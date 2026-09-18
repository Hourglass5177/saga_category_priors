import numpy as np
from category_priors.decision_verification import choose_question,canonical_candidates,scoped_answer_masks


def scene(single=False):
    ids=np.full((20,30),-1,int)
    ids[5:9,5:9]=1  # Anchor, any object identity.
    ids[5:9,12:16]=2 # Candidate boundary difference.
    ids[10:18,10:27]=3 # Large but irrelevant nearby region.
    allowed=ids>=0;disputed=np.isin(ids,[2,3])
    views={v:dict(ids=ids.copy(),allowed=allowed.copy(),disputed=disputed.copy()) for v in ['x','y']}
    return views,[] if single else [('x','y')]


def test_asks_candidate_boundary_not_large_easy_neighbour():
    v,p=scene();q,_=choose_question([[1],[1,2]],v,p,[1],[])
    assert q['b_id']==2 and q['region_ids']==[2]
    assert q['conditional_include_count']==q['conditional_exclude_count']==1


def test_common_members_duplicate_candidates_and_order_cannot_change_question():
    v,p=scene();a,_=choose_question([[1,3],[1,2,3]],v,p,[1],[])
    b,_=choose_question([[1,2,3],[3,1],[1,2,3]],v,p[::-1],[1],[])
    assert a==b


def test_same_candidates_from_swapped_branches_are_one_shared_question():
    v,p=scene();c=[[1],[1,2]];g=[[1,2],[1,3]]
    a,_=choose_question(c+g,v,p,[1],[]);b,_=choose_question(g+c,v,p,[1],[])
    assert a==b


def test_no_independent_pair_does_not_fabricate_validation():
    v,p=scene(single=True);q,why=choose_question([[1],[1,2]],v,p,[1],[])
    assert q is None and why['reason']=='no_actionable_two_view_difference'


def test_missing_core_can_propose_but_not_certify_identity():
    v,p=scene();q,_=choose_question([[1],[1,2]],v,p,[],[1])
    assert q['anchor_status']=='provisional_identity' and not q['boundary_certified']


def test_no_candidate_disagreement_spends_no_query():
    v,p=scene();q,_=choose_question([[1,2],[2,1]],v,p,[1],[])
    assert q is None


def test_visible_decision_mass_beats_tiny_perfectly_balanced_split():
    v,p=scene()
    # Region 2 balances 2/2 but has only 16 visible pixels. Region 3 has
    # a 1/3 split across 136 pixels and affects much more observable geometry.
    q,_=choose_question([[1],[1,2],[1,2,4],[1,3]],v,p,[1],[])
    assert q['b_id']==3


def test_occluded_difference_is_not_a_two_view_question():
    v,p=scene();v['y']['allowed'][v['y']['ids']==2]=False
    q,_=choose_question([[1],[1,2]],v,p,[1],[])
    assert q is None


def test_disconnected_same_signature_does_not_ask_about_both_instances():
    v,p=scene()
    q,_=choose_question([[1],[1,2,3]],v,p,[1],[])
    assert len(q['region_ids'])==1


def test_relation_answer_cannot_certify_unqueried_whole_mask():
    ids=np.zeros((20,20),int);ids[6:14,6:14]=2
    mask=np.ones((3,20,20),bool)
    v=dict(a_masks=mask,b_masks=mask,known=np.ones((20,20),bool))
    fg,bg=scoped_answer_masks(v,[2],ids,'connect','certified')
    assert fg.any() and not fg[ids!=2].any() and not bg.any()
    fg,bg=scoped_answer_masks(v,[2],ids,'connect','provisional_identity')
    assert not fg.any() and not bg.any()


def test_part_whole_disagreement_stays_unknown_in_the_answer():
    ids=np.full((20,20),2);a=np.ones((3,20,20),bool);a[0]=False
    v=dict(a_masks=a,b_masks=np.ones_like(a),known=np.ones((20,20),bool))
    fg,bg=scoped_answer_masks(v,[2],ids,'connect','certified')
    assert not fg.any() and not bg.any()


def test_one_boundary_can_span_distinct_candidate_signatures():
    from category_priors.decision_verification import contrast_regions
    # IDs 2 and 3 form one added boundary relative to base, although an
    # intermediate candidate contains only 2. All-library signatures split it.
    regions=contrast_regions([np.array([1,2]),np.array([1,2,3])],[np.array([1])],
        np.array([2,3]),[np.array([[2,3]])])
    assert any(r.tolist()==[2,3] for r in regions)


def test_runtime_applies_answer_only_to_scoped_region(tmp_path):
    import json
    from types import SimpleNamespace
    from category_priors.regional_evidence_runtime import build_reference,evidence_key
    ids=np.arange(200).reshape(10,20);mask=ids%20<10
    def observation(camera):
        return dict(camera_uid=camera,actual_input={'point_image':[5,5]}),dict(
            sam=mask,observed_pixels=np.ones_like(mask)),None
    rt=SimpleNamespace(sid='s',plan={'regional_evidence_revision':'known-domain-v1'},
        observation=observation,data=lambda _:dict(ids=ids,reliable=np.ones_like(mask)),
        pairs=lambda *_:[('a','b')])
    key=evidence_key(['a','b'],['a','b']);folder=tmp_path/key;folder.mkdir()
    records=[]
    for camera in ('a','b'):
        file=folder/(camera+'.npz');masks=np.ones((3,10,20),bool)
        np.savez(file,a_masks=masks,b_masks=masks,known=np.ones_like(mask))
        records.append(dict(camera=camera,a_xy=[5,5],b_xy=[15,5],file=str(file)))
    result=dict(version='decision-query-v1',evidence_key=key,views=['a','b'],
        anchor_status='certified',a_id=105,region_ids=[115],observations=records)
    (folder/'result.json').write_text(json.dumps(result))
    rt.plan['regional_verification_output']=str(tmp_path)
    refs,_,trace,positive,negative=build_reference(rt,['a','b'],['a','b'])
    assert 115 in positive and 115 not in negative
    assert 116 in negative and 116 not in positive
    assert trace['verification']['certified_pixels_by_view']=={'a':1,'b':1}
