from pathlib import Path
import numpy as np
from category_priors.regional_evidence import competing_families, explanation_quality, choose_explanation
from category_priors.regional_evidence_assembly import commit_selected, compatible_choices
from category_priors.decision_verification import choose_decision_question


def observation(camera, mask, name, known=None):
    mask=np.asarray([mask],bool);known=np.ones_like(mask) if known is None else np.asarray([known],bool)
    ids=np.arange(mask.size).reshape(mask.shape)
    return dict(camera=camera,key=camera+name,path=name,mask=mask,known=known,
                positive_ids=ids[mask&known],negative_ids=ids[~mask&known],known_ids=ids[known])


def test_raw_part_whole_alternatives_and_supported_extension():
    obs=[observation(c,m,n) for c in ('a','b') for n,m in
         [('part',[1,1,1,0,0,0]),('whole',[1,1,1,1,1,0])]]
    families=competing_families(obs,[('a','b')])
    rows=[];members={'part':np.arange(3),'whole':np.arange(5)}
    for name,m in members.items():
        best,_=explanation_quality(lambda c:np.isin(np.arange(6),m)[None],families)
        rows.append(dict(id=name,quality={'energy':best['energy']},core_ids=best['core_ids'],negative_ids=best['negative_ids']))
    winner,_=choose_explanation(rows,members,'part')
    assert winner['id']=='whole'
    assert any(len(f['core_ids'])==5 for f in families)


def test_unknown_expansion_not_certified_by_one_point():
    members={'base':np.arange(3),'new':np.arange(1003)}
    rows=[dict(id=k,quality={'energy':0.},core_ids=np.arange(n),negative_ids=np.array([],int))
          for k,n in [('base',3),('new',4)]]
    chosen,trace=choose_explanation(rows,members,'base')
    assert chosen['id']=='base'
    assert not trace[1]['strict_improvement']


def test_unknown_pixels_do_not_contribute_errors():
    obs=[observation(c,[1,1,1,0,0],'x',[1,1,1,0,0]) for c in ('a','b')]
    fam=competing_families(obs,[('a','b')])
    best,_=explanation_quality(lambda c:np.ones((1,5),bool),fam)
    assert best['fp']==0 and best['fn']==0
    missing,_=explanation_quality(lambda c:np.zeros((1,5),bool),fam)
    assert missing['fn']==6


def test_duplicate_and_order_invariance():
    obs=[observation(c,[1,1,1,0],'x') for c in ('a','b')]
    a=competing_families(obs,[('a','b')]);b=competing_families(obs[::-1]+obs,[('a','b')])
    qa,_=explanation_quality(lambda c:np.array([[1,1,1,0]],bool),a)
    qb,_=explanation_quality(lambda c:np.array([[1,1,1,0]],bool),b)
    assert qa['energy']==qb['energy'] and qa['tp']==qb['tp']


def test_two_disjoint_objects_not_linked_by_same_class():
    obs=[observation(c,m,n) for c in ('a','b') for n,m in
         [('cup1',[1,1,0,0]),('cup2',[0,0,1,1])]]
    fam=competing_families(obs,[('a','b')])
    assert all(not (any(o['path']=='cup1' for o in f['observations']) and
                    any(o['path']=='cup2' for o in f['observations'])) for f in fam)


def test_exact_atomic_commit_removal_and_exchange():
    old=np.array([0,0,0,-1,-1,-1])
    assert np.array_equal(commit_selected(old,{0:np.arange(6)}),np.zeros(6,int))
    assert np.array_equal(commit_selected(old,{0:np.array([0])}),[0,-1,-1,-1,-1,-1])
    assert np.array_equal(commit_selected(np.array([0,0,1,1,9]),{0:[0,2],1:[1,3]}),[0,1,0,1,9])


def test_complete_compatible_alternative_no_trimming():
    options=[[dict(id='a',members=np.array([0,1,2]),changed=True),dict(id='a2',members=np.array([0,1]),changed=False)],
             [dict(id='b',members=np.array([2,3]),changed=True)]]
    best,trace=compatible_choices(options,lambda i,r:0.)
    assert best[0]['id']=='a2' and best[1]['members'].tolist()==[2,3]
    assert trace['search']=='exact'


def test_queries_that_cannot_change_actual_decision_are_rejected():
    ids=np.arange(400).reshape(20,20);allowed=np.ones((20,20),bool)
    views={c:dict(ids=ids,allowed=allowed) for c in ('a','b')}
    winner=ids[5:10,5:10].ravel();whole=ids[5:15,5:10].ravel()
    no,_=choose_decision_question(winner,whole,winner,views,[('a','b')],lambda r,a:'same')
    yes,_=choose_decision_question(winner,whole,winner,views,[('a','b')],lambda r,a:a)
    assert no is None and yes['counterfactual_decisions']['supports_winner']!='supports_challenger'


def test_actual_assembler_honors_local_choice_without_neighbor(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import category_priors.regional_evidence_runtime as runtime
    from category_priors.regional_evidence_assembly import assemble_selected
    monkeypatch.setattr(runtime,'score_object',lambda *a:({}, {}, [], [], []))
    class RT:
        plan={'local_cases':[]};assets=SimpleNamespace(saga20=['cup'])
        b0={'point_labels':[0,0,0,-1,-1,-1],
            'instances':{'0':{'class':'cup','score':1.,'object_uid':'object'}}}
        def classify_members(self,*a):return {'class':'cup','status':'complete','score':1.}
        def actual_for(self,p,u):
            ids=[int(k) for k,v in p['instances'].items() if v['object_uid']==u]
            return np.flatnonzero(np.isin(p['point_labels'],ids))
    path=tmp_path/'selected.npz';np.savez(path,members=np.arange(6))
    bank=dict(selected_id='whole',candidates=[dict(id='whole',members_file=str(path))],
              views=['a','b'],reference_pool=[],ranked_ids=['whole'])
    rt=RT();result=assemble_selected(rt,{'object':bank},tmp_path/'out')
    assert rt.actual_for(result,'object').tolist()==list(range(6))


def test_small_crop_cannot_hide_background_seen_in_wide_observation():
    obs=[observation(c,[1,1,1,0,0,0],n,k) for c in ('a','b')
         for n,k in [('small',[1,1,1,0,0,0]),('wide',[1,1,1,1,1,1])]]
    families=competing_families(obs,[('a','b')])
    wrong,_=explanation_quality(lambda c:np.ones((1,6),bool),families)
    correct,_=explanation_quality(lambda c:np.array([[1,1,1,0,0,0]],bool),families)
    assert wrong['fp']==6 and wrong['energy']>correct['energy']
    assert all(s['fp']>0 for s in explanation_quality(lambda c:np.ones((1,6),bool),families)[1])


def test_ambiguous_crop_extension_keeps_coverage_hole_not_fake_background():
    from category_priors.regional_evidence import aligned_observation_domains
    small=observation('a',[1,1,0,0],'small',[1,1,0,0])
    part=observation('a',[1,1,0,0],'part')
    whole=observation('a',[1,1,1,1],'whole')
    domain=aligned_observation_domains([small,part,whole])[small['key']]
    assert domain['uncovered']==2
    assert not domain['known'][0,2:].any()


def test_regional_semantics_changes_identity_association_not_boundary_energy():
    from category_priors.regional_evidence import identity_links
    obs=[]
    for c in ('a','b'):
        for name,mask,anchor,cos in [('x',[1,1,0],1,[1.,0.,-1.]),('y',[1,0,1],2,[-1.,0.,1.])]:
            o=observation(c,mask,name);o.update(anchor_ids=np.array([anchor]),semantics={'cosines':cos,'top1':'same_name'})
            obs.append(o)
    rejected,evidence=identity_links(obs,[('a','b')])
    assert frozenset(('ax','by')) in rejected and evidence
    fam=competing_families(obs,[('a','b')])
    assert not any(set(f['key'])=={'ax','by'} for f in fam)
    unknown=[dict(o,semantics={'status':'pending_model'}) for o in obs]
    assert not identity_links(unknown,[('a','b')])[0]
    assert any(set(f['key'])=={'ax','by'} for f in competing_families(unknown,[('a','b')]))
    # Merely changing the highest class does not affect a fixed identity's energy.
    x=[o for o in obs if o['key'].endswith('x')]
    a,_=explanation_quality(lambda c:np.array([[1,1,0]],bool),competing_families(x,[('a','b')]))
    b,_=explanation_quality(lambda c:np.array([[1,1,0]],bool),competing_families([dict(o,semantics={'cosines':[-1,0,1]}) for o in x],[('a','b')]))
    assert a['energy']==b['energy']


def test_large_merge_witness_does_not_overrule_reciprocal_separation():
    from category_priors.regional_evidence import separated_families
    obs=[]
    for c in ('a','b'):
        for name,mask,anchor in [('x',[1,1,1,0,0,0],1),('y',[0,0,0,1,1,1],4),('bridge',[1]*6,1)]:
            o=observation(c,mask,name);o['anchor_ids']=np.array([anchor]);obs.append(o)
    fam=competing_families(obs,[('a','b')])
    x=next(f for f in fam if set(f['key'])=={'ax','bx'})
    y=next(f for f in fam if set(f['key'])=={'ay','by'})
    bridge=next(f for f in fam if len(f['core_ids'])==6)
    from category_priors.regional_evidence import identity_partitions
    assert identity_partitions(fam,bridge)  # Also split when the large family won.
    assert separated_families(x,y,[('a','b')])==[('a','b')]
    # Same-prompt part outputs cannot certify different identities.
    for o in y['observations']:o['anchor_ids']=np.array([1])
    assert not separated_families(x,y,[('a','b')])


def test_region_composition_does_not_require_every_splat_to_be_a_core():
    from category_priors.regional_evidence import compose_observed_regions
    a=observation('a',[1]*6,'whole',[1,1,1,1,1,0])
    b=observation('b',[1]*6,'whole',[1,1,1,0,1,1])
    ids=np.arange(6)[None]
    result,trace=compose_observed_regions(np.arange(3),np.arange(6),[a,b],[('a','b')],[ids,ids])
    assert result.tolist()==list(range(6))
    assert trace[0]['independent_pairs']==[('a','b')]
    # A single view still cannot commit a new region.
    result,_=compose_observed_regions(np.arange(3),np.arange(6),[a],[],[ids])
    assert result.tolist()==[0,1,2]
    b['negative_ids']=np.array([4]);b['positive_ids']=np.array([0,1,2,5])
    result,_=compose_observed_regions(np.arange(3),np.arange(6),[a,b],[('a','b')],[ids,ids])
    assert result.tolist()==[0,1,2]


def test_real_assembler_preserves_uncontested_changes_as_explicit_composition(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import category_priors.regional_evidence_runtime as runtime
    from category_priors.regional_evidence_assembly import assemble_selected
    monkeypatch.setattr(runtime,'score_object',lambda *a:({}, {}, [], [], []))
    class RT:
        plan={'local_cases':[]};assets=SimpleNamespace(saga20=['cup'])
        b0={'point_labels':[0,0,0,1,1,1,-1,-1,-1],
            'instances':{'0':{'class':'cup','score':1.,'object_uid':'a'},
                         '1':{'class':'cup','score':1.,'object_uid':'b'}}}
        def classify_members(self,*a):return {'class':'cup','status':'complete','score':1.}
        def actual_for(self,p,u):
            ids=[int(k) for k,v in p['instances'].items() if v['object_uid']==u]
            return np.flatnonzero(np.isin(p['point_labels'],ids))
    path=tmp_path/'selected.npz';np.savez(path,members=[0,1,2,3,6,7,8])
    bank=dict(selected_id='whole',candidates=[dict(id='whole',members_file=str(path))],
              views=['a','b'],reference_pool=[],ranked_ids=['whole'])
    rt=RT()
    pure=assemble_selected(rt,{'a':bank},tmp_path/'pure')
    combined=assemble_selected(rt,{'a':bank},tmp_path/'combined',include_regional=True)
    assert rt.actual_for(pure,'a').tolist()==[0,1,2]
    assert rt.actual_for(combined,'a').tolist()==[0,1,2,6,7,8]
    assert rt.actual_for(combined,'b').tolist()==[3,4,5]
    meta=next(v for v in combined['instances'].values() if v['object_uid']=='a')
    assert meta['selected_candidate']=='ownership-composition'


def test_verification_masks_are_real_g0_g2_observations(tmp_path):
    from types import SimpleNamespace
    from category_priors.multiview_repair_experiment import MultiviewRuntime
    from category_priors.object_verification.model_adapter import CropTransform
    from run_effective_repair import read
    class Renderer:
        def alpha(self,camera,masks,known):
            assert known.all()  # Full interpretation, not three-mask consensus.
            return SimpleNamespace(inside_mass=np.array([[1.,2.,3.]]),visible_mass=np.array([2.,3.,4.]))
    class RT:
        renderer=Renderer();cameras={'a':'camera'}
        def data(self,camera):return dict(rgb=np.zeros((10,12,3),np.uint8),ids=np.zeros((10,12),int),reliable=np.ones((10,12),bool))
        def occlusion_unknown(self,camera):return np.array([],int)
    masks=np.zeros((3,10,12),bool);masks[0,:,:4]=True;masks[1,:,:8]=True;masks[2]=True
    paths=MultiviewRuntime.adopt_raw(RT(),'a',masks,CropTransform((10,12),0,0,12,10),tmp_path/'verification',
        actual_input={'positive_points_image':[[2,5]]},qualities=[.1,.2,.3],source='verification',interpretation_observation=True)
    assert len(paths)==3
    for path in paths:
        with np.load(Path(path)/'alpha.npz') as z:assert z['inside'].tolist()==[1.,2.,3.]
        with np.load(Path(path)/'prediction.npz') as z:assert z['actual_observed_pixels'].all()
        assert read(Path(path)/'observation.json')['source_state']=='verification'


def test_preflight_matches_real_bank_and_actual_writeback(tmp_path):
    from types import SimpleNamespace
    from category_priors.regional_evidence_runtime import object_bank,preview_object_answer,merge_object_observations,evidence_key
    from category_priors.regional_evidence_assembly import assemble_selected
    from category_priors.decision_verification import content_id
    original=[observation(c,[1,1,1,0,0,0],'part') for c in ('a','b')]
    answers=[observation(c,[1,1,1,1,1,0],'whole') for c in ('a','b')]
    class RT:
        plan={'local_cases':[]};sid='scene';assets=SimpleNamespace(saga20=['cup'])
        b0={'point_labels':[0,0,0,-1,-1,-1],
            'instances':{'0':{'class':'cup','score':1.,'object_uid':'object'}}}
        def pairs(self,*a):return [('a','b')]
        def project(self,camera,m):return np.isin(np.arange(6),m)[None]
        def data(self,camera):return {'ids':np.arange(6)[None]}
        def classify_members(self,*a):return {'class':'cup','status':'complete','score':1.}
        def actual_for(self,p,u):
            ids=[int(k) for k,v in p['instances'].items() if v['object_uid']==u]
            return np.flatnonzero(np.isin(p['point_labels'],ids))
    rt=RT();key=(rt.sid,evidence_key([],['a','b']))
    rt.object_observation_overrides={key:(original,competing_families(original,[('a','b')]),{})}
    rows=[dict(id='part',kind='old',members=np.arange(3)),dict(id='whole',kind='new',members=np.arange(5))]
    before=object_bank(rt,'object',rows,[],['a','b'],tmp_path/'before',{'baseline_id':'part'},[])
    initial=assemble_selected(rt,{'object':before},tmp_path/'initial')
    preview=preview_object_answer(rt,before,initial,answers)
    merged=merge_object_observations(original,answers)
    rt.object_observation_overrides={key:(merged,competing_families(merged,[('a','b')]),{})}
    after=object_bank(rt,'object',rows,[],['a','b'],tmp_path/'after',{'baseline_id':before['selected_id']},[])
    written=assemble_selected(rt,{'object':after},tmp_path/'actual',baseline=initial)
    assert after['selected_id']=='whole'
    assert preview['selected_members']==content_id(np.arange(5))
    assert preview['actual_members']==content_id(rt.actual_for(written,'object'))
    assert preview['identity_support']==content_id(np.arange(5))
    assert after['composition_missing']  # No invented alpha masses on CPU.


def test_neutral_semantic_cache_is_shared_and_missing_is_explicit(tmp_path):
    import hashlib,json
    from types import SimpleNamespace
    from category_priors.regional_evidence_runtime import cached_region_semantics
    mask=np.ones((4,5),bool);digest=hashlib.blake2b(mask.tobytes(),digest_size=16).hexdigest()
    folder=tmp_path/'shared'/'semantics'/'camera';folder.mkdir(parents=True)
    row={'cosines':list(np.linspace(-1,1,32)),'status':'complete'}
    (folder/(digest+'.json')).write_text(json.dumps(row))
    a=SimpleNamespace(plan={'neutral_semantic_roots':[str(tmp_path/'shared')]},out=tmp_path/'category')
    b=SimpleNamespace(plan=a.plan,out=tmp_path/'global')
    assert cached_region_semantics(a,'camera',mask)==cached_region_semantics(b,'camera',mask)==row
    assert cached_region_semantics(a,'missing',mask)['status']=='pending_model'


def test_support_diagnostic_change_alone_does_not_authorize_a_query():
    ids=np.arange(400).reshape(20,20);views={c:dict(ids=ids,allowed=np.ones((20,20),bool)) for c in ('a','b')}
    small=ids[5:10,5:10].ravel();whole=ids[5:15,5:10].ravel()
    q,_=choose_decision_question(small,whole,small,views,[('a','b')],
        lambda region,answer:dict(selected_members='same',actual_members='same',identity_support=answer))
    assert q is None


def test_original_repair_exact_mask_encoding_is_reused(tmp_path):
    import hashlib,json
    from types import SimpleNamespace
    from category_priors.regional_evidence_runtime import cached_region_semantics
    mask=np.ones((4,5),bool);digest=hashlib.blake2b(mask.tobytes(),digest_size=16).hexdigest()
    folder=tmp_path/'effective-repair-01'/'semantics'/'camera';folder.mkdir(parents=True)
    row={'cosines':list(np.linspace(-1,1,32)),'status':'complete'}
    (folder/(digest+'.json')).write_text(json.dumps(row))
    rt=SimpleNamespace(plan={'neutral_semantic_roots':[]},reference=tmp_path/'effective-repair-01')
    assert cached_region_semantics(rt,'camera',mask)==row
    assert cached_region_semantics(rt,'camera',~mask)['status']=='pending_model'
