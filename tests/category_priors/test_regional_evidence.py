import numpy as np
from category_priors.regional_evidence import (consensus,mask_key,score_projection,
    select_regional,signed_reciprocity,member_regions,repair_regions)


def hypothesis(mask,known=None):
    m=np.asarray(mask,bool);k=np.ones_like(m) if known is None else np.asarray(known,bool)
    return dict(mask=m,known=k,key=mask_key(m,k))


def rows_for(shapes, targets, positive, negative=()):
    refs=[]
    for camera in ('a','b'):
        hs=[hypothesis(t) for t in targets];fg,bg,_=consensus(hs)
        refs.append(dict(camera=camera,hypotheses=hs,foreground=fg,background=bg))
    rows=[];members={}
    for name,shape in shapes.items():
        m=np.flatnonzero(shape);members[name]=m
        q=score_projection(lambda _:np.asarray(shape,bool),refs);q['independent_pairs']=[('a','b')]
        rows.append(dict(id=name,kind='original' if name=='original' else 'new',quality=q,
            core_ids=np.intersect1d(m,positive),negative_ids=np.array(negative),
            semantics={'class':'cup'}))
    return rows,members


def test_disagreement_is_symmetric_and_duplicates_do_not_vote():
    a=hypothesis([1,1,0,0]);b=hypothesis([1,0,1,0])
    fg,bg,unknown=consensus([a,b,a])
    assert fg.tolist()==[True,False,False,False]
    assert bg.tolist()==[False,False,False,True]
    assert unknown.tolist()==[False,True,True,False]
    assert all(np.array_equal(x,y) for x,y in zip((fg,bg,unknown),consensus([b,a])))


def test_outside_crop_is_not_negative():
    fg,bg,_=consensus([hypothesis([1,1],[1,1]),hypothesis([1,0],[1,0])])
    assert fg.all() and not bg.any()


def test_whole_replaces_part_when_additions_have_independent_support():
    rows,m=rows_for({'original':[1,1,1,0,0,0,0],'whole':[1,1,1,1,1,1,0]},
                    [[1,1,1,1,1,1,0]],range(6),[6])
    assert select_regional(rows,m)[0]['id']=='whole'
    rows[1]['semantics']={'class':'unknown','score':-999}
    assert select_regional(rows[::-1],m)[0]['id']=='whole'


def test_picture_wall_cooccurrence_is_not_a_certificate():
    rows,m=rows_for({'original':[1,1,1,0,0,0],'wall':[1,1,1,1,1,1]},
                    [[1,1,1,0,0,0],[1,1,1,1,1,1]],range(3))
    assert select_regional(rows,m)[0]['id']=='original'


def test_unknown_additions_do_not_become_whole_object_confidence():
    rows,m=rows_for({'original':[1,1,1,0,0,0],'whole':[1,1,1,1,1,1]},
                    [[1,1,1,1,1,1]],range(4))
    winner,trace=select_regional(rows,m)
    assert winner['id']=='whole'
    assert next(t for t in trace if t['selected'])['status']=='partially_resolved'


def test_single_view_cannot_certify_a_replacement():
    rows,m=rows_for({'original':[1,1,1,0],'new':[1,1,1,1]},[[1,1,1,1]],range(4))
    for r in rows:r['quality']['independent_pairs']=[]
    assert select_regional(rows,m)[0]['id']=='original'


def test_background_and_removed_core_prevent_replacement():
    rows,m=rows_for({'original':[1,1,1,0,0],'spill':[1,1,1,1,1]},[[1,1,1,0,0]],range(3),[3,4])
    assert select_regional(rows,m)[0]['id']=='original'


def test_partial_region_repairs_without_changing_unknown():
    baseline=np.array([0,1,2,6]);regions=[np.array([0,1,2]),np.array([3,4]),np.array([5]),np.array([6])]
    result,trace=repair_regions(baseline,regions,[0,1,2,3,4],[6])
    assert result.tolist()==[0,1,2,3,4]
    assert trace[2]['status']=='unresolved'


def test_regions_do_not_connect_neighbour_cups_through_table():
    members={'a':np.array([1,2,3,4]),'b':np.array([1,2])}
    image=np.array([[1,2,-1,3,4]])
    groups=member_regions(members,[image])
    assert sorted(sorted(g.tolist()) for g in groups)==[[1,2],[3,4]]


def relation(camera,kind):
    a=np.zeros((3,3,5),bool);b=a.copy();a[:,1,1]=1;b[:,1,3]=1
    if kind=='connect':a[:,1,3]=1;b[:,1,1]=1
    if kind=='ambiguous':a[0,1,3]=1;b[0,1,1]=1
    return dict(camera=camera,a_masks=a,b_masks=b,a_xy=[1,1],b_xy=[3,1],known=np.ones((3,5),bool))


def test_reciprocal_separation_requires_both_directions_and_two_views():
    assert signed_reciprocity([relation('a','separate'),relation('b','separate')],[('a','b')])[0]=='separate'
    assert signed_reciprocity([relation('a','separate')],[('a','b')])[0]=='unresolved'
    assert signed_reciprocity([relation('a','separate'),relation('b','ambiguous')],[('a','b')])[0]=='unresolved'


def test_chair_part_alternative_cannot_certify_separate_instances():
    assert signed_reciprocity([relation('a','ambiguous'),relation('b','ambiguous')],[('a','b')])[0]=='unresolved'


def test_occluded_anchor_cannot_cast_a_vote():
    a=relation('a','connect');b=relation('b','connect');b['known'][1,1]=False
    assert signed_reciprocity([a,b],[('a','b')])[0]=='unresolved'


def test_qualified_unknown_cannot_reappear_as_pairwise_background_veto():
    # Both interpretations confirm pixel 3, disagree about pixels 4..10.
    # The latter must neither provide support nor veto the confirmed addition.
    part=[1,1,1,1,0,0,0,0,0,0,0];whole=[1]*11
    shapes={'original':[1,1,1,0,0,0,0,0,0,0,0],'whole':whole}
    rows,members=rows_for(shapes,[part,whole],range(4))
    assert select_regional(rows,members)[0]['id']=='original'  # Reproduce v2 bug.
    hs=[hypothesis(part),hypothesis(whole)];fg,bg,_=consensus(hs)
    refs=[dict(camera=c,hypotheses=hs,foreground=fg,background=bg) for c in ('a','b')]
    for row in rows:
        row['quality']=score_projection(lambda _:np.asarray(shapes[row['id']],bool),refs,known_domain=True)
        row['quality']['independent_pairs']=[('a','b')]
    chosen,trace=select_regional(rows,members)
    assert chosen['id']=='whole'
    assert next(t for t in trace if t['selected'])['status']=='partially_resolved'
    assert chosen['quality']['per_view'][0]['fp']==0


def test_boundary_splat_with_both_labels_is_unknown_not_background():
    from category_priors.regional_evidence import symmetric_member_evidence
    fg,bg,unknown=symmetric_member_evidence([1,2,3],[3,4,5])
    assert fg.tolist()==[1,2] and bg.tolist()==[4,5] and unknown.tolist()==[3]
    fg2,bg2,unknown2=symmetric_member_evidence([3,4,5],[1,2,3])
    assert np.array_equal(fg,bg2) and np.array_equal(bg,fg2) and np.array_equal(unknown,unknown2)


def test_regional_assembly_transfers_only_exclusive_evidence_and_keeps_unknown(tmp_path):
    from types import SimpleNamespace
    from category_priors.regional_evidence_assembly import assemble
    from category_priors.regional_evidence import VERSION
    n=12;ids=np.arange(n)
    def score(m,pool,views):
        refs=[]
        if pool:
            for camera in views:
                h=hypothesis(np.isin(ids,[6,7,8]));f,b,_=consensus([h])
                refs.append(dict(camera=camera,hypotheses=[h],foreground=f,background=b))
        q=score_projection(lambda _:np.isin(ids,m),refs);q['independent_pairs']=[('A','B')] if refs else []
        core=np.intersect1d(m,[6,7,8]) if refs else np.array([],int)
        return {'class':None},q,core,np.array([],int),refs
    def actual(payload,u):
        u=payload.get('instance_aliases',{}).get(u,u)
        return np.array([i for i,k in enumerate(payload['point_labels']) if k>=0 and
            payload['instances'].get(str(k),payload['instances'].get(k,{})).get('object_uid')==u],int)
    rt=SimpleNamespace(plan={'selector':VERSION},assets=SimpleNamespace(saga20=['table']),
        b0={'point_labels':[0]*10+[-1]*2,'instances':{'0':{'class':'table','object_uid':'B0:0','score':.8}}},
        descriptor=lambda p:{'camera':p},score_evidence=score,actual_for=actual)
    path=tmp_path/'members.npz';np.savez(path,members=np.array([6,7,8,10]))
    bank=dict(views=['A','B'],reference_pool=['A','B'],selected_id='original',
        candidates=[dict(id='original',kind='original',members_file=str(path))])
    result=assemble(rt,{'source':bank},tmp_path/'out')
    assert actual(result,'source').tolist()==[6,7,8,10]
    assert result['point_labels'][11]==-1  # No geometry outside the saved proposals.
    assert actual(result,'B0:0').tolist()==[0,1,2,3,4,5,9]
    assert any(m['class']=='unknown' for m in result['instances'].values())
    duplicate=assemble(rt,{'source':bank,'duplicate':bank},tmp_path/'deduplicated')
    assert actual(duplicate,'source').tolist()==actual(duplicate,'duplicate').tolist()==[6,7,8,10]
    assert len(duplicate['instances'])==2  # The table plus one object, not zero or two duplicates.
