from types import SimpleNamespace

import numpy as np

from category_priors.common_geometry_selector import (
    geometry_quality, pixel_evidence, reference_family, select_geometry, semantic_state)
from category_priors.multiview_repair_experiment import MultiviewRuntime


def observation(path, camera, positive, negative=()):
    return dict(path=path, camera=camera, positive_ids=np.array(positive, int),
                negative_ids=np.array(negative, int), known_ids=np.union1d(positive, negative))


def candidate(name, members, target, *, kind='new', core=(), negative=()):
    ids = np.arange(12)
    per_view = [dict(camera_uid=v, **pixel_evidence(np.isin(ids, members),
                np.isin(ids, target), ~np.isin(ids, target))) for v in ('A', 'B')]
    return dict(id=name, kind=kind, member_count=len(members), quality=geometry_quality(per_view),
                core_ids=np.array(core, int), negative_ids=np.array(negative, int),
                semantics={'class': 'cup', 'score': .1})


def test_missing_foreground_and_contamination_are_not_erased():
    full = pixel_evidence([1,1,1,0], [1,1,1,0], [0,0,0,1])
    part = pixel_evidence([1,0,0,0], [1,1,1,0], [0,0,0,1])
    spill = pixel_evidence([1,1,1,1], [1,1,1,0], [0,0,0,1])
    assert full['iou'] == 1 and part['fn'] == 2 and part['iou'] == 1/3
    assert spill['fp'] == 1 and spill['iou'] == .75
    assert pixel_evidence([0,0], [1,1], [0,0])['iou'] == 0


def test_unknown_and_sam_ambiguity_are_neither_foreground_nor_background():
    assert pixel_evidence([1,1,0], [1,0,0], [0,0,0])['iou'] == 1
    assert pixel_evidence([0,0,0], [0,0,0], [0,0,0])['iou'] is None


def test_reference_is_content_based_and_duplicate_camera_is_not_another_vote():
    rows = [observation('z','A',[1,2,3],[4]), observation('a','B',[1,2,3],[4])]
    a, trace = reference_family(rows, [('A','B')])
    rows2 = [dict(rows[1], path='different'), dict(rows[0], path='elsewhere'), rows[0]]
    b, trace2 = reference_family(rows2, [('A','B')])
    assert trace == trace2 and len(a) == len(b) == 2
    assert reference_family(rows, [])[0] == []


def test_disjoint_same_class_cups_have_no_identity_edge():
    rows = [observation('a','A',[1,2,3]), observation('b','B',[4,5,6])]
    assert reference_family(rows, [('A','B')])[0] == []


def test_whole_candidate_wins_without_semantic_or_size_advantage():
    members = {'original': np.arange(3), 'whole': np.arange(6), 'spill': np.arange(9)}
    rows = [candidate(k, m, np.arange(6), kind='original' if k=='original' else 'new',
                      core=np.arange(6), negative=np.arange(6,12)) for k,m in members.items()]
    rows[0]['semantics'] = {'class':'cup', 'score':1.}
    rows[1]['semantics'] = {'class':'computer', 'score':0.}
    winner, _ = select_geometry(rows, members)
    assert winner['id'] == 'whole'
    rows[1]['semantics'] = {'class':None, 'score':999}
    assert select_geometry(rows[::-1], members)[0]['id'] == 'whole'


def test_view_disagreement_cannot_be_hidden_by_mean_improvement():
    a = candidate('original', [0,1,2], [0,1,2], kind='original')
    b = candidate('new', [3,4,5], [0,1,2])
    a['quality'] = geometry_quality([dict(camera_uid='A',iou=.5),dict(camera_uid='B',iou=.5)])
    b['quality'] = geometry_quality([dict(camera_uid='A',iou=1.),dict(camera_uid='B',iou=.4)])
    assert select_geometry([a,b], {'original':np.arange(3),'new':np.arange(3,6)})[0]['id']=='original'


def test_unsupported_class_is_explicit_not_forced_into_evaluation_vocabulary():
    assert semantic_state({'class':'computer'}, ['cup']) == ('computer','out_of_eval_vocabulary')
    assert semantic_state({'class':None}, ['cup']) == ('unknown','unknown')


def runtime_for_assembly():
    rt = MultiviewRuntime.__new__(MultiviewRuntime)
    rt.plan = {'selector':'common-geometry-v1'}
    rt.assets = SimpleNamespace(saga20=['cup'])
    rt.b0 = dict(point_labels=[-1]*12, instances={})
    rt.descriptor = lambda p: {'camera':p}
    rt.pairs = lambda views, members: [('A','B')] if len(views)==2 else []
    def score(members, pool, views):
        c = candidate('x', members, [0,1,2,3,4,5], core=[0,1,2,3,4,5], negative=[6,7,8])
        core = np.intersect1d(members, [0,1,2,3,4,5])
        refs = [observation(v,v,[0,1,2,3,4,5],[6,7,8]) for v in views]
        return {'class':'computer','score':.1}, c['quality'], core, np.array([6,7,8]), refs
    rt.score_evidence = score
    return rt


def test_all_candidates_and_unknown_geometry_survive_standard_export(tmp_path):
    rt = runtime_for_assembly()
    rows = []
    for name, members, kind in [('original',[0,1,2],'original'),('whole',list(range(6)),'new')]:
        path = tmp_path/(name+'.npz')
        np.savez(path, members=members)
        rows.append(dict(id=name,kind=kind,members_file=str(path)))
    bank = dict(candidates=rows, selected_id='original', reference_pool=['A','B'], views=['A','B'])
    payload = rt.assemble_evidence({'target':bank}, tmp_path/'scene')
    assert rt.actual_for(payload,'target').tolist() == list(range(6))
    assert payload['instances']['0']['class']=='computer'
    from run_effective_repair import read
    assert read(tmp_path/'scene'/'scene-evaluation.json')['instances'] == {}
    assert read(tmp_path/'scene'/'scene.json')['instances']['0']['classification_status']=='out_of_eval_vocabulary'


def test_fixed_reference_disputed_single_view_extension_remains_unknown(tmp_path):
    rt = MultiviewRuntime.__new__(MultiviewRuntime)
    rt.plan = {'selector':'common-geometry-v1'}
    rt.sid = 'test'; rt.assets = SimpleNamespace(saga20=['cup'])
    ids = np.arange(6)[None]
    rt.data = lambda view: dict(ids=ids, reliable=np.ones_like(ids, bool))
    rt.pairs = lambda views, members: [('A','B')]
    rows = {}
    for view in ('A','B'):
        group = tmp_path/view; group.mkdir()
        small = np.isin(ids,[0,1,2])
        whole = np.isin(ids,[0,1,2,3]) if view=='A' else small
        np.savez(group/'raw.npz', alternatives=np.stack([small,whole,small]))
        row = dict(camera_uid=view,raw_group=str(group),actual_input={'point_image':[0,0]})
        arrays = dict(sam=whole,observed_pixels=np.ones_like(ids,bool))
        # A's disputed extension is unknown in B; it has no independent support.
        descriptor = observation(view,view,[0,1,2,3] if view=='A' else [0,1,2],[4,5])
        rows[view] = row, arrays, descriptor
    rt.observation = lambda path: (*rows[path][:2], {})
    rt.descriptor = lambda path: rows[path][2]
    refs, _, _ = rt.fixed_reference(['A','B'],['A','B'])
    a = next(r for r in refs if r['camera']=='A')
    assert not a['foreground'][0,3] and not a['background'][0,3]
    assert a['foreground'][0,:3].all()


def test_geometry_evaluator_keeps_unknown_and_class_aware_export_does_not(tmp_path):
    from run_effective_repair import save
    from run_effective_repair_evaluation import geometry_rows_from_export
    save(tmp_path/'geometry.json', dict(point_labels=[0,0,0],
        instances={'0':dict(**{'class':'unknown'},score=.8)}))
    save(tmp_path/'export.json', dict(point_labels=[-1,-1,-1],instances={},
                                    geometry_source=str(tmp_path/'geometry.json')))
    rows = geometry_rows_from_export(tmp_path/'export.json',np.arange(3),3)
    assert len(rows)==1 and rows[0].mask.all() and rows[0].class_name=='unknown'


def test_b0_source_alias_uses_its_own_reference_and_preserves_geometry(tmp_path):
    rt=runtime_for_assembly()
    rt.b0=dict(point_labels=[0,0,0]+[-1]*9,instances={'0':{'class':'cup','score':.3}})
    path=tmp_path/'whole.npz'; np.savez(path,members=np.arange(6))
    bank=dict(candidates=[dict(id='whole',kind='new',members_file=str(path))],
              reference_pool=['A','B'],views=['A','B'])
    result=rt.assemble_evidence({'scene:B0:000000':bank},tmp_path/'scene')
    assert rt.actual_for(result,'scene:B0:000000').tolist()==list(range(6))
    assert len(result['instances'])==1


def test_empty_winner_gets_saved_backup_without_stealing_verified_core(tmp_path):
    rt=runtime_for_assembly()
    def score(members,pool,views):
        rival='rival' in pool
        value=1. if rival or np.isin(members,[0,1,2]).all() else .8
        per=[dict(camera_uid=v,iou=value,tp=3,fp=0,fn=0) for v in views]
        core=np.intersect1d(members,[0,1,2]) if rival else np.array([],int)
        refs=[observation(v,v,[0,1,2],[]) for v in views]
        return {'class':'computer','score':.5},geometry_quality(per),core,np.array([],int),refs
    rt.score_evidence=score; rt.descriptor=lambda p: {'camera':'A'}
    def bank(name, variants, pool):
        rows=[]
        for cid,ids in variants:
            path=tmp_path/(name+cid+'.npz');np.savez(path,members=ids)
            rows.append(dict(id=cid,kind='original' if cid=='original' else 'new',members_file=str(path)))
        return dict(candidates=rows,reference_pool=pool,views=['A','B'])
    a=bank('a',[('original',[0,1,2]),('backup',[6,7,8])],['target'])
    b=bank('b',[('original',[0,1,2])],['rival'])
    result=rt.assemble_evidence({'a':a,'b':b},tmp_path/'scene')
    assert rt.actual_for(result,'a').tolist()==[6,7,8]
    assert rt.actual_for(result,'b').tolist()==[0,1,2]


def test_unobserved_incumbent_is_not_a_fabricated_protected_core(tmp_path):
    rt=runtime_for_assembly()
    rt.b0=dict(point_labels=[0]*12,instances={'0':{'class':'cup','score':.3}})
    def score(members,pool,views):
        per=[dict(camera_uid='A',**pixel_evidence(np.isin(np.arange(12),members),
             np.isin(np.arange(12),[0,1,2]),np.zeros(12,bool)))] if pool else []
        return {'class':'cup','score':.5},geometry_quality(per),np.array([],int),np.array([],int),[
            observation('A','A',[0,1,2])] if pool else []
    rt.score_evidence=score;rt.descriptor=lambda p: {'camera':'A'}
    path=tmp_path/'visible.npz';np.savez(path,members=[0,1,2])
    bank=dict(candidates=[dict(id='original',kind='original',members_file=str(path))],
              reference_pool=['A'],views=['A'])
    result=rt.assemble_evidence({'target':bank},tmp_path/'scene')
    assert rt.actual_for(result,'target').tolist()==[0,1,2]


def test_other_neutral_crop_foreground_cannot_become_hard_background(tmp_path,monkeypatch):
    import category_priors.multiview_repair_experiment as module
    rt=MultiviewRuntime.__new__(MultiviewRuntime);rt.plan={'selector':'common-geometry-v1'};rt.sid='test'
    ids=np.arange(6)[None]; data=dict(ids=ids,reliable=np.ones_like(ids,bool))
    rt.data=lambda view:data;rt.pairs=lambda views,members:[('A','B')]
    rows={}
    for path,camera,positive in [('small-A','A',[0,1,2]),('large-A','A',[0,1,2,3]),('small-B','B',[0,1,2])]:
        group=tmp_path/path;group.mkdir(); mask=np.isin(ids,positive)
        np.savez(group/'raw.npz',alternatives=np.stack([mask]*3))
        rows[path]=(dict(camera_uid=camera,raw_group=str(group),actual_input={'point_image':[0,0]}),
                    dict(sam=mask,observed_pixels=np.ones_like(mask)),
                    observation(path,camera,positive,np.setdiff1d(np.arange(6),positive)))
    rt.observation=lambda path:(*rows[path][:2],{})
    rt.descriptor=lambda path:rows[path][2]
    monkeypatch.setattr(module,'reference_family',lambda observations,pairs:
        ([dict(rows[p][2]) for p in ('small-A','small-B')],{'status':'complete'}))
    refs,_,_=rt.fixed_reference(list(rows),['A','B'])
    a=next(r for r in refs if r['camera']=='A')
    assert not a['background'][0,3] and not a['foreground'][0,3]
    assert pixel_evidence(np.isin(ids,[0,1,2,3]),a['foreground'],a['background'])['fp']==0
