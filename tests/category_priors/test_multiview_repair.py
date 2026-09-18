from types import SimpleNamespace
import numpy as np

from category_priors.multiview_repair import (metric_crop, interior_points, known_domain,
    boundary_reasons, depth_visibility, mask_families, independent_support,
    evidence_score, select_evidence, select_view_pair)
from category_priors.multiview_repair_experiment import MultiviewRuntime
from category_priors.object_verification.model_adapter import CropTransform, InjectedModelAdapter
from run_effective_repair import save


def test_competing_instances_use_the_same_observation_pool(tmp_path):
    rt = assembly_runtime()
    rt.assets.sources.append(SimpleNamespace(candidate_uid='second', support_ids=np.arange(5)))
    seen = []
    original = rt.score_evidence
    def score(members, pool, views):
        seen.append((tuple(members), tuple(pool), tuple(views)))
        return original(members, pool, views)
    rt.score_evidence = score
    rt.assemble_evidence({'new': dict(pool=['crop-a'], views=['A']),
                          'second': dict(pool=['crop-b'], views=['B'])}, tmp_path)
    assert {p for m, p, v in seen if m == tuple(range(5))} == {('crop-a', 'crop-b')}
    assert {v for m, p, v in seen if m == tuple(range(5))} == {('A', 'B')}


def test_feedback_uses_actual_core_and_retains_open_candidates(tmp_path, monkeypatch):
    import category_priors.multiview_repair_experiment as module
    rt = MultiviewRuntime.__new__(MultiviewRuntime)
    rt.plan = {'classes32': ['cup']}; rt.mode = 'category'; rt.scale_priors = {}
    rt.saved_global = tmp_path / 'old'; rt.sid = 'scene'; rt.shared = tmp_path / 'shared'
    def bank(name, members, core):
        path = tmp_path / (name + '.npz')
        np.savez(path, members=members, core=core)
        return dict(selected_id=name, candidates=[dict(id=name, kind='new', paths=[],
            members_file=str(path))], pool=[name], paths_by_branch={})
    initial = bank('initial', [0, 1, 2], [0])
    open_bank = bank('open', [3, 4, 5], [3, 4])
    common = dict(bank=initial, groups_by_q={}, first_views=['A', 'B'])
    def initial_evidence(members, pool, views):
        assert list(members) == [7,8,9] and views == ['A','B']
        return {}, {}, np.array([9]), [], []
    rt.score_evidence = initial_evidence
    rt.classify_members = lambda members, views: {'mean_cosines': [1.]}
    monkeypatch.setattr(module, 'scale_slots', lambda *args: [])
    captured = {}
    def save_bank(uid, rows, pool, views, dest, metadata):
        captured.update(rows=rows, pool=pool, **metadata)
        return captured
    rt.bank = save_bank; rt.addition_trace = lambda *args: None
    rt.final_bank('test', common, initial, dict(construction=['A','B','C','D'], changed=False),
        np.array([7,8,9]), 'feedback', tmp_path / 'feedback', open_bank=open_bank)
    assert captured['feedback_applicable']
    assert captured['reliable_written_core_count'] == 1
    assert any(r['members'].tolist() == [3,4,5] for r in captured['rows'])
    assert 'open' in captured['pool']


def test_feedback_reads_initial_writeback_but_rolls_back_to_open_scene(tmp_path):
    rt = MultiviewRuntime.__new__(MultiviewRuntime)
    rt.out = tmp_path / 'output'; rt.reference = tmp_path / 'reference'; rt.shared = tmp_path / 'shared'
    rt.plan = {'bindings': {'scene': {}}, 'local_cases': []}; rt.stats = {}
    rt.assets = SimpleNamespace(sources=[SimpleNamespace(candidate_uid='target')])
    rt.obs_cache = {}; rt.scene = lambda sid: None
    save(rt.reference / 'scenes' / 'scene' / 'panels.json',
         {'target': {'construction': ['A','B','C','D'], 'heldout': None}})
    rt.common_initial = lambda *args: {}
    rt.choose_views = lambda uid, common, panel, dest: dict(panel, changed=False)
    rt.mode_initial = lambda *args: {}
    rt.actual_for = lambda scene, uid: scene['actual']
    input_members, baselines = {}, {}
    def final(uid, common, initial, panel, actual, protocol, dest, *, open_bank):
        input_members[protocol] = actual
        if protocol == 'feedback':
            assert open_bank == {'protocol': 'matched-open'}
        return {'protocol': protocol}
    rt.final_bank = final
    def assemble(banks, dest, *, baseline=None, excluded_views=None):
        baselines[dest.name] = baseline
        return {'actual': dest.name}
    rt.assemble_evidence = assemble; rt.export_legacy = lambda *args: None
    rt.full()
    assert input_members == {'matched-open': 'initial', 'feedback': 'initial'}
    assert baselines['feedback'] == {'actual': 'matched-open'}


def test_shared_assembly_pool_cannot_borrow_another_objects_evaluation_view(tmp_path):
    rt = assembly_runtime()
    rt.plan = {'local_cases': [{'candidate_uid': 'new', 'views': [
        {'camera_uid': 'A'}, {'camera_uid': 'B'}, {'camera_uid': 'H'}]}]}
    rt.descriptor = lambda p: {'camera': p}
    original = rt.score_evidence
    def score(members, pool, views):
        assert 'H' not in views and 'H' not in pool
        return original(members, pool, views)
    rt.score_evidence = score
    rt.assemble_evidence({'new': dict(pool=['A','B'], views=['A','B']),
                          'second': dict(pool=['A','H'], views=['A','H'])}, tmp_path)


def test_unchanged_baseline_outside_semantic_export_taxonomy_does_not_clear_phone(tmp_path):
    # Reproduces B0:2 -> computer (32-way), unchanged geometry, then an invalid
    # 20-way class filter triggering rollback of a neighbouring repaired phone.
    rt = MultiviewRuntime.__new__(MultiviewRuntime)
    phone = np.array([0, 6, 7, 8, 9, 10, 11, 12, 13])
    rt.assets = SimpleNamespace(saga20={'cabinet', 'phone'}, sources=[
        SimpleNamespace(candidate_uid='repair', support_ids=phone)])
    rt.b0 = dict(point_labels=[0]*6+[-1]*8, instances={
        '0': {'class': 'cabinet', 'score': .5, 'point_count': 6}})
    rt.selected = lambda bank: ({}, phone)
    def score(members, pool, views):
        baseline = set(members) == set(range(6))
        return ({'class': 'computer' if baseline else 'phone'},
                {'score': .2 if baseline else .8},
                np.array([0, 1, 2, 3] if baseline else [6, 7, 8]),
                np.array([], int), [])
    rt.score_evidence = score
    result = rt.assemble_evidence({'repair': {'pool': [], 'views': []}}, tmp_path)
    assert len(rt.actual_for(result, 'repair')) == 8
    assert rt.actual_for(result, 'B0:0').tolist() == list(range(6))
    assert {m['class'] for m in result['instances'].values()} == {'cabinet', 'phone'}


def test_residual_baseline_class_conflict_is_not_geometric_core_loss(tmp_path):
    rt = MultiviewRuntime.__new__(MultiviewRuntime)
    phone = np.arange(4,14)
    rt.assets = SimpleNamespace(saga20={'cabinet','phone'}, sources=[])
    rt.b0 = dict(point_labels=[0]*6+[-1]*8,
                 instances={'0': {'class':'cabinet','score':.5}})
    rt.selected = lambda bank: ({},phone)
    def score(members, pool, views):
        residual = set(members).issubset(set(range(6)))
        return ({'class':'computer' if residual else 'phone', 'score':.6},
                {'score':.2 if residual else .8},
                np.array([0,1] if residual else [4,5,6]),np.array([],int),[])
    rt.score_evidence = score
    result = rt.assemble_evidence({'repair': {'pool':[], 'views':[]}}, tmp_path)
    assert rt.actual_for(result,'B0:0').tolist() == [0,1,2,3]
    assert rt.actual_for(result,'repair').tolist() == phone.tolist()
    baseline = next(m for m in result['instances'].values() if m['object_uid']=='B0:0')
    assert baseline['class']=='cabinet'
    assert baseline['classification_fallback']['reestimated_class']=='computer'
    assert baseline['classification_fallback']['reason']=='unsupported_residual_class'


def test_unsupported_new_object_does_not_inherit_a_baseline_class(tmp_path):
    rt=assembly_runtime()
    original=rt.score_evidence
    def score(members,pool,views):
        semantics,quality,core,negative,refs=original(members,pool,views)
        if len(members)==5: semantics['class']='computer'
        return semantics,quality,core,negative,refs
    rt.score_evidence=score
    result=rt.assemble_evidence({'new':{'pool':[], 'views':['A','B']}},tmp_path)
    assert result['point_labels']==[0,0,0,-1,-1]
    assert rt.actual_for(result,'new').size==0


def test_metric_crop_center_is_not_the_prompt_or_huge_old_box():
    xy = np.array([[200,200],[220,200],[200,220],[220,220]])
    crop, trace = metric_crop((600,1000),xy,800,.2,2)
    assert crop.width == 120
    assert trace['center_image_xy'] == [210,210]
    assert crop.image_to_crop_points([[200,200]]).tolist() == [[50,50]]
    crop2,_ = metric_crop((600,1000),xy,800,.4,2)
    assert crop2.width == 240


def test_photo_edge_crop_edge_and_padding_are_distinct():
    observed = np.zeros((20,30),bool); observed[:,0:12]=True
    photo = np.zeros_like(observed); photo[5:10,0:3]=True
    algorithm = np.zeros_like(observed); algorithm[5:10,10:12]=True
    assert boundary_reasons(photo,observed) == dict(image_truncated=True,crop_truncated=False)
    assert boundary_reasons(algorithm,observed) == dict(image_truncated=False,crop_truncated=True)
    crop = CropTransform((20,30),-10,0,20,20)
    _,valid = crop.extract(np.zeros((20,30,3),np.uint8))
    assert not valid[:,:10].any()
    assert crop.mask_to_image(valid).sum() == 200


def test_whole_part_disagreement_and_unobserved_pixels_are_unknown():
    whole = np.zeros((10,10),bool); whole[2:8,2:8]=True
    part = np.zeros_like(whole); part[3:5,3:5]=True
    observed=np.ones_like(whole); observed[:,8:]=False
    known = known_domain(part,[part,whole,part],observed)
    assert not known[whole & ~part].any()
    assert known[part].all() and not known[:,8:].any()
    assert known[0,0]


def test_depth_cues_never_assign_occluder_to_target():
    out=depth_visibility([[1,1],[2,1],[4,1],[0,1]],[2,2,2,2],
        np.array([[0,0,0],[7,9,8]]),np.ones((2,3)),
        np.array([[1,1,1],[0,1,1]],bool),[4,5,6,10],[4,5,6,9,10])
    assert out['suspected_occluded'].tolist() == [True,True,False,False]
    assert out['suspected_self_occluded'].tolist() == [True,False,False,False]
    assert out['outside_image'][2] and out['unreliable'][3]
    assert out['gaussian_ids'].tolist() == [4,5,6,10]


def test_no_reliable_positive_falls_back_instead_of_inventing_prompt():
    assert interior_points(np.ones((8,8),bool),np.zeros((8,8),bool),3)==[]


def test_three_prompts_are_real_internal_foreground_pixels():
    mask=np.zeros((20,20),bool);mask[3:18,2:12]=True
    points=interior_points(mask,np.ones_like(mask),3)
    assert len(points)==len(set(map(tuple,points)))==3
    assert all(mask[y,x] for x,y in points)
    class Sam:
        def set_image(self,image): pass
        def predict(self,**kwargs):
            assert kwargs['box'] is None
            assert kwargs['point_coords'].shape==(3,2)
            assert kwargs['point_labels'].tolist()==[1,1,1]
            return np.stack([mask,mask,mask]),np.array([.2,.8,.4]),None
    result=InjectedModelAdapter(sam_predictor=Sam())._sam_masks(
        image=np.zeros((20,20,3),np.uint8),crop=CropTransform((20,20),0,0,20,20),
        box_crop=None,point_crop=points,uid='multi')
    assert len(result)==3


def obs(path,camera,positive,known=None):
    return dict(path=path,camera=camera,positive_ids=np.array(positive,int),
        known_ids=np.array(known if known is not None else positive,int),negative_ids=np.array([],int))


def test_adjacent_same_class_instances_without_common_surface_never_join():
    families=mask_families([obs('a','A',[1,2]),obs('b','B',[3,4])])
    assert families==[['a'],['b']]


def test_every_raw_mask_is_a_seed_and_new_visible_area_is_not_contradiction():
    rows=[obs('a0','A',[1]),obs('a1','A',[1,2]),obs('a2','A',[2]),
          obs('b0','B',[1,3]),obs('b1','B',[1,2,3]),obs('b2','B',[2,4])]
    families=mask_families(rows)
    assert set(sum(families,[]))=={r['path'] for r in rows}
    assert len(families)<=6
    assert all(len(f)==2 for f in families)
    assert independent_support([rows[1],rows[4]],[('A','B')]).tolist()==[1,2]
    assert independent_support([obs('a','A',[1,2])],[]).size==0


def test_common_visibility_prefers_two_independent_confirmations_and_falls_back():
    rows=[dict(camera='old1',revealed=[],visible_count=100,angle=20),
          dict(camera='old2',revealed=[],visible_count=100,angle=20),
          dict(camera='new1',revealed=[9],visible_count=5,angle=21),
          dict(camera='new2',revealed=[9],visible_count=5,angle=22)]
    assert select_view_pair(['old1','old2'],rows,lambda a,b:True)==(['new1','new2'],True)
    assert select_view_pair(['old1','old2'],rows[:2],lambda a,b:True)==(['old1','old2'],False)
    assert select_view_pair([],rows,lambda a,b:True)==([],False)
    assert len(select_view_pair(['old1'],rows,lambda a,b:True)[0])==1
    assert select_view_pair(['old1','old2'],rows,lambda a,b:False)==(['old1','old2'],False)


def test_size_and_sam_quality_cannot_change_score_or_break_ties():
    quality=evidence_score([.8,.8],.6)
    assert np.isclose(quality['score'],.7125)
    assert quality['size_penalty']==quality['sam_weight']==0
    rows=[dict(id='original',kind='original',member_count=3,quality=quality,semantics={'class':'cup'},core_ids=[],negative_ids=[]),
          dict(id='new',kind='new',member_count=3,quality=quality,semantics={'class':'cup'},core_ids=[],negative_ids=[],sam_quality=1.,size=.1)]
    assert select_evidence(rows,{'original':np.arange(3),'new':np.arange(3)})['id']=='original'


def test_tied_whole_object_requires_independent_added_support():
    q=evidence_score([1.,1.],.6)
    rows=[dict(id='original',kind='original',member_count=3,quality=q,semantics={'class':'phone'},core_ids=[0,1,2],negative_ids=[]),
          dict(id='whole',kind='new',member_count=5,quality=q,semantics={'class':'phone'},core_ids=[0,1,2,3,4],negative_ids=[])]
    members={'original':np.arange(3),'whole':np.arange(5)}
    assert select_evidence(rows,members)['id']=='whole'
    rows[1]['core_ids']=[0,1,2]
    assert select_evidence(rows,members)['id']=='original'


def test_single_view_candidate_is_eligible_but_has_no_protected_core():
    rows=[dict(id='original',kind='original',member_count=3,quality=evidence_score([.1],.4),semantics={'class':'cup'},core_ids=[],negative_ids=[]),
          dict(id='new',kind='new',member_count=3,quality=evidence_score([.8],.7),semantics={'class':'cup'},core_ids=[],negative_ids=[])]
    assert select_evidence(rows,{'original':np.arange(3),'new':np.arange(3,6)})['id']=='new'


def test_score_depends_on_members_and_shared_pool_not_generation_or_duplicate_crops():
    runtime=MultiviewRuntime.__new__(MultiviewRuntime)
    runtime.assets=SimpleNamespace(xyz_scene=np.zeros((8,3)))
    ids=np.arange(8).reshape(2,4)
    descriptors={'a':obs('a','A',[1,2,3],range(8)), 'b':obs('b','B',[1,2,3],range(8))}
    runtime.descriptor=lambda p:descriptors[p]
    runtime.pairs=lambda views,members:[('A','B')]
    runtime.data=lambda camera:dict(ids=ids,reliable=np.ones((2,4),bool))
    runtime.project=lambda camera,members:np.isin(ids,members)
    runtime.observation=lambda p:({},dict(sam=np.isin(ids,[1,2,3]),observed_pixels=np.ones((2,4),bool)),{})
    runtime.classify_members=lambda members,views:dict(score=.7)
    first=runtime.score_evidence([1,2,3],['a','b'],['A','B'])
    second=runtime.score_evidence([1,2,3],['b','a','a'],['A','B'])
    assert first[1]==second[1] and first[1]['valid_view_count']==2
    assert first[2].tolist()==[1,2,3]
    contaminated=runtime.score_evidence([1,2,3,4],['a','b'],['A','B'])
    assert contaminated[1]['projection_iou']==.75  # Reliable background member must be penalized.


def test_actual_feedback_resolves_retained_whole_object_and_missing_is_empty():
    runtime=MultiviewRuntime.__new__(MultiviewRuntime)
    payload=dict(point_labels=[0,0,0,-1],instances={'0':{'object_uid':'B0:0'}},instance_aliases={'C0:3':'B0:0'})
    assert runtime.actual_for(payload,'C0:3').tolist()==[0,1,2]
    assert runtime.actual_for(payload,'C0:4').size==0


def assembly_runtime():
    runtime=MultiviewRuntime.__new__(MultiviewRuntime)
    runtime.b0=dict(point_labels=[0,0,0,-1,-1],instances={'0':{'class':'cup','score':.3}})
    runtime.assets=SimpleNamespace(sources=[SimpleNamespace(candidate_uid='new',support_ids=np.arange(5))],saga20=['cup'])
    runtime.selected=lambda bank:({},np.arange(5))
    runtime.pairs=lambda views,members:[('A','B')] if len(views)>1 else []
    runtime.data=lambda view:dict(reliable=np.ones((1,5),bool),ids=np.arange(5)[None])
    runtime.project=lambda view,members:np.isin(np.arange(5)[None],members)
    def score(members,pool,views):
        refs=[obs(v,v,members,range(5)) for v in ('A','B')]
        return dict(**{'class':'cup'},score=.6),dict(score=len(members)/10),np.asarray(members),np.array([],int),refs
    runtime.score_evidence=score
    return runtime


def test_verified_whole_replacement_keeps_core_under_new_instance_id(tmp_path):
    runtime=assembly_runtime()
    payload=runtime.assemble_evidence({'new':dict(pool=[],views=['A','B'])},tmp_path)
    assert runtime.actual_for(payload,'new').tolist()==list(range(5))
    assert runtime.actual_for(payload,'B0:0').tolist()==list(range(5))
    # Existing evaluator consumes the same normalized output shape.
    assert len(payload['point_labels'])==5 and len(payload['instances'])==1


def test_duplicate_replacement_carries_old_independent_core_before_allocation(tmp_path):
    rt=assembly_runtime()
    rt.b0=dict(point_labels=[0]*6+[-1],instances={'0':{'class':'cup','score':.3}})
    rt.selected=lambda bank:({},np.arange(1,7))
    rt.data=lambda view:dict(reliable=np.ones((1,7),bool),ids=np.arange(7)[None])
    rt.project=lambda view,members:np.isin(np.arange(7)[None],members)
    def score(members,pool,views):
        # The shifted proposal has a higher score, but must not drop old core 0.
        quality=.6 if set(members)==set(range(6)) else .8
        refs=[obs(v,v,members,range(7)) for v in ('A','B')]
        return {'class':'cup','score':.6},{'score':quality},np.array(members),np.array([],int),refs
    rt.score_evidence=score
    result=rt.assemble_evidence({'new':dict(pool=[],views=['A','B'])},tmp_path)
    assert rt.actual_for(result,'new').tolist()==list(range(7))
    assert rt.actual_for(result,'B0:0').tolist()==list(range(7))


def test_no_evidence_improvement_rolls_back_actual_scene(tmp_path):
    runtime=assembly_runtime()
    original=runtime.score_evidence
    def flat_score(members,pool,views):
        semantics,quality,core,negative,refs=original(members,pool,views)
        quality['score']=.3
        return semantics,quality,core,negative,refs
    runtime.score_evidence=flat_score
    payload=runtime.assemble_evidence({'new':dict(pool=[],views=['A','B'])},tmp_path)
    assert payload['point_labels']==[0,0,0,-1,-1]
    assert runtime.actual_for(payload,'new').tolist()==[0,1,2]


def test_soft_alpha_evidence_does_not_require_one_dominant_gaussian_per_pixel(tmp_path):
    runtime=MultiviewRuntime.__new__(MultiviewRuntime)
    runtime.data=lambda v:dict(rgb=np.zeros((2,2,3),np.uint8),ids=np.zeros((2,2),int),reliable=np.zeros((2,2),bool))
    runtime.cameras={'A':None}
    def alpha(camera,masks,valid):
        assert valid.all()  # Three comparable contributors are still an observed surface.
        return SimpleNamespace(inside_mass=np.array([[.6,.6,.6]]),visible_mass=np.array([.6,.6,.6]))
    runtime.renderer=SimpleNamespace(alpha=alpha)
    runtime.occlusion_unknown=lambda v:np.array([],int)
    paths=runtime.adopt_raw('A',np.ones((3,2,2),bool),CropTransform((2,2),0,0,2,2),tmp_path,
        actual_input={},qualities=[.9,.8,.7],source='test')
    with np.load(tmp_path/'mask-0'/'prediction.npz') as z:
        assert z['hard_ids'].size==0 and z['alpha_ids'].tolist()==[0,1,2]
    assert len(paths)==3
