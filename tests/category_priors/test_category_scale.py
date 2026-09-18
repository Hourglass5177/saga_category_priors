import math
from types import SimpleNamespace

import numpy as np

from category_priors.category_scale import (candidate_score, merge_ranked, resolve_prior,
                                           projection_iou, scale_slots, select_candidate)
from category_priors.object_verification.model_adapter import (
    InjectedModelAdapter, point_prior_crop, prior_crop,
)
from run_effective_repair_evaluation import object_iou_assignment


def priors():
    def node(size, active=True):
        return dict(active=active, shrunk={'geometry': {'log_bbox_diag_m': {
            q: math.log(size * factor) for q, factor in [('q25', .5), ('q50', 1), ('q75', 1.5), ('q95', 2)]}}})
    return dict(global_=None, **{'global': node(2)},
                categories={'phone': node(.2), 'book': dict(active=False, fallback='portable')},
                parents={'portable': node(.4)})


def test_metric_point_crop_changes_actual_input_despite_giant_old_box():
    args = dict(image_shape=(600, 1000), focal_geometric_mean=800,
                prior_diagonal_m=.2, positive_optical_z=2)
    old = prior_crop(bbox_xyxy=[0, 0, 1000, 600], **args)
    new = point_prior_crop(point_xy=(900, 570), **args)
    assert old.width == 1000 and new.width == 120
    assert new.image_to_crop_points([[900, 570]]).tolist() == [[60., 60.]]
    image = np.ones((600, 1000, 3), dtype=np.uint8)
    encoded, valid = new.extract(image)
    assert encoded.shape == (120, 120, 3)
    assert not valid.all()  # image edge pads without moving the target


def test_sam_point_only_passes_no_box_and_retains_all_masks():
    class Sam:
        def set_image(self, image):
            self.shape = image.shape[:2]

        def predict(self, **kwargs):
            assert kwargs['box'] is None
            assert kwargs['point_coords'].tolist() == [[32., 32.]]
            return np.ones((3, *self.shape), bool), np.array([.4, .8, .6]), None

    crop = point_prior_crop(image_shape=(64, 64), point_xy=(32, 32), focal_geometric_mean=10,
                            prior_diagonal_m=.1, positive_optical_z=2)
    results = InjectedModelAdapter(sam_predictor=Sam())._sam_masks(
        image=np.zeros((64, 64, 3), np.uint8), crop=crop, box_crop=None,
        point_crop=[[32, 32]], uid='test')
    assert len(results) == 3 and results[1].sam_quality == .8


def test_global_control_keeps_two_slots_three_scales_and_records_parent_fallback():
    p = priors()
    category = scale_slots(p, ['phone', 'book'], [.8, .7], 'category')
    control = scale_slots(p, ['phone', 'book'], [.8, .7], 'global')
    assert len(category) == len(control) == 6
    assert [(r['slot'], r['quantile']) for r in category] == [(r['slot'], r['quantile']) for r in control]
    assert category[0]['source'] == 'phone' and category[3]['source'] == 'portable'
    assert category[0]['category_specific'] and not category[3]['category_specific']
    assert all(r['source'] == 'global' and not r['category_specific'] for r in control)
    assert [r['diagonal_m'] for r in control[:3]] == [r['diagonal_m'] for r in control[3:]]


def test_partial_object_not_forced_to_typical_size_and_ties_keep_original():
    p = resolve_prior(priors(), 'phone')
    quality = candidate_score(overlaps=[.8], semantic_score=.8, sam_qualities=[.8], diagonal_m=.01, prior=p)
    assert quality['size_penalty'] == 0
    oversized = candidate_score(overlaps=[.8], semantic_score=.8, sam_qualities=[.8], diagonal_m=40, prior=p)
    assert oversized['size_penalty'] == .1
    rows = [dict(id=k, kind=k, member_count=3, quality=quality) for k in ('category', 'legacy', 'original')]
    assert select_candidate(rows)['id'] == 'original'


def test_cropped_observation_does_not_penalize_unseen_projected_members():
    # The same observed object, with extra members visible only outside this crop.
    observed = np.array([[1, 1, 0, 0]], bool)
    sam = np.array([[1, 0, 0, 0]], bool)
    compact = sam.copy()
    extended = np.array([[1, 0, 1, 1]], bool)
    assert projection_iou(compact, sam, observed) == 1.
    assert projection_iou(extended, sam, observed) == 1.
    assert projection_iou(extended, sam) == 1 / 3  # reproducible historical behavior
    wrong_inside = np.array([[1, 1, 1, 1]], bool)
    assert projection_iou(wrong_inside, sam, observed) == .5
    assert projection_iou(extended, sam, np.zeros_like(observed)) == 0.


def test_runtime_uses_observation_domain_only_when_explicitly_enabled():
    from category_priors.category_scale_experiment import ScaleRuntime
    runtime = ScaleRuntime.__new__(ScaleRuntime)
    runtime.mode, runtime.scale_priors, runtime.plan = 'category', priors(), {}
    runtime.assets = SimpleNamespace(xyz_scene=np.arange(12).reshape(4, 3)*.001, scale_m_per_unit=1.)
    runtime.project = lambda uid, ids: np.array([[1, 0, 1, 1]], bool)
    runtime.classify_members = lambda ids, views: {'class': 'phone', 'score': .8}
    runtime.observation = lambda path: (
        dict(camera_uid='a', qualities=[.9], chosen=0),
        dict(sam=np.array([[1, 0, 0, 0]], bool), observed_pixels=np.array([[1, 1, 0, 0]], bool)),
        dict(inside=np.ones(4), visible=np.ones(4)))
    assert runtime.score_members(np.array([0, 2, 3]), ['a'])[1]['projection_iou'] == 1/3
    runtime.plan['projection_domain'] = 'observed'
    assert runtime.score_members(np.array([0, 2, 3]), ['a'])[1]['projection_iou'] == 1.


def test_one_to_one_object_iou_maximizes_sum_and_retains_unmatched_gt():
    objects = [SimpleNamespace(gt_id=i, class_name=c) for i, c in enumerate(['phone', 'phone', 'chair'])]
    predictions = [SimpleNamespace(prediction_id=i, class_name='phone') for i in range(2)]
    matrix = np.array([[.9, .8, 0], [.85, 0, .99]])
    aware = object_iou_assignment(matrix, predictions, objects)
    assert [r['iou'] for r in aware] == [.85, .8, 0]
    assert [r['prediction_id'] for r in aware] == [1, 0, None]
    agnostic = object_iou_assignment(matrix, predictions, objects, class_aware=False)
    assert [r['iou'] for r in agnostic] == [.9, 0, .99]
    assert len(object_iou_assignment(np.empty((0, 3)), [], objects)) == 3


def test_ranked_merge_deduplicates_whole_objects_and_preserves_b0_ties():
    b0 = dict(point_labels=[0]*6 + [-1]*4,
              instances={'0': {'class': 'phone', 'score': .8}})
    proposals = [dict(uid='duplicate', members=np.arange(6), **{'class': 'phone', 'score': .9}, selection_score=.8),
                 dict(uid='new', members=np.array([6, 7, 8, 9]), **{'class': 'book', 'score': .8}, selection_score=.7)]
    result = merge_ranked(b0, proposals, {'B0:0': .8}, {'phone', 'book'})
    assert result['suppressed'] == {'duplicate': 'B0:0'}
    assert len(result['payload']['instances']) == 2
    assert len(result['assigned']['B0:0']) == 6
    assert len(result['assigned']['new']) == 4


def test_missing_reliable_point_skips_new_sam_without_substituting_a_box(tmp_path):
    from run_effective_repair import Runtime, read
    runtime = Runtime.__new__(Runtime)
    runtime.np = np
    runtime.assets = SimpleNamespace(xyz_scene=np.zeros((3, 3)), priors=priors())
    runtime.cameras = {'v': SimpleNamespace(fx=100, fy=100,
                                           optical_z_m=lambda xyz: np.ones(len(xyz)))}
    data = dict(ids=np.zeros((4, 4), np.int64), maximum=np.ones((4, 4))*.1,
                opacity=np.ones((4, 4))*.2, rgb=np.zeros((4, 4, 3), np.uint8))
    runtime.data = lambda uid: data
    runtime.project = lambda uid, members: np.ones((4, 4), bool)
    class ForbiddenSam:
        def _sam_masks(self, **kwargs):
            raise AssertionError('new branch must not run without a reliable point')
    runtime.sam = ForbiddenSam()
    runtime.observe('v', np.arange(3), [0], tmp_path, scale_prior={'diagonal_m': .2})
    assert read(tmp_path / 'observation.json')['status'] == 'no_reliable_point'
    assert not (tmp_path / 'prediction.npz').exists()


def test_new_observation_does_not_treat_outside_crop_as_negative(tmp_path):
    from run_effective_repair import Runtime
    runtime = Runtime.__new__(Runtime)
    runtime.np = np
    count = 128**2
    runtime.assets = SimpleNamespace(xyz_scene=np.zeros((count, 3)), priors=priors())
    runtime.cameras = {'v': SimpleNamespace(fx=100, fy=100,
                                           optical_z_m=lambda xyz: np.ones(len(xyz))*2)}
    data = dict(ids=np.arange(count).reshape(128, 128), maximum=np.ones((128, 128)),
                opacity=np.ones((128, 128)), rgb=np.zeros((128, 128, 3), np.uint8),
                reliable=np.ones((128, 128), bool))
    runtime.data = lambda uid: data
    runtime.project = lambda uid, members: np.ones((128, 128), bool)
    def masks(**kwargs):
        assert kwargs['box_crop'] is None
        crop = kwargs['crop']
        mask = np.zeros((128, 128), bool); mask[0, 0] = True
        return [SimpleNamespace(sam_quality=.9, mask_image=mask) for _ in range(3)]
    runtime.sam = SimpleNamespace(_sam_masks=masks)
    def alpha(camera, masks, valid):
        assert valid.sum() == 32**2
        assert not valid[-1, -1]
        return SimpleNamespace(inside_mass=np.zeros((1, count)), visible_mass=np.zeros(count),
                               qualified=lambda i: np.array([0], np.int64))
    runtime.renderer = SimpleNamespace(alpha=alpha)
    runtime.classify = lambda *args: {'status': 'complete', 'cosines': [0.]*32}
    runtime.stats = dict(observation_count=0, observation_seconds=0.)
    runtime.observe('v', np.arange(count), [0], tmp_path, scale_prior={'diagonal_m': .2})
    with np.load(tmp_path / 'prediction.npz') as z:
        assert count-1 not in z['negative_ids']
        assert 1 in z['negative_ids']
        assert not z['observed_pixels'][-1, -1]


def test_candidate_bank_to_scene_export_without_evaluation_data(tmp_path, monkeypatch):
    from PIL import Image
    from run_effective_repair import save, read
    from category_priors.category_scale_experiment import ScaleRuntime
    from category_priors.object_verification.observation import CameraView
    monkeypatch.setattr(Image, 'open', lambda *a, **k: (_ for _ in ()).throw(AssertionError('annotation read')))
    runtime = ScaleRuntime.__new__(ScaleRuntime)
    runtime.np, runtime.out, runtime.mode, runtime.companion = np, tmp_path, 'category', None
    runtime.scale_priors = priors()
    runtime.plan = {'classes32': ['phone', 'book'] + [f'c{i}' for i in range(30)]}
    runtime.assets = SimpleNamespace(xyz_scene=np.arange(18).reshape(6, 3)*.001,
                                    scale_m_per_unit=1., saga20=['phone', 'book', 'table'])
    runtime.b0 = dict(point_labels=[0]*6, instances={'0': {'class': 'table', 'score': .6}})
    runtime.project = lambda uid, ids: np.isin(np.arange(6).reshape(2, 3), ids)
    runtime.classify_members = lambda ids, views: {'class': 'phone', 'score': .8}
    runtime.camera_geometry = lambda uid, ids: CameraView(uid, (0, 0, 1), (0 if uid == 'a' else 1, 0, 0), 2)
    def observe(uid, members, anchors, dest, **kwargs):
        dest.mkdir(parents=True, exist_ok=True)
        mask = np.array([[1, 1, 1], [0, 0, 0]], bool)
        save(dest / 'observation.json', dict(camera_uid=uid, status='complete', chosen=0,
            qualities=[.9, .8, .7], semantics={'cosines': [.8, .7]+[0.]*30}))
        np.savez(dest / 'prediction.npz', sam=mask, hard_ids=np.arange(3), alpha_ids=np.arange(3),
                 negative_ids=np.arange(3, 6))
        np.savez(dest / 'alpha.npz', inside=np.array([1., 1., 1., 0., 0., 0.]), visible=np.ones(6))
        return str(dest)
    runtime.observe = observe
    runtime.old_paths = lambda uid, members, anchors, views, dest, **kw: [
        observe(v, members, anchors, dest / ('old-'+v)) for v in views]
    bank = runtime.build_bank('s:C0:0', np.arange(5), [0], ['a', 'b'], tmp_path / 'bank')
    assert len(bank['candidates']) == 15
    assert len(bank['hypotheses']) == 6
    assert bank['selected_id'].startswith('legacy-')
    actual = runtime.export({'s:C0:0': bank}, tmp_path / 'export')
    assert actual['s:C0:0'].tolist() == [0, 1, 2]
    assert len(read(tmp_path / 'export' / 'scene.json')['instances']) == 2


def test_size_fitting_excludes_all_scans_of_evaluation_physical_scenes(tmp_path, monkeypatch):
    import pytest
    import category_priors.fit_category_sizes as fit
    train = tmp_path / 'train.txt'
    train.write_text('scene0025_00\nscene0025_01\nscene0645_00\nscene0645_02\nscene0001_00\n')
    called = []
    def unavailable(root, sid):
        called.append(sid)
        raise FileNotFoundError(sid)
    monkeypatch.setattr(fit, 'discover_scene_files', unavailable)
    with pytest.raises(ValueError, match='No usable official-training'):
        fit.fit_available_sizes(tmp_path, train, tmp_path / 'priors.json')
    assert called == ['scene0001_00']


def test_bank_oracle_counts_mapped_gt_points_and_uses_each_source_only_once():
    from category_priors.category_scale_diagnostics import gaussian_overlap_index, member_overlaps, assign_sources
    objects = [SimpleNamespace(gt_id='a', class_name='phone', mask=np.array([1, 1, 0, 0, 0], bool)),
               SimpleNamespace(gt_id='b', class_name='phone', mask=np.array([0, 0, 1, 1, 0], bool))]
    # Two GT vertices map to Gaussian zero; unmapped GT remains in the denominator.
    mapping = np.array([0, 0, 1, -1, 2])
    index = gaussian_overlap_index(mapping, objects, 3)
    assert np.allclose(member_overlaps(np.array([0]), index), [1, 0])
    assert np.allclose(member_overlaps(np.array([1]), index), [0, .5])
    matrix = np.array([[1., .9]])
    chosen = assign_sources(matrix, np.array([['variant-a', 'variant-b']], dtype=object), ['source'], objects)
    assert [r['iou'] for r in chosen] == [1., 0.]
