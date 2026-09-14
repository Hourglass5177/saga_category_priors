"""Approved DEV2 guided SAM -> actual scene output -> matched-view feedback study."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

BASE = Path(os.environ.get('SAGA_ROOT', Path(__file__).resolve().parent))
VERSION = 'dev2-object-scope-v2-20260909'
ROOT = BASE / 'artifacts' / VERSION


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, value, *, compact=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                               indent=None if compact else 2), encoding='utf8')
    temp.replace(path)


def slug(uid):
    return uid.replace(':', '_')


def prepare(out):
    if (out / 'plan.json').exists():
        return
    previous = read(ROOT / 'c4-v1-mask-transfer-01/plan.json')
    plan = {k: previous[k] for k in ('bindings', 'classes32', 'sam_spec', 'alpha_spec')}
    plan.update(experiment='effective-repair-20260911', experiment_version=VERSION,
                methods=['A', 'B', 'C'], protocols=['initial', 'open', 'feedback'],
                input_rule='full source projection bbox; original global D50 crop; reliable positive or box only',
                selection='highest SAM quality; original index breaks exact ties',
                G0='two independent hard-or-qualified-alpha supports minus every reliable negative',
                G2='sum inside >= .5 and sum inside / sum visible >= .5; minimum 3',
                semantics='equal-weight 32 cosine mean; unique top1; (1+top1)/2 uncalibrated',
                C='G2, SAM region class, unique maximum member normalized visible contribution',
                view_order='original C0 visible contributor Gaussian count descending, image_name tie; independent',
                local_cap_seconds=3600, recovery_reserve_seconds=3600, estimate_margin=1.5,
                denominator=374, local_cases=[])
    evaluations = []
    for row in previous['cases']:
        case = {k: row[k] for k in ('case_id', 'scene_id', 'candidate_uid', 'views')}
        case.update(group='c4', source_kind='C0', reuse=str(ROOT / 'c4-v1-mask-transfer-01/old_saved' / row['case_id']))
        plan['local_cases'].append(case)
    evaluations.extend(read(ROOT / 'c4-v1-mask-transfer-01/offline-evaluation-inputs.json'))
    catalog = read(ROOT / 'supplementary-catalog-02/report.json')
    for row in catalog['records']:
        uid = row['proposal']['uid']
        case_id = 'supp-' + slug(uid)
        plan['local_cases'].append(dict(case_id=case_id, scene_id=row['proposal']['scene_id'],
            candidate_uid=uid, group='supplementary', source_kind='B0',
            views=[dict(camera_uid=v['camera_uid'], rgb=v['rgb']['path'], role=('I1', 'I2', 'H')[i])
                   for i, v in enumerate(row['views'])]))
        evaluations.append(dict(case_id=case_id, group='supplementary',
            human_class=row['human']['interpreted']['human_class'],
            views=[dict(camera_uid=v['camera_uid'], foreground=v['foreground']['path'],
                        uncertain=v['uncertain']['path']) for v in row['views']]))
    # Existing contributor arrays: the four diagnostic mask NPZs are not contributor arrays.
    contributor_files = {}
    for case in previous['cases']:
        for view in case['views']:
            contributor_files[view['camera_uid']] = dict(path=view['contributors'], kind='c4')
    historic = read(out / 'historical-contributors.json')
    for row in historic['rows']:
        if row.get('status') == 'measured_from_saved_arrays':
            contributor_files[row['camera_uid']] = dict(path=row['cache'], kind='old_npz')
    fresh = read(ROOT / 'contributor-output-01/result.json')
    for row in fresh['rows']:
        ref = row['cache_record']['path']
        contributor_files[row['camera_uid']] = dict(kind='npy_record', path=ref, arrays=row['arrays'])
    plan['contributor_files'] = contributor_files
    save(out / 'plan.json', plan)
    save(out / 'offline-evaluation-inputs.json', evaluations)


class Runtime:
    def __init__(self, out):
        import numpy as np
        import torch
        from category_priors.object_scope.native_models import load_sam
        from category_priors.object_scope.alpha_clip_model import load_local_alpha_clip
        from category_priors.object_verification.model_adapter import InjectedModelAdapter
        self.np, self.torch = np, torch
        self.out, self.plan = out, read(out / 'plan.json')
        self.stats = dict(observation_count=0, observation_seconds=0., semantic_count=0,
                          semantic_seconds=0., contributor_count=0, contributor_seconds=0., reused_observations=0)
        self.alpha = load_local_alpha_clip(read(self.plan['alpha_spec']), device='cuda')
        self.sam = InjectedModelAdapter(sam_predictor=load_sam(read(self.plan['sam_spec']), device='cuda').predictor)
        with torch.inference_mode():
            tokens = self.alpha.tokenize([f'a photo of a {c}.' for c in self.plan['classes32']]).to('cuda')
            self.text_features = self.alpha.model.encode_text(tokens).float().cpu().numpy()
        np.save(out / 'text_features.npy', self.text_features)
        self.data_cache = OrderedDict()

    def scene(self, sid):
        from category_priors.object_verification.frozen_scene import FrozenSceneAssets, NativeMeasurementRenderer
        if hasattr(self, 'renderer'):
            del self.renderer, self.assets
            self.data_cache.clear()
            self.torch.cuda.empty_cache()
        binding = self.plan['bindings'][sid]
        self.assets = FrozenSceneAssets.load(binding['spec']['path'], binding['assets']['path'])
        self.renderer = NativeMeasurementRenderer(self.assets)
        self.cameras = {c.uid: c for c in self.assets.cameras}
        self.sources = {s.candidate_uid: s for s in self.assets.sources}
        self.b0 = read(read(binding['spec']['path'])['b0_output'])
        self.original = {o.uid: o for o in self.assets.b0_scene.objects}
        self.sid = sid

    def data(self, uid):
        np = self.np
        from PIL import Image
        if uid in self.data_cache:
            self.data_cache.move_to_end(uid)
            return self.data_cache[uid]
        camera = self.cameras[uid]
        rgb = np.asarray(Image.open(camera.image_path).convert('RGB'))
        saved = self.out / 'contributors' / self.sid / (slug(uid) + '.npz')
        ref = self.plan['contributor_files'].get(uid)
        if saved.exists():
            with np.load(saved, allow_pickle=False) as z:
                ids, maximum, opacity = z['ids'], z['maximum'], z['opacity']
        elif ref:
            if ref['kind'] == 'npy_record':
                root = Path(ref['path']).parents[2]
                arr = ref['arrays']
                ids = np.load(root / arr['ids']['path'], allow_pickle=False)
                maximum = np.load(root / arr['max']['path'], allow_pickle=False)
                opacity = np.load(root / arr['opacity']['path'], allow_pickle=False)
            else:
                with np.load(ref['path'], allow_pickle=False) as z:
                    ids = z['contributor_ids'] if 'contributor_ids' in z else z['ids']
                    maximum = z['max_contribution'] if 'max_contribution' in z else z['weights']
                    opacity = z['opacity']
        else:
            began = time.monotonic()
            ids, maximum, opacity = self.renderer.contributors(camera, rgb)
            self.stats['contributor_count'] += 1
            self.stats['contributor_seconds'] += time.monotonic() - began
        if not saved.exists():
            saved.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(saved, ids=ids, maximum=maximum, opacity=opacity)
        if rgb.shape[:2] != ids.shape or ids.shape != opacity.shape:
            raise ValueError('RGB/contributor axes do not match ' + uid)
        positive = (ids >= 0) & (maximum > 0) & (opacity > 0)
        data = dict(rgb=rgb, ids=ids, maximum=maximum, opacity=opacity, positive=positive,
                    reliable=positive & (opacity >= .5) & (maximum >= .5 * opacity))
        self.data_cache[uid] = data
        while len(self.data_cache) > 5:
            self.data_cache.popitem(last=False)
        return data

    def project(self, uid, members):
        data = self.data(uid)
        return data['positive'] & self.np.isin(data['ids'], members)

    def classify(self, uid, mask):
        np, torch = self.np, self.torch
        from category_priors.object_scope.encoding import CropEncodingPlan, ENCODING_VERSION, _box, _rectangle, encode_region, full_image_plan
        from category_priors.object_scope.semantics import cosine_scores
        key = hashlib.blake2b(mask.tobytes(), digest_size=16).hexdigest()
        prefix = self.out / 'semantics' / slug(uid) / key
        if prefix.with_suffix('.json').exists():
            return read(prefix.with_suffix('.json'))
        began = time.monotonic()
        rgb = self.data(uid)['rgb']
        box = _box(mask)
        if box is None:
            cp = full_image_plan(mask.shape)
        else:
            side = max(1, math.ceil(1.5 * max(box[2] - box[0], box[3] - box[1])))
            cp = CropEncodingPlan(mask.shape, _rectangle(mask.shape, box, side), 'detail', side,
                                 dict(bbox_xyxy=box, rule='1.5-current-bbox', version=ENCODING_VERSION))
        encoded = encode_region(rgb, mask, np.ones(mask.shape, bool), cp, alpha_mode='object')
        row = dict(top1=None, cosines=None, status='unknown', encoding=encoded.trace)
        if encoded.valid:
            x = torch.from_numpy(encoded.rgb_tensor[None].copy()).to('cuda', dtype=self.alpha.model.dtype)
            a = torch.from_numpy(encoded.alpha_tensor[None].copy()).to('cuda', dtype=self.alpha.model.dtype)
            with torch.inference_mode():
                feature = self.alpha.model.visual(x, a).float().cpu().numpy()[0]
            cos = cosine_scores(feature, self.text_features)
            winners = np.flatnonzero(cos == cos.max())
            row.update(top1=self.plan['classes32'][int(winners[0])] if len(winners) == 1 else None,
                       cosines=cos.tolist(), status='complete')
            prefix.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(prefix.with_suffix('.npz'), features=feature,
                                encoded_rgb=encoded.rgb_tensor, encoded_alpha=encoded.alpha_tensor)
            row['input_npz'] = str(prefix.with_suffix('.npz'))
            self.stats['semantic_count'] += 1
        row['camera_uid'] = uid
        save(prefix.with_suffix('.json'), row)
        self.stats['semantic_seconds'] += time.monotonic() - began
        return row

    def classify_members(self, members, views):
        from category_priors.effective_repair_core import mean32
        values = [self.classify(v, self.project(v, members)) for v in views]
        result = mean32([v['cosines'] for v in values], self.plan['classes32'])
        result['views'] = values
        return result

    def camera_geometry(self, uid, members):
        from category_priors.object_verification.observation import CameraView
        np = self.np
        camera = self.cameras[uid]
        center = np.median(self.assets.xyz_scene[members], axis=0) * self.assets.scale_m_per_unit
        ray = center - camera.center_m
        if np.linalg.norm(ray) == 0 or float(camera.optical_z_m(center / self.assets.scale_m_per_unit)) <= 0:
            return None
        return CameraView(uid, tuple(ray), tuple(camera.center_m), float(np.linalg.norm(ray)))

    def observe(self, uid, members, anchors, dest, *, reuse=None, source_state='original'):
        np = self.np
        from category_priors.object_verification.model_adapter import prior_crop, reliable_prompt_point
        if (dest / 'observation.json').exists():
            return str(dest)
        began = time.monotonic()
        d = self.data(uid)
        pixels = self.project(uid, members)
        original_status = read(Path(reuse) / 'observation.json').get('status') if reuse else None
        row = dict(camera_uid=uid, source_state=source_state, status='unlocalized', point=None,
                   input_changed=bool(reuse and original_status == 'complete'), replay_of=None)
        if not pixels.any():
            save(dest / 'observation.json', row)
            return str(dest)
        camera = self.cameras[uid]
        visible_ids = np.unique(d['ids'][pixels])
        depths = camera.optical_z_m(self.assets.xyz_scene[visible_ids])
        depths = depths[depths > 0]
        if not len(depths):
            row['status'] = 'no_positive_optical_depth'
            save(dest / 'observation.json', row)
            return str(dest)
        y, x = np.where(pixels)
        box = [float(x.min()), float(y.min()), float(x.max() + 1), float(y.max() + 1)]
        prior = self.assets.priors['global']
        statistic = 'shrunk' if 'shrunk' in prior else 'raw'
        d50 = math.exp(prior[statistic]['geometry']['log_bbox_diag_m']['q50'])
        depth = float(np.median(depths))
        crop = prior_crop(image_shape=pixels.shape, bbox_xyxy=box,
                          focal_geometric_mean=math.sqrt(camera.fx * camera.fy),
                          prior_diagonal_m=d50, positive_optical_z=depth)
        encoded, crop_valid = crop.extract(d['rgb'])
        valid = crop.mask_to_image(crop_valid)
        locator = SimpleNamespace(bbox_xyxy=box, anchor_ids=anchors, support_ids=members)
        point = reliable_prompt_point(locator=locator, contributor_ids=d['ids'],
            max_contribution=d['maximum'], opacity=d['opacity'], valid_pixels=valid)
        point_image = None if point is None else list(point['point_image_xy'])
        box_crop = list(crop.image_to_crop_box(box, clip=True))
        actual_input = dict(crop=asdict(crop), box_crop=box_crop, point_image=point_image)
        # requested_side is bookkeeping: the encoder sees integer crop dimensions and the prompts.
        comparison_input = dict(crop={k: v for k, v in actual_input['crop'].items() if k != 'requested_side'},
                                box_crop=box_crop, point_image=point_image)
        comparison_input = json.loads(json.dumps(comparison_input))
        if reuse and (Path(reuse) / 'observation.json').exists():
            old = read(Path(reuse) / 'observation.json')
            if old.get('comparison_input') == comparison_input:
                row.update(status=old['status'], replay_of=str(reuse), input_changed=False,
                           actual_input=actual_input, comparison_input=comparison_input)
                save(dest / 'observation.json', row)
                self.stats['reused_observations'] += 1
                return str(dest)
        point_crop = None if point_image is None else crop.image_to_crop_points(np.asarray(point_image))
        alternatives = self.sam._sam_masks(image=encoded, crop=crop, box_crop=box_crop,
                                          point_crop=point_crop, uid=slug(uid))
        qualities = np.asarray([m.sam_quality for m in alternatives])
        chosen = int(np.argmax(qualities))
        masks = np.stack([m.mask_image for m in alternatives])
        selected = masks[chosen]
        measured = self.renderer.alpha(camera, selected[None], np.ones(selected.shape, bool))
        hard = np.unique(d['ids'][d['reliable'] & selected])
        negative = np.unique(d['ids'][d['reliable'] & ~selected])
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest / 'prediction.npz', sam=selected, alternatives=masks, qualities=qualities,
                            encoded_rgb=encoded, source=pixels, source_members=members,
                            hard_ids=hard, negative_ids=negative, alpha_ids=np.asarray(measured.qualified(0), np.int64))
        np.savez_compressed(dest / 'alpha.npz', inside=measured.inside_mass[0], visible=measured.visible_mass)
        semantic = self.classify(uid, selected)
        row.update(status='complete', chosen=chosen, qualities=qualities.tolist(), point=point,
                   actual_input=actual_input, comparison_input=comparison_input,
                   prior_D50_m=d50, positive_optical_depth_m=depth, semantics=semantic,
                   input_changed=bool(reuse), seconds=time.monotonic() - began)
        save(dest / 'observation.json', row)
        self.stats['observation_count'] += 1
        self.stats['observation_seconds'] += row['seconds']
        return str(dest)

    def observation(self, path):
        path = Path(path)
        row = read(path / 'observation.json')
        if row.get('replay_of'):
            return self.observation(row['replay_of'])
        if row['status'] != 'complete':
            return row, None, None
        with self.np.load(path / 'prediction.npz', allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        with self.np.load(path / 'alpha.npz', allow_pickle=False) as z:
            masses = dict(inside=z['inside'], visible=z['visible'])
        return row, arrays, masses

    def geometry(self, paths, members, dest):
        np = self.np
        from category_priors.effective_repair_core import g0_members, g2_members, mean32
        if (dest / 'geometry.json').exists():
            with np.load(dest / 'geometry.npz', allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            return read(dest / 'geometry.json'), arrays
        observations, inside, visible, semantic, views = [], [], [], [], []
        for path in paths:
            row, arrays, masses = self.observation(path)
            views.append(row['camera_uid'])
            semantic.append(row.get('semantics', {}).get('cosines'))
            if arrays is not None:
                observations.append(dict(camera_uid=row['camera_uid'], **{k: arrays[k] for k in ('hard_ids', 'alpha_ids', 'negative_ids')}))
                inside.append(masses['inside']); visible.append(masses['visible'])
        from category_priors.object_verification.observation import cameras_independent
        cameras = [self.camera_geometry(v, members) for v in views]
        pairs = [(a.camera_uid, b.camera_uid) for a, b in itertools.combinations(cameras, 2)
                 if a is not None and b is not None and cameras_independent(a, b)]
        g0 = g0_members(observations, pairs)
        if inside:
            g2 = g2_members(np.stack(inside), np.stack(visible))
            g2_ids, ratio = g2['members'], g2['ratio'][g2['members']]
        else:
            g2_ids, ratio = np.empty(0, np.int64), np.empty(0)
        if len(paths) < 2:
            g0, g2_ids, ratio = np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0)
        row = dict(paths=paths, views=views, independent_pairs=pairs, valid_observations=len(observations),
                   sam_semantics=mean32(semantic, self.plan['classes32']), G0_count=len(g0), G2_count=len(g2_ids))
        dest.mkdir(parents=True, exist_ok=True)
        arrays = dict(G0=g0, G2=g2_ids, G2_ratio=ratio)
        np.savez_compressed(dest / 'geometry.npz', **arrays)
        save(dest / 'geometry.json', row)
        return row, arrays

    def proposal(self, uid, paths, members, dest, method):
        row, arrays = self.geometry(paths, members, dest)
        key = 'G0' if method == 'A' else 'G2'
        sempath = dest / (key + '-semantics.json')
        if method == 'C':
            semantics = row['sam_semantics']
        elif sempath.exists():
            semantics = read(sempath)
        else:
            semantics = self.classify_members(arrays[key], row['views'])
            save(sempath, semantics)
        return dict(uid=uid, members=arrays[key], ratio=arrays['G2_ratio'] if method == 'C' else None,
                    **{k: semantics[k] for k in ('class', 'score')}), row, arrays

    def assemble(self, method, protocol, records, dest):
        np = self.np
        from category_priors.effective_repair_core import merge_scene
        if (dest / 'summary.json').exists():
            actual = {}
            for uid in records:
                with np.load(dest / 'proposals' / slug(uid) / 'members.npz', allow_pickle=False) as saved:
                    actual[uid] = saved['exported_members']
            return actual
        proposals, views_by_uid = [], {}
        for uid, item in records.items():
            prop, row, _ = self.proposal(uid, item['paths'], item['members'], Path(item['geometry']), method)
            proposals.append(prop)
            views_by_uid[uid] = row['views']
        def classify_actual(uid, members):
            return self.classify_members(members, views_by_uid[uid])
        merged = merge_scene(self.b0, proposals, self.assets.saga20, policy=method, classify_actual=classify_actual)
        save(dest / 'scene.json', merged['payload'], compact=True)
        actual = {}
        for uid, row in merged['proposals'].items():
            item = dict(row)
            arrays = {k: item.pop(k) for k in ('raw_members', 'assigned_members', 'exported_members')}
            path = dest / 'proposals' / slug(uid)
            path.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path / 'members.npz', **arrays)
            save(path / 'result.json', item)
            actual[uid] = arrays['exported_members']
        save(dest / 'summary.json', dict(method=method, protocol=protocol, scene_id=self.sid,
            **merged['summary'], alias_to_representative=merged['alias_to_representative'],
            export_id_by_uid=merged['export_id_by_uid']))
        return actual

    def import_c4(self, case, dest):
        np = self.np
        old = Path(case['reuse'])
        old_result = read(old / 'result.json')
        paths = []
        for i, view in enumerate(case['views'][:2]):
            obs = dest / f'observation-{i}'
            paths.append(str(obs))
            if (obs / 'observation.json').exists():
                continue
            d = self.data(view['camera_uid'])
            with np.load(old / f'view-{i}-prediction.npz', allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            with np.load(old / f'view-{i}.npz', allow_pickle=False) as z:
                arrays['encoded_rgb'] = z['encoded_rgb']
                input_info = dict(crop=old_result['views'][i]['crop'],
                    box_image=z['actual_box_image_xyxy'].tolist(),
                    box_crop=z['actual_box_crop_xyxy'].tolist(),
                    point_image=z['positive_point_image_xy'].tolist())
                original_masks = {name: z[name] for name in ('sam', 'G0', 'G2')}
            # Seed the same-input semantic cache with previously measured C4 features.
            # Actual changed members after scene assignment still get new encodings.
            for name, original_mask in original_masks.items():
                key = hashlib.blake2b(original_mask.tobytes(), digest_size=16).hexdigest()
                semantic_path = self.out / 'semantics' / slug(view['camera_uid']) / (key + '.json')
                semantic = dict(old_result['views'][i]['semantics'][name])
                semantic.update(camera_uid=view['camera_uid'],
                                input_npz=str(old / f'view-{i}-{name}-semantic.npz'), reused_existing_C4=True)
                if not semantic_path.exists():
                    save(semantic_path, semantic)
            with np.load(old / f'view-{i}-alpha.npz', allow_pickle=False) as z:
                inside, visible = z['inside'], z['visible']
            ratio = np.divide(inside, visible, out=np.zeros_like(inside), where=visible > 0)
            mask = arrays['sam']
            arrays.update(hard_ids=np.unique(d['ids'][d['reliable'] & mask]),
                          negative_ids=np.unique(d['ids'][d['reliable'] & ~mask]),
                          alpha_ids=np.flatnonzero((inside >= .5) & (ratio >= .5)).astype(np.int64))
            obs.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(obs / 'prediction.npz', **arrays)
            np.savez_compressed(obs / 'alpha.npz', inside=inside, visible=visible)
            prior_view = old_result['views'][i]
            save(obs / 'observation.json', dict(camera_uid=view['camera_uid'], status='complete',
                chosen=prior_view['chosen_mask'], qualities=prior_view['qualities'],
                semantics=prior_view['semantics']['sam'], reused_existing_C4=True, reuse_source=str(old),
                actual_input=input_info, input_changed=False))
        return paths

    def local(self):
        np = self.np
        from category_priors.effective_repair_core import pixel_metrics
        from PIL import Image
        began = time.monotonic()
        results = []
        for sid in self.plan['bindings']:
            self.scene(sid)
            cases = [c for c in self.plan['local_cases'] if c['scene_id'] == sid]
            records = {}
            for case in cases:
                uid = case['candidate_uid']
                source = self.sources[uid] if case['source_kind'] == 'C0' else self.original[uid]
                members = np.asarray(source.support_ids if case['source_kind'] == 'C0' else source.members, np.int64)
                anchors = source.anchor_ids if case['source_kind'] == 'C0' else members
                dest = self.out / 'local' / case['case_id']
                if case.get('reuse'):
                    paths = self.import_c4(case, dest)
                else:
                    paths = [self.observe(v['camera_uid'], members, anchors, dest / f'observation-{i}')
                             for i, v in enumerate(case['views'][:2])]
                records[uid] = dict(paths=paths, members=members, geometry=str(dest / 'geometry'))
                self.geometry(paths, members, dest / 'geometry')
                print('local observations', case['case_id'], flush=True)
            final = {m: self.assemble(m, 'initial', records, self.out / 'local-scenes' / sid / m) for m in 'ABC'}
            # Every prediction in this scene has been saved before human evaluation pixels are opened.
            evals = {c['case_id']: c for c in read(self.out / 'offline-evaluation-inputs.json')}
            for case in cases:
                uid, dest = case['candidate_uid'], self.out / 'local' / case['case_id']
                item = records[uid]
                geom, arrays = self.geometry(item['paths'], item['members'], dest / 'geometry')
                if case.get('reuse'):
                    with np.load(Path(case['reuse']) / 'proposal.npz', allow_pickle=False) as old:
                        if not np.array_equal(arrays['G0'], old['G0']) or not np.array_equal(arrays['G2'], old['G2']):
                            raise ValueError('C4 member reconstruction differs: ' + uid)
                views = []
                source_semantics = self.classify_members(item['members'], geom['views'])
                save(dest / 'source-semantics.json', source_semantics)
                for i, view in enumerate(case['views']):
                    camera_uid = view['camera_uid']
                    data = self.data(camera_uid)
                    evaluation = evals[case['case_id']]['views'][i]
                    fg = np.asarray(Image.open(evaluation['foreground'])) > 0
                    uncertain = np.asarray(Image.open(evaluation['uncertain'])) > 0
                    masks = dict(source=self.project(camera_uid, item['members']),
                                 G0=self.project(camera_uid, arrays['G0']), G2=self.project(camera_uid, arrays['G2']))
                    masks.update({f'final{m}': self.project(camera_uid, final[m][uid]) for m in 'ABC'})
                    semantics = {}
                    if i < 2:
                        obs, sam, _ = self.observation(item['paths'][i])
                        if sam is not None:
                            masks['sam'] = sam['sam']
                            masks['alternatives'] = sam['alternatives']
                        semantics['source'] = source_semantics['views'][i]
                        semantics['sam'] = obs.get('semantics', {})
                        for key in ('G0', 'G2', 'finalA', 'finalB', 'finalC'):
                            semantics[key] = self.classify(camera_uid, masks[key])
                    scores = {k: pixel_metrics(mask, fg, uncertain) for k, mask in masks.items() if mask.ndim == 2}
                    base = masks['source']
                    changes = {k: dict(added_foreground=int((mask & ~base & fg & ~uncertain).sum()),
                        added_background=int((mask & ~base & ~fg & ~uncertain).sum()),
                        lost_correct=int((~mask & base & fg & ~uncertain).sum()))
                        for k, mask in masks.items() if mask.ndim == 2 and k != 'source'}
                    np.savez_compressed(dest / f'view-{i}.npz', rgb=data['rgb'], foreground=fg, uncertain=uncertain, **masks)
                    vr = dict(camera_uid=camera_uid, role=('I1', 'I2', 'H')[i], npz=f'view-{i}.npz',
                              metrics=scores, changes=changes, semantics=semantics)
                    save(dest / f'view-{i}.json', vr)
                    views.append(vr)
                row = dict(case_id=case['case_id'], scene_id=sid, group=case['group'], candidate_uid=uid,
                           execution_complete=True, views=views, geometry=geom)
                save(dest / 'result.json', row)
                results.append(row)
                print('local complete', case['case_id'], 'H', {k: v['iou'] for k, v in views[-1]['metrics'].items()}, flush=True)
        save(self.out / 'local-result.json', dict(execution_complete=len(results) == 10, results=results,
             stats=self.stats, seconds=time.monotonic() - began, C4_reused=4, new_supplementary_targets=6))

    def panels(self):
        np = self.np
        from scipy.sparse import csr_matrix
        from category_priors.object_verification.observation import cameras_independent
        dest = self.out / 'scenes' / self.sid
        if (dest / 'panels.json').exists():
            return read(dest / 'panels.json')
        sources = sorted(self.assets.sources, key=lambda s: s.candidate_uid)
        counts = np.zeros((len(sources), len(self.assets.cameras)), np.int32)
        lengths = [len(s.support_ids) for s in sources]
        indices = np.concatenate([np.asarray(s.support_ids, np.int32) for s in sources])
        incidence = csr_matrix((np.ones(len(indices), np.int32), indices,
                                np.r_[0, np.cumsum(lengths)]), shape=(len(sources), len(self.assets.xyz_scene)))
        for j, camera in enumerate(self.assets.cameras):
            d = self.data(camera.uid)
            visible = np.zeros(len(self.assets.xyz_scene), np.int32)
            visible[np.unique(d['ids'][d['positive']])] = 1
            counts[:, j] = incidence @ visible
            if j % 40 == 0:
                print('view order', self.sid, j, '/', len(self.assets.cameras), flush=True)
        panel = {}
        for i, source in enumerate(sources):
            ordered = sorted(range(len(self.assets.cameras)), key=lambda j: (-int(counts[i, j]), self.assets.cameras[j].image_name))
            chosen, geometries = [], []
            members = np.asarray(source.support_ids, np.int64)
            for j in ordered:
                if counts[i, j] == 0:
                    break
                camera = self.assets.cameras[j]
                geometry = self.camera_geometry(camera.uid, members)
                if geometry is not None and all(cameras_independent(geometry, old) for old in geometries):
                    chosen.append(camera.uid); geometries.append(geometry)
                    if len(chosen) == 5:
                        break
            if len(chosen) >= 3:
                construction, heldout = chosen[:-1], chosen[-1]
            else:
                construction, heldout = chosen, None
            panel[source.candidate_uid] = dict(construction=construction, heldout=heldout,
                selected=chosen, camera_geometry=[asdict(v) for v in geometries],
                ranked_visible_counts=[dict(camera_uid=self.assets.cameras[j].uid, count=int(counts[i, j])) for j in ordered])
        dest.mkdir(parents=True, exist_ok=True)
        np.save(dest / 'view_counts.npy', counts)
        save(dest / 'panels.json', panel)
        return panel

    def full(self):
        np = self.np
        scene_results = []
        for sid in self.plan['bindings']:
            self.scene(sid)
            dest = self.out / 'scenes' / sid
            panel = self.panels()  # Frozen from original C0 before any SAM inference on this scene.
            initial, opened = {}, {}
            for index, source in enumerate(sorted(self.assets.sources, key=lambda s: s.candidate_uid)):
                uid = source.candidate_uid
                members = np.asarray(source.support_ids, np.int64)
                views = panel[uid]['construction']
                obsdir = dest / 'observations' / slug(uid) / 'original'
                paths = [self.observe(v, members, source.anchor_ids, obsdir / slug(v)) for v in views[:2]]
                initial[uid] = dict(paths=paths, members=members, geometry=str(dest / 'geometry' / slug(uid) / 'initial'))
                if index % 10 == 0:
                    print('initial observations', sid, index + 1, '/', len(self.assets.sources), flush=True)
            initial_actual = {m: self.assemble(m, 'initial', initial, dest / f'{m}_initial') for m in 'ABC'}
            for index, source in enumerate(sorted(self.assets.sources, key=lambda s: s.candidate_uid)):
                uid = source.candidate_uid
                views = panel[uid]['construction']
                original = initial[uid]
                extra = [self.observe(v, original['members'], source.anchor_ids,
                                     dest / 'observations' / slug(uid) / 'original' / slug(v)) for v in views[2:]]
                opened[uid] = dict(paths=original['paths'] + extra, members=original['members'],
                                   geometry=str(dest / 'geometry' / slug(uid) / ('open' if extra else 'initial')))
                if index % 15 == 0:
                    print('open observations', sid, index + 1, '/', len(self.assets.sources), flush=True)
            open_actual = {m: self.assemble(m, 'open', opened, dest / f'{m}_open') for m in 'ABC'}
            feedback_actual, feedback_records = {}, {}
            for method in 'ABC':
                records, changes = {}, []
                for index, source in enumerate(sorted(self.assets.sources, key=lambda s: s.candidate_uid)):
                    uid = source.candidate_uid
                    original = initial[uid]
                    actual = initial_actual[method][uid]
                    committed = len(actual) >= 3
                    locator = actual if committed else original['members']
                    anchors = locator if committed else source.anchor_ids
                    paths = list(original['paths'])
                    for i, v in enumerate(panel[uid]['construction'][2:], 2):
                        ref = opened[uid]['paths'][i]
                        target = dest / 'observations' / slug(uid) / ('feedback-' + method) / slug(v)
                        paths.append(self.observe(v, locator, anchors, target, reuse=ref,
                            source_state=str(dest / f'{method}_initial' / 'proposals' / slug(uid) / 'members.npz') if committed else 'original_no_actual_write'))
                    # A failed feedback locator has no encoded input. It must not
                    # borrow a successful open observation merely because its
                    # early-return metadata omitted the input_changed flag.
                    # Conversely, replay wrappers with the same actual input
                    # are equivalent, as are two absent model observations.
                    view_changes = []
                    for path, ref in zip(paths[2:], opened[uid]['paths'][2:]):
                        observed = read(Path(path) / 'observation.json')
                        original_view = read(Path(ref) / 'observation.json')
                        input_changed = (
                            observed.get('comparison_input') != original_view.get('comparison_input')
                            or (observed['status'] == 'complete') != (original_view['status'] == 'complete'))
                        view_changes.append(dict(camera_uid=observed['camera_uid'], input_changed=input_changed,
                            status=observed['status'], open_status=original_view['status'],
                            replay_of=observed.get('replay_of')))
                    changed = any(row['input_changed'] for row in view_changes)
                    recdir = dest / 'geometry' / slug(uid) / ('feedback-' + method if changed else ('open' if len(paths) > 2 else 'initial'))
                    records[uid] = dict(paths=paths, members=original['members'], geometry=str(recdir))
                    changes.append(dict(uid=uid, actual_initial_member_count=len(actual),
                        original_member_count=len(original['members']), used_actual_initial=committed,
                        new_view_count=max(0, len(paths) - 2), input_changed=changed,
                        new_views=view_changes,
                        actual_members_file=str(dest / f'{method}_initial' / 'proposals' / slug(uid) / 'members.npz')))
                    if index % 15 == 0:
                        print('feedback observations', sid, method, index + 1, '/', len(self.assets.sources), flush=True)
                feedback_actual[method] = self.assemble(method, 'feedback', records, dest / f'{method}_feedback')
                save(dest / f'{method}_feedback' / 'feedback.json', changes)
                feedback_records[method] = changes
            # Candidate projections on H are computed after all nine final outputs are saved.
            for source in self.assets.sources:
                uid = source.candidate_uid
                h = panel[uid]['heldout']
                if h is None:
                    continue
                d = self.data(h)
                masks = dict(source=self.project(h, source.support_ids))
                for m in 'ABC':
                    masks[m + '_initial'] = self.project(h, initial_actual[m][uid])
                    masks[m + '_open'] = self.project(h, open_actual[m][uid])
                    masks[m + '_feedback'] = self.project(h, feedback_actual[m][uid])
                path = dest / 'heldout' / (slug(uid) + '.npz')
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, rgb=d['rgb'], **masks)
            summary = dict(scene_id=sid, C0_count=len(self.assets.sources), conditions_complete=9,
                feedback_changed={m: sum(r['input_changed'] for r in feedback_records[m]) for m in 'ABC'},
                with_extra_views=sum(len(p['construction']) > 2 for p in panel.values()),
                insufficient_initial_views=sum(len(p['construction']) < 2 for p in panel.values()))
            save(dest / 'result.json', summary)
            scene_results.append(summary)
            print('scene complete', summary, flush=True)
        count = sum(s['C0_count'] for s in scene_results)
        if count != self.plan['denominator']:
            raise ValueError(f'Expected 374 original C0 inputs, got {count}')
        save(self.out / 'scene-result.json', dict(execution_complete=True, scenes=scene_results, stats=self.stats))


def worker(out, stage):
    began = time.monotonic()
    try:
        runtime = Runtime(out)
        runtime.local() if stage == 'local' else runtime.full()
        save(out / f'{stage}-worker.json', dict(execution_complete=True, seconds=time.monotonic() - began, stats=runtime.stats))
        return 0
    except Exception:
        error = dict(stage=stage, execution_complete=False, seconds=time.monotonic() - began, traceback=traceback.format_exc())
        save(out / f'{stage}-failure-{int(time.time())}.json', error)
        print(error['traceback'], flush=True)
        return 1


def evaluation_manifest(out):
    original = read(BASE / 'artifacts/dev2-crop-prior-v2-20260907/continuous-v3/continuous_jobs/dev2-continuous-v3-20260907/build-formal-evaluation-manifests/attempt-001/output/evaluation_round2.json')
    conditions = [m + '_' + protocol for m in 'ABC' for protocol in ('initial', 'open', 'feedback')]
    scenes = []
    for row in original['scenes']:
        scene = {k: row[k] for k in ('scene_id', 'gt_npz', 'gaussian_ply', 'gaussian_to_gt_transform', 'b0_output_json')}
        scene['condition_outputs'] = {c: dict(output_json=str(out / 'scenes' / row['scene_id'] / c / 'scene.json')) for c in conditions}
        scenes.append(scene)
    save(out / 'evaluation-manifest.json', dict(conditions=conditions, scenes=scenes,
        minimum_mapped_fraction=original['minimum_mapped_fraction']))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=['local', 'scene', 'all'], default='all')
    p.add_argument('--worker', action='store_true')
    p.add_argument('--attempt', default='01')
    args = p.parse_args()
    if args.worker:
        return worker(args.output, args.stage)
    prepare(args.output)
    from category_priors.object_verification.study import StudyLedger
    from category_priors.object_verification.supervisor import run_supervised_task
    ledger = StudyLedger(ROOT / 'study-ledger.jsonl', VERSION)
    for stage in (['local', 'scene'] if args.stage == 'all' else [args.stage]):
        resultfile = args.output / (stage + '-result.json')
        if resultfile.exists() and read(resultfile).get('execution_complete'):
            continue
        state = ledger.status()
        remaining = state['remaining_gpu_seconds']
        if stage == 'local':
            spent = sum(a.get('occupied_gpu_seconds', 0.) for a in state['attempts']
                        if a['task_id'].startswith('effective-repair-local-'))
            cap = min(int(3600 - spent), int(remaining) - 3600)
        else:
            local = read(args.output / 'local-result.json')
            stats = local['stats']
            observation = stats['observation_seconds'] / max(1, stats['observation_count'])
            semantic = stats['semantic_seconds'] / max(1, stats['semantic_count'])
            # Upper bounds: ten observations and sixty semantic inputs per C0, plus scene scans/IO.
            estimate = 1.5 * (374 * 10 * observation + 374 * 60 * semantic + 1800)
            save(args.output / 'cost-estimate.json', dict(remaining_seconds=remaining, estimated_seconds=estimate,
                observation_seconds=observation, semantic_seconds=semantic, recovery_reserve_seconds=3600,
                maximum_new_observations=3740, maximum_semantic_inputs=22440, margin=1.5))
            if estimate > remaining - 3600:
                save(args.output / 'budget-shortfall.json', dict(estimated_seconds=estimate, available_seconds=remaining - 3600,
                    completed_group='local', missing_group='all 374 C0 nine conditions'))
                return 2
            cap = int(remaining) - 3600
        if cap <= 0:
            raise RuntimeError('Original remaining budget cannot retain the required recovery reserve')
        argv = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=5', str(cap - 5), sys.executable,
                str(Path(__file__).resolve()), '--output', str(args.output), '--stage', stage, '--worker']
        env = dict(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        spec = dict(experiment_version=VERSION, kind='gpu', argv=argv, cwd=str(BASE), env=env,
                    comparison='effective-repair-20260911', stage=stage, cap_seconds=cap)
        result = run_supervised_task(ledger, f'effective-repair-{stage}-{args.attempt}', spec=spec, argv=argv, cwd=BASE,
            attempt_dir=args.output / f'job-{stage}-{args.attempt}', env=env, poll_seconds=1,
            gate_validator=lambda _: {'authorization': '2026-09-11 user approved complete effective-repair experiment'})
        save(args.output / f'process-{stage}-{args.attempt}.json', result)
        if not result['process_complete']:
            return 1
    if (args.output / 'scene-result.json').exists():
        evaluation_manifest(args.output)
        argv = [sys.executable, str(Path(__file__).parent / 'run_effective_repair_evaluation.py'),
                '--manifest', str(args.output / 'evaluation-manifest.json'), '--output-dir', str(args.output)]
        result = subprocess.run(argv, cwd=BASE)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
