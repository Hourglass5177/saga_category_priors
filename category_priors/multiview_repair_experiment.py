"""Bounded, paired multiview repair. Inference never reads evaluation annotations."""
from __future__ import annotations

import argparse
from collections import defaultdict, OrderedDict
from dataclasses import asdict
import hashlib
import itertools
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from run_effective_repair import BASE, ROOT, VERSION, Runtime, read, save, slug
from .category_scale_experiment import ScaleRuntime, comparison_image
from .category_scale import resolve_prior, scale_slots, projection_iou
from .effective_repair_core import g0_members, g2_members, merge_scene
from .multiview_repair import (metric_crop, interior_points, image_edge, boundary_reasons,
    known_domain, depth_visibility, mask_families, independent_support, evidence_score,
    select_evidence, select_view_pair, member_components)
from .object_verification.model_adapter import CropTransform
from .object_verification.observation import cameras_independent
from .prediction_contract import normalize_prediction
from .common_geometry_selector import (VERSION as GEOMETRY_SELECTOR, reference_family,
    geometry_quality, pixel_evidence, select_geometry, semantic_state, stable_improvement)
from .regional_evidence import VERSION as REGIONAL_SELECTOR

EXPERIMENT = 'multiview-repair-20260918'
TAIL_SOURCES = {'scene0025_01': [23, 28, 109, 122], 'scene0645_00': [41]}


def ids_file(path):
    with np.load(path, allow_pickle=False) as z:
        return z['members'].copy()


class MultiviewRuntime(ScaleRuntime):
    @property
    def common_geometry(self):
        return getattr(self, 'plan', {}).get('selector') in (GEOMETRY_SELECTOR, REGIONAL_SELECTOR)

    @property
    def regional_evidence(self):
        return getattr(self, 'plan', {}).get('selector') == REGIONAL_SELECTOR

    def __init__(self, out):
        super().__init__(out)
        self.shared = Path(self.plan['shared_output'])
        self.saved_global = Path(self.plan['saved_global'])
        self.obs_cache = OrderedDict()
        self.descriptor_cache = {}
        self.occluded_cache = {}

    def scene(self, sid):
        super().scene(sid)
        self.obs_cache.clear()
        self.descriptor_cache.clear()
        self.occluded_cache.clear()
        self.__dict__.pop('fixed_reference_cache', None)

    def observation(self, path):
        key = str(path)
        if key not in self.obs_cache:
            self.obs_cache[key] = super().observation(path)
        self.obs_cache.move_to_end(key)
        while len(self.obs_cache) > 128:
            self.obs_cache.popitem(last=False)
        return self.obs_cache[key]

    def descriptor(self, path):
        if str(path) in self.descriptor_cache:
            return self.descriptor_cache[str(path)]
        row, arrays, _ = self.observation(path)
        if arrays is None:
            return None
        result = dict(path=str(path), camera=row['camera_uid'],
                    positive_ids=np.union1d(arrays['hard_ids'], arrays['alpha_ids']),
                    negative_ids=arrays['negative_ids'], known_ids=arrays['known_ids'])
        self.descriptor_cache[str(path)] = result
        return result

    def adopt_raw(self, camera_uid, masks, crop, dest, *, actual_input, qualities, source):
        """Reuse raw masks, not historic full-image negatives or old alpha denominators."""
        result = dest / 'group.json'
        if result.exists():
            return read(result)['paths']
        d = self.data(camera_uid)
        encoded, crop_valid = crop.extract(d['rgb'])
        observed = crop.mask_to_image(crop_valid)
        masks = np.asarray(masks, bool) & observed[None]
        if masks.shape != (3, *observed.shape):
            raise ValueError('all three raw SAM outputs must use original image coordinates')
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest / 'raw.npz', alternatives=masks, observed_pixels=observed,
                            encoded_rgb=encoded, qualities=qualities)
        paths = []
        for index, mask in enumerate(masks):
            path = dest / f'mask-{index}'
            paths.append(str(path))
            if (path / 'observation.json').exists():
                continue
            # Alpha integrates multiple contributors. The single-dominant-contributor
            # rule is for hard IDs/prompts, not a pixel veto on the existing G2 renderer.
            known = known_domain(mask, masks, observed)
            mass = self.renderer.alpha(self.cameras[camera_uid], mask[None], known)
            hard = np.unique(d['ids'][d['reliable'] & mask])
            negative = np.unique(d['ids'][d['reliable'] & known & ~mask])
            # A hidden Gaussian's nonzero splat tail is not observed background or
            # foreground. Ignore that view's vote; never delete scene members here.
            unknown_ids = self.occlusion_unknown(camera_uid)
            inside, visible = mass.inside_mass[0].copy(), mass.visible_mass.copy()
            inside[unknown_ids] = 0.; visible[unknown_ids] = 0.
            ratio = np.divide(inside, visible, out=np.zeros_like(inside), where=visible > 0)
            alpha = np.flatnonzero((inside >= .5) & (ratio >= .5))
            known_ids = np.union1d(np.unique(d['ids'][d['reliable'] & known]), alpha)
            path.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path / 'prediction.npz', sam=mask, observed_pixels=known,
                actual_observed_pixels=observed, hard_ids=hard, negative_ids=negative,
                alpha_ids=alpha, known_ids=known_ids)
            np.savez_compressed(path / 'alpha.npz', inside=inside, visible=visible)
            save(path / 'observation.json', dict(camera_uid=camera_uid, status='complete',
                actual_input=actual_input, chosen=index, qualities=list(map(float, qualities)),
                raw_group=str(dest), source_state=source, semantics={},
                suspected_occluded_unknown_count=len(unknown_ids),
                boundary=boundary_reasons(mask & d['reliable'], observed)))
        save(result, dict(paths=paths, camera_uid=camera_uid, actual_input=actual_input, source=source))
        return paths

    def occlusion_unknown(self, view):
        if view not in self.occluded_cache:
            camera, data = self.cameras[view], self.data(view)
            ids = np.arange(len(self.assets.xyz_scene))
            xy, depth = camera.project(self.assets.xyz_scene)
            front = np.zeros(data['ids'].shape, np.float32)
            valid = data['ids'] >= 0
            front[valid] = camera.optical_z_m(self.assets.xyz_scene[data['ids'][valid]])
            state = depth_visibility(xy, depth, data['ids'], front, data['reliable'], ids, [])
            hidden = state['suspected_occluded']
            # A center-depth approximation cannot overrule directly visible support.
            hidden[np.unique(data['ids'][data['reliable']])] = False
            self.occluded_cache[view] = np.flatnonzero(hidden)
        return self.occluded_cache[view]

    def old_group(self, old_path, dest):
        if (dest / 'group.json').exists():
            return read(dest / 'group.json')['paths']
        row, arrays, _ = self.observation(old_path)
        if arrays is None:
            save(dest / 'group.json', dict(paths=[], status=row['status']))
            return []
        crop = CropTransform(**row['actual_input']['crop'])
        if 'alternatives' not in arrays:
            raise ValueError('raw SAM alternatives missing: ' + str(old_path))
        return self.adopt_raw(row['camera_uid'], arrays['alternatives'], crop, dest,
            actual_input=row['actual_input'], qualities=row['qualities'], source=str(old_path))

    def observe_group(self, camera_uid, locator, prior, dest, *, prompt_count=1, source_state):
        if (dest / 'group.json').exists():
            return read(dest / 'group.json')['paths']
        began = time.monotonic()
        d = self.data(camera_uid)
        foreground = self.project(camera_uid, locator)
        reliable = foreground & d['reliable']
        yy, xx = np.where(reliable)
        if not len(xx):
            save(dest / 'group.json', dict(paths=[], status='no_reliable_point', camera_uid=camera_uid,
                                           source_state=source_state))
            return []
        camera = self.cameras[camera_uid]
        visible_ids = np.unique(d['ids'][reliable])
        z = camera.optical_z_m(self.assets.xyz_scene[visible_ids])
        z = z[np.isfinite(z) & (z > 0)]
        if not len(z):
            save(dest / 'group.json', dict(paths=[], status='no_positive_depth'))
            return []
        crop, location = metric_crop(foreground.shape, np.c_[xx, yy], math.sqrt(camera.fx*camera.fy),
                                     prior['diagonal_m'], float(np.median(z)))
        encoded, valid = crop.extract(d['rgb'])
        image_valid = crop.mask_to_image(valid)
        points = interior_points(foreground & image_valid, d['reliable'], prompt_count)
        if not points:
            save(dest / 'group.json', dict(paths=[], status='no_reliable_point_after_crop'))
            return []
        actual = dict(crop=asdict(crop), point_image=points[0], positive_points_image=points,
                      box_crop=None, location=location, depth_m=float(np.median(z)), prior=prior,
                      source_state=source_state)
        # Repeated global slots/unchanged feedback may share exactly the same model input.
        signature = hashlib.blake2b(repr((camera_uid, crop.left, crop.top, crop.width, crop.height, points)).encode(), digest_size=16).hexdigest()
        canonical = self.shared / 'model-inputs' / self.sid / signature
        if (canonical / 'group.json').exists():
            paths = read(canonical / 'group.json')['paths']
            save(dest / 'group.json', dict(paths=paths, status='complete', actual_input=actual,
                source_state=source_state, reused_input=str(canonical)))
            self.stats['reused_observations'] += 1
            return paths
        previous_raw = next((p for old in ('shared-v2','shared')
            if (p := self.shared.parent / old / 'model-inputs' / self.sid / signature / 'raw.npz').exists()
            and p.parent != canonical), None)
        if previous_raw is not None:
            with np.load(previous_raw, allow_pickle=False) as z:
                paths = self.adopt_raw(camera_uid, z['alternatives'], crop, canonical,
                    actual_input=actual, qualities=z['qualities'], source=source_state)
            save(dest / 'group.json', dict(paths=paths, status='complete', actual_input=actual,
                source_state=source_state, reused_raw=str(previous_raw)))
            self.stats['reused_observations'] += 1
            return paths
        alternatives = self.sam._sam_masks(image=encoded, crop=crop, box_crop=None,
            point_crop=crop.image_to_crop_points(points), uid=slug(camera_uid))
        paths = self.adopt_raw(camera_uid, [m.mask_image for m in alternatives], crop, canonical,
            actual_input=actual, qualities=[m.sam_quality for m in alternatives], source=source_state)
        save(dest / 'group.json', dict(paths=paths, status='complete', actual_input=actual, source_state=source_state))
        self.stats['observation_count'] += 1
        self.stats['observation_seconds'] += time.monotonic()-began
        return paths

    def pairs(self, views, members):
        geometry = {v: self.camera_geometry(v, members) if len(members) else None for v in views}
        return [(a, b) for a, b in itertools.combinations(views, 2) if geometry[a] is not None
                and geometry[b] is not None and cameras_independent(geometry[a], geometry[b])]

    def family_members(self, paths, locator):
        descriptors = [self.descriptor(p) for p in paths]
        descriptors = [d for d in descriptors if d is not None]
        pairs = self.pairs([r['camera'] for r in descriptors], locator)
        rows, inside, visible = [], [], []
        for d in descriptors:
            row, arrays, mass = self.observation(d['path'])
            rows.append(dict(camera_uid=d['camera'], **{k: arrays[k] for k in ('hard_ids', 'alpha_ids', 'negative_ids')}))
            inside.append(mass['inside']); visible.append(mass['visible'])
        first = g0_members(rows, pairs)
        # A single-view mask stays in the candidate bank, not an independently verified core.
        second = g2_members(np.stack(inside), np.stack(visible))['members'] if inside else np.empty(0, np.int64)
        return {'G0': first, 'G2': second}

    def geometry_rows(self, groups, locator, prefix, kind):
        observations = [self.descriptor(p) for group in groups for p in group]
        observations = [x for x in observations if x is not None]
        output = []
        for index, paths in enumerate(mask_families(observations)):
            for rule, members in self.family_members(paths, locator).items():
                output.append(dict(id=f'{prefix}-family{index}-{rule}', kind=kind,
                    members=members, paths=paths, branch=prefix))
        return output

    def score_evidence(self, members, pool, views):
        """A common pool and final membership determine score; ancestry cannot affect it."""
        if self.common_geometry:
            return self.score_common_geometry(members, pool, views)
        members = np.asarray(members, np.int64)
        by_camera = defaultdict(list)
        for path in sorted(set(pool)):
            d = self.descriptor(path)
            if d is not None:
                by_camera[d['camera']].append(d)
        selected = []
        membership = np.zeros(len(self.assets.xyz_scene), bool); membership[members] = True
        for view in views:
            options = []
            for d in by_camera[view]:
                positive, known = d['positive_ids'], d['known_ids']
                intersection = int(membership[positive].sum())
                union = int(membership[known].sum()) + len(positive)-intersection
                options.append((-intersection/max(1, union), d['path'], d))
            if options:
                selected.append(min(options, key=lambda x: x[:2])[2])
        pairs = self.pairs(views, members)
        pair_set = {frozenset(p) for p in pairs}
        overlaps, coverage, per_view = [], [], []
        for d in selected:
            others = [o for o in selected if frozenset((d['camera'], o['camera'])) in pair_set]
            if not others:
                per_view.append(dict(camera_uid=d['camera'], iou=None, reason='no_independent_view'))
                continue
            positive_other = np.unique(np.concatenate([o['positive_ids'] for o in others]))
            known_other = np.unique(np.concatenate([o['known_ids'] for o in others]))
            validated = np.intersect1d(members, known_other)
            data = self.data(d['camera'])
            _, arrays, _ = self.observation(d['path'])
            known = arrays['observed_pixels'] & data['reliable'] & np.isin(data['ids'], known_other)
            # Include members reliably rejected by the other view. Filtering down to
            # positive_other before validation erased false positives from the score.
            projected = self.project(d['camera'], validated)
            if not ((arrays['sam'] | projected) & known).any():
                per_view.append(dict(camera_uid=d['camera'], iou=None, reason='no_common_observed_surface'))
                continue
            value = projection_iou(projected, arrays['sam'], known)
            overlaps.append(value); coverage.append(int(known.sum()))
            per_view.append(dict(camera_uid=d['camera'], iou=value, known_pixels=int(known.sum()),
                                 foreground_union_pixels=int(((projected | arrays['sam']) & known).sum()),
                                 independently_positive_members=len(np.intersect1d(members, positive_other)),
                                 reference_mask=d['path']))
        semantics = self.classify_members(members, views)
        quality = evidence_score(overlaps, semantics['score'])
        quality.update(per_view=per_view, known_pixels=coverage)
        core = np.intersect1d(members, independent_support(selected, pairs))
        negative = independent_support(selected, pairs, 'negative_ids')
        return semantics, quality, core, negative, selected

    def fixed_reference(self, pool, views):
        if self.regional_evidence:
            from .regional_evidence_runtime import build_reference
            refs,pairs,trace,_,_=build_reference(self,pool,views)
            return refs,pairs,trace
        cache = self.__dict__.setdefault('fixed_reference_cache', {})
        key = (getattr(self, 'sid', ''), tuple(sorted(set(pool))), tuple(sorted(set(views))))
        if key in cache:
            return cache[key]
        observations, anchor_ids = [], []
        for path in sorted(set(pool)):
            row, arrays, _ = self.observation(path)
            if arrays is None or row['camera_uid'] not in views:
                continue
            data = self.data(row['camera_uid'])
            actual = row.get('actual_input', {})
            points = actual.get('positive_points_image') or [actual.get('point_image')]
            anchors = []
            for point in points:
                if point is None:
                    continue
                x, y = np.floor(np.asarray(point)+.5).astype(int)
                if (0 <= y < data['ids'].shape[0] and 0 <= x < data['ids'].shape[1]
                        and data['reliable'][y,x] and arrays['sam'][y,x]):
                    anchors.append(int(data['ids'][y,x]))
            if not anchors:
                continue
            anchor_ids.extend(anchors)
            observations.append(self.descriptor(path))
        pairs = self.pairs(sorted(set(views)), np.unique(anchor_ids)) if anchor_ids else []
        refs, trace = reference_family(observations, pairs)
        independent = independent_support(refs, pairs)
        # Ambiguity is a property of the entire neutral observation pool, not
        # only the three masks from the crop that happened to win association.
        # A small crop must not declare the whole phone's other visible surface
        # background when another identity-eligible neutral crop observes it.
        possible_foreground = {}
        for observation in observations:
            camera = observation['camera']
            possible_foreground[camera] = np.union1d(possible_foreground.get(camera, []),
                                                    observation['positive_ids'])
        for ref in refs:
            observation, arrays, _ = self.observation(ref['path'])
            data = self.data(ref['camera'])
            known = arrays['observed_pixels'] & data['reliable']
            # A larger SAM alternative is not itself proof that its disputed
            # extension belongs to the object. Require independent continuation.
            raw_path = Path(observation['raw_group']) / 'raw.npz'
            with np.load(raw_path, allow_pickle=False) as raw:
                alternatives = raw['alternatives']
                ambiguous = np.any(alternatives, axis=0) & ~np.all(alternatives, axis=0)
            known &= ~ambiguous | np.isin(data['ids'], independent)
            foreground, background = known & arrays['sam'], known & ~arrays['sam']
            background &= ~np.isin(data['ids'], possible_foreground[ref['camera']])
            known = foreground | background
            ref.update(foreground=foreground, background=background,
                positive_ids=np.unique(data['ids'][foreground]),
                negative_ids=np.unique(data['ids'][background]),
                known_ids=np.unique(data['ids'][known]))
        trace.update(views=list(views), pool=list(sorted(set(pool))),
                     missing_views=sorted(set(views)-{r['camera'] for r in refs}))
        cache[key] = refs, pairs, trace
        return cache[key]

    def score_common_geometry(self, members, pool, views):
        if self.regional_evidence:
            from .regional_evidence_runtime import score
            return score(self,members,pool,views)
        refs, pairs, _ = self.fixed_reference(pool, views)
        per_view = []
        for ref in refs:
            # Observed foreground is fixed before any candidate is considered.
            # A missing member must remain in the FN denominator.
            foreground, background = ref['foreground'], ref['background']
            item = pixel_evidence(self.project(ref['camera'], members), foreground, background)
            per_view.append(dict(camera_uid=ref['camera'], reference_mask=ref['path'], **item))
        quality = geometry_quality(per_view)
        semantics = self.classify_members(members, [r['camera'] for r in refs])
        semantics = dict(semantics)
        semantics['class'], semantics['classification_status'] = semantic_state(semantics, self.assets.saga20)
        core = np.intersect1d(members, independent_support(refs, pairs))
        negative = independent_support(refs, pairs, 'negative_ids')
        return semantics, quality, core, negative, refs

    def bank(self, uid, rows, pool, views, dest, *, metadata=None, reference_pool=None):
        if (dest / 'bank.json').exists():
            return read(dest / 'bank.json')
        dest.mkdir(parents=True, exist_ok=True)
        if self.common_geometry:
            if reference_pool is None:
                # Only bootstrap/common banks may implicitly use their own pool.
                if any(r['kind'] == 'new' for r in rows):
                    raise ValueError('common geometry requires an explicit neutral reference pool')
                reference_pool = pool
            refs, _, reference_trace = self.fixed_reference(reference_pool, views)
            saved_trace=dict(reference_trace,selector=self.plan['selector'])
            if self.regional_evidence:
                saved_trace['representative_paths']=[r['path'] for r in refs]
            else:
                saved_trace['paths']=[r['path'] for r in refs]
            save(dest / 'reference.json',saved_trace)
        scoring_pool = reference_pool if self.common_geometry else pool
        unique, aliases = {}, {}
        for row in rows:
            members = np.unique(row['members']).astype(np.int64)
            key = members.tobytes()
            if key in unique and row['id'] not in ('original', 'legacy-G0', 'legacy-G2'):
                aliases[row['id']] = unique[key]['id']; continue
            row = dict(row, members=members)
            if key not in unique or row['id'] in ('original', 'legacy-G0', 'legacy-G2'):
                unique[(key, row['id']) if key in unique else key] = row
        candidates, member_arrays, score_cache = [], {}, {}
        for row in unique.values():
            members = row['members']; key = members.tobytes()
            if key not in score_cache:
                score_cache[key] = self.score_evidence(members, scoring_pool, views)
            semantics, quality, core, negative, associated = score_cache[key]
            target = dest / 'candidates' / row['id']
            target.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target / 'members.npz', members=members, core=core, negative=negative,
                                ratio=np.zeros(len(members), np.float32))
            candidate = dict(id=row['id'], kind=row['kind'], member_count=len(members),
                members_file=str(target / 'members.npz'), paths=row.get('paths', []),
                branch=row.get('branch'), semantics=semantics, quality=quality,
                core_ids=core, negative_ids=negative, associated_paths=[a['path'] for a in associated])
            candidates.append(candidate); member_arrays[row['id']] = members
        if self.common_geometry:
            winner, selection_trace = select_geometry(candidates, member_arrays,
                baseline_id=(metadata or {}).get('baseline_id'))
            save(dest / 'selection.json', selection_trace)
        else:
            winner = select_evidence(candidates, member_arrays)
        for candidate in candidates:
            candidate['core_count'] = len(candidate.pop('core_ids'))
            candidate['negative_count'] = len(candidate.pop('negative_ids'))
        result = dict(uid=uid, views=views, pool=sorted(set(pool)), candidates=candidates,
                      selected_id=winner['id'], aliases=aliases, **(metadata or {}))
        if self.common_geometry:
            result.update(selector=self.plan['selector'], reference_pool=sorted(set(reference_pool)),
                          ranked_ids=[r['candidate_id'] for r in selection_trace])
            if self.regional_evidence:
                from .regional_evidence_runtime import regional_candidate
                patch=regional_candidate(self,result,dest)
                result['assembly_candidates']=[patch] if patch is not None else []
        save(dest / 'bank.json', result)
        print('bank', uid, dest.name, 'candidates', len(candidates), 'selected', winner['id'],
              'score', round(winner['quality']['score'], 4), flush=True)
        return result

    def selected(self, bank):
        row = next(r for r in bank['candidates'] if r['id'] == bank['selected_id'])
        return row, ids_file(row['members_file'])

    def rows_from_bank(self, bank, *, prefix='', kind=None):
        return [dict(id=prefix+r['id'], kind=kind or r['kind'], members=ids_file(r['members_file']),
                     paths=r['paths'], branch=r.get('branch')) for r in bank['candidates']]

    def old_candidates(self, uid, source, views, dest, case=None):
        members = np.asarray(source.support_ids if hasattr(source, 'support_ids') else source.members, np.int64)
        anchors = source.anchor_ids if hasattr(source, 'anchor_ids') else members
        old = self.old_paths(uid, members, anchors, views, dest, case=case)
        groups = [self.old_group(p, dest / f'corrected-old-{i}') for i, p in enumerate(old)]
        # Preserve the saved old choices exactly; additionally build corrected alternatives.
        _, historic = Runtime.geometry(self, old, members, dest / 'historic-geometry')
        rows = [dict(id='original', kind='original', members=members, paths=old)]
        rows += [dict(id='legacy-'+r, kind='legacy', members=historic[r], paths=old) for r in ('G0', 'G2')]
        rows += self.geometry_rows(groups, members, 'corrected-old', 'common')
        return members, groups, rows

    def common_initial(self, uid, source, first_views, dest, *, case=None):
        result = dest / 'common.json'
        if result.exists():
            return read(result)
        members, old_groups, rows = self.old_candidates(uid, source, first_views, dest, case)
        pool = [p for g in old_groups for p in g]
        bootstrap = self.bank(uid, rows, pool, first_views, dest / 'bootstrap')
        _, locator = self.selected(bootstrap)
        groups_by_q = {}
        for q in ('q25', 'q50', 'q75'):
            prior = resolve_prior(self.scale_priors, 'global', 'global')
            prior.update(quantile=q, diagonal_m=prior['sizes_m'][q])
            groups = [self.observe_group(v, locator, prior, dest / q / slug(v),
                source_state='common_initial') for v in first_views]
            groups_by_q[q] = groups
            rows += self.geometry_rows(groups, locator, 'global-'+q, 'common')
            pool += [p for g in groups for p in g]
        common = self.bank(uid, rows, pool, first_views, dest / 'common-bank')
        row, locator = self.selected(common)
        result_data = dict(bank=common, groups_by_q=groups_by_q, original_members=members.tolist(),
                           first_views=first_views, locator_file=row['members_file'])
        save(result, result_data)
        return result_data

    def visibility(self, view, members):
        camera, data = self.cameras[view], self.data(view)
        xy, depth = camera.project(self.assets.xyz_scene[members])
        front = np.zeros(data['ids'].shape, np.float32)
        valid = data['ids'] >= 0
        front[valid] = camera.optical_z_m(self.assets.xyz_scene[data['ids'][valid]])
        return depth_visibility(xy, depth, data['ids'], front, data['reliable'], members, members)

    def choose_views(self, uid, common, original_panel, dest):
        if dest.exists():
            return read(dest)
        first = list(original_panel['construction'][:2])
        old_extra = list(original_panel['construction'][2:4])
        _, locator = self.selected(common['bank'])
        frontier, reasons = set(), {}
        for view in first:
            d = self.data(view)
            edge_ids = np.unique(d['ids'][self.project(view, locator) & d['reliable'] & image_edge(d['ids'].shape)])
            visibility = self.visibility(view, locator)
            hidden = locator[visibility['suspected_occluded']]
            frontier.update(map(int, edge_ids)); frontier.update(map(int, hidden))
            reasons[view] = dict(image_edge_ids=edge_ids.tolist(), suspected_occluded_ids=hidden.tolist(),
                suspected_self_occluded_ids=locator[visibility['suspected_self_occluded']].tolist())
        excluded = {original_panel.get('heldout')} | set(first)
        for case in self.plan['local_cases']:
            if case['candidate_uid'] == uid:
                excluded.update(v['camera_uid'] for v in case['views'][2:])
        candidates, geoms = [], {}
        first_geoms = [self.camera_geometry(v, locator) for v in first]
        if frontier and len(locator):
            for camera in sorted(self.assets.cameras, key=lambda c: c.uid):
                if camera.uid in excluded:
                    continue
                geom = self.camera_geometry(camera.uid, locator)
                eligible = geom is not None and all(g is not None and cameras_independent(geom, g) for g in first_geoms)
                if not eligible and camera.uid not in old_extra:
                    continue
                data = self.data(camera.uid)
                reliable = self.project(camera.uid, locator) & data['reliable']
                if not reliable.any():
                    continue
                seen = np.unique(data['ids'][reliable & ~image_edge(reliable.shape)])
                angle = min((float(np.degrees(np.arccos(np.clip(np.dot(geom.view_ray, g.view_ray), -1, 1))))
                             for g in first_geoms if g is not None and geom is not None), default=0.)
                candidates.append(dict(camera=camera.uid, revealed=sorted(set(map(int, seen)) & frontier),
                                       visible_count=len(np.unique(data['ids'][reliable])), angle=angle, eligible=eligible))
                geoms[camera.uid] = geom
        extra, changed = select_view_pair(old_extra, candidates,
            lambda a, b: cameras_independent(geoms[a], geoms[b]))
        # Never add a feedback stage to a source with fewer than two initial views.
        if len(first) < 2:
            extra, changed = old_extra, False
        continuation = ('insufficient_initial_views' if len(first) < 2 else
                        'no_later_slots' if not old_extra else
                        'no_edge_or_occlusion_cue' if not frontier else
                        'replaced' if changed else
                        'no_reliable_independent_continuation' if not any(c['eligible'] for c in candidates) else
                        'no_better_combination')
        result = dict(construction=first+extra, heldout=original_panel.get('heldout'),
            original_construction=original_panel['construction'], changed=changed,
            continuation_status=continuation,
            frontier_reasons=reasons, ranked_geometry=candidates,
            selection_source='common global first-two observations only')
        save(dest, result)
        return result

    def mode_initial(self, uid, common, dest):
        if (dest / 'bank.json').exists():
            return read(dest / 'bank.json')
        row, locator = self.selected(common['bank'])
        slots = scale_slots(self.scale_priors, self.plan['classes32'], row['semantics']['mean_cosines'], self.mode)
        rows = self.rows_from_bank(common['bank'])
        pool, paths_by_branch = list(common['bank']['pool']), {}
        for prior in slots:
            groups = [self.observe_group(v, locator, prior, dest / 'observations' / prior['branch'] / slug(v),
                      source_state='initial_'+self.mode) for v in common['first_views']]
            paths_by_branch[prior['branch']] = groups
            rows += self.geometry_rows(groups, locator, prior['branch'], 'new')
            pool += [p for g in groups for p in g]
        return self.bank(uid, rows, pool, common['first_views'], dest,
            metadata=dict(slots=slots, paths_by_branch=paths_by_branch, initial_locator=common['locator_file']),
            reference_pool=common['bank']['pool'])

    def final_bank(self, uid, common, initial, panel, actual_initial, protocol, dest, *, open_bank=None):
        if (dest / 'bank.json').exists():
            return read(dest / 'bank.json')
        _, frozen_locator = self.selected(common['bank'])
        # A retained B0/whole replacement can have a different core from the raw
        # selected proposal. Re-evaluate the actual writeback, using initial views only.
        _, _, reliable_core, _, _ = self.score_evidence(actual_initial,
            initial.get('reference_pool', initial['pool']), common['first_views'])
        applicable = protocol == 'feedback' and len(actual_initial) >= 3 and len(reliable_core) > 0
        locator = actual_initial if applicable else frozen_locator
        views = panel['construction']
        semantic = self.classify_members(locator, views[:2])
        slots = scale_slots(self.scale_priors, self.plan['classes32'], semantic['mean_cosines'], self.mode)
        rows = self.rows_from_bank(initial)
        pool = list(initial['pool'])
        reference_pool = list(common['bank']['pool'])
        if protocol == 'feedback':
            if open_bank is None:
                raise ValueError('feedback requires the saved matched-open bank for fallback')
            rows += self.rows_from_bank(open_bank, prefix='pre-feedback-', kind='pre_feedback')
            pool += open_bank['pool']
        # Historical global members are identical in all four conditions. Their later
        # observations never enter initial selection, view selection, or feedback inputs.
        historic_path = self.saved_global / 'scenes' / self.sid / 'banks' / slug(uid) / 'bank.json'
        if historic_path.exists():
            saved = read(historic_path)
            rows += self.rows_from_bank(saved, prefix='saved-global-', kind='global_saved')
        # Common new-view global observations are also shared, with frozen localization.
        for q, first_groups in common['groups_by_q'].items():
            prior = resolve_prior(self.scale_priors, 'global', 'global')
            prior.update(diagonal_m=prior['sizes_m'][q], quantile=q)
            groups = list(first_groups) + [self.observe_group(v, frozen_locator, prior,
                self.shared / 'common-final' / self.sid / slug(uid) / q / slug(v), prompt_count=3,
                source_state='common_open') for v in views[2:]]
            rows += self.geometry_rows(groups, frozen_locator, 'common-final-'+q, 'common')
            pool += [p for g in groups for p in g]
            reference_pool += [p for g in groups for p in g]
        for prior in slots:
            # First-view evidence is immutable even when feedback changes the category.
            first_groups = initial['paths_by_branch'].get(prior['branch'], [])
            groups = list(first_groups) + [self.observe_group(v, locator, prior,
                dest / 'observations' / prior['branch'] / slug(v), prompt_count=3,
                source_state=('actual_initial_writeback' if applicable else 'frozen_common_initial')) for v in views[2:]]
            rows += self.geometry_rows(groups, locator, 'final-'+prior['branch'], 'new')
            pool += [p for g in groups for p in g]
        # The exact committed open result, not its pre-allocation winner, is the
        # feedback fallback. Caller supplies it after open assembly.
        baseline_id = None
        if self.common_geometry and protocol == 'feedback':
            baseline_id = 'committed-open'
            rows.append(dict(id=baseline_id, kind='pre_feedback',
                members=np.asarray(open_bank['actual_members'], np.int64), paths=[]))
        bank = self.bank(uid, rows, pool, views, dest,
            **({'reference_pool': reference_pool} if self.common_geometry else {}),
            metadata=dict(slots=slots, protocol=protocol, feedback_applicable=applicable,
                          baseline_id=baseline_id,
                          actual_initial_count=len(actual_initial), reliable_written_core_count=len(reliable_core),
                          actual_initial_members=actual_initial.tolist(), view_changed=panel['changed']))
        self.addition_trace(bank, common, dest)
        return bank

    def addition_trace(self, bank, common, dest):
        _, selected = self.selected(bank)
        added = np.setdiff1d(selected, common['original_members'])
        initial = next(r for r in common['bank']['candidates'] if r['id'] == 'original')
        traces, summaries = {}, []
        for i, view in enumerate(common['first_views']):
            state = self.visibility(view, added)
            data = self.data(view)
            directly_visible = np.isin(added, np.unique(data['ids'][data['reliable']]))
            old = next((p for p in initial['paths'] if read(Path(p)/'observation.json')['camera_uid'] == view), None)
            old_positive = []
            if old is not None:
                _, arrays, _ = self.observation(old)
                if arrays is not None:
                    old_positive = arrays['hard_ids']
            flags = dict(photo_outside=state['outside_image'] & ~directly_visible,
                         suspected_occlusion=state['suspected_occluded'] & ~directly_visible,
                         visible_but_omitted=directly_visible & ~np.isin(added, old_positive))
            flags['unknown'] = ~np.any(np.stack(list(flags.values())), axis=0)
            traces.update({f'view{i}_{k}':v for k,v in flags.items()})
            summaries.append(dict(camera_uid=view, **{k:int(v.sum()) for k,v in flags.items()}))
        np.savez_compressed(dest/'added-region-reasons.npz', members=added, **traces)
        save(dest/'added-region-reasons.json', dict(added_member_count=len(added), views=summaries,
             reasons_are_observation_cues_not_gt=True))

    def assemble_evidence(self, banks, dest, *, baseline=None, excluded_views=None):
        if self.regional_evidence:
            from .regional_evidence_assembly import assemble
            assemble(self,banks,dest.parent/(dest.name+'-pure'),baseline=baseline,
                     excluded_views=excluded_views,include_regional=False)
            return assemble(self,banks,dest,baseline=baseline,excluded_views=excluded_views)
        if self.common_geometry:
            from .common_geometry_assembly import assemble_common
            return assemble_common(self, banks, dest, baseline=baseline, excluded_views=excluded_views)
        if (dest / 'summary.json').exists():
            return read(dest / 'scene.json')
        baseline = baseline or self.b0
        base_labels = np.asarray(baseline['point_labels'], np.int64)
        uid_by_label = {int(k): m.get('object_uid', f'B0:{k}') for k, m in baseline['instances'].items()}
        n = len(base_labels)
        baseline_members = {uid_by_label[int(label)]: np.flatnonzero(base_labels == int(label))
                            for label in baseline['instances']}
        proposed = {uid: self.selected(bank)[1] for uid, bank in banks.items()}
        original_members = {}
        if baseline is not self.b0:
            original_labels = np.asarray(self.b0['point_labels'])
            original_members = {f'B0:{label}': np.flatnonzero(original_labels == int(label))
                                for label in self.b0['instances']}
        components = member_components(list(baseline_members.items()) + list(proposed.items())
                                       + list(original_members.items()), n)
        heldout = {u: set(v) for u, v in (excluded_views or {}).items()}
        for case in getattr(self, 'plan', {}).get('local_cases', []):
            heldout.setdefault(case['candidate_uid'], set()).update(v['camera_uid'] for v in case['views'][2:])
        contexts = {}
        for component in components:
            pool = sorted({p for uid in component if uid in banks for p in banks[uid]['pool']})
            views = sorted({v for uid in component if uid in banks for v in banks[uid]['views']})
            forbidden = set().union(*(heldout.get(uid, set()) for uid in component))
            if forbidden:
                views = [v for v in views if v not in forbidden]
                pool = [p for p in pool if self.descriptor(p) is not None
                        and self.descriptor(p)['camera'] in views]
            contexts.update({uid: (pool, views) for uid in component})
        def context(uid, members):
            # All contenders and incumbents in a fixed conflict component share
            # observations. The UID or generating crop cannot change their score.
            return contexts[uid]
        def make_row(uid, members, incumbent, meta=None):
            pool, views = context(uid, members)
            semantics, quality, core, negative, refs = self.score_evidence(members, pool, views)
            return dict(uid=uid, members=np.asarray(members, np.int64), semantics=semantics,
                quality=quality, core=core, negative=negative, refs=refs, incumbent=incumbent, meta=meta)
        before = {}
        for label, meta in baseline['instances'].items():
            uid = uid_by_label[int(label)]
            before[uid] = make_row(uid, baseline_members[uid], True, meta)
        alternatives = list(before.values())
        for uid, bank in banks.items():
            alternatives.append(make_row(uid, proposed[uid], False))
        # Original B0 remains an available whole-object candidate after the initial pass.
        if baseline is not self.b0:
            for label, meta in self.b0['instances'].items():
                alternatives.append(make_row(f'B0:{label}', original_members[f'B0:{label}'], False, meta))
        choices = {}
        for row in alternatives:
            uid = row['uid']
            if len(row['members']) < 3:
                continue
            if row['semantics']['class'] not in self.assets.saga20 and not row['incumbent']:
                continue
            previous = choices.get(uid)
            if previous is None or (row['quality']['score'], row['incumbent']) > (previous['quality']['score'], previous['incumbent']):
                choices[uid] = row
        ordered = sorted(choices.values(), key=lambda r: (-r['quality']['score'], not r['uid'].startswith('B0:'), not r['incumbent'], r['uid']))
        kept, suppressed = [], {}
        for row in ordered:
            for other in kept:
                if row['semantics']['class'] != other['semantics']['class']:
                    continue
                intersection = len(np.intersect1d(row['members'], other['members']))
                union = len(row['members'])+len(other['members'])-intersection
                nested = intersection / max(1, min(len(row['members']), len(other['members']))) >= .9
                # Containment needs actual common, independent camera evidence.
                common_views = set(r['camera'] for r in row['refs']) & set(r['camera'] for r in other['refs'])
                small, large = sorted((row, other), key=lambda r: len(r['members']))
                confirmed_views = []
                if nested:
                    for view in sorted(common_views):
                        data = self.data(view)
                        a = self.project(view, small['members']) & data['reliable']
                        b = self.project(view, large['members']) & data['reliable']
                        common_positive = np.intersect1d(
                            next(r for r in small['refs'] if r['camera'] == view)['positive_ids'],
                            next(r for r in large['refs'] if r['camera'] == view)['positive_ids'])
                        supported = a & np.isin(data['ids'], common_positive)
                        if a.any() and (a & b & supported).sum() / a.sum() >= .9:
                            confirmed_views.append(view)
                nested_supported = nested and bool(self.pairs(confirmed_views, np.union1d(row['members'], other['members'])))
                if intersection/max(1, union) > .5 or nested_supported:
                    suppressed[row['uid']] = other['uid']; break
            else:
                kept.append(row)
        # NMS changes instance identity, not the existence of independently
        # observed object surfaces. Carry uncontradicted cores of verified
        # duplicates before ownership allocation; never merge an unrelated core.
        core_transfers = []
        merged_members = {r['uid']: r['members'] for r in kept}
        kept_by_uid = {r['uid']: r for r in kept}
        for uid, target in suppressed.items():
            donor, winner = choices[uid], kept_by_uid[target]
            additions = np.setdiff1d(donor['core'], np.union1d(donor['negative'], winner['negative']))
            additions = np.setdiff1d(additions, merged_members[target])
            for other in kept:
                if other['uid'] != target:
                    additions = np.setdiff1d(additions, np.setdiff1d(other['core'], other['negative']))
            if len(additions):
                merged_members[target] = np.union1d(merged_members[target], additions)
                core_transfers.append(dict(source=uid, target=target, count=len(additions),
                                           members=additions.tolist()))
        for i, row in enumerate(kept):
            if not np.array_equal(merged_members[row['uid']], row['members']):
                kept[i] = make_row(row['uid'], merged_members[row['uid']], row['incumbent'], row['meta'])
        kept.sort(key=lambda r: (-r['quality']['score'], not r['uid'].startswith('B0:'), not r['incumbent'], r['uid']))
        # Conflicting members are compared on the intersection of available cameras,
        # never with a candidate's larger number of private crops/views as extra votes.
        all_views = sorted({r['camera'] for row in kept for r in row['refs']})
        cameras_for_point = np.zeros((len(all_views), n), np.int16)
        contender_count = np.zeros(n, np.int16)
        for row in kept:
            contender_count[row['members']] += 1
            cameras = {r['camera'] for r in row['refs']}
            for vi, view in enumerate(all_views):
                if view in cameras:
                    cameras_for_point[vi, row['members']] += 1
        core_count = np.zeros(n, np.int32)
        for row in kept:
            core_count[np.setdiff1d(row['core'], row['negative'])] += 1
        owner = np.full(n, -1, np.int32)
        best_core = np.full(n, -1, np.int8)
        best_votes = np.full(n, -100, np.int16)
        best_score = np.full(n, -1., np.float64)
        ownership_rows = []
        for i, row in enumerate(kept):
            members = row['members']
            votes = np.zeros(len(members), np.int16)
            # One adopted observation per camera. Unknown contributes no negative vote.
            for ref in row['refs']:
                common = cameras_for_point[all_views.index(ref['camera']), members] == contender_count[members]
                votes += (common & np.isin(members, ref['positive_ids'])).astype(np.int16)
                votes -= (common & np.isin(members, ref['negative_ids'])).astype(np.int16)
            protected = np.isin(members, row['core']) & (core_count[members] == 1) & ~np.isin(members, row['negative'])
            core_rank = protected.astype(np.int8)
            score = row['quality']['score']
            win = (core_rank > best_core[members]) | ((core_rank == best_core[members]) &
                ((votes > best_votes[members]) | ((votes == best_votes[members]) & (score > best_score[members]))))
            owner[members[win]] = i; best_core[members[win]] = core_rank[win]
            best_votes[members[win]] = votes[win]; best_score[members[win]] = score
            ownership_rows.append(dict(uid=row['uid'], raw_members=len(members), protected_core_count=int(protected.sum())))
        assigned = {row['uid']: np.flatnonzero(owner == i) for i, row in enumerate(kept)}
        after = {}
        classification_fallbacks = []
        for uid, ids in assigned.items():
            if len(ids) < 3:
                continue
            previous = before.get(uid)
            unchanged = previous is not None and np.array_equal(ids, previous['members'])
            # Reclassification uses 32 labels; the scene export accepts 20. An
            # unchanged, valid baseline object must not disappear just because
            # an observation intended for a neighbouring repair predicts label 21.
            row = previous if unchanged else make_row(uid, ids, False)
            residual = (previous is not None and previous['meta'] is not None
                        and previous['meta'].get('class') in self.assets.saga20
                        and not len(np.setdiff1d(ids, previous['members'])))
            if not unchanged and residual and row['semantics']['class'] not in self.assets.saga20:
                # A 32-way label unsupported by the export vocabulary is not
                # geometric evidence that an existing object's residual vanished.
                # Retain its last committed class, but expose the conflict. No new
                # object or expansion can acquire a class through this fallback.
                fallback = dict(reason='unsupported_residual_class',
                    reestimated_class=row['semantics']['class'],
                    reestimated_score=row['semantics'].get('score'),
                    retained_class=previous['meta']['class'])
                row['meta'] = dict(previous['meta'], classification_fallback=fallback)
                row['incumbent'] = True
                classification_fallbacks.append(dict(uid=uid, members=len(ids), **fallback))
                after[uid] = row
            if unchanged or row['semantics']['class'] in self.assets.saga20:
                after[uid] = row
        # Connected conflicts, using the fixed before/proposed memberships, define the
        # objects whose mean must improve. This is one rollback pass, not an optimizer.
        checks = []
        for component in components:
            old_score = sum(before[u]['quality']['score'] for u in component if u in before)
            new_score = sum(after[u]['quality']['score'] for u in component if u in after)
            lost_core = []
            changed = any(not np.array_equal(before[u]['members'] if u in before else [],
                                             after[u]['members'] if u in after else []) for u in component)
            for uid in component:
                if uid not in before:
                    continue
                b = before[uid]
                exclusive = b['core'][core_count[b['core']] <= 1]
                # A verified duplicate may legitimately retain the same physical core
                # under another instance ID. That is not a loss to a different object.
                retained_uid = uid if uid in after else suppressed.get(uid)
                retained = after[retained_uid]['members'] if retained_uid in after else []
                removed = np.setdiff1d(exclusive, retained)
                unsupported = np.setdiff1d(removed, b['negative'])
                if len(unsupported):
                    lost_core.append(dict(uid=uid, count=len(unsupported)))
            rollback = changed and (new_score <= old_score + 1e-12 or bool(lost_core))
            if rollback:
                for uid in component:
                    after.pop(uid, None)
                    if uid in before:
                        after[uid] = before[uid]
            checks.append(dict(uids=component, before_mean=old_score/len(component),
                proposed_mean=new_score/len(component), rollback=rollback, unsupported_core_loss=lost_core))
        labels = np.full(n, -1, np.int64); metadata = {}
        for label, (uid, row) in enumerate(sorted(after.items())):
            members = row['members']
            if (labels[members] >= 0).any():
                raise AssertionError('conflict rollback produced overlapping instances')
            labels[members] = label
            if row['incumbent'] and row['meta']:
                meta = dict(row['meta'])
            else:
                meta = dict(**{'class': row['semantics']['class']}, score=row['quality']['score'])
            metadata[label] = dict(meta, point_count=len(members), object_uid=uid,
                                   score_source='common_multiview_evidence')
        contracted = normalize_prediction(labels, metadata)
        payload = dict(point_labels=contracted.point_labels.tolist(), instances=contracted.instances,
                       prediction_contract=contracted.audit, repair_policy='multiview-evidence',
                       instance_aliases={u: v for u, v in suppressed.items() if v in after and u not in after})
        save(dest / 'scene.json', payload, compact=True)
        actual = {u: r['members'] for u, r in after.items()}
        actual.update({u: self.actual_for(payload, u) for u in banks if u not in actual})
        np.savez_compressed(dest / 'actual-members.npz', **actual)
        for row in ownership_rows:
            uid = row['uid']; original = choices[uid]['members']
            final = after[uid]['members'] if uid in after else np.empty(0, np.int64)
            row['final_members'] = len(final)
            row['lost_members'] = len(np.setdiff1d(original, final))
            targets, counts = np.unique(contracted.point_labels[np.setdiff1d(original, final)], return_counts=True)
            row['lost_to'] = {str(int(t)): int(c) for t, c in zip(targets, counts)}
        save(dest / 'summary.json', dict(assembly='residual-class-core-v7', suppressed=suppressed,
            duplicate_core_transfers=core_transfers,
            classification_fallbacks=classification_fallbacks,
            conflicts=checks, ownership=ownership_rows, execution_complete=True))
        return payload

    def actual_for(self, payload, uid):
        labels = np.asarray(payload['point_labels'])
        uid = payload.get('instance_aliases', {}).get(uid, uid)
        keys = [int(k) for k, m in payload['instances'].items() if m.get('object_uid') == uid]
        return np.flatnonzero(np.isin(labels, keys))

    def finalize_cached_semantics(self, dest):
        """Budgeted writeback classification only; membership is immutable here."""
        payload = read(dest / 'scene.json')
        if payload.get('classification_complete', True):
            return
        labels = np.asarray(payload['point_labels'], np.int64)
        for key, meta in payload['instances'].items():
            if meta.get('classification_status') != 'pending_model':
                continue
            semantics = self.classify_members(np.flatnonzero(labels == int(key)), meta['semantic_views'])
            label, status = semantic_state(semantics, self.assets.saga20)
            meta.update(**{'class':label}, classification_status=status, semantics=semantics)
        payload['classification_complete'] = True
        save(dest / 'scene.json', payload, compact=True)
        allowed = {k:m for k,m in payload['instances'].items() if m['class'] in self.assets.saga20}
        exported = normalize_prediction(labels, allowed)
        save(dest / 'scene-evaluation.json', dict(point_labels=exported.point_labels.tolist(),
            instances=exported.instances, prediction_contract=exported.audit,
            geometry_source=str(dest/'scene.json')), compact=True)

    def export_legacy(self, banks, dest):
        """Same selected library through historic C ownership, evaluated separately."""
        if (dest / 'scene.json').exists():
            return
        proposals = []
        for uid, bank in banks.items():
            row, members = self.selected(bank)
            insides, visibles = [], []
            for path in row['associated_paths']:
                _, _, mass = self.observation(path)
                insides.append(mass['inside'][members]); visibles.append(mass['visible'][members])
            if insides:
                a, v = np.sum(insides, axis=0), np.sum(visibles, axis=0)
                ratio = np.divide(a, v, out=np.zeros_like(a), where=v > 0)
            else:
                ratio = np.zeros(len(members))
            proposals.append(dict(uid=uid, members=members, ratio=ratio,
                **{'class': row['semantics']['class'], 'score': row['semantics']['score']}))
        merged = merge_scene(self.b0, proposals, self.assets.saga20, policy='C')
        save(dest / 'scene.json', merged['payload'], compact=True)
        save(dest / 'summary.json', merged['summary'])

    def full(self, *, diagnostic=False):
        results = []
        for sid in self.plan['bindings']:
            self.scene(sid)
            original_panels = read(self.reference / 'scenes' / sid / 'panels.json')
            sources = sorted(self.assets.sources, key=lambda s: s.candidate_uid)
            if diagnostic:
                wanted = {f'{sid}:C0:{i:06d}' for i in TAIL_SOURCES[sid]}
                wanted |= {c['candidate_uid'] for c in self.plan['local_cases'] if c['scene_id'] == sid and c['source_kind'] == 'C0'}
                sources = [s for s in sources if s.candidate_uid in wanted]
            dest = self.out / ('diagnostic-scenes' if diagnostic else 'scenes') / sid
            shared_scene = self.shared / 'sources' / sid
            common_all, panels, initial = {}, {}, {}
            for source in sources:
                self.obs_cache.clear()
                uid = source.candidate_uid
                common = self.common_initial(uid, source, original_panels[uid]['construction'][:2], shared_scene / slug(uid))
                panel = self.choose_views(uid, common, original_panels[uid], shared_scene / slug(uid) / 'views.json')
                common_all[uid], panels[uid] = common, panel
                initial[uid] = self.mode_initial(uid, common, dest / 'banks-initial' / slug(uid))
            save(dest / 'panels.json', panels)
            excluded = {uid: [p['heldout']] if p.get('heldout') else [] for uid, p in panels.items()}
            initial_scene = self.assemble_evidence(initial, dest / 'initial', excluded_views=excluded)
            open_banks, open_scene = {}, None
            for protocol in ('matched-open', 'feedback'):
                banks = {}
                for source in sources:
                    self.obs_cache.clear()
                    uid = source.candidate_uid
                    banks[uid] = self.final_bank(uid, common_all[uid], initial[uid], panels[uid],
                        self.actual_for(initial_scene, uid), protocol, dest / ('banks-'+protocol) / slug(uid),
                        open_bank=open_banks.get(uid))
                payload = self.assemble_evidence(banks, dest / protocol,
                    baseline=initial_scene if protocol == 'matched-open' else open_scene, excluded_views=excluded)
                if protocol == 'matched-open':
                    open_banks, open_scene = banks, payload
                    if self.common_geometry:
                        for uid in open_banks:
                            open_banks[uid]['actual_members'] = self.actual_for(payload, uid).tolist()
                self.export_legacy(banks, dest / (protocol+'-legacy-assembly'))
                save(dest / (protocol+'-banks.json'), {u: str(dest / ('banks-'+protocol) / slug(u) / 'bank.json') for u in banks})
                for source in sources:
                    uid = source.candidate_uid; view = panels[uid]['heldout']
                    if view is None:
                        continue
                    row, members = self.selected(banks[uid])
                    comparison_image(self.data(view)['rgb'], [None, self.project(view, source.support_ids),
                        self.project(view, members), self.project(view, self.actual_for(payload, uid))],
                        ['RGB', 'Original', 'Selected 3D', 'Actual scene'],
                        dest / protocol / 'images' / (slug(uid)+'.jpg'))
                    if diagnostic:
                        self.draw_inputs(banks[uid], dest / protocol / 'inputs' / slug(uid), actual=self.actual_for(payload, uid))
                if diagnostic and protocol == 'matched-open':
                    # One view-policy diagnostic, not another full experiment arm.
                    original_banks = {}
                    for source in sources:
                        self.obs_cache.clear(); uid = source.candidate_uid
                        old_panel = dict(panels[uid], construction=original_panels[uid]['construction'], changed=False)
                        original_banks[uid] = self.final_bank(uid, common_all[uid], initial[uid], old_panel,
                            self.actual_for(initial_scene, uid), protocol, dest / 'banks-original-views' / slug(uid))
                    self.assemble_evidence(original_banks, dest / 'original-views', baseline=initial_scene,
                                           excluded_views=excluded)
            results.append(dict(scene_id=sid, source_count=len(sources), view_changes=sum(p['changed'] for p in panels.values())))
        save(self.out / ('diagnostic-result.json' if diagnostic else 'scene-result.json'),
             dict(execution_complete=diagnostic or sum(r['source_count'] for r in results) == 374,
                  scenes=results, stats=self.stats, diagnostic_only=diagnostic))

    def local(self):
        # Existing local annotations have two construction views, no feedback RGB.
        for sid in self.plan['bindings']:
            self.scene(sid)
            banks = {}
            for case in self.plan['local_cases']:
                if case['scene_id'] != sid:
                    continue
                self.obs_cache.clear()
                uid = case['candidate_uid']
                source = self.sources[uid] if case['source_kind'] == 'C0' else self.original[uid]
                views = [v['camera_uid'] for v in case['views'][:2]]
                common = self.common_initial(uid, source, views, self.shared / 'local' / case['case_id'], case=case)
                banks[uid] = self.mode_initial(uid, common, self.out / 'local' / case['case_id'])
            self.assemble_evidence(banks, self.out / 'local-scenes' / sid)
        # All local predictions saved before opening any annotation.
        self.evaluate_local({sid: {case['candidate_uid']: read(self.out / 'local' / case['case_id'] / 'bank.json')
                            for case in self.plan['local_cases'] if case['scene_id'] == sid}
                             for sid in self.plan['bindings']})

    def draw_inputs(self, bank, dest, actual=None):
        from PIL import Image, ImageDraw
        groups = {}
        _, selected = self.selected(bank)
        for path in bank['pool']:
            row = read(Path(path) / 'observation.json')
            groups[row['raw_group']] = row
        for i, (group, row) in enumerate(sorted(groups.items())):
            with np.load(Path(group) / 'raw.npz', allow_pickle=False) as z:
                alternatives, observed = z['alternatives'], z['observed_pixels']
            rgb = self.data(row['camera_uid'])['rgb']
            path = dest / f'input-{i:03d}.png'
            masks = [None, observed, *alternatives, self.project(row['camera_uid'], selected)]
            labels = ['RGB / crop / prompts', 'Observed', 'SAM 0', 'SAM 1', 'SAM 2', 'Selected 3D']
            if actual is not None:
                masks.append(self.project(row['camera_uid'], actual)); labels.append('Actual scene 3D')
            comparison_image(rgb, masks, labels, path)
            canvas = Image.open(path); draw = ImageDraw.Draw(canvas)
            scale = 360 / rgb.shape[1]; crop = row['actual_input']['crop']
            x0, y0 = crop['left']*scale, crop['top']*scale+28
            x1, y1 = x0+crop['width']*scale, y0+crop['height']*scale
            draw.rectangle((max(0,x0), max(28,y0), min(359,x1), min(canvas.height-1,y1)), outline='yellow', width=2)
            draw.rectangle((0,28,359,canvas.height-1), outline='orange', width=2)
            points = row['actual_input'].get('positive_points_image', [row['actual_input'].get('point_image')])
            for point in points:
                if point is not None:
                    x, y = point[0]*scale, point[1]*scale+28
                    draw.ellipse((x-3,y-3,x+3,y+3), fill='red')
            canvas.save(path)
        save(dest / 'input-index.json', [dict(r, raw_group=g) for g,r in groups.items()])


def prepare(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    fields = dict(reference_output=str(args.reference_output.resolve()), prior_mode=args.prior_mode,
        scale_priors=str(args.priors.resolve()), reuse_output=None, projection_domain='observed',
        shared_output=str(args.shared_output.resolve()), saved_global=str(args.saved_global.resolve()))
    if args.selector != 'legacy':
        fields['selector'] = args.selector
    if (output / 'plan.json').exists():
        plan = read(output / 'plan.json')
        if any(plan.get(k) != v for k,v in fields.items()):
            raise ValueError('Changed inference inputs require a new output directory')
        return
    plan = read(args.reference_output / 'plan.json')
    plan.update(fields, experiment=EXPERIMENT, score_weights=[.45/.8,.35/.8,0.],
        size_penalty_max=0., no_feedback=False, max_feedback_rounds=1,
        quantiles=['q25','q50','q75'], photo_edge_pixels=2, occlusion_depth_ratio=1.05,
        max_replaced_views=2, round_cap_seconds=11.2*3600,
        protocols=['initial','matched-open','feedback'],
        selection='source-independent shared-pool 3D evidence; no size or SAM-quality ranking',
        input_rule='separate robust object center and reliable internal positive points; all three SAM masks',
        view_order='fixed initial two, common geometry-only bounded replacement of original later slots',
        unknown_domain='outside crop/photo, padding, unreliable, ambiguous SAM parts, suspected occlusion',
        max_mask_seeds_per_branch=12)
    save(output / 'plan.json', plan)
    # Exact masked-image semantics are reusable across all conditions and stages.
    target = args.shared_output / 'semantics'
    target.mkdir(parents=True, exist_ok=True)
    if not (output / 'semantics').exists():
        (output / 'semantics').symlink_to(target.resolve(), target_is_directory=True)


def evaluate_scenes(output):
    from run_effective_repair_evaluation import evaluate
    from .taxonomy import load_taxonomy
    from .evaluation_strata import load_evaluation_strata
    plan = read(output / 'plan.json')
    original = read(Path(plan['reference_output']) / 'evaluation-manifest.json')
    conditions = ['legacy_C_open', 'initial', 'matched-open', 'feedback',
                  'matched-open-legacy-assembly', 'feedback-legacy-assembly']
    scenes = []
    for old in original['scenes']:
        sid = old['scene_id']
        row = {k:old[k] for k in ('scene_id','gt_npz','gaussian_ply','gaussian_to_gt_transform','b0_output_json')}
        row['condition_outputs'] = {'legacy_C_open':old['condition_outputs']['C_open']}
        row['condition_outputs'].update({c:{'output_json':str(output/'scenes'/sid/c/
            ('scene-evaluation.json' if plan.get('selector') in (GEOMETRY_SELECTOR, REGIONAL_SELECTOR)
             and c in ('initial','matched-open','feedback') else 'scene.json'))} for c in conditions[1:]})
        scenes.append(row)
    save(output / 'evaluation-manifest.json', dict(original, conditions=conditions, scenes=scenes))
    evaluate(output / 'evaluation-manifest.json', output, load_taxonomy(), load_evaluation_strata())
    from .category_scale_diagnostics import evaluate_banks
    for protocol in ('matched-open','feedback'):
        evaluate_banks(output, bank_directory='banks-'+protocol, result_prefix=protocol+'-')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=['multiview-repair'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-output', type=Path, default=ROOT/'effective-repair-01')
    parser.add_argument('--priors', type=Path, default=BASE/'artifacts/category-priors-20260804/category_priors.json')
    parser.add_argument('--saved-global', type=Path, default=ROOT/'category-scale-20260918/global')
    parser.add_argument('--shared-output', type=Path, default=ROOT/EXPERIMENT/'shared')
    parser.add_argument('--prior-mode', choices=['category','global'], default='category')
    parser.add_argument('--selector', choices=['legacy', GEOMETRY_SELECTOR, REGIONAL_SELECTOR], default='legacy')
    parser.add_argument('--stage', choices=['local','writeback','diagnostic','scene','all','evaluate'], default='all')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--attempt', default='01')
    parser.add_argument('--error-rerun', action='store_true', help='Specific implementation error rerun, charged to reserve')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        began = time.monotonic()
        try:
            runtime = MultiviewRuntime(args.output)
            if args.stage == 'writeback':
                # Only exact saved local banks: no SAM or new candidate search.
                by_scene = {}
                for sid in runtime.plan['bindings']:
                    runtime.scene(sid)
                    banks = {c['candidate_uid']: read(args.output/'local'/c['case_id']/'bank.json')
                             for c in runtime.plan['local_cases'] if c['scene_id'] == sid}
                    runtime.assemble_evidence(banks, args.output/'local-scenes'/sid)
                    if runtime.common_geometry:
                        runtime.finalize_cached_semantics(args.output/'local-scenes'/sid)
                    by_scene[sid] = banks
                runtime.evaluate_local(by_scene)
                save(args.output/'writeback-result.json', dict(execution_complete=True,
                    source='saved local candidate banks only',
                    assembly=GEOMETRY_SELECTOR if runtime.common_geometry else 'residual-class-core-v7'))
            elif args.stage == 'local':
                runtime.local()
            else:
                runtime.full(diagnostic=args.stage == 'diagnostic')
            save(args.output/f'{args.stage}-worker.json', dict(execution_complete=True,
                seconds=time.monotonic()-began, stats=runtime.stats))
            return 0
        except Exception:
            save(args.output/f'{args.stage}-failure-{int(time.time())}.json',
                dict(execution_complete=False, traceback=traceback.format_exc(), seconds=time.monotonic()-began))
            raise
    prepare(args)
    if args.stage == 'evaluate':
        evaluate_scenes(args.output)
        return 0
    from .object_verification.study import StudyLedger
    from .object_verification.supervisor import run_supervised_task
    ledger = StudyLedger(ROOT/'study-ledger.jsonl', VERSION)
    for stage in (['local','diagnostic','scene'] if args.stage == 'all' else [args.stage]):
        result_file = args.output/f'{stage}-result.json'
        if result_file.exists() and read(result_file).get('execution_complete'):
            continue
        state = ledger.status()
        phase = [a for a in state['attempts'] if a['task_id'].startswith('category-scale-')]
        current = [a for a in phase if '-multiview-' in a['task_id']]
        spent = lambda rows: sum(a.get('occupied_gpu_seconds',0.) for a in rows)
        # The approved original plan has a separate two-hour repair reserve.
        # A concrete error rerun uses it, rather than also consuming an exhausted
        # E4 cap. Historical attempts remain charged to their original groups;
        # older reserve-labelled attempts also count against the reserve below.
        group = 'reserve' if args.error_rerun else ('E3' if stage == 'writeback' else
            'E4' if stage in ('local','diagnostic') else ('E1' if args.prior_mode == 'category' else 'E2'))
        bucket = group
        bucket_cap = {'E4':.9,'E1':5.,'E2':2.8,'E3':1.,'reserve':1.5}[bucket]*3600
        bucket_spent = spent([a for a in current if f'-multiview-{bucket}-' in a['task_id']])
        group_attempts = [a for a in phase if a['task_id'].startswith('category-scale-'+group+'-')
                          or (group == 'reserve' and '-multiview-reserve-' in a['task_id'])]
        cap = int(min(state['remaining_gpu_seconds'],18*3600-spent(phase),11.2*3600-spent(current),
                      {'E1':8,'E2':4,'E3':3,'E4':1,'reserve':2}[group]*3600-spent(group_attempts),
                      bucket_cap-bucket_spent))
        save(args.output.parent/'budget.json', dict(cumulative_gpu_hours=state['occupied_gpu_seconds']/3600
             if 'occupied_gpu_seconds' in state else (24-state['remaining_gpu_seconds']/3600),
             round_gpu_hours=spent(current)/3600, phase_gpu_hours=spent(phase)/3600,
             next_cap_seconds=cap, group=group, bucket=bucket, attempts=current))
        if cap <= 5:
            raise RuntimeError('Approved GPU budget exhausted; preserve all partial results')
        task = f'category-scale-{group}-multiview-{bucket}-{args.prior_mode}-{stage}-{args.attempt}'
        argv = ['/usr/bin/timeout','--signal=TERM','--kill-after=5',str(cap-5),sys.executable,
                str(Path(__file__).resolve().parents[1]/'run_effective_repair.py'),
                '--experiment','multiview-repair','--output',str(args.output),'--stage',stage,'--worker']
        env = dict(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
        spec = dict(experiment_version=VERSION,kind='gpu',argv=argv,cwd=str(BASE),env=env,
                    comparison=EXPERIMENT,stage=stage,cap_seconds=cap)
        result = run_supervised_task(ledger,task,spec=spec,argv=argv,cwd=BASE,
            attempt_dir=args.output/f'job-{stage}-{args.attempt}',env=env,poll_seconds=1,
            gate_validator=lambda _: {'authorization':'User explicitly requested multiview repair implementation and execution'})
        save(args.output/f'process-{stage}-{args.attempt}.json', result)
        if not result['process_complete']:
            return 1
    if (args.output/'scene-result.json').exists():
        evaluate_scenes(args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
