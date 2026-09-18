"""DEV2 category-scale experiment, invoked through run_effective_repair.py.

Inference writes complete candidate banks before the separate evaluation methods
open annotations. Existing observations and the cumulative GPU ledger are reused.
"""
from __future__ import annotations

import argparse
import hashlib
import html
from pathlib import Path
import os
import sys
import time
import traceback

import numpy as np

from run_effective_repair import BASE, ROOT, VERSION, Runtime, read, save, slug
from .category_scale import candidate_score, merge_ranked, projection_iou, resolve_prior, scale_slots, select_candidate
from .effective_repair_core import merge_scene, pixel_metrics


def comparison_image(rgb, masks, labels, target):
    from PIL import Image, ImageDraw
    width = 360
    height = round(rgb.shape[0] * width / rgb.shape[1])
    canvas = Image.new('RGB', (width * len(labels), height + 28), 'white')
    draw = ImageDraw.Draw(canvas)
    for j, (label, mask) in enumerate(zip(labels, masks)):
        pixels = rgb.copy()
        if mask is not None:
            pixels[mask] = (.55 * pixels[mask] + .45 * np.array([35, 115, 255])).astype(np.uint8)
        canvas.paste(Image.fromarray(pixels).resize((width, height)), (width*j, 28))
        draw.text((width*j+5, 6), label, fill='black')
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)


class ScaleRuntime(Runtime):
    def __init__(self, out):
        super().__init__(out)
        self.reference = Path(self.plan['reference_output'])
        self.mode = self.plan['prior_mode']
        self.companion = Path(self.plan['reuse_output']) if self.plan.get('reuse_output') else None

    def scene(self, sid):
        super().scene(sid)
        self.scale_priors = read(self.plan['scale_priors']) if self.plan.get('scale_priors') else self.assets.priors
        if self.scale_priors.get('provenance', {}).get('splits') != ['train']:
            raise ValueError('Category-scale inference requires train-only size statistics')
        for camera in self.assets.cameras:
            cached = self.reference / 'contributors' / sid / (slug(camera.uid) + '.npz')
            if cached.exists():
                self.plan['contributor_files'][camera.uid] = dict(path=str(cached), kind='old_npz')
        coverage = {name: resolve_prior(self.scale_priors, name, self.mode)
                    for name in self.plan['classes32']}
        save(self.out / 'scenes' / sid / 'prior-coverage.json', coverage)
        # A run where every class falls back is not a category-prior experiment.
        if self.mode == 'category' and not any(r['category_specific'] for r in coverage.values()):
            raise ValueError('No active category size statistics. Fit train-only priors before E1.')

    def classify(self, uid, mask):
        key = hashlib.blake2b(mask.tobytes(), digest_size=16).hexdigest()
        for root in (getattr(self, 'reference', None), getattr(self, 'companion', None)):
            if root is not None:
                path = root / 'semantics' / slug(uid) / (key + '.json')
                if path.exists():
                    return read(path)
        return super().classify(uid, mask)

    def old_paths(self, uid, members, anchors, views, dest, *, case=None):
        if case is not None:
            geom = self.reference / 'local' / case['case_id'] / 'geometry' / 'geometry.json'
            if geom.exists():
                return read(geom)['paths']
        paths = []
        for i, view in enumerate(views):
            saved = self.reference / 'scenes' / self.sid / 'observations' / slug(uid) / 'original' / slug(view)
            if (saved / 'observation.json').exists():
                paths.append(str(saved))
            else:
                paths.append(self.observe(view, members, anchors, dest / f'old-{i}'))
        return paths

    def score_members(self, members, paths, *, prior=None):
        from .scannet import pca_obb
        overlaps, qualities, views, inside, visible = [], [], [], [], []
        for path in paths:
            row, arrays, mass = self.observation(path)
            views.append(row['camera_uid'])
            if arrays is None:
                continue
            projected = self.project(row['camera_uid'], members)
            mask = arrays['sam']
            domain = self.plan.get('projection_domain', 'full')
            if domain not in ('full', 'observed'):
                raise ValueError('projection_domain must be full or observed')
            observed = arrays.get('observed_pixels') if domain == 'observed' else None
            overlaps.append(projection_iou(projected, mask, observed))
            qualities.append(float(row['qualities'][row['chosen']]))
            inside.append(mass['inside'][members]); visible.append(mass['visible'][members])
        semantic = self.classify_members(members, views)
        effective = prior or resolve_prior(self.scale_priors, semantic['class'], self.mode)
        if len(members):
            extents, _, _ = pca_obb(self.assets.xyz_scene[members])
            diagonal = float(np.linalg.norm(extents) * self.assets.scale_m_per_unit)
        else:
            diagonal = 0.
        quality = candidate_score(overlaps=overlaps, semantic_score=semantic['score'],
            sam_qualities=qualities, diagonal_m=diagonal, prior=effective)
        if inside:
            a, v = np.sum(inside, axis=0), np.sum(visible, axis=0)
            ratio = np.divide(a, v, out=np.zeros_like(a), where=v > 0)
        else:
            ratio = np.zeros(len(members), dtype=np.float32)
        return semantic, quality, ratio

    def hypothesis_slots(self, old_geometry):
        return scale_slots(self.scale_priors, self.plan['classes32'],
                           old_geometry['sam_semantics']['mean_cosines'], self.mode)

    def build_bank(self, uid, members, anchors, views, dest, *, case=None):
        if (dest / 'bank.json').exists():
            return read(dest / 'bank.json')
        old = self.old_paths(uid, members, anchors, views, dest, case=case)
        old_geometry, old_arrays = self.geometry(old, members, dest / 'old-geometry')
        slots = self.hypothesis_slots(old_geometry)
        rows = []

        def add(identifier, kind, ids, paths, prior=None):
            ids = np.asarray(ids, dtype=np.int64)
            semantic, quality, ratio = self.score_members(ids, paths, prior=prior)
            target = dest / 'candidates' / identifier
            target.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target / 'members.npz', members=ids, ratio=ratio)
            row = dict(id=identifier, kind=kind, member_count=len(ids), paths=paths,
                       semantics=semantic, quality=quality, prior=prior,
                       members_file=str(target / 'members.npz'))
            save(target / 'candidate.json', row)
            rows.append(row)

        add('original', 'original', members, old)
        for rule in ('G0', 'G2'):
            add('legacy-' + rule, 'legacy', old_arrays[rule], old)
        previous_paths = {}
        for prior in slots:
            branch = prior['branch']
            paths = []
            for i, view in enumerate(views):
                reuse = previous_paths.get((prior['quantile'], view))
                if reuse is None and self.companion is not None:
                    other = self.companion / dest.relative_to(self.out) / 'observations' / branch / slug(view)
                    if (other / 'observation.json').exists():
                        reuse = str(other)
                path = self.observe(view, members, anchors, dest / 'observations' / branch / slug(view),
                                    scale_prior=prior, reuse=reuse, source_state='category_scale')
                paths.append(path)
                previous_paths[(prior['quantile'], view)] = path
            geometry, arrays = self.geometry(paths, members, dest / 'geometry' / branch)
            # Incomplete evidence stays in the bank; old observations are not silently substituted.
            for rule in ('G0', 'G2'):
                add(branch + '-' + rule, 'category', arrays[rule], paths, prior)
        selected = select_candidate(rows)
        result = dict(uid=uid, views=views, hypotheses=slots, candidates=rows,
                      selected_id=selected['id'], original_members=len(members),
                      class_coverage=sum(p['category_specific'] for p in slots),
                      no_semantic_hypothesis=not slots)
        save(dest / 'bank.json', result)
        print('selected', uid, selected['id'], 'quality', round(selected['quality']['score'], 4), flush=True)
        return result

    def proposal_from_bank(self, bank):
        row = next(r for r in bank['candidates'] if r['id'] == bank['selected_id'])
        with np.load(row['members_file'], allow_pickle=False) as z:
            members, ratio = z['members'], z['ratio']
        return dict(uid=bank['uid'], members=members, ratio=ratio,
                    **{key: row['semantics'][key] for key in ('class', 'score')},
                    selection_score=row['quality']['score'])

    def draw_inputs(self, bank, dest):
        """Every distinct branch's actual crop and SAM output, without selecting by GT."""
        from PIL import Image, ImageDraw
        branches = {'legacy': next(r for r in bank['candidates'] if r['id'] == 'legacy-G2')}
        branches.update({r['id']: r for r in bank['candidates'] if r['kind'] == 'category' and r['id'].endswith('-G2')})
        for branch, candidate in branches.items():
            for i, path in enumerate(candidate['paths']):
                row, arrays, _ = self.observation(path)
                if arrays is None:
                    continue
                rgb = self.data(row['camera_uid'])['rgb']
                annotated = Image.fromarray(rgb.copy())
                draw = ImageDraw.Draw(annotated)
                actual = row.get('actual_input', {})
                crop = actual.get('crop', {})
                if crop:
                    x, y = crop['left'], crop['top']
                    draw.rectangle((x, y, x+crop['width'], y+crop['height']), outline='orange', width=3)
                point = actual.get('point_image')
                if point:
                    x, y = point
                    draw.ellipse((x-5, y-5, x+5, y+5), fill='red')
                box = actual.get('box_crop')
                if box is not None and crop:
                    x0, y0, x1, y1 = box
                    draw.rectangle((x0+crop['left'], y0+crop['top'], x1+crop['left'], y1+crop['top']),
                                   outline='yellow', width=3)
                width = 360
                height = round(rgb.shape[0] * width / rgb.shape[1])
                canvas = Image.new('RGB', (3*width, height+28), 'white')
                canvas.paste(annotated.resize((width, height)), (0, 28))
                if 'encoded_rgb' in arrays:
                    encoded = Image.fromarray(arrays['encoded_rgb'])
                    encoded.thumbnail((width, height))
                    canvas.paste(encoded, (width, 28))
                sam = rgb.copy()
                sam[arrays['sam']] = (.55*sam[arrays['sam']] + .45*np.array([35, 115, 255])).astype(np.uint8)
                canvas.paste(Image.fromarray(sam).resize((width, height)), (2*width, 28))
                drawer = ImageDraw.Draw(canvas)
                for j, label in enumerate(('Actual crop / prompt', 'SAM image input', 'SAM output')):
                    drawer.text((j*width+5, 6), label, fill='black')
                canvas.save(dest / f'input-{branch}-{i}.png')

    def export(self, banks, dest, *, ranked=False, panel=None):
        proposals = [self.proposal_from_bank(bank) for bank in banks.values()]
        if ranked:
            b0_quality = {}
            for key in self.b0['instances']:
                uid = f'B0:{key}'
                ids = np.flatnonzero(np.asarray(self.b0['point_labels']) == int(key))
                # Choose views solely by source-member overlap with the frozen C0 panel.
                overlaps = [(len(np.intersect1d(ids, s.support_ids)), s.candidate_uid) for s in self.assets.sources]
                best = min(overlaps, key=lambda v: (-v[0], v[1]))[1]
                views = panel[best]['construction']
                paths = self.old_paths(uid, ids, ids, views, dest / 'b0-quality' / slug(uid))
                _, quality, _ = self.score_members(ids, paths)
                b0_quality[uid] = quality['score']
            merged = merge_ranked(self.b0, proposals, b0_quality, self.assets.saga20)
            actual = {p['uid']: merged['assigned'].get(p['uid'], np.empty(0, np.int64)) for p in proposals}
            summary = dict(assembly='ranked', b0_quality=b0_quality, suppressed=merged['suppressed'])
        else:
            merged = merge_scene(self.b0, proposals, self.assets.saga20, policy='C')
            actual = {uid: row['exported_members'] for uid, row in merged['proposals'].items()}
            summary = dict(assembly='legacy-C', **merged['summary'])
        proposal_ids = {p['uid'] for p in proposals}
        for metadata in merged['payload']['instances'].values():
            if metadata.get('object_uid') in proposal_ids:
                metadata['score_source'] = 'member_projection_mean32'
        save(dest / 'scene.json', merged['payload'], compact=True)
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest / 'actual-members.npz', **actual)
        save(dest / 'summary.json', summary)
        return actual

    def local(self):
        banks_by_scene = {}
        # Finish ALL inference/selection before opening any evaluation annotations.
        for sid in self.plan['bindings']:
            self.scene(sid)
            banks = {}
            for case in self.plan['local_cases']:
                if case['scene_id'] != sid:
                    continue
                uid = case['candidate_uid']
                source = self.sources[uid] if case['source_kind'] == 'C0' else self.original[uid]
                members = np.asarray(source.support_ids if case['source_kind'] == 'C0' else source.members, np.int64)
                anchors = source.anchor_ids if case['source_kind'] == 'C0' else members
                views = [v['camera_uid'] for v in case['views'][:2]]
                banks[uid] = self.build_bank(uid, members, anchors, views,
                    self.out / 'local' / case['case_id'], case=case)
            self.export(banks, self.out / 'local-scenes' / sid)
            banks_by_scene[sid] = banks
        self.evaluate_local(banks_by_scene)

    def evaluate_local(self, banks_by_scene):
        from PIL import Image
        evaluations = {r['case_id']: r for r in read(self.reference / 'offline-evaluation-inputs.json')}
        results = []
        for sid, banks in banks_by_scene.items():
            self.scene(sid)
            with np.load(self.out / 'local-scenes' / sid / 'actual-members.npz', allow_pickle=False) as z:
                actual = {key: z[key] for key in z.files}
            for case in self.plan['local_cases']:
                if case['scene_id'] != sid:
                    continue
                uid, dest = case['candidate_uid'], self.out / 'local' / case['case_id']
                bank = banks[uid]
                members = {}
                for row in bank['candidates']:
                    with np.load(row['members_file'], allow_pickle=False) as z:
                        members[row['id']] = z['members']
                views = []
                for i, view in enumerate(case['views']):
                    camera = view['camera_uid']
                    annotation = evaluations[case['case_id']]['views'][i]
                    fg = np.asarray(Image.open(annotation['foreground'])) > 0
                    uncertain = np.asarray(Image.open(annotation['uncertain'])) > 0
                    projections = {key: self.project(camera, ids) for key, ids in members.items()}
                    scores = {key: pixel_metrics(mask, fg, uncertain) for key, mask in projections.items()}
                    final = self.project(camera, actual[uid])
                    scores['final_scene'] = pixel_metrics(final, fg, uncertain)
                    oracle = max(scores.keys() - {'final_scene'}, key=lambda key: scores[key]['iou'] or 0.)
                    scores['selected'] = scores[bank['selected_id']]
                    row = dict(camera_uid=camera, role=('I1', 'I2', 'H')[i], metrics=scores,
                        posthoc_oracle_id=oracle, oracle_not_used_for_selection=True)
                    save(dest / f'view-{i}.json', row)
                    views.append(row)
                    rgb = self.data(camera)['rgb']
                    np.savez_compressed(dest / f'view-{i}.npz', rgb=rgb, foreground=fg, uncertain=uncertain,
                        original=projections['original'], legacy_G0=projections['legacy-G0'], legacy_G2=projections['legacy-G2'],
                        selected=projections[bank['selected_id']], final_scene=final)
                    # Full-image panels keep all distant false positives visible.
                    labels = ['RGB', 'Original', 'Legacy G0', 'Legacy G2', 'Selected 3D', 'Scene 3D']
                    masks = [None, projections['original'], projections['legacy-G0'], projections['legacy-G2'], projections[bank['selected_id']], final]
                    comparison_image(rgb, masks, labels, dest / f'comparison-{i}.png')
                self.draw_inputs(bank, dest)
                original_ids, selected_ids = members['original'], members[bank['selected_id']]
                result = dict(case_id=case['case_id'], scene_id=sid, candidate_uid=uid,
                    selected_id=bank['selected_id'], views=views,
                    added_members=len(np.setdiff1d(selected_ids, original_ids)),
                    removed_members=len(np.setdiff1d(original_ids, selected_ids)))
                save(dest / 'result.json', result)
                results.append(result)
        save(self.out / 'local-result.json', dict(execution_complete=len(results) == len(self.plan['local_cases']),
            results=results, stats=self.stats, evaluation_only=True))

    def full(self, *, ranked=False):
        results = []
        for sid in self.plan['bindings']:
            self.scene(sid)
            panel = read(self.reference / 'scenes' / sid / 'panels.json')
            banks = {}
            for source in sorted(self.assets.sources, key=lambda s: s.candidate_uid):
                uid = source.candidate_uid
                dest = self.out / 'scenes' / sid / 'banks' / slug(uid)
                if ranked and not (dest / 'bank.json').exists():
                    raise FileNotFoundError('E3 requires the completed E1/E2 bank: ' + str(dest))
                banks[uid] = self.build_bank(uid, np.asarray(source.support_ids, np.int64),
                    source.anchor_ids, panel[uid]['construction'], dest)
            condition = 'ranked' if ranked else 'selected'
            actual = self.export(banks, self.out / 'scenes' / sid / condition, ranked=ranked, panel=panel)
            for uid, bank in banks.items():
                view = panel[uid]['heldout'] or next(iter(panel[uid]['construction']), None)
                if view is None:
                    continue
                variants = {}
                for candidate in bank['candidates']:
                    if candidate['id'] in ('original', 'legacy-G2', bank['selected_id']):
                        with np.load(candidate['members_file'], allow_pickle=False) as z:
                            variants[candidate['id']] = self.project(view, z['members'])
                comparison_image(self.data(view)['rgb'], [None, variants['original'], variants['legacy-G2'],
                    variants[bank['selected_id']], self.project(view, actual[uid])],
                    ['RGB', 'Original', 'Legacy G2', 'Selected 3D', 'Scene 3D'],
                    self.out / 'scenes' / sid / condition / 'images' / (slug(uid) + '.jpg'))
            results.append(dict(scene_id=sid, candidate_count=len(banks), condition=condition))
        save(self.out / ('ranked-result.json' if ranked else 'scene-result.json'),
             dict(execution_complete=sum(r['candidate_count'] for r in results) == 374, scenes=results, stats=self.stats))


def prepare(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    reference = args.reference_output.resolve()
    if output.resolve() == reference:
        raise ValueError('Category experiment output must differ from the frozen reference')
    fields = dict(reference_output=str(reference), prior_mode=args.prior_mode,
                  scale_priors=str(args.priors.resolve()) if args.priors else None,
                  reuse_output=str(args.reuse_output.resolve()) if args.reuse_output else None,
                  projection_domain=getattr(args, 'projection_domain', 'full'))
    if (output / 'plan.json').exists():
        old = read(output / 'plan.json')
        if any(old.get(k, 'full' if k == 'projection_domain' else None) != v for k, v in fields.items()):
            raise ValueError('Use a new output directory when changing priors, inputs or projection scoring')
        return
    plan = read(reference / 'plan.json')
    plan.update(fields, experiment='category-scale-20260917', score_weights=[.45, .35, .20],
                size_penalty_max=.10, quantiles=['q25', 'q50', 'q75'],
                phase_cap_seconds=18*3600, no_feedback=True)
    save(output / 'plan.json', plan)


def evaluate_scenes(output):
    from run_effective_repair_evaluation import evaluate
    from .taxonomy import load_taxonomy
    from .evaluation_strata import load_evaluation_strata
    plan = read(output / 'plan.json')
    original = read(Path(plan['reference_output']) / 'evaluation-manifest.json')
    conditions = ['legacy_C_open', plan['prior_mode']]
    ranked = (output / 'ranked-result.json').exists()
    if ranked:
        conditions.append(plan['prior_mode'] + '_ranked')
    scenes = []
    for old in original['scenes']:
        sid = old['scene_id']
        row = {key: old[key] for key in ('scene_id', 'gt_npz', 'gaussian_ply', 'gaussian_to_gt_transform', 'b0_output_json')}
        row['condition_outputs'] = {'legacy_C_open': old['condition_outputs']['C_open'],
            plan['prior_mode']: {'output_json': str(output / 'scenes' / sid / 'selected' / 'scene.json')}}
        if ranked:
            row['condition_outputs'][plan['prior_mode'] + '_ranked'] = {
                'output_json': str(output / 'scenes' / sid / 'ranked' / 'scene.json')}
        scenes.append(row)
    manifest = dict(original, conditions=conditions, scenes=scenes)
    save(output / 'evaluation-manifest.json', manifest)
    evaluate(output / 'evaluation-manifest.json', output, load_taxonomy(), load_evaluation_strata())
    summarize(output)


def summarize(output):
    import csv
    from run_effective_repair_evaluation import write_csv
    plan = read(output / 'plan.json')
    mode = plan['prior_mode']
    def load_iou(root, condition):
        with (root / 'object_iou.csv').open(encoding='utf-8-sig', newline='') as stream:
            return {(r['matching'], r['scene_id'], r['gt_id']): r for r in csv.DictReader(stream)
                    if r['condition'] == condition}
    current = load_iou(output, mode)
    baseline = load_iou(output, 'legacy_C_open')
    comparison = 'legacy_C_open'
    other = Path(plan['reuse_output']) if plan.get('reuse_output') else None
    if mode == 'global' and other and (other / 'object_iou.csv').exists():
        current, baseline = load_iou(other, 'category'), current
        comparison = 'matched_global'
    rows = []
    for key, row in current.items():
        old = baseline[key]
        rows.append(dict(matching=key[0], scene_id=key[1], gt_id=key[2],
            **{'class': row['class']}, comparison=comparison,
            candidate_iou=float(row['iou']), reference_iou=float(old['iou']),
            delta=float(row['iou'])-float(old['iou'])))
    write_csv(output / 'paired-object-effects.csv', rows)
    aware = [r for r in rows if r['matching'] == 'class_aware']
    improves = [r for r in aware if r['delta'] >= .10]
    declines = [r for r in aware if r['delta'] <= -.10]
    summary = dict(comparison=comparison, gt_count=len(aware),
        mean_iou_delta=float(np.mean([r['delta'] for r in aware])) if aware else None,
        improved_at_least_10pp=improves, declined_at_least_10pp=declines,
        improved_classes=sorted({r['class'] for r in improves}),
        per_scene={sid: float(np.mean([r['delta'] for r in aware if r['scene_id'] == sid]))
                   for sid in sorted({r['scene_id'] for r in aware})},
        category_attribution_available=comparison == 'matched_global',
        evidence_scope='Exploratory DEV2 only; targets are not a reporting gate')
    # Keep every source, including failures to intervene, in the delivered table.
    source_rows = []
    for path in sorted(output.glob('scenes/*/banks/*/bank.json')):
        bank = read(path)
        selected = next(r for r in bank['candidates'] if r['id'] == bank['selected_id'])
        observations = {p for r in bank['candidates'] if r['kind'] == 'category' for p in r['paths']}
        statuses = [read(Path(p) / 'observation.json')['status'] for p in sorted(observations)]
        source_rows.append(dict(scene_id=path.parents[2].name, source_uid=bank['uid'],
            hypotheses=';'.join(dict.fromkeys(p['hypothesis'] for p in bank['hypotheses'])),
            prior_sources=';'.join(dict.fromkeys(p['source'] for p in bank['hypotheses'])),
            category_prior_slots=bank['class_coverage'], candidate_count=len(bank['candidates']),
            successful_scale_observations=statuses.count('complete'),
            no_reliable_point_observations=statuses.count('no_reliable_point'),
            scale_observation_intervened='complete' in statuses,
            selected_id=selected['id'], selected_class=selected['semantics']['class'],
            original_members=bank['original_members'], selected_members=selected['member_count'],
            **selected['quality']))
    write_csv(output / 'source-selection.csv', source_rows)
    summary['intervention'] = dict(source_count=len(source_rows),
        without_category_prior=sum(r['category_prior_slots'] == 0 for r in source_rows),
        without_scale_observation=sum(not r['scale_observation_intervened'] for r in source_rows))
    if (output / 'local-result.json').exists():
        local = read(output / 'local-result.json')['results']
        summary['local_heldout'] = [dict(case_id=r['case_id'],
            selected=r['views'][2]['metrics']['selected']['iou'],
            legacy_G2=r['views'][2]['metrics']['legacy-G2']['iou'],
            oracle=r['views'][2]['metrics'][r['views'][2]['posthoc_oracle_id']]['iou'],
            oracle_is_posthoc=True) for r in local]
    save(output / 'scientific-summary.json', summary)
    pictures = sorted(output.glob('local/*/comparison-2.png'))
    sections = ['<!doctype html><meta charset="utf-8"><title>Category scale results</title>',
                '<style>body{font:16px sans-serif;max-width:1800px;margin:24px auto}img{width:100%}</style>',
                '<p>Blue: prediction. Full-image views retain distant errors. Numerical results: '
                '<a href="paired-object-effects.csv">all paired objects</a>.</p>']
    for picture in pictures:
        sections.append(f'<p>{html.escape(picture.parent.name)}</p><img src="{picture.relative_to(output).as_posix()}">')
    sections.append('<p>All crop / prompt / SAM input images:</p>')
    for picture in sorted(output.glob('local/*/input-*.png')):
        rel = picture.relative_to(output).as_posix()
        sections.append(f'<a href="{rel}">{html.escape(rel)}</a><br>')
    (output / 'index.html').write_text('\n'.join(sections), encoding='utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=['category-scale'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-output', type=Path, default=ROOT / 'effective-repair-01')
    parser.add_argument('--priors', type=Path)
    parser.add_argument('--prior-mode', choices=['category', 'global'], default='category')
    parser.add_argument('--projection-domain', choices=['full', 'observed'], default='full',
                        help='observed ignores pixels outside the SAM crop; full reproduces frozen E1/E2')
    parser.add_argument('--reuse-output', type=Path)
    parser.add_argument('--stage', choices=['local', 'scene', 'all', 'ranked', 'evaluate'], default='all')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--attempt', default='01')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        began = time.monotonic()
        try:
            runtime = ScaleRuntime(args.output)
            if args.stage == 'local':
                runtime.local()
            else:
                runtime.full(ranked=args.stage == 'ranked')
            save(args.output / f'{args.stage}-worker.json', dict(execution_complete=True,
                seconds=time.monotonic()-began, stats=runtime.stats))
            return 0
        except Exception:
            save(args.output / f'{args.stage}-failure-{int(time.time())}.json',
                 dict(execution_complete=False, traceback=traceback.format_exc(), seconds=time.monotonic()-began))
            raise
    prepare(args)
    if args.stage == 'evaluate':
        evaluate_scenes(args.output)
        return 0
    from .object_verification.study import StudyLedger
    from .object_verification.supervisor import run_supervised_task
    ledger = StudyLedger(ROOT / 'study-ledger.jsonl', VERSION)
    for stage in (['local', 'scene'] if args.stage == 'all' else [args.stage]):
        completed = args.output / f'{stage}-result.json'
        if completed.exists() and read(completed).get('execution_complete'):
            continue
        state = ledger.status()
        phase_attempts = [a for a in state['attempts'] if a['task_id'].startswith('category-scale-')]
        phase_spent = sum(a.get('occupied_gpu_seconds', 0.) for a in phase_attempts)
        group = 'E3' if stage == 'ranked' else 'E1' if args.prior_mode == 'category' else 'E2'
        group_spent = sum(a.get('occupied_gpu_seconds', 0.) for a in phase_attempts
                          if a['task_id'].startswith('category-scale-' + group + '-'))
        group_cap = {'E1': 8, 'E2': 4, 'E3': 3}[group] * 3600
        cap = int(min(state['remaining_gpu_seconds'], 18*3600-phase_spent, group_cap-group_spent))
        if cap <= 5:
            raise RuntimeError(f'{group}: cumulative experiment budget exhausted')
        task = f'category-scale-{group}-{args.prior_mode}-{stage}-{args.attempt}'
        argv = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=5', str(cap-5), sys.executable,
            str(Path(__file__).resolve().parents[1] / 'run_effective_repair.py'),
            '--experiment', 'category-scale', '--output', str(args.output), '--stage', stage, '--worker']
        env = dict(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        spec = dict(experiment_version=VERSION, kind='gpu', argv=argv, cwd=str(BASE), env=env,
                    comparison='category-scale-20260917', stage=stage, cap_seconds=cap)
        result = run_supervised_task(ledger, task, spec=spec, argv=argv, cwd=BASE,
            attempt_dir=args.output / f'job-{stage}-{args.attempt}', env=env, poll_seconds=1,
            gate_validator=lambda _: {'authorization': 'User requested implementation and execution of category-scale plan'})
        save(args.output / f'process-{stage}-{args.attempt}.json', result)
        if not result['process_complete']:
            return 1
    if (args.output / 'scene-result.json').exists():
        evaluate_scenes(args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
