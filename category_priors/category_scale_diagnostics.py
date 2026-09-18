"""Posthoc candidate-bank evaluation. This module never selects inference outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix

from run_effective_repair import read, save
from run_effective_repair_evaluation import write_csv, resolve
from .evaluator import load_ground_truth_npz, load_ply_xyz, apply_transform, map_gaussians_to_gt
from .recheck_evaluation import ground_truth_objects
from .taxonomy import load_taxonomy


def gaussian_overlap_index(mapping, objects, gaussian_count):
    valid = mapping >= 0
    totals = np.bincount(mapping[valid], minlength=gaussian_count)
    counts = [np.bincount(mapping[valid & obj.mask], minlength=gaussian_count) for obj in objects]
    sparse = csr_matrix(np.stack(counts) if counts else np.empty((0, gaussian_count)))
    sizes = np.array([np.count_nonzero(obj.mask) for obj in objects])
    return totals, sparse, sizes


def member_overlaps(members, index):
    totals, sparse, sizes = index
    intersections = np.asarray(sparse[:, members].sum(axis=1)).ravel()
    union = sizes + totals[members].sum() - intersections
    return np.divide(intersections, union, out=np.zeros(len(sizes)), where=union > 0)


def assign_sources(matrix, candidate_ids, source_ids, objects):
    """One source may contribute only one variant to one GT, even in the oracle."""
    matches = {}
    if matrix.size:
        ss, gg = linear_sum_assignment(-matrix)
        matches = {int(g): int(s) for s, g in zip(ss, gg) if matrix[s, g] > 0}
    return [dict(gt_id=obj.gt_id, **{'class': obj.class_name},
        iou=float(matrix[matches[g], g]) if g in matches else 0.,
        source_uid=source_ids[matches[g]] if g in matches else None,
        candidate_id=candidate_ids[matches[g], g] if g in matches else None)
        for g, obj in enumerate(objects)]


def evaluate_banks(output, *, bank_directory='banks', result_prefix=''):
    output = Path(output)
    manifest = read(output / 'evaluation-manifest.json')
    taxonomy = load_taxonomy()
    all_rows = []
    for spec in manifest['scenes']:
        sid = spec['scene_id']
        xyz_gt, gt = load_ground_truth_npz(resolve(output, spec['gt_npz']), sid)
        xyz = apply_transform(load_ply_xyz(resolve(output, spec['gaussian_ply'])), spec['gaussian_to_gt_transform'])
        mapping, _ = map_gaussians_to_gt(xyz_gt, xyz, np.arange(len(xyz)), .05)
        objects = ground_truth_objects(gt, xyz_gt, taxonomy.canonical_classes, 100)
        index = gaussian_overlap_index(mapping, objects, len(xyz))
        banks = [read(p) for p in sorted((output / 'scenes' / sid / bank_directory).glob('*/bank.json'))]
        source_ids = [b['uid'] for b in banks]
        kinds = ['selected_class_aware', 'selected_class_agnostic', 'oracle_class_aware', 'oracle_class_agnostic',
                 'legacy_G2_class_aware', 'legacy_G2_class_agnostic']
        matrices = {k: np.zeros((len(banks), len(objects))) for k in kinds}
        candidates = {k: np.full((len(banks), len(objects)), None, dtype=object) for k in kinds}
        for s, bank in enumerate(banks):
            for row in bank['candidates']:
                if row['member_count'] < 3:
                    continue
                with np.load(row['members_file'], allow_pickle=False) as z:
                    overlaps = member_overlaps(z['members'], index)
                aware = overlaps * np.array([obj.class_name == row['semantics']['class'] for obj in objects])
                for suffix, scores in [('class_aware', aware), ('class_agnostic', overlaps)]:
                    keys = ['oracle_' + suffix]
                    if row['id'] == bank['selected_id']:
                        keys.append('selected_' + suffix)
                    if row['id'] == 'legacy-G2':
                        keys.append('legacy_G2_' + suffix)
                    for key in keys:
                        better = scores > matrices[key][s]
                        matrices[key][s, better] = scores[better]
                        candidates[key][s, better] = row['id']
        for kind in kinds:
            all_rows.extend(dict(scene_id=sid, comparison=kind, posthoc=kind.startswith('oracle'), **row)
                            for row in assign_sources(matrices[kind], candidates[kind], source_ids, objects))
        print('bank evaluation', sid, len(banks), 'sources', len(objects), 'GT objects', flush=True)
    write_csv(output / (result_prefix+'bank-object-iou.csv'), all_rows)
    summary = {kind: float(np.mean([r['iou'] for r in all_rows if r['comparison'] == kind]))
               for kind in sorted({r['comparison'] for r in all_rows})}
    save(output / (result_prefix+'bank-diagnostic-summary.json'), dict(means=summary,
        oracle_is_posthoc_not_automatic=True, max_one_candidate_per_source_and_one_source_per_gt=True,
        selected_is_before_scene_ownership=True))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(evaluate_banks(args.output))


if __name__ == '__main__':
    main()
