"""Fit only the train-scene size statistics needed by the category-scale experiment.

python -m category_priors.fit_category_sizes --dataset-root scans --train-list scannetv2_train.txt --output priors.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .io import write_rows, hash_json
from .priors import fit_priors, write_priors
from .scannet import (discover_scene_files, load_mesh, read_axis_alignment, apply_transform,
                      physical_scene_id, read_scene_ids, _instance_assignment, triangle_areas,
                      sample_faces, voxelize, pca_obb, _stable_seed)
from .taxonomy import load_taxonomy


def fit_available_sizes(dataset_root, train_list, output, exclude=('scene0025_01', 'scene0645_00')):
    taxonomy = load_taxonomy()
    excluded = {physical_scene_id(s) for s in exclude}
    train_ids = sorted(set(read_scene_ids(train_list)))
    rows, missing, included = [], [], []
    for sid in train_ids:
        if physical_scene_id(sid) in excluded:
            continue
        try:
            files = discover_scene_files(dataset_root, sid)
        except FileNotFoundError:
            missing.append(sid)
            continue
        vertices, faces = load_mesh(files.mesh)
        vertices = apply_transform(vertices, read_axis_alignment(files.metadata))
        segments = np.asarray(json.loads(files.segments.read_text())['segIndices'], np.int64)
        if len(segments) != len(vertices):
            raise ValueError(sid + ': mesh and segment indices do not align')
        groups = json.loads(files.aggregation.read_text())['segGroups']
        instance_ids, labels = _instance_assignment(segments, groups)
        face_ids = instance_ids[faces]
        valid = np.all(face_ids == face_ids[:, :1], axis=1) & (face_ids[:, 0] >= 0)
        areas = triangle_areas(vertices, faces)
        for instance_id in sorted(set(face_ids[valid, 0].tolist())):
            category = taxonomy.map_label('scannet200', labels[instance_id])
            if category is None:
                continue
            selected = valid & (face_ids[:, 0] == instance_id)
            area = float(areas[selected].sum())
            if area <= 0:
                continue
            n = min(200000, max(30, math.ceil(4 * area / .02**2)))
            sampled = sample_faces(vertices, faces[selected], areas[selected], n, _stable_seed(sid, instance_id))
            points = voxelize(sampled, .02)
            if len(points) < 3:
                continue
            extents, _, _ = pca_obb(points)
            diagonal = float(np.linalg.norm(extents))
            if not np.isfinite(diagonal) or diagonal <= 0:
                continue
            rows.append(dict(dataset='scannet200', split='train', scene_id=sid,
                physical_scene_id=physical_scene_id(sid), canonical_class=category,
                instance_id=int(instance_id), units='meters', quality_valid=True,
                bbox_diag_m=diagonal, surface_area_m2=area))
        included.append(sid)
        print('size statistics', sid, 'rows', len(rows), flush=True)
    if not rows:
        raise ValueError('No usable official-training instances; do not substitute evaluation scenes')
    output = Path(output)
    table = output.with_suffix('.instances.jsonl')
    write_rows(table, rows)
    priors = fit_priors(rows, taxonomy, table)
    priors['provenance'].update(training_list=str(Path(train_list).resolve()),
        included_scene_ids=included, excluded_physical_scenes=sorted(excluded),
        missing_training_scene_ids=missing, extractor='surface-sampled-2cm-voxel-PCA-size-only')
    priors['content_sha256'] = hash_json({k: v for k, v in priors.items() if k != 'content_sha256'})
    write_priors(output, priors)
    return priors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--train-list', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exclude-scene', action='append', default=['scene0025_01', 'scene0645_00'])
    args = parser.parse_args()
    result = fit_available_sizes(args.dataset_root, args.train_list, args.output, args.exclude_scene)
    print('active categories:', [k for k, v in result['categories'].items() if v['active']], flush=True)


if __name__ == '__main__':
    main()
