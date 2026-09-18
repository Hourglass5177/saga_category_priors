"""CPU evaluation of complete final scene exports against the unchanged B0.

Usage: python run_effective_repair_evaluation.py --manifest inputs.json --output-dir results

The manifest uses the existing evaluation fields: conditions; scenes containing
scene_id, gt_npz, gaussian_ply, gaussian_to_gt_transform, b0_output_json, and
condition_outputs mapping each condition to a path or {"output_json": path}.
Paths are relative to the manifest. Predictions are standard point_labels and
instances dictionaries; only instances' own class/score are evaluated.

This reuses the original coordinate mapper, AP evaluator, strict one-to-one
matching and frozen small/tail strata. A scene's coordinate mapping is computed
once because every condition shares the same frozen Gaussian order. Manual
C4/supplementary/D12 records never enter the scene-level denominator.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any
from types import SimpleNamespace

import numpy as np

from category_priors.evaluation_strata import load_evaluation_strata
from category_priors.evaluator import (
    SCANNET_OFFICIAL_OVERLAPS,
    PredictedInstance,
    apply_transform,
    evaluate_instances,
    load_ground_truth_npz,
    load_ply_xyz,
    map_gaussians_to_gt,
)
from category_priors.instance_projection import project_declared_instances
from category_priors.io import write_json
from category_priors.recheck_evaluation import (
    _prediction_rows,
    aggregate_reconciliation_scenes,
    evaluate_scene_reconciliation,
    ground_truth_objects,
)
from category_priors.taxonomy import load_taxonomy


THRESHOLDS = (0.25, 0.50)
STRATA = ("overall", "small", "tail", "small_tail")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def predictions_from_mapping(scene_id, path, gt_to_gaussian, gaussian_count, taxonomy):
    """Original saga_scene_predictions semantics with a reused coordinate map."""
    payload = read_json(path)
    projection = project_declared_instances(payload["point_labels"], payload.get("instances", {}))
    labels = projection.point_labels
    if labels.shape != (gaussian_count,):
        raise ValueError(f"{scene_id}: {path} changes the frozen Gaussian domain")
    mapped = np.full(len(gt_to_gaussian), -1, dtype=np.int64)
    valid = gt_to_gaussian >= 0
    mapped[valid] = labels[gt_to_gaussian[valid]]
    classes = {name: i for i, name in enumerate(taxonomy.canonical_classes)}
    predictions = []
    for key, properties in payload.get("instances", {}).items():
        identifier = int(key)
        if identifier < 0:
            continue
        category = str(properties.get("class", "")).strip().lower()
        # Do not silently drop exported unknown/unsupported classes: finalize
        # them as abstentions before evaluation, as in the original manifest.
        if category not in classes:
            raise ValueError(f"{scene_id}: export {identifier} has unsupported class {category!r}")
        score = float(properties["score"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"{scene_id}: export {identifier} has invalid AP score {score}")
        predictions.append(PredictedInstance(
            scene_id, identifier, classes[category], score, mapped == identifier,
        ))
    rows = _prediction_rows(predictions, taxonomy.canonical_classes, gaussian_labels=labels)
    counts = dict(zip(*np.unique(labels[labels >= 0], return_counts=True)))
    diagnostics = projection.numeric_stats()
    diagnostics["gt_nearest_declared_fraction"] = float(np.mean(mapped >= 0)) if len(mapped) else 0.0
    diagnostics["output_json"] = str(path.resolve())
    return predictions, rows, {int(k): int(v) for k, v in counts.items()}, diagnostics


def overlap_details(predictions, objects):
    """Posthoc best overlaps explain misses; they never choose model outputs."""
    matrix = np.zeros((len(predictions), len(objects)), dtype=np.float64)
    for p, prediction in enumerate(predictions):
        for g, obj in enumerate(objects):
            union = np.count_nonzero(prediction.mask | obj.mask)
            matrix[p, g] = np.count_nonzero(prediction.mask & obj.mask) / union if union else 0.0
    return matrix


def geometry_rows_from_export(path, gt_to_gaussian, gaussian_count):
    """Posthoc geometry includes unknown classes; official AP still uses the export."""
    payload = read_json(path)
    if 'geometry_source' not in payload:
        return None
    geometry = read_json(resolve(Path(path).parent, payload['geometry_source']))
    projection = project_declared_instances(geometry['point_labels'], geometry['instances'])
    labels = projection.point_labels
    if labels.shape != (gaussian_count,):
        raise ValueError('geometry changes the frozen Gaussian domain')
    mapped = np.full(len(gt_to_gaussian), -1, np.int64)
    valid = gt_to_gaussian >= 0
    mapped[valid] = labels[gt_to_gaussian[valid]]
    return [SimpleNamespace(prediction_id=int(k), class_name=m['class'], score=m['score'],
                            mask=mapped == int(k)) for k,m in geometry['instances'].items()]


def object_iou_assignment(overlaps, predictions, objects, *, class_aware=True):
    """Maximum total IoU, one-to-one, with a fixed GT denominator and zeros for misses."""
    from scipy.optimize import linear_sum_assignment
    scores = np.asarray(overlaps, dtype=np.float64).copy()
    if class_aware:
        for p, prediction in enumerate(predictions):
            for g, obj in enumerate(objects):
                if prediction.class_name != obj.class_name:
                    scores[p, g] = 0.
    assigned = {}
    if scores.size:
        pp, gg = linear_sum_assignment(-scores)
        assigned = {int(g): (int(p), float(scores[p, g])) for p, g in zip(pp, gg) if scores[p, g] > 0}
    return [dict(gt_id=obj.gt_id, **{'class': obj.class_name},
                 prediction_id=predictions[assigned[g][0]].prediction_id if g in assigned else None,
                 iou=assigned[g][1] if g in assigned else 0.) for g, obj in enumerate(objects)]


def outcome_rows(condition, scene, final_rows, gaussian_counts, reconciliation, overlaps, strata, min_region_size):
    objects = scene["objects"]
    b0_matches = {row["gt_id"]: row for row in reconciliation["b0_matches"]}
    final_matches = {row["gt_id"]: row for row in reconciliation["final_matches"]}
    gt_rows = []
    for g, obj in enumerate(objects):
        before, after = b0_matches.get(obj.gt_id), final_matches.get(obj.gt_id)
        state = "retained" if before and after else "lost" if before else "rescued" if after else "still_missed"
        compatible = [p for p, prediction in enumerate(final_rows) if prediction.class_name == obj.class_name]
        best_same = max(compatible, key=lambda p: overlaps[p, g]) if compatible else None
        best_any = int(np.argmax(overlaps[:, g])) if len(final_rows) else None
        gt_rows.append({
            "condition": condition, "scene_id": scene["scene_id"], "iou_threshold": reconciliation["iou_threshold"],
            "gt_id": obj.gt_id, "class": obj.class_name, "gt_points": int(np.count_nonzero(obj.mask)),
            "bbox_diagonal_m": obj.bbox_diagonal_m,
            "small": strata.is_small(obj.bbox_diagonal_m), "tail": strata.is_tail(obj.class_name),
            "small_tail": strata.is_small(obj.bbox_diagonal_m) and strata.is_tail(obj.class_name),
            "outcome": state,
            "b0_export_id": before["prediction_id"] if before else None,
            "b0_match_iou": before["iou"] if before else None,
            "final_export_id": after["prediction_id"] if after else None,
            "final_match_iou": after["iou"] if after else None,
            "best_same_class_export_id": final_rows[best_same].prediction_id if best_same is not None else None,
            "best_same_class_iou": float(overlaps[best_same, g]) if best_same is not None else 0.0,
            "best_any_class_export_id": final_rows[best_any].prediction_id if best_any is not None else None,
            "best_any_class": final_rows[best_any].class_name if best_any is not None else None,
            "best_any_class_iou": float(overlaps[best_any, g]) if best_any is not None else 0.0,
        })
    matched = {row["prediction_id"]: row for row in reconciliation["final_matches"]}
    new_fp = set(reconciliation["new_false_positive_ids"])
    inherited = set(reconciliation["inherited_false_positive_ids"])
    displaced = set(reconciliation["unchanged_b0_displaced_prediction_ids"])
    duplicates = set(reconciliation["duplicate_prediction_ids"])
    unchanged = {row["final_export_id"]: row["b0_export_id"] for row in reconciliation["unchanged_prediction_pairs"]}
    prediction_rows = []
    for p, prediction in enumerate(final_rows):
        identifier = prediction.prediction_id
        match = matched.get(identifier)
        state = ("matched" if match else "new_false_positive" if identifier in new_fp
                 else "inherited_false_positive" if identifier in inherited
                 else "unchanged_b0_displaced" if identifier in displaced else "unmatched")
        best = int(np.argmax(overlaps[p])) if len(objects) else None
        prediction_rows.append({
            "condition": condition, "scene_id": scene["scene_id"], "iou_threshold": reconciliation["iou_threshold"],
            "export_id": identifier, "class": prediction.class_name, "score": prediction.score,
            "gaussian_count": gaussian_counts.get(identifier, 0),
            "mapped_gt_point_count": int(np.count_nonzero(prediction.mask)),
            "below_ap_min_region": int(np.count_nonzero(prediction.mask)) < min_region_size,
            "outcome": state, "duplicate": identifier in duplicates,
            "new_duplicate": identifier in duplicates and identifier in new_fp,
            "unchanged_b0_export_id": unchanged.get(identifier),
            "matched_gt_id": match["gt_id"] if match else None,
            "matched_iou": match["iou"] if match else None,
            "best_gt_id": objects[best].gt_id if best is not None else None,
            "best_gt_class": objects[best].class_name if best is not None else None,
            "best_gt_iou": float(overlaps[p, best]) if best is not None else 0.0,
        })
    return gt_rows, prediction_rows


def write_csv(path, rows, *, encoding='utf-8-sig'):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(manifest_path, output_dir, taxonomy, strata):
    manifest_path, output_dir = Path(manifest_path), Path(output_dir)
    manifest = read_json(manifest_path)
    base = manifest_path.parent
    conditions = [str(value) for value in manifest["conditions"]]
    if not conditions or len(set(conditions)) != len(conditions) or "B0" in conditions:
        raise ValueError("conditions must be unique non-B0 names")
    scene_specs = manifest["scenes"]
    scene_ids = [str(row["scene_id"]) for row in scene_specs]
    if not scene_specs or len(set(scene_ids)) != len(scene_ids):
        raise ValueError("manifest must contain unique scenes")
    radius, minimum = 0.05, 100
    for key, expected in (("radius_m", radius), ("min_region_size", minimum)):
        if key in manifest and manifest[key] != expected:
            raise ValueError(f"{key} differs from the frozen original evaluation ({expected})")
    scenes = []
    input_paths = {manifest_path.resolve()}
    for spec in scene_specs:
        scene_id = str(spec["scene_id"])
        if set(spec["condition_outputs"]) != set(conditions):
            raise ValueError(f"{scene_id}: final output conditions differ from the manifest")
        paths = {key: resolve(base, spec[key]) for key in ("gt_npz", "gaussian_ply", "b0_output_json")}
        gt_xyz, gt = load_ground_truth_npz(paths["gt_npz"], scene_id)
        xyz = apply_transform(load_ply_xyz(paths["gaussian_ply"]), spec["gaussian_to_gt_transform"])
        # Passing arange through the original mapper returns exactly the same
        # nearest-Gaussian ownership used by saga_scene_predictions.
        mapping, diagnostics = map_gaussians_to_gt(gt_xyz, xyz, np.arange(len(xyz)), radius)
        if diagnostics["mapped_fraction"] < float(manifest.get("minimum_mapped_fraction", 0.90)):
            raise ValueError(f"{scene_id}: supplied transform does not align GT and Gaussians")
        b0, b0_rows, b0_counts, b0_diagnostics = predictions_from_mapping(
            scene_id, paths["b0_output_json"], mapping, len(xyz), taxonomy,
        )
        outputs = {}
        for name, value in spec["condition_outputs"].items():
            outputs[name] = resolve(base, value["output_json"] if isinstance(value, dict) else value)
        input_paths.update(path.resolve() for path in (*paths.values(), *outputs.values()))
        scenes.append({
            "scene_id": scene_id, "gt": gt,
            "objects": ground_truth_objects(gt, gt_xyz, taxonomy.canonical_classes, minimum),
            "mapping": mapping, "gaussian_count": len(xyz), "mapping_diagnostics": diagnostics,
            "transform": spec["gaussian_to_gt_transform"], "paths": paths,
            "outputs": outputs, "b0": b0, "b0_rows": b0_rows,
            "b0_counts": b0_counts, "b0_diagnostics": b0_diagnostics,
        })
        print(f"Loaded {scene_id}: {len(scenes[-1]['objects'])} GT objects, {len(b0)} B0 exports", flush=True)
    result_files = [output_dir / name for name in ("evaluation.json", "metrics.csv", "gt_outcomes.csv", "prediction_outcomes.csv", "object_iou.csv")]
    if any(path.resolve() in input_paths for path in result_files):
        raise ValueError("evaluation outputs must not overwrite inputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "effective-repair-scene-evaluation-v1", "complete": False,
        "scene_ids": scene_ids, "condition_order": conditions, "b0": {}, "conditions": {},
        "strata": {"small_diagonal_threshold_m": strata.small_diagonal_threshold_m,
                   "tail_classes": list(strata.tail_classes)},
        "protocol": {
            "radius_m": radius, "min_region_size": minimum, "matching_thresholds": list(THRESHOLDS),
            "matching": "same-class maximum-cardinality then total-IoU, strict > threshold",
            "object_mean_iou_matching": "maximum total IoU one-to-one; unmatched GT zero; class-aware and class-agnostic reported separately",
            "ap": "original ScanNet official 9 thresholds .50 through .90, AP25 separate",
            "ap_scope": "all evaluation classes; subgroup results are paired object counts, not a new subgroup AP protocol",
            "denominator": "unique scene and GT instance; all final exports; manual diagnostic cohorts not added",
            "new_fp": "unmatched final exports excluding exact unchanged B0 class+Gaussian membership, one-to-one",
            "small_fp": "all-class FP; small is a GT property; tail and small_tail use tail-class FP alongside all-FP",
            "evidence_scope": "two development scenes; descriptive comparison, not independent generalization evidence",
        },
        "inputs": {"manifest": str(manifest_path.resolve()), "taxonomy_sha256": taxonomy.content_hash,
                   "scenes": [{"scene_id": scene["scene_id"],
                               **{key: str(value.resolve()) for key, value in scene["paths"].items()},
                               "gaussian_to_gt_transform": scene["transform"],
                               "mapping": scene["mapping_diagnostics"]} for scene in scenes]},
    }
    all_gt_rows, all_prediction_rows, metric_rows, object_iou_rows = [], [], [], []
    for condition in ("B0", *conditions):
        all_predictions, per_scene, reconciliations = [], {}, []
        for scene in scenes:
            if condition == "B0":
                predictions, rows, counts, diagnostics = scene["b0"], scene["b0_rows"], scene["b0_counts"], scene["b0_diagnostics"]
            else:
                predictions, rows, counts, diagnostics = predictions_from_mapping(
                    scene["scene_id"], scene["outputs"][condition], scene["mapping"], scene["gaussian_count"], taxonomy,
                )
            all_predictions.extend(predictions)
            official = evaluate_instances([scene["gt"]], predictions, taxonomy.canonical_classes,
                                          overlaps=SCANNET_OFFICIAL_OVERLAPS, min_region_size=minimum)
            per_scene[scene["scene_id"]] = {"official_9": official, "diagnostics": diagnostics}
            overlaps = overlap_details(rows, scene["objects"])
            for aware in (True, False):
                geometry_rows = (geometry_rows_from_export(scene['outputs'][condition],
                    scene['mapping'], scene['gaussian_count']) if not aware and condition != 'B0' else None)
                scoring_rows = rows if geometry_rows is None else geometry_rows
                scoring_overlaps = overlaps if geometry_rows is None else overlap_details(geometry_rows, scene['objects'])
                if geometry_rows is not None:
                    diagnostics['geometry_instance_count'] = len(geometry_rows)
                    diagnostics['geometry_out_of_eval_vocabulary_count'] = sum(
                        r.class_name not in taxonomy.canonical_classes for r in geometry_rows)
                    diagnostics['geometry_unknown_count'] = sum(r.class_name == 'unknown' for r in geometry_rows)
                assignments = object_iou_assignment(scoring_overlaps, scoring_rows, scene['objects'], class_aware=aware)
                object_iou_rows.extend(dict(condition=condition, scene_id=scene['scene_id'],
                    matching='class_aware' if aware else 'class_agnostic', **item) for item in assignments)
            for threshold in THRESHOLDS:
                reconciliation = evaluate_scene_reconciliation(
                    scene_id=scene["scene_id"], b0_predictions=scene["b0_rows"], final_predictions=rows,
                    ground_truth=scene["objects"], strata=strata, iou_threshold=threshold, min_region_size=minimum,
                )
                reconciliations.append(reconciliation)
                gt_rows, prediction_rows = outcome_rows(condition, scene, rows, counts, reconciliation, overlaps, strata, minimum)
                all_gt_rows.extend(gt_rows)
                all_prediction_rows.extend(prediction_rows)
        entry = {
            "object_mean_iou": {kind: float(np.mean([r['iou'] for r in object_iou_rows
                if r['condition'] == condition and r['matching'] == kind]))
                for kind in ('class_aware', 'class_agnostic')},
            "official_9": evaluate_instances([scene["gt"] for scene in scenes], all_predictions,
                                             taxonomy.canonical_classes, overlaps=SCANNET_OFFICIAL_OVERLAPS, min_region_size=minimum),
            "per_scene": per_scene,
            "reconciliation": {
                f"iou_{threshold:.2f}": {
                    "aggregate": aggregate_reconciliation_scenes([r for r in reconciliations if r["iou_threshold"] == threshold]),
                    "per_scene": [r for r in reconciliations if r["iou_threshold"] == threshold],
                } for threshold in THRESHOLDS
            },
        }
        if condition == "B0":
            result["b0"] = entry
        else:
            result["conditions"][condition] = entry
        for threshold in THRESHOLDS:
            block = entry["reconciliation"][f"iou_{threshold:.2f}"]
            for sid, stats in [(r["scene_id"], r["strata"]) for r in block["per_scene"]] + [("pooled", block["aggregate"]["strata"])]:
                ap = entry["official_9"] if sid == "pooled" else entry["per_scene"][sid]["official_9"]
                baseline_ap = result["b0"]["official_9"] if sid == "pooled" else result["b0"]["per_scene"][sid]["official_9"]
                for name in STRATA:
                    counts = stats[name]["pooled_counts"] if sid == "pooled" else stats[name]
                    metric_rows.append({
                        "condition": condition, "scene_id": sid, "iou_threshold": threshold, "stratum": name,
                        **counts,
                        **(stats[name]["pooled_metrics"] if sid == "pooled" else {}),
                        **({"map_50_90": ap["aggregate"]["map_50_90"],
                            "ap25": ap["aggregate"]["map_0.25"], "ap50": ap["aggregate"]["map_0.50"],
                            "map_50_90_minus_b0": ap["aggregate"]["map_50_90"] - baseline_ap["aggregate"]["map_50_90"]} if name == "overall" else {}),
                    })
        print(f"{condition}: mAP={entry['official_9']['aggregate']['map_50_90']:.6f}; "
              f"rescued@.25={entry['reconciliation']['iou_0.25']['aggregate']['strata']['overall']['pooled_counts']['rescued']}", flush=True)
    result["complete"] = True
    result["gt_outcomes"] = all_gt_rows
    result["prediction_outcomes"] = all_prediction_rows
    result["files"] = {path.stem: str(path.resolve()) for path in result_files}
    write_csv(output_dir / "metrics.csv", metric_rows)
    write_csv(output_dir / "gt_outcomes.csv", all_gt_rows)
    write_csv(output_dir / "prediction_outcomes.csv", all_prediction_rows)
    write_csv(output_dir / "object_iou.csv", object_iou_rows, encoding='utf-8')
    write_json(output_dir / "evaluation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path)
    parser.add_argument("--strata", type=Path)
    args = parser.parse_args()
    evaluate(args.manifest, args.output_dir, load_taxonomy(args.taxonomy),
             load_evaluation_strata(args.strata) if args.strata else load_evaluation_strata())


if __name__ == "__main__":
    main()
