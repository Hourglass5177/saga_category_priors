from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from category_priors.evaluation_strata import load_evaluation_strata
from category_priors.recheck_evaluation import (
    BranchPrediction,
    GroundTruthObject,
    aggregate_rescue_scenes,
    aggregate_reconciliation_scenes,
    audit_export_lineage,
    evaluate_recheck_manifest,
    evaluate_rescue_scene,
    evaluate_scene_reconciliation,
    match_one_to_one,
)


def _mask(size: int, *indices: int) -> np.ndarray:
    result = np.zeros(size, dtype=bool)
    result[list(indices)] = True
    return result


def _gt(
    gt_id: int,
    class_name: str,
    indices: tuple[int, ...],
    *,
    size: int = 32,
    diagonal_m: float = 0.4,
) -> GroundTruthObject:
    return GroundTruthObject(
        gt_id=gt_id,
        class_name=class_name,
        mask=_mask(size, *indices),
        bbox_diagonal_m=diagonal_m,
    )


def _prediction(
    prediction_id: int,
    class_name: str,
    indices: tuple[int, ...],
    *,
    size: int = 32,
    score: float = 0.8,
    source_candidate_id: int | None = None,
) -> BranchPrediction:
    return BranchPrediction(
        prediction_id=prediction_id,
        class_name=class_name,
        score=score,
        mask=_mask(size, *indices),
        source_candidate_id=(
            prediction_id if source_candidate_id is None else source_candidate_id
        ),
    )


def _evaluate(
    *,
    scene_id: str = "scene0000_00",
    b0: tuple[BranchPrediction, ...] = (),
    branch: tuple[BranchPrediction, ...] = (),
    ground_truth: tuple[GroundTruthObject, ...],
    threshold: float = 0.25,
):
    return evaluate_rescue_scene(
        scene_id=scene_id,
        b0_predictions=b0,
        branch_predictions=branch,
        ground_truth=ground_truth,
        strata=load_evaluation_strata(),
        iou_threshold=threshold,
    )


def test_one_to_one_matching_maximizes_cardinality_before_total_iou() -> None:
    # P1 has the single strongest edge to G1.  Greedily taking it would leave
    # P2 unmatched.  The required solution instead uses P2-G1 and P1-G2 so
    # that two objects, rather than one, are matched.
    ground_truth = (
        _gt(1, "chair", (0, 1, 2, 3)),
        _gt(2, "chair", (4, 5, 6, 7)),
    )
    predictions = (
        _prediction(10, "chair", (0, 1, 2, 3, 4, 5, 6)),
        _prediction(11, "chair", (0, 1, 8, 9)),
    )

    result = match_one_to_one(predictions, ground_truth, iou_threshold=0.25)

    assert {(row.prediction_id, row.gt_id) for row in result.matches} == {
        (10, 2),
        (11, 1),
    }
    assert result.unmatched_prediction_ids == ()
    assert result.unmatched_gt_ids == ()


def test_one_to_one_matching_maximizes_total_iou_after_cardinality() -> None:
    ground_truth = (_gt(1, "chair", (0, 1, 2, 3)),)
    predictions = (
        _prediction(10, "chair", (0, 1, 2, 3)),
        _prediction(11, "chair", (0, 1, 2, 4)),
    )

    result = match_one_to_one(predictions, ground_truth, iou_threshold=0.25)

    assert len(result.matches) == 1
    assert result.matches[0].prediction_id == 10
    assert result.matches[0].gt_id == 1
    assert result.matches[0].iou == pytest.approx(1.0)
    assert result.unmatched_prediction_ids == (11,)


def test_matching_uses_strictly_greater_than_iou_threshold() -> None:
    ground_truth = (_gt(1, "chair", (0, 1)),)
    predictions = (_prediction(10, "chair", (0, 1, 2, 3)),)

    at_threshold = match_one_to_one(predictions, ground_truth, iou_threshold=0.50)
    below_threshold = match_one_to_one(predictions, ground_truth, iou_threshold=0.499)

    assert at_threshold.matches == ()
    assert at_threshold.unmatched_prediction_ids == (10,)
    assert at_threshold.unmatched_gt_ids == (1,)
    assert [(row.prediction_id, row.gt_id) for row in below_threshold.matches] == [
        (10, 1)
    ]


def test_duplicate_branch_predictions_can_rescue_only_one_gt() -> None:
    ground_truth = (_gt(1, "chair", (0, 1, 2, 3)),)
    branch = (
        _prediction(10, "chair", (0, 1, 2, 3), score=0.9),
        _prediction(11, "chair", (0, 1, 2, 3), score=0.8),
    )

    result = _evaluate(ground_truth=ground_truth, branch=branch)
    overall = result["strata"]["overall"]

    assert (overall["tp"], overall["fp"], overall["fn"]) == (1, 1, 0)
    assert overall["precision"] == pytest.approx(0.5)
    assert overall["recall"] == pytest.approx(1.0)
    assert overall["f1"] == pytest.approx(2.0 / 3.0)


def test_branch_prediction_of_b0_hit_is_classified_as_duplicate_b0() -> None:
    ground_truth = (
        _gt(1, "chair", (0, 1, 2, 3)),
        _gt(2, "chair", (4, 5, 6, 7)),
    )
    b0 = (_prediction(1, "chair", (0, 1, 2, 3)),)
    branch = (_prediction(10, "chair", (0, 1, 2, 3)),)

    result = _evaluate(ground_truth=ground_truth, b0=b0, branch=branch)
    overall = result["strata"]["overall"]

    assert (overall["tp"], overall["fp"], overall["fn"]) == (0, 1, 1)
    assert overall["duplicate_b0"] == 1
    assert overall["ignored"] == 0


def test_small_stratum_ignores_correct_hit_on_medium_or_large_object() -> None:
    ground_truth = (
        _gt(1, "chair", (0, 1, 2, 3), diagonal_m=0.4),
        _gt(2, "chair", (4, 5, 6, 7), diagonal_m=1.4),
    )
    branch = (_prediction(10, "chair", (4, 5, 6, 7)),)

    result = _evaluate(ground_truth=ground_truth, branch=branch)

    assert (
        result["strata"]["overall"]["tp"],
        result["strata"]["overall"]["fn"],
    ) == (1, 1)
    small = result["strata"]["small"]
    assert (small["tp"], small["fp"], small["fn"], small["ignored"]) == (
        0,
        0,
        1,
        1,
    )


def test_small_stratum_ignores_only_one_duplicate_large_prediction() -> None:
    ground_truth = (
        _gt(1, "chair", (0, 1, 2, 3), diagonal_m=0.4),
        _gt(2, "chair", (4, 5, 6, 7), diagonal_m=1.4),
    )
    branch = (
        _prediction(10, "chair", (4, 5, 6, 7)),
        _prediction(11, "chair", (4, 5, 6, 7)),
    )

    small = _evaluate(ground_truth=ground_truth, branch=branch)["strata"]["small"]

    assert (small["tp"], small["fp"], small["fn"], small["ignored"]) == (
        0,
        1,
        1,
        1,
    )
    assert small["other_fp"] == 1


def test_zero_denominators_are_reported_as_undefined() -> None:
    result = _evaluate(ground_truth=(), branch=())
    overall = result["strata"]["overall"]

    assert overall["precision"] is None
    assert overall["recall"] is None
    assert overall["f1"] is None
    assert overall["f0_5"] is None


def test_tail_and_small_tail_strata_use_class_and_instance_definitions() -> None:
    ground_truth = (
        _gt(1, "socket", (0, 1, 2, 3), diagonal_m=0.3),
        _gt(2, "socket", (4, 5, 6, 7), diagonal_m=1.2),
        _gt(3, "chair", (8, 9, 10, 11), diagonal_m=0.3),
        _gt(4, "chair", (12, 13, 14, 15), diagonal_m=1.2),
    )
    branch = (
        _prediction(10, "socket", (0, 1, 2, 3)),
        _prediction(11, "socket", (4, 5, 6, 7)),
        _prediction(12, "chair", (8, 9, 10, 11)),
    )

    result = _evaluate(ground_truth=ground_truth, branch=branch)

    assert (
        result["strata"]["overall"]["tp"],
        result["strata"]["overall"]["fn"],
    ) == (3, 1)
    assert (
        result["strata"]["small"]["tp"],
        result["strata"]["small"]["fp"],
        result["strata"]["small"]["fn"],
    ) == (2, 0, 0)
    assert (
        result["strata"]["tail"]["tp"],
        result["strata"]["tail"]["fp"],
        result["strata"]["tail"]["fn"],
    ) == (2, 0, 0)
    assert (
        result["strata"]["small_tail"]["tp"],
        result["strata"]["small_tail"]["fp"],
        result["strata"]["small_tail"]["fn"],
    ) == (1, 0, 0)
    assert result["strata"]["small_tail"]["ignored"] == 2


def test_scene_aggregation_reports_pooled_counts_and_equal_scene_means() -> None:
    scene_one = _evaluate(
        scene_id="scene0001_00",
        ground_truth=(
            _gt(1, "chair", (0, 1, 2, 3)),
            _gt(2, "chair", (4, 5, 6, 7)),
        ),
        branch=(_prediction(10, "chair", (0, 1, 2, 3)),),
    )
    scene_two = _evaluate(
        scene_id="scene0002_00",
        ground_truth=(_gt(1, "chair", (0, 1, 2, 3)),),
        branch=(
            _prediction(20, "chair", (0, 1, 2, 3), score=0.9),
            _prediction(21, "chair", (0, 1, 2, 3), score=0.8),
        ),
    )

    aggregate = aggregate_rescue_scenes((scene_one, scene_two))
    overall = aggregate["strata"]["overall"]

    assert overall["pooled_counts"]["tp"] == 2
    assert overall["pooled_counts"]["fp"] == 1
    assert overall["pooled_counts"]["fn"] == 1
    assert overall["pooled_metrics"]["precision"] == pytest.approx(2.0 / 3.0)
    assert overall["pooled_metrics"]["recall"] == pytest.approx(2.0 / 3.0)
    assert overall["pooled_metrics"]["f1"] == pytest.approx(2.0 / 3.0)
    assert overall["scene_equal_mean"]["precision"] == pytest.approx(0.75)
    assert overall["scene_equal_mean"]["recall"] == pytest.approx(0.75)
    assert overall["scene_equal_mean"]["f1"] == pytest.approx(2.0 / 3.0)


def test_split_children_share_parent_and_are_both_evaluated() -> None:
    result = _evaluate(
        ground_truth=(_gt(1, "chair", (0, 1)), _gt(2, "chair", (2, 3))),
        branch=(
            _prediction(10, "chair", (0, 1), source_candidate_id=77),
            _prediction(11, "chair", (2, 3), source_candidate_id=77),
        ),
    )
    assert result["strata"]["overall"]["tp"] == 2


def _export(instance_id: int):
    from category_priors.evaluator import PredictedInstance
    return PredictedInstance("scene", instance_id, 0, 0.8, _mask(32, instance_id))


def test_historical_full_lineage_recovers_split_and_merged_parents() -> None:
    rows, audit = audit_export_lineage(
        (_export(0), _export(1), _export(2)),
        {"candidate_export_ids": {"77": 1}, "candidate_export_lineage": {"0": [77, 88], "1": [77]}},
        ("chair",),
    )
    assert audit["complete"]
    assert audit["traceable_prediction_count"] == 2
    assert [(row.prediction_id, row.source_candidate_ids) for row in rows] == [(0, (77, 88)), (1, (77,))]


def test_v2_lineage_requires_exact_inventory_and_inverse() -> None:
    payload = {
        "candidate_export_contract_schema": "saga-candidate-export-lineage-v2",
        "candidate_export_ids": {"77": [0, 1], "88": [0]},
        "candidate_export_lineage": {"0": [77, 88], "1": [77]},
        "refined_export_ids": [0, 1],
    }
    predictions = (_export(0), _export(1))
    assert audit_export_lineage(predictions, payload, ("chair",))[1]["complete"]
    payload["refined_export_ids"] = [0]
    rows, audit = audit_export_lineage(predictions, payload, ("chair",))
    assert not rows and not audit["complete"]
    assert "inventory" in audit["reason"]
    payload["refined_export_ids"] = [0, 1]
    payload["candidate_export_ids"] = {"77": [0, 1]}
    assert not audit_export_lineage(predictions, payload, ("chair",))[1]["complete"]


@pytest.mark.parametrize("payload", [
    {"candidate_export_ids": {"77": 0}},
    {"candidate_export_lineage": {"0": []}},
    {"candidate_export_lineage": {"9": [77]}},
    {"candidate_export_lineage": {"0": [77]}, "candidate_export_ids": {"88": 0}},
    {"candidate_export_lineage": {"0": [77.5]}},
    {"candidate_export_lineage": {"0": [True]}},
    {"candidate_export_lineage": {"0": [77], "00": [88]}},
])
def test_missing_partial_or_contradictory_lineage_is_explicitly_incomplete(payload) -> None:
    rows, audit = audit_export_lineage((_export(0),), payload, ("chair",))
    assert rows == ()
    assert not audit["complete"]
    assert audit["traceable_prediction_count"] is None
    assert audit["reason"]


def test_matching_exact_ties_follow_content_not_instance_numbers() -> None:
    ground_truth = (_gt(1, "chair", (0, 1, 2, 3)),)
    left = _prediction(10, "chair", (0, 1, 4, 5))
    right = _prediction(11, "chair", (2, 3, 6, 7))
    first = match_one_to_one((left, right), ground_truth, 0.25)
    permuted = (replace(right, prediction_id=100), replace(left, prediction_id=999))
    second = match_one_to_one(permuted, (replace(ground_truth[0], gt_id=555),), 0.25)
    first_masks = {row.prediction_id: row.mask for row in (left, right)}
    second_masks = {row.prediction_id: row.mask for row in permuted}
    np.testing.assert_array_equal(first_masks[first.matches[0].prediction_id], second_masks[second.matches[0].prediction_id])


def _reconcile(b0=(), final=(), ground_truth=(), threshold=0.25, scene_id="scene"):
    return evaluate_scene_reconciliation(
        scene_id=scene_id, b0_predictions=b0, final_predictions=final,
        ground_truth=ground_truth, strata=load_evaluation_strata(),
        iou_threshold=threshold,
    )


def test_reconciliation_distinguishes_replacement_duplicate_rescue_and_inherited_fp() -> None:
    ground_truth = (
        _gt(1, "chair", (0, 1, 2, 3)),
        _gt(2, "cup", (4, 5, 6, 7)),
        _gt(3, "chair", (12, 13, 14, 15)),
    )
    b0 = (
        _prediction(0, "chair", (0, 1, 2)),
        _prediction(1, "chair", (8, 9)),
        _prediction(2, "chair", (12, 13, 14, 15)),
    )
    final = (
        _prediction(10, "chair", (0, 1, 2, 3)),  # correct replacement
        _prediction(11, "chair", (0, 1)),  # duplicate
        _prediction(12, "cup", (4, 5, 6, 7)),  # rescue
        _prediction(13, "chair", (8, 9)),  # original FP, renumbered
        _prediction(14, "chair", (20, 21)),  # new FP
    )
    result = _reconcile(b0, final, ground_truth)
    row = result["strata"]["overall"]
    assert (row["rescued"], row["retained"], row["lost"]) == (1, 1, 1)
    assert (row["new_false_positives"], row["inherited_false_positives"], row["duplicate_predictions"]) == (2, 1, 1)
    assert row["final_false_positives"] == 3
    assert row["precision"] == pytest.approx(1 / 3)
    assert row["f0_5"] == pytest.approx(1.25 / 3.25)
    assert row["below_ap_min_region_predictions"] == 5


def test_reconciliation_preserves_all_new_fp_in_small_but_filters_tail_by_class() -> None:
    ground_truth = (_gt(1, "cup", (0, 1), diagonal_m=0.2), _gt(2, "chair", (2, 3), diagonal_m=1.5))
    final = (_prediction(10, "cup", (0, 1)), _prediction(11, "chair", (2, 3)), _prediction(12, "chair", (8, 9)))
    result = _reconcile(final=final, ground_truth=ground_truth)
    small, tail = result["strata"]["small"], result["strata"]["tail"]
    assert (small["tp"], small["fp"]) == (1, 1)
    assert (tail["tp"], tail["fp"], tail["all_new_false_positives"]) == (1, 0, 1)


def test_unchanged_identity_uses_gaussian_membership_not_lossy_gt_mask() -> None:
    b0 = replace(_prediction(0, "chair", (5,)), member_sha256="original-gaussian-members")
    final = replace(_prediction(9, "chair", (5,)), member_sha256="different-gaussian-members")
    result = _reconcile((b0,), (final,))
    assert result["strata"]["overall"]["new_false_positives"] == 1
    assert result["strata"]["overall"]["inherited_false_positives"] == 0


def test_scene_reconciliation_counts_are_invariant_to_all_export_renumbering() -> None:
    b0 = (_prediction(0, "chair", (0, 1)), _prediction(1, "cup", (10, 11)))
    final = (_prediction(2, "chair", (0, 1, 2)), _prediction(3, "cup", (10, 11)), _prediction(4, "chair", (5, 6)))
    gt = (_gt(1, "chair", (0, 1, 2)), _gt(2, "chair", (5, 6)))
    original = _reconcile(b0, final, gt)
    renamed = _reconcile(
        tuple(replace(row, prediction_id=100 - index) for index, row in enumerate(reversed(b0))),
        tuple(replace(row, prediction_id=900 - index) for index, row in enumerate(reversed(final))),
        tuple(replace(row, gt_id=500 - index) for index, row in enumerate(reversed(gt))),
    )
    assert original["strata"] == renamed["strata"]


def test_unchanged_b0_displaced_by_better_prediction_remains_visible_in_total_fp() -> None:
    b0 = _prediction(0, "chair", (0, 1))
    final = (replace(b0, prediction_id=5), _prediction(6, "chair", (0, 1, 2, 3)))
    result = _reconcile((b0,), final, (_gt(1, "chair", (0, 1, 2, 3)),))
    row = result["strata"]["overall"]
    assert row["retained"] == 1
    assert row["new_false_positives"] == 0
    assert row["unchanged_b0_displaced_predictions"] == row["duplicate_predictions"] == row["final_false_positives"] == 1


def test_reconciliation_strict_threshold_null_metrics_and_aggregation() -> None:
    empty = _reconcile(scene_id="empty")
    assert all(empty["strata"]["overall"][key] is None for key in ("precision", "recall", "f1", "f0_5"))
    scene = _reconcile(final=(_prediction(10, "chair", (0, 1, 2, 3)),), ground_truth=(_gt(1, "chair", (0, 1)),), threshold=0.5)
    assert scene["strata"]["overall"]["rescued"] == 0
    empty["iou_threshold"] = 0.5
    pooled = aggregate_reconciliation_scenes((empty, scene))["strata"]["overall"]
    assert pooled["pooled_counts"]["fp"] == 1
    assert pooled["defined_scene_count"]["precision"] == 1


@pytest.mark.parametrize("complete_lineage", [True, False])
def test_manifest_keeps_ap_and_original_artifacts_while_reporting_lineage(monkeypatch, tmp_path, complete_lineage) -> None:
    from category_priors import recheck_evaluation as module
    from category_priors.evaluator import PredictedInstance, evaluate_instances, GroundTruthScene, SCANNET_OFFICIAL_OVERLAPS
    from category_priors.taxonomy import load_taxonomy
    from category_priors.io import sha256_file

    taxonomy = load_taxonomy()
    chair = taxonomy.canonical_classes.index("chair")
    cup = taxonomy.canonical_classes.index("cup")
    semantic = np.array([chair] * 100 + [cup] * 100)
    instance = np.array([1] * 100 + [2] * 100)
    np.savez(tmp_path / "gt.npz", coords=np.zeros((200, 3)), semantic=semantic, instance=instance)
    (tmp_path / "gaussians.ply").write_text("test geometry supplied by test adapter", encoding="utf-8")
    b0 = {"point_labels": [0] * 100 + [-1] * 100, "instances": {"0": {"class": "chair", "score": 0.8}}}
    final = {"point_labels": [0] * 100 + [1] * 100, "instances": {"0": {"class": "chair", "score": 0.8}, "1": {"class": "cup", "score": 0.9}}, "candidate_export_ids": {"77": 1}}
    if complete_lineage:
        final["candidate_export_lineage"] = {"1": [77]}
    for name, payload in (("b0.json", b0), ("final.json", final)):
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    manifest = {
        "schema": "saga-instance-recheck-evaluation-manifest-v2", "conditions": ["D-class"],
        "scenes": [{"scene_id": "scene", "gt_npz": "gt.npz", "gaussian_ply": "gaussians.ply", "b0_output_json": "b0.json", "gaussian_to_gt_transform": np.eye(4).tolist(), "condition_outputs": {"D-class": "final.json"}}],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    original_hashes = {path: sha256_file(path) for path in tmp_path.iterdir()}

    def supplied_predictions(**kwargs):
        payload = json.loads(kwargs["output_json"].read_text(encoding="utf-8"))
        labels = np.asarray(payload["point_labels"])
        return [PredictedInstance("scene", int(key), taxonomy.canonical_classes.index(value["class"]), value["score"], labels == int(key)) for key, value in payload["instances"].items()], {"mapped_fraction": 1.0}

    monkeypatch.setattr(module, "saga_scene_predictions", supplied_predictions)
    result = evaluate_recheck_manifest(manifest_path, output_path=tmp_path / "evaluation.json", taxonomy=taxonomy, strata=load_evaluation_strata())
    condition = result["conditions"]["D-class"]
    assert (condition["rescue"] is not None) == complete_lineage
    assert condition["rescue_status"] == ("complete" if complete_lineage else "incomplete")
    assert condition["reconciliation"]["iou_0.50"]["aggregate"]["strata"]["overall"]["pooled_counts"]["rescued"] == 1
    expected_predictions, _ = supplied_predictions(output_json=tmp_path / "final.json")
    expected = evaluate_instances([GroundTruthScene("scene", semantic, instance)], expected_predictions, taxonomy.canonical_classes, overlaps=SCANNET_OFFICIAL_OVERLAPS)
    assert condition["official_9"] == expected
    assert all(sha256_file(path) == digest for path, digest in original_hashes.items())
    with pytest.raises(ValueError, match="overwrite"):
        evaluate_recheck_manifest(manifest_path, output_path=tmp_path / "final.json", taxonomy=taxonomy, strata=load_evaluation_strata())
