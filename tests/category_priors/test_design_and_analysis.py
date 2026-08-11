from __future__ import annotations

import numpy as np
import pytest

from category_priors.analysis import (
    CompiledEvaluation,
    compile_predictions,
    evaluate_compiled,
    factorial_bootstrap,
    holm_adjust,
    merge_compiled_evaluations,
    paired_scene_bootstrap,
    paired_scene_permutation_test,
    weighted_scene_metric,
)
from category_priors.evaluator import (
    GroundTruthScene,
    PredictedInstance,
    evaluate_instances,
)
from category_priors.io import write_json
from category_priors.mapping import (
    build_run_schedule,
    choose_best_config,
    latin_hypercube_design,
)


def test_config_selection_rejects_locked_or_test_metrics() -> None:
    design = latin_hypercube_design("global", samples=2, seed=7)
    metrics = [
        {
            "config_id": design["configurations"][0]["config_id"],
            "split": "val-locked",
            "map_50_95": 0.5,
            "runtime_seconds": 1.0,
        }
    ]
    with pytest.raises(ValueError, match="val-tune only"):
        choose_best_config(metrics, design)


def test_config_selection_requires_complete_balanced_design() -> None:
    design = latin_hypercube_design("global", samples=2, seed=7)
    one = [
        {
            "config_id": design["configurations"][0]["config_id"],
            "split": "val-tune",
            "map_50_95": 0.5,
            "runtime_seconds": 1.0,
            "scene_count": 24,
        }
    ]
    with pytest.raises(ValueError, match="incomplete"):
        choose_best_config(one, design)


def test_runtime_tie_break_uses_fractional_ap_units() -> None:
    design = latin_hypercube_design("global", samples=2, seed=7)
    rows = [
        {
            "config_id": design["configurations"][0]["config_id"],
            "split": "val-tune",
            "map_50_95": 0.500,
            "runtime_seconds": 100.0,
            "scene_count": 24,
        },
        {
            "config_id": design["configurations"][1]["config_id"],
            "split": "val-tune",
            "map_50_95": 0.499,
            "runtime_seconds": 1.0,
            "scene_count": 24,
        },
    ]
    selected = choose_best_config(rows, design)
    assert selected["config_id"] == design["configurations"][1]["config_id"]


def test_factorial_requires_all_eight_combinations() -> None:
    gt = GroundTruthScene(
        "scene", np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64)
    )
    prediction = PredictedInstance("scene", 1, 0, 1.0, np.ones(1, dtype=bool))
    bits = {
        f"condition-{index}": value
        for index, value in enumerate(
            [
                (0, 0, 0),
                (0, 0, 1),
                (0, 1, 0),
                (0, 1, 1),
                (1, 0, 0),
                (1, 0, 1),
                (1, 1, 0),
            ]
        )
    }
    predictions = {name: [prediction] for name in bits}
    with pytest.raises(ValueError, match="eight"):
        factorial_bootstrap(
            [gt],
            predictions,
            bits,
            {"scene": "physical"},
            ["chair"],
            samples=1,
            min_region_size=1,
        )


def test_holm_adjustment_is_monotone_and_bounded() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03, 1.0])
    assert all(0.0 <= value <= 1.0 for value in adjusted)
    assert adjusted[0] <= adjusted[2] <= adjusted[1] <= adjusted[3]


def _single_instance_scene(scene_id: str, class_id: int = 0) -> GroundTruthScene:
    return GroundTruthScene(
        scene_id,
        np.full(3, class_id, dtype=np.int64),
        np.ones(3, dtype=np.int64),
    )


def _perfect_prediction(scene_id: str, class_id: int = 0) -> PredictedInstance:
    return PredictedInstance(
        scene_id,
        1,
        class_id,
        0.9,
        np.ones(3, dtype=bool),
    )


def test_positive_weight_bootstrap_keeps_every_globally_evaluable_class() -> None:
    ground_truth = [
        _single_instance_scene("chair-scene", 0),
        _single_instance_scene("cup-scene", 1),
    ]
    predictions = {
        "P000": [_perfect_prediction("chair-scene", 0)],
        "P111": [
            _perfect_prediction("chair-scene", 0),
            _perfect_prediction("cup-scene", 1),
        ],
    }
    result = paired_scene_bootstrap(
        ground_truth,
        predictions,
        {"chair-scene": "chair", "cup-scene": "cup"},
        ["chair", "cup"],
        "P000",
        "P111",
        samples=50,
        seed=7,
        min_region_size=1,
    )
    assert result["difference"] == pytest.approx(0.5)
    assert result["ci95"] == pytest.approx([0.5, 0.5])
    assert result["bootstrap_method"] == "physical_scene_exp1_positive_weights"


def test_globally_unsupported_classes_are_fixed_map_exclusions() -> None:
    ground_truth = [
        _single_instance_scene("chair-scene", 0),
        _single_instance_scene("switch-scene", 1),
    ]
    predictions = {
        "P000": [_perfect_prediction("chair-scene", 0)],
        "P111": [
            _perfect_prediction("chair-scene", 0),
            _perfect_prediction("switch-scene", 1),
            PredictedInstance(
                "chair-scene",
                99,
                2,
                0.99,
                np.ones(3, dtype=bool),
            ),
        ],
    }
    classes = ["chair", "switch", "socket"]
    groups = {"chair-scene": "chair", "switch-scene": "switch"}

    compiled = evaluate_compiled(
        compile_predictions(ground_truth, predictions["P111"], classes, 1)
    )
    assert compiled["aggregate"]["map_50_95"] == pytest.approx(1.0)
    assert compiled["class_evaluation"] == {
        "evaluable_classes": ["chair", "switch"],
        "globally_unevaluable_classes": ["socket"],
        "globally_unevaluable_reason": "no_gt_instances_at_min_region_size",
        "map_denominator_class_count": 2,
    }

    bootstrap = paired_scene_bootstrap(
        ground_truth,
        predictions,
        groups,
        classes,
        "P000",
        "P111",
        samples=50,
        seed=11,
        min_region_size=1,
    )
    assert bootstrap["difference"] == pytest.approx(0.5)
    assert bootstrap["ci95"] == pytest.approx([0.5, 0.5])
    assert bootstrap["evaluable_classes"] == ["chair", "switch"]
    assert bootstrap["globally_unevaluable_classes"] == ["socket"]

    permutation = paired_scene_permutation_test(
        ground_truth,
        predictions,
        groups,
        classes,
        "P000",
        "P111",
        samples=100,
        seed=11,
        min_region_size=1,
    )
    assert permutation["observed"] == pytest.approx(0.5)
    assert permutation["evaluable_classes"] == ["chair", "switch"]
    assert permutation["globally_unevaluable_classes"] == ["socket"]


def test_min_region_filter_can_make_canonical_class_globally_unevaluable() -> None:
    scene = GroundTruthScene(
        "scene",
        semantic=np.asarray([0, 0, 0, 1]),
        instance=np.asarray([1, 1, 1, 2]),
    )
    chair_prediction = PredictedInstance(
        "scene",
        1,
        0,
        0.9,
        np.asarray([True, True, True, False]),
    )
    compiled = evaluate_compiled(
        compile_predictions(
            [scene], [chair_prediction], ["chair", "socket"], 2
        )
    )
    assert compiled["class_evaluation"]["evaluable_classes"] == ["chair"]
    assert compiled["class_evaluation"]["globally_unevaluable_classes"] == [
        "socket"
    ]
    assert compiled["per_class"]["socket"]["gt_instances"] == 0


def test_weighted_metric_matches_unweighted_perfect_result() -> None:
    ground_truth = [
        _single_instance_scene("chair-scene", 0),
        _single_instance_scene("cup-scene", 1),
    ]
    predictions = [
        _perfect_prediction("chair-scene", 0),
        _perfect_prediction("cup-scene", 1),
    ]
    result = weighted_scene_metric(
        ground_truth,
        predictions,
        ["chair", "cup"],
        {"chair-scene": 1.0, "cup-scene": 1.0},
        min_region_size=1,
    )
    assert result == pytest.approx(1.0)


def test_unit_weights_match_official_edge_case_semantics() -> None:
    scene = GroundTruthScene(
        "scene",
        semantic=np.asarray([0, 0, 0, 0, 0, 1, 1, -1, -1, -1]),
        instance=np.asarray([1, 1, 1, 1, 2, 3, 3, -1, -1, -1]),
    )
    masks = [
        [0, 1],  # exact IoU 0.5: ScanNet's strict comparison does not match
        [0, 1, 2, 3],
        [0, 1, 2, 3],  # duplicate lower-score prediction becomes an FP event
        [4, 7, 8],  # small GT and void-dominated prediction is ignored
    ]
    predictions = []
    for instance_id, (indices, score) in enumerate(
        zip(masks, [0.95, 0.9, 0.8, 0.7], strict=True), start=1
    ):
        mask = np.zeros(10, dtype=bool)
        mask[indices] = True
        predictions.append(
            PredictedInstance("scene", instance_id, 0, score, mask)
        )
    official_result = evaluate_instances(
        [scene], predictions, ["chair"], min_region_size=2
    )
    official = official_result["aggregate"]["map_50_95"]
    weighted = weighted_scene_metric(
        [scene],
        predictions,
        ["chair"],
        {"scene": 1.0},
        min_region_size=2,
    )
    assert weighted == pytest.approx(official)
    compiled_result = evaluate_compiled(
        compile_predictions(
            [scene], predictions, ["chair"], min_region_size=2
        )
    )
    assert compiled_result["aggregate"] == pytest.approx(
        official_result["aggregate"]
    )
    assert compiled_result["per_class"]["chair"] == pytest.approx(
        official_result["per_class"]["chair"]
    )
    assert compiled_result["aggregate"]["map_0.25"] is not None


def test_valid_semantic_with_negative_instance_is_not_void() -> None:
    scene = GroundTruthScene(
        "scene",
        semantic=np.asarray([0, 0]),
        instance=np.asarray([1, -1]),
    )
    false_positive = PredictedInstance(
        "scene", 1, 0, 0.99, np.asarray([False, True])
    )
    true_positive = PredictedInstance(
        "scene", 2, 0, 0.8, np.asarray([True, False])
    )
    predictions = [false_positive, true_positive]
    official = evaluate_instances(
        [scene], predictions, ["chair"], min_region_size=1
    )
    compiled = evaluate_compiled(
        compile_predictions(
            [scene], predictions, ["chair"], min_region_size=1
        )
    )
    assert official["aggregate"]["map_50_95"] < 1.0
    assert compiled["aggregate"] == pytest.approx(official["aggregate"])
    assert compiled["per_class"]["chair"] == pytest.approx(
        official["per_class"]["chair"]
    )


def test_paired_scene_permutation_uses_plus_one_correction() -> None:
    ground_truth = [_single_instance_scene(f"scene-{index}") for index in range(8)]
    treatment = [_perfect_prediction(scene.scene_id) for scene in ground_truth]
    predictions = {"P000": [], "P111": treatment}
    groups = {scene.scene_id: scene.scene_id for scene in ground_truth}
    result = paired_scene_permutation_test(
        ground_truth,
        predictions,
        groups,
        ["chair"],
        "P000",
        "P111",
        samples=5_000,
        seed=11,
        min_region_size=1,
    )
    assert result["observed"] == pytest.approx(1.0)
    assert result["p_two_sided"] == pytest.approx(
        (result["extreme_permutations"] + 1) / 5_001
    )
    assert result["p_two_sided"] < 0.05


def test_paired_scene_permutation_returns_one_for_zero_effect() -> None:
    ground_truth = [_single_instance_scene(f"scene-{index}") for index in range(3)]
    perfect = [_perfect_prediction(scene.scene_id) for scene in ground_truth]
    result = paired_scene_permutation_test(
        ground_truth,
        {"P000": perfect, "P111": perfect},
        {scene.scene_id: scene.scene_id for scene in ground_truth},
        ["chair"],
        "P000",
        "P111",
        samples=100,
        seed=3,
        min_region_size=1,
    )
    assert result["observed"] == pytest.approx(0.0)
    assert result["p_two_sided"] == pytest.approx(1.0)


def test_pooled_permutation_matches_primary_in_non_additive_case() -> None:
    ground_truth = [
        GroundTruthScene(
            f"scene-{index}",
            semantic=np.asarray([0, 1]),
            instance=np.asarray([1, 2]),
        )
        for index in range(3)
    ]

    def prediction(
        scene_id: str, instance_id: int, score: float, point: int
    ) -> PredictedInstance:
        mask = np.zeros(2, dtype=bool)
        mask[point] = True
        return PredictedInstance(scene_id, instance_id, 0, score, mask)

    reference = [prediction("scene-2", 1, 0.9, 0)]
    treatment = [
        prediction("scene-1", 1, 0.5, 0),
        prediction("scene-2", 1, 0.9, 1),
        prediction("scene-2", 2, 0.8, 0),
    ]
    predictions = {"P000": reference, "P111": treatment}
    groups = {scene.scene_id: scene.scene_id for scene in ground_truth}
    bootstrap = paired_scene_bootstrap(
        ground_truth,
        predictions,
        groups,
        ["target", "other"],
        "P000",
        "P111",
        samples=20,
        seed=5,
        min_region_size=1,
    )
    permutation = paired_scene_permutation_test(
        ground_truth,
        predictions,
        groups,
        ["target", "other"],
        "P000",
        "P111",
        samples=200,
        seed=5,
        min_region_size=1,
    )
    scene_differences = []
    for selected_scene in groups:
        weights = {
            scene_id: float(scene_id == selected_scene) for scene_id in groups
        }
        scene_differences.append(
            weighted_scene_metric(
                ground_truth,
                treatment,
                ["target", "other"],
                weights,
                min_region_size=1,
                require_all_classes=False,
            )
            - weighted_scene_metric(
                ground_truth,
                reference,
                ["target", "other"],
                weights,
                min_region_size=1,
                require_all_classes=False,
            )
        )
    assert bootstrap["difference"] < 0.0
    assert np.mean(scene_differences) > 0.0
    assert permutation["observed"] == pytest.approx(bootstrap["difference"])
    assert permutation["statistic"] == "fixed_class_pooled_map_50_95_difference"
    pattern_differences: dict[tuple[int, ...], float] = {}
    scene_ids = sorted(groups)
    unit_weights = {scene_id: 1.0 for scene_id in scene_ids}
    for encoded in range(2 ** len(scene_ids)):
        pattern = tuple((encoded >> index) & 1 for index in range(len(scene_ids)))
        mixed_reference: list[PredictedInstance] = []
        mixed_treatment: list[PredictedInstance] = []
        for scene_id, swap in zip(scene_ids, pattern, strict=True):
            reference_scene = [item for item in reference if item.scene_id == scene_id]
            treatment_scene = [item for item in treatment if item.scene_id == scene_id]
            mixed_reference.extend(treatment_scene if swap else reference_scene)
            mixed_treatment.extend(reference_scene if swap else treatment_scene)
        pattern_differences[pattern] = weighted_scene_metric(
            ground_truth,
            mixed_treatment,
            ["target", "other"],
            unit_weights,
            min_region_size=1,
        ) - weighted_scene_metric(
            ground_truth,
            mixed_reference,
            ["target", "other"],
            unit_weights,
            min_region_size=1,
        )
    random_patterns = np.random.default_rng(5).integers(
        0, 2, size=(200, len(scene_ids)), dtype=np.int8
    )
    expected_extreme = sum(
        abs(pattern_differences[tuple(int(value) for value in row)])
        >= abs(bootstrap["difference"]) - 1e-12
        for row in random_patterns
    )
    assert permutation["extreme_permutations"] == expected_extreme


def test_compiled_predictions_are_reusable_without_dense_masks() -> None:
    ground_truth = [
        _single_instance_scene("chair-scene", 0),
        _single_instance_scene("cup-scene", 1),
    ]
    reference = [_perfect_prediction("chair-scene", 0)]
    treatment = [
        _perfect_prediction("chair-scene", 0),
        _perfect_prediction("cup-scene", 1),
    ]
    compiled_reference = compile_predictions(
        ground_truth, reference, ["chair", "cup"], min_region_size=1
    )
    compiled_treatment = compile_predictions(
        ground_truth, treatment, ["chair", "cup"], min_region_size=1
    )
    assert isinstance(compiled_reference, CompiledEvaluation)
    assert not hasattr(compiled_reference, "predictions")
    result = paired_scene_bootstrap(
        ground_truth,
        {"P000": {42: compiled_reference}, "P111": {42: compiled_treatment}},
        {"chair-scene": "chair", "cup-scene": "cup"},
        ["chair", "cup"],
        "P000",
        "P111",
        samples=10,
        seed=9,
        min_region_size=1,
    )
    assert result["difference"] == pytest.approx(0.5)


def test_scene_wise_compilation_merges_to_one_shot_evaluation() -> None:
    ground_truth = [
        GroundTruthScene(
            f"scene-{index}",
            semantic=np.asarray([0, 0, 1, 1]),
            instance=np.asarray([1, 1, 2, 2]),
        )
        for index in range(3)
    ]

    def mask(*indices: int) -> np.ndarray:
        result = np.zeros(4, dtype=bool)
        result[list(indices)] = True
        return result

    predictions = [
        PredictedInstance("scene-0", 1, 0, 0.9, mask(0, 1)),
        PredictedInstance("scene-0", 2, 1, 0.8, mask(2, 3)),
        PredictedInstance("scene-1", 1, 0, 0.95, mask(2)),
        PredictedInstance("scene-1", 2, 0, 0.7, mask(0, 1)),
        PredictedInstance("scene-2", 1, 1, 0.85, mask(2, 3)),
    ]
    classes = ["chair", "cup"]
    one_shot = compile_predictions(
        ground_truth, predictions, classes, min_region_size=1
    )
    parts = [
        compile_predictions(
            [scene],
            [item for item in predictions if item.scene_id == scene.scene_id],
            classes,
            min_region_size=1,
        )
        for scene in ground_truth
    ]
    merged = merge_compiled_evaluations(parts)
    expected = evaluate_compiled(one_shot)
    actual = evaluate_compiled(merged)
    assert actual["aggregate"] == pytest.approx(expected["aggregate"])
    for class_name in classes:
        assert actual["per_class"][class_name] == pytest.approx(
            expected["per_class"][class_name]
        )

    weights = {"scene-0": 0.2, "scene-1": 2.0, "scene-2": 0.7}
    raw_metric = weighted_scene_metric(
        ground_truth,
        predictions,
        classes,
        weights,
        min_region_size=1,
    )
    assert weighted_scene_metric(
        ground_truth,
        one_shot,
        classes,
        weights,
        min_region_size=1,
    ) == pytest.approx(raw_metric)
    assert weighted_scene_metric(
        ground_truth,
        merged,
        classes,
        weights,
        min_region_size=1,
    ) == pytest.approx(raw_metric)

    with pytest.raises(ValueError, match="repeat scenes"):
        merge_compiled_evaluations([parts[0], parts[0]])
    incompatible = compile_predictions(
        [ground_truth[0]],
        [item for item in predictions if item.scene_id == "scene-0"],
        classes,
        min_region_size=2,
    )
    with pytest.raises(ValueError, match="incompatible protocols"):
        merge_compiled_evaluations([parts[0], incompatible])


def test_technical_replicates_are_averaged_not_overwritten() -> None:
    ground_truth = [
        _single_instance_scene("chair-scene", 0),
        _single_instance_scene("cup-scene", 1),
    ]
    perfect = [
        _perfect_prediction("chair-scene", 0),
        _perfect_prediction("cup-scene", 1),
    ]
    predictions = {
        "P000": {42: [], 3407: perfect},
        "P111": {42: perfect, 3407: []},
    }
    result = paired_scene_bootstrap(
        ground_truth,
        predictions,
        {"chair-scene": "chair", "cup-scene": "cup"},
        ["chair", "cup"],
        "P000",
        "P111",
        samples=20,
        seed=2,
        min_region_size=1,
    )
    assert result["difference"] == pytest.approx(0.0)
    assert result["ci95"] == pytest.approx([0.0, 0.0])
    assert result["technical_replicates"] == ["3407", "42"]
    permutation = paired_scene_permutation_test(
        ground_truth,
        predictions,
        {"chair-scene": "chair", "cup-scene": "cup"},
        ["chair", "cup"],
        "P000",
        "P111",
        samples=20,
        seed=2,
        min_region_size=1,
    )
    assert permutation["observed"] == pytest.approx(0.0)
    assert permutation["p_two_sided"] == pytest.approx(1.0)
    assert permutation["technical_replicates"] == ["3407", "42"]


def test_unbalanced_technical_replicates_are_rejected() -> None:
    ground_truth = [_single_instance_scene("scene")]
    with pytest.raises(ValueError, match="same technical replicates"):
        paired_scene_bootstrap(
            ground_truth,
            {
                "P000": {42: [], 3407: []},
                "P111": {42: [_perfect_prediction("scene")]},
            },
            {"scene": "scene"},
            ["chair"],
            "P000",
            "P111",
            samples=1,
            min_region_size=1,
        )


def test_factorial_reports_all_effects_with_holm_uncertainty() -> None:
    ground_truth = [_single_instance_scene(f"scene-{index}") for index in range(2)]
    perfect = [_perfect_prediction(scene.scene_id) for scene in ground_truth]
    bits = {
        f"P{size}{smooth}{small}": (size, smooth, small)
        for size in (0, 1)
        for smooth in (0, 1)
        for small in (0, 1)
    }
    predictions = {
        condition: perfect if factor_bits[0] else []
        for condition, factor_bits in bits.items()
    }
    result = factorial_bootstrap(
        ground_truth,
        predictions,
        bits,
        {scene.scene_id: scene.scene_id for scene in ground_truth},
        ["chair"],
        samples=1_000,
        seed=13,
        min_region_size=1,
    )
    assert len(result["effects"]) == 7
    assert result["effects"]["size"]["effect"] == pytest.approx(1.0)
    assert result["effects"]["size"]["ci95"] == pytest.approx([1.0, 1.0])
    assert result["effects"]["size"]["p_holm"] < 0.05
    for name, effect in result["effects"].items():
        if name != "size":
            assert effect["effect"] == pytest.approx(0.0)
            assert effect["p_holm"] == pytest.approx(1.0)


def test_schedule_is_seeded_and_block_complete(tmp_path) -> None:
    selection = tmp_path / "selection.json"
    write_json(
        selection, {"selection": {"tune": ["scene1"], "locked": ["scene2", "scene3"]}}
    )
    first = build_run_schedule(selection, "locked", ("A", "B", "C"), (7, 9), 11)
    second = build_run_schedule(selection, "locked", ("A", "B", "C"), (7, 9), 11)
    assert first == second
    assert len(first["runs"]) == 2 * 2 * 3
    for block in {row["block"] for row in first["runs"]}:
        assert {row["condition"] for row in first["runs"] if row["block"] == block} == {
            "A",
            "B",
            "C",
        }
