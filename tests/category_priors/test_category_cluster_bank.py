from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import numpy as np

from category_priors.category_cluster_bank import (
    CLUSTER_CONDITIONS,
    G1_MUTUAL_LOCAL_GRAPH,
    R0_LEGACY,
    R1_METRIC_HDBSCAN,
    R2_ANCHORED_HDBSCAN,
    build_cluster_bank_family,
    load_cluster_raw_audit,
    measure_cluster_family_determinism,
    save_cluster_raw_audit,
)


class _TwoRawClusterer:
    """Deterministic test double with one retained and one dropped raw cluster."""

    def fit_predict(self, distance: np.ndarray) -> np.ndarray:
        count = len(distance)
        self.probabilities_ = np.full(count, 0.8, dtype=np.float64)
        labels = np.zeros(count, dtype=np.int64)
        labels[3:] = 1
        return labels


def _factory(**_: object) -> _TwoRawClusterer:
    return _TwoRawClusterer()


def _family():
    class_names = tuple(f"class-{index:02d}" for index in range(32))
    label_features = np.eye(32, dtype=np.float64)
    semantic = np.zeros((6, 32), dtype=np.float64)
    semantic[0, 1] = 1.0  # Non-SAGA20 winner: never enters the branch.
    semantic[1:, 0] = 0.8
    semantic[1:, 1] = 0.6  # Unit norm and class-0 score exactly 0.8.
    return build_cluster_bank_family(
        np.tile([1.0, 0.0], (6, 1)),
        semantic,
        np.zeros((6, 3), dtype=np.float64),
        label_features,
        class_names,
        (class_names[0],),
        np.full(6, -1, dtype=np.int64),
        1.0,
        1.0,
        scene_id="scene-test_00",
        conditions=CLUSTER_CONDITIONS,
        hdbscan_factory=_factory,
    )


def test_bank_aggregates_measured_distance_and_all_raw_members() -> None:
    family = _family()

    r0 = family.banks[R0_LEGACY]
    r1 = family.banks[R1_METRIC_HDBSCAN]
    r2 = family.banks[R2_ANCHORED_HDBSCAN]
    g1 = family.banks[G1_MUTUAL_LOCAL_GRAPH]

    # The fake HDBSCAN result contains five raw members, but the two-member
    # raw cluster is dropped.  Denominators must not be reconstructed from
    # emitted candidate rows (which only contain the retained three members).
    for bank in (r0, r1, r2):
        assert bank.diagnostics["raw_member_count"] == 5
        assert bank.diagnostics["raw_member_retained_count"] == 3
    assert g1.diagnostics["raw_member_count"] == 0
    assert g1.diagnostics["raw_member_retained_count"] == 0

    for bank in (r1, r2):
        assert bank.diagnostics["distance_matrix_count"] == 1
        assert bank.diagnostics["distance_all_finite"] is True
        assert bank.diagnostics["distance_symmetry_max_abs"] == 0.0
        assert bank.diagnostics["distance_diagonal_max_abs"] == 0.0
        assert bank.diagnostics["corrected_distance_contract_passed"] is True
        assert bank.diagnostics["corrected_distance_contract_measured"] is True

    # The legacy semantic-confidence matrix has a genuinely non-zero
    # diagonal here (0.2 * (1 - 0.8**2)); the audit must report that measured
    # fact rather than incorrectly stamping a zero-diagonal declaration.
    legacy_distance = family.r0_diagnostics["legacy_distance_measurements"]
    assert legacy_distance["distance_matrix_count"] == 1
    assert legacy_distance["distance_diagonal_max_abs"] > 0.0

    # G1 reports both the class-local ordering key and the actual scene point.
    assert g1.candidates[0]["minimum_class_local_index"] == 0
    assert g1.candidates[0]["minimum_global_point_index"] == 1


def test_r0_raw_audit_is_standalone_validated_and_reports_actual_path(
    tmp_path: Path,
) -> None:
    family = _family()

    saved = save_cluster_raw_audit(family, tmp_path / "audit-directory")
    loaded = load_cluster_raw_audit(tmp_path / "audit-directory")

    assert saved == tmp_path / "audit-directory" / "r0_raw_trace.npz"
    assert loaded["schema"] == "saga-category-cluster-r0-raw-audit-v2"
    np.testing.assert_array_equal(loaded["sample_rank"] >= 0, [False] + [True] * 5)
    np.testing.assert_array_equal(loaded["sample_class_index"], [-1, 0, 0, 0, 0, 0])
    assert loaded["diagnostics"]["raw_member_count"] == 5
    assert (
        loaded["diagnostics"]["legacy_distance_measurements"]
        ["distance_diagonal_max_abs"]
        > 0.0
    )


def test_shell_forwards_repeatable_cluster_conditions_and_actual_audit_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    shell = (repo_root / "run_pipeline.sh").read_text(encoding="utf-8")
    postprocess = (repo_root / "postprocess.py").read_text(encoding="utf-8")

    assert 'category_cluster_conditions+=("$2")' in shell
    assert 'for cluster_condition in "${category_cluster_conditions[@]}"' in shell
    assert 'prior_args+=(--category-cluster-condition "$cluster_condition")' in shell
    assert 'prior_args+=(--category-cluster-verify-determinism)' in shell
    assert 'command.append("--category-cluster-verify-determinism")' not in postprocess
    assert "if args.category_cluster_verify_determinism:" in postprocess
    assert "category_cluster_conditions: $cluster_conditions_display" in shell
    assert "raw_audit_path = save_cluster_raw_audit(" in postprocess
    assert '"raw_audit_path": str(raw_audit_path.resolve())' in postprocess


def test_determinism_is_unmeasured_until_independent_family_is_compared() -> None:
    first = _family()
    assert first.banks[R1_METRIC_HDBSCAN].diagnostics["determinism_measured"] is False
    assert "determinism_violation_count" not in first.banks[R1_METRIC_HDBSCAN].diagnostics

    measured = measure_cluster_family_determinism(first, _family())
    for bank in measured.banks.values():
        assert bank.diagnostics["determinism_measured_this_scene"] is True
        assert bank.diagnostics["determinism_contract_verified"] is True
        assert bank.diagnostics["determinism_violation_count"] == 0


def test_pointwise_determinism_check_detects_a_changed_bank_label() -> None:
    first = _family()
    repeated = _family()
    bank = repeated.banks[R2_ANCHORED_HDBSCAN]
    changed = np.asarray(bank.branch_full_labels).copy()
    changed[1] = -1 if changed[1] >= 0 else 0
    repeated = replace(
        repeated,
        banks={
            **dict(repeated.banks),
            R2_ANCHORED_HDBSCAN: replace(bank, branch_full_labels=changed),
        },
    )

    measured = measure_cluster_family_determinism(first, repeated)

    diagnostics = measured.banks[R2_ANCHORED_HDBSCAN].diagnostics
    assert diagnostics["determinism_measured_this_scene"] is True
    assert diagnostics["determinism_contract_verified"] is False
    assert diagnostics["determinism_point_violation_count"] == 1
    assert diagnostics["determinism_violation_count"] >= 1
