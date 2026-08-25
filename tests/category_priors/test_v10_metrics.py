from __future__ import annotations

import json

import numpy as np
import pytest

from category_priors.io import load_json, read_rows
from category_priors.v10_metrics import (
    V10_STAGE_ORDER,
    GaussianGTIndex,
    V10AcceptedEdge,
    V10FragmentSupport,
    V10GroundTruthObject,
    V10StageCandidate,
    adapt_v10_persisted_bank,
    analyse_v10_rows,
    evaluate_v10_audit,
    recompute_saved_v10_analysis,
    write_v10_results,
)


def _fixture() -> tuple[GaussianGTIndex, list[V10GroundTruthObject]]:
    # GT points 0..5 belong to chair 1 and 6..11 belong to table 2.
    # Gaussian 4 has no inverse GT support and is therefore a real FP.
    nearest = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    index = GaussianGTIndex.from_nearest(
        nearest,
        np.ones(len(nearest), dtype=bool),
        gaussian_count=5,
    )
    ground_truth = [
        V10GroundTruthObject("scene", "chair", 1, np.arange(0, 6), True, "tiny"),
        V10GroundTruthObject("scene", "table", 2, np.arange(6, 12), True),
    ]
    return index, ground_truth


def test_accepted_edges_separate_same_different_and_unknown_gt() -> None:
    index, ground_truth = _fixture()
    fragments = [
        V10FragmentSupport("scene", 0, np.asarray([0, 1])),
        V10FragmentSupport("scene", 1, np.asarray([0])),
        V10FragmentSupport("scene", 2, np.asarray([2, 3])),
        V10FragmentSupport("scene", 3, np.asarray([4])),
    ]
    edges = [
        V10AcceptedEdge("scene", 0, 1),
        V10AcceptedEdge("scene", 0, 2),
        V10AcceptedEdge("scene", 0, 3),
    ]

    result = evaluate_v10_audit(
        ground_truth=ground_truth,
        gaussian_gt_index=index,
        fragments=fragments,
        accepted_edges=edges,
    )
    edge_rows = [
        row for row in result["rows"] if row["row_type"] == "accepted_fragment_pair"
    ]
    assert [row["classification"] for row in edge_rows] == [
        "same_gt",
        "different_gt",
        "unknown",
    ]
    summary = result["analysis"]["accepted_fragment_pairs"]
    assert summary["identifiable_edge_count"] == 2
    assert summary["identifiable_precision"] == pytest.approx(0.5)
    assert summary["all_edge_precision"] == pytest.approx(1 / 3)
    assert summary["unknown_rate"] == pytest.approx(1 / 3)


def test_low_purity_fragment_is_identifiable_and_remains_in_precision_denominator() -> None:
    index = GaussianGTIndex.from_nearest(
        np.asarray([0, 0, 0]),
        np.ones(3, dtype=bool),
        gaussian_count=5,
    )
    ground_truth = [
        V10GroundTruthObject("scene", "chair", 1, np.arange(3), True, "tiny")
    ]
    result = evaluate_v10_audit(
        ground_truth=ground_truth,
        gaussian_gt_index=index,
        fragments=[
            V10FragmentSupport("scene", 0, np.arange(5)),
            V10FragmentSupport("scene", 1, np.asarray([0])),
        ],
        accepted_edges=[V10AcceptedEdge("scene", 0, 1)],
    )
    edge = next(
        row for row in result["rows"] if row["row_type"] == "accepted_fragment_pair"
    )
    assert edge["left_best_purity"] == pytest.approx(3 / 7)
    assert edge["left_identifiable"] is True
    assert edge["classification"] == "same_gt"
    summary = result["analysis"]["accepted_fragment_pairs"]
    assert summary["identifiable_edge_count"] == 1
    assert summary["unknown_edge_count"] == 0
    assert result["analysis"]["association_thresholds"]["min_intersection"] == 1
    assert result["analysis"]["association_thresholds"][
        "iou_and_purity_are_diagnostics_only"
    ]


def test_only_actual_accepted_edges_are_scored_not_component_proxy_pairs() -> None:
    index, ground_truth = _fixture()
    fragments = [
        V10FragmentSupport("scene", 10, np.asarray([0, 1])),
        V10FragmentSupport("scene", 11, np.asarray([0])),
        V10FragmentSupport("scene", 12, np.asarray([1])),
    ]
    # These two edges connect all three fragments as one component.  The old
    # proxy interpretation could accidentally score the unaccepted 10--12
    # transitive pair as well; V10 must use the recorded accepted edge list.
    edges = [
        V10AcceptedEdge("scene", 10, 11),
        V10AcceptedEdge("scene", 11, 12),
    ]
    result = evaluate_v10_audit(
        ground_truth=ground_truth,
        gaussian_gt_index=index,
        fragments=fragments,
        accepted_edges=edges,
    )
    rows = [
        row for row in result["rows"] if row["row_type"] == "accepted_fragment_pair"
    ]
    assert [(row["left_fragment_id"], row["right_fragment_id"]) for row in rows] == [
        (10, 11),
        (11, 12),
    ]
    assert result["analysis"]["accepted_fragment_pairs"]["accepted_edge_count"] == 2


def test_every_unmapped_gaussian_has_a_unique_false_positive_sentinel() -> None:
    nearest = np.asarray([0, 0, 0, 1, 1, 1])
    index = GaussianGTIndex.from_nearest(
        nearest,
        np.ones(len(nearest), dtype=bool),
        gaussian_count=4,
    )
    projected = index.project(np.asarray([0, 1, 2, 3]))

    assert projected.unmapped_gaussian_count == 2
    assert len(projected.support_ids) == 8
    assert set(projected.support_ids[-2:]) == {8, 9}

    ground_truth = [
        V10GroundTruthObject("scene", "chair", 1, np.arange(6), True)
    ]
    result = evaluate_v10_audit(
        ground_truth=ground_truth,
        gaussian_gt_index=index,
        stage_candidates=[
            V10StageCandidate(
                "scene", "final_candidate", 0, np.asarray([0, 1, 2, 3]), "chair"
            )
        ],
    )
    row = next(row for row in result["rows"] if row["row_type"] == "stage_candidate")
    assert row["best_intersection"] == 6
    assert row["best_purity"] == pytest.approx(6 / 8)
    assert row["best_iou"] == pytest.approx(6 / 8)
    assert row["unmapped_gaussian_count"] == 2


def test_stage_funnel_has_all_frozen_support_levels() -> None:
    index, ground_truth = _fixture()
    candidates = [
        V10StageCandidate("scene", stage, number, np.asarray([0, 1]), "chair")
        for number, stage in enumerate(V10_STAGE_ORDER)
    ]
    result = evaluate_v10_audit(
        ground_truth=ground_truth,
        gaussian_gt_index=index,
        stage_candidates=candidates,
    )

    assert result["analysis"]["stage_order"] == list(V10_STAGE_ORDER)
    assert set(result["analysis"]["stages"]) == set(V10_STAGE_ORDER)
    for stage in V10_STAGE_ORDER:
        block = result["analysis"]["stages"][stage]
        assert block["candidate_count"] == 1
        assert block["candidate_precision_050"] == pytest.approx(1.0)
        assert block["candidate_match_050_scene_count"] == 1
        assert block["covered_official_gt_050_count"] == 1
        assert block["same_class_candidate_precision_050"] == pytest.approx(1.0)
        assert block["same_class_official_gt_recall_050"] == pytest.approx(0.5)
        assert block["geometric_tiny_small_recall_025"] == pytest.approx(1.0)
        assert block["geometric_tiny_small_recall_050"] == pytest.approx(1.0)
        assert block["same_class_tiny_small_recall_050"] == pytest.approx(1.0)


def test_tiny_small_recall_deduplicates_official_gt_instances() -> None:
    index, ground_truth = _fixture()
    result = evaluate_v10_audit(
        ground_truth=ground_truth,
        gaussian_gt_index=index,
        stage_candidates=[
            V10StageCandidate("scene", "final_candidate", 0, np.asarray([0]), "chair"),
            V10StageCandidate("scene", "final_candidate", 1, np.asarray([1]), "chair"),
        ],
    )
    block = result["analysis"]["stages"]["final_candidate"]
    assert block["candidate_match_050_count"] == 2
    assert block["official_tiny_small_gt_count"] == 1
    assert block["covered_tiny_small_gt_050_count"] == 1
    assert block["geometric_tiny_small_recall_050"] == pytest.approx(1.0)


def test_final_candidate_score_iou_spearman_uses_selected_class_iou() -> None:
    index, ground_truth = _fixture()
    result = evaluate_v10_audit(
        ground_truth=ground_truth,
        gaussian_gt_index=index,
        stage_candidates=[
            V10StageCandidate(
                "scene", "final_candidate", 0, np.asarray([0]), "chair", 0.9
            ),
            V10StageCandidate(
                "scene", "final_candidate", 1, np.asarray([2]), "chair", 0.1
            ),
        ],
    )
    block = result["analysis"]["stages"]["final_candidate"]
    assert block["scored_candidate_count"] == 2
    assert block["score_iou_spearman"] == pytest.approx(1.0)


def test_tiny_recall_uses_all_gt_overlaps_not_only_candidate_dominant_gt() -> None:
    index, ground_truth = _fixture()
    # Gaussian 0 contributes three chair points; Gaussians 2/3 contribute all
    # six table points.  Table is the dominant GT, but chair still has IoU .25.
    result = evaluate_v10_audit(
        ground_truth=ground_truth,
        gaussian_gt_index=index,
        stage_candidates=[
            V10StageCandidate(
                "scene", "final_candidate", 0, np.asarray([0, 2, 3]), None
            )
        ],
    )
    candidate = next(
        row for row in result["rows"] if row["row_type"] == "stage_candidate"
    )
    assert candidate["best_gt_class_name"] == "table"
    block = result["analysis"]["stages"]["final_candidate"]
    assert block["geometric_tiny_small_recall_025"] == pytest.approx(1.0)
    assert block["geometric_tiny_small_recall_050"] == pytest.approx(0.0)


def test_late_classifier_accuracy_uses_the_best_geometric_gt_class() -> None:
    index, ground_truth = _fixture()
    # The prediction is geometrically dominated by table (IoU 2/3), while its
    # selected chair class still overlaps a chair at exactly IoU .25.  That is
    # a valid same-class detection overlap but an incorrect late-classifier
    # decision for this geometric candidate.
    result = evaluate_v10_audit(
        ground_truth=ground_truth,
        gaussian_gt_index=index,
        stage_candidates=[
            V10StageCandidate(
                "scene", "final_candidate", 0, np.asarray([0, 2, 3]), "chair"
            )
        ],
    )
    candidate = next(
        row for row in result["rows"] if row["row_type"] == "stage_candidate"
    )
    assert candidate["best_official_gt_class_name"] == "table"
    assert candidate["best_same_class_official_iou"] == pytest.approx(0.25)
    block = result["analysis"]["stages"]["final_candidate"]
    assert block["same_class_candidate_match_025_count"] == 1
    assert block["late_classifier_correct_025_count"] == 0


def _persisted_payload():
    fragments = [
        {"fragment_id": 10, "frame_id": 0},
        {"fragment_id": 11, "frame_id": 1},
    ]
    raw_supports = {
        "single_full": [
            {"candidate_id": 10, "gaussian_ids": [0, 1], "class_name": None},
            {"candidate_id": 11, "gaussian_ids": [2, 3], "class_name": None},
        ],
        "single_core": [
            {"candidate_id": 10, "gaussian_ids": [0], "class_name": None},
            {"candidate_id": 11, "gaussian_ids": [2], "class_name": None},
        ],
    }
    for stage, support in (
        ("component_full_union", [0, 1, 2, 3]),
        ("component_core_union", [0, 2]),
        ("pre_conflict", [0, 1, 2, 3]),
        ("post_conflict", [0, 1, 2, 3]),
        ("unique_ownership", [0, 1, 2, 3]),
        ("final_candidate", [0, 1, 2, 3]),
    ):
        raw_supports[stage] = [
            {"candidate_id": 0, "gaussian_ids": support, "class_name": "chair"}
        ]
    # The runner canonicalizes every stage candidate ID to its row index.  A
    # fragment's stable fragment_id remains in metadata.fragments and is used
    # by actual association edges; stage candidate IDs are a separate contract.
    for stage in ("single_full", "single_core"):
        for index, row in enumerate(raw_supports[stage]):
            row["candidate_id"] = index
    supports = {
        stage: tuple(np.asarray(row["gaussian_ids"], dtype=np.int32) for row in rows)
        for stage, rows in raw_supports.items()
    }
    descriptors = {
        stage: [
            {
                "candidate_id": int(row["candidate_id"]),
                "class_name": row["class_name"],
                "support_count": len(row["gaussian_ids"]),
            }
            for row in rows
        ]
        for stage, rows in raw_supports.items()
    }
    metadata = {
        "scene_id": "scene",
        "fragments": fragments,
        "accepted_edges": [
            {
                "left_fragment_id": 10,
                "right_fragment_id": 11,
                "left_frame_id": 0,
                "right_frame_id": 1,
                "kind": "pair",
                "score": 0.9,
                "shared": 3,
                "frame_weighted_jaccard": 0.7,
                "p0_overlap": 1.0,
                "left_coverage": 0.9,
                "right_coverage": 0.8,
                "row_margin": 0.2,
                "column_margin": 0.3,
                "component_support_ratio": 1.0,
                "strong": False,
                "cycle_supported": False,
            }
        ],
        "stage_supports": descriptors,
        "candidates": [
            {
                "candidate_id": 0,
                "classifiers": {
                    "mv-label": {
                        "branch_class": "chair",
                        "classification_eligible": True,
                        "base_score": 0.9,
                    },
                    "codebook": {
                        "branch_class": "table",
                        "classification_eligible": True,
                        "base_score": 0.8,
                    },
                },
            }
        ],
    }
    return metadata, supports


def test_persisted_bank_adapter_uses_actual_edges_and_all_eight_stages(
    monkeypatch,
) -> None:
    import category_priors.v10_runner as v10_runner

    metadata, supports = _persisted_payload()
    bank = type("Bank", (), {"full_ids": (np.arange(4, dtype=np.int32),)})()
    monkeypatch.setattr(
        v10_runner, "load_v10_candidate_bank", lambda _path: (metadata, bank)
    )
    monkeypatch.setattr(
        v10_runner, "load_v10_audit_supports", lambda _path: (metadata, supports)
    )

    adapted = adapt_v10_persisted_bank("ignored")
    assert len(adapted.fragments) == 2
    edge_ids = [
        (edge.left_fragment_id, edge.right_fragment_id)
        for edge in adapted.accepted_edges
    ]
    assert edge_ids == [(10, 11)]
    assert adapted.accepted_edges[0].left_coverage == pytest.approx(0.9)
    assert adapted.accepted_edges[0].right_coverage == pytest.approx(0.8)
    assert adapted.accepted_edges[0].component_support_ratio == pytest.approx(1.0)
    assert {row.stage for row in adapted.stage_candidates} == set(V10_STAGE_ORDER)
    assert len(adapted.stage_candidates) == 10


def test_persisted_bank_adapter_rejects_missing_stage_instead_of_inventing_it(
    monkeypatch,
) -> None:
    import category_priors.v10_runner as v10_runner

    metadata, supports = _persisted_payload()
    del metadata["stage_supports"]["post_conflict"]
    bank = type("Bank", (), {"full_ids": (np.arange(4, dtype=np.int32),)})()
    monkeypatch.setattr(
        v10_runner, "load_v10_candidate_bank", lambda _path: (metadata, bank)
    )
    monkeypatch.setattr(
        v10_runner, "load_v10_audit_supports", lambda _path: (metadata, supports)
    )

    with pytest.raises(ValueError, match="missing=.*post_conflict"):
        adapt_v10_persisted_bank("ignored")


def test_adapter_reads_the_actual_runner_json_and_ragged_npz_contract(
    tmp_path, monkeypatch,
) -> None:
    import category_priors.v10_runner as v10_runner
    from category_priors.v9_lifting import V9_LIFTING_SCHEMA

    scene_id = "scene0000_00"
    lifting_dir = tmp_path / "lifting" / scene_id
    lifting_dir.mkdir(parents=True)
    lifting_metadata = {
        "schema": V9_LIFTING_SCHEMA,
        "scene_id": scene_id,
        "point_count": 5,
        "frame_count": 2,
        "identity": {
            "schema": "test-lifting-identity",
            "scene_id": scene_id,
            "git_commit": "producer",
        },
    }
    (lifting_dir / "lifting_bank.json").write_text(
        json.dumps(lifting_metadata), encoding="utf-8"
    )
    metadata, supports = _persisted_payload()

    def fake_load(_source):
        return lifting_metadata, {"xyz_m": np.zeros((5, 3), dtype=np.float32)}

    def builder(_metadata, _arrays, *, condition: str):
        stage_supports = {
            stage: [
                {
                    "candidate_id": descriptor["candidate_id"],
                    "class_name": descriptor["class_name"],
                    "gaussian_ids": support,
                }
                for descriptor, support in zip(metadata["stage_supports"][stage], rows)
            ]
            for stage, rows in supports.items()
        }
        return {
            "point_count": 5,
            "fragments": metadata["fragments"],
            "tracks": [{"track_id": 0, "fragment_ids": [10, 11]}],
            "accepted_edges": metadata["accepted_edges"],
            "stage_supports": stage_supports,
            "candidates": [
                {
                    "candidate_id": 0,
                    "track_id": 0,
                    "branch_class": "chair",
                    "classification_eligible": True,
                    "classifiers": {
                        "mv-label": {
                            "branch_class": "chair",
                            "classification_eligible": True,
                            "base_score": 0.9,
                        },
                        "codebook": {
                            "branch_class": "chair",
                            "classification_eligible": True,
                            "base_score": 0.8,
                        },
                    },
                    "full_point_count": 4,
                    "core_point_count": 2,
                    "base_score": 0.9,
                    "metric_extents_m": [0.1, 0.2, 0.3],
                    "local_surface_density": 10.0,
                    "boundary_ratio_5cm": 0.2,
                    "structure_condition": condition,
                }
            ],
            "full_ids": [np.arange(4, dtype=np.int32)],
            "core_ids": [np.asarray([0, 2], dtype=np.int32)],
        }

    monkeypatch.setattr(v10_runner, "load_lifting_bank", fake_load)
    output_root = tmp_path / "banks"
    v10_runner.run_v10_banks(
        lifting_root=tmp_path / "lifting",
        output_root=output_root,
        scene_ids=[scene_id],
        conditions=["P0R0"],
        git_commit="consumer",
        builder=builder,
    )

    adapted = adapt_v10_persisted_bank(output_root / "P0R0" / scene_id)
    assert adapted.scene_id == scene_id
    assert len(adapted.fragments) == 2
    assert len(adapted.accepted_edges) == 1
    assert {row.stage for row in adapted.stage_candidates} == set(V10_STAGE_ORDER)


@pytest.mark.parametrize("suffix", [".jsonl", ".parquet"])
def test_saved_rows_recompute_analysis_independently_and_exactly(
    tmp_path, suffix: str
) -> None:
    index, ground_truth = _fixture()
    result = evaluate_v10_audit(
        ground_truth=ground_truth,
        gaussian_gt_index=index,
        fragments=[
            V10FragmentSupport("scene", 0, np.asarray([0, 1])),
            V10FragmentSupport("scene", 1, np.asarray([2, 3])),
        ],
        accepted_edges=[V10AcceptedEdge("scene", 0, 1)],
        stage_candidates=[
            V10StageCandidate(
                "scene", "single_full", 0, np.asarray([0, 1, 4]), "chair"
            ),
            V10StageCandidate(
                "scene", "final_candidate", 0, np.asarray([0, 1]), "chair"
            ),
        ],
    )
    rows_path = tmp_path / f"v10_rows{suffix}"
    analysis_path = tmp_path / "v10_analysis.json"
    written = write_v10_results(
        rows_output=rows_path,
        analysis_output=analysis_path,
        rows=result["rows"],
    )

    persisted_rows = read_rows(rows_path)
    independent = analyse_v10_rows(persisted_rows)
    assert independent == written == load_json(analysis_path)
    assert recompute_saved_v10_analysis(rows_path, analysis_path) == independent
