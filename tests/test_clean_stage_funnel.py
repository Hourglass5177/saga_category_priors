from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from category_priors.clean_baseline.evidence import (
    build_sparse_frame_evidence,
    load_evidence_bank,
    save_evidence_bank,
)
from category_priors.clean_baseline.models import AlphaMaskEvidenceBank
from category_priors.clean_baseline.pipeline import run_consensus_condition
from category_priors.clean_baseline.stage_funnel import (
    FunnelObject,
    STAGE_NAMES,
    _partition_equivalence,
    audit_frozen_clean_scene,
    reconstruct_clean_stage_funnel,
)
from category_priors.taxonomy import load_taxonomy


CLASSES = (
    "chair",
    "table",
    "plant",
    "flower",
    "foliage",
    "tv",
    "painting",
    "sofa",
    "cabinet",
    "bed",
    "wall",
    "floor",
    "ceiling",
    "person",
    "socket",
    "remote",
    "key",
    "book",
    "lighting",
    "switch",
    "door",
    "window",
    "lamp",
    "speaker",
    "computer",
    "fan",
    "refrigerator",
    "robot",
    "cup",
    "vase",
    "phone",
    "trash can",
)


def _frame(frame_id: int, mask_id: int) -> object:
    ids = np.arange(4, dtype=np.int32)
    posterior = np.zeros((1, len(CLASSES)), dtype=np.float32)
    posterior[0, 0] = 1.0
    return build_sparse_frame_evidence(
        frame_id=frame_id,
        image_name=f"frame-{frame_id}",
        point_count=4,
        visible_ids=ids,
        visible_mass=np.ones(4, dtype=np.float32),
        mask_gaussian_ids=[ids],
        mask_inside_mass=[np.ones(4, dtype=np.float32)],
        mask_inside_ratio=[np.ones(4, dtype=np.float32)],
        semantic_posteriors=posterior,
        semantic_abstained=np.asarray([False]),
        global_mask_ids=[mask_id],
        valid_pixel_count=4,
        class_count=len(CLASSES),
    )


def _write_bank(path: Path) -> None:
    bank = AlphaMaskEvidenceBank.from_frames(
        scene_id="scene-test",
        point_count=4,
        xyz_m=np.asarray(
            [
                [0.00, 0.00, 0.00],
                [0.01, 0.00, 0.00],
                [0.02, 0.00, 0.00],
                [0.03, 0.00, 0.00],
            ],
            dtype=np.float32,
        ),
        class_names=CLASSES,
        frames=[_frame(0, 0), _frame(1, 1)],
        source={"test": True},
    )
    save_evidence_bank(bank, path)


def _run_small_condition(tmp_path: Path) -> tuple[Path, dict[str, object], dict[str, object]]:
    bank_dir = tmp_path / "bank"
    output_dir = tmp_path / "condition"
    _write_bank(bank_dir)
    run_consensus_condition(
        scene_id="scene-test",
        bank_dir=bank_dir,
        condition="C0-no-prior",
        output_dir=output_dir,
        allowed_classes=load_taxonomy().canonical_classes,
        consumer_commit="0" * 40,
    )
    diagnostics = json.loads((output_dir / "diagnostics.json").read_text(encoding="utf-8"))
    output = json.loads((output_dir / "output.json").read_text(encoding="utf-8"))
    return bank_dir, diagnostics, output


def test_stage_funnel_reconstructs_frozen_partition_exactly(tmp_path: Path) -> None:
    bank_dir, diagnostics, output = _run_small_condition(tmp_path)
    funnel = audit_frozen_clean_scene(
        bank_dir=bank_dir,
        diagnostics_path=tmp_path / "condition" / "diagnostics.json",
        output_path=tmp_path / "condition" / "output.json",
        allowed_classes=load_taxonomy().canonical_classes,
    )

    assert tuple(stage.name for stage in funnel.stages) == STAGE_NAMES
    assert all(stage.available for stage in funnel.stages)
    assert funnel.final_equivalence is not None
    assert funnel.final_equivalence.equivalent is True
    assert funnel.final_equivalence.changed_points == 0
    assert funnel.stage("complete_mask_support").summary["object_count"] == 2
    assert funnel.stage("association_support").summary["object_count"] == 2
    assert funnel.stage("accepted_edge_components").summary["object_count"] == 1
    assert funnel.stage("detection_ratio_filtered").summary["object_count"] == 1
    assert funnel.stage("physical_split_and_deduplicated").summary["object_count"] == 1
    assert funnel.stage("unique_gaussian_ownership").summary["object_count"] == 1
    assert funnel.stage("final_export").summary["object_count"] == 1
    assert diagnostics["consensus"]["component_count_before_output_filters"] == 1
    assert len(output["instances"]) == 1


def test_stage_funnel_marks_missing_lineage_unavailable(tmp_path: Path) -> None:
    bank_dir, diagnostics, output = _run_small_condition(tmp_path)
    del diagnostics["accepted_edges"]
    bank = load_evidence_bank(bank_dir)

    funnel = reconstruct_clean_stage_funnel(
        bank=bank,
        diagnostics=diagnostics,
        output=output,
        allowed_classes=load_taxonomy().canonical_classes,
    )

    assert funnel.stage("undersegmentation_filtered").available is True
    for name in STAGE_NAMES[3:7]:
        stage = funnel.stage(name)
        assert stage.available is False
        assert "accepted_edges" in str(stage.reason)
    assert funnel.stage("final_export").available is True
    assert funnel.final_equivalence is None


def test_stage_funnel_detects_final_partition_drift(tmp_path: Path) -> None:
    bank_dir, diagnostics, output = _run_small_condition(tmp_path)
    bank = load_evidence_bank(bank_dir)
    output["point_labels"][0] = -1
    output["instances"]["0"]["point_count"] = 3

    funnel = reconstruct_clean_stage_funnel(
        bank=bank,
        diagnostics=diagnostics,
        output=output,
        allowed_classes=load_taxonomy().canonical_classes,
    )

    assert funnel.final_equivalence is not None
    assert funnel.final_equivalence.equivalent is False
    assert funnel.final_equivalence.changed_points > 0
    assert funnel.issues


def test_stage_funnel_exposes_each_stage_to_metric_callback(tmp_path: Path) -> None:
    bank_dir, diagnostics, output = _run_small_condition(tmp_path)
    bank = load_evidence_bank(bank_dir)
    visited: list[str] = []

    def metric(stage: str, objects: tuple[object, ...]) -> dict[str, int]:
        visited.append(stage)
        return {"callback_object_count": len(objects)}

    funnel = reconstruct_clean_stage_funnel(
        bank=bank,
        diagnostics=diagnostics,
        output=output,
        allowed_classes=load_taxonomy().canonical_classes,
        metric_callback=metric,
    )

    assert tuple(visited) == STAGE_NAMES
    for stage in funnel.stages:
        assert stage.metrics["callback_object_count"] == stage.summary["object_count"]


def test_partition_equivalence_does_not_turn_one_inserted_object_into_id_cascade() -> None:
    reconstructed = (
        FunnelObject("a", np.asarray([0, 1]), class_name="chair"),
        FunnelObject("b", np.asarray([2, 3]), class_name="table"),
    )
    frozen = (
        FunnelObject("inserted", np.asarray([4]), class_name="cup"),
        FunnelObject("renumbered-a", np.asarray([0, 1]), class_name="chair"),
        FunnelObject("renumbered-b", np.asarray([2, 3]), class_name="table"),
    )

    audit = _partition_equivalence(reconstructed, frozen, point_count=5)

    assert audit.equivalent is False
    assert audit.changed_points == 1
