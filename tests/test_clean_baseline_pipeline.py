from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from category_priors.clean_baseline.evidence import (
    build_sparse_frame_evidence,
    save_evidence_bank,
)
from category_priors.clean_baseline.models import AlphaMaskEvidenceBank
from category_priors.clean_baseline.consensus import ConsensusEdge
from category_priors.clean_baseline.pipeline import (
    _final_support_statistics,
    run_consensus_condition,
)
from category_priors.io import hash_json
from category_priors.taxonomy import load_taxonomy


CLASSES = (
    "chair", "table", "plant", "flower", "foliage", "tv", "painting",
    "sofa", "cabinet", "bed", "wall", "floor", "ceiling", "person",
    "socket", "remote", "key", "book", "lighting", "switch", "door",
    "window", "lamp", "speaker", "computer", "fan", "refrigerator",
    "robot", "cup", "vase", "phone", "trash can",
)


def _frame(frame_id: int, mask_id: int, *, semantic_class_index: int = 0) -> object:
    ids = np.arange(4, dtype=np.int32)
    posterior = np.zeros((1, len(CLASSES)), dtype=np.float32)
    posterior[0, int(semantic_class_index)] = 1.0
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


def _geometry_abstained_frame(frame_id: int) -> object:
    ids = np.arange(4, dtype=np.int32)
    return build_sparse_frame_evidence(
        frame_id=frame_id,
        image_name=f"frame-{frame_id}",
        point_count=4,
        visible_ids=ids,
        visible_mass=np.ones(4, dtype=np.float32),
        mask_gaussian_ids=[],
        mask_inside_mass=[],
        mask_inside_ratio=[],
        semantic_posteriors=np.empty((0, len(CLASSES)), dtype=np.float32),
        semantic_abstained=np.empty(0, dtype=bool),
        valid_pixel_count=4,
        geometry_abstained=True,
        class_count=len(CLASSES),
    )


def _write_bank(path: Path, *, semantic_class_index: int = 0) -> None:
    xyz = np.asarray(
        [[0.00, 0.00, 0.00], [0.02, 0.00, 0.00], [0.04, 0.00, 0.00], [0.06, 0.00, 0.00]],
        dtype=np.float32,
    )
    bank = AlphaMaskEvidenceBank.from_frames(
        scene_id="scene-test",
        point_count=4,
        xyz_m=xyz,
        class_names=CLASSES,
        frames=[
            _frame(0, 0, semantic_class_index=semantic_class_index),
            _frame(1, 1, semantic_class_index=semantic_class_index),
        ],
        source={"test": True},
    )
    save_evidence_bank(bank, path)


def _write_bank_with_geometry_abstention(path: Path) -> None:
    xyz = np.asarray(
        [[0.00, 0.00, 0.00], [0.02, 0.00, 0.00], [0.04, 0.00, 0.00], [0.06, 0.00, 0.00]],
        dtype=np.float32,
    )
    bank = AlphaMaskEvidenceBank.from_frames(
        scene_id="scene-test",
        point_count=4,
        xyz_m=xyz,
        class_names=CLASSES,
        frames=[_frame(0, 0), _frame(1, 1), _geometry_abstained_frame(2)],
        source={"test": True},
    )
    save_evidence_bank(bank, path)


def test_final_ownership_recomputes_detection_ratio_for_retained_gaussians() -> None:
    def frame(frame_id: int, mask_id: int, support: np.ndarray) -> object:
        ids = np.arange(4, dtype=np.int32)
        posterior = np.zeros((1, len(CLASSES)), dtype=np.float32)
        posterior[0, 0] = 1.0
        selected = np.asarray(support, dtype=np.int32)
        return build_sparse_frame_evidence(
            frame_id=frame_id,
            image_name=f"frame-{frame_id}",
            point_count=4,
            visible_ids=ids,
            visible_mass=np.ones(4, dtype=np.float32),
            mask_gaussian_ids=[selected],
            mask_inside_mass=[np.ones(len(selected), dtype=np.float32)],
            mask_inside_ratio=[np.ones(len(selected), dtype=np.float32)],
            semantic_posteriors=posterior,
            semantic_abstained=np.asarray([False]),
            global_mask_ids=[mask_id],
            valid_pixel_count=4,
            class_count=len(CLASSES),
        )

    bank = AlphaMaskEvidenceBank.from_frames(
        scene_id="scene-test",
        point_count=4,
        xyz_m=np.column_stack((np.arange(4) * 0.01, np.zeros((4, 2)))).astype(
            np.float32
        ),
        class_names=CLASSES,
        frames=[
            frame(0, 0, np.arange(4)),
            frame(1, 1, np.arange(4)),
            frame(2, 2, np.asarray([0, 1])),
        ],
        source={"test": True},
    )
    edges = (
        # Mask 2 belonged to the accepted parent edge, but supports only the
        # region removed by final ownership.  The surviving 0--1 sides must
        # still be recomputed rather than dropping the whole historical edge.
        ConsensusEdge((0, 2), (1,), 3, 3, 1.0, 3),
    )
    masks, frames, consensus, detection = _final_support_statistics(
        bank=bank,
        mask_ids=(0, 1, 2),
        gaussian_ids=np.asarray([2, 3]),
        accepted_edges=edges,
    )
    assert masks == (0, 1)
    assert frames == (0, 1)
    # The stored edge value (1.0) described the four-point parent support.
    # On the retained part, frame 2 still observes the points but no retained
    # mask supports both sides, so final Q must use the recomputed 2/3 value.
    assert consensus == pytest.approx(2 / 3)
    # The third frame sees these points but its retained mask does not detect
    # them.  Keeping the pre-ownership value of 1.0 would overstate Q.
    assert detection == pytest.approx(2 / 3)


def test_geometry_abstained_frame_is_not_negative_consensus_evidence(
    tmp_path: Path,
) -> None:
    bank_dir = tmp_path / "bank"
    _write_bank_with_geometry_abstention(bank_dir)

    output = tmp_path / "out"
    result = run_consensus_condition(
        scene_id="scene-test",
        bank_dir=bank_dir,
        condition="C0-no-prior",
        output_dir=output,
        allowed_classes=load_taxonomy().canonical_classes,
        consumer_commit="a" * 40,
    )

    payload = json.loads((output / "output.json").read_text("utf-8"))
    assert result["consensus"]["object_count"] == 1
    assert payload["point_labels"] == [0, 0, 0, 0]


def _write_priors(path: Path) -> None:
    def node(short: float, mid: float, long: float) -> dict[str, object]:
        return {
            "shrunk": {
                "geometry": {
                    "log_extent_short_m": {"q95": short},
                    "log_extent_mid_m": {"q95": mid},
                    "log_extent_long_m": {"q95": long},
                }
            }
        }

    payload = {
        "kind": "category_priors",
        "schema_version": "1.0",
        "provenance": {"splits": ["train"]},
        "global": node(0.0, 0.0, 0.0),
        "categories": {"chair": node(-30.0, -30.0, -30.0)},
    }
    payload["content_sha256"] = hash_json(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_same_bank_runs_no_prior_global_and_predicted_size(tmp_path: Path) -> None:
    bank_dir = tmp_path / "bank"
    _write_bank(bank_dir)
    priors = tmp_path / "priors.json"
    _write_priors(priors)

    c0 = tmp_path / "c0"
    uniform = tmp_path / "u"
    data = tmp_path / "d"
    allowed = load_taxonomy().canonical_classes
    commit = "a" * 40
    run_consensus_condition(
        scene_id="scene-test",
        bank_dir=bank_dir,
        condition="C0-no-prior",
        output_dir=c0,
        allowed_classes=allowed,
        consumer_commit=commit,
    )
    run_consensus_condition(
        scene_id="scene-test",
        bank_dir=bank_dir,
        condition="U-global",
        output_dir=uniform,
        priors_path=priors,
        allowed_classes=allowed,
        consumer_commit=commit,
    )
    run_consensus_condition(
        scene_id="scene-test",
        bank_dir=bank_dir,
        condition="D-predicted",
        output_dir=data,
        priors_path=priors,
        allowed_classes=allowed,
        consumer_commit=commit,
    )

    c0_output = json.loads((c0 / "output.json").read_text("utf-8"))
    u_output = json.loads((uniform / "output.json").read_text("utf-8"))
    d_output = json.loads((data / "output.json").read_text("utf-8"))
    assert len(c0_output["instances"]) == 1
    assert c0_output["point_labels"] == u_output["point_labels"] == [0, 0, 0, 0]
    assert d_output["instances"] == {}
    assert d_output["point_labels"] == [-1, -1, -1, -1]

    u_diag = json.loads((uniform / "diagnostics.json").read_text("utf-8"))
    d_diag = json.loads((data / "diagnostics.json").read_text("utf-8"))
    assert u_diag["prior_in_ap_score"] is False
    assert d_diag["prior_in_ap_score"] is False
    assert (
        u_diag["consensus"]["raw_graph_identity"]
        == d_diag["consensus"]["raw_graph_identity"]
    )
    assert u_diag["size_merge_decisions"][0]["accepted"] is True
    assert d_diag["size_merge_decisions"][0]["accepted"] is False


def test_c0_geometry_is_invariant_to_semantic_posteriors(tmp_path: Path) -> None:
    chair_bank = tmp_path / "chair-bank"
    table_bank = tmp_path / "table-bank"
    _write_bank(chair_bank, semantic_class_index=0)
    _write_bank(table_bank, semantic_class_index=1)
    allowed = load_taxonomy().canonical_classes

    chair = run_consensus_condition(
        scene_id="scene-test",
        bank_dir=chair_bank,
        condition="C0-no-prior",
        output_dir=tmp_path / "chair-output",
        allowed_classes=allowed,
        consumer_commit="a" * 40,
    )
    table = run_consensus_condition(
        scene_id="scene-test",
        bank_dir=table_bank,
        condition="C0-no-prior",
        output_dir=tmp_path / "table-output",
        allowed_classes=allowed,
        consumer_commit="a" * 40,
    )
    chair_output = json.loads(
        (tmp_path / "chair-output" / "output.json").read_text("utf-8")
    )
    table_output = json.loads(
        (tmp_path / "table-output" / "output.json").read_text("utf-8")
    )

    assert chair["consensus"]["raw_graph_identity"] == table["consensus"][
        "raw_graph_identity"
    ]
    assert chair["accepted_edges"] == table["accepted_edges"]
    assert chair["rejected_undersegmented_mask_ids"] == table[
        "rejected_undersegmented_mask_ids"
    ]
    assert chair_output["point_labels"] == table_output["point_labels"]
    assert chair_output["instances"]["0"]["class"] == "chair"
    assert table_output["instances"]["0"]["class"] == "table"


def test_c0_rejects_prior_or_gt_runtime_inputs(tmp_path: Path) -> None:
    bank_dir = tmp_path / "bank"
    _write_bank(bank_dir)
    priors = tmp_path / "priors.json"
    _write_priors(priors)
    shared = {
        "scene_id": "scene-test",
        "bank_dir": bank_dir,
        "condition": "C0-no-prior",
        "output_dir": tmp_path / "output",
        "allowed_classes": load_taxonomy().canonical_classes,
        "consumer_commit": "a" * 40,
    }
    with pytest.raises(ValueError, match="forbids a category-prior input"):
        run_consensus_condition(**shared, priors_path=priors)
    with pytest.raises(TypeError):
        run_consensus_condition(**shared, gt_path=tmp_path / "gt.npz")


def test_prediction_resume_is_bound_to_commit_bank_and_diagnostics(
    tmp_path: Path,
) -> None:
    bank_dir = tmp_path / "bank"
    _write_bank(bank_dir)
    output = tmp_path / "output"
    allowed = load_taxonomy().canonical_classes
    first = run_consensus_condition(
        scene_id="scene-test",
        bank_dir=bank_dir,
        condition="C0-no-prior",
        output_dir=output,
        allowed_classes=allowed,
        consumer_commit="a" * 40,
    )
    assert first["runner_status"] == "complete"
    second = run_consensus_condition(
        scene_id="scene-test",
        bank_dir=bank_dir,
        condition="C0-no-prior",
        output_dir=output,
        allowed_classes=allowed,
        consumer_commit="a" * 40,
    )
    assert second["runner_status"] == "skipped-complete"

    changed = run_consensus_condition(
        scene_id="scene-test",
        bank_dir=bank_dir,
        condition="C0-no-prior",
        output_dir=output,
        allowed_classes=allowed,
        consumer_commit="b" * 40,
    )
    assert changed["runner_status"] == "complete"
    payload = json.loads((output / "output.json").read_text("utf-8"))
    assert payload["run_identity"]["consumer_commit"] == "b" * 40

    diagnostics = json.loads((output / "diagnostics.json").read_text("utf-8"))
    diagnostics["run_identity"]["consumer_commit"] = "corrupt"
    (output / "diagnostics.json").write_text(json.dumps(diagnostics), encoding="utf-8")
    repaired = run_consensus_condition(
        scene_id="scene-test",
        bank_dir=bank_dir,
        condition="C0-no-prior",
        output_dir=output,
        allowed_classes=allowed,
        consumer_commit="b" * 40,
    )
    assert repaired["runner_status"] == "complete"

    complete_diagnostics = json.loads(
        (output / "diagnostics.json").read_text("utf-8")
    )
    (output / "diagnostics.json").write_text(
        json.dumps({"run_identity": complete_diagnostics["run_identity"]}),
        encoding="utf-8",
    )
    rebuilt_sidecar = run_consensus_condition(
        scene_id="scene-test",
        bank_dir=bank_dir,
        condition="C0-no-prior",
        output_dir=output,
        allowed_classes=allowed,
        consumer_commit="b" * 40,
    )
    assert rebuilt_sidecar["runner_status"] == "complete"
    restored = json.loads((output / "diagnostics.json").read_text("utf-8"))
    assert restored["schema"] == "saga-clean-alpha-mask-condition-diagnostics-v1"
    assert "consensus" in restored
    assert "objects" in restored
    assert "prediction_contract" in restored

    complete_again = run_consensus_condition(
        scene_id="scene-test",
        bank_dir=bank_dir,
        condition="C0-no-prior",
        output_dir=output,
        allowed_classes=allowed,
        consumer_commit="b" * 40,
    )
    assert complete_again["runner_status"] == "skipped-complete"
