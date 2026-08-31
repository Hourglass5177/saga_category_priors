from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from category_priors.clean_baseline.evidence import (
    build_sparse_frame_evidence,
    save_evidence_bank,
)
from category_priors.clean_baseline.models import AlphaMaskEvidenceBank
from category_priors.clean_baseline.pipeline import run_consensus_condition
from category_priors.io import hash_json
from category_priors.taxonomy import load_taxonomy


CLASSES = (
    "chair", "table", "plant", "flower", "foliage", "tv", "painting",
    "sofa", "cabinet", "bed", "wall", "floor", "ceiling", "person",
    "socket", "remote", "key", "book", "lighting", "switch", "door",
    "window", "lamp", "speaker", "computer", "fan", "refrigerator",
    "robot", "cup", "vase", "phone", "trash can",
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
    xyz = np.asarray(
        [[0.00, 0.00, 0.00], [0.02, 0.00, 0.00], [0.04, 0.00, 0.00], [0.06, 0.00, 0.00]],
        dtype=np.float32,
    )
    bank = AlphaMaskEvidenceBank.from_frames(
        scene_id="scene-test",
        point_count=4,
        xyz_m=xyz,
        class_names=CLASSES,
        frames=[_frame(0, 0), _frame(1, 1)],
        source={"test": True},
    )
    save_evidence_bank(bank, path)


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
