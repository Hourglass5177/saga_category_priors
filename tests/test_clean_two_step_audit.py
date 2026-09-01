from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from plyfile import PlyData, PlyElement

from category_priors.clean_baseline.evidence import (
    build_sparse_frame_evidence,
    save_evidence_bank,
)
from category_priors.clean_baseline.models import AlphaMaskEvidenceBank
from category_priors.clean_baseline.pipeline import run_consensus_condition
from category_priors.clean_baseline.two_step_audit import (
    MANIFEST_SCHEMA,
    audit_clean_baseline_manifest,
)
from category_priors.io import hash_json, sha256_file
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


def _xyz() -> np.ndarray:
    grid_x, grid_y = np.meshgrid(np.arange(10), np.arange(10), indexing="ij")
    return np.column_stack(
        (grid_x.reshape(-1) * 0.004, grid_y.reshape(-1) * 0.004, np.zeros(100))
    ).astype(np.float32)


def _frame(frame_id: int, mask_id: int, point_count: int) -> object:
    ids = np.arange(point_count, dtype=np.int32)
    posterior = np.zeros((1, len(CLASSES)), dtype=np.float32)
    posterior[0, 0] = 1.0
    return build_sparse_frame_evidence(
        frame_id=frame_id,
        image_name=f"frame-{frame_id}",
        point_count=point_count,
        visible_ids=ids,
        visible_mass=np.ones(point_count, dtype=np.float32),
        mask_gaussian_ids=[ids],
        mask_inside_mass=[np.ones(point_count, dtype=np.float32)],
        mask_inside_ratio=[np.ones(point_count, dtype=np.float32)],
        semantic_posteriors=posterior,
        semantic_abstained=np.asarray([False]),
        global_mask_ids=[mask_id],
        valid_pixel_count=point_count,
        class_count=len(CLASSES),
    )


def _write_ply(path: Path, xyz: np.ndarray) -> None:
    vertex = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    vertex["x"], vertex["y"], vertex["z"] = xyz.T
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(path))


def _write_priors(path: Path) -> None:
    def node() -> dict[str, object]:
        return {
            "shrunk": {
                "geometry": {
                    "log_extent_short_m": {"q95": 10.0},
                    "log_extent_mid_m": {"q95": 10.0},
                    "log_extent_long_m": {"q95": 10.0},
                }
            }
        }

    payload = {
        "kind": "category_priors",
        "schema_version": "1.0",
        "provenance": {"splits": ["train"]},
        "global": node(),
        "categories": {},
    }
    payload["content_sha256"] = hash_json(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, list[Path]]:
    xyz = _xyz()
    bank_dir = tmp_path / "frozen" / "bank"
    bank = AlphaMaskEvidenceBank.from_frames(
        scene_id="scene-test",
        point_count=len(xyz),
        xyz_m=xyz,
        class_names=CLASSES,
        frames=[_frame(0, 0, len(xyz)), _frame(1, 1, len(xyz))],
        source={"test": "frozen-source"},
    )
    save_evidence_bank(bank, bank_dir)
    priors = tmp_path / "priors.json"
    _write_priors(priors)
    taxonomy = load_taxonomy().canonical_classes
    condition_paths: dict[str, dict[str, str]] = {}
    for condition in ("C0-no-prior", "U-global"):
        destination = tmp_path / "frozen" / condition
        run_consensus_condition(
            scene_id="scene-test",
            bank_dir=bank_dir,
            condition=condition,
            output_dir=destination,
            priors_path=None if condition == "C0-no-prior" else priors,
            allowed_classes=taxonomy,
            consumer_commit="a" * 40,
        )
        output_path = destination / "output.json"
        diagnostics_path = destination / "diagnostics.json"
        output_payload = json.loads(output_path.read_text(encoding="utf-8"))
        condition_paths[condition] = {
            "output": str(output_path),
            "diagnostics": str(diagnostics_path),
            "output_sha256": sha256_file(output_path),
            "diagnostics_sha256": sha256_file(diagnostics_path),
            "run_identity_sha256": output_payload["run_identity"][
                "content_sha256"
            ],
            "consumer_commit": "a" * 40,
        }
    gt_path = tmp_path / "frozen" / "gt.npz"
    np.savez(
        gt_path,
        coords=xyz.astype(np.float64),
        semantic=np.zeros(len(xyz), dtype=np.int64),
        instance=np.ones(len(xyz), dtype=np.int64),
    )
    gaussian_path = tmp_path / "frozen" / "gaussians.ply"
    _write_ply(gaussian_path, xyz)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "taxonomy": {
            "class_names": list(CLASSES),
            "allowed_classes": list(taxonomy),
        },
        "size_bins": {
            "boundaries_m": {
                "tiny_max_m": 0.10,
                "small_max_m": 0.20,
                "medium_max_m": 1.00,
            }
        },
        "min_region_size": 100,
        "metric_tolerance": 1e-12,
        "expected_metrics": {
            condition: {
                "official_ap25": 1.0,
                "official_ap50": 1.0,
                "historical_map_50_95": 1.0,
            }
            for condition in ("C0-no-prior", "U-global")
        },
        "scenes": [
            {
                "scene_id": "scene-test",
                "bank_dir": str(bank_dir),
                "gt_npz": str(gt_path),
                "gaussian_ply": str(gaussian_path),
                "transform": np.eye(4).tolist(),
                "source_evidence_request": {"test": True},
                "evidence_import_identity": {"test": "registered"},
                "conditions": condition_paths,
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "category_priors.clean_baseline.two_step_audit.evidence_request_source",
        lambda **_: {"test": "frozen-source"},
    )
    monkeypatch.setattr(
        "category_priors.clean_baseline.two_step_audit._audit_registered_evidence",
        lambda **_: {"scene_id": "scene-test", "passed": True, "checks": {}},
    )
    frozen_files = [
        path
        for path in (tmp_path / "frozen").rglob("*")
        if path.is_file()
    ]
    return manifest_path, frozen_files


def test_read_only_dev_audit_writes_all_products_and_passes_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, frozen_files = _fixture(tmp_path, monkeypatch)
    before = {path: sha256_file(path) for path in frozen_files}

    result = audit_clean_baseline_manifest(
        manifest_path,
        output_dir=tmp_path / "audit",
        expected_scene_count=1,
    )

    assert result["technical_gates"]["passed"] is True
    assert result["metric"]["gt_as_prediction"]["gt_as_prediction_parity"] is True
    assert result["metric"]["mapping_gate_rows"][0]["mapped_fraction_5cm"] == 1.0
    assert result["metric"]["frozen_inputs_unchanged"] is True
    assert all(
        row["passed"] for row in result["funnel"]["final_partition_gate_rows"]
    )
    assert before == {path: sha256_file(path) for path in frozen_files}
    products = {
        "clean_metric_reaudit_dev8.json",
        "clean_metric_reaudit_dev8.parquet",
        "clean_stage_funnel_dev8.json",
        "clean_stage_funnel_dev8.parquet",
    }
    assert products == {path.name for path in (tmp_path / "audit").iterdir()}
    metric_table = pd.read_parquet(
        tmp_path / "audit" / "clean_metric_reaudit_dev8.parquet"
    )
    funnel_table = pd.read_parquet(
        tmp_path / "audit" / "clean_stage_funnel_dev8.parquet"
    )
    assert set(metric_table["condition"]) == {"C0-no-prior", "U-global"}
    assert set(funnel_table["stage"]) == {
        "complete_mask_support",
        "association_support",
        "undersegmentation_filtered",
        "accepted_edge_components",
        "detection_ratio_filtered",
        "physical_split_and_deduplicated",
        "unique_gaussian_ownership",
        "final_export",
    }


def test_manifest_must_supply_old_metrics_instead_of_guessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, _ = _fixture(tmp_path, monkeypatch)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    del payload["expected_metrics"]["U-global"]["historical_map_50_95"]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="omitted"):
        audit_clean_baseline_manifest(
            manifest_path,
            output_dir=tmp_path / "audit",
            expected_scene_count=1,
        )


def test_audit_rejects_forged_legacy_hierarchy_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, _ = _fixture(tmp_path, monkeypatch)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    unknown_producer = "b" * 40
    payload["scenes"][0]["evidence_import_identity"] = {
        "producer_commit": unknown_producer,
        "legacy_hierarchy_mode_proof": {
            "producer_commit": unknown_producer,
            "assumed_mode": "hierarchy",
            "missing_fields": ["mask_observation_mode"],
        },
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        "category_priors.clean_baseline.two_step_audit.evidence_request_source",
        lambda **_: {
            "test": "frozen-source",
            "mask_observation_mode": "hierarchy",
        },
    )

    with pytest.raises(ValueError, match="not allowlisted"):
        audit_clean_baseline_manifest(
            manifest_path,
            output_dir=tmp_path / "audit",
            expected_scene_count=1,
        )


def test_audit_rejects_output_inside_frozen_artifact_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, _ = _fixture(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="frozen"):
        audit_clean_baseline_manifest(
            manifest_path,
            output_dir=tmp_path / "frozen" / "bank" / "audit",
            expected_scene_count=1,
        )


def test_manifest_may_be_colocated_with_registered_audit_products(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, _ = _fixture(tmp_path, monkeypatch)
    artifact_root = tmp_path / "registered-audit"
    artifact_root.mkdir()
    colocated = artifact_root / "manifest.json"
    colocated.write_bytes(manifest_path.read_bytes())

    result = audit_clean_baseline_manifest(
        colocated,
        output_dir=artifact_root,
        expected_scene_count=1,
    )

    assert result["technical_gates"]["passed"] is True
