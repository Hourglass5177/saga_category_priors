from __future__ import annotations

import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from category_priors.clean_baseline import mask_ablation
from category_priors.clean_baseline import late_filter_experiment
from category_priors.clean_baseline.consensus import ConsensusConfig
from category_priors.clean_baseline.evidence import (
    EVIDENCE_ARRAY_FILE,
    EVIDENCE_DIAGNOSTICS_FILE,
    EVIDENCE_METADATA_FILE,
    build_sparse_frame_evidence,
    path_content_identity,
    save_evidence_bank,
)
from category_priors.clean_baseline.evaluation import RUN_IDENTITY_SCHEMA
from category_priors.clean_baseline.models import AlphaMaskEvidenceBank
from category_priors.io import hash_json, sha256_file


CLASSES = ("chair",) + tuple(f"unused-{index}" for index in range(31))
SCENE_A, SCENE_B = mask_ablation.REGISTERED_DEV2_SCENE_IDS


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _xyz() -> np.ndarray:
    rows: list[list[float]] = []
    for object_index in range(3):
        for point_index in range(100):
            rows.append(
                [
                    object_index * 1.0 + (point_index % 10) * 0.005,
                    (point_index // 10) * 0.005,
                    0.0,
                ]
            )
    return np.asarray(rows, dtype=np.float32)


def _bank(
    scene_id: str, *, mode: str, xyz: np.ndarray, gaussian_path: Path
) -> AlphaMaskEvidenceBank:
    frames = []
    all_ids = np.arange(len(xyz), dtype=np.int32)
    for frame_id in range(2):
        mask_rows = [
            np.arange(index * 100, (index + 1) * 100, dtype=np.int32)
            for index in range(3)
        ]
        posterior = np.zeros((3, len(CLASSES)), dtype=np.float32)
        posterior[:, 0] = 1.0
        frames.append(
            build_sparse_frame_evidence(
                frame_id=frame_id,
                image_name=f"frame-{frame_id}",
                point_count=len(xyz),
                visible_ids=all_ids,
                visible_mass=np.ones(len(xyz), dtype=np.float32),
                mask_gaussian_ids=mask_rows,
                mask_inside_mass=[np.ones(100, dtype=np.float32) for _ in mask_rows],
                mask_inside_ratio=[np.ones(100, dtype=np.float32) for _ in mask_rows],
                semantic_posteriors=posterior,
                global_mask_id_start=frame_id * 3,
                mask_indices=[0, 1, 2],
                valid_pixel_count=1000,
                class_count=len(CLASSES),
            )
        )
    source = {
        "worker": "test",
        "producer_commit": "a" * 40,
        "rgb_ply": str(gaussian_path.resolve()),
        "scene_scale_m_per_unit": 1.0,
        "sam_masks": f"/masks/{mode}",
        "mask_observation_mode": mode,
        "producer_inputs": {
            "sam_everything_masks": {
                "path": f"/masks/{mode}",
                "kind": "registered-sam-masks",
                "file_count": 2,
                "relative_paths": ["frame-0.npz", "frame-1.npz"],
                "manifest_sha256": ("a" if mode == "hierarchy" else "b") * 64,
            },
            "rgb": "same",
            "gaussian_ply": path_content_identity(gaussian_path),
        },
        "shared": "same",
    }
    return AlphaMaskEvidenceBank.from_frames(
        scene_id=scene_id,
        point_count=len(xyz),
        xyz_m=xyz,
        class_names=CLASSES,
        frames=frames,
        source=source,
    )


def _run_identity(
    scene_id: str, bank: AlphaMaskEvidenceBank, bank_dir: Path
) -> dict[str, object]:
    identity: dict[str, object] = {
        "schema": RUN_IDENTITY_SCHEMA,
        "consumer_commit": "a" * 40,
        "scene_id": scene_id,
        "condition": "C0-no-prior",
        "evidence": {
            "schema": bank.schema,
            "scene_id": bank.scene_id,
            "point_count": bank.point_count,
            "frame_count": bank.frame_count,
            "mask_count": bank.mask_count,
            "thresholds": bank.thresholds.to_dict(),
            "source": dict(bank.source),
            "class_names": list(bank.class_names),
            "files": {
                name: sha256_file(bank_dir / name)
                for name in (
                    EVIDENCE_ARRAY_FILE,
                    EVIDENCE_DIAGNOSTICS_FILE,
                    EVIDENCE_METADATA_FILE,
                )
            },
        },
        "consensus_config": asdict(ConsensusConfig()),
        "taxonomy": {"content_sha256": "test", "allowed_classes": ["chair"]},
        "ap_score": {
            "formula": "winner_probability*sqrt(view_consensus*detection_ratio)",
            "prior_in_score": False,
        },
        "prior": None,
    }
    identity["content_sha256"] = hash_json(identity)
    return identity


def _output(
    scene_id: str,
    kept_objects: list[int],
    *,
    run_identity: dict[str, object],
) -> dict[str, object]:
    labels = np.full(300, -1, dtype=np.int64)
    instances: dict[str, dict[str, object]] = {}
    for export_id, object_index in enumerate(kept_objects):
        labels[object_index * 100 : (object_index + 1) * 100] = export_id
        instances[str(export_id)] = {
            "class": "chair",
            "score": 1.0,
            "winner_probability": 1.0,
            "view_consensus": 1.0,
            "detection_ratio": 1.0,
        }
    return {
        "schema": "saga-clean-alpha-mask-prediction-v2",
        "scene_id": scene_id,
        "condition": "C0-no-prior",
        "point_labels": labels.tolist(),
        "instances": instances,
        "run_identity": run_identity,
    }


def _diagnostics(
    scene_id: str,
    kept_objects: list[int],
    *,
    run_identity: dict[str, object],
    bank_schema: str,
) -> dict[str, object]:
    kept = tuple(sorted(set(map(int, kept_objects))))
    rejected = [
        frame_id * 3 + object_index
        for frame_id in range(2)
        for object_index in range(3)
        if object_index not in kept
    ]
    accepted_edges = [
        {
            "left_mask_ids": [object_index],
            "right_mask_ids": [object_index + 3],
            "observer_count": 2,
            "supporter_count": 2,
            "consensus": 1.0,
            "observer_level": 2,
        }
        for object_index in kept
    ]
    return {
        "schema": "saga-clean-alpha-mask-condition-diagnostics-v1",
        "scene_id": scene_id,
        "condition": "C0-no-prior",
        "run_identity": run_identity,
        "bank_schema": bank_schema,
        "config": asdict(ConsensusConfig()),
        "consensus": {
            "component_count_before_output_filters": len(kept),
        },
        "accepted_edges": accepted_edges,
        "rejected_undersegmented_mask_ids": rejected,
        "prediction_contract": {
            "scene_id": scene_id,
            "condition": "C0-no-prior",
            "run_identity": run_identity,
            "contract": {
                "orphan_gaussian_count": 0,
                "negative_metadata_count": 0,
                "duplicate_ownership_count": 0,
            }
        },
    }


def _scene(
    tmp_path: Path,
    scene_id: str,
    *,
    hierarchy_objects: list[int] | None = None,
    flat_objects: list[int] | None = None,
    include_input_audit: bool = True,
    repeat_identity: bool | None = True,
    include_full_repeat: bool = True,
) -> dict[str, object]:
    xyz = _xyz()
    gt_path = tmp_path / scene_id / "gt.npz"
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        gt_path,
        coords=xyz,
        semantic=np.zeros(len(xyz), dtype=np.int64),
        instance=np.repeat(np.arange(3, dtype=np.int64), 100),
    )
    gaussian_path = tmp_path / scene_id / "gaussian.ply"
    gaussian_path.write_bytes(b"test-double")
    conditions: dict[str, object] = {}
    banks: dict[str, AlphaMaskEvidenceBank] = {}
    for arm, mode, objects in (
        ("H-hierarchy", "hierarchy", hierarchy_objects or [0, 1]),
        ("P-flat", "flat-highest-quality", flat_objects or [0, 1, 2]),
    ):
        root = tmp_path / scene_id / arm
        bank_dir = root / "bank"
        bank = _bank(
            scene_id, mode=mode, xyz=xyz, gaussian_path=gaussian_path
        )
        banks[arm] = bank
        save_evidence_bank(bank, bank_dir)
        run_identity = _run_identity(scene_id, bank, bank_dir)
        condition_root = root / "C0-no-prior"
        output_path = condition_root / "output.json"
        diagnostics_path = condition_root / "diagnostics.json"
        _write_json(
            output_path,
            _output(scene_id, objects, run_identity=run_identity),
        )
        _write_json(
            diagnostics_path,
            _diagnostics(
                scene_id,
                objects,
                run_identity=run_identity,
                bank_schema=bank.schema,
            ),
        )
        conditions[arm] = {
            "bank_dir": str(bank_dir),
            "output": str(output_path),
            "diagnostics": str(diagnostics_path),
        }
    result: dict[str, object] = {
        "scene_id": scene_id,
        "gt_npz": str(gt_path),
        "gaussian_ply": str(gaussian_path),
        "transform": np.eye(4).tolist(),
        "mask_control_conditions": conditions,
    }
    if include_input_audit:
        bindings: dict[str, object] = {}
        for arm, mode in (
            ("H-hierarchy", "hierarchy"),
            ("P-flat", "flat-highest-quality"),
        ):
            bank = banks[arm]
            request_path = tmp_path / scene_id / f"{arm}-request.json"
            request_payload = {
                "scene": {"scene_id": scene_id},
                "producer_commit": "a" * 40,
                "sam_masks": bank.source["sam_masks"],
                "mask_observation_mode": mode,
            }
            _write_json(request_path, request_payload)
            bindings[arm] = {
                "arm": arm,
                "scene_id": scene_id,
                "mask_root": bank.source["sam_masks"],
                "mask_manifest": bank.source["producer_inputs"][
                    "sam_everything_masks"
                ],
                "evidence_request": {
                    "path": str(request_path.resolve()),
                    "size": request_path.stat().st_size,
                    "sha256": sha256_file(request_path),
                },
                "expected_bank_source": bank.source,
            }
        result["flat_mask_input_audit"] = {
            "scene_id": scene_id,
            "mechanical_contract_pass": True,
            "union_changed_pixel_count": 0,
            "flat_overlap_pixel_count": 0,
            "input_binding_pass": True,
            "flat_repeat_identity_pass": repeat_identity is True,
            "repeat_input_manifest_before": [],
            "repeat_input_manifest_after": [],
            "stranded_part_files": [],
            "input_bindings": bindings,
        }
    if repeat_identity is not None:
        result["flat_repeat_identity_pass"] = repeat_identity
    if include_full_repeat:
        primary = conditions["P-flat"]
        repeat_root = tmp_path / scene_id / "P-flat-repeat"
        repeat_bank = repeat_root / "bank"
        repeat_condition = repeat_root / "C0-no-prior"
        shutil.copytree(Path(primary["bank_dir"]), repeat_bank)
        repeat_condition.mkdir(parents=True)
        shutil.copy2(Path(primary["output"]), repeat_condition / "output.json")
        shutil.copy2(
            Path(primary["diagnostics"]), repeat_condition / "diagnostics.json"
        )
        bank_left = mask_ablation._tree_byte_manifest(
            Path(primary["bank_dir"]),
            expected_relative_paths=("diagnostics.json", "evidence.npz", "masks.json"),
        )
        bank_right = mask_ablation._tree_byte_manifest(
            repeat_bank,
            expected_relative_paths=("diagnostics.json", "evidence.npz", "masks.json"),
        )
        condition_left = mask_ablation._tree_byte_manifest(
            Path(primary["output"]).parent,
            expected_relative_paths=("diagnostics.json", "output.json"),
        )
        condition_right = mask_ablation._tree_byte_manifest(
            repeat_condition,
            expected_relative_paths=("diagnostics.json", "output.json"),
        )
        bank_identity = {
            "passed": True,
            "primary": bank_left,
            "repeat": bank_right,
        }
        condition_identity = {
            "passed": True,
            "primary": condition_left,
            "repeat": condition_right,
        }
        repeat_audit = {
            "schema": mask_ablation.FLAT_REPEAT_SCHEMA,
            "scene_id": scene_id,
            "primary": {
                "bank_dir": str(Path(primary["bank_dir"]).resolve()),
                "output": str(Path(primary["output"]).resolve()),
                "diagnostics": str(Path(primary["diagnostics"]).resolve()),
            },
            "repeat": {
                "bank_dir": str(repeat_bank.resolve()),
                "output": str((repeat_condition / "output.json").resolve()),
                "diagnostics": str((repeat_condition / "diagnostics.json").resolve()),
            },
            "bank_byte_identity": bank_identity,
            "condition_byte_identity": condition_identity,
            "passed": True,
        }
        audit_path = repeat_root / "flat_full_repeat_audit.json"
        _write_json(audit_path, repeat_audit)
        result["flat_full_repeat_audit"] = str(audit_path)
    return result


def _manifest(tmp_path: Path, scenes: list[dict[str, object]]) -> Path:
    path = tmp_path / "manifest.json"
    _write_json(
        path,
        {
            "schema": "saga-clean-mask-contract-manifest-v1",
            "min_region_size": 100,
            "taxonomy": {
                "class_names": list(CLASSES),
                "allowed_classes": ["chair"],
            },
            "size_bins": {
                "boundaries_m": {
                    "tiny_max_m": 0.1,
                    "small_max_m": 0.2,
                    "medium_max_m": 0.4,
                }
            },
            "dev2_scene_ids": [SCENE_A, SCENE_B],
            "scenes": scenes,
        },
    )
    return path


def test_registered_two_scene_gate_passes_and_writes_flat_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenes = [_scene(tmp_path, SCENE_A), _scene(tmp_path, SCENE_B)]
    xyz = _xyz()
    monkeypatch.setattr(mask_ablation, "load_ply_xyz", lambda _: xyz.copy())

    result = mask_ablation.evaluate_mask_contract_ablation_manifest(
        _manifest(tmp_path, scenes), tmp_path / "analysis"
    )

    analysis = result["analysis"]
    assert analysis["mechanical_gate"]["passed"] is True
    assert analysis["scientific_gate"]["passed"] is True
    assert analysis["conclusion"] == "flat-mask-contract-passed"
    assert analysis["aggregate"]["H-hierarchy"]["official_geometry_050_tp"] == 4
    assert analysis["aggregate"]["P-flat"]["official_geometry_050_tp"] == 6
    assert analysis["aggregate"]["P-flat"][
        "official_geometry_025_tiny_small_recall"
    ] == 1.0
    assert analysis["gt_as_prediction"]["gt_as_prediction_parity"] is True
    assert analysis["category_prior_tested"] is False
    assert analysis["affinity_feature_used_for_geometric_association"] is False
    assert analysis["geometric_identity_unit"] == "complete-frame-mask-observation"
    table = pd.read_parquet(
        tmp_path / "analysis" / "mask_contract_ablation_dev2.parquet"
    )
    assert len(table) == 38
    assert set(table["scope"]) == {"scene", "stage", "aggregate"}
    stage_rows = table[table["scope"] == "stage"]
    assert len(stage_rows) == 2 * 2 * 8
    assert set(stage_rows["stage"]) == {
        "complete_mask_support",
        "association_support",
        "undersegmentation_filtered",
        "accepted_edge_components",
        "detection_ratio_filtered",
        "physical_split_and_deduplicated",
        "unique_gaussian_ownership",
        "final_export",
    }
    assert "all_geometry_025_fp" in table.columns
    assert "geometric_pollution_fraction_5cm" in table.columns
    preclassification = stage_rows[stage_rows["stage"] != "final_export"]
    classified = stage_rows[stage_rows["stage"] == "final_export"]
    assert set(preclassification["semantic_classification_applied"]) == {False}
    assert preclassification["all_same_class_025_match_count"].isna().all()
    assert preclassification["official_evaluable_same_class_050_precision"].isna().all()
    assert set(classified["semantic_classification_applied"]) == {True}
    assert classified["all_same_class_025_match_count"].notna().all()
    assert classified["official_evaluable_same_class_050_precision"].notna().all()
    for scene_id in (SCENE_A, SCENE_B):
        for arm in mask_ablation.MASK_ARMS:
            arm_result = analysis["scene_analysis"][scene_id]["arms"][arm]
            assert arm_result["stage_funnel_diagnostics_schema_current"] is True
            assert arm_result["stage_funnel_lineage_complete"] is True
            assert arm_result["stage_funnel_final_equivalence_exact"] is True
            assert arm_result["stage_funnel_contract_pass"] is True


@pytest.mark.parametrize(
    "outer_schema",
    [
        "saga-clean-evidence-input-content-v1",
        "saga-clean-evidence-input-content-v2",
    ],
)
def test_gaussian_binding_uses_the_producer_content_identity_schema(
    tmp_path: Path, outer_schema: str
) -> None:
    gaussian_path = tmp_path / "gaussian.ply"
    gaussian_path.write_bytes(b"canonical-gaussian-content")
    bank = _bank(
        SCENE_A, mode="hierarchy", xyz=_xyz(), gaussian_path=gaussian_path
    )
    source = dict(bank.source)
    producer_inputs = dict(source["producer_inputs"])
    producer_inputs["schema"] = outer_schema
    source["producer_inputs"] = producer_inputs
    bank = replace(bank, source=source)

    contract = mask_ablation._gaussian_binding_contract(
        bank=bank,
        gaussian_path=gaussian_path,
        gaussian_raw=_xyz(),
    )

    assert contract["passed"] is True
    assert contract["checks"]["embedded_ply_content_identity_matches"] is True
    assert contract["registered_content_identity"] == path_content_identity(
        gaussian_path
    )
    assert contract["content_identity_mismatch_fields"] == []


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("manifest_sha256", "0" * 64),
        ("total_bytes", 999),
        ("relative_paths", ["different.ply"]),
    ],
)
def test_gaussian_binding_rejects_any_registered_content_identity_drift(
    tmp_path: Path, field: str, replacement: object
) -> None:
    gaussian_path = tmp_path / "gaussian.ply"
    gaussian_path.write_bytes(b"canonical-gaussian-content")
    bank = _bank(
        SCENE_A, mode="hierarchy", xyz=_xyz(), gaussian_path=gaussian_path
    )
    source = dict(bank.source)
    producer_inputs = dict(source["producer_inputs"])
    registered = dict(producer_inputs["gaussian_ply"])
    registered[field] = replacement
    producer_inputs["gaussian_ply"] = registered
    source["producer_inputs"] = producer_inputs
    bank = replace(bank, source=source)

    contract = mask_ablation._gaussian_binding_contract(
        bank=bank,
        gaussian_path=gaussian_path,
        gaussian_raw=_xyz(),
    )

    assert contract["passed"] is False
    assert contract["checks"]["embedded_ply_content_identity_matches"] is False
    assert field in contract["content_identity_mismatch_fields"]


def test_gaussian_binding_rejects_legacy_size_sha_fixture(
    tmp_path: Path,
) -> None:
    gaussian_path = tmp_path / "gaussian.ply"
    gaussian_path.write_bytes(b"canonical-gaussian-content")
    bank = _bank(
        SCENE_A, mode="hierarchy", xyz=_xyz(), gaussian_path=gaussian_path
    )
    source = dict(bank.source)
    producer_inputs = dict(source["producer_inputs"])
    producer_inputs["gaussian_ply"] = {
        "path": str(gaussian_path.resolve()),
        "size": gaussian_path.stat().st_size,
        "sha256": sha256_file(gaussian_path),
    }
    source["producer_inputs"] = producer_inputs
    bank = replace(bank, source=source)

    contract = mask_ablation._gaussian_binding_contract(
        bank=bank,
        gaussian_path=gaussian_path,
        gaussian_raw=_xyz(),
    )

    assert contract["passed"] is False
    assert contract["checks"]["embedded_ply_content_identity_matches"] is False
    assert "size" in contract["content_identity_mismatch_fields"]
    assert "manifest_sha256" in contract["content_identity_mismatch_fields"]


def test_gaussian_binding_rejects_same_path_after_file_bytes_change(
    tmp_path: Path,
) -> None:
    gaussian_path = tmp_path / "gaussian.ply"
    gaussian_path.write_bytes(b"original-gaussian-content")
    bank = _bank(
        SCENE_A, mode="hierarchy", xyz=_xyz(), gaussian_path=gaussian_path
    )
    gaussian_path.write_bytes(b"different-gaussian-content")

    contract = mask_ablation._gaussian_binding_contract(
        bank=bank,
        gaussian_path=gaussian_path,
        gaussian_raw=_xyz(),
    )

    assert contract["passed"] is False
    assert contract["checks"]["source_rgb_ply_matches_manifest"] is True
    assert contract["checks"]["bank_xyz_matches_raw_ply_times_scale"] is True
    assert contract["checks"]["embedded_ply_content_identity_matches"] is False
    assert "manifest_sha256" in contract["content_identity_mismatch_fields"]


@pytest.mark.parametrize("schema", [None, "unknown-condition-diagnostics-v999"])
def test_missing_or_unknown_diagnostics_schema_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schema: str | None,
) -> None:
    scenes = [_scene(tmp_path, SCENE_A), _scene(tmp_path, SCENE_B)]
    spec = scenes[0]["mask_control_conditions"]["H-hierarchy"]
    diagnostics_path = Path(spec["diagnostics"])
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    if schema is None:
        diagnostics.pop("schema")
    else:
        diagnostics["schema"] = schema
    _write_json(diagnostics_path, diagnostics)
    monkeypatch.setattr(mask_ablation, "load_ply_xyz", lambda _: _xyz())

    result = mask_ablation.evaluate_mask_contract_ablation_manifest(
        _manifest(tmp_path, scenes), tmp_path / "analysis"
    )

    arm = result["analysis"]["scene_analysis"][SCENE_A]["arms"][
        "H-hierarchy"
    ]
    assert arm["stage_funnel_diagnostics_schema_current"] is False
    assert arm["stage_funnel_lineage_complete"] is False
    assert arm["stage_funnel_final_equivalence_exact"] is False
    assert arm["stage_funnel_contract_pass"] is False
    assert result["analysis"]["mechanical_gate"]["passed"] is False
    assert result["analysis"]["scientific_gate"]["eligible"] is False


def test_current_schema_with_incomplete_lineage_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenes = [_scene(tmp_path, SCENE_A), _scene(tmp_path, SCENE_B)]
    spec = scenes[0]["mask_control_conditions"]["P-flat"]
    diagnostics_path = Path(spec["diagnostics"])
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics.pop("accepted_edges")
    _write_json(diagnostics_path, diagnostics)
    monkeypatch.setattr(mask_ablation, "load_ply_xyz", lambda _: _xyz())

    result = mask_ablation.evaluate_mask_contract_ablation_manifest(
        _manifest(tmp_path, scenes), tmp_path / "analysis"
    )

    arm = result["analysis"]["scene_analysis"][SCENE_A]["arms"]["P-flat"]
    assert arm["stage_funnel_diagnostics_schema_current"] is True
    assert arm["stage_funnel_lineage_complete"] is False
    assert arm["stage_funnel_final_equivalence_exact"] is False
    assert arm["stage_funnel_contract_pass"] is False
    assert result["analysis"]["mechanical_gate"]["passed"] is False


def test_mixed_nested_prediction_identity_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenes = [_scene(tmp_path, SCENE_A), _scene(tmp_path, SCENE_B)]
    hierarchy_spec = scenes[0]["mask_control_conditions"]["H-hierarchy"]
    flat_spec = scenes[0]["mask_control_conditions"]["P-flat"]
    hierarchy_diagnostics_path = Path(hierarchy_spec["diagnostics"])
    hierarchy_diagnostics = json.loads(
        hierarchy_diagnostics_path.read_text(encoding="utf-8")
    )
    flat_output = json.loads(Path(flat_spec["output"]).read_text(encoding="utf-8"))
    hierarchy_diagnostics["prediction_contract"]["run_identity"] = flat_output[
        "run_identity"
    ]
    _write_json(hierarchy_diagnostics_path, hierarchy_diagnostics)
    monkeypatch.setattr(mask_ablation, "load_ply_xyz", lambda _: _xyz())

    result = mask_ablation.evaluate_mask_contract_ablation_manifest(
        _manifest(tmp_path, scenes), tmp_path / "analysis"
    )

    contract = result["analysis"]["scene_analysis"][SCENE_A]["arms"][
        "H-hierarchy"
    ]["output_contract"]
    assert contract["passed"] is False
    assert contract["identity_checks"][
        "nested_prediction_contract_identity_exact"
    ] is False
    assert result["analysis"]["mechanical_gate"]["passed"] is False


def test_output_from_other_bank_cannot_bind_to_selected_bank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenes = [_scene(tmp_path, SCENE_A), _scene(tmp_path, SCENE_B)]
    hierarchy_spec = scenes[0]["mask_control_conditions"]["H-hierarchy"]
    flat_spec = scenes[0]["mask_control_conditions"]["P-flat"]
    hierarchy_spec["output"] = flat_spec["output"]
    hierarchy_spec["diagnostics"] = flat_spec["diagnostics"]
    monkeypatch.setattr(mask_ablation, "load_ply_xyz", lambda _: _xyz())

    result = mask_ablation.evaluate_mask_contract_ablation_manifest(
        _manifest(tmp_path, scenes), tmp_path / "analysis"
    )

    contract = result["analysis"]["scene_analysis"][SCENE_A]["arms"][
        "H-hierarchy"
    ]["output_contract"]
    assert contract["passed"] is False
    assert contract["identity_checks"]["evidence_bank_identity_exact"] is False
    assert contract["identity_checks"]["evidence_files_current"] is False
    assert result["analysis"]["mechanical_gate"]["passed"] is False


def test_missing_actual_pixel_or_repeat_audit_blocks_mechanical_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenes = [
        _scene(
            tmp_path,
            SCENE_A,
            include_input_audit=False,
            repeat_identity=None,
        ),
        _scene(tmp_path, SCENE_B),
    ]
    monkeypatch.setattr(mask_ablation, "load_ply_xyz", lambda _: _xyz())

    result = mask_ablation.evaluate_mask_contract_ablation_manifest(
        _manifest(tmp_path, scenes), tmp_path / "analysis"
    )

    assert result["analysis"]["mechanical_gate"]["passed"] is False
    assert result["analysis"]["scientific_gate"]["eligible"] is False
    assert result["analysis"]["scientific_gate"]["passed"] is False
    assert result["analysis"]["conclusion"] == "mechanical-contract-failed"


def test_duplicate_predictions_receive_only_one_one_to_one_true_positive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Both P candidates project to the same GT object.  The evaluator must not
    # count the duplicate as a second successful object.
    scenes = [
        _scene(tmp_path, SCENE_A, hierarchy_objects=[0], flat_objects=[0, 0]),
        _scene(tmp_path, SCENE_B, hierarchy_objects=[0], flat_objects=[0, 0]),
    ]
    # _output normally assigns each object's points once, so explicitly split
    # the same GT object's 100 Gaussians into two declared candidates.
    for scene in scenes:
        spec = scene["mask_control_conditions"]["P-flat"]
        original = json.loads(Path(spec["output"]).read_text(encoding="utf-8"))
        payload = _output(
            str(scene["scene_id"]),
            [0, 1],
            run_identity=original["run_identity"],
        )
        labels = np.asarray(payload["point_labels"], dtype=np.int64)
        labels[:] = -1
        labels[:50] = 0
        labels[50:100] = 1
        payload["point_labels"] = labels.tolist()
        _write_json(Path(spec["output"]), payload)
    monkeypatch.setattr(mask_ablation, "load_ply_xyz", lambda _: _xyz())

    result = mask_ablation.evaluate_mask_contract_ablation_manifest(
        _manifest(tmp_path, scenes), tmp_path / "analysis"
    )

    for scene_id in (SCENE_A, SCENE_B):
        matching = result["analysis"]["scene_analysis"][scene_id]["arms"][
            "P-flat"
        ]["three_spaces"]["subsets"]["all"]["matching"]["geometry"]["0.25"]
        assert matching["true_positive_count"] == 1
        assert matching["false_positive_count"] == 1


def test_pair_runtime_drift_is_a_mechanical_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenes = [_scene(tmp_path, SCENE_A), _scene(tmp_path, SCENE_B)]
    # Alter an actual non-mask runtime input in one flat bank.
    flat_dir = Path(
        scenes[0]["mask_control_conditions"]["P-flat"]["bank_dir"]
    )
    bank = mask_ablation.load_evidence_bank(flat_dir)
    source = dict(bank.source)
    source["shared"] = "drifted"
    drifted = AlphaMaskEvidenceBank(
        scene_id=bank.scene_id,
        point_count=bank.point_count,
        xyz_m=bank.xyz_m,
        class_names=bank.class_names,
        thresholds=bank.thresholds,
        frames=bank.frames,
        masks=bank.masks,
        mask_support=bank.mask_support,
        frame_visibility=bank.frame_visibility,
        frame_ambiguity=bank.frame_ambiguity,
        semantic_posteriors=bank.semantic_posteriors,
        semantic_abstained=bank.semantic_abstained,
        source=source,
    )
    save_evidence_bank(drifted, flat_dir, overwrite=True)
    monkeypatch.setattr(mask_ablation, "load_ply_xyz", lambda _: _xyz())

    result = mask_ablation.evaluate_mask_contract_ablation_manifest(
        _manifest(tmp_path, scenes), tmp_path / "analysis"
    )

    contract = result["analysis"]["scene_analysis"][SCENE_A]["pair_contract"]
    assert contract["checks"]["shared_runtime_identity_except_mask_mode"] is False
    assert result["analysis"]["mechanical_gate"]["passed"] is False


def test_stale_input_audit_cannot_certify_a_different_flat_bank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenes = [_scene(tmp_path, SCENE_A), _scene(tmp_path, SCENE_B)]
    audit = scenes[0]["flat_mask_input_audit"]
    audit["scene_id"] = "some-other-scene"
    monkeypatch.setattr(mask_ablation, "load_ply_xyz", lambda _: _xyz())

    result = mask_ablation.evaluate_mask_contract_ablation_manifest(
        _manifest(tmp_path, scenes), tmp_path / "analysis"
    )

    binding = result["analysis"]["scene_analysis"][SCENE_A]["pair_contract"][
        "input_binding_contract"
    ]
    assert binding["checks"]["audit_scene_identity"] is False
    assert binding["passed"] is False
    assert result["analysis"]["mechanical_gate"]["passed"] is False


def test_changed_repeat_output_fails_full_repeat_mechanical_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenes = [_scene(tmp_path, SCENE_A), _scene(tmp_path, SCENE_B)]
    repeat_audit = json.loads(
        Path(scenes[0]["flat_full_repeat_audit"]).read_text(encoding="utf-8")
    )
    repeat_output = Path(repeat_audit["repeat"]["output"])
    payload = json.loads(repeat_output.read_text(encoding="utf-8"))
    payload["instances"]["0"]["score"] = 0.5
    _write_json(repeat_output, payload)
    monkeypatch.setattr(mask_ablation, "load_ply_xyz", lambda _: _xyz())

    result = mask_ablation.evaluate_mask_contract_ablation_manifest(
        _manifest(tmp_path, scenes), tmp_path / "analysis"
    )

    repeat = result["analysis"]["scene_analysis"][SCENE_A]["pair_contract"][
        "flat_full_repeat_contract"
    ]
    assert repeat["checks"]["condition_bytes_identical"] is False
    assert repeat["passed"] is False
    assert result["analysis"]["mechanical_gate"]["passed"] is False


def test_manifest_gaussian_path_must_match_the_bank_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenes = [_scene(tmp_path, SCENE_A), _scene(tmp_path, SCENE_B)]
    replacement = tmp_path / SCENE_A / "same-bytes-other-path.ply"
    replacement.write_bytes(Path(scenes[0]["gaussian_ply"]).read_bytes())
    scenes[0]["gaussian_ply"] = str(replacement)
    monkeypatch.setattr(mask_ablation, "load_ply_xyz", lambda _: _xyz())

    result = mask_ablation.evaluate_mask_contract_ablation_manifest(
        _manifest(tmp_path, scenes), tmp_path / "analysis"
    )

    contract = result["analysis"]["scene_analysis"][SCENE_A]["pair_contract"]
    binding = contract["gaussian_asset_binding"]["H-hierarchy"]
    assert binding["checks"]["source_rgb_ply_matches_manifest"] is False
    assert binding["passed"] is False
    assert result["analysis"]["mechanical_gate"]["passed"] is False


def test_tiny_small_uses_complete_gt_extent_not_successful_or_dominant_subset() -> None:
    xyz = np.zeros((101, 3), dtype=np.float64)
    xyz[:100, 0] = np.linspace(0.0, 0.05, 100)
    xyz[100, 0] = 2.0
    semantic = np.concatenate(
        (np.zeros(100, dtype=np.int64), np.ones(1, dtype=np.int64))
    )
    instance = np.full(101, 7, dtype=np.int64)

    selected = mask_ablation._official_tiny_small_ids(
        gt_xyz=xyz,
        semantic=semantic,
        instance=instance,
        class_count=2,
        small_max_m=0.2,
        min_region_size=100,
    )

    assert selected == set()


def test_late_filter_factorial_uses_frozen_dev2_and_never_exports_b0_ap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenes = [_scene(tmp_path, SCENE_A), _scene(tmp_path, SCENE_B)]
    manifest = _manifest(tmp_path, scenes)
    monkeypatch.setattr(
        late_filter_experiment, "load_ply_xyz", lambda _: _xyz().copy()
    )

    result = late_filter_experiment.audit_clean_late_filters(
        manifest, tmp_path / "late-filter-analysis"
    )

    assert result["technical_gates"]["passed"] is True
    assert result["decision"] == "stop-accepted-components-or-earlier-insufficient"
    assert result["exporter_authorized"] is False
    assert result["category_prior_tested"] is False
    table = pd.read_parquet(
        tmp_path
        / "late-filter-analysis"
        / "clean_late_filter_factorial_dev2.parquet"
    )
    assert len(table) == 16
    assert set(table["arm_code"]) == {"A1B1", "A0B1", "A1B0", "A0B0"}
    assert table.loc[table["arm_code"].str.endswith("B0"), "official_ap_reported"].eq(
        False
    ).all()
    assert table.loc[table["arm_code"] == "A1B1", "official_ap_reported"].eq(
        True
    ).all()
    assert (
        tmp_path
        / "late-filter-analysis"
        / "clean_late_filter_analysis_dev2.json"
    ).is_file()
    repeated = late_filter_experiment.audit_clean_late_filters(
        manifest, tmp_path / "late-filter-analysis"
    )
    assert repeated["runner_status"] == "skipped-complete"


def test_late_filter_gate_uses_all_candidates_for_capacity_and_scene_recall() -> None:
    rows: list[dict[str, object]] = []
    for scene_id in (SCENE_A, SCENE_B):
        rows.append(
            {
                "scene_id": scene_id,
                "official_candidate_count": 10,
                "official_geometry_025_tp": 1,
                "official_geometry_025_fp": 9,
                "official_geometry_050_tp": 1,
                "official_geometry_025_tiny_small_recall": 0.5,
                "official_tiny_small_gt_count": 2,
                "all_geometry_025_recall": 0.75,
                "all_geometry_050_tp": 3,
                "accepted_all_candidate_count": 4,
                "accepted_all_geometry_025_tp": 3,
                "accepted_all_geometry_050_tp": 3,
                "accepted_official_geometry_025_tiny_small_recall": 0.5,
                "technical_contract_pass": True,
            }
        )

    aggregate = late_filter_experiment._aggregate_arm_rows(rows)

    assert aggregate["geometry_025_precision"] == pytest.approx(0.1)
    assert aggregate["geometry_050_tp"] == 6
    assert aggregate["accepted_geometry_050_tp"] == 6
    assert aggregate["scene_geometry_025_recall"] == {
        SCENE_A: 0.75,
        SCENE_B: 0.75,
    }


def test_late_filter_replay_score_drift_fails_technical_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenes = [_scene(tmp_path, SCENE_A), _scene(tmp_path, SCENE_B)]
    hierarchy_output = Path(
        scenes[0]["mask_control_conditions"]["H-hierarchy"]["output"]
    )
    payload = json.loads(hierarchy_output.read_text(encoding="utf-8"))
    payload["instances"]["0"]["score"] = 0.5
    _write_json(hierarchy_output, payload)
    monkeypatch.setattr(
        late_filter_experiment, "load_ply_xyz", lambda _: _xyz().copy()
    )

    destination = tmp_path / "late-filter-score-drift"
    result = late_filter_experiment.audit_clean_late_filters(
        _manifest(tmp_path, scenes), destination
    )
    factorial = json.loads(
        (destination / "clean_late_filter_factorial_dev2.json").read_text(
            encoding="utf-8"
        )
    )
    checks = factorial["scenes"][SCENE_A]["H-hierarchy"]["arms"]["A1B1"][
        "technical_contract"
    ]["checks"]

    assert checks["a1b1_frozen_partition_exact"] is True
    assert checks["a1b1_frozen_score_exact"] is False
    assert result["technical_gates"]["passed"] is False


def test_late_filter_reports_full_p_repeat_not_preparation_repeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenes = [_scene(tmp_path, SCENE_A), _scene(tmp_path, SCENE_B)]
    repeat_audit = Path(scenes[0]["flat_full_repeat_audit"])
    payload = json.loads(repeat_audit.read_text(encoding="utf-8"))
    payload["passed"] = False
    _write_json(repeat_audit, payload)
    monkeypatch.setattr(
        late_filter_experiment, "load_ply_xyz", lambda _: _xyz().copy()
    )

    destination = tmp_path / "late-filter-p-repeat"
    late_filter_experiment.audit_clean_late_filters(
        _manifest(tmp_path, scenes), destination
    )
    table = pd.read_parquet(destination / "clean_late_filter_factorial_dev2.parquet")
    hierarchy = table[
        (table["scene_id"] == SCENE_A)
        & (table["mask_mode"] == "H-hierarchy")
    ]
    flat = table[
        (table["scene_id"] == SCENE_A) & (table["mask_mode"] == "P-flat")
    ]

    assert hierarchy["historical_lifting_repeat_pass"].eq(True).all()
    assert flat["historical_lifting_repeat_pass"].eq(False).all()


def test_late_filter_does_not_reuse_mismatched_parquet_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenes = [_scene(tmp_path, SCENE_A), _scene(tmp_path, SCENE_B)]
    manifest = _manifest(tmp_path, scenes)
    destination = tmp_path / "late-filter-content-reuse"
    monkeypatch.setattr(
        late_filter_experiment, "load_ply_xyz", lambda _: _xyz().copy()
    )
    late_filter_experiment.audit_clean_late_filters(manifest, destination)
    table_path = destination / "clean_late_filter_factorial_dev2.parquet"
    table = pd.read_parquet(table_path)
    table.loc[0, "arm_name"] = "tampered"
    table.to_parquet(table_path, index=False)

    repeated = late_filter_experiment.audit_clean_late_filters(
        manifest, destination
    )
    repaired = pd.read_parquet(table_path)

    assert repeated.get("runner_status") != "skipped-complete"
    assert "tampered" not in set(repaired["arm_name"])


def test_late_filter_implementation_identity_covers_replay_and_metric_dependencies() -> None:
    paths = {
        str(row["path"])
        for row in late_filter_experiment._implementation_manifest()
    }

    assert {
        "clean_baseline/late_filter_experiment.py",
        "clean_baseline/late_filter_audit.py",
        "clean_baseline/stage_funnel.py",
        "clean_baseline/metric_reaudit.py",
        "clean_baseline/consensus.py",
        "clean_baseline/pipeline.py",
        "clean_baseline/models.py",
        "clean_baseline/evidence.py",
        "clean_baseline/evaluation.py",
        "clean_baseline/two_step_audit.py",
        "evaluator.py",
        "prediction_contract.py",
        "io.py",
        "scannet.py",
    }.issubset(paths)
