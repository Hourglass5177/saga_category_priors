from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from category_priors.clean_baseline import materialize_config as module
from category_priors.clean_baseline.experiment import CleanExperimentConfig, DEV8
from category_priors.clean_baseline.identity_control import (
    IDENTITY_CONTROL_REGISTRATION_SCHEMA,
    IDENTITY_CONTROL_SCHEMA,
)
from category_priors.clean_baseline.validation import HOLDOUT5
from category_priors.clean_baseline.worker import DEFAULT_CLASSES
from category_priors.io import hash_json, load_json, sha256_file, write_json
from category_priors.taxonomy import default_taxonomy_path, load_taxonomy


def _tune24() -> tuple[str, ...]:
    base = DEV8 + HOLDOUT5
    return base + tuple(f"{value.rsplit('_', 1)[0]}_99" for value in base[:11])


def _final48() -> tuple[str, ...]:
    return ("scene0019_01",) + tuple(
        f"scene{index:04d}_00" for index in range(1000, 1047)
    )


def _write_ascii_ply(path: Path, xyz: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(xyz)}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
        *(" ".join(map(str, row)) for row in np.asarray(xyz)),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _prior_payload() -> dict[str, object]:
    classes = load_taxonomy().canonical_classes

    def node() -> dict[str, object]:
        return {
            "shrunk": {
                "geometry": {
                    key: {"q95": 0.0}
                    for key in (
                        "log_extent_short_m",
                        "log_extent_mid_m",
                        "log_extent_long_m",
                    )
                }
            }
        }

    payload: dict[str, object] = {
        "schema_version": "1.0",
        "kind": "category_priors",
        "provenance": {"splits": ["train"]},
        "normalization": {"units": "meters"},
        "global": node(),
        "categories": {name: node() for name in classes},
    }
    payload["content_sha256"] = hash_json(payload)
    return payload


def _fixture(tmp_path: Path) -> dict[str, Path]:
    tune, final = _tune24(), _final48()
    tune_gt, final_gt = tmp_path / "tune-gt", tmp_path / "final-gt"
    sam_root = tmp_path / "sam"
    for directory in (tune_gt, final_gt, sam_root):
        directory.mkdir(parents=True)

    def runtime_row(scene_id: str) -> dict[str, object]:
        base = tmp_path / "scenes" / scene_id
        ply = base / "output_models/point_cloud/iteration_30000/point_cloud.ply"
        _write_ascii_ply(ply, np.asarray([[0.0, 0.0, 0.0]]))
        for relative in (
            "fastRecon/dense/sparse/0",
            "fastRecon/dense/sparse/0/images",
            "saga/masks",
            "saga/labels",
        ):
            (base / relative).mkdir(parents=True, exist_ok=True)
        (sam_root / scene_id).mkdir(parents=True)
        np.savez_compressed(
            sam_root / scene_id / "frame0000.npz",
            packed=np.empty((0, 1), dtype=np.uint8),
            count=np.asarray(0),
            height=np.asarray(1),
            width=np.asarray(1),
        )
        (base / "saga/masks/frame0000.pt").write_bytes(b"mask")
        (base / "saga/labels/frame0000.pt").write_bytes(b"label")
        gt_root = final_gt if scene_id in final else tune_gt
        np.savez_compressed(
            gt_root / f"{scene_id}.npz",
            coords=np.zeros((1, 3), dtype=np.float32),
            semantic=np.zeros(1, dtype=np.int64),
            instance=np.zeros(1, dtype=np.int64),
        )
        return {
            "scene_id": scene_id,
            "base_path": str(base),
            "scene_scale_m_per_unit": 1.0,
            "gaussian_to_gt_transform": np.eye(4).tolist(),
        }

    tune_manifest, final_manifest = tmp_path / "tune.json", tmp_path / "final.json"
    write_json(tune_manifest, {"kind": "scene_runtime_manifest", "scenes": [runtime_row(value) for value in tune]})
    write_json(final_manifest, {"kind": "scene_runtime_manifest", "scenes": [runtime_row(value) for value in final]})
    locked = tmp_path / "locked.json"
    write_json(
        locked,
        {
            "kind": "locked_evaluation_scenes",
            "scenes": [
                {"scene_id": value, "physical_scene_id": value.rsplit("_", 1)[0]}
                for value in final
            ],
        },
    )
    train = tmp_path / "train.txt"
    train.write_text("scene0800_00\nscene0801_01\n", encoding="utf-8")
    priors = tmp_path / "priors.json"
    write_json(priors, _prior_payload())
    size_bins = tmp_path / "size-bins.json"
    write_json(
        size_bins,
        {
            "boundaries_m": {
                "tiny_max_m": 0.5,
                "small_max_m": 1.0,
                "medium_max_m": 2.0,
            }
        },
    )
    b1 = tmp_path / "b1.json"
    write_json(b1, {"map_50_95": 0.05, "map_0.50": 0.10})
    repo = tmp_path / "repo"
    repo.mkdir()
    return {
        "tune_runtime_manifest": tune_manifest,
        "final_runtime_manifest": final_manifest,
        "locked_evaluation_scenes": locked,
        "tune_gt_root": tune_gt,
        "final_gt_root": final_gt,
        "train_scene_list": train,
        "category_priors": priors,
        "size_bins": size_bins,
        "taxonomy": default_taxonomy_path(),
        "b1_fixed_metrics": b1,
        "repo_root": repo,
        "run_root": tmp_path / "runs",
        "artifact_root": tmp_path / "artifacts",
        "output_dir": tmp_path / "registration",
        "tune_sam_root": sam_root,
        "final_sam_root": sam_root,
    }


def test_formal_runtime_sanitizer_removes_nested_evaluation_inputs() -> None:
    source = {
        "scene_id": "scene0000_00",
        "gt_npz": "top-level-gt.npz",
        "nested": {
            "ground_truth_path": "nested-gt.npz",
            "keep": {"value": 7},
        },
        "items": [
            {"replacement_gt": "replacement.npz", "image": "frame.jpg"}
        ],
    }
    assert module._formal_runtime_fields(source) == {
        "scene_id": "scene0000_00",
        "nested": {"keep": {"value": 7}},
        "items": [{"image": "frame.jpg"}],
    }


def test_materializes_strict_config_and_separates_32_from_20(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    monkeypatch.setattr(
        module,
        "_tiny_small_instance_ids",
        lambda **_kwargs: ((7,), {"official_mapped_instance_count": 1, "tiny_small_official_mapped_instance_count": 1, "gt_point_mapping": {}}),
    )
    commit = "a" * 40
    result = module.materialize_clean_baseline_config(
        **paths,
        code_commit=commit,
        git_head_reader=lambda _path: commit,
    )
    assert result["status"] == "complete"
    assert result["tune_scan_count"] == 24
    assert result["tune_physical_scene_count"] == 13
    assert result["final48_physical_scene_count"] == 48
    config = load_json(Path(result["config"]))
    assert config["class_names"] == list(DEFAULT_CLASSES)
    assert config["evidence_class_names"] == list(DEFAULT_CLASSES)
    assert config["evaluation_class_names"] == list(load_taxonomy().canonical_classes)
    assert config["allowed_classes"] == config["evaluation_class_names"]
    assert len(config["scenes"]) == 72
    request = load_json(Path(config["scenes"][DEV8[0]]["evidence_request"]))
    assert request["producer_commit"] == commit
    assert request["classes"] == list(DEFAULT_CLASSES)
    assert config["scenes"][DEV8[0]]["tiny_small_instance_ids"] == []
    assert config["scenes"][DEV8[0]]["tiny_small_deferred"] is True
    assert config["runtime_registration"][DEV8[0]]["scene_id"] == DEV8[0]
    parsed = CleanExperimentConfig.from_json(Path(result["config"]))
    assert parsed.evidence_class_names == tuple(DEFAULT_CLASSES)
    assert parsed.evaluation_class_names == tuple(load_taxonomy().canonical_classes)
    assert config["identity_control_registration"]["schema"] == (
        IDENTITY_CONTROL_REGISTRATION_SCHEMA
    )
    assert config["identity_control_registration"]["status"] == "unavailable"
    assert result["identity_control_status"] == "unavailable"
    assert "identity_control" not in config
    # Exact reruns are allowed; a changed registration is not overwritten.
    again = module.materialize_clean_baseline_config(
        **paths,
        code_commit=commit,
        git_head_reader=lambda _path: commit,
    )
    assert again == result


def test_materializer_registers_old_evidence_producer_without_changing_other_scenes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    old_commit = "a" * 40
    old_paths = dict(paths)
    old_paths["run_root"] = tmp_path / "old-runs"
    old_paths["artifact_root"] = tmp_path / "old-artifacts"
    old_paths["output_dir"] = tmp_path / "old-registration"
    old = module.materialize_clean_baseline_config(
        **old_paths,
        code_commit=old_commit,
        git_head_reader=lambda _path: old_commit,
    )
    old_config = load_json(Path(old["config"]))
    imported_scene = DEV8[0]
    imported_bank = tmp_path / "copied-old-bank" / imported_scene
    imported_bank.mkdir(parents=True)
    evidence_files = ("evidence.npz", "masks.json", "diagnostics.json")
    for name in evidence_files:
        (imported_bank / name).write_bytes(f"old-{name}".encode("ascii"))
    imports = tmp_path / "evidence-imports.json"
    write_json(
        imports,
        {
            "schema": module.EVIDENCE_IMPORT_MANIFEST_SCHEMA,
            "scenes": {
                imported_scene: {
                    "bank_dir": str(imported_bank),
                    "source_request": old_config["scenes"][imported_scene][
                        "evidence_request"
                    ],
                    "producer_commit": old_commit,
                    "files": {
                        name: sha256_file(imported_bank / name)
                        for name in evidence_files
                    },
                }
            },
        },
    )
    monkeypatch.setattr(
        module,
        "evidence_bank_is_complete",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        module,
        "evidence_request_source",
        lambda **_kwargs: {"producer_commit": old_commit},
    )
    new_commit = "b" * 40
    new_paths = dict(paths)
    new_paths["run_root"] = tmp_path / "new-runs"
    new_paths["artifact_root"] = tmp_path / "new-artifacts"
    new_paths["output_dir"] = tmp_path / "new-registration"
    result = module.materialize_clean_baseline_config(
        **new_paths,
        code_commit=new_commit,
        evidence_imports=imports,
        git_head_reader=lambda _path: new_commit,
    )
    config = load_json(Path(result["config"]))
    assert result["imported_evidence_scene_ids"] == [imported_scene]
    assert config["evidence_imports"][imported_scene]["producer_commit"] == old_commit
    assert config["evidence_imports"][imported_scene]["bank_dir"] == str(
        imported_bank.resolve()
    )
    imported_request = load_json(
        Path(config["scenes"][imported_scene]["evidence_request"])
    )
    normal_request = load_json(Path(config["scenes"][DEV8[1]]["evidence_request"]))
    assert imported_request["producer_commit"] == old_commit
    assert normal_request["producer_commit"] == new_commit
    parsed = CleanExperimentConfig.from_json(Path(result["config"]))
    assert parsed.bank_dir(imported_scene) == imported_bank.resolve()
    assert parsed.bank_dir(DEV8[1]) == Path(new_paths["run_root"]) / "bank" / DEV8[1]


def test_legacy_hierarchy_mode_compatibility_is_explicit_and_single_field_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    scene_id = DEV8[0]
    producer = next(iter(module.LEGACY_HIERARCHY_PRODUCER_COMMITS))
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    files = {}
    for name in ("evidence.npz", "masks.json", "diagnostics.json"):
        path = bank_dir / name
        path.write_bytes(name.encode("ascii"))
        files[name] = sha256_file(path)
    request = tmp_path / "request.json"
    write_json(
        request,
        {
            "schema": module.REQUEST_SCHEMA,
            "producer_commit": producer,
        },
    )
    imports = tmp_path / "imports.json"
    write_json(
        imports,
        {
            "schema": module.EVIDENCE_IMPORT_MANIFEST_SCHEMA,
            "scenes": {
                scene_id: {
                    "bank_dir": str(bank_dir),
                    "source_request": str(request),
                    "producer_commit": producer,
                    "files": files,
                }
            },
        },
    )
    expected = {
        "producer_commit": producer,
        "sam_masks": "/frozen/hierarchy",
        "mask_observation_mode": "hierarchy",
    }
    monkeypatch.setattr(module, "evidence_request_source", lambda **_: expected)
    monkeypatch.setattr(module, "evidence_bank_is_complete", lambda *_, **__: False)
    monkeypatch.setattr(
        module,
        "load_evidence_bank",
        lambda *_args, **_kwargs: SimpleNamespace(
            source={key: value for key, value in expected.items() if key != "mask_observation_mode"}
        ),
    )
    with pytest.raises(ValueError, match="incomplete"):
        module._load_evidence_imports(imports)
    loaded = module._load_evidence_imports(
        imports,
        legacy_hierarchy_producer_commits=module.LEGACY_HIERARCHY_PRODUCER_COMMITS,
    )
    assert loaded[scene_id]["legacy_hierarchy_mode_proof"] == {
        "producer_commit": producer,
        "assumed_mode": "hierarchy",
        "missing_fields": ["mask_observation_mode"],
    }

    flat_expected = {**expected, "mask_observation_mode": "flat-highest-quality"}
    monkeypatch.setattr(module, "evidence_request_source", lambda **_: flat_expected)
    monkeypatch.setattr(
        module,
        "load_evidence_bank",
        lambda *_args, **_kwargs: SimpleNamespace(
            source={
                key: value
                for key, value in flat_expected.items()
                if key != "mask_observation_mode"
            }
        ),
    )
    with pytest.raises(ValueError, match="incomplete"):
        module._load_evidence_imports(
            imports,
            legacy_hierarchy_producer_commits=module.LEGACY_HIERARCHY_PRODUCER_COMMITS,
        )

    monkeypatch.setattr(module, "evidence_request_source", lambda **_: expected)
    monkeypatch.setattr(
        module,
        "load_evidence_bank",
        lambda *_args, **_kwargs: SimpleNamespace(
            source={"producer_commit": producer, "sam_masks": "/different"}
        ),
    )
    with pytest.raises(ValueError, match="incomplete"):
        module._load_evidence_imports(
            imports,
            legacy_hierarchy_producer_commits=module.LEGACY_HIERARCHY_PRODUCER_COMMITS,
        )

    with pytest.raises(ValueError, match="incomplete"):
        module._load_evidence_imports(
            imports,
            legacy_hierarchy_producer_commits=frozenset({"b" * 40}),
        )


def test_registers_existing_three_scene_identity_control_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    tune_manifest = load_json(paths["tune_runtime_manifest"])
    by_scene = {row["scene_id"]: row for row in tune_manifest["scenes"]}
    for scene_id in (*DEV8[:2], DEV8[2]):
        feature = (
            Path(by_scene[scene_id]["base_path"])
            / "saga/contrastive_feature_point_cloud.ply"
        )
        feature.parent.mkdir(parents=True, exist_ok=True)
        feature.write_bytes(b"registered-native-2k")
    monkeypatch.setattr(
        module,
        "_tiny_small_instance_ids",
        lambda **_kwargs: ((), {}),
    )
    monkeypatch.setattr(
        module,
        "load_affinity_feature_ply",
        lambda _path: (
            np.zeros((1, 3), dtype=np.float64),
            np.zeros((1, 32), dtype=np.float64),
        ),
    )
    monkeypatch.setattr(
        module,
        "load_gaussian_attributes_ply",
        lambda _path: (
            np.zeros((1, 3), dtype=np.float64),
            np.ones((1, 3), dtype=np.float64),
            np.ones(1, dtype=np.float64),
        ),
    )
    commit = "b" * 40
    result = module.materialize_clean_baseline_config(
        **paths,
        code_commit=commit,
        git_head_reader=lambda _path: commit,
    )
    config = load_json(Path(result["config"]))
    assert result["identity_control_status"] == "available"
    assert result["identity_control_issues"] == []
    assert config["identity_control_registration"]["status"] == "available"
    assert config["identity_control"]["schema"] == IDENTITY_CONTROL_SCHEMA
    assert config["identity_control"]["train_scene_ids"] == list(DEV8[:2])
    assert config["identity_control"]["validation_scene_id"] == DEV8[2]
    assert set(config["identity_control"]["assets"]) == set(DEV8[:3])
    parsed = CleanExperimentConfig.from_json(Path(result["config"]))
    assert parsed.identity_control is not None


def test_final48_assets_are_registered_but_not_eagerly_required(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    final_manifest = load_json(paths["final_runtime_manifest"])
    row = final_manifest["scenes"][0]
    scene_id = row["scene_id"]
    ply = (
        Path(row["base_path"])
        / "output_models/point_cloud/iteration_30000/point_cloud.ply"
    )
    row["gaussian_ply"] = str(ply)
    write_json(paths["final_runtime_manifest"], final_manifest)
    gt = Path(paths["final_gt_root"]) / f"{scene_id}.npz"
    ply.unlink()
    gt.unlink()
    commit = "c" * 40
    result = module.materialize_clean_baseline_config(
        **paths,
        code_commit=commit,
        git_head_reader=lambda _path: commit,
    )
    config = load_json(Path(result["config"]))
    assert scene_id in config["final48"]
    assert config["scenes"][scene_id]["gaussian_ply"] == str(ply.resolve())
    assert config["scenes"][scene_id]["gt_npz"] == str(gt.resolve())
    report = next(row for row in result["scenes"] if row["scene_id"] == scene_id)
    assert report["asset_validation"] == "deferred-until-stage"


def test_rejects_train_dev_physical_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    Path(paths["train_scene_list"]).write_text(f"{DEV8[0]}\n", encoding="utf-8")
    monkeypatch.setattr(module, "_tiny_small_instance_ids", lambda **_kwargs: ((), {}))
    commit = "a" * 40
    with pytest.raises(ValueError, match="physical-scene split overlap"):
        module.materialize_clean_baseline_config(
            **paths,
            code_commit=commit,
            git_head_reader=lambda _path: commit,
        )


def test_rejects_locked_runtime_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    locked = load_json(paths["locked_evaluation_scenes"])
    locked["scenes"][0]["scene_id"] = "scene9999_00"
    locked["scenes"][0]["physical_scene_id"] = "scene9999"
    write_json(paths["locked_evaluation_scenes"], locked)
    monkeypatch.setattr(module, "_tiny_small_instance_ids", lambda **_kwargs: ((), {}))
    commit = "a" * 40
    with pytest.raises(ValueError, match="exactly match"):
        module.materialize_clean_baseline_config(
            **paths,
            code_commit=commit,
            git_head_reader=lambda _path: commit,
        )


def test_tiny_small_requires_at_least_100_mapped_official_points(tmp_path: Path) -> None:
    gt_path, ply_path = tmp_path / "gt.npz", tmp_path / "iteration_30000/point_cloud.ply"
    xyz = np.column_stack(
        (np.linspace(0.0, 0.2, 100), np.zeros(100), np.zeros(100))
    ).astype(np.float32)
    np.savez_compressed(
        gt_path,
        coords=xyz,
        semantic=np.zeros(100, dtype=np.int64),
        instance=np.full(100, 7, dtype=np.int64),
    )
    size_bins = {
        "boundaries_m": {
            "tiny_max_m": 0.1,
            "small_max_m": 0.5,
            "medium_max_m": 1.0,
        }
    }
    _write_ascii_ply(ply_path, xyz)
    selected, report = module._tiny_small_instance_ids(
        scene_id="scene0000_00",
        gt_npz=gt_path,
        gaussian_ply=ply_path,
        transform=np.eye(4),
        size_bins=size_bins,
        evaluation_class_count=20,
    )
    assert selected == (7,)
    assert report["official_mapped_instance_count"] == 1

    _write_ascii_ply(ply_path, xyz[:99])
    selected, report = module._tiny_small_instance_ids(
        scene_id="scene0000_00",
        gt_npz=gt_path,
        gaussian_ply=ply_path,
        transform=np.eye(4),
        size_bins=size_bins,
        evaluation_class_count=20,
        radius_m=1e-6,
    )
    assert selected == ()
    assert report["official_mapped_instance_count"] == 0


def test_commit_and_30k_identity_are_strict(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="differs from repo HEAD"):
        paths = _fixture(tmp_path)
        module.materialize_clean_baseline_config(
            **paths,
            code_commit="a" * 40,
            git_head_reader=lambda _path: "b" * 40,
        )
    with pytest.raises(ValueError, match="cannot prove"):
        module._prove_30k_ply("scene0000_00", {}, tmp_path / "point_cloud.ply")
