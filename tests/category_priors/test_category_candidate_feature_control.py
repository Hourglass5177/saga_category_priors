from __future__ import annotations

import ast
from pathlib import Path

import pytest

from category_priors.category_candidate_feature_control import (
    CONTROL_SCHEMA,
    bind_control_candidate_root,
    materialize_feature_runtime_manifest,
    same_source_training_assets,
)
from category_priors.io import load_json, write_json


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def test_same_source_assets_follow_active_runtime_overrides(tmp_path: Path) -> None:
    base = tmp_path / "scene"
    directories = {
        name: base / name
        for name in ("images", "sparse", "masks", "labels", "scales")
    }
    for path in directories.values():
        path.mkdir(parents=True)
    point_cloud = _touch(base / "gaussians.ply")
    label_features = _touch(base / "label_features.pt")
    scene = {
        "base_path": str(base),
        "images_path": str(directories["images"]),
        "sparse_path": str(directories["sparse"]),
        "point_cloud_path": str(point_cloud),
        "masks_path": str(directories["masks"]),
        "grounded_labels_path": str(directories["labels"]),
        "label_features_path": str(label_features),
        "mask_scales_path": str(directories["scales"]),
    }

    assets = same_source_training_assets(scene)

    assert assets == {
        "images": directories["images"].resolve(),
        "sparse": directories["sparse"].resolve(),
        "point_cloud": point_cloud.resolve(),
        "masks": directories["masks"].resolve(),
        "labels": directories["labels"].resolve(),
        "label_features": label_features.resolve(),
        "mask_scales": directories["scales"].resolve(),
    }


def test_derived_runtime_binds_only_validated_10k_feature_pair(
    tmp_path: Path,
) -> None:
    source = tmp_path / "runtime.json"
    write_json(
        source,
        {
            "kind": "scene_runtime_manifest",
            "scenes": [
                {
                    "scene_id": "scene0645_00",
                    "base_path": str(tmp_path / "scene"),
                    "scene_scale_m_per_unit": 1.0,
                    "contrastive_feature_point_cloud_path": "old.ply",
                    "scale_gate_path": "old.pt",
                },
                {
                    "scene_id": "scene9999_00",
                    "base_path": str(tmp_path / "other"),
                    "scene_scale_m_per_unit": 1.0,
                },
            ],
        },
    )
    feature = _touch(tmp_path / "10k" / "feature.ply")
    gate = _touch(tmp_path / "10k" / "gate.pt")
    training = {
        "schema": CONTROL_SCHEMA,
        "status": "complete",
        "feature_iterations": 10_000,
        "scenes": [
            {
                "scene_id": "scene0645_00",
                "checkpoints": {
                    "10k": {
                        "feature_ply": str(feature),
                        "scale_gate": str(gate),
                    }
                },
            }
        ],
    }
    output = tmp_path / "derived.json"

    materialize_feature_runtime_manifest(
        source_manifest=source,
        training_payload=training,
        output=output,
    )

    rows = {row["scene_id"]: row for row in load_json(output)["scenes"]}
    controlled = rows["scene0645_00"]
    assert controlled["contrastive_feature_point_cloud_path"] == str(
        feature.resolve()
    )
    assert controlled["scale_gate_path"] == str(gate.resolve())
    assert controlled["feature_training_iterations"] == 10_000
    assert "contrastive_feature_point_cloud_path" not in rows["scene9999_00"]


def test_candidate_cli_has_no_eager_retired_experiment_imports() -> None:
    cli_path = Path(__file__).parents[2] / "category_priors" / "cli.py"
    tree = ast.parse(cli_path.read_text(encoding="utf-8"))
    forbidden = ("v8_", "v9_", "v10_")
    eager = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            leaf = node.module.rsplit(".", 1)[-1]
            if leaf.startswith(forbidden):
                eager.append(node.module)
    assert eager == []


def test_control_bank_root_rejects_changed_feature_pair(tmp_path: Path) -> None:
    feature = _touch(tmp_path / "feature.ply")
    gate = _touch(tmp_path / "gate.pt")
    runtime = tmp_path / "runtime.json"
    write_json(
        runtime,
        {
            "kind": "scene_runtime_manifest",
            "scenes": [
                {
                    "scene_id": "scene0645_00",
                    "base_path": str(tmp_path),
                    "scene_scale_m_per_unit": 1.0,
                    "contrastive_feature_point_cloud_path": str(feature),
                    "scale_gate_path": str(gate),
                }
            ],
        },
    )
    root = tmp_path / "control"
    bind_control_candidate_root(
        runtime_manifest=runtime,
        control_root=root,
        scene_ids=("scene0645_00",),
        sample_cap=5000,
        seed=42,
    )
    feature.write_bytes(b"changed")

    with pytest.raises(ValueError, match="different feature inputs"):
        bind_control_candidate_root(
            runtime_manifest=runtime,
            control_root=root,
            scene_ids=("scene0645_00",),
            sample_cap=5000,
            seed=42,
        )
