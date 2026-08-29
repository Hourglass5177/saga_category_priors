from __future__ import annotations

import json
from pathlib import Path

import pytest

import category_priors.category_cluster_runner as runner
from category_priors.category_cluster_bank import (
    R0_LEGACY,
    R1_METRIC_HDBSCAN,
)


def _repo_stub(root: Path) -> Path:
    (root / "category_priors").mkdir(parents=True)
    (root / "postprocess.py").write_text("# producer\n", encoding="utf-8")
    (root / "category_priors" / "category_cluster_bank.py").write_text(
        "# bank\n", encoding="utf-8"
    )
    return root


def _dev2_reference(path: Path) -> Path:
    conditions = {}
    for condition in (R0_LEGACY, R1_METRIC_HDBSCAN):
        conditions[condition] = {
            "scene_count": 2,
            "determinism_measured_this_scene_count": 2,
            "determinism_reference_scene_count": 0,
            "determinism_violation_count": 0,
        }
    path.write_text(
        json.dumps(
            {
                "schema": "saga-category-cluster-evaluation-v1",
                "phase": "dev2",
                "scene_ids": ["scene0645_00", "scene0025_01"],
                "conditions": conditions,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_embedded_bank_identity_changes_with_manifest_or_prior_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo_stub(tmp_path / "repo")
    manifest = tmp_path / "runtime.json"
    priors = tmp_path / "priors.json"
    manifest.write_text('{"scene":"first"}', encoding="utf-8")
    priors.write_text('{"global":1}', encoding="utf-8")
    monkeypatch.setattr(runner, "_git_commit", lambda _: "a" * 40)

    first = runner._cluster_bank_identity(
        repository=repo,
        runtime_manifest=manifest,
        category_priors=priors,
        scene_id="scene0645_00",
        seed=42,
        conditions=(R0_LEGACY, R1_METRIC_HDBSCAN),
        verify_determinism=True,
        determinism_reference=None,
    )
    manifest.write_text('{"scene":"second"}', encoding="utf-8")
    second = runner._cluster_bank_identity(
        repository=repo,
        runtime_manifest=manifest,
        category_priors=priors,
        scene_id="scene0645_00",
        seed=42,
        conditions=(R0_LEGACY, R1_METRIC_HDBSCAN),
        verify_determinism=True,
        determinism_reference=None,
    )

    assert first != second
    assert first["producer"]["git_commit"] == "a" * 40
    assert first["runtime_manifest"]["path"] == str(manifest.resolve())
    assert first["runtime_manifest"]["sha256"] != second["runtime_manifest"]["sha256"]
    assert first["category_priors"]["sha256"]
    assert not list(tmp_path.rglob("*.sha"))
    assert not list(tmp_path.rglob("*.sha256"))


def test_dev2_determinism_reference_requires_direct_measured_witnesses(
    tmp_path: Path,
) -> None:
    path = _dev2_reference(tmp_path / "dev2.json")

    observed = runner._load_determinism_reference(
        path, conditions=(R0_LEGACY, R1_METRIC_HDBSCAN)
    )

    assert observed["schema"] == runner.DETERMINISM_REFERENCE_SCHEMA
    assert observed["source_phase"] == "dev2"
    assert observed["conditions"][R1_METRIC_HDBSCAN][
        "measured_this_scene_count"
    ] == 2
    assert observed["artifact"]["sha256"]

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["conditions"][R1_METRIC_HDBSCAN][
        "determinism_measured_this_scene_count"
    ] = 0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not directly measured"):
        runner._load_determinism_reference(
            path, conditions=(R0_LEGACY, R1_METRIC_HDBSCAN)
        )


def test_unmeasured_runner_requires_an_explicit_dev2_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner,
        "_validate_common_inputs",
        lambda *_: ({"scene0645_00": {}}, tmp_path / "pipeline.sh", tmp_path / "priors.json", tmp_path / "out"),
    )
    monkeypatch.setattr(
        runner, "_normalize_scene_ids", lambda *_: ("scene0645_00",)
    )

    with pytest.raises(ValueError, match="require a verified DEV2"):
        runner.run_category_cluster_bank(
            "runtime.json",
            tmp_path / "out",
            tmp_path / "repo",
            tmp_path / "priors.json",
            scene_ids=("scene0645_00",),
            conditions=(R0_LEGACY, R1_METRIC_HDBSCAN),
            verify_determinism=False,
        )
