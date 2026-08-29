from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from category_priors import category_candidate_runner as runner
from category_priors.io import write_json


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    scene_id = "scene0001_00"
    base = tmp_path / scene_id
    base.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "run_pipeline.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    priors = tmp_path / "priors.json"
    write_json(priors, {"global": {"shrunk": {}}, "categories": {}})
    manifest = tmp_path / "runtime.json"
    write_json(
        manifest,
        {
            "kind": "scene_runtime_manifest",
            "scenes": [
                {
                    "scene_id": scene_id,
                    "base_path": str(base),
                    "scene_scale_m_per_unit": 1.0,
                }
            ],
        },
    )
    return manifest, repo, priors, scene_id


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_candidate_runner_is_gt_free_single_postprocess_and_resumable(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, repo, priors, scene_id = _fixture(tmp_path)
    output = tmp_path / "runs"
    checks = iter((False, True))
    monkeypatch.setattr(
        runner,
        "_candidate_scene_complete",
        lambda *args, **kwargs: next(checks),
    )
    captured: list[list[str]] = []

    def fake_run(command, *, cwd, log_path):
        captured.append(list(command))
        return 0

    monkeypatch.setattr(runner, "_run_command", fake_run)
    result = runner.repair_category_candidates(
        manifest,
        output,
        repo,
        priors,
        [scene_id],
        sample_cap=10_000,
    )

    assert result["runs"][0]["status"] == "complete"
    assert len(captured) == 1
    command = captured[0]
    assert _option(command, "--stage") == "postprocess"
    assert _option(command, "--category-denoise-action") == "candidate-repair"
    assert _option(command, "--category-candidate-trace-path") == str(
        (output / "candidate_trace" / scene_id).resolve()
    )
    assert _option(command, "--category-candidate-sample-cap") == "10000"
    assert result["sample_cap"] == 10_000
    assert "--gt-dir" not in command
    assert "--iterations" not in command

    monkeypatch.setattr(
        runner, "_candidate_scene_complete", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        runner,
        "_run_command",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("complete scene must be skipped")
        ),
    )
    resumed = runner.repair_category_candidates(
        manifest,
        output,
        repo,
        priors,
        [scene_id],
    )
    assert resumed["runs"][0]["status"] == "skipped_complete"


def test_candidate_replay_uses_frozen_threshold_and_no_protection(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, repo, priors, scene_id = _fixture(tmp_path)
    bank_root = tmp_path / "bank"
    (bank_root / scene_id).mkdir(parents=True)
    monkeypatch.setattr(runner, "_valid_candidate_bank", lambda *args, **kwargs: True)
    monkeypatch.setattr(runner, "load_candidate_bank", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        runner,
        "_replay_identity",
        lambda **kwargs: {"schema": "test-replay-identity"},
    )
    checks = iter((False, True))
    monkeypatch.setattr(
        runner, "_candidate_replay_complete", lambda *args, **kwargs: next(checks)
    )
    captured: list[list[str]] = []
    monkeypatch.setattr(
        runner,
        "_run_command",
        lambda command, **kwargs: captured.append(list(command)) or 0,
    )

    result = runner.replay_repaired_category_candidates(
        manifest,
        bank_root,
        tmp_path / "runs",
        repo,
        priors,
        [scene_id],
        modes=("uniform",),
        score_threshold=0.15,
    )

    assert result["action"] == "candidate-replay"
    command = captured[0]
    assert _option(command, "--category-denoise-action") == "candidate-replay"
    assert _option(command, "--category-candidate-score-threshold") == "0.15"
    assert "--gt-dir" not in command


def test_replay_identity_changes_when_bank_q_or_prior_changes(
    tmp_path: Path,
) -> None:
    priors = tmp_path / "priors.json"
    write_json(priors, {"global": {"shrunk": {}}, "categories": {}})
    bank_path = tmp_path / "bank" / "scene0001_00"
    bank_path.mkdir(parents=True)

    def bank(q: float):
        return SimpleNamespace(
            global_pre_knn=np.asarray([-1, 0], dtype=np.int64),
            semantic_top1=np.asarray([0, 0], dtype=np.int64),
            semantic_top1_score=np.asarray([0.8, 0.9], dtype=np.float64),
            branch_full_labels=np.asarray([0, -1], dtype=np.int64),
            branch_core_labels=np.asarray([0, -1], dtype=np.int64),
            assignment_confidence=np.asarray([0.5, 0.0], dtype=np.float64),
            candidates=({"candidate_id": 0, "Q": q},),
            diagnostics={
                "candidate_repair_condition": "C1-consistent-envelope",
                "sample_cap": 5000,
            },
        )

    first = runner._replay_identity(
        bank=bank(0.4),
        bank_path=bank_path,
        priors_path=priors,
        scene_id="scene0001_00",
        mode="uniform",
        score_threshold=0.2,
        seed=42,
    )
    changed_q = runner._replay_identity(
        bank=bank(0.5),
        bank_path=bank_path,
        priors_path=priors,
        scene_id="scene0001_00",
        mode="uniform",
        score_threshold=0.2,
        seed=42,
    )
    assert first["bank_digest"] != changed_q["bank_digest"]
    assert first["candidate_id_q_digest"] != changed_q["candidate_id_q_digest"]

    write_json(priors, {"global": {"shrunk": {"changed": True}}, "categories": {}})
    changed_prior = runner._replay_identity(
        bank=bank(0.4),
        bank_path=bank_path,
        priors_path=priors,
        scene_id="scene0001_00",
        mode="uniform",
        score_threshold=0.2,
        seed=42,
    )
    assert first["category_priors_sha256"] != changed_prior["category_priors_sha256"]
