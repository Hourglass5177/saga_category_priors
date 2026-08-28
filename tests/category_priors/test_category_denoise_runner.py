from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from category_priors.category_denoise import CandidateBank, save_candidate_bank
from category_priors.category_denoise_runner import (
    _bank_complete,
    _build_command,
    _replay_complete,
    replay_category_denoise,
    run_category_denoise_bank,
)
from category_priors.io import write_json

REQUIRED_BANK_ARRAYS = {
    "global_pre_knn": np.asarray([-1, 0, 0], dtype=np.int64),
    "semantic_top1": np.asarray([2, 0, 0], dtype=np.int64),
    "semantic_top1_score": np.asarray([0.8, 0.9, 0.9], dtype=np.float64),
    "branch_full_labels": np.asarray([-1, 0, 0], dtype=np.int64),
    "branch_core_labels": np.asarray([-1, 0, -1], dtype=np.int64),
    "assignment_confidence": np.asarray([0.0, 0.8, 0.7], dtype=np.float32),
}


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _runtime_fixture(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    scene_id = "scene0001_00"
    base = tmp_path / scene_id
    base.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "run_pipeline.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    priors = tmp_path / "category_priors.json"
    write_json(
        priors,
        {
            "global": {
                "shrunk": {
                    "geometry": {
                        "log_extent_short_m": {"q50": 0.0, "q75": math.log(2.0)},
                        "log_extent_mid_m": {"q50": 0.0, "q75": math.log(2.0)},
                        "log_extent_long_m": {"q50": 0.0, "q75": math.log(2.0)},
                        "log_surface_area_m2": {"q50": 0.0},
                    },
                    "neighborhood": {
                        "boundary_fixed:0.05": {"q50": 0.1, "q75": 0.2}
                    },
                }
            },
            "categories": {},
        },
    )
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


def _write_prediction(path: Path) -> None:
    write_json(
        path,
        {
            "point_labels": [-1, 0, 0],
            "instances": {"0": {"class": "chair", "score": 0.75}},
        },
    )


def _write_complete_bank(root: Path, *, scene_id: str = "scene0001_00") -> None:
    root.mkdir(parents=True, exist_ok=True)
    _write_prediction(root / "output.json")
    class_names = ("chair",) + tuple(f"class-{index}" for index in range(1, 32))
    bank = CandidateBank(
        class_names=class_names,
        saga20_names=("chair",),
        scene_scale_m_per_unit=1.0,
        seed=42,
        global_pre_knn=REQUIRED_BANK_ARRAYS["global_pre_knn"],
        semantic_top1=REQUIRED_BANK_ARRAYS["semantic_top1"],
        semantic_top1_score=REQUIRED_BANK_ARRAYS["semantic_top1_score"],
        branch_full_labels=REQUIRED_BANK_ARRAYS["branch_full_labels"],
        branch_core_labels=REQUIRED_BANK_ARRAYS["branch_core_labels"],
        assignment_confidence=REQUIRED_BANK_ARRAYS["assignment_confidence"],
        candidates=(
            {
                "candidate_id": 0,
                "branch_class": "chair",
                "branch_class_index": 0,
                "core_point_count": 1,
                "full_point_count": 2,
                "assignment_confidence_mean": 0.75,
                "metric_extents_m": [0.1, 0.2, 0.3],
                "boundary_ratio_5cm": 0.1,
            },
        ),
        diagnostics={"scene_id": scene_id},
    )
    save_candidate_bank(bank, root)
    (root / "bank.log").write_text("complete\n", encoding="utf-8")


def _write_complete_replay(
    root: Path, *, scene_id: str, mode: str
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _write_prediction(root / "output.json")
    write_json(
        root / "diagnostics.json",
        {
            "schema": "category-denoise-replay-v1",
            "scene_id": scene_id,
            "mode": mode,
            "point_count": 3,
            "status": "complete",
        },
    )
    (root / "postprocess.log").write_text("complete\n", encoding="utf-8")


def test_bank_complete_rejects_missing_arrays_and_inconsistent_lengths(
    tmp_path: Path,
) -> None:
    bank = tmp_path / "bank"
    _write_complete_bank(bank)
    assert _bank_complete(bank)

    np.savez_compressed(bank / "bank_labels.npz", global_pre_knn=REQUIRED_BANK_ARRAYS["global_pre_knn"])
    assert not _bank_complete(bank)

    _write_complete_bank(bank)
    with np.load(bank / "bank_labels.npz", allow_pickle=False) as arrays:
        broken = {name: np.asarray(arrays[name]) for name in arrays.files}
    broken["branch_core_labels"] = np.asarray([-1, 0], dtype=np.int64)
    np.savez_compressed(bank / "bank_labels.npz", **broken)
    assert not _bank_complete(bank)


def test_bank_complete_rejects_corrupt_candidate_metadata(tmp_path: Path) -> None:
    bank = tmp_path / "bank"
    _write_complete_bank(bank)
    (bank / "candidates.json").write_text("{broken", encoding="utf-8")

    assert not _bank_complete(bank)


def test_replay_complete_requires_diagnostics_not_only_output_and_log(
    tmp_path: Path,
) -> None:
    replay = tmp_path / "uniform" / "scene0001_00"
    replay.mkdir(parents=True)
    _write_prediction(replay / "output.json")
    (replay / "postprocess.log").write_text("complete\n", encoding="utf-8")

    assert not _replay_complete(replay)

    write_json(
        replay / "diagnostics.json",
        {
            "schema": "category-denoise-replay-v1",
            "scene_id": "scene0001_00",
            "mode": "uniform",
            "point_count": 3,
            "status": "complete",
        },
    )
    assert _replay_complete(replay)


def test_command_is_fixed_to_legacy_postprocess_without_training_or_gt(
    tmp_path: Path,
) -> None:
    pipeline = tmp_path / "run_pipeline.sh"
    priors = tmp_path / "priors.json"
    scene = {
        "base_path": str(tmp_path / "scene"),
        "scene_scale_m_per_unit": 1.25,
        "python_bin": str(tmp_path / "registered-python"),
    }
    override_python = tmp_path / "gpu-compatible-python"
    command = _build_command(
        pipeline_path=pipeline,
        priors_path=priors,
        scene_id="scene0001_00",
        scene=scene,
        output_path=tmp_path / "output.json",
        progress_path=tmp_path / "progress.txt",
        diagnostics_path=tmp_path / "diagnostics.json",
        bank_path=tmp_path / "bank",
        action="replay",
        mode="class",
        seed=42,
        python_bin=override_python,
    )

    assert _option(command, "--category-denoise-action") == "replay"
    assert _option(command, "--category-denoise-mode") == "class"
    assert _option(command, "--seed") == "42"
    assert _option(command, "--prior-mode") == "off"
    assert _option(command, "--clustering-mode") == "legacy"
    assert _option(command, "--python") == str(override_python.resolve())
    assert "--disable-other-classes" in command
    forbidden = {
        "--gt-dir",
        "--clean",
        "--iterations",
        "--download",
        "--schedule-hash",
        "--artifact-hash",
        "--lock",
    }
    assert forbidden.isdisjoint(command)


def test_bank_and_two_replays_are_ordered_resumable_and_share_one_bank(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, repo, priors, scene_id = _runtime_fixture(tmp_path)
    bank_output = tmp_path / "bank-output"
    replay_output = tmp_path / "replay-output"
    calls: list[tuple[str, str, str, str]] = []

    def fake_run(command, *, cwd: Path, log_path: Path) -> int:
        action = _option(command, "--category-denoise-action")
        mode = _option(command, "--category-denoise-mode")
        bank_path = _option(command, "--category-denoise-bank-path")
        output_path = Path(_option(command, "--json-path"))
        calls.append((action, mode, bank_path, str(output_path)))
        if action == "bank":
            _write_complete_bank(Path(bank_path), scene_id=scene_id)
        else:
            _write_complete_replay(output_path.parent, scene_id=scene_id, mode=mode)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("complete\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(
        "category_priors.category_denoise_runner._run_command", fake_run
    )

    first_bank = run_category_denoise_bank(
        manifest, bank_output, repo, priors, [scene_id]
    )
    second_bank = run_category_denoise_bank(
        manifest, bank_output, repo, priors, [scene_id]
    )
    first_replay = replay_category_denoise(
        manifest,
        bank_output / "bank",
        replay_output,
        repo,
        priors,
        [scene_id],
        mode=("uniform", "class"),
    )
    second_replay = replay_category_denoise(
        manifest,
        bank_output / "bank",
        replay_output,
        repo,
        priors,
        [scene_id],
        mode=("uniform", "class"),
    )

    assert [row[0:2] for row in calls] == [
        ("bank", "uniform"),
        ("replay", "uniform"),
        ("replay", "class"),
    ]
    assert calls[1][2] == calls[2][2]
    assert calls[1][3] != calls[2][3]
    assert first_bank["complete"] == 1
    assert second_bank["runs"][0]["status"] == "skipped_complete"
    assert first_replay["complete"] == 2
    assert all(row["status"] == "skipped_complete" for row in second_replay["runs"])
