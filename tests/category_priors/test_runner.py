from __future__ import annotations

from category_priors.io import hash_json, sha256_file, write_json
from category_priors.runner import build_postprocess_command, execute_schedule


def run(condition: str) -> dict[str, object]:
    return {"condition": condition, "scene_id": "scene0000_00", "run_seed": 42}


def scene(tmp_path) -> dict[str, object]:
    return {
        "base_path": str(tmp_path / "scene"),
        "scene_scale_m_per_unit": 1.0,
        "python_bin": str(tmp_path / "env" / "python"),
    }


def test_baseline_conditions_are_distinct(tmp_path) -> None:
    b0, _ = build_postprocess_command(
        tmp_path / "run_pipeline.sh",
        run("B0-legacy"),
        scene(tmp_path),
        tmp_path / "runs",
        None,
        None,
    )
    b1, _ = build_postprocess_command(
        tmp_path / "run_pipeline.sh",
        run("B1-other-classes"),
        scene(tmp_path),
        tmp_path / "runs",
        None,
        None,
    )
    assert "--disable-other-classes" in b0
    assert "--disable-other-classes" not in b1
    assert b0[b0.index("--python") + 1] == str(tmp_path / "env" / "python")


def test_factorial_and_gate_arguments_are_exact(tmp_path) -> None:
    command, paths = build_postprocess_command(
        tmp_path / "run_pipeline.sh",
        run("P111-no-gate"),
        scene(tmp_path),
        tmp_path / "runs",
        tmp_path / "priors.json",
        tmp_path / "mapping.json",
    )
    assert command[command.index("--prior-mode") + 1] == "combined"
    assert command[command.index("--prior-gate") + 1] == "off"
    assert "--clean" not in command
    assert paths["metadata_json"].name == "output.json.metadata.json"


def test_search_config_gets_isolated_output_directory(tmp_path) -> None:
    search_run = {
        **run("P000-B2"),
        "config_id": "global-007",
    }
    _, paths = build_postprocess_command(
        tmp_path / "run_pipeline.sh",
        search_run,
        scene(tmp_path),
        tmp_path / "runs",
        tmp_path / "priors.json",
        tmp_path / "mapping.json",
    )
    assert paths["run_dir"] == (
        tmp_path
        / "runs"
        / "P000-B2"
        / "global-007"
        / "scene0000_00"
        / "seed-42"
    )


def test_dry_run_writes_a_resumable_execution_manifest(tmp_path) -> None:
    schedule_path = tmp_path / "schedule.json"
    schedule = {
        "kind": "run_schedule",
        "runs": [
            {
                "sequence": 0,
                "scene_id": "scene0000_00",
                "condition": "B0-legacy",
                "run_seed": 42,
            }
        ],
    }
    schedule["content_sha256"] = hash_json(schedule)
    write_json(schedule_path, schedule)
    scene_manifest = tmp_path / "scenes.json"
    write_json(
        scene_manifest,
        {
            "kind": "scene_runtime_manifest",
            "scenes": [
                {
                    "scene_id": "scene0000_00",
                    "base_path": "scene",
                    "scene_scale_m_per_unit": 1.0,
                }
            ],
        },
    )
    pipeline = tmp_path / "run_pipeline.sh"
    pipeline.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    result_path = tmp_path / "execution.json"
    result = execute_schedule(
        schedule_path,
        scene_manifest,
        tmp_path / "runs",
        result_path,
        pipeline,
        dry_run=True,
    )
    assert result["runs"][0]["status"] == "planned"
    assert result_path.is_file()


def test_resume_preserves_recorded_runtime(tmp_path) -> None:
    schedule_path = tmp_path / "schedule.json"
    schedule = {
        "kind": "run_schedule",
        "runs": [
            {
                "sequence": 0,
                "scene_id": "scene0000_00",
                "condition": "B0-legacy",
                "run_seed": 42,
            }
        ],
    }
    schedule["content_sha256"] = hash_json(schedule)
    write_json(schedule_path, schedule)
    scene_manifest = tmp_path / "scenes.json"
    write_json(
        scene_manifest,
        {
            "kind": "scene_runtime_manifest",
            "scenes": [
                {
                    "scene_id": "scene0000_00",
                    "base_path": "scene",
                    "scene_scale_m_per_unit": 1.0,
                }
            ],
        },
    )
    pipeline = tmp_path / "run_pipeline.sh"
    pipeline.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    output_root = tmp_path / "runs"
    completed_dir = output_root / "B0-legacy" / "scene0000_00" / "seed-42"
    completed_dir.mkdir(parents=True)
    (completed_dir / "output.json").write_text("{}", encoding="utf-8")
    (completed_dir / "output.json.metadata.json").write_text("{}", encoding="utf-8")
    result_path = tmp_path / "execution.json"
    write_json(
        result_path,
        {
            "kind": "run_execution",
            "schedule_sha256": sha256_file(schedule_path),
            "runs": [{"sequence": 0, "runtime_seconds": 12.5}],
        },
    )
    result = execute_schedule(
        schedule_path,
        scene_manifest,
        output_root,
        result_path,
        pipeline,
        resume=True,
    )
    assert result["runs"][0]["status"] == "skipped_complete"
    assert result["runs"][0]["runtime_seconds"] == 12.5
