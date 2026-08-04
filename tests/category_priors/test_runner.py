from __future__ import annotations

from category_priors.io import hash_json, write_json
from category_priors.runner import build_postprocess_command, execute_schedule


def run(condition: str) -> dict[str, object]:
    return {"condition": condition, "scene_id": "scene0000_00", "run_seed": 42}


def scene(tmp_path) -> dict[str, object]:
    return {"base_path": str(tmp_path / "scene"), "scene_scale_m_per_unit": 1.0}


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
