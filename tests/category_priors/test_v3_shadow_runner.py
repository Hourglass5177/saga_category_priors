from __future__ import annotations

from pathlib import Path

from category_priors.v3_shadow_runner import build_v3_shadow_command, v3_shadow_run_paths


def test_v3_shadow_command_captures_both_modes_in_one_legacy_run(tmp_path: Path) -> None:
    scene = {
        "base_path": str(tmp_path / "scene"),
        "python_bin": "/env/python",
        "scene_scale_m_per_unit": 1.0,
    }
    command, paths = build_v3_shadow_command(
        tmp_path / "run_pipeline.sh", scene, tmp_path / "runs", "scene0000_00", 42, "abc123"
    )
    joined = " ".join(command)
    assert "--teacher-prior-mode original" in joined
    assert "--v3-shadow-mode both" in joined
    assert "shadow-{mode}.json" in joined
    assert "branch-labels-{mode}.npz" in joined
    assert "--v3-shadow-scene-id scene0000_00" in joined
    assert paths["exact_json"] != paths["exclusive_json"]


def test_v3_shadow_seed_paths_do_not_overlap(tmp_path: Path) -> None:
    first = v3_shadow_run_paths(tmp_path, "scene0000_00", 42)
    second = v3_shadow_run_paths(tmp_path, "scene0000_00", 3407)
    assert not ({str(value) for value in first.values()} & {str(value) for value in second.values()})
