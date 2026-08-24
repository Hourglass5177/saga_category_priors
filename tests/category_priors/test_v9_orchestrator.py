from pathlib import Path

from category_priors import v9_orchestrator as module


def _arguments(tmp_path: Path) -> dict[str, object]:
    return {
        "runtime_manifest": tmp_path / "tune.json",
        "locked_runtime_manifest": tmp_path / "locked.json",
        "locked_evaluation_scenes": tmp_path / "locked-scenes.json",
        "workspace": tmp_path / "workspace",
        "runs_root": tmp_path / "runs",
        "artifacts_root": tmp_path / "artifacts",
        "gt_dir": tmp_path / "gt-tune",
        "locked_gt_dir": tmp_path / "gt-locked",
        "sam_reusable_root": tmp_path / "sam-reusable",
        "sam_checkpoint": tmp_path / "sam.pth",
        "label_features": tmp_path / "labels.pt",
        "size_bins": tmp_path / "bins.json",
        "category_priors": tmp_path / "priors.json",
        "git_commit": "v9-test-commit",
    }


def test_orchestrator_honors_stage2_stop(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        module,
        "execute_v9_t1_runs",
        lambda **_: {"runs": [{"status": "complete"}]},
    )
    monkeypatch.setattr(
        module,
        "run_v9_stage2",
        lambda _config: {
            "state": "stopped",
            "checkpoint": "stage2-geometric-oracle-failed",
            "stop_reason": "registered stop",
        },
    )

    def forbidden(_config):  # pragma: no cover - assertion helper
        raise AssertionError("Stage 3 must not run past a Stage 2 stop")

    monkeypatch.setattr(module, "run_v9_stage3_to_6", forbidden)
    result = module.run_v9_orchestrator(**_arguments(tmp_path))
    assert result["state"] == "stopped"
    assert result["checkpoint"] == "stage2-geometric-oracle-failed"


def test_orchestrator_binds_corrected_t1_to_continuation(
    monkeypatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}

    def t1(**kwargs):
        calls["t1"] = kwargs
        return {"runs": [{"status": "skipped_complete"}]}

    monkeypatch.setattr(module, "execute_v9_t1_runs", t1)
    monkeypatch.setattr(
        module,
        "run_v9_stage2",
        lambda config: {
            "state": "complete",
            "checkpoint": "stage2-objectbank-selected",
        },
    )

    def continuation(config):
        calls["continuation"] = config
        return {"state": "complete", "checkpoint": "stage6-final48-complete"}

    monkeypatch.setattr(module, "run_v9_stage3_to_6", continuation)
    result = module.run_v9_orchestrator(**_arguments(tmp_path))

    assert result["state"] == "complete"
    assert calls["t1"]["scene_ids"] == module.V9_T1_DEV8
    config = calls["continuation"]
    assert config.t1_b1_root == (tmp_path / "runs" / "t1-legacy").resolve()
    assert config.t1_b1_condition == "T1-B1"
    assert config.git_commit == "v9-test-commit"
