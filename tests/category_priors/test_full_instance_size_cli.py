from __future__ import annotations

import sys
import types
from pathlib import Path

from category_priors.cli import build_parser


def test_full_instance_size_command_parses_without_loading_experiment() -> None:
    module_name = "category_priors.full_instance_size_experiment"
    sys.modules.pop(module_name, None)

    args = build_parser().parse_args(
        [
            "run-full-instance-size-prior",
            "--workspace",
            "workspace",
            "--runtime-manifest",
            "runtime.json",
            "--locked-runtime-manifest",
            "locked-runtime.json",
            "--t1-root",
            "t1",
            "--rebuild-t1-root",
            "rebuild",
            "--gt-dir",
            "gt",
            "--locked-gt-dir",
            "locked-gt",
            "--train-stats",
            "train-stats.parquet",
            "--category-priors",
            "priors.json",
            "--size-bins",
            "sizes.json",
            "--locked-evaluation-scenes",
            "final.json",
            "--runs-root",
            "runs",
            "--artifacts-root",
            "artifacts",
            "--taxonomy",
            "taxonomy.json",
            "--git-commit",
            "abc123",
            "--python-bin",
            "environment/bin/python",
            "--allow-rebuild-missing-traces",
        ]
    )

    assert args.command == "run-full-instance-size-prior"
    assert args.workspace == Path("workspace")
    assert args.runtime_manifest == Path("runtime.json")
    assert args.locked_runtime_manifest == Path("locked-runtime.json")
    assert args.locked_gt_dir == Path("locked-gt")
    assert args.train_stats == Path("train-stats.parquet")
    assert args.rebuild_t1_root == Path("rebuild")
    assert args.locked_evaluation_scenes == Path("final.json")
    assert args.python_bin == Path("environment/bin/python")
    assert args.allow_rebuild_missing_traces is True
    assert args.disk_floor_gib == 80.0
    assert args.cgroup_root == Path("/sys/fs/cgroup")
    assert module_name not in sys.modules


def test_full_instance_size_command_forwards_python_bin_to_worker(
    monkeypatch,
) -> None:
    captured: list[str] = []

    def fake_main(argv: list[str]) -> int:
        captured.extend(argv)
        return 0

    experiment = types.ModuleType("category_priors.full_instance_size_experiment")
    experiment.main = fake_main  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, experiment.__name__, experiment)
    args = build_parser().parse_args(
        [
            "run-full-instance-size-prior",
            "--workspace", "workspace",
            "--runtime-manifest", "runtime.json",
            "--locked-runtime-manifest", "locked-runtime.json",
            "--t1-root", "t1",
            "--gt-dir", "gt",
            "--locked-gt-dir", "locked-gt",
            "--train-stats", "train-stats.parquet",
            "--category-priors", "priors.json",
            "--size-bins", "sizes.json",
            "--locked-evaluation-scenes", "final.json",
            "--runs-root", "runs",
            "--artifacts-root", "artifacts",
            "--python-bin", "preflight/bin/python",
        ]
    )

    args.func(args)

    position = captured.index("--python-bin")
    assert Path(captured[position + 1]) == Path("preflight/bin/python")
