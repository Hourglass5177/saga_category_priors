from __future__ import annotations

from pathlib import Path

import pytest

import category_priors.baseline_closure_precision as precision


def test_precision_parser_accepts_closeout_paths() -> None:
    args = precision.build_parser().parse_args(
        [
            "--closure-root",
            "closure",
            "--gt-dir",
            "gt",
            "--runtime-manifest",
            "runtime.json",
            "--output-dir",
            "artifacts",
        ]
    )
    assert args.closure_root == Path("closure")
    assert precision.AUDIT_RADII_M == (0.02, 0.05, 0.10)


def test_precision_refuses_to_invent_missing_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(precision, "_output_runs", lambda _root: iter(()))
    with pytest.raises(FileNotFoundError, match="No baseline output"):
        precision.evaluate_teacher_handoff_precision(
            closure_root=tmp_path / "closure",
            gt_dir=tmp_path / "gt",
            runtime_manifest=tmp_path / "runtime.json",
            output_dir=tmp_path / "artifacts",
        )
