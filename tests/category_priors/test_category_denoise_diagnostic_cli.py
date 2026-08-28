from __future__ import annotations

import pytest

from category_priors.cli import build_parser


DIAGNOSTIC_COMMANDS = {
    "diagnose-category-denoise-funnel",
    "prepare-category-denoise-knn-oracle",
    "replay-category-denoise-knn-oracle",
    "evaluate-category-denoise-knn-oracle",
    "diagnose-category-prior-oracle",
}


def test_diagnostic_commands_are_public_and_have_help(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert DIAGNOSTIC_COMMANDS <= set(choices)

    for command in sorted(DIAGNOSTIC_COMMANDS):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args([command, "--help"])
        assert exc.value.code == 0
        assert command in capsys.readouterr().out


def test_knn_replay_cli_has_no_ground_truth_surface() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "replay-category-denoise-knn-oracle",
            "--runtime-manifest",
            "runtime.json",
            "--bank-root",
            "bank",
            "--b0-root",
            "bank",
            "--oracle-plan",
            "plan.json",
            "--output-root",
            "replay",
        ]
    )
    assert args.command == "replay-category-denoise-knn-oracle"
    assert not hasattr(args, "gt_dir")
    assert not hasattr(args, "radius_m")
    assert not hasattr(args, "min_region_size")
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "replay-category-denoise-knn-oracle",
                "--runtime-manifest",
                "runtime.json",
                "--bank-root",
                "bank",
                "--b0-root",
                "bank",
                "--oracle-plan",
                "plan.json",
                "--output-root",
                "replay",
                "--gt-dir",
                "gt",
            ]
        )


def test_prepare_and_evaluate_keep_gt_on_evaluation_side() -> None:
    parser = build_parser()
    prepared = parser.parse_args(
        [
            "prepare-category-denoise-knn-oracle",
            "--runtime-manifest",
            "runtime.json",
            "--gt-dir",
            "gt",
            "--bank-root",
            "bank",
            "--output",
            "plan.json",
            "--scene",
            "scene0001_00",
        ]
    )
    evaluated = parser.parse_args(
        [
            "evaluate-category-denoise-knn-oracle",
            "--runtime-manifest",
            "runtime.json",
            "--gt-dir",
            "gt",
            "--prediction-root",
            "replay",
            "--oracle-plan",
            "plan.json",
            "--output-dir",
            "evaluation",
        ]
    )
    assert prepared.gt_dir == "gt"
    assert prepared.iou_threshold == 0.50
    assert evaluated.gt_dir == "gt"
