from __future__ import annotations

import pytest

from category_priors.cli import build_parser


def test_category_denoise_public_commands_are_registered() -> None:
    choices = build_parser()._subparsers._group_actions[0].choices

    assert {
        "run-category-denoise-bank",
        "replay-category-denoise",
        "evaluate-category-denoise",
    } <= set(choices)


def test_category_denoise_bank_and_replay_do_not_accept_ground_truth() -> None:
    parser = build_parser()
    bank = parser.parse_args(
        [
            "run-category-denoise-bank",
            "--runtime-manifest",
            "runtime.json",
            "--output-root",
            "banks",
            "--category-priors",
            "priors.json",
            "--scene",
            "scene0001_00",
        ]
    )
    replay = parser.parse_args(
        [
            "replay-category-denoise",
            "--runtime-manifest",
            "runtime.json",
            "--bank-root",
            "banks/bank",
            "--output-root",
            "replay",
            "--category-priors",
            "priors.json",
            "--mode",
            "class",
            "--scene",
            "scene0001_00",
        ]
    )

    assert bank.command == "run-category-denoise-bank"
    assert replay.command == "replay-category-denoise"
    assert replay.mode == "class"
    assert not hasattr(bank, "gt_dir")
    assert not hasattr(replay, "gt_dir")


def test_category_denoise_replay_rejects_unknown_mode() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "replay-category-denoise",
                "--runtime-manifest",
                "runtime.json",
                "--bank-root",
                "banks/bank",
                "--output-root",
                "replay",
                "--category-priors",
                "priors.json",
                "--mode",
                "combined",
                "--scene",
                "scene0001_00",
            ]
        )


def test_category_denoise_evaluator_is_the_only_entrypoint_with_gt() -> None:
    args = build_parser().parse_args(
        [
            "evaluate-category-denoise",
            "--runtime-manifest",
            "runtime.json",
            "--gt-dir",
            "gt",
            "--bank-root",
            "banks/bank",
            "--prediction-root",
            "replay",
            "--scene",
            "scene0001_00",
            "--condition",
            "uniform",
            "--condition",
            "class",
            "--metrics-output",
            "metrics.parquet",
            "--analysis-output",
            "analysis.json",
        ]
    )

    assert args.gt_dir == "gt"
    assert args.condition == ["uniform", "class"]
