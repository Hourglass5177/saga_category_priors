from __future__ import annotations

from category_priors.cli import build_parser


def test_public_cli_registers_only_the_four_clean_baseline_entrypoints() -> None:
    parser = build_parser()
    build = parser.parse_args(
        [
            "build-alpha-mask-evidence",
            "--scene-id",
            "scene0000_00",
            "--request-json",
            "request.json",
            "--output-dir",
            "bank",
        ]
    )
    assert build.clean_baseline_command == "build-alpha-mask-evidence"

    consensus = parser.parse_args(
        [
            "run-mask-consensus",
            "--scene-id",
            "scene0000_00",
            "--bank-dir",
            "bank",
            "--output-dir",
            "prediction",
            "--condition",
            "C0-no-prior",
            "--gaussian-count",
            "10",
        ]
    )
    assert consensus.clean_baseline_command == "run-mask-consensus"

    replay = parser.parse_args(
        [
            "replay-size-prior",
            "--scene-id",
            "scene0000_00",
            "--bank-dir",
            "bank",
            "--output-dir",
            "prediction",
            "--condition",
            "D-predicted",
            "--gaussian-count",
            "10",
            "--priors",
            "priors.json",
        ]
    )
    assert replay.clean_baseline_command == "replay-size-prior"

    evaluate = parser.parse_args(
        [
            "evaluate-clean-baseline",
            "--manifest",
            "manifest.json",
            "--output",
            "metrics.json",
        ]
    )
    assert evaluate.clean_baseline_command == "evaluate-clean-baseline"
