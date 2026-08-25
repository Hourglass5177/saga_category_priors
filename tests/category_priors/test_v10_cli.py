from __future__ import annotations

from category_priors.cli import build_parser


def test_v10_cli_registers_four_public_entrypoints() -> None:
    parser = build_parser()
    audit = parser.parse_args(
        [
            "audit-v10-association",
            "--runtime-manifest", "runtime.json",
            "--gt-dir", "gt",
            "--bank-root", "banks",
            "--scene", "scene0645_00",
            "--condition", "VC1",
            "--rows-output", "rows.parquet",
            "--analysis-output", "analysis.json",
        ]
    )
    assert audit.command == "audit-v10-association"

    bank = parser.parse_args(
        [
            "run-v10-view-consensus",
            "--lifting-root", "lifting",
            "--output-root", "banks",
            "--scene", "scene0645_00",
            "--git-commit", "commit",
        ]
    )
    assert bank.condition is None

    replay = parser.parse_args(
        [
            "replay-v10-priors",
            "--bank-root", "banks",
            "--output-root", "replay",
            "--category-priors", "priors.json",
            "--scene", "scene0645_00",
            "--structure-condition", "VC1",
            "--classifier", "mv-label",
            "--acceptance-threshold", "0.2",
            "--git-commit", "commit",
        ]
    )
    assert replay.condition is None

    evaluate = parser.parse_args(
        [
            "evaluate-v10",
            "--runtime-manifest", "runtime.json",
            "--gt-dir", "gt",
            "--replay-root", "replay",
            "--structure-condition", "VC1",
            "--classifier", "mv-label",
            "--scene", "scene0645_00",
            "--metrics-output", "metrics.parquet",
            "--analysis-output", "analysis.json",
        ]
    )
    assert evaluate.command == "evaluate-v10"
