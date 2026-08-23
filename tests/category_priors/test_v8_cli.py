from category_priors.cli import build_parser


def test_v8_cli_exposes_only_the_registered_public_entry_points() -> None:
    parser = build_parser()
    subparsers = next(
        action for action in parser._actions if hasattr(action, "choices") and action.choices
    )
    for name in (
        "run-v8-lifting-audit",
        "run-v8-bank",
        "replay-v8-priors",
        "evaluate-v8",
    ):
        assert name in subparsers.choices
    for retired in (
        "run-v7-bank",
        "evaluate-v7-bank",
        "replay-v7-priors",
        "evaluate-v7",
    ):
        assert retired not in subparsers.choices
