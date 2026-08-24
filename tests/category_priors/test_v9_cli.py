from category_priors.cli import build_parser


def test_v9_public_commands_are_registered() -> None:
    parser = build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert {
        "audit-teacher-baseline",
        "train-object-features-10k",
        "run-object-bank",
        "replay-object-priors",
        "evaluate-object-system",
    } <= set(choices)


def test_replay_v9_parser_freezes_explicit_structure() -> None:
    args = build_parser().parse_args(
        [
            "replay-object-priors",
            "--bank-root",
            "bank",
            "--output-root",
            "out",
            "--category-priors",
            "priors.json",
            "--association-mode",
            "A2",
            "--classifier",
            "mv-label",
            "--scene",
            "scene0000_00",
            "--acceptance-threshold",
            "0.15",
        ]
    )
    assert args.association_mode == "A2"
    assert args.classifier == "mv-label"
    assert args.acceptance_threshold == 0.15
