from category_priors.cli import build_parser


def test_fragment_merge_commands_are_registered_with_frozen_choices() -> None:
    choices = build_parser()._subparsers._group_actions[0].choices
    assert {
        "build-category-fragment-graph",
        "merge-category-fragments",
        "evaluate-category-fragment-merge",
    }.issubset(choices)

    merge = choices["merge-category-fragments"]
    mode = next(action for action in merge._actions if action.dest == "mode")
    assert tuple(mode.choices) == ("global", "class")

    evaluate = choices["evaluate-category-fragment-merge"]
    phase = next(action for action in evaluate._actions if action.dest == "phase")
    assert tuple(phase.choices) == ("dev2", "dev8")
