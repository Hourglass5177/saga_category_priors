from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_recheck_contract_reviews_every_projectable_candidate() -> None:
    standard = (ROOT / "category_priors" / "INSTANCE_RECHECK_BASELINE_STANDARD.md").read_text(
        encoding="utf-8"
    )
    assert "所有至少有一个可用投影视角的候选做复核" in standard
    assert "`raw` 接纳全部候选" in standard
    assert "emergent_unreviewed" not in standard


def test_crop_cap_and_instance_strata_are_frozen_before_implementation() -> None:
    standard = (ROOT / "category_priors" / "INSTANCE_RECHECK_BASELINE_STANDARD.md").read_text(
        encoding="utf-8"
    )
    assert "min(crop_side, max(image_width, image_height))" in standard
    assert "0.886095877588466 m" in standard
    assert "socket、speaker、switch、fan、refrigerator、cup、phone" in standard


def test_retired_experiments_have_a_navigation_index() -> None:
    index = (ROOT / "docs" / "RETIRED_EXPERIMENT_INDEX.md").read_text(
        encoding="utf-8"
    )
    for anchor in ("74a745b", "b41533e", "e6a9bfe"):
        assert anchor in index
