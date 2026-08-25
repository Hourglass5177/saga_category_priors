from __future__ import annotations

from pathlib import Path

import pytest

from category_priors.io import load_json, write_json
from category_priors.v10_closeout import write_v10_v9_closeout


def test_v10_closeout_withdraws_only_the_premature_v9_conclusion(
    tmp_path: Path,
) -> None:
    source = tmp_path / "v9"
    source.mkdir()
    write_json(
        source / "v9_orchestrator_status.json",
        {
            "state": "stopped",
            "checkpoint": "stage2-objectbank-failed",
            "git_commit": "producer",
            "category_prior_tested": False,
        },
    )
    target = tmp_path / "v10_v9_closeout.json"
    result = write_v10_v9_closeout(
        v9_artifacts_root=source,
        output_path=target,
        git_commit="consumer",
    )
    assert result["passed"] is True
    assert result["category_prior_tested"] is False
    assert "did not prove" in result["withdrawn_conclusion"]
    assert load_json(target) == result
    assert "hash" not in str(result).lower()


def test_v10_closeout_rejects_a_claim_that_prior_was_already_tested(
    tmp_path: Path,
) -> None:
    source = tmp_path / "v9"
    source.mkdir()
    write_json(
        source / "v9_status.json",
        {"state": "stopped", "category_prior_tested": True},
    )
    with pytest.raises(ValueError, match="unexpectedly claims"):
        write_v10_v9_closeout(
            v9_artifacts_root=source,
            output_path=tmp_path / "closeout.json",
            git_commit="consumer",
        )
