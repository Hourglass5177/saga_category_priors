from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

import pytest

from category_priors.baseline_closure_budget import (
    ITERATIONS_DEFAULT_NEW,
    ITERATIONS_DEFAULT_OLD,
    materialize_iterations_cli_variant,
    repair_iterations_cli,
)


def test_iterations_cli_repair_preserves_a_falsey_integer_default() -> None:
    repaired = repair_iterations_cli(f"before\n{ITERATIONS_DEFAULT_OLD}\nafter\n")
    assert ITERATIONS_DEFAULT_OLD not in repaired
    assert ITERATIONS_DEFAULT_NEW in repaired


def test_repair_changes_real_argparse_behavior_without_changing_adaptive_truthiness(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
class OptimizationParams:
    def __init__(self, parser):
        self.iterations = None
        value = self.iterations
        parser.add_argument("--iterations", default=value, type=type(value))
"""
    historical: dict[str, object] = {}
    exec(source, historical)  # noqa: S102 - fixed test-only source
    parser = ArgumentParser()
    historical["OptimizationParams"](parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--iterations", "10000"])
    capsys.readouterr()

    repaired_namespace: dict[str, object] = {}
    exec(  # noqa: S102 - fixed test-only source
        repair_iterations_cli(source), repaired_namespace
    )
    repaired_parser = ArgumentParser()
    repaired_namespace["OptimizationParams"](repaired_parser)
    default = repaired_parser.parse_args([]).iterations
    explicit = repaired_parser.parse_args(["--iterations", "10000"]).iterations
    assert default == 0 and not default
    assert explicit == 10000


def test_iterations_cli_repair_requires_exact_historical_sentinel() -> None:
    with pytest.raises(ValueError, match="expected one"):
        repair_iterations_cli("self.iterations = 30000")


def test_materialized_variant_is_isolated_and_auditable(tmp_path: Path) -> None:
    source = tmp_path / "full950"
    (source / "arguments").mkdir(parents=True)
    (source / "arguments/__init__.py").write_text(
        ITERATIONS_DEFAULT_OLD + "\n", encoding="utf-8"
    )
    (source / "train_contrastive_feature.py").write_text("teacher\n", encoding="utf-8")
    destination = tmp_path / "full950-iterations-cli"
    result = materialize_iterations_cli_variant(source, destination)

    assert ITERATIONS_DEFAULT_OLD in (source / "arguments/__init__.py").read_text(
        encoding="utf-8"
    )
    assert ITERATIONS_DEFAULT_NEW in (destination / "arguments/__init__.py").read_text(
        encoding="utf-8"
    )
    assert result["adaptive_semantics_preserved"] is True
    recorded = json.loads(
        (destination / "BASELINE_VARIANT.json").read_text(encoding="utf-8")
    )
    assert recorded["variant"] == "full950-iterations-cli"


def test_existing_budget_variant_is_not_overwritten(tmp_path: Path) -> None:
    source = tmp_path / "full950"
    (source / "arguments").mkdir(parents=True)
    (source / "arguments/__init__.py").write_text(
        ITERATIONS_DEFAULT_OLD, encoding="utf-8"
    )
    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        materialize_iterations_cli_variant(source, destination)
