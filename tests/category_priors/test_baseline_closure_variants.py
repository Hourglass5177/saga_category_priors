from __future__ import annotations

import json
from pathlib import Path

import pytest

from category_priors.baseline_closure_variants import (
    ARGS_CALL_OLD,
    ARGS_SIGNATURE_OLD,
    NORMALIZE_NEW,
    NORMALIZE_OLD,
    materialize_training_variant,
    repair_training_source,
)


def _source() -> str:
    return f"{ARGS_SIGNATURE_OLD}\n{NORMALIZE_OLD}\n{ARGS_CALL_OLD}"


def test_args_only_changes_only_executability() -> None:
    repaired = repair_training_source(_source(), "args-only")
    assert ARGS_SIGNATURE_OLD not in repaired
    assert ARGS_CALL_OLD not in repaired
    assert NORMALIZE_OLD in repaired
    assert NORMALIZE_NEW not in repaired


def test_args_norm_adds_channel_normalization() -> None:
    repaired = repair_training_source(_source(), "args-norm")
    assert NORMALIZE_OLD not in repaired
    assert NORMALIZE_NEW in repaired


def test_source_sentinel_mismatch_fails() -> None:
    with pytest.raises(ValueError, match="expected one source sentinel"):
        repair_training_source("not bfc", "args-only")


def test_materialization_is_isolated_and_records_provenance(tmp_path: Path) -> None:
    source = tmp_path / "bfc"
    source.mkdir()
    (source / "train_contrastive_feature.py").write_text(_source(), encoding="utf-8")
    (source / "untouched.txt").write_text("teacher", encoding="utf-8")
    destination = tmp_path / "args-only"
    result = materialize_training_variant(source, destination, "args-only")

    assert (source / "train_contrastive_feature.py").read_text(
        encoding="utf-8"
    ) == _source()
    assert (destination / "untouched.txt").read_text(encoding="utf-8") == "teacher"
    assert result["variant"] == "args-only"
    recorded = json.loads(
        (destination / "BASELINE_VARIANT.json").read_text(encoding="utf-8")
    )
    assert recorded["source_commit"].startswith("bfc2192")


def test_existing_destination_is_not_overwritten(tmp_path: Path) -> None:
    source = tmp_path / "bfc"
    source.mkdir()
    (source / "train_contrastive_feature.py").write_text(_source(), encoding="utf-8")
    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        materialize_training_variant(source, destination, "args-only")
