from __future__ import annotations

from pathlib import Path

import pytest

from category_priors.priors import fit_priors, validate_priors
from category_priors.taxonomy import Taxonomy, load_taxonomy


def test_default_taxonomy_is_exact_and_complete() -> None:
    taxonomy = load_taxonomy()
    mapping = taxonomy.dataset_mappings["scannet200"]
    assert len(taxonomy.canonical_classes) == 20
    assert set(mapping.values()) == set(taxonomy.canonical_classes)
    assert taxonomy.map_label("scannet200", "power_outlet") == "socket"
    assert taxonomy.map_label("scannet200", "telephone") == "phone"
    assert taxonomy.map_label("scannet200", "wall") is None


def test_v1_taxonomy_rejects_many_to_one_mapping() -> None:
    taxonomy = Taxonomy(
        schema_version="1.0",
        benchmark_name="test",
        canonical_classes=("chair",),
        dataset_mappings={"test": {"chair": "chair", "seat": "chair"}},
        parents={"chair": "global"},
        unsupported_saga_classes=(),
        content_hash="unused",
    )
    with pytest.raises(ValueError, match="one-to-one"):
        taxonomy.validate()


def test_fit_is_train_only_and_hash_valid(tmp_path: Path, stats_rows) -> None:
    source = tmp_path / "stats.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    rows = [dict(row) for row in stats_rows]
    rows[0] = {**rows[0], "split": "val"}
    with pytest.raises(ValueError, match="train-only"):
        fit_priors(rows, load_taxonomy(), source, bootstrap_samples=10)

    priors = fit_priors(
        stats_rows,
        load_taxonomy(),
        source,
        bootstrap_samples=20,
        min_physical_scenes=3,
    )
    validate_priors(priors)
    assert priors["categories"]["chair"]["active"] is True
    assert (
        priors["categories"]["cup"]["small_score"]
        > priors["categories"]["chair"]["small_score"]
    )
    assert priors["categories"]["phone"]["active"] is False
