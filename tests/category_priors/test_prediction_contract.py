from __future__ import annotations

import numpy as np
import pytest

from category_priors.prediction_contract import (
    normalize_prediction,
    normalize_score,
    validate_prediction_contract,
)


def test_contract_projects_orphans_and_reindexes_instances() -> None:
    raw = np.asarray([8, 8, 4, 9, -1, 4])
    result = normalize_prediction(
        raw,
        {
            "8": {"class": "book", "score": 0.8},
            "4": {"class": "chair", "score": 0.7},
            "-1": {"class": "cabinet", "score": 1.0},
        },
    )

    assert result.point_labels.tolist() == [1, 1, 0, -1, -1, 0]
    assert list(result.instances) == ["0", "1"]
    assert result.audit["ignored_negative_metadata_ids"] == [-1]
    assert result.audit["orphan_instance_ids"] == [9]
    assert result.audit["orphan_gaussian_count"] == 1


def test_contract_drops_empty_metadata_and_requires_score() -> None:
    result = normalize_prediction(
        [2, -1],
        {
            "2": {"class": "book", "score": 0.5},
            "7": {"class": "chair", "score": 0.4},
        },
    )
    assert result.audit["empty_declared_instance_ids"] == [7]
    with pytest.raises(ValueError, match="score must be numeric"):
        normalize_prediction([2], {"2": {"class": "book"}})


def test_contract_does_not_mutate_inputs() -> None:
    raw = np.asarray([5, -1])
    metadata = {"5": {"class": "book", "score": 0.9}}
    normalize_prediction(raw, metadata)
    assert raw.tolist() == [5, -1]
    assert metadata == {"5": {"class": "book", "score": 0.9}}


@pytest.mark.parametrize("score", [-0.01, 1.01, float("nan"), float("inf"), True])
def test_contract_rejects_scores_outside_shared_unit_interval(score) -> None:
    with pytest.raises(ValueError, match="score"):
        normalize_prediction([0], {"0": {"class": "book", "score": score}})
    with pytest.raises(ValueError, match="score"):
        normalize_score(score)


def test_shared_validator_accepts_only_the_normalized_contract() -> None:
    normalized = normalize_prediction(
        [4, -1], {"4": {"class": "book", "score": np.float32(0.25)}}
    )
    validate_prediction_contract(normalized.point_labels, normalized.instances)
    with pytest.raises(ValueError, match="agree exactly"):
        validate_prediction_contract(np.asarray([0, 1]), normalized.instances)
