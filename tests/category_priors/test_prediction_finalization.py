from __future__ import annotations

import json

import numpy as np
import pytest

from category_priors.prediction_finalization import (
    finalize_prediction,
    prediction_output_payload,
    write_prediction_output_atomic,
)


def _classes() -> tuple[str, ...]:
    return ("chair", "table", "wall")


def test_shared_finalizer_applies_vote_bbox_and_export_mapping() -> None:
    labels = np.asarray([10, 10, 3, 3, -1], dtype=np.int64)
    xyz = np.asarray(
        [[0, 0, 0], [1, 1, 1], [2, 0, 0], [3, 1, 1], [9, 9, 9]],
        dtype=np.float64,
    )
    finalized = finalize_prediction(
        point_labels=labels,
        xyz_scene=xyz,
        is_big_gaussian=np.zeros(5, dtype=bool),
        vote_ratios_by_raw={3: [0.7, 0.2, 0.0], 10: [0.1, 0.8, 0.0]},
        class_names=_classes(),
        selected_classes=("chair", "table"),
        label_threshold=0.3,
    )

    assert finalized.class_by_raw == {3: "chair", 10: "table"}
    assert finalized.score_by_raw[3] == pytest.approx(0.7)
    assert len(finalized.bbox_by_raw[3]) == 24
    assert dict(finalized.contracted.export_id_by_raw) == {3: 0, 10: 1}
    assert finalized.contracted.point_labels.tolist() == [1, 1, 0, 0, -1]


def test_finalizer_projects_background_and_non_saga_classes_out() -> None:
    finalized = finalize_prediction(
        point_labels=[0, 0, 1, 1],
        xyz_scene=np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]),
        is_big_gaussian=[False] * 4,
        vote_ratios_by_raw={0: [0.2, 0.1, 0.0], 1: [0.1, 0.2, 0.6]},
        class_names=_classes(),
        selected_classes=("chair", "table"),
        label_threshold=0.3,
    )
    assert finalized.class_by_raw == {0: "background", 1: "wall"}
    assert finalized.contracted.instances == {}
    assert finalized.contracted.point_labels.tolist() == [-1, -1, -1, -1]


def test_output_payload_and_atomic_writer_share_the_contract(tmp_path) -> None:
    finalized = finalize_prediction(
        point_labels=[4],
        xyz_scene=[[1.0, 2.0, 3.0]],
        is_big_gaussian=[False],
        vote_ratios_by_raw={4: [0.9, 0.0, 0.0]},
        class_names=_classes(),
        selected_classes=("chair",),
        label_threshold=0.3,
    )
    payload = prediction_output_payload(
        finalized,
        is_big_gaussian=[False],
        is_transparent_gaussian=[True],
    )
    destination = tmp_path / "output.json"
    write_prediction_output_atomic(destination, payload)
    observed = json.loads(destination.read_text(encoding="utf-8"))
    assert observed["point_labels"] == [0]
    assert observed["is_transparent_gaissian"] == [True]
    assert not (tmp_path / "output.json.part").exists()


def test_finalizer_requires_vote_for_every_raw_instance() -> None:
    with pytest.raises(ValueError, match="missing final vote ratios"):
        finalize_prediction(
            point_labels=[2],
            xyz_scene=[[0.0, 0.0, 0.0]],
            is_big_gaussian=[False],
            vote_ratios_by_raw={},
            class_names=_classes(),
            selected_classes=("chair",),
            label_threshold=0.3,
        )

