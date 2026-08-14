from __future__ import annotations

from pathlib import Path

import numpy as np

from category_priors.io import load_json
from category_priors.v3_shadow import (
    all_class_top1_labels,
    candidate_survival,
    label_overlap,
    load_shadow_arrays,
    target_top1_masks,
    vote_summary,
    write_shadow_capture,
)


def test_top1_competes_over_complete_codebook_before_target_filter() -> None:
    semantic = np.asarray([[0.8, 0.6], [1.0, 0.0], [0.4, 0.3]])
    labels = np.asarray([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
    masks, winners, scores, margins = target_top1_masks(
        semantic, labels, (0,), threshold=0.7
    )
    assert winners.tolist() == [1, 0, 1]
    assert masks[0].tolist() == [False, True, False]
    assert np.all(scores >= 0.7)
    assert np.all(margins >= 0)


def test_top1_rejects_below_threshold() -> None:
    winners, _, _ = all_class_top1_labels(
        np.asarray([[1.0, 1.0]]), np.asarray([[1.0, 0.0], [0.0, 1.0]]), threshold=0.8
    )
    assert winners.tolist() == [-1]


def test_candidate_survival_overlap_and_vote_summary() -> None:
    mask = np.asarray([True, True, True, False])
    survival = candidate_survival(mask, 7, [7, 7, 2, -1], [7, -1, 2, -1])
    assert survival["after_knn_points"] == 2
    assert survival["after_filter_points"] == 1
    assert label_overlap([3, 3, 4, -1], mask) == {
        "instance_id": 3,
        "point_count": 2,
        "fraction": 2 / 3,
    }
    vote = vote_summary([0.2, 0.3], ("chair", "cup"), "cup")
    assert vote["winner"] == "background"
    assert vote["background_ratio"] == 0.5
    assert not vote["winner_matches_branch"]


def test_shadow_capture_roundtrip_uses_one_compact_npz(tmp_path: Path) -> None:
    json_path = tmp_path / "shadow.json"
    labels_path = tmp_path / "branch_labels.npz"
    payload = write_shadow_capture(
        json_path=json_path,
        labels_path=labels_path,
        scene_id="scene0000_00",
        seed=42,
        mode="exact",
        git_commit="abc123",
        class_names=("chair", "cup", "wall"),
        affinity_gate=[1.0, 0.5],
        branch_labels=[0, -1, 1, -1, 1],
        semantic_top1=[0, 1, 2, -1, 2],
        semantic_top1_score=[0.9, 0.8, 0.7, 0.2, 0.95],
        semantic_margin=[0.2, 0.1, 0.05, 0.01, 0.3],
        sam_covered=[True, False, True, False, True],
        candidates=[{"candidate_id": 0, "branch_class": "chair"}],
        class_diagnostics={"chair": {"candidate_points": 1}},
    )
    assert payload["candidate_count"] == 1
    assert load_json(json_path)["kind"] == "v3_shadow_capture"
    arrays = load_shadow_arrays(labels_path)
    assert arrays["branch_labels"].tolist() == [0, -1, 1, -1, 1]
    assert arrays["sam_covered"].tolist() == [True, False, True, False, True]
