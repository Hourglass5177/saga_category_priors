from __future__ import annotations

import json

import numpy as np

from category_priors.v8_bank import build_v8_object_bank, object_bank_is_complete

DEFAULT_CLASSES = (
    "chair", "table", "plant", "flower", "foliage", "tv", "painting",
    "sofa", "cabinet", "bed", "wall", "floor", "ceiling", "person",
    "socket", "remote", "key", "book", "lighting", "switch", "door",
    "window", "lamp", "speaker", "computer", "fan", "refrigerator",
    "robot", "cup", "vase", "phone", "trash can",
)


def test_build_object_bank_from_frozen_lifting_artifact(tmp_path) -> None:
    source = tmp_path / "lifting"
    target = tmp_path / "objects"
    source.mkdir()
    point_count = 12
    ids = np.arange(point_count, dtype=np.int32)
    indptr = np.array([0, point_count, 2 * point_count], dtype=np.int64)
    (source / "lifting_bank.json").write_text(
        json.dumps(
            {
                "schema": "saga-v8-lifting-bank-v1",
                "scene_id": "scene0000_00",
                "git_commit": "test",
                "mask_source": "G",
                "lifting_source": "M1",
                "point_count": point_count,
                "frame_count": 2,
                "fragment_count": 2,
                "classes": list(DEFAULT_CLASSES),
                "arrays_npz": "lifting_bank.npz",
            }
        ),
        encoding="utf-8",
    )
    label_features = np.zeros((len(DEFAULT_CLASSES), 2), dtype=np.float32)
    label_features[:, 1] = 1.0
    label_features[0] = (1.0, 0.0)
    semantic = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (point_count, 1))
    np.savez_compressed(
        source / "lifting_bank.npz",
        xyz_m=np.column_stack((ids, np.zeros(point_count), np.zeros(point_count))).astype(np.float32),
        semantic=semantic,
        label_features=label_features,
        fragment_id=np.array([0, 1], dtype=np.int64),
        fragment_frame=np.array([0, 1], dtype=np.int32),
        fragment_mask_index=np.array([0, 0], dtype=np.int32),
        fragment_full_indptr=indptr,
        fragment_full_ids=np.tile(ids, 2),
        fragment_full_mass=np.full(2 * point_count, 2.0, dtype=np.float32),
        fragment_core_indptr=indptr,
        fragment_core_ids=np.tile(ids, 2),
        fragment_core_mass=np.full(2 * point_count, 2.0, dtype=np.float32),
        frame_visible_indptr=indptr,
        frame_visible_ids=np.tile(ids, 2),
        frame_visible_mass=np.full(2 * point_count, 2.0, dtype=np.float32),
        frame_geometry_abstained=np.array([False, False]),
        frame_grounded_missing=np.array([False, False]),
        semantic_fragment_id=np.array([0, 1], dtype=np.int64),
        semantic_fragment_frame=np.array([0, 1], dtype=np.int32),
        semantic_fragment_class=np.array([0, 0], dtype=np.int16),
        semantic_fragment_full_indptr=indptr,
        semantic_fragment_full_ids=np.tile(ids, 2),
        semantic_fragment_full_mass=np.full(2 * point_count, 2.0, dtype=np.float32),
    )

    metadata = build_v8_object_bank(source, target)

    assert object_bank_is_complete(target)
    assert metadata["valid_track_count"] == 1
    assert metadata["classifiers"]["mv-label"]["candidate_count"] == 1
    assert metadata["classifiers"]["codebook"]["candidate_count"] == 1

    arrays_path = target / "object_bank.npz"
    with np.load(arrays_path, allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    arrays["full_candidate_ids_mv"] = np.array([point_count], dtype=np.int32)
    arrays["full_candidate_indptr_mv"] = np.array([0, 1], dtype=np.int64)
    np.savez_compressed(arrays_path, **arrays)
    assert not object_bank_is_complete(target)
