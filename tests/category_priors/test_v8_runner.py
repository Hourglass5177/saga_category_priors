from __future__ import annotations

import json

import numpy as np

from category_priors.v8_runner import LIFTING_ARMS, lifting_bank_is_complete


def test_factorial_arms_are_complete_and_stable() -> None:
    assert LIFTING_ARMS == ("G-M1", "G-AM", "S-M1", "S-AM")


def test_lifting_bank_complete_rejects_corruption(tmp_path) -> None:
    bank = tmp_path / "scene"
    bank.mkdir()
    (bank / "lifting_bank.json").write_text(
        json.dumps(
            {
                "schema": "saga-v8-lifting-bank-v1",
                "scene_id": "scene",
                "mask_source": "G",
                "lifting_source": "M1",
                "point_count": 3,
                "fragment_count": 1,
                "frame_count": 2,
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        bank / "lifting_bank.npz",
        xyz_m=np.zeros((3, 3), dtype=np.float32),
        opacity=np.ones(3, dtype=np.float32),
        affinity=np.zeros((3, 2), dtype=np.float32),
        semantic=np.zeros((3, 2), dtype=np.float32),
        label_features=np.zeros((2, 2), dtype=np.float32),
        fragment_full_indptr=np.array([0, 2], dtype=np.int64),
        fragment_full_ids=np.array([0, 1], dtype=np.int32),
        fragment_full_mass=np.ones(2, dtype=np.float32),
        fragment_core_indptr=np.array([0, 1], dtype=np.int64),
        fragment_core_ids=np.array([0], dtype=np.int32),
        fragment_core_mass=np.ones(1, dtype=np.float32),
        fragment_core_ratio=np.ones(1, dtype=np.float32),
        fragment_id=np.array([0], dtype=np.int64),
        fragment_frame=np.array([0], dtype=np.int32),
        fragment_mask_index=np.array([0], dtype=np.int32),
        fragment_source_class=np.array([0], dtype=np.int16),
        frame_visible_indptr=np.array([0, 2, 3], dtype=np.int64),
        frame_visible_ids=np.array([0, 1, 2], dtype=np.int32),
        frame_visible_mass=np.ones(3, dtype=np.float32),
        frame_geometry_abstained=np.zeros(2, dtype=np.bool_),
        frame_grounded_missing=np.zeros(2, dtype=np.bool_),
        semantic_fragment_full_indptr=np.array([0, 1], dtype=np.int64),
        semantic_fragment_full_ids=np.array([0], dtype=np.int32),
        semantic_fragment_full_mass=np.ones(1, dtype=np.float32),
        semantic_fragment_id=np.array([0], dtype=np.int64),
        semantic_fragment_frame=np.array([0], dtype=np.int32),
        semantic_fragment_mask_index=np.array([0], dtype=np.int32),
        semantic_fragment_class=np.array([0], dtype=np.int16),
    )
    assert lifting_bank_is_complete(
        bank,
        expected_scene_id="scene",
        expected_mask_source="G",
        expected_lifting_source="M1",
    )
    assert not lifting_bank_is_complete(bank, expected_mask_source="S")
    assert not lifting_bank_is_complete(
        bank, expected_contributor_audit=True
    )
    metadata = json.loads((bank / "lifting_bank.json").read_text())
    metadata["contributor_audit_requested"] = True
    (bank / "lifting_bank.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    assert lifting_bank_is_complete(
        bank, expected_contributor_audit=True
    )

    (bank / "lifting_bank.json").write_text("{broken", encoding="utf-8")
    assert not lifting_bank_is_complete(bank)
