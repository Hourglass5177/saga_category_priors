from __future__ import annotations

import json

import numpy as np

from category_priors.v7_runner import bank_is_complete


def test_complete_bank_is_reused_and_corrupt_bank_is_rejected(tmp_path) -> None:
    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "object_bank.json").write_text(
        json.dumps({"point_count": 3}), encoding="utf-8"
    )
    np.savez_compressed(
        bank / "object_bank.npz",
        core_track_id=np.array([-1, 0, 0]),
        final_track_id=np.array([-1, 0, 0]),
        candidate_labels=np.array([-1, 0, 0]),
    )
    assert bank_is_complete(bank)
    (bank / "object_bank.json").write_text("{broken", encoding="utf-8")
    assert not bank_is_complete(bank)
