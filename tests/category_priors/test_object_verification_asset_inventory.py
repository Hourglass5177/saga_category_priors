import numpy as np
import pytest

from category_priors.object_verification.asset_inventory import file_identity, tree_identity, xyz_hash, validate_csr


def test_xyz_hash_preserves_full_point_order():
    xyz = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    assert xyz_hash(xyz) == xyz_hash(xyz.astype(np.float64))
    assert xyz_hash(xyz) != xyz_hash(xyz[::-1])
    with pytest.raises(ValueError):
        xyz_hash([[1, 2, float("nan")]])


def test_csr_rejects_wrong_domain_offsets_and_duplicates():
    assert validate_csr([0, 2, 2], [1, 3], 2, 4)["empty_rows"] == 1
    for offsets, ids in [([0, 2], [1, 1]), ([0, 2], [1, 4]), ([1, 2], [0, 1])]:
        with pytest.raises(ValueError):
            validate_csr(offsets, ids, 1, 4)
    with pytest.raises(ValueError):
        validate_csr([0, 1], [1.0], 1, 4)


def test_tree_hash_covers_names_and_contents_not_absolute_root(tmp_path):
    one, two = tmp_path / "a", tmp_path / "b"
    one.mkdir(); two.mkdir()
    (one / "x").write_bytes(b"abc"); (two / "x").write_bytes(b"abc")
    assert tree_identity(one)["sha256"] == tree_identity(two)["sha256"]
    (two / "x").rename(two / "y")
    assert tree_identity(one)["sha256"] != tree_identity(two)["sha256"]
    assert file_identity(one / "absent")["exists"] is False
