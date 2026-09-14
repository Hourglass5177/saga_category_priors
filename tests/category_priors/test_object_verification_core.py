import json
import numpy as np
import pytest

from category_priors.object_verification.artifacts import EvidenceStore, cache_identity, write_once
from category_priors.object_verification.contracts import CandidateSource, EvidenceVersion, RawEvidenceRef
from category_priors.object_verification.measurement import AlphaMeasurement, analytic_alpha


def test_identity_change_does_not_inherit_and_revocation_recomputes():
    a = RawEvidenceRef("a", "1", "construction", "cup-a", "sha", 1)
    b = RawEvidenceRef("b", "2", "construction", "cup-a", "sha2", 2)
    v = EvidenceVersion("object", "cup-a", 0).append((a, b))
    assert len(v.active()) == 2
    revoked = v.revoke(("a",), "camera-linked-explicit-exclusion")
    assert revoked.active() == (b,)
    assert len(v.active()) == 2
    assert v.change_identity("cup-b").active() == ()
    with pytest.raises(ValueError):
        v.revoke(("absent",), "reason")


def test_offline_role_cannot_masquerade_as_model():
    with pytest.raises(ValueError):
        RawEvidenceRef("a", "1", "construction", "cup", "sha", 1, "human")
    source = CandidateSource("c", ("p",), (1,), (1, 2), ("1",), "source")
    assert isinstance(source.support_ids, tuple)


def test_cache_is_role_condition_backend_bound_and_first_value_immutable(tmp_path):
    kwargs = dict(scene="scene0645_00", condition="U1", role="construction", camera="1",
                  inputs={"mask": "masksha"}, algorithm_sha="code", config_sha="config",
                  backend={"backend": "gradient-reference", "module_sha256": "module", "binary_sha256": "binary"})
    identity = cache_identity(**kwargs)
    store = EvidenceStore(tmp_path)
    store.put(identity, {"status": "measured_empty", "score": None})
    assert EvidenceStore(tmp_path).get(identity)["score"] is None
    with pytest.raises(FileExistsError):
        store.put(identity, {"status": "nonempty-better-retry"})
    with pytest.raises(FileNotFoundError):
        store.get(cache_identity(**{**kwargs, "condition": "D1"}))
    with pytest.raises(FileNotFoundError):
        store.get(cache_identity(**{**kwargs, "role": "online_verification"}))
    file = next(tmp_path.rglob("*.json"))
    payload = json.loads(file.read_text())
    payload["result"]["score"] = .99
    file.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="content"):
        store.get(identity)


def test_analytic_alpha_axes_threshold_and_all_contributor_support():
    matrix = np.array([[[.3, .2], [.01, .01]], [[.1, .4], [0, 0]]])
    masks = np.array([[[1, 1], [0, 1]], [[0, 0], [1, 0]]])
    result = analytic_alpha(matrix, masks)
    np.testing.assert_allclose(result.visible_mass, [.8, 1.2])
    np.testing.assert_allclose(result.inside_mass, [[.6, .4], [.2, .8]])
    assert result.valid_pixels == 2
    assert result.qualified(0) == (0,)
    assert result.qualified(1) == (1,)
    with pytest.raises(ValueError):
        AlphaMeasurement(np.array([[2.]]), np.array([1.]), 1)


def test_immutable_artifact_write(tmp_path):
    path = tmp_path / "one.json"
    first = write_once(path, {"round": 1})
    assert write_once(path, {"round": 1}) == first
    with pytest.raises(FileExistsError):
        write_once(path, {"round": 2})
