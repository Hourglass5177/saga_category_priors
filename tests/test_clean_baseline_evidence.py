from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from category_priors.clean_baseline import (
    AlphaMaskEvidenceBank,
    EvidenceThresholds,
    accumulate_alpha_mass_from_contributors,
    build_alpha_mask_evidence,
    build_frame_evidence,
    build_sparse_frame_evidence,
    evidence_bank_is_complete,
    evidence_request_source,
    load_evidence_bank,
    save_evidence_bank,
)
from category_priors.clean_baseline.worker import DEFAULT_CLASSES


CLASSES = DEFAULT_CLASSES
COMMIT = "a" * 40


def _materialize_identity_inputs(base: Path):
    from category_priors.clean_baseline import worker

    values = {
        "rgb_ply": base / "points.ply",
        "sparse": base / "sparse",
        "images": base / "images",
        "sam_masks": base / "sam",
        "grounded_masks": base / "grounded-masks",
        "grounded_labels": base / "grounded-labels",
    }
    values["rgb_ply"].write_bytes(b"ply-input-v1")
    for directory in values.values():
        if directory == values["rgb_ply"]:
            continue
        directory.mkdir(parents=True, exist_ok=True)
    (values["sparse"] / "images.bin").write_bytes(b"colmap-images")
    (values["sparse"] / "cameras.bin").write_bytes(b"colmap-cameras")
    (values["images"] / "zero.jpg").write_bytes(b"rgb-frame")
    (values["sam_masks"] / "zero.jpg.npz").write_bytes(b"packed-sam")
    (values["grounded_masks"] / "zero.jpg.pt").write_bytes(b"grounded-mask")
    (values["grounded_labels"] / "zero.jpg.pt").write_bytes(b"grounded-label")
    return worker.CleanSceneInputs(base_path=base, **values)


def _mass(masks: np.ndarray | None = None):
    # Two pixels, two contributors.  The normalized masses are [.75, .25]
    # and [.10, .90], respectively.
    alpha = np.asarray([[[0.6, 0.2], [0.1, 0.9]]], dtype=np.float64)
    transmittance = np.ones_like(alpha)
    ids = np.asarray([[[0, 1], [0, 1]]], dtype=np.int32)
    return accumulate_alpha_mass_from_contributors(
        alpha, transmittance, ids, masks, point_count=3
    )


def _bank() -> AlphaMaskEvidenceBank:
    masks = np.asarray([[[1, 0]], [[1, 0]]], dtype=bool)
    mass = _mass(masks)
    semantics = np.zeros((2, 32), dtype=np.float32)
    semantics[0, 3] = 1
    semantics[1, 7] = 1
    frame0 = build_frame_evidence(
        frame_id=10,
        image_name="000010.jpg",
        alpha_mass=mass,
        global_mask_id_start=100,
        semantic_posteriors=semantics,
    )
    abstained_mass = _mass(None)
    frame1 = build_frame_evidence(
        frame_id=20,
        image_name="000020.jpg",
        alpha_mass=abstained_mass,
        global_mask_id_start=200,
    )
    return AlphaMaskEvidenceBank.from_frames(
        scene_id="scene0000_00",
        point_count=3,
        xyz_m=np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32),
        class_names=CLASSES,
        frames=(frame1, frame0),  # construction sorts by stable frame ID
        thresholds=EvidenceThresholds(),
        source={"point_cloud": "/read-only/point_cloud.ply", "commit": "abc"},
    )


def test_alpha_tprev_mass_is_normalized_per_pixel() -> None:
    mass = _mass(np.asarray([[[1, 0]]], dtype=bool))
    np.testing.assert_allclose(mass.visible_mass, [0.85, 1.15, 0.0])
    np.testing.assert_allclose(mass.inside_mass[0], [0.75, 0.25, 0.0])
    assert mass.valid_pixel_count == 2
    assert not mass.geometry_abstained


def test_duplicate_contributor_ids_accumulate_and_empty_pixels_do_not() -> None:
    alpha = np.asarray([[[0.2, 0.3, 0.0], [0.0, 0.0, 0.0]]])
    previous = np.ones_like(alpha)
    ids = np.asarray([[[1, 1, -1], [-1, -1, -1]]], dtype=np.int32)
    mass = accumulate_alpha_mass_from_contributors(
        alpha, previous, ids, np.asarray([[[1, 1]]]), point_count=3
    )
    np.testing.assert_allclose(mass.visible_mass, [0, 1, 0])
    np.testing.assert_allclose(mass.inside_mass[0], [0, 1, 0])
    assert mass.valid_pixel_count == 1


def test_positive_mass_with_invalid_id_and_wrong_tprev_are_rejected() -> None:
    alpha = np.asarray([[[0.5]]])
    with pytest.raises(ValueError, match="invalid Gaussian ID"):
        accumulate_alpha_mass_from_contributors(
            alpha, np.ones_like(alpha), np.asarray([[[-1]]]), None, 2
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        accumulate_alpha_mass_from_contributors(
            alpha, np.asarray([[[1.1]]]), np.asarray([[[0]]]), None, 2
        )


def test_geometry_abstention_preserves_visibility_but_creates_no_mask_evidence() -> None:
    mass = _mass(None)
    assert mass.geometry_abstained
    assert mass.inside_mass.shape == (0, 3)
    np.testing.assert_allclose(mass.visible_mass, [0.85, 1.15, 0])
    frame = build_frame_evidence(
        frame_id=1, image_name="one.jpg", alpha_mass=mass
    )
    assert frame.metadata.geometry_abstained
    assert frame.metadata.semantic_abstained
    assert not frame.masks
    np.testing.assert_array_equal(frame.visibility.row(0)[0], [0, 1])


def test_thresholds_and_same_frame_ambiguity_are_explicit() -> None:
    bank = _bank()
    # Gaussian 0 passes both identical same-frame masks, so it is retained for
    # audit but excluded from positive consensus support by default.
    np.testing.assert_array_equal(bank.ambiguous_for_frame(10), [0])
    np.testing.assert_array_equal(bank.support_for_mask(100)[0], [])
    ids, mass, ratio, ambiguous = bank.support_for_mask(
        100, include_ambiguous=True
    )
    np.testing.assert_array_equal(ids, [0])
    np.testing.assert_allclose(mass, [0.75])
    np.testing.assert_allclose(ratio, [0.75 / 0.85])
    np.testing.assert_array_equal(ambiguous, [True])
    np.testing.assert_array_equal(bank.masks_for_gaussian(0), [])
    np.testing.assert_array_equal(
        bank.masks_for_gaussian(0, include_ambiguous=True), [100, 101]
    )
    frame_ids, visible_mass = bank.frames_for_gaussian(0)
    np.testing.assert_array_equal(frame_ids, [10, 20])
    np.testing.assert_allclose(visible_mass, [0.85, 0.85])


def test_semantic_posterior_or_abstention_contract() -> None:
    masks = np.asarray([[[1, 0]], [[0, 1]]], dtype=bool)
    mass = _mass(masks)
    posterior = np.zeros((2, 32), dtype=np.float32)
    posterior[0, 2] = 1
    frame = build_frame_evidence(
        frame_id=1,
        image_name="one.jpg",
        alpha_mass=mass,
        semantic_posteriors=posterior,
        semantic_abstained=[False, True],
    )
    assert not frame.metadata.semantic_abstained
    np.testing.assert_array_equal(frame.semantic_abstained, [False, True])
    with pytest.raises(ValueError, match="sum to one"):
        build_frame_evidence(
            frame_id=2,
            image_name="two.jpg",
            alpha_mass=mass,
            semantic_posteriors=np.full((2, 32), 0.01, dtype=np.float32),
            semantic_abstained=[False, False],
        )


def test_a_complete_100_gaussian_mask_is_not_reduced_to_a_core() -> None:
    alpha = np.ones((10, 10, 1), dtype=np.float64)
    ids = np.arange(100, dtype=np.int32).reshape(10, 10, 1)
    mass = accumulate_alpha_mass_from_contributors(
        alpha, np.ones_like(alpha), ids, np.ones((1, 10, 10), dtype=bool), 100
    )
    frame = build_frame_evidence(
        frame_id=0,
        image_name="zero.jpg",
        alpha_mass=mass,
        semantic_posteriors=np.eye(1, 32, dtype=np.float32),
    )
    assert len(frame.support.row(0)[0]) == 100


def test_sparse_worker_adapter_matches_dense_thresholding() -> None:
    dense = build_frame_evidence(
        frame_id=4,
        image_name="four.jpg",
        alpha_mass=_mass(np.asarray([[[1, 0]], [[1, 0]]], dtype=bool)),
        global_mask_id_start=40,
    )
    support = [dense.support.row(index) for index in range(2)]
    visible_ids, visible_mass = dense.visibility.row(0)
    sparse = build_sparse_frame_evidence(
        frame_id=4,
        image_name="four.jpg",
        point_count=3,
        visible_ids=visible_ids,
        visible_mass=visible_mass,
        mask_gaussian_ids=[row[0] for row in support],
        mask_inside_mass=[row[1] for row in support],
        mask_inside_ratio=[row[2] for row in support],
        ambiguous_ids=[np.asarray([0]), np.asarray([0])],
        global_mask_id_start=40,
        valid_pixel_count=2,
    )
    np.testing.assert_array_equal(
        sparse.support.gaussian_ids, dense.support.gaussian_ids
    )
    np.testing.assert_array_equal(
        sparse.support.ambiguous, dense.support.ambiguous
    )
    np.testing.assert_array_equal(
        sparse.ambiguous_gaussians, dense.ambiguous_gaussians
    )


def test_sparse_worker_adapter_rejects_forged_ambiguity() -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        build_sparse_frame_evidence(
            frame_id=0,
            image_name="zero.jpg",
            point_count=2,
            visible_ids=np.asarray([0]),
            visible_mass=np.asarray([1.0]),
            mask_gaussian_ids=[np.asarray([0]), np.asarray([0])],
            mask_inside_mass=[np.asarray([1.0]), np.asarray([1.0])],
            mask_inside_ratio=[np.asarray([1.0]), np.asarray([1.0])],
            ambiguous_ids=np.empty(0, dtype=np.int32),
        )


def test_save_load_round_trip_and_strict_completion(tmp_path: Path) -> None:
    bank = _bank()
    output = tmp_path / "bank" / bank.scene_id
    bank.save(output)
    assert evidence_bank_is_complete(
        output,
        expected_scene_id="scene0000_00",
        expected_point_count=3,
        expected_source=bank.source,
    )
    loaded = load_evidence_bank(output)
    assert loaded.scene_id == bank.scene_id
    assert loaded.frames == bank.frames
    assert loaded.masks == bank.masks
    np.testing.assert_array_equal(
        loaded.mask_support.gaussian_ids, bank.mask_support.gaussian_ids
    )
    np.testing.assert_allclose(
        loaded.semantic_posteriors, bank.semantic_posteriors
    )
    np.testing.assert_allclose(loaded.xyz_m, bank.xyz_m)
    with pytest.raises(FileExistsError):
        save_evidence_bank(bank, output)


def test_load_rejects_diagnostics_or_metadata_that_disagree(tmp_path: Path) -> None:
    output = tmp_path / "bank"
    save_evidence_bank(_bank(), output)
    diagnostics = json.loads((output / "diagnostics.json").read_text("utf-8"))
    diagnostics["mask_count"] += 1
    (output / "diagnostics.json").write_text(
        json.dumps(diagnostics), encoding="utf-8"
    )
    assert not evidence_bank_is_complete(output)
    with pytest.raises(ValueError, match="diagnostics"):
        load_evidence_bank(output)


def test_load_rejects_corrupt_csr(tmp_path: Path) -> None:
    output = tmp_path / "bank"
    save_evidence_bank(_bank(), output)
    with np.load(output / "evidence.npz", allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    arrays["mask_support_indptr"] = np.asarray([0, 999, 999], dtype=np.int64)
    np.savez_compressed(output / "evidence.npz", **arrays)
    assert not evidence_bank_is_complete(output)
    with pytest.raises(ValueError, match="terminate"):
        load_evidence_bank(output)


def test_arrays_are_immutable_after_validation() -> None:
    bank = _bank()
    assert not bank.mask_support.gaussian_ids.flags.writeable
    assert not bank.frame_visibility.visible_mass.flags.writeable
    assert not bank.semantic_posteriors.flags.writeable


def test_scene_worker_adapter_persists_sparse_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from category_priors.clean_baseline import worker

    base = tmp_path / "scene"
    base.mkdir()
    inputs = _materialize_identity_inputs(base)
    support = worker.RenderedMaskSupport(
        mask_index=0,
        gaussian_ids=np.asarray([0], dtype=np.int32),
        inside_mass=np.asarray([1.0], dtype=np.float32),
        inside_ratio=np.asarray([1.0], dtype=np.float32),
        ambiguous_ids=np.empty(0, dtype=np.int32),
        class_probabilities=np.eye(1, 32, dtype=np.float32)[0],
    )
    rendered = worker.RenderedFrameEvidence(
        frame_id=0,
        image_name="zero.jpg",
        visible_ids=np.asarray([0], dtype=np.int32),
        visible_mass=np.asarray([1.0], dtype=np.float32),
        masks=(support,),
        grounded_abstained=False,
        valid_pixel_count=1,
    )
    monkeypatch.setattr(worker, "resolve_clean_scene_inputs", lambda *_a, **_k: inputs)
    monkeypatch.setattr(
        worker,
        "render_scene_frames",
        lambda *_a, **_k: (np.zeros((2, 3), dtype=np.float32), (rendered,)),
    )
    output = tmp_path / "output"
    result = build_alpha_mask_evidence(
        scene_id="scene0000_00",
        request={
            "producer_commit": COMMIT,
            "scene": {"scene_id": "scene0000_00", "base_path": str(base)},
        },
        output_dir=output,
    )
    assert result["mask_count"] == 1
    assert evidence_bank_is_complete(output, expected_scene_id="scene0000_00")


def test_scene_worker_adapter_converts_raw_xyz_to_meters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from category_priors.clean_baseline import worker

    base = tmp_path / "scene"
    base.mkdir()
    inputs = _materialize_identity_inputs(base)
    empty = worker.RenderedFrameEvidence(
        0,
        "zero.jpg",
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float32),
        (),
        True,
        0,
    )
    monkeypatch.setattr(worker, "resolve_clean_scene_inputs", lambda *_a, **_k: inputs)
    monkeypatch.setattr(
        worker,
        "render_scene_frames",
        lambda *_a, **_k: (np.asarray([[2.0, 0.0, 0.0]], dtype=np.float32), (empty,)),
    )
    output = tmp_path / "metric"
    build_alpha_mask_evidence(
        scene_id="scene0000_00",
        request={
            "producer_commit": COMMIT,
            "scene": {
                "scene_id": "scene0000_00",
                "base_path": str(base),
                "scene_scale_m_per_unit": 0.25,
            }
        },
        output_dir=output,
    )
    bank = load_evidence_bank(output)
    np.testing.assert_allclose(bank.xyz_m, [[0.5, 0.0, 0.0]])
    assert bank.source["scene_scale_m_per_unit"] == 0.25


def test_request_identity_prevents_scene_only_cache_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from category_priors.clean_baseline import worker

    base = tmp_path / "scene"
    base.mkdir()
    inputs = _materialize_identity_inputs(base)
    monkeypatch.setattr(worker, "resolve_clean_scene_inputs", lambda *_a, **_k: inputs)
    request = {
        "producer_commit": COMMIT,
        "classes": list(CLASSES),
        "scene": {
            "scene_id": "scene0000_00",
            "base_path": str(base),
            "scene_scale_m_per_unit": 1.0,
        },
    }
    source = evidence_request_source(scene_id="scene0000_00", request=request)
    assert source["producer_commit"] == COMMIT
    assert source["class_names"] == list(CLASSES)
    identities = source["producer_inputs"]
    assert identities["schema"] == "saga-clean-evidence-input-content-v1"
    assert identities["gaussian_ply"]["file_count"] == 1
    assert identities["colmap_cameras"]["relative_paths"] == [
        "images.bin",
        "cameras.bin",
    ]
    assert identities["image_inputs"]["relative_paths"] == ["zero.jpg"]
    wrong_order = json.loads(json.dumps(request))
    wrong_order["classes"] = list(reversed(CLASSES))
    with pytest.raises(ValueError, match="registered 32-class"):
        evidence_request_source(
            scene_id="scene0000_00", request=wrong_order
        )
    changed = json.loads(json.dumps(request))
    changed["scene"]["scene_scale_m_per_unit"] = 0.5
    assert evidence_request_source(
        scene_id="scene0000_00", request=changed
    ) != source

    output = tmp_path / "bank"
    bank = replace(_bank(), source=source)
    save_evidence_bank(bank, output)
    assert evidence_bank_is_complete(
        output,
        expected_scene_id=bank.scene_id,
        expected_source=source,
    )

    # Replacing an input at the exact same path must invalidate the old bank.
    inputs.rgb_ply.write_bytes(b"ply-input-v2-with-different-content")
    changed_content_source = evidence_request_source(
        scene_id="scene0000_00", request=request
    )
    assert changed_content_source != source
    assert not evidence_bank_is_complete(
        output,
        expected_scene_id=bank.scene_id,
        expected_source=changed_content_source,
    )

    runtime_request = {
        "producer_commit": COMMIT,
        "classes": list(CLASSES),
        "runtime": dict(request["scene"]),
    }
    runtime_source = evidence_request_source(
        scene_id="scene0000_00", request=runtime_request
    )
    assert runtime_source["base_path"] == source["base_path"]
    assert runtime_source["class_names"] == source["class_names"]

    invalid_commit = json.loads(json.dumps(request))
    invalid_commit["producer_commit"] = "abc123"
    with pytest.raises(ValueError, match="40-character"):
        evidence_request_source(
            scene_id="scene0000_00", request=invalid_commit
        )
