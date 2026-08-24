from __future__ import annotations

import json
import importlib
import sys
from types import ModuleType

import numpy as np
import pytest

from category_priors.v9_lifting import (
    AttributionMass,
    FragmentConfig,
    V9_LIFTING_SCHEMA,
    build_lifting_identity,
    hybrid_fragments,
    lifting_bank_is_complete,
    mass_from_max_contributor,
)
from category_priors.v9_objectbank import (
    AssociationResult,
    ConsensusResult,
    Fragment,
    ObjectTrack,
    SparseCounts,
    TrackClassification,
    V9Config,
    associate_fragments,
    attach_local_halo,
    materialize_candidate_bank,
)
from category_priors.v9_lifting_runner import (
    _registered_sam_directory_is_complete,
    _run_logged,
)
from category_priors.v9_feature_training import validate_v8_sam_everything_source


def test_native_m1_rejects_empty_and_invalid_pixels() -> None:
    mass = mass_from_max_contributor(
        np.asarray([[0, -1], [2, 1]]),
        np.asarray([[0.4, 0.0], [0.0, 0.2]]),
        np.asarray([[[True, True], [True, False]]]),
        3,
    )
    assert mass.visible_mass.tolist() == [1.0, 1.0, 0.0]
    assert mass.inside_mass[0].tolist() == [1.0, 0.0, 0.0]


def test_native_subprocess_failure_is_not_silently_accepted(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="exited 3"):
        _run_logged(
            (sys.executable, "-c", "raise SystemExit(3)"),
            cwd=tmp_path,
            log_path=tmp_path / "worker.log",
        )


def test_hybrid_uses_m1_core_am_full_and_records_same_view_conflict() -> None:
    m1 = AttributionMass(
        "M1",
        np.asarray([[2, 2, 2, 0], [0, 0, 2, 2]], dtype=float),
        np.asarray([2, 2, 2, 2], dtype=float),
        8,
    )
    am = AttributionMass(
        "AM",
        np.asarray([[2, 2, 2, 1], [1, 1, 2, 2]], dtype=float),
        np.asarray([2, 2, 2, 2], dtype=float),
        8,
    )
    fragments = hybrid_fragments(
        m1,
        am,
        0,
        0,
        config=FragmentConfig(fragment_min_core=1, fragment_min_full=1),
    )
    assert fragments[0].full_ids.tolist() == [0, 1, 2, 3]
    assert fragments[0].core_ids.tolist() == [0, 1, 2]
    assert fragments[0].conflict_ratio == pytest.approx(1 / 3)


def _fragment(fragment_id: int, frame: int, ids: list[int], conflict: float | None) -> Fragment:
    values = np.asarray(ids, dtype=np.int32)
    return Fragment(
        fragment_id,
        frame,
        0,
        values,
        values,
        np.ones(len(values)),
        np.ones(len(values)),
        conflict,
    )


def test_a2_rejects_missing_conflict_instead_of_assuming_zero() -> None:
    fragments = (
        _fragment(0, 0, [0, 1, 2], 0.0),
        _fragment(1, 1, [0, 1, 2], 0.0),
        _fragment(2, 2, [3, 4, 5], None),
    )
    xyz = np.column_stack((np.arange(6) * 0.001, np.zeros((6, 2))))
    with pytest.raises(ValueError, match="conflict evidence"):
        associate_fragments(fragments, "A2", xyz_m=xyz, affinity=np.ones((6, 2)))


def test_local_attach_considers_all_anchors_inside_radius() -> None:
    wrong = np.column_stack((np.linspace(0.001, 0.012, 12), np.zeros((12, 2))))
    correct = np.asarray([[0.040, 0, 0], [0.041, 0, 0], [0.042, 0, 0]])
    xyz = np.vstack((wrong, correct, [[0, 0, 0]]))
    affinity = np.vstack((
        np.tile([0.0, 1.0], (12, 1)),
        np.tile([1.0, 0.0], (3, 1)),
        [[1.0, 0.0]],
    ))
    labels = np.asarray([0] * 12 + [1] * 3 + [-1], dtype=np.int32)
    empty = SparseCounts(np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), len(labels))
    consensus = ConsensusResult(
        labels,
        np.ones(len(labels), dtype=np.int32),
        np.zeros(len(labels), dtype=np.float32),
        (0, 1),
        {0: empty, 1: empty},
        {0: empty, 1: empty},
        {0: np.arange(12, dtype=np.int32), 1: np.arange(12, 15, dtype=np.int32)},
    )
    output = attach_local_halo(xyz, affinity, consensus, V9Config())
    assert output[-1] == 1


def test_non_saga_classification_does_not_remove_geometry_candidate() -> None:
    association = AssociationResult(
        "A1", (ObjectTrack(7, (0, 1), (0, 1), (0.8,), "A1"),), ()
    )
    point_count = 5
    labels = np.asarray([7, 7, 7, -1, -1], dtype=np.int32)
    counts = SparseCounts(np.asarray([0, 1, 2]), np.asarray([2, 2, 2]), point_count)
    consensus = ConsensusResult(
        labels,
        np.ones(point_count, dtype=np.int32) * 2,
        np.ones(point_count, dtype=np.float32),
        (7,),
        {7: counts},
        {7: SparseCounts(np.asarray([0, 1, 2]), np.zeros(3, dtype=np.int32), point_count)},
        {7: np.asarray([0, 1, 2], dtype=np.int32)},
    )
    bank = materialize_candidate_bank(
        np.column_stack((np.arange(point_count) * 0.01, np.zeros((point_count, 2)))),
        np.tile([1.0, 0.0], (point_count, 1)),
        association,
        consensus,
        labels,
        {7: TrackClassification(7, 10, "wall", 0.9, 0.5, 2, "mv-label", False)},
        V9Config(),
    )
    assert len(bank.candidates) == 1
    assert bank.candidates[0]["track_id"] == 7
    assert bank.candidates[0]["classification_eligible"] is False


def _write_native_bank(tmp_path, *, core_id: int = 0, semantic_class: int = 0) -> None:
    feature = tmp_path / "feature.ply"
    feature.write_bytes(b"ply\n")
    scale_gate = tmp_path / "scale_gate_10k.pt"
    scale_gate.write_bytes(b"gate")
    (tmp_path / "train_progress.txt").write_text("100", "utf-8")
    label_features = tmp_path / "labels.pt"
    label_features.write_bytes(b"labels")
    feature_record = tmp_path / "train_10k.json"
    feature_identity = {
        "schema": "saga-v9-dual-source-feature-v1",
        "scene_id": "scene",
        "iterations": 10_000,
        "seed": 42,
        "outputs": {
            "feature_ply": str(feature.resolve()),
            "scale_gate": str(scale_gate.resolve()),
        },
    }
    feature_record.write_text(
        json.dumps(
            {
                "kind": "v9_feature_training_run",
                "status": "complete",
                "git_commit": "commit",
                "identity": feature_identity,
            }
        ),
        "utf-8",
    )
    sam = tmp_path / "sam"
    sam.mkdir(exist_ok=True)
    np.savez_compressed(
        sam / "frame.npz",
        packed=np.packbits(
            np.asarray(
                [
                    [True, False, False, True],
                    [False, True, True, False],
                ],
                dtype=np.bool_,
            ),
            axis=1,
        ),
        count=np.asarray(2, dtype=np.int32),
        height=np.asarray(2, dtype=np.int32),
        width=np.asarray(2, dtype=np.int32),
    )
    (sam / "summary.json").write_text(
        json.dumps(
            {
                "schema": "saga-v9-segment-everything-v1",
                "image_root": str((tmp_path / "images").resolve()),
                "output_root": str(sam.resolve()),
                "sam_arch": "vit_h",
                "config": {
                    "points_per_side": 32,
                    "pred_iou_thresh": 0.88,
                    "stability_score_thresh": 0.95,
                    "box_nms_thresh": 0.70,
                    "crop_n_layers": 0,
                    "crop_n_points_downscale_factor": 1,
                    "min_mask_region_area": 100,
                },
                "image_count": 1,
                "mask_count": 2,
                "images": [
                    {
                        "image": "frame.jpg",
                        "height": 2,
                        "width": 2,
                        "mask_count": 2,
                    }
                ],
            }
        ),
        "utf-8",
    )
    materialization = tmp_path / "affinity-inputs/sam_everything_materialization.json"
    materialization.parent.mkdir(parents=True, exist_ok=True)
    materialization.write_text(
        json.dumps({
            "kind": "v9_sam_everything_materialization",
            "source": validate_v8_sam_everything_source(sam),
        }),
        "utf-8",
    )
    identity = build_lifting_identity(
        scene_id="scene",
        git_commit="commit",
        feature_ply=feature,
        feature_record=feature_record,
        label_features=label_features,
        segment_everything_root=sam,
        classes=("chair",),
    )
    metadata = {
        "schema": V9_LIFTING_SCHEMA,
        "scene_id": "scene",
        "git_commit": "commit",
        "identity": identity,
        "lifting_source": "M1-core+AM-full",
        "mask_source": "SAM-everything",
        "feature_source": "v9-10k-objectbank",
        "config": {
            key: value for key, value in vars(FragmentConfig()).items()
        },
        "point_count": 3,
        "fragment_count": 1,
        "frame_count": 1,
        "semantic_fragment_count": 1,
        "classes": ["chair"],
    }
    (tmp_path / "lifting_bank.json").write_text(json.dumps(metadata), "utf-8")
    np.savez_compressed(
        tmp_path / "lifting_bank.npz",
        xyz_m=np.zeros((3, 3)), affinity=np.ones((3, 2)), semantic=np.ones((3, 2)),
        label_features=np.asarray([[1.0, 0.0]]),
        fragment_full_indptr=np.asarray([0, 1]), fragment_full_ids=np.asarray([0]),
        fragment_full_mass=np.asarray([1.0]),
        fragment_core_indptr=np.asarray([0, 1]), fragment_core_ids=np.asarray([core_id]),
        fragment_core_mass=np.asarray([1.0]), fragment_id=np.asarray([0]),
        fragment_frame=np.asarray([0]), fragment_mask_index=np.asarray([0]),
        fragment_conflict_ratio=np.asarray([0.0]),
        frame_visible_indptr=np.asarray([0, 1]), frame_visible_ids=np.asarray([0]),
        frame_visible_mass=np.asarray([1.0]), frame_geometry_abstained=np.asarray([False]),
        frame_grounded_missing=np.asarray([False]),
        semantic_fragment_full_indptr=np.asarray([0, 1]), semantic_fragment_full_ids=np.asarray([0]),
        semantic_fragment_full_mass=np.asarray([1.0]),
        semantic_fragment_frame=np.asarray([0]), semantic_fragment_class=np.asarray([semantic_class]),
    )


def test_native_bank_validation_accepts_complete_identified_bank(tmp_path) -> None:
    _write_native_bank(tmp_path)
    assert lifting_bank_is_complete(
        tmp_path, expected_scene_id="scene", expected_git_commit="commit"
    )


def test_lifting_strictly_reuses_registered_v8_masks_and_pins_feature_producer(
    tmp_path,
) -> None:
    _write_native_bank(tmp_path)
    sam = tmp_path / "sam"
    summary = json.loads((sam / "summary.json").read_text("utf-8"))
    summary["schema"] = "saga-v8-segment-everything-v1"
    (sam / "summary.json").write_text(json.dumps(summary), "utf-8")
    materialization_path = tmp_path / "affinity-inputs/sam_everything_materialization.json"
    materialization = json.loads(materialization_path.read_text("utf-8"))
    materialization["source"] = validate_v8_sam_everything_source(sam)
    materialization_path.write_text(json.dumps(materialization), "utf-8")
    record_path = tmp_path / "train_10k.json"
    record = json.loads(record_path.read_text("utf-8"))
    record["git_commit"] = "feature-producer"
    record_path.write_text(json.dumps(record), "utf-8")

    identity = build_lifting_identity(
        scene_id="scene",
        git_commit="lifting-consumer",
        feature_ply=tmp_path / "feature.ply",
        feature_record=record_path,
        label_features=tmp_path / "labels.pt",
        segment_everything_root=sam,
        classes=("chair",),
    )

    assert identity["git_commit"] == "lifting-consumer"
    assert identity["feature_producer_git_commit"] == "feature-producer"
    assert identity["segment_everything"]["registered_summary_schema"] == (
        "saga-v8-segment-everything-v1"
    )
    assert json.loads(record_path.read_text("utf-8"))["git_commit"] == (
        "feature-producer"
    )


def test_registered_sam_source_binds_scene_root_and_frozen_config(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_native_bank(tmp_path)
    sam = tmp_path / "sam"
    monkeypatch.setattr(
        "category_priors.v9_lifting_runner.sam_directory_is_complete",
        lambda directory, image_root: True,
    )
    assert _registered_sam_directory_is_complete(sam, tmp_path / "images")
    assert not _registered_sam_directory_is_complete(sam, tmp_path / "other-images")

    summary = json.loads((sam / "summary.json").read_text("utf-8"))
    summary["config"]["pred_iou_thresh"] = 0.5
    (sam / "summary.json").write_text(json.dumps(summary), "utf-8")
    assert not _registered_sam_directory_is_complete(sam, tmp_path / "images")


def test_native_bank_validation_rejects_core_outside_full(tmp_path) -> None:
    _write_native_bank(tmp_path, core_id=1)
    assert not lifting_bank_is_complete(tmp_path)


def test_native_bank_validation_rejects_invalid_class_and_nonfinite_mass(tmp_path) -> None:
    _write_native_bank(tmp_path, semantic_class=1)
    assert not lifting_bank_is_complete(tmp_path)
    _write_native_bank(tmp_path)
    with np.load(tmp_path / "lifting_bank.npz", allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    arrays["frame_visible_mass"] = np.asarray([np.nan])
    np.savez_compressed(tmp_path / "lifting_bank.npz", **arrays)
    assert not lifting_bank_is_complete(tmp_path)


def test_worker_does_not_skip_bank_without_current_full_identity(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The local pure-function test environment intentionally does not install
    # CUDA PyTorch.  Worker import itself must remain testable because torch is
    # only dereferenced after the resume decision.
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    v9_lifting_worker = importlib.import_module(
        "category_priors.v9_lifting_worker"
    )
    identity = {
        "schema": "saga-v9-native-lifting-identity-v1",
        "scene_id": "scene",
        "git_commit": "current",
        "feature_record_identity": {"scene_id": "scene"},
    }
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        v9_lifting_worker,
        "build_lifting_identity",
        lambda **kwargs: dict(identity),
    )

    def fake_complete(directory, **kwargs):
        observed["complete_kwargs"] = kwargs
        return False

    monkeypatch.setattr(v9_lifting_worker, "lifting_bank_is_complete", fake_complete)

    def fake_run(**kwargs):
        observed["run_kwargs"] = kwargs
        return {"scene_id": "scene", "fragment_count": 1, "runtime_seconds": 0.1}

    monkeypatch.setattr(v9_lifting_worker, "run_v9_lifting_bank", fake_run)
    assert v9_lifting_worker.main(
        [
            "--scene-id", "scene",
            "--base-path", str(tmp_path / "scene"),
            "--output-dir", str(tmp_path / "bank"),
            "--scene-scale-m-per-unit", "1.0",
            "--segment-everything-root", str(tmp_path / "sam"),
            "--feature-ply", str(tmp_path / "feature.ply"),
            "--feature-record", str(tmp_path / "train_10k.json"),
            "--label-features", str(tmp_path / "labels.pt"),
            "--git-commit", "current",
        ]
    ) == 0
    complete_kwargs = observed["complete_kwargs"]
    assert complete_kwargs["expected_git_commit"] == "current"
    assert complete_kwargs["expected_identity"] == identity
    assert complete_kwargs["expected_feature_record_identity"] == {
        "scene_id": "scene"
    }
    assert observed["run_kwargs"]["git_commit"] == "current"
