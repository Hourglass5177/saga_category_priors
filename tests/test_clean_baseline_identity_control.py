from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from category_priors.clean_baseline import (
    AlphaMaskEvidenceBank,
    AlphaMassFrame,
    EvidenceThresholds,
    build_frame_evidence,
)
from category_priors.clean_baseline.evaluation import (
    GroundTruthObject,
    ground_truth_objects_from_arrays,
)
from category_priors.clean_baseline.identity_control import (
    IDENTITY_FEATURE_NAMES,
    IdentityAssetPaths,
    IdentityControlConfig,
    IdentitySceneInput,
    binary_auroc,
    edge_components,
    edge_feature_matrix,
    fit_balanced_l2_logistic,
    run_identity_edge_control,
)
from category_priors.clean_baseline.worker import DEFAULT_CLASSES
from category_priors.io import load_json, write_json


CLASSES = DEFAULT_CLASSES
EVALUATION_CLASSES = (
    "chair",
    "table",
    "plant",
    "tv",
    "painting",
    "sofa",
    "cabinet",
    "bed",
    "socket",
    "book",
    "switch",
    "door",
    "window",
    "lamp",
    "speaker",
    "fan",
    "refrigerator",
    "cup",
    "phone",
    "trash can",
)


def _bank() -> AlphaMaskEvidenceBank:
    visible = np.ones(4, dtype=np.float64)
    inside = np.asarray(
        [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    posterior = np.zeros((2, 32), dtype=np.float32)
    posterior[0, 0] = 1.0
    posterior[1, 1] = 1.0
    frames = []
    for frame_id in (0, 1):
        frames.append(
            build_frame_evidence(
                frame_id=frame_id,
                image_name=f"{frame_id}.jpg",
                alpha_mass=AlphaMassFrame(inside, visible, valid_pixel_count=4),
                global_mask_id_start=2 * frame_id,
                semantic_posteriors=posterior,
            )
        )
    return AlphaMaskEvidenceBank.from_frames(
        scene_id="scene",
        point_count=4,
        xyz_m=np.asarray(
            [[0.00, 0, 0], [0.01, 0, 0], [0.08, 0, 0], [0.09, 0, 0]],
            dtype=np.float32,
        ),
        class_names=CLASSES,
        frames=frames,
        thresholds=EvidenceThresholds(),
    )


def test_identity_features_include_registered_gaussian_and_covisibility_evidence() -> None:
    bank = _bank()
    edges = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    affinity = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float64)
    semantics = np.asarray(
        [[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float64
    )
    values = edge_feature_matrix(
        edge_index=edges,
        xyz_m=bank.xyz_m,
        affinity=affinity,
        soft_semantic=semantics,
        gaussian_scale_m=np.asarray(
            [[0.01, 0.02, 0.03]] * 4, dtype=np.float64
        ),
        opacity=np.asarray([0.8, 0.8, 0.7, 0.7]),
        bank=bank,
    )
    assert values.shape == (2, len(IDENTITY_FEATURE_NAMES))
    assert values[0, 0] == 1.0
    assert values[0, 1] == 1.0
    assert values[1, 0] == 0.0
    assert values[1, 1] == 0.0
    assert values[0, -1] == 1.0
    assert values[1, -1] == 0.0


def test_balanced_l2_logistic_and_auroc_are_deterministic() -> None:
    labels = np.asarray([0] * 20 + [1] * 5, dtype=np.int8)
    values = np.zeros((len(labels), 3), dtype=np.float64)
    values[:, 1] = np.where(labels == 1, 2.0, -2.0)
    first = fit_balanced_l2_logistic(values, labels)
    second = fit_balanced_l2_logistic(values, labels)
    np.testing.assert_allclose(first.coefficients, second.coefficients)
    probabilities = first.predict_proba(values)
    assert binary_auroc(labels, probabilities) == 1.0
    assert np.all(probabilities[labels == 1] > probabilities[labels == 0].max())


def test_edge_components_do_not_emit_untouched_singletons() -> None:
    edges = np.asarray([[0, 1], [1, 2], [3, 4]], dtype=np.int64)
    components = edge_components(6, edges, np.asarray([True, True, False]), min_component_points=3)
    assert len(components) == 1
    np.testing.assert_array_equal(components[0], [0, 1, 2])


def test_identity_scene_uses_scannet20_evaluation_order_not_32_class_codebook(
    tmp_path: Path,
) -> None:
    scene = IdentitySceneInput(
        scene_id="scene",
        bank_dir=tmp_path,
        gt_npz=tmp_path / "gt.npz",
        gaussian_to_gt_transform=tuple(tuple(row) for row in np.eye(4)),
        uniform_output_json=tmp_path / "output.json",
        evaluation_class_names=EVALUATION_CLASSES,
    )
    objects = ground_truth_objects_from_arrays(
        np.full(100, 3, dtype=np.int64),
        np.zeros(100, dtype=np.int64),
        class_names=scene.evaluation_class_names,
        min_region_size=100,
    )
    assert objects[0].class_id == "tv"
    assert CLASSES[3] == "flower"  # evidence taxonomy is deliberately separate


def test_three_scene_identity_control_is_offline_and_uses_held_out_gate(
    tmp_path: Path, monkeypatch,
) -> None:
    train_scenes = ("scene0645_00", "scene0025_01")
    validation_scene = "scene0046_00"
    scene_ids = (*train_scenes, validation_scene)
    assets = {}
    inputs = {}
    for scene_id in scene_ids:
        feature = tmp_path / f"{scene_id}-feature.ply"
        gaussian = tmp_path / f"{scene_id}-gaussian.ply"
        feature.write_bytes(b"registered")
        gaussian.write_bytes(b"registered")
        assets[scene_id] = IdentityAssetPaths(feature, gaussian)
        output = tmp_path / f"{scene_id}-uniform.json"
        write_json(
            output,
            {
                "scene_id": scene_id,
                "condition": "U-global",
                "point_labels": [-1] * 8,
                "instances": {},
            },
        )
        inputs[scene_id] = IdentitySceneInput(
            scene_id=scene_id,
            bank_dir=tmp_path / f"{scene_id}-bank",
            gt_npz=tmp_path / f"{scene_id}-gt.npz",
            gaussian_to_gt_transform=tuple(tuple(row) for row in np.eye(4)),
            uniform_output_json=output,
            evaluation_class_names=EVALUATION_CLASSES,
        )
    control = IdentityControlConfig(assets=assets)

    def prepared(scene_id: str):
        feature_count = len(IDENTITY_FEATURE_NAMES)
        if scene_id != validation_scene:
            labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
            features = np.zeros((8, feature_count), dtype=np.float64)
            features[:, 1] = np.where(labels == 1, 2.0, -2.0)
            return {
                "labelled_index": np.arange(8),
                "labels": labels,
                "features": features,
            }
        edges = np.asarray(
            [[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 7], [3, 4]],
            dtype=np.int64,
        )
        labels = np.asarray([1, 1, 1, 1, 1, 1, 0], dtype=np.int8)
        features = np.zeros((7, feature_count), dtype=np.float64)
        features[:, 0] = 0.5  # raw affinity has no ranking capacity
        features[:, 1] = np.where(labels == 1, 2.0, -2.0)
        objects = [
            GroundTruthObject(0, "class-0", np.arange(4), official_valid=True),
            GroundTruthObject(1, "class-1", np.arange(4, 8), official_valid=True),
        ]
        return {
            "scene": inputs[scene_id],
            "bank": SimpleNamespace(point_count=8),
            "edge_index": edges,
            "features": features,
            "affinity_cosine": features[:, 0],
            "labelled_index": np.arange(7),
            "labels": labels,
            "mapping": np.arange(8),
            "mapping_diagnostics": {"mapped_fraction": 1.0},
            "gt_objects": objects,
        }

    monkeypatch.setattr(
        "category_priors.clean_baseline.identity_control._prepare_scene",
        lambda _control, scene: prepared(scene.scene_id),
    )
    output = tmp_path / "identity-control.json"
    result = run_identity_edge_control(
        control=control, scenes=inputs, output_path=output
    )
    assert result["formal_method"] is False
    assert result["category_prior_tested"] is False
    assert result["validation"]["scene_id"] == validation_scene
    assert result["validation"]["raw_affinity_auroc"] == 0.5
    assert result["validation"]["learned_edge_auroc"] == 1.0
    assert result["validation"]["new_matched_gt_iou050_count"] == 2
    assert result["gate"]["passed"] is True
    assert load_json(output)["identity"] == control.identity()

    monkeypatch.setattr(
        "category_priors.clean_baseline.identity_control._prepare_scene",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("a complete identity-control result must be reused")
        ),
    )
    assert run_identity_edge_control(
        control=control, scenes=inputs, output_path=output
    ) == result
