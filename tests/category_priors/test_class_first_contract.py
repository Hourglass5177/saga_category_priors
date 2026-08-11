from __future__ import annotations

import math

import numpy as np

from category_priors.class_first import (
    ClassFirstConfig,
    _cluster_sample,
    build_class_masks,
    run_class_first,
)
from category_priors.class_first_runner import build_class_first_command


def test_semantic_threshold_uses_normalized_cosine_similarity() -> None:
    point_semantic = np.asarray(
        [
            [100.0, 100.0],  # cosine 1/sqrt(2), despite a very large dot product
            [25.0, 0.0],
            [0.0, 7.0],
        ],
        dtype=np.float32,
    )
    label_features = np.asarray(
        [
            [9.0, 0.0],
            [0.0, 13.0],
        ],
        dtype=np.float32,
    )

    top_class, top_similarity, masks = build_class_masks(
        point_semantic,
        label_features,
        ("chair", "cup"),
        threshold=0.8,
    )

    assert top_class.tolist() == [0, 0, 1]
    np.testing.assert_allclose(
        top_similarity[0], 1.0 / math.sqrt(2.0), atol=1e-6
    )
    assert masks["chair"].tolist() == [False, True, False]
    assert masks["cup"].tolist() == [False, False, True]


def test_unknown_category_uses_uniform_parameters(
    fitted_priors: dict[str, object],
) -> None:
    captured: dict[str, object] = {}

    class FakeClusterer:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def fit_predict(self, distance: np.ndarray) -> np.ndarray:
            return np.zeros(len(distance), dtype=np.int64)

    config = ClassFirstConfig(
        semantic_threshold=0.5,
        sample_fraction=1.0,
        min_samples=2,
        use_sor=False,
    )
    count = 40
    result = run_class_first(
        point_features=np.column_stack(
            (np.ones(count), np.linspace(0.0, 0.2, count))
        ),
        point_semantic_features=np.tile([3.0, 0.0], (count, 1)),
        point_xyz=np.column_stack(
            (np.linspace(0.0, 1.0, count), np.zeros(count), np.zeros(count))
        ),
        label_features=np.asarray([[11.0, 0.0]], dtype=np.float32),
        classes=("not-in-priors",),
        priors=fitted_priors,
        config=config,
        mode="combined",
        scene_scale_m_per_unit=1.0,
        clusterer_factory=FakeClusterer,
    )

    parameters = result.diagnostics["classes"]["not-in-priors"]["parameters"]
    assert parameters["supported"] is False
    assert parameters["min_cluster_size"] == config.min_cluster_size
    assert parameters["min_samples"] == config.min_samples
    assert parameters["knn_k"] == config.knn_k
    assert parameters["coordinate_mode"] == "robust_scale"
    assert parameters["rescue_enabled"] is False
    assert parameters["d_c_m"] is None
    assert parameters["A_c_m2"] is None
    assert parameters["b_c"] is None
    assert captured["min_samples"] == config.min_samples


def test_hdbscan_min_samples_is_always_explicit() -> None:
    captured: dict[str, object] = {}

    class FakeClusterer:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def fit_predict(self, distance: np.ndarray) -> np.ndarray:
            return np.full(len(distance), -1, dtype=np.int64)

    config = ClassFirstConfig(min_cluster_size=7, min_samples=5)
    labels = _cluster_sample(
        np.zeros((8, 8), dtype=np.float64),
        min_cluster_size=7,
        config=config,
        clusterer_factory=FakeClusterer,
    )

    assert labels.tolist() == [-1] * 8
    assert captured["min_cluster_size"] == 7
    assert captured["min_samples"] == 5
    assert captured["metric"] == "precomputed"


def test_class_first_output_paths_are_isolated_by_seed(tmp_path) -> None:
    scene = {
        "base_path": str(tmp_path / "scene"),
        "scene_scale_m_per_unit": 1.0,
        "python_bin": str(tmp_path / "python"),
    }
    common = (
        tmp_path / "run_pipeline.sh",
        scene,
        tmp_path / "runs",
        "D-combined",
        "scene0000_00",
    )
    _, seed_42 = build_class_first_command(
        *common,
        42,
        tmp_path / "priors.json",
        tmp_path / "config.json",
    )
    _, seed_3407 = build_class_first_command(
        *common,
        3407,
        tmp_path / "priors.json",
        tmp_path / "config.json",
    )

    assert seed_42["run_dir"].name == "seed-42"
    assert seed_3407["run_dir"].name == "seed-3407"
    for name in ("output_json", "diagnostics_json", "progress", "log"):
        assert seed_42[name] != seed_3407[name]
