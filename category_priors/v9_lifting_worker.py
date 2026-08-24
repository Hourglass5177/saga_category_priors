from __future__ import annotations

"""Native V9 GPU worker for hybrid M1-core / alpha-mass ObjectBank lifting."""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .v9_lifting import (
    AttributionMass,
    ChannelBatch,
    DEFAULT_CLASSES,
    Fragment,
    FragmentConfig,
    V9_LIFTING_SCHEMA,
    build_lifting_identity,
    build_objectives,
    fragments_from_mass,
    hybrid_fragments,
    iter_mask_batches,
    lifting_bank_is_complete,
    mass_from_gradients,
    mass_from_max_contributor,
    pack_ragged,
)


@dataclass(frozen=True)
class MaskPayload:
    masks: np.ndarray | None
    labels: np.ndarray
    abstained: bool

    @property
    def count(self) -> int:
        return 0 if self.masks is None else len(self.masks)


@dataclass(frozen=True)
class FrameRecord:
    frame_id: int
    image_name: str
    fragments: tuple[Fragment, ...]
    labels: np.ndarray
    visible_ids: np.ndarray
    visible_mass: np.ndarray
    abstained: bool
    mask_count: int


def _paths(base: Path) -> dict[str, Path]:
    candidates = (
        base / "output_models/point_cloud/iteration_30000/scene_point_cloud.ply",
        base / "output_models/point_cloud/iteration_30000/point_cloud.ply",
    )
    return {
        "rgb": next((path for path in candidates if path.is_file()), candidates[0]),
        "feature": base / "saga/contrastive_feature_point_cloud.ply",
        "sparse": base / "fastRecon/dense/sparse/0",
        "images": base / "fastRecon/dense/sparse/0/images",
        "masks": base / "saga/masks",
        "labels": base / "saga/labels",
    }


def _load_cameras(paths: Mapping[str, Path]) -> list[Any]:
    from scene.colmap_loader import (
        read_extrinsics_binary,
        read_extrinsics_text,
        read_intrinsics_binary,
        read_intrinsics_text,
    )
    from scene.dataset_readers import readColmapCameras
    from utils.camera_utils import cameraList_from_camInfos

    try:
        extrinsics = read_extrinsics_binary(str(paths["sparse"] / "images.bin"))
        intrinsics = read_intrinsics_binary(str(paths["sparse"] / "cameras.bin"))
    except (FileNotFoundError, OSError):
        extrinsics = read_extrinsics_text(str(paths["sparse"] / "images.txt"))
        intrinsics = read_intrinsics_text(str(paths["sparse"] / "cameras.txt"))
    infos = readColmapCameras(
        extrinsics,
        intrinsics,
        str(paths["images"]),
        masks_folder=str(paths["masks"]),
        labels_folder=str(paths["labels"]),
    )
    return cameraList_from_camInfos(
        infos, 1, SimpleNamespace(resolution=1, data_device="cuda")
    )


def _resize_masks(masks: Any, height: int, width: int) -> np.ndarray:
    tensor = torch.as_tensor(masks).detach().cpu()
    if tensor.ndim != 3:
        raise ValueError("mask tensor must be MxHxW")
    if tuple(tensor.shape[-2:]) != (height, width):
        tensor = torch.nn.functional.interpolate(
            tensor.float().unsqueeze(1), size=(height, width), mode="nearest"
        ).squeeze(1)
    return tensor.bool().numpy()


def grounded_payload(camera: Any) -> MaskPayload:
    masks, labels = camera.original_masks, camera.labels
    if masks is None and labels is None:
        return MaskPayload(None, np.empty(0, dtype=np.int16), True)
    if masks is None or labels is None:
        raise ValueError("Grounded masks and labels must both exist or both be absent")
    mask_array = _resize_masks(masks, camera.image_height, camera.image_width)
    label_array = np.asarray(torch.as_tensor(labels).detach().cpu()).reshape(-1)
    if len(mask_array) != len(label_array):
        raise ValueError("Grounded masks and labels have different lengths")
    return MaskPayload(mask_array, label_array.astype(np.int16), False)


def segment_everything_payload(root: Path, camera: Any) -> MaskPayload:
    path = root / f"{camera.image_name}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen SAM-everything masks: {path}")
    with np.load(path, allow_pickle=False) as payload:
        packed = np.asarray(payload["packed"], dtype=np.uint8)
        count = int(np.asarray(payload["count"]).item())
        height = int(np.asarray(payload["height"]).item())
        width = int(np.asarray(payload["width"]).item())
    if (height, width) != (camera.image_height, camera.image_width):
        raise ValueError("SAM-everything and camera shapes differ")
    expected = (count, (height * width + 7) // 8)
    if count < 0 or packed.shape != expected:
        raise ValueError("invalid frozen SAM-everything payload")
    masks = np.unpackbits(packed, axis=1, count=height * width).reshape(
        count, height, width
    ).astype(bool, copy=False)
    return MaskPayload(masks, np.full(count, -1, dtype=np.int16), False)


def render_alpha_mass(
    camera: Any,
    gaussians: Any,
    pipeline: Any,
    background: torch.Tensor,
    masks: np.ndarray | None,
    point_count: int,
    *,
    render_mask_fn: Callable[..., Mapping[str, Any]] | None = None,
) -> AttributionMass:
    if render_mask_fn is None:
        from gaussian_renderer import render_mask as render_mask_fn

    height, width = int(camera.image_height), int(camera.image_width)
    mask_array = (
        np.zeros((0, height, width), dtype=bool)
        if masks is None
        else np.asarray(masks, dtype=bool)
    )
    if mask_array.ndim != 3 or mask_array.shape[1:] != (height, width):
        raise ValueError("AM masks do not match camera")
    batches = list(iter_mask_batches(mask_array))
    if not batches:
        batches = [ChannelBatch((), np.zeros((3, height, width), dtype=np.float32))]
    visible: np.ndarray | None = None
    inside: list[tuple[ChannelBatch, np.ndarray]] = []
    valid_pixels = 0
    xyz = gaussians.get_xyz
    for batch_number, batch in enumerate(batches):
        probe = torch.ones(
            (point_count, 3), dtype=xyz.dtype, device=xyz.device, requires_grad=True
        )
        rendered = render_mask_fn(
            camera, gaussians, pipeline, background, precomputed_mask=probe
        )
        image = rendered.get("mask")
        if not isinstance(image, torch.Tensor) or image.shape != (3, height, width) or not image.requires_grad:
            raise RuntimeError("differentiable mask renderer returned invalid output")
        objectives = build_objectives(batch, image.detach()[0].float().cpu().numpy())
        if batch_number == 0:
            coefficient = torch.as_tensor(
                objectives.visible_coefficient, dtype=image.dtype, device=image.device
            )
            gradient = torch.autograd.grad(
                torch.sum(image[0] * coefficient), probe, retain_graph=bool(batch.mask_indices)
            )[0]
            visible = gradient.detach().cpu().numpy()
            valid_pixels = int(np.count_nonzero(objectives.valid_pixels))
        if batch.mask_indices:
            coefficient = torch.as_tensor(
                objectives.inside_coefficients, dtype=image.dtype, device=image.device
            )
            gradient = torch.autograd.grad(torch.sum(image * coefficient), probe)[0]
            inside.append((batch, gradient.detach().cpu().numpy()))
    if visible is None:
        raise RuntimeError("AM visibility gradient was not produced")
    return mass_from_gradients(
        visible,
        inside,
        len(mask_array),
        valid_pixels,
        abstained=masks is None,
    )


def _record(
    frame_id: int,
    image_name: str,
    mass: AttributionMass,
    fragments: tuple[Fragment, ...],
    labels: np.ndarray,
) -> FrameRecord:
    visible_ids = np.flatnonzero(mass.visible_mass > 0).astype(np.int32)
    return FrameRecord(
        frame_id,
        image_name,
        fragments,
        np.asarray(labels, dtype=np.int16),
        visible_ids,
        mass.visible_mass[visible_ids].astype(np.float32),
        mass.abstained,
        mass.mask_count,
    )


def _serialize(
    output: Path,
    *,
    xyz_m: np.ndarray,
    affinity: np.ndarray,
    semantic: np.ndarray,
    label_features: np.ndarray,
    geometry: Sequence[FrameRecord],
    semantics: Sequence[FrameRecord],
) -> None:
    fragments = [fragment for frame in geometry for fragment in frame.fragments]
    semantic_fragments = [fragment for frame in semantics for fragment in frame.fragments]
    full_i, full_v = pack_ragged([row.full_ids for row in fragments], np.int32)
    _, full_m = pack_ragged([row.full_mass for row in fragments], np.float32)
    core_i, core_v = pack_ragged([row.core_ids for row in fragments], np.int32)
    _, core_m = pack_ragged([row.core_mass for row in fragments], np.float32)
    visible_i, visible_v = pack_ragged([row.visible_ids for row in geometry], np.int32)
    _, visible_m = pack_ragged([row.visible_mass for row in geometry], np.float32)
    semantic_i, semantic_v = pack_ragged(
        [row.full_ids for row in semantic_fragments], np.int32
    )
    _, semantic_m = pack_ragged(
        [row.full_mass for row in semantic_fragments], np.float32
    )
    semantic_classes: list[int] = []
    for frame in semantics:
        semantic_classes.extend(int(frame.labels[row.mask_index]) for row in frame.fragments)
    np.savez_compressed(
        output / "lifting_bank.npz",
        xyz_m=np.asarray(xyz_m, dtype=np.float32),
        affinity=np.asarray(affinity, dtype=np.float32),
        semantic=np.asarray(semantic, dtype=np.float32),
        label_features=np.asarray(label_features, dtype=np.float32),
        fragment_full_indptr=full_i,
        fragment_full_ids=full_v,
        fragment_full_mass=full_m,
        fragment_core_indptr=core_i,
        fragment_core_ids=core_v,
        fragment_core_mass=core_m,
        fragment_id=np.asarray([row.fragment_id for row in fragments], dtype=np.int64),
        fragment_frame=np.asarray([row.frame_id for row in fragments], dtype=np.int32),
        fragment_mask_index=np.asarray([row.mask_index for row in fragments], dtype=np.int32),
        fragment_conflict_ratio=np.asarray([row.conflict_ratio for row in fragments], dtype=np.float32),
        frame_visible_indptr=visible_i,
        frame_visible_ids=visible_v,
        frame_visible_mass=visible_m,
        frame_geometry_abstained=np.asarray([row.abstained for row in geometry], dtype=bool),
        frame_grounded_missing=np.asarray([row.abstained for row in semantics], dtype=bool),
        semantic_fragment_full_indptr=semantic_i,
        semantic_fragment_full_ids=semantic_v,
        semantic_fragment_full_mass=semantic_m,
        semantic_fragment_frame=np.asarray([row.frame_id for row in semantic_fragments], dtype=np.int32),
        semantic_fragment_class=np.asarray(semantic_classes, dtype=np.int16),
    )


def run_v9_lifting_bank(
    *,
    scene_id: str,
    base_path: Path,
    output_dir: Path,
    scene_scale_m_per_unit: float,
    segment_everything_root: Path,
    feature_ply_path: Path,
    feature_record_path: Path,
    label_features_path: Path,
    git_commit: str,
    classes: Sequence[str] = DEFAULT_CLASSES,
) -> dict[str, Any]:
    """Freeze one geometry-first bank from the registered 10k feature PLY."""

    if scene_scale_m_per_unit <= 0:
        raise ValueError("scene scale must be positive")
    from gaussian_renderer import render_with_max_contributor
    from scene import FeatureGaussianModel, GaussianModel

    paths = _paths(base_path)
    # The historical 2k feature PLY is deliberately not an input.  Requiring
    # it here would make a healthy isolated 10k scene fail for an irrelevant
    # legacy artifact.
    required = (
        paths["rgb"],
        paths["sparse"],
        paths["images"],
        paths["masks"],
        paths["labels"],
        segment_everything_root,
        feature_ply_path,
        feature_record_path,
        label_features_path,
    )
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"missing V9 lifting inputs: {missing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rgb = GaussianModel(0)
    rgb.load_ply(str(paths["rgb"]))
    feature = FeatureGaussianModel(32, 32)
    feature.load_ply(str(feature_ply_path))
    rgb_xyz = rgb.get_xyz.detach().cpu().numpy()
    feature_xyz = feature.get_xyz.detach().cpu().numpy()
    if rgb_xyz.shape != feature_xyz.shape or not np.allclose(rgb_xyz, feature_xyz, atol=1e-6, rtol=0):
        raise ValueError("RGB and 10k feature Gaussian order differs")
    point_count = len(rgb_xyz)
    affinity = feature.get_point_features.detach().cpu().numpy()
    semantic = feature.get_point_semantic_features.detach().cpu().numpy()
    affinity /= np.maximum(np.linalg.norm(affinity, axis=1, keepdims=True), 1e-12)
    semantic /= np.maximum(np.linalg.norm(semantic, axis=1, keepdims=True), 1e-12)
    loaded_labels = torch.load(label_features_path, map_location="cpu")
    if not isinstance(loaded_labels, torch.Tensor) or loaded_labels.ndim != 2:
        raise ValueError("label feature codebook must be a matrix")
    label_features = loaded_labels.detach().cpu().numpy().astype(np.float32)
    if label_features.shape != (len(classes), semantic.shape[1]):
        raise ValueError("label codebook does not match 32-class semantic feature")
    label_features /= np.maximum(np.linalg.norm(label_features, axis=1, keepdims=True), 1e-12)

    pipeline = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    geometry_records: list[FrameRecord] = []
    semantic_records: list[FrameRecord] = []
    geometry_offset = 0
    semantic_offset = 0
    for frame_id, camera in enumerate(sorted(_load_cameras(paths), key=lambda item: item.image_name)):
        geometry_payload = segment_everything_payload(segment_everything_root, camera)
        semantic_payload = grounded_payload(camera)
        geometry_masks = geometry_payload.masks
        assert geometry_masks is not None
        semantic_masks = (
            np.zeros((0, camera.image_height, camera.image_width), dtype=bool)
            if semantic_payload.masks is None
            else semantic_payload.masks
        )
        combined_masks = np.concatenate((geometry_masks, semantic_masks), axis=0)
        alpha_all = render_alpha_mass(
            camera, rgb, pipeline, background, combined_masks, point_count
        )
        boundary = len(geometry_masks)
        alpha_geometry = AttributionMass(
            "AM", alpha_all.inside_mass[:boundary], alpha_all.visible_mass,
            alpha_all.valid_pixel_count, False,
        )
        alpha_semantic = AttributionMass(
            "AM", alpha_all.inside_mass[boundary:], alpha_all.visible_mass,
            alpha_all.valid_pixel_count, semantic_payload.abstained,
        )
        with torch.no_grad():
            rendered = render_with_max_contributor(camera, rgb, pipeline, background)
        max_id = rendered["max_contributor"].detach().cpu().numpy()
        max_weight = rendered["max_contribute"].detach().cpu().numpy()
        maximum = mass_from_max_contributor(max_id, max_weight, geometry_masks, point_count)
        geometry_fragments = hybrid_fragments(
            maximum, alpha_geometry, frame_id, geometry_offset
        )
        # Semantic masks are late evidence and must not be censored by object
        # fragment size gates.
        semantic_fragments = fragments_from_mass(
            alpha_semantic,
            frame_id,
            semantic_offset,
            config=FragmentConfig(fragment_min_core=0, fragment_min_full=1),
        )
        geometry_records.append(_record(
            frame_id, str(camera.image_name), alpha_geometry,
            geometry_fragments, geometry_payload.labels,
        ))
        semantic_records.append(_record(
            frame_id, str(camera.image_name), alpha_semantic,
            semantic_fragments, semantic_payload.labels,
        ))
        geometry_offset += geometry_payload.count
        semantic_offset += semantic_payload.count

    _serialize(
        output_dir,
        xyz_m=feature_xyz * float(scene_scale_m_per_unit),
        affinity=affinity,
        semantic=semantic,
        label_features=label_features,
        geometry=geometry_records,
        semantics=semantic_records,
    )
    identity = build_lifting_identity(
        scene_id=scene_id,
        git_commit=git_commit,
        feature_ply=feature_ply_path,
        feature_record=feature_record_path,
        label_features=label_features_path,
        segment_everything_root=segment_everything_root,
        classes=classes,
        config=FragmentConfig(),
    )
    metadata = {
        "schema": V9_LIFTING_SCHEMA,
        "scene_id": str(scene_id),
        "git_commit": str(git_commit),
        "identity": identity,
        "lifting_source": "M1-core+AM-full",
        "mask_source": "SAM-everything",
        "feature_source": "v9-10k-objectbank",
        "point_count": int(point_count),
        "frame_count": len(geometry_records),
        "fragment_count": sum(len(row.fragments) for row in geometry_records),
        "semantic_fragment_count": sum(len(row.fragments) for row in semantic_records),
        "mask_count": sum(row.mask_count for row in geometry_records),
        "grounded_abstention_frame_count": sum(row.abstained for row in semantic_records),
        "classes": list(classes),
        "config": asdict(FragmentConfig()),
        "arrays_npz": "lifting_bank.npz",
        "runtime_seconds": float(time.monotonic() - started),
    }
    (output_dir / "lifting_bank.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), "utf-8"
    )
    if not lifting_bank_is_complete(
        output_dir,
        expected_scene_id=scene_id,
        expected_git_commit=git_commit,
        expected_identity=identity,
        expected_feature_record_identity=identity["feature_record_identity"],
    ):
        raise RuntimeError("serialized native V9 lifting bank is invalid")
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--base-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-scale-m-per-unit", type=float, required=True)
    parser.add_argument("--segment-everything-root", type=Path, required=True)
    parser.add_argument("--feature-ply", type=Path, required=True)
    parser.add_argument("--feature-record", type=Path, required=True)
    parser.add_argument("--label-features", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args(argv)
    expected_identity = build_lifting_identity(
        scene_id=args.scene_id,
        git_commit=args.git_commit,
        feature_ply=args.feature_ply,
        feature_record=args.feature_record,
        label_features=args.label_features,
        segment_everything_root=args.segment_everything_root,
        classes=DEFAULT_CLASSES,
        config=FragmentConfig(),
    )
    if lifting_bank_is_complete(
        args.output_dir,
        expected_scene_id=args.scene_id,
        expected_git_commit=args.git_commit,
        expected_identity=expected_identity,
        expected_feature_record_identity=expected_identity[
            "feature_record_identity"
        ],
    ):
        print(f"complete native V9 lifting bank exists: {args.output_dir}")
        return 0
    result = run_v9_lifting_bank(
        scene_id=args.scene_id,
        base_path=args.base_path,
        output_dir=args.output_dir,
        scene_scale_m_per_unit=args.scene_scale_m_per_unit,
        segment_everything_root=args.segment_everything_root,
        feature_ply_path=args.feature_ply,
        feature_record_path=args.feature_record,
        label_features_path=args.label_features,
        git_commit=args.git_commit,
    )
    print(json.dumps({key: result[key] for key in ("scene_id", "fragment_count", "runtime_seconds")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
