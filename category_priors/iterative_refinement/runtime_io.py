from __future__ import annotations

"""Runtime-only SAGA I/O kept separate from the historical postprocessor."""

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    GaussianEvidence,
    LineageRecord,
    MaskHypothesis,
    ObjectState,
    SCHEMA,
)


def load_cameras(args: Any) -> list[Any]:
    from scene.dataset_readers import (
        readColmapCameras,
        read_extrinsics_binary,
        read_extrinsics_text,
        read_intrinsics_binary,
        read_intrinsics_text,
    )
    from utils.camera_utils import cameraList_from_camInfos

    sparse = Path(args.sparse_path)
    binary = (sparse / "images.bin", sparse / "cameras.bin")
    text = (sparse / "images.txt", sparse / "cameras.txt")
    if all(path.is_file() for path in binary):
        infos = readColmapCameras(read_extrinsics_binary(str(binary[0])), read_intrinsics_binary(str(binary[1])), args.images_path)
    elif all(path.is_file() for path in text):
        infos = readColmapCameras(read_extrinsics_text(str(text[0])), read_intrinsics_text(str(text[1])), args.images_path)
    else:
        raise FileNotFoundError("COLMAP cameras require a complete binary or text pair")
    return cameraList_from_camInfos(infos, 1, args)


def camera_rgb(camera: Any) -> np.ndarray:
    image = camera.original_image.detach().cpu().clamp(0, 1).numpy()
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"{camera.image_name}: invalid RGB tensor")
    return np.rint(np.moveaxis(image, 0, 2) * 255.0).astype(np.uint8)


def camera_center(camera: Any) -> np.ndarray:
    value = camera.camera_center.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(3)


def focal_geometric_mean(camera: Any) -> float:
    # Gaussian Splatting camera objects store horizontal/vertical FoV.
    fx = float(camera.image_width) / (2.0 * np.tan(float(camera.FoVx) / 2.0))
    fy = float(camera.image_height) / (2.0 * np.tan(float(camera.FoVy) / 2.0))
    return float(np.sqrt(fx * fy))


def json_atomic(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{uuid.uuid4().hex}.part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, destination)


def npz_atomic(path: str | Path, **arrays: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{uuid.uuid4().hex}.part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, destination)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_scene_cache(
    output_dir: str | Path,
    *,
    hypotheses: Sequence[MaskHypothesis],
    evidence: Mapping[int, GaussianEvidence],
    states_by_profile: Mapping[str, Sequence[ObjectState]],
    lineage: Sequence[LineageRecord],
    provenance: Mapping[str, Any],
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    hypothesis_rows = []
    for index, row in enumerate(hypotheses):
        key = f"mask_{index}"
        arrays[key] = np.asarray(row.packed_mask, dtype=np.uint8)
        hypothesis_rows.append({
            key: value for key, value in row.__dict__.items() if key != "packed_mask"
        } | {"packed_key": key})
    evidence_rows = []
    for candidate_id, row in sorted(evidence.items()):
        prefix = f"e_{candidate_id}"
        arrays[prefix + "_ids"] = row.point_ids
        arrays[prefix + "_hp"] = row.hard_positive_views
        arrays[prefix + "_hn"] = row.hard_negative_views
        arrays[prefix + "_soft"] = row.alpha_soft_support
        evidence_rows.append({
            "candidate_id": candidate_id,
            "prefix": prefix,
            "independent_positive_views": row.independent_positive_views,
            "independent_negative_views": row.independent_negative_views,
            "selected_hypothesis_ids": list(row.selected_hypothesis_ids),
        })
    state_rows = []
    for profile, states in sorted(states_by_profile.items()):
        for ordinal, row in enumerate(states):
            prefix = f"s_{profile}_{ordinal}"
            arrays[prefix + "_ids"] = row.point_ids
            arrays[prefix + "_anchors"] = row.anchor_ids
            arrays[prefix + "_hard"] = row.hard_positive_ids
            arrays[prefix + "_counts"] = row.hard_positive_counts
            arrays[prefix + "_margin"] = row.evidence_margin
            state_rows.append({
                "profile": profile, "prefix": prefix,
                "object_id": row.object_id,
                "parent_candidate_ids": list(row.parent_candidate_ids),
                "review_class": row.review_class,
                "reliable_review_class": row.reliable_review_class,
                "round_index": row.round_index,
                "changed": row.changed,
            })
    npz_atomic(root / "refinement_cache.npz", **arrays)
    json_atomic(root / "refinement_cache.json", {
        "schema": SCHEMA,
        "kind": "iterative-refinement-cache",
        "hypotheses": hypothesis_rows,
        "evidence": evidence_rows,
        "states": state_rows,
        "lineage": [row.__dict__ for row in lineage],
        "provenance": dict(provenance),
    })


__all__ = [
    "camera_center", "camera_rgb", "file_sha256", "focal_geometric_mean",
    "json_atomic", "load_cameras", "npz_atomic", "save_scene_cache",
]
