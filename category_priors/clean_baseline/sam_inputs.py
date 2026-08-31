from __future__ import annotations

"""Stage-local validation and recovery of packed SAM-everything masks.

The experiment registration freezes every runtime row up front, but large
scene assets are deliberately checked only when their preregistered stage is
reached.  This module performs that check against COLMAP's camera names.  If a
packed frame is absent, it may regenerate only that immutable input from an
already present SAM checkpoint into an isolated clean-baseline directory.
Nothing in this module downloads a weight or changes historical masks.
"""

import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


SAM_EVERYTHING_CONFIG: dict[str, Any] = {
    "points_per_side": 32,
    "pred_iou_thresh": 0.88,
    "stability_score_thresh": 0.95,
    "box_nms_thresh": 0.70,
    "crop_n_layers": 0,
    "crop_n_points_downscale_factor": 1,
    "min_mask_region_area": 100,
}


@dataclass(frozen=True)
class ColmapFrameSpec:
    image_name: str
    relative_image_path: str
    height: int
    width: int


def _read_cameras_binary(path: Path) -> dict[int, tuple[int, int]]:
    result: dict[int, tuple[int, int]] = {}
    with path.open("rb") as handle:
        (count,) = struct.unpack("<Q", handle.read(8))
        model_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12,
                        7: 5, 8: 4, 9: 5, 10: 12}
        for _ in range(count):
            camera_id, model_id, width, height = struct.unpack(
                "<iiQQ", handle.read(24)
            )
            if model_id not in model_params:
                raise ValueError(f"unsupported COLMAP camera model {model_id}")
            handle.seek(8 * model_params[model_id], os.SEEK_CUR)
            result[int(camera_id)] = (int(height), int(width))
    return result


def _read_images_binary(
    path: Path, dimensions: Mapping[int, tuple[int, int]]
) -> tuple[ColmapFrameSpec, ...]:
    rows: list[ColmapFrameSpec] = []
    with path.open("rb") as handle:
        (count,) = struct.unpack("<Q", handle.read(8))
        for _ in range(count):
            header = handle.read(64)
            if len(header) != 64:
                raise ValueError("truncated COLMAP images.bin")
            values = struct.unpack("<idddddddi", header)
            camera_id = int(values[-1])
            name_bytes = bytearray()
            while True:
                value = handle.read(1)
                if not value:
                    raise ValueError("truncated COLMAP image name")
                if value == b"\x00":
                    break
                name_bytes.extend(value)
            relative = name_bytes.decode("utf-8")
            (point_count,) = struct.unpack("<Q", handle.read(8))
            handle.seek(int(point_count) * 24, os.SEEK_CUR)
            if camera_id not in dimensions:
                raise ValueError(f"COLMAP image references unknown camera {camera_id}")
            height, width = dimensions[camera_id]
            rows.append(
                ColmapFrameSpec(
                    image_name=str(Path(relative).with_suffix("")),
                    relative_image_path=relative,
                    height=height,
                    width=width,
                )
            )
    return tuple(sorted(rows, key=lambda row: row.image_name))


def _read_cameras_text(path: Path) -> dict[int, tuple[int, int]]:
    result: dict[int, tuple[int, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        fields = value.split()
        result[int(fields[0])] = (int(fields[3]), int(fields[2]))
    return result


def _read_images_text(
    path: Path, dimensions: Mapping[int, tuple[int, int]]
) -> tuple[ColmapFrameSpec, ...]:
    rows: list[ColmapFrameSpec] = []
    data_rows = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    # COLMAP writes one metadata row followed by one (possibly empty) points
    # row.  Some exported files omit empty points rows, so identify metadata
    # by its fixed prefix rather than blindly taking every second line.
    for value in data_rows:
        fields = value.split()
        if len(fields) < 10:
            continue
        try:
            camera_id = int(fields[8])
            int(fields[0])
            tuple(float(item) for item in fields[1:8])
        except (TypeError, ValueError):
            continue
        try:
            float(fields[9])
        except ValueError:
            pass
        else:
            # A POINTS2D row consists only of numeric triples.
            continue
        if camera_id not in dimensions:
            raise ValueError(f"COLMAP image references unknown camera {camera_id}")
        relative = fields[9]
        height, width = dimensions[camera_id]
        rows.append(
            ColmapFrameSpec(
                image_name=str(Path(relative).with_suffix("")),
                relative_image_path=relative,
                height=height,
                width=width,
            )
        )
    return tuple(sorted(rows, key=lambda row: row.image_name))


def colmap_frame_specs(sparse_root: str | Path) -> tuple[ColmapFrameSpec, ...]:
    sparse = Path(sparse_root)
    if (sparse / "cameras.bin").is_file() and (sparse / "images.bin").is_file():
        cameras = _read_cameras_binary(sparse / "cameras.bin")
        rows = _read_images_binary(sparse / "images.bin", cameras)
    elif (sparse / "cameras.txt").is_file() and (sparse / "images.txt").is_file():
        cameras = _read_cameras_text(sparse / "cameras.txt")
        rows = _read_images_text(sparse / "images.txt", cameras)
    else:
        raise FileNotFoundError(f"COLMAP camera registration is missing under {sparse}")
    if not rows or len({row.image_name for row in rows}) != len(rows):
        raise ValueError("COLMAP must register non-empty, unique camera names")
    return rows


def packed_frame_is_valid(path: Path, *, height: int, width: int) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            packed = np.asarray(payload["packed"], dtype=np.uint8)
            count = int(np.asarray(payload["count"]).item())
            stored_height = int(np.asarray(payload["height"]).item())
            stored_width = int(np.asarray(payload["width"]).item())
    except (OSError, ValueError, KeyError, EOFError):
        return False
    return (
        count >= 0
        and stored_height == int(height)
        and stored_width == int(width)
        and packed.shape == (count, (int(height) * int(width) + 7) // 8)
    )


def audit_scene_masks(
    *,
    frames: Sequence[ColmapFrameSpec],
    sam_root: Path,
    grounded_masks_root: Path,
    grounded_labels_root: Path,
) -> dict[str, Any]:
    missing: list[str] = []
    invalid: list[str] = []
    grounded_one_sided: list[str] = []
    for frame in frames:
        target = sam_root / f"{frame.image_name}.npz"
        if not target.is_file():
            missing.append(frame.image_name)
        elif not packed_frame_is_valid(
            target, height=frame.height, width=frame.width
        ):
            invalid.append(frame.image_name)
        mask_exists = (grounded_masks_root / f"{frame.image_name}.pt").is_file()
        label_exists = (grounded_labels_root / f"{frame.image_name}.pt").is_file()
        if mask_exists != label_exists:
            grounded_one_sided.append(frame.image_name)
    return {
        "frame_count": len(frames),
        "valid_sam_frame_count": len(frames) - len(missing) - len(invalid),
        "missing_sam_frames": missing,
        "invalid_sam_frames": invalid,
        "grounded_one_sided_frames": grounded_one_sided,
        "grounded_abstention_allowed": True,
        "complete": not missing and not invalid and not grounded_one_sided,
    }


def _atomic_save_packed(path: Path, masks: np.ndarray) -> None:
    array = np.asarray(masks, dtype=np.bool_)
    if array.ndim != 3:
        raise ValueError("SAM masks must be MxHxW")
    count, height, width = array.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            packed=np.packbits(array.reshape(count, height * width), axis=1),
            count=np.asarray(count, dtype=np.int32),
            height=np.asarray(height, dtype=np.int32),
            width=np.asarray(width, dtype=np.int32),
        )
    os.replace(temporary, path)


def _default_generator_factory(
    checkpoint: Path, arch: str, device: str, config: Mapping[str, Any]
) -> Any:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    model = sam_model_registry[str(arch)](checkpoint=str(checkpoint)).to(device)
    return SamAutomaticMaskGenerator(model=model, **dict(config))


def ensure_scene_sam_masks(
    *,
    frames: Sequence[ColmapFrameSpec],
    images_root: Path,
    primary_root: Path,
    grounded_masks_root: Path,
    grounded_labels_root: Path,
    generation: Mapping[str, Any] | None,
    generator_factory: Callable[[Path, str, str, Mapping[str, Any]], Any]
    | None = None,
) -> dict[str, Any]:
    """Return one complete packed-mask root or an explicit unavailable row."""

    primary = audit_scene_masks(
        frames=frames,
        sam_root=primary_root,
        grounded_masks_root=grounded_masks_root,
        grounded_labels_root=grounded_labels_root,
    )
    if primary["grounded_one_sided_frames"]:
        return {
            "status": "unavailable",
            "reason": "Grounded-SAM mask/label inputs are one-sided",
            "sam_root": str(primary_root),
            "audit": primary,
            "generation_attempted": False,
        }
    if primary["complete"]:
        return {
            "status": "complete",
            "source": "registered",
            "sam_root": str(primary_root),
            "audit": primary,
            "generation_attempted": False,
        }
    if not isinstance(generation, Mapping):
        return {
            "status": "unavailable",
            "reason": "packed SAM frames are missing and no generator is registered",
            "sam_root": str(primary_root),
            "audit": primary,
            "generation_attempted": False,
        }
    output_root = Path(str(generation.get("output_root", ""))).resolve()
    checkpoint = Path(str(generation.get("checkpoint", ""))).resolve()
    if not checkpoint.is_file():
        return {
            "status": "unavailable",
            "reason": f"existing SAM checkpoint is unavailable: {checkpoint}",
            "sam_root": str(primary_root),
            "audit": primary,
            "generation_attempted": False,
            "download_attempted": False,
        }
    output_root.mkdir(parents=True, exist_ok=True)
    # Reuse every valid registered frame without mutating the source root.
    for frame in frames:
        source = primary_root / f"{frame.image_name}.npz"
        target = output_root / f"{frame.image_name}.npz"
        if packed_frame_is_valid(source, height=frame.height, width=frame.width):
            target.parent.mkdir(parents=True, exist_ok=True)
            if not packed_frame_is_valid(
                target, height=frame.height, width=frame.width
            ):
                temporary = target.with_name(target.name + ".part")
                shutil.copy2(source, temporary)
                os.replace(temporary, target)
    current = audit_scene_masks(
        frames=frames,
        sam_root=output_root,
        grounded_masks_root=grounded_masks_root,
        grounded_labels_root=grounded_labels_root,
    )
    pending = set(current["missing_sam_frames"] + current["invalid_sam_frames"])
    if pending:
        try:
            import cv2

            factory = generator_factory or _default_generator_factory
            generator = factory(
                checkpoint,
                str(generation.get("sam_arch", "vit_h")),
                str(generation.get("device", "cuda")),
                generation.get("config", SAM_EVERYTHING_CONFIG),
            )
            for frame in frames:
                if frame.image_name not in pending:
                    continue
                image_path = images_root / frame.relative_image_path
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"failed to decode COLMAP image: {image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if image.shape[:2] != (frame.height, frame.width):
                    raise ValueError(
                        f"{frame.image_name}: image/COLMAP dimensions differ"
                    )
                rows = generator.generate(image)
                masks = (
                    np.stack(
                        [np.asarray(row["segmentation"], dtype=np.bool_) for row in rows]
                    )
                    if rows
                    else np.zeros((0, frame.height, frame.width), dtype=np.bool_)
                )
                _atomic_save_packed(
                    output_root / f"{frame.image_name}.npz", masks
                )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return {
                "status": "unavailable",
                "reason": f"on-demand SAM generation failed: {exc}",
                "sam_root": str(output_root),
                "audit": current,
                "generation_attempted": True,
                "download_attempted": False,
            }
    final = audit_scene_masks(
        frames=frames,
        sam_root=output_root,
        grounded_masks_root=grounded_masks_root,
        grounded_labels_root=grounded_labels_root,
    )
    if not final["complete"]:
        return {
            "status": "unavailable",
            "reason": "on-demand SAM output did not pass the COLMAP-frame audit",
            "sam_root": str(output_root),
            "audit": final,
            "generation_attempted": True,
            "download_attempted": False,
        }
    return {
        "status": "complete",
        "source": "generated-isolated",
        "sam_root": str(output_root),
        "audit": final,
        "generation_attempted": bool(pending),
        "download_attempted": False,
    }
