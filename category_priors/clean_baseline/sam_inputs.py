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
import hashlib
import json
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
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

_GENERATION_MANIFEST_NAME = "generation_manifest.json"
_GENERATION_MANIFEST_SCHEMA = "saga-clean-generated-sam-v1"


@dataclass(frozen=True)
class ColmapFrameSpec:
    image_name: str
    relative_image_path: str
    height: int
    width: int


@dataclass(frozen=True)
class PackedMaskFrame:
    """Canonical packed masks for one COLMAP frame.

    Keeping the masks packed is important in production: a ScanNet frame can
    contain hundreds of SAM proposals, and eagerly expanding every proposal
    to ``MxHxW`` can consume several gigabytes before the first three-channel
    render starts.  Only :meth:`dense_batch` expands the requested rows.
    """

    packed: np.ndarray
    count: int
    height: int
    width: int

    def __post_init__(self) -> None:
        packed = np.asarray(self.packed)
        scalar_values: list[int] = []
        for name, value in (
            ("count", self.count), ("height", self.height), ("width", self.width)
        ):
            raw = np.asarray(value)
            if (
                raw.size != 1
                or np.issubdtype(raw.dtype, np.bool_)
                or not np.issubdtype(raw.dtype, np.integer)
            ):
                raise ValueError(f"packed SAM {name} must be one integer scalar")
            scalar_values.append(int(raw.reshape(()).item()))
        count, height, width = scalar_values
        if packed.dtype != np.uint8 or packed.ndim != 2:
            raise ValueError("packed SAM data must be a two-dimensional uint8 array")
        if count < 0 or height <= 0 or width <= 0:
            raise ValueError("packed SAM count/height/width are invalid")
        byte_count = (height * width + 7) // 8
        if packed.shape != (count, byte_count):
            raise ValueError("packed SAM array shape disagrees with its metadata")
        remainder = (height * width) % 8
        if remainder and count:
            unused_bits = (1 << (8 - remainder)) - 1
            if np.any(np.bitwise_and(packed[:, -1], unused_bits)):
                raise ValueError("packed SAM padding bits must be zero")
        packed = np.ascontiguousarray(packed)
        packed.setflags(write=False)
        object.__setattr__(self, "packed", packed)
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "width", width)

    def dense_batch(self, start: int, stop: int) -> np.ndarray:
        first, last = int(start), int(stop)
        if not 0 <= first <= last <= self.count:
            raise IndexError("packed SAM batch is out of range")
        pixels = self.height * self.width
        return np.unpackbits(
            self.packed[first:last], axis=1, count=pixels
        ).reshape(last - first, self.height, self.width).astype(
            np.bool_, copy=False
        )


def _payload_scalar_int(payload: Mapping[str, Any], name: str) -> int:
    value = np.asarray(payload[name])
    if value.size != 1 or np.issubdtype(value.dtype, np.bool_) or not np.issubdtype(
        value.dtype, np.integer
    ):
        raise ValueError(f"packed SAM {name} must be one integer scalar")
    return int(value.reshape(()).item())


def load_packed_mask_frame(
    path: str | Path, *, height: int | None = None, width: int | None = None
) -> PackedMaskFrame:
    """Load and strictly validate one canonical packed-mask payload."""

    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as payload:
            if set(payload.files) != {"packed", "count", "height", "width"}:
                raise ValueError("packed SAM payload has missing or unexpected arrays")
            raw_packed = np.asarray(payload["packed"])
            if raw_packed.dtype != np.uint8:
                raise ValueError("packed SAM bytes must use uint8 without coercion")
            # ``NpzFile`` returns an owning ndarray; closing the zip container
            # does not invalidate it.  Avoid a second full packed-stack copy.
            packed = raw_packed
            count = _payload_scalar_int(payload, "count")
            stored_height = _payload_scalar_int(payload, "height")
            stored_width = _payload_scalar_int(payload, "width")
    except (OSError, ValueError, KeyError, EOFError) as error:
        raise ValueError(f"invalid packed SAM-everything frame: {source}") from error
    frame = PackedMaskFrame(packed, count, stored_height, stored_width)
    if height is not None and frame.height != int(height):
        raise ValueError("packed SAM height differs from COLMAP")
    if width is not None and frame.width != int(width):
        raise ValueError("packed SAM width differs from COLMAP")
    return frame


def _read_exact(handle: Any, size: int, label: str) -> bytes:
    payload = handle.read(int(size))
    if len(payload) != int(size):
        raise ValueError(f"truncated COLMAP {label}")
    return payload


def _skip_exact(handle: Any, size: int, label: str) -> None:
    count = int(size)
    if count < 0:
        raise ValueError(f"invalid COLMAP {label} byte count")
    start = handle.tell()
    handle.seek(0, os.SEEK_END)
    end = handle.tell()
    if end - start < count:
        raise ValueError(f"truncated COLMAP {label}")
    handle.seek(start + count, os.SEEK_SET)


def _canonical_image_path(value: str) -> tuple[str, str]:
    relative = PurePosixPath(str(value).replace("\\", "/"))
    if (
        relative.is_absolute()
        or not relative.parts
        or ":" in relative.parts[0]
        or any(part in ("", ".", "..") for part in relative.parts)
    ):
        raise ValueError(f"COLMAP image path is not a safe relative path: {value!r}")
    canonical = relative.as_posix()
    image_name = relative.with_suffix("").as_posix()
    if not image_name:
        raise ValueError("COLMAP image name is empty")
    return image_name, canonical


def _read_cameras_binary(path: Path) -> dict[int, tuple[int, int]]:
    result: dict[int, tuple[int, int]] = {}
    with path.open("rb") as handle:
        (count,) = struct.unpack("<Q", _read_exact(handle, 8, "camera count"))
        model_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12,
                        7: 5, 8: 4, 9: 5, 10: 12}
        for _ in range(count):
            camera_id, model_id, width, height = struct.unpack(
                "<iiQQ", _read_exact(handle, 24, "camera record")
            )
            if model_id not in model_params:
                raise ValueError(f"unsupported COLMAP camera model {model_id}")
            if int(camera_id) in result or int(width) <= 0 or int(height) <= 0:
                raise ValueError("COLMAP cameras must have unique IDs and positive dimensions")
            _read_exact(
                handle, 8 * model_params[model_id], "camera parameters"
            )
            result[int(camera_id)] = (int(height), int(width))
    return result


def _read_images_binary(
    path: Path, dimensions: Mapping[int, tuple[int, int]]
) -> tuple[ColmapFrameSpec, ...]:
    rows: list[ColmapFrameSpec] = []
    with path.open("rb") as handle:
        (count,) = struct.unpack("<Q", _read_exact(handle, 8, "image count"))
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
            image_name, relative = _canonical_image_path(
                name_bytes.decode("utf-8")
            )
            (point_count,) = struct.unpack(
                "<Q", _read_exact(handle, 8, "point count")
            )
            _skip_exact(handle, int(point_count) * 24, "points2D")
            if camera_id not in dimensions:
                raise ValueError(f"COLMAP image references unknown camera {camera_id}")
            height, width = dimensions[camera_id]
            rows.append(
                ColmapFrameSpec(
                    image_name=image_name,
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
        if len(fields) < 4:
            raise ValueError("malformed COLMAP text camera record")
        camera_id, width, height = int(fields[0]), int(fields[2]), int(fields[3])
        if camera_id in result or width <= 0 or height <= 0:
            raise ValueError("COLMAP cameras must have unique IDs and positive dimensions")
        result[camera_id] = (height, width)
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
        image_name, relative = _canonical_image_path(fields[9])
        height, width = dimensions[camera_id]
        rows.append(
            ColmapFrameSpec(
                image_name=image_name,
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
        load_packed_mask_frame(path, height=height, width=width)
    except (OSError, ValueError, KeyError, EOFError):
        return False
    return True


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


def _file_digest(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return {"size": size, "sha256": digest.hexdigest()}


def _generation_request_identity(
    *,
    frames: Sequence[ColmapFrameSpec],
    images_root: Path,
    primary_root: Path,
    checkpoint: Path,
    sam_arch: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    frame_rows: list[dict[str, Any]] = []
    for frame in frames:
        image_path = images_root / frame.relative_image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"registered COLMAP image is missing: {image_path}")
        primary_path = primary_root / f"{frame.image_name}.npz"
        primary_identity: dict[str, Any] | None = None
        if packed_frame_is_valid(
            primary_path, height=frame.height, width=frame.width
        ):
            primary_identity = _file_digest(primary_path)
        frame_rows.append(
            {
                "image_name": frame.image_name,
                "relative_image_path": frame.relative_image_path,
                "height": frame.height,
                "width": frame.width,
                "image": _file_digest(image_path),
                "registered_sam": primary_identity,
            }
        )
    return {
        "schema": _GENERATION_MANIFEST_SCHEMA,
        "checkpoint": _file_digest(checkpoint),
        "sam_arch": str(sam_arch),
        "config": dict(config),
        "frames": frame_rows,
    }


def _generated_cache_is_valid(
    *,
    output_root: Path,
    request: Mapping[str, Any],
    frames: Sequence[ColmapFrameSpec],
) -> bool:
    manifest_path = output_root / _GENERATION_MANIFEST_NAME
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if set(payload) != {"request", "outputs"} or payload["request"] != request:
            return False
        outputs = payload["outputs"]
        if not isinstance(outputs, dict) or set(outputs) != {
            frame.image_name for frame in frames
        }:
            return False
        for frame in frames:
            target = output_root / f"{frame.image_name}.npz"
            if not packed_frame_is_valid(
                target, height=frame.height, width=frame.width
            ) or outputs[frame.image_name] != _file_digest(target):
                return False
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False
    return True


def _write_generation_manifest(
    *,
    output_root: Path,
    request: Mapping[str, Any],
    frames: Sequence[ColmapFrameSpec],
) -> None:
    payload = {
        "request": dict(request),
        "outputs": {
            frame.image_name: _file_digest(
                output_root / f"{frame.image_name}.npz"
            )
            for frame in frames
        },
    }
    target = output_root / _GENERATION_MANIFEST_NAME
    temporary = target.with_name(target.name + ".part")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, target)


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
    raw_output_root = generation.get("output_root")
    if not isinstance(raw_output_root, (str, os.PathLike)) or not str(
        raw_output_root
    ).strip():
        return {
            "status": "unavailable",
            "reason": "SAM generation requires an explicit isolated output_root",
            "sam_root": str(primary_root),
            "audit": primary,
            "generation_attempted": False,
            "download_attempted": False,
        }
    output_root = Path(raw_output_root).resolve()
    historical_root = Path(primary_root).resolve()
    if (
        output_root == historical_root
        or output_root.is_relative_to(historical_root)
        or historical_root.is_relative_to(output_root)
    ):
        return {
            "status": "unavailable",
            "reason": (
                "SAM generation output_root must be disjoint from the historical "
                "mask root"
            ),
            "sam_root": str(primary_root),
            "audit": primary,
            "generation_attempted": False,
            "download_attempted": False,
        }
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
    requested_config = dict(generation.get("config", SAM_EVERYTHING_CONFIG))
    if requested_config != SAM_EVERYTHING_CONFIG:
        return {
            "status": "unavailable",
            "reason": "SAM-everything generation parameters differ from the frozen configuration",
            "sam_root": str(primary_root),
            "audit": primary,
            "generation_attempted": False,
            "download_attempted": False,
        }
    sam_arch = str(generation.get("sam_arch", "vit_h"))
    try:
        request_identity = _generation_request_identity(
            frames=frames,
            images_root=images_root,
            primary_root=primary_root,
            checkpoint=checkpoint,
            sam_arch=sam_arch,
            config=requested_config,
        )
    except (OSError, ValueError) as exc:
        return {
            "status": "unavailable",
            "reason": f"SAM generation inputs failed identity validation: {exc}",
            "sam_root": str(primary_root),
            "audit": primary,
            "generation_attempted": False,
            "download_attempted": False,
        }
    output_root.mkdir(parents=True, exist_ok=True)
    cache_valid = _generated_cache_is_valid(
        output_root=output_root, request=request_identity, frames=frames
    )
    if not cache_valid:
        # Rebuild every registered row from its immutable source.  A
        # shape-valid file in the isolated output is not reusable unless its
        # request and output digests are covered by the manifest.
        for frame in frames:
            source = primary_root / f"{frame.image_name}.npz"
            target = output_root / f"{frame.image_name}.npz"
            if packed_frame_is_valid(
                source, height=frame.height, width=frame.width
            ):
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".part")
                shutil.copy2(source, temporary)
                os.replace(temporary, target)
    current = audit_scene_masks(
        frames=frames,
        sam_root=output_root,
        grounded_masks_root=grounded_masks_root,
        grounded_labels_root=grounded_labels_root,
    )
    pending = (
        set()
        if cache_valid
        else {
            frame.image_name
            for frame in frames
            if not packed_frame_is_valid(
                primary_root / f"{frame.image_name}.npz",
                height=frame.height,
                width=frame.width,
            )
        }
    )
    if pending:
        try:
            import cv2

            factory = generator_factory or _default_generator_factory
            generator = factory(
                checkpoint,
                sam_arch,
                str(generation.get("device", "cuda")),
                requested_config,
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
    _write_generation_manifest(
        output_root=output_root, request=request_identity, frames=frames
    )
    return {
        "status": "complete",
        "source": "generated-isolated",
        "sam_root": str(output_root),
        "audit": final,
        "generation_attempted": bool(pending),
        "download_attempted": False,
    }
