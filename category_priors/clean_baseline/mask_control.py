from __future__ import annotations

"""Prepare a paired hierarchy/flat mask control from one SAM generation.

This module owns only immutable input preparation.  It never renders a
Gaussian, reads GT, assigns a class, or runs the consensus algorithm.  Every
frame is generated once into a metadata-rich source payload and both H' and P
are then deterministic projections of that same payload.
"""

import hashlib
import importlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .mask_contract import (
    SamMaskMetadataFrame,
    audit_flat_mask_contract,
    flatten_mask_stack,
    load_sam_mask_metadata,
    metadata_frame_from_sam_rows,
    save_sam_mask_metadata,
)
from .sam_inputs import (
    SAM_EVERYTHING_CONFIG,
    ColmapFrameSpec,
    colmap_frame_specs,
    load_packed_mask_frame,
)
from .worker import DEFAULT_CLASSES, resolve_clean_scene_inputs


MASK_CONTROL_REQUEST_SCHEMA = "saga-clean-mask-control-request-v1"
MASK_CONTROL_STATE_SCHEMA = "saga-clean-mask-control-state-v1"
_FULL_COMMIT = re.compile(r"[0-9a-f]{40}")
_OPENCV_FALLBACK_ENV = "SAGA_EXISTING_OPENCV_SITE_PACKAGES"


def _load_cv2(
    import_module: Callable[[str], Any] = importlib.import_module,
) -> Any:
    """Load OpenCV, optionally reusing one explicitly registered local tree.

    The RTX 5090 runtime intentionally uses the existing Torch/CUDA 12.8
    environment.  Some hosts keep OpenCV only in an older local environment.
    An explicit fallback path is appended (never prepended) so it cannot
    shadow the active Torch, NumPy, or SciPy packages.  Nothing is installed
    or downloaded.
    """

    try:
        return import_module("cv2")
    except ModuleNotFoundError as exc:
        if exc.name not in {None, "cv2"}:
            raise
        raw = os.environ.get(_OPENCV_FALLBACK_ENV, "").strip()
        if not raw:
            raise RuntimeError(
                "OpenCV is unavailable; set SAGA_EXISTING_OPENCV_SITE_PACKAGES "
                "to an existing local site-packages directory"
            ) from exc
        root = Path(raw).resolve(strict=True)
        if not root.is_dir() or not (root / "cv2" / "__init__.py").is_file():
            raise RuntimeError(
                "SAGA_EXISTING_OPENCV_SITE_PACKAGES does not contain cv2"
            ) from exc
        value = str(root)
        if value not in sys.path:
            sys.path.append(value)
        try:
            return import_module("cv2")
        except ModuleNotFoundError as fallback_exc:
            raise RuntimeError(
                "the registered local OpenCV package cannot be imported"
            ) from fallback_exc


def _file_identity(path: Path) -> dict[str, Any]:
    source = path.resolve(strict=True)
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return {"path": str(source), "size": size, "sha256": digest.hexdigest()}


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _save_packed_only(path: Path, frame: SamMaskMetadataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            packed=frame.packed,
            count=np.asarray(frame.count, dtype=np.int32),
            height=np.asarray(frame.height, dtype=np.int32),
            width=np.asarray(frame.width, dtype=np.int32),
        )
    os.replace(temporary, path)


def _save_flat_map(path: Path, source_mask_ids: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            source_mask_ids=np.asarray(source_mask_ids, dtype=np.int32),
        )
    os.replace(temporary, path)


def _default_generator_factory(
    checkpoint: Path, arch: str, device: str, config: Mapping[str, Any]
) -> Any:
    _load_cv2()
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    model = sam_model_registry[str(arch)](checkpoint=str(checkpoint)).to(device)
    return SamAutomaticMaskGenerator(model=model, **dict(config))


def _default_image_loader(path: Path) -> np.ndarray:
    cv2 = _load_cv2()

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to decode COLMAP image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _load_source_request(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        payload = dict(value)
    else:
        payload = json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("source evidence request must be a JSON object")
    return payload


def _request_identity(
    *,
    scene_id: str,
    producer_commit: str,
    checkpoint: Path,
    sam_arch: str,
    config: Mapping[str, Any],
    images_root: Path,
    frames: Sequence[ColmapFrameSpec],
) -> dict[str, Any]:
    return {
        "schema": MASK_CONTROL_REQUEST_SCHEMA,
        "scene_id": str(scene_id),
        "producer_commit": str(producer_commit),
        "checkpoint": _file_identity(checkpoint),
        "sam_arch": str(sam_arch),
        "config": dict(config),
        "frames": [
            {
                "image_name": frame.image_name,
                "relative_image_path": frame.relative_image_path,
                "height": frame.height,
                "width": frame.width,
                "image": _file_identity(images_root / frame.relative_image_path),
            }
            for frame in frames
        ],
    }


def _metadata_is_valid(path: Path, frame: ColmapFrameSpec) -> bool:
    try:
        loaded = load_sam_mask_metadata(path)
    except (OSError, ValueError, KeyError, EOFError):
        return False
    return (loaded.height, loaded.width) == (frame.height, frame.width)


def _content_identity(path: Path) -> dict[str, Any]:
    identity = _file_identity(path)
    return {"size": int(identity["size"]), "sha256": str(identity["sha256"])}


def _packed_matches(
    path: Path, frame: ColmapFrameSpec, expected: SamMaskMetadataFrame
) -> bool:
    try:
        loaded = load_packed_mask_frame(path, height=frame.height, width=frame.width)
    except (OSError, ValueError, KeyError, EOFError):
        return False
    return bool(
        loaded.count == expected.count
        and np.array_equal(loaded.packed, expected.packed)
    )


def _flat_map_matches(path: Path, source_mask_ids: np.ndarray) -> bool:
    try:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != {"source_mask_ids"}:
                return False
            loaded = np.asarray(payload["source_mask_ids"])
    except (OSError, ValueError, KeyError, EOFError):
        return False
    return bool(
        loaded.dtype == np.int32
        and np.array_equal(loaded, np.asarray(source_mask_ids, dtype=np.int32))
    )


def _initial_state(request_identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": MASK_CONTROL_STATE_SCHEMA,
        "request": dict(request_identity),
        "metadata_files": {},
    }


def _load_state(
    path: Path,
    *,
    request_identity: Mapping[str, Any],
    frame_names: set[str],
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise ValueError("existing SAM metadata state is invalid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "request",
        "metadata_files",
    }:
        raise ValueError("existing SAM metadata state has an invalid schema")
    if payload["schema"] != MASK_CONTROL_STATE_SCHEMA:
        raise ValueError("existing SAM metadata state has an invalid schema")
    if payload["request"] != dict(request_identity):
        raise ValueError("existing SAM metadata belongs to a different request")
    files = payload["metadata_files"]
    if not isinstance(files, dict) or not set(files).issubset(frame_names):
        raise ValueError("existing SAM metadata state has invalid frame identities")
    for image_name, identity in files.items():
        if (
            not isinstance(identity, dict)
            or set(identity) != {"size", "sha256"}
            or isinstance(identity["size"], bool)
            or not isinstance(identity["size"], int)
            or identity["size"] <= 0
            or not isinstance(identity["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", identity["sha256"]) is None
        ):
            raise ValueError(
                f"existing SAM metadata state has an invalid identity for {image_name}"
            )
    return payload


def _metadata_matches_state(
    path: Path,
    frame: ColmapFrameSpec,
    identity: Mapping[str, Any] | None,
) -> bool:
    if identity is None or not _metadata_is_valid(path, frame):
        return False
    try:
        return _content_identity(path) == dict(identity)
    except (OSError, ValueError):
        return False


def _derived_requests(
    *,
    source: Mapping[str, Any],
    scene_id: str,
    producer_commit: str,
    hierarchy_root: Path,
    flat_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    scene_value = source.get("scene", source.get("runtime_registration"))
    if not isinstance(scene_value, Mapping):
        raise TypeError("source evidence request lacks a scene object")
    scene = dict(scene_value)
    scene["scene_id"] = str(scene_id)
    classes = list(source.get("classes", DEFAULT_CLASSES))
    if tuple(map(str, classes)) != tuple(DEFAULT_CLASSES):
        raise ValueError("source request does not use the frozen 32-class order")

    def build(mask_root: Path, mode: str) -> dict[str, Any]:
        return {
            "schema": "saga-clean-alpha-mask-evidence-request-v1",
            "producer_commit": str(producer_commit),
            "classes": classes,
            "scene": scene,
            "sam_masks": str(mask_root.resolve()),
            "mask_observation_mode": mode,
        }

    return build(hierarchy_root, "hierarchy"), build(
        flat_root, "flat-highest-quality"
    )


def _bank_input_binding(
    *,
    scene_id: str,
    arm: str,
    request_path: Path,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind one derived request and selected mask tree to its future bank.

    The evidence producer embeds a content manifest for every registered input.
    Persisting that exact expected source here lets the later evaluation prove
    that the H'/P banks were built from these paired trees, instead of merely
    trusting an unrelated ``mechanical_contract_pass`` boolean.
    """

    from .evidence import evidence_request_source

    source = evidence_request_source(scene_id=str(scene_id), request=request)
    producer_inputs = source.get("producer_inputs")
    if not isinstance(producer_inputs, Mapping):
        raise RuntimeError(f"{arm}: evidence source lacks producer_inputs")
    mask_manifest = producer_inputs.get("sam_everything_masks")
    if not isinstance(mask_manifest, Mapping):
        raise RuntimeError(f"{arm}: evidence source lacks the SAM mask manifest")
    mask_root = Path(str(source.get("sam_masks", ""))).resolve()
    if not mask_root.is_dir():
        raise RuntimeError(f"{arm}: registered SAM mask root is unavailable")
    return {
        "arm": str(arm),
        "scene_id": str(scene_id),
        "mask_root": str(mask_root),
        "mask_manifest": dict(mask_manifest),
        "evidence_request": _file_identity(request_path),
        "expected_bank_source": source,
    }


def prepare_flat_mask_control_scene(
    *,
    scene_id: str,
    source_request: str | Path | Mapping[str, Any],
    output_root: str | Path,
    producer_commit: str,
    generator_factory: Callable[[Path, str, str, Mapping[str, Any]], Any]
    | None = None,
    image_loader: Callable[[Path], np.ndarray] | None = None,
) -> dict[str, Any]:
    """Generate one immutable SAM stack and derive paired H'/P roots."""

    if _FULL_COMMIT.fullmatch(str(producer_commit)) is None:
        raise ValueError("producer_commit must be a full lowercase Git commit")
    request = _load_source_request(source_request)
    scene_value = request.get("scene", request.get("runtime_registration"))
    if not isinstance(scene_value, Mapping):
        raise TypeError("source evidence request lacks a scene object")
    scene = dict(scene_value)
    if str(scene.get("scene_id", scene_id)) != str(scene_id):
        raise ValueError("source request scene_id differs from requested scene")
    inputs = resolve_clean_scene_inputs(scene)
    frames = colmap_frame_specs(inputs.sparse)
    generation = request.get("sam_generation")
    if not isinstance(generation, Mapping):
        raise ValueError("source request lacks the frozen SAM generation config")
    config = dict(generation.get("config", {}))
    if config != SAM_EVERYTHING_CONFIG:
        raise ValueError("SAM generation config differs from the frozen configuration")
    checkpoint = Path(str(generation.get("checkpoint", ""))).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"existing SAM checkpoint is unavailable: {checkpoint}")
    sam_arch = str(generation.get("sam_arch", "vit_h"))
    device = str(generation.get("device", "cuda"))

    root = Path(output_root).resolve()
    metadata_root = root / "sam-metadata" / str(scene_id)
    hierarchy_root = root / "masks" / "H-hierarchy" / str(scene_id)
    flat_root = root / "masks" / "P-flat" / str(scene_id)
    flat_map_root = root / "flat-maps" / str(scene_id)
    request_identity = _request_identity(
        scene_id=str(scene_id),
        producer_commit=str(producer_commit),
        checkpoint=checkpoint,
        sam_arch=sam_arch,
        config=config,
        images_root=inputs.images,
        frames=frames,
    )
    state_path = root / "sam-metadata" / str(scene_id) / "generation_state.json"
    frame_names = {frame.image_name for frame in frames}
    if state_path.is_file():
        state = _load_state(
            state_path,
            request_identity=request_identity,
            frame_names=frame_names,
        )
    else:
        state = _initial_state(request_identity)
        _write_json_atomic(state_path, state)

    pending = [
        frame
        for frame in frames
        if not _metadata_matches_state(
            metadata_root / f"{frame.image_name}.npz",
            frame,
            state["metadata_files"].get(frame.image_name),
        )
    ]
    generator: Any | None = None
    factory = generator_factory or _default_generator_factory
    load_image = image_loader or _default_image_loader
    rows: list[dict[str, Any]] = []
    historical_exact = 0
    for frame in frames:
        metadata_path = metadata_root / f"{frame.image_name}.npz"
        generated = False
        if frame in pending:
            if generator is None:
                generator = factory(checkpoint, sam_arch, device, config)
            image = np.asarray(load_image(inputs.images / frame.relative_image_path))
            if image.shape != (frame.height, frame.width, 3):
                raise ValueError(f"{frame.image_name}: image/COLMAP dimensions differ")
            raw_rows = generator.generate(image)
            metadata = metadata_frame_from_sam_rows(
                raw_rows, height=frame.height, width=frame.width
            )
            save_sam_mask_metadata(metadata_path, metadata)
            state["metadata_files"][frame.image_name] = _content_identity(
                metadata_path
            )
            _write_json_atomic(state_path, state)
            generated = True
        metadata = load_sam_mask_metadata(metadata_path)
        flat = flatten_mask_stack(metadata)
        frame_audit = audit_flat_mask_contract(metadata, flat)
        if not frame_audit["mechanical_contract_pass"]:
            raise RuntimeError(f"{frame.image_name}: flat mask contract failed")

        hierarchy_path = hierarchy_root / f"{frame.image_name}.npz"
        flat_path = flat_root / f"{frame.image_name}.npz"
        if not _packed_matches(hierarchy_path, frame, metadata):
            _save_packed_only(hierarchy_path, metadata)
        if not _packed_matches(flat_path, frame, flat.frame):
            _save_packed_only(flat_path, flat.frame)
        flat_map_path = flat_map_root / f"{frame.image_name}.npz"
        if not _flat_map_matches(flat_map_path, flat.source_mask_ids):
            _save_flat_map(flat_map_path, flat.source_mask_ids)

        historical_path = inputs.sam_masks / f"{frame.image_name}.npz"
        exact_historical = False
        if historical_path.is_file():
            try:
                historical = load_packed_mask_frame(
                    historical_path, height=frame.height, width=frame.width
                )
                exact_historical = bool(
                    historical.count == metadata.count
                    and np.array_equal(historical.packed, metadata.packed)
                )
            except (OSError, ValueError, KeyError, EOFError):
                exact_historical = False
        historical_exact += int(exact_historical)
        rows.append(
            {
                "scene_id": str(scene_id),
                "image_name": frame.image_name,
                "generated_this_run": generated,
                "historical_hierarchy_exact": exact_historical,
                **frame_audit,
            }
        )

    hierarchy_request, flat_request = _derived_requests(
        source=request,
        scene_id=str(scene_id),
        producer_commit=str(producer_commit),
        hierarchy_root=hierarchy_root,
        flat_root=flat_root,
    )
    request_root = root / "evidence-requests" / str(scene_id)
    hierarchy_request_path = request_root / "H-hierarchy.json"
    flat_request_path = request_root / "P-flat.json"
    _write_json_atomic(hierarchy_request_path, hierarchy_request)
    _write_json_atomic(flat_request_path, flat_request)
    input_bindings = {
        "H-hierarchy": _bank_input_binding(
            scene_id=str(scene_id),
            arm="H-hierarchy",
            request_path=hierarchy_request_path,
            request=hierarchy_request,
        ),
        "P-flat": _bank_input_binding(
            scene_id=str(scene_id),
            arm="P-flat",
            request_path=flat_request_path,
            request=flat_request,
        ),
    }
    input_binding_pass = bool(
        input_bindings["H-hierarchy"]["mask_root"]
        == str(hierarchy_root.resolve())
        and input_bindings["P-flat"]["mask_root"] == str(flat_root.resolve())
    )

    summary = {
        "schema": MASK_CONTROL_STATE_SCHEMA,
        "scene_id": str(scene_id),
        "frame_count": len(frames),
        "generated_frame_count": sum(bool(row["generated_this_run"]) for row in rows),
        "historical_hierarchy_exact_frame_count": historical_exact,
        "hierarchy_mask_count": sum(int(row["hierarchy_mask_count"]) for row in rows),
        "flat_mask_count": sum(int(row["flat_mask_count"]) for row in rows),
        "union_changed_pixel_count": sum(
            int(row["union_changed_pixel_count"]) for row in rows
        ),
        "flat_overlap_pixel_count": sum(
            int(row["flat_overlap_pixel_count"]) for row in rows
        ),
        "mechanical_contract_pass": all(
            bool(row["mechanical_contract_pass"]) for row in rows
        ),
        "metadata_root": str(metadata_root),
        "hierarchy_mask_root": str(hierarchy_root),
        "flat_mask_root": str(flat_root),
        "hierarchy_evidence_request": str(hierarchy_request_path),
        "flat_evidence_request": str(flat_request_path),
        "sam_generation_request_identity": request_identity,
        "input_bindings": input_bindings,
        "input_binding_pass": input_binding_pass,
        "frames": rows,
    }
    _write_json_atomic(root / f"{scene_id}_flat_mask_input_audit.json", summary)
    return summary


__all__ = [
    "MASK_CONTROL_REQUEST_SCHEMA",
    "MASK_CONTROL_STATE_SCHEMA",
    "prepare_flat_mask_control_scene",
]
