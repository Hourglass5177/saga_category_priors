from __future__ import annotations

"""Small, side-effect-free loader for scene runtime manifests."""

from pathlib import Path
from typing import Any

from .io import hash_json, load_json


def _validate_content_hash(payload: dict[str, Any]) -> None:
    declared = str(payload.get("content_sha256", ""))
    unhashed = {key: value for key, value in payload.items() if key != "content_sha256"}
    if not declared or hash_json(unhashed) != declared:
        raise ValueError("scene runtime manifest content_sha256 does not match")


def load_scene_runtime_manifest(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load paths and metric scale without importing an experiment runner."""

    source = Path(path)
    payload = load_json(source)
    if payload.get("kind") != "scene_runtime_manifest":
        raise ValueError("expected kind=scene_runtime_manifest")
    if "content_sha256" in payload:
        _validate_content_hash(payload)

    scenes: dict[str, dict[str, Any]] = {}
    base = source.parent
    for raw in payload.get("scenes", []):
        item = dict(raw)
        scene_id = str(item.get("scene_id", "")).strip()
        if not scene_id:
            raise ValueError("scene runtime entry is missing scene_id")
        if scene_id in scenes:
            raise ValueError(f"duplicate runtime scene: {scene_id}")
        scale = float(item.get("scene_scale_m_per_unit", 0.0))
        if scale <= 0:
            raise ValueError(f"{scene_id}: scene_scale_m_per_unit must be positive")
        base_path = Path(str(item.get("base_path", "")))
        if not base_path.is_absolute():
            base_path = (base / base_path).resolve()
        item["scene_id"] = scene_id
        item["base_path"] = str(base_path)
        item["scene_scale_m_per_unit"] = scale
        if item.get("python_bin"):
            python_bin = Path(str(item["python_bin"]))
            if not python_bin.is_absolute():
                python_bin = (base / python_bin).resolve()
            item["python_bin"] = str(python_bin)
        scenes[scene_id] = item
    if not scenes:
        raise ValueError("scene runtime manifest contains no scenes")
    return scenes


__all__ = ["load_scene_runtime_manifest"]
