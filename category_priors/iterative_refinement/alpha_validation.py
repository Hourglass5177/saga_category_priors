from __future__ import annotations

"""Cloud-side numerical and performance validation for the fused backend."""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .alpha_backend import (
    AlphaEvidenceCache,
    gaussian_render_sha256,
    render_fused,
    render_gradient_original,
    render_gradient_reference,
)
from .contracts import RefinementConfig
from .runtime_io import json_atomic, load_cameras


def _review_masks(root: Path) -> dict[int, list[np.ndarray]]:
    result: dict[int, dict[str, np.ndarray]] = {}
    for path in sorted(root.glob("review_cache/round*/camera*/candidate*.npz")):
        camera_index = int(path.parent.name.removeprefix("camera"))
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            for row in metadata:
                packed = archive[row["packed_key"]]
                shape = tuple(row["mask_shape"])
                mask = np.unpackbits(packed)[: int(np.prod(shape))].reshape(shape).astype(bool)
                digest = __import__("hashlib").sha256(np.packbits(mask, bitorder="little").tobytes()).hexdigest()
                result.setdefault(camera_index, {})[digest] = mask
    return {index: list(rows.values()) for index, rows in result.items() if rows}


def _stratified_cameras(masks: dict[int, list[np.ndarray]], limit: int = 12) -> list[int]:
    rows = sorted((len(values), index) for index, values in masks.items())
    selected: list[int] = []
    for target in (2, 4, 14, 22):
        if rows:
            _, index = min(rows, key=lambda row: (abs(row[0] - target), row[1]))
            if index not in selected:
                selected.append(index)
    if len(selected) < min(limit, len(rows)):
        positions = np.linspace(0, len(rows) - 1, min(limit, len(rows))).round().astype(int)
        for position in positions:
            index = rows[int(position)][1]
            if index not in selected:
                selected.append(index)
            if len(selected) == limit:
                break
    return selected


def _timed(function: Any) -> tuple[Any, float]:
    import torch
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = function()
    torch.cuda.synchronize()
    return value, time.perf_counter() - start


def validate_alpha_backend(args: Any) -> dict[str, Any]:
    import torch
    from scene import GaussianModel

    config = RefinementConfig()
    model = GaussianModel(args.sh_degree)
    model.load_ply(args.point_cloud_path)
    cameras = load_cameras(args)
    masks = _review_masks(Path(args.review_cache_source))
    selected = _stratified_cameras(masks, 12)
    if not selected:
        raise RuntimeError("no cached review masks found")
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    parity_rows = []
    benchmark_rows = []
    cache = AlphaEvidenceCache(
        args.alpha_cache_dir, mode="readwrite",
        gaussian_identity=gaussian_render_sha256(model),
    )
    for camera_index in selected:
        camera_masks = masks[camera_index]
        if len(camera_masks) > 32:
            camera_masks = camera_masks[:32]
        camera = cameras[camera_index]
        reference, reference_seconds = _timed(lambda: render_gradient_reference(
            camera, model, args, background, camera_masks, config=config,
        ))
        fused, fused_seconds = _timed(lambda: render_fused(
            camera, model, args, camera_masks, config=config,
        ))
        visible_error = np.abs(reference.visible_mass - fused.visible_mass)
        inside_error = np.abs(reference.inside_mass - fused.inside_mass)
        tolerance_visible = 5e-5 * np.maximum(reference.visible_mass, 1.0)
        tolerance_inside = 5e-5 * np.maximum(reference.visible_mass[None, :], 1.0)
        reference_soft = (reference.inside_mass >= config.alpha_inside_mass_min) & (
            np.divide(reference.inside_mass, reference.visible_mass[None], out=np.zeros_like(reference.inside_mass), where=reference.visible_mass[None] > 0)
            >= config.alpha_inside_ratio_min
        )
        fused_soft = (fused.inside_mass >= config.alpha_inside_mass_min) & (
            np.divide(fused.inside_mass, fused.visible_mass[None], out=np.zeros_like(fused.inside_mass), where=fused.visible_mass[None] > 0)
            >= config.alpha_inside_ratio_min
        )
        parity_rows.append({
            "camera_index": camera_index, "mask_count": len(camera_masks),
            "visible_max_abs": float(visible_error.max(initial=0)),
            "inside_max_abs": float(inside_error.max(initial=0)),
            "within_tolerance": bool(np.all(visible_error <= tolerance_visible) and np.all(inside_error <= tolerance_inside)),
            "soft_support_exact": bool(np.array_equal(reference_soft, fused_soft)),
        })
        _, original_seconds = _timed(lambda: render_gradient_original(
            camera, model, args, background, camera_masks, config=config,
        ))
        _, cold_seconds = _timed(lambda: cache.get(
            camera_index, camera, model, args, background, camera_masks,
            backend="fused", config=config,
        ))
        warm, warm_seconds = _timed(lambda: cache.get(
            camera_index, camera, model, args, background, camera_masks,
            backend="fused", config=config,
        ))
        if not np.array_equal(warm.inside_mass, cache.get(
            camera_index, camera, model, args, background, camera_masks,
            backend="fused", config=config,
        ).inside_mass):
            raise RuntimeError("warm cache replay is not byte deterministic")
        benchmark_rows.append({
            "camera_index": camera_index, "mask_count": len(camera_masks),
            "original_seconds": original_seconds,
            "reference_seconds": reference_seconds,
            "fused_cold_seconds": fused_seconds,
            "cache_fill_seconds": cold_seconds,
            "warm_seconds": warm_seconds,
        })
    parity = {
        "schema": "saga-alpha-backend-parity-v1", "camera_indices": selected,
        "rows": parity_rows,
        "passed": all(row["within_tolerance"] and row["soft_support_exact"] for row in parity_rows),
    }
    original = sum(row["original_seconds"] for row in benchmark_rows)
    fused = sum(row["fused_cold_seconds"] for row in benchmark_rows)
    warm = sum(row["warm_seconds"] for row in benchmark_rows)
    benchmark = {
        "schema": "saga-alpha-backend-benchmark-v1", "rows": benchmark_rows,
        "cold_speedup": original / max(fused, 1e-12),
        "warm_speedup": original / max(warm, 1e-12),
        "passed": original / max(fused, 1e-12) >= 3 and original / max(warm, 1e-12) >= 20,
    }
    output = Path(args.output_dir)
    json_atomic(output / "alpha_backend_parity.json", parity)
    json_atomic(output / "alpha_backend_benchmark.json", benchmark)
    json_atomic(output / "alpha_cache_manifest.json", {
        "schema": "saga-alpha-cache-summary-v1", "root": str(Path(args.alpha_cache_dir).resolve()),
        "stats": cache.stats.__dict__, "camera_indices": selected,
    })
    if not parity["passed"]:
        raise RuntimeError("fused alpha backend failed numerical parity")
    if not benchmark["passed"]:
        raise RuntimeError("fused alpha backend failed performance gate")
    return {"parity": parity, "benchmark": benchmark}


__all__ = ["validate_alpha_backend"]
