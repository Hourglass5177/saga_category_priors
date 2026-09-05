from __future__ import annotations

"""Cloud-side numerical and performance validation for the fused backend."""

import json
import shutil
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


def _review_workloads(root: Path) -> dict[tuple[int, int], dict[str, Any]]:
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for path in sorted(root.glob("review_cache/round*/camera*/candidate*.npz")):
        round_index = int(path.parent.parent.name.removeprefix("round"))
        camera_index = int(path.parent.name.removeprefix("camera"))
        key = (round_index, camera_index)
        workload = result.setdefault(key, {"unique": {}, "candidate_groups": []})
        group: list[np.ndarray] = []
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            for row in metadata:
                packed = archive[row["packed_key"]]
                shape = tuple(row["mask_shape"])
                mask = np.unpackbits(packed)[: int(np.prod(shape))].reshape(shape).astype(bool)
                digest = __import__("hashlib").sha256(np.packbits(mask, bitorder="little").tobytes()).hexdigest()
                workload["unique"][digest] = mask
                group.append(mask)
        if group:
            workload["candidate_groups"].append(group)
    return {
        key: {"masks": list(row["unique"].values()), "candidate_groups": row["candidate_groups"]}
        for key, row in result.items() if row["unique"]
    }


def _stratified_workloads(
    workloads: dict[tuple[int, int], dict[str, Any]], limit: int = 12,
) -> list[tuple[int, int]]:
    rows = sorted((len(values["masks"]), key) for key, values in workloads.items())
    selected: list[tuple[int, int]] = []
    for target in (2, 4, 14, 22):
        if rows:
            _, key = min(rows, key=lambda row: (abs(row[0] - target), row[1]))
            if key not in selected:
                selected.append(key)
    if len(selected) < min(limit, len(rows)):
        # The production bottleneck is cameras with many masks. Keep the four
        # registered scale anchors, then benchmark the heaviest remaining work.
        for _, key in reversed(rows):
            if key not in selected:
                selected.append(key)
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


def _resource_snapshot(path: Path) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    usage = shutil.disk_usage(path)
    result["disk_free_bytes"] = int(usage.free)
    for name in ("memory.current", "memory.max", "memory.events"):
        source = Path("/sys/fs/cgroup") / name
        try:
            text = source.read_text(encoding="utf-8").strip()
            result[name.replace(".", "_")] = int(text) if "\n" not in text else text
        except (OSError, ValueError):
            result[name.replace(".", "_")] = None
    return result


def validate_alpha_backend(args: Any) -> dict[str, Any]:
    import torch
    from scene import GaussianModel

    config = RefinementConfig()
    model = GaussianModel(args.sh_degree)
    model.load_ply(args.point_cloud_path)
    cameras = load_cameras(args)
    workloads = _review_workloads(Path(args.review_cache_source))
    selected = _stratified_workloads(workloads, 12)
    if not selected:
        raise RuntimeError("no cached review masks found")
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    parity_rows = []
    benchmark_rows = []
    cache = AlphaEvidenceCache(
        args.alpha_cache_dir, mode="readwrite",
        gaussian_identity=gaussian_render_sha256(model),
    )
    torch.cuda.reset_peak_memory_stats()
    for round_index, camera_index in selected:
        workload = workloads[(round_index, camera_index)]
        camera_masks = workload["masks"]
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
            "round_index": round_index, "camera_index": camera_index,
            "mask_count": len(camera_masks),
            "historical_candidate_calls": len(workload["candidate_groups"]),
            "visible_max_abs": float(visible_error.max(initial=0)),
            "inside_max_abs": float(inside_error.max(initial=0)),
            "within_tolerance": bool(np.all(visible_error <= tolerance_visible) and np.all(inside_error <= tolerance_inside)),
            "soft_support_exact": bool(np.array_equal(reference_soft, fused_soft)),
            "near_threshold_fraction": float(np.mean(
                (np.abs(fused.inside_mass - config.alpha_inside_mass_min) <= 5e-5)
                | ((fused.visible_mass[None, :] > 0) & (
                    np.abs(np.divide(
                        fused.inside_mass, fused.visible_mass[None],
                        out=np.zeros_like(fused.inside_mass), where=fused.visible_mass[None] > 0,
                    ) - config.alpha_inside_ratio_min) <= 5e-5
                ))
            )),
        })
        original_seconds = 0.0
        for group in workload["candidate_groups"]:
            _, seconds = _timed(lambda group=group: render_gradient_original(
                camera, model, args, background, group, config=config,
            ))
            original_seconds += seconds
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
            "round_index": round_index, "camera_index": camera_index,
            "mask_count": len(camera_masks),
            "historical_candidate_calls": len(workload["candidate_groups"]),
            "original_seconds": original_seconds,
            "reference_seconds": reference_seconds,
            "fused_cold_seconds": fused_seconds,
            "cache_fill_seconds": cold_seconds,
            "warm_seconds": warm_seconds,
        })
    parity = {
        "schema": "saga-alpha-backend-parity-v2",
        "workloads": [{"round_index": row[0], "camera_index": row[1]} for row in selected],
        "rows": parity_rows,
        "passed": all(row["within_tolerance"] and row["soft_support_exact"] for row in parity_rows),
    }
    original = sum(row["original_seconds"] for row in benchmark_rows)
    fused = sum(row["fused_cold_seconds"] for row in benchmark_rows)
    warm = sum(row["warm_seconds"] for row in benchmark_rows)
    resources = _resource_snapshot(Path(args.alpha_cache_dir))
    cold_speedup = original / max(fused, 1e-12)
    warm_speedup = original / max(warm, 1e-12)
    projected_scene_hours = (fused / max(len(benchmark_rows), 1)) * len(workloads) / 3600.0
    benchmark = {
        "schema": "saga-alpha-backend-benchmark-v2", "rows": benchmark_rows,
        "cold_speedup": cold_speedup,
        "warm_speedup": warm_speedup,
        "projected_scene_hours": projected_scene_hours,
        "resources": resources,
        "passed": (
            cold_speedup >= 3 and warm_speedup >= 20
            and projected_scene_hours <= 4
            and resources["cuda_peak_reserved_bytes"] < 28 * 1024**3
            and (resources["memory_current"] is None or resources["memory_current"] < 80 * 1024**3)
            and resources["disk_free_bytes"] >= 80 * 1024**3
        ),
    }
    output = Path(args.output_dir)
    json_atomic(output / "alpha_backend_parity.json", parity)
    json_atomic(output / "alpha_backend_benchmark.json", benchmark)
    json_atomic(output / "alpha_cache_manifest.json", {
        "schema": "saga-alpha-cache-summary-v1", "root": str(Path(args.alpha_cache_dir).resolve()),
        "stats": cache.stats.__dict__,
        "workloads": [{"round_index": row[0], "camera_index": row[1]} for row in selected],
    })
    if not parity["passed"]:
        raise RuntimeError("fused alpha backend failed numerical parity")
    if not benchmark["passed"]:
        raise RuntimeError("fused alpha backend failed performance gate")
    return {"parity": parity, "benchmark": benchmark}


__all__ = ["validate_alpha_backend"]
