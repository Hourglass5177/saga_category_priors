from __future__ import annotations

"""Isolated SAM segment-everything extraction for the V8 experiment."""

import json
import argparse
import time
from pathlib import Path
from typing import Any, Sequence

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


def _mask_file_is_complete(path: Path, height: int, width: int) -> bool:
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
        stored_height == int(height)
        and stored_width == int(width)
        and count >= 0
        and packed.shape == (count, (height * width + 7) // 8)
    )


def _save_packed_masks(path: Path, masks: np.ndarray) -> None:
    array = np.asarray(masks, dtype=np.bool_)
    if array.ndim != 3:
        raise ValueError("SAM masks must be MxHxW")
    count, height, width = array.shape
    packed = np.packbits(array.reshape(count, height * width), axis=1)
    np.savez_compressed(
        path,
        packed=packed,
        count=np.asarray(count, dtype=np.int32),
        height=np.asarray(height, dtype=np.int32),
        width=np.asarray(width, dtype=np.int32),
    )


def extract_segment_everything(
    image_root: str | Path,
    output_root: str | Path,
    sam_checkpoint: str | Path,
    *,
    sam_arch: str = "vit_h",
    device: str = "cuda",
    image_extensions: Sequence[str] = (".jpg", ".jpeg", ".png"),
) -> dict[str, Any]:
    """Extract class-agnostic masks without modifying Grounded-SAM assets."""
    import cv2
    import torch
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    images = Path(image_root).resolve()
    output = Path(output_root).resolve()
    checkpoint = Path(sam_checkpoint).resolve()
    if not images.is_dir():
        raise FileNotFoundError(f"image directory not found: {images}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")
    output.mkdir(parents=True, exist_ok=True)

    model = sam_model_registry[sam_arch](checkpoint=str(checkpoint)).to(device)
    generator = SamAutomaticMaskGenerator(model=model, **SAM_EVERYTHING_CONFIG)
    suffixes = {value.lower() for value in image_extensions}
    image_paths = sorted(
        path for path in images.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )
    if not image_paths:
        raise ValueError(f"no images found under {images}")

    started = time.monotonic()
    records: list[dict[str, Any]] = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to decode image: {image_path}")
        # OpenCV decodes BGR while SAM's public predictor API expects RGB.
        # Keeping BGR here changes the mask source itself and would confound
        # the registered G/S factor.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        target = output / f"{image_path.stem}.npz"
        if _mask_file_is_complete(target, height, width):
            with np.load(target, allow_pickle=False) as payload:
                mask_count = int(np.asarray(payload["count"]).item())
            records.append({
                "image": image_path.name,
                "height": height,
                "width": width,
                "mask_count": mask_count,
                "status": "reused",
            })
            continue
        generated = generator.generate(image)
        mask_rows = [
            torch.from_numpy(np.asarray(row["segmentation"], dtype=np.bool_))
            for row in generated
        ]
        masks = (
            torch.stack(mask_rows, dim=0).bool().numpy()
            if mask_rows else np.zeros((0, height, width), dtype=np.bool_)
        )
        _save_packed_masks(target, masks)
        if not _mask_file_is_complete(target, height, width):
            raise RuntimeError(f"invalid SAM-everything output: {target}")
        records.append({
            "image": image_path.name,
            "height": height,
            "width": width,
            "mask_count": int(len(masks)),
            "status": "completed",
        })

    summary = {
        "schema": "saga-v8-segment-everything-v1",
        "image_root": str(images),
        "output_root": str(output),
        "sam_arch": sam_arch,
        "config": dict(SAM_EVERYTHING_CONFIG),
        "image_count": len(records),
        "mask_count": int(sum(row["mask_count"] for row in records)),
        "storage_bytes": int(sum(
            (output / f"{Path(row['image']).stem}.npz").stat().st_size
            for row in records
        )),
        "runtime_seconds": float(time.monotonic() - started),
        "images": records,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--sam-arch", default="vit_h")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    summary = extract_segment_everything(
        args.image_root,
        args.output_root,
        args.sam_checkpoint,
        sam_arch=args.sam_arch,
        device=args.device,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
