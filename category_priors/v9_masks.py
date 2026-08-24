from __future__ import annotations

"""Native, isolated SAM-everything extraction for the V9 ObjectBank."""

import argparse
import json
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


def mask_file_is_complete(path: Path, height: int, width: int) -> bool:
    try:
        with np.load(path, allow_pickle=False) as payload:
            packed = np.asarray(payload["packed"], dtype=np.uint8)
            count = int(np.asarray(payload["count"]).item())
            stored = (
                int(np.asarray(payload["height"]).item()),
                int(np.asarray(payload["width"]).item()),
            )
        return (
            stored == (height, width)
            and count >= 0
            and packed.shape == (count, (height * width + 7) // 8)
        )
    except (OSError, ValueError, KeyError, EOFError):
        return False


def sam_directory_is_complete(directory: Path, image_root: Path) -> bool:
    images = sorted(
        path for path in image_root.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        return False
    try:
        import cv2

        for image in images:
            decoded = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if decoded is None or not mask_file_is_complete(
                directory / f"{image.stem}.npz", *decoded.shape[:2]
            ):
                return False
        return True
    except (OSError, ValueError):
        return False


def extract_segment_everything(
    image_root: Path,
    output_root: Path,
    sam_checkpoint: Path,
    *,
    sam_arch: str = "vit_h",
    device: str = "cuda",
) -> dict[str, Any]:
    import cv2
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    images = Path(image_root).resolve()
    output = Path(output_root).resolve()
    checkpoint = Path(sam_checkpoint).resolve()
    if not images.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError("V9 SAM image root/checkpoint does not exist")
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        path for path in images.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not paths:
        raise ValueError("V9 SAM image root is empty")
    generator = SamAutomaticMaskGenerator(
        model=sam_model_registry[sam_arch](checkpoint=str(checkpoint)).to(device),
        **SAM_EVERYTHING_CONFIG,
    )
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot decode {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        target = output / f"{path.stem}.npz"
        if mask_file_is_complete(target, height, width):
            with np.load(target, allow_pickle=False) as payload:
                count = int(np.asarray(payload["count"]).item())
            rows.append({
                "image": path.name,
                "height": height,
                "width": width,
                "mask_count": count,
                "status": "reused",
            })
            continue
        masks = np.asarray(
            [np.asarray(row["segmentation"], dtype=bool) for row in generator.generate(image)],
            dtype=bool,
        )
        if not len(masks):
            masks = np.zeros((0, height, width), dtype=bool)
        packed = np.packbits(masks.reshape(len(masks), height * width), axis=1)
        np.savez_compressed(
            target,
            packed=packed,
            count=np.asarray(len(masks), dtype=np.int32),
            height=np.asarray(height, dtype=np.int32),
            width=np.asarray(width, dtype=np.int32),
        )
        if not mask_file_is_complete(target, height, width):
            raise RuntimeError(f"invalid V9 SAM output {target}")
        rows.append({
            "image": path.name,
            "height": height,
            "width": width,
            "mask_count": len(masks),
            "status": "completed",
        })
    result = {
        "schema": "saga-v9-segment-everything-v1",
        "image_root": str(images),
        "output_root": str(output),
        "sam_arch": sam_arch,
        "config": SAM_EVERYTHING_CONFIG,
        "image_count": len(rows),
        "mask_count": sum(int(row["mask_count"]) for row in rows),
        "runtime_seconds": float(time.monotonic() - started),
        "images": rows,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), "utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(extract_segment_everything(
        args.image_root, args.output_root, args.sam_checkpoint
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
