from __future__ import annotations

import argparse
import colorsys
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


_SH_C0 = 0.28209479177387814
_SEMANTIC_COLORS = np.asarray(
    [
        (166, 206, 227), (31, 120, 180), (178, 223, 138), (51, 160, 44),
        (251, 154, 153), (227, 26, 28), (253, 191, 111), (255, 127, 0),
        (202, 178, 214), (106, 61, 154), (255, 255, 153), (177, 89, 40),
        (141, 211, 199), (255, 255, 179), (190, 186, 218), (251, 128, 114),
        (128, 177, 211), (253, 180, 98), (179, 222, 105), (252, 205, 229),
    ],
    dtype=np.uint8,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_indices(count: int, limit: int) -> np.ndarray:
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, limit, dtype=np.int64)


def _instance_colors(labels: np.ndarray) -> np.ndarray:
    colors = np.full((len(labels), 3), 96, dtype=np.uint8)
    for label in np.unique(labels):
        if label < 0:
            continue
        hue = (int(label) * 0.6180339887498949) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.72, 0.95)
        colors[labels == label] = np.rint(np.asarray(rgb) * 255).astype(np.uint8)
    return colors


def _semantic_colors(labels: np.ndarray) -> np.ndarray:
    colors = np.full((len(labels), 3), 96, dtype=np.uint8)
    valid = (labels >= 0) & (labels < len(_SEMANTIC_COLORS))
    colors[valid] = _SEMANTIC_COLORS[labels[valid]]
    return colors


def _write_colored_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    from plyfile import PlyData, PlyElement

    vertices = np.empty(
        len(xyz),
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = xyz.T
    vertices["red"], vertices["green"], vertices["blue"] = rgb.T
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


def _prediction_labels(
    output: dict[str, Any], canonical_classes: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    instances = np.asarray(output["point_labels"], dtype=np.int64)
    semantics = np.full(len(instances), -1, dtype=np.int64)
    class_ids = {name: index for index, name in enumerate(canonical_classes)}
    for raw_id, payload in output.get("instances", {}).items():
        class_id = class_ids.get(str(payload.get("class")))
        if class_id is not None:
            semantics[instances == int(raw_id)] = class_id
    return instances, semantics


def build_viewer_materials(
    analysis_path: Path,
    runtime_manifest_path: Path,
    gt_dir: Path,
    runs_root: Path,
    taxonomy_path: Path,
    output_dir: Path,
    seed: int = 42,
    max_points: int = 300_000,
) -> dict[str, Any]:
    from plyfile import PlyData

    analysis = _read_json(analysis_path)
    runtime = _read_json(runtime_manifest_path)
    taxonomy = _read_json(taxonomy_path)
    classes = list(taxonomy["canonical_classes"])
    scene_paths = {
        item["scene_id"]: Path(item["base_path"])
        for item in runtime["scenes"]
    }
    selected = analysis["qualitative_cases"]
    cases = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for role in ("best", "median", "worst"):
        case = selected[role]
        scene_id = str(case["scene_id"])
        base = scene_paths[scene_id]
        case_dir = output_dir / role
        case_dir.mkdir(parents=True, exist_ok=True)

        gaussian_path = (
            base / "output_models/point_cloud/iteration_30000/scene_point_cloud.ply"
        )
        vertices = PlyData.read(gaussian_path)["vertex"].data
        gaussian_xyz = np.column_stack(
            (vertices["x"], vertices["y"], vertices["z"])
        ).astype(np.float32, copy=False)
        gaussian_rgb = np.clip(
            0.5
            + _SH_C0
            * np.column_stack(
                (vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"])
            ),
            0.0,
            1.0,
        )
        gaussian_rgb = np.rint(gaussian_rgb * 255).astype(np.uint8)
        gaussian_sample = _sample_indices(len(gaussian_xyz), max_points)
        _write_colored_ply(
            case_dir / "rgb_gaussians.ply",
            gaussian_xyz[gaussian_sample],
            gaussian_rgb[gaussian_sample],
        )

        gt = np.load(gt_dir / f"{scene_id}.npz")
        gt_xyz = np.asarray(gt["coords"], dtype=np.float32)
        gt_instances = np.asarray(gt["instance"], dtype=np.int64)
        gt_semantics = np.asarray(gt["semantic"], dtype=np.int64)
        gt_sample = _sample_indices(len(gt_xyz), max_points)
        _write_colored_ply(
            case_dir / "gt_instances.ply",
            gt_xyz[gt_sample],
            _instance_colors(gt_instances[gt_sample]),
        )
        _write_colored_ply(
            case_dir / "gt_semantics.ply",
            gt_xyz[gt_sample],
            _semantic_colors(gt_semantics[gt_sample]),
        )

        condition_files: dict[str, dict[str, str]] = {}
        for short_name, condition in (
            ("p000", "P000-B2"),
            ("p111", "P111-combined"),
        ):
            prediction_path = runs_root / condition / scene_id / f"seed-{seed}" / "output.json"
            prediction = _read_json(prediction_path)
            instance_labels, semantic_labels = _prediction_labels(prediction, classes)
            if len(instance_labels) != len(gaussian_xyz):
                raise ValueError(
                    f"{scene_id}/{condition}: labels and Gaussian points differ"
                )
            _write_colored_ply(
                case_dir / f"{short_name}_instances.ply",
                gaussian_xyz[gaussian_sample],
                _instance_colors(instance_labels[gaussian_sample]),
            )
            _write_colored_ply(
                case_dir / f"{short_name}_semantics.ply",
                gaussian_xyz[gaussian_sample],
                _semantic_colors(semantic_labels[gaussian_sample]),
            )
            condition_files[condition] = {
                "source_output": str(prediction_path),
                "instances": f"{short_name}_instances.ply",
                "semantics": f"{short_name}_semantics.ply",
            }

        rgb_sources = sorted((base / "fastRecon/dense/sparse/0/images").glob("*.jpg"))
        rgb_reference = None
        if rgb_sources:
            rgb_reference = "rgb_reference.jpg"
            shutil.copy2(rgb_sources[len(rgb_sources) // 2], case_dir / rgb_reference)

        cases.append(
            {
                "role": role,
                "scene_id": scene_id,
                "paired_delta_map_50_95": float(case["delta_map_50_95"]),
                "seed_for_visualization": seed,
                "rgb_reference": rgb_reference,
                "rgb_point_cloud": "rgb_gaussians.ply",
                "ground_truth": {
                    "instances": "gt_instances.ply",
                    "semantics": "gt_semantics.ply",
                },
                "predictions": condition_files,
                "sampled_gaussian_points": int(len(gaussian_sample)),
                "sampled_gt_points": int(len(gt_sample)),
            }
        )

    payload = {
        "kind": "locked_viewer_materials",
        "split": "val-locked",
        "selection_rule": "best_median_worst_by_preregistered_paired_scene_delta",
        "comparison": ["P000-B2", "P111-combined"],
        "cases": cases,
        "qualitative_only": True,
        "not_for_parameter_selection": True,
    }
    (output_dir / "viewer_case_selection.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# Locked qualitative viewer materials\n\n"
        "Open the PLY files in the Windows viewer using the same camera for each "
        "case. Compare RGB, GT, P000-B2 and P111-combined instance/semantic views. "
        "The cases were selected only after all 1,728 runs completed and are "
        "qualitative illustrations, not inputs to parameter selection.\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-points", type=int, default=300_000)
    args = parser.parse_args()
    build_viewer_materials(
        args.analysis,
        args.scene_manifest,
        args.gt_dir,
        args.runs_root,
        args.taxonomy,
        args.output_dir,
        args.seed,
        args.max_points,
    )


if __name__ == "__main__":
    main()
