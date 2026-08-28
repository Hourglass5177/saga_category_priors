from __future__ import annotations

"""Sequential runner for the all-category denoising experiment.

The runner deliberately has no schedule, lock, hash, cache, or run-record
layer.  It reads the existing scene runtime manifest, invokes the repository's
``run_pipeline.sh --stage postprocess`` once at a time, and resumes only from
artifacts that can be parsed and satisfy the exported prediction contract.
"""

import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .category_denoise import load_candidate_bank
from .io import load_json, write_json
from .prediction_contract import validate_prediction_contract
from .runner import load_scene_runtime_manifest

_SCENE_PATH_OPTIONS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("images_path",), "--images-path"),
    (("sparse_path",), "--sparse-path"),
    (("point_cloud_path", "gaussian_ply"), "--point-cloud-path"),
    (("masks_path",), "--masks-path"),
    (("grounded_labels_path", "labels_path"), "--labels-path"),
    (("label_features_path",), "--label-features-path"),
    (("mask_scales_path",), "--mask-scales-path"),
    (
        (
            "contrastive_feature_point_cloud_path",
            "feature_point_cloud_path",
            "feature_ply_path",
        ),
        "--contrastive-feature-point-cloud-path",
    ),
    (("scale_gate_path",), "--scale-gate-path"),
)


def _normalize_scene_ids(
    scene_ids: str | Sequence[str] | None,
    scenes: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    if scene_ids is None:
        selected = tuple(sorted(scenes))
    elif isinstance(scene_ids, str):
        selected = (scene_ids,)
    else:
        selected = tuple(map(str, scene_ids))
    if not selected:
        raise ValueError("at least one scene is required")
    if len(selected) != len(set(selected)):
        raise ValueError("scene_ids contains duplicates")
    invalid = [scene_id for scene_id in selected if Path(scene_id).name != scene_id]
    if invalid:
        raise ValueError(f"invalid scene IDs: {invalid}")
    missing = sorted(set(selected).difference(scenes))
    if missing:
        raise ValueError(f"runtime manifest is missing scenes: {missing}")
    return selected


def _normalize_modes(mode: str | Sequence[str]) -> tuple[str, ...]:
    modes = (mode,) if isinstance(mode, str) else tuple(map(str, mode))
    if not modes:
        raise ValueError("at least one replay mode is required")
    if len(modes) != len(set(modes)):
        raise ValueError("replay modes contain duplicates")
    unknown = sorted(set(modes).difference({"uniform", "class"}))
    if unknown:
        raise ValueError(f"unsupported category-denoise modes: {unknown}")
    return tuple(value for value in ("uniform", "class") if value in modes)


def _resolved_scene_path(
    scene: Mapping[str, Any], keys: Sequence[str]
) -> Path | None:
    value: Any = None
    for key in keys:
        if scene.get(key) not in {None, ""}:
            value = scene[key]
            break
    if value is None:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = Path(str(scene["base_path"])) / path
    return path.resolve()


def _valid_prediction(path: Path) -> bool:
    try:
        payload = load_json(path)
        if not isinstance(payload, Mapping):
            return False
        labels = payload.get("point_labels")
        instances = payload.get("instances")
        if not isinstance(labels, list) or not isinstance(instances, Mapping):
            return False
        validate_prediction_contract(np.asarray(labels), instances)
        return True
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _valid_candidate_bank(
    scene_root: Path,
    *,
    scene_id: str | None = None,
    seed: int | None = None,
) -> bool:
    try:
        bank = load_candidate_bank(scene_root)
        sidecar = load_json(scene_root / "candidates.json")
    except (OSError, ValueError, TypeError, KeyError, EOFError):
        return False
    if not isinstance(sidecar, Mapping):
        return False
    required = {
        "schema",
        "point_count",
        "class_names",
        "saga20_names",
        "scene_scale_m_per_unit",
        "seed",
        "candidates",
        "diagnostics",
    }
    if not required.issubset(sidecar):
        return False
    valid = (
        sidecar["schema"] == bank.schema
        and int(sidecar["point_count"]) == bank.point_count
        and tuple(map(str, sidecar["class_names"])) == bank.class_names
        and tuple(map(str, sidecar["saga20_names"])) == bank.saga20_names
        and float(sidecar["scene_scale_m_per_unit"])
        == bank.scene_scale_m_per_unit
        and int(sidecar["seed"]) == bank.seed
        and sidecar["candidates"] == [dict(row) for row in bank.candidates]
        and sidecar["diagnostics"] == bank.diagnostics
    )
    if not valid:
        return False
    if seed is not None and bank.seed != int(seed):
        return False
    return scene_id is None or bank.diagnostics.get("scene_id") == scene_id


def _bank_complete(
    scene_root: Path,
    *,
    scene_id: str | None = None,
    seed: int | None = None,
) -> bool:
    return (
        _valid_prediction(scene_root / "output.json")
        and _valid_candidate_bank(scene_root, scene_id=scene_id, seed=seed)
        and (scene_root / "bank.log").is_file()
    )


def _replay_complete(
    scene_root: Path,
    *,
    scene_id: str | None = None,
    mode: str | None = None,
) -> bool:
    try:
        diagnostics = load_json(scene_root / "diagnostics.json")
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    if not (
        _valid_prediction(scene_root / "output.json")
        and isinstance(diagnostics, Mapping)
        and (scene_root / "postprocess.log").is_file()
    ):
        return False
    identity = diagnostics.get("category_denoise", diagnostics)
    if not isinstance(identity, Mapping):
        return False
    if scene_id is not None and identity.get("scene_id") != scene_id:
        return False
    return mode is None or identity.get("mode") == mode


def _b0_complete(scene_root: Path) -> bool:
    return _valid_prediction(scene_root / "output.json") and (
        scene_root / "postprocess.log"
    ).is_file()


def _build_command(
    *,
    pipeline_path: Path,
    priors_path: Path,
    scene_id: str,
    scene: Mapping[str, Any],
    output_path: Path,
    progress_path: Path,
    diagnostics_path: Path,
    bank_path: Path,
    action: str,
    mode: str,
    seed: int,
) -> list[str]:
    command = [
        "bash",
        str(pipeline_path),
        "--stage",
        "postprocess",
        "--base-path",
        str(Path(str(scene["base_path"])).resolve()),
        "--json-path",
        str(output_path),
        "--progress-path",
        str(progress_path),
        "--prior-metadata-path",
        str(diagnostics_path),
        "--scene-scale-m-per-unit",
        str(float(scene["scene_scale_m_per_unit"])),
        "--seed",
        str(int(seed)),
        "--prior-mode",
        "off",
        "--clustering-mode",
        "legacy",
        "--teacher-prior-mode",
        "off",
        "--disable-other-classes",
        "--minimal-metadata",
        "--category-denoise-action",
        action,
        "--category-denoise-bank-path",
        str(bank_path),
        "--category-denoise-mode",
        mode,
        "--category-denoise-scene-id",
        scene_id,
        "--category-priors",
        str(priors_path),
    ]
    python_bin = scene.get("python_bin")
    if python_bin:
        command.extend(("--python", str(Path(str(python_bin)).resolve())))
    for keys, option in _SCENE_PATH_OPTIONS:
        path = _resolved_scene_path(scene, keys)
        if path is not None:
            command.extend((option, str(path)))
    return command


def _run_command(command: Sequence[str], *, cwd: Path, log_path: Path) -> int:
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "42"
    environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(completed.returncode)


def _validate_common_inputs(
    runtime_manifest: str | Path,
    repo_root: str | Path,
    category_priors: str | Path,
    output_root: str | Path,
    seed: int,
) -> tuple[dict[str, dict[str, Any]], Path, Path, Path]:
    if isinstance(seed, bool) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    repository = Path(repo_root).resolve()
    pipeline = repository / "run_pipeline.sh"
    priors = Path(category_priors).resolve()
    if not pipeline.is_file():
        raise FileNotFoundError(pipeline)
    if not priors.is_file():
        raise FileNotFoundError(priors)
    scenes = load_scene_runtime_manifest(runtime_manifest)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return scenes, pipeline, priors, root


def run_category_denoise_bank(
    runtime_manifest: str | Path,
    output_root: str | Path,
    repo_root: str | Path,
    category_priors: str | Path,
    scene_ids: str | Sequence[str] | None = None,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate one shared candidate bank and a B0 output per selected scene."""

    scenes, pipeline, priors, root = _validate_common_inputs(
        runtime_manifest, repo_root, category_priors, output_root, seed
    )
    from .category_denoise import materialize_category_denoise_params
    from .teacher_prior import SAGA20_CLASSES

    write_json(
        root / "category_denoise_params.json",
        materialize_category_denoise_params(load_json(priors), SAGA20_CLASSES),
    )
    selected = _normalize_scene_ids(scene_ids, scenes)
    rows: list[dict[str, Any]] = []
    for scene_id in selected:
        scene_root = root / "bank" / scene_id
        scene_root.mkdir(parents=True, exist_ok=True)
        if _bank_complete(scene_root, scene_id=scene_id, seed=seed):
            rows.append(
                {"scene_id": scene_id, "status": "skipped_complete", "root": str(scene_root)}
            )
            continue
        command = _build_command(
            pipeline_path=pipeline,
            priors_path=priors,
            scene_id=scene_id,
            scene=scenes[scene_id],
            output_path=scene_root / "output.json",
            progress_path=scene_root / "progress.txt",
            diagnostics_path=scene_root / "diagnostics.json",
            bank_path=scene_root,
            action="bank",
            mode="uniform",
            seed=seed,
        )
        return_code = _run_command(command, cwd=pipeline.parent, log_path=scene_root / "bank.log")
        if return_code != 0 or not _bank_complete(
            scene_root, scene_id=scene_id, seed=seed
        ):
            raise RuntimeError(
                f"category-denoise bank failed for {scene_id}; "
                f"inspect {scene_root / 'bank.log'}"
            )
        rows.append({"scene_id": scene_id, "status": "complete", "root": str(scene_root)})
    return {
        "action": "bank",
        "seed": int(seed),
        "total": len(rows),
        "complete": len(rows),
        "runs": rows,
    }


def replay_category_denoise(
    runtime_manifest: str | Path,
    bank_root: str | Path,
    output_root: str | Path,
    repo_root: str | Path,
    category_priors: str | Path,
    scene_ids: str | Sequence[str] | None = None,
    mode: str | Sequence[str] = ("uniform", "class"),
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Replay uniform and/or class statistics over immutable scene banks."""

    scenes, pipeline, priors, root = _validate_common_inputs(
        runtime_manifest, repo_root, category_priors, output_root, seed
    )
    selected = _normalize_scene_ids(scene_ids, scenes)
    modes = _normalize_modes(mode)
    bank_base = Path(bank_root).resolve()
    if not bank_base.is_dir():
        raise FileNotFoundError(bank_base)
    rows: list[dict[str, Any]] = []
    for scene_id in selected:
        scene_bank = bank_base / scene_id
        if not _bank_complete(scene_bank, scene_id=scene_id, seed=seed):
            raise ValueError(
                f"category-denoise bank is missing or corrupt for {scene_id}: {scene_bank}"
            )
        for replay_mode in modes:
            scene_root = root / replay_mode / scene_id
            scene_root.mkdir(parents=True, exist_ok=True)
            if _replay_complete(
                scene_root, scene_id=scene_id, mode=replay_mode
            ):
                rows.append(
                    {
                        "scene_id": scene_id,
                        "mode": replay_mode,
                        "status": "skipped_complete",
                        "root": str(scene_root),
                    }
                )
                continue
            command = _build_command(
                pipeline_path=pipeline,
                priors_path=priors,
                scene_id=scene_id,
                scene=scenes[scene_id],
                output_path=scene_root / "output.json",
                progress_path=scene_root / "progress.txt",
                diagnostics_path=scene_root / "diagnostics.json",
                bank_path=scene_bank,
                action="replay",
                mode=replay_mode,
                seed=seed,
            )
            return_code = _run_command(
                command, cwd=pipeline.parent, log_path=scene_root / "postprocess.log"
            )
            if return_code != 0 or not _replay_complete(
                scene_root, scene_id=scene_id, mode=replay_mode
            ):
                raise RuntimeError(
                    f"category-denoise replay failed for {replay_mode}/{scene_id}; "
                    f"inspect {scene_root / 'postprocess.log'}"
                )
            rows.append(
                {
                    "scene_id": scene_id,
                    "mode": replay_mode,
                    "status": "complete",
                    "root": str(scene_root),
                }
            )
    return {
        "action": "replay",
        "seed": int(seed),
        "total": len(rows),
        "complete": len(rows),
        "runs": rows,
    }


def run_category_denoise_b0_control(
    runtime_manifest: str | Path,
    output_root: str | Path,
    repo_root: str | Path,
    category_priors: str | Path,
    scene_ids: str | Sequence[str] | None = None,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the category-denoising code path fully disabled for DEV2 parity.

    This is intentionally an internal experiment control rather than another
    public method.  It invokes the same legacy B0 command as bank generation,
    but without constructing a bank, so the two exported predictions can be
    compared point-for-point before any U/D result is interpreted.
    """

    scenes, pipeline, priors, root = _validate_common_inputs(
        runtime_manifest, repo_root, category_priors, output_root, seed
    )
    selected = _normalize_scene_ids(scene_ids, scenes)
    rows: list[dict[str, Any]] = []
    for scene_id in selected:
        scene_root = root / "b0-off" / scene_id
        scene_root.mkdir(parents=True, exist_ok=True)
        if _b0_complete(scene_root):
            rows.append(
                {
                    "scene_id": scene_id,
                    "status": "skipped_complete",
                    "root": str(scene_root),
                }
            )
            continue
        command = _build_command(
            pipeline_path=pipeline,
            priors_path=priors,
            scene_id=scene_id,
            scene=scenes[scene_id],
            output_path=scene_root / "output.json",
            progress_path=scene_root / "progress.txt",
            diagnostics_path=scene_root / "diagnostics.json",
            bank_path=scene_root / "unused-bank",
            action="off",
            mode="uniform",
            seed=seed,
        )
        return_code = _run_command(
            command, cwd=pipeline.parent, log_path=scene_root / "postprocess.log"
        )
        if return_code != 0 or not _b0_complete(scene_root):
            raise RuntimeError(
                f"category-denoise disabled B0 failed for {scene_id}; "
                f"inspect {scene_root / 'postprocess.log'}"
            )
        rows.append(
            {"scene_id": scene_id, "status": "complete", "root": str(scene_root)}
        )
    return {
        "action": "b0-off",
        "seed": int(seed),
        "total": len(rows),
        "complete": len(rows),
        "runs": rows,
    }


__all__ = [
    "replay_category_denoise",
    "run_category_denoise_b0_control",
    "run_category_denoise_bank",
]
