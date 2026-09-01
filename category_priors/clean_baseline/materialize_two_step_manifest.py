from __future__ import annotations

"""Materialize the strict two-step manifest from frozen clean-baseline files."""

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..io import load_json, sha256_file, write_json
from ..taxonomy import load_taxonomy
from .evaluation import RUN_IDENTITY_SCHEMA, validate_embedded_identity
from .evaluation import (
    CLEAN_EVALUATION_SCHEMA,
    EVALUATION_IDENTITY_SCHEMA,
)
from .materialize_config import (
    EVIDENCE_IMPORT_MANIFEST_SCHEMA,
    _load_evidence_imports,
)
from .two_step_audit import (
    MANIFEST_SCHEMA,
    REGISTERED_CONDITIONS,
    REGISTERED_DEV2_SCENE_IDS,
    REGISTERED_DEV8_SCENE_IDS,
)
from .worker import resolve_clean_scene_inputs


DEFAULT_DEV2 = REGISTERED_DEV2_SCENE_IDS


def _object(path: str | Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _resolve_registered_path(value: Any, *, base: Path, name: str) -> Path:
    """Resolve one path relative to the manifest that registered it.

    Historical cloud manifests normally contain absolute paths, but resolving
    relative values against the current shell directory would make the same
    immutable manifest mean different things on another host.  Each producer
    manifest therefore owns the base directory for the paths it registers.
    """

    if value in (None, ""):
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(str(value))
    return (path if path.is_absolute() else base / path).resolve()


def _preflight_output_path(
    *,
    output_path: Path,
    input_files: Sequence[Path],
    immutable_directories: Sequence[Path],
) -> None:
    """Reject a manifest destination that could overwrite frozen evidence."""

    target = output_path.resolve()
    for value in input_files:
        source = value.resolve()
        if target == source or source.is_relative_to(target):
            raise ValueError("materialized manifest output overlaps an input file")
    for value in immutable_directories:
        source = value.resolve()
        if (
            target == source
            or target.is_relative_to(source)
            or source.is_relative_to(target)
        ):
            raise ValueError("materialized manifest output overlaps a frozen asset tree")


def materialize_two_step_manifest(
    *,
    legacy_evaluation_manifest: str | Path,
    evidence_import_manifest: str | Path,
    historical_evaluation: str | Path,
    size_bins: str | Path,
    output_path: str | Path,
    dev2_scene_ids: Sequence[str] = DEFAULT_DEV2,
    dev8_scene_ids: Sequence[str] = REGISTERED_DEV8_SCENE_IDS,
) -> dict[str, Any]:
    legacy_manifest_path = Path(legacy_evaluation_manifest).resolve()
    import_manifest_path = Path(evidence_import_manifest).resolve()
    legacy = _object(legacy_manifest_path)
    if legacy.get("kind") != "clean_baseline_evaluation_manifest":
        raise ValueError("legacy evaluation manifest kind mismatch")
    # Reuse the producer's strict import validator: schema, producer commit,
    # request identity and the exact three evidence-file digests are all
    # checked before any frozen bank can enter the two-step manifest.
    import_scenes = _load_evidence_imports(import_manifest_path)
    historical_path = Path(historical_evaluation).resolve()
    historical = _object(historical_path)
    if historical.get("schema") != CLEAN_EVALUATION_SCHEMA:
        raise ValueError("historical evaluation schema mismatch")
    metrics = historical.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("historical evaluation omitted metrics")
    size_payload = _object(size_bins)
    taxonomy = load_taxonomy()

    dev8 = tuple(map(str, dev8_scene_ids))
    if dev8 != REGISTERED_DEV8_SCENE_IDS:
        raise ValueError("DEV8 must exactly match the registered eight-scene order")
    raw_legacy_scenes = legacy.get("scenes")
    if not isinstance(raw_legacy_scenes, list):
        raise ValueError("legacy manifest scenes must be an explicit list")
    legacy_by_scene: dict[str, Mapping[str, Any]] = {}
    for source in raw_legacy_scenes:
        if not isinstance(source, Mapping):
            raise TypeError("legacy scene row must be a mapping")
        scene_id = str(source.get("scene_id", ""))
        if not scene_id or scene_id in legacy_by_scene:
            raise ValueError("legacy scene IDs must be non-empty and unique")
        legacy_by_scene[scene_id] = source
    if set(legacy_by_scene) != set(dev8):
        raise ValueError("legacy manifest must contain exactly the registered DEV8")

    expected: dict[str, dict[str, float]] = {}
    for condition in REGISTERED_CONDITIONS:
        row = metrics.get(condition)
        if not isinstance(row, Mapping) or not isinstance(row.get("aggregate"), Mapping):
            raise ValueError(f"historical evaluation omitted {condition}")
        aggregate = row["aggregate"]
        expected[condition] = {
            "official_ap25": float(aggregate["map_0.25"]),
            "official_ap50": float(aggregate["map_0.50"]),
            "historical_map_50_95": float(aggregate["map_50_95"]),
        }

    scene_rows: list[dict[str, Any]] = []
    for scene_id in dev8:
        source = legacy_by_scene[scene_id]
        imported = import_scenes.get(scene_id)
        if not isinstance(imported, Mapping):
            raise ValueError(f"evidence imports omitted {scene_id}")
        source_request_path = Path(imported["source_request"]).resolve()
        source_request = dict(imported["request"])
        outputs = source.get("outputs")
        if not isinstance(outputs, Mapping):
            raise ValueError(f"legacy outputs omitted {scene_id}")
        conditions: dict[str, Any] = {}
        for condition in REGISTERED_CONDITIONS:
            raw_output = outputs.get(condition)
            output_path_value = (
                raw_output.get("output_json")
                if isinstance(raw_output, Mapping)
                else raw_output
            )
            output = _resolve_registered_path(
                output_path_value,
                base=legacy_manifest_path.parent,
                name=f"legacy {scene_id}/{condition} output",
            )
            diagnostics = output.with_name("diagnostics.json")
            if not output.is_file() or not diagnostics.is_file():
                raise FileNotFoundError(output if not output.is_file() else diagnostics)
            output_payload = _object(output)
            diagnostics_payload = _object(diagnostics)
            output_identity = validate_embedded_identity(
                output_payload.get("run_identity"), expected_schema=RUN_IDENTITY_SCHEMA
            )
            diagnostic_identity = validate_embedded_identity(
                diagnostics_payload.get("run_identity"),
                expected_schema=RUN_IDENTITY_SCHEMA,
            )
            if output_identity != diagnostic_identity:
                raise ValueError(
                    f"legacy {scene_id}/{condition} output/diagnostics identities differ"
                )
            if (
                str(output_payload.get("scene_id")) != scene_id
                or str(diagnostics_payload.get("scene_id")) != scene_id
                or str(output_payload.get("condition")) != condition
                or str(diagnostics_payload.get("condition")) != condition
                or str(output_identity.get("scene_id")) != scene_id
                or str(output_identity.get("condition")) != condition
            ):
                raise ValueError(
                    f"legacy {scene_id}/{condition} artifact identity mismatch"
                )
            conditions[condition] = {
                "output": str(output),
                "diagnostics": str(diagnostics),
                "output_sha256": sha256_file(output),
                "diagnostics_sha256": sha256_file(diagnostics),
                "run_identity_sha256": str(output_identity["content_sha256"]),
                "consumer_commit": str(output_identity["consumer_commit"]),
            }
        scene_rows.append(
            {
                "scene_id": scene_id,
                "gt_npz": str(
                    _resolve_registered_path(
                        source["gt_npz"],
                        base=legacy_manifest_path.parent,
                        name=f"legacy {scene_id}.gt_npz",
                    )
                ),
                "gaussian_ply": str(
                    _resolve_registered_path(
                        source["gaussian_ply"],
                        base=legacy_manifest_path.parent,
                        name=f"legacy {scene_id}.gaussian_ply",
                    )
                ),
                "transform": source["gaussian_to_gt_transform"],
                "bank_dir": str(
                    Path(imported["bank_dir"]).resolve()
                ),
                "source_evidence_request": source_request,
                "evidence_import_identity": {
                    "schema": EVIDENCE_IMPORT_MANIFEST_SCHEMA,
                    "producer_commit": str(imported["producer_commit"]),
                    "source_request": str(source_request_path),
                    "bank_dir": str(Path(imported["bank_dir"]).resolve()),
                    "files": dict(imported["files"]),
                },
                "conditions": conditions,
            }
        )
    scene_ids = [str(row["scene_id"]) for row in scene_rows]
    if tuple(scene_ids) != dev8:
        raise AssertionError("materialized DEV8 order drifted")
    dev2 = tuple(map(str, dev2_scene_ids))
    if dev2 != REGISTERED_DEV2_SCENE_IDS:
        raise ValueError(
            "DEV2 scenes must exactly match the registered scene0645_00/"
            "scene0025_01 order"
        )

    if sorted(map(str, historical.get("scene_ids", ()))) != sorted(dev8):
        raise ValueError("historical evaluation scene set differs from DEV8")
    if tuple(map(str, historical.get("conditions", ()))) != REGISTERED_CONDITIONS:
        raise ValueError("historical evaluation conditions differ from C0/U")
    if float(historical.get("radius_m", -1.0)) != 0.05:
        raise ValueError("historical evaluation must use the registered 5 cm mapping")
    if int(historical.get("min_region_size", -1)) != 100:
        raise ValueError("historical evaluation must use min_region_size=100")
    if historical.get("oracle_class_in_formal_metrics") is not False:
        raise ValueError("historical evaluation cannot contain oracle classes")
    evaluation_identity = validate_embedded_identity(
        historical.get("evaluation_identity"),
        expected_schema=EVALUATION_IDENTITY_SCHEMA,
    )
    if tuple(map(str, evaluation_identity.get("class_names", ()))) != tuple(
        taxonomy.canonical_classes
    ):
        raise ValueError("historical evaluation taxonomy differs from SAGA20")
    if tuple(map(str, evaluation_identity.get("conditions", ()))) != REGISTERED_CONDITIONS:
        raise ValueError("historical evaluation identity conditions differ from C0/U")
    if float(evaluation_identity.get("radius_m", -1.0)) != 0.05:
        raise ValueError("historical evaluation identity radius differs from 5 cm")
    if int(evaluation_identity.get("min_region_size", -1)) != 100:
        raise ValueError("historical evaluation identity min_region_size differs")
    identity_inputs = evaluation_identity.get("inputs")
    if not isinstance(identity_inputs, Mapping) or set(map(str, identity_inputs)) != set(
        dev8
    ):
        raise ValueError("historical evaluation identity input scene set differs")
    for scene in scene_rows:
        scene_id = str(scene["scene_id"])
        registered = identity_inputs.get(scene_id)
        if not isinstance(registered, Mapping):
            raise ValueError(f"historical evaluation identity omitted {scene_id}")
        predictions = registered.get("predictions")
        if not isinstance(predictions, Mapping):
            raise ValueError(f"historical evaluation identity omitted {scene_id} predictions")
        for condition in REGISTERED_CONDITIONS:
            if str(predictions.get(condition, "")) != str(
                scene["conditions"][condition]["run_identity_sha256"]
            ):
                raise ValueError(
                    f"historical evaluation prediction identity differs for {scene_id}/{condition}"
                )
        if str(registered.get("gt_sha256", "")) != sha256_file(scene["gt_npz"]):
            raise ValueError(f"historical evaluation GT identity differs for {scene_id}")
        if str(registered.get("gaussian_sha256", "")) != sha256_file(
            scene["gaussian_ply"]
        ):
            raise ValueError(
                f"historical evaluation Gaussian identity differs for {scene_id}"
            )
        if registered.get("gaussian_to_gt_transform") != scene["transform"]:
            raise ValueError(
                f"historical evaluation transform identity differs for {scene_id}"
            )
    immutable_directories: set[Path] = set()
    input_files: set[Path] = {
        legacy_manifest_path,
        import_manifest_path,
        historical_path,
        Path(size_bins).resolve(),
    }
    for scene in scene_rows:
        immutable_directories.update(
            {
                Path(scene["bank_dir"]).resolve(),
                Path(scene["gt_npz"]).resolve().parent,
                Path(scene["gaussian_ply"]).resolve().parent,
            }
        )
        registration = scene["evidence_import_identity"]
        request_path = Path(registration["source_request"]).resolve()
        input_files.add(request_path)
        immutable_directories.add(request_path.parent)
        for condition in REGISTERED_CONDITIONS:
            immutable_directories.add(
                Path(scene["conditions"][condition]["output"]).resolve().parent
            )
        request = scene["source_evidence_request"]
        scene_value = request.get("scene", request.get("runtime_registration"))
        if isinstance(scene_value, Mapping) and scene_value.get("base_path"):
            inputs = resolve_clean_scene_inputs(scene_value, require_exists=True)
            immutable_directories.update(
                {
                    inputs.base_path,
                    inputs.rgb_ply.parent,
                    inputs.sparse,
                    inputs.images,
                    inputs.sam_masks,
                    inputs.grounded_masks,
                    inputs.grounded_labels,
                }
            )
        generation = request.get("sam_generation")
        if isinstance(generation, Mapping) and generation.get("checkpoint"):
            checkpoint = Path(str(generation["checkpoint"])).resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            input_files.add(checkpoint)
    _preflight_output_path(
        output_path=Path(output_path),
        input_files=tuple(input_files),
        immutable_directories=tuple(immutable_directories),
    )
    payload = {
        "schema": MANIFEST_SCHEMA,
        "dev8_scene_ids": list(dev8),
        "dev2_scene_ids": list(dev2),
        "taxonomy": {
            "class_names": list(taxonomy.canonical_classes),
            "allowed_classes": list(taxonomy.canonical_classes),
        },
        "size_bins": size_payload,
        "min_region_size": 100,
        "metric_tolerance": 1e-12,
        "expected_metrics": expected,
        "scenes": scene_rows,
    }
    write_json(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-evaluation-manifest", required=True)
    parser.add_argument("--evidence-import-manifest", required=True)
    parser.add_argument("--historical-evaluation", required=True)
    parser.add_argument("--size-bins", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    materialize_two_step_manifest(
        legacy_evaluation_manifest=args.legacy_evaluation_manifest,
        evidence_import_manifest=args.evidence_import_manifest,
        historical_evaluation=args.historical_evaluation,
        size_bins=args.size_bins,
        output_path=args.output,
        dev2_scene_ids=DEFAULT_DEV2,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["DEFAULT_DEV2", "materialize_two_step_manifest"]
