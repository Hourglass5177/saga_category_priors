from __future__ import annotations

"""End-to-end orchestration for the registered clean two-step experiment.

Step one is a strictly read-only DEV8 metric/funnel audit.  Step two generates
one metadata-rich SAM stack per DEV2 scene, deterministically derives the
paired hierarchy and flat mask contracts, and runs the same C0 consensus on
both.  Scientific scores never decide whether step two starts; only the
technical integrity gates from step one may stop the pipeline.
"""

import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..io import build_file_manifest, hash_json, load_json, sha256_file, write_json
from .evidence import build_alpha_mask_evidence, load_evidence_bank
from .mask_control import prepare_flat_mask_control_scene
from .pipeline import run_consensus_condition
from .worker import resolve_clean_scene_inputs
from .two_step_audit import (
    DEFAULT_DEV8_SCENE_COUNT,
    MANIFEST_SCHEMA,
    REGISTERED_DEV2_SCENE_IDS,
    REGISTERED_DEV8_SCENE_IDS,
    audit_clean_baseline_manifest,
    preflight_audit_output_directory,
)


STATUS_SCHEMA = "saga-clean-mask-contract-two-step-status-v1"
ANALYSIS_SCHEMA = "saga-clean-mask-contract-two-step-analysis-v1"
FLAT_REPEAT_SCHEMA = "saga-clean-flat-full-repeat-v1"
REGISTERED_ARMS = ("H-hierarchy", "P-flat")
_FULL_COMMIT = re.compile(r"[0-9a-f]{40}")


def _atomic_status(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path).resolve()
    payload = load_json(manifest_path)
    if not isinstance(payload, dict) or payload.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema must be {MANIFEST_SCHEMA}")
    scenes = payload.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("manifest must contain a non-empty scenes list")
    identities = [str(row.get("scene_id", "")) for row in scenes]
    if any(not value for value in identities) or len(set(identities)) != len(
        identities
    ):
        raise ValueError("manifest scene IDs must be non-empty and unique")
    if tuple(identities) != REGISTERED_DEV8_SCENE_IDS:
        raise ValueError("manifest scenes must exactly match the registered DEV8 order")
    if tuple(map(str, payload.get("dev8_scene_ids", ()))) != REGISTERED_DEV8_SCENE_IDS:
        raise ValueError("manifest dev8_scene_ids must exactly match registered DEV8")
    return manifest_path, payload


def _dev2_scene_ids(payload: Mapping[str, Any]) -> tuple[str, str]:
    raw = payload.get("dev2_scene_ids")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("manifest dev2_scene_ids must be an explicit sequence")
    values = tuple(map(str, raw))
    if values != REGISTERED_DEV2_SCENE_IDS:
        raise ValueError("manifest must register the exact frozen DEV2 scene order")
    available = {str(row["scene_id"]) for row in payload["scenes"]}
    if not set(values).issubset(available):
        raise ValueError("DEV2 scene is absent from manifest scenes")
    return values[0], values[1]


def _verify_registered_checkout(producer_commit: str) -> None:
    repository = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        tracked_clean = subprocess.run(
            ["git", "-C", str(repository), "diff", "--quiet", "--"],
            check=False,
        ).returncode == 0
        staged_clean = subprocess.run(
            ["git", "-C", str(repository), "diff", "--cached", "--quiet", "--"],
            check=False,
        ).returncode == 0
    except OSError as exc:
        raise RuntimeError("failed to verify the registered Git checkout") from exc
    if head != str(producer_commit):
        raise ValueError("producer_commit differs from the current Git HEAD")
    if not tracked_clean or not staged_clean:
        raise ValueError("registered checkout has tracked or staged modifications")


def _preflight_two_step_roots(
    *, manifest_file: Path, payload: Mapping[str, Any], artifacts: Path, runs: Path
) -> None:
    for destination in (artifacts, runs):
        preflight_audit_output_directory(
            manifest_file,
            destination,
            expected_scene_ids=REGISTERED_DEV8_SCENE_IDS,
        )
    if (
        artifacts == runs
        or artifacts.is_relative_to(runs)
        or runs.is_relative_to(artifacts)
    ):
        raise ValueError("artifacts and runs roots must be disjoint")
    _preflight_source_asset_roots(payload=payload, destinations=(artifacts, runs))


def _preflight_source_asset_roots(
    *, payload: Mapping[str, Any], destinations: Sequence[Path]
) -> None:
    """Reject writes into any image/COLMAP/mask/PLY/checkpoint source tree."""

    protected_roots: set[Path] = set()
    for scene in payload["scenes"]:
        request = _source_request(scene)
        scene_value = request.get("scene", request.get("runtime_registration"))
        if not isinstance(scene_value, Mapping):
            raise ValueError("source evidence request lacks a scene registration")
        inputs = resolve_clean_scene_inputs(scene_value, require_exists=True)
        protected_roots.update(
            {
                inputs.base_path,
                inputs.rgb_ply,
                inputs.sparse,
                inputs.images,
                inputs.sam_masks,
                inputs.grounded_masks,
                inputs.grounded_labels,
            }
        )
        generation = request.get("sam_generation")
        if not isinstance(generation, Mapping):
            raise ValueError("source evidence request lacks sam_generation")
        checkpoint_value = generation.get("checkpoint")
        if not isinstance(checkpoint_value, str) or not checkpoint_value:
            raise ValueError("sam_generation checkpoint must be an explicit path")
        checkpoint = Path(checkpoint_value).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        protected_roots.add(checkpoint)
    for destination in destinations:
        destination = destination.resolve()
        for protected in protected_roots:
            if (
                destination == protected
                or destination.is_relative_to(protected)
                or protected.is_relative_to(destination)
            ):
                raise ValueError(
                    "two-step output roots must be separate from every source asset tree"
                )


def _run_identity(
    *, manifest_file: Path, producer_commit: str, artifacts: Path, runs: Path
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": "saga-clean-mask-contract-two-step-run-identity-v1",
        "producer_commit": str(producer_commit),
        "manifest_path": str(manifest_file.resolve()),
        "manifest_sha256": sha256_file(manifest_file),
        "artifacts_root": str(artifacts.resolve()),
        "runs_root": str(runs.resolve()),
    }
    value["content_sha256"] = hash_json(value)
    return value


def _load_existing_status(path: Path, *, run_identity: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = load_json(path)
    if not isinstance(payload, Mapping) or payload.get("schema") != STATUS_SCHEMA:
        raise ValueError("existing two-step status has an incompatible schema")
    if payload.get("run_identity") != dict(run_identity):
        raise ValueError("existing two-step status belongs to a different run identity")
    return dict(payload)


def _scene_map(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["scene_id"]): dict(row) for row in payload["scenes"]}


def _source_request(scene: Mapping[str, Any]) -> dict[str, Any]:
    value = scene.get("source_evidence_request")
    if not isinstance(value, Mapping):
        raise ValueError("scene.source_evidence_request must be an embedded mapping")
    return dict(value)


def _files_below(roots: Sequence[Path]) -> tuple[Path, ...]:
    """Enumerate the current file set; callers must rerun this after writes."""

    return tuple(
        sorted(
            (
                path
                for root in roots
                if root.exists()
                for path in root.rglob("*")
                if path.is_file()
            ),
            key=str,
        )
    )


def _part_files(paths: Sequence[Path], *, root: Path) -> list[str]:
    result: list[str] = []
    for path in paths:
        if path.name.endswith(".part"):
            try:
                value = path.resolve().relative_to(root.resolve())
            except ValueError:
                value = path.resolve()
            result.append(str(value).replace("\\", "/"))
    return sorted(result)


def audit_clean_baseline(
    *, manifest_path: str | Path, output_root: str | Path
) -> dict[str, Any]:
    """Public step-one entry point."""

    manifest, payload = _load_manifest(manifest_path)
    destination = Path(output_root).resolve()
    preflight_audit_output_directory(
        manifest,
        destination,
        expected_scene_ids=REGISTERED_DEV8_SCENE_IDS,
    )
    _preflight_source_asset_roots(payload=payload, destinations=(destination,))
    result = audit_clean_baseline_manifest(
        manifest,
        output_dir=destination,
        expected_scene_count=DEFAULT_DEV8_SCENE_COUNT,
    )
    return {
        "command": "audit-clean-baseline",
        "status": "complete" if result["technical_gates"]["passed"] else "failed",
        "technical_gates": result["technical_gates"],
        "output_root": str(destination),
    }


def _prepare_flat_mask_control(
    *,
    payload: Mapping[str, Any],
    storage_root: Path,
    producer_commit: str,
) -> dict[str, Any]:
    scene_lookup = _scene_map(payload)
    scene_summaries: dict[str, Any] = {}
    for scene_id in _dev2_scene_ids(payload):
        first = prepare_flat_mask_control_scene(
            scene_id=scene_id,
            source_request=_source_request(scene_lookup[scene_id]),
            output_root=storage_root,
            producer_commit=producer_commit,
        )
        tracked_roots = (
            storage_root / "sam-metadata" / scene_id,
            storage_root / "masks" / "H-hierarchy" / scene_id,
            storage_root / "masks" / "P-flat" / scene_id,
            storage_root / "flat-maps" / scene_id,
            storage_root / "evidence-requests" / scene_id,
        )
        before_files = _files_below(tracked_roots)
        before = build_file_manifest(before_files, root=storage_root)
        second = prepare_flat_mask_control_scene(
            scene_id=scene_id,
            source_request=_source_request(scene_lookup[scene_id]),
            output_root=storage_root,
            producer_commit=producer_commit,
        )
        # Re-enumerate after the second preparation.  Reusing ``before_files``
        # would miss a newly created, deleted, or stranded partial file.
        after_files = _files_below(tracked_roots)
        after = build_file_manifest(after_files, root=storage_root)
        stranded_parts = _part_files(after_files, root=storage_root)
        repeat_pass = bool(
            before == after
            and int(second["generated_frame_count"]) == 0
            and bool(second["mechanical_contract_pass"])
            and bool(second.get("input_binding_pass"))
            and not stranded_parts
        )
        scene_summary = {
            **first,
            "flat_repeat_identity_pass": repeat_pass,
            "repeat_generated_frame_count": int(second["generated_frame_count"]),
            "repeat_input_manifest_before": before,
            "repeat_input_manifest_after": after,
            "stranded_part_files": stranded_parts,
        }
        scene_summaries[scene_id] = scene_summary
        # The second call wrote its own summary.  Replace it with the complete
        # paired/repeat audit that will be bound into the ablation manifest.
        write_json(storage_root / f"{scene_id}_flat_mask_input_audit.json", scene_summary)
    return {
        "scene_ids": list(_dev2_scene_ids(payload)),
        "scenes": scene_summaries,
        "storage_root": str(storage_root),
    }


def _input_preflight_pass(summary: Mapping[str, Any]) -> bool:
    """Technical gate that must pass before any lifting/GPU work starts."""

    scenes = summary.get("scenes")
    if not isinstance(scenes, Mapping) or not scenes:
        return False
    return all(
        bool(row.get("mechanical_contract_pass"))
        and bool(row.get("input_binding_pass"))
        and bool(row.get("flat_repeat_identity_pass"))
        and int(row.get("repeat_generated_frame_count", -1)) == 0
        and row.get("repeat_input_manifest_before")
        == row.get("repeat_input_manifest_after")
        and not row.get("stranded_part_files")
        for row in scenes.values()
        if isinstance(row, Mapping)
    ) and all(isinstance(row, Mapping) for row in scenes.values())


def _write_input_audits(output_root: Path, summary: Mapping[str, Any]) -> None:
    scenes = summary["scenes"]
    regeneration = {
        "schema": "saga-sam-metadata-regeneration-dev2-v1",
        "scene_ids": list(summary["scene_ids"]),
        "generated_frame_count": sum(
            int(row["generated_frame_count"]) for row in scenes.values()
        ),
        "frame_count": sum(int(row["frame_count"]) for row in scenes.values()),
        "historical_hierarchy_exact_frame_count": sum(
            int(row["historical_hierarchy_exact_frame_count"])
            for row in scenes.values()
        ),
        "scenes": {
            key: {
                "frame_count": int(value["frame_count"]),
                "generated_frame_count": int(value["generated_frame_count"]),
                "historical_hierarchy_exact_frame_count": int(
                    value["historical_hierarchy_exact_frame_count"]
                ),
            }
            for key, value in scenes.items()
        },
    }
    flat_audit = {
        "schema": "saga-flat-mask-input-audit-dev2-v1",
        "scene_ids": list(summary["scene_ids"]),
        "mechanical_contract_pass": all(
            bool(row["mechanical_contract_pass"]) for row in scenes.values()
        ),
        "union_changed_pixel_count": sum(
            int(row["union_changed_pixel_count"]) for row in scenes.values()
        ),
        "flat_overlap_pixel_count": sum(
            int(row["flat_overlap_pixel_count"]) for row in scenes.values()
        ),
        "hierarchy_mask_count": sum(
            int(row["hierarchy_mask_count"]) for row in scenes.values()
        ),
        "flat_mask_count": sum(int(row["flat_mask_count"]) for row in scenes.values()),
        "scenes": dict(scenes),
    }
    write_json(output_root / "sam_metadata_regeneration_dev2.json", regeneration)
    write_json(output_root / "flat_mask_input_audit_dev2.json", flat_audit)


def prepare_flat_mask_control(
    *,
    manifest_path: str | Path,
    output_root: str | Path,
    producer_commit: str,
) -> dict[str, Any]:
    """Public step-two input preparation entry point."""

    if _FULL_COMMIT.fullmatch(str(producer_commit)) is None:
        raise ValueError("producer_commit must be a full lowercase Git commit")
    manifest_file, payload = _load_manifest(manifest_path)
    destination = Path(output_root).resolve()
    _verify_registered_checkout(str(producer_commit))
    preflight_audit_output_directory(
        manifest_file,
        destination,
        expected_scene_ids=REGISTERED_DEV8_SCENE_IDS,
    )
    _preflight_source_asset_roots(payload=payload, destinations=(destination,))
    summary = _prepare_flat_mask_control(
        payload=payload,
        storage_root=destination / "mask-control-inputs",
        producer_commit=str(producer_commit),
    )
    _write_input_audits(destination, summary)
    return {
        "command": "prepare-flat-mask-control",
        "status": "complete" if _input_preflight_pass(summary) else "failed",
        "mechanical_contract_pass": _input_preflight_pass(summary),
        "output_root": str(destination),
    }


def _paired_requests(
    summary: Mapping[str, Any], scene_id: str
) -> dict[str, Path]:
    row = summary["scenes"][scene_id]
    return {
        "H-hierarchy": Path(row["hierarchy_evidence_request"]).resolve(),
        "P-flat": Path(row["flat_evidence_request"]).resolve(),
    }


def _run_paired_conditions(
    *,
    payload: Mapping[str, Any],
    summary: Mapping[str, Any],
    run_root: Path,
    producer_commit: str,
) -> dict[str, dict[str, dict[str, str]]]:
    outputs: dict[str, dict[str, dict[str, str]]] = {}
    for scene_id in _dev2_scene_ids(payload):
        outputs[scene_id] = {}
        for arm, request_path in _paired_requests(summary, scene_id).items():
            request = load_json(request_path)
            bank_dir = run_root / "bank" / arm / scene_id
            build_alpha_mask_evidence(
                scene_id=scene_id,
                request=request,
                output_dir=bank_dir,
            )
            bank = load_evidence_bank(bank_dir, expected_scene_id=scene_id)
            condition_dir = run_root / "conditions" / arm / "C0-no-prior" / scene_id
            run_consensus_condition(
                scene_id=scene_id,
                bank_dir=bank_dir,
                condition="C0-no-prior",
                output_dir=condition_dir,
                priors_path=None,
                consumer_commit=producer_commit,
            )
            outputs[scene_id][arm] = {
                "bank_dir": str(bank_dir),
                "output": str(condition_dir / "output.json"),
                "diagnostics": str(condition_dir / "diagnostics.json"),
            }
    return outputs


def _registered_manifest(
    root: Path, *, expected_relative_paths: Sequence[str]
) -> dict[str, Any]:
    files = _files_below((root,))
    relative = []
    for path in files:
        try:
            value = path.resolve().relative_to(root.resolve())
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"file escaped registered root: {path}") from exc
        relative.append(str(value).replace("\\", "/"))
    expected = sorted(map(str, expected_relative_paths))
    parts = _part_files(files, root=root)
    return {
        "root": str(root.resolve()),
        "expected_relative_paths": expected,
        "actual_relative_paths": sorted(relative),
        "file_set_exact": sorted(relative) == expected,
        "part_files": parts,
        "files": build_file_manifest(files, root=root),
    }


def _compare_registered_roots(
    primary: Path,
    repeat: Path,
    *,
    expected_relative_paths: Sequence[str],
) -> dict[str, Any]:
    left = _registered_manifest(
        primary, expected_relative_paths=expected_relative_paths
    )
    right = _registered_manifest(repeat, expected_relative_paths=expected_relative_paths)
    passed = bool(
        left["file_set_exact"]
        and right["file_set_exact"]
        and not left["part_files"]
        and not right["part_files"]
        and left["files"] == right["files"]
    )
    return {"passed": passed, "primary": left, "repeat": right}


def _run_flat_full_repeat(
    *,
    payload: Mapping[str, Any],
    summary: Mapping[str, Any],
    paired_outputs: Mapping[str, Mapping[str, Mapping[str, str]]],
    run_root: Path,
    producer_commit: str,
) -> dict[str, dict[str, Any]]:
    """Recompute P lifting and C0 in an independent tree, then compare bytes."""

    audits: dict[str, dict[str, Any]] = {}
    for scene_id in _dev2_scene_ids(payload):
        request_path = _paired_requests(summary, scene_id)["P-flat"]
        request = load_json(request_path)
        repeat_root = run_root / "repeat" / "P-flat" / scene_id
        repeat_bank = repeat_root / "bank"
        repeat_condition = repeat_root / "C0-no-prior"
        build_alpha_mask_evidence(
            scene_id=scene_id,
            request=request,
            output_dir=repeat_bank,
        )
        load_evidence_bank(repeat_bank, expected_scene_id=scene_id)
        run_consensus_condition(
            scene_id=scene_id,
            bank_dir=repeat_bank,
            condition="C0-no-prior",
            output_dir=repeat_condition,
            priors_path=None,
            consumer_commit=producer_commit,
        )
        primary = paired_outputs[scene_id]["P-flat"]
        bank_compare = _compare_registered_roots(
            Path(primary["bank_dir"]),
            repeat_bank,
            expected_relative_paths=("diagnostics.json", "evidence.npz", "masks.json"),
        )
        output_compare = _compare_registered_roots(
            Path(primary["output"]).parent,
            repeat_condition,
            expected_relative_paths=("diagnostics.json", "output.json"),
        )
        audit = {
            "schema": FLAT_REPEAT_SCHEMA,
            "scene_id": scene_id,
            "primary": {
                "bank_dir": str(Path(primary["bank_dir"]).resolve()),
                "output": str(Path(primary["output"]).resolve()),
                "diagnostics": str(Path(primary["diagnostics"]).resolve()),
            },
            "repeat": {
                "bank_dir": str(repeat_bank.resolve()),
                "output": str((repeat_condition / "output.json").resolve()),
                "diagnostics": str((repeat_condition / "diagnostics.json").resolve()),
            },
            "bank_byte_identity": bank_compare,
            "condition_byte_identity": output_compare,
        }
        audit["passed"] = bool(bank_compare["passed"] and output_compare["passed"])
        audit_path = repeat_root / "flat_full_repeat_audit.json"
        write_json(audit_path, audit)
        audit["audit_path"] = str(audit_path.resolve())
        audits[scene_id] = audit
    return audits


def _write_ablation_manifest(
    *,
    source_payload: Mapping[str, Any],
    paired_outputs: Mapping[str, Mapping[str, Mapping[str, str]]],
    preparation: Mapping[str, Any],
    flat_repeat_audits: Mapping[str, Mapping[str, Any]],
    destination: Path,
) -> Path:
    selected = set(_dev2_scene_ids(source_payload))
    scenes: list[dict[str, Any]] = []
    for row in source_payload["scenes"]:
        scene_id = str(row["scene_id"])
        if scene_id not in selected:
            continue
        copied = dict(row)
        copied["mask_control_conditions"] = {
            arm: dict(paired_outputs[scene_id][arm]) for arm in REGISTERED_ARMS
        }
        copied["flat_mask_input_audit"] = str(
            Path(preparation["storage_root"])
            / f"{scene_id}_flat_mask_input_audit.json"
        )
        copied["flat_repeat_identity_pass"] = bool(
            preparation["scenes"][scene_id]["flat_repeat_identity_pass"]
        )
        copied["flat_full_repeat_audit"] = str(
            flat_repeat_audits[scene_id]["audit_path"]
        )
        scenes.append(copied)
    payload = {
        key: value
        for key, value in source_payload.items()
        if key not in {"scenes", "expected_metrics"}
    }
    payload["scenes"] = scenes
    payload["dev2_scene_ids"] = list(_dev2_scene_ids(source_payload))
    write_json(destination, payload)
    return destination


def run_clean_baseline_two_step(
    *,
    manifest_path: str | Path,
    output_root: str | Path,
    run_root: str | Path,
    producer_commit: str,
) -> dict[str, Any]:
    """Run both registered steps with an explicit technical stop boundary."""

    if _FULL_COMMIT.fullmatch(str(producer_commit)) is None:
        raise ValueError("producer_commit must be a full lowercase Git commit")
    manifest_file, payload = _load_manifest(manifest_path)
    artifacts = Path(output_root).resolve()
    runs = Path(run_root).resolve()
    _verify_registered_checkout(str(producer_commit))
    _preflight_two_step_roots(
        manifest_file=manifest_file,
        payload=payload,
        artifacts=artifacts,
        runs=runs,
    )
    run_identity = _run_identity(
        manifest_file=manifest_file,
        producer_commit=str(producer_commit),
        artifacts=artifacts,
        runs=runs,
    )
    status_path = artifacts / "clean_two_step_status.json"
    existing_status = _load_existing_status(status_path, run_identity=run_identity)
    # Even a prior ``complete`` state is revalidated stage by stage.  The
    # individual audit/evidence/condition loaders skip exact complete products,
    # while a missing or corrupt file is rebuilt from the nearest checkpoint.
    # A status bit alone is never accepted as proof of completeness.
    artifacts.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {
        "schema": STATUS_SCHEMA,
        "producer_commit": str(producer_commit),
        "manifest": str(manifest_file),
        "run_identity": run_identity,
        "stage": "step1-read-only-audit",
        "status": "running",
    }
    _atomic_status(status_path, status)

    audit_result = audit_clean_baseline_manifest(
        manifest_file,
        output_dir=artifacts,
        expected_scene_count=DEFAULT_DEV8_SCENE_COUNT,
    )
    if not bool(audit_result["technical_gates"]["passed"]):
        status.update(
            {
                "stage": "stopped-step1-technical-integrity",
                "status": "stopped",
                "technical_gates": audit_result["technical_gates"],
            }
        )
        _atomic_status(status_path, status)
        raise RuntimeError(
            "step-one technical integrity gate failed; paired mask control was not run"
        )

    status.update({"stage": "step2-sam-metadata", "technical_gates": audit_result["technical_gates"]})
    _atomic_status(status_path, status)
    preparation = _prepare_flat_mask_control(
        payload=payload,
        storage_root=runs / "mask-control-inputs",
        producer_commit=str(producer_commit),
    )
    _write_input_audits(artifacts, preparation)
    # This is intentionally before ``build_alpha_mask_evidence``: no lifting
    # or other GPU work may start if the paired input/repeat contract is weak.
    if not _input_preflight_pass(preparation):
        status.update(
            {
                "stage": "stopped-flat-mask-mechanical-contract",
                "status": "stopped",
            }
        )
        _atomic_status(status_path, status)
        raise RuntimeError("paired flat-mask input contract failed")

    status["stage"] = "step2-paired-evidence-and-consensus"
    _atomic_status(status_path, status)
    paired_outputs = _run_paired_conditions(
        payload=payload,
        summary=preparation,
        run_root=runs,
        producer_commit=str(producer_commit),
    )
    status["stage"] = "step2-flat-full-repeat"
    _atomic_status(status_path, status)
    flat_repeat_audits = _run_flat_full_repeat(
        payload=payload,
        summary=preparation,
        paired_outputs=paired_outputs,
        run_root=runs,
        producer_commit=str(producer_commit),
    )
    ablation_manifest = _write_ablation_manifest(
        source_payload=payload,
        paired_outputs=paired_outputs,
        preparation=preparation,
        flat_repeat_audits=flat_repeat_audits,
        destination=artifacts / "mask_contract_ablation_manifest_dev2.json",
    )

    status["stage"] = "step2-corrected-evaluation"
    _atomic_status(status_path, status)
    from .mask_ablation import evaluate_mask_contract_ablation_manifest

    ablation = evaluate_mask_contract_ablation_manifest(
        ablation_manifest,
        artifacts,
        expected_scene_count=2,
    )
    ablation_analysis = ablation["analysis"]
    analysis = {
        "schema": ANALYSIS_SCHEMA,
        "producer_commit": str(producer_commit),
        "step1_technical_gates": audit_result["technical_gates"],
        "step2_mechanical_gates": ablation_analysis["mechanical_gate"],
        "step2_scientific_gates": ablation_analysis["scientific_gate"],
        "step2_decision": ablation_analysis["conclusion"],
        "category_prior_tested": False,
        "affinity_feature_used_for_geometric_association": False,
        "geometric_identity_unit": "complete-frame-mask-observation",
        "semantic_category_role": "late-object-classification-only",
        "identity_head_trained": False,
        "historical_inputs_modified": False,
        "run_identity": run_identity,
    }
    write_json(artifacts / "clean_two_step_analysis.json", analysis)
    status.update(
        {
            "stage": "complete",
            "status": "complete",
            "decision": ablation_analysis["conclusion"],
        }
    )
    _atomic_status(status_path, status)
    return {
        "command": "run-clean-baseline-two-step",
        "status": "complete",
        "decision": ablation_analysis["conclusion"],
        "output_root": str(artifacts),
        "run_root": str(runs),
    }


__all__ = [
    "ANALYSIS_SCHEMA",
    "STATUS_SCHEMA",
    "audit_clean_baseline",
    "prepare_flat_mask_control",
    "run_clean_baseline_two_step",
]
