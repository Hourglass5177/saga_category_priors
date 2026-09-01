from __future__ import annotations

"""Recoverable staged controller for the clean alpha-mask baseline.

The controller has one deliberately narrow responsibility: sequence an
immutable evidence bank, class-agnostic geometry, formal C0/U/D predictions,
and GT-only diagnostics without allowing ground truth to leak into either the
evidence worker or a formal replay.  Every gate is preregistered: DEV2 and
DEV8 may stop early, while successful runs continue through holdout5,
tune24 (13 physical environments), and the locked final48.
"""

import argparse
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..evaluator import apply_transform, load_ground_truth_npz, load_ply_xyz
from ..io import hash_json, load_json, sha256_file, write_json, write_rows
from ..priors import validate_priors
from .consensus import ConsensusConfig, MaskObservation, run_mask_consensus
from .evaluation import (
    RUN_IDENTITY_SCHEMA,
    CleanCandidate,
    evaluate_candidates,
    evaluate_clean_baseline_manifest,
    evaluate_geometry_oracles,
    evaluate_ground_truth_parity,
    ground_truth_objects_from_arrays,
    gt_point_to_gaussian_mapping,
    prediction_is_complete,
    project_gaussian_support_to_gt_points,
    validate_embedded_identity,
)
from .evidence import (
    EVIDENCE_ARRAY_FILE,
    EVIDENCE_DIAGNOSTICS_FILE,
    EVIDENCE_METADATA_FILE,
    build_alpha_mask_evidence,
    evidence_bank_is_complete,
    evidence_request_source,
    load_evidence_bank,
)
from .identity_control import (
    IDENTITY_CONTROL_REGISTRATION_SCHEMA,
    IDENTITY_TRAIN_SCENES,
    IDENTITY_VALIDATION_SCENE,
    IdentityControlConfig,
    IdentitySceneInput,
    run_identity_edge_control,
)
from .pipeline import run_consensus_condition
from .provenance import build_clean_baseline_provenance
from .size_prior import (
    SizePriorTable,
    oracle_class_size_compatibility,
    global_size_compatibility,
    pca_sorted_extents_m,
)
from .validation import (
    HOLDOUT5,
    ValidationObservation,
    evaluate_final48,
    evaluate_holdout5,
    evaluate_tune24,
    validate_final48_scene_ids,
)
from ..scannet import physical_scene_id
from ..taxonomy import load_taxonomy
from .worker import DEFAULT_CLASSES
from .sam_inputs import (
    colmap_frame_specs,
    ensure_scene_sam_masks,
)


SCENE_INPUT_REGISTRATION_SCHEMA = "saga-clean-stage-scene-input-v1"
EVIDENCE_IMPORT_SCHEMA = "saga-clean-evidence-import-v1"


DEV2 = ("scene0645_00", "scene0025_01")
DEV8 = (
    "scene0645_00",
    "scene0025_01",
    "scene0046_00",
    "scene0474_01",
    "scene0591_02",
    "scene0329_02",
    "scene0164_03",
    "scene0064_01",
)
FORMAL_CONDITIONS = ("C0-no-prior", "U-global", "D-predicted")
STATE_SCHEMA = "saga-clean-alpha-mask-experiment-state-v1"
CONFIG_KIND = "saga_clean_baseline_experiment"
OFFLINE_ORACLE_IDENTITY_SCHEMA = (
    "saga-clean-alpha-mask-offline-oracle-identity-v1"
)
GEOMETRY_ORACLE_SCHEMA = "saga-clean-alpha-mask-geometry-oracle-v2"
ORACLE_SIZE_SCHEMA = "saga-clean-alpha-mask-oracle-size-v2"
GEOMETRY_ORACLE_IMPLEMENTATION = (
    "complete-mask-single-conservative-greedy-perfect-trim-ceiling-v3"
)
ORACLE_SIZE_IMPLEMENTATION = (
    "gt-class-size-veto-mask-consensus-v2"
)

# A recoverable run is still a preregistered state machine.  Persisting the
# previous checkpoint is not sufficient: a truncated or hand-edited state
# must not be able to jump directly to holdout/final evaluation.
COMPLETED_STAGE_SEQUENCE = (
    "validated",
    "dev2-evidence",
    "dev2-geometry-oracle",
    "dev2-c0-u",
    "dev2-prior",
    "dev8-evidence",
    "dev8-uniform",
    "dev8-prior",
    "holdout5-evidence",
    "holdout5-evaluate",
    "tune24-evidence",
    "tune24-evaluate",
    "final48-evidence",
    "final48-evaluate",
)
ACTIVE_NEXT_STAGE = {
    "initialized": "validated",
    **{
        stage: COMPLETED_STAGE_SEQUENCE[index + 1]
        for index, stage in enumerate(COMPLETED_STAGE_SEQUENCE[:-1])
    },
}
STOP_PREDECESSOR = {
    "dev2-inputs-unavailable": "validated",
    "dev2-geometry-gate-failed": "dev2-evidence",
    "dev2-uniform-gate-failed": "dev2-geometry-oracle",
    "dev2-prior-intervention-inactive": "dev2-c0-u",
    "dev8-inputs-unavailable": "dev2-prior",
    "dev8-uniform-gate-failed": "dev8-evidence",
    "dev8-prior-gate-failed": "dev8-uniform",
    "holdout5-inputs-unavailable": "dev8-prior",
    "holdout5-gate-failed": "holdout5-evidence",
    "tune24-inputs-unavailable": "holdout5-evaluate",
    "tune24-gate-failed": "tune24-evidence",
    "final48-inputs-unavailable": "tune24-evaluate",
    "final48-gate-failed": "final48-evidence",
}

# These fields belong to offline evaluation, never to the formal evidence
# worker.  Historical runtime manifests occasionally carried them alongside
# rendering assets, so the materializer strips them before writing a worker
# request and validation rejects any later reintroduction.
EVALUATION_ONLY_RUNTIME_FIELDS = frozenset(
    {
        "gt",
        "gt_npz",
        "gt_path",
        "ground_truth",
        "ground_truth_path",
        "replacement_gt",
        "replacement_gt_npz",
        "gaussian_to_gt_transform",
        "tiny_small_instance_ids",
    }
)


def is_evaluation_only_runtime_field(name: Any) -> bool:
    normalized = str(name).strip().lower()
    return (
        normalized in EVALUATION_ONLY_RUNTIME_FIELDS
        or normalized.startswith("gt_")
        or normalized.endswith("_gt")
        or "ground_truth" in normalized
    )


def evaluation_only_runtime_paths(
    value: Any, *, prefix: str = ""
) -> tuple[str, ...]:
    """Locate forbidden GT/evaluation fields at any request nesting depth."""

    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key)
            path = f"{prefix}.{name}" if prefix else name
            if is_evaluation_only_runtime_field(name):
                found.append(path)
            else:
                found.extend(evaluation_only_runtime_paths(item, prefix=path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            found.extend(evaluation_only_runtime_paths(item, prefix=path))
    return tuple(found)


def _resolved(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _physical_scene_id(scene_id: str) -> str:
    parts = str(scene_id).rsplit("_", 1)
    return parts[0] if len(parts) == 2 else str(scene_id)


@dataclass(frozen=True)
class CleanSceneSpec:
    scene_id: str
    evidence_request: Path
    gt_npz: Path
    gaussian_ply: Path
    gaussian_to_gt_transform: tuple[tuple[float, float, float, float], ...]
    tiny_small_instance_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        matrix = np.asarray(self.gaussian_to_gt_transform, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"{self.scene_id}: transform must be finite 4x4")
        object.__setattr__(
            self,
            "gaussian_to_gt_transform",
            tuple(tuple(float(value) for value in row) for row in matrix),
        )
        object.__setattr__(
            self,
            "tiny_small_instance_ids",
            tuple(sorted({int(value) for value in self.tiny_small_instance_ids})),
        )


@dataclass(frozen=True)
class CleanEvidenceImport:
    """Immutable registration for one evidence bank made by an older producer.

    The imported bytes remain an input to the current consumer; they are never
    a writable cache target.  Their three persisted files, source request, and
    producer commit are all part of the recoverable experiment identity.
    """

    bank_dir: Path
    producer_commit: str
    files: Mapping[str, str]

    def __post_init__(self) -> None:
        root = Path(self.bank_dir).resolve()
        producer = str(self.producer_commit).strip()
        files = {str(key): str(value) for key, value in self.files.items()}
        expected_files = {
            EVIDENCE_ARRAY_FILE,
            EVIDENCE_METADATA_FILE,
            EVIDENCE_DIAGNOSTICS_FILE,
        }
        if len(producer) != 40 or any(
            character not in "0123456789abcdef" for character in producer
        ):
            raise ValueError(
                "imported evidence producer_commit must be a full lowercase git commit"
            )
        if set(files) != expected_files:
            raise ValueError(
                "imported evidence must register exactly evidence.npz, masks.json, "
                "and diagnostics.json"
            )
        for name, digest in files.items():
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(
                    f"imported evidence {name} must have a lowercase SHA-256"
                )
        object.__setattr__(self, "bank_dir", root)
        object.__setattr__(self, "producer_commit", producer)
        object.__setattr__(self, "files", files)


@dataclass(frozen=True)
class CleanExperimentConfig:
    config_path: Path
    code_commit: str
    repo_root: Path
    run_root: Path
    artifact_root: Path
    category_priors: Path
    size_bins_path: Path
    size_bin_boundaries_m: Mapping[str, float]
    evidence_class_names: tuple[str, ...]
    evaluation_class_names: tuple[str, ...]
    b1_fixed_map_50_95: float
    b1_fixed_map_050: float
    scenes: Mapping[str, CleanSceneSpec]
    evidence_imports: Mapping[str, CleanEvidenceImport]
    dev2: tuple[str, ...] = DEV2
    dev8: tuple[str, ...] = DEV8
    holdout5: tuple[str, ...] = HOLDOUT5
    tune24: tuple[str, ...] = ()
    final48: tuple[str, ...] = ()
    min_region_size: int = 100
    radius_m: float = 0.05
    train_physical_scene_ids: tuple[str, ...] = ()
    holdout_physical_scene_ids: tuple[str, ...] = ()
    final_physical_scene_ids: tuple[str, ...] = ()
    sai3d_asset_paths: tuple[Path, ...] = ()
    identity_control: IdentityControlConfig | None = None
    identity_control_registration: Mapping[str, Any] | None = None

    @property
    def state_path(self) -> Path:
        return self.artifact_root / "clean_baseline_status.json"

    @property
    def analysis_path(self) -> Path:
        return self.artifact_root / "clean_baseline_analysis.json"

    def bank_dir(self, scene_id: str) -> Path:
        imported = self.evidence_imports.get(str(scene_id))
        return (
            imported.bank_dir
            if imported is not None
            else self.run_root / "bank" / scene_id
        )

    def condition_dir(self, scene_id: str, condition: str) -> Path:
        return self.run_root / "conditions" / condition / scene_id

    def oracle_path(self, scene_id: str) -> Path:
        return self.run_root / "oracle" / scene_id / "D-oracle-class.json"

    def prepared_request_path(self, scene_id: str) -> Path:
        return self.artifact_root / "prepared_requests" / f"{scene_id}.json"

    def scene_input_path(self, scene_id: str) -> Path:
        return self.artifact_root / "scene_inputs" / f"{scene_id}.json"

    def prepared_scene_spec(self, scene_id: str) -> CleanSceneSpec:
        """Return the stage-validated scene contract, never an eager guess."""

        source = self.scene_input_path(scene_id)
        if not source.is_file():
            raise RuntimeError(f"{scene_id}: stage-local scene input was not prepared")
        payload = load_json(source)
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != SCENE_INPUT_REGISTRATION_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("scene_id") != scene_id
            or payload.get("code_commit") != self.code_commit
        ):
            raise ValueError(f"{scene_id}: invalid stage-local scene registration")
        base = self.scenes[scene_id]
        request = Path(str(payload["prepared_request"])).resolve()
        if not request.is_file():
            raise FileNotFoundError(request)
        content_identity = payload.get("content_identity")
        if not isinstance(content_identity, Mapping):
            raise ValueError(f"{scene_id}: prepared scene lacks a content identity")
        expected_content = {
            "prepared_request_sha256": sha256_file(request),
            "gt_npz_sha256": sha256_file(base.gt_npz),
            "gaussian_ply_sha256": sha256_file(base.gaussian_ply),
        }
        if dict(content_identity) != expected_content:
            raise ValueError(f"{scene_id}: prepared scene input content changed")
        if Path(str(payload["gt_npz"])).resolve() != base.gt_npz:
            raise ValueError(f"{scene_id}: prepared GT path changed")
        if Path(str(payload["gaussian_ply"])).resolve() != base.gaussian_ply:
            raise ValueError(f"{scene_id}: prepared Gaussian path changed")
        return replace(
            base,
            evidence_request=request,
            tiny_small_instance_ids=tuple(
                int(value) for value in payload["tiny_small_instance_ids"]
            ),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "CleanExperimentConfig":
        source = Path(path).resolve()
        payload = load_json(source)
        if not isinstance(payload, Mapping) or payload.get("kind") != CONFIG_KIND:
            raise ValueError(f"expected registered {CONFIG_KIND} config")
        base = source.parent
        scene_rows = payload.get("scenes")
        if not isinstance(scene_rows, Mapping):
            raise TypeError("config.scenes must be an object keyed by scene ID")
        scenes: dict[str, CleanSceneSpec] = {}
        for scene_id, raw in scene_rows.items():
            if not isinstance(raw, Mapping):
                raise TypeError(f"scene {scene_id!r} must be an object")
            scenes[str(scene_id)] = CleanSceneSpec(
                scene_id=str(scene_id),
                evidence_request=_resolved(base, raw["evidence_request"]),
                gt_npz=_resolved(base, raw["gt_npz"]),
                gaussian_ply=_resolved(base, raw["gaussian_ply"]),
                gaussian_to_gt_transform=tuple(
                    tuple(float(value) for value in row)
                    for row in raw["gaussian_to_gt_transform"]
                ),
                tiny_small_instance_ids=tuple(
                    int(value) for value in raw.get("tiny_small_instance_ids", [])
                ),
            )
        anchor = payload.get("b1_fixed_metrics")
        if not isinstance(anchor, Mapping):
            raise TypeError("config.b1_fixed_metrics is required")
        identity_control_raw = payload.get("identity_control")
        if identity_control_raw is not None and not isinstance(
            identity_control_raw, Mapping
        ):
            raise TypeError("config.identity_control must be an object")
        identity_registration_raw = payload.get("identity_control_registration")
        if not isinstance(identity_registration_raw, Mapping):
            raise TypeError(
                "config.identity_control_registration must be an object"
            )
        raw_imports = payload.get("evidence_imports", {})
        if not isinstance(raw_imports, Mapping):
            raise TypeError("config.evidence_imports must be an object")
        evidence_imports: dict[str, CleanEvidenceImport] = {}
        for scene_id, raw in raw_imports.items():
            if not isinstance(raw, Mapping):
                raise TypeError(
                    f"imported evidence registration for {scene_id!r} must be an object"
                )
            if raw.get("schema") != EVIDENCE_IMPORT_SCHEMA:
                raise ValueError(
                    f"imported evidence registration for {scene_id!r} has the wrong schema"
                )
            files = raw.get("files")
            if not isinstance(files, Mapping):
                raise TypeError(
                    f"imported evidence registration for {scene_id!r} lacks file hashes"
                )
            evidence_imports[str(scene_id)] = CleanEvidenceImport(
                bank_dir=_resolved(base, raw["bank_dir"]),
                producer_commit=str(raw["producer_commit"]),
                files={str(key): str(value) for key, value in files.items()},
            )
        config = cls(
            config_path=source,
            code_commit=str(payload["code_commit"]),
            repo_root=_resolved(base, payload["repo_root"]),
            run_root=_resolved(base, payload["run_root"]),
            artifact_root=_resolved(base, payload["artifact_root"]),
            category_priors=_resolved(base, payload["category_priors"]),
            size_bins_path=_resolved(base, payload["size_bins"]),
            size_bin_boundaries_m={
                str(key): float(value)
                for key, value in payload["size_bin_boundaries_m"].items()
            },
            evidence_class_names=tuple(
                map(str, payload["evidence_class_names"])
            ),
            evaluation_class_names=tuple(
                map(str, payload["evaluation_class_names"])
            ),
            b1_fixed_map_50_95=float(anchor["map_50_95"]),
            b1_fixed_map_050=float(anchor["map_0.50"]),
            scenes=scenes,
            evidence_imports=evidence_imports,
            dev2=tuple(map(str, payload.get("dev2", DEV2))),
            dev8=tuple(map(str, payload.get("dev8", DEV8))),
            holdout5=tuple(map(str, payload.get("holdout5", HOLDOUT5))),
            tune24=tuple(map(str, payload["tune24"])),
            final48=tuple(map(str, payload["final48"])),
            min_region_size=int(payload.get("min_region_size", 100)),
            radius_m=float(payload.get("radius_m", 0.05)),
            train_physical_scene_ids=tuple(
                map(str, payload.get("train_physical_scene_ids", []))
            ),
            holdout_physical_scene_ids=tuple(
                map(
                    str,
                    payload.get(
                        "holdout_physical_scene_ids",
                        sorted({_physical_scene_id(value) for value in payload.get("holdout5", HOLDOUT5)}),
                    ),
                )
            ),
            final_physical_scene_ids=tuple(
                map(
                    str,
                    payload.get(
                        "final_physical_scene_ids",
                        sorted({_physical_scene_id(value) for value in payload["final48"]}),
                    ),
                )
            ),
            sai3d_asset_paths=tuple(
                _resolved(base, value)
                for value in payload.get("sai3d_asset_paths", [])
            ),
            identity_control=(
                None
                if identity_control_raw is None
                else IdentityControlConfig.from_mapping(
                    identity_control_raw, base=base
                )
            ),
            identity_control_registration=dict(identity_registration_raw),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.dev2 != DEV2 or self.dev8 != DEV8:
            raise ValueError("DEV2/DEV8 scene lists differ from preregistration")
        if self.holdout5 != HOLDOUT5:
            raise ValueError("holdout5 scene list differs from preregistration")
        if len(self.tune24) != 24 or len(set(self.tune24)) != 24:
            raise ValueError("tune24 must register exactly 24 unique scans")
        tune_physical = {_physical_scene_id(value) for value in self.tune24}
        if len(tune_physical) != 13:
            raise ValueError("tune24 must resolve to exactly 13 physical scenes")
        if not set(self.dev8 + self.holdout5).issubset(self.tune24):
            raise ValueError("tune24 must contain all DEV8 and holdout5 scans")
        validate_final48_scene_ids(self.final48)
        if tuple(self.dev8[:2]) != self.dev2 or len(set(self.dev8)) != 8:
            raise ValueError("DEV2 must be the first two unique DEV8 scenes")
        registered_scenes = set(self.tune24).union(self.final48)
        missing = sorted(registered_scenes.difference(self.scenes))
        if missing:
            raise ValueError(f"registered scene specifications are missing: {missing}")
        unexpected_imports = sorted(set(self.evidence_imports).difference(registered_scenes))
        if unexpected_imports:
            raise ValueError(
                f"imported evidence references unregistered scenes: {unexpected_imports}"
            )
        if len(self.code_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.code_commit
        ):
            raise ValueError("code_commit must be a full lowercase git commit")
        expected_evidence = tuple(DEFAULT_CLASSES)
        expected_evaluation = tuple(load_taxonomy().canonical_classes)
        if self.evidence_class_names != expected_evidence:
            raise ValueError(
                "evidence_class_names must exactly match the registered "
                "32-class Grounded-SAM/codebook order"
            )
        if self.evaluation_class_names != expected_evaluation:
            raise ValueError(
                "evaluation_class_names must exactly match the canonical "
                "SAGA20 evaluator order"
            )
        if self.evaluation_class_names[3] != "tv":
            raise ValueError("official evaluation class index 3 must be tv")
        if self.evidence_class_names[3] != "flower":
            raise ValueError("32-class evidence index 3 must be flower")
        # The new materializer writes these two legacy-shaped JSON fields only
        # to make the on-disk correspondence explicit.  They are not accepted
        # as a fallback for old configs and are never stored as runtime fields.
        payload = load_json(self.config_path)
        if tuple(map(str, payload.get("class_names", ()))) != self.evidence_class_names:
            raise ValueError("config.class_names must explicitly mirror evidence_class_names")
        if tuple(map(str, payload.get("allowed_classes", ()))) != self.evaluation_class_names:
            raise ValueError(
                "config.allowed_classes must explicitly mirror evaluation_class_names"
            )
        runtime_registration = payload.get("runtime_registration")
        expected_scene_ids = set(self.tune24).union(self.final48)
        if not isinstance(runtime_registration, Mapping) or set(
            map(str, runtime_registration)
        ) != expected_scene_ids:
            raise ValueError(
                "config must freeze one original runtime registration for every scene"
            )
        for scene_id in expected_scene_ids:
            row = runtime_registration[scene_id]
            if not isinstance(row, Mapping):
                raise TypeError(f"{scene_id}: runtime registration must be an object")
            request = load_json(self.scenes[scene_id].evidence_request)
            if request.get("runtime_registration") != row:
                raise ValueError(
                    f"{scene_id}: evidence request runtime registration drifted"
                )
            expected_producer = (
                self.evidence_imports[scene_id].producer_commit
                if scene_id in self.evidence_imports
                else self.code_commit
            )
            if request.get("producer_commit") != expected_producer:
                raise ValueError(
                    f"{scene_id}: evidence request producer_commit differs from its "
                    "registered producer"
                )
            request_scene = request.get("scene")
            if not isinstance(request_scene, Mapping):
                raise TypeError(f"{scene_id}: evidence request.scene must be an object")
            forbidden = sorted(
                set(evaluation_only_runtime_paths(row, prefix="runtime_registration"))
                | set(evaluation_only_runtime_paths(request_scene, prefix="scene"))
            )
            if forbidden:
                raise ValueError(
                    f"{scene_id}: evaluation-only fields leaked into the formal "
                    f"runtime request: {sorted(set(forbidden))}"
                )
        if self.min_region_size != 100 or not math.isclose(self.radius_m, 0.05):
            raise ValueError("official min_region_size=100 and radius=0.05 are frozen")
        expected_size_keys = ("tiny_max_m", "small_max_m", "medium_max_m")
        if set(self.size_bin_boundaries_m) != set(expected_size_keys):
            raise ValueError("size_bin_boundaries_m has unexpected fields")
        size_values = tuple(
            float(self.size_bin_boundaries_m[key]) for key in expected_size_keys
        )
        if (
            any(not math.isfinite(value) or value <= 0 for value in size_values)
            or not size_values[0] <= size_values[1] <= size_values[2]
        ):
            raise ValueError("size-bin metric boundaries must be positive and ordered")
        source_size_bins = load_json(self.size_bins_path)
        source_boundaries = source_size_bins.get(
            "boundaries_m", source_size_bins
        )
        if {
            key: float(source_boundaries[key]) for key in expected_size_keys
        } != dict(self.size_bin_boundaries_m):
            raise ValueError("registered size-bin boundaries differ from their source")
        if not self.train_physical_scene_ids:
            raise ValueError(
                "train_physical_scene_ids must be explicitly registered for split audit"
            )
        for value in (self.b1_fixed_map_50_95, self.b1_fixed_map_050):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("B1-fixed metrics must be finite values in [0,1]")
        expected_holdout_physical = {
            _physical_scene_id(value) for value in self.holdout5
        }
        expected_final_physical = {_physical_scene_id(value) for value in self.final48}
        if set(map(_physical_scene_id, self.holdout_physical_scene_ids)) != expected_holdout_physical:
            raise ValueError("holdout physical-scene registration differs from holdout5")
        if set(map(_physical_scene_id, self.final_physical_scene_ids)) != expected_final_physical:
            raise ValueError("final physical-scene registration differs from final48")
        raw_groups = {
            "ScanNet-train priors": self.train_physical_scene_ids,
            "DEV8": self.dev8,
            "holdout": self.holdout5,
            "final": self.final48,
        }
        groups: dict[str, set[str]] = {}
        for label, values in raw_groups.items():
            physical = tuple(map(_physical_scene_id, values))
            if len(physical) != len(set(physical)):
                raise ValueError(f"{label} contains duplicate physical scenes")
            groups[label] = set(physical)
        labels = tuple(groups)
        for left_index, left in enumerate(labels):
            for right in labels[left_index + 1 :]:
                overlap = groups[left].intersection(groups[right])
                if overlap:
                    raise ValueError(
                        f"{left} overlaps {right}: {sorted(overlap)}"
                    )
        if self.identity_control is not None:
            identity_scenes = set(self.identity_control.train_scene_ids) | {
                self.identity_control.validation_scene_id
            }
            if not identity_scenes.issubset(self.dev8):
                raise ValueError("identity-control scenes must remain inside DEV8")
        registration = self.identity_control_registration
        if not isinstance(registration, Mapping):
            raise ValueError("identity-control registration must be explicit")
        if registration.get("schema") != IDENTITY_CONTROL_REGISTRATION_SCHEMA:
            raise ValueError("identity-control registration schema mismatch")
        registration_status = registration.get("status")
        if registration_status not in {"available", "unavailable"}:
            raise ValueError("identity-control registration status is invalid")
        issues = registration.get("issues")
        if not isinstance(issues, list) or any(
            not isinstance(value, str) or not value for value in issues
        ):
            raise ValueError("identity-control registration issues must be a list")
        if tuple(map(str, registration.get("train_scene_ids", ()))) != (
            IDENTITY_TRAIN_SCENES
        ):
            raise ValueError("identity-control registered training scenes differ")
        if str(registration.get("validation_scene_id", "")) != (
            IDENTITY_VALIDATION_SCENE
        ):
            raise ValueError("identity-control registered validation scene differs")
        if registration_status == "available":
            if self.identity_control is None or issues:
                raise ValueError(
                    "available identity-control registration requires assets and no issues"
                )
        elif self.identity_control is not None or not issues:
            raise ValueError(
                "unavailable identity-control registration requires explicit issues and no control"
            )


Hook = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True)
class CleanExperimentHooks:
    build_evidence: Hook
    geometry_oracle: Hook
    run_formal: Hook
    run_oracle: Hook
    evaluate_stage: Hook
    run_identity_control: Hook | None = None


def _state_identity(config: CleanExperimentConfig) -> dict[str, Any]:
    return {
        "config_content_sha256": sha256_file(config.config_path),
        "config_path": str(config.config_path),
        "code_commit": config.code_commit,
        "repo_root": str(config.repo_root),
        "run_root": str(config.run_root),
        "artifact_root": str(config.artifact_root),
        "category_priors": str(config.category_priors),
        "category_priors_sha256": sha256_file(config.category_priors),
        "size_bins_path": str(config.size_bins_path),
        "size_bins_sha256": sha256_file(config.size_bins_path),
        "size_bin_boundaries_m": dict(config.size_bin_boundaries_m),
        "evidence_class_names": list(config.evidence_class_names),
        "evaluation_class_names": list(config.evaluation_class_names),
        "dev2": list(config.dev2),
        "dev8": list(config.dev8),
        "holdout5": list(config.holdout5),
        "tune24": list(config.tune24),
        "final48": list(config.final48),
        "train_physical_scene_ids": list(config.train_physical_scene_ids),
        "holdout_physical_scene_ids": list(config.holdout_physical_scene_ids),
        "final_physical_scene_ids": list(config.final_physical_scene_ids),
        "identity_control": (
            None
            if config.identity_control is None
            else config.identity_control.identity()
        ),
        "identity_control_registration": _json_safe(
            config.identity_control_registration
        ),
        "evidence_imports": {
            scene_id: {
                "schema": EVIDENCE_IMPORT_SCHEMA,
                "bank_dir": str(registration.bank_dir),
                "producer_commit": registration.producer_commit,
                "files": dict(registration.files),
            }
            for scene_id, registration in sorted(config.evidence_imports.items())
        },
        "radius_m": config.radius_m,
        "min_region_size": config.min_region_size,
        "b1_fixed_map_50_95": config.b1_fixed_map_50_95,
        "b1_fixed_map_050": config.b1_fixed_map_050,
        "scenes": {
            scene_id: {
                "evidence_request": str(config.scenes[scene_id].evidence_request),
                "evidence_request_sha256": sha256_file(
                    config.scenes[scene_id].evidence_request
                ),
                "gt_npz": str(config.scenes[scene_id].gt_npz),
                "gaussian_ply": str(config.scenes[scene_id].gaussian_ply),
                "gaussian_to_gt_transform": [
                    list(row)
                    for row in config.scenes[scene_id].gaussian_to_gt_transform
                ],
                "tiny_small_instance_ids": list(
                    config.scenes[scene_id].tiny_small_instance_ids
                ),
            }
            for scene_id in sorted(set(config.tune24).union(config.final48))
        },
    }


def _new_state(config: CleanExperimentConfig) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "status": "active",
        "checkpoint": "initialized",
        "current_stage": None,
        "next_stage": "validate",
        "stop_reason": None,
        "identity": _state_identity(config),
        "history": [],
        "formal_conditions": list(FORMAL_CONDITIONS),
        "oracle_class_formal_output": False,
        "candidate_prior_tested": False,
        "prior_mechanical_intervention_tested": False,
        "identity_control_run": False,
        "identity_control_formal_method": False,
    }


def _history_stage_prefix(checkpoint: str) -> list[str]:
    if checkpoint == "initialized":
        return []
    try:
        index = COMPLETED_STAGE_SEQUENCE.index(checkpoint)
    except ValueError as exc:
        raise ValueError(f"unknown clean-baseline checkpoint: {checkpoint}") from exc
    return list(COMPLETED_STAGE_SEQUENCE[: index + 1])


def _validate_state_machine(state: Mapping[str, Any]) -> None:
    """Reject an impossible or stage-skipping recoverable state."""

    status = str(state.get("status", ""))
    checkpoint = str(state.get("checkpoint", ""))
    history = state.get("history")
    if not isinstance(history, list) or any(
        not isinstance(row, Mapping) or not isinstance(row.get("stage"), str)
        for row in history
    ):
        raise TypeError("state history must contain stage mappings")
    observed_stages = [str(row["stage"]) for row in history]
    current = state.get("current_stage")
    next_stage = state.get("next_stage")
    active_substage = state.get("active_substage")
    if status == "active":
        if checkpoint not in ACTIVE_NEXT_STAGE:
            raise ValueError("active state has an impossible checkpoint")
        expected_history = _history_stage_prefix(checkpoint)
        if observed_stages != expected_history:
            raise ValueError("active state history skips or reorders stages")
        expected_next = ACTIVE_NEXT_STAGE[checkpoint]
        allowed_next = {expected_next}
        # The pristine state historically displayed the shorter user-facing
        # label; once execution begins `_begin` records the exact stage name.
        if checkpoint == "initialized" and current is None:
            allowed_next.add("validate")
        if next_stage not in allowed_next:
            raise ValueError("active state next_stage is inconsistent")
        if current not in (None, expected_next):
            raise ValueError("active state current_stage is inconsistent")
        if active_substage is not None and not (
            checkpoint == "dev8-evidence"
            and current == "dev8-uniform"
            and next_stage == "dev8-uniform"
            and active_substage == "identity-edge-control"
        ):
            raise ValueError("active state contains an unauthorized substage")
        if state.get("stop_reason") not in (None, ""):
            raise ValueError("active state cannot carry a stop_reason")
        return
    if status == "complete":
        if (
            checkpoint != "final48-complete"
            or observed_stages != list(COMPLETED_STAGE_SEQUENCE)
            or current is not None
            or next_stage is not None
            or state.get("stop_reason") not in (None, "")
            or active_substage is not None
        ):
            raise ValueError("complete state does not contain the full stage chain")
        return
    if status == "stopped":
        predecessor = STOP_PREDECESSOR.get(checkpoint)
        if predecessor is None:
            raise ValueError("stopped state has an unknown stop checkpoint")
        expected_history = _history_stage_prefix(predecessor) + [checkpoint]
        if observed_stages != expected_history:
            raise ValueError("stopped state history skips or reorders stages")
        if current is not None or next_stage is not None or active_substage is not None:
            raise ValueError("stopped state cannot have an active/next stage")
        if not isinstance(state.get("stop_reason"), str) or not state["stop_reason"]:
            raise ValueError("stopped state requires a reason")
        return
    raise ValueError(f"unknown clean-baseline state status: {status!r}")


def _write_state(config: CleanExperimentConfig, state: dict[str, Any]) -> None:
    config.artifact_root.mkdir(parents=True, exist_ok=True)
    state.pop("content_sha256", None)
    state["content_sha256"] = hash_json(state)
    write_json(config.state_path, state)
    write_json(
        config.analysis_path,
        {
            "schema": "saga-clean-alpha-mask-analysis-v1",
            "status": state["status"],
            "checkpoint": state["checkpoint"],
            "next_stage": state.get("next_stage"),
            "stop_reason": state.get("stop_reason"),
            "candidate_prior_tested": state.get("candidate_prior_tested", False),
            "prior_mechanical_intervention_tested": state.get(
                "prior_mechanical_intervention_tested", False
            ),
            "oracle_class_formal_output": False,
            "identity_control_run": state.get("identity_control_run", False),
            "identity_control_formal_method": False,
            "history": state["history"],
        },
    )


def _load_state(config: CleanExperimentConfig) -> dict[str, Any]:
    if not config.state_path.is_file():
        state = _new_state(config)
        _write_state(config, state)
        return state
    payload = load_json(config.state_path)
    if not isinstance(payload, Mapping) or payload.get("schema") != STATE_SCHEMA:
        raise ValueError("invalid clean-baseline state file")
    state = dict(payload)
    expected_hash = state.pop("content_sha256", None)
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("clean-baseline state lacks an embedded content identity")
    if hash_json(state) != expected_hash:
        raise ValueError("clean-baseline state content identity mismatch")
    state["content_sha256"] = expected_hash
    if state.get("identity") != _state_identity(config):
        raise ValueError("registered config differs from the recoverable state")
    if not isinstance(state.get("history"), list):
        raise TypeError("state history must be a list")
    _validate_state_machine(state)
    return state


def _begin(config: CleanExperimentConfig, state: dict[str, Any], stage: str) -> None:
    state.update(current_stage=stage, next_stage=stage)
    state.pop("last_error", None)
    _write_state(config, state)


def _complete(
    config: CleanExperimentConfig,
    state: dict[str, Any],
    stage: str,
    result: Mapping[str, Any],
    next_stage: str | None,
) -> None:
    state.pop("active_substage", None)
    state["history"].append({"stage": stage, **_history_safe(dict(result))})
    state.update(
        checkpoint=stage,
        current_stage=None,
        next_stage=next_stage,
    )
    _write_state(config, state)


def _stop(
    config: CleanExperimentConfig,
    state: dict[str, Any],
    checkpoint: str,
    reason: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    state.pop("active_substage", None)
    state["history"].append({"stage": checkpoint, **_history_safe(dict(result))})
    state.update(
        status="stopped",
        checkpoint=checkpoint,
        current_stage=None,
        next_stage=None,
        stop_reason=reason,
    )
    _write_state(config, state)
    return state


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _history_safe(value: Any) -> Any:
    """Keep state compact while full metrics remain in acceptance artifacts."""

    omitted = {
        "rows",
        "prior_rows",
        "scenes",
        "candidate_rows",
        "gt_rows",
        "results",
        "size_merge_decisions",
        "accepted_edges",
    }
    if isinstance(value, Mapping):
        return {
            str(key): _history_safe(item)
            for key, item in value.items()
            if key not in omitted
        }
    if isinstance(value, (tuple, list)):
        return [_history_safe(item) for item in value]
    return _json_safe(value)


def _aggregate_oracle_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    match_050 = 0
    tiny_match_025 = 0
    tiny_count = 0
    for row in rows:
        # The greedy association diagnostic is a feasible lower bound and can
        # miss a good subset.  Early stopping is therefore based on the true
        # complete-mask support ceiling: perfect deletion of false positives.
        official = row["aggregate"]["official_valid"]["perfect_trim"]
        tiny_group = row["aggregate"]["tiny_small_official_valid"]
        tiny = tiny_group["perfect_trim"]
        match_050 += int(official["match_050_count"])
        tiny_match_025 += int(tiny["match_025_count"])
        tiny_count += int(tiny_group["gt_count"])
    tiny_recall = tiny_match_025 / tiny_count if tiny_count else 0.0
    checks = {
        "geometric_iou050_count_at_least_6": match_050 >= 6,
        "tiny_small_recall025_at_least_020": tiny_recall >= 0.20,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "geometric_iou050_count": match_050,
        "tiny_small_match025_count": tiny_match_025,
        "tiny_small_gt_count": tiny_count,
        "tiny_small_recall025": tiny_recall,
        "gate_metric": "perfect_trim_support_ceiling",
        "greedy_association_used_for_gate": False,
    }


def _aggregate_oracle_gate_dev8(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    base = _aggregate_oracle_gate(rows)
    checks = dict(base["checks"])
    checks["geometric_iou050_count_at_least_16"] = (
        int(base["geometric_iou050_count"]) >= 16
    )
    checks.pop("geometric_iou050_count_at_least_6", None)
    return {**base, "passed": all(checks.values()), "checks": checks}


def _dev2_uniform_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    c0 = report["conditions"]["C0-no-prior"]["candidate"]
    uniform = report["conditions"]["U-global"]["candidate"]
    checks = {
        "iou025_not_lower": int(uniform["geometry_iou_025_count"])
        >= int(c0["geometry_iou_025_count"]),
        "iou050_not_lower": int(uniform["geometry_iou_050_count"])
        >= int(c0["geometry_iou_050_count"]),
        "candidate_count_at_most_1_25x": int(uniform["candidate_count"])
        <= 1.25 * int(c0["candidate_count"]),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _prior_mechanical_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    effect = report["prior_effect"]
    checks = {
        "ten_percent_merge_status_changed": float(
            effect["merge_status_change_fraction"]
        )
        >= 0.10,
        "at_least_five_final_merge_decisions_changed": int(
            effect["final_merge_decision_change_count"]
        )
        >= 5,
    }
    return {"passed": any(checks.values()), "checks": checks}


def _dev8_health_gate(
    config: CleanExperimentConfig, report: Mapping[str, Any]
) -> dict[str, Any]:
    uniform = report["conditions"]["U-global"]
    official = uniform["official"]
    candidate = uniform["candidate"]
    checks = {
        "geometric_iou050_at_least_16": int(candidate["geometry_iou_050_count"])
        >= 16,
        "geometric_iou050_covers_four_scenes": int(
            candidate["geometry_iou_050_scene_count"]
        )
        >= 4,
        "same_class_iou050_at_least_12": int(
            candidate["same_class_iou_050_count"]
        )
        >= 12,
        "same_class_iou050_covers_four_scenes": int(
            candidate["same_class_iou_050_scene_count"]
        )
        >= 4,
        "candidate_precision025_at_least_010": float(
            candidate["candidate_precision_025"]
        )
        >= 0.10,
        "tiny_small_recall025_at_least_020": float(
            candidate["tiny_small_recall_025"]
        )
        >= 0.20,
        "map_safety": float(official["map_50_95"])
        >= config.b1_fixed_map_50_95 - 0.001,
        "ap50_safety": float(official["map_0.50"])
        >= config.b1_fixed_map_050 - 0.002,
        "score_iou_spearman_at_least_020": float(candidate["score_iou_spearman"])
        >= 0.20,
        "orphan_zero": int(uniform["contract"]["orphan_gaussian_count"]) == 0,
        "negative_metadata_zero": int(
            uniform["contract"]["negative_metadata_count"]
        )
        == 0,
        "duplicate_ownership_zero": int(
            uniform["contract"]["duplicate_ownership_count"]
        )
        == 0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _dev8_prior_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    effect = _prior_mechanical_gate(report)
    delta = report["data_minus_uniform"]
    benefit = (
        float(delta["map_50_95_delta"]) >= 0.002
        or (
            float(delta["tiny_small_recall_050_delta"]) >= 0.01
            and float(delta["map_50_95_delta"]) >= -0.0005
        )
    )
    degradation = delta.get("fp_tp_degradation")
    checks = {
        "mechanically_effective": bool(effect["passed"]),
        "registered_benefit": benefit,
        "positive_scenes_at_least_5": int(delta["positive_scene_count"]) >= 5,
        "fp_tp_degradation_at_most_020": degradation is not None
        and float(degradation) <= 0.20,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _validation_observations_from_report(
    report: Mapping[str, Any], scene_ids: Sequence[str]
) -> list[dict[str, Any]]:
    expected = set(map(str, scene_ids))
    observations: list[dict[str, Any]] = []
    for condition in ("U-global", "D-predicted"):
        condition_report = report.get("conditions", {}).get(condition)
        if not isinstance(condition_report, Mapping):
            raise ValueError(f"validation report is missing {condition}")
        rows = condition_report.get("scenes")
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise TypeError(f"{condition}: per-scene validation rows are required")
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, Mapping):
                raise TypeError("per-scene validation rows must be mappings")
            scene_id = str(row["scene_id"])
            if scene_id in seen:
                raise ValueError(f"duplicate {scene_id}/{condition} scene metric")
            seen.add(scene_id)
            official = row.get("official")
            candidate = row.get("candidate")
            if not isinstance(official, Mapping) or not isinstance(candidate, Mapping):
                raise TypeError("validation rows require official and candidate metrics")
            observation = ValidationObservation(
                scene_id=scene_id,
                physical_scene_id=physical_scene_id(scene_id),
                condition=condition,
                map_50_95=float(official["map_50_95"] or 0.0),
                tiny_small_recall_050=float(candidate["tiny_small_recall_050"]),
            )
            observations.append(observation.to_row())
        if seen != expected:
            raise ValueError(
                f"{condition}: validation scene set differs; "
                f"missing={sorted(expected-seen)}, unexpected={sorted(seen-expected)}"
            )
    return observations


def _build_registered_evidence(
    config: CleanExperimentConfig,
    hooks: CleanExperimentHooks,
    scene_ids: Sequence[str],
) -> dict[str, Any]:
    availability = _prepare_registered_scenes(config, scene_ids)
    if not availability["available"]:
        return {
            "scene_ids": list(map(str, scene_ids)),
            "results": [],
            "input_availability": availability,
        }
    results = []
    for scene_id in scene_ids:
        spec = config.prepared_scene_spec(str(scene_id))
        results.append(
            hooks.build_evidence(
                scene_id=str(scene_id),
                request_path=spec.evidence_request,
                output_dir=config.bank_dir(str(scene_id)),
            )
        )
    return {
        "scene_ids": list(map(str, scene_ids)),
        "results": results,
        "input_availability": availability,
    }


def _run_registered_paired_stage(
    config: CleanExperimentConfig,
    hooks: CleanExperimentHooks,
    *,
    scene_ids: Sequence[str],
    stage: str,
) -> dict[str, Any]:
    for scene_id in scene_ids:
        for condition in ("U-global", "D-predicted"):
            hooks.run_formal(
                scene_id=str(scene_id),
                bank_dir=config.bank_dir(str(scene_id)),
                condition=condition,
                output_dir=config.condition_dir(str(scene_id), condition),
                priors_path=config.category_priors,
                allowed_classes=config.evaluation_class_names,
            )
    report = hooks.evaluate_stage(
        scene_ids=tuple(map(str, scene_ids)),
        conditions=("U-global", "D-predicted"),
        output_path=config.artifact_root / f"size_prior_{stage}_evaluation.json",
    )
    observations = _validation_observations_from_report(report, scene_ids)
    if stage == "holdout5":
        validation = evaluate_holdout5(observations)
    elif stage == "tune24":
        validation = evaluate_tune24(observations)
    elif stage == "final48":
        validation = evaluate_final48(observations)
    else:  # pragma: no cover - private call contract
        raise ValueError(f"unknown clean validation stage: {stage}")
    write_rows(config.artifact_root / f"size_prior_{stage}.parquet", observations)
    write_json(
        config.artifact_root / f"size_prior_{stage}_gate.json", validation
    )
    if stage == "final48":
        write_json(
            config.artifact_root / "size_prior_final48_bootstrap.json",
            validation["bootstrap"],
        )
    return {"report": report, "validation": validation}


def _summarize_oracle_results(
    config: CleanExperimentConfig,
    scene_ids: Sequence[str],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_count = 0
    geometry_025 = 0
    geometry_050 = 0
    same_025 = 0
    same_050 = 0
    status_changes = 0
    comparable_decisions = 0
    available = 0
    scene_rows: list[dict[str, Any]] = []
    for scene_id, result in zip(scene_ids, results):
        evaluation = result.get("candidate_evaluation")
        if not isinstance(evaluation, Mapping) or not isinstance(
            evaluation.get("aggregate"), Mapping
        ):
            continue
        aggregate = evaluation["aggregate"]
        available += 1
        candidate_count += int(aggregate["candidate_count"])
        geometry_025 += int(aggregate["geometry_iou_025_count"])
        geometry_050 += int(aggregate["geometry_iou_050_count"])
        same_025 += int(aggregate["same_class_iou_025_count"])
        same_050 += int(aggregate["same_class_iou_050_count"])
        oracle_table = {
            tuple(map(int, row["mask_ids"])): bool(row["accepted"])
            for row in result.get("size_merge_decisions", [])
        }
        uniform_path = config.condition_dir(scene_id, "U-global") / "diagnostics.json"
        if uniform_path.is_file():
            uniform = load_json(uniform_path)
            uniform_table = {
                tuple(map(int, row["mask_ids"])): bool(row["accepted"])
                for row in uniform.get("size_merge_decisions", [])
            }
            common = set(oracle_table).intersection(uniform_table)
            changed = sum(oracle_table[key] != uniform_table[key] for key in common)
            status_changes += changed
            comparable_decisions += len(common)
        else:
            changed = 0
        scene_rows.append(
            {
                "scene_id": scene_id,
                "candidate_count": int(aggregate["candidate_count"]),
                "geometry_iou_025_count": int(aggregate["geometry_iou_025_count"]),
                "geometry_iou_050_count": int(aggregate["geometry_iou_050_count"]),
                "same_class_iou_025_count": int(
                    aggregate["same_class_iou_025_count"]
                ),
                "same_class_iou_050_count": int(
                    aggregate["same_class_iou_050_count"]
                ),
                "uniform_oracle_merge_status_changes": changed,
            }
        )
    return {
        "schema": "saga-clean-alpha-mask-oracle-size-summary-v1",
        "evaluation_only": True,
        "formal_output_written": False,
        "formal_ap_included": False,
        "scene_ids": list(map(str, scene_ids)),
        "available_scene_count": available,
        "candidate_count": candidate_count,
        "geometry_iou_025_count": geometry_025,
        "geometry_iou_050_count": geometry_050,
        "same_class_iou_025_count": same_025,
        "same_class_iou_050_count": same_050,
        "uniform_oracle_comparable_merge_decisions": comparable_decisions,
        "uniform_oracle_merge_status_change_count": status_changes,
        "scenes": scene_rows,
    }


def run_clean_baseline_experiment(
    config: CleanExperimentConfig,
    hooks: CleanExperimentHooks | None = None,
) -> dict[str, Any]:
    """Run or resume the registered DEV2-to-final48 experiment."""

    config.validate()
    production_runtime = hooks is None
    if production_runtime:
        # This gate intentionally precedes state loading and the terminal-state
        # early return.  A stopped/complete run must not be reported from a
        # different commit or a dirty deployment, and a mid-run restart must
        # not skip the provenance check merely because Stage 0 already passed.
        _validate_deployment_environment(config)
    hooks = hooks or default_hooks(config)
    state = _load_state(config)
    if state["status"] in {"stopped", "complete"}:
        return state

    def guarded(stage: str, function: Callable[[], Mapping[str, Any]]) -> Mapping[str, Any]:
        registered_stage = ACTIVE_NEXT_STAGE.get(
            str(state["checkpoint"]), stage
        )
        _begin(config, state, registered_stage)
        if stage != registered_stage:
            # Record conditional diagnostic progress without turning it into a
            # resumable checkpoint.  A hard kill can therefore only replay the
            # enclosing registered stage, never jump into a half-run control.
            state["active_substage"] = stage
            _write_state(config, state)
        try:
            return function()
        except BaseException as exc:
            # Conditional diagnostics (currently only identity-edge-control)
            # are nested inside a preregistered main stage.  The controller
            # dispatches from checkpoints, not arbitrary next_stage strings,
            # so an interruption must return to that checkpoint's registered
            # stage rather than persist an unreachable substage.
            retry_stage = ACTIVE_NEXT_STAGE.get(
                str(state["checkpoint"]), stage
            )
            state.pop("active_substage", None)
            state.update(
                status="active",
                current_stage=None,
                next_stage=retry_stage,
                last_error={"type": type(exc).__name__, "message": str(exc)},
            )
            _write_state(config, state)
            raise

    if state["checkpoint"] == "initialized":
        result = guarded("validated", lambda: _validate_registered_inputs(config))
        _complete(config, state, "validated", result, "dev2-evidence")

    if state["checkpoint"] == "validated":
        def build_dev2() -> Mapping[str, Any]:
            availability = _prepare_registered_scenes(config, config.dev2)
            if not availability["available"]:
                return {
                    "scene_ids": list(config.dev2),
                    "results": [],
                    "input_availability": availability,
                }
            rows = []
            for scene_id in config.dev2:
                spec = config.prepared_scene_spec(scene_id)
                # Formal evidence receives no GT path, transform, or class oracle.
                rows.append(
                    hooks.build_evidence(
                        scene_id=scene_id,
                        request_path=spec.evidence_request,
                        output_dir=config.bank_dir(scene_id),
                    )
                )
            return {
                "scene_ids": list(config.dev2),
                "results": rows,
                "input_availability": availability,
            }
        result = guarded("dev2-evidence", build_dev2)
        if not result["input_availability"]["available"]:
            return _stop(
                config,
                state,
                "dev2-inputs-unavailable",
                "DEV2 scene assets are unavailable at their stage; no geometry or category prior was tested",
                result,
            )
        _complete(config, state, "dev2-evidence", result, "dev2-geometry-oracle")

    if state["checkpoint"] == "dev2-evidence":
        def oracle_dev2() -> Mapping[str, Any]:
            rows = [
                hooks.geometry_oracle(
                    scene_id=scene_id,
                    bank_dir=config.bank_dir(scene_id),
                    scene_spec=config.prepared_scene_spec(scene_id),
                    output_path=config.artifact_root
                    / "geometry_oracle"
                    / f"{scene_id}.json",
                )
                for scene_id in config.dev2
            ]
            gate = _aggregate_oracle_gate(rows)
            write_json(
                config.artifact_root / "alpha_mask_geometry_oracle_dev2.json",
                {"scene_ids": list(config.dev2), "scenes": rows, "gate": gate},
            )
            return {"gate": gate}
        result = guarded("dev2-geometry-oracle", oracle_dev2)
        if not result["gate"]["passed"]:
            return _stop(
                config,
                state,
                "dev2-geometry-gate-failed",
                "complete 2D masks / alpha lifting do not provide enough geometric support; category prior was not tested",
                result,
            )
        _complete(config, state, "dev2-geometry-oracle", result, "dev2-c0-u")

    if state["checkpoint"] == "dev2-geometry-oracle":
        def run_dev2_c0_u() -> Mapping[str, Any]:
            for scene_id in config.dev2:
                for condition in ("C0-no-prior", "U-global"):
                    # This formal hook is intentionally GT-free.
                    hooks.run_formal(
                        scene_id=scene_id,
                        bank_dir=config.bank_dir(scene_id),
                        condition=condition,
                        output_dir=config.condition_dir(scene_id, condition),
                        priors_path=(
                            None if condition == "C0-no-prior" else config.category_priors
                        ),
                        allowed_classes=config.evaluation_class_names,
                    )
            report = hooks.evaluate_stage(
                scene_ids=config.dev2,
                conditions=("C0-no-prior", "U-global"),
                output_path=config.artifact_root / "dev2_c0_u_evaluation.json",
            )
            return {"report": report, "gate": _dev2_uniform_gate(report)}
        result = guarded("dev2-c0-u", run_dev2_c0_u)
        if not result["gate"]["passed"]:
            return _stop(
                config,
                state,
                "dev2-uniform-gate-failed",
                "the clean uniform consensus is less safe than the no-prior geometry; category prior was not tested",
                result,
            )
        _complete(config, state, "dev2-c0-u", result, "dev2-prior")

    if state["checkpoint"] == "dev2-c0-u":
        def run_dev2_prior() -> Mapping[str, Any]:
            oracle_results = []
            for scene_id in config.dev2:
                hooks.run_formal(
                    scene_id=scene_id,
                    bank_dir=config.bank_dir(scene_id),
                    condition="D-predicted",
                    output_dir=config.condition_dir(scene_id, "D-predicted"),
                    priors_path=config.category_priors,
                    allowed_classes=config.evaluation_class_names,
                )
                # GT enters only this explicitly offline path.
                oracle_results.append(hooks.run_oracle(
                    scene_id=scene_id,
                    bank_dir=config.bank_dir(scene_id),
                    scene_spec=config.prepared_scene_spec(scene_id),
                    priors_path=config.category_priors,
                    output_path=config.oracle_path(scene_id),
                ))
            report = hooks.evaluate_stage(
                scene_ids=config.dev2,
                conditions=FORMAL_CONDITIONS,
                output_path=config.artifact_root / "dev2_prior_evaluation.json",
            )
            oracle_summary = _summarize_oracle_results(
                config, config.dev2, oracle_results
            )
            write_json(
                config.artifact_root / "size_prior_oracle_dev2.json",
                {"summary": oracle_summary, "scene_results": oracle_results},
            )
            return {
                "report": report,
                "oracle_summary": oracle_summary,
                "gate": _prior_mechanical_gate(report),
            }
        result = guarded("dev2-prior", run_dev2_prior)
        state["prior_mechanical_intervention_tested"] = True
        if not result["gate"]["passed"]:
            return _stop(
                config,
                state,
                "dev2-prior-intervention-inactive",
                "the registered size prior did not change enough merge decisions; this is not evidence that category priors are ineffective",
                result,
            )
        _complete(config, state, "dev2-prior", result, "dev8-evidence")

    if state["checkpoint"] == "dev2-prior":
        def build_dev8() -> Mapping[str, Any]:
            availability = _prepare_registered_scenes(config, config.dev8)
            if not availability["available"]:
                return {
                    "scene_ids": list(config.dev8),
                    "results": [],
                    "geometry_gate": {"passed": False},
                    "input_availability": availability,
                }
            rows = []
            oracle_rows = []
            for scene_id in config.dev8:
                spec = config.prepared_scene_spec(scene_id)
                rows.append(
                    hooks.build_evidence(
                        scene_id=scene_id,
                        request_path=spec.evidence_request,
                        output_dir=config.bank_dir(scene_id),
                    )
                )
                oracle_rows.append(
                    hooks.geometry_oracle(
                        scene_id=scene_id,
                        bank_dir=config.bank_dir(scene_id),
                        scene_spec=spec,
                        output_path=config.artifact_root
                        / "geometry_oracle"
                        / f"{scene_id}.json",
                    )
                )
            oracle_gate = _aggregate_oracle_gate_dev8(oracle_rows)
            write_json(
                config.artifact_root / "alpha_mask_geometry_oracle_dev8.json",
                {
                    "scene_ids": list(config.dev8),
                    "scenes": oracle_rows,
                    "gate": oracle_gate,
                },
            )
            return {
                "scene_ids": list(config.dev8),
                "results": rows,
                "geometry_gate": oracle_gate,
                "input_availability": availability,
            }
        result = guarded("dev8-evidence", build_dev8)
        if not result["input_availability"]["available"]:
            return _stop(
                config,
                state,
                "dev8-inputs-unavailable",
                "one or more DEV8 scene inputs are unavailable at the DEV8 stage",
                result,
            )
        state["dev8_geometry_gate"] = _json_safe(result["geometry_gate"])
        _complete(config, state, "dev8-evidence", result, "dev8-uniform")

    if state["checkpoint"] == "dev8-evidence":
        def run_dev8_uniform() -> Mapping[str, Any]:
            for scene_id in config.dev8:
                for condition in ("C0-no-prior", "U-global"):
                    hooks.run_formal(
                        scene_id=scene_id,
                        bank_dir=config.bank_dir(scene_id),
                        condition=condition,
                        output_dir=config.condition_dir(scene_id, condition),
                        priors_path=(
                            None if condition == "C0-no-prior" else config.category_priors
                        ),
                        allowed_classes=config.evaluation_class_names,
                    )
            report = hooks.evaluate_stage(
                scene_ids=config.dev8,
                conditions=("C0-no-prior", "U-global"),
                output_path=config.artifact_root / "alpha_mask_consensus_dev8_uniform.json",
            )
            return {
                "report": report,
                "uniform_gate": _dev8_health_gate(config, report),
            }
        result = guarded("dev8-uniform", run_dev8_uniform)
        write_rows(
            config.artifact_root / "alpha_mask_consensus_dev8.parquet",
            result["report"].get("rows", []),
        )
        if not result["uniform_gate"]["passed"]:
            geometry_healthy = bool(
                state.get("dev8_geometry_gate", {}).get("passed", False)
            )
            identity_result: Mapping[str, Any] | None = None
            if geometry_healthy and config.identity_control is not None:
                if hooks.run_identity_control is None:
                    raise RuntimeError(
                        "a registered identity control requires an explicit hook"
                    )
                identity_result = guarded(
                    "identity-edge-control",
                    lambda: hooks.run_identity_control(
                        output_path=config.artifact_root
                        / "identity_edge_control.json"
                    ),
                )
                state["identity_control_run"] = True
                result = {**dict(result), "identity_control": identity_result}
            registration_issues = (
                list(config.identity_control_registration.get("issues", []))
                if isinstance(config.identity_control_registration, Mapping)
                else []
            )
            reason = (
                "the DEV8 complete-mask geometry ceiling is low; 2D masks / alpha lifting are the limiting input and category prior was not tested"
                if not geometry_healthy
                else (
                    (
                        "clean geometry upper bound was healthy but automatic cross-view consensus failed; the fixed held-out hard-edge set contained only one label class, so the isolated identity-edge capacity control was inconclusive (capacity was neither proven nor disproven) and category prior was not tested"
                        if identity_result.get("control_status") == "inconclusive"
                        else "clean geometry upper bound was healthy but automatic cross-view consensus failed; the isolated identity-edge capacity control completed and did not enter the formal U/D method; category prior was not tested"
                    )
                    if identity_result is not None
                    else (
                        "clean geometry upper bound was healthy but automatic cross-view consensus failed; the one permitted identity-edge control was not run because its registered existing assets were unavailable: "
                        + "; ".join(registration_issues)
                        + "; category prior was not tested"
                        if registration_issues
                        else "clean geometry upper bound was healthy but automatic cross-view consensus failed; the one permitted identity-edge control was not registered and category prior was not tested"
                    )
                )
            )
            return _stop(
                config, state, "dev8-uniform-gate-failed", reason, result
            )
        _complete(config, state, "dev8-uniform", result, "dev8-prior")

    if state["checkpoint"] == "dev8-uniform":
        def run_dev8_prior() -> Mapping[str, Any]:
            oracle_results = []
            for scene_id in config.dev8:
                hooks.run_formal(
                    scene_id=scene_id,
                    bank_dir=config.bank_dir(scene_id),
                    condition="D-predicted",
                    output_dir=config.condition_dir(scene_id, "D-predicted"),
                    priors_path=config.category_priors,
                    allowed_classes=config.evaluation_class_names,
                )
                oracle_results.append(
                    hooks.run_oracle(
                        scene_id=scene_id,
                        bank_dir=config.bank_dir(scene_id),
                        scene_spec=config.prepared_scene_spec(scene_id),
                        priors_path=config.category_priors,
                        output_path=config.oracle_path(scene_id),
                    )
                )
            report = hooks.evaluate_stage(
                scene_ids=config.dev8,
                conditions=FORMAL_CONDITIONS,
                output_path=config.artifact_root / "alpha_mask_consensus_dev8.json",
            )
            oracle_summary = _summarize_oracle_results(
                config, config.dev8, oracle_results
            )
            write_json(
                config.artifact_root / "size_prior_oracle_dev8.json",
                {"summary": oracle_summary, "scene_results": oracle_results},
            )
            return {
                "report": report,
                "oracle_summary": oracle_summary,
                "prior_gate": _dev8_prior_gate(report),
            }
        result = guarded("dev8-prior", run_dev8_prior)
        state["candidate_prior_tested"] = True
        write_rows(
            config.artifact_root / "size_prior_dev8.parquet",
            result["report"].get("prior_rows", []),
        )
        if not result["prior_gate"]["passed"]:
            return _stop(
                config,
                state,
                "dev8-prior-gate-failed",
                "the clean baseline is healthy, but predicted-class size priors did not meet the preregistered benefit gate",
                result,
            )
        _complete(config, state, "dev8-prior", result, "holdout5-evidence")

    if state["checkpoint"] == "dev8-prior":
        result = guarded(
            "holdout5-evidence",
            lambda: _build_registered_evidence(config, hooks, config.holdout5),
        )
        if not result["input_availability"]["available"]:
            return _stop(
                config, state, "holdout5-inputs-unavailable",
                "one or more holdout5 inputs are unavailable at the holdout stage",
                result,
            )
        _complete(
            config, state, "holdout5-evidence", result, "holdout5-evaluate"
        )

    if state["checkpoint"] == "holdout5-evidence":
        result = guarded(
            "holdout5-evaluate",
            lambda: _run_registered_paired_stage(
                config, hooks, scene_ids=config.holdout5, stage="holdout5"
            ),
        )
        if not result["validation"]["passed"]:
            return _stop(
                config,
                state,
                "holdout5-gate-failed",
                "predicted-class size prior did not replicate on the five canonical holdout physical scenes",
                result,
            )
        _complete(config, state, "holdout5-evaluate", result, "tune24-evidence")

    if state["checkpoint"] == "holdout5-evaluate":
        result = guarded(
            "tune24-evidence",
            lambda: _build_registered_evidence(config, hooks, config.tune24),
        )
        if not result["input_availability"]["available"]:
            return _stop(
                config, state, "tune24-inputs-unavailable",
                "one or more tune24 inputs are unavailable at the tune stage",
                result,
            )
        _complete(config, state, "tune24-evidence", result, "tune24-evaluate")

    if state["checkpoint"] == "tune24-evidence":
        result = guarded(
            "tune24-evaluate",
            lambda: _run_registered_paired_stage(
                config, hooks, scene_ids=config.tune24, stage="tune24"
            ),
        )
        if not result["validation"]["passed"]:
            return _stop(
                config,
                state,
                "tune24-gate-failed",
                "predicted-class size prior did not reach the preregistered physical-scene macro improvement on tune24",
                result,
            )
        _complete(config, state, "tune24-evaluate", result, "final48-evidence")

    if state["checkpoint"] == "tune24-evaluate":
        result = guarded(
            "final48-evidence",
            lambda: _build_registered_evidence(config, hooks, config.final48),
        )
        if not result["input_availability"]["available"]:
            return _stop(
                config, state, "final48-inputs-unavailable",
                "one or more locked final48 inputs are unavailable at the final stage",
                result,
            )
        _complete(config, state, "final48-evidence", result, "final48-evaluate")

    if state["checkpoint"] == "final48-evidence":
        result = guarded(
            "final48-evaluate",
            lambda: _run_registered_paired_stage(
                config, hooks, scene_ids=config.final48, stage="final48"
            ),
        )
        if not result["validation"]["passed"]:
            return _stop(
                config,
                state,
                "final48-gate-failed",
                "final48 did not satisfy both the 0.002 effect-size floor and the positive paired-bootstrap confidence bound",
                result,
            )
        state.update(
            status="complete",
            checkpoint="final48-complete",
            current_stage=None,
            next_stage=None,
            stop_reason=None,
        )
        state["history"].append(
            {"stage": "final48-evaluate", **_history_safe(result)}
        )
        _write_state(config, state)
    return state


def _validate_deployment_environment(
    config: CleanExperimentConfig,
) -> dict[str, Any]:
    """Revalidate immutable code and train-only priors on every invocation."""

    prior_payload = load_json(config.category_priors)
    if not isinstance(prior_payload, Mapping):
        raise TypeError("category priors must be a JSON object")
    validate_priors(prior_payload)
    if prior_payload.get("provenance", {}).get("splits") != ["train"]:
        raise ValueError("formal category priors must be fitted from train only")
    provenance = build_clean_baseline_provenance(
        repo_root=config.repo_root,
        output_path=None,
    )
    if str(provenance.get("current_commit")) != config.code_commit:
        raise ValueError(
            "registered code_commit does not match the deployed repository HEAD"
        )
    imported_rows: list[dict[str, Any]] = []
    for scene_id, registration in sorted(config.evidence_imports.items()):
        bank = _load_imported_evidence_bank(
            config,
            scene_id=scene_id,
            request_path=config.scenes[scene_id].evidence_request,
            output_dir=registration.bank_dir,
        )
        imported_rows.append(
            {
                "scene_id": scene_id,
                "producer_commit": registration.producer_commit,
                "mask_count": int(bank.mask_count),
                "status": "validated-read-only",
            }
        )
    # Invalid deployments fail without overwriting the last valid provenance.
    write_json(
        config.artifact_root / "clean_baseline_provenance.json", provenance
    )
    return {
        "current_commit": config.code_commit,
        "tracked_worktree_clean": True,
        "category_priors_sha256": sha256_file(config.category_priors),
        "category_prior_splits": ["train"],
        "imported_evidence": imported_rows,
    }


def _validate_registered_inputs(config: CleanExperimentConfig) -> dict[str, Any]:
    config.validate()
    missing: list[str] = []
    registered_scene_ids = tuple(
        sorted(set(config.tune24).union(config.final48))
    )
    # Per-scene GT, PLY, COLMAP and mask assets are intentionally absent from
    # this global check.  A locked final48 disk can be offline while DEV2 is
    # running; each scene is validated immediately before its own stage.
    for scene_id in registered_scene_ids:
        spec = config.scenes[scene_id]
        if not spec.evidence_request.is_file():
            missing.append(str(spec.evidence_request))
    if not config.category_priors.is_file():
        missing.append(str(config.category_priors))
    if config.identity_control is not None:
        for asset in config.identity_control.assets.values():
            for path in (asset.feature_ply, asset.gaussian_ply):
                if not path.is_file():
                    missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"registered inputs are missing: {missing}")
    deployment = _validate_deployment_environment(config)
    provenance = load_json(
        config.artifact_root / "clean_baseline_provenance.json"
    )
    for scene_id in registered_scene_ids:
        request = load_json(config.scenes[scene_id].evidence_request)
        expected_producer = (
            config.evidence_imports[scene_id].producer_commit
            if scene_id in config.evidence_imports
            else config.code_commit
        )
        if request.get("producer_commit") != expected_producer:
            raise ValueError(
                f"{scene_id}: evidence request producer_commit does not exactly "
                "match its registered evidence producer"
            )
        if tuple(map(str, request.get("classes", ()))) != (
            config.evidence_class_names
        ):
            raise ValueError(
                f"{scene_id}: evidence request does not use the registered "
                "32-class evidence vocabulary"
            )
    sai3d_present = bool(config.sai3d_asset_paths) and all(
        path.exists() for path in config.sai3d_asset_paths
    )
    sai3d_audit = {
        "status": "present-not-run" if sai3d_present else "skipped-missing-assets",
        "asset_paths": [str(path) for path in config.sai3d_asset_paths],
        "download_attempted": False,
        "download_allowed": False,
        "external_anchor_run": False,
    }
    write_json(config.artifact_root / "sai3d_asset_audit.json", sai3d_audit)
    return {
        "scene_count": len(registered_scene_ids),
        "dev_physical_scene_count": len(
            {_physical_scene_id(value) for value in config.dev8}
        ),
        "physical_split_overlap": False,
        "missing_inputs": [],
        "gt_as_prediction_parity": "deferred-to-dev2-input-preflight",
        "provenance_schema": provenance["schema"],
        "deployment": deployment,
        "sai3d": sai3d_audit,
    }


def _prepare_registered_scene(
    config: CleanExperimentConfig, scene_id: str
) -> dict[str, Any]:
    """Validate and, if permitted, complete one scene exactly at first use."""

    from .materialize_config import _prove_30k_ply, _tiny_small_instance_ids
    from .worker import resolve_clean_scene_inputs

    base = config.scenes[scene_id]
    existing_path = config.scene_input_path(scene_id)
    if existing_path.is_file():
        existing = load_json(existing_path)
        sam_row = existing.get("sam", {}) if isinstance(existing, Mapping) else {}
        audit = sam_row.get("audit", {}) if isinstance(sam_row, Mapping) else {}
        prepared_existing = Path(
            str(existing.get("prepared_request", ""))
            if isinstance(existing, Mapping)
            else ""
        )
        manifest_ok = False
        content_ok = False
        if prepared_existing.is_file():
            prepared_payload = load_json(prepared_existing)
            manifest = prepared_payload.get("sam_frame_manifest", [])
            if isinstance(manifest, list) and manifest:
                manifest_ok = all(
                    isinstance(row, Mapping)
                    and Path(str(row.get("path", ""))).is_file()
                    and sha256_file(Path(str(row["path"])))
                    == str(row.get("content_sha256", ""))
                    for row in manifest
                )
            elif sam_row.get("source") == "synthetic-test":
                # Unit-test hooks use no real camera assets.  Production rows
                # can never select this source.
                manifest_ok = True
            content_identity = existing.get("content_identity", {})
            content_ok = isinstance(content_identity, Mapping) and dict(
                content_identity
            ) == {
                "prepared_request_sha256": sha256_file(prepared_existing),
                "gt_npz_sha256": (
                    sha256_file(base.gt_npz) if base.gt_npz.is_file() else None
                ),
                "gaussian_ply_sha256": (
                    sha256_file(base.gaussian_ply)
                    if base.gaussian_ply.is_file()
                    else None
                ),
            }
        complete_registration = (
            isinstance(existing, Mapping)
            and existing.get("schema") == SCENE_INPUT_REGISTRATION_SCHEMA
            and existing.get("status") == "complete"
            and existing.get("scene_id") == scene_id
            and existing.get("code_commit") == config.code_commit
        )
        if (
            complete_registration
            and bool(audit.get("complete", False))
            and prepared_existing.is_file()
            and manifest_ok
            and content_ok
            and base.gt_npz.is_file()
            and base.gaussian_ply.is_file()
        ):
            return dict(existing)
        if complete_registration:
            # Once a scene has contributed to a completed gate, silently
            # rebuilding it from changed bytes would mix two datasets inside
            # one recoverable experiment.  Treat any identity/manifest drift
            # as immutable-input corruption and require a fresh registered
            # run instead of rewriting the old registration in place.
            raise ValueError(
                f"{scene_id}: completed scene input registration changed or "
                "became incomplete"
            )
    try:
        request = load_json(base.evidence_request)
        if not isinstance(request, Mapping):
            raise TypeError("evidence request must be an object")
        scene_value = request.get("scene")
        if not isinstance(scene_value, Mapping):
            raise TypeError("evidence request.scene must be an object")
        scene = dict(scene_value)
        inputs = resolve_clean_scene_inputs(scene, require_exists=False)
        for path in (inputs.rgb_ply, inputs.sparse, inputs.images):
            if not path.exists():
                raise FileNotFoundError(path)
        frames = colmap_frame_specs(inputs.sparse)
        sam = ensure_scene_sam_masks(
            frames=frames,
            images_root=inputs.images,
            primary_root=inputs.sam_masks,
            grounded_masks_root=inputs.grounded_masks,
            grounded_labels_root=inputs.grounded_labels,
            generation=(
                request.get("sam_generation")
                if isinstance(request.get("sam_generation"), Mapping)
                else None
            ),
        )
        if sam.get("status") != "complete":
            row = {
                "schema": SCENE_INPUT_REGISTRATION_SCHEMA,
                "status": "unavailable",
                "scene_id": scene_id,
                "code_commit": config.code_commit,
                "reason": str(sam.get("reason", "SAM input is unavailable")),
                "sam": sam,
            }
            write_json(config.scene_input_path(scene_id), row)
            return row
        scene["segment_everything_root"] = str(Path(str(sam["sam_root"])).resolve())
        prepared_request = dict(request)
        prepared_request["scene"] = scene
        prepared_request["resolved_sam_input"] = sam
        sam_root = Path(str(sam["sam_root"])).resolve()
        prepared_request["sam_frame_manifest"] = [
            {
                "image_name": frame.image_name,
                "path": str((sam_root / f"{frame.image_name}.npz").resolve()),
                "content_sha256": sha256_file(
                    sam_root / f"{frame.image_name}.npz"
                ),
            }
            for frame in frames
        ]
        prepared_path = config.prepared_request_path(scene_id)
        write_json(prepared_path, prepared_request)
        prepared_inputs = resolve_clean_scene_inputs(scene, require_exists=True)
        _prove_30k_ply(scene_id, scene, prepared_inputs.rgb_ply)
        if prepared_inputs.rgb_ply != base.gaussian_ply:
            raise ValueError("prepared Gaussian PLY differs from frozen registration")
        if not base.gt_npz.is_file():
            raise FileNotFoundError(base.gt_npz)
        tiny_small, tiny_report = _tiny_small_instance_ids(
            scene_id=scene_id,
            gt_npz=base.gt_npz,
            gaussian_ply=base.gaussian_ply,
            transform=base.gaussian_to_gt_transform,
            size_bins={"boundaries_m": dict(config.size_bin_boundaries_m)},
            evaluation_class_count=len(config.evaluation_class_names),
            radius_m=config.radius_m,
            min_region_size=config.min_region_size,
        )
        source = evidence_request_source(scene_id=scene_id, request=prepared_request)
        expected_producer = (
            config.evidence_imports[scene_id].producer_commit
            if scene_id in config.evidence_imports
            else config.code_commit
        )
        if source.get("producer_commit") != expected_producer:
            raise ValueError("prepared evidence producer identity changed")
        parity_scene = load_ground_truth_npz(base.gt_npz, scene_id)[1]
        parity = evaluate_ground_truth_parity(
            [parity_scene],
            config.evaluation_class_names,
            min_region_size=config.min_region_size,
        )
        row = {
            "schema": SCENE_INPUT_REGISTRATION_SCHEMA,
            "status": "complete",
            "scene_id": scene_id,
            "code_commit": config.code_commit,
            "prepared_request": str(prepared_path),
            "gt_npz": str(base.gt_npz),
            "gaussian_ply": str(base.gaussian_ply),
            "content_identity": {
                "prepared_request_sha256": sha256_file(prepared_path),
                "gt_npz_sha256": sha256_file(base.gt_npz),
                "gaussian_ply_sha256": sha256_file(base.gaussian_ply),
            },
            "tiny_small_instance_ids": list(tiny_small),
            "tiny_small_diagnostics": tiny_report,
            "gt_as_prediction_parity": bool(parity["gt_as_prediction_parity"]),
            "sam": sam,
        }
        write_json(config.scene_input_path(scene_id), row)
        return row
    except (FileNotFoundError, ImportError, KeyError, OSError, TypeError, ValueError) as exc:
        row = {
            "schema": SCENE_INPUT_REGISTRATION_SCHEMA,
            "status": "unavailable",
            "scene_id": scene_id,
            "code_commit": config.code_commit,
            "reason": f"{type(exc).__name__}: {exc}",
            "download_attempted": False,
        }
        write_json(config.scene_input_path(scene_id), row)
        return row


def _prepare_registered_scenes(
    config: CleanExperimentConfig, scene_ids: Sequence[str]
) -> dict[str, Any]:
    rows = [_prepare_registered_scene(config, str(scene_id)) for scene_id in scene_ids]
    unavailable = [row for row in rows if row.get("status") != "complete"]
    return {
        "status": "unavailable" if unavailable else "complete",
        "available": not unavailable,
        "scene_ids": list(map(str, scene_ids)),
        "unavailable": unavailable,
        "scenes": rows,
        "stage_local_validation": True,
    }


def _scene_gt_adapter(
    config: CleanExperimentConfig,
    scene_id: str,
    bank_dir: Path,
) -> tuple[Any, list[Any], np.ndarray, dict[str, Any]]:
    spec = config.prepared_scene_spec(scene_id)
    bank = load_evidence_bank(bank_dir, expected_scene_id=scene_id)
    gt_coords, gt_scene = load_ground_truth_npz(spec.gt_npz, scene_id)
    gaussian_xyz = apply_transform(
        load_ply_xyz(spec.gaussian_ply), spec.gaussian_to_gt_transform
    )
    if len(gaussian_xyz) != bank.point_count:
        raise ValueError(f"{scene_id}: evidence/PLY Gaussian count mismatch")
    _verify_metric_geometry_identity(bank.xyz_m, gaussian_xyz, scene_id=scene_id)
    mapping, diagnostics = gt_point_to_gaussian_mapping(
        gt_coords, gaussian_xyz, radius_m=config.radius_m
    )
    if float(diagnostics["mapped_fraction"]) < 0.90:
        raise ValueError(f"{scene_id}: coordinate alignment gate failed")
    objects = ground_truth_objects_from_arrays(
        gt_scene.semantic,
        gt_scene.instance,
        class_names=config.evaluation_class_names,
        min_region_size=config.min_region_size,
        tiny_small_instance_ids=set(spec.tiny_small_instance_ids),
    )
    return gt_scene, objects, mapping, diagnostics


def _verify_metric_geometry_identity(
    bank_xyz_m: np.ndarray,
    transformed_ply_xyz_m: np.ndarray,
    *,
    scene_id: str,
    max_samples: int = 64,
    atol_m: float = 1e-4,
) -> dict[str, Any]:
    """Verify point order and metric scale without requiring one world frame.

    Pair distances are invariant to rigid rotation and translation.  Sampling
    stable Gaussian indices also catches a reordered or differently scaled PLY
    while keeping the audit bounded for million-point scenes.
    """

    bank = np.asarray(bank_xyz_m, dtype=np.float64)
    ply = np.asarray(transformed_ply_xyz_m, dtype=np.float64)
    if bank.ndim != 2 or bank.shape[1:] != (3,) or bank.shape != ply.shape:
        raise ValueError(f"{scene_id}: evidence/PLY metric XYZ shape mismatch")
    if not np.isfinite(bank).all() or not np.isfinite(ply).all() or len(bank) == 0:
        raise ValueError(f"{scene_id}: evidence/PLY metric XYZ must be finite")
    sample_count = min(int(max_samples), len(bank))
    sample_ids = np.linspace(0, len(bank) - 1, num=sample_count, dtype=np.int64)
    bank_sample = bank[sample_ids]
    ply_sample = ply[sample_ids]
    bank_distances = np.linalg.norm(
        bank_sample[:, None, :] - bank_sample[None, :, :], axis=2
    )
    ply_distances = np.linalg.norm(
        ply_sample[:, None, :] - ply_sample[None, :, :], axis=2
    )
    errors = np.abs(bank_distances - ply_distances)
    maximum_error = float(errors.max(initial=0.0))
    if not np.allclose(bank_distances, ply_distances, atol=atol_m, rtol=1e-4):
        raise ValueError(
            f"{scene_id}: evidence-bank metric geometry differs from transformed "
            f"PLY (sample pair-distance max error {maximum_error:.6g} m)"
        )
    return {
        "sample_count": sample_count,
        "pair_distance_max_abs_error_m": maximum_error,
        "translation_rotation_invariant": True,
    }


def _build_offline_oracle_identity(
    config: CleanExperimentConfig,
    *,
    scene_id: str,
    bank_dir: str | Path,
    artifact_kind: str,
) -> dict[str, Any]:
    """Build the exact, embedded cache boundary for a GT-only artifact.

    Offline diagnostics are resumable, but only when every input that can
    affect the answer is byte-identical.  In particular, a path is not an
    identity: GT, Gaussian geometry, evidence files and train-only priors are
    hashed in place.  The returned self-hash is embedded in the JSON result;
    no SHA sidecar is created.
    """

    if artifact_kind not in {"geometry-oracle", "oracle-class-size"}:
        raise ValueError(f"unsupported offline oracle kind: {artifact_kind}")
    if scene_id not in config.scenes:
        raise KeyError(f"unregistered oracle scene: {scene_id}")
    if len(config.code_commit) != 40 or any(
        value not in "0123456789abcdef" for value in config.code_commit
    ):
        raise ValueError("consumer commit must be a full lowercase git commit")
    spec = config.prepared_scene_spec(scene_id)
    root = Path(bank_dir)
    bank = load_evidence_bank(root, expected_scene_id=scene_id)
    taxonomy = load_taxonomy()
    if tuple(config.evaluation_class_names) != taxonomy.canonical_classes:
        raise ValueError("offline oracle evaluation taxonomy is not canonical")
    prior_payload = load_json(config.category_priors)
    if not isinstance(prior_payload, Mapping):
        raise TypeError("category priors must contain a JSON object")
    implementation = (
        GEOMETRY_ORACLE_IMPLEMENTATION
        if artifact_kind == "geometry-oracle"
        else ORACLE_SIZE_IMPLEMENTATION
    )
    identity: dict[str, Any] = {
        "schema": OFFLINE_ORACLE_IDENTITY_SCHEMA,
        "artifact_kind": artifact_kind,
        "implementation": implementation,
        "consumer_commit": config.code_commit,
        "scene_id": scene_id,
        "evidence": {
            "schema": bank.schema,
            "source": dict(bank.source),
            "scene_id": bank.scene_id,
            "point_count": bank.point_count,
            "frame_count": bank.frame_count,
            "mask_count": bank.mask_count,
            "thresholds": bank.thresholds.to_dict(),
            "class_names": list(map(str, bank.class_names)),
            "files": {
                EVIDENCE_ARRAY_FILE: sha256_file(root / EVIDENCE_ARRAY_FILE),
                EVIDENCE_METADATA_FILE: sha256_file(root / EVIDENCE_METADATA_FILE),
                EVIDENCE_DIAGNOSTICS_FILE: sha256_file(
                    root / EVIDENCE_DIAGNOSTICS_FILE
                ),
            },
        },
        "ground_truth": {
            "content_sha256": sha256_file(spec.gt_npz),
            "gaussian_content_sha256": sha256_file(spec.gaussian_ply),
            "gaussian_to_gt_transform": [
                list(map(float, row)) for row in spec.gaussian_to_gt_transform
            ],
            "radius_m": float(config.radius_m),
            "min_region_size": int(config.min_region_size),
            "tiny_small_instance_ids": list(spec.tiny_small_instance_ids),
        },
        "evaluation_taxonomy": {
            "content_sha256": taxonomy.content_hash,
            "class_names": list(config.evaluation_class_names),
        },
        "category_priors": {
            "file_content_sha256": sha256_file(config.category_priors),
            "declared_content_sha256": prior_payload.get("content_sha256"),
        },
        "consensus_config": (
            None
            if artifact_kind == "geometry-oracle"
            else asdict(ConsensusConfig())
        ),
    }
    identity["content_sha256"] = hash_json(identity)
    return identity


def _offline_oracle_is_complete(
    path: str | Path,
    *,
    expected_schema: str,
    expected_identity: Mapping[str, Any],
) -> bool:
    try:
        payload = load_json(path)
        if payload.get("schema") != expected_schema:
            return False
        actual = validate_embedded_identity(
            payload.get("run_identity"),
            expected_schema=OFFLINE_ORACLE_IDENTITY_SCHEMA,
        )
        expected = validate_embedded_identity(
            expected_identity,
            expected_schema=OFFLINE_ORACLE_IDENTITY_SCHEMA,
        )
        return actual == expected
    except (FileNotFoundError, OSError, KeyError, TypeError, ValueError):
        return False


def _load_imported_evidence_bank(
    config: CleanExperimentConfig,
    *,
    scene_id: str,
    request_path: Path,
    output_dir: Path,
) -> Any:
    """Strictly validate an imported producer artifact without rebuilding it."""

    registration = config.evidence_imports.get(scene_id)
    if registration is None:
        raise KeyError(f"{scene_id}: evidence is not registered as imported")
    if output_dir.resolve() != registration.bank_dir:
        raise ValueError(f"{scene_id}: imported evidence bank path changed")
    request = load_json(request_path)
    expected_source = evidence_request_source(scene_id=scene_id, request=request)
    if expected_source.get("producer_commit") != registration.producer_commit:
        raise ValueError(f"{scene_id}: imported evidence producer identity changed")
    for name, expected_digest in registration.files.items():
        path = registration.bank_dir / name
        if not path.is_file():
            raise FileNotFoundError(
                f"{scene_id}: imported evidence file is missing: {path}"
            )
        actual_digest = sha256_file(path)
        if actual_digest != expected_digest:
            raise ValueError(
                f"{scene_id}: imported evidence byte identity changed for {name}"
            )
    if not evidence_bank_is_complete(
        registration.bank_dir,
        expected_scene_id=scene_id,
        expected_source=expected_source,
    ):
        raise ValueError(
            f"{scene_id}: imported evidence bank failed its source/schema contract"
        )
    bank = load_evidence_bank(
        registration.bank_dir,
        expected_scene_id=scene_id,
        expected_source=expected_source,
    )
    if str(bank.source.get("producer_commit", "")) != registration.producer_commit:
        raise ValueError(f"{scene_id}: imported evidence embedded producer changed")
    return bank


def _default_build_evidence(
    config: CleanExperimentConfig, **kwargs: Any
) -> Mapping[str, Any]:
    scene_id = str(kwargs["scene_id"])
    request_path = Path(kwargs["request_path"])
    request = load_json(request_path)
    expected = evidence_request_source(scene_id=scene_id, request=request)
    output = Path(kwargs["output_dir"])
    if scene_id in config.evidence_imports:
        bank = _load_imported_evidence_bank(
            config,
            scene_id=scene_id,
            request_path=request_path,
            output_dir=output,
        )
        return {
            "scene_id": scene_id,
            "status": "reused-imported",
            "producer_commit": config.evidence_imports[scene_id].producer_commit,
            "mask_count": bank.mask_count,
        }
    if evidence_bank_is_complete(
        output, expected_scene_id=scene_id, expected_source=expected
    ):
        bank = load_evidence_bank(
            output, expected_scene_id=scene_id, expected_source=expected
        )
        return {"scene_id": scene_id, "status": "reused", "mask_count": bank.mask_count}
    result = build_alpha_mask_evidence(
        scene_id=scene_id, request=request, output_dir=output
    )
    return {"scene_id": scene_id, "status": "built", **dict(result)}


def _default_geometry_oracle(
    config: CleanExperimentConfig, **kwargs: Any
) -> Mapping[str, Any]:
    scene_id = str(kwargs["scene_id"])
    output = Path(kwargs["output_path"])
    bank_dir = Path(kwargs["bank_dir"])
    bank = load_evidence_bank(bank_dir, expected_scene_id=scene_id)
    run_identity = _build_offline_oracle_identity(
        config,
        scene_id=scene_id,
        bank_dir=bank_dir,
        artifact_kind="geometry-oracle",
    )
    if _offline_oracle_is_complete(
        output,
        expected_schema=GEOMETRY_ORACLE_SCHEMA,
        expected_identity=run_identity,
    ):
        return {**load_json(output), "runner_status": "skipped-complete"}
    _, gt_objects, mapping, mapping_diagnostics = _scene_gt_adapter(
        config, scene_id, Path(kwargs["bank_dir"])
    )
    # Geometry oracles operate on the complete SAM mask identity.  Ambiguous
    # same-frame hierarchy points abstain only from association evidence; they
    # must not silently shrink the mask supplied to the oracle.
    supports = [
        bank.support_for_mask(mask.global_mask_id, include_ambiguous=True)[0]
        for mask in bank.masks
    ]
    result = evaluate_geometry_oracles(
        supports,
        gt_objects,
        mapping,
        mask_ids=[mask.global_mask_id for mask in bank.masks],
        mask_frame_ids=[mask.frame_id for mask in bank.masks],
        gaussian_count=bank.point_count,
    )
    result["schema"] = GEOMETRY_ORACLE_SCHEMA
    result["scene_id"] = scene_id
    result["mapping_diagnostics"] = mapping_diagnostics
    result["evidence_source"] = dict(bank.source)
    result["run_identity"] = run_identity
    write_json(output, result)
    if not _offline_oracle_is_complete(
        output,
        expected_schema=GEOMETRY_ORACLE_SCHEMA,
        expected_identity=run_identity,
    ):
        raise RuntimeError("geometry oracle did not satisfy its output contract")
    return {**result, "runner_status": "complete"}


def _default_run_formal(**kwargs: Any) -> Mapping[str, Any]:
    # This adapter intentionally has no access to scene specs or GT.
    return run_consensus_condition(**kwargs)


def _oracle_candidates(
    config: CleanExperimentConfig, scene_id: str, bank_dir: Path
) -> tuple[list[CleanCandidate], list[dict[str, Any]], list[dict[str, Any]]]:
    bank = load_evidence_bank(bank_dir, expected_scene_id=scene_id)
    _, gt_objects, mapping, _ = _scene_gt_adapter(config, scene_id, bank_dir)
    priors = SizePriorTable.from_category_priors(load_json(config.category_priors))
    observations = tuple(
        MaskObservation(
            mask_id=mask.global_mask_id,
            frame_id=mask.frame_id,
            gaussian_ids=bank.support_for_mask(mask.global_mask_id, include_ambiguous=True)[0],
            ambiguous_ids=(
                lambda row: row[0][row[3]]
            )(bank.support_for_mask(mask.global_mask_id, include_ambiguous=True)),
        )
        for mask in bank.masks
    )
    visibility = np.zeros((bank.frame_count, bank.point_count), dtype=np.bool_)
    for index, frame in enumerate(bank.frames):
        # Keep the offline oracle on the exact formal visibility protocol.  A
        # frame whose SAM geometry abstained may retain alpha visibility for
        # input auditing, but it supplied no mask observation and therefore
        # cannot count as negative evidence here either.
        if frame.geometry_abstained:
            continue
        ids, _ = bank.visibility_for_frame(frame.frame_id)
        visibility[index, ids] = True
    decisions: list[dict[str, Any]] = []

    def veto(mask_ids: tuple[int, ...], ids: np.ndarray) -> bool:
        projected = project_gaussian_support_to_gt_points(
            ids, mapping, gaussian_count=bank.point_count, include_unmapped_fp=True
        )
        overlaps = [
            (len(np.intersect1d(projected, gt.point_ids)), gt) for gt in gt_objects
        ]
        overlap, target = max(
            overlaps, key=lambda row: (row[0], -row[1].object_id), default=(0, None)
        )
        extents = pca_sorted_extents_m(bank.xyz_m[ids])
        compatibility = (
            global_size_compatibility(extents, priors)
            if target is None or overlap == 0
            else oracle_class_size_compatibility(extents, priors, str(target.class_id))
        )
        accepted = compatibility >= 0.50
        decisions.append(
            {
                "mask_ids": list(mask_ids),
                "oracle_class": (
                    None if target is None or overlap == 0 else str(target.class_id)
                ),
                "global_fallback": target is None or overlap == 0,
                "compatibility": compatibility,
                "accepted": accepted,
            }
        )
        return accepted

    result = run_mask_consensus(
        observations, visibility, bank.xyz_m, config=ConsensusConfig(), merge_veto=veto
    )
    candidates: list[CleanCandidate] = []
    for item in result.objects:
        projected = project_gaussian_support_to_gt_points(
            item.gaussian_ids,
            mapping,
            gaussian_count=bank.point_count,
            include_unmapped_fp=True,
        )
        overlaps = [
            (len(np.intersect1d(projected, gt.point_ids)), gt) for gt in gt_objects
        ]
        overlap, target = max(
            overlaps, key=lambda row: (row[0], -row[1].object_id), default=(0, None)
        )
        candidates.append(
            CleanCandidate(
                object_id=item.object_id,
                gaussian_ids=item.gaussian_ids,
                class_id=None if target is None or overlap == 0 else target.class_id,
                winner_probability=1.0 if target is not None and overlap else 0.0,
                view_consensus=item.mean_view_consensus,
                detection_ratio=item.mean_detection_ratio,
            )
        )
    accepted_edges = [
        {
            "left_mask_ids": list(edge.left_mask_ids),
            "right_mask_ids": list(edge.right_mask_ids),
            "observer_count": edge.observer_count,
            "supporter_count": edge.supporter_count,
            "consensus": edge.consensus,
        }
        for edge in result.accepted_edges
    ]
    return candidates, decisions, accepted_edges


def _default_run_oracle(
    config: CleanExperimentConfig, **kwargs: Any
) -> Mapping[str, Any]:
    scene_id = str(kwargs["scene_id"])
    output = Path(kwargs["output_path"])
    bank_dir = Path(kwargs["bank_dir"])
    run_identity = _build_offline_oracle_identity(
        config,
        scene_id=scene_id,
        bank_dir=bank_dir,
        artifact_kind="oracle-class-size",
    )
    if _offline_oracle_is_complete(
        output,
        expected_schema=ORACLE_SIZE_SCHEMA,
        expected_identity=run_identity,
    ):
        return {**load_json(output), "runner_status": "skipped-complete"}
    candidates, decisions, accepted_edges = _oracle_candidates(
        config, scene_id, bank_dir
    )
    _, gt_objects, mapping, _ = _scene_gt_adapter(
        config, scene_id, bank_dir
    )
    result = {
        "schema": ORACLE_SIZE_SCHEMA,
        "scene_id": scene_id,
        "evaluation_only": True,
        "formal_output_written": False,
        "run_identity": run_identity,
        "candidate_evaluation": evaluate_candidates(
            candidates,
            gt_objects,
            mapping,
            gaussian_count=load_evidence_bank(bank_dir).point_count,
        ),
        "size_merge_decisions": decisions,
        "accepted_edges": accepted_edges,
    }
    write_json(output, result)
    if not _offline_oracle_is_complete(
        output,
        expected_schema=ORACLE_SIZE_SCHEMA,
        expected_identity=run_identity,
    ):
        raise RuntimeError("oracle-class artifact did not satisfy its output contract")
    return {**result, "runner_status": "complete"}


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2:
        return 0.0
    x = _rankdata(np.asarray(left, dtype=np.float64))
    y = _rankdata(np.asarray(right, dtype=np.float64))
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _payload_candidates(payload: Mapping[str, Any]) -> list[CleanCandidate]:
    labels = np.asarray(payload["point_labels"], dtype=np.int64)
    result: list[CleanCandidate] = []
    for raw_id, metadata in payload["instances"].items():
        result.append(
            CleanCandidate(
                object_id=int(raw_id),
                gaussian_ids=np.flatnonzero(labels == int(raw_id)),
                class_id=str(metadata["class"]),
                winner_probability=float(metadata["winner_probability"]),
                view_consensus=float(metadata["view_consensus"]),
                detection_ratio=float(metadata["detection_ratio"]),
            )
        )
    return result


def _prior_effect(config: CleanExperimentConfig, scene_ids: Sequence[str]) -> dict[str, Any]:
    status_changes = 0
    common_decisions = 0
    path_specific_decisions = 0
    edge_changes = 0
    rows: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        u = load_json(config.condition_dir(scene_id, "U-global") / "diagnostics.json")
        d = load_json(config.condition_dir(scene_id, "D-predicted") / "diagnostics.json")
        u_raw = u.get("consensus", {}).get("raw_graph_identity")
        d_raw = d.get("consensus", {}).get("raw_graph_identity")
        if not isinstance(u_raw, str) or u_raw != d_raw:
            raise ValueError(
                f"{scene_id}: U/D do not share the same raw consensus graph"
            )
        def table(payload: Mapping[str, Any]) -> dict[tuple[int, ...], bool]:
            return {
                tuple(map(int, row["mask_ids"])): bool(row["accepted"])
                for row in payload.get("size_merge_decisions", [])
            }
        ut, dt = table(u), table(d)
        common = set(ut).intersection(dt)
        changed = sum(ut[key] != dt[key] for key in common)
        common_decisions += len(common)
        exclusive = len(set(ut).symmetric_difference(dt))
        path_specific_decisions += exclusive
        status_changes += changed
        def edges(payload: Mapping[str, Any]) -> set[tuple[int, ...]]:
            return {
                tuple(sorted(map(int, row["left_mask_ids"] + row["right_mask_ids"])))
                for row in payload.get("accepted_edges", [])
            }
        diff = len(edges(u).symmetric_difference(edges(d)))
        edge_changes += diff
        rows.append(
            {
                "scene_id": scene_id,
                "raw_graph_identity": u_raw,
                "common_merge_decision_count": len(common),
                "path_specific_merge_decision_count": exclusive,
                "status_changes": changed,
                "edge_changes": diff,
            }
        )
    return {
        "merge_status_change_count": status_changes,
        "merge_decision_count": common_decisions,
        "path_specific_merge_decision_count": path_specific_decisions,
        "merge_status_change_fraction": (
            status_changes / common_decisions if common_decisions else 0.0
        ),
        "final_merge_decision_change_count": edge_changes,
        "rows": rows,
    }


def _validate_condition_pair_identity(
    config: CleanExperimentConfig,
    scene_id: str,
    conditions: Sequence[str],
) -> dict[str, Any]:
    """Prove that conditions differ only by the registered size-prior mode."""

    identities: dict[str, dict[str, Any]] = {}
    raw_graphs: dict[str, str] = {}
    for condition in conditions:
        output = load_json(config.condition_dir(scene_id, condition) / "output.json")
        diagnostics = load_json(
            config.condition_dir(scene_id, condition) / "diagnostics.json"
        )
        identity = validate_embedded_identity(
            output.get("run_identity"), expected_schema=RUN_IDENTITY_SCHEMA
        )
        diagnostic_identity = validate_embedded_identity(
            diagnostics.get("run_identity"),
            expected_schema=RUN_IDENTITY_SCHEMA,
        )
        if diagnostic_identity != identity:
            raise ValueError(f"{scene_id}/{condition}: output/diagnostic identity mismatch")
        if identity.get("condition") != condition or identity.get("scene_id") != scene_id:
            raise ValueError(f"{scene_id}/{condition}: embedded formal identity mismatch")
        raw_graph = diagnostics.get("consensus", {}).get("raw_graph_identity")
        if not isinstance(raw_graph, str) or not raw_graph:
            raise ValueError(f"{scene_id}/{condition}: raw graph identity is missing")
        identities[condition] = identity
        raw_graphs[condition] = raw_graph
    first = identities[str(conditions[0])]
    invariant_keys = (
        "consumer_commit",
        "scene_id",
        "evidence",
        "consensus_config",
        "taxonomy",
        "ap_score",
    )
    prior_conditions = [
        condition for condition in conditions if condition != "C0-no-prior"
    ]
    formal_prior = (
        identities[str(prior_conditions[0])].get("prior")
        if prior_conditions
        else None
    )
    if prior_conditions and not isinstance(formal_prior, Mapping):
        raise ValueError(f"{scene_id}: U/D run identity lacks train-only priors")
    for condition, identity in identities.items():
        for key in invariant_keys:
            if identity.get(key) != first.get(key):
                raise ValueError(
                    f"{scene_id}: formal conditions do not share {key}; "
                    f"drift detected at {condition}"
                )
        if condition != "C0-no-prior" and identity.get("prior") != formal_prior:
            raise ValueError(f"{scene_id}: U/D category-prior content differs")
        if condition == "C0-no-prior" and identity.get("prior") is not None:
            raise ValueError(f"{scene_id}: C0 unexpectedly embeds a category prior")
    if len(set(raw_graphs.values())) != 1:
        raise ValueError(f"{scene_id}: formal conditions do not share one raw graph")
    return {
        "scene_id": scene_id,
        "conditions": list(map(str, conditions)),
        "evidence_identity": first["evidence"]["files"],
        "raw_graph_identity": next(iter(raw_graphs.values())),
        "ap_score_identity": first["ap_score"],
        "condition_specific_fields": ["condition", "prior"],
    }


def _default_evaluate_stage(
    config: CleanExperimentConfig, **kwargs: Any
) -> Mapping[str, Any]:
    scene_ids = tuple(map(str, kwargs["scene_ids"]))
    conditions = tuple(map(str, kwargs["conditions"]))
    stage_output = Path(kwargs["output_path"])
    manifest_path = stage_output.with_name(stage_output.stem + "_manifest.json")
    manifest = {
        "kind": "clean_baseline_evaluation_manifest",
        "minimum_mapped_fraction": 0.90,
        "conditions": list(conditions),
        "scenes": [
            {
                "scene_id": scene_id,
                "gt_npz": str(config.prepared_scene_spec(scene_id).gt_npz),
                "gaussian_ply": str(config.prepared_scene_spec(scene_id).gaussian_ply),
                "gaussian_to_gt_transform": [
                    list(row)
                    for row in config.prepared_scene_spec(scene_id).gaussian_to_gt_transform
                ],
                "outputs": {
                    condition: str(
                        config.condition_dir(scene_id, condition) / "output.json"
                    )
                    for condition in conditions
                },
            }
            for scene_id in scene_ids
        ],
    }
    write_json(manifest_path, manifest)
    official_output = stage_output.with_name(stage_output.stem + "_official.json")
    official = evaluate_clean_baseline_manifest(
        manifest_path,
        class_names=config.evaluation_class_names,
        output_path=official_output,
        radius_m=config.radius_m,
        min_region_size=config.min_region_size,
    )
    condition_rows: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in conditions
    }
    table_rows: list[dict[str, Any]] = []
    paired_identity_rows: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        paired_identity_rows.append(
            _validate_condition_pair_identity(config, scene_id, conditions)
        )
        _, gt_objects, mapping, _ = _scene_gt_adapter(
            config, scene_id, config.bank_dir(scene_id)
        )
        bank = load_evidence_bank(config.bank_dir(scene_id))
        for condition in conditions:
            payload = load_json(config.condition_dir(scene_id, condition) / "output.json")
            point_labels = np.asarray(payload["point_labels"], dtype=np.int64)
            declared_ids = {int(value) for value in payload["instances"]}
            observed_ids = {
                int(value) for value in np.unique(point_labels) if int(value) >= 0
            }
            contract_audit = {
                "orphan_gaussian_count": int(
                    np.count_nonzero(
                        np.isin(point_labels, sorted(observed_ids - declared_ids))
                    )
                ),
                "negative_metadata_count": int(
                    sum(value < 0 for value in declared_ids)
                ),
                # A serialized point label has one owner by construction.  The
                # exporter rejects candidate overlap before this projection;
                # verifying the declared/observed sets here ensures the stage
                # report is derived from the actual payload rather than a
                # hard-coded clean bill of health.
                "duplicate_ownership_count": int(
                    sum(
                        1
                        for instance_id in declared_ids
                        if np.count_nonzero(point_labels == instance_id) == 0
                    )
                ),
            }
            candidate_eval = evaluate_candidates(
                _payload_candidates(payload), gt_objects, mapping, gaussian_count=bank.point_count
            )
            from .evaluation import evaluate_prediction_payload
            gt_coords, gt_scene = load_ground_truth_npz(
                config.prepared_scene_spec(scene_id).gt_npz, scene_id
            )
            del gt_coords
            scene_official = evaluate_prediction_payload(
                scene_id=scene_id,
                payload=payload,
                gt_semantic=gt_scene.semantic,
                gt_instance=gt_scene.instance,
                gt_point_to_gaussian=mapping,
                class_names=config.evaluation_class_names,
                min_region_size=config.min_region_size,
            )
            aggregate = candidate_eval["aggregate"]
            condition_rows[condition].append(
                {
                    "scene_id": scene_id,
                    "candidate": aggregate,
                    "candidate_rows": candidate_eval["candidate_rows"],
                    "gt_rows": candidate_eval["gt_rows"],
                    "official": scene_official["aggregate"],
                    "contract": contract_audit,
                }
            )
            table_rows.append(
                {"scene_id": scene_id, "condition": condition, **aggregate}
            )
    result_conditions: dict[str, Any] = {}
    for condition in conditions:
        rows = condition_rows[condition]
        candidate_count = sum(row["candidate"]["candidate_count"] for row in rows)
        same25 = sum(row["candidate"]["same_class_iou_025_count"] for row in rows)
        geometry25 = sum(row["candidate"]["geometry_iou_025_count"] for row in rows)
        geometry50 = sum(row["candidate"]["geometry_iou_050_count"] for row in rows)
        same50 = sum(row["candidate"]["same_class_iou_050_count"] for row in rows)
        tiny_gt = sum(
            sum(bool(gt["is_tiny_small"]) for gt in row["gt_rows"]) for row in rows
        )
        tiny25 = sum(
            sum(
                bool(gt["is_tiny_small"]) and gt["best_same_class_iou"] >= 0.25
                for gt in row["gt_rows"]
            )
            for row in rows
        )
        tiny50 = sum(
            sum(
                bool(gt["is_tiny_small"]) and gt["best_same_class_iou"] >= 0.50
                for gt in row["gt_rows"]
            )
            for row in rows
        )
        all_candidate_rows = [item for row in rows for item in row["candidate_rows"]]
        scores = [float(row["score"]) for row in all_candidate_rows]
        ious = [float(row["best_same_class_iou"]) for row in all_candidate_rows]
        fp_tp = (candidate_count - same25) / same25 if same25 else None
        aggregate = official["metrics"][condition]["aggregate"]
        result_conditions[condition] = {
            "official": {
                "map_50_95": float(aggregate["map_50_95"] or 0.0),
                "map_0.50": float(aggregate["map_0.50"] or 0.0),
                "map_0.25": float(aggregate["map_0.25"] or 0.0),
            },
            "candidate": {
                "candidate_count": candidate_count,
                "geometry_iou_025_count": geometry25,
                "geometry_iou_050_count": geometry50,
                "geometry_iou_050_scene_count": sum(
                    row["candidate"]["geometry_iou_050_count"] > 0 for row in rows
                ),
                "same_class_iou_025_count": same25,
                "same_class_iou_050_count": same50,
                "same_class_iou_050_scene_count": sum(
                    row["candidate"]["same_class_iou_050_count"] > 0 for row in rows
                ),
                "candidate_precision_025": same25 / candidate_count if candidate_count else 0.0,
                "tiny_small_recall_025": tiny25 / tiny_gt if tiny_gt else 0.0,
                "tiny_small_recall_050": tiny50 / tiny_gt if tiny_gt else 0.0,
                "score_iou_spearman": _spearman(scores, ious),
                "fp_tp_ratio_025": fp_tp,
            },
            "contract": {
                key: int(sum(row["contract"][key] for row in rows))
                for key in (
                    "orphan_gaussian_count",
                    "negative_metadata_count",
                    "duplicate_ownership_count",
                )
            },
            "scenes": rows,
        }
    prior_effect = (
        _prior_effect(config, scene_ids)
        if {"U-global", "D-predicted"}.issubset(conditions)
        else {
            "merge_status_change_count": 0,
            "merge_decision_count": 0,
            "merge_status_change_fraction": 0.0,
            "final_merge_decision_change_count": 0,
            "rows": [],
        }
    )
    if {"U-global", "D-predicted"}.issubset(conditions):
        u = result_conditions["U-global"]
        d = result_conditions["D-predicted"]
        u_raw = u["candidate"]["fp_tp_ratio_025"]
        d_raw = d["candidate"]["fp_tp_ratio_025"]
        if u_raw is None:
            degradation = 0.0 if d_raw is None else None
        else:
            u_ratio = float(u_raw)
            if d_raw is None:
                degradation = None
            else:
                d_ratio = float(d_raw)
                degradation = (
                    0.0 if u_ratio == 0 and d_ratio == 0
                    else None if u_ratio == 0
                    else (d_ratio - u_ratio) / u_ratio
                )
        deltas = [
            float(drow["official"]["map_50_95"] or 0.0)
            - float(urow["official"]["map_50_95"] or 0.0)
            for urow, drow in zip(u["scenes"], d["scenes"])
        ]
        data_minus_uniform = {
            "map_50_95_delta": d["official"]["map_50_95"] - u["official"]["map_50_95"],
            "tiny_small_recall_050_delta": d["candidate"]["tiny_small_recall_050"]
            - u["candidate"]["tiny_small_recall_050"],
            "positive_scene_count": sum(value > 0 for value in deltas),
            "fp_tp_degradation": degradation,
        }
    else:
        data_minus_uniform = {}
    result = {
        "schema": "saga-clean-alpha-mask-stage-evaluation-v1",
        "scene_ids": list(scene_ids),
        "conditions": result_conditions,
        "prior_effect": prior_effect,
        "data_minus_uniform": data_minus_uniform,
        "rows": table_rows,
        "prior_rows": prior_effect["rows"],
        "paired_runtime_identity": paired_identity_rows,
        "oracle_class_in_formal_metrics": False,
    }
    write_json(stage_output, result)
    return result


def _default_run_identity_control(
    config: CleanExperimentConfig, *, output_path: str | Path
) -> Mapping[str, Any]:
    control = config.identity_control
    if control is None:
        raise RuntimeError("identity-edge control was not registered")
    scene_ids = (*control.train_scene_ids, control.validation_scene_id)
    scenes: dict[str, IdentitySceneInput] = {}
    for scene_id in scene_ids:
        spec = config.prepared_scene_spec(scene_id)
        scenes[scene_id] = IdentitySceneInput(
            scene_id=scene_id,
            bank_dir=config.bank_dir(scene_id),
            gt_npz=spec.gt_npz,
            gaussian_to_gt_transform=spec.gaussian_to_gt_transform,
            uniform_output_json=config.condition_dir(scene_id, "U-global")
            / "output.json",
            evaluation_class_names=config.evaluation_class_names,
            tiny_small_instance_ids=spec.tiny_small_instance_ids,
            min_region_size=config.min_region_size,
            radius_m=config.radius_m,
        )
    return run_identity_edge_control(
        control=control,
        scenes=scenes,
        output_path=output_path,
    )


def default_hooks(config: CleanExperimentConfig) -> CleanExperimentHooks:
    return CleanExperimentHooks(
        build_evidence=lambda **kwargs: _default_build_evidence(config, **kwargs),
        geometry_oracle=lambda **kwargs: _default_geometry_oracle(config, **kwargs),
        run_formal=lambda **kwargs: run_consensus_condition(
            consumer_commit=config.code_commit, **kwargs
        ),
        run_oracle=lambda **kwargs: _default_run_oracle(config, **kwargs),
        evaluate_stage=lambda **kwargs: _default_evaluate_stage(config, **kwargs),
        run_identity_control=lambda **kwargs: _default_run_identity_control(
            config, **kwargs
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the registered clean baseline")
    parser.add_argument("--config", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_clean_baseline_experiment(CleanExperimentConfig.from_json(args.config))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CONFIG_KIND",
    "DEV2",
    "DEV8",
    "EVIDENCE_IMPORT_SCHEMA",
    "CleanEvidenceImport",
    "CleanExperimentConfig",
    "CleanExperimentHooks",
    "CleanSceneSpec",
    "default_hooks",
    "run_clean_baseline_experiment",
]
