from __future__ import annotations

"""Immutable V10-bank replay using the registered V9 2^3 prior formulas."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_json, write_json
from .prediction_contract import validate_prediction_contract
from .v10_runner import (
    V10_CLASSIFIERS,
    V10CandidateBank,
    load_v10_candidate_bank,
    v10_object_bank_is_complete,
)
from .v9_objectbank import CandidateBank
from .v9_replay import CONDITION_FACTORS, ReplayResult, replay_candidate_bank


V10_REPLAY_SCHEMA = "saga-v10-prior-replay-v1"
V10_PRIOR_CONDITIONS = tuple(CONDITION_FACTORS)


def _file_identity(path: str | Path) -> dict[str, Any]:
    target = Path(path).resolve()
    stat = target.stat()
    return {
        "path": str(target),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _as_v9_bank(bank: V10CandidateBank, classifier: str) -> CandidateBank:
    """Create the V9 scorer's value object without sharing mutable arrays."""

    if classifier not in V10_CLASSIFIERS:
        raise ValueError(f"unknown V10 late classifier: {classifier}")
    candidates: list[dict[str, Any]] = []
    for row in bank.candidates:
        values = dict(row)
        classifiers = values.get("classifiers")
        if not isinstance(classifiers, Mapping) or classifier not in classifiers:
            raise ValueError(f"V10 candidate lacks {classifier} evidence")
        selected = classifiers[classifier]
        if not isinstance(selected, Mapping):
            raise ValueError(f"V10 candidate has invalid {classifier} evidence")
        values.update(dict(selected))
        values["classification_source"] = classifier
        candidates.append(values)

    return CandidateBank(
        point_count=int(bank.point_count),
        association_mode=bank.structure_condition,
        core_candidate_id=np.asarray(bank.core_candidate_id).copy(),
        full_ids=tuple(np.asarray(ids).copy() for ids in bank.full_ids),
        core_ids=tuple(np.asarray(ids).copy() for ids in bank.core_ids),
        candidates=tuple(candidates),
    )


def replay_v10_candidate_bank(
    bank: V10CandidateBank,
    priors: Mapping[str, Any],
    condition: str,
    *,
    classifier: str = "mv-label",
    acceptance_threshold: float,
    nms_core_iou: float = 0.50,
    min_points: int = 10,
) -> ReplayResult:
    """Replay priors without changing V10 fragment/track/full/core identity."""

    if condition not in CONDITION_FACTORS:
        raise ValueError(f"unknown V10 prior condition: {condition}")
    return replay_candidate_bank(
        _as_v9_bank(bank, classifier),
        priors,
        condition,
        acceptance_threshold=float(acceptance_threshold),
        nms_core_iou=float(nms_core_iou),
        min_points=int(min_points),
    )


def v10_replay_is_complete(
    output_dir: str | Path,
    *,
    expected_scene_id: str | None = None,
    expected_structure_condition: str | None = None,
    expected_prior_condition: str | None = None,
    expected_classifier: str | None = None,
    expected_bank_identity: Mapping[str, Any] | None = None,
    expected_category_priors: str | Path | None = None,
    expected_acceptance_threshold: float | None = None,
    expected_git_commit: str | None = None,
    expected_point_count: int | None = None,
) -> bool:
    target = Path(output_dir)
    try:
        output = load_json(target / "output.json")
        diagnostics = load_json(target / "diagnostics.json")
        if diagnostics.get("schema") != V10_REPLAY_SCHEMA:
            return False
        if (
            expected_scene_id is not None
            and diagnostics.get("scene_id") != expected_scene_id
        ):
            return False
        if expected_classifier is not None and diagnostics.get("classifier") != expected_classifier:
            return False
        if (
            expected_structure_condition is not None
            and diagnostics.get("structure_condition") != expected_structure_condition
        ):
            return False
        if (
            expected_prior_condition is not None
            and diagnostics.get("prior_condition") != expected_prior_condition
        ):
            return False
        if (
            expected_bank_identity is not None
            and diagnostics.get("source_bank_identity") != dict(expected_bank_identity)
        ):
            return False
        if expected_category_priors is not None:
            expected_prior_identity = _file_identity(expected_category_priors)
            if (
                Path(str(diagnostics.get("category_priors", ""))).resolve()
                != Path(expected_category_priors).resolve()
                or diagnostics.get("category_priors_identity")
                != expected_prior_identity
            ):
                return False
        if (
            expected_acceptance_threshold is not None
            and float(diagnostics.get("acceptance_threshold", -1.0))
            != float(expected_acceptance_threshold)
        ):
            return False
        if expected_git_commit is not None and diagnostics.get("git_commit") != expected_git_commit:
            return False
        labels = np.asarray(output["point_labels"])
        instances = output["instances"]
        validate_prediction_contract(labels, instances)
        if expected_point_count is not None and labels.shape != (expected_point_count,):
            return False
        candidate_count = int(diagnostics["candidate_count"])
        candidate_scores = diagnostics["candidate_scores"]
        if not isinstance(candidate_scores, list) or len(candidate_scores) != candidate_count:
            return False
        all_ids = set(range(candidate_count))
        score_ids: list[int] = []
        for row in candidate_scores:
            if not isinstance(row, Mapping):
                return False
            raw_candidate_id = row.get("candidate_id")
            if isinstance(raw_candidate_id, bool):
                return False
            candidate_id = int(raw_candidate_id)
            score = float(row.get("score", float("nan")))
            if candidate_id not in all_ids or not np.isfinite(score) or not 0.0 <= score <= 1.0:
                return False
            score_ids.append(candidate_id)
        if len(set(score_ids)) != candidate_count or set(score_ids) != all_ids:
            return False
        disposition_sets: list[set[int]] = []
        for field in (
            "accepted_candidate_ids",
            "rejected_candidate_ids",
            "suppressed_candidate_ids",
            "dropped_small_candidate_ids",
        ):
            values = diagnostics[field]
            if (
                not isinstance(values, list)
                or any(isinstance(value, bool) or int(value) not in all_ids for value in values)
            ):
                return False
            normalized = [int(value) for value in values]
            if len(set(normalized)) != len(normalized):
                return False
            disposition_sets.append(set(normalized))
        if any(
            left.intersection(right)
            for index, left in enumerate(disposition_sets)
            for right in disposition_sets[index + 1 :]
        ):
            return False
        if set().union(*disposition_sets) != all_ids:
            return False
        return True
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def replay_v10_scene(
    *,
    bank_dir: str | Path,
    output_root: str | Path,
    condition: str,
    classifier: str,
    category_priors: str | Path,
    acceptance_threshold: float,
    git_commit: str,
    nms_core_iou: float = 0.50,
    min_points: int = 10,
) -> dict[str, Any]:
    """Replay one prior condition into its isolated structure/prior directory."""

    if condition not in CONDITION_FACTORS:
        raise ValueError(f"unknown V10 prior condition: {condition}")
    if classifier not in V10_CLASSIFIERS:
        raise ValueError(f"unknown V10 late classifier: {classifier}")
    commit = str(git_commit).strip()
    if not commit:
        raise ValueError("V10 replay git_commit must be non-empty")
    source = Path(bank_dir).resolve()
    if not v10_object_bank_is_complete(source):
        raise ValueError(f"incomplete V10 ObjectBank: {source}")
    bank_metadata = load_json(source / "object_bank.json")
    scene_id = str(bank_metadata["scene_id"])
    structure_condition = str(bank_metadata["condition"])
    bank_identity = dict(bank_metadata["identity"])
    priors_path = Path(category_priors).resolve()
    priors_identity = _file_identity(priors_path)
    target = (
        Path(output_root).resolve()
        / structure_condition
        / classifier
        / condition
        / scene_id
    )
    if v10_replay_is_complete(
        target,
        expected_scene_id=scene_id,
        expected_structure_condition=structure_condition,
        expected_prior_condition=condition,
        expected_classifier=classifier,
        expected_bank_identity=bank_identity,
        expected_category_priors=priors_path,
        expected_acceptance_threshold=float(acceptance_threshold),
        expected_git_commit=commit,
        expected_point_count=int(bank_metadata["point_count"]),
    ):
        return load_json(target / "diagnostics.json")

    loaded_metadata, bank = load_v10_candidate_bank(source)
    if loaded_metadata != bank_metadata:
        raise ValueError("V10 bank metadata changed while replaying")
    priors = load_json(priors_path)
    result = replay_v10_candidate_bank(
        bank,
        priors,
        condition,
        classifier=classifier,
        acceptance_threshold=float(acceptance_threshold),
        nms_core_iou=float(nms_core_iou),
        min_points=int(min_points),
    )
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "output.json", result.prediction.output_payload())
    diagnostics = {
        "schema": V10_REPLAY_SCHEMA,
        "scene_id": scene_id,
        "structure_condition": structure_condition,
        "prior_condition": condition,
        "classifier": classifier,
        "git_commit": commit,
        "source_bank": str(source),
        "source_bank_identity": bank_identity,
        "category_priors": str(priors_path),
        "category_priors_identity": priors_identity,
        "acceptance_threshold": float(acceptance_threshold),
        "nms_core_iou": float(nms_core_iou),
        "min_points": int(min_points),
        "candidate_count": len(bank.candidates),
        "fragment_count": len(bank.fragments),
        "track_count": len(bank.tracks),
        "accepted_candidate_ids": list(result.accepted_candidate_ids),
        "rejected_candidate_ids": list(result.rejected_candidate_ids),
        "suppressed_candidate_ids": list(result.suppressed_candidate_ids),
        "dropped_small_candidate_ids": list(result.dropped_small_candidate_ids),
        "candidate_scores": list(result.candidate_scores),
        "instances": result.prediction.instance_metadata,
        "coverage": float(np.mean(result.point_labels >= 0)),
    }
    write_json(target / "diagnostics.json", diagnostics)
    if not v10_replay_is_complete(
        target,
        expected_scene_id=scene_id,
        expected_structure_condition=structure_condition,
        expected_prior_condition=condition,
        expected_classifier=classifier,
        expected_bank_identity=bank_identity,
        expected_category_priors=priors_path,
        expected_acceptance_threshold=float(acceptance_threshold),
        expected_git_commit=commit,
        expected_point_count=int(bank_metadata["point_count"]),
    ):
        raise RuntimeError(f"incomplete V10 prior replay: {target}")
    return diagnostics


def replay_v10_priors(
    *,
    bank_root: str | Path,
    output_root: str | Path,
    scene_ids: Sequence[str],
    structure_conditions: Sequence[str],
    prior_conditions: Sequence[str],
    classifier: str,
    category_priors: str | Path,
    acceptance_threshold: float,
    git_commit: str,
    nms_core_iou: float = 0.50,
    min_points: int = 10,
) -> dict[str, Any]:
    unknown = set(map(str, prior_conditions)).difference(CONDITION_FACTORS)
    if unknown:
        raise ValueError(f"unknown V10 prior conditions: {sorted(unknown)}")
    root = Path(bank_root)
    rows = [
        replay_v10_scene(
            bank_dir=root / structure_condition / scene_id,
            output_root=output_root,
            condition=prior_condition,
            classifier=classifier,
            category_priors=category_priors,
            acceptance_threshold=float(acceptance_threshold),
            git_commit=git_commit,
            nms_core_iou=float(nms_core_iou),
            min_points=int(min_points),
        )
        for scene_id in map(str, scene_ids)
        for structure_condition in map(str, structure_conditions)
        for prior_condition in map(str, prior_conditions)
    ]
    summary = {
        "schema": "saga-v10-prior-replay-summary-v1",
        "git_commit": str(git_commit),
        "structure_conditions": list(map(str, structure_conditions)),
        "prior_conditions": list(map(str, prior_conditions)),
        "classifier": classifier,
        "runs": rows,
    }
    write_json(Path(output_root) / "replay_summary.json", summary)
    return summary
