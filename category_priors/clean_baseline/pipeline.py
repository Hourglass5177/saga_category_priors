from __future__ import annotations

"""Thin runtime pipeline for the clean Gaussian--mask consensus baseline."""

from dataclasses import asdict
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from ..io import hash_json, load_json, sha256_file, write_json
from ..priors import validate_priors
from ..taxonomy import load_taxonomy
from .consensus import (
    ConsensusConfig,
    ConsensusObject,
    MaskObservation,
    run_mask_consensus,
    split_disconnected_support,
)
from .evaluation import (
    RUN_IDENTITY_SCHEMA,
    CleanCandidate,
    build_prediction_payload,
    prediction_is_complete,
    validate_embedded_identity,
)
from .evidence import (
    EVIDENCE_ARRAY_FILE,
    EVIDENCE_DIAGNOSTICS_FILE,
    EVIDENCE_METADATA_FILE,
    load_evidence_bank,
)
from .models import AlphaMaskEvidenceBank
from .size_prior import (
    SizePriorTable,
    global_size_compatibility,
    pca_sorted_extents_m,
    predicted_size_compatibility,
)
from .worker import DEFAULT_CLASSES


CONDITION_TO_PRIOR_MODE = {
    "C0-no-prior": "none",
    "U-global": "global",
    "D-predicted": "predicted",
}


def _current_git_commit() -> str:
    repository = Path(__file__).resolve().parents[2]
    try:
        value = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "clean-baseline output requires an explicit consumer commit"
        ) from exc
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError("git HEAD is not a full lowercase commit identity")
    return value


def build_condition_run_identity(
    *,
    scene_id: str,
    condition: str,
    bank_dir: str | Path,
    bank: AlphaMaskEvidenceBank,
    config: ConsensusConfig,
    allowed_classes: Sequence[str],
    prior_payload: Mapping[str, Any] | None,
    consumer_commit: str | None = None,
) -> dict[str, Any]:
    """Build the exact embedded cache boundary for one formal prediction."""

    root = Path(bank_dir)
    taxonomy = load_taxonomy()
    evidence_classes = tuple(map(str, bank.class_names))
    if evidence_classes != tuple(DEFAULT_CLASSES):
        raise ValueError(
            "evidence bank class_names do not match the registered 32-class order"
        )
    output_classes = tuple(map(str, allowed_classes))
    if output_classes != taxonomy.canonical_classes:
        raise ValueError(
            "allowed_classes must exactly match the registered SAGA20 taxonomy order"
        )
    if prior_payload is None:
        prior_identity: dict[str, Any] | None = None
    else:
        validate_priors(prior_payload)
        prior_identity = {
            "kind": prior_payload["kind"],
            "schema_version": prior_payload["schema_version"],
            "splits": list(prior_payload["provenance"]["splits"]),
            "content_sha256": prior_payload["content_sha256"],
        }
    commit = _current_git_commit() if consumer_commit is None else str(consumer_commit)
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError("consumer_commit must be a full lowercase git commit")
    identity: dict[str, Any] = {
        "schema": RUN_IDENTITY_SCHEMA,
        "consumer_commit": commit,
        "scene_id": str(scene_id),
        "condition": str(condition),
        "evidence": {
            "schema": bank.schema,
            "scene_id": bank.scene_id,
            "point_count": bank.point_count,
            "frame_count": bank.frame_count,
            "mask_count": bank.mask_count,
            "thresholds": bank.thresholds.to_dict(),
            "source": dict(bank.source),
            "class_names": list(evidence_classes),
            "files": {
                EVIDENCE_ARRAY_FILE: sha256_file(root / EVIDENCE_ARRAY_FILE),
                EVIDENCE_METADATA_FILE: sha256_file(root / EVIDENCE_METADATA_FILE),
                EVIDENCE_DIAGNOSTICS_FILE: sha256_file(
                    root / EVIDENCE_DIAGNOSTICS_FILE
                ),
            },
        },
        "consensus_config": asdict(config),
        "taxonomy": {
            "content_sha256": taxonomy.content_hash,
            "allowed_classes": list(output_classes),
        },
        "ap_score": {
            "formula": "winner_probability*sqrt(view_consensus*detection_ratio)",
            "prior_in_score": False,
        },
        "prior": prior_identity,
    }
    identity["content_sha256"] = hash_json(identity)
    return identity


def _observations(bank: AlphaMaskEvidenceBank) -> tuple[MaskObservation, ...]:
    rows: list[MaskObservation] = []
    for metadata in bank.masks:
        ids, _, _, ambiguity = bank.support_for_mask(
            metadata.global_mask_id, include_ambiguous=True
        )
        rows.append(
            MaskObservation(
                mask_id=metadata.global_mask_id,
                frame_id=metadata.frame_id,
                gaussian_ids=ids,
                ambiguous_ids=ids[ambiguity],
            )
        )
    return tuple(rows)


def _dense_visibility(bank: AlphaMaskEvidenceBank) -> np.ndarray:
    """Expand only the compact frame visibility matrix, never pixel evidence."""

    visible = np.zeros((bank.frame_count, bank.point_count), dtype=np.bool_)
    for row, frame in enumerate(bank.frames):
        ids, _ = bank.visibility_for_frame(frame.frame_id)
        visible[row, ids] = True
    return visible


def _mask_probability_mapping(
    bank: AlphaMaskEvidenceBank,
) -> dict[int, dict[str, float]]:
    probabilities: dict[int, dict[str, float]] = {}
    for row, metadata in enumerate(bank.masks):
        if bool(bank.semantic_abstained[row]):
            continue
        values = np.asarray(bank.semantic_posteriors[row], dtype=np.float64)
        probabilities[metadata.global_mask_id] = {
            class_name: float(value)
            for class_name, value in zip(bank.class_names, values)
            if value > 0
        }
    return probabilities


def _object_class_distribution(
    mask_ids: Sequence[int], bank: AlphaMaskEvidenceBank
) -> np.ndarray:
    """Average one normalized 32-class distribution per physical view."""

    by_frame: dict[int, list[np.ndarray]] = {}
    for mask_id in mask_ids:
        row = bank.mask_position(mask_id)
        if bool(bank.semantic_abstained[row]):
            continue
        metadata = bank.masks[row]
        posterior = np.asarray(bank.semantic_posteriors[row], dtype=np.float64)
        if posterior.sum() <= 0:
            continue
        by_frame.setdefault(metadata.frame_id, []).append(posterior)
    frame_probabilities: list[np.ndarray] = []
    for frame_id in sorted(by_frame):
        posterior = np.mean(by_frame[frame_id], axis=0)
        total = float(posterior.sum())
        if total > 0:
            frame_probabilities.append(posterior / total)
    if not frame_probabilities:
        return np.zeros(len(bank.class_names), dtype=np.float64)
    result = np.mean(frame_probabilities, axis=0)
    total = float(result.sum())
    return result / total if total > 0 else np.zeros_like(result)


class _TrackedSizeVeto:
    def __init__(
        self,
        *,
        mode: str,
        xyz_m: np.ndarray,
        priors: SizePriorTable,
        mask_probabilities: Mapping[int, Mapping[str, float]],
        threshold: float = 0.50,
    ) -> None:
        if mode not in {"global", "predicted"}:
            raise ValueError("tracked size veto is only defined for U/D")
        self.mode = mode
        self.xyz_m = np.asarray(xyz_m, dtype=np.float64)
        self.priors = priors
        self.mask_probabilities = mask_probabilities
        self.threshold = float(threshold)
        self.decisions: list[dict[str, Any]] = []

    def _posterior(self, mask_ids: Sequence[int]) -> dict[str, float]:
        accumulated: dict[str, float] = {}
        count = 0
        for mask_id in mask_ids:
            row = self.mask_probabilities.get(int(mask_id))
            if not row:
                continue
            total = float(sum(float(value) for value in row.values()))
            if total <= 0:
                continue
            for class_name, value in row.items():
                accumulated[class_name] = accumulated.get(class_name, 0.0) + float(value) / total
            count += 1
        if count == 0:
            return {"__global_fallback__": 1.0}
        return {key: value / count for key, value in accumulated.items()}

    def __call__(self, mask_ids: tuple[int, ...], gaussian_ids: np.ndarray) -> bool:
        ids = np.asarray(gaussian_ids, dtype=np.int64)
        extents = pca_sorted_extents_m(self.xyz_m[ids])
        posterior: dict[str, float] | None = None
        if self.mode == "global":
            compatibility = global_size_compatibility(extents, self.priors)
        else:
            posterior = self._posterior(mask_ids)
            compatibility = predicted_size_compatibility(
                extents, self.priors, posterior
            )
        accepted = bool(compatibility >= self.threshold)
        self.decisions.append(
            {
                "mask_ids": list(map(int, mask_ids)),
                "gaussian_count": int(len(ids)),
                "extent_short_m": float(extents[0]),
                "extent_mid_m": float(extents[1]),
                "extent_long_m": float(extents[2]),
                "compatibility": float(compatibility),
                "accepted": accepted,
                "posterior": posterior,
            }
        )
        return accepted


def _resolve_unique_ownership(
    objects: Sequence[ConsensusObject],
    xyz_m: np.ndarray,
    *,
    config: ConsensusConfig,
) -> tuple[ConsensusObject, ...]:
    """Resolve residual overlaps using geometry only, then re-split physically."""

    occupied = np.zeros(len(xyz_m), dtype=np.bool_)
    result: list[ConsensusObject] = []
    ranked = sorted(
        objects,
        key=lambda item: (
            -item.geometric_quality,
            -len(item.gaussian_ids),
            item.mask_ids,
            item.object_id,
        ),
    )
    for item in ranked:
        available = item.gaussian_ids[~occupied[item.gaussian_ids]]
        if len(available) < config.dbscan_min_samples:
            continue
        for part in split_disconnected_support(
            available,
            xyz_m,
            eps_m=config.dbscan_eps_m,
            min_samples=config.dbscan_min_samples,
        ):
            occupied[part] = True
            result.append(
                ConsensusObject(
                    object_id=len(result),
                    mask_ids=item.mask_ids,
                    frame_ids=item.frame_ids,
                    gaussian_ids=part,
                    mean_view_consensus=item.mean_view_consensus,
                    mean_detection_ratio=item.mean_detection_ratio,
                    geometric_quality=item.geometric_quality,
                )
            )
    return tuple(result)


def _classify_objects(
    objects: Sequence[ConsensusObject], bank: AlphaMaskEvidenceBank
) -> tuple[tuple[CleanCandidate, ...], list[dict[str, Any]]]:
    candidates: list[CleanCandidate] = []
    rows: list[dict[str, Any]] = []
    for item in objects:
        posterior = _object_class_distribution(item.mask_ids, bank)
        if posterior.sum() > 0:
            winner = int(np.flatnonzero(posterior == posterior.max())[0])
            class_name: str | None = bank.class_names[winner]
            winner_probability = float(posterior[winner])
        else:
            class_name = None
            winner_probability = 0.0
        candidate = CleanCandidate(
            object_id=item.object_id,
            gaussian_ids=item.gaussian_ids,
            class_id=class_name,
            winner_probability=winner_probability,
            view_consensus=item.mean_view_consensus,
            detection_ratio=item.mean_detection_ratio,
        )
        candidates.append(candidate)
        rows.append(
            {
                "object_id": int(item.object_id),
                "mask_ids": list(map(int, item.mask_ids)),
                "frame_ids": list(map(int, item.frame_ids)),
                "gaussian_count": int(len(item.gaussian_ids)),
                "class": class_name,
                "winner_probability": winner_probability,
                "class_probabilities": {
                    name: float(value)
                    for name, value in zip(bank.class_names, posterior)
                    if value > 0
                },
                "view_consensus": float(item.mean_view_consensus),
                "detection_ratio": float(item.mean_detection_ratio),
                "score": float(candidate.score),
            }
        )
    return tuple(candidates), rows


def run_consensus_condition(
    *,
    scene_id: str,
    bank_dir: str | Path,
    condition: str,
    output_dir: str | Path,
    priors_path: str | Path | None = None,
    allowed_classes: Sequence[str] | None = None,
    config: ConsensusConfig = ConsensusConfig(),
    consumer_commit: str | None = None,
) -> dict[str, Any]:
    """Run one formal condition from an immutable evidence bank."""

    if condition not in CONDITION_TO_PRIOR_MODE:
        raise ValueError(f"unknown clean-baseline condition: {condition}")
    bank = load_evidence_bank(bank_dir, expected_scene_id=scene_id)
    output_root = Path(output_dir)
    output_json = output_root / "output.json"
    mode = CONDITION_TO_PRIOR_MODE[condition]
    tracked_veto: _TrackedSizeVeto | None = None
    prior_payload: Mapping[str, Any] | None = None
    if mode == "none":
        merge_veto = None
    else:
        if priors_path is None:
            raise ValueError(f"{condition} requires train-only category priors")
        prior_payload = load_json(priors_path)
        if not isinstance(prior_payload, Mapping):
            raise TypeError("category priors must contain a JSON object")
        table = SizePriorTable.from_category_priors(prior_payload)
        tracked_veto = _TrackedSizeVeto(
            mode=mode,
            xyz_m=bank.xyz_m,
            priors=table,
            mask_probabilities=_mask_probability_mapping(bank),
        )
        merge_veto = tracked_veto

    taxonomy = load_taxonomy()
    taxonomy_classes = (
        tuple(map(str, allowed_classes))
        if allowed_classes is not None
        else taxonomy.canonical_classes
    )
    run_identity = build_condition_run_identity(
        scene_id=scene_id,
        condition=condition,
        bank_dir=bank_dir,
        bank=bank,
        config=config,
        allowed_classes=taxonomy_classes,
        prior_payload=prior_payload,
        consumer_commit=consumer_commit,
    )
    diagnostics_path = output_root / "diagnostics.json"
    if prediction_is_complete(
        output_json,
        expected_scene_id=scene_id,
        expected_condition=condition,
        expected_gaussian_count=bank.point_count,
        expected_run_identity=run_identity,
    ):
        try:
            existing = load_json(diagnostics_path)
            diagnostic_identity = validate_embedded_identity(
                existing.get("run_identity"), expected_schema=RUN_IDENTITY_SCHEMA
            )
            if diagnostic_identity == run_identity:
                return {**existing, "runner_status": "skipped-complete"}
        except (OSError, TypeError, ValueError, KeyError):
            pass

    result = run_mask_consensus(
        _observations(bank),
        _dense_visibility(bank),
        bank.xyz_m,
        config=config,
        merge_veto=merge_veto,
    )
    unique_objects = _resolve_unique_ownership(
        result.objects, bank.xyz_m, config=config
    )
    candidates, object_rows = _classify_objects(unique_objects, bank)
    payload, contract = build_prediction_payload(
        scene_id=scene_id,
        condition=condition,
        gaussian_count=bank.point_count,
        candidates=candidates,
        allowed_classes=taxonomy_classes,
        run_identity=run_identity,
    )
    diagnostics = {
        "schema": "saga-clean-alpha-mask-condition-diagnostics-v1",
        "scene_id": scene_id,
        "condition": condition,
        "run_identity": run_identity,
        "bank_schema": bank.schema,
        "config": asdict(config),
        "consensus": result.diagnostics,
        "unique_object_count": len(unique_objects),
        "objects": object_rows,
        "accepted_edges": [asdict(edge) for edge in result.accepted_edges],
        "rejected_undersegmented_mask_ids": list(
            map(int, result.rejected_undersegmented_mask_ids)
        ),
        "size_merge_decisions": (
            [] if tracked_veto is None else tracked_veto.decisions
        ),
        "prediction_contract": contract,
        "prior_in_ap_score": False,
        "oracle_class_used": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(diagnostics_path, diagnostics)
    # output.json is the completion marker and is written last.
    write_json(output_json, payload)
    if not prediction_is_complete(
        output_json,
        expected_scene_id=scene_id,
        expected_condition=condition,
        expected_gaussian_count=bank.point_count,
        expected_run_identity=run_identity,
    ):
        raise RuntimeError("serialized clean-baseline prediction is invalid")
    return {**diagnostics, "runner_status": "complete"}


def replay_size_prior_condition(**kwargs: Any) -> dict[str, Any]:
    condition = str(kwargs.get("condition", ""))
    if condition not in {"U-global", "D-predicted"}:
        raise ValueError("size-prior replay only accepts U-global or D-predicted")
    return run_consensus_condition(**kwargs)


__all__ = [
    "CONDITION_TO_PRIOR_MODE",
    "build_condition_run_identity",
    "replay_size_prior_condition",
    "run_consensus_condition",
]
