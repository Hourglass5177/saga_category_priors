from __future__ import annotations

"""Thin runtime pipeline for the clean Gaussian--mask consensus baseline."""

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from ..io import hash_json, load_json, sha256_file, write_json
from ..priors import validate_priors
from ..taxonomy import load_taxonomy
from .consensus import (
    ConsensusConfig,
    ConsensusEdge,
    ConsensusObject,
    MaskObservation,
    compute_pair_consensus,
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

CONDITION_DIAGNOSTICS_SCHEMA = "saga-clean-alpha-mask-condition-diagnostics-v1"


def _condition_diagnostics_is_complete(
    payload: object,
    *,
    expected_scene_id: str,
    expected_condition: str,
    expected_bank_schema: str,
    expected_config: ConsensusConfig,
    expected_run_identity: Mapping[str, Any],
    expected_gaussian_count: int,
    expected_exported_instance_count: int,
) -> bool:
    """Validate the sidecar before treating a condition as a cache hit.

    ``output.json`` is the write-last marker, but ``diagnostics.json`` is also
    a production input to the stage-funnel audit.  A matching identity alone
    must not make a truncated sidecar look complete.
    """

    try:
        if not isinstance(payload, Mapping):
            return False
        if payload.get("schema") != CONDITION_DIAGNOSTICS_SCHEMA:
            return False
        if payload.get("scene_id") != str(expected_scene_id):
            return False
        if payload.get("condition") != str(expected_condition):
            return False
        if payload.get("bank_schema") != str(expected_bank_schema):
            return False
        if payload.get("config") != asdict(expected_config):
            return False
        identity = validate_embedded_identity(
            payload.get("run_identity"), expected_schema=RUN_IDENTITY_SCHEMA
        )
        expected_identity = validate_embedded_identity(
            expected_run_identity, expected_schema=RUN_IDENTITY_SCHEMA
        )
        if identity != expected_identity:
            return False

        consensus = payload.get("consensus")
        required_consensus = {
            "observation_count",
            "active_observation_count",
            "raw_graph_identity",
            "accepted_edge_count",
            "undersegmented_mask_count",
            "component_count_before_output_filters",
            "dropped_by_min_views",
            "dropped_by_detection_ratio",
            "dropped_by_physical_connectivity",
            "contained_duplicate_count",
            "object_count",
        }
        if not isinstance(consensus, Mapping) or not required_consensus.issubset(
            consensus
        ):
            return False

        objects = payload.get("objects")
        accepted_edges = payload.get("accepted_edges")
        rejected = payload.get("rejected_undersegmented_mask_ids")
        size_decisions = payload.get("size_merge_decisions")
        if not isinstance(objects, list):
            return False
        if not isinstance(accepted_edges, list):
            return False
        if not isinstance(rejected, list):
            return False
        if not isinstance(size_decisions, list):
            return False
        if payload.get("unique_object_count") != len(objects):
            return False
        if consensus.get("accepted_edge_count") != len(accepted_edges):
            return False
        if consensus.get("undersegmented_mask_count") != len(rejected):
            return False
        required_object_fields = {
            "object_id",
            "mask_ids",
            "frame_ids",
            "gaussian_count",
            "class",
            "winner_probability",
            "class_probabilities",
            "view_consensus",
            "detection_ratio",
            "score",
        }
        if any(
            not isinstance(row, Mapping)
            or not required_object_fields.issubset(row)
            for row in objects
        ):
            return False
        required_edge_fields = {
            "left_mask_ids",
            "right_mask_ids",
            "observer_count",
            "supporter_count",
            "consensus",
            "observer_level",
        }
        if any(
            not isinstance(row, Mapping)
            or not required_edge_fields.issubset(row)
            for row in accepted_edges
        ):
            return False

        contract = payload.get("prediction_contract")
        required_contract = {
            "schema",
            "scene_id",
            "condition",
            "run_identity",
            "gaussian_count",
            "candidate_count",
            "exported_instance_count",
            "skipped_unclassified_object_ids",
            "contract",
            "oracle_class_used",
        }
        if not isinstance(contract, Mapping) or not required_contract.issubset(
            contract
        ):
            return False
        if contract.get("scene_id") != str(expected_scene_id):
            return False
        if contract.get("condition") != str(expected_condition):
            return False
        if contract.get("gaussian_count") != int(expected_gaussian_count):
            return False
        if contract.get("candidate_count") != len(objects):
            return False
        if contract.get("exported_instance_count") != int(
            expected_exported_instance_count
        ):
            return False
        contract_identity = validate_embedded_identity(
            contract.get("run_identity"), expected_schema=RUN_IDENTITY_SCHEMA
        )
        if contract_identity != expected_identity:
            return False
        audit = contract.get("contract")
        if not isinstance(audit, Mapping):
            return False
        if audit.get("schema") != "saga-strict-prediction-contract-v1":
            return False
        if audit.get("point_count") != int(expected_gaussian_count):
            return False
        if audit.get("instance_count") != int(expected_exported_instance_count):
            return False
        if payload.get("prior_in_ap_score") is not False:
            return False
        if payload.get("oracle_class_used") is not False:
            return False
        if contract.get("oracle_class_used") is not False:
            return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


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
    """Expand compact visibility while honoring geometry abstention.

    A frame with no complete SAM mask retains alpha visibility in the evidence
    bank for input auditing, but it has supplied no geometric observation.  It
    therefore cannot vote against an association or lower a final detection
    ratio merely because the mask input was absent.
    """

    visible = np.zeros((bank.frame_count, bank.point_count), dtype=np.bool_)
    for row, frame in enumerate(bank.frames):
        if frame.geometry_abstained:
            continue
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


def _final_support_statistics(
    *,
    bank: AlphaMaskEvidenceBank,
    mask_ids: Sequence[int],
    gaussian_ids: np.ndarray,
    accepted_edges: Sequence[ConsensusEdge],
    rejected_undersegmented_mask_ids: Sequence[int] = (),
    config: ConsensusConfig = ConsensusConfig(),
    observations: Sequence[MaskObservation] | None = None,
    visibility: np.ndarray | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...], float, float]:
    """Recompute view support after overlap removal changes an object mask.

    Consensus calculates detection ratios before residual object overlaps are
    assigned uniquely.  Reusing those values after points have been removed
    makes the exported AP score describe a different Gaussian set.  This
    helper repeats the registered per-view numerator/denominator calculation
    for the final support, excluding same-frame ambiguity and frames whose SAM
    geometry abstained.
    """

    raw_support = np.asarray(gaussian_ids, dtype=np.int64)
    if raw_support.ndim != 1 or len(raw_support) == 0:
        return (), (), 0.0, 0.0
    support = np.unique(raw_support)
    supported_masks: list[int] = []
    masks_by_frame: dict[int, list[int]] = {}
    for mask_id in sorted(map(int, mask_ids)):
        row = bank.mask_position(mask_id)
        metadata = bank.masks[row]
        association_ids = bank.support_for_mask(
            mask_id, include_ambiguous=False
        )[0]
        if np.intersect1d(
            support, association_ids, assume_unique=True
        ).size == 0:
            continue
        supported_masks.append(mask_id)
        masks_by_frame.setdefault(metadata.frame_id, []).append(mask_id)
    frame_ids = tuple(sorted(masks_by_frame))
    if not supported_masks:
        return (), (), 0.0, 0.0

    visible_counts = np.zeros(len(support), dtype=np.int64)
    detected_counts = np.zeros(len(support), dtype=np.int64)
    for frame in bank.frames:
        if frame.geometry_abstained:
            continue
        visible_ids, _ = bank.visibility_for_frame(frame.frame_id)
        _, support_positions, _ = np.intersect1d(
            support,
            visible_ids,
            assume_unique=True,
            return_indices=True,
        )
        if support_positions.size == 0:
            continue
        ambiguous = bank.ambiguous_for_frame(frame.frame_id)
        if ambiguous.size:
            support_positions = support_positions[
                ~np.isin(
                    support[support_positions], ambiguous, assume_unique=True
                )
            ]
        if support_positions.size == 0:
            continue
        visible_counts[support_positions] += 1
        frame_mask_ids = masks_by_frame.get(frame.frame_id, ())
        if not frame_mask_ids:
            continue
        detected = np.unique(
            np.concatenate(
                [
                    bank.support_for_mask(
                        mask_id, include_ambiguous=False
                    )[0]
                    for mask_id in frame_mask_ids
                ]
            )
        )
        detected_positions = support_positions[
            np.isin(
                support[support_positions], detected, assume_unique=True
            )
        ]
        detected_counts[detected_positions] += 1
    ratios = np.divide(
        detected_counts,
        visible_counts,
        out=np.zeros(len(support), dtype=np.float64),
        where=visible_counts > 0,
    )
    retained = set(supported_masks)
    relevant_edges = []
    for edge in accepted_edges:
        left_retained = tuple(
            mask_id for mask_id in edge.left_mask_ids if mask_id in retained
        )
        right_retained = tuple(
            mask_id for mask_id in edge.right_mask_ids if mask_id in retained
        )
        if left_retained and right_retained:
            relevant_edges.append((edge, left_retained, right_retained))
    # An accepted edge's stored consensus describes the component supports at
    # merge time.  DBSCAN and unique ownership can later remove a disconnected
    # or overlapping region, so reusing that number would make Q describe the
    # parent object rather than this final exported part.  Restrict every mask
    # to the final Gaussian domain and recompute the same observer/supporter
    # statistic.  This changes metadata only; it cannot alter graph formation.
    mean_consensus = 0.0
    if relevant_edges:
        base_observations = (
            tuple(observations)
            if observations is not None
            else _observations(bank)
        )
        restricted_observations: list[MaskObservation] = []
        for observation in base_observations:
            restricted_full = np.intersect1d(
                observation.gaussian_ids, support, assume_unique=True
            )
            restricted_ambiguous = np.intersect1d(
                observation.ambiguous_ids, support, assume_unique=True
            )
            restricted_observations.append(
                MaskObservation(
                    mask_id=observation.mask_id,
                    frame_id=observation.frame_id,
                    gaussian_ids=restricted_full,
                    ambiguous_ids=restricted_ambiguous,
                )
            )
        position = {
            observation.mask_id: index
            for index, observation in enumerate(restricted_observations)
        }
        rejected_set = set(map(int, rejected_undersegmented_mask_ids))
        active_indices = tuple(
            index
            for index, observation in enumerate(restricted_observations)
            if observation.mask_id not in rejected_set
        )
        final_visibility = (
            np.asarray(visibility)
            if visibility is not None
            else _dense_visibility(bank)
        )
        recomputed = []
        for _edge, left_retained, right_retained in relevant_edges:
            evidence = compute_pair_consensus(
                tuple(position[mask_id] for mask_id in left_retained),
                tuple(position[mask_id] for mask_id in right_retained),
                restricted_observations,
                final_visibility,
                config=config,
                active_indices=active_indices,
                rejected_mask_ids=tuple(sorted(rejected_set)),
            )
            recomputed.append(evidence.consensus)
        mean_consensus = float(np.mean(recomputed))
    return (
        tuple(supported_masks),
        frame_ids,
        mean_consensus,
        float(np.mean(ratios)),
    )


def _resolve_unique_ownership(
    objects: Sequence[ConsensusObject],
    bank: AlphaMaskEvidenceBank,
    *,
    accepted_edges: Sequence[ConsensusEdge],
    rejected_undersegmented_mask_ids: Sequence[int],
    config: ConsensusConfig,
    observations: Sequence[MaskObservation],
    visibility: np.ndarray,
) -> tuple[ConsensusObject, ...]:
    """Resolve overlaps, then recompute metadata for the final Gaussian set."""

    xyz_m = bank.xyz_m
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
            (
                part_mask_ids,
                part_frame_ids,
                mean_consensus,
                mean_detection,
            ) = _final_support_statistics(
                bank=bank,
                mask_ids=item.mask_ids,
                gaussian_ids=part,
                accepted_edges=accepted_edges,
                rejected_undersegmented_mask_ids=(
                    rejected_undersegmented_mask_ids
                ),
                config=config,
                observations=observations,
                visibility=visibility,
            )
            if len(part_frame_ids) < config.min_views:
                continue
            geometric_quality = float(
                np.sqrt(
                    max(0.0, mean_consensus)
                    * max(0.0, mean_detection)
                )
            )
            occupied[part] = True
            result.append(
                ConsensusObject(
                    object_id=len(result),
                    mask_ids=part_mask_ids,
                    frame_ids=part_frame_ids,
                    gaussian_ids=part,
                    mean_view_consensus=mean_consensus,
                    mean_detection_ratio=mean_detection,
                    geometric_quality=geometric_quality,
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
            output_payload = load_json(output_json)
            output_instances = output_payload.get("instances")
            if not isinstance(output_instances, Mapping):
                raise TypeError("prediction instances must be a mapping")
            if _condition_diagnostics_is_complete(
                existing,
                expected_scene_id=scene_id,
                expected_condition=condition,
                expected_bank_schema=bank.schema,
                expected_config=config,
                expected_run_identity=run_identity,
                expected_gaussian_count=bank.point_count,
                expected_exported_instance_count=len(output_instances),
            ):
                return {**existing, "runner_status": "skipped-complete"}
        except (OSError, TypeError, ValueError, KeyError, AttributeError):
            pass

    observations = _observations(bank)
    visibility = _dense_visibility(bank)

    def report_progress(stage: str, payload: Mapping[str, object]) -> None:
        print(
            json.dumps(
                {
                    "event": "clean-baseline-consensus-progress",
                    "scene_id": scene_id,
                    "condition": condition,
                    "stage": stage,
                    **payload,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )

    result = run_mask_consensus(
        observations,
        visibility,
        bank.xyz_m,
        config=config,
        merge_veto=merge_veto,
        progress_callback=report_progress,
    )
    unique_objects = _resolve_unique_ownership(
        result.objects,
        bank,
        accepted_edges=result.accepted_edges,
        rejected_undersegmented_mask_ids=(
            result.rejected_undersegmented_mask_ids
        ),
        config=config,
        observations=observations,
        visibility=visibility,
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
        "schema": CONDITION_DIAGNOSTICS_SCHEMA,
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
