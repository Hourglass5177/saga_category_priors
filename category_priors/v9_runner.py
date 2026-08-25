from __future__ import annotations

"""Filesystem adapter for the deterministic V9 Clean ObjectBank.

The GPU lifting bank is an immutable input.  This module performs only object
association, late classification, compact persistence, and CPU prior replay.
It never imports the legacy postprocessor and never reads ground truth.
"""

import json
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_json, write_json
from .v9_lifting import load_lifting_bank
from .v9_objectbank import (
    AssociationResult,
    CandidateBank,
    Fragment,
    FrameEvidence,
    MultiviewClassVote,
    V9Config,
    associate_fragments,
    attach_local_halo,
    build_consensus_core,
    classify_tracks_codebook,
    classify_tracks_multiview,
    materialize_candidate_bank,
)
from .v9_replay import CONDITION_FACTORS, replay_candidate_bank


CLASSIFIERS = ("mv-label", "codebook")
ASSOCIATION_MODES = ("A0", "A1", "A2", "A3")


def _ragged_row(indptr: np.ndarray, values: np.ndarray, index: int) -> np.ndarray:
    start, stop = int(indptr[index]), int(indptr[index + 1])
    return np.asarray(values[start:stop])


def _pack_ragged(rows: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.asarray([len(row) for row in rows], dtype=np.int64)
    indptr = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lengths)))
    values = (
        np.concatenate([np.asarray(row, dtype=np.int32) for row in rows])
        if int(indptr[-1])
        else np.empty(0, dtype=np.int32)
    )
    return indptr, values


def _weighted_iou(
    left_ids: np.ndarray,
    left_mass: np.ndarray,
    right_ids: np.ndarray,
    right_mass: np.ndarray,
) -> float:
    union = np.union1d(left_ids, right_ids)
    if not len(union):
        return 0.0
    left = np.zeros(len(union), dtype=np.float64)
    right = np.zeros(len(union), dtype=np.float64)
    left[np.searchsorted(union, left_ids)] = np.asarray(left_mass, dtype=np.float64)
    right[np.searchsorted(union, right_ids)] = np.asarray(right_mass, dtype=np.float64)
    denominator = float(np.maximum(left, right).sum())
    return float(np.minimum(left, right).sum() / denominator) if denominator else 0.0


def _load_fragments(arrays: Mapping[str, np.ndarray]) -> list[Fragment]:
    full_indptr = np.asarray(arrays["fragment_full_indptr"])
    core_indptr = np.asarray(arrays["fragment_core_indptr"])
    output: list[Fragment] = []
    for index, fragment_id in enumerate(np.asarray(arrays["fragment_id"])):
        output.append(
            Fragment(
                fragment_id=int(fragment_id),
                frame_id=int(arrays["fragment_frame"][index]),
                mask_index=int(arrays["fragment_mask_index"][index]),
                full_ids=_ragged_row(full_indptr, arrays["fragment_full_ids"], index),
                core_ids=_ragged_row(core_indptr, arrays["fragment_core_ids"], index),
                full_mass=_ragged_row(full_indptr, arrays["fragment_full_mass"], index),
                core_mass=_ragged_row(core_indptr, arrays["fragment_core_mass"], index),
                conflict_ratio=float(arrays["fragment_conflict_ratio"][index]),
            )
        )
    return output


def _load_frames(
    arrays: Mapping[str, np.ndarray],
    fragments: Sequence[Fragment],
    frame_count: int,
) -> list[FrameEvidence]:
    visible_indptr = np.asarray(arrays["frame_visible_indptr"])
    abstained = np.asarray(
        arrays.get("frame_geometry_abstained", np.zeros(frame_count, dtype=bool)),
        dtype=bool,
    )
    by_frame: dict[int, list[Fragment]] = {}
    for fragment in fragments:
        by_frame.setdefault(int(fragment.frame_id), []).append(fragment)
    return [
        FrameEvidence(
            frame_id=frame_id,
            fragments=tuple(by_frame.get(frame_id, ())),
            visible_ids=_ragged_row(
                visible_indptr, arrays["frame_visible_ids"], frame_id
            ),
            visible_mass=_ragged_row(
                visible_indptr, arrays["frame_visible_mass"], frame_id
            ),
            abstain=bool(abstained[frame_id]),
        )
        for frame_id in range(frame_count)
    ]


def _semantic_rows(
    arrays: Mapping[str, np.ndarray],
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    indptr = np.asarray(arrays["semantic_fragment_full_indptr"])
    classes = np.asarray(arrays["semantic_fragment_class"])
    return [
        (
            int(arrays["semantic_fragment_frame"][index]),
            int(classes[index]),
            _ragged_row(indptr, arrays["semantic_fragment_full_ids"], index),
            _ragged_row(indptr, arrays["semantic_fragment_full_mass"], index),
        )
        for index in range(len(classes))
    ]


def _multiview_votes(
    association: AssociationResult,
    fragments: Sequence[Fragment],
    semantic_rows: Sequence[tuple[int, int, np.ndarray, np.ndarray]],
) -> dict[int, list[MultiviewClassVote]]:
    fragment_by_id = {int(row.fragment_id): row for row in fragments}
    semantic_by_frame: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for frame_id, class_id, ids, mass in semantic_rows:
        semantic_by_frame.setdefault(frame_id, []).append((class_id, ids, mass))
    output: dict[int, list[MultiviewClassVote]] = {}
    for track in association.tracks:
        for fragment_id in track.fragment_ids:
            fragment = fragment_by_id[int(fragment_id)]
            best_by_class: dict[int, float] = {}
            for class_id, ids, mass in semantic_by_frame.get(fragment.frame_id, ()):
                if class_id < 0:
                    continue
                score = _weighted_iou(
                    fragment.full_ids, fragment.full_mass, ids, mass
                )
                best_by_class[class_id] = max(best_by_class.get(class_id, 0.0), score)
            for class_id, score in best_by_class.items():
                if score > 0:
                    output.setdefault(int(track.track_id), []).append(
                        MultiviewClassVote(
                            frame_id=int(fragment.frame_id),
                            class_id=int(class_id),
                            weight=float(score),
                        )
                    )
    return output


def _load_lifting_bank(source: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    return load_lifting_bank(source)


def _bank_arrays(bank: CandidateBank) -> dict[str, np.ndarray]:
    full_indptr, full_ids = _pack_ragged(bank.full_ids)
    core_indptr, core_ids = _pack_ragged(bank.core_ids)
    return {
        "core_candidate_id": np.asarray(bank.core_candidate_id, dtype=np.int32),
        "full_candidate_indptr": full_indptr,
        "full_candidate_ids": full_ids,
        "core_candidate_indptr": core_indptr,
        "core_candidate_ids": core_ids,
    }


def object_bank_is_complete(
    output_dir: Path,
    *,
    expected_scene_id: str | None = None,
    expected_mode: str | None = None,
    expected_source_lifting: str | Path | None = None,
    expected_config: Mapping[str, Any] | None = None,
    expected_git_commit: str | None = None,
    expected_lifting_identity: Mapping[str, Any] | None = None,
) -> bool:
    metadata_path = output_dir / "object_bank.json"
    arrays_path = output_dir / "object_bank.npz"
    if not metadata_path.is_file() or not arrays_path.is_file():
        return False
    try:
        metadata = load_json(metadata_path)
        if metadata.get("schema") != "saga-v9-clean-object-bank-v1":
            return False
        if expected_scene_id is not None and metadata.get("scene_id") != expected_scene_id:
            return False
        if expected_mode is not None and metadata.get("association_mode") != expected_mode:
            return False
        if (
            expected_source_lifting is not None
            and Path(str(metadata.get("source_lifting_bank", ""))).resolve()
            != Path(expected_source_lifting).resolve()
        ):
            return False
        if expected_config is not None and metadata.get("config") != dict(expected_config):
            return False
        if expected_git_commit is not None and metadata.get("git_commit") != expected_git_commit:
            return False
        source_identity = metadata.get("source_lifting_identity")
        if not isinstance(source_identity, Mapping):
            return False
        if (
            expected_lifting_identity is not None
            and dict(source_identity) != dict(expected_lifting_identity)
        ):
            return False
        point_count = int(metadata["point_count"])
        with np.load(arrays_path, allow_pickle=False) as arrays:
            xyz = np.asarray(arrays["xyz_m"])
            if xyz.shape != (point_count, 3) or np.any(~np.isfinite(xyz)):
                return False
            expected_track_ids: list[int] | None = None
            for classifier in CLASSIFIERS:
                key = classifier.replace("-", "_")
                rows = metadata["classifiers"][classifier]["candidates"]
                count = len(rows)
                candidate_ids = [int(row.get("candidate_id", -1)) for row in rows]
                track_ids = [int(row.get("track_id", -1)) for row in rows]
                if candidate_ids != list(range(count)) or len(set(track_ids)) != count:
                    return False
                if expected_track_ids is None:
                    expected_track_ids = track_ids
                elif track_ids != expected_track_ids:
                    # Late classifiers attach attributes to the same frozen
                    # geometry.  They may not censor or renumber tracks.
                    return False
                labels = np.asarray(arrays[f"core_candidate_id_{key}"])
                if (
                    labels.shape != (point_count,)
                    or np.any(labels < -1)
                    or np.any(labels >= count)
                ):
                    return False
                for prefix in ("full_candidate", "core_candidate"):
                    indptr = np.asarray(arrays[f"{prefix}_indptr_{key}"])
                    ids = np.asarray(arrays[f"{prefix}_ids_{key}"])
                    if (
                        indptr.shape != (count + 1,)
                        or int(indptr[0]) != 0
                        or np.any(np.diff(indptr) < 0)
                        or int(indptr[-1]) != len(ids)
                        or np.any(ids < 0)
                        or np.any(ids >= point_count)
                    ):
                        return False
                full_ptr = np.asarray(arrays[f"full_candidate_indptr_{key}"])
                full_ids = np.asarray(arrays[f"full_candidate_ids_{key}"])
                core_ptr = np.asarray(arrays[f"core_candidate_indptr_{key}"])
                core_ids = np.asarray(arrays[f"core_candidate_ids_{key}"])
                for candidate_id in range(count):
                    full = full_ids[full_ptr[candidate_id]:full_ptr[candidate_id + 1]]
                    core = core_ids[core_ptr[candidate_id]:core_ptr[candidate_id + 1]]
                    if (
                        (len(full) and np.any(np.diff(full) <= 0))
                        or (len(core) and np.any(np.diff(core) <= 0))
                        or not np.all(np.isin(core, full))
                    ):
                        return False
                    if not np.array_equal(np.flatnonzero(labels == candidate_id), core):
                        return False
            return True
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def build_v9_object_bank(
    lifting_dir: str | Path,
    output_dir: str | Path,
    *,
    association_mode: str,
    config: V9Config = V9Config(),
) -> dict[str, Any]:
    """Build both late-classifier views over one frozen association bank."""

    if association_mode not in ASSOCIATION_MODES:
        raise ValueError(f"unknown V9 association mode: {association_mode}")
    source = Path(lifting_dir).resolve()
    target = Path(output_dir).resolve()
    lifting = load_json(source / "lifting_bank.json")
    if lifting.get("schema") != "saga-v9-native-lifting-bank-v1":
        raise ValueError("V9 object bank requires native lifting metadata")
    lifting_identity = lifting.get("identity")
    if not isinstance(lifting_identity, Mapping):
        raise ValueError("V9 object bank requires an identified source lifting bank")
    lifting_commit = str(lifting_identity.get("git_commit", ""))
    if object_bank_is_complete(
        target,
        expected_scene_id=str(lifting["scene_id"]),
        expected_mode=association_mode,
        expected_source_lifting=source,
        expected_config=config.as_json(),
        expected_git_commit=lifting_commit,
        expected_lifting_identity=lifting_identity,
    ):
        return load_json(target / "object_bank.json")

    # Only a missing bank needs the large lifting arrays.  This keeps a normal
    # resume path O(metadata) and avoids decompressing 0.5--0.6 GiB NPZ files
    # merely to discover that the registered scene/mode is already complete.
    loaded_lifting, arrays = _load_lifting_bank(source)
    if loaded_lifting != lifting:
        raise ValueError("lifting metadata changed while building the object bank")

    started = time.monotonic()
    fragments = _load_fragments(arrays)
    frames = _load_frames(arrays, fragments, int(lifting["frame_count"]))
    xyz = np.asarray(arrays["xyz_m"], dtype=np.float64)
    affinity = np.asarray(arrays["affinity"], dtype=np.float64)
    association = associate_fragments(
        fragments,
        association_mode,
        xyz_m=xyz if association_mode in {"A2", "A3"} else None,
        affinity=affinity if association_mode in {"A2", "A3"} else None,
        config=config,
    )
    consensus = build_consensus_core(
        association, fragments, frames, int(lifting["point_count"]), config
    )
    final_track_id = attach_local_halo(xyz, affinity, consensus, config)
    classes = tuple(map(str, lifting["classes"]))
    mv = classify_tracks_multiview(
        association,
        _multiview_votes(association, fragments, _semantic_rows(arrays)),
        classes,
    )
    codebook = classify_tracks_codebook(
        association,
        consensus,
        np.asarray(arrays["semantic"]),
        np.asarray(arrays["label_features"]),
        classes,
    )
    banks = {
        "mv-label": materialize_candidate_bank(
            xyz, affinity, association, consensus, final_track_id, mv, config
        ),
        "codebook": materialize_candidate_bank(
            xyz, affinity, association, consensus, final_track_id, codebook, config
        ),
    }
    packed: dict[str, np.ndarray] = {"xyz_m": xyz.astype(np.float32)}
    for classifier, bank in banks.items():
        suffix = classifier.replace("-", "_")
        for name, values in _bank_arrays(bank).items():
            packed[f"{name}_{suffix}"] = values
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target / "object_bank.npz", **packed)
    metadata = {
        "schema": "saga-v9-clean-object-bank-v1",
        "scene_id": str(lifting["scene_id"]),
        "point_count": int(lifting["point_count"]),
        "frame_count": int(lifting["frame_count"]),
        "fragment_count": len(fragments),
        "association_mode": association_mode,
        "git_commit": lifting_commit,
        "source_lifting_bank": str(source),
        "source_lifting_identity": dict(lifting_identity),
        "config": config.as_json(),
        "track_count": len(association.tracks),
        "accepted_edge_count": len(association.accepted_edges),
        "graph_edge_count": int(association.graph_edge_count),
        "tracks": [
            {
                "track_id": int(track.track_id),
                "fragment_ids": list(track.fragment_ids),
                "frame_ids": list(track.frame_ids),
                "merge_scores": list(track.merge_scores),
            }
            for track in association.tracks
        ],
        "accepted_edges": [
            {
                "left_fragment_id": int(edge.left_fragment_id),
                "right_fragment_id": int(edge.right_fragment_id),
                "kind": edge.kind,
                "score": float(edge.score),
                "support": int(edge.support),
            }
            for edge in association.accepted_edges
        ],
        "runtime_seconds": float(time.monotonic() - started),
        "classifiers": {
            classifier: {
                "candidate_count": len(bank.candidates),
                "candidates": [dict(row) for row in bank.candidates],
            }
            for classifier, bank in banks.items()
        },
    }
    write_json(target / "object_bank.json", metadata)
    if not object_bank_is_complete(
        target,
        expected_scene_id=str(lifting["scene_id"]),
        expected_mode=association_mode,
        expected_source_lifting=source,
        expected_config=config.as_json(),
        expected_git_commit=lifting_commit,
        expected_lifting_identity=lifting_identity,
    ):
        raise RuntimeError(f"incomplete V9 object bank: {target}")
    return metadata


def load_v9_candidate_bank(
    bank_dir: str | Path, classifier: str
) -> tuple[dict[str, Any], CandidateBank]:
    if classifier not in CLASSIFIERS:
        raise ValueError(f"unknown V9 classifier: {classifier}")
    source = Path(bank_dir).resolve()
    metadata = load_json(source / "object_bank.json")
    if not object_bank_is_complete(source):
        raise ValueError(f"incomplete V9 object bank: {source}")
    suffix = classifier.replace("-", "_")
    with np.load(source / "object_bank.npz", allow_pickle=False) as arrays:
        full_indptr = np.asarray(arrays[f"full_candidate_indptr_{suffix}"])
        core_indptr = np.asarray(arrays[f"core_candidate_indptr_{suffix}"])
        rows = tuple(metadata["classifiers"][classifier]["candidates"])
        bank = CandidateBank(
            point_count=int(metadata["point_count"]),
            association_mode=str(metadata["association_mode"]),
            core_candidate_id=np.asarray(arrays[f"core_candidate_id_{suffix}"]),
            full_ids=tuple(
                _ragged_row(full_indptr, arrays[f"full_candidate_ids_{suffix}"], index)
                for index in range(len(rows))
            ),
            core_ids=tuple(
                _ragged_row(core_indptr, arrays[f"core_candidate_ids_{suffix}"], index)
                for index in range(len(rows))
            ),
            candidates=rows,
        )
    return dict(metadata), bank


def replay_v9_scene(
    *,
    bank_dir: str | Path,
    output_root: str | Path,
    classifier: str,
    condition: str,
    category_priors: str | Path,
    acceptance_threshold: float,
) -> dict[str, Any]:
    metadata, bank = load_v9_candidate_bank(bank_dir, classifier)
    priors = load_json(category_priors)
    result = replay_candidate_bank(
        bank,
        priors,
        condition,
        acceptance_threshold=float(acceptance_threshold),
    )
    scene_id = str(metadata["scene_id"])
    target = Path(output_root).resolve() / condition / scene_id
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "output.json", result.prediction.output_payload())
    diagnostics = {
        "schema": "saga-v9-prior-replay-v1",
        "scene_id": scene_id,
        "condition": condition,
        "classifier": classifier,
        "association_mode": metadata["association_mode"],
        "acceptance_threshold": float(acceptance_threshold),
        "candidate_count": len(bank.candidates),
        "accepted_candidate_ids": list(result.accepted_candidate_ids),
        "rejected_candidate_ids": list(result.rejected_candidate_ids),
        "suppressed_candidate_ids": list(result.suppressed_candidate_ids),
        "dropped_small_candidate_ids": list(result.dropped_small_candidate_ids),
        "candidate_scores": list(result.candidate_scores),
        "instances": result.prediction.instance_metadata,
        "coverage": float(np.mean(result.point_labels >= 0)),
    }
    write_json(target / "diagnostics.json", diagnostics)
    return diagnostics


def replay_v9_priors(
    *,
    bank_root: str | Path,
    output_root: str | Path,
    scene_ids: Sequence[str],
    classifier: str,
    conditions: Sequence[str],
    category_priors: str | Path,
    acceptance_threshold: float,
) -> dict[str, Any]:
    unknown = set(conditions).difference(CONDITION_FACTORS)
    if unknown:
        raise ValueError(f"unknown V9 replay conditions: {sorted(unknown)}")
    root = Path(bank_root)
    rows = [
        replay_v9_scene(
            bank_dir=root / scene_id,
            output_root=output_root,
            classifier=classifier,
            condition=condition,
            category_priors=category_priors,
            acceptance_threshold=acceptance_threshold,
        )
        for scene_id in map(str, scene_ids)
        for condition in map(str, conditions)
    ]
    return {"schema": "saga-v9-prior-replay-summary-v1", "runs": rows}


def run_v9_banks(
    *,
    lifting_root: str | Path,
    output_root: str | Path,
    scene_ids: Sequence[str],
    association_modes: Sequence[str],
    git_commit: str,
) -> dict[str, Any]:
    target = Path(output_root)
    target.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for mode in association_modes:
        if mode not in ASSOCIATION_MODES:
            raise ValueError(f"unknown V9 association mode: {mode}")
        for scene_id in map(str, scene_ids):
            if shutil.disk_usage(target).free / 1024**3 < 80.0:
                raise RuntimeError("V9 requires at least 80 GiB free")
            lifting_dir = Path(lifting_root) / scene_id
            lifting_metadata = load_json(lifting_dir / "lifting_bank.json")
            lifting_identity = lifting_metadata.get("identity")
            if (
                lifting_metadata.get("schema") != "saga-v9-native-lifting-bank-v1"
                or not isinstance(lifting_identity, Mapping)
                or lifting_identity.get("git_commit") != git_commit
            ):
                raise ValueError(f"{scene_id}: lifting bank is not from current commit")
            metadata = build_v9_object_bank(
                lifting_dir,
                target / mode / scene_id,
                association_mode=mode,
            )
            records.append(
                {
                    "scene_id": scene_id,
                    "association_mode": mode,
                    "candidate_counts": {
                        key: int(value["candidate_count"])
                        for key, value in metadata["classifiers"].items()
                    },
                }
            )
    summary = {
        "schema": "saga-v9-bank-run-summary-v1",
        "git_commit": str(git_commit),
        "runs": records,
    }
    write_json(target / "run_summary.json", summary)
    return summary
