from __future__ import annotations

"""Path-level V8 object-bank construction and immutable prior replay."""

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .io import load_json, write_json
from .v8_objects import (
    CandidateBank,
    Fragment,
    FrameEvidence,
    MultiViewLabelVote,
    V8Config,
    associate_fragments,
    build_consensus_assignment,
    classify_tracks_codebook,
    classify_tracks_multiview,
    materialize_candidates,
)
from .v8_replay import CONDITIONS, replay_candidates


CLASSIFIERS = ("mv-label", "codebook")


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
    left_ids = np.asarray(left_ids, dtype=np.int64)
    right_ids = np.asarray(right_ids, dtype=np.int64)
    left_mass = np.asarray(left_mass, dtype=np.float64)
    right_mass = np.asarray(right_mass, dtype=np.float64)
    union = np.union1d(left_ids, right_ids)
    if not len(union):
        return 0.0
    left = np.zeros(len(union), dtype=np.float64)
    right = np.zeros(len(union), dtype=np.float64)
    left[np.searchsorted(union, left_ids)] = left_mass
    right[np.searchsorted(union, right_ids)] = right_mass
    denominator = float(np.maximum(left, right).sum())
    return float(np.minimum(left, right).sum() / denominator) if denominator else 0.0


def _load_fragments(arrays: Mapping[str, np.ndarray]) -> list[Fragment]:
    full_indptr = np.asarray(arrays["fragment_full_indptr"])
    core_indptr = np.asarray(arrays["fragment_core_indptr"])
    fragments: list[Fragment] = []
    for index, fragment_id in enumerate(np.asarray(arrays["fragment_id"])):
        fragments.append(
            Fragment(
                fragment_id=int(fragment_id),
                frame_id=int(arrays["fragment_frame"][index]),
                mask_index=int(arrays["fragment_mask_index"][index]),
                full_ids=_ragged_row(
                    full_indptr, arrays["fragment_full_ids"], index
                ),
                core_ids=_ragged_row(
                    core_indptr, arrays["fragment_core_ids"], index
                ),
                full_mass=_ragged_row(
                    full_indptr, arrays["fragment_full_mass"], index
                ),
                core_mass=_ragged_row(
                    core_indptr, arrays["fragment_core_mass"], index
                ),
            )
        )
    return fragments


def _load_frames(
    arrays: Mapping[str, np.ndarray],
    fragments: Sequence[Fragment],
    frame_count: int,
) -> list[FrameEvidence]:
    visible_indptr = np.asarray(arrays["frame_visible_indptr"])
    missing = np.asarray(
        arrays.get("frame_geometry_abstained", np.zeros(frame_count, dtype=bool)),
        dtype=bool,
    )
    if missing.shape != (frame_count,):
        raise ValueError("frame_geometry_abstained must have one entry per frame")
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
            grounded_missing=bool(missing[frame_id]),
        )
        for frame_id in range(frame_count)
    ]


def _semantic_fragments(
    arrays: Mapping[str, np.ndarray],
) -> list[tuple[int, int, int, np.ndarray, np.ndarray]]:
    indptr = np.asarray(arrays["semantic_fragment_full_indptr"])
    rows: list[tuple[int, int, int, np.ndarray, np.ndarray]] = []
    classes = np.asarray(arrays["semantic_fragment_class"])
    for index in range(len(classes)):
        rows.append(
            (
                int(arrays["semantic_fragment_id"][index]),
                int(arrays["semantic_fragment_frame"][index]),
                int(classes[index]),
                _ragged_row(indptr, arrays["semantic_fragment_full_ids"], index),
                _ragged_row(indptr, arrays["semantic_fragment_full_mass"], index),
            )
        )
    return rows


def _multiview_votes(
    tracks: Sequence[Any],
    fragments: Sequence[Fragment],
    semantic: Sequence[tuple[int, int, int, np.ndarray, np.ndarray]],
) -> dict[int, list[MultiViewLabelVote]]:
    fragment_by_id = {int(item.fragment_id): item for item in fragments}
    semantic_by_frame: dict[int, list[tuple[int, int, int, np.ndarray, np.ndarray]]] = {}
    for row in semantic:
        semantic_by_frame.setdefault(int(row[1]), []).append(row)
    output: dict[int, list[MultiViewLabelVote]] = {}
    for track in tracks:
        for fragment_id in track.fragment_ids:
            fragment = fragment_by_id[int(fragment_id)]
            best_by_class: dict[int, float] = {}
            for _semantic_id, _frame, class_id, ids, mass in semantic_by_frame.get(
                int(fragment.frame_id), ()
            ):
                if class_id < 0:
                    continue
                score = _weighted_iou(
                    fragment.full_ids, fragment.full_mass, ids, mass
                )
                best_by_class[class_id] = max(best_by_class.get(class_id, 0.0), score)
            for class_id, score in best_by_class.items():
                output.setdefault(int(track.track_id), []).append(
                    MultiViewLabelVote(
                        frame_id=int(fragment.frame_id),
                        class_id=int(class_id),
                        weighted_iou=float(score),
                    )
                )
    return output


def object_bank_is_complete(output_dir: Path) -> bool:
    metadata_path = output_dir / "object_bank.json"
    arrays_path = output_dir / "object_bank.npz"
    if not metadata_path.is_file() or not arrays_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        point_count = int(metadata["point_count"])
        if metadata.get("schema") != "saga-v8-object-bank-v1":
            return False
        with np.load(arrays_path, allow_pickle=False) as arrays:
            valid_track_count = int(metadata.get("valid_track_count", 0))
            required_common = {
                "xyz_m", "core_track_id", "valid_track_ids",
                "track_full_indptr", "track_full_ids",
                "track_core_indptr", "track_core_ids",
            }
            if not required_common.issubset(arrays.files):
                return False
            if arrays["xyz_m"].shape != (point_count, 3):
                return False
            if arrays["core_track_id"].shape != (point_count,):
                return False
            if arrays["valid_track_ids"].shape != (valid_track_count,):
                return False
            if arrays["track_full_indptr"].shape != (valid_track_count + 1,):
                return False
            if arrays["track_core_indptr"].shape != (valid_track_count + 1,):
                return False
            for prefix in ("track_full", "track_core"):
                indptr = np.asarray(arrays[f"{prefix}_indptr"], dtype=np.int64)
                ids = np.asarray(arrays[f"{prefix}_ids"], dtype=np.int64)
                if (
                    not len(indptr)
                    or int(indptr[0]) != 0
                    or np.any(np.diff(indptr) < 0)
                    or int(indptr[-1]) != len(ids)
                    or np.any(ids < 0)
                    or np.any(ids >= point_count)
                ):
                    return False
            core_track = np.asarray(arrays["core_track_id"], dtype=np.int64)
            if np.any(~np.isin(core_track, np.append(
                np.asarray(arrays["valid_track_ids"], dtype=np.int64), -1
            ))):
                return False
            track_core_indptr = np.asarray(arrays["track_core_indptr"], dtype=np.int64)
            track_core_ids = np.asarray(arrays["track_core_ids"], dtype=np.int64)
            for row_index, track_id in enumerate(
                np.asarray(arrays["valid_track_ids"], dtype=np.int64)
            ):
                ids = track_core_ids[
                    track_core_indptr[row_index]:track_core_indptr[row_index + 1]
                ]
                if not np.array_equal(np.flatnonzero(core_track == track_id), ids):
                    return False
            for classifier, metadata_key in (
                ("mv", "mv-label"), ("codebook", "codebook")
            ):
                candidate_count = int(
                    metadata["classifiers"][metadata_key]["candidate_count"]
                )
                required = {
                    f"core_candidate_id_{classifier}",
                    f"full_candidate_indptr_{classifier}",
                    f"full_candidate_ids_{classifier}",
                    f"core_candidate_indptr_{classifier}",
                    f"core_candidate_ids_{classifier}",
                }
                if not required.issubset(arrays.files):
                    return False
                if arrays[f"core_candidate_id_{classifier}"].shape != (point_count,):
                    return False
                if arrays[f"full_candidate_indptr_{classifier}"].shape != (
                    candidate_count + 1,
                ):
                    return False
                if arrays[f"core_candidate_indptr_{classifier}"].shape != (
                    candidate_count + 1,
                ):
                    return False
                for prefix in ("full_candidate", "core_candidate"):
                    indptr = np.asarray(
                        arrays[f"{prefix}_indptr_{classifier}"], dtype=np.int64
                    )
                    ids = np.asarray(
                        arrays[f"{prefix}_ids_{classifier}"], dtype=np.int64
                    )
                    if (
                        not len(indptr)
                        or int(indptr[0]) != 0
                        or np.any(np.diff(indptr) < 0)
                        or int(indptr[-1]) != len(ids)
                        or np.any(ids < 0)
                        or np.any(ids >= point_count)
                    ):
                        return False
                dense = np.asarray(
                    arrays[f"core_candidate_id_{classifier}"], dtype=np.int64
                )
                if np.any((dense < -1) | (dense >= candidate_count)):
                    return False
                core_indptr = np.asarray(
                    arrays[f"core_candidate_indptr_{classifier}"], dtype=np.int64
                )
                core_ids = np.asarray(
                    arrays[f"core_candidate_ids_{classifier}"], dtype=np.int64
                )
                for candidate_id in range(candidate_count):
                    ids = core_ids[
                        core_indptr[candidate_id]:core_indptr[candidate_id + 1]
                    ]
                    if not np.array_equal(
                        np.flatnonzero(dense == candidate_id), ids
                    ):
                        return False
            return True
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def build_v8_object_bank(
    lifting_dir: str | Path,
    output_dir: str | Path,
    *,
    config: V8Config = V8Config(),
) -> dict[str, Any]:
    source = Path(lifting_dir).resolve()
    target = Path(output_dir).resolve()
    lifting = load_json(source / "lifting_bank.json")
    if lifting.get("schema") != "saga-v8-lifting-bank-v1":
        raise ValueError(f"not a V8 lifting bank: {source}")
    if object_bank_is_complete(target):
        existing = load_json(target / "object_bank.json")
        if (
            str(existing.get("scene_id")) == str(lifting.get("scene_id"))
            and str(existing.get("mask_source")) == str(lifting.get("mask_source"))
            and str(existing.get("lifting_source")) == str(lifting.get("lifting_source"))
            and existing.get("config") == config.as_json()
            and Path(str(existing.get("source_lifting_bank", ""))).resolve() == source
        ):
            return existing
    arrays_path = source / str(lifting.get("arrays_npz", "lifting_bank.npz"))
    with np.load(arrays_path, allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    fragments = _load_fragments(arrays)
    frames = _load_frames(arrays, fragments, int(lifting["frame_count"]))
    tracks = associate_fragments(fragments, config)
    consensus = build_consensus_assignment(
        tracks, fragments, frames, int(lifting["point_count"]), config
    )
    class_names = tuple(map(str, lifting["classes"]))
    mv = classify_tracks_multiview(
        tracks, _multiview_votes(tracks, fragments, _semantic_fragments(arrays)),
        class_names, config,
    )
    label_features = np.asarray(arrays.get("label_features", np.empty((0, 0))))
    if label_features.shape[0] == len(class_names):
        codebook = classify_tracks_codebook(
            tracks,
            consensus,
            arrays["semantic"],
            label_features,
            class_names,
        )
    else:
        codebook = {}
    candidates_mv = materialize_candidates(
        arrays["xyz_m"], tracks, consensus, mv, config
    )
    candidates_codebook = materialize_candidates(
        arrays["xyz_m"], tracks, consensus, codebook, config
    )
    valid_track_ids = np.asarray(consensus.valid_track_ids, dtype=np.int32)
    track_full_indptr, track_full_ids = _pack_ragged(
        [consensus.track_full_ids[int(track_id)] for track_id in valid_track_ids]
    )
    track_core_indptr, track_core_ids = _pack_ragged(
        [consensus.track_core_ids[int(track_id)] for track_id in valid_track_ids]
    )
    mv_full_indptr, mv_full_ids = _pack_ragged(candidates_mv.full_ids)
    mv_core_indptr, mv_core_ids = _pack_ragged(candidates_mv.core_ids)
    code_full_indptr, code_full_ids = _pack_ragged(candidates_codebook.full_ids)
    code_core_indptr, code_core_ids = _pack_ragged(candidates_codebook.core_ids)
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target / "object_bank.npz",
        xyz_m=np.asarray(arrays["xyz_m"], dtype=np.float32),
        core_track_id=consensus.core_track_id,
        valid_track_ids=valid_track_ids,
        track_full_indptr=track_full_indptr,
        track_full_ids=track_full_ids,
        track_core_indptr=track_core_indptr,
        track_core_ids=track_core_ids,
        core_candidate_id_mv=candidates_mv.core_candidate_id,
        core_candidate_id_codebook=candidates_codebook.core_candidate_id,
        full_candidate_indptr_mv=mv_full_indptr,
        full_candidate_ids_mv=mv_full_ids,
        core_candidate_indptr_mv=mv_core_indptr,
        core_candidate_ids_mv=mv_core_ids,
        full_candidate_indptr_codebook=code_full_indptr,
        full_candidate_ids_codebook=code_full_ids,
        core_candidate_indptr_codebook=code_core_indptr,
        core_candidate_ids_codebook=code_core_ids,
    )
    metadata = {
        "schema": "saga-v8-object-bank-v1",
        "scene_id": str(lifting["scene_id"]),
        "git_commit": lifting.get("git_commit"),
        "mask_source": lifting["mask_source"],
        "lifting_source": lifting["lifting_source"],
        "point_count": int(lifting["point_count"]),
        "frame_count": int(lifting["frame_count"]),
        "fragment_count": len(fragments),
        "track_count": len(tracks),
        "valid_track_count": len(consensus.valid_track_ids),
        "classifiers": {
            "mv-label": {
                "candidate_count": len(candidates_mv.candidates),
                "candidates": list(candidates_mv.candidates),
            },
            "codebook": {
                "candidate_count": len(candidates_codebook.candidates),
                "candidates": list(candidates_codebook.candidates),
            },
        },
        "config": config.as_json(),
        "source_lifting_bank": str(source),
        "arrays_npz": "object_bank.npz",
    }
    write_json(target / "object_bank.json", metadata)
    if not object_bank_is_complete(target):
        raise RuntimeError(f"serialized V8 object bank failed validation: {target}")
    return metadata


def load_candidate_bank(
    bank_dir: str | Path,
    classifier: str,
) -> tuple[dict[str, Any], CandidateBank]:
    if classifier not in CLASSIFIERS:
        raise ValueError(f"unknown V8 classifier: {classifier}")
    root = Path(bank_dir).resolve()
    metadata = load_json(root / "object_bank.json")
    key = "mv" if classifier == "mv-label" else "codebook"
    with np.load(root / str(metadata["arrays_npz"]), allow_pickle=False) as arrays:
        candidate_count = int(
            metadata["classifiers"][classifier]["candidate_count"]
        )
        full_indptr = np.asarray(arrays[f"full_candidate_indptr_{key}"])
        core_indptr = np.asarray(arrays[f"core_candidate_indptr_{key}"])
        bank = CandidateBank(
            point_count=int(metadata["point_count"]),
            core_candidate_id=np.asarray(
                arrays[f"core_candidate_id_{key}"], dtype=np.int32
            ),
            full_ids=tuple(
                _ragged_row(full_indptr, arrays[f"full_candidate_ids_{key}"], index)
                for index in range(candidate_count)
            ),
            core_ids=tuple(
                _ragged_row(core_indptr, arrays[f"core_candidate_ids_{key}"], index)
                for index in range(candidate_count)
            ),
            candidates=tuple(metadata["classifiers"][classifier]["candidates"]),
        )
    return metadata, bank


def replay_v8_scene(
    *,
    bank_dir: str | Path,
    output_root: str | Path,
    classifier: str,
    condition: str,
    category_priors: str | Path,
) -> dict[str, Any]:
    metadata, bank = load_candidate_bank(bank_dir, classifier)
    priors = load_json(category_priors)
    result = replay_candidates(bank, priors, condition)
    scene_id = str(metadata["scene_id"])
    target = Path(output_root).resolve() / condition / scene_id
    target.mkdir(parents=True, exist_ok=True)
    write_json(
        target / "output.json",
        {"point_labels": result.point_labels.tolist(), "instances": result.instances},
    )
    diagnostics = {
        "schema": "saga-v8-replay-diagnostics-v1",
        "scene_id": scene_id,
        "condition": condition,
        "classifier": classifier,
        "candidate_count": len(bank.candidates),
        "accepted_candidate_ids": list(result.accepted_candidate_ids),
        "rejected_candidate_ids": list(result.rejected_candidate_ids),
        "suppressed_candidate_ids": list(result.suppressed_candidate_ids),
        "dropped_small_candidate_ids": list(result.dropped_small_candidate_ids),
        "candidate_scores": list(result.candidate_scores),
        "instances": result.instance_metadata,
        "assigned_points": int(np.count_nonzero(result.point_labels >= 0)),
        "coverage": float(np.mean(result.point_labels >= 0))
        if len(result.point_labels)
        else 0.0,
    }
    write_json(target / "diagnostics.json", diagnostics)
    return diagnostics


def replay_v8_priors(
    *,
    bank_root: str | Path,
    output_root: str | Path,
    scene_ids: Sequence[str],
    classifier: str,
    conditions: Sequence[str],
    category_priors: str | Path,
) -> dict[str, Any]:
    unknown = set(conditions).difference(CONDITIONS)
    if unknown:
        raise ValueError(f"unknown V8 replay conditions: {sorted(unknown)}")
    rows = [
        replay_v8_scene(
            bank_dir=Path(bank_root) / scene_id,
            output_root=output_root,
            classifier=classifier,
            condition=condition,
            category_priors=category_priors,
        )
        for scene_id in map(str, scene_ids)
        for condition in map(str, conditions)
    ]
    return {"schema": "saga-v8-prior-replay-v1", "runs": rows}
