from __future__ import annotations

"""Read-only V9 lifting adapter and compact V10 ObjectBank persistence.

The association implementation deliberately lives in :mod:`v10_objectbank`.
This module owns only the filesystem contract, resumption, and the narrow
builder protocol used by the five registered V10 structure conditions.
"""

import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .io import load_json, write_json
from .v9_lifting import V9_LIFTING_SCHEMA
from .v10_lifting_worker import (
    V10_LIFTING_SCHEMA,
    load_compatible_lifting_bank as load_lifting_bank,
)


V10_BANK_SCHEMA = "saga-v10-view-consensus-object-bank-v1"
V10_BANK_IDENTITY_SCHEMA = "saga-v10-object-bank-identity-v1"
V10_STRUCTURE_CONDITIONS = ("P0R0", "P1R0", "P0R1", "P1R1", "VC1")
V10_CLASSIFIERS = ("mv-label", "codebook")
V10_FUNNEL_STAGES = (
    "single_full",
    "single_core",
    "component_full_union",
    "component_core_union",
    "pre_conflict",
    "post_conflict",
    "unique_ownership",
    "final_candidate",
)


@dataclass(frozen=True)
class V10CandidateBank:
    """One immutable candidate geometry bank loaded from disk."""

    point_count: int
    structure_condition: str
    core_candidate_id: np.ndarray
    full_ids: tuple[np.ndarray, ...]
    core_ids: tuple[np.ndarray, ...]
    candidates: tuple[Mapping[str, Any], ...]
    fragments: tuple[Mapping[str, Any], ...]
    tracks: tuple[Mapping[str, Any], ...]


ObjectBankBuilder = Callable[..., Any]


def _ragged_row(indptr: np.ndarray, values: np.ndarray, index: int) -> np.ndarray:
    start, stop = int(indptr[index]), int(indptr[index + 1])
    row = np.asarray(values[start:stop], dtype=np.int32).copy()
    row.setflags(write=False)
    return row


def _pack_ragged(rows: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.asarray([len(row) for row in rows], dtype=np.int64)
    indptr = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lengths)))
    values = (
        np.concatenate([np.asarray(row, dtype=np.int32) for row in rows])
        if int(indptr[-1])
        else np.empty(0, dtype=np.int32)
    )
    return indptr, values


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, np.ndarray):
        return tuple(_freeze_json(item) for item in value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _payload_field(payload: Any, name: str, default: Any = None) -> Any:
    if isinstance(payload, Mapping):
        return payload.get(name, default)
    return getattr(payload, name, default)


def _normalise_rows(value: Any, *, name: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"V10 builder {name} must be a sequence of mappings")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise TypeError(f"V10 builder {name}[{index}] must be a mapping")
        normalized = _json_value(row)
        # Reject NaN/Inf and non-JSON objects before touching the output tree.
        json.dumps(normalized, allow_nan=False)
        rows.append(normalized)
    return rows


def _normalise_masks(
    value: Any,
    *,
    name: str,
    candidate_count: int,
    point_count: int,
) -> tuple[np.ndarray, ...]:
    if isinstance(value, np.ndarray) and value.ndim == 2:
        rows: Sequence[Any] = tuple(value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rows = value
    else:
        raise TypeError(f"V10 builder {name} must be a sequence of index arrays")
    if len(rows) != candidate_count:
        raise ValueError(f"V10 builder {name} count must match candidates")
    output: list[np.ndarray] = []
    for candidate_id, raw in enumerate(rows):
        ids = np.asarray(raw)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise TypeError(f"{name}[{candidate_id}] must be a one-dimensional integer array")
        ids = ids.astype(np.int32, copy=True)
        if (
            np.any(ids < 0)
            or np.any(ids >= point_count)
            or (len(ids) and np.any(np.diff(ids) <= 0))
        ):
            raise ValueError(f"{name}[{candidate_id}] must contain sorted unique in-range ids")
        ids.setflags(write=False)
        output.append(ids)
    return tuple(output)


def _normalise_stage_supports(
    value: Any,
    *,
    point_count: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, np.ndarray]]:
    if not isinstance(value, Mapping):
        raise TypeError("V10 builder stage_supports must be a stage mapping")
    unknown = set(map(str, value)).difference(V10_FUNNEL_STAGES)
    missing = set(V10_FUNNEL_STAGES).difference(map(str, value))
    if unknown or missing:
        raise ValueError(
            "V10 builder stage_supports must contain exactly the registered stages"
        )
    metadata: dict[str, list[dict[str, Any]]] = {}
    arrays: dict[str, np.ndarray] = {}
    for stage in V10_FUNNEL_STAGES:
        raw_rows = value[stage]
        if isinstance(raw_rows, (str, bytes)) or not isinstance(raw_rows, Sequence):
            raise TypeError(f"V10 stage_supports[{stage}] must be a sequence")
        descriptors: list[dict[str, Any]] = []
        supports: list[np.ndarray] = []
        for index, raw in enumerate(raw_rows):
            if not isinstance(raw, Mapping):
                raise TypeError(f"V10 stage_supports[{stage}][{index}] must be a mapping")
            candidate_id = int(raw.get("candidate_id", -1))
            if candidate_id != index:
                raise ValueError(f"V10 {stage} candidate IDs must be contiguous")
            ids = np.asarray(raw.get("gaussian_ids"))
            if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
                raise TypeError(f"V10 {stage}[{index}] gaussian_ids must be integer 1-D")
            ids = ids.astype(np.int32, copy=True)
            if (
                not len(ids)
                or np.any(ids < 0)
                or np.any(ids >= point_count)
                or np.any(np.diff(ids) <= 0)
            ):
                raise ValueError(
                    f"V10 {stage}[{index}] support must be non-empty, sorted and unique"
                )
            descriptor = {
                str(key): _json_value(item)
                for key, item in raw.items()
                if key != "gaussian_ids"
            }
            descriptor["candidate_id"] = candidate_id
            descriptor["support_count"] = len(ids)
            json.dumps(descriptor, allow_nan=False)
            descriptors.append(descriptor)
            supports.append(ids)
        indptr, ids = _pack_ragged(supports)
        metadata[stage] = descriptors
        arrays[f"stage_{stage}_indptr"] = indptr
        arrays[f"stage_{stage}_ids"] = ids
    return metadata, arrays


def _validate_candidate_rows(
    candidates: Sequence[Mapping[str, Any]],
    full_ids: Sequence[np.ndarray],
    core_ids: Sequence[np.ndarray],
) -> np.ndarray:
    point_count = max(
        (int(ids[-1]) + 1 for rows in (full_ids, core_ids) for ids in rows if len(ids)),
        default=0,
    )
    labels = np.full(point_count, -1, dtype=np.int32)
    required = (
        "branch_class",
        "base_score",
        "metric_extents_m",
        "local_surface_density",
        "boundary_ratio_5cm",
    )
    for candidate_id, row in enumerate(candidates):
        if int(row.get("candidate_id", -1)) != candidate_id:
            raise ValueError("V10 candidate_id values must be contiguous from zero")
        if any(field not in row for field in required):
            raise ValueError(f"V10 candidate {candidate_id} lacks replay evidence")
        if not isinstance(row["branch_class"], str) or not row["branch_class"]:
            raise ValueError(f"V10 candidate {candidate_id} has no branch class")
        classifiers = row.get("classifiers")
        if not isinstance(classifiers, Mapping) or set(classifiers) != set(
            V10_CLASSIFIERS
        ):
            raise ValueError(
                f"V10 candidate {candidate_id} must preserve both late classifiers"
            )
        for classifier in V10_CLASSIFIERS:
            evidence = classifiers[classifier]
            if (
                not isinstance(evidence, Mapping)
                or not isinstance(evidence.get("branch_class"), str)
                or not str(evidence.get("branch_class", ""))
                or "classification_eligible" not in evidence
            ):
                raise ValueError(
                    f"V10 candidate {candidate_id} has invalid {classifier} evidence"
                )
        score = float(row["base_score"])
        density = float(row["local_surface_density"])
        boundary = float(row["boundary_ratio_5cm"])
        extents = np.asarray(row["metric_extents_m"], dtype=np.float64)
        if (
            not np.isfinite(score)
            or not 0.0 <= score <= 1.0
            or not np.isfinite(density)
            or density < 0.0
            or not np.isfinite(boundary)
            or not 0.0 <= boundary <= 1.0
            or extents.shape != (3,)
            or np.any(~np.isfinite(extents))
            or np.any(extents < 0.0)
        ):
            raise ValueError(f"V10 candidate {candidate_id} has invalid replay evidence")
        if int(row.get("core_point_count", len(core_ids[candidate_id]))) != len(
            core_ids[candidate_id]
        ):
            raise ValueError(f"V10 candidate {candidate_id} core count disagrees with mask")
        if int(row.get("full_point_count", len(full_ids[candidate_id]))) != len(
            full_ids[candidate_id]
        ):
            raise ValueError(f"V10 candidate {candidate_id} full count disagrees with mask")
        if not np.all(np.isin(core_ids[candidate_id], full_ids[candidate_id])):
            raise ValueError(f"V10 candidate {candidate_id} core must be a subset of full")
        if len(core_ids[candidate_id]):
            if len(labels) <= int(core_ids[candidate_id][-1]):
                labels = np.pad(
                    labels,
                    (0, int(core_ids[candidate_id][-1]) + 1 - len(labels)),
                    constant_values=-1,
                )
            if np.any(labels[core_ids[candidate_id]] >= 0):
                raise ValueError("V10 candidate cores must have unique ownership")
            labels[core_ids[candidate_id]] = candidate_id
    return labels


def _normalise_builder_payload(
    payload: Any,
    *,
    point_count: int,
    condition: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    built_point_count = int(_payload_field(payload, "point_count", point_count))
    if built_point_count != point_count or built_point_count <= 0:
        raise ValueError("V10 builder point_count must match the source lifting bank")
    candidates = _normalise_rows(_payload_field(payload, "candidates"), name="candidates")
    fragments = _normalise_rows(_payload_field(payload, "fragments", ()), name="fragments")
    tracks = _normalise_rows(_payload_field(payload, "tracks", ()), name="tracks")
    full_ids = _normalise_masks(
        _payload_field(payload, "full_ids"),
        name="full_ids",
        candidate_count=len(candidates),
        point_count=point_count,
    )
    core_ids = _normalise_masks(
        _payload_field(payload, "core_ids"),
        name="core_ids",
        candidate_count=len(candidates),
        point_count=point_count,
    )
    core_labels = _validate_candidate_rows(candidates, full_ids, core_ids)
    if len(core_labels) < point_count:
        core_labels = np.pad(core_labels, (0, point_count - len(core_labels)), constant_values=-1)
    full_indptr, full_values = _pack_ragged(full_ids)
    core_indptr, core_values = _pack_ragged(core_ids)
    accepted_edges = _normalise_rows(
        _payload_field(payload, "accepted_edges", ()), name="accepted_edges"
    )
    _validate_accepted_edges(
        accepted_edges, fragments, tracks, condition=condition
    )
    stage_metadata, stage_arrays = _normalise_stage_supports(
        _payload_field(payload, "stage_supports"), point_count=point_count
    )
    metadata = {
        "condition": condition,
        "fragments": fragments,
        "tracks": tracks,
        "candidates": candidates,
        "accepted_edges": accepted_edges,
        "stage_supports": stage_metadata,
        "diagnostics": _json_value(_payload_field(payload, "diagnostics", {})),
    }
    arrays = {
        "core_candidate_id": core_labels.astype(np.int32, copy=False),
        "full_candidate_indptr": full_indptr,
        "full_candidate_ids": full_values,
        "core_candidate_indptr": core_indptr,
        "core_candidate_ids": core_values,
        **stage_arrays,
    }
    return metadata, arrays


def _validate_accepted_edges(
    accepted_edges: Sequence[Mapping[str, Any]],
    fragments: Sequence[Mapping[str, Any]],
    tracks: Sequence[Mapping[str, Any]],
    *,
    condition: str,
) -> None:
    """Validate persisted association evidence at the filesystem boundary.

    Metrics consume these rows as the *actual* accepted support pairs.  A stale
    endpoint, copied frame ID, or proxy edge would therefore silently corrupt
    the association precision audit unless rejected before persistence.
    """

    fragments_by_id: dict[int, int] = {}
    for index, fragment in enumerate(fragments):
        fragment_id = int(fragment.get("fragment_id", -1))
        frame_id = int(fragment.get("frame_id", -1))
        if fragment_id < 0 or frame_id < 0:
            raise ValueError(f"V10 fragments[{index}] has invalid identity")
        if fragment_id in fragments_by_id:
            raise ValueError(f"V10 fragment ID {fragment_id} is duplicated")
        fragments_by_id[fragment_id] = frame_id

    fragment_track: dict[int, int] = {}
    track_fragments: dict[int, tuple[int, ...]] = {}
    for index, track in enumerate(tracks):
        track_id = int(track.get("track_id", -1))
        fragment_ids = tuple(int(value) for value in track.get("fragment_ids", ()))
        if track_id < 0 or track_id in track_fragments or not fragment_ids:
            raise ValueError(f"V10 tracks[{index}] has invalid identity or membership")
        if len(set(fragment_ids)) != len(fragment_ids):
            raise ValueError(f"V10 tracks[{index}] duplicates a fragment")
        if any(fragment_id not in fragments_by_id for fragment_id in fragment_ids):
            raise ValueError(f"V10 tracks[{index}] references an unknown fragment")
        frame_ids = tuple(fragments_by_id[fragment_id] for fragment_id in fragment_ids)
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError(f"V10 tracks[{index}] contains same-frame alternatives")
        declared_frames = tuple(int(value) for value in track.get("frame_ids", frame_ids))
        if tuple(sorted(declared_frames)) != tuple(sorted(frame_ids)):
            raise ValueError(f"V10 tracks[{index}] has stale frame membership")
        for fragment_id in fragment_ids:
            if fragment_id in fragment_track:
                raise ValueError(f"V10 fragment {fragment_id} belongs to multiple tracks")
            fragment_track[fragment_id] = track_id
        track_fragments[track_id] = fragment_ids
    if set(fragment_track) != set(fragments_by_id):
        raise ValueError("V10 persisted tracks do not partition all fragments")

    seen_pairs: set[tuple[int, int]] = set()
    parent = {fragment_id: fragment_id for fragment_id in fragments_by_id}

    def find(fragment_id: int) -> int:
        while parent[fragment_id] != fragment_id:
            parent[fragment_id] = parent[parent[fragment_id]]
            fragment_id = parent[fragment_id]
        return fragment_id

    edge_count_by_track = {track_id: 0 for track_id in track_fragments}
    for index, edge in enumerate(accepted_edges):
        prefix = f"V10 accepted_edges[{index}]"
        left = int(edge.get("left_fragment_id", -1))
        right = int(edge.get("right_fragment_id", -1))
        if left < 0 or right < 0 or left == right:
            raise ValueError(f"{prefix} has invalid fragment IDs")
        if left not in fragments_by_id or right not in fragments_by_id:
            raise ValueError(f"{prefix} references an unknown fragment")
        pair = tuple(sorted((left, right)))
        if pair in seen_pairs:
            raise ValueError(f"{prefix} duplicates an accepted fragment pair")
        seen_pairs.add(pair)
        left_track = fragment_track[left]
        right_track = fragment_track[right]
        if left_track != right_track:
            raise ValueError(f"{prefix} is not evidence for one persisted track")
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            raise ValueError(f"{prefix} creates a cycle in the persisted merge forest")
        parent[right_root] = left_root
        edge_count_by_track[left_track] += 1

        left_frame = int(edge.get("left_frame_id", -1))
        right_frame = int(edge.get("right_frame_id", -1))
        if (
            left_frame != fragments_by_id[left]
            or right_frame != fragments_by_id[right]
            or left_frame == right_frame
        ):
            raise ValueError(f"{prefix} has inconsistent or same-frame evidence")

        finite_names = (
            "score",
            "frame_weighted_jaccard",
            "p0_overlap",
            "left_coverage",
            "right_coverage",
            "row_margin",
            "column_margin",
            "component_support_ratio",
        )
        values = {name: float(edge.get(name, float("nan"))) for name in finite_names}
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError(f"{prefix} has non-finite association evidence")
        if int(edge.get("shared", -1)) < 3:
            raise ValueError(f"{prefix} has fewer than three shared Gaussians")
        if values["row_margin"] < 0.10 or values["column_margin"] < 0.10:
            raise ValueError(f"{prefix} violates the registered match margin")

        if condition in {"P0R0", "P0R1"}:
            if values["p0_overlap"] < 0.25:
                raise ValueError(f"{prefix} violates the P0 overlap threshold")
            if bool(edge.get("strong", False)) or bool(edge.get("cycle_supported", False)):
                raise ValueError(f"{prefix} assigns view-consensus flags to a P0 edge")
        else:
            if values["left_coverage"] < 0.25 or values["right_coverage"] < 0.25:
                raise ValueError(f"{prefix} violates the P1 bidirectional coverage threshold")

        if condition == "VC1":
            strong = bool(edge.get("strong", False))
            cycle = bool(edge.get("cycle_supported", False))
            if not (strong or cycle):
                raise ValueError(f"{prefix} is neither a strong nor cycle-supported VC1 edge")
            if strong and (
                values["left_coverage"] < 0.80 or values["right_coverage"] < 0.80
            ):
                raise ValueError(f"{prefix} marks a sub-threshold edge as strong")
            if values["component_support_ratio"] < 0.80:
                raise ValueError(f"{prefix} violates the VC1 component consensus threshold")

    for track_id, fragment_ids in track_fragments.items():
        if edge_count_by_track[track_id] != len(fragment_ids) - 1:
            raise ValueError(
                f"V10 track {track_id} is not connected by its persisted accepted edges"
            )


def _default_builder() -> ObjectBankBuilder:
    # Delayed until a bank is genuinely missing, so a completed resume does
    # not import association code or decompress the large lifting NPZ.
    from .v10_objectbank import build_v10_object_bank

    return build_v10_object_bank


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b", dir=path.parent, prefix=f".{path.name}.", suffix=".part", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def v10_object_bank_is_complete(
    output_dir: str | Path,
    *,
    expected_scene_id: str | None = None,
    expected_condition: str | None = None,
    expected_source_lifting: str | Path | None = None,
    expected_lifting_identity: Mapping[str, Any] | None = None,
    expected_git_commit: str | None = None,
) -> bool:
    target = Path(output_dir)
    try:
        metadata = load_json(target / "object_bank.json")
        if metadata.get("schema") != V10_BANK_SCHEMA:
            return False
        scene_id = str(metadata["scene_id"])
        condition = str(metadata["condition"])
        if condition not in V10_STRUCTURE_CONDITIONS:
            return False
        if expected_scene_id is not None and scene_id != expected_scene_id:
            return False
        if expected_condition is not None and condition != expected_condition:
            return False
        if expected_git_commit is not None and metadata.get("git_commit") != expected_git_commit:
            return False
        if expected_source_lifting is not None and Path(
            str(metadata.get("source_lifting_bank", ""))
        ).resolve() != Path(expected_source_lifting).resolve():
            return False
        source_identity = metadata.get("source_lifting_identity")
        if not isinstance(source_identity, Mapping):
            return False
        if expected_lifting_identity is not None and dict(source_identity) != dict(
            expected_lifting_identity
        ):
            return False
        identity = metadata.get("identity")
        if (
            not isinstance(identity, Mapping)
            or identity.get("schema") != V10_BANK_IDENTITY_SCHEMA
            or identity.get("scene_id") != scene_id
            or identity.get("condition") != condition
            or identity.get("source_lifting_identity") != source_identity
        ):
            return False
        point_count = int(metadata["point_count"])
        candidates = metadata["candidates"]
        fragments = metadata["fragments"]
        tracks = metadata["tracks"]
        accepted_edges = metadata["accepted_edges"]
        stage_supports = metadata["stage_supports"]
        if (
            point_count <= 0
            or not isinstance(candidates, list)
            or not isinstance(fragments, list)
            or not isinstance(tracks, list)
            or not isinstance(accepted_edges, list)
            or not isinstance(stage_supports, Mapping)
            or int(metadata["candidate_count"]) != len(candidates)
            or int(metadata["fragment_count"]) != len(fragments)
            or int(metadata["track_count"]) != len(tracks)
            or int(metadata.get("accepted_edge_count", -1)) != len(accepted_edges)
        ):
            return False
        _validate_accepted_edges(
            accepted_edges, fragments, tracks, condition=condition
        )
        with np.load(target / "object_bank.npz", allow_pickle=False) as arrays:
            required = {
                "core_candidate_id",
                "full_candidate_indptr",
                "full_candidate_ids",
                "core_candidate_indptr",
                "core_candidate_ids",
            }
            for stage in V10_FUNNEL_STAGES:
                required.update(
                    {f"stage_{stage}_indptr", f"stage_{stage}_ids"}
                )
            if not required.issubset(arrays.files):
                return False
            count = len(candidates)
            full_ptr = np.asarray(arrays["full_candidate_indptr"])
            full_ids = np.asarray(arrays["full_candidate_ids"])
            core_ptr = np.asarray(arrays["core_candidate_indptr"])
            core_ids = np.asarray(arrays["core_candidate_ids"])
            labels = np.asarray(arrays["core_candidate_id"])
            if (
                labels.shape != (point_count,)
                or not np.issubdtype(labels.dtype, np.integer)
                or np.any(labels < -1)
                or np.any(labels >= count)
            ):
                return False
            for indptr, values in ((full_ptr, full_ids), (core_ptr, core_ids)):
                if (
                    indptr.shape != (count + 1,)
                    or not np.issubdtype(indptr.dtype, np.integer)
                    or int(indptr[0]) != 0
                    or np.any(np.diff(indptr) < 0)
                    or int(indptr[-1]) != len(values)
                    or not np.issubdtype(values.dtype, np.integer)
                    or np.any(values < 0)
                    or np.any(values >= point_count)
                ):
                    return False
            loaded_full = tuple(
                np.asarray(full_ids[int(full_ptr[i]) : int(full_ptr[i + 1])])
                for i in range(count)
            )
            loaded_core = tuple(
                np.asarray(core_ids[int(core_ptr[i]) : int(core_ptr[i + 1])])
                for i in range(count)
            )
            expected_labels = _validate_candidate_rows(candidates, loaded_full, loaded_core)
            if len(expected_labels) < point_count:
                expected_labels = np.pad(
                    expected_labels,
                    (0, point_count - len(expected_labels)),
                    constant_values=-1,
                )
            if not np.array_equal(labels, expected_labels):
                return False
            if set(stage_supports) != set(V10_FUNNEL_STAGES):
                return False
            for stage in V10_FUNNEL_STAGES:
                descriptors = stage_supports[stage]
                indptr = np.asarray(arrays[f"stage_{stage}_indptr"])
                ids = np.asarray(arrays[f"stage_{stage}_ids"])
                if (
                    not isinstance(descriptors, list)
                    or indptr.shape != (len(descriptors) + 1,)
                    or int(indptr[0]) != 0
                    or np.any(np.diff(indptr) <= 0)
                    or int(indptr[-1]) != len(ids)
                    or np.any(ids < 0)
                    or np.any(ids >= point_count)
                ):
                    return False
                for index, descriptor in enumerate(descriptors):
                    row = ids[int(indptr[index]) : int(indptr[index + 1])]
                    if (
                        int(descriptor.get("candidate_id", -1)) != index
                        or int(descriptor.get("support_count", -1)) != len(row)
                        or np.any(np.diff(row) <= 0)
                    ):
                        return False
            return True
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _lifting_header(lifting_dir: Path) -> dict[str, Any]:
    metadata = load_json(lifting_dir / "lifting_bank.json")
    identity = metadata.get("identity")
    if (
        metadata.get("schema") not in {V9_LIFTING_SCHEMA, V10_LIFTING_SCHEMA}
        or not isinstance(identity, Mapping)
        or not str(identity.get("git_commit", "")).strip()
    ):
        raise ValueError(
            f"V10 requires a registered V9/V10 S-AM lifting bank: {lifting_dir}"
        )
    return dict(metadata)


def _readonly_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for name, raw in arrays.items():
        value = np.asarray(raw)
        value.setflags(write=False)
        output[str(name)] = value
    return output


def _write_bank(
    *,
    target: Path,
    lifting_dir: Path,
    lifting_metadata: Mapping[str, Any],
    condition: str,
    git_commit: str,
    payload: Any,
) -> dict[str, Any]:
    normalized, arrays = _normalise_builder_payload(
        payload,
        point_count=int(lifting_metadata["point_count"]),
        condition=condition,
    )
    source_identity = dict(lifting_metadata["identity"])
    metadata = {
        "schema": V10_BANK_SCHEMA,
        "scene_id": str(lifting_metadata["scene_id"]),
        "condition": condition,
        "git_commit": str(git_commit),
        "point_count": int(lifting_metadata["point_count"]),
        "frame_count": int(lifting_metadata["frame_count"]),
        "fragment_count": len(normalized["fragments"]),
        "track_count": len(normalized["tracks"]),
        "candidate_count": len(normalized["candidates"]),
        "accepted_edge_count": len(normalized["accepted_edges"]),
        "source_lifting_bank": str(lifting_dir.resolve()),
        "source_lifting_identity": source_identity,
        "identity": {
            "schema": V10_BANK_IDENTITY_SCHEMA,
            "scene_id": str(lifting_metadata["scene_id"]),
            "condition": condition,
            "source_lifting_identity": source_identity,
            "git_commit": str(git_commit),
        },
        **normalized,
    }
    target.mkdir(parents=True, exist_ok=True)
    _atomic_npz(target / "object_bank.npz", arrays)
    write_json(target / "object_bank.json", metadata)
    if not v10_object_bank_is_complete(
        target,
        expected_scene_id=str(lifting_metadata["scene_id"]),
        expected_condition=condition,
        expected_source_lifting=lifting_dir,
        expected_lifting_identity=source_identity,
        expected_git_commit=str(git_commit),
    ):
        raise RuntimeError(f"incomplete V10 ObjectBank: {target}")
    return metadata


def run_v10_banks(
    *,
    lifting_root: str | Path,
    output_root: str | Path,
    scene_ids: Sequence[str],
    git_commit: str,
    conditions: Sequence[str] = V10_STRUCTURE_CONDITIONS,
    builder: ObjectBankBuilder | None = None,
) -> dict[str, Any]:
    """Build the five V10 structure arms while loading each lifting NPZ once."""

    requested = tuple(map(str, conditions))
    unknown = set(requested).difference(V10_STRUCTURE_CONDITIONS)
    if unknown or len(set(requested)) != len(requested):
        raise ValueError(f"invalid V10 structure conditions: {sorted(unknown)}")
    commit = str(git_commit).strip()
    if not commit:
        raise ValueError("V10 git_commit must be non-empty")
    lifting_root = Path(lifting_root).resolve()
    output_root = Path(output_root).resolve()
    records: list[dict[str, Any]] = []
    for scene_id in map(str, scene_ids):
        lifting_dir = lifting_root / scene_id
        header = _lifting_header(lifting_dir)
        if str(header.get("scene_id")) != scene_id:
            raise ValueError(f"lifting scene mismatch for {scene_id}")
        identity = dict(header["identity"])
        missing: list[str] = []
        for condition in requested:
            target = output_root / condition / scene_id
            if v10_object_bank_is_complete(
                target,
                expected_scene_id=scene_id,
                expected_condition=condition,
                expected_source_lifting=lifting_dir,
                expected_lifting_identity=identity,
                expected_git_commit=commit,
            ):
                metadata = load_json(target / "object_bank.json")
                records.append(
                    {
                        "scene_id": scene_id,
                        "condition": condition,
                        "status": "reused",
                        "candidate_count": int(metadata["candidate_count"]),
                    }
                )
            else:
                missing.append(condition)
        if not missing:
            continue
        loaded_metadata, loaded_arrays = load_lifting_bank(lifting_dir)
        if loaded_metadata != header:
            raise ValueError(f"lifting metadata changed while reading {scene_id}")
        frozen_metadata = _freeze_json(loaded_metadata)
        frozen_arrays = _readonly_arrays(loaded_arrays)
        active_builder = builder if builder is not None else _default_builder()
        for condition in missing:
            payload = active_builder(
                frozen_metadata,
                MappingProxyType(frozen_arrays),
                condition=condition,
            )
            metadata = _write_bank(
                target=output_root / condition / scene_id,
                lifting_dir=lifting_dir,
                lifting_metadata=loaded_metadata,
                condition=condition,
                git_commit=commit,
                payload=payload,
            )
            records.append(
                {
                    "scene_id": scene_id,
                    "condition": condition,
                    "status": "completed",
                    "candidate_count": int(metadata["candidate_count"]),
                }
            )
    summary = {
        "schema": "saga-v10-bank-run-summary-v1",
        "git_commit": commit,
        "conditions": list(requested),
        "runs": records,
    }
    write_json(output_root / "run_summary.json", summary)
    return summary


def load_v10_candidate_bank(
    bank_dir: str | Path,
) -> tuple[dict[str, Any], V10CandidateBank]:
    source = Path(bank_dir).resolve()
    if not v10_object_bank_is_complete(source):
        raise ValueError(f"incomplete V10 ObjectBank: {source}")
    metadata = load_json(source / "object_bank.json")
    count = int(metadata["candidate_count"])
    with np.load(source / "object_bank.npz", allow_pickle=False) as arrays:
        full_ptr = np.asarray(arrays["full_candidate_indptr"])
        core_ptr = np.asarray(arrays["core_candidate_indptr"])
        labels = np.asarray(arrays["core_candidate_id"], dtype=np.int32).copy()
        labels.setflags(write=False)
        bank = V10CandidateBank(
            point_count=int(metadata["point_count"]),
            structure_condition=str(metadata["condition"]),
            core_candidate_id=labels,
            full_ids=tuple(
                _ragged_row(full_ptr, arrays["full_candidate_ids"], index)
                for index in range(count)
            ),
            core_ids=tuple(
                _ragged_row(core_ptr, arrays["core_candidate_ids"], index)
                for index in range(count)
            ),
            candidates=tuple(_freeze_json(row) for row in metadata["candidates"]),
            fragments=tuple(_freeze_json(row) for row in metadata["fragments"]),
            tracks=tuple(_freeze_json(row) for row in metadata["tracks"]),
        )
    return dict(metadata), bank


def load_v10_audit_supports(
    bank_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, tuple[np.ndarray, ...]]]:
    """Load the immutable eight-stage support funnel for offline GT audit."""

    source = Path(bank_dir).resolve()
    metadata = load_json(source / "object_bank.json")
    if not v10_object_bank_is_complete(source):
        raise ValueError(f"incomplete V10 ObjectBank: {source}")
    result: dict[str, tuple[np.ndarray, ...]] = {}
    with np.load(source / "object_bank.npz", allow_pickle=False) as arrays:
        for stage in V10_FUNNEL_STAGES:
            indptr = np.asarray(arrays[f"stage_{stage}_indptr"])
            ids = np.asarray(arrays[f"stage_{stage}_ids"])
            rows: list[np.ndarray] = []
            for index in range(len(indptr) - 1):
                row = np.asarray(
                    ids[int(indptr[index]) : int(indptr[index + 1])],
                    dtype=np.int32,
                ).copy()
                row.setflags(write=False)
                rows.append(row)
            result[stage] = tuple(rows)
    return dict(metadata), result
