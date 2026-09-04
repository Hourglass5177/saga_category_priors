from __future__ import annotations

"""Build the branch-only refinement reservoir without using ground truth."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..candidate_bank import CandidateBank, load_candidate_bank, validate_candidate_bank
from .contracts import CandidateSeed, SCHEMA


STAGE_PRIORITY = ("exported_prediction", "post_filter", "post_global_knn")


def _support_digest(point_ids: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(point_ids, dtype="<i8"))
    digest = hashlib.sha256()
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _best_overlap_anchor(
    support: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, int | None, int]:
    values = np.asarray(labels, dtype=np.int64)
    stage_values = values[support]
    nonnegative = stage_values[stage_values >= 0]
    if not len(nonnegative):
        return np.empty(0, dtype=np.int64), None, 0
    ids, counts = np.unique(nonnegative, return_counts=True)
    maximum = int(counts.max())
    best_id = int(ids[counts == maximum].min())
    anchor = support[stage_values == best_id]
    return anchor.astype(np.int64, copy=False), best_id, maximum


def load_stage_arrays(path: str | Path, point_count: int) -> dict[str, np.ndarray]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as archive:
        result = {
            name: np.asarray(archive[name], dtype=np.int64)
            for name in STAGE_PRIORITY
            if name in archive.files
        }
    if not result:
        raise ValueError("stage trace contains none of the registered anchor stages")
    for name, values in result.items():
        if values.shape != (point_count,):
            raise ValueError(f"stage {name} does not share the CandidateBank point axis")
        if np.any(values < -1):
            raise ValueError(f"stage {name} contains an invalid negative instance ID")
    return result


def build_candidate_reservoir(
    bank: CandidateBank,
    stage_arrays: Mapping[str, Any],
) -> tuple[CandidateSeed, ...]:
    """Return stable branch candidates and their last surviving in-support anchors.

    Exact duplicate full memberships are collapsed, but every original candidate
    remains in ``parent_candidate_ids``.  Stage instance numbers are never used as
    candidate identity; only maximum overlap inside the frozen support is used.
    """

    validate_candidate_bank(bank)
    point_count = bank.point_count
    stages: dict[str, np.ndarray] = {}
    for name in STAGE_PRIORITY:
        if name not in stage_arrays:
            continue
        values = np.asarray(stage_arrays[name], dtype=np.int64)
        if values.shape != (point_count,):
            raise ValueError(f"stage {name} does not share the CandidateBank point axis")
        stages[name] = values
    if not stages:
        raise ValueError("at least one registered downstream stage is required")

    rows_by_id = {int(row["candidate_id"]): row for row in bank.candidates}
    grouped: dict[str, list[int]] = {}
    supports: dict[str, np.ndarray] = {}
    for candidate_id in sorted(rows_by_id):
        support = np.flatnonzero(
            np.asarray(bank.branch_full_labels, dtype=np.int64) == candidate_id
        ).astype(np.int64)
        digest = _support_digest(support)
        grouped.setdefault(digest, []).append(candidate_id)
        supports[digest] = support

    seeds: list[CandidateSeed] = []
    for stable_id, digest in enumerate(sorted(grouped, key=lambda key: min(grouped[key]))):
        parents = tuple(sorted(grouped[digest]))
        canonical = rows_by_id[parents[0]]
        classes = {str(rows_by_id[item]["branch_class"]) for item in parents}
        if len(classes) != 1:
            raise ValueError("identical candidate memberships disagree on branch class")
        q_values = [
            float(rows_by_id[item].get("base_score", rows_by_id[item].get("vote_winner_ratio", 0.0)))
            for item in parents
        ]
        support = supports[digest]
        anchor = np.empty(0, dtype=np.int64)
        anchor_stage: str | None = None
        matched_stage_id: int | None = None
        for stage_name in STAGE_PRIORITY:
            if stage_name not in stages:
                continue
            candidate_anchor, stage_id, _ = _best_overlap_anchor(support, stages[stage_name])
            if len(candidate_anchor):
                anchor = candidate_anchor
                anchor_stage = stage_name
                matched_stage_id = stage_id
                break
        seeds.append(
            CandidateSeed(
                candidate_id=stable_id,
                parent_candidate_ids=parents,
                branch_class=next(iter(classes)),
                seed_support=support,
                seed_anchor=anchor,
                anchor_stage=anchor_stage,
                q_score=max(q_values),
                diagnostics={
                    "support_sha256": digest,
                    "original_candidate_count": len(parents),
                    "matched_stage_instance_id": matched_stage_id,
                    "original_full_point_count": int(canonical.get("full_point_count", len(support))),
                },
            )
        )
    return tuple(seeds)


def save_reservoir(
    output_dir: str | Path,
    seeds: Sequence[CandidateSeed],
    *,
    candidate_bank_path: str | Path,
    stage_trace_path: str | Path,
    b0_labels: Any,
    provenance: Mapping[str, Any],
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(b0_labels, dtype=np.int64)
    if labels.ndim != 1 or np.any(labels < -1):
        raise ValueError("B0 labels must be one-dimensional and use only -1 as background")
    support_indptr = [0]
    support_values: list[int] = []
    anchor_indptr = [0]
    anchor_values: list[int] = []
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        support_values.extend(seed.seed_support.tolist())
        support_indptr.append(len(support_values))
        anchor_values.extend(seed.seed_anchor.tolist())
        anchor_indptr.append(len(anchor_values))
        rows.append(
            {
                "candidate_id": seed.candidate_id,
                "parent_candidate_ids": list(seed.parent_candidate_ids),
                "branch_class": seed.branch_class,
                "anchor_stage": seed.anchor_stage,
                "q_score": seed.q_score,
                "reachable": seed.reachable,
                "diagnostics": dict(seed.diagnostics),
            }
        )
    npz_path = destination / "reservoir.npz"
    temporary_npz = npz_path.with_name(npz_path.name + ".part")
    with temporary_npz.open("wb") as handle:
        np.savez_compressed(
            handle,
            support_indptr=np.asarray(support_indptr, dtype=np.int64),
            support_ids=np.asarray(support_values, dtype=np.int64),
            anchor_indptr=np.asarray(anchor_indptr, dtype=np.int64),
            anchor_ids=np.asarray(anchor_values, dtype=np.int64),
            b0_labels=labels,
        )
    os.replace(temporary_npz, npz_path)
    metadata = {
        "schema": SCHEMA,
        "kind": "candidate-reservoir",
        "candidate_bank": str(Path(candidate_bank_path).resolve()),
        "stage_trace": str(Path(stage_trace_path).resolve()),
        "candidate_count": len(rows),
        "point_count": int(len(labels)),
        "candidates": rows,
        "provenance": dict(provenance),
    }
    json_path = destination / "reservoir.json"
    temporary_json = json_path.with_name(json_path.name + ".part")
    temporary_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_json, json_path)


def load_reservoir(path: str | Path) -> tuple[tuple[CandidateSeed, ...], np.ndarray, dict[str, Any]]:
    root = Path(path)
    metadata = json.loads((root / "reservoir.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != SCHEMA or metadata.get("kind") != "candidate-reservoir":
        raise ValueError("unsupported refinement reservoir")
    with np.load(root / "reservoir.npz", allow_pickle=False) as archive:
        support_indptr = np.asarray(archive["support_indptr"], dtype=np.int64)
        support_ids = np.asarray(archive["support_ids"], dtype=np.int64)
        anchor_indptr = np.asarray(archive["anchor_indptr"], dtype=np.int64)
        anchor_ids = np.asarray(archive["anchor_ids"], dtype=np.int64)
        b0 = np.asarray(archive["b0_labels"], dtype=np.int64)
    rows = metadata.get("candidates", [])
    if len(support_indptr) != len(rows) + 1 or len(anchor_indptr) != len(rows) + 1:
        raise ValueError("reservoir sparse membership arrays are inconsistent")
    seeds = []
    for index, row in enumerate(rows):
        seeds.append(
            CandidateSeed(
                candidate_id=int(row["candidate_id"]),
                parent_candidate_ids=tuple(int(value) for value in row["parent_candidate_ids"]),
                branch_class=str(row["branch_class"]),
                seed_support=support_ids[support_indptr[index] : support_indptr[index + 1]],
                seed_anchor=anchor_ids[anchor_indptr[index] : anchor_indptr[index + 1]],
                anchor_stage=row.get("anchor_stage"),
                q_score=float(row["q_score"]),
                reachable=bool(row.get("reachable", False)),
                diagnostics=row.get("diagnostics", {}),
            )
        )
    if int(metadata.get("point_count", -1)) != len(b0):
        raise ValueError("reservoir B0 point axis is inconsistent")
    b0.setflags(write=False)
    return tuple(seeds), b0, metadata


def prepare_reservoir(
    *,
    candidate_bank_path: str | Path,
    stage_trace_path: str | Path,
    b0_output_path: str | Path,
    output_dir: str | Path,
    provenance: Mapping[str, Any],
) -> tuple[CandidateSeed, ...]:
    bank = load_candidate_bank(candidate_bank_path)
    stages = load_stage_arrays(stage_trace_path, bank.point_count)
    payload = json.loads(Path(b0_output_path).read_text(encoding="utf-8"))
    labels = np.asarray(payload.get("point_labels"), dtype=np.int64)
    if labels.shape != (bank.point_count,):
        raise ValueError("B0 output does not share the CandidateBank point axis")
    seeds = build_candidate_reservoir(bank, stages)
    save_reservoir(
        output_dir,
        seeds,
        candidate_bank_path=candidate_bank_path,
        stage_trace_path=stage_trace_path,
        b0_labels=labels,
        provenance=provenance,
    )
    return seeds


__all__ = [
    "STAGE_PRIORITY",
    "build_candidate_reservoir",
    "load_reservoir",
    "load_stage_arrays",
    "prepare_reservoir",
    "save_reservoir",
]
