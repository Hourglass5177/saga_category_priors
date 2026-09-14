"""CPU member rules and one-pass scene assembly for effective-repair experiments.

No controller, GT, graph, donor gate or model is imported. Geometry proposals
remain available even when their class or scene ownership prevents export.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .prediction_contract import normalize_prediction, normalize_score


def _ids(values: Any) -> np.ndarray:
    raw = np.asarray(list(values) if isinstance(values, (set, frozenset)) else values)
    if raw.size == 0:
        return np.empty(0, dtype=np.int64)
    if raw.ndim != 1 or raw.dtype.kind not in "iu" or np.any(raw < 0):
        raise ValueError("members must be a vector of nonnegative integer Gaussian IDs")
    return np.unique(raw.astype(np.int64))


def g0_members(observations: Sequence[Mapping[str, Any]], independent_pairs) -> np.ndarray:
    """At least two independent hard-or-alpha positives, minus every negative.

    Alpha + alpha is sufficient; the retired G1 hard-camera condition is absent.
    One observation per physical camera is required. Fewer than three IDs is empty.
    """
    camera_ids = [str(row["camera_uid"]) for row in observations]
    if len(set(camera_ids)) != len(camera_ids):
        raise ValueError("one adopted observation per physical camera is required")
    pairs = {tuple(sorted(map(str, pair))) for pair in independent_pairs}
    if any(len(pair) != 2 or pair[0] == pair[1] for pair in pairs):
        raise ValueError("independent pairs require two different camera IDs")
    positives = {
        camera: np.union1d(_ids(row.get("hard_ids", ())), _ids(row.get("alpha_ids", ())))
        for camera, row in zip(camera_ids, observations)
    }
    selected = np.empty(0, dtype=np.int64)
    for first, second in sorted(pairs):
        if first in positives and second in positives:
            selected = np.union1d(selected, np.intersect1d(positives[first], positives[second]))
    negatives = np.empty(0, dtype=np.int64)
    for row in observations:
        negatives = np.union1d(negatives, _ids(row.get("negative_ids", ())))
    selected = np.setdiff1d(selected, negatives)
    return selected if len(selected) >= 3 else np.empty(0, dtype=np.int64)


def g2_members(inside_mass: Any, visible_mass: Any) -> dict[str, np.ndarray]:
    """Pool all construction views, retaining the measured C4 thresholds."""
    inside_views = np.asarray(inside_mass)
    visible_views = np.asarray(visible_mass)
    if (inside_views.ndim != 2 or visible_views.shape != inside_views.shape
            or not np.isfinite(inside_views).all() or not np.isfinite(visible_views).all()
            or np.any(inside_views < 0) or np.any(visible_views < 0)):
        raise ValueError("inside/visible masses must be finite nonnegative matching V by N arrays")
    inside = inside_views.sum(axis=0)
    visible = visible_views.sum(axis=0)
    ratio = np.divide(inside, visible,
                      out=np.zeros_like(inside, dtype=np.result_type(inside.dtype, np.float32)),
                      where=visible > 0)
    members = np.flatnonzero((inside >= .5) & (ratio >= .5)).astype(np.int64)
    if len(members) < 3:
        members = np.empty(0, dtype=np.int64)
    return {"members": members, "inside": inside, "visible": visible, "ratio": ratio}


def mean32(cosines: Sequence[Any | None], classes32: Sequence[str]) -> dict[str, Any]:
    """Equal-weight mean over valid view vectors, with no agreement/score veto.

    Missing/invalid encodings are supplied as None; nonfinite model vectors are
    errors, not scientific unknowns. Scores are an uncalibrated affine cosine.
    """
    classes = tuple(classes32)
    if len(classes) != 32 or len(set(classes)) != 32 or any(not isinstance(c, str) or not c for c in classes):
        raise ValueError("32 unique class names are required")
    valid, per_view = [], []
    for vector in cosines:
        if vector is None:
            per_view.append(None)
            continue
        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (32,) or not np.isfinite(values).all():
            raise ValueError("each valid observation must contain 32 finite cosine values")
        valid.append(values)
        winners = np.flatnonzero(values == values.max())
        per_view.append(classes[int(winners[0])] if len(winners) == 1 else None)
    result = {"class": None, "score": None, "mean_cosines": None,
              "valid_view_count": len(valid), "total_view_count": len(cosines),
              "per_view_top1": per_view,
              "disagreement": len({name for name in per_view if name is not None}) > 1,
              "margin": None, "status": "unknown", "calibrated_probability": False}
    if not valid:
        return result
    means = np.mean(valid, axis=0)
    ordered = np.sort(means)
    result.update(mean_cosines=means.tolist(), margin=float(ordered[-1] - ordered[-2]))
    winners = np.flatnonzero(means == means.max())
    if len(winners) == 1:
        index = int(winners[0])
        result.update({"class": classes[index], "score": normalize_score((1. + float(means[index])) / 2.),
                       "status": "complete"})
    return result


def _transfer_counts(b0: np.ndarray, members: np.ndarray) -> dict[str, int]:
    labels, counts = np.unique(b0[members], return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(labels, counts)}


def merge_scene(
    b0_payload: Mapping[str, Any], proposals: Sequence[Mapping[str, Any]], saga20: Sequence[str],
    *, policy: str = "A",
    classify_actual: Callable[[str, np.ndarray], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble a scene once from immutable B0 and complete candidate proposals.

    Proposals have uid, members, class, score and optional member-aligned ratio
    (required for C). Exact same-members/same-class proposals alias the smallest
    UID; that representative supplies its score/ratio. A/B retain B0 at any
    contested point. C awards a unique maximum ratio; exact ties retain B0.

    Raw unknown/non-SAGA20 proposals do not claim. For A/B, changed assigned
    membership calls classify_actual(uid, members); unchanged uses raw class.
    A rejected actual class restores only its own writes, without redistribution.
    C calls the callback for diagnostics only and retains its SAM class/score.
    B0 residual metadata is preserved. Returned proposal arrays are not JSON lists;
    payload is JSON-ready and includes the existing evaluator lineage fields.
    """
    if policy not in {"A", "B", "C"}:
        raise ValueError("policy must be A, B or C")
    allowed = set(saga20)
    if not allowed:
        raise ValueError("the frozen export class set is required")
    b0 = np.asarray(b0_payload["point_labels"])
    if b0.ndim != 1 or b0.dtype.kind not in "iu" or np.any(b0 < -1):
        raise ValueError("B0 needs one integer label per Gaussian")
    b0 = b0.astype(np.int64, copy=True)
    original = {int(key): dict(value) for key, value in b0_payload["instances"].items()}
    if set(original) != set(map(int, np.unique(b0[b0 >= 0]))):
        raise ValueError("B0 declared instances and Gaussian labels must agree")
    parsed = {}
    for proposal in proposals:
        uid = proposal["uid"]
        if not isinstance(uid, str) or not uid or uid in parsed:
            raise ValueError("each input proposal needs a unique nonempty UID")
        members = _ids(proposal["members"])
        raw_members = np.asarray(proposal["members"])
        if len(raw_members) != len(members):
            raise ValueError("proposal member IDs must be unique")
        if len(members) and members[-1] >= len(b0):
            raise ValueError("proposal contains an ID outside the full Gaussian domain")
        label = proposal.get("class")
        score = proposal.get("score")
        if label is not None:
            if not isinstance(label, str) or not label:
                raise ValueError("class must be a nonempty name or None")
            score = normalize_score(score)
        ratio = proposal.get("ratio")
        if policy == "C" and ratio is None:
            raise ValueError("policy C requires a ratio aligned with proposal members")
        if ratio is not None:
            ratio = np.asarray(ratio, dtype=np.float64)
            if ratio.shape != raw_members.shape or not np.isfinite(ratio).all() or np.any(ratio < 0):
                raise ValueError("proposal ratio must be finite, nonnegative and member-aligned")
            ratio = ratio[np.argsort(raw_members)]
        parsed[uid] = {"uid": uid, "members": members, "class": label, "score": score, "ratio": ratio}
    groups = {}
    for uid in sorted(parsed):
        row = parsed[uid]
        groups.setdefault((row["class"], row["members"].tobytes()), []).append(uid)
    aliases = {group[0]: group for group in groups.values()}
    alias_to_rep = {uid: representative for representative, group in aliases.items() for uid in group}
    representatives = sorted(aliases)
    rows = [parsed[uid] for uid in representatives]
    next_raw = max(original, default=-1) + 1
    raw_ids = {uid: next_raw + index for index, uid in enumerate(representatives)}
    eligible = [index for index, row in enumerate(rows) if row["class"] in allowed and len(row["members"]) >= 3]
    labels = b0.copy()
    assigned = {uid: np.empty(0, dtype=np.int64) for uid in representatives}
    contested = tied = 0
    if eligible:
        points = np.concatenate([rows[index]["members"] for index in eligible])
        owners = np.concatenate([np.full(len(rows[index]["members"]), index, dtype=np.int32) for index in eligible])
        order = np.argsort(points, kind="stable")
        points, owners = points[order], owners[order]
        unique, starts, counts = np.unique(points, return_index=True, return_counts=True)
        contested = int(np.count_nonzero(counts > 1))
        winner = owners[starts].copy()
        award = counts == 1
        if policy == "C":
            ratios = np.concatenate([rows[index]["ratio"] for index in eligible])[order]
            maximum = np.maximum.reduceat(ratios, starts)
            is_maximum = ratios == np.repeat(maximum, counts)
            max_count = np.add.reduceat(is_maximum.astype(np.int32), starts)
            award = max_count == 1
            tied = int(np.count_nonzero(~award))
            max_owner = np.where(is_maximum, owners, -1)
            winner = np.maximum.reduceat(max_owner, starts)
        for index in eligible:
            uid = rows[index]["uid"]
            members = unique[award & (winner == index)]
            assigned[uid] = members
            labels[members] = raw_ids[uid]
    metadata = {key: dict(value) for key, value in original.items()}
    records, exported = {}, {}
    for row in rows:
        uid, raw = row["uid"], row["members"]
        actual = assigned[uid]
        reason = "exported"
        semantic = {"class": row["class"], "score": row["score"], "source": "raw_members"}
        diagnostic = None
        if row["class"] not in allowed:
            reason = "raw_non_saga20_or_unknown"
        elif len(raw) < 3:
            reason = "raw_fewer_than_three_members"
        elif len(actual) < 3:
            reason = "assigned_fewer_than_three_members"
        elif policy == "C":
            if classify_actual is not None:
                diagnostic = dict(classify_actual(uid, actual.copy()))
        elif not np.array_equal(raw, actual):
            if classify_actual is None:
                raise ValueError("changed A/B membership requires actual-member classification")
            semantic = dict(classify_actual(uid, actual.copy()))
            if semantic.get("class") not in allowed:
                reason = "actual_non_saga20_or_unknown"
            else:
                semantic["score"] = normalize_score(semantic.get("score"))
        kept = actual.copy() if reason == "exported" else np.empty(0, dtype=np.int64)
        if reason != "exported":
            labels[actual] = b0[actual]
        else:
            metadata[raw_ids[uid]] = {"class": semantic["class"], "score": semantic["score"],
                "point_count": len(kept), "object_uid": uid, "aliases": aliases[uid],
                "score_source": "SAM_region_mean32" if policy == "C" else "actual_member_mean32"}
        exported[uid] = kept
        records[uid] = {"uid": uid, "representative_uid": uid, "aliases": aliases[uid],
            "raw_members": raw.copy(), "assigned_members": actual.copy(), "exported_members": kept,
            "raw_class": row["class"], "raw_score": row["score"], "actual_semantics": semantic,
            "actual_semantics_diagnostic": diagnostic, "status": reason,
            "final_class": semantic["class"] if reason == "exported" else None,
            "final_score": semantic["score"] if reason == "exported" else None,
            "raw_source_transfers": _transfer_counts(b0, raw),
            "assigned_source_transfers": _transfer_counts(b0, actual),
            "exported_source_transfers": _transfer_counts(b0, kept)}
    counts_by_raw = dict(zip(*np.unique(labels[labels >= 0], return_counts=True)))
    for raw_id, value in metadata.items():
        value["point_count"] = int(counts_by_raw.get(raw_id, 0))
    contracted = normalize_prediction(labels, metadata)
    export_by_uid = {
        uid: contracted.export_id_by_raw.get(raw_ids[alias_to_rep[uid]]) for uid in sorted(parsed)
    }
    for uid in sorted(parsed):
        representative = alias_to_rep[uid]
        if uid != representative:
            records[uid] = {**records[representative], "uid": uid,
                           "raw_score": parsed[uid]["score"], "alias_of": representative}
        else:
            records[uid]["alias_of"] = None
        records[uid]["export_id"] = export_by_uid[uid]
    parent_indices = {uid: index for index, uid in enumerate(sorted(parsed))}
    lineage = {
        str(export_by_uid[uid]): [parent_indices[alias] for alias in aliases[uid]]
        for uid in representatives if export_by_uid[uid] is not None
    }
    inverse = {str(parent): [int(export)] for export, parents in lineage.items() for parent in parents}
    payload = {"point_labels": contracted.point_labels.tolist(), "instances": contracted.instances,
        "prediction_contract": contracted.audit, "repair_policy": policy,
        "candidate_export_contract_schema": "saga-candidate-export-lineage-v2",
        "candidate_export_ids": inverse, "candidate_export_lineage": lineage,
        "refined_export_ids": sorted(map(int, lineage)),
        "parent_candidate_index": {str(index): uid for uid, index in parent_indices.items()}}
    return {"payload": payload, "proposals": records, "alias_to_representative": alias_to_rep,
        "export_id_by_uid": export_by_uid,
        "summary": {"input_proposal_count": len(parsed), "representative_count": len(representatives),
            "alias_count": len(parsed) - len(representatives), "raw_eligible_count": len(eligible),
            "exported_representative_count": len(lineage), "contested_point_count": contested,
            "ratio_tie_point_count": tied,
            "changed_gaussian_count": int(np.count_nonzero(labels != b0)),
            "restored_assigned_point_count": sum(len(assigned[uid]) - len(exported[uid]) for uid in representatives)}}


__all__ = ["g0_members", "g2_members", "mean32", "merge_scene"]


def pixel_metrics(prediction, foreground, uncertain):
    arrays = [np.asarray(a) for a in (prediction, foreground, uncertain)]
    if any(a.dtype != bool or a.ndim != 2 or a.shape != arrays[0].shape for a in arrays):
        raise ValueError("full RGB shared boolean axes required")
    prediction, foreground, uncertain = arrays
    if (foreground & uncertain).any():
        raise ValueError("foreground and uncertain must be disjoint")
    valid = ~uncertain
    tp = int((prediction & foreground & valid).sum())
    fp = int((prediction & ~foreground & valid).sum())
    fn = int((~prediction & foreground & valid).sum())
    tn = int((~prediction & ~foreground & valid).sum())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
            "foreground_denominator": tp + fn, "prediction_denominator": tp + fp,
            "union_denominator": tp + fp + fn, "valid_pixels": int(valid.sum()),
            "domain": "full_original_RGB_excluding_only_human_uncertain"}
