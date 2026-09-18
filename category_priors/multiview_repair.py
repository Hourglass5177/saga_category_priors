"""Small, inference-only geometry operations for the paired DEV2 repair study."""
from __future__ import annotations

import itertools
import math
import numpy as np

from .object_verification.model_adapter import CropTransform


def metric_crop(image_shape, foreground_xy, focal, diagonal_m, depth_m):
    """Locate the window from the object, not from a prompt or the old mixed box."""
    xy = np.asarray(foreground_xy, float).reshape(-1, 2)
    if not len(xy) or not np.isfinite(xy).all() or min(focal, diagonal_m, depth_m) <= 0:
        raise ValueError('finite foreground and positive metric projection required')
    lo, hi = np.quantile(xy, [.05, .95], axis=0)
    requested = 1.5 * focal * diagonal_m / depth_m
    side = min(max(math.ceil(requested), 64), max(image_shape))
    center = (lo + hi) / 2
    # Translate (never enlarge) to include the robust foreground where possible.
    for axis in range(2):
        a, b = hi[axis] - side / 2 + 1, lo[axis] + side / 2 - 1
        if a <= b:
            center[axis] = np.clip(center[axis], a, b)
    center = np.floor(center + .5).astype(int)
    crop = CropTransform(tuple(image_shape), int(center[0])-side//2,
                         int(center[1])-side//2, side, side, requested)
    return crop, dict(center_image_xy=center.tolist(), foreground_lo=lo.tolist(),
                     foreground_hi=hi.tolist(), foreground_fits=bool(np.all(hi-lo+2 <= side)))


def interior_points(foreground, reliable, count=1):
    """Positive prompts must be observed object pixels; spacing never creates positives."""
    from scipy.ndimage import distance_transform_edt
    foreground, reliable = np.asarray(foreground, bool), np.asarray(reliable, bool)
    valid = foreground & reliable
    if not valid.any():
        return []
    distance = distance_transform_edt(np.pad(foreground, 1))[1:-1, 1:-1]
    yy, xx = np.where(valid)
    interior = distance[yy, xx]
    selected, available = [], np.ones(len(xx), bool)
    spacing = np.full(len(xx), np.inf)
    for _ in range(min(count, len(xx))):
        score = interior if not selected else np.minimum(interior, spacing)
        index = int(np.argmax(np.where(available, score, -1.)))
        selected.append([int(xx[index]), int(yy[index])])
        available[index] = False
        spacing = np.minimum(spacing, np.hypot(xx-xx[index], yy-yy[index]))
    return selected


def image_edge(shape, width=2):
    edge = np.zeros(shape, bool)
    edge[:width] = True; edge[-width:] = True
    edge[:, :width] = True; edge[:, -width:] = True
    return edge


def boundary_reasons(mask, observed):
    from scipy.ndimage import binary_erosion
    mask, observed = np.asarray(mask, bool), np.asarray(observed, bool)
    photo = image_edge(mask.shape)
    crop_edge = observed & ~binary_erosion(observed, iterations=2, border_value=0) & ~photo
    return dict(image_truncated=bool((mask & photo).any()),
                crop_truncated=bool((mask & crop_edge).any()))


def known_domain(mask, alternatives, observed):
    """A different SAM granularity is not evidence of background."""
    union = np.any(np.asarray(alternatives, bool), axis=0)
    return np.asarray(observed, bool) & (np.asarray(mask, bool) | ~union)


def depth_visibility(xy, depth, front_ids, front_depth, reliable, target_ids, members):
    """Center-depth occlusion is only a hypothesis, never positive object identity."""
    xy, depth = np.asarray(xy), np.asarray(depth)
    finite = np.isfinite(xy).all(axis=1) & np.isfinite(depth) & (depth > 0)
    ij = np.zeros((len(xy), 2), np.int64)
    ij[finite] = np.floor(xy[finite]+.5).astype(np.int64)
    h, w = front_ids.shape
    inside = finite & (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
    known, occluded, own = np.zeros((3, len(xy)), bool)
    blocker = np.full(len(xy), -1, np.int64)
    k = np.flatnonzero(inside)
    if len(k):
        x, y = ij[k].T
        known[k] = reliable[y, x]
        blocker[k] = front_ids[y, x]
        occluded[k] = known[k] & (front_depth[y, x] > 0) & (depth[k] > 1.05 * front_depth[y, x])
        own[k] = occluded[k] & np.isin(blocker[k], members)
    return dict(outside_image=~inside, suspected_occluded=occluded,
                suspected_self_occluded=own, unreliable=inside & ~known,
                occluder_ids=blocker, gaussian_ids=np.asarray(target_ids, np.int64))


def mask_families(observations):
    """Seed every raw mask. A positive shared surface is required to join views."""
    seen, result = set(), []
    for seed in observations:
        group = [seed]
        used = {seed['camera']}
        while True:
            options = []
            for other in observations:
                if other['camera'] in used:
                    continue
                best = (0., 0)
                for current in group:
                    common = np.intersect1d(current['known_ids'], other['known_ids'])
                    a = np.intersect1d(current['positive_ids'], common)
                    b = np.intersect1d(other['positive_ids'], common)
                    intersection = len(np.intersect1d(a, b))
                    if intersection:
                        best = max(best, (intersection / max(1, len(np.union1d(a, b))), intersection))
                if best[0] > 0:
                    options.append((-best[0], -best[1], other['path'], other))
            if not options:
                break
            other = min(options, key=lambda x: x[:3])[3]
            group.append(other); used.add(other['camera'])
        key = tuple(sorted(o['path'] for o in group))
        if key not in seen:
            seen.add(key); result.append(list(key))
    return result


def independent_support(rows, pairs, key='positive_ids'):
    by_camera = {r['camera']: np.asarray(r[key], np.int64) for r in rows}
    pieces = [np.intersect1d(by_camera[a], by_camera[b]) for a, b in pairs
              if a in by_camera and b in by_camera]
    return np.unique(np.concatenate(pieces)) if pieces else np.empty(0, np.int64)


def member_components(rows, point_count):
    """Fixed affected object sets, including before and proposed memberships."""
    nodes = sorted({uid for uid, _ in rows})
    index = {uid: i for i, uid in enumerate(nodes)}
    parent = list(range(len(nodes)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    touched = np.full(point_count, -1, np.int32)
    for uid, members in rows:
        members = np.asarray(members, np.int64)
        i = index[uid]
        for other in np.unique(touched[members]):
            if other >= 0:
                a, b = find(i), find(int(other))
                if a != b:
                    parent[b] = a
        touched[members] = i
    groups = {}
    for uid in nodes:
        groups.setdefault(find(index[uid]), []).append(uid)
    return list(groups.values())


def evidence_score(overlaps, semantic_score):
    values = [float(v) for v in overlaps if v is not None]
    v = float(np.mean(values)) if values else 0.
    t = float(semantic_score) if semantic_score is not None else 0.
    return dict(score=(.45*v+.35*t)/.8, projection_iou=v, semantic_score=t,
                valid_view_count=len(values), size_penalty=0., sam_weight=0.)


def select_evidence(rows, members):
    usable = [r for r in rows if r['member_count'] >= 3]
    if not usable:
        return next(r for r in rows if r['id'] == 'original')
    priority = {'original': 0, 'legacy': 1, 'global_saved': 2, 'common': 3, 'pre_feedback': 4, 'new': 5}
    protected = [r for r in usable if r['kind'] != 'new']
    base = min(protected or usable, key=lambda r: (-r['quality']['score'], priority[r['kind']], r['id']))
    # Single-view additions stay eligible. Independence limits protected cores,
    # not the candidate bank or source-independent ranking.
    permitted = usable
    best = min(permitted or [base], key=lambda r: (-r['quality']['score'], priority[r['kind']], r['id']))
    # Exact-score nesting: expand only through an independently observed core.
    for row in sorted(permitted, key=lambda r: (-r['member_count'], r['id'])):
        if abs(row['quality']['score']-best['quality']['score']) > 1e-12 or row['semantics']['class'] != best['semantics']['class']:
            continue
        added = np.setdiff1d(members[row['id']], members[best['id']])
        if len(added) and not len(np.setdiff1d(members[best['id']], members[row['id']])):
            if np.isin(added, row['core_ids']).all() and not np.isin(added, row['negative_ids']).any():
                best = row
    return best


def select_view_pair(original, candidates, compatible):
    """Geometry only; no class, SAM query, or evaluation label enters selection."""
    by_id = {r['camera']: r for r in candidates}
    def value(ids):
        rows = [by_id[c] for c in ids if c in by_id]
        f = [set(r['revealed']) for r in rows]
        twice = len(f[0] & f[1]) if len(f) == 2 else 0
        union = len(set().union(*f)) if f else 0
        return twice, union, sum(r['visible_count'] for r in rows), min((r['angle'] for r in rows), default=0.)
    eligible = [r for r in candidates if r.get('eligible', True)]
    options = [(a['camera'], b['camera']) for a, b in itertools.combinations(eligible, 2)
               if compatible(a['camera'], b['camera'])] if len(original) == 2 else []
    if len(original) == 1:
        options = [(r['camera'],) for r in eligible]
    if not options or not any(r['revealed'] for r in candidates):
        return list(original), False
    best = min(options, key=lambda ids: (tuple(-x for x in value(ids)), ids))
    return (list(best), True) if value(best) > value(original) else (list(original), False)
