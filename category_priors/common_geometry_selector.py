"""Candidate-independent, category-independent comparison of saved observations.

No annotations, model calls, size priors or semantic scores are accepted here.
"""
from __future__ import annotations

import hashlib
import numpy as np

from .multiview_repair import independent_support, mask_families

VERSION = 'common-geometry-v1'
TOL = 1e-12


def content_key(row):
    h = hashlib.sha256()
    for name in ('positive_ids', 'negative_ids', 'known_ids'):
        h.update(np.asarray(np.unique(row[name]), dtype='<i8').tobytes())
        h.update(b'|')
    return row['camera'] + ':' + h.hexdigest()


def reference_family(observations, pairs):
    """Select once from observations, never by overlap with a scored candidate.

    Content keys also prevent filenames and duplicated crops breaking ties.
    Pair connectivity is restricted to independent cameras. A common surface is
    necessary; overlap of boxes or semantic labels is never an identity edge.
    """
    unique = {}
    for row in observations:
        key = content_key(row)
        unique.setdefault(key, dict(row, path=key, observation_path=row['path']))
    rows = [unique[k] for k in sorted(unique)]
    pairset = {frozenset(p) for p in pairs}
    families = []
    for paths in mask_families(rows):
        group = [unique[p] for p in paths]
        # Remove disconnected views rather than letting a nearby duplicate view
        # turn a one-view observation into independent evidence.
        scores = []
        connected = set()
        for i, a in enumerate(group):
            for b in group[i+1:]:
                if frozenset((a['camera'], b['camera'])) not in pairset:
                    continue
                known = np.intersect1d(a['known_ids'], b['known_ids'])
                ap = np.intersect1d(a['positive_ids'], known)
                bp = np.intersect1d(b['positive_ids'], known)
                intersection = len(np.intersect1d(ap, bp))
                if intersection:
                    scores.append(intersection / len(np.union1d(ap, bp)))
                    connected.update((a['camera'], b['camera']))
        refs = [r for r in group if r['camera'] in connected]
        if not refs:
            continue
        core = independent_support(refs, pairs)
        negatives = independent_support(refs, pairs, 'negative_ids')
        families.append((float(np.mean(scores)), len(np.setdiff1d(core, negatives)),
                         tuple(r['path'] for r in refs), refs))
    if not families:
        return [], dict(status='no_independent_identity', family_count=0)
    best = min(families, key=lambda r: (-r[0], -r[1], r[2]))
    selected = [dict(r, path=r['observation_path']) for r in best[3]]
    return selected, dict(status='complete', family_count=len(families),
                          agreement=best[0], independently_supported_members=best[1],
                          content_keys=list(best[2]))


def geometry_quality(per_view):
    values = {r['camera_uid']: r['iou'] for r in per_view if r['iou'] is not None}
    subsets = {}
    if values:
        subsets['all'] = float(np.mean(list(values.values())))
        if len(values) > 1:
            for camera in sorted(values):
                subsets['without:' + camera] = float(np.mean([v for k, v in values.items() if k != camera]))
    return dict(score=min(subsets.values(), default=0.),
                projection_iou=subsets.get('all', 0.), subsets=subsets,
                valid_view_count=len(values), per_view=per_view,
                semantic_score=None, size_penalty=0., sam_weight=0.,
                selector=VERSION, status='complete' if values else 'no_evidence')


def pixel_evidence(projected, foreground, background):
    projected, foreground, background = map(lambda a: np.asarray(a, bool),
                                             (projected, foreground, background))
    if np.any(foreground & background):
        raise ValueError('foreground and background evidence overlap')
    tp = int((projected & foreground).sum())
    fp = int((projected & background).sum())
    fn = int((~projected & foreground).sum())
    # Empty prediction against observed foreground is a real zero, not missing.
    denominator = tp + fp + fn
    return dict(tp=tp, fp=fp, fn=fn, iou=tp/denominator if denominator else None,
                known_pixels=int((foreground | background).sum()))


def semantic_state(semantics, vocabulary):
    label = semantics.get('class')
    if label is None or label == 'unknown':
        return 'unknown', 'unknown'
    return label, 'recognized' if label in vocabulary else 'out_of_eval_vocabulary'


def stable_improvement(a, b):
    av, bv = a['quality'].get('subsets', {}), b['quality'].get('subsets', {})
    # Missing evidence cannot win by changing the comparison denominator.
    if not bv or set(av) != set(bv):
        return False
    delta = [av[k] - bv[k] for k in bv]
    return min(delta) >= -TOL and max(delta) > TOL


def supported_extension(a, b, members):
    ma, mb = members[a['id']], members[b['id']]
    if len(ma) <= len(mb):
        return False
    ap = {r['camera_uid']: r for r in a['quality']['per_view']}
    bp = {r['camera_uid']: r for r in b['quality']['per_view']}
    if set(ap) != set(bp) or not all(ap[v]['fp'] <= bp[v]['fp'] and ap[v]['fn'] <= bp[v]['fn'] for v in bp):
        return False
    added, removed = np.setdiff1d(ma, mb), np.setdiff1d(mb, ma)
    if len(removed) or not len(added):
        return False
    core = a.get('core_ids', [])
    if not np.isin(added, core).all() or np.isin(added, a.get('negative_ids', [])).any():
        return False
    return True


def select_geometry(rows, members, baseline_id=None):
    if rows and rows[0]['quality'].get('selector') == 'regional-evidence-v2':
        from .regional_evidence import select_regional
        return select_regional(rows, members, baseline_id)
    usable = [r for r in rows if len(members[r['id']]) >= 3]
    if not usable:
        return next(r for r in rows if r['id'] == baseline_id or r['id'] == 'original'), []
    def key(r):
        priority = 0 if r['id'] == baseline_id else 1 if r['id'] == 'original' else 2 if r.get('kind') == 'legacy' else 3
        # Content, not candidate names or generating branch, resolves final ties.
        return (-r['quality']['score'], -r['quality']['projection_iou'], priority,
                np.asarray(members[r['id']], dtype='<i8').tobytes())
    base = next((r for r in usable if r['id'] == baseline_id), None)
    if base is None:
        common = [r for r in usable if r.get('kind') in ('original', 'legacy', 'global_saved', 'common')]
        base = min(common or usable, key=key)
    undominated = [b for b in usable if not any(a is not b and supported_extension(a, b, members) for a in usable)]
    eligible = [r for r in undominated if r is base or stable_improvement(r, base) or supported_extension(r, base, members)]
    winner = min(eligible, key=key) if eligible else base
    ranked = sorted(usable, key=key)
    trace = []
    for row in ranked:
        added = np.setdiff1d(members[row['id']], members[base['id']])
        removed = np.setdiff1d(members[base['id']], members[row['id']])
        trace.append(dict(candidate_id=row['id'], baseline_id=base['id'], selected=row is winner,
            stable=stable_improvement(row, base), supported_extension=supported_extension(row, base, members),
            added_count=len(added), removed_count=len(removed),
            added_independent_count=int(np.isin(added, row.get('core_ids', [])).sum()),
            added_negative_count=int(np.isin(added, row.get('negative_ids', [])).sum()),
            removed_core_count=int(np.isin(removed, base.get('core_ids', [])).sum())))
    return winner, trace
