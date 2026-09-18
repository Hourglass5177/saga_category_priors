"""Fixed CPU choices for the DEV2 category-scale experiment; no evaluation inputs."""
from __future__ import annotations

import math
import numpy as np

from .prediction_contract import normalize_prediction

QUANTILES = ('q25', 'q50', 'q75')


def projection_iou(projected, mask, observed_pixels=None):
    """Compare only measured pixels; None reproduces the historical full-image score."""
    projected, mask = np.asarray(projected, bool), np.asarray(mask, bool)
    if projected.shape != mask.shape:
        raise ValueError('projection and SAM mask must share image coordinates')
    if observed_pixels is not None:
        observed = np.asarray(observed_pixels, bool)
        if observed.shape != mask.shape:
            raise ValueError('observation domain must share image coordinates')
        projected, mask = projected & observed, mask & observed
    union = np.count_nonzero(projected | mask)
    return float(np.count_nonzero(projected & mask) / union) if union else 0.


def resolve_prior(priors, category, mode='category'):
    if mode not in ('category', 'global'):
        raise ValueError('prior mode must be category or global')
    name, node = 'global', priors['global']
    if mode == 'category':
        child = priors.get('categories', {}).get(category, {})
        if child.get('active'):
            name, node = category, child
        else:
            parent = child.get('fallback', child.get('parent'))
            ancestor = priors.get('parents', {}).get(parent, {})
            if ancestor.get('active'):
                name, node = parent, ancestor
    geometry = node.get('shrunk', node.get('raw', {}))['geometry']['log_bbox_diag_m']
    return dict(hypothesis=category, mode=mode, source=name,
                category_specific=mode == 'category' and name == category,
                sizes_m={q: math.exp(float(v)) for q, v in geometry.items()})


def scale_slots(priors, classes, mean_cosines, mode):
    """Keep both semantic slots even if they resolve to identical global statistics."""
    if mean_cosines is None:
        return []
    values = np.asarray(mean_cosines, dtype=float)
    if values.shape != (len(classes),) or not np.isfinite(values).all():
        raise ValueError('class hypotheses require one finite score per class')
    top = sorted(range(len(classes)), key=lambda i: (-values[i], i))[:2]
    rows = []
    for slot, index in enumerate(top):
        prior = resolve_prior(priors, classes[index], mode)
        for q in QUANTILES:
            rows.append(dict(prior, slot=slot, quantile=q, diagonal_m=prior['sizes_m'][q],
                             branch=f'class{slot}-{q}'))
    return rows


def candidate_score(*, overlaps, semantic_score, sam_qualities, diagonal_m, prior):
    v = float(np.mean(overlaps)) if len(overlaps) else 0.
    s = float(semantic_score) if semantic_score is not None else 0.
    q = float(np.mean(np.clip(sam_qualities, 0, 1))) if len(sam_qualities) else 0.
    upper = prior['sizes_m']['q95']
    penalty = .10 * min(1., max(0., math.log(max(diagonal_m, upper) / upper) / math.log(2.)))
    return dict(score=.45 * v + .35 * s + .20 * q - penalty,
                projection_iou=v, semantic_score=s, sam_quality=q,
                diagonal_m=float(diagonal_m), size_penalty=penalty, penalty_prior=prior)


def select_candidate(rows):
    usable = [r for r in rows if r['member_count'] >= 3]
    if not usable:
        return next(r for r in rows if r['kind'] == 'original')
    priority = {'original': 0, 'legacy': 1, 'category': 2}
    return min(usable, key=lambda r: (-r['quality']['score'], priority[r['kind']], r['id']))


def merge_ranked(b0_payload, proposals, b0_quality, allowed_classes):
    """Whole-object same-class NMS followed by score-ordered, non-overlapping ownership."""
    b0 = np.asarray(b0_payload['point_labels'], dtype=np.int64)
    rows = []
    for key, meta in b0_payload['instances'].items():
        uid = f'B0:{key}'
        rows.append(dict(uid=uid, members=np.flatnonzero(b0 == int(key)),
                         **{'class': meta['class'], 'score': meta['score']},
                         selection_score=b0_quality[uid], baseline=True))
    rows.extend(dict(p, baseline=False) for p in proposals)
    rows = [r for r in rows if r['class'] in allowed_classes and len(r['members']) >= 3]
    rows.sort(key=lambda r: (-r['selection_score'], not r['baseline'], r['uid']))
    kept, suppressed = [], {}
    for row in rows:
        members = np.unique(row['members'])
        row = dict(row, members=members)
        for other in kept:
            if row['class'] != other['class']:
                continue
            intersection = np.intersect1d(members, other['members'], assume_unique=True).size
            if intersection / (len(members) + len(other['members']) - intersection) > .5:
                suppressed[row['uid']] = other['uid']
                break
        else:
            kept.append(row)
    labels = np.full(len(b0), -1, np.int64)
    metadata, assigned, raw_ids = {}, {}, {}
    for raw_id, row in enumerate(kept):
        members = row['members'][labels[row['members']] < 0]
        if len(members) < 3:
            continue
        labels[members] = raw_id
        raw_ids[row['uid']] = raw_id
        assigned[row['uid']] = members
        metadata[raw_id] = {'class': row['class'], 'score': row['score'],
                            'point_count': len(members), 'object_uid': row['uid']}
    contracted = normalize_prediction(labels, metadata)
    parent_indices = {p['uid']: i for i, p in enumerate(sorted(proposals, key=lambda p: p['uid']))}
    lineage = {str(contracted.export_id_by_raw[raw_ids[uid]]): [index]
               for uid, index in parent_indices.items() if uid in raw_ids}
    payload = dict(point_labels=contracted.point_labels.tolist(), instances=contracted.instances,
        prediction_contract=contracted.audit, repair_policy='ranked',
        candidate_export_contract_schema='saga-candidate-export-lineage-v2',
        candidate_export_lineage=lineage,
        candidate_export_ids={str(i): [int(k)] for k, ids in lineage.items() for i in ids},
        parent_candidate_index={str(i): uid for uid, i in parent_indices.items()},
        refined_export_ids=sorted(map(int, lineage)))
    return dict(payload=payload, assigned=assigned, suppressed=suppressed)
