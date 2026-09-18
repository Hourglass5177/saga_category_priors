"""Region evidence and conservative selection. No classes, annotations or models."""
from __future__ import annotations

import hashlib
import numpy as np

from .common_geometry_selector import TOL, geometry_quality, pixel_evidence
from .multiview_repair import independent_support

VERSION = 'regional-evidence-v2'


def mask_key(positive, known):
    return hashlib.sha256(np.packbits(np.stack([positive, known])).tobytes()).hexdigest()


def consensus(hypotheses):
    """Absence outside an observation is not a negative vote. Duplicates cannot vote."""
    shape = hypotheses[0]['known'].shape
    pos, neg = np.zeros(shape, bool), np.zeros(shape, bool)
    for h in hypotheses:
        pos |= h['known'] & h['mask']
        neg |= h['known'] & ~h['mask']
    return pos & ~neg, neg & ~pos, pos & neg


def score_projection(project, refs, *, known_domain=False):
    per_view, alternatives = [], {}
    for ref in refs:
        prediction = project(ref['camera'])
        per_view.append(dict(camera_uid=ref['camera'],
            **pixel_evidence(prediction, ref['foreground'], ref['background'])))
        # An unresolved interpretation may describe either a part or a whole.
        # Once that disagreement is declared unknown, it cannot become a hard
        # FP/FN again in the pairwise comparison. Preserve crop-specific missing
        # observations, but use the SAME qualified foreground/background.
        alternatives[ref['camera']] = {h['key']: pixel_evidence(
            prediction,
            h['known'] & (ref['foreground'] if known_domain else h['mask']),
            h['known'] & (ref['background'] if known_domain else ~h['mask']))
            for h in ref['hypotheses']}
    q = geometry_quality(per_view)
    # Ranking is worst-case over explanations, then equal-camera mean.
    lows = [min(h['iou'] for h in hs.values() if h['iou'] is not None)
            for hs in alternatives.values() if any(h['iou'] is not None for h in hs.values())]
    q.update(selector=VERSION, explanations=alternatives,
        score=min(lows, default=0.), projection_iou=float(np.mean(lows)) if lows else 0.,
        status='unresolved' if lows else 'missing_observation')
    if known_domain:
        q['evidence_revision']='known-domain-v1'
    return q


def symmetric_member_evidence(positive, negative):
    """One splat may cover both sides of a boundary; neither label wins by order."""
    disputed=np.intersect1d(positive,negative)
    return np.setdiff1d(positive,disputed),np.setdiff1d(negative,disputed),disputed


def paired_difference(a, b):
    av, bv = a['quality']['explanations'], b['quality']['explanations']
    if set(av) != set(bv):
        return None
    differences = {}
    for camera in sorted(av):
        if set(av[camera]) != set(bv[camera]):
            return None
        values = []
        for key in av[camera]:
            x, y = av[camera][key]['iou'], bv[camera][key]['iou']
            if x is None or y is None:
                # A zero-information explanation cannot certify an improvement.
                values.append(0.)
            else:
                values.append(x-y)
        if values:
            differences[camera] = min(values)
    return differences


def delta_evidence(a, b, members):
    added = np.setdiff1d(members[a['id']], members[b['id']])
    removed = np.setdiff1d(members[b['id']], members[a['id']])
    positive = np.union1d(a.get('core_ids', []), b.get('core_ids', []))
    negative = np.asarray(a.get('negative_ids', []), np.int64)
    return dict(added_count=len(added), removed_count=len(removed),
        supported_additions=int(np.isin(added, positive).sum()),
        supported_removals=int(np.isin(removed, negative).sum()),
        added_background=int(np.isin(added, negative).sum()),
        removed_foreground=int(np.isin(removed, positive).sum()),
        unknown_additions=int((~np.isin(added,np.union1d(positive,negative))).sum()),
        unknown_removals=int((~np.isin(removed,np.union1d(positive,negative))).sum()))


def stable(a, b, members):
    d = paired_difference(a,b)
    e = delta_evidence(a,b,members)
    pairs = a['quality'].get('independent_pairs', [])
    if d is None or not any(x in d and y in d for x,y in pairs):
        return False, e, d
    values = list(d.values())
    subsets = [np.mean(values)] + [np.mean(values[:i]+values[i+1:]) for i in range(len(values))]
    safe = not (e['added_background'] or e['removed_foreground'])
    supported = e['supported_additions'] + e['supported_removals'] > 0
    nonworse = min(values) >= -TOL and min(subsets) >= -TOL
    return bool(safe and supported and nonworse), e, d


def select_regional(rows, members, baseline_id=None):
    usable = [r for r in rows if len(members[r['id']]) >= 3]
    if not usable:
        return next(r for r in rows if r['id'] == (baseline_id or 'original')), []
    base = next((r for r in usable if r['id']==baseline_id), None)
    if base is None:
        base = next((r for r in usable if r['id']=='original'), None)
    if base is None:
        base = min(usable, key=lambda r: np.asarray(members[r['id']],dtype='<i8').tobytes())
    eligible, trace = [base], []
    for r in usable:
        valid, evidence, delta = stable(r,base,members)
        if valid:
            eligible.append(r)
        trace.append(dict(candidate_id=r['id'],baseline_id=base['id'],stable=valid,
            delta_by_view=delta,**evidence))
    # Stable extensions have certified positive additions; exact-score ties must
    # not preserve the partial baseline merely because its ID is older.
    def rank(r):
        e=delta_evidence(r,base,members)
        return (-r['quality']['score'],-r['quality']['projection_iou'],
                -(e['supported_additions']+e['supported_removals']),
                r is not base,np.asarray(members[r['id']],dtype='<i8').tobytes())
    winner=min(eligible,key=rank)
    for t in trace:
        t['selected']=t['candidate_id']==winner['id']
        evidence=delta_evidence(winner,base,members)
        t['status']=('partially_resolved' if evidence['unknown_additions'] or evidence['unknown_removals']
                     else 'resolved') if winner is not base else 'unresolved'
    return winner, sorted(trace,key=lambda t: rank(next(r for r in usable if r['id']==t['candidate_id'])))


def signed_reciprocity(views, pairs):
    """Each entry is two complete three-mask responses on one shared known domain."""
    signs={}
    for v in views:
        a,b=v['a_masks'],v['b_masks']; ay,ax=v['a_xy'][1],v['a_xy'][0];by,bx=v['b_xy'][1],v['b_xy'][0]
        if len(a)!=3 or len(b)!=3 or not v['known'][ay,ax] or not v['known'][by,bx]:
            continue
        if not np.all(a[:,ay,ax]) or not np.all(b[:,by,bx]):
            continue
        ab,ba=a[:,by,bx],b[:,ay,ax]
        signs[v['camera']]='connect' if ab.all() and ba.all() else 'separate' if not ab.any() and not ba.any() else 'unresolved'
    certified={s for s in ('connect','separate') if any(signs.get(a)==signs.get(b)==s for a,b in pairs)}
    return (next(iter(certified)) if len(certified)==1 and not any(s not in certified for s in signs.values()) else 'unresolved'),signs


def member_regions(members, id_images):
    """Exact membership signatures, connected only by observed 4-neighbour pixels."""
    arrays=[np.unique(members[k]) for k in sorted(members,key=lambda k: np.asarray(members[k],dtype='<i8').tobytes())]
    union=np.unique(np.concatenate(arrays)) if arrays else np.array([],np.int64)
    if not len(union):return []
    signatures=np.packbits(np.stack([np.isin(union,m) for m in arrays],axis=1),axis=1)
    _,labels=np.unique(signatures,axis=0,return_inverse=True)
    parent=np.arange(len(union))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for im in id_images:
        im=np.asarray(im)
        edges=np.concatenate([np.stack([im[:,:-1].ravel(),im[:,1:].ravel()],1),
                              np.stack([im[:-1].ravel(),im[1:].ravel()],1)])
        edges=np.unique(edges,axis=0)
        ix=np.searchsorted(union,edges);valid=(ix<len(union)).all(1)
        edges,ix=edges[valid],ix[valid]
        valid=(union[ix]==edges).all(1)&(labels[ix[:,0]]==labels[ix[:,1]])
        for a,b in ix[valid]:
            ra,rb=root(a),root(b)
            if ra!=rb:parent[rb]=ra
    groups={}
    for i,m in enumerate(union):groups.setdefault(root(i),[]).append(int(m))
    return [np.array(g,np.int64) for g in groups.values()]


def repair_regions(baseline, regions, positive, negative):
    """All decisions read the original baseline, not preceding region changes."""
    positive=np.setdiff1d(positive,negative)
    add,remove,trace=[],[],[]
    for region in regions:
        if np.isin(region,positive).all():
            ids=np.setdiff1d(region,baseline);add.extend(ids);status='supported_addition'
        elif np.isin(region,negative).all():
            ids=np.intersect1d(region,baseline);remove.extend(ids);status='supported_removal'
        else:
            ids=np.array([],int);status='unresolved'
        trace.append(dict(members=region.tolist(),status=status,changed_count=len(ids)))
    result=np.setdiff1d(np.union1d(baseline,add),remove).astype(np.int64)
    return result,trace
