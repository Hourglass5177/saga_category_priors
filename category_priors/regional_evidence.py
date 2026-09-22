"""Region evidence and conservative selection. No classes, annotations or models."""
from __future__ import annotations

import hashlib
import numpy as np

from .common_geometry_selector import TOL, geometry_quality, pixel_evidence
from .multiview_repair import independent_support

VERSION = 'regional-evidence-v2'
OBJECT_VERSION = 'object-explanation-v1'


def semantic_vector(observation):
    """A full regional signature, never a winning class probability."""
    sem = observation.get('semantics') or {}
    values = sem.get('features', sem.get('cosines'))
    if values is None:
        return None
    x = np.asarray(values, float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        return None
    x = x - x.mean()
    norm = np.linalg.norm(x)
    return x / norm if norm > TOL else None


def identity_links(observations, pairs):
    """Relative semantic correspondence disambiguates competing localized objects.

    No label mismatch is a veto. An association is challenged only when both
    endpoints prefer another geometrically possible, separately localized region.
    Missing signatures leave the geometric interpretation available.
    """
    rejected = set(); evidence = []
    by_camera = {}
    for o in observations:
        by_camera.setdefault(o['camera'], []).append(o)
    vectors = {o['key']: semantic_vector(o) for o in observations}
    def possible(a, b):
        return bool(np.intersect1d(a['positive_ids'], b['positive_ids']).size)
    def affinity(a, b):
        x, y = vectors[a['key']], vectors[b['key']]
        return float(x @ y) if x is not None and y is not None and x.shape == y.shape else None
    def distinct(a, b):
        # A different prompt alone is insufficient: both must lie in their own
        # visible interior and outside the other explanation's foreground.
        x = np.intersect1d(a.get('anchor_ids', []), a['positive_ids'])
        y = np.intersect1d(b.get('anchor_ids', []), b['positive_ids'])
        return (len(x) and len(y) and not np.intersect1d(x,b['positive_ids']).size
                and not np.intersect1d(y,a['positive_ids']).size)
    for ca, cb in pairs:
        aa, bb = by_camera.get(ca, []), by_camera.get(cb, [])
        for a in aa:
            for b in bb:
                ab = affinity(a,b)
                if ab is None or not possible(a,b):
                    continue
                better_b = [d for d in bb if distinct(b,d) and possible(a,d)
                            and affinity(a,d) is not None and affinity(a,d) > ab + TOL]
                better_a = [c for c in aa if distinct(a,c) and possible(c,b)
                            and affinity(c,b) is not None and affinity(c,b) > ab + TOL]
                if better_a and better_b:
                    rejected.add(frozenset((a['key'],b['key'])))
                    evidence.append(dict(a=a['key'],b=b['key'],reason='reciprocal_regional_semantic_conflict',
                        alternatives_a=sorted(c['key'] for c in better_a),
                        alternatives_b=sorted(d['key'] for d in better_b)))
    return rejected, evidence


def aligned_observation_domains(observations):
    """Align crops without turning alternative whole/part masks into one truth.

    Each interpretation keeps its own labels on its observed domain. Outside
    that crop it can inherit only unanimous labels from overlapping, compatible
    raw observations. Remaining coverage holes are explicit, never background.
    """
    result = {}
    for camera in sorted(set(o['camera'] for o in observations)):
        rows = [o for o in observations if o['camera'] == camera]
        common = np.logical_or.reduce([o['known'] for o in rows])
        for o in rows:
            pos = np.zeros_like(common); neg = np.zeros_like(common)
            for other in rows:
                shared = o['known'] & other['known']
                if not (shared & o['mask'] & other['mask']).any():
                    continue
                # Disagreement on a genuinely co-observed pixel is an
                # alternative explanation, not permission to fill this crop.
                if (shared & (o['mask'] != other['mask'])).any():
                    continue
                pos |= other['known'] & other['mask']
                neg |= other['known'] & ~other['mask']
            extra = ~o['known'] & (pos ^ neg)
            known = o['known'] | extra
            mask = (o['mask'] & o['known']) | (extra & pos)
            result[o['key']] = dict(known=known, mask=mask, common=common,
                uncovered=int((common & ~known).sum()), common_pixels=int(common.sum()))
    return result


def separated_families(a, b, pairs):
    """Reciprocal, independently localized exclusion beats a bridging mask.

    Same-prompt SAM part/whole alternatives cannot establish separation.
    Semantics may corroborate, but never substitutes for this observation test.
    """
    witnessed = []
    for ca, cb in pairs:
        valid = []
        for camera in (ca, cb):
            aa = [o for o in a['observations'] if o['camera'] == camera]
            bb = [o for o in b['observations'] if o['camera'] == camera]
            for x in aa:
                for y in bb:
                    ax=np.intersect1d(x.get('anchor_ids',[]),x['positive_ids'])
                    ay=np.intersect1d(y.get('anchor_ids',[]),y['positive_ids'])
                    if (len(ax) and len(ay) and np.intersect1d(ax,y['negative_ids']).size
                            and np.intersect1d(ay,x['negative_ids']).size):
                        valid.append(camera)
        if ca in valid and cb in valid:
            witnessed.append((ca,cb))
    return witnessed


def identity_partitions(families, selected):
    """Find witnessed partitions even when the selected family is the bridge."""
    unique={}
    for f in families:
        core=np.asarray(f['core_ids'],np.int64)
        if not len(core) or not f['pairs']:continue
        key=(core.tobytes(),np.asarray(f['negative_ids'],np.int64).tobytes(),
             tuple(sorted((o['camera'],tuple(o.get('anchor_ids',[])),
                           np.asarray(o['negative_ids'],np.int64).tobytes()) for o in f['observations'])))
        if key not in unique or f['key']<unique[key]['key']:unique[key]=f
    rows=list(unique.values());result=[]
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            if np.intersect1d(a['core_ids'],b['core_ids']).size:continue
            # Both identities must be observations of this selected source, not
            # arbitrary neighbors anywhere in the neutral camera pool.
            if (not np.intersect1d(a['core_ids'],selected['core_ids']).size or
                    not np.intersect1d(b['core_ids'],selected['core_ids']).size):continue
            witness=separated_families(a,b,sorted(set(a['pairs']+b['pairs'])))
            if witness:result.append((a,b,witness))
    return result


def compose_observed_regions(baseline, proposal, observations, pairs, id_images):
    """Decide connected difference regions; do not demand a core per Gaussian."""
    baseline=np.unique(baseline);proposal=np.unique(proposal)
    changes=np.setxor1d(baseline,proposal)
    images=[np.where(o['known'],im,-1) for o,im in zip(observations,id_images)]
    regions=member_regions({'baseline':baseline,'proposal':proposal},images)
    add=[];remove=[];trace=[]
    for region in regions:
        if not np.intersect1d(region,changes).size:continue
        adding=bool(np.intersect1d(region,proposal).size)
        supporting=set();opposing=set();continuing=set();visible=set()
        for o in observations:
            pos=np.intersect1d(region,o['positive_ids']);neg=np.intersect1d(region,o['negative_ids'])
            if len(pos) or len(neg):visible.add(o['camera'])
            if (len(pos) if adding else len(neg)):supporting.add(o['camera'])
            if (len(neg) if adding else len(pos)):opposing.add(o['camera'])
            if np.intersect1d(baseline,o['positive_ids']).size:continuing.add(o['camera'])
        supporting-=opposing
        independent=[(a,b) for a,b in pairs if a in supporting and b in supporting]
        linked=[(a,b) for a,b in independent if a in continuing and b in continuing]
        accepted=bool(independent and not opposing and (not adding or linked))
        if accepted:(add if adding else remove).extend(region.tolist())
        trace.append(dict(members=region.tolist(),operation='add' if adding else 'remove',
            status='supported' if accepted else 'unresolved',support_views=sorted(supporting),
            opposing_views=sorted(opposing),visible_views=sorted(visible),independent_pairs=independent))
    return np.setdiff1d(np.union1d(baseline,add),remove).astype(np.int64),trace


def competing_families(observations, pairs, width=32):
    """Bounded search of whole explanations, without a scored candidate as input.

    Missing cameras are explicit. A same-camera alternative is not a negative
    vote. Semantic signatures constrain ambiguous identity links, never boundary energy.
    """
    by_camera = {}
    for o in observations:
        by_camera.setdefault(o['camera'], {})[o['key']] = o
    cameras = sorted(by_camera)
    pair_set = {frozenset(p) for p in pairs}
    relations = {}
    rejected_links, semantic_evidence = identity_links(observations, pairs)

    def relation(a, b):
        key = tuple(sorted((a['key'], b['key'])))
        if key not in relations:
            common = np.intersect1d(a['known_ids'], b['known_ids'])
            x = np.intersect1d(a['positive_ids'], common)
            y = np.intersect1d(b['positive_ids'], common)
            union = np.union1d(x, y)
            overlap = np.intersect1d(x, y)
            relations[key] = (0 if frozenset(key) in rejected_links else len(overlap),
                              1-len(overlap)/len(union) if len(union) else None)
        return relations[key]

    def describe(group):
        losses, linked = [], []
        for i, a in enumerate(group):
            for b in group[i+1:]:
                if frozenset((a['camera'], b['camera'])) not in pair_set:
                    continue
                overlap, loss = relation(a, b)
                if loss is not None:
                    losses.append(loss)
                if overlap:
                    linked.append((a['camera'], b['camera']))
        # A camera is eligible only if a raw alternative can continue this
        # family. Other identities and absent surfaces do not count as misses.
        eligible = set(o['camera'] for o in group)
        for camera in cameras:
            if any(frozenset((camera, a['camera'])) in pair_set and relation(a, b)[0]
                   for a in group for b in by_camera[camera].values()):
                eligible.add(camera)
        missing = len(eligible-set(o['camera'] for o in group))/max(1, len(eligible))
        return dict(observations=group, cross_loss=float(np.mean(losses)) if losses else None,
                    missing_loss=missing, pairs=linked, eligible_views=sorted(eligible),
                    key=tuple(sorted(o['key'] for o in group)))

    def rank(f):
        return (f['cross_loss'] is None,
                (f['cross_loss'] or 0.)+f['missing_loss'], -len(f['pairs']), f['key'])

    results = {}
    # Every raw mask seeds a search; the cap applies per seed, so a large easy
    # background family cannot evict all small-object seeds before comparison.
    for seed in sorted(observations, key=lambda o:o['key']):
        beam = [describe([seed])]
        for camera in cameras:
            if camera == seed['camera']:
                continue
            expanded = {}
            for f in beam:
                expanded[f['key']] = f
                for other in by_camera[camera].values():
                    if (not any(frozenset((a['key'],other['key'])) in rejected_links for a in f['observations'])
                            and any(frozenset((camera, a['camera'])) in pair_set and relation(a, other)[0]
                                    for a in f['observations'])):
                        nf = describe(f['observations']+[other])
                        expanded[nf['key']] = nf
            beam = sorted(expanded.values(), key=rank)[:width]
        for f in beam:
            results[f['key']] = f
    ordered=sorted(results.values(), key=rank)
    domains=aligned_observation_domains(list({o['key']:o for o in observations}.values()))
    for f in ordered:
        f['domains']={o['key']:domains[o['key']] for o in f['observations']}
        f['semantic_evidence']=semantic_evidence
        f['core_ids']=independent_support(f['observations'],f['pairs'])
        f['negative_ids']=independent_support(f['observations'],f['pairs'],'negative_ids')
        f['core_ids'],f['negative_ids'],_=symmetric_member_evidence(f['core_ids'],f['negative_ids'])
    return ordered


def explanation_quality(project, families):
    """Compare against complete frozen families; no per-camera mask cherry-pick."""
    scored = []
    projections = {}; count_cache={}
    for f in families:
        counts = []
        for o in f['observations']:
            camera = o['camera']
            if camera not in projections:
                projections[camera] = project(camera)
            if o['key'] not in count_cache:
                d=f['domains'][o['key']]
                e = pixel_evidence(projections[camera], d['known'] & d['mask'],
                                   d['known'] & ~d['mask'])
                count_cache[o['key']]=dict(camera=camera, uncovered=d['uncovered'],
                    common_pixels=d['common_pixels'], **e)
            counts.append(count_cache[o['key']])
        tp = sum(x['tp'] for x in counts); fp = sum(x['fp'] for x in counts)
        fn = sum(x['fn'] for x in counts); denom = tp+fp+fn
        boundary = (fp+fn)/denom if denom else None
        # Cross-camera support is assessed from a different observation. Fits
        # here remain in-sample diagnostics, not held-out model evaluation.
        coverage_missing=sum(c['uncovered'] for c in counts)/max(1,sum(c['common_pixels'] for c in counts))
        missing_loss=1-(1-f['missing_loss'])*(1-coverage_missing)
        energy = ((boundary+f['cross_loss']+missing_loss)/3
                  if boundary is not None and f['cross_loss'] is not None else None)
        core=f['core_ids'];negative=f['negative_ids']
        scored.append(dict(family=f, energy=energy, boundary_loss=boundary,
                           counts=counts, coverage_missing=coverage_missing, missing_loss=missing_loss,
                           core_ids=core, negative_ids=negative,
                           tp=tp, fp=fp, fn=fn))
    valid = [s for s in scored if s['energy'] is not None]
    best = min(valid, key=lambda s:(s['energy'], s['family']['key'])) if valid else None
    return best, scored


def select_object_members(members, baseline_id, project, families):
    """One scoring/selection path for real answers and query counterfactuals."""
    evaluations={};rows=[];cache={}
    for cid,m in members.items():
        key=np.asarray(m,dtype='<i8').tobytes()
        if key not in cache:cache[key]=explanation_quality(lambda camera:project(camera,m),families)
        best,scores=cache[key]
        evaluations[cid]=(best,scores)
        rows.append(dict(id=cid,quality={'energy':best['energy'] if best else None},
            core_ids=np.intersect1d(m,best['core_ids']) if best else np.array([],np.int64),
            negative_ids=best['negative_ids'] if best else np.array([],np.int64)))
    winner,trace=choose_explanation(rows,members,baseline_id)
    return winner['id'],trace,evaluations


def choose_explanation(rows, members, baseline_id):
    """Observable improvement or supported whole extension, never unknown growth."""
    base = next(r for r in rows if r['id'] == baseline_id)
    eligible = [base]; trace = []
    for r in rows:
        q = r['quality']; bq = base['quality']; e = delta_evidence(r, base, members)
        observed = e['supported_additions']+e['supported_removals'] > 0
        informed = q.get('energy') is not None
        strict = informed and (bq.get('energy') is None or q['energy'] < bq['energy']-TOL)
        tied = informed and bq.get('energy') is not None and abs(q['energy']-bq['energy']) <= TOL
        # Unknown members do not veto an observed strict improvement, but a
        # tie must not certify 999 unknown additions through one supported point.
        extension = (tied and e['added_count'] > 0 and not e['removed_count']
                     and e['supported_additions'] == e['added_count']
                     and not e['added_background'])
        accept = len(members[r['id']]) >= 3 and observed and (strict or extension)
        if accept and r is not base:
            eligible.append(r)
        trace.append(dict(candidate_id=r['id'], baseline_id=base['id'],
                          stable=bool(accept), strict_improvement=bool(strict and accept),
                          supported_extension=bool(extension), **e))
    def rank(r):
        energy = r['quality'].get('energy')
        return (energy is None, energy if energy is not None else 1.,
                -delta_evidence(r,base,members)['supported_additions'] if any(r is x for x in eligible) else 0,
                r is not base, np.asarray(members[r['id']],dtype='<i8').tobytes())
    winner = min(eligible, key=rank)
    ordered = sorted(rows, key=rank)
    for t in trace:
        t['selected'] = t['candidate_id'] == winner['id']
    return winner, sorted(trace,key=lambda t:next(i for i,r in enumerate(ordered) if r['id']==t['candidate_id']))


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
