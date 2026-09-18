"""Adapter of saved neutral observations to regional evidence (no model calls)."""
from collections import OrderedDict
import hashlib
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion

from .regional_evidence import (VERSION, consensus, mask_key, score_projection,
                               signed_reciprocity, symmetric_member_evidence)
from .multiview_repair import independent_support


def evidence_key(pool, views):
    return hashlib.sha256(('\n'.join(sorted(set(pool)))+'\n'+'\n'.join(sorted(set(views)))).encode()).hexdigest()


def build_reference(rt,pool,views):
    from run_effective_repair import read
    revision=rt.plan.get('regional_evidence_revision','original-v2')
    if revision not in ('original-v2','known-domain-v1'):
        raise ValueError('Unknown regional evidence revision: '+revision)
    key=(rt.sid,evidence_key(pool,views),revision)
    cache=rt.__dict__.setdefault('regional_references',OrderedDict())
    if key in cache:
        cache.move_to_end(key);return cache[key]
    cameras={};paths=[]
    for path in sorted(set(pool)):
        row,arrays,_=rt.observation(path)
        if arrays is None or row['camera_uid'] not in views:continue
        camera=row['camera_uid'];data=rt.data(camera);mask=arrays['sam']
        points=row.get('actual_input',{}).get('positive_points_image') or [row.get('actual_input',{}).get('point_image')]
        anchors=[]
        for point in points:
            if point is None:continue
            x,y=np.floor(np.asarray(point)+.5).astype(int)
            if 2<=y<mask.shape[0]-2 and 2<=x<mask.shape[1]-2 and data['reliable'][y,x] and mask[y,x]:
                anchors.append(int(data['ids'][y,x]))
        if not anchors:continue
        observed=arrays.get('actual_observed_pixels',arrays['observed_pixels']).astype(bool)
        known=binary_erosion(observed,iterations=2,border_value=0)&data['reliable']
        # Exclude both sides of an uncertain boundary, symmetrically.
        interior=binary_erosion(mask,iterations=2,border_value=0)
        exterior=binary_erosion(~mask,iterations=2,border_value=0)
        known &= interior|exterior
        if hasattr(rt,'occlusion_unknown'):
            known &= ~np.isin(data['ids'],rt.occlusion_unknown(camera))
        hkey=mask_key(mask,known)
        value=cameras.setdefault(camera,dict(hypotheses={},anchors=set(),paths=[]))
        value['hypotheses'].setdefault(hkey,dict(key=hkey,mask=mask,known=known,path=path))
        value['anchors'].update(anchors);value['paths'].append(path);paths.append(path)
    refs=[]
    for camera,value in sorted(cameras.items()):
        hs=[value['hypotheses'][k] for k in sorted(value['hypotheses'])]
        fg,bg,dispute=consensus(hs);ids=rt.data(camera)['ids']
        refs.append(dict(camera=camera,path=min(value['paths']),hypotheses=hs,
            foreground=fg,background=bg,disputed=dispute,anchor_ids=np.array(sorted(value['anchors']),np.int64),
            positive_ids=np.unique(ids[fg]),negative_ids=np.unique(ids[bg]),
            known_ids=np.unique(ids[fg|bg]),disputed_ids=np.unique(ids[dispute])))
    anchors=np.unique(np.concatenate([r['anchor_ids'] for r in refs])) if refs else np.array([],int)
    pairs=rt.pairs([r['camera'] for r in refs],anchors) if len(anchors) else []
    bycamera={r['camera']:r for r in refs}
    # Cross-camera reciprocal anchor inclusion is necessary. Do not pool
    # unrelated reliable points into one instance merely because they project.
    linked=[]
    for a,b in pairs:
        ra,rb=bycamera[a],bycamera[b]
        if (np.intersect1d(ra['anchor_ids'],rb['positive_ids']).size and
            np.intersect1d(rb['anchor_ids'],ra['positive_ids']).size):linked.append((a,b))
    verification=None
    root=rt.plan.get('regional_verification_output')
    if root:
        file=Path(root)/key[1]/'result.json'
        if file.exists():
            verification=read(file)
            if verification['evidence_key']!=key[1] or set(verification['views'])-set(views):
                raise ValueError('verification evidence does not match neutral pool and allowed views')
            vrows=[]
            for v in verification['observations']:
                with np.load(v['file'],allow_pickle=False) as z:
                    vrows.append(dict(camera=v['camera'],a_xy=v['a_xy'],b_xy=v['b_xy'],**{k:z[k] for k in z.files}))
            relation,signs=signed_reciprocity(vrows,pairs)
            verification=dict(verification,relation=relation,signs=signs)
            scoped=verification.get('version')=='decision-query-v1'
            if scoped:
                # A planned question is not evidence. A provisional localization
                # may receive an answer but cannot silently become a target core.
                pre_positive=independent_support(refs,linked)
                pre_negative=independent_support(refs,linked,'negative_ids')
                pre_positive,_,_=symmetric_member_evidence(pre_positive,pre_negative)
                if verification['anchor_status']=='certified' and verification['a_id'] not in pre_positive:
                    raise ValueError('Query anchor certification does not match its neutral observations')
                verification['certified_pixels_by_view']={}
            for v in vrows:
                ref=bycamera[v['camera']];ids=rt.data(ref['camera'])['ids']
                if relation=='unresolved':continue
                if scoped:
                    from .decision_verification import scoped_answer_masks
                    fg,bg=scoped_answer_masks(v,verification['region_ids'],ids,relation,verification['anchor_status'])
                    qualified=fg|bg
                    verification['certified_pixels_by_view'][v['camera']]=int(qualified.sum())
                    if not qualified.any():continue
                    # Replace contradictory evidence ONLY at witnessed queried
                    # pixels. Never throw away a complete old interpretation for
                    # a point relation or certify the rest of the new SAM mask.
                    hs=[dict(h,known=h['known']&~qualified,
                        key=mask_key(h['mask'],h['known']&~qualified)) for h in ref['hypotheses']]
                    hs.append(dict(key=mask_key(fg,qualified),mask=fg,known=qualified,path=v['camera']+':scoped-verification'))
                    ref['hypotheses']=list({h['key']:h for h in hs}.values())
                    fg,bg,dispute=consensus(ref['hypotheses'])
                    ref.update(foreground=fg,background=bg,disputed=dispute,
                        positive_ids=np.unique(ids[fg]),negative_ids=np.unique(ids[bg]),
                        known_ids=np.unique(ids[fg|bg]),disputed_ids=np.unique(ids[dispute]))
                    continue
                # Discard an old interpretation only on an observed signed
                # contradiction, not because it agrees poorly with a candidate.
                region=np.all(v['b_masks'],axis=0)&v['known']
                hs=[]
                for h in ref['hypotheses']:
                    tested=region&h['known']
                    contradiction=tested & (h['mask'] if relation=='separate' else ~h['mask'])
                    if not contradiction.any():hs.append(h)
                for mask in v['a_masks']:
                    known=v['known']&(binary_erosion(mask,iterations=2)|binary_erosion(~mask,iterations=2))
                    hs.append(dict(key=mask_key(mask,known),mask=mask,known=known,path=v['camera']+':verification'))
                ref['hypotheses']=list({h['key']:h for h in hs}.values())
                fg,bg,dispute=consensus(ref['hypotheses'])
                ref.update(foreground=fg,background=bg,disputed=dispute,
                    positive_ids=np.unique(ids[fg]),negative_ids=np.unique(ids[bg]),
                    known_ids=np.unique(ids[fg|bg]),disputed_ids=np.unique(ids[dispute]))
            if relation!='unresolved' and (not scoped or verification['anchor_status']=='certified'):
                # A is the same three-dimensional target anchor in both calls;
                # reciprocity certifies only these independently checked views.
                linked=list({tuple(p) for p in linked+[(a,b) for a,b in pairs if a in signs and b in signs]})
    positive=independent_support(refs,linked)
    negative=independent_support(refs,linked,'negative_ids')
    disputed_members=np.intersect1d(positive,negative)
    if revision=='known-domain-v1':
        positive,negative,disputed_members=symmetric_member_evidence(positive,negative)
    else:
        positive=np.setdiff1d(positive,negative)
    trace=dict(selector=VERSION,status='resolved' if linked else 'identity_ambiguous',
        evidence_key=key[1],views=list(views),paths=paths,
        missing_views=sorted(set(views)-set(cameras)),independent_pairs=linked,
        geometric_pairs=pairs,positive_count=len(positive),negative_count=len(negative),
        per_view=[dict(camera=r['camera'],hypothesis_count=len(r['hypotheses']),
            foreground_pixels=int(r['foreground'].sum()),background_pixels=int(r['background'].sum()),
            disputed_pixels=int(r['disputed'].sum()),anchors=r['anchor_ids'].tolist()) for r in refs],
        verification=verification,evidence_revision=revision,
        disputed_member_count=len(disputed_members),
        identity_linked=bool(linked),boundary_resolved=False)
    if revision=='known-domain-v1' and linked:
        trace['status']='identity_linked'  # Identity linkage does not certify the boundary.
    result=(refs,linked,trace,positive,negative)
    cache[key]=result
    while len(cache)>2:cache.popitem(last=False)
    return result


def score(rt,members,pool,views):
    refs,pairs,trace,positive,negative=build_reference(rt,pool,views)
    # Per-object projection cache; never stores evaluation projections here.
    key=trace['evidence_key'];cache=rt.__dict__.setdefault('regional_projections',OrderedDict())
    def project(camera):
        k=(rt.sid,key,camera,hashlib.sha256(np.asarray(members,dtype='<i8').tobytes()).digest())
        if k not in cache:cache[k]=rt.project(camera,members)
        cache.move_to_end(k)
        while len(cache)>96:cache.popitem(last=False)
        return cache[k]
    q=score_projection(project,refs,known_domain=rt.plan.get('regional_evidence_revision')=='known-domain-v1')
    q['independent_pairs']=pairs;q['identity_status']=trace['status']
    sem=rt.classify_members(members,[r['camera'] for r in refs])
    return sem,q,np.intersect1d(members,positive),negative,refs


def regional_candidate(rt,bank,dest):
    from run_effective_repair import save
    from .regional_evidence import member_regions,repair_regions
    from .multiview_repair_experiment import ids_file
    refs,_,trace,positive,negative=build_reference(rt,bank['reference_pool'],bank['views'])
    members={r['id']:ids_file(r['members_file']) for r in bank['candidates']}
    baseline=members[bank['selected_id']]
    images=[np.where(rt.data(r['camera'])['reliable'],rt.data(r['camera'])['ids'],-1) for r in refs]
    regions=member_regions(members,images)
    repaired,changes=repair_regions(baseline,regions,positive,negative)
    save(dest/'regional-repair.json',dict(source_candidate=bank['selected_id'],
        status='partially_resolved' if not np.array_equal(repaired,baseline) else 'unresolved',regions=changes,
        evidence_key=trace['evidence_key'],not_in_original_oracle_library=True))
    if np.array_equal(repaired,baseline):return None
    target=dest/'regional-members.npz';np.savez_compressed(target,members=repaired)
    sem,q,core,neg,_=score(rt,repaired,bank['reference_pool'],bank['views'])
    return dict(id='regional_repair',kind='regional_repair',members_file=str(target),member_count=len(repaired),
        paths=[],semantics=sem,quality=q,core_count=len(core),negative_count=len(neg))
