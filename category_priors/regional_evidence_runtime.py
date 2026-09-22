"""Adapter of saved neutral observations to regional evidence (no model calls)."""
from collections import OrderedDict
import hashlib
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion

from .regional_evidence import (VERSION, consensus, mask_key, score_projection,
                               signed_reciprocity, symmetric_member_evidence)
from .multiview_repair import independent_support


def cached_region_semantics(rt, camera, mask, supplied=None):
    """Read exact-mask signatures only; a CPU replay never loads a model."""
    from run_effective_repair import read, slug
    if supplied and supplied.get('cosines') is not None:
        return supplied
    digest=hashlib.blake2b(np.asarray(mask,bool).tobytes(),digest_size=16).hexdigest()
    roots=list(rt.plan.get('neutral_semantic_roots',[]))
    # The original repair encoded its selected SAM mask before the shared
    # three-mask archive existed. That cache is still a valid exact input.
    roots += [getattr(rt,n,None) for n in ('reference','shared','saved_global')]
    for root in roots:
        if root is None:continue
        file=Path(root)/'semantics'/slug(camera)/(digest+'.json')
        if file.exists():return read(file)
    return {'status':'pending_model','reason':'missing_exact_region_encoding'}


def object_observation(rt, camera, mask, observed, path, points=(), semantics=None):
    """The same full-mask decoder for saved and hypothetical query answers."""
    from .regional_evidence import mask_key
    mask=np.asarray(mask,bool);data=rt.data(camera)
    known=binary_erosion(np.asarray(observed,bool),iterations=2,border_value=0)&data['reliable']
    known &= (binary_erosion(mask,iterations=2,border_value=0)|
              binary_erosion(~mask,iterations=2,border_value=0))
    if hasattr(rt,'occlusion_unknown'):
        known &= ~np.isin(data['ids'],rt.occlusion_unknown(camera))
    ids=data['ids'];pos=np.unique(ids[known&mask]);neg=np.unique(ids[known&~mask])
    pos,neg,_=symmetric_member_evidence(pos[pos>=0],neg[neg>=0])
    anchors=[]
    for point in points:
        if point is None:continue
        x,y=np.floor(np.asarray(point)+.5).astype(int)
        if 0<=y<mask.shape[0] and 0<=x<mask.shape[1] and known[y,x] and mask[y,x]:
            anchors.append(int(ids[y,x]))
    return dict(key=camera+':'+mask_key(mask,known),camera=camera,mask=mask,known=known,
        positive_ids=pos,negative_ids=neg,known_ids=np.union1d(pos,neg),
        path=str(path),points=points,anchor_ids=np.unique(anchors).astype(np.int64),
        semantics=cached_region_semantics(rt,camera,mask,semantics))


def merge_object_observations(original, answers):
    """Add raw alternatives identically in preflight and in real inference.

    Duplicate pixels do not gain a camera vote; duplicate prompts preserve their
    provenance rather than silently adopting the first file's identity.
    """
    merged={}
    for o in list(original)+list(answers):
        if o['key'] not in merged:merged[o['key']]=dict(o)
        else:
            old=merged[o['key']]
            old['anchor_ids']=np.union1d(old.get('anchor_ids',[]),o.get('anchor_ids',[])).astype(np.int64)
            if old.get('semantics',{}).get('cosines') is None and o.get('semantics',{}).get('cosines') is not None:
                old['semantics']=o['semantics']
    return [merged[k] for k in sorted(merged)]


def object_observations(rt, pool, views):
    """Read neutral raw masks without authenticating the old prompt as identity."""
    from .regional_evidence import competing_families
    override=getattr(rt,'object_observation_overrides',{}).get((rt.sid,evidence_key(pool,views)))
    if override is not None:return override
    key = (rt.sid, evidence_key(pool, views), rt.plan.get('object_verification_output'))
    cache = rt.__dict__.setdefault('object_observation_cache', OrderedDict())
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    observations = []; missing = []; semantic_missing = []

    def add(camera, mask, observed, path, points, semantics):
        o=object_observation(rt,camera,mask,observed,path,points,semantics)
        observations.append(o)
        if o['semantics'].get('status')=='pending_model':semantic_missing.append(str(path))

    for path in sorted(set(pool)):
        row, arrays, _ = rt.observation(path)
        if arrays is None:
            missing.append(str(path)); continue
        camera = row['camera_uid']
        if camera not in views:
            continue
        actual = row.get('actual_input',{})
        sem = row.get('semantics') or {'status':'pending_model'}
        add(camera, arrays['sam'].astype(bool),
            arrays.get('actual_observed_pixels',arrays['observed_pixels']), path,
            actual.get('positive_points_image') or [actual.get('point_image')],sem)
    # A new full mask is an alternative, not a blanket certificate. Its own
    # camera cannot create independent support. No provisional-anchor gate.
    root = rt.plan.get('object_verification_output')
    if root:
        from run_effective_repair import read
        file = Path(root)/key[1]/'result.json'
        if file.exists():
            answer = read(file)
            if answer['evidence_key'] != key[1] or set(answer['views'])-set(views):
                raise ValueError('Verification camera/context mismatch')
            for view in answer['observations']:
                # Current verification writes real observation folders, including
                # alpha mass. Legacy masks remain usable for evidence but are
                # explicitly missing composition inputs, never silently skipped.
                if view.get('paths'):
                    for path in view['paths']:
                        row,arrays,_=rt.observation(path)
                        if arrays is None:raise ValueError('Missing verification observation: '+path)
                        points=row['actual_input']['positive_points_image']
                        add(view['camera'],arrays['sam'],arrays['actual_observed_pixels'],path,
                            points,row.get('semantics'))
                else:
                    with np.load(view['file'],allow_pickle=False) as z:
                        for endpoint in ('a','b'):
                            for i,mask in enumerate(z[endpoint+'_masks']):
                                add(view['camera'],mask,z['known'],str(view['file'])+':'+endpoint+str(i),
                                    [view[endpoint+'_xy']],{'status':'pending_model'})
    obs = merge_object_observations([],observations)
    support = np.unique(np.concatenate([o['positive_ids'] for o in obs])) if obs else np.array([],int)
    pairs = rt.pairs(sorted(set(o['camera'] for o in obs)),support) if len(support) else []
    families = competing_families(obs,pairs,32)
    # Semantic agreement describes which interpretation is being observed; it
    # neither authenticates the source's intended object nor ranks its boundary.
    for f in families:
        names=[]
        for o in f['observations']:
            sem=o['semantics'];cos=sem.get('cosines')
            if cos is not None and len(cos)==len(rt.plan['classes32']):
                values=np.asarray(cos);idx=np.flatnonzero(values==values.max())
                if len(idx)==1:names.append(rt.plan['classes32'][int(idx[0])])
        f['semantic_labels']=names
        f['semantic_state']='view_disagreement' if len(set(names))>1 else 'consistent_label' if len(names)>=2 else 'pending_model'
    trace = dict(selector='object-explanation-v1',evidence_key=key[1],views=list(views),
        status='competing_explanations' if any(f['pairs'] for f in families) else 'missing_independent_observation',
        observation_count=len(obs),family_count=len(families),beam_width=32,
        missing_paths=missing,pending_semantic_paths=semantic_missing,
        observations=[dict(key=o['key'],camera=o['camera'],path=o['path'],points=o['points'],
                           semantics=o['semantics'],known_pixels=int(o['known'].sum())) for o in obs],
        families=[dict(key=list(f['key']),paths=[o['path'] for o in f['observations']],
                       cross_loss=f['cross_loss'],missing_loss=f['missing_loss'],pairs=f['pairs'],
                       semantic_state=f['semantic_state'],semantic_labels=f['semantic_labels']) for f in families],
        identity_semantic_constraints=families[0]['semantic_evidence'] if families else [],
        identity_status='unresolved',geometric_link_is_not_target_identity=True)
    cache[key] = (obs,families,trace)
    while len(cache)>2:
        cache.popitem(last=False)
    return cache[key]


def score_object(rt, members, pool, views):
    from .regional_evidence import explanation_quality
    _, families, trace = object_observations(rt,pool,views)
    best, scores = explanation_quality(lambda v:rt.project(v,members),families)
    empty = np.array([],np.int64)
    if best is None:
        q=dict(selector='object-explanation-v1',energy=None,score=0.,projection_iou=0.,
               status='missing_observation',identity_status='unresolved',per_view=[])
        return {'class':None,'status':'pending_model'},q,empty,empty,[]
    q=object_quality(best,scores)
    f=best['family']
    sem=rt.classify_members(members,[o['camera'] for o in f['observations']])
    return sem,q,np.intersect1d(members,best['core_ids']),best['negative_ids'],f['observations']


def object_quality(best,scores):
    if best is None:return dict(selector='object-explanation-v1',energy=None,score=0.,
        status='missing_observation',identity_status='unresolved',per_view=[])
    f=best['family']
    return dict(selector='object-explanation-v1',energy=best['energy'],score=1-best['energy'],
        projection_iou=1-best['boundary_loss'],cross_loss=f['cross_loss'],missing_loss=best['missing_loss'],
        coverage_missing=best['coverage_missing'],status='observed_fit',identity_status='competing_explanations',
        family_key=list(f['key']),independent_pairs=f['pairs'],per_view=best['counts'],
        tp=best['tp'],fp=best['fp'],fn=best['fn'],
        identity_semantic_state=f.get('semantic_state','pending_model'),identity_labels=f.get('semantic_labels',[]),
        semantic_score=None,size_penalty=0.,sam_weight=0.,
        family_energies=[dict(key=list(s['family']['key']),energy=s['energy']) for s in scores])



def object_bank(rt, uid, rows, pool, views, dest, metadata, reference_pool):
    from run_effective_repair import read,save
    from .regional_evidence import select_object_members
    if (dest/'bank.json').exists():
        return read(dest/'bank.json')
    if reference_pool is None:
        if any(r['kind']=='new' for r in rows):
            raise ValueError('Explicit neutral observations required')
        reference_pool=pool
    obs,families,trace=object_observations(rt,reference_pool,views)
    save(dest/'reference.json',trace)
    candidates=[];members={};aliases={};content={}
    baseline_id=(metadata or {}).get('baseline_id','original')
    input_members={r['id']:np.unique(r['members']).astype(np.int64) for r in rows}
    if baseline_id not in input_members:baseline_id='original' if 'original' in input_members else rows[0]['id']
    winner_id,selection,evaluations=select_object_members(input_members,baseline_id,rt.project,families)
    # Preserve the fixed bank, including aliases; equal memberships share score.
    for row in rows:
        m=np.unique(row['members']).astype(np.int64);key=m.tobytes()
        if key not in content:
            best,scores=evaluations[row['id']];q=object_quality(best,scores)
            refs=best['family']['observations'] if best else []
            sem=rt.classify_members(m,[o['camera'] for o in refs])
            core=np.intersect1d(m,best['core_ids']) if best else np.array([],np.int64)
            neg=best['negative_ids'] if best else np.array([],np.int64)
            content[key]=(sem,q,core,neg,refs)
        sem,q,core,neg,refs=content[key]
        target=dest/'candidates'/row['id'];target.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(target/'members.npz',members=m,core=core,negative=neg)
        candidates.append(dict(id=row['id'],kind=row['kind'],members_file=str(target/'members.npz'),
            member_count=len(m),paths=row.get('paths',[]),branch=row.get('branch'),semantics=sem,
            quality=q,core_ids=core,negative_ids=neg,associated_paths=[r['path'] for r in refs]))
        members[row['id']]=m
    if baseline_id not in members:
        baseline_id='original' if 'original' in members else candidates[0]['id']
    winner=next(r for r in candidates if r['id']==winner_id)
    save(dest/'selection.json',selection)
    for r in candidates:
        r['core_count']=len(r.pop('core_ids'));r['negative_count']=len(r.pop('negative_ids'))
    result=dict(uid=uid,views=views,pool=sorted(set(pool)),reference_pool=sorted(set(reference_pool)),
        candidates=candidates,selected_id=winner['id'],aliases=aliases,selector='object-explanation-v1',
        ranked_ids=[t['candidate_id'] for t in selection],assembly_candidates=[],identity_candidates=[],**(metadata or {}))
    # Keep pure selection immutable. Region composition and extra identities are
    # different outputs, all through real observation/G0/G2 paths.
    result['composition_missing']=[];result['region_decisions']=[]
    if winner['quality'].get('family_key'):
        from .regional_evidence import compose_observed_regions, separated_families
        family=next(f for f in families if list(f['key'])==winner['quality']['family_key'])
        generated_cache={}
        def generated(f,locator):
            paths=[o['path'] for o in f['observations']]
            missing=[path for path in paths if not (Path(path)/'alpha.npz').is_file()]
            if missing:
                result['composition_missing'].append(dict(family=list(f['key']),paths=missing,
                    reason='missing_real_alpha_observation'))
                return {}
            key=(f['key'],np.asarray(locator,np.int64).tobytes())
            if key not in generated_cache:generated_cache[key]=rt.family_members(paths,locator)
            return generated_cache[key]
        best=None
        for rule,proposal in generated(family,members[winner['id']]).items():
            m,decisions=compose_observed_regions(members[winner['id']],proposal,family['observations'],
                family['pairs'],[rt.data(o['camera'])['ids'] for o in family['observations']])
            result['region_decisions'].append(dict(rule=rule,regions=decisions))
            if len(m)<3 or np.array_equal(m,members[winner['id']]):continue
            sem,q,core,neg,refs=score_object(rt,m,reference_pool,views)
            if q.get('energy') is None:continue
            if best is None or (q['energy'],m.tobytes())<(best[0],best[1].tobytes()):
                best=(q['energy'],m,sem,q,rule)
        if best:
            _,m,sem,q,rule=best;file=dest/'regional-members.npz';np.savez_compressed(file,members=m)
            result['assembly_candidates']=[dict(id='regional-repair',kind='regional_repair',
                members_file=str(file),member_count=len(m),paths=[o['path'] for o in family['observations']],
                semantics=sem,quality=q,rule=rule)]
        alternatives=[]
        for other in families:
            secondary=other['core_ids']
            if not len(secondary) or np.intersect1d(family['core_ids'],secondary).size:continue
            separation=separated_families(family,other,sorted(set(family['pairs']+other['pairs'])))
            if not separation:continue
            # A bridging explanation loses its merge witness; it is not a veto
            # against independently localized reciprocal separation.
            for rule,m in generated(other,secondary).items():
                if len(m)<3 or np.intersect1d(m,members[winner['id']]).size:continue
                sem,q,core,neg,refs=score_object(rt,m,reference_pool,views)
                if q.get('energy') is None or not np.intersect1d(secondary,m).size:continue
                alternatives.append((q['energy'],m.tobytes(),m,sem,q,
                    [o['path'] for o in other['observations']],rule,separation))
        if alternatives:
            _,_,m,sem,q,p2,rule,separation=min(alternatives,key=lambda x:x[:2])
            file=dest/'secondary-identity.npz';np.savez_compressed(file,members=m)
            result['identity_candidates']=[dict(id='secondary-identity',kind='new_identity',members_file=str(file),
                member_count=len(m),paths=p2,semantics=sem,quality=q,rule=rule,separation_pairs=separation,
                object_uid=uid+'#identity2',source_uid=uid)]
        if not result['identity_candidates']:
            from .regional_evidence import identity_partitions
            splits=[]
            scored_generated={}
            def split_rows(f):
                if f['key'] not in scored_generated:
                    values=[]
                    for rule,m in generated(f,f['core_ids']).items():
                        if len(m)<3:continue
                        sem,q,_,_,_=score_object(rt,m,reference_pool,views)
                        if q.get('energy') is not None:values.append((m,sem,q,rule))
                    scored_generated[f['key']]=values
                return scored_generated[f['key']]
            for a,b,separation in identity_partitions(families,family):
                for ra in split_rows(a):
                    for rb in split_rows(b):
                        if np.intersect1d(ra[0],rb[0]).size:continue
                        splits.append((ra[2]['energy']+rb[2]['energy'],ra[0].tobytes(),rb[0].tobytes(),a,b,ra,rb,separation))
            if splits:
                _,_,_,a,b,ra,rb,separation=min(splits,key=lambda x:x[:3])
                records=[]
                for name,f,data in [('identity-partition-primary',a,ra),('secondary-identity',b,rb)]:
                    m,sem,q,rule=data;file=dest/(name+'.npz');np.savez_compressed(file,members=m)
                    records.append(dict(id=name,kind='identity_partition',members_file=str(file),member_count=len(m),
                        paths=[o['path'] for o in f['observations']],semantics=sem,quality=q,rule=rule,
                        separation_pairs=separation,source_uid=uid))
                result['assembly_candidates']=[records[0]]
                result['identity_candidates']=[dict(records[1],object_uid=uid+'#identity2')]
                result['merge_witness_revoked']=list(family['key'])
    save(dest/'bank.json',result)
    print('object bank',uid,len(candidates),'families',len(families),'selected',winner['id'],flush=True)
    return result


def preview_object_answer(rt,bank,current_scene,answers,*,excluded_views=None):
    """Rebuild with raw answers, select once, then use the actual atomic writer.

    This is CPU-only preflight; no candidate filtering or oracle is permitted.
    The fixed library and current scene are the same inputs used after feedback.
    """
    from .regional_evidence import competing_families,select_object_members
    from .regional_evidence_assembly import assemble_selected
    from .multiview_repair_experiment import ids_file
    from .decision_verification import content_id
    obs,_,trace=object_observations(rt,bank['reference_pool'],bank['views'])
    updated=merge_object_observations(obs,answers)
    support=np.unique(np.concatenate([o['positive_ids'] for o in updated])) if updated else np.array([],int)
    pairs=rt.pairs(bank['views'],support) if len(support) else []
    families=competing_families(updated,pairs,32)
    members={r['id']:ids_file(r['members_file']) for r in bank['candidates']}
    cid,selection,evaluations=select_object_members(members,bank['selected_id'],rt.project,families)
    rows=[dict(r,quality=object_quality(*evaluations[r['id']])) for r in bank['candidates']]
    trial=dict(bank,candidates=rows,selected_id=cid,ranked_ids=[t['candidate_id'] for t in selection],
               assembly_candidates=[],identity_candidates=[])
    key=(rt.sid,evidence_key(bank['reference_pool'],bank['views']))
    old=getattr(rt,'object_observation_overrides',{})
    try:
        rt.object_observation_overrides=dict(old)
        rt.object_observation_overrides[key]=(updated,families,trace)
        scene=assemble_selected(rt,{bank['uid']:trial},Path('.'),baseline=current_scene,
            excluded_views=excluded_views,include_regional=False,dry_run=True)
    finally:rt.object_observation_overrides=old
    best=evaluations[cid][0]
    return dict(selected_members=content_id(members[cid]),
        identity_support=content_id(best['core_ids']) if best else None,
        actual_members=content_id(rt.actual_for(scene,bank['uid'])))


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
