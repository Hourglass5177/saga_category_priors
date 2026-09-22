"""One simultaneous, evidence-qualified ownership update and one fallback pass."""
import numpy as np
from .regional_evidence import VERSION, select_regional, stable, paired_difference
from .common_geometry_selector import semantic_state, TOL
from .prediction_contract import normalize_prediction


def commit_selected(incumbent, selected):
    """Simultaneous exact writeback, with real conflicts returned to the caller."""
    old=np.asarray(incumbent,np.int64)
    proposals={int(k):np.unique(np.asarray(v,np.int64)) for k,v in selected.items()}
    claims=np.full(old.shape,-1,np.int64)
    for label,m in proposals.items():
        if label<0 or np.any((m<0)|(m>=len(old))):raise ValueError('invalid member or label')
        if np.any(claims[m]>=0):raise ValueError('joint ownership required')
        claims[m]=label
    unchanged=(old>=0)&~np.isin(old,list(proposals))
    if np.any(unchanged&(claims>=0)):raise ValueError('joint ownership required')
    result=old.copy();result[np.isin(old,list(proposals))]=-1
    result[claims>=0]=claims[claims>=0]
    return result


def compatible_choices(options, cost, limit=50000, width=32):
    """One complete proposal per identity. Deterministic bounded joint search."""
    import itertools
    count=1
    for opts in options:count*=len(opts)
    def compatible(chosen):
        used=set()
        for r in chosen:
            m=set(map(int,r['members']))
            if used&m:return False
            used.update(m)
        return True
    def rank(combo):
        return (sum(cost(i,r) for i,r in enumerate(combo)),
                sum(r.get('changed',False) for r in combo),
                tuple(np.asarray(r['members'],dtype='<i8').tobytes() for r in combo))
    if count<=limit:
        best=min((c for c in itertools.product(*options) if compatible(c)),key=rank,default=None)
        return best,dict(search='exact',combinations=count)
    beam=[()]
    for opts in options:
        expanded=[c+(r,) for c in beam for r in opts if compatible(c+(r,))]
        beam=sorted(expanded,key=rank)[:width]
    return (min(beam,key=rank) if beam else None),dict(search='beam',combinations=count,width=width)


def localize_conflict(proposals, resolved, region):
    """Commit disjoint improvements; ownership fallback touches only the dispute.

    The result is explicitly a composition. Callers must retain the intact
    library selection separately rather than reporting it as pure selection.
    """
    result={}
    for u,proposal in proposals.items():
        settled=resolved[u]
        m=np.union1d(np.setdiff1d(proposal['members'],region),
                     np.intersect1d(settled['members'],region)).astype(np.int64)
        changed=not np.array_equal(m,proposal['members'])
        result[u]=dict(proposal,members=m,
            id='ownership-composition' if changed else proposal['id'],
            kind='ownership_composition' if changed else proposal.get('kind'),
            intact_candidate=not changed,conflict_choice=settled['id'])
    return result


def assemble_selected(rt,banks,dest,*,baseline=None,excluded_views=None,include_regional=False,dry_run=False):
    """Honor selected IDs; jointly solve only actual complete-proposal overlap."""
    from run_effective_repair import read,save
    from .multiview_repair_experiment import ids_file
    from .multiview_repair import member_components
    from .regional_evidence_runtime import score_object
    from .common_geometry_selector import pixel_evidence
    if not dry_run and (dest/'summary.json').exists():return read(dest/'scene.json')
    if not dry_run:dest.mkdir(parents=True,exist_ok=True)
    base=baseline if baseline is not None else rt.b0
    canon=lambda u:'B0:'+str(int(u.rsplit(':',1)[1])) if ':B0:' in u else u
    original_banks=banks;banks={canon(u):b for u,b in banks.items()}
    if include_regional:
        for u,b in list(banks.items()):
            for r in b.get('identity_candidates',[])[:1]:
                banks[u+'#identity2']=dict(b,uid=u+'#identity2',candidates=[r],selected_id=r['id'],
                    ranked_ids=[r['id']],assembly_candidates=[],identity_candidates=[])
    labels=np.asarray(base['point_labels'],np.int64)
    old={canon(m.get('object_uid','B0:'+str(k))):(np.flatnonzero(labels==int(k)),dict(m))
         for k,m in base['instances'].items()}
    uids=sorted(set(old)|set(banks));indices={u:i for i,u in enumerate(uids)}
    owner0=np.full(len(labels),-1,np.int64)
    for u,(m,_) in old.items():owner0[m]=indices[u]
    proposed={};options={};contexts={};trace=[]
    for u in uids:
        b=banks.get(u)
        if b is None:
            r=dict(id='unchanged',members=old[u][0],changed=False)
            proposed[u]=r;options[u]=[r];continue
        forbidden=set((excluded_views or {}).get(u,[]))
        # Explicit exclusions identify a different (tail/full) task context.
        # Never import the local diagnostic's heldout camera into that context.
        # Query reuse still requires the exact evidence key and camera set.
        if excluded_views is None:
            for case in rt.plan.get('local_cases',[]):
                if canon(case['candidate_uid'])==u:
                    forbidden.update(v['camera_uid'] for v in case['views'][2:])
        if set(b['views'])&forbidden:raise ValueError('Evaluation camera in frozen object context')
        winner=next(r for r in b['candidates'] if r['id']==b['selected_id'])
        if include_regional and b.get('assembly_candidates'):winner=b['assembly_candidates'][0]
        def load(r):return dict(r,members=ids_file(r['members_file']),changed=True)
        proposed[u]=load(winner)
        runners=[r for cid in b.get('ranked_ids',[]) for r in b['candidates']
                 if r['id']==cid and r['id']!=winner['id']]
        opts=[proposed[u]]+([load(runners[0])] if runners else [])
        opts.append(dict(id='actual-incumbent',members=old[u][0] if u in old else np.array([],np.int64),changed=False))
        options[u]=list({r['members'].tobytes():r for r in reversed(opts)}.values())
        contexts[u]=score_object(rt,proposed[u]['members'],b['reference_pool'],b['views'])[-1]
    # The three actual options define ownership components. Camera exclusions
    # are already frozen above and never change with component size.
    comps=member_components([(u,r['members']) for u in uids for r in options[u]],len(labels))
    chosen={};unresolved=[]
    for component in comps:
        if len(component)==1:
            u=component[0];chosen[u]=proposed[u];continue
        restricted=[]
        outside=set(uids)-set(component)
        outside_members=np.unique(np.concatenate([proposed[v]['members'] for v in outside])) if outside else np.array([],int)
        for u in component:
            restricted.append([r for r in options[u] if not np.intersect1d(r['members'],outside_members).size])
        # Actual disputed members, including runner overlaps within this small component.
        claims={}
        for u,opts in zip(component,restricted):
            union=np.unique(np.concatenate([r['members'] for r in opts])) if opts else np.array([],int)
            for g in union:claims.setdefault(int(g),set()).add(u)
        region=np.array([g for g,us in claims.items() if len(us)>1],np.int64)
        costs={};observed=False
        for i,(u,opts) in enumerate(zip(component,restricted)):
            for r in opts:
                tp=fp=fn=0
                for o in contexts.get(u,[]):
                    domain=o['known']&np.isin(rt.data(o['camera'])['ids'],region)
                    e=pixel_evidence(rt.project(o['camera'],r['members']),domain&o['mask'],domain&~o['mask'])
                    tp+=e['tp'];fp+=e['fp'];fn+=e['fn']
                total=tp+fp+fn;observed|=total>0
                costs[i,r['members'].tobytes()]=(fp+fn)/total if total else 0.
        solution,search=compatible_choices(restricted,lambda i,r:costs[i,r['members'].tobytes()])
        if solution is None or not observed:
            solution=tuple(dict(id='actual-incumbent',members=old[u][0] if u in old else np.array([],np.int64),changed=False) for u in component)
            unresolved.extend(component)
        settled=dict(zip(component,solution))
        if include_regional:
            settled=localize_conflict({u:proposed[u] for u in component},settled,region)
        for u,r in settled.items():chosen[u]=r
        trace.append(dict(uids=component,disputed_members=len(region),observed=observed,
                          selected={u:r['id'] for u,r in settled.items()},
                          fallback_scope='disputed_members_only' if include_regional else 'intact_library_options',
                          uncontested_preserved=bool(include_regional),**search))
    # All choices are complete. The atomic writer is the only materialization.
    owner=commit_selected(owner0,{indices[u]:r['members'] for u,r in chosen.items()})
    metadata={};ownership=[]
    for u,r in chosen.items():
        m=r['members'];k=indices[u]
        if len(m)<3:
            owner[owner==k]=-1;continue
        if u in old and np.array_equal(m,old[u][0]):meta=dict(old[u][1])
        else:
            b=banks[u];sem=rt.classify_members(m,b['views'])
            label,status=semantic_state(sem,rt.assets.saga20)
            if sem.get('status')=='pending_model':status='pending_model'
            meta=dict(**{'class':label},score=r.get('quality',{}).get('score',0.),
                      semantics=sem,classification_status=status,semantic_views=b['views'])
        meta.update(object_uid=u,point_count=len(m),selected_candidate=r['id'],score_source='object-explanation-v1')
        metadata[k]=meta
        ownership.append(dict(uid=u,requested_candidate=proposed[u]['id'],written_candidate=r['id'],
            requested_members=len(proposed[u]['members']),written_members=len(m),
            exact=np.array_equal(m,proposed[u]['members']),
            result_kind=r.get('kind','pure_selection'),conflict_choice=r.get('conflict_choice')))
    normalized=normalize_prediction(owner,metadata)
    payload=dict(point_labels=normalized.point_labels.tolist(),instances=normalized.instances,
        prediction_contract=normalized.audit,repair_policy='object-explanation-v1',
        instance_aliases={u:canon(u) for u in original_banks},
        classification_complete=not any(m.get('classification_status')=='pending_model' for m in metadata.values()))
    if dry_run:return payload
    save(dest/'scene.json',payload,compact=True)
    if payload['classification_complete']:
        allowed={k:v for k,v in normalized.instances.items() if v['class'] in rt.assets.saga20}
        exported=normalize_prediction(normalized.point_labels,allowed)
        save(dest/'scene-evaluation.json',dict(point_labels=exported.point_labels.tolist(),instances=exported.instances,
                                             geometry_source=str(dest/'scene.json')),compact=True)
    np.savez_compressed(dest/'actual-members.npz',**{u:rt.actual_for(payload,u) for u in original_banks})
    save(dest/'summary.json',dict(execution_complete=True,selector='object-explanation-v1',
        ownership=ownership,conflicts=trace,unresolved=unresolved,include_regional=include_regional))
    return payload


def assemble(rt,banks,dest,*,baseline=None,excluded_views=None,include_regional=True):
    from run_effective_repair import read,save
    from .multiview_repair_experiment import ids_file
    from .multiview_repair import member_components
    if (dest/'summary.json').exists():return read(dest/'scene.json')
    dest.mkdir(parents=True,exist_ok=True);baseline=baseline or rt.b0
    canon=lambda u:'B0:'+str(int(u.rsplit(':',1)[1])) if ':B0:' in u else u
    input_alias={u:canon(u) for u in banks};banks={canon(u):b for u,b in banks.items()}
    labels=np.asarray(baseline['point_labels'],np.int64);n=len(labels)
    old={canon(m.get('object_uid','B0:'+str(k))):(np.flatnonzero(labels==int(k)),m) for k,m in baseline['instances'].items()}
    libraries={};context={};scores={};before={};choices={};traces={}
    raw={u:[(dict(id='committed',kind='incumbent'),m)] for u,(m,_) in old.items()}
    for u,b in banks.items():
        extra=b.get('assembly_candidates',[]) if include_regional else []
        raw.setdefault(u,[]).extend((r,ids_file(r['members_file'])) for r in b['candidates']+extra)
    components=member_components([(u,m) for u,rs in raw.items() for _,m in rs],n)
    forbidden={canon(u):set(v) for u,v in (excluded_views or {}).items()}
    for c in rt.plan.get('local_cases',[]):
        forbidden.setdefault(canon(c['candidate_uid']),set()).update(v['camera_uid'] for v in c['views'][2:])
    for component in components:
        excluded=set().union(*(forbidden.get(u,set()) for u in component))
        for u in component:
            b=banks.get(u);vs=[v for v in b['views'] if v not in excluded] if b else []
            pool=[p for p in b['reference_pool'] if rt.descriptor(p)['camera'] in vs] if b else []
            context[u]=pool,vs
    def score(u,m,r):
        m=np.unique(m).astype(np.int64);key=u,m.tobytes()
        if key not in scores:scores[key]=rt.score_evidence(m,*context[u])
        sem,q,core,neg,refs=scores[key]
        return dict(r,uid=u,members=m,semantics=sem,quality=q,core_ids=core,negative_ids=neg,refs=refs)
    for u,rs in raw.items():
        lib=[score(u,m,r) for r,m in rs];libraries[u]=lib
        if u in old:before[u]=lib[0]
        winner,trace=select_regional(lib,{r['id']:r['members'] for r in lib},'committed' if u in old else banks[u]['selected_id'])
        choices[u]=winner;traces[u]=trace
    uids=sorted(raw);index={u:i for i,u in enumerate(uids)}
    duplicate_alias={}
    retained=[]
    for u in sorted(uids,key=lambda u:(u not in before,choices[u]['members'].tobytes(),u)):
        row=choices[u];core=np.setdiff1d(row['core_ids'],row['negative_ids'])
        duplicate=None
        if len(core):
            for v in retained:
                other=choices[v];othercore=np.setdiff1d(other['core_ids'],other['negative_ids'])
                intersection=len(np.intersect1d(row['members'],other['members']))
                union=len(np.union1d(row['members'],other['members']))
                if np.array_equal(core,othercore) and intersection/max(1,union)>.5:
                    duplicate=v;break
        if duplicate is not None:duplicate_alias[u]=duplicate
        else:retained.append(u)
    owner0=np.full(n,-1,np.int32)
    for u,(m,_) in old.items():owner0[m]=index[duplicate_alias.get(u,u)]
    owner=owner0.copy();claims=[[] for _ in uids];count=np.zeros(n,np.int32);claimant=np.full(n,-1,np.int32)
    proposed_count=np.zeros(n,np.int32);proposed_owner=np.full(n,-1,np.int32)
    for u,r in choices.items():
        if u in duplicate_alias:continue
        proposed_count[r['members']]+=1;proposed_owner[r['members']]=index[u]
    # Retain a selected whole proposal on unclaimed members. Unknown evidence
    # cannot silently trim it to just its independently protected core.
    free=(owner<0)&(proposed_count==1)
    owner[free]=proposed_owner[free]
    for u,r in choices.items():
        if u in duplicate_alias:continue
        core=np.setdiff1d(r['core_ids'],r['negative_ids']);claims[index[u]]=core
        count[core]+=1;claimant[core]=index[u]
    # Only an exclusive independently supported claim may transfer a member.
    unique=count==1;owner[unique]=claimant[unique]
    for u,r in choices.items():
        if u in duplicate_alias:continue
        if u not in old:continue
        removed=np.setdiff1d(old[u][0],r['members'])
        justified=removed[np.isin(removed,r['negative_ids']) & (owner[removed]==index[u])]
        owner[justified]=-1
    # Unknown regions retain the original owner. Duplicate objects cannot gain
    # weight from candidate counts or take a core just by a higher global score.
    replacements=[]
    def state():
        return {u:score(u,np.flatnonzero(owner==index[u]),choices[u]) for u in uids}
    after=state()
    def valid(u,row):
        if u in duplicate_alias:return True
        if u not in before:return not np.setdiff1d(choices[u]['core_ids'],row['members']).size
        b=before[u];lost=np.setdiff1d(b['core_ids'],row['members'])
        if np.setdiff1d(lost,b['negative_ids']).size:return False
        d=paired_difference(row,b)
        return d is not None and all(v>=-TOL for v in d.values())
    failed=[u for u in uids if not valid(u,after[u])]
    for u in failed:
        occupied=(owner>=0)&(owner!=index[u]);attempted=[];trials=[dict(after[u],id='__assigned_members__')]
        for r in libraries[u]:
            available=r['members'][~occupied[r['members']]]
            candidate=score(u,available,r)
            ok=valid(u,candidate)
            attempted.append(dict(candidate_id=r['id'],valid=ok))
            if ok:trials.append(candidate)
        chosen,_=select_regional(trials,{r['id']:r['members'] for r in trials},'__assigned_members__')
        # Apply only certified differences; a fallback is not another claim round.
        current=np.flatnonzero(owner==index[u]);added=np.setdiff1d(chosen['members'],current)
        added=added[np.isin(added,chosen['core_ids']) & ~occupied[added]]
        removed=np.setdiff1d(current,chosen['members']);removed=removed[np.isin(removed,chosen['negative_ids'])]
        owner[added]=index[u];owner[removed]=-1
        replacements.append(dict(uid=u,selected=chosen['id'],attempts=attempted))
    after=state();rollback=np.zeros(n,bool)
    for u in uids:
        if not valid(u,after[u]):
            rollback |= (owner!=owner0)&((owner==index[u])|(owner0==index[u]))
    owner[rollback]=owner0[rollback];after=state()
    metadata={};outputlabels=np.full(n,-1,np.int64);ownership=[]
    for u,r in after.items():
        m=r['members']
        if len(m)<3:continue
        k=len(metadata);outputlabels[m]=k
        if u in old and np.array_equal(m,old[u][0]):
            meta=dict(old[u][1]);meta.update(object_uid=u,point_count=len(m),score_source=VERSION)
        else:
            sem=r['semantics'];label,status=semantic_state(sem,rt.assets.saga20)
            if sem.get('status')=='pending_model':status='pending_model'
            meta=dict(**{'class':label},object_uid=u,classification_status=status,semantics=sem,
                score=r['quality']['score'],point_count=len(m),score_source=VERSION,
                selected_candidate=r['id'],semantic_views=[x['camera'] for x in r['refs']])
        metadata[k]=meta
        ownership.append(dict(uid=u,selected_candidate=r['id'],final_members=len(m),
            changed_members=int(((owner!=owner0)&((owner==index[u])|(owner0==index[u]))).sum())))
    normalized=normalize_prediction(outputlabels,metadata)
    aliases=dict(duplicate_alias,**{u:duplicate_alias.get(v,v) for u,v in input_alias.items()})
    payload=dict(point_labels=normalized.point_labels.tolist(),instances=normalized.instances,
        prediction_contract=normalized.audit,repair_policy=VERSION,instance_aliases=aliases,
        classification_complete=not any(m.get('classification_status')=='pending_model' for m in metadata.values()))
    save(dest/'scene.json',payload,compact=True)
    if payload['classification_complete']:
        allowed={k:v for k,v in normalized.instances.items() if v['class'] in rt.assets.saga20}
        exported=normalize_prediction(normalized.point_labels,allowed)
        save(dest/'scene-evaluation.json',dict(point_labels=exported.point_labels.tolist(),instances=exported.instances,
            prediction_contract=exported.audit,geometry_source=str(dest/'scene.json')),compact=True)
    actual={u:rt.actual_for(payload,u) for u in raw};actual.update({u:rt.actual_for(payload,u) for u in input_alias})
    np.savez_compressed(dest/'actual-members.npz',**actual)
    save(dest/'summary.json',dict(execution_complete=True,assembly=VERSION,include_regional=include_regional,
        selection=traces,ownership=ownership,replacements=replacements,rollback_members=np.flatnonzero(rollback).tolist()))
    return payload
