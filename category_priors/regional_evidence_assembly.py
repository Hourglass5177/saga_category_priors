"""One simultaneous, evidence-qualified ownership update and one fallback pass."""
import numpy as np
from .regional_evidence import VERSION, select_regional, stable, paired_difference
from .common_geometry_selector import semantic_state, TOL
from .prediction_contract import normalize_prediction


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
