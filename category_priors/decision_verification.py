"""Choose bounded verification questions about candidate differences, without GT.

The input is a source-blind union of saved member sets plus neutral observations.
Counterfactual split counts describe query utility, not predicted correctness.
"""
from __future__ import annotations
import hashlib
import numpy as np
from scipy.ndimage import distance_transform_edt,binary_erosion
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

VERSION='decision-query-v1'


def canonical_candidates(member_sets):
    unique={np.unique(np.asarray(m,np.int64)).astype('<i8').tobytes() for m in member_sets}
    return [np.frombuffer(k,dtype='<i8').copy() for k in sorted(unique) if k]


def content_id(members):
    return hashlib.sha256(np.unique(members).astype('<i8').tobytes()).hexdigest()


def positions(ids,allowed,region_ids):
    """Pick support inside the REGION, never by distance to photo centre."""
    region=allowed & np.isin(ids,region_ids)
    yy,xx=np.where(region)
    if not len(yy):return {}
    h,w=ids.shape
    clearance=distance_transform_edt(np.pad(region,1))[1:-1,1:-1]
    margin=np.minimum.reduce([xx,yy,w-1-xx,h-1-yy])
    keep=margin>2;yy,xx,margin=yy[keep],xx[keep],margin[keep]
    order=np.lexsort((xx,yy,-margin,-clearance[yy,xx]))
    result={}
    for i in order:
        g=int(ids[yy[i],xx[i]])
        result.setdefault(g,(float(clearance[yy[i],xx[i]]),int(margin[i]),[int(xx[i]),int(yy[i])]))
    return result


def best_shared_point(maps,eligible):
    common=set(map(int,eligible))
    for p in maps:common.intersection_update(p)
    if not common:return None
    def rank(g):
        return (-min(p[g][0] for p in maps),-min(p[g][1] for p in maps),g)
    g=min(common,key=rank)
    return dict(id=g,xy=[p[g][2] for p in maps],
                clearance=min(p[g][0] for p in maps),margin=min(p[g][1] for p in maps))


def contrast_regions(candidates,baselines,universe,images):
    """Connected baseline/candidate differences, not tiny all-library signatures.
    A query may span several signature cells when they express the SAME bounded
    contrast. Connectivity proposes a question; it never proves object identity.
    """
    if not len(universe):return []
    edges=[]
    for im in images:
        pairs=np.concatenate((np.stack((im[:,:-1].ravel(),im[:,1:].ravel()),1),
                              np.stack((im[:-1].ravel(),im[1:].ravel()),1)))
        pairs=np.unique(pairs,axis=0);ix=np.searchsorted(universe,pairs)
        keep=(ix<len(universe)).all(1);pairs,ix=pairs[keep],ix[keep]
        keep=(universe[ix]==pairs).all(1)&(ix[:,0]!=ix[:,1]);edges.append(ix[keep])
    edge=np.concatenate(edges) if edges else np.empty((0,2),int)
    graph=coo_matrix((np.ones(len(edge)),(edge[:,0],edge[:,1])),shape=(len(universe),)*2).tocsr()
    regions={}
    for base in baselines:
        for candidate in candidates:
            # Addition and removal have different meanings, so never merge them.
            for delta in (np.setdiff1d(candidate,base),np.setdiff1d(base,candidate)):
                delta=np.intersect1d(delta,universe)
                if not len(delta):continue
                ix=np.searchsorted(universe,delta)
                _,labels=connected_components(graph[ix][:,ix],directed=False)
                for label in np.unique(labels):
                    region=delta[labels==label];regions.setdefault(region.astype('<i8').tobytes(),region)
    return [regions[k] for k in sorted(regions)]


def choose_question(member_sets,views,pairs,certified_ids,provisional_ids,*,baselines=None):
    """One relationship, at most two independently qualified existing cameras.

    views has only camera/ids/allowed/disputed. Labels and scores have no API.
    Bounded baseline/candidate differences are split by observed connectivity.
    A point question may disambiguate only its witnessed region, not a whole mask.
    """
    candidates=canonical_candidates(member_sets)
    if len(candidates)<2:return None,{'reason':'no_distinct_candidate_geometry'}
    certified=np.unique(certified_ids).astype(np.int64)
    baselines=canonical_candidates(baselines) if baselines is not None else candidates[:1]
    provisional=np.setdiff1d(provisional_ids,certified).astype(np.int64)
    best=None;rejections={};pair_count=0
    for a,b in sorted({tuple(sorted(p)) for p in pairs}):
        if a not in views or b not in views:continue
        va,vb=views[a],views[b];vs=[va,vb]
        visible=[np.unique(v['ids'][v['allowed']]) for v in vs]
        common=np.intersect1d(*visible);common=common[common>=0]
        uncertain=np.unique(np.concatenate([v['ids'][v['allowed']&v['disputed']] for v in vs]))
        uncertain=uncertain[uncertain>=0]
        if not len(uncertain):continue
        images=[np.where(v['allowed']&np.isin(v['ids'],uncertain),v['ids'],-1) for v in vs]
        regions=contrast_regions(candidates,baselines,uncertain,images)
        anchors=np.intersect1d(certified,common);status='certified'
        if not len(anchors):
            anchors=np.intersect1d(provisional,common);status='provisional_identity'
        if not len(anchors):
            rejections['no_relocatable_seed']=rejections.get('no_relocatable_seed',0)+1;continue
        anchor_maps=[positions(v['ids'],v['allowed'],anchors) for v in vs]
        anchor_options=[]
        for aid in anchors:
            ap=best_shared_point(anchor_maps,[aid])
            if ap is not None:
                anchor_options.append((int(aid),ap,np.array([aid in m for m in candidates],bool)))
        membership=np.stack([np.isin(uncertain,m) for m in candidates],axis=1)
        pixel_counts=[]
        for v in vs:
            gids,count=np.unique(v['ids'][v['allowed']],return_counts=True)
            pixel_counts.append(dict(zip(gids.tolist(),count.tolist())))
        observed_mass=max(1,min(sum(c.get(int(g),0) for g in uncertain) for c in pixel_counts))
        pair_count+=1
        for region in regions:
            # Both answers must distinguish saved geometries. Unrelated easy
            # objects, unanimous regions and candidate-count duplicates cannot win.
            cells=membership[np.searchsorted(uncertain,region)]
            include=cells.all(axis=0);exclude=~cells.any(axis=0)
            if include.all() or not include.any():continue
            coverage=min(sum(c.get(int(g),0) for g in region) for c in pixel_counts)
            bp=None
            for aid,ap,target in anchor_options:
                if aid in region:continue
                yes=int((target&include).sum());no=int((target&exclude).sum())
                if not yes or not no:continue
                # Minimax split: no assumed SAM answer probabilities, no class
                # preference, no GT-based expected gain and no area-only choice.
                split=min(yes,no)/int(target.sum())
                gain=split*coverage/observed_mass
                if best is not None and (-gain,-coverage)>best[0][:2]:continue
                if bp is None:
                    query_maps=[positions(v['ids'],v['allowed'],region) for v in vs]
                    bp=best_shared_point(query_maps,region)
                    if bp is None:break
                rank=(-gain,-coverage,-min(ap['clearance'],bp['clearance']),
                      a,b,int(aid),bp['id'],content_id(region))
                row=dict(version=VERSION,views=[a,b],a_id=int(aid),b_id=bp['id'],
                    a_xy=ap['xy'],b_xy=bp['xy'],anchor_status=status,
                    region_ids=region.tolist(),region_key=content_id(region),
                    conditional_include_count=yes,conditional_exclude_count=no,
                    mixed_region_candidate_count=int((target&~include&~exclude).sum()),
                    unanchored_candidate_count=int((~target).sum()),
                    worst_answer_split=split,common_visible_pixels=coverage,
                    worst_visible_disagreement_gain=gain,
                    include_candidates=[content_id(m) for m,t,y in zip(candidates,target,include) if t and y],
                    exclude_candidates=[content_id(m) for m,t,y in zip(candidates,target,exclude) if t and y],
                    candidate_count=len(candidates),boundary_certified=False,
                    question='Does the queried visible region belong to the instance at anchor A?',
                    counterfactual_not_observed=True)
                if best is None or rank<best[0]:best=rank,row
    return (best[1] if best else None),dict(reason=None if best else 'no_actionable_two_view_difference',
        candidate_count=len(candidates),eligible_pairs=pair_count,rejections=rejections)


def scoped_answer_masks(view,region_ids,ids,relation,anchor_status):
    """Only the inspected difference can become evidence; endpoints are not a
    certificate for all pixels inside either SAM response. Provisional identity
    questions remain diagnostic until target identity has independent support.
    """
    known=view['known'] & np.isin(ids,region_ids)
    positive=np.zeros_like(known);negative=positive.copy()
    if relation=='unresolved' or anchor_status!='certified':return positive,negative
    a=np.asarray(view['a_masks'],bool);b=np.asarray(view['b_masks'],bool)
    af=binary_erosion(a.all(axis=0),iterations=2,border_value=0)
    ab=binary_erosion(~a.any(axis=0),iterations=2,border_value=0)
    bf=binary_erosion(b.all(axis=0),iterations=2,border_value=0)
    if relation=='connect':positive=known&af&bf
    elif relation=='separate':negative=known&ab&bf
    return positive,negative
