"""One bounded scene allocation and alternative check for common geometry."""
from __future__ import annotations

import numpy as np

from .common_geometry_selector import VERSION, TOL, select_geometry, semantic_state
from .multiview_repair import member_components
from .prediction_contract import normalize_prediction


def assemble_common(rt, banks, dest, *, baseline=None, excluded_views=None):
    from run_effective_repair import read, save
    from .multiview_repair_experiment import ids_file
    if (dest / 'summary.json').exists():
        return read(dest / 'scene.json')
    def canonical(uid):
        # Frozen local B0 sources use scene:B0:000006; scene exports use B0:6.
        # These are the same instance, not two competing objects.
        return 'B0:' + str(int(uid.rsplit(':', 1)[1])) if ':B0:' in uid else uid
    input_aliases = {u: canonical(u) for u in banks if u != canonical(u)}
    banks = {canonical(u): b for u, b in banks.items()}
    baseline = baseline or rt.b0
    labels0 = np.asarray(baseline['point_labels'], np.int64)
    n = len(labels0)
    old = {m.get('object_uid', 'B0:'+str(k)): (np.flatnonzero(labels0 == int(k)), m)
           for k, m in baseline['instances'].items()}
    raw = {u: [(dict(id='committed', kind='incumbent'), m)] for u, (m, _) in old.items()}
    for uid, bank in banks.items():
        raw.setdefault(uid, []).extend((r, ids_file(r['members_file'])) for r in bank['candidates'])
    # Original whole B0 survives the initial pass and feedback bank transitions.
    if baseline is not rt.b0:
        original = np.asarray(rt.b0['point_labels'])
        for key, meta in rt.b0['instances'].items():
            uid = meta.get('object_uid', 'B0:'+str(key))
            raw.setdefault(uid, []).append((dict(id='original-B0', kind='original'),
                                           np.flatnonzero(original == int(key))))
    components = member_components([(u, m) for u, rows in raw.items() for _, m in rows], n)
    forbidden = {canonical(u): set(v) for u, v in (excluded_views or {}).items()}
    for case in rt.plan.get('local_cases', []):
        forbidden.setdefault(canonical(case['candidate_uid']), set()).update(v['camera_uid'] for v in case['views'][2:])
    context = {}
    for component in components:
        excluded = set().union(*(forbidden.get(u, set()) for u in component))
        for uid in component:
            # Neighbour observations do not become this object's foreground.
            bank = banks.get(uid)
            if bank:
                views = [v for v in bank['views'] if v not in excluded]
                pool = [p for p in bank['reference_pool'] if rt.descriptor(p)['camera'] in views]
            else:
                pool, views = [], []
            context[uid] = pool, views
    scored = {}
    def score(uid, members, template):
        members = np.unique(members).astype(np.int64)
        key = uid, members.tobytes()
        if key not in scored:
            scored[key] = rt.score_evidence(members, *context[uid])
        sem, quality, core, negative, refs = scored[key]
        return dict(template, uid=uid, members=members, member_count=len(members),
            semantics=sem, quality=quality, core_ids=core, negative_ids=negative, refs=refs)
    choices, libraries, before, selection = {}, {}, {}, {}
    for uid, entries in raw.items():
        library = [score(uid, members, row) for row, members in entries]
        libraries[uid] = library
        if uid in old:
            before[uid] = library[0]
        nonempty = [r for r in library if r['member_count'] >= 3]
        if not nonempty:
            continue
        winner, trace = select_geometry(nonempty, {r['id']: r['members'] for r in nonempty},
                                        'committed' if uid in old else None)
        choices[uid], selection[uid] = winner, trace

    def sortkey(row):
        return (-row['quality']['score'], row['id'] != 'committed',
                not row['uid'].startswith('B0:'), row['members'].tobytes())

    # An identity merge needs independent shared positive surfaces. Category names
    # are deliberately absent, including when one instance is currently unknown.
    def same_identity(a, b):
        intersection = np.intersect1d(a['members'], b['members'])
        if len(intersection)/max(1, min(len(a['members']), len(b['members']))) < .9:
            if len(intersection)/max(1, len(np.union1d(a['members'], b['members']))) <= .5:
                return False
        shared = np.intersect1d(a['core_ids'], b['core_ids'])
        if not len(shared):
            return False
        # Distinct protected surfaces must not be swallowed just because a large
        # mask covers both. A duplicate explains most of the smaller core.
        if len(shared)/max(1, min(len(a['core_ids']), len(b['core_ids']))) < .9:
            return False
        cameras = set(r['camera'] for r in a['refs']) & set(r['camera'] for r in b['refs'])
        return bool(rt.pairs(sorted(cameras), shared))

    kept, aliases, transfers = {}, {}, []
    for row in sorted(choices.values(), key=sortkey):
        duplicate = next((u for u, other in kept.items() if same_identity(row, other)), None)
        if duplicate is None:
            kept[row['uid']] = row
            continue
        aliases[row['uid']] = duplicate
        other = kept[duplicate]
        additions = np.setdiff1d(row['core_ids'], np.union1d(other['negative_ids'], row['negative_ids']))
        for uid, rival in choices.items():
            if uid not in (row['uid'], duplicate) and not same_identity(row, rival):
                additions = np.setdiff1d(additions, rival['core_ids'])
        additions = np.setdiff1d(additions, other['members'])
        if len(additions):
            kept[duplicate] = score(duplicate, np.union1d(other['members'], additions), other)
            transfers.append(dict(source=row['uid'], target=duplicate, members=additions.tolist()))

    owner = np.full(n, -1, np.int32)
    ordered = sorted(kept.values(), key=sortkey)
    core_count = np.zeros(n, np.int32)
    contender_count = np.zeros(n, np.int32)
    all_views = sorted({r['camera'] for row in ordered for r in row['refs']})
    camera_counts = np.zeros((len(all_views), n), np.int16)
    for row in ordered:
        core_count[np.setdiff1d(row['core_ids'], row['negative_ids'])] += 1
        contender_count[row['members']] += 1
        for ref in row['refs']:
            camera_counts[all_views.index(ref['camera']), row['members']] += 1
    ranks = np.full(n, -1, np.int8)
    votes_best = np.full(n, -100, np.int16)
    scores_best = np.full(n, -1., np.float64)
    for index, row in enumerate(ordered):
        members = row['members']
        votes = np.zeros(len(members), np.int16)
        for ref in row['refs']:
            common = camera_counts[all_views.index(ref['camera']), members] == contender_count[members]
            votes += (common & np.isin(members, ref['positive_ids'])).astype(np.int16)
            votes -= (common & np.isin(members, ref['negative_ids'])).astype(np.int16)
        protected = 2 * (np.isin(members, row['core_ids']) & (core_count[members] == 1)
                     & ~np.isin(members, row['negative_ids'])).astype(np.int8)
        value = row['quality']['score']
        win = (protected > ranks[members]) | ((protected == ranks[members]) &
              ((votes > votes_best[members]) | ((votes == votes_best[members]) & (value > scores_best[members]))))
        owner[members[win]] = index
        ranks[members[win]], votes_best[members[win]], scores_best[members[win]] = protected[win], votes[win], value
    after = {}
    for index, row in enumerate(ordered):
        assigned = np.flatnonzero(owner == index)
        if len(assigned) >= 3:
            after[row['uid']] = score(row['uid'], assigned, row)

    def retained(uid, state):
        target = uid if uid in state else aliases.get(uid)
        return state[target]['members'] if target in state else np.empty(0, np.int64)

    def check(component, state):
        old_score = sum(before[u]['quality']['score'] for u in component if u in before)
        new_score = sum(state[u]['quality']['score'] for u in component if u in state)
        loss = []
        for uid in component:
            if uid in before:
                row = before[uid]
                removed = np.setdiff1d(row['core_ids'], retained(uid, state))
                unsupported = np.setdiff1d(removed, row['negative_ids'])
                if len(unsupported):
                    loss.append(dict(uid=uid, members=unsupported.tolist()))
        return dict(before_mean=old_score/len(component), after_mean=new_score/len(component),
                    core_loss=loss, valid=new_score >= old_score-TOL and not loss)

    checks, replacements = [], []
    for component in components:
        initial_check = check(component, after)
        # Freeze all other objects while trying each saved alternative once.
        failed = [u for u in component if u in choices and u not in aliases and
                  (u not in after or len(np.setdiff1d(choices[u]['core_ids'], retained(u, after)))
                   or not initial_check['valid'])]
        for uid in sorted(failed):
            occupied = np.zeros(n, bool)
            for other, row in after.items():
                if other != uid:
                    occupied[row['members']] = True
            trials, attempted = [], []
            for row in libraries[uid]:
                available = row['members'][~occupied[row['members']]]
                if len(available) < 3:
                    attempted.append(dict(id=row['id'], reason='fewer_than_three_after_ownership'))
                    continue
                candidate = score(uid, available, row)
                lost = np.setdiff1d(row['core_ids'], available)
                if len(np.setdiff1d(lost, row['negative_ids'])):
                    attempted.append(dict(id=row['id'], reason='independent_core_unavailable'))
                    continue
                test = dict(after, **{uid: candidate})
                verdict = check(component, test)
                attempted.append(dict(id=row['id'], reason='eligible' if verdict['valid'] else 'component_evidence_loss'))
                if verdict['valid']:
                    trials.append(candidate)
            if trials:
                winner, _ = select_geometry(trials, {r['id']: r['members'] for r in trials},
                                            after[uid]['id'] if uid in after else None)
                after[uid] = winner
                replacements.append(dict(uid=uid, selected=winner['id'], attempts=attempted))
            else:
                replacements.append(dict(uid=uid, selected=None, attempts=attempted))
        final_check = check(component, after)
        rollback = not final_check['valid']
        if rollback:
            for uid in component:
                after.pop(uid, None)
                if uid in before:
                    after[uid] = before[uid]
                aliases.pop(uid, None)
        checks.append(dict(uids=component, initial=initial_check, final=final_check, rollback=rollback))

    labels = np.full(n, -1, np.int64)
    metadata, ownership = {}, []
    for index, (uid, row) in enumerate(sorted(after.items())):
        members = row['members']
        if np.any(labels[members] >= 0):
            raise AssertionError('bounded assembly produced overlapping members')
        labels[members] = index
        semantics = row['semantics']
        label, status = semantic_state(semantics, rt.assets.saga20)
        # An untouched incumbent with no observations keeps its committed label;
        # no new or changed geometry inherits an old class.
        if not row['quality']['valid_view_count'] and uid in old and np.array_equal(members, old[uid][0]):
            label = old[uid][1]['class']
            status = 'recognized' if label in rt.assets.saga20 else 'out_of_eval_vocabulary'
        metadata[index] = dict(**{'class': label}, classification_status=status,
            semantics=semantics, score=row['quality']['score'], object_uid=uid,
            point_count=len(members), score_source=VERSION, selected_candidate=row['id'],
            semantic_views=[r['camera'] for r in row['refs']])
        if semantics.get('status') == 'pending_model':
            metadata[index]['classification_status'] = 'pending_model'
        source = choices[uid]['members'] if uid in choices else members
        ownership.append(dict(uid=uid, selected_candidate=row['id'], final_members=len(members),
                              lost_ids=np.setdiff1d(source, members).tolist()))
    contracted = normalize_prediction(labels, metadata)
    payload = dict(point_labels=contracted.point_labels.tolist(), instances=contracted.instances,
                   prediction_contract=contracted.audit, repair_policy=VERSION,
                   instance_aliases={u:v for u,v in aliases.items() if v in after and u not in after})
    payload['instance_aliases'].update({u: payload['instance_aliases'].get(v,v)
                                      for u,v in input_aliases.items()})
    payload['classification_complete'] = not any(m.get('classification_status') == 'pending_model'
                                                 for m in contracted.instances.values())
    save(dest / 'scene.json', payload, compact=True)
    # Vocabulary filtering occurs only in the standard export, never in ownership.
    allowed = {k:v for k,v in contracted.instances.items() if v['class'] in rt.assets.saga20}
    exported = normalize_prediction(contracted.point_labels, allowed)
    if payload['classification_complete']:
        save(dest / 'scene-evaluation.json', dict(point_labels=exported.point_labels.tolist(),
             instances=exported.instances, prediction_contract=exported.audit,
             geometry_source=str(dest/'scene.json')), compact=True)
    actual = {u: rt.actual_for(payload, u) for u in raw}
    actual.update({u: rt.actual_for(payload, u) for u in input_aliases})
    np.savez_compressed(dest / 'actual-members.npz', **actual)
    save(dest / 'summary.json', dict(assembly=VERSION, selection=selection,
         duplicate_core_transfers=transfers, replacements=replacements, conflicts=checks,
         ownership=ownership, suppressed=payload['instance_aliases'], execution_complete=True))
    return payload
