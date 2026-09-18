"""CPU-only delivery tables after all four automatic predictions are frozen."""
from __future__ import annotations
import argparse
import csv
import html
from pathlib import Path
import numpy as np
from run_effective_repair import read, save, slug
from run_effective_repair_evaluation import write_csv

TAIL = [('scene0025_01','15:32','fan',28),('scene0025_01','17:28','cup',23),
        ('scene0025_01','17:29','cup',109),('scene0025_01','18:27','phone',122),
        ('scene0645_00','18:13','phone',41)]


def csv_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def collect(root):
    entries, objects, banks = {}, [], {}
    for mode,prefix in [('category','C'),('global','G')]:
        output=root/mode
        assert read(output/'scene-result.json')['execution_complete']
        result=read(output/'evaluation.json')
        names={'initial':prefix+'-initial','matched-open':prefix+'-open','feedback':prefix+'-feedback',
               'matched-open-legacy-assembly':prefix+'-open-old-assembly',
               'feedback-legacy-assembly':prefix+'-feedback-old-assembly'}
        if mode=='category':
            entries.update(B0=result['b0'],legacy_C_open=result['conditions']['legacy_C_open'])
            names.update(B0='B0',legacy_C_open='legacy_C_open')
        entries.update({names[k]:v for k,v in result['conditions'].items() if k in names})
        objects += [dict(r,condition=names[r['condition']]) for r in csv_rows(output/'object_iou.csv') if r['condition'] in names]
        for protocol in ('matched-open','feedback'):
            banks[names[protocol]]={(r['comparison'],r['scene_id'],r['gt_id']):r
                for r in csv_rows(output/(protocol+'-bank-object-iou.csv'))}
    scenes=sorted({r['scene_id'] for r in objects})
    metrics=[]
    for name,entry in entries.items():
        for subgroup in ('overall','tail'):
            for scene in ('pooled',*scenes):
                rows=[r for r in objects if r['condition']==name and (scene=='pooled' or r['scene_id']==scene)
                      and (subgroup=='overall' or (r['scene_id'],r['gt_id']) in {(t[0],t[1]) for t in TAIL})]
                count=sum(r['matching']=='class_aware' for r in rows)
                if scene=='pooled':
                    assert count==(38 if subgroup=='overall' else 5)
                row=dict(condition=name,stratum=subgroup,scene_id=scene,gt_count=count)
                for matching in ('class_aware','class_agnostic'):
                    values=[float(r['iou']) for r in rows if r['matching']==matching]
                    row['mean_3d_iou_'+matching]=float(np.mean(values)) if values else None
                if subgroup=='overall':
                    ap=(entry['official_9'] if scene=='pooled' else entry['per_scene'][scene]['official_9'])['aggregate']
                    row.update(AP25=ap['map_0.25'],AP50=ap['map_0.50'],mAP=ap['map_50_90'])
                for threshold in ('0.25','0.50'):
                    block=entry['reconciliation']['iou_'+threshold]
                    counts=block['aggregate']['strata'][subgroup]['pooled_counts'] if scene=='pooled' else next(
                        r['strata'][subgroup] for r in block['per_scene'] if r['scene_id']==scene)
                    for key in ('rescued','lost','new_false_positives','duplicate_predictions','final_prediction_count'):
                        row[key+'@'+threshold]=counts[key]
                metrics.append(row)
    write_csv(root/'complete-results.csv',metrics)
    lookup={(r['condition'],r['matching'],r['scene_id'],r['gt_id']):r for r in objects}
    paired=[]
    comparisons=[('C-open','G-open'),('C-feedback','G-feedback'),('C-feedback','C-open'),('G-feedback','G-open')]
    comparisons += [(condition, reference) for condition in ('C-open','G-open','C-feedback','G-feedback')
                    for reference in ('B0','legacy_C_open')]
    for original in [r for r in objects if r['condition']=='B0']:
        key=(original['matching'],original['scene_id'],original['gt_id'])
        row=dict(matching=key[0],scene_id=key[1],gt_id=key[2],class_name=original['class'])
        row.update({name:float(lookup[(name,*key)]['iou']) for name in entries})
        for new,reference in comparisons:
            row[new+'_minus_'+reference]=row[new]-row[reference]
        for name in banks:
            for kind in ('selected','oracle'):
                row[('posthoc_' if kind=='oracle' else 'pre_assembly_')+kind+'_'+name]=float(
                    banks[name][(kind+'_'+key[0],key[1],key[2])]['iou'])
        paired.append(row)
    write_csv(root/'all-38-gt.csv',paired)
    write_csv(root/'tail-five-gt.csv',[r for r in paired if (r['scene_id'],r['gt_id']) in {(t[0],t[1]) for t in TAIL}])
    changes=[]
    for r in paired:
        if r['matching']!='class_aware': continue
        for new,reference in comparisons:
            delta=r[new]-r[reference]
            if abs(delta)>.10:
                changes.append(dict(scene_id=r['scene_id'],gt_id=r['gt_id'],class_name=r['class_name'],
                    comparison=new+' minus '+reference,delta=delta,direction='gain' if delta>0 else 'decline'))
    write_csv(root/'all-changes-over-10pp.csv',changes)
    sources=[]
    for mode,prefix in [('category','C'),('global','G')]:
        for sid in scenes:
            panels=read(root/mode/'scenes'/sid/'panels.json')
            other=read(root/('global' if mode=='category' else 'category')/'scenes'/sid/'panels.json')
            assert {u:p['construction'] for u,p in panels.items()}=={u:p['construction'] for u,p in other.items()}
            for protocol,label in [('matched-open','open'),('feedback','feedback')]:
                assembly=read(root/mode/'scenes'/sid/protocol/'summary.json')
                ownership={r['uid']:r for r in assembly['ownership']}
                with np.load(root/mode/'scenes'/sid/protocol/'actual-members.npz',allow_pickle=False) as z:
                    actual_counts={u:len(z[u]) for u in z.files}
                paths=sorted((root/mode/'scenes'/sid/('banks-'+protocol)).glob('*/bank.json'))
                assert len(paths)==(209 if sid=='scene0025_01' else 165)
                for path in paths:
                    bank=read(path); selected=next(r for r in bank['candidates'] if r['id']==bank['selected_id'])
                    uid=bank['uid']; panel=panels[uid]; own=ownership.get(uid,{})
                    sources.append(dict(condition=prefix+'-'+label,scene_id=sid,source_uid=uid,
                        selected_id=selected['id'],selected_class=selected['semantics']['class'],
                        candidate_count=len(bank['candidates']),selected_members=selected['member_count'],
                        actual_members=actual_counts.get(uid,0),own_instance_members=own.get('final_members',0),
                        suppressed_by=assembly['suppressed'].get(uid),
                        hypotheses=';'.join(dict.fromkeys(p['hypothesis'] for p in bank['slots'])),
                        prior_sources=';'.join(dict.fromkeys(p['source'] for p in bank['slots'])),
                        active_category_specific_slots=sum(p['category_specific'] for p in bank['slots']),
                        feedback_applicable=bank['feedback_applicable'],actual_initial_count=bank['actual_initial_count'],
                        reliable_written_core_count=bank['reliable_written_core_count'],
                        view_changed=panel['changed'],
                        continuation_status=panel.get('continuation_status', 'not_recorded_in_previous_cache'),
                        construction=';'.join(panel['construction']),score=selected['quality']['score'],
                        valid_view_count=selected['quality']['valid_view_count'],core_count=selected['core_count']))
    assert len(sources)==374*4
    write_csv(root/'all-374-inputs-four-conditions.csv',sources)
    tail_chain=[]
    for sid,gt,category,source in TAIL:
        uid=f'{sid}:C0:{source:06d}'
        for mode,prefix in [('category','C'),('global','G')]:
            for protocol,label in [('matched-open','open'),('feedback','feedback')]:
                path=root/mode/'scenes'/sid/('banks-'+protocol)/slug(uid)/'bank.json'
                bank=read(path); row=next(r for r in bank['candidates'] if r['id']==bank['selected_id'])
                initial_bank=read(root/mode/'scenes'/sid/'banks-initial'/slug(uid)/'bank.json')
                source_row=next(r for r in sources if r['condition']==prefix+'-'+label and r['source_uid']==uid)
                actual_crops=[]
                for phase,directory in [('initial','banks-initial'),('later','banks-'+protocol)]:
                    base=root/mode/'scenes'/sid/directory/slug(uid)/'observations'
                    for group_file in sorted(base.glob('*/*/group.json')):
                        group=read(group_file); actual=group.get('actual_input',{})
                        prior=actual.get('prior',{}); crop=actual.get('crop',{})
                        control_file=root/'global'/'scenes'/sid/directory/slug(uid)/'observations'/group_file.parent.parent.name/group_file.parent.name/'group.json'
                        control=read(control_file).get('actual_input',{}) if control_file.exists() else {}
                        control_crop=control.get('crop',{})
                        actual_crops.append(dict(phase=phase,branch=group_file.parent.parent.name,
                            camera=group_file.parent.name,status=group.get('status'),hypothesis=prior.get('hypothesis'),
                            prior_source=prior.get('source'),correct_class_prior=bool(prior.get('category_specific') and prior.get('hypothesis')==category),
                            actual_width=crop.get('width'),actual_height=crop.get('height'),
                            requested_side=crop.get('requested_side'),center=actual.get('location',{}).get('center_image_xy'),
                            positive_points=actual.get('positive_points_image'),
                            window_differs_from_global=bool(crop and control_crop and any(crop[k]!=control_crop[k] for k in ('left','top','width','height')))))
                tail_chain.append(dict(gt_id=gt,target_class=category,**source_row,
                    correct_class_in_top2=category in {p['hypothesis'] for p in bank['slots']},
                    correct_class_in_initial_top2=category in {p['hypothesis'] for p in initial_bank['slots']},
                    correct_active_prior_slots=sum(p['category_specific'] and p['hypothesis']==category for p in bank['slots']),
                    selected_kind=row['kind'],selected_branch=row['branch'],
                    attribution_note='Active statistics and class coverage are not evidence of causal benefit; compare actual crops and paired final 3D IoU.',
                    actual_crops=actual_crops,
                    added_region_reasons=read(path.parent/'added-region-reasons.json')))
    save(root/'tail-five-failure-chains.json',tail_chain)
    summary=dict(gt_count=38,tail_count=5,tail_classes=['cup','phone','fan'],source_count=374,
        paired_comparisons={new+'_minus_'+ref:dict(
            mean_iou_delta=float(np.mean([r[new]-r[ref] for r in paired if r['matching']=='class_aware'])),
            tail_delta=float(np.mean([r[new]-r[ref] for r in paired if r['matching']=='class_aware' and (r['scene_id'],r['gt_id']) in {(t[0],t[1]) for t in TAIL}])),
            gains_over_10pp=[r for r in changes if r['comparison']==new+' minus '+ref and r['direction']=='gain'],
            declines_over_10pp=[r for r in changes if r['comparison']==new+' minus '+ref and r['direction']=='decline']) for new,ref in comparisons},
        views_are_paired=True,oracle_is_posthoc=True,evidence_scope='Fixed DEV2 development evidence; five tail objects, three tail classes')
    save(root/'scientific-summary.json',summary)
    rows=['<!doctype html><meta charset="utf-8"><title>DEV2 multiview repair</title>',
          '<style>body{font:16px sans-serif;margin:24px;max-width:1800px}img{width:100%}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}</style>',
          '<h1>DEV2 配对实验</h1><p>固定 374 输入、38 GT、5 个长尾对象。候选上界为事后诊断，不计入自动成绩。</p>',
          '<p><a href="complete-results.csv">完整指标</a> · <a href="all-38-gt.csv">所有 GT 的改善和退化</a> · <a href="tail-five-gt.csv">五个长尾对象</a> · <a href="all-374-inputs-four-conditions.csv">全部输入</a></p>',
          '<table><tr><th>条件</th><th>整体 IoU</th><th>长尾 IoU</th><th>AP50</th></tr>']
    for name in ('B0','legacy_C_open','G-open','C-open','G-feedback','C-feedback'):
        overall=next(r for r in metrics if r['condition']==name and r['scene_id']=='pooled' and r['stratum']=='overall')
        tail=next(r for r in metrics if r['condition']==name and r['scene_id']=='pooled' and r['stratum']=='tail')
        rows.append(f'<tr><td>{name}</td><td>{100*overall["mean_3d_iou_class_aware"]:.2f}%</td><td>{100*tail["mean_3d_iou_class_aware"]:.2f}%</td><td>{100*overall["AP50"]:.2f}%</td></tr>')
    rows.append('</table>')
    for mode in ('category','global'):
        rows.append(f'<h2>{mode}: 全部固定诊断对象</h2>')
        for path in sorted((root/mode/'local').glob('*/comparison-2.png')):
            rows.append(f'<p>{html.escape(path.parent.name)}</p><img src="{path.relative_to(root).as_posix()}">')
        for sid,gt,category,source in TAIL:
            uid=f'{sid}:C0:{source:06d}'
            rows.append(f'<h2>{mode} {sid} {category} {gt}</h2>')
            for protocol in ('matched-open','feedback'):
                path=root/mode/'scenes'/sid/protocol/'images'/(slug(uid)+'.jpg')
                rows.append(f'<p>{protocol}</p><img src="{path.relative_to(root).as_posix()}">')
    (root/'index.html').write_text('\n'.join(rows),encoding='utf8')
    print(summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    collect(parser.parse_args().output)
