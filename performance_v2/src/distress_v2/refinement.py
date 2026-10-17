"""At most three development-only local rounds, including strong incumbents."""
import json
import numpy as np
import pandas as pd
from .common import ROOT,ART,RESULT,dump,digest,hours,now,database
from .search import execute,NN,base

def run():
    marker=ART/'refinement_complete.json'
    if marker.exists():return
    c=database();table=pd.read_sql_query("SELECT * FROM trial_records WHERE status='COMPLETE' AND stage IN ('screen','baseline')",c);c.close()
    tree=table[table.family.isin(['random_forest','extra_trees','catboost','xgboost'])].sort_values('macro_AP',ascending=False).iloc[0]
    neural=table[table.family.isin(NN)].sort_values('macro_AP',ascending=False).iloc[0]
    folds=pd.read_csv(RESULT/'development_by_year.csv')
    mean_cost=folds.groupby('candidate').fit_seconds.mean()
    fixed=['baseline_legacy_strong_inner_l1','baseline_v1_lags_xgb','baseline_v1_raw_catboost','baseline_v1_selected_rf']
    mandatory=list(dict.fromkeys([*fixed,tree.trial_id,neural.trial_id]))
    stability_seconds=sum(12*mean_cost.get(k,60.) for k in mandatory)
    final_seconds=1.3*(10*mean_cost.get(neural.trial_id,120.)+10*mean_cost.get(tree.trial_id,30.)+sum(7*mean_cost.get(k,30.) for k in fixed))
    reserve_hours=(stability_seconds+final_seconds)/3600+.55
    deadline=min(5.45,8-reserve_hours)
    plans=[]
    def add(round_number,parent,change,reason):
        original=json.loads(parent.config_json);config={**original,**change}
        if digest(config)==digest(original) or any(digest(p['config'])==digest(config) for p in plans):return
        plans.append(dict(round=round_number,parent=parent.trial_id,reason=reason,config=config,estimated_seconds=max(5.,parent.seconds)*(config.get('iterations',1)/json.loads(parent.config_json).get('iterations',1) if config['family'] in ['random_forest','extra_trees'] else 1.5)))
    if json.loads(tree.config_json)['features']=='V1_S1':
        # The incumbent's input is available to competing model families too.
        # This is a development-only pipeline repair, not a causal feature claim.
        for family in ['catboost','xgboost']:
            for depth in [3,5]:
                c=base(family,features='V1_S1',preprocess='v1',clip=True,iterations=3000,depth=depth,learning_rate=.03,weight_alpha=0)
                if family=='catboost':c.update(l2_leaf_reg=3.,random_strength=.5,bootstrap_type='Bayesian',bagging_temperature=1.)
                else:c.update(min_child_weight=4.,subsample=.9,colsample_bytree=.9,reg_lambda=3.,reg_alpha=0.)
                plans.append(dict(round=1,parent=tree.trial_id,reason='Give the development-leading incumbent input/preprocessing to competing boosting families; multiple-factor pipeline comparison',config=c,estimated_seconds=90.))
    # The strongest existing core is eligible; S1 is neither forced nor excluded.
    if tree.family in ['random_forest','extra_trees']:
        for change in [dict(iterations=600),dict(iterations=600,leaf=5),dict(iterations=600,depth=10,leaf=5),dict(iterations=1000,leaf=10)]:
            add(1,tree,change,'Capacity/regularization repair around the strongest development-only tree incumbent; multivariable variants are pipeline comparisons')
    else:
        for change in [dict(learning_rate=.02),dict(learning_rate=.08),dict(depth=max(2,int(json.loads(tree.config_json)['depth'])-1)),dict(weight_alpha=.5),dict(features='LEGACY33'),dict(features='SLOG_RATIO')]:
            add(1,tree,change,'Bounded development-only neighborhood of strongest tree')
    ncfg=json.loads(neural.config_json)
    for change in [dict(schedule='cosine' if ncfg.get('schedule','fixed')=='fixed' else 'fixed'),dict(learning_rate=max(1e-4,ncfg.get('learning_rate',.001)/2)),dict(dropout=.15),dict(dropout=.4)]:
        add(2,neural,change,'Same-input neural optimization/regularization neighborhood; inspect saved train/stop curves')
    for parent in [tree,neural]:
        for window in [6,10]:add(3,parent,dict(window=window),'One-factor recent-window sensitivity')
        if json.loads(parent.config_json).get('sampling','none')=='none':
            for half_life in [3,5]:add(3,parent,dict(half_life=half_life),'One-factor train-role time-decay sensitivity')
            for beta in [.5,1]:add(3,parent,dict(company_beta=beta),'One-factor train-role company-weight sensitivity')
    if ncfg.get('sampling','none')=='none':
        for value in [.05,.2]:add(3,neural,dict(pair_lambda=value),'Within-year limited pairwise proxy, added to established BCE backbone')
    curves=[]
    for path in (ART/'logs').glob(neural.trial_id+'__seed42__year*__stop_curve.csv'):
        d=pd.read_csv(path);best=d.loc[d.stop_AP.idxmax()]
        curves.append(dict(curve=str(path.relative_to(ROOT)),best_stop_AP=float(best.stop_AP),train_probe_AP_at_best=float(best.train_probe_AP),best_epoch=int(best.epoch),executed_epochs=int(d.epoch.max()),interpretation='Probe/stop gap and cap contact are optimization clues, not a causal fit diagnosis'))
    dump(ROOT/'configs/local_refinement_plan.json',dict(registered_before_refinement_at_utc=now(),rounds_max=3,deadline_hours=deadline,maximum_deadline_hours=5.45,cost_projected_remaining_hours=reserve_hours,neural_curve_evidence=curves,plan=plans,history_and_confirmation_not_opened=True))
    records=[]
    for i,p in enumerate(plans):
        name=f'refine_r{p["round"]}_{i:02d}_{digest(p["config"])[:8]}'
        if hours()+p['estimated_seconds']/3600>deadline:
            records.append(dict(trial_id=name,round=p['round'],parent=p['parent'],status='NOT_STARTED_BUDGET',reason=p['reason']));continue
        status,value=execute(p['config'],name,'refinement',[2009,2011,2013])
        records.append(dict(trial_id=name,round=p['round'],parent=p['parent'],status=status,macro_AP=value,reason=p['reason']))
    pd.DataFrame(records).to_csv(RESULT/'local_refinement_trace.csv',index=False)
    dump(marker,dict(completed_at_utc=now(),executed=sum(x['status']!='NOT_STARTED_BUDGET' for x in records),not_started=sum(x['status']=='NOT_STARTED_BUDGET' for x in records)))
