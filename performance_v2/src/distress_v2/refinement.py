"""At most three development-only local rounds, including strong incumbents."""
import json
import numpy as np
import pandas as pd
from .common import ROOT,ART,RESULT,dump,digest,hours,now,database
from .search import execute,NN

def run():
    marker=ART/'refinement_complete.json'
    if marker.exists():return
    c=database();table=pd.read_sql_query("SELECT * FROM trial_records WHERE status='COMPLETE' AND stage IN ('screen','baseline')",c);c.close()
    tree=table[table.family.isin(['random_forest','extra_trees','catboost','xgboost'])].sort_values('macro_AP',ascending=False).iloc[0]
    neural=table[table.family.isin(NN)].sort_values('macro_AP',ascending=False).iloc[0]
    plans=[]
    def add(round_number,parent,change,reason):
        original=json.loads(parent.config_json);config={**original,**change}
        if digest(config)==digest(original) or any(digest(p['config'])==digest(config) for p in plans):return
        plans.append(dict(round=round_number,parent=parent.trial_id,reason=reason,config=config,estimated_seconds=max(5.,parent.seconds)*(config.get('iterations',1)/json.loads(parent.config_json).get('iterations',1) if config['family'] in ['random_forest','extra_trees'] else 1.5)))
    # The strongest existing core is eligible; S1 is neither forced nor excluded.
    if tree.family in ['random_forest','extra_trees']:
        for change in [dict(iterations=600),dict(iterations=600,leaf=5),dict(iterations=600,depth=10,leaf=5),dict(iterations=600,depth=16,max_features=.5),dict(iterations=600,depth=None,leaf=5),dict(iterations=1000,leaf=10)]:
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
    dump(ROOT/'configs/local_refinement_plan.json',dict(registered_before_refinement_at_utc=now(),rounds_max=3,deadline_hours=5.45,plan=plans,history_and_confirmation_not_opened=True))
    records=[]
    for i,p in enumerate(plans):
        name=f'refine_r{p["round"]}_{i:02d}_{digest(p["config"])[:8]}'
        if hours()+p['estimated_seconds']/3600>5.45:
            records.append(dict(trial_id=name,round=p['round'],parent=p['parent'],status='NOT_STARTED_BUDGET',reason=p['reason']));continue
        status,value=execute(p['config'],name,'refinement',[2009,2011,2013])
        records.append(dict(trial_id=name,round=p['round'],parent=p['parent'],status=status,macro_AP=value,reason=p['reason']))
    pd.DataFrame(records).to_csv(RESULT/'local_refinement_trace.csv',index=False)
    dump(marker,dict(completed_at_utc=now(),executed=sum(x['status']!='NOT_STARTED_BUDGET' for x in records),not_started=sum(x['status']=='NOT_STARTED_BUDGET' for x in records)))
