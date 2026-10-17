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
    folds=pd.read_csv(RESULT/'development_by_year.csv')
    from .budget import affordable_pair
    tree_id,nn_id,feasibility=affordable_pair(table,folds,hours())
    if tree_id is None or nn_id is None:
        dump(marker,dict(completed_at_utc=now(),executed=0,not_started='Budget feasibility failed; preserve registered seeds',feasibility=feasibility));return
    tree=table[table.trial_id==tree_id].iloc[0];neural=table[table.trial_id==nn_id].iloc[0]
    mean_cost=folds.groupby('candidate').fit_seconds.mean()
    fixed=['baseline_legacy_strong_inner_l1','baseline_v1_lags_xgb','baseline_v1_raw_catboost','baseline_v1_selected_rf']
    mandatory=list(dict.fromkeys([*fixed,tree.trial_id,neural.trial_id]))
    stability_seconds=sum(12*mean_cost.get(k,60.) for k in mandatory)
    final_seconds=1.3*(10*mean_cost.get(neural.trial_id,120.)+10*mean_cost.get(tree.trial_id,30.)+sum(7*mean_cost.get(k,30.) for k in fixed))
    reserve_hours=(stability_seconds+final_seconds)/3600+.55
    deadline=min(5.45,7.15-feasibility['projected_required_seconds']/3600-.10)
    plans=[]
    def add(round_number,parent,change,reason):
        original=json.loads(parent.config_json);config={**original,**change}
        if digest(config)==digest(original) or any(digest(p['config'])==digest(config) for p in plans):return
        plans.append(dict(round=round_number,parent=parent.trial_id,reason=reason,config=config,estimated_seconds=max(5.,parent.seconds)*(config.get('iterations',1)/json.loads(parent.config_json).get('iterations',1) if config['family'] in ['random_forest','extra_trees'] else 1.5)))
    tree_config=json.loads(tree.config_json)
    if tree.family in ['catboost','xgboost','random_forest','extra_trees']:
        # Transfer the observed incumbent input to competing boosting families;
        # the same financial information is not reserved for one algorithm.
        for family in [f for f in ['catboost','xgboost'] if f!=tree.family]:
            for depth in [3,5]:
                c=base(family,features=tree_config['features'],preprocess=tree_config['preprocess'],iterations=3000,depth=depth,learning_rate=.03,weight_alpha=0)
                if 'clip' in tree_config:c['clip']=tree_config['clip']
                if family=='catboost':c.update(l2_leaf_reg=3.,random_strength=.5,bootstrap_type='Bayesian',bagging_temperature=1.)
                else:c.update(min_child_weight=4.,subsample=.9,colsample_bytree=.9,reg_lambda=3.,reg_alpha=0.)
                plans.append(dict(round=1,parent=tree.trial_id,reason='Give the development-leading incumbent input/preprocessing to competing boosting families; multiple-factor pipeline comparison',config=c,estimated_seconds=90.))
    # The strongest existing core is eligible; S1 is neither forced nor excluded.
    if tree.family in ['random_forest','extra_trees']:
        for change in [dict(iterations=600),dict(iterations=600,leaf=5),dict(iterations=600,depth=10,leaf=5),dict(iterations=1000,leaf=10)]:
            add(1,tree,change,'Capacity/regularization repair around the strongest development-only tree incumbent; multivariable variants are pipeline comparisons')
    else:
        for change in [dict(learning_rate=.02),dict(learning_rate=.08),dict(depth=max(2,int(json.loads(tree.config_json)['depth'])-1)),dict(weight_alpha=.5),*[dict(features=f) for f in ['RAW17','RATIO_CORE','LAGS_RAW3','LEGACY33','SLOG_RATIO']]]:
            if tree.family=='xgb_rank' and 'weight_alpha' in change:continue
            add(1,tree,change,'Bounded development-only neighborhood of strongest tree')
    ncfg=json.loads(neural.config_json)
    for change in [dict(schedule='cosine' if ncfg.get('schedule','fixed')=='fixed' else 'fixed'),dict(learning_rate=max(1e-4,ncfg.get('learning_rate',.001)/2)),dict(dropout=.15),dict(dropout=.4)]:
        add(2,neural,change,'Same-input neural optimization/regularization neighborhood; inspect saved train/stop curves')
    for parent in [tree,neural]:
        for window in [6,10]:add(3,parent,dict(window=window),'One-factor recent-window sensitivity')
        if json.loads(parent.config_json).get('sampling','none')=='none' and parent.family!='xgb_rank':
            for half_life in [3,5]:add(3,parent,dict(half_life=half_life),'One-factor train-role time-decay sensitivity')
            for beta in [.5,1]:add(3,parent,dict(company_beta=beta),'One-factor train-role company-weight sensitivity')
    if ncfg.get('sampling','none')=='none':
        for value in [.05,.2]:add(3,neural,dict(pair_lambda=value),'Within-year limited pairwise proxy, added to established BCE backbone')
    curves=[]
    for path in (ART/'logs').glob(neural.trial_id+'__seed42__year*__stop_curve.csv'):
        d=pd.read_csv(path);best=d.loc[d.stop_AP.idxmax()]
        curves.append(dict(curve=str(path.relative_to(ROOT)),best_stop_AP=float(best.stop_AP),train_probe_AP_at_best=float(best.train_probe_AP),best_epoch=int(best.epoch),executed_epochs=int(d.epoch.max()),interpretation='Probe/stop gap and cap contact are optimization clues, not a causal fit diagnosis'))
    dump(ROOT/'configs/local_refinement_plan.json',dict(registered_before_refinement_at_utc=now(),rounds_max=3,deadline_hours=deadline,maximum_deadline_hours=5.45,cost_projected_remaining_hours=reserve_hours,promotion_feasibility=feasibility,neural_curve_evidence=curves,plan=plans,history_and_confirmation_not_opened=True))
    records=[]
    for i,p in enumerate(plans):
        name=f'refine_r{p["round"]}_{i:02d}_{digest(p["config"])[:8]}'
        db=database();prior=db.execute('SELECT status,macro_AP FROM trial_records WHERE trial_id=?',(name,)).fetchone();db.close()
        if prior and prior[0]=='COMPLETE':
            records.append(dict(trial_id=name,round=p['round'],parent=p['parent'],status='COMPLETE',macro_AP=prior[1],reason=p['reason']+'; reused completed trial'));continue
        db=database();current=pd.read_sql_query("SELECT * FROM trial_records WHERE status='COMPLETE' AND stage IN ('screen','baseline','refinement')",db);db.close()
        from .budget import required_seconds,TREE
        best_tree=current[current.family.isin(TREE)].sort_values('macro_AP',ascending=False).iloc[0].trial_id
        best_nn=current[current.family.isin(NN)].sort_values('macro_AP',ascending=False).iloc[0].trial_id
        required=required_seconds(pd.read_csv(RESULT/'development_by_year.csv'),best_tree,best_nn)
        if hours()+required/3600+.15>=7.15:
            records.append(dict(trial_id=name,round=p['round'],parent=p['parent'],status='NOT_STARTED_BUDGET',reason='Preserve full registered-seed promotion after measured incumbent cost changed'));continue
        if hours()+p['estimated_seconds']/3600>deadline:
            if prior and prior[0]=='RUNNING':
                db=database();db.execute('UPDATE trial_records SET status=?,finished=?,error=? WHERE trial_id=?',('INCOMPLETE_BUDGET',now(),'Local refinement budget ended; interrupted partial folds retained',name));db.commit();db.close()
            records.append(dict(trial_id=name,round=p['round'],parent=p['parent'],status='NOT_STARTED_BUDGET',reason=p['reason']));continue
        status,value=execute(p['config'],name,'refinement',[2009,2011,2013])
        records.append(dict(trial_id=name,round=p['round'],parent=p['parent'],status=status,macro_AP=value,reason=p['reason']))
    pd.DataFrame(records).to_csv(RESULT/'local_refinement_trace.csv',index=False)
    dump(marker,dict(completed_at_utc=now(),executed=sum(x['status']!='NOT_STARTED_BUDGET' for x in records),not_started=sum(x['status']=='NOT_STARTED_BUDGET' for x in records)))
