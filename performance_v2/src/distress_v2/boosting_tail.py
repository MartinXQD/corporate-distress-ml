"""Use feasible pre-selection reserve for the registered strong-tree HPO targets."""
import json
import joblib
import optuna
import pandas as pd
from .common import ROOT,ART,RESULT,dump,now,hours,state,database
from .search import execute,propose,export,TARGETS,NN

def run():
    marker=ART/'boosting_tail_complete.json'
    if marker.exists():return
    assert not (ART/'confirmation_opened.json').exists()
    assert not (ROOT/'configs/selection_freeze.json').exists()
    db=database();t=pd.read_sql_query("SELECT * FROM trial_records WHERE status='COMPLETE' AND stage IN ('screen','baseline','refinement')",db);db.close()
    folds=pd.read_csv(RESULT/'development_by_year.csv');cost=folds.groupby('candidate').fit_seconds.mean()
    from .budget import affordable_pair
    best_tree,best_nn,feasibility=affordable_pair(t,folds,hours())
    if best_tree is None or best_nn is None:
        dump(marker,dict(completed_at_utc=now(),executed=0,reason='Remaining budget cannot fund required-seed promotion',feasibility=feasibility));return
    fixed=['baseline_legacy_strong_inner_l1','baseline_v1_lags_xgb','baseline_v1_raw_catboost','baseline_v1_selected_rf']
    mandatory=list(dict.fromkeys([*fixed,best_nn,best_tree]))
    stability=sum(12*cost.get(k,60.) for k in mandatory)
    final=1.3*(10*cost.get(best_nn,120.)+10*cost.get(best_tree,30.)+sum(7*cost.get(k,30.) for k in fixed)+7*cost.get('baseline_v1_selected_dl_corrected_steps',20.))
    reserve=(stability+final)/3600+.55
    deadline=min(5.9,7.15-feasibility['projected_required_seconds']/3600-.10)
    dump(ROOT/'configs/boosting_tail_budget.json',dict(at_utc=now(),deadline_hours=deadline,registered_search_maximum_hours=5.9,training_completion_target_hours=7.15,projected_stability_seconds=stability,projected_final_seconds=final,minimum_delivery_reserve_hours=.55,safety_buffer_hours=.10,promotion_feasibility=feasibility,reason='Finish registered boosting targets when measured cost permits; no new feature space, scoring years, seeds, or historical selection',family_targets={k:TARGETS[k] for k in ['catboost','xgboost']}))
    studies={}
    for family in ['catboost','xgboost']:
        path=ART/f'sampler_{family}.joblib';sampler=joblib.load(path)
        studies[family]=optuna.create_study(study_name=family,storage='sqlite:///'+str(ART/'optuna.sqlite'),load_if_exists=True,direction='maximize',sampler=sampler)
    while hours()<deadline:
        made=False
        for family,study in studies.items():
            complete=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)
            attempts=sum(t.state!=optuna.trial.TrialState.WAITING for t in study.trials)
            if complete>=TARGETS[family] or attempts>=TARGETS[family]+8 or hours()>=deadline:continue
            trial=study.ask();config=propose(trial,family)
            from .common import digest
            name=f'{family}_{trial.number:03d}_{digest(config)[:8]}';trial.set_user_attr('actual_config',config)
            state('boosting_target_completion',current_trial=name,hours_elapsed=hours())
            joblib.dump(study.sampler,ART/f'sampler_{family}.joblib')
            status,value=execute(config,name,'screen',[2009,2011,2013])
            study.tell(trial.number,value if status=='COMPLETE' else None,state=optuna.trial.TrialState.COMPLETE if status=='COMPLETE' else optuna.trial.TrialState.FAIL)
            joblib.dump(study.sampler,ART/f'sampler_{family}.joblib');made=True
        if not made:break
    budget=pd.read_csv(RESULT/'search_budget_summary.csv').set_index('family')
    for family,study in studies.items():
        complete=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)
        budget.loc[family,'actual_complete']=complete;budget.loc[family,'completed_target_shortfall']=max(0,TARGETS[family]-complete)
        budget.loc[family,'failed_attempts']=sum(t.state==optuna.trial.TrialState.FAIL for t in study.trials)
        budget.loc[family,'waiting_enqueued']=sum(t.state==optuna.trial.TrialState.WAITING for t in study.trials)
    budget.reset_index().to_csv(RESULT/'search_budget_summary.csv',index=False)
    export();dump(marker,dict(completed_at_utc=now(),deadline_hours=deadline,confirmation_unopened=True))
