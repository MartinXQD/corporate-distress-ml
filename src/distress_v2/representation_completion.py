"""Finish the full MLP embedding pair and explicitly capped TabM controls."""
import json
import os
import optuna
import joblib
from .common import ROOT,ART,dump,now,hours,state,database
from .search import execute,propose

def run():
    marker=ART/'representation_completion.json'
    if marker.exists():return
    assert not (ART/'confirmation_opened.json').exists()
    state('representation_completion',running_pid=os.getpid(),error=None)
    deadline=5.75
    dump(ROOT/'configs/representation_completion_plan.json',dict(at_utc=now(),deadline_hours=deadline,full_mlp_pair='Original queued RAW17/std01 PLE configuration; same 200-epoch ceiling as the no-embedding control',tabm_pair='Same RAW17/std01, width128/depth2/k16, 20-epoch ceiling for both no-embedding and PLE; limited-budget mechanism control, not a full 200-epoch tuning claim',selection_information='Only completed development years through2013; no confirmation/history'))
    records=[]
    # Complete the already started inexpensive matched-input XGB comparison.
    db=database();pending=db.execute("SELECT trial_id,config_json FROM trial_records WHERE stage='refinement' AND status='RUNNING' AND family IN ('catboost','xgboost')").fetchall();db.close()
    for name,raw in pending:
        if hours()<deadline:records.append(dict(trial_id=name,result=execute(json.loads(raw),name,'refinement',[2009,2011,2013])))
    if hours()<deadline:
        path=ART/'sampler_num_mlp.joblib';study=optuna.create_study(study_name='num_mlp',storage='sqlite:///'+str(ART/'optuna.sqlite'),direction='maximize',load_if_exists=True,sampler=joblib.load(path))
        trial=study.ask();cfg=propose(trial,'num_mlp')
        assert cfg['features']=='RAW17' and cfg['embedding']=='piecewise' and cfg['preprocess']=='std01'
        from .common import digest
        name=f'num_mlp_{trial.number:03d}_{digest(cfg)[:8]}';trial.set_user_attr('actual_config',cfg)
        status,value=execute(cfg,name,'screen',[2009,2011,2013]);study.tell(trial.number,value if status=='COMPLETE' else None,state=optuna.trial.TrialState.COMPLETE if status=='COMPLETE' else optuna.trial.TrialState.FAIL);joblib.dump(study.sampler,path)
        records.append(dict(trial_id=name,status=status,screen_AP=value))
    source=next((ROOT/'configs').glob('trial_tabm_000_*.json'));base=json.loads(source.read_text())
    controls={}
    for mode in ['none','piecewise']:
        name='repr_r2_tabm20_'+mode;controls[mode]=name
        cfg={**base,'embedding':mode,'max_epochs':20,'patience':25,'per_fit_seconds':360}
        if hours()>=deadline:
            records.append(dict(trial_id=name,status='NOT_STARTED_BUDGET'));continue
        status,value=execute(cfg,name,'refinement',[2009,2011,2013]);records.append(dict(trial_id=name,status=status,screen_AP=value))
    dump(ROOT/'configs/representation_matched_controls.json',dict(tabm20=dict(without=controls['none'],with_embedding=controls['piecewise'],maximum_epochs=20,scope='Both use the same 20-epoch ceiling; independent of the full 200-epoch no-embedding trial. A one-seed three-year mechanism control, not fully tuned TabM.')))
    dump(marker,dict(completed_at_utc=now(),records=records,original_wall_deadline_unchanged=True))

if __name__=='__main__':run()
