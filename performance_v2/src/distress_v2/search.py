"""Diverse startup recipes plus persistent per-family Optuna TPE searches."""
import json
import time
import traceback
import numpy as np
import pandas as pd
import optuna
import joblib
from .common import ROOT,ART,RESULT,digest,dump,now,hours,state,database,time_guard
from .data import load,features
from .training import fit,save,NN
from distress_ml.evaluation import budgets
from sklearn.metrics import average_precision_score,roc_auc_score

TARGETS={'catboost':36,'xgboost':36,'random_forest':12,'extra_trees':12,'legacy_residual':18,'mlp':18,'num_mlp':14,'tabm':14,'official_ft':8,'xgb_rank':6}
FEATURES=['RAW17','RATIO_CORE','LAGS_RAW3','SLOG17','SLOG_RATIO','LEGACY33','INNER_L1','LAGS_SLOG3','DYNAMIC_COMPACT']

def threads():return json.loads((ROOT/'configs/device_profile.json').read_text())['selected_threads']
def base(family,**kw):
    cfg=dict(family=family,features='RAW17',preprocess='native' if family in ['catboost','xgboost','xgb_rank'] else ('none' if family in ['random_forest','extra_trees'] else 'std01'),seed=42,threads=threads(),weight_alpha=0,sampling='none',refit_rule='steps',max_epochs=200,patience=25,per_fit_seconds=720)
    cfg.update(kw);return cfg

def baselines():
    return {
      'legacy_strong_inner_l1':base('legacy_residual',features='INNER_L1',width=192,depth=4,dropout=.25,learning_rate=.001,weight_decay=.0002,batch_size=512,weight_alpha=1,max_epochs=100,patience=12,indicators=False,gradient_clip=None),
      'v1_lags_xgb':base('xgboost',features='V1_LAGS',preprocess='v1',clip=True,iterations=180,depth=3,learning_rate=.05,min_child_weight=1.,subsample=.9,colsample_bytree=.9,reg_lambda=1.,reg_alpha=0.),
      'v1_raw_catboost':base('catboost',features='V1_S0',preprocess='v1',clip=True,iterations=180,depth=5,learning_rate=.05),
      'v1_selected_rf':base('random_forest',features='V1_S1',preprocess='v1',clip=True,iterations=120,depth=16,leaf=3,max_features='sqrt'),
      'v1_selected_dl_corrected_steps':base('v1_residual',features='V1_S1',preprocess='v1',clip=True,width=64,depth=2,dropout=.15,learning_rate=.001,weight_decay=.0001,batch_size=1024,sampling='ros',sampling_budget='epoch_equivalent',refit_rule='epochs',refit_contract_revision=2,max_epochs=25,patience=4),
      'logistic':base('logistic',features='RAW17',preprocess='std005',C=1.)
    }

def measures(df,score):
    y=df.y.to_numpy(int);b=budgets(df,score)
    return dict(AP=float(average_precision_score(y,score)),ROC_AUC=float(roc_auc_score(y,score)),recall5=float(b[b.budget==.05].recall.mean()),rows=len(df),prevalence=float(y.mean()),positive_companies=int(df.loc[df.y==1,'company_name'].nunique()))

def run_fold(cfg,year,stage='screen',candidate=None):
    key=digest({k:v for k,v in cfg.items() if k!='seed'})[:12]
    name=f'{candidate or key}__seed{cfg["seed"]}__year{year}'
    done=ART/f'logs/fold_{name}.json'
    if done.exists():return json.loads(done.read_text())
    df=load('search');x=features(df,cfg['features']);tr=df.year<year;va=df.year==year
    assert df.loc[tr,'y'].notna().all() and year<=2013
    p=fit(df.loc[tr],x.loc[tr],cfg,name)
    score=p.predict(x.loc[va]);metrics=measures(df.loc[va],score)
    output=df.loc[va,['row_id','company_name','year','y']].rename(columns={'company_name':'company_id','y':'y_true'}).copy();output['score']=score
    output.to_csv(ART/f'predictions/{name}.csv.gz',index=False)
    result=dict(config_id=key,candidate=candidate or key,family=cfg['family'],features=cfg['features'],seed=cfg['seed'],year=year,stage=stage,**metrics,fit_seconds=p.metadata['fit_seconds'],parameters=p.metadata['final_stats'].get('parameters'),best_steps=p.metadata['inner_stats'].get('best_steps'),best_rounds=p.metadata['inner_stats'].get('best_rounds'),train_AP=p.metadata['final_stats'].get('train_AP'),train_BCE=p.metadata['final_stats'].get('train_BCE'),quantile_saturation=p.pp.saturation(x.loc[va,p.selected]) if hasattr(p.pp,'saturation') else None,prediction_path=str((ART/f'predictions/{name}.csv.gz').relative_to(ROOT)))
    dump(done,result)
    # Predictions and learning curves preserve every trial; full HPO forests are
    # deliberately not duplicated on disk. Frozen final pipelines are serialized.
    export();return result

def export():
    rows=[json.loads(p.read_text()) for p in (ART/'logs').glob('fold_*.json')]
    if rows:pd.DataFrame(rows).to_csv(RESULT/'development_by_year.csv',index=False)
    c=database();trials=pd.read_sql_query('SELECT * FROM trial_records',c);c.close()
    counts={}
    for r in rows:
        if r['seed']==42:counts.setdefault(r['candidate'],set()).add(r['year'])
    trials['completed_development_years']=trials.trial_id.map(lambda k:len(counts.get(k,set())))
    costs={}
    for r in rows:
        if r['seed']==42 and r['year'] in [2009,2011,2013]:costs[r['candidate']]=costs.get(r['candidate'],0.)+r['fit_seconds']
    trials['completed_screen_fold_fit_seconds']=trials.trial_id.map(costs)
    trials['timing_basis']='seconds=current attempt monotonic elapsed; fold-fit sum includes reused completed folds; UTC wall budget includes gaps'
    def functional_recipe(raw):
        c=json.loads(raw)
        if c.get('embedding','none')=='none':
            for field in ['embedding','embedding_dim','bins']:c.pop(field,None)
            if c['family']=='num_mlp':c['family']='legacy_residual' if c.get('backbone')=='residual' else 'mlp'
        if c.get('embedding')=='periodic':c.pop('bins',None)
        if c['family']=='legacy_residual':c.pop('backbone',None)
        if c.get('backbone','mlp')=='mlp':c.pop('backbone',None)
        if c['family']=='official_ft':c.pop('width',None)
        return digest(c)
    trials['functional_recipe_hash']=trials.config_json.map(functional_recipe)
    trials['metric_identity']=trials.status.map(lambda s:'Complete registered trial' if s=='COMPLETE' else 'Partial/invalid/in-progress; not eligible as complete')
    trials.to_csv(RESULT/'trials.csv',index=False)

def execute(cfg,trial_id,stage,years):
    c=database();entry=c.execute('SELECT status FROM trial_records WHERE trial_id=?',(trial_id,)).fetchone()
    if entry and entry[0]=='COMPLETE':
        value=c.execute('SELECT macro_AP FROM trial_records WHERE trial_id=?',(trial_id,)).fetchone()[0]
        c.close();return 'COMPLETE',value
    if entry:
        cursor=c.execute('SELECT * FROM trial_records WHERE trial_id=?',(trial_id,));record=dict(zip([d[0] for d in cursor.description],cursor.fetchone()))
        record['archived_at_utc']=now();record['reason']='Prior attempt preserved before bounded resume'
        with (ART/'logs/trial_attempts.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
    c.execute('INSERT OR REPLACE INTO trial_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',(trial_id,cfg['family'],stage,'RUNNING',json.dumps(cfg,sort_keys=True),now(),None,None,None,None,None,None));c.commit();c.close()
    started=time.monotonic();rows=[];status='COMPLETE';error=None
    dump(ROOT/f'configs/trial_{trial_id}.json',cfg)
    # A bounded resume can reuse completed folds. Preserve curves from an
    # interrupted, incomplete fit before repeating its deterministic fit.
    if entry:
        import shutil
        backup=ART/'logs/interrupted_attempts'/f'{trial_id}_{time.time_ns()}'
        for old in (ART/'logs').glob(trial_id+'__*'):
            backup.mkdir(parents=True,exist_ok=True);shutil.copy2(old,backup/old.name)
    try:
        for year in years:
            time_guard(5.95 if stage in ['screen','baseline','bridge'] else 7.15)
            rows.append(run_fold(cfg,year,stage,candidate=trial_id))
    except Exception as exc:
        status='FAIL';error=type(exc).__name__+': '+str(exc)
        with (ART/'logs/failures.log').open('a') as f:f.write(now()+' '+trial_id+'\n'+traceback.format_exc()+'\n')
    mean=lambda col:float(np.mean([r[col] for r in rows])) if rows else None
    c=database();c.execute('UPDATE trial_records SET status=?,finished=?,seconds=?,macro_AP=?,macro_ROC=?,recall5=?,error=? WHERE trial_id=?',(status,now(),time.monotonic()-started,mean('AP'),mean('ROC_AUC'),mean('recall5'),error,trial_id));c.commit();c.close();export()
    print(json.dumps(dict(trial=trial_id,family=cfg['family'],features=cfg['features'],status=status,AP=mean('AP'),seconds=round(time.monotonic()-started,1),error=error)),flush=True)
    return status,mean('AP')

def bridge():
    d=base('legacy_residual',features='RAW18',preprocess='std01',indicators=False,width=192,depth=4,dropout=.25,learning_rate=.001,weight_decay=.0002,batch_size=512,weight_alpha=1,max_epochs=100,patience=12)
    recipes={'legacy192_raw18':d,'legacy64_raw18':{**d,'width':64,'depth':2},'legacy192_raw17':{**d,'features':'RAW17'},'legacy192_s1':{**d,'features':'V1_S1'},'legacy192_clip005':{**d,'preprocess':'std005'},'legacy192_quantile':{**d,'preprocess':'quantile'},'legacy192_keepbest':{**d,'refit_rule':'keep_best'},'legacy192_epochrefit':{**d,'refit_rule':'epochs'},'legacy192_unweighted':{**d,'weight_alpha':0},'legacy192_ros_steps':{**d,'weight_alpha':0,'sampling':'ros'},'legacy192_ros_epochs':{**d,'weight_alpha':0,'sampling':'ros','sampling_budget':'epoch_equivalent'},'legacy192_inner_l1':{**d,'features':'INNER_L1'}}
    tree=base('xgboost',features='RAW17',iterations=1500,depth=4,learning_rate=.05)
    for feature in ['RAW17','RATIO_CORE','LAGS_RAW3']:
        recipes['xgb_'+feature]={**tree,'features':feature}
    recipes['xgb_raw_keepbest']={**tree,'refit_rule':'keep_best'}
    for name,cfg in recipes.items():
        if hours()>1.:break
        execute(cfg,'bridge_'+name,'bridge',[2012])
    state('bridge_completed',bridge_seconds_limit=3600)

def enqueue(study,family):
    if study.trials:return
    for n,feature in enumerate(FEATURES[:6]):
        if family in ['num_mlp','tabm']:feature=['RAW17','RAW17','SLOG17','SLOG17','RATIO_CORE','LAGS_SLOG3'][n]
        params={'features':feature,'preprocess':('native' if family in ['catboost','xgboost','xgb_rank'] else ('none' if family in ['random_forest','extra_trees'] else ['std01','quantile','std005','robust'][n%4])),'weight_alpha':0}
        if family in ['legacy_residual','mlp','num_mlp','tabm','official_ft']:
            params.update(width=192 if family=='legacy_residual' else 128,depth=4 if family=='legacy_residual' else 2,dropout=.25 if family=='legacy_residual' else .15,learning_rate=.001,weight_decay=.0002,batch_size=512,schedule='fixed',sampling='none')
        if family in ['num_mlp','tabm']:params['preprocess']=['std01','std01','quantile','quantile','std005','std005'][n]
        if family=='num_mlp':params.update(embedding=['none','piecewise','none','piecewise','periodic','piecewise'][n],bins=16,embedding_dim=8,backbone='mlp')
        if family=='tabm':params.update(embedding='none' if n%2==0 else 'piecewise',bins=16,embedding_dim=8,width=128,depth=2)
        if family in ['catboost','xgboost','xgb_rank']:params.update(iterations=3000,depth=4 if family!='catboost' else 5,learning_rate=.05)
        study.enqueue_trial(params,skip_if_exists=True)

def propose(trial,family):
    cfg=base(family,features=trial.suggest_categorical('features',FEATURES))
    choices=['native','none','std01'] if family in ['catboost','xgboost','xgb_rank'] else (['none','std01'] if family in ['random_forest','extra_trees'] else ['std01','std005','robust','quantile'])
    cfg['preprocess']=trial.suggest_categorical('preprocess',choices)
    if family not in ['legacy_residual','mlp','xgb_rank']:cfg['weight_alpha']=trial.suggest_categorical('weight_alpha',[0,.5])
    if family in ['catboost','xgboost','xgb_rank']:
        cfg.update(iterations=trial.suggest_categorical('iterations',[1500,3000]),depth=trial.suggest_int('depth',3 if family=='catboost' else 2,8),learning_rate=trial.suggest_float('learning_rate',.01,.15,log=True))
        if family=='catboost':
            cfg.update(l2_leaf_reg=trial.suggest_float('l2_leaf_reg',1,50,log=True),random_strength=trial.suggest_float('random_strength',0,2),bootstrap_type=trial.suggest_categorical('bootstrap_type',['Bayesian','Bernoulli']))
            if cfg['bootstrap_type']=='Bayesian':cfg['bagging_temperature']=trial.suggest_float('bagging_temperature',0,2)
            else:cfg['subsample']=trial.suggest_float('subsample',.6,1.)
        else:
            cfg.update(min_child_weight=trial.suggest_float('min_child_weight',1,64,log=True),subsample=trial.suggest_float('subsample',.6,1),colsample_bytree=trial.suggest_float('colsample_bytree',.5,1),reg_lambda=trial.suggest_float('reg_lambda',.01,100,log=True),reg_alpha=trial.suggest_float('reg_alpha',1e-6,10,log=True))
            if family=='xgb_rank':cfg.update(objective=trial.suggest_categorical('objective',['rank:map','rank:pairwise']),weight_alpha=0)
    elif family in ['random_forest','extra_trees']:
        depth=trial.suggest_categorical('depth',[6,10,16,0]);cfg.update(iterations=trial.suggest_categorical('iterations',[300,600,1000]),depth=depth or None,leaf=trial.suggest_categorical('leaf',[2,5,10,20,40]),max_features=trial.suggest_categorical('max_features',['sqrt',.5,.8,1.]),bootstrap=trial.suggest_categorical('bootstrap',[True,False]))
    else:
        widths=[128,256,512] if family=='tabm' else [64,128,192,256]
        cfg.update(width=trial.suggest_categorical('width',widths),depth=trial.suggest_categorical('depth',[2,3,4] if family=='tabm' else ([2,3] if family=='official_ft' else [1,2,4])),dropout=trial.suggest_categorical('dropout',[0.,.15,.25,.4]),learning_rate=trial.suggest_float('learning_rate',1e-4,.003,log=True),weight_decay=trial.suggest_categorical('weight_decay',[0.,1e-6,.0001,.0002,.001,.01]),batch_size=trial.suggest_categorical('batch_size',[256,512,1024]),schedule=trial.suggest_categorical('schedule',['fixed','cosine']))
        if family in ['legacy_residual','mlp']:
            cfg['sampling']=trial.suggest_categorical('sampling',['none','none','ros','smote'])
            if cfg['sampling']=='none':cfg['weight_alpha']=trial.suggest_categorical('weight_alpha',[0,.5])
            else:cfg['weight_alpha']=0
        if family in ['num_mlp','tabm']:
            cfg.update(embedding=trial.suggest_categorical('embedding',['none','piecewise'] if family=='tabm' else ['none','piecewise','periodic']),embedding_dim=trial.suggest_categorical('embedding_dim',[8,16]),bins=trial.suggest_categorical('bins',[8,16,32]))
            if family=='num_mlp':cfg['backbone']=trial.suggest_categorical('backbone',['mlp','residual'])
            else:cfg['k']=16
        if family=='official_ft':cfg.update(token_dim=trial.suggest_categorical('token_dim',[32,64,96]),max_epochs=100,patience=20)
    return cfg

def search():
    assert not (ART/'confirmation_opened.json').exists(),'No reopening HPO after confirmation'
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    studies={}
    for family in TARGETS:
        sampler_path=ART/f'sampler_{family}.joblib'
        sampler=joblib.load(sampler_path) if sampler_path.exists() else optuna.samplers.TPESampler(seed=42,n_startup_trials=8)
        studies[family]=optuna.create_study(study_name=family,storage='sqlite:///'+str(ART/'optuna.sqlite'),load_if_exists=True,direction='maximize',sampler=sampler)
        enqueue(studies[family],family)
    dump(ROOT/'configs/resource_reserve.json',dict(search_soft_stop_hours=4.8,registered_search_maximum=5.9,reason='Reserve measured full-fold three-seed stability, five-seed final baselines, company stress, verification and reports inside the eight-hour wall clock. May end earlier only when all search targets complete.'))
    # Measured residual trial 000 took 19 minutes. Cost-based scheduling gives
    # the two high-priority boosting families three turns per cycle. It does
    # not change spaces, scoring, seeds, or promote based on historical data.
    schedule=[f for f in TARGETS for _ in range(3 if f in ['catboost','xgboost'] else 1)]
    dump(ROOT/'configs/search_schedule_revision.json',dict(at_utc=now(),cycle=schedule,coverage_priority='Mandatory high-priority families with fewer than two complete recipes go first, ordered by completed coverage; pending partial trials resume within their family turn',reason='Observed CPU cost; preserve broad tree search and avoid leaving numerical embeddings/TabM unattempted. No score-based family elimination.',registered_protocol_unchanged=True))
    while hours()<4.8:
        made=False
        counts={f:sum(t.state==optuna.trial.TrialState.COMPLETE for t in s.trials) for f,s in studies.items()}
        underexplored=[f for f in list(TARGETS)[:8] if counts[f]<2]
        coverage_order=['num_mlp','tabm','legacy_residual','mlp','extra_trees','random_forest','catboost','xgboost']
        cycle=sorted(underexplored,key=lambda f:(counts[f],coverage_order.index(f)))+[f for f in schedule if f not in underexplored]
        with (ART/'logs/dispatch_cycles.jsonl').open('a') as stream:stream.write(json.dumps(dict(at_utc=now(),complete_counts=counts,cycle=cycle))+'\n')
        for family in cycle:
            target=TARGETS[family]
            study=studies[family]
            completed=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)
            attempts=sum(t.state!=optuna.trial.TrialState.WAITING for t in study.trials)
            if completed>=target or attempts>=target+8:continue
            if family in ['official_ft','xgb_rank'] and min(sum(t.state==optuna.trial.TrialState.COMPLETE for t in studies[f].trials) for f in list(TARGETS)[:8])<6:continue
            if hours()>=4.8:break
            pending=study.get_trials(states=[optuna.trial.TrialState.RUNNING])
            if pending:
                trial=pending[0];cfg=trial.user_attrs.get('actual_config')
                if cfg is None:study.tell(trial.number,state=optuna.trial.TrialState.FAIL);continue
            else:
                trial=study.ask();cfg=propose(trial,family);trial.set_user_attr('actual_config',cfg)
            trial_id=f'{family}_{trial.number:03d}_{digest(cfg)[:8]}'
            state('performance_search',current_trial=trial_id,hours_elapsed=hours())
            joblib.dump(study.sampler,ART/f'sampler_{family}.joblib')
            status,value=execute(cfg,trial_id,'screen',[2009,2011,2013])
            study.tell(trial.number,value if status=='COMPLETE' else None,state=optuna.trial.TrialState.COMPLETE if status=='COMPLETE' else optuna.trial.TrialState.FAIL)
            joblib.dump(study.sampler,ART/f'sampler_{family}.joblib')
            made=True
        if not made:break
    budget_rows=[]
    for family,study in studies.items():
        for pending in study.get_trials(states=[optuna.trial.TrialState.RUNNING]):
            cfg=pending.user_attrs.get('actual_config')
            if cfg:
                key=f'{family}_{pending.number:03d}_{digest(cfg)[:8]}'
                db=database();db.execute('UPDATE trial_records SET status=?,finished=?,error=? WHERE trial_id=?',('INCOMPLETE_BUDGET',now(),'Search budget ended before every registered scoring year completed; preserved folds not eligible for selection',key));db.commit();db.close()
            study.tell(pending.number,state=optuna.trial.TrialState.FAIL)
        complete=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)
        budget_rows.append(dict(family=family,target_complete=TARGETS[family],actual_complete=complete,failed_attempts=sum(t.state==optuna.trial.TrialState.FAIL for t in study.trials),waiting_enqueued=sum(t.state==optuna.trial.TrialState.WAITING for t in study.trials),completed_target_shortfall=max(0,TARGETS[family]-complete),family_closed_for_performance=False))
    pd.DataFrame(budget_rows).to_csv(RESULT/'search_budget_summary.csv',index=False)
    export();state('search_completed_pending_shortlist')

