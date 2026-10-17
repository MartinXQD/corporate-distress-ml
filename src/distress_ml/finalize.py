"""Freeze pre-evaluation artifacts, evaluate once, and verify checkpoint reproduction."""
from __future__ import annotations
import hashlib
import json
import subprocess
import time
import numpy as np
import pandas as pd
from scipy.special import expit
from .common import ROOT,ART,PUBLIC,dump,protocol,sha,state,now,object_hash
from .data import load_data
from .features import build_features
from .splits import groups,row_hash
from .train import NEURAL,base_config,config_id,fit_pipeline,save_pipeline,load_pipeline
from .calibration import fit_calibration,calibrated_predict
from .evaluation import metrics,budgets,reliability,bootstrap_pairs
from .develop import summarize

def get_cfg(cid):
    return json.loads((ART/f'logs/config_{cid}.json').read_text())

def train_final(df,cfg,label,group_fold=None):
    path=ART/f'models/{label}.joblib'
    if path.exists():
        p=load_pipeline(path)
        if hasattr(p,'calibrator'): return path
    f=build_features(df,cfg['feature_set'])
    fit=df.year<=2012; cal=df.year==2013; dec=df.year==2014
    p=fit_pipeline(df.loc[fit],f.loc[fit],cfg,label)
    p=fit_calibration(p,df.loc[fit],df.loc[cal],df.loc[dec],f.loc[cal],f.loc[dec],label)
    p.metadata.update(group_fold=group_fold)
    save_pipeline(p,path)
    # Save natural-prevalence calibration and threshold-role predictions locally.
    frames=[]
    for role,mask in [('calibration',cal),('threshold_selection',dec)]:
        raw,score=calibrated_predict(p,df.loc[mask],f.loc[mask])
        frames.append(prediction_frame(df.loc[mask],p,raw,score,label,role,group_fold))
    pd.concat(frames).to_csv(ART/f'predictions/{label}_pre_evaluation.csv.gz',index=False)
    print('Final fitted',label,p.metadata['fit_seconds'],flush=True)
    return path

def prediction_frame(df,p,raw,score,label,role,group_fold=None):
    pred=df[['row_id','company_name','year','y']].copy().rename(columns={'company_name':'company_id','y':'y_true'})
    return pred.assign(run_id=label,protocol_id=protocol()['protocol_id'],dataset_hash=protocol()['implementation']['source_data_hash'],split_role=role,group_fold=group_fold,seed=p.cfg['seed'],raw_score=raw,calibrated_score=score,fixed_threshold=p.threshold,predicted_label=(score>=p.threshold).astype(int),config_id=config_id(p.cfg))

def freeze():
    if (ROOT/'configs/final_freeze.json').exists():
        print('Final freeze already exists; use historical-evaluate or reproduce.'); return
    state('final_artifact_training')
    s=summarize(); complete=s[s.score_years==7]
    eligible=complete[~complete.family.isin(['dummy','altman'])]
    dl_families=NEURAL+['gru']
    ml=eligible[~eligible.family.isin(dl_families)].iloc[0]
    dl=eligible[eligible.family.isin(dl_families)].iloc[0]
    reps=dict(ML=ml.config_id,DL=dl.config_id)
    from .ensembles import choose_ensemble,RankEnsemblePipeline
    ensemble_choice=choose_ensemble(reps)
    dump(ROOT/'configs/final_selection.json',dict(selected_at_utc=now(),representatives=reps,selection='Equal-weight AP over seven development scoring years; historical labels unavailable to selection',development_summary_sha256=sha(PUBLIC/'development_summary.csv')))
    df=load_data('pre_evaluation'); assert df.year.max()==2014
    configurations={}
    for r in complete[complete.stage.isin(['core_recipe','sequence_control','sequence_extension'])].itertuples(): configurations[r.config_id]=get_cfg(r.config_id)
    for cid in reps.values(): configurations[cid]=get_cfg(cid)
    def fit_cost(cfg):
        selector=0 if cfg['selector']=='none' else 5
        stopping=cfg['family'] in NEURAL+['xgboost','catboost','gru']
        if cfg['family'] in ['dummy','altman']:return 1
        return (2+2*selector if stopping else 1+selector)+1
    used=len((ART/'logs/fit_ledger.jsonl').read_text().splitlines())
    planned_seeds=protocol()['runtime']['final_seeds']
    fixed_cost=sum(fit_cost(cfg) for cfg in configurations.values())+24+int(ensemble_choice['selected'])
    repeat_cost=sum(fit_cost(get_cfg(cid)) for cid in reps.values())
    if used+fixed_cost+4*repeat_cost>600:
        planned_seeds=planned_seeds[:3]
    if used+fixed_cost+(len(planned_seeds)-1)*repeat_cost>600:
        raise RuntimeError('Insufficient registered fit budget for minimum final evidence; save state and report the precise deficit.')
    dump(ROOT/'configs/final_seed_budget.json',dict(seeds=planned_seeds,fit_calls_before_final=used,estimated_final_calls=fixed_cost+(len(planned_seeds)-1)*repeat_cost,reason='Pre-historical budget check; never selected by seed performance'))
    inventory=[]
    for cid,cfg in configurations.items():
        label='final_'+cid+'_seed42'; path=train_final(df,cfg,label)
        inventory.append(dict(label=label,path=str(path.relative_to(ROOT)),config_id=cid,family=cfg['family'],feature_set=cfg['feature_set'],seed=42,role='primary',representative=[k for k,v in reps.items() if v==cid]))
    for kind,cid in reps.items():
        for seed in planned_seeds[1:]:
            cfg=get_cfg(cid); cfg['seed']=seed
            label=f'final_{cid}_seed{seed}'; path=train_final(df,cfg,label)
            inventory.append(dict(label=label,path=str(path.relative_to(ROOT)),config_id=cid,family=cfg['family'],feature_set=cfg['feature_set'],seed=seed,role='seed_repeat',representative=[kind]))
    if ensemble_choice['selected']:
        mlp=load_pipeline(ART/f'models/final_{reps["ML"]}_seed42.joblib')
        dlp=load_pipeline(ART/f'models/final_{reps["DL"]}_seed42.joblib')
        p=RankEnsemblePipeline(mlp,dlp)
        p.metadata=dict(run_id='final_equal_rank_ensemble_seed42',fit_rows=int((df.year<=2012).sum()),train_row_hash=row_hash(df[df.year<=2012]),fit_seconds=mlp.metadata['fit_seconds']+dlp.metadata['fit_seconds'],features=len(p.columns),parameters=None,epochs=0,selected_features=p.columns,fit_calls=[],base_models=reps,annual_cohort_required=True)
        f=build_features(df,'ALL')
        p=fit_calibration(p,df[df.year<=2012],df[df.year==2013],df[df.year==2014],f[df.year==2013],f[df.year==2014],'ensemble')
        path=ART/'models/final_equal_rank_ensemble_seed42.joblib'; save_pipeline(p,path)
        inventory.append(dict(label='final_equal_rank_ensemble_seed42',path=str(path.relative_to(ROOT)),config_id=config_id(p.cfg),family='equal_rank_ensemble',feature_set='ALL',seed=42,role='primary',representative=[]))
    g=groups(df)
    for fold in range(3):
        sub=df.loc[g!=fold]
        for family in ['logistic','xgboost','residual_mlp']:
            cfg=base_config(family,'S1')
            label=f'stress_group{fold}_{family}'; path=train_final(sub,cfg,label,fold)
            inventory.append(dict(label=label,path=str(path.relative_to(ROOT)),config_id=config_id(cfg),family=family,feature_set='S1',seed=42,role='untuned_group_stress',group_fold=fold,representative=[]))
    artifacts={}
    for item in inventory:
        path=ROOT/item['path']
        for ext in ['.joblib','.json','.pt']:
            p=path.with_suffix(ext)
            if p.exists(): artifacts[str(p.relative_to(ROOT))]=sha(p)
    source={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/distress_ml').glob('*.py') if p.stem not in ['reporting','report_content','explain','cli']}
    try: commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError: commit='No commit existed at freeze; source hashes provided'
    freeze=dict(protocol_id=protocol()['protocol_id'],protocol_sha256=sha(ROOT/'configs/protocol_v1.yaml'),data_sha256=protocol()['implementation']['source_data_hash'],frozen_at_utc=now(),representatives=reps,final_seeds=planned_seeds,models=inventory,artifact_hashes=artifacts,source_hashes=source,git_commit=commit,status='frozen_before_historical_label_loading')
    dump(ROOT/'configs/final_freeze.json',freeze)
    state('final_frozen',complete=False)

def verify_freeze():
    f=json.loads((ROOT/'configs/final_freeze.json').read_text())
    assert f['protocol_sha256']==sha(ROOT/'configs/protocol_v1.yaml')
    for path,h in {**f['artifact_hashes'],**f['source_hashes']}.items():
        assert sha(ROOT/path)==h,f'Frozen artifact/source modified: {path}'
    return f

def historical_evaluate():
    freeze=verify_freeze()
    lock=ART/'logs/historical_evaluation_lock.json'
    if lock.exists():
        raise RuntimeError('Historical evaluation was already opened. Use saved predictions, not a second tuning/evaluation cycle. Record a bug amendment before any required rerun.')
    dump(lock,dict(started_at_utc=now(),freeze_sha256=sha(ROOT/'configs/final_freeze.json'),status='opened_once'))
    state('historical_evaluation')
    df=load_data('historical'); hist=df.year>=2015; target=df.loc[hist].copy()
    seen=set(df.loc[df.year<=2012,'company_name']); seen2014=set(df.loc[df.year<=2014,'company_name'])
    metrics_rows=[]; annual_rows=[]; subset_rows=[]; budget_rows=[]; cal_rows=[]; seed_rows=[]; group_rows=[]
    raw_predictions={}; selected={}; baseline=None
    for item in freeze['models']:
        p=load_pipeline(ROOT/item['path'])
        allf=build_features(df,p.cfg['feature_set'])
        if item['role']=='untuned_group_stress':
            fold=item['group_fold']; mask=hist & (groups(df)==fold); sub=df.loc[mask]
            fit_companies=set(df.loc[(df.year<=2012)&(groups(df)!=fold),'company_name'])
            assert not set(sub.company_name)&fit_companies
        else:
            mask=hist; sub=target
        started=time.perf_counter(); raw,score=calibrated_predict(p,sub,allf.loc[mask]); inference=time.perf_counter()-started
        label=item['label']
        prediction_frame(sub,p,raw,score,label,'historical_evaluation',item.get('group_fold')).to_csv(ART/f'predictions/{label}_historical.csv.gz',index=False)
        meta=dict(label=label,config_id=item['config_id'],family=item['family'],feature_set=item['feature_set'],seed=item['seed'],role=item['role'],fit_seconds=p.metadata['fit_seconds'],inference_seconds=inference,features=p.metadata['features'],parameters=p.metadata['parameters'],epochs=p.metadata['epochs'],row_hash=row_hash(sub))
        if item['role']=='untuned_group_stress':
            group_rows.append(dict(**meta,group_fold=item['group_fold'],**metrics(sub,score,p.threshold)))
            continue
        m=metrics(sub,score,p.threshold)
        seed_rows.append(dict(**meta,**m))
        if item['role']=='seed_repeat': continue
        metrics_rows.append(dict(**meta,score_type='calibrated',**m))
        metrics_rows.append(dict(**meta,score_type='uncalibrated',**metrics(sub,expit(raw),.5,ranking=raw)))
        for year in sorted(sub.year.unique()):
            ix=(sub.year==year).to_numpy()
            annual_rows.append(dict(**meta,year=int(year),**metrics(sub.loc[ix],score[ix],p.threshold)))
        for cutoff,seen_set in [(2012,seen),(2014,seen2014)]:
            for novelty in ['seen','unseen']:
                ix=sub.company_name.isin(seen_set).to_numpy()
                if novelty=='unseen': ix=~ix
                subset_rows.append(dict(**meta,seen_cutoff=cutoff,novelty=novelty,**metrics(sub.loc[ix],score[ix],p.threshold)))
        b=budgets(sub,score).assign(label=label,config_id=item['config_id']); budget_rows.extend(b.to_dict('records'))
        for kind,pp in [('calibrated',score),('uncalibrated',expit(raw))]:
            cal_rows.extend(reliability(sub,pp).assign(label=label,score_type=kind).to_dict('records'))
        raw_predictions[label]=score
        for kind in item['representative']: selected[kind]=label
        if item['family']=='logistic' and item['feature_set']=='S0': baseline=label
        print('Historical scored',label,'AP',m['AP'],flush=True)
    for filename,rows in [('historical_metrics',metrics_rows),('historical_yearly',annual_rows),('historical_seen_unseen',subset_rows),('historical_budgets',budget_rows),('reliability_bins',cal_rows),('seed_results',seed_rows),('group_stress_metrics',group_rows)]:
        pd.DataFrame(rows).to_csv(PUBLIC/(filename+'.csv'),index=False)
    # Choices were frozen from development; never select a test winner here.
    bootstrap_scores={name:raw_predictions[name] for name in dict.fromkeys([baseline,*selected.values()])}
    ci=bootstrap_pairs(target,bootstrap_scores,baseline,2000,2026)
    ci.to_csv(PUBLIC/'paired_company_bootstrap.csv',index=False)
    dump(ART/'logs/historical_evaluation_complete.json',dict(completed_at_utc=now(),models=len(freeze['models']),historical_rows=len(target),bootstrap_draws=2000,selected=selected,reference=baseline))
    state('historical_complete',complete=False)

def reproduce(model):
    """Reload a trusted final artifact and compare against saved historical scores."""
    freeze=verify_freeze(); item=next(x for x in freeze['models'] if x['label']==model)
    p=load_pipeline(ROOT/item['path']); df=load_data('historical')
    f=build_features(df,p.cfg['feature_set']); mask=df.year>=2015
    if item['role']=='untuned_group_stress': mask &= groups(df)==item['group_fold']
    raw,score=calibrated_predict(p,df.loc[mask],f.loc[mask])
    saved=pd.read_csv(ART/f'predictions/{model}_historical.csv.gz')
    assert np.array_equal(df.loc[mask,'row_id'].to_numpy(),saved.row_id.to_numpy())
    np.testing.assert_allclose(raw,saved.raw_score.to_numpy(),rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(score,saved.calibrated_score.to_numpy(),rtol=1e-6,atol=1e-6)
    result=dict(model=model,rows=len(saved),max_score_error=float(np.max(np.abs(score-saved.calibrated_score))),passed=True)
    dump(ART/f'logs/reproduce_{model}.json',result); print(json.dumps(result))
