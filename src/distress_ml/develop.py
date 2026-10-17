"""Four expanding calendar folds, staged search and durable out-of-fold predictions."""
from __future__ import annotations
import json
import time
import traceback
import importlib.util
import numpy as np
import pandas as pd
from scipy.special import expit
from .common import ROOT,ART,PUBLIC,protocol,dump,state,now,object_hash
from .data import load_data
from .features import build_features,feature_registry
from .splits import DEVELOPMENT,row_hash
from .train import NEURAL,base_config,config_id,fit_pipeline,save_pipeline
from .evaluation import metrics

def append_table(path,rows):
    path=ROOT/path
    new=pd.DataFrame(rows)
    if path.exists(): new=pd.concat([pd.read_csv(path),new],ignore_index=True)
    new.to_csv(path,index=False)

def run_config(df,cfg,stage):
    cid=config_id(cfg); all_features=build_features(df,cfg['feature_set'])
    run_rows=[]
    for fold,end,lo,hi in DEVELOPMENT:
        rid=f'dev_{cid}_{fold}'; result=ART/f'logs/{rid}_complete.json'
        if result.exists(): continue
        tr=(df.year<=end).to_numpy(); va=df.year.between(lo,hi).to_numpy()
        started=time.perf_counter()
        try:
            fitted=fit_pipeline(df.loc[tr],all_features.loc[tr],cfg,rid)
            infer=time.perf_counter(); score=fitted.predict_raw(df.loc[va],all_features.loc[va]); infer=time.perf_counter()-infer
            save_pipeline(fitted,ART/f'models/{rid}.joblib')
            pred=df.loc[va,['row_id','company_name','year','y']].copy().rename(columns={'company_name':'company_id','y':'y_true'})
            pred=pred.assign(run_id=rid,protocol_id=protocol()['protocol_id'],dataset_hash=protocol()['implementation']['source_data_hash'],split_role='development_score',group_fold=fold,seed=cfg['seed'],raw_score=score,calibrated_score=np.nan,fixed_threshold=np.nan,predicted_label=np.nan,config_id=cid)
            pred.to_csv(ART/f'predictions/{rid}.csv.gz',index=False)
            rows=[]
            for year in sorted(df.loc[va,'year'].unique()):
                sub=df.loc[va & (df.year==year)]; risk=score[df.loc[va,'year'].to_numpy()==year]; m=metrics(sub,expit(risk),ranking=risk)
                rows.append(dict(stage=stage,config_id=cid,run_id=rid,family=cfg['family'],feature_set=cfg['feature_set'],fold=fold,year=int(year),seed=cfg['seed'],fit_seconds=fitted.metadata['fit_seconds'],infer_seconds=infer,features=fitted.metadata['features'],parameters=fitted.metadata['parameters'],epochs=fitted.metadata['epochs'],row_hash=row_hash(sub),**m))
            append_table('results/public/development_yearly.csv',rows)
            dump(ART/f'logs/{rid}_config.json',cfg)
            dump(result,dict(run_id=rid,completed_at_utc=now(),seconds=time.perf_counter()-started))
            print(f"{stage} {fold} {cid}: macro AP={np.mean([r['AP'] for r in rows]):.4f}; {time.perf_counter()-started:.1f}s",flush=True)
        except Exception as e:
            with (ART/'logs/failures.jsonl').open('a') as f:
                f.write(json.dumps(dict(run_id=rid,error=repr(e),traceback=traceback.format_exc(),at_utc=now()))+'\n')
            raise
    dump(ART/f'logs/config_{cid}.json',cfg)
    return cid

def summarize():
    t=pd.read_csv(PUBLIC/'development_yearly.csv')
    keys=['config_id','family','feature_set','stage']
    summary=t.groupby(keys,dropna=False).agg(macro_year_AP=('AP','mean'),macro_ROC_AUC=('ROC_AUC','mean'),macro_recall_top5=('macro_recall_top5','mean'),worst_year_AP=('AP','min'),score_years=('year','nunique')).reset_index()
    cost=t.drop_duplicates(['config_id','run_id']).groupby(keys,dropna=False).agg(mean_fit_seconds=('fit_seconds','mean'),mean_features=('features','mean')).reset_index()
    summary=summary.merge(cost,on=keys,validate='one_to_one')
    summary=summary.sort_values(['macro_year_AP','mean_fit_seconds','config_id'],ascending=[False,True,True])
    summary.to_csv(PUBLIC/'development_summary.csv',index=False)
    return summary

def pilot():
    state('pilot',complete=False)
    df=load_data(); f=build_features(df,'S1'); tr=df.year<=2011
    rows=[]
    pilot_id='pilot_'+str(int(time.time()))
    for family in ['logistic','xgboost','mlp','ft_transformer']:
        cfg=base_config(family,max_epochs=3,patience=2,n_estimators=30)
        p=fit_pipeline(df.loc[tr],f.loc[tr],cfg,pilot_id+'_'+family)
        rows.append(dict(family=family,**p.metadata)); save_pipeline(p,ART/f'models/{pilot_id}_{family}.joblib')
        print('Pilot',family,p.metadata['fit_seconds'],flush=True)
    dump(PUBLIC/'pilot_cost.json',rows)

def develop(stage='all'):
    df=load_data(); assert df.year.max()==2012
    feature_registry().to_csv(PUBLIC/'feature_registry.csv',index=False)
    if stage in ['all','core']:
        state('development_core')
        families=['dummy','altman','logistic','random_forest','xgboost']+NEURAL
        if importlib.util.find_spec('catboost'): families+=['catboost']
        for fs in ['S0','S1']:
            for family in families:
                run_config(df,base_config(family,fs),'core_recipe')
        summarize()
    if stage in ['all','search']:
        state('development_search')
        summary=summarize(); core=summary[(summary.stage=='core_recipe') & (summary.feature_set=='S1') & (summary.score_years==7)]
        ml=core[~core.family.isin(NEURAL+['dummy','altman'])].iloc[0].family
        dl=core[core.family.isin(NEURAL)].iloc[0].family
        dump(ROOT/'configs/development_search_families.json',dict(selected_at_utc=now(),ML=ml,DL=dl,selection='S1 equal-weight seven-year development AP only'))
        grid=protocol()['implementation']['stage_2_candidates']
        for family in [ml,dl]:
            for change in grid['neural' if family in NEURAL else family]:
                run_config(df,base_config(family,'S1',**change),'bounded_search')
        summarize()
    if stage in ['all','ablations']:
        state('development_ablations')
        for family in ['xgboost','residual_mlp']:
            run_config(df,base_config(family,'S2'),'feature_ablation')
            for change in [dict(clip=False),dict(weight_alpha=.5),dict(weight_alpha=1),dict(sampling='ros'),dict(sampling='smote'),dict(selector='elastic_net'),dict(selector='model_aware')]:
                run_config(df,base_config(family,'S1',**change),'controlled_ablation')
        summarize()
    state('development_complete',complete=False)
    (ROOT/'NEXT_ACTIONS.md').write_text('# Next actions\n\nFreeze final configurations using development results, train final models/calibration/thresholds and untuned group stress artifacts before historical evaluation.\n')

def optional_sequences():
    """Run after the mandatory core, within the same training ledger and budget."""
    state('sequence_extension')
    df=load_data()
    for family in ['xgboost','mlp']:
        run_config(df,base_config(family,'LAGS'),'sequence_control')
    cfg=base_config('gru','LAGS',width=32,depth=1,dropout=.15,max_epochs=25,patience=4,batch_size=1024,learning_rate=.001,weight_decay=.0001)
    run_config(df,cfg,'sequence_extension')
    summarize()
    state('development_and_sequences_complete',complete=False)
