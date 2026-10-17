"""One source of truth for fold fitting, early stopping, artifacts and scores."""
from __future__ import annotations
import copy
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from .common import ROOT,ART,PUBLIC,dump,fit_event,now,object_hash,protocol,sha
from .features import build_features
from .splits import row_hash,inner_masks
from .preprocessing import Preprocessor,positive_weight,resample_train
from .selection import select_features
from .classical import make_model,raw_logits,BOOSTING
from .neural import fit_neural,predict_logits,TabularNet

NEURAL=['mlp','residual_mlp','ft_transformer']

def base_config(family,feature_set='S1',**changes):
    p=protocol()['implementation']
    cfg=dict(family=family,feature_set=feature_set,clip=True,sampling='none',selector='none',weight_alpha=0,seed=42)
    cfg.update(p['models'].get(family,{}))
    if family in NEURAL: cfg.update(p['neural_training'])
    cfg.update(changes)
    return cfg

def config_id(cfg):
    return f"{cfg['family']}__{cfg['feature_set']}__{object_hash(cfg)[:10]}"

class FittedPipeline:
    def __init__(self,cfg,preprocessor=None,columns=None,model=None,selected=None):
        self.cfg=cfg; self.preprocessor=preprocessor; self.columns=columns; self.model=model; self.selected=selected
    def predict_raw(self,df,all_features=None):
        if self.cfg['family']=='dummy': return np.full(len(df),self.constant)
        if self.cfg['family']=='altman':
            z=build_features(df,'S1').altman_z.to_numpy()
            return -np.where(np.isfinite(z),z,self.median_z)
        f=all_features if all_features is not None else build_features(df,self.cfg['feature_set'])
        x=self.preprocessor.transform(f[self.selected])
        if self.cfg['family'] in NEURAL:
            return predict_logits(self.model,x)
        return raw_logits(self.model,x)
    def probability(self,df,features=None):
        return expit(self.predict_raw(df,features))

def fit_once(family,x,y,cfg,tag,df,stop=None,epochs=None):
    weight=positive_weight(y,cfg['weight_alpha'])
    xr,yr=resample_train(x,y,cfg['sampling'],cfg['seed'])
    call=fit_event(tag,len(xr),df.company_name,original_rows=len(y),positive_rows=int(y.sum()),weight=float(weight),sampling=cfg['sampling'],max_year=int(df.year.max()),row_hash=row_hash(df),family=family)
    if family in NEURAL:
        m,history,best=fit_neural(family,xr,yr,cfg,cfg['seed'],weight,stop,epochs)
        torch.save({'family':family,'n_features':x.shape[1],'config':cfg,'state_dict':m.state_dict(),'fit_call':call},ART/f'models/fit_{call:04d}.pt')
    else:
        m=make_model(family,cfg,cfg['seed'],weight,iterations=epochs,stopping=stop is not None)
        sw=np.where(yr==1,weight,1.)
        if stop is not None and family=='xgboost' and BOOSTING=='xgboost':
            m.fit(xr,yr,sample_weight=sw,eval_set=[stop],verbose=False)
            best=m.best_iteration+1
            ev=m.evals_result()['validation_0']['logloss']
            history=[dict(epoch=i+1,stop_loss=float(v)) for i,v in enumerate(ev)]
        elif stop is not None and family=='catboost':
            m.fit(xr,yr,sample_weight=sw,eval_set=stop,early_stopping_rounds=15)
            best=m.get_best_iteration()+1
            history=[dict(epoch=i+1,stop_loss=float(v)) for i,v in enumerate(m.get_evals_result()['validation']['Logloss'])]
        else:
            m.fit(xr,yr,sample_weight=sw); best=epochs or cfg.get('n_estimators',cfg.get('iterations',1)); history=[]
        joblib.dump(m,ART/f'models/fit_{call:04d}.joblib',compress=3)
    pd.DataFrame(history).to_csv(ART/f'logs/fit_{call:04d}_history.csv',index=False)
    return m,history,best,call

def fit_pipeline(df,features,cfg,tag):
    """Every learned choice sees only df, never an external score role."""
    started=time.perf_counter(); family=cfg['family']; y=df.y.to_numpy()
    if family=='gru':
        from .sequences import fit_sequence_pipeline
        return fit_sequence_pipeline(df,features,cfg,tag)
    if family in ['dummy','altman']:
        p=FittedPipeline(cfg)
        if family=='dummy': p.constant=float(logit(y.mean()))
        else: p.median_z=float(np.nanmedian(build_features(df,'S1').altman_z))
        p.metadata=dict(run_id=tag,config_id=config_id(cfg),fit_rows=len(df),train_row_hash=row_hash(df),fit_seconds=time.perf_counter()-started,features=0,parameters=1,epochs=0,selected_features=[],fit_calls=[])
        return p
    tr,stop=inner_masks(df)
    history=[]; fit_calls=[]
    fixed_length=None
    # Early stopping never consumes the development scoring years.
    if family in NEURAL+['xgboost','catboost']:
        selected,details=select_features(features.loc[tr],df.loc[tr],cfg['selector'],cfg['seed'],tag+'/stop_selector')
        pp=Preprocessor(cfg['clip'],True if family in NEURAL or cfg['sampling']!='none' else False).fit(features.loc[tr,selected])
        _,h,fixed_length,call=fit_once(family,pp.transform(features.loc[tr,selected]),y[tr],cfg,tag+'/stop',df.loc[tr],stop=(pp.transform(features.loc[stop,selected]),y[stop]))
        history.extend([dict(stage='stop',**r) for r in h]); fit_calls.append(call)
    selected,details=select_features(features,df,cfg['selector'],cfg['seed'],tag+'/full_selector')
    pp=Preprocessor(cfg['clip'],family in NEURAL+['logistic'] or cfg['sampling']!='none').fit(features[selected])
    model,h,epochs,call=fit_once(family,pp.transform(features[selected]),y,cfg,tag+'/full_refit',df,epochs=fixed_length)
    fit_calls.append(call); history.extend([dict(stage='full_refit',**r) for r in h])
    p=FittedPipeline(cfg,pp,pp.columns,model,selected)
    params=sum(v.numel() for v in model.parameters()) if family in NEURAL else (int(model.coef_.size) if hasattr(model,'coef_') else None)
    p.metadata=dict(run_id=tag,config_id=config_id(cfg),fit_rows=len(df),train_row_hash=row_hash(df),fit_seconds=time.perf_counter()-started,features=len(pp.columns),parameters=params,epochs=int(epochs),selected_features=selected,transformed_features=pp.columns,fit_calls=fit_calls,train_company_hash=object_hash(sorted(df.company_name.unique())),early_stop_year=int(df.year.max()))
    pd.DataFrame(history).to_csv(ART/f'logs/{tag}_history.csv',index=False)
    dump(ART/f'logs/{tag}_selection.json',details)
    return p

def save_pipeline(p,path):
    path=Path(path)
    if p.cfg['family'] in NEURAL:
        ckpt=path.with_suffix('.pt')
        torch.save(dict(family=p.cfg['family'],n_features=len(p.columns),config=p.cfg,state_dict=p.model.state_dict()),ckpt)
        model=p.model; p.model=None
        joblib.dump(p,path,compress=3); p.model=model
    else:
        joblib.dump(p,path,compress=3)
    dump(path.with_suffix('.json'),dict(**p.metadata,config=p.cfg,dataset_hash=protocol()['implementation']['source_data_hash'],protocol_hash=sha(ROOT/'configs/protocol_v1.yaml'),generated_at_utc=now()))

def load_pipeline(path):
    """Only load files generated by this project; never source archive pickles."""
    path=Path(path); p=joblib.load(path)
    if p.cfg['family'] in NEURAL:
        c=torch.load(path.with_suffix('.pt'),map_location='cpu',weights_only=True)
        p.model=TabularNet(c['family'],c['n_features'],c['config'])
        p.model.load_state_dict(c['state_dict']); p.model.eval()
    return p
