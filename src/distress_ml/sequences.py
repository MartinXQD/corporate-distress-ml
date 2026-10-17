"""Optional true annual-trajectory GRU with explicit calendar padding masks."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import torch
from .common import ART,fit_event,dump
from .features import build_features
from .data import RAW
from .preprocessing import Preprocessor
from .splits import inner_masks,row_hash
from .neural import fit_neural,predict_logits

S0=[c for c in RAW if c!='X16']

def snapshot_from_lags(f,lag):
    a=f[[f'lag{lag}_{c}' for c in S0]].copy(); a.columns=S0
    return a

def sequence_tensor(f,pp):
    """Keep each of t-2,t-1,t; missing calendar slots are zero with mask 0."""
    steps=[]
    for lag in [2,1,0]:
        z=pp.transform(snapshot_from_lags(f,lag))
        observed=f[f'observed_lag{lag}'].to_numpy(dtype=np.float32)
        z[observed==0]=0
        steps.append(np.column_stack([z,observed]))
    return np.stack(steps,axis=1).astype(np.float32)

class SequencePipeline:
    def __init__(self,cfg,pp,model):
        self.cfg=cfg; self.preprocessor=pp; self.model=model
        self.columns=list(pp.columns)+['observed_calendar_slot']
        self.selected=S0
    def predict_raw(self,df,all_features=None):
        f=all_features if all_features is not None else build_features(df,'LAGS')
        return predict_logits(self.model,sequence_tensor(f,self.preprocessor))

def fit_sequence_pipeline(df,features,cfg,tag):
    started=time.perf_counter(); tr,stop=inner_masks(df); y=df.y.to_numpy()
    pp=Preprocessor(True,True).fit(snapshot_from_lags(features.loc[tr],0))
    x=sequence_tensor(features.loc[tr],pp); sx=sequence_tensor(features.loc[stop],pp)
    a=fit_event(tag+'/GRU_stop',len(x),df.loc[tr,'company_name'],family='gru',max_year=int(df.loc[tr,'year'].max()),row_hash=row_hash(df.loc[tr]))
    m,h,best=fit_neural('gru',x,y[tr],cfg,cfg['seed'],stop=(sx,y[stop]))
    torch.save(dict(config=cfg,state_dict=m.state_dict(),n_features=x.shape[-1]),ART/f'models/fit_{a:04d}.pt')
    pp=Preprocessor(True,True).fit(snapshot_from_lags(features,0)); x=sequence_tensor(features,pp)
    b=fit_event(tag+'/GRU_full_refit',len(x),df.company_name,family='gru',max_year=int(df.year.max()),row_hash=row_hash(df))
    m,h2,_=fit_neural('gru',x,y,cfg,cfg['seed'],epochs=best)
    torch.save(dict(config=cfg,state_dict=m.state_dict(),n_features=x.shape[-1]),ART/f'models/fit_{b:04d}.pt')
    pd.DataFrame([dict(stage='stop',**r) for r in h]+[dict(stage='full_refit',**r) for r in h2]).to_csv(ART/f'logs/{tag}_history.csv',index=False)
    p=SequencePipeline(cfg,pp,m)
    p.metadata=dict(run_id=tag,fit_rows=len(df),train_row_hash=row_hash(df),fit_seconds=time.perf_counter()-started,features=x.shape[-1],parameters=sum(v.numel() for v in m.parameters()),epochs=best,selected_features=S0,transformed_features=p.columns,fit_calls=[a,b],early_stop_year=int(df.year.max()),history_window=3,sequence_shape=list(x.shape),all_target_rows_retained=True)
    return p
