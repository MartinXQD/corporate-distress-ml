"""Measured fold-local fitting, equal-update refits, and checkpoint artifacts."""
import copy
import json
import math
import random
import subprocess
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score,roc_auc_score,log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from scipy.special import expit,logit
from .common import ROOT,ART,RESULT,dump,fit_event,digest,now,time_guard,hours
from .preprocessing import NumericTransform
from .networks import NumericNetwork,logits,probabilities
from distress_ml.preprocessing import Preprocessor as V1Transform,resample_train

NN=['legacy_residual','mlp','num_mlp','tabm','official_ft','v1_residual']

def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)
    torch.use_deterministic_algorithms(True)

def sample_weights(df,cfg):
    y=df.y.to_numpy(int);w=np.ones(len(df))
    alpha=cfg.get('weight_alpha',0)
    w[y==1]*=((len(y)-y.sum())/y.sum())**alpha
    if cfg.get('company_beta',0):
        factor=df.company_name.map(df.company_name.value_counts()).to_numpy(dtype=float)**(-cfg['company_beta']);w*=factor/factor.mean()
    if cfg.get('half_life'):
        factor=2.**(-(df.year.max()-df.year.to_numpy())/cfg['half_life']);w*=factor/factor.mean()
    return w

def make_preprocessor(cfg):
    if cfg.get('preprocess')=='v1':return V1Transform(cfg.get('clip',True),cfg['family'] in NN+['logistic'] or cfg.get('sampling','none')!='none')
    return NumericTransform(cfg.get('preprocess','std01'),indicators=cfg.get('indicators',True))

def selected_columns(x,df,cfg,tag):
    if cfg['features']!='INNER_L1':return list(x.columns),[]
    cache=ART/'cache'/('l1_'+digest(dict(rows=sorted(df.row_id.tolist()),seed=cfg['seed'],columns=list(x),mechanism='std01_l1_C5_ROC_parsimony_v1'))[:24]+'.json')
    if cache.exists():
        stored=json.loads(cache.read_text());return stored['selected'],stored['details']
    end=df.year.max();tr=df.year<end;va=~tr;y=df.y.to_numpy(int)
    pp=NumericTransform('std01',indicators=False).fit(x.loc[tr]);a=pp.transform(x.loc[tr]);b=pp.transform(x.loc[va]);rows=[]
    for C in [.001,.003,.01,.03,.1]:
        fit_event(tag+'/l1_inner',df.loc[tr],cfg,C=C)
        m=LogisticRegression(C=C,penalty='l1',solver='liblinear',max_iter=500,tol=1e-4,random_state=cfg['seed']).fit(a,y[tr])
        rows.append(dict(C=C,nonzero=int((abs(m.coef_[0])>1e-7).sum()),ROC=float(roc_auc_score(y[va],m.predict_proba(b)[:,1]))))
    eligible=[r for r in rows if r['ROC']>=max(v['ROC'] for v in rows)-.01 and r['nonzero']>0]
    best=min(eligible,key=lambda r:(r['nonzero'],r['C'])) if eligible else max(rows,key=lambda r:r['ROC'])
    pp=NumericTransform('std01',indicators=False).fit(x);fit_event(tag+'/l1_refit',df,cfg,C=best['C'])
    m=LogisticRegression(C=best['C'],penalty='l1',solver='liblinear',max_iter=700,tol=1e-4,random_state=cfg['seed']).fit(pp.transform(x),y)
    names=[name for name,w in zip(pp.output_columns,m.coef_[0]) if abs(w)>1e-7]
    chosen=names or list(x.columns);dump(cache,dict(selected=chosen,details=rows,training_row_hash=digest(sorted(df.row_id.tolist())),seed=cfg['seed'],reuse='Identical train-role selector; no new fit call'))
    return chosen,rows

def train_neural(x,df,cfg,tag,n_cont,stop=None,fixed_steps=None,schedule_horizon=None):
    seed(cfg['seed']);started=time.monotonic();y=df.y.to_numpy(int);orig_n=len(y)
    weights=sample_weights(df,cfg).astype(np.float32)
    method=cfg.get('sampling','none')
    assert method=='none' or (not cfg.get('weight_alpha',0) and not cfg.get('company_beta',0) and not cfg.get('half_life'))
    if method!='none':x,y=resample_train(x,y,method,cfg['seed']);weights=np.ones(len(y),np.float32)
    model=NumericNetwork(x.shape[1],n_cont,cfg,x_for_bins=x)
    # Importing the preserved V1 module may set the legacy thread default.
    torch.set_num_threads(cfg.get('threads',2))
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.get('learning_rate',.001),weight_decay=cfg.get('weight_decay',.0002))
    batch=cfg.get('batch_size',512);epoch_steps=math.ceil((len(y) if cfg.get('sampling_budget')=='epoch_equivalent' else orig_n)/batch)
    max_steps=fixed_steps or cfg.get('max_epochs',200)*epoch_steps
    horizon=schedule_horizon or max_steps
    fit_event(tag,df,cfg,original_rows=orig_n,resampled_rows=len(y),optimizer_max_steps=max_steps,positive_weight=((orig_n-df.y.sum())/df.y.sum())**cfg.get('weight_alpha',0),positive_fraction=float(y.mean()))
    xt=torch.from_numpy(x);yt=torch.tensor(y,dtype=torch.float32);wt=torch.from_numpy(weights)
    rng=np.random.default_rng(cfg['seed']);perm=rng.permutation(len(y));cursor=0
    history=[];best=-np.inf;best_state=None;best_steps=0;stale=0;updates=0;exposure=0;negative_exposure=0;total_loss=0.;norms=[]
    probe=np.linspace(0,orig_n-1,min(orig_n,4096),dtype=int)
    # Probe the natural training distribution even when optimization resamples.
    natural_x=x[:orig_n];natural_y=df.y.to_numpy(int)
    while updates<max_steps:
        model.train()
        for _ in range(min(epoch_steps,max_steps-updates)):
            if cursor>=len(perm):perm=rng.permutation(len(y));cursor=0
            ix=perm[cursor:cursor+batch];cursor+=batch
            xb=xt[ix];yb=yt[ix];wb=wt[ix]
            step=updates+1;mult=1.
            if cfg.get('schedule','fixed')=='cosine':
                warm=max(1,int(.05*horizon));mult=min(1.,step/warm)*(.05+.95*.5*(1+math.cos(math.pi*min(step,horizon)/horizon)))
            for g in optimizer.param_groups:g['lr']=cfg.get('learning_rate',.001)*mult
            optimizer.zero_grad(set_to_none=True);z=model(xb)
            if z.ndim==2:
                target=yb[:,None].expand_as(z)
                loss=(torch.nn.functional.binary_cross_entropy_with_logits(z,target,reduction='none')*wb[:,None]).mean()
            else:
                assert z.shape==yb.shape
                loss=(torch.nn.functional.binary_cross_entropy_with_logits(z,yb,reduction='none')*wb).mean()
            if cfg.get('pair_lambda',0):
                assert method=='none'
                pair_losses=[]
                local_years=df.year.to_numpy()[ix]
                for year in np.unique(local_years):
                    positive=np.flatnonzero((local_years==year)&(y[ix]==1));negative=np.flatnonzero((local_years==year)&(y[ix]==0))
                    if len(positive) and len(negative):
                        a=rng.choice(positive,size=min(16,len(positive)),replace=True);b=rng.choice(negative,size=len(a),replace=True)
                        pair_losses.append(torch.nn.functional.softplus(-(z[a]-z[b])).mean())
                if pair_losses:loss=loss+cfg['pair_lambda']*torch.stack(pair_losses).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite optimization loss')
            loss.backward()
            clip=cfg.get('gradient_clip',5.)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf') if clip is None else clip)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
            optimizer.step();updates+=1;exposure+=int(y[ix].sum());negative_exposure+=int((y[ix]==0).sum());total_loss+=loss.item();norms.append(float(norm))
        pred=probabilities(model,natural_x[probe]);loss_probe=log_loss(natural_y[probe],np.clip(pred,1e-7,1-1e-7),labels=[0,1])
        row=dict(epoch=len(history)+1,updates=updates,positive_exposures=exposure,negative_exposures=negative_exposure,optimization_loss=total_loss/updates,train_probe_AP=float(average_precision_score(natural_y[probe],pred)),train_probe_BCE=loss_probe,train_probe_rows=len(probe),gradient_norm_mean=float(np.mean(norms[-epoch_steps:])),saturated_fraction=float(np.mean((pred<1e-4)|(pred>1-1e-4))),learning_rate=optimizer.param_groups[0]['lr'],seconds=time.monotonic()-started)
        if stop is not None:
            sx,sy=stop;q=probabilities(model,sx);score=float(average_precision_score(sy,q));row.update(stop_AP=score,stop_BCE=float(log_loss(sy,np.clip(q,1e-7,1-1e-7),labels=[0,1])))
            if score>best+1e-5:best=score;best_steps=updates;best_state=copy.deepcopy(model.state_dict());stale=0
            else:stale+=1
        history.append(row)
        pd.DataFrame(history).to_csv(ART/f'logs/{tag.replace("/","__")}_curve.csv',index=False)
        time_guard(7.3)
        if time.monotonic()-started>cfg.get('per_fit_seconds',720):raise TimeoutError('Per-fit time limit; incomplete candidate retained, not a completed trial')
        if stop is not None and stale>=cfg.get('patience',25):break
    if stop is not None:
        assert best_state is not None;model.load_state_dict(best_state)
    else:best_steps=updates
    model.eval()
    full=probabilities(model,natural_x)
    stats=dict(best_steps=int(best_steps),actual_updates=updates,best_epoch=int(math.ceil(best_steps/epoch_steps)),epoch_steps=epoch_steps,schedule_horizon=horizon,positive_exposures=exposure,negative_exposures=negative_exposure,weight_ESS=float(weights.sum()**2/(weights**2).sum()),weight_min=float(weights.min()),weight_max=float(weights.max()),train_AP=float(average_precision_score(natural_y,full)),train_BCE=float(log_loss(natural_y,np.clip(full,1e-7,1-1e-7),labels=[0,1])),parameters=sum(p.numel() for p in model.parameters()),history=history)
    return model,stats

def train_tree(x,df,cfg,tag,stop=None,rounds=None):
    fam=cfg['family'];seed(cfg['seed']);y=df.y.to_numpy(int);weights=sample_weights(df,cfg)
    fit_event(tag,df,cfg,weight_ESS=float(weights.sum()**2/(weights**2).sum()),weight_min=float(weights.min()),weight_max=float(weights.max()))
    thread=cfg.get('threads',2)
    if fam=='catboost':
        from catboost import CatBoostClassifier
        pars=dict(iterations=rounds or cfg.get('iterations',2000),depth=cfg.get('depth',5),learning_rate=cfg.get('learning_rate',.05),l2_leaf_reg=cfg.get('l2_leaf_reg',3.),random_strength=cfg.get('random_strength',1.),bootstrap_type=cfg.get('bootstrap_type','Bayesian'),loss_function='Logloss',eval_metric='PRAUC',thread_count=thread,random_seed=cfg['seed'],verbose=False,allow_writing_files=False)
        if pars['bootstrap_type']=='Bernoulli':pars['subsample']=cfg.get('subsample',.8)
        else:pars['bagging_temperature']=cfg.get('bagging_temperature',1.)
        m=CatBoostClassifier(**pars);m.fit(x,y,sample_weight=weights,eval_set=stop,early_stopping_rounds=100 if stop is not None else None,use_best_model=stop is not None)
        best=m.get_best_iteration()+1 if stop is not None else pars['iterations'];history=m.get_evals_result()
    elif fam in ['xgboost','xgb_rank']:
        from xgboost import XGBClassifier,XGBRanker
        pars=dict(n_estimators=rounds or cfg.get('iterations',2000),max_depth=cfg.get('depth',4),learning_rate=cfg.get('learning_rate',.05),min_child_weight=cfg.get('min_child_weight',1.),subsample=cfg.get('subsample',.9),colsample_bytree=cfg.get('colsample_bytree',.9),reg_lambda=cfg.get('reg_lambda',1.),reg_alpha=cfg.get('reg_alpha',0.),tree_method='hist',n_jobs=thread,random_state=cfg['seed'],verbosity=0,early_stopping_rounds=100 if stop is not None else None)
        if fam=='xgb_rank':
            m=XGBRanker(**pars,objective=cfg.get('objective','rank:map'),eval_metric='map')
            order=np.argsort(df.year.to_numpy(),kind='stable');kwargs={}
            if stop is not None:
                sx,sy,years=stop;so=np.argsort(years,kind='stable');kwargs=dict(eval_set=[(sx[so],sy[so])],eval_qid=[years[so]])
            m.fit(x[order],y[order],qid=df.year.to_numpy()[order],verbose=False,**kwargs)
        else:
            m=XGBClassifier(**pars,eval_metric='aucpr');m.fit(x,y,sample_weight=weights,eval_set=[stop] if stop is not None else None,verbose=False)
        best=m.best_iteration+1 if stop is not None else pars['n_estimators'];history=m.evals_result() if stop is not None else {}
    elif fam in ['random_forest','extra_trees']:
        cls=RandomForestClassifier if fam=='random_forest' else ExtraTreesClassifier
        maximum=cfg.get('iterations',600);started=time.monotonic()
        m=cls(n_estimators=min(25,maximum),warm_start=True,max_depth=cfg.get('depth',10),min_samples_leaf=cfg.get('leaf',5),max_features=cfg.get('max_features',.8),bootstrap=cfg.get('bootstrap',fam=='random_forest'),random_state=cfg['seed'],n_jobs=thread)
        for count in range(25,maximum+25,25):
            time_guard(7.25)
            if time.monotonic()-started>cfg.get('per_fit_seconds',720):raise TimeoutError('Forest fit exceeded bounded compute; incomplete trial, not a smaller completed forest')
            m.n_estimators=min(count,maximum);m.fit(x,y,sample_weight=weights)
            dump(ART/f'logs/{tag.replace("/","__")}_forest_resource.json',dict(completed_trees=len(m.estimators_),requested_trees=maximum,seconds=time.monotonic()-started,at_utc=now(),guard='25-tree warm-start batches; same seeded tree sequence'))
        m.warm_start=False
        best=len(m.estimators_);history={}
    elif fam=='logistic':
        m=LogisticRegression(C=cfg.get('C',1.),max_iter=1500,random_state=cfg['seed']).fit(x,y,sample_weight=weights);best=int(m.n_iter_.max());history={}
    else:raise ValueError(fam)
    return m,dict(best_rounds=int(best),stopping_metric='CatBoost PRAUC' if fam=='catboost' else ('XGBoost aucpr' if fam=='xgboost' else ('XGBoost MAP' if fam=='xgb_rank' else 'none')),history=history)

class Fitted:
    def predict(self,x):
        a=self.pp.transform(x[self.selected])
        if self.cfg['family'] in NN:return probabilities(self.model,a)
        if self.cfg['family']=='xgb_rank':return self.model.predict(a)
        assert list(self.model.classes_)==[0,1]
        return self.model.predict_proba(a)[:,1]

def fit(df,x,cfg,tag):
    started=time.monotonic();time_guard(7.25);df=df.copy();assert df.y.notna().all()
    if cfg.get('window'):keep=df.year>df.year.max()-cfg['window'];df=df.loc[keep];x=x.loc[keep]
    internal=df.year==df.year.max();train=~internal;family=cfg['family'];stop_stats={};selector=[]
    if family in NN and cfg.get('fixed_training_steps'):
        names,selector=selected_columns(x,df,cfg,tag+'/fixed_select');pp=make_preprocessor(cfg).fit(x[names]);a=pp.transform(x[names]);fitted_df=df
        model,final_stats=train_neural(a,df,cfg,tag+'/fixed_updates',getattr(pp,'n_cont',len(pp.columns)),fixed_steps=cfg['fixed_training_steps'])
    elif family in NN+['catboost','xgboost','xgb_rank']:
        names,selector=selected_columns(x.loc[train],df.loc[train],cfg,tag+'/inner_select')
        pp=make_preprocessor(cfg).fit(x.loc[train,names]);a=pp.transform(x.loc[train,names]);b=pp.transform(x.loc[internal,names])
        if family in NN:
            inner,stop_stats=train_neural(a,df.loc[train],cfg,tag+'/stop',getattr(pp,'n_cont',len(pp.columns)),stop=(b,df.loc[internal,'y'].to_numpy(int)))
        else:
            stop=(b,df.loc[internal,'y'].to_numpy(int))
            if family=='xgb_rank':stop=(*stop,df.loc[internal,'year'].to_numpy())
            inner,stop_stats=train_tree(a,df.loc[train],cfg,tag+'/stop',stop=stop)
        dump(ART/f'logs/{tag.replace("/","__")}_inner.json',stop_stats)
        if cfg.get('refit_rule')=='keep_best':
            model=inner;final_stats=dict(stop_stats);fitted_df=df.loc[train]
        else:
            names,selector=selected_columns(x,df,cfg,tag+'/full_select');pp=make_preprocessor(cfg).fit(x[names]);a=pp.transform(x[names]);fitted_df=df
            if family in NN:
                steps=stop_stats['best_steps']
                if cfg.get('refit_rule')=='epochs':
                    full_rows=len(df)
                    if cfg.get('sampling_budget')=='epoch_equivalent' and cfg.get('sampling','none')!='none':
                        full_rows=2*int(df.y.astype(int).value_counts().max())
                    steps=stop_stats['best_epoch']*math.ceil(full_rows/cfg.get('batch_size',512))
                model,final_stats=train_neural(a,df,cfg,tag+'/refit',getattr(pp,'n_cont',len(pp.columns)),fixed_steps=steps,schedule_horizon=stop_stats['schedule_horizon'])
            else:model,final_stats=train_tree(a,df,cfg,tag+'/refit',rounds=stop_stats['best_rounds'])
    else:
        names,selector=selected_columns(x,df,cfg,tag+'/select');pp=make_preprocessor(cfg).fit(x[names]);a=pp.transform(x[names]);model,final_stats=train_tree(a,df,cfg,tag+'/fit');fitted_df=df
    p=Fitted();p.cfg=cfg;p.model=model;p.pp=pp;p.selected=names
    p.metadata=dict(tag=tag,config_hash=digest(cfg),fit_seconds=time.monotonic()-started,fit_rows=len(fitted_df),max_fit_year=int(fitted_df.year.max()),requested_fit_year=int(df.year.max()),row_hash=digest(sorted(fitted_df.row_id.tolist())),company_hash=digest(sorted(fitted_df.company_name.unique())),columns=pp.columns,selected_columns=names,inner_stop_year=int(df.year.max()),inner_stats={k:v for k,v in stop_stats.items() if k!='history'},final_stats={k:v for k,v in final_stats.items() if k!='history'},selector=selector,generated_at_utc=now())
    p.metadata['transformed_columns']=getattr(pp,'output_columns',pp.columns)
    p.metadata['git_commit_at_fit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    dump(ART/f'logs/{tag.replace("/","__")}_complete.json',p.metadata)
    return p

def save(p,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(p,path,compress=3)
    if p.cfg['family'] in NN:torch.save(p.model.state_dict(),path.with_suffix('.pt'))
    dump(path.with_suffix('.json'),dict(config=p.cfg,**p.metadata))

def load(path):
    path=Path(path).resolve();assert path.is_relative_to((ART/'models').resolve())
    assert path.with_suffix('.json').exists();p=joblib.load(path)
    if p.cfg['family'] in NN:p.model.eval()
    return p
