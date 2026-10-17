"""Metrics with natural prevalence, annual budgets and paired company bootstrap."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, confusion_matrix, brier_score_loss, log_loss
from .splits import tie_key

def choose_threshold(y,p):
    pr,re,thresholds=precision_recall_curve(y,p)
    f1=2*pr[:-1]*re[:-1]/np.maximum(pr[:-1]+re[:-1],1e-15)
    best=np.flatnonzero(np.isclose(f1,np.max(f1),atol=1e-12))
    return float(thresholds[best[-1]])

def budgets(df,scores):
    result=[]
    scores=np.asarray(scores)
    for year in sorted(df.year.unique()):
        idx=np.flatnonzero(df.year.to_numpy()==year)
        y=df.y.to_numpy()[idx]
        keys=np.array([tie_key(c,t) for c,t in zip(df.iloc[idx].company_name,df.iloc[idx].year)])
        order=np.lexsort((keys,-scores[idx]))
        for b in [.01,.05,.10]:
            k=int(np.ceil(len(idx)*b)); tp=int(y[order[:k]].sum())
            result.append(dict(year=int(year),budget=b,rows=len(idx),positive_rows=int(y.sum()),prevalence=float(y.mean()),selected=k,true_positive=tp,precision=tp/k,recall=tp/y.sum() if y.sum() else np.nan,lift=(tp/k)/y.mean() if y.mean() else np.nan))
    return pd.DataFrame(result)

def metrics(df,scores,threshold=.5,ranking=None):
    y=df.y.to_numpy(); p=np.asarray(scores)
    pred=p>=threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    prob=np.clip(p,1e-7,1-1e-7)
    risk=p if ranking is None else np.asarray(ranking)
    b=budgets(df,risk)
    m=dict(rows=len(df),companies=int(df.company_name.nunique()),positive_rows=int(y.sum()),positive_companies=int(df.loc[df.y==1,'company_name'].nunique()),prevalence=float(y.mean()),AP=float(average_precision_score(y,risk)) if len(np.unique(y))>1 else np.nan,ROC_AUC=float(roc_auc_score(y,risk)) if len(np.unique(y))>1 else np.nan,precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn) if tp+fn else np.nan,F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp),alert_fraction=float(pred.mean()),Brier=float(brier_score_loss(y,prob)),log_loss=float(log_loss(y,prob,labels=[0,1])),threshold=float(threshold))
    for budget in [.01,.05,.10]:
        for col in ["precision","recall","lift"]:
            m[f"macro_{col}_top{int(budget*100)}"]=float(b.loc[b.budget==budget,col].mean())
    yearly=[average_precision_score(g.y,risk[df.index.get_indexer(g.index)]) for _,g in df.groupby("year") if g.y.nunique()>1]
    m["macro_year_AP"]=float(np.mean(yearly))
    return m

def reliability(df,p,bins=10):
    b=np.minimum((np.clip(p,0,1)*bins).astype(int),bins-1)
    return pd.DataFrame({"bin":b,"score":p,"y":df.y.to_numpy()}).groupby("bin").agg(count=("y","size"),positive_rows=("y","sum"),mean_score=("score","mean"),observed_rate=("y","mean")).reset_index()

def bootstrap_pairs(df, predictions, reference, repetitions=2000, seed=2026):
    """Integer company multiplicities preserve all firm-years and paired draws."""
    rng=np.random.default_rng(seed)
    companies,ids=np.unique(df.company_name.to_numpy(),return_inverse=True)
    y=df.y.to_numpy(); years=df.year.to_numpy()
    names=list(predictions)
    yearly_indices={year:np.flatnonzero(years==year) for year in sorted(set(years))}
    tie_keys=np.array([tie_key(c,t) for c,t in zip(df.company_name,df.year)])
    ordered={name:{year:ix[np.lexsort((tie_keys[ix],-np.asarray(p)[ix]))] for year,ix in yearly_indices.items()} for name,p in predictions.items()}
    samples={k:{m:[] for m in ['AP','ROC_AUC','macro_year_AP','macro_year_recall_top5']} for k in names}
    invalid=0
    for _ in range(repetitions):
        counts=np.bincount(rng.integers(0,len(companies),len(companies)),minlength=len(companies))
        w=counts[ids]
        if len(np.unique(y[w>0]))<2:
            invalid+=1; continue
        for name,p in predictions.items():
            p=np.asarray(p)
            samples[name]['AP'].append(average_precision_score(y,p,sample_weight=w))
            samples[name]['ROC_AUC'].append(roc_auc_score(y,p,sample_weight=w))
            annual=[]; recalls=[]
            for year,ix in yearly_indices.items():
                valid=w[ix]>0
                if not valid.any() or len(np.unique(y[ix][valid]))<2: continue
                annual.append(average_precision_score(y[ix],p[ix],sample_weight=w[ix]))
                # Same deterministic score/tie order as review rules; multiplicity is retained.
                order=ordered[name][year]
                repeated=np.repeat(order,w[order])
                k=int(np.ceil(.05*len(repeated)))
                recalls.append(y[repeated[:k]].sum()/y[repeated].sum())
            samples[name]['macro_year_AP'].append(np.mean(annual))
            samples[name]['macro_year_recall_top5'].append(np.mean(recalls))
    rows=[]
    for name in names:
        for metric in samples[name]:
            v=np.array(samples[name][metric]); ref=np.array(samples[reference][metric]); d=v-ref
            rows.append(dict(model=name,reference=reference,metric=metric,lower=float(np.quantile(v,.025)),upper=float(np.quantile(v,.975)),delta_mean=float(d.mean()),delta_lower=float(np.quantile(d,.025)),delta_upper=float(np.quantile(d,.975)),valid_draws=len(v),invalid_draws=invalid,repetitions=repetitions,unit="company",conditional_on_fitted_models=True))
    return pd.DataFrame(rows)
