"""One paired company bootstrap, reused across all frozen references."""
import numpy as np
import pandas as pd
from distress_ml.splits import tie_key

METRICS=['AP','ROC_AUC','macro_year_AP','macro_year_recall_top5']

def ordering(y,q):
    order=np.argsort(-q,kind='stable');scores=np.asarray(q)[order]
    starts=np.r_[0,np.flatnonzero(scores[1:]!=scores[:-1])+1]
    return order,starts,np.asarray(y)[order]

def weighted_ap_roc(order_info,w):
    order,starts,y=order_info;weight=np.asarray(w)[order]
    positive=np.add.reduceat(weight*y,starts);negative=np.add.reduceat(weight*(1-y),starts)
    p=positive.sum();n=negative.sum()
    if p<=0 or n<=0:return np.nan,np.nan
    tp=np.cumsum(positive);total=np.cumsum(positive+negative)
    precision=np.divide(tp,total,out=np.zeros_like(tp,dtype=float),where=total>0)
    ap=np.sum(positive*precision)/p
    # Descending groups: negatives below each group plus half its tied negatives.
    below=n-np.cumsum(negative)
    roc=np.sum(positive*(below+.5*negative))/(p*n)
    return float(ap),float(roc)

def bootstrap(df,predictions,references,repetitions=2000,seed=2026):
    y=df.y.to_numpy(int);years=df.year.to_numpy();companies,ids=np.unique(df.company_name.to_numpy(),return_inverse=True)
    rng=np.random.default_rng(seed);names=list(predictions);indices={year:np.flatnonzero(years==year) for year in sorted(set(years))}
    ties=np.array([tie_key(c,t) for c,t in zip(df.company_name,df.year)])
    prepared={name:ordering(y,np.asarray(q)) for name,q in predictions.items()};annual={};reviews={}
    for name,q in predictions.items():
        q=np.asarray(q);annual[name]={year:ordering(y[ix],q[ix]) for year,ix in indices.items()}
        reviews[name]={year:ix[np.lexsort((ties[ix],-q[ix]))] for year,ix in indices.items()}
    samples={name:[] for name in names};invalid=0;all_year_valid=0
    for _ in range(repetitions):
        counts=np.bincount(rng.integers(0,len(companies),len(companies)),minlength=len(companies));w=counts[ids]
        if (w*y).sum()==0 or (w*(1-y)).sum()==0:invalid+=1;continue
        complete_years=True
        for name in names:
            ap,roc=weighted_ap_roc(prepared[name],w);aps=[];recalls=[]
            for year,ix in indices.items():
                a,_=weighted_ap_roc(annual[name][year],w[ix])
                if not np.isfinite(a):complete_years=False;continue
                aps.append(a);order=reviews[name][year];weight=w[order]
                k=int(np.ceil(.05*weight.sum()));before=np.cumsum(weight)-weight;chosen=np.minimum(weight,np.maximum(k-before,0))
                recalls.append(float((chosen*y[order]).sum()/(weight*y[order]).sum()))
            # Preserve the fixed-year estimand. A resample lacking a class
            # in any year cannot silently average a different set of years.
            full_years=len(aps)==len(indices)
            samples[name].append([ap,roc,np.mean(aps) if full_years else np.nan,np.mean(recalls) if full_years else np.nan])
        all_year_valid+=int(complete_years)
    arrays={name:np.asarray(values) for name,values in samples.items()};rows=[]
    for reference in references:
        for name in names:
            for j,metric in enumerate(METRICS):
                v=arrays[name][:,j];ref=arrays[reference][:,j];valid=np.isfinite(v)&np.isfinite(ref);delta=(v-ref)[valid];v=v[valid]
                quantile=lambda values,q:float(np.quantile(values,q)) if len(values) else np.nan
                rows.append(dict(model=name,reference=reference,metric=metric,lower=quantile(v,.025),upper=quantile(v,.975),delta_mean=float(delta.mean()) if len(delta) else np.nan,delta_lower=quantile(delta,.025),delta_upper=quantile(delta,.975),valid_draws=len(v),invalid_draws=repetitions-len(v),invalid_global_draws=invalid,all_year_valid_draws=all_year_valid,repetitions=repetitions,unit='company',conditional_on_fitted_models=True,fixed_year_set_required=True))
    return pd.DataFrame(rows),arrays

def contract():
    from sklearn.metrics import average_precision_score,roc_auc_score
    from distress_ml.evaluation import bootstrap_pairs
    rng=np.random.default_rng(7);y=rng.integers(0,2,64);q=np.round(rng.normal(size=64),1)
    for _ in range(25):
        w=rng.integers(0,4,64);a,r=weighted_ap_roc(ordering(y,q),w)
        np.testing.assert_allclose(a,average_precision_score(y,q,sample_weight=w),atol=1e-12)
        np.testing.assert_allclose(r,roc_auc_score(y,q,sample_weight=w),atol=1e-12)
    d=pd.DataFrame(dict(y=y,year=np.repeat([2015,2016],32),company_name=[f'fixture_{i%32}' for i in range(64)]))
    predictions={'a':q,'b':q+rng.normal(size=64)}
    old=bootstrap_pairs(d,predictions,'a',repetitions=40,seed=2026).sort_values(['model','metric'])
    new,_=bootstrap(d,predictions,['a'],repetitions=40,seed=2026);new=new.sort_values(['model','metric'])
    np.testing.assert_allclose(old[['lower','upper','delta_mean','delta_lower','delta_upper']],new[['lower','upper','delta_mean','delta_lower','delta_upper']],atol=1e-12)
    rare=pd.DataFrame([(company,year,int(company in [0,1])) for company in range(8) for year in [2015,2016] if not (company==0 and year==2016) and not (company==1 and year==2015)],columns=['company_name','year','y'])
    q=np.arange(len(rare),dtype=float)
    sparse,_=bootstrap(rare,{'a':q},['a'],repetitions=128,seed=2026)
    valid=sparse.set_index('metric').valid_draws
    assert 0<valid['macro_year_AP']<valid['AP']<=128
    assert valid['macro_year_AP']==valid['macro_year_recall_top5']
    return dict(passed=True,weighted_metric_fixtures=25,tied_and_zero_weight_scores=True,paired_bootstrap_draws_compared=40,rare_class_fixed_year_draws=128,annual_single_class_draws_excluded_without_changing_years=True,reference='Preserved V1 bootstrap on complete-class fixtures and sklearn metrics; sparse-year correction explicitly verified',absolute_tolerance=1e-12,performance_evidence=False)
