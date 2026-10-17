"""Structural audit of real local data; no model metrics or event inference."""
import json
import platform
import os
import importlib.metadata
import numpy as np
import pandas as pd
import psutil
import torch
from .common import ROOT,PUBLIC,ART,dump,now,sha
from .data import load_data,RAW
from .features import DESCRIPTIONS,build_features,feature_registry
from .splits import DEVELOPMENT,groups

def audit():
    df=load_data('audit')
    firms=df.groupby('company_name').agg(n=('year','size'),first=('year','min'),last=('year','max'),label=('y','first'))
    candidates=[p for p in [ROOT.parent/'american_bankruptcy.csv',ROOT.parent/'data/raw/american_bankruptcy.csv'] if p.is_file()]
    hashes={str(p.relative_to(ROOT.parent)):sha(p) for p in candidates}
    if len(set(hashes.values()))>1:raise ValueError('Local raw copies disagree')
    if not hashes:
        hashes={'authorized_DISTRESS_DATA_snapshot':__import__('distress_ml.common',fromlist=['protocol']).protocol()['implementation']['source_data_hash']}
    def basic(sub):
        return dict(rows=len(sub),companies=int(sub.company_name.nunique()),positive_rows=int(sub.y.sum()),positive_companies=int(sub.loc[sub.y==1,'company_name'].nunique()),positive_row_fraction=float(sub.y.mean()))
    summary=dict(status='read_only_audit_completed_no_new_model_results',raw_files_sha256=hashes,rows=len(df),columns=len(df.attrs['raw_columns']),companies=len(firms),year_min=int(df.year.min()),year_max=int(df.year.max()),missing_cells=int(df[RAW+['company_name','year','y']].isna().sum().sum()),numeric_nonfinite_cells=int((~np.isfinite(df[RAW])).sum().sum()),duplicate_company_year_rows=int(df.duplicated(['company_name','year']).sum()),duplicate_rows=int(df.drop(columns='row_id').duplicated().sum()),company_label_nunique_distribution={str(k):int(v) for k,v in df.groupby('company_name').y.nunique().value_counts().items()},row_label_counts={'failed':int(df.y.sum()),'alive':int((1-df.y).sum())},company_label_counts={'failed':int(firms.label.sum()),'alive':int((1-firms.label).sum())},exact_duplicate_feature_pairs=[(a,b) for i,a in enumerate(RAW) for b in RAW[i+1:] if df[a].equals(df[b])],negative_values_by_feature={c:int((df[c]<0).sum()) for c in RAW},zero_values_by_feature={c:int((df[c]==0).sum()) for c in RAW},sequence_audit=dict(firms_with_at_least_3_rows=int((firms.n>=3).sum()),firms_with_at_least_4_rows=int((firms.n>=4).sum()),firms_with_noncontiguous_year_coverage=int(((firms['last']-firms['first']+1)!=firms.n).sum())),by_year=[dict(year=int(y),**basic(g)) for y,g in df.groupby('year')],label_timing_verified=False,units_verified=False,historical_period_previously_inspected=True)
    summary['original_splits']={name:basic(df[df.year.between(lo,hi)]) for name,lo,hi in [('train_1999_2011',1999,2011),('validation_2012_2014',2012,2014),('historical_test_2015_2018',2015,2018)]}
    later=df[df.year>=2015]
    summary['historical_test_company_novelty_by_training_cutoff']={}
    for cutoff in [2011,2012,2014]:
        seen=set(df.loc[df.year<=cutoff,'company_name'])
        summary['historical_test_company_novelty_by_training_cutoff'][str(cutoff)]={'seen':basic(later[later.company_name.isin(seen)]),'unseen':basic(later[~later.company_name.isin(seen)])}
    dump(PUBLIC/'data_audit.json',summary)
    feature_registry().to_csv(PUBLIC/'feature_registry.csv',index=False)
    summaries=[]
    def summary(name,d):
        return dict(role=name,rows=len(d),companies=d.company_name.nunique(),positive_rows=int(d.y.sum()),positive_companies=d.loc[d.y==1,'company_name'].nunique(),prevalence=float(d.y.mean()),year_min=int(d.year.min()),year_max=int(d.year.max()))
    for name,lo,hi in [('final_fit',1999,2012),('calibration',2013,2013),('decision',2014,2014),('historical_previously_inspected',2015,2018)]:
        summaries.append(summary(name,df[df.year.between(lo,hi)]))
    for fold,end,lo,hi in DEVELOPMENT:
        summaries.append(summary(fold+'_fit',df[df.year<=end])); summaries.append(summary(fold+'_score',df[df.year.between(lo,hi)]))
    pd.DataFrame(summaries).to_csv(PUBLIC/'split_audit.csv',index=False)
    stats=df[RAW].describe(percentiles=[.005,.01,.5,.99,.995]).T
    stats['negative']=(df[RAW]<0).sum(); stats['zero']=(df[RAW]==0).sum(); stats['nonfinite']=(~np.isfinite(df[RAW])).sum()
    stats.index.name='feature'; stats.reset_index().to_csv(PUBLIC/'financial_field_audit.csv',index=False)
    relations=[]
    for name,a,b in [('sales_equals_revenue',df.X9,df.X16),('gross_profit_sales_less_cogs',df.X13,df.X9-df.X2),('ebit_ebitda_less_depreciation',df.X12,df.X4-df.X3)]:
        error=np.abs(a-b)
        relations.append(dict(relation=name,exact_rows=int((error==0).sum()),close_rows=int(np.isclose(a,b,rtol=1e-5,atol=.01).sum()),rows=len(df),median_abs_residual=float(error.median()),max_abs_residual=float(error.max())))
    pd.DataFrame(relations).to_csv(PUBLIC/'accounting_relationships.csv',index=False)
    g=groups(df)
    pd.DataFrame([summary('hash_group_'+str(i),df[g==i]) for i in range(3)]).to_csv(PUBLIC/'group_audit.csv',index=False)
    feat=build_features(df,'S2')
    pd.DataFrame({'feature':feat.columns,'missing':feat.isna().sum().to_numpy(),'missing_fraction':feat.isna().mean().to_numpy()}).to_csv(PUBLIC/'engineered_quality.csv',index=False)
    env=dict(created_at_utc=now(),platform=platform.platform(),cpu_count=psutil.cpu_count() or os.cpu_count(),memory_bytes=psutil.virtual_memory().total,device='cpu',MPS_available=torch.backends.mps.is_available(),CUDA_available=torch.cuda.is_available(),threads=2,versions={m:importlib.metadata.version(m) for m in ['numpy','pandas','scikit-learn','torch','xgboost','catboost','imbalanced-learn','reportlab','pypdf','nbformat','nbclient']},python=platform.python_version())
    dump(PUBLIC/'environment.json',env)
    return summaries
