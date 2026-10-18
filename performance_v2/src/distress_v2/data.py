"""Input-only feature construction and staged label access."""
import json
import numpy as np
import pandas as pd
from .common import ROOT,SOURCE,ART,sha,protocol
from distress_ml.features import build_features as v1_features,lag_values,RATIOS,ratio

RAW=[f'X{i}' for i in range(1,19)]
LEGACY13=['log_market_value','long_term_debt_to_assets','log_total_assets','X8','X6','X3','X15','working_capital','current_ratio','operating_expense_to_revenue','ebit_to_assets','gross_margin','X17']
_data=None
_features={}

def load(role='search'):
    global _data
    path=SOURCE/'american_bankruptcy.csv'
    assert sha(path)==protocol()['data']['sha256']
    if _data is None:
        names=pd.read_csv(path,nrows=0).columns
        _data=pd.read_csv(path,usecols=[c for c in names if c!='status_label'])
        _data['row_id']=np.arange(len(_data));_data['y']=np.nan
        assert not _data.duplicated(['company_name','year']).any()
    cutoff={'search':2013,'p1':2012,'confirmation':2014,'historical':2018}[role]
    if role=='confirmation':assert (ROOT/'configs/selection_freeze.json').exists()
    if role=='historical':assert (ROOT/'configs/artifact_freeze.json').exists()
    keep=_data.year<=cutoff
    if _data.loc[keep,'y'].notna().all():return _data.copy()
    skip=set((_data.index[~keep]+1).tolist())
    labels=pd.read_csv(path,usecols=['status_label'],skiprows=lambda i:i in skip).status_label
    assert set(labels.unique())=={'alive','failed'}
    _data.loc[keep,'y']=labels.eq('failed').astype(int).to_numpy()
    return _data.copy()

def legacy_features(df):
    f=df[RAW].astype(float).copy();f['working_capital']=df.X1-df.X14
    for name,(numerator,den,_) in RATIOS.items():
        n=df.X1-df.X14 if numerator=='X1-X14' else df[numerator]
        rename={'liabilities_to_assets':'debt_to_assets','operating_expense_to_sales':'operating_expense_to_revenue'}.get(name,name)
        f[rename]=(n/df[den].replace(0,np.nan)).replace([np.inf,-np.inf],np.nan)
    f['revenue_to_assets']=df.X16/df.X10.replace(0,np.nan)
    for c,name in [('X10','log_total_assets'),('X8','log_market_value')]:
        f[name]=np.log1p(df[c].where(df[c]>-1))
    f['altman_z_score']=1.2*f.working_capital_to_assets+1.4*f.retained_earnings_to_assets+3.3*f.ebit_to_assets+.6*f.market_value_to_liabilities+f.sales_to_assets
    assert f.shape[1]==35
    return f

def features(df,name):
    if name in _features:return _features[name]
    raw=df[[c for c in RAW if c!='X16']].astype(float)
    slog=np.sign(raw)*np.log1p(abs(raw));slog.columns=['slog_'+c for c in raw]
    if name=='RAW17':f=raw
    elif name=='RAW18':f=df[RAW].astype(float)
    elif name=='SLOG17':f=slog
    elif name in ['LEGACY35','LEGACY33','INNER_L1','LEGACY_FIXED13']:
        f=legacy_features(df)
        if name=='LEGACY33':f=f.drop(columns=['X16','revenue_to_assets'])
        if name=='LEGACY_FIXED13':f=f[LEGACY13]
    elif name in ['RATIO_CORE','SLOG_RATIO','DYNAMIC_COMPACT']:
        s=v1_features(df,'S1');names=list(RATIOS)
        r=s[names+[c+'_invalid' for c in names]].copy()
        r['log_total_assets']=np.log1p(df.X10.where(df.X10>0));r['log_market_value']=np.log1p(df.X8.where(df.X8>0))
        f=pd.concat([slog,r],axis=1) if name=='SLOG_RATIO' else r
        if name=='DYNAMIC_COMPACT':
            dynamic=v1_features(df,'S2');cols=[c for c in dynamic if c.startswith(('delta1_','std3_','log_growth_','loss_count','observed_year','missing_t'))]
            f=pd.concat([r,dynamic[cols]],axis=1)
    elif name in ['LAGS_RAW3','LAGS_SLOG3']:
        base=raw if name=='LAGS_RAW3' else slog
        out={}
        for k in [2,1,0]:
            for c in base:out[f'lag{k}_{c}']=base[c] if k==0 else lag_values(df,base[c],k)
            out[f'observed_lag{k}']=1. if k==0 else lag_values(df,np.ones(len(df)),k).notna().astype(float)
        f=pd.DataFrame(out,index=df.index)
    elif name in ['V1_S0','V1_S1','V1_LAGS']:
        f=v1_features(df,name.removeprefix('V1_'))
    else:raise ValueError(name)
    f=f.replace([np.inf,-np.inf],np.nan)
    assert len(f)==len(df) and not any(c in f for c in ['year','row_id','company_name','y'])
    _features[name]=f
    return f

