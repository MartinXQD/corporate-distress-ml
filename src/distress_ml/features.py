"""Economically registered snapshot and calendar-causal historical features."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .data import RAW

DESCRIPTIONS = ["Current assets","Cost of goods sold","Depreciation and amortization","EBITDA","Inventory","Net income","Receivables","Market value","Net sales","Total assets","Long-term debt","EBIT","Gross profit","Current liabilities","Retained earnings","Revenue (exact duplicate of net sales)","Total liabilities","Operating expenses"]
RATIOS = {
 "working_capital_to_assets": ("X1-X14","X10","liquidity"),
 "retained_earnings_to_assets": ("X15","X10","profitability"),
 "ebit_to_assets": ("X12","X10","profitability"),
 "ebitda_to_assets": ("X4","X10","profitability"),
 "net_income_to_assets": ("X6","X10","profitability"),
 "market_value_to_liabilities": ("X8","X17","market"),
 "sales_to_assets": ("X9","X10","efficiency"),
 "liabilities_to_assets": ("X17","X10","leverage"),
 "long_term_debt_to_assets": ("X11","X10","leverage"),
 "current_ratio": ("X1","X14","liquidity"),
 "gross_margin": ("X13","X9","profitability"),
 "operating_expense_to_sales": ("X18","X9","efficiency")}

def ratio(num, den):
    return (num/den.where(den > 0)).replace([np.inf,-np.inf], np.nan)

def lag_values(df, values, lag):
    """Exact calendar offset; no previous-observed-row substitution."""
    lookup = pd.Series(np.asarray(values),index=pd.MultiIndex.from_frame(df[["company_name","year"]]))
    keys = pd.MultiIndex.from_arrays([df.company_name,df.year-lag])
    return pd.Series(lookup.reindex(keys).to_numpy(),index=df.index)

def build_features(df, feature_set="S1"):
    """Pure input-only transformation; no fitted population parameters or labels."""
    if feature_set=='ALL':
        return pd.concat([build_features(df,'S2'),build_features(df,'LAGS')],axis=1)
    f = df[[x for x in RAW if x != "X16"]].astype(float).copy()
    if feature_set == "legacy18":
        return df[RAW].astype(float).copy()
    if feature_set == "S0":
        return f
    if feature_set == "LAGS":
        original=f.copy()
        out={}
        for lag in [2,1,0]:
            for c in original:
                out[f"lag{lag}_{c}"] = original[c] if lag==0 else lag_values(df,original[c],lag)
            out[f"observed_lag{lag}"] = np.ones(len(df)) if lag==0 else lag_values(df,np.ones(len(df)),lag).notna().astype(float)
        return pd.DataFrame(out,index=df.index)
    for c in list(f):
        f[f"slog_{c}"]=np.sign(f[c])*np.log1p(np.abs(f[c]))
    f["working_capital"]=df.X1-df.X14
    for name,(num,den,group) in RATIOS.items():
        f[name]=ratio(df.X1-df.X14 if num=="X1-X14" else df[num],df[den])
        f[name+"_invalid"]=f[name].isna().astype(float)
    f["altman_z"]=(1.2*f.working_capital_to_assets + 1.4*f.retained_earnings_to_assets + 3.3*f.ebit_to_assets + 0.6*f.market_value_to_liabilities + f.sales_to_assets)
    if feature_set=="S2":
        for c in ["net_income_to_assets","liabilities_to_assets","current_ratio","sales_to_assets"]:
            prev=lag_values(df,f[c],1)
            f[f"delta1_{c}"]=f[c]-prev
            hist=pd.concat([lag_values(df,f[c],k) for k in [0,1,2]],axis=1)
            f[f"mean3_{c}"]=hist.mean(axis=1)
            f[f"std3_{c}"]=hist.std(axis=1,ddof=0)
        for c in ["X10","X9"]:
            prev=lag_values(df,df[c],1)
            f[f"log_growth_{c}"]=np.log(df[c].where(df[c]>0))-np.log(prev.where(prev>0))
        loss=pd.concat([lag_values(df,(df.X6<0).astype(float),k) for k in [0,1,2]],axis=1)
        f["loss_count3"]=loss.sum(axis=1)
        f["observed_years3"]=loss.notna().sum(axis=1)
        f["missing_tminus1"]=lag_values(df,np.ones(len(df)),1).isna().astype(float)
    return f.replace([np.inf,-np.inf],np.nan)

def feature_group(c):
    base=c.removesuffix("_missing").removesuffix("_invalid")
    if base.startswith(("delta","mean3","std3","log_growth","loss_count","observed","missing_t")):
        return "dynamics"
    if base in RATIOS:
        return RATIOS[base][2]
    if base=="altman_z": return "composite"
    if base=="working_capital": return "liquidity"
    return "raw_amounts" if not base.startswith("slog") else "signed_log_amounts"

def feature_registry():
    rows=[]
    for c,desc in zip(RAW,DESCRIPTIONS):
        rows.append(dict(feature=c,formula=c,source_columns=c,interpretation=desc,group="raw_amounts",time="current fiscal year",invalid_rule="Raw finite values; units and currency unverified",set="legacy18" if c=="X16" else "S0"))
        if c!="X16": rows.append(dict(feature="slog_"+c,formula=f"sign({c})*log1p(abs({c}))",source_columns=c,interpretation="Compress scale while preserving negative values",group="signed_log_amounts",time="current fiscal year",invalid_rule="Finite raw input",set="S1"))
    for name,(num,den,grp) in RATIOS.items():
        rows.append(dict(feature=name,formula=f"({num})/{den}",source_columns=f"{num},{den}",interpretation=name.replace('_',' ')+"; current denominator",group=grp,time="current fiscal year",invalid_rule="Nonpositive denominator -> NaN and explicit invalid indicator; positive tiny values remain for train-only clipping",set="S1"))
        rows.append(dict(feature=name+"_invalid",formula=f"isnan({name})",source_columns=f"{num},{den}",interpretation="Invalid ratio flag",group=grp,time="current fiscal year",invalid_rule="Always finite binary",set="S1"))
    rows += [dict(feature="working_capital",formula="X1-X14",source_columns="X1,X14",interpretation="Current assets less current liabilities",group="liquidity",time="current fiscal year",invalid_rule="Finite",set="S1"),dict(feature="altman_z",formula="1.2*WC/TA+1.4*RE/TA+3.3*EBIT/TA+0.6*MV/TL+Sales/TA",source_columns="X1,X14,X10,X15,X12,X8,X17,X9",interpretation="Original public-manufacturer Z-score formula used descriptively across this unclassified universe",group="composite",time="current fiscal year",invalid_rule="NaN if any component invalid; training median",set="S1")]
    for c in ["net_income_to_assets","liabilities_to_assets","current_ratio","sales_to_assets"]:
        for pre,formula in [("delta1_","value(t)-value(t-1)"),("mean3_","mean of available t,t-1,t-2"),("std3_","population std of available t,t-1,t-2")]:
            rows.append(dict(feature=pre+c,formula=formula,source_columns=c,interpretation="Calendar-causal historical summary",group="dynamics",time="through target year only",invalid_rule="Missing exact calendar lag -> NaN; retain target row",set="S2"))
    for name,formula in [("log_growth_X10","log(X10_t)-log(X10_t-1), positive only"),("log_growth_X9","log(X9_t)-log(X9_t-1), positive only"),("loss_count3","sum(X6<0) over observed t,t-1,t-2"),("observed_years3","count observed exact t,t-1,t-2"),("missing_tminus1","1 if exact t-1 absent else 0")]:
        rows.append(dict(feature=name,formula=formula,source_columns="X6" if "loss" in name else "X10,X9,calendar history",interpretation="Historical change or coverage",group="dynamics",time="through target year only",invalid_rule="No backward fill",set="S2"))
    for lag in [2,1,0]:
        for c in [x for x in RAW if x!='X16']:
            rows.append(dict(feature=f'lag{lag}_{c}',formula=f'{c}(t-{lag})',source_columns=c,interpretation='Exact calendar snapshot for flattened/sequence controls',group='calendar_snapshot',time=f't-{lag}',invalid_rule='Absent slot -> NaN; observed mask retained; no target-row filtering',set='LAGS'))
        rows.append(dict(feature=f'observed_lag{lag}',formula=f'1 if exact t-{lag} exists else 0',source_columns='company/year alignment only',interpretation='Calendar-slot observation mask',group='calendar_snapshot',time=f't-{lag}',invalid_rule='Binary; current slot always observed',set='LAGS'))
    return pd.DataFrame(rows)
