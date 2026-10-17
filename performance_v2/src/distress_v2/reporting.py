"""Concise evidence-first reports; no gain wording before numerical verification."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .common import ROOT,V1,ART,RESULT,dump,now,hours,digest

def context():
    assert json.loads((RESULT/'verification.json').read_text())['passed']
    names=['historical_2015_2018','historical_by_year','trials','development_by_year','selection_trace','confirmation_2014','seeds','paired_company_intervals','ensemble_development','seen_unseen']
    c={k:pd.read_csv(RESULT/f'{k}.csv') for k in names}
    c['selected']=json.loads((ROOT/'configs/selection_freeze.json').read_text())
    c['main']=c['historical_2015_2018'].set_index('name')
    c['bridge']=c['development_by_year'][c['development_by_year'].stage=='bridge']
    return c

def delta_table(c):
    rows=[];archive=pd.read_csv(RESULT/'legacy_archived_metrics.csv')
    for r in archive.itertuples():
        rows.append(dict(recipe='V0 L1' if 'lasso' in r.feature_set else 'V0 base',version='V0',fit_cutoff=2011,input='fixed13/std01' if 'lasso' in r.feature_set else 'RAW18/std01',selected='No' if 'lasso' in r.feature_set else 'Yes',seeds=1,annual_AP=np.nan,pooled_AP=r.test_pr_auc,ROC=r.test_roc_auc,R5=np.nan,seconds=np.nan,identity='Archive only'))
    h=pd.read_csv(V1/'results/public/historical_metrics.csv');h=h[h.score_type=='calibrated']
    for family,feature,label in [('random_forest','S1','V1 RF'),('residual_mlp','S1','V1 DL'),('xgboost','LAGS','V1 lag XGB'),('catboost','S0','V1 Cat raw')]:
        if label in ['V1 RF','V1 DL']:
            selected_id=json.loads((V1/'configs/final_freeze.json').read_text())['representatives']['ML' if label=='V1 RF' else 'DL'];d=h[h.config_id==selected_id]
        else:d=h[(h.family==family)&(h.feature_set==feature)]
        r=d.iloc[0];rows.append(dict(recipe=label,version='V1',fit_cutoff=2012,input=feature,selected='Yes' if label in ['V1 RF','V1 DL'] else 'No',seeds=1,annual_AP=r.macro_year_AP,pooled_AP=r.AP,ROC=r.ROC_AUC,R5=r.macro_recall_top5,seconds=r.fit_seconds,identity='Saved V1'))
    selected=c['selected'];keys=list(dict.fromkeys(['baseline_legacy_strong_inner_l1','baseline_v1_lags_xgb','baseline_v1_raw_catboost',selected['overall'],selected['DL']]))
    for key in keys:
        row=c['main'].loc['P2_seed_mean__'+key];config=json.loads((ROOT/f'configs/trial_{key}.json').read_text())
        short={'baseline_legacy_strong_inner_l1':'P2 old L1','baseline_v1_lags_xgb':'P2 lag XGB','baseline_v1_raw_catboost':'P2 Cat raw'}.get(key,'P2 overall' if key==selected['overall'] else 'P2 DL')
        component=c['historical_2015_2018'];cost=component[(component.candidate==key)&(component.channel=='P2')].fit_seconds.sum()
        rows.append(dict(recipe=short,version='P2',fit_cutoff=2014,input=config['features']+'/'+config['preprocess'],selected='Yes' if key in [selected['overall'],selected['DL']] else 'Fixed',seeds=int(row.seed_count),annual_AP=row.macro_AP,pooled_AP=row.AP,ROC=row.ROC_AUC,R5=row.recall5,seconds=cost,identity='New seed mean'))
    if selected['blend']['admitted']:
        r=c['main'].loc['P2_fixed_rank_blend'];rows.append(dict(recipe='P2 blend',version='P2',fit_cutoff=2014,input='annual ranks',selected='Yes',seeds=5,annual_AP=r.macro_AP,pooled_AP=r.AP,ROC=r.ROC_AUC,R5=r.recall5,seconds=np.nan,identity='Frozen blend'))
    table=pd.DataFrame(rows);table.to_csv(RESULT/'baseline_bridge.csv',index=False);return table

def export_ablations(c):
    b=c['bridge'].copy();b['comparison_scope']='Single 2012 development year, fixed seed; mechanism diagnostic, not independent historical evidence'
    b.to_csv(RESULT/'feature_preprocess_ablation.csv',index=False)
    rows=[]
    for p in (ART/'logs').glob('*_complete.json'):
        if p.name in ['historical_complete.json']:continue
        d=json.loads(p.read_text())
        if not d.get('tag','').startswith('bridge_'):continue
        s=d['final_stats'];rows.append(dict(candidate=d['tag'],actual_updates=s.get('actual_updates'),best_steps=d['inner_stats'].get('best_steps'),positive_exposures=s.get('positive_exposures'),negative_exposures=s.get('negative_exposures'),train_AP=s.get('train_AP'),train_BCE=s.get('train_BCE'),fit_seconds=d.get('fit_seconds'),parameters=s.get('parameters')))
    pd.DataFrame(rows).to_csv(RESULT/'training_budget_ablation.csv',index=False)
    costs=[]
    for item in json.loads((ROOT/'configs/final_inventory.json').read_text()):
        d=json.loads((ROOT/item['path']).with_suffix('.json').read_text());costs.append(dict(name=item['name'],family=d['config']['family'],channel=item['channel'],fit_seconds=d.get('fit_seconds'),parameters=d['final_stats'].get('parameters'),updates=d['final_stats'].get('actual_updates'),rounds=d['final_stats'].get('best_rounds'),input_dimensions=len(d.get('transformed_columns',d['columns'])),fit_rows=d['fit_rows']))
    pd.DataFrame(costs).to_csv(RESULT/'costs.csv',index=False)
    return pd.DataFrame(rows)

def claims():
    rows=[]
    for path in RESULT.glob('*.csv'):
        if path.name=='claims.csv':continue
        table=pd.read_csv(path)
        for i,r in table.iterrows():
            for column in table.select_dtypes(include='number'):
                if pd.notna(r[column]):rows.append(dict(claim_id=f'{path.stem}:{i}:{column}',value=float(r[column]),metric=column,file=str(path.relative_to(ROOT)),data_row=i+1,row_filter=json.dumps({k:str(r[k]) for k in ['name','candidate','trial_id','model','reference','metric','seed','year','channel','recipe','version'] if k in r},sort_keys=True),qualification='See channel/source identity: archives, bridge, development, confirmation and exposed historical evaluation are distinct. Not an annual PD or causal result.'))
    pd.DataFrame(rows).to_csv(RESULT/'claims.csv',index=False)
