"""Independent saved-prediction arithmetic and frozen-pipeline reload checks."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
from scipy.stats import rankdata
from .common import ROOT,SOURCE,ART,RESULT,dump,sha,now
from .data import load,features
from .training import load as load_model,NN
from distress_ml.splits import groups,tie_key

def independent(df,q):
    y=df.y.to_numpy(int);annual=[];roc=[];recall=[]
    for year in sorted(df.year.unique()):
        ix=np.flatnonzero(df.year.to_numpy()==year)
        keys=np.array([tie_key(c,year) for c in df.iloc[ix].company_name])
        order=np.lexsort((keys,-q[ix]));n=int(np.ceil(.05*len(ix)))
        annual.append(average_precision_score(y[ix],q[ix]));roc.append(roc_auc_score(y[ix],q[ix]));recall.append(y[ix][order[:n]].sum()/y[ix].sum())
    return dict(AP=float(average_precision_score(y,q)),ROC_AUC=float(roc_auc_score(y,q)),macro_AP=float(np.mean(annual)),macro_ROC=float(np.mean(roc)),recall5=float(np.mean(recall)))

def verify():
    assert (ART/'historical_complete.json').exists()
    frozen=json.loads((ROOT/'configs/artifact_freeze.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in frozen['artifacts'].items())
    original=json.loads((ART/'audit/protected_sources.json').read_text())
    assert all((SOURCE/p).exists() and sha(SOURCE/p)==h for p,h in original.items()),'A protected V0/V1 file changed'
    df=load('historical');hist=df[df.year>=2015];table=pd.read_csv(RESULT/'historical_2015_2018.csv',float_precision='round_trip').set_index('name');checked=[];families={}
    for item in frozen['models']:
        d=pd.read_csv(ART/f'predictions/{item["name"]}_historical.csv.gz',float_precision='round_trip')
        expected=hist if item['channel']!='post_selection_company_stress' else hist[groups(hist)==item['group_fold']]
        assert set(d.row_id)==set(expected.row_id)
        sub=df.set_index('row_id').loc[d.row_id]
        assert np.array_equal(sub.y.to_numpy(),d.y_true.to_numpy())
        assert np.array_equal(sub.company_name.to_numpy(),d.company_id.to_numpy())
        assert np.array_equal(sub.year.to_numpy(),d.year.to_numpy())
        m=independent(sub,d.score.to_numpy())
        for k,v in m.items():assert np.isclose(v,table.loc[item['name'],k],atol=1e-9,rtol=1e-7),(item['name'],k)
        checked.append(item['name'])
        families[item['name']]=json.loads((ROOT/item['path']).with_suffix('.json').read_text())['config']['family']
    # Mean-seed and rank/blend rows use exactly the same stored row population.
    for name,row in table.iterrows():
        if row.channel not in ['P2_seed_ensemble','P2_annual_rank_seed_ensemble']:continue
        if name=='P2_fixed_rank_blend':path=ART/'predictions/P2_fixed_rank_blend_historical.csv.gz'
        else:path=ART/f'predictions/P2_seed_mean__{row.candidate}_historical.csv.gz'
        d=pd.read_csv(path,float_precision='round_trip');q=d.score.to_numpy()
        assert np.array_equal(hist.row_id,d.row_id)
        if row.channel=='P2_annual_rank_seed_ensemble' and name!='P2_fixed_rank_blend':
            for year in sorted(d.year.unique()):
                ix=np.flatnonzero(d.year.to_numpy()==year);q[ix]=rankdata(q[ix],method='average')/len(ix)
        for key,value in independent(hist,q).items():assert np.isclose(value,row[key],atol=1e-9,rtol=1e-7),(name,key)
    selected=[]
    for neural in [False,True]:
        items=[m for m in frozen['models'] if (families[m['name']] in NN)==neural and m['channel']=='P2']
        item=min(items,key=lambda x:sha(ROOT/x['path']))
        p=load_model(ROOT/item['path']);q=p.predict(features(df,p.cfg['features'])[df.year>=2015])
        d=pd.read_csv(ART/f'predictions/{item["name"]}_historical.csv.gz',float_precision='round_trip')
        np.testing.assert_allclose(q,d.raw_score.to_numpy(),atol=1e-6,rtol=1e-6)
        selected.append(dict(model=item['name'],neural=neural,max_reload_error=float(np.max(abs(q-d.raw_score.to_numpy())))))
    development_checks=verify_development(df)
    result=dict(passed=True,checked_at_utc=now(),saved_prediction_sets_verified=len(checked),development_prediction_sets_verified=development_checks,frozen_hashes_verified=len(frozen['artifacts']),protected_files_unchanged=len(original),same_source_row_alignment=True,primary_metrics_independently_recomputed=True,reloaded_models=selected,tolerance=1e-6)
    uncertainty(df,hist,table)
    dump(RESULT/'verification.json',result)
    print(json.dumps(result,indent=2))

def verify_development(df):
    """Recalculate every saved development fold, including invalidated evidence."""
    indexed=df.set_index('row_id');count=0
    for path in (ART/'logs').glob('fold_*.json'):
        row=json.loads(path.read_text())
        p=pd.read_csv(ROOT/row['prediction_path'],float_precision='round_trip')
        assert p.row_id.is_unique
        sub=indexed.loc[p.row_id]
        assert set(sub.year)=={row['year']} and row['year']<=2013
        np.testing.assert_array_equal(sub.y,p.y_true)
        np.testing.assert_array_equal(sub.company_name,p.company_id)
        m=independent(sub,p.score.to_numpy())
        for metric in ['AP','ROC_AUC','recall5']:
            assert np.isclose(m[metric],row[metric],atol=1e-9,rtol=1e-7),(path.name,metric)
        count+=1
    return count

def uncertainty(df,hist,table):
    from distress_ml.evaluation import bootstrap_pairs
    selected=json.loads((ROOT/'configs/selection_freeze.json').read_text());benchmarks=['baseline_legacy_strong_inner_l1','baseline_v1_lags_xgb','baseline_v1_raw_catboost','baseline_v1_selected_rf'];keys=list(dict.fromkeys([selected['overall'],selected['DL'],*benchmarks]))
    predictions={}
    for key in keys:
        d=pd.read_csv(ART/f'predictions/P2_seed_mean__{key}_historical.csv.gz',float_precision='round_trip');predictions[key]=d.score.to_numpy()
    if selected['blend']['admitted']:
        d=pd.read_csv(ART/'predictions/P2_fixed_rank_blend_historical.csv.gz',float_precision='round_trip');predictions['fixed_annual_rank_blend']=d.score.to_numpy()
        # All pooled AP blend comparisons use the same annual-rank convention.
        for key in keys:
            q=predictions[key].copy()
            for year in sorted(hist.year.unique()):
                ix=np.flatnonzero(hist.year.to_numpy()==year);q[ix]=rankdata(q[ix],method='average')/len(ix)
            predictions[key]=q
    rows=[]
    for reference in benchmarks:
        t=bootstrap_pairs(hist,predictions,reference,repetitions=2000,seed=2026)
        t['score_convention']='annual percentile ranks, frozen cohorts' if selected['blend']['admitted'] else 'mean registered-seed uncalibrated scores'
        rows.append(t)
    pd.concat(rows).to_csv(RESULT/'paired_company_intervals.csv',index=False)

if __name__=='__main__':verify()
