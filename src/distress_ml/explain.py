"""Post-freeze explanations from the exact scoring artifacts, never retraining."""
from __future__ import annotations
import hashlib
import itertools
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from .common import ROOT,ART,PUBLIC,dump
from .data import load_data
from .features import build_features,feature_group
from .train import load_pipeline
from .calibration import calibrated_predict

def explain():
    frozen=json.loads((ROOT/'configs/final_freeze.json').read_text())
    df=load_data('historical'); target=df.year>=2015; sub=df.loc[target]
    rows=[]; cases=[]; costs=[]
    selected={kind:next(m for m in frozen['models'] if kind in m['representative'] and m['seed']==42) for kind in ['ML','DL']}
    for kind,item in selected.items():
        p=load_pipeline(ROOT/item['path']); fs=build_features(df,p.cfg['feature_set']); f=fs.loc[target].copy()
        raw,score=calibrated_predict(p,sub,f)
        base=average_precision_score(sub.y,score); rng=np.random.default_rng(2026)
        groups={g:[c for c in f if feature_group(c)==g] for g in sorted({feature_group(c) for c in f})}
        # Every permutation evaluates this saved model and its exact feature order.
        for group,cols in groups.items():
            drops=[]
            for repeat in range(3):
                pert=f.copy(); order=rng.permutation(len(f)); pert.loc[:,cols]=f.iloc[order][cols].to_numpy()
                _,pred=calibrated_predict(p,sub,pert)
                drops.append(base-average_precision_score(sub.y,pred))
            rows.append(dict(representative=kind,label=item['label'],group=group,mean_AP_drop=float(np.mean(drops)),min_AP_drop=float(np.min(drops)),max_AP_drop=float(np.max(drops)),repetitions=3,role='post_hoc_historical_description',causal=False))
        predicted=score>=p.threshold; y=sub.y.to_numpy()
        for cell,mask in [('TP',(y==1)&predicted),('FP',(y==0)&predicted),('FN',(y==1)&~predicted),('TN',(y==0)&~predicted)]:
            ix=np.flatnonzero(mask)
            if not len(ix): continue
            pick=min(ix,key=lambda i:hashlib.sha256(f'case-v1{sub.iloc[i].company_name}{sub.iloc[i].year}'.encode()).hexdigest())
            cases.append(dict(representative=kind,cell=cell,row_id=int(sub.iloc[pick].row_id),company_id=sub.iloc[pick].company_name,year=int(sub.iloc[pick].year),y_true=int(y[pick]),raw_score=float(raw[pick]),calibrated_score=float(score[pick]),threshold=p.threshold,rule='minimum pre-registered SHA256 within confusion cell',features=f.iloc[pick].to_dict()))
            # Public output describes each cell in aggregate, without company/row IDs.
            snap=build_features(sub.iloc[ix],'S1')
            costs.append(dict(representative=kind,cell=cell,records=len(ix),median_score=float(np.median(score[ix])),median_net_income_to_assets=float(snap.net_income_to_assets.median()),median_liabilities_to_assets=float(snap.liabilities_to_assets.median()),median_current_ratio=float(snap.current_ratio.median())))
    pd.DataFrame(rows).to_csv(PUBLIC/'group_permutation_importance.csv',index=False)
    pd.DataFrame(costs).to_csv(PUBLIC/'confusion_cell_profiles.csv',index=False)
    dump(ART/'audit/local_failure_cases.json',cases)
    stability=[]; coeff=[]
    for cfg_path in (ART/'logs').glob('config_*.json'):
        cfg=json.loads(cfg_path.read_text())
        if cfg.get('selector','none')=='none': continue
        cid=cfg_path.stem.removeprefix('config_'); lists=[]
        for fold in ['D1','D2','D3','D4']:
            pth=ART/f'models/dev_{cid}_{fold}.json'
            if not pth.exists(): continue
            meta=json.loads(pth.read_text()); lists.append(set(meta['selected_features']))
            detail=ART/f'logs/dev_{cid}_{fold}_selection.json'
            for r in json.loads(detail.read_text()):
                if r.get('stage')=='full_coefficient': coeff.append(dict(config_id=cid,fold=fold,**r))
        if not lists: continue
        mean_jaccard=float(np.mean([len(a&b)/len(a|b) if a|b else 1 for a,b in itertools.combinations(lists,2)]))
        for c in sorted(set.union(*lists)):
            stability.append(dict(config_id=cid,family=cfg['family'],selector=cfg['selector'],feature=c,included_folds=sum(c in a for a in lists),folds=len(lists),frequency=sum(c in a for a in lists)/len(lists),mean_pairwise_Jaccard=mean_jaccard))
    pd.DataFrame(stability).to_csv(PUBLIC/'feature_selection_stability.csv',index=False)
    pd.DataFrame(coeff).to_csv(PUBLIC/'elastic_net_coefficients.csv',index=False)
    # Linear coefficients from the same formal final-fit L2 artifact.
    linear=next(m for m in frozen['models'] if m['family']=='logistic' and m['feature_set']=='S1' and m['seed']==42)
    p=load_pipeline(ROOT/linear['path'])
    pd.DataFrame({'feature':p.columns,'coefficient':p.model.coef_[0],'interpretation':'Conditional standardized association; correlated inputs prevent causal interpretation'}).to_csv(PUBLIC/'final_logistic_coefficients.csv',index=False)
    return selected
