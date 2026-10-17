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
    numerical_walkthrough(df,selected,cases)
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

def numerical_walkthrough(df,selected,cases):
    """Keep a complete one-record numerical trace local; publish no row dataset."""
    from scipy.special import expit
    item=selected['DL'];model=load_pipeline(ROOT/item['path'])
    case=next((x for x in cases if x['representative']=='DL' and x['cell']=='FN'),next(x for x in cases if x['representative']=='DL'))
    mask=df.row_id==case['row_id'];f=build_features(df,model.cfg['feature_set']).loc[mask]
    if model.cfg['family']=='gru':
        from .sequences import sequence_tensor,snapshot_from_lags
        tensor=sequence_tensor(f,model.preprocessor)
        input_frame=snapshot_from_lags(f,0)
        feature='X6' if 'X6' in model.preprocessor.columns else model.preprocessor.columns[0]
        j=model.preprocessor.columns.index(feature);transformed=float(tensor[0,-1,j])
    else:
        input_frame=f[model.selected]
        tensor=model.preprocessor.transform(input_frame)
        preferences=['net_income_to_assets','X6','slog_X6','lag0_X6']
        feature=next((x for x in preferences if x in model.preprocessor.columns and x in input_frame),next(x for x in model.preprocessor.columns if x in input_frame))
        j=model.preprocessor.columns.index(feature);transformed=float(tensor[0,j])
    i=model.preprocessor.input_columns.index(feature)
    value=float(input_frame.iloc[0][feature]);median=float(model.preprocessor.medians[i])
    imputed=value if np.isfinite(value) else median
    clipped=float(np.clip(imputed,model.preprocessor.lower[i],model.preprocessor.upper[i]))
    mean=float(model.preprocessor.means[j]);std=float(model.preprocessor.stds[j])
    assert np.isclose((clipped-mean)/std,transformed,atol=1e-5)
    raw,score=calibrated_predict(model,df.loc[mask],f)
    assert np.isclose(raw[0],case['raw_score'],atol=1e-6)
    assert np.isclose(score[0],case['calibrated_score'],atol=1e-6)
    financial=df.loc[mask,[f'X{k}' for k in range(1,19)]].iloc[0].to_dict()
    trace=dict(case_rule=case['rule'],cell=case['cell'],model_label=item['label'],row_id=case['row_id'],company_id=case['company_id'],year=case['year'],raw_financial_fields=financial,traced_feature=feature,feature_value=value,training_median=median,imputed=imputed,training_clip_lower=float(model.preprocessor.lower[i]),training_clip_upper=float(model.preprocessor.upper[i]),clipped=clipped,training_mean=mean,training_std=std,tensor_feature_value=transformed,tensor_shape=list(tensor.shape),full_tensor=tensor.tolist(),raw_logit=float(raw[0]),uncalibrated_sigmoid=float(expit(raw[0])),calibrated_score=float(score[0]),fixed_2014_threshold=model.threshold,predicted_label=int(score[0]>=model.threshold),y_true=case['y_true'],checkpoint_reloaded=True)
    dump(ART/'audit/local_numerical_walkthrough.json',trace)
    text=f'''# Local numerical case walkthrough / 本地数值核验\n\nLocal only. Do not publish this individual record, financial inputs or full tensor. 此文件只供本人核查，不上传 GitHub。\n\n## 1. Registered case / 预登记案例\n\nModel: {item['label']}. Cell: {case['cell']}. Row: {case['row_id']}; company: {case['company_id']}; fiscal year: {case['year']}. Selection follows the registered hash rule, without choosing a persuasive anecdote.\n\n## 2. Raw financial inputs / 原始财务输入\n\nX6 net income = {financial['X6']:.9g}; X10 current total assets = {financial['X10']:.9g}; X17 total liabilities = {financial['X17']:.9g}.\n\nWhen X10 is positive, net-income/assets = {financial['X6']/financial['X10']:.9g}. This ratio is illustrative of feature construction; the traced model input below is an actually retained input in this checkpoint, not a forced ratio for a model that did not use it.\n\n## 3. Actual preprocessing / 正式预处理\n\nTraced input: {feature}. Input value = {value:.9g}. Training median = {median:.9g}; imputed value = {imputed:.9g}. Training clip interval = [{trace['training_clip_lower']:.9g}, {trace['training_clip_upper']:.9g}]; clipped value = {clipped:.9g}. Saved training mean = {mean:.9g}; std = {std:.9g}.\n\nThe actual tensor value is (clipped - mean) / std = {transformed:.9g}. Tensor shape = {list(tensor.shape)}. The complete tensor and all 18 source values are saved in artifacts/audit/local_numerical_walkthrough.json. Only one component is displayed here; the network logit uses the entire actual tensor. 不能把这一个变量的数值误认为整个网络的解释。\n\n## 4. Logit, calibration and decision / 分数到判断\n\nSaved checkpoint raw logit z = {raw[0]:.9g}. Uncalibrated sigmoid(z) = {expit(raw[0]):.9g}. The independently fitted 2013 Platt calibrator gives q = {score[0]:.9g}. The frozen 2014 F1 threshold is {model.threshold:.9g}; predicted_label = {int(score[0]>=model.threshold)}, observed label = {case['y_true']}.\n\nThe case is therefore {case['cell']} at the fixed threshold. A different threshold can change this decision but cannot be chosen from this historical outcome in this version. The calibrated number refers to the company-constant label, not a verified annual default event.\n\n## 5. Recheck / 亲自验证\n\nNotebook 04 reconstructs the same model/features and checks logits and calibrated scores. Compare this file with local_failure_cases.json and the saved historical prediction file. All checks use the formal scoring checkpoint and its exact feature order; no explanatory model was retrained.\n'''
    (ROOT/'reports/Local_Case_Walkthrough_EN_CN.md').write_text(text)
