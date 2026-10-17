"""Development-only selection, one confirmation batch, frozen historical scoring."""
import itertools
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score,roc_auc_score
from .common import ROOT,V1,ART,RESULT,dump,digest,sha,now,hours,state,time_guard,database
from .data import load,features
from .training import fit,save,load as load_model,NN
from .search import run_fold,measures,baselines,execute,export

YEARS=[2009,2010,2011,2012,2013]
STABILITY_SEEDS=[42,2026,31415]
FINAL_SEEDS=[42,2026,31415,7,2718]

def cfg(candidate):return json.loads((ROOT/f'configs/trial_{candidate}.json').read_text())
def dv():return pd.DataFrame([json.loads(p.read_text()) for p in (ART/'logs').glob('fold_*.json')])
def predictions(candidate,seeds=STABILITY_SEEDS):
    frames=[]
    for seed in seeds:
        pieces=[]
        for year in YEARS:
            p=ART/f'predictions/{candidate}__seed{seed}__year{year}.csv.gz'
            if not p.exists():raise FileNotFoundError(p.name)
            pieces.append(pd.read_csv(p,float_precision='round_trip'))
        d=pd.concat(pieces).sort_values('row_id').reset_index(drop=True);frames.append(d)
    for d in frames[1:]:assert np.array_equal(frames[0].row_id,d.row_id)
    result=frames[0].copy();result['score']=np.mean([d.score.to_numpy() for d in frames],axis=0)
    return result

def select_shortlist():
    if (ART/'baseline_steps_correction_required.json').exists() and not (ART/'baseline_steps_correction_applied.json').exists():
        raise RuntimeError('Search finished. Resume the runner in a fresh process for the recorded V2 ROS baseline step correction before shortlist/confirmation; HPO step-refit configurations are unchanged.')
    from .refinement import run as refine
    refine()
    path=ROOT/'configs/shortlist.json'
    if path.exists():chosen=json.loads(path.read_text())['candidates']
    else:
        c=database();t=pd.read_sql_query("SELECT * FROM trial_records WHERE status='COMPLETE' AND stage IN ('screen','baseline','refinement')",c);c.close()
        assert len(t)>0
        neural=t.family.isin(NN);tree=t.family.isin(['catboost','xgboost','random_forest','extra_trees','xgb_rank'])
        mandatory=['baseline_legacy_strong_inner_l1','baseline_v1_lags_xgb','baseline_v1_raw_catboost','baseline_v1_selected_rf']
        mandatory+=[t[tree].sort_values('macro_AP',ascending=False).iloc[0].trial_id,t[neural].sort_values('macro_AP',ascending=False).iloc[0].trial_id]
        chosen=list(dict.fromkeys(mandatory));counts=t.set_index('trial_id').family.loc[chosen].value_counts().to_dict()
        for r in t.sort_values('macro_AP',ascending=False).itertuples():
            if len(chosen)>=8:break
            if r.trial_id in chosen or counts.get(r.family,0)>=2:continue
            chosen.append(r.trial_id);counts[r.family]=counts.get(r.family,0)+1
        # Reduce optional candidates when measured fold cost threatens final fits.
        measured=dv();projection=sum(measured[measured.candidate==x].fit_seconds.mean()*12 for x in chosen)/3600
        while len(chosen)>len(set(mandatory)) and projection>max(.5,6.45-hours()):
            chosen.pop();projection=sum(measured[measured.candidate==x].fit_seconds.mean()*12 for x in chosen)/3600
        dump(path,dict(candidates=chosen,selected_at_utc=now(),reason='Mandatory same-protocol strong legacy and LAGS baselines plus best screened tree/NN; up to two optional high-AP diverse candidates. Measured cost can remove optional candidates before stability.',estimated_remaining_stability_hours=projection,history_labels_unavailable=True))
    for candidate in chosen:
        config=cfg(candidate)
        for seed in STABILITY_SEEDS:
            for year in YEARS:
                time_guard(6.65)
                run_fold({**config,'seed':seed},year,'stability',candidate)
    t=dv();t=t[t.candidate.isin(chosen)&t.seed.isin(STABILITY_SEEDS)]
    counts=t.groupby('candidate').size();assert all(counts.get(k,0)==15 for k in chosen)
    summary=t.groupby('candidate').agg(family=('family','first'),features=('features','first'),macro_AP=('AP','mean'),macro_ROC=('ROC_AUC','mean'),recall5=('recall5','mean'),mean_fit_seconds=('fit_seconds','mean')).reset_index()
    recent=t[t.year.isin([2012,2013])].groupby('candidate').AP.mean();summary['recent_AP']=summary.candidate.map(recent)
    benchmark_names=['baseline_legacy_strong_inner_l1','baseline_v1_lags_xgb','baseline_v1_raw_catboost','baseline_v1_selected_rf']
    incumbent=summary[summary.candidate.isin(benchmark_names)].sort_values('macro_AP',ascending=False).iloc[0]
    summary['guardrail_pass']=(summary.macro_ROC>=incumbent.macro_ROC-.01)&(summary.recall5>=incumbent.recall5-.02)
    eligible=summary[summary.guardrail_pass]
    near=eligible[eligible.macro_AP>=eligible.macro_AP.max()-.005]
    order=['recent_AP','recall5','mean_fit_seconds','candidate']
    winner=near.sort_values(order,ascending=[False,False,True,True]).iloc[0].candidate
    neural=summary[summary.family.isin(NN)]
    near_nn=neural[neural.macro_AP>=neural.macro_AP.max()-.005]
    dl=near_nn.sort_values(order,ascending=[False,False,True,True]).iloc[0].candidate
    trees=summary[summary.family.isin(['catboost','xgboost','random_forest','extra_trees','xgb_rank'])]
    near_trees=trees[trees.macro_AP>=trees.macro_AP.max()-.005]
    ml=near_trees.sort_values(order,ascending=[False,False,True,True]).iloc[0].candidate
    summary.to_csv(RESULT/'selection_trace.csv',index=False)
    selected=dict(overall=winner,ML=ml,DL=dl,shortlist=chosen,selection_years=YEARS,selection_seeds=STABILITY_SEEDS,near_tie=.005,guardrail_reference=incumbent.candidate,chosen_at_utc=now(),history_exposed_before_version=True)
    selected['blend']=choose_blend(selected,summary)
    dump(ROOT/'configs/selection_freeze.json',selected);dump(RESULT/'selected_candidates.json',selected)
    state('selection_frozen_before_2014',overall=winner,DL=dl)
    return selected

def annual_ranks(df,scores):
    result=np.zeros(len(df))
    for year in sorted(df.year.unique()):
        ix=np.flatnonzero(df.year.to_numpy()==year);result[ix]=rankdata(np.asarray(scores)[ix],method='average')/len(ix)
    return result

def measure_frame(df,scores):return measures(df.rename(columns={'company_id':'company_name','y_true':'y'}),scores)

def choose_blend(selected,summary):
    scores={k:predictions(k) for k in selected['shortlist']}
    reference=scores[selected['overall']];ranks={k:annual_ranks(v,v.score.to_numpy()) for k,v in scores.items()}
    for d in scores.values():assert np.array_equal(reference.row_id,d.row_id)
    overall=selected['overall'];dl=selected['DL'];pairs=[]
    if overall!=dl:pairs.append(tuple(sorted([overall,dl])))
    alternatives=summary[summary.candidate!=overall].sort_values('macro_AP',ascending=False)
    if len(alternatives):pairs.append(tuple(sorted([overall,alternatives.iloc[0].candidate])))
    trees=summary[~summary.family.isin(NN)].sort_values('macro_AP',ascending=False)
    if len(trees) and trees.iloc[0].candidate!=dl:pairs.append(tuple(sorted([trees.iloc[0].candidate,dl])))
    rows=[];best=None;baseline=measure_frame(reference,ranks[overall])
    for a,b in dict.fromkeys(pairs):
        for weight in [0,.25,.5,.75,1]:
            score=weight*ranks[a]+(1-weight)*ranks[b];m=measure_frame(reference,score)
            row=dict(component_a=a,component_b=b,weight_a=weight,weight_b=1-weight,**m);rows.append(row)
            if weight in [0,1]:continue
            if m['ROC_AUC']<baseline['ROC_AUC']-.01 or m['recall5']<baseline['recall5']-.02:continue
            if best is None or m['AP']>best['AP']:best=row
    # Blend selection uses macro-year AP, not pooled cross-year rank AP.
    for row in rows:
        score=row['weight_a']*ranks[row['component_a']]+row['weight_b']*ranks[row['component_b']]
        row['macro_AP']=float(np.mean([average_precision_score(reference.loc[reference.year==year,'y_true'],score[reference.year.to_numpy()==year]) for year in YEARS]))
        row['macro_ROC']=float(np.mean([roc_auc_score(reference.loc[reference.year==year,'y_true'],score[reference.year.to_numpy()==year]) for year in YEARS]))
    pd.DataFrame(rows).to_csv(RESULT/'ensemble_development.csv',index=False)
    base_macro=float(np.mean([average_precision_score(reference.loc[reference.year==year,'y_true'],ranks[overall][reference.year.to_numpy()==year]) for year in YEARS]))
    eligible=[r for r in rows if r['weight_a'] not in [0,1] and r['macro_ROC']>=summary.set_index('candidate').loc[overall,'macro_ROC']-.01 and r['recall5']>=baseline['recall5']-.02]
    best=max(eligible,key=lambda r:r['macro_AP']) if eligible else None
    chosen=best is not None and best['macro_AP']>=base_macro+.005
    correlations=[]
    for a,b in itertools.combinations(ranks,2):
        top_a=set();top_b=set()
        for year in YEARS:
            ix=np.flatnonzero(reference.year.to_numpy()==year);n=int(np.ceil(.05*len(ix)))
            top_a.update(ix[np.argsort(-ranks[a][ix],kind='stable')[:n]]);top_b.update(ix[np.argsort(-ranks[b][ix],kind='stable')[:n]])
        correlations.append(dict(a=a,b=b,annual_rank_correlation=float(np.corrcoef(ranks[a],ranks[b])[0,1]),top5_Jaccard=len(top_a&top_b)/len(top_a|top_b),positive_a_only=int(reference.iloc[list(top_a-top_b)].y_true.sum()),positive_b_only=int(reference.iloc[list(top_b-top_a)].y_true.sum())))
    pd.DataFrame(correlations).to_csv(RESULT/'ensemble_complementarity.csv',index=False)
    return dict(admitted=chosen,candidate=best,reference_macro_AP=base_macro,admission='At least +0.005 macro-year development AP, ROC/Top5 guardrails, finite two-model nonnegative grid; NN weight may be zero',score_type='annual percentile rank',reference_cohorts_fixed_for_bootstrap=True)

def all_final_configs(selected):
    result={selected['overall']:cfg(selected['overall']),selected['ML']:cfg(selected['ML']),selected['DL']:cfg(selected['DL'])}
    if selected['blend']['admitted']:
        b=selected['blend']['candidate']
        for key in [b['component_a'],b['component_b']]:result[key]=cfg(key)
    for name,c in baselines().items():result['baseline_'+name]=c
    return result

def confirm_and_fit(selected):
    lock=ART/'confirmation_opened.json'
    if not lock.exists():dump(lock,dict(opened_at_utc=now(),selection_sha256=sha(ROOT/'configs/selection_freeze.json'),rule='Single frozen batch; no selection changes after this point'))
    df=load('confirmation');configurations=all_final_configs(selected);confirmation=[];inventory=[]
    assert int((df.year<=2014).sum())==66400
    for key,c in configurations.items():
        f=features(df,c['features']);done=ART/f'logs/confirmation_{key}.json'
        if done.exists():confirmation.append(json.loads(done.read_text()))
        else:
            p=fit(df[df.year<=2013],f[df.year<=2013],{**c,'seed':42},'confirm_'+key)
            q=p.predict(f[df.year==2014]);m=dict(candidate=key,channel='P2_frozen_confirmation',fit_cutoff=2013,seed=42,**measures(df[df.year==2014],q));confirmation.append(m);dump(done,m)
            out=df.loc[df.year==2014,['row_id','company_name','year','y']].copy();out['score']=q;out.to_csv(ART/f'predictions/confirmation_{key}.csv.gz',index=False)
        seed_list=FINAL_SEEDS if key!='baseline_logistic' else [42]
        for seed in seed_list:
            name=f'P2__{key}__seed{seed}';path=ART/f'models/{name}.joblib'
            if not path.exists():
                time_guard(7.25)
                p=fit(df[df.year<=2014],f[df.year<=2014],{**c,'seed':seed},name);save(p,path)
            inventory.append(dict(name=name,candidate=key,channel='P2',fit_cutoff=2014,seed=seed,path=str(path.relative_to(ROOT)),score_type='raw ranking score' if c['family']=='xgb_rank' else 'uncalibrated risk score'))
    pd.DataFrame(confirmation).to_csv(RESULT/'confirmation_2014.csv',index=False)
    # Strict P1 fixed legacy transfer versus P2-selected cutoff transfer are distinct.
    transfer={'P1_fixed_legacy':baselines()['legacy_strong_inner_l1']}
    transfer.update({'P2_selected__fit_cutoff_2012_transfer__'+k:cfg(v) for k,v in [('overall',selected['overall']),('DL',selected['DL'])]})
    for key,c in transfer.items():
        f=features(df,c['features']);path=ART/f'models/{key}.joblib'
        if not path.exists():
            p=fit(df[df.year<=2012],f[df.year<=2012],c,key)
            if key.startswith('P1_'):
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler
                from sklearn.linear_model import LogisticRegression
                from .common import fit_event
                from distress_ml.evaluation import choose_threshold
                cal=df.year==2013;decision=df.year==2014
                raw=p.predict(f[cal]);fit_event(key+'/Platt2013',df[cal],c)
                p.calibrator=make_pipeline(StandardScaler(),LogisticRegression(C=1e6,max_iter=1000)).fit(logit_safe(raw).reshape(-1,1),df.loc[cal,'y'].to_numpy(int))
                q=p.calibrator.predict_proba(logit_safe(p.predict(f[decision])).reshape(-1,1))[:,1]
                p.threshold=choose_threshold(df.loc[decision,'y'].to_numpy(int),q)
            save(p,path)
        inventory.append(dict(name=key,candidate=key,channel='P1' if key.startswith('P1_') else 'P2_selected__fit_cutoff_2012_transfer',fit_cutoff=2012,seed=42,path=str(path.relative_to(ROOT)),score_type='separate P1 calibrated score' if key.startswith('P1_') else 'uncalibrated ranking score'))
    from .archives import reconstruct
    inventory+=reconstruct(df)
    inventory+=company_stress(selected,df)
    dump(ROOT/'configs/final_inventory.json',inventory)
    hashes={str(p.relative_to(ROOT)):sha(p) for item in inventory for p in [ROOT/item['path'],(ROOT/item['path']).with_suffix('.json')]}
    for item in inventory:
        p=(ROOT/item['path']).with_suffix('.pt')
        if p.exists():hashes[str(p.relative_to(ROOT))]=sha(p)
    for p in (ROOT/'src').rglob('*.py'):
        if p.name not in ['reporting.py','verify.py','inference.py']:hashes[str(p.relative_to(ROOT))]=sha(p)
    dump(ROOT/'configs/artifact_freeze.json',dict(frozen_at_utc=now(),selection_sha256=sha(ROOT/'configs/selection_freeze.json'),artifacts=hashes,models=inventory,historical_exposure='Previously inspected; diagnostics also rechecked saved V1 scores/labels',new_candidate_historical_scoring_not_started=True))
    state('final_artifacts_frozen_before_historical')

def logit_safe(x):
    from scipy.special import logit
    return logit(np.clip(x,1e-7,1-1e-7))

def company_stress(selected,df):
    from distress_ml.splits import groups
    inventory=[]
    for role,candidate in [('ML',selected['ML']),('DL',selected['DL'])]:
        c=cfg(candidate);f=features(df,c['features'])
        for fold in [0,1,2]:
            name=f'post_selection_group{fold}_{role}';path=ART/f'models/{name}.joblib';mask=(groups(df)!=fold)&(df.year<=2014)
            if not path.exists():
                p=fit(df[mask],f[mask],{**c,'seed':42},name);save(p,path)
            inventory.append(dict(name=name,candidate=candidate,channel='post_selection_company_stress',fit_cutoff=2014,seed=42,group_fold=fold,path=str(path.relative_to(ROOT)),score_type='ranking score; selected hyperparameters use broader companies'))
    return inventory

def score_historical():
    lock=ART/'historical_opened.json'
    if lock.exists():
        if (ART/'historical_complete.json').exists():return
        raise RuntimeError('Historical scoring was interrupted; inspect partial evidence and resume the same frozen batch explicitly, never restart search')
    frozen=json.loads((ROOT/'configs/artifact_freeze.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in frozen['artifacts'].items())
    dump(lock,dict(opened_at_utc=now(),freeze_sha256=sha(ROOT/'configs/artifact_freeze.json')))
    df=load('historical');hist=df.year>=2015;rows=[];yearly=[];pred={};probability_rows=[];selected=json.loads((ROOT/'configs/selection_freeze.json').read_text())
    from distress_ml.splits import groups
    for item in frozen['models']:
        p=load_model(ROOT/item['path']);f=features(df,p.cfg['features']);mask=hist.copy()
        if item['channel']=='post_selection_company_stress':mask &= groups(df)==item['group_fold']
        inference_started=time.perf_counter();score=p.predict(f[mask]);inference_seconds=time.perf_counter()-inference_started;sub=df[mask];raw_score=score.copy()
        if item['channel']=='P1':score=p.calibrator.predict_proba(logit_safe(score).reshape(-1,1))[:,1]
        if item['channel']=='P1':
            from distress_ml.evaluation import metrics as probability_metrics
            probability_rows.append(dict(name=item['name'],channel='P1_only',base_fit_cutoff=2012,calibration_year=2013,threshold_year=2014,**probability_metrics(sub,score,p.threshold)))
        out=sub[['row_id','company_name','year','y']].rename(columns={'company_name':'company_id','y':'y_true'}).copy();out['score']=score;out['raw_score']=raw_score;out.to_csv(ART/f'predictions/{item["name"]}_historical.csv.gz',index=False)
        m=historical_metrics(sub,score);rows.append(dict(**item,**m,fit_seconds=p.metadata['fit_seconds'],inference_seconds=inference_seconds))
        for year,g in sub.groupby('year'):yearly.append(dict(**item,year=int(year),**measures(g,score[sub.index.get_indexer(g.index)])))
        if item['channel']=='P2':pred.setdefault(item['candidate'],[]).append(out)
    mean_predictions={}
    for key,frames in pred.items():
        for f in frames:assert np.array_equal(f.row_id,frames[0].row_id)
        out=frames[0].copy();out['score']=np.mean([f.score.to_numpy() for f in frames],axis=0);mean_predictions[key]=out
        out.to_csv(ART/f'predictions/P2_seed_mean__{key}_historical.csv.gz',index=False)
        sub=df.loc[hist];m=historical_metrics(sub,out.score.to_numpy());rows.append(dict(name='P2_seed_mean__'+key,candidate=key,channel='P2_seed_ensemble',fit_cutoff=2014,seed_count=len(frames),score_type='mean uncalibrated score across registered seeds',**m))
    # Same annual-rank score type for every fusion comparison.
    ranked={key:annual_ranks(frame,frame.score.to_numpy()) for key,frame in mean_predictions.items()}
    for key,q in ranked.items():rows.append(dict(name='P2_annual_rank__'+key,candidate=key,channel='P2_annual_rank_seed_ensemble',fit_cutoff=2014,score_type='annual percentile rank of mean seed score',**historical_metrics(df[hist],q)))
    if selected['blend']['admitted']:
        b=selected['blend']['candidate'];q=b['weight_a']*ranked[b['component_a']]+b['weight_b']*ranked[b['component_b']]
        rows.append(dict(name='P2_fixed_rank_blend',candidate='frozen_blend',channel='P2_annual_rank_seed_ensemble',fit_cutoff=2014,score_type='fixed annual-rank blend',**historical_metrics(df[hist],q)))
        out=df.loc[hist,['row_id','company_name','year','y']].rename(columns={'company_name':'company_id','y':'y_true'}).copy();out['score']=q;out.to_csv(ART/'predictions/P2_fixed_rank_blend_historical.csv.gz',index=False)
    pd.DataFrame(rows).to_csv(RESULT/'historical_2015_2018.csv',index=False);pd.DataFrame(yearly).to_csv(RESULT/'historical_by_year.csv',index=False)
    pd.DataFrame(probability_rows).to_csv(RESULT/'p1_probability_supplement.csv',index=False)
    pd.DataFrame([r for r in rows if r['channel']=='P2']).to_csv(RESULT/'seeds.csv',index=False)
    unseen_tables(df,mean_predictions)
    dump(ART/'historical_complete.json',dict(completed_at_utc=now(),pipelines=len(frozen['models']),rows=int(hist.sum()),new_fit_after_history=False));state('historical_complete')

def historical_metrics(df,q):
    m=measures(df,q)
    valid=all(g.y.nunique()==2 for _,g in df.groupby('year'))
    m['macro_AP']=float(np.mean([average_precision_score(g.y,q[df.index.get_indexer(g.index)]) for _,g in df.groupby('year')])) if valid else np.nan
    m['macro_ROC']=float(np.mean([roc_auc_score(g.y,q[df.index.get_indexer(g.index)]) for _,g in df.groupby('year')])) if valid else np.nan
    m['single_class_years']=sum(g.y.nunique()<2 for _,g in df.groupby('year'))
    return m

def unseen_tables(df,predictions):
    hist=df[df.year>=2015];seen=set(df.loc[df.year<=2014,'company_name']);rows=[]
    for key,frame in predictions.items():
        for group in ['seen','unseen']:
            mask=hist.company_name.isin(seen).to_numpy()
            if group=='unseen':mask=~mask
            rows.append(dict(candidate=key,cutoff=2014,novelty=group,companies=int(hist[mask].company_name.nunique()),**historical_metrics(hist[mask],frame.score.to_numpy()[mask])))
    pd.DataFrame(rows).to_csv(RESULT/'seen_unseen.csv',index=False)
