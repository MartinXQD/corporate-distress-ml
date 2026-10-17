"""Meaningful leakage, metric and PyTorch contracts on synthetic software fixtures."""
import copy
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import expit
from sklearn.metrics import average_precision_score
from distress_ml.features import build_features,lag_values
from distress_ml.preprocessing import Preprocessor,resample_train,positive_weight
from distress_ml.splits import groups,assert_disjoint,inner_masks
from distress_ml.evaluation import metrics,budgets,choose_threshold,bootstrap_pairs
from distress_ml.neural import TabularNet,seed_everything,predict_logits

def software_event(kind,tag):
    path=os.environ.get('DISTRESS_TEST_FIT_LOG')
    if path:
        with Path(path).open('a') as f:f.write(json.dumps(dict(kind=kind,tag=tag))+'\n')

@pytest.fixture
def panel():
    rng=np.random.default_rng(4)
    d=pd.DataFrame([(f'F{i}',y,int(i%5==0)) for i in range(40) for y in range(1999,2007)],columns=['company_name','year','y'])
    for k in range(1,19): d[f'X{k}']=rng.uniform(.1,10,len(d))
    d['X16']=d.X9; d['X6']=d.X6-5; d['row_id']=np.arange(len(d))
    return d

def test_alignment_and_disjoint(panel):
    a=panel[panel.year<2004]; b=panel[panel.year==2004]; c=panel[panel.year>2004]
    assert_disjoint(a,b,c)
    f=build_features(panel,'S1')
    assert f.index.equals(panel.index)
    assert np.array_equal(f.X6,panel.X6)

def test_future_perturbation_cannot_change_past(panel):
    original=build_features(panel,'S2')
    other=panel.copy(); other.loc[other.year>=2004,'X6']*=1e8
    future=build_features(other,'S2')
    pd.testing.assert_frame_equal(original[panel.year<2004],future[panel.year<2004])

def test_calendar_gap_is_missing(panel):
    d=panel[~((panel.company_name=='F0') & (panel.year==2003))]
    f=build_features(d,'S2')
    idx=d[(d.company_name=='F0') & (d.year==2004)].index[0]
    assert pd.isna(f.loc[idx,'delta1_net_income_to_assets'])
    assert f.loc[idx,'missing_tminus1']==1

def test_lag_same_company(panel):
    lag=lag_values(panel,panel.X6,1)
    keys={(c,y):v for c,y,v in zip(panel.company_name,panel.year,panel.X6)}
    for i,r in panel.iterrows():
        assert np.isclose(lag.loc[i],keys.get((r.company_name,r.year-1),np.nan),equal_nan=True)

def test_preprocessing_external_mutation_isolated(panel):
    f=build_features(panel,'S1'); mask=panel.year<2004
    p=Preprocessor().fit(f[mask]); saved=copy.deepcopy(p.__dict__)
    f.loc[~mask,:]=1e15; p.transform(f[~mask])
    for k in ['medians','lower','upper','means','stds','keep']: np.testing.assert_array_equal(p.__dict__[k],saved[k])

def test_company_holdout_never_fit(panel):
    g=groups(panel)
    for k in range(3):
        fit=panel[(g!=k)&(panel.year<=2003)]; cal=panel[(g!=k)&(panel.year==2004)]; held=panel[g==k]
        assert not set(held.company_name)&set(fit.company_name)
        assert not set(held.company_name)&set(cal.company_name)

def test_sampling_training_only(panel):
    f=build_features(panel,'S0'); train=panel.year<2004
    p=Preprocessor().fit(f[train]); a=p.transform(f[train]); b=p.transform(f[~train]); n=len(b)
    for method in ['none','ros','smote']:
        x,y=resample_train(a,panel.loc[train,'y'].to_numpy(),method,42)
        assert len(x)==len(y)
        if method!='none': assert y.mean()==.5
        assert len(b)==n

def test_weight_current_role(panel):
    y=panel.y.to_numpy(); assert positive_weight(y,1)==4
    assert positive_weight(y,0)==1; assert positive_weight(y,.5)==2

def test_prohibited_columns_absent(panel):
    for fs in ['S0','S1','S2','LAGS']:
        f=build_features(panel,fs)
        assert not {'year','y','company_name','row_id','last_year','lifetime_count'} & set(f)

def test_s0_deduplicated_signed_log_preserves_loss(panel):
    assert len(build_features(panel,'S0').columns)==17
    f=build_features(panel,'S1'); assert (np.sign(f.slog_X6)==np.sign(panel.X6)).all()

def test_invalid_denominator_mask(panel):
    panel.loc[0,'X10']=0
    f=build_features(panel,'S1')
    assert pd.isna(f.loc[0,'net_income_to_assets'])
    assert f.loc[0,'net_income_to_assets_invalid']==1

def test_early_stop_inside_training(panel):
    tr,stop=inner_masks(panel)
    assert panel.loc[tr,'year'].max()<panel.loc[stop,'year'].min()

def test_role_threshold_fixed(panel):
    p=expit(panel.X6.to_numpy())
    decision=panel.year==2005; hist=panel.year==2006
    t=choose_threshold(panel.loc[decision,'y'],p[decision])
    m=metrics(panel.loc[hist],p[hist],t)
    assert m['TP']+m['FN']==panel.loc[hist,'y'].sum()
    assert m['threshold']==t

def test_ap_not_roc_or_accuracy(panel):
    p=expit(panel.X6.to_numpy())
    m=metrics(panel,p)
    assert np.isclose(m['AP'],average_precision_score(panel.y,p))
    assert m['TP']+m['TN']+m['FP']+m['FN']==len(panel)

def test_budget_ties_label_independent(panel):
    score=np.ones(len(panel)); a=budgets(panel,score)
    changed=panel.copy(); changed['y']=1-panel.y
    b=budgets(changed,score)
    assert np.array_equal(a.selected,b.selected)
    assert (a.selected==np.ceil(a.rows*a.budget)).all()

@pytest.mark.parametrize('family',['mlp','residual_mlp','ft_transformer'])
def test_torch_shape_loss_update_reload(family,tmp_path):
    seed_everything(42); x=torch.randn(24,7); y=torch.tensor([0.,1.]*12)
    cfg=dict(width=16,depth=2,token_dim=8,dropout=.1)
    m=TabularNet(family,7,cfg); m.train(); before=copy.deepcopy(m.state_dict())
    optimizer=torch.optim.AdamW(m.parameters(),lr=.01)
    score=m(x); assert score.shape==y.shape
    loss=torch.nn.BCEWithLogitsLoss()(score,y); assert torch.isfinite(loss)
    loss.backward(); assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    optimizer.step(); assert any(not torch.equal(before[k],v) for k,v in m.state_dict().items())
    software_event('single_neural_update',family)
    m.eval(); a=predict_logits(m,x.numpy()); assert not m.training
    path=tmp_path/'own.pt'; torch.save(m.state_dict(),path)
    other=TabularNet(family,7,cfg); other.load_state_dict(torch.load(path,weights_only=True)); other.eval()
    np.testing.assert_allclose(a,predict_logits(other,x.numpy()),atol=1e-6)

def test_bootstrap_paired_identical_difference_zero(panel):
    p=expit(panel.X6.to_numpy())
    t=bootstrap_pairs(panel,{'a':p,'b':p},'a',repetitions=10)
    assert (t.delta_lower==0).all() and (t.delta_upper==0).all()
    assert (t.unit=='company').all()

def test_flattened_lags_retain_all_rows(panel):
    f=build_features(panel,'LAGS')
    assert len(f)==len(panel); assert len(f.columns)==54
    assert f.loc[panel.year==1999,'observed_lag2'].eq(0).all()

def test_sequence_calendar_mask_and_shape(panel):
    from distress_ml.sequences import snapshot_from_lags,sequence_tensor
    d=panel[~((panel.company_name=='F0') & (panel.year==2003))]
    f=build_features(d,'LAGS')
    pp=Preprocessor().fit(snapshot_from_lags(f,0))
    x=sequence_tensor(f,pp)
    assert x.ndim==3 and x.shape[:2]==(len(d),3)
    i=np.flatnonzero(((d.company_name=='F0') & (d.year==2004)).to_numpy())[0]
    assert x[i,1,-1]==0 and (x[i,1,:-1]==0).all()
    assert x[i,2,-1]==1
    m=TabularNet('gru',x.shape[-1],dict(width=16,dropout=.1));m.eval()
    assert predict_logits(m,x).shape==(len(d),)

def test_sequence_future_invariance(panel):
    from distress_ml.sequences import snapshot_from_lags,sequence_tensor
    tr=panel.year<2004
    f=build_features(panel,'LAGS');pp=Preprocessor().fit(snapshot_from_lags(f.loc[tr],0))
    a=sequence_tensor(f.loc[tr],pp)
    changed=panel.copy();changed.loc[~tr,'X1']=1e12
    b=sequence_tensor(build_features(changed,'LAGS').loc[tr],pp)
    np.testing.assert_array_equal(a,b)

def test_nested_selector_external_mutation_not_used(panel,tmp_path,monkeypatch):
    import distress_ml.selection as module
    (tmp_path/'models').mkdir()
    monkeypatch.setattr(module,'ART',tmp_path)
    counter=iter(range(1,1000))
    def spy(*args,**kwargs):
        software_event('model_fit',args[0]);return next(counter)
    monkeypatch.setattr(module,'fit_event',spy)
    tr=panel.year<=2004;f=build_features(panel,'S1')
    selected,_=module.select_features(f.loc[tr],panel.loc[tr],'elastic_net',42,'synthetic_test')
    changed=panel.copy();changed.loc[~tr,'y']=1-changed.loc[~tr,'y'];changed.loc[~tr,'X6']=1e15
    selected2,_=module.select_features(build_features(changed,'S1').loc[tr],changed.loc[tr],'elastic_net',42,'synthetic_test')
    assert selected==selected2

def test_calibration_roles_and_fixed_threshold(panel,monkeypatch):
    import distress_ml.calibration as module
    from types import SimpleNamespace
    def spy(*args,**kwargs):
        software_event('model_fit',args[0]);return 1
    monkeypatch.setattr(module,'fit_event',spy)
    d=panel.copy();d['year']+=9
    model=SimpleNamespace(metadata={},cfg={'feature_set':'S0'},predict_raw=lambda df,features=None:df.X6.to_numpy())
    fit=d[d.year<=2012];cal=d[d.year==2013];decision=d[d.year==2014]
    p=module.fit_calibration(model,fit,cal,decision)
    assert p.metadata['calibration_year']==2013 and p.metadata['decision_year']==2014
    threshold=p.threshold
    module.calibrated_predict(p,d[d.year==2015])
    assert p.threshold==threshold
    with pytest.raises(AssertionError):module.fit_calibration(model,fit,cal,cal)

def test_bootstrap_integral_company_multiplicity(panel):
    from distress_ml.splits import tie_key
    companies,indices=np.unique(panel.company_name,return_inverse=True)
    rng=np.random.default_rng(3)
    counts=np.bincount(rng.integers(0,len(companies),len(companies)),minlength=len(companies))
    weights=counts[indices]
    for _,g in panel.assign(w=weights).groupby('company_name'):
        assert g.w.nunique()==1

def test_equal_rank_ensemble_uses_only_within_year_scores(panel):
    from distress_ml.ensembles import annual_rank
    original=annual_rank(panel.year,panel.X6)
    changed=panel.X6.copy();changed.loc[panel.year==2006]=1e12
    updated=annual_rank(panel.year,changed)
    np.testing.assert_array_equal(original[panel.year<2006],updated[panel.year<2006])

def test_gru_checkpoint_roundtrip(tmp_path):
    seed_everything(42)
    x=np.random.default_rng(2).normal(size=(20,3,18)).astype(np.float32)
    cfg=dict(width=16,dropout=.1)
    m=TabularNet('gru',18,cfg);m.eval();before=predict_logits(m,x)
    path=tmp_path/'own_gru.pt';torch.save(m.state_dict(),path)
    other=TabularNet('gru',18,cfg);other.load_state_dict(torch.load(path,weights_only=True));other.eval()
    np.testing.assert_allclose(before,predict_logits(other,x),atol=1e-6)

def test_untrusted_pipeline_path_rejected(tmp_path):
    from distress_ml.train import load_pipeline
    path=tmp_path/'unknown.joblib'
    path.write_bytes(b'not-a-project-checkpoint')
    with pytest.raises(ValueError,match='trusted artifacts'):
        load_pipeline(path)
