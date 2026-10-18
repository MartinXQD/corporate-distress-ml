"""Short numerical and optimization contracts; synthetic scores are not research."""
import json
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from .common import ROOT,ART,RESULT,dump,now
from .data import load,features,legacy_features
from .preprocessing import NumericTransform
from .networks import NumericNetwork,LegacyResidual,probabilities,logits
from .training import train_neural
from .search import base,execute

def contracts():
    results=[]
    for d,n in [(18,302209),(13,301249)]:
        assert sum(p.numel() for p in LegacyResidual(d).parameters())==n
    results.append('legacy_parameter_counts')
    x=pd.DataFrame({'amount':np.linspace(-3,3,64),'invalid':[0,1]*32})
    pp=NumericTransform('quantile',indicators=False).fit(x);z=pp.transform(x)
    assert set(z[:,-1])=={0.,1.};assert pp.n_cont==1
    before=pp.median.copy();pp.transform(x.assign(amount=10000));np.testing.assert_array_equal(before,pp.median)
    results.extend(['binary_masks_bypass_quantile','transform_does_not_refit'])
    rng=np.random.default_rng(42);a=rng.normal(size=(64,6)).astype('float32')
    for family,embedding in [('mlp','none'),('legacy_residual','none'),('num_mlp','piecewise'),('num_mlp','periodic'),('tabm','none'),('tabm','piecewise'),('official_ft','none')]:
        cfg=base(family,width=128,depth=2,dropout=.15,embedding=embedding,embedding_dim=8,bins=8,token_dim=32)
        m=NumericNetwork(6,6,cfg,x_for_bins=a);p=probabilities(m,a);raw=logits(m,a)
        assert p.shape==(64,) and np.isfinite(p).all()
        if family=='tabm':
            assert raw.shape==(64,16);np.testing.assert_allclose(p,expit(raw).mean(axis=1),atol=1e-7)
            y=torch.tensor(np.arange(64)%2,dtype=torch.float32);zz=m(torch.from_numpy(a));loss=torch.nn.functional.binary_cross_entropy_with_logits(zz,y[:,None].expand_as(zz));loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        results.append(f'official_or_legacy_forward_{family}_{embedding}')
    # Future financial changes cannot alter earlier exact-calendar features.
    import distress_v2.data as module
    df=load();module._features.clear();original=features(df,'LAGS_SLOG3').copy();future=df.copy();future.loc[future.year>=2014,'X10']*=100
    module._features.clear();changed=features(future,'LAGS_SLOG3');pd.testing.assert_frame_equal(original[df.year<=2013],changed[df.year<=2013]);module._features.clear();results.append('future_input_invariance')
    # Balanced real-training subset is only an overfit sanity diagnostic.
    development=df[df.year<=2011];subset=pd.concat([development[development.y==0].head(32),development[development.y==1].head(32)])
    raw=features(df,'RAW17').loc[subset.index];pp=NumericTransform('std01',indicators=False).fit(raw);config=base('mlp',width=128,depth=2,dropout=0.,learning_rate=.003,weight_decay=0.,batch_size=64,max_epochs=200)
    m,stats=train_neural(pp.transform(raw),subset,config,'software_real_subset_overfit_sanity',pp.n_cont,fixed_steps=200)
    assert stats['train_BCE']<.15 and stats['train_AP']>.95
    dump(RESULT/'optimization_sanity.json',dict(passed=True,training_subset_rows=64,training_subset_AP=stats['train_AP'],training_subset_BCE=stats['train_BCE'],scope='Optimization sanity only on a deliberately balanced training subset; not a performance result or independent experiment'))
    results.append('real_subset_overfit_sanity')
    dump(RESULT/'software_contracts.json',dict(passed=True,checks=len(results),names=results,checked_at_utc=now(),scope='Software/numerical contracts; no synthetic research scores'))
    print('Passed V2 contracts:',len(results),flush=True)

def equal_updates():
    cfg=base('legacy_residual',features='RAW18',preprocess='std01',indicators=False,width=192,depth=4,dropout=.25,learning_rate=.001,weight_decay=.0002,batch_size=512,weight_alpha=0,fixed_training_steps=2000)
    for sampling in ['none','ros','smote']:
        execute({**cfg,'sampling':sampling},'bridge_equal2000_'+sampling,'bridge',[2012])

if __name__=='__main__':contracts();equal_updates()
