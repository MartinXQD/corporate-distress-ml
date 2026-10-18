"""Separate old-role reconstruction, never a P2 development candidate."""
import json
import importlib.util
import sys
import time
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from .common import ROOT,SOURCE,ART,dump,digest,fit_event,now
from .data import features
from .preprocessing import NumericTransform
from .networks import NumericNetwork
from .training import Fitted,train_neural,save
from .search import base

def reconstruct(df):
    assert (ART/'confirmation_opened.json').exists()
    # Original preprocessing and DataLoader inputs were sorted by these keys.
    # Preserve source row IDs, but reproduce that ordering for archive roles.
    df=df.sort_values(['year','company_name'])
    output=[];train=df.year<=2011;val=df.year.between(2012,2014)
    for kind in ['legacy_fixed13_retrained','included_base_checkpoint_epoch31']:
        path=ART/f'models/V0_{kind}.joblib'
        name='LEGACY_FIXED13' if kind=='legacy_fixed13_retrained' else 'RAW18'
        cfg=base('legacy_residual',features=name,preprocess='std01',indicators=False,width=192,depth=4,dropout=.25,learning_rate=.001,weight_decay=.0002,batch_size=512,weight_alpha=1,max_epochs=100,patience=12,gradient_clip=None)
        if not path.exists():
            f=features(df,name).loc[df.index];pp=NumericTransform('std01',indicators=False).fit(f[train]);a=pp.transform(f[train]);b=pp.transform(f[val])
            if kind=='legacy_fixed13_retrained':
                spec=importlib.util.spec_from_file_location('legacy_original_archive_training',SOURCE/'results/nn_team_handoff_clean_20260522/src/neural_network_training.py')
                original=importlib.util.module_from_spec(spec);sys.modules[spec.name]=original;spec.loader.exec_module(original)
                # The quadratic F1 monitoring loop never selects checkpoints.
                # Replace that diagnostic only; retain original optimizer,
                # DataLoader RNG, BCE pos_weight, AP stopping and restoration.
                original.select_threshold_by_f1=lambda y,q:(.5,original.binary_classification_metrics(y,q,.5))
                original.set_seed(42);body=NumericNetwork(13,13,cfg).body
                data=SimpleNamespace(x_train=pd.DataFrame(a),y_train=df.loc[train,'y'].astype(int).reset_index(drop=True),x_val=pd.DataFrame(b),y_val=df.loc[val,'y'].astype(int).reset_index(drop=True))
                training=original.TrainingConfig(batch_size=512,max_epochs=100,patience=12,learning_rate=.001,weight_decay=.0002,seed=42)
                fit_event('V0_original_fixed13_archive_retrain',df[train],cfg,stopping_years=[2012,2013,2014],identity='Original code with diagnostic-only F1 monitoring shortcut')
                started=time.perf_counter();body,history,best=original.train_supervised_model(body,data,training,torch.device('cpu'),float((data.y_train==0).sum()/(data.y_train==1).sum()),ART/'audit/legacy_fixed13_original_loop.pt',cfg)
                model=NumericNetwork(13,13,cfg);model.body.load_state_dict(body.state_dict());model.eval()
                stats=dict(best_epoch=best,actual_updates=len(history)*int(np.ceil(len(a)/512)),best_steps=best*int(np.ceil(len(a)/512)),parameters=sum(p.numel() for p in model.parameters()),fit_seconds=time.perf_counter()-started)
                pd.DataFrame(history).to_csv(ART/'logs/V0_original_fixed13_archive_curve.csv',index=False)
                identity='Original-loop retraining: fixed archival13, no gradient clipping, 2012-2014 AP stopping; F1 diagnostic computation shortened without entering training/selection. Not strict P1/P2 or checkpoint reproduction.'
            else:
                # Verify the complete preprocessing path against original saved matrices.
                old=pd.read_csv(SOURCE/'data/processed/02_X_scaled_for_LR_SVM_ANN/X_train_base_scaled.csv')
                ids=pd.read_csv(SOURCE/'data/processed/ids_train.csv')
                assert np.array_equal(ids.company_name.to_numpy(),df.loc[train,'company_name'].to_numpy())
                assert np.array_equal(ids.year.to_numpy(),df.loc[train,'year'].to_numpy())
                np.testing.assert_allclose(a,old.to_numpy(),atol=5e-5,rtol=5e-6)
                checkpoint=torch.load(ART/'audit/legacy_residual_base.pt',map_location='cpu',weights_only=True)
                model=NumericNetwork(18,18,cfg)
                converted={key.replace('.block.','.'):value for key,value in checkpoint['model_state_dict'].items()}
                model.body.load_state_dict(converted,strict=True);model.eval();stats=dict(best_epoch=int(checkpoint['best_epoch']),parameters=sum(p.numel() for p in model.parameters()),actual_updates=None)
                identity='Included epoch31 checkpoint scored with legacy preprocessing verified against the original saved training matrix; differs from the bridge epoch23 artifact'
            p=Fitted();p.cfg=cfg;p.pp=pp;p.selected=list(f);p.model=model
            p.metadata=dict(tag='V0_'+kind,config_hash=digest(cfg),fit_seconds=stats.get('fit_seconds'),fit_rows=int(train.sum()),max_fit_year=2011,requested_fit_year=2011,row_hash=digest(sorted(df.loc[train,'row_id'].tolist())),company_hash=digest(sorted(df.loc[train,'company_name'].unique())),columns=pp.output_columns,selected_columns=list(f),inner_stop_year='2012-2014 archive validation',inner_stats={k:v for k,v in stats.items() if k!='history'},final_stats={k:v for k,v in stats.items() if k!='history'},source_identity=identity,generated_at_utc=now())
            save(p,path)
        output.append(dict(name='V0_'+kind,candidate=kind,channel='V0_archive_role',fit_cutoff=2011,seed=42,path=str(path.relative_to(ROOT)),score_type='uncalibrated legacy risk score; separate archived selection roles'))
    return output
