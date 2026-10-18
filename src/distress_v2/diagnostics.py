"""Bounded configuration/metric/device checks; no new historical selection."""
import importlib.util
import json
import sys
import time
import zipfile
import io
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from .common import ROOT,V1,SOURCE,ART,RESULT,dump,sha,now
from .data import load,features
from .networks import LegacyResidual,NumericNetwork

def inspect_sources():
    df=load();records=[]
    legacy=SOURCE/'results/nn_team_handoff_clean_20260522/src/neural_network_training.py'
    spec=importlib.util.spec_from_file_location('legacy_training_for_metric_contract',legacy)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    from distress_ml.train import load_pipeline
    freeze=json.loads((V1/'configs/final_freeze.json').read_text())
    config_rows=[]
    for rep in ['ML','DL']:
        item=next(m for m in freeze['models'] if rep in m['representative'] and m['seed']==42)
        model=load_pipeline(V1/item['path']);f=__import__('distress_ml.features',fromlist=['build_features']).build_features(df,model.cfg['feature_set'])
        mask=df.year>=2015;raw=model.predict_raw(df.loc[mask],f.loc[mask])
        saved=pd.read_csv(V1/f'artifacts/predictions/{item["label"]}_historical.csv.gz',float_precision='round_trip')
        assert np.array_equal(saved.row_id,df.loc[mask,'row_id'])
        np.testing.assert_allclose(raw,saved.raw_score,rtol=1e-6,atol=1e-6)
        y=saved.y_true.to_numpy();score=saved.calibrated_score.to_numpy()
        a=module.average_precision_score(y,score);b=average_precision_score(y,score)
        assert np.isclose(a,b,atol=1e-12)
        records.append(dict(check='actual_saved_V1_'+rep,legacy_AP=a,sklearn_AP=b,absolute_difference=abs(a-b),prediction_max_error=float(np.max(abs(raw-saved.raw_score))),source_identity='Reloaded trusted V1 model; previously saved labels/scores, not a new V2 result'))
        config_rows.append(dict(version='V1_selected_'+rep,**model.cfg,parameters=model.metadata.get('parameters'),input_dim=len(model.columns),epochs=model.metadata['epochs'],fit_cutoff=2012,calibration_year=2013,threshold_year=2014,configuration_identity=item['config_id'],refit='inner-tail length then full-window epoch refit',source='Saved trusted V1 artifact'))
    y=np.array([0,1,1,0,1,0,0,1]);scores=np.array([.8,.8,.4,.4,.4,.2,.1,.1])
    a=module.average_precision_score(y,scores);b=average_precision_score(y,scores);assert np.isclose(a,b,atol=1e-12)
    records.append(dict(check='synthetic_ties',legacy_AP=a,sklearn_AP=b,absolute_difference=abs(a-b),source_identity='Synthetic metric contract only'))
    pd.DataFrame(records).to_csv(RESULT/'metric_contract.csv',index=False)
    pd.DataFrame(config_rows).to_csv(RESULT/'legacy_vs_v1_config.csv',index=False)
    z=zipfile.ZipFile(SOURCE/'Final Project.zip')
    names=[n for n in z.namelist() if n.endswith('.pt')]
    path=ART/'audit/legacy_residual_base.pt'
    if not path.exists():path.write_bytes(z.read('Final Project/final_results/checkpoints/residual_mlp_base_scaled.pt'))
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    for d,expected in [(18,302209),(13,301249)]:
        m=LegacyResidual(d);assert sum(p.numel() for p in m.parameters())==expected
    model=LegacyResidual(18)
    converted={k.replace('.block.','.'):v for k,v in checkpoint['model_state_dict'].items()}
    model.load_state_dict(converted,strict=True)
    archived=pd.read_csv(SOURCE/'results/nn_team_handoff_clean_20260522/outputs/neural_networks/neural_network_comparison.csv')
    archived=archived[(archived.model=='residual_mlp')&archived.feature_set.isin(['base_scaled','lasso_selected_scaled'])]
    safe=archived.drop(columns=['checkpoint_path']);safe.to_csv(RESULT/'legacy_archived_metrics.csv',index=False)
    final=pd.read_csv(io.BytesIO(z.read('Final Project/final_results/final_model_comparison.csv')))
    final.to_csv(ART/'audit/legacy_final_model_comparison.csv',index=False)
    dump(ART/'audit/legacy_checkpoint_inventory.json',dict(checkpoint_sha256=sha(path),archive_checkpoint_names=names,loaded_weights_only=True,loaded_strict=True,best_epoch=checkpoint['best_epoch'],archive_report_base_epoch=23,identity_warning='The included final_results checkpoint is epoch31, unlike the bridge-report epoch23 row. They are different run artifacts; do not call their scores reproduced before checking the matching archive table.',architecture_parameters=302209))
    dump(RESULT/'diagnostic_source_checks.json',dict(passed=True,metric_definitions_agree=True,v1_representatives_reloaded=True,legacy_parameter_counts_verified=True,legacy_checkpoint_present=True,legacy_checkpoint_epoch=int(checkpoint['best_epoch']),checkpoint_matches_reported_epoch23_run=False,versions=json.loads((ROOT/'configs/runtime_versions.json').read_text()),at_utc=now()))

def profile():
    torch.manual_seed(42);x=torch.randn(512,17);y=torch.arange(512)%2;rows=[]
    for threads in [2,4]:
        torch.set_num_threads(threads);m=LegacyResidual(17);opt=torch.optim.AdamW(m.parameters(),lr=.001)
        started=time.perf_counter()
        for _ in range(12):
            opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(m(x),y.float());loss.backward();opt.step()
        rows.append(dict(device='cpu',threads=threads,updates=12,seconds=time.perf_counter()-started,finite_loss=bool(torch.isfinite(loss)),final_loss=float(loss.detach())))
    selected=min(rows,key=lambda r:r['seconds'])['threads']
    dump(ROOT/'configs/device_profile.json',dict(selected_device='cpu',selected_threads=selected,MPS_available=torch.backends.mps.is_available(),CUDA_available=torch.cuda.is_available(),benchmark=rows,scope='Synthetic optimizer/device throughput, not research scores'))
    torch.set_num_threads(selected)
    return selected

def run():
    inspect_sources();threads=profile();print('Diagnostic checks passed; selected CPU threads:',threads,flush=True)

if __name__=='__main__':run()
