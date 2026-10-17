"""Correct linear cost labels from trusted saved estimators without fitting/scoring."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from distress_ml.common import ROOT,ART,PUBLIC,dump,now
from distress_ml.train import load_pipeline

path=PUBLIC/'development_yearly.csv'
t=pd.read_csv(path)
before=t[['run_id','year','AP','ROC_AUC']].copy()
backup=ART/'audit/development_yearly_before_cost_label_correction.csv'
if not backup.exists():shutil.copy2(path,backup)
corrections=[]
for rid in t.loc[t.family=='logistic','run_id'].unique():
    p=load_pipeline(ART/f'models/{rid}.joblib')
    iterations=int(np.max(p.model.n_iter_));parameters=int(p.model.coef_.size+p.model.intercept_.size)
    old=t.loc[t.run_id==rid,['epochs','parameters']].iloc[0].to_dict()
    t.loc[t.run_id==rid,'epochs']=iterations
    t.loc[t.run_id==rid,'parameters']=parameters
    corrections.append(dict(run_id=rid,old_cost_labels=old,actual_optimizer_iterations=iterations,actual_coefficients_including_intercept=parameters,evidence='Saved trusted Logistic estimator; no model fit or score recalculation'))
pd.testing.assert_frame_equal(before,t[['run_id','year','AP','ROC_AUC']])
t.to_csv(path,index=False)
dump(ART/'audit/cost_metadata_correction.json',dict(at_utc=now(),corrections=corrections,model_predictions_changed=False,metric_values_changed=False))
print('Corrected linear cost labels from saved estimators:',len(corrections))
