"""Recompute scores, role isolation and original hashes from saved evidence."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from distress_ml.common import ROOT,ART,PUBLIC,sha,dump,object_hash
from distress_ml.data import load_data
from distress_ml.splits import groups,row_hash
from distress_ml.evaluation import metrics
from distress_ml.finalize import verify_freeze

def main():
    freeze=verify_freeze();df=load_data('historical');hist=df[df.year>=2015]
    checks=[];t=pd.read_csv(PUBLIC/'historical_metrics.csv');t=t[t.score_type=='calibrated']
    for item in freeze['models']:
        d=pd.read_csv(ART/f'predictions/{item["label"]}_historical.csv.gz')
        sub=d.rename(columns={'company_id':'company_name','y_true':'y'})
        if item['role']!='untuned_group_stress':
            assert set(sub.row_id)==set(hist.row_id)
        else:
            expected=hist[groups(hist)==item['group_fold']]
            assert set(sub.row_id)==set(expected.row_id)
        original=df.set_index('row_id').loc[sub.row_id]
        assert np.array_equal(original.y.to_numpy(),sub.y.to_numpy())
        assert np.array_equal(original.company_name.to_numpy(),sub.company_name.to_numpy())
        assert np.array_equal(original.year.to_numpy(),sub.year.to_numpy())
        m=metrics(sub,sub.calibrated_score.to_numpy(),sub.fixed_threshold.iloc[0])
        if item['role']=='primary':
            saved=t[t.label==item['label']].iloc[0]
            for metric in ['AP','ROC_AUC','precision','recall','F1','TP','FP','FN','TN','Brier','log_loss','macro_recall_top5']:
                assert np.isclose(m[metric],saved[metric],rtol=1e-7,atol=1e-9),(item['label'],metric)
        checks.append(dict(model=item['label'],prediction_rows=len(d),alignment=True,recomputed_metrics=True))
    ledger=[json.loads(x) for x in (ART/'logs/fit_ledger.jsonl').read_text().splitlines()]
    group_events=[]
    for event in ledger:
        if not event['tag'].startswith('stress_group'):continue
        fold=int(event['tag'][12]);pool=df[groups(df)!=fold]
        fitting=pool[pool.year==2013] if 'Platt2013' in event['tag'] else pool[pool.year<=event['max_year']]
        assert object_hash(sorted(fitting.company_name.unique()))==event['company_hash']
        if 'row_hash' in event:assert row_hash(fitting)==event['row_hash']
        group_events.append(event['fit_call'])
    # Verify every non-calibration fit's maximum fiscal year remains <=2012.
    for event in ledger:
        assert event.get('max_year',2012)<= (2013 if 'Platt2013' in event['tag'] else 2012)
    original=json.loads((ART/'audit/original_manifest.json').read_text())
    changes=[]
    for rel,h in original.items():
        path=ROOT.parent/rel
        if not path.exists() or sha(path)!=h:changes.append(rel)
    assert not changes,'Original source changed: '+repr(changes)
    result=dict(passed=True,models=len(checks),prediction_checks=checks,group_fit_events_checked=group_events,actual_fit_calls=len(ledger),fit_budget_passed=len(ledger)<=600,original_files_verified=len(original),original_changes=changes,frozen_artifact_hashes_verified=len(freeze['artifact_hashes']),all_primary_same_historical_rows=True)
    dump(PUBLIC/'verification_summary.json',{k:v for k,v in result.items() if k!='prediction_checks'})
    dump(ART/'audit/result_verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['prediction_checks','group_fit_events_checked']},indent=2))

if __name__=='__main__':main()
