"""Correct one diagnosed archive-only row permutation without refitting."""
import json
import argparse
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from distress_v2.common import ROOT,ART,RESULT,sha,dump,now
from distress_v2.data import load,features
from distress_v2.training import load as load_model
from distress_v2.finalize import historical_metrics

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['V0_legacy_fixed13_retrained','V0_included_base_checkpoint_epoch31'],default='V0_legacy_fixed13_retrained')
    name=parser.parse_args().model
    receipt=RESULT/('archive_alignment_correction.json' if name=='V0_legacy_fixed13_retrained' else 'archive_base_alignment_correction.json')
    assert not receipt.exists(),'This single correction has already been applied'
    freeze=json.loads((ROOT/'configs/artifact_freeze.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in freeze['artifacts'].items())
    assert sha(ROOT/'configs/selection_freeze.json')==freeze['selection_sha256']
    before_fits=(ART/'logs/fits.jsonl').read_bytes()
    df=load('historical');hist=df[df.year>=2015]
    item=next(m for m in freeze['models'] if m['name']==name)
    pipeline=load_model(ROOT/item['path'])
    matrix=features(df,pipeline.cfg['features']).loc[df.index]
    correct=pipeline.predict(matrix.loc[hist.index])
    ordered=df.sort_values(['year','company_name'])
    bad_order=ordered.loc[ordered.year>=2015].index
    old=pipeline.predict(matrix.loc[bad_order])
    path=ART/f'predictions/{name}_historical.csv.gz'
    saved=pd.read_csv(path,float_precision='round_trip')
    np.testing.assert_array_equal(hist.row_id,saved.row_id)
    np.testing.assert_allclose(old,saved.raw_score,atol=1e-6,rtol=1e-6)
    assert not np.allclose(correct,saved.raw_score,atol=1e-6)
    snapshot=ART/'audit'/('archive_alignment_before_correction_'+name);snapshot.mkdir()
    for p in [path,RESULT/'historical_2015_2018.csv',RESULT/'historical_by_year.csv',RESULT/'verification.json']:
        shutil.copy2(p,snapshot/p.name)
    old_metrics=historical_metrics(hist,saved.score.to_numpy())
    saved['score']=correct;saved['raw_score']=correct;saved.to_csv(path,index=False)
    for file in ['historical_2015_2018.csv','historical_by_year.csv']:
        table=pd.read_csv(RESULT/file,float_precision='round_trip')
        unrelated=table[table.name!=name].copy()
        for i,row in table[table.name==name].iterrows():
            mask=np.ones(len(hist),dtype=bool) if file=='historical_2015_2018.csv' else hist.year.to_numpy()==row.year
            for key,value in historical_metrics(hist.loc[mask],correct[mask]).items():table.loc[i,key]=value
        pd.testing.assert_frame_equal(unrelated,table[table.name!=name])
        table.to_csv(RESULT/file,index=False)
    assert before_fits==(ART/'logs/fits.jsonl').read_bytes()
    assert all(sha(ROOT/p)==h for p,h in freeze['artifacts'].items())
    dump(receipt,dict(passed=True,corrected_at_utc=now(),model=name,
        cause='Archive feature cache retained year/company order while score rows retained source order',
        evidence='Original misaligned scores reproduced by the exact year/company permutation; corrected scores use frozen pipeline and source-row order',
        old_metrics=old_metrics,corrected_metrics=historical_metrics(hist,correct),
        fits_added=0,selection_and_all_frozen_artifacts_unchanged=True,
        other_historical_model_rows_unchanged=True,
        scope='Archive-only inference alignment correction within the completed frozen batch; no reselection, refit, new historical search or metric-definition change'))
    print(receipt.read_text())

if __name__=='__main__':main()
