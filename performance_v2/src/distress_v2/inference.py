"""Score authorized input records using a verified local frozen pipeline."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT,ART,sha
from .data import features
from .training import load

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);parser.add_argument('--input',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    model=Path(args.model).resolve();output=Path(args.output).resolve()
    assert output.is_relative_to(ART.resolve()) and not output.exists()
    frozen=json.loads((ROOT/'configs/artifact_freeze.json').read_text())['artifacts'];rel=str(model.relative_to(ROOT))
    assert rel in frozen and sha(model)==frozen[rel]
    names=pd.read_csv(args.input,nrows=0).columns
    df=pd.read_csv(args.input,usecols=[c for c in names if c not in ['y','y_true','status_label','bankrupt']]);df['row_id']=np.arange(len(df))
    assert not df.duplicated(['company_name','year']).any()
    pipeline=load(model);score=pipeline.predict(features(df,pipeline.cfg['features']))
    result=df[['row_id','company_name','year']].copy();result['score']=score;result['score_type']='uncalibrated P2 ranking score'
    output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(output,index=False)
    print(json.dumps(dict(rows=len(df),written_under_ignored_artifacts=True,model_hash_verified=True,labels_ignored=True)))

if __name__=='__main__':main()
