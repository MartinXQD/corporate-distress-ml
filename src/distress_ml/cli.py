"""Explicit research stages; historical scoring requires a verified freeze."""
import os
import sys
import importlib.util
from pathlib import Path
# macOS wheels can otherwise load incompatible OpenMP runtimes from Conda
# and PyTorch. Re-exec before importing either training stack.
if sys.platform=='darwin' and os.environ.get('DISTRESS_OMP_READY')!='1':
    torch_spec=importlib.util.find_spec('torch')
    if torch_spec:
        os.environ['DYLD_LIBRARY_PATH']=str(Path(torch_spec.origin).parent/'lib')
        os.environ['DISTRESS_OMP_READY']='1'
        os.execv(sys.executable,[sys.executable,'-m','distress_ml.cli',*sys.argv[1:]])
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','2')
import argparse
import json

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('audit'); sub.add_parser('smoke')
    for name in ['develop','train']:
        p=sub.add_parser(name); p.add_argument('--stage',default='all',choices=['all','core','search','ablations'])
    sub.add_parser('freeze'); sub.add_parser('historical-evaluate'); sub.add_parser('report')
    sub.add_parser('sequences')
    p=sub.add_parser('predict'); p.add_argument('--model',required=True); p.add_argument('--input',required=True); p.add_argument('--output',required=True)
    p=sub.add_parser('reproduce'); p.add_argument('--model',required=True)
    args=parser.parse_args()
    if args.command=='audit':
        from .audit import audit
        print(json.dumps(audit(),indent=2,default=int))
    elif args.command=='smoke':
        from .develop import pilot
        pilot()
    elif args.command in ['develop','train']:
        from .develop import develop
        develop(args.stage)
    elif args.command=='freeze':
        from .finalize import freeze
        freeze()
    elif args.command=='sequences':
        from .develop import optional_sequences
        optional_sequences()
    elif args.command=='historical-evaluate':
        from .finalize import historical_evaluate
        historical_evaluate()
    elif args.command=='report':
        from .reporting import report
        report()
    elif args.command=='predict':
        import pandas as pd
        from .train import load_pipeline
        from .calibration import calibrated_predict
        from .common import ART,protocol,sha
        output=Path(args.output).resolve()
        if not output.is_relative_to(ART.resolve()):
            raise ValueError('Prediction outputs must remain under the ignored local artifacts directory')
        if output.exists():raise FileExistsError('Refusing to overwrite an existing prediction output')
        output.parent.mkdir(parents=True,exist_ok=True)
        p=load_pipeline(args.model)
        names=pd.read_csv(args.input,nrows=0).columns
        df=pd.read_csv(args.input,usecols=[name for name in names if name not in ['status_label','y','y_true','bankrupt']])
        raw,cal=calibrated_predict(p,df)
        result=df[['company_name','year']].copy().rename(columns={'company_name':'company_id'})
        result.insert(0,'row_id',range(len(df)))
        result=result.assign(protocol_id=protocol()['protocol_id'],training_dataset_hash=protocol()['implementation']['source_data_hash'],input_file_hash=sha(args.input),raw_score=raw,calibrated_score=cal,fixed_threshold=p.threshold,predicted_label=(cal>=p.threshold).astype(int))
        result.to_csv(output,index=False)
    elif args.command=='reproduce':
        from .finalize import reproduce
        reproduce(args.model)

if __name__=='__main__': main()
