"""Reconstruct NN pair draws from saved steps and verify exposure counters."""
import json
import re
import numpy as np
import pandas as pd
from distress_v2.common import ART,RESULT,digest,dump,now,database
from distress_v2.data import load

def replay(frame,cfg,updates):
    y=frame.y.to_numpy(int);years=frame.year.to_numpy()
    rng=np.random.default_rng(cfg['seed']);perm=rng.permutation(len(y));cursor=0
    batch=cfg['batch_size'];positive_count=0;negative_count=0;pairs=0;queries=set()
    for _ in range(updates):
        if cursor>=len(perm):perm=rng.permutation(len(y));cursor=0
        ix=perm[cursor:cursor+batch];cursor+=batch
        local_y=y[ix];local_years=years[ix]
        positive_count+=int(local_y.sum());negative_count+=int((1-local_y).sum())
        for year in np.unique(local_years):
            positive=np.flatnonzero((local_years==year)&(local_y==1))
            negative=np.flatnonzero((local_years==year)&(local_y==0))
            if len(positive) and len(negative):
                a=rng.choice(positive,size=min(16,len(positive)),replace=True)
                rng.choice(negative,size=len(a),replace=True)
                pairs+=len(a);queries.add(int(year))
    return dict(positive_exposures=positive_count,negative_exposures=negative_count,
                pair_draws=pairs,effective_annual_queries=len(queries),
                minimum_query_year=min(queries),maximum_query_year=max(queries))

def main():
    df=load('search');db=database()
    trials=db.execute("SELECT trial_id,config_json FROM trial_records WHERE status='COMPLETE' AND stage='refinement'").fetchall()
    db.close()
    ledger=[json.loads(line) for line in (ART/'logs/fits.jsonl').read_text().splitlines()]
    events={event['tag']:event for event in ledger}
    rows=[]
    for candidate,raw in trials:
        cfg=json.loads(raw)
        if not cfg.get('pair_lambda',0):continue
        assert cfg.get('sampling','none')=='none' and cfg['refit_rule']=='steps'
        for path in sorted((ART/'logs').glob(candidate+'__seed42__year*_complete.json')):
            metadata=json.loads(path.read_text());year=int(re.search(r'__year(\d+)$',metadata['tag']).group(1))
            outer=df[df.year<year]
            if cfg.get('window'):outer=outer[outer.year>outer.year.max()-cfg['window']]
            for role,key in [('stop','inner_stats'),('refit','final_stats')]:
                frame=outer[outer.year<outer.year.max()] if role=='stop' else outer
                event=events[metadata['tag']+'/'+role]
                assert digest(sorted(frame.row_id.tolist()))==event['row_hash']
                observed=metadata[key];counts=replay(frame,cfg,int(observed['actual_updates']))
                assert counts['positive_exposures']==observed['positive_exposures']
                assert counts['negative_exposures']==observed['negative_exposures']
                rows.append(dict(candidate=candidate,score_year=year,role=role,
                    lambda_value=cfg['pair_lambda'],optimizer_updates=observed['actual_updates'],
                    natural_rows=len(frame),**counts,
                    source='Deterministic sampling-sequence replay, matched to saved actual update/exposure counters and training-row hash',
                    pair_draws_are_independent=False,model_training_performed=False))
    assert rows,'No completed NN ranking-proxy controls are available'
    pd.DataFrame(rows).to_csv(RESULT/'ranking_pair_accounting.csv',index=False)
    result=dict(passed=True,checked_at_utc=now(),verified_role_runs=len(rows),
                source='Replay of the unchanged NumPy sampling sequence using saved executed steps, seed and exact training rows; exposure counters match',
                direct_pair_logging=False,independent_observations_not_claimed=True,
                model_training_performed=False)
    dump(RESULT/'ranking_pair_accounting_verification.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
