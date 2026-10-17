"""Retry only recorded company-beta dtype failures before selection freezes."""
import json
import os
import pandas as pd
from distress_v2.common import ROOT,ART,RESULT,database,state,now,dump,time_guard
from distress_v2.search import execute

def main():
    assert not (ROOT/'configs/selection_freeze.json').exists()
    assert not (ART/'confirmation_opened.json').exists()
    assert not json.loads((ROOT/'EXECUTION_STATE.json').read_text()).get('training_running'),'Stop the recorded runner before the bounded retry'
    state('repairing_company_weighting',running_pid=os.getpid(),training_running=True,error=None)
    db=database()
    pending=db.execute("SELECT trial_id,config_json,error FROM trial_records WHERE status='FAIL' AND stage='refinement'").fetchall()
    db.close();records=[]
    try:
        for name,raw,error in pending:
            cfg=json.loads(raw)
            if cfg.get('company_beta')!=1 or 'Integers to negative integer powers' not in str(error):continue
            time_guard(5.9)
            status,value=execute(cfg,name,'refinement',[2009,2011,2013])
            records.append(dict(trial_id=name,status=status,macro_AP=value,original_failure_preserved=True))
            assert status=='COMPLETE',name
        trace_path=RESULT/'local_refinement_trace.csv'
        trace=pd.read_csv(trace_path)
        for row in records:
            mask=trace.trial_id==row['trial_id']
            trace.loc[mask,'status']=row['status'];trace.loc[mask,'macro_AP']=row['macro_AP']
            trace.loc[mask,'implementation_fix']='Floating-point company counts; original failed attempt retained'
        trace.to_csv(trace_path,index=False)
        dump(ART/'company_weight_repair.json',dict(completed_at_utc=now(),records=records,statistical_protocol_unchanged=True))
    finally:
        state('company_weight_repair_stopped',training_running=False,running_pid=None,active_command=None)

if __name__=='__main__':main()
