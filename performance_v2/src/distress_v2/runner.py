"""Resume experiments within an immutable, explicitly authorized time window."""
import argparse
import json
import traceback
import sys
import os
from .common import ROOT,ART,dump,state,hours,now

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['all','search','finalize','report'],default='all');args=parser.parse_args()
    current=json.loads((ROOT/'EXECUTION_STATE.json').read_text())
    if current.get('complete'):print('This registered run is complete; preserve its freeze.');return
    state('runner_started',active_command='scripts/run_performance_v2.sh -u -m distress_v2.runner --phase '+args.phase,running_pid=os.getpid(),training_running=args.phase!='report',error=None,elapsed_hours=hours())
    try:
        if args.phase in ['all','search']:
            assert not (ART/'confirmation_opened.json').exists(),'Search cannot restart after confirmation opened'
            from .search import baselines,execute,search,bridge
            if not (ROOT/'results/diagnostic_source_checks.json').exists():
                from .diagnostics import run
                run()
            if not (ART/'bridge_complete.json').exists():
                bridge();dump(ART/'bridge_complete.json',dict(completed_at_utc=now()))
            from .contracts import contracts,equal_updates
            if not (ROOT/'results/software_contracts.json').exists():contracts()
            equal_updates()
            for name,cfg in baselines().items():execute(cfg,'baseline_'+name,'baseline',[2009,2011,2013])
            if (ART/'baseline_steps_correction_required.json').exists():
                from .common import database
                db=database();corrected=db.execute("SELECT status FROM trial_records WHERE trial_id='baseline_v1_selected_dl_corrected_steps'").fetchone();db.close()
                assert corrected and corrected[0]=='COMPLETE','Corrected migrated baseline must finish before selection'
                dump(ART/'baseline_steps_correction_applied.json',dict(at_utc=now(),corrected_trial='baseline_v1_selected_dl_corrected_steps',scope='Only V2 migrated V1 ROS epoch-equivalent refit; ordinary P2 step-refit search trials were unaffected'))
            search()
        if args.phase in ['all','finalize']:
            from .finalize import select_shortlist,confirm_and_fit,score_historical
            path=ROOT/'configs/selection_freeze.json'
            selected=json.loads(path.read_text()) if path.exists() else select_shortlist()
            if not (ROOT/'configs/artifact_freeze.json').exists():confirm_and_fit(selected)
            score_historical()
        if args.phase in ['all','report']:
            from .verify import verify
            from .reporting import report
            verify();report()
    except BaseException as exc:
        resume_phase=args.phase
        if (ROOT/'configs/shortlist.json').exists():resume_phase='finalize'
        if (ART/'historical_complete.json').exists():resume_phase='report'
        state('interrupted_or_budget_limited',complete=False,training_running=False,running_pid=None,error=type(exc).__name__+': '+str(exc),elapsed_hours=hours(),resume_command='scripts/resume_performance_v2.sh --phase '+resume_phase)
        with (ART/'logs/runner_failures.log').open('a') as f:f.write(now()+'\n'+traceback.format_exc()+'\n')
        raise

if __name__=='__main__':main()
