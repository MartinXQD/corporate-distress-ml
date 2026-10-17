"""Resume actual experiments without resetting the original eight-hour clock."""
import argparse
import json
import traceback
from .common import ROOT,ART,dump,state,hours,now

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['all','search','finalize','report'],default='all');args=parser.parse_args()
    current=json.loads((ROOT/'EXECUTION_STATE.json').read_text())
    if current.get('complete'):print('This registered run is complete; preserve its freeze.');return
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
        state('interrupted_or_budget_limited',complete=False,error=type(exc).__name__+': '+str(exc),elapsed_hours=hours(),resume_command='scripts/resume_performance_v2.sh --phase '+args.phase)
        with (ART/'logs/runner_failures.log').open('a') as f:f.write(now()+'\n'+traceback.format_exc()+'\n')
        raise

if __name__=='__main__':main()
