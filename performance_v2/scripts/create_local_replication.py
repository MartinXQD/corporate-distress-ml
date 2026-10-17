"""Prepare a separate owner-invoked replication; never reset this run's clock."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import yaml

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description='Create a new sibling research directory using the same authorized source context. Does not train, publish, or change the original run.')
    parser.add_argument('--name',required=True);parser.add_argument('--prepare',action='store_true',help='Actually create the separate copy and register its new eight-hour clock; otherwise print a read-only preflight')
    args=parser.parse_args();assert re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,39}',args.name)
    dest=ROOT.parent/('performance_v2_replication_'+args.name)
    assert not dest.exists(),'Replication destination must be new'
    required=[ROOT/'.venv/bin/python',ROOT/'artifacts/audit/protected_sources.json',ROOT.parent.parent/'american_bankruptcy.csv',ROOT.parent.parent/'Final Project.zip',ROOT.parent.parent/'results/nn_team_handoff_clean_20260522/src/neural_network_training.py']
    missing=[str(p.relative_to(ROOT.parent.parent)) for p in required if not p.exists()]
    if missing:raise RuntimeError('Owner-supplied local runtime/source context is missing: '+json.dumps(missing))
    print(json.dumps(dict(destination=dest.name,will_reset_original_clock=False,will_train=False,will_publish=False,requires_owner_original_source_context=True,prepare_requested=args.prepare)))
    if not args.prepare:return
    dest.mkdir()
    for folder in ['src','scripts']:
        shutil.copytree(ROOT/folder,dest/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for filename in ['.gitignore','requirements_v2.txt','REFERENCES.md','CONTRIBUTIONS.md','PROTOCOL_EXECUTION_NOTES.md']:
        shutil.copy2(ROOT/filename,dest/filename)
    for folder in ['configs','results','reports','artifacts/audit','artifacts/logs','artifacts/models','artifacts/predictions','artifacts/cache']:(dest/folder).mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/'configs/runtime_versions.json',dest/'configs/runtime_versions.json')
    shutil.copy2(ROOT/'artifacts/audit/protected_sources.json',dest/'artifacts/audit/protected_sources.json')
    (dest/'.venv').symlink_to((ROOT/'.venv').resolve(),target_is_directory=True)
    start=dt.datetime.now(dt.timezone.utc);protocol=yaml.safe_load((ROOT/'V2_PROTOCOL.yaml').read_text());source_sha=hashlib.sha256((ROOT/'V2_PROTOCOL.yaml').read_bytes()).hexdigest()
    protocol.update(version='performance-v2-replication-'+args.name,registered_at_utc=start.isoformat(),replication_of_protocol_sha256=source_sha)
    protocol['implementation']['directory']=dest.name
    (dest/'V2_PROTOCOL.yaml').write_text(yaml.safe_dump(protocol,sort_keys=False))
    sha=hashlib.sha256((dest/'V2_PROTOCOL.yaml').read_bytes()).hexdigest()
    registration=dict(sha256=sha,registered_at_utc=start.isoformat(),replication_of_protocol_sha256=source_sha,original_clock_unchanged=True)
    (dest/'configs/protocol_registration.json').write_text(json.dumps(registration,indent=2))
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    state=dict(phase='separate_replication_registered_not_started',complete=False,started_at_utc=start.isoformat(),deadline_utc=(start+dt.timedelta(hours=8)).isoformat(),protocol_sha256=sha,branch=branch,resume_command='scripts/resume_performance_v2.sh --phase all',fits=0,paid_compute=False,replication=True)
    (dest/'EXECUTION_STATE.json').write_text(json.dumps(state,indent=2))
    (dest/'NEXT_ACTIONS.md').write_text('# Separate replication\n\nRun scripts/resume_performance_v2.sh --phase all in this new directory. Original private source context and current local runtime are required. The new eight-hour clock includes setup. No publication is automatic. Earlier bridge snapshots and adaptive wall-budget HPO counts need not be bit-identical to this corrected full run; original commit/config/prediction evidence remains authoritative.\n')
    print('Prepared '+dest.name+'; no training process started.')

if __name__=='__main__':main()
