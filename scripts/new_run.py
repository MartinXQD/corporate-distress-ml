"""Initialize an empty isolated reproduction, preserving all existing evidence."""
from pathlib import Path
import argparse
import datetime
import json
import shutil

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--destination',type=Path,required=True)
a=p.parse_args();destination=a.destination.resolve()
if destination.exists():raise SystemExit('Destination already exists; choose a new directory. Nothing was overwritten.')
destination.mkdir(parents=True)
for name in ['src','scripts','tests','docs']:
    shutil.copytree(ROOT/name,destination/name,ignore=shutil.ignore_patterns('__pycache__'))
for name in ['pyproject.toml','requirements-lock.txt','.gitignore','.env.example','ACKNOWLEDGEMENTS.md','LICENSE_NOTICE.md']:
    shutil.copy2(ROOT/name,destination/name)
for name in ['artifacts/logs','artifacts/models','artifacts/predictions','artifacts/audit','artifacts/cache','artifacts/rendered','results/public','reports','notebooks','configs']:
    (destination/name).mkdir(parents=True,exist_ok=True)
for name in ['protocol_v1.yaml','protocol_registration.json','sequence_extension.yaml']:
    if (ROOT/'configs'/name).exists():shutil.copy2(ROOT/'configs'/name,destination/'configs'/name)
if (ROOT/'.venv').exists():(destination/'.venv').symlink_to(ROOT/'.venv',target_is_directory=True)
state=dict(phase='isolated_reproduction_initialized',started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),complete=False,prior_evidence_preserved=True)
(destination/'RUN_STATUS.json').write_text(json.dumps(state,indent=2))
(destination/'NEXT_ACTIONS.md').write_text('# Next actions\n\nSet DISTRESS_DATA to the authorized pinned raw CSV. Run audit, smoke, develop, sequences, freeze, historical-evaluate and report. This directory contains no prior experiment scores or freezes.\n')
print('Empty isolated reproduction initialized. Set DISTRESS_DATA before running its scripts/run.sh launcher.')
