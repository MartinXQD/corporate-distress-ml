"""Run synthetic tests sequentially and count their model fits outside research scores."""
from pathlib import Path
import datetime
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
stamp=str(int(time.time()))
events=ROOT/f'artifacts/logs/unit_fit_events_{stamp}.jsonl'
junit=ROOT/f'artifacts/logs/unit_tests_{stamp}.xml'
env=os.environ.copy();env['DISTRESS_TEST_FIT_LOG']=str(events)
started=time.time()
r=subprocess.run([sys.executable,'-m','pytest','-q','--junitxml='+str(junit)],cwd=ROOT,env=env)
records=[json.loads(x) for x in events.read_text().splitlines()] if events.exists() else []
fits=sum(x['kind']=='model_fit' for x in records);updates=sum(x['kind']=='single_neural_update' for x in records)
account=ROOT/'artifacts/logs/software_fit_accounting.json'
old=json.loads(account.read_text()) if account.exists() else {'model_fit_calls':0}
old['model_fit_calls']+=fits
old['completed_final_unit_suite_model_fits']=old.get('completed_final_unit_suite_model_fits',0)+fits
old['synthetic_single_update_checks_in_accounted_suites']=old.get('synthetic_single_update_checks_in_accounted_suites',0)+updates
account.write_text(json.dumps(old,indent=2))
tree=ET.parse(junit).getroot();s=tree if tree.tag=='testsuite' else tree.find('testsuite')
result=dict(tests=int(s.attrib['tests']),failures=int(s.attrib['failures']),errors=int(s.attrib['errors']),skipped=int(s.attrib['skipped']),passed=r.returncode==0,seconds=time.time()-started,synthetic_model_fit_calls=fits,synthetic_single_neural_updates=updates,scope='Synthetic software contracts only; no research performance',command='scripts/run.sh scripts/test_with_accounting.py')
(ROOT/'results/public/tests_summary.json').write_text(json.dumps(result,indent=2))
raise SystemExit(r.returncode)
