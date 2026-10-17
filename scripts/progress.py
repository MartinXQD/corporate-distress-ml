"""Persist a compact checkpoint without changing experiment choices."""
from pathlib import Path
import json,datetime
import pandas as pd
r=Path(__file__).resolve().parents[1]
s=json.loads((r/'RUN_STATUS.json').read_text())
p=r/'artifacts/logs/fit_ledger.jsonl'
n=len(p.read_text().splitlines()) if p.exists() else 0
f=r/'results/public/development_yearly.csv'
d=pd.read_csv(f) if f.exists() else pd.DataFrame()
folds=d.run_id.nunique() if len(d) else 0
configs=d.groupby('config_id').year.nunique().eq(7).sum() if len(d) else 0
text=f'''# Experiment status - active checkpoint

Updated UTC: {datetime.datetime.now(datetime.timezone.utc).isoformat()}

Current phase: {s['phase']}. Complete: false.
Recorded actual model fit calls: {n}/600. Completed scoring folds: {folds}. Configurations with all seven development years: {configs}.

The first XGBoost pilot fit crashed from an OpenMP conflict. The shared local runtime repair passed combined-operation checks and the repeat pilot. All formal candidate choices remain under the frozen protocol. No historical model result has been used for development. See the durable fit ledger and fold-completion records for exact recovery state.

Next: finish development and controlled ablations, run registered sequence controls, freeze final configurations and artifacts, perform the one-time historical/company evaluation, then generate and inspect reports, execute notebooks and audit the public release. No paid compute or original-source writes.
'''
(r/'EXPERIMENT_STATUS.md').write_text(text)
(r/'NEXT_ACTIONS.md').write_text('# Next actions\n\n'+text.split('Next: ',1)[-1])
print(s['phase'],n,folds,configs)
