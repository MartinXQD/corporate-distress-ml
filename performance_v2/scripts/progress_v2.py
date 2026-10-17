"""Read durable experiment evidence without inferring results from CPU use."""
from pathlib import Path
import json
import sqlite3
import datetime

root=Path(__file__).resolve().parents[1]
state=json.loads((root/'EXECUTION_STATE.json').read_text())
fits=root/'artifacts/logs/fits.jsonl'
events=[json.loads(s) for s in fits.read_text().splitlines()] if fits.exists() else []
db=root/'artifacts/search.sqlite';counts=[]
if db.exists():
    connection=sqlite3.connect(db)
    counts=connection.execute('SELECT family,stage,status,count(*) FROM trial_records GROUP BY family,stage,status').fetchall()
    connection.close()
print(json.dumps(dict(phase=state['phase'],complete=state['complete'],hours_elapsed=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(state['started_at_utc'])).total_seconds()/3600,deadline_utc=state['deadline_utc'],fit_attempts=len(events),latest_fit=events[-1]['tag'] if events else None,trial_counts=counts),indent=2))
