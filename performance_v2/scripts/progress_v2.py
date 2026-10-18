"""Read durable experiment evidence without inferring results from CPU use."""
from pathlib import Path
import json
import sqlite3
import datetime
from distress_v2.common import hours

root=Path(__file__).resolve().parents[1]
state=json.loads((root/'EXECUTION_STATE.json').read_text())
fits=root/'artifacts/logs/fits.jsonl'
events=[json.loads(s) for s in fits.read_text().splitlines()] if fits.exists() else []
db=root/'artifacts/search.sqlite';counts=[]
if db.exists():
    connection=sqlite3.connect(db)
    counts=connection.execute('SELECT family,stage,status,count(*) FROM trial_records GROUP BY family,stage,status').fetchall()
    connection.close()
print(json.dumps(dict(phase=state['phase'],complete=state['complete'],active_window_hours_elapsed=hours(),original_window_started_at_utc=state['started_at_utc'],original_deadline_utc=state['deadline_utc'],active_deadline_utc=state.get('active_deadline_utc',state['deadline_utc']),active_window_registration=state.get('active_window_registration','original'),fit_attempts=len(events),latest_fit=events[-1]['tag'] if events else None,trial_counts=counts),indent=2))
