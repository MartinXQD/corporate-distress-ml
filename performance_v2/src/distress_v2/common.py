"""Small durable state and time guards for the registered performance run."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import sqlite3
import time
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[2]
V1=ROOT.parent
SOURCE=V1.parent
ART=ROOT/'artifacts'
RESULT=ROOT/'results'

def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
    temporary.replace(path)
def protocol():
    assert sha(ROOT/'V2_PROTOCOL.yaml')==json.loads((ROOT/'configs/protocol_registration.json').read_text())['sha256']
    return yaml.safe_load((ROOT/'V2_PROTOCOL.yaml').read_text())
def state(phase,**kw):
    p=ROOT/'EXECUTION_STATE.json';d=json.loads(p.read_text());d.update(phase=phase,updated_at_utc=now(),**kw);dump(p,d)
def hours():
    d=json.loads((ROOT/'EXECUTION_STATE.json').read_text())
    start=d['started_at_utc']
    if d.get('active_window_registration'):
        p=ROOT/d['active_window_registration']
        assert sha(p)==d['active_window_registration_sha256'],'Continuation registration changed'
        window=json.loads(p.read_text())
        assert window['explicit_user_authorization'] and window['maximum_hours']==8
        assert window['protocol_sha256']==d['protocol_sha256']
        start=window['started_at_utc']
    return (dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(start)).total_seconds()/3600
def time_guard(limit=7.25):
    if hours()>=limit: raise TimeoutError(f'Registered wall-clock phase limit reached ({limit} hours)')
def fit_event(tag,df,cfg,**kw):
    time_guard(7.25);protocol()
    if (ART/'historical_opened.json').exists():raise RuntimeError('No new model fitting after the frozen historical batch opens')
    p=ART/'logs/fits.jsonl';n=sum(1 for _ in p.open()) if p.exists() else 0
    if n>=2400:raise RuntimeError('Registered V2 fit-call cap reached')
    run_state=json.loads((ROOT/'EXECUTION_STATE.json').read_text())
    event=dict(fit_call=n+1,at_utc=now(),window=run_state.get('active_window_registration','original'),tag=tag,rows=len(df),max_fit_year=int(df.year.max()),row_hash=digest(sorted(df.row_id.tolist())),company_hash=digest(sorted(df.company_name.unique())),config_hash=digest(cfg),**kw)
    with p.open('a') as f:f.write(json.dumps(event,default=str)+'\n')
    state('model_fitting',fits=n+1,current_fit_tag=tag)
    return n+1
def database():
    c=sqlite3.connect(ART/'search.sqlite',timeout=30)
    c.execute('PRAGMA journal_mode=WAL')
    c.execute('CREATE TABLE IF NOT EXISTS trial_records (trial_id TEXT PRIMARY KEY, family TEXT, stage TEXT, status TEXT, config_json TEXT, started TEXT, finished TEXT, seconds REAL, macro_AP REAL, macro_ROC REAL, recall5 REAL, error TEXT)')
    c.commit();return c

