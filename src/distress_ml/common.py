"""Durable state, hashes and local resource accounting."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "results/public"
ART = ROOT / "artifacts"

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")
    tmp.replace(path)

def protocol():
    p = yaml.safe_load((ROOT / "configs/protocol_v1.yaml").read_text())
    registered = json.loads((ROOT / "configs/protocol_registration.json").read_text())
    assert sha(ROOT / "configs/protocol_v1.yaml") == registered["protocol_sha256"], "Registration changed"
    return p

def state(phase, **kwargs):
    path = ROOT / "RUN_STATUS.json"
    old = json.loads(path.read_text()) if path.exists() else {"started_at_utc": now()}
    old.update(phase=phase, updated_at_utc=now(), **kwargs)
    dump(path, old)

def fit_event(tag, rows, companies=None, **kwargs):
    """Count every actual model fit, including inner search and stopping refits."""
    protocol()
    path = ART / "logs/fit_ledger.jsonl"
    lines = path.read_text().splitlines() if path.exists() else []
    if len(lines) >= 600:
        raise RuntimeError("Hard fit-call budget reached")
    start = json.loads((ROOT / "RUN_STATUS.json").read_text())["started_at_utc"]
    elapsed = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(start)).total_seconds()
    if elapsed > 7.25 * 3600:
        raise RuntimeError("Delivery reserve reached; no new model fits")
    event = dict(fit_call=len(lines)+1, at_utc=now(), tag=tag, rows=int(rows), pid=os.getpid(), **kwargs)
    if companies is not None:
        event["company_hash"] = object_hash(sorted(set(map(str, companies))))
        event["companies"] = len(set(companies))
    with path.open("a") as f:
        f.write(json.dumps(event)+"\n")
    return event["fit_call"]
