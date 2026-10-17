"""Verify an authorized local snapshot without downloading or redistributing it."""
import argparse
import hashlib
from pathlib import Path
EXPECTED='cff2c899a97ecd629415cb22f59186000e74e1c0a78cfae036c0a53025419b5e'
p=argparse.ArgumentParser();p.add_argument('csv',type=Path);a=p.parse_args()
h=hashlib.sha256(a.csv.read_bytes()).hexdigest()
if h!=EXPECTED:raise SystemExit('Snapshot mismatch; register and audit a new version instead of silently replacing data.')
print('Snapshot hash verified. Set DISTRESS_DATA to this file for local reproduction.')
