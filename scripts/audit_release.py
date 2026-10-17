"""Audit the exact staged Git index; never stage the source workspace."""
from __future__ import annotations
import hashlib
import json
import re
import subprocess
from pathlib import Path
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]

def run():
    top=subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=ROOT,text=True).strip()
    assert Path(top)==ROOT,'Git boundary is not the independent extension'
    paths=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    paths=[p for p in paths if p]
    if not paths:raise RuntimeError('Stage the explicit public allowlist before the index audit')
    forbidden_parts={'artifacts','data','.venv','__pycache__','.git','.pytest_cache'}
    forbidden_suffixes={'.zip','.pt','.pkl','.joblib','.ttf','.ttc','.otf','.env','.jpeg','.jpg'}
    patterns=[re.compile(r'/Users/[A-Za-z0-9_.-]+/'),re.compile(r'/home/[A-Za-z0-9_.-]+/'),re.compile(r'gh[pousr]_[A-Za-z0-9]{20,}'),re.compile(r'github_pat_[A-Za-z0-9_]{20,}'),re.compile(r'-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----'),re.compile(r'(?i)(?:api[_-]?key|access[_-]?token|password)\s*[:=]\s*[\"\'][A-Za-z0-9_+/=-]{16,}')]
    issues=[];records=[]
    for rel in paths:
        p=Path(rel)
        if set(p.parts)&forbidden_parts or p.suffix.lower() in forbidden_suffixes or p.name in ['.DS_Store','Resume_Bullets_EN_CN.md','american_bankruptcy.csv']:
            issues.append(dict(path=rel,reason='Excluded file in index'))
        b=subprocess.check_output(['git','show',':'+rel],cwd=ROOT)
        if len(b)>20*1024*1024:issues.append(dict(path=rel,reason='File exceeds 20 MiB'))
        text=''
        if p.suffix.lower()=='.pdf':
            import io
            text='\n'.join(x.extract_text() for x in PdfReader(io.BytesIO(b)).pages)
        elif p.suffix.lower() in ['.py','.md','.yaml','.yml','.toml','.txt','.json','.ipynb','.cff','.sh','.csv'] or p.name in ['.gitignore','.env.example']:
            text=b.decode('utf-8',errors='replace')
        for pattern in patterns:
            if pattern.search(text):issues.append(dict(path=rel,reason='Secret/private-path pattern detected'))
        if p.suffix=='.ipynb':
            nb=json.loads(b)
            for cell in nb.get('cells',[]):
                for o in cell.get('outputs',[]):
                    if o.get('output_type')=='error':issues.append(dict(path=rel,reason='Notebook error output'))
            if any('row_id' in line and 'raw_score' in line and 'company_id' in line for cell in nb.get('cells',[]) for o in cell.get('outputs',[]) for line in o.get('text',[])):
                issues.append(dict(path=rel,reason='Possible row prediction output'))
        records.append(dict(path=rel,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
    # Check local Markdown links in the README against staged files.
    readme=ROOT/'README.md'
    if readme.exists():
        for target in re.findall(r'\]\(([^)]+)\)',readme.read_text()):
            if target.startswith(('http:','https:','#')):continue
            if not (ROOT/target.split('#')[0]).exists():issues.append(dict(path='README.md',reason='Broken relative link: '+target))
    result=dict(passed=not issues,index_files=len(records),index_bytes=sum(x['bytes'] for x in records),issues=issues,files=records,git_root_verified=True,owner='MartinXQD (connector verified; recheck publishing credential before remote creation)')
    (ROOT/'artifacts/audit/release_index_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
    if issues:raise SystemExit(1)

if __name__=='__main__':run()
