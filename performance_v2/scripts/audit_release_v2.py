"""Inspect the exact public index without writing any preserved V1 audit files."""
from pathlib import Path
import hashlib
import io
import json
import posixpath
import re
import subprocess
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parent

def git(*args):return subprocess.check_output(['git',*args],cwd=REPO)

def main():
    assert Path(git('rev-parse','--show-toplevel').decode().strip())==REPO
    assert git('branch','--show-current').decode().strip()=='research/performance-v2'
    assert git('remote','get-url','origin').decode().strip()=='https://github.com/MartinXQD/corporate-distress-ml.git'
    assert subprocess.run(['git','merge-base','--is-ancestor','b9e78c6454bfc7cbac6560bdfb047589dc8cd2fe','HEAD'],cwd=REPO).returncode==0
    paths=set(filter(None,git('ls-files','-z').decode().split('\0')));issues=[];records=[]
    forbidden={'artifacts','data','.venv','__pycache__','.git','.pytest_cache','Distress_Performance_V2_Codex_Pack'}
    suffixes={'.zip','.pt','.pkl','.joblib','.ttf','.ttc','.otf','.env','.gz','.sqlite'}
    patterns=[re.compile(r'/Users/[A-Za-z0-9_.-]+/'),re.compile(r'/home/[A-Za-z0-9_.-]+/'),re.compile(r'gh[pousr]_[A-Za-z0-9]{20,}'),re.compile(r'github_pat_[A-Za-z0-9_]{20,}'),re.compile(r'-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----'),re.compile(r'(?i)(?:api[_-]?key|access[_-]?token|password)\s*[:=]\s*[\"\'][A-Za-z0-9_+/=-]{16,}')]
    names={'american_bankruptcy.csv','Resume_Bullets_V2_EN_CN.md','Resume_Bullets_EN_CN.md','Local_Case_Walkthrough_EN_CN.md','EXECUTION_STATE.json','NEXT_ACTIONS.md'}
    for rel in sorted(paths):
        p=Path(rel);b=git('show',':'+rel)
        if set(p.parts)&forbidden or p.suffix.lower() in suffixes or p.name in names:issues.append(dict(path=rel,reason='Excluded local/private file'))
        if len(b)>20*1024*1024:issues.append(dict(path=rel,reason='File exceeds 20 MiB'))
        text=''
        if p.suffix=='.pdf':text='\n'.join(page.extract_text() for page in PdfReader(io.BytesIO(b)).pages)
        elif p.suffix in {'.py','.md','.json','.yaml','.yml','.txt','.csv','.sh','.ipynb','.cff'}:text=b.decode('utf-8')
        for pattern in patterns:
            if pattern.search(text):issues.append(dict(path=rel,reason='Secret or private-path pattern'))
        if rel.startswith('performance_v2/') and p.suffix=='.csv':
            header=text.splitlines()[0].lower() if text else ''
            if 'row_id' in header or 'company_id' in header or 'company_name' in header:issues.append(dict(path=rel,reason='Observation-level identifier in public CSV'))
        if rel.startswith('performance_v2/') and p.suffix=='.md':
            for target in re.findall(r'\]\(([^)]+)\)',text):
                if target.startswith(('http:','https:','#')):continue
                normalized=posixpath.normpath(str(p.parent/target.split('#')[0]))
                if normalized not in paths:issues.append(dict(path=rel,reason='Relative link missing from exact index: '+target))
        records.append(dict(path=rel,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
    changed=git('diff','--name-only','b9e78c6454bfc7cbac6560bdfb047589dc8cd2fe','--cached').decode().splitlines()
    if any(not p.startswith('performance_v2/') for p in changed):issues.append(dict(reason='Release changes outside V2',paths=[p for p in changed if not p.startswith('performance_v2/')]))
    for commit in git('rev-list','b9e78c6454bfc7cbac6560bdfb047589dc8cd2fe..HEAD').decode().splitlines():
        for rel in git('diff-tree','--no-commit-id','--name-only','-r',commit).decode().splitlines():
            p=Path(rel)
            if not rel.startswith('performance_v2/') or set(p.parts)&forbidden or p.suffix in suffixes or p.name in names:issues.append(dict(commit=commit,path=rel,reason='Excluded path in new branch history'))
    result=dict(passed=not issues,branch='research/performance-v2',index_files=len(records),index_bytes=sum(r['bytes'] for r in records),issues=issues,files=records,scope='Exact staged index and new-branch path history; preserved V1 audit untouched')
    path=ROOT/'artifacts/audit/release_index_audit_v2.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
    if issues:raise SystemExit(1)

if __name__=='__main__':main()
