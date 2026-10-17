"""Publish only the audited extension to a new public repository, never force-push."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import urllib.request
import urllib.error

ROOT=Path(__file__).resolve().parents[1]
OWNER='MartinXQD'
API='https://api.github.com'

def api(method,path,token,payload=None):
    data=json.dumps(payload).encode() if payload is not None else None
    request=urllib.request.Request(API+path,data=data,method=method,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','Content-Type':'application/json','X-GitHub-Api-Version':'2022-11-28'})
    try:
        with urllib.request.urlopen(request,timeout=40) as response:return response.status,json.loads(response.read())
    except urllib.error.HTTPError as e:
        return e.code,json.loads(e.read())

def main():
    top=subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=ROOT,text=True).strip()
    assert Path(top)==ROOT,'Incorrect Git boundary'
    checks=['results/public/tests_summary.json','results/public/verification_summary.json','results/public/pdf_quality_checks.json','artifacts/audit/release_index_audit.json']
    for name in checks:assert json.loads((ROOT/name).read_text())['passed'],name
    notebooks=json.loads((ROOT/'results/public/notebook_execution.json').read_text())
    assert len(notebooks)==4 and all(n['executed'] and n['errors']==0 for n in notebooks)
    audit=json.loads((ROOT/'artifacts/audit/release_index_audit.json').read_text())
    for record in audit['files']:
        b=subprocess.check_output(['git','show',':'+record['path']],cwd=ROOT)
        assert hashlib.sha256(b).hexdigest()==record['sha256'],'Index changed after release audit'
    assert subprocess.run(['git','diff','--quiet'],cwd=ROOT).returncode==0,'Commit tracked changes before publication'
    assert subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode==0,'Commit the audited index before publication'
    env=os.environ.copy();env['GIT_TERMINAL_PROMPT']='0'
    credential=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',text=True,capture_output=True,cwd=ROOT,env=env,timeout=20)
    fields=dict(line.split('=',1) for line in credential.stdout.splitlines() if '=' in line)
    token=fields.get('password')
    if not token:raise RuntimeError('No noninteractive GitHub HTTPS credential; local deliverables remain complete')
    status,user=api('GET','/user',token)
    assert status==200 and user['login']==OWNER,'Authenticated owner does not match the verified account'
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','remote'],cwd=ROOT,text=True).strip():
        raise RuntimeError('A remote already exists; inspect saved publication receipt before resuming, never replace an unknown remote')
    slug='corporate-distress-ml'
    status,_=api('GET',f'/repos/{OWNER}/{slug}',token)
    if status==200:
        suffix=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        slug+='-independent-'+suffix.lower()
        status,_=api('GET',f'/repos/{OWNER}/{slug}',token)
    assert status==404,'Cannot establish that the proposed repository is new'
    status,repo=api('POST','/user/repos',token,{'name':slug,'description':'Independent retrospective corporate distress ML and PyTorch benchmark; reproducible evidence, temporal/company validation, and teaching reports.','private':False,'auto_init':False})
    assert status==201,'Repository creation failed: '+str(status)+' '+str(repo.get('message',''))
    assert repo['owner']['login']==OWNER and not repo['private']
    receipt=dict(repository=repo['html_url'],owner=OWNER,visibility='public',created_repository=True,local_release_commit=head,pushed=False,created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    receipt_path=ROOT/'artifacts/audit/publication_receipt.json'
    receipt_path.write_text(json.dumps(receipt,indent=2))
    subprocess.run(['git','remote','add','origin',repo['clone_url']],cwd=ROOT,check=True)
    subprocess.run(['git','push','-u','origin','main'],cwd=ROOT,check=True,env=env)
    status,remote=api('GET',f'/repos/{OWNER}/{slug}/commits/main',token)
    assert status==200 and remote['sha']==head,'Remote commit verification failed'
    files={}
    for name in ['README.md','reports/Research_Report_EN.pdf','reports/Project_Interview_Guide_CN.pdf','results/public/historical_metrics.csv','notebooks/04_Results_and_Interpretation.ipynb']:
        status,content=api('GET',f'/repos/{OWNER}/{slug}/contents/{name}',token)
        assert status==200 and content['type']=='file',name
        files[name]=content['sha']
    status,repo_check=api('GET',f'/repos/{OWNER}/{slug}',token)
    assert status==200 and repo_check['owner']['login']==OWNER and repo_check['private'] is False
    receipt.update(pushed=True,remote_commit_verified=head,key_files_verified=files,visibility_verified=True,verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    receipt_path.write_text(json.dumps(receipt,indent=2))
    public=dict(repository=receipt['repository'],owner=OWNER,visibility='public',release_content_commit=head,verified_at_utc=receipt['verified_at_utc'],note='This receipt is recorded in a subsequent documentation commit; the complete content snapshot above was verified remotely. Local artifacts/source data/team materials are excluded.')
    (ROOT/'results/public/publication.json').write_text(json.dumps(public,indent=2))
    print(json.dumps(public,indent=2))

if __name__=='__main__':main()
