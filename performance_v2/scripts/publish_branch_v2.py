"""Push only the audited V2 branch to the authorized existing public repository."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import urllib.request
import urllib.error

ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parent
OWNER='MartinXQD';SLUG='corporate-distress-ml';BRANCH='research/performance-v2'

def git(*args):return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()
def api(path,token):
    req=urllib.request.Request('https://api.github.com'+path,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'})
    try:
        with urllib.request.urlopen(req,timeout=30) as response:return response.status,json.loads(response.read())
    except urllib.error.HTTPError as exc:return exc.code,json.loads(exc.read())

def main():
    assert git('branch','--show-current')==BRANCH
    assert git('remote','get-url','origin')==f'https://github.com/{OWNER}/{SLUG}.git'
    for path in ['results/verification.json','results/pdf_quality_checks.json','artifacts/audit/release_index_audit_v2.json']:
        assert json.loads((ROOT/path).read_text())['passed'],path
    assert subprocess.run(['git','diff','--quiet'],cwd=REPO).returncode==0
    assert subprocess.run(['git','diff','--cached','--quiet'],cwd=REPO).returncode==0
    audit=json.loads((ROOT/'artifacts/audit/release_index_audit_v2.json').read_text())
    for record in audit['files']:
        b=subprocess.check_output(['git','show',':'+record['path']],cwd=REPO)
        assert hashlib.sha256(b).hexdigest()==record['sha256'],'Audited index changed'
    env=os.environ.copy();env['GIT_TERMINAL_PROMPT']='0'
    response=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',capture_output=True,text=True,env=env,cwd=REPO,timeout=20)
    fields=dict(line.split('=',1) for line in response.stdout.splitlines() if '=' in line);token=fields.get('password')
    if not token:raise RuntimeError('No noninteractive GitHub HTTPS credential; no branch pushed')
    code,user=api('/user',token);assert code==200 and user['login']==OWNER,'Authenticated publishing identity mismatch'
    code,repository=api(f'/repos/{OWNER}/{SLUG}',token)
    assert code==200 and repository['owner']['login']==OWNER and repository['private'] is False
    head=git('rev-parse','HEAD');code,prior=api(f'/repos/{OWNER}/{SLUG}/git/ref/heads/{BRANCH}',token)
    known=False;receipt_path=ROOT/'artifacts/audit/publication_receipt_v2.json'
    if code==200 and receipt_path.exists():
        saved=json.loads(receipt_path.read_text());previous=saved.get('remote_commit_verified')
        known=previous==prior['object']['sha'] and subprocess.run(['git','merge-base','--is-ancestor',previous,head],cwd=REPO).returncode==0
    assert code==404 or (code==200 and (prior['object']['sha']==head or known)),'The remote branch already has unknown work; refusing overwrite'
    _,main_before=api(f'/repos/{OWNER}/{SLUG}/git/ref/heads/main',token)
    subprocess.run(['git','push','-u','origin',f'HEAD:refs/heads/{BRANCH}'],cwd=REPO,env=env,check=True,timeout=180)
    code,remote=api(f'/repos/{OWNER}/{SLUG}/git/ref/heads/{BRANCH}',token);assert code==200 and remote['object']['sha']==head
    code,tree=api(f'/repos/{OWNER}/{SLUG}/git/trees/{head}?recursive=1',token);assert code==200 and not tree.get('truncated')
    blobs={r['path']:r['sha'] for r in tree['tree'] if r['type']=='blob'}
    for record in audit['files']:
        expected=git('rev-parse','HEAD:'+record['path']);assert blobs.get(record['path'])==expected,record['path']
    _,main_after=api(f'/repos/{OWNER}/{SLUG}/git/ref/heads/main',token)
    receipt=dict(passed=True,repository=repository['html_url'],branch=BRANCH,branch_url=repository['html_url']+'/tree/'+BRANCH+'/performance_v2',authenticated_owner=OWNER,visibility='public',remote_commit_verified=head,all_index_blobs_verified=len(audit['files']),main_before=main_before['object']['sha'],main_after=main_after['object']['sha'],force_push=False,new_repository_created=False,verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    receipt_path.write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
