"""Audit original files without executing or modifying source material."""
from pathlib import Path, PurePosixPath
import hashlib, json, zipfile, datetime, importlib.util
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'individual_extension'
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
manifest={str(p.relative_to(ROOT)):digest(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and 'individual_extension' not in p.relative_to(ROOT).parts}
(OUT/'artifacts/audit/original_manifest.json').write_text(json.dumps(manifest,indent=2))
archives=[]
seen=set()
for p in ROOT.rglob('*.zip'):
    if 'individual_extension' in p.parts: continue
    h=digest(p)
    if h in seen: continue
    seen.add(h)
    with zipfile.ZipFile(p) as z:
        entries=[]
        for i in z.infolist():
            q=PurePosixPath(i.filename)
            safe=not q.is_absolute() and '..' not in q.parts and not ((i.external_attr>>16)&0o170000)==0o120000
            entries.append(dict(name=i.filename,bytes=i.file_size,compressed_bytes=i.compress_size,safe=safe))
        archives.append(dict(path=str(p.relative_to(ROOT)),sha256=h,entries=entries))
        for name in z.namelist():
            if name.endswith(('martin note.txt','martin comment.txt')) and len(z.read(name))<100000:
                target=OUT/'artifacts/audit'/Path(name).name
                target.write_bytes(z.read(name))
(OUT/'artifacts/audit/archive_inventory.json').write_text(json.dumps(archives,indent=2))
spec=importlib.util.spec_from_file_location('source_audit',ROOT/'Distress_ML_Upgrade_Codex_Pack/audit_local_data.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
a=m.audit(ROOT)
(OUT/'results/public/data_audit.json').write_text(json.dumps(a,indent=2))
print(json.dumps({k:a[k] for k in ['rows','companies','raw_files_sha256','company_label_counts','sequence_audit']},indent=2))
print('Original files:',len(manifest),'Archives:',len(archives))
