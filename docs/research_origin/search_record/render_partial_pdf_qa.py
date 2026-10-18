"""Render the requested report pair and bind visual QA to exact PDF bytes."""
from pathlib import Path
import argparse
import hashlib
import json
import pymupdf
from PIL import Image,ImageOps,ImageDraw

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/pdf_qa';OUT.mkdir(parents=True,exist_ok=True)
parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args()
verification_path=ROOT/'results/verification.json'
if args.final:
    assert json.loads(verification_path.read_text())['passed']
    assert json.loads((ROOT/'results/experiment_summary.json').read_text())['scientific_evaluation_complete']
records=[];thumbs=[]
for name in ['Research_Report_V2_EN','PERFORMANCE_DELTA_CN']:
    doc=pymupdf.open(ROOT/f'reports/{name}.pdf');pages=[]
    for i,page in enumerate(doc):
        path=OUT/f'{name}_{i+1:02d}.png';page.get_pixmap(matrix=pymupdf.Matrix(1.8,1.8),alpha=False).save(path)
        text=page.get_text();outside=[]
        for b in page.get_text('blocks'):
            if b[0]<0 or b[1]<0 or b[2]>page.rect.width+.5 or b[3]>page.rect.height+.5:outside.append(list(b[:4]))
        pages.append(dict(page=i+1,width=page.rect.width,height=page.rect.height,text_characters=len(text),outside_blocks=outside))
        image=Image.open(path);image.thumbnail((305,415));tile=Image.new('RGB',(325,455),'#e6e9ee');tile.paste(image,((325-image.width)//2,25));ImageDraw.Draw(tile).text((8,5),name+' '+str(i+1),fill='black');thumbs.append(tile)
    records.append(dict(file=f'reports/{name}.pdf',sha256=hashlib.sha256((ROOT/f'reports/{name}.pdf').read_bytes()).hexdigest(),pages=pages,page_count=len(doc)))
sheet=Image.new('RGB',(325*3,455*((len(thumbs)+2)//3)),'white')
for i,tile in enumerate(thumbs):sheet.paste(tile,((i%3)*325,(i//3)*455))
sheet.save(OUT/'contact_sheet.png')
result=dict(passed=all(not p['outside_blocks'] and p['text_characters']>80 for r in records for p in r['pages']),scope='Final verified report rendering and bounds; manual inspection additionally required' if args.final else 'Partial report rendering and bounds only; no final experiment verification',reports_are_final=args.final,source_verification_sha256=hashlib.sha256(verification_path.read_bytes()).hexdigest() if args.final else None,manual_visual_review='pending',files=records)
(ROOT/'results/pdf_quality_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
