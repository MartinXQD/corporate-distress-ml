"""Render every PDF page and create review sheets plus machine layout diagnostics."""
from pathlib import Path
import json
import fitz
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/rendered';OUT.mkdir(exist_ok=True)
summaries=[]
for name in ['Research_Report_EN','Project_Interview_Guide_CN']:
    doc=fitz.open(ROOT/f'reports/{name}.pdf')
    folder=OUT/name;folder.mkdir(exist_ok=True)
    pages=[];flags=[]
    for n,page in enumerate(doc):
        pix=page.get_pixmap(matrix=fitz.Matrix(1.4,1.4),alpha=False)
        path=folder/f'page_{n+1:02d}.png';pix.save(path)
        text=page.get_text()
        spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for line in b['lines'] for s in line['spans']]
        headings=[s['text'] for s in spans if s['size']>=17 and s['bbox'][1]<145]
        outside=[s['text'] for s in spans if s['bbox'][0]<35 or s['bbox'][2]>560 or s['bbox'][1]<30 or s['bbox'][3]>825]
        if not headings:flags.append(dict(page=n+1,issue='Continuation without chapter heading; inspect overflow'))
        if len(text.strip())<160:flags.append(dict(page=n+1,issue='Unusually sparse text'))
        if '\ufffd' in text or '\x00' in text:flags.append(dict(page=n+1,issue='Replacement/null glyph in extracted text'))
        if outside:flags.append(dict(page=n+1,issue='Text outside conservative content boundary',text=outside))
        pages.append(dict(page=n+1,characters=len(text),headings=headings,fonts=sorted({s['font'] for s in spans}),png=str(path.relative_to(ROOT))))
    for start in range(0,len(pages),12):
        subset=pages[start:start+12];thumb_w=350;thumb_h=510;cols=3;rows=(len(subset)+cols-1)//cols
        sheet=Image.new('RGB',(cols*thumb_w,rows*thumb_h),'#dce3e8');draw=ImageDraw.Draw(sheet)
        for i,r in enumerate(subset):
            im=Image.open(ROOT/r['png']);im.thumbnail((thumb_w-16,thumb_h-32))
            x=(i%cols)*thumb_w+(thumb_w-im.width)//2;y=(i//cols)*thumb_h+24
            sheet.paste(im,(x,y));draw.text(((i%cols)*thumb_w+10,(i//cols)*thumb_h+6),f'{name}: {r["page"]}',fill='black')
        sheet.save(OUT/f'{name}_contact_{start//12+1}.png')
    summaries.append(dict(report=name,pages=len(doc),machine_layout_flags=flags,page_details=pages,visual_inspection='pending human-model image review; rendering alone is not a pass'))
(OUT/'render_diagnostics.json').write_text(json.dumps(summaries,indent=2,ensure_ascii=False))
print(json.dumps([{k:v for k,v in x.items() if k!='page_details'} for x in summaries],indent=2))
