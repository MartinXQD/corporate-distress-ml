"""V1 visual language with a landscape comparison page and portrait evidence pages."""
from xml.sax.saxutils import escape
from .common import ROOT,dump

def render(pages,pdf_path,source_path,language):
    from reportlab.platypus import BaseDocTemplate,PageTemplate,Frame,Paragraph,Spacer,PageBreak,NextPageTemplate,Table,TableStyle,Image
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from PIL import Image as PILImage
    cn=language=='cn';font='Helvetica';bold='Helvetica-Bold'
    if cn:
        if 'Heiti' not in pdfmetrics.getRegisteredFontNames():pdfmetrics.registerFont(TTFont('Heiti','/System/Library/Fonts/STHeiti Light.ttc',subfontIndex=0))
        font=bold='Heiti'
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='V2Body',fontName=font,fontSize=10.1,leading=16 if cn else 14.8,spaceAfter=9,wordWrap='CJK' if cn else None,textColor=colors.HexColor('#243648')))
    styles.add(ParagraphStyle(name='V2Title',fontName=bold,fontSize=20,leading=27,spaceAfter=13,textColor=colors.HexColor('#10394a')))
    styles.add(ParagraphStyle(name='V2Cell',fontName=font,fontSize=7.6,leading=10.8,wordWrap='CJK' if cn else None,splitLongWords=True))
    styles.add(ParagraphStyle(name='V2Caption',fontName=font,fontSize=8.5,leading=12,spaceAfter=9,wordWrap='CJK' if cn else None,textColor=colors.HexColor('#4c6576')))
    def paragraph(s,style='V2Body'):return Paragraph(escape(str(s)).replace('\n','<br/>'),styles[style])
    def footer(canvas,doc):
        w,h=canvas._pagesize;canvas.saveState();canvas.setStrokeColor(colors.HexColor('#177f86'));canvas.line(44,38,w-44,38)
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#4c6576'));canvas.drawString(44,25,'Qidian (Martin) Xing | Performance V2 | Retrospective evaluation');canvas.drawRightString(w-44,25,str(doc.page));canvas.restoreState()
    document=BaseDocTemplate(str(pdf_path),title='Financial Distress - Performance V2',author='Qidian (Martin) Xing')
    templates=[]
    for name,(w,h) in [('Comparison',(841.89,595.28)),('Evidence',(595.28,841.89))]:
        templates.append(PageTemplate(id=name,frames=[Frame(44,51,w-88,h-91,id=name,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],pagesize=(w,h),onPage=footer,autoNextPageTemplate='Evidence'))
    document.addPageTemplates(templates);story=[];markdown=[]
    for i,p in enumerate(pages):
        if i:story.append(NextPageTemplate('Evidence'));story.append(PageBreak())
        width=753.89 if i==0 else 507.28
        story.append(paragraph(p['title'],'V2Title'));markdown.extend(['# '+p['title'],''])
        for b in p['blocks']:
            if b['type']=='paragraph':story.append(paragraph(b['text']));markdown.extend([b['text'],''])
            elif b['type']=='table':
                n=len(b['header'])
                if n==11:fractions=[.13,.052,.14,.036,.041,.075,.075,.065,.065,.065,.156]
                elif n>3:fractions=[.38]+[.62/(n-1)]*(n-1)
                else:fractions=[.60]+[.40/(n-1)]*(n-1)
                rows=[[paragraph(v,'V2Cell') for v in b['header']]]+[[paragraph(v,'V2Cell') for v in row] for row in b['rows']]
                table=Table(rows,colWidths=[width*x for x in fractions],repeatRows=1,hAlign='LEFT')
                table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dcecef')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f4f7f9')]),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#177f86')),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
                story.extend([table,Spacer(1,10)]);markdown.extend(['| '+' | '.join(b['header'])+' |','| '+' | '.join(['---']*n)+' |',*['| '+' | '.join(map(str,row))+' |' for row in b['rows']],''])
            elif b['type']=='figure':
                path=ROOT/b['path']
                with PILImage.open(path) as image:w,h=image.size
                iw=min(width,490);ih=iw*h/w
                if ih>210:ih=210;iw=ih*w/h
                story.extend([Image(str(path),width=iw,height=ih),Spacer(1,4),paragraph(b['caption'],'V2Caption')]);markdown.extend([f'![{b["caption"]}](../{b["path"]})',''])
    document.build(story);source_path.write_text('\n'.join(markdown),encoding='utf-8');dump(source_path.with_suffix('.json'),pages)
