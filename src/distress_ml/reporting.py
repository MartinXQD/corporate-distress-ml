"""Generate reports and claims directly from saved real experiment outputs."""
from __future__ import annotations
import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,roc_curve
from .common import ROOT,PUBLIC,ART,dump,sha,now,state

DISPLAY={'dummy':'Prevalence','altman':'Altman -Z','logistic':'Logistic L2','random_forest':'Random Forest','xgboost':'XGBoost','mlp':'MLP','residual_mlp':'Residual MLP','ft_transformer':'FT-Transformer','catboost':'CatBoost','gru':'GRU','equal_rank_ensemble':'Equal-rank ensemble'}

def fmt(x,digits=3):
    return 'NA' if pd.isna(x) else f'{x:.{digits}f}'

def load_context():
    c={name:pd.read_csv(PUBLIC/f'{name}.csv') for name in ['development_summary','development_yearly','historical_metrics','historical_yearly','historical_seen_unseen','historical_budgets','seed_results','paired_company_bootstrap','seen_unseen_company_bootstrap','group_stress_metrics','reliability_bins','group_permutation_importance','confusion_cell_profiles','feature_selection_stability','split_audit']}
    c['freeze']=json.loads((ROOT/'configs/final_freeze.json').read_text())
    c['ensemble_choice']=json.loads((ROOT/'configs/ensemble_choice.json').read_text())
    c['audit']=json.loads((PUBLIC/'data_audit.json').read_text())
    c['environment']=json.loads((PUBLIC/'environment.json').read_text())
    c['historical']=c['historical_metrics'][c['historical_metrics'].score_type=='calibrated'].copy()
    c['selected']={k:next(m for m in c['freeze']['models'] if k in m['representative'] and m['seed']==42) for k in ['ML','DL']}
    c['representative']={k:c['historical'].set_index('label').loc[v['label']] for k,v in c['selected'].items()}
    c['baseline']=c['historical'][(c['historical'].family=='logistic')&(c['historical'].feature_set=='S0')].iloc[0]
    c['date']=dt.datetime.now(ZoneInfo('America/Los_Angeles')).date().isoformat()
    return c

def ablation_table(c):
    rows=[]; ds=c['development_summary']
    for family in ['xgboost','residual_mlp']:
        a=ds[(ds.family==family)&(ds.score_years==7)&ds.stage.isin(['core_recipe','feature_ablation','controlled_ablation'])]
        base=a[(a.stage=='core_recipe')&(a.feature_set=='S1')].iloc[0]
        for r in a.itertuples():
            if r.stage=='bounded_search': continue
            cfg=json.loads((ART/f'logs/config_{r.config_id}.json').read_text())
            change='S0 raw' if r.feature_set=='S0' else ('S2 dynamics' if r.feature_set=='S2' else 'S1 base')
            if not cfg['clip']: change='No clipping'
            if cfg['weight_alpha']: change='Weight alpha '+str(cfg['weight_alpha'])
            if cfg['sampling']!='none': change=cfg['sampling'].upper()
            if cfg['selector']!='none': change=cfg['selector']
            rows.append(dict(family=family,recipe=change,config_id=r.config_id,macro_year_AP=r.macro_year_AP,delta_vs_S1=r.macro_year_AP-base.macro_year_AP,mean_features=r.mean_features,mean_fit_seconds=r.mean_fit_seconds))
    t=pd.DataFrame(rows); t.to_csv(PUBLIC/'controlled_ablation_summary.csv',index=False)
    return t

def pre_evaluation_tables(c):
    rows=[]
    for item in c['freeze']['models']:
        if item['role']!='primary':continue
        path=ART/f'predictions/{item["label"]}_pre_evaluation.csv.gz'
        if not path.exists():continue
        d=pd.read_csv(path)
        from .evaluation import metrics
        from scipy.special import expit
        for role in d.split_role.unique():
            s=d[d.split_role==role].rename(columns={'company_id':'company_name','y_true':'y'})
            for typ in ['calibrated','uncalibrated']:
                score=s.calibrated_score.to_numpy() if typ=='calibrated' else expit(s.raw_score.to_numpy())
                rows.append(dict(label=item['label'],role=role,score_type=typ,**metrics(s,score,float(s.fixed_threshold.iloc[0]) if typ=='calibrated' else .5,ranking=None if typ=='calibrated' else s.raw_score)))
    result=pd.DataFrame(rows)
    result.to_csv(PUBLIC/'pre_evaluation_metrics.csv',index=False)
    return result

def figures(c):
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.bbox':'tight'})
    out=PUBLIC/'figures';out.mkdir(exist_ok=True)
    selected=[c['baseline']['label'],*[v['label'] for v in c['selected'].values()]]
    labels={c['baseline']['label']:'S0 Logistic',**{v['label']:f'{k}: {DISPLAY[v["family"]]} ({v["feature_set"]})' for k,v in c['selected'].items()}}
    fig,ax=plt.subplots(figsize=(8,2.7)); a=pd.DataFrame(c['audit']['by_year'])
    ax.plot(a.year,a.positive_row_fraction,'o-',color='#177f86');ax.axvspan(2015,2018,color='#efb84b',alpha=.16);ax.set(xlabel='Fiscal year',ylabel='Failed-label row fraction',title='Label prevalence changes across calendar years');fig.savefig(out/'prevalence.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.1))
    curves=[]
    for name in selected:
        d=pd.read_csv(ART/f'predictions/{name}_historical.csv.gz')
        pr,re,th=precision_recall_curve(d.y_true,d.calibrated_score);fpr,tpr,_=roc_curve(d.y_true,d.calibrated_score)
        axes[0].plot(re,pr,label=labels[name]);axes[1].plot(fpr,tpr,label=labels[name])
        curves.extend(dict(label=name,kind='PR',x=float(x),y=float(y)) for x,y in zip(re,pr))
        curves.extend(dict(label=name,kind='ROC',x=float(x),y=float(y)) for x,y in zip(fpr,tpr))
    axes[0].axhline(c['baseline']['prevalence'],ls='--',color='gray');axes[1].plot([0,1],[0,1],'--',color='gray')
    axes[0].set(xlabel='Recall',ylabel='Precision',title='Previously inspected 2015-2018');axes[1].set(xlabel='False positive rate',ylabel='True positive rate',title='ROC ranking')
    axes[0].legend(fontsize=10.5);fig.tight_layout();fig.savefig(out/'ranking_curves.png');plt.close(fig)
    pd.DataFrame(curves).to_csv(PUBLIC/'ranking_curve_points.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(9,3.1))
    extent=.1
    for name in selected:
        b=c['historical_budgets'];b=b[b.label==name].groupby('budget').recall.mean()
        axes[0].plot(b.index*100,b.values,'o-',label=labels[name])
        cal=c['reliability_bins'];cal=cal[(cal.label==name)&(cal.score_type=='calibrated')]
        axes[1].plot(cal.mean_score,cal.observed_rate,'o-',label=labels[name])
        extent=max(extent,float(cal.mean_score.max()),float(cal.observed_rate.max()))
    extent=min(1,extent*1.08)
    axes[0].set(xlabel='Annual review budget (%)',ylabel='Macro-year recall',title='Annual review budgets');axes[1].plot([0,extent],[0,extent],'--',color='gray');axes[1].set(xlim=(0,extent),ylim=(0,extent),xlabel='Mean calibrated score',ylabel='Observed failed-label fraction',title='Reliability: bins need counts')
    axes[0].legend(fontsize=10.5);fig.tight_layout();fig.savefig(out/'screening_calibration.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.0))
    for name in selected:
        a=c['historical_yearly'];a=a[a.label==name]
        axes[0].plot(a.year,a.AP,'o-',label=labels[name]);axes[1].plot(a.year,a.macro_recall_top5,'o-',label=labels[name])
    axes[0].set(xlabel='Year',ylabel='AP',title='Annual ranking with changing prevalence');axes[1].set(xlabel='Year',ylabel='Recall at annual Top 5%',title='Annual screening')
    axes[0].legend(fontsize=10.5);fig.tight_layout();fig.savefig(out/'annual_performance.png');plt.close(fig)
    core=c['development_summary'];core=core[(core.stage=='core_recipe')&(~core.family.isin(['dummy','altman']))]
    pivot=core.pivot(index='family',columns='feature_set',values='macro_year_AP').rename(index=DISPLAY)
    fig,ax=plt.subplots(figsize=(8,3.3));pivot.plot.barh(ax=ax,color=['#6a899b','#177f86']);ax.set(xlabel='Equal-weight AP across seven development years',ylabel='',title='Controlled S0 / S1 comparison');fig.tight_layout();fig.savefig(out/'development_features.png');plt.close(fig)
    imp=c['group_permutation_importance']
    fig,axes=plt.subplots(1,2,figsize=(9,3.0))
    for ax,kind in zip(axes,['ML','DL']):
        a=imp[imp.representative==kind].sort_values('mean_AP_drop')
        ax.barh(a.group,a.mean_AP_drop,color='#177f86');ax.set(xlabel='Historical AP drop after group shuffle',title=kind+' fixed checkpoint')
    fig.tight_layout();fig.savefig(out/'group_importance.png');plt.close(fig)

def table(df,columns,labels=None,limit=None):
    df=df.head(limit) if limit else df
    header=labels or columns
    body=[]
    for _,r in df.iterrows():
        row=[]
        for col in columns:
            v=r[col]
            if isinstance(v,(float,np.floating)): v=fmt(v)
            elif col=='family':v=DISPLAY.get(v,v)
            row.append(str(v))
        body.append(row)
    return {'type':'table','header':header,'rows':body}

def p(text):return {'type':'paragraph','text':text}
def fig(name,caption):return {'type':'figure','path':f'results/public/figures/{name}.png','caption':caption}
def code(text):return {'type':'code','text':text}

def claims(c):
    rows=[]
    for table_name in ['historical_metrics','historical_yearly','historical_seen_unseen','development_summary','paired_company_bootstrap','seen_unseen_company_bootstrap','group_stress_metrics']:
        t=c[table_name]
        for i,r in t.iterrows():
            for metric in ['AP','ROC_AUC','macro_year_AP','macro_recall_top5','Brier','log_loss','precision','recall','F1','TP','FP','FN','TN','alert_fraction','prevalence','rows','companies','positive_rows','positive_companies','features','parameters','epochs','fit_seconds','inference_seconds','score_years','worst_year_AP','mean_fit_seconds','mean_features','lower','upper','delta_mean','delta_lower','delta_upper','valid_draws','invalid_draws']:
                if metric in r and pd.notna(r[metric]):
                    rows.append(dict(claim_id=f'{table_name}:{i}:{metric}',metric=metric,value=float(r[metric]),metric_file=f'results/public/{table_name}.csv',csv_data_row=i+1,row_selection=json.dumps({k:str(r[k]) for k in ['label','config_id','year','seed','score_type','novelty','seen_cutoff','model','metric','group_fold'] if k in r},sort_keys=True),protocol_id=c['freeze']['protocol_id'],run_id=r.get('label',r.get('config_id',r.get('model','development'))),qualification='Development-selected, previously inspected historical evaluation; company-constant labels; no one-year event horizon. CI conditional on fitted models.'))
    for filename in ['controlled_ablation_summary','seed_results','pre_evaluation_metrics','reliability_bins','group_permutation_importance','confusion_cell_profiles','feature_selection_stability','split_audit','accounting_relationships','financial_field_audit','ensemble_development']:
        path=PUBLIC/f'{filename}.csv'
        if not path.exists():continue
        t=pd.read_csv(path)
        for i,r in t.iterrows():
            for metric in t.select_dtypes(include='number').columns:
                if pd.notna(r[metric]):
                    rows.append(dict(claim_id=f'{filename}:{i}:{metric}',metric=metric,value=float(r[metric]),metric_file=f'results/public/{filename}.csv',csv_data_row=i+1,row_selection=json.dumps({k:str(r[k]) for k in ['label','config_id','family','recipe','year','seed','role','score_type','representative','cell','group','selector','feature'] if k in r},sort_keys=True),protocol_id=c['freeze']['protocol_id'],run_id=r.get('label',r.get('config_id','aggregate_or_audit')),qualification='Descriptive audit/aggregate; see table role. Historical period previously inspected; no causal or event-horizon claim.'))
    for metric in ['rows','companies','missing_cells','duplicate_company_year_rows']:
        rows.append(dict(claim_id='data_audit:'+metric,metric=metric,value=c['audit'][metric],metric_file='results/public/data_audit.json',csv_data_row='',row_selection=metric,protocol_id=c['freeze']['protocol_id'],run_id='structural_audit',qualification='Read-only audit, not model performance'))
    for group in ['row_label_counts','company_label_counts','sequence_audit']:
        for metric,value in c['audit'][group].items():
            rows.append(dict(claim_id=f'data_audit:{group}:{metric}',metric=metric,value=value,metric_file='results/public/data_audit.json',csv_data_row='',row_selection=group+'.'+metric,protocol_id=c['freeze']['protocol_id'],run_id='structural_audit',qualification='Read-only audit, not model performance'))
    pd.DataFrame(rows).to_csv(ROOT/'results/CLAIMS.csv',index=False)

def render_document(pages,pdf_path,source_path,language='en'):
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,Image,KeepTogether,Preformatted
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from xml.sax.saxutils import escape
    font='Helvetica';bold='Helvetica-Bold'
    if language=='cn':
        if 'Heiti' not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont('Heiti','/System/Library/Fonts/STHeiti Light.ttc',subfontIndex=0))
        font=bold='Heiti'
    style=getSampleStyleSheet()
    style.add(ParagraphStyle(name='BodyX',fontName=font,fontSize=10.2 if language=='en' else 10.5,leading=15.0 if language=='en' else 17,spaceAfter=10,wordWrap='CJK' if language=='cn' else None,textColor=colors.HexColor('#243648')))
    style.add(ParagraphStyle(name='TitleX',fontName=bold,fontSize=22 if language=='en' else 21,leading=29,spaceAfter=18,textColor=colors.HexColor('#10394a')))
    style.add(ParagraphStyle(name='SubX',fontName=bold,fontSize=12,leading=17,spaceAfter=10,textColor=colors.HexColor('#177f86')))
    style.add(ParagraphStyle(name='CellX',fontName=font,fontSize=8 if language=='en' else 8.4,leading=11 if language=='en' else 13,wordWrap='CJK' if language=='cn' else None))
    style.add(ParagraphStyle(name='CaptionX',fontName=font,fontSize=8.6,leading=12,spaceAfter=12,wordWrap='CJK' if language=='cn' else None,textColor=colors.HexColor('#4c6576')))
    style.add(ParagraphStyle(name='CodeX',fontName='Courier',fontSize=7.8,leading=11,spaceAfter=12))
    story=[]; md=[]
    def para(s,sty='BodyX'):return Paragraph(escape(s).replace('\n','<br/>'),style[sty])
    for n,page in enumerate(pages):
        if n:story.append(PageBreak())
        story.append(para(page['title'],'TitleX'));md.extend(['# '+page['title'],''])
        if page.get('subtitle'):story.append(para(page['subtitle'],'SubX'));md.extend([page['subtitle'],''])
        for b in page['blocks']:
            if b['type']=='paragraph':story.append(para(b['text']));md.extend([b['text'],''])
            elif b['type']=='code':story.append(Preformatted(b['text'],style['CodeX']));md.extend(['```python',b['text'],'```',''])
            elif b['type']=='figure':
                path=ROOT/b['path']
                from PIL import Image as PILImage
                with PILImage.open(path) as im:w,h=im.size
                width=475;height=width*h/w
                if height>235:height=235;width=height*w/h
                story.extend([Image(str(path),width=width,height=height),Spacer(1,5),para(b['caption'],'CaptionX')])
                md.extend([f'![{b["caption"]}](../{b["path"]})',''])
            elif b['type']=='table':
                rows=[[para(str(x),'CellX') for x in b['header']]]+[[para(str(x),'CellX') for x in row] for row in b['rows']]
                ncol=len(b['header']); widths=[475/ncol]*ncol
                if ncol>3:widths=[475*.27]+[475*.73/(ncol-1)]*(ncol-1)
                t=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
                t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dcecef')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f4f7f9')]),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#177f86')),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
                story.extend([t,Spacer(1,12)])
                md.extend(['| '+' | '.join(b['header'])+' |','| '+' | '.join(['---']*ncol)+' |',*['| '+' | '.join(map(str,r))+' |' for r in b['rows']],''])
    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#177f86'));canvas.setLineWidth(.7);canvas.line(48,39,547,39)
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#4c6576'));canvas.drawString(48,25,'Qidian (Martin) Xing  |  Independent Extension');canvas.drawRightString(547,25,str(doc.page));canvas.restoreState()
    doc=SimpleDocTemplate(str(pdf_path),pagesize=(595.28,841.89),leftMargin=55,rightMargin=55,topMargin=47,bottomMargin=55,title='Corporate Financial Distress - Independent Extension' if language=='en' else 'Project Teaching and Interview Guide',author='Qidian (Martin) Xing')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    source_path.write_text('\n'.join(md),encoding='utf-8')
    dump(source_path.with_suffix('.json'),pages)

def report():
    from .explain import explain
    from .report_content import english_pages,chinese_pages
    state('reporting')
    explain();c=load_context();c['ablations']=ablation_table(c);c['pre_evaluation_metrics']=pre_evaluation_tables(c);figures(c);claims(c)
    en=english_pages(c);cn=chinese_pages(c)
    seed_n=len(c['freeze']['final_seeds'])
    if seed_n!=5:
        for page in en+cn:
            for block in page['blocks']:
                if block['type']=='paragraph':
                    block['text']=block['text'].replace('five',str(seed_n)).replace('Five',str(seed_n)).replace('五种子',str(seed_n)+'种子').replace('五个 seed',str(seed_n)+' 个 seed')
    render_document(en,ROOT/'reports/Research_Report_EN.pdf',ROOT/'reports/Research_Report_EN.md','en')
    render_document(cn,ROOT/'reports/Project_Interview_Guide_CN.pdf',ROOT/'reports/Project_Interview_Guide_CN.md','cn')
    write_summaries(c)
    state('reports_generated_pending_visual_release_checks',complete=False)

def write_summaries(c):
    dev=c['development_summary'];historical=c['historical'];ml=c['representative']['ML'];dl=c['representative']['DL'];base=c['baseline']
    ledger=[json.loads(x) for x in (ART/'logs/fit_ledger.jsonl').read_text().splitlines()]
    software_path=ART/'logs/software_fit_accounting.json'
    software_calls=json.loads(software_path.read_text()).get('model_fit_calls',0) if software_path.exists() else 0
    keep=['fit_call','at_utc','tag','rows','original_rows','positive_rows','sampling','weight','max_year','family','C','l1_ratio','score_year','top_groups']
    pd.DataFrame([{k:event.get(k) for k in keep} for event in ledger]).to_csv(PUBLIC/'research_fit_ledger.csv',index=False)
    started=json.loads((ROOT/'RUN_STATUS.json').read_text())['started_at_utc']
    elapsed=(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(started)).total_seconds()
    status=dict(configurations=int(dev.config_id.nunique()),development_fold_runs=int(c['development_yearly'][['run_id']].drop_duplicates().shape[0]),research_fit_calls=len(ledger),software_fit_calls=software_calls,actual_total_fit_calls=len(ledger)+software_calls,actual_fit_calls=len(ledger),final_artifacts=len(c['freeze']['models']),historical_primary_models=len(historical),workflow_seconds_to_report=elapsed,hardware=c['environment'],selected_representatives=c['selected'],bootstrap_repetitions=2000,final_seeds=c['freeze']['final_seeds'],source_files_preserved='pending final manifest verification',publication='pending release checks',failed_pilot_fit_calls=[2],optional_sequences='completed' if (dev.family=='gru').any() else 'not run',paid_services=False)
    dump(PUBLIC/'experiment_summary.json',status)
    (ROOT/'EXPERIMENT_STATUS.md').write_text(f'''# Experiment status\n\nReal local data only. Protocol: {c['freeze']['protocol_id']}.\n\n- Completed {status['configurations']} development configurations and {status['development_fold_runs']} four-fold runs.\n- Recorded {len(ledger)} actual model fit calls, including selectors, internal stopping refits, calibration and pilot attempts. Fit call 2 crashed during the first software pilot; its repair is documented. Synthetic software tests are separate.\n- Froze {status['final_artifacts']} final artifacts, including untuned company stress and seed repeats; evaluated {status['historical_primary_models']} primary pipelines on the previously inspected historical period.\n- CPU device, two threads; no paid compute. Runtime and model costs are in the aggregate tables and durable fit ledger.\n- Five pre-registered seeds for each selected ML/DL representative. Paired company bootstrap: 2,000 draws, conditional on fitted models.\n- GRU status: {status['optional_sequences']}. CatBoost availability and measured rows are shown explicitly. The equal-rank candidate's decision is in configs/ensemble_choice.json.\n- English research and Chinese teaching PDFs generated; final rendered QA and remote release evidence are recorded separately.\n\nNo source score is represented as a new experiment. Historical evaluation remains previously inspected. No one-year default-event horizon, production readiness or causal explanation is claimed.\n''')
    bullets=f'''# Draft resume bullets - owner review required\n\nThese drafts describe AI-assisted work completed under the owner's direction. Confirm comprehension and personal contribution before use. This file is local only.\n\n## English\n\n- Trained reproducible ML and PyTorch benchmarks on {c['audit']['rows']:,} firm-years, applying expanding-year validation, separate calibration and threshold roles, and company-disjoint stress tests for retrospective failed-company classification.\n- Evaluated development-selected {DISPLAY[ml['family']]} and {DISPLAY[dl['family']]} pipelines, recording historical AP of {ml['AP']:.3f} and {dl['AP']:.3f}; assessed annual screening budgets, five-seed variability, and paired company-bootstrap uncertainty.\n\n## 中文\n\n- 在 {c['audit']['rows']:,} 条公司年度记录上训练可复现的 ML 与 PyTorch 基准，执行逐年扩展验证、独立校准与阈值选择及公司不重叠压力测试；任务限定为回溯式失败公司识别。\n- 评估由开发期选定的 {DISPLAY[ml['family']]} 与 {DISPLAY[dl['family']]}，历史评估 AP 分别为 {ml['AP']:.3f} 与 {dl['AP']:.3f}，并报告年度审查预算、五种子波动及公司聚类 bootstrap 区间。\n\n## Evidence\n\nSee results/public/historical_metrics.csv (calibrated, seed 42), seed_results.csv, paired_company_bootstrap.csv and results/CLAIMS.csv. Previously inspected 2015-2018 is not a new untouched external test. No improvement over the original team's different protocol is claimed. Opening verbs avoid the prohibited list; wording remains subject to owner review.\n'''
    seed_n=len(c['freeze']['final_seeds'])
    bullets=bullets.replace('five-seed',f'{seed_n}-seed').replace('五种子',f'{seed_n}种子')
    bullets=bullets.replace(
        f"Evaluated development-selected {DISPLAY[ml['family']]} and {DISPLAY[dl['family']]} pipelines, recording historical AP of {ml['AP']:.3f} and {dl['AP']:.3f}; assessed annual screening budgets, {seed_n}-seed variability, and paired company-bootstrap uncertainty.",
        f"Evaluated development-selected Random Forest screening at historical AP {ml['AP']:.3f} versus {base['AP']:.3f} for same-protocol Logistic, reporting inconclusive paired company-bootstrap intervals and weaker transfer of the selected network and ensemble."
    ).replace(
        f"评估由开发期选定的 {DISPLAY[ml['family']]} 与 {DISPLAY[dl['family']]}，历史评估 AP 分别为 {ml['AP']:.3f} 与 {dl['AP']:.3f}，并报告年度审查预算、{seed_n}种子波动及公司聚类 bootstrap 区间。",
        f"评估开发期选定的 Random Forest：历史 AP 为 {ml['AP']:.3f}，同协议 Logistic 为 {base['AP']:.3f}；配对公司 bootstrap 区间未支持确定提升，并记录选定网络与集成的泛化失败。"
    )
    (ROOT/'reports/Resume_Bullets_EN_CN.md').write_text(bullets)
    status_path=ROOT/'EXPERIMENT_STATUS.md'
    status_text=status_path.read_text().replace('Five pre-registered seeds',f'{seed_n} pre-registered seeds')
    status_text+=f'\nSupplementary synthetic diagnostic/unit model-fit attempts: {software_calls}. Total counted local model-fit attempts: {len(ledger)+software_calls}/600. These software fixtures never enter research scores.\n'
    status_path.write_text(status_text)
    card=f'''# Model card\n\nIndependent extension by Qidian (Martin) Xing, with disclosed Codex assistance.\n\n## Task and selection\n\nRetrospective company-constant failed-label classification. No verified annual event horizon or statement-publication dates. The final ML representative is {DISPLAY[ml['family']]} ({ml['feature_set']}); the final DL representative is {DISPLAY[dl['family']]} ({dl['feature_set']}). Selection uses equal-weight AP across seven development years. The historical period was previously inspected.\n\n## Measured seed-42 historical performance\n\nML: AP {ml['AP']:.6f}, ROC-AUC {ml['ROC_AUC']:.6f}, macro annual recall at Top5% {ml['macro_recall_top5']:.6f}. DL: AP {dl['AP']:.6f}, ROC-AUC {dl['ROC_AUC']:.6f}, macro recall at Top5% {dl['macro_recall_top5']:.6f}. Same-protocol S0 Logistic AP: {base['AP']:.6f}. See the full table rather than treating these representatives as test-selected winners.\n\n## Validation and uncertainty\n\nFinal fit through 2012, Platt 2013, threshold 2014, historical evaluation 2015-2018. All final representatives use {seed_n} registered seeds, not best-seed selection. Paired company bootstrap uses 2,000 draws, conditional on fitted models. Unseen-positive company count through 2012 is only 16. Untuned three-group company stress excludes each held-out company from all fitting and calibration.\n\n## Artifacts and operation\n\nEach local final artifact includes preprocessing, selected and transformed feature order, weights/estimator, calibration, threshold, budget rule, history window, configuration, fit logs and hashes. Numerical reload tolerance is 1e-6 on the measured CPU environment. Dynamic models require prior calendar inputs; rank ensembles require an annual cohort. Do not load untrusted source pickles or mutate the freeze for a repeat historical search.\n\n## Limits and intended use\n\nResearch/teaching/risk-ranking demonstration only. Scores are not validated one-year PDs or causal explanations. Unknown units, label timing, historical test exposure, anonymous company identity, prevalence shift, repeated observations, small unseen-positive cohorts and macro shocks limit transportability. No production, lending, investment or profit claim is supported.\n'''
    (ROOT/'docs/MODEL_CARD.md').write_text(card)
    readme=f'''# Corporate Financial Distress\n\n**A Reproducible ML & Deep Learning Benchmark - Independent Extension**  \n**Qidian (Martin) Xing** | {c['date']} | Codex-assisted implementation and documentation\n\nThis is retrospective failed-company classification on {c['audit']['rows']:,} firm-years and {c['audit']['companies']:,} companies. Company labels are constant over time; true event and financial-publication dates are unavailable. The previously inspected 2015-2018 period is a historical evaluation, not a new untouched external holdout or verified one-year default prediction.\n\n## Measured evidence\n\n{status['configurations']} complete development configurations, {status['development_fold_runs']} calendar-fold runs, {len(ledger)} recorded model fit calls, {len(c['freeze']['models'])} final artifacts, {seed_n} representative seeds, and 2,000 paired company-bootstrap draws. Required Logistic, Random Forest, XGBoost, PyTorch MLP, Residual MLP and FT-Transformer were run, together with prevalence/Altman baselines, available CatBoost and registered GRU/flattened-history controls.\n\n| Development-selected representative | Set | Historical AP | ROC-AUC | Annual Recall@Top5% |\n| --- | --- | --- | --- | --- |\n| {DISPLAY[ml['family']]} | {ml['feature_set']} | {ml['AP']:.3f} | {ml['ROC_AUC']:.3f} | {ml['macro_recall_top5']:.3f} |\n| {DISPLAY[dl['family']]} | {dl['feature_set']} | {dl['AP']:.3f} | {dl['ROC_AUC']:.3f} | {dl['macro_recall_top5']:.3f} |\n| Same-protocol Logistic baseline | S0 | {base['AP']:.3f} | {base['ROC_AUC']:.3f} | {base['macro_recall_top5']:.3f} |\n\nThese are frozen seed-42 results, not test-selected winners. Read [full historical tables](results/public/historical_metrics.csv), [controlled ablations](results/public/controlled_ablation_summary.csv), [paired company intervals](results/public/paired_company_bootstrap.csv), [seed variation](results/public/seed_results.csv) and [company stress](results/public/group_stress_metrics.csv). Old team scores are not an improvement denominator.\n\n## Read the artifacts\n\n- [English research report](reports/Research_Report_EN.pdf) and [editable source](reports/Research_Report_EN.md)\n- [Chinese teaching and interview guide](reports/Project_Interview_Guide_CN.pdf) and [editable source](reports/Project_Interview_Guide_CN.md)\n- [Four executable notebooks](notebooks/01_Data_and_Features.ipynb) and [notebook execution evidence](results/public/notebook_execution.json)\n- [Registered protocol](docs/RESEARCH_PROTOCOL.md), [data card](docs/DATA_CARD.md), [model card](docs/MODEL_CARD.md), [source audit](docs/SOURCE_AUDIT.md)\n- [Claim-to-result index](results/CLAIMS.csv), [experiment status](EXPERIMENT_STATUS.md), [release checklist](RELEASE_CHECKLIST.md)\n- [Contributions](docs/INDIVIDUAL_CONTRIBUTIONS.md), [provenance](docs/PROVENANCE.md), [AI use](docs/AI_USAGE.md), [acknowledgements](ACKNOWLEDGEMENTS.md) and [license status](LICENSE_NOTICE.md)\n\n## Reproduce locally\n\nUse an authorized raw snapshot with SHA-256 `cff2c899a97ecd629415cb22f59186000e74e1c0a78cfae036c0a53025419b5e`. The [live upstream comparison](results/public/upstream_verification.json) finds matching IDs/years/labels but different financial values; do not silently substitute the current public CSV. Data, original team materials, full row predictions and model artifacts are excluded from Git.\n\n```sh\npython3 -m venv .venv\n.venv/bin/python -m pip install -r requirements-lock.txt\n.venv/bin/python -m pip install --no-deps --no-build-isolation -e .\nexport DISTRESS_DATA=/path/to/american_bankruptcy.csv\nscripts/run.sh -m pytest -q\nscripts/run.sh -m distress_ml.cli audit\n```\n\nFull retraining belongs in a fresh run copy that preserves current freeze/history evidence. Run smoke, develop, sequences, freeze, historical-evaluate and report; details and checkpoint/predict commands are in [reproduction instructions](docs/REPRODUCTION.md). Reports need an available Chinese TrueType font on the report-authoring host. On this macOS run, the launcher shares PyTorch's local OpenMP library to avoid a verified multi-runtime crash. No system library is changed. CI runs only synthetic software tests; it does not download data or perform full research training.\n\n## Attribution and limits\n\nThis repository presents an independent extension of an earlier UCLA MFE 413 team project. The original course submission is not represented as a sole-authored work. Original team authors are acknowledged separately. This is owner-directed, AI-assisted work; personal comprehension and independent manual review are not asserted.\n\nNew-code license is pending owner choice. No MIT or other license is inferred for this extension, the original team code or source data. Research outputs do not establish production readiness, an annual PD, causal financial drivers or investment profitability.\n'''
    (ROOT/'README.md').write_text(readme)
    (ROOT/'README_CN.md').write_text('# 中文阅读入口\n\n请先读 reports/Project_Interview_Guide_CN.pdf，再结合英文研究报告与四份 Notebook 核查。中文手册解释任务定义、方法、失败消融、实测结果和面试边界。\n\n这是回溯式 failed-company classification，历史评估期曾被查看，不是已验证的一年期破产预测。原团队与新个人扩展分别致谢，代码和报告有 Codex 辅助。复现命令、来源、预算、全部聚合表及发布审计见英文 README。完整逐行预测、模型工件与简历草稿仅保留本地。\n')
    model_lines=['# Local model artifact index','','Trusted project-generated files only. These artifacts and row predictions are excluded from public Git.','','| Role | Family | Set | Seed | Artifact |','| --- | --- | --- | --- | --- |']
    for item in c['freeze']['models']:
        model_lines.append(f'| {item["role"]} | {item["family"]} | {item["feature_set"]} | {item["seed"]} | {item["path"]} |')
    model_lines+=['','The frozen artifact hash map, metadata and any adjacent PyTorch .pt checkpoint are in configs/final_freeze.json and artifacts/models/. Each calibrated final pipeline retains the 2013 calibrator and 2014 threshold. Development/intermediate states use separate fit-call IDs.']
    (ART/'MODEL_ARTIFACT_INDEX.md').write_text('\n'.join(model_lines)+'\n')
    (ART/'LOCAL_DELIVERY_INDEX.md').write_text('''# Local delivery index\n\n- Public-safe overview: ../README.md\n- English PDF and editable source: ../reports/Research_Report_EN.pdf and .md\n- Chinese PDF and editable source: ../reports/Project_Interview_Guide_CN.pdf and .md\n- Draft resume bullets, owner review required: ../reports/Resume_Bullets_EN_CN.md\n- Full registered numerical case, local only: ../reports/Local_Case_Walkthrough_EN_CN.md\n- Model inventory: MODEL_ARTIFACT_INDEX.md\n- Frozen checkpoint/config/calibration/threshold hashes: ../configs/final_freeze.json\n- Row predictions: predictions/\n- Original preservation and runtime/release evidence: audit/ and logs/\n- Four executed notebooks: ../notebooks/\n- Aggregate results and claim index: ../results/public/ and ../results/CLAIMS.csv\n\nModel/data/case/resume materials remain local and are excluded from Git. The owner should read the teaching guide and reproduce a registered case before adopting interview wording.\n''')
