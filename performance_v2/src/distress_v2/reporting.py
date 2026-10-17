"""Concise evidence-first reports; no gain wording before numerical verification."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .common import ROOT,V1,ART,RESULT,dump,now,hours,digest,state

def context():
    assert json.loads((RESULT/'verification.json').read_text())['passed']
    names=['historical_2015_2018','historical_by_year','trials','development_by_year','selection_trace','confirmation_2014','seeds','paired_company_intervals','ensemble_development','seen_unseen']
    c={k:pd.read_csv(RESULT/f'{k}.csv') for k in names}
    c['selected']=json.loads((ROOT/'configs/selection_freeze.json').read_text())
    c['main']=c['historical_2015_2018'].set_index('name')
    c['bridge']=c['development_by_year'][c['development_by_year'].stage=='bridge']
    from .data import load
    df=load('historical');c['cohort']=dict(rows=len(df),companies=df.company_name.nunique(),fit_rows=int((df.year<=2014).sum()),historical_rows=int((df.year>=2015).sum()))
    return c

def delta_table(c):
    rows=[];archive=pd.read_csv(RESULT/'legacy_archived_metrics.csv')
    for r in archive.itertuples():
        rows.append(dict(recipe='V0 L1' if 'lasso' in r.feature_set else 'V0 base',version='V0',fit_cutoff=2011,input='fixed13/std01' if 'lasso' in r.feature_set else 'RAW18/std01',selected='No' if 'lasso' in r.feature_set else 'Yes',seeds=1,annual_AP=np.nan,pooled_AP=r.test_pr_auc,ROC=r.test_roc_auc,R5=np.nan,seconds=np.nan,identity='Archive only'))
    h=pd.read_csv(V1/'results/public/historical_metrics.csv');h=h[(h.score_type=='calibrated')&(h.seed==42)]
    for family,feature,label in [('random_forest','S1','V1 RF'),('residual_mlp','S1','V1 DL'),('xgboost','LAGS','V1 lag XGB'),('catboost','S0','V1 Cat raw')]:
        if label in ['V1 RF','V1 DL']:
            selected_id=json.loads((V1/'configs/final_freeze.json').read_text())['representatives']['ML' if label=='V1 RF' else 'DL'];d=h[h.config_id==selected_id]
        else:d=h[(h.family==family)&(h.feature_set==feature)]
        r=d.iloc[0];rows.append(dict(recipe=label,version='V1',fit_cutoff=2012,input=feature,selected='Yes' if label in ['V1 RF','V1 DL'] else 'No',seeds=1,annual_AP=r.macro_year_AP,pooled_AP=r.AP,ROC=r.ROC_AUC,R5=r.macro_recall_top5,seconds=r.fit_seconds,identity='Saved V1'))
    selected=c['selected'];keys=list(dict.fromkeys(['baseline_legacy_strong_inner_l1','baseline_v1_lags_xgb','baseline_v1_raw_catboost','baseline_v1_selected_rf','baseline_v1_selected_dl_corrected_steps','baseline_logistic',selected['overall'],selected['ML'],selected['DL']]))
    for key in keys:
        row=c['main'].loc['P2_seed_mean__'+key];config=json.loads((ROOT/f'configs/trial_{key}.json').read_text())
        short={'baseline_legacy_strong_inner_l1':'P2 old L1','baseline_v1_lags_xgb':'P2 lag XGB','baseline_v1_raw_catboost':'P2 Cat raw','baseline_v1_selected_rf':'P2 V1 RF','baseline_v1_selected_dl_corrected_steps':'P2 V1 DL','baseline_logistic':'P2 Logistic'}.get(key,'P2 overall' if key==selected['overall'] else ('P2 ML' if key==selected['ML'] else 'P2 DL'))
        component=c['historical_2015_2018'];cost=component[(component.candidate==key)&(component.channel=='P2')].fit_seconds.sum()
        rows.append(dict(recipe=short,version='P2',fit_cutoff=2014,input=config['features']+'/'+config['preprocess'],selected='Yes' if key in [selected['overall'],selected['ML'],selected['DL']] else 'Fixed',seeds=int(row.seed_count),annual_AP=row.macro_AP,pooled_AP=row.AP,ROC=row.ROC_AUC,R5=row.recall5,seconds=cost,identity='New seed mean'))
    if selected['blend']['admitted']:
        r=c['main'].loc['P2_fixed_rank_blend'];rows.append(dict(recipe='P2 blend',version='P2',fit_cutoff=2014,input='annual ranks',selected='Yes',seeds=5,annual_AP=r.macro_AP,pooled_AP=r.AP,ROC=r.ROC_AUC,R5=r.recall5,seconds=np.nan,identity='Frozen blend'))
        for key,label in [('baseline_legacy_strong_inner_l1','Rank old L1'),('baseline_v1_lags_xgb','Rank lag XGB'),('baseline_v1_raw_catboost','Rank Cat raw'),('baseline_v1_selected_rf','Rank V1 RF')]:
            r=c['main'].loc['P2_annual_rank__'+key]
            rows.append(dict(recipe=label,version='P2',fit_cutoff=2014,input='annual ranks',selected='Fixed',seeds=5,annual_AP=r.macro_AP,pooled_AP=r.AP,ROC=r.ROC_AUC,R5=r.recall5,seconds=np.nan,identity='Rank reference'))
    for row in c['historical_2015_2018'].itertuples():
        if row.channel not in ['P1','P2_selected__fit_cutoff_2012_transfer','V0_archive_role']:continue
        rows.append(dict(recipe=row.name,version=row.channel,fit_cutoff=row.fit_cutoff,input='See frozen artifact configuration',selected='Pre' if row.channel!='P2_selected__fit_cutoff_2012_transfer' else 'P2',seeds=1,annual_AP=row.macro_AP,pooled_AP=row.AP,ROC=row.ROC_AUC,R5=row.recall5,seconds=row.fit_seconds,identity=row.score_type))
    table=pd.DataFrame(rows);table.to_csv(RESULT/'baseline_bridge.csv',index=False);return table

def export_ablations(c):
    b=c['bridge'].copy();b['comparison_scope']='Single 2012 development year, fixed seed; mechanism diagnostic, not independent historical evidence'
    b.to_csv(RESULT/'feature_preprocess_ablation.csv',index=False)
    rows=[]
    for p in (ART/'logs').glob('*_complete.json'):
        if p.name in ['historical_complete.json']:continue
        d=json.loads(p.read_text())
        if not d.get('tag','').startswith('bridge_'):continue
        s=d['final_stats'];rows.append(dict(candidate=d['tag'],actual_updates=s.get('actual_updates'),best_steps=d['inner_stats'].get('best_steps'),positive_exposures=s.get('positive_exposures'),negative_exposures=s.get('negative_exposures'),train_AP=s.get('train_AP'),train_BCE=s.get('train_BCE'),fit_seconds=d.get('fit_seconds'),parameters=s.get('parameters')))
    pd.DataFrame(rows).to_csv(RESULT/'training_budget_ablation.csv',index=False)
    costs=[]
    for item in json.loads((ROOT/'configs/final_inventory.json').read_text()):
        d=json.loads((ROOT/item['path']).with_suffix('.json').read_text());costs.append(dict(name=item['name'],family=d['config']['family'],channel=item['channel'],fit_seconds=d.get('fit_seconds'),parameters=d['final_stats'].get('parameters'),updates=d['final_stats'].get('actual_updates'),rounds=d['final_stats'].get('best_rounds'),input_dimensions=len(d.get('transformed_columns',d['columns'])),fit_rows=d['fit_rows']))
    pd.DataFrame(costs).to_csv(RESULT/'costs.csv',index=False)
    return pd.DataFrame(rows)

def configuration_bridge():
    """Keep unknown archive training lengths unknown; derive saved V1 updates."""
    rows=[]
    archive=pd.read_csv(RESULT/'legacy_archived_metrics.csv')
    for r in archive.itertuples():
        l1='lasso' in r.feature_set;columns=['log_market_value','long_term_debt_to_assets','log_total_assets','X8','X6','X3','X15','working_capital','current_ratio','operating_expense_to_revenue','ebit_to_assets','gross_margin','X17'] if l1 else [f'X{i}' for i in range(1,19)]
        rows.append(dict(version='V0_archive',family='residual_mlp',configuration_identity=r.feature_set,feature_set='fixed_archive13' if l1 else 'RAW18',input_columns=json.dumps(columns),input_dim=len(columns),parameters=301249 if l1 else 302209,preprocessing='Train-only 1/99% clipping, median and standard scaling',fit_cutoff=2011,stop_years='2012-2014',weight_sampling='natural data, full train-recomputed pos_weight',width=192,depth=4,optimizer='AdamW',early_stop_metric='sklearn average precision',best_epoch=r.best_epoch,actual_optimizer_updates=np.nan,update_evidence='Actual executed length not recorded in bridge metrics; best epoch is not actual executed length',best_state_updates=int(r.best_epoch)*110,refit='Restore best checkpoint, no full-validation-window refit',device='CPU',source='Archived comparison row; not newly reproduced'))
    freeze=json.loads((V1/'configs/final_freeze.json').read_text());events=[json.loads(s) for s in (V1/'artifacts/logs/fit_ledger.jsonl').read_text().splitlines()]
    for role in ['ML','DL']:
        item=next(m for m in freeze['models'] if role in m['representative'] and m['seed']==42)
        m=json.loads((V1/item['path']).with_suffix('.json').read_text());cfg=m['config'];event=next(e for e in events if e['tag']==item['label']+'/full_refit');neural=role=='DL'
        updates=int(m['epochs'])*int(np.ceil(event['rows']/cfg['batch_size'])) if neural else np.nan
        rows.append(dict(version='V1_selected_'+role,family=cfg['family'],configuration_identity=item['config_id'],feature_set=cfg['feature_set'],input_columns=json.dumps(m.get('transformed_features',m['selected_features'])),input_dim=m['features'],parameters=m.get('parameters'),preprocessing='Train-only 0.5/99.5% clipping; V1 median/scale/masks',fit_cutoff=2012,stop_years=2012,weight_sampling=cfg['sampling']+f'; class alpha={cfg["weight_alpha"]}',width=cfg.get('width'),depth=cfg.get('depth',cfg.get('max_depth')),optimizer='AdamW' if neural else 'RandomForest fit',early_stop_metric='sklearn average precision' if neural else 'No early stop',best_epoch=m['epochs'] if neural else np.nan,actual_optimizer_updates=updates,update_evidence='Derived exactly from saved executed epochs and fit-ledger resampled rows' if neural else 'Tree rounds, not optimizer updates',best_state_updates=updates,refit='Inner-tail length then new full-window epoch-equivalent fit',device='CPU',source='Saved artifact reloaded; predictions checked'))
    pd.DataFrame(rows).to_csv(RESULT/'legacy_vs_v1_config.csv',index=False)

def claims():
    rows=[]
    for path in RESULT.glob('*.csv'):
        if path.name=='claims.csv':continue
        table=pd.read_csv(path)
        for i,r in table.iterrows():
            for column in table.select_dtypes(include='number'):
                if pd.notna(r[column]):rows.append(dict(claim_id=f'{path.stem}:{i}:{column}',value=float(r[column]),metric=column,file=str(path.relative_to(ROOT)),data_row=i+1,row_filter=json.dumps({k:str(r[k]) for k in ['name','candidate','trial_id','model','reference','metric','seed','year','channel','recipe','version'] if k in r},sort_keys=True),qualification='See channel/source identity: archives, bridge, development, confirmation and exposed historical evaluation are distinct. Not an annual PD or causal result.'))
    pd.DataFrame(rows).to_csv(RESULT/'claims.csv',index=False)

def matched_numeric_controls(c):
    rows=[]
    for family in ['num_mlp','tabm']:
        pair=[]
        for number in [0,1]:
            t=c['trials'][(c['trials'].family==family)&(c['trials'].stage=='screen')&c['trials'].trial_id.str.contains(f'_{number:03d}_')]
            if len(t)!=1 or t.iloc[0].status!='COMPLETE':break
            pair.append(t.iloc[0])
        if len(pair)!=2:
            rows.append(dict(family=family,status='INCOMPLETE_MATCHED_PAIR'));continue
        a,b=[json.loads(r.config_json) for r in pair]
        ignore={'embedding','bins','embedding_dim'}
        assert {k:v for k,v in a.items() if k not in ignore}=={k:v for k,v in b.items() if k not in ignore}
        rows.append(dict(family=family,status='COMPLETE',without_id=pair[0].trial_id,with_id=pair[1].trial_id,input=a['features'],preprocess=a['preprocess'],no_embedding_AP=pair[0].macro_AP,piecewise_AP=pair[1].macro_AP,delta_AP=pair[1].macro_AP-pair[0].macro_AP,no_embedding_seconds=pair[0].seconds,piecewise_seconds=pair[1].seconds,scope='Same configured backbone/optimization and three development years; a single screening seed, not historical causal attribution'))
    table=pd.DataFrame(rows);table.to_csv(RESULT/'numeric_matched_controls.csv',index=False);return table

def verdict(c):
    selected=c['selected'];blend=selected['blend']['admitted'];target='fixed_annual_rank_blend' if blend else selected['overall'];main=c['main'].loc['P2_fixed_rank_blend' if blend else 'P2_seed_mean__'+target]
    names=['baseline_legacy_strong_inner_l1','baseline_v1_lags_xgb','baseline_v1_raw_catboost','baseline_v1_selected_rf']
    refs={key:c['main'].loc['P2_seed_mean__'+key] for key in names}
    best=max(refs,key=lambda key:refs[key].macro_AP);gap=main.macro_AP-refs[best].macro_AP
    old=c['main'].loc['P2_seed_mean__baseline_legacy_strong_inner_l1']
    ci=c['paired_company_intervals'];q=ci[(ci.model==target)&(ci.metric=='macro_year_AP')]
    positive=len(q)==4 and bool((q.delta_lower>0).all())
    archive=pd.read_csv(RESULT/'legacy_archived_metrics.csv');archival_ap=float(archive.loc[archive.feature_set.str.contains('lasso'),'test_pr_auc'].iloc[0]);single=c['main'].loc['P2_seed_mean__'+selected['overall']]
    return dict(main=main,target=target,strongest_name=best,strongest=refs[best],gap=gap,old=old,all_point_benchmarks_beaten=gap>0,all_conditional_intervals_positive=positive,
      en=f"The development-selected {'fixed annual-rank blend' if blend else 'overall single recipe'} has historical annual AP {main.macro_AP:.3f}, versus {refs[best].macro_AP:.3f} for the strongest frozen same-cutoff baseline point estimate ({best}). The difference is {gap:+.3f}. "+('It exceeds all four baseline point estimates. ' if gap>0 else 'It does not exceed every strong baseline; this is not a comprehensive performance upgrade. ')+('All four paired annual-AP lower bounds are positive under the conditional company bootstrap.' if positive else 'The paired intervals do not uniformly establish a positive improvement over all four strong baselines.'),
      archival_en=f"Descriptively, the overall single recipe's raw pooled AP is {single.AP:.3f}, versus archived legacy L1 {archival_ap:.3f} ({single.AP-archival_ap:+.3f}). Different cutoff/selection/seed roles prevent attributing this archive difference to the algorithm; it is separate from the same-cutoff annual-AP comparison.",
      archival_cn=f"描述性对照：整体单配方的原始 pooled AP={single.AP:.3f}，旧L1档案={archival_ap:.3f}（差{single.AP-archival_ap:+.3f}）。两者截止年、选择和seed角色不同，不能把这个档案差值归因于算法；它与同截止年的年度AP比较分开。",
      cn=f"开发期选定的{'固定年度rank融合' if blend else '整体单配方'}，历史年度 AP={main.macro_AP:.3f}；冻结的同截止年强基准中，点估计最高的是 {best}（{refs[best].macro_AP:.3f}），差值 {gap:+.3f}。"+('点估计超过四条强基准。' if gap>0 else '没有超过所有强基准，不能称全面性能升级。')+('四组配对年度 AP 区间下界均为正，但仍只是固定模型与当前样本下的条件证据。' if positive else '配对区间没有一致支持对四条强基准都有确定的正向差异。'))

def figures(c):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=RESULT/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
    fig,axes=plt.subplots(1,2,figsize=(9,3.0))
    for name,label in [('bridge_legacy192_unweighted','No sampling'),('bridge_legacy192_ros_steps','ROS: original-step epochs'),('bridge_legacy192_ros_epochs','ROS: balanced-data epochs')]:
        p=ART/f'logs/{name}__seed42__year2012__stop_curve.csv'
        if not p.exists():continue
        t=pd.read_csv(p);axes[0].plot(t.updates,t.stop_AP,label=label);axes[1].plot(t.updates,t.positive_exposures,label=label)
    axes[0].set(xlabel='Actual optimizer updates',ylabel='Internal 2011 AP',title='Same epoch does not mean same updates');axes[1].set(xlabel='Actual optimizer updates',ylabel='Positive-example exposures',title='Exposure is not new independent data');axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'update_budget.png',dpi=160);plt.close(fig)
    h=c['historical_by_year'];selected=c['selected'];keys=list(dict.fromkeys([selected['overall'],selected['DL'],'baseline_legacy_strong_inner_l1','baseline_v1_lags_xgb']))
    fig,axes=plt.subplots(1,2,figsize=(9,3.0))
    for key in keys:
        sub=h[(h.candidate==key)&(h.channel=='P2')];g=sub.groupby('year')[['AP','recall5']].mean()
        label={'baseline_legacy_strong_inner_l1':'Legacy recipe','baseline_v1_lags_xgb':'V1 lag XGB'}.get(key,'Overall' if key==selected['overall'] else 'DL')
        axes[0].plot(g.index,g.AP,'o-',label=label);axes[1].plot(g.index,g.recall5,'o-',label=label)
    axes[0].set(xlabel='Previously inspected historical year',ylabel='Mean individual-seed AP',title='Annual seed-average metrics');axes[1].set(xlabel='Year',ylabel='Mean individual-seed Recall@5%',title='Natural annual review population');axes[0].legend(fontsize=9);fig.tight_layout();fig.savefig(out/'historical_years.png',dpi=160);plt.close(fig)
    t=c['trials'];t=t[(t.status=='COMPLETE')&t.stage.isin(['screen','refinement'])]
    fig,ax=plt.subplots(figsize=(8,3.0))
    for family,g in t.groupby('family'):ax.scatter(g.completed_screen_fold_fit_seconds,g.macro_AP,label=family,s=16,alpha=.65)
    ax.set(xscale='log',xlabel='Sum of completed screen-fold fit seconds',ylabel='Screening mean annual AP',title='Optimization cost and development performance');ax.legend(fontsize=7,ncol=3);fig.tight_layout();fig.savefig(out/'search_cost.png',dpi=160);plt.close(fig)

def block(text):return {'type':'paragraph','text':text}
def image(name,caption):return {'type':'figure','path':f'results/figures/{name}.png','caption':caption}
def tab(df,columns,labels=None):
    rows=[]
    for _,r in df.iterrows():
        row=[]
        for col in columns:
            value=r[col]
            count_column=col in ['fit_cutoff','seeds','actual_updates','positive_exposures','valid_draws','year','rows','COMPLETE','FAIL','PRUNED']
            row.append('NA' if pd.isna(value) else (str(int(value)) if count_column else (f'{value:.3f}' if isinstance(value,(float,np.floating)) else str(value))))
        rows.append(row)
    return dict(type='table',header=labels or columns,rows=rows)
def page(title,*blocks):return dict(title=title,subtitle=None,blocks=list(blocks))

def pages(c,delta,budget,language):
    cn=language=='cn';v=verdict(c);selected=c['selected'];cohort=c['cohort']
    overview=delta[delta.version.isin(['V0','V1','P2'])&~delta.recipe.isin(['V1 lag XGB','V1 Cat raw','P2 Cat raw','P2 Logistic'])].copy()
    overview['input']=overview.input.replace({'INNER_L1/std01':'L1/01','V1_LAGS/v1':'Lag3/V1','V1_S1/v1':'S1/V1','fixed13/std01':'13/01','RAW18/std01':'18/01'})
    overview['input']=overview.input.str.replace('RATIO_CORE','Ratio',regex=False).str.replace('SLOG_RATIO','LogR',regex=False).str.replace('LAGS_RAW3','Lag3',regex=False).str.replace('LAGS_SLOG3','Log3',regex=False).str.replace('DYNAMIC_COMPACT','Dyn',regex=False).str.replace('quantile','QT',regex=False).str.replace('std005','005',regex=False).str.replace('std01','01',regex=False)
    overview['selected']=overview.selected.replace({'Yes':'Y','No':'N','Fixed':'Pre'});overview['minutes']=overview.seconds/60;overview['identity']=overview.identity.replace({'Archive only':'Archive','Saved V1':'V1','New seed mean':'Mean score','Frozen blend':'Rank blend'})
    columns=['recipe','fit_cutoff','input','selected','seeds','annual_AP','pooled_AP','ROC','R5','minutes','identity']
    first=tab(overview,columns,['Recipe','Cutoff','Input','Sel','Seeds','Yr AP','Pool AP','ROC','R@5','Min','Source'])
    from .search import TARGETS
    complete=c['trials'];screen=complete[complete.stage=='screen'];statuses=screen.status.map(lambda s:'FAIL' if s.startswith('FAIL') else s);coverage=pd.crosstab(screen.family,statuses).reindex(index=list(TARGETS),columns=['COMPLETE','FAIL','PRUNED'],fill_value=0).rename_axis('family').reset_index()
    bridge=c['bridge'][~c['bridge'].candidate.str.contains('equal2000')][['candidate','AP','ROC_AUC','fit_seconds']].copy();bridge.candidate=bridge.candidate.str.replace('bridge_','',regex=False)
    ci=c['paired_company_intervals'];ci=ci[(ci.model==v['target'])&(ci.metric=='macro_year_AP')][['reference','delta_lower','delta_upper','valid_draws']]
    seeds=c['seeds'];seed_summary=seeds.groupby('candidate').agg(seeds=('seed','nunique'),annual_AP_mean=('macro_AP','mean'),annual_AP_min=('macro_AP','min'),annual_AP_max=('macro_AP','max')).reset_index()
    seed_summary['candidate']=seed_summary.candidate.str.slice(0,38)
    if cn:
        return [
          page('性能优化 V2：先看真实增量',block(f'Qidian (Martin) Xing | {cohort["rows"]:,}条公司年度记录、{cohort["companies"]:,}家公司；截至2014可用于拟合{cohort["fit_rows"]:,}条。2015-2018共{cohort["historical_rows"]:,}条，已被多轮查看。'),first,block(v['cn']),block('Yr AP 为历史四年AP等权均值；Pool AP与ROC为合并统计；R@5为逐年Top5%召回均值。Min为拟合计时分钟；seed mean先平均预测再评分。Sel表示各版本/家族内选定；Archive为旧档案，不是新实验。年度ROC guardrail另见selection_trace，完整身份见baseline_bridge.csv。')),
          page('2. 旧版为何曾经更好：证据与边界',tab(bridge,['candidate','AP','fit_seconds'],['单年控制配方','2012 AP','秒']),block('可确认：旧 Residual 为 192×4、约30万参数；V1 选定网络为64×2、36,801参数、ROS、18个最终epoch。输入、截尾、损失权重、停止与重拟合角色也不同。旧AP函数与sklearn在真实保存分数及ties样例上相同，不能把差距归咎于指标定义。'),block('这些控制实验固定2012评分年与seed42，只支持该环境下的机制诊断。大容量并非自动更优，单个高分也不能证明普遍欠拟合或过拟合。固定旧13列含原2012-2014选择信息，只用于档案角色重训；正式开发每折重新执行L1。原包中epoch31的checkpoint与桥接表epoch23不同，身份分别保留。')),
          page('3. 采样、数值处理与真实训练预算',image('update_budget','横轴是实际optimizer更新次数。右图是正类曝光次数，复制或插值不会产生新的独立公司。'),tab(budget[budget.candidate.str.contains('equal2000')][['candidate','actual_updates','positive_exposures','train_BCE']],['candidate','actual_updates','positive_exposures','train_BCE'],['固定更新控制','更新次数','正类曝光','训练BCE']),block('另做无采样、ROS、SMOTE各2,000次更新控制。NN比较旧1/99%截尾、0.5/99.5%、稳健尺度与训练内normal quantile；二元mask不做quantile。仅在训练角色移除常量和重复mask，保留全部记录。数值embedding使用官方模块，TabM逐head求BCE后平均，推断平均sigmoid概率。'),block('V2迁移的V1 ROS epoch重拟合曾遗漏采样后行数，已将那条trial标记无效、保留证据并重跑；这是本轮迁移修复，不能拿来解释原V1下降。主搜索固定step分支不受影响。')),
          page('4. 搜索不是模型名单：覆盖与选择',tab(coverage,['family','COMPLETE','FAIL','PRUNED'],['家族','完成配置','失败','剪枝']),image('search_cost','每个点是一个真实三年初筛配置，未把fold或seed计作不同方法。'),block('初筛2009/2011/2013；入围补齐2009-2013五年，再按42/2026/31415三seed×年度平均选择。每家最多2个、总共最多8个；AP差0.005以内按近期年度AP、Top5召回和成本取舍。强旧配方、LAGS、CatBoost与原RF基准均可入围，不强迫新模型或S1胜出。')),
          page('5. 是否超过强旧配方，而非弱Logistic',block(v['cn']),block(v['archival_cn']),tab(ci,['reference','delta_lower','delta_upper','valid_draws'],['同协议强基准','年度AP差下界','上界','有效抽样']),image('historical_years','这里是各seed指标的均值，与“先平均预测再算AP”是不同统计量。'),block('旧档案L1的pooled AP=.295332来自训练截至2011、验证2012-2014选epoch的单seed配置，且不是原验证ROC规则选中的代表。P2所有强基准与新模型均按同一截止2014重训。不能把P1/P2截止年差异、年度rank归一化或seed融合收益全部叫作新算法提升。')),
          page('6. 稳定性、公司压力与是否值得',tab(seed_summary,['candidate','seeds','annual_AP_mean','annual_AP_min','annual_AP_max'],['配方','seeds','均值','最小','最大']),block('公司聚类配对bootstrap默认2,000次，整家公司抽样，并保留同一次抽样配对。区间条件于固定模型与当前历史样本，不包含完整HPO、校准、宏观冲击或未来数据版本不确定性。采用年度rank融合时，区间固定原年度参考群体。'),block('公司压力测试排除留组公司的所有拟合标签，但超参数已在更广公司上选择，因此准确名称是post-selection stress。自然未见公司按2014截止定义，failed公司极少；单类年份指标保留NA。计算成本、失败/无效trial与未启动计划均保留，不用参数量增加代替性能证据。')),
          page('7. 如何核查与面试表达',block('先读baseline_bridge、selection_trace和paired_company_intervals，再看全部trials、development_by_year和costs。模型、预处理、列序、训练步数、配置和预测保存在本地artifacts。verify从保存预测独立重算并随机重载一个树和一个NN。'),block('面试应说：在Codex辅助下完成有预算的真实优化，区分旧档案、同协议重训与新选择；能解释哪些变化有控制证据，哪些仍是猜测。个人贡献与原团队成果分别致谢，尚不声称本人手工重写每行或已独立复核。简历草稿仅保留本地。'),block('标签在公司内恒定，缺少真实事件日、财报发布日期与固定horizon。校准分数不是已验证的一年期PD；时间划分、公司hash和本轮优化都不能恢复未接触过的外部测试。原团队致谢和AI说明见父目录及本轮说明。'),block('技术来源：官方TabM、rtdl-num-embeddings、rtdl-revisiting-models、CatBoost、XGBoost和Optuna。版本与合同检查见configs/runtime_versions.json及software_contracts.json。'))]
    return [
      page('Performance V2: measured deltas',block(f'Qidian (Martin) Xing | {cohort["rows"]:,} firm-years, {cohort["companies"]:,} firms; {cohort["fit_rows"]:,} records through 2014; {cohort["historical_rows"]:,} previously inspected 2015-2018 records.'),first,block(v['en']),block('Yr AP is equal-weight annual AP; Pool AP and ROC pool years; R@5 averages annual review recall. Min is measured fit time. Mean-score ensembles differ from mean seed metrics. Sel is selection within each version/family. Archive rows are source observations, not new experiments. Annual ROC guardrails and full identities are in selection_trace.csv and baseline_bridge.csv.')),
      page('2. Why the legacy recipe could look stronger',tab(bridge,['candidate','AP','fit_seconds'],['2012 controlled recipe','AP','Seconds']),block('The legacy residual network has width192/four blocks and about302k parameters, versus36,801 for the V1 selected width64/two-block ROS pipeline. The latter used18 final epochs; the V1 core table used a different16-epoch configuration. Inputs, clipping, loss weights, stopping information and refitting also changed.'),block('Legacy and sklearn AP agree on actual saved predictions and tied-score fixtures. Capacity differences therefore establish a changed experiment, not an automatic explanation of generalization. These seed42/single-year controls provide conditional mechanism evidence, not universal causal attribution. The included epoch31 checkpoint is distinct from the bridge epoch23 record; exact archived13 selection never enters early P2 development folds.')),
      page('3. Training budgets and numerical representation',image('update_budget','Actual updates and positive exposures distinguish optimization budget from sampling.'),tab(budget[budget.candidate.str.contains('equal2000')],['candidate','actual_updates','positive_exposures','train_BCE'],['Fixed-update control','Updates','Positive exposures','Train BCE']),block('Three treatments receive2,000 actual updates. Numerical transforms and bin knots fit current training roles only; binary masks bypass quantile mapping. Constant/duplicate masks are removed without dropping records. Official numerical embedding modules and TabM are used; TabM averages per-head losses during training and per-head sigmoid probabilities at inference.'),block('A V2 migration error in the V1 ROS epoch-equivalent refit used the natural row count. That trial was invalidated, retained and replaced before selection. It was not a defect in preserved V1 code, and ordinary P2 step-refit HPO was unaffected.')),
      page('4. Search coverage, selection and confirmation',tab(coverage,['family','COMPLETE','FAIL','PRUNED'],['Family','Complete','Failed','Pruned']),image('search_cost','Real three-year screening configurations; folds/seeds are not additional methods.'),block('Screening years are 2009/2011/2013. Shortlisted candidates complete all five 2009-2013 years and three registered seeds. Selection averages seed/year AP, with 0.005 near-tie tolerance, recent-year/review-capacity/cost tie-breaks and ROC/Top5 guardrails. Strong fixed baselines remain eligible. Candidate/blend choices freeze before one 2014 confirmation batch; confirmation cannot reopen search.')),
      page('5. Historical evidence against strong baselines',block(v['en']),block(v['archival_en']),tab(ci,['reference','delta_lower','delta_upper','valid_draws'],['Same-cutoff reference','Annual AP delta low','High','Draws']),image('historical_years','Averages of individual-seed metrics, distinct from mean-score ensemble metrics.'),block('The archived L1 pooled AP0.295332 came from one seed and different selection/training roles and was not the old validation-ROC-selected representative. P2 retrains every comparison baseline through2014. P2-selected cutoff2012 transfers are explicitly separate from strict P1. Annual-rank pooled AP is compared only with matching annual-rank baselines.')),
      page('6. Cost, uncertainty and reproducibility',tab(seed_summary,['candidate','seeds','annual_AP_mean','annual_AP_min','annual_AP_max'],['Recipe','Seeds','Mean annual AP','Min','Max']),block('Paired company bootstrap uses2,000 draws and carries all annual records together. Intervals condition on fitted models and the current historical sample; they omit full HPO/training uncertainty and common macro shocks. Rank comparisons additionally fix original cohort maps. The company holdouts are accurately labeled post-selection stress, and2014-cutoff naturally unseen failed firms are few.'),block('All trials, invalidated/failed attempts, prediction coverage, final artifacts and numerical checks are retained. The local model directory contains preprocessing, feature order, states, configs and fit histories; raw data, complete predictions and private cases are excluded from public Git. Reload/prediction commands and exact dependencies are in README and requirements_v2.txt.'),block('This is an owner-directed, Codex-assisted independent extension of an earlier UCLA MFE413 team project. Parent acknowledgements remain intact. Company-constant labels, missing event/publication dates and prior historical exposure preclude verified one-year PD or prospective deployment claims. Official primary implementation sources and versions are documented; no paper benchmark performance is assumed.'))]

def actual_changes_page(c,matched,language):
    cn=language=='cn';selected=c['selected'];rows=[]
    for role in ['overall','DL']:
        key=selected[role];cfg=json.loads((ROOT/f'configs/trial_{key}.json').read_text());trace=c['selection_trace'].set_index('candidate').loc[key]
        details={k:cfg[k] for k in ['family','features','preprocess','width','depth','iterations','leaf','max_features','embedding','sampling','weight_alpha','learning_rate','schedule','window','half_life','company_beta'] if k in cfg}
        rows.append(block((f'{role} 选定配方：' if cn else f'{role} selected recipe: ')+json.dumps(details,ensure_ascii=False,sort_keys=True)+ (f'；五年×三seed AP={trace.macro_AP:.3f}。' if cn else f'; five-year/three-seed AP={trace.macro_AP:.3f}.')))
    valid=matched[matched.status=='COMPLETE']
    if len(valid):rows.append(tab(valid,['family','no_embedding_AP','piecewise_AP','delta_AP'],['主干','无embedding AP','PLE AP','变化'] if cn else ['Backbone','No embedding AP','PLE AP','Change']))
    else:rows.append(block('未完成成对数值embedding控制，不声称其有独立增益。' if cn else 'Matched numerical-embedding pairs were incomplete; no isolated embedding benefit is claimed.'))
    keys=list(dict.fromkeys([selected['overall'],selected['DL'],'baseline_legacy_strong_inner_l1','baseline_v1_lags_xgb','baseline_v1_selected_rf']))
    confirmation=c['confirmation_2014'];confirmation=confirmation[confirmation.candidate.isin(keys)][['candidate','AP','ROC_AUC','recall5']].copy()
    confirmation.candidate=confirmation.candidate.map(lambda k:'Overall' if k==selected['overall'] else ('DL' if k==selected['DL'] else k.replace('baseline_','')))
    rows.append(tab(confirmation,['candidate','AP','ROC_AUC','recall5'],['2014 一次确认','AP','ROC','R@5'] if cn else ['One 2014 confirmation','AP','ROC','R@5']))
    dl=c['main'].loc['P2_seed_mean__'+selected['DL']];old_dl=c['main'].loc['P2_seed_mean__baseline_v1_selected_dl_corrected_steps'];old_l1=c['main'].loc['P2_seed_mean__baseline_legacy_strong_inner_l1']
    rows.append(block(f'神经网络单配方：选定DL年度 AP={dl.macro_AP:.3f}；同截止年V1 DL={old_dl.macro_AP:.3f}（差{dl.macro_AP-old_dl.macro_AP:+.3f}），旧强L1配方={old_l1.macro_AP:.3f}（差{dl.macro_AP-old_l1.macro_AP:+.3f}）。这些是先平均五seed预测再评分的历史点估计，稳定性与区间另列。' if cn else f'Neural single recipe: selected DL annual AP={dl.macro_AP:.3f}; same-cutoff V1 DL={old_dl.macro_AP:.3f} (difference {dl.macro_AP-old_dl.macro_AP:+.3f}), legacy L1={old_l1.macro_AP:.3f} (difference {dl.macro_AP-old_l1.macro_AP:+.3f}). These are historical point estimates from mean five-seed predictions; stability and intervals are reported separately.'))
    rows.append(block('PLE差异仅是同主干、单seed、三个开发年的成对结果。最终配方同时改变多项时，只能归因于整条流水线；2014不会触发重选。' if cn else 'PLE deltas are matched-backbone, one-seed, three-year development results. Multivariable final changes support pipeline comparisons; the 2014 batch never triggers reselection.'))
    early=c['main'].loc['P2_selected__fit_cutoff_2012_transfer__overall'];late=c['main'].loc[f'P2__{selected["overall"]}__seed42']
    rows.append(block(f'固定选定配方和seed42，仅改变训练截止规则：截至2012的年度AP={early.macro_AP:.3f}，截至2014={late.macro_AP:.3f}。这是用过2013选择信息的P2迁移，不能冒称严格P1；滚动窗配方同时按相同窗长规则移动起点。' if cn else f'Fixed selected recipe and seed42, changed fit cutoff: annual AP={early.macro_AP:.3f} through 2012 versus {late.macro_AP:.3f} through 2014. This P2-selected transfer used 2013 selection information and is not strict P1; rolling recipes shift the start with the same window-length rule.'))
    if json.loads((ROOT/'EXECUTION_STATE.json').read_text()).get('active_window_registration'):
        rows.append(block('计算预算：首个8小时窗口到期时仅有部分开发结果；用户随后明确授权续跑，另登记最多8小时。两次时钟、原部分交付和已完成fold均保留；本次最终结果使用额外授权计算，不能写成首轮8小时内全部完成。实际计时与fit调用见experiment_summary和fit_accounting。' if cn else 'Compute disclosure: the first eight-hour window expired with partial development evidence. The owner explicitly authorized a separately registered additional eight-hour continuation. Both clocks and prior partial outputs are preserved; final results use additional authorized compute. Actual timing and fit calls are in experiment_summary and fit_accounting.'))
    return page('本轮究竟改了什么：用新结果回答' if cn else 'What changed: measured recipe and confirmation',*rows)

def write_summaries(c,matched):
    v=verdict(c);selected=c['selected'];ledger=[json.loads(s) for s in (ART/'logs/fits.jsonl').read_text().splitlines()]
    trial=c['trials'];coverage=trial.groupby(['stage','family','status']).size().rename('configurations').reset_index();coverage.to_csv(RESULT/'trial_coverage.csv',index=False)
    preservation=json.loads((RESULT/'verification.json').read_text())
    attempts=ART/'logs/trial_attempts.jsonl'
    if attempts.exists():
        records=[json.loads(s) for s in attempts.read_text().splitlines()]
        for record in records:
            if record.get('error'):record['error']=str(record['error']).replace(str(V1.parent),'[SOURCE_ROOT]')
            record['identity']='Archived interrupted/retried attempt state; not an active process or an extra configuration'
        pd.DataFrame(records).to_csv(RESULT/'trial_attempts.csv',index=False)
    screened=trial[(trial.stage=='screen')&(trial.status=='COMPLETE')]
    summary=dict(author='Qidian (Martin) Xing',created_at_utc=now(),wall_clock_hours_to_report=hours(),hard_budget_hours=8,model_fit_calls=len(ledger),completed_screen_configurations=int(len(screened)),unique_effective_screen_recipes=int(screened.functional_recipe_hash.nunique()),completed_refinement_configurations=int(((trial.stage=='refinement')&(trial.status=='COMPLETE')).sum()),final_pipelines=len(json.loads((ROOT/'configs/final_inventory.json').read_text())),selected=selected,strongest_same_cutoff_baseline=v['strongest_name'],historical_annual_AP_difference=v['gap'],all_point_benchmarks_beaten=v['all_point_benchmarks_beaten'],all_conditional_intervals_positive=v['all_conditional_intervals_positive'],historical_exposure='Previously inspected across versions; not a new holdout',paid_services=False,publication='pending exact release audit and authenticated branch push')
    execution=json.loads((ROOT/'EXECUTION_STATE.json').read_text())
    continuation=execution.get('active_window_registration')
    summary['active_window_hours_to_report']=hours()
    summary['original_window_started_at_utc']=execution['started_at_utc']
    summary['original_window_deadline_utc']=execution['deadline_utc']
    summary['separate_user_authorized_continuation']=json.loads((ROOT/continuation).read_text()) if continuation else None
    summary['completed_fold_fit_seconds']=float(c['development_by_year'].fit_seconds.sum())
    summary['budget_qualification']='Original eight-hour window expired partial; a separate owner-authorized eight-hour continuation reused saved development evidence. Neither clock resets on resume.' if continuation else 'One registered eight-hour local window.'
    dump(RESULT/'experiment_summary.json',summary)
    public_ledger=pd.DataFrame([{k:r.get(k) for k in ['fit_call','at_utc','window','tag','rows','max_fit_year','config_hash','optimizer_max_steps','resampled_rows','positive_weight','positive_fraction','weight_ESS','weight_min','weight_max','C']} for r in ledger]);public_ledger.to_csv(RESULT/'fit_accounting.csv',index=False)
    (ROOT/'README.md').write_text(f'''# Financial Distress - Performance V2

**Qidian (Martin) Xing.** An owner-directed, Codex-assisted performance extension of the earlier team project; see [parent acknowledgements](../ACKNOWLEDGEMENTS.md) and [contribution statement](CONTRIBUTIONS.md).

{v['en']}

This version completed {summary['completed_screen_configurations']} three-year screening trials ({summary['unique_effective_screen_recipes']} effective distinct recipes, accounting for matched no-embedding controls) and {summary['completed_refinement_configurations']} bounded local-refinement configurations, with {len(ledger)} recorded model-fit calls and {summary['final_pipelines']} frozen final pipelines. Folds, seeds and shared TabM heads are not counted as different methods. Coverage targets and actual shortfalls are in [trial coverage](results/trial_coverage.csv). Active-window time to report: {hours():.2f} hours. {summary['budget_qualification']} See [continuation registration](configs/continuation_window_02.json) for the separately authorized window; this is additional compute, not a claim that all work fit the first eight hours. No paid compute. Fit seconds use the monotonic process timer, while the UTC budget includes observed time gaps; these clocks are not interchangeable.

The 2015-2018 historical labels were inspected in earlier versions. V2 is retrospective, not a new holdout or an annual probability-of-default study. All P2 strong baselines are retrained through 2014 on the same permitted records; P1 and P2-selected cutoff-2012 transfers remain separately identified.

## Evidence

- [First comparison table](results/baseline_bridge.csv), [selection trace](results/selection_trace.csv), [2014 confirmation](results/confirmation_2014.csv), [historical results](results/historical_2015_2018.csv).
- [All logical trials](results/trials.csv), [attempt history](results/trial_attempts.csv), [budget shortfalls](results/search_budget_summary.csv), [fold results](results/development_by_year.csv), [matched embedding controls](results/numeric_matched_controls.csv), [training-budget controls](results/training_budget_ablation.csv).
- [Five-seed results](results/seeds.csv), [paired company intervals](results/paired_company_intervals.csv), [costs](results/costs.csv), [numerical verification](results/verification.json).
- [English report](reports/Research_Report_V2_EN.pdf), [Chinese performance report](reports/PERFORMANCE_DELTA_CN.pdf), [Chinese interview addendum](reports/Interview_Addendum_V2_CN.md).
- [Frozen protocol](V2_PROTOCOL.yaml), [execution corrections and deviations](PROTOCOL_EXECUTION_NOTES.md), [sources and installed versions](REFERENCES.md), [AI and contribution disclosure](CONTRIBUTIONS.md).

## Local execution

Work from this `performance_v2` directory. The wrapper uses an isolated environment and the preserved parent implementation; the installed package versions are in `requirements_v2.txt` and `configs/runtime_versions.json`. Supply the authorized original dataset at the parent project's documented path; raw records and all serialized models remain local.

```bash
scripts/run_performance_v2.sh scripts/progress_v2.py
scripts/resume_performance_v2.sh --phase all
scripts/run_performance_v2.sh -m distress_v2.verify
scripts/run_performance_v2.sh -m distress_v2.inference --help
scripts/run_performance_v2.sh scripts/create_local_replication.py --name rerun1
```

Resume never resets either registered wall clock or reopens search after confirmation. After the registered run is complete, preserve its freezes; a new search requires a newly registered version, not deleting a lock. Local model files are indexed by `configs/final_inventory.json`; each has a preprocessing/config sidecar, and neural states additionally have `.pt` files. Models are trusted local joblib artifacts, not safe files to load from strangers. Private data, company-level predictions, audit manifests, runtime state and resume drafts are excluded from public Git.

The replication command above is a read-only preflight. When the owner explicitly adds `--prepare`, it creates a new sibling directory, preserves this directory and clock, reuses the existing local runtime, and registers a separate eight-hour run; training still requires manually running its resume script. Full original-role reconstruction requires the owner's original authorized source archive and data, which are intentionally absent from a public clone. Earlier bridge source snapshots and adaptive budget-dependent HPO counts need not be bit-identical to a fresh corrected run; per-trial configs, source commits and saved predictions identify the original evidence. No additional replication was launched during this bounded run.

The inherited V1 paths and historical lock are unchanged. The V2 bridge used a source snapshot before the compact-mask implementation; its purpose is a controlled mechanism diagnostic, explicitly distinct from P2 HPO. One migrated V1 ROS trial was invalidated for an epoch-refit step error and replaced before selection; it is retained in the ledger and does not explain original V1 performance.

The final preservation check confirmed {preservation['protected_files_unchanged']} readable protected files against their initial hashes. {preservation['protected_files_read_denied']} private original file(s) could not be read under current system permissions, including an escalated read-only attempt; no current hash-equality claim is made for those files. Exact local exceptions remain in the private audit directory and are not published.
''',encoding='utf-8')
    interview=f'''# 性能优化 V2：中文面试增补

作者：Qidian (Martin) Xing。以下是 Codex 辅助完成的新实验，原团队成果按父目录致谢；不把 AI 编写的全部代码声称为本人手写。

## 结果先行

{v['cn']}

开发期 overall={selected['overall']}，DL={selected['DL']}。先按五年×三seed选定，再打开2014确认与已暴露历史期。没有事后把历史最高行补成winner。

## 旧版为什么可能更好？

配置确实不同：旧网络192×4约30万参数，V1选定网络64×2、36,801参数；输入、截尾、权重、采样、停止与重拟合角色也变了。旧AP实现与sklearn对真实分数和ties样例一致。不能仅凭小网络、epoch少或某个历史分数下降就诊断欠拟合。桥接实验固定2012和seed42逐项对照，其结果只支持该环境；全流程改动不能当单项因果效应。

## 如何解释采样和训练充分性？

相同epoch会给ROS/SMOTE更多更新。先比较各2,000次真实更新，再看正类曝光量、训练/内部停止AP和BCE、梯度范数及饱和比例。复制的正类记录不是新增独立公司。V2复现V1时的ROS重训步数错误已标无效并修复；这不是原V1代码的错误。

## 哪些改动有新证据？

成对数值embedding结果见 numeric_matched_controls.csv；不同输入的桥接见 feature_preprocess_ablation.csv；预算见 training_budget_ablation.csv。最终选定配方与一次2014确认写在两份PDF。只对已完成成对控制讨论机制；随机搜索不同参数的两行只能比较整条配方。

## 为什么主指标用年度AP？

任务是自然年度截面的少数类筛查。逐年AP等权避免跨年score尺度和样本量主导比较；每年Top5%对应固定审阅容量。概率均值、seed指标均值和年度rank融合是不同统计量。年度rank融合的pooled AP必须与同rank基准比较。

## 如何说明不确定性和公司泛化？

公司聚类配对bootstrap整家公司抽样2,000次；它条件于固定模型和当前历史样本，未涵盖完整HPO、未来宏观冲击或数据修订。留组拟合不读留组标签，但参数在更广公司上选过，因此称post-selection stress。2014截止下自然unseen的failed公司很少，单类年度指标保留NA。标签在公司内恒定且无事件日/发布日期，不能宣称真实一年期PD或前瞻预警。

## 是否值得增加算力？

先用 costs.csv 比较实际拟合分钟、参数/步数、跨seed范围，再检查对强旧配方、LAGS、CatBoost和V1 RF的同截止年差异及区间。若只恢复V1弱代表或区间跨零，应直接说局部改善或证据不确定。实际完成的配置数是{summary['completed_screen_configurations']}，不是预定目标，更不是fold/seed的总和。

## 能诚实说的个人贡献

本人提出并限定性能目标、授权本地数据与预算；Codex执行迁移、调参、核验和材料生成。可展示真实实验记录并解释取舍。个人独立掌握和手工复核程度尚未由本次自动执行证明；发布简历前应逐行理解并本人确认。
'''
    (ROOT/'reports/Interview_Addendum_V2_CN.md').write_text(interview,encoding='utf-8')
    m=v['main'];old=v['old'];ref=v['strongest']
    resume=f'''# Resume drafts - private owner review required

Author: Qidian (Martin) Xing. Use only after personally reviewing and understanding the Codex-assisted implementation and results. Attribution to the prior team remains in the contribution statement. Historical evaluation is retrospective and previously inspected.

## English

- Benchmarked {summary['unique_effective_screen_recipes']} financial-distress configurations with temporal folds and five-seed checks; a development-selected recipe achieved {m.macro_AP:.3f} historical annual AP against {ref.macro_AP:.3f} for the strongest same-cutoff baseline on previously inspected data.
- Compared legacy residual networks, tuned tree models and numerical representations through a Codex-assisted local workflow, auditing sampling updates, company generalization and paired uncertainty while preserving original team results and source attribution.

## 中文

- 比较{summary['unique_effective_screen_recipes']}个有效不同配方，以时间折和五seed核验财务困境排序；开发期选定模型的历史年度AP为{m.macro_AP:.3f}，同截止年最强基准为{ref.macro_AP:.3f}，未将已暴露历史期称新holdout。
- 在Codex辅助下对照旧Residual、调参树模型与数值表示，核查采样更新步数、公司泛化及配对区间，保留原团队成果与准确来源说明。

No prospective PD, sole-author model-development, or unsupported improvement claim is made. Point difference against old L1 recipe: {m.macro_AP-old.macro_AP:+.3f} annual AP; strongest baseline: {v['strongest_name']}; paired intervals remain in the report.
'''
    (ROOT/'reports/Resume_Bullets_V2_EN_CN.md').write_text(resume,encoding='utf-8')

def report():
    if not (ART/'pdf_operation_started.json').exists():
        state('verified_awaiting_pdf_authoring_marker',complete=False)
        print('Numerical verification complete. PDF authoring marker required before report generation.');return
    from .search import export
    export()
    c=context();configuration_bridge();delta=delta_table(c);budget=export_ablations(c);matched=matched_numeric_controls(c);figures(c)
    from .pdf_layout import render
    (ROOT/'reports').mkdir(exist_ok=True)
    for language,name in [('en','Research_Report_V2_EN'),('cn','PERFORMANCE_DELTA_CN')]:
        document=pages(c,delta,budget,language)
        document.insert(5,actual_changes_page(c,matched,language))
        # Readability fixes do not modify scientific results or their identities.
        for p in document:
            for b in p['blocks']:
                if b['type']=='paragraph' and language=='en':
                    for old,new in [('width192','width 192'),('about302k','about 302k'),('versus36,801','versus 36,801'),('width64','width 64'),('used18','used 18'),('different16','different 16'),('seed42','seed 42'),('epoch31','epoch 31'),('epoch23','epoch 23'),('archived13','archived 13'),('receive2,000','receive 2,000'),('AP0.295332','AP 0.295332'),('through2014','through 2014'),('cutoff2012','cutoff 2012'),('uses2,000','uses 2,000'),('and2014','and 2014'),('MFE413','MFE 413')]:b['text']=b['text'].replace(old,new)
        render(document,ROOT/f'reports/{name}.pdf',ROOT/f'reports/{name}.md',language)
    write_summaries(c,matched);claims()
    state('reports_generated_pending_visual_and_release_checks',complete=False,elapsed_hours=hours())

if __name__=='__main__':report()
