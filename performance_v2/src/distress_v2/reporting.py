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
    return c

def delta_table(c):
    rows=[];archive=pd.read_csv(RESULT/'legacy_archived_metrics.csv')
    for r in archive.itertuples():
        rows.append(dict(recipe='V0 L1' if 'lasso' in r.feature_set else 'V0 base',version='V0',fit_cutoff=2011,input='fixed13/std01' if 'lasso' in r.feature_set else 'RAW18/std01',selected='No' if 'lasso' in r.feature_set else 'Yes',seeds=1,annual_AP=np.nan,pooled_AP=r.test_pr_auc,ROC=r.test_roc_auc,R5=np.nan,seconds=np.nan,identity='Archive only'))
    h=pd.read_csv(V1/'results/public/historical_metrics.csv');h=h[h.score_type=='calibrated']
    for family,feature,label in [('random_forest','S1','V1 RF'),('residual_mlp','S1','V1 DL'),('xgboost','LAGS','V1 lag XGB'),('catboost','S0','V1 Cat raw')]:
        if label in ['V1 RF','V1 DL']:
            selected_id=json.loads((V1/'configs/final_freeze.json').read_text())['representatives']['ML' if label=='V1 RF' else 'DL'];d=h[h.config_id==selected_id]
        else:d=h[(h.family==family)&(h.feature_set==feature)]
        r=d.iloc[0];rows.append(dict(recipe=label,version='V1',fit_cutoff=2012,input=feature,selected='Yes' if label in ['V1 RF','V1 DL'] else 'No',seeds=1,annual_AP=r.macro_year_AP,pooled_AP=r.AP,ROC=r.ROC_AUC,R5=r.macro_recall_top5,seconds=r.fit_seconds,identity='Saved V1'))
    selected=c['selected'];keys=list(dict.fromkeys(['baseline_legacy_strong_inner_l1','baseline_v1_lags_xgb','baseline_v1_raw_catboost','baseline_v1_selected_rf',selected['overall'],selected['DL']]))
    for key in keys:
        row=c['main'].loc['P2_seed_mean__'+key];config=json.loads((ROOT/f'configs/trial_{key}.json').read_text())
        short={'baseline_legacy_strong_inner_l1':'P2 old L1','baseline_v1_lags_xgb':'P2 lag XGB','baseline_v1_raw_catboost':'P2 Cat raw','baseline_v1_selected_rf':'P2 V1 RF'}.get(key,'P2 overall' if key==selected['overall'] else 'P2 DL')
        component=c['historical_2015_2018'];cost=component[(component.candidate==key)&(component.channel=='P2')].fit_seconds.sum()
        rows.append(dict(recipe=short,version='P2',fit_cutoff=2014,input=config['features']+'/'+config['preprocess'],selected='Yes' if key in [selected['overall'],selected['DL']] else 'Fixed',seeds=int(row.seed_count),annual_AP=row.macro_AP,pooled_AP=row.AP,ROC=row.ROC_AUC,R5=row.recall5,seconds=cost,identity='New seed mean'))
    if selected['blend']['admitted']:
        r=c['main'].loc['P2_fixed_rank_blend'];rows.append(dict(recipe='P2 blend',version='P2',fit_cutoff=2014,input='annual ranks',selected='Yes',seeds=5,annual_AP=r.macro_AP,pooled_AP=r.AP,ROC=r.ROC_AUC,R5=r.recall5,seconds=np.nan,identity='Frozen blend'))
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
    selected=c['selected'];main=c['main'].loc['P2_seed_mean__'+selected['overall']]
    names=['baseline_legacy_strong_inner_l1','baseline_v1_lags_xgb','baseline_v1_raw_catboost','baseline_v1_selected_rf']
    refs={key:c['main'].loc['P2_seed_mean__'+key] for key in names}
    best=max(refs,key=lambda key:refs[key].macro_AP);gap=main.macro_AP-refs[best].macro_AP
    old=c['main'].loc['P2_seed_mean__baseline_legacy_strong_inner_l1']
    ci=c['paired_company_intervals'];q=ci[(ci.model==selected['overall'])&(ci.metric=='macro_year_AP')]
    positive=len(q)==4 and bool((q.delta_lower>0).all())
    return dict(main=main,strongest_name=best,strongest=refs[best],gap=gap,old=old,all_point_benchmarks_beaten=gap>0,all_conditional_intervals_positive=positive,
      en=f"The development-selected overall model has historical annual AP {main.macro_AP:.3f}, versus {refs[best].macro_AP:.3f} for the strongest frozen same-cutoff baseline point estimate ({best}). The difference is {gap:+.3f}. "+('It exceeds all four baseline point estimates. ' if gap>0 else 'It does not exceed every strong baseline; this is not a comprehensive performance upgrade. ')+('All four paired annual-AP lower bounds are positive under the conditional company bootstrap.' if positive else 'The paired intervals do not uniformly establish a positive improvement over all four strong baselines.'),
      cn=f"开发期选定的整体模型，历史年度 AP={main.macro_AP:.3f}；冻结的同截止年强基准中，点估计最高的是 {best}（{refs[best].macro_AP:.3f}），差值 {gap:+.3f}。"+('点估计超过四条强基准。' if gap>0 else '没有超过所有强基准，不能称全面性能升级。')+('四组配对年度 AP 区间下界均为正，但仍只是固定模型与当前样本下的条件证据。' if positive else '配对区间没有一致支持对四条强基准都有确定的正向差异。'))

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
    for family,g in t.groupby('family'):ax.scatter(g.seconds,g.macro_AP,label=family,s=16,alpha=.65)
    ax.set(xscale='log',xlabel='Measured three-year trial wall time (seconds)',ylabel='Screening mean annual AP',title='Optimization cost and development performance');ax.legend(fontsize=7,ncol=3);fig.tight_layout();fig.savefig(out/'search_cost.png',dpi=160);plt.close(fig)

def block(text):return {'type':'paragraph','text':text}
def image(name,caption):return {'type':'figure','path':f'results/figures/{name}.png','caption':caption}
def tab(df,columns,labels=None):
    rows=[]
    for _,r in df.iterrows():
        row=[]
        for col in columns:
            value=r[col]
            row.append('NA' if pd.isna(value) else (f'{value:.3f}' if isinstance(value,(float,np.floating)) else str(value)))
        rows.append(row)
    return dict(type='table',header=labels or columns,rows=rows)
def page(title,*blocks):return dict(title=title,subtitle=None,blocks=list(blocks))

def pages(c,delta,budget,language):
    cn=language=='cn';v=verdict(c);selected=c['selected']
    overview=delta[~delta.recipe.isin(['V1 lag XGB','V1 Cat raw','P2 Cat raw'])].copy()
    overview['input']=overview.input.replace({'INNER_L1/std01':'L1/01','V1_LAGS/v1':'Lag3/V1','V1_S1/v1':'S1/V1','fixed13/std01':'13/01','RAW18/std01':'18/01'})
    overview['input']=overview.input.str.replace('RATIO_CORE','Ratio',regex=False).str.replace('SLOG_RATIO','LogR',regex=False).str.replace('LAGS_RAW3','Lag3',regex=False).str.replace('LAGS_SLOG3','Log3',regex=False).str.replace('DYNAMIC_COMPACT','Dyn',regex=False).str.replace('quantile','QT',regex=False).str.replace('std005','005',regex=False).str.replace('std01','01',regex=False)
    overview['selected']=overview.selected.replace({'Yes':'Y','No':'N','Fixed':'Pre'});overview['minutes']=overview.seconds/60;overview['identity']=overview.identity.replace({'Archive only':'Archive','Saved V1':'V1','New seed mean':'Mean score','Frozen blend':'Rank blend'})
    columns=['recipe','fit_cutoff','input','selected','seeds','annual_AP','pooled_AP','ROC','R5','minutes','identity']
    first=tab(overview,columns,['Recipe','Cutoff','Input','Sel','Seeds','Yr AP','Pool AP','ROC','R@5','Min','Source'])
    complete=c['trials'];coverage=pd.crosstab(complete.loc[complete.stage=='screen','family'],complete.loc[complete.stage=='screen','status']).reindex(columns=['COMPLETE','FAIL','PRUNED'],fill_value=0).reset_index()
    bridge=c['bridge'][['candidate','AP','ROC_AUC','fit_seconds']].copy();bridge.candidate=bridge.candidate.str.replace('bridge_','',regex=False)
    ci=c['paired_company_intervals'];ci=ci[(ci.model==selected['overall'])&(ci.metric=='macro_year_AP')][['reference','delta_lower','delta_upper','valid_draws']]
    seeds=c['seeds'];seed_summary=seeds.groupby('candidate').agg(seeds=('seed','nunique'),annual_AP_mean=('macro_AP','mean'),annual_AP_min=('macro_AP','min'),annual_AP_max=('macro_AP','max')).reset_index()
    seed_summary['candidate']=seed_summary.candidate.str.slice(0,38)
    if cn:
        return [
          page('性能优化 V2：先看真实增量',block('Qidian (Martin) Xing | P1 旧角色、P2 ranking-first 新协议分别登记；2015-2018 已被多轮查看。'),first,block(v['cn']),block('表中 Yr AP 是四个历史年份 AP 的等权均值；Pool AP 为跨年合并统计；R@5 为逐年 Top5% 召回均值。Min 为实际拟合分钟；seed mean 表示平均预测形成的模型集合，不是独立数据重复。Archive 行未重新计算，不作为新实验。完整来源与分数类型见 baseline_bridge.csv。')),
          page('2. 旧版为何曾经更好：证据与边界',tab(bridge.head(12),['candidate','AP','fit_seconds'],['单年控制配方','2012 AP','秒']),block('可确认：旧 Residual 为 192×4、约30万参数；V1 选定网络为64×2、36,801参数、ROS、18个最终epoch。输入、截尾、损失权重、停止与重拟合角色也不同。旧AP函数与sklearn在真实保存分数及ties样例上相同，不能把差距归咎于指标定义。'),block('这些控制实验固定2012评分年与seed42，只支持该环境下的机制诊断。大容量并非自动更优，单个高分也不能证明普遍欠拟合或过拟合。固定旧13列含原2012-2014选择信息，只用于档案角色重训；正式开发每折重新执行L1。原包中epoch31的checkpoint与桥接表epoch23不同，身份分别保留。')),
          page('3. 采样、数值处理与真实训练预算',image('update_budget','横轴是实际optimizer更新次数。右图是正类曝光次数，复制或插值不会产生新的独立公司。'),tab(budget[budget.candidate.str.contains('equal2000')][['candidate','actual_updates','positive_exposures','train_BCE']],['candidate','actual_updates','positive_exposures','train_BCE'],['固定更新控制','更新次数','正类曝光','训练BCE']),block('另做无采样、ROS、SMOTE各2,000次更新控制。NN比较旧1/99%截尾、0.5/99.5%、稳健尺度与训练内normal quantile；二元mask不做quantile。仅在训练角色移除常量和重复mask，保留全部记录。数值embedding使用官方模块，TabM逐head求BCE后平均，推断平均sigmoid概率。'),block('V2迁移的V1 ROS epoch重拟合曾遗漏采样后行数，已将那条trial标记无效、保留证据并重跑；这是本轮迁移修复，不能拿来解释原V1下降。主搜索固定step分支不受影响。')),
          page('4. 搜索不是模型名单：覆盖与选择',tab(coverage,['family','COMPLETE','FAIL','PRUNED'],['家族','完成配置','失败','剪枝']),image('search_cost','每个点是一个真实三年初筛配置，未把fold或seed计作不同方法。'),block('初筛2009/2011/2013；入围补齐2009-2013五年，再按42/2026/31415三seed×年度平均选择。每家最多2个、总共最多8个；AP差0.005以内按近期稳定性、Top5召回和成本取舍。强旧配方、LAGS、CatBoost与原RF基准均可入围，不强迫新模型或S1胜出。')),
          page('5. 是否超过强旧配方，而非弱Logistic',block(v['cn']),tab(ci,['reference','delta_lower','delta_upper','valid_draws'],['同协议强基准','年度AP差下界','上界','有效抽样']),image('historical_years','这里是各seed指标的均值，与“先平均预测再算AP”是不同统计量。'),block('旧档案L1的pooled AP=.295332来自训练截至2011、验证2012-2014选epoch的单seed配置，且不是原验证ROC规则选中的代表。P2所有强基准与新模型均按同一截止2014重训。不能把P1/P2截止年差异、年度rank归一化或seed融合收益全部叫作新算法提升。')),
          page('6. 稳定性、公司压力与是否值得',tab(seed_summary,['candidate','seeds','annual_AP_mean','annual_AP_min','annual_AP_max'],['配方','seeds','均值','最小','最大']),block('公司聚类配对bootstrap默认2,000次，整家公司抽样，并保留同一次抽样配对。区间条件于固定模型与当前历史样本，不包含完整HPO、校准、宏观冲击或未来数据版本不确定性。采用年度rank融合时，区间固定原年度参考群体。'),block('公司压力测试排除留组公司的所有拟合标签，但超参数已在更广公司上选择，因此准确名称是post-selection stress。自然未见公司按2014截止定义，failed公司极少；单类年份指标保留NA。计算成本、失败/无效trial与未启动计划均保留，不用参数量增加代替性能证据。')),
          page('7. 如何核查与面试表达',block('先读baseline_bridge、selection_trace和paired_company_intervals，再看全部trials、development_by_year和costs。模型、预处理、列序、训练步数、配置和预测保存在本地artifacts。verify从保存预测独立重算并随机重载一个树和一个NN。'),block('面试应说：在Codex辅助下完成有预算的真实优化，区分旧档案、同协议重训与新选择；能解释哪些变化有控制证据，哪些仍是猜测。个人贡献与原团队成果分别致谢，尚不声称本人手工重写每行或已独立复核。简历草稿仅保留本地。'),block('标签在公司内恒定，缺少真实事件日、财报发布日期与固定horizon。校准分数不是已验证的一年期PD；时间划分、公司hash和本轮优化都不能恢复未接触过的外部测试。原团队致谢和AI说明见父目录及本轮说明。'),block('技术来源：官方TabM、rtdl-num-embeddings、rtdl-revisiting-models、CatBoost、XGBoost和Optuna。版本与合同检查见configs/runtime_versions.json及software_contracts.json。'))]
    return [
      page('Performance V2: measured deltas',block('Qidian (Martin) Xing | Retrospective failed-company classification; previously inspected 2015-2018.'),first,block(v['en']),block('Yr AP is equal-weight annual AP; Pool AP is cross-year AP; R@5 is mean annual review recall. Min is measured fit time. Mean-score ensembles differ from averaging seed metrics. Archive rows are source observations, not new experiments. Full identities, score types and configurations are in baseline_bridge.csv.')),
      page('2. Why the legacy recipe could look stronger',tab(bridge.head(12),['candidate','AP','fit_seconds'],['2012 controlled recipe','AP','Seconds']),block('The legacy residual network has width192/four blocks and about302k parameters, versus36,801 for the V1 selected width64/two-block ROS pipeline. The latter used18 final epochs; the V1 core table used a different16-epoch configuration. Inputs, clipping, loss weights, stopping information and refitting also changed.'),block('Legacy and sklearn AP agree on actual saved predictions and tied-score fixtures. Capacity differences therefore establish a changed experiment, not an automatic explanation of generalization. These seed42/single-year controls provide conditional mechanism evidence, not universal causal attribution. The included epoch31 checkpoint is distinct from the bridge epoch23 record; exact archived13 selection never enters early P2 development folds.')),
      page('3. Training budgets and numerical representation',image('update_budget','Actual updates and positive exposures distinguish optimization budget from sampling.'),tab(budget[budget.candidate.str.contains('equal2000')],['candidate','actual_updates','positive_exposures','train_BCE'],['Fixed-update control','Updates','Positive exposures','Train BCE']),block('Three treatments receive2,000 actual updates. Numerical transforms and bin knots fit current training roles only; binary masks bypass quantile mapping. Constant/duplicate masks are removed without dropping records. Official numerical embedding modules and TabM are used; TabM averages per-head losses during training and per-head sigmoid probabilities at inference.'),block('A V2 migration error in the V1 ROS epoch-equivalent refit used the natural row count. That trial was invalidated, retained and replaced before selection. It was not a defect in preserved V1 code, and ordinary P2 step-refit HPO was unaffected.')),
      page('4. Search coverage, selection and confirmation',tab(coverage,['family','status','configurations'],['Family','Status','Configurations']),image('search_cost','Real three-year screening configurations; folds/seeds are not additional methods.'),block('Screening years are2009/2011/2013. Shortlisted candidates complete all five2009-2013 years and three registered seeds. Selection averages seed/year AP, with0.005 near-tie tolerance, recent-year/review-capacity/cost tie-breaks and ROC/Top5 guardrails. Strong fixed baselines remain eligible. Candidate/blend choices freeze before one2014 confirmation batch; confirmation cannot reopen search.')),
      page('5. Historical evidence against strong baselines',block(v['en']),tab(ci,['reference','delta_lower','delta_upper','valid_draws'],['Same-cutoff reference','Annual AP delta low','High','Draws']),image('historical_years','Averages of individual-seed metrics, distinct from mean-score ensemble metrics.'),block('The archived L1 pooled AP0.295332 came from one seed and different selection/training roles and was not the old validation-ROC-selected representative. P2 retrains every comparison baseline through2014. P2-selected cutoff2012 transfers are explicitly separate from strict P1. Annual-rank pooled AP is compared only with matching annual-rank baselines.')),
      page('6. Cost, uncertainty and reproducibility',tab(seed_summary,['candidate','seeds','annual_AP_mean','annual_AP_min','annual_AP_max'],['Recipe','Seeds','Mean annual AP','Min','Max']),block('Paired company bootstrap uses2,000 draws and carries all annual records together. Intervals condition on fitted models and the current historical sample; they omit full HPO/training uncertainty and common macro shocks. Rank comparisons additionally fix original cohort maps. The company holdouts are accurately labeled post-selection stress, and2014-cutoff naturally unseen failed firms are few.'),block('All trials, invalidated/failed attempts, prediction coverage, final artifacts and numerical checks are retained. The local model directory contains preprocessing, feature order, states, configs and fit histories; raw data, complete predictions and private cases are excluded from public Git. Reload/prediction commands and exact dependencies are in README and requirements_v2.txt.'),block('This is an owner-directed, Codex-assisted independent extension of an earlier UCLA MFE413 team project. Parent acknowledgements remain intact. Company-constant labels, missing event/publication dates and prior historical exposure preclude verified one-year PD or prospective deployment claims. Official primary implementation sources and versions are documented; no paper benchmark performance is assumed.'))]
