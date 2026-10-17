"""Close the expired run using existing evidence only; never fit a model."""
import argparse
import json
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
from .common import ROOT,V1,ART,RESULT,dump,now,hours,database,state
from .data import load
from .verify import verify_development,verify_protected
from .search import export
from .reporting import block,tab,page,configuration_bridge,claims

MISSING=[
    'Full numerical-embedding and paired TabM performance search',
    'Target CatBoost/XGBoost and other family search coverage',
    'Five-year, three-seed shortlist stability and final selection',
    'One frozen 2014 confirmation batch',
    'Same-cutoff 2014 final fits with five registered seeds',
    'New frozen 2015-2018 historical batch and paired company intervals',
    'New company-generalization stress tests and saved final model pipelines',
    'Final publication checks and authenticated GitHub branch push']

def evidence():
    assert hours()>=8,'Closeout is for the expired run only'
    assert not (ART/'historical_opened.json').exists()
    db=database()
    pending=pd.read_sql_query("SELECT * FROM trial_records WHERE status='RUNNING'",db)
    for record in pending.to_dict('records'):
        record.update(archived_at_utc=now(),reason='Expired-run closeout; prior interrupted state retained')
        with (ART/'logs/trial_attempts.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        db.execute('UPDATE trial_records SET status=?,finished=?,error=? WHERE trial_id=?',('INCOMPLETE_BUDGET',now(),'No active training remains; interrupted partial folds retained',record['trial_id']))
    db.commit();db.close();export()
    df=load('search');checked=verify_development(df);preservation=verify_protected()
    t=pd.read_csv(RESULT/'trials.csv');d=pd.read_csv(RESULT/'development_by_year.csv')
    screen=t[(t.stage=='screen')&(t.status=='COMPLETE')];refine=t[(t.stage=='refinement')&(t.status=='COMPLETE')]
    events=[json.loads(s) for s in (ART/'logs/fits.jsonl').read_text().splitlines()]
    deadline=dt.datetime.fromisoformat(json.loads((ROOT/'EXECUTION_STATE.json').read_text())['deadline_utc'])
    after=sum(dt.datetime.fromisoformat(e['at_utc'])>deadline for e in events);assert after==0
    dump(RESULT/'partial_results_verification.json',dict(passed=True,scope='Saved development predictions only; not final historical validation',complete_experiment=False,verified_development_prediction_sets=checked,protected_files_unchanged=preservation['verified'],protected_files_read_denied=preservation['read_denied'],model_fit_calls_after_registered_deadline=after,checked_at_utc=now()))
    summary=dict(status='BUDGET_EXPIRED_PARTIAL',complete=False,author='Qidian (Martin) Xing',completed_screen_trials=len(screen),effective_distinct_screen_recipes=screen.functional_recipe_hash.nunique(),completed_refinement_trials=len(refine),bridge_trials=int(((t.stage=='bridge')&(t.status=='COMPLETE')).sum()),recorded_fit_calls=len(events),completed_fold_fit_seconds=float(d.fit_seconds.sum()),fit_timing_scope='Sum over completed folds only; excludes interrupted work and UTC gaps, not CPU-seconds',last_recorded_fit_start=events[-1]['at_utc'],deadline_utc=deadline.isoformat(),closed_at_utc=now(),closeout_prepared_after_window=True,new_final_model_pipelines=len(list((ART/'models').glob('*.joblib'))),new_historical_batch_run=False,performance_improvement_established=False,publication='NOT PUSHED: required final experiments and validation incomplete',missing=MISSING)
    dump(RESULT/'experiment_summary.json',summary)
    dump(RESULT/'selected_candidates.json',dict(status='NOT_SELECTED_BUDGET',overall=None,DL=None,reason='Registered stability/confirmation/final fits not completed'))
    dump(RESULT/'model_artifacts_status.json',dict(new_final_pipelines=summary['new_final_model_pipelines'],status='NOT_DELIVERED',note='Development predictions, configurations, metadata and learning curves are saved. The copied legacy checkpoint in the private audit is an original artifact, not a newly trained V2 final model.'))
    for filename,columns in [('confirmation_2014.csv',['candidate','AP','ROC_AUC','recall5']),('historical_2015_2018.csv',['name','channel','AP','macro_AP','ROC_AUC','recall5']),('seeds.csv',['candidate','seed','AP','macro_AP']),('paired_company_intervals.csv',['model','reference','metric','delta_lower','delta_upper'])]:
        p=RESULT/filename
        if not p.exists():pd.DataFrame(columns=columns).to_csv(p,index=False)
    d[['candidate','family','stage','seed','year','fit_seconds','parameters','best_steps','best_rounds']].to_csv(RESULT/'costs.csv',index=False)
    d[d.stage=='bridge'].to_csv(RESULT/'feature_preprocess_ablation.csv',index=False)
    budgets=[]
    for path in (ART/'logs').glob('bridge_*_complete.json'):
        m=json.loads(path.read_text());s=m['final_stats'];budgets.append(dict(candidate=m['tag'],actual_updates=s.get('actual_updates'),positive_exposures=s.get('positive_exposures'),negative_exposures=s.get('negative_exposures'),train_AP=s.get('train_AP'),train_BCE=s.get('train_BCE'),fit_seconds=m['fit_seconds']))
    pd.DataFrame(budgets).to_csv(RESULT/'training_budget_ablation.csv',index=False)
    attempts=[json.loads(s) for s in (ART/'logs/trial_attempts.jsonl').read_text().splitlines()]
    for record in attempts:
        if record.get('error'):record['error']=record['error'].replace(str(V1.parent),'[SOURCE_ROOT]')
        record['identity']='Archived interrupted/retried attempt, not an active process or new configuration'
    pd.DataFrame(attempts).to_csv(RESULT/'trial_attempts.csv',index=False)
    configuration_bridge()
    # Every row of this first table has the same three scoring years and seed.
    candidates=['baseline_v1_selected_rf','baseline_legacy_strong_inner_l1','baseline_v1_lags_xgb','baseline_v1_raw_catboost','baseline_v1_selected_dl_corrected_steps','mlp_000_d4daf458','legacy_residual_000_ed2c13d5']
    for family in ['catboost','xgboost','extra_trees','tabm']:
        q=screen[screen.family==family]
        if len(q):candidates.append(q.sort_values('macro_AP',ascending=False).iloc[0].trial_id)
    rows=[]
    labels={'baseline_v1_selected_rf':'V1/P2 RF','baseline_legacy_strong_inner_l1':'V0/P2 inner L1','baseline_v1_lags_xgb':'V1/P2 lag XGB','baseline_v1_raw_catboost':'V1/P2 Cat raw','baseline_v1_selected_dl_corrected_steps':'V1/P2 DL','mlp_000_d4daf458':'V2 MLP raw','legacy_residual_000_ed2c13d5':'V2 old Res raw'}
    indexed=t.set_index('trial_id')
    for key in dict.fromkeys(candidates):
        r=indexed.loc[key];cfg=json.loads(r.config_json);frames=[pd.read_csv(ART/f'predictions/{key}__seed42__year{year}.csv.gz',float_precision='round_trip') for year in [2009,2011,2013]];p=pd.concat(frames)
        rows.append(dict(recipe=labels.get(key,'V2 '+r.family),candidate=key,cutoff='08/10/12',input=cfg['features']+'/'+cfg['preprocess'],selected='Pending',seeds=1,annual_AP=r.macro_AP,pooled_AP=average_precision_score(p.y_true,p.score),ROC=roc_auc_score(p.y_true,p.score),R5=r.recall5,minutes=d[d.candidate==key].fit_seconds.sum()/60,source='DEV 09/11/13'))
    bridge=pd.DataFrame(rows);bridge.to_csv(RESULT/'baseline_bridge.csv',index=False)
    return summary,t,d,bridge,preservation

def documents(summary,t,d,comparison,preservation):
    preserved_en=f'All {preservation["verified"]} protected originals matched their initial hashes.' if not preservation['read_denied'] else f'{preservation["verified"]} readable protected originals matched; {preservation["read_denied"]} private original remains unreadable and unverified.'
    preserved_cn=f'原文件哈希复核：全部{preservation["verified"]}个原文件与初始清单一致。' if not preservation['read_denied'] else f'原文件哈希复核：{preservation["verified"]}个一致，{preservation["read_denied"]}个因系统拒读未能复核。'
    display=comparison.copy();display['selected']='N/A'
    first=tab(display,['recipe','cutoff','input','selected','seeds','annual_AP','pooled_AP','ROC','R5','minutes','source'],['Recipe/version','Fit to','Input/scale','Sel','Seeds','Year AP','Pool AP','ROC','R@5','Min','Identity'])
    archive=pd.read_csv(RESULT/'legacy_archived_metrics.csv')[['feature_set','test_pr_auc','test_roc_auc']]
    b=d[(d.stage=='bridge')&~d.candidate.str.contains('equal2000')][['candidate','AP','fit_seconds']].copy();b.candidate=b.candidate.str.replace('bridge_','',regex=False)
    coverage=pd.read_csv(RESULT/'search_budget_summary.csv')
    rf=float(comparison[comparison.candidate=='baseline_v1_selected_rf'].annual_AP.iloc[0]);mlp=float(comparison[comparison.candidate=='mlp_000_d4daf458'].annual_AP.iloc[0])
    en=[page('Performance V2 - partial budget closeout',block('Qidian (Martin) Xing | NOT a completed performance upgrade. The original UTC budget expired; no final model or new historical score is claimed.'),first,block(f'The strongest completed development baseline remains V1/P2 RF ({rf:.3f} mean annual AP), versus {mlp:.3f} for the plain raw-input MLP. These are seed42, three-year development results. They do not establish an improvement on 2015-2018.'),block('All rows use scoring years 2009/2011/2013 and fit cutoffs 2008/2010/2012. Pooled scores combine different fold-trained models. R@5 averages annual recall. No final representative was selected. This is the same-protocol comparison currently available, not the requested final cutoff-2014 benchmark.')),
        page('2. What the legacy bridge actually established',tab(archive,['feature_set','test_pr_auc','test_roc_auc'],['Old archive identity','Historical AP','Historical ROC']),tab(b,['candidate','AP','fit_seconds'],['Controlled 2012 recipe','AP','Seconds']),block('Archive scores are old observations, not newly reproduced results. Legacy L1 AP 0.295332 was not the old validation-ROC-selected representative. The legacy residual has width192/four blocks and 302,209 parameters at 18 inputs; selected V1 DL has 36,801 parameters and 18 refit epochs. Capacity, inputs, clipping, loss weights and refit roles all changed. AP implementations agree on saved scores and ties.'),block('Single-year/seed controls suggest that preprocessing, weighting and sampling can change results, but do not prove the cause of the historical gap. More capacity was not uniformly better. The included epoch31 checkpoint is not the archived epoch23 run.')),
        page('3. Real coverage and resource limits',tab(coverage,['family','actual_complete','target_complete','completed_target_shortfall'],['Family','Complete','Target','Shortfall']),block(f'{summary["completed_screen_trials"]} complete screening trials represent {summary["effective_distinct_screen_recipes"]} effective distinct recipes; the no-embedding numerical MLP duplicates the plain MLP. There are {summary["completed_refinement_trials"]} complete local-refinement trials, {summary["bridge_trials"]} single-year bridge controls and {summary["recorded_fit_calls"]} recorded fit calls. No-embedding TabM completed one full 200-epoch-ceiling configuration (development AP 0.323). PLE, full TabM tuning, FT and ranking performance searches were not completed.'),block('ROS/SMOTE controls received 2,000 updates each; exposures and learning curves are saved. The incorrect migrated V1 ROS refit was invalidated and replaced. An RF timeout inference based on UTC gaps was retracted. Monotonic completed-fold timings omit gaps and interrupted work; they are not total CPU-seconds. The causes of the large UTC gaps remain unverified.'),block(f'The last recorded model fit started at {summary["last_recorded_fit_start"]}; deadline {summary["deadline_utc"]}. No fit was initiated after that deadline. Closeout documentation was prepared after the window. Operational deviations, including the per-component rather than combined-fold timeout, are disclosed in PROTOCOL_EXECUTION_NOTES.md.')),
        page('4. What remains incomplete',*[block(x) for x in MISSING],block(preserved_en+' No new final pipeline was saved, and the GitHub branch was not pushed because final experimental/validation gates are unmet.'),block('Source, all logical trials, interrupted attempts, development predictions and curves remain local. A new compute window requires fresh owner authorization; the expired clock must not be reset. The original team remains credited in the parent acknowledgements; this extension is owner-directed and Codex-assisted. Company-constant labels and missing event/publication dates preclude verified one-year PD claims. Official implementation sources and versions are in REFERENCES.md.'))]
    cn=[page('性能优化 V2：预算到期的部分结果',block('Qidian (Martin) Xing | 本轮未完成性能升级。下表只有已核验的开发期结果，没有新的最终历史成绩或模型工件。'),first,block(f'共同开发口径下，固定V1/P2 RF仍最高：年度AP={rf:.3f}；普通Raw MLP={mlp:.3f}。均为seed42、三个评分年。不能据此填写“历史表现提升”，也不能与旧档案0.295的历史pooled AP直接比较。'),block('评分年为2009/2011/2013，对应训练截至2008/2010/2012。Pool AP与ROC合并不同折训练模型的评分；R@5为逐年召回均值。所有新配方的最终选择仍为Pending；截至2014的最终共同口径对照尚未完成。')),
        page('2. 旧版为什么曾经更好：能确认什么',tab(archive,['feature_set','test_pr_auc','test_roc_auc'],['旧档案身份','历史AP','历史ROC']),tab(b,['candidate','AP','fit_seconds'],['2012单年控制配方','AP','秒']),block('旧Residual为192×4，18维输入302,209参数；V1所选网络36,801参数、ROS、18个最终epoch。输入、截尾、权重、早停和重拟合角色都变了。旧AP与sklearn在真实保存评分和ties样例上一致，不能把差距归咎于指标定义。'),block('控制实验中，无权重、截尾和采样改变了单年结果；较大容量并非总更好。这里只验证了固定2012和seed42下的变化，不能证明它们导致了2015-2018的差距。旧L1的0.295不是按旧验证ROC选中的代表，也不是性能下界；原包epoch31 checkpoint与档案epoch23不同。')),
        page('3. 方法、取舍与未验证的假设',block('P1基础模型截至2012，2013校准，2014阈值；P2开发期选配置、最终允许拟合至2014。P2若用了2013选择信息，不能再把2013叫独立校准年。本轮没有完成新的P1/P2最终模型，不能借角色切换宣称算法进步。'),block('相同epoch并不代表相同更新次数：ROS/SMOTE改变样本数。已做各2,000更新的无采样/ROS/SMOTE控制并记录正类曝光；复制和插值不产生新的独立公司。曲线同时保存训练/内部停止AP、BCE、梯度和饱和比例，不能只凭epoch少诊断欠拟合。'),block('RAW17、精简比率、三年日历历史等已有真实开发结果。训练内截尾、标准化与quantile不接触评分标签，mask另行处理。无embedding的数值MLP与普通MLP得到完全相同的AP，因此不能算两种独立新方法。官方PLE与TabM合同测试通过，不等于完成了它们的性能实验。'),block('TabM只有一个完整无embedding配置，AP约0.323；尚未完成充分调参或PLE成对性能训练。FT、ranking与权重/滚动窗等覆盖不足，不能下架构优劣结论。所声明的TabM20-epoch控制最终没有启动，不能当作实验结果。')),
        page('4. 实际覆盖、成本与是否值得',tab(coverage,['family','actual_complete','target_complete','completed_target_shortfall'],['家族','完整配置','目标','缺口']),block(f'完整三年初筛{summary["completed_screen_trials"]}个trial，去掉功能重复后{summary["effective_distinct_screen_recipes"]}个有效不同配方；完整局部修正{summary["completed_refinement_trials"]}个，单年桥接{summary["bridge_trials"]}个，记账拟合调用{summary["recorded_fit_calls"]}次。fold、seed和TabM head都没有当作新方法。'),block('真实目标远未达到，不能称“充分调优”。总UTC窗口出现长间隔，但现有证据没有确认原因；单调拟合计时与总wall-clock分开报告。最后拟合启动在原截止前，截止后没有补跑。结项材料是在窗口结束后整理的。'),block('本轮没有建立超过旧强配方的历史证据，也没有完成五seed稳定性和公司泛化。不能用架构名称、参数量或单年高分来替代这些证据。更多计算是否值得，须在下一次获授权的预算内完成真实共同口径比较后再判断。')),
        page('5. 缺少哪一步、如何诚实表达',block('未完成：五年×三seed入围稳定性；2014一次确认；截至2014的五seed最终拟合；2015-2018冻结批量评分；公司留组压力、配对bootstrap和最终模型工件。完整PLE/TabM、CatBoost/XGBoost等搜索目标也有缺口。'),block('因此GitHub未推送：实验和核验门槛尚未满足，随后仍需现场核验GitHub身份与公开文件隐私。没有上传原始数据、公司级预测或私有源文件。'),block(preserved_cn+' 已停止本次训练与临时进程。'),block('面试可说：迁移并核对旧网络，执行了真实的开发期配置与采样控制，发现并纠正迁移错误；尚未完成最终性能验证。不能说已提高历史成绩、完成现代embedding调优或获得五seed公司泛化结论。署名为Qidian (Martin) Xing，原团队按父目录准确致谢；代码与分析由Codex辅助执行。'),block('原8小时预算不可重置。续跑需要新的明确计算授权及单独登记窗口，同时保留此次结果与失败记录。NEXT_ACTIONS.md列出缺项；当前resume命令只作流程说明，不能在过期窗口继续训练。'))]
    return en,cn

def run(render=False):
    if render and (RESULT/'partial_results_verification.json').exists():
        v=json.loads((RESULT/'partial_results_verification.json').read_text());assert v['passed'] and not v['complete_experiment']
        summary=json.loads((RESULT/'experiment_summary.json').read_text());t=pd.read_csv(RESULT/'trials.csv');d=pd.read_csv(RESULT/'development_by_year.csv');comparison=pd.read_csv(RESULT/'baseline_bridge.csv');preservation=dict(verified=v['protected_files_unchanged'],read_denied=v['protected_files_read_denied'])
    else:summary,t,d,comparison,preservation=evidence()
    (ROOT/'reports').mkdir(exist_ok=True)
    en,cn=documents(summary,t,d,comparison,preservation)
    if render:
        assert (ART/'pdf_operation_started.json').exists()
        from .pdf_layout import render as pdf
        for language,name,pages in [('en','Research_Report_V2_EN',en),('cn','PERFORMANCE_DELTA_CN',cn)]:pdf(pages,ROOT/f'reports/{name}.pdf',ROOT/f'reports/{name}.md',language)
    (ROOT/'README.md').write_text(f'''# Performance V2 - budget-expired partial delivery

**Qidian (Martin) Xing. Not a completed performance upgrade.** No new final pipeline or historical batch was produced, and no GitHub branch was pushed.

The run completed {summary['completed_screen_trials']} three-year screening trials ({summary['effective_distinct_screen_recipes']} effective distinct recipes), {summary['completed_refinement_trials']} local-refinement trials and {summary['bridge_trials']} single-year bridge controls, with {summary['recorded_fit_calls']} recorded fit calls. The strongest completed development baseline remains V1/P2 RF (annual AP 0.418); raw MLP recorded 0.382. These are single-seed development results, not new 2015-2018 performance.

- [Partial English report](reports/Research_Report_V2_EN.pdf), [partial Chinese report](reports/PERFORMANCE_DELTA_CN.pdf), [interview addendum](reports/Interview_Addendum_V2_CN.md).
- [Same-development-protocol comparison](results/baseline_bridge.csv), [all logical trials](results/trials.csv), [interrupted attempts](results/trial_attempts.csv), [fold results](results/development_by_year.csv), [target shortfalls](results/search_budget_summary.csv).
- [Development arithmetic verification](results/partial_results_verification.json), [run status](results/experiment_summary.json), [model-artifact status](results/model_artifacts_status.json), [execution deviations](PROTOCOL_EXECUTION_NOTES.md).

Empty confirmation/historical/seed/interval CSV files explicitly mean **not run**. They are not zero-valued results. Local ignored artifacts retain development predictions, curves and configurations. The copied legacy checkpoint is an original artifact; no newly trained V2 final model was saved. Full numerical-embedding training and tuned TabM comparisons remain unfinished; software contract tests do not substitute for those experiments.

Original deadline: {summary['deadline_utc']}. No recorded fit started after it. UTC gaps are not diagnosed as training time; completed-fold fit timing excludes interrupted work. Closeout documentation was prepared after the window expired. The original clock must not be reset; a fresh compute window requires explicit owner authorization. See the private NEXT_ACTIONS.md and state file for outstanding steps. No replication was launched automatically.

Source/code and [contribution disclosure](CONTRIBUTIONS.md) remain available. The [original team acknowledgements](../ACKNOWLEDGEMENTS.md) are preserved. Private data, models and company predictions are excluded from publication; publication remains blocked on the missing final experiments and their checks.
''')
    interview='\n\n'.join('# '+p['title']+'\n\n'+'\n\n'.join(b['text'] for b in p['blocks'] if b['type']=='paragraph') for p in cn[1:]);(ROOT/'reports/Interview_Addendum_V2_CN.md').write_text(interview)
    n=summary['effective_distinct_screen_recipes']
    (ROOT/'reports/Resume_Bullets_V2_EN_CN.md').write_text(f'''# Private resume drafts - partial scope only

Qidian (Martin) Xing. Owner review required; Codex-assisted work. Original team attribution remains intact. No performance improvement or completed final validation is claimed.

## English

- Benchmarked {n} distinct financial-distress recipes across three temporal development years, retaining failed and interrupted trials; reproduced a 192-by-4 residual architecture and distinguished development evidence from previously inspected historical results.
- Compared legacy residual and TabM training with tree baselines, documenting sampling exposures, numerical preprocessing and optimization limits; preserved original team results and recorded unfinished multi-seed validation without unsupported performance claims.

## 中文

- 比较{n}个有效不同配方，覆盖三个时间开发年，保留中断与失败记录；迁移并核对192×4残差架构，明确区分开发证据和已暴露历史期成绩。
- 对照旧Residual、TabM与树基准，记录采样曝光、数值处理和优化限制；保留原团队来源，明确多seed最终验证尚未完成，不声称性能提升。
''')
    (ROOT/'NEXT_ACTIONS.md').write_text('# Budget expired - no active training\n\n'+ '\n'.join('- '+x for x in MISSING)+'\n\nThe original eight-hour clock must not be reset. The archived command is scripts/resume_performance_v2.sh --phase all; it is not authorization to run after expiry. A new bounded continuation/replication requires fresh owner authorization and registration while preserving this evidence. GitHub publication is blocked on final experiments, independent numerical/visual checks, live authenticated-owner verification and privacy review.\n')
    dump(RESULT/'publication_blocked.json',dict(pushed=False,reason='Final experiments and independent final-result verification incomplete',identity_live_recheck='Not performed for an ineligible release',private_data_uploaded=False,missing=MISSING))
    claims()
    state('closed_budget_expired_partial',complete=False,local_partial_delivery_complete=False,partial_reports_rendered=render,training_running=False,running_pid=None,active_command=None,current_trial=None,current_fit_tag=None,model_fit_calls_after_deadline=0,publication='NOT_PUSHED_FINAL_RESULTS_INCOMPLETE')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--render',action='store_true');run(parser.parse_args().render)
