"""Editable report narratives; every displayed result is injected from measured tables."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .reporting import DISPLAY,fmt,p,fig,code,table
from .common import ROOT,ART

def page(title,*blocks,subtitle=None):return dict(title=title,subtitle=subtitle,blocks=list(blocks))

def core_comparison(c):
    d=c['development_summary'];d=d[d.stage=='core_recipe']
    return d.pivot(index='family',columns='feature_set',values='macro_year_AP').reset_index()

def representative_rows(c):
    return pd.DataFrame([dict(representative=k,**v.to_dict()) for k,v in c['representative'].items()])

def ci_rows(c,metric='AP'):
    d=c['paired_company_bootstrap']
    labels=[v['label'] for v in c['selected'].values()]
    return d[(d.model.isin(labels))&(d.metric==metric)].assign(model=lambda x:x.model.map({v['label']:k for k,v in c['selected'].items()}))

def ablation_wide(c):
    a=c['ablations']
    return a.pivot(index='recipe',columns='family',values='macro_year_AP').reset_index()

def calibration_comparison(c):
    rows=[]
    for kind,item in c['selected'].items():
        for role,source in [('2014 decision',c['pre_evaluation_metrics'][c['pre_evaluation_metrics'].role=='threshold_selection']),('2015-2018',c['historical_metrics'])]:
            data=source[source.label==item['label']].set_index('score_type')
            rows.append(dict(representative=kind,period=role,raw_Brier=data.loc['uncalibrated','Brier'],cal_Brier=data.loc['calibrated','Brier'],raw_logloss=data.loc['uncalibrated','log_loss'],cal_logloss=data.loc['calibrated','log_loss']))
    return pd.DataFrame(rows)

def novelty_with_intervals(c):
    labels=[x['label'] for x in c['selected'].values()]
    points=c['historical_seen_unseen']
    points=points[points.label.isin(labels)&(points.seen_cutoff==2012)]
    ci=c['seen_unseen_company_bootstrap']
    ci=ci[(ci.metric=='AP')&(ci.seen_cutoff==2012)][['model','novelty','lower','upper']]
    return points.merge(ci,left_on=['label','novelty'],right_on=['model','novelty'],validate='one_to_one')

def measured_interpretation(c):
    core=core_comparison(c)
    learned=core[~core.family.isin(['dummy','altman'])]
    h1='; '.join(f'{DISPLAY[r.family]} {r.S1-r.S0:+.3f}' for r in learned.itertuples())
    a=c['ablations'];dynamics=a[a.recipe=='S2 dynamics']
    h2='; '.join(f'{DISPLAY[r.family]} {r.delta_vs_S1:+.3f}' for r in dynamics.itertuples())
    intervention=a[~a.recipe.isin(['S0 raw','S1 base','S2 dynamics'])]
    best=intervention.loc[intervention.delta_vs_S1.idxmax()]
    worst=intervention.loc[intervention.delta_vs_S1.idxmin()]
    negative='did not improve the base recipe' if worst.delta_vs_S1<=0 else 'had the smallest measured gain'
    negative_cn='未改善基准' if worst.delta_vs_S1<=0 else '增量最小'
    intervals=[];intervals_cn=[]
    for r in ci_rows(c).itertuples():
        conclusion='is above zero' if r.delta_lower>0 else ('is below zero' if r.delta_upper<0 else 'crosses zero')
        conclusion_cn='整体高于零' if r.delta_lower>0 else ('整体低于零' if r.delta_upper<0 else '跨过零')
        intervals.append(f'{r.model} delta AP [{r.delta_lower:+.3f}, {r.delta_upper:+.3f}] {conclusion}')
        intervals_cn.append(f'{r.model} 的 AP 差区间 [{r.delta_lower:+.3f}, {r.delta_upper:+.3f}] {conclusion_cn}')
    return dict(h1='Measured S1-minus-S0 development AP changes: '+h1+'.',h2='Measured S2-minus-S1 development AP changes: '+h2+'.',h4=f'The largest intervention delta was {DISPLAY[best.family]} / {best.recipe} ({best.delta_vs_S1:+.3f} AP). {DISPLAY[worst.family]} / {worst.recipe} {negative} ({worst.delta_vs_S1:+.3f}). These selected extrema are descriptive development observations, not unbiased final estimates.',h4_cn=f'受控干预中，开发增量最大的是 {DISPLAY[best.family]} / {best.recipe}（AP 差 {best.delta_vs_S1:+.3f}）；{DISPLAY[worst.family]} / {worst.recipe} {negative_cn}（{worst.delta_vs_S1:+.3f}）。这些极值来自开发表的描述，不是独立无偏估计。',ci='; '.join(intervals)+'. All interval statements are conditional on the fitted models and current clustered sample.',ci_cn='；'.join(intervals_cn)+'。结论只限于当前已拟合模型与公司聚类样本，不能推导未来业务或因果效果。')

def english_pages(c):
    ml=c['representative']['ML'];dl=c['representative']['DL'];base=c['baseline'];r=representative_rows(c)
    core=core_comparison(c)
    h=c['historical'];main=h[((h.feature_set=='S1')&(~h.family.isin(['dummy','altman'])))|h.label.isin([base.label,c['selected']['ML']['label'],c['selected']['DL']['label']])|(h.family.isin(['dummy','altman'])&(h.feature_set=='S1'))|h.family.eq('equal_rank_ensemble')]
    main=main.drop_duplicates(['family','feature_set','config_id'])
    selected_ids=set(c['freeze']['representatives'].values())
    main=main.assign(pipeline=[DISPLAY[x.family]+(' (ensemble)' if x.family=='equal_rank_ensemble' else (' (selected)' if x.config_id in selected_ids else ' (core)')) for x in main.itertuples()])
    novelty=novelty_with_intervals(c)
    seeds=c['seed_results'];seeds=seeds[seeds.config_id.isin(c['freeze']['representatives'].values())].groupby(['family','feature_set']).agg(seeds=('seed','nunique'),AP_mean=('AP','mean'),AP_min=('AP','min'),AP_max=('AP','max'),ROC_mean=('ROC_AUC','mean')).reset_index()
    stress=c['group_stress_metrics'].groupby('family').agg(groups=('group_fold','nunique'),AP_min=('AP','min'),AP_mean=('AP','mean'),AP_max=('AP','max'),ROC_mean=('ROC_AUC','mean')).reset_index()
    seq=c['development_summary'];seq=seq[seq.stage.isin(['sequence_control','sequence_extension'])]
    stable=c['feature_selection_stability'].groupby(['family','selector']).agg(mean_Jaccard=('mean_pairwise_Jaccard','first'),candidate_union=('feature','size')).reset_index()
    interpretation=measured_interpretation(c)
    importance=c['group_permutation_importance'].sort_values(['representative','mean_AP_drop'],ascending=[True,False])
    costs=h[(h.feature_set=='S1')&(~h.family.isin(['dummy','altman']))].drop_duplicates('family')
    return [
      page('Corporate Financial Distress',
        p('A Reproducible ML & Deep Learning Benchmark - Independent Extension'),
        p('Qidian (Martin) Xing | Research date: '+c['date']),
        p(f'This study compares financial features, traditional estimators and compact PyTorch networks on {c["audit"]["rows"]:,} firm-year records. The target is the observed company-constant failed label. It is retrospective classification, not verified one-year-ahead bankruptcy prediction.'),
        p(f'Models selected using seven development scoring years achieved previously inspected historical AP of {ml.AP:.3f} ({DISPLAY[ml.family]}, {ml.feature_set}) and {dl.AP:.3f} ({DISPLAY[dl.family]}, {dl.feature_set}), compared with {base.AP:.3f} for the same-protocol S0 Logistic baseline. These numbers describe one frozen seed-42 run; seed variability and paired company intervals are reported separately.'),
        p('The main contribution is an auditable comparison: fold-local preprocessing and selection, separate calibration and decision years, budgeted screening, same-row feature/sampling ablations, and time-constrained company-disjoint stress tests. The earlier team project already compared ML and neural models. This extension does not claim to introduce those methods.'),
        p('Reading map: data/labels (2), validation (3), information sets (4), model training (5), development (6), ablations (7), historical ranking (8), decisions/calibration (9), company generalization (10), uncertainty (11), sequences (12), interpretation (13), reproducibility and limits (14).'),
        p('Scope: research and risk-ranking demonstration. Historical test exposure, unknown event/filing dates, changing class prevalence and repeated company observations constrain every performance claim.')),
      page('2. Data identity and label limits',
        table(c['split_audit'].iloc[:4],['role','rows','companies','positive_rows','prevalence'],['Role','Rows','Companies','Positive rows','Fraction']),
        p('The original CSV copies have identical SHA-256 hashes. The audit found no missing raw cells or duplicate company-years. X9 (net sales) and X16 (revenue) are identical, so the main raw information set contains 17 variables. Negative net income, EBIT and retained earnings are economically meaningful; they are not deleted.'),
        fig('prevalence','Read the vertical scale as the fraction of rows assigned the failed-company label, not an annual bankruptcy hazard. The shaded years are the previously inspected historical evaluation.'),
        p('Labels are constant within all 8,971 companies. True bankruptcy dates, financial-statement publication dates and when the label became knowable are missing. Restricting input fiscal years does not establish point-in-time label availability. The last observed year is never treated as a failure date.'),
        p('Live upstream comparison aligns IDs, years and labels but finds different financial values and extra sector columns. The local snapshot remains fixed. Units, currency, restatement treatment and exact upstream-version relationship remain unresolved. Dataset acquisition/rights checks are documented; no raw data is redistributed.')),
      page('3. Temporal validation and isolation',
        {'type':'table','header':['Fold / role','Fit years','Scoring / function'],'rows':[['D1','1999-2005','2006-2007'],['D2','1999-2007','2008-2009'],['D3','1999-2009','2010-2011'],['D4','1999-2011','2012'],['Final model','1999-2012','No later labels in base fit'],['Calibration','2013','Natural-prevalence Platt fit'],['Decision','2014','Freeze one F1 threshold'],['Historical','2015-2018','Previously inspected evaluation']]},
        p('The selection objective is the equal-weight mean of AP across the seven development scoring years. Each calendar year has equal influence despite different sample sizes. Secondary diagnostics include annual recall at Top 5%, ROC-AUC, worst-year AP and cost. Development selection scores are optimistic estimates of the selection process, not independent final evidence.'),
        p('For neural networks and boosting, the last year inside the current training window is the stopping set. A preliminary model determines training length; a new preprocessor and model are then fitted to the complete training window at that fixed length. The outer scoring labels do not determine epochs, imputation values, quantiles, scaling or selected features.'),
        p('Development loading never parses labels beyond 2012. Calibration and threshold roles become available only to final-artifact training. The historical command verifies protocol, artifact and source hashes before opening historical labels once. Reproduction subsequently reloads saved artifacts and checks predictions; it does not retune.'),
        p('The historical period was exposed in the original project and cannot be made untouched through a new split or company hash. A substantive scoring bug would require an explicit invalidation/amendment and a new run, preserving prior evidence.')),
      page('4. Registered financial information sets',
        {'type':'table','header':['Set','Information','Causality / invalid rules'],'rows':[['S0','17 deduplicated financial amounts','Current fiscal record only'],['S1','S0, signed logs, 12 unique ratios, working capital, Altman composite and invalid flags','Current positive denominators; NaN plus masks otherwise'],['S2','S1 plus exact annual changes, three-calendar-year summaries, positive log growth, losses and coverage','Only t, t-1 and t-2; gaps remain gaps'],['LAGS','Three exact calendar snapshots with observed flags','Same target rows, explicit missing slots']]},
        p('Signed log: sign(x) * log(1 + abs(x)). For example, a loss remains negative while its magnitude is compressed. Ratios express liquidity, profitability, total-liability leverage, operating efficiency and market-value coverage. X17/assets is total liabilities/assets, not exclusively interest-bearing debt. Denominators are current-year values rather than average balances.'),
        p('Nonpositive financial ratio denominators create NaN and an invalid flag. Tiny positive values are retained for transparent train-only clipping rather than replaced by arbitrary constants. Median imputation, missing masks, optional 0.5/99.5-percentile clipping, and scaling are fitted only to the training role. A no-clipping ablation tests whether this protection is beneficial.'),
        p('Dynamic features use exact calendar joins. If t-1 is absent, a t-2 row cannot become a one-year change. Three-year summaries use the observed slots and report coverage. No backward fill, centered window, future record count, final year, remaining lifetime, ID suffix or label enters a model tensor.'),
        p('Every feature has a formula, source columns, economic group, timing and invalid-value rule in feature_registry.csv. Constant and exact duplicate transformed columns are removed using training data only. Additional columns are justified by controlled information-set comparisons, not by feature count.')),
      page('5. Models, optimization and actual cost',
        table(costs,['family','features','parameters','epochs','fit_seconds'],['Family','Input dims','NN / linear params','Fixed epochs / trees','Fit seconds']),
        p('Baselines are training prevalence and the negative Altman Z-score. Learned models comprise L2 Logistic, Random Forest, XGBoost, regularized MLP, Residual MLP, compact FT-Transformer and available CatBoost. Elastic Net is tested as a nested embedded selector. No network is presumed superior: deep learning is a subset of machine learning.'),
        p('MLP uses two dense blocks. Residual MLP adds normalized skip blocks. FT-Transformer learns a weight and bias token for each numeric feature, adds a CLS token and applies two small attention layers. It is an independent educational implementation inspired by the tabular-model paper, not a reproduction of the paper benchmark. Attention acts across features; it does not create a time series.'),
        p('PyTorch training uses AdamW, dropout, weight decay, shuffled deterministic DataLoaders, BCEWithLogitsLoss, finite-gradient checks and gradient clipping. Sigmoid is applied for probabilities after logits. The networks use train mode for updates and eval/inference mode for scoring; round-trip checkpoint tests verify prediction consistency.'),
        p('All measured runs use local CPU, two threads and one neural experiment process. A first XGBoost pilot crashed because two OpenMP runtimes were loaded; sharing the PyTorch runtime resolved the conflict. Exact KD-tree neighbors avoid a separate SMOTE dispatcher crash. Failed diagnostics remain logged. No paid service or system-Python modification was used.')),
      page('6. Development comparison and bounded search',
        table(core,['family','S0','S1'],['Family','S0 macro-year AP','S1 macro-year AP']),
        p(interpretation['h1']),
        fig('development_features','Bars compare the fixed recipes on the same records and seven development scoring years. Higher AP is better, but these are development-selected observations.'),
        p('The first stage runs every required family on S0 and S1 with seed 42. The best S1 ML and DL family each receive only two pre-registered capacity/regularization candidates. The second stage is deliberately small; it is not an exhaustive architecture search or an equal-compute claim for all model families.'),
        p(f'The final ML representative is {DISPLAY[ml.family]} on {ml.feature_set}; the DL representative is {DISPLAY[dl.family]} on {dl.feature_set}. Their configurations were chosen from complete development runs before historical evaluation, using macro-year AP and registered tie rules. The complete candidate table includes weak and failed-to-improve alternatives.'),
        p('A fixed equal-rank ML+DL candidate is assessed on matched OOF predictions only. It is admitted to the historical inventory only if its development macro AP exceeds both constituents; no historical metric controls that decision.')),
      page('7. Controlled feature and imbalance ablations',
        table(ablation_wide(c),['recipe','xgboost','residual_mlp'],['Changed factor','XGBoost AP','Residual MLP AP']),
        p(interpretation['h2']+' '+interpretation['h4']),
        p('Every row uses the same four expanding folds, natural scoring proportions and seed 42. S0/S1/S2 compare information; other rows change one element of the S1 recipe. The full controlled table reports AP deltas, feature dimensions and training cost. The optimized final pipeline comparison is a separate question because its feature set or regularization may differ.'),
        p('Class weights derive from the current training role: (N_negative/N_positive)^alpha, with alpha 0, 0.5 or 1. ROS and SMOTE are separate from weighting. Both resample only training inputs after appropriate scaling. ROS controls for extra minority exposure; SMOTE adds interpolations, which need not preserve accounting relationships or represent new companies.'),
        p('Elastic Net chooses C and L1 ratio in an inner tail-year split, then selects nonzero coefficients after an outer-training refit. Group-aware selection ranks economic groups using two earlier cross-fitted tree validations, then chooses top 2, top 4 or all groups on a later inner year. Neither selector sees outer scoring labels. No selection remains a genuine candidate.'),
        p('A negative AP delta is retained as a failed improvement, not removed. Clipping, selecting variables, resampling and adding architecture are empirical choices with costs; none is assumed to improve every model or year.')),
      page('8. Frozen historical ranking',
        table(main,['pipeline','feature_set','AP','ROC_AUC','macro_recall_top5'],['Pipeline','Set','AP','ROC-AUC','Annual R@5%']),
        fig('ranking_curves','The PR and ROC curves use identical historical rows for the S0 Logistic baseline and development-selected representatives. AP is sklearn average_precision_score, not trapezoidal PR area.'),
        p(f'The evaluation contains {int(base.rows):,} records and {int(base.positive_rows)} positive rows, from {int(base.positive_companies)} positive companies. Its prevalence is {base.prevalence:.4f}. Pooled AP and macro-year AP answer different weighting questions; both are saved. Use same-row paired differences, not an AP ratio across years with different prevalence.'),
        p('Core S1 rows provide a controlled family comparison. Representative rows describe optimized pipelines, including their information set. Extra tuned configurations from the same family remain identifiable by config_id in the machine tables. Old team scores follow different roles and are not used as an improvement denominator.')),
      page('9. Screening and probability quality',
        fig('screening_calibration','Left: average recall under fixed annual review capacities. Right: reliability bins. Sparse high-score bins require their counts and positive counts; a line alone is not evidence of calibration.'),
        table(r,['representative','precision','recall','F1','alert_fraction','Brier'],['Rep.','Precision','Recall','F1','Alert share','Brier']),
        table(calibration_comparison(c).query('period == "2015-2018"'),['representative','raw_Brier','cal_Brier','raw_logloss','cal_logloss'],['Historical','Raw Brier','Platt Brier','Raw log loss','Platt log loss']),
        p('Annual Top 1%, 5% and 10% use score-only ordering, ceil(budget*n) and a deterministic company/year hash for ties. The capacity is fixed before looking at outcomes. Lift divides selected precision by the same year\'s positive fraction. It is a risk-screening statistic, not a realized financial return.'),
        p('One fixed threshold maximizes F1 on 2014 calibrated predictions. Historical precision, recall, F1 and confusion counts use that frozen threshold. A large change in the historical alarm fraction is possible under distribution shift; threshold metrics must be interpreted alongside review budgets.'),
        p('Platt calibration fits only on natural-prevalence 2013 data, without refitting the base model afterward. Uncalibrated scores, 2014 probability quality, historical Brier/log loss and reliability-bin counts are preserved. Brier and log loss assess probability prediction quality, not pure calibration error. A calibrated failed-label probability is still not an annual default PD.')),
      page('10. Time and company generalization',
        fig('annual_performance','Yearly AP and annual recall at Top 5% reveal changing performance. Only four historical years are available, so no strong time-series significance claim follows.'),
        table(novelty,['family','novelty','companies','positive_companies','AP','lower','upper'],['Family','Seen by 2012','Companies','Failed firms','AP','CI low','CI high']),
        table(stress,['family','groups','AP_min','AP_mean','AP_max'],['Untuned stress','Groups','Min AP','Mean AP','Max AP']),
        p('Seen/unseen is relative to the actual 2012 base-fit cutoff. The unseen subset contains only 16 failed companies; its score can be unstable even with many rows. A separate 2014-cutoff sensitivity includes an even smaller positive cohort and is not silently omitted.'),
        p('Company stress holds out each of three fixed SHA256 groups from all fitting, stopping and calibration roles. Logistic, Boosting and Residual MLP use the recipes fixed before development, not globally tuned winners. This is explicitly an untuned sensitivity analysis. Its training population is smaller, so direct differences from full-company models mix training coverage and novelty.')),
      page('11. Uncertainty and seed sensitivity',
        table(ci_rows(c),['model','lower','upper','delta_lower','delta_upper','valid_draws'],['Representative','AP CI low','AP CI high','Delta low','Delta high','Draws']),
        table(seeds,['family','feature_set','seeds','AP_mean','AP_min','AP_max'],['Family','Set','Seeds','Mean AP','Min AP','Max AP']),
        p(interpretation['ci']),
        p('Paired cluster bootstrap samples companies with replacement, carrying all their annual records together. All models share each draw. Integer company multiplicities preserve repeated observations and annual capacities; 2,000 draws produce percentile intervals. Single-class invalid draws are counted rather than assigned artificial scores.'),
        p('Delta intervals compare each development-selected representative with the same-protocol S0 Logistic baseline. An interval crossing zero does not support a confident positive difference under this procedure. An interval wholly above zero is still conditional on these fitted models, this historical population and the cluster scheme; it does not eliminate test exposure or model-selection uncertainty.'),
        p('The five registered seeds are all retained. Their mean and range describe stochastic training variation on the same evaluation population, not five independent tests. Seed 42 is the preselected main run; no best historical seed is substituted. Seed averaging is not represented as a single model or as independent data replication.'),
        p('Company bootstrap does not fully represent shared macroeconomic shocks, upstream data revisions, point-in-time label availability or future-regime uncertainty. Four historical years provide descriptive temporal evidence rather than broad coverage of economic states.')),
      page('12. Annual trajectories and finite ensemble',
        table(seq,['family','feature_set','macro_year_AP','mean_fit_seconds'],['Optional family','History set','Development AP','Mean fit seconds']),
        table(pd.DataFrame(list(c['ensemble_choice']['macro_year_AP'].items()),columns=['candidate','AP']),['candidate','AP'],['Fixed-rank OOF candidate','Development AP']),
        p('The GRU receives [batch, 3 calendar slots, feature] rather than treating financial columns as time. A shared snapshot preprocessor is fitted only on current training-role records. For missing calendar slots, numerical inputs are zeroed and an observed flag is zero; target-year rows remain present. The recurrent output at the target year feeds a logit head.'),
        p('Snapshot S0 tree/MLP, flattened three-year tree/MLP and GRU are evaluated on identical target rows. This separates the availability of historical inputs from a recurrent architecture. The flattened controls may learn lag-specific transforms; the GRU shares feature scaling over time. They compare practical pipelines on the same information, not perfectly identical parameterizations.'),
        p('The optional window is fixed at three years. No ordinary SMOTE is applied to sequence tensors. Short histories and internal gaps are retained rather than filtering failure companies away. Calendar masks carry coverage information and can themselves reflect selection processes; their presence is disclosed.'),
        p('Sequence modeling does not supply missing event dates. All outputs predict the same retrospective ultimate-status label. Any improvement is evidence about information/architecture under this dataset and protocol, not an established advance in one-year event forecasting.'),
        p('The equal-rank candidate uses one half ML and one half DL OOF rank per calendar year. It requires an annual scoring cohort, and any accepted ensemble has its own disjoint 2013 calibration and 2014 threshold. Rejected candidates remain in ensemble_development.csv; there is no stacking on in-sample predictions.')),
      page('13. Interpretation, stability and errors',
        fig('group_importance','Permutation uses the exact frozen scoring checkpoints and preprocessing. Related variables are shuffled as economic groups; three repeat drops show descriptive sensitivity, not causal effects.'),
        table(stable,['family','selector','mean_Jaccard','candidate_union'],['Family','Selector','Mean Jaccard','Union size']),
        p('Linear coefficients are conditional on other correlated inputs and their standardization. Selector coefficients and per-fold feature lists are exported. Jaccard overlap measures descriptive selection stability across time windows; it is not a significance test. A sparse linear selector can discard variables useful in nonlinear interactions.'),
        p('Failure cases follow a registered rule: the smallest SHA256 within each TP, FP, FN and TN cell. IDs, financial context and predictions remain in the local case file. Public profiles report cell counts and medians only. This avoids choosing only persuasive anecdotes and avoids publishing the full row-prediction dataset.'),
        p('False positives may resemble financially weak ultimate survivors; false negatives may have financial patterns similar to ultimate survivors or incomplete temporal information. These are possible interpretations, not verified causal histories. Neither attention weights nor permutation establishes why a company failed. All case analysis occurs after freeze and cannot feed back into this version.')),
      page('14. Reproducibility, attribution and next work',
        code('scripts/run.sh -m pytest -q\nscripts/run.sh -m distress_ml.cli audit\nscripts/run.sh -m distress_ml.cli develop --stage all\nscripts/run.sh -m distress_ml.cli sequences\nscripts/run.sh -m distress_ml.cli freeze\nscripts/run.sh -m distress_ml.cli historical-evaluate\nscripts/run.sh -m distress_ml.cli report'),
        p('To reproduce an existing artifact, run the reproduce command with its final model label; this reloads the checkpoint and checks scores against saved local predictions. predict accepts a raw input CSV, with company/year history when required. Full retraining requires the exact authorized data hash, the pinned environment and a fresh run directory; the historical lock is intentionally not bypassed.'),
        p('The local artifact package stores preprocessing, feature order, model state, calibration, threshold, budget rules, configuration, environment, hashes, train histories and row predictions. Public release includes only new implementation, aggregate results, sanitized notebooks and reports. Source/processed data, original team materials, private communications, credentials and local resume drafts are excluded. No open-source license is inferred.'),
        p('The extension is authored under Qidian (Martin) Xing\'s direction with disclosed Codex assistance. The earlier UCLA MFE 413 course submission remains credited to Gus Guenther, Zhixiang (Eason) Chen, Yingqi Fang, Qidian (Martin) Xing, Yiyang (Faye) Xie, Xiaoyue Wang and Xianghan Zhu. No claim is made that the owner manually wrote every line or has already independently reviewed the outputs.'),
        p('Next research should obtain auditable event and statement-availability dates, verify source units/version history, register a genuine prospective or external holdout, and test calibration and review capacity under realistic operational constraints. New historical insights belong to a new registered version, not a retuned result in this report.'),
        p('Primary references: scikit-learn common pitfalls, AP and calibration documentation; imbalanced-learn resampling pitfalls; Gorishniy et al. (2021), arXiv:2106.11959, and the authors\' official rtdl-revisiting-models repository; sowide/bankruptcy_dataset README/license and requested data papers. Exact checked links appear in docs/REFERENCES.md.'))
    ]

def chinese_pages(c):
    ml=c['representative']['ML'];dl=c['representative']['DL'];base=c['baseline'];r=representative_rows(c)
    ab=ablation_wide(c);core=core_comparison(c)
    chosen=c['selected']['DL']['label']
    profiles=c['confusion_cell_profiles'];profiles=profiles[profiles.representative=='DL']
    seeds=c['seed_results'];seeds=seeds[seeds.config_id.isin(c['freeze']['representatives'].values())].groupby('family').agg(seeds=('seed','nunique'),AP_mean=('AP','mean'),AP_min=('AP','min'),AP_max=('AP','max')).reset_index()
    sequence=c['development_summary'];sequence=sequence[sequence.stage.isin(['sequence_control','sequence_extension'])]
    novelty=novelty_with_intervals(c)
    interpretation=measured_interpretation(c)
    pages=[
      page('从财务记录到可核查的研究',
        p('企业财务困境个人扩展：中文教学与面试手册'),
        p('Qidian (Martin) Xing | '+c['date']),
        p('研究主线：先确认标签到底是什么，再限制模型能看到的信息；用统一时间协议比较财务特征、树模型与 PyTorch 网络；最后检查少数类筛查、概率质量、跨年变化和陌生公司表现。模型复杂度只是候选设计，可信的证据链才是主线。'),
        p(f'本轮实测：开发期选出的 ML 代表为 {DISPLAY[ml.family]}（{ml.feature_set}），历史 AP 为 {ml.AP:.3f}；DL 代表为 {DISPLAY[dl.family]}（{dl.feature_set}），历史 AP 为 {dl.AP:.3f}。同协议 S0 Logistic 的 AP 为 {base.AP:.3f}。这是已被查看过的历史期，不是新外部测试；五种子和公司区间见后文。'),
        {'type':'table','header':['阅读路线','核心问题'],'rows':[['2-5 数据与验证','一行是什么？哪些信息不能用？'],['6-8 特征','金额、比率、稳健变换与历史变化'],['9-15 指标与模型','怎样读分数？ML / DL / GRU 的关系'],['16-20 取舍与结果','采样、校准、完整流程与不确定性'],['21-22 表达与边界','开场、深入版、贡献与自查'],['23-26 追问','24 个问题及实际代码定位']]},
        p('读法：先看每章要解决的问题，再对照公式、小例子和本项目的真实表。示意例子会明确标注，不会冒充实测。技术定位均为仓库相对路径；本地路径不写入公开报告。')),
      page('2. firm-year 与标签：先把题目说对',
        p('firm-year 是一家公司某个会计年度的一条财务记录。同一家公司连续五年出现，会形成五行；它们不是五家不同公司，也不是五次独立破产事件。这会影响划分、置信区间和面试中的样本量说法。'),
        p(f'真实数据共有 {c["audit"]["rows"]:,} 行、{c["audit"]["companies"]:,} 家公司、{c["audit"]["company_label_counts"]["failed"]} 家 failed 公司。正类行有 {c["audit"]["row_label_counts"]["failed"]:,} 条。同一公司所有年份的标签都相同，因此行数多于真正独立的公司数。'),
        p('示意：某公司 2001、2002、2003 年均标 failed，只能说明它在该数据版本中被归为 failed 公司。不能据此说 2001 年输入是在预测 2002 年破产。我们没有逐公司的真实破产日期，也没有财报公开日期或明确预测窗口。'),
        p('为什么不把最后一年加一当事件日？公司停止出现可能来自样本截止、退市、缺报或覆盖改变。这样造标签会把资料缺失变成结果定义，模型可能只是识别数据收集规则。任意增加 180 天 lag 也不能补出真实发布日期。'),
        p('本项目能说：利用历史财务特征对数据中最终状态为 failed 的公司年度记录做回溯识别与风险排序。不能说：建立了一年期违约概率模型，或验证了真实业务上线时的提前预警能力。校准只改变概率刻度，不会改变标签的含义。'),
        code('src/distress_ml/data.py: load_data\ndocs/DATA_CARD.md\nresults/public/data_audit.json')),
      page('3. 源审计为什么是研究的一部分',
        p('两个原始 CSV 的 SHA-256 相同，表示本轮从同一字节版本开始。原文件没有缺失和重复公司年度行，但这不等于所有新特征都无缺失，也不等于数据已经满足前瞻预测要求。'),
        p('字段含义必须使用原有字典。例如 X10 是 Total assets，X17 是 Total liabilities，X8 是 Market value。总负债包含的内容比“有息债务”广，不能为了金融措辞简洁而改写含义。单位、币种和重述处理仍未独立证实。'),
        p('X9 与 X16 在全体原始记录中完全一致。S0 去掉 X16，避免同一收入信息重复进入模型。盈利为负、留存收益为负通常有经济意义，不能见到负数就删除公司。零存货也可能反映业务结构，而非自动等于缺失。'),
        p('会计关系也要算残差，而不是假定等式永远成立。accounting_relationships.csv 检查 sales/revenue、gross profit 与 sales-COGS、EBIT 与 EBITDA-depreciation 的一致性。口径、四舍五入或字段加工都可能造成差异；不符合关系的数值不会被自动改掉。'),
        p('当前上游公开 CSV 与本地公司、年份、标签逐行对应，但财务值不完全相同，还多出行业字段。上游 README 的公司数和事件标签描述也不能覆盖实际 CSV 证据。本轮没有下载替换训练数据，也没有把新增行业字段偷偷加入模型。'),
        p('原始材料全部保留；哈希清单、归档成员目录和原报告封面核对留在本地。公开仓库仅放新代码与聚合输出。检查入口：scripts/audit_sources.py、src/distress_ml/audit.py、docs/SOURCE_AUDIT.md。')),
      page('4. 时间角色：谁负责学，谁负责选',
        {'type':'table','header':['角色','年份','负责的问题'],'rows':[['开发 D1-D4','1999 起逐次扩展','模型设置在后续年份能否工作？'],['最终拟合','1999-2012','确定基础模型与预处理参数'],['校准','2013','分数与标签比例如何对应？'],['决策','2014','固定 F1 阈值放在哪里？'],['历史评估','2015-2018','冻结后的表现与不确定性']]},
        p('开发四折按年份块划分：训练至 2005 看 2006-2007；至 2007 看 2008-2009；至 2009 看 2010-2011；至 2011 看 2012。先按公司排序再直接套普通 TimeSeriesSplit，可能把公司顺序当时间顺序，因此这里不用这种做法。'),
        p('主目标是七个评分年份 AP 的等权平均。理由是每年代表一个不同的时段，不能让记录更多的年份自动支配选择。平均 AP 仍然会受每年正类比例影响，所以逐年表同时保留 prevalence、ROC 和预算召回。'),
        p('2013 不参与基础模型拟合，否则校准器面对的分数分布会改变。2014 用来选阈值，因此其 F1 不是独立泛化成绩。2015-2018 是原团队看过的历史期，冻结后评估可以增加严谨性，却不能消除历史暴露。'),
        p('技术定位：splits.py 的 DEVELOPMENT 与 inner_masks；data.py 对 development、pre_evaluation、historical 的标签读取边界；finalize.py 的 verify_freeze 和 historical_evaluate。')),
      page('5. 泄漏、嵌套选择与 early stopping',
        p('泄漏的关键不是代码里有没有 test 这个词，而是某项学习是否使用了不该知道的结果或分布。例如先用全样本挑特征再分折，评分标签已经影响了输入名单；先对全部公司 SMOTE 再划分，则评分行可能参与生成训练点。'),
        p('本项目每折重新学 median、clipping quantiles、scaling、常数/重复检查及 selector。训练完只调用 transform；评分数据不会更新这些状态。合成数据只在训练角色生成，校准、阈值和历史评估保持真实类别比例。'),
        p('early stopping 也属于选择。若每轮查看外层评分期并选最好一轮，再报告同一评分成绩，成绩会受轮数选择影响。本实现先在训练窗口内部拿尾部年份选轮数，再重建预处理和网络，用完整训练窗口训练固定轮数，最后才看外层评分期。'),
        p('Elastic Net 的 C 与 L1 ratio 在内层时间切分中选；group permutation 的重要性与 top-k 也只在外层训练范围内产生。外层评分标签负责评价整个选择过程，而不是再训练该选择器。内层表现可以被选优影响，外层仍有明确边界。'),
        p('如何主动找 bug？把未来财务值改成极大值，检查过去动态特征与训练预处理是否不变；删去一年的记录，检查 t-1 是否变成缺失；改变评分标签，确认训练特征名单不受影响。测试不能证明所有研究结论正确，但能击中最危险的数据流错误。'),
        code('tests/test_protocol.py\npreprocessing.py: Preprocessor\nselection.py: select_features\ntrain.py: fit_pipeline')),
      page('6. 金额与比率：经济含义先于列数',
        p('金额回答“规模多大”，比率回答“结构怎样”。示意：A 公司总负债 80、资产 100，B 公司总负债 800、资产 2000。B 的负债金额更大，但总负债率为 0.4，低于 A 的 0.8。若只看原始金额，规模可能遮住结构信息。这个例子是教学示意，不是样本行。'),
        p('流动性：Current assets / Current liabilities；盈利：Net income / Total assets；杠杆：Total liabilities / Total assets；效率：Sales / Total assets；市场覆盖：Market value / Total liabilities。它们不是互不相关的独立解释，多个比率共享分子或分母，解释时需要考虑相关性。'),
        p('为什么要求分母为正？负收入的 margin 或非正资产的资产回报率，经济解释容易失真；零分母会产生无穷大。本实现生成 NaN 与异常标记，再由训练内 median 处理。极小正分母不会被一个任意大常数替换；其极端比率由训练分位点截尾对照检验。'),
        p('Altman 构成项把 working capital、retained earnings、EBIT、market value 和 sales 按固定系数组合。这里用 -Z 排序，分数越高代表越弱。它是可解释财务基准，但当前样本未验证制造业限制和完全一致的原论文适用条件，不应将其解释为经过行业校验的事件概率。'),
        p('S1 对 S0 的增量由同配方时间折实测检验。不能用“构造了很多变量”替代经济理由，也不能因为某比率单变量显著，就断言它在控制其他变量后仍然重要。'),
        code('features.py: RATIOS, ratio, feature_registry\nresults/public/feature_registry.csv')),
      page('7. 变换、缺失与截尾：保护也有代价',
        p('signed log = sign(x) × log(1+|x|)。示意：利润 -100 与 +100 变换后绝对值相同、符号相反。它压缩重尾金额，同时保留亏损信息。普通 log 对负数无定义；把所有负数裁为零又会丢掉亏损强弱。'),
        p('标准化为 (x-mean_train)/std_train，有助于 Logistic 的正则化、神经网络优化和基于欧氏距离的 SMOTE。树主要依靠排序和切分，不必强行标准化；但树若使用 SMOTE，其插值前仍需要合理尺度。这里允许各模型采用适合自身的预处理。'),
        p('截尾把训练分位点之外的极端值限制在边界内，减少巨额金额或近零分母对优化的支配。代价是可能压平真实财务危机信号，所以必须保留无截尾版本。边界只从当折训练样本计算，不能因为测试极端而重新设分位点。'),
        table(ab[ab.recipe.isin(['S1 base','No clipping'])],['recipe','xgboost','residual_mlp'],['受控设置','XGBoost 开发 AP','Residual MLP 开发 AP']),
        p('读表：两列各自在自己的模型内比较，只有截尾开关改变。若无截尾更差，说明该设定下截尾有帮助；若更好，也必须保留这一结果。不能把一个网络的选择直接套给所有树模型。'),
        p('实际输入维度由训练内去常数/完全重复检查决定，可能少于登记候选数。missing indicator 让模型区分“观测到 median”与“缺失后填 median”；它也可能学到覆盖机制，因此要如实披露。定位：preprocessing.py。')),
      page('8. 动态特征：上一条不一定是上一年',
        p('710 家公司存在年份不连续。示意：记录为 2001、2003、2004。2003 年的上一条是 2001，却不是 t-1；把两年变化命名为一年变化会引入定义错误。代码以 company + year 做精确查询，不以行位置代替日历。'),
        p('S2 包括盈利能力、总负债率、流动比率和资产周转的一年差；正资产和正收入的对数增长；t、t-1、t-2 的均值/波动；亏损次数与观测覆盖。它们回答“状态是否在恶化”，而单年快照主要回答“当前有多弱”。'),
        p('缺历史时保留目标行，变化设 NaN，附加 mask。三年均值使用实际可得的日历槽，覆盖数同时进入记录。不能为了得到整齐的三年张量删掉大部分短历史公司，再与全样本快照模型比较。'),
        table(ab[ab.recipe.isin(['S1 base','S2 dynamics'])],['recipe','xgboost','residual_mlp'],['信息集','XGBoost 开发 AP','Residual MLP 开发 AP']),
        p('读表时只把 S2-S1 的差异归因于新增历史信息及其缺失标记，不能单凭这一对照宣称 GRU 有效。真正比较序列架构，还需要相同历史窗口的 flattened tree / MLP 对照。'),
        code('features.py: lag_values, build_features\nsequences.py: sequence_tensor\ntests: calendar gap and future perturbation checks')),
      page('9. 指标：先理解漏报和误报',
        p(f'以开发期选出的 DL 代表的固定阈值为例，历史结果为 TP={int(dl.TP)}、FP={int(dl.FP)}、FN={int(dl.FN)}、TN={int(dl.TN)}。这些是真实计数，可从本地预测文件重算；阈值只由 2014 年选择。'),
        {'type':'table','header':['真实 / 判断','标为高风险','标为低风险'],'rows':[['failed=1',f'TP {int(dl.TP)}',f'FN {int(dl.FN)}'],['alive=0',f'FP {int(dl.FP)}',f'TN {int(dl.TN)}']]},
        p(f'Precision=TP/(TP+FP)，回答报警里有多少正类，本例为 {dl.precision:.3f}。Recall=TP/(TP+FN)，回答真实正类抓到多少，本例为 {dl.recall:.3f}。F1 是两者的调和平均，本例为 {dl.F1:.3f}。降低阈值通常增加召回，也会增加误报，不能单凭召回高就说模型更好。'),
        p('ROC-AUC 衡量一个随机正类排在随机负类前面的概率，关注整体排序。AP 汇总 precision 随 recall 增加的贡献，本项目严格用 sklearn.average_precision_score，不与梯形 PR 面积混用。正类稀少时，AP 往往比 accuracy 更贴近少数类筛查。'),
        p(f'历史正类比例只有 {base.prevalence:.4f}。全部预测 alive 可以得到很高 accuracy，却抓不到 failed。不同年份基准比例不同，因此 AP 不能脱离 prevalence 解释；同一批行上的模型差异最有可比性。'),
        p('Top 5% 是每年只看最高分约 5% 的固定工作量，回答有限审查能力下能抓到多少。它无需在测试标签上挑阈值。F1 阈值则是辅助决策，两者不是同一规则。定位：evaluation.py 的 metrics、budgets 与 choose_threshold。')),
      page('10. Logistic、Elastic Net 与筛选取舍',
        p('Logistic 先计算 logit = b + Σ w_j x_j，再用 sigmoid 把它映射到 0-1。它能表达特征的加总关系。若某比率只有在另一比率很高时才危险，纯线性加总未必容易表达这个交互，除非额外构造相应输入。'),
        p('L2 惩罚系数平方和，让系数更平滑；L1 惩罚绝对值和，可把部分系数压成零。Elastic Net 混合 L1 与 L2，在稀疏和相关变量稳定性之间折中。C 越小代表正则化通常越强，L1 ratio 决定两种惩罚比例。'),
        p('本项目先在训练窗口内部的尾部年份比较有限 C/L1 ratio，再在完整外层训练上重拟合选择器。非零系数决定候选名单，随后模型自己的预处理与训练仍重新进行。评分年份没有参与名单生成。'),
        p('为什么不预先固定 13 列？原项目的 13 列来自 L1 筛选器的 AUC 简洁性规则，并非所有后续模型的最佳名单。新的主目标是年度 AP，而且非线性模型可能利用线性筛选器忽略的交互信息，因此“不做学习式筛选”必须保留。'),
        table(ab[ab.recipe.isin(['S1 base','elastic_net','model_aware'])],['recipe','xgboost','residual_mlp'],['筛选策略','XGBoost 开发 AP','Residual MLP 开发 AP']),
        p('系数方向是控制其他输入后的条件关联。高度相关盈利指标可能分担或交换系数，不能将一个负系数解释成“增加该变量一定降低失败概率”。完整系数与跨折入选频率见 elastic_net_coefficients.csv 和 feature_selection_stability.csv。')),
      page('11. 树、Random Forest 与 Boosting',
        p('决策树通过一系列阈值划分样本。例如先判断流动比率低不低，再在低流动性的公司中判断负债率高不高，就能表达交互。单树容易追逐噪声，所以需要深度、叶节点样本数等限制。'),
        p('Random Forest 用多个随机化样本/特征的树平均，减少单树波动。树太深仍可能利用数据中的公司特征指纹；公司 ID 虽然被禁止，独特财务组合仍可能让已见公司更容易被识别，所以必须做公司不重叠测试。'),
        p('Boosting 顺序增加树，每一轮修正当前损失下尚未解释的部分。它不等于“每轮只学错分样本”，而是根据损失梯度调整。学习率、树深和轮数共同影响拟合能力；不能用训练分数高来选无限加树。'),
        table(core[core.family.isin(['logistic','random_forest','xgboost','catboost'])],['family','S0','S1'],['模型','S0 开发 AP','S1 开发 AP']),
        p('XGBoost 是本轮实际完成的 Boosting 实现；规格也允许依赖不可用时使用 HistGradientBoosting，但不能把替代结果叫作 XGBoost。本轮先诊断并修复 OpenMP 冲突，保留了真实 XGBoost 实验。CatBoost 只在依赖可用且确实完成时进入表格。'),
        p('树模型是否胜过神经网络，要先看同信息集固定配方，再看各自适合的优化 pipeline。不能一边给树原始金额、一边给网络精选比率，再把差别全部归因于算法。定位：classical.py、develop.py。')),
      page('12. 神经元、MLP 与真实训练',
        p('一个神经元先做加权和，再经过激活函数。ReLU 将负输入截为零，使多层网络能表示非线性函数。MLP 把多个这样的层连接起来；参数通过损失的梯度反向传播更新，不是把财务比率手工“调到正确答案”。'),
        p('本项目输出的是一个 logit，标签 shape 为 [batch]。BCEWithLogitsLoss 内部结合 sigmoid 与二元交叉熵，数值上更稳定；训练前不能再手动 sigmoid 一次。评价时才 sigmoid，得到未校准分数，再进入独立校准器。'),
        p('AdamW 根据梯度更新参数，并采用 weight decay。Dropout 在 train 模式随机屏蔽部分激活，降低依赖特定路径的程度；eval 模式关闭这类随机行为。因此“训练完直接预测”若漏了 eval，会让同一 checkpoint 的输出漂移。'),
        p('每次更新都检查 loss 和 gradient 是否有限，再做 gradient clipping。随机种子控制初始化与 DataLoader 顺序，但跨设备、版本和库的逐位一致不能保证。可复现承诺应包括环境版本、保存工件和数值容差。'),
        code('model.train()\nlogits = model(x_batch)\nloss = BCEWithLogitsLoss()(logits, y_batch)\nloss.backward()\noptimizer.step()\nmodel.eval()\n# inference_mode + sigmoid during evaluation'),
        p('这些是教学步骤缩写；正式实现还包含 zero_grad、权重、有限值检查和固定种子。真实训练与停止历史保存为每轮 CSV，模型 state_dict 保存为 checkpoint。定位：neural.py 的 fit_neural、predict_logits。')),
      page('13. 残差、正则化与内部停止',
        p('残差块计算 x + F(x)。直觉是网络可以学习对已有表示做修正，而不必每层都完全重造表示；恒等路径也有助于梯度传播。它不意味着任何小样本任务都需要更深网络，更不能保证超过树模型。'),
        p('Residual MLP 使用输入投影、归一化残差块和最终 logit head。Dropout、weight decay、宽度和深度共同限制容量。小数据或重尾特征下，优化速度与泛化未必同步：训练 AP 持续提高，不表示后续年份更好。'),
        p('内部停止看训练窗口尾部年份的 AP。选定最佳 epoch 后，丢弃那次预处理和模型，在完整训练窗口上从头训练固定轮数。这样最终评分使用了完整训练信息，同时外层评分期没有参与停止决策。'),
        table(core[core.family.isin(['mlp','residual_mlp','ft_transformer'])],['family','S0','S1'],['PyTorch 架构','S0 开发 AP','S1 开发 AP']),
        p('读表不能只盯最大值：不同年份可能出现反转，复杂网络也增加训练时间。完整表还有最差年度、成本、参数量和实际 epochs。若残差结构没有带来稳定增益，应如实归为当前设定下未证明有效的复杂度。'),
        p('面试解释重点：残差改变网络的函数参数化，不改变标签定义；early stopping 改变训练长度选择方式，不允许读取外层评分标签。定位：neural.py 的 ResidualBlock、train.py 的 fit_pipeline。')),
      page('14. FT-Transformer：注意力看的是特征',
        p('FT-Transformer 把每个数值特征变成一个 token：token_j = x_j × weight_j + bias_j。这样模型知道不同位置代表流动性、盈利或金额等不同输入，再通过 attention 让特征表示互相交流。CLS token 汇总信息，用于最终分类。'),
        p('这里的 token 轴是特征轴，不是年度轴。把 18 个财务字段 reshape 成 18 个时间点，不能称为学习了企业历史。真正的时间序列必须由同一家公司不同年度的记录组成，GRU 扩展另有日历窗口。'),
        p('自注意力会根据当前输入改变特征间的信息混合方式。它可以表达复杂交互，也引入额外参数和计算。表格任务的样本规模、变量质量、正则化和验证协议往往比架构名字更决定结果。'),
        p('本轮是小型、独立组织的教学实现：数值 token、CLS、两层 attention、归一化及 logit head。引用原论文并不表示复现了其全部数据、搜索预算或成绩，也不表示个人发明了 Transformer。'),
        p('注意力权重不是因果解释。权重大可能只是内部表征选择，且受相关变量影响；正式解释使用固定 checkpoint 的分组 permutation。若打乱某组使 AP 下降，只能说明该模型在这批样本上依赖该组信息。'),
        code('neural.py: TabularNet(family="ft_transformer")\nresults/public/development_summary.csv\nresults/public/group_permutation_importance.csv')),
      page('15. GRU：真正的年度轨迹对照',
        p('GRU 通过门控制旧状态保留多少、新输入写入多少，用于压缩年度轨迹。本项目 tensor 为 [batch, 3, features]，按 t-2、t-1、t 排列。缺失年度的数值部分归零，另附 observed flag；目标年度始终保留。'),
        p('共享 snapshot preprocessor 只在当次训练角色拟合，三个年度槽使用同一组尺度。若 t-1 不存在，不会将 t-2 前移填补。padding/mask 是覆盖描述，不是合成一份真实财务报表。'),
        table(sequence,['family','feature_set','macro_year_AP','mean_fit_seconds'],['模型','信息集','开发 AP','平均训练秒']),
        p('对照包含当年 S0 tree/MLP，以及同三年窗口的 flattened tree/MLP。若历史 flattened 模型已获得同等增益，就不能把提升全部归功于循环结构。所有对照在相同 target row_id 上评价；历史不足公司没有被静默删除。'),
        p('本轮固定三年窗口，没有根据历史测试反复试 2/3/4 年；也没有把普通 SMOTE 用在序列张量上。序列模型仍然预测最终状态标签，不自动获得未来一年事件含义。'),
        code('configs/sequence_extension.yaml\nsequences.py: sequence_tensor, fit_sequence_pipeline\nfeatures.py: build_features(..., "LAGS")')),
      page('16. 不平衡：权重、复制和插值不同',
        p('少数类占比低，会让模型更容易优先解释多数类。可选方案包括不处理、改损失权重、复制正类和插值生成训练点。它们改变训练目标或曝光频率，不会产生新的真实公司。'),
        p('Class weight 使用当前训练折的负/正样本比，再取 alpha 次方。alpha=0 为不加权，0.5 为温和加权，1 为完整比率。它强调正类误差，但可能扭曲未校准概率，也可能降低 AP，因此需要实测。'),
        p('ROS 复制已有少数类样本；SMOTE 在少数类近邻之间插值。后者对尺度敏感，且插值后的资产/利润/负债未必仍满足财务关系。加入 ROS 能判断收益来自“多看到正类”，还是插值本身。'),
        table(ab[ab.recipe.isin(['S1 base','Weight alpha 0.5','Weight alpha 1','ROS','SMOTE'])],['recipe','xgboost','residual_mlp'],['训练处理','XGBoost 开发 AP','Residual MLP 开发 AP']),
        p('所有方法只在当前训练角色执行，停止集、开发评分、2013、2014 和历史评估保持自然比例。SMOTE 默认与 class weight 分开；否则难以知道是哪一项带来改变。本轮使用精确 KD-tree 近邻，修复的是运行环境冲突，而不是挑有利结果。'),
        p('读表时保留未改善的行。若加权提高 recall 却降低 AP，说明决策权衡发生变化，不能只报道召回上升。定位：preprocessing.py 的 positive_weight、resample_train；controlled_ablation_summary.csv。')),
      page('17. 校准与阈值：排序好不等于概率准',
        p('两个模型可能排序相同，但一个把最高风险样本打 0.9，另一个打 0.2。AP 可能相同，概率质量却不同。校准要让相近分数的一组样本，其实际标签比例与分数更接近。'),
        p('Platt calibration 对基础 raw score 拟合一个 sigmoid 映射，数据来自自然比例的 2013 年。之后不把 2013 加回基础模型再沿用原校准器，否则映射所针对的分数机制已改变。'),
        fig('screening_calibration','左图按年度固定审查比例，右图为历史期可靠性分箱。右图每个点必须结合 reliability_bins.csv 的样本数和正类数理解；高分箱很小可能大幅波动。'),
        table(calibration_comparison(c),['representative','period','raw_Brier','cal_Brier'],['代表','检验期','未校准 Brier','Platt Brier']),
        p('Brier 是概率与 0/1 标签的平方误差均值，log loss 对高置信度错误更敏感。两者包含区分度和概率刻度等影响，不是纯粹的“校准误差”。本项目同时保留未校准对照、2014 检验与历史可靠性分箱。'),
        p('2014 选一个 F1 阈值并固定应用到 2015-2018。Top 5% 则按年度排序分配固定工作量。不要在历史 PR 曲线上挑最好阈值后称为部署效果，更不要把 failed 标签下的校准概率称作已验证的一年期 PD。'),
        code('calibration.py: fit_calibration\nevaluation.py: choose_threshold, reliability')),
      page('18. 一条记录如何穿过整个 pipeline',
        p('以下数值纯属教学示意，不是实测模型、训练参数或研究成绩：设净利润=-5、资产=100、总负债=80、流动资产=30、流动负债=60。'),
        {'type':'table','header':['示意步骤','数值'],'rows':[['财务结构','ROA=-0.05，负债率=0.8，流动比率=0.5'],['假设训练均值/标准差','分别用 (0,0.1)、(0.5,0.25)、(1,0.5)'],['标准化 tensor','[[-0.5, 1.2, -1.0]]'],['定义的教学权重','w=[-1,0.8,-0.4]，b=-1，因此 z=0.86'],['sigmoid 与示意校准','sigmoid(z)=0.703；sigmoid(0.8z-1.5)=0.307'],['示意判断','若阈值0.25，则标高风险；这些都不是正式参数']]},
        p('正式流程先用 company/year 对齐与查询历史，再构造财务特征；ID 不进入 tensor。随后加载原 checkpoint 的 median、截尾边界、尺度及列顺序，不能对当前数据重新 fit。'),
        p('tensor 在 eval/inference 模式输出 logit，sigmoid 得到未校准分数，2013 Platt 映射得到 q，再比较 2014 固定阈值。年度 Top 5% 则在完整年度 cohort 中排序。形状正确但列顺序错误，仍不是原模型。'),
        p('正式模型的完整实测数值链另见本地 Local_Case_Walkthrough_EN_CN.md 与 local_numerical_walkthrough.json：包含预登记案例的原始字段、实际截尾/尺度、完整 tensor、logit、校准和固定阈值。Notebook 04 重载正式工件核对；公开输出只显示一致性与聚合统计。'),
        p('真实 TP/FP/FN/TN 组的计数、分数及财务比率中位数另见 confusion_cell_profiles.csv，不代表每家公司都符合组内画像。亲自核查时，从注册案例逐项追踪 build_features、transform、predict_raw、calibrated_predict 和 threshold。')),
      page('19. 真实结果：哪些变化有效，哪些无效',
        table(core,['family','S0','S1'],['固定配方','S0 开发 AP','S1 开发 AP']),
        p(interpretation['h4_cn']),
        p(f'历史期主报告采用开发选出的 ML/DL 配置：{DISPLAY[ml.family]} 的 AP={ml.AP:.3f}、ROC={ml.ROC_AUC:.3f}；{DISPLAY[dl.family]} 的 AP={dl.AP:.3f}、ROC={dl.ROC_AUC:.3f}。同协议 S0 Logistic 为 AP={base.AP:.3f}。不要从这几行中事后挑一个测试最大值倒填为开发选择。'),
        p('H1 看 S1-S0：金融结构与稳健变换是否提供增量。H2 看 S2-S1 与 flattened controls：历史信息是否有用。H3 先看 S1 同信息集，再看优化 pipeline。H4 保留 no selection、无加权、ROS、SMOTE 的完整对照。H5 结合陌生公司与 hash 留组，而不是只看总体 AP。'),
        p('失败实验也有研究价值：没有提升的筛选或采样说明该干预在当前信息集/优化设置下未获支持，而不是方法永远无效。先解释变化因素和成本，再谈是否值得保留。完整 ablation 表的负差异不能删除。'),
        p('旧报告采用不同训练/验证角色，其保存成绩只属于档案。新结果只能与本轮同协议重训的基准配对比较。本报告不将旧 AUC 与新 AP 混算提升，也不把不同正类比例年份的 AP 随意相除。')),
      page('20. 公司泛化、区间与种子波动',
        table(novelty,['family','novelty','companies','positive_companies','AP','lower','upper'],['模型','相对2012','公司数','failed公司','AP','区间下界','区间上界']),
        table(ci_rows(c),['model','lower','upper','delta_lower','delta_upper'],['代表','AP下界','AP上界','相对基准差下界','差上界']),
        p(interpretation['ci_cn']),
        p('未见公司按基础模型真实拟合截止 2012 定义。该子集仅 16 家 failed 公司，所以即使有很多年度行，结论仍可能很不稳定。另有截至 2014 的敏感性表，正类更少时也不能静默隐藏。'),
        p('公司 hash 压力测试将固定一组公司的所有年份从拟合、选择、停止和校准中排除；Logistic、Boosting、Residual MLP 使用训练前固定配方，因此称 untuned sensitivity check。不能把包含这些公司标签调过的全局最优配置伪装成严格独立调参。'),
        p('Bootstrap 每次整家公司抽样，同一抽样用于模型对比，保留其所有年度行。本轮 2,000 次。差值区间跨零时，不支持在该方法下有把握地声称正向差异；即使全为正，也只针对已拟合模型和当前样本，不涵盖全部调参、宏观冲击或标签时点问题。'),
        p('五个 seed 都在同一个历史集上，不是五份独立测试集。它们描述随机训练波动，不能取最好的 seed 当主结果。主 run 固定 seed=42。定位：paired_company_bootstrap.csv、seed_results.csv、group_stress_metrics.csv。')),
      page('21. 90 秒开场：先说证据边界',
        p('中文开场（按自然语速练习，时长因人而异）：'),
        p(f'我把原来的团队财务困境项目扩展成了一套可复现的个人研究。第一步不是换更复杂的模型，而是重新核对数据：共有 {c["audit"]["rows"]:,} 条公司年度记录，同一家公司所有年份的标签都一样，又没有可靠的事件日期，所以我把任务明确限定为回溯式失败公司识别。然后我冻结了四个按年份扩展的开发折，把最终训练、2013 校准、2014 阈值和 2015-2018 历史评估分开。在相同记录上，我比较原始财务金额、比率和历史变化，训练了传统模型与 PyTorch 网络，并用筛选和采样消融检查增益来源。开发期选出的 {DISPLAY[ml.family]} 和 {DISPLAY[dl.family]} 在历史期 AP 分别为 {ml.AP:.3f} 和 {dl.AP:.3f}。我同时检查年度 Top 5% 筛查、未见公司、公司留组、五种子和公司聚类区间。结论只针对这份历史标签和已被查看过的数据，不声称是一年期违约预测。代码实现和文稿有 Codex 辅助，个人归属与原团队成果分开记录。'),
        p('English opening:'),
        p(f'I extended an earlier team project into an auditable benchmark for retrospective failed-company classification. I first checked the label: it is constant within companies, while event and statement-publication dates are unavailable. I therefore avoid claiming one-year default prediction. I registered four expanding-year development folds, then separated final fitting, 2013 calibration, 2014 threshold selection and the previously inspected 2015-2018 historical evaluation. I compared raw amounts, interpretable financial ratios and causal historical features across classical models and compact PyTorch networks. The development-selected {DISPLAY[ml.family]} and {DISPLAY[dl.family]} pipelines recorded historical AP of {ml.AP:.3f} and {dl.AP:.3f}. I also examined annual review budgets, sampling and selection ablations, unseen companies, company-disjoint stress tests, five seeds and paired company-bootstrap uncertainty. The contribution is the controlled, reproducible evidence chain and its limitations, rather than a claim that deeper models must win. Codex assisted implementation and drafting; original team work is separately acknowledged.'),
        p('使用前先亲自核对表、代码与贡献表。若尚不能解释某一步，应说“在工具辅助下实现并正在核验”，不要背成亲手从零完成所有代码。')),
      page('22. 三分钟深入版与可主张范围',
        p('第一分钟：从任务定义讲起。原团队已经比较过 ML 和神经网络，新扩展解决的是标签解释、特征含义衔接、时间角色和可核查性。说明 firm-year 与公司的区别，解释为什么公司最终状态不能自动转成一年内破产事件。'),
        p('第二分钟：选择一项受控实验深入。以 S1 对 S0 为例，说明净利润/资产与负债/资产为什么比单独金额更有经济解释；再展示相同年度、行集合、随机种子和模型配方下 AP 如何变化。接着讲训练内预处理、内层停止后完整重拟合，以及无筛选/无加权为何必须存在。'),
        p('第三分钟：从最佳点回到证据强度。展示历史 AP 与年度预算召回，解释固定 F1 阈值为何只看 2014；说明未见公司正类仅 16 家，bootstrap 按公司而非按行，五个 seed 也不是五份测试。以一项无效的采样或筛选结果收尾，说明你会接受什么证据推翻原来的直觉。'),
        {'type':'table','header':['可以主张','目前证据不支持'],'rows':[['完成真实本地比较与可复现工件','已获得未来业务上线表现'],['在同协议、同记录上比较改动','跨旧协议直接计算提升'],['个人主导范围与AI辅助透明','亲手独立编写每行或发明算法'],['历史公司泛化压力分析','彻底消除历史数据暴露'],['最终状态标签下的校准分数','已验证的一年期 PD 或因果结论']]},
        p('亲自验证清单：读 DATA_CARD；画出时间角色；从一个本地案例重算特征/score；解释一项失败消融；读 paired interval；运行一次 checkpoint reproduce；核对贡献与来源；再决定是否采用简历表述。手册帮助形成可核查理解，不保证覆盖所有面试问题。'),
        p('简历中英文各两条另存于本地 Resume_Bullets_EN_CN.md，明确为待本人核对。原课程团队名单与新个人扩展署名分开，不把原提交改写为个人独立作品。'))
    ]
    qa=[
      ('1. 你的目标变量是什么？','数据中 failed/alive 的公司最终状态，主任务是回溯式公司年度识别。没有可靠事件日和固定 horizon，不能称一年期破产事件预测。核查：DATA_CARD.md、data.py: load_data。'),
      ('2. 为什么 78,682 行不等于同样多的独立样本？','同一公司重复出现且标签固定，财务与误差会相关。公司数为 8,971；区间按公司成组抽样。核查：data_audit.json、evaluation.py: bootstrap_pairs。'),
      ('3. 标签是什么时候可获得的？','源数据没有提供这一点。时间输入分离只限制财务年份，不能证明训练当时知道最终状态。不能凭最后出现年补事件日期。核查：docs/RESEARCH_PROTOCOL.md。'),
      ('4. 为什么不用随机切分？','随机行切分会把同一公司的相邻年份混入各侧，也不对应向后续年份迁移。这里按年份块验证，另做全公司留组。核查：splits.py: DEVELOPMENT、group_id。'),
      ('5. 你如何保证没用历史测试调参？','development 不解析 2013 之后标签；最终角色单独开放，模型与工件冻结后才读取历史标签。原测试已暴露的事实仍保留。核查：data.py、configs/final_freeze.json。'),
      ('6. 报告中的 improved 是相对谁？','只能与本轮同协议、同记录重训的基准配对。旧团队训练角色和指标不同，不做跨协议提升百分比。核查：historical_metrics.csv、results/CLAIMS.csv。'),
      ('7. X17/资产为什么不叫有息债务率？','X17 的字段定义是 total liabilities，范围比有息债务更广。经济解释必须遵守字典。核查：features.py: RATIOS、feature_registry.csv。'),
      ('8. 金额和比率各提供什么？','金额包含规模信息，比率表达财务结构；二者都可能有用。先登记经济理由，再在同配方 S0/S1 中比较。核查：controlled_ablation_summary.csv 的 S0 raw 与 S1 base。'),
      ('9. 为什么保留负净利润？','亏损是有效信号，不是自动异常。signed log 保留符号并压缩重尾；普通 log 不能直接处理负数。核查：features.py: build_features；符号保持测试。'),
      ('10. 截尾是否必然改善？','不必然。它降低极端值支配，也可能损失危机信号。因此保留 No clipping 受控行，阈值仅从训练计算。核查：preprocessing.py、ablation 表。'),
      ('11. 为什么不统一只用最稀疏名单？','线性筛选可能删去非线性交互所需变量，AUC 简洁性规则也不等于 AP 最优。无学习式筛选是正式候选。核查：selection.py、feature_selection_stability.csv。'),
      ('12. permutation importance 怎么做才不泄漏？','用于选择时，树与置换评分都限于外层训练内部；用于解释时加载正式 checkpoint，历史解释只作事后描述。两种角色分开。核查：selection.py 与 explain.py。'),
      ('13. 你怎样处理 early stopping？','在训练窗口尾部年份选 epoch/轮数，再从头重拟合完整训练窗，固定长度后评分外层年份。不能用外层评分期停训再当独立成绩。核查：train.py: fit_pipeline。'),
      ('14. BCEWithLogitsLoss 前要 sigmoid 吗？','不要。该损失已经稳定地组合 sigmoid 与交叉熵。训练输入 logit，推断时才转换分数；标签与输出均为 [batch]。核查：neural.py 与 torch shape/loss tests。'),
      ('15. MLP、残差、Transformer 是三个领域吗？','它们都属于神经网络/深度学习，深度学习又属于 ML。差异在结构与参数化，不能按名字决定优劣。核查：TabularNet 与同 S1 开发比较。'),
      ('16. attention 能解释破产原因吗？','不能。权重是内部计算的一部分，相关输入会改变其含义。分组置换也只是当前模型的预测依赖，不证明因果。核查：group_permutation_importance.csv。'),
      ('17. 为什么财务列不能当时间点？','不同财务变量没有年度先后关系。GRU 的时间槽来自同公司 t-2/t-1/t，缺年份有明确 mask。核查：sequences.py: sequence_tensor。'),
      ('18. GRU 改善是否证明时序架构更好？','先与同历史信息的 flattened tree/MLP 比，再与当前快照比。若仅比快照，增益可能来自历史可用性而非 GRU。核查：sequence_control 与 sequence_extension 开发结果。'),
      ('19. class weight 与 SMOTE 有何区别？','前者改变损失贡献，后者生成少数类插值训练点。ROS 控制额外曝光效应，三者不默认叠加；外部角色不重采样。核查：positive_weight、resample_train 和受控表。'),
      ('20. AP、ROC 与 Top 5% 怎样一起读？','AP 强调 precision-recall，ROC 看整体两类排序；Top 5% 对应年度有限审查容量。必须列 prevalence，避免只报高 accuracy 或无工作量约束的 recall。核查：evaluation.py。'),
      ('21. 校准后就是违约概率吗？','只是在此标签定义下的概率刻度。2013 拟合，2014/历史检查；没有事件窗口就不能叫年度 PD。Brier/log loss 也不是纯校准误差。核查：calibration.py、reliability_bins.csv。'),
      ('22. 未见公司测试为什么仍不算全新外部证据？','公司分组来自同一历史数据，且原项目已接触该历史期；它提供更严格的公司隔离压力，但不消除来源、标签时点与历史暴露。核查：group_stress_metrics.csv。'),
      ('23. 区间为什么按公司而不按 seed？','公司是重复记录的聚类单位；同一次抽样用于所有模型才是配对。seed 只反映同一测试集上的训练波动，不是新增样本。核查：bootstrap_pairs、seed_results.csv。'),
      ('24. 哪些部分是你做的，哪些有工具辅助？','可说本人提出并主导本次扩展范围，实际实现、调试和报告有 Codex 辅助，尚需本人阅读和复核。不能把原团队成果或通用算法据为个人发明。核查：INDIVIDUAL_CONTRIBUTIONS.md、AI_USAGE.md、ACKNOWLEDGEMENTS.md。')
    ]
    for i in range(4):
        blocks=[]
        for q,a in qa[i*6:(i+1)*6]:blocks.extend([p(q),p(a)])
        pages.append(page(f'{23+i}. 面试追问 {i*6+1}-{(i+1)*6}',*blocks))
    return pages
