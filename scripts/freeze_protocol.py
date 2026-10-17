"""Create the research registration before any model training."""
from pathlib import Path
import datetime, hashlib, json, yaml
ROOT=Path(__file__).resolve().parents[1]
p=yaml.safe_load((ROOT.parent/'Distress_ML_Upgrade_Codex_Pack/RESEARCH_PROTOCOL_DRAFT.yaml').read_text())
p['protocol_id']='distress_individual_extension_v1'
p['source_root']='environment_variable_DISTRESS_DATA_or_parent_raw_csv'
p['frozen_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
p['runtime']['threads']=2
p['implementation']={
 'primary_device':'cpu','device_reason':'Small tabular networks; CPU provides a shared conservative device and reproducible checkpoint reloads.',
 'base_recipe':{'clip':True,'quantiles':[0.005,0.995],'sampling':'none','selector':'none','weight_alpha':0,'seed':42},
 'models':{'logistic':{'C':1,'max_iter':1500},'random_forest':{'n_estimators':120,'max_depth':10,'min_samples_leaf':5},
 'xgboost':{'n_estimators':180,'max_depth':3,'learning_rate':0.05,'subsample':0.9,'colsample_bytree':0.9},
 'catboost':{'iterations':180,'depth':5,'learning_rate':0.05},
 'mlp':{'width':64,'depth':2,'dropout':0.15},'residual_mlp':{'width':64,'depth':2,'dropout':0.15},
 'ft_transformer':{'token_dim':16,'heads':2,'depth':2,'dropout':0.15}},
 'neural_training':{'max_epochs':25,'patience':4,'batch_size':1024,'learning_rate':0.001,'weight_decay':0.0001,'gradient_clip':5},
 'stage_2':'Best development ML and best development DL family only; each two extra S1 candidates: capacity and regularization. No test feedback.',
 'stage_2_candidates':{'xgboost':[{'max_depth':2,'learning_rate':0.08},{'max_depth':5,'learning_rate':0.03}],
 'random_forest':[{'max_depth':6,'min_samples_leaf':10},{'max_depth':16,'min_samples_leaf':3}],
 'logistic':[{'C':0.1},{'C':10}], 'catboost':[{'depth':3},{'depth':7}],
 'neural':[{'width':96,'dropout':0.1,'learning_rate':0.0007},{'width':48,'dropout':0.3,'weight_decay':0.001}]},
 'ablation_representatives':['xgboost','residual_mlp'],
 'ablations':'Each representative: S2 none, S1 no clipping, S1 weight alpha .5 and 1, S1 ROS and SMOTE, S1 elastic net and grouped permutation selection. Fixed recipe otherwise.',
 'elastic_net_inner_grid':{'C':[0.01,0.1],'l1_ratio':[0.5,0.9]},
 'model_aware_inner':'Two expanding tail-year folds; economic feature groups jointly permuted; choose top 2, top 4, or all groups using last inner year AP.',
 'selection_ties':'Prefer lower cost, then lexicographic config id.',
 'final_representatives':'Best ML and DL config by equal-weight seven development-year AP, plus same-protocol S0 baselines and all S1 core recipes at seed42. Repeat best ML/DL across five seeds.',
 'stress_fixed_recipe':{'families':['logistic','xgboost','residual_mlp'],'feature_set':'S1','clip':True,'sampling':'none','selector':'none','weight_alpha':0,'seed':42,'untuned':True},
 'bootstrap':{'repetitions':2000,'seed':2026,'metrics':['AP','ROC_AUC','macro_year_AP','macro_year_recall_top5'],'unit':'company','paired':True},
 'case_rule':'For each TP/FP/FN/TN: smallest sha256(case-v1 + company + year), one case per cell; local identifiers only, public aggregates.',
 'optional':'After core: CatBoost if import works; GRU with fixed three calendar years and flattened-lag XGBoost/MLP controls; equal-rank ML+DL ensemble selected on OOF only.',
 'top_budget_rounding':'ceil(budget*n_year); score descending, SHA256(company|year) ascending for deterministic label-free ties.',
 'source_data_hash':'cff2c899a97ecd629415cb22f59186000e74e1c0a78cfae036c0a53025419b5e'}
path=ROOT/'configs/protocol_v1.yaml'
if path.exists(): raise SystemExit('Protocol already exists; refusing to overwrite registration.')
path.write_text(yaml.safe_dump(p,sort_keys=False,allow_unicode=False))
h=hashlib.sha256(path.read_bytes()).hexdigest()
(ROOT/'configs/protocol_registration.json').write_text(json.dumps({'protocol_sha256':h,'registered_at_utc':p['frozen_at_utc'],'status':'registered_before_training'},indent=2))
(ROOT/'docs/RESEARCH_PROTOCOL.md').write_text('''# Registered research protocol\n\nOwner: Qidian (Martin) Xing. Independent extension.\n\nThe immutable machine specification is `configs/protocol_v1.yaml`; its registration hash is in `configs/protocol_registration.json`. H1 compares S0/S1, H2 S1/S2, H3 model families, H4 selection and sampling, and H5 company novelty/stress.\n\nThe task is retrospective failed-company classification. Financial-statement publication dates, true bankruptcy event dates and a fixed event horizon are unavailable. All historical years have previously been inspected. The 2015-2018 evaluation cannot be described as a new untouched holdout.\n\nD1 trains through 2005 and scores 2006-2007; D2 through 2007 scores 2008-2009; D3 through 2009 scores 2010-2011; D4 through 2011 scores 2012. Select by equal-weight mean AP over these seven scoring years. Early stopping uses only the last training year, followed by full-window refitting at fixed length.\n\nFinal base-model fit: 1999-2012. Calibration: 2013. Threshold: 2014. Historical evaluation: 2015-2018, once after model/config/artifact freeze. Group stress uses three fixed SHA256 groups and pre-registered untuned recipes; held-out companies cannot enter any fit, calibration or stopping role.\n\nAll data-dependent transforms and selectors are refit within their training role. No company ID, year or lifetime summary is a model feature. Dynamic windows use exact calendar offsets. All target rows are retained. Model artifacts and row-level predictions remain local.\n\nTraining is bounded by 6 hours soft, 8 hours hard including delivery, and 600 fit calls including internal selection and stopping refits. Every fit starts with a durable ledger event. Two threads, one neural training process. No paid resources.\n''')
(ROOT/'docs/DECISION_LOG.md').write_text('# Decision log\n\n- '+p['frozen_at_utc']+': Registered before first training. Structural audit confirms both raw copies agree, company-constant labels and calendar gaps. Retain original labels. No new event dates inferred.\n- CPU selected for conservative tabular workload; MPS availability will be recorded but is not required. Default quantile clipping is controlled by an explicit no-clipping ablation. Positive ratio denominators are required; invalid ratios become NaN with masks. Units/currency remain unverified.\n- Writing uses the requested research format; no unrelated account or writing history is accessed.\n')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
(ROOT/'RUN_STATUS.json').write_text(json.dumps({'phase':'protocol_registered','started_at_utc':now,'complete':False,'protocol_sha256':h},indent=2))
(ROOT/'NEXT_ACTIONS.md').write_text('# Next actions\n\n1. Implement and test fold isolation.\n2. Run pilot and four development folds.\n3. Freeze models, run historical and company stress evaluation.\n4. Generate and inspect reports, notebooks, and release audit.\n')
print(h)
