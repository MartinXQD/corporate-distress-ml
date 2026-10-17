"""Classical estimators with explicit CPU and probability-score conventions."""
import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    BOOSTING='xgboost'
except (ImportError,OSError):
    XGBClassifier=None
    BOOSTING='hist_gradient_boosting'

def make_model(family,cfg,seed,weight=1,iterations=None,stopping=False):
    if family=='logistic':
        return LogisticRegression(C=cfg.get('C',1),max_iter=cfg.get('max_iter',1500),solver='lbfgs',random_state=seed)
    if family=='random_forest':
        return RandomForestClassifier(n_estimators=cfg.get('n_estimators',120),max_depth=cfg.get('max_depth',10),min_samples_leaf=cfg.get('min_samples_leaf',5),n_jobs=2,random_state=seed)
    if family=='xgboost' and XGBClassifier is not None:
        return XGBClassifier(n_estimators=iterations or cfg.get('n_estimators',180),max_depth=cfg.get('max_depth',3),learning_rate=cfg.get('learning_rate',.05),subsample=.9,colsample_bytree=.9,tree_method='hist',n_jobs=2,random_state=seed,eval_metric='logloss',early_stopping_rounds=15 if stopping else None,verbosity=0)
    if family in ['xgboost','hist_gradient_boosting']:
        return HistGradientBoostingClassifier(max_iter=iterations or cfg.get('n_estimators',180),max_depth=cfg.get('max_depth',3),learning_rate=cfg.get('learning_rate',.05),early_stopping=False,random_state=seed)
    if family=='catboost':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=iterations or cfg.get('iterations',180),depth=cfg.get('depth',5),learning_rate=cfg.get('learning_rate',.05),thread_count=2,random_seed=seed,verbose=False,allow_writing_files=False,loss_function='Logloss')
    raise ValueError(family)

def raw_logits(model,x):
    p=np.clip(model.predict_proba(x)[:,1],1e-7,1-1e-7)
    return logit(p)
