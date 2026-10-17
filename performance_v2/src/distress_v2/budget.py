"""Cost feasibility preserves registered seeds instead of picking one lucky run."""
from .training import NN

FIXED=['baseline_legacy_strong_inner_l1','baseline_v1_lags_xgb','baseline_v1_raw_catboost','baseline_v1_selected_rf']
TREE=['catboost','xgboost','random_forest','extra_trees','xgb_rank']

def affordable_pair(table,folds,elapsed):
    cost=folds.groupby('candidate').fit_seconds.mean();latest=folds[folds.year==2013].groupby('candidate').fit_seconds.mean();available=max(0.,(7.15-elapsed)*3600)
    def required(tree,neural):
        stable=sum(12*cost.get(k,60.) for k in set([*FIXED,tree,neural]))
        finals=sum(7*cost.get(k,30.) for k in [*FIXED,'baseline_v1_selected_dl_corrected_steps'])
        finals+=sum(10*max(cost.get(k,60.),latest.get(k,60.)) for k in set([tree,neural]))
        return float(stable+1.3*finals+360.)
    trees=table[table.family.isin(TREE)].sort_values('macro_AP',ascending=False)
    feasible_tree=[r.trial_id for r in trees.itertuples() if required(r.trial_id,'baseline_legacy_strong_inner_l1')<=available]
    if not feasible_tree:return None,None,dict(remaining_training_seconds=available,reason='No complete registered-seed pair fits the conservative cost forecast')
    tree=feasible_tree[0];neural=table[table.family.isin(NN)].sort_values('macro_AP',ascending=False)
    feasible_nn=[r.trial_id for r in neural.itertuples() if required(tree,r.trial_id)<=available]
    nn=feasible_nn[0] if feasible_nn else None
    return tree,nn,dict(remaining_training_seconds=available,projected_required_seconds=required(tree,nn) if nn else None,best_screened_neural=neural.iloc[0].trial_id,promotion_neural=nn,screened_best_is_affordable=nn==neural.iloc[0].trial_id,reason='Cost forecast only; no 2014/historical scores; preserve all required seeds')
