"""Create and freshly execute the final aggregate/reload research notebook."""
from pathlib import Path
import json
import hashlib
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]

def main():
    verification=json.loads((ROOT/'results/verification.json').read_text())
    assert verification['passed'] and (ROOT/'artifacts/historical_complete.json').exists()
    fit_log=ROOT/'artifacts/logs/fits.jsonl';fit_calls_before=len(fit_log.read_text().splitlines())
    md=nbformat.v4.new_markdown_cell
    code=nbformat.v4.new_code_cell
    cells=[
        md("""# Performance V2: frozen evidence and reproducible checks

Qidian (Martin) Xing. An owner-directed, Codex-assisted extension of the earlier team project. Team credit and primary sources remain in the project acknowledgements.

This notebook reports actual completed experiments. The 2015-2018 period was inspected in earlier versions; it is not a fresh holdout. P2 ranks retrospective company status and does not establish one-year default probabilities. Final baselines and selected recipes share the 2014 fitting cutoff. Search targets, failures and budget shortfalls remain explicit.
"""),
        code("""from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from IPython.display import display

ROOT=Path.cwd()
if not (ROOT/'V2_PROTOCOL.yaml').exists():
    ROOT=ROOT.parent
assert (ROOT/'V2_PROTOCOL.yaml').exists()
for directory in [ROOT/'src',ROOT.parent/'src']:
    sys.path.insert(0,str(directory))
saved_verification=json.loads((ROOT/'results/verification.json').read_text())
assert saved_verification['passed']
selected=json.loads((ROOT/'configs/selection_freeze.json').read_text())
table=pd.read_csv(ROOT/'results/historical_2015_2018.csv').set_index('name')
print('Saved final verification passed; private-artifact checks below run separately.')
display(pd.DataFrame([{
    'protected_files':saved_verification['protected_files_unchanged'],
    'prediction_sets':saved_verification['saved_prediction_sets_verified'],
    'development_sets':saved_verification['development_prediction_sets_verified'],
    'authorized_windows':saved_verification['authorized_time_windows_verified']
}]))
"""),
        md("""## Same-cutoff comparison

Annual AP averages the four historical years equally. Pooled AP and ROC use combined years. Recall@5% applies the review budget separately each year. Averaging predictions across seeds differs from averaging seed-level metrics. Archived and strict P1 rows retain their separate roles in the complete comparison CSV.
"""),
        code("""baseline_keys=['baseline_legacy_strong_inner_l1','baseline_v1_lags_xgb',
               'baseline_v1_raw_catboost','baseline_v1_selected_rf']
keys=list(dict.fromkeys([selected['overall'],selected['DL'],*baseline_keys]))
names=['P2_seed_mean__'+key for key in keys]
primary_name='P2_seed_mean__'+selected['overall']
if selected['blend']['admitted']:
    primary_name='P2_fixed_rank_blend'
    names=[primary_name]+['P2_annual_rank__'+key for key in keys]+names
display(table.loc[names,['candidate','channel','fit_cutoff','macro_AP','AP','ROC_AUC','recall5']].round(6))
"""),
        md("""## Independent arithmetic from saved predictions

Private company-level predictions are intentionally excluded from public Git. When those local files are present, this cell checks annual/pooled metrics and the exact public results. Its outputs contain only aggregates. A public clone without private artifacts still displays the released comparison.
"""),
        code("""from distress_ml.splits import tie_key
prediction_name=('P2_fixed_rank_blend_historical.csv.gz' if selected['blend']['admitted']
                 else 'P2_seed_mean__'+selected['overall']+'_historical.csv.gz')
prediction_path=ROOT/'artifacts/predictions'/prediction_name
if prediction_path.exists():
    prediction=pd.read_csv(prediction_path,float_precision='round_trip')
    assert prediction.row_id.is_unique
    annual=[]
    for year,g in prediction.groupby('year'):
        y=g.y_true.to_numpy(int);q=g.score.to_numpy()
        order=np.lexsort(([tie_key(company,int(year)) for company in g.company_id],-q))
        count=int(np.ceil(.05*len(g)))
        annual.append(dict(year=int(year),rows=len(g),positives=int(y.sum()),
                           AP=average_precision_score(y,q),ROC_AUC=roc_auc_score(y,q),
                           recall5=float(y[order[:count]].sum()/y.sum())))
    annual=pd.DataFrame(annual)
    reference=table.loc[primary_name]
    np.testing.assert_allclose(annual.AP.mean(),reference.macro_AP,atol=1e-9)
    np.testing.assert_allclose(annual.recall5.mean(),reference.recall5,atol=1e-9)
    np.testing.assert_allclose(average_precision_score(prediction.y_true,prediction.score),
                               reference.AP,atol=1e-9)
    np.testing.assert_allclose(roc_auc_score(prediction.y_true,prediction.score),
                               reference.ROC_AUC,atol=1e-9)
    released=pd.read_csv(ROOT/'results/historical_by_year.csv')
    released=released[released.name==primary_name].sort_values('year')
    np.testing.assert_allclose(annual[['AP','ROC_AUC','recall5']],
                               released[['AP','ROC_AUC','recall5']],atol=1e-9)
    display(annual.round(6))
    print('Independent annual and pooled arithmetic passed.')
else:
    print('Private predictions are absent. No independent local arithmetic is claimed.')
"""),
        md("""## Reload one selected tree and one selected neural pipeline

Only trusted local artifacts generated by this project are loaded. Frozen hashes, source-row alignment and saved raw scores are checked. This cell performs inference only and cannot reopen model selection.
"""),
        code("""model_root=ROOT/'artifacts/models'
if model_root.exists() and (ROOT/'artifacts/historical_complete.json').exists():
    import hashlib
    from distress_v2.training import load as load_pipeline
    from distress_v2.data import load as load_data, features
    freeze=json.loads((ROOT/'configs/artifact_freeze.json').read_text())
    data=load_data('historical');mask=data.year>=2015
    reloaded=[]
    for role in ['ML','DL']:
        item=next(m for m in freeze['models'] if m['channel']=='P2'
                  and m['candidate']==selected[role] and m['seed']==42)
        path=ROOT/item['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==freeze['artifacts'][item['path']]
        fitted=load_pipeline(path)
        score=fitted.predict(features(data,fitted.cfg['features']).loc[mask])
        saved=pd.read_csv(ROOT/'artifacts/predictions'/
                          (item['name']+'_historical.csv.gz'),float_precision='round_trip')
        np.testing.assert_array_equal(data.loc[mask,'row_id'],saved.row_id)
        np.testing.assert_allclose(score,saved.raw_score,atol=1e-6,rtol=1e-6)
        reloaded.append(dict(role=role,family=fitted.cfg['family'],seed=42,
                             fit_cutoff=2014,rows=len(score),
                             maximum_reload_error=float(np.max(abs(score-saved.raw_score)))))
    display(pd.DataFrame(reloaded))
    print('Selected tree/neural pipeline reload checks passed.')
else:
    print('Trusted private models are absent. No model reload is claimed.')
"""),
        md("""## Search coverage, uncertainty and company stress

Completed configurations are distinct from folds, seeds and TabM heads. Bootstrap intervals condition on fitted models and the current sample, omit full HPO/macro uncertainty, and retain paired company draws. Annual metrics require the same fixed year set. Company group tests use hyperparameters selected on broader companies and are post-selection stress tests.
"""),
        code("""trials=pd.read_csv(ROOT/'results/trials.csv')
display(trials.groupby(['stage','family','status']).size().rename('configurations').unstack(fill_value=0))
target='fixed_annual_rank_blend' if selected['blend']['admitted'] else selected['overall']
intervals=pd.read_csv(ROOT/'results/paired_company_intervals.csv')
display(intervals[(intervals.model==target)&intervals.metric.isin(
    ['macro_year_AP','macro_year_recall_top5'])][
    ['reference','metric','delta_mean','delta_lower','delta_upper','valid_draws']].round(6))
display(pd.read_csv(ROOT/'results/company_stress_summary.csv').round(6))
print('Company stress ranges are not confidence intervals.')
"""),
        md("""## Reproduction boundary

The final run is frozen. Do not delete confirmation or historical locks. Per-trial configurations, all aggregate trials and source code are retained. Private data, company-level predictions and serialized models are supplied locally by the owner, not downloaded from this repository.

From the performance_v2 directory, the local wrapper can run the verification module. A new training replication requires the separate owner-invoked preparation script and its own registered clock; it never resets this run.

    scripts/run_performance_v2.sh -m distress_v2.verify
    scripts/run_performance_v2.sh scripts/create_local_replication.py --name rerun1

The original eight-hour window expired with a preserved partial delivery. The owner explicitly authorized a separate bounded continuation. Exact timing, costs and shortfalls are recorded in the execution notes and experiment summary. No paid service was used.
""")
    ]
    notebook=nbformat.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},
        'author':'Qidian (Martin) Xing'
    })
    private=ROOT/'artifacts/notebook_qa';private.mkdir(parents=True,exist_ok=True)
    try:
        NotebookClient(notebook,timeout=300,kernel_name='python3',
                       resources={'metadata':{'path':str(ROOT)}}).execute()
    finally:
        nbformat.write(notebook,private/'Performance_V2_Results.executed.ipynb')
    assert all(output.output_type!='error' for cell in notebook.cells
               if cell.cell_type=='code' for output in cell.get('outputs',[]))
    encoded=nbformat.writes(notebook)
    assert '/Users/' not in encoded and '/home/' not in encoded,'Private path leaked into notebook output'
    destination=ROOT/'notebooks/Performance_V2_Results.ipynb'
    destination.parent.mkdir(parents=True,exist_ok=True)
    nbformat.write(notebook,destination)
    fit_calls_after=len(fit_log.read_text().splitlines());assert fit_calls_after==fit_calls_before
    result=dict(passed=True,fresh_kernel=True,code_cells=sum(c.cell_type=='code' for c in cells),
                outputs_contain_only_aggregates=True,fit_calls_added=fit_calls_after-fit_calls_before,
                sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                source_verification_sha256=hashlib.sha256((ROOT/'results/verification.json').read_bytes()).hexdigest(),
                file=str(destination.relative_to(ROOT)))
    (ROOT/'results/notebook_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
