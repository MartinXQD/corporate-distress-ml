"""Create English teaching notebooks that call the shared implementation."""
from pathlib import Path
import json
import os
import nbformat as nbf
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
INIT='''from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from IPython.display import display, Image
root = Path.cwd()
if root.name == "notebooks":
    root = root.parent
sys.path.insert(0, str(root / "src"))
from distress_ml.common import ROOT, PUBLIC, ART
pd.set_option("display.max_columns", 12)
print("Independent extension: saved real-data evidence; no training is triggered.")
'''

def notebook(name,title,cells):
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('# '+title+'\n\nQidian (Martin) Xing. Independent extension with disclosed Codex assistance. The historical period was previously inspected. Source data and row-level predictions are not redistributed.'),nbf.v4.new_code_cell(INIT)])
    for typ,text in cells: nb.cells.append(nbf.v4.new_markdown_cell(text) if typ=='md' else nbf.v4.new_code_cell(text))
    nb.metadata.kernelspec={'display_name':'Python (distress local)','language':'python','name':'distress-local'}
    path=ROOT/'notebooks'/name
    nbf.write(nb,path)
    return path

def main():
    paths=[]
    paths.append(notebook('01_Data_and_Features.ipynb','01 - Data, labels and causal financial features',[
      ('md','## Question\nWhat does a row represent, and which information can legitimately enter a model?\n\n## Logic\nAudit the pinned snapshot, keep the ultimate-status label interpretation, then register snapshot, ratio and exact-calendar history features. Audit counts are not new model performance.'),
      ('code','audit = json.loads((PUBLIC / "data_audit.json").read_text())\nprint({k: audit[k] for k in ["rows", "companies", "row_label_counts", "company_label_counts", "sequence_audit"]})\ndisplay(pd.read_csv(PUBLIC / "split_audit.csv"))'),
      ('md','Raw units and publication/event dates are unverified. X9 and X16 are identical. Negative financial values are retained. Ratios use positive current denominators and train-role imputation/masks. Exact t-1 absence is never replaced by t-2.'),
      ('code','registry = pd.read_csv(PUBLIC / "feature_registry.csv")\ndisplay(registry[["feature", "formula", "group", "time", "set"]].head(12))\ndisplay(Image(filename=str(PUBLIC / "figures/prevalence.png")))'),
      ('md','## Local code verification\nThe following branch loads only permitted development labels and checks all feature sets retain the same target rows. No raw company identifiers or financial rows are displayed.'),
      ('code','from distress_ml.data import load_data\nfrom distress_ml.features import build_features\nraw_available = (ROOT.parent / "american_bankruptcy.csv").exists() or bool(__import__("os").environ.get("DISTRESS_DATA"))\nif raw_available:\n    data = load_data("development")\n    shapes = {s: build_features(data, s).shape for s in ["S0", "S1", "S2", "LAGS"]}\n    assert len({shape[0] for shape in shapes.values()}) == 1\n    print("Development-only feature shapes:", shapes)\nelse:\n    print("Authorized raw snapshot excluded from public release; aggregate audit remains available.")')]))
    paths.append(notebook('02_Validation_and_ML.ipynb','02 - Calendar validation, classical ML and ablations',[
      ('md','## Question\nWhere do improvements come from?\n\n## Logic\nSeparate fixed-recipe information/sampling ablations from optimized pipelines. Select by seven-year development AP; keep calibration and decision roles separate. Original team scores are archival and are not controlled baselines.'),
      ('code','from distress_ml.splits import DEVELOPMENT\ndisplay(pd.DataFrame(DEVELOPMENT, columns=["fold", "fit_end", "score_start", "score_end"]))\ndevelopment = pd.read_csv(PUBLIC / "development_summary.csv")\ndisplay(development[["family", "feature_set", "stage", "macro_year_AP", "worst_year_AP", "mean_fit_seconds"]].head(14))'),
      ('code','ablation = pd.read_csv(PUBLIC / "controlled_ablation_summary.csv")\ndisplay(ablation.pivot(index="recipe", columns="family", values="macro_year_AP"))\ndisplay(Image(filename=str(PUBLIC / "figures/development_features.png")))'),
      ('md','## Selection and cost\nElastic Net and grouped permutation are nested inside each training window. No selection is a real candidate. Class weights derive from each current training role; ROS and SMOTE never alter scoring/calibration proportions.'),
      ('code','stability = pd.read_csv(PUBLIC / "feature_selection_stability.csv")\ndisplay(stability.groupby(["family", "selector"]).agg(union_features=("feature", "size"), mean_jaccard=("mean_pairwise_Jaccard", "first")))\nselection = json.loads((ROOT / "configs/final_selection.json").read_text())\nprint("Selected using development only:", selection["representatives"])'),
      ('md','Inspect `selection.py`, `preprocessing.py` and `train.py`. Full training commands are in the README; running this notebook does not start an unbounded search or reopen historical selection.')]))
    paths.append(notebook('03_PyTorch_and_Sequences.ipynb','03 - PyTorch models, logits and genuine calendar sequences',[
      ('md','## Question\nDoes a more complex network justify its cost, and does history matter independently of architecture?\n\n## Logic\nMLP, Residual MLP and FT-Transformer all consume tabular features. Feature attention is not temporal recurrence. GRU consumes actual calendar slots, with masks, plus flattened-lag controls on the same targets.'),
      ('code','dev = pd.read_csv(PUBLIC / "development_summary.csv")\ndisplay(dev[dev.family.isin(["mlp", "residual_mlp", "ft_transformer", "gru"])][["family", "feature_set", "stage", "macro_year_AP", "mean_fit_seconds"]])'),
      ('md','`BCEWithLogitsLoss` consumes logits directly. AdamW updates train-mode parameters; scoring uses eval/inference mode. Stopping uses an inner tail year, followed by a new full-window fixed-length refit. The saved artifact below is the actual formal scoring model, not a new explanatory retrain.'),
      ('code','freeze = json.loads((ROOT / "configs/final_freeze.json").read_text())\ndl = next(m for m in freeze["models"] if "DL" in m["representative"] and m["seed"] == 42)\nfrom distress_ml.train import load_pipeline\nif (ROOT / dl["path"]).exists():\n    model = load_pipeline(ROOT / dl["path"])\n    print("Formal model family:", model.cfg["family"])\n    print("Parameter count:", model.metadata["parameters"])\n    print("Selected full-refit epochs:", model.metadata["epochs"])\n    print("Eval mode:", not model.model.training)\n    assert not model.model.training\nelse:\n    print("Local model artifacts excluded; aggregate architecture and cost tables are available.")'),
      ('code','display(dev[dev.stage.isin(["sequence_control", "sequence_extension"])][["family", "feature_set", "macro_year_AP", "mean_fit_seconds"]])\nprint("Sequence implementation: sequences.py; feature tensors: [batch, calendar_time, feature].")'),
      ('md','Changing seeds does not create independent test data. Cross-device/version bit equality is not promised. The synthetic software tests check shape, finite loss/gradients, one update and checkpoint reload; they do not supply research performance.')]))
    paths.append(notebook('04_Results_and_Interpretation.ipynb','04 - Historical results, uncertainty and checkpoint verification',[
      ('md','## Question\nHow strong is the frozen historical evidence?\n\n## Logic\nRead natural prevalence, AP/ROC and annual review budgets together. Separate seen/unseen companies, untuned group stress, seed variation and paired company intervals. This is a previously inspected historical period.'),
      ('code','hist = pd.read_csv(PUBLIC / "historical_metrics.csv")\ndisplay(hist[hist.score_type == "calibrated"][["family", "feature_set", "AP", "ROC_AUC", "macro_recall_top5", "Brier"]])\ndisplay(Image(filename=str(PUBLIC / "figures/ranking_curves.png")))'),
      ('code','display(pd.read_csv(PUBLIC / "paired_company_bootstrap.csv"))\ndisplay(pd.read_csv(PUBLIC / "group_stress_metrics.csv")[["family", "group_fold", "companies", "positive_companies", "AP", "ROC_AUC"]])\ndisplay(Image(filename=str(PUBLIC / "figures/annual_performance.png")))'),
      ('md','## Same-checkpoint walkthrough\nWhen local artifacts and the authorized snapshot exist, reload the exact DL representative and compare its full historical scores with saved predictions. For a registered error case, reconstruct the financial features, raw logit, calibration and threshold locally. Only consistency flags are displayed; identifiers, source financial rows and full predictions stay local.'),
      ('code','from distress_ml.finalize import reproduce\nfreeze = json.loads((ROOT / "configs/final_freeze.json").read_text())\ndl = next(m for m in freeze["models"] if "DL" in m["representative"] and m["seed"] == 42)\nif (ROOT / dl["path"]).exists():\n    reproduce(dl["label"])\nelse:\n    print("Checkpoint reproduction requires excluded local artifacts and the authorized raw snapshot.")'),
      ('code','case_path = ART / "audit/local_failure_cases.json"\nif case_path.exists() and (ROOT / dl["path"]).exists():\n    from distress_ml.data import load_data\n    from distress_ml.features import build_features\n    from distress_ml.train import load_pipeline\n    from distress_ml.calibration import calibrated_predict\n    data = load_data("historical")\n    model = load_pipeline(ROOT / dl["path"])\n    full_features = build_features(data, model.cfg["feature_set"])\n    case = next(x for x in json.loads(case_path.read_text()) if x["representative"] == "DL" and x["cell"] == "FN")\n    mask = data.row_id == case["row_id"]\n    raw, calibrated = calibrated_predict(model, data.loc[mask], full_features.loc[mask])\n    assert np.isclose(raw[0], case["raw_score"], atol=1e-6)\n    assert np.isclose(calibrated[0], case["calibrated_score"], atol=1e-6)\n    assert (calibrated[0] >= model.threshold) == False\n    print("Registered FN walkthrough: feature/logit/calibration/threshold consistency passed.")\nelse:\n    print("Individual cases remain local; public confusion-cell profiles are aggregated.")'),
      ('md','Interpretations are post-hoc predictive descriptions, not causes of failure. See the two reports, model/data cards, contribution disclosure and `results/CLAIMS.csv` before using claims in an interview.')]))
    kernel_dir=ROOT/'artifacts/kernels/distress-local';kernel_dir.mkdir(parents=True,exist_ok=True)
    python=ROOT/'.venv/bin/python'
    import importlib.util
    torch_lib=str(Path(importlib.util.find_spec('torch').origin).parent/'lib')
    kernel=dict(argv=[str(python),'-m','ipykernel_launcher','-f','{connection_file}'],display_name='Python (distress local)',language='python',env={'PYTHONPATH':str(ROOT/'src'),'DYLD_LIBRARY_PATH':torch_lib,'DISTRESS_OMP_READY':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
    (kernel_dir/'kernel.json').write_text(json.dumps(kernel))
    os.environ['JUPYTER_PATH']=str(ROOT/'artifacts')
    status=[]
    for path in paths:
        nb=nbf.read(path,as_version=4)
        client=NotebookClient(nb,timeout=240,kernel_name='distress-local',resources={'metadata':{'path':str(ROOT)}})
        client.execute()
        nbf.write(nb,path)
        status.append(dict(notebook=path.name,executed=True,cells=sum(c.cell_type=='code' for c in nb.cells),errors=sum(o.output_type=='error' for cell in nb.cells if cell.cell_type=='code' for o in cell.outputs)))
        print('Executed',path.name,flush=True)
    (ROOT/'results/public/notebook_execution.json').write_text(json.dumps(status,indent=2))

if __name__=='__main__':main()
