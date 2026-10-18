"""Check public evidence without training, private files, or network access."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]


def verify():
    checks = []
    def require(condition, description):
        if not condition:
            raise AssertionError(description)
        checks.append(description)

    manifest = json.loads((ROOT / "docs/research_origin/aggregate_manifest.json").read_text())
    for item in manifest["files"]:
        require(hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item["sha256"],
                "immutable source: " + item["path"])
    registration = json.loads((ROOT / "configs/protocol_registration.json").read_text())
    require(hashlib.sha256((ROOT / "configs/protocol.yaml").read_bytes()).hexdigest() == registration["sha256"],
            "registered statistical protocol preserved")
    freeze = json.loads((ROOT / "configs/artifact_freeze.json").read_text())
    require(hashlib.sha256((ROOT / "configs/selection_freeze.json").read_bytes()).hexdigest() == freeze["selection_sha256"],
            "original model choice remains frozen")
    with (ROOT / "results/historical_2015_2018.csv").open() as stream:
        table = {r["name"]: r for r in csv.DictReader(stream)}
    blend = table["P2_fixed_rank_blend"]
    baseline = table["P2_seed_mean__baseline_v1_lags_xgb"]
    with (ROOT / "results/historical_by_year.csv").open() as stream:
        annual = [r for r in csv.DictReader(stream) if r["name"] == "P2_fixed_rank_blend"]
    require(sorted(int(r["year"]) for r in annual) == [2015, 2016, 2017, 2018], "fixed historical year set")
    for key, source in [("macro_AP", "AP"), ("recall5", "recall5")]:
        require(abs(sum(float(r[source]) for r in annual) / 4 - float(blend[key])) < 1e-12,
                "annual arithmetic: " + key)
    require(sum(int(r["rows"]) for r in annual) == int(blend["rows"]) == 12282, "historical row accounting")
    summary = json.loads((ROOT / "results/experiment_summary.json").read_text())
    require(abs(float(blend["macro_AP"]) - float(baseline["macro_AP"]) - summary["historical_annual_AP_difference"]) < 1e-12,
            "same-cutoff AP difference agrees with frozen summary")
    with (ROOT / "results/model_comparison.csv").open() as stream:
        view = list(csv.DictReader(stream))
    require(len(view) == len(table), "complete comparison retains every score channel")
    for row in view:
        source = table[row["source_name"]]
        for key, source_key in [("mean_annual_AP", "macro_AP"), ("pooled_AP", "AP"),
                                ("pooled_ROC_AUC", "ROC_AUC"), ("mean_annual_recall_top5", "recall5")]:
            require(row[key] == source[source_key], "comparison source identity: " + row["source_name"] + ":" + key)
    links = []
    for path in sorted(ROOT.rglob("*.md")):
        if any(part in {".git", ".venv", "artifacts", "__pycache__"} for part in path.relative_to(ROOT).parts):
            continue
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", path.read_text()):
            if re.match(r"[a-zA-Z]+:|#", target):
                continue
            target = urllib.parse.unquote(target.split("#")[0].split("?")[0].strip("<>"))
            destination = (path.parent / target).resolve()
            require(destination.is_relative_to(ROOT) and destination.exists(),
                    "local link: " + str(path.relative_to(ROOT)) + " -> " + target)
            links.append(str(path.relative_to(ROOT)) + ":" + target)
    notebooks = []
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text())
        cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        require(all(c.get("execution_count") is not None for c in cells), "executed notebook: " + path.name)
        require(all(o.get("output_type") != "error" for c in cells for o in c.get("outputs", [])),
                "no notebook error outputs: " + path.name)
        require(not re.search(r"/Users/|/home/|[A-Z]:\\\\Users\\\\", path.read_text()), "no private notebook paths: " + path.name)
        notebooks.append(path.name)
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(p in {".git", ".venv", "artifacts", "__pycache__", ".pytest_cache"}
                                     or p.endswith(".egg-info") for p in path.relative_to(ROOT).parts):
            continue
        require(path.suffix.lower() not in {".pt", ".pkl", ".joblib", ".zip", ".docx", ".pem", ".key"},
                "public artifact type: " + str(path.relative_to(ROOT)))
        require(path.stat().st_size < 10_000_000, "bounded public file size: " + str(path.relative_to(ROOT)))
        if path.suffix == ".csv":
            with path.open() as stream:
                columns = set(next(csv.reader(stream), []))
            require(not ({"company_name", "company_id"} & columns and {"score", "raw_score", "y_true"} & columns),
                    "no company predictions: " + str(path.relative_to(ROOT)))
    return {"passed": True, "checks": len(checks), "immutable_source_files": len(manifest["files"]),
            "local_markdown_links": len(links), "executed_notebooks": notebooks,
            "private_model_reloads_performed": False, "research_fits_added": 0,
            "metrics": {"mean_annual_AP": float(blend["macro_AP"]),
                        "baseline_mean_annual_AP": float(baseline["macro_AP"]),
                        "AP_difference": float(blend["macro_AP"]) - float(baseline["macro_AP"]),
                        "pooled_ROC_AUC": float(blend["ROC_AUC"]), "mean_annual_recall_top5": float(blend["recall5"])}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-receipt", action="store_true")
    args = parser.parse_args()
    result = verify()
    if args.write_receipt:
        (ROOT / "results/release_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
