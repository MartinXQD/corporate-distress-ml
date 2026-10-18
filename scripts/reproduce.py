"""Replay frozen recipes in a fresh private run; default is a read-only preflight."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.environ.get("DISTRESS_DATA"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--run", action="store_true", help="Start a fresh eight-hour frozen-recipe replay")
    args = parser.parse_args()
    if not args.data:
        parser.error("supply --data or DISTRESS_DATA with the authorized snapshot")
    import yaml
    import pandas as pd
    data_path = Path(args.data).expanduser().resolve()
    protocol = yaml.safe_load((ROOT / "configs/protocol.yaml").read_text())
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == protocol["data"]["sha256"], "Dataset hash differs"
    output = Path(args.output).expanduser().resolve()
    assert output.is_relative_to((ROOT / "artifacts/replications").resolve()), "Output must be under ignored artifacts/replications"
    assert not output.exists(), "Never overwrite or resume a released/frozen run"
    choice = json.loads((ROOT / "configs/final_model.json").read_text())
    frame = pd.read_csv(data_path, usecols=["company_name", "year"])
    assert len(frame) == 78682 and frame.company_name.nunique() == 8971
    print(json.dumps({"preflight_passed": True, "rows": len(frame), "companies": frame.company_name.nunique(),
                      "selected_families": [c["family"] for c in choice["components"]],
                      "seeds": choice["seeds"], "fit_cutoff": 2014, "training_requested": args.run}))
    if not args.run:
        return
    # Create only a new private run root. Import run-scoped modules afterward.
    for folder in ["configs", "results", "artifacts/models", "artifacts/predictions", "artifacts/logs", "artifacts/cache"]:
        (output / folder).mkdir(parents=True, exist_ok=False)
    for path in (ROOT / "configs").glob("*.json"):
        shutil.copy2(path, output / "configs" / path.name)
    shutil.copy2(ROOT / "configs/protocol.yaml", output / "configs/protocol.yaml")
    start = dt.datetime.now(dt.timezone.utc)
    (output / "EXECUTION_STATE.json").write_text(json.dumps({"started_at_utc": start.isoformat(),
        "deadline_utc": (start + dt.timedelta(hours=8)).isoformat(), "replay_of_frozen_choices": True,
        "original_research_clock_unchanged": True, "new_model_selection": False}, indent=2))
    os.environ["DISTRESS_PROJECT_ROOT"] = str(output)
    os.environ["DISTRESS_DATA"] = str(data_path)
    from distress_v2.data import load, features
    from distress_v2.training import fit, save
    from distress_v2.search import measures
    from distress_v2.finalize import annual_ranks
    from distress_v2.common import dump
    import numpy as np
    data = load("historical")
    train = data.year <= 2014
    historical = data.year.between(2015, 2018)
    family_scores = {}
    inventory = []
    candidates = [c["candidate"] for c in choice["components"]] + [choice["strongest_fixed_baseline"]]
    for candidate in candidates:
        config = json.loads((output / f"configs/trial_{candidate}.json").read_text())
        matrix = features(data, config["features"]).loc[data.index]
        scores = []
        for seed in choice["seeds"]:
            config = {**config, "seed": seed}
            fitted = fit(data.loc[train], matrix.loc[train], config, f"replay_{candidate}_seed{seed}")
            model_path = output / f"artifacts/models/{candidate}_seed{seed}.joblib"
            save(fitted, model_path)
            inventory.append({"name": f"{candidate}_seed{seed}", "candidate": candidate,
                              "seed": seed, "channel": "P2", "fit_cutoff": 2014,
                              "path": str(model_path.relative_to(output))})
            scores.append(fitted.predict(matrix.loc[historical]))
        family_scores[candidate] = np.mean(scores, axis=0)
    history = data.loc[historical]
    blend = sum(c["weight"] * annual_ranks(history, family_scores[c["candidate"]]) for c in choice["components"])
    predictions = {**family_scores, "frozen_rank_blend": blend}
    aggregate = []
    for candidate, scores in predictions.items():
        aggregate.append({"candidate": candidate, **measures(history, scores)})
        scored = history[["row_id", "company_name", "year", "y"]].copy()
        scored["score"] = scores
        scored.to_csv(output / f"artifacts/predictions/{candidate}.csv", index=False)
    pd.DataFrame(aggregate).to_csv(output / "results/replay_metrics.csv", index=False)
    hashes = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (output / "artifacts/models").iterdir() if p.is_file()}
    dump(output / "configs/final_inventory.json", inventory)
    dump(output / "configs/artifact_freeze.json", {"models": inventory, "artifacts": hashes,
         "selection_sha256": hashlib.sha256((output / "configs/selection_freeze.json").read_bytes()).hexdigest(),
         "replay_of_frozen_choices": True})
    dump(output / "artifacts/historical_opened.json", {"replay_complete": True})
    dump(output / "results/replay_receipt.json", {"complete": True, "new_selection": False,
         "historical_period_previously_exposed": True, "source_data_hash": protocol["data"]["sha256"],
         "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    print("Frozen-recipe replay complete; released results were not overwritten.")


if __name__ == "__main__":
    main()
