"""Load raw accounting records with a hard development-label boundary."""
from pathlib import Path
import os
import numpy as np
import pandas as pd
from .common import ROOT, sha, protocol

RAW = [f"X{i}" for i in range(1,19)]

def load_data(role="development", path=None):
    path = Path(path or os.environ.get("DISTRESS_DATA", ROOT.parent / "american_bankruptcy.csv"))
    expected = protocol()["implementation"]["source_data_hash"]
    if sha(path) != expected:
        raise ValueError("Raw data hash does not match the registered source")
    # Parse only financial inputs first. Development never loads historical labels.
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    x = pd.read_csv(path, usecols=[c for c in columns if c != "status_label"])
    x["row_id"] = np.arange(len(x), dtype=int)
    if role == "development":
        keep = x.year <= 2012
    elif role == "pre_evaluation":
        keep = x.year <= 2014
    elif role == "historical":
        if not (ROOT / "configs/final_freeze.json").exists():
            raise RuntimeError("Historical labels require final freeze")
        keep = np.ones(len(x), dtype=bool)
    elif role == "audit":
        keep = np.ones(len(x), dtype=bool)
    else:
        raise ValueError(role)
    # skiprows prevents parsing labels outside the permitted period.
    row_indices = x.index[keep].to_numpy()
    skip = set((x.index[~keep] + 1).tolist())
    labels = pd.read_csv(path, usecols=["status_label"], skiprows=lambda i: i in skip)
    x = x.loc[row_indices].copy()
    assert len(labels) == len(x)
    assert set(labels.status_label.unique()) <= {"alive", "failed"}
    x["y"] = labels.status_label.eq("failed").astype("int8").to_numpy()
    assert not x.duplicated(["company_name", "year"]).any()
    assert np.isfinite(x[RAW]).all().all()
    return x.reset_index(drop=True)
