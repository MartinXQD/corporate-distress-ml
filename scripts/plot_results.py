"""Render presentation figures from saved aggregates without model fitting."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.titleweight": "bold"})
annual = pd.read_csv(ROOT / "results/historical_by_year.csv")
series = [("P2_fixed_rank_blend", "Frozen rank blend", "#15616d", 2.8),
          ("P2_seed_mean__refine_r3_21_bbf84fb1", "Selected MLP", "#dd8452", 1.8),
          ("P2_seed_mean__refine_r3_15_a028aa7a", "Selected XGBoost", "#758abd", 1.8),
          ("P2_seed_mean__baseline_v1_lags_xgb", "Fixed lag-XGBoost baseline", "#6c757d", 1.8)]
fig, ax = plt.subplots(figsize=(10.5, 5.2), layout="constrained")
for key, label, color, width in series:
    part = annual[annual.name == key].sort_values("year")
    ax.plot(part.year, part.AP, marker="o", label=label, color=color, linewidth=width)
ax.set(title="Historical ranking quality by year", xlabel="Evaluation year", ylabel="Average precision (AP)",
       xticks=[2015, 2016, 2017, 2018], ylim=(0, .65))
ax.grid(axis="y", alpha=.18)
ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=9)
fig.supxlabel("Previously inspected historical period · frozen recipes fitted through 2014 · five-seed prediction ensembles", fontsize=9, color="#555555")
fig.savefig(ROOT / "results/figures/annual_comparison.png", dpi=180)
plt.close(fig)

controls = pd.read_csv(ROOT / "results/optimization_controls.csv")
part = controls[(controls.factor == "half_life") & (controls.value.astype(str).isin(["3", "3.0"]))]
fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
for i, family in enumerate(["xgboost", "mlp"]):
    row = part[part.family == family].iloc[0]
    ax.plot([float(row.parent_AP), float(row.AP)], [i, i], color="#bac4ca", linewidth=5, zorder=1)
    ax.scatter([float(row.parent_AP)], [i], color="#758abd", s=85, zorder=2, label="No temporal decay" if i == 0 else None)
    ax.scatter([float(row.AP)], [i], color="#15616d", s=85, zorder=2, label="Three-year half-life" if i == 0 else None)
    ax.annotate(f"+{row.delta_AP:.3f} AP", (float(row.AP)+.012, i), va="center", color="#15616d", weight="bold")
ax.set(title="Recency weighting helped both selected backbones", xlabel="Mean annual development AP",
       yticks=[0, 1], yticklabels=["XGBoost", "PyTorch MLP"], xlim=(.32, .64), ylim=(-.6, 1.65))
ax.grid(axis="x", alpha=.18)
ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=9)
fig.supxlabel("Matched configured factor · same three development years and seed 42 · conditional mechanism evidence", fontsize=9, color="#555555")
fig.savefig(ROOT / "results/figures/time_weighting.png", dpi=180)
plt.close(fig)
print("Two figures rendered from unchanged aggregate tables.")
