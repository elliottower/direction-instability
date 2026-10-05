"""
Replicate noise floor: compute within-cell-line replicate instability (D_rep)
as a noise ceiling, then compare against cross-cell-line direction instability.

Uses Level 5 signatures grouped by (drug, cell, dose, timepoint) — signatures
within such a group are true replicates (same conditions, different plates).
D_rep measures how much a drug's signature rotates due to assay noise alone.
Cross-cell D is only meaningful when it exceeds D_rep.
"""
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

OUTDIR = Path("results/05_replicate_noise")
OUTDIR.mkdir(parents=True, exist_ok=True)

print(f"[{datetime.now():%H:%M:%S}] Loading data...")
data_npz = np.load("data/lincs_subset.npz", allow_pickle=True)
all_sigs = data_npz["signatures"]
all_sig_ids = list(data_npz["sig_ids"])
sig_id_to_idx = {sid: i for i, sid in enumerate(all_sig_ids)}

siginfo = pd.read_csv("data/GSE92742_Broad_LINCS_sig_info.txt.gz", sep="\t", low_memory=False)
cp = siginfo[siginfo["pert_type"] == "trt_cp"].copy()

with open("results/01_cross_cellline/real_results.json") as f:
    exp1 = json.load(f)
exp1_by_name = {r["drug_name"]: r for r in exp1}

print(f"[{datetime.now():%H:%M:%S}] Building replicate groups...")

# Group by (drug, cell, dose, timepoint) — true replicates
groups = cp.groupby(["pert_iname", "cell_id", "pert_dose", "pert_time"])

# For each drug, compute mean within-group replicate instability
drug_rep_instabilities = {}
n_groups_used = 0

for (drug, cell, dose, time), group_df in tqdm(groups, desc="Replicate groups"):
    sids = [sid for sid in group_df["sig_id"] if sid in sig_id_to_idx]
    if len(sids) < 2:
        continue

    idxs = [sig_id_to_idx[sid] for sid in sids]
    sigs = all_sigs[idxs]

    # Normalize
    norms = np.linalg.norm(sigs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    unit_sigs = sigs / norms

    # Pairwise cosine -> D_rep for this group
    K = len(unit_sigs)
    cos_sum = 0.0
    n_pairs = 0
    for i in range(K):
        for j in range(i + 1, K):
            cos_sum += float(np.dot(unit_sigs[i], unit_sigs[j]))
            n_pairs += 1

    mean_cos = cos_sum / n_pairs
    d_rep = 1.0 - mean_cos

    drug_rep_instabilities.setdefault(drug, []).append(d_rep)
    n_groups_used += 1

print(f"[{datetime.now():%H:%M:%S}] {n_groups_used} replicate groups across {len(drug_rep_instabilities)} drugs")

# Compute per-drug mean D_rep (average across all replicate groups for that drug)
drug_d_rep = {}
for drug, reps in drug_rep_instabilities.items():
    drug_d_rep[drug] = float(np.mean(reps))

# Match with cross-cell D
matched_drugs = []
for drug, d_rep in drug_d_rep.items():
    if drug in exp1_by_name:
        cross_d = exp1_by_name[drug]["direction_instability"]
        moa = exp1_by_name[drug].get("moa") or ""
        n_cells = exp1_by_name[drug]["n_cell_lines"]
        matched_drugs.append({
            "drug": drug,
            "d_rep": d_rep,
            "cross_d": cross_d,
            "excess": cross_d - d_rep,
            "ratio": cross_d / max(d_rep, 1e-6),
            "moa": moa,
            "n_cells": n_cells,
            "n_rep_groups": len(drug_rep_instabilities[drug]),
        })

print(f"[{datetime.now():%H:%M:%S}] {len(matched_drugs)} drugs with both D_rep and cross-cell D")

# Summary stats
d_reps = np.array([d["d_rep"] for d in matched_drugs])
cross_ds = np.array([d["cross_d"] for d in matched_drugs])
excesses = np.array([d["excess"] for d in matched_drugs])
ratios = np.array([d["ratio"] for d in matched_drugs])

print(f"\n  D_rep (replicate noise): mean={d_reps.mean():.4f}, median={np.median(d_reps):.4f}")
print(f"  Cross-cell D:            mean={cross_ds.mean():.4f}, median={np.median(cross_ds):.4f}")
print(f"  Excess (cross - rep):    mean={excesses.mean():.4f}, median={np.median(excesses):.4f}")
print(f"  Ratio (cross / rep):     mean={ratios.mean():.2f}, median={np.median(ratios):.2f}")
print(f"  Fraction where cross_D > D_rep: {(excesses > 0).mean():.4f}")

rho, p = spearmanr(d_reps, cross_ds)
print(f"  Spearman(D_rep, cross_D) = {rho:.4f}, p = {p:.2e}")

# HDAC panel
print(f"\n  === HDAC inhibitor panel ===")
hdac_drugs = [d for d in matched_drugs if "HDAC inhibitor" in d["moa"]]
hdac_drugs.sort(key=lambda x: x["cross_d"])
for d in hdac_drugs:
    print(f"  {d['drug']:20s}  D_rep={d['d_rep']:.3f}  cross_D={d['cross_d']:.3f}  "
          f"excess={d['excess']:.3f}  ratio={d['ratio']:.1f}x  ({d['n_rep_groups']} groups)")

# ── Figure ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2))

# Panel A: D_rep vs cross-cell D scatter
ax = axes[0]
is_hdac = np.array(["HDAC inhibitor" in d["moa"] for d in matched_drugs])

ax.scatter(d_reps[~is_hdac], cross_ds[~is_hdac], alpha=0.04, s=2,
           color="#4878CF", rasterized=True)
ax.scatter(d_reps[is_hdac], cross_ds[is_hdac], s=18, color="#C44E52",
           zorder=5, edgecolors="white", linewidths=0.3, label="HDAC inhibitors")

# Identity line
lim = max(d_reps.max(), cross_ds.max()) * 1.05
ax.plot([0, lim], [0, lim], "--", color="#999", linewidth=0.8, alpha=0.5, label="$D = D_{rep}$")

ax.set_xlabel("Replicate instability $D_{rep}$", fontsize=8)
ax.set_ylabel("Cross-cell-line instability $D$", fontsize=8)
ax.set_title(f"(a) Noise floor vs.\ cross-context signal", fontsize=7.5)
ax.legend(fontsize=6, loc="lower right", framealpha=0.9)
ax.tick_params(labelsize=7)

# Panel B: HDAC bar chart — D_rep vs cross_D side by side
ax = axes[1]
hdac_sorted = sorted(hdac_drugs, key=lambda x: x["cross_d"])
y = range(len(hdac_sorted))
w = 0.35
ax.barh([yi - w / 2 for yi in y], [d["d_rep"] for d in hdac_sorted],
        height=w, color="#55A868", label="$D_{rep}$ (noise floor)")
ax.barh([yi + w / 2 for yi in y], [d["cross_d"] for d in hdac_sorted],
        height=w, color="#4878CF", label="$D$ (cross-cell)")
ax.set_yticks(list(y))
ax.set_yticklabels([d["drug"] for d in hdac_sorted], fontsize=5)
ax.set_xlabel("Instability", fontsize=8)
ax.set_title("(b) HDAC: noise floor vs.\ cross-cell", fontsize=7.5)
ax.legend(fontsize=5.5, loc="lower right", framealpha=0.9)
ax.tick_params(axis="x", labelsize=7)
ax.invert_yaxis()

plt.tight_layout(w_pad=1.5)
plt.savefig("paper/fig_replicate_noise.pdf", dpi=300, bbox_inches="tight")
plt.savefig("paper/fig_replicate_noise.png", dpi=200, bbox_inches="tight")
print(f"\n[{datetime.now():%H:%M:%S}] Saved paper/fig_replicate_noise.pdf")

# Save results
result = {
    "n_drugs": len(matched_drugs),
    "n_replicate_groups": n_groups_used,
    "d_rep_mean": float(d_reps.mean()),
    "d_rep_median": float(np.median(d_reps)),
    "cross_d_mean": float(cross_ds.mean()),
    "excess_mean": float(excesses.mean()),
    "ratio_mean": float(ratios.mean()),
    "ratio_median": float(np.median(ratios)),
    "frac_cross_exceeds_rep": float((excesses > 0).mean()),
    "spearman_rho": float(rho),
    "spearman_p": float(p),
    "hdac_panel": [{
        "drug": d["drug"], "d_rep": d["d_rep"], "cross_d": d["cross_d"],
        "excess": d["excess"], "ratio": d["ratio"],
    } for d in hdac_drugs],
}
with open(OUTDIR / "replicate_noise_floor.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"[{datetime.now():%H:%M:%S}] Done.")
