"""Generate all paper figures from pre-computed results."""
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = "paper"

with open("results/01_cross_cellline/real_results.json") as f:
    exp1 = json.load(f)
exp1_by_name = {r["drug_name"]: r for r in exp1}

with open("results/02_invariance/invariance_results.json") as f:
    invariance = json.load(f)

with open("results/03_core_defenses/all_results.json") as f:
    defenses = json.load(f)

with open("results/03_core_defenses/genetic_triangulation.json") as f:
    triangulation = json.load(f)

with open("results/03_core_defenses/holdout_prediction.json") as f:
    holdout = json.load(f)


# ── Figure 1: MOA stratification ────────────────────────────────
def fig_moa_stratification():
    moa_vals = defaultdict(list)
    for d in exp1:
        moa = d.get("moa")
        if not moa:
            continue
        for m in moa.split("|"):
            m = m.strip()
            moa_vals[m].append(d["direction_instability"])

    # Filter to n>=15 and sort by mean
    classes = [(m, vals) for m, vals in moa_vals.items() if len(vals) >= 15]
    classes.sort(key=lambda x: np.mean(x[1]))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    positions = range(len(classes))
    means = [np.mean(v) for _, v in classes]
    stds = [np.std(v) for _, v in classes]
    names = [f"{m} ($n$={len(v)})" for m, v in classes]

    colors = []
    for m, v in classes:
        mean_d = np.mean(v)
        if mean_d < 0.85:
            colors.append("#C44E52")  # red — machinery-targeting
        elif mean_d < 0.92:
            colors.append("#DD8452")  # orange — intermediate
        else:
            colors.append("#4878CF")  # blue — receptor-mediated

    bars = ax.barh(positions, means, xerr=stds, height=0.7,
                   color=colors, edgecolor="white", linewidth=0.3,
                   capsize=2, error_kw={"linewidth": 0.8, "color": "#555"})

    # Population mean line
    pop_mean = np.mean([d["direction_instability"] for d in exp1])
    ax.axvline(pop_mean, color="#333", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(pop_mean + 0.003, len(classes) - 0.5, f"pop. mean\n({pop_mean:.3f})",
            fontsize=5.5, color="#333", va="top")

    ax.set_yticks(list(positions))
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("Direction instability $D$", fontsize=8)
    ax.set_xlim(0.65, 1.02)
    ax.tick_params(axis="x", labelsize=7)
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#C44E52", label="Machinery-targeting"),
        Patch(facecolor="#DD8452", label="Intermediate"),
        Patch(facecolor="#4878CF", label="Receptor-mediated"),
    ]
    ax.legend(handles=legend_elements, fontsize=6, loc="lower right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig_moa_stratification.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{OUTDIR}/fig_moa_stratification.png", dpi=200, bbox_inches="tight")
    print("Saved fig_moa_stratification")
    plt.close()


# ── Figure 2: HDAC selectivity gradient ─────────────────────────
def fig_hdac_gradient():
    hdac_inv = [d for d in invariance if d.get("moa") and "HDAC inhibitor" in str(d["moa"])]

    # Get instability from exp1
    hdac_data = []
    for d in hdac_inv:
        name = d["drug_name"]
        if name not in exp1_by_name:
            continue
        di = exp1_by_name[name]["direction_instability"]
        density = d["thresh_0.50"]["density"]
        n_comps = d["thresh_0.50"]["n_components"]
        n_cells = d["n_cell_lines"]
        hdac_data.append({"drug": name, "di": di, "density": density,
                          "n_comps": n_comps, "n_cells": n_cells})
    hdac_data.sort(key=lambda x: x["di"])

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.0))

    # Panel A: direction instability
    ax = axes[0]
    y = range(len(hdac_data))
    di_vals = [d["di"] for d in hdac_data]
    colors = ["#C44E52" if d["di"] < 0.75 else "#DD8452" if d["di"] < 0.90 else "#4878CF"
              for d in hdac_data]
    ax.barh(y, di_vals, color=colors, height=0.7, edgecolor="white", linewidth=0.3)
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{d['drug']} ($n$={d['n_cells']})" for d in hdac_data], fontsize=5.5)
    ax.set_xlabel("Direction instability $D$", fontsize=8)
    ax.set_title("(a) Instability gradient", fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.tick_params(axis="x", labelsize=7)
    ax.invert_yaxis()

    # Panel B: invariance graph density at threshold 0.5
    ax = axes[1]
    density_vals = [d["density"] for d in hdac_data]
    ax.barh(y, density_vals, color=colors, height=0.7, edgecolor="white", linewidth=0.3)
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{d['n_comps']} comp." for d in hdac_data], fontsize=5.5)
    ax.set_xlabel("Invariance graph density (threshold = 0.5)", fontsize=8)
    ax.set_title("(b) Graph connectivity", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.invert_yaxis()

    plt.tight_layout(w_pad=1.0)
    plt.savefig(f"{OUTDIR}/fig_hdac_gradient.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{OUTDIR}/fig_hdac_gradient.png", dpi=200, bbox_inches="tight")
    print("Saved fig_hdac_gradient")
    plt.close()


# ── Figure 3: Toxicity confound defense ─────────────────────────
def fig_toxicity_defense():
    hdac_rows = defenses["toxicity"]["hdac"]

    # Need to regenerate the scatter from raw data
    # Just load and compute
    cytotoxic_moas = [
        "topoisomerase inhibitor", "DNA alkylating agent",
        "tubulin polymerization inhibitor", "proteasome inhibitor",
        "RNA polymerase inhibitor", "protein synthesis inhibitor",
    ]

    # Load signatures for cytotoxic mean
    data_npz = np.load("data/lincs_subset.npz", allow_pickle=True)
    all_sigs = data_npz["signatures"]
    all_sig_ids = list(data_npz["sig_ids"])
    sig_id_to_idx = {sid: i for i, sid in enumerate(all_sig_ids)}

    import pandas as pd
    siginfo = pd.read_csv("data/GSE92742_Broad_LINCS_sig_info.txt.gz", sep="\t",
                          low_memory=False, compression="infer")

    # Build consensus
    drug_cell_map = {}
    for _, row in siginfo.iterrows():
        if row["pert_type"] != "trt_cp":
            continue
        sid = row["sig_id"]
        if sid not in sig_id_to_idx:
            continue
        name = row["pert_iname"]
        cell = row["cell_id"]
        drug_cell_map.setdefault(name, {}).setdefault(cell, []).append(sig_id_to_idx[sid])

    drug_data = {}
    for drug, cells in drug_cell_map.items():
        if len(cells) < 5:
            continue
        cell_names = sorted(cells.keys())
        sigs = np.array([all_sigs[cells[c]].mean(axis=0) for c in cell_names])
        drug_data[drug] = {"cell_lines": cell_names, "signatures": sigs}

    # Cytotoxic mean
    cyto_sigs = []
    for name, dd in drug_data.items():
        if name not in exp1_by_name:
            continue
        moa = exp1_by_name[name].get("moa") or ""
        if any(m in moa for m in cytotoxic_moas):
            cyto_sigs.append(dd["signatures"].mean(axis=0))

    mean_cyto = np.mean(cyto_sigs, axis=0)
    mean_cyto_unit = mean_cyto / np.linalg.norm(mean_cyto)

    cosines = []
    instabilities = []
    is_cyto = []
    for name, dd in drug_data.items():
        if name not in exp1_by_name:
            continue
        mean_sig = dd["signatures"].mean(axis=0)
        mean_unit = mean_sig / max(np.linalg.norm(mean_sig), 1e-10)
        cosines.append(float(np.dot(mean_unit, mean_cyto_unit)))
        instabilities.append(exp1_by_name[name]["direction_instability"])
        moa = exp1_by_name[name].get("moa") or ""
        is_cyto.append(any(m in moa for m in cytotoxic_moas))

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2))

    # Panel A: scatter
    ax = axes[0]
    non_cyto = [(inst, cos) for inst, cos, c in zip(instabilities, cosines, is_cyto) if not c]
    cyto_pts = [(inst, cos) for inst, cos, c in zip(instabilities, cosines, is_cyto) if c]

    ax.scatter([x[0] for x in non_cyto], [x[1] for x in non_cyto],
               alpha=0.04, s=2, color="#4878CF", rasterized=True)
    ax.scatter([x[0] for x in cyto_pts], [x[1] for x in cyto_pts],
               alpha=0.6, s=12, color="#C44E52", zorder=5, label="Known cytotoxic",
               edgecolors="white", linewidths=0.3)
    from scipy.stats import spearmanr
    nc_inst = [x[0] for x in non_cyto]
    nc_cos = [x[1] for x in non_cyto]
    rho, _ = spearmanr(nc_inst, nc_cos)
    ax.set_xlabel("Direction instability", fontsize=8)
    ax.set_ylabel("Cosine with cytotoxic signature", fontsize=8)
    ax.set_title(f"(a) Stress confound ($\\rho = {rho:.2f}$, excl. cytotoxic)", fontsize=7.5)
    ax.legend(fontsize=6, loc="upper left", framealpha=0.9)
    ax.tick_params(labelsize=7)

    # Panel B: HDAC gradient survives stress-gene removal
    ax = axes[1]
    hdac_sorted = sorted(hdac_rows, key=lambda x: x["di_orig"])
    y = range(len(hdac_sorted))
    w = 0.35
    ax.barh([yi - w/2 for yi in y], [r["di_orig"] for r in hdac_sorted],
            height=w, color="#4878CF", label="Original")
    ax.barh([yi + w/2 for yi in y], [r["di_clean"] for r in hdac_sorted],
            height=w, color="#DD8452", label="Stress genes removed")
    ax.set_yticks(list(y))
    ax.set_yticklabels([r["drug"] for r in hdac_sorted], fontsize=5)
    ax.set_xlabel("Direction instability $D$", fontsize=8)
    rho_preserved = defenses["toxicity"]["h10_gradient_rho"]
    ax.set_title(f"(b) HDAC gradient preserved ($\\rho = {rho_preserved:.2f}$)", fontsize=7.5)
    ax.legend(fontsize=6, loc="lower right", framealpha=0.9)
    ax.tick_params(axis="x", labelsize=7)
    ax.invert_yaxis()

    plt.tight_layout(w_pad=1.5)
    plt.savefig(f"{OUTDIR}/fig_toxicity_defense.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{OUTDIR}/fig_toxicity_defense.png", dpi=200, bbox_inches="tight")
    print("Saved fig_toxicity_defense")
    plt.close()


# ── Figure 4: Genetic triangulation ─────────────────────────────
def fig_genetic_triangulation():
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    dis = [r["di"] for r in triangulation]
    coss = [abs(r["mean_cosine"]) for r in triangulation]

    ax.scatter(dis, coss, alpha=0.12, s=4, color="#4878CF", rasterized=True)

    # Highlight HDAC
    hdac = [r for r in triangulation if exp1_by_name.get(r["drug"], {}).get("moa")
            and "HDAC inhibitor" in str(exp1_by_name[r["drug"]]["moa"])]
    hdac_dis = [r["di"] for r in hdac]
    hdac_cos = [abs(r["mean_cosine"]) for r in hdac]
    ax.scatter(hdac_dis, hdac_cos, s=20, color="#C44E52", zorder=5,
               edgecolors="white", linewidths=0.3, label="HDAC inhibitors")

    # Label a few key drugs (not all — too dense)
    labeled = set()
    for r in sorted(hdac, key=lambda x: -abs(x["mean_cosine"])):
        drug = r["drug"]
        if drug in labeled:
            continue
        if abs(r["mean_cosine"]) > 0.15 or r["di"] > 0.95:
            ax.annotate(f"{drug[:15]}→{r['target']}",
                        (r["di"], abs(r["mean_cosine"])),
                        fontsize=4, alpha=0.7, xytext=(3, 2),
                        textcoords="offset points")
            labeled.add(drug)

    # Highlight proteasome (top cosines)
    proto = [r for r in triangulation if exp1_by_name.get(r["drug"], {}).get("moa")
             and "proteasome" in str(exp1_by_name[r["drug"]]["moa"])]
    if proto:
        ax.scatter([r["di"] for r in proto], [abs(r["mean_cosine"]) for r in proto],
                   s=20, color="#55A868", zorder=5, edgecolors="white", linewidths=0.3,
                   label="Proteasome inhibitors", marker="D")

    from scipy.stats import spearmanr
    rho, p = spearmanr(dis, coss)
    ax.set_xlabel("Direction instability", fontsize=8)
    ax.set_ylabel("|Drug–shRNA cosine|", fontsize=8)
    ax.set_title(f"Genetic triangulation: global $\\rho = {rho:.2f}$, "
                 f"$p = {p:.1e}$", fontsize=8)
    ax.legend(fontsize=6, loc="upper right", framealpha=0.9)
    ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig_genetic_triangulation.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{OUTDIR}/fig_genetic_triangulation.png", dpi=200, bbox_inches="tight")
    print("Saved fig_genetic_triangulation")
    plt.close()


if __name__ == "__main__":
    fig_moa_stratification()
    fig_hdac_gradient()
    fig_toxicity_defense()
    fig_genetic_triangulation()
    print("All figures saved.")
