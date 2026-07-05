"""
Three hole-plugging analyses:
1. Permutation control for AUROC (tautology defense)
2. Signature quality confound (norm vs instability)
3. Jaccard scatter figure
"""
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

OUTDIR = Path("results/04_hole_plugs")
OUTDIR.mkdir(parents=True, exist_ok=True)

with open("results/01_cross_cellline/real_results.json") as f:
    exp1 = json.load(f)
exp1_by_name = {r["drug_name"]: r for r in exp1}

with open("results/03_core_defenses/holdout_prediction.json") as f:
    holdout = json.load(f)


# ── 1. Permutation control for AUROC ────────────────────────────
def permutation_auroc():
    print(f"[{datetime.now():%H:%M:%S}] === Permutation control for AUROC ===")

    real_dis = np.array([r["mean_loo_di"] for r in holdout])
    real_cos = np.array([r["mean_heldout_cosine"] for r in holdout])
    frac_consistent = np.array([r["frac_consistent"] for r in holdout])
    binary_transport = (frac_consistent > 0.5).astype(int)
    neg_di = -real_dis

    real_auroc = roc_auc_score(binary_transport, neg_di)
    real_rho, _ = spearmanr(real_dis, real_cos)
    print(f"  Real AUROC: {real_auroc:.4f}")
    print(f"  Real rho:   {real_rho:.4f}")

    n_perms = 1000
    perm_aurocs = []
    perm_rhos = []
    for _ in tqdm(range(n_perms), desc="Permutations"):
        shuffled_di = np.random.permutation(real_dis)
        perm_rho, _ = spearmanr(shuffled_di, real_cos)
        perm_rhos.append(perm_rho)
        neg_shuffled = -shuffled_di
        try:
            perm_auroc = roc_auc_score(binary_transport, neg_shuffled)
            perm_aurocs.append(perm_auroc)
        except ValueError:
            pass

    perm_aurocs = np.array(perm_aurocs)
    perm_rhos = np.array(perm_rhos)

    print(f"  Permuted AUROC: {perm_aurocs.mean():.4f} +/- {perm_aurocs.std():.4f}")
    print(f"  Permuted rho:   {perm_rhos.mean():.4f} +/- {perm_rhos.std():.4f}")
    print(f"  p(AUROC >= real): {(perm_aurocs >= real_auroc).mean():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))

    ax = axes[0]
    ax.hist(perm_aurocs, bins=40, color="#4878CF", alpha=0.7, density=True, edgecolor="white")
    ax.axvline(real_auroc, color="#C44E52", linewidth=2, label=f"Real ({real_auroc:.3f})")
    ax.set_xlabel("AUROC", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.set_title("(a) Permutation null for AUROC", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    ax = axes[1]
    ax.hist(perm_rhos, bins=40, color="#4878CF", alpha=0.7, density=True, edgecolor="white")
    ax.axvline(real_rho, color="#C44E52", linewidth=2, label=f"Real ({real_rho:.3f})")
    ax.set_xlabel("Spearman $\\rho$", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.set_title("(b) Permutation null for $\\rho$", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig("paper/fig_permutation_control.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("paper/fig_permutation_control.png", dpi=200, bbox_inches="tight")
    print(f"  Saved paper/fig_permutation_control.pdf")

    result = {
        "real_auroc": float(real_auroc), "real_rho": float(real_rho),
        "perm_auroc_mean": float(perm_aurocs.mean()),
        "perm_auroc_std": float(perm_aurocs.std()),
        "perm_rho_mean": float(perm_rhos.mean()),
        "perm_rho_std": float(perm_rhos.std()),
        "p_auroc": float((perm_aurocs >= real_auroc).mean()),
    }
    with open(OUTDIR / "permutation_control.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


# ── 2. Signature quality confound ───────────────────────────────
def quality_confound():
    print(f"\n[{datetime.now():%H:%M:%S}] === Signature quality confound ===")

    dis = []
    norms = []
    for r in exp1:
        dis.append(r["direction_instability"])
        norms.append(r["mean_norm"])

    rho_norm, p_norm = spearmanr(dis, norms)
    print(f"  Spearman(instability, mean_norm) = {rho_norm:.4f}, p = {p_norm:.2e}")

    # Also check if norm predicts holdout cosine as well as instability does
    holdout_dis = [r["mean_loo_di"] for r in holdout]
    holdout_cos = [r["mean_heldout_cosine"] for r in holdout]
    holdout_norms = [exp1_by_name[r["drug"]]["mean_norm"] for r in holdout
                     if r["drug"] in exp1_by_name]
    holdout_cos_matched = [r["mean_heldout_cosine"] for r in holdout
                           if r["drug"] in exp1_by_name]

    rho_norm_holdout, p_norm_holdout = spearmanr(holdout_norms, holdout_cos_matched)
    rho_di_holdout, _ = spearmanr(holdout_dis, holdout_cos)
    print(f"  Spearman(mean_norm, heldout_cos)  = {rho_norm_holdout:.4f}, p = {p_norm_holdout:.2e}")
    print(f"  Spearman(loo_di, heldout_cos)     = {rho_di_holdout:.4f}")
    print(f"  Instability explains {abs(rho_di_holdout)/max(abs(rho_norm_holdout), 0.001):.1f}x more variance than norm")

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))

    ax = axes[0]
    ax.scatter(dis, norms, alpha=0.03, s=2, color="#4878CF", rasterized=True)
    ax.set_xlabel("Direction instability", fontsize=8)
    ax.set_ylabel("Mean signature $L_2$ norm", fontsize=8)
    ax.set_title(f"(a) Instability vs.\ signal strength ($\\rho = {rho_norm:.2f}$)", fontsize=7.5)
    ax.tick_params(labelsize=7)

    ax = axes[1]
    ax.scatter(holdout_norms, holdout_cos_matched, alpha=0.03, s=2, color="#999", rasterized=True, label=f"Norm ($\\rho={rho_norm_holdout:.2f}$)")
    ax.scatter(holdout_dis, holdout_cos, alpha=0.03, s=2, color="#4878CF", rasterized=True, label=f"Instability ($\\rho={rho_di_holdout:.2f}$)")
    ax.set_xlabel("Predictor value", fontsize=8)
    ax.set_ylabel("Mean held-out cosine", fontsize=8)
    ax.set_title("(b) Norm vs.\ instability as predictors", fontsize=7.5)
    ax.legend(fontsize=6, markerscale=5)
    ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig("paper/fig_quality_confound.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("paper/fig_quality_confound.png", dpi=200, bbox_inches="tight")
    print(f"  Saved paper/fig_quality_confound.pdf")

    result = {
        "rho_instability_norm": float(rho_norm), "p_instability_norm": float(p_norm),
        "rho_norm_holdout": float(rho_norm_holdout), "p_norm_holdout": float(p_norm_holdout),
        "rho_di_holdout": float(rho_di_holdout),
    }
    with open(OUTDIR / "quality_confound.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


# ── 3. Jaccard scatter ──────────────────────────────────────────
def jaccard_scatter():
    print(f"\n[{datetime.now():%H:%M:%S}] === Jaccard scatter ===")

    dis = [r["direction_instability"] for r in exp1]
    jaccards = [r["mean_top_gene_jaccard"] for r in exp1]
    moas = [r.get("moa") or "" for r in exp1]

    rho, p = spearmanr(dis, jaccards)
    print(f"  Spearman(instability, Jaccard) = {rho:.4f}, p = {p:.2e}")

    fig, ax = plt.subplots(figsize=(5.0, 4.0))

    # Color by MOA category
    colors = []
    for moa in moas:
        if "HDAC inhibitor" in moa:
            colors.append("#C44E52")
        elif any(m in moa for m in ["topoisomerase", "proteasome", "CDK inhibitor"]):
            colors.append("#DD8452")
        else:
            colors.append("#4878CF")

    ax.scatter(dis, jaccards, alpha=0.06, s=3, c=colors, rasterized=True)

    # Overlay HDAC
    hdac_idx = [i for i, m in enumerate(moas) if "HDAC inhibitor" in m]
    ax.scatter([dis[i] for i in hdac_idx], [jaccards[i] for i in hdac_idx],
               s=18, color="#C44E52", edgecolors="white", linewidths=0.3,
               zorder=5, label="HDAC inhibitors")

    ax.set_xlabel("Direction instability $D$", fontsize=9)
    ax.set_ylabel("Mean top-gene Jaccard overlap", fontsize=9)
    ax.set_title(f"Direction instability vs.\ gene-level consistency "
                 f"($\\rho = {rho:.2f}$)", fontsize=9)
    ax.legend(fontsize=7, framealpha=0.9)
    ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig("paper/fig_jaccard_scatter.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("paper/fig_jaccard_scatter.png", dpi=200, bbox_inches="tight")
    print(f"  Saved paper/fig_jaccard_scatter.pdf")

    result = {"rho": float(rho), "p": float(p)}
    with open(OUTDIR / "jaccard_scatter.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    r1 = permutation_auroc()
    r2 = quality_confound()
    r3 = jaccard_scatter()
    print(f"\n[{datetime.now():%H:%M:%S}] All done.")
