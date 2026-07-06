"""
Geometric validity tests for direction instability.

Six tests that break cosine circularity at different points:
1. Cross-metric held-out prediction (Pearson, L1, Jaccard outcomes)
2. Target expression breadth (CCLE, no perturbation data)
3. Drug sensitivity concordance (PRISM viability CV)
4. Synthetic null with matched geometry
5. Split-half subspace stability
6. MOA classification (k-NN on DI + magnitude CV)

Pre-registration: PREREGISTRATION_VALIDITY.md
"""
import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm


def load_drug_results(path):
    with open(path) as f:
        results = json.load(f)
    return {r["drug_name"]: r for r in results}


def build_consensus(siginfo, sig_id_to_idx, all_sigs, min_cells=5):
    drug_cell_map = {}
    for _, row in tqdm(siginfo.iterrows(), total=len(siginfo), desc="Indexing drugs"):
        if row["pert_type"] != "trt_cp":
            continue
        sid = row["sig_id"]
        if sid not in sig_id_to_idx:
            continue
        name = row["pert_iname"]
        cell = row["cell_id"]
        drug_cell_map.setdefault(name, {}).setdefault(cell, []).append(sig_id_to_idx[sid])

    result = {}
    for drug, cells in drug_cell_map.items():
        if len(cells) < min_cells:
            continue
        cell_names = sorted(cells.keys())
        sigs = np.array([all_sigs[cells[c]].mean(axis=0) for c in cell_names])
        result[drug] = {"cell_lines": cell_names, "signatures": sigs}
    return result


def magnitude_correct_di(drug_results):
    names = sorted(drug_results.keys())
    di_raw = np.array([drug_results[n]["direction_instability"] for n in names])
    norms = np.array([drug_results[n]["mean_norm"] for n in names])
    A = np.column_stack([norms, np.ones(len(norms))])
    coeffs, _, _, _ = np.linalg.lstsq(A, di_raw, rcond=None)
    residuals = di_raw - A @ coeffs
    di_corrected = residuals + coeffs[1]
    return {n: float(di_corrected[i]) for i, n in enumerate(names)}


def pairwise_cosine_matrix(sigs):
    norms = np.linalg.norm(sigs, axis=1, keepdims=True)
    unit = sigs / np.maximum(norms, 1e-10)
    return unit @ unit.T


# ── Test 1: Cross-metric held-out prediction ────────────────────
def run_cross_metric_holdout(drug_data, drug_results, di_corrected, outdir, min_cells=10):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{ts}] === Test 1: Cross-metric held-out prediction ===")

    results_per_drug = []
    for name, dd in tqdm(drug_data.items(), desc="Cross-metric LOO"):
        if name not in drug_results or name not in di_corrected:
            continue
        sigs = dd["signatures"]
        n = sigs.shape[0]
        if n < min_cells:
            continue

        pearson_vals = []
        l1_vals = []
        jaccard_vals = []
        for i in range(n):
            train = np.delete(sigs, i, axis=0)
            held = sigs[i]
            consensus = train.mean(axis=0)

            pearson_vals.append(float(np.corrcoef(held, consensus)[0, 1]))

            l1_vals.append(float(np.sum(np.abs(held - consensus))))

            held_up = set(np.argsort(held)[-50:])
            held_down = set(np.argsort(held)[:50])
            cons_up = set(np.argsort(consensus)[-50:])
            cons_down = set(np.argsort(consensus)[:50])
            j_up = len(held_up & cons_up) / len(held_up | cons_up)
            j_down = len(held_down & cons_down) / len(held_down | cons_down)
            jaccard_vals.append((j_up + j_down) / 2)

        results_per_drug.append({
            "drug": name,
            "di_corrected": di_corrected[name],
            "mean_pearson": float(np.mean(pearson_vals)),
            "mean_l1": float(np.mean(l1_vals)),
            "mean_jaccard": float(np.mean(jaccard_vals)),
        })

    dis = [r["di_corrected"] for r in results_per_drug]
    pearsons = [r["mean_pearson"] for r in results_per_drug]
    l1s = [r["mean_l1"] for r in results_per_drug]
    jaccards = [r["mean_jaccard"] for r in results_per_drug]

    rho_pearson, p_pearson = spearmanr(dis, pearsons)
    rho_l1, p_l1 = spearmanr(dis, l1s)
    rho_jaccard, p_jaccard = spearmanr(dis, jaccards)

    binary_pearson = [1 if p > 0.3 else 0 for p in pearsons]
    binary_jaccard = [1 if j > 0.3 else 0 for j in jaccards]
    neg_di = [-d for d in dis]

    auroc_pearson = roc_auc_score(binary_pearson, neg_di) if len(set(binary_pearson)) > 1 else None
    auroc_jaccard = roc_auc_score(binary_jaccard, neg_di) if len(set(binary_jaccard)) > 1 else None

    print(f"  N drugs: {len(results_per_drug)}")
    print(f"  DI vs Pearson:  rho = {rho_pearson:.4f}, p = {p_pearson:.2e}, AUROC = {auroc_pearson}")
    print(f"  DI vs L1:       rho = {rho_l1:.4f}, p = {p_l1:.2e}")
    print(f"  DI vs Jaccard:  rho = {rho_jaccard:.4f}, p = {p_jaccard:.2e}, AUROC = {auroc_jaccard}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].scatter(dis, pearsons, alpha=0.03, s=3, color="steelblue")
    axes[0].set_xlabel("DI (magnitude-corrected)")
    axes[0].set_ylabel("Mean held-out Pearson r")
    axes[0].set_title(f"Pearson (rho={rho_pearson:.3f})")

    axes[1].scatter(dis, l1s, alpha=0.03, s=3, color="darkorange")
    axes[1].set_xlabel("DI (magnitude-corrected)")
    axes[1].set_ylabel("Mean held-out L1 distance")
    axes[1].set_title(f"L1 (rho={rho_l1:.3f})")

    axes[2].scatter(dis, jaccards, alpha=0.03, s=3, color="forestgreen")
    axes[2].set_xlabel("DI (magnitude-corrected)")
    axes[2].set_ylabel("Mean held-out Jaccard")
    axes[2].set_title(f"Jaccard (rho={rho_jaccard:.3f})")

    plt.tight_layout()
    plt.savefig(outdir / "test1_cross_metric.png", dpi=150, bbox_inches="tight")

    return {
        "test": "cross_metric_holdout",
        "n_drugs": len(results_per_drug),
        "rho_pearson": float(rho_pearson), "p_pearson": float(p_pearson),
        "auroc_pearson": float(auroc_pearson) if auroc_pearson else None,
        "rho_l1": float(rho_l1), "p_l1": float(p_l1),
        "rho_jaccard": float(rho_jaccard), "p_jaccard": float(p_jaccard),
        "auroc_jaccard": float(auroc_jaccard) if auroc_jaccard else None,
        "per_drug": results_per_drug,
    }


# ── Test 2: Target expression breadth ───────────────────────────
def run_target_breadth(drug_results, di_corrected, repurposing_hub_path, ccle_path, outdir):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{ts}] === Test 2: Target expression breadth ===")

    rh = pd.read_csv(repurposing_hub_path, sep="\t", skiprows=9)
    single_target = rh[rh["target"].notna()].copy()
    single_target = single_target[~single_target["target"].str.contains(r"\|", na=False)]
    drug_to_target = dict(zip(single_target["pert_iname"], single_target["target"]))
    print(f"  Drugs with single annotated target: {len(drug_to_target)}")

    print(f"  Loading CCLE expression data...")
    ccle = pd.read_csv(ccle_path, index_col=0)
    gene_name_to_col = {}
    for col in ccle.columns:
        gene_name = col.split(" (")[0] if " (" in col else col
        gene_name_to_col[gene_name] = col
    print(f"  CCLE: {ccle.shape[0]} cell lines, {ccle.shape[1]} genes")

    cell_medians = ccle.median(axis=1)

    matched = []
    for drug, target in drug_to_target.items():
        if drug not in di_corrected:
            continue
        if target not in gene_name_to_col:
            continue
        col = gene_name_to_col[target]
        expr = ccle[col]
        breadth = float((expr > cell_medians).mean())
        matched.append({
            "drug": drug,
            "target": target,
            "expression_breadth": breadth,
            "di_corrected": di_corrected[drug],
            "di_raw": drug_results[drug]["direction_instability"],
        })

    print(f"  Matched drugs: {len(matched)}")

    if len(matched) < 10:
        print("  Too few matched drugs for meaningful test")
        return {"test": "target_breadth", "n_matched": len(matched), "status": "insufficient_data"}

    dis = [m["di_corrected"] for m in matched]
    breadths = [m["expression_breadth"] for m in matched]
    rho, p = spearmanr(dis, breadths)
    print(f"  Spearman(DI, breadth) = {rho:.4f}, p = {p:.2e}")
    print(f"  Pass criterion: rho < -0.10 and p < 0.05 -> {'PASS' if rho < -0.10 and p < 0.05 else 'FAIL'}")

    hdac_targets = {"HDAC1", "HDAC2", "HDAC3", "HDAC4", "HDAC5", "HDAC6", "HDAC7", "HDAC8", "HDAC9", "HDAC10", "HDAC11"}
    hdac_breadths = {}
    for target in hdac_targets:
        if target in gene_name_to_col:
            col = gene_name_to_col[target]
            expr = ccle[col]
            breadth = float((expr > cell_medians).mean())
            hdac_breadths[target] = breadth
            print(f"    {target}: expressed in {100*breadth:.1f}% of cell lines")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(breadths, dis, alpha=0.15, s=8, color="steelblue")
    ax.set_xlabel("Target expression breadth (fraction of cell lines)")
    ax.set_ylabel("DI (magnitude-corrected)")
    ax.set_title(f"Test 2: Target breadth vs DI (rho={rho:.3f}, n={len(matched)})")
    plt.tight_layout()
    plt.savefig(outdir / "test2_target_breadth.png", dpi=150, bbox_inches="tight")

    return {
        "test": "target_breadth",
        "n_matched": len(matched),
        "rho": float(rho), "p": float(p),
        "hdac_breadths": hdac_breadths,
        "per_drug": matched,
    }


# ── Test 3: Drug sensitivity concordance (PRISM) ────────────────
def run_prism_concordance(drug_results, di_corrected, prism_cv_path, outdir):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{ts}] === Test 3: Drug sensitivity concordance (PRISM) ===")

    prism_cv = pd.read_csv(prism_cv_path)
    print(f"  PRISM drugs: {len(prism_cv)}")

    prism_by_name = {}
    for _, row in prism_cv.iterrows():
        name = row["drug_name"].lower().replace("-", "").replace(" ", "")
        prism_by_name[name] = row

    matched = []
    for drug_name, di in di_corrected.items():
        normalized = drug_name.lower().replace("-", "").replace(" ", "")
        if normalized in prism_by_name:
            row = prism_by_name[normalized]
            matched.append({
                "drug": drug_name,
                "di_corrected": di,
                "di_raw": drug_results[drug_name]["direction_instability"],
                "viability_cv": float(row["viability_cv"]),
                "mean_lfc": float(row["mean_lfc"]),
                "n_prism_cells": int(row["n_cell_lines"]),
            })

    print(f"  Matched LINCS-PRISM drugs: {len(matched)}")

    if len(matched) < 10:
        print("  Too few matched drugs")
        return {"test": "prism_concordance", "n_matched": len(matched), "status": "insufficient_data"}

    dis = [m["di_corrected"] for m in matched]
    cvs = [m["viability_cv"] for m in matched]
    rho, p = spearmanr(dis, cvs)
    print(f"  Spearman(DI, viability_CV) = {rho:.4f}, p = {p:.2e}")
    print(f"  Pass criterion: rho > 0.10 and p < 0.05 -> {'PASS' if rho > 0.10 and p < 0.05 else 'FAIL'}")

    active_matched = [m for m in matched if abs(m["mean_lfc"]) >= 0.1]
    if len(active_matched) >= 10:
        dis_active = [m["di_corrected"] for m in active_matched]
        cvs_active = [m["viability_cv"] for m in active_matched]
        rho_active, p_active = spearmanr(dis_active, cvs_active)
        print(f"  Active drugs (|LFC| >= 0.1): n={len(active_matched)}, rho={rho_active:.4f}, p={p_active:.2e}")
    else:
        rho_active, p_active = None, None

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(dis, cvs, alpha=0.3, s=10, color="steelblue")
    ax.set_xlabel("LINCS DI (magnitude-corrected)")
    ax.set_ylabel("PRISM viability CV")
    ax.set_title(f"Test 3: PRISM concordance (rho={rho:.3f}, n={len(matched)})")
    plt.tight_layout()
    plt.savefig(outdir / "test3_prism_concordance.png", dpi=150, bbox_inches="tight")

    return {
        "test": "prism_concordance",
        "n_matched": len(matched),
        "rho": float(rho), "p": float(p),
        "rho_active": float(rho_active) if rho_active is not None else None,
        "p_active": float(p_active) if p_active is not None else None,
        "n_active": len(active_matched),
        "per_drug": matched,
    }


# ── Test 4: Synthetic null with matched geometry ─────────────────
def run_synthetic_null(drug_data, drug_results, di_corrected, all_sigs, outdir, n_repeats=5, min_cells=10):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{ts}] === Test 4: Synthetic null with matched geometry ===")

    rng = np.random.default_rng(42)
    eligible = {name: dd for name, dd in drug_data.items()
                if name in di_corrected and dd["signatures"].shape[0] >= min_cells}
    print(f"  Eligible drugs: {len(eligible)}")

    all_sig_norms = np.linalg.norm(all_sigs, axis=1)
    n_total = all_sigs.shape[0]

    aurocs = []
    for rep in range(n_repeats):
        print(f"  Repeat {rep+1}/{n_repeats}...")
        synth_results = []
        for name, dd in tqdm(eligible.items(), desc=f"Synthetic rep {rep+1}", leave=False):
            K = dd["signatures"].shape[0]
            real_norms = np.linalg.norm(dd["signatures"], axis=1)
            mean_norm = real_norms.mean()
            std_norm = max(real_norms.std(), 1e-6)

            sampled_idxs = rng.choice(n_total, size=K * 10, replace=False)
            sampled_norms = all_sig_norms[sampled_idxs]
            norm_scores = np.abs(sampled_norms - mean_norm)
            best_idxs = sampled_idxs[np.argsort(norm_scores)[:K]]
            synth_sigs = all_sigs[best_idxs]

            cos_mat = pairwise_cosine_matrix(synth_sigs)
            triu = cos_mat[np.triu_indices_from(cos_mat, k=1)]
            synth_di = 1.0 - float(triu.mean())

            heldout_cosines = []
            for i in range(K):
                train = np.delete(synth_sigs, i, axis=0)
                held = synth_sigs[i]
                consensus = train.mean(axis=0)
                cos = float(np.dot(held, consensus) /
                            (np.linalg.norm(held) * np.linalg.norm(consensus) + 1e-10))
                heldout_cosines.append(cos)

            synth_results.append({
                "di": synth_di,
                "mean_heldout_cos": float(np.mean(heldout_cosines)),
                "frac_consistent": float(np.mean([c > 0.3 for c in heldout_cosines])),
            })

        dis_synth = [r["di"] for r in synth_results]
        frac_synth = [r["frac_consistent"] for r in synth_results]
        binary_synth = [1 if f > 0.5 else 0 for f in frac_synth]
        neg_di_synth = [-d for d in dis_synth]

        if len(set(binary_synth)) > 1:
            auroc = roc_auc_score(binary_synth, neg_di_synth)
        else:
            auroc = 0.5
        aurocs.append(auroc)
        print(f"    AUROC = {auroc:.4f}")

    mean_auroc = float(np.mean(aurocs))
    std_auroc = float(np.std(aurocs))
    print(f"  Synthetic AUROC: {mean_auroc:.4f} +/- {std_auroc:.4f}")
    print(f"  Real AUROC: 0.986")
    print(f"  Pass criterion: synthetic AUROC < 0.60 -> {'PASS' if mean_auroc < 0.60 else 'FAIL'}")

    return {
        "test": "synthetic_null",
        "n_drugs": len(eligible),
        "n_repeats": n_repeats,
        "aurocs": [float(a) for a in aurocs],
        "mean_auroc": mean_auroc,
        "std_auroc": std_auroc,
        "real_auroc": 0.986,
    }


# ── Test 5: Split-half subspace stability ────────────────────────
def run_split_half(drug_data, drug_results, outdir, n_splits=100, min_cells=5):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{ts}] === Test 5: Split-half subspace stability ===")

    rng = np.random.default_rng(42)
    eligible = {name: dd for name, dd in drug_data.items()
                if name in drug_results and dd["signatures"].shape[0] >= min_cells}
    n_genes = next(iter(eligible.values()))["signatures"].shape[1]
    print(f"  Eligible drugs: {len(eligible)}, genes: {n_genes}")

    drug_names = sorted(eligible.keys())
    rhos = []
    for split_i in tqdm(range(n_splits), desc="Split-half"):
        perm = rng.permutation(n_genes)
        half1 = perm[:n_genes // 2]
        half2 = perm[n_genes // 2: 2 * (n_genes // 2)]

        di_half1 = []
        di_half2 = []
        for name in drug_names:
            sigs = eligible[name]["signatures"]

            sigs1 = sigs[:, half1]
            cos1 = pairwise_cosine_matrix(sigs1)
            triu1 = cos1[np.triu_indices_from(cos1, k=1)]
            di_half1.append(1.0 - float(triu1.mean()))

            sigs2 = sigs[:, half2]
            cos2 = pairwise_cosine_matrix(sigs2)
            triu2 = cos2[np.triu_indices_from(cos2, k=1)]
            di_half2.append(1.0 - float(triu2.mean()))

        rho, _ = spearmanr(di_half1, di_half2)
        rhos.append(float(rho))

    mean_rho = float(np.mean(rhos))
    ci_low = float(np.percentile(rhos, 2.5))
    ci_high = float(np.percentile(rhos, 97.5))
    print(f"  Mean rho: {mean_rho:.4f} (95% CI [{ci_low:.4f}, {ci_high:.4f}])")
    print(f"  Pass criterion: mean > 0.70, CI lower > 0.60 -> "
          f"{'PASS' if mean_rho > 0.70 and ci_low > 0.60 else 'FAIL'}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rhos, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(mean_rho, color="crimson", linewidth=2, label=f"Mean = {mean_rho:.3f}")
    ax.axvline(ci_low, color="crimson", linewidth=1, linestyle="--", label=f"95% CI [{ci_low:.3f}, {ci_high:.3f}]")
    ax.axvline(ci_high, color="crimson", linewidth=1, linestyle="--")
    ax.set_xlabel("Split-half Spearman rho")
    ax.set_ylabel("Count")
    ax.set_title(f"Test 5: Split-half stability ({n_splits} splits)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outdir / "test5_split_half.png", dpi=150, bbox_inches="tight")

    return {
        "test": "split_half",
        "n_drugs": len(eligible),
        "n_genes": n_genes,
        "n_splits": n_splits,
        "mean_rho": mean_rho,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "all_rhos": rhos,
    }


# ── Test 6: MOA classification ──────────────────────────────────
def run_moa_classification(drug_results, di_corrected, outdir, min_class_size=15, k=5, n_permutations=1000):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{ts}] === Test 6: MOA classification ===")

    moa_drugs = {}
    for name, r in drug_results.items():
        if name not in di_corrected:
            continue
        moa = r.get("moa")
        if not moa or moa == "nan":
            continue
        primary_moa = str(moa).split("|")[0].strip()
        moa_drugs.setdefault(primary_moa, []).append(name)

    large_moas = {m: drugs for m, drugs in moa_drugs.items() if len(drugs) >= min_class_size}
    print(f"  MOA families with >= {min_class_size} drugs: {len(large_moas)}")
    for m, drugs in sorted(large_moas.items(), key=lambda x: -len(x[1])):
        print(f"    {m}: {len(drugs)} drugs")

    all_drugs = []
    all_labels = []
    for moa, drugs in large_moas.items():
        for d in drugs:
            all_drugs.append(d)
            all_labels.append(moa)

    X = np.array([[di_corrected[d], drug_results[d]["magnitude_cv"]] for d in all_drugs])
    y = np.array(all_labels)
    n = len(y)
    print(f"  Total drugs: {n}, classes: {len(large_moas)}")

    correct = 0
    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        X_test = X[i:i+1]
        clf = KNeighborsClassifier(n_neighbors=min(k, len(X_train)))
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)[0]
        if pred == y[i]:
            correct += 1

    accuracy = correct / n
    most_freq = Counter(y).most_common(1)[0][1] / n
    print(f"  LOO accuracy: {100*accuracy:.1f}%")
    print(f"  Baseline (most frequent): {100*most_freq:.1f}%")

    rng = np.random.default_rng(42)
    perm_accs = []
    for _ in tqdm(range(n_permutations), desc="Permutation test"):
        y_perm = rng.permutation(y)
        correct_perm = 0
        for i in range(n):
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y_perm, i)
            X_test = X[i:i+1]
            clf = KNeighborsClassifier(n_neighbors=min(k, len(X_train)))
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)[0]
            if pred == y_perm[i]:
                correct_perm += 1
        perm_accs.append(correct_perm / n)

    mean_perm = float(np.mean(perm_accs))
    std_perm = float(np.std(perm_accs))
    p_val = float(np.mean([a >= accuracy for a in perm_accs]))
    print(f"  Permutation: {100*mean_perm:.1f} +/- {100*std_perm:.1f}%")
    print(f"  p-value (permutation): {p_val:.4f}")
    print(f"  Pass criterion: accuracy > baseline + 2*SD -> "
          f"{'PASS' if accuracy > most_freq + 2 * std_perm else 'FAIL'}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(perm_accs, bins=30, color="gray", edgecolor="white", alpha=0.7, label="Permutation null")
    ax.axvline(accuracy, color="crimson", linewidth=2, label=f"Real = {100*accuracy:.1f}%")
    ax.axvline(most_freq, color="orange", linewidth=2, linestyle="--", label=f"Baseline = {100*most_freq:.1f}%")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Count")
    ax.set_title(f"Test 6: MOA classification (k-NN, k={k})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outdir / "test6_moa_classification.png", dpi=150, bbox_inches="tight")

    return {
        "test": "moa_classification",
        "n_drugs": n,
        "n_classes": len(large_moas),
        "k": k,
        "accuracy": float(accuracy),
        "baseline_accuracy": float(most_freq),
        "perm_mean": mean_perm,
        "perm_std": std_perm,
        "p_value": p_val,
        "class_sizes": {m: len(drugs) for m, drugs in large_moas.items()},
    }


# ── Main ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Geometric validity tests for DI")
    parser.add_argument("--data", default="data/lincs_subset.npz")
    parser.add_argument("--metadata", default="data/GSE92742_Broad_LINCS_sig_info.txt.gz")
    parser.add_argument("--drug-results", default="zenodo_v1/drug_instability_results.json")
    parser.add_argument("--repurposing-hub", default="data/repurposing_hub_drugs.txt")
    parser.add_argument("--ccle-expression",
                        default="../direction-instability-atlas/data/depmap/OmicsExpressionProteinCodingGenesTPMLogp1.csv")
    parser.add_argument("--prism-cv", default="../direction-instability-atlas/results/prism_cv.csv")
    parser.add_argument("--outdir", default="results/06_geometric_validity")
    parser.add_argument("--tests", nargs="+", default=["1", "2", "3", "4", "5", "6"],
                        help="Which tests to run (1-6)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] Loading drug results...")
    drug_results = load_drug_results(args.drug_results)
    di_corrected = magnitude_correct_di(drug_results)
    print(f"  {len(drug_results)} drugs, {len(di_corrected)} with corrected DI")

    need_sigs = any(t in args.tests for t in ["1", "4", "5"])
    drug_data = None
    all_sigs = None

    if need_sigs:
        print(f"[{datetime.now():%H:%M:%S}] Loading signatures...")
        npz = np.load(args.data, allow_pickle=True)
        all_sigs = npz["signatures"]
        all_sig_ids = list(npz["sig_ids"])
        sig_id_to_idx = {sid: i for i, sid in enumerate(all_sig_ids)}
        print(f"  Signatures: {all_sigs.shape}")

        siginfo = pd.read_csv(args.metadata, sep="\t", low_memory=False, compression="infer")
        drug_data = build_consensus(siginfo, sig_id_to_idx, all_sigs)
        print(f"  Drugs with >= 5 cell lines: {len(drug_data)}")

    all_results = {}

    if "1" in args.tests:
        all_results["test1"] = run_cross_metric_holdout(drug_data, drug_results, di_corrected, outdir)

    if "2" in args.tests:
        all_results["test2"] = run_target_breadth(
            drug_results, di_corrected, args.repurposing_hub, args.ccle_expression, outdir)

    if "3" in args.tests:
        all_results["test3"] = run_prism_concordance(drug_results, di_corrected, args.prism_cv, outdir)

    if "4" in args.tests:
        all_results["test4"] = run_synthetic_null(drug_data, drug_results, di_corrected, all_sigs, outdir)

    if "5" in args.tests:
        all_results["test5"] = run_split_half(drug_data, drug_results, outdir)

    if "6" in args.tests:
        all_results["test6"] = run_moa_classification(drug_results, di_corrected, outdir)

    summary_path = outdir / "all_validity_results.json"
    serializable = {}
    for k, v in all_results.items():
        clean = {kk: vv for kk, vv in v.items() if kk != "per_drug"}
        serializable[k] = clean

    with open(summary_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\n[{datetime.now():%H:%M:%S}] All results saved to {summary_path}")

    for k, v in all_results.items():
        if "per_drug" in v:
            detail_path = outdir / f"{k}_per_drug.json"
            with open(detail_path, "w") as f:
                json.dump(v["per_drug"], f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for k, v in sorted(all_results.items()):
        name = v.get("test", k)
        if k == "test1":
            print(f"  1. Cross-metric: Pearson rho={v['rho_pearson']:.3f}, "
                  f"Jaccard rho={v['rho_jaccard']:.3f}")
        elif k == "test2":
            print(f"  2. Target breadth: rho={v.get('rho', 'N/A')}, n={v.get('n_matched', 'N/A')}")
        elif k == "test3":
            print(f"  3. PRISM concordance: rho={v.get('rho', 'N/A')}, n={v.get('n_matched', 'N/A')}")
        elif k == "test4":
            print(f"  4. Synthetic null: AUROC={v['mean_auroc']:.3f} (real=0.986)")
        elif k == "test5":
            print(f"  5. Split-half: rho={v['mean_rho']:.3f} "
                  f"[{v['ci_low']:.3f}, {v['ci_high']:.3f}]")
        elif k == "test6":
            print(f"  6. MOA classification: {100*v['accuracy']:.1f}% "
                  f"(baseline={100*v['baseline_accuracy']:.1f}%, "
                  f"perm={100*v['perm_mean']:.1f}%)")


if __name__ == "__main__":
    main()
