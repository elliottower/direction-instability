"""
Experiments 03-09: Comprehensive analysis suite.

Runs all pre-registered analyses on existing LINCS data:
- 03: Toxicity confound check
- 04: Component-tissue enrichment
- 05: Cell-line similarity matrix
- 06: Gene program decomposition
- 07: Dose-time stability
- 08: Genetic vs drug perturbation
- 09: Clinical phase prediction
"""
import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, fisher_exact
from tqdm import tqdm


def load_data(args):
    print(f"[{datetime.now():%H:%M:%S}] Loading data...")
    data = np.load(args.data, allow_pickle=True)
    all_sigs = data["signatures"]
    all_sig_ids = list(data["sig_ids"])
    gene_ids = list(data["gene_ids"])
    sig_id_to_idx = {sid: i for i, sid in enumerate(all_sig_ids)}

    siginfo = pd.read_csv(args.metadata, sep="\t", low_memory=False, compression="infer")
    cellinfo = pd.read_csv(args.cellinfo, sep="\t", low_memory=False, compression="infer")

    with open(args.labels) as f:
        labels = json.load(f)
    drug_labels = {d["pert_iname"]: d for d in labels["drugs"]}

    with open(args.exp1_results) as f:
        exp1 = json.load(f)
    exp1_by_name = {r["drug_name"]: r for r in exp1}

    print(f"  Signatures: {all_sigs.shape}")
    print(f"  Siginfo: {len(siginfo)} rows")
    print(f"  Cell lines: {len(cellinfo)}")
    print(f"  Drug labels: {len(drug_labels)}")
    return all_sigs, all_sig_ids, gene_ids, sig_id_to_idx, siginfo, cellinfo, drug_labels, exp1_by_name


def build_drug_cell_map(siginfo, sig_id_to_idx, pert_type="trt_cp"):
    drugs = {}
    for _, row in tqdm(siginfo.iterrows(), total=len(siginfo), desc="Building drug-cell map"):
        if row["pert_type"] != pert_type:
            continue
        name = row["pert_iname"]
        cell = row["cell_id"]
        sid = row["sig_id"]
        if sid not in sig_id_to_idx:
            continue
        if name not in drugs:
            drugs[name] = {}
        if cell not in drugs[name]:
            drugs[name][cell] = []
        drugs[name][cell].append(sig_id_to_idx[sid])
    return drugs


def consensus_signatures(drug_cell_map, all_sigs, min_cells=5):
    result = {}
    for drug, cells in drug_cell_map.items():
        if len(cells) < min_cells:
            continue
        cell_names = sorted(cells.keys())
        sigs = []
        for cell in cell_names:
            idxs = cells[cell]
            sigs.append(all_sigs[idxs].mean(axis=0))
        result[drug] = {"cell_lines": cell_names, "signatures": np.array(sigs)}
    return result


def pairwise_cosine(sigs):
    norms = np.linalg.norm(sigs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    unit = sigs / norms
    return unit @ unit.T


# ============================================================
# Experiment 03: Toxicity confound
# ============================================================
def exp03_toxicity_confound(drug_data, exp1_by_name, all_sigs, drug_cell_map, outdir):
    print(f"\n[{datetime.now():%H:%M:%S}] === Experiment 03: Toxicity confound ===")

    known_cytotoxic_moas = [
        "topoisomerase inhibitor", "DNA alkylating agent",
        "tubulin polymerization inhibitor", "proteasome inhibitor",
        "RNA polymerase inhibitor", "protein synthesis inhibitor",
    ]

    cytotoxic_drugs = []
    for name, dd in drug_data.items():
        if name not in exp1_by_name:
            continue
        moa = exp1_by_name[name].get("moa") or ""
        if any(m in moa for m in known_cytotoxic_moas):
            cytotoxic_drugs.append(name)

    if not cytotoxic_drugs:
        print("  No cytotoxic drugs found, skipping")
        return {}

    all_drug_sigs = []
    for name in cytotoxic_drugs:
        dd = drug_data[name]
        all_drug_sigs.append(dd["signatures"].mean(axis=0))
    mean_cytotoxic_sig = np.mean(all_drug_sigs, axis=0)
    mean_cytotoxic_unit = mean_cytotoxic_sig / np.linalg.norm(mean_cytotoxic_sig)

    stress_gene_idxs = np.argsort(np.abs(mean_cytotoxic_sig))[-50:]

    cosines_with_stress = []
    instabilities = []
    for name, dd in drug_data.items():
        if name not in exp1_by_name:
            continue
        mean_sig = dd["signatures"].mean(axis=0)
        mean_unit = mean_sig / max(np.linalg.norm(mean_sig), 1e-10)
        cos_stress = float(np.dot(mean_unit, mean_cytotoxic_unit))
        cosines_with_stress.append(cos_stress)
        instabilities.append(exp1_by_name[name]["direction_instability"])

    rho, p = spearmanr(instabilities, cosines_with_stress)
    print(f"  H9: Spearman(instability, stress_cosine) = {rho:.4f}, p = {p:.2e}")

    non_cytotoxic_instabilities = []
    non_cytotoxic_cosines = []
    for name, dd in drug_data.items():
        if name not in exp1_by_name:
            continue
        moa = exp1_by_name[name].get("moa") or ""
        if any(m in moa for m in known_cytotoxic_moas):
            continue
        mean_sig = dd["signatures"].mean(axis=0)
        mean_unit = mean_sig / max(np.linalg.norm(mean_sig), 1e-10)
        cos_stress = float(np.dot(mean_unit, mean_cytotoxic_unit))
        non_cytotoxic_cosines.append(cos_stress)
        non_cytotoxic_instabilities.append(exp1_by_name[name]["direction_instability"])

    rho_nc, p_nc = spearmanr(non_cytotoxic_instabilities, non_cytotoxic_cosines)
    print(f"  H9 (excluding cytotoxic drugs): rho = {rho_nc:.4f}, p = {p_nc:.2e}")

    print(f"\n  H10: Recomputing instability after removing top 50 stress genes...")
    mask = np.ones(drug_data[list(drug_data.keys())[0]]["signatures"].shape[1], dtype=bool)
    mask[stress_gene_idxs] = False

    hdac_drugs = {name: dd for name, dd in drug_data.items()
                  if name in exp1_by_name and exp1_by_name[name].get("moa")
                  and "HDAC inhibitor" in str(exp1_by_name[name]["moa"])}

    print(f"  HDAC inhibitors (stress genes removed):")
    hdac_results = []
    for name in sorted(hdac_drugs.keys()):
        dd = hdac_drugs[name]
        sigs_masked = dd["signatures"][:, mask]
        cos_mat = pairwise_cosine(sigs_masked)
        triu = cos_mat[np.triu_indices_from(cos_mat, k=1)]
        di_masked = 1.0 - float(triu.mean())
        di_orig = exp1_by_name[name]["direction_instability"]
        hdac_results.append({"drug": name, "di_orig": di_orig, "di_masked": di_masked})
        print(f"    {name:25s}  orig={di_orig:.4f}  masked={di_masked:.4f}  delta={di_masked-di_orig:+.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(instabilities, cosines_with_stress, alpha=0.1, s=5, color="gray")
    axes[0].set_xlabel("Direction instability")
    axes[0].set_ylabel("Cosine with mean cytotoxic signature")
    axes[0].set_title(f"H9: Stress confound (rho={rho:.3f})")

    if hdac_results:
        orig = [r["di_orig"] for r in hdac_results]
        masked = [r["di_masked"] for r in hdac_results]
        names = [r["drug"] for r in hdac_results]
        y = range(len(names))
        axes[1].barh(y, orig, height=0.4, label="Original", color="steelblue", align="edge")
        axes[1].barh([yi + 0.4 for yi in y], masked, height=0.4, label="Stress genes removed", color="darkorange", align="edge")
        axes[1].set_yticks([yi + 0.4 for yi in y])
        axes[1].set_yticklabels(names, fontsize=7)
        axes[1].set_xlabel("Direction instability")
        axes[1].set_title("H10: HDAC gradient after stress removal")
        axes[1].legend(fontsize=8)
        axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(outdir / "exp03_toxicity_confound.png", dpi=150, bbox_inches="tight")
    print(f"  Saved {outdir / 'exp03_toxicity_confound.png'}")

    return {
        "h9_rho": float(rho), "h9_p": float(p),
        "h9_rho_excl_cytotoxic": float(rho_nc), "h9_p_excl_cytotoxic": float(p_nc),
        "n_stress_genes_removed": 50,
        "hdac_stress_removal": hdac_results,
    }


# ============================================================
# Experiment 04: Component-tissue enrichment
# ============================================================
def exp04_tissue_enrichment(drug_data, exp1_by_name, cellinfo, outdir):
    print(f"\n[{datetime.now():%H:%M:%S}] === Experiment 04: Component-tissue enrichment ===")

    cell_tissue = dict(zip(cellinfo["cell_id"], cellinfo["primary_site"]))
    threshold = 0.5

    enrichment_results = []
    n_tested = 0
    n_enriched = 0

    for name, dd in tqdm(drug_data.items(), desc="Tissue enrichment"):
        if name not in exp1_by_name:
            continue
        sigs = dd["signatures"]
        cells = dd["cell_lines"]
        if len(cells) < 8:
            continue

        cos_mat = pairwise_cosine(sigs)
        adj = (cos_mat >= threshold).astype(int)
        np.fill_diagonal(adj, 0)

        visited = [False] * len(cells)
        components = []
        for start in range(len(cells)):
            if visited[start]:
                continue
            comp = []
            stack = [start]
            while stack:
                node = stack.pop()
                if visited[node]:
                    continue
                visited[node] = True
                comp.append(node)
                for nb in range(len(cells)):
                    if adj[node, nb] and not visited[nb]:
                        stack.append(nb)
            components.append(comp)

        n_comps = len(components)
        if not (2 <= n_comps <= 5):
            continue

        n_tested += 1
        tissues_per_comp = []
        has_enrichment = False

        for comp in components:
            if len(comp) < 2:
                continue
            comp_tissues = [cell_tissue.get(cells[i], "unknown") for i in comp]
            other_tissues = [cell_tissue.get(cells[i], "unknown") for i in range(len(cells)) if i not in comp]

            tissue_counts = Counter(comp_tissues)
            for tissue, count in tissue_counts.items():
                if tissue in ("-666", "unknown"):
                    continue
                a = count
                b = len(comp) - count
                c = sum(1 for t in other_tissues if t == tissue)
                d = len(other_tissues) - c
                if a + c < 3:
                    continue
                _, p_val = fisher_exact([[a, b], [c, d]], alternative="greater")
                if p_val < 0.05:
                    has_enrichment = True
                    tissues_per_comp.append({"tissue": tissue, "p": float(p_val), "count_in": a, "count_out": c})

        if has_enrichment:
            n_enriched += 1
        enrichment_results.append({
            "drug": name, "n_comps": n_comps, "n_cells": len(cells),
            "has_enrichment": has_enrichment, "enrichments": tissues_per_comp,
            "moa": exp1_by_name[name].get("moa"),
        })

    frac = n_enriched / max(n_tested, 1)
    print(f"  H11: {n_enriched}/{n_tested} drugs ({100*frac:.1f}%) with tissue-enriched components")
    print(f"  (threshold: >20% for confirmation)")

    kinase_moas = {"EGFR inhibitor": ["lung", "large intestine"],
                   "VEGFR inhibitor": ["vascular system", "kidney"],
                   "MEK inhibitor": ["skin", "large intestine"]}
    print(f"\n  H12: Kinase inhibitor tissue predictions:")
    for moa, expected_tissues in kinase_moas.items():
        hits = [r for r in enrichment_results if r.get("moa") and moa in str(r["moa"])]
        if not hits:
            print(f"    {moa}: no drugs with 2-5 components")
            continue
        for r in hits:
            if r["enrichments"]:
                found = [e["tissue"] for e in r["enrichments"]]
                match = any(t in found for t in expected_tissues)
                print(f"    {r['drug']:25s} ({moa}): enriched for {found}, expected {expected_tissues}, match={match}")
            else:
                print(f"    {r['drug']:25s} ({moa}): no enrichments")

    with open(outdir / "exp04_tissue_enrichment.json", "w") as f:
        json.dump(enrichment_results, f, indent=2, default=str)

    return {"n_tested": n_tested, "n_enriched": n_enriched, "frac_enriched": frac}


# ============================================================
# Experiment 05: Cell-line similarity matrix
# ============================================================
def exp05_cellline_similarity(drug_data, exp1_by_name, cellinfo, outdir):
    print(f"\n[{datetime.now():%H:%M:%S}] === Experiment 05: Cell-line similarity matrix ===")

    all_cells = set()
    for dd in drug_data.values():
        all_cells.update(dd["cell_lines"])
    all_cells = sorted(all_cells)
    cell_idx = {c: i for i, c in enumerate(all_cells)}
    n = len(all_cells)

    cooccur = np.zeros((n, n), dtype=np.float32)
    pair_counts = np.zeros((n, n), dtype=np.float32)
    threshold = 0.5

    for name, dd in tqdm(drug_data.items(), desc="Cell-line cooccurrence"):
        sigs = dd["signatures"]
        cells = dd["cell_lines"]
        if len(cells) < 5:
            continue
        cos_mat = pairwise_cosine(sigs)

        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                ci, cj = cell_idx[cells[i]], cell_idx[cells[j]]
                pair_counts[ci, cj] += 1
                pair_counts[cj, ci] += 1
                if cos_mat[i, j] >= threshold:
                    cooccur[ci, cj] += 1
                    cooccur[cj, ci] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.where(pair_counts > 0, cooccur / pair_counts, 0)

    cell_tissue = dict(zip(cellinfo["cell_id"], cellinfo["primary_site"]))
    tissues = [cell_tissue.get(c, "unknown") for c in all_cells]

    same_tissue = []
    diff_tissue = []
    for i in range(n):
        for j in range(i + 1, n):
            if pair_counts[i, j] < 10:
                continue
            if tissues[i] in ("-666", "unknown") or tissues[j] in ("-666", "unknown"):
                continue
            if tissues[i] == tissues[j]:
                same_tissue.append(similarity[i, j])
            else:
                diff_tissue.append(similarity[i, j])

    if same_tissue and diff_tissue:
        u, p = mannwhitneyu(same_tissue, diff_tissue, alternative="greater")
        print(f"  H13: Same-tissue cooccurrence = {np.mean(same_tissue):.4f} (n={len(same_tissue)})")
        print(f"        Diff-tissue cooccurrence = {np.mean(diff_tissue):.4f} (n={len(diff_tissue)})")
        print(f"        Mann-Whitney p = {p:.2e}")

    top_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if pair_counts[i, j] >= 20 and tissues[i] != tissues[j]:
                if tissues[i] not in ("-666", "unknown") and tissues[j] not in ("-666", "unknown"):
                    top_pairs.append((all_cells[i], all_cells[j], tissues[i], tissues[j], float(similarity[i, j]), int(pair_counts[i, j])))

    top_pairs.sort(key=lambda x: -x[4])
    print(f"\n  H14: Top 10 cross-tissue high-cooccurrence pairs:")
    for c1, c2, t1, t2, sim, cnt in top_pairs[:10]:
        print(f"    {c1:10s} ({t1:20s}) <-> {c2:10s} ({t2:20s})  sim={sim:.3f}  n_drugs={cnt}")

    top_cells = [c for c in all_cells if sum(1 for dd in drug_data.values() if c in dd["cell_lines"]) > 50]
    if len(top_cells) > 5:
        top_idx = [cell_idx[c] for c in top_cells]
        sub_sim = similarity[np.ix_(top_idx, top_idx)]
        sub_tissues = [cell_tissue.get(c, "?") for c in top_cells]

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(sub_sim, cmap="YlOrRd", vmin=0, vmax=max(0.1, sub_sim.max()))
        ax.set_xticks(range(len(top_cells)))
        ax.set_xticklabels([f"{c}\n({t[:8]})" for c, t in zip(top_cells, sub_tissues)], rotation=90, fontsize=7)
        ax.set_yticks(range(len(top_cells)))
        ax.set_yticklabels([f"{c} ({t[:8]})" for c, t in zip(top_cells, sub_tissues)], fontsize=7)
        plt.colorbar(im, label="Cooccurrence rate")
        ax.set_title("Drug-mechanism-derived cell line similarity")
        plt.tight_layout()
        plt.savefig(outdir / "exp05_cellline_similarity.png", dpi=150, bbox_inches="tight")
        print(f"  Saved {outdir / 'exp05_cellline_similarity.png'}")

    result = {
        "n_cell_lines": n,
        "h13_same_tissue_mean": float(np.mean(same_tissue)) if same_tissue else None,
        "h13_diff_tissue_mean": float(np.mean(diff_tissue)) if diff_tissue else None,
        "h13_p": float(p) if same_tissue and diff_tissue else None,
        "h14_top_cross_tissue": [{"c1": c1, "c2": c2, "t1": t1, "t2": t2, "sim": sim} for c1, c2, t1, t2, sim, _ in top_pairs[:20]],
    }
    with open(outdir / "exp05_cellline_similarity.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


# ============================================================
# Experiment 06: Gene program decomposition
# ============================================================
def exp06_gene_programs(drug_data, exp1_by_name, gene_ids, outdir):
    print(f"\n[{datetime.now():%H:%M:%S}] === Experiment 06: Gene program decomposition ===")

    core_sizes = []
    instabilities = []
    gene_consistency = defaultdict(list)

    for name, dd in tqdm(drug_data.items(), desc="Gene programs"):
        if name not in exp1_by_name:
            continue
        sigs = dd["signatures"]
        n_cells = sigs.shape[0]
        if n_cells < 5:
            continue

        sign_consistency = np.mean(np.sign(sigs) == np.sign(sigs.mean(axis=0, keepdims=True)), axis=0)
        magnitude_cv = np.std(np.abs(sigs), axis=0) / np.maximum(np.mean(np.abs(sigs), axis=0), 1e-10)

        core_mask = (sign_consistency > 0.8) & (magnitude_cv < 1.0) & (np.abs(sigs.mean(axis=0)) > 0.5)
        core_size = int(core_mask.sum())
        core_sizes.append(core_size)
        instabilities.append(exp1_by_name[name]["direction_instability"])

        for gi in np.where(core_mask)[0]:
            gene_consistency[gene_ids[gi]].append(name)

    rho, p = spearmanr([-d for d in instabilities], core_sizes)
    print(f"  H15: Spearman(-instability, core_size) = {rho:.4f}, p = {p:.2e}")
    print(f"  Mean core size: {np.mean(core_sizes):.1f}, median: {np.median(core_sizes):.0f}")

    top_core_genes = sorted(gene_consistency.items(), key=lambda x: -len(x[1]))[:30]
    print(f"\n  H16/H17: Top 30 most consistently perturbed genes (across all drugs):")
    for gene, drugs in top_core_genes[:30]:
        print(f"    {gene:10s}  core in {len(drugs)} drugs")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(instabilities, core_sizes, alpha=0.1, s=5, color="steelblue")
    ax.set_xlabel("Direction instability")
    ax.set_ylabel("Core gene set size")
    ax.set_title(f"H15: Core genes vs instability (rho={rho:.3f})")
    plt.tight_layout()
    plt.savefig(outdir / "exp06_gene_programs.png", dpi=150, bbox_inches="tight")
    print(f"  Saved {outdir / 'exp06_gene_programs.png'}")

    return {"h15_rho": float(rho), "h15_p": float(p),
            "mean_core_size": float(np.mean(core_sizes)),
            "top_core_genes": [(g, len(d)) for g, d in top_core_genes[:30]]}


# ============================================================
# Experiment 07: Dose-time stability
# ============================================================
def exp07_dose_time(all_sigs, sig_id_to_idx, siginfo, exp1_by_name, outdir):
    print(f"\n[{datetime.now():%H:%M:%S}] === Experiment 07: Dose-time stability ===")

    drugs_dt = siginfo[siginfo["pert_type"] == "trt_cp"].copy()
    drugs_dt = drugs_dt[drugs_dt["sig_id"].isin(sig_id_to_idx)]

    dose_instabilities = []
    time_instabilities = []
    cross_cell_instabilities = []

    drug_cell_groups = drugs_dt.groupby(["pert_iname", "cell_id"])

    drugs_with_dose_var = defaultdict(list)
    drugs_with_time_var = defaultdict(list)

    for (drug, cell), group in drug_cell_groups:
        doses = group["pert_idose"].unique()
        times = group["pert_itime"].unique()
        if len(doses) >= 2:
            drugs_with_dose_var[drug].append((cell, group))
        if len(times) >= 2:
            drugs_with_time_var[drug].append((cell, group))

    print(f"  Drugs with dose variation in >=1 cell line: {len(drugs_with_dose_var)}")
    print(f"  Drugs with time variation in >=1 cell line: {len(drugs_with_time_var)}")

    dose_rotations = []
    time_rotations = []

    for drug, cell_groups in tqdm(list(drugs_with_dose_var.items())[:2000], desc="Dose variation"):
        for cell, group in cell_groups:
            doses = group["pert_idose"].unique()
            if len(doses) < 2:
                continue
            dose_sigs = []
            for dose in doses:
                idxs = [sig_id_to_idx[sid] for sid in group[group["pert_idose"] == dose]["sig_id"] if sid in sig_id_to_idx]
                if idxs:
                    dose_sigs.append(all_sigs[idxs].mean(axis=0))
            if len(dose_sigs) >= 2:
                dose_sigs = np.array(dose_sigs)
                cos_mat = pairwise_cosine(dose_sigs)
                triu = cos_mat[np.triu_indices_from(cos_mat, k=1)]
                dose_rotations.append(float(1 - triu.mean()))

    for drug, cell_groups in tqdm(list(drugs_with_time_var.items())[:2000], desc="Time variation"):
        for cell, group in cell_groups:
            times = group["pert_itime"].unique()
            if len(times) < 2:
                continue
            time_sigs = []
            for time in times:
                idxs = [sig_id_to_idx[sid] for sid in group[group["pert_itime"] == time]["sig_id"] if sid in sig_id_to_idx]
                if idxs:
                    time_sigs.append(all_sigs[idxs].mean(axis=0))
            if len(time_sigs) >= 2:
                time_sigs = np.array(time_sigs)
                cos_mat = pairwise_cosine(time_sigs)
                triu = cos_mat[np.triu_indices_from(cos_mat, k=1)]
                time_rotations.append(float(1 - triu.mean()))

    if dose_rotations and time_rotations:
        print(f"  H19: Mean dose rotation: {np.mean(dose_rotations):.4f} (n={len(dose_rotations)})")
        print(f"        Mean time rotation: {np.mean(time_rotations):.4f} (n={len(time_rotations)})")
        u, p = mannwhitneyu(time_rotations, dose_rotations, alternative="greater")
        print(f"        Mann-Whitney (time > dose): p = {p:.2e}")

    dt_drug_instabilities = []
    for drug in tqdm(list(drugs_with_time_var.keys())[:2000], desc="Dose-time vs cross-cell"):
        if drug not in exp1_by_name:
            continue
        within_rotations = []
        for cell, group in drugs_with_time_var[drug]:
            times = group["pert_itime"].unique()
            if len(times) < 2:
                continue
            time_sigs = []
            for time in times:
                idxs = [sig_id_to_idx[sid] for sid in group[group["pert_itime"] == time]["sig_id"] if sid in sig_id_to_idx]
                if idxs:
                    time_sigs.append(all_sigs[idxs].mean(axis=0))
            if len(time_sigs) >= 2:
                time_sigs_arr = np.array(time_sigs)
                cos_mat = pairwise_cosine(time_sigs_arr)
                triu = cos_mat[np.triu_indices_from(cos_mat, k=1)]
                within_rotations.append(float(1 - triu.mean()))
        if within_rotations:
            dt_drug_instabilities.append((
                exp1_by_name[drug]["direction_instability"],
                float(np.mean(within_rotations)),
            ))

    if dt_drug_instabilities:
        cross_cell, within_dt = zip(*dt_drug_instabilities)
        rho, p = spearmanr(cross_cell, within_dt)
        print(f"\n  H18: Spearman(cross-cell instability, within-cell dose-time instability)")
        print(f"        rho = {rho:.4f}, p = {p:.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if dose_rotations and time_rotations:
        axes[0].hist(dose_rotations, bins=50, alpha=0.6, label=f"Dose (n={len(dose_rotations)})", color="steelblue")
        axes[0].hist(time_rotations, bins=50, alpha=0.6, label=f"Time (n={len(time_rotations)})", color="darkorange")
        axes[0].set_xlabel("Within-cell-line direction instability")
        axes[0].set_title("H19: Dose vs time rotation")
        axes[0].legend()

    if dt_drug_instabilities:
        axes[1].scatter(cross_cell, within_dt, alpha=0.1, s=5, color="steelblue")
        axes[1].set_xlabel("Cross-cell-line instability")
        axes[1].set_ylabel("Within-cell dose-time instability")
        axes[1].set_title(f"H18: Cross-cell vs dose-time (rho={rho:.3f})")

    plt.tight_layout()
    plt.savefig(outdir / "exp07_dose_time.png", dpi=150, bbox_inches="tight")
    print(f"  Saved {outdir / 'exp07_dose_time.png'}")

    return {
        "h18_rho": float(rho) if dt_drug_instabilities else None,
        "h19_mean_dose_rotation": float(np.mean(dose_rotations)) if dose_rotations else None,
        "h19_mean_time_rotation": float(np.mean(time_rotations)) if time_rotations else None,
    }


# ============================================================
# Experiment 08: Genetic vs drug perturbation
# ============================================================
def exp08_genetic_comparison(all_sigs, sig_id_to_idx, siginfo, drug_data, exp1_by_name, drug_labels, outdir):
    print(f"\n[{datetime.now():%H:%M:%S}] === Experiment 08: Genetic vs drug comparison ===")

    shrna = siginfo[siginfo["pert_type"] == "trt_sh"]
    shrna = shrna[shrna["sig_id"].isin(sig_id_to_idx)]
    print(f"  shRNA signatures in extracted data: {len(shrna)}")

    if len(shrna) == 0:
        print("  No shRNA signatures in extracted subset — need to extract trt_sh from GCTX")
        print("  (Current extraction only contains trt_cp)")
        return {"status": "skipped", "reason": "shRNA signatures not in extracted subset"}

    gene_cell_sigs = defaultdict(lambda: defaultdict(list))
    for _, row in tqdm(shrna.iterrows(), total=len(shrna), desc="Building gene-cell map"):
        gene_cell_sigs[row["pert_iname"]][row["cell_id"]].append(sig_id_to_idx[row["sig_id"]])

    drug_targets = {}
    for name, label in drug_labels.items():
        target = label.get("target")
        if target and name in drug_data:
            targets = [t.strip() for t in str(target).split("|")]
            drug_targets[name] = targets

    print(f"  Drugs with targets and signatures: {len(drug_targets)}")
    print(f"  Genes with shRNA signatures: {len(gene_cell_sigs)}")

    matched = []
    for drug, targets in drug_targets.items():
        for target in targets:
            if target in gene_cell_sigs:
                shared_cells = set(drug_data[drug]["cell_lines"]) & set(gene_cell_sigs[target].keys())
                if len(shared_cells) >= 3:
                    matched.append((drug, target, shared_cells))

    print(f"  Drug-gene pairs with >=3 shared cell lines: {len(matched)}")

    if not matched:
        return {"status": "no_matches", "n_drug_targets": len(drug_targets), "n_genes": len(gene_cell_sigs)}

    drug_gene_cosines = []
    for drug, target, shared_cells in tqdm(matched, desc="Drug-gene cosines"):
        dd = drug_data[drug]
        cosines = []
        for cell in shared_cells:
            drug_idx = dd["cell_lines"].index(cell)
            drug_sig = dd["signatures"][drug_idx]
            gene_idxs = gene_cell_sigs[target][cell]
            gene_sig = all_sigs[gene_idxs].mean(axis=0)
            cos = float(np.dot(drug_sig, gene_sig) / (np.linalg.norm(drug_sig) * np.linalg.norm(gene_sig) + 1e-10))
            cosines.append(cos)
        mean_cos = float(np.mean(cosines))
        di = exp1_by_name.get(drug, {}).get("direction_instability")
        if di is not None:
            drug_gene_cosines.append({"drug": drug, "target": target, "mean_cosine": mean_cos, "di": di, "n_shared": len(shared_cells)})

    if drug_gene_cosines:
        dis = [r["di"] for r in drug_gene_cosines]
        coss = [r["mean_cosine"] for r in drug_gene_cosines]
        rho, p = spearmanr(dis, coss)
        print(f"  H20: Spearman(instability, drug-gene cosine) = {rho:.4f}, p = {p:.2e}")

        sorted_dgc = sorted(drug_gene_cosines, key=lambda x: -abs(x["mean_cosine"]))
        print(f"\n  Top 10 drug-gene matches:")
        for r in sorted_dgc[:10]:
            print(f"    {r['drug']:25s} <-> {r['target']:15s}  cos={r['mean_cosine']:.3f}  di={r['di']:.3f}  n={r['n_shared']}")

    with open(outdir / "exp08_genetic_comparison.json", "w") as f:
        json.dump(drug_gene_cosines, f, indent=2, default=str)

    return {"n_matched": len(matched), "h20_rho": float(rho) if drug_gene_cosines else None}


# ============================================================
# Experiment 09: Clinical phase prediction
# ============================================================
def exp09_clinical_phase(drug_data, exp1_by_name, drug_labels, outdir):
    print(f"\n[{datetime.now():%H:%M:%S}] === Experiment 09: Clinical phase prediction ===")

    phase_order = {"Preclinical": 0, "Phase 1": 1, "Phase 1/Phase 2": 1.5,
                   "Phase 2": 2, "Phase 2/Phase 3": 2.5, "Phase 3": 3,
                   "Launched": 4, "Withdrawn": -1}

    drugs_with_phase = []
    for name in drug_data:
        if name not in exp1_by_name:
            continue
        label = drug_labels.get(name, {})
        phase = label.get("clinical_phase")
        if not phase or phase not in phase_order:
            continue
        if phase == "Withdrawn":
            continue
        drugs_with_phase.append({
            "drug": name,
            "phase": phase,
            "phase_num": phase_order[phase],
            "di": exp1_by_name[name]["direction_instability"],
            "mag_cv": exp1_by_name[name]["magnitude_cv"],
            "moa": label.get("moa"),
        })

    print(f"  Drugs with clinical phase (excl withdrawn): {len(drugs_with_phase)}")

    phases = [r["phase_num"] for r in drugs_with_phase]
    dis = [r["di"] for r in drugs_with_phase]
    rho, p = spearmanr(phases, dis)
    print(f"  H22: Spearman(phase, instability) = {rho:.4f}, p = {p:.2e}")

    for phase_name in ["Preclinical", "Phase 1", "Phase 2", "Phase 3", "Launched"]:
        vals = [r["di"] for r in drugs_with_phase if r["phase"] == phase_name]
        if vals:
            print(f"    {phase_name:15s}: n={len(vals):4d}  mean_di={np.mean(vals):.4f}")

    print(f"\n  H23: Within-MOA-class phase-instability correlation:")
    moa_groups = defaultdict(list)
    for r in drugs_with_phase:
        if r.get("moa"):
            for m in str(r["moa"]).split("|"):
                moa_groups[m.strip()].append(r)

    significant_moas = []
    for moa, group in sorted(moa_groups.items(), key=lambda x: -len(x[1])):
        if len(group) < 8:
            continue
        phases_g = [r["phase_num"] for r in group]
        dis_g = [r["di"] for r in group]
        if len(set(phases_g)) < 2:
            continue
        rho_g, p_g = spearmanr(phases_g, dis_g)
        sig = "*" if p_g < 0.05 else ""
        print(f"    {moa:40s}  n={len(group):3d}  rho={rho_g:+.3f}  p={p_g:.3f} {sig}")
        if p_g < 0.05:
            significant_moas.append(moa)

    withdrawn = [r for r in exp1_by_name.values() if drug_labels.get(r["drug_name"], {}).get("clinical_phase") == "Withdrawn"]
    launched = [r for r in exp1_by_name.values() if drug_labels.get(r["drug_name"], {}).get("clinical_phase") == "Launched"]

    if withdrawn and launched:
        w_di = [r["direction_instability"] for r in withdrawn]
        l_di = [r["direction_instability"] for r in launched]
        w_mag = [r["magnitude_cv"] for r in withdrawn]
        l_mag = [r["magnitude_cv"] for r in launched]
        print(f"\n  H24: Withdrawn vs Launched:")
        print(f"    Withdrawn: n={len(withdrawn)}, mean_di={np.mean(w_di):.4f}, mean_mag_cv={np.mean(w_mag):.4f}")
        print(f"    Launched:  n={len(launched)}, mean_di={np.mean(l_di):.4f}, mean_mag_cv={np.mean(l_mag):.4f}")
        u_di, p_di = mannwhitneyu(w_di, l_di)
        u_mag, p_mag = mannwhitneyu(w_mag, l_mag)
        print(f"    DI difference: p={p_di:.2e}")
        print(f"    Mag CV difference: p={p_mag:.2e}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    phase_labels = ["Preclinical", "Phase 1", "Phase 2", "Phase 3", "Launched"]
    phase_data = [[r["di"] for r in drugs_with_phase if r["phase"] == pl] for pl in phase_labels]
    axes[0].violinplot([d for d in phase_data if d], positions=[i for i, d in enumerate(phase_data) if d], showmedians=True)
    axes[0].set_xticks(range(len(phase_labels)))
    axes[0].set_xticklabels(phase_labels, rotation=45, fontsize=8)
    axes[0].set_ylabel("Direction instability")
    axes[0].set_title(f"H22: Phase vs instability (rho={rho:.3f})")

    if withdrawn and launched:
        axes[1].violinplot([w_di, l_di], positions=[0, 1], showmedians=True)
        axes[1].set_xticks([0, 1])
        axes[1].set_xticklabels(["Withdrawn", "Launched"])
        axes[1].set_ylabel("Direction instability")
        axes[1].set_title(f"H24: Withdrawn vs Launched (p={p_di:.2e})")

    if significant_moas:
        moa_rhos = []
        moa_names = []
        for moa in significant_moas[:10]:
            group = moa_groups[moa]
            phases_g = [r["phase_num"] for r in group]
            dis_g = [r["di"] for r in group]
            rho_g, _ = spearmanr(phases_g, dis_g)
            moa_rhos.append(rho_g)
            moa_names.append(moa[:30])
        axes[2].barh(range(len(moa_names)), moa_rhos, color="steelblue")
        axes[2].set_yticks(range(len(moa_names)))
        axes[2].set_yticklabels(moa_names, fontsize=7)
        axes[2].set_xlabel("Spearman rho (phase vs instability)")
        axes[2].set_title("H23: Within-MOA correlations (p<0.05)")
        axes[2].axvline(0, color="gray", linewidth=0.5)
    else:
        axes[2].text(0.5, 0.5, "No significant\nwithin-MOA correlations", transform=axes[2].transAxes, ha="center")

    plt.tight_layout()
    plt.savefig(outdir / "exp09_clinical_phase.png", dpi=150, bbox_inches="tight")
    print(f"  Saved {outdir / 'exp09_clinical_phase.png'}")

    return {
        "h22_rho": float(rho), "h22_p": float(p),
        "n_significant_moas": len(significant_moas),
        "significant_moas": significant_moas,
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", default="data/frozen_drug_labels.json")
    parser.add_argument("--metadata", default="data/GSE92742_Broad_LINCS_sig_info.txt.gz")
    parser.add_argument("--cellinfo", default="data/GSE92742_Broad_LINCS_cell_info.txt.gz")
    parser.add_argument("--exp1-results", default="results/01_cross_cellline/real_results.json")
    parser.add_argument("--outdir", default="results/03_comprehensive")
    parser.add_argument("--min-cells", type=int, default=5)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_sigs, all_sig_ids, gene_ids, sig_id_to_idx, siginfo, cellinfo, drug_labels, exp1_by_name = load_data(args)

    print(f"[{datetime.now():%H:%M:%S}] Building drug-cell map...")
    drug_cell_map = build_drug_cell_map(siginfo, sig_id_to_idx, "trt_cp")
    drug_data = consensus_signatures(drug_cell_map, all_sigs, min_cells=args.min_cells)
    print(f"  Drugs with >={args.min_cells} cell lines: {len(drug_data)}")

    all_results = {}

    all_results["exp03"] = exp03_toxicity_confound(drug_data, exp1_by_name, all_sigs, drug_cell_map, outdir)
    all_results["exp04"] = exp04_tissue_enrichment(drug_data, exp1_by_name, cellinfo, outdir)
    all_results["exp05"] = exp05_cellline_similarity(drug_data, exp1_by_name, cellinfo, outdir)
    all_results["exp06"] = exp06_gene_programs(drug_data, exp1_by_name, gene_ids, outdir)
    all_results["exp07"] = exp07_dose_time(all_sigs, sig_id_to_idx, siginfo, exp1_by_name, outdir)
    all_results["exp08"] = exp08_genetic_comparison(all_sigs, sig_id_to_idx, siginfo, drug_data, exp1_by_name, drug_labels, outdir)
    all_results["exp09"] = exp09_clinical_phase(drug_data, exp1_by_name, drug_labels, outdir)

    with open(outdir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[{datetime.now():%H:%M:%S}] All results saved to {outdir / 'all_results.json'}")


if __name__ == "__main__":
    main()
