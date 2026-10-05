"""
Core defense experiments for the drug transport paper.

Four experiments that defend or extend the single claim:
  "Target breadth — not drug class — determines cross-context transport,
   and the invariance group G captures this structure."

- Toxicity confound (H9-H10): transporting ≠ cytotoxic
- Component-tissue enrichment (H11-H12): G has biological structure
- Core gene pathway (H16): transporting core = target mechanism
- Genetic triangulation (H20-H21): drug transport predicts drug↔shRNA match
"""
import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, fisher_exact
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def load_all(args):
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

    print(f"  Signatures: {all_sigs.shape}, Drugs: {len(drug_labels)}")
    return all_sigs, gene_ids, sig_id_to_idx, siginfo, cellinfo, drug_labels, exp1_by_name


def build_consensus(siginfo, sig_id_to_idx, all_sigs, pert_type="trt_cp", min_cells=5):
    drug_cell_map = {}
    for _, row in tqdm(siginfo.iterrows(), total=len(siginfo), desc=f"Indexing {pert_type}"):
        if row["pert_type"] != pert_type:
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
    return result, drug_cell_map


def pairwise_cosine(sigs):
    norms = np.linalg.norm(sigs, axis=1, keepdims=True)
    unit = sigs / np.maximum(norms, 1e-10)
    return unit @ unit.T


def connected_components(adj):
    n = adj.shape[0]
    visited = [False] * n
    components = []
    for start in range(n):
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
            for nb in range(n):
                if adj[node, nb] and not visited[nb]:
                    stack.append(nb)
        components.append(sorted(comp))
    return components


# ── Experiment 03: Toxicity confound ──────────────────────────
def run_toxicity_confound(drug_data, exp1_by_name, outdir):
    print(f"\n[{datetime.now():%H:%M:%S}] === Toxicity confound (H9-H10) ===")

    cytotoxic_moas = [
        "topoisomerase inhibitor", "DNA alkylating agent",
        "tubulin polymerization inhibitor", "proteasome inhibitor",
        "RNA polymerase inhibitor", "protein synthesis inhibitor",
    ]

    cytotoxic_sigs = []
    for name, dd in drug_data.items():
        if name not in exp1_by_name:
            continue
        moa = exp1_by_name[name].get("moa") or ""
        if any(m in moa for m in cytotoxic_moas):
            cytotoxic_sigs.append(dd["signatures"].mean(axis=0))

    if not cytotoxic_sigs:
        print("  No cytotoxic drugs found")
        return {}

    mean_cytotoxic = np.mean(cytotoxic_sigs, axis=0)
    mean_cytotoxic_unit = mean_cytotoxic / np.linalg.norm(mean_cytotoxic)
    stress_gene_idxs = np.argsort(np.abs(mean_cytotoxic))[-50:]

    cosines = []
    instabilities = []
    is_cytotoxic = []
    for name, dd in drug_data.items():
        if name not in exp1_by_name:
            continue
        mean_sig = dd["signatures"].mean(axis=0)
        mean_unit = mean_sig / max(np.linalg.norm(mean_sig), 1e-10)
        cosines.append(float(np.dot(mean_unit, mean_cytotoxic_unit)))
        instabilities.append(exp1_by_name[name]["direction_instability"])
        moa = exp1_by_name[name].get("moa") or ""
        is_cytotoxic.append(any(m in moa for m in cytotoxic_moas))

    rho_all, p_all = spearmanr(instabilities, cosines)
    non_cyto_idx = [i for i, c in enumerate(is_cytotoxic) if not c]
    rho_nc, p_nc = spearmanr(
        [instabilities[i] for i in non_cyto_idx],
        [cosines[i] for i in non_cyto_idx],
    )
    print(f"  H9 (all drugs):        rho = {rho_all:.4f}, p = {p_all:.2e}")
    print(f"  H9 (excl cytotoxic):   rho = {rho_nc:.4f}, p = {p_nc:.2e}")
    print(f"  Criterion: |rho| < 0.2 → {'PASS' if abs(rho_nc) < 0.2 else 'FAIL'}")

    # H10: recompute HDAC gradient with stress genes removed
    mask = np.ones(drug_data[next(iter(drug_data))]["signatures"].shape[1], dtype=bool)
    mask[stress_gene_idxs] = False

    hdac = {name: dd for name, dd in drug_data.items()
            if name in exp1_by_name and exp1_by_name[name].get("moa")
            and "HDAC inhibitor" in str(exp1_by_name[name]["moa"])}

    print(f"\n  H10: HDAC gradient after removing 50 stress genes:")
    hdac_rows = []
    for name in sorted(hdac.keys(), key=lambda n: exp1_by_name[n]["direction_instability"]):
        dd = hdac[name]
        sigs_clean = dd["signatures"][:, mask]
        cos_mat = pairwise_cosine(sigs_clean)
        triu = cos_mat[np.triu_indices_from(cos_mat, k=1)]
        di_clean = 1.0 - float(triu.mean())
        di_orig = exp1_by_name[name]["direction_instability"]
        hdac_rows.append({"drug": name, "di_orig": di_orig, "di_clean": di_clean})
        print(f"    {name:25s}  orig={di_orig:.4f}  clean={di_clean:.4f}  Δ={di_clean - di_orig:+.4f}")

    orig_vals = [r["di_orig"] for r in hdac_rows]
    clean_vals = [r["di_clean"] for r in hdac_rows]
    rho_preserved, _ = spearmanr(orig_vals, clean_vals)
    print(f"  Gradient preserved: Spearman(orig, clean) = {rho_preserved:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(instabilities, cosines, alpha=0.05, s=3, color="gray")
    cyto_idx = [i for i, c in enumerate(is_cytotoxic) if c]
    axes[0].scatter([instabilities[i] for i in cyto_idx], [cosines[i] for i in cyto_idx],
                    alpha=0.5, s=15, color="crimson", label="Known cytotoxic")
    axes[0].set_xlabel("Direction instability")
    axes[0].set_ylabel("Cosine with mean cytotoxic signature")
    axes[0].set_title(f"H9: Stress confound check (ρ={rho_nc:.3f} excl cytotoxic)")
    axes[0].legend(fontsize=8)

    y = range(len(hdac_rows))
    axes[1].barh(y, orig_vals, height=0.35, label="Original", color="steelblue", align="edge")
    axes[1].barh([yi + 0.35 for yi in y], clean_vals, height=0.35,
                 label="Stress genes removed", color="darkorange", align="edge")
    axes[1].set_yticks([yi + 0.35 for yi in y])
    axes[1].set_yticklabels([r["drug"] for r in hdac_rows], fontsize=7)
    axes[1].set_xlabel("Direction instability")
    axes[1].set_title(f"H10: HDAC gradient survives (ρ={rho_preserved:.3f})")
    axes[1].legend(fontsize=8)
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(outdir / "toxicity_confound.png", dpi=150, bbox_inches="tight")
    print(f"  Saved {outdir / 'toxicity_confound.png'}")

    return {"h9_rho_all": float(rho_all), "h9_rho_excl_cytotoxic": float(rho_nc),
            "h10_gradient_rho": float(rho_preserved), "hdac": hdac_rows}


# ── Experiment 04: Component-tissue enrichment ────────────────
def run_tissue_enrichment(drug_data, exp1_by_name, cellinfo, outdir):
    print(f"\n[{datetime.now():%H:%M:%S}] === Component-tissue enrichment (H11-H12) ===")

    cell_tissue = dict(zip(cellinfo["cell_id"], cellinfo["primary_site"]))
    threshold = 0.5
    n_tested = 0
    n_enriched = 0
    enrichments = []

    for name, dd in tqdm(drug_data.items(), desc="Tissue enrichment"):
        if name not in exp1_by_name:
            continue
        cells = dd["cell_lines"]
        if len(cells) < 8:
            continue
        cos_mat = pairwise_cosine(dd["signatures"])
        adj = (cos_mat >= threshold).astype(int)
        np.fill_diagonal(adj, 0)
        comps = connected_components(adj)

        if not (2 <= len(comps) <= 5):
            continue
        n_tested += 1

        drug_enrichments = []
        for comp in comps:
            if len(comp) < 2:
                continue
            comp_tissues = [cell_tissue.get(cells[i], "unknown") for i in comp]
            other_tissues = [cell_tissue.get(cells[i], "unknown") for i in range(len(cells)) if i not in comp]

            for tissue, count in Counter(comp_tissues).items():
                if tissue in ("-666", "unknown"):
                    continue
                c_out = sum(1 for t in other_tissues if t == tissue)
                if count + c_out < 3:
                    continue
                _, p = fisher_exact([[count, len(comp) - count],
                                     [c_out, len(other_tissues) - c_out]], alternative="greater")
                if p < 0.05:
                    drug_enrichments.append({"tissue": tissue, "p": float(p),
                                             "in_comp": count, "total_comp": len(comp)})

        if drug_enrichments:
            n_enriched += 1
        enrichments.append({
            "drug": name, "n_comps": len(comps), "n_cells": len(cells),
            "moa": exp1_by_name[name].get("moa"),
            "enrichments": drug_enrichments,
        })

    frac = n_enriched / max(n_tested, 1)
    print(f"  H11: {n_enriched}/{n_tested} ({100*frac:.1f}%) drugs with tissue-enriched components")
    print(f"  Criterion: >20% → {'PASS' if frac > 0.2 else 'FAIL'}")

    # H12: specific kinase predictions
    print(f"\n  H12: Kinase inhibitor tissue predictions:")
    kinase_expected = {
        "EGFR inhibitor": ["lung", "large intestine"],
        "VEGFR inhibitor": ["kidney"],
        "MEK inhibitor": ["skin", "large intestine"],
    }
    kinase_confirmed = 0
    kinase_tested = 0
    for moa, expected in kinase_expected.items():
        hits = [e for e in enrichments if e.get("moa") and moa in str(e["moa"]) and e["enrichments"]]
        if not hits:
            print(f"    {moa}: no drugs with 2-5 components and enrichment")
            continue
        kinase_tested += 1
        for e in hits:
            found_tissues = [x["tissue"] for x in e["enrichments"]]
            match = any(t in found_tissues for t in expected)
            if match:
                kinase_confirmed += 1
            print(f"    {e['drug']:25s} ({moa}): found={found_tissues}, expected={expected}, match={match}")

    print(f"  H12: {kinase_confirmed}/{kinase_tested} kinase predictions confirmed")

    # Show examples
    print(f"\n  Best examples of tissue-enriched components:")
    for e in sorted([x for x in enrichments if x["enrichments"]], key=lambda x: min(y["p"] for y in x["enrichments"]))[:10]:
        best = min(e["enrichments"], key=lambda x: x["p"])
        moa_str = str(e.get("moa") or "no MOA")[:40]
        print(f"    {e['drug']:25s}  {best['tissue']:20s}  p={best['p']:.3f}  "
              f"({best['in_comp']}/{best['total_comp']} in comp)  MOA={moa_str}")

    with open(outdir / "tissue_enrichment.json", "w") as f:
        json.dump(enrichments, f, indent=2, default=str)

    return {"h11_frac": frac, "h11_n_tested": n_tested, "h11_n_enriched": n_enriched,
            "h12_confirmed": kinase_confirmed, "h12_tested": kinase_tested}


# ── Experiment 06: Core gene pathway enrichment ───────────────
def run_core_genes(drug_data, exp1_by_name, gene_ids, outdir):
    print(f"\n[{datetime.now():%H:%M:%S}] === Core gene pathway enrichment (H15-H16) ===")

    core_sizes = []
    instabilities = []
    drug_cores = {}

    for name, dd in tqdm(drug_data.items(), desc="Core genes"):
        if name not in exp1_by_name:
            continue
        sigs = dd["signatures"]
        if sigs.shape[0] < 5:
            continue

        sign_consistency = np.mean(
            np.sign(sigs) == np.sign(sigs.mean(axis=0, keepdims=True)), axis=0
        )
        mag_cv = np.std(np.abs(sigs), axis=0) / np.maximum(np.mean(np.abs(sigs), axis=0), 1e-10)
        mean_effect = np.abs(sigs.mean(axis=0))

        core_mask = (sign_consistency > 0.8) & (mag_cv < 1.0) & (mean_effect > 0.5)
        core_size = int(core_mask.sum())
        core_sizes.append(core_size)
        instabilities.append(exp1_by_name[name]["direction_instability"])
        core_gene_names = [gene_ids[i] for i in np.where(core_mask)[0]]
        drug_cores[name] = core_gene_names

    rho, p = spearmanr([-d for d in instabilities], core_sizes)
    print(f"  H15: Spearman(-instability, core_size) = {rho:.4f}, p = {p:.2e}")
    print(f"  Criterion: |rho| > 0.4 → {'PASS' if abs(rho) > 0.4 else 'FAIL'}")
    print(f"  Mean core: {np.mean(core_sizes):.1f}, median: {np.median(core_sizes):.0f}")

    # H16: HDAC inhibitor core genes should relate to chromatin
    print(f"\n  H16: Core genes for top-transporting HDAC inhibitors:")
    hdac_core_genes = Counter()
    for name, genes in drug_cores.items():
        if exp1_by_name.get(name, {}).get("moa") and "HDAC inhibitor" in str(exp1_by_name[name]["moa"]):
            di = exp1_by_name[name]["direction_instability"]
            if di < 0.7:
                for g in genes:
                    hdac_core_genes[g] += 1
                print(f"    {name:25s}  di={di:.3f}  core_size={len(genes)}")

    if hdac_core_genes:
        print(f"\n  Genes in core of >=3 pan-HDAC inhibitors:")
        for gene, count in hdac_core_genes.most_common(30):
            if count >= 3:
                print(f"    {gene:10s}  in {count} pan-HDAC cores")

    # Same for topoisomerase inhibitors
    topo_core_genes = Counter()
    for name, genes in drug_cores.items():
        if exp1_by_name.get(name, {}).get("moa") and "topoisomerase inhibitor" in str(exp1_by_name[name]["moa"]):
            di = exp1_by_name[name]["direction_instability"]
            if di < 0.8:
                for g in genes:
                    topo_core_genes[g] += 1

    if topo_core_genes:
        print(f"\n  Genes in core of >=2 topoisomerase inhibitors:")
        for gene, count in topo_core_genes.most_common(20):
            if count >= 2:
                print(f"    {gene:10s}  in {count} topo cores")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(instabilities, core_sizes, alpha=0.05, s=3, color="steelblue")
    ax.set_xlabel("Direction instability")
    ax.set_ylabel("Core gene set size")
    ax.set_title(f"H15: Transporting core size (ρ={rho:.3f})")
    plt.tight_layout()
    plt.savefig(outdir / "core_genes.png", dpi=150, bbox_inches="tight")
    print(f"  Saved {outdir / 'core_genes.png'}")

    return {"h15_rho": float(rho), "h15_p": float(p),
            "mean_core": float(np.mean(core_sizes)),
            "hdac_shared_core": dict(hdac_core_genes.most_common(30))}


# ── Experiment 08: Genetic triangulation ──────────────────────
def run_genetic_triangulation(drug_data, exp1_by_name, drug_labels, shrna_data_path, shrna_meta_path, outdir):
    print(f"\n[{datetime.now():%H:%M:%S}] === Genetic triangulation (H20-H21) ===")

    if not Path(shrna_data_path).exists():
        print(f"  shRNA data not found at {shrna_data_path}")
        print("  Run: modal volume get drug-perturbation-vol lincs_shrna.npz data/lincs_shrna.npz")
        return {"status": "needs_shrna_extraction"}

    print(f"  Loading shRNA data from {shrna_data_path}...")
    shrna_npz = np.load(shrna_data_path, allow_pickle=True)
    shrna_sigs = shrna_npz["signatures"]
    shrna_sig_ids = list(shrna_npz["sig_ids"])
    shrna_id_to_idx = {sid: i for i, sid in enumerate(shrna_sig_ids)}
    print(f"  shRNA signatures: {shrna_sigs.shape}")

    shrna_meta = pd.read_csv(shrna_meta_path)
    print(f"  shRNA metadata: {len(shrna_meta)} rows, {shrna_meta['pert_iname'].nunique()} genes")

    gene_cell_sigs = defaultdict(lambda: defaultdict(list))
    for _, row in tqdm(shrna_meta.iterrows(), total=len(shrna_meta), desc="shRNA index"):
        sid = row["sig_id"]
        if sid in shrna_id_to_idx:
            gene_cell_sigs[row["pert_iname"]][row["cell_id"]].append(shrna_id_to_idx[sid])

    matched = []
    for drug_name, dd in drug_data.items():
        label = drug_labels.get(drug_name, {})
        target = label.get("target")
        if not target:
            continue
        for t in str(target).split("|"):
            t = t.strip()
            if t in gene_cell_sigs:
                shared = set(dd["cell_lines"]) & set(gene_cell_sigs[t].keys())
                if len(shared) >= 3:
                    matched.append((drug_name, t, sorted(shared)))

    print(f"  Drug-gene pairs with >=3 shared cell lines: {len(matched)}")

    if not matched:
        return {"status": "no_matches", "n_genes_with_shrna": len(gene_cell_sigs)}

    results = []
    for drug, target, shared_cells in tqdm(matched, desc="Drug-gene cosines"):
        dd = drug_data[drug]
        cosines_per_cell = {}
        for cell in shared_cells:
            drug_idx = dd["cell_lines"].index(cell)
            drug_sig = dd["signatures"][drug_idx]
            gene_idxs = gene_cell_sigs[target][cell]
            gene_sig = shrna_sigs[gene_idxs].mean(axis=0)
            cos = float(np.dot(drug_sig, gene_sig) /
                        (np.linalg.norm(drug_sig) * np.linalg.norm(gene_sig) + 1e-10))
            cosines_per_cell[cell] = cos

        di = exp1_by_name.get(drug, {}).get("direction_instability")
        if di is None:
            continue
        results.append({
            "drug": drug, "target": target,
            "mean_cosine": float(np.mean(list(cosines_per_cell.values()))),
            "di": di, "n_shared": len(shared_cells),
            "per_cell": cosines_per_cell,
        })

    rho, p = None, None
    if results:
        dis = [r["di"] for r in results]
        coss = [abs(r["mean_cosine"]) for r in results]
        rho, p = spearmanr(dis, coss)
        print(f"\n  H20: Spearman(instability, |drug-gene cosine|) = {rho:.4f}, p = {p:.2e}")
        print(f"  Criterion: |rho| > 0.3 → {'PASS' if abs(rho) > 0.3 else 'FAIL'}")

        # H21: HDAC-specific triangulation
        hdac_results = [r for r in results if exp1_by_name.get(r["drug"], {}).get("moa")
                        and "HDAC inhibitor" in str(exp1_by_name[r["drug"]]["moa"])]
        if hdac_results:
            print(f"\n  H21: HDAC inhibitor triangulation ({len(hdac_results)} pairs):")
            for r in sorted(hdac_results, key=lambda x: x["di"]):
                moa = str(exp1_by_name[r["drug"]].get("moa", ""))[:40]
                print(f"    {r['drug']:25s} <-> {r['target']:15s}  cos={r['mean_cosine']:+.3f}  di={r['di']:.3f}")

        print(f"\n  Top 15 drug-gene matches (by |cosine|):")
        for r in sorted(results, key=lambda x: -abs(x["mean_cosine"]))[:15]:
            print(f"    {r['drug']:25s} <-> {r['target']:15s}  cos={r['mean_cosine']:+.3f}  di={r['di']:.3f}")

    # Plot
    if results:
        fig, ax = plt.subplots(figsize=(8, 6))
        dis_plot = [r["di"] for r in results]
        cos_plot = [abs(r["mean_cosine"]) for r in results]
        ax.scatter(dis_plot, cos_plot, alpha=0.15, s=8, color="steelblue")
        if hdac_results:
            hdac_dis = [r["di"] for r in hdac_results]
            hdac_cos = [abs(r["mean_cosine"]) for r in hdac_results]
            ax.scatter(hdac_dis, hdac_cos, s=30, color="crimson", zorder=5, label="HDAC inhibitors")
            for r in hdac_results:
                ax.annotate(r["drug"][:12], (r["di"], abs(r["mean_cosine"])),
                            fontsize=5, alpha=0.7, xytext=(3, 3), textcoords="offset points")
            ax.legend(fontsize=8)
        ax.set_xlabel("Direction instability")
        ax.set_ylabel("|Drug–shRNA cosine|")
        ax.set_title(f"H20: Genetic triangulation (ρ={rho:.3f}, p={p:.2e})")
        plt.tight_layout()
        plt.savefig(outdir / "genetic_triangulation.png", dpi=150, bbox_inches="tight")
        print(f"  Saved {outdir / 'genetic_triangulation.png'}")

    with open(outdir / "genetic_triangulation.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return {"n_matched": len(matched),
            "h20_rho": float(rho) if rho is not None else None,
            "h20_p": float(p) if p is not None else None}


# ── Experiment 10: Held-out context prediction ─────────────────
def run_holdout_prediction(drug_data, exp1_by_name, outdir, min_cells=10):
    print(f"\n[{datetime.now():%H:%M:%S}] === Held-out context prediction (H25-H27) ===")

    loo_results = []
    for name, dd in tqdm(drug_data.items(), desc="LOO prediction"):
        if name not in exp1_by_name:
            continue
        sigs = dd["signatures"]
        n = sigs.shape[0]
        if n < min_cells:
            continue

        heldout_cosines = []
        loo_instabilities = []
        for i in range(n):
            train = np.delete(sigs, i, axis=0)
            held = sigs[i]
            cos_mat = pairwise_cosine(train)
            triu = cos_mat[np.triu_indices_from(cos_mat, k=1)]
            loo_di = 1.0 - float(triu.mean())
            loo_instabilities.append(loo_di)
            consensus = train.mean(axis=0)
            cos = float(np.dot(held, consensus) /
                        (np.linalg.norm(held) * np.linalg.norm(consensus) + 1e-10))
            heldout_cosines.append(cos)

        loo_results.append({
            "drug": name,
            "n_cells": n,
            "mean_loo_di": float(np.mean(loo_instabilities)),
            "mean_heldout_cosine": float(np.mean(heldout_cosines)),
            "min_heldout_cosine": float(np.min(heldout_cosines)),
            "frac_consistent": float(np.mean([c > 0.3 for c in heldout_cosines])),
            "moa": exp1_by_name[name].get("moa"),
            "full_di": exp1_by_name[name]["direction_instability"],
        })

    print(f"  Drugs tested: {len(loo_results)}")

    # H25: correlation
    dis = [r["mean_loo_di"] for r in loo_results]
    coss = [r["mean_heldout_cosine"] for r in loo_results]
    rho, p = spearmanr(dis, coss)
    print(f"\n  H25: Spearman(loo_instability, mean_heldout_cosine) = {rho:.4f}, p = {p:.2e}")
    print(f"  Criterion: |rho| > 0.3 → {'PASS' if abs(rho) > 0.3 else 'FAIL'}")

    # H26: HDAC gradient
    print(f"\n  H26: HDAC selectivity gradient in LOO prediction:")
    hdac_results = [r for r in loo_results if r["moa"] and "HDAC inhibitor" in str(r["moa"])]
    for r in sorted(hdac_results, key=lambda x: x["mean_loo_di"]):
        print(f"    {r['drug']:25s}  loo_di={r['mean_loo_di']:.3f}  "
              f"heldout_cos={r['mean_heldout_cosine']:.3f}  frac_consistent={r['frac_consistent']:.2f}")

    # H27: AUROC for binary prediction
    frac_consistent = [r["frac_consistent"] for r in loo_results]
    binary_transport = [1 if f > 0.5 else 0 for f in frac_consistent]
    neg_di = [-d for d in dis]
    if len(set(binary_transport)) > 1:
        auroc = roc_auc_score(binary_transport, neg_di)
        print(f"\n  H27: AUROC(loo_instability → transport) = {auroc:.4f}")
        print(f"  Criterion: AUROC > 0.65 → {'PASS' if auroc > 0.65 else 'FAIL'}")
    else:
        auroc = None
        print(f"\n  H27: Only one class present, cannot compute AUROC")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(dis, coss, alpha=0.05, s=3, color="steelblue")
    if hdac_results:
        hdac_dis = [r["mean_loo_di"] for r in hdac_results]
        hdac_cos = [r["mean_heldout_cosine"] for r in hdac_results]
        axes[0].scatter(hdac_dis, hdac_cos, s=30, color="crimson", zorder=5, label="HDAC inhibitors")
        for r in hdac_results:
            axes[0].annotate(r["drug"][:12], (r["mean_loo_di"], r["mean_heldout_cosine"]),
                             fontsize=5, alpha=0.7, xytext=(3, 3), textcoords="offset points")
        axes[0].legend(fontsize=8)
    axes[0].set_xlabel("LOO direction instability")
    axes[0].set_ylabel("Mean held-out cosine")
    axes[0].set_title(f"H25: LOO prediction (ρ={rho:.3f})")

    # Panel 2: HDAC bar chart
    if hdac_results:
        hdac_sorted = sorted(hdac_results, key=lambda x: x["mean_loo_di"])
        y = range(len(hdac_sorted))
        axes[1].barh(y, [r["mean_heldout_cosine"] for r in hdac_sorted],
                     color="steelblue", height=0.6)
        axes[1].set_yticks(list(y))
        axes[1].set_yticklabels([r["drug"] for r in hdac_sorted], fontsize=7)
        axes[1].set_xlabel("Mean held-out cosine")
        axes[1].set_title("H26: HDAC selectivity gradient (LOO)")
        axes[1].invert_yaxis()

    # Panel 3: histogram of held-out cosines by instability quartile
    di_arr = np.array(dis)
    cos_arr = np.array(coss)
    q25, q75 = np.percentile(di_arr, [25, 75])
    low_mask = di_arr <= q25
    high_mask = di_arr >= q75
    axes[2].hist(cos_arr[low_mask], bins=40, alpha=0.6, color="steelblue",
                 label=f"Low instability (Q1, n={low_mask.sum()})", density=True)
    axes[2].hist(cos_arr[high_mask], bins=40, alpha=0.6, color="crimson",
                 label=f"High instability (Q4, n={high_mask.sum()})", density=True)
    axes[2].set_xlabel("Mean held-out cosine")
    axes[2].set_ylabel("Density")
    axes[2].set_title("Held-out cosine by instability quartile")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(outdir / "holdout_prediction.png", dpi=150, bbox_inches="tight")
    print(f"  Saved {outdir / 'holdout_prediction.png'}")

    with open(outdir / "holdout_prediction.json", "w") as f:
        json.dump(loo_results, f, indent=2, default=str)

    return {"n_drugs": len(loo_results),
            "h25_rho": float(rho), "h25_p": float(p),
            "h27_auroc": float(auroc) if auroc is not None else None,
            "mean_heldout_cos_q1": float(cos_arr[low_mask].mean()),
            "mean_heldout_cos_q4": float(cos_arr[high_mask].mean())}


# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", default="data/frozen_drug_labels.json")
    parser.add_argument("--metadata", default="data/GSE92742_Broad_LINCS_sig_info.txt.gz")
    parser.add_argument("--cellinfo", default="data/GSE92742_Broad_LINCS_cell_info.txt.gz")
    parser.add_argument("--exp1-results", default="results/01_cross_cellline/real_results.json")
    parser.add_argument("--shrna-data", default="data/lincs_shrna.npz")
    parser.add_argument("--shrna-meta", default="data/lincs_shrna_siginfo.csv.gz")
    parser.add_argument("--outdir", default="results/03_core_defenses")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_sigs, gene_ids, sig_id_to_idx, siginfo, cellinfo, drug_labels, exp1_by_name = load_all(args)
    drug_data, drug_cell_map = build_consensus(siginfo, sig_id_to_idx, all_sigs)
    print(f"  Drugs with >=5 cell lines: {len(drug_data)}")

    results = {}
    results["toxicity"] = run_toxicity_confound(drug_data, exp1_by_name, outdir)
    results["tissue"] = run_tissue_enrichment(drug_data, exp1_by_name, cellinfo, outdir)
    results["core_genes"] = run_core_genes(drug_data, exp1_by_name, gene_ids, outdir)
    results["genetic"] = run_genetic_triangulation(
        drug_data, exp1_by_name, drug_labels, args.shrna_data, args.shrna_meta, outdir
    )
    results["holdout"] = run_holdout_prediction(drug_data, exp1_by_name, outdir)

    with open(outdir / "all_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[{datetime.now():%H:%M:%S}] Done. Results in {outdir}/")


if __name__ == "__main__":
    main()
