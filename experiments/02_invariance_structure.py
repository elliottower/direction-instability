"""
Experiment 02: Invariance structure analysis.

For each drug, find the invariance group G — the set of cell-line pairs
where the drug's mechanism is indistinguishable (high cosine similarity).
The structure of G (connected components, clique number, graph density)
captures WHICH contexts are equivalent, not just how many.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm


def pairwise_cosine_matrix(signatures: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(signatures, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    unit = signatures / norms
    return unit @ unit.T


def invariance_graph(cos_matrix: np.ndarray, threshold: float) -> np.ndarray:
    n = cos_matrix.shape[0]
    adj = (cos_matrix >= threshold).astype(int)
    np.fill_diagonal(adj, 0)
    return adj


def connected_components(adj: np.ndarray) -> list[list[int]]:
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
            for neighbor in range(n):
                if adj[node, neighbor] and not visited[neighbor]:
                    stack.append(neighbor)
        components.append(sorted(comp))
    return components


def graph_stats(adj: np.ndarray) -> dict:
    n = adj.shape[0]
    n_edges = adj.sum() // 2
    max_edges = n * (n - 1) // 2
    density = n_edges / max_edges if max_edges > 0 else 0.0

    comps = connected_components(adj)
    comp_sizes = sorted([len(c) for c in comps], reverse=True)
    largest_frac = comp_sizes[0] / n if n > 0 else 0.0

    degrees = adj.sum(axis=1)

    return {
        "n_nodes": int(n),
        "n_edges": int(n_edges),
        "density": float(density),
        "n_components": len(comps),
        "component_sizes": comp_sizes,
        "largest_component_frac": float(largest_frac),
        "mean_degree": float(degrees.mean()),
        "max_degree": int(degrees.max()),
        "isolated_nodes": int((degrees == 0).sum()),
    }


def analyze_drug_invariance(
    signatures: np.ndarray,
    cell_lines: list[str],
    thresholds: list[float],
) -> dict:
    cos_mat = pairwise_cosine_matrix(signatures)

    results = {
        "n_cell_lines": len(cell_lines),
        "cell_lines": cell_lines,
        "mean_pairwise_cosine": float(cos_mat[np.triu_indices_from(cos_mat, k=1)].mean()),
        "cosine_matrix": cos_mat.tolist(),
    }

    for thresh in thresholds:
        adj = invariance_graph(cos_mat, thresh)
        stats = graph_stats(adj)
        results[f"thresh_{thresh:.2f}"] = stats

    return results


def run(args):
    print(f"[{datetime.now():%H:%M:%S}] Loading signatures from {args.data}")
    data = np.load(args.data, allow_pickle=True)
    all_sigs = data["signatures"]
    all_sig_ids = data["sig_ids"]
    gene_ids = data["gene_ids"]
    print(f"  Signatures: {all_sigs.shape}")

    with open(args.labels) as f:
        labels = json.load(f)
    drug_labels = {d["pert_iname"]: d for d in labels["drugs"]}
    print(f"  Labels: {len(drug_labels)} drugs")

    import pandas as pd
    siginfo = pd.read_csv(args.metadata, sep="\t", low_memory=False, compression="infer")

    sig_id_to_idx = {sid: i for i, sid in enumerate(all_sig_ids)}

    drugs_by_name = {}
    for _, row in siginfo.iterrows():
        if row["pert_type"] != "trt_cp":
            continue
        name = row["pert_iname"]
        cell = row["cell_id"]
        sig_id = row["sig_id"]
        if sig_id not in sig_id_to_idx:
            continue
        if name not in drugs_by_name:
            drugs_by_name[name] = {}
        if cell not in drugs_by_name[name]:
            drugs_by_name[name][cell] = []
        drugs_by_name[name][cell].append(sig_id_to_idx[sig_id])

    thresholds = [0.3, 0.5, 0.7, 0.85]
    min_cells = args.min_cells

    eligible = {
        name: cells
        for name, cells in drugs_by_name.items()
        if len(cells) >= min_cells
    }
    print(f"  Drugs with >= {min_cells} cell lines: {len(eligible)}")

    results = []
    for drug_name in tqdm(sorted(eligible.keys()), desc="Invariance analysis"):
        cells = eligible[drug_name]
        cell_names = sorted(cells.keys())
        consensus_sigs = []
        for cell in cell_names:
            idxs = cells[cell]
            consensus_sigs.append(all_sigs[idxs].mean(axis=0))
        consensus_sigs = np.array(consensus_sigs)

        inv = analyze_drug_invariance(consensus_sigs, cell_names, thresholds)
        inv["drug_name"] = drug_name

        label = drug_labels.get(drug_name, {})
        inv["moa"] = label.get("moa")
        inv["target"] = label.get("target")

        del inv["cosine_matrix"]
        results.append(inv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "invariance_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[{datetime.now():%H:%M:%S}] Saved {len(results)} drugs to {outdir / 'invariance_results.json'}")

    print_summary(results, thresholds)
    plot_invariance(results, thresholds, outdir)


def print_summary(results: list[dict], thresholds: list[float]):
    print(f"\n{'='*70}")
    print("INVARIANCE STRUCTURE SUMMARY")
    print(f"{'='*70}")

    for thresh in thresholds:
        key = f"thresh_{thresh:.2f}"
        densities = [r[key]["density"] for r in results]
        n_comps = [r[key]["n_components"] for r in results]
        largest_fracs = [r[key]["largest_component_frac"] for r in results]

        print(f"\n  Threshold = {thresh}:")
        print(f"    Mean density:              {np.mean(densities):.4f}")
        print(f"    Mean n_components:         {np.mean(n_comps):.1f}")
        print(f"    Mean largest_comp_frac:    {np.mean(largest_fracs):.4f}")

        fully_connected = sum(1 for d in densities if d > 0.99)
        fully_fragmented = sum(1 for n in n_comps if n == len(results[0].get("cell_lines", [])))
        print(f"    Fully connected (density>0.99): {fully_connected}")

    thresh_key = f"thresh_{thresholds[1]:.2f}"

    hdac = [r for r in results if r.get("moa") and "HDAC inhibitor" in str(r["moa"])]
    if hdac:
        print(f"\n  HDAC inhibitors (threshold={thresholds[1]}):")
        hdac_sorted = sorted(hdac, key=lambda x: -x[thresh_key]["density"])
        for r in hdac_sorted:
            stats = r[thresh_key]
            print(f"    {r['drug_name']:25s}  density={stats['density']:.3f}  "
                  f"comps={stats['n_components']}  largest={stats['largest_component_frac']:.2f}  "
                  f"n={r['n_cell_lines']}")

    print(f"\n  Top 15 most invariant drugs (threshold={thresholds[1]}, density):")
    sorted_by_density = sorted(results, key=lambda x: -x[thresh_key]["density"])
    for r in sorted_by_density[:15]:
        stats = r[thresh_key]
        moa = str(r.get("moa") or "no MOA")[:45]
        print(f"    {r['drug_name']:25s}  density={stats['density']:.3f}  "
              f"comps={stats['n_components']}  n={r['n_cell_lines']}  MOA={moa}")

    print(f"\n  Top 15 most fragmented drugs (threshold={thresholds[1]}):")
    sorted_by_frag = sorted(results, key=lambda x: x[thresh_key]["density"])
    for r in sorted_by_frag[:15]:
        stats = r[thresh_key]
        moa = str(r.get("moa") or "no MOA")[:45]
        print(f"    {r['drug_name']:25s}  density={stats['density']:.3f}  "
              f"comps={stats['n_components']}  n={r['n_cell_lines']}  MOA={moa}")


def plot_invariance(results: list[dict], thresholds: list[float], outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, thresh in zip(axes.flat, thresholds):
        key = f"thresh_{thresh:.2f}"
        densities = [r[key]["density"] for r in results]

        hdac = [r for r in results if r.get("moa") and "HDAC inhibitor" in str(r["moa"])]
        topo = [r for r in results if r.get("moa") and "topoisomerase inhibitor" in str(r["moa"])]
        receptor = [r for r in results if r.get("moa") and "receptor" in str(r.get("moa", "")).lower()]
        other = [r for r in results if r not in hdac and r not in topo and r not in receptor]

        ax.hist([r[key]["density"] for r in other], bins=50, alpha=0.5, label=f"Other (n={len(other)})", color="gray")
        if receptor:
            ax.hist([r[key]["density"] for r in receptor], bins=30, alpha=0.7, label=f"Receptor (n={len(receptor)})", color="steelblue")
        if topo:
            ax.hist([r[key]["density"] for r in topo], bins=15, alpha=0.8, label=f"Topoisomerase (n={len(topo)})", color="darkorange")
        if hdac:
            ax.hist([r[key]["density"] for r in hdac], bins=15, alpha=0.8, label=f"HDAC (n={len(hdac)})", color="crimson")

        ax.set_xlabel("Graph density (fraction of invariant pairs)")
        ax.set_ylabel("Count")
        ax.set_title(f"Cosine threshold = {thresh}")
        ax.legend(fontsize=8)

    fig.suptitle("Invariance graph density by MOA class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(outdir / "invariance_density_by_moa.png", dpi=150, bbox_inches="tight")
    print(f"[{datetime.now():%H:%M:%S}] Saved {outdir / 'invariance_density_by_moa.png'}")

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    thresh_key = f"thresh_{thresholds[1]:.2f}"
    hdac = [r for r in results if r.get("moa") and "HDAC inhibitor" in str(r["moa"])]
    if hdac:
        hdac_sorted = sorted(hdac, key=lambda x: x[thresh_key]["density"], reverse=True)
        names = [r["drug_name"] for r in hdac_sorted]
        densities = [r[thresh_key]["density"] for r in hdac_sorted]
        n_comps = [r[thresh_key]["n_components"] for r in hdac_sorted]
        n_cells = [r["n_cell_lines"] for r in hdac_sorted]

        colors = ["crimson" if d > 0.3 else "lightcoral" if d > 0.1 else "lightgray" for d in densities]
        axes2[0].barh(range(len(names)), densities, color=colors)
        axes2[0].set_yticks(range(len(names)))
        axes2[0].set_yticklabels([f"{n} (n={nc})" for n, nc in zip(names, n_cells)], fontsize=8)
        axes2[0].set_xlabel(f"Invariance graph density (threshold={thresholds[1]})")
        axes2[0].set_title("HDAC inhibitors: invariance density")
        axes2[0].invert_yaxis()

        axes2[1].barh(range(len(names)), n_comps, color=colors)
        axes2[1].set_yticks(range(len(names)))
        axes2[1].set_yticklabels([f"{n} (n={nc})" for n, nc in zip(names, n_cells)], fontsize=8)
        axes2[1].set_xlabel("Number of equivalence classes (components)")
        axes2[1].set_title("HDAC inhibitors: fragmentation")
        axes2[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(outdir / "hdac_invariance_structure.png", dpi=150, bbox_inches="tight")
    print(f"[{datetime.now():%H:%M:%S}] Saved {outdir / 'hdac_invariance_structure.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to lincs_subset.npz")
    parser.add_argument("--labels", default="data/frozen_drug_labels.json")
    parser.add_argument("--metadata", default="data/GSE92742_Broad_LINCS_sig_info.txt.gz")
    parser.add_argument("--outdir", default="results/02_invariance")
    parser.add_argument("--min-cells", type=int, default=5)
    args = parser.parse_args()
    run(args)
