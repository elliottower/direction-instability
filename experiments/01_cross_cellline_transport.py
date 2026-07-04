"""Experiment 01: Cross-cell-line transport of drug perturbation signatures.

Core question: does the direction of a drug's perturbation signature
stay stable across cell lines? The bracket norm analogue (direction
instability) should predict which drugs have transportable mechanisms.

Modes:
    --synthetic     Run on synthetic data to validate pipeline (default)
    --real          Run on real LINCS data (requires extracted .npz)
    --metadata-only Download metadata and report drug/cell-line landscape

Usage:
    uv run python experiments/01_cross_cellline_transport.py --synthetic
    uv run python experiments/01_cross_cellline_transport.py --metadata-only
    uv run python experiments/01_cross_cellline_transport.py --real --data data/lincs_subset.npz
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.lincs_loader import (
    download_metadata,
    find_multi_cellline_drugs,
    load_pertinfo,
    load_siginfo,
    make_synthetic_data,
    save_results,
)
from geometry.drug_transport import compute_all_transport_metrics


def timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def run_synthetic(n_drugs: int = 200, n_cell_lines: int = 8, seed: int = 42) -> None:
    """Validate pipeline on synthetic data with known ground truth."""
    print(f"[{timestamp()}] Generating synthetic data: {n_drugs} drugs, {n_cell_lines} cell lines")

    rng = np.random.default_rng(seed)
    drug_sigs, drug_meta = make_synthetic_data(
        n_drugs=n_drugs,
        n_cell_lines=n_cell_lines,
        frac_transporting=0.3,
        rng=rng,
    )

    results = []
    for drug_name in tqdm(sorted(drug_sigs.keys()), desc="Computing transport metrics"):
        sigs = drug_sigs[drug_name]
        metrics = compute_all_transport_metrics(sigs)
        metrics["drug_name"] = drug_name
        metrics["transports"] = drug_meta[drug_name]["transports"]
        metrics["moa"] = drug_meta[drug_name]["moa"]
        results.append(metrics)

    results_dir = Path("results/01_cross_cellline")
    results_dir.mkdir(parents=True, exist_ok=True)
    save_results(results, results_dir / "synthetic_results.json")

    df = pd.DataFrame(results)
    print(f"\n[{timestamp()}] Results summary:")
    print(f"  Total drugs: {len(df)}")
    print(f"  Transporting: {df['transports'].sum()}")
    print(f"  Non-transporting: {(~df['transports']).sum()}")

    for metric in ["direction_instability", "mean_pairwise_cosine", "frechet_variance",
                    "magnitude_cv", "mean_top_gene_jaccard"]:
        if metric in df.columns:
            t_vals = df[df["transports"]][metric].dropna()
            nt_vals = df[~df["transports"]][metric].dropna()
            if len(t_vals) > 0 and len(nt_vals) > 0:
                print(f"\n  {metric}:")
                print(f"    Transporting:     {t_vals.mean():.4f} +/- {t_vals.std():.4f}")
                print(f"    Non-transporting: {nt_vals.mean():.4f} +/- {nt_vals.std():.4f}")

    from scipy.stats import mannwhitneyu
    for metric in ["direction_instability", "frechet_variance", "mean_top_gene_jaccard"]:
        if metric in df.columns:
            t_vals = df[df["transports"]][metric].dropna()
            nt_vals = df[~df["transports"]][metric].dropna()
            if len(t_vals) > 1 and len(nt_vals) > 1:
                stat, p = mannwhitneyu(t_vals, nt_vals, alternative="two-sided")
                print(f"\n  Mann-Whitney U ({metric}): U={stat:.0f}, p={p:.2e}")

    plot_synthetic_results(df, results_dir)
    print(f"\n[{timestamp()}] Done. Results in {results_dir}/")


def plot_synthetic_results(df: pd.DataFrame, out_dir: Path) -> None:
    """Generate diagnostic plots for synthetic validation."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Synthetic validation: transporting vs non-transporting drugs", fontsize=14)

    metrics = [
        ("direction_instability", "Direction instability\n(bracket norm analogue)"),
        ("frechet_variance", "Frechet variance\n(spherical dispersion)"),
        ("magnitude_cv", "Magnitude CV\n(effect size stability)"),
        ("mean_top_gene_jaccard", "Top gene Jaccard\n(gene-level consistency)"),
    ]

    for ax, (metric, label) in zip(axes.flat, metrics):
        if metric not in df.columns:
            ax.set_visible(False)
            continue

        t_vals = df[df["transports"]][metric].dropna()
        nt_vals = df[~df["transports"]][metric].dropna()

        ax.hist(t_vals, bins=20, alpha=0.6, label=f"Transporting (n={len(t_vals)})", color="#2196F3")
        ax.hist(nt_vals, bins=20, alpha=0.6, label=f"Non-transporting (n={len(nt_vals)})", color="#F44336")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out_dir / "synthetic_validation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{timestamp()}] Saved plot: {out_dir / 'synthetic_validation.png'}")


def run_metadata_only(data_dir: Path) -> None:
    """Download metadata and report the drug/cell-line landscape."""
    print(f"[{timestamp()}] Downloading LINCS metadata to {data_dir}")
    paths = download_metadata(data_dir)

    print(f"\n[{timestamp()}] Loading signature metadata...")
    siginfo = load_siginfo(paths["siginfo"])
    print(f"  Total compound signatures: {len(siginfo):,}")
    print(f"  Unique compounds: {siginfo['pert_iname'].nunique():,}")
    print(f"  Unique cell lines: {siginfo['cell_id'].nunique():,}")

    print(f"\n[{timestamp()}] Finding drugs across multiple cell lines...")
    for min_cl in [3, 5, 8, 10, 15, 20]:
        multi = find_multi_cellline_drugs(siginfo, min_cell_lines=min_cl)
        print(f"  >= {min_cl:2d} cell lines: {len(multi):5d} drugs")

    multi5 = find_multi_cellline_drugs(siginfo, min_cell_lines=5)
    print(f"\n[{timestamp()}] Top 20 drugs by cell line coverage:")
    for _, row in multi5.head(20).iterrows():
        print(f"  {row['pert_iname']:30s}  {row['n_cell_lines']:3d} cell lines  {row['n_total_signatures']:5d} sigs")

    results_dir = Path("results/01_cross_cellline")
    results_dir.mkdir(parents=True, exist_ok=True)
    multi5.to_csv(results_dir / "drugs_5plus_celllines.csv", index=False)
    print(f"\n[{timestamp()}] Saved drug list to {results_dir / 'drugs_5plus_celllines.csv'}")

    print(f"\n[{timestamp()}] Loading perturbation metadata (MOA annotations)...")
    pertinfo = load_pertinfo(paths["pertinfo"])
    if "moa" in pertinfo.columns:
        has_moa = pertinfo[pertinfo["moa"].notna() & (pertinfo["moa"] != "-666")]
        print(f"  Perturbations with MOA annotation: {len(has_moa):,} / {len(pertinfo):,}")

        merged = multi5.merge(pertinfo[["pert_iname", "moa"]], on="pert_iname", how="left")
        merged_moa = merged[merged["moa"].notna() & (merged["moa"] != "-666")]
        print(f"  Multi-cell-line drugs with MOA: {merged_moa['pert_iname'].nunique()}")

        moa_counts = merged_moa["moa"].value_counts().head(20)
        print(f"\n  Top MOA classes (drugs with 5+ cell lines):")
        for moa, count in moa_counts.items():
            print(f"    {moa:50s}  {count:4d}")

    siginfo_target = siginfo[siginfo["pert_iname"].isin(multi5["pert_iname"])]
    target_sig_ids = siginfo_target["sig_id"].tolist()
    sig_ids_path = results_dir / "target_sig_ids.json"
    with open(sig_ids_path, "w") as f:
        json.dump(target_sig_ids, f)
    print(f"\n[{timestamp()}] Saved {len(target_sig_ids):,} target signature IDs to {sig_ids_path}")
    print(f"  (Use these to extract signatures from GCTX on Modal)")


def load_frozen_labels(labels_path: Path) -> dict[str, dict]:
    """Load pre-registered frozen drug labels (MOA, target, clinical phase)."""
    with open(labels_path) as f:
        data = json.load(f)

    lookup = {}
    for drug in data["drugs"]:
        lookup[drug["pert_iname"].lower()] = drug

    print(f"[{timestamp()}] Loaded frozen labels: {len(lookup)} drugs")
    print(f"  Source: {data.get('moa_source', 'unknown')}")
    print(f"  Frozen date: {data.get('frozen_date', 'unknown')}")
    return lookup


def permutation_null(
    signatures: dict[str, np.ndarray],
    n_perms: int = 1000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Permutation null for direction instability.

    Shuffles cell-line labels and recomputes direction instability.
    The observed value should exceed the null for drugs with genuinely
    context-dependent mechanisms.
    """
    if rng is None:
        rng = np.random.default_rng()

    from geometry.drug_transport import direction_stability

    observed = direction_stability(signatures)["direction_instability"]

    names = sorted(signatures.keys())
    vecs = [signatures[n] for n in names]
    null_dist = np.zeros(n_perms)
    for p in range(n_perms):
        perm = rng.permutation(len(vecs))
        shuffled = {names[i]: vecs[perm[i]] for i in range(len(names))}
        null_dist[p] = direction_stability(shuffled)["direction_instability"]

    p_value = float(np.mean(null_dist >= observed))
    return {
        "observed": observed,
        "null_mean": float(np.mean(null_dist)),
        "null_std": float(np.std(null_dist)),
        "p_value_permutation": p_value,
    }


def run_real(data_path: Path, metadata_dir: Path, n_perms: int = 200) -> None:
    """Run transport analysis on real LINCS data (extracted .npz)."""
    print(f"[{timestamp()}] Loading extracted signatures from {data_path}")

    data = np.load(data_path, allow_pickle=True)
    signatures_matrix = data["signatures"]  # (n_sigs, n_genes)
    sig_ids = data["sig_ids"].tolist()
    gene_ids = data["gene_ids"].tolist() if "gene_ids" in data else None

    print(f"  Signatures: {signatures_matrix.shape}")
    print(f"  Genes: {len(gene_ids) if gene_ids else 'unknown'}")

    # Load frozen labels (pre-registered MOA annotations)
    labels_path = metadata_dir / "frozen_drug_labels.json"
    frozen_labels = load_frozen_labels(labels_path)

    # Load siginfo for signature metadata
    from data.lincs_loader import METADATA_FILES
    siginfo_path = metadata_dir / METADATA_FILES["siginfo"]
    siginfo = load_siginfo(siginfo_path)

    target_siginfo = siginfo[siginfo["sig_id"].isin(sig_ids)]
    drugs = target_siginfo["pert_iname"].unique()
    print(f"  Drugs in extracted data: {len(drugs)}")

    rng = np.random.default_rng(42)
    results = []
    for drug_name in tqdm(drugs, desc="Computing transport metrics"):
        drug_sigs = target_siginfo[target_siginfo["pert_iname"] == drug_name]
        sig_id_to_row = {sid: i for i, sid in enumerate(sig_ids)}

        if "distil_ss" in drug_sigs.columns:
            best = drug_sigs.loc[drug_sigs.groupby("cell_id")["distil_ss"].idxmax()]
        else:
            best = drug_sigs.groupby("cell_id").first().reset_index()

        cell_sigs = {}
        for _, row in best.iterrows():
            if row["sig_id"] in sig_id_to_row:
                cell_sigs[row["cell_id"]] = signatures_matrix[sig_id_to_row[row["sig_id"]]]

        if len(cell_sigs) < 3:
            continue

        metrics = compute_all_transport_metrics(cell_sigs)
        metrics["drug_name"] = drug_name
        metrics["n_cell_lines"] = len(cell_sigs)

        # Attach frozen labels (pre-registered, not derived from data)
        label = frozen_labels.get(drug_name.lower(), {})
        metrics["moa"] = label.get("moa")
        metrics["target"] = label.get("target")
        metrics["clinical_phase"] = label.get("clinical_phase")
        metrics["disease_area"] = label.get("disease_area")

        # Permutation null (subsample for speed)
        if n_perms > 0 and len(cell_sigs) >= 4:
            perm = permutation_null(cell_sigs, n_perms=n_perms, rng=rng)
            metrics["p_value_permutation"] = perm["p_value_permutation"]
            metrics["null_mean"] = perm["null_mean"]

        results.append(metrics)

    results_dir = Path("results/01_cross_cellline")
    results_dir.mkdir(parents=True, exist_ok=True)
    save_results(results, results_dir / "real_results.json")

    df = pd.DataFrame(results)
    print(f"\n[{timestamp()}] Results summary:")
    print(f"  Drugs analyzed: {len(df)}")
    print(f"  With MOA annotation: {df['moa'].notna().sum()}")
    print(f"  Launched drugs: {(df['clinical_phase'] == 'Launched').sum()}")

    print(f"\n  Direction instability (bracket norm analogue):")
    print(f"    Mean: {df['direction_instability'].mean():.4f}")
    print(f"    Median: {df['direction_instability'].median():.4f}")
    print(f"    Std: {df['direction_instability'].std():.4f}")

    # H3: direction instability vs gene Jaccard
    from scipy.stats import spearmanr
    if "mean_top_gene_jaccard" in df.columns:
        valid = df[["direction_instability", "mean_top_gene_jaccard"]].dropna()
        if len(valid) > 10:
            rho, p = spearmanr(valid["direction_instability"], valid["mean_top_gene_jaccard"])
            print(f"\n  H3 test: direction_instability vs gene Jaccard:")
            print(f"    Spearman rho = {rho:.4f}, p = {p:.2e}")

    if df["moa"].notna().sum() > 10:
        plot_moa_stratified(df, results_dir)

    # Summary of pre-registered hypotheses
    print(f"\n[{timestamp()}] Pre-registered hypothesis results:")
    if "p_value_permutation" in df.columns:
        sig_perm = (df["p_value_permutation"] < 0.05).sum()
        print(f"  Drugs with significant permutation test (p<0.05): {sig_perm} / {len(df)}")

    print(f"\n[{timestamp()}] Done. Results in {results_dir}/")


def plot_moa_stratified(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot transport metrics stratified by mechanism of action."""
    df_moa = df[df["moa"].notna()].copy()

    moa_counts = df_moa["moa"].value_counts()
    top_moa = moa_counts[moa_counts >= 5].index.tolist()[:12]
    df_top = df_moa[df_moa["moa"].isin(top_moa)]

    if len(df_top) < 10:
        print(f"[{timestamp()}] Too few drugs with frequent MOA for stratified plot")
        return

    moa_medians = df_top.groupby("moa")["direction_instability"].median().sort_values()
    ordered_moa = moa_medians.index.tolist()

    fig, ax = plt.subplots(figsize=(12, max(6, len(ordered_moa) * 0.5)))

    positions = []
    labels = []
    for i, moa in enumerate(ordered_moa):
        vals = df_top[df_top["moa"] == moa]["direction_instability"].values
        positions.append(vals)
        n = len(vals)
        labels.append(f"{moa}\n(n={n})")

    parts = ax.violinplot(positions, vert=False, showmedians=True)
    ax.set_yticks(range(1, len(labels) + 1))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Direction instability (bracket norm analogue)\n← transports | context-specific →")
    ax.set_title("Drug mechanism transport by MOA class")

    plt.tight_layout()
    fig.savefig(out_dir / "moa_stratified_transport.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{timestamp()}] Saved MOA plot: {out_dir / 'moa_stratified_transport.png'}")


def main():
    parser = argparse.ArgumentParser(description="Cross-cell-line drug transport analysis")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic", action="store_true", help="Run on synthetic data")
    mode.add_argument("--metadata-only", action="store_true", help="Download metadata and report landscape")
    mode.add_argument("--real", action="store_true", help="Run on real LINCS data")

    parser.add_argument("--data", type=Path, help="Path to extracted .npz (for --real)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory for metadata")
    parser.add_argument("--n-drugs", type=int, default=200, help="Number of synthetic drugs")
    parser.add_argument("--n-perms", type=int, default=200, help="Permutation null iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic data")

    args = parser.parse_args()

    if args.synthetic:
        run_synthetic(n_drugs=args.n_drugs, seed=args.seed)
    elif args.metadata_only:
        run_metadata_only(args.data_dir)
    elif args.real:
        if args.data is None:
            parser.error("--real requires --data path to extracted .npz")
        run_real(args.data, args.data_dir, n_perms=args.n_perms)


if __name__ == "__main__":
    main()
