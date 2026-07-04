"""Modal script: download LINCS GCTX on cloud, extract target drug signatures.

Downloads the full Level 5 GCTX from GEO (~4GB compressed), extracts
signatures for drugs profiled across 5+ cell lines, saves a compact .npz
file (~50-100MB) that you pull back to local.

Usage:
    # First: run metadata-only locally to get target sig IDs
    uv run python experiments/01_cross_cellline_transport.py --metadata-only

    # Then: extract on Modal
    modal run --detach experiments/modal_extract_lincs.py

    # Then: download the .npz from Modal volume
    modal volume get drug-perturbation-vol lincs_subset.npz data/lincs_subset.npz
"""
import modal

app = modal.App("lincs-extract")

vol = modal.Volume.from_name("drug-perturbation-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pandas", "cmapPy", "h5py", "tqdm", "matplotlib")
)

GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl"
GCTX_FILE = "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz"
SIGINFO_FILE = "GSE92742_Broad_LINCS_sig_info.txt.gz"
PERTINFO_FILE = "GSE92742_Broad_LINCS_pert_info.txt.gz"
GENEINFO_FILE = "GSE92742_Broad_LINCS_gene_info.txt.gz"


@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=86400,
    cpu=4,
    memory=32768,
)
def extract_signatures(min_cell_lines: int = 5):
    import gzip
    import json
    import shutil
    import urllib.request
    from datetime import datetime
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    work_dir = Path("/data")

    def ts():
        return datetime.now().strftime("%H:%M:%S")

    # Download metadata
    for name, filename in [("siginfo", SIGINFO_FILE), ("pertinfo", PERTINFO_FILE),
                           ("geneinfo", GENEINFO_FILE)]:
        local = work_dir / filename
        if not local.exists():
            url = f"{GEO_BASE}/{filename}"
            print(f"[{ts()}] Downloading {name}: {url}")
            urllib.request.urlretrieve(url, local)
            print(f"[{ts()}] Downloaded {name}: {local.stat().st_size / 1e6:.1f} MB")
        else:
            print(f"[{ts()}] {name} already exists")

    # Download GCTX
    gctx_gz = work_dir / GCTX_FILE
    gctx_path = work_dir / GCTX_FILE.replace(".gz", "")

    if not gctx_path.exists():
        if not gctx_gz.exists():
            url = f"{GEO_BASE}/{GCTX_FILE}"
            print(f"[{ts()}] Downloading GCTX (~4GB): {url}")
            urllib.request.urlretrieve(url, gctx_gz)
            print(f"[{ts()}] Downloaded GCTX: {gctx_gz.stat().st_size / 1e9:.2f} GB")

        print(f"[{ts()}] Decompressing GCTX...")
        with gzip.open(gctx_gz, "rb") as f_in, open(gctx_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        print(f"[{ts()}] Decompressed: {gctx_path.stat().st_size / 1e9:.2f} GB")

    # Load metadata
    print(f"[{ts()}] Loading siginfo...")
    siginfo = pd.read_csv(work_dir / SIGINFO_FILE, sep="\t", low_memory=False)
    siginfo_cp = siginfo[siginfo["pert_type"] == "trt_cp"].copy()
    print(f"  Compound signatures: {len(siginfo_cp):,}")

    print(f"[{ts()}] Loading geneinfo...")
    geneinfo = pd.read_csv(work_dir / GENEINFO_FILE, sep="\t", low_memory=False)
    landmark_genes = geneinfo[geneinfo["pr_is_lm"] == 1]["pr_gene_id"].astype(str).tolist()
    print(f"  Landmark genes: {len(landmark_genes)}")

    # Find drugs across 5+ cell lines
    drug_cells = (
        siginfo_cp.groupby("pert_iname")
        .agg(n_cell_lines=("cell_id", "nunique"), n_sigs=("sig_id", "count"))
        .reset_index()
    )
    target_drugs = drug_cells[drug_cells["n_cell_lines"] >= min_cell_lines]
    target_drug_names = set(target_drugs["pert_iname"])
    print(f"[{ts()}] Drugs with {min_cell_lines}+ cell lines: {len(target_drug_names)}")

    target_sigs = siginfo_cp[siginfo_cp["pert_iname"].isin(target_drug_names)]
    target_sig_ids = target_sigs["sig_id"].tolist()
    print(f"  Target signatures to extract: {len(target_sig_ids):,}")

    # Extract from GCTX (landmark genes only)
    print(f"[{ts()}] Extracting signatures from GCTX (landmark genes only)...")
    from cmapPy.pandasGEXpress import parse

    # Extract in batches to manage memory
    batch_size = 5000
    all_data = []
    all_ids = []

    for i in tqdm(range(0, len(target_sig_ids), batch_size),
                  desc="Extracting batches"):
        batch_ids = target_sig_ids[i:i + batch_size]
        try:
            gctoo = parse.parse(str(gctx_path), cid=batch_ids, rid=landmark_genes)
            all_data.append(gctoo.data_df.values.T)  # (n_sigs, n_genes)
            all_ids.extend(gctoo.data_df.columns.tolist())
        except Exception as e:
            print(f"  Warning: batch {i} failed: {e}")
            continue

    signatures = np.vstack(all_data).astype(np.float32)
    sig_ids = all_ids
    gene_ids = landmark_genes

    print(f"[{ts()}] Extracted: {signatures.shape} ({signatures.nbytes / 1e6:.1f} MB)")

    # Save compact .npz
    out_path = work_dir / "lincs_subset.npz"
    np.savez_compressed(
        out_path,
        signatures=signatures,
        sig_ids=np.array(sig_ids, dtype=object),
        gene_ids=np.array(gene_ids, dtype=object),
    )
    print(f"[{ts()}] Saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Also save the filtered siginfo for these drugs
    target_sigs.to_parquet(work_dir / "lincs_subset_siginfo.parquet")

    # Save pertinfo
    pertinfo = pd.read_csv(work_dir / PERTINFO_FILE, sep="\t", low_memory=False)
    target_pertinfo = pertinfo[pertinfo["pert_iname"].isin(target_drug_names)]
    target_pertinfo.to_parquet(work_dir / "lincs_subset_pertinfo.parquet")

    vol.commit()
    print(f"[{ts()}] Done. Volume committed.")
    print(f"\nTo download locally:")
    print(f"  modal volume get drug-perturbation-vol lincs_subset.npz data/lincs_subset.npz")
    print(f"  modal volume get drug-perturbation-vol lincs_subset_siginfo.parquet data/lincs_subset_siginfo.parquet")
    print(f"  modal volume get drug-perturbation-vol lincs_subset_pertinfo.parquet data/lincs_subset_pertinfo.parquet")

    return {
        "n_drugs": len(target_drug_names),
        "n_signatures": len(sig_ids),
        "n_genes": len(gene_ids),
        "file_size_mb": out_path.stat().st_size / 1e6,
    }


@app.local_entrypoint()
def main():
    result = extract_signatures.remote(min_cell_lines=5)
    print(f"\nExtraction complete: {result}")
