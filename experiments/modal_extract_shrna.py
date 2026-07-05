"""Modal script: extract shRNA signatures from LINCS GCTX.

Reuses the already-downloaded GCTX on the drug-perturbation-vol volume.
Extracts trt_sh (shRNA knockdown) signatures for genes that are targets
of drugs in our dataset, to enable genetic triangulation (H20-H21).

Usage:
    modal run --detach experiments/modal_extract_shrna.py
    modal volume get drug-perturbation-vol lincs_shrna.npz data/lincs_shrna.npz
"""
import modal

app = modal.App("lincs-shrna-extract")

vol = modal.Volume.from_name("drug-perturbation-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pandas", "cmapPy", "h5py", "tqdm", "matplotlib")
)

GCTX_FILE = "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx"
SIGINFO_FILE = "GSE92742_Broad_LINCS_sig_info.txt.gz"
GENEINFO_FILE = "GSE92742_Broad_LINCS_gene_info.txt.gz"


@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=86400,
    cpu=4,
    memory=32768,
)
def extract_shrna():
    import json
    from datetime import datetime
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    work_dir = Path("/data")

    def ts():
        return datetime.now().strftime("%H:%M:%S")

    gctx_path = work_dir / GCTX_FILE
    if not gctx_path.exists():
        print(f"[{ts()}] ERROR: GCTX not found at {gctx_path}")
        print("  Run modal_extract_lincs.py first to download the GCTX")
        return {"error": "GCTX not found"}

    print(f"[{ts()}] Loading siginfo...")
    siginfo = pd.read_csv(work_dir / SIGINFO_FILE, sep="\t", low_memory=False)

    shrna_sigs = siginfo[siginfo["pert_type"] == "trt_sh"].copy()
    print(f"  shRNA signatures total: {len(shrna_sigs):,}")
    print(f"  Unique genes targeted: {shrna_sigs['pert_iname'].nunique()}")

    print(f"[{ts()}] Loading geneinfo...")
    geneinfo = pd.read_csv(work_dir / GENEINFO_FILE, sep="\t", low_memory=False)
    landmark_genes = geneinfo[geneinfo["pr_is_lm"] == 1]["pr_gene_id"].astype(str).tolist()
    print(f"  Landmark genes: {len(landmark_genes)}")

    target_sig_ids = shrna_sigs["sig_id"].tolist()
    print(f"[{ts()}] Extracting {len(target_sig_ids):,} shRNA signatures...")

    from cmapPy.pandasGEXpress import parse

    batch_size = 5000
    all_data = []
    all_ids = []

    for i in tqdm(range(0, len(target_sig_ids), batch_size), desc="Extracting batches"):
        batch_ids = target_sig_ids[i:i + batch_size]
        try:
            gctoo = parse.parse(str(gctx_path), cid=batch_ids, rid=landmark_genes)
            all_data.append(gctoo.data_df.values.T)
            all_ids.extend(gctoo.data_df.columns.tolist())
        except Exception as e:
            print(f"  Warning: batch {i} failed: {e}")
            continue

    signatures = np.vstack(all_data).astype(np.float32)
    print(f"[{ts()}] Extracted: {signatures.shape} ({signatures.nbytes / 1e6:.1f} MB)")

    out_path = work_dir / "lincs_shrna.npz"
    np.savez_compressed(
        out_path,
        signatures=signatures,
        sig_ids=np.array(all_ids, dtype=object),
        gene_ids=np.array(landmark_genes, dtype=object),
    )
    print(f"[{ts()}] Saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    shrna_meta = shrna_sigs[shrna_sigs["sig_id"].isin(all_ids)]
    shrna_meta.to_csv(work_dir / "lincs_shrna_siginfo.csv.gz", index=False, compression="gzip")
    print(f"[{ts()}] Saved shrna siginfo: {len(shrna_meta)} rows")

    vol.commit()
    print(f"[{ts()}] Done. Volume committed.")
    print(f"\nTo download locally:")
    print(f"  modal volume get drug-perturbation-vol lincs_shrna.npz data/lincs_shrna.npz")
    print(f"  modal volume get drug-perturbation-vol lincs_shrna_siginfo.csv.gz data/lincs_shrna_siginfo.csv.gz")

    return {
        "n_signatures": len(all_ids),
        "n_genes": len(landmark_genes),
        "n_unique_targets": shrna_meta["pert_iname"].nunique(),
        "file_size_mb": out_path.stat().st_size / 1e6,
    }


@app.local_entrypoint()
def main():
    result = extract_shrna.remote()
    print(f"\nExtraction complete: {result}")
