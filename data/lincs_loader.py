"""LINCS L1000 data loading and metadata handling.

Data source: GEO GSE92742 (Phase 1) and GSE70138 (Phase 2).
Level 5 = replicate-collapsed z-scores (MODZ), 978 landmark genes.

Workflow:
1. download_metadata() — pulls siginfo, pertinfo, cellinfo (~50MB total)
2. find_multi_cellline_drugs() — identifies drugs across N+ cell lines
3. load_signatures() — reads actual signatures from GCTX file

The GCTX file (GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz)
is ~4GB compressed and must be downloaded manually from GEO. All other
metadata files are small enough to download programmatically.
"""
import gzip
import json
import logging
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl"

METADATA_FILES = {
    "siginfo": "GSE92742_Broad_LINCS_sig_info.txt.gz",
    "pertinfo": "GSE92742_Broad_LINCS_pert_info.txt.gz",
    "cellinfo": "GSE92742_Broad_LINCS_cell_info.txt.gz",
    "geneinfo": "GSE92742_Broad_LINCS_gene_info.txt.gz",
}

GCTX_FILE = "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz"

N_LANDMARK_GENES = 978


def download_metadata(data_dir: Path, force: bool = False) -> dict[str, Path]:
    """Download LINCS metadata files from GEO FTP.

    Returns dict mapping metadata type to local file path.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for name, filename in METADATA_FILES.items():
        local_path = data_dir / filename
        if local_path.exists() and not force:
            logger.info(f"{name}: already exists at {local_path}")
            paths[name] = local_path
            continue

        url = f"{GEO_BASE}/{filename}"
        logger.info(f"Downloading {name} from {url}...")
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] Downloading {name}: {url}")

        urllib.request.urlretrieve(url, local_path)

        ts = datetime.now().strftime("%H:%M:%S")
        size_mb = local_path.stat().st_size / 1e6
        print(f"[{ts}] Downloaded {name}: {size_mb:.1f} MB")
        paths[name] = local_path

    return paths


def load_siginfo(path: Path, compound_only: bool = True) -> pd.DataFrame:
    """Load signature metadata.

    Args:
        path: path to siginfo.txt.gz
        compound_only: if True, filter to pert_type == 'trt_cp'
    """
    df = pd.read_csv(path, sep="\t", low_memory=False)
    if compound_only:
        df = df[df["pert_type"] == "trt_cp"].copy()
    return df


def load_pertinfo(path: Path) -> pd.DataFrame:
    """Load perturbation metadata (MOA annotations, targets, etc.)."""
    return pd.read_csv(path, sep="\t", low_memory=False)


def load_geneinfo(path: Path, landmark_only: bool = True) -> pd.DataFrame:
    """Load gene metadata. If landmark_only, filter to the 978 landmark genes."""
    df = pd.read_csv(path, sep="\t", low_memory=False)
    if landmark_only:
        df = df[df["pr_is_lm"] == 1].copy()
    return df


def find_multi_cellline_drugs(
    siginfo: pd.DataFrame,
    min_cell_lines: int = 5,
    min_signatures_per_cellline: int = 1,
) -> pd.DataFrame:
    """Identify drugs profiled across multiple cell lines.

    Returns a summary DataFrame with columns:
        pert_iname, n_cell_lines, cell_lines, n_total_signatures
    """
    drug_cells = (
        siginfo.groupby("pert_iname")
        .agg(
            n_cell_lines=("cell_id", "nunique"),
            cell_lines=("cell_id", lambda x: sorted(x.unique().tolist())),
            n_total_signatures=("sig_id", "count"),
        )
        .reset_index()
    )
    drug_cells = drug_cells[drug_cells["n_cell_lines"] >= min_cell_lines]
    drug_cells = drug_cells.sort_values("n_cell_lines", ascending=False)
    return drug_cells


def get_consensus_signatures(
    siginfo: pd.DataFrame,
    drug_name: str,
    gctx_path: Path | None = None,
    signatures_matrix: np.ndarray | None = None,
    sig_ids_index: list[str] | None = None,
    gene_ids: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Get one consensus signature per cell line for a drug.

    Uses the signature with highest distil_ss (signature strength) per cell line.
    If no GCTX file, returns sig_ids for manual loading.

    Args:
        siginfo: full siginfo DataFrame
        drug_name: pert_iname value
        gctx_path: path to GCTX file (optional)
        signatures_matrix: pre-loaded (n_sigs, n_genes) matrix (optional)
        sig_ids_index: sig_id list matching rows of signatures_matrix
        gene_ids: gene_id list matching columns

    Returns:
        {cell_line: (n_genes,) z-score vector}
    """
    drug_sigs = siginfo[siginfo["pert_iname"] == drug_name].copy()

    if "distil_ss" in drug_sigs.columns:
        best_per_cell = drug_sigs.loc[
            drug_sigs.groupby("cell_id")["distil_ss"].idxmax()
        ]
    else:
        best_per_cell = drug_sigs.groupby("cell_id").first().reset_index()

    if signatures_matrix is not None and sig_ids_index is not None:
        sig_id_to_row = {sid: i for i, sid in enumerate(sig_ids_index)}
        result = {}
        for _, row in best_per_cell.iterrows():
            sid = row["sig_id"]
            if sid in sig_id_to_row:
                result[row["cell_id"]] = signatures_matrix[sig_id_to_row[sid]]
        return result

    if gctx_path is not None:
        return _load_from_gctx(gctx_path, best_per_cell, gene_ids)

    logger.warning("No signature data source provided. Returning empty dict.")
    return {}


def _load_from_gctx(
    gctx_path: Path,
    sig_rows: pd.DataFrame,
    gene_ids: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Load specific signatures from a GCTX file using cmapPy."""
    try:
        from cmapPy.pandasGEXpress import parse
    except ImportError:
        raise ImportError("cmapPy required for GCTX loading: pip install cmapPy")

    cids = sig_rows["sig_id"].tolist()
    gctoo = parse.parse(str(gctx_path), cid=cids, rid=gene_ids)

    result = {}
    for _, row in sig_rows.iterrows():
        sid = row["sig_id"]
        if sid in gctoo.data_df.columns:
            result[row["cell_id"]] = gctoo.data_df[sid].values.astype(np.float64)

    return result


def get_multi_dose_signatures(
    siginfo: pd.DataFrame,
    drug_name: str,
    signatures_matrix: np.ndarray | None = None,
    sig_ids_index: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Get all signatures per cell line (multiple doses/times).

    Returns {cell_line: (n_sigs, n_genes) matrix}.
    """
    drug_sigs = siginfo[siginfo["pert_iname"] == drug_name]

    if signatures_matrix is None or sig_ids_index is None:
        return {}

    sig_id_to_row = {sid: i for i, sid in enumerate(sig_ids_index)}

    result = {}
    for cell_id, group in drug_sigs.groupby("cell_id"):
        rows = []
        for sid in group["sig_id"]:
            if sid in sig_id_to_row:
                rows.append(signatures_matrix[sig_id_to_row[sid]])
        if rows:
            result[cell_id] = np.array(rows)

    return result


def make_synthetic_data(
    n_drugs: int = 100,
    n_cell_lines: int = 8,
    n_genes: int = 978,
    frac_transporting: float = 0.3,
    rng: np.random.Generator | None = None,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict]]:
    """Generate synthetic drug signatures for pipeline testing.

    Creates two classes of drugs:
    - Transporting: same direction across cell lines (small perturbation)
    - Non-transporting: direction rotates across cell lines

    Returns:
        (drug_signatures, drug_metadata)
        drug_signatures: {drug_name: {cell_line: (n_genes,) vector}}
        drug_metadata: {drug_name: {"transports": bool, "moa": str}}
    """
    if rng is None:
        rng = np.random.default_rng()

    n_transport = int(n_drugs * frac_transporting)
    moa_classes = [
        "kinase inhibitor",
        "HDAC inhibitor",
        "proteasome inhibitor",
        "topoisomerase inhibitor",
        "mTOR inhibitor",
        "MEK inhibitor",
        "VEGFR inhibitor",
        "CDK inhibitor",
    ]
    cell_lines = [f"CELL_{i}" for i in range(n_cell_lines)]

    drug_signatures = {}
    drug_metadata = {}

    for i in range(n_drugs):
        drug_name = f"BRD-{'T' if i < n_transport else 'N'}{i:04d}"
        transports = i < n_transport
        moa = moa_classes[i % len(moa_classes)]

        base_direction = rng.standard_normal(n_genes)
        base_direction = base_direction / np.linalg.norm(base_direction)
        base_magnitude = rng.uniform(2, 10)

        sigs = {}
        for cl in cell_lines:
            if transports:
                noise = rng.standard_normal(n_genes) * 0.1
            else:
                noise = rng.standard_normal(n_genes) * 0.8
            sig = base_direction * base_magnitude + noise
            mag_scale = rng.uniform(0.8, 1.2) if transports else rng.uniform(0.3, 3.0)
            sigs[cl] = sig * mag_scale

        drug_signatures[drug_name] = sigs
        drug_metadata[drug_name] = {
            "transports": transports,
            "moa": moa,
            "pert_iname": drug_name,
        }

    return drug_signatures, drug_metadata


def save_results(results: list[dict], path: Path) -> None:
    """Save per-drug results to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    for r in results:
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)):
                r[k] = float(v) if isinstance(v, np.floating) else int(v)
            elif isinstance(v, np.ndarray):
                r[k] = v.tolist()

    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Saved {len(results)} drug results to {path}")
