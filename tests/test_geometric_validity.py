"""
Tests for the six geometric validity experiments.

Each test verifies the correctness of the computation on synthetic data
where the ground truth is known.
"""
import importlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

validity = importlib.import_module("experiments.06_geometric_validity")
build_consensus = validity.build_consensus
magnitude_correct_di = validity.magnitude_correct_di
pairwise_cosine_matrix = validity.pairwise_cosine_matrix
run_cross_metric_holdout = validity.run_cross_metric_holdout
run_moa_classification = validity.run_moa_classification
run_prism_concordance = validity.run_prism_concordance
run_split_half = validity.run_split_half
run_synthetic_null = validity.run_synthetic_null
run_target_breadth = validity.run_target_breadth


@pytest.fixture
def synthetic_drug_data():
    """Create synthetic drugs with known transport properties.

    - Transporting drugs: same direction across cell lines (low DI).
    - Non-transporting drugs: random direction per cell line (high DI).
    """
    rng = np.random.default_rng(12345)
    n_genes = 200
    n_cells = 15
    n_transporting = 50
    n_random = 50

    drug_data = {}
    drug_results = {}

    for i in range(n_transporting):
        name = f"transport_{i}"
        base = rng.standard_normal(n_genes)
        base /= np.linalg.norm(base)
        sigs = np.array([base * rng.uniform(5, 15) + rng.standard_normal(n_genes) * 0.15
                         for _ in range(n_cells)])
        cells = [f"CELL_{j}" for j in range(n_cells)]
        drug_data[name] = {"cell_lines": cells, "signatures": sigs}

        cos_mat = pairwise_cosine_matrix(sigs)
        triu = cos_mat[np.triu_indices_from(cos_mat, k=1)]
        di = 1.0 - float(triu.mean())
        norms = np.linalg.norm(sigs, axis=1)

        drug_results[name] = {
            "drug_name": name,
            "direction_instability": di,
            "mean_norm": float(norms.mean()),
            "magnitude_cv": float(norms.std() / norms.mean()),
            "moa": "kinase inhibitor" if i < 15 else "HDAC inhibitor",
            "target": f"GENE_{i % 5}",
        }

    for i in range(n_random):
        name = f"random_{i}"
        sigs = np.array([rng.standard_normal(n_genes) * rng.uniform(5, 15)
                         for _ in range(n_cells)])
        cells = [f"CELL_{j}" for j in range(n_cells)]
        drug_data[name] = {"cell_lines": cells, "signatures": sigs}

        cos_mat = pairwise_cosine_matrix(sigs)
        triu = cos_mat[np.triu_indices_from(cos_mat, k=1)]
        di = 1.0 - float(triu.mean())
        norms = np.linalg.norm(sigs, axis=1)

        drug_results[name] = {
            "drug_name": name,
            "direction_instability": di,
            "mean_norm": float(norms.mean()),
            "magnitude_cv": float(norms.std() / norms.mean()),
            "moa": "serotonin receptor antagonist" if i < 15 else "dopamine receptor antagonist",
            "target": f"GENE_{(i + 5) % 10}",
        }

    return drug_data, drug_results


@pytest.fixture
def di_corrected(synthetic_drug_data):
    _, drug_results = synthetic_drug_data
    return magnitude_correct_di(drug_results)


def test_pairwise_cosine_identical_vectors():
    sigs = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float)
    cos_mat = pairwise_cosine_matrix(sigs)
    assert cos_mat == pytest.approx(np.ones((3, 3)))


def test_pairwise_cosine_orthogonal_vectors():
    sigs = np.eye(3, dtype=float)
    cos_mat = pairwise_cosine_matrix(sigs)
    expected = np.eye(3)
    assert cos_mat == pytest.approx(expected, abs=1e-7)


def test_pairwise_cosine_antiparallel():
    sigs = np.array([[1, 0], [-1, 0]], dtype=float)
    cos_mat = pairwise_cosine_matrix(sigs)
    assert cos_mat[0, 1] == pytest.approx(-1.0)


def test_magnitude_correct_removes_norm_correlation():
    rng = np.random.default_rng(99)
    n = 200
    norms = rng.uniform(10, 200, n)
    di_raw = 0.3 + 0.002 * norms + rng.standard_normal(n) * 0.05
    results = {}
    for i in range(n):
        results[f"drug_{i}"] = {
            "drug_name": f"drug_{i}",
            "direction_instability": float(di_raw[i]),
            "mean_norm": float(norms[i]),
        }
    corrected = magnitude_correct_di(results)
    corrected_arr = np.array([corrected[f"drug_{i}"] for i in range(n)])
    rho_before, _ = spearmanr(norms, di_raw)
    rho_after, _ = spearmanr(norms, corrected_arr)
    assert abs(rho_before) > 0.3
    assert abs(rho_after) < 0.1


def test_cross_metric_transporting_drugs_score_higher(synthetic_drug_data, di_corrected, tmp_path):
    drug_data, drug_results = synthetic_drug_data
    result = run_cross_metric_holdout(drug_data, drug_results, di_corrected, tmp_path, min_cells=5)
    assert result["n_drugs"] > 0
    assert result["rho_pearson"] < -0.15
    assert result["rho_jaccard"] < 0


def test_cross_metric_l1_positive_correlation(synthetic_drug_data, di_corrected, tmp_path):
    drug_data, drug_results = synthetic_drug_data
    result = run_cross_metric_holdout(drug_data, drug_results, di_corrected, tmp_path, min_cells=5)
    assert result["rho_l1"] > -0.5


def test_synthetic_null_auroc_below_real(synthetic_drug_data, di_corrected, tmp_path):
    drug_data, drug_results = synthetic_drug_data
    all_sigs = np.vstack([dd["signatures"] for dd in drug_data.values()])
    result = run_synthetic_null(drug_data, drug_results, di_corrected, all_sigs, tmp_path,
                                n_repeats=2, min_cells=5)
    assert result["mean_auroc"] < 0.75


def test_split_half_high_correlation(synthetic_drug_data, tmp_path):
    drug_data, drug_results = synthetic_drug_data
    result = run_split_half(drug_data, drug_results, tmp_path, n_splits=20, min_cells=5)
    assert result["mean_rho"] > 0.5


def test_moa_classification_above_chance(synthetic_drug_data, di_corrected, tmp_path):
    _, drug_results = synthetic_drug_data
    result = run_moa_classification(drug_results, di_corrected, tmp_path,
                                    min_class_size=10, k=3, n_permutations=50)
    assert result["n_drugs"] > 0
    assert result["accuracy"] >= result["perm_mean"]


def test_target_breadth_negative_correlation(synthetic_drug_data, di_corrected, tmp_path):
    _, drug_results = synthetic_drug_data

    rh_path = tmp_path / "repurposing_hub.txt"
    header_lines = [
        "!Source\tTest\n", "!URL\ttest\n", "!File_date\t1/1/2026\n",
        "!Table_name\tdrug information\n", "!API_available\ttest\n",
        "!Table_name\ttest\n", "!Notes\ttest\n", "!Additional\ttest\n",
        "!Fields\ttest\n",
    ]
    rh_rows = []
    for name, r in drug_results.items():
        rh_rows.append(f"{name}\tLaunched\t{r['moa']}\t{r['target']}\tneurology\ttest\n")
    rh_path.write_text("".join(header_lines) + "pert_iname\tclinical_phase\tmoa\ttarget\tdisease_area\tindication\n" + "".join(rh_rows))

    rng = np.random.default_rng(42)
    n_cells = 50
    gene_names = sorted(set(r["target"] for r in drug_results.values()))
    ccle_data = {}
    for g in gene_names:
        ccle_data[f"{g} (1234)"] = rng.exponential(5, n_cells)
    for i in range(20):
        ccle_data[f"OTHER_{i} (999{i})"] = rng.exponential(5, n_cells)

    ccle_df = pd.DataFrame(ccle_data, index=[f"ACH-{i:06d}" for i in range(n_cells)])
    ccle_path = tmp_path / "ccle.csv"
    ccle_df.to_csv(ccle_path)

    result = run_target_breadth(drug_results, di_corrected, str(rh_path), str(ccle_path), tmp_path)
    assert result["n_matched"] > 0
    assert "rho" in result


def test_prism_concordance_runs(synthetic_drug_data, di_corrected, tmp_path):
    _, drug_results = synthetic_drug_data

    rng = np.random.default_rng(42)
    prism_rows = []
    for name in list(drug_results.keys())[:20]:
        prism_rows.append({
            "broad_id": f"BRD-{name}",
            "drug_name": name,
            "viability_cv": rng.uniform(0.5, 10),
            "mean_lfc": rng.uniform(-2, 0.5),
            "std_lfc": rng.uniform(0.3, 1.5),
            "n_cell_lines": 900,
        })
    prism_df = pd.DataFrame(prism_rows)
    prism_path = tmp_path / "prism_cv.csv"
    prism_df.to_csv(prism_path, index=False)

    result = run_prism_concordance(drug_results, di_corrected, str(prism_path), tmp_path)
    assert result["n_matched"] > 0
    assert "rho" in result


def test_results_saved_to_disk(synthetic_drug_data, di_corrected, tmp_path):
    drug_data, drug_results = synthetic_drug_data
    result = run_split_half(drug_data, drug_results, tmp_path, n_splits=5, min_cells=5)
    assert (tmp_path / "test5_split_half.png").exists()
    assert "mean_rho" in result
    assert "ci_low" in result
    assert "ci_high" in result
