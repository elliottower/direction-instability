# Pre-registration: Geometric Validity Tests

**Filed before running any validity analyses on real data.**

## Purpose

These six tests address the concern that the held-out AUROC (0.986) from the
confound paper may partly reflect cosine self-consistency rather than biological
structure. Each test breaks the shared cosine geometry at a different point.

## Scorer

Direction instability: `DI = 1 - mean_{i<j} cos(s_i / ||s_i||, s_j / ||s_j||)`

Magnitude correction: OLS regression of DI on mean signature L2 norm per drug,
retain residuals + intercept. All tests use magnitude-corrected DI unless stated
otherwise.

## Test 1: Cross-metric held-out prediction

**Hypothesis:** DI predicts held-out outcomes computed with non-cosine metrics.
If the AUROC is cosine-circular, replacing the outcome metric should destroy it.

**Procedure:** Repeat the LOO held-out analysis from Section 3 using three
alternative outcomes: (i) Pearson correlation between held-out signature and
N-1 consensus (raw z-scores, not unit-normalized), (ii) L1 distance between
held-out and consensus, (iii) gene-level Jaccard overlap of top 50 up + top 50
down regulated genes.

**Pass criterion:** Spearman rho between DI and each alternative outcome has
|rho| > 0.3, and the binarized AUROC exceeds 0.80.

## Test 2: Target expression breadth

**Hypothesis:** Drugs targeting broadly expressed genes have lower DI, because
their mechanism can engage in every cell type. This test uses zero perturbation
data.

**Procedure:** For drugs with single annotated targets in the Repurposing Hub
that appear in the CCLE expression panel, compute expression breadth = fraction
of cell lines where the target gene TPM > median TPM across all genes in that
cell line. Correlate with magnitude-corrected DI.

**Pass criterion:** Spearman rho < -0.10 with p < 0.05.

## Test 3: Drug sensitivity concordance (PRISM)

**Hypothesis:** DI from transcriptomics predicts variability in a functional
readout (cell killing) measured on an independent platform.

**Procedure:** For drugs present in both LINCS L1000 and PRISM Repurposing
(24Q2), compute the CV of viability (log-fold change) across cell lines.
Correlate LINCS DI with PRISM viability CV.

**Pass criterion:** Spearman rho > 0.10 with p < 0.05.

## Test 4: Synthetic null with matched geometry

**Hypothesis:** The held-out AUROC depends on within-drug biological coherence,
not cosine-space structure.

**Procedure:** For each real drug with K cell lines, construct a synthetic drug
by sampling K signatures uniformly at random from other drugs' signatures,
matching K and the distribution of L2 norms. Compute DI and run LOO prediction
on synthetic drugs. Repeat 5 times.

**Pass criterion:** Synthetic AUROC < 0.60 (vs 0.986 for real drugs).

## Test 5: Split-half subspace stability

**Hypothesis:** DI reflects gene-level biology distributed across the
transcriptome, not an artifact of any particular gene subset.

**Procedure:** Randomly split the 978 landmark genes into two disjoint halves
(489 each). Compute DI independently on each half. Measure Spearman rank
correlation between DI_half1 and DI_half2. Repeat 100 times.

**Pass criterion:** Mean rho > 0.70 with 95% CI lower bound > 0.60.

## Test 6: MOA classification

**Hypothesis:** DI encodes classifiable biological information about mechanism
conservation beyond cosine geometry.

**Procedure:** Train a k-NN classifier (k=5) on (DI, magnitude_CV) to predict
MOA family among drugs in families with >= 15 members. LOO cross-validation.
Compare to most-frequent-class baseline and 1,000 label-permutation shuffles.

**Pass criterion:** Classification accuracy > baseline + 2 * permutation SD.

## Data sources

- LINCS L1000: data/lincs_subset.npz (167,266 signatures, 978 genes)
- Drug annotations: data/repurposing_hub_drugs.txt (Repurposing Hub)
- PRISM: direction-instability-atlas/results/prism_cv.csv (pre-computed)
- CCLE: direction-instability-atlas/data/depmap/OmicsExpressionProteinCodingGenesTPMLogp1.csv
- Pre-computed DI: zenodo_v1/drug_instability_results.json (8,949 drugs)

## Implementation

Script: experiments/06_geometric_validity.py
Tests: tests/test_geometric_validity.py
Results: results/06_geometric_validity/
