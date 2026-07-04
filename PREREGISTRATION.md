# Pre-registration: Cross-Cell-Line Drug Mechanism Transport

## Frozen before seeing real LINCS results

This document and the scoring code were committed before running the
analysis on real LINCS L1000 data. The commit SHA proves that the
scorer, hypotheses, and validation labels were not modified to fit
observed outcomes.

## Hypotheses

### H1: Direction instability predicts cross-cell-line transport failure
Drugs with high direction instability (the bracket norm analogue —
measuring how much the perturbation signature rotates across cell
lines) will have less reproducible effects across contexts.

### H2: Mechanism class stratifies transport
Abundance-modulating drugs (degraders, expression modulators) will
show lower direction instability than receptor blockers, allosteric
modulators, and pleiotropic compounds. This mirrors the MR
abundance/activity boundary (Tower, 2026).

### H3: Top-gene consistency tracks direction stability
Gene-level Jaccard overlap of top up/down-regulated genes correlates
with direction stability, confirming that direction change reflects
biological mechanism change, not just noise.

### H4: Magnitude instability is dissociable from direction instability
Some drugs will have stable direction but unstable magnitude (dose-
sensitive mechanisms that still transport qualitatively). These are
valid mechanisms with context-dependent penetrance.

## Frozen scorer

The scoring functions are in `geometry/drug_transport.py`. The metrics:

1. **direction_instability** = 1 - mean(pairwise cosine of unit signatures)
   Range [0, 2]. Low = transports. High = context-dependent.

2. **frechet_variance** = mean squared geodesic deviation from Frechet
   mean on the unit sphere. Zero = identical directions.

3. **magnitude_cv** = coefficient of variation of signature L2 norms.

4. **mean_top_gene_jaccard** = mean Jaccard overlap of top-50
   up/down-regulated genes across cell line pairs.

5. **mean_grassmannian_distance** = geodesic distance between per-cell-
   line response subspaces on Gr(k, d). Only computed when multiple
   signatures per cell line exist (dose/time variation).

## Validation labels (external, not derived from LINCS)

### Source 1: Broad Repurposing Hub MOA annotations
Mechanism of action labels from the Drug Repurposing Hub
(https://repo-hub.broadinstitute.org/repurposing). Downloaded and
frozen before analysis. Used for H2.

### Source 2: Known cell-line-specific vs. general drugs
Curated from literature. Examples:
- Vemurafenib: BRAF V600E-specific, should NOT transport to non-BRAF lines
- Trichostatin-a: HDAC inhibitor, broad mechanism, should transport
- Sirolimus: mTOR inhibitor, relatively general, should transport
- Imatinib: BCR-ABL-specific in CML context, context-dependent

### Source 3: Touchstone compound annotations
LINCS "is_touchstone" flag marks well-characterized reference compounds.
These should have higher-quality signatures and serve as a positive
control for measurement quality.

## Analysis plan

1. Compute all five metrics for every drug with 5+ cell lines
2. Rank drugs by direction instability
3. Stratify by MOA class (from Repurposing Hub)
4. Test H1: direction instability correlates with cross-cell-line
   Spearman correlation of gene-level effects
5. Test H2: Mann-Whitney U comparing MOA classes
6. Test H3: Spearman correlation between direction_instability and
   mean_top_gene_jaccard
7. Test H4: 2x2 table of high/low direction vs magnitude instability
8. Report all results including nulls

## Null model

Permutation null: for each drug, randomly shuffle cell-line labels
and recompute direction instability. The observed value should exceed
the 95th percentile of the null distribution for drugs we claim have
genuinely context-dependent mechanisms.

## What counts as success

- H1 confirmed with Spearman |rho| > 0.3
- H2 confirmed with p < 0.01 for at least one MOA contrast
- H3 confirmed with Spearman |rho| > 0.5
- A null result on any hypothesis is reported as-is

## What counts as failure

- Direction instability does not separate known general vs. specific drugs
- Permutation null produces comparable instability scores
- MOA stratification shows no pattern
- All signal is driven by signature quality (distil_ss) rather than biology
