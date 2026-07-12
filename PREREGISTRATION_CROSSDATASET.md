# Pre-Registration: Cross-Dataset DI Replication

**Date:** 2026-07-06
**Author:** [Anonymous]
**Status:** Frozen before analysis

## Hypothesis

Direction instability (DI) computed on LINCS L1000 (978 landmark genes,
dose-collapsed signatures) correlates positively with DI computed on
Tahoe-100M (whole-transcriptome, different cell panel) for drugs measured
in both datasets.

## Primary test

Spearman rank correlation between LINCS DI and Tahoe-100M DI for all
overlapping drugs (matched by drug name, case-insensitive).

**Success criterion:** Spearman rho > 0.15, p < 0.05 (one-sided).

## Secondary tests

1. Pearson correlation (sensitive to magnitude, may differ from Spearman).
2. Concordance in the tails: among drugs in the bottom quartile of LINCS DI
   (most stable), what fraction are also below-median in Tahoe DI?
3. HDAC inhibitor subset: do HDAC inhibitors that are stable in LINCS
   also appear stable in Tahoe?

## Magnitude correction

Report both raw and magnitude-corrected DI for the cross-dataset comparison.
If the raw correlation is driven by magnitude confounding, the corrected
correlation should be weaker.

## What we will report regardless of outcome

- The Spearman and Pearson correlations with confidence intervals.
- A scatter plot of LINCS DI vs Tahoe DI.
- If the primary test fails (rho <= 0.15 or p >= 0.05), we will report
  this and interpret it as evidence that DI may be platform-specific.

## Scorer and data

- LINCS DI: `zenodo_v1/drug_instability_results.json` (frozen at SHA 1dc20a2)
- Tahoe DI: `direction-instability-atlas/results/tahoe_di.csv` (from atlas repo)
- Drug name matching: case-insensitive exact match, no fuzzy matching.
