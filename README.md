# Bracket Norm Predicts Drug Mechanism Transport Across Cell Lines

Does a drug's mechanism of action transport from one cell type to another? LINCS L1000 provides ~1.3M gene expression profiles of cells treated with ~20,000 compounds across multiple cell lines --- interventional data, not observational.

We apply bracket norm (Tower, 2026) to drug perturbation signatures and test whether it predicts which drug mechanisms transport across cell lines vs which fail. Bracket norm captures how much a perturbation response changes *direction* (not just magnitude) across conditions. In neural data, it predicted optogenetic causal importance; here, we test whether it predicts cross-cell-line mechanism stability.

## Core question

Standard drug development assumes a mechanism found in one cell line (e.g., MCF7 breast cancer) will hold in another (e.g., A549 lung cancer). The 90% clinical trial failure rate suggests this assumption fails far more often than it succeeds. Can geometry predict *which* mechanisms transport before you run the trial?

## Approach

1. For each drug, compute the perturbation signature (gene expression change) in each cell line
2. Compute bracket norm across the population of perturbation responses within each cell line
3. Test whether bracket norm predicts cross-cell-line signature correlation (mechanism transport)
4. Validate against known clinical outcomes: drugs that worked in one cancer type and failed in another

## Data

- [LINCS L1000](https://clue.io/) --- ~1.3M perturbation profiles, 978 landmark genes, ~70 cell lines
- Level 5 (replicate-collapsed z-scores) is the primary analysis level
- Freely available via clue.io API or GEO (GSE92742, GSE70138)

## Geometry (from bracket-norm)

- `geometry/bracket_norm.py` --- Bracket norm computation on perturbation populations
- `geometry/subspace.py` --- PCA/LDA subspace extraction
- `geometry/distances.py` --- Grassmannian geodesic distance, principal angles, CKA

## Connection to Mechanistic Reference

This is a direct empirical test of the transport hierarchy (Tower, 2026, Mechanistic Reference). A drug perturbation signature that transports across cell lines satisfies Role-level or Subspace-level transport. One that doesn't is accumulating reference debt. Bracket norm is the geometric diagnostic.

## Status

Setting up data pipeline and adapting bracket norm code from neural data to gene expression.

## License

MIT
