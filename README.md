# Direction Instability Predicts Cross-Cell-Line Drug Mechanism Transport

Does a drug's mechanism of action transport from one cell type to another? We introduce *direction instability*, a geometric metric that quantifies how much a drug's perturbation signature rotates (vs. merely rescales) across cell lines.

## Core question

Standard drug development assumes a mechanism found in one cell line will hold in another. Can geometry predict *which* mechanisms transport before you run the experiment?

## Approach

1. For each drug, compute the perturbation signature (gene expression change) in each cell line
2. Compute direction instability: the mean pairwise cosine distance among unit-normalized signatures
3. Test whether instability predicts cross-cell-line mechanism conservation
4. Validate with genetic triangulation (shRNA knockdown signatures) and leave-one-out cross-validation

## Data

- [LINCS L1000](https://clue.io/) --- ~1.3M perturbation profiles, 978 landmark genes, ~70 cell lines
- Level 5 (replicate-collapsed z-scores) is the primary analysis level
- Freely available via clue.io API or GEO (GSE92742, GSE70138)

## Structure

- `geometry/` --- Direction instability computation, subspace extraction, distance metrics
- `experiments/` --- Analysis scripts (cross-cell-line transport, invariance, core defenses, replicate noise)
- `data/` --- LINCS metadata loaders and frozen pre-registered labels
- `paper/` --- Manuscript and figures
- `results/` --- Per-experiment output (JSON, PNG)

## License

MIT
