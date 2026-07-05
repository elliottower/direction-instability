# Pre-registration: Cross-Cell-Line Drug Mechanism Transport

## Experiment 01: Direction instability

**Registered:** 2026-07-04T19:15:00-04:00 (commit SHA `1dc20a2`)

## Frozen before seeing real LINCS results

This document and the scoring code were committed before running the
analysis on real LINCS L1000 data. The commit SHA proves that the
scorer, hypotheses, and validation labels were not modified to fit
observed outcomes.

## Hypotheses

### H1: Direction instability predicts cross-cell-line transport failure
Drugs with high direction instability (measuring how much the
perturbation signature rotates across cell lines) will have less
reproducible effects across contexts.

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

---

## Experiment 02: Invariance structure

**Registered:** 2026-07-04T20:37:00-04:00

### Motivation

Direction instability (Experiment 01) summarizes cross-context conservation
as a single number. The invariance structure analysis goes further: for each
drug, build a graph over cell lines where an edge means "mechanism is
indistinguishable" (cosine similarity above threshold). The graph's
connected components are the drug's equivalence classes under context change.
This operationalizes the idea of finding the invariance group G — the set
of context transformations under which the mechanism is preserved.

### Hypotheses

#### H5: Invariance graph density correlates with direction instability
Graph density (fraction of cell-line pairs above cosine threshold) should
anticorrelate with direction instability. This is a consistency check: both
metrics should identify the same drugs as transporting.

#### H6: Pan-HDAC inhibitors form single connected components
At cosine threshold 0.5, pan-isoform HDAC inhibitors (vorinostat,
panobinostat, trichostatin-a) should have 1-2 connected components, while
selective HDAC inhibitors (PCI-34051, tubastatin-a) should fragment into
n_cell_lines components (each cell line isolated).

#### H7: Component structure reveals tissue-specific mechanism subgroups
For drugs with intermediate instability, the connected components should
cluster cell lines by tissue type or lineage. A kinase inhibitor might
transport across all carcinoma lines but not to hematopoietic lines. The
equivalence classes should be biologically interpretable, not random.

#### H8: Invariance rank predicts MOA breadth
Drugs with higher invariance rank (larger largest-component fraction) should
target more broadly expressed genes/proteins. Receptor-mediated drugs should
have low invariance rank because receptor expression is tissue-specific.

### Metrics

1. **Graph density**: fraction of cell-line pairs above cosine threshold
2. **Number of connected components**: how many equivalence classes
3. **Largest component fraction**: what fraction of cell lines share one
   equivalence class
4. **Component-tissue enrichment**: whether components correspond to known
   tissue/lineage groupings (future analysis)

### Thresholds

We test four cosine thresholds: 0.3, 0.5, 0.7, 0.85. We expect 0.5 to be
the most informative (stringent enough to be meaningful, permissive enough
to find structure). Results at all four thresholds are reported.

### What counts as success

- H5 confirmed with Spearman |rho| > 0.5 between density and -instability
- H6 confirmed if >=3 pan-HDAC inhibitors have 1-2 components and >=3
  selective HDAC inhibitors have n_cell_lines components at threshold 0.5
- H7 exploratory — reported as-is
- H8 confirmed if receptor-mediated MOA classes have significantly lower
  mean largest-component fraction than mechanism-universal classes

---

## Experiment 03: Toxicity confound check

**Registered:** 2026-07-04T20:45:00-04:00

### Motivation

Top-transporting drugs might just be cytotoxic — killing cells the same way
everywhere produces a conserved signature, but that's not mechanism transport.
SPEC 04 calls this "violence is not control."

### Hypotheses

#### H9: Cytotoxicity gene overlap does not explain direction instability
Define a stress/apoptosis gene set (top genes shared across known cytotoxic
compounds). Direction instability should NOT correlate with signature overlap
with this gene set, after controlling for the transporting drugs that target
fundamental machinery. If it does, the metric is confounded.

#### H10: Removing stress genes preserves the MOA stratification
Recompute direction instability after removing the top stress/apoptosis genes
(e.g., top 50 genes most correlated with cell death across LINCS). The MOA
stratification pattern should survive.

### What counts as success
- H9: correlation between stress-gene-overlap and instability is |rho| < 0.2
- H10: HDAC pan/selective gradient persists after stress gene removal

---

## Experiment 04: Component-tissue enrichment

**Registered:** 2026-07-04T20:45:00-04:00

### Motivation

For drugs with intermediate invariance (not fully transporting, not fully
fragmented), the equivalence classes should correspond to tissue lineages.
A kinase inhibitor might transport across all carcinoma lines but not to
hematopoietic lines. This tests whether the invariance graph captures
biologically meaningful structure.

### Hypotheses

#### H11: Components are tissue-enriched above chance
For drugs with 2-5 components at threshold 0.5, connected components should
show significant enrichment for cell lines from the same primary_site
(Fisher's exact test per component vs rest). Expect at least 20% of
intermediate-invariance drugs to have at least one tissue-enriched component
(p < 0.05).

#### H12: Tissue-specific kinase inhibitors cluster relevant lineages
EGFR inhibitors should have a component enriched for lung/colorectal lines.
VEGFR inhibitors should cluster by vascular/endothelial-adjacent lines.
MEK inhibitors should separate BRAF-mutant from wild-type lines.

### What counts as success
- H11: >20% of 2-5 component drugs have tissue-enriched components
- H12: at least 2/3 of the kinase inhibitor predictions are confirmed

---

## Experiment 05: Cell-line similarity matrix

**Registered:** 2026-07-04T20:45:00-04:00

### Motivation

Across all drugs, some cell lines are frequently co-grouped in the same
equivalence class. This gives a drug-mechanism-derived "functional
similarity" between cell lines, independent of expression clustering.

### Hypotheses

#### H13: Drug-derived cell similarity correlates with tissue of origin
Cell lines from the same primary_site should co-occur in equivalence
classes more often than cross-tissue pairs. Expect Spearman |rho| > 0.3
between co-occurrence frequency and tissue-match indicator.

#### H14: Drug-derived similarity reveals non-obvious groupings
Some cross-tissue cell line pairs should have high co-occurrence that is
not predicted by expression similarity alone. These are "functionally
equivalent" cell lines from a drug-response perspective.

### What counts as success
- H13: significant tissue enrichment in co-occurrence matrix
- H14: exploratory — reported as-is

---

## Experiment 06: Gene program decomposition

**Registered:** 2026-07-04T20:45:00-04:00

### Motivation

For each drug, separate genes into a "transporting core" (consistently
perturbed across cell lines) and a "context-dependent periphery" (variable).
The core should correspond to the drug's primary mechanism; the periphery
should reflect tissue-specific secondary effects.

### Hypotheses

#### H15: Core gene set size correlates with direction stability
Drugs with low direction instability should have larger core gene sets
(more genes consistently perturbed). Expect Spearman |rho| > 0.4.

#### H16: Core genes are enriched for drug target pathways
For drugs with MOA annotation, the core gene set should be enriched for
genes in the target pathway (e.g., HDAC inhibitor core genes enriched for
chromatin/histone modification genes).

#### H17: Periphery genes are enriched for tissue-specific programs
The variable genes should enrich for tissue-specific pathways
(differentiation markers, tissue-specific transcription factors).

### What counts as success
- H15: |rho| > 0.4 between core size and -instability
- H16, H17: exploratory — reported as-is

---

## Experiment 07: Dose-time stability

**Registered:** 2026-07-04T20:45:00-04:00

### Motivation

LINCS profiles drugs at multiple doses (mostly 10µM and 5µM) and timepoints
(6h and 24h). Direction instability across dose-time within a single cell
line measures whether the mechanism is stable across the dose-time manifold.
A drug whose signature rotates from 6h to 24h has time-dependent mechanism
engagement.

### Hypotheses

#### H18: Cross-cell-line instability and dose-time instability are independent
A drug can transport across cell lines but be dose-sensitive (stable
direction, variable magnitude with dose). Expect weak correlation
(|rho| < 0.3) between cross-cell-line direction instability and within-
cell-line dose-time direction instability.

#### H19: 6h vs 24h signatures differ more than dose variation
Time is a stronger source of signature rotation than dose, because early
timepoints capture direct target effects while later timepoints capture
downstream cascades. Expect mean rotation angle (6h vs 24h) > mean rotation
angle (5µM vs 10µM).

### What counts as success
- H18: |rho| < 0.3
- H19: paired comparison significant at p < 0.01

---

## Experiment 08: Genetic vs drug perturbation comparison

**Registered:** 2026-07-04T20:45:00-04:00

### Motivation

LINCS has 155K shRNA signatures. For drugs with known gene targets, the
drug signature should resemble the genetic knockdown signature of that
target — if the drug's mechanism is truly target-mediated. Direction
instability of the drug should predict how well the drug-genetic match
holds across cell lines.

### Hypotheses

#### H20: Drug-genetic cosine correlates with drug direction stability
Transporting drugs (low instability) should have higher cosine similarity
to their target gene's shRNA signature, because their effect is dominated
by the on-target mechanism. Context-dependent drugs should diverge from
the genetic signature in cell lines where off-target effects dominate.

#### H21: The drug-genetic match is cell-line dependent
For drugs with intermediate instability, the drug-genetic cosine should be
high in cell lines within the drug's largest equivalence class and low in
cell lines outside it. The invariance structure predicts WHERE the drug
acts through its labeled target.

### What counts as success
- H20: Spearman |rho| > 0.3 between drug instability and drug-genetic cosine
- H21: within-component vs outside-component cosine difference significant
  at p < 0.01

---

## Experiment 09: Clinical phase prediction

**Registered:** 2026-07-04T20:45:00-04:00

### Motivation

The Repurposing Hub provides clinical phase for 1,793 drugs (893 launched,
572 preclinical, 111 Phase 2, 79 Phase 3, 78 Phase 1, 43 withdrawn).
Does direction instability predict clinical advancement?

### Hypotheses

#### H22: Launched drugs do NOT have lower direction instability overall
Most launched drugs are receptor-mediated (designed for specific tissues),
so they should have HIGH instability. The relationship between instability
and clinical success is NOT monotonic — it depends on whether the
indication requires broad or narrow mechanism.

#### H23: Within MOA class, lower instability predicts advancement
Controlling for MOA class, drugs with lower instability within their class
should be more advanced clinically. A pan-HDAC inhibitor with the most
conserved signature should be the one that reached market.

#### H24: Withdrawn drugs have anomalous instability profiles
Withdrawn drugs should differ from launched drugs in direction-magnitude
dissociation: high magnitude variation (dose-sensitivity or toxicity) even
when direction is stable, or unstable direction indicating off-target
effects discovered post-approval.

### What counts as success
- H22: no overall correlation between instability and phase (expected null)
- H23: within-class correlation significant for at least 3 MOA classes
- H24: exploratory — reported as-is

---

## Experiment 10: Held-out context prediction

**Registered:** 2026-07-05T10:42:40+0000

### Motivation

Experiments 01-09 show that direction instability *correlates with*
various measures of mechanism conservation. Correlation is not
prediction. This experiment tests whether direction instability computed
on N-1 cell lines *predicts* whether the held-out Nth cell line will
show a consistent drug response. Leave-one-out cross-validation ensures
no data leakage: the held-out cell line is never used to compute the
instability score.

### Hypotheses

#### H25: Direction instability predicts held-out signature consistency
For each drug with >=10 cell lines, hold out one cell line at a time.
Compute direction instability on the remaining N-1. Compute cosine
similarity between the held-out signature and the consensus (mean) of
the N-1. Average across all held-out folds. Drugs with low LOO
instability should have high mean held-out cosine.

#### H26: HDAC selectivity gradient survives in LOO prediction
Pan-HDAC inhibitors (vorinostat, panobinostat, trichostatin-a) should
have higher mean held-out cosine than selective HDAC inhibitors
(PCI-34051, tubastatin-a, entinostat). The selectivity gradient
observed in full-sample instability should persist in the predictive
setting.

#### H27: LOO instability discriminates "will-transport" vs "will-fail"
Binarize held-out outcomes: "consistent" if cosine > 0.3, "inconsistent"
otherwise. Use LOO instability as a classifier (threshold at median).
AUROC should exceed 0.65 for this binary prediction task.

### Metrics

1. **loo_instability**: direction instability computed on N-1 cell lines
   (averaged across all N held-out folds)
2. **mean_heldout_cosine**: mean cosine between held-out signature and
   consensus of remaining N-1
3. **auroc**: area under ROC curve for binary prediction

### What counts as success
- H25: Spearman |rho| > 0.3 between loo_instability and mean_heldout_cosine
- H26: pan-HDAC mean_heldout_cosine > selective-HDAC mean_heldout_cosine
- H27: AUROC > 0.65
