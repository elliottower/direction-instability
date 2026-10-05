# Provenance: the LINCS extraction carries a mislabeled gene axis

**Found:** 2026-10-05, by the registered reconstruction gate of
`direction-instability-drug-validity`, which compares artifacts against an
independent read of the pinned GEO GCTX.

## The defect

Three sites in this repository parse the GCTX with a requested row order and then
label the values with the request rather than with the file:

| site | |
|---|---|
| `experiments/modal_extract_lincs.py:125-134` | `parse.parse(..., rid=landmark_genes)`, then `gctoo.data_df.values.T` with `gene_ids = landmark_genes` |
| `experiments/modal_extract_shrna.py:78-93` | the same three lines |
| `data/lincs_loader.py:183-190` | `gctoo.data_df[sid].values`, labeled by the caller's `gene_ids` |

cmapPy **ignores the `rid` order it is given and returns the file's own row order**.
Measured on the pinned GCTX: asked for the landmark rows in reversed order, the
parser returned them in source order
(`direction-instability-drug-validity/results/03c_h3_sensitivity/reader_agreement.json`,
`parser_returned_the_requested_order: false`). Nothing in this repository calls
`reindex`, so every `.npz` these scripts have ever written carries the same
permutation: correct values, wrong column labels.

`direction-instability-drug-validity/experiments/modal_h3_rebuild.py:218` shows the
fix, one line, with the comment *"rid= selects, it does not order"*.

## What is affected, and what is not

A permutation applied to a whole matrix leaves every inner product, norm, cosine
and pairwise distance unchanged, and column-wise statistics travel with their
columns. So:

**Unaffected.** Direction instability, cross-cell-line transport, invariance
structure, the replicate noise floor, holdout prediction, every cosine and every
correlation between them. Core *sizes*, and H15's rank correlation between
instability and core size. These are computed from the extraction alone and are
invariant to the permutation.

**Affected.** Every gene *identity* obtained by indexing the declared labels:

- `experiments/03_core_defenses.py:325` — `core_gene_names = [gene_ids[i] for i in np.where(core_mask)[0]]`
- `experiments/03_comprehensive_analysis.py:443` — `gene_consistency[gene_ids[gi]]`
- H16, the claim that core genes are enriched for drug target pathways, which rests
  on those identities.

Drug and target names are unaffected: they come from the annotation files, not from
the gene axis.

## The pan-HDAC core, relabeled

`results/03_core_defenses/all_results.json`, `core_genes.hdac_shared_core`, holds 26
Entrez ids with a count of 8. Relabeled to the axis recovered from the pinned
source, **none of the 26 identities is unchanged**.

| printed | actually |
|---|---|
| BIRC5 | PLK1 |
| CDK6 | DYRK3 |
| MYC | BIRC5 |
| ORC1 | PCBD1 |
| SUV39H1 | CCNB2 |
| USP14 | SUV39H1 |

The corrected 26: PLK1, CCNA2, CCNB2, SMC4, SMARCC1, USP22, SUV39H1, BIRC5, LIG1,
FOXO4, ATF1, ZFP36, KEAP1, STUB1, DYRK3, NCOA3, MRPL12, PCBD1, MRPS16, GLOD4,
PCMT1, CNDP2, IFRD2, GABPB1, MLEC, TIMM9.

The full mapping is at
`direction-instability-drug-validity/results/03c_h3_sensitivity/core_gene_relabeling.json`,
bound to run `core-gene-relabeling-2026-10-05`.

MYC, CDK6 and ORC1 are not in the corrected core. SUV39H1 and BIRC5 are, at
positions other than the ones they were printed for.

## Consequence for the paper

`paper/drug_transport_bioinformatics_v4.tex` through `v7.tex` and
`paper/drug_transport_plos_v1.tex` all name the five genes above as the shared core
of the 8 pan-HDAC inhibitors, and read a chromatin/cell-cycle program from them.

The reading survives and is better supported by the corrected list: PLK1, CCNA2,
CCNB2 and SMC4 are cell cycle, SMARCC1 is SWI/SNF chromatin remodeling, USP22 is a
histone deubiquitinase, SUV39H1 is a histone methyltransferase. The exemplars are
what change.

Why this was not visible: the 978 LINCS landmarks are functionally central genes,
so a permutation maps one plausible chromatin story onto another, and the printed
list read as confirmation.

## What to do

1. Fix the three extraction sites with a `reindex` onto the requested order, and
   assert the order afterwards.
2. Do not regenerate the `.npz` in place. Write a corrected derivative with
   truthful `gene_ids`, recording the parent hash, the source GCTX hash, the
   permutation and its hash, and an all-cohort comparison against the source.
3. Replace the gene identities in the paper from the corrected mapping, as a new
   version rather than an edit in place.
4. Re-run H16 on the corrected identities. H15 and every geometric result stand.
