# Which held-out outcomes survive a null that regroups the signatures, and which are settled by algebra?

**Date:** 2026-10-08
**Status:** DRAFT v2. Not frozen. Revises
`PREREG_JACCARD_CONSTRUCTION_NULL_DRAFT.md`, which is kept unaltered. No
synthetic Jaccard, Pearson or L1 statistic has been computed under any
construction.
**Commit SHA:** pending
**Kind:** post-hoc diagnostic of a reported analysis. It computes no new
registered criterion for the metric and cannot change any verdict in
`PREREGISTRATION_VALIDITY.md`.
**Relates to:** `PREREGISTRATION_VALIDITY.md` Tests 1 and 4;
`experiments/06_geometric_validity.py`;
`experiments/03_core_defenses.py:496` (`run_holdout_prediction`), whose
leave-one-out predictor the synthetic arm did not match.

## The questions

Three, separated because the first is settled analytically and the other two are
not.

**The algebra.** For unit vectors `x_1 ... x_n` with mean `xbar`,

    D = 1 - mean_{i<j} <x_i, x_j> = n / (n - 1) * (1 - ||xbar||^2)

exactly. Expanding `||sum_i x_i||^2 = n + 2 sum_{i<j} <x_i, x_j>` gives it in one
step, and it is verified numerically below. Direction instability is therefore a
strictly decreasing function of the squared length of the mean unit vector, which
is the concentration of the direction set. The held-out cosine outcome is the
cosine between one unit vector and the mean of the remaining `n - 1`. Its
relationship to `D` is algebraic, not empirical, and the published synthetic
result is not what establishes it.

**The gene-rank channel.** Top-fifty-gene overlap between a held-out signature
and the consensus of its siblings is not a function of `||xbar||^2`. It depends on
coordinate rankings and on margins near the fiftieth position, so whether it
inherits the concentration strongly enough to reproduce its reported
discrimination is an empirical question with no algebraic answer.

**The predictor.** `run_holdout_prediction` recomputes `D` inside the
leave-one-out loop on training signatures only, so the reported 0.986 is a
genuine held-out prediction. `run_cross_metric_holdout` does not: at
`06_geometric_validity.py:112` it takes the predictor from the full signature set
through `di_corrected`, including the signature held out for the outcome. The
reported Pearson, L1 and Jaccard figures are therefore retrospective consistency
statistics. `run_synthetic_null` also uses a full-set predictor, so the published
0.996 is not built like the 0.986 it is compared against.

## What the null is, and is not

`run_synthetic_null` draws real measured signature rows and regroups them under
synthetic drug labels, selecting rows whose norms sit near each real drug's mean
norm. It therefore retains gene-gene covariance, common transcriptional
responses, cell-line, dose, time and batch structure, assay and preprocessing
artifacts, and may draw rows from the source drug itself. What it destroys is the
original grouping of signatures into drugs.

It is named here an **empirical signature-regrouping null with norm matching**. It
is not biology-free and it is not a full geometry match, and no hypothesis below
is worded as though it were. A high statistic under it licenses the statement
that the original drug groupings are not required to produce that statistic, and
nothing stronger.

## Foreknowledge

Published and read before this plan: held-out cosine AUROC 0.986, Jaccard AUROC
0.957, Spearman rho between direction instability and mean top-gene Jaccard
-0.79, synthetic cosine AUROC 0.996 +/- 0.001 over five repeats, CCLE target
expression breadth rho -0.170, PRISM viability concordance rho 0.261 at p 0.019,
MOA classification 6.0% against an 8.9% baseline.

The closed form above was derived and checked numerically before this plan was
written: over 400 random cohorts with `n` from 3 to 39 and dimension from 2 to
199, the maximum absolute difference between the pairwise definition and the
closed form was 2.22e-16. That check is a proposition about the definition and
involves no cohort data.

No synthetic Pearson, L1 or Jaccard statistic exists on disk or has been computed
in any form. The 0.60 threshold retained below was fixed in
`PREREGISTRATION_VALIDITY.md` Test 4 before any synthetic statistic existed; its
transfer to a different outcome is an operational choice made here, not an
inherited justification, and is labeled as such.

## The cohort and the estimands, fixed once

One ordered list of eligible drugs serves every arm: drugs with at least ten
signature rows that are present in both the per-drug results file and the
magnitude-corrected predictor. The list, its length and its sha256 are written
before any arm runs, and any arm that cannot score a drug on the list records why
rather than dropping it silently.

Two predictors are carried through every arm, never mixed:

| predictor | definition | what it audits |
|---|---|---|
| `D_full` | magnitude-corrected `D` over all of a drug's rows | the statistic the manuscript reports |
| `D_loo` | mean over `i` of `D` computed on the rows excluding `i` | the statistic `run_holdout_prediction` reports, and the one a held-out claim requires |

Outcomes and their reductions, fixed here:

| outcome | per-drug reduction | binary endpoint | continuous endpoint |
|---|---|---|---|
| cosine | fraction of held-out cosines strictly above 0.3 | that fraction strictly above 0.5 | mean held-out cosine |
| Pearson | mean held-out Pearson | mean strictly above 0.3 | mean held-out Pearson |
| Jaccard | mean of top-fifty-up and top-fifty-down Jaccard over held-out rows | mean strictly above 0.3 | the mean itself |
| L1 | mean held-out L1 distance | none; L1 has no binary endpoint and no AUROC is computed for it | the mean itself |

The continuous endpoint is co-primary with the binary one for every outcome, so a
cohort in which no drug crosses a binary threshold still yields a result.

Nonfinite and degenerate cases have one policy applied identically to every arm:
a drug whose consensus or held-out row has norm below 1e-12, whose signature is
constant, or whose reduction is nonfinite is excluded from that outcome, counted,
and listed by identifier in the output. Ties at the fiftieth gene position are
broken by ascending gene index, fixed here. Gene order is asserted identical
across all signatures before any outcome is computed.

**An AUROC with one outcome class present is undefined and is recorded as
undefined.** It is never recorded as 0.5 and never enters a mean. The existing
code returns 0.5 in that case at `06_geometric_validity.py:369`; that branch is
replaced, and a level whose binary endpoint is undefined reports its continuous
endpoint and no binary verdict.

## Hypotheses

**H1.** The regrouping null reproduces high Jaccard discrimination: the mean
synthetic Jaccard AUROC under `D_loo` is at least 0.90, and the synthetic
continuous Spearman reaches at least 0.70 of the observed value in absolute terms
with the same sign.

**H2.** It does not: the mean synthetic Jaccard AUROC is below 0.60 and the
synthetic continuous Spearman is below 0.70 of the observed value in absolute
terms.

**H3.** The published full-set predictor inflates the synthetic cosine result
relative to the held-out predictor: the synthetic cosine AUROC under `D_full`
exceeds the synthetic cosine AUROC under `D_loo`.

**H4.** Matching each real drug's within-drug norm standard deviation in addition
to its mean norm moves the synthetic cosine AUROC by at most 0.02, judged as a
paired difference between the two generators run on the same cohort with the same
per-repeat seeds.

**H5.** The scorer returns a permutation distribution centered on 0.5. For each
valid synthetic cohort, 999 permutations of the predictor against the fixed
outcome give a mean AUROC within 0.01 of 0.5, and the observed spread agrees with
`(n + 1) / (12 * n_pos * n_neg)` to within a factor of two.

**H5 carries the design.** Without it, a high statistic under every arm is
equally consistent with a scorer that returns a high statistic whatever it is
given. H5 is a software and calibration control; it does not validate the
regrouping generator, and a mis-specified generator can pass it.

**H3 and H4 are the outcomes that can favor the reported analysis.** If H4 holds,
the published cosine null stands under a tighter norm match than the one it was
computed under. H3 is reported whichever way it falls and tells the manuscript
which predictor its synthetic comparison should use.

## Inference criteria

| hypothesis | holds when |
|---|---|
| H1 | mean synthetic Jaccard AUROC under `D_loo` >= 0.90 and continuous ratio >= 0.70, same sign |
| H2 | mean synthetic Jaccard AUROC under `D_loo` < 0.60 and continuous ratio < 0.70 |
| H3 | mean synthetic cosine AUROC under `D_full` exceeds that under `D_loo`, with a paired bootstrap interval excluding zero |
| H4 | the paired generator difference in mean synthetic cosine AUROC has a bootstrap interval contained in (-0.02, 0.02) |
| H5 | permutation mean within 0.01 of 0.5 on every valid cohort, and the spread within a factor of two of the closed-form variance |

**All hypotheses are void if H5 fails.** **H1 and H2 are void if the binary
endpoint is undefined on more than one repeat**, in which case the continuous
endpoints are reported and no verdict on the Jaccard channel is issued.

Neither H1 nor H2 holding is a possible outcome: a mean AUROC between 0.60 and
0.90, or criteria that split, is reported with its interval and the gap to the
observed value, and yields no categorical verdict. The 0.90 and 0.60 bounds are
operational thresholds declared here, not estimates of comparability with 0.957,
and the wording of any resulting claim is bounded by the maximum-claim paragraph.

## Sample size

The eligible cohort is every drug meeting the filter above, 2,694 under the
published eligibility. Thirty regrouping repeats, not the five the published null
used, because a 0.02 paired difference is registered in H4 and five repeats do
not resolve it; thirty gives a standard error near 0.004 at the published
per-repeat spread of 0.001 to 0.01, which does. H5 uses 999 permutations per
valid cohort.

Reported separately, never as one interval: the spread across regroupings, the
Monte Carlo error of each mean, and the paired generator difference with its own
interval.

## Procedure

The eligibility filter, the magnitude correction, the outcome reductions and the
two predictors are implemented once and imported by every arm, so no arm can
diverge from another in a way the code does not make visible. The achieved norm
matching error per drug, per generator, is recorded.

Both generators run: the published nearest-mean-norm selection, and a variant
that additionally matches within-drug norm standard deviation. The published
generator is run first and its mean synthetic cosine AUROC under `D_full` is
compared to the published 0.996 as a reproduction check, reported whether or not
it agrees.

Seeds come from one `SeedSequence` with entropy 20261008003, spawned
deterministically by (generator, repeat, drug index, permutation index) over the
frozen drug order, so a resumed run reproduces an uninterrupted one. Every child
seed is written beside the statistic it produced.

Checkpointing: one append-only shard per (generator, repeat) holding the per-drug
synthetic statistics, written as the repeat completes and also every 500 drugs
within it, with completed shards skipped on restart.

## What is written

`results/07_jaccard_construction_null/construction_null.json` with every
per-repeat AUROC and continuous statistic for every outcome, predictor and
generator, the permutation calibration, the undefined-endpoint counts and
identifiers, the achieved norm-matching errors, the reproduction check against
0.996, the input hashes, the seed entropy and all child seeds, and the published
values the comparison is against. Per-drug statistics go to the shards.

## Maximum claim under this registration

If H1 holds and H5 holds, the paper may say that high gene-overlap discrimination
is also obtained after the original grouping of signatures into drugs is
destroyed by the specified regrouping procedure, and that the reported Jaccard
figure therefore does not by itself establish validation independent of
within-signature-set consistency. If H2 holds and H5 holds, the paper may say
that the specified regrouping procedure did not reproduce the observed gene-overlap
discrimination, which is a limited robustness result and does not establish
independence from the shared construction or identify a biological mechanism.

In either case the paper may say that the cosine channel's relationship to the
metric is algebraic, on the closed form rather than on any null, and that the
Pearson, L1 and Jaccard figures as published use a predictor that includes the
held-out row and are retrospective consistency statistics rather than held-out
predictions.

It may not say that the metric is invalid, that the Jaccard association is zero,
that biology causes any difference between a real and a synthetic statistic, or
that the external CCLE and PRISM channels are thereby established. External
provenance removes the shared construction and does not remove confounding,
selection or measurement limits, which this computes nothing about. Nothing here
licenses a claim about the 8,949-drug global correlation of -0.79, which is not a
held-out statistic and is not tested.
