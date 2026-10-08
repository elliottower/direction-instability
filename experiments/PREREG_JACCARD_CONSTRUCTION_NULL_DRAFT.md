# Which of the four held-out outcomes survive a null that preserves the construction?

**Date:** 2026-10-08
**Status:** DRAFT. Not frozen. No synthetic Jaccard, Pearson or L1 statistic has
been computed under any construction.
**Commit SHA:** pending
**Kind:** post-hoc diagnostic of a published analysis. It computes no new
registered criterion for the metric itself and cannot change any verdict in
`PREREGISTRATION_VALIDITY.md`, whose Test 4 it extends to the three outcomes that
test omitted.
**Relates to:** `PREREGISTRATION_VALIDITY.md` Test 1 and Test 4;
`experiments/06_geometric_validity.py`, whose `run_cross_metric_holdout` computes
the four outcomes and whose `run_synthetic_null` scores only the first.

## The question

Test 4 generates synthetic drugs whose signature norms and cell-line counts match
the real cohort, carries no biology, and asks what held-out AUROC the geometry
alone produces. It returned 0.996 against the real 0.986, and the published text
reads that as establishing the cosine-based prediction to be a geometric
tautology. That reading is correct.

Test 4 scores one outcome: the held-out cosine. Test 1 computes three more from
the same signatures under the same leave-one-out split with the same predictor,
swapping only the final similarity function, and reports a Jaccard AUROC of
0.957. The published text groups that Jaccard result with two external channels
and describes all three as validating the metric "through channels that share no
cosine geometry with the metric definition."

Sharing no cosine geometry and sharing no construction are different properties.
Top-fifty-gene overlap between a held-out signature and the consensus of its
siblings is not a cosine, and it still rises with angular agreement among the
same signatures. Whether it rises enough to reproduce 0.957 with no biology
present has never been computed.

## Foreknowledge

The real values are published and were read before this plan was written: held-out
cosine AUROC 0.986, Jaccard AUROC 0.957, Spearman rho between direction
instability and mean top-gene Jaccard -0.79, synthetic cosine AUROC 0.996 +/- 0.001
over five repeats, CCLE target expression breadth rho -0.170, PRISM viability
concordance rho 0.261 at p 0.019, MOA classification 6.0% against an 8.9%
baseline. The 0.60 pass threshold below is taken from Test 4 of
`PREREGISTRATION_VALIDITY.md`, where it was fixed before any synthetic statistic
was computed, and is not chosen here.

No synthetic Pearson, L1 or Jaccard statistic exists on disk or has been
computed in any form. The generator is the one already written; this plan changes
it in one respect, stated in full below.

## Hypotheses

**H1.** The synthetic Jaccard AUROC reaches the neighborhood of the real 0.957,
which would place the Jaccard channel inside the construction rather than outside
it.

**H2.** The synthetic Pearson AUROC and the synthetic L1 association likewise
reach their real values, since both are computed from the same signatures under
the same split.

**H3.** Scrambling which synthetic drug's direction instability is paired with
which synthetic outcome returns an AUROC indistinguishable from 0.5.

**H4.** Matching the real within-drug norm dispersion in addition to the mean
norm, which the current generator computes and discards, does not move the
synthetic cosine AUROC by more than 0.01 from the published 0.996.

**H3 carries the design.** Without it, a high synthetic AUROC on every outcome is
equally consistent with a pipeline that returns a high AUROC whatever it is given.
H3 is the condition under which this diagnostic can return a null at all.

**H4 is the outcome that strengthens the published analysis.** The generator's
unused norm dispersion is a gap in the geometry match, and if closing it leaves
the synthetic cosine AUROC where it was, the published 0.996 stands on a match
that is tighter than the one it was computed under.

## Inference criteria

| hypothesis | holds when |
|---|---|
| H1 | the mean synthetic Jaccard AUROC over five repeats is at least 0.90 |
| H1 refuted | the mean synthetic Jaccard AUROC is below 0.60, the threshold Test 4 fixed for its own outcome |
| H2 | the mean synthetic Pearson AUROC is at least 0.90, and the synthetic L1 Spearman rho reaches at least 0.70 of the real value in absolute terms with the same sign |
| H3 | the scrambled AUROC falls within 0.5 plus or minus 0.05 on every repeat |
| H4 | the mean synthetic cosine AUROC under the dispersion-matched generator differs from 0.996 by at most 0.01 |

**All hypotheses are void if H3 fails.** A pipeline that cannot return 0.5 when
the pairing is destroyed establishes nothing about any outcome.

A synthetic Jaccard AUROC between 0.60 and 0.90 is reported with its interval and
the gap to the real value, and yields no categorical verdict. The published
Jaccard result is then neither inside nor outside the construction at the
resolution of this diagnostic, and that is what the record says.

## Sample size

The eligible cohort is every drug with at least ten cell lines and a
magnitude-corrected direction instability, 2,694 drugs under the published
filter. Five repeats, as Test 4 used. This resolves whether a synthetic AUROC
sits near 0.95 or near 0.55; it does not resolve differences of 0.02, and no
criterion above depends on one.

## Procedure

The generator, the eligibility filter, the magnitude correction and the two-stage
outcome thresholds are the published ones, reused by import rather than restated,
so this diagnostic cannot silently diverge from the analysis it examines.

One change: synthetic signatures are drawn to match each real drug's mean
signature norm and its within-drug norm standard deviation, rather than the mean
alone. The current generator computes the standard deviation at
`06_geometric_validity.py:339` and does not use it. Both generators are run, which
is what makes H4 answerable.

For each synthetic drug, all four held-out outcomes are computed by the same
code path that computes them for real drugs: held-out cosine, Pearson correlation
on raw z-scores, L1 distance, and the mean of top-fifty-up and top-fifty-down
gene Jaccard. Each outcome is reduced to a per-drug summary and an AUROC against
negative direction instability using the published thresholds.

The scramble for H3 permutes the direction instability vector against the fixed
outcome vector within each repeat.

Seeds are drawn from one `SeedSequence` whose entropy is fixed here:
20261008003. The per-repeat child seeds and the scramble seeds are recorded in
the output file beside the statistics they produced.

## What is written

One file, `results/07_jaccard_construction_null/construction_null.json`,
carrying every per-repeat AUROC for every outcome under both generators, the
scramble results, the input hashes, the seed entropy and child seeds, and the
published real values the comparison is against. The per-drug synthetic
statistics are written as an append-only shard per repeat so a killed run resumes
rather than restarts.

## Maximum claim under this registration

If H1 holds and H3 holds, the paper may say that the gene-level Jaccard channel
reproduces its reported AUROC under a construction-matched null carrying no
biology, and therefore does not provide validation independent of the metric's
construction; and that the metric's construction-independent support rests on the
two external channels, CCLE target expression breadth and PRISM viability
concordance. It may not say that the Jaccard association is zero, that the metric
is invalid, or that the published cosine result was wrong, none of which this
computes. If H1 is refuted the paper may say the Jaccard channel survives a
construction-matched null that had not previously been applied to it. Nothing
here licenses a claim about the 8,949-drug global correlation of -0.79, which is
not a held-out prediction and is not tested.
