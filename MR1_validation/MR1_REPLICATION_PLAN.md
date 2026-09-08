# MR1 targeted replication plan

## Purpose

This analysis asks whether the key findings from the frozen, corrected MIDUS 3
(M3) analysis reproduce in the independent MIDUS Refresher (MR1) neuroimaging
sample. It is a targeted replication, not a second comprehensive analysis of
all possible MR1 associations.

The corrected M3 analysis frozen at tag `m3-corrected-stc-v1` defines the
replication targets and their expected directions. MR1 statistical significance
must not determine which effects are designated as replication targets.

Some provisional MR1 sensitivity results were viewed before this plan was
finalized, including findings involving the right amygdala. Accordingly, all
sensitivity analyses are explicitly secondary/exploratory. They will be run and
reported as complete, predefined families rather than selected according to
their MR1 results.

**Completion status (September 8, 2026):** the protected-data preprocessing pipeline,
four direct analyses, and all predefined sensitivity families are complete. None of the
four direct tests reached the prespecified significance threshold. Under the frozen decision
rule, no MR1 motion-adjusted follow-up is required. The MR1 synthetic suite passes all 118 tests.

## Analysis sample and models

- The primary sample is the conservative MR1 fMRI sample: all three
  EmotionRegulation runs pass historical visual QC, mean framewise displacement
  across available task runs is below 0.5 mm, left-amygdala negative persistence
  has all six cross-run pairs (n_pairs == 6), and all three EmotionRegulation
  runs are present with exactly 231 volumes.
- Analyses involving diary affect additionally require the applicable diary
  outcome and its prespecified diary covariates.
- Adjusted OLS is the primary MR1 replication model. The available MR1 P5
  analytic inputs do not contain a usable family, household, sibling, twin-pair,
  or other grouping identifier; participant-level OLS is therefore used. This
  does not assert that the underlying MR1 cohort contains no related
  participants.
- Pearson correlation is descriptive and secondary.
- Full-sample analyses that retain participants failing conservative fMRI QC
  may be generated for transparency but will not support scientific
  conclusions.
- Directional tests retain the corrected M3 expected direction. Two-tailed
  tests are used where no direction was specified in M3.
- Raw negative affect is the principal negative-affect outcome. Its log
  transform is a robustness analysis, not an independent replication target.
- Exact estimates, uncertainty, sample sizes, and p-values are read from the
  generated result CSVs rather than copied into this plan.

## Direct replication targets

The following effects preserve the corrected M3 predictor, hemisphere,
contrast, outcome, expected direction, and covariate structure as closely as
the MR1 data allow.

| Target | Predictor | Outcome | Expected direction |
|---|---|---|---|
| R1 | Left-amygdala negative cross-run persistence | Daily positive affect | Negative |
| R2 | Left-amygdala negative cross-run persistence | Daily negative affect | Positive |
| R3 | Age at the MR1 neuroscience visit | Left-amygdala negative cross-run persistence | Negative |
| R4 | Left-amygdala–anterior-vmPFC negative-versus-neutral FC | Daily negative affect | Negative |

The negative-affect log transform will accompany R2 and R4 as a robustness
analysis. Posterior-vmPFC FC is not substituted for the anterior-vmPFC R4
predictor when judging direct replication.

The corrected M3 FC-to-persistence analysis was null. Its MR1 analogue may be
reported as a specificity check, but it is not a positive finding to replicate
and another nonsignificant result will not be interpreted as proof of
equivalence.

## Predefined sensitivity and extension families

These analyses guard against an overly narrow conclusion while keeping the
scope tied to the corrected M3 analysis scripts.

### Persistence and affect (02_sensitivity.py)

- Right-amygdala negative persistence with diary positive and negative affect
  (two-tailed; hemisphere specificity).
- Left-amygdala negative persistence with neuroscience-visit PANAS positive and
  negative affect as convergent-validity outcomes (one-tailed).

### Age and persistence (03_sensitivity.py)

- Age with right-amygdala negative persistence (two-tailed).
- Age with positive persistence, left and right (two-tailed).

### Functional connectivity and affect (04_sensitivity.py)

- Right-amygdala FC (neg−neu) with diary affect (two-tailed; hemisphere
  specificity).
- Left-amygdala positive-versus-neutral FC with diary affect (two-tailed;
  condition specificity).
- Left-amygdala FC (neg−neu) with PANAS as convergent-validity outcomes
  (one-tailed).
- Left-amygdala negative-condition-only FC with diary affect (one-tailed;
  operationalisation robustness).

All three sensitivity scripts execute their complete predefined family without
significance gating. No section is conditioned on the corresponding main
analysis being significant. Nominal p-values are reported without multiplicity
adjustment, matching the corrected M3 sensitivity-analysis workflow.

**Motion-adjusted analyses**: Motion-adjusted models are deferred until after
the main analyses. They will be considered only if a main effect shows
meaningful signal, matching the corrected M3 follow-up workflow. No motion
analysis is required when there is no main signal whose robustness needs
assessment. This conditional robustness step does not redefine replication
targets or authorize selecting new predictor/outcome pairs based on MR1 results.

No additional analysis will be added solely because its provisional MR1 result
is statistically significant. If a new analysis becomes scientifically
necessary, it will be labeled post hoc and justified separately.

## Interpretation rules

Direct replication is judged from the conservative-sample adjusted OLS model.
The correlation and robustness analyses provide context but do not replace that
primary test.

- **Supported:** the MR1 primary estimate is in the corrected M3 direction and
  the prespecified directional test is statistically significant.
- **Directionally consistent but inconclusive:** the estimate is in the M3
  direction, but its uncertainty includes zero.
- **Not supported:** the estimate reverses direction or is materially
  inconsistent with the corrected M3 effect.

A significant right-amygdala sensitivity result with an unsupported left-sided
direct target is an exploratory cross-hemisphere extension, not a direct
replication. Evidence for hemispheric specificity requires a direct comparison
of the left and right coefficients; significance in one hemisphere and not the
other is insufficient.

All results within each predefined sensitivity family will remain visible.
Nominal p-values will be reported without multiplicity adjustment, matching the
corrected M3 sensitivity-analysis workflow. Sensitivity analyses are interpreted
as robustness, specificity, or convergent-validity checks rather than
independent confirmatory tests. Isolated nominal significance in a sensitivity
analysis will not override the direct-replication classification. No additional
analyses will be selected based on provisional significance.

## Required validation before analysis

Before regenerating MR1 results:

1. Verify from fMRIPrep provenance whether BIDS slice-timing correction was
   applied; do not infer this from comments in the SLURM wrapper.
2. Reconcile raw EmotionRegulation run counts with derivative BOLD and confound
   counts.
3. Restrict motion extraction to EmotionRegulation runs 1–3 and validate one
   record per participant and run.
4. Apply the corrected PANAS missing-code handling after confirming the MR1
   codebook definitions and scale range.
5. Add required-column, unique-key, and merge-cardinality checks to the QC and
   fMRI merge steps.
6. Correct participant-level fMRI availability flags using participant ID
   membership rather than group-file existence.
7. Archive provisional result tables, including stale MLM files, before the
   clean rerun.
8. Run synthetic tests and syntax checks in the MR1 environment.

## Deferred M3 handedness audit

If a future manuscript revision makes a claim about hemispheric differences between M3 and
MR1, first perform the focused handedness audit below. The current report does not make a
lateralization claim: the nominal MR1 right-amygdala findings are described only as exploratory
cross-hemisphere sensitivity results. The audit is therefore outside the current frozen
manuscript scope rather than an incomplete analysis required for the present conclusions.

The audit will verify that:

1. Corrected M3 normalized images have valid and mutually consistent qform and
   sform matrices.
2. The Harvard–Oxford left-amygdala mask has a negative MNI x centroid and the
   right-amygdala mask has a positive MNI x centroid.
3. The two masks visibly overlay the intended hemispheres on representative
   normalized M3 images.
4. Corrected M3 BOLD orientation agrees with the participant T1 image and the
   corresponding fMRIPrep registration report.
5. fMRIPrep reports and logs contain no unresolved orientation, affine, or
   handedness warnings.
6. The M3 axis-transposition/reorientation step preserved anatomical
   handedness; downstream `L` and `R` labels are checked against image geometry,
   not accepted from filenames alone.

Until this audit is complete, a significant MR1 right-amygdala sensitivity
result will be described as an exploratory cross-hemisphere finding. It will
not be interpreted as evidence of lateralized replication or as evidence that
the M3 hemispheres were swapped.

## Reporting boundary

The MR1 report will distinguish direct replication, robustness analyses,
sensitivity/extension analyses, and specificity checks. It will report null and
divergent results within the frozen scope, but it will not expand into a general
search across all MR1 imaging and behavioral variables.
