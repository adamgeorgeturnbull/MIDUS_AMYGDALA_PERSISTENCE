# Manuscript Preparation Freeze

**Freeze date:** September 8, 2026  
**State:** Corrected M3 and targeted MR1 analyses are computationally complete and frozen for
manuscript preparation.

## Frozen analysis state

- The corrected M3 preprocessing starts from untouched raw BOLD images, uses the corrected
  1-based FSL slice order, and applies slice-timing correction exactly once.
- Corrected M3 GLM, persistence, beta-series connectivity, QC, primary analyses, predefined
  sensitivities, motion checks, exploratory moderation analyses, and supplementary tables are
  complete.
- The targeted MR1 reproduction, protected-data preprocessing, imaging merge, QC, four direct
  tests, and all predefined sensitivity families are complete.
- The final MR1 active outputs use correlations and participant-level OLS only. No `mlm.csv`
  belongs in an active MR1 result directory.
- The MR1 synthetic test suite passes: **118 passed** in the real Box project on September 8,
  2026.
- None of the four direct MR1 tests reached the prespecified significance threshold. Under the
  frozen decision rule, no MR1 motion-adjusted follow-up is required.
- M3 sensitivity/exploratory results and MR1 sensitivity results use nominal p values without
  BH/FDR adjustment and do not redefine the direct findings.

## Manuscript evidence state

- Corrected Methods and Results were drafted from the final M3 and MR1 aggregate outputs.
- An independent read-only audit checked more than 80 numerical claims against those outputs and
  found no numerical errors.
- Follow-up checks confirmed the M3 sample demographics, Analysis 01 adjusted N of 1,168, and
  65 residual degrees of freedom for the reported FC-to-log-NA OLS result.
- `MAP_manuscript.docx` is the authoritative Box manuscript and will remain untouched while a
  separate revised copy is prepared.

## Freeze boundary

No new predictors, outcomes, samples, model variants, sensitivity families, or result-driven
analyses should be added for the current manuscript. Any proposed departure requires a new,
dated analysis-plan amendment and must not be represented as part of this frozen analysis.

The following are manuscript-preparation tasks, not reasons to alter or rerun the frozen models:

- integrate the audited Methods and Results into a new Word copy;
- complete manual author review;
- revise the Abstract, Discussion, Table 1, figures, and captions for consistency;
- identify residualized scatterplots as descriptive and base inference on the specified models;
- if normalized/resampled voxel dimensions are retained or reintroduced, verify them against the
  analyzed images before submission;
- if a future revision makes a lateralization claim comparing M3 and MR1, first complete the
  conditional handedness audit specified in `MR1_validation/MR1_REPLICATION_PLAN.md`.

The current MR1 right-amygdala results are reported only as nominal exploratory sensitivity
findings, not as evidence of hemispheric specificity or lateralized replication; therefore the
conditional handedness audit is outside the present reporting scope.

