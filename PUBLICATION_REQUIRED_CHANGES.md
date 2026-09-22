# M3 Publication Required Changes — Completion and Revision Record

This document tracks every factual manuscript and reporting change identified in the M3 code
review and corrected-STC reanalysis. For each item, the status is one of:

- **COMPLETED** — computational correction or verification finished; result is in the repository
- **DRAFTED** — corrected manuscript prose exists and has been checked against aggregate outputs,
  but has not yet completed Word integration and author review
- **PENDING** — manuscript/figure/reporting edit still required

**Current phase (September 8, 2026): manuscript preparation.** The corrected M3 analysis and
targeted MR1 replication are computationally frozen. The replacement Methods and Results in
`MANUSCRIPT_METHODS_RESULTS_DRAFT.md` passed an independent numerical audit against the final
aggregate outputs. Remaining unchecked items are document integration, figures/captions, and
manual author review rather than new statistical analyses.

Authoritative working manuscript: `MAP_manuscript.docx`

---

## 1. Behavioral Preprocessing and Sample Description

### 1.1 Diary date missing-value audit — COMPLETED

The official M3 Daily Diary codebook confirms `C2DIMON` uses **98** and `C2DIYEAR` uses **9998**
as numeric missing codes. `02_construct_daily_diary_affect.py` already recodes exactly those values
before constructing `StartMonth` and `StartYear`. No preprocessing change was required.

### 1.2 PANAS missing-value recode — COMPLETED

**Correction:** `06_clean_merged_data.py` now recodes official missing codes **8, 98, and 99** for
both `C5SPGP` and `C5SPGN` to NaN **before** computing skewness, kurtosis, the log offset, or
`C5SPGN_log`. C5SPGP and C5SPGN are **mean item scores** (valid range 1–5, not summed scores).

Audit results: exactly one code-8 value was recoded for each variable; no 98 or 99 values were
present. Valid N = 230 for each variable after recoding.

Audit file: `results/tables/panas_missing_code_recode.csv`

Downstream regeneration completed: `midus_merged_clean.csv`, `midus_with_fmri.csv`,
`panas_skew_kurtosis.csv`, and all PANAS-dependent analyses (00a, 00d, 01, 02, 04, 06, 07,
and their sensitivity variants).

**Manuscript update — DRAFTED; pending Word integration:**
- Compare corrected vs. prior PANAS sample sizes, coefficients, confidence intervals, and p-values
- Update every affected manuscript claim; do not assume prior conclusions remain valid

### 1.3 Race wording — DRAFTED; pending Word integration

Replace all manuscript references to "race/ethnicity covariates" or "race/ethnicity" with
**"race"** or **"race dummy variables (reference = White)"**. Do not call the race categories
themselves "race/ethnicity."

### 1.4 Sex coding — DRAFTED; pending Word integration

Replace "0 = male, 1 = female" in the manuscript with **"1 = male, 2 = female"** (original MIDUS
coding). No model rerun is required; the sex variable was used as intended.

### 1.5 Final fMRI sample demographics — DRAFTED; pending Word integration

Replace the stale N = 128 demographics with the correct post-rerun values:

**Conservative fMRI (N = 127, no diary requirement; Analysis 03, PANAS sensitivities):**
- Age: M = 65.7, SD = 9.5, range 48–95
- 51 male, 76 female (sex: 1 = male, 2 = female)
- Race (no missing): White 96, Black 24, Native American 1, Asian 2, Pacific Islander 0, Other 4

**Diary + conservative fMRI (N = 81; Analyses 02, 04, 06, 07):**
- Age: M = 67.2, SD = 9.8, range 48–95
- 31 male, 50 female
- Race (no missing): White 67, Black 12, Native American 1, Asian 1, Pacific Islander 0, Other 0

**Moderation complete cases (N = 80; Analyses 06 and 07):**
- Age: M = 67.1, SD = 9.7, range 48–95
- 31 male, 49 female
- Analysis 06 and Analysis 07 have **identical participant membership** for each moderator;
  no separate FC-moderation sample row is needed

Full descriptives: `results/tables/sample_descriptives.csv`

### 1.6 PANAS sample wording — DRAFTED; pending Word integration

PANAS sensitivity analyses use the **full conservative fMRI sample (N = 127)**, irrespective of
diary availability. Replace any wording like "PANAS data only (no diary)" with:

> "PANAS outcomes were examined in the full QC-passing fMRI sample (N = 127), irrespective of
> diary availability."

---

## 2. fMRI Methods and Figure 1

### 2.1 Corrected STC rerun — COMPLETED

The original `M3_slice_order.txt` contained 0-based indices (0–39), which is invalid for FSL
`slicetimer --ocustom` (requires 1-based). The corrected rerun:
- Started from untouched raw BOLD data
- Applied 1-based FSL STC via the manual reorientation → STC → orientation-restoration workflow
- Ran fMRIPrep with `--ignore slicetiming` so STC was applied exactly once
- Completed GLM for 469 runs across 158 participants

### 2.2 Corrected FD sourcing — COMPLETED

`08_process_fmri_qc.py` now sources quantitative FD exclusively from
`data/fMRI/fd_summary.csv` (derived from corrected fMRIPrep confounds). The Excel file
`data/fMRI/task_fMRI_QC.xlsx` is read for visual QC ratings only (`subject`, `run1`, `run2`,
`run3` columns); any FD columns present in Excel are discarded.

### 2.3 QC reconstruction and conservative-sample redefinition — COMPLETED

Conservative-sample criterion (all three required):
1. All three visual run ratings = Pass
2. Participant mean FD < 0.5 mm (from `fd_summary.csv`)
3. Left-amygdala negative-persistence n_pairs == 6

**Final conservative N = 127 (no diary); diary + conservative N = 81.**

`fmri_qc_processed.csv` has been regenerated. All downstream analyses have been rerun.

### 2.4 Corrected canonical fMRI inputs — COMPLETED

Seven corrected products promoted into `data/fMRI/`. Obsolete bad-STC, concatenated,
seed-based, and aCompCor-derived products removed from canonical paths and preserved in
labelled archival directories. Git history also preserves previously tracked files.

### 2.5 Figure 1 persistence definition — PENDING

Revise Figure 1 and its caption to depict persistence as:
- Voxelwise amygdala pattern for a **negative image** in run *i*
- Correlated with the pattern for a **neutral face** (following a negative image) in run *j*
- *i* ≠ *j*
- Averaged in **Fisher-z space** across all six directional cross-run pairs
- Calculated from **condition-level GLM beta maps**, not trial-level betas

Revise the FC caption to state that beta-series correlations were calculated
**separately by condition within each run**.

### 2.6 LSS event duration — DRAFTED; pending Word integration

Add to Methods:

> "For the LSS models, each image event was modeled for 6 s, encompassing the 4-s image
> presentation and the following 2-s fixation interval; the subsequent 0.5-s face was not
> included."

### 2.7 Task-activation Results correction — DRAFTED; pending Word integration

Read corrected condition-by-condition values from
`results/tables/00b_task_conditions/roi_one_sample.csv`.

The key qualitative correction: left-amygdala neutral activation does not differ significantly
from zero (*p* = .070), whereas negative and positive activation do. The manuscript must not
claim that all amygdala conditions were significant. Use exact values from the CSV when writing
the revised Results sentence.

### 2.8 Voxel-size audit — CONDITIONAL PRE-SUBMISSION CHECK

A systematic voxel-size audit of the corrected preprocessed BOLD files has not been documented
in this repository. The replacement Methods draft does not report normalized/resampled voxel
dimensions. If such a statement is retained or reintroduced during manuscript integration, it
must be verified against the analyzed images (e.g., via `fslhd` or `nibabel`) before submission
and the audit record archived. This does not require a model rerun.

### 2.9 vmPFC Fisher-z correction — COMPLETED

`prepare_persistence_vars()` in `analysis_utils.py` now Fisher-z transforms both amygdala and
vmPFC persistence predictors (`ant_vmPFC_*_image_mean_r` → `*_mean_z`) consistently.
Sensitivity analyses using vmPFC persistence (`02_sensitivity.py`, `03_sensitivity.py`) now use
`*_mean_z` predictors.

---

## 3. Brain–Behavior Results and Discussion

### 3.1 Analysis 02 — Persistence → Daily Affect (N = 81) — DRAFTED; pending Word integration

Read exact estimates from `results/tables/02_persistence_affect/` (`mlm.csv`,
`regressions.csv`, `correlations.csv`).

Key qualitative corrections:
- Left negative persistence predicts raw NA in the primary MLM but not PA or log NA.
- Correlation and OLS results differ from MLM across outcomes; each must be described
  by its own method. Do not present all three as equally primary; the MLM is the primary
  adjusted model.

### 3.2 Analysis 03 — Age → Persistence (N = 127) — DRAFTED; pending Word integration

Read exact estimates from `results/tables/03_persistence_age/`. Older age predicts lower left
negative persistence in the one-tailed correlation and MLM. OLS is not significant.
Tests are one-tailed (prespecified direction: older age → lower persistence).

### 3.3 Analysis 04 — FC → Daily Affect (N = 81) — DRAFTED; pending Word integration

Read exact estimates from `results/tables/04_fc_affect/` (`mlm.csv`, `regressions.csv`,
`correlations.csv`).

Key qualitative corrections:
- Left anterior amygdala–vmPFC FC (neg−neu) predicts NA outcomes in the primary MLM, but
  results differ across raw NA and log NA, and across methods (correlation, OLS, MLM).
- Left anterior FC does not significantly predict PA in any method.
- Posterior FC is not significant for any outcome.
- Report method-specific evidence accurately; do not conflate OLS or correlation results
  with the MLM, and do not conflate the raw-NA and log-NA findings.

### 3.4 Analysis 05 — FC → Persistence (N = 127) — DRAFTED; pending Word integration

No significant results in correlation, OLS, or MLM for either anterior or posterior FC
predicting left amygdala persistence. Read exact null results from
`results/tables/05_fc_persistence/`.

### 3.5 Moderation Analyses 06 and 07 — DRAFTED; pending Word integration

Treat all moderation findings as **exploratory**. Key correction:

**The reappraisal × anterior vmPFC FC interaction for diary PA did not reproduce in the
corrected conservative analysis and must not be retained as a main finding.** Remove all
manuscript claims presenting this as a confirmed result. If any moderation interaction appears
nominally significant in the corrected results, describe it as an exploratory observation
requiring replication.

Read all corrected moderation estimates from:
- `results/tables/06_persistence_affect_moderation/`
- `results/tables/07_fc_affect_moderation/`

Do not promote isolated sensitivity interactions without multiplicity-aware caution.

### 3.6 FC moderation simple slopes — PENDING

If any FC moderation interaction survives to merit reporting:
- Calculate simple slopes from the **exact reported MLM** (not a separate OLS model)
- Use the same complete-case sample, centered variables, covariates, and family random-intercept
- Evaluate slopes at mean ± 1 SD of the moderator
- Obtain uncertainty intervals from the fixed-effect covariance matrix
- Replace any figures and caption text based on the updated model

If the interaction does not reproduce, remove or substantially revise the simple slopes figure
and all related manuscript prose.

### 3.7 Independence and specificity claims — DRAFTED; pending Word integration

- Remove "independently of persistence" unless a joint model containing both FC and persistence
  is reported
- Replace "specific to the left hemisphere / negative condition" with wording such as:
  > "the association was observed for the left-amygdala negative-condition measure, whereas
  > corresponding right-hemisphere and positive-condition sensitivity analyses were not significant"
- Do not interpret null sensitivity results as evidence of specificity

### 3.8 PANAS sensitivity wording — DRAFTED; pending Word integration

PANAS analyses are **sensitivity analyses** (same construct, different instrument). Do not
describe them as replacements for, or confirmations of, diary-outcome effects. Use language such
as "consistent with the diary findings" or "in the PANAS sensitivity analysis."

### 3.9 Motion-reporting correction — COMPLETED

`00e_motion_check.py` Part 2 now identifies all significant primary MLM predictor–outcome pairs
from Analyses 02–04 and runs motion-adjusted OLS and MLM models for those pairs. Correlations are
not used for adjustment because they cannot control for covariates. The Part 2 console summary
reports the beta coefficient rather than NaN for adjusted models.

Motion was not significantly associated with primary persistence or FC metrics in Part 1
correlations, OLS, or MLM (minimum two-tailed *p*s = .143, .449, and .129, respectively).
The earlier missing FC MLM rows were caused by unsanitized hyphens in outcome names, which
prevented formula parsing before optimization. After the formula-name correction, all four FC
motion models converged. Read exact values from `results/tables/00e_motion_check/`.

Motion-adjusted results: `results/tables/00e_motion_check/`.

The final Part 2 set comprises persistence with raw NA, age with persistence, and anterior
connectivity with raw and log-transformed NA. All four significant primary MLM findings remained
significant after separate adjustment for mean FD across runs and mean FD during
negative-condition trials (one-tailed *p*s = .038/.026, .019/.019, .048/.040, and .018/.015,
respectively).

---

## 4. Figures and Supplementary Table

### 4.1 Supplementary all-methods table N correction — COMPLETED

The supplementary table now reports **three separate N columns** (`N correlation`, `N OLS`,
`N MLM`), each read from that method's own result file. A blank N means that method produced
no row for that predictor × outcome pair. Neither N nor statistics are copied across methods.

Files regenerated:
- `results/tables/supplementary/supp_all_methods.csv`
- `results/tables/supplementary/supp_all_methods.docx`

Verify Analysis 01 in particular: the correlation N = 1,174 while the covariate-adjusted
OLS and MLM use a smaller N after listwise deletion on covariates.

### 4.2 Main residualized scatterplots — PENDING

- Add an explicit statement that residualized scatterplots are **descriptive visualizations**
  and that statistical inference is from the MLM
- Remove "partial r" annotations unless the adjustment set is explicitly described and
  consistent with the MLM
- Update Figures 2 and 3 (persistence–affect and FC–affect) to use the corrected N = 81 sample

### 4.3 Figure 1 — PENDING (see Section 2.5)

---

## 5. Methodological Documentation

### 5.1 Sample descriptives script — COMPLETED

`07_sample_descriptives.py` now reports the nine analytic samples accurately, with separate N
columns for race (six categories) and sex (1 = male, 2 = female). Sample definitions match the
exact `get_samples()` calls in each analysis script. Denominators are explicit in column names.
"race" is used throughout; "ethnicity" is not reported.

### 5.2 Method-specific Ns in supplementary table — COMPLETED (see Section 4.1)

### 5.3 Persistence definition documentation — PENDING (see Section 2.5)

---

## 6. MR1 Validation — COMPLETED

The targeted MR1 pipeline, four direct analyses, and all predefined sensitivity families are
complete. The final MR1 interface uses correlations and participant-level OLS only; active result
directories contain no stale MLM files. None of the four direct tests reached the prespecified
significance threshold, so the conditional MR1 motion follow-up was not required. The synthetic
test suite passes all 118 tests. See `MR1_validation/` for the frozen plan, code, tests, and
aggregate results.

---

## 7. Completion Checklist

### Computational corrections — COMPLETED

- [x] Recode PANAS missing codes 8, 98, 99 in `06_clean_merged_data.py`; generate audit CSV
- [x] Regenerate cleaned/merged datasets and all PANAS-dependent analyses
- [x] Correct STC rerun from untouched raw BOLD data with 1-based FSL order
- [x] Source quantitative FD from `fd_summary.csv` (corrected fMRIPrep confounds)
- [x] Reconstruct `fmri_qc_processed.csv` with corrected FD and n_pairs criterion
- [x] Determine final conservative N = 127 (no diary) and N = 81 (diary+fMRI)
- [x] Promote seven corrected canonical fMRI products; archive obsolete bad-STC products
- [x] Regenerate all fMRI-dependent analyses with the corrected sample
- [x] Fisher-z transform vmPFC persistence predictors in `prepare_persistence_vars()`
- [x] Motion-reporting correction: Part 2 OLS/MLM only; beta coefficient in summary
- [x] Supplementary table: separate N columns per method
- [x] Confirm diary date missing codes are correctly recoded (98, 9998) — no change needed
- [x] Confirm primary FC sample uses neg-vs-neu contrast — no mismatch

### Manuscript and reporting changes — DRAFTED; integration and author review pending

- [x] Draft corrected Methods and Results from final M3 and MR1 aggregate outputs
- [x] Independently audit all traceable numerical claims against the aggregate CSVs

- [ ] Replace N = 128 with N = 127, N = 80 with N = 81 throughout
- [ ] Update conservative-sample demographics table (age, sex, race) to N = 127 values
- [ ] Correct sex coding: "1 = male, 2 = female"
- [ ] Replace "race/ethnicity" with "race" or "race dummy variables (reference = White)"
- [ ] Update PANAS sample wording (not "no diary"; instead N = 127 irrespective of diary)
- [ ] Revise Figure 1 persistence definition and FC caption
- [ ] Add 6-s LSS event duration to Methods
- [ ] Correct task-activation language: left-amygdala neutral activation is not significant
      (*p* = .070), whereas negative and positive activation are; use values from
      `results/tables/00b_task_conditions/roi_one_sample.csv`
- [ ] Update moderation prose: reappraisal × anterior FC interaction did not reproduce;
      treat all moderation as exploratory
- [ ] Remove or substantially revise simple slopes figure and prose if interaction not reproduced
- [ ] Temper "independent" and "specific" claims; add wording about null sensitivities
- [ ] Clarify descriptive status of residualized scatterplots; update Figures 2 and 3
- [ ] Describe method-specific evidence accurately in Analyses 02 and 04 Results
- [ ] Update PANAS sensitivity prose to describe as sensitivity, not primary evidence
- [x] Correct Part 1 FC MLM formula parsing, rerun the models, and report the converged null results
- [ ] Final consistency audit of all manuscript N values, statistics, and figure labels

### MR1 targeted replication

- [x] Complete and version the MR1 pipeline, direct analyses, predefined sensitivities, and tests

### Pending verification

- [ ] If normalized/resampled voxel dimensions are retained in the integrated manuscript,
      document pixdim from analyzed BOLD files before submission
