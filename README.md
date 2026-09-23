# MIDUS Amygdala Persistence

This repository contains analysis code for a preregistered study using the MIDUS 3 Neuroscience dataset examining age-related differences in affect, amygdala persistence to emotional stimuli, and model-free emotion regulation mechanisms.

The study combines daily diary data, survey-based demographics, and neuroimaging measures to test confirmatory hypotheses derived from prior work (e.g., Puccetti et al., 2021), alongside preregistered exploratory analyses of task-based functional connectivity.

**Project status (September 23, 2026): author review in progress.**
M3 uses the corrected slice-timing pipeline and the corrected vmPFC image-to-face persistence definition. The affected aggregate results and manuscript tables have been refreshed. See [M3 imaging source](scripts/fMRI/README.md) for the active script set. Earlier freeze notes and workflow examples below describe historical checkpoints; current source and aggregate tables take precedence.

**Preregistration** (updated Oct 15, 2025): OSF

---

## Corrected M3 Analysis

The original FSL custom slice-order file used 0-based indices, which is invalid for
`slicetimer --ocustom`. The corrected M3 reanalysis (`M3_stc_rerun`) began from untouched raw
BOLD data, applied a 1-based FSL slice-timing correction via the manual reorientation workflow,
and reran fMRIPrep with `--ignore slicetiming`. All downstream GLM, persistence, FC, and
brain–behavior analyses were regenerated from the corrected outputs.

**Final analytic samples (corrected rerun):**
- Conservative fMRI (no diary): **N = 127** — Analyses 03, 05 and PANAS sensitivities
- Diary + conservative fMRI: **N = 81** — Analyses 02, 04, 06, 07
- Moderation complete cases: **N = 80**

See `PREPROCESSING_SUMMARY.md` for the full provenance record and `PUBLICATION_REQUIRED_CHANGES.md`
for the item-by-item completion status.

**Data versioning:** Raw data (`data/raw/`, `data/processed/`) are not tracked by git and must
be obtained through authorized MIDUS access. Selected deidentified fMRI summary CSVs under
`data/fMRI/` and statistical results under `results/tables/` are versioned. Publication figures
under `results/figures/` are generated from scripts.

---

## MR1 Validation

The targeted MIDUS Refresher 1 (MR1) replication is complete and versioned under
`MR1_validation/`. The final primary analyses use participant-level OLS, with conservative
QC-restricted results as primary and unrestricted results retained as secondary archives.
None of the four direct MR1 tests reached the prespecified significance threshold. See
`MR1_validation/MR1_REPLICATION_PLAN.md` and `MR1_validation/README.md`.

---

## Study Goals

### Replications
1. Can we replicate age-related differences in daily life positive and negative affect?
2. Can we replicate associations between amygdala persistence to negative images and daily life affect?

### Extensions (Confirmatory)
1. Is amygdala persistence in response to negative images decreased in older adults?
2. Do differences in amygdala persistence explain age-related differences in daily life affect?

### Novel Analyses (Exploratory) — Model-Free Emotion Regulation
1. Does model-free emotion regulation capacity (vmPFC/sgACC–amygdala connectivity in response to negative images) relate to amygdala persistence?
2. Does model-free emotion regulation capacity relate to daily life positive and negative affect?
3. Is the relationship between model-free emotion regulation capacity and daily life affect moderated by the use of reappraisal or suppression?
4. Do these effects vary by age?

---

## Repository Organization

```
MIDUS_AMYGDALA_PERSISTENCE/
├── data/
│   ├── raw/                        ← not tracked by git; obtain via MIDUS access
│   │   ├── M3P5_variables.csv
│   │   ├── M3P2_variables.csv
│   │   ├── MKE2_variables.csv
│   │   └── README.md
│   ├── processed/                  ← not tracked by git
│   │   ├── daily_diary_processed.csv
│   │   ├── daily_diary_descriptives.csv
│   │   ├── m3p5_ids.csv
│   │   ├── mke2_ids.csv
│   │   ├── m3p5_demos.csv
│   │   ├── mke2_demos.csv
│   │   ├── m3p5_covariates.csv
│   │   ├── mke2_covariates.csv
│   │   ├── midus_merged.csv
│   │   ├── midus_merged_clean.csv
│   │   └── midus_with_fmri.csv
│   └── fMRI/                       ← corrected M3_stc_rerun canonical outputs (versioned)
│       ├── task_fMRI_QC.xlsx                                  ← visual QC ratings only
│       ├── all_subjects_betaSeries_LSS_all_conditions_M2ID.csv ← 156 participants
│       ├── negative_persistence_cross_run.csv                  ← 156 subj × L/R/BI
│       ├── positive_persistence_cross_run.csv                  ← 156 subj × L/R/BI
│       ├── fd_summary.csv                                      ← 469 run records, 158 subj
│       ├── fmri_qc_processed.csv                              ← derived by script 08
│       ├── vmPFC_persistence_wide.csv                         ← 151 participants
│       ├── all_subjects_roi_activations.csv                   ← 148 participants
│       └── condition_fd_wide.csv                              ← 156 participants
├── results/
│   ├── tables/
│   │   ├── sample_descriptives.csv
│   │   ├── panas_missing_code_recode.csv
│   │   ├── panas_skew_kurtosis.csv
│   │   ├── daily_diary_descriptives.csv
│   │   ├── 00a_diary_panas/          ← correlations.csv, regressions.csv, mlm.csv
│   │   ├── 00b_task_conditions/      ← roi_one_sample.csv, roi_paired.csv,
│   │   │                               fc_one_sample.csv, fc_paired.csv
│   │   ├── 00c_vmPFC_convergence/    ← ant_post_correlations.csv
│   │   ├── 00d_erq_context/          ← four subdirectories
│   │   ├── 00e_motion_check/         ← part1 and part2 CSVs
│   │   ├── 01_affect_age/            ← correlations.csv, regressions.csv, mlm.csv
│   │   │   ├── neuro_sample/
│   │   │   └── sensitivity_panas/
│   │   ├── 02_persistence_affect/    ← and sensitivity subdirectories
│   │   ├── 03_persistence_age/       ← and sensitivity subdirectories
│   │   ├── 04_fc_affect/             ← and sensitivity subdirectories
│   │   ├── 05_fc_persistence/        ← and sensitivity subdirectories
│   │   ├── 06_persistence_affect_moderation/
│   │   ├── 07_fc_affect_moderation/
│   │   └── supplementary/
│   │       ├── supp_all_methods.csv  ← N correlation, N OLS, N MLM columns
│   │       └── supp_all_methods.docx
│   └── figures/                     ← versioned generated outputs; regenerate and review after analysis changes
│       ├── figure1_roi.tiff
│       ├── figure2_persistence_affect.tiff
│       └── figure3_fc_affect.tiff
├── scripts/
│   ├── preprocessing/
│   │   ├── 01_harmonize_ids.py
│   │   ├── 02_construct_daily_diary_affect.py
│   │   ├── 03_construct_demographics.py
│   │   ├── 04_construct_covariates.py
│   │   ├── 05_merge_master_dataset.py
│   │   ├── 06_clean_merged_data.py
│   │   ├── 07_sample_descriptives.py
│   │   ├── 08_process_fmri_qc.py
│   │   └── 09_merge_fmri_data.py
│   ├── analysis/
│   │   ├── analysis_utils.py
│   │   ├── 00a_diary_panas_convergence.py
│   │   ├── 00b_task_condition_differences.py
│   │   ├── 00c_vmPFC_convergence.py
│   │   ├── 00d_erq_context.py
│   │   ├── 00e_motion_check.py
│   │   ├── 01_affect_age.py
│   │   ├── 01_sensitivity.py
│   │   ├── 02_persistence_affect.py
│   │   ├── 02_sensitivity.py
│   │   ├── 03_persistence_age.py
│   │   ├── 03_sensitivity.py
│   │   ├── 04_fc_affect.py
│   │   ├── 04_sensitivity.py
│   │   ├── 04ex_fc_affect_antpost.py
│   │   ├── 05_fc_persistence.py
│   │   ├── 05_sensitivity.py
│   │   ├── 05ex_fc_persistence_antpost.py
│   │   ├── 06_persistence_affect_moderation.py
│   │   ├── 06_sensitivity.py
│   │   ├── 07_fc_affect_moderation.py
│   │   └── 07_sensitivity.py
│   ├── fMRI/
│   │   ├── preprocessing/
│   │   │   ├── fixOrientation.py
│   │   │   ├── extractSliceTiming.py
│   │   │   ├── M3_slice_time_correction.sh
│   │   │   ├── slurm_M3_stc_parallel.sh
│   │   │   ├── slurm_fmriprep_parallel.sh
│   │   │   ├── slurm_fmriprep_parallel_pre_fs.sh
│   │   │   ├── slurm_recon_all_parallel.sh
│   │   │   ├── getMotion.py
│   │   │   └── getMotion.sh
│   │   └── analysis/
│   │       ├── runGLM.sh
│   │       ├── runGLM_aCompCor.sh
│   │       ├── runGLM_concat.sh
│   │       ├── extract_amygdala.sh
│   │       ├── extract_amygdala_aCompCor.sh
│   │       ├── extract_amygdala_concat.sh
│   │       ├── extract_vmPFC.sh
│   │       ├── extract_vmPFC_aCompCor.sh
│   │       ├── extract_roi_activations.sh
│   │       ├── extract_roi_activations_aCompCor.sh
│   │       ├── run_cross_corr.py
│   │       ├── run_cross_corr_aCompCor.py
│   │       ├── run_cross_corr_concat.py
│   │       ├── run_cross_corr_vmPFC.py
│   │       ├── run_cross_corr_vmPFC_aCompCor.py
│   │       ├── combineROIActivations.py
│   │       ├── combineROIActivations_aCompCor.py
│   │       ├── runBStaskFC_LSS.sh
│   │       ├── combineBTS_LSS.py
│   │       ├── runBStaskFC.sh
│   │       ├── combineBTS.py
│   │       ├── seedbasedBStaskFC_LSS.sh
│   │       ├── seedbasedBStaskFC.sh
│   │       ├── grouplevelSeedBasedFC.sh
│   │       └── archive/
│   └── visualization/
│       └── publication_figures.py
├── README.md
└── requirements.txt
```

---

## Analysis Run Order

Run all scripts from the **project root directory**.

| Phase | Scripts |
|-------|---------|
| 1 — Behavioral preprocessing | `01_harmonize_ids.py` → `06_clean_merged_data.py` (in order) |
| 2 — fMRI QC and merge | `08_process_fmri_qc.py`, then `09_merge_fmri_data.py` |
| 3 — Sample descriptives | `07_sample_descriptives.py` |
| 4 — Context checks | `00a`, `00b`, `00c`, `00d` (order-independent) |
| 5 — Primary analyses | `01` → `07` (order-independent within phase) |
| 6 — Sensitivities and motion | `01_sensitivity` → `07_sensitivity`, `00e_motion_check` |
| 7 — Output | `supplementary_tables.py`, `supplementary_tables_docx.py`, `publication_figures.py` |

Full numbered run order with exact commands: see `PREPROCESSING_SUMMARY.md`, Section 10.

---

## Data Download and Access

### Data Source

Data are obtained from the MIDUS repository:
**https://midus.colectica.org**

Neuroimaging data require a Data Usage Agreement and are accessed separately via ICPSR (Study 38862).

---

## Raw Data Structure

The preprocessing pipeline uses three raw datasets, each corresponding to a distinct MIDUS project. Each dataset is treated as authoritative for its domain, eliminating redundancy across files.

---

## Raw Dataset 1: MIDUS 3 Project 5 (Neuroscience and Survey Demographics)

**File:** `M3P5_variables.csv`

Contains all neuroscience project variables as well as demographic and survey variables.

**Participant Identifiers:**
- `MIDUSID` - MIDUS participant ID
- `M2ID` - MIDUS 2 ID (primary key)
- `SAMPLMAJ` - Sample membership indicator
- `M2FAMNUM` - Family number (for twins)

**Demographics:**
- `C1PRSEX` - Sex
- `C1PB1` - Education
- `C1PF7A` - Race
- `C1PF1` - Ethnicity
- `C1PBYEAR` - Birth year
- `C1PRAGE` - Age at baseline

**Interview Timing:**
- `C1PIDATE_MO`, `C1PIDATE_YR` - Baseline interview date
- `C5PDATE_MO`, `C5PDATE_YR` - Neuroscience visit date

**Neuroscience Variables:**
- `C5IC` - MRI completion indicator
- `C5PAGE` - Age at neuroscience visit
- `C5HAND` - Handedness
- `C5SER`, `C5SES` - Emotion Regulation Questionnaire (Reappraisal, Suppression)
- `C5SPGP`, `C5SPGN` - PANAS positive and negative affect

This dataset serves as the primary source of demographics and neuroscience participation variables.

---

## Raw Dataset 2: MIDUS 3 Project 2 (Daily Diary Affect)

**File:** `M3P2_variables.csv`

Contains daily diary affect data collected over an 8-day period.

**Variables:**
- `C2DC1` through `C2DC27` - Daily affect items
- `C2DDAY` - Diary day number
- `C2DIMON`, `C2DIYEAR` - Diary start date
- `M2ID` - Participant identifier

This dataset is used to compute participant-level affect measures including mean positive affect, mean negative affect, and log-transformed negative affect.

---

## Raw Dataset 3: MIDUS Milwaukee Sample Demographics

**File:** `MKE2_variables.csv`

Contains demographic information for participants in the Milwaukee oversample.

**Participant Identifiers:**
- `M2ID` - MIDUS 2 ID (primary key)
- `SAMPLMAJ` - Sample membership indicator

**Demographics:**
- `CACRSEX` - Sex
- `CACB1` - Education
- `CACF7A` - Race
- `CACF1` - Ethnicity
- `CACRAGE` - Age at interview

**Interview Timing:**
- `CACIDATE_MO`, `CACIDATE_YR` - Interview date

These variables supplement demographic information when not available from Project 5.

---

## Preprocessing Pipeline

### Script 01: Harmonize IDs

**File:** `scripts/preprocessing/01_harmonize_ids.py`

Ensures participant IDs are consistent across all three raw datasets.

**Processing Steps:**
- Standardizes ID formats
- Verifies uniqueness of M2ID within each dataset
- Logs missing or mismatched IDs across projects

**Inputs:**
- `data/raw/M3P5_variables.csv`
- `data/raw/M3P2_variables.csv`
- `data/raw/MKE2_variables.csv`

**Outputs:**
- `data/processed/m3p5_ids.csv`
- `data/processed/mke2_ids.csv`
- `data/processed/p2_ids.csv`
- `logs/id_harmonization_log.txt`

---

### Script 02: Construct Daily Diary Affect

**File:** `scripts/preprocessing/02_construct_daily_diary_affect.py`

Processes daily diary affect data to generate participant-level summary measures.

**Processing Steps:**
- Recodes MIDUS missing value codes (7, 8, 9) to NaN
- Computes participant-level mean positive affect (PA_score) and negative affect (NA_score)
- Log-transforms negative affect (NA_score_log) with small offset to handle zeros
- Counts number of diary days completed per participant
- Extracts diary start date (StartMonth, StartYear)
- Generates descriptive statistics including Cronbach's alpha, distributional statistics

**Input:**
- `data/raw/M3P2_variables.csv`

**Outputs:**
- `data/processed/daily_diary_processed.csv`
- `data/processed/daily_diary_descriptives.csv`

---

### Script 03: Construct Demographics

**File:** `scripts/preprocessing/03_construct_demographics.py`

Constructs cleaned, harmonized demographics for the MIDUS neuroscience (M3P5) and Milwaukee (MKE2) datasets independently.

**Processing Steps:**
- Preserves all original variables from each dataset
- Adds harmonized demographic columns:
  - `sex` - Sex (1 = Male, 2 = Female)
  - `educ` - Education (1-12 scale)
  - `ethnicity` - Hispanic/Latino (0 = No, 1 = Yes)
  - `race` - Race category (1 = White, 2 = Black, 3 = Native American, 4 = Asian, 5 = Pacific Islander, 6 = Other)
- Converts MIDUS missing codes to NaN
- Keeps one row per participant
- Does not merge the datasets—demographics are cleaned separately for M3P5 and MKE2

**Inputs:**
- `data/processed/m3p5_ids.csv`
- `data/processed/mke2_ids.csv`

**Outputs:**
- `data/processed/m3p5_demos.csv`
- `data/processed/mke2_demos.csv`

---

### Script 04: Construct Covariates

**File:** `scripts/preprocessing/04_construct_covariates.py`

Generates analysis-ready covariates from cleaned demographics files.

**Processing Steps:**
- **Race dummy coding:** Creates dummy variables for non-reference race categories
  - Reference category: White (race = 1)
  - Dummy variables: `race_2`, `race_3`, `race_4`, `race_5`, `race_6`
- **Twin pair dummy coding (M3P5 only):** Creates dummy variable for each twin family
  - Applies only to participants in twin sample (SAMPLMAJ = 3)
  - Creates `twin_pair_fam_XXXXX` for each twin family with 2+ participants
  - MKE2 participants have no twin pairs

**Inputs:**
- `data/processed/m3p5_demos.csv`
- `data/processed/mke2_demos.csv`

**Outputs:**
- `data/processed/m3p5_covariates.csv`
- `data/processed/mke2_covariates.csv`

---

### Script 05: Merge Master Dataset

**File:** `scripts/preprocessing/05_merge_master_dataset.py`

Merges multiple processed MIDUS datasets into a single master file keyed on M2ID.

**Inputs:**
- `data/processed/daily_diary_processed.csv` - Participant-level daily diary affect
- `data/processed/m3p5_covariates.csv` - Project 5 demographics and covariates
- `data/processed/mke2_covariates.csv` - Milwaukee demographics and covariates

**Processing Steps:**
1. Merge all datasets on M2ID using outer joins
2. For duplicate variables across datasets, combine into single harmonized column
3. Set twin_pair_ variables to 0 for MKE2 participants
4. Compute derived variables:
   - **Age at P2 (C2PAGE):**
     - M3P5: `StartYear - C1PBYEAR`
     - MKE2: Estimated from CACRAGE and interview dates
   - **Time between P2 and P5 (time_P2_P5):** Calculated in months from diary start to P5 visit

**Output:**
- `data/processed/midus_merged.csv`

---

### Script 06: Clean Merged Data

**File:** `scripts/preprocessing/06_clean_merged_data.py`

Performs final cleaning and quality checks on the merged dataset.

**Processing Steps:**
1. Verify twin_pair_ variables are 0 for MKE2 participants
2. Compute age at P2 (C2PAGE) and time between P2 and P5 in months (time_P2_P5)
3. Recode ERQ missing codes (≥97) to NaN for C5SER and C5SES
4. **Recode PANAS missing codes:** Convert C5SPGP and C5SPGN to numeric, then recode
   official codebook missing values **8, 98, and 99** to NaN before any distributional
   statistics or log transformation. C5SPGP and C5SPGN are **mean item scores** (valid
   range 1–5). An unparseable or out-of-range value causes an immediate exit.
5. Compute skewness and kurtosis for C5SPGP and C5SPGN (after recoding)
6. Compute log-transformed negative affect (C5SPGN_log) using half the minimum non-zero value
   as an offset (after recoding)
7. Save PANAS missing-code audit and distribution statistics

**Input:**
- `data/processed/midus_merged.csv`

**Outputs:**
- `data/processed/midus_merged_clean.csv`
- `results/tables/panas_missing_code_recode.csv` — one row per variable; n_code_8, n_code_98,
  n_code_99, n_total_recoded, n_missing_before, n_missing_after, n_valid, valid_min, valid_max
- `results/tables/panas_skew_kurtosis.csv`

---

### Script 07: Sample Descriptives

**File:** `scripts/preprocessing/07_sample_descriptives.py`

Generates publication-ready sample descriptives for the analytic samples used in the paper.

**Samples reported:**
- `daily_diary` — ≥1 of PA_score/NA_score nonmissing; age variable C2PAGE
- `neuroscience_age` — C5PAGE nonmissing (broader than the Analysis 01 neuro subsample)
- `diary_neuroscience_age_overlap` — diary AND C5PAGE; the exact Analysis 01 neuro subsample
- `neuroscience_panas` — ≥1 of C5SPGP/C5SPGN nonmissing; defined from PANAS variables directly
- `diary_panas_overlap` — diary AND PANAS
- `fmri_conservative` (N = 127) — conservative fMRI, no diary; Analyses 03 and PANAS sensitivities
- `fmri_fc_conservative` — conservative + FC, no diary; Analysis 05 and FC PANAS sensitivities
- `diary_fmri_conservative` (N = 81) — diary + conservative; Analyses 02 and 06
- `diary_fmri_fc_conservative` — diary + conservative + FC; Analyses 04 and 07
- `reappraisal_complete_case` / `suppression_complete_case` (N = 80) — moderation complete cases
- FC moderation versions reported only when M2ID membership differs from persistence versions

**Descriptives computed per sample:** N, N_age, age mean/SD/min/max; sex (n male, n female,
n missing, % of nonmissing); race (n and % for each of six categories, n missing).

Sex coding: 1 = male, 2 = female. Percentage denominators are nonmissing sex and nonmissing
race respectively, named explicitly in column headers. Ethnicity is not reported.

**Inputs:**
- `data/processed/midus_merged_clean.csv` (behavioral samples)
- `data/processed/midus_with_fmri.csv` via `load_master()` (fMRI samples)

**Output:**
- `results/tables/sample_descriptives.csv`

---

### Script 08: Process fMRI QC

**File:** `scripts/preprocessing/08_process_fmri_qc.py`

Generates `fmri_qc_processed.csv` from three inputs, each serving a distinct role.

**Inputs:**
- `data/fMRI/task_fMRI_QC.xlsx` — manual visual QC ratings; uses only `subject`, `run1`,
  `run2`, `run3` columns. Any FD columns present in this file are discarded immediately
  after reading to prevent merging conflicts with the corrected FD source.
- `data/fMRI/fd_summary.csv` — **authoritative quantitative FD** from corrected fMRIPrep
  confounds (469 run records). The `flagged` column is ignored.
- `data/fMRI/negative_persistence_cross_run.csv` — supplies `n_pairs` for each participant
  (left-hemisphere row only) to define task completeness.

**QC flags created:**
- `run1_pass`, `run2_pass`, `run3_pass` — individual run ratings (1 = Pass)
- `all_runs_pass` — all three runs pass (1 = yes)
- `run1_fd`, `run2_fd`, `run3_fd`, `mean_fd` — from fd_summary.csv
- `fd_pass` — mean FD < 0.5 mm (1 = yes; NaN counts as fail)
- `n_pairs` — cross-run pairs from left-hemisphere persistence file
- `task_complete` — n_pairs == 6 (1 = yes)
- `qc_conservative` — all_runs_pass AND fd_pass AND task_complete

**Output:**
- `data/fMRI/fmri_qc_processed.csv` (1 row per participant)

**Conservative sample definition (all three required):**
- All 3 runs pass visual QC
- Mean FD < 0.5 mm (from corrected fMRIPrep confounds)
- n_pairs == 6 (all six directional cross-run pairs available)

---

### Script 09: Merge fMRI Data

**File:** `scripts/preprocessing/09_merge_fmri_data.py`

Merges corrected-STC fMRI summary measures into the cleaned MIDUS master dataset.

**Required inputs:**
- `data/processed/midus_merged_clean.csv`
- `data/fMRI/all_subjects_betaSeries_LSS_all_conditions_M2ID.csv` — LSS beta-series
  (used only to set `has_beta_series` flag; FC columns are loaded separately by
  `analysis_utils.load_master(fc=True)` to avoid duplicate/suffixed columns)
- `data/fMRI/negative_persistence_cross_run.csv` — cross-run amygdala persistence
- `data/fMRI/positive_persistence_cross_run.csv`
- `data/fMRI/fd_summary.csv` — corrected FD, aggregated across runs
- `data/fMRI/fmri_qc_processed.csv` — QC flags including `qc_conservative`

**Optional inputs (merged when present):**
- `data/fMRI/vmPFC_persistence_wide.csv`
- `data/fMRI/all_subjects_roi_activations.csv`
- `data/fMRI/condition_fd_wide.csv`

**Processing:**
1. Load master dataset; all merges are left joins on M2ID; every merge is validated to
   preserve row count
2. Pivot persistence files from long (subject × hemisphere) to wide format; Fisher-z
   transformation of persistence predictors is applied in analyses via `prepare_persistence_vars()`
3. Aggregate FD across runs (mean, max, n_runs); the `flagged` column is ignored
4. Create availability flags: `has_beta_series`, `has_neg_persistence`, `has_pos_persistence`,
   `has_fd_data`, `has_imaging_data`

**Output:**
- `data/processed/midus_with_fmri.csv`

---

## Analysis Scripts

### Analysis 00a: Diary–PANAS Convergence

**File:** `scripts/analysis/00a_diary_panas_convergence.py`

Tests convergent validity between daily diary affect (P2 wave) and PANAS affect measured at the neuroimaging visit (P5 wave). Establishes convergent validity between the two instruments. PANAS is subsequently used as a convergent-validity sensitivity outcome, not as an interchangeable replacement for diary affect.

**Input:** `data/processed/midus_merged_clean.csv`

**Predictors:** PA_score, NA_score, NA_score_log (daily diary)

**Outcomes:** C5SPGP, C5SPGN, C5SPGN_log (PANAS)

**Covariates:** sex, race dummies, twin pair dummies, n_days_complete, time_P2_P5

**Analyses:** Pearson correlations, OLS, MLM

**Outputs:** `results/tables/00a_diary_panas/`
- `correlations.csv`, `regressions.csv`, `mlm.csv`, `_methods.txt`

---

### Analysis 00b: Task Condition Differences

**File:** `scripts/analysis/00b_task_condition_differences.py`

Validates task-evoked effects by testing whether ROI activations and beta-series FC differ significantly across emotional conditions. Confirms that the task is producing expected condition-level differences before using these measures in primary analyses.

**Inputs:**
- `data/processed/midus_with_fmri.csv`
- `data/fMRI/all_subjects_roi_activations.csv`
- `data/fMRI/all_subjects_betaSeries_LSS_all_conditions_M2ID.csv`

**Analyses:**
- One-sample t-tests against zero for each ROI × condition and FC × condition
- Paired contrasts: neg vs. neu, neg vs. pos

**Outputs:** `results/tables/00b_task_conditions/`
- `roi_one_sample.csv`, `roi_paired.csv`
- `fc_one_sample.csv`, `fc_paired.csv`

---

### Analysis 00c: Anterior vs. Posterior vmPFC Convergence

**File:** `scripts/analysis/00c_vmPFC_convergence.py`

Tests whether anterior and posterior vmPFC are tracking the same or distinct signals across activations, persistence, and FC measures. High correlations would suggest redundancy; low correlations support treating them as separate targets.

**Inputs:**
- `data/processed/midus_with_fmri.csv`
- `data/fMRI/all_subjects_roi_activations.csv`
- `data/fMRI/vmPFC_persistence_wide.csv`
- `data/fMRI/all_subjects_betaSeries_LSS_all_conditions_M2ID.csv`

**Outputs:** `results/tables/00c_vmPFC_convergence/`
- `ant_post_correlations.csv`

---

### Analysis 00d: ERQ Context

**File:** `scripts/analysis/00d_erq_context.py`

Provides contextual characterization of emotion regulation strategy use (ERQ reappraisal and suppression) before moderation analyses (scripts 06–07). Tests the reappraisal–suppression relationship, whether ERQ predicts affect, and whether age predicts ERQ use.

**Input:** `data/processed/midus_merged_clean.csv`

**Analyses:** Pearson correlations, OLS, MLM across four sections:
1. Reappraisal × suppression relationship
2. ERQ → daily diary affect
3. ERQ → PANAS affect
4. Age → ERQ use

**Outputs:** `results/tables/00d_erq_context/`
- `reappraisal_suppression/` — correlations.csv, regressions.csv, mlm.csv
- `erq_affect_diary/` — same
- `erq_affect_panas/` — same
- `age_erq/` — same

---

### Analysis 00e: Motion Sensitivity Check

**File:** `scripts/analysis/00e_motion_check.py`

Tests whether head motion confounds the primary fMRI findings.

**Part 1 — Motion as predictor:** Correlations + OLS + MLM testing whether mean FD (all runs) and mean FD during negative-condition trials predict amygdala persistence and amygdala–vmPFC FC. Two-tailed tests; conservative sample.

**Part 2 — Motion-controlled replication:** Automatically reads the existing conservative correlation results for scripts 02 and 04, identifies effects with p < .05, and re-runs those predictor–outcome pairs with each motion measure added as an additional covariate. Reports whether effects survive.

**Motion measures:**
- `fd_mean_across_runs` — mean FD averaged across all three runs
- `fd_neg_mean` — mean FD during negative-image trials (the persistence window)

**Outputs:** `results/tables/00e_motion_check/`
- `part1_{motion_var}_correlations.csv`, `_regressions.csv`, `_mlm.csv`
- `part2_motion_controlled_{motion_var}.csv`

---

### Analysis 01: Affect–Age Replication

**File:** `scripts/analysis/01_affect_age.py`

Replicates age-related differences in daily life affect in MIDUS 3. Tests the expected pattern that older age is associated with higher positive affect and lower negative affect.

**Input:**
- `data/processed/midus_merged_clean.csv`

**Samples:**
1. **Full diary sample** — all participants with diary affect data; predictor: C2PAGE (age at diary wave)
2. **Neuroscience subsample** — participants with both diary and fMRI visit data; predictor: C5PAGE (age at neuroscience visit)

**Outcomes:** PA_score, NA_score, NA_score_log

**Covariates:** sex, race dummies, twin pair dummies, n_days_complete

**Analyses:**
1. Zero-order Pearson correlations (age × affect); one-tailed tests in pre-specified directions
2. OLS regressions: `outcome ~ age + covariates`
3. MLM (random intercept for family): same fixed effects, twin pair dummies replaced by family grouping

**Outputs:** `results/tables/01_affect_age/`
- `correlations.csv`, `regressions.csv`, `mlm.csv`, `_methods.txt`
- `neuro_sample/` — same outputs for the neuroscience subsample (C5PAGE predictor)
- `sensitivity_panas/` — PANAS outcomes in place of daily diary

---

### Statistical Decisions and Shared Helpers

All analyses (02–07) share infrastructure via `scripts/analysis/analysis_utils.py`, which centralizes sample definitions, covariate lists, outcome lists, model fitting, and output formatting.

**Primary variables (confirmatory, no correction):**
- Persistence predictor: left amygdala cross-run negative persistence (`neg_persist_crossrun_mean_z_L`, Fisher z)
- FC predictors: left amygdala–anterior vmPFC and left amygdala–posterior vmPFC differential FC (`*_neg_vs_neu`, Fisher z)
- Affect outcomes: daily diary PA (`PA_score`), NA (`NA_score`), and log-transformed NA (`NA_score_log`)

**Directional p-values:**
- One-tailed for prespecified directional main effects, including age–affect,
  persistence–affect, persistence–age, FC–affect, and FC–persistence
- Two-tailed for exploratory moderation interactions and nondirectional hemisphere,
  condition, or other specificity analyses unless an explicit directional prediction was set

**Samples:**
- **Conservative (primary, N = 127 / N = 81):** `qc_conservative == 1` — all 3 runs pass
  visual QC, mean FD < 0.5 mm (from `fd_summary.csv`), AND n_pairs = 6 (all six directional
  cross-run pairs available). The conservative sample is the inferential sample. Full-sample
  results are retained for transparency only and must not be used for substantive conclusions.
- **Full (transparency only):** `has_neg_persistence == 1` — any imaging data present,
  includes participants with excessive motion

All analyses report conservative sample results as primary; full sample results are archived in
`full_sample/` subdirectories.

**Sensitivity analyses** (run within each main script, results saved to subdirectories):
- *Right hemisphere:* right amygdala persistence (`neg_persist_crossrun_mean_z_R`) or right amygdala FC seed
- *vmPFC persistence:* Fisher-z transformed anterior and posterior vmPFC cross-run persistence
- *Other persistence operationalizations:* positive persistence (`pos_persist_crossrun_mean_z_L`)
- *PANAS:* `C5SPGP`, `C5SPGN`, `C5SPGN_log` in the conservative fMRI sample (N = 127) —
  convergent-validity sensitivity using a different measurement instrument; reported as
  sensitivity only and not as confirmatory evidence for the diary-affect findings
- *FC condition specificity:* raw negative condition FC; positive-vs-neutral contrast
- *ROI activations:* mean amygdala and vmPFC beta per condition as alternative to persistence
- *Motion control:* mean FD (all runs) and condition-level FD added as covariates (script 00e);
  Part 2 runs OLS and MLM only (correlations cannot adjust for covariates)

---

### Analysis 02: Persistence × Affect

**File:** `scripts/analysis/02_persistence_affect.py`

Tests whether left amygdala negative persistence is associated with daily life affect (replication of Puccetti et al., 2021). Expected directions: higher persistence → lower PA, higher NA (one-tailed).

**Input:**
- `data/processed/midus_with_fmri.csv`

**Predictor:** `neg_persist_crossrun_mean_z_L` (left amygdala cross-run negative persistence, Fisher z)

**Outcomes:** PA_score, NA_score, NA_score_log

**Covariates:** C5PAGE, sex, race dummies, twin pair dummies, time_P2_P5, n_days_complete

**Analyses:**
1. Zero-order Pearson correlations (one-tailed)
2. OLS: `affect ~ persistence + covariates`
3. MLM (random intercept for family): same fixed effects, twin pair dummies replaced by family grouping

**Outputs:** `results/tables/02_persistence_affect/`
- `correlations.csv`, `regressions.csv`, `mlm.csv`, `_methods.txt` (conservative sample)
- `full_sample/` — same outputs for full sample
- `sensitivity_right_hemisphere/` — right amygdala persistence
- `sensitivity_vmpfc_persistence/` — vmPFC persistence (anterior and posterior)
- `sensitivity_other_persistence/` — positive persistence (`pos_persist_crossrun_mean_z_L`)
- `sensitivity_panas/` — PANAS outcomes
- `sensitivity_roi_activations/` — mean ROI betas as alternative to persistence

---

### Analysis 03: Persistence × Age

**File:** `scripts/analysis/03_persistence_age.py`

Tests whether left amygdala negative persistence decreases with age (Extension #1, Confirmatory). Expected direction: older age → lower persistence (one-tailed).

**Input:**
- `data/processed/midus_with_fmri.csv`

**Outcome:** `neg_persist_crossrun_mean_z_L` (left amygdala cross-run negative persistence, Fisher z)

**Predictor:** C5PAGE (age at neuroscience visit)

**Covariates:** sex, race dummies, twin pair dummies (no diary-specific covariates — diary data not required)

**Analyses:**
1. Zero-order Pearson correlations (one-tailed)
2. OLS: `persistence ~ C5PAGE + covariates`
3. MLM (random intercept for family): same fixed effects, twin pair dummies replaced by family grouping

**Outputs:** `results/tables/03_persistence_age/`
- `correlations.csv`, `regressions.csv`, `mlm.csv`, `_methods.txt` (conservative sample)
- `full_sample/` — same outputs for full sample

---

### Analysis 04: FC × Affect

**File:** `scripts/analysis/04_fc_affect.py`

Tests whether amygdala–vmPFC functional connectivity (negative-vs-neutral differential) relates to daily life affect (Exploratory Analysis #1). Expected directions: higher differential FC → higher PA, lower NA (one-tailed).

**Input:**
- `data/processed/midus_with_fmri.csv`
- `data/fMRI/all_subjects_betaSeries_LSS_all_conditions_M2ID.csv`

**FC Predictors (Fisher z-transformed, LSS beta-series correlations, neg−neu contrast):**
- `l_amyg-ant_vmPFC_neg_vs_neu` — left amygdala to anterior vmPFC
- `l_amyg-post_vmPFC_neg_vs_neu` — left amygdala to posterior vmPFC

**Outcomes:** PA_score, NA_score, NA_score_log

**Covariates:** C5PAGE, sex, race dummies, twin pair dummies, time_P2_P5, n_days_complete

**Analyses:**
1. Zero-order Pearson correlations (one-tailed)
2. OLS: `affect ~ FC + covariates`
3. MLM (random intercept for family): same fixed effects, twin pair dummies replaced by family grouping

**Outputs:** `results/tables/04_fc_affect/`
- `correlations.csv`, `regressions.csv`, `mlm.csv`, `_methods.txt` (conservative sample)
- `full_sample/` — same outputs for full sample
- `sensitivity_right_amygdala/` — right amygdala seed
- `sensitivity_neg_condition/` — raw negative condition FC (not contrast)
- `sensitivity_pos_vs_neu/` — positive-vs-neutral contrast
- `sensitivity_panas/` — PANAS outcomes

---

### Analysis 04ex: FC Anterior–Posterior Contrast → Affect

**File:** `scripts/analysis/04ex_fc_affect_antpost.py`

Exploratory follow-up to Analysis 04. Tests whether an anterior-minus-posterior vmPFC connectivity contrast (neg−neu) predicts daily diary affect and PANAS affect. The contrast captures safety-biased connectivity (relatively greater coupling with anterior vs. posterior vmPFC during negative stimuli). Not pre-registered; interpret with caution.

**Predictor:** `l_amyg-ant_minus_post_vmPFC_neg_vs_neu` (computed; right hemisphere if available)

**Outputs:** `results/tables/04ex_fc_affect_antpost/diary/`, `results/tables/04ex_fc_affect_antpost/panas/`

---

### Analysis 05: FC × Persistence

**File:** `scripts/analysis/05_fc_persistence.py`

Tests whether amygdala–vmPFC FC (negative-vs-neutral) relates to amygdala persistence (Exploratory Analysis #2). Expected direction: higher differential FC → lower persistence (one-tailed).

**Input:**
- `data/processed/midus_with_fmri.csv`
- `data/fMRI/all_subjects_betaSeries_LSS_all_conditions_M2ID.csv`

**FC Predictors:** Same two as Analysis 04 (`l_amyg-ant_vmPFC_neg_vs_neu`, `l_amyg-post_vmPFC_neg_vs_neu`)

**Outcome:** `neg_persist_crossrun_mean_z_L`

**Covariates:** C5PAGE, sex, race dummies, twin pair dummies (no diary covariates)

**Analyses:**
1. Zero-order Pearson correlations (one-tailed)
2. OLS: `persistence ~ FC + covariates`
3. MLM (random intercept for family)

**Outputs:** `results/tables/05_fc_persistence/`
- `correlations.csv`, `regressions.csv`, `mlm.csv`, `_methods.txt` (conservative sample)
- `full_sample/` — same outputs for full sample

---

### Analysis 05ex: FC Anterior–Posterior Contrast → Persistence

**File:** `scripts/analysis/05ex_fc_persistence_antpost.py`

Exploratory follow-up to Analysis 05. Tests whether the anterior-minus-posterior vmPFC connectivity contrast predicts amygdala persistence. Parallel to 04ex but with persistence as the outcome.

**Outputs:** `results/tables/05ex_fc_persistence_antpost/`

---

### Analysis 06: Persistence × Affect — Moderation by Emotion Regulation

**File:** `scripts/analysis/06_persistence_affect_moderation.py`

Tests whether self-reported emotion regulation strategy use moderates the persistence–affect association (Exploratory Analysis #3). Two-tailed tests for all interaction terms.

**Predictor:** `neg_persist_crossrun_mean_z_L` (mean-centered)

**Moderators (mean-centered):**
- C5SER — ERQ Reappraisal (1–7 scale)
- C5SES — ERQ Suppression (1–7 scale)

**Model:** `affect ~ persistence_c + moderator_c + persistence_c × moderator_c + covariates`

**Outcomes:** PA_score, NA_score, NA_score_log

**Covariates:** C5PAGE, sex, race dummies, twin pair dummies, time_P2_P5, n_days_complete

**Analyses:** OLS and MLM (random intercept for family), separately for each moderator

**Outputs:** `results/tables/06_persistence_affect_moderation/`
- `reappraisal/moderation_ols.csv`, `reappraisal/moderation_mlm.csv` (conservative sample)
- `reappraisal/full_sample/` — same for full sample
- `suppression/` — same structure for suppression moderator
- `sensitivity_right_hemisphere/` — right amygdala persistence as predictor
- `sensitivity_panas/` — PANAS outcomes

---

### Analysis 07: FC × Affect — Moderation by Emotion Regulation

**File:** `scripts/analysis/07_fc_affect_moderation.py`

Tests whether emotion regulation strategy use moderates the FC–affect association (Exploratory Analysis #3). Same moderation framework as Analysis 06, with FC as the predictor. Two-tailed tests for all interaction terms.

**FC Predictors (mean-centered):** `l_amyg-ant_vmPFC_neg_vs_neu`, `l_amyg-post_vmPFC_neg_vs_neu`

**Moderators (mean-centered):** C5SER (reappraisal), C5SES (suppression)

**Model:** `affect ~ FC_c + moderator_c + FC_c × moderator_c + covariates`

**Outcomes:** PA_score, NA_score, NA_score_log

**Covariates:** C5PAGE, sex, race dummies, twin pair dummies, time_P2_P5, n_days_complete

**Analyses:** OLS and MLM (random intercept for family), separately for each FC predictor and each moderator

**Outputs:** `results/tables/07_fc_affect_moderation/`
- `reappraisal/moderation_ols.csv`, `reappraisal/moderation_mlm.csv` (conservative sample)
- `reappraisal/full_sample/` — same for full sample
- `suppression/` — same structure for suppression moderator
- `sensitivity_right_amygdala/` — right amygdala FC predictors
- `sensitivity_panas/` — PANAS outcomes

---

### Sensitivity Variants

Sensitivity coverage varies by analysis and should be read from the current result subdirectories under `results/tables/`. Not every analysis has a paired sensitivity script, and not every paired script has been regenerated after the corrected-STC rerun. In particular, Analysis 05 sensitivities were not regenerated because the corrected primary Analysis 05 (FC → persistence) was entirely null.

MLM is integrated directly into the main scripts (01–07) rather than in separate sensitivity scripts. Each main script runs correlations, OLS, and MLM together and saves all three output tables. The MLM replaces OLS twin-pair dummies with a random intercept for family, using REML estimation with sequential optimizer fallback.

**Family grouping (`family_id`):**
- Twins (`SAMPLMAJ == 3` AND 2+ members share `M2FAMNUM`): grouped by `M2FAMNUM`
- Everyone else: own cluster (`M2ID`)

---

## Visualization Scripts

### Publication Figures

**File:** `scripts/visualization/publication_figures.py`

Generates all three main publication figures.

**Inputs:**
- `data/processed/midus_with_fmri.csv`
- `data/fMRI/all_subjects_betaSeries_LSS_all_conditions_M2ID.csv`

**Figures:**
1. **Figure 1 — ROI brain visualization:** Glass brain showing left/right amygdala seeds (yellow shades, Harvard-Oxford atlas) and anterior/posterior vmPFC target spheres (green shades, 10mm radius at MNI [-2,46,-10] and [0,26,-12]). Display mode: left sagittal, axial, right sagittal, coronal.
2. **Figure 2 — Persistence × Affect:** Two-panel residualized scatterplot (PA and NA) for left amygdala negative persistence in the conservative diary+fMRI sample. Both axes residualized on age, sex, race, diary days, and time between visits.
3. **Figure 3 — FC × Affect:** Two-panel residualized scatterplot (PA and NA) for left amygdala–anterior vmPFC FC (neg−neu contrast) in the conservative diary+fMRI sample.

**Outputs:**
- `results/figures/figure1_roi.tiff`
- `results/figures/figure2_persistence_affect.tiff`
- `results/figures/figure3_fc_affect.tiff`

**Figure Settings:**
- Format: TIFF, 300 dpi

---

### Simple Slopes: Suppression × Posterior vmPFC FC

**File:** `scripts/visualization/simple_slopes_07_suppression_vmPFC.py`

Simple slopes plot for the significant MLM interaction: suppression × posterior vmPFC–amygdala FC → PANAS negative affect. Predicted values computed from MLM fixed effects at −1 SD, mean, and +1 SD suppression. Confidence bands via delta method.

**Output:** `results/figures/simple_slopes_07_suppression_vmPFC.png`

---

### Simple Slopes: Reappraisal × Anterior vmPFC FC

**File:** `scripts/visualization/simple_slopes_07_reappraisal_antFC.py`

**Note:** The reappraisal × anterior vmPFC FC interaction did not reproduce in the corrected
conservative analysis. This script and its output require review before use; the figure and
related manuscript prose must be updated or removed based on the corrected moderation results.
If retained, simple slopes must be calculated from the exact reported MLM (same sample,
covariates, and family random-intercept structure), not a separate OLS model.

**Output:** `results/figures/simple_slopes_07_reappraisal_antFC.png`

---

### Suppression × vmPFC Moderation: Residualized Scatter

**File:** `scripts/visualization/suppression_vmPFC_moderation_residualized.py`

Residualized median-split scatterplot for the suppression × posterior vmPFC FC interaction.
Review whether the underlying interaction reproduces in the corrected results before finalizing.

**Output:** `results/figures/suppression_vmPFC_moderation_residualized.png`

---

### Supplementary Tables

**Files:**
- `scripts/visualization/supplementary_tables.py` — reads primary analysis results (01–05) and
  writes a combined CSV with correlation, OLS, and MLM results side by side. The predictor ×
  outcome index is the **union** across all three method files. Each method contributes its own
  N column (`N correlation`, `N OLS`, `N MLM`) read from its own result file; Ns are never
  copied across methods. A missing N and blank statistics indicate that method has no row for
  that pair.
- `scripts/visualization/supplementary_tables_docx.py` — formats the CSV into a
  publication-ready Word table (landscape, styled headers, all three N columns right-aligned,
  significant p-values bolded)

**Outputs:**
- `results/tables/supplementary/supp_all_methods.csv`
- `results/tables/supplementary/supp_all_methods.docx`

---

### Participant Flowchart

**File:** `scripts/visualization/participant_flowchart.py` (run `participant_flowchart.py`, assemble boxes externally)

Generates individual transparent PNG boxes for the participant flowchart, one per analytic sample. Sample sizes and descriptives loaded from `results/tables/sample_descriptives.csv`. Boxes are saved to `results/figures/flowchart_boxes/` for assembly in PowerPoint or Keynote.

---

## fMRI Processing Pipeline

These scripts were run on the Stanford Sherlock HPC cluster. Most are SLURM array jobs.
The corrected M3_stc_rerun pipeline applies a **1-based FSL slice-timing correction**
via the manual reorientation workflow (fixOrientation → FSL STC → orientation restore)
before fMRIPrep, which runs with `--ignore slicetiming`.

Pipeline order:

1. Fix orientation → 2. Slice timing correction (1-based) → 3. FreeSurfer recon-all → 4. fMRIPrep (`--ignore slicetiming`) → 5. Motion QC from confounds → 6. GLM → 7. Feature extraction (amygdala, vmPFC, ROI activations) → 8. Persistence computation → 9. Beta-series connectivity (LSS)

### fMRI Preprocessing

| Script | Description |
|--------|-------------|
| `fixOrientation.py` | Fix transposed NIfTI axes in raw BOLD data (run before all processing) |
| `extractSliceTiming.py` | Extract slice acquisition order from BIDS JSON sidecar for FSL |
| `M3_slice_time_correction.sh` | Single-subject STC prototype using FSL slicetimer with axis swapping |
| `slurm_M3_stc_parallel.sh` | SLURM array: STC for all 160 subjects (production version) |
| `slurm_recon_all_parallel.sh` | SLURM array: FreeSurfer recon-all for subjects that failed during fMRIPrep |
| `slurm_fmriprep_parallel.sh` | SLURM array: fMRIPrep v24.1.0 (main batch) |
| `slurm_fmriprep_parallel_pre_fs.sh` | SLURM array: fMRIPrep with pre-computed FreeSurfer surfaces |
| `getMotion.py` | Extract framewise displacement metrics from fMRIPrep confounds |
| `getMotion.sh` | SLURM wrapper for FD extraction (alternative to getMotion.py) |

### fMRI Analysis

**GLM and feature extraction:**

| Script | Description |
|--------|-------------|
| `runGLM.sh` | SLURM array: Per-run first-level GLM (nilearn, 6 conditions, 24 motion params) — primary |
| `runGLM_aCompCor.sh` | SLURM array: Per-run GLM with 24 motion + 6 aCompCor regressors — preprocessing comparison |
| `runGLM_concat.sh` | SLURM array: Concatenated all-runs GLM (sensitivity analysis) |
| `extract_amygdala.sh` | SLURM array: Extract voxelwise amygdala betas from per-run GLM (Harvard-Oxford 50%) — primary |
| `extract_amygdala_aCompCor.sh` | SLURM array: Extract amygdala betas from aCompCor GLM — prepared for preprocessing comparison, not used in primary analyses |
| `extract_amygdala_concat.sh` | SLURM array: Extract voxelwise amygdala betas from concatenated GLM |
| `extract_vmPFC.sh` | SLURM array: Extract voxelwise vmPFC betas (ant/post spheres × 3 image conditions) for vmPFC persistence — primary |
| `extract_vmPFC_aCompCor.sh` | SLURM array: Extract vmPFC betas from aCompCor GLM — prepared for preprocessing comparison, not used in primary analyses |
| `extract_roi_activations.sh` | SLURM array: Extract mean beta per ROI (L/R amygdala, ant/post vmPFC) per condition for task validation — primary |
| `extract_roi_activations_aCompCor.sh` | SLURM array: Extract mean ROI betas from aCompCor GLM — prepared for preprocessing comparison, not used in primary analyses |
| `combineROIActivations.py` | Combine per-subject ROI activation CSVs into group file (`all_subjects_roi_activations.csv`) |
| `combineROIActivations_aCompCor.py` | Same as above for aCompCor GLM output |

**Persistence computation:**

| Script | Description |
|--------|-------------|
| `run_cross_corr.py` | Compute cross-run voxelwise persistence for amygdala (primary measure) |
| `run_cross_corr_aCompCor.py` | Same as above using aCompCor GLM output — prepared for preprocessing comparison, not used in primary analyses |
| `run_cross_corr_concat.py` | Compute concatenated persistence (sensitivity measure) |
| `run_cross_corr_vmPFC.py` | Compute cross-run voxelwise persistence for vmPFC seeds (secondary comparison) |
| `run_cross_corr_vmPFC_aCompCor.py` | Same as above using aCompCor GLM output — prepared for preprocessing comparison, not used in primary analyses |

**Beta-series functional connectivity (LSS — primary):**

| Script | Description |
|--------|-------------|
| `runBStaskFC_LSS.sh` | SLURM array: ROI-level LSS beta-series FC (amygdala–vmPFC, 3 conditions: neg, neu, pos) — PRIMARY |
| `combineBTS_LSS.py` | Combine per-subject LSS ROI CSVs into group file (`all_subjects_betaSeries_LSS_all_conditions_M2ID.csv`) |
| `runBStaskFC.sh` | SLURM array: ROI-level beta-series FC using standard LSA estimation — prepared for method comparison; LSS preferred and used in all primary analyses |
| `combineBTS.py` | Combine per-subject LSA ROI CSVs — companion to `runBStaskFC.sh` |
| `seedbasedBStaskFC_LSS.sh` | SLURM array: Voxelwise LSS seed-based FC (outputs per-condition maps for neg, neu, pos) |
| `seedbasedBStaskFC.sh` | SLURM array: Voxelwise LSA seed-based FC — prepared for method comparison; not used in primary analyses |
| `grouplevelSeedBasedFC.sh` | Group-level TFCE permutation testing (5000 perms) for per-condition maps × 2 seeds × 2 analyses |

**Beta-series functional connectivity (LSA — archived):**

Original LSA scripts are preserved in `archive/` for reference. LSA beta-series is invalid for rapid event-related designs with ISI < 12s; LSS should be used instead.

| Script | Description |
|--------|-------------|
| `archive/runBStaskFC_LSA.sh` | (Archived) LSA ROI-level beta-series FC |
| `archive/combineBTS_LSA.py` | (Archived) Combine per-subject LSA CSVs |
| `archive/seedbasedBStaskFC_LSA.sh` | (Archived) LSA voxelwise seed-based FC |
| `archive/grouplevelSeedBasedFC_LSA.sh` | (Archived) Group-level TFCE for LSA maps |

**Common GLM parameters across scripts:**
- TR = 2.0s, HRF = Glover, drift = cosine (1/128 Hz), noise = AR(1)
- 24 motion regressors (6 params + temporal derivatives + quadratic terms; primary)
- Optional: + 6 aCompCor components (a_comp_cor_00 through a_comp_cor_05; preprocessing comparison)
- First 4 dummy scans removed
- Amygdala mask: Harvard-Oxford atlas, 50% probability threshold, 2mm
- vmPFC seeds: anterior vmPFC (10mm sphere at [-2, 46, -10]), posterior vmPFC (10mm sphere at [0, 26, -12])

---

## Citation and Acknowledgment

Data are provided by the Midlife in the United States (MIDUS) study. Users must comply with all
MIDUS data use agreements and citation requirements. Original MIDUS data are not included in
this repository; access may be requested at [midus.wisc.edu](https://midus.wisc.edu).

Deidentified fMRI summary CSVs under `data/fMRI/`, statistical result tables under
`results/tables/`, and publication figures under `results/figures/` are versioned in this
repository. Raw behavioral data (`data/raw/`) and processed individual-level data
(`data/processed/`) are not tracked by git and must be regenerated from authorized MIDUS data.

Script development and initial drafts of the Methods and Results sections were assisted by
Claude Code (Anthropic), with all content verified and edited by the authors.
