# MIDUS Amygdala Persistence

This repository contains analysis code for a preregistered study using the MIDUS 3 Neuroscience dataset examining age-related differences in affect, amygdala persistence to emotional stimuli, and model-free emotion regulation mechanisms.

The study combines daily diary data, survey-based demographics, and neuroimaging measures to test confirmatory hypotheses derived from prior work (e.g., Puccetti et al., 2021), alongside preregistered exploratory analyses of task-based functional connectivity.

**Preregistration** (updated Oct 15, 2025): OSF

---

## Study Goals

### Replication (Confirmatory)
1. Replicate age-related decreases in daily life negative affect and increases in positive affect
2. Replicate associations between amygdala persistence to negative images and daily life affect

### Extensions (Confirmatory)
1. Test whether amygdala persistence to negative images decreases with age
2. Test whether age-related differences in affect are mediated by amygdala persistence

### Extensions (Methodological)
1. Compare amygdala persistence to vmPFC persistence (same measure, comparison ROI)
2. Validate task effects via amygdala and vmPFC activation maps and individual-level ROI activations
3. Compare preprocessing strategies: 24 motion parameters (original) vs. 24 motion + 6 aCompCor (modern standard)

### Novel Analyses (Exploratory)
1. Examine model-free emotion regulation indexed by task-based amygdala–vmPFC connectivity using Least Squares Separate (LSS) beta-series modeling
2. Test whether model-free emotion regulation relates to amygdala persistence and daily life affect
3. Examine moderation by emotion regulation strategy use (reappraisal, suppression)
4. Test age-related differences in these effects, including analyses restricted to older adults

---

## Project Structure
```
MIDUS_AMYGDALA_PERSISTENCE/
├── data/
│   ├── raw/
│   │   ├── M3P5_variables.csv
│   │   ├── M3P2_variables.csv
│   │   ├── MKE2_variables.csv
│   │   └── README.md
│   ├── processed/
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
│   └── fMRI/
│       ├── all_subjects_betaSeries_LSS_all_conditions_M2ID.csv
│       ├── all_subjects_roi_activations.csv
│       ├── negative_persistence_cross_run.csv
│       ├── positive_persistence_cross_run.csv
│       └── fd_summary.csv
├── results/
│   ├── tables/
│   │   ├── 01_affect_age_correlations.csv
│   │   ├── 01_affect_age_regressions.csv
│   │   ├── 02_persistence_affect_correlations_{full,conservative}.csv
│   │   ├── 02_persistence_affect_regressions_{full,conservative}.csv
│   │   ├── 03_persistence_age_correlations_{full,conservative}.csv
│   │   ├── 03_persistence_age_regressions_{full,conservative}.csv
│   │   ├── 04_fc_affect_correlations_{full,conservative}.csv
│   │   ├── 04_fc_affect_regressions_{full,conservative}.csv
│   │   ├── 05_fc_persistence_correlations_{full,conservative}.csv
│   │   ├── 05_fc_persistence_regressions_{full,conservative}.csv
│   │   ├── 06_persistence_affect_moderation_{full,conservative}.csv
│   │   └── 07_fc_affect_moderation_{full,conservative}.csv
│   └── figures/
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
│   │   ├── analysis_utils.py              ← shared helpers, sample definitions, tiering
│   │   ├── 00a_diary_panas_convergence.py ← preliminary: diary–PANAS convergence check
│   │   ├── 00b_task_condition_differences.py ← preliminary: task activation condition check
│   │   ├── 00c_vmPFC_convergence.py       ← preliminary: vmPFC persistence convergence
│   │   ├── 00d_erq_context.py             ← preliminary: ERQ descriptives
│   │   ├── 01_affect_age.py               ← main: affect–age replication (OLS)
│   │   ├── 01_sensitivity.py              ← sensitivity: MLM variant
│   │   ├── 02_persistence_affect.py       ← main: persistence–affect (OLS)
│   │   ├── 02_sensitivity.py              ← sensitivity: MLM variant
│   │   ├── 03_persistence_age.py          ← main: persistence–age (OLS)
│   │   ├── 03_sensitivity.py              ← sensitivity: MLM variant
│   │   ├── 04_fc_affect.py                ← main: FC–affect (OLS)
│   │   ├── 04_sensitivity.py              ← sensitivity: MLM variant
│   │   ├── 04ex_fc_affect_antpost.py      ← exploratory: anterior vs posterior vmPFC
│   │   ├── 05_fc_persistence.py           ← main: FC–persistence (OLS)
│   │   ├── 05_sensitivity.py              ← sensitivity: MLM variant
│   │   ├── 05ex_fc_persistence_antpost.py ← exploratory: anterior vs posterior vmPFC
│   │   ├── 06_persistence_affect_moderation.py ← main: persistence × ER moderation (OLS)
│   │   ├── 06_sensitivity.py              ← sensitivity: MLM + right hemisphere
│   │   ├── 07_fc_affect_moderation.py     ← main: FC × ER moderation (OLS)
│   │   └── 07_sensitivity.py              ← sensitivity: MLM + right amygdala
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
│   │       ├── runGLM.sh                        ← PRIMARY: per-run GLM (24 motion params)
│   │       ├── runGLM_aCompCor.sh               ← COMPARISON: GLM with aCompCor regressors
│   │       ├── runGLM_concat.sh                 ← concatenated GLM (sensitivity)
│   │       ├── extract_amygdala.sh              ← voxelwise amygdala betas (primary)
│   │       ├── extract_amygdala_aCompCor.sh     ← voxelwise amygdala betas (aCompCor)
│   │       ├── extract_amygdala_concat.sh       ← concatenated betas (sensitivity)
│   │       ├── extract_vmPFC.sh                 ← vmPFC betas (primary)
│   │       ├── extract_vmPFC_aCompCor.sh        ← vmPFC betas (aCompCor)
│   │       ├── extract_roi_activations.sh       ← mean ROI betas for task validation
│   │       ├── extract_roi_activations_aCompCor.sh
│   │       ├── run_cross_corr.py                ← amygdala cross-run persistence (primary)
│   │       ├── run_cross_corr_aCompCor.py       ← amygdala persistence (aCompCor)
│   │       ├── run_cross_corr_concat.py         ← concatenated persistence (sensitivity)
│   │       ├── run_cross_corr_vmPFC.py          ← vmPFC cross-run persistence
│   │       ├── run_cross_corr_vmPFC_aCompCor.py
│   │       ├── combineROIActivations.py
│   │       ├── combineROIActivations_aCompCor.py
│   │       ├── runBStaskFC_LSS.sh               ← PRIMARY: ROI-level LSS beta-series FC
│   │       ├── combineBTS_LSS.py
│   │       ├── runBStaskFC.sh                   ← LSA beta-series FC (archived method)
│   │       ├── combineBTS.py
│   │       ├── seedbasedBStaskFC_LSS.sh         ← voxelwise LSS seed-based FC maps
│   │       ├── seedbasedBStaskFC.sh
│   │       ├── grouplevelSeedBasedFC.sh         ← group-level TFCE permutation testing
│   │       └── archive/                         ← superseded LSA scripts
│   └── visualization/
│       └── publication_figures.py               ← all 3 publication figures
├── README.md
└── requirements.txt
```

**Note:** Raw and processed data are not tracked by git and must be obtained through authorized MIDUS access.

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
2. Ensure MIDUS missing codes are set to NaN
3. Compute PANAS affect statistics:
   - Skewness and kurtosis for C5SPGP and C5SPGN
   - Log-transformed negative affect (C5SPGN_log)
4. Verify variable types and remove redundant columns
5. Save summary table of PANAS distributions

**Input:**
- `data/processed/midus_merged.csv`

**Outputs:**
- `data/processed/midus_merged_clean.csv`
- `results/tables/panas_skew_kurtosis.csv`

---

### Script 07: Sample Descriptives

**File:** `scripts/preprocessing/07_sample_descriptives.py`

Generates publication-ready sample descriptives for 5 MIDUS analytic samples.

**Samples:**
1. **daily_diary_full** - All participants with diary data (StartYear present)
2. **neuro_full** - All participants with neuroscience data (C5PDATE_YR present)
3. **daily_neuro_overlap** - Participants with both diary and neuroscience data
4. **neuro_imaging** - Participants who completed neuroimaging (C5IC = 1)
5. **imaging_daily_overlap** - Participants with both imaging and diary data

**Descriptives Computed:**
- **Age:** Mean ± SD, range (C2PAGE for diary-only, C5PAGE for neuroscience samples)
- **Sex:** % Female
- **Education:** Mean ± SD (1-12 scale)
- **Ethnicity:** % Hispanic/Latino
- **Race:** % White, Black, Native American, Asian, Pacific Islander, Other

**Input:**
- `data/processed/midus_merged_clean.csv`

**Output:**
- `results/tables/sample_descriptives.csv`

---

### Script 08: Process fMRI QC

**File:** `scripts/preprocessing/08_process_fmri_qc.py`

Processes fMRI quality control data from manual inspection.

**Input:**
- `data/fMRI/task_fMRI_QC.xlsx` (manual QC ratings from visual inspection)

**Processing Steps:**
1. Load QC Excel file with run-level pass/fail ratings and framewise displacement
2. Convert subject IDs to M2ID format
3. Create binary QC flags:
   - `run1_pass`, `run2_pass`, `run3_pass` - Individual run QC (1 = Pass, 0 = Fail/NaN)
   - `all_runs_pass` - All 3 runs pass QC (1 = yes, 0 = no)
   - `fd_pass` - Mean FD < 0.5 mm (1 = yes, 0 = no)
   - `qc_conservative` - Both all_runs_pass AND fd_pass (conservative sample criterion)
4. Generate summary statistics and exclusion breakdown

**Output:**
- `data/fMRI/fmri_qc_processed.csv` (processed QC flags, 1 row per participant)

**Conservative Sample Criteria:**
- All 3 functional runs pass visual QC
- Mean framewise displacement < 0.5 mm across all runs

---

### Script 09: Merge fMRI Data

**File:** `scripts/preprocessing/09_merge_fmri_data.py`

Merges fMRI-derived participant-level measures into the cleaned MIDUS master dataset.

**Inputs:**
- `data/processed/midus_merged_clean.csv`
- **fMRI data files (required):**
  - `data/fMRI/betaSeries_neg_vs_neu_threat_safety.csv` - Beta-series connectivity (1 row/participant)
  - `data/fMRI/negative_persistence_concat.csv` - Negative persistence concatenated (3 rows/participant)
  - `data/fMRI/negative_persistence_cross_run.csv` - Negative persistence cross-run (3 rows/participant)
  - `data/fMRI/positive_persistence_cross_run.csv` - Positive persistence cross-run (3 rows/participant)
  - `data/fMRI/fd_summary.csv` - Framewise displacement (multiple rows/participant)
  - `data/fMRI/fmri_qc_processed.csv` - Quality control flags (1 row/participant)
- **fMRI data files (optional — merged when present):**
  - `data/fMRI/vmPFC_persistence_wide.csv` - vmPFC cross-run persistence (from `run_cross_corr_vmPFC.py`)
  - `data/fMRI/all_subjects_roi_activations.csv` - ROI activation means (from `extract_roi_activations.sh`)

**Processing Steps:**
1. Load cleaned master dataset
2. Load and process fMRI files:
   - Rename beta-series connectivity columns (e.g., `l_amyg-ant_vmPFC` → `conn_l_amyg_ant_vmPFC_neg_vs_neu`)
   - Pivot hemisphere-wise persistence data from long to wide format
   - Aggregate framewise displacement across runs
3. Merge all fMRI data with master dataset using left joins on M2ID
4. Optionally merge vmPFC persistence and ROI activations if files are present
5. Merge QC flags from processed QC file
6. Create availability flags:
   - `has_beta_series` - Has beta-series connectivity data
   - `has_neg_persistence` - Has negative persistence data
   - `has_pos_persistence` - Has positive persistence data
   - `has_fd_data` - Has framewise displacement data
   - `has_imaging_data` - Has any imaging data
7. No participant exclusions applied (QC flags used for sample definition in analyses)

**Output:**
- `data/processed/midus_with_fmri.csv`

---

## Analysis Scripts

### Analysis 01: Affect–Age Replication

**File:** `scripts/analysis/01_affect_age.py`

Replicates age-related differences in affect in MIDUS 3.

**Input:**
- `data/processed/midus_merged_clean.csv`

**Samples:**
1. **Daily diary full sample**
   - Filter: PA_score not missing
   - Outcomes: PA_score, NA_score, NA_score_log
   - Predictor: C2PAGE (age at P2)

2. **Neuroscience sample**
   - Filter: C5PAGE not missing
   - Outcomes: PA_score, NA_score, NA_score_log, C5SPGP, C5SPGN
   - Predictors:
     - Daily diary outcomes: C2PAGE and C5PAGE
     - PANAS outcomes: C5PAGE only

**Covariates:**
- Sex, education, race dummies, twin pair dummies

**Analyses:**
1. Zero-order Pearson correlations between age and affect outcomes
2. OLS regressions: `outcome ~ age + covariates`

**Outputs:**
- `results/tables/01_affect_age_correlations.csv` - Correlation results
- `results/tables/01_affect_age_regressions.csv` - Regression results

**Notes:**
- All analyses use complete cases for outcome and covariates
- Minimum N: 10 for correlations, 20 for regressions

---

### Analysis Tiering and Statistical Decisions

All analyses (02-07) assign each result to a **tier** (primary, secondary, or sensitivity) based on the variables involved. This organizes the large number of tests into a clear hierarchy aligned with the pre-registered hypotheses, without requiring formal multiple comparisons correction.

**Persistence variable tier:**
- **Primary:** Left amygdala cross-run negative persistence (`neg_persist_crossrun_mean_z_L`)
- **Secondary:** Right amygdala cross-run negative persistence (`neg_persist_crossrun_mean_z_R`); vmPFC negative image persistence (`ant_vmPFC_neg_image_mean_z`, `post_vmPFC_neg_image_mean_z`)
- **Sensitivity:** Positive persistence, concatenated negative persistence

**Affect variable tier:**
- **Primary:** Daily diary PA (`PA_score`) and NA (`NA_score`)
- **Sensitivity:** `NA_score_log`, PANAS outcomes (`C5SPGP`, `C5SPGN`, `C5SPGN_log`)

**FC variable tier (condition-level scripts 04, 05, 07):**
- **Primary:** Negative condition, left amygdala seed, anterior/posterior vmPFC target
- **Secondary:** Negative condition, right amygdala seed, anterior/posterior vmPFC target
- **Sensitivity:** Neutral and positive conditions (neu, pos), safety_vs_threat derived variables

**Row-level tier** = worst (least primary) tier among its component variables. For example, a primary FC variable paired with a sensitivity affect variable yields a sensitivity-tier result.

**One-tailed p-values** are reported for persistence analyses with directional hypotheses:
- Scripts 02, 06: Persistence-affect association (p/2 for the observed direction)
- Script 03: Persistence decreases with age (p/2 if coefficient < 0, else 1 - p/2)

**Two-tailed p-values** are used for all FC analyses (04, 05, 07) and all interaction terms (exploratory).

Tier assignments and shared helpers (sample definitions, covariate lists, outcome lists) are centralized in `scripts/analysis/analysis_utils.py`.

---

### Analysis 02: Persistence × Affect

**File:** `scripts/analysis/02_persistence_affect.py`

Tests associations between amygdala persistence to negative images and daily life affect (replication of Puccetti et al., 2021).

**Input:**
- `data/processed/midus_with_fmri.csv`

**Samples:**
1. **Full sample** - All participants with imaging + affect data
   - Filter: `has_neg_persistence == 1` AND any affect measure present
2. **Conservative sample** - Strict QC criteria
   - Filter: `qc_conservative == 1` (all 3 runs pass QC AND mean FD < 0.5 mm)

**Persistence Measures (Fisher z-transformed):**
- **Primary (Confirmatory):**
  - Cross-run negative persistence (L, R)
- **Secondary (Comparison ROI):**
  - vmPFC negative image persistence (ant_vmPFC, post_vmPFC) — if `vmPFC_persistence_wide.csv` present
- **Sensitivity:**
  - Cross-run positive persistence (L, R)
  - Concatenated negative persistence (L, R)

**Affect Outcomes:**
- **Primary:** Daily diary PA (`PA_score`), NA (`NA_score`)
- **Sensitivity:** `NA_score_log`, PANAS (`C5SPGP`, `C5SPGN`, `C5SPGN_log`)

**Covariates:**
- Age (C5PAGE)
- Sex
- Race (dummy-coded)
- Twin pairs (dummy-coded)
- Time between visits (time_P2_P5)
- Number of diary days completed (n_days_complete)

**Analyses:**
1. Zero-order Pearson correlations between persistence and affect
2. OLS regressions: `affect ~ persistence + covariates`

**Outputs:**
- `results/tables/02_persistence_affect_correlations_full.csv`
- `results/tables/02_persistence_affect_regressions_full.csv`
- `results/tables/02_persistence_affect_correlations_conservative.csv`
- `results/tables/02_persistence_affect_regressions_conservative.csv`

**Notes:**
- Primary analyses use cross-run negative persistence (L, R); vmPFC persistence included as secondary if data available
- Conservative sample is primary per preregistration
- One-tailed p-values reported for persistence-affect associations
- All results tiered (primary/secondary/sensitivity) using `tier_utils.py`

---

### Analysis 03: Persistence × Age

**File:** `scripts/analysis/03_persistence_age.py`

Tests whether amygdala persistence to negative images decreases with age (Extension #1, Confirmatory). The hypothesis is directional: negative persistence should be negatively associated with age.

**Input:**
- `data/processed/midus_with_fmri.csv`

**Samples:**
1. **Full sample** - `has_neg_persistence == 1` AND valid `C5PAGE`
2. **Conservative sample** - Full + `qc_conservative == 1`

**Persistence Outcomes (Fisher z-transformed):**
- **Primary:** Cross-run negative persistence (L, R)
- **Secondary:** vmPFC negative image persistence (ant_vmPFC, post_vmPFC) — if data available
- **Sensitivity:** Cross-run positive persistence, concatenated negative persistence

**Predictor:** C5PAGE (age at neuroscience visit)

**Covariates:** sex, race dummies, twin pair dummies, time_P2_P5, n_days_complete
- Note: C5PAGE is the predictor, not a covariate

**Analyses:**
1. Zero-order Pearson correlations (persistence ~ age)
2. OLS regressions: `persistence ~ C5PAGE + covariates`

**Outputs:**
- `results/tables/03_persistence_age_correlations_full.csv`
- `results/tables/03_persistence_age_regressions_full.csv`
- `results/tables/03_persistence_age_correlations_conservative.csv`
- `results/tables/03_persistence_age_regressions_conservative.csv`

---

### Analysis 04: Condition-Level FC × Affect

**File:** `scripts/analysis/04_fc_affect.py`

Tests whether condition-level amygdala–vmPFC functional connectivity relates to daily-life affect (Exploratory Analysis #1). Uses per-condition FC from a 3-condition GLM (negative, neutral, positive) rather than relying solely on a contrast.

**Input:**
- `data/processed/midus_with_fmri.csv`
- `data/fMRI/all_subjects_betaSeries_LSS_all_conditions_M2ID.csv` (LSS; preferred)
  Fallback: `data/fMRI/all_subjects_betaSeries_all_conditions_M2ID.csv` (LSA)

**FC Variables:**
- ROI-level beta-series correlations (Fisher z-transformed) between amygdala seeds (L, R) and vmPFC targets (anterior = safety signaling, posterior = threat signaling), per Tashjian et al. (2021, TICS)
- 4 seed–target pairs × 3 conditions (neg, neu, pos) = 12 base FC variables
- 2 derived safety-vs-threat variables per condition (anterior minus posterior) × 3 conditions = 6 derived variables

**Affect Outcomes:** 6 (PA_score, NA_score, NA_score_log, C5SPGP, C5SPGN, C5SPGN_log)

**Analyses:**
1. Bivariate Pearson correlations between all FC × affect pairs
2. OLS regressions: `affect ~ FC + covariates`

**Samples:**
- Full: `has_beta_series == 1` AND `has_neg_persistence == 1`
- Conservative: full + `qc_conservative == 1`

**Outputs:**
- `results/tables/04_fc_affect_correlations_full.csv`
- `results/tables/04_fc_affect_correlations_conservative.csv`
- `results/tables/04_fc_affect_regressions_full.csv`
- `results/tables/04_fc_affect_regressions_conservative.csv`

---

### Analysis 05: Condition-Level FC × Persistence

**File:** `scripts/analysis/05_fc_persistence.py`

Tests whether condition-level amygdala–vmPFC functional connectivity relates to amygdala (and vmPFC) persistence (Exploratory Analysis #2). Same condition-level approach as Analysis 04.

**Input:**
- `data/processed/midus_with_fmri.csv`
- `data/fMRI/all_subjects_betaSeries_LSS_all_conditions_M2ID.csv` (LSS; preferred)
  Fallback: `data/fMRI/all_subjects_betaSeries_all_conditions_M2ID.csv` (LSA)

**FC Variables:** Same 18 FC variables as Analysis 04 (12 base + 6 safety-vs-threat derived).

**Persistence Measures (Fisher z-transformed):**
- **Primary:** Cross-run negative persistence (L, R)
- **Secondary:** vmPFC negative image persistence (ant_vmPFC, post_vmPFC) — if data available
- **Sensitivity:** Cross-run positive persistence (L, R), concatenated negative persistence (L, R)

**Analyses:**
1. Bivariate Pearson correlations between all FC × persistence pairs
2. OLS regressions: `persistence ~ FC + covariates`

**Samples:**
- Full: `has_neg_persistence == 1` AND FC data present
- Conservative: full + `qc_conservative == 1`

**Outputs:**
- `results/tables/05_fc_persistence_correlations_full.csv`
- `results/tables/05_fc_persistence_correlations_conservative.csv`
- `results/tables/05_fc_persistence_regressions_full.csv`
- `results/tables/05_fc_persistence_regressions_conservative.csv`

---

### Analysis 06: Persistence × Affect — Moderation by Emotion Regulation

**File:** `scripts/analysis/06_persistence_affect_moderation.py`

Tests whether self-reported emotion regulation strategy use moderates the persistence–affect association (Exploratory Analysis #3).

**Moderators:**
- C5SER — ERQ Reappraisal (1–7 scale)
- C5SES — ERQ Suppression (1–7 scale)

**Model:** `affect ~ persistence + moderator + persistence×moderator + covariates`

Both persistence and moderator are mean-centered before creating the interaction term to reduce multicollinearity and aid interpretation. The interaction term is the key test: does the persistence–affect slope change as a function of the moderator?

**Variables:**
- Persistence: amygdala (L, R) + vmPFC (ant, post, if available) cross-run negative persistence, positive persistence (L, R), concatenated negative persistence (L, R) — all Fisher z-transformed
- Affect: 6 outcomes (PA_score, NA_score, NA_score_log, C5SPGP, C5SPGN, C5SPGN_log)
- Covariates: C5PAGE, sex, race dummies, twin dummies, time_P2_P5, n_days_complete
- Diary-specific covariates (time_P2_P5, n_days_complete) only for daily diary outcomes

**Samples:**
- Full: all participants with persistence + affect + ER data
- Conservative: full + strict QC (qc_conservative == 1)

**Outputs:**
- `results/tables/06_persistence_affect_moderation_full.csv`
- `results/tables/06_persistence_affect_moderation_conservative.csv`

---

### Analysis 07: FC × Affect — Moderation by Emotion Regulation

**File:** `scripts/analysis/07_fc_affect_moderation.py`

Tests whether self-reported emotion regulation strategy use moderates the FC-affect association (Exploratory Analysis #3). Same moderation framework as Analysis 06, but with per-condition FC as the predictor instead of persistence.

**Input:**
- `data/processed/midus_with_fmri.csv`
- `data/fMRI/all_subjects_betaSeries_LSS_all_conditions_M2ID.csv` (LSS; preferred)
  Fallback: `data/fMRI/all_subjects_betaSeries_all_conditions_M2ID.csv` (LSA)

**Moderators:**
- C5SER - ERQ Reappraisal (1-7 scale)
- C5SES - ERQ Suppression (1-7 scale)

**Model:** `affect ~ FC + moderator + FC*moderator + covariates`

**FC Variables:** Same 18 per-condition FC variables as Analyses 04/05 (4 seed-target pairs × 3 conditions + 6 safety-vs-threat derived)

**Affect Outcomes:** 6 (PA_score, NA_score, NA_score_log, C5SPGP, C5SPGN, C5SPGN_log)

**Covariates:** C5PAGE, sex, race dummies, twin dummies; diary-specific covariates (time_P2_P5, n_days_complete) only for daily diary outcomes.

**Samples:**
- Full: FC data present AND `has_neg_persistence == 1`
- Conservative: full + `qc_conservative == 1`

**Notes:**
- All results tiered (primary/secondary/sensitivity); negative condition is primary
- Two-tailed tests for all interaction and FC effects (exploratory)

**Outputs:**
- `results/tables/07_fc_affect_moderation_full.csv`
- `results/tables/07_fc_affect_moderation_conservative.csv`

---

### Sensitivity (MLM) Variants

Each primary OLS analysis (01–07) has a paired sensitivity script (`01_sensitivity.py` through `07_sensitivity.py`). These replace OLS regressions (with twin pair dummies) with linear mixed-effects models, avoiding the degrees-of-freedom cost of per-pair dummies while properly accounting for non-independence within twin families via a random intercept. Sensitivity scripts for 06 and 07 also include right hemisphere/right amygdala specificity checks.

**Grouping Variable (`family_id`):**
- Twins (`SAMPLMAJ == 3` AND 2+ members share `M2FAMNUM`): grouped by `M2FAMNUM`
- Everyone else: own cluster (`M2ID`)

**Model:**
- Fixed effects: same predictors as paired OLS script, minus twin pair dummies
- Random effects: random intercept for `family_id`
- Estimation: REML (optimizer: LBFGS with Powell fallback)

**Notes:**
- Correlations are identical to the parent OLS script and are not re-run
- Zero-variance covariates are automatically dropped
- Directional p-values preserved (one-tailed for persistence; two-tailed for FC)

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
2. **Figure 2 — Persistence × Affect:** Two-panel scatterplot (PA and log-NA) for left amygdala negative persistence (Fisher z) in the conservative diary+fMRI sample.
3. **Figure 3 — FC × Affect:** Two-panel scatterplot (PA and log-NA) for left amygdala–anterior vmPFC FC (neg−neu contrast) in the conservative diary+fMRI sample.

**Outputs:**
- `results/figures/figure1_roi.tiff`
- `results/figures/figure2_persistence_affect.tiff`
- `results/figures/figure3_fc_affect.tiff`

**Figure Settings:**
- Format: TIFF, 300 dpi

---

## fMRI Processing Pipeline

These scripts were run on the Stanford Sherlock HPC cluster. Most are SLURM array jobs that process subjects in parallel. The pipeline order is:

1. Fix orientation → 2. Slice timing correction → 3. FreeSurfer recon-all → 4. fMRIPrep → 5. Motion QC → 6. GLM → 7. Feature extraction (amygdala, vmPFC, ROI activations) → 8. Persistence computation → 9. Beta-series connectivity (LSS)

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
| `extract_amygdala.sh` | SLURM array: Extract voxelwise amygdala betas from per-run GLM (Harvard-Oxford 50%) |
| `extract_amygdala_concat.sh` | SLURM array: Extract voxelwise amygdala betas from concatenated GLM |
| `extract_vmPFC.sh` | SLURM array: Extract voxelwise vmPFC betas (ant/post spheres × 3 image conditions) for vmPFC persistence |
| `extract_roi_activations.sh` | SLURM array: Extract mean beta per ROI (L/R amygdala, ant/post vmPFC) per condition for task validation |
| `combineROIActivations.py` | Combine per-subject ROI activation CSVs into group file (`all_subjects_roi_activations.csv`) |

**Persistence computation:**

| Script | Description |
|--------|-------------|
| `run_cross_corr.py` | Compute cross-run voxelwise persistence for amygdala (primary measure) |
| `run_cross_corr_concat.py` | Compute concatenated persistence (sensitivity measure) |
| `run_cross_corr_vmPFC.py` | Compute cross-run voxelwise persistence for vmPFC seeds (secondary comparison) |

**Beta-series functional connectivity (LSS — primary):**

| Script | Description |
|--------|-------------|
| `runBStaskFC_LSS.sh` | SLURM array: ROI-level LSS beta-series FC (amygdala–vmPFC, 3 conditions: neg, neu, pos) — PRIMARY |
| `combineBTS_LSS.py` | Combine per-subject LSS ROI CSVs into group file (`all_subjects_betaSeries_LSS_all_conditions_M2ID.csv`) |
| `seedbasedBStaskFC_LSS.sh` | SLURM array: Voxelwise LSS seed-based FC (outputs per-condition maps for neg, neu, pos) |
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

Data are provided by the Midlife in the United States (MIDUS) study. Users must comply with all MIDUS data use agreements and citation requirements. Original MIDUS data are not included in this repository; access may be requested at [midus.wisc.edu](https://midus.wisc.edu). Data files included in this repository contain synthetic dummy data generated to match the structure and format of the original dataset for reproducibility purposes only.

Script development and initial drafts of the Methods and Results sections were assisted by Claude Code (Anthropic), with all content verified and edited by the authors.
