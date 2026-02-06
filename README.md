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

### Novel Analyses (Exploratory)
1. Examine model-free emotion regulation indexed by task-based amygdala–vmPFC connectivity using beta-series modeling
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
│   │   ├── midus_with_fmri.csv
│   │   └── README.md
│   └── fMRI/
│       ├── betaSeries_neg_vs_neu.csv
│       ├── negative_persistence_concat.csv
│       ├── negative_persistence_cross_run.csv
│       ├── positive_persistence_cross_run.csv
│       └── fd_summary.csv
├── results/
│   ├── tables/
│   │   ├── sample_descriptives.csv
│   │   ├── panas_skew_kurtosis.csv
│   │   ├── 01_affect_age_correlations.csv
│   │   ├── 01_affect_age_regressions.csv
│   │   ├── 02_persistence_affect_correlations_full.csv
│   │   ├── 02_persistence_affect_correlations_conservative.csv
│   │   ├── 02_persistence_affect_regressions_full.csv
│   │   ├── 02_persistence_affect_regressions_conservative.csv
│   │   └── 02_persistence_affect_summary.txt
│   └── figures/
│       ├── analysis_01/
│       └── analysis_02/
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
│   │   ├── 01_affect_age_replication.py
│   │   ├── 02_persistence_affect.py
│   │   └── 02_persistence_affect_summary.py
│   ├── fMRI/
│   │   ├── preprocessing/
│   │   │   ├── extractSliceTiming.py
│   │   │   ├── fixOrientation.py
│   │   │   ├── getMotion.py
│   │   │   ├── getMotion.sh
│   │   │   ├── M3_slice_time_correction.sh
│   │   │   ├── slurm_M3_stc_parallel.sh
│   │   │   ├── slurm_fmriprep_parallel.sh
│   │   │   ├── slurm_fmriprep_parallel_pre_fs.sh
│   │   │   └── slurm_recon_all_parallel.sh
│   │   └── analysis/
│   │       ├── runGLM.sh
│   │       ├── runGLM_concat.sh
│   │       ├── extract_amygdala.sh
│   │       ├── extract_amygdala_concat.sh
│   │       ├── run_cross_corr.py
│   │       ├── run_cross_corr_concat.py
│   │       ├── combineBTS.py
│   │       ├── runBStaskFC.sh
│   │       ├── seedbasedBStaskFC.sh
│   │       └── grouplevelSeedBasedFC.sh
│   ├── visualization/
│   │   ├── 01_affect_age_figures.py
│   │   └── 02_persistence_affect_figures.py
│   ├── generate_dummy_data.py
│   ├── generate_dummy_fmri_data.py
│   └── compare_outputs.py
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
- `C5SER`, `C5SES` - Socioeconomic indicators
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
- **fMRI data files:**
  - `data/fMRI/betaSeries_neg_vs_neu.csv` - Beta-series connectivity (1 row/participant)
  - `data/fMRI/negative_persistence_concat.csv` - Negative persistence concatenated (3 rows/participant)
  - `data/fMRI/negative_persistence_cross_run.csv` - Negative persistence cross-run (3 rows/participant)
  - `data/fMRI/positive_persistence_cross_run.csv` - Positive persistence cross-run (3 rows/participant)
  - `data/fMRI/fd_summary.csv` - Framewise displacement (multiple rows/participant)
  - `data/fMRI/fmri_qc_processed.csv` - Quality control flags (1 row/participant)

**Processing Steps:**
1. Load cleaned master dataset
2. Load and process fMRI files:
   - Rename beta-series connectivity columns (e.g., `l_amyg-ant_vmPFC` → `conn_l_amyg_ant_vmPFC_neg_vs_neu`)
   - Pivot hemisphere-wise persistence data from long to wide format
   - Aggregate framewise displacement across runs
3. Merge all fMRI data with master dataset using left joins on M2ID
4. Merge QC flags from processed QC file
5. Create availability flags:
   - `has_beta_series` - Has beta-series connectivity data
   - `has_neg_persistence` - Has negative persistence data
   - `has_pos_persistence` - Has positive persistence data
   - `has_fd_data` - Has framewise displacement data
   - `has_imaging_data` - Has any imaging data
6. No participant exclusions applied (QC flags used for sample definition in analyses)

**Output:**
- `data/processed/midus_with_fmri.csv`

---

## Analysis Scripts

### Analysis 01: Affect-Age Replication

**File:** `scripts/analysis/01_affect_age_replication.py`

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
  - Cross-run negative persistence (L, R, bilateral)
- **Sensitivity:**
  - Cross-run positive persistence (L, R, bilateral)
  - Concatenated negative persistence (L, R, bilateral)

**Affect Outcomes:**
- **Primary:** Daily diary PA, NA, and NA_log
- **Secondary:** PANAS (C5SPGP, C5SPGN, C5SPGN_log)

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
- Primary analyses use cross-run negative persistence (replication)
- Conservative sample is primary per preregistration
- Full sample results are exploratory if conservative sample unavailable

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
- **Primary:** Cross-run negative persistence (L, R, bilateral)
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

### Analysis 02: Sensitivity — Mixed-Effects Models

**File:** `scripts/analysis/02_persistence_affect_mlm.py`

Sensitivity reanalysis replacing OLS regressions (with twin pair dummies) with linear mixed-effects models. This avoids the degrees-of-freedom cost of one dummy variable per twin pair while properly accounting for non-independence within twin families via a random intercept.

**Input:**
- `data/processed/midus_with_fmri.csv`

**Grouping Variable (`family_id`):**
- Twins (`SAMPLMAJ == 3` AND 2+ members share `M2FAMNUM`): grouped by `M2FAMNUM`
- Everyone else: own cluster (`M2ID`)

**Model:**
- Fixed effects: `affect ~ persistence + C5PAGE + sex + race dummies + time_P2_P5 + n_days_complete`
- Random effects: random intercept for `family_id`
- Estimation: REML (optimizer: LBFGS with Powell fallback)

**Outputs:**
- `results/tables/02_persistence_affect_mlm_full.csv`
- `results/tables/02_persistence_affect_mlm_conservative.csv`

**Notes:**
- Correlations are identical to `02_persistence_affect.py` and are not re-run
- Only the covariate-adjusted models differ (MLM vs OLS)
- Zero-variance covariates (e.g., race dummies with no cases) are automatically dropped

---

### Analysis 02: Summary

**File:** `scripts/analysis/02_persistence_affect_summary.py`

Summarizes persistence × affect results, focusing on consistency across analytical methods.

**Inputs:**
- `results/tables/02_persistence_affect_correlations_full.csv`
- `results/tables/02_persistence_affect_regressions_full.csv`
- `results/tables/02_persistence_affect_correlations_conservative.csv`
- `results/tables/02_persistence_affect_regressions_conservative.csv`
- `results/tables/02_persistence_affect_mlm_full.csv` (optional)
- `results/tables/02_persistence_affect_mlm_conservative.csv` (optional)

**Summary Approach:**
1. Identifies findings significant in BOTH correlations AND regressions (p < 0.05)
2. Reports findings significant in only ONE method (with non-significant pair for comparison)
3. Categorizes results as primary (confirmatory) vs sensitivity analyses
4. Compares OLS vs MLM regression results (agreement rate and disagreements)

**Output:**
- `results/tables/02_persistence_affect_summary.txt` - Human-readable summary

**Structure:**
- Sample status (conservative vs full)
- Analysis categories breakdown
- Consistent findings (significant in both methods)
- Single-method findings (significant in one method only)
- OLS vs MLM comparison (if MLM results available)

---

## Visualization Scripts

### Visualization 01: Affect-Age Figures

**File:** `scripts/visualization/01_affect_age_figures.py`

Generates publication-ready figures for the affect-age replication analysis.

**Inputs:**
- `data/processed/midus_merged_clean.csv` (for scatterplots)
- `results/tables/01_affect_age_regressions.csv` (for coefficient plot)

**Figures Created:**
1. **Scatterplots:** Age × affect relationships with regression lines
2. **Coefficient plot (forest plot):** Summarizes all regression results

**Outputs:**
- `results/figures/analysis_01/01_affect_age_scatterplots.pdf`
- `results/figures/analysis_01/01_affect_age_coefficients.pdf`

**Figure Settings:**
- Format: PDF
- DPI: 300
- Color palette distinguishes daily diary vs neuroscience samples

---

### Visualization 02: Persistence-Affect Figures

**File:** `scripts/visualization/02_persistence_affect_figures.py`

Creates publication-quality figures for persistence × affect associations.

**Inputs:**
- `data/processed/midus_with_fmri.csv` (for scatterplots)
- `results/tables/02_persistence_affect_regressions_full.csv`
- `results/tables/02_persistence_affect_regressions_conservative.csv`

**Figures Created:**
1. **Scatterplots:** Individual plots for each persistence × affect relationship
2. **Forest plots:** Regression coefficients (primary vs sensitivity analyses)

**Outputs:**
- `results/figures/analysis_02/primary/` - Primary analysis figures
- `results/figures/analysis_02/sensitivity/` - Sensitivity analysis figures

**Figure Organization:**
- Separate subdirectories for full sample vs conservative sample
- Primary analyses (cross-run negative persistence) vs sensitivity analyses
- Color coding distinguishes persistence types and samples

**Figure Settings:**
- Format: PNG
- DPI: 300
- Organized by analysis type (primary/sensitivity) and persistence measure

---

## fMRI Processing Pipeline

These scripts were run on the Stanford Sherlock HPC cluster. Most are SLURM array jobs that process subjects in parallel. The pipeline order is:

1. Fix orientation → 2. Slice timing correction → 3. FreeSurfer recon-all → 4. fMRIPrep → 5. Motion QC → 6. GLM → 7. Amygdala extraction → 8. Persistence computation → 9. Beta-series connectivity

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
| `runGLM.sh` | SLURM array: Per-run first-level GLM (nilearn, 6 conditions, 24 motion params) |
| `runGLM_concat.sh` | SLURM array: Concatenated all-runs GLM (sensitivity analysis) |
| `extract_amygdala.sh` | SLURM array: Extract voxelwise amygdala betas from per-run GLM (Harvard-Oxford 50%) |
| `extract_amygdala_concat.sh` | SLURM array: Extract voxelwise amygdala betas from concatenated GLM |

**Persistence computation:**

| Script | Description |
|--------|-------------|
| `run_cross_corr.py` | Compute cross-run voxelwise persistence (primary measure) |
| `run_cross_corr_concat.py` | Compute concatenated persistence (sensitivity measure) |

**Beta-series functional connectivity:**

| Script | Description |
|--------|-------------|
| `runBStaskFC.sh` | SLURM array: ROI-level beta-series connectivity (amygdala–anterior/posterior vmPFC, neg vs neu; Tashjian et al., 2021 TICS) |
| `combineBTS.py` | Combine per-subject beta-series ROI CSVs into group file |
| `seedbasedBStaskFC.sh` | SLURM array: Voxelwise seed-based beta-series FC (whole-brain maps) |
| `grouplevelSeedBasedFC.sh` | Group-level second-level analysis of seed-based FC maps |

**Common GLM parameters across scripts:**
- TR = 2.0s, HRF = Glover, drift = cosine (1/128 Hz), noise = AR(1)
- 24 motion regressors (6 params + temporal derivatives + quadratic terms)
- First 4 dummy scans removed
- Amygdala mask: Harvard-Oxford atlas, 50% probability threshold, 2mm

---

## Citation and Acknowledgment

Data are provided by the Midlife in the United States (MIDUS) study. Users must comply with all MIDUS data use agreements and citation requirements.
