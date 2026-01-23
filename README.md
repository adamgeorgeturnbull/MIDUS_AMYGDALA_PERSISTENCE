# MIDUS Amygdala Persistence

This repository contains analysis code for a preregistered study using the MIDUS 3 Neuroscience dataset examining age-related differences in affect, amygdala persistence to emotional stimuli, and model-free emotion regulation mechanisms.

The study combines daily diary data, survey-based demographics, and neuroimaging measures to test confirmatory hypotheses derived from prior work (e.g., Puccetti et al., 2021), alongside preregistered exploratory analyses of task-based functional connectivity.

Preregistration (updated Oct 15, 2025): OSF

---

## Study Goals

### Replication (confirmatory)
1. Replicate age-related decreases in daily life negative affect and increases in positive affect.
2. Replicate associations between amygdala persistence to negative images and daily life affect.

### Extensions (confirmatory)
1. Test whether amygdala persistence to negative images decreases with age.
2. Test whether age-related differences in affect are mediated by amygdala persistence.

### Novel analyses (exploratory)
1. Examine model-free emotion regulation indexed by task-based amygdala–vmPFC/sgACC connectivity using beta-series modeling.
2. Test whether model-free emotion regulation relates to amygdala persistence and daily life affect.
3. Examine moderation by emotion regulation strategy use (reappraisal, suppression).
4. Test age-related differences in these effects, including analyses restricted to older adults.

---

## Project Structure

# MIDUS Amygdala Persistence

This repository contains analysis code for a preregistered study using the MIDUS 3 Neuroscience dataset examining age-related differences in affect, amygdala persistence to emotional stimuli, and model-free emotion regulation mechanisms.

The study combines daily diary data, survey-based demographics, and neuroimaging measures to test confirmatory hypotheses derived from prior work (e.g., Puccetti et al., 2021), alongside preregistered exploratory analyses of task-based functional connectivity.

Preregistration (updated Oct 15, 2025): OSF

---

## Study Goals

### Replication (confirmatory)
1. Replicate age-related decreases in daily life negative affect and increases in positive affect.
2. Replicate associations between amygdala persistence to negative images and daily life affect.

### Extensions (confirmatory)
1. Test whether amygdala persistence to negative images decreases with age.
2. Test whether age-related differences in affect are mediated by amygdala persistence.

### Novel analyses (exploratory)
1. Examine model-free emotion regulation indexed by task-based amygdala–vmPFC/sgACC connectivity using beta-series modeling.
2. Test whether model-free emotion regulation relates to amygdala persistence and daily life affect.
3. Examine moderation by emotion regulation strategy use (reappraisal, suppression).
4. Test age-related differences in these effects, including analyses restricted to older adults.

---

## Project Structure
MIDUS_AMYGDALA_PERSISTENCE/
├─ data/
│  ├─ raw/
│  │  ├─ M3P5_variables.csv
│  │  ├─ M3P2_variables.csv
│  │  ├─ MKE2_variables.csv
│  │  └─ README.md
│  ├─ processed/
│  │  ├─ daily_diary_processed.csv
│  │  ├─ m3p5_ids.csv
│  │  ├─ mke2_ids.csv
│  │  ├─ m3p5_demos.csv
│  │  ├─ mke2_demos.csv
│  │  ├─ m3p5_covariates.csv
│  │  ├─ mke2_covariates.csv
│  │  ├─ midus_merged.csv
│  │  ├─ midus_merged_clean.csv
│  │  ├─ daily_diary_descriptives.csv
│  │  └─ README.md
├─ results/
│  ├─ tables/
│  │  └─ sample_descriptives.csv
│  └─ figures/
├─ scripts/
│  ├─ preprocessing/
│  │  ├─ 01_harmonize_ids.py
│  │  ├─ 02_construct_daily_diary_affect.py
│  │  ├─ 03_construct_demographics.py
│  │  ├─ 04_construct_covariates.py
│  │  ├─ 05_merge_master_dataset.py
│  │  ├─ 06_clean_merged_data.py
│  │  └─ 07_sample_descriptives.py
│  └─ fMRI/
│     ├─ preprocessing/
│     └─ analysis/
├─ README.md
├─ requirements.txt


Raw and processed data are not tracked by git and must be obtained through authorized MIDUS access.

---

## Preprocessing Step 1: Data Download and Collation

### Data Source

Data are obtained from the MIDUS repository via:
https://midus.colectica.org

Neuroimaging data require a Data Usage Agreement and are accessed separately via ICPSR (Study 38862).

---

## Raw Data Structure

The preprocessing pipeline begins from three non-overlapping raw datasets, each corresponding to a distinct MIDUS 2 project. Each dataset is treated as authoritative for its domain, eliminating redundancy across files.

---

## Raw Dataset 1: MIDUS 2 Project 5 (Neuroscience and Survey Demographics)

File: M2P5_variables.csv

This dataset contains all neuroscience project variables as well as demographic and survey variables drawn from MIDUS Project 1.

Participant identifiers:
- MIDUSID
- M2ID
- SAMPLMAJ
- M2FAMNUM

Demographics:
- Sex (C1PRSEX)
- Education (C1PB1)
- Race (C1PF7A)
- Ethnicity (C1PF1)
- Birth year (C1PBYEAR)
- Age at MIDUS 3 baseline (C1PRAGE)

Interview timing:
- Baseline interview month/year (C1PIDATE_MO, C1PIDATE_YR)
- Neuroscience visit month/year (C5PDATE_MO, C5PDATE_YR)

Neuroscience-related variables:
- MRI completion indicator (C5IC)
- Age at neuroscience visit (C5PAGE)
- Handedness (C5HAND)
- Socioeconomic indicators (C5SER, C5SES)
- PANAS affect variables (C5SPGP, C5SPGN)

This dataset serves as the primary source of demographics and neuroscience participation variables.

---

## Raw Dataset 2: MIDUS 2 Project 2 (Daily Diary Affect)

File: M2P2_variables.csv

This dataset contains daily diary affect data collected over an 8-day period.

Variables include:
- Daily affect items C2DC1 through C2DC27
- Diary day number (C2DDAY)
- Diary interview month/year (C2DIMON, C2DIYEAR)
- Participant identifier (M2ID)

This dataset is used to compute daily diary-based affect measures, including mean positive affect, mean negative affect, and persistence-related summaries.

---

## Raw Dataset 3: MIDUS 2 Milwaukee Sample Demographics

File: MKE2_variables.csv

This dataset contains demographic information for participants in the Milwaukee oversample.

Participant identifiers:
- M2ID
- SAMPLMAJ

Demographics:
- Education (CACB1)
- Race (CACF7A)
- Ethnicity (CACF1)
- Sex (CACRSEX)
- Age at interview (CACRAGE)

Interview timing:
- Interview month/year (CACIDATE_MO, CACIDATE_YR)

These variables are used to supplement demographic information when not available from Project 5.

---

## Behavioral Preprocessing Scripts

### 01_harmonize_ids.py

Ensures participant IDs are consistent across all three raw datasets.

Tasks performed:
- Standardizes ID formats.
- Verifies uniqueness of M2ID within each dataset.
- Logs missing or mismatched IDs across projects.

Inputs:
- data/raw/M2P5_variables.csv
- data/raw/M2P2_variables.csv
- data/raw/MKE2_variables.csv

Outputs:
- ID-consistent datasets saved to data/processed

---

### 02_construct_daily_diary_affect.py

Processes daily diary affect data to generate participant-level summary measures.

Tasks performed:
- Recodes invalid responses to missing values.
- Computes participant-level mean positive affect and negative affect.
- Computes persistence-related affect measures.
- Counts number of diary days completed per participant.
- Generates descriptive statistics for scale reliability and distributions.

Input:
- data/raw/M2P2_variables.csv

Outputs:
- data/processed/daily_diary_processed.csv
- data/processed/daily_diary_descriptives.csv

---

### 03_construct_demographics.py

Constructs cleaned, harmonized demographics for the MIDUS neuroscience (M3P5) and Milwaukee (MKE2) datasets independently.

Key points:

- Preserves all original variables from each dataset.
- Adds harmonized demographic columns:
  - sex
  - educ (highest education)
  - ethnicity
  - race
- Converts MIDUS missing codes to NA.
- Keeps one row per participant.
- Does not merge the datasets—demographics are cleaned separately for M3P5 and MKE2.

Inputs:
- data/processed/m3p5_ids.csv
- data/processed/mke2_ids.csv

Outputs:
- data/processed/m3p5_demos.csv
- data/processed/mke2_demos.csv

---

### 04_construct_covariates.py

Generates analysis-ready covariates for each dataset from the cleaned demographics files.

Tasks performed:

- Race dummy coding: Creates dummy variables for each non-reference race category (reference = White) while keeping coding consistent across datasets.
- Twin pair dummy coding (M3P5 only): Creates a dummy variable for each twin pair using M2FAMNUM (applies only to participants in the twin sample, SAMPLMAJ = 3).  
  - MKE2 participants do not have twin pairs; all twin dummies are 0.

Inputs:
- data/processed/m3p5_demos.csv
- data/processed/mke2_demos.csv

Outputs:
- data/processed/m3p5_covariates.csv
- data/processed/mke2_covariates.csv

---

### 05_merge_master_dataset.py

Merges multiple processed MIDUS datasets into a single master analysis file keyed on M2ID.

Inputs:
- daily_diary_processed.csv: participant-level daily diary affect summaries
- m3p5_covariates.csv: P5 neuroscience demographics and covariates
- mke2_covariates.csv: MKE2 demographics and covariates

Processing steps:
1. Merge all datasets on M2ID, keeping all variables.
2. For duplicate variables across datasets, combine into a single harmonized column.
3. Add 0s for all twin_pair_ variables for participants from MKE2.
4. Compute new variables:
   - Age at P2 (C2PAGE):
     - M3 participants: C2PAGE = StartYear (from diary) - C1PBYEAR
     - MKE2 participants: estimated from CACRAGE and baseline interview date (CACIDATE_MO, CACIDATE_YR) relative to diary start (StartMonth, StartYear)
   - Time between P2 and P5 (time_P2_P5): in months, calculated from diary start (StartMonth, StartYear) and P5 visit date (C5PDATE_MO, C5PDATE_YR)

Output:
- midus_merged.csv: cleaned master dataset with harmonized variables, age calculations, twin-pair indicators, and P2–P5 intervals ready for analysis

---

### 06_clean_merged_data.py

Performs additional cleaning of the master dataset to prepare for analysis.

Processing steps:
1. Ensures twin_pair_ variables are set to 0 for participants from MKE2 who do not have twins.
2. Harmonizes demographic variables and ensures missing codes are set to NA.
3. Computes derived variables needed for analysis:
   - Age at P2 (C2PAGE)
   - Time between P2 and P5 (time_P2_P5) in months
   - Kurtosis and skewness for affect variables (C5SPGP, C5SPGN)
   - Log-transformed negative affect (C5SPGN_log)
4. Verifies consistency of variable types and removes any redundant columns if present.

Input:
- midus_merged.csv

Output:
- midus_merged_clean.csv

### 07_sample_descriptives.py

Generates publication-ready sample descriptives for 5 MIDUS samples.

Samples included:
1. daily diary full: all participants with diary data (StartYear present)
2. neuroscience full: all participants with P5 neuroscience data (C5PDATE_YR present)
3. daily diary + neuroscience overlap: participants with both diary and neuroscience data
4. neuroimaging sample: participants who completed neuroimaging (C5IC = 1)
5. imaging + diary overlap: participants with both completed imaging and diary data

Descriptives computed for each sample:
- Age: mean ± SD, range
  - C2PAGE used for diary-only sample
  - C5PAGE used for all other samples
- Sex: % Female
- Education: mean ± SD (harmonized 1–12 coding)
- Ethnicity: % Hispanic
- Race: % White / Black / Native American / Asian / Pacific Islander / Other

Output:
- results/tables/sample_descriptives.csv: publication-ready table with all sample characteristics

---

### 08_merge_fmri_data.py

Merge fMRI-derived participant-level measures into the cleaned MIDUS master dataset.

Processing steps:
1. Load cleaned master dataset from 06_clean_merged_data.py.
2. Load fMRI-derived CSV files:
   - betaSeries_neg_vs_neu.csv
   - positive_persistence_cross_run.csv
   - negative_persistence_cross_run.csv
   - negative_persistence_concat.csv
   - fd_summary.csv
3. Pivot multi-row fMRI data (by hemisphere) to wide format so each participant has a single row.
4. Merge all fMRI data with the master dataset using left joins on M2ID.
5. Create availability flags for each modality:
   - has_beta_series
   - has_neg_persistence
   - has_pos_persistence
   - has_fd_data
   - has_imaging_data
6. No participant exclusions are applied.

Input:
- midus_merged_clean.csv
- fMRI-derived CSV files in data/fMRI/

Output:
- data/processed/midus_with_fmri.csv


## Behavioral Analysis Scripts

### 01_affect_age_replication.py

Replication of age-related differences in affect in MIDUS 3.

Processing steps:
1. Load processed master dataset with fMRI data (midus_with_fmri.csv).
2. Define affect outcomes:
   - Daily diary: PA_score, NA_score, NA_score_log
   - PANAS (neuro sample only): C5SPGP (positive), C5SPGN (negative), C5SPGN_log
3. Define two analysis samples:
   - Daily diary sample: participants with daily diary data
       - Outcome: daily diary affect (PA_score, NA_score, NA_score_log)
       - Predictor: age at P2 (C2PAGE)
   - Neuro sample: participants with fMRI and/or PANAS data
       - Outcomes: daily diary affect (PA_score, NA_score, NA_score_log) and PANAS scores (C5SPGP, C5SPGN, C5SPGN_log)
       - Predictors:
           - Daily diary affect: age at P2 (C2PAGE) and age at P5 (C5PAGE)
           - PANAS scores: age at P5 (C5PAGE)
4. Covariates included in regressions: sex, educ, race dummies, twin dummies.
5. Compute zero-order correlations and linear regressions for each outcome × age variable combination.
6. Save output tables:
   - results/tables/01_affect_age_correlations.csv
   - results/tables/01_affect_age_regressions.csv

Notes:
- All analyses use complete cases for the specific outcome and covariates.
- Log-transformed negative affect variables are included to normalize distributions.


## Citation and Acknowledgment

Data are provided by the Midlife in the United States (MIDUS) study. Users must comply with all MIDUS data use agreements and citation requirements.
