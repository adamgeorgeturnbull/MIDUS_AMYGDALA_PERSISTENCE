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
│ ├─ raw/
│ │ ├─ M2P5_variables.csv
│ │ ├─ M2P2_variables.csv
│ │ ├─ MKE2_variables.csv
│ │ └─ README.md
│ ├─ processed/
│ │ ├─ daily_diary_processed.csv
│ │ ├─ demographics_processed.csv
│ │ ├─ covariates_processed.csv
│ │ ├─ combined_data.csv
│ │ └─ combined_data_filtered.csv
├─ results/
│ ├─ tables/
│ └─ figures/
├─ scripts/
│ ├─ preprocessing/
│ │ ├─ 01_harmonize_ids.py
│ │ ├─ 02_construct_daily_diary_affect.py
│ │ ├─ 03_construct_demographics.py
│ │ ├─ 04_construct_covariates.py
│ │ ├─ 05_merge_master_dataset.py
│ │ └─ 06_sample_descriptives.py
│ └─ fMRI/
│ ├─ preprocessing/
│ └─ analysis/
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

Merges daily diary summaries, demographics, covariates, and neuroscience variables into a master analysis dataset keyed on M2ID.

---

### 06_sample_descriptives.py

Generates final sample descriptives for the manuscript, including demographic distributions, affect means, and completion rates.

---

## Citation and Acknowledgment

Data are provided by the Midlife in the United States (MIDUS) study. Users must comply with all MIDUS data use agreements and citation requirements.
