# MIDUS Amygdala Persistence

This repository contains analysis code for a preregistered study using the MIDUS 3 Neuroscience dataset examining age-related differences in affect, amygdala persistence to emotional stimuli, and model-free emotion regulation mechanisms.

The study combines behavioral, daily diary, and neuroimaging data to test confirmatory hypotheses derived from prior work (e.g., Puccetti et al., 2021), alongside preregistered exploratory analyses of task-based functional connectivity.

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

MIDUS_AMYGDALA_PERSISTANCE/
├─ data/
│  ├─ raw/
│  │  ├─ M3P2_variables.csv
│  │  ├─ M3P5_variables_and_demos.csv
│  │  └─ README.md  # detailed variable coding
│  ├─ processed/
│  │  ├─ m3p2_ids.csv
│  │  ├─ m3p5_ids.csv
│  │  ├─ demos_ids.csv
│  │  ├─ daily_diary_processed.csv
│  │  ├─ demographics_processed.csv
│  │  ├─ covariates_processed.csv
│  │  ├─ combined_data.csv
│  │  └─ combined_data_filtered.csv
├─ results/
│  ├─ tables/
│  └─ figures/
├─ scripts/
│  ├─ preprocessing/  # contains Steps 01–06
│  │  ├─ 01_harmonize_ids.py
│  │  ├─ 02_construct_daily_diary_affect.py
│  │  ├─ 03_construct_demographics.py
│  │  ├─ 04_construct_covariates.py
│  │  ├─ 05_merge_master_dataset.py
│  │  └─ 06_sample_descriptives.py
│  └─ fMRI/
│     ├─ preprocessing/  # all fMRI preprocessing scripts you run on cluster
│     └─ analysis/       # all fMRI analysis scripts you run on cluster
├─ README.md  # high-level project overview
├─ requirements.txt

Raw and processed data are not tracked by git and must be obtained through authorized MIDUS access.

---

## Preprocessing Step 1: Data Download and Collation

### Data Source

Data are obtained from the MIDUS repository via:
https://midus.colectica.org

Neuroimaging data require a Data Usage Agreement and are accessed separately via ICPSR (Study 38862).

### Behavioral and Survey Data Downloads

The first preprocessing step involves downloading and collating behavioral, affective, and demographic variables from multiple MIDUS 3 projects using Colectica baskets.

#### Basket 1: MIDUS 3 Project 2 (Daily Diary Affect)

File: `M3P2_variables.csv`

This basket includes daily diary affect items assessed over an 8-day period, along with demographic identifiers. Variables include:

- Daily negative affect items (e.g., restless, nervous, sad, hopeless, angry)
- Daily positive affect items (e.g., cheerful, calm, satisfied, enthusiastic, confident)
- Interview date variables (day, month, year)
- Respondent identifiers (MIDUS 2 ID, family number)
- Demographics (age at MIDUS 3, sex, major sample ID)
- Repetitive thought / rumination frequency items (C2DY1–C2DY6)

These variables are used to compute mean daily positive and negative affect scores and exploratory measures of repetitive negative thinking.

#### Basket 2: MIDUS 3 Neuroscience (Project 5) and Survey Data

File: `M3P5_variables_and_demos.csv`

This basket includes demographic, affect, emotion regulation, and neuroscience participation variables drawn from:
- MIDUS 3 Project 1 (Survey)
- MIDUS 3 Project 5 (Neuroscience)
- MIDUS 3 Milwaukee (MKE2)

Key variables include:
- Age at neuroscience visit
- Sex, education, race, ethnicity
- Handedness
- MRI completion status
- General positive and negative affect (PANAS-style)
- Emotion regulation strategy use (ERQ reappraisal and suppression)
- Interview timing variables

## Behavioral Preprocessing Scripts (scripts/preprocessing)

### 01_harmonize_ids.py
This script ensures that participant IDs are consistent across datasets.  

Main tasks:
- Checks and standardizes ID formats (e.g., numeric vs. string).  
- Validates that all expected participants are present in each dataset.  
- Logs data types and any issues with duplicate or missing IDs.  

Input:  
- Raw neuroscience dataset (data/raw/neuroscience_raw.csv)  
- Raw daily diary dataset (data/raw/M3P2_variables.csv)  

Output:  
- Harmonized datasets saved in the data/processed/ folder (for downstream merging).  
- Log file summarizing any warnings or ID inconsistencies.  

---

### 02_construct_daily_diary_affect.py
This script processes the MIDUS daily diary data to generate participant-level and summary scores.  

Main tasks:
- Recodes invalid responses to missing values.  
- Calculates participant-level mean Positive Affect (PA), Negative Affect (NA), and Persistent Thought (PT) scores.  
- Computes log-transformed NA and PT scores to reduce skew/kurtosis.  
- Saves participant-level averages for all raw items and computed summary scores.  
- Counts number of days completed per participant (both any response and fully complete days).  
- Generates descriptive statistics for the dataset, including:  
  - Number of participants before and after filtering  
  - Cronbach’s alpha for PA and NA scales  
  - Skewness and kurtosis for all summary variables  
  - Mean and standard deviation for daily diary items and summary scores  

Input:  
- Raw daily diary dataset (data/raw/M3P2_variables.csv)  

Output:  
- Processed participant-level dataset: data/processed/daily_diary_processed.csv  
- Descriptive statistics for the paper: data/processed/daily_diary_descriptives.csv  

### 03_construct_demographics.py

This script creates a clean demographics file by merging MIDUS Project 2 (daily diary) and Project 5 (neuroscience) participant data.

Inputs:
- data/processed/m3p2_ids.csv – P2 participant-level daily diary identifiers and demographics
- data/processed/m3p5_ids.csv – P5 participant-level neuroscience identifiers and demographics

Outputs:
- data/processed/demographics_processed.csv – unified demographics for all participants, with:
  - educ – highest education level
  - ethnicity – Hispanic/Latino status
  - race – primary racial origin
  - sex – participant sex
  - C2PAGE – estimated age at daily diary collection
  - All original P2 and P5 variables preserved, with overlapping variables suffixed _p2 or _p5

Key steps performed:
1. Read processed P2 and P5 datasets.
2. Rename overlapping columns (_p2 and _p5).
3. Merge datasets on M2ID.
4. Create unified variables for education, ethnicity, race, and sex.
5. Compute age at daily diary (C2PAGE) using baseline age and interview date.
6. Preserve all other variables for later analyses.

## Citation and Acknowledgment

Data are provided by the Midlife in the United States (MIDUS) study. Users must comply with all MIDUS data use agreements and citation requirements.
