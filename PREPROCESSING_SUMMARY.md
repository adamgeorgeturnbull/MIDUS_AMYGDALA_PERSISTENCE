# M3 Corrected-STC Preprocessing and Reproduction Summary

This document describes the scientific preprocessing pipeline for the MIDUS 3 (M3) corrected
slice-timing reanalysis, the provenance of all canonical fMRI outputs, QC procedures, sample
construction, and the run order for reproducing the statistical results. It replaces the prior
code-standardization document.

MR1 validation was completed as a separate pipeline after the corrected M3 freeze and is
documented under `MR1_validation/`. The September 8 freeze is a historical checkpoint. The September 23 author review corrected vmPFC persistence to six directional image-to-following-face comparisons and refreshed affected analyses. See [M3 imaging source](scripts/fMRI/README.md) for the active script set; historical workflow references below may name superseded scripts.

---

## 1. Discovery of the Slice-Order Error

The original preprocessing used `extractSliceTiming.py`, which called `np.argsort(slice_times)`
to produce a custom FSL slice-order file. NumPy argsort yields **0-based** indices (0–39 for 40
slices), but FSL `slicetimer --ocustom` requires **1-based** slice numbers (1–40). The production
file `M3_slice_order.txt` was therefore invalid.

Both fMRIPrep launch scripts used `--ignore slicetiming`, meaning fMRIPrep did not independently
apply slice-timing correction. The analysis consequently used BOLD data with an incorrectly applied
STC pass. The intended workflow — reorient, apply FSL STC manually with the corrected 1-based order,
restore orientation, then run fMRIPrep with `--ignore slicetiming` — was implemented in the corrected
rerun.

---

## 2. Corrected Rerun Provenance

The corrected rerun (`M3_stc_rerun`) began from **untouched raw BOLD data** that had not been
processed by the invalid STC workflow. The pipeline steps were:

1. `fixOrientation.py` — Fix transposed NIfTI axes in raw BOLD files
2. `M3_slice_time_correction.sh` / `slurm_M3_stc_parallel.sh` — FSL `slicetimer` with the
   corrected **1-based** interleaved order (`np.argsort(slice_times) + 1`); axis-swapped before
   STC and restored afterward
3. `slurm_fmriprep_parallel.sh` — fMRIPrep 24.1.0 via Singularity with `--ignore slicetiming`,
   so STC is applied **exactly once** (by step 2)
4. HTML visual QC of corrected fMRIPrep outputs — no new participant failed QC compared with the
   prior review
5. GLM, feature extraction, persistence, and LSS beta-series steps run on corrected derivatives

Software: FSL 6.0.7.10, FreeSurfer 7.4.1, fMRIPrep 24.1.0, Python 3.12.1, nilearn.
HPC: Stanford Sherlock cluster.

---

## 3. Quantitative FD Source

Quantitative framewise displacement comes exclusively from
**`data/fMRI/fd_summary.csv`**, which is derived from the corrected fMRIPrep confound timeseries.
The file contains **469 run records** from **158 participants** (one row per subject × run).

The Excel file `data/fMRI/task_fMRI_QC.xlsx` provides **manual anatomical and functional visual
QC ratings only**. Any FD columns present in that file are ignored by `08_process_fmri_qc.py`
because they reflect the prior (invalid) preprocessing.

---

## 4. Output Counts from the Corrected Run

| Output | Count | File |
|--------|-------|------|
| GLM runs completed | 469 runs | — |
| LSS beta-series participants | 156 | `data/fMRI/all_subjects_betaSeries_LSS_all_conditions_M2ID.csv` |
| Negative amygdala persistence rows | 468 (156 × L/R/BI) | `data/fMRI/negative_persistence_cross_run.csv` |
| Positive amygdala persistence rows | 468 (156 × L/R/BI) | `data/fMRI/positive_persistence_cross_run.csv` |
| vmPFC persistence participants | 151 | `data/fMRI/vmPFC_persistence_wide.csv` |
| ROI activation participants | 148 | `data/fMRI/all_subjects_roi_activations.csv` |
| Condition-specific FD participants | 156 | `data/fMRI/condition_fd_wide.csv` |
| FD run records / participants | 469 / 158 | `data/fMRI/fd_summary.csv` |

---

## 5. Conservative-Sample Construction

**Script:** `scripts/preprocessing/08_process_fmri_qc.py`

Inputs:
- `data/fMRI/task_fMRI_QC.xlsx` — visual QC, columns `subject`, `run1`, `run2`, `run3` only
- `data/fMRI/fd_summary.csv` — corrected quantitative FD
- `data/fMRI/negative_persistence_cross_run.csv` — n_pairs for the n_pairs=6 criterion

**Conservative sample criterion (all three required):**
1. All three visual run ratings are `Pass` (`all_runs_pass == 1`)
2. Participant mean FD < 0.5 mm across three task runs (`fd_pass == 1`)
3. Left-amygdala negative persistence n_pairs == 6 — all six directional cross-run pairs valid
   (`task_complete == 1`)

Output: `data/fMRI/fmri_qc_processed.csv`

**Script:** `scripts/preprocessing/09_merge_fmri_data.py`

Merges QC flags and corrected fMRI summary measures into the master dataset.
FC columns are NOT merged here — `analysis_utils.load_master(fc=True)` loads
`all_subjects_betaSeries_LSS_all_conditions_M2ID.csv` directly when FC data are needed.

Output: `data/processed/midus_with_fmri.csv`

---

## 6. PANAS Missing-Code Correction

**Script:** `scripts/preprocessing/06_clean_merged_data.py`

The MIDUS 3 Neuroscience codebook designates **8, 98, and 99** as missing-value codes for both
`C5SPGP` (PANAS positive affect mean) and `C5SPGN` (PANAS negative affect mean). These are
**mean item scores** with a valid range of **1–5** (not summed scores). Codes are recoded to NaN
before any distributional statistics, the log offset, or `C5SPGN_log` are computed.

Audit results:

| Variable | n_code_8 | n_code_98 | n_code_99 | valid_N | valid_range |
|----------|---------|---------|---------|---------|-------------|
| C5SPGP   | 1       | 0       | 0       | 230     | 1–5         |
| C5SPGN   | 1       | 0       | 0       | 230     | 1–5         |

Audit file: `results/tables/panas_missing_code_recode.csv`

All downstream analyses containing PANAS outcomes were regenerated after this correction.

---

## 7. Final Sample Sizes

| Sample | N | Age M (SD) | Range | Male | Female |
|--------|---|------------|-------|------|--------|
| Daily diary (`daily_diary`) | 1174 | 67.6 (10.3) | 47–94 | 501 | 673 |
| Neuroscience-age availability (`neuroscience_age`) | 231 | — | — | — | — |
| Analysis 01 neuro subsample (`diary_neuroscience_age_overlap`) | 137 | — | — | — | — |
| Valid PANAS (`neuroscience_panas`) | 230 | — | — | — | — |
| Diary + PANAS overlap (`diary_panas_overlap`) | 136 | — | — | — | — |
| Conservative fMRI — no diary (`fmri_conservative`) | **127** | 65.7 (9.5) | 48–95 | 51 | 76 |
| Diary + conservative fMRI (`diary_fmri_conservative`) | **81** | 67.2 (9.8) | 48–95 | 31 | 50 |
| Moderation complete cases (`reappraisal_complete_case`, `suppression_complete_case`) | **80** | 67.1 (9.7) | 48–95 | 31 | 49 |

**Analysis 06 and Analysis 07 have identical participant membership for each moderator** (N = 80);
no separate FC moderation complete-case rows are reported.

Race counts for conservative fMRI (N = 127, no missing race):
White 96, Black 24, Native American 1, Asian 2, Pacific Islander 0, Other 4.

Race counts for diary + conservative fMRI (N = 81, no missing race):
White 67, Black 12, Native American 1, Asian 1, Pacific Islander 0, Other 0.

Sex coding: 1 = male, 2 = female. Race is reported as race (not race/ethnicity);
reference category in analyses is White.

Full sample descriptives: `results/tables/sample_descriptives.csv`
(generated by `scripts/preprocessing/07_sample_descriptives.py`).

---

## 8. Canonical Output Files in `data/fMRI/`

The seven corrected products promoted into `data/fMRI/` after the M3_stc_rerun:

| File | Provenance |
|------|-----------|
| `all_subjects_betaSeries_LSS_all_conditions_M2ID.csv` | LSS ROI-level FC, 156 participants |
| `negative_persistence_cross_run.csv` | Amygdala negative persistence, 156 participants × 3 hemispheres |
| `positive_persistence_cross_run.csv` | Amygdala positive persistence, 156 participants × 3 hemispheres |
| `fd_summary.csv` | Per-run FD from corrected fMRIPrep confounds, 469 records |
| `vmPFC_persistence_wide.csv` | vmPFC cross-run persistence, 151 participants |
| `all_subjects_roi_activations.csv` | Mean ROI beta per condition, 148 participants |
| `condition_fd_wide.csv` | Condition-specific FD, 156 participants |

Derived from those inputs by local scripts:
- `data/fMRI/fmri_qc_processed.csv` — QC flags (from `08_process_fmri_qc.py`)
- `data/processed/midus_with_fmri.csv` — fMRI-merged master dataset (from `09_merge_fmri_data.py`)

---

## 9. Archived and Obsolete Products

Obsolete bad-STC, concatenated, seed-based, and aCompCor-derived products were moved to local
`bad_stc_0based_slice_order` archival directories and are absent from canonical `data/fMRI/`
paths. Git history preserves versions that were previously tracked. These products must not be
used as analysis inputs.

Archived files (not at canonical `data/fMRI/` paths):
- `negative_persistence_concat.csv` — concatenated operationalization (sensitivity only)
- `betaSeries_neg_vs_neu_threat_safety.csv` — superseded by LSS file
- `betaSeries_neg_vs_neu.csv` — superseded by LSS file
- `all_subjects_betaSeries_all_conditions_M2ID.csv` — LSA file (LSS preferred)
- `vmPFC_persistence_wide_aCompCor.csv` — aCompCor sensitivity, not primary
- `all_subjects_roi_activations_aCompCor.csv` — aCompCor sensitivity, not primary

---

## 10. Analysis Run Order

Run all scripts from the **project root directory**.

### Phase 1 — Behavioral preprocessing

```
python scripts/preprocessing/01_harmonize_ids.py
python scripts/preprocessing/02_construct_daily_diary_affect.py
python scripts/preprocessing/03_construct_demographics.py
python scripts/preprocessing/04_construct_covariates.py
python scripts/preprocessing/05_merge_master_dataset.py
python scripts/preprocessing/06_clean_merged_data.py
```

Outputs include `data/processed/midus_merged_clean.csv` and
`results/tables/panas_missing_code_recode.csv`.

### Phase 2 — fMRI QC and merge

```
python scripts/preprocessing/08_process_fmri_qc.py
python scripts/preprocessing/09_merge_fmri_data.py
```

Outputs include `data/fMRI/fmri_qc_processed.csv` and
`data/processed/midus_with_fmri.csv`.

### Phase 3 — Sample descriptives

```
python scripts/preprocessing/07_sample_descriptives.py
```

Output: `results/tables/sample_descriptives.csv`.

### Phase 4 — Context and validation checks

```
python scripts/analysis/00a_diary_panas_convergence.py
python scripts/analysis/00b_task_condition_differences.py
python scripts/analysis/00c_vmPFC_convergence.py
python scripts/analysis/00d_erq_context.py
```

### Phase 5 — Primary analyses

```
python scripts/analysis/01_affect_age.py
python scripts/analysis/02_persistence_affect.py
python scripts/analysis/03_persistence_age.py
python scripts/analysis/04_fc_affect.py
python scripts/analysis/05_fc_persistence.py
python scripts/analysis/06_persistence_affect_moderation.py
python scripts/analysis/07_fc_affect_moderation.py
```

### Phase 6 — Sensitivity and motion analyses

```
python scripts/analysis/01_sensitivity.py
python scripts/analysis/02_sensitivity.py
python scripts/analysis/03_sensitivity.py
python scripts/analysis/04_sensitivity.py
python scripts/analysis/04ex_fc_affect_antpost.py
python scripts/analysis/05_sensitivity.py
python scripts/analysis/05ex_fc_persistence_antpost.py
python scripts/analysis/06_sensitivity.py
python scripts/analysis/06b_persistence_affect_moderation_mlm.py
python scripts/analysis/07_sensitivity.py
python scripts/analysis/07b_fc_affect_moderation_mlm.py
python scripts/analysis/00e_motion_check.py
```

### Phase 7 — Supplementary tables and figures

```
python scripts/visualization/supplementary_tables.py
python scripts/visualization/supplementary_tables_docx.py
python scripts/visualization/publication_figures.py
python scripts/preprocessing/participant_flowchart.py
```

---

## 11. Completed Verification

- Corrected fMRIPrep HTML outputs underwent visual QC; no new exclusions vs. prior review.
- Quantitative FD sourced exclusively from `fd_summary.csv` (corrected fMRIPrep confounds).
- `08_process_fmri_qc.py` reads only the `subject`, `run1`, `run2`, `run3` columns from the
  Excel file; any Excel FD columns are discarded before merging.
- PANAS missing codes 8, 98, and 99 recoded before any distributional statistics or log
  transformation. Audit saved to `results/tables/panas_missing_code_recode.csv`.
- Conservative sample defined with three required criteria (visual QC + FD + n_pairs=6);
  `flagged` column in fd_summary.csv is not used.
- Fisher-z transformation applied consistently to both amygdala and vmPFC persistence
  predictors via `prepare_persistence_vars()`.
- Supplementary table now reports separate N columns (`N correlation`, `N OLS`, `N MLM`)
  populated from each method's own result file.
- Motion analyses (00e) confirm no significant mixed-effects association between head motion and
  the primary persistence or FC metrics (all two-tailed *p*s ≥ .129). A September 9 formula-name
  correction allowed the four FC-as-outcome MLMs to run; all converged and were nonsignificant.
  A September 10 coverage correction aligned Part 2 with the significant primary MLM findings.
  Persistence with raw NA, age with persistence, and anterior connectivity with raw and
  log-transformed NA all remained significant when controlling separately for both motion
  measures (one-tailed *p*s = .038/.026, .019/.019, .048/.040, and .018/.015, respectively).

---

## 12. Remaining Manuscript-Preparation Work

The following items are manuscript and figure changes; no further computational correction
or analysis rerun is required for the frozen scope. Corrected Methods and Results prose has
been drafted and independently checked against the aggregate CSV outputs. It still requires
integration into the Word manuscript and manual author review.

- Replace N = 128 with N = 127 (conservative fMRI) and N = 80 with N = 81 (diary+fMRI)
  throughout the manuscript
- Correct sex coding from "0 = male, 1 = female" to "1 = male, 2 = female"
- Replace "race/ethnicity" with "race" or "race dummy variables (reference = White)"
- Update sample demographics table with N = 127 values (including race counts)
- Revise Figure 1 persistence definition (negative-image → neutral-face cross-run;
  condition-level GLM betas, not trial-level; averaged in Fisher-z space)
- Add LSS event duration (6 s = 4-s image + 2-s fixation) to Methods
- Correct task-activation language: left-amygdala neutral activation does not differ
  significantly from zero (*p* = .070), whereas negative and positive activation do; read
  corrected values from
  `results/tables/00b_task_conditions/roi_one_sample.csv`
- Update moderation prose: the reappraisal × anterior FC interaction did not reproduce in
  the corrected conservative analysis; treat moderation as exploratory
- Update simple slopes if interaction findings change
- Temper "independent" and "specific" language per independence-testing requirements
- Add clarification that residualized scatterplots are descriptive; inference from MLM
- If normalized/resampled voxel dimensions are retained in the manuscript, verify them against
  the analyzed images before submission; the replacement Methods draft does not make that claim

See `PUBLICATION_REQUIRED_CHANGES.md` for the full item-by-item tracking record.
