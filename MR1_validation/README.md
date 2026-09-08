# MR1 Validation — MIDUS Refresher replication

**Status (September 8, 2026): complete and computationally frozen for manuscript
preparation.** The protected-data preprocessing sequence, main analyses, and predefined
sensitivity families were run successfully. The active result directories contain only the
final OLS/correlation outputs and methods notes; obsolete pre-final MLM outputs remain only
in the explicitly labelled local archive.

Targeted replication of key findings from the frozen corrected MIDUS 3 (M3)
analysis, using the independent **MIDUS Refresher (MR1)** neuroimaging cohort
(Refresher Project 5 imaging session, `MR_P5`).

The frozen scope and interpretation rules are in
[`MR1_REPLICATION_PLAN.md`](MR1_REPLICATION_PLAN.md). Direct targets cover
persistence → affect, age → persistence, and FC → affect. FC → persistence is
a specificity check because the corrected M3 analysis was null. Predefined
sensitivity families are retained but are secondary/exploratory, particularly
because some provisional MR1 sensitivity results were viewed before the plan
was finalized.

**Primary adjusted model: OLS.** The available MR1 P5 analytic inputs do not
contain a usable family, household, sibling, twin-pair, or other grouping
identifier. Participant-level OLS is therefore used. This does not assert that
the underlying MR1 cohort contains no related participants. Results are saved as
`correlations.csv`, `regressions.csv`, and `_methods.txt`. No current MR1
analysis writes `mlm.csv`.

## Established facts

- **Paradigm is identical** to M3 (EmotionRegulation task, 3 runs, neg/neu/pos
  image+face, `valence`/`valenceFollowing` events, TR=2.0).
- **fMRI reproduction complete** — fMRIPrep 24.1.0 ran on Sherlock under
  `/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828`. 123 participants
  completed; 368 EmotionRegulation runs total.
- **Slice-timing correction established** — BIDS slice-timing correction was
  handled normally by fMRIPrep for MR1. No separate manual STC step was
  applied and none is needed.
- **FD summary verified** — the reproduced framewise-displacement summary is
  numerically identical to the historical FD summary.
- **Run completeness established** — strict run-completeness summary:
  - 365 runs with 231 volumes (expected)
  - 1 run with 215 volumes
  - 1 run with 130 volumes
  - 1 run with 6 volumes
  - **120 participants** have all three EmotionRegulation runs with exactly 231
    volumes (the all-three-at-231 criterion used in `qc_conservative`)
- **Historical visual QC carried forward** — manual visual-QC decisions from the
  historical analysis are used as-is. No fMRI preprocessing or first-level
  imaging stage needs to be rerun for the current local analysis.
- **Behavioral data available** — MR1 P2 diary and P5 survey data (plus MKER1
  demographics supplement) are in the PHI-compliant folder. The behavioral
  pipeline below is fully operational.
- **Audited historical scripts retained** — fMRIPrep wrappers, GLM, LSS, and
  extraction scripts under `historical_sherlock_scripts/` and
  `submitted_reproduction_scripts/` are kept for provenance. They do not need
  to be rerun for the current local analysis.

## Variable mapping vs M3

| Concept | M3 variable | MR1 variable |
|---------|-------------|--------------|
| Participant ID | M2ID | MIDUSID (= MRID) |
| Age at neuro visit | C5PAGE | RA5PAGE |
| Sex | C1PRSEX | RA1PRSEX → `sex` |
| Education | C1PB1 | RA1PB1 → `educ` (RAACB1 fallback) |
| Race | C1PF7A | RA1PF7A → `race` (RAACF7A fallback) |
| Birth year | C1PBYEAR | RA1PBYEAR → `birth_year` (RAACBYEAR fallback) |
| Diary items | C2DC1–C2DC27 | RA2DC1–RA2DC27 |
| PANAS | C5SPGP, C5SPGN | RA5SPGP, RA5SPGN |
| ERQ | C5SER, C5SES | RA5SER, RA5SES |
| P5 date | C5PDATE_YR/MO | RA5PDATE_YR, RA5PDATE_MO |

`MKER1_variables1.tsv` provides `RAAC*` fallback demographics for MR1
participants whose P5 (`RA1P*`) values are missing. `MKER1_variables2.tsv` is
not consumed by the current pipeline.

## Raw input files

Place these files in `data/raw/` before running the preprocessing pipeline:

| File | Description | Required |
|------|-------------|---------|
| `MR1_P2_variables.csv` | Daily diary — long format (multiple rows per participant) | Yes |
| `MR1_P5_variables.csv` | Neuroscience + survey — participant-level | Yes |
| `MKER1_variables1.tsv` | MKE Refresher 1 demographics supplement | Yes (demographic fallback) |

The processed MKER1 output is `data/processed/mker1_ids.csv`. `MKER1_variables1.tsv`
provides the `RAAC*` fallback columns; MKER1-only participants do not enter the
analytic universe.

## Sherlock layout (reproduction root)

The reproduction root on Sherlock is
`/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/`.

The following imaging-derived outputs have been copied locally into `data/fMRI/`:

| File | Description |
|------|-------------|
| `all_subjects_betaSeries_LSS_all_conditions_M2ID.csv` | ROI beta-series FC (legacy filename; identifier column is MIDUSID) |
| `negative_persistence_cross_run.csv` | Cross-run negative persistence summaries |
| `positive_persistence_cross_run.csv` | Cross-run positive persistence summaries |
| `fd_summary.csv` | Per-run framewise displacement summary |
| `condition_fd_wide.csv` | Per-condition FD (wide format) |
| `run_completeness.csv` | Per-participant run volume counts |
| `task_fMRI_QC_MR1.xlsx` | Historical manual visual-QC decisions |

## Preprocessing run order

Run all steps from the `MR1_validation/` directory. Script 07 runs after scripts
08 and 09 because it reads the completed `mr1_with_fmri.csv` and uses the shared
analysis sample definitions.

```
1   python scripts/preprocessing/00_validate_inputs.py
    # Validates all raw input files: required columns, no duplicate IDs,
    # expected SAMPLMAJ codes. Aborts on hard failures.
    # Privacy-safe: prints aggregate counts only.

2   python scripts/preprocessing/01_harmonize_ids.py
    # ID harmonization across P2, P5, MKER1_variables1.tsv.
    # Outputs: data/processed/mr1p2_ids.csv
    #          data/processed/mr1p5_ids.csv
    #          data/processed/mker1_ids.csv

3   python scripts/preprocessing/02_construct_daily_diary_affect.py
    # Aggregates P2 long-format diary rows to participant-level affect scores.
    # Output: data/processed/daily_diary_processed.csv

4   python scripts/preprocessing/03_construct_demographics.py
    # Harmonized demographics: RA1P* primary, RAAC* fallback from MKER1.
    # Output: data/processed/mr1p5_demos.csv

5   python scripts/preprocessing/04_construct_covariates.py
    # Race dummies race_2..race_6 (White = reference category).
    # Output: data/processed/mr1p5_covariates.csv

6   python scripts/preprocessing/05_merge_master_dataset.py
    # Analytic universe = diary ∪ P5 via outer merge (merge_combine).
    # MKER1 is a required left-join supplement; it does not add
    # MKER1-only participants to the universe.
    # Output: data/processed/mr1_merged.csv

7   python scripts/preprocessing/06_clean_merged_data.py
    # Computes RA2PAGE, time_P2_P5, recodes ERQ, applies corrected PANAS
    # missing-code handling (8/98/99 → NaN for RA5SPGP, RA5SPGN), and writes
    # the PANAS recoding audit.
    # Outputs: data/processed/mr1_merged_clean.csv
    #          results/tables/panas_skew_kurtosis.csv
    #          results/tables/panas_missing_code_recode.csv

8   python scripts/preprocessing/08_process_fmri_qc.py
    # Computes QC flags: historical visual QC, mean FD < 0.5 mm, task
    # completeness (n_pairs == 6), and all three runs at exactly 231 volumes.
    # These four criteria together define qc_conservative == 1.
    # Output: data/fMRI/fmri_qc_processed.csv

9   python scripts/preprocessing/09_merge_fmri_data.py
    # Merges mr1_merged_clean.csv with fMRI persistence, FC, FD, and QC files
    # into the master analysis file.
    # Output: data/processed/mr1_with_fmri.csv

10  python scripts/preprocessing/07_sample_descriptives.py
    # Reads mr1_with_fmri.csv; uses shared analysis sample definitions to
    # report aggregate descriptive statistics (age, sex, education, race) for
    # behavioral and fMRI samples.
    # Output: results/tables/sample_descriptives.csv
```

## Analysis runbook

Run the four main analyses first, then run the sensitivity scripts. Sensitivity
scripts do not need to wait for any specific main-analysis result — they execute
the complete predefined family unconditionally.

### Main analyses

```
python scripts/analysis/02_persistence_affect.py
    # L amygdala neg persistence → PA_score, NA_score, NA_score_log
    # Conservative sample (primary) + full sample (archive)
    # Output: results/tables/02_persistence_affect/

python scripts/analysis/03_persistence_age.py
    # Age at neuroscience visit → L amygdala neg persistence
    # Conservative sample (primary) + full sample (archive)
    # Output: results/tables/03_persistence_age/

python scripts/analysis/04_fc_affect.py
    # L amyg–ant/post vmPFC FC (neg−neu) → PA_score, NA_score, NA_score_log
    # Conservative sample (primary) + full sample (archive)
    # Output: results/tables/04_fc_affect/

python scripts/analysis/05_fc_persistence.py
    # L amyg–ant/post vmPFC FC (neg−neu) → neg persistence (specificity check)
    # Conservative sample (primary) + full sample (archive)
    # Output: results/tables/05_fc_persistence/
```

### Sensitivity analyses

Run after all four main analyses. These are predefined families run in full;
no section is gated on whether the corresponding main analysis is significant.
Nominal p-values are reported without multiplicity adjustment, matching the
corrected M3 sensitivity-analysis workflow.

```
python scripts/analysis/02_sensitivity.py
    # Section 1: Right-hemisphere negative persistence → diary affect (two-tailed)
    # Section 2: Left-hemisphere negative persistence → PANAS (one-tailed)
    # Output: results/tables/02_persistence_affect/sensitivity_*/

python scripts/analysis/03_sensitivity.py
    # Section 1: Age → right-hemisphere negative persistence (two-tailed)
    # Section 2: Age → bilateral positive persistence (two-tailed)
    # Output: results/tables/03_persistence_age/sensitivity_*/

python scripts/analysis/04_sensitivity.py
    # Section 1: Right-amygdala FC (neg−neu) → diary affect (two-tailed)
    # Section 2: L amygdala pos−neu FC → diary affect (two-tailed)
    # Section 3: L amygdala FC (neg−neu) → PANAS (one-tailed)
    # Section 4: L amygdala neg-only FC → diary affect (one-tailed)
    # Output: results/tables/04_fc_affect/sensitivity_*/
```

**Motion follow-up**: The frozen rule specified motion-adjusted models only if a
direct main effect showed meaningful signal. None of the four direct MR1 tests
reached *p* < .05, so no MR1 motion-adjusted follow-up is required. No
motion-adjusted script was added to this pipeline.

## Analysis outputs

Each analysis directory contains three files:

| File | Contents |
|------|----------|
| `correlations.csv` | Pearson r, n, two-tailed and one-tailed p (where pre-registered) |
| `regressions.csv` | Participant-level OLS: β, SE, t, p, R²; covariates: age, sex, race dummies, plus diary covariates for diary outcomes |
| `_methods.txt` | Variables, covariates, inference type (one- vs. two-tailed), and model provenance |

No BH/FDR-adjusted p-value columns are written. No `mlm.csv` files are produced.

## Testing

Synthetic tests (no real data required or accessed):

```
python -m pytest tests/ -v
```

Tests cover: ID harmonization, missing-value recoding, P5/MKER1 harmonization
precedence, race dummy construction (all five dummies, race_2–race_6),
operator-precedence fix, MKE interview-age formula, strict covariate validation
(`get_covariates` exact ordered list, `SystemExit` on missing columns),
strict sample construction (`get_samples` diary+fMRI mask, conservative subset,
flag validation), persistence transformation (`prepare_persistence_vars` z-column
creation, raw r unchanged, out-of-range detection), diary ∪ P5 outer-join
universe (MKER1 supplement does not add participants), PANAS recoding
(codes 8/98/99 → NaN, valid 1–5 preserved, log offset from recoded values),
`run_analysis_set` two-value return (no MLM), strict missing-input validation
(SystemExit on missing predictor/outcome/covariate), diary covariate requirement
for diary outcomes, `save_results` three-file output including `_methods.txt`.

## Status

| Component | State |
|-----------|-------|
| fMRI reproduction (Sherlock) | ✅ complete — 123 participants, 368 runs, fMRIPrep 24.1.0 |
| FD summary | ✅ verified numerically identical to historical |
| Run completeness | ✅ established — 120 participants at all-three-231-volumes |
| Historical visual QC | ✅ carried forward |
| Local imaging summary inputs | ✅ copied to `data/fMRI/` |
| Validate inputs (00) | ✅ audited |
| ID harmonization (01) | ✅ audited — MKER1_variables1.tsv integrated |
| Diary affect (02) | ✅ audited |
| Demographics (03) | ✅ audited — RA1P* primary, RAAC* fallback |
| Covariates (04) | ✅ audited — all five race dummies |
| Master merge (05) | ✅ audited — diary ∪ P5 outer join, MKER1 left-join supplement |
| Clean / PANAS (06) | ✅ audited — corrected PANAS recoding, audit written |
| fMRI QC (08) | ✅ audited — four-criterion qc_conservative |
| fMRI merge (09) | ✅ audited — writes mr1_with_fmri.csv |
| Sample descriptives (07) | ✅ audited — runs after 08+09 |
| Analysis utils | ✅ audited — OLS primary, no MLM, strict validation |
| Main analyses (02–05) | ✅ audited — conservative + full sample, two-output interface |
| Sensitivity analysis 02 | ✅ audited — right hemisphere + PANAS |
| Sensitivity analysis 03 | ✅ audited — right-negative + bilateral-positive persistence |
| Sensitivity analysis 04 | ✅ audited — four predefined families |
| Synthetic tests | ✅ 118 passed in the MIDUS environment |
| Behavioral preprocessing rerun | ✅ complete on protected real data |
| Main analyses rerun | ✅ complete; no direct test reached p < .05 |
| Sensitivity analyses | ✅ complete predefined families; two nominal OLS findings retained as secondary |
| Motion follow-up | ✅ not required under the frozen decision rule because no direct main effect reached p < .05 |
