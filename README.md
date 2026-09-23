# MIDUS Amygdala Persistence

Code and aggregate results for analyses of age, daily affect, amygdala persistence, and amygdala–vmPFC connectivity in MIDUS 3 (M3), with targeted replication in MIDUS Refresher 1 (MR1).

## Data access

Participant-level MIDUS data, derived imaging measures, and quality-control summaries require authorized access under the applicable MIDUS data-use agreements and are not distributed here. The pipelines expect restricted local inputs under `data/` and `MR1_validation/data/`; both directories are excluded from Git.

The repository provides analysis code, disclosure-reviewed aggregate results, and publication-approved figures. Manuscripts and submission documents are maintained separately. Review generated outputs for disclosure before sharing.

## Repository structure

| Directory | Contents |
| --- | --- |
| `scripts/preprocessing/` | Behavioral preprocessing, covariates, imaging QC, and data integration |
| `scripts/fMRI/` | M3 imaging preprocessing and measure extraction |
| `scripts/analysis/` | Main analyses, sensitivities, and shared statistical helpers |
| `scripts/diagnostics/` | Confidence-interval and model-fit checks |
| `scripts/tests/` | Synthetic unit tests |
| `scripts/visualization/` | Figure and supplementary-table generators |
| `results/` | Aggregate tables, methods descriptions, and figures |
| `MR1_validation/` | MR1 pipeline, tests, and aggregate results |

## Analysis specifications

M3 analyses cover age–affect associations (`01`), persistence–affect (`02`), age–persistence (`03`), statistical indirect effects (`03b`), connectivity–affect (`04`), connectivity–persistence (`05`), and moderation by reappraisal and suppression (`06`–`07`). Scripts `00a`–`00e` provide convergence, task-condition, contextual, and motion checks. Companion scripts implement sensitivity analyses.

Primary M3 brain–behavior analyses use mixed-effects models with family random intercepts. The QC-retained imaging sample contains 127 participants, the diary overlap contains 81, and primary moderation models contain 80 complete cases. Consult individual exports for model-specific sample sizes. Broader pre-QC outputs in `full_sample` directories are not the primary imaging analyses. MR1 uses participant-level OLS because its available analytic inputs lack a usable grouping identifier.

Key measurement definitions:

- **Persistence:** six directional cross-run image-to-following-face spatial correlations combined in Fisher-z space, for amygdala and vmPFC. Legacy vmPFC columns ending `_image_mean_r` represent image-to-face persistence.
- **Connectivity:** LSS beta-series models use 6-second image-plus-fixation events without separate face regressors. Negative-minus-neutral Fisher-z contrasts are calculated within runs and averaged across runs.
- **Negative affect:** diary scores use an offset log transformation; positive-scale PANAS scores use a natural log without an added constant.

Exports specify confidence-interval methods, sidedness, and fit status. Two-sided 95% intervals accompany some one-tailed tests. Invalid-variance or nonconverged inference must not be interpreted as a null finding. Indirect-effect estimates are statistical associations, not evidence of causal mediation.

## Reproduction

Use code and aggregate outputs from the same revision. Python dependencies are recorded in [requirements.txt](requirements.txt). Imaging software, filesystem paths, and scheduler configuration require separate setup in an authorized environment.

### M3

Run behavioral and statistical scripts from the project root:

1. Prepare imaging summaries using the [M3 imaging guide](scripts/fMRI/README.md). The pipeline applies corrected slice timing to raw BOLD data before fMRIPrep, which runs with its own slice-timing correction disabled. Do not repeat this correction on already corrected inputs.
2. Run `scripts/preprocessing/01_harmonize_ids.py` through `06_clean_merged_data.py` in numerical order.
3. Run `08_process_fmri_qc.py`, then `09_merge_fmri_data.py`, then `07_sample_descriptives.py` in the same directory.
4. Run the numbered context and main scripts in `scripts/analysis/`, including `03b_persistence_age_mediation.py`, followed by the companion sensitivity and exploratory scripts for the required outputs.
5. Check sample sizes, fit status, and confidence-interval exports before generating figures or supplementary tables with `scripts/visualization/`.

See [PREPROCESSING_SUMMARY.md](PREPROCESSING_SUMMARY.md) for detailed provenance. Historical checkpoints in that document describe earlier revisions.

### MR1

Run commands from `MR1_validation/`, following its [runbook](MR1_validation/README.md) and [replication plan](MR1_validation/MR1_REPLICATION_PLAN.md). The shared PANAS plain-log definition above supersedes older offset descriptions in the dated runbook.

### Synthetic tests

From the project root:

```sh
python -m unittest discover -s scripts/tests -v
```

The separate MR1 synthetic suite is documented in its runbook. Tests do not require participant data.
