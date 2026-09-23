# M3 imaging source

`preprocessing/` and `analysis/` contain the definitive corrected-slice-timing M3 pipeline, with the vmPFC image-to-face correction. MR1 has its separate pipeline under `MR1_validation/`.

## Preprocessing

Start from untouched raw inputs: validate the subject list (`make_subject_list.py`), apply the documented orientation correction (`fixOrientation.py`), extract the 1-based slice order (`extractSliceTiming.py JSON_PATH`), and run `slurm_M3_stc_parallel.sh`. fMRIPrep is launched with slice-timing correction disabled. FreeSurfer preparation/retry launchers are retained; participant-specific retry lists remain restricted. `fix_invalid_physio_json.py` repairs metadata syntax; it does not perform physiological noise regression. These scripts are provenance and reproduction source, not instructions to reprocess existing corrected inputs in place.

## Measures

- `runGLM.sh`: condition-level image and following-face maps; 24 motion regressors, cosine drift, AR(1).
- `extract_amygdala.sh` and `run_cross_corr.py`: six directional cross-run image-to-following-face spatial correlations, averaged in Fisher-z space.
- `runBStaskFC_LSS.sh` and `combineBTS_LSS.py`: trialwise beta-series connectivity, with six-second image/fixation events, then run averaging.
- `extract_vmPFC.sh` and `run_cross_corr_vmPFC.py`: the same six directional image-to-following-face comparisons in anterior/posterior vmPFC. Corrected outputs use separate `_image_face` directories. The wide names ending `_image_mean_r` remain for compatibility and denote image-to-face persistence, not image-to-image stability. The correlation script refuses an existing output directory; specify a fresh `--output-dir` for a new run.
- `extract_roi_activations.sh` and `combineROIActivations.py`: condition-level regional means averaged across runs.
- `getMotion.sh`, `extract_condition_fd.sh`, and `combine_condition_fd.py`: run-level and condition-window FD summaries. Inclusion thresholds are applied by `scripts/preprocessing/08_process_fmri_qc.py`.

Paths and scheduler settings target the restricted Sherlock environment. Participant outputs and private logs do not belong in this repository. Superseded variants are preserved in local private archives.
