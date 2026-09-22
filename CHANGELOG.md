# Changelog

## 2026-09-10 — Motion follow-up aligned with primary MLM findings

- Updated the prespecified M3 motion follow-up to select significant primary mixed-effects
  findings rather than significant zero-order associations.
- This added the significant age–persistence finding and the raw-NA connectivity finding to the
  already tested persistence–raw-NA and connectivity–log-NA findings.
- Reran the complete motion analysis from the real Box project. All four primary findings
  remained significant when controlling separately for mean FD across runs and mean FD during
  negative-condition trials (one-tailed *p*s = .038/.026, .019/.019, .048/.040, and .018/.015,
  respectively).
- Updated the manuscript draft and manuscript-preparation documentation to match the final MLM
  results and motion coverage. No new predictor, outcome, or post hoc sensitivity family was
  introduced.

## 2026-09-09 — Motion MLM formula-name correction

- Corrected `run_mlm()` to make both predictor and outcome names safe for statsmodels formulas.
- Reran the complete M3 motion analysis after identifying that hyphenated FC outcome names had
  caused formula-parsing errors before optimization.
- Restored the four motion-to-FC mixed-effects results; all converged and were nonsignificant
  (two-tailed *p*s = .987, .237, .606, and .129).
- Updated the manuscript draft and project documentation. The substantive motion conclusion and
  motion-adjusted primary findings were unchanged.

## 2026-09-08 — Manuscript-preparation computational freeze

- Froze the corrected M3 analysis after the corrected slice-timing rerun, QC reconstruction,
  regenerated primary/sensitivity analyses, motion checks, and corrected supplementary outputs.
- Completed the targeted MR1 reproduction, behavioral and imaging merge, four direct analyses,
  and all predefined sensitivity families.
- Confirmed that all 118 MR1 synthetic tests pass in the real Box project.
- Drafted corrected manuscript Methods and Results and independently checked the numerical claims
  against the final aggregate result CSVs.
- Entered manuscript preparation. Remaining work is document integration, author review, and
  figure/caption revision; no additional computation is required within the frozen scope.
