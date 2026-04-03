#!/usr/bin/env python3
"""
compare_midus2_overlap.py

Construct a refined MIDUS II comparison sample (diary data + completed
neuroimaging) and compare with the conservative diary+fMRI sample from
the current MIDUS III study.

Inputs:
  data/raw/M2P2_variables.csv        — MIDUS II diary data (B2DC1-27, M2ID)
  data/raw/M2P5_variables.csv        — MIDUS II neuroscience visit (M2ID, B5IC)
  data/processed/midus_with_fmri.csv — current study master dataset

Run from project root directory.
"""

import pandas as pd
import numpy as np
from pathlib import Path

M2P2_FILE   = Path("data/raw/M2P2_variables.csv")
M2P5_FILE   = Path("data/raw/M2P5_variables.csv")
MASTER_FILE = Path("data/processed/midus_with_fmri.csv")

# ── Build MIDUS II comparison sample ─────────────────────────────────────────
# Diary: at least one non-missing affect item (B2DC1-B2DC27)
p2 = pd.read_csv(M2P2_FILE)
p2["M2ID"] = pd.to_numeric(p2["M2ID"], errors="coerce").astype("Int64")
diary_cols = [c for c in p2.columns if c.upper().startswith("B2DC")]
p2["has_diary"] = p2[diary_cols].notna().any(axis=1)
has_diary = set(p2.loc[p2["has_diary"], "M2ID"].dropna().astype(int))
print(f"MIDUS II — has diary data:        N = {len(has_diary)}")

# Neuroimaging: B5IC == 1 (completed MRI)
p5 = pd.read_csv(M2P5_FILE)
p5["M2ID"] = pd.to_numeric(p5["M2ID"], errors="coerce").astype("Int64")
ic_col = next((c for c in p5.columns if "b5ic" in c.lower() or "ic" in c.lower()), None)
if ic_col:
    has_imaging = set(p5.loc[p5[ic_col] == 1, "M2ID"].dropna().astype(int))
else:
    # Fallback: anyone present in the neuroscience file
    has_imaging = set(p5["M2ID"].dropna().astype(int))
    print(f"  (B5IC column not found — using all N in M2P5 as proxy)")
print(f"MIDUS II — completed neuroimaging: N = {len(has_imaging)}")

midus2_ids = has_diary & has_imaging
print(f"MIDUS II — diary + neuroimaging:   N = {len(midus2_ids)}")

# ── Load current conservative sample (MIDUS III) ──────────────────────────────
df = pd.read_csv(MASTER_FILE)
df["M2ID_int"] = pd.to_numeric(df["M2ID"], errors="coerce").astype("Int64")

cons = df[(df.get("qc_conservative", pd.Series(0, index=df.index)) == 1) &
          df[["PA_score", "NA_score"]].notna().any(axis=1)]
cons_ids = set(cons["M2ID_int"].dropna().astype(int))
print(f"\nMIDUS III — conservative diary+fMRI sample: N = {len(cons_ids)}")

# ── Overlap ───────────────────────────────────────────────────────────────────
overlap  = midus2_ids & cons_ids
only_m2  = midus2_ids - cons_ids
only_m3  = cons_ids   - midus2_ids

print(f"\nOverlap (same participants in both studies):")
print(f"  N = {len(overlap)}")
print(f"  {len(overlap)/len(midus2_ids)*100:.1f}% of MIDUS II diary+neuroimaging sample")
print(f"  {len(overlap)/len(cons_ids)*100:.1f}% of current conservative sample")
print(f"\nOnly in MIDUS II: N = {len(only_m2)}")
print(f"Only in MIDUS III: N = {len(only_m3)}")

# Also check against full MIDUS III fMRI sample
full = df[df.get("has_neg_persistence", pd.Series(0, index=df.index)) == 1]
full_ids = set(full["M2ID_int"].dropna().astype(int))
overlap_full = midus2_ids & full_ids
print(f"\nFor reference — overlap with full MIDUS III fMRI sample (N={len(full_ids)}): "
      f"N = {len(overlap_full)} ({len(overlap_full)/len(midus2_ids)*100:.1f}% of MIDUS II)")
