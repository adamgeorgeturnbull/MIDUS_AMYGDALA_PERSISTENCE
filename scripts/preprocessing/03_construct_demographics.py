#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construct clean demographics file from MIDUS P2 (daily diary) and P5 (neuroscience) datasets.
Saves 'demographics_processed.csv' to data/processed.
"""

import pandas as pd
import os

# Directories
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

# Filenames
P2_FILE = os.path.join(PROCESSED_DIR, "m3p2_ids.csv")
P5_FILE = os.path.join(PROCESSED_DIR, "m3p5_ids.csv")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "demographics_processed.csv")

# ------------------------------
# Load data
# ------------------------------
p2_cols = [
    "C2DDAY", "C2DIMON", "C2DIYEAR", "M2ID", "SAMPLMAJ", 
    "C1PRAGE", "C1PRSEX", "M2FAMNUM"
]
p5_cols = [
    "SAMPLMAJ", "M2ID", "C1PF1", "C1PF7A", "C1PB1", "C1PIDATE_YR", 
    "C1PRSEX", "C1PIDATE_MO", "CACB1", "CACF1", "CACF7A", "CACRAGE",
    "CACIDATE_MO", "CACIDATE_YR", "CACRSEX", "C1PRAGE", "C5PDATE_MO", 
    "C5PDATE_YR", "C5IC", "C5PAGE", "C5HAND"
]

df_p2 = pd.read_csv(P2_FILE, usecols=p2_cols)
df_p5 = pd.read_csv(P5_FILE, usecols=p5_cols)

# ------------------------------
# Rename overlapping columns to keep track
# ------------------------------
df_p2 = df_p2.rename(columns={
    "C1PRAGE": "C1PRAGE_p2",
    "C1PRSEX": "C1PRSEX_p2",
    "SAMPLMAJ": "SAMPLMAJ_p2",
    "M2ID": "M2ID_p2",
    "M2FAMNUM": "M2FAMNUM_p2"
})

df_p5 = df_p5.rename(columns={
    "C1PRAGE": "C1PRAGE_p5",
    "C1PRSEX": "C1PRSEX_p5",
    "SAMPLMAJ": "SAMPLMAJ_p5",
    "M2ID": "M2ID_p5"
})

# ------------------------------
# Merge datasets safely
# ------------------------------
# Create a merged frame, combine M2ID and SAMPLMAJ safely
merged = pd.DataFrame()
merged['M2ID'] = df_p5['M2ID_p5'].fillna('').combine_first(df_p2['M2ID_p2'].fillna(''))
merged['SAMPLMAJ'] = df_p5['SAMPLMAJ_p5'].fillna('').combine_first(df_p2['SAMPLMAJ_p2'].fillna(''))

# Combine other overlapping variables
merged['C1PRAGE'] = df_p5['C1PRAGE_p5'].combine_first(df_p2['C1PRAGE_p2'])
merged['C1PRSEX'] = df_p5['C1PRSEX_p5'].combine_first(df_p2['C1PRSEX_p2'])
merged['M2FAMNUM'] = df_p2['M2FAMNUM_p2']  # only exists in P2

# Non-overlapping P5 variables
p5_unique = df_p5.drop(columns=['SAMPLMAJ_p5', 'M2ID_p5', 'C1PRAGE_p5', 'C1PRSEX_p5'])
merged = pd.concat([merged, p5_unique], axis=1)

# Non-overlapping P2 variables
p2_unique = df_p2.drop(columns=['SAMPLMAJ_p2', 'M2ID_p2', 'C1PRAGE_p2', 'C1PRSEX_p2'])
merged = pd.concat([merged, p2_unique], axis=1)

# ------------------------------
# Create unified demographics
# ------------------------------
merged['educ'] = merged['C1PB1'].combine_first(merged['CACB1'])
merged['ethnicity'] = merged['C1PF1'].combine_first(merged['CACF1'])
merged['race'] = merged['C1PF7A'].combine_first(merged['CACF7A'])
merged['sex'] = merged['C1PRSEX']

# Replace missing codes
merged['ethnicity'] = merged['ethnicity'].replace({97: pd.NA, 98: pd.NA})
merged['educ'] = merged['educ'].replace({97: pd.NA, 98: pd.NA})
merged['race'] = merged['race'].replace({7: pd.NA, 8: pd.NA})

# ------------------------------
# Calculate age at daily diary collection (C2PAGE)
# ------------------------------
# Use P5 baseline first, then fallback to P2
merged['age_baseline'] = merged['C1PRAGE'].combine_first(merged['CACRAGE'])
merged['baseline_year'] = merged['C1PIDATE_YR'].combine_first(merged['CACIDATE_YR'])
merged['baseline_month'] = merged['C1PIDATE_MO'].combine_first(merged['CACIDATE_MO'])

# Approximate birth year/month
merged['birth_year'] = merged['baseline_year'] - merged['age_baseline']
merged['birth_month'] = merged['baseline_month']

# Get start month/year from P2 daily diary
start_date_info = (
    df_p2[df_p2['C2DDAY'] == 1][['M2ID_p2', 'C2DIMON', 'C2DIYEAR']]
    .drop_duplicates('M2ID_p2')
    .rename(columns={'C2DIMON': 'StartMonth', 'C2DIYEAR': 'StartYear', 'M2ID_p2': 'M2ID'})
)
merged = merged.merge(start_date_info, on='M2ID', how='left')

# Compute C2PAGE
merged['C2PAGE'] = merged['StartYear'] - merged['birth_year']
month_diff = merged['StartMonth'] - merged['birth_month']
merged['C2PAGE'] -= (month_diff < 0).astype(int)

# Clean up intermediate columns
merged.drop(columns=['age_baseline', 'baseline_year', 'baseline_month', 'birth_year', 'birth_month'], inplace=True)

# ------------------------------
# Save processed demographics
# ------------------------------
merged.to_csv(OUTPUT_FILE, index=False)
print(f"Processed demographics saved to: {OUTPUT_FILE}")
