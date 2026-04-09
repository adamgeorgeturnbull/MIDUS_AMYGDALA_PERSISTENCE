import pandas as pd

df = pd.read_csv("data/processed/midus_with_fmri.csv")

sample = df.dropna(subset=["neg_persist_crossrun_mean_r_L", "PA_score"])
print("N diary+fMRI sample:", len(sample))
print("N with C5SER:", sample["C5SER"].notna().sum())
print("N with C5SES:", sample["C5SES"].notna().sum())
print("Missing C5SER:", sample["C5SER"].isna().sum())
print("Missing C5SES:", sample["C5SES"].isna().sum())

missing_ser = sample[sample["C5SER"].isna()][["M2ID", "C5SER", "C5SES", "neg_persist_crossrun_mean_r_L"]]
missing_ses = sample[sample["C5SES"].isna()][["M2ID", "C5SER", "C5SES", "neg_persist_crossrun_mean_r_L"]]
missing = pd.concat([missing_ser, missing_ses]).drop_duplicates()
print("\nSubjects missing ERQ:")
print(missing)
