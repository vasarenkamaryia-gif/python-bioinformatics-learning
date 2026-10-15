"""
Step 1: prepare the NHANES 2011-2012 analysis dataset (participants aged 60+).

Input:  data/raw/*.XPT  (CFQ_G, SLQ_G, DEMO_G, BMX_G, SMQ_G, DIQ_G)
Output: data/processed/nhanes_cognitive_60plus.csv
"""

from pathlib import Path

import pandas as pd

# The script is in nhanes-cognition/scripts/, so the project root is one level up
project_dir = Path(__file__).resolve().parent.parent
raw_dir = project_dir / "data" / "raw"
processed_dir = project_dir / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

# Check that all raw files are in data/raw/
required_files = [
    "CFQ_G.XPT", "SLQ_G.XPT", "DEMO_G.XPT",
    "BMX_G.XPT", "SMQ_G.XPT", "DIQ_G.XPT",
]
missing_files = [name for name in required_files if not (raw_dir / name).exists()]
if missing_files:
    raise FileNotFoundError(f"Missing files in {raw_dir}: {missing_files}")

# Load the datasets
cfq = pd.read_sas(raw_dir / "CFQ_G.XPT")
slq = pd.read_sas(raw_dir / "SLQ_G.XPT")
demo = pd.read_sas(raw_dir / "DEMO_G.XPT")
bmx = pd.read_sas(raw_dir / "BMX_G.XPT")
smq = pd.read_sas(raw_dir / "SMQ_G.XPT")
diq = pd.read_sas(raw_dir / "DIQ_G.XPT")

# Keep only the variables we need
cfq = cfq[["SEQN", "CFDDS", "CFDAST", "CFDCSR"]]
slq = slq[["SEQN", "SLD010H"]]
demo = demo[["SEQN", "RIDAGEYR", "RIAGENDR", "DMDEDUC2"]]
bmx = bmx[["SEQN", "BMXBMI"]]
smq = smq[["SEQN", "SMQ020"]]
diq = diq[["SEQN", "DIQ010"]]

# Merge all datasets using participant ID
df = cfq.merge(slq, on="SEQN", how="inner")
df = df.merge(demo, on="SEQN", how="inner")
df = df.merge(bmx, on="SEQN", how="inner")
df = df.merge(smq, on="SEQN", how="inner")
df = df.merge(diq, on="SEQN", how="inner")

# Keep participants aged 60+
df = df[df["RIDAGEYR"] >= 60]

# Replace special NHANES codes (7/9 and 77/99 = refused / don't know) with missing values
df["DMDEDUC2"] = df["DMDEDUC2"].replace({7: float("nan"), 9: float("nan")})
df["SMQ020"] = df["SMQ020"].replace({7: float("nan"), 9: float("nan")})
df["DIQ010"] = df["DIQ010"].replace({7: float("nan"), 9: float("nan")})
df["SLD010H"] = df["SLD010H"].replace({77: float("nan"), 99: float("nan")})

print("Final dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isna().sum())

# Check the distribution of values and the codes of categorical variables
print("\nCategorical variables:")
print("\nSex:")
print(df["RIAGENDR"].value_counts(dropna=False))

print("\nEducation:")
print(df["DMDEDUC2"].value_counts(dropna=False))

print("\nSmoking:")
print(df["SMQ020"].value_counts(dropna=False))

print("\nDiabetes:")
print(df["DIQ010"].value_counts(dropna=False))

print("\nSleep:")
print(df["SLD010H"].value_counts(dropna=False))

# Save the prepared dataset
output_path = processed_dir / "nhanes_cognitive_60plus.csv"
df.to_csv(output_path, index=False)

print(f"\nDataset saved to: {output_path}")