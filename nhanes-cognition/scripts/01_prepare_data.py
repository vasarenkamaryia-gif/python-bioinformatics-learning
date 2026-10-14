import pandas as pd
from pathlib import Path

folder = Path(r"D:\Documents\Desktop\Bioinformatics\ПРОЕКТ 1")

# Load the datasets
cfq = pd.read_sas(folder / "CFQ_G.XPT")
slq = pd.read_sas(folder / "SLQ_G.XPT")
demo = pd.read_sas(folder / "DEMO_G.XPT")
bmx = pd.read_sas(folder / "BMX_G.XPT")
smq = pd.read_sas(folder / "SMQ_G.XPT")
diq = pd.read_sas(folder / "DIQ_G.xpt")

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

# Replace special NHANES codes with missing values
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

#посмотреть распределение значений и коды категориальных переменных (SMQ020, DIQ010, DMDEDUC2, RIAGENDR, SLD010H).
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
#сохранить подготовленный датасет
from pathlib import Path

project_dir = Path(r"D:\Documents\GitHub\python-bioinformatics-learning")
output_path = project_dir / "nhanes_cognitive_60plus.csv"

df.to_csv(output_path, index=False)

print(f"\nDataset saved to: {output_path}")