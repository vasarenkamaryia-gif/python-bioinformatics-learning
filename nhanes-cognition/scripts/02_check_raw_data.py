import pandas as pd
from pathlib import Path

folder = Path(r"D:\Documents\Desktop\Bioinformatics\ПРОЕКТ 1")

for file in folder.glob("*.XPT"):
    df = pd.read_sas(file)

    print("\n" + "=" * 60)
    print(file.name)
    print("Shape:", df.shape)
    print("Columns:")
    print(df.columns.tolist())