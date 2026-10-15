"""
Step 2: quick check of the raw NHANES files (number of rows and columns, column names).

Input: data/raw/*.XPT
"""

from pathlib import Path

import pandas as pd

# The script is in nhanes-cognition/scripts/, so the project root is one level up
project_dir = Path(__file__).resolve().parent.parent
raw_dir = project_dir / "data" / "raw"

files = sorted(f for f in raw_dir.iterdir() if f.suffix.lower() == ".xpt")
if not files:
    raise FileNotFoundError(f"No .XPT files found in {raw_dir}")

for file in files:
    df = pd.read_sas(file)

    print("\n" + "=" * 60)
    print(file.name)
    print("Shape:", df.shape)
    print("Columns:")
    print(df.columns.tolist())