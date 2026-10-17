"""
Step 4: descriptive statistics, distributions, normality and correlations.

Input:  data/processed/nhanes_cognitive_60plus.csv
Output: results/tables/*.csv, results/figures/*.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ============================================================
# 1. SETUP: paths, labels, analysis sample
# ============================================================
# The script is in nhanes-cognition/scripts/, so the project root is one level up
project_dir = Path(__file__).resolve().parent.parent
file_path = project_dir / "data" / "processed" / "nhanes_cognitive_60plus.csv"

tables_dir = project_dir / "results" / "tables"
figures_dir = project_dir / "results" / "figures"
tables_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
pd.set_option("display.width", 200)

outcomes = ["CFDDS", "CFDAST", "CFDCSR"]
outcome_names = {
    "CFDDS": "Digit Symbol",
    "CFDAST": "Animal Fluency",
    "CFDCSR": "Delayed Recall",
}
sex_labels = {1: "Male", 2: "Female"}
education_labels = {
    1: "<9th grade",
    2: "9-11th grade",
    3: "High school/GED",
    4: "Some college/AA",
    5: "College graduate+",
}
smoking_labels = {1: "Smoked 100+ cigarettes", 2: "Never smoked 100+ cigarettes"}
diabetes_labels = {1: "Diabetes", 2: "No diabetes", 3: "Borderline"}

# The same analysis sample as in the regression models (complete cases)
df = pd.read_csv(file_path)
needed = outcomes + [
    "DMDEDUC2", "RIDAGEYR", "RIAGENDR",
    "SLD010H", "BMXBMI", "SMQ020", "DIQ010",
]
d = df.dropna(subset=needed).copy()
for col in ["DMDEDUC2", "RIAGENDR", "SMQ020", "DIQ010"]:
    d[col] = d[col].astype(int)
print("Analysis sample N:", len(d))

# ============================================================
# 2. TABLE 1: characteristics of the analysis sample
# ============================================================
def mean_sd(s):
    return f"{s.mean():.1f} ({s.std():.1f})"


def median_iqr(s):
    q1, q3 = s.quantile([0.25, 0.75])
    return f"{s.median():.1f} [{q1:.1f}, {q3:.1f}]"


rows = [("N", str(len(d)))]

continuous = [
    ("RIDAGEYR", "Age, years"),
    ("SLD010H", "Sleep, hours"),
    ("BMXBMI", "BMI, kg/m2"),
    ("CFDDS", "Digit Symbol score"),
    ("CFDAST", "Animal Fluency score"),
    ("CFDCSR", "Delayed Recall score"),
]
for col, label in continuous:
    rows.append((f"{label}, mean (SD)", mean_sd(d[col])))
    rows.append((f"{label}, median [IQR]", median_iqr(d[col])))

categorical = [
    ("RIAGENDR", "Sex", sex_labels),
    ("DMDEDUC2", "Education", education_labels),
    ("SMQ020", "Smoking history", smoking_labels),
    ("DIQ010", "Diabetes", diabetes_labels),
]
for col, label, mapping in categorical:
    rows.append((f"{label}, n (%)", ""))
    counts = d[col].value_counts()
    for code, name in mapping.items():
        n = int(counts.get(code, 0))
        rows.append((f"    {name}", f"{n} ({n / len(d) * 100:.1f}%)"))

table1 = pd.DataFrame(rows, columns=["Characteristic", "Overall"])
table1.to_csv(tables_dir / "table1_characteristics.csv", index=False)
print("\nTable 1: characteristics of the analysis sample")
print(table1.to_string(index=False))
print("\nTable saved in:", tables_dir / "table1_characteristics.csv")

# ============================================================
# 3. DISTRIBUTIONS AND NORMALITY OF THE COGNITIVE TESTS
# ============================================================
norm_rows = []
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, y in zip(axes, outcomes):
    sns.histplot(d[y], kde=True, ax=ax, color="steelblue")
    ax.set_title(outcome_names[y])
    ax.set_xlabel("Score")

    # Shapiro-Wilk test: H0 = the data come from a normal distribution
    w_stat, p_value = stats.shapiro(d[y])
    norm_rows.append({
        "test": outcome_names[y],
        "mean": d[y].mean(),
        "SD": d[y].std(),
        "skewness": stats.skew(d[y]),
        "excess_kurtosis": stats.kurtosis(d[y]),
        "Shapiro_W": w_stat,
        "Shapiro_p": p_value,
    })

plt.tight_layout()
plt.savefig(figures_dir / "distributions_cognitive_tests.png", dpi=150)
plt.close(fig)

norm_table = pd.DataFrame(norm_rows)
norm_table.round(4).to_csv(tables_dir / "normality_tests.csv", index=False)
print("\nDistribution and normality (Shapiro-Wilk) of the cognitive tests:")
print(norm_table.round(4).to_string(index=False))
# With N > 1000 the Shapiro-Wilk test detects even tiny deviations from normality,
# so skewness and kurtosis are reported together with the p-value.

# ============================================================
# 4. SPEARMAN CORRELATION MATRIX
# ============================================================
corr_cols = ["RIDAGEYR", "SLD010H", "BMXBMI"] + outcomes
corr_names = {
    "RIDAGEYR": "Age",
    "SLD010H": "Sleep",
    "BMXBMI": "BMI",
    "CFDDS": "Digit Symbol",
    "CFDAST": "Animal Fluency",
    "CFDCSR": "Delayed Recall",
}
corr = d[corr_cols].corr(method="spearman").rename(index=corr_names, columns=corr_names)
corr.round(3).to_csv(tables_dir / "spearman_correlations.csv")
print("\nSpearman correlations:")
print(corr.round(3).to_string())

fig, ax = plt.subplots(figsize=(7.5, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1,
            square=True, ax=ax)
ax.set_title("Spearman correlations")
plt.tight_layout()
plt.savefig(figures_dir / "spearman_heatmap.png", dpi=150)
plt.close(fig)

# ============================================================
# 5. COGNITIVE SCORES BY EDUCATION LEVEL (boxplots)
# ============================================================
d["education"] = d["DMDEDUC2"].map(education_labels)
order = list(education_labels.values())

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, y in zip(axes, outcomes):
    sns.boxplot(data=d, x="education", y=y, order=order, color="lightsteelblue", ax=ax)
    ax.set_title(outcome_names[y])
    ax.set_xlabel("Education")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig(figures_dir / "boxplots_by_education.png", dpi=150)
plt.close(fig)

print("\nFigures saved in:", figures_dir)
print("\nDescriptive analysis finished.")

# ============================================================
# 6. GROUP COMPARISONS: tests, effect sizes and FDR correction
# ============================================================
# Family 1: comparisons of cognitive scores between groups (12 tests in total).
# - Two groups: Mann-Whitney U test (Wilcoxon rank-sum) is the primary test because the
#   cognitive tests are not exactly normal. Welch's t-test is shown for comparison.
# - Five education groups: Kruskal-Wallis test.
# Effect sizes: Cohen's d and rank-biserial correlation (two groups), epsilon squared
# (Kruskal-Wallis). p-values are corrected for multiple testing with the
# Benjamini-Hochberg false discovery rate (FDR) procedure.
binary_comparisons = [
    # (label, column, code of group A, code of group B, name of A, name of B)
    ("Sex", "RIAGENDR", 2, 1, "Female", "Male"),
    ("Diabetes", "DIQ010", 1, 2, "Diabetes", "No diabetes"),
    ("Smoking history", "SMQ020", 1, 2, "Smoked 100+ cigarettes", "Never smoked"),
]

test_rows = []
for label, col, code_a, code_b, name_a, name_b in binary_comparisons:
    for y in outcomes:
        a = d.loc[d[col] == code_a, y]
        b = d.loc[d[col] == code_b, y]

        u_stat, p_mwu = stats.mannwhitneyu(a, b, alternative="two-sided")
        _, p_welch = stats.ttest_ind(a, b, equal_var=False)

        pooled_sd = np.sqrt(
            ((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2)
        )
        cohens_d = (a.mean() - b.mean()) / pooled_sd
        rank_biserial = 2 * u_stat / (len(a) * len(b)) - 1

        test_rows.append({
            "comparison": f"{label}: {name_a} vs {name_b}",
            "outcome": outcome_names[y],
            "n": f"{len(a)} vs {len(b)}",
            "test": "Mann-Whitney U",
            "statistic": u_stat,
            "p_raw": p_mwu,
            "effect_size_type": "rank-biserial r",
            "effect_size": rank_biserial,
            "median_A": a.median(),
            "median_B": b.median(),
            "cohens_d": cohens_d,
            "welch_t_p": p_welch,
        })

for y in outcomes:
    groups = [d.loc[d["DMDEDUC2"] == code, y] for code in education_labels]
    h_stat, p_kw = stats.kruskal(*groups)
    test_rows.append({
        "comparison": "Education: 5 groups",
        "outcome": outcome_names[y],
        "n": str(len(d)),
        "test": "Kruskal-Wallis",
        "statistic": h_stat,
        "p_raw": p_kw,
        "effect_size_type": "epsilon squared",
        "effect_size": h_stat / (len(d) - 1),
        "median_A": float("nan"),
        "median_B": float("nan"),
        "cohens_d": float("nan"),
        "welch_t_p": float("nan"),
    })

tests = pd.DataFrame(test_rows)
# Benjamini-Hochberg FDR correction across all 12 tests of this family
tests["p_FDR"] = multipletests(tests["p_raw"], method="fdr_bh")[1]
tests["significant_FDR"] = tests["p_FDR"] < 0.05

tests.to_csv(tables_dir / "group_comparison_tests.csv", index=False, float_format="%.4g")
print("\nGroup comparisons of cognitive scores (FDR-corrected, 12 tests):")
print(tests[["comparison", "outcome", "test", "p_raw", "p_FDR",
             "effect_size_type", "effect_size", "cohens_d"]]
      .to_string(index=False, float_format=lambda x: f"{x:.4g}"))
print("\nTable saved in:", tables_dir / "group_comparison_tests.csv")

# Family 2: chi-square tests of association between categorical variables (3 tests).
# Effect size: Cramer's V. The expected cell counts should be at least 5.
label_maps = {
    "DIQ010": diabetes_labels,
    "DMDEDUC2": education_labels,
    "RIAGENDR": sex_labels,
    "SMQ020": smoking_labels,
}
chi_pairs = [
    ("Diabetes by education", "DIQ010", "DMDEDUC2"),
    ("Diabetes by sex", "DIQ010", "RIAGENDR"),
    ("Smoking history by sex", "SMQ020", "RIAGENDR"),
]

chi_rows = []
for label, row_var, col_var in chi_pairs:
    table = pd.crosstab(d[row_var], d[col_var])
    chi2, p_chi, dof, expected = stats.chi2_contingency(table)
    cramers_v = np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1)))
    chi_rows.append({
        "association": label,
        "chi2": chi2,
        "df": dof,
        "p_raw": p_chi,
        "cramers_v": cramers_v,
        "min_expected_count": expected.min(),
    })

    percents = pd.crosstab(d[row_var], d[col_var], normalize="columns") * 100
    percents = percents.rename(index=label_maps[row_var], columns=label_maps[col_var])
    print(f"\n{label}: % of each column group (columns add up to 100%)")
    print(percents.round(1).to_string())

chi_table = pd.DataFrame(chi_rows)
# FDR correction across the 3 chi-square tests
chi_table["p_FDR"] = multipletests(chi_table["p_raw"], method="fdr_bh")[1]
chi_table["significant_FDR"] = chi_table["p_FDR"] < 0.05

chi_table.to_csv(tables_dir / "chi_square_tests.csv", index=False, float_format="%.4g")
print("\nChi-square tests of association (FDR-corrected, 3 tests):")
print(chi_table.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
print("\nTable saved in:", tables_dir / "chi_square_tests.csv")

print("\nDescriptive analysis and group comparisons finished.")