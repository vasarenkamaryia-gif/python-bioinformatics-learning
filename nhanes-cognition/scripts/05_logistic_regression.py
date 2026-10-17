"""
Step 5: logistic regression for low cognitive performance (lowest quartile of each test).

Input:  data/processed/nhanes_cognitive_60plus.csv
Output: results/tables/*.csv, results/figures/forest_plot_logistic.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

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

pd.set_option("display.width", 220)

outcomes = ["CFDDS", "CFDAST", "CFDCSR"]
outcome_names = {
    "CFDDS": "Digit Symbol",
    "CFDAST": "Animal Fluency",
    "CFDCSR": "Delayed Recall",
}

# The same analysis sample as in the linear regression models (complete cases)
df = pd.read_csv(file_path)
needed = outcomes + [
    "DMDEDUC2", "RIDAGEYR", "RIAGENDR",
    "SLD010H", "BMXBMI", "SMQ020", "DIQ010",
]
d = df.dropna(subset=needed).copy()
print("Analysis sample N:", len(d))

# ============================================================
# 2. PREDICTORS: binary indicators (1 = yes) and rescaled variables
# ============================================================
# Education: separate indicators for levels 1-4. The reference group is level 5
# (college graduate or above), so odds ratios above 1 mean a higher risk than the
# most educated group.
for code in [1, 2, 3, 4]:
    d[f"edu_{code}"] = (d["DMDEDUC2"] == code).astype(int)

d["age_10y"] = d["RIDAGEYR"] / 10      # odds ratio per 10 years of age
d["bmi_5"] = d["BMXBMI"] / 5           # odds ratio per 5 BMI units
d["sleep_h"] = d["SLD010H"]            # odds ratio per 1 hour of sleep
d["female"] = (d["RIAGENDR"] == 2).astype(int)
d["smoker_ever"] = (d["SMQ020"] == 1).astype(int)
d["diabetes"] = (d["DIQ010"] == 1).astype(int)
d["prediabetes"] = (d["DIQ010"] == 3).astype(int)

terms = {
    "edu_1": "Education: <9th grade vs college graduate+",
    "edu_2": "Education: 9-11th grade vs college graduate+",
    "edu_3": "Education: High school/GED vs college graduate+",
    "edu_4": "Education: Some college/AA vs college graduate+",
    "age_10y": "Age (per 10 years)",
    "female": "Female vs male",
    "sleep_h": "Sleep (per hour)",
    "bmi_5": "BMI (per 5 units)",
    "smoker_ever": "Ever smoked vs never",
    "diabetes": "Diabetes vs no diabetes",
    "prediabetes": "Borderline diabetes vs no diabetes",
}
predictors = " + ".join(terms)

# ============================================================
# 3. OUTCOME: low performance = score in the lowest quartile of the test
# ============================================================
cutoff_rows = []
for y in outcomes:
    cutoff = d[y].quantile(0.25)
    d[f"{y}_low"] = (d[y] <= cutoff).astype(int)
    cutoff_rows.append({
        "test": outcome_names[y],
        "cutoff_score": cutoff,
        "n_low": int(d[f"{y}_low"].sum()),
        "percent_low": d[f"{y}_low"].mean() * 100,
    })

cutoffs = pd.DataFrame(cutoff_rows)
cutoffs.to_csv(tables_dir / "low_performance_definition.csv", index=False, float_format="%.4g")
print("\nDefinition of low performance (score <= 25th percentile of the sample):")
print(cutoffs.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
# Because test scores are whole numbers, more than 25% of people can fall at or below
# the cutoff (for example, in Delayed Recall).

# ============================================================
# 4. LOGISTIC REGRESSION: odds ratios, 95% CI, model fit, forest plot
# ============================================================
or_rows = []
fit_rows = []
fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True)

for ax, y in zip(axes, outcomes):
    model = smf.logit(f"{y}_low ~ {predictors}", data=d).fit(disp=0)
    ci = model.conf_int()

    names = list(terms)
    odds = np.exp(model.params[names]).to_numpy()
    ci_low = np.exp(ci.loc[names, 0]).to_numpy()
    ci_high = np.exp(ci.loc[names, 1]).to_numpy()
    p_values = model.pvalues[names].to_numpy()

    for name, o, lo, hi, p in zip(names, odds, ci_low, ci_high, p_values):
        or_rows.append({
            "outcome": outcome_names[y],
            "term": terms[name],
            "odds_ratio": o,
            "ci_low": lo,
            "ci_high": hi,
            "p_value": p,
        })

    # Model fit: discrimination (AUC) and number of events per predictor
    probs = np.asarray(model.predict())
    low = d[f"{y}_low"].to_numpy()
    u_stat, _ = stats.mannwhitneyu(probs[low == 1], probs[low == 0])
    auc = u_stat / ((low == 1).sum() * (low == 0).sum())
    fit_rows.append({
        "outcome": outcome_names[y],
        "N": int(model.nobs),
        "events": int(low.sum()),
        "events_per_predictor": low.sum() / len(names),
        "McFadden_R2": model.prsquared,
        "AUC": auc,
    })

    # Forest plot panel (log scale: an odds ratio of 1 means no association)
    ypos = list(range(len(names)))
    ax.errorbar(odds, ypos, xerr=[odds - ci_low, ci_high - odds], fmt="o", capsize=3)
    ax.axvline(1, linestyle="--", color="grey")
    ax.set_xscale("log")
    ax.set_yticks(ypos)
    ax.set_yticklabels([terms[n] for n in names])
    ax.set_title(f"{outcome_names[y]}: odds of low performance")
    ax.set_xlabel("Odds ratio (95% CI, log scale)")

axes[0].invert_yaxis()
plt.tight_layout()
plt.savefig(figures_dir / "forest_plot_logistic.png", dpi=150)
plt.close(fig)

or_table = pd.DataFrame(or_rows)
or_table.to_csv(tables_dir / "logistic_odds_ratios.csv", index=False, float_format="%.4g")
print("\nOdds ratios of low performance (logistic regression):")
print(or_table.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

fit_table = pd.DataFrame(fit_rows)
fit_table.to_csv(tables_dir / "logistic_model_fit.csv", index=False, float_format="%.4g")
print("\nModel fit (AUC = ability to separate low performers from the others):")
print(fit_table.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

print("\nTables saved in:", tables_dir)
print("Forest plot saved in:", figures_dir / "forest_plot_logistic.png")
print("\nLogistic regression finished.")