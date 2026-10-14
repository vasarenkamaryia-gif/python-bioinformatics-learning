"""
NHANES 2011-2012: biological and lifestyle factors associated with
cognitive performance in adults aged 60+.

Run order: 01_prepare_data.py -> 02_check_raw_data.py -> 03_analysis.py
Input:  data/processed/nhanes_cognitive_60plus.csv
Output: results/tables/*.csv, results/figures/*.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ============================================================
# 1. SETUP: paths and settings
# ============================================================
# The script is in nhanes-cognition/scripts/, so the project root is one level up
project_dir = Path(__file__).resolve().parent.parent
file_path = project_dir / "data" / "processed" / "nhanes_cognitive_60plus.csv"

tables_dir = project_dir / "results" / "tables"
figures_dir = project_dir / "results" / "figures"
tables_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", None)

outcomes = ["CFDDS", "CFDAST", "CFDCSR"]
outcome_names = {
    "CFDDS": "Digit Symbol",
    "CFDAST": "Animal Fluency",
    "CFDCSR": "Delayed Recall",
}

# ============================================================
# 2. LOAD DATA AND CHECK IT
# ============================================================
df = pd.read_csv(file_path)

print("File loaded successfully")
print("Dataset shape:", df.shape)
print("\nMissing values:")
print(df.isna().sum())
print("\nData types:")
print(df.dtypes)

# ============================================================
# 3. DESCRIPTIVE ANALYSIS
# ============================================================
print("\nCognitive test summary:")
print(df[outcomes].describe().round(2))

print("\nAge summary:")
print(df["RIDAGEYR"].describe().round(2))
# Note: age is top-coded at 80 in NHANES (80 means "80 years and older")

print("\nCorrelations with age:")
print(df[["RIDAGEYR"] + outcomes].corr().round(3))
# All three tests are negatively correlated with age (simple correlations,
# not adjusted for other variables, not proof of causality).

print("\nCorrelations with sleep duration:")
print(df[["SLD010H"] + outcomes].corr().round(3))
# Simple correlations with sleep are weak; they are not adjusted for age
# or other variables.

print("\nCognitive scores by education level (means):")
print(df.groupby("DMDEDUC2")[outcomes].mean().round(2))

print("\nNumber of participants by education level:")
print(df.groupby("DMDEDUC2").size())

print("\nAvailable CFDDS scores by education level:")
print(df.groupby("DMDEDUC2")["CFDDS"].count())

print("\nMean age by education level:")
print(df.groupby("DMDEDUC2")["RIDAGEYR"].agg(["mean", "std", "count"]).round(2))
# People with lower education are older on average, so adjusting for age matters.

# ============================================================
# 4. MISSING DATA ANALYSIS: who has no test result?
# ============================================================
miss_tables = []
for y in outcomes:
    dm = df.dropna(subset=["DMDEDUC2", "RIAGENDR", "RIDAGEYR"]).copy()
    dm["missing"] = dm[y].isna().astype(int)

    print(f"\n{y}: {dm['missing'].mean() * 100:.1f}% of participants have no result")
    print("Percent missing by education level:")
    print((dm.groupby("DMDEDUC2")["missing"].mean() * 100).round(1))

    # Logistic regression: what is associated with a missing test result?
    # Odds ratio (OR) > 1 means higher odds of a missing result.
    miss_model = smf.logit(
        "missing ~ RIDAGEYR + C(RIAGENDR) + C(DMDEDUC2)", data=dm
    ).fit(disp=0)
    miss_ci = miss_model.conf_int()
    miss_out = pd.DataFrame({
        "outcome": y,
        "OR": np.exp(miss_model.params),
        "ci_low": np.exp(miss_ci[0]),
        "ci_high": np.exp(miss_ci[1]),
        "p": miss_model.pvalues,
    })
    miss_out.index.name = "term"
    print(f"\nLogistic regression for missing {y} (odds ratios):")
    print(miss_out.round(3))
    miss_tables.append(miss_out)

pd.concat(miss_tables).round(4).to_csv(tables_dir / "missingness_analysis.csv")
print("\nTable saved in:", tables_dir / "missingness_analysis.csv")

# ============================================================
# 5. FINAL MODELS: education + age + sex + sleep + BMI + smoking + diabetes
# ============================================================
# Reference categories (the first level of each categorical variable):
#   education: 1 (less than 9th grade); sex: 1 (male);
#   smoking: 1 (smoked at least 100 cigarettes); diabetes: 1 (yes)
predictors = (
    "C(DMDEDUC2) + RIDAGEYR + C(RIAGENDR) + SLD010H + BMXBMI"
    " + C(SMQ020) + C(DIQ010)"
)
needed = outcomes + [
    "DMDEDUC2", "RIDAGEYR", "RIAGENDR",
    "SLD010H", "BMXBMI", "SMQ020", "DIQ010",
]

# The same participants for all three tests (complete cases)
d = df.dropna(subset=needed).copy()
print("\nN final sample:", len(d))
print("\nDiabetes (DIQ010): 1 = yes, 2 = no, 3 = borderline")
print(d["DIQ010"].value_counts())

rows = []
for y in outcomes:
    m = smf.ols(f"{y} ~ {predictors}", data=d).fit(cov_type="HC3")
    ci = m.conf_int()
    t = pd.DataFrame({
        "outcome": y,
        "coef": m.params,
        "ci_low": ci[0],
        "ci_high": ci[1],
        "p_HC3": m.pvalues,
    })
    t["N"] = int(m.nobs)
    t["R2"] = m.rsquared
    t["adj_R2"] = m.rsquared_adj
    rows.append(t)

results = pd.concat(rows)
results.index.name = "term"
results.round(4).to_csv(tables_dir / "regression_main.csv")

print("\nFinal models (OLS with HC3 robust standard errors):")
print(results.round(3))
print("\nTable saved in:", tables_dir / "regression_main.csv")

# ============================================================
# 6. BMI AND DIABETES: does adjusting for diabetes change the BMI estimate?
# ============================================================
print("\nBMI in diabetes groups (1 = diabetes, 2 = no, 3 = borderline):")
print(d.groupby("DIQ010")["BMXBMI"].agg(["mean", "std", "count"]).round(2))

d["has_diabetes"] = (d["DIQ010"] == 1).astype(int)
print("\nCorrelation of BMI and diabetes:")
print(d[["BMXBMI", "has_diabetes"]].corr().round(3))

pred_without_diabetes = (
    "C(DMDEDUC2) + RIDAGEYR + C(RIAGENDR) + SLD010H + BMXBMI + C(SMQ020)"
)
for label, pred in [("with diabetes", predictors),
                    ("without diabetes", pred_without_diabetes)]:
    m = smf.ols(f"CFDAST ~ {pred}", data=d).fit(cov_type="HC3")
    print(f"\nCFDAST, model {label}: BMI coef = {m.params['BMXBMI']:.3f}, "
          f"p = {m.pvalues['BMXBMI']:.3f}")

# ============================================================
# 7. STANDARDIZED EFFECTS AND FOREST PLOT
# ============================================================
# Easier-to-read variables: 1 = yes for binary variables, effects per
# 10 years of age and per 5 BMI units
d["female"] = (d["RIAGENDR"] == 2).astype(int)
d["smoker_ever"] = (d["SMQ020"] == 1).astype(int)
d["diabetes"] = (d["DIQ010"] == 1).astype(int)
d["prediabetes"] = (d["DIQ010"] == 3).astype(int)
d["age_10y"] = d["RIDAGEYR"] / 10
d["bmi_5"] = d["BMXBMI"] / 5
d["sleep_h"] = d["SLD010H"]

predictors_plot = (
    "C(DMDEDUC2) + age_10y + female + sleep_h + bmi_5"
    " + smoker_ever + diabetes + prediabetes"
)

terms = [
    "C(DMDEDUC2)[T.2.0]", "C(DMDEDUC2)[T.3.0]",
    "C(DMDEDUC2)[T.4.0]", "C(DMDEDUC2)[T.5.0]",
    "age_10y", "female", "sleep_h", "bmi_5",
    "smoker_ever", "diabetes", "prediabetes",
]
labels = {
    "C(DMDEDUC2)[T.2.0]": "Education 2 vs 1",
    "C(DMDEDUC2)[T.3.0]": "Education 3 vs 1",
    "C(DMDEDUC2)[T.4.0]": "Education 4 vs 1",
    "C(DMDEDUC2)[T.5.0]": "Education 5 vs 1",
    "age_10y": "Age (per 10 years)",
    "female": "Female vs male",
    "sleep_h": "Sleep (per hour)",
    "bmi_5": "BMI (per 5 units)",
    "smoker_ever": "Ever smoked vs never",
    "diabetes": "Diabetes vs no diabetes",
    "prediabetes": "Borderline diabetes vs no diabetes",
}

fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True)
clean_rows = []
for ax, y in zip(axes, outcomes):
    # Outcome in standard deviation units, so the three tests can be compared
    d[y + "_z"] = (d[y] - d[y].mean()) / d[y].std()
    m = smf.ols(f"{y}_z ~ {predictors_plot}", data=d).fit(cov_type="HC3")
    ci = m.conf_int()
    est = m.params[terms]
    err_low = est - ci.loc[terms, 0]
    err_high = ci.loc[terms, 1] - est

    ypos = list(range(len(terms)))
    ax.errorbar(est, ypos, xerr=[err_low, err_high], fmt="o", capsize=3)
    ax.axvline(0, linestyle="--", color="grey")
    ax.set_yticks(ypos)
    ax.set_yticklabels([labels[t] for t in terms])
    ax.set_title(outcome_names[y])
    ax.set_xlabel("Effect (SD of test score), 95% CI")

    for t in terms:
        clean_rows.append({
            "outcome": y,
            "term": labels[t],
            "effect_SD": est[t],
            "ci_low": ci.loc[t, 0],
            "ci_high": ci.loc[t, 1],
            "p_HC3": m.pvalues[t],
        })

axes[0].invert_yaxis()
plt.tight_layout()
plt.savefig(figures_dir / "forest_plot_main.png", dpi=150)
plt.close(fig)

clean_table = pd.DataFrame(clean_rows)
clean_table.round(4).to_csv(tables_dir / "regression_standardized.csv", index=False)
print("\nStandardized effects (outcome in SD units, HC3):")
print(clean_table.round(3))
print("\nForest plot saved in:", figures_dir / "forest_plot_main.png")

# ============================================================
# 8. MODEL DIAGNOSTICS FOR ALL THREE OUTCOMES
# ============================================================
fig, axes = plt.subplots(3, 2, figsize=(12, 14))
bp_rows = []
for i, y in enumerate(outcomes):
    m_ols = smf.ols(f"{y} ~ {predictors}", data=d).fit()

    # Residuals vs fitted values
    axes[i, 0].scatter(m_ols.fittedvalues, m_ols.resid, alpha=0.4)
    axes[i, 0].axhline(y=0, linestyle="--", color="grey")
    axes[i, 0].set_xlabel("Fitted values")
    axes[i, 0].set_ylabel("Residuals")
    axes[i, 0].set_title(f"{outcome_names[y]}: Residuals vs Fitted")

    # Normal Q-Q plot of residuals
    sm.qqplot(m_ols.resid, line="45", fit=True, ax=axes[i, 1])
    axes[i, 1].set_title(f"{outcome_names[y]}: Normal Q-Q Plot")

    # Breusch-Pagan test for heteroskedasticity
    # H0: constant variance of residuals
    bp = het_breuschpagan(m_ols.resid, m_ols.model.exog)
    bp_rows.append({
        "outcome": y,
        "LM_statistic": bp[0],
        "LM_p_value": bp[1],
        "F_statistic": bp[2],
        "F_p_value": bp[3],
    })

plt.tight_layout()
plt.savefig(figures_dir / "cognitive_model_diagnostics.png", dpi=150)
plt.close(fig)

bp_table = pd.DataFrame(bp_rows)
bp_table.round(4).to_csv(tables_dir / "breusch_pagan_tests.csv", index=False)
print("\nBreusch-Pagan tests:")
print(bp_table.round(4))

# Variance inflation factors (multicollinearity); same predictors in all models
m_vif = smf.ols(f"CFDDS ~ {predictors}", data=d).fit()
exog = m_vif.model.exog
vif = pd.DataFrame({
    "term": m_vif.model.exog_names,
    "VIF": [variance_inflation_factor(exog, j) for j in range(exog.shape[1])],
})
vif = vif[vif["term"] != "Intercept"]
vif.round(3).to_csv(tables_dir / "vif.csv", index=False)
print("\nVariance inflation factors (values above ~5 suggest collinearity):")
print(vif.round(3))

print("\nDiagnostics figure saved in:", figures_dir / "cognitive_model_diagnostics.png")
print("\nAnalysis finished.")

# ============================================================
# 9. SLEEP: is the association non-linear? (sleep categories)
# ============================================================
# Short: less than 7 hours, normal: 7-9 hours (reference), long: more than 9 hours
d["sleep_cat"] = pd.cut(
    d["SLD010H"],
    bins=[0, 6.99, 9, 24],
    labels=["short_lt7", "normal_7_9", "long_gt9"],
).astype(str)
print("\nParticipants by sleep category:")
print(d["sleep_cat"].value_counts())

sleep_formula = (
    "C(DMDEDUC2) + RIDAGEYR + C(RIAGENDR)"
    ' + C(sleep_cat, Treatment(reference="normal_7_9"))'
    " + BMXBMI + C(SMQ020) + C(DIQ010)"
)

sleep_rows = []
for y in outcomes:
    m = smf.ols(f"{y} ~ {sleep_formula}", data=d).fit(cov_type="HC3")
    ci = m.conf_int()
    for term in m.params.index:
        if "sleep_cat" in term:
            sleep_rows.append({
                "outcome": y,
                "term": term.split("[T.")[-1].rstrip("]"),
                "coef": m.params[term],
                "ci_low": ci.loc[term, 0],
                "ci_high": ci.loc[term, 1],
                "p_HC3": m.pvalues[term],
            })

sleep_table = pd.DataFrame(sleep_rows)
sleep_table.round(4).to_csv(tables_dir / "sleep_categories.csv", index=False)
print("\nSleep categories vs normal sleep (7-9 hours), HC3:")
print(sleep_table.round(3))
print("\nSleep analysis finished.")