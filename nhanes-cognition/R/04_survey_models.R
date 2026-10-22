# ============================================================
# Step R-4: design-based regression models (survey package).
# Weighted counterpart of the Python models (03_analysis.py and 05_logistic_regression.py).
#
# Input:  data/processed/nhanes_2011_2012_survey.rds
#         results/tables/regression_main.csv        (Python linear models)
#         results/tables/logistic_odds_ratios.csv   (Python logistic models)
# Output: results/tables/survey_linear_models.csv
#         results/tables/survey_vs_python_linear.csv
#         results/tables/survey_logistic_odds_ratios.csv
#         results/tables/survey_vs_python_logistic.csv
# ============================================================

library(here)
library(dplyr)
library(survey)

tables_dir <- here("results", "tables")
dir.create(tables_dir, showWarnings = FALSE, recursive = TRUE)

nhanes <- readRDS(here("data", "processed", "nhanes_2011_2012_survey.rds"))

outcomes <- c(CFDDS = "Digit Symbol", CFDAST = "Animal Fluency", CFDCSR = "Delayed Recall")

# ------------------------------------------------------------
# 1. Variables (the same coding and reference categories as in Python)
# ------------------------------------------------------------
# "Low performance" = score <= the 25th percentile of the analysis sample
dat60 <- nhanes[nhanes$analysis_sample, ]
cutoffs <- sapply(names(outcomes), function(y) unname(quantile(dat60[[y]], 0.25)))
cat("Cut-off scores for low performance (Python: 33, 13, 4):\n")
print(cutoffs)

nhanes <- nhanes |>
  mutate(
    # linear models: factors with reference level = first level
    edu     = factor(DMDEDUC2, levels = 1:5),   # reference: 1 (<9th grade)
    sex_f   = factor(RIAGENDR, levels = 1:2),   # reference: 1 (male)
    smoke_f = factor(SMQ020, levels = 1:2),     # reference: 1 (smoked 100+ cigarettes)
    diab_f  = factor(DIQ010, levels = 1:3),     # reference: 1 (diabetes)
    # logistic models: indicators; reference education = college graduate+
    edu_1 = as.integer(DMDEDUC2 == 1),
    edu_2 = as.integer(DMDEDUC2 == 2),
    edu_3 = as.integer(DMDEDUC2 == 3),
    edu_4 = as.integer(DMDEDUC2 == 4),
    age_10y = RIDAGEYR / 10,
    bmi_5 = BMXBMI / 5,
    sleep_h = SLD010H,
    female = as.integer(RIAGENDR == 2),
    smoker_ever = as.integer(SMQ020 == 1),
    diabetes = as.integer(DIQ010 == 1),
    prediabetes = as.integer(DIQ010 == 3),
    low_CFDDS  = as.integer(CFDDS  <= cutoffs[["CFDDS"]]),
    low_CFDAST = as.integer(CFDAST <= cutoffs[["CFDAST"]]),
    low_CFDCSR = as.integer(CFDCSR <= cutoffs[["CFDCSR"]])
  )

# ------------------------------------------------------------
# 2. Survey design on ALL participants; the analysis sample is selected INSIDE the design
# ------------------------------------------------------------
design <- svydesign(id = ~SDMVPSU, strata = ~SDMVSTRA, weights = ~WTMEC2YR,
                    nest = TRUE, data = nhanes)
des_60 <- subset(design, analysis_sample)

cat("\nParticipants in the analysis subpopulation:", nrow(des_60), "\n")
cat("Design degrees of freedom (clusters - strata):", degf(des_60), "\n")

# Helper: coefficients with t-based 95% CI (t distribution with the residual
# degrees of freedom of the design) and p-values
tidy_svyglm <- function(fit) {
  s <- summary(fit)$coefficients
  df <- df.residual(fit)
  crit <- qt(0.975, df)
  data.frame(
    term_r = rownames(s),
    estimate = s[, "Estimate"],
    se = s[, "Std. Error"],
    ci_low = s[, "Estimate"] - crit * s[, "Std. Error"],
    ci_high = s[, "Estimate"] + crit * s[, "Std. Error"],
    p_value = s[, "Pr(>|t|)"],
    df = df,
    row.names = NULL
  )
}

# ------------------------------------------------------------
# 3. Linear models: education + age + sex + sleep + BMI + smoking + diabetes
# ------------------------------------------------------------
lin_formula <- "edu + RIDAGEYR + sex_f + SLD010H + BMXBMI + smoke_f + diab_f"

# R coefficient names -> the term names used in the Python table
term_map <- c(
  "(Intercept)" = "Intercept",
  edu2 = "C(DMDEDUC2)[T.2.0]", edu3 = "C(DMDEDUC2)[T.3.0]",
  edu4 = "C(DMDEDUC2)[T.4.0]", edu5 = "C(DMDEDUC2)[T.5.0]",
  sex_f2 = "C(RIAGENDR)[T.2.0]",
  smoke_f2 = "C(SMQ020)[T.2.0]",
  diab_f2 = "C(DIQ010)[T.2.0]", diab_f3 = "C(DIQ010)[T.3.0]",
  RIDAGEYR = "RIDAGEYR", SLD010H = "SLD010H", BMXBMI = "BMXBMI"
)

lin_rows <- list()
for (y in names(outcomes)) {
  fit <- svyglm(as.formula(paste(y, "~", lin_formula)), design = des_60)
  tab <- tidy_svyglm(fit)
  tab$outcome <- y
  tab$term <- unname(term_map[tab$term_r])
  lin_rows[[length(lin_rows) + 1]] <- tab
}

lin_table <- bind_rows(lin_rows) |>
  select(outcome, term, coef = estimate, se, ci_low, ci_high, p_value, df)

write.csv(lin_table |> mutate(across(where(is.numeric), ~ signif(.x, 4))),
          file.path(tables_dir, "survey_linear_models.csv"), row.names = FALSE)

# Comparison with the Python models (OLS with HC3 standard errors, unweighted)
py_lin <- read.csv(here("results", "tables", "regression_main.csv")) |>
  select(outcome, term, python_coef = coef, python_ci_low = ci_low,
         python_ci_high = ci_high, python_p = p_HC3)

lin_compare <- lin_table |>
  filter(term != "Intercept") |>
  left_join(py_lin, by = c("outcome", "term")) |>
  mutate(
    same_direction = sign(coef) == sign(python_coef),
    significant_survey = p_value < 0.05,
    significant_python = python_p < 0.05,
    same_significance = significant_survey == significant_python
  )

write.csv(lin_compare |> mutate(across(where(is.numeric), ~ signif(.x, 4))),
          file.path(tables_dir, "survey_vs_python_linear.csv"), row.names = FALSE)

cat("\nLinear models: design-based (survey) vs Python (unweighted, HC3):\n")
print(lin_compare |>
        select(outcome, term, python_coef, coef, python_p, p_value,
               same_direction, same_significance) |>
        mutate(across(where(is.numeric), ~ signif(.x, 3))),
      row.names = FALSE)

sig_py <- lin_compare$significant_python
cat("\nLinear models: coefficients with the same direction:",
    sum(lin_compare$same_direction), "of", nrow(lin_compare), "\n")
cat("Terms significant in Python:", sum(sig_py),
    "| of them also significant in the survey models:",
    sum(sig_py & lin_compare$significant_survey), "\n")
cat("Terms with the same significance conclusion:",
    sum(lin_compare$same_significance), "of", nrow(lin_compare), "\n")

# ------------------------------------------------------------
# 4. Logistic models: odds of low performance (lowest quartile)
# ------------------------------------------------------------
log_terms <- c(
  edu_1 = "Education: <9th grade vs college graduate+",
  edu_2 = "Education: 9-11th grade vs college graduate+",
  edu_3 = "Education: High school/GED vs college graduate+",
  edu_4 = "Education: Some college/AA vs college graduate+",
  age_10y = "Age (per 10 years)",
  female = "Female vs male",
  sleep_h = "Sleep (per hour)",
  bmi_5 = "BMI (per 5 units)",
  smoker_ever = "Ever smoked vs never",
  diabetes = "Diabetes vs no diabetes",
  prediabetes = "Borderline diabetes vs no diabetes"
)
log_formula <- paste(names(log_terms), collapse = " + ")

log_rows <- list()
for (y in names(outcomes)) {
  fit <- svyglm(as.formula(paste0("low_", y, " ~ ", log_formula)),
                design = des_60, family = quasibinomial())
  tab <- tidy_svyglm(fit) |> filter(term_r != "(Intercept)")
  tab$outcome <- unname(outcomes[y])
  tab$term <- unname(log_terms[tab$term_r])
  log_rows[[length(log_rows) + 1]] <- tab
}

or_table <- bind_rows(log_rows) |>
  transmute(outcome, term,
            odds_ratio = exp(estimate), ci_low = exp(ci_low), ci_high = exp(ci_high),
            p_value, df)

write.csv(or_table |> mutate(across(where(is.numeric), ~ signif(.x, 4))),
          file.path(tables_dir, "survey_logistic_odds_ratios.csv"), row.names = FALSE)

py_log <- read.csv(here("results", "tables", "logistic_odds_ratios.csv")) |>
  select(outcome, term, python_or = odds_ratio, python_ci_low = ci_low,
         python_ci_high = ci_high, python_p = p_value)

log_compare <- or_table |>
  left_join(py_log, by = c("outcome", "term")) |>
  mutate(
    same_direction = (odds_ratio > 1) == (python_or > 1),
    significant_survey = p_value < 0.05,
    significant_python = python_p < 0.05,
    same_significance = significant_survey == significant_python
  )

write.csv(log_compare |> mutate(across(where(is.numeric), ~ signif(.x, 4))),
          file.path(tables_dir, "survey_vs_python_logistic.csv"), row.names = FALSE)

cat("\nLogistic models (odds of low performance): design-based (survey) vs Python:\n")
print(log_compare |>
        select(outcome, term, python_or, odds_ratio, python_p, p_value,
               same_direction, same_significance) |>
        mutate(across(where(is.numeric), ~ signif(.x, 3))),
      row.names = FALSE)

sig_py <- log_compare$significant_python
cat("\nLogistic models: odds ratios with the same direction:",
    sum(log_compare$same_direction), "of", nrow(log_compare), "\n")
cat("Terms significant in Python:", sum(sig_py),
    "| of them also significant in the survey models:",
    sum(sig_py & log_compare$significant_survey), "\n")
cat("Terms with the same significance conclusion:",
    sum(log_compare$same_significance), "of", nrow(log_compare), "\n")

cat("\nTables saved in:", tables_dir, "\n")
cat("Design-based regression models finished.\n")