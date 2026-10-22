# ============================================================
# Step R-2: define the survey design and compute weighted descriptive statistics
# (a weighted version of Table 1).
#
# Input:  data/processed/nhanes_2011_2012_survey.rds
# Output: results/tables/survey_weighted_means.csv
#         results/tables/survey_weighted_proportions.csv
# ============================================================

library(here)
library(dplyr)
library(survey)

tables_dir <- here("results", "tables")
dir.create(tables_dir, showWarnings = FALSE, recursive = TRUE)

nhanes <- readRDS(here("data", "processed", "nhanes_2011_2012_survey.rds"))

# ------------------------------------------------------------
# 1. Factors with readable labels
# ------------------------------------------------------------
nhanes <- nhanes |>
  mutate(
    sex = factor(RIAGENDR, levels = c(1, 2), labels = c("Male", "Female")),
    education = factor(DMDEDUC2, levels = 1:5,
                       labels = c("<9th grade", "9-11th grade", "High school/GED",
                                  "Some college/AA", "College graduate+")),
    smoking = factor(SMQ020, levels = c(1, 2),
                     labels = c("Smoked 100+ cigarettes", "Never smoked 100+ cigarettes")),
    diabetes = factor(DIQ010, levels = c(1, 2, 3),
                      labels = c("Diabetes", "No diabetes", "Borderline"))
  )

# ------------------------------------------------------------
# 2. Survey design: defined on ALL participants
# ------------------------------------------------------------
# id      = cluster (primary sampling unit, PSU)
# strata  = stratum
# weights = examination (MEC) sample weight
# nest    = TRUE because PSU numbers are repeated in different strata
design <- svydesign(
  id = ~SDMVPSU,
  strata = ~SDMVSTRA,
  weights = ~WTMEC2YR,
  nest = TRUE,
  data = nhanes
)
print(design)

# The analysis subpopulation (age 60+, complete data) is selected INSIDE the design.
# Filtering the data before svydesign() would give wrong standard errors.
des_60 <- subset(design, analysis_sample)

cat("\nParticipants in the full design:", nrow(design), "\n")
cat("Participants in the analysis subpopulation:", nrow(des_60), "\n")
cat("Sum of weights in the analysis subpopulation (people represented):",
    format(round(sum(nhanes$WTMEC2YR[nhanes$analysis_sample])), big.mark = ","), "\n")

# ------------------------------------------------------------
# 3. Weighted means with 95% confidence intervals
# ------------------------------------------------------------
cont_vars <- c(
  RIDAGEYR = "Age, years",
  SLD010H  = "Sleep, hours",
  BMXBMI   = "BMI, kg/m2",
  CFDDS    = "Digit Symbol score",
  CFDAST   = "Animal Fluency score",
  CFDCSR   = "Delayed Recall score"
)

est <- svymean(as.formula(paste("~", paste(names(cont_vars), collapse = " + "))), des_60)
ci <- confint(est)
unweighted <- colMeans(nhanes[nhanes$analysis_sample, names(cont_vars)])

weighted_means <- data.frame(
  variable = unname(cont_vars[names(est)]),
  unweighted_mean = unname(unweighted[names(est)]),
  weighted_mean = unname(coef(est)),
  weighted_SE = unname(SE(est)),
  ci_low = unname(ci[, 1]),
  ci_high = unname(ci[, 2]),
  row.names = NULL
)

write.csv(weighted_means, file.path(tables_dir, "survey_weighted_means.csv"), row.names = FALSE)
cat("\nWeighted means (design-based) vs unweighted means:\n")
print(weighted_means |> mutate(across(where(is.numeric), ~ round(.x, 2))))

# ------------------------------------------------------------
# 4. Weighted percentages of categorical variables
# ------------------------------------------------------------
prop_table <- function(var) {
  est <- svymean(as.formula(paste0("~", var)), des_60)
  ci <- confint(est)
  levels_found <- sub(paste0("^", var), "", names(est))
  
  unweighted <- 100 * prop.table(table(nhanes[[var]][nhanes$analysis_sample]))
  
  data.frame(
    variable = var,
    level = levels_found,
    unweighted_percent = as.numeric(unweighted[levels_found]),
    weighted_percent = 100 * unname(coef(est)),
    weighted_SE = 100 * unname(SE(est)),
    ci_low = 100 * unname(ci[, 1]),
    ci_high = 100 * unname(ci[, 2]),
    row.names = NULL
  )
}

weighted_props <- bind_rows(lapply(c("sex", "education", "smoking", "diabetes"), prop_table))

write.csv(weighted_props, file.path(tables_dir, "survey_weighted_proportions.csv"), row.names = FALSE)
cat("\nWeighted percentages (design-based) vs unweighted percentages:\n")
print(weighted_props |> mutate(across(where(is.numeric), ~ round(.x, 2))))

cat("\nTables saved in:", tables_dir, "\n")
cat("Weighted descriptive statistics finished.\n")