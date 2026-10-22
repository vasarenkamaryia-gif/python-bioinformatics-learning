# ============================================================
# Step R-3: design-based group comparisons (survey package).
# Weighted counterpart of 04_group_comparisons.py.
#
# Input:  data/processed/nhanes_2011_2012_survey.rds
#         results/tables/group_comparison_tests.csv  (Python results, for comparison)
# Output: results/tables/survey_group_comparison_tests.csv
#         results/tables/survey_chi_square_tests.csv
#         results/tables/survey_vs_python_group_tests.csv
# ============================================================

library(here)
library(dplyr)
library(survey)

tables_dir <- here("results", "tables")
dir.create(tables_dir, showWarnings = FALSE, recursive = TRUE)

nhanes <- readRDS(here("data", "processed", "nhanes_2011_2012_survey.rds"))

outcomes <- c(CFDDS = "Digit Symbol", CFDAST = "Animal Fluency", CFDCSR = "Delayed Recall")

# ------------------------------------------------------------
# 1. Factors. For the two-group comparisons the SECOND level is group A,
#    so the estimated difference is (group A - group B), as in the Python analysis.
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
                      labels = c("Diabetes", "No diabetes", "Borderline")),
    sex_bin = sex,
    diabetes_bin = factor(
      case_when(diabetes == "Diabetes" ~ "Diabetes",
                diabetes == "No diabetes" ~ "No diabetes",
                TRUE ~ NA_character_),
      levels = c("No diabetes", "Diabetes")),
    smoking_bin = factor(
      case_when(smoking == "Smoked 100+ cigarettes" ~ "Smoked 100+ cigarettes",
                smoking == "Never smoked 100+ cigarettes" ~ "Never smoked",
                TRUE ~ NA_character_),
      levels = c("Never smoked", "Smoked 100+ cigarettes"))
  )

# ------------------------------------------------------------
# 2. Survey design on ALL participants; subpopulations are selected INSIDE the design
# ------------------------------------------------------------
design <- svydesign(id = ~SDMVPSU, strata = ~SDMVSTRA, weights = ~WTMEC2YR,
                    nest = TRUE, data = nhanes)

des_60 <- subset(design, analysis_sample)
designs_binary <- list(
  sex_bin      = subset(design, analysis_sample),
  diabetes_bin = subset(design, analysis_sample & !is.na(diabetes_bin)),  # borderline excluded
  smoking_bin  = subset(design, analysis_sample & !is.na(smoking_bin))
)

# Weighted standard deviation of each test in the analysis population,
# used to express differences in SD units (a weighted analogue of Cohen's d)
weighted_sd <- function(x, w) {
  m <- weighted.mean(x, w)
  sqrt(sum(w * (x - m)^2) / sum(w))
}
dat60 <- nhanes[nhanes$analysis_sample, ]
sd_w <- sapply(names(outcomes), function(y) weighted_sd(dat60[[y]], dat60$WTMEC2YR))

# ------------------------------------------------------------
# 3. Family 1: comparisons of cognitive scores between groups (12 tests)
# ------------------------------------------------------------
binary_comparisons <- list(
  list(label = "Sex: Female vs Male", group = "sex_bin"),
  list(label = "Diabetes: Diabetes vs No diabetes", group = "diabetes_bin"),
  list(label = "Smoking history: Smoked 100+ cigarettes vs Never smoked", group = "smoking_bin")
)

rows <- list()

for (cmp in binary_comparisons) {
  des_g <- designs_binary[[cmp$group]]
  for (y in names(outcomes)) {
    fml <- as.formula(paste(y, "~", cmp$group))
    
    rk <- svyranktest(fml, des_g, test = "wilcoxon")   # design-based Wilcoxon rank test
    tt <- svyttest(fml, des_g)                         # design-based t-test
    fit <- svyglm(fml, des_g)                          # same model: gives the CI of the difference
    ci <- confint(fit)[2, ]
    means <- svyby(as.formula(paste0("~", y)), as.formula(paste0("~", cmp$group)),
                   des_g, svymean)
    
    rows[[length(rows) + 1]] <- data.frame(
      comparison = cmp$label,
      outcome = unname(outcomes[y]),
      test = "Design-based Wilcoxon rank test",
      n = nrow(des_g),
      statistic = unname(rk$statistic),
      p_raw = unname(rk$p.value),
      weighted_mean_A = means[[y]][2],
      weighted_mean_B = means[[y]][1],
      mean_difference = unname(coef(fit)[2]),
      ci_low = unname(ci[1]),
      ci_high = unname(ci[2]),
      std_difference = unname(coef(fit)[2]) / sd_w[[y]],
      t_test_p = unname(tt$p.value)
    )
  }
}

# Education: five groups, design-based Kruskal-Wallis test.
# Effect: weighted difference between the most and the least educated group.
for (y in names(outcomes)) {
  fml <- as.formula(paste(y, "~ education"))
  
  kw <- svyranktest(fml, des_60, test = "KruskalWallis")
  fit <- svyglm(fml, des_60)
  coef_name <- "educationCollege graduate+"
  ci <- confint(fit)[coef_name, ]
  
  rows[[length(rows) + 1]] <- data.frame(
    comparison = "Education: 5 groups",
    outcome = unname(outcomes[y]),
    test = "Design-based Kruskal-Wallis test",
    n = nrow(des_60),
    statistic = unname(kw$statistic),
    p_raw = unname(kw$p.value),
    weighted_mean_A = NA_real_,
    weighted_mean_B = NA_real_,
    mean_difference = unname(coef(fit)[coef_name]),   # college graduate+ vs <9th grade
    ci_low = unname(ci[1]),
    ci_high = unname(ci[2]),
    std_difference = unname(coef(fit)[coef_name]) / sd_w[[y]],
    t_test_p = NA_real_
  )
}

tests <- bind_rows(rows)

# Benjamini-Hochberg FDR correction across all 12 tests
tests$p_FDR <- p.adjust(tests$p_raw, method = "BH")
tests$significant_FDR <- tests$p_FDR < 0.05

write.csv(tests |> mutate(across(where(is.numeric), ~ signif(.x, 4))),
          file.path(tables_dir, "survey_group_comparison_tests.csv"), row.names = FALSE)

cat("\nDesign-based group comparisons of cognitive scores (FDR-corrected, 12 tests):\n")
print(tests |>
        select(comparison, outcome, p_raw, p_FDR, mean_difference, ci_low, ci_high, std_difference) |>
        mutate(across(where(is.numeric), ~ signif(.x, 3))),
      row.names = FALSE)
cat("(For sex, diabetes and smoking: difference = group A - group B in test points.",
    "For education: college graduate+ minus <9th grade.)\n")

# ------------------------------------------------------------
# 4. Family 2: chi-square tests of association (3 tests, Rao-Scott F test)
# ------------------------------------------------------------
cramers_v_weighted <- function(tab) {
  p <- prop.table(tab)
  expected <- outer(rowSums(p), colSums(p))
  phi2 <- sum((p - expected)^2 / expected)
  sqrt(phi2 / (min(dim(p)) - 1))
}

chi_pairs <- list(
  list(label = "Diabetes by education", a = "diabetes", b = "education"),
  list(label = "Diabetes by sex", a = "diabetes", b = "sex"),
  list(label = "Smoking history by sex", a = "smoking", b = "sex")
)

chi_rows <- list()
for (cp in chi_pairs) {
  fml <- as.formula(paste0("~", cp$a, " + ", cp$b))
  ch <- svychisq(fml, des_60, statistic = "F")
  tab <- as.matrix(svytable(fml, des_60))
  
  chi_rows[[length(chi_rows) + 1]] <- data.frame(
    association = cp$label,
    F_statistic = unname(ch$statistic),
    ndf = unname(ch$parameter["ndf"]),
    ddf = unname(ch$parameter["ddf"]),
    p_raw = unname(ch$p.value),
    cramers_v_weighted = cramers_v_weighted(tab)
  )
  
  cat("\n", cp$label, ": weighted % of each column group (columns add up to 100%)\n", sep = "")
  print(round(100 * prop.table(tab, 2), 1))
}

chi_table <- bind_rows(chi_rows)
chi_table$p_FDR <- p.adjust(chi_table$p_raw, method = "BH")
chi_table$significant_FDR <- chi_table$p_FDR < 0.05

write.csv(chi_table |> mutate(across(where(is.numeric), ~ signif(.x, 4))),
          file.path(tables_dir, "survey_chi_square_tests.csv"), row.names = FALSE)
cat("\nDesign-based chi-square tests (Rao-Scott, FDR-corrected, 3 tests):\n")
print(chi_table |> mutate(across(where(is.numeric), ~ signif(.x, 3))), row.names = FALSE)

# ------------------------------------------------------------
# 5. Compare with the Python (unweighted) results
# ------------------------------------------------------------
py_path <- here("results", "tables", "group_comparison_tests.csv")
if (file.exists(py_path)) {
  py <- read.csv(py_path) |>
    select(comparison, outcome, python_p_FDR = p_FDR, python_cohens_d = cohens_d)
  
  comparison_table <- tests |>
    select(comparison, outcome, survey_p_FDR = p_FDR, survey_std_difference = std_difference) |>
    left_join(py, by = c("comparison", "outcome")) |>
    mutate(same_conclusion = (survey_p_FDR < 0.05) == (python_p_FDR < 0.05))
  
  write.csv(comparison_table |> mutate(across(where(is.numeric), ~ signif(.x, 4))),
            file.path(tables_dir, "survey_vs_python_group_tests.csv"), row.names = FALSE)
  cat("\nSurvey-weighted vs unweighted (Python) group comparisons:\n")
  print(comparison_table |> mutate(across(where(is.numeric), ~ signif(.x, 3))), row.names = FALSE)
  cat("\nSame conclusion after FDR correction:", sum(comparison_table$same_conclusion),
      "of", nrow(comparison_table), "tests\n")
} else {
  cat("\nPython results not found; skipping the comparison.\n")
}

cat("\nTables saved in:", tables_dir, "\n")
cat("Design-based group comparisons finished.\n")