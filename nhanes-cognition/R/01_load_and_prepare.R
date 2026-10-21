# ============================================================
# Step R-1: read the raw NHANES 2011-2012 files and prepare the data
# for design-based (survey) analysis.
#
# Input:  data/raw/*.XPT
# Output: data/processed/nhanes_2011_2012_survey.rds
# ============================================================

library(here)
library(haven)
library(dplyr)

# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------
# here() points to the project root (the folder with nhanes-cognition.Rproj)
raw_dir <- here("data", "raw")
processed_dir <- here("data", "processed")
dir.create(processed_dir, showWarnings = FALSE, recursive = TRUE)

# ------------------------------------------------------------
# 2. Read the raw files (keep only the variables we need)
# ------------------------------------------------------------
read_nhanes <- function(file, vars) {
  path <- file.path(raw_dir, file)
  if (!file.exists(path)) stop("Missing file: ", path)
  read_xpt(path) |>
    select(all_of(vars)) |>
    mutate(across(everything(), as.numeric))
}

# DEMO_G contains the survey design variables:
#   WTMEC2YR = examination (MEC) sample weight, SDMVSTRA = stratum, SDMVPSU = cluster (PSU)
demo <- read_nhanes("DEMO_G.XPT", c("SEQN", "RIDAGEYR", "RIAGENDR", "DMDEDUC2",
                                    "WTINT2YR", "WTMEC2YR", "SDMVPSU", "SDMVSTRA"))
cfq <- read_nhanes("CFQ_G.XPT", c("SEQN", "CFDDS", "CFDAST", "CFDCSR"))
slq <- read_nhanes("SLQ_G.XPT", c("SEQN", "SLD010H"))
bmx <- read_nhanes("BMX_G.XPT", c("SEQN", "BMXBMI"))
smq <- read_nhanes("SMQ_G.XPT", c("SEQN", "SMQ020"))
diq <- read_nhanes("DIQ_G.XPT", c("SEQN", "DIQ010"))

# ------------------------------------------------------------
# 3. Merge by participant ID
# ------------------------------------------------------------
# left_join keeps ALL participants from DEMO_G. The survey design must be
# defined on the full sample; the subgroup (age 60+) is selected later.
nhanes <- demo |>
  left_join(cfq, by = "SEQN") |>
  left_join(slq, by = "SEQN") |>
  left_join(bmx, by = "SEQN") |>
  left_join(smq, by = "SEQN") |>
  left_join(diq, by = "SEQN")

# ------------------------------------------------------------
# 4. Recode special NHANES codes (refused / don't know) as missing
# ------------------------------------------------------------
nhanes <- nhanes |>
  mutate(
    DMDEDUC2 = if_else(DMDEDUC2 %in% c(7, 9), NA_real_, DMDEDUC2),
    SMQ020   = if_else(SMQ020 %in% c(7, 9), NA_real_, SMQ020),
    DIQ010   = if_else(DIQ010 %in% c(7, 9), NA_real_, DIQ010),
    SLD010H  = if_else(SLD010H %in% c(77, 99), NA_real_, SLD010H)
  )

# ------------------------------------------------------------
# 5. Flag the analysis sample (the same as in the Python analysis)
# ------------------------------------------------------------
needed <- c("CFDDS", "CFDAST", "CFDCSR", "DMDEDUC2", "RIDAGEYR", "RIAGENDR",
            "SLD010H", "BMXBMI", "SMQ020", "DIQ010")

ids_in_all_files <- Reduce(intersect, list(demo$SEQN, cfq$SEQN, slq$SEQN,
                                           bmx$SEQN, smq$SEQN, diq$SEQN))

nhanes <- nhanes |>
  mutate(
    age60 = RIDAGEYR >= 60,
    age60_in_all_files = age60 & SEQN %in% ids_in_all_files,
    analysis_sample = age60 & complete.cases(across(all_of(needed)))
  )

# ------------------------------------------------------------
# 6. Checks: compare with the Python analysis
# ------------------------------------------------------------
cat("All NHANES 2011-2012 participants in DEMO_G:", nrow(nhanes), "\n")
cat("Participants aged 60+ in DEMO_G:", sum(nhanes$age60), "\n")
cat("Aged 60+ and present in all six files (Python: 1687):",
    sum(nhanes$age60_in_all_files), "\n")
cat("Analysis sample, complete data (Python: 1332):",
    sum(nhanes$analysis_sample), "\n")

if (sum(nhanes$analysis_sample) != 1332) {
  stop("The analysis sample differs from the Python analysis (N = 1332). Check the recoding.")
}

analysis <- nhanes |> filter(analysis_sample)

cat("\nEducation levels in the analysis sample (Python: 174, 196, 295, 374, 293):\n")
print(table(analysis$DMDEDUC2))

cat("\nMeans in the analysis sample (Python: CFDDS 45.3806, CFDAST 16.4865, CFDCSR 5.5803):\n")
print(round(colMeans(analysis[c("CFDDS", "CFDAST", "CFDCSR", "RIDAGEYR", "SLD010H", "BMXBMI")]), 4))

# Survey design variables
w <- analysis$WTMEC2YR
if (anyNA(w) || any(w <= 0)) {
  stop("Some participants in the analysis sample have a missing or zero MEC weight.")
}
cat("\nMEC weight in the analysis sample:\n")
print(summary(w))
cat("Sum of weights (population represented by the analysis sample):",
    format(round(sum(w)), big.mark = ","), "\n")
cat("Participants with zero MEC weight (interviewed but not examined):",
    sum(nhanes$WTMEC2YR == 0, na.rm = TRUE), "\n")
cat("Number of strata:", n_distinct(nhanes$SDMVSTRA), "\n")
cat("Participants by PSU (cluster) number:\n")
print(table(nhanes$SDMVPSU))

# ------------------------------------------------------------
# 7. Save the prepared data (all participants, with the analysis_sample flag)
# ------------------------------------------------------------
saveRDS(nhanes, file.path(processed_dir, "nhanes_2011_2012_survey.rds"))
cat("\nSaved:", file.path(processed_dir, "nhanes_2011_2012_survey.rds"), "\n")
cat("Data preparation finished.\n")