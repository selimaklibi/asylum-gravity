# ==============================================================================
# Determinants of asylum destination choice in Europe: gravity model
# R / fixest version of analysis/run_analysis.py (same five specifications).
# Run from the repository root:  Rscript R/gravity_fixest.R
# ==============================================================================
library(data.table)
library(fixest)

d <- fread("data/Asylum_data.csv")
d <- d[!is.na(number_app_tot)]
d[, `:=`(
  log_app      = log1p(number_app_tot),
  log_diasp    = log1p(diasp_tot_10),
  log_distcap  = log(distcap),
  log_gdpcap_d = log(gdpcap_d),
  pair         = paste(cname_o, cname_d, sep = "|")
)]

# Index used in the original report: plain mean of the four indicators.
# Caveat: fh_polity2_o is 0-10 with 10 = most democratic, the three others
# increase with repression (correlation polity2 / political rights = -0.96).
d[, politic_j := rowMeans(.SD, na.rm = TRUE),
  .SDcols = c("fh_civlib_o", "fh_polity2_o", "fh_polrights_o", "ptscale_o")]

# Corrected index: components rescaled to [0, 1], 1 = most repressive.
d[, repress_j := rowMeans(cbind((fh_civlib_o - 1) / 6, (fh_polrights_o - 1) / 6,
                                (10 - fh_polity2_o) / 10, (ptscale_o - 1) / 4), na.rm = TRUE)]

base <- "log_diasp + log_distcap + comlang_off + colony + contig + log_gdpcap_d + hdi_d"
f <- function(lhs, rhs, fe) as.formula(paste(lhs, "~", rhs, "|", fe))
fe3 <- "year + cname_o + cname_d"

m1 <- feols(f("log_app", paste(base, "+ politic_j"), fe3), d, cluster = ~pair)
m2 <- feols(f("log_app", paste(base, "+ repress_j"), fe3), d, cluster = ~pair)
m3 <- fepois(f("number_app_tot", paste(base, "+ repress_j"), fe3), d, cluster = ~pair)
m4 <- fepois(f("number_app_tot",
               "log_diasp + log_distcap + comlang_off + colony + contig + log_gdpcap_d + repress_j", fe3),
             d, cluster = ~pair)
m5 <- fepois(number_app_tot ~ log_diasp + log_distcap + comlang_off + colony + contig |
               cname_o^year + cname_d^year, d, cluster = ~pair)

etable(m1, m2, m3, m4, m5,
       headers = c("OLS report", "OLS fixed index", "PPML", "PPML no HDI", "PPML structural"),
       signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10))
