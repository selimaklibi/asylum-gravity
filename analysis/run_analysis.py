"""Full pipeline: data preparation, descriptive statistics, five gravity specifications, figures.

    python analysis/run_analysis.py            (expects data/Asylum_data.csv, see data/README.md)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hdfe import feols, fepois  # noqa: E402

DATA, RES, FIG = ROOT / "data" / "Asylum_data.csv", ROOT / "results", ROOT / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

C = {"blue": "#2a78d6", "orange": "#eb6834", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
     "grid": "#e6e5e1"}
plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": C["muted"], "axes.labelcolor": C["ink2"],
    "xtick.color": C["muted"], "ytick.color": C["muted"], "axes.grid": True, "grid.color": C["grid"],
    "grid.linewidth": 0.6, "axes.titleweight": "bold", "axes.titlesize": 11, "legend.frameon": False,
    "axes.axisbelow": True,
})

LABELS = {
    "log_diasp": "log diaspora stock (2010)", "log_distcap": "log distance between capitals",
    "comlang_off": "common official language", "colony": "colonial link", "contig": "contiguity",
    "log_gdpcap_d": "log GDP per capita (dest.)", "hdi_d": "HDI (dest.)",
    "politic_j": "political index (original, report)", "repress_j": "repression index (corrected)",
}


# ----------------------------------------------------------------------------
def load() -> pd.DataFrame:
    d = pd.read_csv(DATA)
    d = d[d.number_app_tot.notna()].copy()
    d["log_app"] = np.log1p(d.number_app_tot)
    d["log_diasp"] = np.log1p(d.diasp_tot_10)
    d["log_distcap"] = np.log(d.distcap)
    d["log_gdpcap_d"] = np.log(d.gdpcap_d)
    d["pair"] = d.cname_o + "|" + d.cname_d
    d["orig_year"] = d.cname_o + "|" + d.year.astype(str)
    d["dest_year"] = d.cname_d + "|" + d.year.astype(str)
    # Report's index: plain mean of four indicators. fh_polity2_o is coded 0-10 with 10 = most
    # democratic, the other three increase with repression, so the plain mean mixes directions.
    d["politic_j"] = d[["fh_civlib_o", "fh_polity2_o", "fh_polrights_o", "ptscale_o"]].mean(axis=1)
    # Corrected index: each component rescaled to [0, 1], oriented so that 1 = most repressive.
    d["repress_j"] = pd.concat([(d.fh_civlib_o - 1) / 6, (d.fh_polrights_o - 1) / 6,
                                (10 - d.fh_polity2_o) / 10, (d.ptscale_o - 1) / 4], axis=1).mean(axis=1)
    return d


def describe(d: pd.DataFrame) -> pd.DataFrame:
    cols = {"number_app_tot": "asylum applications (pair-year)", "diasp_tot_10": "diaspora stock 2010",
            "distcap": "distance between capitals (km)", "gdpcap_d": "GDP per capita, destination (USD)",
            "hdi_d": "HDI, destination", "repress_j": "repression index (corrected, 0-1)",
            "comlang_off": "common official language", "colony": "colonial link", "contig": "contiguity"}
    t = d[list(cols)].describe().T[["count", "mean", "std", "min", "50%", "max"]]
    t.index = list(cols.values())
    t.loc["share of zero flows"] = [len(d), (d.number_app_tot == 0).mean(), np.nan, np.nan, np.nan, np.nan]
    return t


# ----------------------------------------------------------------------------
def specifications(d: pd.DataFrame):
    FE3 = ["year", "cname_o", "cname_d"]
    base = ["log_diasp", "log_distcap", "comlang_off", "colony", "contig", "log_gdpcap_d", "hdi_d"]
    bilateral = base[:5]
    no_hdi = [v for v in base if v != "hdi_d"]
    return {
        "(1) OLS, report": (feols, "log_app", base + ["politic_j"], FE3, d),
        "(2) OLS, fixed index": (feols, "log_app", base + ["repress_j"], FE3, d),
        "(3) PPML": (fepois, "number_app_tot", base + ["repress_j"], FE3, d),
        "(4) PPML, 2010-15, no HDI": (fepois, "number_app_tot", no_hdi + ["repress_j"], FE3, d),
        "(5) PPML, structural": (fepois, "number_app_tot", bilateral, ["orig_year", "dest_year"], d),
    }


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def regression_table(results: dict) -> str:
    rows = list(dict.fromkeys(v for r in results.values() for v in r.coef.index))
    head = "| | " + " | ".join(results) + " |\n|---|" + "---|" * len(results) + "\n"
    body = ""
    for v in rows:
        cells, ses = [], []
        for r in results.values():
            if v in r.coef.index:
                cells.append(f"{r.coef[v]:.3f}{stars(r.pvalue[v])}")
                ses.append(f"({r.se[v]:.3f})")
            else:
                cells.append("")
                ses.append("")
        body += f"| {LABELS[v]} | " + " | ".join(cells) + " |\n| | " + " | ".join(ses) + " |\n"
    fe = {"(5) PPML, structural": "origin×year, dest.×year"}
    body += "| Fixed effects | " + " | ".join(fe.get(k, "origin, dest., year") for k in results) + " |\n"
    body += "| Observations | " + " | ".join(f"{r.nobs:,}" for r in results.values()) + " |\n"
    body += "| Pair clusters | " + " | ".join(f"{r.n_clusters:,}" for r in results.values()) + " |\n"
    fit = [f"adj. R² {r.stat['adj_r2']:.3f}" if "adj_r2" in r.stat else f"pseudo-R² {r.stat['pseudo_r2']:.3f}"
           for r in results.values()]
    body += "| Fit | " + " | ".join(fit) + " |\n"
    return head + body


# ----------------------------------------------------------------------------
def fig_flows(d):
    by_year = d.groupby("year").number_app_tot.sum() / 1e6
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(by_year.index, by_year.values, color=C["blue"], width=0.7)
    for x, y in by_year.items():
        ax.text(x, y + 0.02, f"{y:.2f}", ha="center", fontsize=8.5, color=C["ink2"])
    ax.set(ylabel="applications (millions)", title="Asylum applications to 28 European countries")
    ax.grid(axis="x", visible=False)
    fig.savefig(FIG / "applications_by_year.png")


def fig_residuals(r):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.scatter(r.stat["fitted"], r.stat["resid"], s=3, alpha=0.15, color=C["blue"], linewidths=0)
    ax.axhline(0, color=C["orange"], lw=1.2, ls="--")
    ax.set(xlabel="fitted log(1 + applications)", ylabel="within residual",
           title="OLS on log(1+y): zeros (82% of pairs) create residual bands")
    fig.savefig(FIG / "ols_residuals.png")


def fig_coefficients(ols, ppml):
    keys = ["log_diasp", "comlang_off", "colony", "contig", "log_distcap"]
    y = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for res, off, col, lab in [(ols, 0.15, C["blue"], "OLS on log(1+y)  (2)"),
                               (ppml, -0.15, C["orange"], "PPML  (3)")]:
        t = res.table().loc[keys]
        ax.errorbar(t.coef, y + off, xerr=[t.coef - t.ci_low, t.ci_high - t.coef], fmt="o", ms=6,
                    color=col, ecolor=col, elinewidth=1.5, capsize=0, label=lab)
    ax.axvline(0, color=C["muted"], lw=1)
    ax.set_yticks(y, [LABELS[k] for k in keys])
    ax.invert_yaxis()
    ax.set(xlabel="coefficient (95% CI, pair-clustered)",
           title="Same data, same fixed effects: contiguity flips sign under PPML")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.savefig(FIG / "ols_vs_ppml.png")


# ----------------------------------------------------------------------------
def main():
    d = load()
    desc = describe(d)
    desc.round(3).to_csv(RES / "descriptive_statistics.csv")
    results = {}
    for name, (est, y, x, fe, data) in specifications(d).items():
        results[name] = est(data, y, x, fe, "pair")
        results[name].table().round(4).to_csv(RES / f"spec{name[1]}.csv")
    table = regression_table(results)
    (RES / "regressions.md").write_text(
        table + "\nPair-clustered standard errors in parentheses. *** p<0.01, ** p<0.05, * p<0.10.\n")
    print(desc.round(3).to_string(), "\n")
    print(table)
    fig_flows(d)
    fig_residuals(results["(2) OLS, fixed index"])
    fig_coefficients(results["(2) OLS, fixed index"], results["(3) PPML"])
    corr = d[["fh_civlib_o", "fh_polity2_o", "fh_polrights_o", "ptscale_o"]].corr().round(2)
    print("Correlation of the index components:\n", corr)


if __name__ == "__main__":
    main()
