"""Minimal high-dimensional fixed-effects estimators (OLS and Poisson PML).

Fixed effects are partialled out with the method of alternating projections
(Frisch-Waugh-Lovell + Gauss-Seidel demeaning), as in Gaure (2013) / fixest.
PPML uses IRLS with weighted demeaning at every iteration (Correia, Guimaraes
& Zylkin, 2020).  Standard errors are cluster-robust (one-way).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["feols", "fepois", "FEResult"]


# ----------------------------------------------------------------------------
# Demeaning
# ----------------------------------------------------------------------------
def _encode(fe: pd.DataFrame) -> list[np.ndarray]:
    return [pd.factorize(fe[c])[0] for c in fe.columns]


def demean(M: np.ndarray, codes: list[np.ndarray], w: np.ndarray | None = None,
           tol: float = 1e-10, maxiter: int = 10_000) -> np.ndarray:
    """Project the columns of M off the span of the fixed-effect dummies.

    Alternating projections: subtract (weighted) group means for each FE
    dimension in turn until the update is below `tol`.
    """
    M = np.array(M, dtype=float, copy=True)
    if M.ndim == 1:
        M = M[:, None]
    w = np.ones(M.shape[0]) if w is None else w
    sw = [np.bincount(c, weights=w) for c in codes]
    for _ in range(maxiter):
        delta = 0.0
        for c, s in zip(codes, sw):
            means = np.vstack([np.bincount(c, weights=w * M[:, j]) for j in range(M.shape[1])]).T / s[:, None]
            M -= means[c]
            delta = max(delta, np.abs(means).max())
        if delta < tol:
            return M
    raise RuntimeError("demeaning did not converge")


def _drop_singletons(fe_codes: list[np.ndarray]) -> np.ndarray:
    """Iteratively flag observations that are alone in some FE group."""
    keep = np.ones(len(fe_codes[0]), dtype=bool)
    changed = True
    while changed:
        changed = False
        for c in fe_codes:
            counts = np.bincount(c[keep], minlength=c.max() + 1)
            bad = keep & (counts[c] == 1)
            if bad.any():
                keep &= ~bad
                changed = True
    return keep


# ----------------------------------------------------------------------------
# Results container
# ----------------------------------------------------------------------------
@dataclass
class FEResult:
    coef: pd.Series
    se: pd.Series
    nobs: int
    n_clusters: int
    stat: dict

    @property
    def tstat(self) -> pd.Series:
        return self.coef / self.se

    @property
    def pvalue(self) -> pd.Series:
        # t distribution with G-1 dof, as fixest does for clustered SEs
        return pd.Series(2 * stats.t.sf(np.abs(self.tstat), self.n_clusters - 1), index=self.coef.index)

    def table(self) -> pd.DataFrame:
        ci = stats.t.ppf(0.975, self.n_clusters - 1) * self.se
        return pd.DataFrame({"coef": self.coef, "se": self.se, "t": self.tstat,
                             "p": self.pvalue, "ci_low": self.coef - ci, "ci_high": self.coef + ci})


def _cluster_vcov(Xt: np.ndarray, scores: np.ndarray, bread: np.ndarray,
                  cl: np.ndarray, n: int, k: int) -> tuple[np.ndarray, int]:
    G = cl.max() + 1
    S = np.vstack([np.bincount(cl, weights=scores[:, j], minlength=G) for j in range(scores.shape[1])]).T
    meat = S.T @ S
    adj = (G / (G - 1)) * ((n - 1) / (n - k))
    return adj * bread @ meat @ bread, G


def _prepare(df: pd.DataFrame, y: str, x: list[str], fe: list[str], cluster: str):
    d = df[[y, *x, *fe, cluster]].dropna().reset_index(drop=True)
    codes = _encode(d[fe])
    keep = _drop_singletons(codes)
    d = d.loc[keep].reset_index(drop=True)
    return d


# ----------------------------------------------------------------------------
# OLS with HDFE
# ----------------------------------------------------------------------------
def feols(df: pd.DataFrame, y: str, x: list[str], fe: list[str], cluster: str) -> FEResult:
    d = _prepare(df, y, x, fe, cluster)
    codes = _encode(d[fe])
    Z = demean(d[[y, *x]].to_numpy(float), codes)
    yt, Xt = Z[:, 0], Z[:, 1:]
    bread = np.linalg.inv(Xt.T @ Xt)
    beta = bread @ Xt.T @ yt
    e = yt - Xt @ beta
    n = len(d)
    # fixest "nested" convention: FE nested in the cluster variable are not counted in K
    cl = pd.factorize(d[cluster])[0]
    k = len(x) + sum(int(c.max()) for c, f in zip(codes, fe) if not _nested(d[f], d[cluster]))
    V, G = _cluster_vcov(Xt, Xt * e[:, None], bread, cl, n, k)
    # R2 on the original scale (FE fitted values included)
    ybar = d[y].mean()
    tss = ((d[y] - ybar) ** 2).sum()
    rss = (e ** 2).sum()
    k_all = len(x) + sum(int(c.max()) + 0 for c in codes) + 1
    r2 = 1 - rss / tss
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k_all)
    wr2 = 1 - rss / (yt ** 2).sum()
    return FEResult(pd.Series(beta, index=x), pd.Series(np.sqrt(np.diag(V)), index=x), n, G,
                    {"r2": r2, "adj_r2": adj_r2, "within_r2": wr2, "resid": e,
                     "fitted": d[y].to_numpy() - e, "data": d})


def _nested(f: pd.Series, cl: pd.Series) -> bool:
    """True if the FE is nested in the cluster variable: each cluster contains a single FE level
    (e.g. origin and destination FE inside origin-destination pair clusters)."""
    return bool((pd.DataFrame({"f": f.values, "c": cl.values}).groupby("c")["f"].nunique() == 1).all())


# ----------------------------------------------------------------------------
# Poisson pseudo-maximum likelihood with HDFE
# ----------------------------------------------------------------------------
def fepois(df: pd.DataFrame, y: str, x: list[str], fe: list[str], cluster: str,
           tol: float = 1e-9, maxiter: int = 200) -> FEResult:
    d = df[[y, *x, *fe, cluster]].dropna().reset_index(drop=True)
    # Drop FE groups whose outcome is identically zero: their FE -> -inf
    # (no information on beta; this is what fixest/ppmlhdfe do).
    changed = True
    while changed:
        n0 = len(d)
        for f in fe:
            d = d[d.groupby(f)[y].transform("sum") > 0]
        d = d.reset_index(drop=True)
        changed = len(d) < n0
    codes = _encode(d[fe])
    yv = d[y].to_numpy(float)
    X = d[x].to_numpy(float)

    mu = (yv + yv.mean()) / 2
    eta = np.log(mu)
    beta = np.zeros(len(x))
    dev_old = np.inf
    for it in range(maxiter):
        z = eta + (yv - mu) / mu
        Zt = demean(np.column_stack([z, X]), codes, w=mu, tol=1e-11)
        zt, Xt = Zt[:, 0], Zt[:, 1:]
        WX = Xt * mu[:, None]
        beta = np.linalg.solve(Xt.T @ WX, WX.T @ zt)
        resid = zt - Xt @ beta
        eta = z - resid
        mu = np.exp(eta)
        with np.errstate(divide="ignore", invalid="ignore"):
            dev = 2 * np.sum(np.where(yv > 0, yv * np.log(yv / mu), 0) - (yv - mu))
        if abs(dev - dev_old) / (0.1 + abs(dev)) < tol:
            break
        dev_old = dev
    else:
        raise RuntimeError("PPML did not converge")

    bread = np.linalg.inv(Xt.T @ (Xt * mu[:, None]))
    scores = Xt * (yv - mu)[:, None]
    cl = pd.factorize(d[cluster])[0]
    n = len(d)
    k = len(x) + sum(int(c.max()) for c, f in zip(codes, fe) if not _nested(d[f], d[cluster]))
    V, G = _cluster_vcov(Xt, scores, bread, cl, n, k)
    # pseudo R2 = squared correlation between y and fitted mu (Santos Silva & Tenreyro)
    pr2 = np.corrcoef(yv, mu)[0, 1] ** 2
    return FEResult(pd.Series(beta, index=x), pd.Series(np.sqrt(np.diag(V)), index=x), n, G,
                    {"pseudo_r2": pr2, "deviance": dev, "iterations": it + 1, "data": d})
