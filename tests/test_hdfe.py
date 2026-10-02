"""Validate the HDFE estimators against brute-force dummy-variable estimation on simulated panels."""
import numpy as np
import pandas as pd
import pytest

from hdfe import feols, fepois, demean


def _panel(seed=0, n_o=15, n_d=8, n_t=5, beta=(0.5, -0.3)):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([range(n_o), range(n_d), range(n_t)], names=["o", "d", "t"])
    df = idx.to_frame(index=False)
    a, g, th = rng.normal(size=n_o), rng.normal(size=n_d), rng.normal(size=n_t)
    df["x1"] = rng.normal(size=len(df)) + a[df.o]          # correlated with the FE
    df["x2"] = rng.normal(size=len(df)) + g[df.d]
    eta = beta[0] * df.x1 + beta[1] * df.x2 + a[df.o] + g[df.d] + th[df.t]
    df["y_lin"] = eta + rng.normal(size=len(df))
    df["y_cnt"] = rng.poisson(np.exp(eta - 1))
    df["pair"] = df.o.astype(str) + "_" + df.d.astype(str)
    return df


def _dummies(df):
    return pd.get_dummies(df[["o", "d", "t"]].astype(str), drop_first=True).to_numpy(float)


def test_demeaning_is_a_projection():
    df = _panel()
    codes = [pd.factorize(df[c])[0] for c in ["o", "d", "t"]]
    M = demean(df[["x1"]].to_numpy(), codes)
    D = np.column_stack([np.ones(len(df)), _dummies(df)])
    resid = df.x1.to_numpy() - D @ np.linalg.lstsq(D, df.x1.to_numpy(), rcond=None)[0]
    np.testing.assert_allclose(M[:, 0], resid, atol=1e-8)


def test_feols_equals_dummy_ols_and_clustered_se():
    df = _panel()
    r = feols(df, "y_lin", ["x1", "x2"], ["o", "d", "t"], "pair")
    X = np.column_stack([df[["x1", "x2"]].to_numpy(), np.ones(len(df)), _dummies(df)])
    b = np.linalg.lstsq(X, df.y_lin.to_numpy(), rcond=None)[0]
    np.testing.assert_allclose(r.coef.values, b[:2], atol=1e-8)
    # cluster-robust sandwich on the full dummy model (same small-sample factor conventions)
    e = df.y_lin.to_numpy() - X @ b
    bread = np.linalg.pinv(X.T @ X)
    cl = pd.factorize(df.pair)[0]
    S = np.vstack([np.bincount(cl, weights=X[:, j] * e) for j in range(X.shape[1])]).T
    V = bread @ (S.T @ S) @ bread
    G, n = cl.max() + 1, len(df)
    k = 2 + (df.t.nunique() - 1)  # o and d FE are nested in pairs (fixest "nested" rule)
    V *= G / (G - 1) * (n - 1) / (n - k)
    np.testing.assert_allclose(r.se.values, np.sqrt(np.diag(V))[:2], rtol=1e-6)


def _poisson_newton(X, y, iters=100):
    b = np.zeros(X.shape[1])
    b[0] = np.log(y.mean())
    for _ in range(iters):
        mu = np.exp(X @ b)
        step = np.linalg.solve(X.T @ (X * mu[:, None]), X.T @ (y - mu))
        b += step
        if np.abs(step).max() < 1e-12:
            break
    return b


def test_fepois_equals_dummy_poisson():
    df = _panel(seed=3)
    r = fepois(df, "y_cnt", ["x1", "x2"], ["o", "d", "t"], "pair")
    d = r.stat["data"]
    X = np.column_stack([np.ones(len(d)), d[["x1", "x2"]].to_numpy(), _dummies(d)])
    b = _poisson_newton(X, d.y_cnt.to_numpy(float))
    np.testing.assert_allclose(r.coef.values, b[1:3], atol=1e-6)


def test_fepois_is_consistent_with_many_zeros():
    rng_betas = []
    for s in range(5):
        df = _panel(seed=10 + s, n_o=40, n_d=20, n_t=6)
        rng_betas.append(fepois(df, "y_cnt", ["x1", "x2"], ["o", "d", "t"], "pair").coef.values)
    np.testing.assert_allclose(np.mean(rng_betas, axis=0), [0.5, -0.3], atol=0.03)


def test_log1p_ols_is_biased_under_poisson_dgp():
    """The motivating fact for PPML (Santos Silva & Tenreyro, 2006)."""
    df = _panel(seed=1, n_o=40, n_d=20, n_t=6)
    df["log1p"] = np.log1p(df.y_cnt)
    b = feols(df, "log1p", ["x1", "x2"], ["o", "d", "t"], "pair").coef.values
    assert abs(b[0] - 0.5) > 0.1
