"""Model fitting in numpy/scipy, matching R's lm / aov / TukeyHSD / glm(binomial)."""
import itertools
import math
import numpy as np
from scipy import stats


def ols(X, y):
    n, p = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    fitted = X @ b
    res = y - fitted
    df = n - p
    sse = float(res @ res)
    mse = sse / df
    se = np.sqrt(np.diag(XtX_inv) * mse)
    t = b / se
    pv = 2 * stats.t.sf(np.abs(t), df)
    sst = float(((y - y.mean()) ** 2).sum())
    ssr = sst - sse
    r2 = ssr / sst
    adj = 1 - mse / (sst / (n - 1))
    F = (ssr / (p - 1)) / mse
    pF = stats.f.sf(F, p - 1, df)
    return dict(b=b, se=se, t=t, p=pv, res=res, fitted=fitted, df=df, sse=sse, mse=mse, s=math.sqrt(mse),
                sst=sst, ssr=ssr, r2=r2, adj=adj, F=F, pF=pF, n=n, k=p, XtX_inv=XtX_inv)


def predict(fit, x0row, level=0.95):
    x0 = np.asarray(x0row, float)
    yhat = float(x0 @ fit["b"])
    se_m = math.sqrt(float(x0 @ fit["XtX_inv"] @ x0) * fit["mse"])
    se_p = math.sqrt(se_m ** 2 + fit["mse"])
    tc = stats.t.ppf(0.5 + level / 2, fit["df"])
    return dict(fit=yhat, ci=(yhat - tc * se_m, yhat + tc * se_m), pi=(yhat - tc * se_p, yhat + tc * se_p),
                se_m=se_m, se_p=se_p, tc=tc)


def oneway(groups):
    """groups: dict level -> np.array (levels in R order)."""
    levels = list(groups)
    allv = np.concatenate([groups[l] for l in levels])
    N, k = len(allv), len(levels)
    grand = allv.mean()
    means = {l: groups[l].mean() for l in levels}
    ns = {l: len(groups[l]) for l in levels}
    ssb = sum(ns[l] * (means[l] - grand) ** 2 for l in levels)
    ssw = sum(((groups[l] - means[l]) ** 2).sum() for l in levels)
    dfb, dfw = k - 1, N - k
    msb, msw = ssb / dfb, ssw / dfw
    F = msb / msw
    p = stats.f.sf(F, dfb, dfw)
    return dict(levels=levels, means=means, ns=ns, ssb=ssb, ssw=ssw, dfb=dfb, dfw=dfw, msb=msb, msw=msw,
                F=F, p=p, N=N, k=k, grand=grand)


def tukey(ow, level=0.95):
    lv = ow["levels"]
    k, df, mse = ow["k"], ow["dfw"], ow["msw"]
    q = stats.studentized_range.ppf(level, k, df)
    rows = []
    for i, j in itertools.combinations(range(k), 2):
        a, b = lv[i], lv[j]
        diff = ow["means"][b] - ow["means"][a]
        se = math.sqrt(mse / 2 * (1 / ow["ns"][a] + 1 / ow["ns"][b]))
        lwr, upr = diff - q * se, diff + q * se
        padj = float(stats.studentized_range.sf(abs(diff) / se, k, df))
        rows.append(dict(name=f"{b}-{a}", hi=b, lo=a, diff=diff, lwr=lwr, upr=upr, padj=min(1.0, padj)))
    return rows


def _glm_irls(X, y, offset, family, maxit=25, eps=1e-8):
    """Replicates R's glm.fit: start from mustart, stop when |dev-devold|/(|dev|+0.1) < eps.
    Returns coefficients and the iteration count R would print."""
    n, p = X.shape
    off = np.zeros(n) if offset is None else np.asarray(offset, float)
    if family == "binomial":
        mu = (y + 0.5) / 2.0
        eta = np.log(mu / (1 - mu))
        linkinv = lambda e: 1 / (1 + np.exp(-e))
        mu_eta = lambda e, m: m * (1 - m)
        var = lambda m: m * (1 - m)

        def devf(m):
            m = np.clip(m, 1e-15, 1 - 1e-15)
            return float(-2 * np.sum(y * np.log(m) + (1 - y) * np.log(1 - m)))
    else:  # poisson, log link
        mu = y + 0.1
        eta = np.log(mu)
        linkinv = np.exp
        mu_eta = lambda e, m: m
        var = lambda m: m

        def devf(m):
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(y > 0, y * np.log(y / m), 0.0)
            return float(2 * np.sum(t - (y - m)))
    devold = devf(mu)
    b = np.zeros(p)
    it = 0
    for it in range(1, maxit + 1):
        me = mu_eta(eta, mu)
        z = (eta - off) + (y - mu) / me
        w = me ** 2 / var(mu)
        XtW = X.T * w
        b = np.linalg.solve(XtW @ X, XtW @ z)
        eta = X @ b + off
        mu = linkinv(eta)
        dev = devf(mu)
        if abs(dev - devold) / (abs(dev) + 0.1) < eps:
            break
        devold = dev
    me = mu_eta(eta, mu)
    w = me ** 2 / var(mu)
    cov = np.linalg.inv((X.T * w) @ X)
    return b, mu, cov, dev, it


def logistic(X, y, maxit=25):
    n, p = X.shape
    b, mu, cov, res_dev, it = _glm_irls(X, y, None, "binomial", maxit)
    se = np.sqrt(np.diag(cov))
    z = b / se
    pv = 2 * stats.norm.sf(np.abs(z))
    m0 = np.clip(np.full(n, y.mean()), 1e-15, 1 - 1e-15)
    null_dev = float(-2 * np.sum(y * np.log(m0) + (1 - y) * np.log(1 - m0)))
    aic = res_dev + 2 * p
    return dict(b=b, se=se, z=z, p=pv, res_dev=res_dev, null_dev=null_dev, aic=aic, n=n, k=p,
                df_null=n - 1, df_res=n - p, iters=it, mu=mu)


def poisson(X, y, offset=None, maxit=25):
    """glm(..., family = poisson) with optional offset (already on log scale)."""
    n, p = X.shape
    b, mu, cov, res_dev, it = _glm_irls(X, y, offset, "poisson", maxit)
    se = np.sqrt(np.diag(cov))
    z = b / se
    pv = 2 * stats.norm.sf(np.abs(z))
    off = np.zeros(n) if offset is None else np.asarray(offset, float)
    mu0 = np.exp(off) * y.sum() / np.exp(off).sum()  # intercept-only model with the same offset

    def dev(m):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(y > 0, y * np.log(y / m), 0.0)
        return float(2 * np.sum(t - (y - m)))

    null_dev = dev(mu0)
    from scipy.special import gammaln
    ll = float(np.sum(y * np.log(mu) - mu - gammaln(y + 1)))
    aic = -2 * ll + 2 * p
    pearson = float(np.sum((y - mu) ** 2 / mu))
    return dict(b=b, se=se, z=z, p=pv, res_dev=res_dev, null_dev=null_dev, aic=aic, n=n, k=p,
                df_null=n - 1, df_res=n - p, iters=it, mu=mu, pearson=pearson, ll=ll)
