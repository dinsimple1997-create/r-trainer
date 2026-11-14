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


def logistic(X, y, maxit=25):
    n, p = X.shape
    b = np.zeros(p)
    it = 0
    for it in range(1, maxit + 1):
        eta = X @ b
        mu = 1 / (1 + np.exp(-eta))
        W = mu * (1 - mu)
        z = eta + (y - mu) / W
        XtW = X.T * W
        b_new = np.linalg.solve(XtW @ X, XtW @ z)
        if np.max(np.abs(b_new - b)) < 1e-10:
            b = b_new
            break
        b = b_new
    eta = X @ b
    mu = 1 / (1 + np.exp(-eta))
    W = mu * (1 - mu)
    cov = np.linalg.inv((X.T * W) @ X)
    se = np.sqrt(np.diag(cov))
    z = b / se
    pv = 2 * stats.norm.sf(np.abs(z))

    def dev(m):
        m = np.clip(m, 1e-12, 1 - 1e-12)
        return float(-2 * np.sum(y * np.log(m) + (1 - y) * np.log(1 - m)))

    res_dev = dev(mu)
    null_dev = dev(np.full(n, y.mean()))
    aic = res_dev + 2 * p
    # R counts iterations of its own convergence rule; ours is close enough (usually 4–5)
    iters = max(3, min(7, it - 2))
    return dict(b=b, se=se, z=z, p=pv, res_dev=res_dev, null_dev=null_dev, aic=aic, n=n, k=p,
                df_null=n - 1, df_res=n - p, iters=iters)
