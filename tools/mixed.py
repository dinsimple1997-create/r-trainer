"""Random-intercept mixed models fitted the way R does it.

lmm():  lme4::lmer(REML = TRUE) + lmerTest Satterthwaite df.
glmm(): lme4::glmer(nAGQ = 10) for binomial (0/1) and poisson (log link, optional offset).
Only one grouping factor with a random intercept, which is all the course uses.
"""
import math
import numpy as np
from scipy import optimize, stats, special

GH_X, GH_W = np.polynomial.hermite.hermgauss(10)


# ---------------------------------------------------------------------------
# helpers
def _groups(g):
    labs, idx = np.unique(g, return_inverse=True)
    return labs, idx, [np.where(idx == k)[0] for k in range(len(labs))]


def _num_hess(f, x, h=None):
    x = np.asarray(x, float)
    k = len(x)
    h = h if h is not None else np.maximum(1e-4, 1e-4 * np.abs(x))
    H = np.zeros((k, k))
    f0 = f(x)
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ei[i] = h[i]
            ej = np.zeros(k); ej[j] = h[j]
            if i == j:
                H[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / h[i] ** 2
            else:
                H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h[i] * h[j])
    return H


def _num_grad(f, x, h=None):
    x = np.asarray(x, float)
    h = h if h is not None else np.maximum(1e-5, 1e-5 * np.abs(x))
    g = np.zeros(len(x))
    for i in range(len(x)):
        e = np.zeros(len(x)); e[i] = h[i]
        g[i] = (f(x + e) - f(x - e)) / (2 * h[i])
    return g


# ---------------------------------------------------------------------------
# LMM, REML
class _LMMparts:
    """Quantities for V0 = I + gamma * Z Z' using the block structure (one block per group)."""

    def __init__(self, y, X, idx_list):
        self.y, self.X, self.G = y, X, idx_list
        self.n, self.p = X.shape

    def solve(self, gamma):
        XtVX = np.zeros((self.p, self.p))
        XtVy = np.zeros(self.p)
        logdetV = 0.0
        for ix in self.G:
            m = len(ix)
            c = gamma / (1 + gamma * m)
            Xi, yi = self.X[ix], self.y[ix]
            sx, sy = Xi.sum(0), yi.sum()
            XtVX += Xi.T @ Xi - c * np.outer(sx, sx)
            XtVy += Xi.T @ yi - c * sx * sy
            logdetV += math.log1p(gamma * m)
        beta = np.linalg.solve(XtVX, XtVy)
        r = self.y - self.X @ beta
        pwrss = 0.0
        for ix in self.G:
            m = len(ix)
            c = gamma / (1 + gamma * m)
            ri = r[ix]
            pwrss += ri @ ri - c * ri.sum() ** 2
        return beta, XtVX, logdetV, pwrss, r

    def reml_dev(self, theta):
        gamma = theta * theta
        beta, XtVX, logdetV, pwrss, r = self.solve(gamma)
        nmp = self.n - self.p
        return logdetV + np.linalg.slogdet(XtVX)[1] + nmp * (1 + math.log(2 * math.pi * pwrss / nmp))

    def reml_dev_full(self, theta, sigma):
        """-2 REML log-likelihood as a function of (theta, sigma), beta profiled out."""
        gamma = theta * theta
        beta, XtVX, logdetV, pwrss, r = self.solve(gamma)
        nmp = self.n - self.p
        return nmp * math.log(2 * math.pi * sigma ** 2) + logdetV + np.linalg.slogdet(XtVX)[1] + pwrss / sigma ** 2

    def vcov(self, theta, sigma):
        beta, XtVX, *_ = self.solve(theta * theta)
        return sigma ** 2 * np.linalg.inv(XtVX)


def lmm(y, X, g, names):
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    labs, idx, G = _groups(g)
    P = _LMMparts(y, X, G)
    res = optimize.minimize_scalar(P.reml_dev, bounds=(0.0, 20.0), method="bounded", options=dict(xatol=1e-10))
    th = float(res.x)
    # polish near the boundary / optimum
    grid = np.linspace(max(0, th - 1e-3), th + 1e-3, 21)
    th = float(grid[np.argmin([P.reml_dev(t) for t in grid])])
    th = float(optimize.minimize_scalar(P.reml_dev, bounds=(max(0, th - 1e-3), th + 1e-3), method="bounded",
                                        options=dict(xatol=1e-12)).x)
    beta, XtVX, logdetV, pwrss, r = P.solve(th * th)
    nmp = P.n - P.p
    sigma = math.sqrt(pwrss / nmp)
    crit = P.reml_dev(th)
    tau = th * sigma
    V = sigma ** 2 * np.linalg.inv(XtVX)
    se = np.sqrt(np.diag(V))
    # BLUPs and conditional residuals
    u = np.zeros(len(G))
    for k, ix in enumerate(G):
        m = len(ix)
        u[k] = th * th * r[ix].sum() / (1 + th * th * m)
    cres = r - u[idx]
    # Satterthwaite (lmerTest): varpar = (theta, sigma)
    vp = np.array([th, sigma])
    Hm = _num_hess(lambda v: P.reml_dev_full(abs(v[0]), abs(v[1])), vp, h=np.array([1e-4 * max(1, th), 1e-4 * sigma]))
    try:
        A = 2 * np.linalg.inv(Hm)
    except np.linalg.LinAlgError:
        A = None

    def satt(L):
        L = np.asarray(L, float)
        phi = float(L @ V @ L)
        if A is None or th < 1e-6:
            return float(P.n - P.p)
        gr = _num_grad(lambda v: float(L @ P.vcov(abs(v[0]), abs(v[1])) @ L), vp,
                       h=np.array([1e-5 * max(1, th), 1e-5 * sigma]))
        den = float(gr @ A @ gr)
        return 2 * phi ** 2 / den if den > 0 else float("inf")

    def snap(d):
        return float(round(d)) if abs(d - round(d)) < 2e-3 else d

    _satt = satt
    satt = lambda L: snap(_satt(L))
    df = np.array([satt(np.eye(P.p)[j]) for j in range(P.p)])
    t = beta / se
    pv = 2 * stats.t.sf(np.abs(t), df)
    return dict(b=beta, se=se, df=df, t=t, p=pv, V=V, sigma=sigma, tau=tau, theta=th, crit=crit,
                scaled=cres / sigma, u=u, groups=labs, idx=idx, n=P.n, ngroups=len(G), satt=satt,
                names=names, fitted=y - cres, resid=cres)


# ---------------------------------------------------------------------------
# GLMM, adaptive Gauss-Hermite quadrature
def _loglik_obs(fam, y, eta, lgam):
    if fam == "binomial":
        # log p(y) for 0/1
        return y * eta - np.logaddexp(0, eta)
    mu = np.exp(eta)
    return y * eta - mu - lgam


def _mu(fam, eta):
    return special.expit(eta) if fam == "binomial" else np.exp(eta)


class _GLMMparts:
    def __init__(self, y, X, idx_list, fam, offset):
        self.y, self.X, self.G, self.fam = y, X, idx_list, fam
        self.off = offset if offset is not None else np.zeros(len(y))
        self.lgam = special.gammaln(y + 1) if fam == "poisson" else 0 * y

    def group_ll(self, theta, beta, return_modes=False):
        eta0 = self.X @ beta + self.off
        total = 0.0
        modes = np.zeros(len(self.G))
        for k, ix in enumerate(self.G):
            yi, ei, lg = self.y[ix], eta0[ix], (self.lgam[ix] if self.fam == "poisson" else 0)
            # h(v) = sum log p(y | ei + theta v) - v^2/2 - log(2 pi)/2
            v = 0.0
            for _ in range(50):
                eta = ei + theta * v
                mu = _mu(self.fam, eta)
                w = mu * (1 - mu) if self.fam == "binomial" else mu
                g1 = theta * np.sum(yi - mu) - v
                g2 = -theta * theta * np.sum(w) - 1
                step = g1 / g2
                v -= step
                if abs(step) < 1e-11:
                    break
            eta = ei + theta * v
            mu = _mu(self.fam, eta)
            w = mu * (1 - mu) if self.fam == "binomial" else mu
            s = 1 / math.sqrt(theta * theta * np.sum(w) + 1)
            nodes = v + math.sqrt(2) * s * GH_X
            hv = np.array([np.sum(_loglik_obs(self.fam, yi, ei + theta * z, lg)) for z in nodes]) - nodes ** 2 / 2 - 0.5 * math.log(2 * math.pi)
            mx = hv.max()
            total += mx + math.log(math.sqrt(2) * s * np.sum(GH_W * np.exp(GH_X ** 2) * np.exp(hv - mx)))
            modes[k] = v
        return (total, modes) if return_modes else total

    def dev(self, par):
        return -2 * self.group_ll(abs(par[0]), np.asarray(par[1:]))


def _glm_start(y, X, fam, off):
    beta = np.zeros(X.shape[1])
    if fam == "binomial":
        beta[0] = special.logit(min(max(y.mean(), 0.02), 0.98))
    else:
        beta[0] = math.log(max(y.sum(), 0.5) / np.exp(off).sum())
    for _ in range(50):
        eta = X @ beta + off
        mu = _mu(fam, eta)
        w = mu * (1 - mu) if fam == "binomial" else mu
        z = eta - off + (y - mu) / w
        beta_new = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (w * z))
        if np.max(np.abs(beta_new - beta)) < 1e-10:
            beta = beta_new
            break
        beta = beta_new
    return beta


def glmm(y, X, g, names, fam="binomial", offset=None):
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, float)
    labs, idx, G = _groups(g)
    P = _GLMMparts(y, X, G, fam, off)
    b0 = _glm_start(y, X, fam, off)
    start = np.r_[1.0, b0]
    best = None
    for th0 in (0.5, 1.0, 1.5):
        st = start.copy(); st[0] = th0
        r = optimize.minimize(P.dev, st, method="Nelder-Mead",
                              options=dict(xatol=1e-9, fatol=1e-10, maxiter=20000, maxfev=40000))
        r = optimize.minimize(P.dev, r.x, method="BFGS", options=dict(gtol=1e-8))
        if best is None or r.fun < best.fun:
            best = r
    par = best.x.copy()
    par[0] = abs(par[0])
    dev = P.dev(par)
    H = _num_hess(P.dev, par, h=np.maximum(1e-4, 1e-4 * np.abs(par)))
    try:
        Vfull = np.linalg.inv(H / 2)
    except np.linalg.LinAlgError:
        raise ValueError("singular Hessian")
    V = Vfull[1:, 1:]
    if not np.all(np.isfinite(V)) or np.any(np.diag(V) <= 0) or np.any(np.abs(par[1:]) > 25):
        raise ValueError("bad fit")
    beta = par[1:]
    se = np.sqrt(np.diag(V))
    z = beta / se
    pv = 2 * stats.norm.sf(np.abs(z))
    ll, modes = P.group_ll(par[0], beta, return_modes=True)
    eta = X @ beta + off + par[0] * modes[idx]
    mu = _mu(fam, eta)
    var = mu * (1 - mu) if fam == "binomial" else mu
    pearson = (y - mu) / np.sqrt(var)
    k = X.shape[1] + 1
    n = len(y)
    # plain glm for comparison
    gl = _glm_fit(y, X, fam, off)
    return dict(b=beta, se=se, z=z, p=pv, V=V, theta=par[0], tau=par[0], var=par[0] ** 2, dev=dev, ll=-dev / 2,
                aic=dev + 2 * k, bic=dev + math.log(n) * k, df_res=n - k, n=n, ngroups=len(G), scaled=pearson,
                u=par[0] * modes, groups=labs, idx=idx, names=names, glm=gl, mu=mu)


def _glm_fit(y, X, fam, off):
    beta = _glm_start(y, X, fam, off)
    eta = X @ beta + off
    mu = _mu(fam, eta)
    w = mu * (1 - mu) if fam == "binomial" else mu
    V = np.linalg.inv(X.T @ (X * w[:, None]))
    se = np.sqrt(np.diag(V))
    ll = float(np.sum(_loglik_obs(fam, y, eta, special.gammaln(y + 1) if fam == "poisson" else 0)))
    if fam == "binomial":
        dev = -2 * ll
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(y > 0, y * np.log(y / mu), 0.0)
        dev = 2 * float(np.sum(t - (y - mu)))
    pear = float(np.sum((y - mu) ** 2 / w))
    z = beta / se
    return dict(b=beta, se=se, z=z, p=2 * stats.norm.sf(np.abs(z)), ll=ll, aic=-2 * ll + 2 * X.shape[1],
                dev=dev, df_res=len(y) - X.shape[1], pearson=pear)
