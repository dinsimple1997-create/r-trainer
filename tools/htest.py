"""Classical tests computed like R and printed like print.htest."""
import math
import numpy as np
from scipy import stats, optimize
import rfmt as R

RELERR = 1 + 1e-7


# ---------------------------------------------------------------------------
# printing
def fmt5(x):
    """format(x, digits = 5): statistic and parameter lines."""
    if abs(x - round(x)) < 1e-9 and abs(x) < 1e6:
        return str(int(round(x)))
    return R.fmt1(x, 5)


def pval(p):
    """format.pval(p, digits = 4) for one value, as print.htest shows it: fixed unless scientific is shorter."""
    if p < R.EPS:
        return "< 2.2e-16"
    fx = R.fmt1(p, 4)
    sc = R._sci_joint([p], 4)[0]
    return fx if len(fx) <= len(sc) else sc


def fmt_ci(lo, hi):
    a, b = R.fmt_joint([lo, hi], 7)
    w = max(len(a), len(b))
    return " " + a.rjust(w) + " " + b.rjust(w)


def htest(method, data, stat_pairs, p, alt, ci=None, conf=95, est=None):
    """stat_pairs: [(name, value)], est: [(name, value)]"""
    out = ["", "\t" + method, "", "data:  " + data]
    parts = [f"{n} = {fmt5(v)}" for n, v in stat_pairs]
    if p is not None:
        fp = pval(p)
        parts.append("p-value " + (fp if fp.startswith("<") else "= " + fp))
    line = ", ".join(parts)
    # strwrap at width ~72 (only very long lines wrap)
    if len(line) > 72:
        words, cur, lines = line.split(" "), "", []
        for w in words:
            if cur and len(cur) + 1 + len(w) >= 72:
                lines.append(cur)
                cur = w
            else:
                cur = (cur + " " + w) if cur else w
        lines.append(cur)
        out += lines
    else:
        out.append(line)
    out.append("alternative hypothesis: " + alt)
    if ci is not None:
        out.append(f"{conf:g} percent confidence interval:")
        out.append(fmt_ci(*ci))
    if est:
        out.append("sample estimates:")
        out += R.named_vector([n for n, _ in est], R.fmt_joint([v for _, v in est], 7))
    out.append("")
    return out


# ---------------------------------------------------------------------------
# t tests
def t_two(x, y, var_equal, conf=0.95):
    n1, n2 = len(x), len(y)
    m1, m2 = float(np.mean(x)), float(np.mean(y))
    v1, v2 = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
    if var_equal:
        df = n1 + n2 - 2
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
        se = math.sqrt(sp2 * (1 / n1 + 1 / n2))
    else:
        a, b = v1 / n1, v2 / n2
        se = math.sqrt(a + b)
        df = (a + b) ** 2 / (a ** 2 / (n1 - 1) + b ** 2 / (n2 - 1))
    t = (m1 - m2) / se
    p = 2 * stats.t.sf(abs(t), df)
    q = stats.t.ppf(1 - (1 - conf) / 2, df)
    return dict(t=t, df=df, p=p, ci=(m1 - m2 - q * se, m1 - m2 + q * se), m1=m1, m2=m2, se=se,
                s1=math.sqrt(v1), s2=math.sqrt(v2), n1=n1, n2=n2)


def t_one(x, mu=0.0, conf=0.95):
    n = len(x)
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    se = s / math.sqrt(n)
    t = (m - mu) / se
    df = n - 1
    p = 2 * stats.t.sf(abs(t), df)
    q = stats.t.ppf(1 - (1 - conf) / 2, df)
    return dict(t=t, df=df, p=p, ci=(m - q * se, m + q * se), m=m, s=s, se=se, n=n)


# ---------------------------------------------------------------------------
# Wilcoxon
def _wilcox_counts(m, n):
    """number of ways to pick m of the ranks 1..m+n with each value of U = ranksum - m(m+1)/2."""
    N = m + n
    maxs = sum(range(N - m + 1, N + 1))
    dp = [[0] * (maxs + 1) for _ in range(m + 1)]
    dp[0][0] = 1
    for r in range(1, N + 1):
        for k in range(min(r, m), 0, -1):
            row, prev = dp[k], dp[k - 1]
            for s_ in range(maxs, r - 1, -1):
                if prev[s_ - r]:
                    row[s_] += prev[s_ - r]
    base = m * (m + 1) // 2
    return [dp[m][u + base] if u + base <= maxs else 0 for u in range(m * n + 1)]


def pwilcox(q, m, n, upper=False):
    cnt = _wilcox_counts(m, n)
    tot = math.comb(m + n, n)
    q = int(math.floor(q + 1e-7))
    if upper:
        return sum(cnt[q + 1:]) / tot
    return sum(cnt[:q + 1]) / tot


def rank_sum(x, y, exact=None, correct=True):
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    r = stats.rankdata(np.r_[x, y])
    W = float(r[:nx].sum() - nx * (nx + 1) / 2)
    ties = len(np.unique(np.r_[x, y])) < nx + ny
    if exact is None:
        exact = nx < 50 and ny < 50
    warn = False
    if exact and not ties:
        if W > nx * ny / 2:
            p = pwilcox(W - 1, nx, ny, upper=True)
        else:
            p = pwilcox(W, nx, ny)
        p = min(2 * p, 1.0)
        method = "Wilcoxon rank sum exact test"
    else:
        if exact and ties:
            warn = True
        _, nt = np.unique(np.r_[x, y], return_counts=True)
        N = nx + ny
        z = W - nx * ny / 2
        sigma = math.sqrt((nx * ny / 12) * ((N + 1) - np.sum(nt ** 3 - nt) / (N * (N - 1))))
        corr = (0.5 * np.sign(z)) if correct else 0.0
        z = (z - corr) / sigma
        p = 2 * min(stats.norm.cdf(z), stats.norm.sf(z))
        method = "Wilcoxon rank sum test" + (" with continuity correction" if correct else "")
    return dict(W=W, p=p, method=method, warn=warn, R1=float(r[:nx].sum()), R2=float(r[nx:].sum()), nx=nx, ny=ny, exact=exact and not ties)


def _signrank_counts(n):
    tot = n * (n + 1) // 2
    c = [1] + [0] * tot
    for k in range(1, n + 1):
        for s in range(tot, k - 1, -1):
            c[s] += c[s - k]
    return c


def psignrank(q, n, upper=False):
    c = _signrank_counts(n)
    q = int(math.floor(q + 1e-7))
    tot = 2 ** n
    if upper:
        return sum(c[q + 1:]) / tot
    return sum(c[:q + 1]) / tot


def signed_rank(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[d != 0]
    n = len(d)
    r = stats.rankdata(np.abs(d))
    V = float(r[d > 0].sum())
    ties = len(np.unique(np.abs(d))) < n
    if n < 50 and not ties:
        if V > n * (n + 1) / 4:
            p = psignrank(V - 1, n, upper=True)
        else:
            p = psignrank(V, n)
        p = min(2 * p, 1.0)
        method = "Wilcoxon signed rank exact test"
    else:
        _, nt = np.unique(np.abs(d), return_counts=True)
        z = V - n * (n + 1) / 4
        sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - np.sum(nt ** 3 - nt) / 48)
        z = (z - 0.5 * np.sign(z)) / sigma
        p = 2 * min(stats.norm.cdf(z), stats.norm.sf(z))
        method = "Wilcoxon signed rank test with continuity correction"
    return dict(V=V, p=p, method=method, n=n, d=d, ranks=r)


def binom_test(x, n, p0=0.5, conf=0.95):
    d = stats.binom.pmf(x, n, p0)
    m = n * p0
    if x == m:
        pv = 1.0
    elif x < m:
        ks = np.arange(math.ceil(m), n + 1)
        yv = int(np.sum(stats.binom.pmf(ks, n, p0) <= d * RELERR))
        pv = stats.binom.cdf(x, n, p0) + stats.binom.sf(n - yv, n, p0)
    else:
        ks = np.arange(0, math.floor(m) + 1)
        yv = int(np.sum(stats.binom.pmf(ks, n, p0) <= d * RELERR))
        pv = stats.binom.cdf(yv - 1, n, p0) + stats.binom.sf(x - 1, n, p0)
    pv = min(1.0, float(pv))
    a = (1 - conf) / 2
    lo = 0.0 if x == 0 else stats.beta.ppf(a, x, n - x + 1)
    hi = 1.0 if x == n else stats.beta.ppf(1 - a, x + 1, n - x)
    return dict(p=pv, ci=(lo, hi), est=x / n)


# ---------------------------------------------------------------------------
# categorical
def chisq(tab, correct=True):
    O = np.asarray(tab, float)
    n = O.sum()
    E = np.outer(O.sum(1), O.sum(0)) / n
    if correct and O.shape == (2, 2):
        Y = np.minimum(0.5, np.abs(O - E))
        stat = float(np.sum((np.abs(O - E) - Y) ** 2 / E))
        method = "Pearson's Chi-squared test with Yates' continuity correction"
    else:
        stat = float(np.sum((O - E) ** 2 / E))
        method = "Pearson's Chi-squared test"
    df = (O.shape[0] - 1) * (O.shape[1] - 1)
    p = float(stats.chi2.sf(stat, df))
    return dict(X2=stat, df=df, p=p, E=E, res2=(O - E) ** 2 / E, method=method, small=bool((E < 5).any()))


def prop_test(tab, correct=True, conf=0.95):
    O = np.asarray(tab, float)
    x, n = O[:, 0], O.sum(1)
    est = x / n
    pc = x.sum() / n.sum()
    E = np.c_[n * pc, n * (1 - pc)]
    if correct:
        YATES = min(0.5, float(np.min(np.abs(O - E))))
    else:
        YATES = 0.0
    stat = float(np.sum((np.abs(O - E) - YATES) ** 2 / E))
    p = float(stats.chi2.sf(stat, 1))
    delta = est[0] - est[1]
    yc = min(YATES, abs(delta) / np.sum(1 / n)) if correct else 0.0
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    width = z * math.sqrt(np.sum(est * (1 - est) / n)) + yc * np.sum(1 / n)
    ci = (max(delta - width, -1), min(delta + width, 1))
    method = "2-sample test for equality of proportions " + ("with" if correct else "without") + " continuity correction"
    return dict(X2=stat, p=p, ci=ci, est=est, method=method, n=n, x=x)


def _brent(f, ax, bx, tol=np.finfo(float).eps ** 0.25, maxit=1000):
    """R's uniroot (R_zeroin2, Brent), so roots match R to the printed digits."""
    EPS = np.finfo(float).eps
    a, b = ax, bx
    fa, fb = f(a), f(b)
    c, fc = a, fa
    if fa == 0:
        return a
    if fb == 0:
        return b
    for _ in range(maxit + 1):
        prev_step = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol_act = 2 * EPS * abs(b) + tol / 2
        new_step = (c - b) / 2
        if abs(new_step) <= tol_act or fb == 0:
            return b
        if abs(prev_step) >= tol_act and abs(fa) > abs(fb):
            cb = c - b
            if a == c:
                t1 = fb / fa
                p = cb * t1
                q = 1.0 - t1
            else:
                q = fa / fc
                t1 = fb / fc
                t2 = fb / fa
                p = t2 * (cb * q * (q - t1) - (b - a) * (t1 - 1.0))
                q = (q - 1.0) * (t1 - 1.0) * (t2 - 1.0)
            if p > 0:
                q = -q
            else:
                p = -p
            if p < (0.75 * cb * q - abs(tol_act * q) / 2) and p < abs(prev_step * q / 2):
                new_step = p / q
        if abs(new_step) < tol_act:
            new_step = tol_act if new_step > 0 else -tol_act
        a, fa = b, fb
        b += new_step
        fb = f(b)
        if (fb > 0 and fc > 0) or (fb < 0 and fc < 0):
            c, fc = a, fa
    return b


def fisher(tab, conf=0.95):
    O = np.asarray(tab, int)
    x = O[0, 0]
    m = O[:, 0].sum()
    n = O[:, 1].sum()
    k = O[0, :].sum()
    lo, hi = max(0, k - n), min(k, m)
    sup = np.arange(lo, hi + 1)
    logdc = stats.hypergeom.logpmf(sup, m + n, m, k)

    def dnhyper(ncp):
        d = logdc + math.log(ncp) * sup
        d = np.exp(d - d.max())
        return d / d.sum()

    def mnhyper(ncp):
        if ncp == 0:
            return lo
        if ncp == math.inf:
            return hi
        return float(np.sum(sup * dnhyper(ncp)))

    def pnhyper(q, ncp, upper=False):
        if ncp == 1:
            if upper:
                return float(stats.hypergeom.sf(q - 1, m + n, m, k))
            return float(stats.hypergeom.cdf(q, m + n, m, k))
        if ncp == 0:
            return float(q <= lo) if upper else float(q >= lo)
        if ncp == math.inf:
            return float(q <= hi) if upper else float(q >= hi)
        d = dnhyper(ncp)
        return float(d[sup >= q].sum()) if upper else float(d[sup <= q].sum())

    d1 = dnhyper(1.0)
    pv = float(d1[d1 <= d1[x - lo] * RELERR].sum())
    eps = np.finfo(float).eps

    def mle(xx):
        if xx == lo:
            return 0.0
        if xx == hi:
            return math.inf
        mu = mnhyper(1.0)
        if mu > xx:
            return _brent(lambda t: mnhyper(t) - xx, 0, 1)
        if mu < xx:
            return 1 / _brent(lambda t: mnhyper(1 / t) - xx, eps, 1)
        return 1.0

    def ncp_u(xx, alpha):
        if xx == hi:
            return math.inf
        p = pnhyper(xx, 1)
        if p < alpha:
            return _brent(lambda t: pnhyper(xx, t) - alpha, 0, 1)
        if p > alpha:
            return 1 / _brent(lambda t: pnhyper(xx, 1 / t) - alpha, eps, 1)
        return 1.0

    def ncp_l(xx, alpha):
        if xx == lo:
            return 0.0
        p = pnhyper(xx, 1, upper=True)
        if p > alpha:
            return _brent(lambda t: pnhyper(xx, t, upper=True) - alpha, 0, 1)
        if p < alpha:
            return 1 / _brent(lambda t: pnhyper(xx, 1 / t, upper=True) - alpha, eps, 1)
        return 1.0

    a = (1 - conf) / 2
    return dict(p=min(1.0, pv), est=mle(x), ci=(ncp_l(x, a), ncp_u(x, a)))


def mcnemar(tab, correct=True):
    O = np.asarray(tab, float)
    b, c = O[0, 1], O[1, 0]
    y = 1.0 if correct else 0.0
    stat = (abs(b - c) - y) ** 2 / (b + c)
    p = float(stats.chi2.sf(stat, 1))
    method = "McNemar's Chi-squared test" + (" with continuity correction" if correct else "")
    return dict(X2=stat, p=p, method=method, b=b, c=c)
