"""Helpers for the end-to-end analysis workflows (wf_bank.py): R-like printing of data frames,
question builders, plot payloads."""
import math
import numpy as np
from scipy import stats
import rfmt as R
from gen import ru, rua, mc, num
import diag_bank as DB


# ---------------------------------------------------------------------------
# numbers
def r3(vals):
    """str()-style: format jointly with 3 significant digits, then drop trailing zeros."""
    out = []
    for s in R.fmt_joint(list(vals), 3):
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        out.append(s)
    return out


def rnum(v, d):
    return ru(v, d)


def pr(p):
    """p-value in Russian text."""
    if p < 0.0001:
        return "&lt; 0,0001"
    if p < 0.001:
        return ru(p, 4)
    return ru(p, 3)


# ---------------------------------------------------------------------------
# data frame printing
def str_df(n, cols, cls="'data.frame'"):
    """cols: [(name, kind, values)] kind in num, int, chr, factor."""
    w = max(len(c[0]) for c in cols)
    out = [f"{cls}:\t{n} obs. of  {len(cols)} variable{'s' if len(cols) > 1 else ''}:"]
    for name, kind, vals in cols:
        vals = list(vals)
        if kind == "int":
            shown = [str(int(v)) for v in vals[:10]]
            line = "int  " + " ".join(shown)
            more = len(vals) > 10
        elif kind == "num":
            shown = r3(vals[:5])
            line = "num  " + " ".join(shown)
            more = len(vals) > 5
        elif kind == "chr":
            shown = [f'"{v}"' for v in vals[:4]]
            line = "chr  " + " ".join(shown)
            more = len(vals) > 4
        else:  # factor
            lev = sorted(set(vals))
            codes = [str(lev.index(v) + 1) for v in vals[:10]]
            ls = ",".join(f'"{l}"' for l in lev)
            if len(ls) > 30:
                ls = ",".join(f'"{l}"' for l in lev[:2]) + ",.."
            line = f"Factor w/ {len(lev)} level{'s' if len(lev) > 1 else ''} {ls}: " + " ".join(codes)
            more = len(vals) > 10
        out.append(f" $ {name.ljust(w)}: {line}{' ...' if more else ''}")
    return out


def df_print(cols, rows=None, rownames=None):
    """print(data.frame): cols = [(name, kind, values)], numeric formatted jointly with 7 digits."""
    n = len(cols[0][2])
    idx = list(range(n)) if rows is None else rows
    rn = rownames or [str(i + 1) for i in idx]
    fmt_cols = []
    for name, kind, vals in cols:
        v = [vals[i] for i in idx]
        if kind in ("num", "int"):
            s = [x.strip() for x in R.format_real([float(x) for x in v], 7)]
            if kind == "int":
                s = [str(int(x)) for x in v]
        else:
            s = [str(x) for x in v]
        fmt_cols.append((name, s))
    wr = max(len(r) for r in rn)
    widths = [max(len(h), max(len(x) for x in s)) for h, s in fmt_cols]
    out = [" " * wr + "".join(" " + h.rjust(w) for (h, _), w in zip(fmt_cols, widths))]
    for i, r in enumerate(rn):
        out.append(r.ljust(wr) + "".join(" " + s[i].rjust(w) for (_, s), w in zip(fmt_cols, widths)))
    return out


def agg_print(gname, levels, yname, stats_cols, dec=1):
    """aggregate(y ~ g, data, FUN = function(v) round(c(n=..., mean=..., sd=...), dec))"""
    names = [f"{yname}.{k}" for k, _ in stats_cols]
    allv = [round(float(v), dec) for _, vals in stats_cols for v in vals]
    fm = [x.strip() for x in R.format_real(allv, 7)]
    # matrix column: common decimals
    decs = max((len(x.split(".")[1]) if "." in x else 0) for x in fm)
    fm = [f"{v:.{decs}f}" for v in allv]
    k = len(levels)
    cols = [(gname, [str(l) for l in levels])]
    for j, nm in enumerate(names):
        cols.append((nm, fm[j * k:(j + 1) * k]))
    widths = [max(len(h), max(len(x) for x in s)) for h, s in cols]
    out = [" " + "".join(" " + h.rjust(w) for (h, _), w in zip(cols, widths))]
    for i in range(k):
        out.append(str(i + 1) + "".join(" " + s[i].rjust(w) for (_, s), w in zip(cols, widths)))
    return out


def table2(rn_title, cn_title, rows, cols, M, cells=None):
    """print(table(a, b)) with named dimnames; cells: preformatted strings (else integers)."""
    if cells is None:
        cells = [[str(int(v)) for v in r] for r in np.asarray(M)]
    labs = ["  " + r for r in rows]
    wr = max(len(rn_title), max(len(r) for r in labs))
    cw = [max(len(c), max(len(cells[i][j]) for i in range(len(rows)))) for j, c in enumerate(cols)]
    out = [" " * wr + " " + cn_title, rn_title.ljust(wr) + "".join(" " + c.rjust(w) for c, w in zip(cols, cw))]
    for i, r in enumerate(labs):
        out.append(r.ljust(wr) + "".join(" " + cells[i][j].rjust(cw[j]) for j in range(len(cols))))
    return out


def joint_cells(M, digits=7):
    flat = [float(v) for r in M for v in r]
    f = [x.strip() for x in R.format_real(flat, digits)]
    k = len(M[0])
    return [f[i * k:(i + 1) * k] for i in range(len(M))]


def aov_summary(rows, resid, rn_width=None):
    """summary(aov): rows = [(name, df, ss)], resid = (df, ss)."""
    rdf, rss = resid
    mse = rss / rdf
    Fs = [(ss / df) / mse for _, df, ss in rows]
    ps = [float(stats.f.sf(F, df, rdf)) for F, (_, df, _) in zip(Fs, rows)]
    names = [r[0] for r in rows] + ["Residuals"]
    ss = R.fmt_joint([r[2] for r in rows] + [rss], 4)
    ms = R.fmt_joint([r[2] / r[1] for r in rows] + [mse], 4)
    Ff = R.fmt_joint(Fs, 4) + [""]
    pf = R.pval_col(ps) + [""]
    st = [R.stars(p) for p in ps] + [""]
    w = rn_width or max(11, max(len(n) for n in names) + 1)
    lines = R.table(names, [("Df", [str(r[1]) for r in rows] + [str(rdf)]), ("Sum Sq", ss), ("Mean Sq", ms), ("F value", Ff), ("Pr(>F)", pf)],
                    st, rn_width=w)
    if any(p < 0.1 for p in ps):
        lines += ["---", R.SIGNIF]
    return lines, dict(F=[float(x) for x in Ff[:-1]], p=ps, mse=float(ms[-1]), ms=[float(x) for x in ms])


def anova_lm(rows, resid, response):
    """anova(m) for lm: 'Analysis of Variance Table'."""
    lines, d = aov_summary(rows, resid, rn_width=max(len(r[0]) for r in rows + [("Residuals",)]))
    return ["Analysis of Variance Table", "", f"Response: {response}"] + lines, d


def anova2(f1, f2, rdf, rss, extra_names=None):
    """anova(m0, m1) for two nested lm fits: rdf, rss = [m0, m1]."""
    df = rdf[0] - rdf[1]
    ssd = rss[0] - rss[1]
    F = (ssd / df) / (rss[1] / rdf[1])
    p = float(stats.f.sf(F, df, rdf[1]))
    rssf = R.fmt_joint(rss, 5)
    cols = [("Res.Df", [str(rdf[0]), str(rdf[1])]), ("RSS", rssf), ("Df", ["", str(df)]), ("Sum of Sq", ["", R.fmt1(ssd, 5)]),
            ("F", ["", R.fmt1(F, 5)]), ("Pr(>F)", ["", R.pval_col([p])[0]])]
    lines = ["Analysis of Variance Table", "", f"Model 1: {f1}", f"Model 2: {f2}"] + R.table(["1", "2"], cols, ["", R.stars(p)])
    if p < 0.1:
        lines += ["---", R.SIGNIF]
    return lines, dict(F=F, p=p, df=df, ss=ssd)


def summary_vec(v, digits=4):
    q = np.quantile(v, [0, .25, .5, .75, 1], method="linear")
    vals = [q[0], q[1], q[2], float(np.mean(v)), q[3], q[4]]
    f = [x.strip() for x in R.format_real([float(f"{x:.{digits}g}") for x in vals], digits)]
    return R.named_vector(["Min.", "1st Qu.", "Median", "Mean", "3rd Qu.", "Max."], f)


def named(names, vals, digits=7):
    return R.named_vector(names, [s.strip() for s in R.format_real(list(vals), digits)])


def vec1(v):
    return "[1] " + R.r7(v)


# ---------------------------------------------------------------------------
# questions
def code_pick(rng, q, right, wrong, ex):
    w = lambda s: f"<code>{s}</code>"
    return mc(rng, q, w(right), [w(x) for x in wrong], ex)


def multi(rng, q, items, ex):
    """items: [(text, is_true)]"""
    order = list(rng.permutation(len(items)))
    opts = [items[i][0] for i in order]
    a = sorted(j for j, i in enumerate(order) if items[i][1])
    return dict(t="multi", q=q, o=opts, a=a, ex=ex)


def step(h, task=None, pick=None, code=None, out=None, plots=None, qs=None, note=None):
    s = dict(h=h)
    if task:
        s["task"] = task
    if pick:
        s["pick"] = pick
    if code:
        s["code"] = code if isinstance(code, str) else "\n".join(code)
    if out is not None:
        s["out"] = out if isinstance(out, str) else "\n".join(out)
    if plots:
        s["plots"] = plots
    s["qs"] = [q for q in (qs or []) if q]
    if note:
        s["note"] = note
    return s


# ---------------------------------------------------------------------------
# plots (drawn by the app)
def rd(a, d=4):
    return [float(f"{v:.{d}g}") for v in a]


def p_box(groups, title=None, yl=None):
    return dict(k="box", g=list(groups.keys()), v=[rd(np.asarray(v)) for v in groups.values()], t=title or "", yl=yl or "")


def p_xy(x, y, line=None, title="", xl="", yl="", lab=None):
    d = dict(k="xy", x=rd(x), y=rd(y), t=title, xl=xl, yl=yl)
    if line is not None:
        d["ab"] = [float(line[0]), float(line[1])]
    if lab:
        d["lab"] = lab
    return d


def p_int(xlev, series, title="", xl="", yl=""):
    """interaction plot: series = {name: [means per xlev]}"""
    return dict(k="int", xl=xl, yl=yl, t=title, xs=list(xlev), sr=[[k, rd(v)] for k, v in series.items()])


def p_lm(X, y, which=(1, 2)):
    F = DB.ols(np.asarray(X, float), np.asarray(y, float))
    P = DB.plots_lm(F, which)
    for p in P:
        if p["k"] == "qq":
            ids = p.pop("id")
            p["l"] = [[ids.index(i + 1), i + 1] for i in p["l"]]
    return P, F


def p_qq(vals, title="Normal Q-Q Plot", yl="Sample Quantiles", raw=True):
    v = np.sort(np.asarray(vals, float))
    return dict(k="qq", y=rd(v), t=title, yl=yl, l=[], raw=1 if raw else 0)


def p_hist(vals, title="", xl=""):
    return dict(k="hist", v=rd(vals), t=title, xl=xl)
