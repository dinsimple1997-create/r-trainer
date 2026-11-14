"""Formatting helpers that imitate R's console printing."""
import math

SIGNIF = "Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1"
EPS = 2.220446e-16


def _rgt(v, digits):
    """Decimals R needs to show v with `digits` significant digits (trailing zeros dropped)."""
    if v == 0 or not math.isfinite(v):
        return 0
    s = f"{abs(v):.{digits - 1}e}"
    mant, exp = s.split("e")
    exp = int(exp)
    mant = mant.replace(".", "").rstrip("0")
    nsig = max(1, len(mant))
    return max(0, nsig - 1 - exp)


def fmt_joint(vals, digits):
    rgt = min(12, max(_rgt(v, digits) for v in vals))
    out = []
    for v in vals:
        s = f"{v:.{rgt}f}"
        if float(s) == 0:
            s = s.lstrip("-")
        out.append(s)
    return out


def fmt1(v, digits):
    return fmt_joint([v], digits)[0]


def round_dec(v, d):
    return round(v + 0.0, d)


def coef_cols(est, se, digits=4):
    allv = [abs(x) for x in list(est) + list(se) if x != 0]
    dec = max(1, digits - (1 + math.floor(math.log10(min(allv)))))
    vals = [round(x, dec) for x in list(est) + list(se)]
    f = fmt_joint(vals, digits)
    n = len(est)
    return f[:n], f[n:]


def tval_col(tv, digits=4):
    return fmt_joint([round(x, 3) for x in tv], digits)


def _sci_joint(ps, digits):
    need = 1
    for p in ps:
        mant = f"{p:.{digits - 1}e}".split("e")[0].replace(".", "").rstrip("0")
        need = max(need, len(mant))
    return [f"{p:.{need - 1}e}" for p in ps]


def pval_col(ps, digits=3, eps_text="< 2e-16"):
    out = [None] * len(ps)
    fix = [i for i, p in enumerate(ps) if p >= EPS and math.floor(math.log10(p)) >= -3]
    sci = [i for i, p in enumerate(ps) if p >= EPS and math.floor(math.log10(p)) < -3]
    for i, p in enumerate(ps):
        if p < EPS:
            out[i] = eps_text
    if fix:
        f = fmt_joint([ps[i] for i in fix], digits)
        for i, s in zip(fix, f):
            out[i] = s
    if sci:
        f = _sci_joint([ps[i] for i in sci], digits)
        for i, s in zip(sci, f):
            out[i] = s
    return out


def pval1(p, digits=4):
    if p < EPS:
        return "< 2.2e-16"
    return pval_col([p], digits)[0]


def stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return ""


def table(rownames, cols, star_list=None, rn_pad=0, rn_width=None):
    """cols = [(header, [strings])]. Right-aligned columns, left-aligned row names."""
    w_rn = rn_width if rn_width is not None else max(len(r) for r in rownames) + rn_pad
    widths = [max(len(h), max(len(v) for v in vals)) for h, vals in cols]
    lines = []
    head = " " * w_rn + "".join(" " + h.rjust(w) for (h, _), w in zip(cols, widths))
    lines.append(head.rstrip())
    for i, rn in enumerate(rownames):
        line = rn.ljust(w_rn) + "".join(" " + vals[i].rjust(w) for (_, vals), w in zip(cols, widths))
        if star_list is not None and star_list[i]:
            line += " " + star_list[i]
        lines.append(line.rstrip())
    return lines


def named_vector(names, vals_str):
    w = max(max(len(n) for n in names), max(len(v) for v in vals_str))
    widths = [w] * len(names)
    l1 = " ".join(n.rjust(w) for n, w in zip(names, widths))
    l2 = " ".join(v.rjust(w) for v, w in zip(vals_str, widths))
    return [l1, l2]


def matrix(rownames, colnames, cols_vals, digits=7, fixed_dec=None):
    """Print a numeric matrix like R (each column formatted jointly)."""
    cols = []
    for j, (cn, vals) in enumerate(zip(colnames, cols_vals)):
        if fixed_dec is not None and fixed_dec.get(j) is not None:
            s = [f"{v:.{fixed_dec[j]}f}" for v in vals]
        else:
            s = fmt_joint(vals, digits)
        cols.append((cn, s))
    return table(rownames, cols)


def r7(v):
    """One number printed by R with 7 significant digits."""
    return fmt1(v, 7)
