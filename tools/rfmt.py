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
        f = [x.strip() for x in format_real([ps[i] for i in sci], digits)]
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
            s = [x.strip() for x in format_real(list(vals), digits)]
        cols.append((cn, s))
    return table(rownames, cols)


def r7(v):
    """One number printed by R with 7 significant digits."""
    return fmt1(v, 7)


# ---------------------------------------------------------------------------
# R's formatReal: joint fixed / scientific choice (used by format(), print())
def _sig_info(x, digits):
    """(kpower, nsig, neg) for x shown with `digits` significant digits."""
    if x == 0 or not math.isfinite(x):
        return 0, 1, x < 0
    s = f"{abs(x):.{digits - 1}e}"
    mant, ex = s.split("e")
    kp = int(ex)
    m = mant.replace(".", "").rstrip("0")
    return kp, max(1, len(m)), x < 0


def format_real(vals, digits=7):
    """Format a numeric vector the way R's format(x, digits) does (common width, right-justified)."""
    info = [_sig_info(v, digits) for v in vals]
    neg = 1 if any(v < 0 for v in vals) else 0
    mxsl, rgt, mxns, mxe, mne = 1, 0, 1, -999, 999
    for v, (kp, ns, _) in zip(vals, info):
        if v == 0:
            continue
        left = kp + 1 if kp >= 0 else 1
        mxsl = max(mxsl, left)
        rgt = max(rgt, ns - kp - 1)
        mxns = max(mxns, ns)
        mxe, mne = max(mxe, kp), min(mne, kp)
    rgt = max(0, min(rgt, 15))
    wF = neg + mxsl + (rgt + 1 if rgt else 0)
    e = 2 if (mxe >= 100 or mne <= -100) else 1
    d = mxns - 1
    wE = neg + (1 if d > 0 else 0) + d + 4 + e
    if all(v == 0 for v in vals) or wF <= wE:
        out = [f"{v:.{rgt}f}" for v in vals]
        out = [s.lstrip("-") if float(s) == 0 else s for s in out]
    else:
        out = []
        for v in vals:
            s = f"{v:.{d}e}"
            mant, ex = s.split("e")
            sign, digs = ex[0], ex[1:].lstrip("0") or "0"
            out.append(f"{mant}e{sign}{digs.zfill(2 if e == 1 else 3)}")
    w = max(len(s) for s in out)
    return [s.rjust(w) for s in out]


def abbreviate(s, minlength):
    """R abbreviate(): drop lower-case vowels from the right, then lower-case letters, then truncate."""
    if len(s) <= minlength:
        return s
    ch = list(s)
    for pool in ("aeiou", "abcdefghijklmnopqrstuvwxyz"):
        i = len(ch) - 1
        while len(ch) > minlength and i > 0:
            if ch[i] in pool:
                del ch[i]
            i -= 1
    return "".join(ch[:max(minlength, 0)]) if len(ch) > minlength else "".join(ch)


def coefmat(names, cols, star_p=None, digits=4):
    """printCoefmat: cols = [(header, values, kind)], kind in cs (joint coef/se/df), tst, p."""
    cs = [(h, v) for h, v, k in cols if k == "cs"]
    allv = [abs(x) for _, v in cs for x in v if x != 0 and math.isfinite(x)]
    dec = max(1, digits - (1 + math.floor(math.log10(min(allv))))) if allv else 1
    joint = format_real([round(x, dec) for _, v in cs for x in v], digits)
    out_cols, pos = [], 0
    n = len(names)
    for h, v, k in cols:
        if k == "cs":
            out_cols.append((h, joint[pos:pos + n]))
            pos += n
        elif k == "tst":
            out_cols.append((h, format_real([round(x, 3) for x in v], digits)))
        elif k == "p":
            out_cols.append((h, pval_col(list(v))))
    stars_l = [stars(p) for p in star_p] if star_p is not None else None
    return table(names, out_cols, stars_l)
