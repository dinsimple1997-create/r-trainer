"""Format statsmodels results the way R prints them (close enough for exam reading)."""
import math
import numpy as np
from scipy import stats

SIGCODES = "Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"


def stars(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    if p < 0.1: return '.'
    return ' '


def fpval(p, digits=3):
    if p < 2.2e-16:
        return '<2e-16'
    if p < 1e-4:
        s = f'{p:.{digits-1}e}'
        m, e = s.split('e')
        e = int(e)
        m = m.rstrip('0').rstrip('.') if '.' in m else m
        return f'{m}e-{abs(e):02d}'
    return f'{p:.{digits}g}' if p >= 0.001 else f'{p:.{digits}g}'


def sig(x, d=4):
    """format with d significant digits, R style (no sci for usual ranges)."""
    if x == 0:
        return '0'
    mag = math.floor(math.log10(abs(x)))
    dec = max(0, d - 1 - mag)
    return f'{x:.{dec}f}'


def col_fmt(vals, d=4):
    """Format a column so the smallest |value| has d significant digits; same decimals for all."""
    nz = [abs(v) for v in vals if v != 0]
    mn = min(nz) if nz else 1
    mag = math.floor(math.log10(mn))
    dec = max(0, min(7, d - 1 - mag))
    return [f'{v:.{dec}f}' for v in vals]


def coef_table(names, est, se, stat, p, statname='t'):
    est_s = col_fmt(list(est) + list(se))
    e_s, s_s = est_s[:len(est)], est_s[len(est):]
    t_s = [f'{v:.3f}' for v in stat]
    p_s = [fpval(v) for v in p]
    st = [stars(v) for v in p]
    w0 = max(len(n) for n in names)
    hdr_p = 'Pr(>|t|)' if statname == 't' else 'Pr(>|z|)'
    hdr_t = 't value' if statname == 't' else 'z value'
    w1 = max(len('Estimate'), max(len(x) for x in e_s))
    w2 = max(len('Std. Error'), max(len(x) for x in s_s))
    w3 = max(len(hdr_t), max(len(x) for x in t_s))
    w4 = max(len(hdr_p), max(len(x) for x in p_s))
    lines = [' ' * w0 + ' ' + 'Estimate'.rjust(w1) + ' ' + 'Std. Error'.rjust(w2) + ' ' + hdr_t.rjust(w3) + ' ' + hdr_p.rjust(w4) + '    ']
    for i, n in enumerate(names):
        lines.append(n.ljust(w0) + ' ' + e_s[i].rjust(w1) + ' ' + s_s[i].rjust(w2) + ' ' + t_s[i].rjust(w3) + ' ' + p_s[i].rjust(w4) + ' ' + st[i])
    return lines


def resid_block(res):
    q = np.quantile(res, [0, .25, .5, .75, 1])
    vals = [sig(v, 4) if abs(v) >= 1e-4 else f'{v:.2e}' for v in q]
    # R uses common formatting; approximate with column formatter
    vals = col_fmt(list(q))
    labs = ['Min', '1Q', 'Median', '3Q', 'Max']
    w = max(max(len(v) for v in vals), 3)
    return [' '.join(l.rjust(w) for l in labs) + ' ', ' '.join(v.rjust(w) for v in vals) + ' ']


def _order(m, names):
    idx = list(m.params.index)
    if names and isinstance(names[0], tuple):
        keys = [a for a, b in names]; labs = [b for a, b in names]
    else:
        keys = idx; labs = names or idx
    labs = ['(Intercept)' if n == 'Intercept' else n for n in labs]
    return keys, labs


def lm_summary(m, call, names=None):
    keys, names = _order(m, names)
    out = ['Call:', call, '', 'Residuals:']
    out += resid_block(m.resid.values)
    out += ['', 'Coefficients:']
    out += coef_table(names, m.params[keys].values, m.bse[keys].values, m.tvalues[keys].values, m.pvalues[keys].values, 't')
    out += ['---', SIGCODES, '']
    s = math.sqrt(m.scale)
    out.append(f'Residual standard error: {sig(s,4)} on {int(m.df_resid)} degrees of freedom')
    out.append(f'Multiple R-squared:  {sig(m.rsquared,4)},\tAdjusted R-squared:  {sig(m.rsquared_adj,4)} ')
    out.append(f'F-statistic: {sig(m.fvalue,4)} on {int(m.df_model)} and {int(m.df_resid)} DF,  p-value: {fpval(m.f_pvalue,4)}')
    return '\n'.join(out)


def glm_summary(m, call, family='binomial', names=None, dispersion=None):
    keys, names = _order(m, names)
    statname = 't' if family.startswith('quasi') else 'z'
    out = ['Call:', call, '', 'Coefficients:']
    tv = m.tvalues[keys].values
    pv = m.pvalues[keys].values if statname == 'z' else 2 * stats.t.sf(np.abs(tv), m.df_resid)
    out += coef_table(names, m.params[keys].values, m.bse[keys].values, tv, pv, statname)
    out += ['---', SIGCODES, '']
    if dispersion is None:
        out.append(f'(Dispersion parameter for {family} family taken to be 1)')
    else:
        out.append(f'(Dispersion parameter for {family} family taken to be {sig(dispersion,4)})')
    out.append('')
    nd, rd = m.null_deviance, m.deviance
    ndf = int(m.df_model + m.df_resid)
    out.append(f'    Null deviance: {nd:.2f}  on {ndf}  degrees of freedom')
    out.append(f'Residual deviance: {rd:.2f}  on {int(m.df_resid)}  degrees of freedom')
    if dispersion is None:
        out.append(f'AIC: {m.aic:.2f}')
    else:
        out.append('AIC: NA')
    out.append('')
    out.append('Number of Fisher Scoring iterations: 4')
    return '\n'.join(out)


def anova_compare(m1, m2, f1, f2):
    rdf1, rdf2 = int(m1.df_resid), int(m2.df_resid)
    rss1, rss2 = m1.ssr, m2.ssr
    df = rdf1 - rdf2
    ss = rss1 - rss2
    F = (ss / df) / (rss2 / rdf2)
    p = stats.f.sf(F, df, rdf2)
    out = ['Analysis of Variance Table', '', f'Model 1: {f1}', f'Model 2: {f2}',
           '  Res.Df    RSS Df Sum of Sq      F    Pr(>F)    ',
           f'1 {rdf1:6d} {sig(rss1,5):>6}                                  ',
           f'2 {rdf2:6d} {sig(rss2,5):>6} {df:2d} {sig(ss,5):>9} {sig(F,4):>6} {fpval(p,4):>9} {stars(p)}',
           '---', SIGCODES]
    return '\n'.join(out), dict(df=df, ss=ss, F=F, p=p, rdf2=rdf2, rss1=rss1, rss2=rss2, rdf1=rdf1)


def anova_glm_compare(m1, m2, f1, f2):
    rdf1, rdf2 = int(m1.df_resid), int(m2.df_resid)
    d1, d2 = m1.deviance, m2.deviance
    df = rdf1 - rdf2
    dd = d1 - d2
    p = stats.chi2.sf(dd, df)
    out = ['Analysis of Deviance Table', '', f'Model 1: {f1}', f'Model 2: {f2}',
           '  Resid. Df Resid. Dev Df Deviance  Pr(>Chi)    ',
           f'1 {rdf1:9d} {d1:10.3f}                             ',
           f'2 {rdf2:9d} {d2:10.3f} {df:2d} {dd:8.3f} {fpval(p,4):>9} {stars(p)}',
           '---', SIGCODES]
    return '\n'.join(out), dict(df=df, dd=dd, p=p)


def anova_table(rows, resid, response):
    """rows: list of (name, df, ss); resid: (df, ss). Sequential (type I) table as anova(m)."""
    rdf, rss = resid
    mse = rss / rdf
    names = [r[0] for r in rows] + ['Residuals']
    w0 = max(len(n) for n in names)
    out = ['Analysis of Variance Table', '', f'Response: {response}',
           ' ' * w0 + '   Df  Sum Sq Mean Sq F value    Pr(>F)    ']
    for n, df, ss in rows:
        F = (ss / df) / mse
        p = stats.f.sf(F, df, rdf)
        out.append(f'{n.ljust(w0)} {df:4d} {sig(ss,5):>7} {sig(ss/df,5):>7} {sig(F,4):>7} {fpval(p,4):>9} {stars(p)}')
    out.append(f'{"Residuals".ljust(w0)} {rdf:4d} {sig(rss,5):>7} {sig(mse,5):>7}                      ')
    out += ['---', SIGCODES]
    return '\n'.join(out)


def anova_type3(rows, resid, response):
    """car::Anova(m, type=3) style. rows: (name, ss, df)."""
    rdf, rss = resid
    mse = rss / rdf
    names = [r[0] for r in rows] + ['Residuals']
    w0 = max(len(n) for n in names)
    out = ['Anova Table (Type III tests)', '', f'Response: {response}',
           ' ' * w0 + '  Sum Sq Df  F value    Pr(>F)    ']
    for n, ss, df in rows:
        F = (ss / df) / mse
        p = stats.f.sf(F, df, rdf)
        out.append(f'{n.ljust(w0)} {sig(ss,5):>7} {df:2d} {sig(F,4):>8} {fpval(p,4):>9} {stars(p)}')
    out.append(f'{"Residuals".ljust(w0)} {sig(rss,5):>7} {rdf:2d}                      ')
    out += ['---', SIGCODES]
    return '\n'.join(out)


def predict_block(rows, cols=('fit', 'lwr', 'upr')):
    out = ['       ' + '      '.join(c.rjust(4) for c in cols)]
    for i, r in enumerate(rows, 1):
        out.append(f'{i} ' + ' '.join(f'{v:10.4f}' for v in r))
    return '\n'.join(out)


def confint_block(names, lo, hi, level=95):
    a = (100 - level) / 2
    l1, l2 = f'{a:g} %', f'{100-a:g} %'
    lo_s = col_fmt(list(lo) + list(hi))
    w0 = max(len(n) for n in names)
    out = [' ' * w0 + ' ' + l1.rjust(11) + ' ' + l2.rjust(11)]
    for i, n in enumerate(names):
        out.append(n.ljust(w0) + ' ' + lo_s[i].rjust(11) + ' ' + lo_s[len(names) + i].rjust(11))
    return '\n'.join(out)
