import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
import json, math, sys, itertools
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from statsmodels.stats.anova import anova_lm
sys.path.insert(0, HERE)
import rfmt as R

OUT, V = {}, {}

# A1: iron dose (mg, day 3) -> haemoglobin at weaning (g/dL); randomized, 4 doses x 8 piglets
rng = np.random.default_rng(201)
dose = np.repeat([50, 100, 200, 400], 8)
hb = 6.4 + 0.95 * np.log(dose) + rng.normal(0, 0.55, len(dose))
a1 = pd.DataFrame(dict(dose=dose, hb=np.round(hb, 1)))
mA1 = smf.ols('hb ~ np.log(dose)', a1).fit()
OUT['A1'] = R.lm_summary(mA1, 'lm(formula = hb ~ log(dose), data = iron)', names=['(Intercept)', 'log(dose)'])
pr = mA1.get_prediction(pd.DataFrame(dict(dose=[200]))).summary_frame(alpha=0.05)
OUT['A1_pred'] = ('> predict(fit, newdata = data.frame(dose = 200), interval = "confidence")\n' + R.predict_block(pr[['mean', 'mean_ci_lower', 'mean_ci_upper']].values) +
                  '\n> predict(fit, newdata = data.frame(dose = 200), interval = "prediction")\n' + R.predict_block(pr[['mean', 'obs_ci_lower', 'obs_ci_upper']].values))
mA1f = smf.ols('hb ~ C(dose)', a1).fit()
txt, info = R.anova_compare(mA1, mA1f, 'hb ~ log(dose)', 'hb ~ factor(dose)')
OUT['A1_lof'] = txt
V['A1'] = dict(b0=mA1.params.iloc[0], b1=mA1.params.iloc[1], se1=mA1.bse.iloc[1], df=mA1.df_resid, s=math.sqrt(mA1.scale), r2=mA1.rsquared,
               fit=pr['mean'].iloc[0], clo=pr['mean_ci_lower'].iloc[0], chi=pr['mean_ci_upper'].iloc[0], plo=pr['obs_ci_lower'].iloc[0], phi=pr['obs_ci_upper'].iloc[0],
               se_fit=pr['mean_se'].iloc[0], tcrit=stats.t.ppf(0.975, mA1.df_resid), lof=info, meanlog=np.log(dose).mean())

# A2: M. hyo seropositivity vs age in weeks (cross-sectional sample, 1 pig per pen)
rng = np.random.default_rng(202)
n = 240
age = rng.integers(4, 25, n)
p = 1 / (1 + np.exp(-(-4.1 + 0.27 * age)))
pos = rng.binomial(1, p)
a2 = pd.DataFrame(dict(age=age, pos=pos))
mA2 = smf.glm('pos ~ age', a2, family=sm.families.Binomial()).fit()
mA20 = smf.glm('pos ~ 1', a2, family=sm.families.Binomial()).fit()
OUT['A2'] = R.glm_summary(mA2, 'glm(formula = pos ~ age, family = binomial, data = sero)')
txt, info = R.anova_glm_compare(mA20, mA2, 'pos ~ 1', 'pos ~ age')
OUT['A2_lrt'] = txt
from scipy.optimize import brentq, minimize_scalar
y_, x_ = a2.pos.values, a2.age.values
def ll(b0, b1):
    eta = b0 + b1 * x_
    return np.sum(y_ * eta - np.log1p(np.exp(eta)))
LLmax = mA2.llf
def prof_b1(b1):
    r = minimize_scalar(lambda b0: -ll(b0, b1), bracket=(-8, -2)); return -r.fun
def prof_b0(b0):
    r = minimize_scalar(lambda b1: -ll(b0, b1), bracket=(0.05, 0.6)); return -r.fun
crit = stats.chi2.ppf(0.95, 1) / 2
B0, B1 = mA2.params.iloc[0], mA2.params.iloc[1]
lo1 = brentq(lambda b: LLmax - prof_b1(b) - crit, B1 - 6 * mA2.bse.iloc[1], B1)
hi1 = brentq(lambda b: LLmax - prof_b1(b) - crit, B1, B1 + 6 * mA2.bse.iloc[1])
lo0 = brentq(lambda b: LLmax - prof_b0(b) - crit, B0 - 6 * mA2.bse.iloc[0], B0)
hi0 = brentq(lambda b: LLmax - prof_b0(b) - crit, B0, B0 + 6 * mA2.bse.iloc[0])
OUT['A2_ci'] = ('> confint(fit)   # profile likelihood\nWaiting for profiling to be done...\n' + R.confint_block(['(Intercept)', 'age'], [lo0, lo1], [hi0, hi1]))
ci = pd.DataFrame([[lo0, hi0], [lo1, hi1]])
V['A2'] = dict(b0=mA2.params.iloc[0], b1=mA2.params.iloc[1], se1=mA2.bse.iloc[1], z=mA2.tvalues.iloc[1], p=mA2.pvalues.iloc[1], lrt=info, n=n, npos=int(pos.sum()),
               p16=1 / (1 + math.exp(-(mA2.params.iloc[0] + 16 * mA2.params.iloc[1]))), ci=[lo0, lo1, hi0, hi1], wald=[B1 - 1.96 * mA2.bse.iloc[1], B1 + 1.96 * mA2.bse.iloc[1]])

# A3: coughs per pen in 15 min vs ammonia (ppm), overdispersed
rng = np.random.default_rng(203)
k = 48
nh3 = np.round(rng.uniform(5, 30, k))
mu = np.exp(0.6 + 0.055 * nh3)
lam = rng.gamma(2.5, mu / 2.5)
cough = rng.poisson(lam)
a3 = pd.DataFrame(dict(nh3=nh3, cough=cough))
mA3 = smf.glm('cough ~ nh3', a3, family=sm.families.Poisson()).fit()
phi = (mA3.resid_pearson ** 2).sum() / mA3.df_resid
mA3q = smf.glm('cough ~ nh3', a3, family=sm.families.Poisson()).fit(scale='X2')
OUT['A3'] = R.glm_summary(mA3, 'glm(formula = cough ~ nh3, family = poisson, data = barns)', family='poisson')
OUT['A3_q'] = R.glm_summary(mA3q, 'glm(formula = cough ~ nh3, family = quasipoisson, data = barns)', family='quasipoisson', dispersion=phi)
V['A3'] = dict(b0=mA3.params.iloc[0], b1=mA3.params.iloc[1], se1=mA3.bse.iloc[1], rd=mA3.deviance, rdf=mA3.df_resid, phi=phi, se1q=mA3q.bse.iloc[1],
               mu10=math.exp(mA3.params.iloc[0] + 10 * mA3.params.iloc[1]), mu25=math.exp(mA3.params.iloc[0] + 25 * mA3.params.iloc[1]))

# B1: lean % ~ backfat + loin + sex (gilt indicator); observational, n = 64
rng = np.random.default_rng(204)
k = 64
gilt = rng.integers(0, 2, k)
bf = rng.normal(16, 3, k) - 1.8 * gilt
loin = 58 + 0.35 * (bf - 16) + 2.5 * gilt + rng.normal(0, 4, k)
lean = 62 - 0.55 * bf + 0.12 * loin + 0.8 * gilt + rng.normal(0, 1.1, k)
b1d = pd.DataFrame(dict(backfat=np.round(bf, 1), loin=np.round(loin, 1), gilt=gilt, lean=np.round(lean, 1)))
mB1 = smf.ols('lean ~ backfat + loin + gilt', b1d).fit()
mB1r = smf.ols('lean ~ backfat', b1d).fit()
mB1s = smf.ols('lean ~ loin', b1d).fit()
OUT['B1'] = R.lm_summary(mB1, 'lm(formula = lean ~ backfat + loin + gilt, data = carcass)')
OUT['B1_slr'] = R.lm_summary(mB1s, 'lm(formula = lean ~ loin, data = carcass)')
txt, info = R.anova_compare(mB1r, mB1, 'lean ~ backfat', 'lean ~ backfat + loin + gilt')
OUT['B1_cmp'] = txt
X = sm.add_constant(b1d[['backfat', 'loin', 'gilt']])
V['B1'] = dict(**{f'b_{c}': mB1.params[c] for c in ['Intercept', 'backfat', 'loin', 'gilt']}, **{f'se_{c}': mB1.bse[c] for c in ['backfat', 'loin', 'gilt']},
               **{f'p_{c}': mB1.pvalues[c] for c in ['backfat', 'loin', 'gilt']}, s=math.sqrt(mB1.scale), df=mB1.df_resid, r2=mB1.rsquared, F=mB1.fvalue,
               slr_loin=mB1s.params['loin'], slr_p=mB1s.pvalues['loin'], cmp=info, sse_full=mB1.ssr, sse_red=mB1r.ssr, n=k,
               vif=[vif(X.values, i) for i in range(1, 4)], tcrit=stats.t.ppf(0.975, mB1.df_resid))

# B2: feeder type (dry vs wetdry) x body weight -> feed intake; interaction present
rng = np.random.default_rng(205)
k = 40
feeder = np.repeat(['dry', 'wetdry'], 20)
bw = np.round(rng.uniform(30, 100, k))
fi = np.where(feeder == 'dry', 0.55 + 0.024 * bw, 0.40 + 0.031 * bw) + rng.normal(0, 0.12, k)
b2d = pd.DataFrame(dict(feeder=pd.Categorical(feeder, ['dry', 'wetdry']), bw=bw, intake=np.round(fi, 2)))
mB2 = smf.ols('intake ~ bw * feeder', b2d).fit()
mB2a = smf.ols('intake ~ bw + feeder', b2d).fit()
names = [('Intercept', '(Intercept)'), ('bw', 'bw'), ('feeder[T.wetdry]', 'feederwetdry'), ('bw:feeder[T.wetdry]', 'bw:feederwetdry')]
OUT['B2'] = R.lm_summary(mB2, 'lm(formula = intake ~ bw * feeder, data = feeders)', names=names)
txt, info = R.anova_compare(mB2a, mB2, 'intake ~ bw + feeder', 'intake ~ bw * feeder')
OUT['B2_cmp'] = txt
V['B2'] = dict(b0=mB2.params['Intercept'], b_bw=mB2.params['bw'], b_w=mB2.params['feeder[T.wetdry]'], b_int=mB2.params['bw:feeder[T.wetdry]'], p_int=mB2.pvalues['bw:feeder[T.wetdry]'], cmp=info)

# B3: quadratic: ADG vs dietary net energy (Mcal/kg)
rng = np.random.default_rng(206)
k = 36
ne = np.repeat([2.20, 2.30, 2.40, 2.50, 2.60, 2.70], 6)
adg = -2183.6 + 2081 * ne - 408 * ne ** 2 + rng.normal(0, 15, k)
b3d = pd.DataFrame(dict(ne=ne, adg=np.round(adg)))
mB3 = smf.ols('adg ~ ne + I(ne**2)', b3d).fit()
OUT['B3'] = R.lm_summary(mB3, 'lm(formula = adg ~ ne + I(ne^2), data = energy)', names=['(Intercept)', 'ne', 'I(ne^2)'])
Xq = sm.add_constant(pd.DataFrame(dict(a=ne, b=ne ** 2)))
V['B3'] = dict(b0=mB3.params.iloc[0], b1=mB3.params.iloc[1], b2=mB3.params.iloc[2], p2=mB3.pvalues.iloc[2], xopt=-mB3.params.iloc[1] / (2 * mB3.params.iloc[2]), vif=vif(Xq.values, 1))
V['B3']['yopt'] = mB3.params.iloc[0] + mB3.params.iloc[1] * V['B3']['xopt'] + mB3.params.iloc[2] * V['B3']['xopt'] ** 2
OUT['B3_vif'] = '> vif(fit)\n       ne   I(ne^2) \n' + f'{vif(Xq.values,1):9.1f} {vif(Xq.values,2):9.1f} '

# B4: diagnostics + model selection: mortality % per batch? use ADG in nursery with 4 candidates
rng = np.random.default_rng(207)
k = 50
d = pd.DataFrame(dict(weanwt=rng.uniform(5, 8, k), creep=rng.integers(0, 2, k), temp=rng.uniform(24, 30, k), stocking=rng.uniform(0.25, 0.4, k)))
d['weanage'] = 4 * d.weanwt + rng.normal(0, 1.0, k)  # strongly collinear with weanwt
d['adg'] = 120 + 32 * d.weanwt + 25 * d.creep - 4 * (d.temp - 27) + 300 * d.stocking + rng.normal(0, 22, k)
cands = ['weanwt', 'weanage', 'creep', 'temp', 'stocking']
rows = []
for r_ in range(0, 6):
    for comb in itertools.combinations(cands, r_):
        f = 'adg ~ ' + (' + '.join(comb) if comb else '1')
        mm = smf.ols(f, d).fit(); kk = len(comb) + 2
        rows.append(dict(model=comb, k=kk, llf=mm.llf, aicc=-2 * mm.llf + 2 * kk + 2 * kk * (kk + 1) / (k - kk - 1), bic=-2 * mm.llf + kk * math.log(k), coefs=mm.params))
rows.sort(key=lambda r: r['aicc'])
best = rows[0]['aicc']
ws = np.array([math.exp(-0.5 * (r['aicc'] - best)) for r in rows]); ws /= ws.sum()
cols = ['(Int)', 'creep', 'stckn', 'temp', 'weanag', 'weanwt']
keymap = {'(Int)': 'Intercept', 'creep': 'creep', 'stckn': 'stocking', 'temp': 'temp', 'weanag': 'weanage', 'weanwt': 'weanwt'}
lines = ['   ' + '  '.join(c.rjust(6) for c in cols) + ' df   logLik  AICc delta weight']
for i, r in enumerate(rows[:6]):
    cells = []
    for c in cols:
        key = keymap[c]
        if c == '(Int)' or key in r['model']:
            v = r['coefs'][key]; cells.append(f'{v:6.1f}')
        else:
            cells.append(' ' * 6)
    lines.append(f'{i+1:2d} ' + '  '.join(cells) + f" {r['k']:2d} {r['llf']:8.2f} {r['aicc']:5.1f} {r['aicc']-best:5.2f} {ws[i]:6.3f}")
OUT['B4_dredge'] = 'Model selection table (all subsets, ranked by AICc)\n' + '\n'.join(lines)
full = smf.ols('adg ~ weanwt + weanage + creep + temp + stocking', d).fit()
Xf = sm.add_constant(d[cands])
vv = [vif(Xf.values, i) for i in range(1, 6)]
OUT['B4_vif'] = '> vif(full)\n  weanwt  weanage    creep     temp stocking \n' + ' '.join(f'{x:8.2f}' for x in vv)
OUT['B4_full'] = R.lm_summary(full, 'lm(formula = adg ~ weanwt + weanage + creep + temp + stocking, data = nursery)')
infl = full.get_influence(); cd = infl.cooks_distance[0]; rs = infl.resid_studentized_internal
top = np.argsort(-cd)[:3]
OUT['B4_cook'] = '> head(sort(cooks.distance(full), decreasing = TRUE), 3)\n' + '    '.join(f'{i+1:>5}' for i in top) + '\n' + '  '.join(f'{cd[i]:7.4f}' for i in top) + \
                 '\n> range(rstandard(full))\n' + f'[1] {rs.min():.3f}  {rs.max():.3f}'
V['B4'] = dict(rows=[dict(model=list(r['model']), k=r['k'], aicc=r['aicc'], bic=r['bic'], d=r['aicc'] - best) for r in rows[:6]], vif=vv, cook_max=float(cd.max()),
               rs=[float(rs.min()), float(rs.max())], F=full.fvalue, pvals={c: full.pvalues[c] for c in cands}, n=k)

json.dump(V, open(HERE + '/exam_values.json', 'w'), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else (o.tolist() if isinstance(o, np.ndarray) else str(o)))
json.dump(OUT, open(HERE + '/exam_outputs.json', 'w'), indent=1)
for kk, t in OUT.items():
    print('#' * 10, kk); print(t)
print(json.dumps(V, indent=1, default=str)[:6000])
