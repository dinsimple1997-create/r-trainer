import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
import json, math, sys
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
sys.path.insert(0, HERE)
import rfmt as R

OUT = {}
V = {}


def put(key, text, **vals):
    OUT[key] = text
    V[key] = vals


# 1. SLR: weaning weight -> weight at 70 d (one piglet per litter, observational)
rng = np.random.default_rng(11)
n = 30
ww = np.round(rng.uniform(4.6, 8.4, n), 1)
w70 = np.round(8 + 3.0 * ww + rng.normal(0, 2.4, n), 1)
d1 = pd.DataFrame(dict(wean=ww, w70=w70))
m1 = smf.ols('w70 ~ wean', d1).fit()
put('slr', R.lm_summary(m1, 'lm(formula = w70 ~ wean, data = piglets)'),
    b0=m1.params.iloc[0], b1=m1.params.iloc[1], se0=m1.bse.iloc[0], se1=m1.bse.iloc[1], t1=m1.tvalues.iloc[1], p1=m1.pvalues.iloc[1],
    s=math.sqrt(m1.scale), df=m1.df_resid, r2=m1.rsquared, F=m1.fvalue, xbar=ww.mean(), xmin=ww.min(), xmax=ww.max(), sx=ww.std(ddof=1), sse=m1.ssr, ybar=w70.mean())
an = R.anova_table([('wean', 1, m1.ess)], (int(m1.df_resid), m1.ssr), 'w70')
put('slr_anova', an, ssr=m1.ess, sse=m1.ssr)
ci = m1.conf_int()
put('slr_confint', R.confint_block(['(Intercept)', 'wean'], ci[0].values, ci[1].values), lo=list(ci[0]), hi=list(ci[1]))
new = pd.DataFrame(dict(wean=[6.0, 8.0, 10.0]))
pc = m1.get_prediction(new).summary_frame(alpha=0.05)
put('slr_pred_conf', R.predict_block(pc[['mean', 'mean_ci_lower', 'mean_ci_upper']].values),
    fit=list(pc['mean']), se=list(pc['mean_se']), lo=list(pc['mean_ci_lower']), hi=list(pc['mean_ci_upper']))
put('slr_pred_pred', R.predict_block(pc[['mean', 'obs_ci_lower', 'obs_ci_upper']].values),
    lo=list(pc['obs_ci_lower']), hi=list(pc['obs_ci_upper']))
V['slr']['tcrit'] = stats.t.ppf(0.975, m1.df_resid)
d1.to_csv(HERE + '/piglets.csv', index=False)

# 2. log Y: antibody titre after vaccination
rng = np.random.default_rng(21)
wk = np.repeat(np.arange(1, 13), 2)
lt = 7.6 - 0.16 * wk + rng.normal(0, 0.25, len(wk))
titre = np.round(np.exp(lt))
d2 = pd.DataFrame(dict(week=wk, titre=titre))
m2 = smf.ols('np.log(titre) ~ week', d2).fit()
put('logy', R.lm_summary(m2, 'lm(formula = log(titre) ~ week, data = vacc)', names=['(Intercept)', 'week']),
    b0=m2.params.iloc[0], b1=m2.params.iloc[1], se1=m2.bse.iloc[1], df=m2.df_resid, s=math.sqrt(m2.scale), r2=m2.rsquared,
    lo=m2.conf_int().iloc[1, 0], hi=m2.conf_int().iloc[1, 1], tcrit=stats.t.ppf(0.975, m2.df_resid))

# 3. log-log: feed intake vs body weight
rng = np.random.default_rng(31)
bw = np.round(np.exp(rng.uniform(np.log(10), np.log(120), 40)), 1)
fi = np.round(0.115 * bw ** 0.75 * np.exp(rng.normal(0, 0.09, 40)), 2)
d3 = pd.DataFrame(dict(bw=bw, feed=fi))
m3 = smf.ols('np.log(feed) ~ np.log(bw)', d3).fit()
put('loglog', R.lm_summary(m3, 'lm(formula = log(feed) ~ log(bw), data = pigs)', names=['(Intercept)', 'log(bw)']),
    b0=m3.params.iloc[0], b1=m3.params.iloc[1], se1=m3.bse.iloc[1], df=m3.df_resid, lo=m3.conf_int().iloc[1, 0], hi=m3.conf_int().iloc[1, 1])

# 3b. log X: meat pH (course data)
meat = pd.read_csv(HERE + '/meat.txt', sep=r'\s+')
m3b = smf.ols('ph ~ np.log(time)', meat).fit()
put('logx', R.lm_summary(m3b, 'lm(formula = ph ~ log(time), data = meat)', names=['(Intercept)', 'log(time)']),
    b0=m3b.params.iloc[0], b1=m3b.params.iloc[1], se1=m3b.bse.iloc[1], df=m3b.df_resid, s=math.sqrt(m3b.scale))
mf = smf.ols('ph ~ C(time)', meat).fit()
txt, info = R.anova_compare(m3b, mf, 'ph ~ log(time)', 'ph ~ factor(time)')
put('lof', txt, **info)
mlin = smf.ols('ph ~ time', meat).fit()
txt2, info2 = R.anova_compare(mlin, mf, 'ph ~ time', 'ph ~ factor(time)')
put('lof_bad', txt2, **info2)

# 4. logistic: post-weaning death vs weaning weight
rng = np.random.default_rng(41)
n4 = 400
wt = np.round(rng.uniform(3.5, 9.0, n4), 1)
p = 1 / (1 + np.exp(-(1.6 - 0.62 * wt)))
died = rng.binomial(1, p)
d4 = pd.DataFrame(dict(wean=wt, died=died))
m4 = smf.glm('died ~ wean', d4, family=sm.families.Binomial()).fit()
m40 = smf.glm('died ~ 1', d4, family=sm.families.Binomial()).fit()
put('logit', R.glm_summary(m4, 'glm(formula = died ~ wean, family = binomial, data = weaners)'),
    b0=m4.params.iloc[0], b1=m4.params.iloc[1], se1=m4.bse.iloc[1], z1=m4.tvalues.iloc[1], p1=m4.pvalues.iloc[1], nd=m4.null_deviance, rd=m4.deviance, n=n4, deaths=int(died.sum()))
txt, info = R.anova_glm_compare(m40, m4, 'died ~ 1', 'died ~ wean')
put('logit_lrt', txt, **info)
newd = pd.DataFrame(dict(wean=[4.0, 7.0]))
Xn = np.column_stack([np.ones(2), newd.wean.values])
lm_ = Xn @ m4.params.values
se_ = np.sqrt(np.einsum('ij,jk,ik->i', Xn, m4.cov_params().values, Xn))
link = pd.DataFrame(dict(mean=lm_, mean_se=se_))
V['logit']['pred_link'] = list(link['mean']); V['logit']['pred_link_se'] = list(link['mean_se'])
V['logit']['pred_p'] = list(1 / (1 + np.exp(-link['mean'])))
# profile-like CI (use statsmodels Wald) for slope
V['logit']['wald'] = [m4.params.iloc[1] - 1.96 * m4.bse.iloc[1], m4.params.iloc[1] + 1.96 * m4.bse.iloc[1]]
OUT['logit_predict'] = ('> predict(fit, newdata = data.frame(wean = c(4, 7)), se.fit = TRUE)\n$fit\n' +
                        '         1          2 \n' + f'{link["mean"].iloc[0]:10.6f} {link["mean"].iloc[1]:10.6f} \n\n$se.fit\n' +
                        '        1         2 \n' + f'{link["mean_se"].iloc[0]:9.6f} {link["mean_se"].iloc[1]:9.6f} ')
pp = 1 / (1 + np.exp(-link['mean']))
OUT['logit_predict_resp'] = ('> predict(fit, newdata = data.frame(wean = c(4, 7)), type = "response")\n' +
                             '         1          2 \n' + f'{pp.iloc[0]:10.7f} {pp.iloc[1]:10.7f} ')

# 5. binomial counts: deaths per pen vs hours of transport
rng = np.random.default_rng(51)
k = 24
hrs = np.round(rng.uniform(2, 14, k), 1)
N = rng.integers(180, 260, k)
pp5 = 1 / (1 + np.exp(-(-4.4 + 0.11 * hrs)))
dead = rng.binomial(N, pp5)
d5 = pd.DataFrame(dict(hours=hrs, placed=N, dead=dead))
m5 = smf.glm('dead + I(placed-dead) ~ hours', d5, family=sm.families.Binomial()).fit()
put('binom', R.glm_summary(m5, 'glm(formula = cbind(dead, placed - dead) ~ hours, family = binomial, data = trucks)', names=['(Intercept)', 'hours']),
    b0=m5.params.iloc[0], b1=m5.params.iloc[1], se1=m5.bse.iloc[1], rd=m5.deviance, rdf=m5.df_resid)

# 6. Poisson: stillborn piglets vs parity
rng = np.random.default_rng(61)
k = 80
par = rng.integers(1, 9, k)
mu = np.exp(-0.35 + 0.13 * par)
sb = rng.poisson(mu)
d6 = pd.DataFrame(dict(parity=par, stillborn=sb))
m6 = smf.glm('stillborn ~ parity', d6, family=sm.families.Poisson()).fit()
put('pois', R.glm_summary(m6, 'glm(formula = stillborn ~ parity, family = poisson, data = sows)', family='poisson'),
    b0=m6.params.iloc[0], b1=m6.params.iloc[1], se1=m6.bse.iloc[1], rd=m6.deviance, rdf=m6.df_resid, nd=m6.null_deviance)
# overdispersed version: tail biting incidents vs stocking density (negative binomial generated)
rng = np.random.default_rng(62)
k = 50
dens = np.round(rng.uniform(0.6, 1.2, k), 2)  # m2 per pig
mu = np.exp(3.2 - 2.0 * dens)
r_ = 2.0
lam = rng.gamma(r_, mu / r_)
tb = rng.poisson(lam)
d6b = pd.DataFrame(dict(space=dens, bites=tb))
m6b = smf.glm('bites ~ space', d6b, family=sm.families.Poisson()).fit()
phi = (m6b.resid_pearson ** 2).sum() / m6b.df_resid
m6q = smf.glm('bites ~ space', d6b, family=sm.families.Poisson()).fit(scale='X2')
put('pois_od', R.glm_summary(m6b, 'glm(formula = bites ~ space, family = poisson, data = pens)', family='poisson'),
    b1=m6b.params.iloc[1], se1=m6b.bse.iloc[1], rd=m6b.deviance, rdf=m6b.df_resid, phi=phi)
put('pois_q', R.glm_summary(m6q, 'glm(formula = bites ~ space, family = quasipoisson, data = pens)', family='quasipoisson', dispersion=phi),
    b1=m6q.params.iloc[1], se1=m6q.bse.iloc[1], t1=m6q.tvalues.iloc[1], p1=m6q.pvalues.iloc[1])

# 7. MLR with confounding: weaning weight ~ age + litter
rng = np.random.default_rng(71)
k = 60
age = rng.uniform(18, 28, k)
litter = np.clip(np.round(6 + 0.25 * age + rng.normal(0, 1.1, k)), 7, 15)
ww7 = 1.0 + 0.26 * age - 0.17 * litter + rng.normal(0, 0.35, k)
d7 = pd.DataFrame(dict(age=np.round(age), litter=litter, wean=np.round(ww7, 2)))
m7s = smf.ols('wean ~ litter', d7).fit()
m7 = smf.ols('wean ~ age + litter', d7).fit()
m7a = smf.ols('wean ~ age', d7).fit()
put('mlr_slr', R.lm_summary(m7s, 'lm(formula = wean ~ litter, data = litters)'), b1=m7s.params.iloc[1], se1=m7s.bse.iloc[1], p1=m7s.pvalues.iloc[1], r2=m7s.rsquared)
put('mlr', R.lm_summary(m7, 'lm(formula = wean ~ age + litter, data = litters)'),
    b0=m7.params.iloc[0], b_age=m7.params.iloc[1], b_lit=m7.params.iloc[2], se_lit=m7.bse.iloc[2], se_age=m7.bse.iloc[1], s=math.sqrt(m7.scale), df=m7.df_resid, r2=m7.rsquared, r2a=m7.rsquared_adj, F=m7.fvalue,
    t_lit=m7.tvalues.iloc[2], p_lit=m7.pvalues.iloc[2], corr=np.corrcoef(d7.age, d7.litter)[0, 1], sse=m7.ssr, n=k)
txt, info = R.anova_compare(m7a, m7, 'wean ~ age', 'wean ~ age + litter')
put('mlr_cmp', txt, **info)
# 7b: three predictors + sequential vs type III
rng2 = np.random.default_rng(72)
parity = np.clip(np.round(1 + 0.15 * (litter - 7) + rng2.uniform(0, 5, k)), 1, 7)
ww7b = ww7 + 0.06 * parity + rng2.normal(0, 0.05, k)
d7['parity'] = parity
d7['wean2'] = np.round(ww7b, 2)
m7b = smf.ols('wean2 ~ litter + age + parity', d7).fit()
from statsmodels.stats.anova import anova_lm
a1 = anova_lm(m7b, typ=1)
a3 = anova_lm(m7b, typ=3)
put('mlr3', R.lm_summary(m7b, 'lm(formula = wean ~ litter + age + parity, data = litters)'), sse=m7b.ssr, df=m7b.df_resid)
put('typ1', R.anova_table([(nme, int(a1.loc[nme, 'df']), a1.loc[nme, 'sum_sq']) for nme in ['litter', 'age', 'parity']], (int(m7b.df_resid), m7b.ssr), 'wean'),
    **{f'ss_{k_}': a1.loc[k_, 'sum_sq'] for k_ in ['litter', 'age', 'parity']}, **{f'F_{k_}': a1.loc[k_, 'F'] for k_ in ['litter', 'age', 'parity']}, **{f'p_{k_}': a1.loc[k_, 'PR(>F)'] for k_ in ['litter', 'age', 'parity']})
put('typ3', R.anova_type3([(nme, a3.loc[nme, 'sum_sq'], 1) for nme in ['(Intercept)', 'litter', 'age', 'parity'] if (nme in a3.index or nme == '(Intercept)')] if '(Intercept)' in a3.index else [('(Intercept)', a3.loc['Intercept', 'sum_sq'], 1)] + [(nme, a3.loc[nme, 'sum_sq'], 1) for nme in ['litter', 'age', 'parity']], (int(m7b.df_resid), m7b.ssr), 'wean'),
    **{f'ss_{k_}': a3.loc[k_, 'sum_sq'] for k_ in ['litter', 'age', 'parity']}, **{f'F_{k_}': a3.loc[k_, 'F'] for k_ in ['litter', 'age', 'parity']}, **{f't_{k_}': m7b.tvalues[k_] for k_ in ['litter', 'age', 'parity']})
d7.to_csv(HERE + '/litters.csv', index=False)

# 8. ANCOVA: 3 diets + start weight, ADG
rng = np.random.default_rng(89)
k = 45
diet = np.repeat(['CON', 'PRO', 'ACID'], 15)
start = np.round(rng.uniform(6, 10, k), 1)
eff = {'CON': 0, 'PRO': 32, 'ACID': 14}
adg = 250 + 24 * start + np.array([eff[d] for d in diet]) + rng.normal(0, 18, k)
d8 = pd.DataFrame(dict(diet=pd.Categorical(diet, categories=['CON', 'ACID', 'PRO']), start=start, adg=np.round(adg)))
m8 = smf.ols('adg ~ start + diet', d8).fit()
m8i = smf.ols('adg ~ start * diet', d8).fit()
m8d = smf.ols('adg ~ diet', d8).fit()
nm8 = [('Intercept','(Intercept)'), ('start','start'), ('diet[T.ACID]','dietACID'), ('diet[T.PRO]','dietPRO')]
put('ancova', R.lm_summary(m8, 'lm(formula = adg ~ start + diet, data = pens)', names=nm8),
    b0=m8.params['Intercept'], b_start=m8.params['start'], b_acid=m8.params['diet[T.ACID]'], b_pro=m8.params['diet[T.PRO]'], se_pro=m8.bse['diet[T.PRO]'], se_acid=m8.bse['diet[T.ACID]'], se_start=m8.bse['start'], s=math.sqrt(m8.scale), df=m8.df_resid, mean_start=start.mean())
nm8i = [('Intercept','(Intercept)'), ('start','start'), ('diet[T.ACID]','dietACID'), ('diet[T.PRO]','dietPRO'), ('start:diet[T.ACID]','start:dietACID'), ('start:diet[T.PRO]','start:dietPRO')]
put('ancova_int', R.lm_summary(m8i, 'lm(formula = adg ~ start * diet, data = pens)', names=nm8i),
    b0=m8i.params['Intercept'], b_start=m8i.params['start'], b_acid=m8i.params['diet[T.ACID]'], b_pro=m8i.params['diet[T.PRO]'], b_sa=m8i.params['start:diet[T.ACID]'], b_sp=m8i.params['start:diet[T.PRO]'], p_sa=m8i.pvalues['start:diet[T.ACID]'], p_sp=m8i.pvalues['start:diet[T.PRO]'])
txt, info = R.anova_compare(m8, m8i, 'adg ~ start + diet', 'adg ~ start * diet')
put('ancova_cmp', txt, **info)
put('diet_only', R.lm_summary(m8d, 'lm(formula = adg ~ diet, data = pens)', names=['(Intercept)', 'dietACID', 'dietPRO']),
    b0=m8d.params.iloc[0], b_acid=m8d.params.iloc[1], b_pro=m8d.params.iloc[2], means={g: d8[d8.diet == g].adg.mean() for g in ['CON', 'ACID', 'PRO']}, s=math.sqrt(m8d.scale))
V['ancova']['group_start'] = {g: d8[d8.diet == g].start.mean() for g in ['CON', 'ACID', 'PRO']}
V['ancova']['group_adg'] = {g: d8[d8.diet == g].adg.mean() for g in ['CON', 'ACID', 'PRO']}

# 9. Quadratic: ADG vs room temperature
rng = np.random.default_rng(91)
k = 32
T = np.round(rng.uniform(15, 29, k), 1)
adg9 = -400 + 110 * T - 2.5 * T ** 2 + rng.normal(0, 28, k)
d9 = pd.DataFrame(dict(temp=T, adg=np.round(adg9)))
m9 = smf.ols('adg ~ temp + I(temp**2)', d9).fit()
put('quad', R.lm_summary(m9, 'lm(formula = adg ~ temp + I(temp^2), data = rooms)', names=['(Intercept)', 'temp', 'I(temp^2)']),
    b0=m9.params.iloc[0], b1=m9.params.iloc[1], b2=m9.params.iloc[2], se2=m9.bse.iloc[2], p2=m9.pvalues.iloc[2], xmax=-m9.params.iloc[1] / (2 * m9.params.iloc[2]))
X = sm.add_constant(pd.DataFrame(dict(t=T, t2=T ** 2)))
V['quad']['vif'] = [vif(X.values, 1), vif(X.values, 2)]
Tc = T - 22
d9['tc'] = Tc
m9c = smf.ols('adg ~ tc + I(tc**2)', d9).fit()
Xc = sm.add_constant(pd.DataFrame(dict(t=Tc, t2=Tc ** 2)))
V['quad']['vif_c'] = [vif(Xc.values, 1), vif(Xc.values, 2)]
put('quad_c', R.lm_summary(m9c, 'lm(formula = adg ~ I(temp - 22) + I((temp - 22)^2), data = rooms)', names=['(Intercept)', 'I(temp - 22)', 'I((temp - 22)^2)']),
    b0=m9c.params.iloc[0], b1=m9c.params.iloc[1], b2=m9c.params.iloc[2])
m9l = smf.ols('adg ~ temp', d9).fit()
put('quad_lin', R.lm_summary(m9l, 'lm(formula = adg ~ temp, data = rooms)'), b1=m9l.params.iloc[1], p1=m9l.pvalues.iloc[1], r2=m9l.rsquared)

# 10. Multicollinearity: body weight from chest girth and body length
rng = np.random.default_rng(101)
k = 40
girth = rng.uniform(80, 120, k)
length = 0.95 * girth + rng.normal(0, 2.2, k)
bw10 = -95 + 1.05 * girth + 0.85 * length + rng.normal(0, 4.5, k)
d10 = pd.DataFrame(dict(girth=np.round(girth, 1), length=np.round(length, 1), bw=np.round(bw10, 1)))
m10 = smf.ols('bw ~ girth + length', d10).fit()
m10g = smf.ols('bw ~ girth', d10).fit()
X10 = sm.add_constant(d10[['girth', 'length']])
v10 = [vif(X10.values, 1), vif(X10.values, 2)]
put('vif_lm', R.lm_summary(m10, 'lm(formula = bw ~ girth + length, data = pigs)'),
    b_g=m10.params.iloc[1], b_l=m10.params.iloc[2], p_g=m10.pvalues.iloc[1], p_l=m10.pvalues.iloc[2], se_g=m10.bse.iloc[1], F=m10.fvalue, r2=m10.rsquared, corr=np.corrcoef(d10.girth, d10.length)[0, 1], vif=v10)
put('vif_out', '> library(car)\n> vif(fit)\n   girth   length \n' + f'{v10[0]:8.4f} {v10[1]:8.4f} ')
put('vif_g', R.lm_summary(m10g, 'lm(formula = bw ~ girth, data = pigs)'), b_g=m10g.params.iloc[1], se_g=m10g.bse.iloc[1])

# 11. Influence: ADG vs feed intake with one bad pen
rng = np.random.default_rng(111)
k = 25
fi11 = np.round(rng.uniform(1.6, 2.4, k), 2)
adg11 = np.round(80 + 340 * fi11 + rng.normal(0, 25, k))
fi11[-1] = 3.4  # data-entry error pen: high intake, low adg
adg11[-1] = 820
d11 = pd.DataFrame(dict(intake=fi11, adg=adg11))
m11 = smf.ols('adg ~ intake', d11).fit()
infl = m11.get_influence()
cd = infl.cooks_distance[0]
rs = infl.resid_studentized_internal
h = infl.hat_matrix_diag
m11b = smf.ols('adg ~ intake', d11.iloc[:-1]).fit()
V['infl'] = dict(cook=float(cd[-1]), rstd=float(rs[-1]), hat=float(h[-1]), b1=m11.params.iloc[1], b1_wo=m11b.params.iloc[1], maxcook_others=float(np.max(cd[:-1])), p=2, n=k)
top = np.argsort(-cd)[:4]
lines = ['> data.frame(intake, adg, rstandard = round(rstandard(fit), 2),', '+            hat = round(hatvalues(fit), 3), cooks = round(cooks.distance(fit), 3))[c(' + ', '.join(str(i + 1) for i in sorted(top)) + '), ]',
         '   intake  adg rstandard   hat cooks']
for i in sorted(top):
    lines.append(f'{i+1:2d}   {fi11[i]:.2f} {int(adg11[i]):4d} {rs[i]:9.2f} {h[i]:.3f} {cd[i]:.3f}')
OUT['infl'] = '\n'.join(lines)
OUT['infl_lm'] = R.lm_summary(m11, 'lm(formula = adg ~ intake, data = pens)')
OUT['infl_lm_wo'] = R.lm_summary(m11b, 'lm(formula = adg ~ intake, data = pens[-25, ])')

# 12. Model selection table (all subsets, AICc) for finishing pens ADG
rng = np.random.default_rng(121)
k = 60
d12 = pd.DataFrame(dict(start=rng.uniform(25, 40, k), space=rng.uniform(0.65, 1.0, k), temp=rng.uniform(16, 26, k),
                        water=rng.uniform(0.6, 1.6, k), males=rng.uniform(0.3, 0.7, k)))
d12['adg'] = 500 + 6 * d12.start + 280 * d12.space - 6 * (d12.temp - 20) + 20 * d12.water + rng.normal(0, 38, k)
import itertools
cands = ['start', 'space', 'temp', 'water', 'males']
rows = []
for r_ in range(0, 6):
    for comb in itertools.combinations(cands, r_):
        f = 'adg ~ ' + (' + '.join(comb) if comb else '1')
        mm = smf.ols(f, d12).fit()
        kk = len(comb) + 2
        aicc = -2 * mm.llf + 2 * kk + 2 * kk * (kk + 1) / (k - kk - 1)
        rows.append(dict(model=comb, k=kk, llf=mm.llf, aicc=aicc, bic=-2 * mm.llf + kk * math.log(k), coefs=mm.params))
rows.sort(key=lambda r: r['aicc'])
best = rows[0]['aicc']
ws = np.array([math.exp(-0.5 * (r['aicc'] - best)) for r in rows])
ws = ws / ws.sum()
hdr = 'Global model call: lm(formula = adg ~ start + space + temp + water + males, data = fin)\n---\nModel selection table \n'
cols = ['(Int)', 'males', 'space', 'start', 'temp', 'water']
lines = ['   ' + '  '.join(c.rjust(6) for c in cols) + ' df   logLik  AICc delta weight']
for i, r in enumerate(rows[:6]):
    cells = []
    for c in cols:
        key = 'Intercept' if c == '(Int)' else c
        if c == '(Int)' or c in r['model']:
            cells.append(f"{r['coefs'][key]:6.1f}" if abs(r['coefs'][key]) >= 1 else f"{r['coefs'][key]:6.3f}")
        else:
            cells.append(' ' * 6)
    lines.append(f'{i+1:2d} ' + '  '.join(cells) + f" {r['k']:2d} {r['llf']:8.2f} {r['aicc']:5.1f} {r['aicc']-best:5.2f} {ws[i]:6.3f}")
OUT['dredge'] = hdr + '\n'.join(lines) + '\nModels ranked by AICc(x) '
V['dredge'] = dict(rows=[dict(model=list(r['model']), k=r['k'], aicc=r['aicc'], bic=r['bic'], delta=r['aicc'] - best) for r in rows[:8]], n=k)

json.dump(V, open(HERE + '/values.json', 'w'), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else (o.tolist() if isinstance(o, np.ndarray) else str(o)))
json.dump(OUT, open(HERE + '/outputs.json', 'w'), indent=1)
for kk, t in OUT.items():
    print('#' * 10, kk)
    print(t)

# profile CI for weaners logistic
from scipy.optimize import brentq, minimize_scalar
y_, x_ = d4.died.values, d4.wean.values
def ll(b0, b1):
    eta = b0 + b1 * x_
    return np.sum(y_ * eta - np.log1p(np.exp(eta)))
LLmax = m4.llf
B0, B1 = m4.params.iloc[0], m4.params.iloc[1]
def prof_b1(b1):
    return -minimize_scalar(lambda b0: -ll(b0, b1), bracket=(B0 - 3, B0 + 3)).fun
def prof_b0(b0):
    return -minimize_scalar(lambda b1: -ll(b0, b1), bracket=(B1 - 0.5, B1 + 0.5)).fun
crit = stats.chi2.ppf(0.95, 1) / 2
lo1 = brentq(lambda b: LLmax - prof_b1(b) - crit, B1 - 6 * m4.bse.iloc[1], B1)
hi1 = brentq(lambda b: LLmax - prof_b1(b) - crit, B1, B1 + 6 * m4.bse.iloc[1])
lo0 = brentq(lambda b: LLmax - prof_b0(b) - crit, B0 - 6 * m4.bse.iloc[0], B0)
hi0 = brentq(lambda b: LLmax - prof_b0(b) - crit, B0, B0 + 6 * m4.bse.iloc[0])
OUT['logit_confint'] = '> confint(fit)\nWaiting for profiling to be done...\n' + R.confint_block(['(Intercept)', 'wean'], [lo0, lo1], [hi0, hi1])
V['logit']['profile'] = [lo1, hi1]
json.dump(V, open(HERE + '/values.json', 'w'), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else (o.tolist() if isinstance(o, np.ndarray) else str(o)))
json.dump(OUT, open(HERE + '/outputs.json', 'w'), indent=1)
print(OUT['logit_confint']); print(V['logit'])
