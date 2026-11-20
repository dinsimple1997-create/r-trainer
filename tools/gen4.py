"""v1.4: mixed models — lmer (lmerTest), glmer binomial / poisson (nAGQ = 10), emmeans + pairs.

Follows VDPAM 5290 weeks 7–12: hierarchical data, experimental studies with pen/block random,
repeated measures, GLMM logistic and Poisson with offset.
"""
import math
import numpy as np
from scipy import stats, special
import rfmt as R
from mixed import lmm, glmm
from gen import ru, rua, mc, num, pick, code
from exp_core import pl, nn

SIG = R.SIGNIF
LATENT = math.pi ** 2 / 3  # 3.29: residual variance on the latent logit scale


# ---------------------------------------------------------------------------
# printing
def quant7(x, p):
    x = np.sort(np.asarray(x, float))
    h = (len(x) - 1) * p
    lo = int(math.floor(h))
    hi = min(lo + 1, len(x) - 1)
    return x[lo] + (h - lo) * (x[hi] - x[lo])


def resid_lines(scaled):
    q = [quant7(scaled, p) for p in (0, .25, .5, .75, 1)]
    mx = max(abs(v) for v in q)
    q = [round(v, max(0, 5 - math.ceil(math.log10(mx)))) for v in q]
    return ["Scaled residuals: "] + R.named_vector(["Min", "1Q", "Median", "3Q", "Max"], [s.strip() for s in R.format_real(q, 4)])


def vc_lines(rows):
    """rows = [(Groups, Name, variance)] -> lme4 'Random effects:' block."""
    var = R.format_real([r[2] for r in rows], 4)
    sd = R.format_real([math.sqrt(r[2]) for r in rows], 4)
    cols = [("Groups", [r[0] for r in rows]), ("Name", [r[1] for r in rows]), ("Variance", var), ("Std.Dev.", sd)]
    w = [max(len(h), max(len(v) for v in vals)) for h, vals in cols]
    out = [" " + " ".join(h.ljust(wi) for (h, _), wi in zip(cols, w))]
    for i in range(len(rows)):
        out.append(" " + " ".join(vals[i].ljust(wi) for (_, vals), wi in zip(cols, w)))
    return [l.rstrip() for l in out]


def corr_lines(names, V):
    p = len(names)
    if p < 2:
        return []
    sd = np.sqrt(np.diag(V))
    C = V / np.outer(sd, sd)
    cells = R.format_real([round(float(C[i, j]), 3) for i in range(p) for j in range(p)], 7)
    # nsmall = 3
    cells = [c if "." in c else c + ".000" for c in cells]
    cells = [f"{float(c):.3f}" for c in cells]
    w = max(len(c) for c in cells)
    cells = [c.rjust(w) for c in cells]
    rn = [R.abbreviate(n, 11) for n in names]
    cn = [R.abbreviate(n, 6) for n in names]
    wr = max(len(r) for r in rn[1:])
    colw = [max(len(cn[j]), w) for j in range(p - 1)]
    out = ["Correlation of Fixed Effects:", " " * wr + " " + " ".join(cn[j].ljust(colw[j]) for j in range(p - 1))]
    for i in range(1, p):
        row = [cells[i * p + j] if j < i else "" for j in range(p - 1)]
        out.append(rn[i].ljust(wr) + " " + " ".join(row[j].ljust(colw[j]) for j in range(p - 1)))
    return [l.rstrip() for l in out]


def lmer_print(formula, data, F, names, group):
    lines = ["Linear mixed model fit by REML. t-tests use Satterthwaite's method [", "lmerModLmerTest]",
             f"Formula: {formula}", f"   Data: {data}", "", f"REML criterion at convergence: {round(F['crit'], 1):.1f}", ""]
    lines += resid_lines(F["scaled"]) + ["", "Random effects:"]
    lines += vc_lines([(group, "(Intercept)", F["tau"] ** 2), ("Residual", "", F["sigma"] ** 2)])
    lines += [f"Number of obs: {F['n']}, groups:  {group}, {F['ngroups']}", "", "Fixed effects:"]
    cols = [("Estimate", list(F["b"]), "cs"), ("Std. Error", list(F["se"]), "cs"), ("df", list(F["df"]), "cs"),
            ("t value", list(F["t"]), "tst"), ("Pr(>|t|)", list(F["p"]), "p")]
    sig = any(p < 0.1 for p in F["p"])
    lines += R.coefmat(names, cols, list(F["p"]) if sig else None)
    if sig:
        lines += ["---", SIG]
    lines += [""] + corr_lines(names, F["V"])
    return lines


def glmer_print(formula, data, F, names, group, fam, control=True):
    lines = ["Generalized linear mixed model fit by maximum likelihood (Adaptive",
             "  Gauss-Hermite Quadrature, nAGQ = 10) [glmerMod]",
             " Family: binomial  ( logit )" if fam == "binomial" else " Family: poisson  ( log )",
             f"Formula: {formula}", f"   Data: {data}"]
    if control:
        lines.append('Control: glmerControl(optimizer = "bobyqa")')
    t4 = [round(F["aic"], 1), round(F["bic"], 1), round(F["ll"], 1), round(F["dev"], 1)]
    vals = [s.strip() for s in R.format_real(t4 + [float(F["df_res"])], 7)][:4] + [str(F["df_res"])]
    lines += [""] + R.named_vector(["AIC", "BIC", "logLik", "deviance", "df.resid"], vals) + [""]
    lines += resid_lines(F["scaled"]) + ["", "Random effects:"]
    lines += vc_lines([(group, "(Intercept)", F["var"])])
    lines += [f"Number of obs: {F['n']}, groups:  {group}, {F['ngroups']}", "", "Fixed effects:"]
    cols = [("Estimate", list(F["b"]), "cs"), ("Std. Error", list(F["se"]), "cs"),
            ("z value", list(F["z"]), "tst"), ("Pr(>|z|)", list(F["p"]), "p")]
    sig = any(p < 0.1 for p in F["p"])
    lines += R.coefmat(names, cols, list(F["p"]) if sig else None)
    if sig:
        lines += ["---", SIG]
    lines += [""] + corr_lines(names, F["V"])
    return lines


def shown(lines, header):
    """Read back the numbers as printed (so answers match the screen)."""
    out = {}
    start = lines.index(header)
    for l in lines[start + 1:]:
        if not l or l.startswith("---"):
            break
        parts = l.split()
        out[parts[0]] = parts[1:]
    return out


def fnum(s):
    return float(s.replace("< 2e-16", "1e-16").replace("<", ""))


def vc_shown(lines):
    """Variance values as printed in 'Random effects:'."""
    i = lines.index("Random effects:")
    res = {}
    for l in lines[i + 2:]:
        if l.startswith("Number of obs"):
            break
        parts = l.split()
        if parts[0] == "Residual":
            res["Residual"] = (float(parts[1]), float(parts[2]))
        else:
            res[parts[0]] = (float(parts[2]), float(parts[3]))
    return res


def pr(p):
    return "p < 0,001" if p < 0.001 else "p = " + ru(p, 3 if p >= 0.01 else 4)


def pct(x, d=1):
    return ru(100 * x, d) + " %"


def sig_word(p):
    return "значим" if p < 0.05 else "не значим"


# ---------------------------------------------------------------------------
# emmeans printing
def emm_fmt(vals, sig=3):
    return [s.strip() for s in R.format_real([float(f"{v:.{sig}g}") for v in vals], sig)]


def emm_table(cols, left):
    """cols: [(header, [str], left_align)]"""
    w = [max(len(h), max(len(v) for v in vals)) for h, vals, _ in cols]
    out = [" " + " ".join((h.ljust(wi) if la else h.rjust(wi)) for (h, _, la), wi in zip(cols, w))]
    n = len(cols[0][1])
    for i in range(n):
        out.append(" " + " ".join((vals[i].ljust(wi) if la else vals[i].rjust(wi)) for (_, vals, la), wi in zip(cols, w)))
    return [l.rstrip() for l in out]


def emmeans_block(F, levels, fac, ref_names):
    """LS means for a one-factor fixed part (other fixed effects absent) from an lmm fit."""
    p = len(F["b"])
    k = len(levels)
    Ls, est, se, df = [], [], [], []
    for j in range(k):
        L = np.zeros(p)
        L[0] = 1
        if j > 0:
            L[j] = 1
        Ls.append(L)
        est.append(float(L @ F["b"]))
        se.append(math.sqrt(float(L @ F["V"] @ L)))
        df.append(F["satt"](L))
    tq = [stats.t.ppf(0.975, d) for d in df]
    lo = [e - t * s for e, t, s in zip(est, tq, se)]
    hi = [e + t * s for e, t, s in zip(est, tq, se)]
    dfs = emm_fmt(df, 3)
    lines = [f"> emmeans(m, specs = \"{fac}\")"]
    lines += emm_table([(fac, levels, True), ("emmean", emm_fmt(est), False), ("SE", emm_fmt(se), False), ("df", dfs, False),
                        ("lower.CL", emm_fmt(lo), False), ("upper.CL", emm_fmt(hi), False)], True)
    lines += ["", "Degrees-of-freedom method: kenward-roger ", "Confidence level used: 0.95 "]
    # pairs, Tukey
    pr_rows = []
    for a in range(k):
        for b in range(a + 1, k):
            L = Ls[a] - Ls[b]
            e = float(L @ F["b"])
            s = math.sqrt(float(L @ F["V"] @ L))
            d = F["satt"](L)
            t = e / s
            pv = float(stats.studentized_range.sf(abs(t) * math.sqrt(2), k, d)) if k > 2 else float(2 * stats.t.sf(abs(t), d))
            pr_rows.append((f"{levels[a]} - {levels[b]}", e, s, d, t, pv))
    lines += ["", "> pairs(emmeans(m, specs = \"" + fac + "\"), adjust = \"tukey\")"]
    pvs = []
    for r in pr_rows:
        pvs.append("<.0001" if r[5] < 0.0001 else f"{r[5]:.4f}")
    lines += emm_table([("contrast", [r[0] for r in pr_rows], True), ("estimate", emm_fmt([r[1] for r in pr_rows], 4), False),
                        ("SE", emm_fmt([r[2] for r in pr_rows]), False), ("df", emm_fmt([r[3] for r in pr_rows]), False),
                        ("t.ratio", [f"{r[4]:.3f}" for r in pr_rows], False), ("p.value", pvs, False)], True)
    lines += ["", "Degrees-of-freedom method: kenward-roger "]
    if k > 2:
        lines.append(f"P value adjustment: tukey method for comparing a family of {k} estimates ")
    return lines, dict(est=emm_fmt(est), se=emm_fmt(se), pairs=pr_rows, pvs=pvs)


# ---------------------------------------------------------------------------
# contexts
H_CTX = [
    dict(key="scc", y="lnscc", yru="логарифм числа соматических клеток (lnscc)", unit="", log=True, ylab="SCC",
         grp="herdid", gru="ферма", gpl="ферм", gacc="ферму", ind="корова", indpl="коров", data="cows", gforms=("ферма", "фермы", "ферм"),
         iforms=("корова", "коровы", "коров"), gins="фермами", iins="коровами", same="на той же ферме", sgen="первотёлок", who="первотёлки",
         ng=(30, 45), m=(15, 60), b0=4.7, tau=0.39, sig=1.25, dec=2,
         facs=[dict(v="heifer", lev=("0", "1"), name="heifer1", ru="первотёлки (heifer = 1) против старших коров", short="первотёлки",
                    eff=(-0.9, -0.5), lvl=1, prob=0.3)],
         cont=[dict(v="dim_sc", ru="дни лактации (dim_sc = (dim − 60) / 100)", eff=(0.15, 0.4), lvl=1, mean=0, sd=0.7, step="+100 дней лактации")],
         farm=dict(v="herd_size_sc", ru="размер стада (herd_size_sc = (herd_size − 50) / 100)", eff=(-0.3, 0.3), mean=0, sd=0.5)),
    dict(key="adg", y="adg", yru="среднесуточный привес (adg, г/сут)", unit="г/сут", log=False, ylab="ADG",
         grp="farm", gru="ферма", gpl="ферм", gacc="ферму", ind="свинья", indpl="свиней", data="pigs", gforms=("ферма", "фермы", "ферм"),
         iforms=("свинья", "свиньи", "свиней"), gins="фермами", iins="свиньями", same="на той же ферме", sgen="свинок", who="свинки",
         ng=(15, 30), m=(15, 40), b0=820, tau=35, sig=70, dec=1,
         facs=[dict(v="sex", lev=("barrow", "gilt"), name="sexgilt", ru="свинки (gilt) против боровков (barrow)", short="свинки",
                    eff=(-40, -15), lvl=1, prob=0.5)],
         cont=[dict(v="start_wt", ru="масса при постановке (start_wt, кг)", eff=(3, 8), lvl=1, mean=28, sd=4, step="+1 кг массы при постановке")],
         farm=dict(v="feeder", lev=("dry", "wet"), name="feederwet", ru="влажное кормление (feeder = wet) против сухого", eff=(-10, 40))),
    dict(key="bf", y="bf", yru="толщина шпика (bf, мм)", unit="мм", log=False, ylab="шпик",
         grp="farm", gru="ферма", gpl="ферм", gacc="ферму", ind="свинья", indpl="свиней", data="carcass", gforms=("ферма", "фермы", "ферм"),
         iforms=("свинья", "свиньи", "свиней"), gins="фермами", iins="свиньями", same="на той же ферме", sgen="свинок", who="свинки",
         ng=(15, 30), m=(20, 50), b0=10, tau=1.4, sig=2.6, dec=2,
         facs=[dict(v="sex", lev=("barrow", "gilt"), name="sexgilt", ru="свинки (gilt) против боровков (barrow)", short="свинки",
                    eff=(-2.5, -1.0), lvl=1, prob=0.5)],
         cont=[dict(v="hcw", ru="масса туши (hcw, кг)", eff=(0.05, 0.12), lvl=1, mean=95, sd=8, step="+1 кг массы туши")],
         farm=dict(v="genetics", lev=("A", "B"), name="geneticsB", ru="генетика B против A", eff=(-2, 1))),
    dict(key="wwt", y="wwt", yru="масса поросёнка при отъёме (wwt, кг)", unit="кг", log=False, ylab="масса при отъёме",
         grp="sow", gru="свиноматка (гнездо)", gpl="гнёзд", gacc="гнездо", ind="поросёнок", indpl="поросят", data="piglets", gforms=("гнездо", "гнезда", "гнёзд"),
         iforms=("поросёнок", "поросёнка", "поросят"), gins="гнёздами", iins="поросятами", same="в том же гнезде", sgen="хрячков", who="хрячка",
         ng=(40, 80), m=(8, 14), b0=6.4, tau=0.65, sig=0.95, dec=2,
         facs=[dict(v="sex", lev=("F", "M"), name="sexM", ru="хрячки (M) против свинок (F)", short="хрячки",
                    eff=(0.05, 0.25), lvl=1, prob=0.5)],
         cont=[dict(v="bw_c", ru="масса при рождении (bw_c = масса − 1,4 кг)", eff=(1.6, 2.6), lvl=1, mean=0, sd=0.3, step="+1 кг массы при рождении")],
         farm=dict(v="creep", lev=("no", "yes"), name="creepyes", ru="подкормка в гнезде (creep = yes) против без неё", eff=(0.1, 0.5))),
]


def sim_h(rng, c, with_farm):
    ng = int(rng.integers(*c["ng"]))
    sizes = rng.integers(c["m"][0], c["m"][1] + 1, ng)
    g = np.repeat(np.arange(1, ng + 1), sizes)
    n = len(g)
    f = c["facs"][0]
    ct = c["cont"][0]
    beta_f = rng.uniform(*f["eff"])
    beta_c = rng.uniform(*ct["eff"])
    xf = (rng.random(n) < f["prob"]).astype(float)
    xc = np.round(rng.normal(ct["mean"], ct["sd"], n), 2)
    tau = c["tau"] * rng.uniform(0.6, 1.4)
    sig = c["sig"] * rng.uniform(0.85, 1.15)
    u = rng.normal(0, tau, ng)
    y = c["b0"] + beta_f * xf + beta_c * (xc - ct["mean"]) + u[g - 1] + rng.normal(0, sig, n)
    X = [np.ones(n), xf, xc]
    names = ["(Intercept)", f["name"], ct["v"]]
    fd = c["farm"]
    if with_farm:
        if "lev" in fd:
            gf = (rng.random(ng) < 0.5).astype(float)
            beta_g = rng.uniform(*fd["eff"])
            y = y + beta_g * gf[g - 1]
            X.append(gf[g - 1])
            names.append(fd["name"])
        else:
            gf = np.round(rng.normal(fd["mean"], fd["sd"], ng), 2)
            beta_g = rng.uniform(*fd["eff"])
            y = y + beta_g * gf[g - 1]
            X.append(gf[g - 1])
            names.append(fd["v"])
    y = np.round(y, c["dec"])
    return np.column_stack(X), y, g, names


def gen_lmmh(rng, c):
    with_farm = rng.random() < 0.5
    for _ in range(20):
        X, y, g, names = sim_h(rng, c, with_farm)
        F = lmm(y, X, g, names)
        if F["theta"] > 0.05:
            break
    f, ct, fd = c["facs"][0], c["cont"][0], c["farm"]
    terms = [f["v"], ct["v"]] + ([fd["v"]] if with_farm else [])
    formula = f"{c['y']} ~ {' + '.join(terms)} + (1 | {c['grp']})"
    lines = ["> library(lmerTest)", f"> m <- lmer({c['y']} ~ {' + '.join(terms)} + (1|{c['grp']}), REML = TRUE, data = {c['data']})",
             "> summary(m)"] + lmer_print(formula, c["data"], F, names, c["grp"])
    fx = shown(lines, "Fixed effects:")
    vc = vc_shown(lines)
    tau2, sig2 = vc[c["grp"]][0], vc["Residual"][0]
    icc = tau2 / (tau2 + sig2)
    b0 = fnum(fx["(Intercept)"][0])
    bf = fnum(fx[f["name"]][0])
    bc = fnum(fx[ct["v"]][0])
    pf = F["p"][1]
    pc = F["p"][2]
    gw = c["gru"].split(" ")[0]
    qs_req, qs_opt = [], []
    qs_req.append(num(f"Какая доля необъяснённой изменчивости {c['y']} приходится на различия между {c['gins']} (ICC), %?",
                      100 * icc,
                      f"Берём столбец Variance: между группами ({c['grp']}) = {rua(tau2, 4)}, внутри (Residual) = {rua(sig2, 4)}. "
                      f"ICC = {rua(tau2, 4)} / ({rua(tau2, 4)} + {rua(sig2, 4)}) = {rua(icc, 3)} = <b>{rua(100 * icc, 3)} %</b>. "
                      f"Остальные {rua(100 - 100 * icc, 3)} % — разница между животными внутри одной группы. "
                      f"Это же число — корреляция двух животных из одной группы.", rel=0.03, abs_=0.5, unit="%"))
    unit = (" " + c["unit"]) if c["unit"] else ""
    direction = "выше" if bf > 0 else "ниже"
    log_note = ""
    if c["log"]:
        log_note = f" Поскольку y — логарифм, в разах это e<sup>{rua(bf, 3)}</sup> = {rua(math.exp(bf), 3)}."
    qs_req.append(mc(rng, f"Как прочитать строку {f['name']} = {rua(bf, 4)}?",
                     f"У {c['sgen']} {c['y']} на {rua(abs(bf), 3)}{unit} {direction}, чем у референтной группы {c['same']}",
                     [f"{c['gforms'][1].capitalize()}, где больше {c['sgen']}, в среднем на {rua(abs(bf), 3)}{unit} {direction} остальных",
                      f"У {c['sgen']} {c['y']} на {rua(abs(bf), 3)} % {direction}, чем у референтной группы {c['same']}",
                      f"У {c['sgen']} разброс {c['y']} на {rua(abs(bf), 3)} больше, чем у референтной группы {c['same']}"],
                     f"Фиксированный эффект читается как в обычной регрессии: {f['ru']}. "
                     f"Разница — «при прочих равных», включая одну и ту же группу ({c['same']}): случайный интерсепт уже снял различия между {c['gins']}.{log_note}"))
    qs_opt.append(mc(rng, f"Что показывает строка Residual в блоке Random effects (Variance = {rua(sig2, 4)})?",
                     f"Разброс внутри групп: между {c['iins']} {c['same']}",
                     [f"Разброс между группами ({c['grp']}): насколько отличаются их средние", "Ошибку оценки интерсепта (Std. Error из Fixed effects)",
                      "Сумму квадратов остатков всей модели, делённую на n"],
                     f"В random intercept модели две дисперсии: между группами ({c['grp']}, {rua(tau2, 4)}) и внутри групп (Residual, {rua(sig2, 4)}). "
                     f"Их сумма — вся необъяснённая изменчивость."))
    qs_opt.append(num(f"Сколько групп ({c['grp']}) в данных?", F["ngroups"],
                      f"Строка «Number of obs: {F['n']}, groups: {c['grp']}, {F['ngroups']}». Наблюдений {F['n']}, групп <b>{F['ngroups']}</b>. "
                      f"Для эффектов, которые меняются только между группами, информации столько, сколько групп, а не наблюдений.", rel=0, abs_=0.5))
    sd_tau = vc[c["grp"]][1]
    qs_opt.append(mc(rng, f"Std.Dev. у {c['grp']} = {rua(sd_tau, 4)}. Что это значит?",
                     f"Средние групп обычно отличаются от общего среднего на ±{rua(sd_tau, 3)}{unit}",
                     [f"Это стандартная ошибка интерсепта: точность общего среднего",
                      f"Каждое животное обычно отклоняется от среднего своей группы на ±{rua(sd_tau, 3)}{unit}",
                      f"Эффект группы значим, раз {rua(sd_tau, 3)} больше нуля, p не нужен"],
                     f"Std.Dev. — корень из Variance: √{rua(tau2, 4)} = {rua(sd_tau, 3)}. Случайные сдвиги групп u ~ N(0, {rua(sd_tau, 3)}²): "
                     f"примерно 95 % групп лежат в пределах ±1,96 × {rua(sd_tau, 3)} = ±{rua(1.96 * sd_tau, 3)}{unit} от общего среднего."))
    yes_t, no_t = f"Да, эффект значим на уровне 0,05: {pr(pc)}", f"Нет, не значим на уровне 0,05: {pr(pc)}"
    qs_opt.append(mc(rng, f"Значим ли эффект {ct['v']} на уровне 0,05?",
                     yes_t if pc < 0.05 else no_t,
                     [no_t if pc < 0.05 else yes_t, f"Нельзя сказать: в выводе lmer нет p-value",
                      f"Да, потому что df = {rua(F['df'][2], 4)} — это много"],
                     f"lmerTest добавляет df (Satterthwaite) и Pr(>|t|). У {ct['v']} {pr(pc)}, значит эффект {sig_word(pc)}. "
                     f"Оценка {rua(bc, 4)}: при {ct['step']} {c['y']} меняется в среднем на {rua(bc, 3)}{unit} (внутри одной и той же группы)."))
    xs = 1.0 if ct["v"] in ("dim_sc",) else round(float(ct["mean"] + (1 if ct["sd"] < 2 else 5) * rng.choice([-1, 1])), 2)
    if ct["v"] == "bw_c":
        xs = float(rng.choice([-0.3, -0.2, 0.2, 0.3]))
    pred = b0 + bf + bc * xs + (0 if not with_farm else 0)
    farm_term = ""
    if with_farm:
        bg = fnum(fx[names[3]][0])
        farm_term = f" при {fd['v']} = {fd['lev'][0]}" if "lev" in fd else f" при {names[3]} = 0"
    qs_opt.append(num(f"Предскажи {c['y']} для {c['who']} с {ct['v']} = {ru(xs, 2)} в «типичной» группе (u = 0){farm_term}.",
                      pred, f"ŷ = b₀ + {f['name']} + {ct['v']} × {ru(xs, 2)} = {rua(b0, 5)} + ({rua(bf, 4)}) + {rua(bc, 4)} × {ru(xs, 2)} = <b>{rua(pred, 4)}</b>. "
                            f"«Типичная» группа — со случайным сдвигом u = 0. Для конкретной фермы добавили бы её BLUP из ranef(m).",
                      rel=0.01, abs_=0.02))
    if with_farm:
        dfg = F["df"][3]
        qs_req.append(mc(rng, f"Почему у {names[3]} df ≈ {rua(dfg, 3)}, а у {f['name']} df ≈ {rua(F['df'][1], 3)}?",
                         f"Он меняется только между группами — сведений столько, сколько групп",
                         [f"Его измеряли с ошибкой, и Satterthwaite за это урезал степени свободы",
                          f"У него меньше уровней, чем у признака животного, поэтому и df меньше",
                          "Satterthwaite всегда даёт меньше df последнему предиктору в формуле"],
                         f"Satterthwaite считает «эффективное» число независимых сведений для каждого коэффициента. "
                         f"Признак уровня группы ({names[3]}) сравнивает целые группы, поэтому df близок к числу групп минус параметры уровня группы. "
                         f"Признак уровня животного ({f['name']}) сравнивает животных внутри групп — df близок к числу наблюдений. "
                         f"Обычный lm() дал бы обоим ~{F['n'] - len(names)} df и слишком маленький p для {names[3]}."))
    else:
        qs_req.append(mc(rng, f"Что делает часть (1 | {c['grp']}) в формуле?",
                         f"Случайный сдвиг среднего для каждой группы — учёт сходства внутри них",
                         [f"Добавляет {c['grp']} как обычный фактор: отдельный фиксированный коэффициент на каждую группу",
                          f"Исключает из анализа группы, где меньше двух животных", f"Делит {c['y']} на среднее своей группы, чтобы убрать различия"],
                         f"(1 | {c['grp']}) — случайный интерсепт. Группы считаются выборкой из совокупности ферм/гнёзд: модель оценивает не "
                         f"{F['ngroups']} отдельных сдвигов, а одну дисперсию между группами ({rua(tau2, 4)}). Так учитывается кластеризация "
                         f"и выводы распространяются на совокупность групп."))
    if c["log"]:
        qs_opt.append(num(f"Во сколько раз SCC у первотёлок отличается от старших коров той же фермы? (e^b)", math.exp(bf),
                          f"y = ln(SCC), значит разница логарифмов → отношение: e<sup>{rua(bf, 4)}</sup> = <b>{rua(math.exp(bf), 3)}</b>. "
                          f"SCC у первотёлок примерно в {rua(math.exp(bf), 3)} раза от уровня старших коров ({rua(100 * (math.exp(bf) - 1), 3)} %).", rel=0.01))
    qs = pick(rng, qs_req, qs_opt)
    intro = (f"Наблюдательные данные: {nn(F['n'], c['iforms'])} из {F['ngroups']} {pl(F['ngroups'], ('фермы', 'ферм', 'ферм')) if c['grp'] != 'sow' else pl(F['ngroups'], ('гнезда', 'гнёзд', 'гнёзд'))}. Животные одной группы похожи друг на друга "
             f"(общие корм, климат, менеджмент), поэтому группа ({c['grp']}) включена как случайный интерсепт. Исход — {c['yru']}. "
             f"Предикторы: {f['ru']}; {ct['ru']}" + (f"; {fd['ru']} — признак уровня группы" if with_farm else "") + ".")
    sub = "lmmh"
    return dict(cat="mixed", sub=sub, title=f"lmer: {c['y']} ~ … + (1 | {c['grp']})", intro=intro, console="\n".join(lines), qs=qs)


# ---------------------------------------------------------------------------
E_CTX = [
    dict(key="adg", y="adg", yru="среднесуточный привес (г/сут)", unit="г/сут", data="trial", what="рационы на доращивании-откорме",
         lev=["A", "B"], eff=[0, (15, 50)], b0=780, tau=(15, 35), sig=(55, 80), pen_n=(8, 20), pens=(12, 30), dec=0),
    dict(key="adg3", y="adg", yru="среднесуточный привес (г/сут)", unit="г/сут", data="trial", what="три кормовые добавки",
         lev=["Ctrl", "Enz", "Prob"], eff=[0, (5, 35), (10, 45)], b0=760, tau=(15, 35), sig=(55, 80), pen_n=(8, 18), pens=(15, 30), dec=0),
    dict(key="gain", y="gain", yru="прирост за 21 день на доращивании (кг)", unit="кг", data="nursery", what="дозы оксида цинка",
         lev=["Z0", "Z2500"], eff=[0, (0.4, 1.4)], b0=8.5, tau=(0.3, 0.7), sig=(0.9, 1.4), pen_n=(10, 25), pens=(12, 24), dec=2),
]


def gen_lmme(rng, c):
    k = len(c["lev"])
    for _ in range(20):
        npen = int(rng.integers(*c["pens"]))
        npen -= npen % k
        per = int(rng.integers(*c["pen_n"]))
        tr = np.array([c["lev"][i % k] for i in range(npen)])
        rng.shuffle(tr)
        sizes = np.clip(per + rng.integers(-1, 2, npen), 5, None)
        pen = np.repeat(np.arange(1, npen + 1), sizes)
        trt = tr[pen - 1]
        effs = [0.0] + [rng.uniform(*e) for e in c["eff"][1:]]
        if rng.random() < 0.25:
            effs = [e * rng.uniform(0.05, 0.3) for e in effs]
        tau = rng.uniform(*c["tau"])
        sig = rng.uniform(*c["sig"])
        mu = c["b0"] + np.array([effs[c["lev"].index(t)] for t in trt])
        y = mu + rng.normal(0, tau, npen)[pen - 1] + rng.normal(0, sig, len(pen))
        y = np.round(y, c["dec"])
        X = np.column_stack([np.ones(len(y))] + [(trt == l).astype(float) for l in c["lev"][1:]])
        names = ["(Intercept)"] + [f"trt{l}" for l in c["lev"][1:]]
        F = lmm(y, X, pen, names)
        if F["theta"] > 0.08:
            break
    formula = f"{c['y']} ~ trt + (1 | pen)"
    show_lm = rng.random() < 0.55
    lines = ["> library(lmerTest)", f"> m <- lmer({c['y']} ~ trt + (1|pen), data = {c['data']})", "> summary(m)"]
    lines += lmer_print(formula, c["data"], F, names, "pen")
    # naive lm for comparison
    XtX = X.T @ X
    bl = np.linalg.solve(XtX, X.T @ y)
    rl = y - X @ bl
    s2 = rl @ rl / (len(y) - X.shape[1])
    sel = np.sqrt(np.diag(s2 * np.linalg.inv(XtX)))
    tl = bl / sel
    pl = 2 * stats.t.sf(np.abs(tl), len(y) - X.shape[1])
    if show_lm:
        lines += ["", f"> coef(summary(lm({c['y']} ~ trt, data = {c['data']})))"]
        lines += R.matrix(names, ["Estimate", "Std. Error", "t value", "Pr(>|t|)"],
                          [list(bl), list(sel), list(tl), list(pl)], digits=7)
    fx = shown(lines, "Fixed effects:")
    vc = vc_shown(lines)
    tau2, sig2 = vc["pen"][0], vc["Residual"][0]
    icc = tau2 / (tau2 + sig2)
    j = int(rng.integers(1, k))
    lj = c["lev"][j]
    bj = fnum(fx[names[j]][0])
    b0 = fnum(fx["(Intercept)"][0])
    dfj = F["df"][j]
    pj = F["p"][j]
    npen = F["ngroups"]
    unit = " " + c["unit"]
    qs_req, qs_opt = [], []
    qs_req.append(mc(rng, f"Что здесь единица эксперимента?", f"Загон ({npen})",
                     [f"Свинья ({F['n']})", "Рацион", "Ферма"],
                     f"Рацион назначали загону целиком, поэтому независимых повторностей {npen}, а не {F['n']}. "
                     f"Свиньи — единицы наблюдения. (1 | pen) говорит модели, что свиньи одного загона связаны."))
    qs_req.append(mc(rng, f"Почему у {names[j]} df ≈ {rua(dfj, 3)}, а не ~{F['n'] - k}?",
                     f"Рацион сравнивают по загонам: загонов {npen} − рационов {k} = {npen - k}",
                     [f"Satterthwaite вычитает по степени свободы за каждую свинью",
                      "Это случайное число: df в lmer не важен",
                      f"Из-за разного числа свиней в загонах часть df отбросили"],
                     f"Для эффекта, который назначен загону, информация — это загоны. В сбалансированном опыте Satterthwaite даёт "
                     f"ровно столько же df, сколько t-тест по средним загонов: {npen} − {k} = <b>{npen - k}</b>."))
    direction = "больше" if bj > 0 else "меньше"
    qs_req.append(mc(rng, f"Как прочитать {names[j]} = {rua(bj, 4)}?",
                     f"На рационе {lj} {c['y']} в среднем на {rua(abs(bj), 3)}{unit} {direction}, чем на {c['lev'][0]}",
                     [f"Рацион {lj} даёт {rua(bj, 3)}{unit} — это и есть среднее группы {lj}",
                      f"На рационе {lj} разброс между загонами на {rua(abs(bj), 3)} {direction}, чем на {c['lev'][0]}",
                      f"Каждая свинья на рационе {lj} дала ровно на {rua(abs(bj), 3)}{unit} {direction}"],
                     f"Строка trt{lj} — разница {lj} − {c['lev'][0]} (референт в интерсепте). Среднее {lj} = {rua(b0, 5)} + ({rua(bj, 4)}) = {rua(b0 + bj, 4)}{unit}. "
                     f"{pr(pj).capitalize()}: разница {sig_word(pj)}."))
    qs_opt.append(num(f"Какая доля изменчивости {c['y']} приходится на загоны (ICC), %?", 100 * icc,
                      f"ICC = pen / (pen + Residual) = {rua(tau2, 4)} / ({rua(tau2, 4)} + {rua(sig2, 4)}) = <b>{rua(100 * icc, 3)} %</b>. "
                      f"Свиньи одного загона похожи — вот почему их нельзя считать независимыми.", rel=0.03, abs_=0.5, unit="%"))
    qs_opt.append(num(f"Средний {c['y']} на рационе {lj} (по выводу lmer)?", b0 + bj,
                      f"(Intercept) + {names[j]} = {rua(b0, 5)} + ({rua(bj, 4)}) = <b>{rua(b0 + bj, 4)}</b>{unit}.", rel=0.005))
    if show_lm:
        pl_j = float(pl[j])
        qs_req.append(mc(rng, f"lm() без загона даёт для {names[j]} SE {rua(sel[j], 3)} и {pr(pl_j)}; lmer — SE {rua(F['se'][j], 3)} и {pr(pj)}. Каким верить?",
                         "lmer: свиньи одного загона не независимы, а lm это игнорирует",
                         ["lm: у него больше степеней свободы, значит, он точнее", "Обоим: оценки разниц одинаковые, значит, и выводы будут одинаковые",
                          "lm: в нём нет случайных эффектов, поэтому он надёжнее"],
                         f"Оценки разниц в сбалансированном опыте совпадают, а SE — нет. lm делает вид, что у нас {F['n']} независимых свиней; "
                         f"на самом деле рацион сравнивается на {npen} загонах. lmer с (1 | pen) это учитывает: SE больше, p честнее. "
                         f"Тот же результат даст t-тест или ANOVA по средним загонов."))
    else:
        qs_opt.append(mc(rng, "Чем эта модель лучше lm(adg ~ trt) по всем свиньям?" if c["y"] == "adg" else f"Чем эта модель лучше lm({c['y']} ~ trt) по всем животным?",
                         "Учитывает связь свиней внутри загона и не завышает n",
                         ["Даёт другие, более точные оценки средних по рационам", "Автоматически находит и убирает выбросы",
                          "Позволяет не проверять предпосылки модели"],
                         "Псевдорепликация: lm по всем свиньям занижает SE и p. Случайный интерсепт загона возвращает честную точность — "
                         "как анализ средних по загонам."))
    qs = pick(rng, qs_req, qs_opt)
    intro = (f"Опыт: {c['what']}. {nn(npen, ('загон', 'загона', 'загонов'))} распределили по вариантам жребием ({', '.join(c['lev'])}); вариант задаётся на загон целиком, "
             f"а {c['y']} измеряли у каждой свиньи ({F['n']} голов). Исход — {c['yru']}. Загон — случайный интерсепт.")
    return dict(cat="mixed", sub="lmme", title=f"lmer: {c['y']} ~ trt + (1 | pen)", intro=intro, console="\n".join(lines), qs=qs)


# ---------------------------------------------------------------------------
B_RCBD = [
    dict(y="ofg", yru="конверсия корма за весь опыт (корм : прирост)", unit="", what="пять кормовых добавок", lev=["1", "2", "3", "4", "5"],
         eff=(-0.12, 0.08), b0=2.55, tau=(0.04, 0.09), sig=(0.03, 0.07), blocks=(5, 9), dec=3, better="ниже"),
    dict(y="adg", yru="среднесуточный привес загона (г/сут)", unit="г/сут", what="четыре рациона", lev=["A", "B", "C", "D"],
         eff=(-15, 45), b0=790, tau=(20, 45), sig=(15, 30), blocks=(5, 10), dec=1, better="выше"),
    dict(y="gain", yru="прирост загона за доращивание (кг на голову)", unit="кг", what="три подкислителя", lev=["Ctrl", "OrgA", "OrgB"],
         eff=(-0.2, 1.0), b0=14.0, tau=(0.5, 1.0), sig=(0.3, 0.6), blocks=(6, 10), dec=2, better="выше"),
]


def gen_rcbd(rng, c):
    k = len(c["lev"])
    for _ in range(30):
        nb = int(rng.integers(*c["blocks"]))
        blk = np.repeat(np.arange(1, nb + 1), k)
        trt = np.tile(np.array(c["lev"]), nb)
        effs = [0.0] + list(rng.uniform(*c["eff"], k - 1))
        tau = rng.uniform(*c["tau"])
        sig = rng.uniform(*c["sig"])
        y = c["b0"] + np.array([effs[c["lev"].index(t)] for t in trt]) + rng.normal(0, tau, nb)[blk - 1] + rng.normal(0, sig, len(blk))
        y = np.round(y, c["dec"])
        X = np.column_stack([np.ones(len(y))] + [(trt == l).astype(float) for l in c["lev"][1:]])
        names = ["(Intercept)"] + [f"Trt{l}" for l in c["lev"][1:]]
        F = lmm(y, X, blk, names)
        if F["theta"] > 0.1:
            break
    formula = f"{c['y']} ~ Trt + (1 | Block)"
    lines = ["> library(lmerTest); library(emmeans)", f"> m <- lmer({c['y']} ~ Trt + (1|Block), data = pens)", "> summary(m)"]
    lines += lmer_print(formula, "pens", F, names, "Block")
    el, E = emmeans_block(F, c["lev"], "Trt", names)
    lines += [""] + el
    vc = vc_shown(lines)
    tau2, sig2 = vc["Block"][0], vc["Residual"][0]
    pairs = E["pairs"]
    npairs = len(pairs)
    sigp = [r for r, pv in zip(pairs, E["pvs"]) if (pv == "<.0001" or float(pv) < 0.05)]
    nsp = [r for r, pv in zip(pairs, E["pvs"]) if not (pv == "<.0001" or float(pv) < 0.05)]
    unit = (" " + c["unit"]) if c["unit"] else ""
    qs_req, qs_opt = [], []
    qs_req.append(mc(rng, "Почему блок (Block) взят случайным эффектом?",
                     "Блоки — выборка условий: важен их разброс, а не каждый блок",
                     ["Блок — главный изучаемый фактор, поэтому его эффект и проверяем отдельно", "Без случайного блока не посчитать средние по вариантам",
                      "Случайный эффект сам находит и убирает выбросы"],
                     "Структура опыта (блоки) идёт в случайную часть, изучаемые варианты (Trt) — в фиксированную. "
                     f"Модель оценивает одну дисперсию между блоками ({rua(tau2, 4)}) и вычитает её из ошибки сравнения вариантов."))
    qs_req.append(num("Сколько попарных сравнений делает Tukey?", npairs,
                      f"k = {k} вариантов: k(k − 1)/2 = {k}·{k - 1}/2 = <b>{npairs}</b>. Tukey поправляет p-value на всё семейство сравнений, "
                      "чтобы общая вероятность хотя бы одной ложной находки оставалась 5 %.", rel=0, abs_=0.5))
    if sigp and nsp:
        r = sigp[int(rng.integers(len(sigp)))]
        wr = [x[0] for x in nsp]
        rng.shuffle(wr)
        wr = (wr + ["Ни одна пара", "Все пары"])[:3]
        qs_req.append(mc(rng, "Какая пара вариантов различается по Tukey (p.value < 0,05)?", r[0], wr,
                         f"Смотрим столбец p.value в выводе pairs(): у {r[0]} p.value = {E['pvs'][pairs.index(r)]} < 0,05. "
                         "Это уже скорректированный p — второй раз поправку делать не нужно."))
    elif not sigp:
        qs_req.append(mc(rng, "Какие пары вариантов различаются по Tukey (p.value < 0,05)?", "Ни одна пара",
                         [f"Только {pairs[0][0]}", f"Только {pairs[-1][0]} и {pairs[0][0]}", "Все пары без исключения"],
                         "Все скорректированные p.value больше 0,05: при поправке на множественность ни одна разница не подтверждена."))
    else:
        qs_req.append(mc(rng, "Какие пары вариантов различаются по Tukey (p.value < 0,05)?", "Все пары",
                         [pairs[0][0] + " и больше никакие", "Ни одна пара", "Только пары с вариантом " + c["lev"][0]],
                         "Все скорректированные p.value меньше 0,05."))
    jj = int(rng.integers(k))
    qs_opt.append(num(f"LS-среднее (emmean) варианта {c['lev'][jj]}?", float(E["est"][jj]),
                      f"Строка {c['lev'][jj]} в таблице emmeans: emmean = <b>{E['est'][jj]}</b>. Это (Intercept) "
                      + (f"+ Trt{c['lev'][jj]} " if jj > 0 else "") + "— среднее для «типичного» блока.", rel=0.005))
    qs_opt.append(num("Какая доля изменчивости приходится на блоки, %?", 100 * tau2 / (tau2 + sig2),
                      f"Block / (Block + Residual) = {rua(tau2, 4)} / ({rua(tau2, 4)} + {rua(sig2, 4)}) = <b>{rua(100 * tau2 / (tau2 + sig2), 3)} %</b>. "
                      "Чем больше эта доля, тем больше выиграли от блоков: эта часть разброса не попала в ошибку.", rel=0.03, abs_=0.5, unit="%"))
    # summary says "significant vs reference", Tukey does not
    for j in range(1, k):
        if F["p"][j] < 0.05:
            r = pairs[j - 1]
            pv = E["pvs"][j - 1]
            if pv != "<.0001" and float(pv) >= 0.05:
                qs_req.append(mc(rng, f"В summary у {names[j]} {pr(F['p'][j])}, а в pairs() для {r[0]} p.value = {pv}. Почему?",
                                 "pairs() с Tukey поправляет на число сравнений, summary — нет",
                                 ["В summary другой референтный уровень, поэтому и сравнение там другое", "pairs() считает по другим данным — средним блоков",
                                  "Это просто ошибка округления в выводе emmeans"],
                                 f"Обе строки сравнивают {c['lev'][j]} и {c['lev'][0]}, оценки одинаковые. Но pairs() проверяет сразу {npairs} пар и "
                                 "раздувает p-value, чтобы общий риск ложной находки был 5 %. Если вопрос про одну заранее выбранную пару — "
                                 "годится обычный p; если смотрим все пары — нужен Tukey."))
                break
    dft = F["df"][1]
    nb = F["ngroups"]
    qs_opt.append(mc(rng, f"Почему df у сравнений ≈ {rua(dft, 3)}?",
                     f"(вариантов − 1)(блоков − 1) = ({k} − 1)({nb} − 1) = {(k - 1) * (nb - 1)}",
                     [f"Число загонов минус число вариантов и блоков: {k * nb} − {k} − {nb} = {k * nb - k - nb}", f"Блоков минус один: {nb} − 1 = {nb - 1}",
                      "Satterthwaite выдаёт случайное число, смысла нет"],
                     "В сбалансированном RCBD смешанная модель даёт те же df и тот же F-тест, что и aov(y ~ Trt + Block)."))
    qs = pick(rng, qs_req, qs_opt)
    intro = (f"Опыт RCBD: {c['what']} ({', '.join(c['lev'])}). Загоны разбили на {nb} блоков по стартовой массе; в каждом блоке по одному загону на вариант, "
             f"жребий внутри блока. Исход — {c['yru']} (одна строка на загон; лучше, когда {c['better']}). Блок — случайный интерсепт; "
             f"после модели — LS-средние и попарные сравнения Тьюки.")
    return dict(cat="mixed", sub="lmme", title=f"lmer + emmeans: {c['y']} ~ Trt + (1 | Block)", intro=intro, console="\n".join(lines), qs=qs)


# ---------------------------------------------------------------------------
RM_CTX = [
    dict(y="adg", yru="среднесуточный привес гнезда (кг/сут)", yacc="среднесуточный привес гнезда", unit="кг/сут", grp="litter", gru="гнездо",
         same_acc="Одно и то же гнездо", gen1="одного гнезда", lev=["A", "B"],
         days=["7", "14", "21"], b0=0.22, day_eff=(0.03, 0.08), trt0=(-0.01, 0.02), inter=(0.0, 0.04), tau=(0.02, 0.05), sig=(0.015, 0.035), n=(16, 30), dec=3,
         what="два заменителя молока"),
    dict(y="wt", yru="масса свиньи (кг)", yacc="массу свиньи (кг)", unit="кг", grp="pig", gru="свинья",
         same_acc="Одну и ту же свинью", gen1="одной свиньи", lev=["Ctrl", "Vac"],
         days=["0", "28", "56", "84"], b0=30, day_eff=(6, 9), trt0=(-0.5, 0.5), inter=(0.0, 1.2), tau=(2.0, 4.0), sig=(1.0, 2.0), n=(20, 40), dec=1,
         what="вакцина против цирковируса (дни после вакцинации)"),
]


def gen_lmmr(rng, c):
    for _ in range(20):
        nsub = int(rng.integers(*c["n"]))
        nsub -= nsub % 2
        tr = np.array([c["lev"][i % 2] for i in range(nsub)])
        rng.shuffle(tr)
        T = len(c["days"])
        sub = np.repeat(np.arange(1, nsub + 1), T)
        day = np.tile(np.arange(T), nsub)
        trt = tr[sub - 1]
        de = np.cumsum([0] + list(rng.uniform(*c["day_eff"], T - 1)))
        t0 = rng.uniform(*c["trt0"])
        it = np.cumsum([0] + list(rng.uniform(*c["inter"], T - 1)))
        if rng.random() < 0.3:
            it = it * 0.1
        tau = rng.uniform(*c["tau"])
        sig = rng.uniform(*c["sig"])
        isB = (trt == c["lev"][1]).astype(float)
        y = c["b0"] + de[day] + isB * (t0 + it[day]) + rng.normal(0, tau, nsub)[sub - 1] + rng.normal(0, sig, len(sub))
        y = np.round(y, c["dec"])
        cols = [np.ones(len(y)), isB] + [(day == d).astype(float) for d in range(1, T)] + [isB * (day == d) for d in range(1, T)]
        X = np.column_stack(cols)
        L1, D = c["lev"][1], c["days"]
        names = ["(Intercept)", f"treatm{L1}"] + [f"Day{d}" for d in D[1:]] + [f"treatm{L1}:Day{d}" for d in D[1:]]
        F = lmm(y, X, sub, names)
        if F["theta"] > 0.1:
            break
    formula = f"{c['y']} ~ treatm * Day + (1 | {c['grp']})"
    lines = ["> library(lmerTest)", f"> m <- lmer({c['y']} ~ treatm * Day + (1|{c['grp']}), data = rm)", "> summary(m)"]
    lines += lmer_print(formula, "rm", F, names, c["grp"])
    fx = shown(lines, "Fixed effects:")
    vc = vc_shown(lines)
    tau2, sig2 = vc[c["grp"]][0], vc["Residual"][0]
    L1, L0, D = c["lev"][1], c["lev"][0], c["days"]
    unit = " " + c["unit"]
    b = {nm: fnum(fx[nm][0]) for nm in names}
    T = len(D)
    dl = int(rng.integers(1, T))
    bt = b[f"treatm{L1}"]
    bi = b[f"treatm{L1}:Day{D[dl]}"]
    eff_d = bt + bi
    qs_req, qs_opt = [], []
    qs_req.append(mc(rng, f"Что показывает строка treatm{L1} = {rua(bt, 4)}?",
                     f"Разницу {L1} − {L0} только в референтный день (Day = {D[0]})",
                     [f"Разницу {L1} − {L0} в среднем по всем дням опыта, без учёта дня", f"Разницу {L1} − {L0} в последний день опыта ({D[-1]})",
                      f"Прирост {c['y']} у группы {L0} от дня {D[0]} до дня {D[-1]} опыта"],
                     f"При взаимодействии treatm * Day главный эффект treatm относится к референтному уровню Day ({D[0]}). "
                     f"В другие дни к нему прибавляется соответствующий коэффициент treatm{L1}:Day…"))
    qs_req.append(num(f"Чему равна разница {L1} − {L0} в день {D[dl]}?", eff_d,
                      f"treatm{L1} + treatm{L1}:Day{D[dl]} = {rua(bt, 4)} + ({rua(bi, 4)}) = <b>{rua(eff_d, 3)}</b>{unit}.", rel=0.01, abs_=0.002))
    qs_req.append(mc(rng, f"Зачем в модели (1 | {c['grp']})?",
                     f"{c['same_acc']} мерили {T} раза — эти измерения связаны",
                     ["Чтобы сравнить дни между собой и найти лучший", "Чтобы учесть, что в группах было разное число животных и гнёзд",
                      "Без случайного эффекта lmer не выдаст p-value"],
                     f"Повторные измерения — частный случай кластеризации: кластер — {c['gru']}. Случайный интерсепт задаёт одинаковую корреляцию "
                     f"между любыми двумя измерениями {c['gen1']} (compound symmetry) = {rua(tau2, 4)} / ({rua(tau2, 4)} + {rua(sig2, 4)}) = {rua(tau2 / (tau2 + sig2), 2)}."))
    qs_opt.append(num(f"Корреляция двух измерений одного и того же {'гнезда' if c['grp'] == 'litter' else 'животного'} (ICC)?", tau2 / (tau2 + sig2),
                      f"{c['grp']} / ({c['grp']} + Residual) = {rua(tau2, 4)} / ({rua(tau2, 4)} + {rua(sig2, 4)}) = <b>{rua(tau2 / (tau2 + sig2), 3)}</b>.",
                      rel=0.03, abs_=0.01))
    pi = F["p"][names.index(f"treatm{L1}:Day{D[-1]}")]
    qs_opt.append(mc(rng, f"Что значит treatm{L1}:Day{D[-1]} = {rua(b[f'treatm{L1}:Day{D[-1]}'], 4)} ({pr(pi)})?",
                     f"Насколько разница {L1} − {L0} в день {D[-1]} больше (меньше), чем в день {D[0]}",
                     [f"Средний {c['y']} группы {L1} в день {D[-1]} (в единицах {c['y']})", f"Прирост группы {L0} от дня {D[0]} до дня {D[-1]} опыта",
                      f"Разница между днями {D[0]} и {D[-1]} у всех животных обеих групп вместе (в среднем)"],
                     f"Коэффициент взаимодействия — «разница разниц». {pr(pi).capitalize()}: "
                     + ("кривые групп расходятся со временем." if pi < 0.05 else "нет доказательств, что разница групп меняется со временем.")))
    mean_b_last = b["(Intercept)"] + bt + b[f"Day{D[-1]}"] + b[f"treatm{L1}:Day{D[-1]}"]
    qs_opt.append(num(f"Средний {c['y']} группы {L1} в день {D[-1]}?", mean_b_last,
                      f"(Intercept) + treatm{L1} + Day{D[-1]} + treatm{L1}:Day{D[-1]} = {rua(b['(Intercept)'], 4)} + ({rua(bt, 4)}) + ({rua(b[f'Day{D[-1]}'], 4)}) + ({rua(b[f'treatm{L1}:Day{D[-1]}'], 4)}) = <b>{rua(mean_b_last, 4)}</b>{unit}.",
                      rel=0.005))
    qs = pick(rng, qs_req, qs_opt)
    intro = (f"Повторные измерения: {c['what']}. {F['ngroups']} {'гнёзд' if c['grp'] == 'litter' else 'свиней'} распределили по вариантам {L0} и {L1} жребием; "
             f"{c['yacc']} измеряли в дни {', '.join(D)}. Day — фактор, treatm * Day = главные эффекты + взаимодействие; "
             f"{c['grp']} — случайный интерсепт.")
    return dict(cat="mixed", sub="lmmr", title=f"lmer: {c['y']} ~ treatm * Day + (1 | {c['grp']})", intro=intro, console="\n".join(lines), qs=qs)


# ---------------------------------------------------------------------------
GB_CTX = [
    dict(y="pn", yru="пневмония на бойне (pn: 1 — есть поражения)", x="ar_1", xlev=("0", "1"), xname="ar_11",
         xru="атрофический ринит (ar_1 = 1, если балл > 1)", grp="farm", gru="ферма", gpl="ферм", ind="свиней", data="pig_adg", within="внутри одной и той же фермы",
         b0=(-1.4, -0.6), b1=(0.2, 0.8), tau=(0.6, 1.1), ng=(12, 22), m=(15, 40), obs=True, what="поражения лёгких"),
    dict(y="diar", yru="диарея у поросёнка (1 — да)", x="vacc", xlev=("no", "yes"), xname="vaccyes",
         xru="вакцинация свиноматки перед опоросом (vacc = yes)", grp="farm", gru="ферма", gpl="ферм", ind="поросят", data="piglets", within="внутри одной и той же фермы",
         b0=(-0.6, 0.2), b1=(-1.1, -0.3), tau=(0.5, 1.0), ng=(12, 25), m=(20, 45), obs=True, what="диарею"),
    dict(y="dead", yru="падёж до отъёма (1 — пал)", x="lowbw", xlev=("0", "1"), xname="lowbw1",
         xru="низкая масса при рождении (lowbw = 1, если < 1 кг)", grp="batch", gru="партия опоросов", gpl="партий", ind="поросят", data="litters", within="внутри одной и той же партии",
         b0=(-2.6, -1.8), b1=(0.8, 1.6), tau=(0.3, 0.7), ng=(12, 20), m=(40, 80), obs=True, what="падёж"),
    dict(y="infection", yru="инфекция (1 — да)", x="treatment", xlev=("Control", "Vaccine"), xname="treatmentVaccine",
         xru="вакцина против контроля", grp="block_id", gru="блок", gpl="блоков", ind="животных", data="infection", within="внутри одного и того же блока",
         b0=(-0.3, 0.5), b1=(-1.4, -0.5), tau=(0.4, 0.9), ng=(8, 14), m=(20, 40), obs=False, what="инфекцию"),
    dict(y="bite", yru="укус хвоста за опыт (1 — да)", x="enrich", xlev=("none", "straw"), xname="enrichstraw",
         xru="солома в загоне (enrich = straw) против пустого загона", grp="pen", gru="загон", gpl="загонов", ind="свиней", data="pens", within="внутри одного и того же загона",
         b0=(-1.6, -0.8), b1=(-1.2, -0.4), tau=(0.5, 1.0), ng=(16, 30), m=(10, 20), obs=False, pen_level=True, what="укусы хвостов"),
]


def gen_glmmb(rng, c):
    for _ in range(25):
        ng = int(rng.integers(*c["ng"]))
        sizes = rng.integers(c["m"][0], c["m"][1] + 1, ng)
        g = np.repeat(np.arange(1, ng + 1), sizes)
        n = len(g)
        tau = rng.uniform(*c["tau"])
        u = rng.normal(0, tau, ng)
        if c.get("pen_level"):
            xg = np.array([i % 2 for i in range(ng)], float)
            rng.shuffle(xg)
            x = xg[g - 1]
        elif c["obs"]:
            # exposure partly clustered by farm (confounding with farm effect)
            px = special.expit(-0.3 + 0.8 * u / max(tau, 1e-6) * rng.uniform(0, 1))
            x = (rng.random(n) < px[g - 1]).astype(float)
        else:
            x = (rng.random(n) < 0.5).astype(float)
        b0, b1 = rng.uniform(*c["b0"]), rng.uniform(*c["b1"])
        if rng.random() < 0.25:
            b1 *= rng.uniform(0.0, 0.3)
        y = (rng.random(n) < special.expit(b0 + b1 * x + u[g - 1])).astype(float)
        X = np.column_stack([np.ones(n), x])
        names = ["(Intercept)", c["xname"]]
        try:
            F = glmm(y, X, g, names, "binomial")
        except ValueError:
            continue
        if F["var"] > 0.05 and 0.05 < y.mean() < 0.9:
            break
    formula = f"{c['y']} ~ {c['x']} + (1 | {c['grp']})"
    lines = [f"> m <- glmer({c['y']} ~ {c['x']} + (1 | {c['grp']}), family = binomial,",
             f"+            control = glmerControl(optimizer = \"bobyqa\"), nAGQ = 10, data = {c['data']})", "> summary(m)"]
    lines += glmer_print(formula, c["data"], F, names, c["grp"], "binomial")
    show_glm = rng.random() < 0.45
    G = F["glm"]
    if show_glm:
        lines += ["", f"> coef(summary(glm({c['y']} ~ {c['x']}, family = binomial, data = {c['data']})))"]
        lines += R.matrix(names, ["Estimate", "Std. Error", "z value", "Pr(>|z|)"], [list(G["b"]), list(G["se"]), list(G["z"]), list(G["p"])], digits=7)
    fx = shown(lines, "Fixed effects:")
    vc = vc_shown(lines)
    var = vc[c["grp"]][0]
    b0 = fnum(fx["(Intercept)"][0])
    b1 = fnum(fx[c["xname"]][0])
    p1 = F["p"][1]
    OR = math.exp(b1)
    icc = var / (var + LATENT)
    qs_req, qs_opt = [], []
    qs_req.append(num(f"Отношение шансов (OR) для {c['xname']}?", OR,
                      f"OR = e<sup>b</sup> = e<sup>{rua(b1, 4)}</sup> = <b>{rua(OR, 3)}</b>. Шансы ({c['what']}) у группы «{c['xlev'][1]}» "
                      f"в {rua(OR, 3)} раза {'выше' if OR > 1 else 'ниже'}, чем у «{c['xlev'][0]}»" + (f" (на {rua(100 * (1 - OR), 3)} % ниже)." if OR < 1 else "."),
                      rel=0.01))
    qs_req.append(mc(rng, f"Как правильно прочитать этот OR = {rua(OR, 3)} из glmer?",
                     f"Сравнение шансов {c['within']} (subject-specific)",
                     ["Сравнение средних вероятностей по всей совокупности ферм (population-averaged)", "Отношение вероятностей (RR) между группами, а не шансов",
                      "Разница вероятностей в процентных пунктах между группами"],
                     "В GLMM OR условный: «при прочих равных», включая тот же случайный эффект группы. Это эффект «для ветврача конкретной фермы». "
                     "Популяционно-усреднённый OR (PA) ближе к 1: приблизительно b<sub>PA</sub> ≈ b / √(1 + 0,346·σ²)."))
    qs_req.append(num(f"ICC на латентной шкале (доля изменчивости из-за {c['grp']}), %?", 100 * icc,
                      f"У логистической модели нет Residual, поэтому берут π²/3 = 3,29 как дисперсию ошибки на шкале логитов: "
                      f"ICC = {rua(var, 4)} / ({rua(var, 4)} + 3,29) = <b>{rua(100 * icc, 3)} %</b>.", rel=0.03, abs_=0.5, unit="%"))
    pref = special.expit(b0)
    qs_opt.append(num(f"Вероятность ({c['what']}) в группе «{c['xlev'][0]}» на «типичной» {'ферме' if c['grp'] == 'farm' else 'группе'} (u = 0), %?",
                      100 * pref,
                      f"p = 1 / (1 + e<sup>−b₀</sup>) = 1 / (1 + e<sup>{rua(-b0, 4)}</sup>) = {rua(pref, 3)} = <b>{rua(100 * pref, 3)} %</b>. "
                      "Это медианная группа (u = 0), а не среднее по всем фермам: из-за нелинейности средняя вероятность другая.", rel=0.02, abs_=0.3, unit="%"))
    qs_opt.append(mc(rng, f"Variance у {c['grp']} = {rua(var, 4)}. Что это?",
                     f"Насколько группы различаются по исходным шансам — на шкале логитов",
                     ["Разброс вероятностей болезни между группами в процентах", "Стандартная ошибка коэффициента, возведённая в квадрат",
                      "Дисперсия остатков внутри каждой группы — то же, что Residual в lmer"],
                     f"σ² = {rua(var, 4)} — дисперсия случайных сдвигов логита. Std.Dev. {rua(math.sqrt(var), 3)}: шансы у «типичной» группы на ±1 SD "
                     f"отличаются в e<sup>{rua(math.sqrt(var), 3)}</sup> = {rua(math.exp(math.sqrt(var)), 3)} раза. 0 — нет кластеризации."))
    pa = b1 / math.sqrt(1 + 0.346 * var)
    qs_opt.append(num(f"Приближённый популяционно-усреднённый OR: e^(b / √(1 + 0,346·σ²))?", math.exp(pa),
                      f"b<sub>PA</sub> = {rua(b1, 4)} / √(1 + 0,346 × {rua(var, 4)}) = {rua(pa, 3)}; OR<sub>PA</sub> = e<sup>{rua(pa, 3)}</sup> = <b>{rua(math.exp(pa), 3)}</b> — "
                      "эффект для всех ферм сразу, он ближе к 1, чем фермо-специфический.", rel=0.02))
    if show_glm:
        pg = float(G["p"][1])
        qs_req.append(mc(rng, f"glm без {c['grp']}: b = {rua(G['b'][1], 4)}, SE {rua(G['se'][1], 3)}; glmer: b = {rua(b1, 4)}, SE {rua(F['se'][1], 3)}. Почему разница?",
                         f"glm считает всех {c['ind']} независимыми, а glmer учитывает {c['grp']}",
                         ["glmer просто иначе округляет коэффициенты и SE", "glm точнее: в нём меньше параметров для оценки",
                          "Разницы по сути нет: это шум оптимизатора bobyqa"],
                         f"Когда {c['ind']} внутри {c['gpl']} похожи, glm занижает SE. Если же экспозиция чаще встречается в «плохих» группах, "
                         f"glm ещё и смещает оценку — группа работает как конфаундер. glmer сравнивает животных внутри групп."))
    qs_opt.append(mc(rng, f"Значим ли {c['x']} на уровне 0,05?", f"{'Да' if p1 < 0.05 else 'Нет'}: {pr(p1)} (Wald z-тест)",
                     [f"{'Нет' if p1 < 0.05 else 'Да'}: {pr(p1)} (Wald z-тест)", "Нельзя сказать без df", "Да, раз OR не равен 1"],
                     f"В glmer p-value — по z (Wald). {pr(p1).capitalize()} → эффект {sig_word(p1)}."))
    qs = pick(rng, qs_req, qs_opt)
    kind = "Наблюдательные данные" if c["obs"] else "Опыт"
    intro = (f"{kind}: {F['n']} {c['ind']} из {F['ngroups']} {c['gpl']}. Исход — {c['yru']}; предиктор — {c['xru']}. "
             f"{c['gru'].capitalize()} — случайный интерсепт: животные одной группы похожи. Модель — логистическая GLMM, glmer с nAGQ = 10.")
    return dict(cat="mixed", sub="glmmb", title=f"glmer binomial: {c['y']} ~ {c['x']} + (1 | {c['grp']})", intro=intro, console="\n".join(lines), qs=qs)


# ---------------------------------------------------------------------------
GP_CTX = [
    dict(y="reactors", yru="число туберкулиновых реакторов в группе", x="age", lev=["1", "2", "3"], levru="1 — моложе 12 мес., 2 — 12–24 мес., 3 — старше 24 мес.", off="par",
         offru="животных под риском (par)", grp="farm_id", gru="ферма", gpl="ферм", data="tb", per=10000, perru="на 10 000 животных",
         b0=(-9.0, -7.5), eff=[(1.5, 3.0), (1.2, 2.8)], tau=(0.8, 1.4), ng=(8, 15), m=(8, 18), offr=(20, 400), what="ТБ"),
    dict(y="coughs", yru="число кашлей в загоне за 15 мин", x="vacc", lev=["no", "yes"], off="pigs",
         offru="свиней в загоне (pigs)", grp="farm", gru="ферма", gpl="ферм", data="cough", per=100, perru="на 100 свиней за 15 мин",
         b0=(-1.6, -0.8), eff=[(-1.0, -0.2)], tau=(0.3, 0.7), ng=(10, 20), m=(6, 12), offr=(15, 35), what="кашель"),
    dict(y="lame", yru="число хромых свиноматок в загоне", x="floor", lev=["slat", "solid"], off="sows",
         offru="свиноматок в загоне (sows)", grp="farm", gru="ферма", gpl="ферм", data="sowpens", per=100, perru="на 100 свиноматок",
         b0=(-2.4, -1.6), eff=[(-0.9, -0.2)], tau=(0.3, 0.6), ng=(10, 20), m=(5, 10), offr=(10, 40), what="хромоту"),
]


def gen_glmmp(rng, c):
    k = len(c["lev"])
    for _ in range(25):
        ng = int(rng.integers(*c["ng"]))
        sizes = rng.integers(c["m"][0], c["m"][1] + 1, ng)
        g = np.repeat(np.arange(1, ng + 1), sizes)
        n = len(g)
        lv = np.array([c["lev"][i % k] for i in range(n)])
        rng.shuffle(lv)
        E = rng.integers(c["offr"][0], c["offr"][1] + 1, n).astype(float)
        effs = [0.0] + [rng.uniform(*e) for e in c["eff"]]
        if rng.random() < 0.25:
            effs = [e * rng.uniform(0, 0.3) for e in effs]
        tau = rng.uniform(*c["tau"])
        b0 = rng.uniform(*c["b0"])
        eta = b0 + np.array([effs[c["lev"].index(l)] for l in lv]) + rng.normal(0, tau, ng)[g - 1] + np.log(E)
        y = rng.poisson(np.exp(eta)).astype(float)
        X = np.column_stack([np.ones(n)] + [(lv == l).astype(float) for l in c["lev"][1:]])
        names = ["(Intercept)"] + [f"{c['x']}{l}" for l in c["lev"][1:]]
        if any(y[lv == l].sum() < 3 for l in c["lev"]):
            continue
        try:
            F = glmm(y, X, g, names, "poisson", offset=np.log(E))
        except ValueError:
            continue
        if F["var"] > 0.05 and y.sum() > 15:
            break
    formula = f"{c['y']} ~ {c['x']} + offset(log({c['off']})) + (1 | {c['grp']})"
    lines = [f"> m <- glmer({c['y']} ~ {c['x']} + offset(log({c['off']})) + (1 | {c['grp']}),",
             f"+            family = poisson, control = glmerControl(optimizer = \"bobyqa\"),",
             f"+            nAGQ = 10, data = {c['data']})", "> summary(m)"]
    lines += glmer_print(formula, c["data"], F, names, c["grp"], "poisson")
    # dispersion_glmer (blmeco): sqrt(sum(c(deviance residuals, spherical u)^2) / n)
    mu = F["mu"]
    yv = y
    with np.errstate(divide="ignore", invalid="ignore"):
        dr = np.sign(yv - mu) * np.sqrt(2 * (np.where(yv > 0, yv * np.log(yv / mu), 0) - (yv - mu)))
    sph = F["u"] / F["theta"] if F["theta"] > 0 else F["u"]
    disp = math.sqrt(float(np.sum(np.r_[dr, sph] ** 2)) / len(yv))
    show_disp = rng.random() < 0.5
    if show_disp:
        lines += ["", "> blmeco::dispersion_glmer(m)", "[1] " + R.r7(disp)]
    fx = shown(lines, "Fixed effects:")
    vc = vc_shown(lines)
    var = vc[c["grp"]][0]
    b0s = fnum(fx["(Intercept)"][0])
    j = int(rng.integers(1, k))
    bj = fnum(fx[names[j]][0])
    IRR = math.exp(bj)
    pj = F["p"][j]
    qs_req, qs_opt = [], []
    qs_req.append(num(f"Отношение частот (IRR) для {names[j]}?", IRR,
                      f"IRR = e<sup>{rua(bj, 4)}</sup> = <b>{rua(IRR, 3)}</b>: частота ({c['what']}) в группе «{c['lev'][j]}» в {rua(IRR, 3)} раза "
                      f"{'выше' if IRR > 1 else 'ниже'}, чем в «{c['lev'][0]}», на той же {'ферме' if 'farm' in c['grp'] else 'группе'}.", rel=0.01))
    qs_req.append(mc(rng, f"Зачем в формуле offset(log({c['off']}))?",
                     "Чтобы моделировать частоту на одно животное",
                     ["Чтобы убрать из данных строки с нулями", f"Чтобы оценить отдельный коэффициент при {c['off']}",
                      "Чтобы исправить сверхдисперсию данных"],
                     f"log(μ) = log({c['off']}) + b₀ + …, т.е. log(μ / {c['off']}) = b₀ + … Коэффициент при log({c['off']}) зафиксирован на 1, "
                     "поэтому e<sup>b</sup> — отношения частот, а не числа случаев."))
    rate = c["per"] * math.exp(b0s)
    qs_opt.append(num(f"Частота в группе «{c['lev'][0]}» {c['perru']} на «типичной» ферме (u = 0)?", rate,
                      f"e<sup>b₀</sup> = e<sup>{rua(b0s, 4)}</sup> = {rua(math.exp(b0s), 3)} на одно животное. × {c['per']} = <b>{rua(rate, 3)}</b> {c['perru']}.",
                      rel=0.02))
    qs_opt.append(mc(rng, f"Что значит {names[j]} = {rua(bj, 4)} ({pr(pj)})?",
                     f"Логарифм частоты в группе «{c['lev'][j]}» на {rua(abs(bj), 3)} {'выше' if bj > 0 else 'ниже'}, чем в «{c['lev'][0]}»",
                     [f"В группе «{c['lev'][j]}» на {rua(abs(bj), 3)} случая {'больше' if bj > 0 else 'меньше'} в каждой строке",
                      f"Вероятность случая в группе «{c['lev'][j]}» на {rua(abs(bj), 3)} {'выше' if bj > 0 else 'ниже'}, чем в группе «{c['lev'][0]}»",
                      f"Шансы в «{c['lev'][j]}» в {rua(IRR, 3)} раза {'выше' if bj > 0 else 'ниже'}"],
                     f"В пуассоновской модели коэффициенты — на шкале log(частоты). Удобнее e<sup>b</sup> = IRR = {rua(IRR, 3)}. "
                     f"{pr(pj).capitalize()}: эффект {sig_word(pj)}."))
    if show_disp:
        ok = 0.75 <= disp <= 1.4
        qs_req.append(mc(rng, f"dispersion_glmer(m) = {R.r7(disp)}. Есть ли проблема сверхдисперсии?",
                         "Нет: значение в пределах 0,75–1,4" if ok else ("Да: значение больше 1,4 — сверхдисперсия" if disp > 1.4 else "Значение меньше 0,75 — недодисперсия, обычно не страшно"),
                         ["Да: любое значение больше 1 — сверхдисперсия" if ok else "Нет: значение в пределах 0,75–1,4",
                          "Нужно сравнить с 0,05", "Нельзя сказать без p-value"],
                         "Правило из курса (Д. Бейтс): если dispersion_glmer между 0,75 и 1,4, сверхдисперсии, скорее всего, нет. "
                         "Если больше — добавить случайный эффект / предикторы или перейти к отрицательной биномиальной (glmmTMB, family = nbinom2)."))
    qs_opt.append(mc(rng, f"Variance у {c['grp']} = {rua(var, 4)}. Как это понимать?",
                     f"Фермы различаются по исходной частоте: SD {rua(math.sqrt(var), 3)} на шкале log",
                     ["Это сверхдисперсия внутри каждой фермы", "Это стандартная ошибка оценки IRR",
                      "Это доля ферм, где за всё время наблюдения не было ни одного случая"],
                     f"Случайный интерсепт: log-частота каждой фермы сдвинута на u ~ N(0, {rua(var, 4)}). "
                     f"Ферма на +1 SD имеет частоту в e<sup>{rua(math.sqrt(var), 3)}</sup> = {rua(math.exp(math.sqrt(var)), 3)} раза выше типичной."))
    qs = pick(rng, qs_req, qs_opt)
    intro = (f"Данные: {nn(F['n'], ('строка', 'строки', 'строк'))} из {F['ngroups']} {c['gpl']}. Исход — {c['yru']}; в строках разное число животных: {c['offru']} — offset. "
             f"Предиктор — {c['x']} ({c.get('levru', ', '.join(c['lev']))}). {c['gru'].capitalize()} — случайный интерсепт. Модель — пуассоновская GLMM (glmer, nAGQ = 10).")
    return dict(cat="mixed", sub="glmmp", title=f"glmer poisson: {c['y']} ~ {c['x']} + offset + (1 | {c['grp']})", intro=intro,
                console="\n".join(lines), qs=qs)


PLAN4 = [
    ("lmmh", gen_lmmh, H_CTX, 70),
    ("lmme", gen_lmme, E_CTX, 40),
    ("rcbd", gen_rcbd, B_RCBD, 30),
    ("lmmr", gen_lmmr, RM_CTX, 40),
    ("glmmb", gen_glmmb, GB_CTX, 60),
    ("glmmp", gen_glmmp, GP_CTX, 40),
]
