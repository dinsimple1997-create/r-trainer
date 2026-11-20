"""Workflows, part C: logistic, Poisson with offset, lmer (pens, repeated measures), glmer, power."""
import math
import numpy as np
from scipy import stats
import rfmt as R
import htest as H
from models import ols, logistic, poisson
from mixed import lmm, glmm
from gen import ru, rua, mc, num, glm_summary, coef_block
from gen2 import pois_summary, irr_table, disp_lines, deparse_call, dev_lines
from gen4 import lmer_print, glmer_print
from wf_lib import (str_df, df_print, agg_print, named, code_pick, multi, step, p_lm, p_xy, p_int, p_qq, pr, table2, joint_cells,
                    aov_summary, anova2, summary_vec, rd)
from wf_a import shuffle_rows, lm_out
import diag_bank as DB


def anova_dev(f1, f2, rdf, dev):
    df = rdf[0] - rdf[1]
    dd = dev[0] - dev[1]
    p = float(stats.chi2.sf(dd, df))
    devf = R.fmt_joint(dev, 5)
    cols = [("Resid. Df", [str(rdf[0]), str(rdf[1])]), ("Resid. Dev", devf), ("Df", ["", str(df)]), ("Deviance", ["", R.fmt1(dd, 5)]),
            ("Pr(>Chi)", ["", R.pval_col([p])[0]])]
    lines = ["Analysis of Deviance Table", "", f"Model 1: {f1}", f"Model 2: {f2}"] + R.table(["1", "2"], cols, ["", R.stars(p)])
    if p < 0.1:
        lines += ["---", R.SIGNIF]
    return lines, dict(dd=dd, p=p, df=df)


# =====================================================================
# 10. Logistic regression: pre-weaning mortality
# =====================================================================
LOGIT_VAR = [dict(b_par=0.65, kind="par"), dict(b_par=0.0, kind="nopar")]


def wf_logit(rng, v):
    n = 360
    for _ in range(3000):
        bw = np.round(np.clip(rng.normal(1.38, 0.28, n), 0.6, 2.2), 2)
        gilt = (rng.random(n) < 0.35).astype(float)
        eta = 2.6 - 3.3 * bw + v["b_par"] * gilt
        died = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
        X = np.c_[np.ones(n), bw, (1 - gilt)]  # parity: gilt (base) / sow
        L = logistic(X, died)
        X0 = np.c_[np.ones(n), bw]
        L0 = logistic(X0, died)
        lr = L0["res_dev"] - L["res_dev"]
        p_lr = stats.chi2.sf(lr, 1)
        if L["p"][1] < 1e-4 and ((v["kind"] == "par" and p_lr < 0.01) or (v["kind"] == "nopar" and p_lr > 0.3)) and 0.10 < died.mean() < 0.2:
            break
    else:
        raise RuntimeError("logit")
    parity = np.where(gilt == 1, "gilt", "sow")
    S = []
    S.append(step("Загрузить", task="Таблица <code>pw</code>: по одному случайному поросёнку из 360 гнёзд; масса при рождении (кг), чей помёт (свинка-первоопоросок <code>gilt</code> или свиноматка <code>sow</code>) и пал ли до отъёма (1/0).",
                  code=['> pw <- read.csv("preweaning.csv")', "> str(pw)"],
                  out=str_df(n, [("piglet", "int", range(1, n + 1)), ("parity", "chr", parity), ("birth_wt", "num", bw), ("died", "int", died.astype(int))]),
                  qs=[num("Какая доля поросят пала до отъёма, %? (среднее столбца 0/1)", 100 * died.mean(),
                          f"Среднее 0/1 — доля единиц: mean(pw$died) = {R.r7(died.mean())} → <b>{ru(100 * died.mean(), 1)}</b>%.", rel=0.01)]))
    cuts = [0, 1.0, 1.2, 1.4, 1.6, 3]
    labs = ["(0,1]", "(1,1.2]", "(1.2,1.4]", "(1.4,1.6]", "(1.6,3]"]
    cls = np.digitize(bw, cuts[1:-1], right=True)
    obs = [died[cls == k].mean() for k in range(5)]
    w0 = max(len(l) for l in labs + ["wcl"])
    agg = ["      ".rjust(1) + " " + "wcl".rjust(w0) + " " + "died".rjust(9)]
    obs_s = [s.strip() for s in R.format_real(obs, 7)]
    agg = [" " + " " + "wcl".rjust(w0) + " " + "died".rjust(max(4, max(len(s) for s in obs_s)))]
    for k in range(5):
        agg.append(str(k + 1) + " " + labs[k].rjust(w0) + " " + obs_s[k].rjust(max(4, max(len(s) for s in obs_s))))
    S.append(step("Смертность по классам массы", task="Прежде чем строить модель, посмотри на сырые доли павших по классам массы.",
                  code=["> pw$wcl <- cut(pw$birth_wt, breaks = c(0, 1, 1.2, 1.4, 1.6, 3))", "> aggregate(died ~ wcl, data = pw, FUN = mean)"], out=agg,
                  qs=[mc(rng, "Что видно?", "Чем легче поросёнок, тем выше доля павших; связь нелинейная по доле — похоже на логистическую кривую",
                         ["Смертность одинакова во всех классах", "Тяжёлые поросята гибнут чаще", "Связь прямая: каждые 200 г — минус 5% смертности ровно"],
                         "Доли 0/1-исхода по классам X — простой способ увидеть форму связи. Логистическая модель описывает плавное падение риска.")]))
    names = ["(Intercept)", "birth_wt", "paritysow"]
    gl, gd = glm_summary("died ~ birth_wt + parity", "pw", L, names)
    S.append(step("Логистическая регрессия",
                  pick=code_pick(rng, "Какой код?", "m <- glm(died ~ birth_wt + parity, family = binomial, data = pw)",
                                 ["m <- lm(died ~ birth_wt + parity, data = pw)", "m <- glm(died ~ birth_wt + parity, family = poisson, data = pw)",
                                  "m <- glm(died ~ birth_wt + parity, data = pw)"],
                                 "0/1-исход → <code>family = binomial</code>. Без family glm делает обычную линейную регрессию; poisson — для счёта."),
                  code="> m <- glm(died ~ birth_wt + parity, family = binomial, data = pw)\n> summary(m)", out=gl[2:],
                  qs=[mc(rng, f"birth_wt = {gd['est'][1]}. Что это значит?", "На каждый +1 кг массы log-шансы гибели меньше на " + rua(-L["b"][1], 3),
                         ["Каждый +1 кг снижает вероятность гибели на " + rua(-L["b"][1], 3), "Каждый +1 кг снижает смертность на " + rua(-L["b"][1], 3) + "%",
                          "Связи нет: коэффициент отрицательный"],
                         "Коэффициенты логистической регрессии — в log-шансах. Чтобы читать в разах, берут exp().")]))
    tq = 1.959964
    ors = [list(np.exp(L["b"])), list(np.exp(L["b"] - tq * L["se"])), list(np.exp(L["b"] + tq * L["se"]))]
    or100 = math.exp(L["b"][1] / 10)
    S.append(step("Отношения шансов", code="> exp(cbind(OR = coef(m), confint.default(m)))",
                  out=R.matrix(names, ["OR", "2.5 %", "97.5 %"], ors),
                  qs=[num("OR гибели на каждые +100 г массы при рождении?", or100,
                          f"+100 г = 0,1 кг: OR = exp({R.r7(L['b'][1])} × 0,1) = <b>{rua(or100, 3)}</b>. Каждые 100 г снижают шансы гибели на {ru(100 * (1 - or100), 0)}%.", rel=0.02),
                      mc(rng, "Как прочитать OR для paritysow?",
                         (f"Шансы гибели у поросят свиноматок в {rua(1 / ors[0][2], 2)} раза ниже, чем у поросят свинок, при той же массе" if v["kind"] == "par"
                          else f"Разница не доказана: ДИ OR ({rua(ors[1][2], 2)}; {rua(ors[2][2], 2)}) включает 1"),
                         ["Поросята свиноматок тяжелее в " + rua(ors[0][2], 2) + " раза", "Риск у свиноматок выше на " + ru(100 * (ors[0][2] - 1), 0) + "%",
                          "OR нельзя читать без p-value"],
                         "Базовый уровень — gilt (по алфавиту). OR &lt; 1 — шансы ниже, чем у базового. «При той же массе» — потому что масса в модели.")]))
    nd = [(0.9, "sow"), (1.5, "sow")]
    pp = [1 / (1 + math.exp(-(L["b"][0] + L["b"][1] * x + L["b"][2]))) for x, _ in nd]
    S.append(step("Прогноз риска", code='> predict(m, data.frame(birth_wt = c(0.9, 1.5), parity = "sow"), type = "response")',
                  out=named(["1", "2"], pp),
                  qs=[num("Риск гибели поросёнка 0,9 кг от свиноматки, %?", 100 * pp[0], f"type = \"response\" даёт вероятность: {R.r7(pp[0])} → <b>{ru(100 * pp[0], 1)}</b>%.", rel=0.02),
                      mc(rng, "Что будет без <code>type = \"response\"</code>?", "R вернёт log-шансы, а не вероятность",
                         ["То же самое", "Ошибку", "Шансы (odds)"], "По умолчанию predict для glm даёт линейный предиктор (logit). Для вероятности — type = \"response\".")]))
    al, ad = anova_dev("died ~ birth_wt", "died ~ birth_wt + parity", [L0["df_res"], L["df_res"]], [L0["res_dev"], L["res_dev"]])
    S.append(step("Тест отношения правдоподобия для parity", code=["> m0 <- glm(died ~ birth_wt, family = binomial, data = pw)", '> anova(m0, m, test = "Chisq")'], out=al,
                  qs=[num("Разница девиансов (Deviance)?", ad["dd"], f"{R.r7(L0['res_dev'])} − {R.r7(L['res_dev'])} = <b>{rua(ad['dd'], 4)}</b> на 1 df.", rel=0.01),
                      mc(rng, "Зачем этот тест, если в summary уже есть p для paritysow?", "Это LRT — обычно точнее теста Вальда; для фактора с 3+ уровнями — единственный способ проверить его целиком",
                         ["Он проверяет массу", "summary всегда неверен", "Он нужен только для lm"],
                         f"Вальд (summary): p = {pr(L['p'][2])}; LRT: p = {pr(ad['p'])}. Обычно близки.")]))
    fitted = L["mu"]
    pred_cls = [fitted[cls == k].mean() for k in range(5)]
    cal = [" " + " wcl".rjust(w0 + 1) + "       died          p"]
    o_s = [s.strip() for s in R.format_real(obs, 7)]
    p_s = [s.strip() for s in R.format_real(pred_cls, 7)]
    wo, wp = max(4, max(len(s) for s in o_s)), max(1, max(len(s) for s in p_s))
    cal = ["  " + "wcl".rjust(w0) + " " + "died".rjust(wo) + " " + "p".rjust(wp)]
    for k in range(5):
        cal.append(str(k + 1) + " " + labs[k].rjust(w0) + " " + o_s[k].rjust(wo) + " " + p_s[k].rjust(wp))
    gap = max(abs(a - b) for a, b in zip(obs, pred_cls))
    S.append(step("Проверить, что модель описывает данные", task="У логистической регрессии не смотрят на нормальность остатков. Проверяют, совпадают ли наблюдаемые и предсказанные доли по классам.",
                  code=["> pw$p <- fitted(m)", "> aggregate(cbind(died, p) ~ wcl, data = pw, FUN = mean)"], out=cal,
                  qs=[mc(rng, "Модель хорошо описывает данные?", "Да: наблюдаемые доли близки к средним предсказанным в каждом классе",
                         ["Нет: остатки 0/1 не нормальные", "Нет: доли не совпадают до третьего знака", "Проверить нельзя: нужен R²"],
                         f"Самое большое расхождение — {ru(100 * gap, 1)} п.п. при десятках поросят в классе — в пределах случайности. "
                         "Для формального теста есть Hosmer–Lemeshow, но таблица нагляднее.")]))
    rep = (f"Каждые +100 г массы при рождении снижали шансы гибели до отъёма в {rua(1 / or100, 2)} раза (OR = {rua(or100, 2)}; 95% ДИ {rua(math.exp((L['b'][1] - tq * L['se'][1]) / 10), 2)}–{rua(math.exp((L['b'][1] + tq * L['se'][1]) / 10), 2)})")
    S.append(step("Записать результат", qs=[mc(rng, "Какая фраза верна?", rep,
                                                [f"Каждые 100 г снижают смертность на {ru(100 * (1 - or100), 0)} процентных пунктов",
                                                 "Масса при рождении — причина гибели поросят", "OR меньше 1, значит связь не значима"],
                                                "OR — про шансы, а не про процентные пункты риска. Это наблюдение: масса связана с гибелью, но вес может отражать другие вещи (жизнеспособность, место в гнезде).")]))
    items = [("По одному поросёнку из гнезда — строки независимы", True), ("Если брать всех поросят гнезда, нужна glmer с (1 | litter)", True),
             ("Проверяли нормальность остатков", False), ("Сравнили наблюдаемые и предсказанные доли", True)]
    S.append(step("Проверка", qs=[multi(rng, "Отметь верное.", items, "Поросята одного гнезда делят мать, молоко и микроклимат — они не независимы.")]))
    return dict(g="glm", key="logit", title="Логистическая: гибель до отъёма", vt=("помёт важен" if v["kind"] == "par" else "помёт не важен"),
                goal="<p><b>Вопрос.</b> Как масса при рождении и вид матери (свинка или свиноматка) связаны с гибелью поросёнка до отъёма? Наблюдательные данные фермы.</p>", steps=S)


# =====================================================================
# 11. Poisson with offset (+ overdispersion)
# =====================================================================
POIS_VAR = [dict(theta=None, kind="ok"), dict(theta=2.5, kind="over")]


def quasi_summary(call_args, P, names, phi):
    se = P["se"] * math.sqrt(phi)
    t = P["b"] / se
    pv = 2 * stats.t.sf(np.abs(t), P["df_res"])
    lines = ["", "Call:"] + deparse_call("glm", call_args) + ["", "Coefficients:"]
    cl, est, sse, tv, ps = coef_block(names, P["b"], se, t, pv, statname="t")
    lines += cl + ["---", R.SIGNIF, "", f"(Dispersion parameter for quasipoisson family taken to be {R.fmt1(phi, 7)})", ""]
    dl = dev_lines(P)
    dl[2] = "AIC: NA"
    lines += dl
    return lines, dict(se=se, t=t, p=pv)


def wf_pois(rng, v):
    npen = 30
    for _ in range(3000):
        trt = np.repeat(["ctrl", "probio"], npen // 2)
        pd = np.round(rng.uniform(560, 760, npen)).astype(int)
        rate = np.where(trt == "probio", 0.0085, 0.0125)
        mu = rate * pd
        if v["theta"]:
            lam = rng.gamma(v["theta"], mu / v["theta"])
        else:
            lam = mu
        cases = rng.poisson(lam)
        X = np.c_[np.ones(npen), (trt == "probio").astype(float)]
        P = poisson(X, cases.astype(float), np.log(pd))
        disp = P["res_dev"] / P["df_res"]
        phi = P["pearson"] / P["df_res"]
        if v["kind"] == "ok" and P["p"][1] < 0.005 and disp < 1.25:
            break
        if v["kind"] == "over" and P["p"][1] < 0.01 and disp > 2.3 and 2 * stats.t.sf(abs(P["b"][1] / (P["se"][1] * math.sqrt(phi))), P["df_res"]) > 0.05:
            break
    else:
        raise RuntimeError("pois")
    o = rng.permutation(npen)
    trt, pd, cases = trt[o], pd[o], cases[o]
    X = np.c_[np.ones(npen), (trt == "probio").astype(float)]
    P = poisson(X, cases.astype(float), np.log(pd))
    S = []
    S.append(step("Загрузить", task="Таблица <code>pens</code>: 30 загонов доращивания, пробиотик или контроль (жребий на загон), число случаев диареи за 4 недели и число свиноднeй (свиньи × дни в загоне).",
                  code=['> pens <- read.csv("diarrhea.csv")', "> str(pens)"],
                  out=str_df(npen, [("pen", "int", range(1, npen + 1)), ("trt", "chr", trt), ("cases", "int", cases), ("pigdays", "int", pd)]),
                  qs=[mc(rng, "Почему загоны сравнивают не по числу случаев, а по частоте на свинодень?", "Загоны «под риском» разное время: где больше свиней и дней, там больше и случаев",
                         ["Потому что случаи — целые числа", "Так требует R", "Число случаев нельзя сравнивать никогда"],
                         "Свинодни — «время под риском». Частота = случаи / свинодни. В модели это учитывает offset(log(pigdays)).")]))
    sums = {t: (cases[trt == t].sum(), pd[trt == t].sum()) for t in ("ctrl", "probio")}
    rates = [1000 * sums[t][0] / sums[t][1] for t in ("ctrl", "probio")]
    S.append(step("Частота по группам", code=["> r <- aggregate(cbind(cases, pigdays) ~ trt, data = pens, FUN = sum)", "> r$per1000 <- 1000 * r$cases / r$pigdays", "> r"],
                  out=df_print([("trt", "chr", ["ctrl", "probio"]), ("cases", "int", [sums["ctrl"][0], sums["probio"][0]]),
                                ("pigdays", "int", [sums["ctrl"][1], sums["probio"][1]]), ("per1000", "num", rates)]),
                  qs=[num("Отношение частот probio / ctrl?", rates[1] / rates[0], f"{R.r7(rates[1])} / {R.r7(rates[0])} = <b>{rua(rates[1] / rates[0], 3)}</b>. Это сырой IRR — модель даст то же число.", rel=0.01)]))
    names = ["(Intercept)", "trtprobio"]
    call_args = ["formula = cases ~ trt + offset(log(pigdays))", "family = poisson", "data = pens"]
    pl, pd_ = pois_summary("glm(cases ~ trt + offset(log(pigdays)), family = poisson, data = pens)", call_args, P, names)
    S.append(step("Пуассоновская модель с offset",
                  pick=code_pick(rng, "Какой код?", "m <- glm(cases ~ trt + offset(log(pigdays)), family = poisson, data = pens)",
                                 ["m <- glm(cases ~ trt, family = poisson, data = pens)", "m <- glm(cases ~ trt + offset(pigdays), family = poisson, data = pens)",
                                  "m <- lm(cases / pigdays ~ trt, data = pens)"],
                                 "Модель на log-шкале: log(μ) = log(свинодни) + β₀ + β₁·trt. Поэтому offset — <b>логарифм</b> времени под риском. Без offset сравниваются сырые числа случаев."),
                  code="> m <- glm(cases ~ trt + offset(log(pigdays)), family = poisson, data = pens)\n> summary(m)", out=pl[2:],
                  qs=[mc(rng, f"exp(trtprobio) = {rua(math.exp(P['b'][1]), 3)}. Что это?", f"Частота диареи на пробиотике в {rua(math.exp(P['b'][1]), 3)} от контрольной (IRR)",
                         ["Число случаев в загоне на пробиотике", f"Риск снизился на {rua(math.exp(P['b'][1]), 3)}%", "Доля больных свиней"],
                         f"exp(коэффициента) — отношение частот (incidence rate ratio). Снижение на {ru(100 * (1 - math.exp(P['b'][1])), 0)}%.")]))
    dl, dv = disp_lines(P, pearson=True)
    phi = P["pearson"] / P["df_res"]
    S.append(step("Проверить сверхдисперсию", task="Пуассон предполагает, что дисперсия равна среднему. Проверь отношение девианса к df.",
                  code=dl[0], out=dl[1:],
                  qs=[mc(rng, "Что делать?",
                         ("Всё в порядке: отношение около 1, пуассоновская модель подходит" if v["kind"] == "ok" else
                          "Сверхдисперсия: отношение больше 2, SE занижены — нужен quasipoisson или отрицательная биномиальная"),
                         (["Сверхдисперсия: нужен quasipoisson", "Отношение должно быть ровно 0", "Перейти на lm"] if v["kind"] == "ok"
                          else ["Ничего: 2,5 — близко к 1", "Удалить загоны с большим числом случаев", "Добавить offset ещё раз"]),
                         "Загоны различаются сильнее, чем допускает Пуассон (кто-то заболел вспышкой, кто-то нет). Тогда SE и p по Пуассону слишком оптимистичны.")]))
    est = P["b"]
    se_use = P["se"]
    if v["kind"] == "over":
        ql, qd = quasi_summary(["formula = cases ~ trt + offset(log(pigdays))", "family = quasipoisson", "data = pens"], P, names, phi)
        S.append(step("Квази-Пуассон", code="> mq <- glm(cases ~ trt + offset(log(pigdays)), family = quasipoisson, data = pens)\n> summary(mq)", out=ql,
                      qs=[num("Во сколько раз выросла SE коэффициента trtprobio?", math.sqrt(phi), f"SE умножается на √φ = √{R.fmt1(phi, 4)} = <b>{rua(math.sqrt(phi), 3)}</b>. Оценка та же.", rel=0.02),
                          mc(rng, f"Пуассон: p = {pr(P['p'][1])}, квази: p = {pr(qd['p'][1])}. Чему верить?", "Квази-Пуассону: он учитывает реальный разброс загонов",
                             ["Пуассону: у него p меньше", "Обоим: выводы одинаковые", "Ни одному: нужно больше загонов"],
                             "Сверхдисперсия не меняет оценку IRR, но расширяет интервал. Вывод «пробиотик точно работает» по Пуассону был слишком смелым.")]))
        se_use = qd["se"]
    it = irr_table(names, est, se_use)
    it[0] = "> exp(cbind(IRR = coef(" + ("mq" if v["kind"] == "over" else "m") + "), confint.default(" + ("mq" if v["kind"] == "over" else "m") + ")))"
    irr = math.exp(est[1])
    lo, hi = math.exp(est[1] - 1.959964 * se_use[1]), math.exp(est[1] + 1.959964 * se_use[1])
    S.append(step("IRR с интервалом", code=it[0], out=it[1:],
                  qs=[mc(rng, "Как прочитать интервал для trtprobio?", f"Частота на пробиотике — от {rua(lo, 2)} до {rua(hi, 2)} от контрольной",
                         ["От " + rua(lo, 2) + " до " + rua(hi, 2) + " случаев на загон", "95% загонов в этом интервале", "Интервал для числа свинодней"],
                         "Интервал в разах. Включает 1 — разница не доказана.")]))
    pp = qd["p"][1] if v["kind"] == "over" else P["p"][1]
    rep = f"Частота диареи на пробиотике составила {rua(irr, 2)} от контрольной (IRR; 95% ДИ {rua(lo, 2)}–{rua(hi, 2)}; " + ("квази-Пуассон" if v["kind"] == "over" else "Пуассон") + f" с offset свинодней, p = {pr(pp)})"
    S.append(step("Записать результат", qs=[mc(rng, "Какая фраза верна?", rep,
                                                [f"У пробиотика на {ru(100 * (1 - irr), 0)} случаев меньше в каждом загоне", "Пробиотик снижает число больных свиней в " + rua(1 / irr, 2) + " раза у всех ферм",
                                                 "Пуассоновская модель показала, что случаи распределены нормально"],
                                                "IRR — отношение частот на свинодень, а не разница числа случаев.")]))
    items = [("Единица — загон, жребий на загон", True), ("Offset — логарифм свинодней", True), ("Проверили сверхдисперсию", True),
             ("Сверхдисперсия меняет оценку IRR", False)]
    S.append(step("Проверка", qs=[multi(rng, "Отметь верное.", items, "Сверхдисперсия меняет SE и p, но не точечную оценку.")]))
    return dict(g="glm", key="pois", title="Пуассон с offset: диарея в загонах", vt=("Пуассон подходит" if v["kind"] == "ok" else "сверхдисперсия"),
                goal="<p><b>Вопрос.</b> Снижает ли пробиотик частоту диареи на доращивании? 30 загонов, пробиотик назначали загону жребием.</p>", steps=S)


# =====================================================================
# 12. LMM: pens as experimental units (pseudo-replication)
# =====================================================================
LMM_VAR = [dict(eff=22, kind="pseudo"), dict(eff=70, kind="real")]


def wf_lmm(rng, v):
    npen, m = 12, 12
    for _ in range(300):
        pen = np.repeat(np.arange(1, npen + 1), m)
        diet = np.where(pen <= npen // 2, "ctrl", "new")
        rng.shuffle(diet.reshape(npen, m)[:, :1]) if False else None
        pdiet = rng.permutation(np.repeat(["ctrl", "new"], npen // 2))
        diet = pdiet[pen - 1]
        u = rng.normal(0, 38, npen)
        adg = np.round(800 + v["eff"] * (diet == "new") + u[pen - 1] + rng.normal(0, 60, npen * m))
        naive = H.t_two(adg[diet == "ctrl"], adg[diet == "new"], True)
        X = np.c_[np.ones(npen * m), (diet == "new").astype(float)]
        F = lmm(adg, X, pen, ["(Intercept)", "dietnew"])
        if v["kind"] == "pseudo" and naive["p"] < 0.02 and F["p"][1] > 0.12 and F["tau"] > 25:
            break
        if v["kind"] == "real" and naive["p"] < 0.001 and F["p"][1] < 0.02:
            break
    else:
        raise RuntimeError("lmm")
    N = npen * m
    S = []
    S.append(step("Загрузить", task="Таблица <code>d</code>: 144 свиньи в 12 загонах по 12; рацион назначали загону жребием.",
                  code=['> d <- read.csv("pens_trial.csv")', "> str(d)"],
                  out=str_df(N, [("pig", "int", range(1, N + 1)), ("pen", "int", pen), ("diet", "chr", diet), ("adg", "num", adg)]),
                  qs=[mc(rng, "Что здесь единица эксперимента?", "Загон: рацион назначали загону", ["Свинья: привес меряли у свиньи", "Рацион", "Ферма"],
                         "Единица эксперимента — то, чему назначают вариант жребием. Свинья здесь — единица наблюдения.")]))
    S.append(step("Сколько независимых единиц", code='> with(unique(d[c("pen", "diet")]), table(diet))',
                  out=["diet", "ctrl  new ", "   6    6 "],
                  qs=[num("Сколько единиц эксперимента на рацион?", 6, "6 загонов на рацион — вот настоящее n, а не 72 свиньи.", rel=0)]))
    nl = ["> t.test(adg ~ diet, data = d, var.equal = TRUE)"] + H.htest("Two Sample t-test", "adg by diet", [("t", naive["t"]), ("df", naive["df"])], naive["p"],
                                                                       "true difference in means between group ctrl and group new is not equal to 0", naive["ci"], 95,
                                                                       [("mean in group ctrl", naive["m1"]), ("mean in group new", naive["m2"])])
    S.append(step("Наивный анализ (ошибка)", task="Так часто делают — посмотри, что получится.", code=nl[0], out=nl[1:],
                  qs=[mc(rng, f"df = {N - 2}. Что не так?", "Свиньи одного загона не независимы: df и SE посчитаны так, будто 144 независимых опыта",
                         ["Ничего, df правильные", "Надо было Welch", "df должны быть 143"],
                         "Псевдоповторность: загон — единица, свиньи внутри похожи (общий микроклимат, кормушка). SE занижена, p слишком маленькое.")]))
    ll = lmer_print("adg ~ diet + (1 | pen)", "d", F, ["(Intercept)", "dietnew"], "pen")
    S.append(step("Смешанная модель",
                  pick=code_pick(rng, "Какой код учитывает загон?", "m <- lmer(adg ~ diet + (1 | pen), data = d)",
                                 ["m <- lmer(adg ~ diet + (1 | pig), data = d)", "m <- lm(adg ~ diet + factor(pen), data = d)", "m <- lmer(adg ~ (1 | diet) + pen, data = d)"],
                                 "Случайный эффект загона (1 | pen) — у каждого загона своё смещение. (1 | pig) бессмысленно: одна строка на свинью. "
                                 "Загон фиксированным фактором нельзя — он полностью совпадает с рационом."),
                  code="> library(lmerTest)\n> m <- lmer(adg ~ diet + (1 | pen), data = d)\n> summary(m)", out=ll,
                  qs=[mc(rng, f"df у dietnew ≈ {rua(F['df'][1], 3)}. Откуда такое число?", "Почти загоны − 2 = 10: модель поняла, что независимых единиц 12",
                         ["Это число свиней минус 2", "Ошибка Satterthwaite", "Число свиней в загоне"],
                         "Satterthwaite «видит» структуру: рацион меняется только между загонами, поэтому информация о нём — от 12 загонов."),
                      num("ICC: доля дисперсии между загонами = τ² / (τ² + σ²)?", F["tau"] ** 2 / (F["tau"] ** 2 + F["sigma"] ** 2),
                          f"{rua(F['tau'] ** 2, 4)} / ({rua(F['tau'] ** 2, 4)} + {rua(F['sigma'] ** 2, 4)}) = <b>{rua(F['tau'] ** 2 / (F['tau'] ** 2 + F['sigma'] ** 2), 3)}</b>.", rel=0.03)]))
    pm = np.array([adg[pen == k].mean() for k in range(1, npen + 1)])
    pdg = pdiet
    tm = H.t_two(pm[pdg == "ctrl"], pm[pdg == "new"], True)
    tl = ["> pm <- aggregate(adg ~ pen + diet, data = d, FUN = mean)", "> t.test(adg ~ diet, data = pm, var.equal = TRUE)"]
    out = H.htest("Two Sample t-test", "adg by diet", [("t", tm["t"]), ("df", tm["df"])], tm["p"],
                  "true difference in means between group ctrl and group new is not equal to 0", tm["ci"], 95,
                  [("mean in group ctrl", tm["m1"]), ("mean in group new", tm["m2"])])
    S.append(step("Проверка другим путём: средние загонов", task="При одинаковом числе свиней в загонах t-тест по 12 средним даёт тот же ответ, что lmer.",
                  code=tl, out=out,
                  qs=[mc(rng, "Совпадает ли с lmer?", f"Да: |t| = {H.fmt5(abs(tm['t']))}, df = 10, p = {pr(tm['p'])} — как у lmer",
                         ["Нет: lmer всегда точнее", "Нет: средние загонов терять нельзя", "Совпадает только оценка, p разные"],
                         "Сбалансированный план: анализ средних загонов эквивалентен смешанной модели. Это хорошая проверка, что модель записана правильно.")]))
    icc = F["tau"] ** 2 / (F["tau"] ** 2 + F["sigma"] ** 2)
    de = 1 + (m - 1) * icc
    S.append(step("Эффективный размер выборки", task="Design effect DE = 1 + (m − 1)·ICC показывает, во сколько раз зависимость «съедает» информацию.",
                  qs=[num("Чему равен DE при 12 свиньях в загоне?", de, f"1 + 11 × {rua(icc, 3)} = <b>{rua(de, 3)}</b>.", rel=0.03),
                      num("Сколько «независимых свиней» стоят 144 свиньи?", N / de, f"144 / {rua(de, 3)} ≈ <b>{rua(N / de, 3)}</b>.", rel=0.03)]))
    u = F["u"]
    plots = [dict(k="rvf", x=rd(F["fitted"]), y=rd(F["resid"]), s=None, l=[int(i) for i in np.argsort(-np.abs(F["resid"]))[:3]], t="Residuals vs Fitted"),
             p_qq(u, title="Q-Q: random effects (pen)")]
    S.append(step("Остатки обоих уровней", code=["> plot(m)", "> qqnorm(ranef(m)$pen[, 1]); qqline(ranef(m)$pen[, 1])"], plots=plots,
                  qs=[mc(rng, "Почему на Q-Q всего 12 точек?", "Это случайные эффекты (BLUP) — по одному на загон",
                         ["Это 12 выбросов", "R показывает каждую 12-ю свинью", "Ошибка: должно быть 144"],
                         "У смешанной модели два уровня остатков: внутри загонов (свиньи) и между загонами (BLUP загонов).")]))
    rep = (f"Разница привеса new − ctrl {rua(F['b'][1], 3)} г/сут (lmer с загоном как случайным эффектом, df ≈ {rua(F['df'][1], 2)}, p = {pr(F['p'][1])}; 6 загонов на рацион)")
    S.append(step("Записать результат", qs=[mc(rng, "Какая фраза верна?", rep,
                                                [f"Новый рацион повышает привес (t-тест по 144 свиньям, p = {pr(naive['p'])})", "Загоны различаются, значит рацион работает",
                                                 "Смешанная модель уменьшила разницу средних"],
                                                "Оценка разницы та же; меняются SE и p. " + ("Здесь вывод поменялся: наивный тест «нашёл» эффект, которого данные не подтверждают." if v["kind"] == "pseudo" else "Здесь эффект большой и виден при любом честном анализе."))]))
    items = [("Единица — загон", True), ("Наивный t-тест по свиньям занижает p", True), ("Загон можно вставить фиксированным фактором вместе с рационом", False),
             ("Для планирования важно число загонов, а не свиней", True)]
    S.append(step("Проверка", qs=[multi(rng, "Отметь верное.", items, "Загон вложен в рацион: как фиксированный фактор он съедает весь эффект рациона.")]))
    return dict(g="mixed", key="lmm", title="Загоны: псевдоповторность и lmer", vt=("эффект только кажется" if v["kind"] == "pseudo" else "эффект настоящий"),
                goal="<p><b>Вопрос.</b> Повышает ли новый рацион привес на откорме? 12 загонов по 12 свиней, рацион назначали загону жребием.</p>", steps=S)


# =====================================================================
# 13. Repeated measures: weekly weights
# =====================================================================
REP_VAR = [dict(slope_d=0.9, kind="int"), dict(slope_d=0.0, kind="noint")]


def wf_rep(rng, v):
    npig, weeks = 20, np.arange(5)
    for _ in range(300):
        pig = np.repeat(np.arange(1, npig + 1), 5)
        tr = rng.permutation(np.repeat(["A", "B"], npig // 2))
        trt = tr[pig - 1]
        wk = np.tile(weeks, npig)
        u = rng.normal(0, 1.8, npig)
        wt = np.round(25 + 4.6 * wk + v["slope_d"] * wk * (trt == "B") + u[pig - 1] + rng.normal(0, 0.9, len(pig)), 1)
        B = (trt == "B").astype(float)
        X = np.c_[np.ones(len(pig)), wk, B, wk * B]
        F = lmm(wt, X, pig, ["(Intercept)", "week", "trtB", "week:trtB"])
        if v["kind"] == "int" and F["p"][3] < 0.001 and F["p"][2] > 0.2:
            break
        if v["kind"] == "noint" and F["p"][3] > 0.3 and F["p"][2] > 0.2:
            break
    else:
        raise RuntimeError("rep")
    N = len(pig)
    S = []
    S.append(step("Загрузить (длинный формат)", task="Таблица <code>d</code>: 20 свиней взвешивали каждую неделю 5 раз (неделя 0–4). Добавку A или B назначали свинье жребием.",
                  code=['> d <- read.csv("weights_long.csv")', "> head(d, 7)"],
                  out=df_print([("pig", "int", pig), ("trt", "chr", trt), ("week", "int", wk), ("wt", "num", wt)], rows=list(range(7))),
                  qs=[mc(rng, "Почему одна свинья занимает 5 строк?", "Длинный формат: одна строка — одно взвешивание",
                         ["Ошибка: дубли", "В каждой строке — другая свинья", "Так R хранит факторы"],
                         "Для lmer данные должны быть в длинном формате: свинья, время, значение. Широкий (wt0, wt1, …) переводят через tidyr::pivot_longer.")]))
    mm = {t: [wt[(trt == t) & (wk == w)].mean() for w in weeks] for t in ("A", "B")}
    S.append(step("Средние по неделям", code="> with(d, interaction.plot(week, trt, wt))",
                  plots=[p_int([str(w) for w in weeks], mm, xl="week", yl="mean of wt")],
                  qs=[mc(rng, "Что видно?", ("Линии расходятся: на B растут быстрее" if v["kind"] == "int" else "Линии почти параллельны: скорость роста похожая"),
                         (["Линии параллельны", "B стартовали тяжелее", "Свиньи не росли"] if v["kind"] == "int" else ["Линии расходятся", "A растут вдвое быстрее", "Рост не линейный"]),
                         "Нас интересует не масса в какой-то день, а <b>скорость роста</b> — наклон линии. Разница наклонов — взаимодействие week × trt.")]))
    ll = lmer_print("wt ~ week * trt + (1 | pig)", "d", F, ["(Intercept)", "week", "trtB", "week:trtB"], "pig")
    S.append(step("Модель повторных измерений",
                  pick=code_pick(rng, "Какой код?", "m <- lmer(wt ~ week * trt + (1 | pig), data = d)",
                                 ["m <- lm(wt ~ week * trt, data = d)", "m <- lmer(wt ~ week * trt + (1 | week), data = d)", "m <- t.test(wt ~ trt, data = d)"],
                                 "Одна свинья — 5 строк: нужен (1 | pig). lm считал бы 100 независимых взвешиваний. Неделя — фиксированный эффект (наклон), а не группа."),
                  code="> m <- lmer(wt ~ week * trt + (1 | pig), data = d)\n> summary(m)", out=ll,
                  qs=[mc(rng, "Что значит week:trtB?", f"На B свиньи прибавляют на {rua(F['b'][3], 2)} кг в неделю больше, чем на A",
                         [f"B тяжелее A на {rua(F['b'][3], 2)} кг всё время", "Разница в неделе 0", "Эффект недели у A"],
                         "week — наклон (кг/нед.) у базовой группы A; week:trtB — прибавка к наклону у B; trtB — разница в неделе 0 (при жребии около 0)."),
                      num("Скорость роста у B, кг/нед.?", F["b"][1] + F["b"][3], f"week + week:trtB = {rua(F['b'][1], 4)} + {rua(F['b'][3], 4)} = <b>{rua(F['b'][1] + F['b'][3], 4)}</b>.", rel=0.01)]))
    S.append(step("Почему не lm", qs=[mc(rng, "Что было бы с lm(wt ~ week * trt)?", "Те же оценки, но неверные SE: lm считает 100 независимых наблюдений",
                                          ["Оценки были бы другими, SE те же", "R выдал бы ошибку", "Ничего, lm и lmer одинаковы"],
                                          "Взвешивания одной свиньи похожи (тяжёлая остаётся тяжёлой). Для эффектов между свиньями (trt) lm занижает SE, для эффектов внутри (week) — обычно завышает.")]))
    rep = (f"Скорость роста на B была выше на {rua(F['b'][3], 2)} кг/нед. (взаимодействие week × trt, p = {pr(F['p'][3])}; lmer со случайным эффектом свиньи)" if v["kind"] == "int"
           else f"Скорость роста на A и B не различалась (взаимодействие week × trt: {rua(F['b'][3], 2)} кг/нед., p = {pr(F['p'][3])})")
    S.append(step("Записать результат", qs=[mc(rng, "Какая фраза верна?", rep,
                                                ["trtB не значим, значит добавка не влияет", "Свиньи на B тяжелее с первой недели", "Неделя значима — добавка работает"],
                                                "Главный вопрос опыта с ростом — разница наклонов.")]))
    items = [("Свинья — случайный эффект", True), ("Данные в длинном формате", True), ("(1 | pig) допускает, что скорость роста у всех свиней одинаковая внутри группы", True),
             ("Можно усложнить: (week | pig) — свой наклон у каждой свиньи", True), ("Неделю надо делать случайным эффектом", False)]
    S.append(step("Проверка", qs=[multi(rng, "Отметь верное.", items, "Случайный наклон (week | pig) позволяет свиньям расти с разной скоростью. Сравнивают модели по AIC или LRT.")]))
    return dict(g="mixed", key="rep", title="Повторные взвешивания: lmer", vt=("скорость разная" if v["kind"] == "int" else "скорость одинаковая"),
                goal="<p><b>Вопрос.</b> Растут ли свиньи на добавке B быстрее, чем на A? 20 свиней, 5 еженедельных взвешиваний.</p>", steps=S)


# =====================================================================
# 14. GLMM: mortality by pen
# =====================================================================
GLMM_VAR = [dict(b=-0.75, kind="eff"), dict(b=-0.1, kind="noeff")]


def wf_glmm(rng, v):
    npen, m = 24, 25
    for _ in range(200):
        pen = np.repeat(np.arange(1, npen + 1), m)
        pt = rng.permutation(np.repeat(["ctrl", "vacc"], npen // 2))
        trt = pt[pen - 1]
        u = rng.normal(0, 0.7, npen)
        eta = -2.3 + v["b"] * (trt == "vacc") + u[pen - 1]
        dead = (rng.random(npen * m) < 1 / (1 + np.exp(-eta))).astype(float)
        X = np.c_[np.ones(npen * m), (trt == "vacc").astype(float)]
        try:
            F = glmm(dead, X, pen, ["(Intercept)", "trtvacc"], fam="binomial")
        except ValueError:
            continue
        L = logistic(X, dead)
        if F["var"] < 0.15:
            continue
        if v["kind"] == "eff" and F["p"][1] < 0.03 and L["p"][1] < 0.001:
            break
        if v["kind"] == "noeff" and F["p"][1] > 0.3 and L["p"][1] < 0.2:
            break
    else:
        raise RuntimeError("glmm")
    N = npen * m
    S = []
    S.append(step("Загрузить", task="Таблица <code>d</code>: 600 поросят в 24 загонах по 25; вакцину или контроль назначали загону жребием; пал ли поросёнок за доращивание.",
                  code=['> d <- read.csv("nursery_mort.csv")', "> str(d)"],
                  out=str_df(N, [("pig", "int", range(1, N + 1)), ("pen", "int", pen), ("trt", "chr", trt), ("dead", "int", dead.astype(int))]),
                  qs=[mc(rng, "Почему нельзя просто glm по 600 поросятам?", "Падёж в загоне заразен и зависит от загона: поросята одного загона не независимы",
                         ["Потому что 0/1", "glm не работает с 600 строками", "Можно, это правильно"], "Единица — загон (вакцину назначали загону). Нужен случайный эффект загона.")]))
    pm = np.array([dead[pen == k].mean() for k in range(1, npen + 1)])
    S.append(step("Падёж по загонам", code=["> pm <- aggregate(dead ~ pen + trt, data = d, FUN = mean)", "> summary(pm$dead)"], out=[l for l in __import__("wf_lib").summary_vec(pm)],
                  qs=[mc(rng, "О чём говорит разброс падежа между загонами?", "Загоны сильно различаются — больше, чем дал бы случай при одинаковом риске",
                         ["Все загоны одинаковые", "Надо удалить загоны с высоким падежом", "Разброс маленький"],
                         f"От {ru(100 * pm.min(), 0)}% до {ru(100 * pm.max(), 0)}% при 25 поросятах в загоне — это сверхдисперсия между загонами.")]))
    gl, gd = glm_summary("dead ~ trt", "d", L, ["(Intercept)", "trtvacc"])
    S.append(step("Наивная логистическая (ошибка)", code="> summary(glm(dead ~ trt, family = binomial, data = d))", out=gl[2:],
                  qs=[num("SE у trtvacc в наивной модели?", L["se"][1], f"<b>{R.r7(L['se'][1])}</b> — запомни и сравни с glmer.", rel=0.02)]))
    gp = glmer_print("dead ~ trt + (1 | pen)", "d", F, ["(Intercept)", "trtvacc"], "pen", "binomial")
    S.append(step("GLMM",
                  pick=code_pick(rng, "Какой код?", "m <- glmer(dead ~ trt + (1 | pen), family = binomial, data = d, nAGQ = 10)",
                                 ["m <- lmer(dead ~ trt + (1 | pen), data = d)", "m <- glm(dead ~ trt + factor(pen), family = binomial, data = d)",
                                  "m <- glmer(dead ~ trt + (1 | dead), family = binomial, data = d)"],
                                 "0/1-исход и случайный эффект — glmer с family = binomial. lmer — для непрерывного отклика. Загон фиксированным фактором смешается с вакциной."),
                  code="> m <- glmer(dead ~ trt + (1 | pen), family = binomial, data = d, nAGQ = 10,\n+            control = glmerControl(optimizer = \"bobyqa\"))\n> summary(m)",
                  out=gp,
                  qs=[num("Во сколько раз SE у trtvacc больше, чем в наивной модели?", F["se"][1] / L["se"][1],
                          f"{R.r7(F['se'][1])} / {R.r7(L['se'][1])} = <b>{rua(F['se'][1] / L['se'][1], 3)}</b>. Наивная модель занижала неопределённость.", rel=0.03),
                      num("OR вакцины = exp(trtvacc)?", math.exp(F["b"][1]), f"exp({R.r7(F['b'][1])}) = <b>{rua(math.exp(F['b'][1]), 3)}</b> — шансы гибели в типичном загоне.", rel=0.02)]))
    rep = (f"Шансы гибели в загонах с вакциной составили {rua(math.exp(F['b'][1]), 2)} от контрольных (glmer, загон — случайный эффект, p = {pr(F['p'][1])}; 12 загонов на группу)")
    S.append(step("Записать результат", qs=[mc(rng, "Какая фраза верна?", rep,
                                                [f"Вакцина снижает гибель (glm по 600 поросятам, p = {pr(L['p'][1])})", "Дисперсия загонов значима, значит вакцина работает",
                                                 f"Падёж снизился на {ru(100 * (1 - math.exp(F['b'][1])), 0)} процентных пунктов"],
                                                "OR glmer — «внутри типичного загона» (условный). Это не процентные пункты.")]))
    items = [("Единица — загон", True), ("Наивный glm занижает SE", True), ("Отклик 0/1 — family = binomial", True), ("OR из glmer — разница рисков", False)]
    S.append(step("Проверка", qs=[multi(rng, "Отметь верное.", items, "Альтернатива — агрегировать: cbind(dead, alive) ~ trt + (1 | pen) — то же самое по загонам.")]))
    return dict(g="mixed", key="glmm", title="GLMM: падёж в загонах", vt=("вакцина работает" if v["kind"] == "eff" else "эффект не доказан"),
                goal="<p><b>Вопрос.</b> Снижает ли вакцина падёж на доращивании? 24 загона по 25 поросят, вакцину назначали загону жребием.</p>", steps=S)


# =====================================================================
# 15. Planning: power.t.test
# =====================================================================
POW_VAR = [dict(sd=38.4, sd_b=24.0, delta=25, kind="many"), dict(sd=21.5, sd_b=15.0, delta=30, kind="few")]


def power_t(n, delta, sd, alpha=0.05):
    nu = (n - 1) * 2
    q = stats.t.isf(alpha / 2, nu)
    return float(stats.nct.sf(q, nu, math.sqrt(n / 2) * delta / sd))


def solve_n(delta, sd, power=0.8):
    return H._brent(lambda n: power_t(n, delta, sd) - power, 2, 1e7)


def pt_print(n, delta, sd, power, nfmt=None):
    rows = [("n", n), ("delta", delta), ("sd", sd), ("sig.level", 0.05), ("power", power)]
    out = ["", "     Two-sample t test power calculation ", ""]
    for k, val in rows:
        out.append(f"{k.rjust(15)} = {R.r7(val) if not isinstance(val, str) else val}")
    out.append(f"{'alternative'.rjust(15)} = two.sided")
    out += ["", "NOTE: n is number in *each* group", ""]
    return out


def wf_power(rng, v):
    S = []
    pilot = np.round(rng.normal(820, v["sd"], 16))
    pilot = 820 + (pilot - pilot.mean()) / pilot.std(ddof=1) * v["sd"]
    pilot = np.round(pilot)
    sd = float(np.std(pilot, ddof=1))
    S.append(step("SD по прошлым данным", task="Прошлая партия: средний привес 16 загонов. Разброс <b>между загонами</b> — основа расчёта, раз единица — загон.",
                  code=['> pilot <- read.csv("last_batch_pens.csv")', "> sd(pilot$adg)"], out=[f"[1] {R.r7(sd)}"],
                  qs=[mc(rng, "Почему SD загонов, а не свиней?", "Рацион будем назначать загонам: ошибка в опыте — разброс между загонами",
                         ["SD свиней всегда меньше", "Так проще", "SD загонов больше, это надёжнее"],
                         "Сколько единиц эксперимента нужно, зависит от разброса <b>единиц</b>. Разброс свиней внутри загона в этот SD уже «спрятан» через средние.")]))
    n1 = solve_n(v["delta"], sd)
    S.append(step("Сколько загонов на рацион",
                  pick=code_pick(rng, f"Хотим заметить разницу {v['delta']} г/сут с мощностью 80%. Какой код?",
                                 f"power.t.test(delta = {v['delta']}, sd = {R.r7(sd)}, power = 0.8)",
                                 [f"power.t.test(n = {v['delta']}, sd = {R.r7(sd)}, power = 0.8)", f"t.test(pilot$adg, mu = {v['delta']})",
                                  f"power.t.test(delta = {R.r7(sd)}, sd = {v['delta']}, power = 0.8)"],
                                 "Задаём всё, кроме неизвестного (n), — R решит уравнение. delta — минимальная важная разница, sd — разброс единиц."),
                  code=f"> power.t.test(delta = {v['delta']}, sd = {R.r7(sd)}, power = 0.8)", out=pt_print(n1, v["delta"], float(R.r7(sd)), 0.8),
                  qs=[num("Сколько загонов на рацион надо взять (округли вверх)?", math.ceil(n1), f"n = {R.r7(n1)} → <b>{math.ceil(n1)}</b> загонов в <b>каждой</b> группе (всего {2 * math.ceil(n1)}).", rel=0)]))
    pw = power_t(12, v["delta"], float(R.r7(sd)))
    S.append(step("А если загонов только 12 на рацион?", code=f"> power.t.test(n = 12, delta = {v['delta']}, sd = {R.r7(sd)})", out=pt_print(12, v["delta"], float(R.r7(sd)), pw),
                  qs=[num("Мощность при 12 загонах на рацион, %?", 100 * pw, f"power = {R.r7(pw)} → <b>{ru(100 * pw, 0)}</b>%: с такой вероятностью опыт заметит разницу {v['delta']} г/сут, если она есть.", rel=0.02, abs_=0.5),
                      mc(rng, "Что значит низкая мощность?", "Даже если эффект есть, опыт часто его не найдёт — и «не значимо» ничего не будет значить",
                         ["Опыт найдёт эффект, просто p будет больше", "Ошибок I рода станет больше", "Ничего, если делать опыт аккуратно"],
                         "Мощность — шанс найти эффект заданного размера. При 50% мощности опыт — подбрасывание монеты.")]))
    nb = solve_n(v["delta"], v["sd_b"])
    S.append(step("Блоки уменьшают SD", task=f"Если разбить загоны на блоки по комнатам, остаточный SD внутри блоков по прошлым данным ≈ {R.r7(v['sd_b'])} г/сут.",
                  code=f"> power.t.test(delta = {v['delta']}, sd = {R.r7(v['sd_b'])}, power = 0.8)$n", out=[f"[1] {R.r7(nb)}"],
                  qs=[num("Сколько загонов на рацион теперь?", math.ceil(nb), f"{R.r7(nb)} → <b>{math.ceil(nb)}</b>. Блоки экономят загоны, когда комнаты разные.", rel=0)]))
    S.append(step("Почему не считать свиней", qs=[mc(rng, "Коллега говорит: «В загоне 25 свиней, значит 5 загонов = 125 свиней, хватит». Что ответить?",
                                                     "Свиньи загона зависимы: эффективное n ≈ загоны × 25 / DE, а для рациона на загон нужно считать загоны",
                                                     ["Прав: 125 &gt; 38", "Надо считать свиней, но добавить 10%", "Свиньи или загоны — всё равно"],
                                                     "Когда вариант назначают загону, информация о нём — в числе загонов. Свиньи внутри загона уточняют среднее загона, но не заменяют другие загоны.")]))
    rep = f"Для обнаружения разницы {v['delta']} г/сут при SD загонов {rua(sd, 3)} г/сут (α = 0,05, мощность 80%) нужно {math.ceil(n1)} загонов на рацион; с блоками по комнатам — {math.ceil(nb)}"
    S.append(step("Записать план", qs=[mc(rng, "Какая фраза для протокола верна?", rep,
                                           [f"Нужно {math.ceil(n1)} свиней на рацион", "Мощность 80% означает, что эффект будет найден с p = 0,8",
                                            "Размер выборки не нужен: опыт покажет"],
                                           "В протоколе: эффект, SD и откуда он, α, мощность, единица и итоговое число единиц.")]))
    items = [("delta — минимальная экономически важная разница", True), ("SD взят по единицам эксперимента", True),
             ("Если опыт окажется «не значимым», это доказывает отсутствие эффекта", False), ("Расчёт делают до опыта", True)]
    S.append(step("Проверка", qs=[multi(rng, "Отметь верное.", items, "Расчёт мощности после опыта по полученному эффекту бессмыслен — он просто повторяет p.")]))
    return dict(g="plan", key="power", title="Сколько загонов нужно: power.t.test", vt=("большой разброс" if v["kind"] == "many" else "малый разброс"),
                goal="<p><b>Вопрос.</b> Сколько загонов нужно для опыта с рационом? Готовим протокол по данным прошлой партии.</p>", steps=S)


def build_c(rng):
    out = []
    for v in LOGIT_VAR:
        out.append(wf_logit(rng, v))
    for v in POIS_VAR:
        out.append(wf_pois(rng, v))
    for v in LMM_VAR:
        out.append(wf_lmm(rng, v))
    for v in REP_VAR:
        out.append(wf_rep(rng, v))
    for v in GLMM_VAR:
        out.append(wf_glmm(rng, v))
    for v in POW_VAR:
        out.append(wf_power(rng, v))
    return out
