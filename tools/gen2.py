"""v1.1 additions: Poisson regression (groups + offset, continuous X) and multiple linear regression."""
import math
import numpy as np
from scipy import stats
import rfmt as R
from models import ols, predict, poisson
import contexts as C
from gen import ru, rua, u, code, mc, num, pick, coef_block, lm_summary, qt_line, design_ru


# ---------------------------------------------------------------------------
# helpers
def deparse_call(fn, args):
    """Mimic R deparse(width.cutoff = 60): break after an argument's ', ' once the line is longer than 60."""
    out, cur = [], fn + "("
    for i, a in enumerate(args):
        last = i == len(args) - 1
        cur += a + (")" if last else ", ")
        if not last and len(cur) > 60:
            out.append(cur)
            cur = "    "
    out.append(cur)
    return out


def dev_lines(L):
    nd, rd = R.fmt_joint([L["null_dev"], L["res_dev"]], 5)
    w = max(len(nd), len(rd))
    dfs = [str(L["df_null"]), str(L["df_res"])]
    wd = max(len(s) for s in dfs)
    return [f"    Null deviance: {nd.rjust(w)}  on {dfs[0].rjust(wd)}  degrees of freedom",
            f"Residual deviance: {rd.rjust(w)}  on {dfs[1].rjust(wd)}  degrees of freedom",
            f"AIC: {R.fmt1(L['aic'], 5)}", "", f"Number of Fisher Scoring iterations: {L['iters']}"]


def pois_summary(call_user, call_args, P, names):
    lines = [f"> m <- {call_user}", "> summary(m)", "", "Call:"] + deparse_call("glm", call_args) + ["", "Coefficients:"]
    cl, est, se, zv, ps = coef_block(names, P["b"], P["se"], P["z"], P["p"], statname="z")
    lines += cl + ["---", R.SIGNIF, "", "(Dispersion parameter for poisson family taken to be 1)", ""]
    lines += dev_lines(P)
    return lines, dict(est=est, se=se, z=zv, p=ps)


def irr_table(names, est, se):
    b = np.array(est)
    s = np.array(se)
    cols = [list(np.exp(b)), list(np.exp(b - 1.959964 * s)), list(np.exp(b + 1.959964 * s))]
    return ["> exp(cbind(IRR = coef(m), confint.default(m)))"] + R.matrix(names, ["IRR", "2.5 %", "97.5 %"], cols)


def disp_lines(P, pearson=False):
    r = P["res_dev"] / P["df_res"]
    out = ["> deviance(m) / df.residual(m)", "[1] " + R.r7(r)]
    if pearson:
        out += ["> sum(residuals(m, type = \"pearson\")^2) / df.residual(m)", "[1] " + R.r7(P["pearson"] / P["df_res"])]
    return out, float(R.r7(r))


def gamma_pois(rng, mu, theta):
    if theta is None:
        return rng.poisson(mu)
    lam = rng.gamma(theta, mu / theta)
    return rng.poisson(lam)


def times(irr):
    return "выше" if irr > 1 else "ниже"


def ratio(irr):
    """'в 1,8 раза выше' / 'в 1,8 раза ниже' (always a factor >= 1)."""
    return f"в {rua(irr if irr >= 1 else 1 / irr, 3)} раза {times(irr)}"


TWO = {"pigs": "две свиньи", "carcass": "две туши", "cows": "две коровы", "litters": "два гнезда",
       "piglets": "два поросёнка", "kids": "двое детей", "plots": "две делянки"}


def disp_verdict(r):
    if r < 1.25:
        return "ok"
    return "over"


def gr(v):
    return f"{v:g}".replace(".", ",")


def intro_pois(c, extra):
    return f"<p>{c.get('what', '')}</p><p>Отклик {code(c['y'])}: {c['y_ru']} (счёт: 0, 1, 2, …). {extra}</p><p class='dz'>{design_ru(c)}</p>"


DISP_EX = ("Пуассон считает, что дисперсия = среднему. Проверка: Residual deviance / df. "
           "Около 1 — нормально. Больше ~1,25–1,5 — <b>передисперсия</b>: SE и p слишком оптимистичны, "
           "нужен quasipoisson или отрицательная биномиальная (MASS::glm.nb). "
           "Меньше 1 — недодисперсия, она не так опасна.")


# ---------------------------------------------------------------------------
# Poisson with groups and offset
def gen_pois_grp(rng, c):
    lv = sorted(c["rates"])
    k = len(lv)
    ref = lv[0]
    n_rows = int(rng.integers(c["n"][0], c["n"][1] + 1))
    shrink = rng.choice([0.1, 0.7, 1.0, 1.0, 1.0, 1.2])
    base = c["rates"][ref] * rng.uniform(0.85, 1.15)
    rates = {l: base * (c["rates"][l] / c["rates"][ref]) ** shrink for l in lv}
    theta = rng.choice([None, None, None, 12.0, 4.0, 2.5])
    g = np.array([lv[i % k] for i in range(n_rows)])
    rng.shuffle(g)
    E = np.round(rng.uniform(c["olo"], c["ohi"], n_rows), c["odec"])
    mu = np.array([rates[gi] for gi in g]) * E
    y = gamma_pois(rng, mu, theta).astype(float)
    X = np.c_[np.ones(n_rows)]
    for l in lv[1:]:
        X = np.c_[X, (g == l).astype(float)]
    P = poisson(X, y, np.log(E))
    names = ["(Intercept)"] + [f"{c['g']}{l}" for l in lv[1:]]
    style = rng.integers(0, 3)
    fam = 'poisson(link = "log")' if rng.random() < 0.35 else "poisson"
    if style == 0:
        form = f"{c['y']} ~ {c['g']} + offset(log({c['off']}))"
        call_user = f"glm({form}, family = {fam}, data = {c['data']})"
        call_args = [f"formula = {form}", f"family = {fam}", f"data = {c['data']}"]
    else:
        form = f"{c['y']} ~ {c['g']}"
        call_user = f"glm({form}, offset = log({c['off']}), family = {fam}, data = {c['data']})"
        call_args = [f"formula = {form}", f"family = {fam}", f"data = {c['data']}", f"offset = log({c['off']})"]
        # R keeps the user's argument order
        call_args = [f"formula = {form}", f"offset = log({c['off']})", f"family = {fam}", f"data = {c['data']}"]
    lines, d = pois_summary(call_user, call_args, P, names)
    show_irr = rng.random() < 0.45
    if show_irr:
        lines += irr_table(names, d["est"], d["se"])
    show_disp = rng.random() < 0.5
    rdisp = P["res_dev"] / P["df_res"]
    if show_disp:
        dl, rdisp_shown = disp_lines(P, pearson=rng.random() < 0.4)
        lines += dl
    est, se = d["est"], d["se"]
    b0 = est[0]
    j = int(rng.integers(1, k))  # which non-reference level to ask about
    oth = lv[j]
    bj, sej = est[j], se[j]
    IRR = math.exp(bj)
    per, per_ru, unit1 = c["per"], c["per_ru"], c["unit1"]
    qs_req, qs_opt = [], []
    qs_req.append(mc(rng, "Какая группа референтная (её нет среди строк коэффициентов)?", ref,
                     ([l for l in lv[1:]] + ["Никакая: (Intercept) — общее среднее", "Все группы сразу"])[:3],
                     f"R берёт первую по алфавиту группу ({ref}) за базу. Она спрятана в (Intercept): "
                     f"log(частоты {ref}) = b₀. Строки {', '.join(names[1:])} — разницы с {ref} на шкале логарифма."))
    qs_req.append(num(f"Отношение частот (IRR) «{oth} против {ref}»?", IRR,
                      f"IRR = e<sup>b</sup> = e<sup>{rua(bj, 5)}</sup> = <b>{rua(IRR, 4)}</b>. "
                      f"Частота ({c['ev']}) в {oth} {ratio(IRR)}, чем в {ref}"
                      f"{'' if IRR >= 1 else f' (1 / {rua(IRR, 4)} = {rua(1 / IRR, 3)})'}.", rel=0.005))
    pct = abs(IRR - 1) * 100
    qs_req.append(mc(rng, f"Как правильно прочитать IRR = {rua(IRR, 3)} для {names[j]}?",
                     f"Частота ({c['ev']}) в {oth} {ratio(IRR)}, чем в {ref}",
                     [f"Шансы события в {oth} {ratio(IRR)}, чем в {ref}",
                      f"В каждой строке {oth} на {rua(abs(bj), 3)} случая {'больше' if bj > 0 else 'меньше'}, чем в {ref}",
                      f"Частота в {ref} {ratio(IRR)}, чем в {oth}"],
                     f"В пуассоновской e<sup>b</sup> — отношение <b>частот</b> (incidence rate ratio), не шансов. "
                     f"Направление: группа из строки против референтной. Эффекты умножают частоту, а не прибавляют случаи. "
                     f"В процентах: {'+' if IRR > 1 else '−'}{rua(pct, 3)} %."
                     f"<br><i>EN:</i> The rate of {c['y']} in the {oth} group is estimated to be {rua(IRR, 3).replace(',', '.')} times the rate in the {ref} group."))
    qs_opt.append(num(f"На сколько процентов частота в {oth} {times(IRR)}, чем в {ref}? (число без знака)", pct,
                      f"(IRR − 1) × 100 % = ({rua(IRR, 4)} − 1) × 100 = {'+' if IRR > 1 else '−'}<b>{rua(pct, 3)}</b> %.",
                      rel=0.01, abs_=0.3, unit="%", absval=True))
    rate_ref = per * math.exp(b0)
    qs_opt.append(num(f"Частота в группе {ref} {per_ru}?", rate_ref,
                      f"b₀ = log(частоты {ref} {unit1}). e<sup>{rua(b0, 5)}</sup> = {rua(math.exp(b0), 4)} {unit1}. "
                      f"× {per} = <b>{rua(rate_ref, 4)}</b> {per_ru}. Это работает только потому, что в модели есть offset: "
                      f"без него e<sup>b₀</sup> было бы ожидаемым числом случаев на одну строку данных.", rel=0.01))
    E0 = float(np.round(rng.uniform(c["olo"], c["ohi"]), c["odec"]))
    E0s = gr(E0)
    lam = E0 * math.exp(b0 + bj)
    qs_opt.append(num(f"Сколько случаев ждём в строке группы {oth} с {c['off']} = {E0s}?", lam,
                      f"log(μ) = log({c['off']}) + b₀ + b = log({E0s}) + {rua(b0, 5)} + ({rua(bj, 5)}). "
                      f"Частота {oth} = e<sup>{rua(b0 + bj, 4)}</sup> = {rua(math.exp(b0 + bj), 4)} {unit1}. "
                      f"μ = {E0s} × {rua(math.exp(b0 + bj), 4)} = <b>{rua(lam, 4)}</b> случая.", rel=0.01))
    qs_opt.append(mc(rng, f"Зачем в модели log({c['off']}) как offset?",
                     f"Чтобы модель описывала частоту {unit1}: {c['off']} у строк разный, а коэффициент у log({c['off']}) зафиксирован на 1",
                     [f"Чтобы проверить, влияет ли {c['off']} на {c['y']}: R оценит для него коэффициент",
                      f"Чтобы убрать нули из {c['y']}", f"Чтобы перевести {c['y']} в логарифм, как в lm(log(y) ~ x)"],
                     f"Offset — «предиктор» с коэффициентом ровно 1, его не оценивают: вдвое больше {c['off']} при той же частоте → вдвое больше случаев. "
                     f"Поэтому в выводе нет строки для {c['off']}. В модель он входит как логарифм, потому что модель живёт в мире log(μ)."))
    dex = (f"Residual deviance / df = {R.fmt_joint([P['res_dev']], 5)[0]} / {P['df_res']} = <b>{rua(rdisp, 3)}</b>. " + DISP_EX)
    qs_opt.append(num("Посчитай Residual deviance / df (проверка передисперсии).", rdisp, dex, rel=0.02))
    over = rdisp >= 1.25
    qs_opt.append(mc(rng, f"Residual deviance / df ≈ {rua(rdisp, 3)}. Что делать?",
                     ("Есть передисперсия: SE занижены, перейти к quasipoisson или MASS::glm.nb" if over else
                      "Передисперсии нет, пуассоновской модели можно доверять"),
                     (["Всё хорошо: чем больше отношение, тем лучше модель", "Убрать offset", "Перейти к логистической регрессии"] if over else
                      ["Есть передисперсия: перейти к glm.nb", "Отношение должно быть около 0", "Модель плохая: p-value надо умножить на это число"]),
                     dex))
    up = math.exp(bj + 1.96 * sej)
    lo_ = math.exp(bj - 1.96 * sej)
    qs_opt.append(num(f"95% ДИ для IRR {names[j]}: верхняя граница (±1,96·SE)?", up,
                      f"На шкале логарифма: {rua(bj, 5)} + 1,96 × {rua(sej, 4)} = {rua(bj + 1.96 * sej, 4)}. Потом e: <b>{rua(up, 4)}</b>. "
                      f"Нижняя: e<sup>{rua(bj - 1.96 * sej, 4)}</sup> = {rua(lo_, 4)}. "
                      f"{'1 внутри → нет доказательств различия частот.' if lo_ < 1 < up else '1 снаружи → частоты различаются значимо.'}",
                      rel=0.01))
    sig = P["p"][j] < 0.05
    qs_opt.append(mc(rng, f"Строка {names[j]}: p = {d['p'][j]}. Что проверяет тест и какой вывод?",
                     ("H₀: β = 0 (IRR = 1, частоты равны) — отвергаем" if sig else "H₀: β = 0 (IRR = 1, частоты равны) — не отвергаем"),
                     (["H₀: частота в группе = 0 — отвергаем", "H₀: IRR = 0 — отвергаем", "H₀: β = 1 — отвергаем"] if sig else
                      ["H₀: β = 0 — отвергаем", "H₀: IRR = 0 — не отвергаем", "Доказано, что частоты одинаковы"]),
                     "b = 0 на шкале логарифма ⇔ IRR = e⁰ = 1. z = Estimate / SE. Если есть передисперсия, этому p верить нельзя: SE занижена."))
    qs_opt.append(num(f"IRR наоборот: «{ref} против {oth}»?", 1 / IRR,
                      f"1 / {rua(IRR, 4)} = <b>{rua(1 / IRR, 4)}</b> (то же, что e<sup>−b</sup>). Отдельной строки для этого в выводе нет.",
                      rel=0.005))
    lr = P["null_dev"] - P["res_dev"]
    nd_s, rd_s = R.fmt_joint([P["null_dev"], P["res_dev"]], 5)
    qs_opt.append(num(f"Тест всей модели: χ² = Null deviance − Residual deviance. Чему он равен?", float(nd_s) - float(rd_s),
                      f"{nd_s} − {rd_s} = <b>{rua(float(nd_s) - float(rd_s), 4)}</b> при df = {P['df_null']} − {P['df_res']} = {k - 1}. "
                      f"p = {R.pval1(stats.chi2.sf(lr, k - 1))}: {'группы различаются' if stats.chi2.sf(lr, k - 1) < 0.05 else 'нет доказательств различий'}.",
                      rel=0.005))
    if k == 3:
        a_, b_ = lv[1], lv[2]
        ba, bb = est[1], est[2]
        qs_opt.append(num(f"IRR «{b_} против {a_}» (обе не референтные)?", math.exp(bb - ba),
                          f"Обе строки — разницы с {ref}. Разница между ними: {rua(bb, 5)} − ({rua(ba, 5)}) = {rua(bb - ba, 4)}. "
                          f"IRR = e<sup>{rua(bb - ba, 4)}</sup> = <b>{rua(math.exp(bb - ba), 4)}</b>. Или так: {rua(math.exp(bb), 4)} / {rua(math.exp(ba), 4)}.",
                          rel=0.005))
        qs_req.append(qs_opt.pop())
    qs = pick(rng, qs_req, qs_opt, 5)
    extra = (f"Группа {code(c['g'])}: {', '.join(lv)}. {code(c['off'])} — {c['off_ru']}. "
             f"Всего {c['eu']}: {n_rows}.")
    sub = "p2" if k == 2 else "p3"
    return dict(cat="pois", sub=sub, title=f"Пуассоновская: {form}", intro=intro_pois(c, extra),
                console="\n".join(lines), qs=qs)


# ---------------------------------------------------------------------------
# Poisson with continuous X
def gen_poisx(rng, c):
    n = int(rng.integers(c["n"][0], c["n"][1] + 1))
    x = np.round(rng.uniform(c["xlo"], c["xhi"], n), c["xdec"])
    f = rng.choice([0.05, 0.6, 1.0, 1.0, 1.0, 1.3])
    b1t = c["b1"] * f
    b0t = c["b0"] + (c["b1"] - b1t) * x.mean()
    theta = rng.choice([None, None, None, 10.0, 3.0])
    y = gamma_pois(rng, np.exp(b0t + b1t * x), theta).astype(float)
    P = poisson(np.c_[np.ones(n), x], y)
    names = ["(Intercept)", c["x"]]
    fam = 'poisson(link = "log")' if rng.random() < 0.35 else "poisson"
    form = f"{c['y']} ~ {c['x']}"
    call_user = f"glm({form}, family = {fam}, data = {c['data']})"
    lines, d = pois_summary(call_user, [f"formula = {form}", f"family = {fam}", f"data = {c['data']}"], P, names)
    if rng.random() < 0.4:
        lines += ["> exp(coef(m))"] + R.named_vector(names, R.fmt_joint([math.exp(b) for b in d["est"]], 7))
    rdisp = P["res_dev"] / P["df_res"]
    if rng.random() < 0.5:
        dl, _ = disp_lines(P, pearson=rng.random() < 0.4)
        lines += dl
    b0, b1 = d["est"]
    se1 = d["se"][1]
    IRR = math.exp(b1)
    xu = c["xu"]
    step = c["step"]
    qs_req, qs_opt = [], []
    qs_req.append(num(f"Во сколько раз меняется ожидаемое {c['y']} при +1 {xu} к {c['x']} (IRR)?", IRR,
                      f"IRR = e<sup>b₁</sup> = e<sup>{rua(b1, 5)}</sup> = <b>{rua(IRR, 4)}</b>. "
                      f"Каждая +1 {xu} умножает ожидаемый счёт на {rua(IRR, 4)}.", rel=0.003))
    pct = abs(IRR - 1) * 100
    qs_req.append(mc(rng, f"Как читать b₁ = {rua(b1, 4)} для {c['x']}?",
                     f"При +1 {xu} к {c['x']} ожидаемое {c['y']} умножается на e<sup>{rua(b1, 4)}</sup> = {rua(IRR, 3)} "
                     f"({'+' if IRR > 1 else '−'}{rua(pct, 3)} %)",
                     [f"При +1 {xu} к {c['x']} ожидаемое {c['y']} {'растёт' if b1 > 0 else 'падает'} на {rua(abs(b1), 3)} случая",
                      f"При +1 {xu} к {c['x']} вероятность события {'растёт' if b1 > 0 else 'падает'} на {rua(abs(b1), 3)}",
                      f"При +1 {xu} к {c['x']} шансы события умножаются на {rua(IRR, 3)}"],
                     "Модель: log(μ) = b₀ + b₁·x. Плюс b₁ в мире логарифма = умножение на e<sup>b₁</sup> в мире счёта. "
                     "Поэтому эффект читают «в разах» или в процентах, а не «плюс столько-то случаев». Шансы — это логистическая."))
    big = math.exp(step * b1)
    qs_opt.append(num(f"Во сколько раз меняется ожидаемое {c['y']} при +{gr(step)} {xu} к {c['x']}?", big,
                      f"e<sup>{gr(step)} × b₁</sup> = e<sup>{gr(step)} × {rua(b1, 5)}</sup> = e<sup>{rua(step * b1, 4)}</sup> = <b>{rua(big, 4)}</b>. "
                      f"Или {rua(IRR, 4)}<sup>{gr(step)}</sup>. Не {gr(step)} × {rua(IRR, 3)}: эффекты перемножаются.", rel=0.01))
    qs_opt.append(num(f"На сколько процентов меняется ожидаемое {c['y']} при +1 {xu}? (число без знака)", pct,
                      f"(IRR − 1) × 100 = ({rua(IRR, 4)} − 1) × 100 = {'+' if IRR > 1 else '−'}<b>{rua(pct, 3)}</b> %.",
                      rel=0.02, abs_=0.1, unit="%", absval=True))
    x0 = float(np.round(rng.uniform(c["xlo"] + 0.2 * (c["xhi"] - c["xlo"]), c["xhi"] - 0.2 * (c["xhi"] - c["xlo"])), c["xdec"]))
    mu0 = math.exp(b0 + b1 * x0)
    qs_opt.append(num(f"Ожидаемое {c['y']} при {c['x']} = {gr(x0)} {xu}?", mu0,
                      f"log(μ) = b₀ + b₁·x = {rua(b0, 5)} + {rua(b1, 5)} × {gr(x0)} = {rua(b0 + b1 * x0, 4)}. "
                      f"μ = e<sup>{rua(b0 + b1 * x0, 4)}</sup> = <b>{rua(mu0, 4)}</b>. Последний шаг — e, иначе получишь логарифм счёта.",
                      rel=0.01))
    dex = (f"Residual deviance / df = {R.fmt_joint([P['res_dev']], 5)[0]} / {P['df_res']} = <b>{rua(rdisp, 3)}</b>. " + DISP_EX)
    qs_opt.append(num("Посчитай Residual deviance / df.", rdisp, dex, rel=0.02))
    over = rdisp >= 1.25
    qs_opt.append(mc(rng, f"Residual deviance / df ≈ {rua(rdisp, 3)}. Вывод?",
                     ("Передисперсия: p-value слишком оптимистичны, нужен quasipoisson или glm.nb" if over else
                      "Передисперсии нет: допущение «дисперсия = среднему» выглядит разумным"),
                     (["Модель отличная: большое отношение — хорошее согласие", "Нужно добавить offset", "Нужно логарифмировать x"] if over else
                      ["Передисперсия: нужен glm.nb", "Отношение должно быть около 0", "Нужно перейти к логистической регрессии"]),
                     dex))
    up = math.exp(b1 + 1.96 * se1)
    qs_opt.append(num(f"95% ДИ для IRR (на +1 {xu}): верхняя граница?", up,
                      f"e<sup>{rua(b1, 5)} + 1,96 × {rua(se1, 4)}</sup> = e<sup>{rua(b1 + 1.96 * se1, 4)}</sup> = <b>{rua(up, 4)}</b>. "
                      f"Нижняя: {rua(math.exp(b1 - 1.96 * se1), 4)}.", rel=0.01))
    nd_s, rd_s = R.fmt_joint([P["null_dev"], P["res_dev"]], 5)
    lr = float(nd_s) - float(rd_s)
    qs_opt.append(num("χ² теста отношения правдоподобий: Null deviance − Residual deviance?", lr,
                      f"{nd_s} − {rd_s} = <b>{rua(lr, 4)}</b>, df = {P['df_null']} − {P['df_res']} = 1. "
                      f"p = {R.pval1(stats.chi2.sf(lr, 1))}. Это та же идея, что в логистической: насколько модель лучше «пустой».",
                      rel=0.005))
    qs_opt.append(mc(rng, f"Что такое e<sup>b₀</sup> = {rua(math.exp(b0), 4)}?",
                     f"Ожидаемое {c['y']} при {c['x']} = 0" + ("" if c["xlo"] <= 0 else " (это за пределами данных, сам по себе смысла нет)"),
                     [f"Среднее {c['y']} по всем данным", "Частота события в референтной группе",
                      f"Во сколько раз растёт {c['y']} при +1 к {c['x']}"],
                     "Intercept ставит кривую на место: log(μ) при x = 0. e<sup>b₀</sup> — ожидаемый счёт при x = 0. Если x = 0 нет в данных, это экстраполяция."))
    qs = pick(rng, qs_req, qs_opt, 5)
    extra = f"Предиктор {code(c['x'])} — {c['x_ru']}, {xu}; в данных от {rua(x.min(), 3)} до {rua(x.max(), 3)}. Всего {c['eu']}: {n}."
    return dict(cat="pois", sub="px", title=f"Пуассоновская: {form}", intro=intro_pois(c, extra),
                console="\n".join(lines), qs=qs)


# ---------------------------------------------------------------------------
# Multiple linear regression
def vif_vals(X):
    """X without intercept column; VIF_j = 1/(1-R2_j)."""
    out = []
    for j in range(X.shape[1]):
        yj = X[:, j]
        others = np.c_[np.ones(len(yj)), np.delete(X, j, axis=1)]
        f = ols(others, yj)
        out.append(1 / (1 - f["r2"]))
    return out


def gen_mlr(rng, c):
    n = int(rng.integers(c["n"][0], c["n"][1] + 1))
    z1 = rng.normal(size=n)
    rho = c["rho"]
    if c["data"] == "kids":
        rho = rng.choice([0.75, 0.85, 0.93])
    z2 = rho * z1 + math.sqrt(1 - rho ** 2) * rng.normal(size=n)

    def scale(z, lo, hi, dec):
        v = (lo + hi) / 2 + z * (hi - lo) / 4.5
        return np.round(np.clip(v, lo, hi), dec)

    x1 = scale(z1, c["x1lo"], c["x1hi"], c["x1dec"])
    x2 = scale(z2, c["x2lo"], c["x2hi"], c["x2dec"])
    b2t = c["b2"] * rng.choice([0.03, 0.6, 1.0, 1.0, 1.0])
    b1t = c["b1"] * rng.choice([0.8, 1.0, 1.0, 1.2])
    mean = c["b0"] + b1t * x1 + b2t * x2 + (c["b2"] - b2t) * x2.mean()
    has_g = bool(c["g"]) and rng.random() < 0.75
    cols = [np.ones(n), x1, x2]
    names = ["(Intercept)", c["x1"], c["x2"]]
    if has_g:
        lv = sorted(c["levels"])
        gi = rng.integers(0, 2, n)
        g = np.array([lv[i] for i in gi])
        bg = c["bg"] * rng.choice([0.05, 0.7, 1.0, 1.0])
        # bg is defined for levels[1] vs levels[0] as written; convert to sorted reference
        sign = 1 if c["levels"][0] == lv[0] else -1
        mean = mean + sign * bg * (g == lv[1])
        cols.append((g == lv[1]).astype(float))
        names.append(f"{c['g']}{lv[1]}")
    y = np.round(mean + rng.normal(0, c["sd"], n), c["ydec"])
    X = np.column_stack(cols)
    f = ols(X, y)
    formula = f"{c['y']} ~ {c['x1']} + {c['x2']}" + (f" + {c['g']}" if has_g else "")
    lines, d = lm_summary(formula, c["data"], f, names)
    est, se = d["est"], d["se"]
    show_vif = rng.random() < 0.35 or c["data"] == "kids"
    if show_vif:
        v = vif_vals(X[:, 1:])
        lines += ["> car::vif(m)"] + R.named_vector(names[1:], R.fmt_joint(v, 7))
    tc = None
    if rng.random() < 0.5:
        tc = qt_line(f["df"], lines)
    show_pred = rng.random() < 0.4
    x1v = float(np.round(rng.uniform(np.quantile(x1, 0.25), np.quantile(x1, 0.75)), c["x1dec"]))
    x2v = float(np.round(rng.uniform(np.quantile(x2, 0.25), np.quantile(x2, 0.75)), c["x2dec"]))
    gv = None
    if has_g:
        gv = lv[int(rng.integers(0, 2))]
    yhat = est[0] + est[1] * x1v + est[2] * x2v + (est[3] if has_g and gv == lv[1] else 0.0)
    if show_pred:
        x1p = float(np.round(np.quantile(x1, 0.5), c["x1dec"]))
        x2p = float(np.round(np.quantile(x2, 0.5), c["x2dec"]))
        gp = lv[0] if has_g else None
        kind_iv = "confidence" if rng.random() < 0.5 else "prediction"
        nd = f"{c['x1']} = {x1p:g}, {c['x2']} = {x2p:g}" + (f", {c['g']} = \"{gp}\"" if has_g else "")
        x0 = [1, x1p, x2p] + ([0.0] if has_g else [])
        pr = predict(f, x0)
        iv = pr["ci"] if kind_iv == "confidence" else pr["pi"]
        lines += [f"> predict(m, newdata = data.frame({nd}),", f"+         interval = \"{kind_iv}\")"]
        lines += R.matrix(["1"], ["fit", "lwr", "upr"], [[pr["fit"]], [iv[0]], [iv[1]]])
    yu = c["yu"]
    b1, b2 = est[1], est[2]
    qs_req, qs_opt = [], []
    others = c["x2"] + (f" и {c['g']}" if has_g else "")
    dirw = "выше" if b1 > 0 else "ниже"
    qs_req.append(mc(rng, f"Как читать коэффициент {c['x1']} = {rua(b1, 4)}?",
                     f"При +1 {c['x1u']} к {c['x1']} и тех же {others} среднее {c['y']} {dirw} на {rua(abs(b1), 4)}{u(yu)}",
                     [f"При +1 {c['x1u']} к {c['x1']} среднее {c['y']} {dirw} на {rua(abs(b1), 4)}{u(yu)}, независимо от {c['x2']}: {c['x2']} не учитывается",
                      f"При +1 {c['x1u']} к {c['x1']} {c['y']} {dirw} на {rua(abs(b1), 4)} %",
                      f"{c['x1']} объясняет {rua(abs(b1), 4)} разброса {c['y']}"],
                     f"Во множественной регрессии каждый коэффициент — эффект <b>при прочих равных</b>: остальные предикторы ({others}) зафиксированы. "
                     f"Это отличает его от наклона в простой регрессии {c['y']} ~ {c['x1']}, где {c['x2']} «прячется» внутри наклона."
                     f"<br><i>EN:</i> Holding {c['x2']}{' and ' + c['g'] if has_g else ''} constant, each 1 {c['x1u']} increase in {c['x1']} is associated with "
                     f"a {rua(abs(b1), 4).replace(',', '.')} {yu} {'increase' if b1 > 0 else 'decrease'} in mean {c['y']}."))
    gtxt = (f", {c['g']} = {gv}" if has_g else "")
    calc = f"{rua(est[0], 5)} + {rua(b1, 5)} × {gr(x1v)} + {rua(b2, 5)} × {gr(x2v)}"
    if has_g:
        calc += f" + {rua(est[3], 5)} × {1 if gv == lv[1] else 0}"
    qs_req.append(num(f"Прогноз среднего {c['y']} при {c['x1']} = {gr(x1v)}, {c['x2']} = {gr(x2v)}{gtxt}?", yhat,
                      f"Ŷ = {calc} = <b>{rua(yhat, 5)}</b>{u(yu)}. "
                      + (f"Дамми {names[3]} = 1 только для {lv[1]}; для {lv[0]} это слагаемое равно 0." if has_g else ""),
                      rel=0.004, abs_=0.02 * c["sd"], unit=yu))
    delta = float(np.round((c["x1hi"] - c["x1lo"]) * rng.choice([0.1, 0.2, 0.25]), c["x1dec"])) or 1.0
    qs_opt.append(num(f"Есть {TWO[c['data']]}: {others} одинаковые, {c['x1']} различается на {gr(delta)} {c['x1u']}. "
                      f"Насколько различается среднее {c['y']}? (больший {c['x1']} минус меньший, со знаком)",
                      delta * b1,
                      f"Всё одинаковое, кроме {c['x1']}, поэтому остальные слагаемые сокращаются: {gr(delta)} × {rua(b1, 5)} = <b>{rua(delta * b1, 4)}</b>{u(yu)}.",
                      rel=0.01, abs_=1e-6, unit=yu))
    if has_g:
        bg_ = est[3]
        qs_opt.append(num(f"На сколько среднее {c['y']} у {lv[1]} отличается от {lv[0]} при одинаковых {c['x1']} и {c['x2']}? ({lv[1]} − {lv[0]}, со знаком)",
                          bg_,
                          f"Это сам коэффициент {names[3]} = <b>{rua(bg_, 4)}</b>{u(yu)}. {lv[0]} — референт (первый по алфавиту), "
                          f"дамми {names[3]} = 1 для {lv[1]}. «При одинаковых {c['x1']} и {c['x2']}» — потому что они тоже в модели.",
                          rel=0.002, abs_=1e-6, unit=yu))
    k = f["k"]
    qs_opt.append(mc(rng, f"F-statistic внизу: {R.fmt1(f['F'], 4)} on {k - 1} and {f['df']} DF. Какую H₀ проверяет?",
                     "Все коэффициенты при предикторах равны 0 (модель не лучше одного среднего)",
                     ["Все коэффициенты, включая (Intercept), равны 0", f"Коэффициент {c['x1']} равен 0", "Остатки распределены нормально"],
                     f"Общий F-тест: H₀: β₁ = β₂ = …{' = β₃' if has_g else ''} = 0. df числителя = число предикторов = {k - 1}, "
                     f"df знаменателя = n − {k} = {f['df']}. Маленькое p → хотя бы один предиктор полезен. Какой именно — смотрят t-тесты."))
    qs_opt.append(num(f"Сколько {c['eu']} в анализе?", f["n"],
                      f"Residual df = n − (число коэффициентов). Коэффициентов {k} (с Intercept). n = {f['df']} + {k} = <b>{f['n']}</b>.", rel=0))
    qs_opt.append(mc(rng, f"R² = {R.fmt1(f['r2'], 4)}, Adjusted R² = {R.fmt1(f['adj'], 4)}. Что брать, чтобы сравнить модели с разным числом предикторов?",
                     "Adjusted R²: он штрафует за лишние предикторы",
                     ["Multiple R²: он всегда точнее", "Любой: они всегда одинаковы", "Тот, что больше"],
                     "Обычный R² никогда не падает при добавлении переменной, даже если это шум. Adjusted R² = 1 − (1 − R²)(n − 1)/(n − k) "
                     "может упасть, если новый предиктор бесполезен."))
    se1 = se[1]
    if tc is not None:
        up = b1 + tc * se1
        qs_opt.append(num(f"95% ДИ для коэффициента {c['x1']}: верхняя граница (qt в выводе)?", up,
                          f"b + qt × SE = {rua(b1, 5)} + {R.r7(tc)} × {rua(se1, 4)} = <b>{rua(up, 4)}</b>. "
                          f"df для qt — Residual df = {f['df']}, а не n − 2: коэффициентов больше двух.",
                          rel=0.01, abs_=1e-5))
    pv = [f["p"][i] for i in range(1, k)]
    ns_idx = [i for i in range(1, k) if f["p"][i] >= 0.05]
    if len(ns_idx) == 1:
        j = ns_idx[0]
        wr = ([nm for i, nm in enumerate(names) if i >= 1 and i != j] + ["Все предикторы значимы", "(Intercept)"])[:3]
        qs_opt.append(mc(rng, "Какой предиктор незначим на уровне 0,05 (при остальных в модели)?", names[j], wr,
                         f"Смотрим Pr(>|t|) в строках предикторов: у {names[j]} p = {d['p'][j]} ≥ 0,05. "
                         "Это значит «нет доказательств эффекта при прочих равных», а не «эффекта точно нет»."))
    elif len(ns_idx) == 0:
        qs_opt.append(mc(rng, "Какой предиктор незначим на уровне 0,05?", "Все предикторы значимы",
                         names[1:4] if len(names) >= 4 else names[1:3] + ["(Intercept)"],
                         "У всех строк предикторов Pr(>|t|) &lt; 0,05. Строку (Intercept) не проверяют: её p — про «Y = 0 при всех x = 0»."))
    qs_opt.append(mc(rng, f"Residual standard error = {R.fmt1(f['s'], 4)}. Что это?",
                     f"Типичный промах модели для одного наблюдения: ≈ {R.fmt1(f['s'], 4)}{u(yu)}",
                     ["SE коэффициента " + c["x1"], "Доля необъяснённого разброса", "Средняя ошибка прогноза среднего"],
                     f"s = √MSE = √(SSE / {f['df']}). Он про разброс отдельных точек вокруг плоскости модели, в единицах Y."))
    if show_vif:
        v = vif_vals(X[:, 1:])
        vmax = max(v)
        jm = int(np.argmax(v))
        bad = vmax > 4
        qs_opt.append(mc(rng, f"car::vif: максимум {rua(vmax, 3)} у {names[1 + jm]}. Вывод?",
                         ("Заметная коллинеарность: SE раздуты, эффекты трудно разделить" if bad else
                          "Коллинеарность не мешает (VIF меньше 4)"),
                         (["Проблем нет: VIF должен быть больше 10", "Предиктор надо логарифмировать", "Коэффициенты смещены из-за выбросов"] if bad else
                          ["Сильная коллинеарность: надо удалить предиктор", "VIF показывает значимость предиктора", "VIF — это R² модели"]),
                         f"VIF = 1 / (1 − R²ⱼ), R²ⱼ — насколько предиктор предсказывается остальными. SE растёт в √VIF раз: "
                         f"√{rua(vmax, 3)} = {rua(math.sqrt(vmax), 3)}. Пороги курса: &gt; 4 — разобраться, &gt; 10 — серьёзно."))
    if show_pred:
        what = ("ДИ для <b>среднего</b> " + c["y"] + " у всех с такими значениями предикторов") if kind_iv == "confidence" else \
            "интервал прогноза для <b>одного нового</b> наблюдения с такими значениями"
        qs_opt.append(mc(rng, f"predict(…, interval = \"{kind_iv}\"): что показывают lwr и upr?",
                         "ДИ для среднего Y при этих значениях предикторов" if kind_iv == "confidence" else
                         "Интервал прогноза для одного нового наблюдения",
                         (["Интервал прогноза для одного нового наблюдения", f"ДИ для коэффициента {c['x1']}", "Разброс Y в данных"]
                          if kind_iv == "confidence" else
                          ["ДИ для среднего Y при этих значениях предикторов", f"ДИ для коэффициента {c['x1']}", "Разброс Y в данных"]),
                         f"{what.capitalize()}. confidence — про среднее (узкий), prediction — про одного (шире: добавляется разброс s)."))
    qs = pick(rng, qs_req, qs_opt, 5)
    xl = (f"{code(c['x1'])} — {c['x1_ru']}, {c['x1u']}; {code(c['x2'])} — {c['x2_ru']}, {c['x2u']}"
          + (f"; {code(c['g'])} — {c['g_ru']}" if has_g else ""))
    intro = (f"<p>{c['eu'].capitalize()}: n = {f['n']}. Отклик {code(c['y'])} — {c['y_ru']}, {yu}. Предикторы: {xl}.</p>"
             f"<p class='dz'>{design_ru(c)}</p>")
    return dict(cat="lin", sub="mlr", title=f"Множественная: {formula}", intro=intro, console="\n".join(lines), qs=qs)


PLAN2 = [
    ("pg", gen_pois_grp, C.POISG, 60),
    ("px", gen_poisx, C.POISX, 40),
    ("mlr", gen_mlr, C.MLR, 60),
]
