"""Generate 500 R-output examples with questions -> examples.json"""
import json, math, sys
import numpy as np
from scipy import stats
import rfmt as R
from models import ols, predict, oneway, tukey, logistic
import contexts as C

LN2, LN10 = math.log(2), math.log(10)


# ---------- small helpers ----------
def ru(x, d=2):
    s = f"{x:.{d}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace(".", ",").replace("-", "−")


def rua(x, sig=3):
    """Russian number with sensible precision."""
    ax = abs(x)
    if ax == 0:
        return "0"
    d = max(0, sig - 1 - math.floor(math.log10(ax)))
    d = min(d, 6)
    return ru(x, d)


def u(unit):
    return f" {unit}" if unit else ""


def code(s):
    return f"<code>{s}</code>"


def mc(rng, q, correct, wrongs, ex):
    opts = [correct] + list(wrongs)
    order = list(rng.permutation(len(opts)))
    return dict(t="mc", q=q, o=[opts[i] for i in order], a=order.index(0), ex=ex)


def num(q, ans, ex, rel=0.02, abs_=0.0, unit="", absval=False):
    tol = max(abs_, rel * abs(ans))
    d = dict(t="num", q=q, a=float(f"{ans:.6g}"), tol=float(f"{tol:.4g}"), ex=ex, unit=unit)
    if absval:
        d["abs"] = 1
    return d


def pick(rng, required, optional, total=5):
    opt = list(optional)
    rng.shuffle(opt)
    qs = list(required) + opt[: max(0, total - len(required))]
    return [q for q in qs if q is not None][:total]


def lin_x(rng, lo, hi, n, dec, logscale=False):
    if logscale:
        x = np.exp(rng.uniform(math.log(lo), math.log(hi), n))
    else:
        x = rng.uniform(lo, hi, n)
    x = np.round(x, dec)
    if logscale:
        x = np.maximum(x, 10 ** (-dec) if dec else 1)
    return x


def coef_block(names, est, se, stat, pv, statname="t"):
    e, s = R.coef_cols(est, se)
    t = R.tval_col(stat)
    p = R.pval_col(list(pv))
    lines = R.table(names, [("Estimate", e), ("Std. Error", s), (f"{statname} value", t), (f"Pr(>|{statname}|)", p)],
                    [R.stars(v) for v in pv])
    return lines, [float(x) for x in e], [float(x) for x in s], [float(x) for x in t], p


def lm_summary(formula, data, f, names):
    lines = ["> m <- lm(" + formula + ", data = " + data + ")", "> summary(m)", "", "Call:",
             f"lm(formula = {formula}, data = {data})", "", "Residuals:"]
    q = np.quantile(f["res"], [0, .25, .5, .75, 1])
    lines += R.named_vector(["Min", "1Q", "Median", "3Q", "Max"], R.fmt_joint(list(q), 4))
    lines += ["", "Coefficients:"]
    cl, est, se, tv, ps = coef_block(names, f["b"], f["se"], f["t"], f["p"])
    lines += cl
    lines += ["---", R.SIGNIF, ""]
    rse = R.fmt1(f["s"], 4)
    r2, adj = R.fmt1(f["r2"], 4), R.fmt1(f["adj"], 4)
    Fs = R.fmt1(f["F"], 4)
    lines += [f"Residual standard error: {rse} on {f['df']} degrees of freedom",
              f"Multiple R-squared:  {r2},  Adjusted R-squared:  {adj} ",
              f"F-statistic: {Fs} on {f['k'] - 1} and {f['df']} DF,  p-value: {R.pval1(f['pF'])}"]
    disp = dict(est=est, se=se, t=tv, p=ps, rse=float(rse), r2=float(r2), adj=float(adj), F=float(Fs))
    return lines, disp


def qt_line(df, lines):
    tc = stats.t.ppf(0.975, df)
    lines += [f"> qt(0.975, df = {df})", "[1] " + R.r7(tc)]
    return float(R.r7(tc))


def design_ru(c):
    return ("Опыт: условия назначали случайно." if c["design"] == "exp"
            else "Наблюдение: условия никто не назначал, данные собраны как есть.")


def causal_word(c):
    return "вызывает" if c["design"] == "exp" else "связан с"


# =====================================================================
# ANOVA family
# =====================================================================
def make_groups(rng, c, k_use=None):
    levels = c["levels"]
    mult = rng.choice([0.25, 1.0, 1.0, 1.4])
    effects = rng.normal(0, 1, len(levels)) * c["eff"] * mult
    n_equal = rng.random() < 0.6
    nmin, nmax = (6, 12) if c["eu"] in ("станков", "делянок", "гнёзд") else (8, 30)
    base_n = int(rng.integers(nmin, nmax + 1))
    groups = {}
    for l, e in zip(levels, effects):
        n = base_n if n_equal else int(rng.integers(nmin, nmax + 1))
        v = np.round(rng.normal(c["base"] + e, c["sd"], n), c["dec"])
        groups[l] = v
    return groups


def aov_lines(c, ow):
    ssb_s, ssw_s = R.fmt_joint([ow["ssb"], ow["ssw"]], 4)
    msb_s, msw_s = R.fmt_joint([ow["msb"], ow["msw"]], 4)
    F_s = R.fmt1(ow["F"], 4)
    p_s = R.pval_col([ow["p"]])[0]
    rn = [c["g"], "Residuals"]
    lines = R.table(rn, [("Df", [str(ow["dfb"]), str(ow["dfw"])]), ("Sum Sq", [ssb_s, ssw_s]),
                         ("Mean Sq", [msb_s, msw_s]), ("F value", [F_s, ""]), ("Pr(>F)", [p_s, ""])],
                    [R.stars(ow["p"]), ""], rn_width=11)
    return lines, dict(ssb=float(ssb_s), ssw=float(ssw_s), msb=float(msb_s), msw=float(msw_s), F=float(F_s), p=ow["p"])


def means_lines(c, ow):
    lv = ow["levels"]
    mdec = 2 if c["dec"] >= 1 else 1
    if c["dec"] >= 2:
        mdec = 3
    ms = [round(ow["means"][l], mdec) for l in lv]
    ms_s = [f"{m:.{mdec}f}" for m in ms]
    lines = [f"> round(tapply({c['data']}${c['y']}, {c['data']}${c['g']}, mean), {mdec})"]
    lines += R.named_vector(lv, ms_s)
    lines += [f"> table({c['data']}${c['g']})", ""]
    lines += R.named_vector(lv, [str(ow["ns"][l]) for l in lv])
    return lines, {l: m for l, m in zip(lv, ms)}


def intro_anova(c, ow):
    n_txt = ", ".join(f"{l}: {ow['ns'][l]}" for l in ow["levels"])
    return (f"<p>{c['what']}</p><p>Отклик {code(c['y'])} — {c['y_ru']}{(', ' + c['unit']) if c['unit'] else ''}. "
            f"Фактор {code(c['g'])} — {c['g_ru']}. Число {c['eu']} в группах: {n_txt}.</p>"
            f"<p class='dz'>{design_ru(c)}</p>")


def gen_aov(rng, c):
    groups = make_groups(rng, c)
    ow = oneway(groups)
    lv = ow["levels"]
    lines = [f"> fit <- aov({c['y']} ~ {c['g']}, data = {c['data']})", "> summary(fit)"]
    al, d = aov_lines(c, ow)
    lines += al + ["---", R.SIGNIF]
    ml, means = means_lines(c, ow)
    lines += ml
    tc = qt_line(ow["dfw"], lines)
    k, N = ow["k"], ow["N"]
    un = c["unit"]
    sd = math.sqrt(d["msw"])
    A, B = lv[0], lv[1]
    Cc = lv[-1]
    nA, nB, nC = ow["ns"][A], ow["ns"][B], ow["ns"][Cc]
    qs_req, qs_opt = [], []

    qs_opt.append(num(f"Сколько групп ({c['g_ru']}) сравнивали? Найди по таблице, а не по описанию.", k,
                      f"Строка {code(c['g'])}: Df = {ow['dfb']} = k − 1, значит k = {ow['dfb']} + 1 = <b>{k}</b>.", rel=0, abs_=0))
    qs_opt.append(num("Сколько всего наблюдений (N) было в анализе?", N,
                      f"Df фактора + Df остатков = Df total = N − 1: {ow['dfb']} + {ow['dfw']} = {ow['dfb'] + ow['dfw']} → N = <b>{N}</b>.", rel=0))
    qs_req.append(num(f"Чему равна оценка σ — типичный разброс внутри групп (pooled SD)?{(' В ' + un + '.') if un else ''}",
                      sd, f"σ̂ = √MSE. MSE — это Mean Sq в строке Residuals = {rua(d['msw'], 4)}. √{rua(d['msw'], 4)} = <b>{rua(sd)}</b>{u(un)}. "
                          f"Отдельное наблюдение обычно отличается от среднего своей группы примерно на столько.", unit=un))
    qs_opt.append(num("Проверь, как R получил F value: посчитай F сам.", d["msb"] / d["msw"],
                      f"F = Mean Sq фактора / Mean Sq Residuals = {rua(d['msb'], 4)} / {rua(d['msw'], 4)} = <b>{rua(d['msb'] / d['msw'])}</b>. "
                      f"Сигнал (разброс между средними) делим на шум (разброс внутри групп).", rel=0.01))
    sig = d["p"] < 0.05
    p_s = R.pval_col([ow["p"]])[0]
    if sig:
        corr = "Хотя бы одно среднее отличается от остальных; какие именно — F не говорит"
        wr = ["Все средние отличаются друг от друга", "Средние равны", f"{c['g_ru'].capitalize()} не влияет на отклик"]
    else:
        corr = "Нет доказательств, что средние различаются (но и равенство не доказано)"
        wr = ["Доказано, что все средние равны", "Хотя бы одно среднее отличается", "Все средние отличаются друг от друга"]
    qs_req.append(mc(rng, f"p-value = {p_s}. Какой вывод правильный (α = 0,05)?", corr, wr,
                     f"H₀: все средние в популяции равны. p {'&lt;' if sig else '≥'} 0,05 → H₀ {'отвергаем' if sig else 'не отвергаем'}. "
                     f"F-тест отвечает только на вопрос «все ли средние одинаковы», а не «какие различаются»."))
    r2 = d["ssb"] / (d["ssb"] + d["ssw"])
    qs_opt.append(num("Какая доля всего разброса объясняется группами (R²)?", r2,
                      f"R² = SS фактора / (SS фактора + SS остатков) = {rua(d['ssb'], 4)} / ({rua(d['ssb'], 4)} + {rua(d['ssw'], 4)}) = <b>{rua(r2)}</b>.",
                      rel=0.02, abs_=0.005))
    diff = means[B] - means[A]
    se = math.sqrt(d["msw"] * (1 / nA + 1 / nB))
    qs_opt.append(num(f"t для сравнения {B} с {A} (без поправки на много сравнений)?", diff / se,
                      f"Разница = {rua(means[B], 5)} − {rua(means[A], 5)} = {rua(diff)}. "
                      f"SE = √(MSE·(1/n{A} + 1/n{B})) = √({rua(d['msw'], 4)}·(1/{nA} + 1/{nB})) = {rua(se)}. "
                      f"t = {rua(diff)} / {rua(se)} = <b>{rua(diff / se)}</b>, df = {ow['dfw']} (df остатков).", rel=0.03, abs_=0.03))
    up = diff + tc * se
    qs_opt.append(num(f"95% ДИ для разницы средних {Cc if Cc != B else B} − {A}: найди <b>верхнюю</b> границу. qt есть в выводе.",
                      (means[Cc] - means[A]) + tc * math.sqrt(d["msw"] * (1 / nA + 1 / nC)),
                      f"Разница = {rua(means[Cc], 5)} − {rua(means[A], 5)} = {rua(means[Cc] - means[A])}. "
                      f"SE = √({rua(d['msw'], 4)}·(1/{nA} + 1/{nC})) = {rua(math.sqrt(d['msw'] * (1 / nA + 1 / nC)))}. "
                      f"Верх = разница + qt × SE = {rua(means[Cc] - means[A])} + {R.r7(tc)} × {rua(math.sqrt(d['msw'] * (1 / nA + 1 / nC)))} = "
                      f"<b>{rua((means[Cc] - means[A]) + tc * math.sqrt(d['msw'] * (1 / nA + 1 / nC)))}</b>{u(un)}. "
                      f"MSE и df берём из всей ANOVA, даже если сравниваем две группы.", rel=0.02, abs_=0.02 * c["sd"], unit=un))
    qs_opt.append(mc(rng, "Что показывает строка Residuals?",
                     "Разброс внутри групп — шум, который группы не объясняют",
                     ["Разницу между средними групп", "Ошибки ввода данных", "Остатки после удаления выбросов"],
                     "Residuals = Error = Within. Её Mean Sq — это MSE, оценка σ²: насколько наблюдения отличаются от среднего своей группы."))
    if k >= 3:
        others = [l for l in lv if l != A]
        cA = 1.0
        cO = -1.0 / len(others)
        g = means[A] - sum(means[o] for o in others) / len(others)
        sum_c2n = 1 / nA + sum((cO ** 2) / ow["ns"][o] for o in others)
        se_c = math.sqrt(d["msw"] * sum_c2n)
        frac = "½" if len(others) == 2 else "⅓"
        qs_opt.append(num(f"Контраст «{A} против среднего остальных групп», коэффициенты (1, −{frac}, …). Найди SE контраста.",
                          se_c, f"SE = √(MSE × Σc²/n). Σc²/n = 1²/{nA} + " +
                          " + ".join(f"({frac})²/{ow['ns'][o]}" for o in others) + f" = {rua(sum_c2n, 4)}. "
                          f"SE = √({rua(d['msw'], 4)} × {rua(sum_c2n, 4)}) = <b>{rua(se_c)}</b>. Оценка контраста g = {rua(g)}.",
                          rel=0.03))
    qs = pick(rng, qs_req, qs_opt, 5)
    return dict(cat="anova", sub="aov", title=f"ANOVA: {c['y']} ~ {c['g']}", intro=intro_anova(c, ow),
                console="\n".join(lines), qs=qs)


def gen_lmfactor(rng, c):
    groups = make_groups(rng, c)
    ow = oneway(groups)
    lv = ow["levels"]
    y = np.concatenate([groups[l] for l in lv])
    lab = np.concatenate([[l] * len(groups[l]) for l in lv])
    X = np.column_stack([np.ones(len(y))] + [(lab == l).astype(float) for l in lv[1:]])
    f = ols(X, y)
    names = ["(Intercept)"] + [f"{c['g']}{l}" for l in lv[1:]]
    lines, d = lm_summary(f"{c['y']} ~ {c['g']}", c["data"], f, names)
    tc = qt_line(f["df"], lines)
    un = c["unit"]
    ref = lv[0]
    est = d["est"]
    qs_req, qs_opt = [], []
    qs_req.append(mc(rng, "Какая группа референтная (базовая), то есть «спрятана» в (Intercept)?",
                     f"{ref} — её нет среди строк коэффициентов",
                     [f"{lv[-1]}", "Никакая: (Intercept) — общее среднее всех данных", f"{lv[1]}"],
                     f"R кодирует фактор через 0/1. Первая по алфавиту группа ({ref}) становится базой: для неё все "
                     f"«{c['g']}…» = 0, и её среднее = (Intercept). Остальные строки — <b>разницы с {ref}</b>."))
    qs_opt.append(num(f"Чему равно среднее группы {ref}?", est[0],
                      f"Это просто (Intercept) = <b>{rua(est[0], 5)}</b>{u(un)}: у референтной группы все dummy = 0, "
                      f"поэтому Ŷ = b₀.", rel=0.002, abs_=0.01, unit=un))
    j = int(rng.integers(1, len(lv)))
    gj = lv[j]
    qs_req.append(num(f"Чему равно среднее группы {gj}?", est[0] + est[j],
                      f"Среднее {gj} = (Intercept) + {c['g']}{gj} = {rua(est[0], 5)} + ({rua(est[j], 5)}) = <b>{rua(est[0] + est[j], 5)}</b>{u(un)}. "
                      f"Коэффициент {c['g']}{gj} — это не среднее группы, а её отличие от {ref}.", rel=0.002, abs_=0.01, unit=un))
    if len(lv) >= 3:
        a, b = 1, len(lv) - 1
        qs_req.append(num(f"На сколько среднее {lv[b]} отличается от среднего {lv[a]} ({lv[b]} − {lv[a]})? Ни одна из них не референтная.",
                          est[b] - est[a],
                          f"Обе разницы отсчитаны от {ref}, поэтому {ref} сокращается: "
                          f"({ref} + {rua(est[b], 5)}) − ({ref} + {rua(est[a], 5)}) = {rua(est[b], 5)} − ({rua(est[a], 5)}) = "
                          f"<b>{rua(est[b] - est[a], 4)}</b>{u(un)}.", rel=0.005, abs_=0.01, unit=un))
    qs_opt.append(mc(rng, f"Что проверяет p-value в строке {c['g']}{gj}?",
                     f"Равно ли среднее {gj} среднему референтной группы {ref}",
                     [f"Равно ли среднее {gj} нулю", "Различаются ли все группы", f"Отличается ли {gj} от всех остальных групп"],
                     f"Каждая строка — разница «{gj} минус {ref}». H₀: эта разница = 0 в популяции. "
                     f"Про сравнение всех групп сразу говорит F-statistic внизу."))
    qs_opt.append(mc(rng, f"F-statistic внизу вывода ({R.fmt1(f['F'], 4)} на {f['k'] - 1} и {f['df']} DF) — это…",
                     "Тот же F, что в таблице ANOVA: проверяет, равны ли все средние групп",
                     ["Тест только для (Intercept)", f"Тест разницы {gj} и {ref}", "Квадрат t для первого коэффициента"],
                     f"lm с фактором и aov — одна и та же модель. F = MS между / MSE, df = (k − 1, N − k) = ({f['k'] - 1}, {f['df']})."))
    up = est[j] + tc * d["se"][j]
    qs_opt.append(num(f"95% ДИ для разницы {gj} − {ref}: найди верхнюю границу. qt есть в выводе.", up,
                      f"Оценка ± qt × SE = {rua(est[j], 5)} + {R.r7(tc)} × {rua(d['se'][j], 4)} = <b>{rua(up, 4)}</b>{u(un)}. "
                      f"Оценку и SE берём из строки {c['g']}{gj}.", rel=0.01, abs_=0.01, unit=un))
    qs_opt.append(num("Сколько наблюдений в анализе?", f["n"],
                      f"Residual standard error … on {f['df']} degrees of freedom: df = N − k = N − {f['k']}. N = {f['df']} + {f['k']} = <b>{f['n']}</b>.", rel=0))
    qs_opt.append(mc(rng, f"Residual standard error = {R.fmt1(f['s'], 4)}. Что это?",
                     "Pooled SD: типичное отклонение наблюдения от среднего своей группы (√MSE)",
                     ["SE разницы между группами", "Стандартное отклонение средних групп", "Ошибка (Intercept)"],
                     f"RSE = √MSE — то же σ̂, что в ANOVA. Здесь ≈ {R.fmt1(f['s'], 4)}{u(un)}."))
    qs = pick(rng, qs_req, qs_opt, 5)
    return dict(cat="anova", sub="lmfac", title=f"lm с фактором: {c['y']} ~ {c['g']}", intro=intro_anova(c, ow),
                console="\n".join(lines), qs=qs)


def gen_tukey(rng, c):
    groups = make_groups(rng, c)
    ow = oneway(groups)
    lv = ow["levels"]
    lines = [f"> fit <- aov({c['y']} ~ {c['g']}, data = {c['data']})"]
    ml, means = means_lines(c, ow)
    lines += ml
    rows = tukey(ow)
    lines += ["> TukeyHSD(fit)", "  Tukey multiple comparisons of means", "    95% family-wise confidence level", "",
              f"Fit: aov(formula = {c['y']} ~ {c['g']}, data = {c['data']})", "", f"${c['g']}"]
    mat = R.matrix([r["name"] for r in rows], ["diff", "lwr", "upr", "p adj"],
                   [[r["diff"] for r in rows], [r["lwr"] for r in rows], [r["upr"] for r in rows], [r["padj"] for r in rows]],
                   fixed_dec={3: 7})
    lines += mat
    dif = [float(l.split()[1]) for l in mat[1:]]
    lw = [float(l.split()[2]) for l in mat[1:]]
    upv = [float(l.split()[3]) for l in mat[1:]]
    pa = [float(l.split()[4]) for l in mat[1:]]
    un = c["unit"]
    k = len(lv)
    npairs = k * (k - 1) // 2
    nsig = sum(p < 0.05 for p in pa)
    qs_req, qs_opt = [], []
    qs_req.append(num("Сколько пар различаются значимо (p adj &lt; 0,05)?", nsig,
                      "Смотрим столбец p adj и считаем числа меньше 0,05: " +
                      ", ".join(f"{r['name']} {'✓' if p < 0.05 else '✗'}" for r, p in zip(rows, pa)) + f". Итого <b>{nsig}</b>.", rel=0))
    i = int(rng.integers(0, len(rows)))
    r = rows[i]
    hi_ = r["hi"] if dif[i] > 0 else r["lo"]
    lo_ = r["lo"] if dif[i] > 0 else r["hi"]
    qs_req.append(mc(rng, f"Строка {r['name']}: diff = {R.fmt_joint(dif, 7)[i]}. Что это значит?",
                     f"Среднее {r['hi']} минус среднее {r['lo']}; у {hi_} среднее выше на {rua(abs(dif[i]))}{u(un)}",
                     [f"Среднее {r['lo']} минус среднее {r['hi']}", f"Разница медиан {r['hi']} и {r['lo']}",
                      f"{r['hi']} выше {r['lo']} в {rua(abs(dif[i]))} раз"],
                     f"В TukeyHSD имя «{r['name']}» читается как «{r['hi']} минус {r['lo']}». Проверка по средним: "
                     f"{rua(means[r['hi']], 5)} − {rua(means[r['lo']], 5)} = {rua(means[r['hi']] - means[r['lo']], 4)}."))
    ns = [j for j, p in enumerate(pa) if p >= 0.05]
    if ns:
        j = ns[int(rng.integers(0, len(ns)))]
        qs_opt.append(mc(rng, f"Пара {rows[j]['name']}: ДИ от {R.fmt_joint(lw, 7)[j]} до {R.fmt_joint(upv, 7)[j]}. Вывод?",
                         "ДИ содержит 0 → нет доказательств различия этих двух групп",
                         ["Группы различаются, потому что diff не равен 0", "Группы точно одинаковые",
                          "Вывод нельзя сделать без p-value обычного t-теста"],
                         f"Ноль внутри интервала ⇔ p adj ≥ 0,05 (здесь {pa[j]:.4f}). «Нет доказательств» ≠ «доказано, что равны»."))
    sg = [j for j, p in enumerate(pa) if p < 0.05]
    if sg:
        j = sg[int(rng.integers(0, len(sg)))]
        wrong = [rows[x]["name"] for x in range(len(rows)) if x != j and pa[x] >= 0.05][:3]
        if len(wrong) >= 2:
            qs_opt.append(mc(rng, "Какая из этих пар различается значимо?", rows[j]["name"], wrong,
                             f"У {rows[j]['name']} p adj = {pa[j]:.7f} &lt; 0,05 и ДИ ({R.fmt_joint(lw, 7)[j]}; {R.fmt_joint(upv, 7)[j]}) не содержит 0."))
    qs_opt.append(num("Сколько попарных сравнений сделал TukeyHSD?", npairs,
                      f"k = {k} групп → k(k − 1)/2 = {k}·{k - 1}/2 = <b>{npairs}</b> пар (столько строк в выводе).", rel=0))
    qs_opt.append(mc(rng, "Почему p adj больше, чем p обычного t-теста для той же пары?",
                     f"Тьюки поправляет на {npairs} сравнений сразу: держит риск хотя бы одной ложной находки на всём наборе ≤ 5 %",
                     ["Потому что Тьюки использует медианы вместо средних", "Потому что SE берётся только из двух групп",
                      "Это ошибка округления"],
                     "Чем больше пар, тем выше шанс случайной «значимости». Поправка поднимает планку (множитель Тьюки вместо t*), "
                     "поэтому скорректированные p больше, а ДИ шире."))
    j = int(rng.integers(0, len(rows)))
    qs_opt.append(num(f"Проверь diff для {rows[j]['name']} по таблице средних.", means[rows[j]["hi"]] - means[rows[j]["lo"]],
                      f"diff = среднее {rows[j]['hi']} − среднее {rows[j]['lo']} = {rua(means[rows[j]['hi']], 5)} − {rua(means[rows[j]['lo']], 5)} = "
                      f"<b>{rua(means[rows[j]['hi']] - means[rows[j]['lo']], 4)}</b>{u(un)}.", rel=0.01, abs_=0.02 * c["sd"], unit=un))
    qs = pick(rng, qs_req, qs_opt, 5)
    return dict(cat="anova", sub="tukey", title=f"TukeyHSD: {c['y']} ~ {c['g']}", intro=intro_anova(c, ow),
                console="\n".join(lines), qs=qs)


# =====================================================================
# Linear regression family
# =====================================================================
def sim_lin(rng, c, kind):
    n = int(rng.integers(c["n"][0], c["n"][1] + 1))
    logx = kind in ("logx", "loglog")
    x = lin_x(rng, max(c["xlo"], 1e-9) if logx else c["xlo"], c["xhi"], n, c["xdec"], logscale=logx)
    f = rng.choice([0.15, 0.7, 1.0, 1.0, 1.3]) if rng.random() < 0.9 else 1.0
    b1 = c["b1"] * f
    xx = np.log(x) if logx else x
    mean_t = c["b0"] + b1 * xx
    if kind == "lin" or kind == "logx":
        # keep mean of Y similar when slope is shrunk
        mean_t = mean_t + (c["b1"] - b1) * xx.mean()
        y = np.round(mean_t + rng.normal(0, c["sd"], n), c["ydec"])
        yy = y
    else:
        mean_t = mean_t + (c["b1"] - b1) * xx.mean()
        y = np.round(np.exp(mean_t + rng.normal(0, c["sd"], n)), c["ydec"])
        y = np.maximum(y, 10 ** (-c["ydec"]) if c["ydec"] else 1)
        yy = np.log(y)
    return x, y, xx, yy


def intro_lin(c, kind, n, x):
    yl = f"{code(c['y'])} — {c['y_ru']}{(', ' + c['yu']) if c['yu'] else ''}"
    xl = f"{code(c['x'])} — {c['x_ru']}, {c['xu']}"
    form = {"lin": "", "logy": f"<p>Y сильно растёт или падает «в разах», поэтому в модель взяли {code('log(' + c['y'] + ')')}.</p>",
            "logx": f"<p>X растянут в разы, эффект затухает, поэтому в модель взяли {code('log(' + c['x'] + ')')}.</p>",
            "loglog": f"<p>Обе величины меняются в разах, поэтому логарифмировали обе: {code('log(' + c['y'] + ')')} и {code('log(' + c['x'] + ')')}.</p>"}[kind]
    xmin, xmax = x.min(), x.max()
    return (f"<p>{c['eu'].capitalize()}: n = {n}. Отклик {yl}. Предиктор {xl}; в данных от {rua(xmin, 3)} до {rua(xmax, 3)}.</p>"
            f"{form}<p class='dz'>{design_ru(c)} R: {code('log()')} — натуральный логарифм.</p>")


def slope_word(b, up="выше", down="ниже"):
    return up if b > 0 else down


def gen_lin(rng, c):
    x, y, xx, yy = sim_lin(rng, c, "lin")
    f = ols(np.c_[np.ones(len(x)), x], y)
    formula = f"{c['y']} ~ {c['x']}"
    lines, d = lm_summary(formula, c["data"], f, ["(Intercept)", c["x"]])
    b0, b1 = d["est"]
    se1 = d["se"][1]
    yu, xu = c["yu"], c["xu"]
    show_ci = rng.random() < 0.5
    show_pred = rng.random() < 0.45
    lo, hi = x.min(), x.max()
    x0 = float(np.round(rng.uniform(lo + 0.2 * (hi - lo), hi - 0.2 * (hi - lo)), c["xdec"]))
    if show_ci:
        lines += ["> confint(m)"]
        ci = [[b - stats.t.ppf(.975, f["df"]) * s for b, s in zip(f["b"], f["se"])],
              [b + stats.t.ppf(.975, f["df"]) * s for b, s in zip(f["b"], f["se"])]]
        lines += R.matrix(["(Intercept)", c["x"]], ["2.5 %", "97.5 %"], ci)
        ci_s = R.matrix(["(Intercept)", c["x"]], ["2.5 %", "97.5 %"], ci)
        ci_lo, ci_hi = float(ci_s[2].split()[1]), float(ci_s[2].split()[2])
    else:
        tc = qt_line(f["df"], lines)
    if show_pred:
        pr = predict(f, [1, x0])
        kind_iv = rng.choice(["confidence", "prediction"])
        lines += [f"> predict(m, newdata = data.frame({c['x']} = {x0:g}), interval = \"{kind_iv}\")"]
        iv = pr["ci"] if kind_iv == "confidence" else pr["pi"]
        lines += R.matrix(["1"], ["fit", "lwr", "upr"], [[pr["fit"]], [iv[0]], [iv[1]]])
    qs_req, qs_opt = [], []
    dirw = slope_word(b1)
    wrongs = [f"При {c['x']} = 1 значение {c['y']} равно {rua(b1, 4)}{u(yu)}",
              f"При +1 к {c['x']} значение {c['y']} меняется на {rua(b1, 4)} %",
              f"При +1 к {c['x']} у каждого объекта {c['y']} меняется ровно на {rua(b1, 4)}{u(yu)}"]
    qs_req.append(mc(rng, f"Как читать наклон {c['x']} = {rua(b1, 5)}?",
                     f"При +1 к {c['x']} ({xu}) среднее {c['y']} {dirw} на {rua(abs(b1), 4)}{u(yu)}",
                     wrongs,
                     f"Наклон b₁ — на сколько единиц Y меняется <b>среднее</b> Y, если X больше на 1 единицу. "
                     f"Знак {'плюс' if b1 > 0 else 'минус'} → среднее {dirw}. Не «у каждого объекта» (точки разбросаны вокруг прямой), не проценты (логарифмов нет). "
                     f"{'Опыт: можно говорить «увеличение X вызывает».' if c['design'] == 'exp' else 'Наблюдение: говорим «связано с», не «вызывает».'}"
                     f"<br><i>EN:</i> Each 1 {xu} increase in {c['x']} is associated with a {rua(abs(b1), 4).replace(',', '.')} {yu} "
                     f"{'increase' if b1 > 0 else 'decrease'} in mean {c['y']}."))
    step = float(np.round((hi - lo) * rng.choice([0.2, 0.3, 0.5]), c["xdec"])) or 1.0
    xa = float(np.round(rng.uniform(lo, hi - step), c["xdec"]))
    xb = round(xa + step, c["xdec"])
    qs_req.append(num(f"Насколько изменится среднее {c['y']}, если {c['x']} увеличить с {ru(xa, c['xdec'])} до {ru(xb, c['xdec'])} {xu}? (со знаком)",
                      (xb - xa) * b1,
                      f"Разница по X = {ru(xb - xa, c['xdec'])} {xu}. Каждая единица X даёт {rua(b1, 5)}{u(yu)}. "
                      f"Итого {ru(xb - xa, c['xdec'])} × {rua(b1, 5)} = <b>{rua((xb - xa) * b1, 4)}</b>{u(yu)}. "
                      f"Сдвиг b₀ не нужен — он сокращается.", rel=0.01, abs_=1e-6, unit=yu))
    qs_req.append(num(f"Прогноз среднего {c['y']} при {c['x']} = {x0:g} {xu}?", b0 + b1 * x0,
                      f"Ŷ = b₀ + b₁·X = {rua(b0, 5)} + {rua(b1, 5)} × {x0:g} = <b>{rua(b0 + b1 * x0, 5)}</b>{u(yu)}. Числа из столбца Estimate.",
                      rel=0.005, abs_=0.01 * c["sd"], unit=yu))
    qs_opt.append(num("Сколько наблюдений в данных?", f["n"],
                      f"«on {f['df']} degrees of freedom»: df = n − 2 → n = {f['df']} + 2 = <b>{f['n']}</b>.", rel=0))
    qs_opt.append(mc(rng, f"Multiple R-squared = {R.fmt1(f['r2'], 4)}. Что это значит?",
                     f"{rua(100 * d['r2'], 3)} % разброса {c['y']} объясняется линейной связью с {c['x']}",
                     [f"Прогноз верен в {rua(100 * d['r2'], 3)} % случаев", f"{rua(100 * d['r2'], 3)} % точек лежат на прямой",
                      f"{c['x']} увеличивает {c['y']} на {rua(100 * d['r2'], 3)} %"],
                     "R² = SSR/SST: доля разброса Y, которую убрала прямая. Ничего не говорит о причинности и о точности одного прогноза."))
    qs_opt.append(mc(rng, f"Residual standard error = {R.fmt1(f['s'], 4)}. Что это?",
                     f"Типичное отклонение отдельной точки от прямой: ≈ {R.fmt1(f['s'], 4)}{u(yu)}",
                     ["SE наклона", "Средняя ошибка (Intercept)", "Насколько наклон может отличаться от истинного"],
                     f"s = √MSE, df = n − 2 = {f['df']}. Не путать с Std. Error наклона ({rua(se1, 4)}): та про то, насколько «дрожит» сам наклон."))
    if show_ci:
        qs_opt.append(mc(rng, f"confint: для {c['x']} ДИ от {rua(ci_lo, 4)} до {rua(ci_hi, 4)}. Как сказать?",
                         f"С уверенностью 95 % +1 к {c['x']} связано с изменением среднего {c['y']} на величину от {rua(ci_lo, 4)} до {rua(ci_hi, 4)}{u(yu)}",
                         [f"95 % объектов имеют наклон в этом интервале", f"Если повторить опыт 100 раз, наклон всегда будет в этом интервале",
                          f"95 % значений {c['y']} лежат между {rua(ci_lo, 4)} и {rua(ci_hi, 4)}"],
                         f"ДИ — про <b>средний</b> эффект (β₁). {'0 внутри → нет доказательств связи.' if ci_lo < 0 < ci_hi else '0 снаружи → связь значима (p &lt; 0,05).'}"))
    else:
        up = b1 + tc * se1
        qs_opt.append(num(f"95% ДИ для наклона: найди верхнюю границу (qt в выводе).", up,
                          f"b₁ + qt × SE = {rua(b1, 5)} + {R.r7(tc)} × {rua(se1, 4)} = <b>{rua(up, 4)}</b>. "
                          f"Нижняя: {rua(b1 - tc * se1, 4)}. {'0 внутри — не значимо.' if (b1 - tc * se1) < 0 < up else '0 снаружи — значимо.'}",
                          rel=0.01, abs_=1e-4))
    qs_opt.append(num(f"Проверь t value для {c['x']}: посчитай сам.", b1 / se1,
                      f"t = Estimate / Std. Error = {rua(b1, 5)} / {rua(se1, 4)} = <b>{rua(b1 / se1, 4)}</b>. "
                      f"А F внизу = t² = {rua((b1 / se1) ** 2, 4)}.", rel=0.01))
    inside = lo <= 0 <= hi
    qs_opt.append(mc(rng, f"Как понимать (Intercept) = {rua(b0, 5)}?",
                     (f"Средний {c['y']} при {c['x']} = 0; 0 внутри данных, число имеет смысл" if inside else
                      f"Средний {c['y']} при {c['x']} = 0; но 0 далеко за пределами данных ({rua(lo, 3)}–{rua(hi, 3)}), смысла нет — это экстраполяция"),
                     ([f"Средний {c['y']} по всем данным", "Наклон прямой в начале координат", f"Минимальный {c['y']} в данных"]),
                     "Сдвиг b₀ — значение прямой при X = 0. Читать его по смыслу можно, только если X = 0 есть в данных."))
    p1 = d["p"][1]
    sig = f["p"][1] < 0.05
    qs_opt.append(mc(rng, f"Строка {c['x']}: p = {p1}. Вывод для H₀: β₁ = 0?",
                     ("Отвергаем H₀: есть доказательства линейной связи" if sig else "Не отвергаем H₀: нет доказательств линейной связи"),
                     (["Не отвергаем H₀: связи нет", "Доказано, что X вызывает изменение Y", "Отвергаем H₀: наклон равен 0"] if sig else
                      ["Отвергаем H₀: связь есть", "Доказано, что наклон равен 0", "Наклон значим, потому что он не равен 0"]),
                     f"p {'&lt;' if sig else '≥'} 0,05. Гипотезы пишут через β (популяция), не через b. "
                     f"{'Направление: ' + ('положительная' if b1 > 0 else 'отрицательная') + ' связь.' if sig else 'Большое p ≠ «связи нет»: просто не хватает доказательств.'}"))
    if show_pred:
        what = ("95 % ДИ для <b>среднего</b> " + c["y"] + f" у всех объектов с {c['x']} = {x0:g}") if kind_iv == "confidence" else \
            ("95 % интервал прогноза для <b>одного нового</b> объекта с " + f"{c['x']} = {x0:g}")
        qs_opt.append(mc(rng, f"Что показывают lwr и upr в predict(…, interval = \"{kind_iv}\")?",
                         "ДИ для среднего Y при этом X" if kind_iv == "confidence" else "Интервал прогноза для одного нового наблюдения",
                         (["Интервал прогноза для одного нового наблюдения", "ДИ для наклона", "Разброс X в данных"] if kind_iv == "confidence" else
                          ["ДИ для среднего Y при этом X", "ДИ для наклона", "Разброс X в данных"]),
                         f"{what}. confidence — про среднее (узкий), prediction — про один объект (шире: добавляется разброс s)."))
    qs = pick(rng, qs_req, qs_opt, 5)
    return dict(cat="lin", sub="lin", title=f"Регрессия: {formula}", intro=intro_lin(c, "lin", f["n"], x),
                console="\n".join(lines), qs=qs)


def gen_logy(rng, c):
    x, y, xx, yy = sim_lin(rng, c, "logy")
    f = ols(np.c_[np.ones(len(x)), x], yy)
    formula = f"log({c['y']}) ~ {c['x']}"
    lines, d = lm_summary(formula, c["data"], f, ["(Intercept)", c["x"]])
    lines += ["> confint(m)"]
    tcr = stats.t.ppf(.975, f["df"])
    ci = [[b - tcr * s for b, s in zip(f["b"], f["se"])], [b + tcr * s for b, s in zip(f["b"], f["se"])]]
    cis = R.matrix(["(Intercept)", c["x"]], ["2.5 %", "97.5 %"], ci)
    lines += cis
    ci_lo, ci_hi = float(cis[2].split()[1]), float(cis[2].split()[2])
    b0, b1 = d["est"]
    yu, xu = c["yu"], c["xu"]
    m1 = math.exp(b1)
    pct = (m1 - 1) * 100
    lo, hi = x.min(), x.max()
    x0 = float(np.round(rng.uniform(lo + 0.2 * (hi - lo), hi - 0.2 * (hi - lo)), c["xdec"]))
    qs_req, qs_opt = [], []
    qs_req.append(num(f"Во сколько раз меняется медиана {c['y']} при +1 к {c['x']} ({xu})?", m1,
                      f"Y под логарифмом → берём e от наклона: e<sup>{rua(b1, 5)}</sup> = <b>{rua(m1, 4)}</b>. "
                      f"{'Больше 1 — рост.' if m1 > 1 else 'Меньше 1 — падение.'} На калькуляторе для минуса: 1 / e<sup>{rua(abs(b1), 5)}</sup>.",
                      rel=0.003, abs_=1e-4))
    qs_opt.append(num(f"На сколько процентов меняется медиана {c['y']} при +1 к {c['x']}? Со знаком: рост +, падение −.", pct,
                      f"(e<sup>b₁</sup> − 1) × 100 = ({rua(m1, 4)} − 1) × 100 = <b>{rua(pct, 3)} %</b>. "
                      f"Прикидка 100·b₁ = {rua(100 * b1, 3)} % работает только при маленьком b₁.", rel=0.02, abs_=0.05, unit="%"))
    step = float(np.round((hi - lo) * rng.choice([0.25, 0.5]), c["xdec"])) or 1.0
    qs_req.append(num(f"Во сколько раз изменится медиана {c['y']}, если {c['x']} вырастет на {step:g} {xu}?", math.exp(step * b1),
                      f"Не {step:g} × процент, а e<sup>{step:g} × b₁</sup> = e<sup>{rua(step * b1, 4)}</sup> = <b>{rua(math.exp(step * b1), 4)}</b>. "
                      f"Проценты перемножаются: {rua(m1, 4)}<sup>{step:g}</sup> = то же самое.", rel=0.01))
    qs_opt.append(num(f"Медиана {c['y']} при {c['x']} = {x0:g} {xu}?", math.exp(b0 + b1 * x0),
                      f"Сначала на лог-шкале: {rua(b0, 5)} + {rua(b1, 5)} × {x0:g} = {rua(b0 + b1 * x0, 4)}. "
                      f"Потом e: e<sup>{rua(b0 + b1 * x0, 4)}</sup> = <b>{rua(math.exp(b0 + b1 * x0), 4)}</b>{u(yu)} — это медиана, не среднее.",
                      rel=0.01, unit=yu))
    wr = [f"При +1 к {c['x']} медиана {c['y']} меняется на {rua(b1, 4)}{u(yu)}",
          f"При +1 к {c['x']} <b>среднее</b> {c['y']} умножается на {rua(m1, 4)}",
          f"При +1 к {c['x']} медиана {c['y']} умножается на {rua(b1, 4)}"]
    qs_req.append(mc(rng, f"Как правильно прочитать наклон {rua(b1, 5)} на исходной шкале?",
                     f"При +1 к {c['x']} ({xu}) медиана {c['y']} умножается на {rua(m1, 4)} ({'+' if pct > 0 else '−'}{rua(abs(pct), 3)} %)",
                     wr,
                     f"Y под log → e<sup>b₁</sup>, ответ «в разах / на %», слово <b>медиана</b> (e от среднего логарифмов = медиана). "
                     f"<br><i>EN:</i> Each 1 {xu} increase in {c['x']} is associated with median {c['y']} being multiplied by {rua(m1, 4).replace(',', '.')}."))
    qs_opt.append(num(f"ДИ из confint для {c['x']}: ({rua(ci_lo, 4)}; {rua(ci_hi, 4)}). Переведи <b>верхний</b> конец в множитель.",
                      math.exp(ci_hi),
                      f"Концы ДИ переводим так же, как наклон: e<sup>{rua(ci_hi, 4)}</sup> = <b>{rua(math.exp(ci_hi), 4)}</b>; нижний: e<sup>{rua(ci_lo, 4)}</sup> = {rua(math.exp(ci_lo), 4)}.",
                      rel=0.005))
    qs_opt.append(mc(rng, "Какое число из вывода НЕ переводят через e?", f"Multiple R-squared ({R.fmt1(f['r2'], 4)})",
                     [f"Наклон {c['x']}", "Концы confint для наклона", "(Intercept)"],
                     "R², t, p, F — доли и тесты, единиц у них нет. Наклон, ДИ, сдвиг переводим через e."))
    qs_opt.append(mc(rng, f"Нужно ли брать e от наклона в модели {formula}?",
                     "Да: Y под логарифмом, чтобы вернуть Y в обычные единицы",
                     ["Нет: X без логарифма", "Только если p &lt; 0,05", "Нет: e берут только в логистической регрессии"],
                     "Правило: Y под log → e от коэффициентов. X под log → говорим про «X в 2 раза». Здесь log только у Y."))
    qs_opt.append(num(f"e от (Intercept): что за число? Посчитай.", math.exp(b0),
                      f"e<sup>{rua(b0, 5)}</sup> = <b>{rua(math.exp(b0), 4)}</b>{u(yu)} — медиана {c['y']} при {c['x']} = 0. "
                      f"{'0 в данных есть — число имеет смысл.' if lo <= 0 <= hi else 'X = 0 вне данных — это экстраполяция.'}", rel=0.01, unit=yu))
    qs = pick(rng, qs_req, qs_opt, 5)
    return dict(cat="lin", sub="logy", title=f"log(Y): {formula}", intro=intro_lin(c, "logy", f["n"], x),
                console="\n".join(lines), qs=qs)


def gen_logx(rng, c):
    x, y, xx, yy = sim_lin(rng, c, "logx")
    f = ols(np.c_[np.ones(len(x)), xx], y)
    formula = f"{c['y']} ~ log({c['x']})"
    lines, d = lm_summary(formula, c["data"], f, ["(Intercept)", f"log({c['x']})"])
    b0, b1 = d["est"]
    yu, xu = c["yu"], c["xu"]
    lo, hi = x.min(), x.max()
    x0 = float(np.round(math.exp(rng.uniform(math.log(lo) + 0.2, math.log(hi) - 0.2)), c["xdec"]))
    qs_req, qs_opt = [], []
    qs_req.append(num(f"Насколько меняется среднее {c['y']}, если {c['x']} увеличить в 2 раза? (со знаком)", b1 * LN2,
                      f"X под log → удвоение X = +ln 2 = +0,693 на шкале log(X). Y меняется на 0,693 × b₁ = 0,693 × {rua(b1, 5)} = "
                      f"<b>{rua(b1 * LN2, 4)}</b>{u(yu)}. e не берём: Y без логарифма.", rel=0.01, abs_=1e-4, unit=yu))
    qs_req.append(num(f"Прогноз среднего {c['y']} при {c['x']} = {x0:g} {xu}?", b0 + b1 * math.log(x0),
                      f"Модель видит ln X, поэтому подставляем ln {x0:g} = {rua(math.log(x0), 4)}, а не {x0:g}: "
                      f"{rua(b0, 5)} + {rua(b1, 5)} × {rua(math.log(x0), 4)} = <b>{rua(b0 + b1 * math.log(x0), 4)}</b>{u(yu)}.",
                      rel=0.005, abs_=0.01 * c["sd"], unit=yu))
    qs_req.append(mc(rng, f"Нужно ли брать e от наклона {rua(b1, 5)}?",
                     "Нет: Y без логарифма, наклон уже в единицах Y; X под log → говорим про удвоение X",
                     ["Да: в модели есть log", "Да, и потом умножить на 2", "Только для (Intercept)"],
                     "e нужен, только когда под логарифмом Y. Здесь log у X, значит: «X × 2 → Y + 0,693·b₁»."))
    qs_opt.append(num(f"Насколько меняется среднее {c['y']}, если {c['x']} увеличить в 10 раз?", b1 * LN10,
                      f"×10 по X = +ln 10 = +2,303 на шкале log(X). 2,303 × {rua(b1, 5)} = <b>{rua(b1 * LN10, 4)}</b>{u(yu)}.", rel=0.01, unit=yu))
    qs_opt.append(mc(rng, f"Как прочитать наклон {rua(b1, 5)}?",
                     f"При удвоении {c['x']} среднее {c['y']} меняется на {rua(b1 * LN2, 4)}{u(yu)}",
                     [f"При +1 к {c['x']} ({xu}) {c['y']} меняется на {rua(b1, 4)}{u(yu)}",
                      f"При +1 к {c['x']} {c['y']} умножается на e<sup>{rua(b1, 3)}</sup>",
                      f"При удвоении {c['x']} {c['y']} умножается на 2<sup>{rua(b1, 3)}</sup>"],
                     f"«+1 к log X» = «X × 2,718», неудобно, поэтому берут удвоение: 0,693 × b₁. «+1 {xu}» здесь не постоянен: "
                     f"от {rua(lo, 2)} до {rua(lo + 1, 2)} прирост больше, чем от {rua(hi - 1, 2)} до {rua(hi, 2)}."))
    xa = float(np.round(lo * 1.2, c["xdec"])) or lo
    qs_opt.append(num(f"Разница средних прогнозов при {c['x']} = {xa:g} и {c['x']} = {2 * xa:g} {xu}?", b1 * (math.log(2 * xa) - math.log(xa)),
                      f"Это удвоение: {rua(b1, 5)} × (ln {2 * xa:g} − ln {xa:g}) = {rua(b1, 5)} × 0,693 = <b>{rua(b1 * LN2, 4)}</b>{u(yu)}.", rel=0.01, unit=yu))
    qs_opt.append(mc(rng, f"Почему на графике в обычных единицах {c['x']} линия модели кривая?",
                     "Прямая по log(X); на обычной оси каждое удвоение даёт тот же прирост, а удвоения всё длиннее — линия выполаживается",
                     ["Потому что модель нелинейная", "Из-за выбросов", "Потому что R² меньше 1"],
                     "«Линейная» — по параметрам β₀, β₁. Прямая по ln X на шкале X выглядит кривой."))
    inside = lo <= 1 <= hi
    qs_opt.append(mc(rng, f"Что означает (Intercept) = {rua(b0, 5)}?",
                     f"Средний {c['y']} при {c['x']} = 1 (потому что ln 1 = 0)" + ("" if inside else " — вне данных"),
                     [f"Средний {c['y']} при {c['x']} = 0", f"Средний {c['y']} по всем данным", "Медиана Y при X = 0"],
                     "Подставь log(X) = 0 → X = e⁰ = 1. Поэтому сдвиг — прогноз при X = 1, а не при 0 (ln 0 не существует)."))
    qs = pick(rng, qs_req, qs_opt, 5)
    return dict(cat="lin", sub="logx", title=f"log(X): {formula}", intro=intro_lin(c, "logx", f["n"], x),
                console="\n".join(lines), qs=qs)


def gen_loglog(rng, c):
    x, y, xx, yy = sim_lin(rng, c, "loglog")
    f = ols(np.c_[np.ones(len(x)), xx], yy)
    formula = f"log({c['y']}) ~ log({c['x']})"
    lines, d = lm_summary(formula, c["data"], f, ["(Intercept)", f"log({c['x']})"])
    b0, b1 = d["est"]
    yu, xu = c["yu"], c["xu"]
    lo, hi = x.min(), x.max()
    x0 = float(np.round(math.exp(rng.uniform(math.log(lo) + 0.2, math.log(hi) - 0.2)), c["xdec"]))
    qs_req, qs_opt = [], []
    m2 = 2 ** b1
    qs_req.append(num(f"Во сколько раз меняется медиана {c['y']}, если {c['x']} увеличить в 2 раза?", m2,
                      f"Оба под log: удвоение X → +0,693 на log(X) → +0,693·b₁ на log(Y) → медиана Y × e<sup>0,693·b₁</sup> = 2<sup>b₁</sup> = "
                      f"2<sup>{rua(b1, 4)}</sup> = <b>{rua(m2, 4)}</b>.", rel=0.005))
    qs_req.append(mc(rng, f"Как прочитать наклон {rua(b1, 4)}?",
                     f"+1 % к {c['x']} связан примерно с {'+' if b1 > 0 else '−'}{rua(abs(b1), 3)} % к медиане {c['y']}",
                     [f"+1 {xu} к {c['x']} → {c['y']} + {rua(b1, 3)}{u(yu)}", f"+1 {xu} к {c['x']} → медиана {c['y']} × e<sup>{rua(b1, 3)}</sup>",
                      f"+1 % к {c['x']} → {c['y']} + {rua(b1, 3)}{u(yu)}"],
                     "В log-log наклон — «эластичность»: процент к проценту. Для удвоения точнее: 2<sup>b₁</sup>."))
    qs_req.append(num(f"Медиана {c['y']} при {c['x']} = {x0:g} {xu}?", math.exp(b0 + b1 * math.log(x0)),
                      f"ln {x0:g} = {rua(math.log(x0), 4)}. log(Y) = {rua(b0, 5)} + {rua(b1, 5)} × {rua(math.log(x0), 4)} = {rua(b0 + b1 * math.log(x0), 4)}. "
                      f"e<sup>{rua(b0 + b1 * math.log(x0), 4)}</sup> = <b>{rua(math.exp(b0 + b1 * math.log(x0)), 4)}</b>{u(yu)}.", rel=0.01, unit=yu))
    qs_opt.append(mc(rng, "Что делаем с X и с Ŷ в этой модели?",
                     "X: берём ln перед подстановкой; Ŷ: берём e в конце (получаем медиану)",
                     ["X: ничего; Ŷ: берём e", "X: берём ln; Ŷ: ничего", "Ничего: логарифмы сокращаются"],
                     "Два вопроса: Y под log? да → e. X под log? да → ln X при подстановке, «X × 2» в трактовке."))
    qs_opt.append(num(f"Во сколько раз меняется медиана {c['y']}, если {c['x']} увеличить в 10 раз?", 10 ** b1,
                      f"10<sup>b₁</sup> = 10<sup>{rua(b1, 4)}</sup> = <b>{rua(10 ** b1, 4)}</b>.", rel=0.01))
    m15 = 1.5 ** b1
    qs_opt.append(num(f"{c['x'].capitalize()} больше на 50 % (× 1,5). Во сколько раз больше медиана {c['y']}?", m15,
                      f"1,5<sup>b₁</sup> = 1,5<sup>{rua(b1, 4)}</sup> = <b>{rua(m15, 4)}</b>.", rel=0.01))
    qs_opt.append(mc(rng, f"Multiple R-squared = {R.fmt1(f['r2'], 4)}. Переводим через e?",
                     f"Нет: {rua(100 * d['r2'], 3)} % разброса log({c['y']}) объяснено log({c['x']})",
                     ["Да: e<sup>R²</sup>", "Да, но только если p &lt; 0,05", "Нет, его надо умножить на 2"],
                     "R², t, p, F не переводят никогда."))
    qs = pick(rng, qs_req, qs_opt, 5)
    return dict(cat="lin", sub="loglog", title=f"log-log: {formula}", intro=intro_lin(c, "loglog", f["n"], x),
                console="\n".join(lines), qs=qs)


# =====================================================================
# Logistic regression family
# =====================================================================
def glm_summary(formula, data, L, names):
    lines = [f"> m <- glm({formula}, family = binomial, data = {data})", "> summary(m)", "", "Call:",
             f"glm(formula = {formula}, family = binomial, data = {data})", "", "Coefficients:"]
    cl, est, se, zv, ps = coef_block(names, L["b"], L["se"], L["z"], L["p"], statname="z")
    lines += cl + ["---", R.SIGNIF, "", "(Dispersion parameter for binomial family taken to be 1)", ""]
    nd, rd = R.fmt_joint([L["null_dev"], L["res_dev"]], 5)
    w = max(len(nd), len(rd))
    dfs = [str(L["df_null"]), str(L["df_res"])]
    wd = max(len(s) for s in dfs)
    lines += [f"    Null deviance: {nd.rjust(w)}  on {dfs[0].rjust(wd)}  degrees of freedom",
              f"Residual deviance: {rd.rjust(w)}  on {dfs[1].rjust(wd)}  degrees of freedom",
              f"AIC: {R.fmt1(L['aic'], 5)}", "", f"Number of Fisher Scoring iterations: {L['iters']}"]
    return lines, dict(est=est, se=se, z=zv, p=ps)


def expcoef_lines(names, est):
    vals = [math.exp(b) for b in est]
    return ["> exp(coef(m))"] + R.named_vector(names, R.fmt_joint(vals, 7))


def prob(lo):
    return math.exp(lo) / (1 + math.exp(lo))


def intro_logit(c, extra):
    return f"<p>{c.get('what', '')}</p><p>Отклик {code(c['y'])}: {c['y_ru'].split(': ', 1)[1]}. {extra}</p><p class='dz'>{design_ru(c)}</p>"


def gen_logit2(rng, c):
    lv_written = c["levels"]
    p_first = min(0.93, max(0.04, c["p0"] * rng.uniform(0.8, 1.2)))
    orr = c["orr"] ** rng.choice([0.15, 0.8, 1.0, 1.0, 1.2])
    odds1 = p_first / (1 - p_first) * orr
    p_second = odds1 / (1 + odds1)
    ptrue = {lv_written[0]: p_first, lv_written[1]: p_second}
    lv = sorted(lv_written)
    ref, oth = lv[0], lv[1]
    n1 = int(rng.integers(60, 220))
    n2 = n1 if rng.random() < 0.5 else int(rng.integers(60, 220))
    ns = {ref: n1, oth: n2}
    ys, gs = [], []
    for l in lv:
        k = int(rng.binomial(ns[l], ptrue[l]))
        k = min(max(k, 3), ns[l] - 3)
        ys += [1] * k + [0] * (ns[l] - k)
        gs += [l] * ns[l]
    y = np.array(ys, float)
    g = np.array(gs)
    X = np.c_[np.ones(len(y)), (g == oth).astype(float)]
    L = logistic(X, y)
    names = ["(Intercept)", f"{c['g']}{oth}"]
    formula = f"{c['y']} ~ {c['g']}"
    lines, d = glm_summary(formula, c["data"], L, names)
    if rng.random() < 0.5:
        lines += expcoef_lines(names, L["b"])
    if rng.random() < 0.4:
        cnt = {l: int(y[g == l].sum()) for l in lv}
        lines += [f"> table({c['data']}${c['g']}, {c['data']}${c['y']})", "", " " * (len(max(lv, key=len)) + 2) + "  0   1"]
        w = len(max(lv, key=len))
        for l in lv:
            lines.append(f"  {l.ljust(w)} {str(ns[l] - cnt[l]).rjust(3)} {str(cnt[l]).rjust(3)}")
    b0, b1 = d["est"]
    se1 = d["se"][1]
    OR = math.exp(b1)
    ev, evn = c["ev"], c["evn"]
    qs_req, qs_opt = [], []
    qs_req.append(mc(rng, "Какая группа референтная (её нет среди строк коэффициентов)?",
                     f"{ref}", [f"{oth}", "Никакая: (Intercept) — общая вероятность", "Обе группы сразу"],
                     f"R берёт первую по алфавиту группу ({ref}) за базу. Где «вторая» группа? Она — строка {c['g']}{oth}. "
                     f"А референтная ({ref}) сидит в (Intercept): для неё {c['g']}{oth} = 0, и log-odds = b₀."))
    qs_req.append(num(f"Отношение шансов (OR) «{oth} против {ref}»?", OR,
                      f"OR = e<sup>b₁</sup> = e<sup>{rua(b1, 5)}</sup> = <b>{rua(OR, 4)}</b>. Шансы {evn} в группе {oth} в {rua(OR, 3)} раза "
                      f"{'больше' if OR > 1 else 'меньше'}, чем в {ref}.", rel=0.005))
    qs_req.append(num(f"OR наоборот: «{ref} против {oth}»?", 1 / OR,
                      f"Переворачиваем: 1 / OR = 1 / {rua(OR, 4)} = <b>{rua(1 / OR, 4)}</b> (то же, что e<sup>−b₁</sup> = e<sup>{rua(-b1, 5)}</sup>). "
                      f"Отдельной строки для «{ref} против {oth}» в выводе нет — её получают так.", rel=0.005))
    qs_opt.append(num(f"Вероятность события ({ev}) в референтной группе {ref}?", prob(b0),
                      f"log-odds {ref} = b₀ = {rua(b0, 5)}. Шансы = e<sup>{rua(b0, 5)}</sup> = {rua(math.exp(b0), 4)}. "
                      f"Вероятность = шансы / (1 + шансы) = {rua(math.exp(b0), 4)} / {rua(1 + math.exp(b0), 4)} = <b>{rua(prob(b0), 3)}</b>.",
                      rel=0.01, abs_=0.003))
    qs_req.append(num(f"Вероятность события ({ev}) в группе {oth}?", prob(b0 + b1),
                      f"log-odds {oth} = b₀ + b₁ = {rua(b0, 5)} + ({rua(b1, 5)}) = {rua(b0 + b1, 4)}. "
                      f"Шансы = e<sup>{rua(b0 + b1, 4)}</sup> = {rua(math.exp(b0 + b1), 4)}. "
                      f"Вероятность = {rua(math.exp(b0 + b1), 4)} / (1 + {rua(math.exp(b0 + b1), 4)}) = <b>{rua(prob(b0 + b1), 3)}</b>.",
                      rel=0.01, abs_=0.003))
    qs_opt.append(num(f"Шансы (odds) события в группе {ref}?", math.exp(b0),
                      f"Шансы = e<sup>b₀</sup> = e<sup>{rua(b0, 5)}</sup> = <b>{rua(math.exp(b0), 4)}</b>: на 1 «нет» приходится {rua(math.exp(b0), 3)} «да».",
                      rel=0.005))
    qs_opt.append(mc(rng, f"Как правильно сказать про OR = {rua(OR, 3)}?",
                     f"Шансы ({evn}) в группе {oth} в {rua(OR, 3)} раза {'выше' if OR > 1 else 'ниже'}, чем в {ref}",
                     [f"Вероятность в группе {oth} в {rua(OR, 3)} раза {'выше' if OR > 1 else 'ниже'}, чем в {ref}",
                      f"Шансы в группе {ref} в {rua(OR, 3)} раза {'выше' if OR > 1 else 'ниже'}, чем в {oth}",
                      f"Вероятность в группе {oth} на {rua(b1, 3)} {'выше' if b1 > 0 else 'ниже'}"],
                     "e<sup>b</sup> в логистической регрессии — отношение <b>шансов</b>, не вероятностей. Направление: группа из строки против референтной. "
                     f"<br><i>EN:</i> The odds of {c['y']} in the {oth} group are estimated to be {rua(OR, 3).replace(',', '.')} times the odds in the {ref} group."))
    up = math.exp(b1 + 1.96 * se1)
    lo_ = math.exp(b1 - 1.96 * se1)
    qs_opt.append(num("95% ДИ для OR: найди верхнюю границу (Wald: ±1,96·SE).", up,
                      f"Сначала на шкале log-odds: {rua(b1, 5)} + 1,96 × {rua(se1, 4)} = {rua(b1 + 1.96 * se1, 4)}. "
                      f"Потом e: <b>{rua(up, 4)}</b>. Нижняя: e<sup>{rua(b1 - 1.96 * se1, 4)}</sup> = {rua(lo_, 4)}. "
                      f"{'1 внутри → нет доказательств различия.' if lo_ < 1 < up else '1 снаружи → различие значимо.'}", rel=0.01))
    sig = L["p"][1] < 0.05
    qs_opt.append(mc(rng, f"Строка {c['g']}{oth}: p = {d['p'][1]}. Что проверяет этот тест и какой вывод?",
                     (f"H₀: β₁ = 0 (OR = 1, шансы одинаковы) — отвергаем" if sig else f"H₀: β₁ = 0 (OR = 1, шансы одинаковы) — не отвергаем"),
                     ([f"H₀: вероятность в {oth} = 0 — отвергаем", "H₀: OR = 0 — отвергаем", "H₀: β₁ = 1 — отвергаем"] if sig else
                      [f"H₀: β₁ = 0 — отвергаем", "H₀: OR = 0 — не отвергаем", "Доказано, что группы одинаковы"]),
                     "b = 0 на шкале log-odds ⇔ OR = e⁰ = 1 (нет разницы в шансах). z = Estimate / SE, p — по нормальному распределению."))
    qs = pick(rng, qs_req, qs_opt, 5)
    extra = f"Группа {code(c['g'])}: {lv_written[0]} или {lv_written[1]}. Всего n = {len(y)}."
    return dict(cat="logit", sub="g2", title=f"Логистическая: {formula}", intro=intro_logit(c, extra),
                console="\n".join(lines), qs=qs)


def gen_logitx(rng, c):
    n = int(rng.integers(c["n"][0], c["n"][1] + 1))
    x = np.round(rng.uniform(c["xlo"], c["xhi"], n), c["xdec"])
    b1t = c["b1"] * rng.choice([0.2, 0.8, 1.0, 1.0, 1.2])
    b0t = c["b0"] + (c["b1"] - b1t) * x.mean()
    pr = 1 / (1 + np.exp(-(b0t + b1t * x)))
    y = (rng.random(n) < pr).astype(float)
    L = logistic(np.c_[np.ones(n), x], y)
    formula = f"{c['y']} ~ {c['x']}"
    names = ["(Intercept)", c["x"]]
    lines, d = glm_summary(formula, c["data"], L, names)
    if rng.random() < 0.5:
        lines += expcoef_lines(names, L["b"])
    b0, b1 = d["est"]
    se1 = d["se"][1]
    xu = c["xu"]
    lo, hi = x.min(), x.max()
    x0 = float(np.round(rng.uniform(lo + 0.2 * (hi - lo), hi - 0.2 * (hi - lo)), c["xdec"]))
    OR = math.exp(b1)
    qs_req, qs_opt = [], []
    qs_req.append(num(f"OR для +1 к {c['x']} ({xu})?", OR,
                      f"e<sup>b₁</sup> = e<sup>{rua(b1, 5)}</sup> = <b>{rua(OR, 4)}</b>: каждая +1 к {c['x']} умножает шансы ({c['evn']}) на {rua(OR, 4)}.",
                      rel=0.005))
    k = float(np.round((hi - lo) * rng.choice([0.2, 0.3]), c["xdec"])) or 1.0
    qs_req.append(num(f"Во сколько раз меняются шансы, если {c['x']} больше на {k:g} {xu}?", math.exp(k * b1),
                      f"e<sup>{k:g} × b₁</sup> = e<sup>{rua(k * b1, 4)}</sup> = <b>{rua(math.exp(k * b1), 4)}</b> "
                      f"(то же, что {rua(OR, 4)}<sup>{k:g}</sup>). Не {k:g} × OR!", rel=0.01))
    qs_req.append(num(f"Вероятность события ({c['ev']}) при {c['x']} = {x0:g} {xu}?", prob(b0 + b1 * x0),
                      f"log-odds = {rua(b0, 5)} + {rua(b1, 5)} × {x0:g} = {rua(b0 + b1 * x0, 4)}. Шансы = e<sup>{rua(b0 + b1 * x0, 4)}</sup> = "
                      f"{rua(math.exp(b0 + b1 * x0), 4)}. Вероятность = шансы / (1 + шансы) = <b>{rua(prob(b0 + b1 * x0), 3)}</b>.",
                      rel=0.01, abs_=0.003))
    qs_opt.append(mc(rng, f"Как правильно прочитать наклон {rua(b1, 5)}?",
                     f"При +1 к {c['x']} шансы ({c['evn']}) умножаются на {rua(OR, 4)}",
                     [f"При +1 к {c['x']} вероятность меняется на {rua(b1, 4)}",
                      f"При +1 к {c['x']} вероятность умножается на {rua(OR, 4)}",
                      f"При +1 к {c['x']} шансы меняются на {rua(b1, 4)}"],
                     "b₁ — изменение log-odds. e<sup>b₁</sup> — во сколько раз меняются шансы. Вероятность меняется неравномерно: "
                     "в середине (около 0,5) быстрее, у краёв (около 0 и 1) медленнее."))
    qs_opt.append(num(f"На сколько процентов меняются шансы при +1 к {c['x']}? (со знаком)", (OR - 1) * 100,
                      f"(OR − 1) × 100 = ({rua(OR, 4)} − 1) × 100 = <b>{rua((OR - 1) * 100, 3)} %</b>.", rel=0.02, abs_=0.05, unit="%"))
    up = math.exp(b1 + 1.96 * se1)
    qs_opt.append(num("95% ДИ для OR на +1 единицу: верхняя граница (Wald)?", up,
                      f"e<sup>{rua(b1, 5)} + 1,96 × {rua(se1, 4)}</sup> = e<sup>{rua(b1 + 1.96 * se1, 4)}</sup> = <b>{rua(up, 4)}</b>. "
                      f"Нижняя = e<sup>{rua(b1 - 1.96 * se1, 4)}</sup> = {rua(math.exp(b1 - 1.96 * se1), 4)}.", rel=0.01))
    qs_opt.append(mc(rng, "Почему нельзя сказать «+1 единица X меняет вероятность на одно и то же число»?",
                     "Модель линейна по log-odds; на шкале вероятностей это S-кривая, шаг зависит от того, где мы на кривой",
                     ["Потому что X не под логарифмом", "Потому что p-value больше 0,05", "Можно, это и есть b₁"],
                     "Одинаково меняются только log-odds (+b₁) и шансы (× e<sup>b₁</sup>)."))
    qs_opt.append(num(f"Шансы события при {c['x']} = 0?", math.exp(b0),
                      f"e<sup>b₀</sup> = e<sup>{rua(b0, 5)}</sup> = <b>{rua(math.exp(b0), 4)}</b>. "
                      f"{'0 в данных есть.' if lo <= 0 <= hi else 'X = 0 вне данных — это экстраполяция, по смыслу не читаем.'}", rel=0.01))
    qs = pick(rng, qs_req, qs_opt, 5)
    extra = f"Предиктор {code(c['x'])} — {c['x_ru']}, {xu}; в данных от {rua(lo, 3)} до {rua(hi, 3)}. n = {n}."
    return dict(cat="logit", sub="gx", title=f"Логистическая: {formula}", intro=intro_logit(dict(c, what=""), extra),
                console="\n".join(lines), qs=qs)


def gen_logit3(rng, c):
    lv = sorted(c["levels"])
    p0 = c["p0"] * rng.uniform(0.8, 1.2)
    o0 = p0 / (1 - p0)
    ors = [o ** rng.choice([0.3, 1.0, 1.0, 1.2]) for o in c["ors"]]
    ptrue = {lv[0]: p0}
    for l, o in zip(lv[1:], ors):
        ptrue[l] = o0 * o / (1 + o0 * o)
    ys, gs = [], []
    for l in lv:
        n = int(rng.integers(60, 200))
        k = int(rng.binomial(n, ptrue[l]))
        k = min(max(k, 3), n - 3)
        ys += [1] * k + [0] * (n - k)
        gs += [l] * n
    y = np.array(ys, float)
    g = np.array(gs)
    X = np.column_stack([np.ones(len(y))] + [(g == l).astype(float) for l in lv[1:]])
    L = logistic(X, y)
    names = ["(Intercept)"] + [f"{c['g']}{l}" for l in lv[1:]]
    formula = f"{c['y']} ~ {c['g']}"
    lines, d = glm_summary(formula, c["data"], L, names)
    if rng.random() < 0.6:
        lines += expcoef_lines(names, L["b"])
    b0, bB, bC = d["est"]
    A, B, Cc = lv
    qs_req, qs_opt = [], []
    qs_req.append(mc(rng, "Какая группа референтная?", A, [B, Cc, "Никакая"],
                     f"Строк {c['g']}{B} и {c['g']}{Cc} есть, а {c['g']}{A} — нет. Значит {A} — база (первая по алфавиту), она в (Intercept)."))
    qs_req.append(num(f"OR «{Cc} против {B}»? (ни одна не референтная)", math.exp(bC - bB),
                      f"Обе строки — против {A}. Разница log-odds: {rua(bC, 5)} − ({rua(bB, 5)}) = {rua(bC - bB, 4)}. "
                      f"OR = e<sup>{rua(bC - bB, 4)}</sup> = <b>{rua(math.exp(bC - bB), 4)}</b>. "
                      f"Или делим OR: {rua(math.exp(bC), 4)} / {rua(math.exp(bB), 4)}.", rel=0.01))
    qs_req.append(num(f"Вероятность события ({c['ev']}) в группе {Cc}?", prob(b0 + bC),
                      f"log-odds {Cc} = b₀ + b{Cc} = {rua(b0, 5)} + ({rua(bC, 5)}) = {rua(b0 + bC, 4)}. "
                      f"Шансы = {rua(math.exp(b0 + bC), 4)}. Вероятность = <b>{rua(prob(b0 + bC), 3)}</b>.", rel=0.01, abs_=0.003))
    qs_opt.append(num(f"OR «{B} против {A}»?", math.exp(bB),
                      f"e<sup>{rua(bB, 5)}</sup> = <b>{rua(math.exp(bB), 4)}</b>.", rel=0.005))
    qs_opt.append(num(f"OR «{A} против {Cc}»?", math.exp(-bC),
                      f"Строка {c['g']}{Cc} даёт OR «{Cc} против {A}» = e<sup>{rua(bC, 5)}</sup> = {rua(math.exp(bC), 4)}. "
                      f"Наоборот: 1 / {rua(math.exp(bC), 4)} = <b>{rua(math.exp(-bC), 4)}</b>.", rel=0.005))
    qs_opt.append(num(f"Вероятность события в референтной группе {A}?", prob(b0),
                      f"e<sup>{rua(b0, 5)}</sup> = {rua(math.exp(b0), 4)} (шансы) → {rua(math.exp(b0), 4)} / (1 + {rua(math.exp(b0), 4)}) = "
                      f"<b>{rua(prob(b0), 3)}</b>.", rel=0.01, abs_=0.003))
    sigB = L["p"][1] < 0.05
    qs_opt.append(mc(rng, f"p для {c['g']}{B} = {d['p'][1]}. Что он проверяет?",
                     f"Отличаются ли шансы в {B} от шансов в {A}",
                     [f"Отличаются ли шансы в {B} от шансов в {Cc}", f"Равна ли вероятность в {B} нулю",
                      "Различаются ли все три группы сразу"],
                     f"Каждая строка — сравнение с референтной {A}. {'Здесь p &lt; 0,05 — различие значимо.' if sigB else 'Здесь p ≥ 0,05 — нет доказательств различия.'} "
                     f"Сравнение {B} с {Cc} в выводе нет — его получают через разницу коэффициентов (или сменив референтную группу)."))
    qs_opt.append(mc(rng, f"OR {Cc} против {A} = {rua(math.exp(bC), 3)}. Как сказать?",
                     f"Шансы ({c['evn']}) в {Cc} в {rua(math.exp(bC), 3)} раза {'выше' if bC > 0 else 'ниже'}, чем в {A}",
                     [f"Вероятность в {Cc} в {rua(math.exp(bC), 3)} раза {'выше' if bC > 0 else 'ниже'}, чем в {A}",
                      f"Шансы в {A} в {rua(math.exp(bC), 3)} раза {'выше' if bC > 0 else 'ниже'}, чем в {Cc}",
                      f"Шансы в {Cc} в {rua(math.exp(bC), 3)} раза {'выше' if bC > 0 else 'ниже'}, чем в среднем по всем фермам"],
                     "OR из строки — всегда «эта группа против референтной», и это про шансы, не вероятности."))
    qs = pick(rng, qs_req, qs_opt, 5)
    extra = f"Группа {code(c['g'])}: {', '.join(lv)}. Всего n = {len(y)}."
    return dict(cat="logit", sub="g3", title=f"Логистическая, 3 группы: {formula}", intro=intro_logit(c, extra),
                console="\n".join(lines), qs=qs)


PLAN = [
    ("aov", gen_aov, C.ANOVA, 70),
    ("lmfac", gen_lmfactor, C.ANOVA, 60),
    ("tukey", gen_tukey, C.ANOVA, 40),
    ("lin", gen_lin, C.LIN, 60),
    ("logy", gen_logy, C.LOGY, 50),
    ("logx", gen_logx, C.LOGX, 40),
    ("loglog", gen_loglog, C.LOGLOG, 30),
    ("g2", gen_logit2, C.LOGIT2, 60),
    ("gx", gen_logitx, C.LOGITX, 50),
    ("g3", gen_logit3, C.LOGIT3, 40),
]


def main():
    import gen2
    seed = 20261114
    out = []
    for sub, fn, bank, count in PLAN:
        for i in range(count):
            rng = np.random.default_rng(seed + len(out) * 7919)
            out.append(fn(rng, bank[i % len(bank)]))
    # v1.0 order and ids stay the same, so saved progress keeps matching
    order = np.random.default_rng(7).permutation(len(out))
    out = [out[i] for i in order]
    # v1.1: Poisson + multiple regression, appended after id 500
    new = []
    for sub, fn, bank, count in gen2.PLAN2:
        for i in range(count):
            rng = np.random.default_rng(seed + (len(out) + len(new)) * 7919)
            new.append(fn(rng, bank[i % len(bank)]))
    order2 = np.random.default_rng(8).permutation(len(new))
    out += [new[i] for i in order2]
    # v1.2: t-tests, nonparametric, categorical, appended after id 660
    import gen3
    new3 = []
    for sub, fn, bank, count in gen3.PLAN3:
        for i in range(count):
            rng = np.random.default_rng(seed + (len(out) + len(new3)) * 7919)
            new3.append(fn(rng, bank[i % len(bank)]))
    order3 = np.random.default_rng(9).permutation(len(new3))
    out += [new3[i] for i in order3]
    for i, e in enumerate(out):
        e["id"] = i + 1
    json.dump(out, open("examples.json", "w"), ensure_ascii=False, separators=(",", ":"))
    print(len(out), "examples", sum(len(e["qs"]) for e in out), "questions")


if __name__ == "__main__":
    main()
