"""v1.2: t-tests, nonparametric tests, categorical data (STAT 5101, module 2)."""
import math
import numpy as np
from scipy import stats
import rfmt as R
import htest as H
from gen import ru, rua, u, code, mc, num, pick, design_ru


def rf(v, d=5):
    """R-printed number, written the Russian way for question text."""
    return R.fmt1(v, d).replace(".", ",").replace("-", "−")


def f5(v):
    return H.fmt5(v).replace(".", ",").replace("-", "−")


def pfmt(p):
    return H.pval(p).replace(".", ",")


def gr(v):
    return f"{v:g}".replace(".", ",").replace("-", "−")


def evidence(p):
    if p < 0.001:
        return "очень сильные доказательства"
    if p < 0.01:
        return "сильные доказательства"
    if p < 0.05:
        return "умеренные доказательства"
    if p < 0.10:
        return "слабые, наводящие доказательства"
    return "доказательств нет"


EVID = ["очень сильные доказательства", "сильные доказательства", "умеренные доказательства",
        "слабые, наводящие доказательства", "доказательств нет"]


def evid_q(rng, p, ptxt, what):
    right = evidence(p)
    wrongs = [e for e in EVID if e != right]
    rng.shuffle(wrongs)
    return mc(rng, f"p-value = {ptxt}. Как сказать о силе доказательств ({what})?", right.capitalize(), [w.capitalize() for w in wrongs[:3]],
              "Шкала курса: &lt; 0,001 — очень сильные; 0,001–0,01 — сильные; 0,01–0,05 — умеренные; 0,05–0,10 — слабые, наводящие; "
              "&gt; 0,10 — доказательств нет. Всегда добавляй само p. Большое p не значит «разницы нет»: данных мало, чтобы её увидеть.")


def named(names, vals, digits=7):
    return R.named_vector(names, R.fmt_joint(vals, digits))


def design_txt(c):
    if c["design"] == "cc":
        return "Исследование случай-контроль: группы набраны по исходу."
    return design_ru(c)


def causal_q(rng, c):
    if c["design"] == "exp":
        return mc(rng, "Можно ли делать причинный вывод?", "Да: условия назначали случайно, значит, разница вызвана обработкой",
                  ["Нет: для причинного вывода нужен p &lt; 0,001", "Нет: никакой статистический тест не даёт причинного вывода",
                   "Да, потому что выборка большая"],
                  "Причинный вывод даёт дизайн (случайное назначение), а не величина p.")
    return mc(rng, "Можно ли делать причинный вывод?", "Нет: это наблюдение, вывод только о связи",
              ["Да, если p &lt; 0,05", "Да, если выборка больше 30", "Да, потому что группы сравнили t-тестом"],
              "Без случайного назначения группы могут различаться чем-то ещё (конфаундеры). Вывод — только о связи.")


# =====================================================================
# t-tests
TT2 = [
    dict(data="pigs", y="adg", g="diet", lv={"Control": (780, 70), "Probiotic": (815, 70)}, yu="г/сут", ydec=0, n=(10, 30), design="exp",
         y_ru="среднесуточный привес", what="Поросят на доращивании случайно разделили: обычный корм (Control) или корм с пробиотиком (Probiotic)."),
    dict(data="piglets", y="weanwt", g="creep", lv={"No": (6.8, 0.9), "Yes": (7.3, 0.9)}, yu="кг", ydec=2, n=(12, 35), design="exp",
         y_ru="масса при отъёме", what="Гнёзда случайно получали престартер (Yes) или нет (No); у поросят записали массу при отъёме."),
    dict(data="sows", y="born", g="parity", lv={"Gilt": (12.8, 2.6), "Sow": (14.1, 2.8)}, yu="пор.", ydec=0, n=(15, 40), design="obs",
         y_ru="живорождённых в гнезде", what="Записи опоросов: свинки (Gilt) и свиноматки (Sow)."),
    dict(data="carcass", y="backfat", g="sex", lv={"Barrow": (18.5, 3.2), "Gilt": (16.4, 3.0)}, yu="мм", ydec=1, n=(12, 35), design="obs",
         y_ru="толщина шпика", what="Туши с бойни: боровы-кастраты (Barrow) и свинки (Gilt)."),
    dict(data="cows", y="milk", g="bedding", lv={"Sand": (32.5, 5.0), "Straw": (30.8, 5.0)}, yu="кг/сут", ydec=1, n=(15, 40), design="obs",
         y_ru="удой", what="Коровы на двух фермах: подстилка из песка (Sand) или соломы (Straw)."),
    dict(data="sows", y="igg", g="vacc", lv={"No": (62, 14), "Yes": (71, 14)}, yu="г/л", ydec=0, n=(10, 25), design="exp",
         y_ru="IgG в молозиве", what="Свиноматок случайно вакцинировали (Yes) или нет (No) до опороса; измерили IgG в молозиве."),
    dict(data="pigs", y="cortisol", g="housing", lv={"Crate": (145, 42), "Group": (118, 18)}, yu="нмоль/л", ydec=0, n=(8, 25), design="obs",
         y_ru="кортизол в слюне", what="Свиньи в индивидуальных станках (Crate) и в групповых загонах (Group)."),
    dict(data="pens", y="water", g="drinker", lv={"Bowl": (6.1, 0.6), "Nipple": (7.4, 1.7)}, yu="л/сут", ydec=2, n=(8, 22), design="exp",
         y_ru="расход воды на свинью", what="Загоны случайно оборудовали чашечными (Bowl) или ниппельными (Nipple) поилками."),
    dict(data="calves", y="igg", g="feeding", lv={"Bottle": (22, 6), "Tube": (19, 11)}, yu="г/л", ydec=1, n=(8, 25), design="exp",
         y_ru="IgG в сыворотке телёнка", what="Телятам случайно выпаивали молозиво из бутылки (Bottle) или через зонд (Tube)."),
]

TPAIR = [
    dict(data="pigs", x="after", y="before", mu=95, sb=25, delta=28, sd=18, xu="нмоль/л", dec=0, n=(8, 20),
         unit="свиней", what="У каждой свиньи измерили кортизол до (before) и после (after) перевозки.", design="obs"),
    dict(data="litters", x="trt", y="ctl", mu=7.0, sb=1.0, delta=0.35, sd=0.4, xu="кг", dec=2, n=(8, 18),
         unit="пар поросят-сибсов", what="В каждом гнезде двух поросят-сибсов случайно разделили: один получал добавку (trt), другой нет (ctl).", design="exp"),
    dict(data="sows", x="day28", y="day0", mu=18, sb=3, delta=-2.4, sd=1.2, xu="мм", dec=1, n=(10, 25),
         unit="свиноматок", what="Толщину шпика у свиноматок измерили на опоросе (day0) и через 28 дней лактации (day28).", design="obs"),
    dict(data="cows", x="after", y="before", mu=30, sb=6, delta=1.2, sd=2.0, xu="кг/сут", dec=1, n=(10, 25),
         unit="коров", what="Удой каждой коровы записали за неделю до (before) и через неделю после (after) обрезки копыт.", design="obs"),
    dict(data="pigs", x="post", y="pre", mu=39.2, sb=0.3, delta=0.4, sd=0.35, xu="°C", dec=1, n=(10, 25),
         unit="свиней", what="Ректальную температуру измерили до (pre) и через 6 часов после (post) вакцинации.", design="obs"),
    dict(data="farms", x="y2026", y="y2025", mu=4.5, sb=1.2, delta=-0.6, sd=0.9, xu="%", dec=2, n=(8, 20),
         unit="ферм", what="Падёж на доращивании на одних и тех же фермах в 2025 (y2025) и 2026 (y2026) году.", design="obs"),
]

T1 = [
    dict(data="pigs", x="adg", mu0=800, m=780, s=60, xu="г/сут", dec=0, n=(12, 40), what="Цель компании — среднесуточный привес 800 г/сут. Взяли случайную выборку свиней с фермы."),
    dict(data="piglets", x="weanwt", mu0=7, m=6.8, s=0.9, xu="кг", dec=2, n=(15, 40), what="Норматив массы при отъёме — 7 кг. Взвесили случайную выборку поросят."),
    dict(data="feed", x="cp", mu0=17.5, m=17.2, s=0.5, xu="%", dec=2, n=(8, 20), what="На этикетке комбикорма сырой протеин 17,5 %. Лаборатория проверила пробы из партий."),
    dict(data="vials", x="dose", mu0=2.0, m=2.03, s=0.05, xu="мл", dec=3, n=(10, 30), what="Дозатор должен выдавать 2,0 мл вакцины. Измерили объём доз."),
    dict(data="sows", x="born", mu0=14, m=13.5, s=2.8, xu="пор.", dec=0, n=(20, 50), what="Генетическая компания обещает 14 живорождённых. Взяли записи опоросов."),
]


def tt_intro(c, n_txt):
    return f"<p>{c['what']}</p><p>{n_txt}</p><p class='dz'>{design_txt(c)}</p>"


def gen_t2(rng, c, var_equal):
    lv = sorted(c["lv"])
    g1, g2 = lv
    n1 = int(rng.integers(*c["n"]))
    n2 = n1 if rng.random() < 0.35 else int(rng.integers(*c["n"]))
    f = rng.choice([0.0, 0.5, 1.0, 1.0, 1.3])
    m1, s1 = c["lv"][g1]
    m2, s2 = c["lv"][g2]
    mid = (m1 + m2) / 2
    m1, m2 = mid + (m1 - mid) * f, mid + (m2 - mid) * f
    if var_equal:
        s = (s1 + s2) / 2
        s1 = s2 = s
    x = np.round(rng.normal(m1, s1, n1), c["ydec"])
    y = np.round(rng.normal(m2, s2, n2), c["ydec"])
    r = H.t_two(x, y, var_equal)
    lines = []
    show_sd = rng.random() < 0.55
    if show_sd:
        lines += [f"> tapply({c['data']}$" + c["y"] + f", {c['data']}$" + c["g"] + ", sd)"] + named(lv, [r["s1"], r["s2"]])
    call = f"t.test({c['y']} ~ {c['g']}, data = {c['data']}" + (", var.equal = TRUE)" if var_equal else ")")
    lines.append("> " + call)
    method = "Two Sample t-test" if var_equal else "Welch Two Sample t-test"
    lines += H.htest(method, f"{c['y']} by {c['g']}", [("t", r["t"]), ("df", r["df"])], r["p"],
                     f"true difference in means between group {g1} and group {g2} is not equal to 0", r["ci"], 95,
                     [(f"mean in group {g1}", r["m1"]), (f"mean in group {g2}", r["m2"])])
    pt = pfmt(r["p"])
    diff = r["m1"] - r["m2"]
    yu = c["yu"]
    qs_req, qs_opt = [], []
    qs_req.append(mc(rng, "Какой тест сделал R?", "Двухвыборочный t-тест с объединённым SD (pooled)" if var_equal else "t-тест Уэлча (Welch)",
                     (["t-тест Уэлча (Welch)", "Парный t-тест", "Одновыборочный t-тест"] if var_equal else
                      ["Двухвыборочный t-тест с объединённым SD (pooled)", "Парный t-тест", "Ранговая сумма Вилкоксона"]),
                     ("Первая строка вывода — «Two Sample t-test», df = n₁ + n₂ − 2 = " + fmt_i(r["df"]) + " — целое. Это pooled, его даёт var.equal = TRUE."
                      if var_equal else
                      f"Первая строка — «Welch Two Sample t-test», df = {f5(r['df'])} — дробные. В R это вариант по умолчанию, когда нет var.equal = TRUE.")))
    qs_req.append(num(f"Разность средних, которую проверяет R ({g1} − {g2})?", diff,
                      f"R вычитает вторую группу из первой по алфавиту: {rf(r['m1'], 7)} − {rf(r['m2'], 7)} = <b>{rua(diff, 4)}</b>{u(yu)}. "
                      "Концы ДИ — в тех же единицах и том же направлении.", rel=0.01, abs_=1e-6, unit=yu))
    se = diff / r["t"]
    qs_opt.append(num("SE разности средних (t.test его не печатает)?", se,
                      f"t = разность / SE, значит SE = разность / t = {rua(diff, 5)} / {f5(r['t'])} = <b>{rua(se, 4)}</b>{u(yu)}.",
                      rel=0.01, unit=yu))
    lo, hi = r["ci"]
    inside = lo < 0 < hi
    if inside:
        corr = f"Средние {g1} и {g2} могут быть равны: 0 внутри интервала, это согласуется с p = {pt} &gt; 0,05"
    else:
        sign = "выше" if diff > 0 else "ниже"
        corr = f"Мы на 95% уверены, что среднее у {g1} {sign} среднего у {g2} на величину от {rua(min(abs(lo), abs(hi)), 3)} до {rua(max(abs(lo), abs(hi)), 3)}{u(yu)}"
    qs_opt.append(mc(rng, f"Как прочитать 95% ДИ ({rf(lo, 5)}; {rf(hi, 5)})?", corr,
                     ["С вероятностью 95% каждое животное группы " + g1 + " попадёт в этот интервал",
                      "95% наблюдений разницы лежат в этом интервале",
                      "Интервал показывает, где лежит среднее группы " + g1],
                     "ДИ — правдоподобные значения <b>разности средних</b> μ₁ − μ₂, а не отдельных животных. "
                     "0 внутри ⇔ p &gt; 0,05; 0 снаружи ⇔ p &lt; 0,05."))
    qs_opt.append(evid_q(rng, r["p"], pt, f"различие средних {c['y']}"))
    if var_equal:
        qs_opt.append(num(f"Сколько животных всего в анализе (по df)?", r["df"] + 2,
                          f"У pooled df = n₁ + n₂ − 2, значит n₁ + n₂ = {fmt_i(r['df'])} + 2 = <b>{fmt_i(r['df'] + 2)}</b>.", rel=0))
    else:
        qs_opt.append(mc(rng, f"Почему df = {f5(r['df'])} дробные?",
                         "Уэлч считает df по формуле Сатертуэйта: дисперсии групп оцениваются отдельно",
                         ["R округлил ошибку вычислений", "Часть данных пропущена", "Так бывает, когда группы одного размера"],
                         "Welch не объединяет дисперсии и «платит» степенями свободы. Чем сильнее различаются s²/n групп, тем меньше df."))
    half = (hi - lo) / 2
    qs_opt.append(num("Погрешность ДИ: половина его ширины?", half,
                      f"({rf(hi, 7)} − ({rf(lo, 7)})) / 2 = <b>{rua(half, 4)}</b>{u(yu)}. Это t*·SE: ДИ = разность ± погрешность.",
                      rel=0.01, unit=yu))
    if show_sd:
        ratio = max(r["s1"], r["s2"]) / min(r["s1"], r["s2"])
        ok = ratio <= 2
        qs_opt.append(mc(rng, f"SD групп {rf(r['s1'], 4)} и {rf(r['s2'], 4)}. Подходит ли pooled по правилу курса?",
                         ("Да: отношение SD " + rua(ratio, 3) + " — в пределах 0,5–2") if ok else ("Нет: отношение SD " + rua(ratio, 3) + " больше 2, лучше Welch"),
                         (["Нет: SD должны совпадать точно", "Нет: нужен тест Шапиро", "Да, если p &lt; 0,05"] if ok else
                          ["Да: pooled подходит всегда", "Да, если n больше 10", "Нет, нужен парный t-тест"]),
                         "Правило курса: pooled уместен, если отношение SD от 0,5 до 2. Опасный случай — большой разброс в маленькой группе."))
        if var_equal:
            sp = math.sqrt(((n1 - 1) * r["s1"] ** 2 + (n2 - 1) * r["s2"] ** 2) / (n1 + n2 - 2))
            qs_opt.append(num(f"Объединённое SD s_p (n₁ = {n1}, n₂ = {n2})?", sp,
                              f"s_p = √[((n₁−1)s₁² + (n₂−1)s₂²) / (n₁+n₂−2)] = √[({n1 - 1}·{rf(r['s1'], 4)}² + {n2 - 1}·{rf(r['s2'], 4)}²) / {n1 + n2 - 2}] = <b>{rua(sp, 4)}</b>{u(yu)}.",
                              rel=0.01, unit=yu))
    qs_opt.append(causal_q(rng, c))
    qs = pick(rng, qs_req, qs_opt, 5)
    intro = tt_intro(c, f"Отклик {code(c['y'])} — {c['y_ru']}, {yu}. Группы {code(c['g'])}: {g1} (n = {n1}) и {g2} (n = {n2}).")
    return dict(cat="ttest", sub="t2p" if var_equal else "t2w", title=f"t-тест: {c['y']} ~ {c['g']}", intro=intro, console="\n".join(lines), qs=qs)


def fmt_i(v):
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else H.fmt5(v)


def gen_t2p(rng, c):
    return gen_t2(rng, c, True)


def gen_t2w(rng, c):
    return gen_t2(rng, c, False)


def gen_tpair(rng, c):
    n = int(rng.integers(*c["n"]))
    f = rng.choice([0.0, 0.5, 1.0, 1.0, 1.3])
    base = rng.normal(c["mu"], c["sb"], n)
    d = rng.normal(c["delta"] * f, c["sd"], n)
    y = np.round(base, c["dec"])
    x = np.round(base + d, c["dec"])
    dd = x - y
    r = H.t_one(dd)
    xs, ys = f"{c['data']}${c['x']}", f"{c['data']}${c['y']}"
    lines = []
    if rng.random() < 0.4:
        lines += [f"> sd({xs} - {ys})", "[1] " + R.r7(r["s"])]
    lines.append(f"> t.test({xs}, {ys}, paired = TRUE)")
    lines += H.htest("Paired t-test", f"{xs} and {ys}", [("t", r["t"]), ("df", r["df"])], r["p"],
                     "true mean difference is not equal to 0", r["ci"], 95, [("mean difference", r["m"])])
    pt = pfmt(r["p"])
    xu = c["xu"]
    qs_req, qs_opt = [], []
    qs_req.append(mc(rng, "Какую разность усредняет парный t-тест?", f"{c['x']} − {c['y']} внутри каждой пары",
                     [f"{c['y']} − {c['x']} внутри каждой пары", f"Среднее {c['x']} минус среднее {c['y']} как у независимых групп",
                      "Разность медиан"],
                     f"R вычитает второй аргумент из первого для каждой пары: {xs} − {ys}. "
                     f"mean difference = {rf(r['m'], 7)}: в среднем {c['x']} {'больше' if r['m'] > 0 else 'меньше'} {c['y']}."))
    qs_req.append(num(f"Сколько пар в анализе ({c['unit']})?", r["df"] + 1,
                      f"У парного теста df = n − 1 (n — число пар). n = {r['df']} + 1 = <b>{r['df'] + 1}</b>.", rel=0))
    se = r["m"] / r["t"]
    qs_opt.append(num("SE средней разности?", se,
                      f"SE = mean difference / t = {rf(r['m'], 7)} / {f5(r['t'])} = <b>{rua(se, 4)}</b>{u(xu)}. Это s_D / √n.",
                      rel=0.01, unit=xu))
    qs_opt.append(num("SD разностей s_D?", se * math.sqrt(r["n"]),
                      f"SE = s_D / √n ⇒ s_D = SE·√n = {rua(se, 4)} × √{r['n']} = <b>{rua(se * math.sqrt(r['n']), 4)}</b>{u(xu)}.",
                      rel=0.015, unit=xu))
    qs_opt.append(mc(rng, "Почему здесь парный тест, а не двухвыборочный?",
                     "Каждому значению " + c["x"] + " соответствует конкретный партнёр " + c["y"] + " (та же единица или сибс)",
                     ["Потому что групп две", "Потому что в группах одинаковое число наблюдений", "Потому что данные не нормальны"],
                     "Вопрос-проверка: можно ли для каждого наблюдения назвать одного конкретного партнёра? Равные n ничего не доказывают. "
                     "Разность внутри пары убирает различия между животными, поэтому SE парного теста обычно намного меньше."))
    qs_opt.append(mc(rng, "Какие допущения у парного t-теста?", "Пары независимы друг от друга, разности примерно нормальны",
                     ["Обе группы нормальны и их SD равны", "Нет выбросов и тест Шапиро незначим", "n больше 30 в каждой группе"],
                     "Только два: независимость пар (по дизайну) и нормальность разностей (QQ-график разностей). Равенство SD групп не нужно."))
    lo, hi = r["ci"]
    qs_opt.append(mc(rng, f"95% ДИ ({rf(lo, 5)}; {rf(hi, 5)}). Что он говорит?",
                     ("0 внутри: средняя разность может быть нулевой" if lo < 0 < hi else
                      f"Средняя разность {c['x']} − {c['y']} с 95% уверенностью от {rf(lo, 4)} до {rf(hi, 4)}{u(xu)}, 0 не входит"),
                     ["95% животных изменились на величину в этом интервале", "Интервал для среднего " + c["x"], "Интервал для SD разностей"],
                     "ДИ — для средней разности в популяции пар. 0 внутри ⇔ p &gt; 0,05."))
    qs_opt.append(evid_q(rng, r["p"], pt, "средняя разность ≠ 0"))
    qs_opt.append(mc(rng, "Что будет, если по ошибке сделать t.test(x, y) без paired = TRUE?",
                     "SE станет больше (различия между животными попадут в шум), p вырастет",
                     ["Ничего: результат тот же", "p станет меньше", "R выдаст ошибку"],
                     "Var(D) = s₁² + s₂² − 2r·s₁·s₂. Двухвыборочный тест выбрасывает член с корреляцией r, поэтому шум завышен."))
    qs = pick(rng, qs_req, qs_opt, 5)
    intro = f"<p>{c['what']}</p><p>Всего {c['unit']}: {r['n']}. Единицы — {xu}.</p><p class='dz'>{design_txt(c)}</p>"
    return dict(cat="ttest", sub="tpair", title=f"Парный t-тест: {c['x']} − {c['y']}", intro=intro, console="\n".join(lines), qs=qs)


def gen_t1(rng, c):
    n = int(rng.integers(*c["n"]))
    m = c["mu0"] + (c["m"] - c["mu0"]) * rng.choice([0.0, 0.6, 1.0, 1.0, 1.4])
    x = np.round(rng.normal(m, c["s"], n), c["dec"])
    r = H.t_one(x, c["mu0"])
    xs = f"{c['data']}${c['x']}"
    lines = [f"> t.test({xs}, mu = {c['mu0']:g})"]
    lines += H.htest("One Sample t-test", xs, [("t", r["t"]), ("df", r["df"])], r["p"],
                     f"true mean is not equal to {c['mu0']:g}", r["ci"], 95, [("mean of x", r["m"])])
    pt = pfmt(r["p"])
    xu = c["xu"]
    qs_req = [mc(rng, "Какую H₀ проверяет тест?", f"Среднее в популяции равно {gr(c['mu0'])}{u(xu)}",
                 [f"Выборочное среднее равно {gr(c['mu0'])}", "Среднее равно 0", f"Все значения равны {gr(c['mu0'])}"],
                 f"mu = {c['mu0']:g} — значение из H₀ для <b>популяционного</b> среднего μ. Выборочное среднее ({rf(r['m'], 7)}) мы знаем точно, его не проверяют.")]
    qs_opt = []
    se = (r["m"] - c["mu0"]) / r["t"]
    qs_opt.append(num("SE среднего?", se,
                      f"t = (x̄ − μ₀) / SE ⇒ SE = ({rf(r['m'], 7)} − {gr(c['mu0'])}) / {f5(r['t'])} = <b>{rua(se, 4)}</b>{u(xu)}. Это s / √n.",
                      rel=0.01, unit=xu))
    qs_opt.append(num("Сколько наблюдений?", r["df"] + 1, f"df = n − 1 ⇒ n = {r['df']} + 1 = <b>{r['df'] + 1}</b>.", rel=0))
    lo, hi = r["ci"]
    inside = lo < c["mu0"] < hi
    qs_opt.append(mc(rng, f"Попадает ли {gr(c['mu0'])} в 95% ДИ и что это значит?",
                     ("Да, попадает: среднее может быть равно нормативу, p &gt; 0,05" if inside else
                      "Нет: весь ДИ " + ("ниже" if hi < c["mu0"] else "выше") + " норматива, p &lt; 0,05"),
                     (["Нет: весь ДИ выше норматива", "Да, и это доказывает, что среднее равно нормативу", "ДИ тут ни при чём"] if inside else
                      ["Да, попадает", "Нет, но это не связано с p", "Не попадает, значит каждое животное ниже норматива"]),
                     f"ДИ ({rf(lo, 5)}; {rf(hi, 5)}) — правдоподобные значения μ. Значение H₀ внутри ⇔ двусторонний p &gt; 0,05. "
                     "«Попадает» не доказывает равенство: данных просто мало, чтобы увидеть разницу."))
    qs_opt.append(evid_q(rng, r["p"], pt, f"среднее ≠ {gr(c['mu0'])}"))
    qs_opt.append(num(f"Насколько выборочное среднее отличается от {gr(c['mu0'])}? (x̄ − μ₀, со знаком)", r["m"] - c["mu0"],
                      f"{rf(r['m'], 7)} − {gr(c['mu0'])} = <b>{rua(r['m'] - c['mu0'], 4)}</b>{u(xu)}.", rel=0.01, abs_=1e-6, unit=xu))
    qs = pick(rng, qs_req, qs_opt, 5)
    intro = f"<p>{c['what']}</p><p>{code(c['x'])}, {xu}; n = {r['n']}.</p>"
    return dict(cat="ttest", sub="t1", title=f"Одновыборочный t-тест: {c['x']}", intro=intro, console="\n".join(lines), qs=qs)


# =====================================================================
# nonparametric
WRS = [
    dict(data="swabs", y="cfu", g="wash", lv={"Control": (0.6, 0.9), "Wash": (0.12, 0.8)}, dec=2, yu="КОЕ/см²", n=(5, 12),
         y_ru="E. coli на смыве с туши", design="exp", what="Туши случайно обрабатывали раствором (Wash) или нет (Control) и считали E. coli на смыве."),
    dict(data="pigs", y="epg", g="drug", lv={"Control": (400, 0.9), "Ivermectin": (90, 1.1)}, dec=0, yu="яиц/г", n=(6, 14),
         y_ru="яиц гельминтов в грамме фекалий", design="exp", what="Свиней случайно обработали ивермектином или оставили без обработки."),
    dict(data="sows", y="wei", g="treat", lv={"Control": (6.0, 0.5), "Hormone": (4.8, 0.4)}, dec=1, yu="дн.", n=(6, 14),
         y_ru="дней от отъёма до охоты", design="exp", what="После отъёма свиноматки случайно получали гормональный препарат (Hormone) или нет (Control)."),
    dict(data="pigs", y="hp", g="vacc", lv={"No": (1.4, 0.6), "Yes": (0.9, 0.6)}, dec=2, yu="г/л", n=(6, 14),
         y_ru="гаптоглобин в сыворотке", design="obs", what="Свиньи из вакцинированных (Yes) и невакцинированных (No) партий."),
    dict(data="pens", y="score", g="floor", lv={"Rubber": (1.0, 0), "Slatted": (2.2, 0)}, dec=0, yu="балл", n=(15, 30), ordinal=True,
         y_ru="балл хромоты 0–5", design="obs", what="Свиноматки на резиновых матах (Rubber) и щелевом полу (Slatted); ветеринар оценил хромоту по шкале 0–5."),
    dict(data="calves", y="days", g="colostrum", lv={"Good": (3.0, 0.7), "Poor": (5.5, 0.7)}, dec=0, yu="дн.", n=(15, 30), ordinal=True,
         y_ru="дней с диареей", design="obs", what="Телята с хорошей (Good) и плохой (Poor) передачей иммунитета от молозива."),
]


def gen_wrs(rng, c):
    lv = sorted(c["lv"])
    g1, g2 = lv
    n1, n2 = int(rng.integers(*c["n"])), int(rng.integers(*c["n"]))
    f = rng.choice([0.0, 0.6, 1.0, 1.0])
    ordinal = c.get("ordinal", False)
    for _ in range(200):
        if ordinal:
            m1, m2 = c["lv"][g1][0], c["lv"][g2][0]
            mid = (m1 + m2) / 2
            x = np.clip(np.round(rng.normal(mid + (m1 - mid) * f, 1.2, n1)), 0, 5)
            y = np.clip(np.round(rng.normal(mid + (m2 - mid) * f, 1.2, n2)), 0, 5)
            break
        (med1, sl1), (med2, sl2) = c["lv"][g1], c["lv"][g2]
        mid = math.sqrt(med1 * med2)
        a1 = math.log(mid) + (math.log(med1) - math.log(mid)) * f
        a2 = math.log(mid) + (math.log(med2) - math.log(mid)) * f
        x = np.round(np.exp(rng.normal(a1, sl1, n1)), c["dec"])
        y = np.round(np.exp(rng.normal(a2, sl2, n2)), c["dec"])
        if len(np.unique(np.r_[x, y])) == n1 + n2 and (x > 0).all() and (y > 0).all():
            break
    ties = len(np.unique(np.r_[x, y])) < n1 + n2
    if ties or ordinal:
        style = "approx"
        r = H.rank_sum(x, y, exact=False, correct=False)
        call = f"wilcox.test({c['y']} ~ {c['g']}, data = {c['data']}, exact = FALSE, correct = FALSE)"
    else:
        style = "exact"
        r = H.rank_sum(x, y)
        call = f"wilcox.test({c['y']} ~ {c['g']}, data = {c['data']})"
    lines = []
    if rng.random() < 0.6:
        lines += [f"> tapply({c['data']}${c['y']}, {c['data']}${c['g']}, median)"] + named(lv, [float(np.median(x)), float(np.median(y))])
    lines.append("> " + call)
    lines += H.htest(r["method"], f"{c['y']} by {c['g']}", [("W", r["W"])], r["p"], "true location shift is not equal to 0")
    pt = pfmt(r["p"])
    W = r["W"]
    qs_req, qs_opt = [], []
    qs_req.append(mc(rng, "Что такое W в выводе?", f"Сумма рангов группы {g1} минус её минимум n₁(n₁+1)/2",
                     [f"Сумма рангов группы {g2}", "Разность медиан групп", "Число наблюдений больше общей медианы"],
                     f"R считает W для первого уровня по алфавиту ({g1}): W = R₁ − n₁(n₁+1)/2. "
                     f"Второй смысл: в стольких парах «{g1} против {g2}» значение {g1} больше (ничья = ½)."))
    R1 = W + n1 * (n1 + 1) / 2
    qs_req.append(num(f"Сумма рангов группы {g1} (n₁ = {n1})?", R1,
                      f"R₁ = W + n₁(n₁+1)/2 = {f5(W)} + {n1}·{n1 + 1}/2 = <b>{gr(R1)}</b>.", rel=0, abs_=0.01))
    qs_opt.append(num(f"Каким было бы W в среднем, если группы не различаются (n₁ = {n1}, n₂ = {n2})?", n1 * n2 / 2,
                      f"E₀(W) = n₁·n₂ / 2 = {n1}·{n2} / 2 = <b>{gr(n1 * n2 / 2)}</b>. Наблюдаемое W = {f5(W)}: "
                      f"{'меньше' if W < n1 * n2 / 2 else 'больше'} середины.", rel=0, abs_=0.01))
    share = W / (n1 * n2) * 100
    qs_opt.append(num(f"В каком проценте пар «{g1} против {g2}» значение {g1} больше?", share,
                      f"W / (n₁n₂) = {f5(W)} / {n1 * n2} = <b>{rua(share, 3)} %</b>. Если группы одинаковы, ждём 50 %.",
                      rel=0.01, abs_=0.2, unit="%"))
    qs_opt.append(mc(rng, "Какую H₀ проверяет ранговая сумма Вилкоксона?",
                     "Распределение величины одинаково в двух популяциях",
                     ["Средние двух групп равны", "Дисперсии групп равны", "Данные нормальны"],
                     "H₀ курса: распределения одинаковы. Если формы одинаковы и отличаются только сдвигом, это «медианы равны». "
                     "Средние ранговый тест не сравнивает."))
    if style == "exact":
        qs_opt.append(mc(rng, "Почему в выводе «exact test»?", "Групп меньше 50 наблюдений и нет ничьих: R перебрал все раскладки рангов",
                         ["Данные нормальны", "R использовал t-распределение", "Задан exact = TRUE обязательно"],
                         "По умолчанию R считает точный p при n &lt; 50 в каждой группе без ничьих. Это рандомизационный тест по рангам."))
    else:
        qs_opt.append(mc(rng, "Почему здесь exact = FALSE и нормальное приближение?",
                         "В данных есть ничьи (одинаковые значения), точный p классическим способом не считается",
                         ["Данные нормальны", "Группы больше 1000", "Так ранговый тест работает всегда"],
                         "С ничьими R предупредил бы «cannot compute exact p-value with ties». В курсе пишут exact = FALSE, correct = FALSE: "
                         "z по средним рангам с поправкой на ничьи, без поправки на непрерывность."))
    qs_opt.append(num(f"Чему было бы W, если считать для группы {g2}?", n1 * n2 - W,
                      f"W₁ + W₂ = n₁n₂ ⇒ W₂ = {n1 * n2} − {f5(W)} = <b>{gr(n1 * n2 - W)}</b>. p тот же: оба равноудалены от {gr(n1 * n2 / 2)}.",
                      rel=0, abs_=0.01))
    qs_opt.append(evid_q(rng, r["p"], pt, "распределения различаются"))
    qs_opt.append(mc(rng, "Почему здесь выбрали ранговый тест, а не t-тест?",
                     "Маленькие группы и скошенные данные или выбросы (или порядковая шкала)",
                     ["Потому что групп две", "Потому что ранговый тест всегда мощнее", "Потому что данные парные"],
                     "Ранги устойчивы к выбросам: самое большое значение просто получает последний ранг. Плата — чуть меньшая мощность при нормальных данных."))
    qs = pick(rng, qs_req, qs_opt, 5)
    intro = f"<p>{c['what']}</p><p>Отклик {code(c['y'])} — {c['y_ru']}, {c['yu']}. {g1}: n = {n1}, {g2}: n = {n2}.</p><p class='dz'>{design_txt(c)}</p>"
    return dict(cat="nonpar", sub="wrs", title=f"Вилкоксон: {c['y']} ~ {c['g']}", intro=intro, console="\n".join(lines), qs=qs)


WSR = [
    dict(data="pigs", x="post", y="pre", mu=1.2, sl=0.5, eff=0.45, sd=0.4, dec=2, xu="г/л", n=(8, 20),
         unit="свиней", what="Гаптоглобин у каждой свиньи до (pre) и после (post) перевода в новый корпус.", design="obs"),
    dict(data="cows", x="after", y="before", mu=250, sl=0.6, eff=-90, sd=70, dec=0, xu="тыс./мл", n=(8, 20),
         unit="коров", what="Соматические клетки в молоке каждой коровы до (before) и после (after) лечения мастита.", design="obs"),
    dict(data="pens", x="straw", y="bare", mu=6, sl=0.4, eff=-1.8, sd=1.6, dec=1, xu="укусов/ч", n=(8, 20),
         unit="загонов", what="В каждом загоне считали укусы хвостов неделю без соломы (bare) и неделю с соломой (straw).", design="exp"),
    dict(data="litters", x="trt", y="ctl", mu=5.5, sl=0.15, eff=0.3, sd=0.35, dec=2, xu="кг", n=(8, 20),
         unit="пар поросят-сибсов", what="Внутри гнезда поросята-сибсы: один получил железо дважды (trt), другой один раз (ctl).", design="exp"),
]


def gen_wsr(rng, c):
    n = int(rng.integers(*c["n"]))
    f = rng.choice([0.0, 0.6, 1.0, 1.0, 1.3])
    for _ in range(500):
        base = np.exp(rng.normal(math.log(c["mu"]), c["sl"], n))
        d = rng.normal(c["eff"] * f, c["sd"], n)
        if rng.random() < 0.5:
            k = int(rng.integers(0, n))
            d[k] = d[k] - 3 * c["sd"] * np.sign(c["eff"] or 1)  # one discordant outlier
        y = np.round(base, c["dec"])
        x = np.round(base + d, c["dec"])
        dd = x - y
        if (dd != 0).all() and len(np.unique(np.round(np.abs(dd), 9))) == n and (x > 0).all():
            break
    r = H.signed_rank(x, y)
    xs, ys = f"{c['data']}${c['x']}", f"{c['data']}${c['y']}"
    lines = []
    if rng.random() < 0.5:
        lines += [f"> median({xs} - {ys})", "[1] " + R.r7(float(np.median(dd)))]
    lines.append(f"> wilcox.test({xs}, {ys}, paired = TRUE)")
    lines += H.htest(r["method"], f"{xs} and {ys}", [("V", r["V"])], r["p"], "true location shift is not equal to 0")
    pt = pfmt(r["p"])
    V = r["V"]
    tot = n * (n + 1) / 2
    npos = int((dd > 0).sum())
    qs_req = [mc(rng, "Что такое V?", f"Сумма рангов |разностей| у пар, где {c['x']} − {c['y']} &gt; 0",
                 ["Число положительных разностей", "Сумма самих положительных разностей", "Медиана разностей"],
                 f"Разности ранжируют по модулю, возвращают знаки и складывают ранги положительных. Положительных разностей здесь {npos} из {n}."),
              num(f"Сумма рангов отрицательных разностей (всего {n} пар)?", tot - V,
                  f"Все ранги 1 + … + n = n(n+1)/2 = {gr(tot)}. Отрицательные: {gr(tot)} − V = {gr(tot)} − {f5(V)} = <b>{gr(tot - V)}</b>.",
                  rel=0, abs_=0.01)]
    qs_opt = [num("Каким было бы V в среднем при H₀?", tot / 2,
                  f"При H₀ каждый ранг с вероятностью ½ плюс или минус: E₀(V) = n(n+1)/4 = {n}·{n + 1}/4 = <b>{gr(tot / 2)}</b>.", rel=0, abs_=0.01),
              mc(rng, "Какую H₀ проверяет знаково-ранговый тест?", "Медиана разностей внутри пар равна 0",
                 ["Средние двух групп равны", "Медианы двух независимых групп равны", "Разности нормальны"],
                 "Это тест для пар. Допущения: пары независимы, разности симметричны. Медиана разностей — не то же, что разность медиан."),
              mc(rng, "Что будет без paired = TRUE?", "R сделает ранговую сумму двух независимых групп — неверный тест для пар",
                 ["То же самое", "R выдаст ошибку", "R сделает парный t-тест"],
                 "wilcox.test(x, y) без paired = TRUE считает W для двух независимых выборок и игнорирует, кто с кем в паре."),
              evid_q(rng, r["p"], pt, "медиана разностей ≠ 0"),
              mc(rng, "Почему не парный t-тест?", "Мало пар и разности скошены или есть выброс",
                 ["Потому что разности нормальны", "Потому что пар больше 30", "Парный t-тест нельзя для пар"],
                 "Знаково-ранговый тест устойчив к выбросам в разностях: выброс получает лишь самый большой ранг. При нормальных разностях парный t чуть мощнее.")]
    qs = pick(rng, qs_req, qs_opt, 5)
    intro = f"<p>{c['what']}</p><p>{c['unit'].capitalize()}: {n}. Единицы — {c['xu']}.</p><p class='dz'>{design_txt(c)}</p>"
    return dict(cat="nonpar", sub="wsr", title=f"Знаково-ранговый: {c['x']} − {c['y']}", intro=intro, console="\n".join(lines), qs=qs)


SIGN = [
    dict(data="schiz", x="unaff", y="aff", what="Пары однояйцевых близнецов: объём гиппокампа у здорового (unaff) и больного (aff) близнеца.", unit="пар близнецов", n=(10, 20), p=0.85),
    dict(data="pens", x="after", y="before", what="Падёж в каждом загоне до (before) и после (after) смены вакцины.", unit="загонов", n=(8, 25), p=0.3),
    dict(data="sows", x="new", y="old", what="Каждая свиноматка прошла два опороса: на старом (old) и новом (new) корме; записали массу гнезда при отъёме.", unit="свиноматок", n=(10, 30), p=0.7),
    dict(data="darwin", x="cross", y="self", what="В каждом горшке растение перекрёстного (cross) и самоопыления (self).", unit="горшков", n=(10, 20), p=0.8),
]


def gen_sign(rng, c):
    n = int(rng.integers(*c["n"]))
    p = c["p"] if rng.random() < 0.7 else 0.5
    S = int(rng.binomial(n, p))
    b = H.binom_test(S, n)
    xs, ys = f"{c['data']}${c['x']}", f"{c['data']}${c['y']}"
    lines = [f"> d <- {xs} - {ys}", "> sum(d > 0)", f"[1] {S}", "> length(d)", f"[1] {n}", f"> binom.test({S}, {n})"]
    lines += H.htest("Exact binomial test", f"{S} and {n}", [("number of successes", S), ("number of trials", n)], b["p"],
                     "true probability of success is not equal to 0.5", b["ci"], 95, [("probability of success", b["est"])])
    pt = pfmt(b["p"])
    qs_req = [mc(rng, "Почему p = 0.5 в H₀?", "Если медиана разностей равна 0, каждая разность — как бросок монетки: плюс или минус с вероятностью ½",
                 ["Потому что половина животных получила обработку", "Это уровень значимости", "Так R делает всегда, это не связано с задачей"],
                 "Тест знаков: S = число положительных разностей ~ Bin(n, 0,5) при H₀. Нули выбрасывают."),
              mc(rng, f"S = {S} из {n}. Что показывает «probability of success» = {rf(b['est'], 7)}?",
                 f"Долю пар, где {c['x']} &gt; {c['y']}",
                 ["Вероятность, что H₀ верна", "Среднюю разность", f"Долю {c['unit']}, у которых нет разницы"],
                 "Оценка — S/n. ДИ — для доли положительных разностей в популяции пар. 0,5 внутри ⇔ p &gt; 0,05.")]
    qs_opt = [evid_q(rng, b["p"], pt, "медиана разностей ≠ 0"),
              mc(rng, "Чем тест знаков отличается от знаково-рангового?", "Учитывает только знак разности, не её размер",
                 ["Учитывает размер разностей через ранги", "Требует нормальности разностей", "Подходит для независимых групп"],
                 "Тест знаков выбрасывает всё, кроме направления. Ему нужна только независимость пар, выбросы ему безразличны. Цена — мощность."),
              mc(rng, "Можно ли тот же тест сделать в DescTools?", "Да: SignTest(x, y) печатает тот же S и p",
                 ["Нет, только binom.test", "Да, это t.test(x, y, paired = TRUE)", "Да, это wilcox.test(x, y)"],
                 "В Lab 4 курса — SignTest(schiz$unaff, schiz$aff). Он дополнительно печатает медиану разностей и её ДИ.")]
    if S in (0, n):
        qs_opt.append(num("Посчитай p вручную: все разности одного знака.", 2 * 0.5 ** n,
                          f"P(S = {n}) = 0,5<sup>{n}</sup>; двусторонний p = 2·0,5<sup>{n}</sup> = <b>{rua(2 * 0.5 ** n, 3)}</b>.", rel=0.01))
    elif S in (1, n - 1):
        qs_opt.append(num("Посчитай p вручную.", min(1, 2 * (n + 1) * 0.5 ** n),
                          f"p = 2·0,5<sup>{n}</sup>·[C({n},{max(S, n - S)}) + C({n},{n})] = 2·({n} + 1) / 2<sup>{n}</sup> = <b>{rua(2 * (n + 1) * 0.5 ** n, 3)}</b>.", rel=0.01))
    qs = pick(rng, qs_req, qs_opt, 5)
    intro = f"<p>{c['what']}</p><p>{c['unit'].capitalize()}: {n}. Тест знаков: считаем, у скольких пар разность {c['x']} − {c['y']} положительна.</p>"
    return dict(cat="nonpar", sub="sign", title=f"Тест знаков: {S} из {n}", intro=intro, console="\n".join(lines), qs=qs)


# =====================================================================
# categorical
C22 = [
    dict(rows=["Vacc", "Control"], cols=["Sick", "Healthy"], p=(0.18, 0.32), n=(60, 220), design="exp",
         what="Поросят случайно вакцинировали (Vacc) или нет (Control) и считали заболевших диареей (Sick).", ev="заболеть"),
    dict(rows=["Probiotic", "Control"], cols=["Died", "Alive"], p=(0.04, 0.07), n=(300, 900), design="exp",
         what="Партии на доращивании случайно получали пробиотик или нет; считали павших.", ev="пасть"),
    dict(rows=["DrugA", "DrugB"], cols=["Cured", "Not"], p=(0.72, 0.58), n=(40, 140), design="exp",
         what="Больных свиней случайно лечили препаратом A или B; исход — выздоровела ли.", ev="выздороветь"),
    dict(rows=["Gilt", "Sow"], cols=["Stillborn", "None"], p=(0.35, 0.47), n=(80, 260), design="obs",
         what="Гнёзда свинок (Gilt) и свиноматок (Sow): было ли хоть одно мертворождённое.", ev="получить гнездо с мертворождённым"),
    dict(rows=["Low", "High"], cols=["PED", "Free"], p=(0.30, 0.12), n=(40, 120), design="obs",
         what="Фермы с низким (Low) и высоким (High) уровнем биобезопасности: была ли вспышка PED за год.", ev="получить вспышку"),
    dict(rows=["Case", "Control"], cols=["Slatted", "Solid"], p=(0.62, 0.45), n=(60, 200), design="cc",
         what="Набрали хромых свиноматок (Case) и здоровых (Control) и выяснили, на каком полу их держали.", ev="стоять на щелевом полу"),
    dict(rows=["Case", "Control"], cols=["Bought", "Own"], p=(0.40, 0.24), n=(50, 160), design="cc",
         what="Фермы со вспышкой PRRS (Case) и без неё (Control): покупали ли ремонтных свинок на стороне.", ev="покупать свинок"),
]


def tab_lines(name, rows, cols, M):
    vals = ", ".join(str(int(v)) for v in np.asarray(M).ravel())
    rr = ", ".join(f'"{r}"' for r in rows)
    cc = ", ".join(f'"{c_}"' for c_ in cols)
    lines = [f"> {name} <- matrix(c({vals}), nrow = {len(rows)}, byrow = TRUE,",
             f"+               dimnames = list(c({rr}), c({cc})))", f"> {name}"]
    lines += R.matrix(rows, cols, [list(map(float, col)) for col in np.asarray(M).T])
    return lines


def gen_c22(rng, c):
    n1, n2 = int(rng.integers(*c["n"])), int(rng.integers(*c["n"]))
    f = rng.choice([0.0, 0.6, 1.0, 1.0, 1.3])
    p1, p2 = c["p"]
    mid = (p1 + p2) / 2
    p1, p2 = mid + (p1 - mid) * f, mid + (p2 - mid) * f
    a = int(rng.binomial(n1, p1))
    cc_ = int(rng.binomial(n2, p2))
    a, cc_ = max(a, 2), max(cc_, 2)
    a, cc_ = min(a, n1 - 2), min(cc_, n2 - 2)
    M = np.array([[a, n1 - a], [cc_, n2 - cc_]])
    r1, r2 = c["rows"]
    e1, e0 = c["cols"]
    lines = tab_lines("tab", c["rows"], c["cols"], M)
    kind = rng.choice(["prop", "prop", "chisq", "chisqY"])
    if kind == "prop":
        pr = H.prop_test(M, correct=False)
        lines.append("> prop.test(tab, correct = FALSE)")
        lines += H.htest(pr["method"], "tab", [("X-squared", pr["X2"]), ("df", 1)], pr["p"], "two.sided", pr["ci"], 95,
                         [("prop 1", pr["est"][0]), ("prop 2", pr["est"][1])])
        X2, pv = pr["X2"], pr["p"]
    else:
        ch = H.chisq(M, correct=(kind == "chisqY"))
        lines.append("> chisq.test(tab" + (")" if kind == "chisqY" else ", correct = FALSE)"))
        out = H.htest(ch["method"], "tab", [("X-squared", ch["X2"]), ("df", 1)], ch["p"], "")
        lines += [l for l in out if not l.startswith("alternative hypothesis: ")]  # chisq.test has no alternative line
        X2, pv = ch["X2"], ch["p"]
    pt = pfmt(pv)
    b, d = n1 - a, n2 - cc_
    OR = (a * d) / (b * cc_)
    se = math.sqrt(1 / a + 1 / b + 1 / cc_ + 1 / d)
    ph1, ph2 = a / n1, cc_ / n2
    qs_req, qs_opt = [], []
    if c["design"] == "cc":
        qs_req.append(mc(rng, "Это исследование случай-контроль. Что из него можно оценить?",
                         "Отношение шансов (OR), но не вероятность болезни при факторе",
                         ["Вероятность болезни в каждой группе", "Относительный риск RR", "Только p-value, никаких мер эффекта"],
                         "Группы набраны по исходу: доля случаев в выборке задана исследователем. OR не зависит от того, сколько контролей набрали, поэтому его оценить можно."))
    qs_req.append(num(f"Доля «{e1}» в строке {r1}?", ph1,
                      f"{a} / ({a} + {b}) = {a} / {n1} = <b>{rua(ph1, 4)}</b>. Знаменатель — итог строки {r1}, не всё исследование.", rel=0.005))
    qs_req.append(num(f"Отношение шансов «{e1}»: {r1} против {r2}?", OR,
                      f"Шансы в {r1}: {a}/{b} = {rua(a / b, 4)}; в {r2}: {cc_}/{d} = {rua(cc_ / d, 4)}. OR = ad/bc = {a}·{d} / ({b}·{cc_}) = <b>{rua(OR, 4)}</b>. "
                      f"{'Больше 1' if OR > 1 else 'Меньше 1'}: шансы в {r1} {'выше' if OR > 1 else 'ниже'}.", rel=0.005))
    qs_opt.append(num("SE(log OR)?", se, f"√(1/a + 1/b + 1/c + 1/d) = √(1/{a} + 1/{b} + 1/{cc_} + 1/{d}) = <b>{rua(se, 4)}</b>.", rel=0.005))
    up = math.exp(math.log(OR) + 1.96 * se)
    lo = math.exp(math.log(OR) - 1.96 * se)
    qs_opt.append(num("Верхняя граница 95% ДИ для OR?", up,
                      f"Интервал строят для log OR: ln({rua(OR, 4)}) ± 1,96·{rua(se, 4)} = ({rua(math.log(OR) - 1.96 * se, 4)}; {rua(math.log(OR) + 1.96 * se, 4)}). "
                      f"Потом e: ({rua(lo, 4)}; <b>{rua(up, 4)}</b>). Интервал несимметричен вокруг OR — так и должно быть.", rel=0.01))
    qs_opt.append(num(f"OR наоборот: {r2} против {r1}?", 1 / OR,
                      f"Перевернул строки — получил 1/OR = 1 / {rua(OR, 4)} = <b>{rua(1 / OR, 4)}</b>, интервал (1/U; 1/L).", rel=0.005))
    if kind == "prop":
        lo_d, hi_d = pr["ci"]
        inside = lo_d < 0 < hi_d
        qs_opt.append(mc(rng, f"prop.test: 95% ДИ ({rf(lo_d, 4)}; {rf(hi_d, 4)}). Для чего он?",
                         f"Для разности долей «{e1}»: {r1} − {r2}" + (" — 0 внутри, разница может быть нулевой" if inside else " — 0 не входит"),
                         [f"Для отношения шансов {r1} против {r2}", f"Для доли «{e1}» в {r1}", "Для X²"],
                         f"prop.test берёт первый столбец как «успех» и печатает ДИ для prop 1 − prop 2 = {rua(ph1, 4)} − {rua(ph2, 4)} = {rua(ph1 - ph2, 4)}. "
                         "Для разности ноль = нет эффекта, для OR — единица."))
    if kind in ("prop", "chisq"):
        z = math.sqrt(X2)
        qs_opt.append(num("z теста двух долей (без поправки)?", z,
                          f"Для 2×2 X² = z². z = √{f5(X2)} = <b>{rua(z, 4)}</b> (знак — по направлению разности долей).", rel=0.005, absval=True))
    else:
        qs_opt.append(mc(rng, "Почему в названии «with Yates' continuity correction»?", "chisq.test для 2×2 по умолчанию делает поправку Йейтса",
                         ["Ожидаемые частоты меньше 5", "Таблица парная", "Так считается точный тест"],
                         "Поправка немного уменьшает X² и увеличивает p. Курс просит correct = FALSE."))
    qs_opt.append(mc(rng, "Какая H₀ у этого теста для 2×2?", f"Доли «{e1}» в двух строках равны (⇔ OR = 1 ⇔ независимость)",
                     ["OR = 0", f"Доля «{e1}» равна 0,5", "Строки и столбцы связаны"],
                     "Для 2×2 все три формулировки — одна гипотеза. Направление X² не показывает: смотри на доли или OR."))
    qs_opt.append(evid_q(rng, pv, pt, "доли различаются"))
    if c["design"] != "cc":
        qs_opt.append(num(f"Относительный риск RR: {r1} против {r2}?", ph1 / ph2,
                          f"RR = p₁/p₂ = {rua(ph1, 4)} / {rua(ph2, 4)} = <b>{rua(ph1 / ph2, 4)}</b>. Сравни с OR = {rua(OR, 4)}: близки, только если событие редкое.",
                          rel=0.01))
        qs_opt.append(causal_q(rng, c))
    qs = pick(rng, qs_req, qs_opt, 5)
    intro = f"<p>{c['what']}</p><p>Строки — группы ({r1}, {r2}), первый столбец — событие ({e1}).</p><p class='dz'>{design_txt(c)}</p>"
    return dict(cat="cat", sub="c22", title=f"2×2: {r1}/{r2} × {e1}/{e0}", intro=intro, console="\n".join(lines), qs=qs)


CRC = [
    dict(rows=["Small", "Medium", "Large"], cols=["Pos", "Neg"], p=[[0.2, 0.8], [0.32, 0.68], [0.45, 0.55]], n=(150, 400), kind="ind",
         what="Случайная выборка ферм региона: размер фермы и статус по PRRS (Pos/Neg)."),
    dict(rows=["Crate", "Pen", "Outdoor"], cols=["None", "Mild", "Severe"], p=[[0.5, 0.3, 0.2], [0.62, 0.28, 0.10], [0.7, 0.24, 0.06]], n=(80, 150), kind="hom",
         what="Взяли по одинаковому числу свиноматок из каждой системы содержания и оценили хромоту."),
    dict(rows=["Duroc", "Landrace", "Yorkshire"], cols=["Bitten", "Intact"], p=[[0.18, 0.82], [0.10, 0.90], [0.12, 0.88]], n=(150, 400), kind="hom",
         what="Взяли по одинаковому числу свиней каждой породы и осмотрели хвосты."),
    dict(rows=["Farrow", "Nursery", "Finish"], cols=["Low", "Medium", "High"], p=[[0.6, 0.3, 0.1], [0.45, 0.35, 0.2], [0.35, 0.35, 0.3]], n=(200, 450), kind="ind",
         what="Случайная выборка ферм: тип производства и уровень антител к сальмонелле (Low/Medium/High)."),
    dict(rows=["North", "South", "East", "West"], cols=["Pos", "Neg"], p=[[0.08, 0.92], [0.15, 0.85], [0.1, 0.9], [0.2, 0.8]], n=(300, 600), kind="ind",
         what="Пробы с ферм страны: регион и результат ПЦР на цирковирус."),
]


def gen_crc(rng, c):
    k = len(c["rows"])
    f = rng.choice([0.0, 0.5, 1.0, 1.0])
    P = np.array(c["p"], float)
    Pm = P.mean(0)
    P = Pm + (P - Pm) * f
    if c["kind"] == "hom":
        nr = int(rng.integers(*c["n"]))
        M = np.array([rng.multinomial(nr, p / p.sum()) for p in P])
    else:
        N = int(rng.integers(*c["n"]))
        rw = rng.dirichlet(np.ones(k) * 6)
        M = np.array([rng.multinomial(int(max(20, N * w)), p / p.sum()) for w, p in zip(rw, P)])
    ch = H.chisq(M, correct=False)
    lines = tab_lines("tab", c["rows"], c["cols"], M)
    lines.append("> res <- chisq.test(tab)")
    if ch["small"]:
        lines += ["Warning message:", "In chisq.test(tab) : Chi-squared approximation may be incorrect"]
    lines.append("> res")
    out = H.htest(ch["method"], "tab", [("X-squared", ch["X2"]), ("df", ch["df"])], ch["p"], "")
    lines += [l for l in out if not l.startswith("alternative hypothesis: ")]
    show_e = rng.random() < 0.6
    if show_e:
        lines.append("> round(res$expected, 1)")
        lines += R.matrix(c["rows"], c["cols"], [list(np.round(ch["E"][:, j], 1)) for j in range(M.shape[1])])
    show_r = rng.random() < 0.6
    if show_r:
        lines.append("> round(res$residuals^2, 2)")
        lines += R.matrix(c["rows"], c["cols"], [list(np.round(ch["res2"][:, j], 2)) for j in range(M.shape[1])])
    pt = pfmt(ch["p"])
    I, J = M.shape
    qs_req, qs_opt = [], []
    qs_req.append(num("Степени свободы — проверь сам.", (I - 1) * (J - 1),
                      f"df = (строк − 1)(столбцов − 1) = ({I} − 1)({J} − 1) = <b>{(I - 1) * (J - 1)}</b>. Итоги строк и столбцов известны, свободных клеток столько.", rel=0))
    i, j = int(rng.integers(0, I)), int(rng.integers(0, J))
    E = M[i].sum() * M[:, j].sum() / M.sum()
    qs_req.append(num(f"Ожидаемая частота клетки {c['rows'][i]} × {c['cols'][j]} при H₀?", E,
                      f"E = итог строки × итог столбца / n = {M[i].sum()} × {M[:, j].sum()} / {M.sum()} = <b>{rua(E, 4)}</b>.", rel=0.005))
    hom = c["kind"] == "hom"
    qs_opt.append(mc(rng, "Как сформулировать H₀ по дизайну этого исследования?",
                     ("Распределение " + "/".join(c["cols"]) + " одинаково во всех группах (однородность)") if hom else
                     "Строковая и столбцовая переменные независимы (независимость)",
                     (["Переменные независимы: выборка одна", "Все клетки равны", "Средние групп равны"] if hom else
                      ["Распределение одинаково в группах с заранее заданным n", "Все клетки равны", "Доли в первом столбце равны 0,5"]),
                     ("Итоги строк заданы дизайном (по одинаковому числу в группе) — это несколько выборок, тест однородности." if hom else
                      "Одна выборка, обе переменные измерены у каждого — тест независимости.") +
                     " E, X², df и p при этом одинаковые."))
    if show_r:
        rr = ch["res2"]
        ii, jj = np.unravel_index(np.argmax(rr), rr.shape)
        cells = [(a_, b_) for a_ in range(I) for b_ in range(J) if (a_, b_) != (ii, jj)]
        rng.shuffle(cells)
        wr = [f"{c['rows'][a_]} × {c['cols'][b_]}" for a_, b_ in cells[:3]]
        more = "больше" if M[ii, jj] > ch["E"][ii, jj] else "меньше"
        qs_opt.append(mc(rng, "Какая клетка больше всех вкладывает в X²?", f"{c['rows'][ii]} × {c['cols'][jj]}", wr,
                         f"residuals² = (O − E)²/E — вклад клетки. Максимум {rua(rr[ii, jj], 3)} у {c['rows'][ii]} × {c['cols'][jj]}: "
                         f"наблюдали {M[ii, jj]}, это {more} ожидаемого {rua(ch['E'][ii, jj], 3)}."))
    qs_opt.append(mc(rng, "Когда p из хи-квадрата можно доверять?", "Когда все ожидаемые частоты не меньше 5",
                     ["Когда все наблюдаемые частоты не меньше 5", "Когда n больше 30", "Когда таблица 2×2"],
                     "Правило касается <b>ожидаемых</b> частот. Если оно нарушено, R предупреждает «Chi-squared approximation may be incorrect»: "
                     "объедини редкие категории или считай точный p (Фишер)." + (" Здесь предупреждение есть." if ch["small"] else "")))
    qs_opt.append(evid_q(rng, ch["p"], pt, "связь / различие распределений"))
    pr_ = M[i, j] / M[i].sum()
    qs_opt.append(num(f"Условная доля: какая часть строки {c['rows'][i]} в столбце {c['cols'][j]}?", pr_,
                      f"{M[i, j]} / {M[i].sum()} = <b>{rua(pr_, 4)}</b>. Сравнивая такие доли по строкам, видно, где связь.", rel=0.005))
    qs_opt.append(mc(rng, "Показывает ли X² направление связи?", "Нет: X² — сумма квадратов, направление смотри по долям и вкладам клеток",
                     ["Да: положительный X² — прямая связь", "Да, через знак p", "Да, если df = 1"],
                     "Где наблюдаемое больше ожидаемого, а где меньше, видно по таблице и residuals (без квадрата — со знаком)."))
    qs = pick(rng, qs_req, qs_opt, 5)
    intro = f"<p>{c['what']}</p><p>Таблица {I}×{J}, всего {int(M.sum())}.</p>"
    return dict(cat="cat", sub="crc", title=f"Хи-квадрат {I}×{J}", intro=intro, console="\n".join(lines), qs=qs)


FISH = [
    dict(rows=["Treated", "Control"], cols=["Pos", "Neg"], p=(0.08, 0.45), n=(8, 16), what="Загоны случайно обработали (Treated) или нет; через месяц — смыв на сальмонеллу (Pos/Neg).", design="exp"),
    dict(rows=["Vacc", "Control"], cols=["Abort", "None"], p=(0.04, 0.25), n=(10, 22), what="Свиноматок случайно вакцинировали или нет; исход — аборт.", design="exp"),
    dict(rows=["Imported", "Local"], cols=["ASFpos", "ASFneg"], p=(0.25, 0.03), n=(8, 25), what="Партии свиней: завезённые и местные; результат ПЦР на АЧС.", design="obs"),
    dict(rows=["New", "Old"], cols=["Lame", "Sound"], p=(0.1, 0.4), n=(8, 18), what="Свиноматки в новом и старом корпусе: хромота при осмотре.", design="obs"),
]


def gen_fish(rng, c):
    for _ in range(200):
        n1, n2 = int(rng.integers(*c["n"])), int(rng.integers(*c["n"]))
        f = rng.choice([0.0, 0.7, 1.0, 1.0, 1.2])
        p1, p2 = c["p"]
        mid = (p1 + p2) / 2
        a = int(rng.binomial(n1, mid + (p1 - mid) * f))
        cc_ = int(rng.binomial(n2, mid + (p2 - mid) * f))
        M = np.array([[a, n1 - a], [cc_, n2 - cc_]])
        ch = H.chisq(M, correct=True)
        k_, m_, n_ = M[0].sum(), M[:, 0].sum(), M[:, 1].sum()
        if ch["small"] and max(0, k_ - n_) < a < min(k_, m_):  # finite, non-zero conditional OR
            break
    fi = H.fisher(M)
    r1, r2 = c["rows"]
    e1 = c["cols"][0]
    lines = tab_lines("tab", c["rows"], c["cols"], M)
    lines.append("> chisq.test(tab)")
    out = H.htest(ch["method"], "tab", [("X-squared", ch["X2"]), ("df", 1)], ch["p"], "")
    lines += [l for l in out if not l.startswith("alternative hypothesis: ")][:-1]
    lines += ["", "Warning message:", "In chisq.test(tab) : Chi-squared approximation may be incorrect", "> fisher.test(tab)"]
    est = float(fi["est"])
    ci = tuple(float(v) for v in fi["ci"])
    lines += H.htest("Fisher's Exact Test for Count Data", "tab", [], fi["p"], "true odds ratio is not equal to 1",
                     ci, 95, [("odds ratio", est)])
    pt = pfmt(fi["p"])
    E = ch["E"]
    a, b, cc_, d = M.ravel()
    qs_req = [mc(rng, "Почему здесь нужен тест Фишера?", f"Есть ожидаемые частоты меньше 5 (минимум {rua(E.min(), 3)}), χ²-приближение ненадёжно",
                 ["Таблица парная", "Наблюдаемых нулей нет", "Потому что p &lt; 0,05"],
                 "R предупредил: «Chi-squared approximation may be incorrect». Фишер фиксирует итоги строк и столбцов и считает точные вероятности всех таблиц с такими итогами (гипергеометрическое распределение)."),
              mc(rng, "Какая H₀ у fisher.test?", "OR = 1: строка и столбец не связаны",
                 ["OR = 0", "Все клетки равны", "Доли равны 0,5"],
                 "p — сумма вероятностей таблиц, не более вероятных, чем наблюдаемая, при H₀.")]
    qs_opt = [evid_q(rng, fi["p"], pt, "связь есть"),
              mc(rng, "Чем «odds ratio» в выводе Фишера отличается от ad/bc?", "Это условная оценка максимального правдоподобия, она немного отличается от ad/bc",
                 ["Ничем, это ad/bc", "Это относительный риск", "Это обратный OR"],
                 "fisher.test печатает условную MLE отношения шансов. Для больших таблиц она почти равна ad/bc, для маленьких — заметно ближе к 1.")]
    if min(a, b, cc_, d) > 0:
        orr = a * d / (b * cc_)
        qs_opt.append(num(f"Выборочное OR (ad/bc) для «{e1}»: {r1} против {r2}?", orr,
                          f"OR = {a}·{d} / ({b}·{cc_}) = <b>{rua(orr, 4)}</b>. fisher.test печатает близкое, но другое число — условную MLE.", rel=0.005))
    if math.isfinite(ci[1]) and math.isfinite(ci[0]):
        inside = ci[0] < 1 < ci[1]
        qs_opt.append(mc(rng, f"ДИ для OR ({rf(ci[0], 4)}; {rf(ci[1], 4)}). Вывод?",
                         "1 внутри — нет доказательств связи" if inside else "1 снаружи — связь есть, " + ("шансы в " + r1 + " ниже" if ci[1] < 1 else "шансы в " + r1 + " выше"),
                         (["1 снаружи — связь есть", "Интервал слишком широкий, тест неверен", "Нужно смотреть, внутри ли 0"] if inside else
                          ["1 внутри — связи нет", "Нужно смотреть, внутри ли 0", "Интервал ничего не говорит"]),
                         "Для OR «нет связи» — это 1, не 0. Широкий интервал — плата за маленькую выборку."))
    qs_opt.append(num(f"Минимальная ожидаемая частота (итог строки × итог столбца / n)?", E.min(),
                      f"Клетка с наименьшими итогами: {int(M.sum(1)[np.unravel_index(E.argmin(), E.shape)[0]])} × {int(M.sum(0)[np.unravel_index(E.argmin(), E.shape)[1]])} / {int(M.sum())} = <b>{rua(E.min(), 3)}</b> &lt; 5.",
                      rel=0.01))
    qs = pick(rng, qs_req, qs_opt, 5)
    intro = f"<p>{c['what']}</p><p>Маленькая таблица 2×2, всего {int(M.sum())}.</p><p class='dz'>{design_txt(c)}</p>"
    return dict(cat="cat", sub="fish", title=f"Фишер: {r1}/{r2} × {e1}", intro=intro, console="\n".join(lines), qs=qs)


MCN = [
    dict(rows=["Pos", "Neg"], name="ELISA × PCR", what="Одних и тех же свиней проверили двумя тестами: строки — ELISA, столбцы — ПЦР.", n=(150, 600),
         p=dict(a=0.25, b=0.07, c=0.03)),
    dict(rows=["Yes", "No"], name="до × после", what="Одни и те же свиноматки: строки — серопозитивность до вакцинации, столбцы — через 2 месяца.", n=(100, 400),
         p=dict(a=0.3, b=0.04, c=0.12)),
    dict(rows=["Lame", "Sound"], name="визит 1 × визит 2", what="Одних и тех же свиноматок осмотрели дважды: строки — первый визит, столбцы — второй.", n=(100, 400),
         p=dict(a=0.12, b=0.10, c=0.05)),
    dict(rows=["Approve", "Not"], name="опрос 1 × опрос 2", what="Одни и те же владельцы ферм дважды ответили, одобряют ли программу вакцинации.", n=(300, 1600),
         p=dict(a=0.5, b=0.09, c=0.055)),
]


def gen_mcn(rng, c):
    n = int(rng.integers(*c["n"]))
    f = rng.choice([0.0, 0.6, 1.0, 1.0, 1.3])
    pa, pb, pc_ = c["p"]["a"], c["p"]["b"], c["p"]["c"]
    m_ = (pb + pc_) / 2
    pb, pc_ = m_ + (pb - m_) * f, m_ + (pc_ - m_) * f
    probs = [pa, pb, pc_, 1 - pa - pb - pc_]
    a, b, cc_, d = rng.multinomial(n, probs)
    if b + cc_ == 0:
        b = 1
    M = np.array([[a, b], [cc_, d]])
    correct = rng.random() < 0.3
    mn = H.mcnemar(M, correct)
    rows = c["rows"]
    lines = tab_lines("tab", rows, rows, M)
    lines.append("> mcnemar.test(tab" + (")" if correct else ", correct = FALSE)"))
    out = H.htest(mn["method"], "tab", [("McNemar's chi-squared", mn["X2"]), ("df", 1)], mn["p"], "")
    lines += [l for l in out if not l.startswith("alternative hypothesis: ")]
    pt = pfmt(mn["p"])
    p1 = (a + b) / n
    p2 = (a + cc_) / n
    qs_req = [mc(rng, "Какие клетки использует тест Макнемара?", f"Только несогласные: {rows[0]}→{rows[1]} ({b}) и {rows[1]}→{rows[0]} ({cc_})",
                 ["Все четыре клетки", "Только диагональ", "Итоги строк"],
                 "Те, кто ответил одинаково оба раза, ничего не говорят об изменении. При H₀ «перешедших» в обе стороны поровну."),
              num("Посчитай статистику без поправки: (b − c)² / (b + c).", (b - cc_) ** 2 / (b + cc_),
                  f"({b} − {cc_})² / ({b} + {cc_}) = <b>{rua((b - cc_) ** 2 / (b + cc_), 4)}</b>." +
                  (f" В выводе с поправкой: (|b − c| − 1)²/(b + c) = {f5(mn['X2'])}." if correct else ""), rel=0.005)]
    qs_opt = [num(f"Доля «{rows[0]}» по строкам (первое измерение)?", p1, f"({a} + {b}) / {n} = <b>{rua(p1, 4)}</b>.", rel=0.005),
              num(f"Доля «{rows[0]}» по столбцам (второе измерение)?", p2, f"({a} + {cc_}) / {n} = <b>{rua(p2, 4)}</b>.", rel=0.005),
              num("Разность долей p̂₁. − p̂.₁?", p1 - p2, f"= (b − c) / n = ({b} − {cc_}) / {n} = <b>{rua(p1 - p2, 4)}</b>.", rel=0.01, abs_=1e-5),
              mc(rng, "Почему не chisq.test или prop.test?", "Оба измерения у одних и тех же единиц: доли зависимы",
                 ["Потому что таблица 2×2", "Потому что n большое", "Потому что есть нули"],
                 "chisq.test проверяет связь строк и столбцов — здесь она очевидна (люди в основном не меняют ответ). "
                 "Вопрос другой: изменилась ли доля. На него отвечает McNemar."),
              mc(rng, "Условие для нормального приближения McNemar?", "b + c ≥ 20 (несогласных пар достаточно)",
                 ["Все ожидаемые ≥ 5", "n ≥ 30", "a + d ≥ 20"],
                 f"Здесь b + c = {b + cc_}." + (" Мало — лучше точная версия: binom.test(b, b + c)." if b + cc_ < 20 else "")),
              evid_q(rng, mn["p"], pt, "доля изменилась")]
    se = math.sqrt((b + cc_) - (b - cc_) ** 2 / n) / n
    qs_opt.append(num("SE разности долей (формула слайда 15)?", se,
                      f"SE = √[(b + c) − (b − c)²/n] / n = √[{b + cc_} − {(b - cc_) ** 2}/{n}] / {n} = <b>{rua(se, 4)}</b>.", rel=0.01))
    qs = pick(rng, qs_req, qs_opt, 5)
    intro = f"<p>{c['what']}</p><p>Парная таблица ({c['name']}), n = {n}.</p>"
    return dict(cat="cat", sub="mcn", title=f"Макнемар: {c['name']}", intro=intro, console="\n".join(lines), qs=qs)


PLAN3 = [
    ("t2p", gen_t2p, [c for c in TT2 if max(v[1] for v in c["lv"].values()) / min(v[1] for v in c["lv"].values()) < 1.3], 25),
    ("t2w", gen_t2w, TT2, 25),
    ("tpair", gen_tpair, TPAIR, 25),
    ("t1", gen_t1, T1, 15),
    ("wrs", gen_wrs, WRS, 30),
    ("wsr", gen_wsr, WSR, 15),
    ("sign", gen_sign, SIGN, 15),
    ("c22", gen_c22, C22, 30),
    ("crc", gen_crc, CRC, 20),
    ("fish", gen_fish, FISH, 15),
    ("mcn", gen_mcn, MCN, 15),
]
