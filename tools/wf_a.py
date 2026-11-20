"""Workflows, part A: comparing means (t-test, paired, Wilcoxon, one-way ANOVA)."""
import math
import numpy as np
from scipy import stats
import rfmt as R
import htest as H
from models import ols, oneway, tukey
from gen import ru, rua, mc, num, lm_summary
from wf_lib import (str_df, df_print, agg_print, named, vec1, code_pick, multi, step, p_box, p_lm, p_qq, p_xy, pr, r3)


def box_stats(v):
    """boxplot.stats: Tukey hinges (fivenum), whiskers to the most extreme point within 1.5 * IQR."""
    v = np.sort(np.asarray(v, float))
    n = len(v)
    n4 = math.floor((n + 3) / 2) / 2
    d = [1, n4, (n + 1) / 2, n + 1 - n4, n]
    f = [0.5 * (v[math.floor(x) - 1] + v[math.ceil(x) - 1]) for x in d]
    q1, med, q3 = f[1], f[2], f[3]
    c = 1.5 * (q3 - q1)
    inside = v[(v >= q1 - c) & (v <= q3 + c)]
    out = v[(v < q1 - c) | (v > q3 + c)]
    return dict(q1=q1, med=med, q3=q3, lo=float(inside.min()), hi=float(inside.max()), out=[float(x) for x in out])


def box(groups, yl=""):
    p = p_box(groups, yl=yl)
    p["st"] = [box_stats(v) for v in groups.values()]
    return p


def lm_out(formula, data, X, y, names):
    f = ols(np.asarray(X, float), np.asarray(y, float))
    lines, disp = lm_summary(formula, data, f, names)
    return lines[2:], disp, f


def ttest_lines(call, method, data_txt, r, g1, g2):
    return [f"> {call}"] + H.htest(method, data_txt, [("t", r["t"]), ("df", r["df"])], r["p"],
                                   f"true difference in means between group {g1} and group {g2} is not equal to 0", r["ci"], 95,
                                   [(f"mean in group {g1}", r["m1"]), (f"mean in group {g2}", r["m2"])])


def shuffle_rows(rng, *cols):
    o = rng.permutation(len(cols[0]))
    return [np.asarray(c)[o] for c in cols]


# =====================================================================
# 1. Two independent means: t-test
# =====================================================================
T2_VAR = [
    dict(nA=20, nB=20, mA=780, mB=832, sA=58, sB=60, kind="clear"),
    dict(nA=14, nB=14, mA=790, mB=806, sA=72, sB=70, kind="ns"),
    dict(nA=26, nB=12, mA=795, mB=840, sA=38, sB=96, kind="unequal"),
]


def wf_t2(rng, v):
    for _ in range(5000):
        a = np.round(rng.normal(v["mA"], v["sA"], v["nA"]))
        b = np.round(rng.normal(v["mB"], v["sB"], v["nB"]))
        w = H.t_two(a, b, False)
        pz = H.t_two(a, b, True)
        ratio = max(w["s1"], w["s2"]) / min(w["s1"], w["s2"])
        res = np.r_[a - a.mean(), b - b.mean()]
        shp = stats.shapiro(res).pvalue
        bs = [box_stats(a), box_stats(b)]
        blo, bhi = -w["ci"][1], -w["ci"][0]
        if v["kind"] == "clear" and w["p"] < 0.01 and ratio < 1.4 and shp > 0.3 and blo < 25 and not bs[0]["out"]:
            break
        if v["kind"] == "ns" and w["p"] > 0.3 and ratio < 1.4 and shp > 0.3 and blo < -20 and bhi > 45 and not bs[0]["out"]:
            break
        if v["kind"] == "unequal" and pz["p"] < 0.04 and w["p"] > 0.07 and ratio > 2.2 and b.mean() > a.mean():
            break
    else:
        raise RuntimeError("t2")
    diet = np.array(["A"] * len(a) + ["B"] * len(b))
    adg = np.r_[a, b]
    diet, adg = shuffle_rows(rng, diet, adg)
    n = len(adg)
    mA, mB = a.mean(), b.mean()
    sA, sB = a.std(ddof=1), b.std(ddof=1)
    diff = mB - mA
    T = 25
    S = []
    # 1. load
    S.append(step("Загрузить данные и посмотреть структуру",
                  task="Файл <code>diet_trial.csv</code> лежит в рабочей папке. Прочитай его в таблицу <code>pigs</code>, потом посмотри, что внутри.",
                  pick=code_pick(rng, "Какой командой прочитать файл?", 'pigs <- read.csv("diet_trial.csv")',
                                 ['pigs <- read.table("diet_trial.csv")', 'pigs <- load("diet_trial.csv")', 'pigs <- source("diet_trial.csv")'],
                                 "<code>read.csv</code> читает текстовую таблицу, где столбцы разделены запятой, а первая строка — имена столбцов. "
                                 "<code>read.table</code> по умолчанию ждёт пробелы и не считает первую строку заголовком; <code>load</code> — для файлов .RData; "
                                 "<code>source</code> выполняет R-скрипт."),
                  code=['> pigs <- read.csv("diet_trial.csv")', "> str(pigs)"],
                  out=str_df(n, [("pig", "int", range(1, n + 1)), ("diet", "chr", diet), ("adg", "num", adg)]),
                  qs=[num("Сколько свиней в опыте?", n, f"Первая строка <code>str</code>: «{n} obs.» — число строк. Одна строка — одна свинья, значит {n}.", rel=0),
                      mc(rng, "Столбец <code>diet</code> — текст (chr). Какой рацион R возьмёт за базовый уровень в <code>lm</code>?",
                         "A — первый по алфавиту",
                         ["B — потому что он идёт вторым", "Тот, что записан в первой строке файла", "Никакой: текст в модели использовать нельзя"],
                         "R превращает текст в фактор, уровни идут по алфавиту: A, B. Базовый — A, а коэффициент <code>dietB</code> покажет разность B − A. "
                         "Поменять базу можно так: <code>pigs$diet &lt;- relevel(factor(pigs$diet), ref = \"B\")</code>.")]))
    # 2. describe
    agg = agg_print("diet", ["A", "B"], "adg", [("n", [len(a), len(b)]), ("mean", [mA, mB]), ("sd", [sA, sB])])
    shown_sd = [round(sA, 1), round(sB, 1)]
    rs = max(shown_sd) / min(shown_sd)
    if v["kind"] == "unequal":
        verdict = ("SD различаются больше чем вдвое, и группы разного размера: t-тест с общим SD (pooled) может ошибиться. Нужен Welch",
                   ["SD почти одинаковые: любой t-тест подойдёт", "Разные SD означают, что рационы точно различаются",
                    "Надо удалить свиней с большим привесом, чтобы выровнять SD"])
    else:
        verdict = ("SD близки (r<sub>s</sub> ≤ 2): pooled и Welch дадут почти одно и то же",
                   ["SD различаются: нужен только непараметрический тест", "По SD видно, что разница средних значима",
                    "SD надо сравнивать со средними: SD больше 50 — плохо"])
    S.append(step("Описательная статистика по группам",
                  task="Посчитай для каждого рациона число свиней, среднее и SD привеса.",
                  pick=code_pick(rng, "Какой код это сделает?",
                                 "aggregate(adg ~ diet, data = pigs, FUN = function(v) round(c(n = length(v), mean = mean(v), sd = sd(v)), 1))",
                                 ["mean(pigs$adg); sd(pigs$adg)", "tapply(pigs$diet, pigs$adg, mean)", "table(pigs$diet, pigs$adg)"],
                                 "<code>aggregate(y ~ группа, data, FUN)</code> применяет функцию к y отдельно в каждой группе. "
                                 "<code>mean(pigs$adg)</code> — одно общее среднее; в <code>tapply</code> первым идёт то, что усредняем, вторым — группировка; "
                                 "<code>table</code> считает частоты, а не средние."),
                  code="> aggregate(adg ~ diet, data = pigs, FUN = function(v)\n+   round(c(n = length(v), mean = mean(v), sd = sd(v)), 1))",
                  out=agg,
                  qs=[num("Отношение большего SD к меньшему (r<sub>s</sub>)?", rs,
                          f"r<sub>s</sub> = {ru(max(shown_sd), 1)} / {ru(min(shown_sd), 1)} = <b>{rua(rs, 3)}</b>. Правило курса: до 2 — равные дисперсии разумны.",
                          rel=0.02),
                      mc(rng, "Что это значит для выбора t-теста?", verdict[0], verdict[1],
                         "Pooled t-тест считает одну общую σ для обеих групп. Если SD сильно разные, а n неравные, его SE и p неверны. "
                         "Welch считает SE для каждой группы отдельно и почти ничего не теряет, когда SD равны, — поэтому в R он по умолчанию.")]))
    # 3. boxplot
    bp = box({"A": a, "B": b}, yl="adg")
    hi_g = "B" if bs[1]["med"] > bs[0]["med"] else "A"
    iqr = [bs[0]["q3"] - bs[0]["q1"], bs[1]["q3"] - bs[1]["q1"]]
    nout = len(bs[0]["out"]) + len(bs[1]["out"])
    if max(iqr) / min(iqr) > 1.8:
        wide = "B" if iqr[1] > iqr[0] else "A"
        sp = f"ящик {wide} заметно выше и шире: разброс там больше"
        sp_w = f"ящик {'A' if wide == 'B' else 'B'} шире: разброс там больше"
    else:
        sp = "ящики примерно одной высоты: разброс похож"
        sp_w = "у одной группы разброс в несколько раз больше"
    overlap = bs[1]["q1"] < bs[0]["q3"] and bs[0]["q1"] < bs[1]["q3"]
    ov = "центральные половины данных перекрываются" if overlap else "центральные половины данных не перекрываются"
    S.append(step("Посмотреть на данные",
                  task="Нарисуй ящики с усами по рационам: сразу видно уровень, разброс и выбросы.",
                  code='> boxplot(adg ~ diet, data = pigs)',
                  plots=[bp],
                  qs=[mc(rng, "Что видно на графике?", f"Медиана выше у {hi_g}; {sp}",
                         [f"Медиана выше у {'A' if hi_g == 'B' else 'B'}; {sp}", f"Медиана выше у {hi_g}; {sp_w}",
                          "Ящики показывают среднее ± SD каждой группы"],
                         f"Жирная линия — медиана, ящик — от 1-го до 3-го квартиля (половина данных), усы — до самых далёких точек в пределах 1,5 высоты ящика. "
                         f"Здесь {ov}. Ящик не показывает ни среднее, ни SD."),
                      num("Сколько точек R нарисовал отдельно, за усами?", nout,
                          "Точки за усами — те, что дальше 1,5 высоты ящика от его краёв. " +
                          (f"Здесь их <b>{nout}</b>. Это не приговор: такие точки проверяют (ошибка записи?), но не удаляют автоматически."
                           if nout else "Здесь таких точек нет: <b>0</b>."), rel=0)]))
    # 4. residuals
    X = np.c_[np.ones(n), (diet == "B").astype(float)]
    P, F = p_lm(X, adg, (1, 2))
    if v["kind"] == "unequal":
        right_d = "Правый столбик заметно шире левого: у группы B разброс больше — дисперсии разные"
        wr_d = ["Всё в порядке: ширина столбиков одинаковая", "Дуга на красной линии: нарушена линейность",
                "Точки на Q-Q лежат на прямой, значит дисперсии равны"]
        ex_d = ("У модели с одним фактором прогноз — среднее группы, поэтому точки стоят двумя столбиками. Сравнивают их высоту: здесь у B разброс в "
                f"{rua(sB / sA, 2)} раза больше. Хвосты на Q-Q загибаются по той же причине: остатки с разным разбросом, сложенные вместе, выглядят «тяжелохвостыми».")
    else:
        right_d = "Столбики похожей ширины, точки на Q-Q около прямой: предпосылки в порядке"
        wr_d = ["Правый столбик шире: дисперсии разные", "Есть дуга: нарушена линейность",
                "Подписаны три точки — это выбросы, их надо удалить"]
        ex_d = "Ширина столбиков похожа, на Q-Q нет загиба хвостов. R всегда подписывает три самые крайние точки, это не диагноз."
    S.append(step("Проверить предпосылки по остаткам",
                  task="Подгони ту же модель через <code>lm</code> и построй два главных графика остатков.",
                  code=["> m <- lm(adg ~ diet, data = pigs)", "> par(mfrow = c(1, 2))", "> plot(m, which = 1:2)"],
                  plots=P,
                  qs=[mc(rng, "Почему на Residuals vs Fitted всего два столбика точек?", "Прогноз модели — среднее своей группы, а групп две",
                         ["В данных ошибка: часть точек потерялась", "Так выглядит нарушение линейности", "R рисует только по одной точке на группу"],
                         "Fitted value для каждой свиньи — среднее её рациона. Значений прогноза всего два, поэтому точки выстраиваются в два вертикальных столбика."),
                      mc(rng, "Что с предпосылками?", right_d, wr_d, ex_d)]))
    # 5. t-test
    tl = ttest_lines("t.test(adg ~ diet, data = pigs)", "Welch Two Sample t-test", "adg by diet", w, "A", "B")
    lo, hi = -w["ci"][1], -w["ci"][0]
    if w["p"] < 0.05:
        concl = f"Средний привес на B выше, чем на A; разница от {rua(lo, 3)} до {rua(hi, 3)} г/сут правдоподобна"
    else:
        concl = f"Разница не доказана: ДИ для B − A от {rua(lo, 3)} до {rua(hi, 3)} г/сут включает 0"
    S.append(step("Сравнить средние t-тестом",
                  task="Сравни средние. В R по умолчанию — t-тест Уэлча.",
                  pick=code_pick(rng, "Какой код?", "t.test(adg ~ diet, data = pigs)",
                                 ["t.test(adg ~ diet, data = pigs, paired = TRUE)", "t.test(pigs$adg, mu = 0)", "t.test(diet ~ adg, data = pigs)"],
                                 "Формула <code>y ~ группа</code>: слева отклик, справа группы. <code>paired = TRUE</code> — для пар (одно животное дважды), здесь пар нет. "
                                 "<code>mu = 0</code> — одна выборка против нуля."),
                  code=tl[0], out=tl[1:],
                  qs=[num("Разность средних B − A, г/сут?", diff,
                          f"R печатает средние A = {R.r7(mA)} и B = {R.r7(mB)}, а ДИ считает для A − B (первая группа минус вторая). "
                          f"B − A = <b>{rua(diff, 4)}</b> г/сут; ДИ для B − A — тот же интервал с обратными знаками: ({rua(lo, 4)}; {rua(hi, 4)}).", rel=0.01),
                      mc(rng, "Какой вывод верный?", concl,
                         [f"p = {pr(w['p'])} — это вероятность того, что рационы одинаковы",
                          "95% свиней на рационе B растут быстрее, чем на A",
                          "Раз p считается по t, разница в граммах неважна"],
                         f"p — вероятность получить такую или бо́льшую разницу, <i>если бы</i> рационы были одинаковы. Здесь p = {pr(w['p'])}. "
                         "ДИ говорит о разнице <b>средних</b>, а не об отдельных свиньях.")]))
    if v["kind"] == "unequal":
        tp = ttest_lines("t.test(adg ~ diet, data = pigs, var.equal = TRUE)", " Two Sample t-test", "adg by diet", pz, "A", "B")
        tp = [tp[0]] + [l.replace("\t ", "\t") for l in tp[1:]]
        S.append(step("Сравнить с t-тестом с общим SD",
                      task="Для сравнения запусти pooled t-тест (<code>var.equal = TRUE</code>).",
                      code=tp[0], out=tp[1:],
                      qs=[mc(rng, f"Pooled даёт p = {pr(pz['p'])}, Welch — p = {pr(w['p'])}. Какому верить?",
                             "Уэлчу: SD разные, а в маленькой группе разброс больше — pooled занижает SE",
                             ["Pooled: у него больше df, значит он точнее", "Тому, где p меньше 0,05 — так мы нашли эффект",
                              "Никакому: надо удалить свиней с большим привесом и пересчитать"],
                             f"Pooled усредняет дисперсии с весами по n. Большая группа A (n = {len(a)}) с маленьким SD тянет общую σ вниз, "
                             f"а группа B (n = {len(b)}) на самом деле разбросана сильно. SE получается слишком маленьким, p — слишком оптимистичным. "
                             "Выбирать тест по тому, где p меньше, — p-hacking."),
                          num("Насколько больше df у pooled, чем у Welch?", pz["df"] - w["df"],
                              f"df pooled = n₁ + n₂ − 2 = {len(a) + len(b) - 2}; df Welch = {H.fmt5(w['df'])}. Разница <b>{rua(pz['df'] - w['df'], 3)}</b>: "
                              "Welch «теряет» df, когда дисперсии сильно разные, — это честная плата за неопределённость.", rel=0.02)]))
    # 6. lm check
    ll, disp, f = lm_out("adg ~ diet", "pigs", X, adg, ["(Intercept)", "dietB"])
    S.append(step("Перепроверить через lm",
                  task="Та же задача — это регрессия с фактором. Посмотри <code>summary(m)</code> и найди в нём разность средних.",
                  code="> summary(m)", out=ll,
                  qs=[mc(rng, "Какое число в <code>summary(m)</code> — разность средних B − A?", "Estimate в строке dietB",
                         ["Estimate в строке (Intercept)", "Residual standard error", "Multiple R-squared"],
                         f"(Intercept) = среднее базового уровня A ({R.r7(mA)}), dietB = B − A = {rua(diff, 4)}. Это то же, что в t-тесте."),
                      mc(rng, "|t value| у dietB совпадает с t какого теста?", "t-теста с общим SD (var.equal = TRUE)",
                         ["t-теста Уэлча", "Парного t-теста", "Ни с каким: lm считает по-другому"],
                         "lm оценивает одну общую σ (Residual standard error) для всех свиней — ровно как pooled t-тест. "
                         + ("Поэтому здесь lm повторяет ошибку pooled: SD групп разные." if v["kind"] == "unequal" else "Welch даст почти то же, раз SD близки."))]))
    # 7. practical
    pct = 100 * diff / mA
    lo_w, hi_w = lo, hi
    if lo_w > T:
        pay = (f"Да: весь ДИ ({rua(lo_w, 3)}; {rua(hi_w, 3)}) выше порога {T} г/сут", "весь интервал выше порога")
    elif hi_w < T:
        pay = (f"Скорее нет: весь ДИ ({rua(lo_w, 3)}; {rua(hi_w, 3)}) ниже порога {T} г/сут", "весь интервал ниже порога")
    elif lo_w > 0:
        pay = (f"Не наверняка: эффект есть (ДИ выше 0), но нижняя граница {rua(lo_w, 3)} — ниже порога {T}", "эффект доказан, окупаемость — нет")
    else:
        pay = (f"Неизвестно: ДИ ({rua(lo_w, 3)}; {rua(hi_w, 3)}) накрывает и 0, и порог {T}", "данных мало для ответа")
    pay_w = [f"Да, раз p {'&lt;' if w['p'] < 0.05 else '&gt;'} 0,05", "Нет: проценты меньше 10% никогда не окупаются",
             f"Да: оценка {rua(diff, 3)} больше {T}, остальное неважно" if diff > T else f"Нет: оценка {rua(diff, 3)} меньше {T}, остальное неважно"]
    S.append(step("Практический смысл: окупится ли добавка",
                  task=f"Экономист посчитал: фермент окупается, если привес выше хотя бы на {T} г/сут. Переведи разницу в проценты и сравни ДИ с порогом.",
                  code="> mm <- tapply(pigs$adg, pigs$diet, mean)\n> 100 * (mm[\"B\"] - mm[\"A\"]) / mm[\"A\"]",
                  out=named(["B"], [pct]),
                  qs=[num("На сколько процентов привес на B выше, чем на A?", pct,
                          f"100 × (B − A) / A = 100 × {rua(diff, 4)} / {R.r7(mA)} = <b>{rua(pct, 3)}</b>%. Имя «B» у результата R взял от первого вектора.", rel=0.02),
                      mc(rng, "Окупится ли фермент?", pay[0], pay_w,
                         f"Порог сравнивают с <b>интервалом</b>, а не с одной оценкой и не с p: {pay[1]}. Статистическая значимость отвечает на вопрос «есть ли разница вообще», "
                         "а экономике нужна величина разницы.")]))
    # 8. report
    tw, dfw = H.fmt5(-w["t"]).replace(".", ",").replace("-", "−"), H.fmt5(w["df"]).replace(".", ",")
    rep = (f"Привес на рационе B был {'выше' if diff > 0 else 'ниже'} на {rua(abs(diff), 3)} г/сут (95% ДИ от {rua(lo, 3)} до {rua(hi, 3)}; "
           f"t-тест Уэлча, t = {tw}, df = {dfw}, p = {pr(w['p'])})")
    S.append(step("Записать результат",
                  task="Выбери фразу для отчёта: оценка, интервал, тест, p.",
                  qs=[mc(rng, "Как правильно записать результат?", rep,
                         [f"Рацион B повышает привес на 95% (p = {pr(w['p'])})",
                          f"С вероятностью {ru(100 * (1 - w['p']), 1)}% рацион B лучше рациона A",
                          (f"Рационы не различаются (p = {pr(w['p'])}), значит фермент не работает" if w["p"] >= 0.05 else
                           f"Рацион B лучше рациона A у каждой свиньи (p = {pr(w['p'])})")],
                         "В отчёте — величина эффекта в единицах измерения, 95% ДИ, название теста и p. «95%» — уровень доверия интервала, а не размер эффекта; "
                         "1 − p — не вероятность того, что гипотеза верна; p &gt; 0,05 — не доказательство отсутствия эффекта.")]))
    # 9. checklist
    items = [("Рацион назначали каждой свинье жребием, свиньи в отдельных станках: строки независимы", True),
             ("SD групп различаются не больше чем вдвое" + (" — проверено" if v["kind"] != "unequal" else ""), v["kind"] != "unequal"),
             ("Остатки без явных выбросов и сильной асимметрии", True),
             ("Нужен парный t-тест: свиньи сравнивались попарно", False),
             ("Можно говорить о причине: рационы назначали случайно", True),
             ("Вывод верен для любой фермы и любого возраста свиней", False)]
    S.append(step("Проверка: анализ сделан правильно?",
                  task="Последний шаг любого анализа — пройти по списку и честно отметить, что выполнено.",
                  qs=[multi(rng, "Отметь всё, что верно для этого анализа.", items,
                            "Независимость — по дизайну (что назначали жребием). Равенство дисперсий — по SD и графику остатков"
                            + (": здесь оно нарушено, поэтому Welch." if v["kind"] == "unequal" else ".") +
                            " Причинный вывод законен в опыте с жребием, но переносится только на похожих свиней и условия.")]))
    names = {"clear": "разница есть", "ns": "разница не доказана", "unequal": "разные дисперсии"}
    return dict(g="means", key="t2", title="Два рациона: t-тест", vt=names[v["kind"]],
                goal=(f"<p><b>Вопрос.</b> Повышает ли кормовой фермент среднесуточный привес? Опыт: {n} свиней в индивидуальных станках, "
                      f"рацион A (контроль) или B (с ферментом) назначали жребием. Отклик <code>adg</code> — привес за опыт, г/сут.</p>"),
                steps=S)


# =====================================================================
# 2. Paired: backfat at farrowing and at weaning
# =====================================================================
PAIR_VAR = [dict(n=16, loss=2.1, sd_d=1.6, kind="normal"), dict(n=13, loss=1.5, sd_d=1.0, kind="outlier")]


def wf_paired(rng, v):
    for _ in range(5000):
        n = v["n"]
        far = np.round(rng.normal(19, 3.0, n), 1)
        d = rng.normal(-v["loss"], v["sd_d"], n)
        if v["kind"] == "outlier":
            d[int(rng.integers(n))] = rng.uniform(6.5, 8.5)
        wean = np.round(far + d, 1)
        dd = np.round(wean - far, 1)
        r = H.t_one(dd)
        tw = H.t_two(wean, far, True)
        sr = H.signed_rank(wean, far)
        shp = stats.shapiro(dd).pvalue
        if np.any(dd == 0) or (v["kind"] == "outlier" and len(np.unique(np.abs(dd))) < n):
            continue
        if v["kind"] == "normal" and r["p"] < 0.001 and tw["p"] > 0.02 and shp > 0.3:
            break
        if v["kind"] == "outlier" and shp < 0.01 and r["p"] > 0.06 and sr["p"] < 0.03:
            break
    else:
        raise RuntimeError("paired")
    sow = np.arange(1, n + 1)
    S = []
    S.append(step("Загрузить данные",
                  task="Таблица <code>sows</code>: одна строка — одна свиноматка, толщина шпика (мм) при опоросе и при отъёме.",
                  code=['> sows <- read.csv("backfat.csv")', "> head(sows)"],
                  out=df_print([("sow", "int", sow), ("bf_farrow", "num", far), ("bf_wean", "num", wean)], rows=list(range(6))),
                  qs=[mc(rng, "Почему здесь нужен парный анализ?", "Каждую свиноматку измерили дважды: два числа в строке связаны",
                         ["Потому что свиноматок чётное число", "Потому что измерения в миллиметрах", "Парный нужен всегда, когда групп две"],
                         "Свиноматка с толстым шпиком при опоросе обычно остаётся «толстой» и при отъёме. Парный анализ сравнивает каждую саму с собой "
                         "и убирает эту разницу между свиноматками из ошибки.")]))
    S.append(step("Посчитать разности",
                  task="Создай вектор разностей «отъём − опорос» и посмотри на его среднее и SD.",
                  pick=code_pick(rng, "Как создать разности?", "d <- sows$bf_wean - sows$bf_farrow",
                                 ["d <- mean(sows$bf_wean) - mean(sows$bf_farrow)", "d <- sows$bf_wean / sows$bf_farrow", "d <- c(sows$bf_wean, sows$bf_farrow)"],
                                 "Вычитание векторов идёт поэлементно: для каждой свиноматки своя разность. "
                                 "Разность средних — одно число, а нам нужны n разностей; c() склеивает столбцы в один длинный вектор."),
                  code=["> d <- sows$bf_wean - sows$bf_farrow", "> round(c(n = length(d), mean = mean(d), sd = sd(d)), 2)"],
                  out=named(["n", "mean", "sd"], [n, round(dd.mean(), 2), round(dd.std(ddof=1), 2)]),
                  qs=[num("Сколько мм шпика в среднем теряет свиноматка за лактацию?", -dd.mean(),
                          f"Среднее разностей = {ru(dd.mean(), 2)} мм: минус значит «при отъёме тоньше». Потеря = <b>{ru(-dd.mean(), 2)}</b> мм.", rel=0.02, abs_=0.01)]))
    S.append(step("Проверить нормальность разностей",
                  task="Для парного t-теста нормальными должны быть <b>разности</b>, а не сами измерения.",
                  code=["> qqnorm(d); qqline(d)"],
                  plots=[p_qq(dd, yl="Sample Quantiles")],
                  qs=[mc(rng, "Что показывает график?",
                         ("Одна разность далеко вверху: у одной свиноматки шпик будто вырос на много мм" if v["kind"] == "outlier" else "Точки около прямой: разности примерно нормальные"),
                         (["Точки около прямой: всё в порядке", "Правый хвост загибается вверх: асимметрия", "Точки идут ступеньками: данные дискретные"]
                          if v["kind"] == "outlier" else
                          ["Одна точка далеко: выброс", "Хвосты загибаются: тяжёлые хвосты", "Нужно проверять нормальность опороса и отъёма отдельно"]),
                         ("Самая верхняя точка сильно отстоит от прямой: у одной свиноматки шпик «вырос» на " + ru(dd.max(), 1) +
                          " мм за лактацию. Так почти не бывает — проверь запись (перепутали строки? ошибка датчика?) и проверь вывод тестом по рангам."
                          if v["kind"] == "outlier" else
                          "Немного неровностей при n = " + str(n) + " бывает всегда. Явного загиба или далёких точек нет."))]))
    tl = [f"> t.test(sows$bf_wean, sows$bf_farrow, paired = TRUE)"] + H.htest(
        "Paired t-test", "sows$bf_wean and sows$bf_farrow", [("t", r["t"]), ("df", r["df"])], r["p"],
        "true mean difference is not equal to 0", r["ci"], 95, [("mean difference", r["m"])])
    S.append(step("Парный t-тест",
                  task="Сравни шпик при отъёме и при опоросе парным t-тестом.",
                  pick=code_pick(rng, "Какой код?", "t.test(sows$bf_wean, sows$bf_farrow, paired = TRUE)",
                                 ["t.test(sows$bf_wean, sows$bf_farrow)", "t.test(bf_wean ~ bf_farrow, data = sows)", "cor.test(sows$bf_wean, sows$bf_farrow)"],
                                 "<code>paired = TRUE</code> велит вычитать попарно. Без него R сделает тест Уэлча для двух независимых групп. "
                                 "<code>cor.test</code> проверяет связь, а не разницу."),
                  code=tl[0], out=tl[1:],
                  qs=[num("df парного теста?", r["df"], f"df = число пар − 1 = {n} − 1 = <b>{n - 1}</b>.", rel=0),
                      mc(rng, "Как прочитать 95% ДИ?",
                         f"Средняя потеря шпика от {ru(-r['ci'][1], 2)} до {ru(-r['ci'][0], 2)} мм за лактацию",
                         ["95% свиноматок теряют от " + ru(-r['ci'][1], 2) + " до " + ru(-r['ci'][0], 2) + " мм",
                          "Шпик при отъёме лежит в этом интервале", "Интервал не нужен, раз p маленькое"],
                         "Это интервал для <b>средней</b> разности, а не для отдельных свиноматок. Отдельные свиноматки разбросаны намного шире (SD разностей "
                         f"{ru(dd.std(ddof=1), 2)} мм).")]))
    if v["kind"] == "normal":
        tl2 = ["> t.test(sows$bf_wean, sows$bf_farrow)"] + H.htest(
            "Welch Two Sample t-test", "sows$bf_wean and sows$bf_farrow", [("t", tw["t"]), ("df", H.t_two(wean, far, False)["df"])],
            H.t_two(wean, far, False)["p"], "true difference in means is not equal to 0", H.t_two(wean, far, False)["ci"], 95,
            [("mean of x", wean.mean()), ("mean of y", far.mean())])
        tw2 = H.t_two(wean, far, False)
        S.append(step("Что было бы без пар",
                      task="Ошибка, которую часто делают: забыть <code>paired = TRUE</code>.",
                      code=tl2[0], out=tl2[1:],
                      qs=[mc(rng, f"Без пар p = {pr(tw2['p'])}, с парами p = {pr(r['p'])}. Почему такая разница?",
                             "Без пар в ошибку попадает разница между свиноматками — она намного больше, чем изменение за лактацию",
                             ["Без пар больше df, поэтому тест слабее", "Потому что средние в двух тестах разные",
                              "Парный тест всегда даёт p в 100 раз меньше"],
                             f"Средняя разность та же ({ru(dd.mean(), 2)} мм), меняется SE. SD шпика между свиноматками около {ru(far.std(ddof=1), 1)} мм, "
                             f"а SD разностей — {ru(dd.std(ddof=1), 2)} мм. Парный дизайн убирает первое из ошибки.")]))
    else:
        sl = ["> wilcox.test(sows$bf_wean, sows$bf_farrow, paired = TRUE)"] + H.htest(
            sr["method"], "sows$bf_wean and sows$bf_farrow", [("V", sr["V"])], sr["p"], "true location shift is not equal to 0")
        S.append(step("Проверить знаково-ранговым тестом",
                      task="Из-за выброса проверь вывод тестом, который смотрит на ранги, а не на сами числа.",
                      code=sl[0], out=sl[1:],
                      qs=[mc(rng, f"t-тест: p = {pr(r['p'])}; знаково-ранговый: p = {pr(sr['p'])}. Почему они расходятся?",
                             "Одна странная свиноматка раздувает SD разностей, и t-тест теряет эффект; ранговому тесту важен только её ранг",
                             ["Знаково-ранговый тест всегда сильнее t-теста", "t-тест посчитан с ошибкой: df неверные",
                              "Тесты проверяют разные данные"],
                             f"SD разностей с этой свиноматкой {ru(dd.std(ddof=1), 2)} мм, без неё — {ru(np.delete(dd, int(np.argmax(dd))).std(ddof=1), 2)} мм. "
                             "Что делать: проверить её запись, показать анализ с ней и без неё. Выбирать тест по меньшему p нельзя."),
                          num("Чему равно V (сумма рангов положительных разностей)?", sr["V"],
                              f"V — сумма рангов |d| у разностей больше нуля (где шпик «вырос»). Здесь такая разность {int((dd > 0).sum())}, "
                              f"и у подозрительной свиноматки самый большой ранг {n}: V = <b>{H.fmt5(sr['V'])}</b>.", rel=0)]))
    rep_ok = (f"Свиноматки потеряли в среднем {ru(-dd.mean(), 1)} мм шпика за лактацию (95% ДИ {ru(-r['ci'][1], 1)}–{ru(-r['ci'][0], 1)} мм; "
              f"парный t-тест, t = {H.fmt5(r['t']).replace('.', ',').replace('-', '−')}, df = {n - 1}, p = {pr(r['p'])})")
    if v["kind"] == "outlier":
        rep_ok = (f"Медиана изменения шпика за лактацию {ru(np.median(dd), 1)} мм; знаково-ранговый тест p = {pr(sr['p'])}, парный t-тест p = {pr(r['p'])}: "
                  f"расхождение из-за одной свиноматки с ростом шпика {ru(dd.max(), 1)} мм, её запись проверяется")
    S.append(step("Записать результат",
                  qs=[mc(rng, "Как правильно записать результат?", rep_ok,
                         ["Шпик при опоросе и при отъёме различается (p &lt; 0,05), опыт подтверждает влияние корма",
                          f"У 95% свиноматок шпик уменьшился на {ru(-r['ci'][1], 1)}–{ru(-r['ci'][0], 1)} мм",
                          "Двухвыборочный t-тест показал разницу между группами опороса и отъёма"],
                         "Это наблюдение за изменением во времени: корм никто не менял, о его влиянии вывода нет. Интервал — для средней потери. "
                         "Групп здесь нет: одни и те же свиноматки дважды.")]))
    items = [("Каждая свиноматка дала пару измерений, анализ по разностям", True),
             ("Нормальность проверяли по разностям", True),
             ("Свиноматки — независимые единицы (разные животные)", True),
             ("Вывод про причину: лактация уменьшает шпик", False),
             ("Если разности не нормальные — надо брать двухвыборочный тест", False)]
    S.append(step("Проверка", qs=[multi(rng, "Отметь всё, что верно.", items,
                                        "Без контрольной группы это описание изменения, а не опыт: шпик мог меняться и по другим причинам (сезон, корм). "
                                        "При ненормальных разностях — знаково-ранговый тест или тест знаков, но пары сохраняются.")]))
    return dict(g="means", key="paired", title="До и после: парный t-тест", vt=("чистые данные" if v["kind"] == "normal" else "одна странная свиноматка"),
                goal=f"<p><b>Вопрос.</b> Сколько шпика теряет свиноматка за лактацию? У {n} свиноматок измерили толщину шпика ультразвуком при опоросе и при отъёме.</p>",
                steps=S)


# =====================================================================
# 3. Skewed outcome: Wilcoxon rank sum vs t on log scale
# =====================================================================
WRS_VAR = [dict(n1=10, n2=12, r=1.9, kind="sig"), dict(n1=9, n2=9, r=1.15, kind="ns")]


def wf_wrs(rng, v):
    for _ in range(5000):
        a = np.maximum(1, np.round(rng.lognormal(math.log(4), 0.55, v["n1"])))
        b = np.maximum(1, np.round(rng.lognormal(math.log(4 * v["r"]), 0.55, v["n2"])))
        w = H.rank_sum(a, b)
        tl = H.t_two(np.log(a), np.log(b), False)
        sk = stats.skew(np.r_[a - a.mean(), b - b.mean()])
        if v["kind"] == "sig" and w["p"] < 0.02 and tl["p"] < 0.02 and sk > 0.8 and b.max() >= 14:
            break
        if v["kind"] == "ns" and w["p"] > 0.25 and tl["p"] > 0.25 and sk > 0.8:
            break
    else:
        raise RuntimeError("wrs")
    prot = np.array(["new"] * len(a) + ["std"] * len(b))
    days = np.r_[a, b]
    prot, days = shuffle_rows(rng, prot, days)
    n = len(days)
    S = []
    S.append(step("Загрузить и посмотреть", task="Таблица <code>tr</code>: свинья, схема лечения (<code>new</code> или <code>std</code>) и число дней до выздоровления.",
                  code=['> tr <- read.csv("treat_days.csv")', "> str(tr)"],
                  out=str_df(n, [("pig", "int", range(1, n + 1)), ("prot", "chr", prot), ("days", "int", days)]),
                  qs=[mc(rng, "Почему у <code>days</code> тип int?", "Дни — целые числа: R читает столбец без дробей как целый",
                         ["Это ошибка чтения, надо as.numeric", "int означает «интервальная шкала»", "R так помечает столбцы-группы"],
                         "int — целые числа. Для моделей это обычное число. Для группировки R использует текст или фактор.")]))
    meds = [np.median(a), np.median(b)]
    S.append(step("Описать: медианы и средние", task="При асимметрии смотрят и на медиану, и на среднее.",
                  code="> aggregate(days ~ prot, data = tr, FUN = function(v)\n+   c(n = length(v), median = median(v), mean = round(mean(v), 1)))",
                  out=agg_print("prot", ["new", "std"], "days", [("n", [len(a), len(b)]), ("median", meds), ("mean", [a.mean(), b.mean()])]),
                  qs=[mc(rng, "Среднее больше медианы в обеих группах. О чём это говорит?", "Правая асимметрия: несколько свиней болели очень долго",
                         ["Данные нормальные", "Левая асимметрия: многие выздоровели мгновенно", "Ошибка в расчёте"],
                         "Длинный правый хвост тянет среднее вверх, а медиана на него не реагирует. Время до события почти всегда так выглядит.")]))
    P, F = p_lm(np.c_[np.ones(n), (prot == "std").astype(float)], days, (1, 2))
    S.append(step("Ящики и остатки", task="Посмотри на ящики и на Q-Q остатков обычной модели.",
                  code=["> boxplot(days ~ prot, data = tr)", "> m <- lm(days ~ prot, data = tr)", "> plot(m, which = 2)"],
                  plots=[box({"new": a, "std": b}, yl="days"), P[1]],
                  qs=[mc(rng, "Что видно на Q-Q?", "Правый хвост загибается вверх: остатки асимметричны",
                         ["Точки на прямой: t-тест подходит без оговорок", "Левый хвост уходит вниз: левая асимметрия", "Две полосы: исход 0/1"],
                         f"При n = {n} асимметрия заметно влияет на t-тест. Варианты: ранговый тест (Вилкоксон) или log(days) и t-тест на лог-шкале.")]))
    wl = ["> wilcox.test(days ~ prot, data = tr)"] + H.htest(w["method"], "days by prot", [("W", w["W"])], w["p"], "true location shift is not equal to 0")
    if w["warn"]:
        wl += ["Warning message:", "In wilcox.test.default(x = DATA[[1L]], y = DATA[[2L]], ...) :", "  cannot compute exact p-value with ties"]
    S.append(step("Ранговый тест Вилкоксона", task="Сравни схемы тестом, который работает с рангами.",
                  pick=code_pick(rng, "Какой код?", "wilcox.test(days ~ prot, data = tr)",
                                 ["wilcox.test(days ~ prot, data = tr, paired = TRUE)", "kruskal.test(prot ~ days, data = tr)", "t.test(rank(days), data = tr)"],
                                 "Ранговая сумма Вилкоксона (Манна — Уитни) — <code>wilcox.test</code> с формулой. paired = TRUE — для пар, здесь свиньи разные."),
                  code=wl[0], out=wl[1:],
                  qs=[mc(rng, "Почему R предупреждает про ties?" if w["warn"] else "Что значит «exact» в названии теста?",
                         ("Есть одинаковые значения дней: точный p не считается, R взял нормальное приближение" if w["warn"]
                          else "p посчитан точно по всем перестановкам рангов, без приближения"),
                         (["Ошибка: тест нельзя использовать", "Данные не нормальные", "Группы разного размера"] if w["warn"]
                          else ["Тест точнее t-теста при любых данных", "Средние посчитаны без округления", "Выборки одинакового размера"]),
                         "Дни — целые числа, совпадения (ties) неизбежны. Это не ошибка, а пометка о способе расчёта p." if w["warn"]
                         else "Без совпадений и при n &lt; 50 R перебирает все возможные расклады рангов."),
                      mc(rng, "Что проверяет этот тест?", "Сдвинуто ли распределение дней в одной группе относительно другой",
                         ["Равны ли средние", "Нормальны ли данные", "Равны ли дисперсии"],
                         f"Ранговый тест сравнивает, у кого значения обычно больше. Здесь p = {pr(w['p'])}.")]))
    gm = math.exp(-tl["m1"] + tl["m2"])
    lt = ["> t.test(log(days) ~ prot, data = tr)"] + H.htest("Welch Two Sample t-test", "log(days) by prot", [("t", tl["t"]), ("df", tl["df"])], tl["p"],
                                                            "true difference in means between group new and group std is not equal to 0", tl["ci"], 95,
                                                            [("mean in group new", tl["m1"]), ("mean in group std", tl["m2"])])
    rlo, rhi = math.exp(-tl["ci"][1]), math.exp(-tl["ci"][0])
    S.append(step("Тот же вопрос на лог-шкале", task="Второй путь: t-тест на log(days). Ответ получается в разах, его легко объяснить.",
                  code=lt[0], out=lt[1:] + ["> exp(-rev(t.test(log(days) ~ prot, data = tr)$conf.int))", f"[1] {R.r7(rlo)} {R.r7(rhi)}"],
                  qs=[num("Во сколько раз дольше болеют на схеме std (отношение геометрических средних)?", gm,
                          f"exp(разность средних логарифмов) = exp({R.r7(tl['m2'])} − {R.r7(tl['m1'])}) = <b>{rua(gm, 3)}</b>. "
                          f"ДИ в разах: от {rua(rlo, 3)} до {rua(rhi, 3)}.", rel=0.02),
                      mc(rng, "Совпадают ли выводы двух тестов?",
                         ("Да: оба находят разницу" if v["kind"] == "sig" else "Да: оба не находят разницы"),
                         ["Нет, и надо выбрать тот, где p меньше", "Нет: t-тест на логарифмах неприменим к дням", "Сравнивать нельзя: разные данные"],
                         "Когда разумные методы согласны, вывод устойчив. Если бы расходились — показали бы оба и разобрались почему.")]))
    rep = (f"Медиана выздоровления: {ru(meds[0], 1).replace(',0', '')} дн. на new и {ru(meds[1], 1).replace(',0', '')} дн. на std "
           f"(ранговый тест Вилкоксона, p = {pr(w['p'])}); на std болеют в {rua(gm, 2)} раза дольше (95% ДИ {rua(rlo, 2)}–{rua(rhi, 2)})")
    S.append(step("Записать результат", qs=[mc(rng, "Какая фраза верна?", rep,
                                                [f"Средние различаются (W = {H.fmt5(w['W'])}), значит лечение работает у всех свиней",
                                                 "Данные ненормальные, поэтому сравнить схемы нельзя",
                                                 f"Схема new сокращает болезнь на {ru(100 * (1 - 1 / gm), 0)} дней"],
                                                "Для асимметричных данных пишут медианы и ранговый тест или отношение в разах с лог-шкалы. "
                                                "Проценты и дни путать нельзя.")]))
    items = [("Схему назначали жребием, свиньи независимы", True), ("Асимметрию увидели по ящикам и Q-Q", True),
             ("Ранговый тест требует нормальности", False), ("Отношение в разах получили через exp от разности логарифмов", True),
             ("Если p &gt; 0,05, схемы одинаково хороши", False)]
    S.append(step("Проверка", qs=[multi(rng, "Отметь верное.", items,
                                        "Ранговые тесты не требуют нормальности. «Не нашли разницы» ≠ «схемы равны»: для равенства нужен узкий ДИ около нуля.")]))
    return dict(g="means", key="wrs", title="Асимметричные данные: Вилкоксон и log", vt=("разница есть" if v["kind"] == "sig" else "разница не доказана"),
                goal=f"<p><b>Вопрос.</b> Быстрее ли выздоравливают свиньи на новой схеме лечения? {n} больных свиней, схему назначали жребием. Отклик — дни до выздоровления.</p>",
                steps=S)


# =====================================================================
# 4. One-way ANOVA + Tukey (+ Welch ANOVA when SDs differ)
# =====================================================================
AOV_VAR = [dict(mu=[8.0, 8.9, 9.4], sd=[1.0, 1.0, 1.0], n=12, kind="sig"),
           dict(mu=[8.6, 8.8, 8.7], sd=[1.1, 1.1, 1.1], n=10, kind="ns"),
           dict(mu=[8.2, 8.8, 9.6], sd=[0.5, 1.0, 2.0], n=10, kind="var")]


def welch_anova(groups):
    k = len(groups)
    n = np.array([len(g) for g in groups], float)
    m = np.array([g.mean() for g in groups])
    s2 = np.array([g.var(ddof=1) for g in groups])
    w = n / s2
    mw = (w * m).sum() / w.sum()
    A = (w * (m - mw) ** 2).sum() / (k - 1)
    tmp = ((1 - w / w.sum()) ** 2 / (n - 1)).sum()
    B = 1 + 2 * (k - 2) / (k ** 2 - 1) * tmp
    F = A / B
    df2 = (k ** 2 - 1) / (3 * tmp)
    return dict(F=F, df1=k - 1, df2=df2, p=float(stats.f.sf(F, k - 1, df2)))


def wf_aov(rng, v):
    lv = ["Z0", "Z1500", "Z3000"]
    for _ in range(5000):
        G = {l: np.round(rng.normal(mu, sd, v["n"]), 1) for l, mu, sd in zip(lv, v["mu"], v["sd"])}
        ow = oneway(G)
        tk = tukey(ow)
        sds = [G[l].std(ddof=1) for l in lv]
        rs = max(sds) / min(sds)
        nsig = sum(r["padj"] < 0.05 for r in tk)
        if v["kind"] == "sig" and ow["p"] < 0.001 and rs < 1.6 and nsig == 2:
            break
        if v["kind"] == "ns" and ow["p"] > 0.4 and rs < 1.6:
            break
        if v["kind"] == "var" and rs > 3.2:
            wa = welch_anova([G[l] for l in lv])
            if wa["p"] < 0.05 and ow["p"] < 0.05:
                break
    else:
        raise RuntimeError("aov")
    zinc = np.repeat(lv, v["n"])
    gain = np.concatenate([G[l] for l in lv])
    zinc, gain = shuffle_rows(rng, zinc, gain)
    N = len(gain)
    S = []
    S.append(step("Загрузить данные", task="Таблица <code>nurs</code>: поросёнок, доза оксида цинка (<code>zinc</code>) и прирост за 21 день (кг). Поросята в отдельных клетках.",
                  code=['> nurs <- read.csv("zinc.csv")', "> str(nurs)"],
                  out=str_df(N, [("pig", "int", range(1, N + 1)), ("zinc", "chr", zinc), ("gain", "num", gain)]),
                  qs=[mc(rng, "В каком порядке R расставит уровни zinc?", "Z0, Z1500, Z3000 — по алфавиту, и здесь это совпало с дозой",
                         ["Z3000, Z1500, Z0 — по убыванию дозы", "В порядке первой встречи в файле", "Порядок не важен ни для чего"],
                         "Текст сортируется по алфавиту (посимвольно). Здесь повезло. Будь уровни «low, mid, high», порядок был бы high, low, mid — тогда задают "
                         "<code>factor(zinc, levels = c(\"low\", \"mid\", \"high\"))</code>.")]))
    ag = agg_print("zinc", lv, "gain", [("n", [v["n"]] * 3), ("mean", [G[l].mean() for l in lv]), ("sd", sds)], dec=2)
    shown = [round(s, 2) for s in sds]
    rs_s = max(shown) / min(shown)
    S.append(step("Средние и SD по группам", code="> aggregate(gain ~ zinc, data = nurs, FUN = function(v)\n+   round(c(n = length(v), mean = mean(v), sd = sd(v)), 2))", out=ag,
                  qs=[num("Отношение самого большого SD к самому маленькому?", rs_s,
                          f"{ru(max(shown), 2)} / {ru(min(shown), 2)} = <b>{rua(rs_s, 3)}</b>. " +
                          ("Больше 3 — обычный ANOVA ненадёжен, особенно если большие SD у групп с другим n." if rs_s > 3 else "До 2 — дисперсии можно считать равными."), rel=0.02)]))
    fit_code = "> fit <- aov(gain ~ zinc, data = nurs)"
    P, F = p_lm(np.c_[np.ones(N), (zinc == "Z1500").astype(float), (zinc == "Z3000").astype(float)], gain, (1, 2))
    S.append(step("Модель и остатки", task="Подгони ANOVA и посмотри на остатки до чтения F.",
                  pick=code_pick(rng, "Как подогнать однофакторный ANOVA?", "fit <- aov(gain ~ zinc, data = nurs)",
                                 ["fit <- aov(zinc ~ gain, data = nurs)", "fit <- anova(gain, zinc)", "fit <- t.test(gain ~ zinc, data = nurs)"],
                                 "Отклик слева от ~, фактор справа. t.test работает только с двумя группами."),
                  code=[fit_code, "> par(mfrow = c(1, 2))", "> plot(fit, which = 1:2)"], plots=P,
                  qs=[mc(rng, "Что с остатками?",
                         ("Столбики разной ширины: самый правый (Z3000) самый широкий — дисперсии разные" if v["kind"] == "var"
                          else "Три столбика похожей ширины, Q-Q около прямой: всё в порядке"),
                         (["Всё в порядке", "Дуга: нелинейность", "Две полосы: исход 0/1"] if v["kind"] == "var"
                          else ["Столбики разной ширины: дисперсии разные", "Дуга: нелинейность", "Три столбика — значит в данных три выброса"]),
                         "У ANOVA прогноз — среднее группы, поэтому точки в столбиках. Сравнивают их высоту.")]))
    ssb, ssw = ow["ssb"], ow["ssw"]
    from wf_lib import aov_summary
    al, ad0 = aov_summary([("zinc", 2, ow["ssb"])], (N - 3, ow["ssw"]))
    ad = dict(F=ad0["F"][0], msw=ad0["mse"])
    S.append(step("Таблица ANOVA", code="> summary(fit)", out=al,
                  qs=[num("Чему равно MSE (оценка σ² внутри групп)?", ad["msw"], f"Mean Sq в строке Residuals = <b>{R.r7(ad['msw'])}</b>; σ̂ = √MSE = {rua(math.sqrt(ad['msw']), 3)} кг.", rel=0.01),
                      mc(rng, f"F = {R.fmt1(ad['F'], 4)}, p = {pr(ow['p'])}. Что это значит?",
                         ("Хотя бы одна доза отличается от других по среднему приросту" if ow["p"] < 0.05 else "Разница между дозами не доказана"),
                         ["Все три дозы различаются между собой", "Z3000 лучше всех", "Дисперсии групп равны"],
                         "F-тест отвечает только на вопрос «все ли средние равны». Какие именно пары различаются — покажут сравнения после F (Tukey).")]))
    if v["kind"] == "var":
        wa = welch_anova([G[l] for l in lv])
        wl = ["> oneway.test(gain ~ zinc, data = nurs)", "", "\tOne-way analysis of means (not assuming equal variances)", "",
              "data:  gain and zinc", f"F = {H.fmt5(wa['F'])}, num df = {wa['df1']}.000, denom df = {H.fmt5(wa['df2'])}, p-value = {H.pval(wa['p'])}", ""]
        wl[5] = f"F = {H.fmt5(wa['F'])}, num df = {wa['df1']}.000, denom df = {R.fmt1(wa['df2'], 5)}, p-value = {H.pval(wa['p'])}"
        S.append(step("ANOVA Уэлча", task="При разных дисперсиях есть аналог t-теста Уэлча для трёх и больше групп.",
                      code=wl[0], out=wl[1:],
                      qs=[mc(rng, "Почему denom df дробные и меньше 27?", "Как у Уэлча: df уменьшены из-за разных дисперсий",
                             ["Ошибка округления", "Потому что выкинули выбросы", "denom df всегда дробные"],
                             f"В обычном ANOVA df ошибки = N − k = {N} − 3 = {N - 3}. Уэлч считает с весами 1/SD² и платит за разные дисперсии меньшими df.")]))
    tl = ["> TukeyHSD(fit)", "  Tukey multiple comparisons of means", "    95% family-wise confidence level", "",
          "Fit: aov(formula = gain ~ zinc, data = nurs)", "", "$zinc"]
    mat = R.matrix([r["name"] for r in tk], ["diff", "lwr", "upr", "p adj"],
                   [[r["diff"] for r in tk], [r["lwr"] for r in tk], [r["upr"] for r in tk], [r["padj"] for r in tk]], fixed_dec={3: 7})
    tl += mat
    pa = [float(l.split()[4]) for l in mat[1:]]
    nsig = sum(p < 0.05 for p in pa)
    S.append(step("Сравнения после F: Tukey", code=tl[0], out=tl[1:],
                  qs=[num("Сколько пар различаются значимо (p adj &lt; 0,05)?", nsig,
                          "Смотрим столбец p adj: " + ", ".join(f"{r['name']} {'✓' if p < 0.05 else '✗'}" for r, p in zip(tk, pa)) + f". Итого <b>{nsig}</b>.", rel=0),
                      mc(rng, "Зачем Tukey, а не три обычных t-теста?", "Три теста по 5% дают шанс ложной тревоги около 14%; Tukey держит общий риск 5%",
                         ["Tukey точнее считает средние", "t-тест не работает с тремя группами в одной таблице", "Tukey не требует равных дисперсий"],
                         "Каждое лишнее сравнение — ещё один шанс на ложную находку. Tukey расширяет интервалы так, чтобы риск хотя бы одной ошибки на все пары был 5%. "
                         "Он, как и обычный ANOVA, предполагает равные дисперсии" + (" — здесь это нарушено, поэтому лучше Games — Howell или попарные Уэлчи с поправкой." if v["kind"] == "var" else "."))]))
    if v["kind"] == "ns":
        rep = f"Средний прирост на трёх дозах цинка не различался (ANOVA: F = {R.fmt1(ad['F'], 3).replace('.', ',')}, df = 2 и {N - 3}, p = {pr(ow['p'])}); ни одна пара не различалась по Tukey"
    else:
        rep = (f"Доза цинка влияла на прирост (F = {R.fmt1(ad['F'], 3).replace('.', ',')}, df = 2 и {N - 3}, p = {pr(ow['p'])}" +
               (f"; ANOVA Уэлча p = {pr(welch_anova([G[l] for l in lv])['p'])}" if v["kind"] == "var" else "") +
               f"); по Tukey значимо различаются {nsig} пары из 3")
    S.append(step("Записать результат", qs=[mc(rng, "Какая фраза верна?", rep,
                                                ["Цинк улучшает прирост у всех поросят", "Z3000 лучше Z1500 на 95%", "ANOVA доказал, что все дозы разные"],
                                                "В отчёте — F, оба df, p и итог парных сравнений с поправкой.")]))
    items = [("Дозу назначали каждому поросёнку жребием", True), ("Сначала смотрели на остатки, потом читали F", True),
             ("Сравнения пар делали с поправкой на множественность", True),
             ("Равенство дисперсий выполнено", v["kind"] != "var"),
             ("Можно сказать, какая доза оптимальна для любых условий", False)]
    S.append(step("Проверка", qs=[multi(rng, "Отметь верное.", items, "Оптимум зависит от условий, цен и здоровья стада; опыт отвечает только про эти дозы и этих поросят.")]))
    nm = {"sig": "дозы различаются", "ns": "разница не доказана", "var": "разные дисперсии"}
    return dict(g="means", key="aov", title="Три дозы цинка: ANOVA и Tukey", vt=nm[v["kind"]],
                goal=f"<p><b>Вопрос.</b> Влияет ли доза оксида цинка в стартере на прирост? {N} поросят в отдельных клетках, дозу (0, 1500 или 3000 мг/кг) назначали жребием.</p>",
                steps=S)


def build_a(rng):
    out = []
    for v in T2_VAR:
        out.append(wf_t2(rng, v))
    for v in PAIR_VAR:
        out.append(wf_paired(rng, v))
    for v in WRS_VAR:
        out.append(wf_wrs(rng, v))
    for v in AOV_VAR:
        out.append(wf_aov(rng, v))
    return out
