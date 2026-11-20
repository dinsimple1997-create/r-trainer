"""Assumption-checking trainer: ~1000 residual-plot sets, normal and broken -> diagdata.json

Each item: a model (two groups / ANOVA, simple or multiple regression, mixed model, logistic),
R code that drew the plots, the plots themselves (data only, the app draws them like plot.lm),
and 3 questions: what is broken (multi), what to do, and one about reading the plots.
Labels are checked numerically, so an "OK" item really has no strong pattern and a broken one
shows exactly the planted problem.
"""
import json, math
import numpy as np
from scipy import stats

OPTS = ["Линейность (форма связи)", "Равенство дисперсий", "Нормальность остатков", "Выбросы / влиятельные точки",
        "Независимость", "Всё в порядке"]
LIN, VAR, NORM, OUT, IND, OK = range(6)


# ---------------------------------------------------------------------------
# numerics
def lowess(x, y, f=2 / 3, it=3):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    o = np.argsort(x)
    xs, ys = x[o], y[o]
    r = max(2, int(math.ceil(f * n)))
    w_rob = np.ones(n)
    fit = np.zeros(n)
    for _ in range(it + 1):
        for i in range(n):
            d = np.abs(xs - xs[i])
            h = np.sort(d)[r - 1] or 1e-12
            w = np.clip(1 - (d / h) ** 3, 0, None) ** 3 * w_rob
            sw = w.sum()
            if sw <= 0:
                fit[i] = ys[i]
                continue
            mx = (w * xs).sum() / sw
            my = (w * ys).sum() / sw
            vx = (w * (xs - mx) ** 2).sum()
            b = (w * (xs - mx) * (ys - my)).sum() / vx if vx > 1e-12 else 0
            fit[i] = my + b * (xs[i] - mx)
        res = ys - fit
        s = np.median(np.abs(res))
        if s == 0:
            break
        u = res / (6 * s)
        w_rob = np.clip(1 - u ** 2, 0, None) ** 2
    return xs, fit


def ols(X, y):
    XtX = X.T @ X
    b = np.linalg.solve(XtX, X.T @ y)
    H = X @ np.linalg.solve(XtX, X.T)
    h = np.diag(H)
    r = y - X @ b
    n, p = X.shape
    s = math.sqrt(r @ r / (n - p))
    rs = r / (s * np.sqrt(np.clip(1 - h, 1e-9, None)))
    cook = rs ** 2 * h / (p * np.clip(1 - h, 1e-9, None))
    return dict(b=b, fit=X @ b, r=r, rs=rs, h=h, cook=cook, s=s, n=n, p=p)


def checks(F, X, x_for_quad=None, order=None):
    """Numbers used to make sure the label is right."""
    r, rs, fit = F["r"], F["rs"], F["fit"]
    out = {}
    out["shapiro"] = stats.shapiro(rs).pvalue if len(rs) >= 3 else 1
    out["het"] = stats.spearmanr(np.abs(rs), fit).correlation if np.ptp(fit) > 0 else 0
    out["het_p"] = stats.spearmanr(np.abs(rs), fit).pvalue if np.ptp(fit) > 0 else 1
    out["maxrs"] = float(np.max(np.abs(rs)))
    out["cook"] = float(np.max(F["cook"]))
    out["skew"] = float(stats.skew(rs))
    if x_for_quad is not None:
        Xq = np.column_stack([X, (x_for_quad - x_for_quad.mean()) ** 2])
        bq = np.linalg.lstsq(Xq, np.r_[F["fit"] + F["r"]], rcond=None)[0]
        rq = (F["fit"] + F["r"]) - Xq @ bq
        sq = rq @ rq / (len(rq) - Xq.shape[1])
        cov = sq * np.linalg.inv(Xq.T @ Xq)
        t = bq[-1] / math.sqrt(cov[-1, -1])
        out["quad_p"] = 2 * stats.t.sf(abs(t), len(rq) - Xq.shape[1])
    if order is not None:
        rr = r[np.argsort(order)]
        out["lag1"] = float(np.corrcoef(rr[:-1], rr[1:])[0, 1])
    return out


def rnd(a, d=3):
    return [float(f"{v:.{d}g}") for v in a]


def top3(v):
    return [int(i) for i in np.argsort(-np.abs(np.asarray(v)))[:3]]


def smooth_pts(x, y):
    xs, ys = lowess(x, y)
    k = max(1, len(xs) // 30)
    idx = list(range(0, len(xs), k))
    if idx[-1] != len(xs) - 1:
        idx.append(len(xs) - 1)
    return [rnd(xs[idx], 4), rnd(ys[idx], 3)]


def plots_lm(F, which=(1, 2), title_extra=None):
    """plot.lm panels: 1 residuals vs fitted, 2 Q-Q, 3 scale-location, 4 Cook's distance."""
    P = []
    lab = [i + 1 for i in range(F["n"])]
    if 1 in which:
        P.append(dict(k="rvf", x=rnd(F["fit"], 4), y=rnd(F["r"], 3), s=smooth_pts(F["fit"], F["r"]), l=top3(F["r"])))
    if 2 in which:
        o = np.argsort(F["rs"])
        P.append(dict(k="qq", y=rnd(F["rs"][o], 3), id=[int(i) + 1 for i in o], l=top3(F["rs"])))
    if 3 in which:
        v = np.sqrt(np.abs(F["rs"]))
        P.append(dict(k="sl", x=rnd(F["fit"], 4), y=rnd(v, 3), s=smooth_pts(F["fit"], v), l=top3(F["rs"])))
    if 4 in which:
        P.append(dict(k="cook", y=rnd(F["cook"], 3), l=top3(F["cook"])))
    return P


# ---------------------------------------------------------------------------
# texts
REMEDY = {
    LIN: "Изменить форму связи: добавить X² или взять log(X)",
    VAR: "Взять log(Y) или модель с разными дисперсиями (Welch, varIdent)",
    NORM: "Взять log(Y) или ранговый (непараметрический) метод",
    OUT: "Проверить запись; если ошибки нет — анализ с ней и без неё",
    "infl": "Проверить точку и сравнить выводы с ней и без неё",
    "serial": "Учесть порядок во времени: модель с корреляцией остатков",
    "cluster": "Анализ по средним загонов или смешанная модель с (1 | pen)",
    OK: "Ничего не менять: предпосылки в порядке, анализ как есть",
    "logit": "Ничего: у 0/1 исхода две полосы остатков — это норма",
    "logvar": "Взять log(Y): он обычно чинит и «трубу», и правый хвост",
    "farm": "Проверить эту ферму и сравнить выводы с ней и без неё",
}
WRONG_REMEDY = ["Удалить самые далёкие точки и не упоминать об этом", "Сделать тест Шапиро — Уилка и, если p > 0,05, ничего не делать",
                "Увеличить уровень значимости до 0,10", "Всегда переходить на непараметрический тест"]

WHERE = {
    LIN: ("Residuals vs Fitted: дуга вместо горизонтальной полосы",
          "Красная линия сглаживания на Residuals vs Fitted изгибается дугой: средняя связь не прямая."),
    VAR: ("Residuals vs Fitted: разброс растёт слева направо — «труба»",
          "На Residuals vs Fitted вертикальный разброс расширяется слева направо; на Scale-Location линия идёт вверх."),
    NORM: ("Q-Q Residuals: точки на хвостах уходят от пунктирной прямой",
           "На Q-Q правый хвост загибается вверх (правая асимметрия): остатки не нормальные."),
    OUT: ("Одна точка далеко от остальных, |стандартизованный остаток| > 3",
          "R подписывает номера самых крайних точек; у выброса стандартизованный остаток больше 3–4 по модулю."),
    "infl": ("Cook's distance: у одной точки D больше 1, у остальных около 0",
             "Cook's D объединяет «далеко по X» (рычаг) и «далеко по Y» (остаток). D > 1 — точка сама тянет линию."),
    "serial": ("Остатки по порядку: длинные серии одного знака, волны",
               "Соседние по времени остатки похожи: волны вместо случайного шума."),
    "cluster": ("Ни на каком: это видно только по дизайну исследования",
                "Независимость проверяют по дизайну: что назначали жребием (единица эксперимента) и что записали в строку (единица наблюдения)."),
    OK: ("Нигде: горизонтальная полоса без формы, точки около прямой на Q-Q",
         "Немного неровностей бывает всегда, особенно при малом n. Ищем явный узор, а не идеальную картинку."),
}

IMPORTANCE = {
    LIN: ("Всё сразу: и наклон, и тесты, и прогнозы",
          ["Только интервалы прогноза, а оценки наклона остаются в порядке", "Только p-value, сам наклон оценён верно", "Ничего важного, если R² всё равно высокий"],
          "Неверная форма связи искажает сам наклон — критично для любых выводов."),
    VAR: ("SE, p-value и интервалы прогноза; сами оценки не смещены",
          ["Сами оценки средних и наклона: они смещаются", "Ничего: равенство дисперсий нужно только для красоты графиков",
           "Только R² модели, всё остальное в порядке"],
          "Оценки средних и наклона не смещены, но SE и тесты врут; сильнее всего при разных n в группах. Интервал прогноза для отдельного животного — самый чувствительный."),
    NORM: ("В основном интервалы прогноза; тесты при большом n устойчивы",
           ["Всё: без нормальности модель вообще нельзя использовать для выводов", "Оценки наклона: они смещаются к нулю", "Независимость наблюдений друг от друга"],
           "Нормальность — наименее важная предпосылка: при умеренном n работает ЦПТ. Но смотри на выбросы и сильную асимметрию при малом n."),
    IND: ("Все выводы: SE, p-value и доверительные интервалы",
          ["Только интервалы прогноза для новых животных, остальное в порядке", "Только оценки средних, а SE и p верны", "Ничего: при большом n независимость не важна"],
          "Независимость критична: при зависимых данных SE обычно занижены, p слишком маленькие."),
    OUT: ("Оценки и выводы могут держаться на одной точке",
          ["Ничего: одна точка на выводы не влияет", "Только нормальность остатков на Q-Q графике, но не сами выводы", "Только R² модели, а наклон в порядке"],
          "Поэтому сравнивают анализ с точкой и без неё."),
}


def multi_q(correct, why):
    return dict(t="multi", q="Что не так с моделью? Отметь все проблемы.", o=OPTS, a=sorted(correct), ex=why)


def mc(rng, q, right, wrong, ex):
    opts = [right] + list(wrong[:3])
    order = list(rng.permutation(4))
    return dict(t="mc", q=q, o=[opts[i] for i in order], a=order.index(0), ex=ex)


CONFLICT = [{VAR, NORM, "logvar"}, {OUT, "infl", "farm"}]


def remedy_q(rng, key, extra_wrong=(), kind=None):
    right = REMEDY[key]
    bad = {OK, "logit"}
    for grp in CONFLICT:
        if key in grp:
            bad |= grp
    if kind != "lmm":
        bad.add("farm")
    pool = [v for k, v in REMEDY.items() if v != right and k not in bad]
    rng.shuffle(pool)
    wr = list(extra_wrong) + pool[:2] + [WRONG_REMEDY[int(rng.integers(len(WRONG_REMEDY)))]]
    seen, wr2 = set(), []
    for w in wr:
        if w != right and w not in seen:
            wr2.append(w)
            seen.add(w)
    return mc(rng, "Что лучше всего сделать?", right, wr2[:3], REMEDY_EX[key])


REMEDY_EX = {
    LIN: "Если средняя связь изогнута, ни один тест по прямой не годится. Добавляют квадратичный член (I(x^2)) или берут log(X), потом снова смотрят на остатки.",
    VAR: "Когда разброс растёт вместе со средним, log(Y) обычно выравнивает дисперсии. Для групп можно Welch t-тест или модель с varIdent в nlme::lme.",
    NORM: "Правый хвост часто лечится log(Y). Если не помогает и выборка маленькая — ранговый тест. При большом n небольшая асимметрия почти не мешает тестам.",
    OUT: "Выброс сначала исследуют: ошибка ввода или измерения? Животное из другой совокупности? Если ошибки нет — оставить, но показать анализ с ним и без него.",
    "infl": "Влиятельная точка в одиночку меняет линию. Удалять «потому что мешает» нельзя: проверяют данные и смотрят, меняются ли выводы без неё.",
    "serial": "Если измерения идут во времени и соседние похожи, нужна модель, которая это учитывает (повторные измерения, corAR1). Обычный lm занижает SE.",
    "cluster": "Свиньи одного загона не независимы, если корм назначали загону. Либо одна строка на загон (среднее), либо lmer(y ~ trt + (1 | pen)).",
    OK: "Все предпосылки выглядят разумно. Не надо «чинить» то, что не сломано: лишние преобразования только усложняют интерпретацию.",
    "logit": "В логистической регрессии остатки 0/1-исхода всегда лежат двумя полосами. Нормальность и равные дисперсии там не требуются — проверяют другое (форму связи, влиятельные точки, сверхдисперсию для биномиальных долей).",
    "logvar": "Мультипликативная ошибка (разброс растёт со средним, правый хвост) — классический случай для log(Y). Потом коэффициенты читаются в разах.",
    "farm": "BLUP одной фермы далеко от остальных. Это выброс на уровне 2. Проверяют, нет ли ошибки и не другая ли это совокупность; сравнивают выводы с фермой и без.",
}


# ---------------------------------------------------------------------------
SLR_CTX = [
    dict(y="adg", x="start_wt", xru="масса при постановке (кг)", yru="среднесуточный привес (г/сут)", xr=(18, 40), b0=500, b1=10, s=60, data="pigs"),
    dict(y="hcw", x="age", xru="возраст при убое (дни)", yru="масса туши (кг)", xr=(150, 200), b0=5, b1=0.55, s=6, data="carcass"),
    dict(y="intake", x="bw", xru="живая масса (кг)", yru="потребление корма (кг/сут)", xr=(30, 120), b0=0.6, b1=0.022, s=0.25, data="feed"),
    dict(y="bf", x="hcw", xru="масса туши (кг)", yru="толщина шпика (мм)", xr=(75, 115), b0=-3, b1=0.15, s=2.0, data="carcass"),
    dict(y="wean_wt", x="birth_wt", xru="масса при рождении (кг)", yru="масса при отъёме (кг)", xr=(0.8, 2.0), b0=3.0, b1=2.3, s=0.8, data="piglets"),
    dict(y="milk", x="dim", xru="день лактации", yru="удой (кг/сут)", xr=(10, 300), b0=38, b1=-0.05, s=3.5, data="cows"),
    dict(y="water", x="temp", xru="температура в корпусе (°C)", yru="потребление воды (л/сут)", xr=(16, 32), b0=2, b1=0.25, s=0.9, data="barn"),
]
MLR_CTX = [
    dict(y="adg", xs=["start_wt", "lysine", "days_sick"], b=[480, 9, 120, -12], xr=[(18, 40), (0.9, 1.3), (0, 8)], s=55, data="pigs",
         yru="среднесуточный привес (г/сут)"),
    dict(y="hcw", xs=["age", "adg", "sex_m"], b=[-60, 0.4, 0.09, 2.5], xr=[(150, 200), (700, 1000), (0, 1)], s=5, data="carcass",
         yru="масса туши (кг)"),
    dict(y="lnscc", xs=["dim", "parity", "herd_size"], b=[3.8, 0.004, 0.25, -0.001], xr=[(5, 300), (1, 6), (40, 600)], s=0.9, data="cows",
         yru="log числа соматических клеток"),
    dict(y="fcr", xs=["energy", "lysine", "temp"], b=[4.5, -0.12, -0.5, 0.02], xr=[(9.5, 10.8), (0.9, 1.3), (16, 30)], s=0.12, data="pens",
         yru="конверсия корма"),
]
GRP_CTX = [
    dict(y="adg", g="diet", lev=["A", "B", "C"], mu=[780, 810, 835], s=55, n=(8, 20), yru="среднесуточный привес (г/сут)", data="trial", unitru="свинья"),
    dict(y="cck", g="group", lev=["Ctrl", "Gall", "Ulcer"], mu=[0.25, 0.32, 0.45], s=0.06, n=(8, 12), yru="активность фермента", data="gp", unitru="животное"),
    dict(y="titer", g="vacc", lev=["V1", "V2", "V3", "V4"], mu=[6.0, 6.6, 7.1, 6.3], s=0.8, n=(6, 15), yru="титр антител (log2)", data="sera", unitru="свинья"),
    dict(y="gain", g="zinc", lev=["Z0", "Z1500", "Z3000"], mu=[8.2, 8.9, 9.3], s=1.0, n=(10, 20), yru="прирост за 21 день (кг)", data="nursery", unitru="поросёнок"),
    dict(y="bwt", g="parity", lev=["P1", "P2"], mu=[1.32, 1.41], s=0.22, n=(15, 40), yru="масса при рождении (кг)", data="piglets", unitru="поросёнок"),
]
LMM_CTX = [
    dict(y="adg", grp="farm", b0=820, b1=-25, x="sex", s=65, tau=35, ng=(15, 30), m=(10, 30), data="pigs"),
    dict(y="lnscc", grp="herdid", b0=4.7, b1=-0.7, x="heifer", s=1.2, tau=0.4, ng=(20, 40), m=(15, 40), data="cows"),
    dict(y="wwt", grp="sow", b0=6.4, b1=0.15, x="sex", s=0.9, tau=0.6, ng=(30, 60), m=(8, 13), data="piglets"),
]


def err(rng, n, kind, s):
    if kind == "skew":
        z = rng.lognormal(0, 0.9, n)
        return s * (z - z.mean()) / z.std()
    return rng.normal(0, s, n)


def code_lm(formula, data, which):
    if which == (1, 2):
        return f"> m <- lm({formula}, data = {data})\n> par(mfrow = c(1, 2))\n> plot(m, which = 1:2)"
    if which == (1, 2, 3, 4):
        return f"> m <- lm({formula}, data = {data})\n> par(mfrow = c(2, 2))\n> plot(m, which = 1:4)"
    return f"> m <- lm({formula}, data = {data})\n> plot(m, which = c({', '.join(map(str, which))}))"


# ---------------------------------------------------------------------------
def gen_slr(rng, c, flaw):
    for _ in range(300):
        n = int(rng.integers(25, 90))
        x = rng.uniform(*c["xr"], n)
        xc = (x - x.mean()) / (c["xr"][1] - c["xr"][0])
        mu = c["b0"] + c["b1"] * x
        s = c["s"]
        e = rng.normal(0, s, n)
        order = None
        if flaw == "lin":
            mu = mu + s * rng.choice([-1, 1]) * rng.uniform(5, 9) * (xc ** 2 - np.mean(xc ** 2))
        elif flaw == "var":
            e = rng.normal(0, 1, n) * s * (0.25 + 2.2 * (xc - xc.min()) / np.ptp(xc))
        elif flaw == "skew":
            e = err(rng, n, "skew", s)
        elif flaw == "logvar":
            m0 = mu.mean()
            e = mu * (np.exp(rng.normal(0, 0.45, n)) - 1) * (s * 3 / m0)
            e = e * (1 + 1.6 * (xc - xc.min()) / np.ptp(xc))
        elif flaw == "out":
            k = int(rng.integers(n))
            x[k] = np.quantile(x, rng.uniform(0.3, 0.7))
            mu[k] = c["b0"] + c["b1"] * x[k]
            e[k] = rng.choice([-1, 1]) * s * rng.uniform(5.5, 8)
        elif flaw == "infl":
            k = int(rng.integers(n))
            x[k] = c["xr"][1] + (c["xr"][1] - c["xr"][0]) * rng.uniform(0.9, 1.4)
            mu[k] = c["b0"] + c["b1"] * x[k]
            e[k] = -np.sign(c["b1"]) * s * rng.uniform(4, 6)
        elif flaw == "serial":
            order = np.arange(n)
            x = np.sort(x) if rng.random() < 0 else x
            rho = rng.uniform(0.75, 0.9)
            ee = np.zeros(n)
            ee[0] = rng.normal(0, s)
            for i in range(1, n):
                ee[i] = rho * ee[i - 1] + rng.normal(0, s * math.sqrt(1 - rho ** 2))
            e = ee
        y = mu + e
        X = np.column_stack([np.ones(n), x])
        F = ols(X, y)
        ch = checks(F, X, x_for_quad=x, order=np.arange(n) if flaw == "serial" else None)
        if label_ok(flaw, ch, n):
            break
    else:
        raise RuntimeError(flaw)
    which = (1, 2, 3, 4) if flaw == "infl" else ((1, 2, 3) if flaw in ("var", "logvar") and rng.random() < 0.5 else (1, 2))
    P = plots_lm(F, which)
    code = code_lm(f"{c['y']} ~ {c['x']}", c["data"], which)
    if flaw == "serial":
        P.append(dict(k="ord", y=rnd(F["r"], 3)))
        code += "\n> plot(resid(m), type = \"b\", xlab = \"Порядок взвешивания (день)\")"
    story = (f"Простая регрессия: {c['yru']} ~ {c['xru']}, n = {n}." +
             (" Данные записаны подряд: одна и та же группа, день за днём." if flaw == "serial" else
              " Каждая строка — отдельное животное, животных отбирали случайно."))
    return finish(rng, "slr", flaw, story, code, P, n, ch)


def gen_mlr(rng, c, flaw):
    for _ in range(300):
        n = int(rng.integers(40, 110))
        Xs = [rng.uniform(*r, n) if r != (0, 1) else rng.integers(0, 2, n).astype(float) for r in c["xr"]]
        if c["xr"][1] == (1, 6):
            Xs[1] = rng.integers(1, 7, n).astype(float)
        mu = c["b"][0] + sum(b * x for b, x in zip(c["b"][1:], Xs))
        s = c["s"]
        e = rng.normal(0, s, n)
        x0 = Xs[0]
        xc = (x0 - x0.mean()) / np.ptp(x0)
        if flaw == "lin":
            mu = mu + s * rng.choice([-1, 1]) * rng.uniform(6, 10) * (xc ** 2 - np.mean(xc ** 2))
        elif flaw == "var":
            f0 = (mu - mu.min()) / np.ptp(mu)
            e = rng.normal(0, 1, n) * s * (0.25 + 2.3 * f0)
        elif flaw == "skew":
            e = err(rng, n, "skew", s)
        elif flaw == "out":
            k = int(rng.integers(n))
            e[k] = rng.choice([-1, 1]) * s * rng.uniform(5.5, 8)
        elif flaw == "infl":
            k = int(rng.integers(n))
            lo, hi = c["xr"][0]
            Xs[0][k] = hi + (hi - lo) * rng.uniform(1.0, 1.5)
            mu[k] = c["b"][0] + sum(b * x[k] for b, x in zip(c["b"][1:], Xs))
            e[k] = -np.sign(c["b"][1]) * s * rng.uniform(4, 6)
        y = mu + e
        X = np.column_stack([np.ones(n)] + Xs)
        F = ols(X, y)
        ch = checks(F, X, x_for_quad=Xs[0])
        if label_ok(flaw, ch, n):
            break
    else:
        raise RuntimeError(flaw)
    which = (1, 2, 3, 4)
    P = plots_lm(F, which)
    code = code_lm(f"{c['y']} ~ {' + '.join(c['xs'])}", c["data"], which)
    story = f"Множественная регрессия: {c['yru']} от {', '.join(c['xs'])}; n = {n}, одна строка — одно животное (или загон), отбор случайный."
    return finish(rng, "mlr", flaw, story, code, P, n, ch)


def gen_grp(rng, c, flaw):
    k = len(c["lev"])
    for _ in range(300):
        ns = rng.integers(*c["n"], k)
        g = np.repeat(np.arange(k), ns)
        n = len(g)
        mu = np.array(c["mu"])[g]
        s = c["s"]
        sds = np.full(k, s)
        if flaw in ("var",):
            # largest mean gets the largest spread (trumpet)
            ratio = rng.uniform(3.2, 5)
            order = np.argsort(c["mu"])
            sds = s * np.interp(np.arange(k), [0, k - 1], [1, ratio])[np.argsort(order)]
        e = rng.normal(0, 1, n) * sds[g]
        if flaw == "skew":
            e = err(rng, n, "skew", s)
        if flaw == "logvar":
            e = mu * (np.exp(rng.normal(0, 0.5, n)) - 1) * (s * 2.5 / np.mean(c["mu"]))
            e = e * (np.array(c["mu"])[g] / min(c["mu"])) ** 3
        if flaw == "out":
            j = int(rng.integers(n))
            e[j] = rng.choice([-1, 1]) * s * rng.uniform(5.5, 8)
        pens = None
        if flaw == "cluster":
            per = int(rng.integers(4, 8))
            pens = np.arange(n) // per
            pe = rng.normal(0, s * 0.8, pens.max() + 1)
            e = rng.normal(0, s * 0.6, n) + pe[pens]
        y = mu + e
        X = np.column_stack([np.ones(n)] + [(g == j).astype(float) for j in range(1, k)])
        F = ols(X, y)
        ch = checks(F, X)
        gs = [float(np.std(F["r"][g == j], ddof=1)) for j in range(k)]
        ch["rs_ratio"] = max(gs) / min(gs)
        if label_ok(flaw, ch, n, groups=True):
            break
    else:
        raise RuntimeError(flaw)
    P = plots_lm(F, (1, 2))
    code = code_lm(f"{c['y']} ~ {c['g']}", c["data"], (1, 2))
    if flaw == "cluster":
        story = (f"Сравнение групп ({c['g']}: {', '.join(c['lev'])}) по {c['yru']}. Вариант назначали загону целиком, "
                 f"в каждом загоне {per} голов, а в анализ вошла каждая голова отдельной строкой (n = {n}).")
    else:
        story = (f"Сравнение групп ({c['g']}: {', '.join(c['lev'])}) по {c['yru']}: lm/aov с одним фактором, "
                 f"n = {n}; каждое животное — отдельная единица, варианты назначали жребием.")
    it = finish(rng, "grp", flaw, story, code, P, n, ch)
    it["sds"] = [round(v, 4) for v in gs]
    if flaw in ("var", "logvar", OK, "ok") or rng.random() < 0.3:
        # ratio of SDs question
        r = ch["rs_ratio"]
        it["qs"][2] = dict(t="num", q=f"SD остатков по группам: {', '.join(f'{l} = {fmt(v)}' for l, v in zip(c['lev'], gs))}. Чему равно отношение самого большого SD к самому маленькому?",
                           a=float(f"{r:.4g}"), tol=float(f"{max(0.05, 0.03 * r):.3g}"), unit="",
                           ex=f"r<sub>s</sub> = {fmt(max(gs))} / {fmt(min(gs))} = <b>{fmt(r)}</b>. Правило курса: r<sub>s</sub> ≤ 2 — равенство дисперсий разумно; "
                              f"r<sub>s</sub> ≥ 3 — сомнительно. {'Здесь дисперсии явно разные.' if r >= 3 else ('Здесь всё в порядке.' if r <= 2 else 'Пограничный случай: смотрим и на график.')}")
    return it


def fmt(v):
    d = max(0, 3 - 1 - math.floor(math.log10(abs(v)))) if v else 0
    return f"{v:.{min(d, 5)}f}".replace(".", ",")


def gen_lmm(rng, c, flaw):
    from mixed import lmm
    for _ in range(200):
        ng = int(rng.integers(*c["ng"]))
        sizes = rng.integers(c["m"][0], c["m"][1] + 1, ng)
        g = np.repeat(np.arange(ng), sizes)
        n = len(g)
        x = (rng.random(n) < 0.5).astype(float)
        u = rng.normal(0, c["tau"], ng)
        if flaw == "farm":
            u[int(rng.integers(ng))] = rng.choice([-1, 1]) * c["tau"] * rng.uniform(4.5, 6)
        s = c["s"]
        e = rng.normal(0, s, n)
        if flaw == "skew":
            e = err(rng, n, "skew", s)
        if flaw == "var":
            base = u[g]
            f0 = (base - base.min()) / np.ptp(base)
            e = rng.normal(0, 1, n) * s * (0.3 + 2.4 * f0)
        y = c["b0"] + c["b1"] * x + u[g] + e
        X = np.column_stack([np.ones(n), x])
        F = lmm(y, X, g, ["(Intercept)", "x"])
        res = F["resid"]
        fit = F["fitted"]
        rs = res / F["sigma"]
        blup = F["u"]
        ch = dict(shapiro=stats.shapiro(rs).pvalue, het=stats.spearmanr(np.abs(rs), fit).correlation,
                  het_p=stats.spearmanr(np.abs(rs), fit).pvalue, maxrs=float(np.max(np.abs(rs))),
                  blup_sh=stats.shapiro(blup).pvalue, blup_max=float(np.max(np.abs((blup - blup.mean()) / blup.std()))), skew=float(stats.skew(rs)))
        if label_ok(flaw, ch, n, mixed=True):
            break
    else:
        raise RuntimeError(flaw)
    P = [dict(k="rvf", x=rnd(fit, 4), y=rnd(res, 3), s=None, l=top3(res), t="Residuals vs Fitted (level 1)"),
         dict(k="qq", y=rnd(np.sort(rs), 3), id=[int(i) + 1 for i in np.argsort(rs)], l=top3(rs), t="Q-Q: residuals (level 1)"),
         dict(k="qq", y=rnd(np.sort(blup / blup.std()), 3), id=[int(i) + 1 for i in np.argsort(blup)], l=top3(blup), t=f"Q-Q: random effects ({c['grp']})")]
    code = (f"> m <- lmer({c['y']} ~ {c['x']} + (1|{c['grp']}), data = {c['data']})\n> plot(m)                         # level 1\n"
            f"> qqnorm(resid(m)); qqline(resid(m))\n> qqnorm(ranef(m)${c['grp']}[, 1])     # level 2: BLUPs")
    story = f"Смешанная модель: {c['y']} ~ {c['x']} + (1 | {c['grp']}); {n} наблюдений в {ng} группах ({c['grp']}). Проверяем остатки уровня 1 и случайные эффекты уровня 2."
    return finish(rng, "lmm", flaw, story, code, P, n, ch)


def gen_logit(rng, c, flaw):
    n = int(rng.integers(60, 200))
    x = rng.uniform(*c["xr"], n)
    xc = (x - x.mean()) / x.std()
    p = 1 / (1 + np.exp(-(-0.5 + 0.9 * xc)))
    y = (rng.random(n) < p).astype(float)
    # glm fit
    X = np.column_stack([np.ones(n), x])
    b = np.zeros(2)
    for _ in range(30):
        eta = X @ b
        mu = 1 / (1 + np.exp(-eta))
        w = mu * (1 - mu)
        z = eta + (y - mu) / w
        b = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (w * z))
    eta = X @ b
    mu = 1 / (1 + np.exp(-eta))
    dres = np.sign(y - mu) * np.sqrt(-2 * (y * np.log(mu) + (1 - y) * np.log(1 - mu)))
    W = mu * (1 - mu)
    Hm = (X * W[:, None]) @ np.linalg.solve(X.T @ (X * W[:, None]), X.T)
    h = np.diag(Hm)
    rs = dres / np.sqrt(1 - h)
    P = [dict(k="rvf", x=rnd(eta, 4), y=rnd(dres, 3), s=smooth_pts(eta, dres), l=top3(dres), xl="Predicted values"),
         dict(k="qq", y=rnd(np.sort(rs), 3), id=[int(i) + 1 for i in np.argsort(rs)], l=top3(rs), yl="Std. deviance resid.")]
    code = f"> m <- glm({c['yb']} ~ {c['x']}, family = binomial, data = {c['data']})\n> par(mfrow = c(1, 2))\n> plot(m, which = 1:2)"
    story = f"Логистическая регрессия: {c['ybru']} (0/1) ~ {c['xru']}, n = {n}. Кто-то построил те же графики, что для lm."
    return finish(rng, "logit", "logit", story, code, P, n, {})


LOGIT_CTX = [dict(yb="pn", ybru="пневмония на бойне", x="age", xru="возраст (дни)", xr=(150, 200), data="pigs"),
             dict(yb="dead", ybru="падёж до отъёма", x="birth_wt", xru="масса при рождении (кг)", xr=(0.6, 2.0), data="piglets"),
             dict(yb="mastitis", ybru="клинический мастит", x="dim", xru="день лактации", xr=(5, 300), data="cows")]


def label_ok(flaw, ch, n, groups=False, mixed=False):
    """Accept a simulated data set only if the plots show the planted problem and nothing else."""
    sh = ch["shapiro"]
    het = abs(ch["het"])
    mx = ch["maxrs"]
    cook = ch.get("cook", 0)
    quad = ch.get("quad_p", 1)
    lag = ch.get("lag1", 0)
    clean_norm = sh > 0.1 and abs(ch["skew"]) < 0.5
    clean_var = (het < 0.15 or ch["het_p"] > 0.15) if not groups else ch.get("rs_ratio", 1) < 1.8
    clean_out = mx < 3.0 and cook < 0.5
    clean_lin = quad > 0.1
    if mixed:
        clean_blup = ch["blup_sh"] > 0.1 and ch["blup_max"] < 2.8
        if flaw in ("ok", OK):
            return clean_norm and (het < 0.12) and mx < 3.2 and clean_blup
        if flaw == "skew":
            return sh < 1e-4 and ch["skew"] > 1 and het < 0.15 and clean_blup
        if flaw == "var":
            return het > 0.35 and ch["het_p"] < 1e-4 and clean_blup and abs(ch["skew"]) < 0.6
        if flaw == "farm":
            return ch["blup_max"] > 3.6 and clean_norm and het < 0.15
        return False
    if flaw in ("ok", OK, "cluster"):
        return clean_norm and clean_var and clean_out and clean_lin
    if flaw == "lin":
        return quad < 1e-5 and sh > 0.02 and clean_out
    if flaw == "var":
        if groups:
            return ch["rs_ratio"] >= 3 and mx < 4.5
        return het > 0.4 and ch["het_p"] < 1e-4 and clean_lin and cook < 0.6 and abs(ch["skew"]) < 0.6
    if flaw == "skew":
        return sh < 1e-4 and ch["skew"] > 1 and (clean_var if not groups else ch.get("rs_ratio", 1) < 2) and clean_lin
    if flaw == "logvar":
        return sh < 1e-3 and ch["skew"] > 0.8 and ((het > 0.3 and ch["het_p"] < 1e-3) if not groups else ch["rs_ratio"] >= 3)
    if flaw == "out":
        return mx > 4.5 and cook < 1.0 and clean_lin and (clean_var if not groups else True)
    if flaw == "infl":
        return cook > 1.0
    if flaw == "serial":
        return lag > 0.5 and clean_out and clean_lin
    return False


def finish(rng, kind, flaw, story, code, P, n, ch):
    flaw = OK if flaw == "ok" else flaw
    key = {OK: OK, "lin": LIN, "var": VAR, "skew": NORM, "logvar": "logvar", "out": OUT, "infl": "infl", "serial": "serial",
           "cluster": "cluster", "logit": "logit", "farm": "farm"}[flaw]
    correct = {OK: [OK], "lin": [LIN], "var": [VAR], "skew": [NORM], "logvar": [VAR, NORM], "out": [OUT], "infl": [OUT],
               "serial": [IND], "cluster": [IND], "logit": [OK], "farm": [OUT]}[flaw]
    if flaw == "var":
        why_extra = (" Если хвосты на Q-Q загибаются, это следствие разных дисперсий: остатки с разным разбросом, сложенные вместе, "
                     "всегда выглядят «тяжелохвостыми». Нормальность судят после того, как починили дисперсии.")
    else:
        why_extra = ""
    WHY = {
        OK: "Остатки — горизонтальная полоса без формы, на Q-Q точки около прямой, нет далёких точек. Предпосылки выглядят разумно.",
        "lin": "Линия сглаживания на Residuals vs Fitted — дуга: прямая не описывает связь. Остальное в порядке.",
        "var": "Вертикальный разброс остатков растёт со средним — «труба» (trumpet). Нарушено равенство дисперсий." + why_extra,
        "skew": "Q-Q: правый хвост уходит вверх от прямой, остатки асимметричны. Разброс по графику Residuals vs Fitted ровный.",
        "logvar": "Сразу два признака мультипликативной ошибки: разброс растёт со средним (труба) и правый хвост на Q-Q.",
        "out": "Одна точка с огромным остатком (|стандартизованный| > 4), остальное в порядке.",
        "infl": "Cook's D одной точки больше 1: точка далеко по X и не на линии — она одна тянет наклон.",
        "serial": "Остатки по порядку идут волнами: соседние похожи. Нарушена независимость.",
        "cluster": "Графики выглядят прилично, но дизайн ломает независимость: вариант назначали загону, а строки — свиньи. "
                   "Свиньи одного загона похожи, независимых единиц столько, сколько загонов.",
        "logit": "Для логистической регрессии две полосы остатков и «кривая» Q-Q — норма: исход 0/1 не бывает нормальным. Нарушений нет.",
        "farm": "Остатки уровня 1 в порядке, но на Q-Q случайных эффектов одна ферма далеко от прямой — выброс на уровне 2.",
    }
    qs = [multi_q(correct, WHY[flaw]), remedy_q(rng, key, kind=kind)]
    # third question
    if flaw in (OK, "logit"):
        qs.append(mc(rng, "R подписал номера трёх крайних точек. Это повод беспокоиться?",
                     "Нет: R всегда подписывает 3 крайние точки",
                     ["Да: подписанные точки — выбросы, их надо удалить", "Да: значит, нарушена нормальность остатков", "Да: значит, данные зависимы между собой"],
                     "plot.lm по умолчанию (id.n = 3) подписывает три точки с самыми большими остатками. Подпись — не диагноз; смотрят, насколько они далеко."))
    elif flaw == "cluster":
        qs.append(mc(rng, "Насколько опасно такое нарушение?", IMPORTANCE[IND][0], IMPORTANCE[IND][1], IMPORTANCE[IND][2]))
    else:
        mk = {"lin": LIN, "var": VAR, "skew": NORM, "logvar": VAR, "out": OUT, "infl": "infl", "serial": "serial", "farm": OUT}[flaw]
        if rng.random() < 0.5 and mk in WHERE:
            right, ex = WHERE[mk]
            excl = {OK, "cluster", mk}
            if mk in (OUT, "infl"):
                excl |= {OUT, "infl"}
            if flaw == "logvar":
                excl |= {NORM}
            wr = [v[0] for k2, v in WHERE.items() if k2 not in excl]
            rng.shuffle(wr)
            qs.append(mc(rng, "По какому признаку на графиках это видно?", right, wr, ex))
        else:
            ik = {"lin": LIN, "var": VAR, "skew": NORM, "logvar": VAR, "out": OUT, "infl": OUT, "serial": IND, "farm": OUT}[flaw]
            right, wr, ex = IMPORTANCE[ik]
            qs.append(mc(rng, "Что пострадает от этого нарушения сильнее всего?", right, list(wr), ex))
    return dict(d=kind, f=flaw if isinstance(flaw, str) else "ok", story=story, code=code, plots=P, qs=qs)


PLAN = [
    (gen_slr, SLR_CTX, [("ok", 70), ("lin", 50), ("var", 45), ("skew", 40), ("logvar", 25), ("out", 35), ("infl", 35), ("serial", 35)]),
    (gen_mlr, MLR_CTX, [("ok", 40), ("lin", 30), ("var", 35), ("skew", 35), ("out", 25), ("infl", 40)]),
    (gen_grp, GRP_CTX, [("ok", 55), ("var", 70), ("skew", 40), ("logvar", 25), ("out", 40), ("cluster", 35)]),
    (gen_lmm, LMM_CTX, [("ok", 50), ("var", 30), ("skew", 30), ("farm", 35)]),
    (gen_logit, LOGIT_CTX, [("logit", 50)]),
]
KIND_RU = {"slr": "Простая регрессия", "mlr": "Множественная", "grp": "Группы (t / ANOVA)", "lmm": "Смешанная", "logit": "Логистическая"}


def build():
    rng = np.random.default_rng(20261120)
    items = []
    for fn, ctx, flaws in PLAN:
        for flaw, cnt in flaws:
            for i in range(cnt):
                c = ctx[i % len(ctx)]
                for _try in range(10):
                    try:
                        it = fn(rng, c, flaw)
                        break
                    except RuntimeError:
                        continue
                else:
                    raise RuntimeError((fn.__name__, flaw))
                it["title"] = KIND_RU[it["d"]]
                items.append(it)
    order = np.random.default_rng(20261121).permutation(len(items))
    items = [items[k] for k in order]
    for i, it in enumerate(items):
        it["id"] = i + 1
    return items


B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"


def enc(arr):
    """Quantize to 0..4095 between min and max, two base64 characters per value."""
    lo, hi = float(f"{min(arr):.4g}"), float(f"{max(arr):.4g}")
    if hi <= lo:
        hi = lo + 1e-9
    out = []
    for v in arr:
        q = int(round((min(max(v, lo), hi) - lo) / (hi - lo) * 4095))
        out.append(B64[q >> 6] + B64[q & 63])
    return [lo, hi, "".join(out)]


def pack(items):
    from collections import Counter
    cnt = Counter()
    for it in items:
        for q in it["qs"]:
            cnt[q["q"]] += 1
            cnt[q["ex"]] += 1
            for o in q.get("o", []):
                cnt[o] += 1
    S = [t for t, n in cnt.items() if n > 1 and len(t) > 6]
    ix = {t: i for i, t in enumerate(S)}
    f = lambda t: ix.get(t, t)
    out = []
    for it in items:
        it = dict(it)
        it.pop("sds", None)
        it["qs"] = [dict(q, q=f(q["q"]), ex=f(q["ex"]), **({"o": [f(o) for o in q["o"]]} if "o" in q else {})) for q in it["qs"]]
        P = []
        for p in it["plots"]:
            p = dict(p)
            if p["k"] == "qq":
                ids = p.pop("id")
                p["l"] = [[ids.index(i + 1), i + 1] for i in p["l"]]
            for key in ("x", "y"):
                if key in p:
                    p[key] = enc(p[key])
            if p.get("s"):
                p["s"] = [enc(p["s"][0]), enc(p["s"][1])]
            P.append(p)
        it["plots"] = P
        out.append(it)
    return dict(S=S, X=out)


if __name__ == "__main__":
    data = build()
    json.dump(data, open("/tmp/diagdata_full.json", "w"), ensure_ascii=False, separators=(",", ":"))
    json.dump(pack(data), open("diagdata.json", "w"), ensure_ascii=False, separators=(",", ":"))
    from collections import Counter
    print(len(data), "items", Counter(d["d"] for d in data), Counter(d["f"] for d in data))
