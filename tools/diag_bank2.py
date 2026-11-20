"""Level 2 of 'Предпосылки моделей': kinds of violations, not just the fact of one.

Linearity alone comes in 8 shapes (arc with an optimum, saturation, broken line / plateau, exponential growth,
S-curve, a missed group, a missed interaction, one bad predictor in a multiple regression); normality in 3
(right tail, left tail, heavy tails); plus wrong response type (counts, proportions, 0/1 in lm), seasonality,
three kinds of unusual points, and 'before / after the fix' pairs. Each item asks: what is broken (multi), which
pattern exactly, which R code fixes it, and one reading question. Every label is checked with numbers: the
planted shape must beat the other shapes by AIC, assumption checks must agree with the multi answer.
Items are appended to diagdata.json after the 1000 level-1 items (ids 1001+), progress keys stay valid.
"""
import json, math
import numpy as np
from scipy import stats
import diag_bank as DB
from diag_bank import OPTS, LIN, VAR, NORM, OUT, IND, OK, rnd, top3, smooth_pts, plots_lm, fmt


# ---------------------------------------------------------------------------
# fitting helpers
def fit(X, y):
    return DB.ols(np.asarray(X, float), np.asarray(y, float))


def aic(F):
    n, p = F["n"], F["p"]
    sse = float(F["r"] @ F["r"])
    return n * math.log(sse / n) + 2 * (p + 1)


def aic_logy(X, y):
    F = fit(X, np.log(y))
    return aic(F) + 2 * float(np.sum(np.log(y)))


def broken(x, y):
    """best two-slope (hinge) model y = a + b1 x + b2 (x - c)+ over a grid of c."""
    best = None
    for c in np.quantile(x, np.linspace(0.12, 0.88, 40)):
        X = np.c_[np.ones(len(x)), x, np.maximum(x - c, 0)]
        F = fit(X, y)
        a = aic(F) + 2  # c is a parameter too
        if best is None or a < best[0]:
            best = (a, c, F)
    return best


def shapes(x, y):
    one = np.ones(len(x))
    xc = x - x.mean()
    A = dict(lin=aic(fit(np.c_[one, x], y)), quad=aic(fit(np.c_[one, xc, xc ** 2], y)),
             cub=aic(fit(np.c_[one, xc, xc ** 2, xc ** 3], y)), brk=broken(x, y)[0])
    if np.all(x > 0):
        A["logx"] = aic(fit(np.c_[one, np.log(x)], y))
    if np.all(y > 0):
        A["logy"] = aic_logy(np.c_[one, x], y)
    return A


def basic(F, X, xq=None, order=None):
    ch = DB.checks(F, X, x_for_quad=xq, order=order)
    return ch


def P_rvf(F, title=None, xl=None):
    p = dict(k="rvf", x=rnd(F["fit"], 4), y=rnd(F["r"], 3), s=smooth_pts(F["fit"], F["r"]), l=top3(F["r"]))
    if title:
        p["t"] = title
    if xl:
        p["xl"] = xl
    return p


def P_qq(F, title=None):
    o = np.argsort(F["rs"])
    p = dict(k="qq", y=rnd(F["rs"][o], 3), id=[int(i) + 1 for i in o], l=top3(F["rs"]))
    if title:
        p["t"] = title
    return p


def P_rvx(F, x, name):
    return dict(k="rvf", x=rnd(x, 4), y=rnd(F["r"], 3), s=smooth_pts(x, F["r"]), l=top3(F["r"]), t=f"Residuals vs {name}", xl=name)


def P_xy(x, y, F=None, g=None, xl="", yl="", title=""):
    p = dict(k="xy", x=rnd(x, 4), y=rnd(y, 4), xl=xl, yl=yl, t=title)
    if F is not None:
        p["ab"] = [float(F["b"][0]), float(F["b"][1])]
    if g is not None:
        p["g"] = [int(v) for v in g]
    return p


def P_cook(F):
    return dict(k="cook", y=rnd(F["cook"], 3), l=top3(F["cook"]))


def P_ord(F, xl="Месяц"):
    return dict(k="ord", y=rnd(F["r"], 3), xl=xl)


def mc(rng, q, right, wrong, ex):
    wrong = [w for w in dict.fromkeys(wrong) if w != right][:3]
    opts = [right] + wrong
    order = list(rng.permutation(len(opts)))
    return dict(t="mc", q=q, o=[opts[i] for i in order], a=order.index(0), ex=ex)


def code(s):
    return f"<code>{s}</code>"


def multi_q(correct, why):
    return dict(t="multi", q="Что не так с моделью? Отметь все проблемы.", o=OPTS, a=sorted(set(correct)), ex=why)


# ---------------------------------------------------------------------------
# texts
PAT = {
    "quad": "Дуга: у связи есть оптимум (или минимум)",
    "sat": "Насыщение: сначала быстрый рост, потом всё медленнее",
    "brk": "Излом: одна прямая до порога, после порога — другая (плато)",
    "expo": "Экспонента: рост ускоряется, и разброс растёт вместе со средним",
    "sig": "S-кривая: медленно — быстро — снова медленно",
    "grp": "Две параллельные полосы: в модели не хватает группы",
    "inter": "Разные наклоны у групп: пропущено взаимодействие",
    "which": "Кривая только по одному предиктору",
    "rskew": "Правый хвост: несколько очень больших значений",
    "lskew": "Левый хвост: несколько очень маленьких значений",
    "heavy": "Тяжёлые хвосты с обеих сторон (S на Q-Q)",
    "diamond": "Разброс больше всего в середине, отклик ограничен 0 и 1 (доля)",
    "count": "Косые параллельные полосы: отклик — маленькие целые числа",
    "binlm": "Две косые линии остатков: отклик 0/1 в обычной lm",
    "season": "Волна по времени: сезонность",
    "lev_ok": "Точка далеко по X, но лежит на линии: рычаг без влияния",
    "typo": "Точка далеко по X и в стороне от линии: тянет наклон (Cook's D &gt; 1)",
    "out_y": "Большой остаток в середине по X: выброс по Y, наклон почти не двигает",
    "ok": "Узора нет: обычный шум малой выборки",
}
PAT_FAM = {"lin": ["quad", "sat", "brk", "expo", "sig", "grp", "inter"], "norm": ["rskew", "lskew", "heavy", "grp"],
           "type": ["diamond", "count", "binlm", "expo"], "pt": ["lev_ok", "typo", "out_y", "ok"]}


def pattern_q(rng, key, fam, ex):
    pool = [k for k in PAT_FAM[fam] if k != key]
    rng.shuffle(pool)
    extra = [k for k in PAT if k not in pool and k != key and k not in ("which",)]
    rng.shuffle(extra)
    wr = [PAT[k] for k in (pool + extra)[:3]]
    return mc(rng, "Какой это узор?", PAT[key], wr, ex)


def fix_q(rng, right, wrong, ex):
    return mc(rng, "Какой код это исправит?", code(right), [code(w) for w in wrong], ex)


# ---------------------------------------------------------------------------
# contexts
LINCTX = {
    "quad": [dict(y="adg", x="temp", yru="привес (г/сут)", xru="температура в секции (°C)", xr=(12, 32), d="barn", opt=(19, 22), top=880, curv=2.3, s=38),
             dict(y="adg", x="ca", yru="привес (г/сут)", xru="кальций в рационе (%)", xr=(0.45, 1.15), d="trial", opt=(0.75, 0.85), top=860, curv=2600, s=32),
             dict(y="born", x="age_m", yru="родилось поросят (среднее по свиноматке)", xru="возраст свиноматки (мес.)", xr=(9, 60), d="sows", opt=(30, 38), top=14.2, curv=0.0035, s=0.9)],
    "sat": [dict(y="feed", x="bw", yru="потребление корма (кг/сут)", xru="живая масса (кг)", xr=(12, 130), d="pigs", a=-1.4, b=0.85, s=0.17),
            dict(y="igg", x="colostrum", yru="IgG в сыворотке (г/л)", xru="выпито молозива (мл)", xr=(40, 480), d="piglets", a=-38, b=10.5, s=2.6),
            dict(y="adg", x="space", yru="привес (г/сут)", xru="площадь на голову (м²)", xr=(0.25, 1.4), d="pens", a=905, b=170, s=18)],
    "brk": [dict(y="adg", x="lys", yru="привес (г/сут)", xru="SID-лизин (%)", xr=(0.7, 1.3), d="trial", c=(0.98, 1.08), kind="plateau", b=650, base=700, s=24),
            dict(y="water", x="temp", yru="потребление воды (л/сут)", xru="температура (°C)", xr=(14, 34), d="barn", c=(22, 25), kind="rise", b=0.55, base=6.0, s=0.45),
            dict(y="resp", x="temp", yru="частота дыхания (в мин)", xr=(15, 33), xru="температура (°C)", d="sows", c=(23, 26), kind="rise", b=6.5, base=28, s=4.5)],
    "expo": [dict(y="cfu", x="hours", yru="бактерии в жидком корме (тыс. КОЕ/г)", xru="часы хранения", xr=(0, 24), d="feed", a=8, k=0.13, s=0.28),
             dict(y="wt", x="age", yru="масса поросёнка (кг)", xru="возраст (дни)", xr=(21, 70), d="nursery", a=1.9, k=0.042, s=0.12)],
    "sig": [dict(y="wt", x="age", yru="живая масса (кг)", xru="возраст (дни)", xr=(5, 250), d="growth", A=255, s=5.0)],
    "grp": [dict(y="bf", x="hcw", g="sex", gl=("barrow", "gilt"), yru="толщина шпика (мм)", xru="масса туши (кг)", xr=(78, 112), d="carcass", b0=-4, b1=0.17, dg=-3.6, s=1.1),
            dict(y="adg", x="start_wt", g="farm", gl=("F1", "F2"), yru="привес (г/сут)", xru="масса при постановке (кг)", xr=(18, 36), d="pigs", b0=520, b1=10, dg=95, s=26),
            dict(y="wean", x="birth", g="foster", gl=("own", "foster"), yru="масса при отъёме (кг)", xru="масса при рождении (кг)", xr=(0.8, 2.0), d="piglets", b0=1.8, b1=3.0, dg=-1.0, s=0.32)],
    "inter": [dict(y="adg", x="start_wt", g="line", gl=("L1", "L2"), yru="привес (г/сут)", xru="масса при постановке (кг)", xr=(18, 38), d="pigs", b0=560, b1=(14, 1.5), s=24),
              dict(y="water", x="temp", g="stage", gl=("gest", "lact"), yru="потребление воды (л/сут)", xru="температура (°C)", xr=(14, 32), d="sows", b0=12, b1=(0.15, 1.1), s=1.1),
              dict(y="titer", x="dose", g="vaccine", gl=("VA", "VB"), yru="титр (log2)", xru="доза (мл)", xr=(0.5, 3.0), d="sera", b0=4.5, b1=(1.6, 0.1), s=0.45)],
}
FIX = {
    "quad": ("lm({y} ~ {x} + I({x}^2), data = {d})", ["lm(log({y}) ~ {x}, data = {d})", "lm({y} ~ log({x}), data = {d})", "glm({y} ~ {x}, family = poisson, data = {d})"],
             "Оптимум описывает квадратичный член. log не создаёт максимума — только сгибает связь в одну сторону."),
    "sat": ("lm({y} ~ log({x}), data = {d})", ["lm({y} ~ {x} + I({x}^2), data = {d})", "lm(log({y}) ~ {x}, data = {d})", "lm({y} ~ {x}, weights = {x}, data = {d})"],
            "Убывающая отдача: каждый следующий шаг X даёт меньше. log(X) выпрямляет такую связь. Парабола тоже изогнётся, но на краю она начнёт падать — для насыщения это неправда."),
    "brk": ("segmented::segmented(lm({y} ~ {x}, data = {d}), seg.Z = ~ {x})", ["lm({y} ~ {x} + I({x}^2), data = {d})", "lm(log({y}) ~ {x}, data = {d})", "lm({y} ~ log({x}), data = {d})"],
            "Ломаная регрессия (broken-line) ищет точку излома. Так в кормлении свиней оценивают потребность: до излома привес растёт, после — плато. Парабола «размажет» излом и даст неверный оптимум."),
    "expo": ("lm(log({y}) ~ {x}, data = {d})", ["lm({y} ~ {x} + I({x}^2), data = {d})", "lm({y} ~ log({x}), data = {d})", "lm({y} ~ {x}, data = {d}, subset = {y} &lt; 100)"],
             "Экспоненциальный рост с мультипликативной ошибкой: log(Y) выпрямляет связь и выравнивает разброс сразу. Наклон потом читают в процентах за единицу X."),
    "sig": ("lm({y} ~ splines::ns({x}, df = 3), data = {d})", ["lm({y} ~ {x} + I({x}^2), data = {d})", "lm(log({y}) ~ {x}, data = {d})", "lm({y} ~ log({x}), data = {d})"],
            "У S-кривой два изгиба — парабола описывает только один. Нужен сплайн или нелинейная модель роста (Гомперц, логистическая через nls)."),
    "grp": ("lm({y} ~ {x} + {g}, data = {d})", ["lm({y} ~ {x} + I({x}^2), data = {d})", "lm(log({y}) ~ {x}, data = {d})", "lm({y} ~ {x} * {x}, data = {d})"],
            "Две параллельные полосы — два сдвинутых облака: у групп одинаковый наклон, но разный уровень. Добавляют фактор группы."),
    "inter": ("lm({y} ~ {x} * {g}, data = {d})", ["lm({y} ~ {x} + {g}, data = {d})", "lm(log({y}) ~ {x}, data = {d})", "lm({y} ~ {x} + I({x}^2), data = {d})"],
              "Наклоны разные — нужно взаимодействие x × группа. Фактор без взаимодействия даст только сдвиг."),
}
WHYMSG = {
    "quad": "Красная линия на Residuals vs Fitted — дуга: прямая проходит мимо вершины.",
    "sat": "Остатки отрицательные на краях и положительные в середине, но дуга несимметрична: круче слева, где связь растёт быстрее всего.",
    "brk": "Остатки лежат двумя прямыми кусками с углом: до порога связь одна, после — другая.",
    "expo": "Дуга вверх и одновременно «труба»: при больших прогнозах и среднее, и разброс растут быстрее, чем допускает прямая.",
    "sig": "Остатки делают волну с двумя изгибами — признак S-образной связи.",
    "grp": "Остатки разбиты на две полосы: одна группа систематически выше. На Q-Q середина «проваливается»: распределение двугорбое.",
    "inter": "Облако остатков расходится веером, а на диаграмме видно две прямые с разным наклоном.",
}


def gen_shape(rng, key, c):
    """Simulate one data set with the planted shape and accept it only if numbers agree."""
    for _ in range(400):
        n = int(rng.integers(45, 95))
        x = rng.uniform(*c["xr"], n)
        g = None
        if key == "quad":
            o = rng.uniform(*c["opt"])
            mu = c["top"] - c["curv"] * (x - o) ** 2
            y = mu + rng.normal(0, c["s"], n)
        elif key == "sat":
            y = c["a"] + c["b"] * np.log(x) + rng.normal(0, c["s"], n)
        elif key == "brk":
            cc = rng.uniform(*c["c"])
            if c["kind"] == "plateau":
                y = c["base"] + c["b"] * (np.minimum(x, cc) - c["xr"][0]) + rng.normal(0, c["s"], n)
            else:
                y = c["base"] + c["b"] * np.maximum(x - cc, 0) + rng.normal(0, c["s"], n)
        elif key == "expo":
            y = c["a"] * np.exp(c["k"] * x + rng.normal(0, c["s"], n))
        elif key == "sig":
            x = rng.uniform(*c["xr"], n)
            b = math.log(c["A"] / 1.5)
            k = math.log(b) / rng.uniform(115, 140)
            y = c["A"] * np.exp(-b * np.exp(-k * x)) + rng.normal(0, c["s"], n)
        elif key == "grp":
            g = (rng.random(n) < 0.5).astype(int)
            y = c["b0"] + c["b1"] * x + c["dg"] * g + rng.normal(0, c["s"], n)
        elif key == "inter":
            g = (rng.random(n) < 0.5).astype(int)
            b1 = np.where(g == 1, c["b1"][1], c["b1"][0])
            xm = np.mean(c["xr"])
            y = c["b0"] + b1 * (x - xm) + c["b1"][0] * xm + rng.normal(0, c["s"], n)
        X = np.c_[np.ones(n), x]
        F = fit(X, y)
        ch = basic(F, X, xq=x)
        A = shapes(x, y)
        ok = False
        if key == "quad":
            ok = ch["quad_p"] < 1e-5 and A["quad"] < min(A.get("logx", 1e9), A["brk"] - 2, A.get("logy", 1e9)) - 4 and ch["shapiro"] > 0.02
        elif key == "sat":
            ok = ch["quad_p"] < 1e-4 and A["logx"] < min(A["quad"], A["brk"]) - 4 and ch["shapiro"] > 0.02
        elif key == "brk":
            ok = A["brk"] < min(A["quad"], A["cub"], A.get("logx", 1e9), A.get("logy", 1e9)) - 4 and A["brk"] < A["lin"] - 20 and ch["shapiro"] > 0.02
        elif key == "expo":
            ok = A["logy"] < min(A["quad"], A["lin"]) - 6 and ch["quad_p"] < 1e-4 and abs(ch["het"]) > 0.3 and ch["het_p"] < 1e-3
        elif key == "sig":
            ok = A["cub"] < A["quad"] - 8 and A["cub"] < A["lin"] - 20
        elif key == "grp":
            Xg = np.c_[X, g]
            Fg = fit(Xg, y)
            ok = ch["quad_p"] > 0.1 and stats.shapiro(F["rs"]).pvalue < 0.05 and aic(Fg) < aic(F) - 20 and abs(ch["het"]) < 0.2
        elif key == "inter":
            Xi = np.c_[X, g, g * x]
            Xa = np.c_[X, g]
            ok = aic(fit(Xi, y)) < aic(fit(Xa, y)) - 15 and ch["quad_p"] > 0.01
        if ok:
            return x, y, g, F, ch, n
    raise RuntimeError(key)


def lin_item(rng, key, c):
    x, y, g, F, ch, n = gen_shape(rng, key, c)
    show_xy = rng.random() < (0.85 if key in ("inter", "grp") else 0.5)
    plots = []
    gtxt = ""
    if show_xy:
        plots.append(P_xy(x, y, F, g=g if key in ("inter", "grp") else None, xl=c["x"], yl=c["y"], title=f"{c['y']} vs {c['x']}"))
        if key in ("inter", "grp"):
            gtxt = f" Кружки — {c['gl'][0]}, треугольники — {c['gl'][1]}."
    plots += [P_rvf(F), P_qq(F)]
    codel = f"> m <- lm({c['y']} ~ {c['x']}, data = {c['d']})"
    if show_xy:
        codel = f"> plot({c['y']} ~ {c['x']}, data = {c['d']}" + (f", pch = ifelse({c['g']} == \"{c['gl'][0]}\", 1, 17))" if key in ("inter", "grp") else ")") + "; abline(lm(" + c["y"] + " ~ " + c["x"] + ", data = " + c["d"] + "))\n" + codel
    codel += "\n> par(mfrow = c(1, 2))\n> plot(m, which = 1:2)"
    story = f"{c['yru'][0].upper() + c['yru'][1:]} ~ {c['xru']}, n = {n}; простая линейная регрессия."
    if key in ("inter", "grp"):
        story += f" В данных есть столбец {c['g']} ({c['gl'][0]} / {c['gl'][1]}), но в модель его не включили.{gtxt}"
    correct = [LIN]
    why = WHYMSG[key]
    if key == "expo":
        correct = [LIN, VAR]
        if ch["shapiro"] < 1e-3 and ch["skew"] > 0.6:
            correct.append(NORM)
    if key == "grp":
        correct = [NORM]
        why += " Это не нелинейность (наклон верный) и не «плохие» данные: в модели не хватает переменной."
    if key == "inter":
        correct = [LIN]
        if abs(ch["het"]) > 0.3 and ch["het_p"] < 1e-3:
            correct.append(VAR)
            why += " Разброс по Residuals vs Fitted растёт — это тоже следствие пропущенного взаимодействия."
        why += " Средняя связь (форма) задана неверно: наклон зависит от группы."
    if key == "sig" and abs(ch["het"]) > 0.3 and ch["het_p"] < 1e-3:
        correct.append(VAR)
    if key in ("quad", "sat", "brk", "sig") and (ch["shapiro"] < 1e-3 and abs(ch["skew"]) > 0.8):
        correct.append(NORM)
    fam = "lin"
    r, wr, ex = FIX[key]
    fmt_ = dict(y=c["y"], x=c["x"], d=c["d"], g=c.get("g", "g"))
    qs = [multi_q(correct, why), pattern_q(rng, key, fam, why), fix_q(rng, r.format(**fmt_), [w.format(**fmt_) for w in wr], ex)]
    # 4th question: reading
    if key == "brk":
        A = broken(x, y)
        cc = A[1]
        qs.append(dict(t="num", q=f"На какой X примерно излом ({c['x']})? Оцени по графику{'' if show_xy else ' остатков'}.", a=float(f"{cc:.3g}"),
                       tol=float(f"{(c['xr'][1] - c['xr'][0]) * 0.09:.3g}"),
                       ex=f"Лучшая ломаная (перебор точки излома) даёт излом около <b>{fmt(cc)}</b>. Принимается ответ ±{fmt((c['xr'][1] - c['xr'][0]) * 0.09)}. "
                          "В кормлении эту точку называют потребностью: дальше прибавка лизина привес не повышает."))
    elif key == "quad":
        xc = x - x.mean()
        Fq = fit(np.c_[np.ones(n), xc, xc ** 2], y)
        vx = x.mean() - Fq["b"][1] / (2 * Fq["b"][2])
        qs.append(dict(t="num", q=f"Модель с I({c['x']}^2) дала оптимум −b₁ / (2·b₂). Где примерно вершина ({c['x']})? Оцени по графику.",
                       a=float(f"{vx:.3g}"), tol=float(f"{(c['xr'][1] - c['xr'][0]) * 0.1:.3g}"),
                       ex=f"Вершина параболы = <b>{fmt(vx)}</b> (±{fmt((c['xr'][1] - c['xr'][0]) * 0.1)} засчитывается). "
                          "Для температуры это зона комфорта: выше и ниже привес падает."))
    elif key == "expo":
        Fl = fit(np.c_[np.ones(n), x], np.log(y))
        pct = 100 * (math.exp(Fl["b"][1]) - 1)
        qs.append(dict(t="num", q=f"В модели log({c['y']}) ~ {c['x']} наклон = {Fl['b'][1]:.4f}. На сколько процентов растёт {c['y']} на +1 {c['x']}?",
                       a=float(f"{pct:.3g}"), tol=float(f"{max(0.3, abs(pct) * 0.03):.2g}"),
                       ex=f"(exp({Fl['b'][1]:.4f}) − 1) × 100 = <b>{fmt(pct)}</b>%. При малом наклоне ≈ наклон × 100."))
    elif key in ("grp", "inter"):
        qs.append(mc(rng, "Почему так бывает чаще всего?", "Две разные совокупности смешаны в одну модель без учёта группы",
                     ["Измерения неточные", "Слишком мало животных", "R неправильно посчитал остатки"],
                     "Пол, ферма, линия, физиологическое состояние — частые «невидимые» группы. Перед моделью полезно раскрасить график по каждому фактору."))
    else:
        imp = mc(rng, "Что будет, если оставить прямую?", "Наклон и прогнозы будут систематически неверны в разных частях диапазона",
                 ["Только p-value чуть изменится", "Ничего, если R² высокий", "Только интервалы прогноза станут шире"],
                 "Неверная форма связи искажает сам эффект: где-то прямая завышает, где-то занижает. Высокий R² этого не исправляет.")
        qs.append(imp)
    return dict(d="slr", f=key, story=story, code=codel, plots=plots, qs=qs)


# ---------------------------------------------------------------------------
# which predictor (multiple regression)
WCTX = [dict(y="adg", d="pigs", xs=["start_wt", "temp", "lys"], xr=[(18, 38), (12, 32), (0.8, 1.25)], b=[380, 10, 0, 320], quad=1, opt=21, curv=2.2, s=34,
             ru="привес (г/сут) от массы при постановке, температуры секции и SID-лизина"),
        dict(y="fcr", d="pens", xs=["energy", "temp", "days_sick"], xr=[(9.4, 10.9), (14, 31), (0, 9)], b=[4.6, -0.13, 0, 0.025], quad=1, opt=20, curv=0.0028, s=0.07,
             ru="конверсия корма от энергии рациона, температуры и дней болезни"),
        dict(y="wean_wt", d="litters", xs=["birth_wt", "lact_len", "parity"], xr=[(0.9, 1.8), (18, 30), (1, 7)], b=[0.2, 2.6, 0.25, 0], quad=2, opt=3.8, curv=0.09, s=0.35,
             ru="масса при отъёме от массы при рождении, длины лактации и номера опороса")]


def which_item(rng, c):
    for _ in range(400):
        n = int(rng.integers(60, 110))
        Xs = [rng.uniform(*r, n) for r in c["xr"]]
        if c["xs"][2] == "parity" or c["xs"][1] == "parity":
            j = c["xs"].index("parity")
            Xs[j] = rng.integers(1, 8, n).astype(float)
        mu = c["b"][0] + sum(b * x for b, x in zip(c["b"][1:], Xs))
        q = c["quad"]
        mu = mu - np.sign(c["b"][0] if c["y"] != "fcr" else -1) * c["curv"] * (Xs[q] - c["opt"]) ** 2 * (1 if c["y"] != "fcr" else -1)
        y = mu + rng.normal(0, c["s"], n)
        X = np.c_[np.ones(n)] if False else np.column_stack([np.ones(n)] + Xs)
        F = fit(X, y)
        ps = []
        for j in range(3):
            xc = Xs[j] - Xs[j].mean()
            Fq = fit(np.c_[X, xc ** 2], y)
            se = math.sqrt(Fq["s"] ** 2 * np.linalg.inv(np.c_[X, xc ** 2].T @ np.c_[X, xc ** 2])[-1, -1])
            ps.append(2 * stats.t.sf(abs(Fq["b"][-1] / se), n - 5))
        if ps[q] < 1e-6 and all(ps[j] > 0.1 for j in range(3) if j != q) and stats.shapiro(F["rs"]).pvalue > 0.01:
            break
    else:
        raise RuntimeError("which")
    plots = [P_rvx(F, Xs[j], c["xs"][j]) for j in range(3)] + [P_rvf(F)]
    form = f"{c['y']} ~ {' + '.join(c['xs'])}"
    code_ = f"> m <- lm({form}, data = {c['d']})\n> car::residualPlots(m, tests = FALSE)   # остатки против каждого X\n> plot(m, which = 1)"
    bad = c["xs"][q]
    story = f"Множественная регрессия: {c['ru']}; n = {n}. Residuals vs Fitted показывает дугу — нужно найти, какой предиктор виноват."
    qs = [multi_q([LIN], "Дуга на остатках — нарушена форма связи. Остальное в порядке."),
          mc(rng, "Какой предиктор надо изменить?", code(bad), [code(x) for x in c["xs"] if x != bad] + [code("Все три одновременно")],
             f"Только на графике против {bad} красная линия изгибается. Против остальных — горизонтальная полоса. "
             "Residuals vs Fitted показывает, что форма не та, но не говорит где; графики против каждого X — говорят."),
          fix_q(rng, f"lm({c['y']} ~ " + " + ".join(x if x != bad else f"{x} + I({x}^2)" for x in c["xs"]) + f", data = {c['d']})",
                [f"lm(log({c['y']}) ~ {' + '.join(c['xs'])}, data = {c['d']})",
                 f"lm({c['y']} ~ " + " + ".join(x if x != c['xs'][(q + 1) % 3] else f"{x} + I({x}^2)" for x in c["xs"]) + f", data = {c['d']})",
                 f"lm({c['y']} ~ {' * '.join(c['xs'])}, data = {c['d']})"],
                "Кривизну добавляют только туда, где она есть. log(Y) изогнул бы связь со всеми X сразу; взаимодействия всех со всеми — не про форму."),
          mc(rng, "Что делает <code>car::residualPlots</code>?", "Рисует остатки против каждого предиктора отдельно",
             ["Удаляет выбросы", "Проверяет нормальность", "Строит модель без лишних X"], "Это главный способ найти «виноватый» X во множественной регрессии. В базовом R то же: plot(pigs$x, resid(m)).")]
    return dict(d="mlr", f="which", story=story, code=code_, plots=plots, qs=qs)


# ---------------------------------------------------------------------------
# normality shapes
NCTX = [dict(y="adg", x="start_wt", yru="привес (г/сут)", xru="масса при постановке (кг)", xr=(18, 36), d="pigs", b0=520, b1=10, s=40),
        dict(y="dress", x="hcw", yru="убойный выход (%)", xru="масса туши (кг)", xr=(80, 112), d="carcass", b0=67, b1=0.06, s=0.9),
        dict(y="ph24", x="hcw", yru="pH через 24 ч", xru="масса туши (кг)", xr=(80, 112), d="carcass", b0=5.3, b1=0.003, s=0.07)]


def norm_item(rng, key):
    for _ in range(600):
        c = NCTX[int(rng.integers(len(NCTX)))] if key != "lskew" else NCTX[1]
        n = int(rng.integers(40, 100))
        x = rng.uniform(*c["xr"], n)
        if key == "rskew":
            z = rng.lognormal(0, 0.85, n)
            e = c["s"] * (z - z.mean()) / z.std()
        elif key == "lskew":
            z = -rng.lognormal(0, 0.85, n)
            e = c["s"] * (z - z.mean()) / z.std()
        else:
            e = c["s"] * rng.standard_t(2.5, n) / 1.6
        y = c["b0"] + c["b1"] * x + e
        X = np.c_[np.ones(n), x]
        F = fit(X, y)
        ch = basic(F, X, xq=x)
        if ch["quad_p"] < 0.1 or abs(ch["het"]) > 0.2:
            continue
        sk = ch["skew"]
        if key == "rskew" and ch["shapiro"] < 1e-4 and sk > 1.0:
            break
        if key == "lskew" and ch["shapiro"] < 1e-4 and sk < -1.0:
            break
        if key == "heavy" and ch["shapiro"] < 1e-3 and abs(sk) < 0.35 and stats.kurtosis(F["rs"]) > 2.5:
            break
    else:
        raise RuntimeError(key)
    story = f"{c['yru'][0].upper() + c['yru'][1:]} ~ {c['xru']}, n = {n}."
    code_ = f"> m <- lm({c['y']} ~ {c['x']}, data = {c['d']})\n> par(mfrow = c(1, 2))\n> plot(m, which = 1:2)"
    plots = [P_rvf(F), P_qq(F)]
    why = {"rskew": "Q-Q: правый конец загибается вверх — несколько остатков намного больше ожидаемого. Слева точки чуть выше прямой. Это правая асимметрия.",
           "lskew": "Q-Q: левый конец уходит вниз — несколько очень низких значений. Справа хвост короткий (точки ниже прямой). Это левая асимметрия.",
           "heavy": "Q-Q: оба конца уходят от прямой в разные стороны (слева вниз, справа вверх), середина ровная. Хвосты тяжелее нормальных: крайних значений больше, чем ждёт нормальное распределение."}[key]
    fixes = {"rskew": ("lm(log({y}) ~ {x}, data = {d})", ["lm({y} ~ {x} + I({x}^2), data = {d})", "lm(exp({y}) ~ {x}, data = {d})", "lm({y} ~ log({x}), data = {d})"],
                       "Правый хвост сжимает логарифм (если Y &gt; 0). Логарифм X меняет форму связи, а не распределение остатков."),
             "lskew": ("lm({y} ~ {x}, data = {d})   # и проверить животных с низкими значениями", ["lm(log({y}) ~ {x}, data = {d})", "lm({y} ~ {x} + I({x}^2), data = {d})", "lm(exp(-{y}) ~ {x}, data = {d})"],
                       "Логарифм усиливает левый хвост, а не лечит. Низкий выход туши часто у больных или травмированных животных — сначала смотрят, кто это. При n ≥ 30 оценки наклона устойчивы; если хвост от отдельной подгруппы — её добавляют в модель или анализируют отдельно."),
             "heavy": ("MASS::rlm({y} ~ {x}, data = {d})", ["lm(log({y}) ~ {x}, data = {d})", "lm({y} ~ {x} + I({x}^2), data = {d})", "glm({y} ~ {x}, family = poisson, data = {d})"],
                       "Тяжёлые хвосты симметричны — логарифм не поможет. Робастная регрессия (rlm) уменьшает вес далёких точек. Проверь и данные: тяжёлые хвосты иногда — смесь двух групп.")}
    r, wr, ex = fixes[key]
    f_ = dict(y=c["y"], x=c["x"], d=c["d"])
    qs = [multi_q([NORM], why), pattern_q(rng, key, "norm", why), fix_q(rng, r.format(**f_), [w.format(**f_) for w in wr], ex),
          mc(rng, "Насколько это опасно для вывода о наклоне?", "При n ≈ " + str(n) + " умеренно: оценка не смещена, p примерно верны; страдают интервалы прогноза",
             ["Очень: наклон смещён к нулю", "Совсем не опасно ни для чего", "Опасно только для R²"],
             "Нормальность — самая «мягкая» предпосылка: при умеренном n работает ЦПТ. Но интервал прогноза для отдельного животного опирается на форму распределения.")]
    return dict(d="slr", f=key, story=story, code=code_, plots=plots, qs=qs)


# ---------------------------------------------------------------------------
# wrong response type
def type_item(rng, key):
    for _ in range(600):
        n = int(rng.integers(50, 120))
        if key == "count":
            x = rng.uniform(2, 10, n)
            y = rng.poisson(np.exp(-1.3 + 0.24 * x)).astype(float)
            c = dict(y="stillborn", x="farrow_h", yru="мертворождённых в помёте", xru="длительность опороса (ч)", d="litters")
        elif key == "diamond":
            x = rng.uniform(0.45, 1.25, n)
            p = 1 / (1 + np.exp(-(-6.0 + 7.5 * x)))
            m = 25
            y = rng.binomial(m, p) / m
            c = dict(y="p_sick", x="density", yru="доля поросят с диареей в загоне (из 25)", xru="плотность (гол/м²)", d="pens")
        else:
            x = np.round(rng.uniform(0.6, 2.1, n), 2)
            p = 1 / (1 + np.exp(-(3.0 - 3.6 * x)))
            y = (rng.random(n) < p).astype(float)
            c = dict(y="died", x="birth_wt", yru="пал до отъёма (0/1)", xru="масса при рождении (кг)", d="piglets")
        X = np.c_[np.ones(n), x]
        F = fit(X, y)
        ch = basic(F, X, xq=x)
        if key == "count" and abs(ch["het"]) > 0.25 and ch["het_p"] < 0.01 and y.max() <= 9 and (y == 0).mean() > 0.15:
            break
        if key == "diamond" and 0.05 < y.min() + 0.06 and y.max() > 0.85 and y.min() < 0.12:
            break
        if key == "binlm" and 0.1 < y.mean() < 0.5 and ch["shapiro"] < 0.001:
            break
    else:
        raise RuntimeError(key)
    correct = []
    if ch["quad_p"] < 1e-3:
        correct.append(LIN)
    if (abs(ch["het"]) > 0.2 and ch["het_p"] < 0.01) or key in ("binlm", "diamond"):
        correct.append(VAR)
    if ch["shapiro"] < 1e-3:
        correct.append(NORM)
    if not correct:
        correct = [VAR]
    story = f"{c['yru'][0].upper() + c['yru'][1:]} ~ {c['xru']}, n = {n}; кто-то подогнал обычную lm."
    code_ = f"> m <- lm({c['y']} ~ {c['x']}, data = {c['d']})\n> par(mfrow = c(1, 2))\n> plot(m, which = 1:2)"
    plots = [P_rvf(F), P_qq(F)]
    names = {LIN: "форма", VAR: "дисперсии", NORM: "нормальность"}
    why = {"count": "Отклик — маленькие целые числа. Для каждого значения Y остатки лежат на своей косой линии (Y − прогноз), отсюда косые полосы; разброс растёт со средним, как у счётных данных.",
           "diamond": "Доля не может выйти за 0 и 1: у краёв разброс сжимается, в середине (p ≈ 0,5) — самый большой. Дисперсия доли = p(1 − p)/n зависит от среднего.",
           "binlm": "У 0/1-отклика для каждого прогноза остаток может быть только 0 − ŷ или 1 − ŷ: две косые линии. Дисперсия p(1 − p) зависит от прогноза, нормальности нет и быть не может."}[key]
    why += " Нарушено: " + ", ".join(names[k] for k in sorted(correct)) + ". Лечится не преобразованием, а правильной моделью."
    fixes = {"count": ("glm(stillborn ~ farrow_h, family = poisson, data = litters)", ["lm(log(stillborn) ~ farrow_h, data = litters)", "lm(stillborn ~ farrow_h + I(farrow_h^2), data = litters)",
                                                                                         "glm(stillborn ~ farrow_h, family = binomial, data = litters)"],
                       "Счёт → Пуассон (или отрицательная биномиальная при сверхдисперсии). log(0) не существует — log(Y) для счёта с нулями не работает."),
             "diamond": ("glm(cbind(sick, 25 - sick) ~ density, family = binomial, data = pens)", ["lm(p_sick ~ density + I(density^2), data = pens)", "lm(log(p_sick) ~ density, data = pens)",
                                                                                                   "glm(p_sick ~ density, family = poisson, data = pens)"],
                         "Доля из известного n — биномиальная модель с cbind(успехи, неудачи). Она знает, что разброс доли максимален около 0,5."),
             "binlm": ("glm(died ~ birth_wt, family = binomial, data = piglets)", ["lm(log(died) ~ birth_wt, data = piglets)", "glm(died ~ birth_wt, family = poisson, data = piglets)",
                                                                                  "lm(died ~ birth_wt + I(birth_wt^2), data = piglets)"],
                       "0/1 → логистическая регрессия. Прямая для вероятности ещё и выходит за 0 и 1 при крайних X.")}
    r, wr, ex = fixes[key]
    qs = [multi_q(correct, why), pattern_q(rng, key, "type", why), fix_q(rng, r, wr, ex)]
    if key == "binlm":
        pr_ = F["b"][0] + F["b"][1] * 2.3
        qs.append(dict(t="num", q=f"Прямая lm: {fmt(F['b'][0])} + ({fmt(F['b'][1])}) × birth_wt. Какую «вероятность гибели» она даст для поросёнка 2,3 кг?",
                       a=float(f"{pr_:.3g}"), tol=float(f"{max(0.01, abs(pr_) * 0.05):.2g}"),
                       ex=f"{fmt(F['b'][0])} + ({fmt(F['b'][1])}) × 2,3 = <b>{fmt(pr_)}</b>. " + ("Отрицательная вероятность — абсурд. " if pr_ < 0 else "") +
                          "Логистическая кривая всегда остаётся между 0 и 1."))
    elif key == "count":
        qs.append(mc(rng, "Почему не log(stillborn)?", "Много нулей: log(0) = −∞, такие строки R выкинет или выдаст ошибку",
                     ["Можно, это стандартный путь", "Потому что X не логарифмирован", "Потому что R² станет меньше"],
                     f"Нулей здесь {int((y == 0).sum())} из {n}. Приём log(Y + 1) существует, но коэффициенты потом трудно читать; glm с Пуассоном — прямой путь."))
    else:
        qs.append(mc(rng, "Почему дисперсия доли больше в середине?", "Дисперсия доли из n равна p(1 − p)/n: максимум при p = 0,5",
                     ["Из-за выбросов", "Так получилось случайно", "Потому что в середине больше загонов"],
                     "При p = 0,05 у доли почти некуда «гулять», при p = 0,5 разброс максимален. Обычная lm считает разброс постоянным."))
    return dict(d="glm", f=key, story=story, code=code_, plots=plots, qs=qs)


# ---------------------------------------------------------------------------
# seasonality
def season_item(rng):
    for _ in range(600):
        n = 36 + 12 * int(rng.integers(0, 2))
        mon = np.arange(n)
        x = rng.uniform(19, 28, n)
        amp = rng.uniform(0.7, 1.2)
        ph = rng.uniform(0, 2 * math.pi)
        y = 6.5 - 0.12 * x + amp * np.sin(2 * math.pi * mon / 12 + ph) + rng.normal(0, 0.35, n)
        X = np.c_[np.ones(n), x]
        F = fit(X, y)
        ch = basic(F, X, xq=x, order=mon)
        r = F["r"]
        lag12 = float(np.corrcoef(r[:-12], r[12:])[0, 1])
        if ch["lag1"] > 0.5 and lag12 > 0.4 and ch["quad_p"] > 0.05:
            break
    else:
        raise RuntimeError("season")
    story = f"Падёж на доращивании (%) по месяцам ~ средний возраст отъёма (дни); {n} месяцев одной фермы подряд."
    code_ = "> m <- lm(mort ~ wean_age, data = monthly)\n> par(mfrow = c(1, 2))\n> plot(m, which = 1)\n> plot(resid(m), type = \"b\", xlab = \"Месяц\")"
    plots = [P_rvf(F), P_ord(F)]
    why = "Остатки по порядку месяцев идут волной с периодом около 12: зимой падёж выше, летом ниже. Соседние месяцы похожи — независимость нарушена."
    qs = [multi_q([IND], why), pattern_q(rng, "season", "pt", why),
          fix_q(rng, "lm(mort ~ wean_age + factor(month), data = monthly)", ["lm(log(mort) ~ wean_age, data = monthly)", "lm(mort ~ wean_age + I(wean_age^2), data = monthly)",
                                                                             "lm(mort ~ wean_age, data = monthly, subset = month %in% 5:9)"],
                "Сезон — пропущенная переменная. Месяц (или sin/cos от месяца) в модели убирает волну. Если остатки всё равно связаны — модель с автокорреляцией (nlme::gls с corAR1)."),
          dict(t="num", q="Через сколько месяцев волна повторяется? (на графике по порядку)", a=12, tol=1.5,
               ex=f"Пики на графике остатков по порядку разделены примерно <b>12</b> месяцами. Корреляция остатков с лагом 12: {fmt(lag12)}.")]
    return dict(d="slr", f="season", story=story, code=code_, plots=plots, qs=qs)


# ---------------------------------------------------------------------------
# unusual points
PCTX = DB.SLR_CTX


def point_item(rng, key):
    for _ in range(800):
        c = PCTX[int(rng.integers(len(PCTX)))]
        n = int(rng.integers(30, 70))
        x = rng.uniform(*c["xr"], n)
        mu = c["b0"] + c["b1"] * x
        y = mu + rng.normal(0, c["s"], n)
        k = int(rng.integers(n))
        span = c["xr"][1] - c["xr"][0]
        if key == "lev_ok":
            x[k] = c["xr"][1] + span * rng.uniform(0.9, 1.3)
            y[k] = c["b0"] + c["b1"] * x[k] + rng.normal(0, c["s"] * 0.3)
        elif key == "typo":
            x[k] = c["xr"][1] + span * rng.uniform(1.0, 1.5)
            y[k] = c["b0"] + c["b1"] * np.mean(c["xr"]) + rng.normal(0, c["s"] * 0.5)
        else:
            x[k] = np.mean(c["xr"]) + span * rng.uniform(-0.1, 0.1)
            y[k] = c["b0"] + c["b1"] * x[k] + rng.choice([-1, 1]) * c["s"] * rng.uniform(5, 7)
        X = np.c_[np.ones(n), x]
        F = fit(X, y)
        X0 = np.delete(X, k, 0)
        F0 = fit(X0, np.delete(y, k))
        ch = basic(F, X, xq=None)
        h = F["h"][k]
        others = np.delete(F["cook"], k)
        if others.max() > 0.25:
            continue
        if key == "lev_ok" and h > 0.25 and F["cook"][k] < 0.3 and abs(F["rs"][k]) < 1.5 and ch["shapiro"] > 0.05:
            break
        if key == "typo" and F["cook"][k] > 1.5 and h > 0.25:
            break
        if key == "out_y" and abs(F["rs"][k]) > 4 and F["cook"][k] < 0.5 and h < 0.06:
            break
    else:
        raise RuntimeError(key)
    story = f"Простая регрессия: {c['yru']} ~ {c['xru']}, n = {n}. Одна точка (№ {k + 1}) выделяется."
    code_ = (f"> plot({c['y']} ~ {c['x']}, data = {c['data']}); abline(lm({c['y']} ~ {c['x']}, data = {c['data']}))\n"
             f"> m <- lm({c['y']} ~ {c['x']}, data = {c['data']})\n> par(mfrow = c(1, 3))\n> plot(m, which = c(1, 2, 4))")
    plots = [P_xy(x, y, F, xl=c["x"], yl=c["y"]), P_rvf(F), P_qq(F), P_cook(F)]
    db = 100 * (F["b"][1] - F0["b"][1]) / abs(F0["b"][1])
    if key == "lev_ok":
        correct, why = [OK], (f"Точка № {k + 1} далеко по X (рычаг h = {fmt(h)}), но лежит почти на линии остальных: остаток маленький, Cook's D = {fmt(F['cook'][k])}. "
                              f"Без неё наклон меняется на {fmt(abs(db))}% — она не искажает, а подтверждает связь. Нарушений нет.")
        fixr = (f"lm({c['y']} ~ {c['x']}, data = {c['data']})   # оставить как есть", [f"lm({c['y']} ~ {c['x']}, data = {c['data']}[-{k + 1}, ])",
                                                                                     f"lm(log({c['y']}) ~ {c['x']}, data = {c['data']})",
                                                                                     f"MASS::rlm({c['y']} ~ {c['x']}, data = {c['data']})"],
                "Точка, которая согласна с остальными, полезна: она сужает интервал наклона. Удалять её — выбрасывать информацию.")
    elif key == "typo":
        correct, why = [OUT], (f"Точка № {k + 1} далеко по X и не на линии: Cook's D = {fmt(F['cook'][k])} (&gt; 1). Без неё наклон меняется на {fmt(abs(db))}%. "
                               "Сначала — проверить запись (запятая не там? другое животное?).")
        fixr = (f"{c['data']}[{k + 1}, ]   # сначала посмотреть строку и сверить с журналом", [f"lm({c['y']} ~ {c['x']}, data = {c['data']})   # оставить без проверки",
                                                                                          f"lm(log({c['y']}) ~ {c['x']}, data = {c['data']})",
                                                                                          f"lm({c['y']} ~ {c['x']} + I({c['x']}^2), data = {c['data']})"],
                "Влиятельную точку не удаляют вслепую и не «лечат» преобразованием. Ошибка ввода → исправить по журналу; настоящая — показать анализ с ней и без неё.")
    else:
        correct, why = [OUT], (f"Точка № {k + 1}: стандартизованный остаток {fmt(F['rs'][k])}, но она в середине по X (h = {fmt(h)}), поэтому Cook's D всего {fmt(F['cook'][k])}. "
                               f"Наклон без неё меняется на {fmt(abs(db))}% — мало. Она раздувает σ и SE, но почти не двигает линию.")
        fixr = (f"{c['data']}[{k + 1}, ]   # проверить запись; анализ с точкой и без неё", [f"lm({c['y']} ~ {c['x']}, data = {c['data']}[-{k + 1}, ])   # удалить и не сообщать",
                                                                                         f"lm({c['y']} ~ {c['x']} + I({c['x']}^2), data = {c['data']})",
                                                                                         f"glm({c['y']} ~ {c['x']}, family = poisson, data = {c['data']})"],
                "Выброс по Y проверяют так же: ошибка ли это. Если нет — оставляют и показывают, меняются ли выводы.")
    qs = [multi_q(correct, why), pattern_q(rng, key, "pt", why), fix_q(rng, *fixr),
          dict(t="num", q=f"Cook's D точки № {k + 1}?", a=float(f"{F['cook'][k]:.3g}"), tol=float(f"{max(0.02, F['cook'][k] * 0.08):.2g}"),
               ex=f"На графике Cook's distance столбик № {k + 1} = <b>{fmt(F['cook'][k])}</b>. Меньше 0,5 — не страшно; больше 1 — точка одна меняет выводы.")]
    return dict(d="slr", f=key, story=story, code=code_, plots=plots, qs=qs)


# ---------------------------------------------------------------------------
# before / after the fix
def fix_item(rng, key):
    for _ in range(600):
        n = int(rng.integers(45, 90))
        if key == "fixed":
            c = dict(y="titer", x="day", yru="титр антител", xru="дни после вакцинации", d="sera")
            x = rng.uniform(14, 120, n)
            y = 2400 * np.exp(-0.022 * x + rng.normal(0, 0.35, n))
        else:
            c = dict(y="adg", x="temp", yru="привес (г/сут)", xru="температура (°C)", d="barn")
            x = rng.uniform(12, 32, n)
            y = 880 - 2.4 * (x - rng.uniform(19, 22)) ** 2 + rng.normal(0, 35, n)
        X = np.c_[np.ones(n), x]
        F1 = fit(X, y)
        F2 = fit(X, np.log(y))
        c1 = basic(F1, X, xq=x)
        c2 = basic(F2, X, xq=x)
        if key == "fixed" and abs(c1["het"]) > 0.35 and c1["het_p"] < 1e-3 and c1["shapiro"] < 1e-3 and c2["shapiro"] > 0.1 and abs(c2["het"]) < 0.15 and c2["quad_p"] > 0.1:
            break
        if key == "notfixed" and c1["quad_p"] < 1e-6 and c2["quad_p"] < 1e-6:
            break
    else:
        raise RuntimeError(key)
    m1 = f"lm({c['y']} ~ {c['x']})"
    m2 = f"lm(log({c['y']}) ~ {c['x']})"
    plots = [P_rvf(F1, "m1: Residuals vs Fitted"), P_qq(F1, "m1: Q-Q"), P_rvf(F2, "m2: Residuals vs Fitted"), P_qq(F2, "m2: Q-Q")]
    code_ = (f"> m1 <- lm({c['y']} ~ {c['x']}, data = {c['d']})\n> m2 <- lm(log({c['y']}) ~ {c['x']}, data = {c['d']})\n"
             "> par(mfrow = c(2, 2))\n> plot(m1, which = 1:2); plot(m2, which = 1:2)")
    story = f"{c['yru'][0].upper() + c['yru'][1:]} ~ {c['xru']}, n = {n}. Сверху — исходная модель m1, снизу — m2 с log({c['y']})."
    if key == "fixed":
        corr1 = [VAR, NORM] + ([LIN] if c1["quad_p"] < 1e-3 else [])
        why = "В m1 разброс растёт со средним и правый хвост на Q-Q — мультипликативная ошибка. Это видно на верхних графиках."
        pct = 100 * (math.exp(F2["b"][1]) - 1)
        qs = [dict(t="multi", q="Что не так с m1 (верхние графики)?", o=OPTS, a=sorted(set(corr1)), ex=why),
              mc(rng, "Помог ли log(Y)?", "Да: в m2 полоса ровная, Q-Q около прямой", ["Нет: в m2 осталась труба", "Нет: log не лечит дисперсии", "Частично: появилась дуга"],
                 "Нижние графики выглядят как учебный пример без нарушений. Значит, разумно работать с m2."),
              dict(t="num", q=f"В m2 наклон {F2['b'][1]:.5f}. На сколько процентов меняется титр за день?", a=float(f"{pct:.3g}"), tol=float(f"{max(0.05, abs(pct) * 0.04):.2g}"),
                   ex=f"(exp({F2['b'][1]:.5f}) − 1) × 100 = <b>{fmt(pct)}</b>% в день: титр каждый день умножается на {fmt(math.exp(F2['b'][1]))}."),
              mc(rng, "Можно ли сравнить m1 и m2 по R²?", "Нет: у них разные отклики (Y и log Y), R² не сравнимы", ["Да, берём больший R²", "Да, если n одинаковый", "Только по AIC без поправок"],
                 "R² и AIC сравнивают только для одного и того же Y. Преобразование выбирают по остаткам и по смыслу.")]
    else:
        why = "В m1 — дуга (оптимум температуры). В m2 дуга осталась: log сжимает большие значения, но не создаёт и не убирает вершину."
        qs = [dict(t="multi", q="Что не так с m1 (верхние графики)?", o=OPTS, a=[LIN], ex=why),
              mc(rng, "Помог ли log(Y)?", "Нет: дуга на m2 осталась — нужна другая форма связи (квадратичный член)", ["Да: всё исправилось", "Да: дисперсии стали равными, этого достаточно",
                                                                                                                   "Нет: надо взять log ещё раз"],
                 "Преобразование Y лечит разброс и асимметрию, а форму «с оптимумом» описывает только модель с вершиной."),
              fix_q(rng, f"lm({c['y']} ~ {c['x']} + I({c['x']}^2), data = {c['d']})", [f"lm(log(log({c['y']})) ~ {c['x']}, data = {c['d']})", f"lm(sqrt({c['y']}) ~ {c['x']}, data = {c['d']})",
                                                                                           f"lm({c['y']} ~ log({c['x']}), data = {c['d']})"],
                    "Вершину даёт квадратичный член. Корни и логарифмы Y только сжимают шкалу."),
              mc(rng, "Чему учит этот пример?", "Преобразование подбирают под конкретное нарушение и проверяют остатки снова",
                 ["log(Y) — универсальное лекарство", "Если R² вырос — модель правильная", "Остатки после преобразования смотреть не нужно"],
                 "Любое исправление проверяют теми же графиками.")]
    return dict(d="slr", f=key, story=story, code=code_, plots=plots, qs=qs)


# ---------------------------------------------------------------------------
# clean small samples (false alarms)
def ok_item(rng):
    for _ in range(800):
        c = PCTX[int(rng.integers(len(PCTX)))]
        n = int(rng.integers(10, 19))
        x = rng.uniform(*c["xr"], n)
        y = c["b0"] + c["b1"] * x + rng.normal(0, c["s"], n)
        X = np.c_[np.ones(n), x]
        F = fit(X, y)
        ch = basic(F, X, xq=x)
        if ch["shapiro"] > 0.2 and ch["quad_p"] > 0.2 and ch["maxrs"] < 2.4 and F["cook"].max() < 0.6 and (abs(ch["het"]) < 0.3 or ch["het_p"] > 0.2):
            # but a wiggly lowess (looks like something)
            xs, ys = DB.lowess(F["fit"], F["r"])
            if np.ptp(ys) > 0.8 * F["s"]:
                break
    else:
        raise RuntimeError("ok")
    story = f"Простая регрессия: {c['yru']} ~ {c['xru']}, всего n = {n}."
    code_ = f"> m <- lm({c['y']} ~ {c['x']}, data = {c['data']})\n> par(mfrow = c(1, 2))\n> plot(m, which = 1:2)"
    plots = [P_rvf(F), P_qq(F)]
    why = (f"При n = {n} красная линия всегда гуляет, а точки на Q-Q неровные. Проверка числами: квадратичный член p = {fmt(ch['quad_p'])}, "
           f"Шапиро p = {fmt(ch['shapiro'])}, максимальный |станд. остаток| = {fmt(ch['maxrs'])}. Узора нет.")
    qs = [multi_q([OK], why), pattern_q(rng, "ok", "pt", why),
          mc(rng, "Почему у маленькой выборки графики выглядят «хуже»?", "Мало точек — случайные неровности не усредняются; ищут явный узор, а не идеальную картинку",
             ["Маленькая выборка всегда нарушает предпосылки", "Надо добавить квадратичный член на всякий случай", "R рисует малые выборки неточно"],
             "Обратная сторона: при малом n и настоящее нарушение трудно увидеть. Поэтому важны знания о данных (что это за показатель, как он обычно распределён)."),
          mc(rng, "Что делать?", "Ничего не менять: модель подходит; помнить, что при n = " + str(n) + " вывод неточный",
             ["Перейти на непараметрический метод", "Добавить I(x^2)", "Взять log(Y)"], "Не чини то, что не сломано. Широкие ДИ честно покажут неопределённость.")]
    return dict(d="slr", f="ok2", story=story, code=code_, plots=plots, qs=qs)


# ---------------------------------------------------------------------------
PLAN2 = ([("quad", 45), ("sat", 40), ("brk", 50), ("expo", 35), ("sig", 25), ("grp", 40), ("inter", 40)], [("which", 45)],
         [("rskew", 20), ("lskew", 25), ("heavy", 30)], [("count", 35), ("diamond", 25), ("binlm", 25)], 30,
         [("lev_ok", 30), ("typo", 25), ("out_y", 25)], [("fixed", 30), ("notfixed", 20)], 40)
KIND_RU = {"slr": "Простая регрессия", "mlr": "Множественная", "glm": "Тип отклика"}
FAMILY = {"quad": "form", "sat": "form", "brk": "form", "expo": "form", "sig": "form", "grp": "form", "inter": "form", "which": "form",
          "rskew": "norm", "lskew": "norm", "heavy": "norm", "count": "type", "diamond": "type", "binlm": "type", "season": "ind",
          "lev_ok": "pts", "typo": "pts", "out_y": "pts", "fixed": "fix", "notfixed": "fix", "ok2": "pts"}


def build2():
    rng = np.random.default_rng(20261122)
    items = []
    lin, wh, nm, ty, ns, pt, fx, nok = PLAN2
    for key, cnt in lin:
        for i in range(cnt):
            items.append(lin_item(rng, key, LINCTX[key][i % len(LINCTX[key])]))
    for key, cnt in wh:
        for i in range(cnt):
            items.append(which_item(rng, WCTX[i % len(WCTX)]))
    for key, cnt in nm:
        for i in range(cnt):
            items.append(norm_item(rng, key))
    for key, cnt in ty:
        for i in range(cnt):
            items.append(type_item(rng, key))
    for i in range(ns):
        items.append(season_item(rng))
    for key, cnt in pt:
        for i in range(cnt):
            items.append(point_item(rng, key))
    for key, cnt in fx:
        for i in range(cnt):
            items.append(fix_item(rng, key))
    for i in range(nok):
        items.append(ok_item(rng))
    order = np.random.default_rng(20261123).permutation(len(items))
    items = [items[k] for k in order]
    for it in items:
        it["lv"] = 2
        it["fam"] = FAMILY[it["f"]]
        it["title"] = "Уровень 2 · " + {"form": "форма связи", "norm": "распределение", "type": "тип отклика", "ind": "время", "pts": "точки и шум", "fix": "до и после"}[it["fam"]]
    return items


if __name__ == "__main__":
    import sys, time
    t = time.time()
    lvl1 = DB.build()
    for it in lvl1:
        it["lv"] = 1
    lvl2 = build2()
    for i, it in enumerate(lvl2):
        it["id"] = 1001 + i
    data = lvl1 + lvl2
    json.dump(DB.pack(data), open("diagdata.json", "w"), ensure_ascii=False, separators=(",", ":"))
    from collections import Counter
    print(len(lvl1), "+", len(lvl2), "items;", Counter(d["f"] for d in lvl2), round(time.time() - t), "s")
