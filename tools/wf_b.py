"""Workflows, part B: blocks, factorial, 2x2 table, simple and multiple regression."""
import math
import numpy as np
from scipy import stats
import rfmt as R
import htest as H
from models import ols, predict

from gen import ru, rua, mc, num, lm_summary
from gen4 import emm_table, emm_fmt
from wf_lib import (str_df, df_print, agg_print, named, code_pick, multi, step, p_lm, p_xy, p_int, pr, table2, joint_cells,
                    aov_summary, anova_lm, anova2, summary_vec)
from wf_a import box, shuffle_rows, lm_out
import diag_bank as DB


def tukey_rows(levels, means, n, mse, df):
    k = len(levels)
    q = stats.studentized_range.ppf(0.95, k, df)
    rows = []
    for i in range(k):
        for j in range(i + 1, k):
            d = means[j] - means[i]
            se = math.sqrt(mse / n)
            rows.append(dict(name=f"{levels[j]}-{levels[i]}", diff=d, lwr=d - q * se, upr=d + q * se,
                             padj=min(1.0, float(stats.studentized_range.sf(abs(d) / se, k, df)))))
    return rows


def tukey_print(call, fac, rows):
    out = ["  Tukey multiple comparisons of means", "    95% family-wise confidence level", "", f"Fit: {call}", "", f"${fac}"]
    out += R.matrix([r["name"] for r in rows], ["diff", "lwr", "upr", "p adj"],
                    [[r["diff"] for r in rows], [r["lwr"] for r in rows], [r["upr"] for r in rows], [r["padj"] for r in rows]], fixed_dec={3: 7})
    return out


# =====================================================================
# 5. RCBD: rooms as blocks
# =====================================================================
RCBD_VAR = [dict(room_sd=38, eff=[0, 22, 40], sd=13, kind="strong"), dict(room_sd=3, eff=[0, 30, 48], sd=14, kind="weak")]


def wf_rcbd(rng, v):
    lv = ["ctrl", "mcfa", "probio"]
    nb = 6
    for _ in range(5000):
        rb = rng.normal(0, v["room_sd"], nb)
        Y = np.round(780 + rb[:, None] + np.array(v["eff"])[None, :] + rng.normal(0, v["sd"], (nb, 3)))
        gm = Y.mean()
        ss_room = 3 * ((Y.mean(1) - gm) ** 2).sum()
        ss_diet = nb * ((Y.mean(0) - gm) ** 2).sum()
        ss_tot = ((Y - gm) ** 2).sum()
        ss_res = ss_tot - ss_room - ss_diet
        df_res = (nb - 1) * 2
        mse = ss_res / df_res
        p_diet = stats.f.sf(ss_diet / 2 / mse, 2, df_res)
        p_room = stats.f.sf(ss_room / 5 / mse, 5, df_res)
        mse0 = (ss_tot - ss_diet) / (3 * nb - 3)
        p0 = stats.f.sf(ss_diet / 2 / mse0, 2, 3 * nb - 3)
        if v["kind"] == "strong" and p_diet < 0.005 and p0 > 0.08 and p_room < 0.001:
            break
        if v["kind"] == "weak" and p_diet < 0.01 and p0 < 0.01 and p_room > 0.3:
            break
    else:
        raise RuntimeError("rcbd")
    room = np.repeat(np.arange(1, nb + 1), 3)
    diet = np.tile(lv, nb)
    adg = Y.ravel()
    order = np.argsort(rng.random(len(adg)) + room * 10)  # keep rooms together, shuffle within
    pen = np.arange(1, 19)
    room, diet, adg = room[order], diet[order], adg[order]
    S = []
    S.append(step("Загрузить данные", task="Таблица <code>pens</code>: 18 загонов, средний привес загона (г/сут), комната (1–6) и рацион.",
                  code=['> pens <- read.csv("rooms.csv")', "> str(pens)"],
                  out=str_df(18, [("pen", "int", pen), ("room", "int", room), ("diet", "chr", diet), ("adg", "num", adg)]),
                  qs=[mc(rng, "<code>room</code> хранится как int (числа 1–6). Чем это опасно?",
                         "В модели R сочтёт номер комнаты числом и подгонит один наклон по номеру вместо 5 df на различия комнат",
                         ["Ничем: номер есть номер", "R выдаст ошибку и не посчитает модель", "Комнаты с большим номером получат больший вес"],
                         "Номер комнаты — метка, а не количество: комната 4 не «вдвое больше» комнаты 2. Поэтому пишут <code>factor(room)</code> или переводят столбец в фактор заранее.")]))
    S.append(step("Проверить план", task="Убедись, что в каждой комнате есть каждый рацион.",
                  code="> with(pens, table(room, diet))",
                  out=table2("room", "diet", [str(i) for i in range(1, nb + 1)], lv, np.ones((nb, 3))),
                  qs=[mc(rng, "Что это за план?", "Рандомизированные блоки: комната — блок, в каждой комнате все три рациона по одному загону",
                         ["Полностью рандомизированный план: комнаты не важны", "Факторный план 6 × 3 с повторностями", "Split-plot: рацион на комнату"],
                         "Комнаты отличаются (температура, вентиляция, возраст поросят при заселении). Блок «уравнивает» эти различия: каждый рацион попадает в каждую комнату. "
                         "Внутри комнаты рацион загонам назначали жребием."),
                      num("Сколько df останется на ошибку в модели с блоками?", df_res,
                          f"(блоков − 1) × (рационов − 1) = 5 × 2 = <b>{df_res}</b>. Это всё, что осталось от 18 − 1 = 17 после 5 df на комнаты и 2 на рационы.", rel=0)]))
    al, ad = aov_summary([("factor(room)", 5, ss_room), ("diet", 2, ss_diet)], (df_res, ss_res))
    S.append(step("ANOVA с блоками",
                  pick=code_pick(rng, "Какой код соответствует плану?", "fit <- aov(adg ~ factor(room) + diet, data = pens)",
                                 ["fit <- aov(adg ~ room + diet, data = pens)", "fit <- aov(adg ~ factor(room) * diet, data = pens)", "fit <- aov(adg ~ diet + pen, data = pens)"],
                                 "Блок входит в модель как фактор, без взаимодействия с рационом: на каждую пару «комната × рацион» один загон, "
                                 "и взаимодействие съело бы все df ошибки. Без factor() комната станет числом."),
                  code=["> fit <- aov(adg ~ factor(room) + diet, data = pens)", "> summary(fit)"], out=al,
                  qs=[mc(rng, f"Рацион: F = {R.fmt1(ad['F'][1], 4)}, p = {pr(ad['p'][1])}. Что это значит?",
                         "Хотя бы один рацион отличается по среднему привесу" if ad["p"][1] < 0.05 else "Разница рационов не доказана",
                         ["Рационы различаются в каждой комнате", "Комнаты различаются из-за рационов", "Все три рациона разные"],
                         "Строка diet проверяет рационы <b>после</b> учёта комнат. Строка factor(room) — различия комнат; сами по себе они редко интересны, "
                         "но показывают, помог ли блок."),
                      num("Чему равен MSE (Mean Sq Residuals)?", ad["mse"], f"MSE = <b>{R.r7(ad['mse'])}</b>; σ̂ = √MSE ≈ {rua(math.sqrt(ad['mse']), 3)} г/сут.", rel=0.01)]))
    al0, ad0 = aov_summary([("diet", 2, ss_diet)], (15, ss_tot - ss_diet))
    ratio = ad0["mse"] / ad["mse"]
    S.append(step("Что было бы без блоков", task="Посмотри, как изменится анализ, если забыть про комнаты.",
                  code=["> summary(aov(adg ~ diet, data = pens))"], out=al0,
                  qs=[num("Во сколько раз остаточная дисперсия без блоков больше, чем с блоками?", ratio,
                          f"{R.r7(ad0['mse'])} / {R.r7(ad['mse'])} = <b>{rua(ratio, 3)}</b>. " +
                          ("Различия комнат ушли в ошибку — F для рациона упал, и разница стала не видна." if v["kind"] == "strong"
                           else "Комнаты почти одинаковые: блоки почти ничего не дали, но и не помешали."), rel=0.02),
                      mc(rng, "Какой анализ правильный?", "С блоками: так был устроен опыт",
                         ["Без блоков: у него больше df", "Тот, где p меньше", "Оба неверны: нужен t-тест"],
                         "Анализ повторяет дизайн. Блоки выбирают до опыта, а не по результату. Даже если комнаты почти одинаковые, блоки оставляют в модели.")]))
    X = np.c_[np.ones(18)]
    Xr = np.column_stack([np.ones(18)] + [(room == r).astype(float) for r in range(2, nb + 1)] + [(diet == l).astype(float) for l in lv[1:]])
    P, F = p_lm(Xr, adg, (1, 2))
    S.append(step("Остатки", code=["> par(mfrow = c(1, 2))", "> plot(fit, which = 1:2)"], plots=P,
                  qs=[mc(rng, "Есть ли явные проблемы?", "Нет: облако без формы, Q-Q около прямой (при 18 загонах неровности нормальны)",
                         ["Да: подписаны 3 точки, это выбросы", "Да: облако не идеально ровное — нужен непараметрический тест", "Да: точек мало, ANOVA нельзя"],
                         "При малом числе единиц графики всегда «шумные». Ищут явный узор: трубу, дугу, далёкую точку.")]))
    means = [Y[:, j].mean() for j in range(3)]
    tr = tukey_rows(lv, means, nb, ss_res / df_res, df_res)
    tl = tukey_print("aov(formula = adg ~ factor(room) + diet, data = pens)", "diet", tr)
    pa = [r["padj"] for r in tr]
    S.append(step("Какие рационы различаются", code='> TukeyHSD(fit, which = "diet")', out=tl,
                  qs=[num("Сколько пар значимо различаются?", sum(p < 0.05 for p in pa),
                          "По p adj: " + ", ".join(f"{r['name']} {'✓' if r['padj'] < 0.05 else '✗'}" for r in tr) + ".", rel=0),
                      mc(rng, "Чем Tukey здесь отличается от Tukey без блоков?", "Использует MSE модели с блоками: интервалы уже",
                         ["Ничем: Tukey не видит модель", "Сравнивает комнаты, а не рационы", "Делит p на число комнат"],
                         "TukeyHSD берёт MSE и df из подогнанной модели. Чем меньше MSE, тем уже интервалы для разностей.")]))
    rep = (f"Рацион влиял на привес загона (ANOVA с комнатой как блоком: F = {R.fmt1(ad['F'][1], 3).replace('.', ',')}, df = 2 и {df_res}, p = {pr(ad['p'][1])})")
    S.append(step("Записать результат", qs=[mc(rng, "Какая фраза верна?", rep,
                                                [f"Рацион влиял на привес каждой свиньи (n = 18, p = {pr(ad['p'][1])})",
                                                 "Комнаты различаются, значит рационы работают по-разному в разных комнатах",
                                                 "Блоки не нужны были: их p значим, поэтому их исключили"],
                                                "Единица — загон (18), а не свинья. Взаимодействие рацион × комната здесь не проверялось — для этого нужны повторности внутри комнаты.")]))
    items = [("Единица эксперимента — загон", True), ("Комната в модели как фактор", True), ("Взаимодействие комната × рацион оценено", False),
             ("Рацион назначали загонам жребием внутри комнаты", True), ("Блоки добавили после того, как увидели данные", False)]
    S.append(step("Проверка", qs=[multi(rng, "Отметь верное.", items, "Блоки определяют до опыта. Взаимодействие блока с рационом в RCBD без повторностей не оценить.")]))
    return dict(g="means", key="rcbd", title="Блоки: 6 комнат × 3 рациона", vt=("комнаты сильно разные" if v["kind"] == "strong" else "комнаты похожи"),
                goal="<p><b>Вопрос.</b> Различаются ли три рациона (контроль, MCFA, пробиотик) по привесу? В каждой из 6 комнат три загона, рацион загонам назначали жребием внутри комнаты. Отклик — средний привес загона.</p>",
                steps=S)


# =====================================================================
# 6. Factorial 2x2 with interaction
# =====================================================================
FACT_VAR = [dict(e=[0, 70, 0, 5], kind="int"), dict(e=[0, 35, -40, -5], kind="add")]


def wf_fact(rng, v):
    n = 10
    cells = [("low", "barrow"), ("high", "barrow"), ("low", "gilt"), ("high", "gilt")]
    for _ in range(5000):
        Y = {c: np.round(rng.normal(820 + e, 40, n)) for c, e in zip(cells, v["e"])}
        en = np.repeat([c[0] for c in cells], n)
        sx = np.repeat([c[1] for c in cells], n)
        y = np.concatenate([Y[c] for c in cells])
        H_ = (en == "high").astype(float)
        G_ = (sx == "gilt").astype(float)
        X = np.c_[np.ones(4 * n), H_, G_, H_ * G_]
        f = ols(X, y)
        X0 = np.c_[np.ones(4 * n), H_, G_]
        f0 = ols(X0, y)
        gm = y.mean()
        ss_e = 2 * n * sum((y[en == l].mean() - gm) ** 2 for l in ("low", "high"))
        ss_s = 2 * n * sum((y[sx == l].mean() - gm) ** 2 for l in ("barrow", "gilt"))
        ss_i = f0["sse"] - f["sse"]
        mse = f["mse"]
        p_i = stats.f.sf(ss_i / mse, 1, f["df"])
        p_e = stats.f.sf(ss_e / mse, 1, f["df"])
        gse = math.sqrt(2 * mse / n)
        g_eff = Y[("high", "gilt")].mean() - Y[("low", "gilt")].mean()
        p_g = 2 * stats.t.sf(abs(g_eff) / gse, f["df"])
        if v["kind"] == "int" and p_i < 0.02 and f["p"][1] < 0.001 and p_g > 0.25:
            break
        if v["kind"] == "add" and p_i > 0.4 and p_e < 0.01:
            break
    else:
        raise RuntimeError("fact")
    en, sx, y = shuffle_rows(rng, en, sx, y)
    N = 4 * n
    S = []
    S.append(step("Загрузить и задать порядок уровней", task="Таблица <code>pigs</code>: энергия рациона (low/high), пол (barrow — боров, gilt — свинка) и привес. "
                  "По алфавиту high шёл бы первым, а нам нужен low как контроль.",
                  pick=code_pick(rng, "Как сделать low базовым уровнем?", 'pigs$energy <- factor(pigs$energy, levels = c("low", "high"))',
                                 ['pigs$energy <- as.numeric(pigs$energy)', 'pigs$energy <- sort(pigs$energy)', 'levels(pigs$energy) <- c("low", "high")'],
                                 "<code>factor(x, levels = …)</code> задаёт порядок уровней, первый — базовый. <code>sort</code> переставит значения и перепутает строки; "
                                 "присваивание <code>levels&lt;-</code> переименовало бы уровни, а не переставило."),
                  code=['> pigs <- read.csv("energy_sex.csv")', '> pigs$energy <- factor(pigs$energy, levels = c("low", "high"))', "> str(pigs)"],
                  out=str_df(N, [("pig", "int", range(1, N + 1)), ("energy", "factor", [("low", "high").index(e) and "high" or "low" for e in en]),
                                 ("sex", "chr", sx), ("adg", "num", y)]),
                  qs=[]))
    # fix factor str manually (levels order low, high)
    codes = " ".join(str(1 if e == "low" else 2) for e in en[:10])
    S[0]["out"] = S[0]["out"].replace(S[0]["out"].split("\n")[2], f' $ energy: Factor w/ 2 levels "low","high": {codes} ...')
    cm = np.array([[y[(en == e) & (sx == s)].mean() for s in ("barrow", "gilt")] for e in ("low", "high")])
    cells_txt = joint_cells(cm, 7)
    S.append(step("Средние по клеткам", code="> with(pigs, tapply(adg, list(energy, sex), mean))",
                  out=[" " * 4 + " " + " ".join(c.rjust(max(len(c), max(len(cells_txt[i][j]) for i in range(2)))) for j, c in enumerate(["barrow", "gilt"]))] +
                      [e.ljust(4) + " " + " ".join(cells_txt[i][j].rjust(max(len(c), max(len(cells_txt[k][j]) for k in range(2)))) for j, c in enumerate(["barrow", "gilt"]))
                       for i, e in enumerate(["low", "high"])],
                  qs=[num("Эффект высокой энергии у боровов (high − low), г/сут?", cm[1, 0] - cm[0, 0],
                          f"{R.r7(cm[1, 0])} − {R.r7(cm[0, 0])} = <b>{rua(cm[1, 0] - cm[0, 0], 4)}</b>.", rel=0.01, abs_=0.1),
                      num("А у свинок?", cm[1, 1] - cm[0, 1], f"{R.r7(cm[1, 1])} − {R.r7(cm[0, 1])} = <b>{rua(cm[1, 1] - cm[0, 1], 4)}</b>.", rel=0.01, abs_=0.5)]))
    diffdiff = (cm[1, 0] - cm[0, 0]) - (cm[1, 1] - cm[0, 1])
    S.append(step("График взаимодействия", code="> with(pigs, interaction.plot(energy, sex, adg))",
                  plots=[p_int(["low", "high"], {"barrow": list(cm[:, 0]), "gilt": list(cm[:, 1])}, xl="energy", yl="mean of adg")],
                  qs=[mc(rng, "Что говорит график?",
                         ("Линии не параллельны: энергия помогает боровам заметно сильнее, чем свинкам" if v["kind"] == "int"
                          else "Линии почти параллельны: эффект энергии похож у обоих полов"),
                         (["Линии параллельны: взаимодействия нет", "Свинки растут быстрее боровов на любом рационе", "График показывает разброс, а не средние"]
                          if v["kind"] == "int" else
                          ["Линии пересекаются: сильное взаимодействие", "Энергия не влияет ни на кого", "Свинки реагируют на энергию вдвое сильнее"]),
                         f"Взаимодействие — это «разница разниц»: ({rua(cm[1, 0] - cm[0, 0], 3)}) − ({rua(cm[1, 1] - cm[0, 1], 3)}) = {rua(diffdiff, 3)} г/сут. "
                         "Параллельные линии = разница одинакова = взаимодействия нет. Значимость покажет тест.")]))
    al, ad = anova_lm([("energy", 1, ss_e), ("sex", 1, ss_s), ("energy:sex", 1, ss_i)], (f["df"], f["sse"]), "adg")
    S.append(step("Модель с взаимодействием",
                  pick=code_pick(rng, "Какая формула включает оба фактора и взаимодействие?", "m <- lm(adg ~ energy * sex, data = pigs)",
                                 ["m <- lm(adg ~ energy + sex, data = pigs)", "m <- lm(adg ~ energy:sex, data = pigs)", "m <- lm(adg ~ energy * sex * pig, data = pigs)"],
                                 "<code>a * b</code> = <code>a + b + a:b</code>. Плюс без звёздочки — только главные эффекты, а <code>a:b</code> в одиночку — только взаимодействие, "
                                 "что почти никогда не нужно."),
                  code=["> m <- lm(adg ~ energy * sex, data = pigs)", "> anova(m)"], out=al,
                  qs=[mc(rng, f"energy:sex: p = {pr(ad['p'][2])}. Что дальше?",
                         ("Взаимодействие есть: эффект энергии смотрим отдельно для каждого пола" if v["kind"] == "int"
                          else "Взаимодействие не найдено: можно смотреть главные эффекты (или упростить модель)"),
                         (["Читать строку energy: она даёт средний эффект для всех", "Удалить взаимодействие, раз главный эффект не значим", "Ничего: модель неверна"]
                          if v["kind"] == "int" else
                          ["Эффект энергии считать отдельно в каждом поле обязательно", "Главные эффекты теперь читать нельзя", "Выкинуть пол из модели"]),
                         "Если взаимодействие значимо, «средний» эффект энергии смешивает два разных эффекта и вводит в заблуждение. "
                         "Если нет — эффект энергии одинаков у обоих полов, и главный эффект имеет смысл."),
                      num("df ошибки?", f["df"], f"N − число средних клеток = {N} − 4 = <b>{f['df']}</b>.", rel=0)]))
    se = math.sqrt(f["mse"] * 2 / n)
    if v["kind"] == "int":
        est = [cm[1, 0] - cm[0, 0], cm[1, 1] - cm[0, 1]]
        tv = [e / se for e in est]
        pv = [2 * stats.t.sf(abs(t), f["df"]) for t in tv]
        el = ["> library(emmeans)", "> pairs(emmeans(m, ~ energy | sex), reverse = TRUE)"]
        for s_, e_, t_, p_ in zip(["barrow", "gilt"], est, tv, pv):
            el += [f"sex = {s_}:"] + emm_table([("contrast", ["high - low"], True), ("estimate", emm_fmt([e_], 3), False), ("SE", emm_fmt([se]), False),
                                                 ("df", [str(f['df'])], False), ("t.ratio", [f"{t_:.3f}"], False),
                                                 ("p.value", ["<.0001" if p_ < 1e-4 else f"{p_:.4f}"], False)], True) + [""]
        S.append(step("Простые эффекты", task="Сравни энергии отдельно у боровов и у свинок.",
                      code=el[:2], out=el[2:],
                      qs=[mc(rng, "Какой вывод?", f"У боровов высокая энергия добавляет около {rua(est[0], 2)} г/сут (p = {pr(pv[0])}), у свинок эффект не доказан (p = {pr(pv[1])})",
                             ["Энергия не работает: главный эффект energy не значим", "Энергия работает одинаково у обоих полов",
                              "У свинок эффект отрицательный и значимый"],
                             "Главный эффект energy в модели с взаимодействием — это эффект у базового пола или среднее по полам, в зависимости от кодирования. "
                             "При взаимодействии честнее показать эффекты по группам.")]))
    else:
        f0p = ols(X0[np.argsort(np.argsort(np.zeros(N)))], y) if False else None
        H_ = (en == "high").astype(float)
        G_ = (sx == "gilt").astype(float)
        ll, disp, fa = lm_out("adg ~ energy + sex", "pigs", np.c_[np.ones(N), H_, G_], y, ["(Intercept)", "energyhigh", "sexgilt"])
        S.append(step("Упростить модель", task="Взаимодействие не нужно — подгони модель только с главными эффектами.",
                      code=["> m2 <- lm(adg ~ energy + sex, data = pigs)", "> summary(m2)"], out=ll,
                      qs=[mc(rng, "Что значит коэффициент energyhigh?", f"Высокая энергия добавляет около {rua(fa['b'][1], 3)} г/сут при одном и том же поле",
                             ["Привес свиней на высокой энергии", "Разница между полами", "Эффект энергии только у боровов"],
                             "В аддитивной модели эффект энергии один для обоих полов — так мы и решили, раз взаимодействия нет.")]))
    P, _ = p_lm(np.c_[np.ones(N), (en == "high"), (sx == "gilt"), (en == "high") * (sx == "gilt")].astype(float), y, (1, 2))
    S.append(step("Остатки", code=["> par(mfrow = c(1, 2))", "> plot(m, which = 1:2)"], plots=P,
                  qs=[mc(rng, "Почему точки стоят 4 столбиками?", "Прогноз — среднее своей клетки, а клеток 4 (2 × 2)",
                         ["Четыре выброса", "Нарушена линейность", "Четыре свиньи на загон"], "Как в ANOVA: прогноз модели с факторами — средние групп.")]))
    rep = ("Эффект энергии зависел от пола (взаимодействие p = " + pr(ad["p"][2]) + "): у боровов +" + rua(cm[1, 0] - cm[0, 0], 2) + " г/сут, у свинок разница не доказана"
           if v["kind"] == "int" else
           "Взаимодействия энергии и пола нет (p = " + pr(ad["p"][2]) + "); высокая энергия повышала привес в среднем на " + rua((cm[1, 0] - cm[0, 0] + cm[1, 1] - cm[0, 1]) / 2, 2) + " г/сут у обоих полов")
    S.append(step("Записать результат", qs=[mc(rng, "Какая фраза верна?", rep,
                                                ["Высокая энергия и пол взаимодействуют, значит пол вызывает разницу в рационах",
                                                 "Главный эффект energy незначим, значит энергия не нужна", "Свинки и боровы — разные опыты, их нельзя сравнивать"],
                                                "Сначала — взаимодействие, потом главные эффекты.")]))
    items = [("Порядок уровней задан вручную (low базовый)", True), ("Взаимодействие проверено до чтения главных эффектов", True),
             ("Пол назначали жребием", False), ("Энергию назначали жребием, поэтому её эффект причинный", True)]
    S.append(step("Проверка", qs=[multi(rng, "Отметь верное.", items, "Пол не назначают — это свойство животного. Причинный вывод — только про энергию.")]))
    return dict(g="means", key="fact", title="Факторный 2 × 2: энергия × пол", vt=("есть взаимодействие" if v["kind"] == "int" else "без взаимодействия"),
                goal=f"<p><b>Вопрос.</b> Одинаково ли высокоэнергетический рацион действует на боровов и свинок? {N} свиней в индивидуальных станках, рацион назначали жребием внутри пола.</p>",
                steps=S)


# =====================================================================
# 7. 2x2 table: vaccine trial
# =====================================================================
CHI_VAR = [dict(n=100, p=(0.30, 0.15), kind="large"), dict(n=18, p=(0.33, 0.04), kind="small")]


def wf_chisq(rng, v):
    for _ in range(5000):
        yp = int(rng.binomial(v["n"], v["p"][0]))
        yv = int(rng.binomial(v["n"], v["p"][1]))
        tab = np.array([[v["n"] - yp, yp], [v["n"] - yv, yv]])
        ch = H.chisq(tab)
        fi = H.fisher(tab)
        if v["kind"] == "large" and ch["p"] < 0.02 and not ch["small"]:
            break
        if v["kind"] == "small" and ch["small"] and fi["p"] < 0.05 and yv >= 1:
            break
    else:
        raise RuntimeError("chisq")
    grp = np.array(["placebo"] * v["n"] + ["vacc"] * v["n"])
    dis = np.array(["no"] * tab[0, 0] + ["yes"] * tab[0, 1] + ["no"] * tab[1, 0] + ["yes"] * tab[1, 1])
    grp, dis = shuffle_rows(rng, grp, dis)
    N = 2 * v["n"]
    S = []
    S.append(step("Загрузить", task="Таблица <code>vt</code>: одна строка — один поросёнок; группа (placebo или vacc) и была ли клиническая пневмония до конца доращивания.",
                  code=['> vt <- read.csv("vaccine.csv")', "> str(vt)"],
                  out=str_df(N, [("pig", "int", range(1, N + 1)), ("group", "chr", grp), ("disease", "chr", dis)]),
                  qs=[mc(rng, "Какой тип данных у отклика?", "Бинарный: да/нет у каждого поросёнка",
                         ["Непрерывный", "Счётный", "Порядковый"], "Два исхода → таблица 2 × 2, χ², Фишер или логистическая регрессия.")]))
    S.append(step("Таблица сопряжённости",
                  pick=code_pick(rng, "Как получить таблицу 2 × 2?", "tab <- with(vt, table(group, disease))",
                                 ["tab <- table(vt)", "tab <- with(vt, tapply(disease, group, mean))", "tab <- data.frame(vt$group, vt$disease)"],
                                 "<code>table(a, b)</code> считает частоты сочетаний. <code>table(vt)</code> взяла бы все три столбца, включая номер поросёнка."),
                  code=["> tab <- with(vt, table(group, disease))", "> tab"], out=table2("group", "disease", ["placebo", "vacc"], ["no", "yes"], tab),
                  qs=[num("Доля заболевших в плацебо, %?", 100 * tab[0, 1] / v["n"], f"{tab[0, 1]} / {v['n']} = <b>{ru(100 * tab[0, 1] / v['n'], 1)}</b>%.", rel=0.01)]))
    rp = tab / tab.sum(1, keepdims=True)
    S.append(step("Доли по строкам", code="> prop.table(tab, 1)", out=table2("group", "disease", ["placebo", "vacc"], ["no", "yes"], rp, cells=joint_cells(rp, 7)),
                  qs=[mc(rng, "Почему <code>prop.table(tab, 1)</code>, а не <code>prop.table(tab)</code>?", "1 — доли внутри каждой строки (группы): риск заболеть в группе",
                         ["1 — первый столбец", "Без 1 R выдаст ошибку", "1 — округление до 1 знака"],
                         "margin = 1 — по строкам, 2 — по столбцам, без него — доли от общего числа. Риск в группе — доля по строке.")]))
    E = ch["E"]
    el = ["> chisq.test(tab)$expected"] + table2("group", "disease", ["placebo", "vacc"], ["no", "yes"], E, cells=joint_cells(E, 7))
    cl = ["> chisq.test(tab)"] + H.htest(ch["method"], "tab", [("X-squared", ch["X2"]), ("df", ch["df"])], ch["p"], "")
    cl = [l for l in cl if not l.startswith("alternative")]
    if ch["small"]:
        cl += ["Warning message:", "In chisq.test(tab) : Chi-squared approximation may be incorrect"]
    S.append(step("Хи-квадрат и ожидаемые частоты", code=el[0], out=el[1:] + [cl[0]] + cl[1:],
                  qs=[num("Самая маленькая ожидаемая частота?", E.min(), f"Минимум в таблице expected = <b>{R.r7(E.min())}</b>. Правило: все ожидаемые ≥ 5 — χ² надёжен.", rel=0.01),
                      mc(rng, "Можно ли верить p из χ²?",
                         ("Нет: ожидаемые частоты меньше 5, R сам предупреждает — нужен точный тест Фишера" if ch["small"] else "Да: все ожидаемые частоты больше 5"),
                         (["Да: p меньше 0,05", "Да: предупреждение можно игнорировать", "Нет: χ² не работает с 2 × 2"] if ch["small"]
                          else ["Нет: нужен Фишер всегда", "Нет: Yates' correction делает тест неверным", "Нет: наблюдаемые частоты разные"]),
                         "χ² — приближение, которое хорошо работает при достаточных ожидаемых частотах. Иначе — fisher.test.")]))
    fl = ["> fisher.test(tab)"] + H.htest("Fisher's Exact Test for Count Data", "tab", [], fi["p"], "true odds ratio is not equal to 1", fi["ci"], 95, [("odds ratio", fi["est"])])
    S.append(step("Точный тест Фишера", code=fl[0], out=fl[1:],
                  qs=[mc(rng, "Что значит odds ratio в выводе?", f"Шансы заболеть у vacc в {rua(1 / fi['est'], 3) if fi['est'] > 0 else '∞'} раза ниже, чем у placebo",
                         [f"Риск у vacc ниже на {ru(100 * (1 - fi['est']), 0)}%", "Вакцина защищает в " + rua(fi['est'], 2) + " случаях из 100",
                          "Это p-value в другой форме"],
                         f"OR = шансы во 2-й строке / шансы в 1-й: {rua(fi['est'], 3)}. Fisher печатает условную оценку (не ровно ad/bc). ДИ OR: {rua(fi['ci'][0], 3)}–{rua(fi['ci'][1], 3)}.")]))
    p1, p2 = tab[0, 1] / v["n"], tab[1, 1] / v["n"]
    RR = p2 / p1
    S.append(step("Риск, RR и эффективность вакцины", task="Для опыта (когорты) понятнее относительный риск и эффективность вакцины VE = 1 − RR.",
                  code=['> p <- prop.table(tab, 1)[, "yes"]', '> p["vacc"] / p["placebo"]'], out=named(["vacc"], [RR]),
                  qs=[num("Эффективность вакцины VE = (1 − RR) × 100, %?", 100 * (1 - RR),
                          f"RR = {ru(p2, 3)} / {ru(p1, 3)} = {rua(RR, 3)}; VE = (1 − {rua(RR, 3)}) × 100 = <b>{rua(100 * (1 - RR), 3)}</b>%: вакцина сократила риск пневмонии на столько процентов.", rel=0.02),
                      mc(rng, "Почему здесь можно RR, а в исследовании «случай-контроль» нельзя?", "Здесь группы набраны по воздействию, и доля заболевших в группе — настоящий риск",
                         ["RR всегда можно, разницы нет", "В случае-контроль нет таблицы 2 × 2", "Потому что здесь больше 30 животных"],
                         "В случай-контроль число больных задаёт исследователь, поэтому «риск» в группах искусственный. Остаётся только OR.")]))
    test_name = "точный тест Фишера" if ch["small"] else "χ² с поправкой Йейтса"
    pp = fi["p"] if ch["small"] else ch["p"]
    rep = f"Пневмония: {tab[0, 1]}/{v['n']} в плацебо и {tab[1, 1]}/{v['n']} у вакцинированных; RR = {rua(RR, 2)}, VE = {ru(100 * (1 - RR), 0)}% ({test_name}, p = {pr(pp)})"
    S.append(step("Записать результат", qs=[mc(rng, "Какая фраза верна?", rep,
                                                [f"Вакцина защищает {ru(100 * (1 - RR), 0)}% поросят навсегда", f"OR = {rua(fi['est'], 2)}, значит риск снижен на {ru(100 * (1 - fi['est']), 0)}%",
                                                 "χ² показал, что вакцина и болезнь независимы"],
                                                "OR и RR близки только при редком исходе. VE — снижение риска в этом опыте за этот срок.")]))
    items = [("Группу назначали жребием каждому поросёнку", True),
             ("Поросята в общих загонах могли заражать друг друга: независимость под вопросом", True),
             ("Проверили ожидаемые частоты перед χ²", True), ("Если p &lt; 0,05, вакцина работает на всех фермах", False)]
    S.append(step("Проверка", qs=[multi(rng, "Отметь верное.", items,
                                        "Инфекция передаётся между соседями: если вакцинированные и невакцинированные живут вместе, заражение зависит от соседей. "
                                        "Строже — рандомизировать загоны и анализировать с учётом загона (glmer).")]))
    return dict(g="tables", key="chisq", title="Вакцина: таблица 2 × 2", vt=("большая выборка" if v["kind"] == "large" else "мало животных"),
                goal=f"<p><b>Вопрос.</b> Снижает ли вакцина риск пневмонии? {N} поросят, вакцину или плацебо назначали жребием.</p>", steps=S)


# =====================================================================
# 8. Simple linear regression (with a data-entry error in variant 2)
# =====================================================================
SLR_VAR = [dict(kind="clean"), dict(kind="typo")]


def wf_slr(rng, v):
    n = 40
    for _ in range(5000):
        bw = np.round(rng.uniform(0.85, 1.95, n), 2)
        ww = np.round(1.8 + 3.1 * bw + rng.normal(0, 0.55, n), 2)
        X = np.c_[np.ones(n), bw]
        f = ols(X, ww)
        ch = DB.checks(DB.ols(X, ww), X, x_for_quad=bw)
        if ch["shapiro"] > 0.3 and ch["quad_p"] > 0.3 and ch["maxrs"] < 2.6 and abs(ch["het"]) < 0.15:
            break
    k = int(rng.integers(5, n - 5))
    bw_obs = bw.copy()
    if v["kind"] == "typo":
        bw_obs[k] = round(bw[k] * 10, 1)
    S = []
    S.append(step("Загрузить", task="Таблица <code>pig</code>: по одному поросёнку из 40 гнёзд, масса при рождении (<code>birth</code>, кг) и при отъёме в 21 день (<code>wean</code>, кг).",
                  code=['> pig <- read.csv("piglets.csv")', "> summary(pig$birth)"], out=summary_vec(bw_obs),
                  qs=[mc(rng, "Есть ли в summary что-то подозрительное?",
                         (f"Да: максимум {ru(bw_obs.max(), 1)} кг — поросёнок не может весить столько при рождении" if v["kind"] == "typo"
                          else "Нет: от " + ru(bw.min(), 2) + " до " + ru(bw.max(), 2) + " кг — обычные массы при рождении"),
                         (["Нет: среднее около 1,5 кг, всё нормально", "Да: медиана меньше среднего", "Да: минимум меньше 1 кг"] if v["kind"] == "typo"
                          else ["Да: среднее не равно медиане", "Да: минимум меньше 1 кг — это ошибка", "Да: нет отрицательных значений"]),
                         "Первая проверка любого датасета — диапазоны. Масса при рождении 0,8–2 кг; " +
                         ("12 кг — явная ошибка ввода (запятая не там)." if v["kind"] == "typo" else "всё в разумных пределах."))]))
    if v["kind"] == "typo":
        S.append(step("Найти строку с ошибкой",
                      pick=code_pick(rng, "Как вывести строки, где birth больше 5 кг?", "pig[pig$birth > 5, ]",
                                     ["pig[, pig$birth > 5]", "pig$birth > 5", "subset(pig, birth = 5)"],
                                     "<code>[строки, столбцы]</code>: условие ставят на место строк, после запятой пусто — все столбцы. "
                                     "<code>pig$birth &gt; 5</code> вернёт только TRUE/FALSE."),
                      code="> pig[pig$birth > 5, ]",
                      out=df_print([("piglet", "int", [k + 1]), ("birth", "num", [bw_obs[k]]), ("wean", "num", [ww[k]])], rownames=[str(k + 1)]),
                      qs=[mc(rng, "Что делать с этой строкой?", "Проверить по журналу; если там " + ru(bw[k], 2) + " — исправить и записать, что исправлено",
                             ["Удалить молча", "Оставить: это реальные данные", "Заменить на среднее по всем"],
                             "Ошибки ввода исправляют по первичным записям и фиксируют в скрипте (а не в Excel руками), чтобы правка была видна. "
                             "Если первичной записи нет — исключают и пишут об этом.")]))
        Xo = np.c_[np.ones(n), bw_obs]
        fo = DB.ols(Xo, ww)
        Po = DB.plots_lm(fo, (4,))
        ll0, d0, f0 = lm_out("wean ~ birth", "pig", Xo, ww, ["(Intercept)", "birth"])
        S.append(step("Что ошибка делает с моделью", task="Для наглядности — модель по данным с ошибкой.",
                      code=["> m_bad <- lm(wean ~ birth, data = pig)", "> coef(m_bad)"],
                      out=named(["(Intercept)", "birth"], f0["b"]) + ["> plot(m_bad, which = 4)"], plots=[dict(Po[0])],
                      qs=[num(f"Cook's D у строки {k + 1}?", float(fo["cook"][k]), f"Столбик с номером {k + 1} — Cook's D = <b>{rua(float(fo['cook'][k]), 3)}</b>. Больше 1 — точка одна тянет линию.", rel=0.05),
                          mc(rng, "Что случилось с наклоном?", f"Одна точка положила линию почти плашмя: наклон {rua(f0['b'][1], 2)} вместо ~3 кг на кг",
                             ["Наклон не изменился: одна точка из 40 не влияет", "Наклон стал круче", "Модель не посчиталась"],
                             "Точка далеко по X (большой рычаг) и не на линии тянет наклон на себя. Поэтому смотрят на данные до модели.")]))
        S.append(step("Исправить", code=[f"> pig$birth[pig$piglet == {k + 1}] <- {bw[k]}   # журнал опоросов, гнездо {k + 1}", "> summary(pig$birth)"],
                      out=summary_vec(bw), qs=[]))
    S.append(step("Диаграмма рассеяния", code=["> plot(wean ~ birth, data = pig)", "> abline(lm(wean ~ birth, data = pig))"],
                  plots=[p_xy(bw, ww, line=(f["b"][0], f["b"][1]), xl="birth", yl="wean")],
                  qs=[mc(rng, "Форма связи?", "Прямая с положительным наклоном и примерно ровным разбросом",
                         ["Дуга: нужен квадратичный член", "Связи нет", "Разброс растёт с массой — труба"], "Рассеяние — первый взгляд на форму связи до модели.")]))
    ll, disp, _ = lm_out("wean ~ birth", "pig", X, ww, ["(Intercept)", "birth"])
    S.append(step("Модель", pick=code_pick(rng, "Какой код подгонит прямую?", "m <- lm(wean ~ birth, data = pig)",
                                           ["m <- lm(birth ~ wean, data = pig)", "m <- cor(pig$wean, pig$birth)", "m <- glm(wean ~ birth, family = binomial, data = pig)"],
                                           "Отклик (что объясняем) слева. birth ~ wean объясняло бы массу при рождении массой при отъёме — наоборот."),
                  code=["> m <- lm(wean ~ birth, data = pig)", "> summary(m)"], out=ll,
                  qs=[mc(rng, f"Как прочитать наклон {R.r7(disp['est'][1])}?", f"Поросёнок, родившийся на 1 кг тяжелее, к отъёму в среднем тяжелее на {rua(disp['est'][1], 3)} кг",
                         [f"Масса при отъёме в {rua(disp['est'][1], 3)} раза больше массы при рождении", f"Каждый поросёнок прибавляет {rua(disp['est'][1], 3)} кг",
                          "Наклон — это коэффициент корреляции"],
                         "Наклон — изменение среднего Y на единицу X. На 100 г при рождении — это " + rua(disp['est'][1] / 10, 3) + " кг при отъёме."),
                      num("R² в процентах?", 100 * disp["r2"], f"Multiple R-squared = {disp['r2']}: масса при рождении объясняет <b>{ru(100 * disp['r2'], 1)}</b>% разброса массы при отъёме.", rel=0.01)]))
    P, F = p_lm(X, ww, (1, 2))
    S.append(step("Остатки", code=["> par(mfrow = c(1, 2))", "> plot(m, which = 1:2)"], plots=P,
                  qs=[mc(rng, "Что с предпосылками?", "Полоса без формы, Q-Q около прямой: в порядке",
                         ["Дуга: нелинейность", "Труба: разные дисперсии", "Подписанные точки надо удалить"], "Ищем узор, а не идеальную картинку.")]))
    tq = stats.t.ppf(0.975, n - 2)
    lo = f["b"] - tq * f["se"]
    hi = f["b"] + tq * f["se"]
    S.append(step("Доверительные интервалы коэффициентов", code="> confint(m)",
                  out=R.matrix(["(Intercept)", "birth"], ["2.5 %", "97.5 %"], [list(lo), list(hi)]),
                  qs=[mc(rng, "Что говорит интервал для birth?", f"Каждый лишний килограмм при рождении — от {rua(lo[1], 3)} до {rua(hi[1], 3)} кг при отъёме",
                         ["95% поросят в этом интервале", "Наклон точно равен середине", "Интервал включает 0"], "Это ДИ для наклона: правдоподобные значения истинного наклона.")]))
    pr_rows = []
    for x0 in (1.0, 1.6):
        p_ = predict(f, [1, x0])
        pr_rows.append(p_)
    conf = R.matrix(["1", "2"], ["fit", "lwr", "upr"], [[p["fit"] for p in pr_rows], [p["ci"][0] for p in pr_rows], [p["ci"][1] for p in pr_rows]])
    pred = R.matrix(["1", "2"], ["fit", "lwr", "upr"], [[p["fit"] for p in pr_rows], [p["pi"][0] for p in pr_rows], [p["pi"][1] for p in pr_rows]])
    S.append(step("Прогноз: среднее и отдельный поросёнок",
                  code=['> new <- data.frame(birth = c(1.0, 1.6))', '> predict(m, new, interval = "confidence")'],
                  out=conf + ['> predict(m, new, interval = "prediction")'] + pred,
                  qs=[mc(rng, "Фермер спрашивает: «Мой поросёнок родился 1,0 кг — сколько он будет весить при отъёме?» Какой интервал ему дать?",
                         f"prediction: от {rua(pr_rows[0]['pi'][0], 3)} до {rua(pr_rows[0]['pi'][1], 3)} кг",
                         [f"confidence: от {rua(pr_rows[0]['ci'][0], 3)} до {rua(pr_rows[0]['ci'][1], 3)} кг", "Только fit, интервалы не нужны",
                          "Интервал наклона из confint"],
                         "confidence — для <b>среднего</b> всех поросят с такой массой; prediction — для <b>одного</b> поросёнка, он шире, потому что добавляет разброс между животными (σ)."),
                      mc(rng, "Можно ли так же предсказать для поросёнка 0,5 кг?", "Осторожно: 0,5 кг вне диапазона данных — это экстраполяция",
                         ["Да, формула работает для любых X", "Нет, только для 1,0 и 1,6", "Да, если R² больше 0,5"],
                         f"Данные от {ru(bw.min(), 2)} до {ru(bw.max(), 2)} кг. Совсем мелкие поросята часто растут иначе — прямая там может не работать.")]))
    rep = f"Масса при отъёме росла на {rua(f['b'][1], 2)} кг на каждый кг массы при рождении (95% ДИ {rua(lo[1], 2)}–{rua(hi[1], 2)}; R² = {ru(f['r2'], 2)}; n = {n})"
    S.append(step("Записать результат", qs=[mc(rng, "Какая фраза верна?", rep,
                                                ["Масса при рождении вызывает " + ru(100 * f['r2'], 0) + "% массы при отъёме", "Связь значима, значит прогноз для каждого поросёнка точный",
                                                 "Наклон " + rua(f['b'][1], 2) + " означает рост в " + rua(f['b'][1], 2) + " раза"],
                                                "Это наблюдение: массу при рождении не назначали. R² — доля объяснённого разброса, а не «процент причины».")]))
    items = [("Один поросёнок из гнезда: строки независимы", True), ("Диапазоны проверили до модели", True),
             ("Для отдельного поросёнка — интервал prediction", True), ("Можно говорить, что масса при рождении — причина", False)]
    if v["kind"] == "typo":
        items.append(("Ошибку ввода исправили в скрипте по журналу", True))
    S.append(step("Проверка", qs=[multi(rng, "Отметь верное.", items, "Если бы взяли всех поросят гнезда, они были бы зависимы (общая мать) — понадобилась бы смешанная модель с (1 | litter).")]))
    return dict(g="regr", key="slr", title="Простая регрессия: рождение → отъём", vt=("чистые данные" if v["kind"] == "clean" else "ошибка ввода"),
                goal="<p><b>Вопрос.</b> Насколько масса при рождении предсказывает массу при отъёме? Из каждого из 40 гнёзд взвесили одного случайного поросёнка.</p>", steps=S)


# =====================================================================
# 9. Multiple regression: confounding and collinearity
# =====================================================================
MLR_VAR = [dict(kind="conf"), dict(kind="coll")]


def vif(Xn):
    out = []
    for j in range(Xn.shape[1]):
        others = np.c_[np.ones(len(Xn)), np.delete(Xn, j, axis=1)]
        f = ols(others, Xn[:, j])
        out.append(1 / (1 - f["r2"]))
    return out


def wf_mlr(rng, v):
    n = 60
    for _ in range(5000):
        sex = np.where(rng.random(n) < 0.5, "barrow", "gilt")
        gilt = (sex == "gilt").astype(float)
        start = np.round(rng.normal(30 - 3.0 * gilt, 3.0, n), 1)
        lys = np.round(rng.choice([0.95, 1.05, 1.15, 1.25], n), 2)
        age = np.round(70 + 2.6 * (start - 28) + rng.normal(0, 1.4, n))
        adg = np.round(400 + 12 * start + 300 * lys - 15 * gilt + rng.normal(0, 38, n))
        Xc = np.c_[np.ones(n), gilt]
        fc = ols(Xc, adg)
        if v["kind"] == "conf":
            Xf = np.c_[np.ones(n), start, lys, gilt]
            ff = ols(Xf, adg)
            if fc["p"][1] < 0.005 and ff["p"][3] > 0.1 and ff["p"][2] < 0.01 and fc["b"][1] < -50:
                break
        else:
            Xf = np.c_[np.ones(n), start, age, lys, gilt]
            ff = ols(Xf, adg)
            vf = vif(Xf[:, 1:])
            if vf[0] > 10 and ff["p"][1] > 0.1 and ff["p"][2] > 0.1 and ff["F"] > 5 and ff["pF"] < 1e-4:
                break
    else:
        raise RuntimeError("mlr")
    S = []
    cols = [("pig", "int", range(1, n + 1)), ("sex", "chr", sex), ("start_wt", "num", start)] + \
           ([("age", "num", age)] if v["kind"] == "coll" else []) + [("lys", "num", lys), ("adg", "num", adg)]
    S.append(step("Загрузить", task="Таблица <code>pigs</code>: пол, масса при постановке (кг), " + ("возраст при постановке (дни), " if v["kind"] == "coll" else "") +
                  "уровень SID-лизина в рационе (%, назначали жребием) и привес (г/сут).",
                  code=['> pigs <- read.csv("finisher.csv")', "> str(pigs)"], out=str_df(n, cols),
                  qs=[mc(rng, "Какие переменные назначали жребием?", "Только уровень лизина", ["Все", "Пол и лизин", "Никакие"],
                         "Пол и масса — свойства животного. Причинный вывод — только для лизина; остальное — поправки.")]))
    nm = ["adg", "start_wt"] + (["age"] if v["kind"] == "coll" else []) + ["lys"]
    M = np.c_[adg, start] if v["kind"] == "conf" else np.c_[adg, start, age]
    M = np.c_[M, lys]
    C = np.corrcoef(M.T)
    S.append(step("Корреляции", code=f'> round(cor(pigs[, c({", ".join(chr(34) + x + chr(34) for x in nm)})]), 2)',
                  out=R.matrix(nm, nm, [list(np.round(C[:, j], 2)) for j in range(len(nm))], fixed_dec={j: 2 for j in range(len(nm))}),
                  qs=[mc(rng, "Что настораживает?" if v["kind"] == "coll" else "Что видно?",
                         (f"start_wt и age почти одно и то же (r = {ru(C[1, 2], 2)}): вместе в модели они будут мешать друг другу" if v["kind"] == "coll"
                          else f"Привес заметно связан с начальной массой (r = {ru(C[0, 1], 2)})"),
                         (["Связь adg и lys слабая — значит лизин не важен", "Все корреляции меньше 1 — всё в порядке", "Корреляция 0,9 значит, что age вызывает start_wt"]
                          if v["kind"] == "coll" else
                          ["Лизин и масса сильно связаны", "Привес не связан ни с чем", "Корреляция 1 на диагонали — ошибка"]),
                         "Высокая корреляция предикторов (мультиколлинеарность) не портит прогноз, но делает отдельные коэффициенты неустойчивыми.")]))
    cr = named(["(Intercept)", "sexgilt"], fc["b"])
    S.append(step("Сырая разница полов", code="> coef(lm(adg ~ sex, data = pigs))", out=cr,
                  qs=[num("Насколько свинки растут медленнее боровов без поправок, г/сут?", -fc["b"][1], f"sexgilt = {R.r7(fc['b'][1])}: свинки медленнее на <b>{rua(-fc['b'][1], 3)}</b> г/сут.", rel=0.01)]))
    names = ["(Intercept)", "start_wt"] + (["age"] if v["kind"] == "coll" else []) + ["lys", "sexgilt"]
    form = "adg ~ start_wt + " + ("age + " if v["kind"] == "coll" else "") + "lys + sex"
    ll, disp, _ = lm_out(form, "pigs", Xf, adg, names)
    if v["kind"] == "conf":
        q2 = mc(rng, f"С поправками sexgilt = {R.r7(ff['b'][3])} (p = {pr(ff['p'][3])}), а без них {R.r7(fc['b'][1])}. Почему?",
                "Свинки пришли легче, а лёгкие растут медленнее: большая часть «эффекта пола» — эффект начальной массы",
                ["Модель с поправками неверна", "Поправка всегда уменьшает эффекты", "Из-за лизина: свинкам дали меньше"],
                "Начальная масса связана и с полом, и с привесом — это вмешивающийся фактор (конфаундер). Сравнение «при одинаковой массе» честнее.")
    else:
        q2 = mc(rng, f"F-тест модели p = {pr(ff['pF'])}, а start_wt и age по отдельности незначимы. Как так?",
                "Они почти дублируют друг друга: модель не может разделить их вклад, и SE обоих раздуваются",
                ["Масса и возраст не влияют на привес", "Ошибка в R", "F-тест врёт при 60 свиньях"],
                "Каждый коэффициент — эффект «при прочих равных». Нельзя изменить массу при том же возрасте — данных таких почти нет, поэтому неопределённость огромная.")
    S.append(step("Множественная регрессия", code=[f"> m <- lm({form}, data = pigs)", "> summary(m)"], out=ll,
                  qs=[mc(rng, f"Как прочитать коэффициент lys = {R.r7(ff['b'][-2])}?",
                         f"+0,1% лизина — привес выше в среднем на {rua(ff['b'][-2] / 10, 3)} г/сут при той же массе и том же поле",
                         [f"+1% лизина — привес в {rua(ff['b'][-2], 3)} раз выше", "Лизин объясняет столько-то процентов привеса",
                          f"+0,1% лизина — привес выше на {rua(ff['b'][-2], 3)} г/сут"],
                         "Коэффициент — на единицу X (1 процентный пункт). Шаг в опыте 0,1%, поэтому делим на 10."), q2]))
    vv = vif(Xf[:, 1:])
    S.append(step("Мультиколлинеарность: VIF", code=["> library(car)", "> vif(m)"],
                  out=named(names[1:-1] + ["sex"], vv),
                  qs=[mc(rng, "Что делать?", ("Убрать age (или start_wt): VIF больше 10, они несут одну информацию" if v["kind"] == "coll" else "Ничего: все VIF около 1–2"),
                         (["Ничего: VIF ни на что не влияет", "Убрать lys", "Удалить свиней с большим возрастом"] if v["kind"] == "coll"
                          else ["Убрать start_wt: VIF больше 1", "Удалить sex", "Добавить ещё предикторов"]),
                         "VIF = во сколько раз дисперсия коэффициента больше, чем без коллинеарности. До 5 — спокойно, больше 10 — проблема.")]))
    if v["kind"] == "coll":
        Xf2 = np.c_[np.ones(n), start, lys, gilt]
        ll2, d2, ff2 = lm_out("adg ~ start_wt + lys + sex", "pigs", Xf2, adg, ["(Intercept)", "start_wt", "lys", "sexgilt"])
        S.append(step("Модель без age", code=["> m <- lm(adg ~ start_wt + lys + sex, data = pigs)", "> summary(m)"], out=ll2,
                      qs=[num("Во сколько раз уменьшилась SE у start_wt?", ff["se"][1] / ff2["se"][1],
                              f"{R.r7(ff['se'][1])} / {R.r7(ff2['se'][1])} = <b>{rua(ff['se'][1] / ff2['se'][1], 3)}</b>. Примерно √VIF.", rel=0.03)]))
        Xf, ff = Xf2, ff2
    P, F = p_lm(Xf, adg, (1, 2, 3, 4))
    S.append(step("Диагностика", code=["> par(mfrow = c(2, 2))", "> plot(m, which = 1:4)"], plots=P,
                  qs=[mc(rng, "Есть проблемы?", "Нет явных: полоса без формы, Q-Q ровный, Cook's D все меньше 0,5",
                         ["Да: Scale-Location не идеально горизонтальна", "Да: подписаны 3 точки", "Да: Cook's D у некоторых больше 0"],
                         "Cook's D &gt; 1 — сигнал. Лёгкие колебания линий — норма.")]))
    X0 = np.delete(Xf, Xf.shape[1] - 2, axis=1)
    f0 = ols(X0, adg)
    al, ad = anova2("adg ~ start_wt + sex", "adg ~ start_wt + lys + sex", [f0["df"], ff["df"]], [f0["sse"], ff["sse"]])
    S.append(step("Нужен ли лизин в модели", code=["> m0 <- lm(adg ~ start_wt + sex, data = pigs)", "> anova(m0, m)"], out=al,
                  qs=[mc(rng, "Как связаны этот F и t для lys в summary(m)?", "F = t²: при одном добавленном коэффициенте это один и тот же тест",
                         ["Никак", "F всегда больше t в 10 раз", "F проверяет все коэффициенты сразу"],
                         f"t для lys = {R.fmt1(ff['t'][-2], 4)}, t² = {R.fmt1(ff['t'][-2] ** 2, 4)} = F. anova(m0, m) нужен, когда добавляют несколько коэффициентов сразу (например, фактор с 3+ уровнями).")]))
    rep = f"При поправке на начальную массу и пол каждые +0,1% SID-лизина повышали привес на {rua(ff['b'][-2] / 10, 2)} г/сут (p = {pr(ff['p'][-2])})"
    S.append(step("Записать результат", qs=[mc(rng, "Какая фраза верна?", rep,
                                                ["Пол вызывает разницу в привесе 15 г/сут", "Лизин объясняет R² × 100% привеса",
                                                 "Начальная масса — главный эффект опыта, потому что у неё самый большой t"],
                                                "Причинный вывод — для назначенного жребием лизина. Пол и масса — поправки.")]))
    items = [("Лизин назначали жребием", True), ("Проверили VIF", True), ("Эффект пола после поправки — причинный", False),
             ("Сырая и скорректированная разница полов различаются из-за начальной массы", v["kind"] == "conf")]
    S.append(step("Проверка", qs=[multi(rng, "Отметь верное.", items, "Свиньи — независимые единицы, если сидели по одной. В групповых загонах нужен (1 | pen).")]))
    return dict(g="regr", key="mlr", title="Множественная регрессия: лизин, масса, пол", vt=("конфаундер" if v["kind"] == "conf" else "коллинеарность"),
                goal="<p><b>Вопрос.</b> Насколько уровень лизина повышает привес на откорме, если учесть начальную массу и пол? 60 свиней в индивидуальных станках.</p>", steps=S)


def build_b(rng):
    out = []
    for v in RCBD_VAR:
        out.append(wf_rcbd(rng, v))
    for v in FACT_VAR:
        out.append(wf_fact(rng, v))
    for v in CHI_VAR:
        out.append(wf_chisq(rng, v))
    for v in SLR_VAR:
        out.append(wf_slr(rng, v))
    for v in MLR_VAR:
        out.append(wf_mlr(rng, v))
    return out
