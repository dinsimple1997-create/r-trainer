"""Generated experiments: description + 7 questions, with 0–3 planted errors."""
import numpy as np
from exp_core import U, TP, TR, TA, TX, OBS, CASECTRL, CROSS, BEFORE, pl, nn, num_ru, p_ru, t_report, f_report

NOERR = "Ошибок нет: опыт и анализ сделаны правильно"
DESIGNS = {
    "crd": "Полностью рандомизированный (CRD)", "rcbd": "Рандомизированные блоки (RCBD)", "pair": "Парный план (paired comparison)",
    "latin": "Латинский квадрат (Latin square)", "fact": "Факторный план 2×2 (factorial)", "split": "Расщеплённые делянки (split-plot)",
    "rep": "Повторные измерения (repeated)", "cross": "Перекрёстный план (cross-over)", "ba": "Сравнение до/после (before–after)",
    "cohort": "Когортное исследование (cohort)", "cc": "Случай-контроль (ретроспективное)", "xs": "Поперечный срез (cross-sectional)",
}
EXP_D = ["crd", "rcbd", "pair", "latin", "fact", "split", "rep", "cross"]


def mc(q, correct, wrongs, ex):
    assert len(set([correct] + list(wrongs[:3]))) == 4, (q, correct, wrongs)
    return dict(t="mc", q=q, correct=correct, wrongs=list(wrongs[:3]), ex=ex)


def num(q, a, ex, tol=0):
    return dict(t="num", q=q, a=float(a), tol=tol, ex=ex, unit="")


def design_q(code, rng, extra_ex=""):
    pool = [d for d in EXP_D if d != code] if code in EXP_D else [d for d in ["cohort", "cc", "xs", "ba", "crd"] if d != code]
    rng.shuffle(pool)
    exs = {
        "crd": "Обработки назначили единицам одним общим жребием, без блоков и второго фактора.",
        "rcbd": "Единицы разбиты на блоки, в каждом блоке все варианты по разу, жребий внутри блока.",
        "pair": "В каждой паре похожих единиц (сибсы) по одной на вариант — блоки размера 2.",
        "latin": "Каждый вариант по разу в каждой строке и каждом столбце: два мешающих фактора сразу.",
        "fact": "Все комбинации двух факторов назначены единицам одним жребием.",
        "split": "Один фактор на больших единицах (комнатах), другой на маленьких внутри них — два жребия.",
        "rep": "Одну и ту же единицу измеряют много раз во времени.",
        "cross": "Каждая единица получает все варианты по очереди, порядок по жребию, между периодами отмывка.",
        "ba": "Одну и ту же ферму сравнивают до и после изменения, параллельного контроля нет.",
        "cohort": "Группы набраны по фактору, который никто не назначал, и за ними следят до исхода.",
        "cc": "Набрали по исходу (случаи и контроли) и выясняли фактор в прошлом.",
        "xs": "Фактор и исход измерены одновременно в одной выборке.",
    }
    return mc("Какой это план исследования?", DESIGNS[code], [DESIGNS[d] for d in pool[:3]], exs[code] + extra_ex)


def obs_q(what):
    return mc("Это эксперимент или наблюдение?", f"Наблюдение: {what} никто не назначал",
              ["Эксперимент: группы сравнивали между собой", "Эксперимент с блоками: группы — это блоки", "Перекрёстный эксперимент без отмывки"],
              "Эксперимент — когда исследователь сам назначает фактор (лучше жребием). Здесь фактор уже был, его только записали.")


def _lc(t):
    return t[0].lower() + t[1:] if len(t) > 1 and t[1].islower() else t


def errors_q(errs, nonerrs, rng):
    k = len(errs)
    rng.shuffle(nonerrs)
    used = nonerrs[:4 - k]
    items = [(e[0], True) for e in errs] + [(n[0], False) for n in used] + [(NOERR, k == 0)]
    parts = []
    if k:
        parts.append("<b>Ошибки.</b> " + " ".join(f"{i + 1}) <i>{e[0]}</i>: {_lc(e[1])}" for i, e in enumerate(errs)))
    else:
        parts.append("<b>Здесь всё сделано правильно.</b>")
    if used:
        parts.append("<b>Не ошибки.</b> " + " ".join(f"<i>{n[0]}</i>: {_lc(n[1])}" for n in used))
    return dict(t="multi", q="Отметь все ошибки в исследовании и отчёте.", opts=items, ex="<br>".join(parts))


def pick_errors(rng, cand, exclusive=(), kmax=3):
    w = [0.15, 0.35, 0.35, 0.15]
    k = int(rng.choice(4, p=w))
    k = min(k, kmax, len(cand))
    if k == 0:
        return []
    order = list(rng.permutation(len(cand)))
    chosen = []
    for i in order:
        e = cand[i]
        if any(e in grp and any(c in grp for c in chosen) for grp in exclusive):
            continue
        chosen.append(e)
        if len(chosen) == k:
            break
    return chosen


def concl_q(kind, rng, errs=(), scope="этой фермы и похожих условий", avoid=()):
    opts = {
        "causal": "Причинный вывод, но только для этой фермы",
        "noncausal": "Причинный вывод нельзя: жребия не было",
        "noev": "Эффект не доказан, но и не исключён",
        "none_effect": "Эффекта нет: p больше 0,05, значит разницы нет",
        "all": "Причинный вывод для всех ферм страны",
        "nothing": "Никакого вывода: опыт ничего не говорит",
        "confounded": "Нельзя: эффект смешан с другим фактором",
        "assoc": "Только связь: фактор никто не назначал",
        "attr": "Под вопросом: выбывание сместило сравнение",
        "redo": "Не обоснован: p посчитан не по той единице",
        "multi": "Не доказан: одна находка из многих тестов",
        "interact": "Эффект зависит от уровня второго фактора",
    }
    exs = {
        "causal": "Жребий даёт право на причинный вывод. Но вывод распространяется только на совокупность, из которой взяты единицы: эту ферму и похожие условия.",
        "noncausal": "Без жребия выбор зависел от состояния единиц — это конфаундинг. Разницу нельзя приписать обработке.",
        "noev": "Большое p — недостаток данных, а не доказательство, что эффекта нет. Полезно посмотреть ДИ эффекта: он покажет, какие эффекты ещё возможны.",
        "confounded": "Если обработка совпадает с комнатой, местом, временем, порядком или последействием прошлого варианта, их эффекты не разделить никаким тестом.",
        "assoc": "В наблюдательном исследовании группы различаются не только фактором. Вывод — о связи; конфаундеры можно учесть моделью, но не исключить.",
        "attr": "Жребий уравнял группы в начале, но выбывание, связанное с исходом, разрушило это равенство. Нужен анализ всех назначенных животных.",
        "redo": "План хороший, но авторы посчитали p так, будто наблюдения независимы. Настоящий p больше; что он покажет, узнаем только после правильного анализа.",
        "multi": "Чем больше тестов, тем больше случайных «находок». Нужна поправка на множественные сравнения или один заранее выбранный главный показатель.",
        "interact": "Значимое взаимодействие значит, что общего «эффекта фактора» нет: смотрят средние по комбинациям или эффект внутри каждого уровня второго фактора.",
    }
    G = {"noncausal", "confounded", "assoc", "attr"}
    bad = set(avoid) | {kind}
    if kind in G:
        bad |= G
    if kind not in ("causal", "noev"):
        bad |= {"noev"}
    if "nullp" in errs and kind != "noev":
        bad |= {"none_effect"}
    wrong_pool = [o for o in ["causal", "noncausal", "noev", "none_effect", "all", "nothing", "assoc", "confounded"] if o not in bad]
    rng.shuffle(wrong_pool)
    for first in {"causal": ["all"], "noev": ["none_effect"]}.get(kind, []):
        if first in wrong_pool:
            wrong_pool = [first] + [o for o in wrong_pool if o != first]
    return mc("Какой вывод здесь корректен?", opts[kind], [opts[o] for o in wrong_pool[:3]], exs[kind])


def kind_of(errs):
    e = set(errs)
    if e & {"norand", "system", "pick"}:
        return "noncausal"
    if e & {"rooms", "posfix", "allAB", "nowash"}:
        return "confounded"
    if "attr" in e:
        return "attr"
    if e & {"pseudo", "indep"}:
        return "redo"
    if e & {"multi", "weekly"}:
        return "multi"
    if "nullp" in e:
        return "noev"
    return "causal"


BEST = {"prob": 0, "zinc": 2, "lys": 1, "pellet": 0, "phyt": 0, "enr": 1, "acid": 0, "fib": 1, "mats": 0, "energy": 1}


def best(c):
    return BEST.get(c["key"], 0)


def p_sig(rng):
    return float(rng.choice([rng.uniform(0.0005, 0.009), rng.uniform(0.01, 0.045), rng.uniform(0.01, 0.045)]))


def p_ns(rng):
    return float(rng.uniform(0.06, 0.45))


def animal_of(c):
    if c.get("animal") == "sow":
        return "sow"
    return "piglet" if c["stage"] == "доращивание" else "pig"


def golov(n):
    return f"{n} {pl(n, ('голове', 'головы', 'голов'))}"


def codes_str(lv):
    return ", ".join(f"{a} — {b}" for a, b in lv)


# =====================================================================
def d_crd_pen(rng):
    c = TP[int(rng.integers(len(TP)))]
    t = len(c["lv"])
    r = int(rng.integers(4, 11))
    N = t * r
    ppp = int(rng.integers(10, 26))
    an = animal_of(c)
    A = U[an]
    obs_pig = c["out_on"] == "pig"
    nobs = N * ppp
    cand = ["norand", "system", "general", "nullp", "attr"]
    if obs_pig:
        cand.append("pseudo")
    if t == 2:
        cand.append("rooms")
    if c.get("subjective"):
        cand.append("blind")
    cand.append("multi")
    errs = pick_errors(rng, cand, exclusive=[{"norand", "system", "rooms"}, {"nullp", "multi"}])
    lv = c["lv"]
    codes = [a for a, _ in lv]
    bc, bl = lv[best(c)]
    # allocation text and layout
    if "norand" in errs:
        alloc = f"Зоотехник сам выбрал {nn(r, U['pen']['forms'])} для варианта {bc} — те, где животные выглядели крепче; остальные загоны разделил между остальными вариантами."
        layout = [("Корпус", list(rng.permutation(codes * r)))]
    elif "system" in errs:
        alloc = f"Вариант {codes[0]} получили первые {nn(r, U['pen']['forms'])} от входа, " + ", ".join(f"{codes[i]} — следующие {r}" for i in range(1, t)) + "."
        layout = [("Корпус", sum(([cd] * r for cd in codes), []))]
    elif "rooms" in errs:
        alloc = f"Все {nn(r, U['pen']['forms'])} варианта {codes[0]} стоят в комнате 1, все {nn(r, U['pen']['forms'])} {codes[1]} — в комнате 2; внутри комнат загоны выбрали жребием."
        layout = [("Комната 1", [codes[0]] * r), ("Комната 2", [codes[1]] * r)]
    else:
        alloc = f"Загоны распределили жребием: по {r} на каждый вариант."
        layout = [("Корпус", list(rng.permutation(codes * r)))]
    days = int(rng.choice([28, 35, 42, 56, 70]))
    meas = (f"Через {nn(days, ('день', 'дня', 'дней'))} измерили {c['out']} ({c['unit']}) у каждого животного — всего {nn(nobs, A['forms'])}." if obs_pig
            else f"Для каждого загона за {nn(days, ('день', 'дня', 'дней'))} посчитали: {c['out']} ({c['unit']}).")
    if "multi" in errs:
        meas += " Кроме того, измерили ещё 11 показателей (поведение, гематология, качество туш и другие)."
    if "attr" in errs:
        meas += f" Павших и выбракованных из анализа исключили: в группе {codes[0]} выбыло {int(rng.integers(8, 15))}, в остальных — по {int(rng.integers(1, 4))}."
    if c.get("subjective"):
        meas += (" Хромоту оценивал зоотехник, который знал, где какая группа." if "blind" in errs
                 else " Оценку проводил ветврач, который не знал групп (слепая оценка).")
    # report
    p = p_ns(rng) if "nullp" in errs else p_sig(rng)
    if "pseudo" in errs:
        test = (f"Двухвыборочный t-тест по {nobs} животным: {t_report(p, nobs - 2)}" if t == 2
                else f"Однофакторный ANOVA по {nobs} животным: {f_report(p, t - 1, nobs - t)}")
    else:
        unit_txt = "средним загонов" if obs_pig else "загонам"
        test = (f"t-тест по {unit_txt} ({N}): {t_report(p, N - 2)}" if t == 2
                else f"Однофакторный ANOVA по {unit_txt} ({N}): {f_report(p, t - 1, N - t)}")
    if "multi" in errs:
        test += f". Из 12 показателей значимым оказался только этот"
    if "nullp" in errs:
        concl = f"«{c['what'].capitalize()} не влияет: разницы нет»."
    else:
        concl = f"«Вариант {bc} ({bl}) {c['verb']}" + (" — рекомендуем всем фермам страны»." if "general" in errs else "»." )
    story = (f"<p><b>Цель.</b> Сравнить варианты фактора «{c['what']}» ({c['stage']}): {codes_str(lv)}.</p>"
             f"<p><b>Условия.</b> {nn(N, U['pen']['forms'])} по {golov(ppp)}. Вариант задаётся на загон целиком: общая кормушка и одинаковые условия для всех животных загона.</p>"
             f"<p><b>Распределение.</b> {alloc}</p><p><b>Измерения.</b> {meas}</p>"
             f"<div class='report'><b>Из отчёта.</b> {test}. Вывод: {concl}</div>")
    # errors and non-errors
    E = {
        "pseudo": (f"Анализ сделан по всем {nobs} животным",
                   f"Вариант назначали загонам, поэтому n = {N} загонов, а не {nobs} животных; p авторов слишком маленький."),
        "norand": (f"Загоны для варианта {bc} выбирал зоотехник",
                   "Выбор «крепких» загонов смешивает эффект корма с исходным состоянием животных."),
        "system": ("Варианты расставлены по порядку от входа",
                   "Место в корпусе (сквозняк, свет, первыми кормят) совпадает с вариантом — систематическая ошибка."),
        "rooms": ("Каждый вариант стоит в своей комнате",
                  "По сути единица — комната, и у каждого варианта по одной: повторностей нет."),
        "general": ("Вывод сделан для всех ферм страны", "Одна ферма — один корм, генетика и здоровье стада."),
        "nullp": ("По p > 0,05 сделан вывод «не влияет»", "Большое p — мало данных, а не «нет эффекта»."),
        "multi": ("Вывод сделан по 1 значимому показателю из 12",
                  "При 12 тестах один p < 0,05 легко получить случайно (ожидаемо ~0,6 ложной находки)."),
        "attr": ("Павших и выбракованных исключили из анализа",
                 "Если выбывали самые слабые, в этой группе остались лучшие животные, и группа выглядит лучше."),
        "blind": ("Хромоту оценивал зоотехник, знавший группы", "Нужна слепая оценка, иначе ожидания влияют на баллы."),
    }
    nonerr = [("В опыте есть контрольный вариант", "Контроль есть — это правильно.")] if set(codes) & {"C", "N", "Z0", "K"} else []
    if not set(errs) & {"norand", "system", "rooms"}:
        nonerr.append(("Загоны распределили жребием", "Жребий — правильно: он даёт право на причинный вывод."))
    if "rooms" not in errs:
        nonerr.append(("У всех вариантов одинаковое число загонов", "Равное число единиц — удобно и правильно."))
    if "pseudo" not in errs:
        nonerr.append(("Анализ сделан по загонам, а не по животным", "Единица анализа совпадает с единицей эксперимента — верно."))
    nonerr.append(("Вариант задавали на загон целиком", "Это не ошибка, а причина, почему единица — загон."))
    nonerr.append(("В каждом загоне одинаковое число голов", "Равные загоны — удобно и правильно."))
    if c.get("subjective") and "blind" not in errs:
        nonerr.append(("Оценку делал ветврач, не знавший групп", "Слепая оценка — правильно."))
    qs = []
    if "rooms" in errs:
        qs.append(mc("Что здесь по сути единица эксперимента?", "Комната: по одной на вариант",
                     [f"Загон: по {r} на каждый вариант", f"{A['cap']}: всего {nobs} в опыте", "Ферма: опыт на одной ферме"],
                     "Раз все загоны варианта стоят в одной комнате, сравниваются комнаты. Загоны внутри комнаты — не независимые повторности варианта."))
    else:
        qs.append(mc("Что здесь единица эксперимента?", f"Загон ({N})", [f"{A['cap']} ({nobs})", "Комната", "Ферма"],
                     "Единица эксперимента — то, чему независимо назначали вариант. Вариант задавали на загон."))
    if obs_pig:
        qs.append(mc("А что единица наблюдения?", A["cap"], ["Загон", "Кормушка", "Ферма"],
                     f"Измеряли каждое животное: на одну единицу эксперимента приходится {ppp} наблюдений. n считают по загонам."))
    else:
        qs.append(mc("А что единица наблюдения?", "Загон", [A["cap"], "Кормушка", "Ферма"],
                     "Показатель считали для загона, поэтому единицы наблюдения и эксперимента совпадают."))
    qs.append(design_q("crd", rng, " Ошибки в распределении (если есть) ловим в следующем вопросе."))
    qs.append(errors_q([E[e] for e in errs], nonerr, rng))
    var = c["var"]
    if "rooms" in errs:
        qs.append(mc("Как правильно анализировать?", "Никак: нужен новый опыт с комнатами-блоками",
                     [f"t-тест по средним {N} загонов", f"lmer({var} ~ trt + (1 | pen), data = d)", f"t-тест по всем {nobs} животным"],
                     "Эффект корма и эффект комнаты совпали. Правильный план — комнаты как блоки (в каждой комнате все варианты) или несколько комнат на вариант."))
        qs.append(num("Сколько независимых повторностей у каждого варианта?", 1, "По одной комнате на вариант — повторность одна, ошибку эксперимента оценить нельзя."))
    else:
        dat = "pen_means" if obs_pig else "pens"
        fn = "t.test" if t == 2 else "aov"
        wr = ([f"{fn}({var} ~ trt, data = pigs)", f"lm({var} ~ trt + pen, data = pigs)"] if obs_pig else
              [f"{fn}({var} ~ trt + pen, data = pens)", f"lmer({var} ~ trt + (1 | pen), data = pens)"])
        wr.append(f"aov({var} ~ trt * room, data = {dat})")
        qs.append(mc("Как правильно анализировать?", f"{fn}({var} ~ trt, data = {dat})", wr,
                     ("<code>pen_means</code> — таблица, где одна строка на загон (средняя по животным загона). " if obs_pig else
                      "<code>pens</code> — одна строка на загон. ") +
                     "Анализ идёт по единицам эксперимента." + (f" Эквивалент на уровне животных — <code>lmer({var} ~ trt + (1 | pen), data = pigs)</code>." if obs_pig else
                     " Загон в модели лишний: на каждый загон одно число, и он совпадает с ошибкой.") + " Комнат как фактора в этом опыте нет."))
        qs.append(num("Сколько степеней свободы у ошибки при правильном анализе?", N - t,
                      f"Загонов {N}, вариантов {t}: df = {N} − {t} = <b>{N - t}</b>."))
    qs.append(concl_q(kind_of(errs), rng, errs))
    return dict(design="crd", title=f"{c['what'].capitalize()}", topic=f"{c['stage'].capitalize()} · загоны", story=story,
                scheme=dict(title=nn(N, U['pen']['forms']), rooms=layout, legend={a: b for a, b in lv}), qs=qs)


def d_crd_ind(rng):
    c = TA[int(rng.integers(len(TA)))]
    while c["key"] in ("iron", "teeth"):
        c = TA[int(rng.integers(len(TA)))]
    A = U[c["unit"]]
    r = int(rng.integers(10, 41))
    N = 2 * r
    lv = c["lv"]
    codes = [a for a, _ in lv]
    subj = c.get("subjective")
    cand = ["norand", "general", "nullp", "attr"] + (["blind"] if subj else [])
    errs = pick_errors(rng, cand, exclusive=[{"nullp"}])
    alloc = (f"Ветврач сам решал, кому дать вариант {codes[0]}: тем, кто выглядел слабее." if "norand" in errs
             else f"Вариант для каждого животного определяли по списку случайных чисел: по {r} на вариант.")
    meas = f"Измеряли: {c['out']} ({c['unit_y']})."
    if subj:
        meas += (" Оценку делал тот же ветврач, который давал препарат." if "blind" in errs else " Оценку делал сотрудник, не знавший, кто что получил.")
    if "attr" in errs:
        meas += f" Животных, которым стало хуже, исключили из опыта: {int(rng.integers(5, 10))} в группе {codes[1]}, ни одного в группе {codes[0]}."
    p = p_ns(rng) if "nullp" in errs else p_sig(rng)
    if subj:
        test = f"Ранговая сумма Вилкоксона: W = {int(rng.integers(r * r // 4, r * r // 2))}, {p_ru(p)}"
    else:
        test = f"Двухвыборочный t-тест: {t_report(p, N - 2)}"
    concl = (f"«{c['what'].capitalize()} не влияет»." if "nullp" in errs else
             f"«{lv[0][1].capitalize()} {c['verb']}" + (" у всех свиней страны»." if "general" in errs else "»."))
    story = (f"<p><b>Цель.</b> Сравнить варианты фактора «{c['what']}»: {codes_str(lv)}.</p><p><b>Животные.</b> {nn(N, A['forms'])} одной фермы; каждое животное получает вариант индивидуально.</p>"
             f"<p><b>Распределение.</b> {alloc}</p><p><b>Измерения.</b> {meas}</p><div class='report'><b>Из отчёта.</b> {test}. Вывод: {concl}</div>")
    E = {
        "norand": ("Вариант выбирал ветврач по виду животного", "Выбор «слабых» смешивает эффект препарата с исходным состоянием."),
        "general": ("Вывод сделан для всех свиней страны", "Совокупность — одна ферма."),
        "nullp": ("По p > 0,05 сделан вывод «не влияет»", "Большое p — недостаток данных, не доказательство отсутствия эффекта."),
        "attr": ("Ухудшившихся животных исключили из опыта", "Выбывание, связанное с исходом, смещает сравнение (нужен анализ всех назначенных — intention to treat)."),
        "blind": ("Балл ставил тот, кто давал препарат", "Нужна слепая оценка."),
    }
    nonerr = [("Каждое животное получало вариант индивидуально", "Поэтому единица — животное, это верно."),
              ("Есть контрольный вариант", "Контроль есть — правильно."),
              ("Группы одинакового размера", "Это удобно, не ошибка.")]
    if "norand" not in errs:
        nonerr.append(("Варианты распределили по случайным числам", "Рандомизация — правильно."))
    if subj and "blind" not in errs:
        nonerr.append(("Оценку делал сотрудник, не знавший групп", "Слепая оценка — правильно."))
    if subj:
        nonerr.append(("Для баллов использовали ранговый тест", "Порядковая шкала — ранговый тест уместен."))
    qs = [mc("Что здесь единица эксперимента?", f"{A['cap']} ({N})", ["Загон", "Ферма", "Одно измерение"],
             "Вариант назначали каждому животному отдельно, и измеряли каждое — единицы эксперимента и наблюдения совпадают."),
          num("Сколько животных получил каждый вариант?", r, f"Всего {N}, вариантов 2: по <b>{r}</b>."),
          design_q("crd", rng), errors_q([E[e] for e in errs], nonerr, rng)]
    var = c["var"]
    if subj:
        qs.append(mc("Как правильно анализировать?", f"wilcox.test({var} ~ trt, data = d)",
                     [f"t.test({var} ~ trt, data = d, paired = TRUE)", f"aov({var} ~ trt * sex, data = d)", f"lmer({var} ~ trt + (1 | room), data = d)"],
                     "Балл 0–5 — порядковая шкала: ранговая сумма Вилкоксона. Пар здесь нет, загоны не назначали."))
        qs.append(num("Сколько всего животных в анализе?", N, f"По {r} в группе: <b>{N}</b>."))
    else:
        qs.append(mc("Как правильно анализировать?", f"t.test({var} ~ trt, data = d, var.equal = TRUE)",
                     [f"t.test({var} ~ trt, data = d, paired = TRUE)", f"chisq.test(table(d$trt, d${var} > 0))", f"lmer({var} ~ trt + (1 | room), data = d)"],
                     "Две независимые группы животных с числовым исходом — двухвыборочный t-тест."))
        qs.append(num("Сколько степеней свободы у t-теста?", N - 2, f"df = n₁ + n₂ − 2 = {N} − 2 = <b>{N - 2}</b>."))
    qs.append(concl_q(kind_of(errs), rng, errs))
    return dict(design="crd", title=c["what"].capitalize(), topic=f"{A['cap']} · ветеринария", story=story, scheme=None, qs=qs)


def d_rcbd(rng):
    c = TP[int(rng.integers(len(TP)))]
    t = len(c["lv"])
    b = int(rng.integers(4, 9))
    N = t * b
    ppp = int(rng.integers(10, 25))
    an = animal_of(c)
    A = U[an]
    obs_pig = c["out_on"] == "pig"
    bt = rng.choice(["room", "weight", "batch"])
    if bt == "room":
        bname, bforms, why = "комната", U["room"]["forms"], "комнаты различаются микроклиматом: у торцов холоднее"
    elif bt == "weight":
        bname, bforms, why = "весовой блок", ("весовой блок", "весовых блока", "весовых блоков"), f"животные пришли разной массы: самых тяжёлых посадили в загоны блока 1, самых лёгких — в блок {b}"
    else:
        bname, bforms, why = "партия", U["batch"]["forms"], "заселения шли в разные месяцы, с разными поставками животных"
    lv = c["lv"]
    codes = [a for a, _ in lv]
    cand = ["noblock", "nullp", "general", "attr", "posfix"] + (["pseudo"] if obs_pig else [])
    errs = pick_errors(rng, cand, exclusive=[{"nullp"}])
    if "posfix" in errs:
        alloc = f"В каждом блоке ({bname}) {codes[0]} всегда ставили в первый загон от прохода, остальные — дальше по порядку."
        layout = [(f"{bname.capitalize()} {i + 1}", codes[:]) for i in range(b)]
    else:
        alloc = f"Внутри каждого блока ({bname}) варианты распределили жребием — по одному загону на вариант."
        layout = [(f"{bname.capitalize()} {i + 1}", list(rng.permutation(codes))) for i in range(b)]
    meas = (f"Измерили {c['out']} ({c['unit']}) у каждого животного — всего {nobs_txt(N * ppp, A)}." if obs_pig
            else f"Для каждого загона посчитали: {c['out']} ({c['unit']}).")
    if "attr" in errs:
        meas += f" Павших исключили; в группе {codes[-1]} их было втрое больше."
    p = p_ns(rng) if "nullp" in errs else p_sig(rng)
    if "pseudo" in errs:
        test = f"Однофакторный ANOVA по {N * ppp} животным: {f_report(p, t - 1, N * ppp - t)}"
    elif "noblock" in errs:
        test = f"Однофакторный ANOVA по {N} загонам: {f_report(p, t - 1, N - t)}"
    else:
        test = f"ANOVA с блоком по {N} загонам: {f_report(p, t - 1, (t - 1) * (b - 1))}"
    concl = (f"«{c['what'].capitalize()} не влияет»." if "nullp" in errs else
             f"«{lv[best(c)][1].capitalize()} {c['verb']}" + (" на всех фермах»." if "general" in errs else "»."))
    story = (f"<p><b>Цель.</b> Сравнить варианты фактора «{c['what']}» ({c['stage']}): {codes_str(lv)}.</p>"
             f"<p><b>Условия.</b> {nn(b, bforms)} по {nn(t, U['pen']['forms'])}, в загоне {nn(ppp, ('голова', 'головы', 'голов'))}. Блоки нужны, потому что {why}.</p>"
             f"<p><b>Распределение.</b> {alloc}</p><p><b>Измерения.</b> {meas}</p>"
             f"<div class='report'><b>Из отчёта.</b> {test}. Вывод: {concl}</div>")
    E = {
        "noblock": ("Блок не включили в модель", "Блок обязан входить в модель: aov(y ~ trt + block)."),
        "nullp": ("По p > 0,05 сделан вывод «не влияет»", "Большое p — недостаток данных."),
        "general": ("Вывод сделан для всех ферм страны", "Совокупность — одна ферма."),
        "attr": ("Павших исключили из анализа", "Выбывание, связанное с вариантом, смещает сравнение."),
        "posfix": (f"В каждом блоке {codes[0]} стоял первым от прохода", "Положение загона (у прохода) совпадает с вариантом — нужна рандомизация внутри блока."),
        "pseudo": (f"Анализ сделан по всем {N * ppp} животным", "Единица эксперимента — загон."),
    }
    nonerr = [("В каждом блоке все варианты по одному разу", "Это и есть полный блок — верно."),
              ("Блоки сформировали по известному мешающему фактору", "Так и нужно формировать блоки."),
              ("По одному загону на вариант в блоке", "Повторность дают сами блоки — не ошибка.")]
    if "posfix" not in errs:
        nonerr.append(("Жребий бросали внутри каждого блока", "Правило RCBD."))
    if "noblock" not in errs and "pseudo" not in errs:
        nonerr.append(("Блок включён в модель без взаимодействия", "Верно: при одной единице на вариант в блоке взаимодействие не оценить."))
    var = c["var"]
    dat = "pen_means" if obs_pig else "pens"
    qs = [mc("Что здесь единица эксперимента?", "Загон", [A["cap"], bname.capitalize(), "Ферма"], "Вариант назначали загону. Блок — группа загонов."),
          mc(f"Какую роль играет «{bname}»?", "Блок: похожие загоны, в нём все варианты",
             ["Единица эксперимента: блоку — один вариант", "Повторность: в нём загоны одного варианта", "Второй изучаемый фактор, как и корм"],
             "Блок убирает из ошибки разброс по мешающему фактору — сравнение вариантов внутри блока точнее."),
          design_q("rcbd", rng), errors_q([E[e] for e in errs], nonerr, rng),
          mc("Какая модель правильная?", f"aov({var} ~ trt + block, data = {dat})",
             [f"aov({var} ~ trt, data = {dat})", f"aov({var} ~ trt * block, data = {dat})",
              f"aov({var} ~ trt + block, data = pigs)" if obs_pig else f"kruskal.test({var} ~ trt, data = pens)"],
             "Блок — фактор в модели, без взаимодействия (на взаимодействие не осталось бы df ошибки)." + (" Для двух вариантов это то же, что парный t-тест по блокам." if t == 2 else "")),
          num("Сколько степеней свободы у ошибки в правильной модели?", (t - 1) * (b - 1),
              f"(вариантов − 1)(блоков − 1) = ({t} − 1)({b} − 1) = <b>{(t - 1) * (b - 1)}</b>."),
          concl_q(kind_of(errs), rng, errs)]
    return dict(design="rcbd", title=c["what"].capitalize(), topic=f"{c['stage'].capitalize()} · загоны", story=story,
                scheme=dict(title=f"{nn(b, ('блок', 'блока', 'блоков'))} × {nn(t, U['pen']['forms'])}", rooms=layout, legend={a: bb for a, bb in lv}), qs=qs)


def nobs_txt(n, A):
    return nn(n, A["forms"])


def d_pair(rng):
    c = [x for x in TA if x.get("block") == "litter"][int(rng.integers(2))]
    n = int(rng.integers(8, 21))
    lv = c["lv"]
    codes = [a for a, _ in lv]
    subj = c.get("subjective")
    cand = ["unpaired", "pick", "nullp", "general"] + (["blind"] if subj else [])
    errs = pick_errors(rng, cand, exclusive=[{"nullp"}])
    alloc = (f"В каждом гнезде выбрали двух поросят одного пола и похожей массы. {codes[0]} получал самый крупный из пары — так решил оператор." if "pick" in errs
             else f"В каждом гнезде выбрали двух поросят одного пола и похожей массы; монетка решала, кто из пары получит {codes[0]}.")
    meas = f"Измеряли: {c['out']} ({c['unit_y']})."
    if subj:
        meas += " Оценку делал оператор, знавший варианты." if "blind" in errs else " Оценку делал сотрудник, не знавший вариантов."
    p = p_ns(rng) if "nullp" in errs else p_sig(rng)
    test = (f"Двухвыборочный t-тест по {2 * n} поросятам: {t_report(p, 2 * n - 2)}" if "unpaired" in errs
            else f"Парный t-тест по {n} гнёздам: {t_report(p, n - 1)}")
    concl = (f"«{c['what'].capitalize()} не влияет»." if "nullp" in errs else
             f"«{lv[0][1].capitalize()} {c['verb']}" + (" — рекомендуем всем фермам»." if "general" in errs else "»."))
    story = (f"<p><b>Цель.</b> Сравнить варианты фактора «{c['what']}»: {codes_str(lv)}.</p><p><b>Животные.</b> {nn(n, U['litter']['forms'])} одной фермы.</p>"
             f"<p><b>Распределение.</b> {alloc}</p><p><b>Измерения.</b> {meas}</p><div class='report'><b>Из отчёта.</b> {test}. Вывод: {concl}</div>")
    E = {
        "unpaired": ("Использовали двухвыборочный t-тест", "Пары-сибсы похожи: двухвыборочный тест выбрасывает эту связь и завышает шум."),
        "pick": ("Вариант внутри пары выбирал оператор", "Крупный поросёнок всегда получал один вариант — эффект смешан с массой."),
        "nullp": ("По p > 0,05 сделан вывод «не влияет»", "Мало данных ≠ нет эффекта."),
        "general": ("Вывод сделан для всех ферм страны", "Совокупность — одна ферма."),
        "blind": ("Балл ставил оператор, знавший варианты", "Нужна слепая оценка."),
    }
    nonerr = [("Пары подобрали по полу и массе внутри гнезда", "Так и формируют пары — верно."),
              ("В каждом гнезде оба варианта", "Это и делает план парным."),
              ("В анализ вошли все гнёзда", "Не ошибка.")]
    if "pick" not in errs:
        nonerr.append(("Внутри пары вариант решала монетка", "Рандомизация внутри блока — верно."))
    if "unpaired" not in errs:
        nonerr.append(("Использовали парный t-тест по гнёздам", "Верный анализ."))
    qs = [mc("Что здесь единица эксперимента?", "Поросёнок", ["Гнездо", "Свиноматка", "Ферма"], "Вариант назначали каждому поросёнку внутри пары."),
          mc("Что здесь блок?", "Гнездо: два поросёнка-сибса", ["Поросёнок: он получает вариант", "Вариант: их два в опыте", "Ферма: все гнёзда с неё"], "Сибсы похожи генетикой и материнским молоком — блок размера 2."),
          design_q("pair", rng), errors_q([E[e] for e in errs], nonerr, rng),
          mc("Как правильно анализировать?", "Парный t-тест по разностям в гнёздах",
             [f"Двухвыборочный t-тест по {2 * n} поросятам", "chisq.test по числу улучшившихся", "aov(y ~ trt) без учёта гнезда"],
             f"Или эквивалентно: <code>lmer(y ~ trt + (1 | litter))</code>." + (" Для баллов — wilcox.test(x, y, paired = TRUE)." if subj else "")),
          num("Сколько степеней свободы у парного t-теста?", n - 1, f"Пар {n}: df = {n} − 1 = <b>{n - 1}</b>."),
          concl_q(kind_of(errs), rng, errs)]
    return dict(design="pair", title=c["what"].capitalize(), topic="Поросёнок · ветеринария", story=story, scheme=None, qs=qs)


def d_latin(rng):
    c = [x for x in TP if len(x["lv"]) == 3][int(rng.integers(3))]
    t = 3
    lv = c["lv"]
    codes = [a for a, _ in lv]
    base = [list(np.roll(codes, -i)) for i in range(t)]
    rows = [base[i] for i in rng.permutation(t)]
    cols = list(rng.permutation(t))
    sq = [[r_[j] for j in cols] for r_ in rows]
    cand = ["noblocks", "nullp", "general", "pseudo"]
    errs = pick_errors(rng, cand, exclusive=[{"noblocks", "pseudo"}, {"nullp"}])
    ppr = int(rng.integers(60, 200))
    p = p_ns(rng) if "nullp" in errs else p_sig(rng)
    if "noblocks" in errs:
        test = f"Однофакторный ANOVA по 9 значениям: {f_report(p, 2, 6)}"
    elif "pseudo" in errs:
        test = f"ANOVA по всем {9 * ppr} животным: {f_report(p, 2, 9 * ppr - 3)}"
    else:
        test = f"ANOVA с комнатой и партией: {f_report(p, 2, 2)}"
    concl = (f"«{c['what'].capitalize()} не влияет»." if "nullp" in errs else
             f"«{lv[best(c)][1].capitalize()} {c['verb']}" + (" на любой ферме»." if "general" in errs else "»."))
    story = (f"<p><b>Цель.</b> Сравнить варианты фактора «{c['what']}» ({c['stage']}): {codes_str(lv)}.</p>"
             f"<p><b>Условия.</b> 3 комнаты по {golov(ppr)}; опыт шёл в 3 партиях (заселениях) подряд. Комнаты различаются, и партии тоже (сезон, поставка поросят). В каждой партии вся комната получает один вариант.</p>"
             f"<p><b>Распределение.</b> Варианты расставили так, что каждый был ровно один раз в каждой комнате и в каждой партии; какие строки и столбцы — решил жребий.</p>"
             f"<p><b>Измерения.</b> Для каждой комнаты в каждой партии — {c['out']} ({c['unit']}).</p>"
             f"<div class='report'><b>Из отчёта.</b> {test}. Вывод: {concl}</div>")
    E = {"noblocks": ("В модели только вариант, без комнаты и партии", "В латинском квадрате оба мешающих фактора — в модели: aov(y ~ trt + room + batch)."),
         "nullp": ("По p > 0,05 сделан вывод «не влияет»", "Особенно при df ошибки = 2: данных очень мало."),
         "general": ("Вывод сделан для любой фермы", "Совокупность — одна ферма."),
         "pseudo": ("Анализ сделан по всем животным", "Единиц эксперимента всего 9.")}
    nonerr = [("Каждый вариант по разу в каждой комнате и в каждой партии", "Это и есть латинский квадрат."),
              ("Расстановку строк и столбцов выбрали жребием", "Рандомизация квадрата — верно."),
              ("В каждой партии вся комната получала один вариант", "Поэтому единица — комната в партии."),
              ("Исход считали по комнате целиком", "Исход на уровне единицы эксперимента — верно."),
              ("Число вариантов равно числу комнат и партий", "Так и должно быть в латинском квадрате.")]
    qs = [mc("Что здесь единица эксперимента?", "Комната в одной партии", ["Комната на всё время опыта", "Партия (заселение)", f"{U[animal_of(c)]['cap']} в комнате"],
             "Вариант назначали комнате на время одной партии: 3 комнаты × 3 партии = 9 единиц."),
          mc("Что здесь «строки» и «столбцы» квадрата?", "Комнаты и партии",
             ["Варианты и повторности", "Комнаты и варианты", "Дни и загоны"],
             "Это два мешающих фактора: латинский квадрат контролирует их одновременно, а варианты стоят в клетках."),
          design_q("latin", rng), errors_q([E[e] for e in errs], nonerr, rng),
          mc("Какая модель правильная?", f"aov({c['var']} ~ trt + room + batch, data = d)",
             [f"aov({c['var']} ~ trt * room * batch, data = d)", f"aov({c['var']} ~ trt + room, data = d)", f"lmer({c['var']} ~ trt + (1 | room), data = d)"],
             "Оба мешающих фактора — в модели, без взаимодействий."),
          num("Сколько степеней свободы у ошибки?", 2, "(t − 1)(t − 2) = (3 − 1)(3 − 2) = <b>2</b>. Маленький квадрат даёт очень мало df — часто его повторяют."),
          concl_q(kind_of(errs), rng, errs)]
    return dict(design="latin", title=c["what"].capitalize(), topic=f"{c['stage'].capitalize()} · комнаты", story=story,
                scheme=dict(title="строки — комнаты, слева направо — партии 1, 2, 3", rooms=[(f"Комната {i + 1}", sq[i]) for i in range(t)],
                            legend={a: bb for a, bb in lv}), qs=qs)


FACT = [
    (("Энергия корма", [("L", "низкая"), ("H", "высокая")]), ("Фитаза", [("F", "есть"), ("C", "нет")]), "откорм", "среднесуточный привес", "adg", "pig"),
    (("Лизин", [("L", "низкий"), ("H", "высокий")]), ("Форма корма", [("G", "гранулы"), ("M", "рассыпной")]), "откорм", "конверсия корма", "fcr", "pen"),
    (("Обогащение среды", [("N", "нет"), ("S", "солома")]), ("Плотность посадки", [("D7", "0,7 м² на голову"), ("D9", "0,9 м² на голову")]), "откорм", "число укусов хвостов за час", "bites", "pen"),
    (("Пробиотик", [("C", "нет"), ("P", "есть")]), ("Клетчатка", [("K3", "3 %"), ("K5", "5 %")]), "доращивание", "доля дней с диареей", "diar", "pen"),
]


def d_fact(rng):
    (fa, la), (fb, lb), stage, out, var, on = FACT[int(rng.integers(len(FACT)))]
    r = int(rng.integers(3, 7))
    N = 4 * r
    ppp = int(rng.integers(10, 22))
    combos = [(x[0] + y[0], f"{fa.lower()} {x[1]}, {fb.lower()} {y[1]}") for x in la for y in lb]
    cand = ["separate", "interact", "nullp", "general"] + (["pseudo"] if on == "pig" else [])
    errs = pick_errors(rng, cand, exclusive=[{"separate", "pseudo"}, {"nullp", "interact"}, {"separate", "interact"}])
    p = p_ns(rng) if "nullp" in errs else p_sig(rng)
    if "separate" in errs:
        test = f"Два отдельных t-теста: {fa.lower()} — {t_report(p, N - 2)}; {fb.lower()} — {p_ru(p_ns(rng))}"
    elif "pseudo" in errs:
        test = f"Двухфакторный ANOVA по {N * ppp} животным: {fa.lower()} {f_report(p, 1, N * ppp - 4)}"
    else:
        test = f"Двухфакторный ANOVA по {N} загонам: {fa.lower()} {f_report(p, 1, N - 4)}"
    if "interact" in errs:
        test += f"; взаимодействие {p_ru(float(rng.uniform(0.002, 0.02)))}: эффект есть только при варианте {lb[0][0]}"
    elif "separate" not in errs:
        test += f"; взаимодействие {p_ru(p_ns(rng))}"
    hi = la[1]
    concl = (f"«{fa} не влияет»." if "nullp" in errs else
             f"«Вариант {hi[0]} ({fa.lower()} {hi[1]}) лучше при любом уровне фактора «{fb.lower()}»" + (" — на всех фермах»." if "general" in errs else "»."))
    story = (f"<p><b>Цель.</b> Изучить вместе два фактора ({stage}): {fa.lower()} ({la[0][0]} — {la[0][1]}, {la[1][0]} — {la[1][1]}) "
             f"и {fb.lower()} ({lb[0][0]} — {lb[0][1]}, {lb[1][0]} — {lb[1][1]}).</p>"
             f"<p><b>Условия.</b> {nn(N, U['pen']['forms'])} по {golov(ppp)}, условия задаются на загон.</p>"
             f"<p><b>Распределение.</b> Четыре комбинации ({', '.join(cc for cc, _ in combos)}) назначили загонам общим жребием, по {r} на комбинацию.</p>"
             f"<p><b>Измерения.</b> {out.capitalize()} — " + ("у каждого животного." if on == "pig" else "по загону.") + "</p>"
             f"<div class='report'><b>Из отчёта.</b> {test}. Вывод: {concl}</div>")
    E = {"separate": ("Каждый фактор проверили своим t-тестом", "Нужен двухфакторный ANOVA с взаимодействием: aov(y ~ A * B)."),
         "interact": ("Вывод: эффект одинаков при любом уровне второго",
                      "При взаимодействии эффект одного фактора зависит от уровня другого — его описывают отдельно для каждого уровня."),
         "nullp": ("По p > 0,05 сделан вывод «не влияет»", "Мало данных ≠ нет эффекта."),
         "general": ("Вывод сделан для всех ферм страны", "Совокупность — одна ферма."),
         "pseudo": (f"Анализ сделан по всем {N * ppp} животным", "Единица — загон.")}
    nonerr = [("Все четыре комбинации есть в опыте", "Полный факторный план — верно."),
              ("Комбинации назначили общим жребием", "Так и делают в факторном CRD."),
              ("Одинаковое число загонов на комбинацию", "Сбалансированный план — удобно.")]
    if not set(errs) & {"separate", "pseudo"}:
        nonerr.append(("В модель включили взаимодействие факторов", "Верно: факторный план нужен именно для этого."))
    qs = [mc("Что здесь единица эксперимента?", "Загон", [U["pig"]["cap"], "Комбинация факторов", "Ферма"], "Комбинацию назначали загону."),
          num("Сколько групп (комбинаций) в опыте?", 4, f"2 уровня × 2 уровня = <b>4</b>: {', '.join(cc for cc, _ in combos)}."),
          design_q("fact", rng), errors_q([E[e] for e in errs], nonerr, rng),
          mc("Какая модель правильная?", f"aov({var} ~ A * B, data = {'pen_means' if on == 'pig' else 'pens'})",
             [f"t.test({var} ~ A); t.test({var} ~ B)", f"aov({var} ~ A + B, data = pens)", f"lmer({var} ~ A * B + (1 | room), data = pens)"],
             "A * B = A + B + A:B. Комнат и двух уровней рандомизации здесь нет — это не split-plot."),
          num("Сколько степеней свободы у ошибки?", 4 * (r - 1), f"Загонов {N}, групп 4: df = {N} − 4 = <b>{4 * (r - 1)}</b>."),
          concl_q("interact" if "interact" in errs else kind_of(errs), rng, errs)]
    return dict(design="fact", title=f"{fa} × {fb.lower()}", topic=f"{stage.capitalize()} · загоны", story=story,
                scheme=dict(title=nn(N, U['pen']['forms']) + ", общий жребий", rooms=[("Корпус", list(rng.permutation([cc for cc, _ in combos] * r)))],
                            legend=dict(combos)), qs=qs)


def d_split(rng):
    w = TR[int(rng.integers(len(TR)))]
    two = [x for x in TP if len(x["lv"]) == 2]
    s = two[int(rng.integers(len(two)))]
    R = int(rng.choice([4, 6, 8]))
    k = int(rng.choice([2, 4]))
    N = R * k
    obs_pig = s["out_on"] == "pig"
    wa, sa = [a for a, _ in w["lv"]], [a for a, _ in s["lv"]]
    cand = ["wrongerr", "nullp", "general"] + (["pseudo"] if obs_pig else []) + (["overclaim"] if R <= 6 else [])
    errs = pick_errors(rng, cand, exclusive=[{"wrongerr", "pseudo"}])
    p = p_sig(rng)
    pb = p_ns(rng) if "nullp" in errs else p_sig(rng)
    if "pseudo" in errs:
        test = f"Двухфакторный ANOVA по всем животным: {w['what']} {p_ru(p / 3)}, {s['what']} {p_ru(pb)}"
    elif "wrongerr" in errs:
        test = f"Двухфакторный ANOVA по {N} загонам как по независимым: {w['what']} {f_report(p / 3, 1, N - 4)}; {s['what']} {p_ru(pb)}"
    else:
        test = f"Смешанная модель с комнатой как случайным эффектом: {w['what']} {f_report(min(0.2, p * 4), 1, R - 2)}; {s['what']} {p_ru(pb)}"
    cl = []
    if "overclaim" in errs:
        cl.append(f"{w['what'].capitalize()} «{w['lv'][0][1]}» надёжно улучшает результат")
    if "nullp" in errs:
        cl.append(f"{s['what']} не влияет")
    elif not cl:
        cl.append(f"{s['lv'][best(s)][1]} {s['verb']}")
    cl = "; ".join(cl)
    concl = "«" + cl[0].upper() + cl[1:] + (" — на всех фермах»." if "general" in errs else "».")
    layout = []
    wname = dict(w["lv"])
    wl = list(rng.permutation(wa * (R // 2)))
    for i in range(R):
        layout.append((f"Комната {i + 1}: {wname[wl[i]]}", list(rng.permutation(sa * (k // 2)))))
    story = (f"<p><b>Цель.</b> Изучить вместе два фактора: «{w['what']}» ({wa[0]} — {w['lv'][0][1]}, {wa[1]} — {w['lv'][1][1]}) и «{s['what']}» ({codes_str(s['lv'])}).</p>"
             f"<p><b>Условия.</b> {nn(R, U['room']['forms'])} по {nn(k, U['pen']['forms'])}. {w['why'].capitalize()}.</p>"
             f"<p><b>Распределение.</b> Жребием половина комнат получила {wa[0]}, половина — {wa[1]}. Затем в каждой комнате жребием половина загонов получила {sa[0]}, половина — {sa[1]}.</p>"
             f"<p><b>Измерения.</b> {s['out'].capitalize()} — " + ("у каждого животного." if obs_pig else "по загону.") + "</p>"
             f"<div class='report'><b>Из отчёта.</b> {test}. Вывод: {concl}</div>")
    E = {"wrongerr": (f"{w['what'].capitalize()} проверили по {N} загонам", f"Ошибка для фактора комнат — разброс между комнатами: df = {R} − 2 = {R - 2}."),
         "overclaim": (f"О факторе «{w['what']}» сделан уверенный вывод", f"По {R // 2} комнаты на уровень — точность для этого фактора низкая."),
         "nullp": (f"По p > 0,05: «{s['what']} не влияет»", "Мало данных ≠ нет эффекта."),
         "general": ("Вывод сделан для всех ферм страны", "Совокупность — одна ферма."),
         "pseudo": ("Анализ сделан по всем животным", "Единицы — комнаты и загоны.")}
    nonerr = [("Два отдельных жребия: для комнат и для загонов", "Так и устроен split-plot."),
              ("Фактор комнат назначен комнатам целиком", "Иначе нельзя: " + w["why"] + "."),
              ("В каждой комнате оба варианта второго фактора", "Так второй фактор сравнивается внутри комнат — точнее.")]
    if not set(errs) & {"wrongerr", "pseudo"}:
        nonerr.append(("Комната в модели как случайный эффект", "Верный анализ split-plot."))
    qs = [mc(f"Что единица эксперимента для фактора «{w['what']}»?", f"Комната ({R})", [f"Загон ({N})", U[animal_of(s)]["cap"], "Корпус целиком"],
             f"{w['what'].capitalize()} назначали комнатам: повторностей {R}, по {R // 2} на уровень."),
          mc(f"А для фактора «{s['what']}»?", "Загон", ["Комната", U[animal_of(s)]["cap"], "Ферма"], "Этот фактор назначали загонам внутри комнаты."),
          design_q("split", rng), errors_q([E[e] for e in errs], nonerr, rng),
          mc("Какая модель правильная?", f"lmer({s['var']} ~ A * B + (1 | room), data = pens)",
             [f"aov({s['var']} ~ A * B, data = pens)", f"lmer({s['var']} ~ A * B + (1 | pen), data = pens)", f"t.test({s['var']} ~ A); t.test({s['var']} ~ B)"],
             "Комната — случайный эффект: две ошибки, между комнатами (для A) и внутри (для B). Классически: aov(y ~ A * B + Error(room))."),
          num(f"Сколько степеней свободы у ошибки для фактора «{w['what']}»?", R - 2, f"Комнат {R}, уровней 2: df = {R} − 2 = <b>{R - 2}</b>."),
          concl_q(kind_of(errs), rng, errs)]
    return dict(design="split", title=f"{w['what'].capitalize()} и {s['what']}", topic=f"{s['stage'].capitalize()} · комнаты", story=story,
                scheme=dict(title=f"{nn(R, U['room']['forms'])} × {nn(k, U['pen']['forms'])}: «{w['what']}» на комнату, «{s['what']}» на загон", rooms=layout,
                            legend={a: b for a, b in s["lv"]}), qs=qs)


def d_rep(rng):
    c = [x for x in TP if len(x["lv"]) == 2 and x["var"] == "adg"][int(rng.integers(4))]
    r = int(rng.integers(4, 11))
    N = 2 * r
    W = int(rng.integers(4, 9))
    codes = [a for a, _ in c["lv"]]
    cand = ["indep", "weekly", "nullp", "general"]
    errs = pick_errors(rng, cand, exclusive=[{"indep", "weekly"}, {"nullp", "weekly"}])
    p = p_ns(rng) if "nullp" in errs else p_sig(rng)
    if "indep" in errs:
        test = f"t-тест по всем {N * W} взвешиваниям загонов как по независимым: {t_report(p, N * W - 2)}"
    elif "weekly" in errs:
        wk = int(rng.integers(2, W + 1))
        test = f"Отдельный t-тест каждую неделю; на {wk}-й неделе из {W} p = 0,03, в остальные p > 0,05"
    else:
        test = f"Смешанная модель (загон — случайный эффект): вариант × неделя, вариант {p_ru(p)}"
    concl = (f"«{c['what'].capitalize()} не влияет»." if "nullp" in errs else
             f"«{c['lv'][best(c)][1].capitalize()} {c['verb']}" + (" на всех фермах»." if "general" in errs else "»."))
    story = (f"<p><b>Цель.</b> Сравнить варианты фактора «{c['what']}» ({c['stage']}): {codes_str(c['lv'])} — по динамике массы.</p>"
             f"<p><b>Условия.</b> {nn(N, U['pen']['forms'])}, жребием по {r} на вариант.</p>"
             f"<p><b>Измерения.</b> Каждую неделю, {nn(W, ("неделю", "недели", "недель"))} подряд, взвешивали загон целиком и считали среднюю массу.</p>"
             f"<div class='report'><b>Из отчёта.</b> {test}. Вывод: {concl}</div>")
    E = {"indep": (f"Все {N * W} взвешиваний взяты как независимые", f"Взвешивания одного загона связаны: независимых единиц {N}."),
         "weekly": ("Тест сделан отдельно для каждой недели", f"{nn(W, ('тест', 'теста', 'тестов'))} подряд — множественные сравнения; анализируют всю кривую сразу."),
         "nullp": ("По p > 0,05 сделан вывод «не влияет»", "Мало данных ≠ нет эффекта."),
         "general": ("Вывод сделан для всех ферм страны", "Совокупность — одна ферма.")}
    nonerr = [("Загоны распределили жребием", "Рандомизация — верно."),
              ("Взвешивали один и тот же загон каждую неделю", "Это суть повторных измерений, не ошибка."),
              ("Взвешивали весь загон, а не одну свинью", "Исход на уровне единицы эксперимента — верно.")]
    if not set(errs) & {"indep", "weekly"}:
        nonerr.append(("Загон — случайный эффект в модели", "Учитывает связь измерений одного загона."))
    qs = [mc("Что здесь единица эксперимента?", f"Загон ({N})", [f"Одно взвешивание ({N * W})", "Неделя", U[animal_of(c)]["cap"]],
             "Вариант назначали загону; взвешивания — повторные наблюдения той же единицы."),
          num("Сколько измерений приходится на один загон?", W, f"По одному в неделю, {nn(W, ('неделя', 'недели', 'недель'))}: <b>{W}</b>."),
          design_q("rep", rng), errors_q([E[e] for e in errs], nonerr, rng),
          mc("Какая модель правильная?", "lmer(weight ~ trt * week + (1 | pen), data = d)",
             [f"t-тест по всем {N * W} взвешиваниям", "Отдельный t-тест каждую неделю", "aov(weight ~ trt, data = d) по всем взвешиваниям"],
             "Загон — случайный эффект; trt × week показывает, расходятся ли кривые. Простой честный вариант — одно число на загон (например, наклон или итоговый привес) и t-тест."),
          num("Сколько единиц эксперимента на каждый вариант?", r, f"<b>{r}</b> загонов на вариант — по ним считается точность сравнения."),
          concl_q(kind_of(errs), rng, errs)]
    return dict(design="rep", title=f"{c['what'].capitalize()}: масса по неделям", topic=f"{c['stage'].capitalize()} · загоны", story=story,
                scheme=None, qs=qs)


def d_cross(rng):
    c = TX[int(rng.integers(len(TX)))]
    A = U[c["unit"]]
    n = int(rng.integers(8, 21))
    codes = [a for a, _ in c["lv"]]
    cand = ["nowash", "allAB", "unpaired", "nullp", "general"]
    errs = pick_errors(rng, cand, exclusive=[{"nullp"}])
    seq = (f"Все {nn(n, A['forms'])} сначала получили {codes[0]}, потом {codes[1]}." if "allAB" in errs
           else f"Порядок ({codes[0]}→{codes[1]} или {codes[1]}→{codes[0]}) для каждой единицы выбрали жребием, поровну.")
    wash = ("Второй вариант начали сразу на следующий день." if "nowash" in errs else f"Между периодами — {c['wash']}.")
    p = p_ns(rng) if "nullp" in errs else p_sig(rng)
    test = (f"Двухвыборочный t-тест {codes[0]} против {codes[1]}: {t_report(p, 2 * n - 2)}" if "unpaired" in errs
            else f"Парный t-тест (каждая единица — сама себе контроль): {t_report(p, n - 1)}")
    concl = (f"«{c['what'].capitalize()}: разницы нет»." if "nullp" in errs else
             f"«{c['lv'][0][1].capitalize()} лучше" + (" для всех ферм»." if "general" in errs else "»."))
    story = (f"<p><b>Цель.</b> Сравнить варианты фактора «{c['what']}»: {codes_str(c['lv'])}.</p><p><b>Условия.</b> {c['where'].capitalize()}, {nn(n, A['forms'])}. Каждая единица получает оба варианта по очереди; один период длится {c['period']}.</p>"
             f"<p><b>Распределение.</b> {seq} {wash}</p><p><b>Измерения.</b> {c['out'].capitalize()} ({c['unit_y']}) в конце каждого периода.</p>"
             f"<div class='report'><b>Из отчёта.</b> {test}. Вывод: {concl}</div>")
    E = {"nowash": ("Второй вариант начали сразу после первого", "Эффект первого варианта переносится во второй период (последействие)."),
         "allAB": ("Все единицы получили варианты в одном порядке", "Вариант совпал с периодом: эффект времени не отделить."),
         "unpaired": ("Использовали двухвыборочный t-тест", "Каждая единица дала оба значения — нужен парный анализ."),
         "nullp": ("По p > 0,05 сделан вывод «разницы нет»", "Мало данных ≠ нет эффекта."),
         "general": ("Вывод сделан для всех ферм страны", "Совокупность — одна ферма.")}
    nonerr = [("Каждая единица получила оба варианта", "Суть перекрёстного плана."),
              ("Периоды одинаковой длины", "Так и должно быть.")]
    if "allAB" not in errs:
        nonerr.append(("Порядок вариантов выбрали жребием", "Рандомизация последовательностей — верно."))
    if "nowash" not in errs:
        nonerr.append(("Между периодами был отмывочный период", "Защищает от последействия."))
    if "unpaired" not in errs:
        nonerr.append(("Использовали парный анализ", "Верно: каждая единица — сама себе контроль."))
    qs = [mc("Что здесь единица эксперимента?", f"{A['cap']} в одном периоде", [f"{A['cap']} целиком", "Период", "Последовательность вариантов"],
             f"Вариант назначают на один период, и каждая единица ({A['nom']}) даёт два наблюдения. Поэтому {A['nom']} здесь — блок, а единица эксперимента — {A['nom']} в одном периоде."),
          mc("Зачем нужен отмывочный период?", "Чтобы эффект первого варианта не переносился во второй",
             ["Чтобы животные отдохнули от взвешиваний и стресса", "Чтобы получить больше наблюдений на каждую единицу", "Чтобы уравнять животных по массе перед вторым периодом"],
             "Последействие (carryover) — главный риск перекрёстного плана."),
          design_q("cross", rng), errors_q([E[e] for e in errs], nonerr, rng),
          mc("Какая модель правильная?", f"lmer(y ~ trt + period + (1 | {c['unit']}), data = d)",
             [f"t.test(y ~ trt, data = d) как две независимые группы", "aov(y ~ trt, data = d)", "chisq.test по числу улучшений"],
             "Или парный t-тест по разностям (если эффект периода мал). Период в модели защищает от эффекта времени."),
          num("Сколько степеней свободы у парного t-теста?", n - 1, f"Единиц {n}: df = {n} − 1 = <b>{n - 1}</b>."),
          concl_q(kind_of(errs), rng, errs)]
    return dict(design="cross", title=c["what"].capitalize(), topic=c["where"].capitalize(), story=story, scheme=None, qs=qs)


def d_ba(rng):
    c = BEFORE[int(rng.integers(len(BEFORE)))]
    x1 = float(rng.uniform(*c["before"]))
    x2 = x1 * (1 - c["after_drop"] * rng.uniform(0.6, 1.4))
    d = 0 if x1 >= 100 else 1
    x1, x2 = round(x1, d), round(x2, d)
    errs = pick_errors(rng, ["season", "rtm", "causal", "general"], exclusive=[{"season", "rtm"}], kmax=2)
    errs = ["nocontrol"] + errs
    if "season" in errs:
        per = "«до» — октябрь–декабрь прошлого года, «после» — январь–март этого года"
    elif "rtm" in errs:
        per = "«до» — весь прошлый год, самый плохой за пять лет (из-за него и решили что-то менять), «после» — этот год"
    else:
        per = "«до» — январь–март прошлого года, «после» — январь–март этого года"
    u = (" " + c["unit_y"]) if c["unit_y"] else ""
    better = (x2 < x1) if c["after_drop"] > 0 else (x2 > x1)
    story = (f"<p><b>Что сделали.</b> На ферме {c['change']}.</p>"
             f"<p><b>Сравнение.</b> {c['out'].capitalize()}: {per}.</p>"
             f"<p><b>Результат.</b> Было {num_ru(x1, d)}{u}, стало {num_ru(x2, d)}{u}.</p>"
             f"<div class='report'><b>Из отчёта.</b> " + ("«Изменение вызвало " if "causal" in errs else "«После внедрения произошло ")
             + ("улучшение" if better else "изменение") + (" — рекомендуем всем фермам»." if "general" in errs else "».") + "</div>")
    E = {"nocontrol": ("Сравнивали только с периодом «до»", "Всё, что изменилось между периодами (сезон, корм, болезни), смешано с изменением."),
         "season": ("«До» и «после» — разные месяцы года", "Падёж и привес сильно зависят от сезона."),
         "rtm": ("«До» — самый плохой год за пять лет", "Регрессия к среднему: после худшего периода показатель обычно улучшается и сам."),
         "causal": ("Отчёт говорит, что изменение «вызвало»", "До/после без контроля даёт только совпадение во времени."),
         "general": ("Вывод сделан для всех ферм страны", "Совокупность — одна ферма.")}
    nonerr = [("Показатель считали одинаково до и после", "Это правильно."),
              ("Сравнивали одну и ту же ферму", "Это суть плана, а не ошибка сама по себе."),
              ("Записали значения до и после", "Без этого не было бы и сравнения.")]
    if "season" not in errs:
        nonerr.append(("Периоды «до» и «после» одинаковой длины", "Не ошибка."))
    rel = (x2 - x1) / x1 * 100
    qs = [mc("Есть ли здесь контрольная группа?", "Нет: ту же ферму сравнивают до и после", ["Да: период «до» и есть контроль", "Да: другие фермы региона за тот же год", "Да: половина корпусов осталась без изменений"],
             "Период «до» — не контроль: он отличается не только изменением, но и временем."),
          mc("Что ещё изменилось вместе с вмешательством?", "Время: сезон, поставки, болезни",
             ["Ничего: ферма та же самая", "Только число животных в корпусах", "Только сам вариант обработки"],
             "Между периодами меняется всё, что зависит от времени. Это главный изъян плана до/после."),
          design_q("ba", rng), errors_q([E[e] for e in errs], nonerr, rng),
          mc("Как улучшить план?", "Параллельный контроль по жребию",
             ["Взять периоды «до» и «после» подлиннее", "Посчитать p-value", "Сравнить показатель с нормативом по отрасли"],
             "Часть корпусов или ферм оставить без изменения, выбрав их жребием. Только одновременный контроль отделяет эффект изменения от эффекта времени."),
          num("На сколько процентов изменился показатель: (стало − было) / было × 100? (снижение — с минусом)", round(rel, 1),
              f"({num_ru(x2, d)} − {num_ru(x1, d)}) / {num_ru(x1, d)} × 100 = <b>{num_ru(rel, 1)} %</b>.", tol=0.6),
          concl_q("confounded", rng, errs, avoid={"nothing"})]
    return dict(design="ba", title=c["change"][0].upper() + c["change"][1:], topic="Ферма · производство", story=story, scheme=None, qs=qs)


CONF = {"farm": "фермы первой группы крупнее и стоят в районе, где много свиноферм", "sow": "свиноматки первой группы чаще стоят в старом корпусе",
        "piglet": "поросята первой группы чаще из больших гнёзд", "pen": "загоны первой группы чаще стоят в старом корпусе"}


def grp(A, label, n):
    return f"«{label}» ({nn(n, A['forms'])})"


def d_cohort(rng):
    c = OBS[int(rng.integers(len(OBS)))]
    A = U[c["unit"]]
    n1, n2 = int(rng.integers(40, 300)), int(rng.integers(40, 300))
    p1, p2 = rng.uniform(0.15, 0.45), rng.uniform(0.05, 0.3)
    a, cc = max(1, int(n1 * p1)), max(1, int(n2 * p2))
    errs = pick_errors(rng, ["causal", "confound", "general", "lost"])
    rr = (a / n1) / (cc / n2)
    lost = int(n1 * 0.2)
    story = (f"<p><b>Вопрос.</b> Связан ли признак «{c['exp'][0]}» с исходом «{c['out']}»?</p>"
             f"<p><b>Как собирали.</b> Взяли две группы: {grp(A, c['exp'][0], n1)} и {grp(A, c['exp'][1], n2)} — и наблюдали вперёд, пока не стало ясно, был ли исход. Признак никто не назначал.</p>"
             + (f"<p>{CONF[c['unit']][0].upper() + CONF[c['unit']][1:]}; в анализе это не учитывали.</p>" if "confound" in errs else "")
             + (f"<p>Из первой группы {lost} выбыли из наблюдения; их просто убрали из таблицы.</p>" if "lost" in errs else "")
             + f"<p><b>Результат.</b> Исход: {a} из {n1 - (lost if 'lost' in errs else 0)} в первой группе и {cc} из {n2} во второй.</p>"
             f"<div class='report'><b>Из отчёта.</b> χ²-тест: {p_ru(p_sig(rng))}. Вывод: «"
             + ("Признак вызывает этот исход" if "causal" in errs else "Признак связан с этим исходом")
             + (" на всех фермах страны»." if "general" in errs else "».") + "</div>")
    if "lost" in errs:
        n1 = n1 - lost
        rr = (a / n1) / (cc / n2)
    E = {"causal": ("Отчёт говорит: признак «вызывает» исход", "Признак никто не назначал — только связь."),
         "confound": ("Различия групп по другим признакам не учли", "Нужна модель с конфаундерами: glm(y ~ x + конфаундеры, family = binomial)."),
         "general": ("Вывод сделан для всей страны", "Совокупность — та, из которой брали."),
         "lost": ("Выбывших убрали из таблицы", "Если выбывали из-за исхода или близко к нему, доли смещены.")}
    nonerr = [("Группы набрали до того, как исход стал известен", "Так устроено когортное исследование."),
              ("Исход учитывали одинаково в обеих группах", "Правильно."), ("Сравнили доли χ²-тестом", "Для таблицы 2×2 это уместно."),
              ("Группы разного размера", "Это не ошибка: доли считаются в каждой группе отдельно.")]
    qs = [obs_q("признак"),
          mc("Что здесь можно оценить?", "Риск в каждой группе, RR и OR", ["Только OR, как в случай-контроль", "Только средние в группах", "Ничего: нет контроля"],
             "Группы набраны по фактору, исход наступал сам — доли честные, поэтому RR считать можно."),
          design_q("cohort", rng), errors_q([E[e] for e in errs], nonerr, rng),
          mc("Как правильно анализировать?", "chisq.test или glm(y ~ x, family = binomial)",
             ["t.test(y ~ x, var.equal = TRUE) по исходу 0/1", "Парный t-тест: t.test(x, y, paired = TRUE)", "aov(y ~ x, data = d) по долям в группах"],
             "Исход да/нет: доли, OR/RR, χ²; конфаундеры добавляют в логистическую регрессию: <code>glm(y ~ x + z, family = binomial)</code>."),
          num("Относительный риск RR (доля исхода в первой группе / доля во второй)?", round(rr, 2),
              f"({a}/{n1}) / ({cc}/{n2}) = {num_ru(a / n1, 3)} / {num_ru(cc / n2, 3)} = <b>{num_ru(rr, 2)}</b>.", tol=0.02),
          concl_q("assoc", rng, errs)]
    return dict(design="cohort", title=f"{c['exp'][0][0].upper() + c['exp'][0][1:]} → {c['ev']}", topic=f"Здоровье стада · {A['nom']}", story=story, scheme=None, qs=qs)


def d_cc(rng):
    c = CASECTRL[int(rng.integers(len(CASECTRL)))]
    A = U[c["unit"]]
    n1, n2 = int(rng.integers(30, 150)), int(rng.integers(60, 300))
    a, cc = int(n1 * rng.uniform(0.3, 0.7)), int(n2 * rng.uniform(0.1, 0.45))
    a, cc = max(1, min(a, n1 - 1)), max(1, min(cc, n2 - 1))
    errs = pick_errors(rng, ["risk", "causal", "recall", "ctrlsel"])
    OR = (a / (n1 - a)) / (cc / (n2 - cc))
    story = (f"<p><b>Вопрос.</b> Связан ли фактор «{c['exp']}» с проблемой?</p>"
             f"<p><b>Как собирали.</b> Случаи — {c['case']}: {n1}. Контроли — {c['ctrl']}: {n2}" + (", их взяли из другого региона" if "ctrlsel" in errs else ", из того же района") + ". "
             "Затем выясняли, был ли фактор в прошлом" + (": случаи опрашивали подробно на месте, контроли — коротко по телефону." if "recall" in errs else ", одной и той же анкетой.") + "</p>"
             f"<p><b>Результат.</b> Фактор был у {a} из {n1} случаев и у {cc} из {n2} контролей.</p>"
             f"<div class='report'><b>Из отчёта.</b> " + (f"«Риск проблемы при наличии фактора — {num_ru(a / (a + cc) * 100, 0)} %»; " if "risk" in errs else "")
             + f"OR = {num_ru(OR, 2)}, {p_ru(p_sig(rng))}. Вывод: «" + ("Фактор вызывает проблему»." if "causal" in errs else "Фактор связан с проблемой».") + "</div>")
    E = {"risk": ("В отчёте посчитан риск проблемы в %", "Число случаев в выборке задал исследователь — риск не оценить, только OR."),
         "causal": ("Отчёт говорит: фактор «вызывает» проблему", "Фактор никто не назначал — только связь."),
         "recall": ("Случаи и контроли опрашивали по-разному", "Смещение опроса — у случаев фактор найдут чаще, потому что ищут тщательнее."),
         "ctrlsel": ("Контроли взяли из другого региона", "Контроли должны представлять ту же совокупность, из которой пришли случаи.")}
    nonerr = [("Набрали по исходу: случаи и контроли", "Так устроен план случай-контроль."), ("Посчитали отношение шансов", "Единственная мера связи, которую здесь можно."),
              ("Контролей больше, чем случаев", "Это нормально и даже повышает точность.")]
    if "recall" not in errs:
        nonerr.append(("Фактор выясняли одинаковой анкетой", "Правильно."))
    if "ctrlsel" not in errs:
        nonerr.append(("Контроли из того же района", "Правильно."))
    qs = [obs_q("фактор"),
          mc("Какую меру связи можно оценить?", "Только отношение шансов (OR)", ["Риск проблемы в каждой группе", "Относительный риск RR и OR", "Среднее время до появления проблемы"],
             "Долю случаев в выборке задал исследователь, поэтому вероятность проблемы и RR не оценить. OR не зависит от того, сколько контролей набрали."),
          design_q("cc", rng), errors_q([E[e] for e in errs], nonerr, rng),
          mc("Как правильно анализировать?", "OR с ДИ и chisq.test или fisher.test", ["RR с ДИ и двухвыборочный t-тест", "Парный t-тест: t.test(x, y, paired = TRUE)", "aov по числу случаев в группах"],
             "fisher.test — когда частоты малы. Для конфаундеров — логистическая регрессия: её OR тоже корректны в случай-контроль."),
          num("Отношение шансов OR (шансы фактора у случаев / у контролей)?", round(OR, 2),
              f"({a}/{n1 - a}) / ({cc}/{n2 - cc}) = <b>{num_ru(OR, 2)}</b>.", tol=0.02),
          concl_q("assoc", rng, errs)]
    return dict(design="cc", title=f"{c['case'][0].upper() + c['case'][1:]}: {c['short']}", topic=f"Здоровье стада · {A['nom']}", story=story, scheme=None, qs=qs)


def d_xs(rng):
    import re
    c = CROSS[int(rng.integers(len(CROSS)))]
    A = U[c["unit"]]
    tot = int(re.search(r"\d+", c["what"]).group())
    n1 = int(tot * rng.uniform(0.3, 0.7))
    n2 = tot - n1
    a, cc = int(n1 * rng.uniform(0.1, 0.5)), int(n2 * rng.uniform(0.1, 0.5))
    errs = pick_errors(rng, ["causal", "selection", "general"])
    prev = (a + cc) / tot * 100
    sel = {"farm": "участвовали только фермы, которые сами согласились", "sow": "только в ближнем к лаборатории корпусе",
           "pig": "только туши утренней смены, где забивают свиней с ближайших ферм"}[c["unit"]]
    story = (f"<p><b>Как собирали.</b> {c['what'][0].upper() + c['what'][1:]}" + (f": {sel}" if "selection" in errs else ", выборка случайная") + ". "
             f"Для каждой единицы сразу записали два признака: «{c['exp']}» (да/нет) и «{c['out']}» (да/нет).</p>"
             f"<p><b>Результат.</b> Среди «{c['exp']}» исход у {a} из {n1}; среди остальных — у {cc} из {n2}.</p>"
             f"<div class='report'><b>Из отчёта.</b> χ²-тест: {p_ru(p_sig(rng))}. Вывод: «" + ("Признак вызывает этот исход" if "causal" in errs else "Признак связан с этим исходом")
             + (" во всей стране»." if "general" in errs else "».") + "</div>")
    E = {"causal": ("Отчёт говорит: признак «вызывает» исход", "В один момент не ясно даже, что было раньше — фактор или исход."),
         "selection": ("В выборку попали согласившиеся или удобные", "Такие единицы могут отличаться от остальных, и распространённость сместится."),
         "general": ("Вывод сделан для всей страны", "Совокупность — та, из которой брали.")}
    nonerr = [("Фактор и исход записали одновременно", "Так устроен срез."), ("Сравнили доли χ²-тестом", "Для 2×2 уместно."),
              ("Записали фактор и исход у каждой единицы", "Правильно."), ("Группы по фактору разного размера", "В срезе так почти всегда, это не ошибка.")]
    if "selection" not in errs:
        nonerr.append(("Выборку взяли случайно", "Правильно: так распространённость оценивается без смещения."))
    qs = [obs_q("признак"),
          mc("Что даёт такой план?", "Распространённость и связь на один момент",
             ["Причинный эффект фактора", "Риск заболеть за следующий год в каждой группе", "Только OR, как в исследовании случай-контроль"],
             "Распространённость — доля с исходом сейчас. Связь — OR или χ²."),
          design_q("xs", rng), errors_q([E[e] for e in errs], nonerr, rng),
          mc("Как правильно анализировать?", "chisq.test по таблице 2×2 или glm(binomial)",
             ["t.test(y ~ x, var.equal = TRUE) по исходу 0/1", "lmer(y ~ x + (1 | pen)) по загонам", "Парный t-тест: t.test(x, y, paired = TRUE)"],
             "Исход да/нет — доли, OR, χ²; конфаундеры учитывает логистическая регрессия."),
          num("Общая распространённость исхода, %?", round(prev, 1), f"({a} + {cc}) / {tot} × 100 = <b>{num_ru(prev, 1)} %</b>.", tol=0.3),
          concl_q("assoc", rng, errs)]
    xt = {"сальмонелла": "Биобезопасность и сальмонелла", "антитела": "Номер опороса и антитела к PRRS", "пневмония": "Размер фермы и пневмония"}
    return dict(design="xs", title=xt[c["ev"]], topic=f"Здоровье стада · {A['nom']}", story=story, scheme=None, qs=qs)


PLAN = [(d_crd_pen, 125), (d_crd_ind, 50), (d_rcbd, 115), (d_pair, 60), (d_latin, 60), (d_fact, 90), (d_split, 90),
        (d_rep, 80), (d_cross, 77), (d_ba, 60), (d_cohort, 70), (d_cc, 70), (d_xs, 50)]
