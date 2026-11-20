import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
import html as _h, json, re

OUT = json.load(open(HERE + '/outputs.json'))
OUT.update(json.load(open(HERE + '/exam_outputs.json')))
V = json.load(open(HERE + '/values.json'))
EV = json.load(open(HERE + '/exam_values.json'))

JS_EXPL = {}      # id -> {key: [title, html, [related]]}
TRAINERS = []     # ids


def ru(x, d=2):
    s = f'{x:.{d}f}'
    return s.replace('-', '−').replace('.', ',')


def en(x, d=2):
    return f'{x:.{d}f}'


def esc(s):
    return _h.escape(s, quote=False)


def pre(key_or_text, cls=''):
    t = OUT.get(key_or_text, key_or_text)
    c = f' class="{cls}"' if cls else ''
    return f'<pre{c}>{esc(t)}</pre>'


def rclick(rid, key, marks, expl, title='Нажимай на подсвеченные числа — справа откроется объяснение'):
    """marks: list of (substring, k). Each substring is searched after the previous match."""
    text = OUT[key]
    pos = 0
    parts = []
    for sub, k in marks:
        i = text.find(sub, pos)
        if i < 0:
            raise ValueError(f'{rid}: cannot find {sub!r} after {pos}')
        parts.append(esc(text[pos:i]))
        parts.append(f'<span class="rk" data-k="{k}">{esc(sub)}</span>')
        pos = i + len(sub)
    parts.append(esc(text[pos:]))
    JS_EXPL[rid] = expl
    return (f'<h4>{title}</h4>\n<div class="rwrap" data-rid="{rid}">\n<pre class="rout">{"".join(parts)}</pre>\n'
            f'<div class="rexp"><span class="lbl">Объяснение</span><div class="rexpBody">Нажми на любое подсвеченное место в выводе.</div></div>\n</div>\n'
            f'<details><summary>Все объяснения списком</summary><div class="body rexpAll" data-rid="{rid}"></div></details>\n')


def box(kind, label, inner):
    return f'<div class="{kind} box"><span class="lbl">{label}</span>{inner}</div>\n'


def idea(inner, label='Суть'):
    return box('idea', label, inner)


def mini(inner, label='Посчитай сам'):
    return box('mini', label, inner)


def remember(inner, label='Запомни'):
    return box('remember', label, inner)


def analogy(inner, label='Аналогия'):
    return box('analogy', label, inner)


def vet(inner, label='На ферме'):
    return f'<div class="vet"><span class="lbl">{label}</span>{inner}</div>\n'


def trap(inner, label='Ловушка'):
    return f'<div class="trap"><span class="lbl">{label}</span>{inner}</div>\n'


def exam(inner, label='На экзамене (по-английски)'):
    return f'<div class="exam"><span class="lbl">{label}</span><div class="en">{inner}</div></div>\n'


def steps(items):
    return '<ol class="steps">' + ''.join(f'<li>{x}</li>' for x in items) + '</ol>\n'


def table(headers, rows, cls=''):
    th = ''.join(f'<th>{h}</th>' for h in headers)
    trs = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f'<div class="tbl"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>\n'


def qa(items, start=1):
    out = []
    for i, (q, a) in enumerate(items, start):
        out.append(f'<details class="q"><summary><span class="qnum">{i}</span>{q}</summary><div class="body">{a}</div></details>')
    return '\n'.join(out) + '\n'


def checklist(day, items):
    lis = ''.join(f'<li><input type="checkbox" id="c{day}_{i}"><label for="c{day}_{i}">{t}</label><span class="t">{m}</span></li>' for i, (t, m) in enumerate(items))
    return f'<ul class="check">{lis}</ul>\n'


def calc(s):
    return f'<span class="calc">{s}</span>'


def calci(s):
    return f'<span class="calci">{s}</span>'


# ---------- trainer ----------
def trainer(tid, title, intro, blocks):
    """blocks: list of (h4, intro_html, questions). question dicts:
       num: dict(k='num', q, ans, tol, unit, hint, sol)
       mc:  dict(k='mc', q, opts=[(text, ok, why)])
       txt: dict(k='txt', q, chk=[...], model)"""
    TRAINERS.append(tid)
    n = 0
    out = [f'<h3>{title}</h3>\n<div class="trainer" id="{tid}">\n{intro}\n<div class="trhead"><span>Верно: <b class="trscore">0</b></span><button class="btn trreset" type="button">Начать заново</button></div>']
    for h4, bi, qs in blocks:
        out.append(f'<h4>{h4}</h4>')
        if bi:
            out.append(bi)
        for q in qs:
            n += 1
            qid = f'{tid}_{n}'
            if q['k'] == 'num':
                unit = f'<span class="unit">{q.get("unit","")}</span>' if q.get('unit') else ''
                out.append(f'<div class="tq" id="{qid}" data-kind="num" data-ans="{q["ans"]}" data-tol="{q["tol"]}"{" data-abs=\"1\"" if q.get("abs") else ""}>'
                           f'<div class="qh"><span class="tn">{n}</span>{q["q"]}</div>'
                           f'<div class="row"><input class="ans" type="text" inputmode="decimal" autocomplete="off" aria-label="Ответ к заданию {n}">{unit}'
                           f'<button class="btn chk" type="button">Проверить</button><span class="res"></span><button class="btn ask" type="button">Не понял — спросить Claude</button><span class="cpok"></span></div>'
                           f'<div class="fb hint">Подсказка: {q["hint"]}</div>'
                           f'<details class="sol"><summary>Решение</summary><div class="body">{q["sol"]}</div></details></div>')
            elif q['k'] == 'mc':
                opts = ''.join(f'<button class="opt" type="button" data-c="{1 if ok else 0}" data-why="{_h.escape(why, quote=True)}">{t}</button>' for t, ok, why in q['opts'])
                out.append(f'<div class="tq" id="{qid}" data-kind="mc"><div class="qh"><span class="tn">{n}</span>{q["q"]}</div>'
                           f'<div class="opts">{opts}</div><div class="fb why"></div>'
                           f'<div class="row"><button class="btn ask" type="button">Не понял — спросить Claude</button><span class="cpok"></span></div></div>')
            else:
                chk = ''.join(f'<li>{c}</li>' for c in q['chk'])
                out.append(f'<div class="tq" id="{qid}" data-kind="txt"><div class="qh"><span class="tn">{n}</span>{q["q"]}</div>'
                           f'<textarea placeholder="Напиши ответ по-английски, как на экзамене"></textarea>'
                           f'<div class="row"><button class="btn cp" type="button">Скопировать ответ для проверки в чате</button><span class="cpok"></span></div>'
                           f'<details class="sol"><summary>Образец и что проверяющий ищет</summary><div class="body">{q["model"]}<p><b>Что должно быть в ответе:</b></p><ul class="chk">{chk}</ul></div></details></div>')
    out.append('</div>\n')
    return '\n'.join(out)


def N(q, ans, tol, hint, sol, unit='', abs_=False):
    return dict(k='num', q=q, ans=ans, tol=tol, hint=hint, sol=sol, unit=unit, abs=abs_)


def M(q, *opts):
    return dict(k='mc', q=q, opts=list(opts))


def T(q, model, chk):
    return dict(k='txt', q=q, model=model, chk=chk)
