import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
import json, re, sys
sys.path.insert(0, HERE)
from lib import JS_EXPL, TRAINERS
import w1, w1b, w2, w2b, w3, exam, ref

css_main = open(HERE + '/style_main.css', encoding='utf-8').read()
css_blocks = open(HERE + '/style_blocks.css', encoding='utf-8').read()
tr_css = open(HERE + '/style_trainer.css', encoding='utf-8').read().replace('#tr4', '.trainer')
APP = '--app' in sys.argv   # build for the R-trainer PWA (exam3.html): back link + manifest

extra_css = '''
.trainer .trscore{font-variant-numeric:tabular-nums}
.rwrap .rout{max-height:none}
details.eq > summary{font-weight:500}
details.eq .en{background:var(--exam-bg);border-radius:8px;padding:6px 10px}
.timer .clock{min-width:96px}
.appback{display:inline-flex;align-items:center;gap:6px;margin:0 0 10px;font-size:14px;font-weight:600;text-decoration:none;color:var(--accent);border:1px solid var(--line);background:var(--surface);border-radius:10px;padding:5px 10px}
.weekbar{display:inline-block;font-size:12px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;background:var(--accent);color:#fff;border-radius:6px;padding:2px 8px;margin-right:6px}
ol li{margin:3px 0}
.grid2 > div{min-width:0}
figure > .chart{max-width:620px;margin:0 auto}
p code, li code, td code{overflow-wrap:anywhere}
'''

sections = [w1.start(), w1.day1(), w1.day2(), w1.day3(), w1.day4(), w1b.day5(), w1b.day6(),
            w2.day8(), w2.day9(), w2.day10(), w2b.day11(), w2b.day12(), w2b.day13(), w2b.day14(),
            w3.day15(), w3.day16(), w3.day17(), w3.day18(), exam.mockA(), exam.mockB(), w3.day21(),
            ref.atlas(), ref.formulas(), ref.traps(), ref.gloss(), ref.sources()]

nav = [('plan', 'Старт'), ('d1', '1 · Прямая'), ('d2', '2 · SE, t, ДИ'), ('d3', '3 · Прогноз'), ('d4', '4 · Логарифмы'), ('d5', '5 · Допущения'), ('d6', '6 · Практика'),
       ('d8', '8 · Логистическая'), ('d9', '9 · LRT, Пуассон'), ('d10', '10 · Практика'), ('d11', '11 · Множественная'), ('d12', '12 · F, Type I/III'), ('d13', '13 · Группы'), ('d14', '14 · Тренажёр'),
       ('d15', '15 · Кривые'), ('d16', '16 · Диагностика'), ('d17', '17 · AIC'), ('d18', '18 · Практика'), ('d19', '19 · Пробник A'), ('d20', '20 · Пробник B'), ('d21', '21 · Шпаргалка'),
       ('atlas', 'Атлас R'), ('cheat', 'Формулы'), ('traps', 'Ловушки'), ('gloss', 'Словарик'), ('src', 'Источники')]

js = r'''
<script>
(function(){
  var store = {get:function(k){try{return localStorage.getItem(k);}catch(e){return null;}}, set:function(k,v){try{localStorage.setItem(k,v);}catch(e){}}};
  var EXPL = __EXPL__;
  // clickable R outputs
  document.querySelectorAll('.rwrap').forEach(function(w){
    var rid = w.dataset.rid, E = EXPL[rid] || {}, spans = w.querySelectorAll('.rk'), body = w.querySelector('.rexpBody');
    function show(k){
      var e = E[k]; if(!e) return;
      spans.forEach(function(s){ s.classList.remove('on','rel'); });
      spans.forEach(function(s){ if(s.dataset.k===k) s.classList.add('on'); if(e[2].indexOf(s.dataset.k)>=0) s.classList.add('rel'); });
      body.innerHTML = '<b>'+e[0]+'</b><br>'+e[1];
    }
    spans.forEach(function(s){ s.setAttribute('tabindex','0'); s.setAttribute('role','button');
      s.addEventListener('click', function(){ show(s.dataset.k); });
      s.addEventListener('keydown', function(ev){ if(ev.key==='Enter'||ev.key===' '){ ev.preventDefault(); show(s.dataset.k); } }); });
  });
  document.querySelectorAll('.rexpAll').forEach(function(all){
    var E = EXPL[all.dataset.rid] || {}, h = '<ul>';
    Object.keys(E).forEach(function(k){ h += '<li><b>'+E[k][0]+'.</b> '+E[k][1]+'</li>'; });
    all.innerHTML = h + '</ul>';
  });

  // progress
  var KEY = "stat5101-exam3-progress-v1";
  var boxes = Array.prototype.slice.call(document.querySelectorAll('ul.check input[type=checkbox]'));
  var saved = {}; try { saved = JSON.parse(store.get(KEY) || "{}"); } catch(e){ saved = {}; }
  function update(){
    var done = boxes.filter(function(b){ return b.checked; }).length;
    document.getElementById('ptext').textContent = done + " из " + boxes.length;
    document.getElementById('pbar').style.width = (boxes.length ? 100*done/boxes.length : 0) + "%";
  }
  boxes.forEach(function(b){ if (saved[b.id]) b.checked = true;
    b.addEventListener('change', function(){ saved[b.id] = b.checked; store.set(KEY, JSON.stringify(saved)); update(); }); });
  document.getElementById('resetProg').addEventListener('click', function(){ boxes.forEach(function(b){ b.checked = false; }); saved = {}; store.set(KEY, "{}"); update(); });
  update();

  // trainers
  function parse(v){ v = String(v).replace(/−/g,'-').replace(/\s/g,'').replace(/,/g,'.').replace(/[^0-9.\-]/g,''); return parseFloat(v); }
  function copyText(text, ok){
    function done(){ ok.textContent = 'Скопировано — вставь в чат'; setTimeout(function(){ ok.textContent=''; }, 4000); }
    function fallback(){ try { var t=document.createElement('textarea'); t.value=text; document.body.appendChild(t); t.select(); document.execCommand('copy'); document.body.removeChild(t); done(); } catch(e){ ok.textContent='Не удалось скопировать — выдели текст вручную'; } }
    try { if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(done, fallback); else fallback(); } catch(e){ fallback(); }
  }
  document.querySelectorAll('.trainer').forEach(function(root){
    var tid = root.id, TK = "stat5101-exam3-"+tid, st = {};
    try { st = JSON.parse(store.get(TK) || "{}") || {}; } catch(e){ st = {}; }
    function save(){ store.set(TK, JSON.stringify(st)); }
    var title = (root.previousElementSibling && root.previousElementSibling.textContent || tid).split(':')[0];
    var auto = [].slice.call(root.querySelectorAll('.tq[data-kind=num], .tq[data-kind=mc]'));
    var scoreEl = root.querySelector('.trscore');
    function score(){ var ok=0, first=0; auto.forEach(function(t){ var s=st[t.id]; if(s&&s.ok){ ok++; if(s.tries===1) first++; } });
      scoreEl.textContent = ok + " из " + auto.length + " (с первой попытки: " + first + ")"; }
    function rec(t){ return st[t.id] || (st[t.id] = {tries:0, ok:false}); }
    function qtext(t){ return t.querySelector('.qh').textContent.replace(/^\d+/, '').trim(); }
    root.querySelectorAll('.tq .ask').forEach(function(b){
      var t = b.closest('.tq'), ok = b.parentNode.querySelector('.cpok'), n = t.querySelector('.tn').textContent;
      b.addEventListener('click', function(){ var mine=''; var inp=t.querySelector('input.ans'); if(inp) mine=inp.value.trim(); if(t.dataset.picked) mine=t.dataset.picked;
        copyText(title + ", задание " + n + ". " + qtext(t) + "\nМой ответ: " + (mine||"—") + "\nНе понял, объясни.", ok); });
    });
    root.querySelectorAll('.tq[data-kind=num]').forEach(function(t){
      var inp=t.querySelector('input.ans'), res=t.querySelector('.res'), hint=t.querySelector('.hint'), sol=t.querySelector('details.sol');
      var ans=parseFloat(t.dataset.ans), tol=parseFloat(t.dataset.tol), useAbs=t.dataset.abs==="1";
      function check(){ var x=parse(inp.value); if(isNaN(x)){ res.className='res bad'; res.textContent='Впиши число'; return; }
        if(useAbs) x=Math.abs(x); var r=rec(t); if(!r.ok) r.tries++; r.val=inp.value;
        if(Math.abs(x-ans) <= tol+1e-9){ r.ok=true; res.className='res ok'; res.textContent='✓ Верно'; t.classList.add('done'); hint.classList.remove('on'); sol.open=true; }
        else { res.className='res bad'; res.textContent='✗ Не то, попробуй ещё'; hint.classList.add('on'); }
        save(); score(); }
      t.querySelector('.chk').addEventListener('click', check);
      inp.addEventListener('keydown', function(e){ if(e.key==='Enter') check(); });
      var s=st[t.id]; if(s&&s.ok){ inp.value=s.val||''; res.className='res ok'; res.textContent='✓ Верно'; t.classList.add('done'); }
    });
    root.querySelectorAll('.tq[data-kind=mc]').forEach(function(t){
      var why=t.querySelector('.why'), opts=[].slice.call(t.querySelectorAll('.opt'));
      opts.forEach(function(b){ b.addEventListener('click', function(){ var r=rec(t); if(!r.ok) r.tries++; t.dataset.picked=b.textContent;
        b.classList.add(b.dataset.c==="1"?'ok':'bad');
        why.innerHTML=(b.dataset.c==="1"?'<b style="color:var(--ok)">✓</b> ':'<b style="color:var(--warn)">✗</b> ')+b.dataset.why; why.classList.add('on');
        if(b.dataset.c==="1"){ r.ok=true; t.classList.add('done'); } save(); score(); }); });
      var s=st[t.id]; if(s&&s.ok){ opts.forEach(function(b){ if(b.dataset.c==="1"){ b.classList.add('ok'); why.innerHTML='<b style="color:var(--ok)">✓</b> '+b.dataset.why; why.classList.add('on'); } }); t.classList.add('done'); }
    });
    root.querySelectorAll('.tq[data-kind=txt]').forEach(function(t){
      var ta=t.querySelector('textarea'), ok=t.querySelector('.cpok'), n=t.querySelector('.tn').textContent, q=qtext(t);
      ta.value = st['txt_'+t.id] || '';
      ta.addEventListener('input', function(){ st['txt_'+t.id]=ta.value; save(); });
      t.querySelector('.cp').addEventListener('click', function(){ copyText(title + ", задание " + n + ". " + q + "\nМой ответ: " + ta.value.trim(), ok); });
    });
    root.querySelector('.trreset').addEventListener('click', function(){ st={}; save();
      root.querySelectorAll('.tq').forEach(function(t){ t.classList.remove('done'); delete t.dataset.picked; });
      root.querySelectorAll('input.ans, textarea').forEach(function(x){ x.value=''; });
      root.querySelectorAll('.res').forEach(function(x){ x.className='res'; x.textContent=''; });
      root.querySelectorAll('.fb').forEach(function(x){ x.classList.remove('on'); });
      root.querySelectorAll('.opt').forEach(function(x){ x.classList.remove('ok','bad'); });
      root.querySelectorAll('details.sol').forEach(function(x){ x.open=false; });
      score(); });
    score();
  });

  // timers
  document.querySelectorAll('.timer').forEach(function(tb){
    var clock=tb.querySelector('.clock'), tmin=tb.querySelector('.tmin'), iv=null, remaining=parseInt(tb.dataset.min,10)*60;
    function fmt(s){ var m=Math.floor(s/60), r=s%60; return (m<10?"0":"")+m+":"+(r<10?"0":"")+r; }
    function draw(){ clock.textContent=fmt(remaining); }
    tb.querySelector('.tstart').addEventListener('click', function(){ if(iv) return; tb.classList.remove('done');
      iv=setInterval(function(){ remaining=Math.max(0,remaining-1); draw(); if(remaining===0){ clearInterval(iv); iv=null; tb.classList.add('done'); clock.textContent="Время!"; } },1000); });
    tb.querySelector('.tpause').addEventListener('click', function(){ clearInterval(iv); iv=null; });
    tb.querySelector('.treset').addEventListener('click', function(){ clearInterval(iv); iv=null; tb.classList.remove('done'); remaining=Math.max(1,parseInt(tmin.value,10)||120)*60; draw(); });
    tmin.addEventListener('change', function(){ if(!iv){ remaining=Math.max(1,parseInt(tmin.value,10)||120)*60; draw(); } });
  });

  // error log
  var log=document.getElementById('errlog'), LK="stat5101-exam3-errlog";
  if(log){ log.value=store.get(LK)||""; log.addEventListener('input', function(){ store.set(LK, log.value); }); }

  window.addEventListener('beforeprint', function(){ document.querySelectorAll('details').forEach(function(d){ d.open=true; }); });
  var links=Array.prototype.slice.call(document.querySelectorAll('nav.top a')); var navw=document.querySelector('nav.top .wrap');
  if('IntersectionObserver' in window){
    var obs=new IntersectionObserver(function(entries){ entries.forEach(function(en){ if(en.isIntersecting){ links.forEach(function(a){ var on=a.getAttribute('href')==='#'+en.target.id; a.classList.toggle('active', on); if(on && navw){ navw.scrollLeft = a.offsetLeft - navw.clientWidth/2 + a.clientWidth/2; } }); } }); }, {rootMargin:"-40% 0px -55% 0px"});
    document.querySelectorAll('main > section').forEach(function(s){ obs.observe(s); });
  }
})();
</script>
'''.replace('__EXPL__', json.dumps(JS_EXPL, ensure_ascii=False))

html = f'''<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>STAT 5101 · Экзамен 3 · программа на 3 недели</title>
__APPHEAD__
<style>{css_main}</style>
<style>{css_blocks}</style>
<style>{tr_css}{extra_css}</style>
</head>
<body>
<header class="hero">
  <div class="wrap">
    __APPBACK__<h1>Экзамен 3 по STAT 5101: программа на 3 недели</h1>
    <p class="sub">Модули 5 и 6: простая, логистическая, пуассоновская и множественная регрессия. Главное — научиться читать вывод R: каждое число в нём можно нажать и посмотреть, что оно значит и как его пересчитать на калькуляторе. Примеры со свинофермы, задачи из Canvas и два пробных экзамена на английском.</p>
    <div class="chips">
      <span class="chip">21 день по 2–2,5 часа</span>
      <span class="chip">__NCLICK__ кликабельных выводов R</span>
      <span class="chip">4 тренажёра, __NTASK__ заданий</span>
      <span class="chip">2 пробника по 26 вопросов</span>
    </div>
    <div class="progress"><span>Сделано:</span><div class="bar"><span id="pbar"></span></div><span id="ptext">0 из 0</span><button class="btn" id="resetProg" type="button">Сбросить</button></div>
  </div>
</header>
<nav class="top"><div class="wrap">
__APPNAV__{''.join(f'  <a href="#{a}">{b}</a>' for a, b in nav)}
</div></nav>
<main class="wrap">
{''.join(sections)}
</main>
<footer>STAT 5101 · Exam 3 · Fall 2026 · материалы из Canvas, примеры пересчитаны</footer>
{js}
</body>
</html>
'''
ntask = html.count('class="tq"')
apphead = '''<link rel="manifest" href="manifest.webmanifest">
<meta name="theme-color" content="#101821">
<meta name="mobile-web-app-capable" content="yes">
<link rel="icon" href="icon-192.png">
<link rel="apple-touch-icon" href="apple-touch-icon.png">''' if APP else ''
html = (html.replace('__APPHEAD__', apphead)
            .replace('__APPBACK__', '<a class="appback" href="./#home">☰ Меню тренажёра</a>' if APP else '')
            .replace('__APPNAV__', '  <a href="./#home">☰ Меню</a>' if APP else ''))
html = html.replace('__NCLICK__', str(len(JS_EXPL))).replace('__NTASK__', str(ntask))
outname = 'exam3.html' if APP else 'STAT5101_Exam3_3-week_plan.html'
open(HERE + '/' + outname, 'w', encoding='utf-8').write(html)
print('wrote', outname, 'checkboxes', html.count('type="checkbox"'))
print('tasks', ntask)
print(len(html), 'bytes; trainers', TRAINERS, 'rclick', list(JS_EXPL))
