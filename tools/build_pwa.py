import json, os, re, shutil
from PIL import Image, ImageDraw, ImageFont

VERSION = "1.6"
DATE = "21.11.2026"
import sys
OUT = sys.argv[1] if len(sys.argv) > 1 else ".."
os.makedirs(OUT, exist_ok=True)

# ---------- icons ----------
def icon(size, maskable=False):
    bg = (16, 24, 33)
    im = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(im)
    if not maskable:
        im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([0, 0, size - 1, size - 1], radius=int(size * 0.22), fill=bg)
    safe = 0.62 if maskable else 0.78
    fbig = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", int(size * 0.46 * safe / 0.78))
    fsm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", int(size * 0.20 * safe / 0.78))
    txt = "R"
    w = d.textlength(txt, font=fbig)
    cx, cy = size / 2, size / 2
    d.text((cx - w / 2 + size * 0.06 * safe, cy - size * 0.30 * safe / 0.78), txt, font=fbig, fill=(138, 184, 255))
    d.text((cx - size * 0.30 * safe / 0.78, cy - size * 0.17 * safe / 0.78), ">", font=fsm, fill=(219, 228, 238))
    bw = size * 0.30 * safe / 0.78
    y = cy + size * 0.22 * safe / 0.78
    d.rectangle([cx - bw / 2, y, cx + bw / 2, y + size * 0.035], fill=(219, 228, 238))
    return im

icon(192).save(f"{OUT}/icon-192.png")
icon(512).save(f"{OUT}/icon-512.png")
icon(512, maskable=True).save(f"{OUT}/icon-maskable-512.png")
icon(180, maskable=True).save(f"{OUT}/apple-touch-icon.png")

# ---------- manifest ----------
manifest = {
    "name": "Тренажёр R",
    "short_name": "R-тренажёр",
    "description": "Тренажёр выводов R (t-тесты, непараметрические тесты, таблицы, ANOVA, регрессии), объяснения методов и тренажёр синтаксиса R.",
    "lang": "ru",
    "start_url": "./",
    "scope": "./",
    "display": "standalone",
    "orientation": "portrait",
    "background_color": "#f4f6f9",
    "theme_color": "#101821",
    "icons": [
        {"src": "icon-192.png", "sizes": "192x192", "type": "image/png"},
        {"src": "icon-512.png", "sizes": "512x512", "type": "image/png"},
        {"src": "icon-maskable-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable"},
    ],
}
json.dump(manifest, open(f"{OUT}/manifest.webmanifest", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# ---------- service worker ----------
sw = """const CACHE = 'r-trainer-v__VER__';
const CORE = ['./', 'index.html', 'exam3.html', 'manifest.webmanifest', 'icon-192.png', 'icon-512.png', 'icon-maskable-512.png', 'apple-touch-icon.png'];
self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(CORE)));
});
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE && k.startsWith('r-trainer-')).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener('message', e => { if (e.data === 'skip') self.skipWaiting(); });
self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin === location.origin) {
    e.respondWith(caches.match(req, {ignoreSearch: true}).then(hit => hit || fetch(req).then(res => {
      if (res.ok) { const copy = res.clone(); caches.open(CACHE).then(c => c.put(req, copy)); }
      return res;
    }).catch(() => req.mode === 'navigate' ? caches.match('index.html') : undefined)));
    return;
  }
  if (url.hostname === 'fonts.googleapis.com' || url.hostname === 'fonts.gstatic.com') {
    e.respondWith(caches.open('r-trainer-fonts').then(c => c.match(req).then(hit => {
      const net = fetch(req).then(res => { c.put(req, res.clone()); return res; }).catch(() => hit);
      return hit || net;
    })));
  }
});
""".replace("__VER__", VERSION)
open(f"{OUT}/sw.js", "w").write(sw)

# ---------- exam 3 lesson ----------
shutil.copy("exam3/exam3.html", f"{OUT}/exam3.html")

# ---------- index.html ----------
html = open("r-trainer.html", encoding="utf-8").read()
head_add = """<link rel="manifest" href="manifest.webmanifest">
<meta name="theme-color" content="#101821">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<link rel="icon" href="icon-192.png">
<link rel="apple-touch-icon" href="apple-touch-icon.png">
"""
html = re.sub(r'(<title>[^<]*</title>)', lambda m: m.group(1) + '\n' + head_add, html, count=1)
css_add = """.upd{position:fixed;left:12px;right:12px;bottom:76px;z-index:9;background:var(--console-bg);color:var(--console-ink);border-radius:12px;padding:12px 14px;display:flex;gap:10px;align-items:center;justify-content:space-between;font-size:14px}
.upd .btn{flex:none}
"""
html = html.replace("[hidden]{display:none!important}", css_add + "[hidden]{display:none!important}", 1)
# версия и список изменений — только в README.md и в имени кэша sw.js; в меню приложения их не показываем
upd = """<div class="upd" id="upd" hidden><span>Вышло обновление тренажёра</span><button class="btn" id="upd-btn" type="button">Обновить</button></div>
<script>
(function(){
  if (!('serviceWorker' in navigator)) return;
  var waiting = null, refreshing = false;
  function show(w){ waiting = w; document.getElementById('upd').hidden = false; }
  document.getElementById('upd-btn').onclick = function(){ if (waiting) waiting.postMessage('skip'); };
  navigator.serviceWorker.addEventListener('controllerchange', function(){ if (refreshing) return; refreshing = true; location.reload(); });
  navigator.serviceWorker.register('sw.js').then(function(reg){
    if (reg.waiting && navigator.serviceWorker.controller) show(reg.waiting);
    reg.addEventListener('updatefound', function(){
      var nw = reg.installing;
      nw.addEventListener('statechange', function(){ if (nw.state === 'installed' && navigator.serviceWorker.controller) show(nw); });
    });
    document.addEventListener('visibilitychange', function(){ if (document.visibilityState === 'visible') reg.update().catch(function(){}); });
  }).catch(function(){});
})();
</script>
</body>"""
html = html.replace("</body>", upd, 1)
open(f"{OUT}/index.html", "w", encoding="utf-8").write(html)
open(f"{OUT}/.nojekyll", "w").write("")
print("ok", os.listdir(OUT))
