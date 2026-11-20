"""Tiny SVG chart helper in the reference style (uses CSS vars)."""
import numpy as np
_CID = [0]


def nice_ticks(lo, hi, n=5):
    span = hi - lo
    raw = span / n
    mag = 10 ** np.floor(np.log10(raw))
    for m in [1, 2, 2.5, 5, 10]:
        step = m * mag
        if span / step <= n + 0.5:
            break
    start = np.ceil(lo / step) * step
    ticks = []
    t = start
    while t <= hi + 1e-9:
        ticks.append(round(t, 10))
        t += step
    return ticks


def fmt(t):
    if abs(t - round(t)) < 1e-9:
        return str(int(round(t))).replace('-', '−')
    s = f'{t:.2f}'.rstrip('0').rstrip('.')
    return s.replace('.', ',').replace('-', '−')


class Plot:
    def __init__(self, xr, yr, w=460, h=300, xlab='', ylab='', title='', ml=56, mr=14, mt=30, mb=44, xticks=None, yticks=None):
        self.x0, self.x1 = xr
        self.y0, self.y1 = yr
        self.w, self.h = w, h
        self.ml, self.mr, self.mt, self.mb = ml, mr, mt, mb
        self.items = []
        self.texts = []
        self.xlab, self.ylab, self.title = xlab, ylab, title
        self.xticks = xticks if xticks is not None else nice_ticks(self.x0, self.x1)
        self.yticks = yticks if yticks is not None else nice_ticks(self.y0, self.y1)

    def X(self, x):
        return self.ml + (x - self.x0) / (self.x1 - self.x0) * (self.w - self.ml - self.mr)

    def Y(self, y):
        return self.h - self.mb - (y - self.y0) / (self.y1 - self.y0) * (self.h - self.mt - self.mb)

    def points(self, xs, ys, color='var(--series-1)', r=3.2, op=0.8):
        for x, y in zip(xs, ys):
            self.items.append(f'<circle cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="{r}" fill="{color}" fill-opacity="{op}"/>')

    def line(self, xs, ys, color='var(--series-2)', width=2.2, dash=None):
        d = ' '.join(f'{"M" if i == 0 else "L"}{self.X(x):.1f},{self.Y(y):.1f}' for i, (x, y) in enumerate(zip(xs, ys)))
        da = f' stroke-dasharray="{dash}"' if dash else ''
        self.items.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{da}/>')

    def band(self, xs, lo, hi, color='var(--series-1)', op=0.15):
        pts = [(self.X(x), self.Y(y)) for x, y in zip(xs, hi)] + [(self.X(x), self.Y(y)) for x, y in zip(xs[::-1], lo[::-1])]
        d = 'M' + ' L'.join(f'{a:.1f},{b:.1f}' for a, b in pts) + ' Z'
        self.items.append(f'<path d="{d}" fill="{color}" fill-opacity="{op}" stroke="none"/>')

    def seg(self, x1, y1, x2, y2, color='var(--axis)', width=1.2, dash='3 3'):
        self.items.append(f'<line x1="{self.X(x1):.1f}" y1="{self.Y(y1):.1f}" x2="{self.X(x2):.1f}" y2="{self.Y(y2):.1f}" stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>')

    def text(self, x, y, s, cls='note', anchor='start', dx=0, dy=0, color=None):
        f = f' fill="{color}"' if color else ''
        self.texts.append(f'<text class="{cls}" x="{self.X(x)+dx:.1f}" y="{self.Y(y)+dy:.1f}" text-anchor="{anchor}"{f}>{s}</text>')

    def hline(self, y, color='var(--axis)', dash='4 4'):
        self.items.append(f'<line x1="{self.ml}" x2="{self.w-self.mr}" y1="{self.Y(y):.1f}" y2="{self.Y(y):.1f}" stroke="{color}" stroke-dasharray="{dash}"/>')

    def render(self, aria=''):
        g = []
        for t in self.yticks:
            if self.y0 <= t <= self.y1:
                y = self.Y(t)
                g.append(f'<line x1="{self.ml}" x2="{self.w-self.mr}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--grid)"/>')
                g.append(f'<text class="tick" x="{self.ml-6}" y="{y+4:.1f}" text-anchor="end">{fmt(t)}</text>')
        for t in self.xticks:
            if self.x0 <= t <= self.x1:
                x = self.X(t)
                g.append(f'<text class="tick" x="{x:.1f}" y="{self.h-self.mb+16}" text-anchor="middle">{fmt(t)}</text>')
        g.append(f'<line x1="{self.ml}" x2="{self.w-self.mr}" y1="{self.h-self.mb}" y2="{self.h-self.mb}" stroke="var(--axis)"/>')
        g.append(f'<line x1="{self.ml}" x2="{self.ml}" y1="{self.mt}" y2="{self.h-self.mb}" stroke="var(--axis)"/>')
        if self.xlab:
            g.append(f'<text class="axlab" x="{(self.ml+self.w-self.mr)/2:.1f}" y="{self.h-8}" text-anchor="middle">{self.xlab}</text>')
        if self.ylab:
            g.append(f'<text class="axlab" x="14" y="{(self.mt+self.h-self.mb)/2:.1f}" text-anchor="middle" transform="rotate(-90 14 {(self.mt+self.h-self.mb)/2:.1f})">{self.ylab}</text>')
        if self.title:
            g.append(f'<text class="ptitle" x="{self.ml}" y="18">{self.title}</text>')
        _CID[0] += 1
        cid = f'clip{_CID[0]}'
        clip = f'<clipPath id="{cid}"><rect x="{self.ml}" y="{self.mt}" width="{self.w-self.ml-self.mr}" height="{self.h-self.mt-self.mb}"/></clipPath>'
        return (f'<svg class="chart" viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{aria}"><defs>{clip}</defs>' + ''.join(g) + f'<g clip-path="url(#{cid})">' + ''.join(self.items) + '</g>' + ''.join(self.texts) + '</svg>')


def figure(svgs, caption, cols=None):
    if len(svgs) == 1:
        inner = svgs[0]
    else:
        inner = '<div class="grid2">' + ''.join(f'<div>{s}</div>' for s in svgs) + '</div>'
    return f'<figure>{inner}<figcaption>{caption}</figcaption></figure>\n'
