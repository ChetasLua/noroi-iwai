"""Kit of parts for the sets: facades, signs, crowd, cars, poles, wires, rubble, trees."""
import numpy as np, math
from .paint import rgb, f32, mix
from .world import C, Scene, Wall

WARM = ['#ffdca0', '#ffcf8a', '#ffe6bd', '#ffc97a']; NEUT = ['#fff1da', '#f6f3ea', '#ffeccb']; COOL = ['#cfe4ff', '#b8dcf4', '#d8f0ea', '#e6f2ff']; DIM = ['#b98244', '#8c5c38', '#5f80a4', '#9a6f55']
NEON = [('#c8182e', '#fff0e2', '#ffcf5a'), ('#f4ead6', '#b81c2e', '#1a1a24'), ('#f7f0e0', '#1d2a52', '#c8202e'), ('#e8f0f4', '#14507a', '#e0503a'), ('#e07a2c', '#fff6e0', '#2a1410'), ('#173a8a', '#e8f2ff', '#ffcf5a'),
        ('#f2c548', '#2a1a14', '#b81c2e'), ('#101119', '#ff4a52', '#f6efe0'), ('#101119', '#7fe0ff', '#f6efe0'), ('#101119', '#ffc25a', '#f6efe0'), ('#b8243e', '#ffe6d0', '#14121a'), ('#1b7f86', '#eafcff', '#ffe08a'), ('#d8d2c2', '#23242e', '#b81c2e'),
        ('#c23a78', '#fff0f4', '#2a0c1c'), ('#2d6a4a', '#f0f4e0', '#ffd66a'), ('#f6f2ea', '#e0702a', '#23242e')]


def pick(r, L): return L[int(r.integers(len(L)))]


def wcols(r, n, mixw=(.45, .28, .20, .07)):
    """n window light colours"""
    k = r.choice(4, n, p=np.array(mixw) / sum(mixw)); g = [WARM, NEUT, COOL, DIM]
    return np.stack([rgb(pick(r, g[i])) for i in k]) * r.uniform(.72, 1.12, (n, 1)).astype(f32)


# ------------------------------------------------------------------------------------------------ graphics without words
def graphic(W, u0, u1, v0, v1, d, r, pal=None, kind=None, em=1.0, bg=True, dd=.012):
    """abstract sign art inside a rectangle of a Wall frame"""
    bgc, fg, ac = [rgb(c) for c in (pal or pick(r, NEON))]
    w, h = u1 - u0, v1 - v0; cu, cv = (u0 + u1) / 2, (v0 + v1) / 2; m = min(w, h)
    if bg: W.rect(u0, u1, v0, v1, d, bgc * .86, ct=bgc, em=em, tex=0)
    d += dd; kw = dict(em=em, snow=0)
    if kind == 'bars' and m > 1.5: kind = None
    if kind is None: kind = pick(r, ['bars' if m <= 1.5 else 'wave', 'disc', 'wave', 'rings', 'chev', 'petal', 'fish', 'cup', 'stripes', 'mount', 'note', 'blocks', 'eye', 'drop', 'grid', 'arc'])
    if kind == 'bars':
        n = int(max(2, min(9, round((w if w > h else h) / (m * .55)))));
        for i in range(n):
            t = (i + .5) / n; a, b = (.5 * m * r.uniform(.45, .8), m * r.uniform(.2, .36))
            if w > h: x = u0 + w * t; _glyph(W, x, cv, min(w / n * .36, m * .34), m * .32, d, fg, r, kw)
            else: y = v1 - h * t; _glyph(W, cu, y, m * .32, min(h / n * .36, m * .34), d, fg, r, kw)
    elif kind == 'disc':
        W.disc(cu + r.uniform(-.12, .12) * w, cv, m * .34, d, fg, **kw); W.disc(cu + r.uniform(-.05, .05) * w, cv - m * .04, m * .16, d + dd, ac, **kw)
        W.rect(u0 + w * .06, u1 - w * .06, v0 + h * .06, v0 + h * .10, d, ac, **kw)
    elif kind == 'wave':
        for k in range(3):
            t = np.linspace(0, 1, 13); y = v0 + h * (.18 + .22 * k); amp = h * .09; pts = [(u0 + w * a, y + amp * math.sin(a * 9 + k * 1.3)) for a in t]
            for (a, b), (c_, e) in zip(pts[:-1], pts[1:]): W.poly([(a, b), (c_, e), (c_, e + h * .10), (a, b + h * .10)], d + dd * k, fg if k != 1 else ac, **kw)
    elif kind == 'rings':
        for k, c_ in enumerate((fg, ac, fg)): W.ring(cu, cv, m * (.40 - .13 * k) - m * .06, m * (.40 - .13 * k), d + dd * k * .3, c_, **kw)
    elif kind == 'chev':
        n = int(max(3, w / (m * .4)))
        for i in range(n):
            x = u0 + w * (i + .15) / n; q = w / n * .5
            W.poly([(x, v0 + h * .2), (x + q * .5, v0 + h * .2), (x + q * 1.2, cv), (x + q * .7, cv)], d, fg if i % 3 else ac, **kw); W.poly([(x, v1 - h * .2), (x + q * .7, cv), (x + q * 1.2, cv), (x + q * .5, v1 - h * .2)], d, fg if i % 3 else ac, **kw)
    elif kind == 'petal':
        a0 = r.uniform(0, 1)
        for k in range(6):
            a = a0 + k * math.pi / 3; c_, s_ = math.cos(a), math.sin(a); R = m * .4
            W.poly([(cu, cv), (cu + c_ * R * .55 - s_ * R * .24, cv + s_ * R * .55 + c_ * R * .24), (cu + c_ * R, cv + s_ * R), (cu + c_ * R * .55 + s_ * R * .24, cv + s_ * R * .55 - c_ * R * .24)], d, fg if k % 2 else ac, **kw)
    elif kind == 'fish':
        x, y, q = cu, cv, m * .42
        W.poly([(x - q * 1.1, y), (x - q * .4, y + q * .5), (x + q * .5, y + q * .42), (x + q * .9, y), (x + q * .5, y - q * .42), (x - q * .4, y - q * .5)], d, fg, **kw)
        W.poly([(x + q * .75, y), (x + q * 1.35, y + q * .5), (x + q * 1.2, y), (x + q * 1.35, y - q * .5)], d, ac, **kw); W.disc(x - q * .62, y + q * .1, q * .10, d + dd, ac, 10, **kw)
    elif kind == 'cup':
        q = m * .34; W.poly([(cu - q, cv + q * .5), (cu - q * .7, cv - q), (cu + q * .7, cv - q), (cu + q, cv + q * .5)], d, fg, **kw); W.ring(cu + q * 1.05, cv - q * .05, q * .28, q * .5, d, fg, a0=-1.7, a1=1.7, **kw)
        for k in (-.4, .1, .55): W.poly([(cu + q * k - q * .08, cv + q * .65), (cu + q * k + q * .08, cv + q * .65), (cu + q * k + q * .22, cv + q * 1.2), (cu + q * k + q * .06, cv + q * 1.2)], d, ac, **kw)
    elif kind == 'stripes':
        for k in range(5): x = u0 + w * (k + .2) / 5; W.poly([(x, v0 + h * .08), (x + w * .1, v0 + h * .08), (x + w * .1 + h * .25, v1 - h * .08), (x + h * .25, v1 - h * .08)], d, (fg, ac)[k % 2], **kw)
    elif kind == 'mount':
        W.poly([(u0 + w * .05, v0 + h * .12), (u1 - w * .05, v0 + h * .12), (cu + w * .04, v1 - h * .18)], d, fg, **kw); W.poly([(cu - w * .09, v0 + h * .58), (cu + w * .17, v0 + h * .58), (cu + w * .04, v1 - h * .18)], d + dd, ac, **kw); W.disc(u0 + w * .22, v1 - h * .26, m * .10, d, ac, **kw)
    elif kind == 'note':
        q = m * .3; W.disc(cu - q * .6, cv - q * .8, q * .42, d, fg, 14, q * .32, **kw); W.disc(cu + q * .8, cv - q * .55, q * .42, d, fg, 14, q * .32, **kw)
        W.rect(cu - q * .32, cu - q * .18, cv - q * .8, cv + q * .9, d, fg, **kw); W.rect(cu + q * 1.06, cu + q * 1.2, cv - q * .5, cv + q * 1.15, d, fg, **kw); W.poly([(cu - q * .32, cv + q * .6), (cu + q * 1.2, cv + q * .85), (cu + q * 1.2, cv + q * 1.2), (cu - q * .32, cv + q * .95)], d, ac, **kw)
    elif kind == 'blocks':
        nx = max(2, int(round(w / m * 2))); ny = max(2, int(round(h / m * 2)))
        for i in range(nx):
            for j in range(ny):
                if r.random() < .62: q = .12; c_ = pick(r, [fg, ac, bgc * .55]); W.rect(u0 + w * (i + q) / nx, u0 + w * (i + 1 - q) / nx, v0 + h * (j + q) / ny, v0 + h * (j + 1 - q) / ny, d, c_, **kw)
    elif kind == 'eye':
        t = np.linspace(0, math.pi, 12); q = min(w * .42, h * .8); up = np.c_[cu + np.cos(t) * q, cv + np.sin(t) * q * .46]; dn = np.c_[cu - np.cos(t) * q, cv - np.sin(t) * q * .34]
        W.poly(np.vstack([up, dn]), d, fg, **kw); W.disc(cu, cv, q * .34, d + dd, ac, **kw); W.disc(cu, cv, q * .15, d + 2 * dd, bgc * .4, **kw)
    elif kind == 'drop':
        q = m * .36
        for k in (-1, 0, 1):
            x = cu + k * min(w * .28, q * 1.3); W.disc(x, cv - q * .2, q * .42, d, (fg, ac, fg)[k + 1], 16, **kw); W.poly([(x - q * .4, cv - q * .05), (x + q * .4, cv - q * .05), (x, cv + q * .95)], d, (fg, ac, fg)[k + 1], **kw)
    elif kind == 'grid':
        for k in range(4): W.rect(u0 + w * .1 + w * .8 * k / 3.6, u0 + w * .1 + w * .8 * (k + .55) / 3.6, v0 + h * .14, v0 + h * (.3 + .55 * r.random()), d, (fg, ac)[k % 2], **kw)
    else:
        W.ring(cu, v0 + h * .1, m * .5, m * .66, d, fg, a0=0, a1=math.pi, **kw); W.ring(cu, v0 + h * .1, m * .26, m * .4, d, ac, a0=0, a1=math.pi, **kw)


def lightbox(W, u0, u1, v0, v1, d, r, pal=None, em=1., kind=None):
    """backlit sign: tubes hot in the middle, dull rim, panel seams, art"""
    pal = pal or pick(r, NEON); c = C(pal[0]); w, h = u1 - u0, v1 - v0; q = min(w, h) * .22
    P = lambda u, v: W.p(u, v, d)
    ui, vi = np.array([u0, u0 + q, u1 - q, u1]), np.array([v0, v0 + q, v1 - q, v1])
    g = np.full((4, 4), .58); g[1:3, 1:3] = 1.06; cols = g[..., None] * c
    for i in range(3):
        for j in range(3):
            W.S.quads(np.stack([P(ui[i], vi[j]), P(ui[i + 1], vi[j]), P(ui[i + 1], vi[j + 1]), P(ui[i], vi[j + 1])])[None], np.stack([cols[i, j], cols[i + 1, j], cols[i + 1, j + 1], cols[i, j + 1]])[None], em=em, snow=0, grp=W.grp, lab=W.S.nlab)
    W.S.nlab += 1
    m = max(1, int(w / 2.4)); x = u0 + w * np.arange(1, m) / m
    if m > 1 and w > 3: W.rect(x - .02, x + .02, v0, v1, d + .006, c * .5, em=em * .8, snow=0)
    graphic(W, u0 + q * .4, u1 - q * .4, v0 + q * .4, v1 - q * .4, d + .012, r, pal, kind, em=em, bg=False)
    W.slab(u0 - .06, u1 + .06, np.array([v0 - .06, v1]), np.array([v0, v1 + .06]), d - .1, d + .06, '#1c1e26', snow=.5); W.slab(np.array([u0 - .06, u1]), np.array([u0, u1 + .06]), v0, v1, d - .1, d + .06, '#1c1e26', snow=0)


def _glyph(W, x, y, a, b, d, col, r, kw):
    """an invented mark: dense crossed strokes, a box or splayed legs, dots.  never a letter of any script"""
    t = max(a, b) * .13
    def bar(x0, y0, x1, y1, w=1.):
        v = np.array([x1 - x0, y1 - y0]); L = math.hypot(*v) + 1e-9; n = np.array([-v[1], v[0]]) / L * t * w / 2
        W.poly([(x0 - n[0], y0 - n[1]), (x1 - n[0] * .7, y1 - n[1] * .7), (x1 + n[0] * .7, y1 + n[1] * .7), (x0 + n[0], y0 + n[1])], d, col, **kw)
    k = int(r.integers(0, 4)); cx = x + a * r.uniform(-.15, .15)
    cx = x + a * r.choice([-.45, .4, -.3, .5])
    for yy in np.sort(r.uniform(-.25, .95, 3)): l_, r_ = r.uniform(.1, .95), r.uniform(.1, .95); bar(x - a * l_, y + b * yy, x + a * r_, y + b * (yy + r.uniform(-.28, .28)))
    bar(cx, y + b, cx + a * r.uniform(-.5, .5), y - b * r.uniform(.1, .9), 1.15); bar(x + a * r.uniform(-.9, .9), y + b * r.uniform(0, .9), x + a * r.uniform(-.9, .9), y - b * r.uniform(0, .9), .8)
    if k == 0: bar(cx, y, x - a, y - b, .9); bar(cx, y, x + a, y - b, 1.1)
    elif k == 1:
        q = a * r.uniform(.5, .8); y0, y1 = y - b * .95, y - b * r.uniform(.2, .4); bar(x - q, y0, x - q, y1); bar(x + q, y0, x + q, y1); bar(x - q, y0, x + q, y0); bar(x - q, y1, x + q, y1)
    elif k == 2:
        for u in (-.7, -.25, .25, .7): bar(x + a * u, y - b * .55, x + a * (u + .06), y - b * .95, 1.2)
    else: bar(x - a * .9, y + b * .5, x - a * .6, y - b * .9, .9); bar(x - a * .2, y - b * .1, x + a * .9, y - b * .9, .9); bar(x + a * .5, y - b * .05, x - a * .1, y - b * .95, .8)


def screen_art(W, u0, u1, v0, v1, d, r, kind='iris', em=1.15, pal=None):
    """big LED screen pictures: iris, koi, city, bloom, face-less silhouette"""
    w, h = u1 - u0, v1 - v0; cu, cv = (u0 + u1) / 2, (v0 + v1) / 2; kw = dict(em=em, snow=0); m = min(w, h)
    if kind == 'iris':
        A, B, C_, D_ = [rgb(c) for c in (pal or ('#0b1c4a', '#38b6ff', '#e8fbff', '#ff3f7e'))]
        W.rect(u0, u1, v0, v1, d, A, ct=mix(A, B, .35), **kw)
        x, y = u0 + w * .58, v0 + h * .5
        for k, (q, c) in enumerate(((.46, mix(A, B, .55)), (.38, B), (.27, mix(B, C_, .5)), (.16, A * .35))): W.disc(x, y, m * q, d + .02 * (k + 1), c, 36, **kw)
        for k in range(34): a = k * 2 * math.pi / 34 + r.uniform(-.05, .05); r0, r1 = m * .17, m * r.uniform(.25, .37); q = .03; W.poly([(x + math.cos(a - q) * r0, y + math.sin(a - q) * r0), (x + math.cos(a) * r1, y + math.sin(a) * r1), (x + math.cos(a + q) * r0, y + math.sin(a + q) * r0)], d + .12, pick(r, [C_, B, mix(B, C_, .5)]), **kw)
        W.disc(x + m * .10, y + m * .12, m * .06, d + .16, '#ffffff', **kw); W.rect(u0, u0 + w * .10, v0, v1, d + .03, D_, **kw); W.rect(u0 + w * .12, u0 + w * .30, v0 + h * .14, v0 + h * .18, d + .03, C_, **kw); W.rect(u0 + w * .12, u0 + w * .24, v0 + h * .22, v0 + h * .25, d + .03, C_, **kw)
    elif kind == 'koi':
        A, B, C_, D_ = [rgb(c) for c in (pal or ('#c8102e', '#ff5a3c', '#fff2e0', '#1a0a12'))]
        W.rect(u0, u1, v0, v1, d, A * .8, ct=B, **kw)
        for k in range(5): y = v0 + h * (.1 + .19 * k); W.ring(u0 + w * r.uniform(.1, .9), y, m * .18, m * .21, d + .02, mix(A, C_, .25), **kw)
        for (fx, fy, q, fl) in ((.36, .60, .30, 1), (.70, .30, .22, -1)):
            x, y, q = u0 + w * fx, v0 + h * fy, m * q
            W.poly([(x - fl * q * 1.2, y - q * .1), (x - fl * q * .4, y + q * .5), (x + fl * q * .6, y + q * .4), (x + fl * q * 1.1, y - q * .15), (x + fl * q * .4, y - q * .42), (x - fl * q * .5, y - q * .45)], d + .04, C_, **kw)
            W.poly([(x + fl * q * 1.0, y - q * .1), (x + fl * q * 1.75, y + q * .45), (x + fl * q * 1.5, y - q * .1), (x + fl * q * 1.75, y - q * .7)], d + .04, C_, **kw)
            W.poly([(x - fl * q * .3, y + q * .45), (x + fl * q * .3, y + q * .42), (x + fl * q * .1, y - q * .1)], d + .06, D_, **kw); W.poly([(x + fl * q * .5, y + q * .3), (x + fl * q * .9, y - q * .1), (x + fl * q * .4, y - q * .3)], d + .06, B, **kw)
            W.disc(x - fl * q * .82, y + q * .06, q * .07, d + .07, D_, 10, **kw)
    elif kind == 'bloom':
        A, B, C_, D_ = [rgb(c) for c in (pal or ('#2a0f4a', '#ff4fa0', '#ffe0f0', '#ffd24a'))]
        W.rect(u0, u1, v0, v1, d, A, ct=mix(A, B, .5), **kw)
        for j, (x, y, q) in enumerate(((.34, .46, .40), (.80, .70, .24), (.72, .16, .18))):
            x, y, q = u0 + w * x, v0 + h * y, m * q
            for L_, (rr, c) in enumerate(((1., B), (.74, mix(B, C_, .5)), (.46, C_))):
                for k in range(5): a = k * 2 * math.pi / 5 + j + L_ * .6; R = q * rr; ca, sa = math.cos(a), math.sin(a); W.poly([(x, y), (x + ca * R * .6 - sa * R * .42, y + sa * R * .6 + ca * R * .42), (x + ca * R, y + sa * R), (x + ca * R * .6 + sa * R * .42, y + sa * R * .6 - ca * R * .42)], d + .02 + .02 * L_, c, **kw)
            W.disc(x, y, q * .16, d + .12, D_, **kw)
    elif kind == 'city':
        A, B, C_, D_ = [rgb(c) for c in (pal or ('#ff7a3c', '#ffd27a', '#3a1c5a', '#fff4d8'))]
        W.rect(u0, u1, v0, v1, d, B, ct=A, **kw); W.disc(u0 + w * .62, v0 + h * .46, m * .26, d + .02, D_, **kw)
        x = u0
        while x < u1 - w * .04: q = w * r.uniform(.05, .12); W.rect(x, min(x + q, u1), v0, v0 + h * r.uniform(.16, .5), d + .04, C_ * r.uniform(.8, 1.3), **kw); x += q * 1.05
        for k in range(3): W.rect(u0, u1, v0 + h * (.40 + .05 * k), v0 + h * (.412 + .05 * k), d + .03, A, **kw)
    else:
        A, B, C_, D_ = [rgb(c) for c in (pal or ('#0a2a3a', '#1fd0c0', '#f2fff8', '#ffcf3a'))]
        W.rect(u0, u1, v0, v1, d, A, ct=mix(A, B, .6), **kw)
        for k in range(9):
            t = np.linspace(0, 1, 15); y = v0 + h * (.08 + .1 * k)
            p = [(u0 + w * a, y + h * .08 * math.sin(a * 7 + k * .8) * (1 - .06 * k)) for a in t]
            for (a, b), (c_, e) in zip(p[:-1], p[1:]): W.poly([(a, b), (c_, e), (c_, e + h * .045), (a, b + h * .045)], d + .02, mix(B, C_, k / 9.), **kw)
        W.disc(u0 + w * .75, v0 + h * .72, m * .13, d + .05, D_, **kw)


# ------------------------------------------------------------------------------------------------ facades
def glass(W, u0, u1, v0, v1, d, r, lit, det=2, cols=None, frame=None, dark=('#0a1022', '#2a3a5e'), em=.85, own=False):
    """window panes with interiors: blinds, ceiling tubes, furniture, people"""
    u0, u1, v0, v1 = [np.atleast_1d(np.asarray(t, float)) for t in np.broadcast_arrays(u0, u1, v0, v1)]; n = len(u0)
    lit = np.broadcast_to(lit, (n,))
    dk = ~lit
    if dk.any():
        a, b = C(dark[0]), C(dark[1]); t = r.uniform(.3, 1., (dk.sum(), 1)).astype(f32)
        W.rect(u0[dk], u1[dk], v0[dk], v1[dk], d, a * (.7 + .6 * t), ct=mix(a, b, t), em=.35, snow=0)
        if det >= 2 and dk.sum() > 2:            # a curtain edge, a slanting reflection
            k = np.nonzero(dk)[0]; k = k[r.random(len(k)) < .4]
            if len(k):
                w = u1[k] - u0[k]; f = r.uniform(.2, .5, len(k)); W.rect(u0[k], u0[k] + w * f, v0[k], v1[k], d + .004, mix(C(dark[1]), C('#8a92a8'), .25) * .8, em=.25, snow=0)
    if det >= 2 and frame is not False:
        W.rect(u0, u1, v1 - (v1 - v0) * .30, v1 - (v1 - v0) * .30 + .05, d + .016, '#1a1c24', snow=0); W.rect(np.r_[u0, u1 - .05], np.r_[u0 + .05, u1], np.r_[v0, v0], np.r_[v1, v1], d + .016, '#1a1c24', snow=0)
    if lit.any():
        k = np.nonzero(lit)[0]; m = len(k); c = wcols(r, m) if cols is None else np.asarray(cols, f32)[k]
        W.rect(u0[k], u1[k], v0[k], v1[k], d, c * r.uniform(.45, .62, (m, 1)).astype(f32), ct=c, em=em, snow=0, own=own)
        if det >= 1:
            h = v1[k] - v0[k]; w = u1[k] - u0[k]
            bl = r.random(m) < .42; f = r.uniform(.15, .75, m)
            if bl.any(): W.rect(u0[k][bl], u1[k][bl], (v1[k] - h * f)[bl], v1[k][bl], d + .006, c[bl] * r.uniform(.62, .95, (bl.sum(), 1)).astype(f32) * np.array([1, .96, .9], f32), em=em * .9, snow=0)
        if det >= 2:
            tb = (r.random(m) < .5) & ~bl
            if tb.any(): q = np.nonzero(tb)[0]; o = r.uniform(.1, .5, len(q)); W.rect(u0[k][q] + w[q] * o, u0[k][q] + w[q] * (o + .4), v1[k][q] - h[q] * .16, v1[k][q] - h[q] * .10, d + .008, '#fffdf4', em=1.25, snow=0)
            fu = r.random(m) < .6
            if fu.any(): q = np.nonzero(fu)[0]; o = r.uniform(0, .5, len(q)); W.rect(u0[k][q] + w[q] * o, u0[k][q] + w[q] * (o + r.uniform(.25, .5, len(q))), v0[k][q], v0[k][q] + h[q] * r.uniform(.12, .34, len(q)), d + .01, c[q] * .28, em=em * .7, snow=0)
            pp = r.random(m) < .12
            if pp.any():
                q = np.nonzero(pp)[0]; x = u0[k][q] + w[q] * r.uniform(.2, .8, len(q)); hh = np.minimum(h[q] * .62, 1.5)
                W.rect(x - .19, x + .19, v0[k][q], v0[k][q] + hh * .75, d + .012, '#1a1420', em=.4, snow=0); W.rect(x - .10, x + .10, v0[k][q] + hh * .78, v0[k][q] + hh, d + .012, '#1a1420', em=.4, snow=0)


def litmap(r, nf, nb, p, kind='office'):
    """which panes are lit: whole floors, zones and strays, never a uniform sprinkle"""
    L = np.zeros((nf, nb), bool)
    for f in range(nf):
        q = r.random()
        if kind == 'office':
            if q < p * .55: L[f] = r.random(nb) < .93
            elif q < p * 1.05: a = int(r.integers(0, nb)); b = a + int(r.integers(1, max(2, nb * .6) + 1)); L[f, a:b] = True; L[f] &= r.random(nb) < .9
            else: L[f] = r.random(nb) < p * .10
        else: L[f] = r.random(nb) < p * r.uniform(.5, 1.3)
    return L


def facade(W, style, r, lit=.5, pal='#6a6f82', det=2, base=0., trim=None, grp=None, glow=1.):
    """dress one wall.  det 0 far, 1 mid, 2 near (reveals, sills, interiors).  base = height reserved for the shop floor"""
    S = W.S; w, h = W.w, W.h; pc = C(pal); tr = pc * 1.25 if trim is None else C(trim)
    if grp is not None: W.grp = grp
    body = pc;
    tex = {'office': 7, 'grid': 1, 'curtain': 0, 'tenant': 1, 'dept': 7, 'resid': 14, 'blank': 1, 'fins': 7, 'brick': 4}.get(style, 1)
    rec = .22 if (det >= 2 and style in ('office', 'fins', 'grid', 'brick') and h - base > 3 and w > 2) else 0.
    W.rect(0, w, 0, h, -rec, body * (.9 if not rec else .35), ct=body * (1.08 if not rec else .4), tex=tex if det > 0 else 0)
    H0 = base
    if w < 2 or h - H0 < 3: return
    if style in ('office', 'fins'):
        fh = min(r.uniform(3.5, 4.1), max(3.2, (h - H0 - .8) / max(1, round((h - H0 - .8) / 3.8)))); nf = int((h - H0 - .8) / fh); bw = r.uniform(1.3, 1.9); mg = min(1.2, w * .05)
        if nf < 1: return
        zn = max(1, int(round((w - 2 * mg) / (bw * 4)))); nb = zn * 4; bw = (w - 2 * mg) / nb
        L = litmap(r, nf, nb, lit); F, B = np.mgrid[0:nf, 0:nb]
        u0 = (mg + B * bw + .06).ravel(); u1 = u0 + bw - .12; v0 = (H0 + F * fh + 1.0).ravel(); v1 = v0 + fh - 1.35
        fc = wcols(r, nf, (.2, .5, .28, .02)); cols = np.repeat(fc, nb, 0) * r.uniform(.9, 1.08, (nf * nb, 1)).astype(f32)
        glass(W, u0, u1, v0, v1, -rec + .01, r, L.ravel(), det, cols, em=.82 * glow)
        if rec:
            yb = np.r_[0., H0 + np.arange(1, nf) * fh - .35, H0 + (nf - 1) * fh + fh - .35]; yt = np.r_[H0 + 1.0, H0 + np.arange(1, nf) * fh + 1.0, h]
            W.slab(0, w, yb, yt, -rec, 0, body, side=body * .6, top=body * 1.1, tex=tex); W.slab(np.array([0, w - mg]), np.array([mg, w]), 0, h, -rec, .01, body, side=body * .6, tex=tex)
        if det >= 1:
            y = H0 + np.arange(nf) * fh + .95
            W.slab(mg * .4, w - mg * .4, y - .28, y, 0, .22, tr, snow=1.)
            ux = mg + np.arange(0, nb + 1, 4) * bw
            if style == 'fins': W.slab(ux - .16, ux + .16, H0, h, 0, .55, tr * 1.05)
            else: W.slab(ux - .14, ux + .14, H0, h, 0, .10, pc)
    elif style in ('grid', 'brick'):
        fh = r.uniform(3.0, 3.5); nf = int((h - H0 - .6) / fh); bw = r.uniform(2.3, 3.4); nb = max(1, int((w - .8) / bw)); bw = (w - .8) / nb; ww = bw * r.uniform(.48, .66); wh = fh * r.uniform(.46, .58)
        if nf < 1: return
        F, B = np.mgrid[0:nf, 0:nb]; u0 = (.4 + B * bw + (bw - ww) / 2).ravel(); v0 = (H0 + F * fh + fh * .30).ravel(); L = litmap(r, nf, nb, lit, 'home').ravel()
        glass(W, u0, u0 + ww, v0, v0 + wh, -rec + .01, r, L, det, em=.85 * glow)
        if det >= 2:
            xs = .4 + np.arange(nb) * bw + (bw - ww) / 2; ys = H0 + np.arange(nf) * fh + fh * .30
            W.slab(np.r_[0., xs + ww], np.r_[xs, w], 0, h, -rec, 0, body * .98, side=body * .62, tex=tex); W.slab(0, w, np.r_[0., ys + wh], np.r_[ys, h], -rec, -.004, body * .98, side=body * .62, top=body * 1.1, tex=tex)
            W.rect(u0 + ww / 2 - .03, u0 + ww / 2 + .03, v0, v0 + wh, -rec + .05, '#1c1e28')
            W.slab(u0 - .08, u0 + ww + .08, v0 - .10, v0, 0, .14, tr)
            k = r.random(len(u0)) < .16
            if k.any(): a = u0[k] + r.uniform(0, ww * .4, k.sum()); W.slab(a, a + .8, v0[k] - .85, v0[k] - .28, 0, .36, '#9a9aa0'); W.rect(a + .12, a + .62, v0[k] - .78, v0[k] - .36, .365, '#4a4a52')
        if det >= 1:
            y = H0 + np.arange(1, nf + 1) * fh; W.slab(0, w, y - .05, y + .05, 0, .05, tr * .9)
    elif style == 'curtain':
        fh = r.uniform(3.7, 4.3); nf = int((h - H0) / fh); bw = r.uniform(1.5, 2.2); nb = max(2, int(w / bw)); bw = w / nb
        if nf < 1: return
        a, b = C('#141f3a') * (.6 + pc), C('#3d5680') * (.6 + pc)
        W.rect(0, w, H0, h, .02, a, ct=b, em=.3)
        F, B = np.mgrid[0:nf, 0:nb]; L = litmap(r, nf, nb, lit).ravel(); u0 = (B * bw + .05).ravel(); v0 = (H0 + F * fh + .55).ravel()
        fc = wcols(r, nf, (.15, .45, .38, .02)); cols = np.repeat(fc, nb, 0) * r.uniform(.6, 1.0, (nf * nb, 1)).astype(f32)
        k = L; glass(W, u0[k], u0[k] + bw - .1, v0[k], v0[k] + fh - .75, .035, r, np.ones(k.sum(), bool), min(det, 2), cols[k], em=.72 * glow) if k.any() else None
        k = ~L & (r.random(len(L)) < .4)
        if k.any(): q = r.uniform(.2, .8, (k.sum(), 1)).astype(f32); W.rect(u0[k], u0[k] + bw - .1, v0[k] - .5, v0[k] + fh - .6, .03, mix(a, b, q * .5), ct=mix(a, b, q), em=.34)
        if det >= 1:
            y = H0 + np.arange(nf + 1) * fh; W.slab(0, w, y - .09, y + .09, 0, .12, tr * .6); x = np.arange(nb + 1) * bw; W.slab(x - .045, x + .045, H0, h, 0, .14, tr * .5)
    elif style == 'tenant':
        fh = r.uniform(3.2, 3.7); nf = max(1, int((h - H0 - .4) / fh)); fh = (h - H0 - .4) / nf
        for f in range(nf):
            y = H0 + f * fh; on = r.random() < min(.9, lit * 1.22)
            nb = max(1, int(w / r.uniform(1.4, 2.6))); bw = (w - .5) / nb; B = np.arange(nb); L = (r.random(nb) < .85) if on else (r.random(nb) < .06)
            fcol = wcols(r, 1) * r.uniform(.55, 1.05) * (np.array([[1., .86, .9]], f32) if r.random() < .2 else 1.); glass(W, .25 + B * bw + .05, .25 + (B + 1) * bw - .05, np.full(nb, y + 1.05), np.full(nb, y + fh - .25), .02, r, L, det, np.repeat(fcol, nb, 0) * r.uniform(.8, 1.1, (nb, 1)).astype(f32), em=.86 * glow)
            if r.random() < .72:
                q = r.uniform(.55, 1.0); u0 = 0 if r.random() < .7 else w * (1 - q); pal_ = pick(r, NEON); on2 = .95 * glow if r.random() < .85 else .12
                W.slab(u0, u0 + w * q, y + .1, y + 1.0, 0, .20, C(pal_[0]) * .4, side=C(pal_[0]) * .35, em=on2 * .4, snow=0)
                lightbox(W, u0 + .05, u0 + w * q - .05, y + .14, y + .96, .22, r, pal_, on2, 'bars' if r.random() < .75 else None)
            else: W.slab(0, w, y, y + .2, 0, .18, tr)
    elif style == 'dept':
        n = max(2, int(w / 6)); x = np.linspace(0, w, n + 1)
        W.slab(x - .22, x + .22, H0, h, 0, .4, tr)
        for i in range(n):
            if r.random() < .5: yy = np.arange(H0 + 2, h - 2, r.uniform(.9, 1.4)); W.slab(x[i] + .5, x[i + 1] - .5, yy, yy + .25, 0, .16, tr * .9)
        nl = int(r.integers(2, 5)); xx = np.sort(r.uniform(.06, .94, nl)) * w
        W.slab(xx - .10, xx + .10, H0 + 1, h - .6, .1, .3, '#ffe9c0', em=1. * glow, snow=0)
    elif style == 'resid':
        fh = 2.9; nf = int((h - H0 - .4) / fh); bw = r.uniform(3.2, 4.6); nb = max(1, int(w / bw)); bw = w / nb
        if nf < 1: return
        F, B = np.mgrid[0:nf, 0:nb]; u = (B * bw).ravel(); v = (H0 + F * fh).ravel(); L = (r.random(nf * nb) < lit * .8)
        glass(W, u + .5, u + bw - .5, v + .15, v + 2.25, .01, r, L, det, em=.85 * glow)
        y = H0 + np.arange(nf) * fh
        W.slab(0, w, y - .16, y, 0, 1.2, tr); W.slab(.05, w - .05, y + .05, y + 1.05, 1.12, 1.2, pc * 1.1 if r.random() < .6 else mix(C('#5a7a8c'), pc, .4))
        if det >= 1:
            x = np.arange(nb + 1) * bw; W.slab(x - .06, x + .06, H0, H0 + nf * fh, 0, 1.2, pc)
            k = r.random(nf * nb) < .5
            if k.any(): W.slab(u[k] + .25, u[k] + 1.0, v[k] + .02, v[k] + .6, .3, .65, '#a6a6aa')
            if det >= 2:
                k = np.nonzero(r.random(nf * nb) < .22)[0]
                for i in k:
                    m = int(r.integers(2, 5)); a = u[i] + r.uniform(.4, 1.) + np.arange(m) * r.uniform(.4, .6); a = a[a < u[i] + bw - .6]
                    if len(a): W.rect(a, a + .36, v[i] + r.uniform(1.0, 1.4, len(a)), v[i] + 1.95, .9, np.stack([C(pick(r, ['#e8e4da', '#8fb0c4', '#c86a5a', '#d8c27a', '#5a6a8a'])) for _ in a]))
    elif style == 'blank':
        y = np.arange(H0 + 3.4, h, 3.4)
        if len(y): W.slab(0, w, y - .04, y + .04, 0, .03, pc * .85)
        if r.random() < .8:
            x = r.uniform(.15, .85, int(r.integers(1, 4))) * w; W.slab(x - .07, x + .07, H0, h - .4, .05, .2, pc * .6)
        k = int(r.integers(1, 5))
        for _ in range(k): a, b = r.uniform(.1, .8) * w, H0 + r.uniform(.1, .9) * (h - H0 - 2); glass(W, a, a + .9, b, b + 1.1, .015, r, r.random() < lit * .5, 1, em=.8 * glow)


def shopfront(W, r, h=4.2, em=1., pal=None):
    """lit ground floor: glass, goods, awning, light box; spills light"""
    w = W.w
    if w < 2.5: return
    x = .2
    while x < w - 2:
        q = min(r.uniform(3.5, 7.5), w - x - .2)
        if q < 2: break
        p = pal or pick(r, NEON); c = wcols(r, 1, (.45, .4, .15, 0))[0]
        W.rect(x, x + q, .15, h - 1.1, .05, c * .62, ct=c * 1.02, em=.95 * em, snow=0)
        n = max(2, int(q / .9))                                  # goods, shelves, mannequins as flat dark/bright shapes
        for i in range(n):
            a = x + q * (i + .15) / n; b = a + q / n * r.uniform(.35, .75); t = r.uniform(.3, 1.5)
            W.rect(a, b, .15, .15 + t, .07, c * r.uniform(.12, .5) * np.array([r.uniform(.8, 1.2), r.uniform(.8, 1.1), r.uniform(.7, 1.2)], f32), em=.8 * em, snow=0)
        k = int(r.integers(1, 4)); xs = x + q * r.uniform(.1, .85, k); W.rect(xs, xs + q * .12, h - 1.5, h - 1.38, .075, '#fffdf2', em=1.3 * em, snow=0)
        m = np.linspace(x, x + q, max(2, int(q / 1.6)) + 1); W.slab(m - .04, m + .04, .1, h - 1.1, .05, .13, '#23242c')
        W.slab(x, x + q, h - 1.1, h - .15, 0, .45, C(p[0]) * .5, side=C(p[0]) * .4, em=.4 * em, side_em=.3 * em, snow=0); lightbox(W, x + .06, x + q - .06, h - 1.04, h - .22, .47, r, p, .95 * em, 'bars' if r.random() < .65 else None)
        if r.random() < .35:
            ac = C(pick(r, ['#b8323a', '#2f6a5a', '#c8a24a', '#e6e0d4']))
            for i in range(int(q / .5)): a = x + i * .5; W.S.quads(np.stack([W.p(a, h - 1.15, .45), W.p(a + .5, h - 1.15, .45), W.p(a + .5, h - 1.75, 1.3), W.p(a, h - 1.75, 1.3)])[None], ac if i % 2 else ac * .55 + .35, grp=W.grp)
        x += q + r.uniform(.2, .7)


def blade(S, W, u, v0, v1, out, r, em=1., n=None, d0=.25, grp=None):
    """vertical stack of boxed signs sticking out from a wall at u (both faces dressed)"""
    H_ = v1 - v0; n = n or max(1, int(H_ / r.uniform(1.3, 2.6))); th = .28; g = W.grp if grp is None else grp
    y = v0
    for i in range(n):
        hh = H_ / n * r.uniform(.82, .97); p = pick(r, NEON); c = C(p[0]); e = em if r.random() < .88 else .1
        W.slab(u - th / 2, u + th / 2, y, y + hh, d0, d0 + out, c * .35, side=c * .9, em=.25 * e, side_em=.72 * e, snow=0)
        kind = None if out > .62 * hh and r.random() < .6 else 'bars'
        for sd in (-1, 1):
            F = Wall.frame(S, W.p(u + sd * (th / 2 + .012), y, d0 + out if sd > 0 else d0), -W.n * sd, W.ez, out, hh, g)
            rr = np.random.default_rng(int(r.integers(1 << 30))); graphic(F, out * .1, out * .9, hh * .06, hh * .94, 0, rr, p, kind, em=.85 * e, bg=False)
        y += H_ / n
    W.slab(u - .06, u + .06, v0 - .4, v1 + .3, d0 - .12, d0, '#2a2c34')


def rooftop(S, foot, z, r, det=2, lights=True, grp=0, billboard=0., eye=None):
    """parapet, plant, tanks, ducts, antennas with red obstruction lamps, sign frames"""
    P = np.asarray(foot, float); n = len(P); c = P.mean(0)
    e = np.roll(P, -1, 0) - P; L = np.hypot(*e.T)
    for i in range(n):
        W = Wall(S, P[i], P[(i + 1) % n], z - .1, z + 1.1); W.slab(0, W.w, 0, 1.1, -.3, 0, '#6c7080', snow=1.); S.seg([W.p(0., 1.1, .01)], [W.p(W.w, 1.1, .01)], .06, '#8f98c0', em=.10, min_px=1.1) if det else None
    k = int(np.argmax(L)); ex = e[k] / L[k]; ey = np.array([-ex[1], ex[0]]); a0 = math.atan2(ex[1], ex[0])
    sz = min(L.min(), 18)
    def at(u, v): return c + ex * u + ey * v
    near = det >= 2 and sz > 8
    if near:
        for j in range(5):                                             # patched membrane
            a_, b_ = r.uniform(-.45, .3, 2) * sz; q_ = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * r.uniform(.12, .5, 2) * sz + [a_, b_]; pts_ = np.array([at(*v) for v in q_]); S.quads(np.c_[pts_, np.full(4, z + .03 + j * .004)][None], C('#7c8090') * r.uniform(.6, 1.1), tex=1, bias=3e-4)
        nr = int(r.integers(2, 4))
        for j in range(nr):                                            # ranks of condensers on rails
            v_ = r.uniform(-.4, .4) * sz; k_ = int(r.integers(3, 8)); us = r.uniform(-.42, 0) * sz + np.arange(k_) * 1.25; us = us[us < .42 * sz]; pts = np.array([at(u_, v_) for u_ in us])
            if len(pts): S.boxes(np.c_[pts, np.full(len(pts), z + .75)], [1.0, .75, 1.1], a0, '#a4a6ac'); S.boxes(np.c_[pts, np.full(len(pts), z + 1.31)], [.7, .6, .04], a0, '#3a3d48'); S.seg([np.r_[pts[0], z + .18]], [np.r_[pts[-1], z + .18]], .12, '#4a4e5a')
        for j in range(3):                                             # ducts and pipe runs
            u0_, u1_ = np.sort(r.uniform(-.44, .44, 2)) * sz; v_ = r.uniform(-.44, .44) * sz; A_, B_ = at(u0_, v_), at(u1_, v_); S.tube([[*A_, z + .1], [*A_, z + .5], [*B_, z + .5], [*B_, z + .1]], .16 if j else .34, '#8d9098', 6)
        pr = [at(u_, -.46 * sz) for u_ in np.linspace(-.46, .46, 9) * sz]; S.seg([np.r_[q_, z + 1.] for q_ in pr], [np.r_[q_, z + 2.2] for q_ in pr], .06, '#7c8090'); S.seg([np.r_[q_, z + 2.2] for q_ in pr[:-1]], [np.r_[q_, z + 2.2] for q_ in pr[1:]], .05, '#7c8090'); S.seg([np.r_[q_, z + 1.6] for q_ in pr[:-1]], [np.r_[q_, z + 1.6] for q_ in pr[1:]], .04, '#7c8090')
        p_ = at(.2 * sz, .22 * sz); S.boxes([*p_, z + 1.5], [3.4, 2.8, 3.0], a0, '#888c9a', tex=1); W_ = Wall.frame(S, np.r_[at(.2 * sz - .5, .22 * sz - 1.42), z], np.r_[ex, 0], (0, 0, 1), 1., 2.1, grp); W_.rect(0, 1., 0, 2.1, .02, '#4a5266'); W_.rect(.1, .9, 2.25, 2.4, .05, '#ffe9b8', em=1.2, grp=grp); S.light(np.r_[at(.2 * sz, .22 * sz - 2.4), z + 2.2], '#ffe0a8', 9, 2.4, 1.6, grp)
    m = int(r.integers(2, 5)) if det else 1
    for _ in range(m):
        p = at(*(r.uniform(-.3, .3, 2) * sz)); s = (r.uniform(.14, .4) * sz, r.uniform(.12, .3) * sz, r.uniform(1.6, 4.4))
        S.boxes([p[0], p[1], z + s[2] / 2], s, a0, pick(r, ['#5d6272', '#747884', '#4b5062']), tex=6 if r.random() < .5 else 1)
    if det >= 1:
        q = int(r.integers(3, 9)); pts = np.array([at(*(r.uniform(-.42, .42, 2) * sz)) for _ in range(q)]); S.boxes(np.c_[pts, np.full(q, z + .6)], [1.3, .9, 1.2], a0, '#8a8d96');
        if r.random() < .5: p = at(*(r.uniform(-.25, .25, 2) * sz)); S.cyl(p, min(1.6, sz * .12), z + 1.4, z + 3.6, '#8f9099', 12); S.boxes([p[0], p[1], z + .7], [2.2, 2.2, 1.4], a0, '#3c404c')
        if r.random() < .6:
            p = at(*(r.uniform(-.3, .3, 2) * sz)); hh = r.uniform(4, 11); S.seg([[*p, z]], [[*p, z + hh]], .12, '#565a68');
            for t in (.45, .7, .88): S.seg([[p[0] - ex[0] * .8, p[1] - ex[1] * .8, z + hh * t]], [[p[0] + ex[0] * .8, p[1] + ex[1] * .8, z + hh * t]], .07, '#565a68')
            if lights: S.boxes([p[0], p[1], z + hh], [.5, .5, .5], 0, '#ff3a30', em=1.6, grp=grp, snow=0); S.light((p[0], p[1], z + hh + .4), '#ff3a30', 9, 2., .8, grp)
    if billboard > 0 and r.random() < billboard:
        if eye is not None:
            sc = [L[i] * max(0, np.dot((eye[:2] - (P[i] + P[(i + 1) % n]) / 2) / np.hypot(*(eye[:2] - (P[i] + P[(i + 1) % n]) / 2)), [e[i][1] / L[i], -e[i][0] / L[i]])) for i in range(n)]; k = int(np.argmax(sc))
        W = Wall(S, P[k], P[(k + 1) % n], z + 1.2, z + 1.2 + min(7., L[k] * .4), grp); hh = W.h; p = pick(r, NEON)
        x = np.linspace(.5, W.w - .5, 6); S.seg(W.p(x, 0, -.4), W.p(x, hh, -.4), .16, '#2c2f3a'); S.seg(W.p(x, 0, -2.6), W.p(x, hh * .8, -.4), .12, '#2c2f3a')
        W.slab(.2, W.w - .2, .3, hh, -.35, -.1, '#14151c', side='#23252e', snow=0); lightbox(W, .3, W.w - .3, .4, hh - .1, -.07, r, p, .95); S.light(W.p(W.w / 2, hh / 2, 5), C(p[0]) * .6 + .3, 26, 7, .9, grp)


def building(S, foot, z1, style, r, eye, f=1200., lit=.5, pal='#6a6f82', shop=0., roof=True, grp=0, z0=0., bill=0., trim=None, glow=1., blades=0, side_style=None, detmax=2):
    """one building: dress every wall that can be seen from `eye`"""
    P = np.asarray(foot, float); n = len(P); eye = np.asarray(eye, float)
    c = P.mean(0); dist = max(4., float(np.hypot(*(c - eye[:2])) - np.hypot(*(P - c).T).max())); ppm = f / dist
    det = min(detmax, 2 if ppm > 9 else 1 if ppm > 3.2 else 0)
    ws = []
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]; W = Wall(S, a, b, z0, z1, grp)
        if np.dot(eye - W.o, W.n) <= 0.02:
            continue
        st = style if (W.w > 5.5 or side_style is None) else side_style
        facade(W, st, r, lit, pal, det, shop, trim, glow=glow); ws.append(W)
        if shop and det >= 1: shopfront(Wall(S, a, b, z0, z0 + shop, grp), r, shop, glow)
        if blades and det >= 1 and W.w > 3:
            for _ in range(int(blades)): u = pick(r, [.35, W.w - .35, W.w * r.uniform(.2, .8)]); v = shop + r.uniform(.5, 2); blade(S, W, u, v, min(z1 - z0 - .5, v + r.uniform(.3, .85) * (z1 - z0 - v)), r.uniform(.8, 1.5), r, glow)
    S.poly(np.c_[P, np.full(n, z1)], C(pal) * .8, tex=1)
    if roof: rooftop(S, foot, z1, r, det, True, grp, bill, eye)
    return ws


def lot(a, b, depth):
    """CCW footprint from street edge a->b (street on the right hand) going `depth` to the left"""
    a, b = np.asarray(a, float), np.asarray(b, float); d = (b - a) / np.hypot(*(b - a)); L = np.array([-d[1], d[0]])
    return np.array([a, b, b + L * depth, a + L * depth])


# ------------------------------------------------------------------------------------------------ people, cars, street furniture
def crowd(S, pts, cam, r, z=0., head=1.7, dark=.5, pal=None, umb=0., grp=0, rimc=None, walk=1.):
    """drawn walkers as camera-facing cut-outs (head, hair, coat, stride, bag).  pts (n,2)"""
    pts = np.asarray(pts, float); n = len(pts)
    if n == 0: return
    pal = pal or ['#1a1c28', '#23222e', '#2e2a34', '#15161f', '#3a3440', '#4a3c3c', '#26323e', '#6a5a50', '#8a8078', '#52303a', '#1f3a4a']
    zz = np.broadcast_to(z, (n,)).astype(float)
    R = cam.R[0].copy(); R[2] = 0; R /= np.linalg.norm(R); U = np.array([0, 0, 1.])
    H_ = head * r.normal(1, .06, n); bw = r.uniform(.9, 1.15, n)
    O = np.c_[pts, zz]; side = r.choice([-1, 1], n); ph = r.uniform(0, 2 * np.pi, n); st = .20 * np.sin(ph) * walk
    top = np.stack([C(pick(r, pal)) for _ in range(n)]); bot = np.stack([C(pick(r, pal)) for _ in range(n)]) * .8; skin = np.stack([C(pick(r, ['#c99a7e', '#b98a70', '#d8ab8c'])) for _ in range(n)]) * .75; hair = np.stack([C(pick(r, ['#14121a', '#1d1820', '#2c2024', '#4a3226', '#7a5a3a'])) for _ in range(n)])
    to_eye = cam.eye - O; to_eye[:, 2] = 0; to_eye /= np.linalg.norm(to_eye, axis=1, keepdims=True)
    lab = S.newlab(n)
    def q(pts2, col, k=0, gain=(1., 1.)):
        """pts2: list of 4 (x,y) arrays in body units (x shoulder-widths, y fraction of height)"""
        P = np.stack([O + R * ((np.broadcast_to(px, (n,)) * bw * H_ * .30))[:, None] + U * (np.broadcast_to(py, (n,)) * H_)[:, None] + to_eye * .015 * k for px, py in pts2], 1)
        c = np.stack([col * gain[0], col * gain[0], col * gain[1], col * gain[1]], 1) if len(pts2) == 4 else col
        S.quads(P, c, lab=lab, snow=.4, grp=grp, d=(0, 0, 1))
    s = st
    q([(-.10 + s - .16, 0.), (-.10 + s + .12, 0.), (.16, .50), (-.20, .50)], bot, 0, (.8, 1.))
    q([(.10 - s - .12, 0.), (.10 - s + .16, 0.), (.22, .50), (-.14, .50)], bot * .85, 1, (.8, 1.))
    long = r.random(n) < .35; hem = np.where(long, .30, .46)
    q([(-.42, hem), (.42, hem), (.46, .80), (-.46, .80)], top, 2, (.85, 1.1))
    q([(-.46, .80), (.46, .80), (.18, .865), (-.18, .865)], top, 2, (1.1, 1.25))
    q([(-.56 + s * .6, .46), (-.40 + s * .6, .46), (-.38, .80), (-.52, .79)], top * .9, 3, (.8, 1.)); q([(.40 - s * .6, .46), (.56 - s * .6, .46), (.52, .79), (.38, .80)], top * .9, 3, (.8, 1.))
    q([(-.17, .85), (.17, .85), (.20, .95), (-.20, .95)], skin, 3); q([(-.23, .90), (.23, .90), (.17, 1.005), (-.17, 1.005)], hair, 2, (.9, 1.15))
    back = r.random(n) < .55                                           # turned away: hair covers the head
    q([(np.where(back, -.22, -.24), np.where(back, .87, .955)), (np.where(back, .22, .24), np.where(back, .87, .955)), (.16, 1.01), (-.16, 1.01)], hair, 5, (.9, 1.2))
    bag = r.random(n) < .4; bx = side * .62
    q([(np.where(bag, bx - .18, 0), np.where(bag, .26, .5)), (np.where(bag, bx + .18, 0), np.where(bag, .26, .5)), (np.where(bag, bx + .15, 0), np.where(bag, .44, .5)), (np.where(bag, bx - .15, 0), np.where(bag, .44, .5))], top[::-1] * 1.3, 5)
    if umb > 0:
        k = np.nonzero(r.random(n) < umb)[0]
        for i in k:
            c = O[i] + U * (H_[i] * 1.22); col = C(pick(r, ['#c8d4e0', '#1a1c26', '#8a2a36', '#2a4a6a', '#d8d2c0'])); a = np.linspace(0, 2 * np.pi, 9)[:-1]; ring = c + np.c_[np.cos(a) * .55, np.sin(a) * .55, np.full(8, -.20)]
            for j in range(8): S.tris(np.array([[c, ring[j], ring[(j + 1) % 8]]]), col * (.8 + .25 * (j % 2)), grp=grp)
            S.seg([c], [O[i] + U * H_[i] * .6], .02, '#30323a')
    a = np.linspace(0, 2 * np.pi, 9)[:-1]                               # contact shadow
    for j in range(0, 8, 2):
        e = np.stack([np.c_[pts[:, 0] + np.cos(a[(j + t) % 8]) * .42, pts[:, 1] + np.sin(a[(j + t) % 8]) * .24, zz + .015] for t in (0, 1, 2)], 1)
        e = np.concatenate([np.c_[pts, zz + .015][:, None], e], 1); S.quads(e, np.broadcast_to(C('#07070c'), (n, 3)), snow=0, grp=grp, bias=2e-4)


CARS = {'taxi': ('#1c2a22', 1), 'black': ('#16171d', 0), 'white': ('#c8cad0', 0), 'red': ('#7a1c22', 0), 'van': ('#aeb2ba', 0), 'blue': ('#24344e', 0), 'yellow': ('#c89a24', 1)}


def car(S, x, y, ang, r, kind='black', z=0., lights=1., grp=0, wreck=0., axes=None):
    """saloon / van / bus built from a drawn side profile"""
    col = C(CARS.get(kind, ('#444', 0))[0]) if kind != 'bus' else C('#c9c3b4'); van = kind in ('van', 'bus')
    L, Wd, Hh = (4.5, 1.72, 1.45) if not van else ((4.8, 1.8, 2.0) if kind == 'van' else (10.5, 2.4, 3.0))
    ca, sa = math.cos(ang), math.sin(ang); ax = np.array([[ca, sa, 0], [-sa, ca, 0], [0, 0, 1.]]) if axes is None else np.asarray(axes, float)
    o = np.array([x, y, z], float)
    def P(u, v, w): return o + ax[0] * u + ax[1] * v + ax[2] * w
    if not van: prof = [(-.5, .28), (-.5, .55), (-.48, .64), (-.30, .66), (-.19, .98), (.16, 1.), (.32, .68), (.46, .60), (.5, .50), (.5, .26)]
    else: prof = [(-.5, .16), (-.5, .93), (-.47, 1.), (.40, 1.), (.46, .62 if kind == 'van' else .5), (.5, .42), (.5, .16)]
    pr = np.array(prof) * [L, Hh]
    if wreck: pr = pr + r.normal(0, .10 * wreck, pr.shape) * [1, .8]
    tw = np.where(pr[:, 1] > Hh * .7, .80, 1.) if not van else np.where(pr[:, 1] > Hh * .9, .94, 1.)
    A = np.array([P(u, -Wd / 2 * t, w) for (u, w), t in zip(pr, tw)]); B = np.array([P(u, Wd / 2 * t, w) for (u, w), t in zip(pr, tw)]); n = len(pr)
    for i in range(n - 1): S.quads(np.stack([A[i], B[i], B[i + 1], A[i + 1]])[None], col * (1.12 if pr[i + 1, 1] >= Hh * .6 else .9), d=ax[0], grp=grp, snow=1.)
    for E in (A, B):
        k = [i for i in range(n)]; c0 = E.mean(0); S.tris(np.array([[c0, E[i], E[(i + 1) % n]] for i in k]), col, d=ax[0], grp=grp)
    g = C('#0c1018') if not wreck else C('#050507')
    if not van:
        for s in (-1, 1):
            w = [(-.17, .70), (-.145, .93), (.13, .94), (.245, .70)]; S.quads(np.array([P(u * L, s * (Wd / 2 * (.905 if h_ < .8 else .815) + .02), h_ * Hh) for u, h_ in w])[None], g, em=.2, grp=grp, snow=0)
        for (u0, h0, u1, h1) in ((-.285, .70, -.195, .955), (.305, .71, .17, .965)): S.quads(np.array([P(u0 * L + np.sign(u0) * .03, -Wd * .40, h0 * Hh + .02), P(u0 * L + np.sign(u0) * .03, Wd * .40, h0 * Hh + .02), P(u1 * L + np.sign(u0) * .03, Wd * .34, h1 * Hh + .02), P(u1 * L + np.sign(u0) * .03, -Wd * .34, h1 * Hh + .02)])[None], g * 1.2, em=.2, grp=grp, snow=0)
    else:
        for s in (-1, 1):
            nw = 3 if kind == 'van' else 7
            for i in range(nw): u0 = -.44 + .84 * i / nw; S.quads(np.array([P(u0 * L, s * (Wd / 2 + .015), .58 * Hh), P((u0 + .84 / nw - .02) * L, s * (Wd / 2 + .015), .58 * Hh), P((u0 + .84 / nw - .02) * L, s * (Wd / 2 * .985 + .015), .86 * Hh), P(u0 * L, s * (Wd / 2 * .985 + .015), .86 * Hh)])[None], '#ffe2b0' if kind == 'bus' and lights else g, em=.7 * lights if kind == 'bus' else .2, grp=grp, snow=0)
        S.quads(np.array([P(.44 * L, -Wd * .45, .64 * Hh), P(.44 * L, Wd * .45, .64 * Hh), P(.405 * L, Wd * .44, .95 * Hh), P(.405 * L, -Wd * .44, .95 * Hh)])[None] + ax[0] * .06, g * 1.3, em=.25, grp=grp, snow=0)
        S.quads(np.array([P(-.5 * L - .02, -Wd * .4, .55 * Hh), P(-.5 * L - .02, Wd * .4, .55 * Hh), P(-.5 * L - .02, Wd * .4, .88 * Hh), P(-.5 * L - .02, -Wd * .4, .88 * Hh)])[None], g * 1.3, em=.25, grp=grp, snow=0)
    for u in (-.31, .31):                                              # wheels
        for s in (-1, 1):
            a = np.linspace(0, 2 * np.pi, 11)[:-1]; c_ = P(u * L, s * (Wd / 2 + .02), .33 if not van else .4); R_ = .33 if not van else .42
            S.poly(c_ + np.outer(np.cos(a) * R_, ax[0]) + np.outer(np.sin(a) * R_, ax[2]), '#0a0a0d', grp=grp, snow=0); S.poly(c_ + ax[1] * s * .01 + np.outer(np.cos(a) * R_ * .5, ax[0]) + np.outer(np.sin(a) * R_ * .5, ax[2]), '#6a6d76', grp=grp, snow=0)
    fr = .50 * L
    for s in (-.36, .36):
        for (u, c_, h_, e) in ((fr + .02, '#fff4d4', .50 if not van else .34, 1.9), (-.5 * L - .02, '#e01822', .56 if not van else .34, 1.2)):
            S.quads(np.array([P(u, (s - .10) * Wd, h_ * Hh - .07), P(u, (s + .10) * Wd, h_ * Hh - .07), P(u, (s + .10) * Wd, h_ * Hh + .07), P(u, (s - .10) * Wd, h_ * Hh + .07)])[None], c_ if lights else C(c_) * .25, em=e * lights, grp=grp, snow=0, bias=3e-4)
    if lights and not wreck:
        p = P(fr + 1.2, 0, .5); S.light(p, '#ffe9c0', 13, 3.5, .9 * lights, grp, dir=ax[0] - ax[2] * .16, cos=.80, soft=.8); p = P(-.5 * L - .4, 0, .5); S.light(p, '#e02030', 4.5, 1.2, .8 * lights, grp)
    if CARS.get(kind, ('', 0))[1]: S.boxes(P(-.02 * L, 0, Hh + .10), [.5, .26, .2], ang, '#ffdf9a', em=1.1 * lights, grp=grp, snow=0)


def signal(S, base, arm, h=6., r=None, col='red', grp=0, z=0., heads=1, face=(0, -1), k=1.):
    """mast-arm traffic light; lamps: red / amber / green row like Japanese signals"""
    b = np.asarray(base, float); a = np.asarray(arm, float); S.tube([[*b, z], [*b, z + h]], [.13, .10], '#8a8e98', 7); S.tube([[*b, z + h - .35], [*(b + a * .5), z + h + .25], [*(b + a), z + h + .25]], .06, '#8a8e98', 6)
    fx = np.asarray(face, float); fx /= np.hypot(*fx); n3 = np.array([fx[0], fx[1], 0]); ex = np.cross([0, 0, 1.], n3);
    for q in range(heads):
        p = b + a * (1 - .5 * q); o = np.array([p[0], p[1], z + h - .42]) - ex * .62 * k
        W = Wall.frame(S, o, ex, (0, 0, 1), 1.24, .42, grp); W.slab(0, 1.24 * k, 0, .42 * k, -.22, 0, '#3c4048', snow=1.)
        for i, (c, nm) in enumerate((('#22d28a', 'green'), ('#ffb43a', 'amber'), ('#ff2a24', 'red'))):
            on = nm == col; W.disc((.22 + i * .40) * k, .21 * k, .155 * k, .02, c if on else C(c) * .16, 14, em=2.4 if on else .4, snow=0); W.slab((.06 + i * .40) * k, (.38 + i * .40) * k, .36 * k, .39 * k, 0, .24, '#2a2d34')
            if on: S.light(W.p((.22 + i * .4) * k, .21 * k, .6), c, 18, 4., 1.6 * k, grp)


def lamp(S, p, h=8., arm=(1.6, 0), col='#ffe0a8', grp=0, z=0., amt=2.6, rad=22, em=1.8, pole='#6f737e'):
    x, y = p; S.tube([[x, y, z], [x, y, z + h * .86], [x + arm[0] * .4, y + arm[1] * .4, z + h * .985], [x + arm[0], y + arm[1], z + h]], [.12, .08, .06, .05], pole, 7, grp=grp)
    S.boxes([x + arm[0], y + arm[1], z + h - .06], [.85, .42, .16], math.atan2(arm[1], arm[0]), '#4a4e58', grp=grp); S.boxes([x + arm[0], y + arm[1], z + h - .17], [.72, .34, .08], math.atan2(arm[1], arm[0]), col, em=em, grp=grp, snow=0)
    S.light((x + arm[0], y + arm[1], z + h - .4), col, rad, rad * .28, amt, grp)


def pole(S, p, h=11., r=None, z=0., arms=((-1, 0),), col='#7d7a78', xf=True):
    """Japanese concrete utility pole: cross-arms, insulators, transformer can, step bolts.  returns wire anchor points"""
    x, y = p; S.tube([[x, y, z], [x, y, z + h]], [.19, .13], col, 9, tex=1); out = []
    ax = np.array([arms[0][0], arms[0][1]], float); ax /= np.hypot(*ax)
    for k, (zz, hw) in enumerate(((h - .5, 1.15), (h - 1.5, .9))):
        a = np.array([x, y, z + zz]) - np.r_[ax, 0] * hw; b = np.array([x, y, z + zz]) + np.r_[ax, 0] * hw; S.tube([a, b], .055, '#5c5a5c', 5)
        for t in (-.95, -.4, .4, .95): q = np.array([x, y, z + zz]) + np.r_[ax, 0] * hw * t; S.boxes(q + [0, 0, .12], [.10, .10, .2], 0, '#d8d2c4'); out.append(q + [0, 0, .22])
    if xf: pp = np.array([x, y]) + np.array([-ax[1], ax[0]]) * .42; S.cyl(pp, .30, z + h - 3.6, z + h - 2.5, '#8f9098', 10); S.boxes([pp[0], pp[1], z + h - 3.1], [.75, .1, .12], math.atan2(ax[1], ax[0]), '#54545a')
    S.boxes([x, y, z + h * .46], [.42, .42, .9], 0, '#9a9690') if r is not None and r.random() < .5 else None
    for i in range(6): S.seg([[x - .28, y, z + 2.4 + i * .7]], [[x + .28, y, z + 2.4 + i * .7]], .03, '#4a4a4e')
    return out


def rebar(S, p0, r, n=6, spread=.4, L=(.5, 1.6), dir=(0, 0, 1), col='#4a2c22'):
    """bent rods out of a broken edge"""
    p0 = np.atleast_2d(np.asarray(p0, float)); d0 = np.asarray(dir, float)
    for k in range(n):
        p = p0[int(r.integers(len(p0)))] + r.normal(0, .05, 3); d = d0 + r.normal(0, spread, 3); d /= np.linalg.norm(d); ln = r.uniform(*L); m = p + d * ln * .6; e = m + (d * .6 + r.normal(0, .5, 3)) * ln * .4
        S.seg([p, m], [m, e], .035, col, min_px=1.0)


_CUBE = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)


def rock(S, c, size, r, col='#7a7a80', chips=2, tex=9, **k):
    """one faceted lump: a jittered box with cut corners (every one different)"""
    from scipy.spatial import ConvexHull
    n = 9 + chips * 2; P = r.normal(0, 1, (n, 3)); P /= np.linalg.norm(P, axis=1, keepdims=True); P = np.vstack([P * r.uniform(.8, 1., (n, 1)), _CUBE * r.uniform(.42, .66)])
    P = P * np.asarray(size, float) / 2
    a = r.uniform(0, 2 * np.pi, 3); from .cam import rot_x, rot_y, rot_z
    P = P @ (rot_z(a[0]) @ rot_x(a[1] * .18) @ rot_y(a[2] * .18)).T + np.asarray(c, float)
    h = ConvexHull(P); T = P[h.simplices]; nrm = h.equations[:, :3]; key = np.round(nrm * 5).astype(int); _, lab = np.unique(key, axis=0, return_inverse=True)
    S.tris(T, C(col) * r.uniform(.82, 1.15), lab=S.newlab(int(lab.max()) + 2)[0] + lab.ravel(), tex=tex, **k)


def rubble(S, pts, sizes, r, z=0., cols=('#77777c', '#6a6468', '#8a8580', '#5c5a60', '#7d6a60'), slabs=.35, bars=.15):
    """pile of broken concrete: lumps and tilted slabs; pts (n,2), sizes metres"""
    from .cam import rot_x, rot_y, rot_z
    z = np.broadcast_to(z, (len(pts),))
    for (x, y), s, zz in zip(pts, sizes, z):
        c = pick(r, cols)
        if r.random() < slabs and s > .5:
            A = rot_z(r.uniform(0, 6.28)) @ rot_x(r.normal(0, .5)) @ rot_y(r.normal(0, .35)); L = np.array([s * r.uniform(1, 1.9), s * r.uniform(.6, 1.2), s * r.uniform(.14, .28)])
            poly = np.array([(-.5, -.5), (.5 * r.uniform(.6, 1), -.5), (.5, .5 * r.uniform(.2, 1)), (.2 * r.uniform(0, 1), .5), (-.5, .5 * r.uniform(.5, 1))]) * L[:2]
            lo = np.c_[poly, np.full(5, -L[2] / 2)] @ A.T + [x, y, zz + s * .25]; hi = np.c_[poly, np.full(5, L[2] / 2)] @ A.T + [x, y, zz + s * .25]
            S.poly(hi, c, tex=1); S.poly(lo, c, tex=9)
            for i in range(5): S.quads(np.stack([lo[i], lo[(i + 1) % 5], hi[(i + 1) % 5], hi[i]])[None], C(c) * .82, tex=9)
            if r.random() < bars * 2: rebar(S, hi[:3], r, 4, .5, (.3 * s, .9 * s), A @ [1, 0, .3])
        else: rock(S, (x, y, zz + s * .18), (s * r.uniform(.8, 1.3), s * r.uniform(.8, 1.3), s * r.uniform(.5, .95)), r, c)


def tree(S, p, h, r, z=0., cols=('#2f4a34', '#3f6040', '#23382c'), crown=.42, n=60, trunk='#4a3a34', leaf=.26):
    """street tree: forked trunk and clumps of leaf cards (hundreds of small masked dabs)"""
    x, y = p; t = h * .42; S.tube([[x, y, z], [x + r.normal(0, .1), y, z + t * .6], [x + r.normal(0, .2), y + r.normal(0, .2), z + t]], [h * .030, h * .024, h * .018], trunk, 6)
    top = np.array([x, y, z + t]); R = h * crown
    cl = []
    for k in range(5):
        a = r.uniform(0, 6.28); e = top + [math.cos(a) * R * .7, math.sin(a) * R * .7, R * r.uniform(.3, .9)]; S.tube([top - [0, 0, .2], (top + e) / 2 + [0, 0, .1 * R], e], [h * .016, h * .01, h * .006], trunk, 5); cl.append((e, R * r.uniform(.42, .62)))
    cl += [(top + [0, 0, R * 1.0], R * .6), (top + [r.normal(0, .3) * R, r.normal(0, .3) * R, R * .65], R * .65)]
    for (c, q) in cl:
        m = int(n * 5); d = r.normal(0, 1, (m, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True); P = c + d * (q * r.uniform(.55, 1., (m, 1)) ** .6) * [1, 1, .72]
        s = leaf * h * .09 * r.uniform(.6, 1.4, m); u = np.cross(d, r.normal(0, 1, (m, 3))); u /= np.linalg.norm(u, axis=1, keepdims=True); v = np.cross(d, u); u = u * s[:, None]; v = v * (s * .55)[:, None]
        lev = np.clip(.5 + .5 * d[:, 2] + r.normal(0, .18, m), 0, 1); cc = np.where(lev[:, None] > .72, C(cols[1]), np.where(lev[:, None] > .36, C(cols[0]), C(cols[2]))) * r.uniform(.86, 1.14, (m, 1))
        S.quads(np.stack([P - u, P - v, P + u, P + v], 1), cc.astype(f32), lab=S.newlab(m), d=(0, 0, 1))


def zebra(S, p0, p1, wid, col='#d9d7d0', stripe=.45, gap=.45, z=.02, skip=0., r=None, tex=8):
    """crossing from p0 to p1 (walking direction), bars run across it"""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float); d = p1 - p0; L = np.hypot(*d); d /= L; n = np.array([-d[1], d[0]]); k = int(L / (stripe + gap)); t = np.arange(k) * (stripe + gap)
    if skip and r is not None: t = t[r.random(k) > skip]
    a = p0 + d * t[:, None]; b = a + d * stripe; w = n * wid / 2; zc = np.full(len(t), z)
    q = np.stack([np.c_[a - w, zc], np.c_[b - w, zc], np.c_[b + w, zc], np.c_[a + w, zc]], 1)
    S.quads(q, col, tex=tex, bias=4e-4, d=(n[0], n[1], 0), lab=S.newlab(len(t)))


def strip(S, P, wid, col='#d4d2ca', z=.02, tex=8, dash=None, **k):
    """painted road line along a polyline (n,2)"""
    P = np.asarray(P, float)
    for a, b in zip(P[:-1], P[1:]):
        d = b - a; L = np.hypot(*d); d /= L; n = np.array([-d[1], d[0]]) * wid / 2; m = max(1, int(L / 3)) if dash is None else 0
        ts = np.linspace(0, L, m + 1) if dash is None else None
        if dash is None: A = a + d * ts[:-1, None]; B = a + d * ts[1:, None]
        else: t = np.arange(0, L - dash[0], dash[0] + dash[1]); A = a + d * t[:, None]; B = A + d * dash[0]
        zc = np.full(len(A), z); S.quads(np.stack([np.c_[A - n, zc], np.c_[B - n, zc], np.c_[B + n, zc], np.c_[A + n, zc]], 1), col, tex=tex, bias=4e-4, d=(d[0], d[1], 0), **k)


def kerb(S, P, h=.16, w=.30, col='#8a8c94', z=0.):
    """granite kerb along a polyline; pavement lies on its left"""
    P = np.asarray(P, float)
    for a, b in zip(P[:-1], P[1:]):
        W = Wall(S, a, b, z, z + h); W.slab(0, W.w, 0, h, -w, 0, col, top=C(col) * 1.15, tex=1)
