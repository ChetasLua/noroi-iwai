"""fx ink: ink burst, ink run / curtain, ink pool, brush splash, paper burn."""
from .fx_core import *

INK = '#0c0a0d'; BLOODRED = '#b0101f'


def ink_burst(cel, x, y, r, t=1., seed=0, col=INK, n=26, drops=40, ang=None, aniso=0., core=.34, dry=True):
    """thrown ink: one closed splash (bulbed rays, webs between them) that snaps out in the first third, satellites, dry scratches"""
    e = 1 - (1 - min(1., t * 2.2)) ** 3; P = []; a = np.linspace(0, 6.283, 90, endpoint=False)
    body = r * core * (.5 + .5 * e) * (.72 + .16 * noise(np.cos(a) * 1.8, np.sin(a) * 1.8, 0, seed) + .10 * noise(np.cos(a) * 6, np.sin(a) * 6, 0, seed + 1)); cel.fill(np.c_[x + np.cos(a) * body, y + np.sin(a) * body], col)
    for i in range(n):
        a = (i + .8 * (h1(seed, i, 1) - .5)) / n * 6.283; z = bms(h1(seed, i, 2), .2, .3); g = 1 + (aniso * max(-.5, math.cos(a - ang)) if ang is not None else 0.); l = r * (core + (1 - core) * z * e) * g; w = r * (.030 + .045 * h1(seed, i, 3)); d = np.array([math.cos(a), math.sin(a)]); m = np.array([-d[1], d[0]]); q = r * core * .45; bow = r * .05 * (h1(seed, i, 7) - .5)
        v = np.linspace(0, 1, 16); hw = w * (1 - v) ** 1.6 + w * .10; c = d * (q + (l - q) * v)[:, None] + m * (bow * v * v)[:, None]; cel.fill(np.vstack([c + m * hw[:, None], (c - m * hw[:, None])[::-1]]) + [x, y], col)
        bl = w * (.16 + .42 * h1(seed, i, 5)); th = np.linspace(0, 6.283, 14, endpoint=False); st = 1 + .9 * h1(seed, i, 6); Q = c[-1] + d * (np.cos(th) * bl * st)[:, None] + m * (np.sin(th) * bl * (1 + .12 * np.sin(th * 3 + i)))[:, None]; cel.fill(Q + [x, y], col)
    for i in range(drops):
        a = h1(seed, i, 21) * 6.283 if ang is None or h1(seed, i, 20) > aniso * .6 else ang + (h1(seed, i, 21) - .5) * 1.4; d = r * (.5 + 1.1 * h1(seed, i, 22) ** .7) * e; s = r * .050 * h1(seed, i, 23) ** 3 + 1.; th = np.linspace(0, 6.283, 10, endpoint=False); st = 1 + 1.8 * h1(seed, i, 24); Q = np.c_[np.cos(th) * s * st * (1 + .2 * np.sin(th * 3 + i)), np.sin(th) * s]; c, q = math.cos(a), math.sin(a); cel.fill(spl(np.c_[Q[:, 0] * c - Q[:, 1] * q + x + c * d, Q[:, 0] * q + Q[:, 1] * c + y + q * d], True, n=3), col)
    if dry:
        for i in range(18):
            a = h1(seed, i, 31) * 6.283; d0 = r * (.4 + .4 * h1(seed, i, 32)); d1 = d0 + r * (.2 + .6 * h1(seed, i, 33)) * e; c, q = math.cos(a), math.sin(a); o = np.array([-q, c]) * r * .02 * (h1(seed, i, 34) - .5); thin(cel, np.array([[x + c * d0, y + q * d0], [x + c * (d0 + d1) / 2 + o[0] * 3, y + q * (d0 + d1) / 2 + o[1] * 3], [x + c * d1, y + q * d1]]), r * .006 + .8, col, .9, .1, .7, i)
    return cel


def ink_runs(cel, top, lengths, t, seed=0, col=INK, w=(3, 11), wob=3., gleam=None, stagger=.5):
    """drips running down from a line.  top = [(x, y)], lengths px (list or scalar), t 0..1; each with a bulb head, thin neck, its own start and speed"""
    n = len(top)
    for i, (x, y) in enumerate(top):
        L = lengths[i] if np.ndim(lengths) else lengths * (.25 + .9 * bms(h1(seed, i, 1), .25, .35)); q = min(1., max(0., (t - stagger * h1(seed, i, 2)) / max(1e-3, 1 - stagger * h1(seed, i, 2)))); l = L * (1 - (1 - q) ** (1.4 + 1.2 * h1(seed, i, 3))); ww = w[0] + (w[1] - w[0]) * h1(seed, i, 4) ** 1.5
        if l < 2: continue
        v = np.linspace(0, 1, max(10, int(l / 9))); cx = x + wob * noise(v * l / 46., i * 5., 0, seed); neck = ww * (.30 + .12 * noise(v * l / 30., i, 2, seed)); root = ww * 1.5 * np.exp(-v * l / (ww * 2.2)); head = ww * .62 * np.exp(-((1 - v) * l / (ww * .95)) ** 2); hw = neck + root + head; hw[-1] *= .55; yy = y + v * l
        tip = np.array([[cx[-1] + ww * .30, yy[-1] + ww * .42], [cx[-1], yy[-1] + ww * .62], [cx[-1] - ww * .30, yy[-1] + ww * .42]]); cel.fill(np.vstack([np.c_[cx + hw, yy], tip, np.c_[cx - hw, yy][::-1]]), col)
        if gleam and ww > 5: thin(cel, np.c_[cx[-4:] - ww * .28, yy[-4:] - ww * .1], ww * .22, gleam, .8, .4, .4, i, 0, 8)
    return cel


def ink_curtain(cel, t, seed=0, col=INK, y0=-40., y1=H + 60., n=34, x0=-60., x1=W + 60., shape=None, edge=None, rag=.35):
    """ink wall running down the frame (the veil): one closed front of broad fingers, narrow runners ahead of it.  t 0..1; shape(u) -> extra delay 0..1 across x"""
    N = 480; u = np.linspace(0, 1, N); x = x0 + (x1 - x0) * u; e = t * (1.35 + rag); base = np.zeros(N)
    for i in range(n):
        c = (i + .5 + .8 * (h1(seed, i, 1) - .5)) / n; w = (.35 + 1.5 * bms(h1(seed, i, 2), .2, .4)) / n; sp = .55 + .75 * h1(seed, i, 3) ** 1.3; d = np.abs(u - c) / w; base = np.maximum(base, sp * np.exp(-d ** 2.6) * (1.0))
    dl = np.array([shape(q) for q in u]) if shape is not None else 0.; k = np.clip(e - dl, 0, 2); y = y0 + (y1 - y0) * (k ** 1.25) * (.50 + .50 * base) + 8 * noise(u * 40, t * 2, 0, seed); P = np.vstack([[x0, y0 - 200], np.c_[x, y], [x1, y0 - 200]])
    if edge: cel.fill(P + [0, 5], edge)
    cel.fill(P, col); top = []; ln = []
    for i in range(int(n * 1.6)):
        q = h1(seed, i, 11); j = int(q * (N - 1)); top.append((x[j], y[j] - 14)); ln.append(260 * k[j] * h1(seed, i, 12) ** 2 if np.ndim(k) else 260 * k * h1(seed, i, 12) ** 2)
    return ink_runs(cel, top, ln, 1., seed + 3, col, (2, 8))


def ink_pool(cel, x, y, rx, ry, t, seed=0, col=INK, fingers=9, rim=None, wet=None):
    """ink spreading on a surface seen in perspective: creeping lobes, fingers running ahead, a wet highlight"""
    e = 1 - (1 - min(1., max(0., t))) ** 2; a = np.linspace(0, 6.283, 180, endpoint=False); rad = .62 + .10 * noise(np.cos(a) * 1.6, np.sin(a) * 1.6, t * .6, seed) + .06 * noise(np.cos(a) * 4, np.sin(a) * 4, t, seed + 1)
    for i in range(fingers):
        c = h1(seed, i, 1) * 6.283; w = .06 + .10 * h1(seed, i, 2); d = np.abs(np.arctan2(np.sin(a - c), np.cos(a - c))); rad += (.18 + .30 * bms(h1(seed, i, 3), .3, .3)) * e * np.exp(-(d / w) ** 2)
    P = np.c_[x + np.cos(a) * rad * rx * e, y + np.sin(a) * rad * ry * e]
    if rim: cel.fill(np.array([x, y]) + (P - [x, y]) * 1.02 + [0, 2.5], rim)
    cel.fill(P, col)
    if wet: thin(cel, np.c_[x + np.cos(a[100:150]) * rad[100:150] * rx * e * .8, y + np.sin(a[100:150]) * rad[100:150] * ry * e * .8], max(1.5, ry * .06 * e), wet, .8, .4, .5, seed)
    return P


def ink_wash(layer_h=H, layer_w=W, P=None, w=120., seed=0, col=INK, dry=.45, n=14):
    """broad dry-brush sweep made of bristle streaks along P (fast vector version): streaks run out at different lengths"""
    cel = Cel(layer_w, layer_h); p = Path(P, 6.); L = p.L
    for i in range(int(n * 6)):
        o = (h1(seed, i, 1) * 2 - 1); z = h1(seed, i, 2); s0 = L * (.02 * h1(seed, i, 3)); s1 = L * (1 - dry * h1(seed, i, 4) ** 1.2 * (.4 + .6 * abs(o))); q = p.sub(s0, s1, lambda v: o * w * .5 * (.85 + .15 * np.sin(v * 3 + i)) + 2 * noise(v * 9, i, 0, seed), 40); thin(cel, q, w * (.02 + .10 * z ** 2), col, 1., .02 + .04 * z, .15 + .4 * h1(seed, i, 5), i, .3, 40)
    return cel.layer(layer_h, layer_w)


def paper_burn(img, t, seed=0, x=W * .55, y=H * .55, under=None, f=0, scale=170., glow='#ff7a1e', char='#1a0f0a', brown='#6a3a1a', width=1.0):
    """a picture burns through like paper from (x, y): brown scorch, black char with a torn bright ember edge, hole shows `under` (RGB / colour).  t 0..1 -> RGB"""
    h, w = img.shape[:2]; gx, gy = grid(h, w); rm = math.hypot(max(x, w - x), max(y, h - y)); d = np.sqrt((gx - x) ** 2 + ((gy - y) * 1.15) ** 2) / rm; n = noise_img(h, w, scale, 0, seed, 4, 2) * .15 + noise_img(h, w, scale * .18, 0, seed + 5, 3, 2) * .016; e = (t * 1.75 - .12) - (d + n); e = e.astype(f32)
    hole = smooth(0, .004, e)[..., None]; k = width; br = smooth(-.16 * k, -.03 * k, e)[..., None]; ch = smooth(-.055 * k, -.028 * k, e + .012 * noise_img(h, w, 14, 0, seed + 9, 2, 2))[..., None]; em = np.exp(-((e + .004 * k) / (.006 * k)) ** 2)[..., None]; em2 = np.exp(-((e + .010 * k) / (.020 * k)) ** 2)[..., None] * (.6 + .4 * noise_img(h, w, 60, f / 24. * 3, seed + 4, 2, 4))[..., None]
    o = img * (1 - br * .85) + (img * C(brown) * 1.1) * br * .85; o = o * (1 - ch) + C(char) * ch; un = 0. if under is None else (C(under) if isinstance(under, str) else under); o = o * (1 - hole) + un * hole; live = min(1., t * 8) * min(1., (1.02 - t) * 6)
    return o + (em * C('#ffe9a0') * 1.2 + np.clip(em2, 0, 1) * C(glow) * .9) * live * (1 if t > 0 else 0)
