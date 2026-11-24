"""Drawn gags for b3: the mouth that opens on skin, slap star, cheek jelly."""
import numpy as np, math
from jj.ink import Cel, spl, resample
from jj.kit import outline
from jj.paint import rgb
from chars.yuji import mixc
A = np.array


def skin_mouth(c, P, o=1., tongue=0., wag=0., s=100., seed=1, skin=('#f2cdb3', '#c9897a'), ink='#17131a', grin=.22, w=5., laugh=0., lips=1.):
    """a toothy mouth on skin.  P(local (n,2)) -> screen; local x -1..1 along the mouth, y up.  o = opening 0..1.2, tongue 0..1.2 sticks out, s = px per local unit"""
    x = np.linspace(-1, 1, 61); cor = grin * np.abs(x) ** 2.2
    env = np.clip(1 - np.abs(x) ** 2.1, 0, 1) ** .62
    up = cor + o * .46 * env * (1 - .16 * np.exp(-(x / .16) ** 2)) + .02; lo = cor - o * .62 * env * (1 + .05 * np.sin(x * 3 + .5)) - .02
    U, L = np.c_[x, up], np.c_[x, lo]; poly = np.vstack([U, L[::-1]])
    # the bulge the mouth makes in the skin: a soft flat shadow on the far side, swollen lips
    lipU = np.c_[x * 1.06, up + .13 * env + .02]; lipL = np.c_[x * 1.06, lo - .17 * env - .02]
    c.fill(P(np.vstack([lipU, lipL[::-1]])), mixc(skin[0], '#d8705e', .30 * lips))
    c.fill(P(np.vstack([np.c_[x[8:-8] * 1.02, (lo - .17 * env)[8:-8]], np.c_[x[8:-8] * .9, (lo - .34 * env - .02)[8:-8]][::-1]])), skin[1], .9)
    if o < .10:
        c.line(P(np.c_[x, (up + lo) / 2]), w * 1.1, ink, taper=(.15, .15), seed=seed); return
    Pp = P(poly); c.fill(Pp, '#3a0a14'); c.clip([Pp])
    # gums and teeth: uneven, sharp, crowded
    r = np.random.default_rng(seed + 5)
    for sg, edge, n, h0 in ((1, up, 9, .30), (-1, lo, 8, .26)):
        gum = np.c_[x, edge - sg * .10 * env]; c.fill(P(np.vstack([np.c_[x, edge + sg * .05], gum[::-1]])), '#a8434e')
        xs = np.linspace(-.90, .90, n + 1) + np.r_[0, r.normal(0, .016, n - 1), 0]
        for a, b in zip(xs[:-1], xs[1:]):
            m = (a + b) / 2; ea, eb, em = [float(np.interp(v, x, edge)) - sg * .07 * float(np.interp(v, x, env)) for v in (a, b, m)]
            h = h0 * (.75 + .45 * r.random()) * (.55 + .45 * float(np.interp(m, x, env))) * min(1., .35 + o * .65); tip = m + r.normal(0, .02)
            T = P(spl([(a, ea), (a + (tip - a) * .45, ea - sg * h * .55), (tip - .012, em - sg * h), (tip + .012, em - sg * h), (b + (tip - b) * .45, eb - sg * h * .55), (b, eb)], n=4))
            c.fill(T, '#fff8ea'); c.fill(P(A([(tip, em - sg * h), (b + (tip - b) * .45, eb - sg * h * .55), (b, eb), (m + (b - m) * .35, em - sg * h * .3)])), '#d9c9bc')
            c.line(T, max(.8, w * .34), ink, taper=(.1, .1), seed=seed)
    c.unclip()
    c.line(P(U), w * 1.15, ink, taper=(.06, .06), press=.35, seed=seed + 1); c.line(P(L), w, ink, taper=(.06, .06), press=.35, seed=seed + 2)
    for q in (-1, 1):                                                 # corners and the skin they pull
        y = grin
        c.line(P(spl([(q * .98, y - .02), (q * 1.08, y + .05), (q * 1.12, y + .16)], n=6)), w * .8, ink, taper=(.1, .6), seed=seed + 3)
        for k, (dx, dy, l) in enumerate(((.10, .20, .20), (.16, .02, .22), (.12, -.18, .18))):
            c.line(P(spl([(q * (1.08 + dx * .3), y + dy * .5), (q * (1.10 + dx + l * .4), y + dy), (q * (1.08 + dx + l), y + dy * 1.5)], n=6)), w * .42, ink, taper=(.3, .6), seed=seed + 4 + k)
    c.line(P(np.c_[x[20:42] * 1.0, (lipL[:, 1] - .03)[20:42]]), w * .45, ink, taper=(.4, .4), seed=seed + 9)
    if tongue > .02:
        tl = 1.35 * tongue; wv = wag
        cx = np.linspace(0, 1, 16); mid = np.c_[.04 + .10 * wv * cx ** 1.5 * 2, -.02 - cx * tl * .95 + .10 * np.sin(cx * 3) * tongue]
        T = np.gradient(mid, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-6; N = np.c_[-T[:, 1], T[:, 0]]; wd = (.36 - .05 * cx) * np.clip(1 - np.maximum(0, cx - .72) ** 2 / .0785, 0, 1) ** .5
        a, b = mid + N * wd[:, None], mid - N * wd[:, None]; Pt = P(np.vstack([a, b[::-1]]))
        c.fill(Pt, '#e0727c'); c.fill(P(np.vstack([mid + N * (wd * .10)[:, None], b[::-1]])), '#b84a5a')
        c.line(P(mid[2:13]), w * .5, '#8c2f42', taper=(.2, .5), seed=seed + 20); outline(c, Pt, w * .95, ink, seed + 21)
        c.fill(P(spl([(a[9, 0] + .05, a[9, 1]), (a[6, 0] + .07, a[6, 1]), (a[6, 0] + .13, a[6, 1]), (a[9, 0] + .10, a[9, 1])], True)), '#ffd2d0', .9)


def star(c, cx, cy, r0, r1, n, seed, fill='#ffffff', ink='#17131a', w=6., asp=1., rot=0.):
    """comic burst: uneven hand-cut spikes"""
    r = np.random.default_rng(seed); a = (np.arange(n) + r.normal(0, .16, n)) / n * 2 * math.pi + rot; P = []
    for i in range(n):
        R = r1 * r.choice([1., .72, .52], p=[.4, .35, .25]) * r.uniform(.88, 1.12); d = math.pi / n * r.uniform(.5, .75)
        for th, q in ((a[i] - d, r0 * r.uniform(.85, 1.1)), (a[i], R), (a[i] + d, r0 * r.uniform(.85, 1.1))): P.append((cx + math.cos(th) * q, cy + math.sin(th) * q * asp))
    P = A(P); c.fill(P, fill)
    if w: outline(c, P, w, ink, seed, press=.4)
    return P


def little_star(c, x, y, r, rot, fill='#ffd84a', ink='#17131a', w=3.5):
    a = np.arange(10) * math.pi / 5 + rot; R = np.where(np.arange(10) % 2 == 0, r, r * .46); P = np.c_[x + np.cos(a) * R, y + np.sin(a) * R]; c.fill(P, fill); outline(c, P, w, ink, int(x))


def jelly(cx, cy, R, amp, phase, k=.035, dx=-1., dy=.15):
    """screen warp: a decaying ripple over the cheek (amp px) along (dx, dy)"""
    def f(P):
        P = np.asarray(P, float); d = P - A([cx, cy]); r = np.hypot(d[..., 0], d[..., 1]); g = np.exp(-(r / R) ** 2) * amp * np.cos(phase - k * r)
        return P + np.stack([g * dx, g * dy], -1)
    return f


def chain(*fs):
    def f(P):
        for q in fs:
            if q is not None: P = q(P)
        return P
    return f


def press(cx, cy, R, depth, dx=-1., dy=0.):
    """screen warp: the hand pushes the cheek in (depth px), flesh piles up round it"""
    def f(P):
        P = np.asarray(P, float); d = P - A([cx, cy]); u = d[..., 0] * dx + d[..., 1] * dy; r = np.hypot(d[..., 0], d[..., 1]); g = np.exp(-(r / R) ** 2)
        pile = np.exp(-((r - R * .95) / (R * .42)) ** 2) * np.clip(u / R, 0, 1) * .40
        return P + np.stack([depth * (g * dx + pile * dx), depth * g * dy], -1)
    return f
