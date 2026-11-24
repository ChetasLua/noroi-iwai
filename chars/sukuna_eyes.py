"""SUKUNA lane: the four giant ink eyes.  Each is one slash of the brush that tears open.
    e = InkEye(cx, cy, L, H, ang, seed)      L = half length px, H = half height px when open, ang = axis (radians)
    img = e.draw(img, o, look=(lx, ly), slash=1.)      o 0..1 opening; slash 0..1 draws the closed stroke on; look in -1..1 (along axis, across)
    four(kind) -> list of four placed eyes for a2 / f1"""
import numpy as np, cv2, math
from jj import W, H
from jj.paint import rgb, f32, smooth, mix, Canvas
from jj.ink import Cel, spl, resample, noise1
import cairo
from jj.comp import over
from chars.sukuna_ink import *


def bristle(c, P, ws, col=BLACK, seed=0, n=None, dry0=.12, dry1=.45, spread=1.08, hair=1.0, alpha=1.):
    """a loaded brush stroke seen as hairs: solid in the body, splitting into streaks toward both ends"""
    r = np.random.default_rng(abs(int(seed))); P = resample(P, 3.); m = len(P)
    if m < 8: return c
    T = np.gradient(P, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]; w = prof(m, ws)
    n = n or int(np.clip(w.max() / 2.6, 4, 60))
    for j in range(n):
        u = (j + .5) / n * 2 - 1 + r.normal(0, .5 / n); u *= spread; e = abs(u)
        t0 = dry0 * r.random() ** 1.4 * (.4 + e); t1 = 1 - dry1 * r.random() ** 1.7 * (.35 + .9 * e)
        i0, i1 = int(m * t0), int(m * t1)
        if i1 - i0 < 6: continue
        L = P + N * (u * w * .5)[:, None]; hw = max(.8, w.max() / n * 1.5) * r.uniform(.6, 1.3) * hair
        segs = [(i0, i1)]
        if r.random() < .5:                                   # skipped patch near the end of the pull
            g = int(i0 + (i1 - i0) * r.uniform(.55, .85)); k = int((i1 - i0) * r.uniform(.03, .10)); segs = [(i0, g), (g + k, i1)]
        for a, b in segs:
            if b - a < 5: continue
            R = stroke_poly(L[a:b], np.array([.3, 1, 1, 1, .8, .25]) * hw, int(r.integers(1 << 30)), .35, 2.2, .3)
            if R is not None: c.fill(R, col, alpha)
    return c


def fly(c, P, ws, seed=0, n=None, t0=(.45, .85), w=2.0, reach=.96):
    """flying white: hairs lifted out of a finished stroke, opening toward its end"""
    r = np.random.default_rng(abs(int(seed))); P = resample(P, 3.); m = len(P)
    if m < 12: return c
    T = np.gradient(P, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]; wd = prof(m, ws); n = n or int(wd.max() / 4)
    c.ctx.save(); c.ctx.set_operator(cairo.OPERATOR_DEST_OUT)
    for j in range(n):
        u = r.uniform(-reach, reach); a = r.uniform(*t0); i0 = int(m * a); i1 = int(m * min(1, a + r.uniform(.10, .5)))
        if i1 - i0 < 6: continue
        L = (P + N * (u * wd * .5)[:, None])[i0:i1]; R = stroke_poly(L, np.array([.1, .8, 1, 1.2, 1.4]) * w * r.uniform(.4, 1.5), int(r.integers(1 << 30)), .5, 2.2, .4)
        if R is not None: c.fill(R, '#000000')
    c.ctx.restore(); return c


class InkEye:
    def __init__(self, cx, cy, L, H_, ang, seed=0, flip=1, bend=.10, tone=0.):
        self.cx, self.cy, self.L, self.Hh, self.ang, self.seed, self.flip, self.bend, self.tone = cx, cy, L, H_, ang, seed, flip, bend, tone
        c, s = math.cos(ang), math.sin(ang); self.ax = np.array([c, s]); self.nx = np.array([s, -c]) * flip      # nx points "up" out of the upper lid
        self._p = None

    def X(self, u, v):
        u, v = np.asarray(u, float), np.asarray(v, float)
        return np.array([self.cx, self.cy]) + np.multiply.outer(u * self.L, self.ax) + np.multiply.outer(v * self.Hh, self.nx)

    def base(self, u): return self.bend * np.sin(math.pi * u * .9) * 1.0 - .10 * u * u

    def lid(self, o, up=True):
        u = np.linspace(-1, 1, 61); q = np.random.default_rng(self.seed + 5)
        if up: k = np.clip(1 - np.abs(u + .10 * 1) ** 2.1 / 1.22, 0, 1) ** .95; a = 1.0
        else: k = np.clip(1 - np.abs(u - .18) ** 2.4 / 1.35, 0, 1) ** 1.05; a = -.72
        k = k * np.clip((1 - u * u) * 5, 0, 1)
        return u, self.base(u) + a * k * o

    # ------------------------------------------------ painted inside (cached)
    def paint(self):
        if self._p is not None: return self._p
        r = np.random.default_rng(self.seed + 11); m = 60
        q = np.vstack([self.X(-1.45, 0), self.X(1.45, 0), self.X(0, 1.9), self.X(0, -1.9)]); lo = np.floor(q.min(0) - m).astype(int); hi = np.ceil(q.max(0) + m).astype(int)
        x0, y0 = max(0, lo[0]), max(0, lo[1]); x1, y1 = min(W, hi[0]), min(H, hi[1])
        if x1 - x0 < 8 or y1 - y0 < 8: self._p = (0, 0, None, None); return self._p
        h, w = y1 - y0, x1 - x0; org = np.array([x0, y0])
        cv = Canvas(h, w, color='#3c070d', layer=True); cv.img[:] = rgb('#4a0910')
        def lay(n, hw, cols, ur, vr, op=1., dr=.15, ln=.28, jit=.12, shape='flat'):
            u = r.uniform(*ur, n); v = r.uniform(*vr, n); Ps = []; C = []
            for a, b in zip(u, v):
                l = ln * r.uniform(.5, 1.5); uu = np.linspace(a - l, a + l, 6); vv = self.base(uu) * .9 + b * (1 - .62 * np.abs(uu) ** 2) + r.normal(0, jit) * (uu - a)
                Ps.append(self.X(uu, vv) - org); d = math.hypot((a - .03) * 1.0, (b + .16) * .85) + .25 * max(0, b); k = np.clip(d * 1.2 + r.normal(0, .09), 0, 1)
                cc = np.array([np.interp(k, [0, .30, .60, 1.], [c_[i] for c_ in cols]) for i in range(3)]); C.append(cc * (1 + r.normal(0, .05)))
            cv.strokes(np.array(Ps), hw * r.uniform(.5, 1.3, n), np.clip(np.array(C), 0, 1), op=op, dry=dr * r.uniform(.3, 2, n), seed=int(r.integers(1 << 20)), bristle=.12, shape=shape)
        S = self.Hh / 190.
        lay(70, 34 * S, [rgb('#b81823'), rgb('#a5101b'), rgb('#6f0b14'), rgb('#33060b')], (-1.05, 1.05), (-1.1, 1.1), 1., .0, .36, shape='round')
        lay(170, 13 * S, [rgb('#e0454a'), rgb('#cc1f2a'), rgb('#8c0d18'), rgb('#3f080f')], (-1, 1), (-1.0, 1.0), .95, .18, .20, shape='round')
        lay(120, 8 * S, [rgb('#f2938a'), rgb('#e2373d'), rgb('#9c101b'), rgb('#480910')], (-.9, .9), (-.9, .7), .9, .30, .22)
        lay(60, 5 * S, [rgb('#f9b9ad'), rgb('#f0625c'), rgb('#b8131f'), rgb('#560a13')], (-.6, .6), (-.8, .4), .85, .42, .17)
        # shadow thrown by the upper lid is laid per frame; veins:
        cc = Cel(w, h, 0, 0)
        for k in range(16):
            s = -1 if k % 2 else 1; a = r.uniform(.45, .95) * s; b = r.uniform(-.5, .5) * .5; uu = np.linspace(a, a - s * r.uniform(.16, .42), 7); vv = self.base(uu) + b + np.cumsum(r.normal(0, .05, 7))
            brush(cc, self.X(uu, vv) - org, np.array([2.6, 1.8, 1.4, .3]) * max(.6, S), '#6a0a14', k + self.seed, .6)
        g = over(cv.img, cc.rgba())
        self._p = (x0, y0, g.astype(f32), None); return self._p

    def smear_layer(self):
        """dark blood dragged along the eye before it opens"""
        if getattr(self, '_s', None) is not None: return self._s
        r = np.random.default_rng(self.seed + 31); c = Cel(); S = self.Hh / 190.
        for k in range(26):
            s = r.uniform(-1, 1); v = s * 1.20 + r.normal(0, .08); a = r.uniform(-1.5, -.2); b = a + r.uniform(.7, 1.9); uu = np.linspace(a, b, 11); vv = self.base(uu) + v * (1 - .5 * np.clip(np.abs(uu), 0, 1.3) ** 2)
            col = ['#2a0508', '#4a0810', '#6d0c16', '#7d0f1a', '#1c0406'][int(r.integers(5))]; w = r.uniform(26, 78) * S
            q = self.X(uu, vv); q = spl(q, n=5); bristle(c, q, np.array([.4, 1, 1, .9, .7, .5]) * w, col, int(r.integers(1 << 20)), n=int(np.clip(w / 4.0, 5, 18)), dry0=.4, dry1=.75, hair=1.3)
        self._s = c.layer(); return self._s

    # ------------------------------------------------ one frame, in three passes
    def under(self, base, o):
        return over(base, self.smear_layer(), min(1., o * 1.6) * .9) if o > 0 else base

    def _lids(self, o):
        u, vu = self.lid(o, True); _, vl = self.lid(o, False); return u, vu, vl, self.X(u, vu), self.X(u, vl)

    def _open(self, o, hold):
        u, vu, vl, U, D_ = self._lids(o); S = self.Hh / 190.; sd = self.seed * 101 + hold * 7
        jag = lambda a, q: a + self.nx * (noise1(len(a), 2.6, np.random.default_rng(sd + q)) * 3.2 * S * min(1, o * 3))[:, None]
        return np.vstack([jag(U, 2), jag(D_, 3)[::-1]])

    def inside(self, base, o, hold=0):
        if o <= .01: return base
        x0, y0, g, _ = self.paint()
        if g is None: return base
        u, vu, vl, U, D_ = self._lids(o); op = self._open(o, hold)
        mc = Cel(g.shape[1], g.shape[0], x0, y0); mc.fill(op, '#ffffff'); m = mc.rgba()[..., 3:4]
        sc = Cel(g.shape[1], g.shape[0], x0, y0); dv = .16 + .10 * o
        sc.fill(np.vstack([U, self.X(u, vu - dv * np.clip(1 - u * u, 0, 1) ** .5 * o)[::-1]]), '#ffffff'); sm = washify(sc.rgba(), ox=x0, oy=y0)[..., 3:4]
        gg = g * (1 - sm * .62 * np.array([.8, .95, .95], f32)); b = base[y0:y0 + g.shape[0], x0:x0 + g.shape[1]]
        base = base.copy(); base[y0:y0 + g.shape[0], x0:x0 + g.shape[1]] = b * (1 - m) + gg * m
        return base

    def ink(self, c, o, look=(0., 0.), slash=1., smear=0., lid_w=1., strands=True, hold=0, parts='all', rk=1., pk=1., upright=0.):
        """all black marks into an existing Cel.  hold = drawing number (changes the hand, not the design)"""
        if slash <= 0 and o <= 0: return
        L, Hh = self.L, self.Hh; S = Hh / 190.; sd = self.seed * 101 + hold * 7; r = np.random.default_rng(sd + 1)
        u, vu, vl, U, D_ = self._lids(o)
        if o > .01:
            op = self._open(o, hold); c.clip(op)
            if parts != 'lids': self._iris(c, o, look, S, sd, smear, rk, pk, upright)
            if parts == 'iris': c.unclip(); return
            feather(c, U[6:-6], self.X(u, vu - .42 * o)[6:-6], 7.5 * max(.7, S), 5. * max(.7, S), BLACK, sd + 4, (.15, .85), skew=.04, group=6, hole=.7, env=lambda t: .35 + .9 * abs(t - .5) * 2, bow=.05)
            feather(c, D_[4:26], self.X(u, vl + .40 * o)[4:26], 8. * max(.7, S), 4. * max(.7, S), BLACK, sd + 5, (.1, .7), skew=.05, group=5, hole=.7, env=lambda t: 1 - t)
            feather(c, D_[36:-3], self.X(u, vl + .40 * o)[36:-3], 8. * max(.7, S), 4. * max(.7, S), BLACK, sd + 6, (.1, .7), skew=-.05, group=5, hole=.7, env=lambda t: t)
            if strands and .02 < o < .75:                      # sinews of ink that stretch across and snap
                for k in range(13):
                    a = r.uniform(-.8, .8); life = r.uniform(.25, .95)
                    if o / .75 > life: continue
                    th = (1 - o / .75 / life); p0 = self.X(a, np.interp(a, u, vu)); p1 = self.X(a + r.normal(0, .06), np.interp(a, u, vl)); mid = (p0 + p1) / 2 + self.ax * r.normal(0, 8) * S
                    brush(c, np.array([p0, mid, p1]), np.array([16, 4 + 5 * th, 12]) * S * (.35 + .65 * th), BLACK, sd + 40 + k, .5)
            c.unclip()
        if parts == 'iris': return
        e = min(1., slash); w = lid_w * S; i1 = max(4, int(61 * e)); wide = .55 + .45 * min(1, o * 1.5)
        tail = lambda Q, k0, k1: np.vstack([Q[:1] + (Q[:1] - Q[4:5]) * k0, Q, Q[-1:] + (Q[-1:] - Q[-5:-4]) * k1])
        if o <= .02:
            Q = tail(U[:i1], .9, 1.3 if e >= 1 else 0); brush(c, Q[::3], np.array([1, 16, 30, 34, 26, 16, 3]) * w, BLACK, sd + 8, .8); bristle(c, Q, np.array([10, 40, 54, 56, 44, 26]) * w, BLACK, sd + 9, dry0=.25, dry1=.6)
        else:
            Qu = tail(U, 1.1, 2.4); Ql = tail(D_, .7, 1.5); W1 = np.array([1, 20, 46, 60, 58, 46, 30, 12, .5]) * w * wide
            brush(c, Qu[::3] + self.nx * 7 * w, W1, BLACK, sd + 10, 1.0)
            W2 = np.array([4, 26, 60, 92, 96, 70, 46, 30, 14, 4]) * w * wide; bristle(c, Qu + self.nx * 13 * w, W2, BLACK, sd + 11, dry0=.45, dry1=.60)
            # second pull: starts over the pupil and shoots past the outer corner, splitting in three
            m = len(Qu); q2 = Qu[int(m * .42):] + self.nx * 16 * w; bristle(c, q2, np.array([4, 34, 48, 44, 36, 26]) * w * wide, BLACK, sd + 17, dry0=.3, dry1=.5)
            for k, (da, ln, ww) in enumerate([(-.16, .62, 22), (.05, .84, 17), (.24, .46, 11)]):
                p = Qu[-9]; d = (Qu[-6] - Qu[-14]); d /= np.hypot(*d); ca, sa = math.cos(da * self.flip), math.sin(da * self.flip); d2 = np.array([d[0] * ca - d[1] * sa, d[0] * sa + d[1] * ca]); q = np.array([p - d * 60 * w, p, p + d2 * L * ln * .25, p + d2 * L * ln * .55 + self.nx * 6 * w])
                brush(c, q, np.array([2, 1.2, .8, .05]) * ww * w, BLACK, sd + 80 + k, .9); bristle(c, spl(q, n=8), np.array([1.5, 1.5, 1.4, .8]) * ww * w, BLACK, sd + 83 + k, dry0=.2, dry1=.6)
            fly(c, Qu + self.nx * 12 * w, W2, sd + 14, t0=(.50, .92), w=2.4 * max(.7, w)); fly(c, (Qu + self.nx * 12 * w)[::-1], W2[::-1], sd + 15, t0=(.62, .95), w=2.0 * max(.7, w))
            # lower lid: two pulls meeting under the iris, thin in the middle
            brush(c, Ql[::3] - self.nx * 4 * w, np.array([.5, 26, 38, 24, 11, 14, 30, 38, 22, .5]) * w * wide, BLACK, sd + 12, 1.0)
            W3 = np.array([6, 40, 56, 40, 18, 20, 44, 56, 36, 10]) * w * wide; bristle(c, Ql - self.nx * 9 * w, W3, BLACK, sd + 13, dry0=.4, dry1=.6); fly(c, Ql - self.nx * 8 * w, W3, sd + 16, t0=(.55, .95), w=2. * max(.7, w))
            # a run of ink from the lowest point
            for k in (range(3) if parts == 'all' else ()):
                a_ = r.uniform(-.35, .45); p = self.X(a_, np.interp(a_, u, vl) - .10); ln = r.uniform(40, 150) * S * min(1, o * 1.2); brush(c, np.array([p, p + (r.normal(0, 2), ln * .5), p + (r.normal(0, 3), ln)]), np.array([9, 5, 4, 6.5, .5]) * w, BLACK, sd + 90 + k, .5)
            # wrinkles that echo the lids, each a different length
            for k, (v, a, b, ww) in enumerate([(.30, -.95, .35, 15), (.52, -.60, .95, 10), (.78, -.25, .55, 7), (.40, .25, 1.25, 12), (-.26, -1.15, .10, 11), (-.46, -.30, .9, 9), (-.62, .30, 1.2, 6)]):
                uu = np.linspace(a, b, 15); f = np.interp(np.clip(uu, -1, 1), u, vu if v > 0 else vl); vv = f + v * (.55 + .45 * o) * np.clip(1.25 - uu * uu * .55, .2, 1) + .02 * noise1(15, 5, r)
                q = self.X(uu, vv); brush(c, q, np.array([.5, 1, .9, .6, .1]) * ww * w * r.uniform(.7, 1.2), BLACK, sd + 20 + k, .9)
                if ww > 9: bristle(c, q, np.array([.4, 1.5, 1.6, 1.0, .5]) * ww * w, BLACK, sd + 30 + k, dry0=.4, dry1=.75)

    def draw(self, base, o, look=(0., 0.), slash=1., smear=0., hold=0, **k):
        return draw_eyes(base, [self], [o], [look], [slash], [smear], hold)

    def _iris(self, c, o, look, S, sd, smear, rk=1., pk=1., upright=0.):
        sd = self.seed * 101; r = np.random.default_rng(sd + 60); L, Hh = self.L, self.Hh
        ctr = self.X(look[0] * .42 - .03, self.base(np.array(look[0] * .42 - .03)) + .12 * o + look[1] * .30); R = Hh * .40 * rk; asp = 1.0
        ex, ny = self.ax, self.nx
        pt = lambda th, k, rx=1., ry=1.12: ctr + ex * math.cos(th) * R * k * rx + ny * math.sin(th) * R * k * ry
        a = np.linspace(0, 6.283, 30, endpoint=False); c.fill(spl(np.array([pt(t, 1.0 + .03 * math.sin(5 * t)) for t in a]), True, n=3), '#8c0c18', .80); c.fill(spl(np.array([pt(t + .4, .60, 1., 1.1) for t in a]), True, n=3), '#d92a33', .75)
        if smear:                                   # the whole iris dragged along the move
            for k in range(22):
                th = r.uniform(0, 6.28); p = pt(th, r.uniform(.1, 1.05)); d = ex * smear * L * r.uniform(.10, .34); brush(c, np.array([p - d, p - d * .4, p]), np.array([.3, 6, 10]) * S, BLACK, sd + 300 + k, .6)
        # broken double ring
        for j, (k0, ww, gaps) in enumerate([(1.0, 13, (.02, .30, .55)), (.80, 7, (.15, .48, .80))]):
            a = r.uniform(0, 6.28)
            for q in range(3):
                th = np.linspace(a + q * 2.09 + gaps[0], a + (q + 1) * 2.09 - .10 - .3 * r.random(), 14); k = k0 * (1 + .03 * np.sin(3 * th + j)); brush(c, np.array([pt(t, kk) for t, kk in zip(th, k)]), np.array([2, 1, .8, 1.1, .2]) * ww * S * r.uniform(.7, 1.2), BLACK, sd + 70 + j * 9 + q, .7)
        # radial dashes driving in
        n = int(26 + 10 * min(1.5, S))
        for k in range(n):
            th = 2 * math.pi * (k + r.uniform(-.3, .3)) / n; k0 = r.uniform(.9, 1.22); k1 = r.uniform(.46, .74); brush(c, np.array([pt(th, k0), pt(th + .03, (k0 + k1) / 2), pt(th + .02, k1)]), np.array([2, 8, .3]) * S * r.uniform(.5, 1.1), BLACK if r.random() < .75 else '#5c0a12', sd + 100 + k, .5)
        # pupil: slash of a split brush; upright = 1 turns it to the screen vertical (all four lock on the lens)
        py = ny * (1 - upright) + np.array([0., -1.]) * upright; py = py / np.hypot(*py); px = np.array([-py[1], py[0]])
        for k in range(9):
            x = (k - 4) / 4. * .17 * pk; h = (1 - abs(x) / (.24 * max(pk, .3))) * r.uniform(.62, .98); brush(c, np.array([ctr + px * R * x + py * R * h * 1.05, ctr + px * R * (x * .8), ctr + px * R * (x * .9) - py * R * h * .95]), np.array([.4, 15 * max(pk, .35), 1.2]) * S * r.uniform(.8, 1.2), BLACK, sd + 160 + k, .7)

    # ------------------------------------------------ cached states for animation (premultiplied uint8 ROIs)
    def state(self, o, slash=1., hold=0):
        """-> dict(u = (x, y, rgba) smear + painted inside, l = (x, y, rgba) lids): the iris is drawn live between them"""
        def crop(L):
            a = L[..., 3] > .004; ys, xs = np.nonzero(a.any(1))[0], np.nonzero(a.any(0))[0]
            if len(ys) == 0: return (0, 0, np.zeros((1, 1, 4), np.uint8))
            y0, y1, x0, x1 = ys[0], ys[-1] + 1, xs[0], xs[-1] + 1; return (int(x0), int(y0), np.clip(L[y0:y1, x0:x1] * 255 + .5, 0, 255).astype(np.uint8))
        u = np.zeros((H, W, 4), f32)
        if o > .01:
            u = self.smear_layer() * min(1., o * 1.6) * .9; blk = self.inside(np.zeros((H, W, 3), f32), o, hold); wht = self.inside(np.ones((H, W, 3), f32), o, hold); a = 1 - (wht - blk).mean(-1, keepdims=True); u = u * (1 - a) + np.dstack([blk, a])
        c = Cel(); self.ink(c, o, slash=slash, hold=hold, parts='lids'); return dict(u=crop(u), l=crop(inkify(c.rgba(), .4, 1, .2)))

    def runs(self, c, o, k, seed=0):
        """ink that runs from the lower lid, k frames after the tear"""
        r = np.random.default_rng(self.seed + 71 + seed); S = self.Hh / 190.; u, vu, vl, U, D_ = self._lids(o)
        for j in range(4):
            a_ = r.uniform(-.5, .55); p = self.X(a_, np.interp(a_, u, vl) - .08); ln = r.uniform(50, 210) * S * (1 - math.exp(-max(0, k - 3 - 3 * j) / r.uniform(7, 16)))
            if ln > 6: w = r.uniform(5, 10) * S; brush(c, np.array([p, p + (r.normal(0, 2), ln * .5), p + (r.normal(0, 3), ln)]), np.array([w, w * .55, w * .45, w * .8, .6]), BLACK, self.seed * 7 + j, .5)

    def throw(self, c, k, n=54, power=1., ground=None):
        """ink thrown out of the slit when it tears (k = frames since the slash): long lashes of ink off both corners, then drops that fly, stretch, land and stay"""
        if k < 1: return
        r = np.random.default_rng(self.seed * 13 + 5); S = max(.6, self.Hh / 190.)
        if k <= 5:                                              # the crack of the brush: streaks off the corners
            g = min(1., k / 3.); fade = 1 if k < 4 else .45
            for j in range(22):
                e = 1 if r.random() < .6 else -1; th = r.normal(0, .42); d = self.ax * e * math.cos(th) + self.nx * math.sin(th); L0 = self.L * r.uniform(.25, 1.15) * power; p0 = self.X(e * r.uniform(.75, 1.05), r.normal(0, .12)); a, b_ = p0 + d * L0 * (.55 * max(0, g - .45)), p0 + d * L0 * g; w = r.uniform(2, 11) * S * fade
                brush(c, np.array([a, (a + b_) / 2 + self.nx * r.normal(0, 6), b_]), np.array([.4, 1, .8, .05]) * w, BLACK if r.random() < .6 else BLOOD_D, j + self.seed, .7)
        for j in range(n):
            u0 = r.uniform(-1.05, 1.05); sg = 1 if r.random() < .58 else -1; d = self.nx * sg * r.uniform(.55, 1) + self.ax * (u0 * 1.0 + r.normal(0, .35)); d = d / np.hypot(*d); v = max(self.L * (.16 + .95 * r.random() ** 1.5) * power, abs(d @ self.nx) * self.Hh * 1.9); R = (1.6 + 15 * r.random() ** 3.2) * S; land = 3 + 4 * r.random(); col = r.random() < .40; kk = min(k - 1 + .6, land)
            pos = lambda t: self.X(u0, 0.) + d * v * (1 - math.exp(-t * .62)); p = pos(kk); cc = (BLOOD_D if r.random() < .45 else '#a3131e') if col else BLACK
            if k - 1 < land: q = pos(max(0, kk - 1.1)); brush(c, np.array([q, (q + p) / 2, p]), [.4, R * .8, R * 1.5, R * .9], cc, j, .4)
            else:
                g = ground or c; g.fill(drop_shape(p[0], p[1], R, j + self.seed, 1 + .9 * r.random() ** 2, math.atan2(d[1], d[0])), cc)
                if R > 5:
                    for q_ in range(3): t2 = math.atan2(d[1], d[0]) + r.normal(0, .7); e = R * r.uniform(1.4, 3.2); g.fill(drop_shape(p[0] + math.cos(t2) * e, p[1] + math.sin(t2) * e, R * r.uniform(.1, .24), j + q_), cc)
                    if d[1] > -.2 and k > land + 2: ln = R * r.uniform(2, 7) * (1 - math.exp(-(k - land) / 9.)); brush(g, np.array([p, p + (0, ln * .5), p + (r.normal(0, 1.5), ln)]), [R * .8, R * .4, R * .5, .5], cc, j + 9, .3)

    def iris_roi(self, o, look, smear=0., hold=0, rk=1., pk=1., upright=0.):
        """live iris as a small inked ROI -> (x, y, rgba)"""
        ctr = self.X(look[0] * .42 - .03, self.base(np.array(look[0] * .42 - .03)) + .12 * o + look[1] * .30); m = self.Hh * .40 * rk * 1.45 + 8 + abs(smear) * self.L * .36; x0, y0 = int(ctr[0] - m), int(ctr[1] - m); c = Cel(int(2 * m), int(2 * m), x0, y0)
        self.ink(c, o, look, smear=smear, hold=hold, parts='iris', rk=rk, pk=pk, upright=upright); return x0, y0, inkify(c.rgba(), .35, 2, .1, x0 % 997, y0 % 991)


def draw_eyes(base, eyes, o, look=None, slash=None, smear=None, hold=0, spat=True):
    """several eyes in one go: all smears, then all insides, then one ink cel"""
    n = len(eyes); look = look or [(0., 0.)] * n; slash = slash or [1.] * n; smear = smear or [0.] * n
    for e, q in zip(eyes, o): base = e.under(base, q)
    for e, q in zip(eyes, o): base = e.inside(base, q, hold)
    c = Cel(); red = Cel()
    for e, q, l, s_, m in zip(eyes, o, look, slash, smear):
        e.ink(c, q, l, s_, m, hold=hold)
        if spat and q > 0:
            S = e.Hh / 190.; k = min(1., q * 1.8)
            spray(c, e.X(1.25, -.1), e.ang + (0 if True else 0), 26, 420 * S * k, .30, 11 * S, BLACK, e.seed * 3 + 1); spray(red, e.X(.9, .35 * e.flip), e.ang - .12 * e.flip, 22, 520 * S * k, .34, 10 * S, BLOOD_D, e.seed * 3 + 2); spray(red, e.X(-1.1, -.2), e.ang + math.pi, 14, 300 * S * k, .4, 8 * S, '#8d0f1b', e.seed * 3 + 3)
    base = over(base, red.rgba())
    return over(base, inkify(c.rgba(), .4, 1, .2))


def charcoal(seed=2):
    """charcoal sheet: burnt paper, rubbed verticals, tooth"""
    k = ('charcoal', seed)
    if k in _C: return _C[k].copy()
    n = fbm(H, W, (260, 90, 30), (1, .6, .4), seed); t = fbm(H, W, (2.6, 1.3), (1, .8), seed + 1)
    b = np.empty((H, W, 3), f32); b[:] = rgb('#2a2626'); b *= (1 + .20 * n)[..., None]
    r = np.random.default_rng(seed); cv = Canvas(H, W, img=b)
    for k in range(90):
        a = -1.25 + r.normal(0, .28); L = r.uniform(240, 900); x, y = r.uniform(-100, W + 100), r.uniform(-80, H + 80); d = np.array([math.cos(a), math.sin(a)]); nn = np.array([-d[1], d[0]]); q = np.linspace(-.5, .5, 7)[:, None]; P = np.array([x, y]) + d * L * q + nn * 30 * r.normal() * (q * q)
        g = rgb(['#3a3534', '#1d1a1a', '#454040', '#151213', '#332a2a'][int(r.integers(5))]); cv.strokes(P[None], r.uniform(18, 90), g[None], op=r.uniform(.25, .6), dry=r.uniform(.45, .9), seed=k, bristle=.1)
    b = cv.img * (1 + .10 * t)[..., None]
    y, x = np.mgrid[0:H, 0:W].astype(f32); e = np.clip(np.hypot((x - W / 2) / (W * .62), (y - H / 2) / (H * .7)), 0, 1.5)
    b *= (1.08 - .50 * e ** 2)[..., None]
    _C[k] = np.clip(b, 0, 1).astype(f32); return _C[k].copy()


_C = {}


def four(kind='a2'):
    """four placed eyes, biggest first; asymmetric on purpose"""
    if kind == 'a2':
        return [InkEye(610, 330, 560, 205, .30, 3, 1, .10), InkEye(1420, 300, 400, 150, -.34, 8, 1, -.12), InkEye(500, 760, 380, 132, .16, 13, 1, .08), InkEye(1330, 740, 520, 176, -.24, 21, 1, -.10)]
    return [InkEye(560, 300, 520, 190, .42, 3, 1, .10), InkEye(1560, 250, 360, 130, -.5, 8, 1, -.12), InkEye(300, 690, 340, 120, .30, 13, 1, .08), InkEye(1700, 640, 260, 92, -.62, 21, 1, -.10)]
