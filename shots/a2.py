"""A2  Sukuna's four eyes.  Four slashes of the brush on charcoal; each tears open on a beat; at 30 all look at us.  36 f."""
import numpy as np, cv2, math, os
from jj import comp, W, H
from jj.paint import rgb, f32, smooth, fbm
from jj.ink import Cel, over_roi
from jj.kit import shake, focus_lines
from chars.sukuna_ink import *
from chars.sukuna_eyes import InkEye, charcoal, bristle

VER = 'v3'
START = [0, 9, 18, 27]
# big lower left, medium upper right, small upper left, long lower right: no two share a size, a tilt or a line
EYES = lambda: [InkEye(616, 662, 590, 206, .27, 3, 1, .10), InkEye(1500, 232, 396, 146, -.46, 8, 1, -.12), InkEye(330, 168, 270, 92, .56, 13, 1, .08), InkEye(1600, 800, 430, 136, -.33, 21, 1, -.10)]
STATES = dict(s0=(0., .55), s1=(0., 1.), t1=(.20, 1.), t2=(.62, 1.), over=(1.15, 1.), set=(.90, 1.), A=(1., 1.), B=(.965, 1.), C=(1.02, 1.), wide=(1.10, 1.), glare=(.80, 1.))
SEQ = ['s0', 's1', 't1', 't2', 'over', 'over', 'set', 'set']
IDLE = ['A', 'A', 'A', 'B', 'B', 'B', 'A', 'A', 'A', 'C', 'C', 'C']
# independent wandering: (frame, look) saccades; the eye jumps in two frames and then holds dead still
WANDER = [[(0, (-.2, .1)), (7, (-.55, -.10)), (13, (.50, .12)), (19, (.10, -.30)), (25, (-.60, .22))],
          [(0, (.1, .0)), (16, (.45, -.20)), (22, (-.50, .15)), (27, (.20, .25))],
          [(0, (0., 0.)), (24, (-.45, .10)), (28, (.50, -.15))],
          [(0, (.3, .1))]]


class Shot:
    id = 'a2'; n = 36
    finish = dict(vig=.16, bloom=.16, grain=.011)
    EVENTS = ([(s, 'brush_slash', dict(size=z, pan=p, note='dry brush slit, 2 frames')) for s, z, p in zip(START, (1., .75, .55, .9), (-.35, .55, -.7, .6))]
              + [(s + 2, 'tear', dict(size=z, pan=p, note='paper-like wet tear, sinews snap, ink thrown')) for s, z, p in zip(START, (1., .75, .55, .9), (-.35, .55, -.7, .6))]
              + [(s + 4, 'lid_settle', dict(size=z * .6, pan=p)) for s, z, p in zip(START, (1., .75, .55, .9), (-.35, .55, -.7, .6))]
              + [(s + 3, 'ink_splatter', dict(size=z * .7, pan=p, dur=4)) for s, z, p in zip(START, (1., .75, .55, .9), (-.35, .55, -.7, .6))]
              + [(7, 'eye_dart', dict(size=.5, pan=-.35)), (13, 'eye_dart', dict(size=.5, pan=-.3)), (16, 'eye_dart', dict(size=.4, pan=.55)), (19, 'eye_dart', dict(size=.4, pan=-.35)), (22, 'eye_dart', dict(size=.4, pan=.5)), (24, 'eye_dart', dict(size=.3, pan=-.7)), (25, 'eye_dart', dict(size=.5, pan=-.35)), (27, 'eye_dart', dict(size=.3, pan=.55)), (28, 'eye_dart', dict(size=.3, pan=-.7))]
              + [(0, 'drone', dict(dur=34, note='charcoal room breathing, low')), (30, 'snap', dict(size=1., note='all four irises lock on the lens in 2 frames')), (32, 'impact_low', dict(size=.7, note='they arrive')), (33, 'clamp', dict(size=.9, note='pupils clamp to needles, lids press')), (34, 'brush_wipe', dict(dur=2, size=1., pan=0., note='giant black brush crosses left to right, white paper behind it'))])

    def setup(self):
        os.makedirs('cache', exist_ok=True); self.eyes = EYES(); self.st = {}
        for i, e in enumerate(self.eyes):
            for nm, (o, sl) in STATES.items():
                for part in 'ul':
                    self.st[i, nm, part] = f'cache/a2_{VER}_{i}_{nm}_{part}'
                if not all(os.path.exists(self.st[i, nm, q] + '.npy') for q in 'ul'):
                    d = e.state(o, sl, hold=list(STATES).index(nm) if nm in ('s0', 's1', 't1', 't2', 'over', 'set') else 6)
                    for q in 'ul':
                        x, y, a = d[q]; tmp = self.st[i, nm, q] + f'.{os.getpid()}.npy'; np.save(tmp, a); os.replace(tmp, self.st[i, nm, q] + '.npy'); np.save(self.st[i, nm, q] + '_xy.npy', [x, y])
        self.G = cv2.resize(charcoal(), None, fx=1.10, fy=1.10, interpolation=cv2.INTER_CUBIC); self.nx = cv2.resize(fbm(H // 4, W // 4, (60, 24), (1, .5), 5), (W, H), interpolation=cv2.INTER_CUBIC); self.ny = cv2.resize(fbm(H // 4, W // 4, (60, 24), (1, .5), 6), (W, H), interpolation=cv2.INTER_CUBIC)
        r = np.random.default_rng(4); self.dust = np.c_[r.uniform(-60, W + 60, 90), r.uniform(-60, H + 60, 90), r.uniform(-1, 1, 90), r.uniform(.15, .9, 90), r.uniform(.6, 2.6, 90)]

    def put(self, im, i, nm, part):
        x, y = np.load(self.st[i, nm, part] + '_xy.npy'); a = np.load(self.st[i, nm, part] + '.npy', mmap_mode='r'); return over_roi(im, np.asarray(a, f32) / 255., int(x), int(y))

    def look(self, i, f):
        """-> (lx, ly), smear, (radius k, pupil k, upright)"""
        if f >= 30:
            a = self._w(i, 29)[0]; b = np.array([(.02, -.03), (-.03, .02), (.0, .0), (.02, .02)][i])
            if f == 30: return a + (b - a) * .62, float(np.sign(b[0] - a[0]) or 1) * 1.3, (1.05, 1.15, .6)
            if f == 31: return b + (b - a) * .07, .35, (1.12, 1.25, 1.)
            if f == 32: return b, 0., (1.06, 1.10, 1.)
            return b, 0., (.84, .34, 1.)
        l, s = self._w(i, f); return l, s, (1., 1., 0.)

    def _w(self, i, f):
        K = WANDER[i]; j = max([q for q, (a, _) in enumerate(K) if a <= f] or [0]); a, v = K[j]; v = np.array(v, float)
        if j == 0: return v, 0.
        p = np.array(K[j - 1][1], float); k = f - a
        if k == 0: return p + (v - p) * .55, float(np.sign(v[0] - p[0])) * .9
        if k == 1: return v + (v - p) * .10, 0.
        return v, 0.

    def frame(self, f):
        # ---- charcoal sheet that breathes
        ph = f / 18. * 2 * math.pi; z = 1.045 - .012 * math.cos(ph) + .020 * f / 36.; kick = np.zeros(2)
        for s, e in zip(START, self.eyes):
            if s + 2 <= f < s + 7: kick += shake(f, (9, 6, 5, 8)[START.index(s)], 3, 1.6, s + 2)
        if f >= 31: kick += shake(f, 10, 9, 1.4, 31)
        if f >= 33: kick += shake(f, 7, 12, 2., 33)
        g = comp.cam2d(self.G, W * .55 + kick[0] * .4, H * .55 + kick[1] * .4, z, .004 * math.sin(ph * .5))
        x, y = np.meshgrid(np.arange(W, dtype=f32), np.arange(H, dtype=f32)); q = 5.5 * math.sin(ph)
        g = cv2.remap(g, x + self.nx * q, y + self.ny * q, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        g = g * (.80 + .09 * math.sin(ph - .6)) * (1 if f < 31 else .78 if f < 33 else .60)
        im = np.ascontiguousarray(g.astype(f32))
        # ---- eyes
        live = []
        for i, s in enumerate(START):
            k = f - s
            if k < 0: continue
            nm = SEQ[k] if k < len(SEQ) else IDLE[(k - 8 + 3 * i) % 12]
            if f in (31, 32) and k >= 4: nm = 'wide'
            if f >= 33 and k >= 4: nm = 'glare'
            if f in (30,) and k >= 6: nm = 'set'
            live.append((i, nm, k))
        if f >= 31:                                             # the dark presses in from the edges, behind the eyes
            fl = Cel(); focus_lines(fl, 960, 560, 900 - 120 * (f - 31), 1700, 190, 16, BLACK, 20 + f, ry=.62); im = over(im, fl.rgba())
        c = Cel(); gr = Cel()
        for i, nm, k in live: self.eyes[i].throw(c, k, n=(70, 50, 38, 58)[i], ground=gr)
        im = over(im, gr.rgba())
        for i, nm, k in live: im = self.put(im, i, nm, 'u')
        for i, nm, k in live:
            o = STATES[nm][0]
            if o > .15:
                l, sm, (rk, pk, up) = self.look(i, f); x0, y0, a_ = self.eyes[i].iris_roi(o, l, sm, 0 if nm in ('A', 'B', 'C') else 1, rk, pk, up); im = over_roi(im, a_, x0, y0)
        for i, nm, k in live: im = self.put(im, i, nm, 'l')
        for a, b, vx, vy, R in self.dust:
            px, py = (a + vx * 9 * f / 2.4 + 8 * math.sin(f * .21 + a)) % (W + 120) - 60, (b + vy * 4.2 * f) % (H + 120) - 60; c.fill(drop_shape(px, py, R, int(a), 1.), '#57504e' if vx > 0 else '#100d0e', .8)
        for i, nm, k in live:
            if STATES[nm][0] > .5: self.eyes[i].runs(c, STATES[nm][0], k)
        im = over(im, c.rgba())
        if f >= 31: im = comp.cam2d(im, W / 2 + kick[0], H / 2 + kick[1], 1.0 + (.018 if f == 31 else .010 if f == 32 else .03))
        elif np.abs(kick).max() > .3: im = comp.cam2d(im, W / 2 + kick[0], H / 2 + kick[1], 1.012)
        if f >= 34: im = self.wipe(im, f - 34)
        return im.astype(f32)

    def wipe(self, im, k):
        """the brush crosses: black, then the white sheet it drags behind it"""
        lead, tail = ((1330, -260), (2560, 1180))[k]; r = np.random.default_rng(61 + k); cv = Cel(); Y = lambda t, y: y + 26 * np.sin(t * 2.2 + y * .002) - 60 * (t - .5)
        def sweep(x0, x1, col, n, jit, lo=-80, hi=1180, thick=1.35):
            for j in range(n):
                y = lo + (hi - lo) * (j + .5 + r.normal(0, .25)) / n; e = x1 + r.normal(0, jit) - 260 * ((y - 520) / 620) ** 2; t = np.linspace(0, 1, 14); xs = x0 + (e - x0) * t; P = np.c_[xs, Y(xs / 1900., y)]; w = (hi - lo) / n * thick * r.uniform(.7, 1.25)
                R = stroke_poly(P, np.array([w, w, w, w * .9, w * .6, .5]), int(r.integers(1 << 30)), 1.6, 5., 1.2)
                if R is not None: cv.fill(R, col)
        sweep(-300, lead, BLACK, 96, 120); cv_l = cv.rgba(); im = over(im, inkify(cv_l, .8, 4, .25))
        c2 = Cel(); spray(c2, (lead - 160, 560), 0., 70, 560, .55, 18, BLACK, 5 + k, 3.2); im = over(im, c2.rgba())
        if tail > 0:
            cv = Cel(); sweep(-300, tail, '#f3eee3', 120, 170, thick=1.25); L = inkify(cv.rgba(), .7, 5, .0); im = over(im, L)
        return im
