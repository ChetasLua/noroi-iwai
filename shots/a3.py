"""A3  The word.  呪 is slammed on white paper in eight strokes, burns, runs, and the paper goes blind.  36 f."""
import numpy as np, cv2, math, os
from jj import comp, W, H
from jj.paint import rgb, f32, smooth, fbm, grid
from jj.ink import Cel
from jj.sumi import (Page, Mark, splash, fling, drip, Spray, paper, lay, page_state, wet_light, page_save, page_load, impact_inv, blob, SHEET, BLACK)
from jj.letters import write

D = math.radians
X, Y, SZ, ROT = 352, 14, 1030, D(-3)
LAND = [0, 2, 5, 7, 9, 12, 13, 15]                                             # frame on which each stroke lands: fast fast hold | fast fast hold | fast, the long one
TIMES = [(-.35, 1.15), (1.9, 4.0), (4.9, 6.2), (6.9, 8.1), (8.8, 11.2), (11.9, 13.0), (12.8, 14.8), (14.8, 17.8)]
KICK = [(13, .9, 9.), (10, .5, 11.), (9, 2.6, 8.), (12, 1.1, 10.), (11, .4, 14.), (8, 2.8, 8.), (12, 1.9, 13.), (10, 1.3, 18.)]   # (zoom mil, direction rad, px)
VER = 'v3'


def B(a, b, k=SZ / 1000., x=X, y=Y):
    c, s = math.cos(ROT), math.sin(ROT); return (x + (500 + (a - 500) * c - (b - 500) * s) * k, y + (500 + (a - 500) * s + (b - 500) * c) * k)


class Shot:
    id = 'a3'; n = 36
    finish = dict(bloom=.10, vig=.10, soft=.03, grain=.011, lift=.0)
    EVENTS = ([(0, 'cut', dict(note='hard cut to white paper, silence of a room')), (0, 'paper_room', dict(size=.3, note='room tone: a big sheet on a board'))]
              + [(LAND[i], 'brush_slap', dict(size=s, stroke=i + 1, pan=p, note=n)) for i, (s, p, n) in enumerate([(.75, -.6, 'first stroke, short vertical'), (.85, -.4, 'corner stroke'), (.6, -.45, 'closing stroke, then a hold'), (.75, .1, 'short vertical'), (.9, .4, 'corner stroke'), (.6, .2, 'closing stroke, then a hold'), (.95, -.05, 'left sweep'), (1.0, .35, 'last stroke, the long hook')])]
              + [(2, 'brush_drag', dict(dur=2, pan=-.3)), (9, 'brush_drag', dict(dur=2, pan=.4)), (13, 'brush_drag', dict(dur=2, pan=-.2, note='left sweep leaves the paper')), (15, 'brush_drag', dict(dur=3, pan=.5, note='long pull, pressed, flicked upward')),
                 (17, 'ink_splash', dict(size=.9, pan=.6, note='thrown drops past the lens')), (18, 'impact', dict(size=.9, note='inversion: black and blood, the kanji burns white; held 4 frames')), (18, 'paper_impact', dict(size=1.0)), (20, 'impact_cut', dict(size=.6, note='second crop of the inversion, full word')),
                 (22, 'cut', dict(note='back on white, wet ink')), (23, 'ink_run', dict(dur=12, note='seven runs of ink')), (25, 'ink_drip', dict(pan=-.5)), (28, 'ink_drip', dict(pan=.3)), (30, 'overexpose', dict(dur=6, note='paper burns out to white, full at 35'))])

    def setup(self):
        os.makedirs('cache', exist_ok=True); pa, pb = f'cache/a3_page_{VER}.npz', f'cache/a3_closemap_{VER}.npz'
        if os.path.exists(pa): pg = page_load(pa)
        else:
            pg = Page(seed=4); write(pg, '呪', (X, Y, SZ), rot=ROT, times=TIMES)
            S = lambda i, m, dt=0., d=2.2: pg.add(self._t(m, LAND[i] + dt, LAND[i] + dt + d))
            S(0, splash(*B(70, 350), 180, seed=11, ang=D(200), spread=1.5, n=140, big=10, fingers=10, lobes=4, core=22))
            S(1, splash(*B(350, 330), 120, seed=18, ang=D(-40), spread=1.3, n=70, big=7, fingers=7, lobes=3, core=18), 1.)
            S(2, fling(*B(308, 575), D(20), 110, seed=22, n=50, big=5, fingers=4))
            S(3, splash(*B(505, 160), 190, seed=12, ang=D(-120), spread=1.3, n=130, big=11, fingers=9, lobes=4, core=22))
            S(4, splash(*B(880, 160), 200, seed=13, ang=D(-30), spread=1.2, n=150, big=11, fingers=10, lobes=4, core=26), 1.)
            S(5, splash(*B(640, 400), 200, seed=23, n=50, big=5, fingers=0, lobes=5, core=0, mist=1.4))
            S(6, fling(*B(250, 984), D(155), 240, seed=14, big=8, src=[B(246, 970), B(270, 990)]), 1.4)
            S(7, splash(*B(1110, 850), 340, seed=15, ang=D(5), spread=1.6, n=300, big=19, fingers=24, lobes=7, core=44), 1.8, 2.6)
            S(7, fling(*B(1096, 530), D(-88), 250, seed=16, n=130, big=9, fingers=5), 2.6)
            S(7, splash(*B(640, 430), 420, seed=21, n=120, big=7, fingers=0, lobes=9, core=0, mist=1.5), 2.0, 3.)
            page_save(pg, pa)
        self.nk = 8; self.pg = pg
        for x, y, l, w_, sd, t0, d in ((98, 650, 260, 13, 3, 20.5, 15), (312, 606, 120, 9, 8, 23, 12), (540, 432, 130, 9, 7, 21.5, 14), (905, 956, 120, 12, 5, 22, 13), (1010, 940, 90, 9, 12, 24.5, 11), (820, 398, 150, 10, 9, 22.5, 14), (420, 890, 120, 8, 21, 23.5, 13)):
            pg.add(drip(*B(x, y), l, w_, t0, t0 + d, seed=sd))
        self.P = paper(); x, y = grid(); self.light = (1.055 - .09 * smooth(.2, 1.2, np.hypot((x - 700) / 1500, (y - 380) / 1000)))[..., None]
        # extreme close: inside the hook, where the bristles split
        if os.path.exists(pb): z = np.load(pb); self.cl = (z['k'], z['p'], z['s'])
        else:
            k = 3.0; cx, cy = B(985, 800); z = SZ * k; bx, by = 960 - (cx - X) * k, 560 - (cy - Y) * k
            pc = Page(seed=4, ss=1); write(pc, '呪', (bx, by, z), rot=ROT, times=[(0, 1)] * 8, bleed=2.0); nc = len(pc.marks)
            pc.add(splash(960 + 360, 560 + 120, 620, seed=15, ang=D(5), spread=1.8, n=220, big=30, fingers=22, lobes=6, core=0, ss=1)); pc.add(splash(520, 300, 500, seed=31, n=60, big=16, fingers=0, lobes=7, core=0, ss=1))
            K_, _, Po = page_state(pc, 99, 9e9, only=set(range(nc))); Sp = page_state(pc, 99, 9e9, skip=set(range(nc)))[0]; self.cl = (K_.astype(np.float16), Po.astype(np.float16), Sp.astype(np.float16)); q = pb + f'.{os.getpid()}.npz'; np.savez(q, k=self.cl[0], p=self.cl[1], s=self.cl[2]); os.replace(q, pb)
        self.inversion(18); self.inversion(20)                                 # warms the burst cache
        # ink in the air: (landing, Spray)
        self.air = [(0., Spray(*B(70, 350), seed=9, power=44, n=70, ang=D(205), spread=2.0, big=11, lig=0, R0=60, lift=1.9)), (2.8, Spray(*B(352, 330), seed=41, power=34, n=34, ang=D(-30), spread=2.0, big=8, lig=0, R0=50, lift=1.8)),
                    (7., Spray(*B(505, 160), seed=42, power=42, n=60, ang=D(-100), spread=2.0, big=10, lig=0, R0=60, lift=1.8)), (9.8, Spray(*B(880, 150), seed=8, power=46, n=80, ang=D(-40), spread=2.2, big=11, lig=0, R0=70, lift=1.9)),
                    (14.2, Spray(*B(250, 984), seed=43, power=52, n=70, ang=D(160), spread=1.0, big=10, lig=0, R0=40, lift=1.6)), (16.6, Spray(*B(1100, 850), seed=5, power=64, n=240, ang=D(-30), spread=2.8, big=17, lig=0, R0=80, lift=1.8)), (17.8, Spray(*B(1096, 540), seed=44, power=48, n=50, ang=D(-85), spread=.8, big=9, lig=0, R0=20, lift=1.5))]
        self.VK = None

    @staticmethod
    def _t(m, a, b):
        m.t0, m.t1 = a, b; return m

    # ------------------------------------------------------------- white page at time t (frames)
    def white(self, f, shadow=True):
        t = f + .5; D_, Fr, Po = page_state(self.pg, t, tau=22.)
        base = self.P * self.light
        if shadow:                                                             # the brush is never seen: only its shadow crosses the sheet
            x, y, lift = self.tip(t)
            if lift < 240:
                s = np.zeros((H // 4, W // 4), f32); p0 = np.array([x + 40 + lift * .9, y + 46 + lift * 1.05]); p1 = p0 + np.array([1500, 1350]); n = np.array([-.67, .74])
                q = np.vstack([p0 + n * 18, p0 + np.array([60, 54]) + n * 58, p0 + np.array([170, 153]) + n * 88, p1 + n * 190, p1 - n * 190, p0 + np.array([170, 153]) - n * 88, p0 + np.array([60, 54]) - n * 58, p0 - n * 18]) / 4
                cv2.fillPoly(s, [np.round(q * 16).astype(np.int32)], 1., cv2.LINE_AA, 4); s = cv2.GaussianBlur(s, (0, 0), 6 + lift * .05); far = cv2.GaussianBlur(s, (0, 0), 26); gx, gy = grid(H // 4, W // 4); u = np.clip(((gx * 4 - p0[0]) * .743 + (gy * 4 - p0[1]) * .669) / 900., 0, 1); s = s * (1 - u) + far * u * .8
                s = cv2.resize(s, (W, H), interpolation=cv2.INTER_LINEAR); base = base * (1 - (.17 * max(0., 1 - lift / 240.)) * s[..., None] * np.array([.94, 1., 1.06], f32))
        im = lay(base, D_, None, Po); im = im + wet_light(D_, Fr, Po, 1.0, phase=f * .05)
        for a, sp in self.air:
            ag = t - a
            if .15 < ag < 9.5:
                L, sh = sp.layers(ag); fade = float(np.clip((9.5 - ag) / 3., 0, 1)); im = im * (1 - .24 * fade * sh[..., None]); im = im * (1 - L[..., 3:4]) + L[..., :3]
        return im

    def tip(self, t):
        """where the unseen brush is: (x, y, height)"""
        P = lambda i, u: (lambda inf: (lambda k: np.array([float(inf['x'][k]), float(inf['y'][k])]))(int(np.clip(np.searchsorted(inf['t'], u), 0, len(inf['x']) - 1))))(self.pg.marks[i].info)
        for i, (a, b) in enumerate(TIMES):
            if t < a: 
                if i == 0: return (*P(0, 0.), 200.)
                a0 = TIMES[i - 1][1]; u = (t - a0) / max(1e-3, a - a0); e = u * u * (3 - 2 * u); p = P(i - 1, 1.) * (1 - e) + P(i, 0.) * e; return p[0], p[1], 150 * math.sin(u * math.pi) * min(1.4, (a - a0) / 1.2)
            if t <= b: p = P(i, (t - a) / (b - a)); return p[0], p[1], 0.
        p = P(7, 1.); u = t - TIMES[-1][1]; return p[0], p[1] - 60 * u, 160 * u

    def kick(self, f):
        dx = dy = 0.; z = 1.; r = 0.; rings = []
        for i, l in enumerate(LAND):
            a = f - l
            if 0 <= a < 2:
                zm, th, px = KICK[i]; k = (1., -.42)[a]; dx += math.cos(th) * px * k; dy += math.sin(th) * px * k; z += zm * .001 * k * .6; r += D(.14) * k * (1 if i % 2 else -1)
            if 0 <= a < 3: rings.append((i, a))
        return dx, dy, z, r, rings

    def ripple(self, im, rings):
        if not rings: return im
        x, y = grid(); mx, my = x.copy(), y.copy()
        for i, a in rings:
            inf = self.pg.marks[i].info; cx, cy = float(inf['x'][len(inf['x']) // 5]), float(inf['y'][len(inf['x']) // 5]); dx, dy = x - cx, y - cy; d = np.sqrt(dx * dx + dy * dy) + 1
            g = (4.2, 3.0, 1.4)[a] * np.exp(-((d - (70 + 230 * a)) / 90.) ** 2); mx += dx / d * g; my += dy / d * g
        return cv2.remap(im, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)

    def inversion(self, f):
        if f < 20:
            K_, Po, Sp = [q.astype(f32) for q in self.cl]
            v = impact_inv(K_, Sp, 1180, 700, seed=21, pool=Po)
            if f == 19: v = comp.cam2d(v, 960 + 22, 540 - 12, 1.045, D(.5))
            return v
        K_, Fr, Po = page_state(self.pg, 18.2, 9e9, only=set(range(8))); Sp = page_state(self.pg, 19.5, 9e9, skip=set(range(8)))[0]
        v = impact_inv(K_, Sp, *B(1000, 760), seed=3, pool=Po)
        if f == 21: v = comp.cam2d(v, 960 - 8, 540 + 8, 1.022, D(-.35))
        return v

    def frame(self, f):
        if 18 <= f < 22: return self.inversion(f).astype(f32)
        im = self.white(f)
        if f < 18:
            dx, dy, z, r, rings = self.kick(f); im = self.ripple(im, rings); im = comp.cam2d(im, W / 2 + dx, H / 2 + dy, z + .004, r, interp=cv2.INTER_CUBIC)
        else:
            u = (f - 22) / 13.; s = math.exp(-(f - 22) / 2.2); rr = np.random.default_rng(f); o = rr.normal(0, 5, 2) * s
            im = comp.cam2d(im, W / 2 + o[0] + 10 * u, H / 2 + o[1] + 6 * u, 1.006 + .030 * u, D(-.25) * u, interp=cv2.INTER_CUBIC)
        if f >= 30:                                                            # the paper burns: light eats the ink from its edges inward
            k = (.07, .20, .38, .62, .88, 1.)[f - 30]; q = cv2.resize(np.clip(im, 0, 1.1), (W // 4, H // 4), interpolation=cv2.INTER_AREA)
            g = .6 * cv2.GaussianBlur(q, (0, 0), 2.5 + 3 * k) + .9 * cv2.GaussianBlur(q, (0, 0), 7 + 8 * k) + .7 * cv2.GaussianBlur(q, (0, 0), 30); g = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)
            im = im * (1 + 1.4 * k) + np.clip(g / 2.2, 0, 1) ** (2.6 - 1.5 * k) * k * 1.55 * rgb('#fff6ea'); e = max(0., (k - .6) / .4) ** 2; im = im * (1 - e) + rgb('#fffaf0') * 1.06 * e
        return np.maximum(im, 0).astype(f32)
