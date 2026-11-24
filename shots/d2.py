"""d2 NOBARA: three nails flung up, the hammer, blue fire, hairpin inside the curse, the grin."""
import numpy as np, cv2, math
from jj import comp, W, H
from jj.paint import rgb, f32, fbm, smooth, ramp, grid
from jj.ink import Cel, spl, resample, noise1
from jj.kit import para, focus_lines, speed_lines, shifted, shake, smear, zoomblur, outline, splatter, contours
from chars import mn, dog
from chars.mn_keys import facets, licks, blueflame, steel_hammer, hpal, nz, boil_layer
from chars.nobara import Nobara as N, BODY, EXPR
from jj import fx

E3 = lambda t, p=3: 1 - (1 - np.clip(t, 0, 1)) ** p
INK = '#07060f'
VIOLET = [(0, '#f6e6ff'), (.12, '#d9a6ff'), (.28, '#9a4df2'), (.5, '#4b1fb0'), (.8, '#1c0f58'), (1.3, '#0a0724')]


def nail(c, p, ang, L, fs=1., w=2.6, lit=-1, pal=('#e2e8f6', '#8790b4', '#343a58')):
    """one iron nail centred at p, pointing along ang; fs = foreshortening 0..1 (spinning)"""
    d = np.array([math.cos(ang), math.sin(ang)]); n = np.array([-d[1], d[0]]); l = L * max(.16, abs(fs)); t = L * .030; p = np.asarray(p, float); a, b = p - d * l * .5, p + d * l * .5
    sh = np.vstack([a + n * t, p + d * l * .30 + n * t * .95, b, p + d * l * .30 - n * t * .95, a - n * t]); outline(c, sh, w * 1.5, INK, 1); c.fill(sh, pal[1]); c.fill(np.vstack([a + n * t * lit, p + d * l * .30 + n * t * .9 * lit, b, p + d * l * .30 + n * t * .2 * lit, a + n * t * .2 * lit]), pal[0]); c.fill(np.vstack([a - n * t * lit, p + d * l * .30 - n * t * .9 * lit, b, p + d * l * .30 - n * t * .4 * lit, a - n * t * .4 * lit]), pal[2])
    hw, hh = t * 2.7, t * 1.1 + t * 1.6 * (1 - abs(fs)); u = np.linspace(0, 6.283, 22)[:-1]; hd = a + np.outer(np.cos(u) * hh, d) + np.outer(np.sin(u) * hw, n); outline(c, hd, w * 1.4, INK, 2); c.fill(hd, pal[1]); c.fill(a + np.outer(np.cos(u) * hh * .7, d) + np.outer(np.sin(u) * hw * .6 + hw * .25 * lit, n), pal[0])


class Curse:
    """the hulking mass: silhouette, mouths, and where the nails sit (screen px of shot D)"""
    # x, y, corner
    BODYP = [(430, 1200, 0), (404, 1084, 0), (352, 1006, 0), (296, 968, 1), (250, 1010, 0), (232, 1084, 1), (206, 1052, 1), (176, 1124, 1), (150, 1066, 1), (104, 1122, 1), (96, 1052, 1), (40, 1080, 1), (92, 985, 0), (112, 912, 0), (92, 836, 0), (128, 744, 0), (214, 690, 0), (300, 676, 0), (352, 628, 1), (316, 566, 0), (256, 520, 0), (236, 440, 0), (280, 362, 0), (372, 330, 0), (456, 350, 0), (512, 404, 1), (548, 330, 0), (528, 220, 0), (556, 116, 0), (634, 48, 0), (730, 42, 0), (796, 104, 0), (806, 196, 0), (776, 262, 1), (842, 262, 0), (930, 196, 0), (1046, 136, 0), (1176, 118, 0), (1282, 158, 0), (1346, 226, 1), (1372, 168, 0), (1440, 148, 0), (1498, 196, 0), (1488, 262, 0), (1440, 322, 1), (1506, 326, 0), (1572, 282, 0), (1618, 204, 0), (1632, 112, 0), (1600, 46, 1), (1560, 6, 1), (1618, 10, 1), (1622, -56, 1), (1662, 0, 1), (1700, -60, 1), (1712, 8, 1), (1780, -26, 1), (1764, 50, 0), (1748, 130, 0), (1736, 226, 0), (1700, 330, 0), (1648, 420, 0), (1596, 486, 1), (1636, 566, 0), (1668, 656, 0), (1640, 748, 0), (1582, 806, 1), (1616, 880, 0), (1580, 960, 0), (1494, 978, 1), (1520, 1046, 0), (1590, 1094, 0), (1640, 1160, 1), (1560, 1200, 0)]
    # x, y, half width, half height, angle, seed, kind
    MOUTHS = [(985, 420, 290, 116, -.12, 4, 'grin'), (1160, 812, 270, 90, .15, 8, 'm'), (640, 650, 150, 60, 1.22, 5, 'm'), (1350, 570, 130, 122, 0, 6, 'ring'), (372, 452, 96, 46, .22, 1, 'm'), (668, 158, 70, 72, 0, 3, 'ring'), (166, 862, 70, 28, -1.2, 9, 'm'), (1690, 256, 66, 26, -1.15, 10, 'm'), (1436, 222, 36, 17, .3, 12, 'm'), (560, 930, 100, 40, .3, 11, 'm'), (850, 712, 70, 28, .2, 15, 'm'), (1520, 872, 50, 46, 0, 16, 'ring'), (1180, 200, 60, 24, .15, 14, 'm')]
    NAILS = [(560 * .86 + 264, 520 * .86 + 185), (1120 * .86 + 264, 640 * .86 + 185), (960 * .86 + 264, 270 * .86 + 185)]

    @staticmethod
    def poly(f, breath=1.):
        Q = np.array(Curse.BODYP, float); P = Q[:, :2]; k = math.sin(f * .35) * 5 * breath; c = np.array([960, 1000]); P = c + (P - c) * [1 + .004 * k, 1 + .006 * k]; r = np.random.default_rng(f // 3); P = P + r.normal(0, 1.6, P.shape); Z = resample(spl(P, True, corners=[i for i in range(len(Q)) if Q[i, 2]], n=8), 5.); T = np.gradient(Z, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-6; rr = np.random.default_rng(6); q = f // 3; return Z + np.c_[-T[:, 1], T[:, 0]] * (9 * noise1(len(Z), 15, rr) + 3.2 * noise1(len(Z), 5, rr) + 1.5 * noise1(len(Z), 9, np.random.default_rng(q)))[:, None]

    @staticmethod
    def tendrils(f):
        """ropes of it that grope the air: rooted on the outline, drawn under the body"""
        r = np.random.default_rng(23); Z = Curse.poly(f); out = []
        for j, (x, y, a0, L_) in enumerate(((300, 365, -2.5, 190), (262, 500, 2.8, 150), (560, 120, -2.4, 170), (760, 70, -.9, 150), (930, 200, -1.9, 230), (1050, 140, -1.5, 150), (1240, 140, -1.2, 260), (1330, 220, -.9, 130), (1480, 170, -.8, 140), (1660, 560, -.2, 190), (1640, 740, .4, 230), (1600, 900, .3, 150), (120, 740, -2.6, 170), (100, 860, 3.0, 120), (1700, 330, .1, 130), (640, 45, -1.7, 120))):
            th = a0 + .18 * math.sin(f * .35 + j); cu = (1 if j % 2 else -1) * (.26 + .10 * math.sin(f * .28 + j * 1.7)); p = np.array([x, y], float) - 40 * np.array([math.cos(a0), math.sin(a0)]); pts = [p]
            for i in range(7): p = p + np.array([math.cos(th), math.sin(th)]) * L_ / 6.; pts.append(p); th += cu
            out.append(dog.Lock.d2(pts, r.uniform(34, 62), 'ink', 0, 5000 + j))
        return out

    @staticmethod
    def mouth(c, m, open_, f, w=2.6, lit='#a15cff'):
        x, y, a, b, an, sd, kd = m; r = np.random.default_rng(sd); R = np.array([[math.cos(an), math.sin(an)], [-math.sin(an), math.cos(an)]]); X = lambda q: np.asarray(q, float) @ R + [x, y]; k = a / 120.
        if kd == 'ring':                                     # lamprey: rings of teeth round a hole
            b = b * (.7 + .3 * open_); t = np.linspace(0, 6.283, 60)[:-1]; E = lambda q, j=0.: X(np.c_[np.cos(t) * a * q * (1 + j * np.sin(t * 5 + sd)), np.sin(t) * b * q * (1 + j * np.cos(t * 4))]); outline(c, E(1.16, .04), w * 1.2, lit, sd, press=.5); c.fill(E(1.16, .04), '#160b30'); c.fill(E(1.0, .03), '#5c1c5a'); c.fill(E(.78, .03), '#2a0a3e'); c.fill(E(.40, .04), '#0a0414')
            for ring, (q0, q1, n) in enumerate(((.98, .64, 15), (.72, .40, 11))):
                for j in range(n):
                    th = 6.283 * (j + .5 * ring + r.uniform(-.2, .2)) / n; d = .20 * (1 + .3 * r.random()) * 10 / n; p = lambda q, th_: X(np.array([[math.cos(th_) * a * q, math.sin(th_) * b * q]]))[0]; A_, B_, T_ = p(q0, th - d), p(q0, th + d), p(q1 * r.uniform(.8, 1.1), th); c.fill(np.array([A_, T_, B_]), '#f1e6ff'); c.fill(np.array([B_, T_, A_ * .45 + B_ * .55]), '#8f6cc8'); c.line(np.array([A_, T_, B_]), w * .7, INK, taper=(.1, .1), seed=j)
            for j in range(26): th = 6.283 * j / 26 + r.normal(0, .05); c.line(X(np.array([[math.cos(th) * a * 1.17, math.sin(th) * b * 1.17], [math.cos(th) * a * 1.28, math.sin(th) * b * 1.28], [math.cos(th) * a * r.uniform(1.35, 1.55), math.sin(th) * b * r.uniform(1.35, 1.55)]])), w * 1.1, '#341a72', taper=(.1, .8), seed=j)
            c.line(np.vstack([E(1.0, .03), E(1.0, .03)[:1]]), w * 1.1, INK, taper=(.02, .02), seed=3); return
        b = b * (.30 + .80 * open_); lift = -.38 if kd == 'grin' else r.uniform(-.25, .1)
        up = np.array([(-a, b * lift), (-a * .66, -b * r.uniform(.25, .5)), (-a * .12, -b * r.uniform(.55, .8)), (a * .48, -b * r.uniform(.3, .6)), (a, b * (lift - .1))]); lo = np.array([(a, up[-1][1]), (a * .72, b * r.uniform(.62, .9)), (a * .08, b * r.uniform(1.0, 1.25)), (-a * .62, b * r.uniform(.7, .95)), (-a, b * lift)]); U, D = spl(up, n=9), spl(lo, n=9)
        lipU = spl(np.vstack([up[0] * [1.06, 1.2], up[1] + [0, -b * .26 - 8], up[2] + [0, -b * .20 - 8], up[3] + [0, -b * .28 - 8], up[4] * [1.06, 1.2]]), n=9); lipD = spl(np.vstack([up[4] * [1.06, 1.2], lo[1] + [0, b * .24 + 8], lo[2] + [0, b * .20 + 10], lo[3] + [0, b * .24 + 8], up[0] * [1.06, 1.2]]), n=9); lip = X(np.vstack([lipU, lipD])); outline(c, lip, w * 1.1, lit, sd, press=.6); c.fill(lip, '#160b30')
        for S_, L_ in ((U, lipU), (D, lipD)):                # cracked lips
            n_ = int(6 + 9 * k)
            for j in range(n_): i = int(r.integers(3, len(S_) - 3)); i2 = min(len(L_) - 1, max(0, i + int(r.integers(-2, 3)))); c.line(X(np.array([S_[i], (S_[i] + L_[i2]) / 2 + r.normal(0, 2, 2), L_[i2] * 1.04])), w * .9, '#3a1c7c' if j % 3 else '#6a38c8', taper=(.2, .7), seed=j)
        M = X(np.vstack([U, D])); c.fill(M, '#1c0630'); c.clip(M)
        tg = spl([(-a * .46, b * 1.4), (-a * .34, b * .55), (-a * .02, b * .22), (a * .34, b * .48), (a * .52, b * 1.4)], n=6); c.fill(X(tg), '#8a2a6c'); c.fill(X(spl([(-a * .30, b * 1.4), (-a * .22, b * .62), (-a * .05, b * .42), (-a * .02, b * 1.4)], n=6)), '#c24a8c'); c.line(X(spl([(a * .02, b * .34), (a * .05, b * .8), (a * .03, b * 1.3)], n=6)), w, '#4a1244', taper=(.2, .4), seed=sd)
        for row, S_, g in ((-1, U, 1), (1, D[::-1], -1)):
            c.fill(X(np.vstack([S_, (S_ + [0, -row * b * .20])[::-1]])), '#6b2a68'); c.fill(X(np.vstack([S_ + [0, -row * b * .06], (S_ + [0, -row * b * .11])[::-1]])), '#a14a98')
            n = int(np.clip(r.integers(8, 13) * k ** .55, 5, 20)); e = np.linspace(.02, .98, n + 1); e[1:-1] += r.normal(0, .26 / n, n - 1); e = np.sort(e); G_ = S_ + [0, -row * b * .17]
            for t0, t1 in zip(e[:-1], e[1:]):
                i0, i1 = int(t0 * (len(G_) - 1)), int(t1 * (len(G_) - 1)); p0, p1 = G_[i0], G_[i1]; wd = abs(p1[0] - p0[0])
                if wd < 5: continue
                ln = min(wd * r.uniform(1.0, 2.3), b * (1.25 if kd == 'grin' else .9)); tp = (p0 + p1) / 2 + [r.normal(0, wd * .12), -row * ln]; q0 = (p0 + tp) / 2 + [-wd * .10, 0]; q1 = (p1 + tp) / 2 + [wd * .10, 0]; T = X(np.vstack([spl([p0, q0, tp], n=5), spl([tp, q1, p1], n=5)])); c.fill(T, '#f3eaff'); c.fill(X(np.vstack([spl([tp, q1, p1], n=5), [p0 * .40 + p1 * .60]])), '#9272cc'); c.line(T, w * .8, INK, taper=(.08, .08), seed=sd)
        c.unclip(); c.line(np.vstack([M, M[:1]]), w * 1.2, INK, taper=(.02, .02), press=.5, seed=sd + 3)
        if kd == 'grin':                                     # drool
            for q in (-.42, .18, .55): p = X(np.array([[a * q, b * .95]]))[0]; c.line(spl([p, p + [3, 60 * open_], p + [-2, 130 * open_ + 30]], n=8), 7, '#c9a0ff', taper=(.1, .3), seed=int(q * 90), alpha=.7)


class Shot:
    id = 'd2'; n = 72
    finish = dict(bloom=.34, vig=.20)
    EVENTS = [(0, 'nail_throw', dict(size=.5, note='three iron nails flung up, sleeve snap')), (1, 'nail_spin', dict(len=11, size=.4, note='three spinning metallic rings, slowing')), (12, 'breath', dict(len=4, size=.4, note='wind-up, cloth creak')), (16, 'whoosh', dict(len=2, size=.8, note='hammer swing')), (18, 'hammer_hit', dict(size=1.0, note='iron on iron, one impact frame')), (19, 'cursed_fire', dict(len=15, size=.8, note='nails streak away with blue fire, doppler L>R')),
              (25, 'curse_drone', dict(len=10, size=.5, note='many mouths muttering')), (32, 'nail_thunk', dict(size=.6)), (33, 'nail_thunk', dict(size=.6)), (34, 'nail_thunk', dict(size=.7)), (36, 'hairpin', dict(size=.8, stage=1, note='spikes burst inside it')), (39, 'hairpin', dict(size=.9, stage=2)), (42, 'hairpin', dict(size=1.0, stage=3, note='biggest, the shape tears')), (43, 'curse_scream', dict(len=3, size=.8, note='many mouths at once, cut off')), (45, 'shred', dict(len=9, size=.7, note='wet paper tearing, ink drops')),
              (54, 'hammer_swing', dict(size=.5, note='hammer flips up')), (57, 'hammer_shoulder', dict(size=.5, note='shaft lands on the shoulder')), (58, 'fire_bed', dict(len=14, size=.4)), (63, 'chin_up', dict(size=.2, note='small cloth, proud exhale'))]

    def setup(self):
        self.C = {}; x, y = grid(); n = fbm(H, W, (420, 160, 60), seed=21)
        d = np.sqrt(((x - 620) / 1200) ** 2 + ((y - 320) / 900) ** 2); self.bgA = ramp(np.clip(d + n * .05, 0, 1.4), [(0, '#5d86f0'), (.25, '#2c4ec6'), (.55, '#1a2478'), (.9, '#0c0d36'), (1.4, '#07061a')])
        d = np.sqrt(((x - 1080) / 1150) ** 2 + ((y - 760) / 760) ** 2); self.bgE = ramp(np.clip(d + n * .05, 0, 1.4), [(0, '#3f6fe0'), (.3, '#2340a8'), (.6, '#141f66'), (.95, '#0a0c30'), (1.4, '#06061a')])
        d = np.sqrt(((x - 1000) / 1080) ** 2 + ((y - 470) / 720) ** 2); self.bgD = ramp(np.clip(d + n * .06, 0, 1.4), VIOLET)
        # low ruined skyline in front of the violet light
        c = Cel(); r = np.random.default_rng(9)
        for col, base, hh in (('#3a1c86', 900, 300), ('#1a0f4c', 985, 230)):
            xx = -60; P = [(-60, 1100)]
            while xx < W + 60: wd = r.uniform(50, 170); h_ = r.uniform(.3, 1) * hh; P += [(xx, base - h_), (xx + wd * r.uniform(.5, 1), base - h_ - r.uniform(-30, 30)), (xx + wd, base - h_ * r.uniform(.5, .9))]; xx += wd
            c.fill(np.array(P + [(W + 60, 1100)]), col)
        self.sky = comp.blur(c.layer(), 2.0)

    def cached(self, k, fn):
        if k not in self.C:
            if len(self.C) > 8: self.C.pop(next(iter(self.C)))
            self.C[k] = fn()
        return self.C[k]

    def frame(self, f):
        if f < 12: return self.throw(f)
        if f < 25: return self.strike(f)
        if f < 31: return self.fly(f)
        if f < 54: return self.hairpin(f)
        return self.grin(f)

    # ------------------------------------------------------------ shared Nobara drawing
    def nobara(self, cx, cy, s, yaw, pitch, roll, expr, f, wind, tyaw=None, troll=None, drop=1.10, tdy=0., sleeves=(1, 1), before=None, after=None, light='blue'):
        L = mn.LIGHT[light]; w = mn.lw(s); cel = Cel(); Hd = N.pose(cx, cy, s, yaw, pitch, roll); hair = N._hair(Hd, f, wind); o = Hd.P3(np.array([(0, -drop, -.06)]))[0]
        T = mn.Torso(o[0], o[1] + tdy, s, yaw * .6 if tyaw is None else tyaw, 0, roll * .3 if troll is None else troll)
        N.back(cel, Hd, light, f, (0, 0), 3, hair); mn.uniform(cel, T, BODY, L, w, 3, 'back'); mn.neck(cel, Hd, L, w, .285, seed=3)
        if before: before(cel, T, L, w)
        mn.uniform(cel, T, BODY, L, w, 3, 'front', sleeves=sleeves)
        if after: after(cel, T, L, w)
        N.head(cel, Hd, expr, light, f, (0, 0), 3, parts='f', L=hair)
        return mn.grade(cel.layer(), L, (cx - 420, cy - 320, cx + 420, cy + 680), 1.0), T

    # ------------------------------------------------------------ 0-12 nails up, eyes follow
    def npos(self, i, t):
        u = t / 12.; e = 1 - (1 - min(1., u)) ** 2.2; x0, x1 = 820 - 40 * i, (560, 400, 640)[i]; y0, y1 = 800, (250, 330, 150)[i]
        return np.array([x0 + (x1 - x0) * e + 30 * math.sin(u * 2 + i), y0 + (y1 - y0) * e]), (1.15 - .75 * u) * t * (1, 1.12, .9)[i] + i * 1.3

    def throw(self, f):
        from jj import hands as HD
        t = f; k = 0 if f < 1 else 1 if f < 3 else f // 3 * 3; u = np.clip((k - 1) / 9., 0, 1); e = float(E3(u, 2)); L = mn.LIGHT['blue']
        im = comp.over(self.bgA.copy(), self.cached(('bf', f // 2), lambda: comp.blur(blueflame([(-100, 2020, 960, 330, 12), (-100, 2020, 1050, 240, 14)], 14, 2.2, '#080f44', cols=('#14247e', '#2146c0', '#3f7ae8'), f=f // 2 * 2, boil=.2), 3.0)))
        if f < 5: im = comp.over(im, self.cached(('sl', f // 2), lambda: (lambda c: (speed_lines(c, -1.95, 60, 7, (500, 1500), '#8fb4ff', 40 + f // 2, alpha=.7), speed_lines(c, -1.95, 46, 10, (500, 1500), '#0a0b30', 60 + f // 2, alpha=.7), c.layer())[-1])(Cel())), (.5, .5, .4, .25, .1)[f])
        ex = dict(EXPR['calm'], lid=1.06, gaze=(-.018 - .016 * e, .012 + .022 * e), brow=(.018, .06, 0), mouth=dict(kind='line', pts=[(-.06, -.66), (0, -.668), (.07, -.650)], lip=[(-.035, -.735), (0, -.742), (.035, -.735)])) if k > 1 else dict(EXPR['fierce'])
        pt, yw, rl = -.10 + .34 * e, -.34 - .10 * e, -.06 - .08 * e
        hx, hy = [(1010, 1010), (900, 800), (880, 770), (874, 776), (872, 782), (872, 786)][min(5, {0: 0, 1: 1, 3: 2, 6: 3, 9: 4}.get(k, 5))]
        def drawN(): return self.nobara(1290, 560 + (18 if k == 0 else 0), 252, yw, pt, rl, ex, k, (.02, .16 - .40 * (1 - u) * (k > 0), .35), tyaw=-.26, sleeves=(0, 1))[0]
        body = self.cached(('A', k), drawN)
        def hand():
            hc = Cel(); w = mn.lw(252); cam = HD.Cam(hx, hy, 182, 6.0); po = 'fist' if k == 0 else 'reach' if k == 1 else 'open'; h = HD.Hand(po, 'R', HD.euler(118 if k else 150, -20 if k else 10, 26 if k else -30), (0, 0, 0), HD.FEMALE); g = HD.render([h], cam, light=(.75, .3, .55)); W_ = lambda q: cam.px(h.world(np.array([0, q, 0])))
            HD.ink(hc, g, hpal(L), w[0] * .85, 3, inner=.6); b, m = W_(-.26), W_(-1.5); sh = np.array([1080, 1000]); mn.sleeve2(hc, [b, m + [10, 10], (m + sh) / 2 + [-20, 110], sh + [40, 240]], [(0, 74), (.35, 104), (.7, 130), (1, 150)], L, w, 27, bunch=(.12,)); return mn.grade(hc.layer(), L, (700, 500, 1200, 1080), .8)
        hl = self.cached(('Ah', k), hand); im = comp.over(im, body)
        if f in (0, 1, 2): hl = smear(hl, 40, -190 + 60 * f, 9) * .55 + hl * .45
        im = comp.over(im, hl)
        # the nails: smeared arcs while they are fast, clear and glinting as they slow
        c = Cel()
        for i in range(3):
            p, a = self.npos(i, t); v = self.npos(i, t)[0] - self.npos(i, t - 1)[0] if t > 0 else np.array([0, -200.]); L_ = (250, 220, 190)[i]
            if t < 5:
                tr = np.array([self.npos(i, max(0, t - q))[0] for q in (0, .5, 1., 1.6)]); c.line(spl(tr + np.array([0, 4.]), n=8), L_ * .10, '#9fc4ff', taper=(.02, .95), alpha=.55, seed=i, press=0)
            for q, al in ((.55, .28), (.28, .5)):
                if t > 0 and t < 9: pp, aa = self.npos(i, t - q); c.group(); nail(c, pp, aa, L_, math.cos(aa * .8)); c.ungroup(al)
            nail(c, p, a, L_, .55 + .45 * abs(math.cos(a * .8 + i)))
        lay = dog.rim2(c.layer(), 2, 2, '#cfe6ff', .8, .5); im = comp.over(im, lay)
        for i in range(3):
            p, a = self.npos(i, t); g = abs(math.sin(a * 1.3 + i)) ** 6
            if g > .25: im = im + (comp.radial(p[0], p[1] - 10, 120, 8, 1.4) + comp.radial(p[0], p[1] - 10, 8, 90, 1.4))[..., None] * rgb('#d8eaff') * g * .9 + comp.radial(p[0], p[1], 90, 90, 2.)[..., None] * rgb('#4f86ff') * .35 * g
        dx, dy = shake(f, 3, 3, 3, 0); return comp.cam2d(im, W / 2 + dx, H / 2 - 18 * e + 18 + dy, 1.06 - .02 * e, -.02 + .03 * e)

    # ------------------------------------------------------------ 12-25 wind-up, smear, hit, follow-through
    def strike(self, f):
        from jj import hands as HD
        L = mn.LIGHT['blue']; w = mn.lw(215); S = 205
        # key: head (cx, cy, yaw, pitch, roll), torso yaw (right shoulder back, then thrown forward), torso roll, elbow, hand, hammer angle (screen), wind
        K = {12: ((800, 405, .60, .02, .02), .30, -.04, (362, 716), (448, 490), -2.10, (.10, -.04, .3)), 14: ((778, 420, .58, -.06, -.04), .20, -.08, (326, 664), (424, 440), -2.45, (.16, .02, .3)), 16: ((762, 430, .56, -.10, -.08), .14, -.11, (308, 636), (420, 405), -2.68, (.20, .04, .2)),
             17: ((850, 420, .72, -.08, .08), .80, .06, (900, 872), (1118, 870), -1.44, (-.50, -.10, .3)), 18: ((890, 430, .82, -.10, .12), .96, .10, (928, 884), (1140, 868), -1.47, (-.64, -.04, .3)), 20: ((915, 446, .86, -.12, .16), 1.02, .12, (980, 900), (1226, 852), -.46, (-.38, .10, .3)), 22: ((908, 442, .82, -.10, .14), .98, .10, (968, 898), (1210, 870), -.32, (-.12, .14, .3))}
        kf = [q for q in sorted(K) if q <= f][-1]; hd, ty, tr, el, hn, ang, wind = K[kf]; t = f - 12
        Hd_ = N.pose(hd[0], hd[1], S, *hd[2:]); o_ = Hd_.P3(np.array([(0, -1.10, -.06)]))[0]; T_ = mn.Torso(o_[0], o_[1], S, ty, 0, tr); sh = T_.P3(np.array([(-1.04, -.72, .12)]))[0]
        bgk = self.bgA[:, ::-1]; im = bgk.copy()
        c = Cel(); speed_lines(c, .04, 60 if f >= 17 else 26, 8, (300, 1700), '#7fa8ff', 50 + f, alpha=.6); speed_lines(c, .04, 54 if f >= 17 else 26, 11, (300, 1700), '#080a2c', 90 + f, alpha=.75); im = comp.over(im, c.layer(), .75 if f >= 17 else .35)
        ex = dict(EXPR['fierce']) if f < 17 else dict(EXPR['fierce'], lid=1.05, mouth=dict(kind='open', up=[(-.120, -.640), (-.05, -.622), (.03, -.626), (.130, -.616)], lo=[(-.120, -.640), (-.10, -.740), (.0, -.775), (.10, -.735), (.130, -.616)], teeth=[(-.12, -.664), (0, -.660), (.13, -.645)], tongue=[(-.06, -.76), (0, -.715), (.06, -.76), (0, -.80)], lip=[(-.04, -.825), (0, -.832), (.04, -.825)]))
        def drawN(): return self.nobara(hd[0], hd[1], S, hd[2], hd[3], hd[4], ex, kf, wind, tyaw=ty, troll=tr, sleeves=(0, 1))[0]
        body = self.cached(('B', kf), drawN)
        def arm():
            hc = Cel(); cam = HD.Cam(hn[0], hn[1], 128, 7.0); el_ = np.array(el, float); d = np.array(hn, float) - el_; fa = math.degrees(math.atan2(-d[1], d[0])) - 90; h = HD.Hand('hammer', 'R', HD.euler(18, -6, fa), (0, 0, 0), HD.FEMALE); hp = cam.px(h.world(np.array([0., .80, -.1]))); e = np.array([math.cos(ang), math.sin(ang)]); b = cam.px(h.world(np.array([0, -.26, 0])))
            u_ = (el_ - hn) / np.hypot(*(el_ - hn)) * 110; bx = (min(hn[0], hn[0] + u_[0]) - 115, min(hn[1], hn[1] + u_[1]) - 115, max(hn[0], hn[0] + u_[0]) + 115, max(hn[1], hn[1] + u_[1]) + 115); g = HD.render([h], cam, light=(.6, .4, .55), clip=bx)
            wd = [v * S / 127. * .80 for v in (36, 43, 50, 57, 64)]; up_ = dict(order='uf', bow=10 if kf < 17 else -8, folds=3, bold=1.6, bunch=False, openend=True)
            steel_hammer(hc, hp - e * 78, hp + e * 246, 88, w[0] * .9, flip=-1 if kf < 17 else 1, shw=.15); HD.ink(hc, g, hpal(L), w[0] * .8, 3, inner=.6)
            mn.arm(hc, sh, el_, b, wd, L, w, 31, **up_)
            # armhole seam and the roll of the deltoid, so the arm is hung on the shoulder
            s0, s1, s2 = T_.P3(np.array([(-.84, -.30, .10), (-.80, -.72, .52), (-.92, -1.18, .40)])); hc.line(spl([s0, s1, s2], n=10), w[0] * .8, L['cloth_ln'], taper=(.25, .3), press=.3, seed=5); mn.fold(hc, spl([s1 + [4, -6], s1 * .6 + sh * .4 + [-6, 14], sh * .7 + el_ * .3 + [0, 10]], n=10), 15, L, w[1], 6)
            return mn.grade(hc.layer(), L, (hn[0] - 300, hn[1] - 300, hn[0] + 300, hn[1] + 300), .8), hp + e * 246
        al, head = self.cached(('Ba', kf), arm)
        # nails hang in a row in front of her, waiting
        nx, ny = 1352, 624; sp_ = 62 if f < 17 else 46
        def nails(dx=0., fire=0.):
            c = Cel()
            for i in range(3): y_ = ny - sp_ + sp_ * i; nail(c, (nx + dx + (0, 24, 6)[i] + 4 * math.sin(f * .9 + i), y_ + (3 * math.sin(f * .7 + i * 2) if f < 18 else 0)), .02 * (i - 1), 230, 1., 2.8)
            return dog.rim2(c.layer(), 2, 2, '#cfe6ff', .8, .5)
        if f < 17: im = comp.over(im, body); im = comp.over(im, al); im = comp.over(im, nails())
        elif f == 17:                                        # smear: one crescent of iron
            c = Cel(); pv = np.array([700, 792]); a0, a1 = -2.40, -.36; aa = np.linspace(a0, a1, 40); R1 = np.interp(np.linspace(0, 1, 40), [0, .4, 1], [735, 715, 560]); R0 = np.interp(np.linspace(0, 1, 40), [0, .35, 1], [729, 628, 440])
            o = pv + np.c_[np.cos(aa) * R1, np.sin(aa) * R1]; q = pv + np.c_[np.cos(aa) * R0, np.sin(aa) * R0]; P_ = np.vstack([o, q[::-1]]); outline(c, P_, 4, INK, 3); c.fill(P_, '#99a3c4'); m = (o * .62 + q * .38); c.fill(np.vstack([o, m[::-1]]), '#e9f0ff'); m2 = o * .18 + q * .82; c.fill(np.vstack([m2, q[::-1]]), '#4a5278')
            r = np.random.default_rng(4)
            for j in range(22): i0 = int(r.integers(0, 18)); i1 = i0 + int(r.integers(8, 20)); v = r.uniform(.05, .95); c.line((o * v + q * (1 - v))[i0:i1], r.uniform(2, 6), INK if r.random() < .5 else '#ffffff', taper=(.6, .3), seed=j)
            im = comp.over(im, body); lay = c.layer(); im = comp.over(im, smear(lay, 60, 60, 9) * .6 + lay * .5); im = comp.over(im, smear(al, 70, 40, 7) * .5 + al * .6); im = comp.over(im, nails())
        elif f == 18:                                        # impact drawing
            c = Cel(); c.fill(np.array([(0, 0), (W, 0), (W, H), (0, H)]), '#f2f6ff'); focus_lines(c, 1240, 622, 170, 1900, 150, 26, '#05060f', 5, ry=.8, clump=.9); focus_lines(c, 1240, 622, 290, 1900, 50, 16, '#2b5fe0', 6, ry=.8, clump=.9)
            base = c.layer()[..., :3]; sub = comp.over(np.ones((H, W, 3), f32), body); sub = comp.over(sub, al); a = np.maximum(body[..., 3], al[..., 3])[..., None]; Lm = cv2.GaussianBlur((sub @ np.array([.3, .59, .11], f32)), (0, 0), 1.2); tn = smooth(.34, .38, Lm)[..., None]; sb = rgb('#05060f') * (1 - tn) + rgb('#f2f6ff') * tn; sb = np.where((Lm > .20)[..., None] & (Lm < .36)[..., None], rgb('#2b5fe0'), sb); im = base * (1 - a) + sb * a
            c = Cel(); st = fx.star_poly(1240, 622, 70, 300, 17, 5, asp=.9); c.fill(np.array(st) if not isinstance(st, list) else st, '#05060f'); st = fx.star_poly(1240, 622, 40, 190, 15, 8, asp=.9); c.fill(np.array(st) if not isinstance(st, list) else st, '#ffffff'); im = comp.over(im, c.layer()); dx, dy = shake(f, 20, 5); return comp.cam2d(im, W / 2 + dx, H / 2 + dy, 1.08, .015)
        else:                                                # follow-through, nails already fire
            u = f - 19; c = Cel(); sx = 1440 + u * 190
            for i in range(3):
                y_ = ny - 52 + 52 * i + (i - 1) * u * 22; x1 = sx + (0, 120, 40)[i]
                fx.flame_stream(c, np.array([(1290 + u * 40, y_ + 6), (1450 + u * 90, y_ - 6), (x1 - 120, y_), (x1 + 160, y_)])[::-1], 50 + 10 * (i == 1), f / 24., 11 + i, fx.BLUEPAL, speed=1400, smooth=.35)
                if x1 < W + 200: nail(c, (x1, y_), 0.02 * (i - 1), 240, 1., 2.8)
            fl = c.layer(); im = comp.screen(im, comp.blur(fl, 30)[..., :3] * .5); im = comp.over(im, body); im = comp.over(im, al if f > 19 else smear(al, 60, 120, 9) * .6 + al * .4); im = comp.over(im, fl); im = im + fx.hot(fl, .75, amt=.2)
            if u < 4: im = im + comp.radial(1380, 622, 420 - 60 * u, 300, 1.7)[..., None] * rgb('#8fc0ff') * (.6 - .15 * u)
            c = Cel(); fx.shards(c, 1290, 610, (u + .5) / 24., 5, 26, 900, size=22, black='#0a1a6a', red='#8fd2ff') if False else None
            r = np.random.default_rng(77); c = Cel()
            for q in range(36): a = r.uniform(-1.2, 1.2) + (math.pi if r.random() < .3 else 0); v = r.uniform(120, 900); p = np.array([1260, 622]) + np.array([math.cos(a), math.sin(a)]) * v * (u + 1) / 8.; e = np.array([math.cos(a), math.sin(a)]); c.line(np.array([p - e * 26, p - e * 8, p]), r.uniform(1.5, 4), '#cfeaff', taper=(.7, .1), seed=q)
            im = im + c.layer()[..., :3] * max(0, 1 - u / 6.)
        dx, dy = shake(f, 16, 5, 2.4, 18) if f >= 18 else (0, 0); z = 1.04 + (.0035 * t if f < 17 else .02)
        return comp.cam2d(im, W / 2 + dx + (-14 * (f - 12) / 5. if f < 17 else 0), H / 2 + dy, z, -.012 if f < 17 else .012)

    # ------------------------------------------------------------ 25-31 with the nails
    def fly(self, f):
        t = f - 25; x, y = grid(); d = np.abs(y - 540) / 620.; im = ramp(np.clip(d + .06 * nz((600, 60), (1, .5), 31), 0, 1.3), [(0, '#4a7cf2'), (.25, '#2546b8'), (.6, '#141a62'), (1.2, '#080820')]); v = smooth(900, 1920, x)[..., None]; im = im * (1 - v * .8) + ramp(np.clip(d, 0, 1.2), [(0, '#e9c8ff'), (.3, '#8b45e8'), (.7, '#341690'), (1.2, '#120a3c')]) * v * .8
        c = Cel(); r = np.random.default_rng(300 + f); speed_lines(c, 0.0, 90, 10, (500, 2200), '#9fc4ff', 400 + f, alpha=.6); speed_lines(c, 0.0, 80, 16, (500, 2200), '#070926', 500 + f, alpha=.8); im = comp.over(im, smear(c.layer(), 160, 0, 7), .9)
        c = Cel(); DP = [(980, 300, .62, 2), (1150, 800, 1.25, 1), (820, 560, .95, 0)]
        for (nx, ny, sc, i) in sorted(DP, key=lambda q: q[2]):
            nx += t * 34 * sc + 12 * math.sin(f * 1.7 + i); ny += 7 * math.sin(f * 2.1 + i * 2) * sc; L_ = 620 * sc
            P = np.array([(nx + L_ * .56, ny), (nx + L_ * .25, ny - 3), (nx - L_ * .3, ny + 6 * math.sin(f + i)), (nx - L_ * .9, ny + 10 * math.sin(f * .8 + i)), (nx - L_ * 1.6, ny)]); fx.flame_stream(c, P, 108 * sc, f / 24., 21 + i, fx.BLUEPAL, speed=2400, smooth=.30, core=True)
            nail(c, (nx, ny), .0, L_, 1., 2.4 + 2.4 * sc); c.line(np.array([(nx - L_ * .4, ny - L_ * .022), (nx, ny - L_ * .024), (nx + L_ * .34, ny - L_ * .012)]), 5 * sc, '#ffffff', taper=(.3, .4), seed=i)
        lay = c.layer(); im = comp.screen(im, comp.blur(lay, 34)[..., :3] * .45); im = comp.over(im, lay); im = im + fx.hot(lay, .75, amt=.18)
        dx, dy = shake(f, 7, 9); return comp.cam2d(im, W / 2 + dx, H / 2 + dy, 1.07, -.06 + .010 * t)

    # ------------------------------------------------------------ 31-54 hairpin
    STAGE = [(36, 0), (39, 1), (42, 2)]

    def spikes(self, c, i, t, seed):
        """one burst: long black-blue thorns thrown out of a nail, grown on two drawings with overshoot"""
        x, y = Curse.NAILS[i]; g = [.62, 1.08, 1.0][min(int(t), 2)]; r = np.random.default_rng(seed); n = (11, 14, 18)[i]; big = (330, 430, 620)[i]; a0 = r.uniform(0, 6.28)
        sp = []
        for j in range(n):
            a = a0 + 6.283 * (j + r.uniform(-.35, .35)) / n; k = r.choice([.45, .75, 1.0, 1.5], p=[.25, .3, .3, .15]); ln = big * k * g * r.uniform(.85, 1.15); w = (26, 34, 44)[i] * k ** .5 * r.uniform(.8, 1.3); e = np.array([math.cos(a), math.sin(a)]); q = np.array([-e[1], e[0]]); b = q * ln * r.normal(0, .05); p0 = np.array([x, y]) + e * 18
            sp.append((k, p0, e, q, ln, w, b))
        for k, p0, e, q, ln, w, b in sorted(sp, key=lambda s: s[0]):
            L_ = spl([p0 - q * w * .6, p0 + e * ln * .22 - q * w + b * .2, p0 + e * ln * .6 - q * w * .42 + b * .7, p0 + e * ln + b], n=7); R_ = spl([p0 + e * ln + b, p0 + e * ln * .6 + q * w * .42 + b * .7, p0 + e * ln * .22 + q * w + b * .2, p0 + q * w * .6], n=7); P = np.vstack([L_, R_]); outline(c, P, 5, '#2b5fe0', int(k * 10)); c.fill(P, '#05060f'); s_ = -1 if (q @ np.array([-.6, -.8])) < 0 else 1
            A_ = L_ if s_ < 0 else R_[::-1]; M_ = (L_ * .62 + R_[::-1] * .38) if s_ < 0 else (L_ * .38 + R_[::-1] * .62); c.fill(np.vstack([A_, M_[::-1]]), '#2446b4'); M2 = A_ * .80 + M_ * .20; c.fill(np.vstack([A_, M2[::-1]]), '#8fd0ff'); c.line(P, 2.2, '#02030a', taper=(.05, .05), seed=3)

    def curse(self, f, open_):
        def draw():
            c = Cel(W, H + 130, 0, -120); P = Curse.poly(f); tl = Curse.tendrils(f)
            for l in tl: outline(c, l.poly, 7, '#7a3cf0', l.id, press=.6)
            outline(c, P, 9, '#7a3cf0', 3, press=.6)
            for l in tl: c.fill(l.poly, '#080512'); c.fill(l.side(.15, .8, -1.0, -.5), '#2c1470')
            c.fill(P, '#080512'); c.clip(P); r = np.random.default_rng(17)
            for q in range(70):                              # folds and ridges that catch a little violet
                x0, y0 = r.uniform(250, 1700), r.uniform(120, 1080); a = math.atan2(y0 - 560, x0 - 960) + r.normal(1.57, .5); l = r.uniform(60, 260); e = np.array([math.cos(a), math.sin(a)]); p = np.array([x0, y0]); c.line(spl([p, p + e * l * .5 + np.array([-e[1], e[0]]) * l * .14, p + e * l], n=8), r.uniform(1.5, 6), '#24124e' if r.random() < .75 else '#4a2596', taper=(.4, .6), seed=q)
            c.unclip()
            for m in Curse.MOUTHS: Curse.mouth(c, m, np.clip(open_ + .22 * math.sin(f * .8 + m[5] * 1.7), 0, 1.2), f)
            lay = c.rgba(); lay = cv2.warpAffine(lay, np.float32([[.86, 0, 960 - .86 * 960 + 130], [0, .86, 1080 - .86 * 1080 + 34 - .86 * 120]]), (W, H), flags=cv2.INTER_AREA); dark = (lay[..., 0] + lay[..., 2] < .16).astype(f32); lay = facets(lay, 15, 17, '#3d1f96', .46, .30, 3, only=dark); lay = facets(lay, -15, 17, '#3d1f96', .40, .32, 5, only=dark); lay = facets(lay, 5, 6, '#b888ff', .30, .30, 7, only=dark); return facets(lay, -5, 6, '#b888ff', .30, .30, 9, only=dark)
        return self.cached(('cu', f, round(open_, 2)), draw)

    def scraps(self, f, k):
        """what is left of it: torn slabs of ink (some still with a mouth), curling ribbons, broken thorns, thrown out and shrinking"""
        c = Cel(); r = np.random.default_rng(8); e = 1 - (1 - min(1., k)) ** 1.8; C0 = np.array([990, 520]); sc = max(.12, 1 - .62 * k)
        def fly(p, v, lift): d = p - C0; L_ = np.hypot(*d) + 1; return p + d / L_ * v * e + np.array([0, -lift * k + 420 * k * k])
        # ribbons
        for j in range(22):
            a = r.uniform(0, 6.283); p = C0 + np.array([math.cos(a) * r.uniform(120, 560), math.sin(a) * r.uniform(80, 400)]); q = fly(p, r.uniform(260, 760), r.uniform(40, 200)); th = a + r.normal(0, .5) + 1.6 * e * r.choice([-1, 1]); L_ = r.uniform(170, 460) * sc; cu = r.choice([-1, 1]) * (.5 + 1.2 * e); w = r.uniform(26, 70) * sc; pts = [q]; x_ = q
            for i in range(6): th += cu * .34; x_ = x_ + np.array([math.cos(th), math.sin(th)]) * L_ / 6; pts.append(x_)
            l = dog.Lock.d2(pts, w, 'swp', r.uniform(-.2, .2), 4000 + j); outline(c, l.poly, 4, '#02020a', j); c.fill(l.poly, '#080512'); c.fill(l.side(.05, .7, -1.02, -.45), '#3d1f96'); c.fill(l.side(.12, .55, -1.0, -.82), '#c79aff')
        # thorns that broke off
        for j in range(16):
            i = j % 3; a = r.uniform(0, 6.283); p = np.array(Curse.NAILS[i]) + np.array([math.cos(a), math.sin(a)]) * r.uniform(40, 220); q = fly(p, r.uniform(320, 900), 120); th = a + r.normal(0, 1.2) + 4 * e * r.choice([-1, 1]); d = np.array([math.cos(th), math.sin(th)]); n = np.array([-d[1], d[0]]); L_, w = r.uniform(90, 300) * sc, r.uniform(12, 34) * sc; b0 = np.array([q - d * L_ * .4 - n * w * .7, q - d * L_ * .30 + n * w * .2, q - d * L_ * .42 + n * w * .8]); P = np.vstack([b0, [q + n * w * .7, q + d * L_ * .6, q - n * w * .7]]); outline(c, P, 4, '#2b5fe0', j); c.fill(P, '#05060f'); c.fill(np.array([b0[0], q - n * w * .7, q + d * L_ * .6, q - n * w * .1]), '#2446b4'); c.fill(np.array([q - n * w * .7, q + d * L_ * .6, q - n * w * .45]), '#8fd0ff')
        # slabs
        S_ = [(560, 380, 150), (760, 300, 170), (1060, 280, 190), (1330, 330, 170), (520, 640, 150), (830, 560, 200), (1180, 600, 210), (1460, 620, 160), (700, 860, 180), (1040, 880, 210), (1360, 900, 170), (430, 900, 130), (1640, 420, 110), (960, 170, 110)]
        for j, (x, y, R_) in enumerate(S_):
            p = np.array([x, y]) * [.9, .9] + [110, 90]; q = fly(p, (.55 + .5 * r.random()) * 620, r.uniform(20, 160)); ro = r.normal(0, 1.1) * e; n = int(r.integers(9, 14)); an = np.sort(np.linspace(0, 6.283, n, endpoint=False) + r.normal(0, .16, n)); rad = R_ * sc * (.62 + .55 * r.random(n)) * (1 - .22 * e * r.random(n)); rad[r.integers(0, n, 3)] *= 1.45; V = np.c_[np.cos(an + ro) * rad * 1.15, np.sin(an + ro) * rad * .85] + q; T = [V[0]]
            for i in range(n):                               # torn edge: short rips between the corners
                A_, B_ = V[i], V[(i + 1) % n]; m = int(r.integers(2, 5)); dv = B_ - A_; nv = np.array([-dv[1], dv[0]]) / (np.hypot(*dv) + 1e-6)
                for t in np.sort(r.uniform(.1, .9, m)): T.append(A_ + dv * t + nv * r.normal(0, .10) * R_ * sc * (1 + e))
                T.append(B_)
            T = np.array(T[:-1]); outline(c, T, 6, '#7a3cf0', j, press=.6); c.fill(T, '#080512'); c.clip(T); c.fill(T + np.array([14, 18]) * sc, '#3d1f96'); c.fill(T + np.array([26, 32]) * sc, '#080512')
            if j in (1, 3, 5, 6, 9, 10): Curse.mouth(c, (q[0], q[1], R_ * .62 * sc, R_ * .34 * sc, ro + r.normal(0, .3), 30 + j, 'ring' if j == 3 else 'm'), 1.0, f)
            else:
                for z in range(4): a0 = r.uniform(0, 6.283); pp = q + r.normal(0, R_ * .3 * sc, 2); c.line(spl([pp, pp + np.array([math.cos(a0), math.sin(a0)]) * R_ * .3 * sc, pp + np.array([math.cos(a0 + .5), math.sin(a0 + .5)]) * R_ * .6 * sc], n=6), 4 * sc, '#4a2596', taper=(.4, .6), seed=z)
            c.unclip(); c.line(T[: len(T) // 3], 3.2, '#d0a8ff', taper=(.3, .3), seed=j)
        return c.layer()

    def hairpin(self, f):
        t = f - 31; tw = f // 2 * 2; x, y = grid(); im = self.bgD.copy()
        rays = self.cached(('ray', tw // 4), lambda: (lambda c: (focus_lines(c, 990, 470, 420, 1800, 70, 34, '#e6c4ff', 7 + tw // 4, ry=.8, clump=.8), comp.blur(c.layer(), 5))[-1])(Cel())); im = comp.over(im, rays, .16); im = comp.over(im, self.sky)
        st = [s for s, i in self.STAGE if f >= s]; nst = len(st); ls = f - st[-1] if st else 99
        op = .30 + .15 * math.sin(f * .5) if f < 32 else .55 if f < 36 else [.8, .95, 1.15][max(0, nst - 1)]
        cur = self.curse(f // 3 * 3 if f < 36 else f // 2 * 2, op); ox, oy = (0, 0)
        if f >= 36 and ls < 3: ox, oy = ([14, -10, 4][ls] * (1, -1, 1)[nst - 1], [-12, 8, -3][ls])
        sh = f - 45
        sc = Cel()
        for s, i in self.STAGE:
            if f >= s: self.spikes(sc, i, f - s, 30 + i)
        spl_ = sc.layer(); spl_ = dog.rim2(spl_, 2, 2, '#bfe4ff', .6, .5)
        if sh < 0:
            lay = shifted(cur, ox, oy) if (ox or oy) else cur
            if f >= 43: q_ = 1.025 + .02 * (f - 43); lay = cv2.warpAffine(lay, np.float32([[q_, 0, 960 * (1 - q_)], [0, q_, 560 * (1 - q_)]]), (W, H))
            a = lay[..., 3]
            im = comp.screen(im, np.clip(comp.blur(a, 26) - a, 0, 1)[..., None] * rgb('#c88cff') * .5); im = comp.over(im, lay)
            im = comp.over(im, spl_)
            if f >= 43:                                      # light leaks through it before it goes
                c = Cel(); r = np.random.default_rng(70)
                for j in range(9 + 9 * (f - 43)):
                    i = j % 3; a0 = r.uniform(0, 6.283); p = np.array(Curse.NAILS[i], float); L_ = r.uniform(120, 560); pts = [p]
                    for q in range(5): a0 += r.normal(0, .35); p = p + np.array([math.cos(a0), math.sin(a0)]) * L_ / 5; pts.append(p)
                    c.line(np.array(pts), r.uniform(4, 13), '#f6e8ff', taper=(.05, .8), press=.6, tremor=.6, seed=j)
                lk = c.layer() * lay[..., 3:4]; im = comp.over(im, lk); im = im + comp.blur(lk[..., :3], 14) * rgb('#b06cff') * .9
        else:                                                # it comes apart: every cell is thrown out, edges eaten away
            k = .05 + (sh + .6) / 10.; im = im + comp.radial(990, 470, 900, 640, 1.4)[..., None] * rgb('#e8ccff') * (.20 + .30 * min(1, k)); im = comp.over(im, self.scraps(f, k)); c = Cel(); r = np.random.default_rng(90)
            for j in range(130):                             # ink rain of what it was
                a = r.uniform(0, 6.283); v = r.uniform(260, 1500); s0 = r.uniform(0, 260); p = np.array([980 + math.cos(a) * (s0 + v * (1 - (1 - min(1, k)) ** 2)), 520 + math.sin(a) * (s0 + v * .7 * (1 - (1 - min(1, k)) ** 2)) + 260 * k * k]); e = np.array([math.cos(a), math.sin(a) + .4 * k]); e /= np.hypot(*e); q = np.array([-e[1], e[0]]); z = (2 + 20 * r.random() ** 2.4) * (1 - .5 * k); c.fill(spl([p - e * z * 2.6, p + q * z * .45, p + e * z, p - q * z * .45], True, corners=(0,), n=5), '#080512' if r.random() < .8 else '#3f78ff')
            im = comp.over(im, c.layer())
            # the light that was inside it, in drawn shafts, and the three nails falling out
            if sh >= 1:
                c = Cel(); focus_lines(c, 990, 500, 150 + 26 * sh, 1700, 46, 40, '#fff4ff', 90 + sh // 2, ry=.85, clump=.9); im = comp.screen(im, comp.blur(c.layer(), 1.4)[..., :3] * rgb('#e2b8ff') * .34)
            c = Cel()
            for i, (hx, hy) in enumerate(Curse.NAILS): u = sh + 1; nail(c, (hx + (-34, 30, 10)[i] * u * .8 + (960 - hx) * min(1, u / 9.) * .35, hy + (-26, -40, -60)[i] * u + 8.6 * u * u), (.7, -.9, .5)[i] * u + i, 250, .45 + .55 * abs(math.cos(u * .9 + i)), 3.2)
            im = comp.over(im, dog.rim2(c.layer(), 2, -2, '#f2dcff', .9, .5))
        # nails coming in and where they sit
        c = Cel()
        for i, (hx, hy) in enumerate(Curse.NAILS):
            hit = 32 + i; p1 = np.array([hx, hy], float); d = np.array([.94, .34 - .30 * (i == 2)]); d /= np.hypot(*d)
            if f < hit: p = p1 - d * (hit - f) * 620 - d * 80; fx.flame_stream(c, np.array([p + d * 130, p, p - d * 260, p - d * 620]), 46, f / 24., 5 + i, fx.BLUEPAL, speed=2400, smooth=.35); nail(c, p, math.atan2(d[1], d[0]), 220, 1., 2.6)
            elif f < 43: nail(c, p1 - d * 60 + np.array([ox, oy]), math.atan2(d[1], d[0]) + (.10 * (-1) ** f if f - hit < 2 else 0), 190, 1., 2.8)
        nl = c.layer(); im = comp.screen(im, comp.blur(nl, 20)[..., :3] * .5); im = comp.over(im, nl)
        for i, (hx, hy) in enumerate(Curse.NAILS):
            u = f - (32 + i)
            if 0 <= u < 4: im = im + comp.radial(hx, hy, 240 - 40 * u, 200, 1.6)[..., None] * rgb('#8fc8ff') * (.9 - .22 * u); c = Cel(); st_ = fx.star_poly(hx, hy, 16, 150 - 30 * u, 11, 7 + i + u); c.fill(np.array(st_) if not isinstance(st_, list) else st_, '#dff2ff'); im = comp.over(im, c.layer(), .9 if u < 2 else .4)
            if 34 <= f < 36: im = im + comp.radial(hx, hy, 150, 150, 1.8)[..., None] * rgb('#3f86ff') * .7   # they glow, waiting
        for s, i in self.STAGE:
            u = f - s; hx, hy = Curse.NAILS[i]
            if u == 0: im = im * .35 + (rgb('#0b1236') * .3); c = Cel(); focus_lines(c, hx, hy, 120, 1900, 90, 18, '#bfe4ff', s, ry=.85, clump=.8); im = comp.over(im, c.layer(), .9); im = comp.over(im, spl_); im = im + comp.radial(hx, hy, 420, 360, 1.5)[..., None] * rgb('#9fd8ff') * 1.1
            elif 0 < u < 5: im = im + comp.radial(hx, hy, 700, 520, 1.8)[..., None] * rgb('#3f7cff') * (.6 - .13 * u); c = Cel(); fx.shockwave(c, hx, hy, u / 6., 60, 900, .85, 22, s, col='#cfe8ff', shade='#3a56c0'); im = comp.over(im, c.layer(), .8)
        dx, dy = (0, 0)
        for s, i in self.STAGE:
            if f >= s: q = shake(f, 13 + 5 * i, 5 + i, 1.9, s); dx += q[0]; dy += q[1]
        z = 1.05 + .0018 * t + (.02 * (1 - ls / 3.) if ls < 3 else 0); fade = smooth(51, 54, f) * .5
        im = im + fade * rgb('#f0e2ff') if fade else im
        return comp.cam2d(im, W / 2 + dx, H / 2 - 6 + dy, z, .004 * math.sin(f * .7))

    # ------------------------------------------------------------ 54-72 hammer on the shoulder
    def grin(self, f):
        from jj import hands as HD
        t = f - 54; L = mn.LIGHT['blue']; fk = f // 2 * 2; s = 288
        def fire():
            b = comp.over(self.bgE.copy(), comp.blur(blueflame([(-100, 2020, 860, 520, 13), (-100, 2020, 960, 400, 15)], 4, 2.2, '#080f44', cols=('#121f74', '#1b37ac', '#2a55d8'), f=fk, boil=.22), 2.4)); bx = []
            r = np.random.default_rng(3)
            for i in range(12): v = r.uniform(26, 60); bx.append((r.uniform(40, 1880) + 20 * math.sin(t * .3 + i), (r.uniform(0, 700) - v * t * .4) % 760, r.uniform(12, 32)))
            fl = blueflame([(-100, 2020, 900, 470, 12), (-100, 2020, 985, 390, 15), (-100, 2020, 1070, 250, 19)], 9, 2.6, '#0a1766', boxes=bx, f=fk, boil=.24); g = comp.blur(fl, 34); b = comp.screen(b, g[..., :3] * .5); return comp.over(b, fl)
        im = self.cached(('fire', fk), fire).copy()
        im = im + comp.radial(1500, 330, 900, 620, 1.8)[..., None] * rgb('#8a4df0') * .30 * math.exp(-t / 7.)        # the last violet of what died
        # keys on ones for the flip, then a settle: (hand x, y, roll, dip, face)
        K = [(0, 1010, 1080, 60, 10, 0), (1, 900, 1000, 14, -6, 0), (2, 800, 930, -24, -16, 0), (3, 786, 962, -46, 4, 1), (4, 800, 1000, -42, 16, 1), (6, 800, 990, -37, 6, 2), (8, 800, 985, -38, 0, 2), (12, 800, 984, -38, 1, 2)]
        k = [q for q in K if q[0] <= t][-1]; hx, hy, hr, dip, face = k[1:]; settle = math.exp(-t / 5.5); wx = -.12 + .62 * settle * math.cos(t * .62); wy = -.10 + .22 * settle * math.sin(t * .62 + .6); hk = f // 2 * 2 if t > 4 else f
        chin = float(E3((t - 9) / 3.)) if t >= 9 else 0.; br = math.sin(t * .35) * 2
        ex = dict(EXPR['fierce'], lid=.9) if face == 0 else dict(EXPR['smug'], lid=.18, squint=.1) if face == 1 else dict(EXPR['smug'], gaze=(-.006, .0))
        if face == 1: ex['mouth'] = dict(kind='line', pts=[(-.10, -.652), (-.03, -.664), (.05, -.655), (.15, -.612)], lip=[(-.03, -.75), (.015, -.757), (.065, -.748)])
        kk = (k[0], hk, round(chin, 1))
        def drawN(): return self.nobara(1020, 428 + dip * .55, s, .27 - .03 * chin, .02 + .07 * chin + (0 if face else -.05), -.15 + (0 if face else .07) - .03 * chin, ex, hk, (wx, wy, .3), tyaw=.14, troll=.10, tdy=dip * .5)[0]
        body = self.cached(('E', kk), drawN)
        def arm():
            hc = Cel(); w = mn.lw(s); pal = hpal(L); cam = HD.Cam(hx, hy + dip * .5, 168, 7.5); h = HD.Hand('hammer', 'R', HD.euler(16, -8, hr), (0, 0, 0), HD.FEMALE); ax = np.array((.0, .86, -.345)); dr = HD.unit(np.array((1., .10, .03))); q0 = cam.px(h.world(ax - dr * .62)); q1 = cam.px(h.world(ax + dr * 1.95)); steel_hammer(hc, q0, q1, 112, w[0] * .9, flip=-1, shw=.16)
            wr = cam.px(h.world(np.array([0, -.15, 0]))); g = HD.render([h], cam, light=(.75, .3, .55), clip=(200, 300, 1300, 1080)); HD.ink(hc, g, pal, w[0] * .8, 3, inner=.6); mn.sleeve2(hc, [wr, wr + [-40, 60], wr + [-90, 160], wr + [-150, 320]], [(0, 66), (.5, 92), (1, 118)], L, w, 23); return mn.grade(hc.layer(), L, (300, 400, 900, 1100), .7)
        hl = self.cached(('Eh', k[0]), arm); im = comp.over(im, body)
        sh = cv2.GaussianBlur(shifted(hl[..., 3], -14, 24), (0, 0), 2.0) * body[..., 3]; im = im * (1 - sh[..., None] * (1 - rgb('#8478b4')))
        if t in (0, 1, 2): hl = smear(hl, -90 + 10 * t, -150, 9) * .6 + hl * .45
        im = comp.over(im, hl)
        # sparks and near fire
        im = comp.over(im, comp.bokeh(blueflame([(-200, 560, 1120, 230, 4), (1560, 2100, 1150, 230, 4)], 31, 2, '#0a1766', f=fk, boil=.2), 8, 0)); c = Cel(); r = np.random.default_rng(12)
        for i in range(80):
            x0, y0, v, sz = r.uniform(0, W), r.uniform(0, 1200), r.uniform(8, 30), 1.5 + 6 * r.random() ** 2.5; p = np.array([x0 + 30 * math.sin(t * .25 + i * 1.3) + t * 3, (y0 - v * t) % 1200 - 60]); a = -math.pi / 2 + .4 * math.sin(t * .25 + i); e = np.array([math.cos(a), math.sin(a)]); q = np.array([-e[1], e[0]]); c.fill(spl([p - e * sz, p + q * sz * .5, p + e * sz * 2.2, p - q * sz * .5], True, corners=(2,), n=5), '#c9f1ff' if r.random() < .6 else '#4fa8ff')
        im = im + c.layer()[..., :3] * 1.1
        if t < 3: im = im + (.26, .12, .05)[t] * rgb('#f0e2ff')
        dx, dy = shake(f, 4, 9, 2, 57) if f >= 57 else (0, 0); z = 1.10 - .05 * float(E3(t / 10.)) + .0012 * t
        return comp.cam2d(im, W / 2 + dx + 10 - t * .5, H / 2 + dy + 6, z, .010 * (1 - t / 18.))
