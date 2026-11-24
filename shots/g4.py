"""G4  The last image.  呪 loosens, its mouth runs off the paper, and 祝 is written over the stain.  Seal, a line, light.  108 f."""
import numpy as np, cv2, math, os
from numba import njit
from jj import comp, W, H
from jj.paint import rgb, f32, smooth, fbm, grid, mix
from jj.ink import Cel, spl, resample, noise1
from jj.sumi import (Page, splash, paper, lay, page_state, wet_light, page_save, page_load, tooth, _fibre, SHEET)
from jj.letters import write, KANJI, latin, latin_ink, seal

D = math.radians
X, Y, SZ = 462, 46, 730                                                        # em box of the word
RO = (452, 214, 470, 800)                                                      # region where the ink runs
TIMES = [(39.0, 42.6), (44.2, 50.4), (51.9, 57.0), (58.2, 60.8)]            # the blessing, four calm strokes
RUN0, PEN, STAMP, FADE = 21, (60.5, 75.5), 72, 100                             # ink lets go | the line is written | seal | fade
SEAL = (1408, 742, 108)
VER = 'v5'
B = lambda a, b: (X + a * SZ / 1000., Y + b * SZ / 1000.)


@njit(cache=True, fastmath=True)
def _step(F, C, Wt, S, M, V, src, ramp, n, kd, g, dw, ev, ka, smax, nw):
    h, w = F.shape
    for it in range(n):
        # source + dissolve
        for y in range(h):
            for x in range(w):
                Wt[y, x] += src[y, x] * ramp
                q = kd * min(1., Wt[y, x]) * F[y, x]
                F[y, x] -= q; C[y, x] += q
        # gravity: down the wet channels (upwind, bottom up)
        for y in range(h - 2, -1, -1):
            for x in range(1, w - 1):
                wt = Wt[y, x]
                if wt < .02: continue
                v = g * M[y, x] * max(0., wt - .10) ** 1.2
                if v > .62: v = .62
                fx = V[y, x]
                dn = wt * v; dc = C[y, x] * v
                xx = x + (1 if fx > 0 else -1); af = min(.8, abs(fx))
                Wt[y, x] -= dn; C[y, x] -= dc
                Wt[y + 1, x] += dn * (1 - af); C[y + 1, x] += dc * (1 - af)
                Wt[y + 1, xx] += dn * af; C[y + 1, xx] += dc * af
        # wicking along fibres, evaporation, adsorption
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                wt = Wt[y, x]
                if wt > .004:
                    d = dw * (.25 + 1.3 * M[y, x]) * .25
                    for k in range(4):
                        yy = y + (1 if k == 0 else (-1 if k == 1 else 0)); xx = x + (1 if k == 2 else (-1 if k == 3 else 0))
                        df = Wt[y, x] - Wt[yy, xx]
                        if df > 0:
                            m = d * df
                            c = C[y, x] * (m / (Wt[y, x] + 1e-4))
                            if c > C[y, x] * .24: c = C[y, x] * .24
                            Wt[y, x] -= m; Wt[yy, xx] += m; C[y, x] -= c; C[yy, xx] += c
                Wt[y, x] *= 1 - ev
                a = ka * C[y, x] * (.35 + 1.6 * max(0., .5 - Wt[y, x]))
                if a > C[y, x]: a = C[y, x]
                C[y, x] -= a; r = max(0., 1 - S[y, x] / smax); S[y, x] += a * .42 * r


def dissolve(F0, seed=3, f0=27, f1=76):
    """frames f0..f1 of the running ink: (ink left, ink in motion, stain, water)"""
    h, w = F0.shape; r = np.random.default_rng(seed); y, x = grid(h, w)
    wander = np.cumsum(cv2.GaussianBlur(r.standard_normal((h, 1)).astype(f32), (0, 0), 14), 0); wander = (wander - wander.mean()) / (wander.std() + 1e-6) * 5.
    cs = np.arange(4, w + 76, 21) + r.normal(0, 7, len(np.arange(4, w + 76, 21))); col = np.zeros(w + 80, f32)
    for c0 in cs: col += np.exp(-((np.arange(w + 80) - c0) / r.uniform(2.4, 5.4)) ** 2) * r.gamma(2.2, .42)
    ch = np.stack([np.interp(np.arange(w) + 40 + wander[j, 0] + 4 * math.sin(j / 47.), np.arange(w + 80), col) for j in range(h)]).astype(f32)
    M = (.10 + 1.15 * ch) * (1 + .30 * fbm(h, w, (14, 6), (1, .6), seed)) * (.75 + .5 * _fibre(h, w)); M = np.clip(M, .04, 2.4).astype(f32)
    V = np.clip(np.gradient(wander[:, 0])[:, None] * -1.0 + .16 * fbm(h, w, (36, 12), (1, .5), seed + 2), -.7, .7).astype(f32)
    ms = cv2.GaussianBlur(F0, (0, 0), 3.); src = ms * (.8 + .45 * np.clip(fbm(h, w, (34, 13), (1, .6), seed + 4), -1, 1.6)); src = (np.clip(src, 0, 2) * .052).astype(f32)
    F, C, Wt, S = F0.astype(f32).copy(), np.zeros((h, w), f32), np.zeros((h, w), f32), np.zeros((h, w), f32); out = []
    for f in range(f0, f1 + 1):
        u = f - f0; ramp = float(np.interp(u, [0, 4, 12, 18, 24], [.0, .7, 1., .5, 0.]))
        kd = float(np.interp(u, [0, 6, 16, 30], [.008, .030, .060, .09])); ev = float(np.interp(u, [0, 12, 20, 27, 34], [.003, .006, .022, .08, .18])); ka = float(np.interp(u, [0, 12, 20, 28, 38], [.003, .008, .030, .10, .24]))
        _step(F, C, Wt, S, M, V, src, ramp * .75, 24, kd / 2.6, 1.25, .30, ev / 2.6, ka / 2.6, .21, 0)
        out.append((F.copy(), cv2.GaussianBlur(C, (0, 0), sigmaX=.9, sigmaY=2.4), cv2.GaussianBlur(S, (0, 0), 1.2), np.clip(cv2.GaussianBlur(Wt, (0, 0), 1.), 0, 1)))
    return out


def petal_mesh(cup=.5, n=13, m=19):
    """cherry petal: u across -1..1, v base 0 -> tip 1 (notched)"""
    v = np.linspace(0, 1, m); hw = .62 * np.sin(np.clip(v * 1.08, 0, 1) ** .82 * math.pi * .86 + .10) * (1 - .10 * v); P = []
    for i, vv in enumerate(v):
        uu = np.linspace(-1, 1, n); x = uu * hw[i]; y = vv * (1 - .16 * np.exp(-(uu / .30) ** 2) * (vv > .55) * (vv - .55) / .45) ; z = cup * (x * x * 1.6 - .20 * np.sin(vv * math.pi) + .18 * vv * vv); P.append(np.c_[x, y - .5, z])
    return np.array(P)


class Petal:
    def __init__(s, t0, p0, p1, size, seed, land=None, z0=260., dur=60., spin=1.):
        s.t0, s.p0, s.p1, s.size, s.land, s.z0, s.dur, s.spin = t0, np.array(p0, float), np.array(p1, float), size, land, z0, dur, spin
        r = np.random.default_rng(seed); s.a = r.uniform(0, 6.28, 6); s.w = r.uniform(.07, .14, 3); s.m = petal_mesh(r.uniform(.35, .7)); s.seed = seed; s.tint = r.uniform(-.5, 1)

    def at(s, t):
        u = (t - s.t0) / s.dur
        if u < -.05 or (s.land is None and u > 1.05): return None
        e = min(1., u); k = e if s.land is None else 1 - (1 - max(0, e)) ** 1.5; sway = np.array([math.sin(t * s.w[0] + s.a[0]) * 46, math.sin(t * s.w[1] + s.a[1]) * 16]) * (1 - (smooth(.55, 1., e) if s.land is not None else 0))
        p = s.p0 * (1 - k) + s.p1 * k + sway; z = s.z0 * (1 - .35 * max(0, e) + .10 * math.sin(t * s.w[2] + s.a[2])) if s.land is None else s.z0 * (1 - k) ** 1.1
        fl = 1. if s.land is None else 1 - float(smooth(.62, 1.02, e)); ang = np.array([s.a[3] + t * .13 * s.spin, s.a[4] + t * .16 * s.spin, s.a[5] + t * .035]) * np.array([1., 1., 1.])
        if s.land is not None: tg = np.array([.30, -.16, s.land]); A = np.array([math.sin(ang[0]) * 1.25, math.sin(ang[1]) * 1.25, ang[2]]); ang = A * fl + tg * (1 - fl)
        return p, max(0., z), ang

    def draw(s, cel, t, k=1.):
        q = s.at(t)
        if q is None: return None
        p, z, a = q; ca, sa = np.cos(a), np.sin(a); Rx = np.array([[1, 0, 0], [0, ca[0], -sa[0]], [0, sa[0], ca[0]]]); Ry = np.array([[ca[1], 0, sa[1]], [0, 1, 0], [-sa[1], 0, ca[1]]]); Rz = np.array([[ca[2], -sa[2], 0], [sa[2], ca[2], 0], [0, 0, 1]])
        sc = s.size * 1200. / (1200. - z); M = s.m @ (Rz @ Ry @ Rx).T * sc; XY = M[..., :2] + p; n_, m_ = M.shape[0], M.shape[1]
        quads = []
        for i in range(n_ - 1):
            for j in range(m_ - 1):
                A, B_, C, E = M[i, j], M[i, j + 1], M[i + 1, j + 1], M[i + 1, j]; nn = np.cross(C - A, E - B_); nn /= np.linalg.norm(nn) + 1e-9; v = i / (n_ - 1.)
                fr = nn[2] < 0; ln = abs(float(nn @ np.array([-.45, -.55, .70]))); base = mix(rgb('#fdeeec'), rgb('#f3bcc2'), float(np.clip(1.0 - v * 1.6, 0, 1)) * .85 + .06 * s.tint) if fr else mix(rgb('#f8dfdf'), rgb('#eab2bb'), float(np.clip(1. - v * 1.4, 0, 1)))
                h_ = float(smooth(.30, .52, ln)); col = base * (.88 + .12 * h_) * (np.array([1., .955, .97]) * (1 - h_) + np.array([1, .99, .985]) * h_); quads.append((A[2] + C[2], i, j, col))
        for _, i, j, col in sorted(quads, key=lambda t_: t_[0]):
            P = np.array([XY[i, j], XY[i, j + 1], XY[i + 1, j + 1], XY[i + 1, j]]); c = P.mean(0); cel.fill(c + (P - c) * 1.07, np.clip(col, 0, 1))
        rim = np.vstack([XY[:, 0], XY[-1, 1:], XY[::-1, -1][1:]]); cel.line(rim, 1.5 * k, '#b87f86', taper=(.12, .12), press=.5, seed=s.seed, alpha=.8)
        mid = XY[:, m_ // 2]; cel.line(mid[1:11], 1.0 * k, '#d08e9c', taper=(.1, .8), seed=s.seed + 1, alpha=.6)
        return rim, z


class Shot:
    id = 'g4'; n = 108
    finish = dict(bloom=.10, vig=.10, soft=.05, grain=.009, lift=.0, halation='#fff0dc')
    EVENTS = [(0, 'washi_room', dict(note='quiet morning: paper, far birds, the held chord')), (4, 'birds', dict(size=.2, note='very far')), (RUN0, 'ink_run', dict(dur=26, size=.5, note='the left half loosens and runs down')), (27, 'ink_drip', dict(size=.3, pan=-.4)), (34, 'ink_drip', dict(size=.22, pan=-.3)),
              (39, 'brush_write', dict(stroke=1, dur=4, pan=-.35, note='dot')), (44, 'brush_write', dict(stroke=2, dur=6, pan=-.3, note='across, then the long fall to the left')), (52, 'brush_write', dict(stroke=3, dur=5, pan=-.3, note='down')), (58, 'brush_write', dict(stroke=4, dur=3, pan=-.2, note='last dot')),
              (60, 'pencil_letters', dict(dur=16, pan=0., note='two small lines, pen, done at 76')), (STAMP, 'seal_stamp', dict(size=.9, pan=.4, note='pressed with a squash, paper dips; lifted at 76')), (90, 'petal_land', dict(size=.3, pan=.5, note='one petal settles by the seal')), (FADE, 'fade_white', dict(dur=8, note='finished picture held 76-100, then slow fade to warm white, full at 107'))]

    def setup(self):
        os.makedirs('cache', exist_ok=True); pa = f'cache/g4_page_{VER}.npz'; pd = f'cache/g4_run_{VER}.npy'
        if os.path.exists(pa):
            pg = page_load(pa)
            for m, (t0, t1) in zip(pg.marks[8:12], TIMES): m.t0, m.t1 = t0, t1
        else:
            pg = Page(seed=4); write(pg, '呪', (X, Y, SZ), times=[(-200, -199)] * 8); write(pg, '祝', (X, Y, SZ), strokes=KANJI['祝'][:4], times=TIMES, dry=.34, deplete=.75, load=.85)
            for a, b, R, sd, ang, sp, n in ((70, 350, 130, 11, D(200), 1.5, 60), (505, 160, 130, 12, D(-120), 1.3, 60), (880, 160, 140, 13, D(-30), 1.2, 70), (1110, 850, 240, 15, D(5), 1.6, 150)): m = splash(*B(a, b), R, seed=sd, ang=ang, spread=sp, n=n, big=9, fingers=6, lobes=4, core=0); m.t0, m.t1 = -200, -199; pg.add(m)
            page_save(pg, pa)
        self.pg = pg; st = lambda idx, t=0.: page_state(pg, t, 9e9, only=set(idx))
        self.old = st((3, 4, 5, 6, 7)); self.mouth = st((0, 1, 2)); self.spl = st(range(12, len(pg.marks)))[0]
        x0, y0, rw, rh = RO
        if not os.path.exists(pd): run = dissolve(self.mouth[0][y0:y0 + rh, x0:x0 + rw]); q = pd + f'.{os.getpid()}.npy'; np.save(q, np.array(run).astype(np.float16)); os.replace(q, pd)
        self.run = np.load(pd, mmap_mode='r')
        # morning: warm washi, a soft window, leaves
        self.P = paper(tone='#fbf6ea', amt=1.25, seed=23); x, y = grid(); d = (x * .55 + y * .84)
        self.warm = mix(np.array([1.06, 1.04, 1.0], f32), np.array([.975, .95, .925], f32), smooth(.0, 1.3, np.hypot((x - 380) / 1750, (y - 60) / 1250))[..., None])
        self.fresh_wear = np.clip(1 - .10 * smooth(.72, 1.0, tooth(H, W, 5)) - .07 * smooth(.2, 1.6, fbm(H, W, (90, 30), (1, .6), 31)), .25, 1)
        hal = cv2.GaussianBlur(self.old[0] + self.mouth[0], (0, 0), 13); self.halo_old = cv2.GaussianBlur(self.old[0], (0, 0), 13); self.halo_mouth = cv2.GaussianBlur(self.mouth[0], (0, 0), 13)
        self.leaves = self._leaves()
        L1 = latin('a fan film', 905, 892, 46, seed=3, align='center'); L2 = latin('characters by Gege Akutami', 905, 950, 46, seed=4, align='center'); self.text = [L1, L2]
        self.seal = seal(SEAL[2], 2)
        self.petals = [Petal(-14, (260, -60), (1120, 1180), 44, 1, z0=240, dur=86, spin=.9), Petal(8, (1180, -70), (1790, 1160), 36, 2, z0=300, dur=74, spin=1.1), Petal(34, (760, -60), (1552, 806), 40, 5, land=2.2, z0=330, dur=56, spin=1.0), Petal(52, (-60, 300), (780, 1150), 48, 7, z0=420, dur=62, spin=.8), Petal(20, (1680, -60), (2010, 700), 34, 9, z0=200, dur=58, spin=1.2), Petal(62, (420, -80), (1010, 700), 38, 11, z0=360, dur=70, spin=1.0)]

    def _leaves(self):
        """shadow of a branch outside the window, larger than the frame: two soft layers (near / far)"""
        out = []
        for sd, n, s0, bl in ((5, 46, 1.25, 8.5), (6, 34, 1.9, 22)):
            r = np.random.default_rng(sd); c = Cel((W + 400) // 2, (H + 300) // 2, ss=1)
            br = spl(np.array([(1250, -30), (1010, 120), (800, 170), (560, 300), (330, 350)], float) + (sd - 5) * np.array([-60, 120]), n=14); c.line(br, 13 * s0, '#000000', taper=(.0, .8)); tw = []
            for k in range(7): i = int(r.uniform(.1, .95) * (len(br) - 1)); q = br[i]; e = q + np.array([r.uniform(-190, -30), r.uniform(-120, 170)]); m = (q + e) / 2 + r.normal(0, 16, 2); tw.append(spl([q, m, e], n=10)); c.line(tw[-1], 6 * s0, '#000000', taper=(.0, .8))
            for k in range(n):
                t = tw[k % len(tw)]; q = t[int(r.uniform(.15, 1) * (len(t) - 1))]; a = r.uniform(0, 6.28); L = r.uniform(34, 62) * s0; d = np.array([math.cos(a), math.sin(a)]); nn = np.array([-d[1], d[0]]); wd = L * r.uniform(.22, .32); c.fill(spl(np.array([q, q + d * L * .4 + nn * wd, q + d * L, q + d * L * .4 - nn * wd]), True, corners=(0, 2), n=6), '#000000')
            m = cv2.resize(c.rgba()[..., 3], (W + 400, H + 300), interpolation=cv2.INTER_LINEAR); out.append(cv2.GaussianBlur(m, (0, 0), bl))
        return out

    def light(self, f):
        s = [(math.sin(f * .050) * 13 + math.sin(f * .017 + 1) * 9, math.sin(f * .041 + 2) * 8), (math.sin(f * .038 + .6) * 22, math.sin(f * .030) * 12)]; L = self.warm.copy()
        for k, (dx, dy) in enumerate(s):
            m = cv2.warpAffine(self.leaves[k], np.float32([[1, 0, -200 + dx], [0, 1, -150 + dy]]), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            L = L * (1 - (.105, .075)[k] * m[..., None] * np.array([1.12, 1., .74], f32))
        return L

    def frame(self, f):
        t = f + .5; base = self.P * self.light(f) * (.985 + .065 * float(smooth(10, 100, f)))
        aged = rgb('#1d1815'); brown = rgb('#b79c78')
        # the old word: dried, worn into the tooth, a breath of oil round it; the new ink wakes it a little
        wake = float(smooth(46, 64, f)); wear = self.fresh_wear + (1 - self.fresh_wear) * .65 * wake
        base = base * (1 - (.10 * np.clip(self.halo_old, 0, 1))[..., None] * (1 - brown)); mo = 1 - .75 * float(smooth(RUN0 + 3, RUN0 + 25, f)); base = base * (1 - (.10 * mo * np.clip(self.halo_mouth, 0, 1))[..., None] * (1 - brown))
        a = (self.spl * .17)[..., None]; base = base * (1 - a) + rgb('#6b5d50') * a
        a = (self.old[0] * wear * (.93 + .05 * wake))[..., None]; g = smooth(.06, .5, self.old[2])[..., None]; base = base * (1 - a) + (mix(aged, rgb('#120f0f'), wake) * (1.25 - .35 * g)) * a
        # its mouth
        x0, y0, rw, rh = RO; q = (slice(y0, y0 + rh), slice(x0, x0 + rw)); wet = np.zeros((H, W), f32)
        if f < RUN0: a = (self.mouth[0] * self.fresh_wear * .93)[..., None]; base = base * (1 - a) + aged * (1.2 - .3 * smooth(.06, .5, self.mouth[2])[..., None]) * a
        else:
            F, C, S, Wt = [np.asarray(self.run[min(f - RUN0, len(self.run) - 1), k], f32) for k in range(4)]; b = base[q]; e = float(smooth(RUN0, RUN0 + 11, f)); wr = self.fresh_wear[q] + (1 - self.fresh_wear[q]) * e
            b = b * (1 - (np.clip(Wt, 0, 1) * .055 * smooth(.02, .3, Wt))[..., None] * np.array([.7, .9, 1.15], f32))          # wet paper is a shade darker
            a = np.clip(S, 0, 1)[..., None]; b = b * (1 - a) + rgb('#5b4d48') * a
            a = (self.mouth[0][q] * self.fresh_wear[q] * .10 * e * (1 - .25 * float(smooth(60, 96, f))))[..., None]; b = b * (1 - a) + rgb('#4a4040') * a
            a = (1 - np.exp(-1.25 * np.clip(C, 0, 3)))[..., None] * .86; b = b * (1 - a) + mix(rgb('#544b4b'), rgb('#1c1819'), smooth(.25, .85, a)) * a
            a = np.clip(F * wr * .93, 0, 1)[..., None]; b = b * (1 - a) + mix(aged, rgb('#2c2625'), e) * a
            base[q] = b; wet[q] = np.clip(1 - np.exp(-1.2 * C), 0, 1) * smooth(.15, .5, Wt)
            gl = wet_light(np.where(wet > .25, 1., 0).astype(f32) * smooth(.2, .5, wet), wet, cv2.GaussianBlur(wet, (0, 0), 2) * .5, .55, phase=f * .02); base = base + gl
        # the unseen brush: a soft shadow passes over the sheet
        for k, (ta, tb) in enumerate(TIMES):
            u = (t - ta) / (tb - ta)
            if -.55 < u < 1.5:
                inf = self.pg.marks[8 + k].info; i = int(np.clip(np.searchsorted(inf['t'], np.clip(u, 0, 1)), 0, len(inf['x']) - 1)); px, py = float(inf['x'][i]), float(inf['y'][i]); lift = max(0., -u * 1.2, (u - 1.) * 1.2) * 190
                s = np.zeros((H // 4, W // 4), f32); p0 = np.array([px + 24 + lift * .8, py + 28 + lift]); p1 = p0 + np.array([1400, 1300]); n = np.array([-.68, .73]); pl = np.vstack([p0 + n * 6, p0 + np.array([40, 37]) + n * 26, p0 + np.array([110, 102]) + n * 42, p1 + n * 120, p1 - n * 120, p0 + np.array([110, 102]) - n * 42, p0 + np.array([40, 37]) - n * 26, p0 - n * 6]) / 4
                cv2.fillPoly(s, [np.round(pl * 16).astype(np.int32)], 1., cv2.LINE_AA, 4); s = cv2.GaussianBlur(s, (0, 0), 5 + lift * .05); far = cv2.GaussianBlur(s, (0, 0), 24); gx, gy = grid(H // 4, W // 4); uu = np.clip(((gx * 4 - p0[0]) * .733 + (gy * 4 - p0[1]) * .68) / 760., 0, 1); s = s * (1 - uu) + far * uu * .7
                s = cv2.resize(s, (W, H), interpolation=cv2.INTER_LINEAR); base = base * (1 - .12 * max(0., 1 - lift / 170.) * s[..., None] * np.array([1.08, 1., .86], f32))
        # the new half: fresh, wet, black
        if f >= int(TIMES[0][0]):
            N, Fr, Po = page_state(self.pg, t, tau=34., only={8, 9, 10, 11}); base = lay(base, N, None, Po, col='#0c0a0c'); base = base + wet_light(N, Fr, Po, .9, phase=1.1 + f * .012)
        # the line
        if f >= int(PEN[0]):
            a = latin_ink(self.text, float(np.clip((t - PEN[0]) / (PEN[1] - PEN[0]), 0, 1)))[..., None] * .90; base = base * (1 - a) + rgb('#2a2322') * a
        # the seal
        sx, sy, ss = SEAL; x, y = grid(); dep = float(np.interp(f, [69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 81], [0, .12, 1.25, 1.12, .95, .80, .40, -.16, -.20, -.06, 0]))
        if f >= 72:
            k = 1.0; s = np.zeros((H, W), f32); sqx, sqy = {72: (1.13, .86), 73: (1.07, .93), 74: (.975, 1.03), 75: (.99, 1.012)}.get(f, (1., 1.)); M = cv2.getRotationMatrix2D((ss / 2, ss / 2), 2.2, 1.); M = np.diag([sqx, sqy]) @ M + np.array([[0, 0, ss / 2 * (1 - sqx)], [0, 0, ss / 2 * (1 - sqy)]]); M[:, 2] += [sx - ss / 2, sy - ss / 2]; s = cv2.warpAffine(self.seal, M, (W, H), flags=cv2.INTER_LINEAR)
            a = (s * .95)[..., None]; base = base * (1 - a) + rgb('#b4161c') * (.60 + .42 * np.clip(base, 0, 1)) * a
            if f < 86: w_ = cv2.GaussianBlur(s, (0, 0), 1.2); base = base + ((np.clip(w_ - cv2.warpAffine(w_, np.float32([[1, 0, 1.2], [0, 1, 1.6]]), (W, H)), 0, 1)) * .30 * (1 - smooth(76, 86, f)))[..., None] * rgb('#ffd9cc')
        if 66 < f < 82:                                                        # shadow of the stone nobody holds, closing and opening
            hgt = float(np.interp(f, [67, 72, 75, 81], [200, 0, 0, 230])); q_ = ss / 2 + 7; ox, oy = hgt * .42 + 3, hgt * .58 + 4; m = 15; u = np.linspace(0, 2 * math.pi, 48, endpoint=False); pl = np.c_[sx + ox + np.sign(np.cos(u)) * np.abs(np.cos(u)) ** .45 * q_, sy + oy + np.sign(np.sin(u)) * np.abs(np.sin(u)) ** .45 * q_]
            s = np.zeros((H // 2, W // 2), f32); cv2.fillPoly(s, [np.round(pl * 8).astype(np.int32)], 1., cv2.LINE_AA, 4); s = cv2.resize(cv2.GaussianBlur(s, (0, 0), 4.5 + hgt * .11), (W, H), interpolation=cv2.INTER_LINEAR)
            base = base * (1 - .22 * float(np.clip(1 - hgt / 240., 0, 1)) ** .8 * s[..., None] * np.array([1.08, 1., .86], f32))
        if abs(dep) > .005:
            dx, dy = x - sx, y - sy; d = np.sqrt(dx * dx + dy * dy) + 1; g = np.exp(-(d / 165.) ** 2); sl = dep * (d / 165.) * g * 1.4; base = base * (1 - .07 * dep * g - .135 * sl * (-(dx * -.55 + dy * -.83) / d))[..., None]
            am = 10.5 * dep * (d / 120.) * np.exp(-(d / 150.) ** 2); base = cv2.remap(base.astype(f32), x + dx / d * am, y + dy / d * am, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
        # petals and their shadows
        sh = np.zeros((H // 2, W // 2), f32); c = Cel(W, H); info = []
        for p in self.petals:
            o = p.draw(c, t)
            if o is not None: info.append(o)
        for rim, z in info:
            l = np.zeros((H // 2, W // 2), f32); cv2.fillPoly(l, [np.round((rim + np.array([.32, .46]) * z + np.array([2.5, 3.5])) * 8).astype(np.int32)], 1., cv2.LINE_AA, 4); sh = np.maximum(sh, cv2.GaussianBlur(l, (0, 0), .9 + z * .022) * (.30 * math.exp(-z / 420.)))
        if info:
            sh = cv2.resize(sh, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]; base = base * (1 - sh * np.array([1.05, 1.0, .88], f32)); L = c.layer(); base = base * (1 - L[..., 3:4]) + L[..., :3]
        # an almost imperceptible push
        z = 1.004 + .020 * comp.ease(f / 107.); im = comp.cam2d(base.astype(f32), W / 2 - 10 * (f / 107.), H / 2 - 8 * (f / 107.) + {72: -2.2, 73: 1.0}.get(f, 0.), z, D(-.10) + D(.16) * f / 107., interp=cv2.INTER_CUBIC)
        if f >= FADE:
            k = float(np.clip((f - FADE + 1.) / 8., 0, 1)) ** 1.35; im = im + np.clip(im - .5, 0, 1) * k * .30; im = im * (1 - k) + rgb('#fff6e8') * 1.03 * k
        return np.maximum(im, 0).astype(f32)
