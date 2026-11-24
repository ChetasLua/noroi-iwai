"""Painted hillside seen from above, alive.

The ground is a gouache painting whose brush marks remember WHERE they sampled their colour: every frame a small
colour guide (value masses + drifting cloud shadow + travelling gusts + contact shadow) is re-read through those marks,
so the painting changes light without a single stroke moving.  On top: tens of thousands of individually directed
tapered blades that bend with the gust, flash their pale undersides, splay round whoever lies in them and overlap
their edges.  Everything is a pure function of the frame number.

    G = Hill(season='summer', blobs=[...], crest=..., seed=3); G.setup()
    img = G.frame(f, layer=rgba, rest=alpha, cam=(cx, cy, zoom, rot))
"""
import numpy as np, cv2, math, numba
from numba import njit, prange
from . import W, H
from .paint import rgb, f32, smooth, mix, ramp, lum, fbm, Canvas, arcs, grid
from .comp import over, blur

numba.set_num_threads(min(4, numba.get_num_threads()))
PW, PH = 2240, 1260                     # painting size; frame sits centred in it (margin 160 x 90)
MX, MY = (PW - W) // 2, (PH - H) // 2
Q = 2                                   # guide is painting / Q
GW, GH = PW // Q, PH // Q


# ------------------------------------------------------------------ periodic noise
def tile(n, feat, seed, aniso=1.0):
    """periodic unit-variance noise n x n, dominant feature size `feat` px (aniso stretches along x)"""
    r = np.random.default_rng(seed)
    g = np.fft.rfft2(r.standard_normal((n, n)))
    ky = np.fft.fftfreq(n)[:, None]; kx = np.fft.rfftfreq(n)[None]
    k = np.sqrt((kx * aniso) ** 2 + ky ** 2)
    a = np.exp(-(k * feat) ** 2 * 1.6)
    o = np.fft.irfft2(g * a, s=(n, n)).astype(f32)
    return (o - o.mean()) / (o.std() + 1e-6)


def samp(t, u, v):
    n = t.shape[0]
    if u.shape[-1] > 30000:
        a, b = np.mod(u, n), np.mod(v, n); i, j = a.astype(int), b.astype(int); p, q = a - i, b - j; i %= n; j %= n; i1, j1 = (i + 1) % n, (j + 1) % n
        return ((t[j, i] * (1 - p) + t[j, i1] * p) * (1 - q) + (t[j1, i] * (1 - p) + t[j1, i1] * p) * q).astype(f32)
    return cv2.remap(t, np.mod(u, n).astype(f32), np.mod(v, n).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


# ------------------------------------------------------------------ blade rasteriser
@njit(parallel=True, cache=True, fastmath=True)
def _blades(img, alpha, hasA, X, Y, MXx, MXy, TX, TY, WD, C0, C1, OP, KIND, nb):
    """quadratic blades root(X,Y) -> mid -> tip.  KIND 0 grass (root melts, tip tapers) 1 petal (round end) 2 dab (oval).
    Sequential painter inside nb horizontal bands."""
    h, w = img.shape[0], img.shape[1]; n = X.shape[0]
    for b in prange(nb):
        y0b = h * b // nb; y1b = h * (b + 1) // nb
        for i in range(n):
            m = WD[i] + 1.5
            lo = min(Y[i], min(MXy[i], TY[i])) - m; hi = max(Y[i], max(MXy[i], TY[i])) + m
            if hi < y0b or lo >= y1b: continue
            if min(X[i], min(MXx[i], TX[i])) - m >= w or max(X[i], max(MXx[i], TX[i])) + m < 0: continue
            k = KIND[i]; S = 5 if k == 0 else 4
            # control point of the quadratic through mid at t=.5
            cx = 2 * MXx[i] - .5 * (X[i] + TX[i]); cy = 2 * MXy[i] - .5 * (Y[i] + TY[i])
            px = X[i]; py = Y[i]
            for s in range(S):
                t1 = (s + 1) / S; t0 = s / S
                u = 1 - t1
                qx = u * u * X[i] + 2 * u * t1 * cx + t1 * t1 * TX[i]; qy = u * u * Y[i] + 2 * u * t1 * cy + t1 * t1 * TY[i]
                dx = qx - px; dy = qy - py; L2 = dx * dx + dy * dy + 1e-9
                ix0 = max(0, int(min(px, qx) - m)); ix1 = min(w - 1, int(max(px, qx) + m) + 1)
                iy0 = max(y0b, int(min(py, qy) - m)); iy1 = min(y1b - 1, int(max(py, qy) + m) + 1)
                for y in range(iy0, iy1 + 1):
                    for x in range(ix0, ix1 + 1):
                        t = ((x - px) * dx + (y - py) * dy) / L2
                        if t < 0: t = 0.0
                        elif t > 1:
                            if s < S - 1: continue
                            t = 1.0
                        ex = px + dx * t - x; ey = py + dy * t - y
                        d = math.sqrt(ex * ex + ey * ey)
                        tt = t0 + (t1 - t0) * t
                        if k == 0:
                            wl = WD[i] * (1 - tt * tt) * (.62 + .38 * min(1.0, tt * 3.5)) + .18
                            o = min(1.0, tt * 3.2)
                        elif k == 1:
                            wl = WD[i] * math.sqrt(max(0.0, 1 - (1.9 * tt - 1.0) ** 2 * .98)) * (.55 + .6 * tt) + .12
                            o = 1.0
                        else:
                            wl = WD[i] * math.sqrt(max(0.0, 1 - (2 * tt - 1.0) ** 2 * .96)) + .1
                            o = 1.0
                        a = wl - d + .5
                        if a <= 0: continue
                        if a > 1: a = 1.0
                        a *= o * OP[i]
                        for c in range(3):
                            v = C0[i, c] + (C1[i, c] - C0[i, c]) * tt
                            img[y, x, c] += (v - img[y, x, c]) * a
                        if hasA: alpha[y, x] += (1 - alpha[y, x]) * a
                px = qx; py = qy


@njit(parallel=True, cache=True, fastmath=True)
def _guide(out, val, g, tr, cl, hue, hue2, far, air, lut, A, C, ac, warm, cool, shade, flash, gk):
    h, w = val.shape; n = lut.shape[0] - 1
    for y in prange(h):
        for x in range(w):
            gg = g[y, x]; cc = cl[y, x]; fr = far[y, x]
            v = val[y, x] + .22 * gk * gg - .15 * gk * tr[y, x] - .17 * cc
            v = min(1.0, max(0.0, v)) * n; i = min(n - 1, int(v)); f = v - i
            r = lut[i, 0] + (lut[i + 1, 0] - lut[i, 0]) * f; e = lut[i, 1] + (lut[i + 1, 1] - lut[i, 1]) * f; b = lut[i, 2] + (lut[i + 1, 2] - lut[i, 2]) * f
            hu = hue[y, x]; L = r * .299 + e * .587 + b * .114
            k = .096 * min(1.6, max(0.0, hu)); q = L * 1.25
            r = r * (1 - k) + warm[0] * q * k; e = e * (1 - k) + warm[1] * q * k; b = b * (1 - k) + warm[2] * q * k
            k = .105 * min(1.6, max(0.0, -hu)); L = r * .299 + e * .587 + b * .114; q = L * .6 + .35
            r = r * (1 - k) + cool[0] * q * k; e = e * (1 - k) + cool[1] * q * k; b = b * (1 - k) + cool[2] * q * k
            m = 1 + .02 * hue2[y, x]; r *= m; e *= m; b *= m
            k = .40 * cc * (1 - .5 * fr)
            r = r * (1 - k) + r * shade[0] * 1.75 * k; e = e * (1 - k) + e * shade[1] * 1.75 * k; b = b * (1 - k) + b * shade[2] * 1.75 * k
            k = min(.9, .56 * (.5 + .5 * gk)) * gg * (1 - .55 * fr)
            r = r * (1 - k) + (flash[0] * .55 + r * .45) * k; e = e * (1 - k) + (flash[1] * .55 + e * .45) * k; b = b * (1 - k) + (flash[2] * .55 + b * .45) * k
            a = A[y, x]; t = .72 + .28 * (1 - cc)
            r = r * (1 - a) + C[y, x, 0] * t; e = e * (1 - a) + C[y, x, 1] * t; b = b * (1 - a) + C[y, x, 2] * t
            a = air[y, x]; k = .22 * cc
            r = r * (1 - a) + (ac[y, x, 0] * (1 - k) + shade[0] * k) * a; e = e * (1 - a) + (ac[y, x, 1] * (1 - k) + shade[1] * k) * a; b = b * (1 - a) + (ac[y, x, 2] * (1 - k) + shade[2] * k) * a
            out[y, x, 0] = r; out[y, x, 1] = e; out[y, x, 2] = b


_NOA = np.zeros((1, 1), f32)


def draw_blades(img, g, alpha=None, nb=8):
    """g = dict of arrays x y mx my tx ty w c0 c1 op kind (already sorted in painting order)"""
    if len(g['x']) == 0: return img
    _blades(img, _NOA if alpha is None else alpha, alpha is not None, *(g[k].astype(f32) for k in ('x', 'y', 'mx', 'my', 'tx', 'ty', 'w')),
            g['c0'].astype(f32), g['c1'].astype(f32), g['op'].astype(f32), g['kind'].astype(np.int32), nb)
    return img


# ------------------------------------------------------------------ seasons
PAL = dict(
    summer=dict(ramp=[(0, '#1f6a5c'), (.2, '#3b8e5f'), (.4, '#62ad5b'), (.58, '#86c561'), (.74, '#a3d66c'), (.88, '#c4e57f'), (1, '#e2f3a4')],
                shade='#2c7f74', air='#a9dccb', air2='#8fcdc4', flash='#e4f3bb', warm='#d8dd62', cool='#5fc09a',
                contact='#1d5a56', flowers=[('daisy', .5), ('butter', .3), ('clover', .2)]),
    spring=dict(ramp=[(0, '#2c7a5a'), (.2, '#4c9c5c'), (.4, '#78ba5c'), (.58, '#9dd063'), (.74, '#b9de70'), (.88, '#d6ea8a'), (1, '#f1f6b6')],
                shade='#3f8f7c', air='#cfe6d2', air2='#b5dccb', flash='#f3f7cf', warm='#ecd968', cool='#79caa0',
                contact='#24604f', flowers=[('daisy', .26), ('butter', .24), ('pink', .26), ('violet', .12), ('clover', .12)]))
PAL['gold'] = dict(ramp=[(0, '#2d6a56'), (.2, '#4f8e55'), (.4, '#80ad55'), (.58, '#abc65a'), (.74, '#cbd769'), (.88, '#e8e489'), (1, '#fbf2b8')],
                   shade='#6a8a80', air='#ffe7b0', air2='#e2e2bd', air3='#c6cbe0', flash='#fff4cb', warm='#f5cf5c', cool='#8cc49a',
                   contact='#3f6450', flowers=[('daisy', .26), ('butter', .2), ('pink', .30), ('violet', .12), ('clover', .12)])


class Hill:
    """One composed hillside.
    blobs  = [(cx, cy, rx, ry, rot_deg, dv, soft)] authored value shapes in painting px (dv>0 light, <0 dark; soft 0..1)
    crest  = None or [(x, y), ...] the ridge line; above it lies a far meadow in blue air
    wind   = screen direction the gusts travel in degrees (0 = right, 90 = down)
    quiet  = [(cx, cy, r)] places kept free of tufts and flowers (where somebody lies)
    """

    def __init__(self, season='summer', seed=3, blobs=(), crest=None, wind=18., wave=1.0, speed=1.0, quiet=(), flowers=1.0,
                 clouds=.55, drifts=(), density=1.0, t0=0., cloud_seed=0, tufts=1.0, contrast=1.0, trees=(), far_blobs=(), period=27., lam=700., crest2=None, pond=None, carpets=(), gust_ref=None, weak=1.0, strips=(), haze=1.0, sunny=(), far_air=(.40, .30), gk=1.0, level=0.):
        self.__dict__.update(locals()); del self.__dict__['self']
        self.p = PAL[season]

    # -------------------------------------------------- static geometry
    def setup(self):
        r = np.random.default_rng(self.seed)
        x, y = grid(GH, GW); x = x * Q + Q / 2; y = y * Q + Q / 2
        if self.crest is not None:
            c = np.asarray(self.crest, float); xs = np.arange(GW) * Q
            from .ink import spl
            cc = spl(c, n=24); o = np.argsort(cc[:, 0]); self.cy_ = np.interp(np.arange(PW), cc[o, 0], cc[o, 1]).astype(f32)
        else: self.cy_ = np.full(PW, -400., f32)
        cyq = self.cy_[np.minimum(np.arange(GW) * Q, PW - 1)][None]
        self.far = (y < cyq)
        self.s = self.scale(x, y)
        # world coordinates: near plane (pitched ground) and far plane
        k = .80 / PH; s0 = .46; cosf = .60
        sn = s0 + k * np.maximum(y, -200); u = (x - PW * .5) / sn; v = np.log(sn) / (k * cosf)
        fy = np.clip((cyq - y), 0, 1200); sf = np.maximum(.30 - .00028 * fy, .13)
        uf = (x - PW * .5) / sf + 5200; vf = -fy / (.5 * .22) + 9000
        self.u = np.where(self.far, uf, u).astype(f32); self.v = np.where(self.far, vf, v).astype(f32)
        self.sn = np.where(self.far, sf, sn).astype(f32)
        dep = np.clip(1 - y / PH, 0, 1)
        self.air = np.where(self.far, self.far_air[0] + self.far_air[1] * np.clip(fy / 420, 0, 1) + (.34 + .12 * self.haze) * np.exp(-fy / 85.), .30 * dep ** 2.0 + .16 * self.haze * np.exp(-np.clip(y - cyq, 0, 999) / 60.) * (self.crest is not None)).astype(f32)
        self.r2 = np.zeros((GH, GW), f32)
        if self.crest2 is not None:
            from .ink import spl
            c2 = spl(np.asarray(self.crest2, float), n=24); o = np.argsort(c2[:, 0]); yy = np.interp(np.arange(GW) * Q, c2[o, 0], c2[o, 1]).astype(f32)[None]
            up = yy - y; self.r2 = np.where(self.far & (up > 0), 1., 0.).astype(f32); self.u = self.u + self.r2 * 1700; self.v = self.v - self.r2 * 900
            self.lip2 = (np.exp(-np.clip(-up, 0, 999) / 26.) * (up <= 0) * self.far).astype(f32)
            self.air = np.clip(self.air + self.r2 * (.33 * self.far_air[1] + .22 * np.exp(-np.clip(up, 0, 999) / 40.)) - .10 * self.lip2, 0, .92).astype(f32)
        # tiles
        self.tA = tile(1024, 120, self.seed + 1, .55); self.tB = tile(1024, 46, self.seed + 2, .55); self.tC = tile(1024, 17, self.seed + 3, .6)
        self.tW1 = tile(1024, 60, self.seed + 4, 2.6); self.tW2 = tile(1024, 34, self.seed + 5, 2.0); self.tWarp = tile(1024, 120, self.seed + 6)
        self.tCl = tile(1024, 150, self.seed + 7 + self.cloud_seed, .62)
        # value masses: independent light and dark shapes with lost-and-found edges (never nested contours)
        wa_ = samp(self.tB, self.u * .12, self.v * .12)
        A = samp(self.tA, self.u * .13 + 40 * wa_, self.v * .15); B = samp(self.tB, self.u * .26 + 200, self.v * .30); C = samp(self.tC, self.u * .24, self.v * .26 + 90)
        D = samp(self.tA, self.u * .20 + 512 + 40 * wa_, self.v * .23 + 300); E = samp(self.tB, self.u * .15 + 640, self.v * .21 + 410)
        hard = np.clip(.5 + .38 * samp(self.tB, self.u * .13 + 400, self.v * .13), .06, 1.0)         # 0 crisp .. 1 lost
        def mass(n, t, w0=.07, w1=.75): w = w0 + (w1 - w0) * hard; return smooth(t - w, t + w, n)
        val = .625 + .030 * A + .018 * B + .012 * C
        val = val + self.contrast * (.095 * (mass(A + .30 * B + .12 * C, -.1) - .5) - .13 * mass(D + .35 * E + .14 * C, .95) + .055 * mass(E - .3 * D + .2 * C, 1.0, .05, .4) - .05 * mass(-E + .2 * B, 1.1, .05, .4))
        for b in self.blobs:
            cx, cy, rx, ry, rot, dv, so = b; a = math.radians(rot)
            dx, dy = x - cx, y - cy; p = (dx * math.cos(a) + dy * math.sin(a)) / rx; q = (-dx * math.sin(a) + dy * math.cos(a)) / ry
            d = np.sqrt(p * p + q * q) + .13 * B + .07 * C + .05 * A
            val = val + dv * (1 - smooth(1 - so, 1 + so * .6, d))
        self.val = (val + self.level * (~self.far)).astype(f32)
        self.air = np.clip(self.air + self.haze * self.far * .16 * smooth(-.1, 1.3, samp(self.tA, self.u * .035 + 60, self.v * .30 + 33)), 0, .93).astype(f32)
        self.hue = (samp(self.tA, self.u * .2 + 333, self.v * .24 + 77)).astype(f32)
        self.hue2 = samp(self.tB, self.u * .2 + 133, self.v * .24 + 577).astype(f32)
        # slope light: the near plane falls away under the crest (bright lip), far plane hazy
        if self.crest is not None:
            lip = np.exp(-np.clip(y - cyq, 0, 900) / 46.) * (~self.far)
            self.val = self.val + .13 * lip.astype(f32) * (.6 + .4 * np.clip(B, -1, 1))
            fv = .52 + (self.val - .57) * .40
            for b in self.far_blobs:
                cx, cy, rx, ry, rot, dv, so = b; a = math.radians(rot); dx, dy = x - cx, y - cy
                p = (dx * math.cos(a) + dy * math.sin(a)) / rx; q = (-dx * math.sin(a) + dy * math.cos(a)) / ry
                fv = fv + dv * (1 - smooth(1 - so, 1 + so * .6, np.sqrt(p * p + q * q) + .10 * B + .08 * C))
            for pts, wd_, dv in self.strips:                          # drawn bands: terraces, mown rows, hedges
                from .ink import spl, resample
                P_ = resample(spl(np.array(pts, float), n=16), 6.) / Q; m_ = np.zeros((GH, GW), f32); n_ = len(P_)
                for i_, (px_, py_) in enumerate(P_): cv2.circle(m_, (int(px_ * 8), int(py_ * 8)), int(wd_ / Q * 8 * (.55 + .45 * math.sin(math.pi * min(1, max(0, i_ / n_ * 1.0 + .0))))), 1., -1, cv2.LINE_AA, 3)
                fv = fv + dv * cv2.GaussianBlur(m_, (0, 0), 1.6) * (.8 + .25 * np.clip(C, -1, 1))
            for tx, ty, tr_ in self.trees:                            # tree shadows, thrown down-right
                for k_ in range(3):
                    d = np.sqrt(((x - tx - tr_ * (.35 + .28 * k_)) / (tr_ * 1.0)) ** 2 + ((y - ty - tr_ * (.28 + .16 * k_)) / (tr_ * .52)) ** 2)
                    fv = np.where(self.far, fv - .13 * (1 - smooth(.7, 1.1, d + .1 * C)) * (1 - .2 * k_), fv)
            if self.crest2 is not None: fv = fv + .09 * self.lip2 - .03 * self.r2
            self.val = np.where(self.far, fv, self.val).astype(f32)
            if self.pond is not None:
                px_, py_, rx, ry = self.pond; d = np.sqrt(((x - px_) / rx) ** 2 + ((y - py_) / ry) ** 2) + .10 * B + .06 * C
                self.paths = getattr(self, 'paths', []) + [(((1 - smooth(.94, 1.10, d)) * .35 * self.far).astype(f32), '#4f9a94'), (((1 - smooth(.86, .96, d)) * .9 * self.far).astype(f32), '#d3edf0')]
        WASH = dict(pink='#f2a3bd', daisy='#eef2dc', butter='#f1d24c', violet='#8f86da', clover='#dcaccb')
        for cx, cy, rx, ry, rot, n, kd in self.carpets:
            a = math.radians(rot); dx, dy = x - cx, y - cy; p = (dx * math.cos(a) + dy * math.sin(a)) / rx; q = (-dx * math.sin(a) + dy * math.cos(a)) / ry - .35 * np.sin(p * 3.1 + cx)
            m = np.exp(-(p * p / .40 + q * q / .34)) * smooth(-.4, .9, samp(self.tC, self.u * .42, self.v * .42) + .5 * B) * (~self.far)
            self.paths = getattr(self, 'paths', []) + [((m * .42).astype(f32), WASH[kd])]
        self._marks(r); self._field(r)
        if self.flowers > 0: self._flowers(r)
        self._order()
        return self

    def add_path(self, pts, w0, w1, col='#d9d6b0', amt=.6):
        """pale trodden path (painting px), call after setup"""
        from .ink import spl, resample
        P = resample(spl(np.array(pts, float), n=14), 4.); m = np.zeros((GH, GW), f32); n = len(P)
        for i, (a, b) in enumerate(P / Q): cv2.circle(m, (int(a * 8), int(b * 8)), int((w0 + (w1 - w0) * i / n) / Q * 8), 1., -1, cv2.LINE_AA, 3)
        m = cv2.GaussianBlur(m, (0, 0), 1.2) * (.75 + .25 * samp(self.tC, self.u * .5, self.v * .5)); self.paths = getattr(self, 'paths', []) + [(np.clip(m, 0, 1) * amt, col)]

    def scale(self, x, y):
        """blade / flower scale at painting px"""
        x = np.asarray(x); y = np.asarray(y)
        s = .46 + .80 * np.clip(y, 0, PH) / PH
        cy = self.cy_[np.clip(x.astype(int), 0, PW - 1)]
        return np.where(y < cy, np.maximum(.30 - .00028 * (cy - y), .13), s).astype(f32)

    def flow(self, x, y):
        """resting blade heading (radians, up = -pi/2) : leans with the slope, wanders in clumps"""
        s = self.scale(x, y)
        n1 = samp(self.tA, (x / s * .34 + 900)[None].astype(f32), (y / s * .5)[None].astype(f32))[0]
        n2 = samp(self.tC, (x / s * .20)[None].astype(f32), (y / s * .3 + 50)[None].astype(f32))[0]
        return -math.pi / 2 + .30 + .34 * n1 + .16 * n2

    def _marks(self, r):
        """the base painting: each stroke stores its sample point (and a value jitter)"""
        cv = Canvas(PH, PW, color=np.zeros(3, f32)); x, y = grid(PH, PW)
        cv.img[..., 0] = x; cv.img[..., 1] = y
        for k, (hw, asp, dens, op, spread) in enumerate([(44, 3.6, 1.2, 1, .45), (22, 4.6, 1.5, 1, .4), (10.5, 5.2, 1.5, .95, .32), (5.2, 6.0, 1.25, .92, .26)]):
            # perspective-jittered sampling: rejection on 1/s^2
            n = int(PW * PH / (hw * hw * asp * 2) * 3.2 * dens)
            px = r.uniform(-30, PW + 30, n * 2); py = r.uniform(-30, PH + 30, n * 2); s = self.scale(px, py)
            keep = r.random(n * 2) < np.clip((.46 / s) ** 1.6, 0, 1) * (1.0 if k < 2 else .9)
            px, py, s = px[keep], py[keep], s[keep]
            o = np.argsort(py + r.normal(0, 30 + hw, len(py))); px, py, s = px[o], py[o], s[o]
            ang = self.flow(px, py) + r.normal(0, spread, len(px)) + (.9 if k < 2 else 0.)
            w = hw * s * r.uniform(.6, 1.15, len(px)); L = w * asp * r.uniform(.6, 1.3, len(px))
            P = arcs(px, py, ang, L, r.normal(0, .5, len(px)))
            root = P[:, 0] * .75 + P[:, 2] * .25          # sample near the root -> ragged, blade-wise edges
            wan = r.normal(0, 1, (len(px), 2)) * (w[:, None] * .8)
            col = np.c_[root[:, 0] + wan[:, 0], root[:, 1] + wan[:, 1] * .6, r.normal(0, 1, len(px))].astype(f32)
            if self.crest is not None:
                lab = (y >= self.cy_[None]).astype(np.int32); LAB = (py >= self.cy_[np.clip(px.astype(int), 0, PW - 1)]).astype(np.int32)
                lim = self.cy_[np.clip(col[:, 0].astype(int), 0, PW - 1)]; col[:, 1] = np.where(LAB > 0, np.maximum(col[:, 1], lim + 4), np.minimum(col[:, 1], lim - 4))
            else: lab = LAB = None
            cv.strokes(P, w, col, op=op, dry=(.0, .06, .16, .22)[k], seed=self.seed * 13 + k, shape='flat' if k < 2 else 'round', bristle=0.0, lab=lab, LAB=LAB)
        mx, my, mj = cv.img[..., 0], cv.img[..., 1], cv.img[..., 2]          # memory: keep only what a frame reads
        self.gx = (mx / Q - .25).astype(f32); self.gy = (my / Q - .25).astype(f32)
        self.kx = ((x + np.clip(mx - x, -11, 11)) / Q - .25).astype(f32); self.ky = ((y + np.clip(my - y, -15, 15)) / Q - .25).astype(f32)
        self.tex = (1 + .016 * mj + fbm(PH, PW, (190, 60, 18, 6), (1, .6, .35, .22), self.seed + 71) * .012).astype(f32)

    # -------------------------------------------------- blades
    def _pts(self, r, n, w_fn=None):
        """n points, perspective-dense (uniform in world), inside the near plane unless w_fn says otherwise"""
        out = []
        got = 0
        while got < n:
            m = int(n * 3)
            px = r.uniform(-20, PW + 20, m); py = r.uniform(-20, PH + 40, m); s = self.scale(px, py)
            p = np.clip((.5 / s) ** 1.7, 0, 1)
            if w_fn is not None: p = p * w_fn(px, py)
            k = r.random(m) < p
            out.append(np.c_[px[k], py[k]]); got += int(k.sum())
            if len(out) > 60: break
        return np.vstack(out)[:n]

    def _near(self, px, py):
        return (py > self.cy_[np.clip(px.astype(int), 0, PW - 1)] - 2)

    def _quiet(self, px, py, k=1.0):
        q = np.ones(len(px))
        for cx, cy, R in self.quiet: q *= smooth(R * .75 * k, R * 1.25 * k, np.hypot(px - cx, (py - cy) * 1.2))
        return q

    def _field(self, r):
        B = []                                   # columns: x y ang len w op kind class seedphase bright rootdark ox oy stemid
        d = self.density
        def add(x, y, ang, L, w, op, cls, br, dk, kind=0, key=None):
            n = len(x); B.append(np.c_[x, y, ang, L, w, np.broadcast_to(op, n), np.full(n, kind), np.full(n, cls), r.uniform(0, 6.283, n), np.broadcast_to(br, n), np.broadcast_to(dk, n), np.asarray(y if key is None else key)])
        vq = self.val; sob = cv2.magnitude(cv2.Sobel(cv2.GaussianBlur(vq, (0, 0), 3), cv2.CV_32F, 1, 0), cv2.Sobel(cv2.GaussianBlur(vq, (0, 0), 3), cv2.CV_32F, 0, 1))
        sob = cv2.GaussianBlur(sob, (0, 0), 5); sob /= np.percentile(sob, 97)
        edge = lambda px, py: np.clip(sob[np.clip((py / Q).astype(int), 0, GH - 1), np.clip((px / Q).astype(int), 0, GW - 1)], 0, 1)
        clump = lambda px, py, k, sd: samp(self.tB if sd % 2 else self.tC, (px / self.scale(px, py) * k + 311 * sd)[None].astype(f32), (py / self.scale(px, py) * k * 1.7 + 97 * sd)[None].astype(f32))[0]
        # A nap: short soft marks that carry the guide's colour over its edges
        P = self._pts(r, int(30000 * d), lambda x, y: self._near(x, y) * (.30 + .70 * smooth(-.4, .9, clump(x, y, .16, 1)) + .5 * edge(x, y)))
        s = self.scale(P[:, 0], P[:, 1]); n = len(P)
        add(P[:, 0], P[:, 1], self.flow(P[:, 0], P[:, 1]) + r.normal(0, .16, n), s * r.uniform(17, 34, n), s * r.uniform(1.3, 2.6, n), r.uniform(.35, .75, n), 0, r.normal(.02, .035, n), .0)
        # B tufts: fans of 6-22 blades sharing a foot, gathered on mass edges
        T = self._pts(r, int(1150 * d * self.tufts), lambda x, y: self._near(x, y) * (.05 + .95 * smooth(.15, 1.0, clump(x, y, .10, 3)) * (.25 + .75 * edge(x, y))) * (.15 + .85 * self._quiet(x, y)))
        for (tx, ty) in T:
            s = float(self.scale(np.array([tx]), np.array([ty]))[0]); n = int(r.integers(6, 23)); sp = r.uniform(12, 46) * s
            a0 = float(self.flow(np.array([tx]), np.array([ty]))[0]); q = np.sort(r.normal(0, .5, n))
            x = tx + q * sp; y = ty + r.normal(0, .10, n) * sp - abs(r.normal(0, .08)) * q * sp + np.abs(q) * sp * .12
            L = s * r.uniform(24, 50) * (1 - .30 * np.abs(q)) * r.uniform(.7, 1.15, n)
            tone = r.choice([-1, -1, 1, .4], p=[.36, .22, .26, .16]); strong = r.uniform(.5, 1.)
            br = tone * strong * np.where(tone < 0, .16, .10) * r.uniform(.6, 1.25, n)
            add(x, y, a0 + q * r.uniform(.35, .75) + r.normal(0, .07, n), L, s * r.uniform(1.6, 3.3, n), r.uniform(.6, .95, n), 1, br, -.10 * (tone > 0))
        # C long light tips, singly and in loose rows
        P = self._pts(r, int(2600 * d), lambda x, y: self._near(x, y) * smooth(.1, 1.1, clump(x, y, .13, 5)) * (.1 + .9 * self._quiet(x, y)))
        s = self.scale(P[:, 0], P[:, 1]); n = len(P)
        add(P[:, 0], P[:, 1], self.flow(P[:, 0], P[:, 1]) + r.normal(.05, .22, n), s * r.uniform(34, 62, n), s * r.uniform(1.2, 2.2, n), r.uniform(.5, .9, n), 2, r.uniform(.06, .20, n), -.06)
        # D dark ticks: a brush touched down in little families
        T = self._pts(r, int(230 * d), lambda x, y: self._near(x, y) * (.15 + .85 * self._quiet(x, y)))
        for (tx, ty) in T:
            s = float(self.scale(np.array([tx]), np.array([ty]))[0]); n = int(r.integers(2, 8))
            x = tx + np.cumsum(r.uniform(6, 17, n)) * s * r.choice([-1, 1]); y = ty + r.normal(0, 4, n) * s
            add(x, y, self.flow(x, y) + r.normal(.1, .18, n), s * r.uniform(8, 19, n), s * r.uniform(1.8, 3.0, n), r.uniform(.5, .85, n), 3, -r.uniform(.10, .22, n), 0., 2)
        # G silver: blades that only show their undersides when a gust lays them over
        P = self._pts(r, int(9000 * d), lambda x, y: self._near(x, y) * (.25 + .75 * smooth(-.5, .8, clump(x, y, .18, 9))))
        s = self.scale(P[:, 0], P[:, 1]); n = len(P)
        add(P[:, 0], P[:, 1], self.flow(P[:, 0], P[:, 1]) + r.normal(.12, .18, n), s * r.uniform(24, 52, n), s * r.uniform(1.0, 2.0, n), r.uniform(.55, .95, n), 6, r.uniform(.04, .16, n), .0)
        # E crest fringe
        if self.crest is not None:
            n = int(3000 * d); x = r.uniform(-10, PW + 10, n); cy = self.cy_[np.clip(x.astype(int), 0, PW - 1)]
            y = cy + np.abs(r.normal(0, 8, n)) + 1; s = self.scale(x, y + 4); gate = smooth(-.9, .5, samp(self.tC, (x * .5)[None].astype(f32), np.full((1, n), 40, f32))[0])
            add(x, y, self.flow(x, y) + r.normal(0, .22, n), s * r.uniform(13, 34, n) * (.45 + .8 * gate), s * r.uniform(1.3, 2.5, n), r.uniform(.6, .95, n), 4, r.uniform(0., .12, n), -.04)
        # F far dabs: the other meadow is only touches
        if self.crest is not None:
            P = self._pts(r, int(3000 * d), lambda x, y: (~self._near(x, y)) * (.2 + .8 * smooth(-.2, 1., clump(x, y, .12, 7))))
            if len(P):
                s = self.scale(P[:, 0], P[:, 1]); n = len(P)
                add(P[:, 0], P[:, 1], -math.pi / 2 + r.normal(.5, .5, n), s * r.uniform(18, 40, n), s * r.uniform(2.2, 4.4, n), r.uniform(.25, .6, n), 5, r.normal(0, .06, n), 0., 2)
        for tx, ty, tr_ in self.trees:
            q_ = np.random.default_rng(int(tx * 7 + ty)); lobes = [(q_.uniform(-.55, .55), q_.uniform(-.30, .30), q_.uniform(.45, .72)) for _ in range(4)] + [(0., .05, .78)]
            for k_, (n, br, rad, ox_, oy) in enumerate([(26, -.24, 1.0, .06, .16), (24, -.11, .92, .0, .02), (17, .03, .72, -.10, -.14), (10, .15, .48, -.20, -.28)]):
                for lx, ly, lr in lobes:
                    th = q_.uniform(0, 6.283, n); rr = np.sqrt(q_.random(n)) * rad * lr * tr_
                    xx = tx + (lx + ox_ * lr) * tr_ + np.cos(th) * rr; yy = ty + (ly + oy * lr) * tr_ + np.sin(th) * rr * .86
                    add(xx, yy, q_.uniform(-2.3, -.8, n), np.full(n, tr_ * .20) * q_.uniform(.7, 1.2, n), tr_ * .075 * q_.uniform(.7, 1.25, n), q_.uniform(.65, .95, n), 5, br + q_.normal(0, .02, n), 0., 2, key=np.full(n, ty - 900 + k_))
        self.B = np.vstack(B)

    # -------------------------------------------------- flowers
    def _flowers(self, r):
        """drifts of small flowers.  Each is a stem blade, a few petals, an eye; all heads ride their stem tip."""
        kinds, wts = zip(*self.p['flowers']); wts = np.array(wts) / sum(wts)
        Fl = []
        dr = list(self.drifts)
        if not dr:
            for _ in range(int(16 * self.flowers)):
                dr.append((r.uniform(0, PW), r.uniform(PH * .15, PH), r.uniform(60, 260), r.uniform(20, 70), int(r.integers(5, 22)), None))
        for cx, cy, rx, ry, n, kd in dr:
            n = int(n * self.flowers); kd = kd or str(r.choice(kinds, p=wts))
            # loose comma-shaped drift
            t = r.normal(0, .5, n); x = cx + t * rx + r.normal(0, .2, n) * rx; y = cy + (t * t - .3) * ry * .8 + r.normal(0, .42, n) * ry
            for i in range(n):
                if not (0 < x[i] < PW and 0 < y[i] < PH + 30): continue
                near = bool(self._near(np.array([x[i]]), np.array([y[i]]))[0])
                if float(self._quiet(np.array([x[i]]), np.array([y[i]]), .82)[0]) < r.random(): continue
                Fl.append((x[i], y[i], kd if r.random() < .86 else str(r.choice(kinds, p=wts)), near, r.uniform(.75, 1.25), r.uniform(0, 6.283), r.uniform(-.35, .35), int(r.integers(1 << 30))))
        for cx, cy, rx, ry, rot, n, kd in self.carpets:
            a = math.radians(rot); n = int(n * self.flowers); t = r.normal(0, .42, n); q = r.normal(0, .38, n) * (1.15 - np.abs(t) * .6); wob = .35 * np.sin(t * 3.1 + cx)
            lx, ly = t * rx, (q + wob) * ry; x = cx + lx * math.cos(a) - ly * math.sin(a); y = cy + lx * math.sin(a) + ly * math.cos(a)
            for i in range(n):
                if not (0 < x[i] < PW and 0 < y[i] < PH + 30) or not self._near(np.array([x[i]]), np.array([y[i]]))[0]: continue
                if float(self._quiet(np.array([x[i]]), np.array([y[i]]), .86)[0]) < r.random(): continue
                Fl.append((x[i], y[i], kd if r.random() < .9 else str(r.choice(kinds, p=wts)), True, float(np.clip(r.normal(1.0, .28), .45, 1.7)), r.uniform(0, 6.283), r.uniform(-.4, .4), int(r.integers(1 << 30))))
        self.Fl = Fl
        rows = []
        for x, y, kd, near, sz, ph, tl, sd in Fl:
            q = np.random.default_rng(sd); s = float(self.scale(np.array([x]), np.array([y]))[0]) * sz
            if not near or s < .3: s = max(s, .24)
            stem = q.uniform(16, 34) * s
            rows += self._flower_rows(kd, x, y, s, stem, ph, tl, q)
        self.F = np.array(rows, float) if rows else np.zeros((0, 22))

    @staticmethod
    def _flower_rows(kd, x, y, s, stem, ph, tl, q):
        """rows: x y stem ph tl | ang len w kind | c0 c1 | off0 off1 | op | z"""
        R = []
        def part(ang, L, w, kind, c0, c1, ox=0., oy=0., op=1., z=1):
            R.append([x, y, stem, ph, tl, ang, L, w, kind, *c0, *c1, ox, oy, op, z, s])
        g0, g1 = rgb('#4f9a56'), rgb('#7fbf60')
        part(-math.pi / 2, stem, .9 * s + .25, 3, g0, g1, op=.8, z=0)
        sq = q.uniform(.5, .72)                                    # foreshortening of the head seen from above at a slant
        def petals(n, L, w, cols, inner=.5, rot=None, droop=.0):
            rot = q.uniform(0, 6.28) if rot is None else rot
            for k in np.argsort([math.sin(rot + 6.283 * i / n) for i in range(n)]):
                a = rot + 6.283 * k / n + q.normal(0, .10); l = L * q.uniform(.85, 1.12)
                vx, vy = math.cos(a), math.sin(a) * sq + droop
                ang = math.atan2(vy, vx); ll = l * math.hypot(vx, vy)
                sh = .5 + .5 * math.sin(a)                           # lower petals turn from the sun
                c1 = cols[0] * (1 - .16 * sh) + cols[2] * .16 * sh; c0 = cols[1]
                part(ang, ll, w * (.75 + .25 * abs(math.cos(a)) + .1), 1, c0, c1, ox=vx * inner, oy=vy * inner)
        S = s
        if kd == 'daisy':
            petals(int(q.integers(7, 11)), 7.4 * S, 2.1 * S, (rgb('#fdfbf2'), rgb('#e6ead8'), rgb('#b9cfc9')), inner=.8 * S)
            part(0., 4.6 * S, 2.5 * S, 2, rgb('#f2b632'), rgb('#f9d75a'), ox=-2.3 * S, oy=-.2 * S, z=2)
        elif kd == 'butter':
            petals(5, 5.6 * S, 3.0 * S, (rgb('#ffe45c'), rgb('#f3b928'), rgb('#d9a52c')), inner=.4 * S)
            part(0., 2.6 * S, 1.3 * S, 2, rgb('#e09a1f'), rgb('#eeb12b'), ox=-1.3 * S, z=2)
        elif kd == 'pink':
            petals(5, 7.6 * S, 3.7 * S, (rgb('#ffc4d6'), rgb('#f07fa6'), rgb('#d98bb4')), inner=.5 * S)
            part(0., 3.0 * S, 1.5 * S, 2, rgb('#f4d35a'), rgb('#fbe58c'), ox=-1.5 * S, z=2)
        elif kd == 'violet':
            petals(5, 5.8 * S, 2.8 * S, (rgb('#9c8be8'), rgb('#6a58c4'), rgb('#6d6fc4')), inner=.3 * S)
            part(0., 2.0 * S, 1.0 * S, 2, rgb('#f7efc0'), rgb('#fff7cf'), ox=-1. * S, z=2)
        elif kd == 'clover':
            for k in range(9):
                a = q.uniform(0, 6.28); d = q.random() ** .5 * 3.4 * S
                part(-math.pi / 2 + q.normal(0, .5), 4.4 * S, 1.3 * S, 1, rgb('#d9a4c8'), rgb('#f7dbe9'), ox=math.cos(a) * d, oy=math.sin(a) * d * .7 + 1.5 * S)
        return R

    def _order(self):
        B = self.B; self.B = B[np.argsort(B[:, 11], kind='stable')]
        self.bq = (np.clip((self.B[:, 1] / Q).astype(int), 0, GH - 1), np.clip((self.B[:, 0] / Q).astype(int), 0, GW - 1))
        key = self.B[:, 11]
        if len(getattr(self, 'F', ())):
            F = self.F; self.F = F[np.lexsort((F[:, 18], F[:, 1]))]
            self.fq = (np.clip((self.F[:, 1] / Q).astype(int), 0, GH - 1), np.clip((self.F[:, 0] / Q).astype(int), 0, GW - 1))
            key = np.r_[key, self.F[:, 1] + 3 * self.F[:, 19]]
        self.perm = np.argsort(key, kind='stable')

    # -------------------------------------------------- time
    def fields(self, f):
        """gust 0..1, trailing dark, cloud shadow 0..1 on the guide grid"""
        if getattr(self, '_fc', (None,))[0] == f: return self._fc[1]
        self._fc = (f, self._fields(f)); return self._fc[1]

    def _fields(self, f):
        t = (f + self.t0) * self.speed; a = math.radians(self.wind); c, s = math.cos(a), math.sin(a)
        u, v = self.u, self.v * .80
        al = u * c + v * s; ac = -u * s + v * c
        if not hasattr(self, '_w0'):                          # wander of the fronts (static in the ground)
            self._w0 = .24 * samp(self.tWarp, al * .045 + 100, ac * .085) + .045 * samp(self.tB, al * .06 + 40, ac * .12)
            self._wp = samp(self.tWarp, u * .18, v * .18)
        wp = self._wp; n1 = samp(self.tA, ac * .13 + 300, al * .13 - t * .9)           # breathes as it goes
        if not hasattr(self, '_p0'):                          # gust_ref = (frame, x, y): a strong front stands on that painting point at that frame
            self._p0 = 0.
            if self.gust_ref is not None:
                fr_, gx_, gy_ = self.gust_ref; iy, ix = int(np.clip(gy_ / Q, 0, GH - 1)), int(np.clip(gx_ / Q, 0, GW - 1))
                self._p0 = -(al[iy, ix] / self.lam + self._w0[iy, ix]) + (fr_ + self.t0) * self.speed / self.period
        ph = al / self.lam - t / self.period + self._w0 + self._p0 + .025 * n1
        k = np.floor(ph); fr = ph - k; dpx = (1 - fr) * self.lam                        # distance behind the front (world px)
        amp = smooth(-1.0, .55, samp(self.tA, ac * .105 + k * 137.3, k * 71.7 + t * .35 + al * .02) + .35 * samp(self.tB, ac * .16 + k * 59., al * .10))
        amp = np.where(np.mod(k + 1, 2) == 0, .50 + .50 * amp if self.weak < 1 else amp, amp * self.weak)
        prof = smooth(0, 74, dpx) * np.exp(-np.clip(dpx - 60, 0, 9999) / 150.)
        comb = .74 + .26 * samp(self.tC, ac * .30, al * .07 - t * 1.6)                 # streaks combed along the wind
        g = np.clip(prof * amp * comb, 0, 1.15) * self.wave
        tr = smooth(95, 270, dpx) * smooth(560, 320, dpx) * amp * self.wave
        cl = samp(self.tCl, u * .115 - t * .42 + 40 * self.cloud_seed, self.v * .115 - t * .10)
        cl = smooth(.15, .75, cl + .10 * wp) * self.clouds
        if self.sunny:                                        # the patch of sun somebody lies in
            if not hasattr(self, '_sun'):
                x_, y_ = grid(GH, GW); self._sun = np.ones((GH, GW), f32)
                for sx_, sy_, sr_ in self.sunny: self._sun *= smooth(sr_ * .62, sr_ * 1.25, np.hypot(x_ * Q - sx_, (y_ * Q - sy_) * 1.25) + 46 * wp)
            cl = cl * self._sun
        g = cv2.GaussianBlur(g.astype(f32), (0, 0), 1.6)
        return g, np.clip(tr - g, 0, 1).astype(f32), cl.astype(f32)

    def _static(self):
        """everything the guide needs that never changes: ramp table, path / flower-wash overlay"""
        lut = ramp(np.linspace(0, 1, 1024, dtype=f32), self.p['ramp']).astype(f32)
        A = np.zeros((GH, GW), f32); C = np.zeros((GH, GW, 3), f32)
        for m, c_ in getattr(self, 'paths', ()): C = C * (1 - m[..., None]) + rgb(c_) * m[..., None]; A = A + m - A * m
        p = self.p; ac = np.where(self.far[..., None], rgb(p['air2']) * .45 + rgb(p['air']) * .55, rgb(p['air'])).astype(f32)
        if 'air3' in p: r2 = (self.r2 * self.far)[..., None] * .75; ac = (ac * (1 - r2) + rgb(p['air3']) * r2).astype(f32)
        self._st = (lut, A, C.astype(f32), ac, *(np.ascontiguousarray(v, f32) for v in (self.val, self.hue, self.hue2, self.far.astype(f32), self.air)), *(rgb(p[k]).astype(f32) for k in ('warm', 'cool', 'shade', 'flash')))
        return self._st

    def guide(self, g, tr, cl, contact=None):
        lut, A, C, ac, val, hue, hue2, far, air, warm, cool, shade, flash = getattr(self, '_st', None) or self._static()
        col = np.empty((GH, GW, 3), f32)
        _guide(col, val, g, tr, cl, hue, hue2, far, air, lut, A, C, ac, warm, cool, shade, flash, float(self.gk))
        if contact is not None: col = self._shadow(col, np.clip(contact, 0, 1)[..., None])
        return col

    def _shadow(self, col, k):
        return col * (1 - k) + (col * .30 * np.array([.72, .95, 1.05], f32) + rgb(self.p['contact']) * .70 * (.65 + .5 * lum(col)[..., None])) * k

    # -------------------------------------------------- contact shadow of whoever lies here (painting space)
    @staticmethod
    def to_painting(alpha, cam=None):
        """full-frame alpha (screen) -> painting space mask"""
        M = cv2.invertAffineTransform(Hill.matrix(cam))
        return cv2.warpAffine(alpha, M, (PW, PH), flags=cv2.INTER_LINEAR)

    @staticmethod
    def matrix(cam=None):
        """painting -> screen affine.  cam = (cx, cy, zoom, rot) painting point at frame centre"""
        cx, cy, z, ro = cam or (PW / 2, PH / 2, 1., 0.); c, s = math.cos(ro) * z, math.sin(ro) * z
        return np.float32([[c, -s, W / 2 - c * cx + s * cy], [s, c, H / 2 - s * cx - c * cy]])

    def _contact(self, m, sun=(13, 17), h=1.0):
        """m painting-space alpha -> guide-space contact/cast shadow"""
        q = cv2.resize(m, (GW, GH), interpolation=cv2.INTER_AREA)
        def sh(dx, dy): return cv2.warpAffine(q, np.float32([[1, 0, dx / Q], [0, 1, dy / Q]]), (GW, GH))
        core = cv2.dilate(q, np.ones((3, 3), np.uint8), iterations=2)
        cast = np.maximum(sh(sun[0] * h, sun[1] * h), np.maximum(sh(sun[0] * .55 * h, sun[1] * .55 * h), core))
        soft_ = cv2.GaussianBlur(np.maximum(cast, sh(sun[0] * 1.5 * h, sun[1] * 1.4 * h)), (0, 0), 9)
        return np.clip(.86 * cv2.GaussianBlur(cast, (0, 0), 1.3) + .30 * soft_, 0, 1)

    # -------------------------------------------------- per-frame blades
    def _blade_frame(self, f, G, g, cl, contact, M, sub=None, press=None, reach=None):
        B = self.B if sub is None else self.B[sub]; bq = self.bq if sub is None else (self.bq[0][sub], self.bq[1][sub])
        x, y, ang, L, w, op, kind, cls, ph, br, dk, _ = B.T
        gg = g[bq]; cc = G[bq]; t = f + self.t0
        near = cls != 5
        wa = math.radians(self.wind); wd = np.array([math.cos(wa), math.sin(wa) * .55 + .12])
        # flutter: small, individual, stronger inside a gust
        fl = np.sin(t * .42 + ph + x * .011) * (.05 + .16 * gg) + .05 * np.sin(t * .19 + ph * 2.3)
        be = np.where(near, .16 + 1.02 * gg ** 1.1 + fl, 0.)
        be = be * np.where(cls == 1, .85, 1.0)
        if press is not None:
            k, px_, py_ = press; k = k[bq]; a2 = np.arctan2(py_[bq], px_[bq])
            da = np.angle(np.exp(1j * (a2 - ang))); ang = ang + da * .62 * k; be = be * (1 - .75 * k); L = L * (1 - .10 * k)
        dx, dy = np.cos(ang), np.sin(ang)
        cb, sb = np.cos(be), np.sin(be); cm, sm = np.cos(be * .42), np.sin(be * .42)
        fs = np.where(near, 1.0, 1.0)
        tx = x + L * (dx * cb + wd[0] * sb) * fs; ty = y + L * (dy * cb + wd[1] * sb) * .92
        mx = x + .54 * L * (dx * cm + wd[0] * sm); my = y + .54 * L * (dy * cm + wd[1] * sm) * .92
        l0 = cc * (1 + dk)[:, None]
        c0 = l0
        warm = np.array([1.06, 1.02, .80]); coolm = np.array([.78, .97, 1.02])
        br = np.where(br < 0, br * (1 - .75 * np.clip(gg, 0, 1)), br); b = br[:, None]
        c1 = np.where(b > 0, cc * (1 + b * warm) + b * .10, cc * (1 + b * np.array([1.35, 1.0, .85])))
        # pale underside flash as the gust passes
        fk = np.where(near, gg ** 1.4 * np.choose(np.clip(cls, 0, 6).astype(int), [.34, .50, .62, 0., .45, .1, .85]) * np.where(br < 0, .55, 1.), 0)[:, None]
        if contact is not None: fk = fk * (1 - .8 * contact[bq][:, None])
        shd = (1 - .45 * cl[bq])[:, None]
        c1 = c1 * (1 - fk) + rgb(self.p['flash']) * fk * shd
        c1 = c1 * (1 - .0)
        kindf = kind; op = np.where(cls == 6, op * np.clip(gg * 1.5 - .12, 0, 1), op)
        d = dict(x=x, y=y, mx=mx, my=my, tx=tx, ty=ty, w=w, c0=c0, c1=c1, op=op, kind=kindf)
        return self._proj(d, M)

    @staticmethod
    def _proj(d, M):
        z = math.hypot(M[0, 0], M[1, 0])
        for a, b in (('x', 'y'), ('mx', 'my'), ('tx', 'ty')):
            X, Y = d[a], d[b]; d[a] = M[0, 0] * X + M[0, 1] * Y + M[0, 2]; d[b] = M[1, 0] * X + M[1, 1] * Y + M[1, 2]
        d['w'] = d['w'] * z
        return d

    def _flower_frame(self, f, G, g, cl, M, sub=None, contact=None):
        F = self.F if sub is None else self.F[sub]
        if len(F) == 0: return None
        fq = self.fq if sub is None else (self.fq[0][sub], self.fq[1][sub])
        x, y, stem, ph, tl, ang, L, w, kind = F[:, :9].T; c0 = F[:, 9:12]; c1 = F[:, 12:15]; ox, oy, op, z, s = F[:, 15:20].T
        gg = g[fq]; t = f + self.t0
        be = tl * .6 + .12 + .75 * gg + (.06 + .14 * gg) * np.sin(t * .33 + ph)
        hx = x + stem * np.sin(be); hy = y - stem * np.cos(be) * .86
        st = kind == 3
        X = np.where(st, x, hx + ox); Y = np.where(st, y, hy + oy)
        TX = np.where(st, hx, X + np.cos(ang) * L); TY = np.where(st, hy, Y + np.sin(ang) * L)
        mx = np.where(st, x + stem * .5 * np.sin(be * .45), (X + TX) / 2); my = np.where(st, y - stem * .5 * np.cos(be * .45) * .86, (Y + TY) / 2)
        a = self.air[fq][:, None]; sh = cl[fq][:, None]
        if contact is not None: sh = np.maximum(sh, contact[fq][:, None] * .8)
        amb = 1 - .38 * sh * np.array([1.15, .85, .6])
        c0, c1 = c0 * amb, c1 * amb
        ac = rgb(self.p['air']); c0 = c0 * (1 - a * .8) + ac * a * .8; c1 = c1 * (1 - a * .8) + ac * a * .8
        d = dict(x=X, y=Y, mx=mx, my=my, tx=TX, ty=TY, w=w, c0=c0, c1=c1, op=op, kind=np.where(st, 0, kind))
        return self._proj(d, M)

    # -------------------------------------------------- who lies here
    def seat(self, masks, sun=(13, 17), h=1.0, fringe=1.0, extra=None):
        """bed for moving cameras and moving limbs.  masks = [painting-space alpha, ...] (body, arm, ...); pass the result to
        frame(seat=..., w=[1., .4, ...]): w lifts a part out of the grass (its shadow thins, its fringe sinks, its blades spring back)."""
        parts = []
        for m in masks:
            mq = cv2.resize(m, (GW, GH), interpolation=cv2.INTER_AREA); dt = cv2.distanceTransform((mq < .5).astype(np.uint8), cv2.DIST_L2, 3)
            ct = self._contact(m, sun, h); ys, xs = np.nonzero(ct > .004); bx = (max(0, ys.min() - 14), min(GH, ys.max() + 14), max(0, xs.min() - 14), min(GW, xs.max() + 14)) if len(ys) else (0, 1, 0, 1)
            parts.append(dict(contact=ct[bx[0]:bx[1], bx[2]:bx[3]].copy(), bx=bx, dt=dt, m=m))
        allm = np.maximum.reduce(masks); out = []
        for p_ in parts: p_['front'] = self._front_set(p_['m'], fringe, allm); del p_['m']
        return dict(parts=parts, extra=extra)

    def _seat_now(self, seat, w, breathe=None):
        """breathe = [b, ...] per part (-1..1 or bigger): its shadow swells and slides a little with the chest"""
        c = np.zeros((GH, GW), f32)
        for i, (p, wi) in enumerate(zip(seat['parts'], w)):
            y0, y1, x0, x1 = p['bx']; r = p['contact']
            if breathe is not None and breathe[i]:
                b = breathe[i]; sc = 1 + .022 * b; h_, w_ = r.shape
                r = cv2.warpAffine(r, np.float32([[sc, 0, (1 - sc) * w_ / 2 + 1.5 * b], [0, sc, (1 - sc) * h_ / 2 + 1.1 * b]]), (w_, h_))
            np.maximum(c[y0:y1, x0:x1], r * wi, out=c[y0:y1, x0:x1])
        if seat['extra'] is not None: c = np.maximum(c, seat['extra'])
        if all(v == 1 for v in w) and 'still' in seat: return (c,) + seat['still']
        k = np.maximum.reduce([np.exp(-p['dt'] * Q / 20.) * wi for p, wi in zip(seat['parts'], w)]).astype(f32)
        ds = cv2.GaussianBlur(np.minimum.reduce([p['dt'] + (0 if wi > .3 else 400) for p, wi in zip(seat['parts'], w)]), (0, 0), 4)
        S = [np.c_[p['front'][:, :3], p['front'][:, 3] * wi, p['front'][:, 4:]] for p, wi in zip(seat['parts'], w) if p['front'] is not None and wi > .05]
        out = ((k, cv2.Sobel(ds, cv2.CV_32F, 1, 0), cv2.Sobel(ds, cv2.CV_32F, 0, 1)), (np.vstack(S) if S else None))
        if all(v == 1 for v in w): seat['still'] = out
        return (c,) + out

    # -------------------------------------------------- frame
    def frame(self, f, layer=None, rest=None, cam=None, sun=(13, 17), shadow_h=1.0, front=True, fringe=1.0, extra_contact=None, seat=None, w=None, breathe=None):
        """finished ground (and, if given, the characters lying in it).
        layer = premultiplied RGBA full frame; rest = alpha used for shadows/fringe (default layer alpha; pass a held
        pose so the fringe never pops).  For moving cameras or limbs use seat()."""
        M = self.matrix(cam); g, tr, cl = self.fields(f)
        contact = press = None; m = None; S = None
        if seat is not None: contact, press, S = self._seat_now(seat, w or [1.] * len(seat['parts']), breathe)
        elif layer is not None or rest is not None:
            a = rest if rest is not None else layer[..., 3]
            m = self.to_painting(a, cam); contact = self._contact(m, sun, shadow_h)
            if extra_contact is not None: contact = np.maximum(contact, extra_contact)
            mq = cv2.resize(m, (GW, GH), interpolation=cv2.INTER_AREA)
            dt = cv2.distanceTransform((mq < .5).astype(np.uint8), cv2.DIST_L2, 3); ds = cv2.GaussianBlur(dt, (0, 0), 4)
            gx = cv2.Sobel(ds, cv2.CV_32F, 1, 0); gy = cv2.Sobel(ds, cv2.CV_32F, 0, 1)
            press = (np.exp(-dt * Q / 20.).astype(f32), gx, gy)
            key = float(m.sum())
            if getattr(self, '_fk', None) is None or abs(self._fk[0] - key) > 1e-3 * max(key, 1): self._fk = (key, self._front_set(m, fringe))
            S = self._fk[1]
        G0 = self.guide(g, tr, cl); G = G0 if contact is None else self._shadow(G0, np.clip(contact, 0, 1)[..., None])
        base = cv2.remap(G0, self.gx, self.gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if contact is not None:                                        # the shadow is read through the marks too, but never far from where it fell
            ys, xs = np.nonzero(contact > .01)
            if len(xs):
                x0, x1, y0, y1 = max(0, xs.min() * Q - 40), min(PW, xs.max() * Q + 40), max(0, ys.min() * Q - 40), min(PH, ys.max() * Q + 40)
                kk = cv2.remap(contact, self.kx[y0:y1, x0:x1], self.ky[y0:y1, x0:x1], cv2.INTER_LINEAR)[..., None]; R = base[y0:y1, x0:x1]
                base[y0:y1, x0:x1] = self._shadow(R, kk)
        base *= self.tex[..., None]
        if cam is None: base = base[MY:MY + H, MX:MX + W].copy()
        else: base = cv2.warpAffine(base, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        base = np.ascontiguousarray(base.astype(f32))
        d = self._blade_frame(f, G, g, cl, contact, M, press=press)
        fr = self._flower_frame(f, G, g, cl, M, contact=contact) if len(getattr(self, 'F', ())) else None
        if fr is not None: d = {k: np.concatenate([d[k], fr[k]]) for k in d}
        d = {k: v[self.perm] for k, v in d.items()}
        draw_blades(base, d)
        if layer is not None:
            ck = cv2.warpAffine(cl, M @ np.array([[Q, 0, 0], [0, Q, 0], [0, 0, 1.]]), (W, H), flags=cv2.INTER_LINEAR)[..., None].astype(f32)
            layer = np.concatenate([layer[..., :3] * (1 - ck * np.array([.27, .18, .12], f32)), layer[..., 3:]], -1)
            base = over(base, layer)
            if front and S is not None: base = self._front(base, f, g, cl, contact, M, S, layer[..., 3])
        return base

    # -------------------------------------------------- blades that cross the sleeper's edge
    def _front(self, base, f, g, cl, contact, M, S, al):
        sx = np.clip(M[0, 0] * S[:, 0] + M[0, 1] * S[:, 1] + M[0, 2], 0, W - 1).astype(int); sy = np.clip(M[1, 0] * S[:, 0] + M[1, 1] * S[:, 1] + M[1, 2] + 1, 0, H - 1).astype(int)
        S = S[al[sy, sx] < .5]                                                        # somebody is lying on that root now
        x, y, ang, L, w, op, ph, br = S.T
        bq = (np.clip((y / Q).astype(int), 0, GH - 1), np.clip((x / Q).astype(int), 0, GW - 1))
        gg = g[bq]; t = f + self.t0; wa = math.radians(self.wind); wd = np.array([math.cos(wa), math.sin(wa) * .5])
        be = .08 + .55 * gg + (.04 + .12 * gg) * np.sin(t * .42 + ph)
        dx, dy = np.cos(ang), np.sin(ang); cb, sb = np.cos(be), np.sin(be); cm, sm = np.cos(be * .42), np.sin(be * .42)
        tx = x + L * (dx * cb + wd[0] * sb); ty = y + L * (dy * cb + wd[1] * sb) * .95
        mx = x + .54 * L * (dx * cm + wd[0] * sm); my = y + .54 * L * (dy * cm + wd[1] * sm) * .95
        col = self.guide_at(bq, g, cl); c0 = col * .80; c1 = col * (1.02 + br)[:, None] + .02
        fk = (gg ** 1.4 * .4)[:, None]; c1 = c1 * (1 - fk) + rgb(self.p['flash']) * fk
        k = contact[bq][:, None] * .62; c0 = c0 * (1 - k) + rgb(self.p['contact']) * k; c1 = c1 * (1 - k * .35) + rgb(self.p['contact']) * k * .35
        d = self._proj(dict(x=x, y=y, mx=mx, my=my, tx=tx, ty=ty, w=w, c0=c0, c1=c1, op=op, kind=np.zeros(len(x))), M)
        return draw_blades(base, d)

    def guide_at(self, bq, g, cl):
        p = self.p; v = self.val[bq] + .085 * g[bq] - .17 * cl[bq]
        col = ramp(v, p['ramp']); k = (.40 * cl[bq])[:, None]; col = col * (1 - k) + col * rgb(p['shade']) * 1.75 * k
        a = self.air[bq][:, None]; return col * (1 - a) + rgb(p['air']) * a

    def _front_set(self, m, fringe, allm=None):
        """static blades rooted just outside the lower contour, reaching over it"""
        ys, xs = np.nonzero(m > .5)
        if len(xs) == 0: return None
        x0, x1, y0, y1 = xs.min() - 40, xs.max() + 40, ys.min() - 20, ys.max() + 60
        r = np.random.default_rng(self.seed * 7 + 5)
        n = int((x1 - x0) * (y1 - y0) / 9.)
        x = r.uniform(x0, x1, n); y = r.uniform(y0, y1, n); s = self.scale(x, y)
        ix = np.clip(x.astype(int), 0, PW - 1); iy = np.clip(y.astype(int), 0, PH - 1)
        out = (m if allm is None else allm)[iy, ix] < .45
        mb = cv2.GaussianBlur(m, (0, 0), 2.0)
        up = mb[np.clip((y - 11 * s).astype(int), 0, PH - 1), ix] > .5            # somebody is just above this root
        close = cv2.dilate((m > .5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)))[iy, ix] > 0
        cl_ = samp(self.tC, (x * .9 + 40)[None].astype(f32), (y * .9)[None].astype(f32))[0]          # clusters, gaps between
        dens = np.where(up, .10 + .80 * smooth(-.1, .9, cl_), .0)
        k = out & close & up & (r.random(n) < dens * fringe * .75)
        x, y, s = x[k], y[k], s[k]
        if len(x) == 0: return None
        o = np.argsort(y); x, y, s = x[o], y[o], s[o]; n = len(x)
        ang = -math.pi / 2 + .10 + r.normal(0, .26, n)
        L = s * r.uniform(11, 27, n); w = s * r.uniform(1.5, 3.0, n)
        return np.c_[x, y, ang, L, w, r.uniform(.85, 1., n), r.uniform(0, 6.283, n), r.normal(.05, .08, n)]


# ------------------------------------------------------------------ foreground out of focus + air
class Foreground:
    """big soft blades and flower heads close to the lens (a blurred frame for depth)"""
    def __init__(self, clumps, season='summer', seed=9, blur_px=9.):
        """clumps = [(x, y, size, n, lean, kind|None)] screen px of the foot"""
        self.cl = clumps; self.p = PAL[season]; self.seed = seed; self.b = blur_px

    def layer(self, f, gust=.3, z=1., dx=0., dy=0.):
        """z, dx, dy: lens zoom and screen shift, so near plants travel faster than the hill"""
        q = 2; h, w = H // q, W // q; img = np.zeros((h, w, 3), f32); al = np.zeros((h, w), f32); img[:] = rgb(self.p['ramp'][2][1])
        r = np.random.default_rng(self.seed); rows = dict(x=[], y=[], mx=[], my=[], tx=[], ty=[], w=[], c0=[], c1=[], op=[], kind=[])
        def add(x, y, mx, my, tx, ty, ww, c0, c1, kind=0, op=1.):
            for k, v in zip(('x', 'y', 'mx', 'my', 'tx', 'ty'), (x, y, mx, my, tx, ty)): rows[k].append(v / q)
            rows['w'].append(ww / q); rows['c0'].append(c0); rows['c1'].append(c1); rows['op'].append(op); rows['kind'].append(kind)
        rp = [rgb(c) for _, c in self.p['ramp']]
        for x, y, sz, n, lean, kd in self.cl:
            x, y, sz = W / 2 + (x - W / 2) * z + dx, H / 2 + (y - H / 2) * z + dy, sz * z
            for i in range(int(n)):
                u = r.normal(0, 1); xx = x + u * sz * .45; L = sz * r.uniform(.6, 1.25) * (1 - .14 * abs(u)); ph = r.uniform(0, 6.28)
                a = -math.pi / 2 + lean + u * .30 + r.normal(0, .10); be = .18 + .32 * gust + .06 * math.sin(f * .13 + ph) + .025 * math.sin(f * .31 + ph * 2)
                dx, dy = math.cos(a), math.sin(a); tx = xx + L * (dx * math.cos(be) + math.sin(be)); ty = y + L * dy * math.cos(be) + L * .25 * math.sin(be)
                mx = xx + .54 * L * (dx * math.cos(be * .4) + math.sin(be * .4)); my = y + .54 * L * dy
                k = r.random(); c = rp[1] * (1 - k) + rp[3] * k; add(xx, y, mx, my, tx, ty, sz * r.uniform(.030, .062), c * .72, c * 1.18)
                if kd and r.random() < .34:
                    col = {'pink': ('#f07fa6', '#ffd0de'), 'daisy': ('#e6ead8', '#fffdf4'), 'butter': ('#f3b928', '#ffe45c'), 'violet': ('#6a58c4', '#a99af0')}[kd]
                    np_ = 5 if kd != 'daisy' else 9; ro = r.uniform(0, 6.28); pl = sz * r.uniform(.075, .15)
                    for j in range(np_):
                        aa = ro + 6.283 * j / np_; ex, ey = math.cos(aa), math.sin(aa) * .7
                        add(tx + ex * pl * .1, ty + ey * pl * .1, tx + ex * pl * .55, ty + ey * pl * .55, tx + ex * pl, ty + ey * pl, pl * (.42 if np_ == 5 else .2), rgb(col[0]), rgb(col[1]), 1)
                    add(tx - pl * .2, ty, tx, ty, tx + pl * .2, ty, pl * .2, rgb('#f2b632'), rgb('#f9d75a'), 2)
        d = {k: np.array(v) for k, v in rows.items()}
        draw_blades(img, d, al)
        L = np.dstack([img * al[..., None], al]); L = cv2.GaussianBlur(L, (0, 0), self.b / q)
        return cv2.resize(L, (W, H), interpolation=cv2.INTER_LINEAR)


def motes(base, f, n=46, seed=4, col='#fff7d6', wind=(1.6, .35), amt=.5, box=(0, 0, W, H), petals=None):
    """pollen in the light (and optional drifting petals = [(hex0, hex1)]), pure function of f"""
    r = np.random.default_rng(seed); q = 2; gl = np.zeros((H // q, W // q), f32)
    for i in range(n):
        z = r.uniform(.4, 1.6); x = r.uniform(0, W) + wind[0] * z * f + 22 * math.sin(f * .05 * z + i); y = r.uniform(0, H) + wind[1] * z * f + 14 * math.sin(f * .07 + i * 2.1)
        x, y = (x + 40) % (W + 80) - 40, (y + 40) % (H + 80) - 40; tw = .55 + .45 * math.sin(f * .11 * z + i * 1.7)
        cv2.circle(gl, (int(x / q * 16), int(y / q * 16)), int((.6 + 1.0 * z) * 16), float(tw), -1, cv2.LINE_AA, 4)
    gl = cv2.GaussianBlur(gl, (0, 0), 1.2); gl = gl + .6 * cv2.GaussianBlur(gl, (0, 0), 5)
    gl = cv2.resize(gl, (W, H), interpolation=cv2.INTER_LINEAR)
    k = np.clip(gl * amt, 0, 1)[..., None]
    out = base + (np.maximum(base, rgb(col)) - base) * k
    if petals:
        rows = dict(x=[], y=[], mx=[], my=[], tx=[], ty=[], w=[], c0=[], c1=[], op=[], kind=[])
        for i in range(len(petals) * 9):
            a, b = petals[i % len(petals)]; z = r.uniform(.6, 1.7)
            x = r.uniform(0, W) + wind[0] * 2.2 * z * f + 40 * math.sin(f * .06 + i); y = r.uniform(0, H) + (wind[1] + .5) * 1.6 * z * f + 26 * math.sin(f * .09 + i * 1.3)
            x, y = (x + 60) % (W + 120) - 60, (y + 60) % (H + 120) - 60; ro = f * .10 * z + i; fl = abs(math.sin(f * .13 * z + i * .7)) * .8 + .2; s = 7.5 * z
            ex, ey = math.cos(ro) * s, math.sin(ro) * s
            for k_, v in zip(('x', 'y', 'mx', 'my', 'tx', 'ty'), (x - ex, y - ey, x + ey * .18, y - ex * .18, x + ex, y + ey)): rows[k_].append(v)
            rows['w'].append(s * .62 * fl); rows['c0'].append(rgb(a)); rows['c1'].append(rgb(b)); rows['op'].append(.95); rows['kind'].append(1)
        d = {k_: np.array(v) for k_, v in rows.items()}
        out = np.ascontiguousarray(out.astype(f32)); draw_blades(out, d)
    return out
