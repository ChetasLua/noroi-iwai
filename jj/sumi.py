"""Sumi: a bristled brush on paper.

A brush is a few thousand hairs in a teardrop footprint, gathered in tufts.  It is dragged along an authored
skeleton (x, y, width) sampled in time; every hair leaves ink where its load, its pressure and its speed beat
the tooth of the paper.  Dry tufts part, skip and fork: that is the flying white.  Every pixel also remembers
WHEN it was first touched, so a stroke can be drawn for any t in 0..1 by a threshold.

    pg = Page()                                  # 1920x1080, simulated at 2x
    pg.stroke([(x, y, w), ...], t0=0, t1=1)      # widths in px, authored pressure
    pg.splash(x, y, power, t=0)                  # landing splash
    ink, wet = pg.ink(t)                         # ink thickness (0 paper .. 1 solid black) and wetness
    img = lay(paper(), ink, wet)                 # composite on washi
"""
import numpy as np, cv2, math
from numba import njit
from scipy.interpolate import PchipInterpolator
from . import W, H
from .paint import rgb, f32, fbm, smooth, mix, grid, _vn, _h2
from .ink import Cel, spl, resample, noise1, ribbon

SS = 2                       # simulation scale
BLACK, BONE, BLOOD, BLOOD_D, WASH, SHEET = '#0d0b0e', '#e6dcc8', '#b3121f', '#6d0c16', '#8d8782', '#f7f4ec'


# ------------------------------------------------------------------ paper
_tooth = {}


def tooth(h, w, seed=5):
    """paper relief 0..1 (1 = deep pit the brush cannot reach), at simulation scale, tiled from a cached sheet"""
    S = 2304
    if seed not in _tooth:
        r = np.random.default_rng(seed)
        n = fbm(S, S, (3.6, 2.0, 1.2), (1, .9, .6), seed) * .9 + fbm(S, S, (26, 11), (1, .7), seed + 4) * .10
        # long-fibre washi: short curved ridges
        fib = np.zeros((S, S), f32)
        for _ in range(5200):
            x, y = r.uniform(0, S, 2); a = r.uniform(0, math.pi); L = r.uniform(18, 110); c = r.normal(0, .7)
            t = np.linspace(0, 1, 9); th = a + c * t
            P = np.c_[x + np.cumsum(np.cos(th)) * L / 9, y + np.cumsum(np.sin(th)) * L / 9]
            cv2.polylines(fib, [np.round(P * 8).astype(np.int32)], False, float(r.uniform(-1, 1)), 1, cv2.LINE_AA, 3)
        fib = cv2.GaussianBlur(fib, (0, 0), .8)
        n = n + fib / fib.std() * .13
        n = (n - n.mean()) / n.std()
        _tooth[seed] = (.5 + .5 * np.tanh(n * .62)).astype(f32)
    t = _tooth[seed]
    if h <= S and w <= S: return t[:h, :w]
    ry, rx = -(-h // S), -(-w // S)
    return np.tile(t, (ry, rx))[:h, :w]


_paper = {}


def paper(h=H, w=W, seed=11, tone=SHEET, amt=1.0):
    """washi: warm white, faintly clouded, long fibres, fine relief and a few flecks.  float RGB"""
    key = (h, w, seed, tone, amt)
    if key in _paper: return _paper[key].copy()
    r = np.random.default_rng(seed)
    base = np.ones((h, w, 3), f32) * rgb(tone)
    m = fbm(h, w, (520, 210, 80), (1, .6, .3), seed); n = fbm(h, w, (9, 4, 1.8), (1, .8, .7), seed + 1)
    base *= (1 + amt * (.0065 * m + .0055 * n))[..., None]
    base += (amt * .0011 * fbm(h, w, (420, 160), (1, .5), seed + 2))[..., None] * np.array([1, .3, -1.2], f32)
    fib = np.zeros((h * 2, w * 2), f32)
    nf = int(1500 * h * w / (W * H))
    for _ in range(nf):
        x, y = r.uniform(0, 2 * w), r.uniform(0, 2 * h); a = r.uniform(0, math.pi); L = r.uniform(30, 240); c = r.normal(0, .9)
        t = np.linspace(0, 1, 14); th = a + c * t + .3 * np.sin(t * r.uniform(2, 8) + r.uniform(0, 6))
        P = np.c_[x + np.cumsum(np.cos(th)) * L / 14, y + np.cumsum(np.sin(th)) * L / 14]
        cv2.polylines(fib, [np.round(P * 8).astype(np.int32)], False, float(r.choice([-1, 1]) * r.uniform(.2, 1) ** 1.5), 1, cv2.LINE_AA, 3)
    fib = cv2.resize(cv2.GaussianBlur(fib, (0, 0), .8), (w, h), interpolation=cv2.INTER_AREA)
    base *= (1 + amt * .040 * fib)[..., None]
    for _ in range(int(16 * h * w / (W * H))):
        x, y = r.uniform(0, w), r.uniform(0, h)
        cv2.ellipse(base, (int(x), int(y)), (int(r.integers(1, 3)), 1), r.uniform(0, 180), 0, 360, tuple(float(v) for v in rgb(tone) * r.uniform(.78, .92)), -1, cv2.LINE_AA)
    base = np.clip(base, 0, 1).astype(f32)
    _paper[key] = base
    return base.copy()


# --------------------------------------------------------------- skeleton -> sampled stroke
def _cr2(P, n):
    """centripetal Catmull-Rom through P (m,2), n samples per span; returns (xy, span parameter)"""
    m = len(P)
    if m == 2:
        t = np.linspace(0, 1, n * 2)
        return P[0] + (P[1] - P[0]) * t[:, None], t
    out, tt = [], []
    for i in range(m - 1):
        p1, p2 = P[i], P[i + 1]
        p0 = P[i - 1] if i > 0 else 2 * p1 - p2
        p3 = P[i + 2] if i + 2 < m else 2 * p2 - p1
        d1 = max(np.hypot(*(p1 - p0)), 1e-4) ** .5; d2 = max(np.hypot(*(p2 - p1)), 1e-4) ** .5; d3 = max(np.hypot(*(p3 - p2)), 1e-4) ** .5
        m1 = (p2 - p1) + d2 * ((p1 - p0) / d1 - (p2 - p0) / (d1 + d2)); m2 = (p2 - p1) + d2 * ((p3 - p2) / d3 - (p3 - p1) / (d2 + d3))
        t = np.linspace(0, 1, n, endpoint=(i == m - 2))[:, None]
        out.append((2 * t ** 3 - 3 * t ** 2 + 1) * p1 + (t ** 3 - 2 * t ** 2 + t) * m1 + (-2 * t ** 3 + 3 * t ** 2) * p2 + (t ** 3 - t ** 2) * m2)
        tt.append(i + t[:, 0])
    return np.concatenate(out), np.concatenate(tt)


def skeleton(keys, corners=(), step=.75, vmin=.22, vkeys=None):
    """keys (x, y, width[, speed]) -> dict of samples every `step` px: x y w hd (heading) s (arc) t (0..1 time) v (speed 0..1)"""
    K = np.asarray(keys, float)
    cs = [0] + sorted(int(c) % len(K) for c in corners if 0 < int(c) % len(K) < len(K) - 1) + [len(K) - 1]
    xy, par = [], []
    for a, b in zip(cs[:-1], cs[1:]):
        q, t = _cr2(K[a:b + 1, :2], 48)
        if b != cs[-1]: q, t = q[:-1], t[:-1]
        xy.append(q); par.append(t + a)
    xy = np.concatenate(xy); par = np.concatenate(par)
    d = np.hypot(*np.diff(xy, axis=0).T); s = np.r_[0, np.cumsum(d)]
    keep = np.r_[True, d > 1e-9]; xy, par, s = xy[keep], par[keep], s[keep]
    n = max(8, int(s[-1] / step))
    S = np.linspace(0, s[-1], n)
    x, y, p = np.interp(S, s, xy[:, 0]), np.interp(S, s, xy[:, 1]), np.interp(S, s, par)
    kp = np.arange(len(K))
    w = PchipInterpolator(kp, K[:, 2])(p) if len(K) > 2 else np.interp(p, kp, K[:, 2])
    dx, dy = np.gradient(x), np.gradient(y)
    hd = np.unwrap(np.arctan2(dy, dx))
    # speed: slow at ends and corners, fast between
    ks = np.interp(kp, p, S)
    slow = [0., S[-1]] + [ks[c] for c in cs[1:-1]]
    v = np.ones(n)
    wm = max(8., float(np.median(w)))
    for q, sl in enumerate(slow):
        reach = wm * (1.3 if q != 1 else .8)
        v = np.minimum(v, vmin + (1 - vmin) * np.clip(np.abs(S - sl) / reach, 0, 1) ** .8)
    if K.shape[1] > 3:
        v = v * np.clip(np.interp(p, kp, K[:, 3]), .05, 3)
    dt = (S[1] - S[0]) / np.clip(v, .05, None); T = np.cumsum(dt); T = (T - T[0]) / (T[-1] - T[0])
    xd = PchipInterpolator(kp, K[:, 4])(p) if K.shape[1] > 4 else np.zeros(n)
    return dict(x=x, y=y, w=w, hd=hd, s=S, t=T, v=np.clip(v, 0, 1.4), L=S[-1], k=p, xd=xd)


# --------------------------------------------------------------- the brush
@njit(cache=True, fastmath=True)
def _drag(ink, wet, arr, tth, X, Y, WD, PH, T, V, DRY, SPL, S, FK, A, B, TB, TR, TL, TID, FID, HW, seed, elong, rad, gain, toff, grip, wref):
    """n steps x m hairs.  ink/wet accumulate, arr keeps first touch."""
    h, w = ink.shape
    n = X.shape[0]; m = A.shape[0]
    th, tw = tth.shape
    for j in range(n):
        wd = WD[j]
        if wd < .5: continue
        c = math.cos(PH[j]); s = math.sin(PH[j])
        L = wd * elong
        dry = DRY[j]; spd = V[j]; spl_ = SPL[j]; t = T[j]; fk = FK[j]
        big = wd / wref
        # hairs that splay do not come back to a point: lateral room kept while the brush lifts
        lat = wd * (1 - fk) + fk * (.40 * wref + .60 * wd)
        for k in range(m):
            tid = TID[k]
            if fk > .01 and big < TL[tid] * fk * 1.6: continue        # this tuft has left the paper
            a = A[k]
            b = TB[k] + (B[k] - TB[k]) * (1 - .86 * spl_)
            if spl_ > .02 or fk > .02:
                b += (_vn(tid * 3.17, S[j] / (wref * 1.6), seed + 11) - .5) * .22 * (spl_ + fk)
            a2 = a * a
            prof = math.pow(a, .66) * math.sqrt(max(0., 1 - a2 * a2 * a)) * 1.20
            u = (a - .52) * L
            v = b * prof * lat * .5
            px = X[j] + c * u - s * v
            py = Y[j] + s * u + c * v
            ix = int(px); iy = int(py)
            if ix < 2 or iy < 2 or ix >= w - 2 or iy >= h - 2: continue
            d = dry * TR[k]
            if dry > .02:
                d += dry * 1.5 * (_vn(tid * 5.3 + 2., S[j] / (wref * .95), seed) - .5)
                d += dry * 1.5 * (_vn(FID[k] * 2.9 + 7., S[j] / (wref * .42), seed + 3) - .5)
                d += dry * .7 * (_vn(k * 1.7 + 3., S[j] / (wref * .16), seed + 5) - .5)
            b2 = B[k] * B[k]; b4 = b2 * b2
            con = 1.14 - .16 * b2 - .62 * b4 * b4 - .10 * a
            if HW[k] < .5:                                             # stray hair: touches now and then
                con = 2.6 * _vn(k * 4.1, S[j] / (wref * .22), seed + 8) - 1.55
            tv = tth[(iy + toff) % th, (ix + toff * 3) % tw]
            g = con - d - tv * (grip + .30 * spd * (.18 + dry * .55))
            g = g * 7.0 + .35
            if g <= 0: continue
            if g > 1: g = 1.0
            am = g * gain * (1.0 if HW[k] > .5 else 1.4)
            fx = px - ix; fy = py - iy
            for oy in range(-1, 2):
                wy = 1 - abs(oy - fy) / rad
                if wy <= 0: continue
                for ox in range(-1, 2):
                    wx = 1 - abs(ox - fx) / rad
                    if wx <= 0: continue
                    q = am * wx * wy
                    yy = iy + oy; xx = ix + ox
                    ink[yy, xx] += q
                    wet[yy, xx] += q * (1.12 - min(1., spd)) * (1 - min(1., dry))
                    if q > .004 and t < arr[yy, xx]: arr[yy, xx] = t


def brush_hairs(width, seed=0, tufts=None, n=None, stray=.007):
    """hair table for a brush of `width` simulation px"""
    r = np.random.default_rng(seed)
    nt = tufts or int(np.clip(6 + width / 8.0, 6, 52))
    n = n or int(np.clip(width * 19, 260, 5600))
    ed = np.sort(r.uniform(-1, 1, nt)); ed = .5 * ed + .5 * np.linspace(-1, 1, nt)
    tw = np.abs(np.gradient(ed)) * .74 + .01
    wt = tw / tw.sum()
    tid = r.choice(nt, n, p=wt)
    b = np.clip(ed[tid] + r.uniform(-1, 1, n) * tw[tid] * 1.02, -1, 1)
    a = r.beta(1.35, 1.2, n)
    tr = r.lognormal(0, .55, nt); tr = np.clip(tr / np.median(tr), .30, 3.0)
    tr = .8 * tr + .2 * np.convolve(np.pad(tr, 1, 'edge'), np.ones(3) / 3, 'valid')
    tl = r.beta(1.0, 1.5, nt)
    fid = (tid * 5 + np.clip(((b - ed[tid]) / (tw[tid] + 1e-6) + 1) * 2.5, 0, 4).astype(int))
    hw = np.ones(n)
    ns = int(n * stray); q = r.choice(n, ns, replace=False)                  # strays stick out at the sides
    sg = np.sign(r.uniform(-1, 1, ns)); b = b.copy(); b[q] = sg * (1.0 + .20 * r.random(ns) ** 2); a[q] = r.uniform(.35, .9, ns); hw[q] = .49
    tb = ed[tid].copy(); tb[q] = b[q]
    return dict(A=a, B=b, TB=tb, TR=tr[tid], TL=tl, TID=tid.astype(np.int64), FID=fid.astype(np.int64), HW=hw)


class Mark:
    """one drawn element: ink, wet, first-touch time on a ROI at 1x, plus when it happens"""
    __slots__ = ('ox', 'oy', 'd', 'wet', 'arr', 't0', 't1', 'kind', 'bleed', 'info')

    def __init__(self, ox, oy, d, wet, arr, t0, t1, kind='stroke', bleed=None, info=None):
        self.ox, self.oy, self.d, self.wet, self.arr, self.t0, self.t1, self.kind, self.bleed, self.info = ox, oy, d, wet, arr, t0, t1, kind, bleed, info


def _down(ink, wet, arr, ss):
    h, w = ink.shape[0] // ss, ink.shape[1] // ss
    a = np.where(arr > 1.5, 0, arr)
    cov = (ink > .02).astype(f32)
    k = cv2.INTER_AREA
    d = cv2.resize(ink, (w, h), interpolation=k); we = cv2.resize(wet, (w, h), interpolation=k)
    c = cv2.resize(cov, (w, h), interpolation=k); t = cv2.resize(a.astype(f32) * cov, (w, h), interpolation=k)
    # a pixel is drawn by its last-touched subpixels: keep it slightly late so fronts do not bleed ahead
    tt = np.where(c > 1e-3, t / np.maximum(c, 1e-3), 2.)
    return d, we, tt.astype(f32)


_fib = {}


def _fibre(h, w, seed=3):
    """capillary map 0..1: how far ink creeps from here (anisotropic, hairy)"""
    S = 1400
    if seed not in _fib:
        r = np.random.default_rng(seed); f = np.zeros((S, S), f32)
        for _ in range(9000):
            x, y = r.uniform(0, S, 2); a = r.uniform(0, math.pi); L = r.uniform(5, 26)
            cv2.line(f, (int(x * 8), int(y * 8)), (int((x + math.cos(a) * L) * 8), int((y + math.sin(a) * L) * 8)), float(r.uniform(.3, 1)), 1, cv2.LINE_AA, 3)
        n = fbm(S, S, (9, 4, 2), (1, .7, .5), seed)
        _fib[seed] = np.clip(.38 * f + .30 * np.clip(n * .5 + .5, 0, 1) + .1, 0, 1).astype(f32)
    f = _fib[seed]; ry, rx = -(-h // S), -(-w // S)
    return np.tile(f, (ry, rx))[:h, :w]


def creep(d, wt, arr, R=4.0, seed=0, amt=.8, rate=.035):
    """capillary bleeding: wet edges feather outward along the fibres.  Returns (d, arr) with the halo added."""
    core = (d > .55) & (cv2.dilate(wt, np.ones((9, 9), np.uint8)) > .12)
    if not core.any(): return d, arr
    dist, lab = cv2.distanceTransformWithLabels((~core).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    iy, ix = np.nonzero(core)
    lut = np.zeros(int(lab.max()) + 1, f32); lut[lab[iy, ix]] = arr[iy, ix]
    w2 = cv2.dilate(wt, np.ones((15, 15), np.uint8))
    lw = np.zeros(int(lab.max()) + 1, f32); lw[lab[iy, ix]] = w2[iy, ix]
    h, w = d.shape
    f = _fibre(h, w)
    f = np.roll(f, (seed * 131 % 997, seed * 57 % 991), (0, 1))
    reach = R * lw[lab] * (.25 + 1.75 * f * f + .35 * f)
    a = np.clip(reach - dist + .6, 0, 1) * np.clip(1.05 - dist / (R * 2.6 + 1), 0, 1) * amt
    a = cv2.GaussianBlur(a.astype(f32), (0, 0), .6)
    nd = np.maximum(d, a)
    at = lut[lab] + .05 + dist * rate
    return nd, np.where((a > d + .02), np.maximum(at, np.where(arr > 1.9, 0, arr) * 0 + at), arr).astype(f32)


def drag(keys, corners=(), seed=0, load=1.0, dry=.18, deplete=.75, split=.55, speed=1.0, tilt=math.radians(50), follow=.55,
         lag=.75, elong=1.36, flat=.84, gain=.55, grip=.26, tufts=None, vmin=.22, t0=0., t1=1., bleed=2.6, fork=.0, forkin=.0, ss=SS, h=H, w=W):
    """simulate one stroke.  keys: (x, y, width px[, speed factor]) at 1x.
    load = how far the ink lasts (in brush widths x 9), dry = dryness at the start, deplete = extra dryness at the end,
    split = how far tufts part when dry, fork = the brush lifts still splayed (dry, torn exit), forkin = same at the entry,
    tilt = angle of the belly (radians), follow = how much the footprint turns with the motion, bleed = capillary feather px."""
    K = np.asarray(keys, float)
    sk = skeleton(K, corners, step=.75 / ss, vmin=vmin)
    wm = float(K[:, 2].max()); m = wm * 1.1 + 10
    x0, y0 = math.floor(K[:, 0].min() - m), math.floor(K[:, 1].min() - m)
    x1, y1 = math.ceil(K[:, 0].max() + m), math.ceil(K[:, 1].max() + m)
    x0, y0 = max(x0, -40), max(y0, -40); x1, y1 = min(x1, w + 40), min(y1, h + 40)
    if x1 - x0 < 4 or y1 - y0 < 4: return None
    rw, rh = (x1 - x0) * ss, (y1 - y0) * ss
    ink = np.zeros((rh, rw), f32); wet = np.zeros((rh, rw), f32); arr = np.full((rh, rw), 2., f32)
    n = len(sk['x'])
    ph = np.empty(n); hd = sk['hd']; ds = sk['s'][1] - sk['s'][0]
    cur = tilt; lam = max(wm * lag, 2.)
    for j in range(n):
        tgt = tilt + follow * ((hd[j] - tilt + math.pi) % (2 * math.pi) - math.pi)
        dlt = (tgt - cur + math.pi) % (2 * math.pi) - math.pi
        cur += dlt * min(1., ds / lam * (1.0 if j else 0)); ph[j] = cur
    dist = np.cumsum(ds * (sk['w'] / wm) ** .5)
    reach = load * 9 * wm
    v = np.clip(sk['v'] * speed, 0, 1.6)
    vs = np.convolve(np.pad(v, 60, 'edge'), np.hanning(121) / np.hanning(121).sum(), 'valid')[:n] if n > 130 else v
    DRY = dry * (.15 + .85 * np.clip(vs, 0, 1)) + deplete * np.clip(dist / reach, 0, 1.6) ** 1.35 * (.30 + .70 * np.clip(vs, 0, 1.3))
    # when the pressure lets go the ink film breaks first
    rel = np.clip(1 - sk['w'] / wm, 0, 1); pos = sk['s'] / sk['L']
    ex = smooth(.55, .95, pos) * smooth(.15, .8, rel); en = smooth(.30, .04, pos) * smooth(.15, .8, rel)
    FK = np.clip(fork * ex + forkin * en, 0, 1)
    DRY = np.maximum(0, 1.15 * DRY + FK * .35 + sk['xd'])
    SPL = np.clip(split * (DRY * 1.25) + .10 * split * np.clip(vs, 0, 1) + FK * .5, 0, .97)
    H_ = brush_hairs(wm * ss, seed, tufts)
    _drag(ink, wet, arr, tooth(2304, 2304), (sk['x'] - x0) * ss, (sk['y'] - y0) * ss, sk['w'] * ss * flat, ph, sk['t'], v, DRY, SPL, sk['s'] * ss, FK,
          H_['A'], H_['B'], H_['TB'], H_['TR'], H_['TL'], H_['TID'], H_['FID'], H_['HW'], int(seed) * 13 + 5, elong / flat, 1.05, gain, int(seed) * 37 % 1500, grip, wm * ss * flat)
    # surface tension: wet ink closes its own pinholes and pulls its edge tight
    q = ss / 2.
    cov = 1 - np.exp(-ink * .46)
    wb = cv2.GaussianBlur(wet, (0, 0), 6.0 * q)
    soak = np.clip(1 - np.exp(-wb * .34), 0, 1) ** 1.3
    cl = cv2.GaussianBlur(cov, (0, 0), 3.2 * q); cl = smooth(.33, .52, cl)
    cov = cov * (1 - soak) + np.maximum(cl, cov * .0) * soak
    cov = cv2.GaussianBlur(cov, (0, 0), .40 * q); cov = np.clip((cov - .05) * 1.10, 0, 1)
    arr = np.where((arr > 1.5) & (cov > .02), cv2.erode(arr, np.ones((9, 9), np.uint8)), arr)
    wt = np.clip(1 - np.exp(-wet * .14), 0, 1) * cov
    d, we, tt = _down(cov.astype(f32), wt.astype(f32), arr.astype(f32), ss)
    if bleed > 0: d, tt = creep(d, we, tt, bleed, int(seed))
    pool = cv2.resize(1 - np.exp(-ink * .045), (d.shape[1], d.shape[0]), interpolation=cv2.INTER_AREA)
    return Mark(x0, y0, d.astype(f32), np.maximum(we, 0).astype(f32), tt, t0, t1, 'stroke', pool.astype(f32),
                dict(x=sk['x'], y=sk['y'], w=sk['w'], t=sk['t'], s=sk['s'], hd=sk['hd'], v=v))


# --------------------------------------------------------------- the page
class Page:
    """A sheet being written on.  Elements are added in order with times (any unit); ink(t) gives the state at t."""

    def __init__(self, h=H, w=W, seed=0, ss=SS):
        self.h, self.w, self.seed, self.ss = h, w, seed, ss; self.marks = []

    def stroke(self, keys, corners=(), **k):
        k.setdefault('seed', self.seed * 101 + len(self.marks) * 17 + 3)
        mk = drag(keys, corners, ss=self.ss, h=self.h, w=self.w, **k)
        if mk is not None: self.marks.append(mk)
        return mk

    def add(self, mk):
        if mk is not None: self.marks.append(mk)
        return mk

    def ink(self, t=1e9, only=None, skip=(), full=False):
        """(thickness, wetness) at time t"""
        D = np.zeros((self.h, self.w), f32); Wt = np.zeros((self.h, self.w), f32); Po = np.zeros((self.h, self.w), f32)
        for i, m in enumerate(self.marks):
            if only is not None and i not in only: continue
            if i in skip or t < m.t0: continue
            if m.kind == 'drip':
                m.info['draw'](D, Wt, t); continue
            u = (t - m.t0) / max(1e-6, m.t1 - m.t0)
            if u >= 1.6: a, we = m.d, m.wet
            else:
                g = np.clip((u + .012 - m.arr) / .02, 0, 1); a, we = m.d * g, m.wet * g
            h, w = a.shape
            x0, y0 = max(0, m.ox), max(0, m.oy); x1, y1 = min(self.w, m.ox + w), min(self.h, m.oy + h)
            if x1 <= x0 or y1 <= y0: continue
            s = (slice(y0 - m.oy, y1 - m.oy), slice(x0 - m.ox, x1 - m.ox)); q = (slice(y0, y1), slice(x0, x1))
            D[q] = 1 - (1 - D[q]) * (1 - a[s]); Wt[q] = np.maximum(Wt[q], we[s])
            if m.bleed is not None: Po[q] = 1 - (1 - Po[q]) * (1 - m.bleed[s] * (a[s] > .02) * .8)
        return (D, Wt, Po) if full else (D, Wt)


# --------------------------------------------------------------- wet ink, storage
def page_state(pg, t, tau=14., only=None, skip=()):
    """(ink, fresh, pool) at time t: like Page.ink, but `fresh` is wetness that dries with age (time constant tau, same unit as t)"""
    D = np.zeros((pg.h, pg.w), f32); Fr = np.zeros((pg.h, pg.w), f32); Po = np.zeros((pg.h, pg.w), f32)
    for i, m in enumerate(pg.marks):
        if only is not None and i not in only: continue
        if i in skip or t < m.t0: continue
        if m.kind == 'drip':
            m.info['draw'](D, Fr, t); continue
        dt = max(1e-6, m.t1 - m.t0); u = (t - m.t0) / dt
        ar = m.arr.astype(f32); g = 1. if u >= 1.6 else np.clip((u + .012 - ar) / .02, 0, 1)
        a = m.d.astype(f32) * g; age = np.maximum(0, t - m.t0 - np.clip(ar, 0, 1) * dt); fr = a * np.exp(-age / tau) * (.45 + .55 * np.clip(m.wet.astype(f32) * 1.6, 0, 1))
        h, w = a.shape; x0, y0 = max(0, m.ox), max(0, m.oy); x1, y1 = min(pg.w, m.ox + w), min(pg.h, m.oy + h)
        if x1 <= x0 or y1 <= y0: continue
        s = (slice(y0 - m.oy, y1 - m.oy), slice(x0 - m.ox, x1 - m.ox)); q = (slice(y0, y1), slice(x0, x1))
        D[q] = 1 - (1 - D[q]) * (1 - a[s]); Fr[q] = np.maximum(Fr[q], fr[s])
        if m.bleed is not None and m.kind == 'stroke': Po[q] = 1 - (1 - Po[q]) * (1 - m.bleed[s] * (a[s] > .02) * .8)
    return D, Fr, Po


def wet_light(ink, fresh, pool, amt=1.0, phase=0., L=(-.46, -.62, .64), col='#e6edf4'):
    """light on standing ink (add to the picture): hair-fine glints along the bristle ridges and a soft window sliding over the black"""
    h, w = ink.shape; q = cv2.GaussianBlur(pool, (0, 0), .9) * 1.2 + cv2.GaussianBlur(ink * (.6 + .4 * fresh), (0, 0), 2.6)
    gx = cv2.Sobel(q, cv2.CV_32F, 1, 0, ksize=3) / 8; gy = cv2.Sobel(q, cv2.CV_32F, 0, 1, ksize=3) / 8
    n = np.dstack([-gx * 9, -gy * 9, np.ones_like(gx)]); n /= np.linalg.norm(n, axis=-1, keepdims=True)
    H_ = np.array(L) / np.linalg.norm(L) + np.array([0, 0, 1.]); H_ /= np.linalg.norm(H_); sp = np.clip(n @ H_.astype(f32), 0, 1) ** 46
    x, y = grid(h, w); u = (x * .42 + y * .92) / 240. + n[..., 0] * 1.6 + n[..., 1] * 1.6 + phase + .35 * cv2.resize(fbm(h // 4, w // 4, (60, 24), (1, .5), 77), (w, h))
    win = smooth(.55, .95, np.sin(u * 2.1)) * .5 + smooth(.8, 1., np.sin(u * .9 + 1.7)) * .5
    k = smooth(.55, .95, ink) * np.clip(fresh, 0, 1) * amt
    return ((sp * .62 + win * .060) * k)[..., None] * rgb(col)


def page_save(pg, path):
    import os
    d = {}; n = 0
    for m in pg.marks:
        if m.kind == 'drip': continue
        d[f'd{n}'] = m.d.astype(np.float16); d[f'w{n}'] = m.wet.astype(np.float16); d[f'a{n}'] = m.arr.astype(np.float16); d[f'p{n}'] = (m.bleed if m.bleed is not None else np.zeros((1, 1))).astype(np.float16)
        d[f'm{n}'] = np.array([m.ox, m.oy, m.t0, m.t1, 1 if m.kind == 'stroke' else 0], float)
        inf = m.info or {}; d[f'i{n}'] = np.stack([inf['x'], inf['y'], inf['w'], inf['t']]).astype(np.float32) if 'x' in inf else np.zeros((4, 1), np.float32); n += 1
    q = path + f'.{os.getpid()}.npz'; np.savez(q, n=n, **d); os.replace(q, path)


def page_load(path, h=H, w=W, seed=0):
    z = np.load(path); pg = Page(h, w, seed)
    for k in range(int(z['n'])):
        ox, oy, t0, t1, st = z[f'm{k}']; p = z[f'p{k}']; i = z[f'i{k}']
        pg.marks.append(Mark(int(ox), int(oy), z[f'd{k}'], z[f'w{k}'], z[f'a{k}'], float(t0), float(t1), 'stroke' if st else 'splash', p if p.size > 1 else None, dict(x=i[0], y=i[1], w=i[2], t=i[3]) if i.shape[1] > 1 else {}))
    return pg


# --------------------------------------------------------------- splashes
def _fill(buf, arr, P, val, t, ss):
    q = np.round(np.asarray(P) * ss * 16).astype(np.int32)
    cv2.fillPoly(buf, [q], val, cv2.LINE_AA, 4)
    if arr is not None: cv2.fillPoly(arr, [q], float(t), cv2.LINE_8, 4)


def blob(x, y, R, r, ang=0., st=1.0, spikes=0., m=26):
    """an ink drop that hit paper: wobbly, stretched along ang, thorned on the far side"""
    a = np.linspace(0, 2 * math.pi, m, endpoint=False)
    n = np.zeros(m)
    for k in (2, 3, 4, 5): n += r.normal(0, .055 / k ** .4) * np.cos(k * a + r.uniform(0, 6))
    rad = R * (1 + n)
    if spikes > 0:
        far = np.clip(np.cos(a), 0, 1)
        sp = (r.random(m) < .22 + .30 * far) * r.gamma(1.2, .30, m) * spikes * (.35 + far)
        rad = rad * (1 + sp)
    u = np.cos(a) * rad * st; v = np.sin(a) * rad / st ** .35
    # tail: pulled out along the throw
    u = np.where(u > 0, u * (1 + (st - 1) * .35), u)
    c, s = math.cos(ang), math.sin(ang)
    P = np.c_[x + u * c - v * s, y + u * s + v * c]
    return spl(P, True, n=3) if R > 1.6 else P


def splash(x, y, R, seed=0, ang=None, spread=math.pi, n=200, big=12., fingers=11, lobes=5, t0=0., t1=1., core=0., mist=1.0, h=H, w=W, ss=SS, reach=2.2, bleed=1.6, src=None):
    """ink thrown on the paper around (x, y): a blot whose rim breaks into webbed fingers, every finger shooting a chain of drops,
    plus loose drops and mist gathered in a few jets.  R = characteristic distance px, ang +- spread = direction (None all round),
    core = radius of the blot (0: no blot), src = optional polyline the ink leaves from.  Near drops land first."""
    r = np.random.default_rng(seed)
    M = R * reach + core * 2 + 60
    x0, y0, x1, y1 = [int(v) for v in (max(0, x - M), max(0, y - M), min(w, x + M), min(h, y + M))]
    if x1 - x0 < 4 or y1 - y0 < 4: return None
    buf = np.zeros(((y1 - y0) * ss, (x1 - x0) * ss), f32); arr = np.full_like(buf, 2.)
    O = np.array([x - x0, y - y0], float)
    SR = None if src is None else np.asarray(src, float) - [x0, y0]
    org = lambda: O if SR is None else (lambda q: SR[int(q)] * (1 - q % 1) + SR[min(int(q) + 1, len(SR) - 1)] * (q % 1))(r.uniform(0, len(SR) - 1.001))
    dirn = lambda: r.uniform(0, 2 * math.pi) if ang is None else ang + spread * float(np.clip(r.normal(0, .5), -1, 1))
    tmax = R * reach
    def drop(p, rad, th, st=1., ta=None, sp=None):
        _fill(buf, arr, blob(p[0], p[1], rad, r, th, st, (.9 if rad > 3.6 else 0.) if sp is None else sp, 30 if rad > 4 else 12), 1., min(1, np.hypot(*(p - O)) / tmax) if ta is None else ta, ss)
    # ---- fingers
    F = [dict(th=dirn(), L=R * r.gamma(2.2, .26), w=r.uniform(.010, .032) * (1 + .9 * (r.random() < .18)), o=org()) for _ in range(fingers)]
    if core > 0:
        a = np.linspace(0, 2 * math.pi, 900, endpoint=False); rad = core * (1 + .07 * noise1(900, 90, r) + .03 * noise1(900, 22, r))
        for q in F:
            dl = np.abs((a - q['th'] + math.pi) % (2 * math.pi) - math.pi)
            hgt = min(q['L'], core * 2.4) * r.uniform(.35, 1.0); u = dl / q['w']
            rad = np.maximum(rad, core * .80 + hgt * np.exp(-u ** .72))
        _fill(buf, arr, np.c_[O[0] + np.cos(a) * rad, O[1] + np.sin(a) * rad], 1., .02, ss)
    for q in F:
        th = q['th']; d0 = core * .5; L = q['L']; bow = r.normal(0, .06)
        P = lambda u: q['o'] + np.array([math.cos(th + bow * u), math.sin(th + bow * u)]) * (d0 + L * u)
        wd = max(1.4, big * r.uniform(.25, .75))
        # the string, torn into long pieces
        u = r.uniform(.0, .10)
        while u < .78:
            ln = r.uniform(.05, .26) * (1.1 - u); e = min(.95, u + ln); t = np.linspace(u, e, 14)
            C = np.array([P(v) for v in t]); N = np.array([-math.sin(th), math.cos(th)])
            hw = wd * (1 - t) ** 1.1 * (.35 + .65 * np.sin(np.linspace(.25, math.pi, 14)) ** .7) * r.uniform(.5, 1.1) * .5 + .2
            hw = hw * (1 + .25 * noise1(14, 3, r))
            _fill(buf, arr, np.vstack([C + N * hw[:, None], (C - N * hw[:, None])[::-1]]), 1., min(1, (d0 + L * u) / tmax), ss)
            u = e + r.uniform(.01, .10)
        u = r.uniform(.5, .8)
        for j in range(r.integers(2, 7)):
            drop(P(u) + r.normal(0, wd * .25, 2), wd * r.uniform(.20, .55) * (1 - .25 * min(u, 2)) + .5, th, 1 + r.uniform(0, 1.2)); u += r.uniform(.06, .30)
    # ---- loose drops in jets
    th0 = np.array([dirn() for _ in range(lobes)]); lw = r.dirichlet(np.ones(lobes) * 1.2); lr = r.uniform(.5, 1.3, lobes); ld = r.uniform(.07, .26, lobes)
    def throw(k):
        if r.random() < .22: th = dirn(); d = R * abs(r.normal(0, .6)) + core
        else: th = th0[k] + r.normal(0, ld[k]); d = R * lr[k] * r.beta(1.2, 2.2) * reach + core * .7
        return th, d
    for k in r.choice(lobes, n, p=lw):
        th, d = throw(k); p = org() + d * np.array([math.cos(th), math.sin(th)])
        rad = (big * r.beta(1.1, 4.2) + .9) if r.random() < .13 else (.5 + 2.2 * r.random() ** 2.6)
        rad *= 1.25 - .55 * min(1, d / tmax)
        st = 1 + r.uniform(0, 1.2) * min(1.3, d / R) * (.4 + .6 * (rad < 3))
        drop(p, rad, th + r.normal(0, .15), st)
        if rad > 3.4:
            for _ in range(r.integers(2, 9)):
                dd = rad * (1.4 + r.gamma(1.4, 1.1)); tt = th + r.normal(0, .5)
                drop(p + dd * np.array([math.cos(tt), math.sin(tt)]), r.uniform(.4, 1.5), tt, r.uniform(1, 2.2), min(1, d / tmax), 0)
    for k in r.choice(lobes, int(n * 2.6 * mist), p=lw):
        th, d = throw(k); d *= r.uniform(.3, 1.0); p = org() + d * np.array([math.cos(th), math.sin(th)])
        _fill(buf, arr, blob(p[0], p[1], r.uniform(.25, .72), r, th, 1., 0, 7), float(r.uniform(.5, 1)), min(1, .05 + d / tmax), ss)
    buf = np.clip(buf, 0, 1)
    ar = np.where((arr > 1.5) & (buf > .02), cv2.erode(arr, np.ones((5, 5), np.uint8)), arr)
    d, we, tt = _down(buf, buf * .8, ar, ss)
    if bleed > 0: d, tt = creep(d, we, tt, bleed, seed, amt=.75)
    return Mark(x0, y0, d, we, tt, t0, t1, 'splash', np.clip(cv2.GaussianBlur(d, (0, 0), 1.5) * .7, 0, 1))


def fling(x, y, ang, R, seed=0, **k):
    """a flick of the brush: one narrow jet"""
    k.setdefault('spread', .30); k.setdefault('lobes', 3); k.setdefault('n', 150); k.setdefault('fingers', 7)
    return splash(x, y, R, seed, ang=ang, **k)


def drip(x, y, L, wd, t0=0., t1=1., seed=0, wander=.35, h=H, w=W, stick=.5):
    """ink running down a vertical sheet: narrow trail, heavy rounded head, stick-slip descent"""
    r = np.random.default_rng(seed); n = max(12, int(L / 2))
    s = np.linspace(0, 1, n)
    px = x + wander * wd * (noise1(n, max(6, n / 5), r) * .9 + .25 * noise1(n, 7, r)) * np.clip(s * 6, 0, 1); py = y + L * s
    tw = .42 + .10 * noise1(n, max(4, n / 7), r); tw = tw * (1 + .9 * np.exp(-s * 10))
    kn = np.sort(r.random(5)); kv = np.sort(r.random(5))
    prog = PchipInterpolator(np.r_[0, kn, 1], np.r_[0, kn * (1 - stick) + kv * stick, 1])
    def draw(D, Wt, t):
        u = min(1., (t - t0) / max(1e-6, t1 - t0)); u = float(prog(1 - (1 - u) ** 1.7))
        k = int(u * (n - 1)) + 1
        if k < 3: return
        hw = np.maximum(.5, wd * .5 * tw[:k]); d = np.arange(k)[::-1] * (L / n)
        hw = hw + wd * .34 * np.exp(-(d / (wd * .95)) ** 2)
        hw = hw * np.sqrt(np.clip((d + .6) / (wd * .55), 0, 1))
        P = np.c_[px[:k], py[:k]]; poly = np.vstack([P - np.c_[hw, 0 * hw], (P + np.c_[hw, 0 * hw])[::-1]])
        bx0, by0 = max(0, int(poly[:, 0].min()) - 3), max(0, int(poly[:, 1].min()) - 3); bx1, by1 = min(w, int(poly[:, 0].max()) + 4), min(h, int(poly[:, 1].max()) + 4)
        if bx1 - bx0 < 2 or by1 - by0 < 2: return
        b = np.zeros(((by1 - by0) * 3, (bx1 - bx0) * 3), f32)
        cv2.fillPoly(b, [np.round((poly - [bx0, by0]) * 48).astype(np.int32)], 1., cv2.LINE_AA, 4)
        b = cv2.resize(b, (bx1 - bx0, by1 - by0), interpolation=cv2.INTER_AREA)
        q = (slice(by0, by1), slice(bx0, bx1)); D[q] = 1 - (1 - D[q]) * (1 - b); Wt[q] = np.maximum(Wt[q], b * .9)
    return Mark(0, 0, None, None, None, t0, t1, 'drip', None, dict(draw=draw))


class Spray:
    """ink in the air after a hit: drops and stretching ligaments thrown off the sheet toward the lens.
    sp = Spray(x, y, seed, power, ...);  layer, shadow = sp.layers(age_frames)"""

    def __init__(self, x, y, seed=0, power=34., n=110, ang=None, spread=math.pi, lift=1.0, big=15., lig=14, R0=20., f=900., g=1.1, h=H, w=W):
        r = np.random.default_rng(seed); self.h, self.w, self.f, self.g = h, w, f, g
        th = r.uniform(0, 2 * math.pi, n) if ang is None else ang + spread * np.clip(r.normal(0, .5, n), -1, 1)
        sp = power * r.lognormal(-.35, .55, n)
        self.p0 = np.c_[x + np.cos(th) * R0 * r.random(n), y + np.sin(th) * R0 * r.random(n), np.zeros(n)]
        self.v = np.c_[np.cos(th) * sp, np.sin(th) * sp, lift * power * r.gamma(1.6, .55, n)]
        self.r = np.where(r.random(n) < .2, big * r.beta(1.4, 2.6, n) + 2, .8 + 3.2 * r.random(n) ** 2)
        self.sd = r.integers(1 << 30, size=n); self.n = n
        tl = r.uniform(0, 2 * math.pi, lig) if ang is None else ang + spread * np.clip(r.normal(0, .5, lig), -1, 1)
        self.L = [dict(th=t, v=power * r.uniform(.55, 1.35), w=big * r.uniform(.35, .9), c=r.normal(0, .25), ln=r.uniform(.5, 1.0), sd=int(r.integers(1 << 30)), z=power * r.uniform(.1, .8), ph=r.uniform(0, 6)) for t in tl]
        self.o = np.array([x, y], float); self.R0 = R0

    def layers(self, age, shutter=.5, dof=1.0):
        h, w, f = self.h, self.w, self.f
        cel = Cel(w, h); sh = Cel(w // 2, h // 2, ss=1); far = Cel(w // 2, h // 2, ss=1)
        C = np.array([w / 2, h / 2])
        drag_ = lambda a: (1 - math.exp(-.10 * a)) / .10
        for L in self.L:                                # ligaments: a sheet pulled into a string, pinching into beads
            d1 = self.R0 + L['v'] * drag_(age) * 1.0; d0 = max(self.R0 * .6, d1 - L['ln'] * (60 + L['v'] * 4.) * min(1, age / 2.5) + max(0, age - 4) * L['v'] * .35)
            if d1 - d0 < 6: continue
            t = np.linspace(0, 1, 40); rr = d0 + (d1 - d0) * t; th = L['th'] + L['c'] * (rr / 600.) * 1.2
            P = self.o + np.c_[np.cos(th), np.sin(th)] * rr[:, None]
            z = L['z'] * drag_(age) * t * 1.2; k = f / np.maximum(120, f - z); P = C + (P - C) * k[:, None]
            pinch = np.clip((age - 1.5) / 7., 0, .95)
            wd = L['w'] * ((1 - t) ** 1.2 * (1 - pinch * .6) + .16) * (1 + pinch * 1.3 * np.sin(t * 19 + L['ph'])) * k
            wd = np.maximum(wd, .6); wd += L['w'] * .9 * np.exp(-((t - .97) / .05) ** 2) * k
            T = np.gradient(P, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-6; N = np.c_[-T[:, 1], T[:, 0]] * wd[:, None] * .5
            poly = np.vstack([P + N, (P - N)[::-1]])
            cel.fill(spl(poly, True, n=2), BLACK); so = np.c_[z * .30, z * .42]; sh.fill(np.vstack([P + N + so, (P - N + so)[::-1]]) / 2, '#000000')
            hl = P[6:28] - N[6:28] * .38; cel.line(hl, max(.8, L['w'] * .12), '#d8dde4', taper=(.5, .5), alpha=.55, seed=L['sd'])
        v = self.v; a = drag_(age)
        pos = self.p0 + np.c_[v[:, 0] * a, v[:, 1] * a + .5 * self.g * age * age, v[:, 2] * a - .8 * self.g * age * age]
        for i in np.argsort(pos[:, 2]):
            z = pos[i, 2]
            if z < 0: continue
            k = f / max(60., f - z)
            if k > 9: continue
            p = C + (pos[i, :2] - C) * k; R = self.r[i] * k
            if not (-R * 2 < p[0] < w + R * 2 and -R * 2 < p[1] < h + R * 2): continue
            vel = (v[i, :2] + (p - C) / max(80., f - z) * v[i, 2] * .6) * math.exp(-.10 * age) * k * shutter
            sp = np.hypot(*vel); ang = math.atan2(vel[1], vel[0]); r = np.random.default_rng(int(self.sd[i]))
            st = 1 + min(3.2, sp / (2.4 * R + 3))
            b = blob(p[0], p[1], R / st ** .28, r, ang, st ** .75, 0, 26)
            q = pos[i, :2] + np.array([.30, .42]) * z
            if z < 500: sh.fill(blob(q[0] / 2, q[1] / 2, self.r[i] * .5, r, ang, st ** .5, 0, 14), '#000000', float(np.clip(1 - z / 500., 0, 1)))
            (far if k > 1.9 else cel).fill(b / 2 if k > 1.9 else b, BLACK, 1.0 if k < 3 else .86)
            if R > 4 and k <= 1.9:
                c, s_ = math.cos(-2.2), math.sin(-2.2); cel.fill(blob(p[0] + c * R * .42, p[1] + s_ * R * .42, R * .16, r, ang, 1.6, 0, 12), '#e7ebf0', .8)
        lay_ = cel.layer(h, w)
        fa = cv2.resize(far.rgba(), (w, h), interpolation=cv2.INTER_LINEAR)
        if dof > 0 and fa[..., 3].max() > .01: fa = cv2.GaussianBlur(fa, (0, 0), 7 * dof)
        out = lay_ * (1 - fa[..., 3:4]) + fa
        sm = cv2.resize(sh.rgba()[..., 3], (w, h), interpolation=cv2.INTER_LINEAR); sm = cv2.GaussianBlur(sm, (0, 0), 6)
        return out, sm


# --------------------------------------------------------------- washes, edges
def wash(mask, seed=0, body=.35, edge=.55, ew=3.0, grain=.35, bloom=.25):
    """watered ink in a mask (0..1): uneven body, darker dried rim, granulation.  Returns ink thickness 0..1"""
    h, w = mask.shape
    m = (mask > .5).astype(np.uint8); d = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    n = fbm(h, w, (120, 46, 18), (1, .6, .35), seed); t = tooth(min(h, 2304), min(w, 2304))[:h, :w]
    o = body * (1 + .28 * n) + edge * np.exp(-d / ew) * (.75 + .25 * fbm(h, w, (60, 22), (1, .5), seed + 3))
    if bloom > 0:
        b = fbm(h, w, (90, 30, 12), (1, .7, .5), seed + 5); o = o + bloom * np.exp(-((b - .55) / .05) ** 2) * .6 - bloom * .5 * smooth(.55, 1.1, b)
    o = o * (1 + grain * (t - .5))
    return np.clip(o, 0, 1) * np.clip(mask, 0, 1)


def inkify(mask, amp=.45, seed=0, sig=1.2, warp=1.6):
    """roughen the edge of a drawn mask so it reads as brushed ink on toothed paper"""
    from .paint import warp_by
    h, w = mask.shape
    m = warp_by(mask.astype(f32), warp, 14, seed) if warp > 0 else mask.astype(f32)
    m = cv2.GaussianBlur(m, (0, 0), sig); t = tooth(min(h, 2304), min(w, 2304), 5); t = np.roll(t, (seed * 97, seed * 41), (0, 1))[:h, :w]
    return smooth(.5 - .18, .5 + .18, m + (.5 - t) * amp * (1 - np.abs(2 * m - 1)))


def lay(base, ink, wet=None, pool=None, col=BLACK, thin='#4a4037', sheen=0.):
    """ink on paper: a thin film is warm grey, a full film is black; pool (how many hairs passed) keeps the brush grain alive
    inside the black and gives standing ink a faint wet sheen from the upper left."""
    a = ink[..., None]; k = smooth(.15, .95, a); c = rgb(thin) * (1 - k) + rgb(col) * k
    if pool is not None:
        g = cv2.GaussianBlur(pool, (0, 0), .7)
        c = c * (1.2 - .5 * smooth(.10, .60, g))[..., None] + (.030 * (1 - smooth(.06, .50, g)) * k[..., 0])[..., None]
        if wet is not None and sheen > 0:
            hgt = cv2.GaussianBlur(pool * wet, (0, 0), 2.2) + .35 * cv2.GaussianBlur(ink, (0, 0), 3.2)
            gx = cv2.Sobel(hgt, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(hgt, cv2.CV_32F, 0, 1, ksize=3)
            n = gx * .55 + gy * .84
            c = c + (np.clip(n - .02, 0, .5) * sheen * 3.0 * smooth(.15, .6, wet) * (ink > .9))[..., None] * np.array([.86, .93, 1.], f32)
    return base * (1 - a) + c * a


def burst(h, w, cx, cy, seed=0, n=150, rmax=1500, wd=26., r0=120.):
    """radial brush slashes from (cx, cy): ink alpha (dry streaks, not wedges).  Traced at half size and kept on disk."""
    import os
    from .manga import streaks
    x, y = grid(h, w); d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) + 1e-3
    c = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache', f'burst_{w}x{h}_{int(cx)}_{int(cy)}_{seed}.npy')
    if os.path.exists(c): m = np.load(c).astype(f32)
    else:
        hx, hy = grid(h // 2, w // 2); hx = hx * 2 - cx; hy = hy * 2 - cy
        m = cv2.resize(.55 * streaks(hx, hy, 46, 2.6, seed) + .45 * streaks(hx, hy, 60, 8., seed + 1), (w, h), interpolation=cv2.INTER_CUBIC)
        os.makedirs(os.path.dirname(c), exist_ok=True); q = c + f'.{os.getpid()}.npy'; np.save(q, m.astype(np.float16)); os.replace(q, c)
    return smooth(.46, .54, m + .22 * smooth(r0, rmax * .55, d) - .12), d


def impact_inv(ink, splat, cx, cy, seed=0, black='#080405', red='#c30f1c', deep='#4d0710', hot='#fff8ee', amt=1.0, scratch=True, pool=None):
    """the hit frame: black and blood, the written mark burning white.  ink = the kanji, splat = thrown ink (both 0..1)"""
    h, w = ink.shape; b, d = burst(h, w, cx, cy, seed)
    n = fbm(h, w, (90, 34, 13), (1, .7, .5), seed); t = np.tile(tooth(2304, 2304), (1, 1))[:h, :w]
    out = np.ones((h, w, 3), f32) * rgb(black)
    far = smooth(260, 1150, d + 60 * n)
    m1 = b * (1 - far * .55) * smooth(.15, .62, .55 + .30 * n - .75 * np.abs(smooth(0, 900, d) - .42) + (t - .5) * .5)
    out = out * (1 - m1[..., None]) + rgb(deep) * m1[..., None]
    m2 = smooth(.60, .70, b * .55 + .26 * n + (t - .5) * .55 + .30 * np.exp(-((d - 420) / 260.) ** 2)); out = out * (1 - m2[..., None]) + rgb(red) * m2[..., None]
    # the mark tears the paper: ragged halo, black first, then blood
    al = np.clip(ink, 0, 1); g1 = cv2.GaussianBlur(al, (0, 0), 30); g2 = cv2.GaussianBlur(al, (0, 0), 11); rr = (cv2.GaussianBlur(b, (0, 0), 1.2) - .5) * .20 + .13 * n + (t - .5) * .10
    k = smooth(.30, .36, g1 * 1.25 + rr * .6); out = out * (1 - k[..., None]) + rgb(black) * k[..., None]
    k = smooth(.20, .27, g2 + rr * .55); out = out * (1 - k[..., None]) + rgb(red) * k[..., None]
    k = smooth(.5, .6, g2 * 1.05 + rr * .2); out = out * (1 - k[..., None]) + rgb(deep) * k[..., None]
    if scratch:
        r = np.random.default_rng(seed + 9); c = Cel(w, h)
        for _ in range(120):
            th = r.uniform(0, 2 * math.pi); q0 = r.uniform(360, 900); L = r.gamma(1.5, 110); cc, ss_ = math.cos(th), math.sin(th)
            c.line(np.array([(cx + cc * q0 * 1.15, cy + ss_ * q0 * .85), (cx + cc * (q0 + L) * 1.15, cy + ss_ * (q0 + L) * .85)]), r.uniform(.8, 3.2), hot if r.random() < .35 else red, taper=(.4, .6), seed=int(r.integers(1 << 30)), alpha=.9)
        l = c.rgba(); out = out * (1 - l[..., 3:4]) + l[..., :3]
    sp = np.clip(splat, 0, 1); kk = smooth(.10, .5, cv2.GaussianBlur(sp, (0, 0), 5))[..., None]; out = out * (1 - kk * .9) + rgb(black) * kk * .9
    sp = sp[..., None]; out = out * (1 - sp) + rgb('#ff2a24') * sp
    si = smooth(.42, .72, cv2.GaussianBlur(np.clip(splat, 0, 1), (0, 0), 2.4))[..., None]; out = out * (1 - si) + rgb(hot) * si
    a = smooth(.34, .86, al)[..., None]; core = smooth(.50, .98, cv2.GaussianBlur(smooth(.2, .9, al), (0, 0), 3.2))[..., None]
    if pool is not None: core = core * (.62 + .38 * smooth(.06, .34, cv2.GaussianBlur(pool, (0, 0), 1.0)))[..., None]
    out = out * (1 - a) + (rgb('#ff4a30') * (1 - core) + rgb(hot) * core * 1.08) * a
    return out
