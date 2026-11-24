"""fx core: deterministic noise, path frames, lick-edged bands, light, layers.  (effects animator)"""
import numpy as np, cv2, math
from collections import namedtuple
from numba import njit, prange
from . import W, H
from .paint import rgb, f32, grid, fbm, smooth
from .ink import Cel, spl, resample, ribbon, noise1, over_roi, paste

FX = namedtuple('FX', 'rgba light')                    # premultiplied RGBA (h,w,4) or None, additive RGB (h,w,3) or None


def put(base, fx, op=1.0, light=1.0):
    """composite an FX (or a bare RGBA layer) over an RGB frame"""
    rgba, li = (fx.rgba, fx.light) if isinstance(fx, FX) else (fx, None)
    if rgba is not None: base = base * (1 - rgba[..., 3:4] * op) + rgba[..., :3] * op
    if li is not None and light: base = base + li * (light * op)
    return base


def merge(*fx):
    """stack several FX (first = bottom) into one"""
    R = L = None
    for q in fx:
        if q is None: continue
        rgba, li = (q.rgba, q.light) if isinstance(q, FX) else (q, None)
        if rgba is not None:
            if R is not None: R = R * (1 - rgba[..., 3:4]) + rgba
            else: R = rgba
            if L is not None: L = L * (1 - rgba[..., 3:4] * .6)
        if li is not None: L = li if L is None else L + li
    return FX(R, L)


def C(c):
    return rgb(c) if isinstance(c, str) else np.asarray(c, f32)


def twos(f, k=2):
    return int(f) // k


# ------------------------------------------------------------- hashes / noise
def h1(*k):
    """hash of integers -> float 0..1 (stable across runs)"""
    n = 2166136261
    for q in k: n = ((n ^ (int(q) & 0xffffffff)) * 16777619) & 0xffffffff
    n ^= n >> 15; n = (n * 2246822519) & 0xffffffff; n ^= n >> 13; n = (n * 3266489917) & 0xffffffff; n ^= n >> 16
    return (n & 0xffffff) / 16777216.


def R(*k):
    """generator seeded by a tuple of integers"""
    return np.random.default_rng([int(q) & 0xffffffff for q in k])


def bms(u, big=.18, med=.34):
    """big / medium / small size law from a uniform number: 18 % big (1), 34 % medium (.55), rest small (.25), jittered"""
    if u < big: return .85 + .3 * (u / big)
    if u < big + med: return .45 + .25 * (u - big) / med
    return .16 + .2 * (u - big - med) / (1 - big - med)


@njit(cache=True, fastmath=True)
def _hash3(x, y, z):
    n = (x * 374761393 + y * 668265263 + z * 1440662683) & 0xffffffff
    n = ((n ^ (n >> 13)) * 1274126177) & 0xffffffff
    return ((n ^ (n >> 16)) & 0xffff) / 32767.5 - 1.0


@njit(cache=True, fastmath=True)
def _vn3(x, y, z):
    ix = math.floor(x); iy = math.floor(y); iz = math.floor(z)
    fx = x - ix; fy = y - iy; fz = z - iz
    fx = fx * fx * fx * (fx * (fx * 6 - 15) + 10); fy = fy * fy * fy * (fy * (fy * 6 - 15) + 10); fz = fz * fz * (3 - 2 * fz)
    ix = int(ix); iy = int(iy); iz = int(iz)
    s = 0.0
    for c in range(2):
        a = _hash3(ix, iy, iz + c); b = _hash3(ix + 1, iy, iz + c); d = _hash3(ix, iy + 1, iz + c); e = _hash3(ix + 1, iy + 1, iz + c)
        q = (a + (b - a) * fx) * (1 - fy) + (d + (e - d) * fx) * fy
        s += q * (fz if c else 1 - fz)
    return s


@njit(cache=True, fastmath=True)
def _vn3a(x, y, z, out):
    for i in range(x.size): out[i] = _vn3(x[i], y[i], z[i])


@njit(cache=True, fastmath=True)
def _vn3x(x, y, z, out):
    for i in range(x.size): out[i] = _vn3(x[i], y, z)


@njit(cache=True, fastmath=True)
def _vn3xy(x, y, z, out):
    for i in range(x.size): out[i] = _vn3(x[i], y[i], z)


def c2(x, y):
    """stack two 1-D arrays into (n, 2)  (np.c_ is slow)"""
    x = np.asarray(x, float); o = np.empty((x.shape[0], 2)); o[:, 0] = x; o[:, 1] = y; return o


_LIN = {}


def lin01(n):
    v = _LIN.get(n)
    if v is None: v = _LIN[n] = np.linspace(0, 1, n)
    return v


def noise(x, y=0., z=0., seed=0):
    """smooth 3-D value noise (-1..1) at points; z is usually time"""
    zz = np.asarray(z) + seed * 37.17; x = np.asarray(x, np.float64)
    if x.ndim == 1 and np.ndim(zz) == 0:
        o = np.empty(x.size)
        if np.ndim(y) == 0: _vn3x(np.ascontiguousarray(x), float(y), float(zz), o); return o
        y = np.asarray(y, np.float64)
        if y.shape == x.shape: _vn3xy(np.ascontiguousarray(x), np.ascontiguousarray(y), float(zz), o); return o
    if x.ndim == 0 and np.ndim(y) == 0 and np.ndim(zz) == 0: return _vn3(float(x), float(y), float(zz))
    x, y, zz = np.broadcast_arrays(x, np.asarray(y, np.float64), np.asarray(zz, np.float64)); o = np.empty(x.size); _vn3a(np.array(x).ravel(), np.array(y).ravel(), np.array(zz).ravel(), o)
    return o.reshape(x.shape)


@njit(cache=True, fastmath=True, parallel=False)
def _vimg(h, w, sx, sy, z, ox, oy, octs, out):
    for j in range(h):
        for i in range(w):
            v = 0.0; a = 1.0; px = (i * sx + ox) * .866 - (j * sy + oy) * .5; py = (i * sx + ox) * .5 + (j * sy + oy) * .866
            for o in range(octs):
                v += a * _vn3(px, py, z + o * 17.3); a *= .5; qx = (px * .707 - py * .707) * 2.03 + 3.1; py = (px * .707 + py * .707) * 2.03 + 1.7; px = qx
            out[j, i] = v


def noise_img(h, w, scale, z=0., seed=0, octs=3, q=4, aniso=1.0, ox=0., oy=0.):
    """fbm picture evaluated at 1/q resolution; scale = feature size px; aniso stretches along x"""
    hh, ww = h // q + 2, w // q + 2; o = np.empty((hh, ww), f32)
    _vimg(hh, ww, q / (scale * aniso), q / scale, float(z) + seed * 37.17, ox / (scale * aniso), oy / scale, octs, o)
    return cv2.resize(o, ((ww) * q, (hh) * q), interpolation=cv2.INTER_LINEAR)[:h, :w]


# ------------------------------------------------------------- paths
_BZ = {}


def bez(p0, c1, c2_, p3, n=10):
    M = _BZ.get(n)
    if M is None: u = np.linspace(0, 1, n + 1)[1:]; M = _BZ[n] = np.stack([(1 - u) ** 3, 3 * (1 - u) ** 2 * u, 3 * (1 - u) * u * u, u ** 3], 1)
    return M @ np.array([p0, c1, c2_, p3], float)


class Path:
    """arc-length frame of a polyline.  at(s, n): point at arc length s (px, may run past the ends), offset n to the left-hand normal"""
    def __init__(self, P, step=5.0):
        P = np.asarray(P, float); self.P = P = resample(P, step)
        if len(P) < 3: P = self.P = np.linspace(P[0], P[-1] + 1e-3, 3)
        T = np.gradient(P, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-9
        self.T = T; self.N = np.c_[T[:, 1], -T[:, 0]]
        self.s = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]; self.L = float(self.s[-1])

    def at(self, s, n=0.):
        s = np.asarray(s, float)
        if s.ndim == 0: return self.at(s[None], n)[0]
        c = np.clip(s, 0, self.L); e = s - c; I = np.interp; q = self.s; o = np.empty((s.shape[0], 2)); nx, ny = I(c, q, self.N[:, 0]), I(c, q, self.N[:, 1]); k = n / (np.sqrt(nx * nx + ny * ny) + 1e-9)
        o[:, 0] = I(c, q, self.P[:, 0]) - ny * e + nx * k; o[:, 1] = I(c, q, self.P[:, 1]) + nx * e + ny * k; return o

    def sub(self, s0, s1, n=0., k=24):
        """offset sub-path as polyline; n = scalar or callable(v 0..1)"""
        v = lin01(k); return self.at(s0 + (s1 - s0) * v, n(v) if callable(n) else n)


def teeth(L, pitch, seed, shift=0., amp=1.0, ln=1.5, t=0., pulse=.25):
    """travelling licks on an edge of length L -> list of (s0, s1, a 0..1).  shift px moves them along (stream)"""
    out = []; k0 = int(math.floor((-shift - 2 * pitch) / pitch)); k1 = int(math.ceil((L - shift + 2 * pitch) / pitch))
    for k in range(k0, k1 + 1):
        u = h1(seed, k, 1); z = bms(u)
        s1 = (k + .15 + .7 * h1(seed, k, 2)) * pitch + shift; ll = pitch * ln * (.45 + .9 * z) * (.75 + .5 * h1(seed, k, 3))
        a = amp * z * (1 + pulse * math.sin(6.283 * (h1(seed, k, 4) + t * (1.3 + h1(seed, k, 5)))))
        if s1 > 6 and s1 - ll < L: out.append((s1 - ll, s1, a, h1(seed, k, 6)))
    return out


def lick_edge(L, wf, tt, under=.34, step=14., s_end=None):
    """(s, n) polyline of one edge: half width wf(s) with swept licks tt [(s0, s1, a_px[, kind 0..1])]; kind < .5 chunky step, else needle"""
    pts = []; cur = 0.; E = L if s_end is None else s_end; K = max(5, int(E / step) + 2); g = lin01(K) * E; wg = np.asarray(wf(g), float); dg = E / (K - 1)
    def w(s):
        q = min(max(s, 0.), E) / dg; i = min(int(q), K - 2); f = q - i; return wg[i] * (1 - f) + wg[i + 1] * f
    def run(a, b):
        if b <= a: return
        k = max(2, int((b - a) / step) + 1); s = a + (b - a) * lin01(k); pts.append(c2(s, np.interp(s, g, wg)))
    for q in tt:
        s0, s1, a = q[:3]; kd = q[3] if len(q) > 3 else .7
        s0 = max(s0, cur); s1 = min(s1, E + a * .6); ln = s1 - s0
        if ln < 5 or a < .6: run(cur, min(s1, E)); cur = max(cur, min(s1, E)); continue
        run(cur, min(s0, E)); w0, w1 = w(s0), w(s1); tip = (s1, w1 + a); un = under * (.35 + 1.0 * kd); bk = s1 - ln * un; wb = w(bk); pts.append(np.array([[s0, w0]]))
        if kd < .5: pts.append(bez((s0, w0), (s0 + .45 * ln, w0 + .10 * a), (s1 - .42 * ln, w1 + a * 1.02), tip, 7)); pts.append(bez(tip, (s1 - .06 * ln, w1 + a * .40), (bk - .02 * ln, wb + a * .30), (bk + .02 * ln, wb - .04 * a), 5))
        else: pts.append(bez((s0, w0), (s0 + .5 * ln, w0 - .06 * a), (s1 - .30 * ln, w1 + a * .62), tip, 7)); pts.append(bez(tip, (s1 - .16 * ln, w1 + a * .52), (bk - .10 * ln, wb + a * .20), (bk + .02 * ln, wb - .04 * a), 6))
        cur = bk + .02 * ln
    run(cur, E)
    return np.concatenate(pts) if pts else np.zeros((0, 2))


def band(P, hw, seed=0, shift=0., pitch=70., amp=.8, ln=1.5, t=0., env=None, sides=(1, 1), wave=.10, under=.34, step=5., bias=0., px=0.):
    """closed lick-edged ribbon along P (flows from start to tip).
    hw = callable(v 0..1) -> half width px (v along path), or (left, right) callables; amp = lick height relative to half width,
    env(v) scales licks, bias shifts the whole band sideways (fraction of hw).  -> (k,2) closed polygon"""
    p = P if isinstance(P, Path) else Path(P, step); L = p.L
    hl, hr = hw if isinstance(hw, tuple) else (hw, hw)
    out = []
    for q, (h, sg) in enumerate(((hl, 1), (hr, -1))):
        if wave: wf = lambda s, h=h, q=q: np.maximum(h(np.clip(s / L, 0, 1)) * (1 + wave * noise(s / (pitch * 1.4) - shift / (pitch * 1.4), q * 9.1, t * .9, seed)), 0)
        else: wf = lambda s, h=h: np.maximum(h(np.clip(s / L, 0, 1)), 0)
        tt = []
        if sides[q] > 0 and (amp or px):
            T = teeth(L, pitch, seed * 13 + q * 7 + 1, shift, 1., ln, t)
            if T:
                v = np.clip(np.array([z[1] for z in T]) / L, 0, 1); e = np.ones(len(T)) if env is None else np.broadcast_to(np.asarray(env(v), float), v.shape); hh = np.broadcast_to(np.asarray(h(v), float), v.shape); tt = [(z[0], z[1], (z[2] * amp * hh[i] + z[2] * px) * e[i] * sides[q], z[3]) for i, z in enumerate(T)]
        E = lick_edge(L, wf, tt)
        if bias: E[:, 1] += bias * np.interp(E[:, 0], lin01(40) * L, wf(lin01(40) * L)) * sg
        out.append(p.at(E[:, 0], E[:, 1] * sg))
    return np.vstack([out[0], out[1][::-1]])


def leaf(v, a=.30, p0=.6, p1=1.0):
    """0..1 width profile: swells to 1 at v=a, sharp at both ends"""
    v = np.clip(v, 0, 1)
    return np.where(v < a, (v / a) ** p0, ((1 - v) / (1 - a)) ** p1) if a > 0 else (1 - v) ** p1


def tongue(v, belly=.28, need=1.35):
    """flame tongue profile: wide base, belly, needle tip"""
    v = np.clip(v, 0, 1)
    return (1 - v) ** (need * (.55 + .55 * v)) * (1 + .38 * np.exp(-((v - belly) / .20) ** 2)) / 1.2


def poly_len(P):
    return float(np.hypot(*np.diff(P, axis=0).T).sum())


def ink_broken(cel, P, w=1.4, col='#5a1408', seed=0, cover=.75, piece=(60, 220), taper=(.2, .35), alpha=1.0):
    """thin broken pen line along an edge (drops gaps, lifts): the interior ink of drawn fire"""
    P = resample(np.asarray(P, float), 3.0); n = len(P); r = R(seed, 77); i = int(r.integers(0, 12)); k = 0
    while i < n - 6:
        m = int(r.uniform(*piece) / 3)
        if r.random() < cover: cel.line(P[i:i + m], w * r.uniform(.75, 1.2), col, taper=taper, press=.3, seed=seed * 31 + k, tremor=.2, alpha=alpha)
        i += m + int(r.uniform(2, 12)); k += 1
    return cel


def thin(cel, P, w, col, alpha=1.0, a=.3, b=.5, seed=0, press=.0, K=28):
    """fast tapered stroke (no tremor): a / b = fraction of the length that tapers in / out"""
    P = np.asarray(P, float)
    if len(P) < 2: return cel
    if len(P) != K: d = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]; u = np.linspace(0, d[-1], K); P = c2(np.interp(u, d, P[:, 0]), np.interp(u, d, P[:, 1]))
    T = np.gradient(P, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; v = np.linspace(0, 1, K); hw = .5 * w * np.minimum(1., v / max(a, 1e-3)) ** .7 * np.minimum(1., (1 - v) / max(b, 1e-3)) ** .7
    if press: hw = hw * (1 + press * np.sin(v * (7 + 5 * h1(seed, 1)) + 6 * h1(seed, 2)))
    N = c2(-T[:, 1], T[:, 0]) * np.maximum(hw, .12)[:, None]; cel.fill(np.vstack([P + N, (P - N)[::-1]]), col, alpha); return cel


# ------------------------------------------------------------- light
def hot(layer, thr=.35, r=(5, 20, 64), wts=(.5, .35, .3), col=None, amt=.8, gain=1.0):
    """additive bloom of the bright part of a layer (computed at 1/4 size)"""
    h, w = layer.shape[:2]
    q = cv2.resize(layer[..., :3], (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    m = np.clip((q.max(-1) - thr) / (1 - thr), 0, 1)[..., None] * q * gain
    o = sum(wt * cv2.GaussianBlur(m, (0, 0), s / 4.) for s, wt in zip(r, wts))
    if col is not None: o = o * C(col)
    return cv2.resize(o.astype(f32), (w, h), interpolation=cv2.INTER_LINEAR) * amt


def glow_poly(polys, col, r=(8, 40), wts=(.6, .5), amt=1., h=H, w=W, q=4):
    """additive glow from polygons rendered small"""
    m = np.zeros((h // q, w // q), np.uint8)
    for P in (polys if isinstance(polys, list) else [polys]):
        if len(P) > 2: cv2.fillPoly(m, [np.round(np.asarray(P) / q * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
    m = m.astype(f32) / 255.
    o = sum(wt * cv2.GaussianBlur(m, (0, 0), s / q) for s, wt in zip(r, wts))
    return cv2.resize(o[..., None] * C(col) * amt, (w, h), interpolation=cv2.INTER_LINEAR)


def displace(img, dx, dy, border=cv2.BORDER_REFLECT_101):
    h, w = img.shape[:2]; x, y = grid(h, w)
    return cv2.remap(img, x + dx.astype(f32), y + dy.astype(f32), cv2.INTER_LINEAR, borderMode=border)


def soft_edge(layer, s):
    return cv2.GaussianBlur(layer, (0, 0), s)


def roi_of(polys, pad=12, w=W, h=H):
    """integer ROI (x0,y0,x1,y1) of polygons clipped to the frame, or None"""
    A = np.concatenate([np.asarray(p) for p in polys if len(p)]) if len(polys) else np.zeros((0, 2))
    if not len(A): return None
    x0, y0 = np.floor(A.min(0) - pad); x1, y1 = np.ceil(A.max(0) + pad)
    x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(w, x1)), int(min(h, y1))
    return None if x1 <= x0 or y1 <= y0 else (x0, y0, x1, y1)
