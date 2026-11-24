"""Brush and pen over the construction sculpt: edge walking, loose broken ink, hatching, dab-textured planes (used by chars/mahoraga.py)."""
import numpy as np, cv2, math, os
from numba import njit, prange
from jj.paint import Canvas, arcs, fbm, smooth, f32
from jj.ink import noise1

# ------------------------------------------------------------------ variable-width stroke rasteriser (max-blended coverage)
@njit(parallel=False, fastmath=True, cache=True)
def _ink(mask, P, HW, ST, AL):
    h, w = mask.shape
    for s in range(len(ST) - 1):
        al = AL[s]
        for k in range(ST[s], ST[s + 1] - 1):
            ax, ay, bx, by = P[k, 0], P[k, 1], P[k + 1, 0], P[k + 1, 1]; ra = HW[k]; rb = HW[k + 1]
            m = max(ra, rb) + 1.5
            x0 = max(0, int(min(ax, bx) - m)); x1 = min(w - 1, int(max(ax, bx) + m) + 1); y0 = max(0, int(min(ay, by) - m)); y1 = min(h - 1, int(max(ay, by) + m) + 1)
            dx = bx - ax; dy = by - ay; l2 = dx * dx + dy * dy + 1e-9
            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    t = ((x - ax) * dx + (y - ay) * dy) / l2
                    t = 0. if t < 0 else (1. if t > 1 else t)
                    qx = ax + dx * t - x; qy = ay + dy * t - y; r = ra + (rb - ra) * t
                    c = r - math.sqrt(qx * qx + qy * qy) + .5
                    if c > 0:
                        c = min(c, 1.) * al
                        if r < .5: c *= r * 2
                        if c > mask[y, x]: mask[y, x] = c


class Pen:
    """collects strokes, rasterises them in one go"""
    def __init__(self): self.P, self.H, self.A = [], [], []

    def add(self, P, hw, al=1.):
        P = np.asarray(P, f32)
        if len(P) < 2: return
        self.P.append(P); self.H.append(np.broadcast_to(np.asarray(hw, f32), (len(P),)).copy()); self.A.append(al)

    def stroke(self, P, w, seed=0, taper=(.22, .35), press=.35, tremor=.5, al=1., tmin=.08, step=3.):
        """pen line: resampled, tapered, pressure, slight tremor.  w = full width"""
        P = _res(P, step); n = len(P)
        if n < 3: return
        r = np.random.default_rng(abs(int(seed)) & 0x7fffffff); t = np.linspace(0, 1, n)
        wl = np.ones(n)
        if taper[0] > 0: wl *= tmin + (1 - tmin) * np.clip(t / taper[0], 0, 1) ** .7
        if taper[1] > 0: wl *= tmin + (1 - tmin) * np.clip((1 - t) / taper[1], 0, 1) ** .8
        wl *= 1 + press * np.clip(noise1(n, 14, r), -1.5, 1.5) * .6
        T = np.gradient(P, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]
        self.add(P + N * (noise1(n, 11, r) * tremor)[:, None], np.maximum(.25, .5 * w * wl), al)

    def raster(self, h, w):
        m = np.zeros((h, w), f32)
        if self.P:
            ST = np.r_[0, np.cumsum([len(p) for p in self.P])].astype(np.int64)
            _ink(m, np.concatenate(self.P), np.concatenate(self.H), ST, np.asarray(self.A, f32))
        return m


def _res(P, step=3.):
    P = np.asarray(P, float)
    d = np.hypot(*np.diff(P, axis=0).T); keep = np.r_[True, d > 1e-6]; P = P[keep]
    if len(P) < 2: return P
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]; n = max(3, int(s[-1] / step) + 1); t = np.linspace(0, s[-1], n)
    return np.c_[np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])]


# ------------------------------------------------------------------ skeleton walking
@njit(cache=True)
def _walk(sk, maxp):
    h, w = sk.shape; vis = np.zeros((h, w), np.uint8); out = np.empty((maxp, 2), np.int32); st = np.empty(maxp // 4 + 2, np.int64); n = 0; ns = 0
    nb = np.zeros((h, w), np.uint8)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if sk[y, x]:
                c = 0
                for j in range(-1, 2):
                    for i in range(-1, 2):
                        if (i or j) and sk[y + j, x + i]: c += 1
                nb[y, x] = c
    for ps in range(2):
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if not sk[y, x] or vis[y, x]: continue
                if ps == 0 and nb[y, x] != 1: continue
                st[ns] = n; ns += 1
                cx = x; cy = y; dx = 0.; dy = 0.
                while True:
                    if n >= maxp - 1 or ns >= len(st) - 2: st[ns] = n; return out[:n], st[:ns + 1]
                    vis[cy, cx] = 1; out[n, 0] = cx; out[n, 1] = cy; n += 1
                    best = -9.; bx = -1; by = -1
                    for j in range(-1, 2):
                        for i in range(-1, 2):
                            if i == 0 and j == 0: continue
                            xx = cx + i; yy = cy + j
                            if xx < 1 or yy < 1 or xx >= w - 1 or yy >= h - 1: continue
                            if sk[yy, xx] and not vis[yy, xx]:
                                l = math.sqrt(i * i + j * j); sc = (i * dx + j * dy) / l if (dx or dy) else 0.
                                if l < 1.1: sc += .15
                                if sc > best: best = sc; bx = xx; by = yy
                    if bx < 0 or best < -.75: break
                    dx = dx * .75 + (bx - cx) * .25; dy = dy * .75 + (by - cy) * .25
                    cx = bx; cy = by
    st[ns] = n
    return out[:n], st[:ns + 1]


def trace(mask, min_len=8):
    """binary edge mask -> list of float polylines (x,y) after thinning"""
    m = (mask > 0).astype(np.uint8) * 255
    sk = cv2.ximgproc.thinning(m, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    pts, st = _walk((sk > 0).astype(np.uint8), int((sk > 0).sum()) + 8)
    return [pts[a:b].astype(float) for a, b in zip(st[:-1], st[1:]) if b - a >= min_len]


def confident(P, eps=1.4, sig=2.0):
    """smooth a pixel chain and simplify it the way a hand would: straights against curves.  -> list of strokes split at corners"""
    n = len(P)
    if n < 5: return [P]
    k = int(sig * 3) | 1; pad = k // 2; g = np.exp(-.5 * ((np.arange(k) - pad) / sig) ** 2); g /= g.sum()
    Q = np.vstack([2 * P[0] - P[1:pad + 1][::-1], P, 2 * P[-1] - P[-pad - 1:-1][::-1]]) if n > pad + 1 else np.pad(P, ((pad, pad), (0, 0)), 'edge')
    S = np.stack([np.convolve(Q[:, i], g, 'valid') for i in (0, 1)], -1)
    A = cv2.approxPolyDP(S.astype(np.float32)[:, None], eps, False)[:, 0].astype(float)
    if len(A) < 3: return [A]
    d = np.diff(A, axis=0); a = np.arctan2(d[:, 1], d[:, 0]); tn = np.abs((np.diff(a) + math.pi) % (2 * math.pi) - math.pi)
    cut = [0] + [i + 1 for i in range(len(tn)) if tn[i] > 1.05] + [len(A) - 1]
    return [A[i:j + 1] for i, j in zip(cut[:-1], cut[1:]) if j > i]


def curve(A, step=3.):
    """Catmull-Rom through sparse points, resampled"""
    from jj.ink import spl
    if len(A) < 3: return _res(A, step)
    return _res(spl(A, n=8), step)


# ------------------------------------------------------------------ brush-dab texture (zero mean, unit range), one hand direction
_DAB = {}
def dabs(h, w, size, seed=0, ang=-.95, spread=.35, aspect=3.2):
    """painted once at 1080p (cached in cache/), resized to the drawing's working size"""
    k = (h, w, size, seed)
    if k not in _DAB:
        fn = os.path.join(os.path.dirname(__file__), '..', 'cache', f'maho_dab_{size}_{seed}.npy')
        try: d = np.load(fn).astype(f32)
        except Exception:
            r = np.random.default_rng(seed + int(size * 7)); hh, ww = 1080, 1920; n = int(hh * ww / (size * size * 1.1))
            cv = Canvas(hh, ww, color=np.array([.5, .5, .5], f32))
            P = arcs(r.uniform(-20, ww + 20, n), r.uniform(-20, hh + 20, n), ang + r.normal(0, spread, n), size * aspect * r.uniform(.6, 1.4, n), r.normal(0, .5, n))
            v = r.uniform(0, 1, n).astype(f32)
            cv.strokes(P, size * r.uniform(.55, 1.2, n), np.repeat(v[:, None], 3, 1), op=r.uniform(.55, 1, n), dry=.10 * r.random(n), seed=seed, shape='flat', bristle=.10)
            d = cv.img[..., 0]; d = ((d - d.mean()) / (d.std() + 1e-6)).astype(f32)
            tmp = fn + '.%d.npy' % os.getpid(); np.save(tmp, d.astype(np.float16)); os.replace(tmp, fn)
        _DAB[k] = d if d.shape == (h, w) else cv2.resize(d, (w, h), interpolation=cv2.INTER_AREA)
    return _DAB[k]


def wash(h, w, seed=0):
    """multi-scale dab field, rolled by seed so each drawing has its own brushwork"""
    big = dabs(h, w, 46, 1); mid = dabs(h, w, 21, 2); sm = dabs(h, w, 9, 3)
    r = np.random.default_rng(seed)
    f = lambda a: np.roll(a, (int(r.integers(0, h)), int(r.integers(0, w))), (0, 1))
    return cv2.GaussianBlur(.68 * f(big) + .44 * f(mid) + .12 * f(sm), (0, 0), 1.1 * h / 1080 + .25)
