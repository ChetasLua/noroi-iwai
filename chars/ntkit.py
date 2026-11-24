"""Shared construction for Nanami and Todo: FHead (Head with visible-side clamping and contour parts), 3-D hair
(shell + locks that shingle over each other), eyes, noses, ink helpers.  Nothing here knows about a character."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, cv2, math
from jj import W, H
from jj.head3 import Head
from jj.ink import Cel, spl, resample, ribbon, noise1, over_roi, boil, xf
from jj.kit import svg, s1, outline, shape, mask_of, contours, hatch, splatter, rim, para, tint, spike, Turn
from jj.paint import rgb, mix, smooth, ramp, fbm, grid, f32, Canvas, arcs, repaint
from jj import comp

INK = '#17131a'
_spl2 = spl


def spl(P, closed=False, corners=(), n=10):
    """jj.ink.spl for 2-D or 3-D control points"""
    P = np.asarray(P, float)
    if P.shape[-1] == 2: return _spl2(P, closed, corners, n)
    # centripetal parametrisation would need the 3-D norm: do xy + z with the same knots by splining (x,y) and (arc,z) jointly
    def cr(Q, closed_):
        m = len(Q); out = []
        for i in range(m if closed_ else m - 1):
            if closed_: p0, p1, p2, p3 = Q[(i - 1) % m], Q[i], Q[(i + 1) % m], Q[(i + 2) % m]
            else:
                p1, p2 = Q[i], Q[i + 1]; p0 = Q[i - 1] if i > 0 else 2 * p1 - p2; p3 = Q[i + 2] if i + 2 < m else 2 * p2 - p1
            d1, d2, d3 = [max(np.linalg.norm(b - a), 1e-4) ** .5 for a, b in ((p0, p1), (p1, p2), (p2, p3))]
            m1 = (p2 - p1) + d2 * ((p1 - p0) / d1 - (p2 - p0) / (d1 + d2)); m2 = (p2 - p1) + d2 * ((p3 - p2) / d3 - (p3 - p1) / (d2 + d3))
            t = np.linspace(0, 1, n, endpoint=False)[:, None]
            out.append((2 * t ** 3 - 3 * t ** 2 + 1) * p1 + (t ** 3 - 2 * t ** 2 + t) * m1 + (-2 * t ** 3 + 3 * t ** 2) * p2 + (t ** 3 - t ** 2) * m2)
        out = np.concatenate(out)
        return out if closed_ else np.vstack([out, Q[-1]])
    c = sorted(set(int(i) % len(P) for i in corners))
    if not c: return cr(P, closed)
    if closed: return np.concatenate([cr(P[[k % len(P) for k in range(a, b + 1)]], False)[:-1] for a, b in zip(c, c[1:] + [c[0] + len(P)])])
    cs = [0] + [k for k in c if 0 < k < len(P) - 1] + [len(P) - 1]
    return np.vstack([cr(P[a:b + 1], False)[:-1] for a, b in zip(cs[:-1], cs[1:])] + [P[-1:]])


def hexmix(a, b, t):
    c = np.clip(rgb(a) * (1 - t) + rgb(b) * t, 0, 1); return '#%02x%02x%02x' % tuple(int(round(v * 255)) for v in c)


def mulcol(a, b, gain=1.0):
    c = np.clip(rgb(a) * rgb(b) * gain, 0, 1); return '#%02x%02x%02x' % tuple(int(round(v * 255)) for v in c)


def bz(p0, p1, p2, p3, n=18):
    u = np.linspace(0, 1, n)[:, None]; p0, p1, p2, p3 = [np.asarray(v, float) for v in (p0, p1, p2, p3)]
    return (1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * p1 + 3 * (1 - u) * u * u * p2 + u ** 3 * p3


# ------------------------------------------------------------------ head
class FHead(Head):
    ZJ = [.64, .64, .58, .48, .32, .16, .04, -.04, -.10, -.10]
    def __init__(self, zj=None, prof=None, low=None, **kw):
        if zj is not None: self.ZJ = zj
        self.prof = prof; self.low = low                     # (y0, k): the face below y0 is k times as long
        super().__init__(**kw)

    def P3(self, Q):
        if self.low:
            Q = np.array(Q, float); y0, k = self.low; y = Q[..., 1]; Q[..., 1] = np.where(y < y0, y0 + (y - y0) * k, y)
        return Head.P3(self, Q)

    def _mesh(self, y0, y1, half, ny=90, nf=90):
        y = np.linspace(y0, y1, ny)
        zj = np.interp(y, [-1.0, -.995, -.96, -.90, -.82, -.72, -.60, -.46, -.30, -.2], self.ZJ)
        A, F, B = (np.interp(y, self.yy, t) for t in (self.A, self.F, self.B)); p = self.k['p']
        ph = np.linspace(-math.pi, math.pi, nf) if not half else np.linspace(-math.pi * .5 * half, math.pi * .5 * half, nf)
        PH = np.repeat(ph[None], ny, 0)
        if half:
            c = np.clip(zj / np.maximum(np.where(zj >= 0, F, B), 1e-3), -1, .98)
            t = np.where(c >= 0, np.arctan2((1 - np.abs(c) ** p) ** (1 / p), np.abs(c)), np.pi / 2 + .0 * c + np.arcsin(np.clip(-c, 0, 1)))
            PH = np.linspace(-1, 1, nf)[None] * np.minimum(math.pi * .5 * half, t)[:, None]
        sx, cz = np.sin(PH), np.cos(PH)
        r = (np.abs(sx) ** p + np.abs(cz) ** p) ** (-1 / p)
        X = A[:, None] * sx * r; Z = np.where(cz >= 0, F[:, None], B[:, None]) * cz * r
        Y = np.repeat(y[:, None], nf, 1); fr = np.clip(cz, 0, 1)
        Z = Z + fr * (self._mid(X, Y) + self.k['cheek'] * np.exp(-((Y + .22) / .13) ** 2) * np.exp(-((np.abs(X) - .44) / .2) ** 2) * 1.4
                      - self.k['socket'] * np.exp(-((Y - .02) / .10) ** 2) * np.exp(-((np.abs(X) - .40) / .24) ** 2) * 1.3)
        return np.stack([X, Y, Z], -1)

    def jaw(self, side=1):
        y = np.array([-.20, -.30, -.46, -.60, -.72, -.82, -.90, -.96, -.995])
        a = np.interp(y, self.yy, self.A)
        z = np.interp(y, [-1.0, -.995, -.96, -.90, -.82, -.72, -.60, -.46, -.30, -.2], self.ZJ)
        x = a * np.array([1., 1.0, 1.0, 1.0, .985, .95, .86, .66, .0]) * side
        return self.P3(np.stack([x, y, z], -1))

    # ---- visible clamp: slide points that have turned away back to the face edge so nothing folds over
    def _face_cos(self, x, y):
        e = 1e-2
        nx = -(self.z(x + e, y) - self.z(x - e, y)) / (2 * e); ny = -(self.z(x, y + e) - self.z(x, y - e)) / (2 * e)
        l = np.sqrt(nx * nx + ny * ny + 1)
        return (self.R[2, 0] * nx + self.R[2, 1] * ny + self.R[2, 2]) / l

    def clampx(self, xy, lim=.03):
        xy = np.array(xy, float)
        if abs(self.yaw) < .12: return xy
        s = -1. if self.yaw < 0 else 1.                     # far side sign
        g = np.linspace(0, .98, 60)
        for k in range(len(xy)):
            x, y = xy[k]
            if x * s <= .05: continue
            a = np.interp(y, self.yy, self.A)
            c = self._face_cos(s * g * a, np.full(60, y))
            bad = np.nonzero(c < lim)[0]
            if len(bad): xy[k, 0] = s * min(x * s, g[max(0, bad[0] - 1)] * a)
        return xy

    def Q(self, xy, dz=0.):
        return self.P(self.clampx(xy), dz)

    def ear(self, side=1):
        ex, t, b, z = self.k['ear']; s = side; h = t - b
        # (u outward, v 0..1 up from the lobe)
        O = [(.0, .10), (.012, .0), (.045, -.015), (.075, .07), (.098, .26), (.135, .48), (.165, .70), (.168, .88), (.135, 1.0), (.085, 1.03), (.035, .97), (.0, .82)]
        I = [(.012, .80), (.045, .91), (.095, .94), (.130, .86), (.128, .70), (.105, .50), (.082, .34)]
        T = [(.010, .56), (.052, .64), (.078, .54), (.066, .38), (.038, .24), (.012, .32)]
        def Q(Lx): return self.P3(np.array([(s * (ex - .012 + u * .95), b + v * h, z + .02 - u * 1.45 - v * .05) for u, v in Lx]))
        return dict(outer=spl(Q(O), n=6), inner=spl(Q(I), n=6), tragus=spl(Q(T), True, n=6), vis=float((self.R @ np.array([s * .84, 0, -.35 + .62]))[2]))

    def ux(self): return np.array([math.cos(self.roll), math.sin(self.roll)])
    def uy(self): return np.array([math.sin(self.roll), -math.cos(self.roll)])

    def near(self):
        """side whose ear/cheek faces us: -1 (picture left) when yaw > 0"""
        return 0 if abs(self.yaw) < .18 else (-1 if self.yaw > 0 else 1)

    def profile_poly(self):
        """authored profile line (sagittal) in screen px, forehead -> chin"""
        z0 = np.array(self.prof, float)                      # list of (z, y)
        return self.P3(spl(np.c_[np.zeros(len(z0)), z0[:, 1], z0[:, 0]][:, 1:], n=8)[:, [0, 1]] @ np.array([[0, 1., 0], [0, 0, 1.]]) + 0.0)

    def parts(self, hair_y=.55, lobe_y=-.34):
        """dict: fill polygon, left / right ink lines (chin -> up)"""
        f = self.face(top=.72)
        if abs(self.yaw) > 1.12 and self.prof is not None:
            s = 1 if self.yaw > 0 else -1
            q = np.array(self.prof, float); pr = spl(np.c_[q[:, 0], q[:, 1]], n=8)
            pl = self.P3(np.c_[np.zeros(len(pr)), pr[:, 1], pr[:, 0]])
            jw = self.jaw(-s)[::-1]                            # chin -> lobe, near side
            up = self.P3(np.array([(-s * .62, -.10, -.10), (-s * .62, .3, -.1), (-s * .5, .72, 0.1)]))
            fill = np.vstack([pl, jw[1:], up])
            near = spl(jw[:-1], n=6) if True else jw
            d = dict(fill=fill)
            d['left' if s > 0 else 'right'] = near; d['right' if s > 0 else 'left'] = pl[::-1]
            return d
        chin = self.P3(np.array([(0, -1., .64)]))[0]
        i = int(np.argmin(np.hypot(*(f - chin).T)))
        # chin is the lowest point along uy near there
        F = np.roll(f, -i, 0)
        a, b = F, F[::-1]
        if (a[14] - b[14]) @ self.ux() > 0: R, L = a, b
        else: R, L = b, a
        out = dict(fill=f)
        for nm, pth, s in (('left', L, -1), ('right', R, 1)):
            far = self.near() != 0 and self.near() != s
            an = self.P3(np.array([(s * np.interp(lobe_y, self.yy, self.A), lobe_y, -.10)]))[0] if not far else None
            h = (pth - chin) @ self.uy()
            if far:
                top = (self.P3(np.array([(s * .5, hair_y, .5)]))[0] - chin) @ self.uy()
            else: top = (an - chin) @ self.uy()
            k = np.argmax(h > top) if (h > top).any() else len(pth) // 2
            o_ = b if pth is a else a
            out[nm] = np.vstack([o_[1:7][::-1], pth[:max(6, k)]])
        return out


# ------------------------------------------------------------------ ink in face coordinates
def L(cel, H_, pts, w, col=INK, seed=0, taper=(.22, .28), press=.28, n=9, dz=0., corners=(), tmin=.2, alpha=1., smooth_=True):
    P = H_.Q(spl(pts, corners=corners, n=n) if smooth_ and len(pts) > 2 else np.asarray(pts, float), dz)
    cel.line(P, w, col, taper=taper, press=press, seed=seed, tmin=tmin, alpha=alpha, tremor=.2); return cel


def S(cel, H_, pts, col, within=None, closed_spl=False, corners=(), alpha=1., dz=0., n=8, rnd=False):
    """shadow / flat polygon in face coordinates; straight decisive edges by default (points resampled so it wraps the form)"""
    P = np.asarray(pts, float)
    if rnd: P = spl(P, True, corners=corners, n=n)
    else: P = resample(np.vstack([P, P[:1]]), .02)
    q = H_.Q(P, dz)
    if within is not None: cel.clip(within)
    cel.fill(q, col, alpha)
    if within is not None: cel.unclip()
    return cel


def sp(pts, closed=False, corners=(), n=8): return spl(pts, closed, corners, n)


# ------------------------------------------------------------------ hair
class Hair:
    """Ellipsoid-ish hair surface on a head.  S(theta_deg, v, lift) -> 3-D.  v in [-1,1]: 0 equator (y=yc), 1 crown, -1 nape line."""
    def __init__(self, yc=.22, ry=.90, rx=.735, rzf=.86, rzb=1.0, zc=-.06, low=.95, box=.82):
        self.box = box; self.yc, self.ry, self.rx, self.rzf, self.rzb, self.zc, self.low = yc, ry, rx, rzf, rzb, zc, low

    def S(self, th, v, lift=0.):
        th = np.radians(np.asarray(th, float)); v = np.asarray(v, float); lift = np.asarray(lift, float)
        el = np.clip(v, 0, 1) * math.pi / 2
        y = np.where(v >= 0, self.yc + self.ry * np.sin(el), self.yc + v * self.low)
        R = np.where(v >= 0, np.cos(el) ** self.box, np.interp(-v, [0, .25, .5, .7, .85, 1.0], [1, .985, .95, .88, .80, .72]))
        Rz = np.where(v >= 0, np.cos(el) ** self.box, np.interp(-v, [0, .25, .5, .7, .85, 1.0], [1, .95, .82, .65, .55, .49]))
        sx, cz = np.sin(th), np.cos(th)
        rz = np.where(cz >= 0, self.rzf, self.rzb)
        p = np.stack(np.broadcast_arrays(self.rx * R * sx, y, self.zc + rz * np.where(cz >= 0, R, Rz) * cz), -1)
        c = np.array([0, self.yc, self.zc]); n = p - c; n = n / np.array([self.rx ** 2, self.ry ** 2, 1.]) ; n = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9)
        return p + n * lift[..., None] if np.ndim(lift) else p + n * lift

    def normal(self, p):
        n = np.asarray(p, float) - np.array([0, self.yc, self.zc]); n = n / np.array([self.rx ** 2, self.ry ** 2, 1.])
        return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9)

    def shell(self, H_, line, v1=1.0, lift=0., nt=120, nv=36, face=.0):
        """polygons of the shell above the hairline line(theta_deg)->v; returns (front quads, back quads) in screen px"""
        th = np.linspace(-180, 180, nt + 1); lo = np.array([line(t) for t in th])
        u = np.linspace(0, 1, nv)
        V = lo[:, None] * (1 - u) + v1 * u; T = np.repeat(th[:, None], nv, 1)
        P = self.S(T, V, lift(T, V) if callable(lift) else lift); q = H_.P3(P); n = self.normal(P); d = (n @ H_.R.T)[..., 2]
        quads = np.stack([q[:-1, :-1], q[1:, :-1], q[1:, 1:], q[:-1, 1:]], 2).reshape(-1, 4, 2)
        dd = (d[:-1, :-1] + d[1:, 1:]).reshape(-1) / 2
        return quads[dd > face], quads[dd <= face]


def quads_contour(quads, k=1.0, close=5):
    """outer contours of a pile of quads -> list of polylines"""
    if len(quads) == 0: return []
    lo = quads.reshape(-1, 2).min(0) - 8; hi = quads.reshape(-1, 2).max(0) + 8
    w, h = np.ceil((hi - lo) * k).astype(int) + 2
    m = np.zeros((h, w), np.uint8)
    for c in np.round((quads - lo) * k * 16).astype(np.int32): cv2.fillConvexPoly(m, c, 255, cv2.LINE_8, 4)
    if close: m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    return [c / k + lo for c in contours(m / 255., min_len=40)]


def union_contour(polys, close=5, k=1.0, smooth_=0):
    """contours of the union of polygons"""
    allp = np.vstack(polys); lo = allp.min(0) - 10; hi = allp.max(0) + 10
    w, h = np.ceil((hi - lo) * k).astype(int) + 2
    m = np.zeros((h, w), np.uint8)
    for p in polys: cv2.fillPoly(m, [np.round((p - lo) * k * 16).astype(np.int32)], 255, cv2.LINE_8, 4)
    d = max(allp.max(0) - allp.min(0)); close = max(close, int(d * .016) | 1) if close else 0
    if close: m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close)))
    cs = [c for c in contours(m / 255., min_len=40) if cv2.contourArea(c.astype(np.float32)) > (d * .10) ** 2]
    return [c / k + lo for c in cs]


def lock(H_, hair, path, w, prof=None, off=None, n=34, wind=None):
    """one lock of hair as a 3-D ribbon.  path = [(theta, v, lift), ...]; off = [(dx,dy,dz)] added per control point (free tips).
    returns dict(left, right, mid: screen polylines root->tip; poly; depth; face)"""
    a = np.array([q[:3] for q in path], float)
    P = hair.S(a[:, 0], a[:, 1], a[:, 2])
    if off is not None: P = P + np.asarray(off, float)
    if wind is not None:
        t = np.linspace(0, 1, len(P))[:, None]; P = P + np.asarray(wind, float)[None] * t ** 2
    # 3-D Catmull-Rom
    u = np.linspace(0, len(P) - 1, n); i = np.clip(u.astype(int), 0, len(P) - 2); f = (u - i)[:, None]
    g = lambda k: P[np.clip(k, 0, len(P) - 1)]
    p0, p1, p2, p3 = g(i - 1), g(i), g(i + 1), g(i + 2)
    C = .5 * ((2 * p1) + (-p0 + p2) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * f ** 3)
    T = np.gradient(C, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
    N = hair.normal(C); B = np.cross(T, N); B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-9
    t = np.linspace(0, 1, n)
    pr = np.interp(t, *np.array(prof or [(0, .55), (.25, 1), (.6, .8), (.85, .42), (1, 0)]).T)
    h = (w * .5 * pr)[:, None]
    Lq, Rq, Mq = H_.P3(C + B * h), H_.P3(C - B * h), H_.P3(C)
    dn = (N @ H_.R.T)[:, 2]
    return dict(left=Lq, right=Rq, mid=Mq, poly=np.vstack([Lq, Rq[::-1]]), depth=float(H_.depth(C).mean()), face=float(dn.mean()), C=C, fz=dn)


def cr3(P, n):
    """uniform Catmull-Rom through 3-D points, n samples"""
    P = np.asarray(P, float); u = np.linspace(0, len(P) - 1, n); i = np.clip(u.astype(int), 0, len(P) - 2); f = (u - i)[:, None]; g = lambda k: P[np.clip(k, 0, len(P) - 1)]
    p0, p1, p2, p3 = g(i - 1), g(i), g(i + 1), g(i + 2)
    return .5 * ((2 * p1) + (-p0 + p2) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * f ** 3)


def patch_lock(H_, hair, e0, e1, off=None, n=44, wind=None):
    """a big lock as a patch of the hair surface between two flow lines e0 (upper) and e1 (lower), each [(theta, v, lift), ...];
    off = [(dx,dy,dz)] added to the last len(off) points of both edges (free tips).  same dict as lock()"""
    Es = []
    for e in (e0, e1):
        a = np.array([q[:3] for q in e], float); P = hair.S(a[:, 0], a[:, 1], a[:, 2])
        if off is not None: P[-len(off):] += np.asarray(off, float)
        if wind is not None: t = np.linspace(0, 1, len(P))[:, None]; P = P + np.asarray(wind, float)[None] * t ** 2.2
        Es.append(cr3(P, n))
    A_, B_ = Es; C = (A_ + B_) / 2; N = hair.normal(C)
    Lq, Rq, Mq = H_.P3(A_), H_.P3(B_), H_.P3(C)
    return dict(left=Lq, right=Rq, mid=Mq, poly=np.vstack([Lq, Rq[::-1]]), depth=float(H_.depth(C).mean()), face=float((N @ H_.R.T)[:, 2].mean()), C=C, fz=(N @ H_.R.T)[:, 2])


def trim_lock(d, lim=-.10):
    """drop the part of a lock that has gone round behind the head (it would fold back over the face)"""
    v = d['fz'] > lim; n = len(v)
    if v.all(): return d
    best = (0, 0); i = 0
    while i < n:
        if not v[i]: i += 1; continue
        j = i
        while j < n and v[j]: j += 1
        if j - i > best[1] - best[0]: best = (i, j)
        i = j
    a, b = best
    if b - a < 5: return None
    o = dict(d); o['left'], o['right'], o['mid'] = d['left'][a:b], d['right'][a:b], d['mid'][a:b]; o['poly'] = np.vstack([o['left'], o['right'][::-1]]); o['fz'] = d['fz'][a:b]; o['C'] = d['C'][a:b]
    return o


def lane(d, u, i0=0, i1=None):
    """polyline inside a lock: u = 0 left edge .. 1 right edge (scalar or per-point array)"""
    Lq, Rq = d['left'][i0:i1], d['right'][i0:i1]; u = np.broadcast_to(np.asarray(u, float), (len(Lq),))[:, None]
    return Lq * (1 - u) + Rq * u


def sliver(cel, d, u, t0, t1, w, col, drift=0., alpha=1., sharp=(.8, 1.2)):
    """pointed streak that follows a lock: lane u, from t0 to t1 of its length, w = fraction of the lock width"""
    n = len(d['mid']); i0, i1 = int(t0 * (n - 1)), int(t1 * (n - 1)) + 1
    if i1 - i0 < 4: return
    q = np.linspace(0, 1, i1 - i0); prof = np.sin(math.pi * q ** sharp[0]) ** sharp[1] * w * .5; uu = u + drift * (q - .5)
    cel.fill(np.vstack([lane(d, np.clip(uu - prof, 0, 1), i0, i1), lane(d, np.clip(uu + prof, 0, 1), i0, i1)[::-1]]), col, alpha)


def draw_locks(cel, locks, col, shade_col, ink, w, side=1, t0=.35, seed=0, shade=True, hi=None, sort=True, root_fade=True, strands=1.):
    """shingle locks far -> near: flat, root + edge shadow broken into pointed streaks, highlight streaks, strand lines, ink edges that grow toward the tip"""
    ls = sorted(locks, key=lambda d: d['depth']) if sort else locks
    for k, d in enumerate(ls):
        r = np.random.default_rng(seed * 13 + k * 101 + 7); c, sc = d.get('col', col), d.get('shade', shade_col)
        cel.fill(d['poly'], c); sd = d.get('side', side); n = len(d['mid'])
        if d.get('root', 0) > 0:                                             # the root lies in shade; teeth run out of it
            q = d['root']; cel.fill(np.vstack([d['left'][:int(n * q * .5) + 2], d['right'][:int(n * q * .5) + 2][::-1]]), sc)
            for j in range(5): sliver(cel, d, (j + .5) / 5 + r.normal(0, .04), 0, q * r.uniform(.6, 1.35), .26, sc, sharp=(.35, .9))
        if shade and d.get('sh', 1) > 0:
            q = d.get('sh', .5); e = 1. if sd > 0 else 0.; g = np.linspace(0, 1, n)
            a = d['right'] if sd > 0 else d['left']
            edge = lane(d, np.clip(e - (1 if sd > 0 else -1) * q * .62 * (1.15 - .6 * g), 0, 1)); cel.fill(np.vstack([a, edge[::-1]]), sc)
            for j in range(3):
                u = e - (1 if sd > 0 else -1) * q * (.60 + .22 * j + r.normal(0, .04)); tt = r.uniform(.0, .45)
                sliver(cel, d, u, tt, min(1, tt + r.uniform(.3, .55)), .15, sc, drift=r.normal(0, .06))
        if hi is not None and d.get('hi', 0):
            h0, h1 = d['hi'][0], d['hi'][1]; e = 0. if sd > 0 else 1.; dr = 1 if sd > 0 else -1
            for j, (uu, ww) in enumerate(((.17, .17), (.36, .10), (.50, .05))):
                tt = h0 + r.normal(0, .03) + .03 * j; sliver(cel, d, e + dr * uu, tt, tt + (h1 - h0) * r.uniform(.7, 1.1), ww, hi)
        if strands > 0 and d.get('strands', 1):
            for j in range(int(d.get('strands', 2) * strands)):
                tt = r.uniform(.02, .5); u = r.uniform(.18, .82); i0, i1 = int(tt * n), int(min(.97, tt + r.uniform(.25, .5)) * n)
                if i1 - i0 > 5: cel.line(lane(d, u + r.normal(0, .05) * np.linspace(-1, 1, i1 - i0), i0, i1), w * .62, ink, taper=(.45, .5), press=.2, seed=seed + k * 17 + j, tmin=.06, tremor=.1)
        i0 = int(n * d.get('t0', t0))
        for e, s_ in (('left', 1), ('right', 2)):
            if d.get('no', '') == e: continue
            P = d[e][i0:]
            if len(P) > 4: cel.line(P, w * d.get('lw', 1.), ink, taper=(.5 if root_fade else .1, .06), press=.25, seed=seed + k * 7 + s_, tmin=.08, tremor=.15)
    return cel


# ------------------------------------------------------------------ eye
def eye_pts(s, ex, ey, ew, eh, tilt, U): return np.array([(s * (ex + u * ew), ey + v * eh + tilt * u) for u, v in U])


def wind_vec(f, amp, seed=0, d=(1., .15, 0.)):
    g = math.sin(f * .31 + seed * 1.7) * .5 + math.sin(f * .13 + seed * 2.9) * .5
    return np.array(d) * amp * (1. + .35 * g) + np.array([0, .25, 0]) * amp * math.sin(f * .22 + seed)


def rim(layer, dx, dy, col, amt=1.0, soft=.6, where=None):
    """kit.rim without the false rim where the cel leaves the frame"""
    a = layer[..., 3]; h, w = a.shape
    sh = cv2.warpAffine(a, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    m = np.clip(a - sh, 0, 1)
    if soft > 0: m = cv2.GaussianBlur(m, (0, 0), soft) * (a > .02)
    if where is not None: m = m * where
    c = rgb(col) if isinstance(col, str) else np.asarray(col, f32); out = layer.copy()
    out[..., :3] = out[..., :3] * (1 - m[..., None] * amt) + c * (m * a)[..., None] * amt
    return out


def qb(img, s):
    """quick gaussian: big sigmas at reduced resolution"""
    q = 1 if s < 4 else 2 if s < 8 else 4 if s < 24 else 8
    if q == 1: return cv2.GaussianBlur(img, (0, 0), s)
    h, w = img.shape[:2]; sm = cv2.resize(img, (w // q, h // q), interpolation=cv2.INTER_AREA); sm = cv2.GaussianBlur(sm, (0, 0), s / q); return cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)
