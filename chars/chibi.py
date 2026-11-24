"""Super-deformed cast: brush-inked, flat cel, one shadow.  Pure function of (who, pose, t).

    L = draw('gojo', 'starfish', x, y, S, rot=.4, t=f)            # premultiplied full-frame RGBA layer
    d = D.frame(); put(d, 'gojo', 'starfish', ...); put(d, ...); L = d.layer()      # several on one cel, painter's order

Local units: head half-width = 1, y down, neck at (0, 0); a body is about 1.9 long under a head 1.85 tall.
"""
import numpy as np, cv2, math, cairo
from jj import W, H
from jj.paint import rgb, f32, smooth
from jj.ink import Cel, spl, resample, ribbon, noise1, paste
from jj.kit import s1, svg

LINE = '#1a1216'; SKINL = '#8a3024'; SKINO = '#45191a'
pi = math.pi


# ================================================================ drawing surface
class D:
    """four registered cels: flat colour, shadow colour, shadow mask, ink"""
    def __init__(self, x0, y0, w, h, ss=2):
        self.x0, self.y0, self.w, self.h, self.ss = int(x0), int(y0), int(w), int(h), ss
        mk = lambda: Cel(self.w, self.h, self.x0, self.y0, ss)
        self.fl, self.sc, self.mk, self.ln = mk(), mk(), mk(), mk()
        self.lw = 4.6; self.soft = 1.1; self.lw_ref = 4.6

    @classmethod
    def frame(cls, ss=2): return cls(0, 0, W, H, ss)

    def _knock(self, P):
        c = self.ln.ctx; c.save(); c.set_operator(cairo.OPERATOR_DEST_OUT); self.ln._path(P); c.set_source_rgba(0, 0, 0, 1); c.fill(); c.restore()

    def part(self, P, col, sh, shade=(), hi=(), w=1.0, ink=LINE, gaps=(), seed=0, press=.2, light=(-.55, -.83), tip=True):
        """closed shape P (px).  shade = polygons of the cast shadow, hi = [(poly, hex)] flat details, gaps = [(t0,t1)] of the
        contour left without ink"""
        P = np.asarray(P, float)
        self._knock(P)
        self.fl.fill(P, col); self.sc.fill(P, sh); self.mk.fill(P, '#000000')
        for q in shade: self.mk.shade(q, '#ffffff', within=P)
        for q, c in hi: self.fl.shade(q, c, within=P); self.sc.shade(q, _mix(c, '#1c1430', .34), within=P)
        if w > 0: self.contour(P, w, ink, gaps, seed, press, light, tip)
        return self

    def shade(self, q, within=None):
        self.mk.shade(q, '#ffffff', within=within); return self

    def detail(self, q, col, within=None, sh=None):
        self.fl.shade(q, col, within=within); self.sc.shade(q, sh or _mix(col, '#2a1c30', .42), within=within); return self

    def stroke(self, P, w=.5, ink=LINE, taper=(.22, .3), seed=0, press=.15, tmin=.12, within=None):
        P = np.asarray(P, float)
        if within is not None: self.ln.clip(within)
        self.ln.line(P, w * self.lw, ink, taper=taper, press=press, seed=seed, tremor=.3, tmin=tmin)
        if within is not None: self.ln.unclip()
        return self

    # ------------------------------------------------ brush contour
    def contour(self, P, w, ink, gaps=(), seed=0, press=.2, light=(-.55, -.83), tip=True):
        P = np.asarray(P, float); P = resample(np.vstack([P, P[:1]]), 1.7)[:-1]; n = len(P)
        if n < 8: return
        area = .5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]); sg = 1. if area > 0 else -1.
        T = np.roll(P, -2, 0) - np.roll(P, 2, 0); T /= np.hypot(*T.T)[:, None] + 1e-9
        N = np.c_[T[:, 1], -T[:, 0]] * sg                                           # outward
        k = -(N[:, 0] * light[0] + N[:, 1] * light[1])                              # >0 away from the light
        wl = .66 + .62 * smooth(-.7, .95, _cbox(k, 9))
        d = 4; a = np.arctan2(P[:, 1] - np.roll(P, d, 0)[:, 1], P[:, 0] - np.roll(P, d, 0)[:, 0]); b = np.arctan2(np.roll(P, -d, 0)[:, 1] - P[:, 1], np.roll(P, -d, 0)[:, 0] - P[:, 0])
        turn = np.angle(np.exp(1j * (b - a))) * sg                                  # + convex, - concave
        cv = _cbox(np.clip(turn, 0, 1.6), 11) / 1.0; cc = _cbox(np.clip(-turn, 0, 1.6), 9)
        wl = wl * (1 - .50 * np.clip(cv, 0, 1)) * (1 + .45 * np.clip(cc, 0, 1))
        r = np.random.default_rng(abs(int(seed)) * 7 + 11); nz = noise1(n, 40, r)
        nz = nz * np.minimum(1, np.minimum(np.arange(n), n - 1 - np.arange(n)) / 30.)
        wl = wl * (1 + press * np.clip(nz, -1.3, 1.3)); Wd = np.maximum(.5, w * self.lw * wl)
        on = np.ones(n, bool); t = np.arange(n) / n; fade = np.ones(n)
        for g0, g1 in gaps:
            m = (t >= g0) & (t <= g1) if g0 <= g1 else (t >= g0) | (t <= g1); on &= ~m
        if gaps:                                                                     # strokes lift into a gap
            di = cv2.distanceTransform(np.r_[on, on, on].astype(np.uint8)[:, None], cv2.DIST_L1, 3)[n:2 * n, 0]; fade = np.minimum(1, di * 1.7 / 16.) ** .7; fade = .25 + .75 * fade
        c = self.ln.ctx; c.save(); c.set_line_cap(cairo.LINE_CAP_ROUND); col = rgb(ink) if isinstance(ink, str) else ink; c.set_source_rgba(*map(float, col), 1)
        Wd = Wd * fade
        for i in range(n):
            j = (i + 1) % n
            if not (on[i] and on[j]): continue
            c.set_line_width(float(.5 * (Wd[i] + Wd[j]))); c.move_to(*P[i]); c.line_to(*P[j]); c.stroke()
        if tip:                                                                      # pen flick past sharp points
            for i in np.nonzero((turn > 1.15) & (turn >= np.roll(turn, 1)) & (turn > np.roll(turn, -1)))[0]:
                if not on[i]: continue
                back = P[i] - .5 * (P[(i - 7) % n] + P[(i + 7) % n]); L = np.hypot(*back) + 1e-6; e = back / L * w * self.lw * .85
                nrm = np.array([-e[1], e[0]]); nrm = nrm / (np.hypot(*nrm) + 1e-6) * Wd[i] * .5
                self.ln.fill(np.array([P[i] - nrm, P[i] + e, P[i] + nrm, P[i] - e * .5]), ink)
        c.restore()

    # ------------------------------------------------ result
    def rgba(self, outer=('#1c3f3c', .72)):
        """outer = (hex, amount): the silhouette's own line leans to the colour of the place (green-navy in grass)"""
        a = self.fl.rgba(); s = self.sc.rgba(); m = self.mk.rgba()[..., 0]; l = self.ln.rgba()
        if outer is not None:
            A = np.maximum(a[..., 3], l[..., 3]); px = self.lw_ref * 1.7; dt = cv2.distanceTransform((A > .5).astype(np.uint8), cv2.DIST_L2, 5)
            k = (np.clip(1.35 - dt / px, 0, 1) * outer[1])[..., None]; l = np.concatenate([l[..., :3] * (1 - k) + rgb(outer[0]) * l[..., 3:] * k, l[..., 3:]], -1)
        m = cv2.GaussianBlur(m, (0, 0), self.soft)[..., None]
        al = np.maximum(a[..., 3:4], s[..., 3:4])
        rgb_ = a[..., :3] * (1 - m) + s[..., :3] * m
        out = np.concatenate([rgb_, al], -1)
        out = out * (1 - l[..., 3:4]) + l
        return out

    def layer(self, **k):
        f = self.rgba(**k)
        if (self.x0, self.y0, self.w, self.h) == (0, 0, W, H): return f
        out = np.zeros((H, W, 4), f32); paste(out, f, self.x0, self.y0); return out


def _cbox(v, k):
    ker = np.hanning(k + 2)[1:-1]; ker /= ker.sum(); p = k // 2 + 1
    return np.convolve(np.r_[v[-p:], v, v[:p]], ker, 'same')[p:-p]


def _mix(a, b, t):
    a = rgb(a) if isinstance(a, str) else a; b = rgb(b) if isinstance(b, str) else b
    c = np.clip(a * (1 - t) + b * t, 0, 1); return '#%02x%02x%02x' % tuple(int(v * 255 + .5) for v in c)


# ================================================================ geometry
class N:
    """2-D node: local -> px"""
    def __init__(self, M=None): self.M = np.eye(3) if M is None else M
    def at(self, x=0., y=0., rot=0., sx=1., sy=None):
        sy = sx if sy is None else sy; c, s = math.cos(rot), math.sin(rot)
        return N(self.M @ np.array([[c * sx, -s * sy, x], [s * sx, c * sy, y], [0, 0, 1.]]))
    def __call__(self, P):
        P = np.asarray(P, float); return P @ self.M[:2, :2].T + self.M[:2, 2]
    @property
    def s(self): return math.sqrt(abs(np.linalg.det(self.M[:2, :2])))
    @property
    def flip(self): return np.linalg.det(self.M[:2, :2]) < 0


def bez(a, c1, c2, b, n=12):
    a, c1, c2, b = [np.asarray(v, float) for v in (a, c1, c2, b)]; u = np.linspace(0, 1, n)[:, None]
    return (1 - u) ** 3 * a + 3 * (1 - u) ** 2 * u * c1 + 3 * (1 - u) * u * u * c2 + u ** 3 * b


def flame(a, t, b, bulge=.16, bend=.0, n=11):
    """one hair lock a -> tip -> b: both sides swept the same way (bend), belly out near the root"""
    a, t, b = [np.asarray(v, float) for v in (a, t, b)]
    ax = t - (a + b) / 2; L = np.hypot(*ax) + 1e-6; nr = np.array([-ax[1], ax[0]]) / L
    s = 1. if np.dot(a - b, nr) > 0 else -1.
    A = bez(a, a + ax * .36 + nr * L * (bulge * s + bend), a + (t - a) * .74 + nr * L * (bulge * .30 * s + bend * .8), t, n)
    B = bez(t, b + (t - b) * .74 + nr * L * (-bulge * .30 * s + bend * .8), b + ax * .36 + nr * L * (-bulge * s + bend), b, n)
    return np.vstack([A, B[1:]])


def flames(v0, seq, bulge=.14):
    """v0, [(tip, valley, bend[, bulge])...] -> open polyline"""
    out = []; a = v0
    for q in seq:
        t, b, bd = q[:3]; bu = q[3] if len(q) > 3 else bulge
        out.append(flame(a, t, b, bu, bd)[:-1]); a = b
    return np.vstack(out + [np.asarray([a], float)])


def limb(P, Wd, n=9):
    """centre line control points + half widths -> left side, right side polylines (root -> end)"""
    P = np.asarray(P, float); C = spl(P, n=n); C = resample(C, .02)
    T = np.gradient(C, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-9; Nn = np.c_[-T[:, 1], T[:, 0]]
    s = np.linspace(0, 1, len(C)); w = np.interp(s, np.linspace(0, 1, len(Wd)), Wd)
    k = np.hanning(15); k /= k.sum(); w = np.convolve(np.r_[[w[0]] * 7, w, [w[-1]] * 7], k, 'same')[7:-7]
    def side(Q):                                                         # drop the loop on the inside of a sharp bend
        k_ = np.r_[True, np.sum(np.diff(Q, axis=0) * T[1:], 1) > 0]
        if k_.all(): return Q
        Q = Q[k_]
        for _ in range(3):                                              # cut the crossing itself
            A, B = Q[:-1], Q[1:]; n_ = len(A); dA = B - A
            ii, jj = np.triu_indices(n_, 2); den = dA[ii, 0] * dA[jj, 1] - dA[ii, 1] * dA[jj, 0]; den = np.where(np.abs(den) < 1e-12, 1e-12, den); d_ = A[jj] - A[ii]
            u = (d_[:, 0] * dA[jj, 1] - d_[:, 1] * dA[jj, 0]) / den; v = (d_[:, 0] * dA[ii, 1] - d_[:, 1] * dA[ii, 0]) / den
            h_ = np.nonzero((u >= 0) & (u <= 1) & (v >= 0) & (v <= 1))[0]
            if not len(h_): break
            q_ = h_[np.argmax(jj[h_] - ii[h_])]; X = A[ii[q_]] + dA[ii[q_]] * u[q_]; Q = np.vstack([Q[:ii[q_] + 1], X, Q[jj[q_] + 1:]])
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(Q, axis=0).T))]; t = np.linspace(0, d[-1], len(C)); Q = np.c_[np.interp(t, d, Q[:, 0]), np.interp(t, d, Q[:, 1])]
        m = np.hanning(7); m /= m.sum(); e = np.vstack([[Q[0]] * 3, Q, [Q[-1]] * 3]); return np.c_[np.convolve(e[:, 0], m, 'valid'), np.convolve(e[:, 1], m, 'valid')]
    return side(C + Nn * w[:, None]), side(C - Nn * w[:, None]), C, T


def ell(cx, cy, rx, ry=None, rot=0., n=40, sq=2.0):
    ry = rx if ry is None else ry; a = np.linspace(0, 2 * pi, n, endpoint=False); c, s = np.cos(a), np.sin(a)
    x = np.sign(c) * np.abs(c) ** (2 / sq) * rx; y = np.sign(s) * np.abs(s) ** (2 / sq) * ry
    cr, sr = math.cos(rot), math.sin(rot); return np.c_[cx + x * cr - y * sr, cy + x * sr + y * cr]


# ================================================================ cast
SK = dict(skin='#fbe3cf', skin2='#e3a992')
CAST = dict(
    gojo=dict(skin='#fdebdc', skin2='#e6b6a2', hair='#f4f0ee', hair2='#b9b3c6', cloth='#474054', cloth2='#2c2638', pant='#3d3748', pant2='#272130', shoe='#26212c', shoe2='#17131b', eye='#69c7f2', tall=1.06, hairl='#27394a'),
    yuji=dict(skin='#fbdcc2', skin2='#dfa083', hair='#f0a596', hair2='#bd6f70', cloth='#3c3d58', cloth2='#26253a', pant='#3c3d58', pant2='#26253a', shoe='#c2352f', shoe2='#82202a', eye='#9a5a38', tall=1.0, hairl='#3a1719'),
    megumi=dict(skin='#fae3d3', skin2='#d9a79a', hair='#2b2c3d', hair2='#191825', cloth='#3b3f5e', cloth2='#26263c', pant='#5d5a52', pant2='#3d3a3c', shoe='#4b3430', shoe2='#2e1f22', eye='#3f8a86', tall=1.0, hairl=LINE),
    nobara=dict(skin='#fde2cd', skin2='#e2a58c', hair='#d98245', hair2='#9c4d33', cloth='#393b5a', cloth2='#252439', pant='#3b2f36', pant2='#241b24', shoe='#7a4430', shoe2='#4d2823', eye='#c8742f', tall=.97, hairl='#3a1a12'),
    nanami=dict(skin='#f8dcc4', skin2='#d7a286', hair='#f0d588', hair2='#bd9552', cloth='#e9dcc4', cloth2='#b9a38f', pant='#e2d4bb', pant2='#b19a86', shoe='#6b4733', shoe2='#432a24', eye='#6f7a4a', tall=1.08, hairl='#3f2a14'),
    todo=dict(skin='#e9b48e', skin2='#b97757', hair='#2c2730', hair2='#1a161d', cloth='#e9b48e', cloth2='#b97757', pant='#343a55', pant2='#212439', shoe='#2b2530', shoe2='#18141b', eye='#6b4a34', tall=1.1, hairl=LINE),
    panda=dict(skin='#f7f4ee', skin2='#bdb9c4', hair='#2e2b33', hair2='#1b1820', cloth='#f7f4ee', cloth2='#bdb9c4', pant='#2e2b33', pant2='#1b1820', shoe='#2e2b33', shoe2='#1b1820', eye='#2a2026', tall=1.0, hairl=LINE),
    sukuna=dict(skin='#f9d8bf', skin2='#d8957d', hair='#f2a397', hair2='#b9646c', cloth='#f3ede2', cloth2='#b9aebb', pant='#f3ede2', pant2='#b9aebb', shoe='#f3ede2', shoe2='#b9aebb', eye='#c1272d', tall=.92, hairl='#3a1719'),
)

FACE_R = [(0, -.98), (.42, -.93), (.76, -.70), (.94, -.34), (.985, .02), (.972, .30), (.87, .56), (.60, .775), (.27, .875), (0, .895)]


def face_pts(puff=0., chin=0.):
    R = np.array(FACE_R, float); R[5:8, 0] += np.array([.02, .035, .03]) * (1 + puff); R[-2:, 1] += chin
    L = R[::-1][1:-1] * [-1, 1]
    return spl(np.vstack([R, L]), True, n=8)


class Head:
    """head node + 2.5-D turn.  h(P, z) -> px.  z: number | 'd' (on the dome)"""
    def __init__(self, node, yaw=0., pitch=0., t=0., wind=(0., 0.), wamp=.0, gust=0.):
        self.n = node; self.yaw, self.pitch = yaw, pitch; self.lid = None; self.mouth = None; self.ear = (0., 0.); self.mb = 0.; self.brow = 0.; self.vein = 0.; self.t, self.wind, self.wamp, self.gust = t, wind, wamp, gust
    def dome(self, P):
        P = np.asarray(P, float); return .80 * np.sqrt(np.clip(1.06 - (P[..., 0] / 1.02) ** 2 - ((P[..., 1] + .05) / .98) ** 2, 0, 2))
    def __call__(self, P, z=0., loose=0.):
        P = np.atleast_2d(np.asarray(P, float))
        if isinstance(z, str): z = self.dome(P) + (float(z[1:]) if len(z) > 1 else 0.)
        Q = P.copy()
        if loose:
            r = np.hypot(P[:, 0], P[:, 1] + .12); w = np.clip((r - .92) / .75, 0, 1.2) ** 1.25 * loose; an = np.arctan2(P[:, 1], P[:, 0])
            k = self.gust + self.wamp * np.sin(self.t * .30 + an * 2.2) + self.wamp * .5 * np.sin(self.t * .53 + an * 4.1 + 1.)
            Q = Q + np.outer(w * k, self.wind)
        sy, cy, sp, cp = math.sin(self.yaw), math.cos(self.yaw), math.sin(self.pitch), math.cos(self.pitch)
        x = Q[:, 0] * cy + z * sy; zz = -Q[:, 0] * sy + z * cy; y = Q[:, 1] * cp + zz * sp
        return self.n(np.c_[x, y])


# ------------------------------------------------------------ hair: individually cut locks on the cranium
CR = (0., -.12, 1.0, .94)                                        # cranium ellipse cx cy rx ry


def root(ang, k=1.0):
    """point on the cranium, ang degrees clockwise from straight up"""
    a = math.radians(ang); return np.array([CR[0] + math.sin(a) * CR[2] * k, CR[1] - math.cos(a) * CR[3] * k])


def lock(ang, hw, L, lean=0., bend=.0, bulge=.07, k=.93):
    """hw half root width in degrees, L length beyond the root, lean degrees (+ clockwise), bend sideways sweep"""
    a, b = root(ang - hw, k), root(ang + hw, k); m = root(ang, k); d = math.radians(ang * .96 + lean)
    tip = m + np.array([math.sin(d), -math.cos(d)]) * L
    return dict(a=a, tip=tip, b=b, bend=bend, bulge=bulge)


def lk(a, tip, b, bend=0., bulge=.08): return dict(a=np.array(a, float), tip=np.array(tip, float), b=np.array(b, float), bend=bend, bulge=bulge)


def draw_lock(d, h, q, c, z=0., loose=1., w=1., shade=.0, sside=1, hi=None, lines=1, seed=0, open_root=True, zroot=None, col=None, sh=None, deep=0.):
    """one lock: fill, sliver of shadow on one side (shade 0..1 of its width), open root, a hair line or two"""
    F = flame(q['a'], q['tip'], q['b'], q['bulge'], q['bend'], n=13); n = len(F); col = col or c['hair']; sh = sh or c['hair2']
    Z = z if zroot is None else np.interp(np.linspace(0, 1, n), [0, .5, 1], [0, 1, 0]) * 0 + z
    if isinstance(z, str): Z = z
    P = h(F, Z, loose); A, B = F[:13], F[12:][::-1]                # root->tip sides
    shd = []
    if shade > 0:
        S1, S0 = (B, A) if sside > 0 else (A, B); inner = S1 * (1 - shade) + S0 * shade
        t = np.linspace(0, 1, 13)[:, None]; inner = S1 * (1 - shade * (1 - t ** 2.2)) + S0 * shade * (1 - t ** 2.2)
        shd.append(h(np.vstack([S1, inner[::-1]]), Z, loose))
    gp = ()
    if open_root:
        per = np.hypot(*np.diff(P, axis=0).T).sum(); rt = np.hypot(*(P[0] - P[-1])); tot = per + rt
        gp = [(1 - rt / tot - .02, .02)] if not deep else [(1 - rt / tot - deep, deep)]
    d.part(P, col, sh, shade=shd, w=w, ink=c['hairl'], seed=seed, gaps=gp)
    ax = q['tip'] - (q['a'] + q['b']) / 2
    for i in range(lines):
        u = .34 + .30 * (i % 2) if lines > 1 else .46; C = A * (1 - u) + B * u
        i0 = 1 + (seed + i) % 2; L = C[i0:i0 + 6 + (seed * 3 + i) % 3]
        d.stroke(h(spl(L, n=5), Z if isinstance(Z, str) else z, loose), .34 + .12 * ((seed + i) % 3) / 2, c['hairl'], taper=(.45, .5), seed=seed * 3 + i)


def cap_fill(d, h, c, line, col=None, sh=None, z0=0., kx=.955):
    """cranium under the locks: hides their roots, closed below by the hairline (right -> left)"""
    an = np.linspace(-118, 118, 40); O = np.array([root(a, kx) for a in an]); L = np.asarray(line, float)
    P = np.vstack([O, L]); Z = np.r_[np.full(len(O), z0), h.dome(L) + .02]
    d.part(h(P, Z), col or c['hair'], sh or c['hair2'], w=0)
    return h(P, Z)


# ================================================================ heads
def skin_shape(d, h, c, expr, seed=3, shade_top=.0, chin=0., puff=0., cheekshadow=True):
    F = face_pts(puff, chin); P = h(F, 0.)
    sh = []
    if cheekshadow:                                                        # the far, lower side turns from the light
        sh.append(h(spl([(1.1, -.2), (.93, .20), (.84, .46), (.60, .67), (.24, .77), (-.2, .80), (-.5, .77), (-.2, 1.1), (1.2, 1.0)], True, n=8, corners=(0, 6)), 0.))
    if shade_top: sh.append(h(np.vstack([[(-1.2, -1.2)], [(1.2, -1.2)], spl([(1.2, shade_top - .02), (.5, shade_top + .05), (0, shade_top + .08), (-.5, shade_top + .05), (-1.2, shade_top - .02)], n=8)]), 'd'))
    d.part(P, c['skin'], c['skin2'], shade=sh, w=.92, ink=SKINO, seed=seed)
    return P


def ear(d, h, c, sgn, y=.20, s=1.0, dx=0.):
    E = spl([(sgn * (.93 + dx), y - .17 * s), (sgn * (1.06 + dx), y - .21 * s), (sgn * (1.16 + dx), y - .08 * s), (sgn * (1.11 + dx), y + .10 * s), (sgn * (.98 + dx), y + .19 * s), (sgn * (.90 + dx), y + .1 * s)], True, n=8)
    P = h(E, -.1); d.part(P, c['skin'], c['skin2'], shade=[h(E + [sgn * .02, .10 * s], -.1)] if sgn > 0 else [h(E * [1, 1] + [.05 * 0, .13 * s], -.1)], w=.8, ink=SKINL, seed=5)
    d.stroke(h(spl([(sgn * (1.08 + dx), y - .10 * s), (sgn * (1.02 + dx), y - .10 * s), (sgn * (1.01 + dx), y + .02 * s)], n=8), -.1), .42, SKINL)


def sweat(d, h, x=.86, y=-.30, s=1.):
    o = h([(x, y)], .5)[0]; u = h.n.s * s; P = spl(np.array([(0, -.17), (.07, -.02), (.085, .07), (0, .15), (-.085, .07), (-.07, -.02)]) * u + o, True, n=7, corners=(0,))
    d.part(P, '#d5eefb', '#9fcbe6', w=.5, ink='#41698a', tip=False); d.detail(ell(o[0] - .03 * u, o[1] + .05 * u, .018 * u, .04 * u), '#ffffff', sh='#ffffff')


def vein(d, h, x=.50, y=-.50, s=1.):
    """the little cross of irritation"""
    o = h([(x, y)], 'd.2')[0]; u = h.n.s * .16 * s
    for k in range(4):
        a = k * pi / 2 + .5; c, q = math.cos(a), math.sin(a); L = np.array([(.28, 1.05), (.30, .34), (1.05, .28)]); L = np.c_[L[:, 0] * c - L[:, 1] * q, L[:, 0] * q + L[:, 1] * c] * u + o
        d.stroke(spl(L, n=7), 1.0, '#c8232b', taper=(.12, .12), tmin=.4)


def blush(d, h, c, x=.60, y=.44, a=.5, w=.17):
    col = _mix(c['skin'], '#f08a86', a)
    for s in (-1, 1):
        d.detail(h(ell(s * x, y, w, w * .52), 'd'), col)
        for k in (-1, 0, 1): d.stroke(h([(s * x + k * .075 + .03, y - .05), (s * x + k * .075 - .03, y + .05)], 'd'), .26, '#d8665f')


def closed_eye(d, h, x, y, w=.15, sag=.07, sgn=1, lash=True, col=LINE, wt=.72, seed=0):
    """sleeping eye: a soft bowl with two lashes"""
    L = spl([(x - w, y - .01), (x - w * .5, y + sag * .8), (x + w * .1, y + sag), (x + w * .65, y + sag * .7), (x + w, y - .015)], n=8)
    d.stroke(h(L, 'd'), wt, col, taper=(.25, .25), seed=seed, tmin=.3)
    if lash:
        o = sgn
        d.stroke(h([(x + o * w * .86, y + .015), (x + o * w * 1.28, y + .045)], 'd'), .42, col); d.stroke(h([(x + o * w * .52, y + .058), (x + o * w * .82, y + .112)], 'd'), .38, col)


def open_eye(d, h, x, y, c, sgn, w=.15, hh=.20, look=(0., 0.), lid=0., iris=None, lash=True, sc='#fbf8f2', pup=1.0):
    """sharp little JJK eye, chibi size.  lid 0..1 droops"""
    iris = iris or c['eye']
    top = y - hh * (1 - lid * .9); E = spl([(x - sgn * w, y + .02), (x - sgn * w * .6, top + .03), (x + sgn * w * .2, top - .005), (x + sgn * w, top + hh * .30), (x + sgn * w * .9, y + hh * .45), (x + sgn * w * .2, y + hh * .82), (x - sgn * w * .7, y + hh * .55)], True, n=7)
    Q = h(E, 'd'); d._knock(Q); d.fl.fill(Q, sc); d.sc.fill(Q, _mix(sc, '#6a6a8a', .35)); d.mk.fill(Q, '#000000')
    ix, iy = x + look[0] + sgn * .01, y + look[1]
    d.detail(h(ell(ix, iy + .02, w * .70, hh * .86), 'd'), iris, within=Q)
    d.detail(h(ell(ix, iy - .05, w * .70, hh * .55), 'd'), _mix(iris, '#1a1018', .55), within=Q)
    d.detail(h(ell(ix, iy + .02, w * .34 * pup, hh * .46 * pup), 'd'), '#1a1018', within=Q)
    d.detail(h(ell(ix - .045, iy - .075, .042, .05), 'd'), '#ffffff', within=Q, sh='#ffffff')
    T = spl([(x - sgn * w * 1.05, y + .0), (x - sgn * w * .6, top + .02), (x + sgn * w * .25, top - .015), (x + sgn * w * 1.02, top + hh * .30), (x + sgn * w * 1.25, top + hh * .30)], n=8)
    d.stroke(h(T, 'd'), 1.0, LINE, taper=(.3, .12), tmin=.3)
    d.stroke(h(spl([(x - sgn * w * .35, y + hh * .80), (x + sgn * w * .3, y + hh * .84), (x + sgn * w * .75, y + hh * .62)], n=6), 'd'), .34, LINE)


def brow(d, h, x, y, sgn, ang=0., w=.17, wt=.75, col=LINE, arch=.03):
    c, s = math.cos(ang), -math.sin(ang)
    L = np.array([(-w, .0), (-w * .2, -arch), (w, .01)]) @ np.array([[c, s], [-s, c]]) * [sgn, 1] + [x, y]
    d.stroke(h(spl(L, n=8), 'd'), wt, col, taper=(.12, .45), tmin=.25)


def mouth(d, h, kind, c, x=0., y=.50, t=0., k=1.0):
    S = lambda L, w=.6, **q: d.stroke(h(spl(np.asarray(L) * k + [x, y], n=8), 'd'), w, '#3a1418', **q)
    if kind == 'flat': S([(-.10, .0), (0, .012), (.10, -.004)], .62)
    elif kind == 'smile': S([(-.13, -.03), (-.05, .035), (.05, .035), (.13, -.035)], .66)
    elif kind == 'cat': S([(-.15, -.02), (-.09, .035), (-.01, -.01)], .6); S([(-.01, -.01), (.07, .035), (.14, -.025)], .6)
    elif kind == 'frown': S([(-.12, .035), (-.02, -.015), (.12, .03)], .66)
    elif kind == 'wave': S([(-.15, .0), (-.08, -.03), (0, .02), (.08, -.03), (.15, .01)], .62)
    elif kind in ('open', 'laugh', 'gape', 'snore'):
        shp = dict(open=[(-.13, -.06), (0, -.085), (.13, -.06), (.10, .08), (0, .135), (-.10, .08)], laugh=[(-.24, -.08), (0, -.05), (.24, -.08), (.19, .11), (0, .21), (-.19, .11)],
                   gape=[(-.15, -.12), (0, -.15), (.15, -.12), (.17, .12), (0, .19), (-.17, .12)], snore=[(-.17, -.055), (-.05, -.085), (.10, -.07), (.17, -.02), (.13, .085), (0, .135), (-.13, .085)])[kind]
        sq = 1 + (.10 * math.sin(t) if kind == 'snore' else 0) + (.34 * getattr(h, 'mb', 0.) if kind in ('snore', 'open') else 0)
        M = spl(np.array(shp) * k * [1, sq] + [x, y], True, n=8); Q = h(M, 'd')
        d.part(Q, '#8c2f35', '#8c2f35', hi=[(h(ell(x + .02 * k, y + .13 * k, .13 * k, .09 * k), 'd'), '#ee8b84')], w=.58, ink='#3a1418', tip=False)
        if kind in ('laugh',): d.detail(h(spl(np.array([(-.22, -.08), (0, -.05), (.22, -.08), (.17, -.02), (0, .005), (-.17, -.02)]) * k + [x, y], True, n=6), 'd'), '#fffaf2', within=Q, sh='#fffaf2')


def drool(d, h, x, y, L=.16, t=0.):
    """runs down the screen whatever way the head lies"""
    o = h([(x, y)], 'd')[0]; u = h.n.s; L = L * (1 + .12 * math.sin(t)); sg = -1 if h.n.flip else 1
    q = lambda P: np.array(P, float) * u + o
    P = spl([(0, -.015), (.03, L * .5), (.048, L), (0, L * 1.25), (-.045, L), (-.018, L * .5)], True, n=7)
    d.part(q(P), '#dff2fb', '#a9d2e8', w=.34, ink='#6b93b0', tip=False); d.detail(q(ell(-.012, L * .9, .012, .03)), '#ffffff', sh='#ffffff')


# ------------------------------------------------------------ per-character heads
def head_gojo(d, h, c, expr, t, seed=0):
    ear(d, h, c, -1); ear(d, h, c, 1)
    skin_shape(d, h, c, expr, puff=.3)
    d.shade(h(np.vstack([spl([(-1.1, .24), (-.5, .36), (-.16, .33), (0, .27), (.16, .33), (.5, .36), (1.1, .24)], n=8), [(1.1, -.6), (-1.1, -.6)]]), 'd'))      # band's shadow on the cheeks
    d.detail(h(ell(0, .345, .02, .012), 'd'), c['skin'])
    d.stroke(h(spl([(-.045, .385), (.0, .405), (.05, .38)], n=6), 'd'), .42, SKINL)                                   # nose
    if h.mouth: mouth(d, h, h.mouth, c, .03, .53)
    elif expr in ('sleep', 'snore'):
        mouth(d, h, 'snore', c, .03, .535, t * .0); drool(d, h, .175, .555, .13)
    else: mouth(d, h, {'smile': 'smile', 'shock': 'gape', 'dot': 'flat', 'laugh': 'laugh'}.get(expr, 'smile'), c, .02, .56)
    blush(d, h, c, .62, .46, .42, .15)
    K = [(-107, 10, .46, -6, -.08, .4, -1, 0), (103, 10, .50, 6, .10, .55, 1, 0), (-91, 11, .66, 8, -.12, .0, 1, 1), (88, 11, .70, -4, .14, .45, 1, 1),
         (-72, 12, .82, -2, -.10, .0, 1, 1), (70, 12, .60, 10, .10, .5, 1, 1), (-30, 10, .62, -14, .10, 0, 1, 0), (16, 9, .70, 12, .08, 0, 1, 0), (-53, 14, .98, 4, -.12, .0, 1, 2), (52, 13, .96, -6, .14, .42, 1, 2),
         (-8, 12, .80, -10, -.10, .0, 1, 1), (33, 13, 1.02, -2, .12, .38, 1, 2), (-31, 15, 1.10, 6, -.10, .0, 1, 2), (8, 15, 1.16, 4, .10, .30, 1, 2), (-64, 9, .50, 6, -.16, 0, 1, 0), (44, 8, .52, 10, .16, .4, 1, 0)]
    for i, (an, hw, L, le, be, sd, ss, ln) in enumerate(K): draw_lock(d, h, lock(an, hw, L, le, be, .085), c, 0., 1., .80, shade=sd, sside=ss, lines=ln, seed=seed + 20 + i)
    line = spl([(.96, .20), (.60, -.02), (0, -.12), (-.60, -.02), (-.96, .20)], n=8)
    cp = cap_fill(d, h, c, line)
    d.shade(h(np.vstack([[(-1.2, .3)], flames((-1.04, -.02), [((-.84, -.36), (-.80, -.10), .0), ((-.44, -.40), (-.36, -.16), 0), ((.02, -.50), (.10, -.18), 0), ((.40, -.40), (.46, -.14), 0), ((.78, -.62), (.94, -.30), 0)], .05), [(1.2, -.3), (1.2, .4)]]), .3), within=cp)
    for a, m, b, w in [((-.70, -.34), (-.86, -.52), (-1.06, -.62), .42), ((-.46, -.42), (-.58, -.70), (-.74, -.98), .5), ((-.20, -.46), (-.28, -.80), (-.40, -1.14), .38), ((.02, -.50), (.06, -.84), (.06, -1.20), .46), ((.30, -.44), (.40, -.74), (.48, -1.02), .40), ((.56, -.36), (.74, -.58), (.94, -.80), .46), ((.76, -.24), (.96, -.32), (1.14, -.40), .36)]:
        d.stroke(h(spl([a, m, b], n=10), .1, .6), w, c['hairl'], taper=(.5, .5), seed=int(abs(a[0]) * 99) % 97)
    # blindfold
    top = [(-1.03, -.12), (-.72, -.22), (-.30, -.275), (.10, -.285), (.50, -.255), (.82, -.19), (1.03, -.10)]
    bot = [(1.03, .22), (.80, .305), (.48, .335), (.22, .30), (.02, .225), (-.20, .29), (-.48, .335), (-.80, .30), (-1.03, .22)]
    Bp = spl(np.vstack([top, bot]), True, n=7, corners=(0, 6, 7, 15)); B = h(Bp, 'd')
    kn = flames((1.0, -.04), [((1.46, .08), (1.06, .10), .10, .16), ((1.36, .40), (1.0, .20), .12, .18)])                    # knot tails
    d.part(h(np.vstack([kn, [(.9, .1)]]), -.2, .6), '#37313d', '#231e29', w=.9, seed=seed + 9)
    d.part(B, '#403a47', '#272230', shade=[h(spl([(-1.1, .12), (-.5, .17), (.0, .05), (.30, .10), (.62, -.02), (1.1, -.16), (1.1, .5), (-1.1, .5)], True, n=8, corners=(5, 6, 7)), 'd')],
           hi=[(h(spl([(-.80, -.12), (-.42, -.19), (-.16, -.19), (-.40, -.10), (-.66, -.02)], True, n=7, corners=(0, 2)), 'd'), '#5d5566')], w=.95, seed=seed + 2)
    for L, w in [([(-.96, .10), (-.62, .10), (-.30, .16)], .4), ([(.02, .215), (-.02, .12), (-.16, .06)], .4), ([(.06, .215), (.14, .13), (.34, .10)], .4), ([(.98, .02), (.70, .06), (.50, .16)], .4), ([(-.92, -.04), (-.70, -.02)], .3)]:
        d.stroke(h(spl(L, n=8), 'd'), w, LINE, taper=(.1, .55))


def head_yuji(d, h, c, expr, t, seed=0):
    cap = spl([(-.98, .12), (-1.03, -.25), (-.94, -.62), (-.5, -.98), (0, -1.06), (.5, -.98), (.94, -.62), (1.03, -.25), (.98, .12)], n=8)
    d.part(h(np.vstack([cap, [(.5, .2), (-.5, .2)]]), 0.), '#5a3d3f', '#3c2730', w=.8, seed=seed)          # undercut
    ear(d, h, c, -1); ear(d, h, c, 1); skin_shape(d, h, c, expr, shade_top=-.12)
    for s in (-1, 1):                                                                                           # sideburn of the undercut
        Q = h(spl([(s * .80, -.45), (s * 1.0, -.45), (s * 1.0, .10), (s * .91, .13), (s * .86, -.10)], True, n=6, corners=(0, 1, 2, 3)), .05); d.detail(Q, '#5a3d3f', sh='#3c2730'); d.stroke(h(spl([(s * .82, -.40), (s * .87, -.10), (s * .915, .12)], n=6), .05), .4, LINE)
    for s in (-1, 1): d.stroke(h(spl([(s * .30, .385), (s * .42, .40), (s * .53, .37)], n=6), 'd'), .4, '#b86a5c')       # Sukuna's closed second eyes
    face_expr(d, h, c, expr, t, brow_col='#8a4a48', bw=.8, mouthk=dict(sleep='open', smile='laugh'))
    K = [(-84, 11, .34, 4, -.10, 0, 1, 0), (82, 11, .36, -4, .12, .5, 1, 0), (-64, 12, .50, -2, -.12, 0, 1, 1), (64, 11, .48, 4, .12, .5, 1, 1), (-20, 9, .46, -14, -.06, 0, 1, 0), (28, 9, .44, 10, .08, .4, 1, 0),
         (-44, 13, .62, 6, -.12, 0, 1, 1), (46, 13, .64, -4, .12, .42, 1, 1), (-28, 10, .56, 10, .10, 0, 1, 0), (4, 12, .58, -6, -.10, 0, 1, 1), (22, 13, .70, 6, .10, .36, 1, 1), (-13, 14, .72, 4, -.08, .0, 1, 1), (-74, 8, .30, 10, -.12, 0, 1, 0), (56, 8, .34, 8, .12, .4, 1, 0)]
    for i, (an, hw, L, le, be, sd, ss, ln) in enumerate(K): draw_lock(d, h, lock(an, hw, L, le, be, .08), c, 0., .9, 1.0, shade=sd, sside=ss, lines=ln, seed=seed + 20 + i)
    line = spl([(.94, -.20), (.74, -.36), (.36, -.44), (0, -.46), (-.36, -.44), (-.74, -.36), (-.94, -.20)], n=7)
    cp = cap_fill(d, h, c, line)
    for i, (a_, t_, b_, be, sd) in enumerate([((-.97, -.34), (-.80, -.02), (-.62, -.50), .10, .0), ((.60, -.50), (.84, -.02), (.97, -.34), -.10, .4), ((-.72, -.56), (-.50, -.10), (-.26, -.56), .10, 0), ((.22, -.58), (.60, -.12), (.74, -.54), -.10, .4), ((-.34, -.60), (-.16, -.16), (.06, -.60), .14, 0), ((.02, -.62), (.28, -.08), (.38, -.58), -.12, .4)]):
        draw_lock(d, h, lk(a_, t_, b_, be, .06), c, 'd.07', .0, .95, shade=.25 + sd, sside=1, lines=1, seed=seed + 50 + i, deep=.07)
    d.part(h(spl([(-.95, -.40), (-.5, -.52), (0, -.57), (.5, -.52), (.95, -.40), (.6, -.80), (0, -.90), (-.6, -.80)], True, n=8), 'd.07'), c['hair'], c['hair2'], w=0)
    for a, m, b, w in [((-.58, -.60), (-.70, -.78), (-.86, -.92), .42), ((-.22, -.66), (-.30, -.92), (-.40, -1.10), .46), ((.06, -.66), (.06, -.94), (.02, -1.20), .40), ((.38, -.62), (.46, -.84), (.54, -1.04), .42), ((.66, -.54), (.80, -.68), (.94, -.80), .4)]:
        d.stroke(h(spl([a, m, b], n=8), .15, .6), w, c['hairl'], taper=(.5, .5))


def head_megumi(d, h, c, expr, t, seed=0):
    ear(d, h, c, -1); ear(d, h, c, 1); skin_shape(d, h, c, expr, shade_top=-.04)
    face_expr(d, h, c, expr, t, brow_col=LINE, bw=.8, lash=True, lid=.38, browang=.16, mouthk=dict(sleep='flat', smile='smile'))
    K = [(-122, 9, .56, -2, -.06, 0, 1, 0), (120, 9, .58, 4, .08, 0, 1, 0), (-104, 11, .82, 6, -.10, 0, 1, 0), (104, 11, .78, -6, .10, 0, 1, 0), (-84, 12, .70, -8, -.10, 0, 1, 0), (83, 12, .92, 2, .12, 0, 1, 0), (-22, 10, .72, -16, -.08, 0, 1, 0), (24, 10, .64, 14, .08, 0, 1, 0),
         (-68, 13, 1.04, 2, -.10, 0, 1, 1), (64, 13, .80, -4, .12, 0, 1, 1), (-34, 11, .66, 12, .10, 0, 1, 0), (6, 12, .86, -8, -.08, 0, 1, 1), (-46, 14, 1.06, 2, -.12, 0, 1, 1), (44, 14, 1.12, 2, .12, 0, 1, 1), (-14, 14, 1.06, 4, -.08, 0, 1, 1), (22, 13, 1.0, 2, .08, 0, 1, 1), (-94, 7, .46, 10, -.14, 0, 1, 0), (74, 7, .46, 8, .14, 0, 1, 0)]
    for i, (an, hw, L, le, be, sd, ss, ln) in enumerate(K):
        q = lock(an, hw, L, le, be, .06); draw_lock(d, h, q, c, 0., .9, 1.0, lines=0, seed=seed + 20 + i)
        if L > .62:                                                     # cool light slivers
            F = flame(q['a'], q['tip'], q['b'], q['bulge'], q['bend'], n=13); A, B = F[:13], F[12:][::-1]
            u0, u1 = (.30, .52) if i % 2 else (.46, .66); t = np.linspace(0, 1, 13)[:, None]; m = .5 + .0 * t; g = np.sin(np.clip((t - .12) / .62, 0, 1) * pi)
            c0 = A * (1 - u0) + B * u0; c1 = A * (1 - (u0 + (u1 - u0) * g)) + B * (u0 + (u1 - u0) * g)
            d.detail(h(np.vstack([c0[1:10], c1[1:10][::-1]]), 0., .9), '#4d5b88', sh='#33395c')
    cap_fill(d, h, c, spl([(.94, .10), (.74, -.20), (.36, -.36), (0, -.40), (-.36, -.36), (-.74, -.20), (-.94, .10)], n=7))
    for i, (a_, t_, b_, be) in enumerate([((-1.0, -.10), (-.90, .52), (-.74, -.26), .06), ((.74, -.26), (.92, .56), (1.0, -.08), -.06), ((-.86, -.40), (-.66, .24), (-.46, -.42), .10), ((.44, -.44), (.72, .20), (.86, -.38), -.10), ((-.52, -.50), (-.36, .06), (-.14, -.50), .10), ((.12, -.50), (.40, .12), (.52, -.48), -.10), ((-.22, -.54), (.04, .30), (.22, -.54), -.10)]):
        q = lk(a_, t_, b_, be, .06); draw_lock(d, h, q, c, 'd.08', .25, .95, lines=0, seed=seed + 50 + i, deep=.07)
        F = flame(q['a'], q['tip'], q['b'], q['bulge'], q['bend'], n=13); A, B = F[:13], F[12:][::-1]
        d.detail(h(np.vstack([(A * .70 + B * .30)[3:10], (A * .48 + B * .52)[3:10][::-1]]), 'd.09', .25), '#46527c', sh='#2d3252')
    d.part(h(spl([(-.99, -.20), (-.6, -.40), (0, -.49), (.6, -.40), (.99, -.20), (.8, -.66), (0, -.90), (-.8, -.66)], True, n=8), 'd.08'), c['hair'], c['hair2'], w=0)
    for q in [[(-.62, -.54), (-.72, -.78), (-.80, -.98), (-.58, -.80), (-.48, -.58)], [(-.10, -.62), (-.16, -.92), (-.16, -1.10), (.02, -.92), (.06, -.64)], [(.40, -.58), (.54, -.82), (.66, -1.0), (.62, -.76), (.54, -.54)], [(.74, -.40), (.94, -.54), (1.08, -.58), (.94, -.40), (.80, -.28)], [(-.86, -.34), (-1.02, -.42), (-1.10, -.42), (-.98, -.28), (-.84, -.22)]]:
        d.detail(h(spl(q, True, n=6, corners=(0, 2, 4)), .25, .4), '#4d5b88', sh='#33395c')


def head_nobara(d, h, c, expr, t, seed=0):
    dk = _mix(c['hair'], c['hair2'], .62)
    # back of the bob, ends cut in uneven locks
    for i, (a_, t_, b_, be) in enumerate([((-.80, .30), (-.64, 1.00), (-.30, .4), .06), ((.80, .30), (.62, 1.02), (.30, .4), -.06)]):
        draw_lock(d, h, lk(a_, t_, b_, be, .10), c, -.15, .35, 1., lines=0, seed=seed + 2 + i, col=dk, sh=c['hair2'])
    back = spl([(0, -1.12), (-.62, -1.02), (-1.06, -.60), (-1.19, -.05), (-1.15, .38), (-.9, .55), (.9, .55), (1.15, .38), (1.19, -.05), (1.06, -.60), (.62, -1.02)], True, n=7)
    d.part(h(back, -.15, .3), dk, c['hair2'], w=1.0, ink=c['hairl'], seed=seed, gaps=[(.40, .56)])
    skin_shape(d, h, c, expr, shade_top=-.14)
    face_expr(d, h, c, expr, t, brow_col='#6a3420', bw=.66, lash=True, browang=.10, mouthk=dict(sleep='cat', smile='laugh'))
    # side curtains
    for s, sd in ((-1, 0.), (1, .42)):
        oc = spl([(s * 1.02, -.62), (s * 1.17, -.22), (s * 1.215, .16), (s * 1.17, .48), (s * 1.02, .76), (s * .80, .90)], n=8); ic = spl([(s * .80, .90), (s * .93, .66), (s * .96, .34), (s * .92, .02), (s * .80, -.3)], n=8)
        tp = flames(tuple(oc[-12]), [((s * 1.06, .94), (s * .97, .72), s * .06, .06)], .05)
        Pq = np.vstack([oc, ic[1:]]); Q = h(Pq, .1, .25); seg = np.hypot(*np.diff(np.vstack([Q, Q[:1]]), axis=0).T); cum = np.cumsum(seg) / seg.sum()
        d.part(h(spl([(s * 1.10, .50), (s * 1.15, .70), (s * 1.08, .88), (s * .97, .78), (s * .92, .6)], True, n=6, corners=(2,)), .05, .25), dk, c['hair2'], w=.9, ink=c['hairl'], seed=seed + 15)
        d.part(Q, c['hair'], c['hair2'], shade=[h(np.vstack([ic, spl([(s * .9, -.3), (s * 1.06, .1), (s * 1.06, .42), (s * .80, .90)], n=6)[::-1]]), .1, .25)] + ([h(spl([(1.08, -.5), (1.12, .1), (1.04, .6), (.8, .9), (1.4, .9), (1.4, -.5)], True, n=6), .1, .25)] if s > 0 else []), w=1.0, ink=c['hairl'], seed=seed + 12 + s, gaps=[(.0, .06), (cum[-1] - .16, 1.0)])
        for L in ([(s * 1.06, -.2), (s * 1.10, .2), (s * 1.02, .60)], [(s * .98, .1), (s * 1.02, .4), (s * .94, .70)]): d.stroke(h(spl(L, n=6), .15, .25), .38, c['hairl'], taper=(.4, .4))
    # crown
    an = np.linspace(-112, 112, 40); O = np.array([root(a_, 1.09 + .05 * math.cos(math.radians(a_ * 1.6))) for a_ in an]); Ln = spl([(1.04, -.02), (.80, -.34), (.40, -.60), (0, -.52), (-.40, -.40), (-.80, -.22), (-1.04, .02)], n=7)
    P = np.vstack([O, Ln]); Z = np.r_[np.zeros(len(O)), h.dome(Ln) + .03]; n_ = len(P); kO = len(O) / n_
    Q = h(P, Z, .2); per = np.hypot(*np.diff(np.vstack([Q, Q[:1]]), axis=0).T); cum = np.cumsum(per) / per.sum(); kO = cum[len(O) - 1]
    d.part(Q, c['hair'], c['hair2'], w=1.05, ink=c['hairl'], seed=seed + 1, gaps=[(kO - .01, 1.0)])
    # fringe swept from the parting on the right
    for i, (a_, t_, b_, be, sd) in enumerate([((.30, -.78), (.80, .02), (.86, -.50), -.14, .35), ((.28, -.80), (.52, -.10), (.62, -.66), -.12, .30), ((-.94, -.42), (-.92, .22), (-.52, -.60), -.10, .0), ((-.74, -.56), (-.60, .10), (-.14, -.74), -.16, .22), ((-.36, -.70), (-.20, .02), (.16, -.82), -.18, .24), ((.02, -.78), (.16, -.12), (.36, -.84), -.14, .28)]):
        draw_lock(d, h, lk(a_, t_, b_, be, .07), c, 'd.06', .12, .95, shade=.20 + sd, sside=1, lines=1, seed=seed + 50 + i, deep=.09)
    d.part(h(spl([(-.98, -.30), (-.60, -.54), (-.20, -.66), (.24, -.72), (.60, -.60), (.96, -.36), (.9, -.66), (.5, -.94), (0, -1.02), (-.5, -.94), (-.9, -.66)], True, n=8), 'd.06'), c['hair'], c['hair2'], w=0)
    hi = [spl([(-.90, -.50), (-.50, -.70), (-.06, -.80), (.18, -.80), (.10, -.88), (-.20, -.92), (-.56, -.84), (-.84, -.66)], True, n=7, corners=(0, 3)), spl([(.42, -.80), (.70, -.66), (.90, -.44), (.88, -.56), (.70, -.76), (.5, -.88)], True, n=7, corners=(0, 2))]
    for q in hi: d.detail(h(q, .35), '#f6b673', sh='#c9743f')
    for L, w in [([(.26, -.80), (-.10, -.66), (-.46, -.42)], .40), ([(.26, -.84), (-.06, -.86), (-.50, -.70), (-.80, -.46)], .42), ([(.28, -.84), (.56, -.66), (.74, -.40)], .40), ([(.24, -.84), (.06, -1.02), (-.2, -1.08)], .42), ([(.28, -.88), (.40, -1.0), (.52, -1.04)], .34)]:
        d.stroke(h(spl(L, n=8), .4, .2), w, c['hairl'], taper=(.2, .5))


def head_nanami(d, h, c, expr, t, seed=0):
    back = spl([(-.98, .20), (-1.05, -.2), (-.96, -.64), (-.52, -1.02), (0, -1.10), (.52, -1.02), (.96, -.64), (1.05, -.2), (.98, .20)], n=8)
    d.part(h(np.vstack([back, [(.4, .3), (-.4, .3)]]), 0.), c['hair2'], _mix(c['hair2'], '#402a1a', .4), w=.9, ink=c['hairl'], seed=seed)
    ear(d, h, c, -1); ear(d, h, c, 1); skin_shape(d, h, c, expr, shade_top=-.34, chin=.02)
    # cheekbones
    for s in (-1, 1): d.stroke(h(spl([(s * .74, .42), (s * .66, .52), (s * .60, .62)], n=6), 'd'), .34, SKINL)
    d.stroke(h(spl([(-.03, .36), (.0, .385), (.05, .365)], n=6), 'd'), .38, SKINL)
    mouth(d, h, dict(sleep='flat', smile='smile', dot='flat', shock='open', read='flat', soft='smile').get(expr, 'flat'), c, .0, .60, k=.85)
    if expr in ('read', 'dot', 'sleep', 'smile', 'soft', 'shock'):
        for s in (-1, 1): brow(d, h, s * .42, -.10 if expr != 'shock' else -.2, s, .10, .18, .7, '#8a6a30')
    # 7:3, combed back; a few locks lift
    for i, (a_, t_, b_, be, sd) in enumerate([((-1.0, .02), (-1.12, .20), (-.96, -.30), -.06, 0), ((.96, -.30), (1.12, .18), (1.0, .02), .06, .4), ((-1.0, -.22), (-1.22, -.16), (-.94, -.60), -.16, 0), ((.94, -.58), (1.20, -.18), (1.0, -.22), .16, .4), ((-.96, -.50), (-1.20, -.60), (-.70, -.86), -.2, 0), ((.72, -.84), (1.16, -.66), (.96, -.50), .18, .4)]):
        draw_lock(d, h, lk(a_, t_, b_, be, .08), c, 0., .6, .95, shade=sd, lines=0, seed=seed + 20 + i)
    an = np.linspace(-84, 84, 30); O = np.array([root(a_, 1.06 + .10 * math.exp(-((a_ + 14) / 34.) ** 2)) for a_ in an])
    Ln = np.vstack([spl([(1.02, -.10), (.93, -.06), (.88, -.28), (.74, -.46), (.52, -.56), (.36, -.64)], n=7), spl([(.36, -.64), (.16, -.44), (-.18, -.38), (-.52, -.42), (-.78, -.34), (-.88, -.16), (-.93, .02), (-1.02, -.10)], n=7)[1:]])
    P = np.vstack([O, Ln]); Z = np.r_[np.zeros(len(O)), h.dome(Ln) + .04]
    d.part(h(P, Z, .3), c['hair'], c['hair2'], shade=[h(np.vstack([spl([(-1.3, -.2), (-.94, -.40), (-.5, -.54), (-.1, -.52), (.22, -.60), (.36, -.72)], n=8), spl([(.36, -.72), (.64, -.66), (.86, -.46), (1.3, -.2)], n=8), [(1.3, .5), (-1.3, .5)]]), .4, .3), h([(.30, -1.12), (.37, -.66), (.48, -.84), (.44, -1.08)], .3, .3), h(spl([(.70, -.98), (.84, -.60), (.98, -.20), (1.3, -.3), (1.0, -.9)], True, n=6), .1, .3)],
           hi=[(h(spl([(-.74, -.68), (-.40, -.90), (.06, -.98), (.20, -.92), (-.14, -.84), (-.58, -.62)], True, n=6, corners=(0, 3)), .3, .3), '#fff2c0')], w=1.0, ink=c['hairl'], seed=seed + 1)
    for L, w in [([(.36, -.68), (.06, -.80), (-.34, -1.0)], .46), ([(.24, -.54), (-.20, -.70), (-.66, -.84)], .42), ([(.0, -.42), (-.40, -.58), (-.82, -.66)], .40), ([(-.34, -.42), (-.62, -.50), (-.92, -.50)], .36), ([(.36, -.70), (.38, -.92), (.30, -1.14)], .48), ([(.50, -.60), (.70, -.74), (.80, -.92)], .38), ([(.70, -.48), (.86, -.52), (.98, -.62)], .34)]:
        d.stroke(h(spl(L, n=8), .35, .3), w, c['hairl'], taper=(.12, .5))
    # goggles: no arms, two round-cornered lenses
    for s in (-1, 1):
        G = h(spl([(s * .09, .06), (s * .16, -.045), (s * .46, -.07), (s * .70, -.02), (s * .74, .12), (s * .62, .30), (s * .34, .34), (s * .14, .24)], True, n=8), 'd.05')
        d.part(G, '#5f7a55', '#3d5340', shade=[h(spl([(s * .05, .20), (s * .4, .10), (s * .8, .16), (s * .8, .5), (s * .0, .5)], True, n=6), 'd.05')], w=.9, seed=seed + 6 + s, tip=False)
        d.detail(h(spl([(s * .20, .02) if s > 0 else (-.60, .0), (s * .20 + .12 if s > 0 else -.48, -.02), (s * .20 + .02 if s > 0 else -.58, .16), (s * .20 - .04 if s > 0 else -.64, .14)], True, n=5), 'd.05'), '#b9d09a', within=G)
        for q, w in (((.40, .02), .030), ((.56, .16), .026), ((.32, .20), .026), ((.48, .26), .02)): d.detail(h(ell(s * q[0], q[1], w, w * .8), 'd.05'), '#34463a', within=G)
    d.stroke(h(spl([(-.10, .06), (0, .025), (.10, .06)], n=6), 'd.08'), .8, LINE, taper=(0, 0))


def head_todo(d, h, c, expr, t, seed=0):
    # pineapple tail
    tail = flames((-.20, -1.0), [((-.62, -1.46), (-.26, -1.30), -.10, .2), ((-.42, -1.86), (-.10, -1.50), -.08, .18), ((.06, -2.02), (.12, -1.50), .02, .18), ((.52, -1.80), (.26, -1.30), .08, .18), ((.66, -1.42), (.20, -1.0), .10, .2)])
    d.part(h(tail, -.1, .6), c['hair'], c['hair2'], hi=[(h(spl([(-.1, -1.4), (-.06, -1.66), (.04, -1.86), (.04, -1.6)], True, n=5), 0., .6), '#4d4658')], w=1.0, seed=seed)
    ear(d, h, c, -1); ear(d, h, c, 1); skin_shape(d, h, c, expr, shade_top=-.30, chin=.04, puff=.4)
    face_expr(d, h, c, expr, t, brow_col=LINE, bw=1.25, browang=-.05, eyes=.8, mouthk=dict(sleep='open', smile='laugh'))
    cap = np.vstack([spl([(-1.0, .02), (-1.04, -.34), (-.92, -.72), (-.5, -1.02), (0, -1.09), (.5, -1.02), (.92, -.72), (1.04, -.34), (1.0, .02)], n=8), spl([(1.0, .02), (.93, -.02), (.90, -.28), (.70, -.46), (.36, -.50), (0, -.40), (-.36, -.50), (-.70, -.46), (-.90, -.28), (-.93, -.02), (-1.0, .02)], n=8)[1:]])
    Z = np.where(cap[:, 1] > -.52, h.dome(cap) + .03, 0.)
    d.part(h(cap, Z), c['hair'], c['hair2'], hi=[(h(spl([(-.74, -.62), (-.4, -.86), (-.08, -.94), (-.34, -.76), (-.62, -.54)], True, n=6, corners=(0, 2)), .3), '#4d4658')], w=1.0, seed=seed + 1)
    for L in ([(-.50, -.50), (-.34, -.80), (-.10, -1.04)], [(.0, -.42), (.0, -.8), (.0, -1.06)], [(.50, -.50), (.34, -.80), (.10, -1.04)], [(-.84, -.34), (-.66, -.72), (-.32, -1.0)], [(.84, -.34), (.66, -.72), (.32, -1.0)]): d.stroke(h(spl(L, n=8), .4), .36, '#5a5266', taper=(.4, .3))
    T = h(spl([(-.24, -1.06), (0, -1.0), (.24, -1.06), (.22, -1.16), (0, -1.12), (-.22, -1.16)], True, n=6), .2); d.part(T, '#c9a24e', '#8a6a3a', w=.6, tip=False)
    # the scar
    d.stroke(h(spl([(.62, -.42), (.56, -.1), (.60, .14), (.50, .40), (.42, .58)], n=8), 'd'), .62, '#a2543f', taper=(.15, .35)); d.stroke(h([(.47, .30), (.60, .36)], 'd'), .36, '#a2543f'); d.stroke(h([(.50, -.2), (.64, -.16)], 'd'), .36, '#a2543f')


def head_panda(d, h, c, expr, t, seed=0):
    for s in (-1, 1):
        E = spl([(s * .46, -.76), (s * .56, -1.10), (s * .82, -1.22), (s * 1.06, -1.04), (s * 1.06, -.74), (s * .84, -.50)], True, n=8); ea = h.ear[0 if s < 0 else 1]
        if ea: ce, se = math.cos(s * ea), math.sin(s * ea); E = (E - [s * .74, -.66]) @ np.array([[ce, se], [-se, ce]]) * [1 - .22 * abs(ea), 1.] + [s * .74, -.66]
        d.part(h(E, -.2), c['hair'], c['hair2'], hi=[(h(ell(s * .82 - .04, -.92, .14, .12), -.2), '#45404c')], w=1.0, seed=seed + s)
    F = spl([(0, -.98), (.5, -.92), (.92, -.62), (1.12, -.10), (1.15, .28), (.98, .66), (.6, .86), (0, .92), (-.6, .86), (-.98, .66), (-1.15, .28), (-1.12, -.10), (-.92, -.62), (-.5, -.92)], True, n=8)
    P = h(F); d.part(P, c['skin'], c['skin2'], shade=[h(spl([(1.3, -.2), (1.02, .24), (.84, .58), (.4, .76), (-.3, .80), (-.2, 1.3), (1.4, 1.2)], True, n=8, corners=(0, 4)))], w=1.05, seed=seed + 2)
    for s in (-1, 1):
        Q = spl([(s * .24, .06), (s * .34, -.18), (s * .58, -.24), (s * .76, -.04), (s * .74, .26), (s * .58, .42), (s * .36, .34)], True, n=8); d.part(h(Q, 'd'), c['hair'], c['hair2'], w=.5, seed=seed + 5 + s)
        if expr in ('sleep', 'snore', 'read'): closed_eye(d, h, s * .48, .10, .13, .06, s, False, '#f3efe8', .7)
        elif expr == 'shock': d.detail(h(ell(s * .48, .06, .15, .17), 'd'), '#fdfbf6', sh='#fdfbf6'); d.detail(h(ell(s * .48, .06, .035), 'd'), LINE)
        else: d.detail(h(ell(s * .47, .06, .10, .12), 'd'), '#fdfbf6', sh='#e6e2ea'); d.detail(h(ell(s * .46, .07, .06, .08), 'd'), '#241a1e'); d.detail(h(ell(s * .46 - .03, .035, .025), 'd'), '#ffffff')
    M = spl([(-.30, .42), (0, .30), (.30, .42), (.32, .62), (0, .74), (-.32, .62)], True, n=8); d.detail(h(M, 'd'), '#fffdf8', sh='#d8d4dc')
    d.part(h(spl([(-.13, .36), (0, .33), (.13, .36), (.07, .45), (0, .48), (-.07, .45)], True, n=7), 'd.1'), '#2a242c', '#1a161c', hi=[(h(ell(-.04, .375, .04, .016), 'd.1'), '#7a727e')], w=.4, tip=False)
    d.stroke(h([(0, .47), (0, .55)], 'd'), .5)
    if expr in ('sleep', 'snore'): mouth(d, h, 'open', c, 0., .635, k=.9); drool(d, h, .10, .66, .13)
    elif expr == 'shock': mouth(d, h, 'gape', c, 0, .66, k=.9)
    else: mouth(d, h, 'cat', c, 0.01, .565, k=1.2)


def head_sukuna(d, h, c, expr, t, seed=0):
    cap = spl([(-.98, .12), (-1.03, -.25), (-.94, -.62), (-.5, -.98), (0, -1.06), (.5, -.98), (.94, -.62), (1.03, -.25), (.98, .12)], n=8)
    d.part(h(np.vstack([cap, [(.5, .2), (-.5, .2)]]), 0.), '#4c3437', '#33212a', w=.8, seed=seed)
    ear(d, h, c, -1); ear(d, h, c, 1); skin_shape(d, h, c, expr, shade_top=-.40)
    ink = '#241418'
    Ls = [([(-.20, -.40), (-.12, -.14), (0, -.04), (.12, -.14), (.20, -.40)], .9), ([(-.98, .30), (-.80, .36), (-.66, .46), (-.60, .58)], .95), ([(.98, .30), (.80, .36), (.66, .46), (.60, .58)], .95), ([(-.24, .345), (-.10, .285), (.10, .285), (.24, .345)], .85), ([(0, .70), (0, .84)], 1.0), ([(-.12, .68), (-.08, .80)], .7), ([(.12, .68), (.08, .80)], .7)]
    for L, w in Ls: d.stroke(h(spl(L, n=8), 'd'), w, ink, taper=(.2, .25), tmin=.2)
    for s in (-1, 1):
        if expr in ('sleep', 'snore'): closed_eye(d, h, s * .40, .16, .15, .05, s, False, LINE, .8); closed_eye(d, h, s * .46, .33, .09, .03, s, False, LINE, .6)
        elif expr == 'shock': d.detail(h(ell(s * .40, .13, .14, .16), 'd'), '#fffdf8', sh='#fffdf8'); d.detail(h(ell(s * .40, .13, .035), 'd'), '#c1272d'); d.stroke(h(ell(s * .40, .13, .14, .16), 'd'), .5, taper=(0, 0))
        else:
            lid = h.lid if h.lid is not None else (.25 if expr == 'grump' else .1); open_eye(d, h, s * .40, .12 + .05 * lid, c, s, .15, .15 * (1 - .45 * lid), lid=lid, look=(0, .0), pup=.5); open_eye(d, h, s * .48, .335, c, s, .08, .075 * (1 - .5 * lid), lid=max(.2, lid), pup=.5)
        brow(d, h, s * .40, -.14, s, .34 * (1 - .7 * (h.lid or 0)) if expr not in ('shock', 'sleep') else .0, .20, 1.0, '#8a3f44', .01)
    mouth(d, h, dict(grump='frown', sleep='flat', shock='gape', smile='laugh').get(expr, 'frown'), c, 0, .57, k=1.05)
    if expr == 'grump': d.stroke(h(spl([(.30, -.46), (.38, -.42), (.36, -.34)], n=5), 'd'), .5, '#c1272d'); d.stroke(h(spl([(.50, -.46), (.43, -.40), (.49, -.34)], n=5), 'd'), .5, '#c1272d')
    line = flames((.955, -.06), [((.84, -.20), (.72, -.48), .06, .08), ((.52, -.40), (.36, -.58), .04, .06), ((.0, -.50), (-.36, -.58), 0, .05), ((-.52, -.40), (-.72, -.48), .04, .06), ((-.84, -.20), (-.955, -.06), .06, .08)])
    K = [(-84, 11, .34, 8, -.10, 0, 1, 0), (82, 11, .36, -6, .12, .5, 1, 0), (-64, 12, .56, 10, -.12, 0, 1, 1), (64, 11, .54, -8, .12, .5, 1, 1), (-20, 9, .5, -10, -.06, 0, 1, 0), (28, 9, .5, 8, .08, .4, 1, 0),
         (-44, 13, .72, 12, -.12, 0, 1, 1), (46, 13, .74, -10, .12, .42, 1, 1), (4, 12, .70, -6, -.10, 0, 1, 1), (22, 13, .84, 2, .10, .36, 1, 1), (-13, 14, .88, 4, -.08, .0, 1, 1)]
    for i, (an, hw, L, le, be, sd, ss, ln) in enumerate(K): draw_lock(d, h, lock(an, hw, L, le, be, .08), c, 0., .9, 1.0, shade=sd, sside=ss, lines=ln, seed=seed + 20 + i)
    ln_ = flames((.95, -.14), [((.84, -.22), (.72, -.50), .06, .08), ((.48, -.44), (.36, -.60), .04, .06), ((.0, -.52), (-.36, -.60), 0, .05), ((-.48, -.44), (-.72, -.50), .04, .06), ((-.84, -.22), (-.95, -.14), .06, .08)])
    cp = cap_fill(d, h, c, ln_); d.stroke(h(ln_, 'd.03'), .85, c['hairl'], taper=(.05, .05))
    d.shade(h(np.vstack([flames((-1.0, -.42), [((-.70, -.72), (-.62, -.54), 0), ((-.28, -.80), (-.24, -.60), 0), ((.16, -.80), (.20, -.60), 0), ((.60, -.72), (.64, -.52), 0), ((.96, -.56), (1.0, -.40), 0)], .05), [(1.1, .2), (-1.1, .2)]]), .5), within=cp)
    for a, m, b, w in [((-.60, -.60), (-.70, -.78), (-.86, -.92), .42), ((-.24, -.66), (-.30, -.92), (-.40, -1.12), .46), ((.02, -.64), (.04, -.94), (0, -1.22), .40), ((.36, -.64), (.46, -.86), (.54, -1.06), .42), ((.66, -.56), (.80, -.70), (.94, -.84), .4)]: d.stroke(h(spl([a, m, b], n=8), .15, .6), w, c['hairl'], taper=(.5, .5))


def face_expr(d, h, c, expr, t, brow_col=LINE, bw=.8, lash=False, lid=.0, browang=0., eyes=1.0, mouthk=None):
    mk = dict(sleep='flat', snore='snore', dot='flat', dotblink='flat', shock='gape', smile='smile', laugh='laugh', grump='frown', read='flat'); mk.update(mouthk or {})
    d.stroke(h(spl([(-.02, .315), (.015, .335), (.045, .315)], n=6), 'd'), .36, SKINL)
    if expr in ('sleep', 'snore'):
        for s in (-1, 1): closed_eye(d, h, s * .40, .17 - .02 * h.brow, .155 * eyes ** .5, .062 * (1 - 1.3 * h.brow), s, lash); brow(d, h, s * (.42 - .05 * h.brow), -.07 + .10 * h.brow, s, -.06 + .62 * h.brow, .13 + .04 * h.brow, bw * (.8 + .5 * h.brow), brow_col)
        blush(d, h, c, .62, .41, .42, .14)
    elif expr == 'dotblink':
        for s in (-1, 1): d.stroke(h([(s * .38 - .10, .185), (s * .38, .192), (s * .38 + .10, .185)], 'd'), .8, LINE, taper=(.1, .1)); brow(d, h, s * .40, -.05, s, .0, .12, bw * .7, brow_col, .0)
    elif expr == 'dot':
        for s in (-1, 1): d.part(h(ell(s * .38, .17, .052, .066), 'd'), '#1a1014', '#1a1014', w=.2, tip=False); brow(d, h, s * .40, -.06, s, .0, .12, bw * .7, brow_col, .0)
    elif expr == 'shock':
        G_ = h(np.vstack([spl([(-.9, -.40), (0, -.46), (.9, -.40)], n=6), spl([(.86, -.04), (.3, -.02), (-.3, -.02), (-.86, -.04)], n=6)]), 'd'); d.detail(G_, _mix(c['skin'], '#8fa6cf', .42))
        for q in (-.62, -.46, .50, .66): d.stroke(h([(q, -.30), (q * 1.02, .02)], 'd'), .34, '#5f74a8', taper=(.1, .6))
        for s in (-1, 1): d.part(h(ell(s * .38, .12, .20, .22), 'd'), '#fffdf8', '#e8e2ea', w=.7, tip=False); d.detail(h(ell(s * .37, .13, .035, .04), 'd'), LINE); brow(d, h, s * .40, -.22, s, -.22, .15, bw, brow_col, .05)
    elif expr == 'laugh':
        for s in (-1, 1):
            L = spl([(s * .24, .18), (s * .34, .08), (s * .44, .07), (s * .56, .17)], n=8); d.stroke(h(L, 'd'), .95, taper=(.2, .2), tmin=.35); brow(d, h, s * .42, -.10, s, -.12, .15, bw, brow_col, .05)
        blush(d, h, c, .62, .40, .42, .14)
    else:
        for s in (-1, 1): open_eye(d, h, s * .40, .15, c, s, .155 * eyes ** .5, .21 * eyes, lid=lid); brow(d, h, s * .41, -.14 + .05 * lid, s, browang, .16, bw, brow_col)
        if expr in ('smile',): blush(d, h, c, .62, .41, .30, .13)
    m = h.mouth or mk.get(expr, 'smile')
    mouth(d, h, m, c, .0, .52 if m not in ('laugh', 'gape') else .54)
    if expr == 'sleep' and m in ('open', 'snore'): drool(d, h, .10, .55, .12)


HEADS = dict(gojo=head_gojo, yuji=head_yuji, megumi=head_megumi, nobara=head_nobara, nanami=head_nanami, todo=head_todo, panda=head_panda, sukuna=head_sukuna)


# ================================================================ bodies
LIGHT = np.array([-.55, -.83])


def limb_part(d, n, pts, wd, col, col2, w=1., seed=0, folds=2, cap=.05, root='open', shade=.46, ink=LINE, rnd=False, hi=None, outer=None):
    """sleeve / trouser leg, shaded on the side away from the light, a few drawn folds.
    root: 'open' straight and unlined (hidden under the torso) | 'ball' rounded shoulder, lined on its outer half | 'round' lined all round"""
    L, R, C, T = limb(pts, wd); Pl, Pr = n(L), n(R)
    nl = Pl[len(Pl) // 2] - Pr[len(Pr) // 2]; dark_left = np.dot(nl, LIGHT) < 0
    if rnd: e = spl([L[-1], C[-1] + T[-1] * wd[-1] * .62 + (L[-1] - C[-1]) * .74, C[-1] + T[-1] * wd[-1] * .92, C[-1] + T[-1] * wd[-1] * .62 + (R[-1] - C[-1]) * .74, R[-1]], n=7)[1:-1]
    else: e = spl([L[-1], C[-1] + T[-1] * cap, R[-1]], n=6)[1:-1]
    body = np.vstack([L, e, R[::-1]]); gaps = ()
    if root == 'open':
        P = n(body); per = np.hypot(*np.diff(np.vstack([P, P[:1]]), axis=0).T).sum(); gaps = [(1 - np.hypot(*(P[0] - P[-1])) / per - .03, .03)]
    else:
        w0 = wd[0]; arc = spl([R[0], C[0] - T[0] * w0 * .62 + (R[0] - C[0]) * .74, C[0] - T[0] * w0 * .86, C[0] - T[0] * w0 * .62 + (L[0] - C[0]) * .74, L[0]], n=8)[1:-1]
        P = n(np.vstack([body, arc])); seg = np.hypot(*np.diff(np.vstack([P, P[:1]]), axis=0).T); cum = np.cumsum(seg) / seg.sum(); i0 = len(body) - 1
        if root == 'ball':
            k = len(arc); outer_is_L = abs(L[0][0]) > abs(R[0][0]) if outer is None else outer == 'L'
            gaps = [(cum[i0] - .10, cum[i0 + int(k * .56)])] if outer_is_L else [(cum[i0 + int(k * .44)], .10)]
    tn = np.abs(np.angle(np.exp(1j * np.diff(np.unwrap(np.arctan2(T[:, 1], T[:, 0])))))).sum()
    if tn > 1.2: shade = min(shade, .24)
    D_, O_ = (L, R) if dark_left else (R, L); s = np.linspace(0, 1, len(C))[:, None]
    inn = D_ + (O_ - D_) * (shade * (.72 + .28 * np.sin(s * 7 + seed)) * np.clip(s * 4 + (.1 if root != 'open' else 1), 0, 1))
    hh = [] if hi is None else [(n(np.vstack([(O_ + (D_ - O_) * .07)[len(C) // 6:len(C) * 4 // 5], (O_ + (D_ - O_) * (.07 + .25 * np.sin(s * pi * 1.1) ** 2))[len(C) // 6:len(C) * 4 // 5][::-1]])), hi)]
    d.part(P, col, col2, shade=[n(np.vstack([D_, inn[::-1]]))], hi=hh, w=w, ink=ink, seed=seed, gaps=gaps, tip=False)
    r = np.random.default_rng(seed + 77); m = len(C)
    for k in range(folds):
        i = int(m * (.34 + .34 * k / max(1, folds - 1)) + r.integers(-2, 3)); i = max(2, min(m - 8, i)); side = D_ if (k + seed) % 2 == 0 else O_
        mid = (D_[i] + O_[i]) / 2; a = side[i]; b = mid + (side[i] - mid) * .10 + T[i] * (.08 if k % 2 else -.05); c_ = (a + b) / 2 + T[i] * .04
        d.stroke(n(spl([a, c_, b], n=6)), .40, ink, taper=(.1, .6), seed=seed + k)
    return L, R, C, T


def hand(d, n, c, s=1.0, kind='open', seed=0, rot=0., flipy=1):
    """local +x away from the wrist; fingers are three soft lobes and a thumb, never a ball"""
    shp = dict(open=[(-.08, -.11), (.03, -.145), (.075, -.215), (.135, -.225), (.165, -.165), (.225, -.165), (.285, -.135), (.295, -.08), (.325, -.055), (.335, .0), (.315, .035), (.315, .085), (.265, .13), (.17, .155), (.06, .145), (-.08, .11)],
               fist=[(-.08, -.11), (.05, -.16), (.15, -.175), (.25, -.12), (.275, -.02), (.26, .09), (.18, .16), (.06, .15), (-.08, .11)],
               flat=[(-.08, -.10), (.08, -.135), (.22, -.11), (.31, -.03), (.30, .05), (.20, .125), (.06, .13), (-.08, .10)])[kind]
    P = np.array(shp) * [1, flipy] * s; m = n.at(0, 0, rot); Q = m(spl(P, True, n=6))
    d.part(Q, c['skin'], c['skin2'], shade=[m(spl(np.array([(-.1, .02), (.1, .06), (.3, .04), (.4, .2), (-.1, .25)]) * [1, flipy] * s, True, n=6))], w=.82, ink=SKINO, seed=seed, tip=False)
    if kind == 'fist':
        for y in (-.07, .0, .07): d.stroke(m(np.array([(.17, y * flipy), (.25, y * 1.0 * flipy)]) * s), .34, SKINL)
    if kind == 'open':
        for x, y in ((.295, -.072), (.315, .03)): d.stroke(m(np.array([(x - .10, y * .8 * flipy), (x, y * flipy)]) * s), .34, SKINL, taper=(.5, .05))
        d.stroke(m(np.array([(.16, -.155 * flipy), (.12, -.10 * flipy)]) * s), .34, SKINL, taper=(.05, .5))
    if kind == 'flat':
        for y in (-.04, .03): d.stroke(m(np.array([(.18, y * flipy), (.29, y * flipy)]) * s), .34, SKINL)
    return m


def shoe(d, n, c, s=1.0, seed=0, kind='shoe', top=False):
    """local +x = toe direction, origin at the ankle.  top: toe points at us (sole shows)"""
    col, col2 = c['shoe'], c['shoe2']
    if kind == 'sole':
        P = n(spl(np.array([(-.20, -.02), (-.16, -.24), (0, -.34), (.16, -.24), (.20, -.02), (.18, .18), (0, .27), (-.18, .18)]) * s, True, n=7)); d.part(P, col, col2, shade=[n(spl(np.array([(.02, -.4), (.3, -.3), (.3, .3), (-.1, .4), (.1, .0)]) * s, True, n=6))], w=.95, seed=seed, tip=False)
        Q = n(spl(np.array([(-.13, -.02), (-.10, -.17), (0, -.25), (.10, -.17), (.13, -.02), (.11, .12), (0, .19), (-.11, .12)]) * s, True, n=7)); d.part(Q, _mix(col, '#d9cdbf', .34), _mix(col2, '#b5a8a4', .22), w=.5, seed=seed + 1, tip=False)
        for y in (-.10, -.02, .06): d.stroke(n(np.array([(-.08, y), (0, y - .012), (.08, y)]) * s), .34)
        return
    P = n(spl(np.array([(-.06, -.17), (.10, -.19), (.28, -.17), (.40, -.09), (.43, .02), (.38, .12), (.26, .18), (.08, .19), (-.06, .17)]) * s, True, n=7))
    d.part(P, col, col2, shade=[n(spl(np.array([(-.1, .03), (.2, .05), (.5, -.02), (.5, .3), (-.1, .3)]) * s, True, n=6))], w=.95, seed=seed, tip=False)
    d.stroke(n(spl(np.array([(.31, -.155), (.335, .0), (.29, .165)]) * s, n=6)), .5)                                    # sole edge
    d.detail(n(spl(np.array([(.335, -.13), (.41, -.06), (.42, .03), (.37, .11), (.315, .15), (.345, .0)]) * s, True, n=6)), _mix(col, '#e9e2d8', .30 if col != '#c2352f' else .75), within=P)
    d.stroke(n(spl(np.array([(.10, -.11), (.17, -.0), (.10, .11)]) * s, n=6)), .36)


class Fr:
    """torso frame with a bend: (u, v) -> local"""
    def __init__(self, n, lean=0., bend=0., sx=1., L=1., hem=(0., 0.), t=0.): self.n, self.lean, self.bend, self.sx, self.L, self.hem, self.t = n, lean, bend, sx, L, hem, t
    def __call__(self, P):
        P = np.atleast_2d(np.asarray(P, float)); v = P[:, 1]; x = P[:, 0] * self.sx + self.lean * v + self.bend * v * v; y = v * self.L
        if self.hem[0] or self.hem[1]:
            w = np.clip((v - .58) / .5, 0, 1.3) ** 1.5; wob = 1 + .7 * np.sin(self.t * .62 + P[:, 0] * 3.6); x = x + self.hem[0] * w * wob; y = y + self.hem[1] * w * wob * .6
        return self.n(np.c_[x, y])


TORSO = [(-.22, -.04), (-.42, .03), (-.56, .18), (-.555, .46), (-.57, .78), (-.63, 1.06), (-.32, 1.13), (0, 1.15), (.32, 1.13), (.63, 1.06), (.57, .78), (.555, .46), (.56, .18), (.42, .03), (.22, -.04)]


def torso_base(d, F, pts, c, col, col2, seed, w=1., gaps=(), hi=None):
    P = F(spl(pts, True, n=7, corners=(0, 5, 9, 14) if len(pts) == 15 else ()))
    sh = [F(spl([(.10, -.1), (.26, .30), (.22, .60), (.34, .90), (.28, 1.3), (.9, 1.3), (.9, -.1)], True, n=7, corners=(0, 4, 5, 6))), F(spl([(-.62, -.1), (-.40, .20), (0, .285), (.40, .20), (.62, -.1)], True, n=7, corners=(0, 4)))]
    if hi: hi = [(F(spl(q, True, n=6)), hc) for q, hc in hi]
    d.part(P, col, col2, shade=sh, hi=hi or (), w=w, seed=seed, gaps=gaps, tip=False)
    return P


def collar(d, F, c, hgt=.20, col=None, col2=None, seed=0, open_=0.):
    """stand collar, drawn over the chin"""
    P = F(spl([(-.43, .12), (-.44, -hgt * .30), (-.37, -hgt * .78), (-.18, -hgt * .42), (0, -hgt * .28), (.18, -hgt * .42), (.37, -hgt * .78), (.44, -hgt * .30), (.43, .12), (.20, .19), (0, .205), (-.20, .19)], True, n=7))
    d.part(P, col or c['cloth'], col2 or c['cloth2'], shade=[F([(.06, -.4), (.5, -.4), (.5, .3), (.12, .3)])], hi=[(F(spl([(-.33, -.02), (-.30, -.14), (-.20, -.10), (-.24, .06)], True, n=5)), _mix(col or c['cloth'], '#8d84a8', .4))], w=.95, seed=seed)
    d.stroke(F(spl([(.0, -hgt * .28), (.015, .06), (.0, .205)], n=6)), .5)


def body_gojo(d, F, c, back, seed, br):
    P = torso_base(d, F, TORSO, c, c['cloth'], c['cloth2'], seed, hi=[([(-.50, .16), (-.34, .20), (-.16, .34), (-.10, .50), (-.24, .44), (-.42, .36)], '#6d6488'), ([(-.54, .70), (-.34, .66), (-.10, .78), (-.14, .86), (-.38, .84)], '#645b7c'), ([(-.54, .48), (-.42, .50), (-.30, .60), (-.40, .63), (-.52, .60)], '#5c5472'), ([(.10, .48), (.24, .40), (.30, .46), (.18, .58)], '#5a526f')])
    if back: d.stroke(F(spl([(0, .1), (.01, .5), (0, 1.1)], n=6)), .4); return
    d.stroke(F(spl([(-.04, .16), (-.02, .55), (-.05, 1.14)], n=8)), .55); d.stroke(F(spl([(.055, .18), (.07, .55), (.05, 1.14)], n=8)), .36)
    for L in ([(.54, .40), (.36, .46), (.22, .56)], [(-.54, .52), (-.40, .56), (-.28, .64)], [(.56, .74), (.36, .74), (.16, .84)], [(-.58, .92), (-.38, .90), (-.18, .98)], [(.60, 1.0), (.44, .96), (.30, 1.0)]): d.stroke(F(spl(L, n=7)), .42, taper=(.08, .6))
    d.stroke(F(spl([(-.50, .86), (-.36, .84), (-.22, .87)], n=5)), .4)


def body_uniform(d, F, c, back, seed, br, btn='#d8b04a', skirt=False):
    top = TORSO if not skirt else [(-.22, -.04), (-.40, .03), (-.53, .18), (-.50, .46), (-.47, .70), (-.49, .84), (-.26, .89), (0, .905), (.26, .89), (.49, .84), (.47, .70), (.50, .46), (.53, .18), (.40, .03), (.22, -.04)]
    if skirt:
        K = [(-.46, .72), (-.62, .98), (-.74, 1.24), (-.44, 1.30), (-.2, 1.26), (.06, 1.31), (.30, 1.27), (.50, 1.30), (.74, 1.22), (.62, .98), (.46, .72)]
        Q = F(spl(K, True, n=6, corners=(0, 2, 4, 6, 8, 10))); d.part(Q, c['cloth'], c['cloth2'], shade=[F([(.0, .7), (.06, 1.4), (.30, 1.4), (.20, .7)]), F([(.36, .7), (.5, 1.4), (.9, 1.4), (.6, .7)]), F([(-.5, .7), (-.46, 1.35), (-.28, 1.35), (-.34, .7)])], w=1., seed=seed + 4, tip=False)
        for x0, x1 in ((-.30, -.44), (-.10, -.2), (.10, .06), (.24, .30), (.38, .50)): d.stroke(F(spl([(x0, .84), ((x0 + x1) / 2, 1.05), (x1, 1.27)], n=6)), .45, taper=(.2, .2))
    torso_base(d, F, top, c, c['cloth'], c['cloth2'], seed, hi=[([(-.48, .16), (-.34, .20), (-.18, .34), (-.14, .46), (-.28, .42), (-.42, .34)], '#626a9c'), ([(-.48, .66), (-.30, .62), (-.12, .70), (-.18, .77), (-.36, .77)], '#575d8c'), ([(.12, .46), (.26, .38), (.30, .44), (.18, .54)], '#51567f')])
    if back: d.stroke(F(spl([(0, .1), (.01, .5), (0, .9)], n=6)), .4); return
    d.stroke(F(spl([(.02, .16), (.05, .5), (.03, .9 if skirt else 1.14)], n=8)), .5)
    for L in ([(.52, .38), (.36, .44), (.22, .54)], [(-.52, .50), (-.38, .54), (-.26, .62)], [(.52, .70), (.34, .70), (.18, .80)]): d.stroke(F(spl(L, n=7)), .40, taper=(.08, .6))
    for y in ((.30,) if not skirt else (.28,)):
        Q = F(ell(.14, y, .062, .062)); d.part(Q, btn, '#93702c', w=.5, tip=False); d.stroke(F(spl([(.13, y + .02), (.12, y - .02), (.15, y - .025), (.165, y + .01)], n=5)), .26)


def body_nanami(d, F, c, back, seed, br):
    torso_base(d, F, TORSO, c, c['cloth'], c['cloth2'], seed)
    if back: d.stroke(F(spl([(0, .1), (.01, .6), (0, 1.14)], n=6)), .4); return
    S = F(spl([(-.22, -.06), (0, -.02), (.22, -.06), (.17, .30), (.05, .66), (-.05, .66), (-.16, .30)], True, n=6, corners=(0, 2, 4, 5))); d.part(S, '#5f83b8', '#42608f', shade=[F(spl([(-.5, -.1), (-.3, .18), (0, .24), (.3, .18), (.5, -.1)], True, n=6))], w=.5, seed=seed + 2, tip=False)
    T = F(spl([(-.055, .06), (.055, .06), (.045, .14), (.085, .46), (.0, .62), (-.085, .46), (-.045, .14)], True, n=6, corners=(0, 1, 4))); d.part(T, '#e3c455', '#a98634', w=.55, seed=seed + 3)
    for x, y in ((-.02, .22), (.035, .30), (-.04, .36), (.02, .43), (-.02, .50)): d.detail(F(ell(x, y, .02, .016)), '#4a3726', within=T)
    for s in (-1, 1):
        Lp = F(spl([(s * .20, -.06), (s * .36, .03), (s * .40, .22), (s * .30, .30), (s * .34, .36), (s * .07, .70), (s * .10, .40), (s * .17, .15)], True, n=6, corners=(0, 3, 4, 5))); d.part(Lp, _mix(c['cloth'], '#ffffff', .25), c['cloth2'], w=.6, seed=seed + 5, tip=False)
    d.stroke(F(spl([(.07, .70), (.10, .86), (.22, 1.14)], n=6)), .5); d.stroke(F(spl([(-.06, .70), (-.04, .9), (-.10, 1.13)], n=6)), .4)
    d.part(F(ell(.02, .74, .03)), '#8a6b4c', '#5d4638', w=.4, tip=False); d.stroke(F(spl([(-.50, .84), (-.36, .82), (-.22, .85)], n=5)), .45); d.stroke(F(spl([(.30, .85), (.42, .83), (.54, .86)], n=5)), .45)
    for L in ([(.54, .44), (.40, .48), (.32, .56)], [(-.54, .50), (-.42, .54), (-.32, .62)]): d.stroke(F(spl(L, n=7)), .38, taper=(.08, .6))


def body_todo(d, F, c, back, seed, br):
    B = [(-.22, -.06), (-.50, .0), (-.74, .16), (-.72, .42), (-.60, .72), (-.56, .98), (-.3, 1.04), (0, 1.06), (.3, 1.04), (.56, .98), (.60, .72), (.72, .42), (.74, .16), (.50, .0), (.22, -.06)]
    torso_base(d, F, B, c, c['skin'], c['skin2'], seed)
    if back: d.stroke(F(spl([(0, .1), (.02, .5), (0, 1.0)], n=6)), .5, SKINL)
    else:
        for s in (-1, 1):
            d.stroke(F(spl([(s * .06, .40), (s * .30, .50), (s * .58, .42), (s * .66, .28)], n=7)), .55, SKINL, taper=(.2, .4)); d.detail(F(ell(s * .40, .34, .035, .028)), '#b9705a')
            d.stroke(F(spl([(s * .24, .60), (s * .22, .80), (s * .16, .96)], n=5)), .36, SKINL)
        d.stroke(F(spl([(0, .28), (.01, .6), (0, .96)], n=6)), .45, SKINL, taper=(.3, .3)); d.stroke(F([(-.17, .70), (0, .72), (.17, .70)]), .34, SKINL); d.stroke(F([(-.15, .84), (0, .86), (.15, .84)]), .34, SKINL)
    S = F(spl([(-.60, .90), (-.2, .97), (.2, .97), (.60, .90), (.63, 1.10), (.2, 1.18), (-.2, 1.18), (-.63, 1.10)], True, n=6, corners=(0, 3, 4, 7))); d.part(S, '#f2ede4', '#b8b0ba', shade=[F([(.1, .8), (.9, .8), (.9, 1.3), (.2, 1.3)])], w=.8, seed=seed + 2, tip=False)
    d.stroke(F(spl([(-.56, 1.02), (0, 1.06), (.56, 1.0)], n=6)), .36); 
    if not back: d.part(F(spl([(-.10, .98), (.12, .97), (.15, 1.14), (.10, 1.36), (-.02, 1.34), (-.12, 1.14)], True, n=6)), '#f2ede4', '#b8b0ba', w=.7, seed=seed + 3, tip=False)


def body_panda(d, F, c, back, seed, br):
    B = [(-.22, -.06), (-.62, .04), (-1.00, .32), (-1.22, .78), (-1.16, 1.20), (-.80, 1.52), (-.36, 1.64), (0, 1.66), (.36, 1.64), (.80, 1.52), (1.16, 1.20), (1.22, .78), (1.00, .32), (.62, .04), (.22, -.06)]
    P = torso_base(d, F, B, c, c['skin'], c['skin2'], seed, hi=[([(-1.6, -.4), (-1.3, .34), (-.94, .50), (-.5, .56), (0, .46), (.5, .56), (.94, .50), (1.3, .34), (1.6, -.4)], c['hair'])])
    d.shade(F(spl([(.5, .5), (.86, .9), (.72, 1.3), (.2, 1.7), (1.5, 1.8), (1.5, .5)], True, n=6)), within=P)
    d.stroke(F(spl([(-.98, .47), (-.5, .56), (0, .46), (.5, .56), (.98, .47)], n=8)), .4, taper=(.1, .1))
    if not back: d.stroke(F(spl([(-.05, 1.16), (0, 1.20), (.05, 1.16)], n=5)), .55)
    for L in ([(-.06, .78), (-.02, .86), (-.05, .92)], [(.62, .74), (.66, .82), (.62, .88)], [(-.62, 1.22), (-.58, 1.30), (-.62, 1.36)]): d.stroke(F(spl(L, n=5)), .3, '#8d8798', taper=(.4, .4))


def body_sukuna(d, F, c, back, seed, br):
    B = [(-.22, -.04), (-.46, .03), (-.60, .20), (-.60, .50), (-.66, .84), (-.76, 1.24), (-.4, 1.31), (0, 1.33), (.4, 1.31), (.76, 1.24), (.66, .84), (.60, .50), (.60, .20), (.46, .03), (.22, -.04)]
    torso_base(d, F, B, c, c['cloth'], c['cloth2'], seed)
    if not back:
        d.part(F(spl([(-.20, -.06), (.20, -.06), (.0, .44)], True, n=5, corners=(0, 1, 2))), c['skin'], c['skin2'], shade=[F(spl([(-.3, -.1), (0, .16), (.3, -.1)], True, n=5))], w=.5, ink=SKINO, tip=False)
        for s in (-1, 1): d.part(F(spl([(s * .20, -.08), (s * .32, -.02), (s * .02, .62), (-s * .08, .60)], True, n=5, corners=(0, 1, 2, 3))), '#33364f', '#20223a', w=.6, tip=False)
    S = F(spl([(-.62, .58), (0, .66), (.62, .58), (.65, .78), (0, .86), (-.65, .78)], True, n=6, corners=(0, 2, 3, 5))); d.part(S, '#2a2c42', '#191a2c', w=.7, tip=False)
    for L in ([(-.10, .88), (-.16, 1.1), (-.12, 1.30)], [(.12, .88), (.22, 1.1), (.20, 1.30)], [(-.44, .84), (-.5, 1.04), (-.52, 1.28)], [(.46, .84), (.52, 1.04), (.54, 1.28)]): d.stroke(F(spl(L, n=6)), .42, taper=(.2, .2))


def scarf(d, F, seed=0):
    P = F(spl([(-.50, .05), (-.30, -.10), (0, -.03), (.30, -.10), (.50, .05), (.40, .22), (0, .28), (-.40, .22)], True, n=7)); d.part(P, '#2f3350', '#1d2036', shade=[F([(.05, -.2), (.6, -.2), (.6, .4), (.1, .4)])], w=.9, seed=seed, tip=False)
    for L in ([(-.40, .10), (-.1, .16), (.2, .10)], [(.36, .0), (.14, .08), (-.06, .06)]): d.stroke(F(spl(L, n=6)), .4)


def hood_back(d, F, c, seed=0):
    P = F(spl([(-.52, .18), (-.74, -.02), (-.66, -.30), (-.2, -.2), (.2, -.2), (.66, -.30), (.74, -.02), (.52, .18)], True, n=7)); d.part(P, '#d63b34', '#932430', shade=[F([(.1, -.5), (1, -.5), (1, .4), (.1, .4)])], w=1., seed=seed, tip=False)


def hood_front(d, F, c, seed=0):
    P = F(spl([(-.58, .08), (-.46, -.12), (-.22, -.02), (0, .02), (.22, -.02), (.46, -.12), (.58, .08), (.44, .26), (.16, .24), (0, .36), (-.16, .24), (-.44, .26)], True, n=7)); d.part(P, '#d63b34', '#932430', shade=[F(spl([(-.5, -.2), (-.25, .06), (0, .10), (.25, .06), (.5, -.2)], True, n=5)), F([(.16, -.2), (.8, -.2), (.8, .5), (.24, .5)])], w=.95, seed=seed, tip=False)
    for L in ([(-.50, .12), (-.3, .17), (-.1, .16)], [(.46, .08), (.28, .15), (.06, .16)], [(0, .06), (.01, .2), (0, .34)]): d.stroke(F(spl(L, n=6)), .42, '#5c1620')


BODY = dict(gojo=body_gojo, yuji=lambda *a: body_uniform(*a), megumi=lambda *a: body_uniform(*a), nobara=lambda *a: body_uniform(*a, skirt=True), nanami=body_nanami, todo=body_todo, panda=body_panda, sukuna=body_sukuna)
LEGW = dict(gojo=(.27, .25, .28), yuji=(.27, .25, .30), megumi=(.26, .24, .27), nobara=(.19, .165, .14), nanami=(.27, .25, .27), todo=(.36, .42, .26), panda=(.46, .44, .38), sukuna=(.30, .30, .30))
ARMW = dict(gojo=(.20, .19, .235), yuji=(.20, .19, .23), megumi=(.20, .19, .22), nobara=(.18, .165, .185), nanami=(.20, .19, .215), todo=(.27, .25, .18), panda=(.36, .33, .29), sukuna=(.22, .28, .36))


def arm(d, n, who, c, pts, seed=0, hk='open', hrot=0., hs=1., flipy=1, root='open'):
    wd = ARMW[who]; a, b = np.asarray(pts[-2], float), np.asarray(pts[-1], float); v = b - a; v /= np.hypot(*v) + 1e-9; an = math.atan2(v[1], v[0])
    if who == 'panda':
        limb_part(d, n, pts, wd, c['hair'], c['hair2'], seed=seed, folds=0, rnd=True, root='round', hi='#47424f'); return
    if who == 'todo':
        hand(d, n.at(*(b + v * .02), an + hrot), c, 1.12 * hs, hk, seed, flipy=flipy); limb_part(d, n, pts, wd, c['skin'], c['skin2'], seed=seed, folds=1, ink=SKINO, root=root, cap=.02); return
    hand(d, n.at(*(b - v * .03), an + hrot), c, hs, hk, seed, flipy=flipy)
    limb_part(d, n, pts, wd, c['cloth'], c['cloth2'], seed=seed, folds=2, root=root, hi=_mix(c['cloth'], '#8c86b4' if who not in ('nanami', 'sukuna') else '#ffffff', .30))
    if who in ('yuji', 'megumi', 'nobara', 'gojo'): pass
    if who == 'nanami':                                          # shirt cuff
        pr = np.array([-v[1], v[0]]); Q = n(spl([b + pr * .17, b + v * .06 + pr * .14, b + v * .06 - pr * .14, b - pr * .17], True, n=4)); d.part(Q, '#6f93c6', '#47679a', w=.4, tip=False)


def leg(d, n, who, c, pts, seed=0, toe=0., sole=False, ss=1., shoe_on=True):
    wd = LEGW[who]; a, b = np.asarray(pts[-2], float), np.asarray(pts[-1], float); v = b - a; v /= np.hypot(*v) + 1e-9; an = math.atan2(v[1], v[0])
    if who == 'panda':
        limb_part(d, n, pts, wd, c['hair'], c['hair2'], seed=seed, folds=0, rnd=True, hi='#47424f')
        if sole:
            e = b + v * .06; d.detail(n(ell(*e, .15, .12)), '#6a5a62')
            for k in (-1, 0, 1): d.detail(n(ell(e[0] + k * .14, e[1] - .19 + .03 * abs(k), .055, .06)), '#6a5a62')
        return
    if shoe_on and not sole: shoe(d, n.at(*(b - v * .04), an + toe), c, ss, seed)
    if who == 'nobara':
        limb_part(d, n, pts, wd, c['pant'], c['pant2'], seed=seed, folds=0)
    elif who == 'todo':
        limb_part(d, n, pts, wd, c['pant'], c['pant2'], seed=seed, folds=3)
    else:
        limb_part(d, n, pts, wd, c['pant'], c['pant2'], seed=seed, folds=2)
    if shoe_on and sole: shoe(d, n.at(*(b + v * .04 + np.array([0, -.05])), toe), c, 1.02 * ss, seed, 'sole')


def newspaper(d, n, w=.86, hgt=.60, seed=0, t=0., fold=.10, turn=None, flut=0.):
    """an open paper seen from the reader's far side: two pages, drawn columns, a little print picture, a lifted corner"""
    fl = .02 * math.sin(t * .35) + flut * math.sin(t * .95)
    L = spl([(-w, -hgt + .05 + fl), (-w * .5, -hgt - .02), (0, -hgt + .10), (0, hgt + .06), (-w * .5, hgt - .04), (-w - .03, hgt + .02)], True, n=6, corners=(0, 2, 3, 5))
    R = spl([(0, -hgt + .10), (w * .5, -hgt - .02), (w, -hgt + .04 - fl), (w + .03, hgt + .0), (w * .5, hgt - .04), (0, hgt + .06)], True, n=6, corners=(0, 2, 3, 5))
    d.part(n(L), '#f4efe2', '#c9c0b6', w=.95, seed=seed, tip=False); d.part(n(R), '#f4efe2', '#c9c0b6', shade=[n(spl([(0, -2), (.22, -2), (.16, 0), (.24, 2), (0, 2)], True, n=5))], w=.95, seed=seed + 1, tip=False)
    r = np.random.default_rng(seed + 3); g = '#8b847f'
    for x0, x1, y0, hd in ((-w + .10, -.52, -hgt + .20, 1), (-.44, -.08, -hgt + .24, 0), (.10, .46, -hgt + .24, 1), (.54, w - .10, -hgt + .18, 0)):
        y = y0
        if hd: d.stroke(n([(x0, y), (x1, y + .01)]), 1.2, '#4d474a', taper=(.02, .05)); y += .12
        while y < hgt - .16:
            d.stroke(n([(x0, y), (x1 - r.uniform(0, .10), y + .006)]), .30, g, taper=(.05, .1), seed=int(y * 99)); y += .085
    for q in ([(-w + .12, .06), (-.54, .04), (-.54, .34), (-w + .12, .36)],): Q = n(q); d.fl.fill(Q, '#b9b2a8'); d.sc.fill(Q, '#948c8c'); d.stroke(n(q + q[:1]), .3, '#6b6566', taper=(0, 0))
    d.detail(n(spl([(-w + .16, .32), (-.70, .16), (-.62, .24), (-.56, .18), (-.55, .33)], True, n=4)), '#8d868a')
    d.stroke(n([(0, -hgt + .10), (0, hgt + .06)]), .5)
    if turn is not None and 0 < turn < 1:                            # one page comes over: edge swings round the spine, belly of air
        a = pi * turn; ex = w * math.cos(a); lf = .34 * math.sin(a); bl = .22 * math.sin(a)
        P = spl([(0, -hgt + .10), (ex * .5 + bl * 1.0, -hgt - .06 - lf * .8), (ex, -hgt + .02 - lf), (ex + bl * .28, 0 - lf * .7), (ex * 1.02, hgt - .02 - lf * .45), (ex * .5 + bl * 1.0, hgt - .02 - lf * .2), (0, hgt + .06)], True, n=7, corners=(0, 2, 4, 6))
        d.part(n(P), '#fbf7ec' if ex < 0 else '#ebe4d6', '#c3b9b0', shade=[n(spl([(0, -2), (ex * .34, -2), (ex * .26, 0), (ex * .36, 2), (0, 2)], True, n=5))], w=.95, seed=seed + 9, tip=False)
        for k in range(7): y = -hgt + .24 + k * .13; d.stroke(n(spl([(ex * .16, y - lf * .15), (ex * .5 + bl * .2, y - .03 - lf * .4), (ex * .86, y - lf * .7)], n=6)), .30, '#8b847f', taper=(.05, .1), seed=k)


# poses in body units.  arm/leg = control points; h = head (x, y, roll), over = arms drawn over the torso
POSES = dict(
    starfish=dict(aL=[(-.44, .18), (-.94, .28), (-1.38, .04)], aR=[(.44, .20), (.92, .34), (1.34, .52)], lL=[(-.32, .95), (-.74, 1.26), (-1.02, 1.58)], lR=[(.32, .95), (.64, 1.34), (.84, 1.72)], h=(.03, -.78, .12), yaw=.14, expr='sleep', toe=(-.75, .60), sole=True, hk=('open', 'open'), pitch=-.16),
    scratch=dict(aL=[(-.44, .18), (-.94, .28), (-1.38, .04)], aR=[(.46, .20), (.92, .52), (.40, .80)], lL=[(-.32, .95), (-.74, 1.26), (-1.02, 1.58)], lR=[(.32, .95), (.64, 1.34), (.84, 1.72)], h=(.03, -.78, .12), yaw=.14, expr='sleep', toe=(-.75, .60), sole=True, over='R', hk=('open', 'flat'), pitch=-.16),
    stare=dict(aL=[(-.46, .20), (-.64, .50), (-.64, .84)], aR=[(.46, .20), (.64, .50), (.64, .84)], lL=[(-.28, .95), (-.29, 1.30), (-.29, 1.60)], lR=[(.28, .95), (.29, 1.30), (.29, 1.60)], h=(0, -.80, 0), yaw=0, expr='dot', toe=(.25, -.25), over='LR', hk=('fist', 'fist')),
    wave=dict(aL=[(-.46, .20), (-.66, .50), (-.62, .84)], aR=[(.44, .22), (.92, .26), (1.22, -.14)], lL=[(-.28, .95), (-.30, 1.30), (-.32, 1.60)], lR=[(.28, .95), (.34, 1.28), (.36, 1.58)], h=(-.02, -.80, -.08), yaw=-.08, expr='laugh', toe=(.25, -.25), over='L', hk=('fist', 'open')),
    shock=dict(aL=[(-.44, .20), (-.92, .22), (-1.16, -.12)], aR=[(.44, .20), (.92, .22), (1.16, -.12)], lL=[(-.28, .95), (-.40, 1.28), (-.46, 1.56)], lR=[(.28, .95), (.40, 1.28), (.46, 1.56)], h=(0, -.84, 0), yaw=0, expr='shock', toe=(.35, -.35), hk=('open', 'open')),
    pillow=dict(aL=[(-.44, .22), (-.84, .52), (-.30, .70)], aR=[(.44, .20), (.90, .24), (1.20, .52)], lL=[(-.30, .95), (-.66, 1.24), (-.34, 1.54)], lR=[(.30, .95), (.42, 1.32), (.50, 1.64)], h=(.06, -.78, .34), yaw=.24, expr='sleep', toe=(-.5, .55), sole=True, over='L', hk=('flat', 'open'), pitch=-.10),
    sit=dict(aL=[(-.46, .22), (-.78, .56), (-.62, .84)], aR=[(.46, .22), (.78, .56), (.62, .84)], lL=[(-.30, .86), (-.48, 1.08), (-.56, 1.22)], lR=[(.30, .86), (.48, 1.08), (.56, 1.22)], h=(0, -.80, 0), yaw=0, expr='read', sole=True, toe=(-.25, .25), over='LR', hk=('fist', 'fist')),
    crossed=dict(aL=[(-.46, .22), (-.72, .54), (.16, .56)], aR=[(.46, .22), (.74, .56), (-.14, .64)], lL=[(-.30, .86), (-.48, 1.08), (-.56, 1.22)], lR=[(.30, .86), (.48, 1.08), (.56, 1.22)], h=(0, -.80, 0), yaw=0, expr='grump', sole=True, toe=(-.25, .25), over='LR', hk=('flat', 'flat')),
)


def put(d, who, pose, x, y, S, rot=0., flip=False, t=0., expr=None, yaw=None, pitch=None, hroll=0., wind=(1., 0.), gust=.06, wamp=.05, breath=None, seed=0, hx=0., hy=0., **kw):
    """draw one chibi.  (x, y) = neck px, S = px per unit.  t = frame (breath, hair, scratch).  wind in screen direction."""
    if pose == 'curl': return put_curl(d, who, x, y, S, rot, flip, t, expr, yaw, pitch, hroll, wind, gust, wamp, breath, seed, **kw)
    c = CAST[who]; q = POSES[pose]; k = c['tall']; lw0 = d.lw; d.lw = S * kw.get('lw', .044); d.lw_ref = d.lw
    if flip:
        n0 = N().at(x, y, rot, S * -1, S)
    else: n0 = N().at(x, y, rot, S)
    b = math.sin(2 * pi * t / 36. - .6) if breath is None else breath
    lie = pose in ('starfish', 'scratch', 'pillow')
    deep = kw.get('deep', 1.); hv = np.linalg.inv(n0.M[:2, :2] / n0.s) @ np.array(wind, float) * kw.get('hem', 0.)
    F = Fr(n0, kw.get('lean', 0.), kw.get('bend', 0.), 1 + (.030 * b if lie else .012 * b) * deep, 1.0 + (.012 * b * (deep - 1) if lie else 0.), hv, t)
    P = {n: [np.array(v, float) for v in q[n]] for n in ('aL', 'aR', 'lL', 'lR')}
    for n in ('aL', 'aR', 'lL', 'lR'):
        if n in kw: P[n] = [np.array(v, float) for v in kw[n]]
    if who == 'panda':
        for n, off in (('aL', (-.42, .14)), ('aR', (.42, .14)), ('lL', (-.34, .46)), ('lR', (.34, .46))): P[n] = [v + np.array(off) * (1 if i == 0 else 1.25) for i, v in enumerate(P[n])]
    if who == 'todo':
        for n, off in (('aL', (-.16, .02)), ('aR', (.16, .02)), ('lL', (-.05, .06)), ('lR', (.05, .06))): P[n] = [v + np.array(off) * (1 if i == 0 else 1.3) for i, v in enumerate(P[n])]
    if who == 'gojo' or who == 'nanami':
        for n in ('lL', 'lR'): P[n] = [P[n][0], P[n][1] + [0, .05], P[n][2] + [0, .11]]
    if lie: P['aL'][-1] = P['aL'][-1] + [0, .012 * b * deep]; P['aR'][-1] = P['aR'][-1] + [0, -.012 * b * deep]
    if pose == 'scratch':
        sc = kw.get('scratch', math.sin(t * 1.4)); P['aR'][-1] = P['aR'][-1] + [.06 * sc, .10 * sc]; P['aR'][-2] = P['aR'][-2] + [.0, .04 * sc]
    if pose == 'wave':
        wv = kw.get('waveph', math.sin(t * .7)); P['aR'][-1] = P['aR'][-1] + [.10 * wv, -.03 * abs(wv)]
    paper = pose == 'sit' and kw.get('paper', who == 'nanami')
    if paper: py_ = kw.get('py', .50) - .50; P['aL'] = [np.array(v) for v in [(-.46, .22), (-.86, .34 + .55 * py_), (-1.00, .58 + py_)]]; P['aR'] = [np.array(v) for v in [(.46, .22), (.86, .34 + .55 * py_), (1.00, .56 + py_)]]
    back = q.get('back', False); over = kw.get('over', q.get('over', '')); hk = kw.get('hk', q['hk']); toe = kw.get('toe', q.get('toe', (0, 0))); sole = kw.get('sole', q.get('sole', False))
    if who == 'yuji' and not back: hood_back(d, F, c, seed + 3)
    if who == 'sukuna' and pose not in ('sit', 'crossed'):
        for n, sg in (('lL', -1), ('lR', 1)): leg(d, n0, who, dict(c, pant=c['skin'], pant2=c['skin2'], shoe='#4a3a36', shoe2='#2f2524'), P[n], seed + 5 + sg, toe[0 if sg < 0 else 1], sole, .9)
    elif pose not in ('sit', 'crossed') or who == 'panda':
        leg(d, n0, who, c, P['lL'], seed + 5, toe[0], sole); leg(d, n0, who, c, P['lR'], seed + 6, toe[1], sole)
    if 'L' not in over: arm(d, n0, who, c, P['aL'], seed + 7, hk[0], kw.get('hrL', 0.), flipy=1)
    if 'R' not in over: arm(d, n0, who, c, P['aR'], seed + 8, hk[1], kw.get('hrR', 0.), flipy=-1)
    BODY[who](d, F, c, back, seed + 9, b)
    if who == 'sukuna' and pose in ('sit', 'crossed'):
        Q = n0(spl([(-.62, .86), (-.96, 1.06), (-1.02, 1.28), (-.6, 1.40), (0, 1.36), (.6, 1.40), (1.02, 1.28), (.96, 1.06), (.62, .86)], True, n=7)); d.part(Q, c['cloth'], c['cloth2'], shade=[n0(spl([(-.2, 1.2), (.4, 1.22), (.8, 1.0), (1.3, 1.0), (1.2, 1.6), (-.3, 1.6)], True, n=6))], w=1., seed=seed + 2, tip=False)
        for L in ([(-.50, .92), (-.64, 1.10), (-.90, 1.22)], [(.50, .92), (.64, 1.10), (.90, 1.20)], [(-.06, .98), (-.02, 1.16), (-.10, 1.34)], [(.16, .98), (.3, 1.14), (.34, 1.34)]): d.stroke(n0(spl(L, n=6)), .45, taper=(.15, .4))
    elif pose in ('sit', 'crossed') and who != 'panda':
        leg(d, n0, who, c, P['lL'], seed + 5, 0, True); leg(d, n0, who, c, P['lR'], seed + 6, 0, True)
    if who == 'sukuna': scarf(d, F, seed + 4)
    # head
    hxx, hyy, hr = q['h']; hn = n0.at(hxx + hx, hyy + hy + (-.012 * b if lie else -.01 * b) * deep, hr + hroll, kw.get('hs', 1.))
    M = n0.M[:2, :2]; wv_ = np.linalg.inv(hn.M[:2, :2] / hn.s) @ np.array(wind, float)
    h = Head(hn, q['yaw'] if yaw is None else yaw, q.get('pitch', 0.) if pitch is None else pitch, t, wv_, wamp, gust); h.lid = kw.get('lid'); h.mouth = kw.get('mouth'); h.ear = kw.get('ear', (0., 0.)); h.mb = kw.get('mb', 0.); h.brow = kw.get('brow', 0.); h.vein = kw.get('vein', 0.)
    ex = expr or q['expr']
    HEADS[who](d, h, c, ex, t, seed + 30)
    if h.vein: vein(d, h, s=h.vein)
    if kw.get('sweat', ex == 'shock'): sweat(d, h)
    if not back:
        if who == 'gojo': collar(d, F, c, .17, seed=seed + 60)
        elif who == 'megumi': collar(d, F, c, .10, seed=seed + 60)
        elif who == 'yuji': hood_front(d, F, c, seed + 60)
        elif who == 'nobara': collar(d, F, c, .03, seed=seed + 60)
    if 'L' in over: arm(d, n0, who, c, P['aL'], seed + 7, hk[0], kw.get('hrL', 0.), flipy=1, root='ball')
    if 'R' in over: arm(d, n0, who, c, P['aR'], seed + 8, hk[1], kw.get('hrR', 0.), flipy=-1, root='ball')
    if paper:
        pn = n0.at(kw.get('px', 0.), kw.get('py', .50), kw.get('prot', -.04), 1., .92); tu = kw.get('turn'); newspaper(d, pn, 1.02, .64, seed=seed + 70, t=t, turn=tu, flut=kw.get('flut', 0.))
        for sg in (-1, 1):
            hx_, hy_, hr_ = sg * 1.06, .10, (pi if sg > 0 else 0.)
            if sg > 0 and tu is not None and 0 < tu < 1: u_ = min(1., tu * 1.5); go = math.sin(pi * u_); hx_ = 1.06 * (1 - .95 * go); hy_ = .10 - .40 * go
            hand(d, pn.at(hx_, hy_, hr_, 1., 1.), c, .95, 'fist', seed + sg, flipy=-sg)
    d.lw = lw0
    return dict(head=h, body=n0, F=F, hn=hn)


def put_curl(d, who, x, y, S, rot=0., flip=False, t=0., expr=None, yaw=None, pitch=None, hroll=0., wind=(1., 0.), gust=.06, wamp=.05, breath=None, seed=0, **kw):
    """asleep on the side, knees up, hands under the cheek"""
    c = CAST[who]; lw0 = d.lw; d.lw = S * kw.get('lw', .044); d.lw_ref = d.lw; n0 = N().at(x, y, rot, S * (-1 if flip else 1), S); b = math.sin(2 * pi * t / 36. - .6) if breath is None else breath
    deep = kw.get('deep', 1.); s = 1 + .022 * b * deep; hv = np.linalg.inv(n0.M[:2, :2] / n0.s) @ np.array(wind, float) * kw.get('hem', 0.)
    hw_ = lambda Q_: np.asarray(Q_, float) + np.outer(np.clip((np.asarray(Q_, float)[:, 1] - .84) / .6, 0, 1.2) ** 1.3 * (1 + .6 * np.sin(t * .62 + np.asarray(Q_, float)[:, 0] * 4)), hv)
    pant, pant2 = c['pant'], c['pant2']; cl, cl2 = c['cloth'], c['cloth2']
    if who == 'todo': cl, cl2 = c['skin'], c['skin2']
    if who == 'panda': pant, pant2, cl, cl2 = c['hair'], c['hair2'], c['skin'], c['skin2']
    lg = LEGW[who]; k = {'panda': 1.22, 'todo': 1.1}.get(who, 1.)
    # far leg, back of the body, near leg
    if who == 'yuji': hood_back(d, Fr(n0.at(.30, -.02, -1.1)), c, seed + 3)
    def foot(p, a, sd):
        if who != 'panda': shoe(d, n0.at(*p, a), c if who != 'sukuna' else dict(c, shoe='#4a3a36', shoe2='#2f2524'), .95, sd)
    foot((.60, 1.46), 2.5, seed + 1); limb_part(d, n0, [(1.02, 1.12), (.44, 1.14), (.16, 1.16), (.36, 1.40), (.62, 1.44)], [lg[0] * 1.0 * k, lg[0] * k, lg[1] * k, lg[1] * .9 * k, lg[2] * .8 * k], pant, pant2, seed=seed + 11, folds=1)
    B = spl(np.array([(.10, .02), (.52, -.14), (.98, .02), (1.32, .42), (1.40, .90), (1.18, 1.30), (.80, 1.36), (.50, 1.06), (.30, .70), (.06, .40)]) * [s, 1 + .5 * (s - 1) * (deep > 1)], True, n=8)
    P = n0(B)
    d.part(P, cl, cl2, shade=[n0(spl([(.0, .5), (.46, .62), (.76, .92), (1.1, 1.0), (1.5, .84), (1.4, 1.6), (.4, 1.6)], True, n=7)), n0(spl([(-.2, .1), (.24, .30), (.56, .22), (.62, -.2), (.1, -.3)], True, n=7))], hi=[(n0(spl([(.62, .02), (.96, .14), (1.18, .44), (1.02, .34), (.74, .18)], True, n=6, corners=(0, 2))), _mix(cl, '#8c86b4' if who not in ('nanami', 'sukuna', 'panda', 'todo') else '#ffffff', .35))], w=1., seed=seed, tip=False)
    if who not in ('panda', 'nobara'):
        R_ = n0(spl([(1.6, .92), (1.16, .98), (.86, 1.06), (.60, .92), (.4, 1.4), (1.5, 1.6)], True, n=6, corners=(0, 3, 4, 5))); d.detail(R_, pant, within=P, sh=pant2)
        d.stroke(n0(spl([(1.385, .95), (1.16, .99), (.90, 1.04)], n=6)), .55, taper=(.05, .3))
    if who == 'panda': d.detail(n0(spl([(.0, -.2), (.5, -.3), (.86, -.1), (.74, .3), (.40, .62), (.0, .5)], True, n=7)), c['hair'], within=P)
    if who == 'todo':
        Q = n0(spl([(1.39, .74), (1.40, .98), (1.0, 1.06), (.78, 1.04), (.66, .90), (1.0, .84)], True, n=5, corners=(0, 1, 3, 4))); d.part(Q, '#f2ede4', '#b8b0ba', w=.7, tip=False)
    for L in ([(1.30, .46), (1.06, .56), (.86, .72)], [(1.38, .86), (1.14, .86), (.96, .96)], [(.34, .70), (.52, .62), (.66, .60)]): d.stroke(n0(spl(L, n=6)), .42, taper=(.08, .6))
    foot((.44, 1.70), 2.75, seed + 2)
    if who == 'nobara': pass
    limb_part(d, n0, [(1.10, 1.10), (.56, 1.06), (.02, 1.12), (.24, 1.50), (.48, 1.66)], [lg[0] * 1.15 * k, lg[0] * 1.1 * k, lg[1] * 1.02 * k, lg[1] * .95 * k, lg[2] * .85 * k], pant, pant2, seed=seed + 12, folds=2, rnd=who == 'panda', root='round' if who == 'panda' else 'ball', outer='R')
    if who == 'nobara':                                      # the skirt falls over the thigh
        K = n0(hw_(spl([(.62, .78), (.98, .80), (1.34, .78), (1.50, 1.10), (1.32, 1.44), (1.02, 1.52), (.72, 1.40), (.46, 1.36), (.36, 1.08), (.48, .88)], True, n=7))); d.part(K, cl, cl2, shade=[n0(spl([(.5, 1.2), (.9, 1.12), (1.5, .9), (1.6, 1.7), (.4, 1.6)], True, n=6))], w=1., seed=seed + 2, tip=False)
        for L in ([(1.24, .84), (1.34, 1.1), (1.26, 1.42)], [(1.06, .84), (1.10, 1.1), (1.02, 1.46)], [(.88, .84), (.82, 1.1), (.76, 1.36)], [(.70, .84), (.58, 1.06), (.52, 1.30)]): d.stroke(n0(hw_(spl(L, n=6))), .45, taper=(.15, .25))
    hn = n0.at(-.42 + kw.get('hx', 0.), -.28 + kw.get('hy', 0.) - .014 * b * (deep - 1), -1.18 + hroll, kw.get('hs', 1.)); wv_ = np.linalg.inv(hn.M[:2, :2] / hn.s) @ np.array(wind, float)
    h = Head(hn, .16 if yaw is None else yaw, 0. if pitch is None else pitch, t, wv_, wamp, gust); h.mouth = kw.get('mouth'); h.mb = kw.get('mb', 0.); h.brow = kw.get('brow', 0.); h.vein = kw.get('vein', 0.)
    # lower arm goes under the head, upper arm over
    arm(d, n0, who, c, [(.50, .34), (.30, .72), (-.10, .74)], seed + 14, 'flat', 0.)
    HEADS[who](d, h, c, expr or 'sleep', t, seed + 30)
    if h.vein: vein(d, h, s=h.vein)
    au = kw.get('armU')
    if au is None: arm(d, n0, who, c, [(.62, .30), (.54, .64), (.14, .62)], seed + 15, 'flat', -.2)
    else: arm(d, n0, who, c, au, seed + 15, kw.get('hkU', 'open'), kw.get('hrU', 0.), root='ball')
    d.lw = lw0
    return dict(head=h, body=n0, hn=hn)


# ================================================================ small things
def bubble(base, x, y, r, t=0., ax=None, ay=None, tint='#dff3ff', pop=None, squash=0.):
    """sleep bubble on an RGB frame: clear film, sky in the upper left, warm bounce below, ink rim.  (ax, ay) = nostril.
    pop = 0..3 draws the burst instead"""
    if r < 2 and pop is None: return base
    R = int(max(r, 10) * 2.6 + 50); x0, y0 = int(x - R), int(y - R); w = h = 2 * R
    c = Cel(w, h, x0, y0, 2); m = Cel(w, h, x0, y0, 2)
    if pop is not None:
        q = np.random.default_rng(8)
        if pop < 1: E = ell(x, y, r * 1.06, r * 1.06, n=60); c.fill(E, '#ffffff', .55)
        for k in range(13):                                          # film tears into curved shreds
            a = 2 * pi * k / 13 + q.uniform(-.16, .16); r0 = r * (.96 + .24 * pop) * q.uniform(.92, 1.1); ln = r * q.uniform(.22, .42) * (1 - .18 * pop); wd = max(1.2, r * q.uniform(.06, .13) * (1 - .22 * pop)); al = max(0., 1 - .2 * pop)
            p0 = (x + math.cos(a) * r0, y + math.sin(a) * r0); p1 = (x + math.cos(a) * (r0 + ln), y + math.sin(a) * (r0 + ln))
            c.line([p0, p1], wd * 1.9, '#4f7f9f', taper=(.05, .6), alpha=al * .55); c.line([p0, p1], wd, '#ffffff', taper=(.05, .6), alpha=al)
            if k % 2 == 0:
                rr = r0 + ln * (1.5 + .5 * pop); yy = y + math.sin(a) * rr + pop * pop * r * .06; c.fill(ell(x + math.cos(a) * rr, yy, r * .055 + 1.2), '#4f7f9f', al * .6); c.fill(ell(x + math.cos(a) * rr - .6, yy - .6, r * .045 + .5), '#ffffff', al)
        l = c.rgba()
    else:
        wb = 1 + .025 * math.sin(t * .5) + squash; E = ell(x, y, r * wb, r / wb, n=60)
        if ax is not None:
            d = np.array([ax - x, ay - y], float); L = np.hypot(*d) + 1e-6; e = d / L; nr = np.array([-e[1], e[0]])
            A_ = np.array([x, y]) + e * r * .92 + nr * r * .22; B_ = np.array([x, y]) + e * r * .92 - nr * r * .22; tip = np.array([ax, ay], float); mid = (A_ + B_) / 2 * .45 + tip * .55
            q1 = spl([A_, mid + nr * r * .06, tip], n=8); q2 = spl([tip, mid - nr * r * .06, B_], n=8); c.fill(np.vstack([q1, q2]), '#ffffff', .62)
            for q_ in (q1, q2[::-1]): c.line(q_, max(1.1, r * .035), '#4f7f9f', taper=(.1, .3), seed=2, alpha=.9)
        c.fill(E, tint, .22)
        c.fill(spl([(x - r * .80, y + r * .06), (x - r * .70, y - r * .46), (x - r * .22, y - r * .82), (x + r * .34, y - r * .76), (x + r * .02, y - r * .66), (x - r * .44, y - r * .38), (x - r * .64, y + r * .10)], True, n=7, corners=(0, 3)), '#ffffff', .74)
        c.fill(ell(x - r * .60, y + r * .34, r * .075, r * .11, -.4), '#ffffff', .85)
        c.fill(spl([(x + r * .02, y + r * .88), (x + r * .56, y + r * .62), (x + r * .86, y + r * .04), (x + r * .74, y + r * .34), (x + r * .44, y + r * .62)], True, n=7, corners=(0, 2)), '#ffd9ec', .62)
        c.fill(spl([(x + r * .30, y + r * .70), (x + r * .64, y + r * .42), (x + r * .80, y + r * .10), (x + r * .64, y + r * .30), (x + r * .48, y + r * .50)], True, n=7, corners=(0, 2)), '#c9fbff', .6)
        c.line(np.vstack([E, E[:2]]), max(2.2, r * .075), '#ffffff', taper=(0, 0), press=.3, seed=4, alpha=.75)
        c.line(np.vstack([E, E[:2]]), max(1.1, r * .030), '#4f7f9f', taper=(0, 0), press=.5, seed=3, alpha=.9)
        l = c.rgba()
    from jj.ink import over_roi
    return over_roi(base, l, x0, y0)


def butterfly(d, x, y, S, t, head=0., pitch=.9, flap=None, col='#fffef8', col2='#c8d3e2', speed=1.0, seed=0):
    """white butterfly; wings are planes hinged on the body, seen by a camera pitched down.  Returns px of the body."""
    if flap is None:
        u = (t * speed / 7.) % 1.0                                   # quick down-stroke, slower recovery
        flap = math.cos(pi * (u / .38)) if u < .38 else -math.cos(pi * ((u - .38) / .62))
    phi = .55 + .75 * flap                                           # wing dihedral, radians above the plane
    cp, sp = math.cos(pitch), math.sin(pitch); ch, sh = math.cos(head), math.sin(head)
    def pr(p3):
        p3 = np.asarray(p3, float); X = p3[:, 0] * ch - p3[:, 1] * sh; Y = p3[:, 0] * sh + p3[:, 1] * ch; Z = p3[:, 2]
        return np.c_[x + S * X, y + S * (Y * sp - Z * cp)]
    up = np.array([(.10, .02), (.34, .40), (.58, .86), (.52, 1.04), (.22, 1.08), (-.06, .84), (-.10, .42), (-.04, .04)]); lo = np.array([(-.02, .03), (-.14, .44), (-.26, .80), (-.52, .84), (-.66, .58), (-.56, .26), (-.22, .03)])
    lw0 = d.lw; d.lw = max(1.4, S * .06)
    order = [-1, 1] if sh * math.cos(phi) <= 2 else [1, -1]
    def wing(W_, sg, ph, i):
        Q = spl(W_, True, n=6); p3 = np.c_[Q[:, 0], sg * Q[:, 1] * math.cos(ph), Q[:, 1] * math.sin(ph)]; P = pr(p3)
        lit = 1 if sg * math.cos(head) < 0 else 0
        inn = np.c_[Q[:, 0] * .72 + W_[:, 0].mean() * .10, Q[:, 1] * .62]; p4 = np.c_[inn[:, 0], sg * inn[:, 1] * math.cos(ph), inn[:, 1] * math.sin(ph)]
        d.part(P, col if not lit else '#f2f3f4', col2, shade=[pr(p4)] if math.sin(ph) > .5 else (), w=.8, ink='#4d4a58', tip=False, seed=seed + i)
        for k in (2, 3, 4):
            if k < len(W_): e = W_[k]; d.stroke(pr(np.array([(W_[0][0], sg * W_[0][1], 0), (e[0] * .8, sg * e[1] * .8 * math.cos(ph), e[1] * .8 * math.sin(ph))])), .3, '#9aa0b2', taper=(.1, .6))
        if W_ is up:
            e = np.array([(.40, .80)]); d.detail(pr(np.c_[ell(.34, .86, .10, .075)[:, 0], sg * ell(.34, .86, .10, .075)[:, 1] * math.cos(ph), ell(.34, .86, .10, .075)[:, 1] * math.sin(ph)]), '#6a6878', within=P)
    for sg in order: wing(lo, sg, phi * .86, 1 + sg); wing(up, sg, phi, 4 + sg)
    b = pr(np.array([(.22, 0, .02), (.0, 0, .05), (-.34, 0, 0)])); d.stroke(spl(b, n=6), 1.9, '#3a3540', taper=(.15, .4), tmin=.4)
    for sg in (-1, 1): d.stroke(spl(pr(np.array([(.22, 0, .04), (.36, sg * .10, .14), (.48, sg * .20, .20)])), n=6), .45, '#3a3540', taper=(0, .2))
    d.lw = lw0
    return np.array([x, y])


def zzz(d, x, y, S, t, n=3, col='#ffffff', drift=(.55, -.9), edge=None):
    """hand-lettered z's that drift up, grow and turn; each lives 54 frames"""
    r = np.random.default_rng(5); lw0 = d.lw
    for k in range(n):
        u = ((t / 54.) + k / n) % 1.0; s = S * (.45 + .75 * u); px, py = x + drift[0] * S * 3.9 * u + S * .16 * math.sin(u * 6 + k), y + drift[1] * S * 3.4 * u
        a = -.18 + .30 * math.sin(u * 5 + k * 2); ca, sa = math.cos(a), math.sin(a); al = min(1, u * 6) * min(1, (1 - u) * 4)
        Z = np.array([(-.42, -.40), (-.10, -.44), (.36, -.42), (.10, -.08), (-.40, .38), (-.02, .36), (.44, .40)])
        seg = [spl(Z[:3], n=6), spl(Z[2:5], n=6), spl(Z[4:], n=6)]
        for j, q in enumerate(seg):
            q = np.c_[q[:, 0] * ca - q[:, 1] * sa, q[:, 0] * sa + q[:, 1] * ca] * s + [px, py]
            if edge: d.ln.line(q, s * .26, edge, taper=(.02, .02), press=.2, seed=k * 3 + j, alpha=al)
        for j, q in enumerate(seg):
            q = np.c_[q[:, 0] * ca - q[:, 1] * sa, q[:, 0] * sa + q[:, 1] * ca] * s + [px, py]
            d.ln.line(q, s * .16, col, taper=(.04, .06), press=.3, seed=k * 3 + j, alpha=al, tmin=.5)
    d.lw = lw0


# ================================================================ a pile of them
def solo(who, pose, x, y, S, pad=3.0, **k):
    """one chibi on its own ROI -> (rgba, x0, y0, info)"""
    P = int(S * pad); d = D(int(x - P), int(y - P), 2 * P, 2 * P, 2); info = put(d, who, pose, x, y, S, **k); return d.rgba(), d.x0, d.y0, info


def stack(items, sun=(7, 9), dark=(.62, .56, .70), soft=2.0, amt=.85):
    """items = [(rgba, x0, y0)] back to front -> full-frame premultiplied RGBA; each one shades those below it"""
    acc = np.zeros((H, W, 4), f32); dk = 1 - np.array(dark, f32)
    for rgba, x0, y0 in [i[:3] for i in items]:
        h, w = rgba.shape[:2]; m = 24; X0, Y0, X1, Y1 = max(0, x0 - m), max(0, y0 - m), min(W, x0 + w + m), min(H, y0 + h + m)
        if X1 <= X0 or Y1 <= Y0: continue
        a = np.zeros((Y1 - Y0, X1 - X0), f32); sx0, sy0 = max(0, X0 - x0), max(0, Y0 - y0); dx, dy = max(0, x0 - X0), max(0, y0 - Y0)
        hh, ww = min(h - sy0, a.shape[0] - dy), min(w - sx0, a.shape[1] - dx)
        if hh <= 0 or ww <= 0: continue
        L = np.zeros(a.shape + (4,), f32); L[dy:dy + hh, dx:dx + ww] = rgba[sy0:sy0 + hh, sx0:sx0 + ww]; a = L[..., 3]
        sh = cv2.warpAffine(a, np.float32([[1, 0, sun[0]], [0, 1, sun[1]]]), (a.shape[1], a.shape[0])); sh = np.maximum(sh, cv2.dilate(a, np.ones((3, 3), np.uint8), iterations=3) * .8); sh = cv2.GaussianBlur(sh, (0, 0), soft) * amt
        R = acc[Y0:Y1, X0:X1]; R[..., :3] *= 1 - sh[..., None] * dk; R[:] = R * (1 - a[..., None]) + L
    return acc


def daylight(L, y0=0., y1=float(H), bounce='#dcefbe', amt=.5):
    """para pass on the whole cel: grass bounce from below, clean light above"""
    from jj.kit import para
    return para(L, (W * .46, y0), (W * .54, y1), [(0, '#ffffff'), (.45, '#ffffff'), (1, bounce)], 'mul', amt)


def stamp(base, d, op=1.0, **k):
    """composite a small D straight onto an RGB frame (in place, only its own rectangle)"""
    from jj.ink import over_roi
    return over_roi(base, d.rgba(**k), d.x0, d.y0, op)


# ================================================================ camera + b3 insert
def through(M, x, y, S, rot=0.):
    """painting position -> (screen x, y, S, rot) under a Hill camera matrix"""
    z = math.hypot(M[0, 0], M[1, 0]); return M[0, 0] * x + M[0, 1] * y + M[0, 2], M[1, 0] * x + M[1, 1] * y + M[1, 2], S * z, rot + math.atan2(M[1, 0], M[0, 0])


_st = {}


def stare_insert(f):
    """b3, local frames 54..62 (0..8 also accepted): Nobara and Megumi look at us, dot-eyed; one slow blink (57-60).  -> float RGB (1080, 1920, 3)"""
    from jj.kit import screentone, cast
    from jj.comp import over
    from jj.paint import fbm, grid
    k = int(f) - 54 if f >= 54 else int(f)
    if 'bg' not in _st:
        x, y = grid(); d_ = (x * .62 + y * .78)
        bg = np.empty((H, W, 3), f32); bg[:] = rgb('#f3dfae')
        n = fbm(H, W, (260, 70, 9), (1, .4, .12), 5); dark = np.clip(.02 + .40 * smooth(.30, 1.0, (x / W * .55 + y / H * .75) + .04 * n), 0, .42)
        tone = screentone(np.ones((H, W), f32), 19., .62, dark)
        bg = bg * (1 - tone[..., None]) + rgb('#e9b96d') * tone[..., None]
        c = Cel()
        r = np.random.default_rng(3)                                  # a lazy horizon of hand-ruled strokes and three dots of silence
        for i in range(26):
            yy = 120 + r.uniform(-60, 60) + i * 3; x0 = r.uniform(-60, 300) if i % 2 else r.uniform(1500, 1860)
            c.line([(x0, yy), (x0 + r.uniform(120, 420), yy + r.uniform(-3, 3))], r.uniform(1.5, 4), '#d79f55', taper=(.1, .7), seed=i)
        bg = over(bg, c.layer()); _st['bg'] = bg
    key = 'blink' if k in (3, 4, 5) else 'open'
    kk = (key, k // 4)
    if kk not in _st:
        out = []
        for who, x, y, rot, yw, sd in (('nobara', 590, 770, .035, .07, 7), ('megumi', 1340, 790, -.03, -.07, 8)):
            d = D(x - 620, y - 880, 1240, 1240, 2)
            put(d, who, 'stare', x, y + (3 if k // 4 else 0), 268, rot=rot, t=0, expr='dotblink' if key == 'blink' else 'dot', yaw=yw, wind=(1, 0), gust=.0, wamp=.0, seed=sd, breath=0., sweat=False, lw=.036)
            out.append((d.rgba(outer=None), d.x0, d.y0))
        _st[kk] = stack(out, sun=(0, 0), amt=0.)
    L = _st[kk]
    im = cast(_st['bg'], L, 26, 22, '#c98a45', .85, 0)                 # flat printed shadow
    im = over(im, L)
    c = Cel()                                                          # "..." drawn one dot at a time
    for i in range(min(3, 1 + k // 2)):
        c.fill(ell(520 + i * 74, 112 + (i % 2) * 5, 19, 17.5, .3 * i), '#3a2418')
    return over(im, c.layer())
