"""Shared drawing hand for Megumi and Nobara: light modes, 3-D hair locks, eyes, brows, nose, mouth, face shading, uniform bust.
Everything is authored in jj.head3 front coordinates and projected, so every angle comes from the same drawing.
Not a character by itself: see chars/megumi.py and chars/nobara.py."""
import numpy as np, cv2, math, cairo
from jj import W, H as FH
from jj.ink import Cel, spl, resample, ribbon, noise1
from jj.kit import outline, shape, spike, rim, para, tint, mask_of, hatch
from jj.paint import rgb, mix
from jj.head3 import Head

INK = '#17131a'


# ------------------------------------------------------------------ light modes
# skin: base, shadow, deep (occlusion), light;  line = face line;  key = screen direction the light COMES from (x right, y down)
LIGHT = {
    'day':    dict(skin='#f9dfce', skin2='#dfa898', skin3='#b8786f', skinhi='#fff1df', blush='#eea690', white='#fbf4ea', white2='#cdbcc0', line='#2a1a1c', lip='#c98579',
                   cloth='#232238', cloth_hi='#4b4778', cloth_dk='#15141f', cloth_ln='#0b0a12', key=(-.75, -.65), rim=None, metal='#e8b64c',
                   para=[(0, '#fff6e2'), (.55, '#ffffff'), (1, '#e6d6e6')]),
    'sunset': dict(skin='#f7bf95', skin2='#b3636a', skin3='#7c3f55', skinhi='#ffe0ad', blush='#e37e5c', white='#f6dcc4', white2='#b48c98', line='#2c1218', lip='#b4605c',
                   cloth='#2a2036', cloth_hi='#9a5560', cloth_dk='#17111d', cloth_ln='#0d080f', key=(-.9, -.15), rim=('#ffc064', 4, 1, .9), metal='#ffc25a',
                   para=[(0, '#ffe4b8'), (.5, '#ffffff'), (1, '#b99ac0')]),
    'under':  dict(skin='#b5b7c2', skin2='#6f7191', skin3='#474a6a', skinhi='#d6dde0', blush='#a892a0', white='#cdd5da', white2='#858aa4', line='#171523', lip='#85708a',
                   cloth='#1c1d2c', cloth_hi='#3b4266', cloth_dk='#101019', cloth_ln='#08080e', key=(.35, -.95), rim=None, metal='#9d9270',
                   para=[(0, '#eef4f4'), (.5, '#d2d8e4'), (1, '#6a6f8e')]),
    'battle': dict(skin='#d3c6cf', skin2='#6a6394', skin3='#403a68', skinhi='#cfeaff', blush='#b98c9c', white='#e6ebf4', white2='#8c8fb8', line='#14101f', lip='#8e6a86',
                   cloth='#1b1a2c', cloth_hi='#41508f', cloth_dk='#0e0d16', cloth_ln='#06060b', key=(-.8, -.35), rim=('#62c8ff', -4, 1, .95), metal='#c7a95a',
                   para=[(0, '#f2f6ff'), (.5, '#dfe2f4'), (1, '#7780b8')]),
    'snow':   dict(skin='#f1dfdb', skin2='#b4a3bd', skin3='#857799', skinhi='#fff6ec', blush='#e6a3a0', white='#f4f3f5', white2='#b9b8cc', line='#26202c', lip='#b98a92',
                   cloth='#272a3c', cloth_hi='#59628c', cloth_dk='#171824', cloth_ln='#0c0c14', key=(.7, -.6), rim=('#ffe6c4', -3, 1, .8), metal='#cfb06a',
                   para=[(0, '#fff8ee'), (.5, '#f2f2fb'), (1, '#a9b2d2')]),
}

LIGHT['blue'] = dict(LIGHT['battle'], skin='#ecd3cb', skin2='#8b74ac', skin3='#4d4280', skinhi='#f7f3f4', blush='#d58f96', white='#f2f4fa', white2='#9ea0c8', lip='#a5627c', cloth='#1d1c33', cloth_hi='#4a5cb4', cloth_dk='#100f1b', key=(.80, -.30), rim=('#74d0ff', 4, 2, .95),
                     para=[(0, '#ffffff'), (.5, '#f4f0fb'), (1, '#8f92d2')])

# golden afternoon, sun behind-left: warm open shade on the faces, hard rim from the left
LIGHT['gold'] = dict(LIGHT['day'], skin='#e3a587', skin2='#bd7468', skin3='#8f4f50', skinhi='#ffe2b2', lit='#ffdcae', blush='#d97862', white='#f6e2cc', white2='#b5948f', line='#2a1216', lip='#a9584f', cloth='#25223f', cloth_hi='#5a5089', cloth_dk='#141226', cloth_ln='#0b0910', key=(-.85, -.40), rim=None, metal='#f0b64e',
                     para=[(0, '#fff2d8'), (.5, '#ffffff'), (1, '#c3aee0')])


def lw(s):
    """line weights (outer, inner, fine) for a head drawn at s px per unit"""
    return float(np.clip(1.0 + s * .0140, 1.2, 6.2)), float(np.clip(.75 + s * .0062, .85, 3.2)), float(np.clip(.6 + s * .0036, .7, 2.0))


# ------------------------------------------------------------------ 3-D helpers
def surf(H, theta, y, lift=0.):
    """point on the skull at azimuth theta (0 front, +right) and height y, plus outward normal"""
    A = np.interp(y, H.yy, H.A); B = np.interp(y, H.yy, H.F if math.cos(theta) >= 0 else H.B); p = H.k['p']
    sx, cz = math.sin(theta), math.cos(theta); r = (abs(sx) ** p + abs(cz) ** p) ** (-1 / p)
    q = np.array([A * sx * r, y, B * cz * r]); n = np.array([sx / max(A, .1), (y - .1) * .95, cz / max(B, .1)]); n /= np.linalg.norm(n)
    return q + n * lift, n


def scalp(H, hl, infl=1.05, ny=22, nf=73, top=1.0, front=True, rise=0., ell=None):
    """screen contour of the hair-covered part of the skull; hl(theta) = hairline height.  front = only the part that faces the camera"""
    from jj.kit import contours
    th = np.linspace(-math.pi, math.pi, nf); M = np.zeros((ny, nf, 3)); N = np.zeros((ny, nf, 3))
    for j, t in enumerate(th):
        for i, y in enumerate(np.linspace(hl(t), top, ny)):
            q, n = surf(H, t, y); M[i, j] = q * [infl, 1, infl] + [0, (infl - 1 + rise) * max(y, 0), 0]; N[i, j] = n
    if ell is not None:
        c, rr = np.array(ell[0]), np.array(ell[1]); D = (M - c) / rr; M = c + rr * D / (np.linalg.norm(D, axis=-1, keepdims=True) + 1e-9)
    if not front: return H._sil([M])
    vis = (N @ H.R.T)[..., 2] > -.12; vis = vis[:-1, :-1] & vis[1:, :-1] & vis[:-1, 1:] & vis[1:, 1:]
    Q = H.P3(M); k = max(.5, min(1., 900. / (H.s * 3.2))); lo = Q.reshape(-1, 2).min(0) - 8; hi = Q.reshape(-1, 2).max(0) + 8
    w, h = np.ceil((hi - lo) * k).astype(int) + 2; m = np.zeros((h, w), np.uint8); q = (Q - lo) * k
    quad = np.stack([q[:-1, :-1], q[:-1, 1:], q[1:, 1:], q[1:, :-1]], 2)[vis]
    for c in np.round(quad * 16).astype(np.int32): cv2.fillConvexPoly(m, c, 255, cv2.LINE_8, 4)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)); cs = contours(m / 255.)
    return resample(max(cs, key=len) / k + lo, 3.0) if cs else Q[0, :3]


def _cr3(P, n=7):
    """Catmull-Rom through 3-D points"""
    P = np.asarray(P, float); E = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]]); out = []
    for i in range(len(P) - 1):
        a, b, c, d = E[i:i + 4]; t = np.linspace(0, 1, n, endpoint=False)[:, None]
        out.append(.5 * (2 * b + (c - a) * t + (2 * a - 5 * b + 4 * c - d) * t * t + (3 * b - a - 3 * c + d) * t ** 3))
    return np.vstack(out + [P[-1:]])


class Lock:
    """one hair lock: 3-D centre line + width profile -> screen polygon.  kind: 'spike' thorn, 'bang' blade, 'sheet' ribbon with a cut end"""
    __slots__ = ('p0', 'p1', 'p2', 'n', 'w', 'kind', 'asym', 'flex', 'id', 'Q', 'A', 'B', 'poly', 'depth', 'vis', 'w1', 'tag', 'pr', 'path')

    def __init__(self, H, theta, y, p1, p2, w, kind='spike', asym=0., flex=1., lift=.02, w1=None, tag='', id=0):
        self.p0, self.n = surf(H, theta, y, lift); self.p1, self.p2 = np.asarray(p1, float), np.asarray(p2, float)
        self.w, self.kind, self.asym, self.flex, self.id, self.w1, self.tag = w, kind, asym, flex, id, w1, tag; self.path = None

    @staticmethod
    def sp(H, theta, y, d, L, w, curl=(0, 0, 0), arc=(0, 0, 0), **k):
        """free spike: direction d, length L, tip curl, middle arc (3-D)"""
        p0, n = surf(H, theta, y, k.get('lift', .02)); d = np.asarray(d, float); d = d / (np.linalg.norm(d) + 1e-9)
        return Lock(H, theta, y, p0 + d * L * .5 + np.asarray(arc, float), p0 + d * L + np.asarray(curl, float), w, **k)

    @staticmethod
    def fr(H, theta, y, tip, w, dz=.06, bulge=.16, up=.0, **k):
        """fringe: from the scalp over the forehead to a tip given in face-front coordinates"""
        p0, n = surf(H, theta, y, k.get('lift', .02)); tx, ty = tip[0], tip[1]
        p2 = np.array([tx, ty, float(H.z(np.clip(tx, -.66, .66), ty)) + dz]) if len(tip) < 3 else np.array(tip, float)
        p1 = (p0 + p2) / 2 + np.array([n[0] * bulge, up + n[1] * bulge * .6, max(n[2], .55) * bulge * 1.6]); p1[1] += .06
        return Lock(H, theta, y, p1, p2, w, kind=k.pop('kind', 'bang'), **k)

    @staticmethod
    def pa(H, pts, w, kind='bob', n=None, **k):
        """lock along explicit 3-D points"""
        P = np.asarray(pts, float); l = Lock(H, 0., .5, P[len(P) // 2], P[-1], w, kind=kind, **k); l.p0 = P[0]; l.path = P
        c = P[len(P) // 2] - [0, .15, -.05]; l.n = c / np.linalg.norm(c) if n is None else np.asarray(n, float); return l

    def build(self, H, off=(0, 0, 0), m=22):
        t = np.linspace(0, 1, m)[:, None]; p0, p1, p2 = self.p0, self.p1, self.p2
        if self.path is not None:
            q = np.column_stack([spl(self.path[:, :2], n=8), spl(self.path[:, 1:], n=8)[:, 1]]) if False else _cr3(self.path)
            d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]; v = t[:, 0] * d[-1]
            C = np.stack([np.interp(v, d, q[:, i]) for i in range(3)], -1) + t ** 2 * np.asarray(off, float) * self.flex
        else: C = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t * t * p2 + t ** 2 * np.asarray(off, float) * self.flex
        Q = H.P3(C); self.depth = float(H.depth(C[m // 3])); self.vis = float((H.R @ self.n)[2])
        return self._poly(Q, self.w * H.s)

    def _poly(self, Q, wpx):
        m = len(Q); u = np.linspace(0, 1, m); T = np.gradient(Q, axis=0); ln = np.hypot(T[:, 0], T[:, 1]); tot = ln.sum()
        T = T / (ln[:, None] + 1e-6)
        if tot < wpx * .6:                     # foreshortened: steady the normals
            g = Q[-1] - Q[0]; g = g / (np.hypot(*g) + 1e-6); T = T * .3 + g * .7; T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-6
        N = np.c_[-T[:, 1], T[:, 0]]; w = wpx * .5
        if self.kind == 'spike': pr = (1 - u) ** 1.12 * (1 + .30 * np.sin(math.pi * np.clip(u * 1.3, 0, 1)))
        elif self.kind == 'flame': pr = np.interp(u, [0, .12, .35, .62, .84, 1], [.40, .78, 1.0, .74, .30, 0])
        elif self.kind == 'ink': pr = np.interp(u, [0, .10, .28, .50, .72, .88, 1], [.45, .80, 1.0, .74, .38, .14, 0])
        elif self.kind == 'flip': pr = np.interp(u, [0, .12, .34, .56, .74, .88, 1], [.06, .46, .90, 1.0, .74, .34, 0])
        elif self.kind == 'swp': pr = np.interp(u, [0, .10, .25, .58, .84, 1], [.06, .42, .84, 1.0, .52, 0])
        elif self.kind == 'bang': pr = (.70 + .30 * np.sin(math.pi * np.clip(u * .9, 0, 1))) * np.clip(1 - u ** 1.9, 0, 1) ** 1.0
        elif self.kind == 'sheet': pr = np.interp(u, [0, .15, .5, .85, 1], [.75, .95, 1.05, 1.0, .0 if self.w1 is None else self.w1])
        elif self.kind == 'bob': pr = np.interp(u, [0, .10, .24, .45, .72, .90, 1], [.10, .52, .86, 1.0, .96, .58, 0])
        else: pr = np.interp(u, [0, .2, .7, 1], [.6, 1, .75, 0])
        b = self.asym * np.sin(math.pi * u) * w
        self.Q = Q; self.A = Q + N * (w * pr + b)[:, None]; self.B = Q - N * (w * pr - b)[:, None]
        self.poly = np.vstack([self.A, self.B[::-1]])
        return self

    @staticmethod
    def d2(pts, wpx, kind='flame', asym=0., id=0, tag='', m=20, rel=.2, depth=0.):
        """lock drawn straight in screen px along a 2-D polyline"""
        l = Lock.__new__(Lock); l.kind, l.asym, l.id, l.tag, l.w1, l.w, l.depth, l.vis = kind, asym, id, tag, None, rel, depth, 1.
        q = spl(np.asarray(pts, float), n=8) if len(pts) > 2 else np.asarray(pts, float); d = np.r_[0, np.cumsum(np.hypot(*np.diff(q, axis=0).T))]; v = np.linspace(0, d[-1], m)
        return l._poly(np.c_[np.interp(v, d, q[:, 0]), np.interp(v, d, q[:, 1])], wpx)

    def side(self, t0, t1, a, b):
        """sliver inside the lock between fractions a..b across (-1 = A side, 1 = B side) for t0..t1"""
        n = len(self.Q); i0, i1 = int(t0 * (n - 1)), int(t1 * (n - 1)) + 1
        mid = (self.A + self.B) / 2; h = (self.B - self.A) / 2
        k = np.sin(np.linspace(0, math.pi, i1 - i0))[:, None]
        c = (a + b) / 2; P = mid[i0:i1] + h[i0:i1] * (c + (a - c) * k); Q = mid[i0:i1] + h[i0:i1] * (c + (b - c) * k)
        return np.vstack([P, Q[::-1]])


def wind_off(H, wind, f, ph, amp=1.0):
    """3-D tip offset of a lock for a screen wind (wx, wy) in head units + flutter"""
    wx, wy = wind[0], -wind[1]; g = wind[2] if len(wind) > 2 else .25
    v = math.hypot(wx, wy) + 1e-6
    a = f * 2 * math.pi / 24.
    fl = g * v * (.65 * math.sin(a * 1.9 + ph * 6.28) + .35 * math.sin(a * 3.7 + ph * 17.1))
    s = np.array([wx + fl * (-wy / v), wy + fl * (wx / v), 0.]) * amp
    return H.R.T @ s


# ------------------------------------------------------------------ inked hair group
def hair_group(cel, locks, cols, wo, wi, wf, seed=0, key=(-.7, -.7), caps=(), line=INK, shade=True, inner=.35):
    """locks sorted far->near.  cols = dict(base, sh, hi, hi2?).  One heavy outer contour, thin broken inner lines, one shadow and one light shape per lock."""
    if cols.get('cast') is not None: cel.ctx.push_group()
    for c in caps: outline(cel, c, wo * 1.7, line, seed + 3, tremor=.2)
    for i, l in enumerate(locks): outline(cel, l.poly, wo * 1.75 * min(1., .30 + float(np.hypot(*(l.A - l.B).T).max()) / (wo * 5.)), line, seed + 11 * i + l.id, tremor=.2)
    for c in caps: cel.fill(c, cols['base'])
    for l in locks: cel.fill(l.poly, cols['base'])
    kx, ky = key; kn = math.hypot(kx, ky); kx, ky = kx / kn, ky / kn
    for i, l in enumerate(locks):
        r = np.random.default_rng(l.id * 131 + seed + 7)
        if cols.get('cast') is not None and l.w > .09:
            cel.ctx.set_operator(cairo.OPERATOR_ATOP); cel.fill(l.poly + np.asarray(cols['cast'], float), cols['sh']); cel.ctx.set_operator(cairo.OPERATOR_OVER)
        cel.fill(l.poly, cols['base'])
        # which side faces the light
        m = len(l.Q) // 2; nv = l.A[m] - l.B[m]; lit = 1 if nv @ np.array([kx, ky]) > 0 else -1
        if shade:
            a = r.uniform(.10, .42)
            if lit > 0: cel.fill(l.side(.0, r.uniform(.82, .97), a, 1.05), cols['sh'])
            else: cel.fill(l.side(.0, r.uniform(.82, .97), -1.05, -a), cols['sh'])
        if cols.get('wedge') and r.random() < cols['wedge']:
            a0 = r.uniform(.02, .12); a1 = r.uniform(.55, .85); e = r.uniform(.30, .62)
            cel.fill(l.side(a0, a1, 1.02, e) if lit < 0 else l.side(a0, a1, -1.02, -e), cols['hi'])
            if cols.get('hi2') and r.random() < .45: cel.fill(l.side(a0 + .12, a1 - .08, 1.0, .86) if lit < 0 else l.side(a0 + .12, a1 - .08, -1.0, -.86), cols['hi2'])
        elif cols.get('hi') and l.w > .10 and r.random() < cols.get('hip', .7):
            c = -r.uniform(.25, .6) * lit if lit > 0 else r.uniform(.25, .6)
            t0 = r.uniform(.10, .3); hw = r.uniform(.07, .16) * cols.get('hiw', 1.)
            cel.fill(l.side(t0, t0 + r.uniform(.22, .42), c - hw, c + hw), cols['hi'])
        if cols.get('stroke'):
            nq = len(l.Q)
            for q in range(int(r.integers(2, 5))):
                c = r.uniform(-.8, .8); t0 = r.uniform(.02, .35); t1 = min(.96, t0 + r.uniform(.25, .5)); i0, i1 = int(t0 * nq), int(t1 * nq); g = (l.A[i0:i1] * (1 - c) + l.B[i0:i1] * (1 + c)) / 2
                if len(g) > 3: cel.line(g, wf * r.uniform(.8, 1.5), cols['stroke'], taper=(.5, .5), press=.3, seed=l.id * 17 + q, alpha=.8)
        bd = cols.get('band')
        if bd and l.w > .16 and l.tag in bd[2] and r.random() < bd[3]:
            n_ = len(l.Q) - 1; W_ = lambda t, c: np.array([np.interp(t * n_, np.arange(n_ + 1), (l.A[:, j] * (1 - c) + l.B[:, j] * (1 + c)) / 2) for j in (0, 1)])
            t0, t1 = bd[0] + r.uniform(-.015, .015), bd[1] + r.uniform(-.01, .015); c0, c1 = sorted((r.uniform(-.92, -.5), r.uniform(.30, .95))); k = 4; xs = np.linspace(c0, c1, 2 * k + 1)
            up = [W_(np.clip(t0 + (.012 if q % 2 else 0) * r.uniform(.5, 1.5), 0, 1), x) for q, x in enumerate(xs)]; lo = [W_(np.clip(t1 + (r.uniform(.03, .10) if q % 2 else -.02), 0, 1), x) for q, x in enumerate(xs)]
            cel.fill(np.vstack([spl(up, n=3), lo[::-1]]), cols['hi'])
        n = len(l.Q)
        for q, P in enumerate((l.A, l.B)):
            i0 = int(n * (inner + r.uniform(-.12, .15))); seg = P[max(1, i0):]
            if len(seg) > 3: cel.line(seg, wi * r.uniform(.8, 1.15), line, taper=(.5, .12), press=.3, seed=l.id * 7 + q + seed)
        if l.w > .16:
            for q in range(int(r.integers(1, 3))):
                c = r.uniform(-.55, .55); t0 = r.uniform(.05, .3); t1 = t0 + r.uniform(.3, .55); i0, i1 = int(t0 * n), min(n, int(t1 * n))
                P = (l.A[i0:i1] * (1 - c) + l.B[i0:i1] * (1 + c)) / 2
                if len(P) > 3: cel.line(P, wf * r.uniform(.8, 1.2), line, taper=(.45, .5), press=.3, seed=l.id * 13 + q + seed)
    if cols.get('cast') is not None: cel.ctx.pop_group_to_source(); cel.ctx.paint()
    return cel


# ------------------------------------------------------------------ face parts
def eye(cel, H, s, C, E, L, w, seed=0):
    """s = -1 screen-left eye (character's right), +1 the other.  C = character eye design, E = expression"""
    wo, wi, wf = w; ex, ey = C['eye']
    if E.get('dots'):                                      # comedy stare: round white eye, pin dot
        if H.facing(s * ex, ey) < -.02: return
        t_ = np.linspace(0, 2 * math.pi, 40)[:-1]; el = lambda rx, ry, dx=0., dy=0.: H.P(np.c_[s * (ex + dx) + np.cos(t_) * rx, ey + dy + np.sin(t_) * ry])
        P_ = el(.125, .135); cel.fill(P_, L['white']); cel.line(np.vstack([P_, P_[:1]]), wi * 1.35, L['line'], taper=(.02, .02), press=.25, seed=seed + s); cel.fill(el(.026, .030, -.010, -.005), L['line'])
        return
    lid = E.get('lid', 1.); lid = lid[(s + 1) // 2] if np.ndim(lid) else lid
    sq = E.get('squint', 0.); tl = E.get('tilt', 0.); wide = E.get('eyew', 1.)
    U = np.array(C['upper'], float); D = np.array(C['lower'], float)
    base = np.interp(U[:, 0], D[:, 0], D[:, 1])
    U[:, 1] = base + (U[:, 1] - base) * lid
    mid = np.interp(D[:, 0], [U[0, 0], U[-1, 0]], [U[0, 1], U[-1, 1]])
    D[:, 1] = D[:, 1] + (mid - D[:, 1]) * sq
    for a in (U, D): a[:, 1] += tl * (a[:, 0] + .02); a[:, 0] *= wide
    def P(L_, dz=0.): L_ = np.asarray(L_, float); return H.P(np.c_[s * (ex + L_[:, 0]), ey + L_[:, 1]], dz)
    Us = spl(U, n=8); Ds = spl(D, n=8)
    vis = H.facing(s * ex, ey)
    if vis < -.02: return
    line = L['line']; lash = C['lash']
    closed = lid < .16
    if not closed:
        op = np.vstack([Us, Ds[::-1]]); OP = P(op)
        cel.fill(OP, L['white'])
        cel.clip(OP)
        # shadow of the lid on the white
        th = .035 * lid + .012
        cel.fill(np.vstack([P(Us), P(Us - [0, th])[::-1]]), L['white2'])
        # iris
        gx, gy = E.get('gaze', (0, 0)); k = E.get('iris', 1.); rx, ry = C['iris'][0] * k, C['iris'][1] * k
        ix, iy = s * gx + C['irisc'][0], C['irisc'][1] + gy
        th_ = np.linspace(0, 2 * math.pi, 40)[:-1]
        ir = lambda a, b, dx=0., dy=0.: np.c_[ix + dx + np.cos(th_) * rx * a, iy + dy + np.sin(th_) * ry * b]
        ic = C['ic']; tnt = L.get('eyemul')
        cel.fill(P(ir(1.0, 1.0)), ic['ring'])
        cel.fill(P(ir(.86, .88)), ic['base'])
        cel.clip(P(ir(.86, .88)))
        # lower light crescent, upper dark cap
        cel.fill(P(ir(.80, .58, 0, -ry * .46)), ic['lo'])
        cel.fill(P(np.array([(ix - rx * 1.2, iy + ry * 1.2), (ix + rx * 1.2, iy + ry * 1.2), (ix + rx * 1.2, iy + ry * .12), (ix + rx * .4, iy + ry * .30), (ix - rx * .3, iy + ry * .22), (ix - rx * 1.2, iy + ry * .34)])), ic['top'])
        cel.unclip()
        pk = E.get('pupil', 1.)
        if pk > 0: cel.fill(P(ir(.36 * pk, .42 * pk)), ic['pup'])
        if ic.get('spoke') and H.s > 120:
            for a in np.linspace(.3, 5.9, 9): cel.line(P(np.c_[[ix + math.cos(a) * rx * .5, ix + math.cos(a) * rx * .78], [iy + math.sin(a) * ry * .5, iy + math.sin(a) * ry * .78]]), wf * .7, ic['ring'], alpha=.55, seed=seed)
        hl = E.get('spark', 1.)
        if hl > 0:
            kx = -1 if L['key'][0] < 0 else 1
            cel.fill(P(ir(.24 * hl, .22 * hl, kx * rx * .36, ry * .36)), '#ffffff', .98)
            if L.get('lit'): cel.fill(P(ir(.62, .16, kx * rx * -.06, -ry * .66)), L['lit'], .55)
            cel.fill(P(ir(.13 * hl, .12 * hl, -kx * rx * .38, -ry * .46)), '#ffffff', .7)
        if E.get('tear'):
            cel.fill(P(np.vstack([Ds[4:-4], (Ds[4:-4] + [0, .030])[::-1]])), '#ffffff', .85)
            cel.fill(P(ir(.4, .22, -rx * .2, -ry * .62)), '#ffffff', .8)
        cel.unclip()
    # upper lash line: wedge, thick to the outside, individual curved lashes
    n = len(Us); u = np.linspace(0, 1, n)
    tk = np.interp(u, np.linspace(0, 1, len(lash['tk'])), lash['tk']) * (1.0 if not closed else .8)
    T = np.gradient(Us, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]
    top = Us + N * tk[:, None]; bot = Us - N * (tk * .12)[:, None]
    tip = Us[-1] + np.array(lash['wing']) + [0, tl * .04]
    tm = (top[-1] + tip) / 2 + [.004, .004]
    wedge = np.vstack([top, [tm, tip], [(Us[-1] + tip) / 2 + [0, -.004]], bot[::-1]])
    cel.fill(P(wedge), lash['col'])
    at = lambda t: np.array([np.interp(t, u, top[:, 0]), np.interp(t, u, top[:, 1])])
    flat = np.clip(lid, .35, 1.1)
    for t, dx, dy, th in lash['up']:
        r0 = at(t) - [.006, .010]; dy *= flat; q = np.array([r0, r0 + [dx * .34, dy * .22], r0 + [dx * .70, dy * .58], r0 + [dx, dy]])
        cel.line(P(spl(q, n=7)), th * H.s, lash['col'], taper=(0, 0), press=0, tremor=0, wfun=lambda v: (1 - v) ** .75 + .08)
    if lash.get('tint') and not closed:
        i0, i1 = int(n * .22), int(n * .78); cel.line(P(Us[i0:i1] + N[i0:i1] * .007), wi * .8, lash['tint'], taper=(.5, .5), seed=seed, alpha=.8)
    # lower lid
    if not closed:
        m = len(Ds); cel.line(P(Ds[int(m * lash['lowr'][0]):int(m * lash['lowr'][1])]), wi * lash.get('loww', 1.), line, taper=(.5, .35), seed=seed + 1)
        for a, dx, dy in lash['low']:
            q = Ds[int(a * (m - 1))]; cel.line(P(np.array([q + [0, .004], q + [dx * .5, dy * .6], q + [dx, dy]])), wf * 1.15, line, taper=(.02, .8), seed=seed + int(a * 100))
        cel.line(P(np.array([Us[1] + [-.002, -.006], Us[0] + [-.006, -.004], Us[0] + [.012, -.014]])), wi * .75, line, taper=(.3, .6), seed=seed + 9)
    if E.get('bag', C.get('bag')):
        k = E.get('bag', C.get('bag')); cel.line(P(spl(np.array([(-.085, -.118), (-.01, -.137), (.075, -.124)]), n=8)), wf * .95, L['line'], taper=(.5, .5), seed=seed + 12, alpha=.55 * k)
    # lid crease
    cr = C.get('crease')
    if cr is not None:
        c = np.array(cr, float); b = np.interp(c[:, 0], U[:, 0], U[:, 1]); c = np.c_[c[:, 0] * wide, b + c[:, 1]]
        cel.line(P(spl(c, n=8)), wf * 1.1, line, taper=(.4, .45), seed=seed + 4, alpha=.85)


def brow(cel, H, s, C, E, L, w, seed=0, col=None):
    ex, ey = C['eye']; B = np.array(C['brow'], float); dy, ang, pinch = E.get('brow', (0, 0, 0))
    if np.ndim(dy): dy = dy[(s + 1) // 2]
    if np.ndim(ang): ang = ang[(s + 1) // 2]
    B[:, 1] += dy + ang * (B[:, 0]); B[:, 0] -= pinch * np.clip(.12 - B[:, 0], 0, .3) / .3
    if H.facing(s * ex, ey + .15) < -.02: return
    c = spl(B[:, :2], n=8); th = np.interp(np.linspace(0, 1, len(c)), np.linspace(0, 1, len(B)), B[:, 2])
    T = np.gradient(c, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]
    P = np.vstack([c + N * th[:, None] * .5, (c - N * th[:, None] * .5)[::-1]])
    Q = H.P(np.c_[s * (ex + P[:, 0]), ey + P[:, 1]])
    cel.fill(Q, col or C['browc'])
    if E.get('frown'):
        for k in range(E['frown']):
            q = np.array([(.035 + k * .03, .10 + k * .01), (.028 + k * .03, .16), (.045 + k * .035, .215 + k * .01)])
            cel.line(H.P(np.c_[s * q[:, 0], q[:, 1]]), w[2] * 1.2, L['line'], taper=(.4, .5), seed=seed + k)


def nose(cel, H, C, L, w, seed=0):
    wo, wi, wf = w; n3 = H.nose_profile(); ya = abs(H.yaw); s = 1 if H.yaw >= 0 else -1; line = L['line']
    if ya > .17:
        b = spl(H.P3(n3[1:6]), n=8); m = len(b)
        if ya < .95: cel.line(b[int(m * .42):], wi * 1.05, line, taper=(.65, .1), press=.2, seed=seed)
        else: pass                                     # profile: contour draws it
        under = np.array([n3[5], n3[6] * [1, 1, 1] + [-s * .035, .012, 0]]); cel.line(spl(H.P3(np.array([n3[4] + [0, -.012, -.01], n3[5], under[1]])), n=6), wi, line, taper=(.1, .5), seed=seed + 1)
        z = H.z(.07, -.43); cel.line(H.P3(np.array([(-s * .085, -.395, z + .03), (-s * .105, -.428, z + .02), (-s * .085, -.456, z + .02)])), wf * 1.1, line, taper=(.4, .4), seed=seed + 2)
    else:
        for q in (-1, 1): cel.line(H.P(np.array([(q * .075, -.428), (q * .052, -.452), (q * .022, -.446)]), .06), wi * .9, line, taper=(.5, .4), seed=seed + q + 3)
        k = 1 if L['key'][0] < 0 else -1
        cel.line(H.P(np.array([(k * .012, -.30), (k * .026, -.36), (k * .030, -.395)]), .10), wf * 1.1, line, taper=(.6, .3), seed=seed)


def planes(cel, H, C, E, L, F):
    """small cel shadow shapes that model the face: eye socket, side/under of the nose, under the lip"""
    k = 1 if L['key'][0] < 0 else -1; cel.clip(F)
    if abs(H.yaw) < .9:
        z = lambda x, y: float(H.z(x, y))
        sg = k if abs(H.yaw) < .17 else (-1 if H.yaw > 0 else 1)
        if abs(H.yaw) < .17:
            T = [(0, -.365, z(0, -.36) + .12), (k * .012, -.410, z(0, -.41) + .13), (k * .062, -.444, z(.06, -.44) + .02), (k * .030, -.470, z(.03, -.47) + .01), (-k * .03, -.452, z(.03, -.45) + .05)]
        else:
            T = [(0, -.405, z(0, -.405) + .155), (sg * .05, -.436, z(.05, -.43) + .07), (sg * .085, -.458, z(.09, -.45) + .02), (sg * .03, -.474, z(.03, -.47) + .02), (0, -.462, z(0, -.46) + .06)]
        cel.fill(H.P3(np.array(T)), L['skin2'])
    ex, ey = C['eye']
    for s in (-1, 1):
        if H.facing(s * ex, ey) < .15 or E.get('nosock') or E.get('dots'): continue
        q = np.array([(-.175, .075), (-.10, .098), (-.02, .104), (-.10, .062), (-.150, .010), (-.172, -.035), (-.195, .02)])
        cel.fill(H.P(np.c_[s * (ex + q[:, 0]), ey + q[:, 1] + (E.get('brow', (0, 0, 0))[0] if np.ndim(E.get('brow', (0, 0, 0))[0]) == 0 else 0) * .5]), L['skin2'])
    cel.unclip()


def mouth(cel, H, M, L, w, seed=0):
    """M: dict(kind='line', pts=[...]) | dict(kind='open', up=[...], lo=[...], teeth=.., tongue=..)"""
    wo, wi, wf = w; line = L['line']
    if M['kind'] == 'line':
        c = spl(M['pts'], n=8); cel.line(H.P(c, .012), wi * 1.15 * M.get('w', 1.), line, taper=(.28, .28), press=.35, seed=seed)
        a, b = np.array(M['pts'][0]), np.array(M['pts'][-1])
        for q, sg in ((a, -1), (b, 1)):
            e = M.get('tick', 0)
            if e: cel.line(H.P(np.array([q + [-sg * .004, 0], q + [sg * .012, -e * .4], q + [sg * .016, -e * 1.4]]), .01), wf, line, taper=(.1, .7), seed=seed + 5)
    else:
        U, D = spl(M['up'], n=8), spl(M['lo'], n=8); op = np.vstack([U, D[::-1]]); OP = H.P(op, .01)
        cel.fill(OP, M.get('dark', '#4a1c26')); cel.clip(OP)
        if M.get('tongue'): cel.fill(H.P(spl(M['tongue'], True, n=8), .0), M.get('tcol', '#b85a58'))
        if M.get('teeth'):
            t = np.array(M['teeth']); cel.fill(H.P(np.vstack([U + [0, .03], spl(t, n=8)[::-1]]), .0), M.get('tw', L['white']))
            cel.line(H.P(spl(t, n=8), .0), wf * .9, '#8c6a70', taper=(.2, .2), seed=seed)
            for x, y0, y1 in M.get('gaps', ()): cel.line(H.P(np.array([(x, y0), (x * 1.01, (y0 + y1) / 2), (x, y1)]), 0), wf * .7, '#b79a9c', seed=seed)
        if M.get('loteeth'):
            t = np.array(M['loteeth']); cel.fill(H.P(np.vstack([spl(t, n=8), (D - [0, .03])[::-1]]), .0), L['white2']); cel.line(H.P(spl(t, n=8), .0), wf * .9, '#8c6a70', seed=seed)
        cel.unclip()
        cel.line(H.P(U, .012), wi * 1.2, line, taper=(.15, .15), press=.3, seed=seed + 2); cel.line(H.P(D, .012), wi * 1.0, line, taper=(.2, .2), press=.3, seed=seed + 3)
    if M.get('crease') is not None: cel.line(H.P(spl(M['crease'], n=8), .012), wf * 1.2, line, taper=(.5, .5), seed=seed + 8)
    if M.get('lip') is not None:
        c = spl(M['lip'], n=8); cel.line(H.P(c, .012), wf * 1.25, L['lip'] if M.get('lipc') is None else M['lipc'], taper=(.45, .45), seed=seed + 7)


def face_line(cel, H, F, w, line, seed=0):
    """face contour with a pen: heavy at chin and jaw corners, light on the cheeks, lifted at the temples"""
    P = resample(np.vstack([F, F[:1]]), 2.0); s = H.s
    c = H.P3(np.array([(0, -.99, .62)]))[0]; i = int(np.argmin(np.hypot(*(P - c).T))); P = np.vstack([P[i:-1], P[:i + 1]]); ov = max(3, int(.05 * s / 2))
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]; tot = d[-1]
    jl = [H.P3(np.array([(q * .56, -.56, .05)]))[0] for q in (-1, 1)]
    n = len(P); half = n // 2
    for k, seg in enumerate((np.vstack([P[-ov:-1], P[:half]]), np.vstack([P[half:], P[1:ov]])[::-1])):
        L = np.r_[0, np.cumsum(np.hypot(*np.diff(seg, axis=0).T))]
        def wf(t, L=L, seg=seg):
            q = np.interp(t, np.linspace(0, 1, len(seg)), L) / s
            v = .95 + .22 * np.exp(-((q - .05) / .16) ** 2) - .30 * np.exp(-((q - .38) / .16) ** 2) + .16 * np.exp(-((q - .70) / .12) ** 2) - .34 * np.exp(-((q - 1.05) / .22) ** 2)
            return v
        cel.line(seg, w[0], line, taper=(0, .04), press=.18, seed=seed + k, tremor=.15, wfun=wf)


def face_shadow(cel, H, L, F, amt=1.0, poly=None):
    """hard cel plane shadow on the side away from the key"""
    sgn = 1 if L['key'][0] < 0 else -1
    p = poly or [(.30, .80), (.41, .44), (.455, .27), (.41, .16), (.475, .02), (.50, -.10), (.455, -.20), (.36, -.32), (.385, -.44), (.30, -.62), (.215, -.79), (.10, -.93), (.04, -1.06), (.95, -1.15), (.98, .85)]
    q = np.array(p, float); q[:, 0] *= sgn
    q[:, 0] += 0 if abs(H.yaw) < .05 else np.clip(H.yaw * sgn, -.5, .5) * .10
    Q = H.P(q[:-2]); xf = H.cx + sgn * 4 * H.s; cel.shade(np.vstack([Q, [(xf, H.cy + 3 * H.s), (xf, H.cy - 3 * H.s)]]), L['skin2'], within=F)


def lit_plane(cel, H, L, F):
    """backlight: the face is in soft shade, the sun catches one hard cheek plane, the side of the nose and the rim of the jaw"""
    sg = -1 if L['key'][0] < 0 else 1; k = np.clip(H.yaw * -sg, -.6, .6); q = np.array([(.50, .62), (.455, .40), (.505, .24), (.470, .12), (.405, .02), (.365, -.10), (.300, -.215), (.345, -.33), (.425, -.40), (.385, -.52), (.300, -.64), (.335, -.75), (.235, -.86), (.180, -.97)], float)
    q[:, 0] = (q[:, 0] - .10 * k) * sg; Q = H.P(q); xf = H.cx + sg * 4 * H.s; cel.clip(F); cel.fill(np.vstack([Q, [(xf, H.cy + 3 * H.s), (xf, H.cy - 3 * H.s)]]), L['lit'])
    if abs(H.yaw) < 1.0:
        z = lambda x, y: float(H.z(x, y)); n = np.array([(sg * .004, -.06, z(0, -.06) + .03), (sg * .004, -.26, z(0, -.26) + .085), (sg * .002, -.385, z(0, -.39) + H.k['nose'][4] * .93), (sg * .030, -.405, z(.03, -.40) + .08), (sg * .052, -.425, z(.05, -.42) + .03), (sg * .045, -.30, z(.045, -.3) + .02), (sg * .040, -.12, z(.04, -.12) + .005)]); cel.fill(H.P3(n), L['lit'])
    cel.unclip()


def ear(cel, H, s, L, w, seed=0):
    e = H.ear(s)
    if e['vis'] <= .02: return
    shape(cel, e['outer'], L['skin'], w[0] * .72, L['line'], seed)
    cel.shade(e['inner'], L['skin2'], within=e['outer']); cel.line(e['inner'], w[2] * 1.1, L['line'], taper=(.3, .5), seed=seed + 1)
    cel.line(e['tragus'], w[2] * 1.1, L['line'], taper=(.3, .4), seed=seed + 2)


def neck_lines(H, nw, length=.95):
    """left and right silhouette of the neck column for any view"""
    ys = [-.30, -.70, -.88, -1.08, -1 - length * .62, -1 - length]; rx = [nw + .06, nw + .015, nw, nw - .005, nw + .05, nw + .24]; rz = [nw + .06, nw + .02, nw, nw, nw + .02, nw + .10]; cz = [-.22, -.18, -.13, -.11, -.10, -.10]
    t = np.linspace(0, 2 * math.pi, 48, endpoint=False); A, B = [], []
    c, sn = math.cos(H.roll), math.sin(H.roll)
    for y, a, b, z in zip(ys, rx, rz, cz):
        q = H.P3(np.c_[np.sin(t) * a, np.full(48, y), z + np.cos(t) * b]); u = q[:, 0] * c + q[:, 1] * sn; A.append(q[u.argmin()]); B.append(q[u.argmax()])
    return np.array(A), np.array(B)


def neck(cel, H, L, w, nw=.40, length=.95, seed=0):
    a, b = neck_lines(H, nw, length); A, B = spl(a, n=8), spl(b, n=8); P = np.vstack([A, B[::-1]]); s = H.s
    cel.fill(P, L['skin']); sg = 1 if L['key'][0] < 0 else -1
    d = np.array([sg * .07, .27]) * s
    J = np.vstack([H.jaw(-1), H.jaw(1)[::-1][1:]]) + d
    cel.clip(P); cel.fill(np.vstack([J, H.P3(np.array([(1.2, .3, 0), (-1.2, .3, 0)]))]), L['skin2'])
    sd = A if sg < 0 else B; k = np.linspace(0, 1, len(sd))[:, None]
    cel.fill(np.vstack([sd, (sd + np.array([-sg, 0]) * s * (.26 * (1 - k) + .07))[::-1]]), L['skin2'])
    cel.unclip()
    for i, c in enumerate((A, B)): cel.line(c, w[0] * .8, L['line'], taper=(.3, .1), seed=seed + i)
    if abs(H.yaw) < 1.1:
        c = H.P3(np.array([(-sg * .24, -.98, .06), (-sg * .13, -1.28, .14), (-sg * .05, -1.55, .2)])); cel.line(spl(c, n=8), w[2] * 1.2, L['line'], taper=(.5, .5), seed=seed + 5, alpha=.7)
    return P


# ------------------------------------------------------------------ torso and uniform
class Torso:
    """body frame: origin at the pit of the neck, x right, y up, z forward, unit = head unit (s px)"""
    def __init__(self, ox, oy, s, yaw=0., pitch=0., roll=0., persp=.06):
        self.o = np.array([ox, oy], float); self.s = s; self.persp = persp; a, b = yaw, pitch; self.yaw = yaw
        Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]); Rx = np.array([[1, 0, 0], [0, math.cos(b), math.sin(b)], [0, -math.sin(b), math.cos(b)]])
        c, n = math.cos(-roll), math.sin(-roll); self.R = np.array([[c, -n, 0], [n, c, 0], [0, 0, 1]]) @ Rx @ Ry

    def P3(self, Q):
        q = np.asarray(Q, float) @ self.R.T; f = 1 / (1 - np.clip(q[..., 2] * self.persp, -.5, .5))
        return np.stack([self.o[0] + q[..., 0] * f * self.s, self.o[1] - q[..., 1] * f * self.s], -1)

    def z(self, x, y, B):
        a = np.interp(-np.asarray(y, float), B['ty'], B['ta']); f = np.interp(-np.asarray(y, float), B['ty'], B['tf'])
        return f * np.sqrt(np.clip(1 - (np.abs(x) / a) ** 2.2, 0, 1))

    def S(self, xy, B, dz=.0):
        xy = np.asarray(xy, float); return self.P3(np.c_[xy, self.z(xy[:, 0], xy[:, 1], B) + dz])

    def ring(self, cx, cy, cz, rx, rz, t0, t1, tilt=0., n=40):
        """arc of a horizontal ring, angle 0 = front, + = right; tilt raises the back"""
        t = np.linspace(t0, t1, n); return np.c_[cx + np.sin(t) * rx, cy - np.cos(t) * tilt, cz + np.cos(t) * rz]


def fold(cel, c, th, L, wi, seed=0, dim=1., line=True):
    """one cloth fold: pen line, lit sliver on the light side, dark sliver on the other"""
    c = np.asarray(c, float); n = len(c); t = np.linspace(0, 1, n); c = c * .48 + (c[0] + (c[-1] - c[0]) * t[:, None]) * .52; T = np.gradient(c, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-6; N = np.c_[-T[:, 1], T[:, 0]]
    k = np.array(L['key']); N = N * (1 if N[n // 2] @ k > 0 else -1); w = (np.clip(t / .13, 0, 1) ** .9 * (1 - t) ** .9 * th * 1.15)[:, None]
    cel.fill(np.vstack([c, (c + N * w)[::-1]]), L['cloth_hi'] if dim >= 1 else mixh(L['cloth'], L['cloth_hi'], dim)); cel.fill(np.vstack([c, (c - N * w * .6)[::-1]]), L['cloth_dk'])
    if line: cel.line(c, wi, L['cloth_ln'], taper=(.05, .65), press=.3, seed=seed)


def mirror(P): P = np.array(P, float); P[:, 0] *= -1; return P


def swirl(cel, c, r, L, w, seed=0, sq=1.0):
    """the Jujutsu High swirl button"""
    t = np.linspace(0, 2 * math.pi, 30)[:-1]; E = np.c_[c[0] + np.cos(t) * r * sq, c[1] + np.sin(t) * r]
    cel.fill(E, L['metal']); cel.shade(E + [r * .35, r * .4], '#8a5a26', within=E, alpha=.7); cel.line(np.vstack([E, E[:1]]), max(1.1, r * .16), L['cloth_ln'], taper=(.05, .05), press=.3, seed=seed)
    u = np.linspace(0, 1, 40); a = -1.2 + u * 9.2; q = np.c_[c[0] + np.cos(a) * r * .74 * (1 - u) * sq, c[1] + np.sin(a) * r * .74 * (1 - u)]
    cel.line(q, max(.9, r * .15), '#5a3516', taper=(.1, .4), seed=seed + 1)
    cel.line(np.array([[c[0] - r * .5 * sq, c[1] - r * .35], [c[0] - r * .25 * sq, c[1] - r * .6]]), max(.9, r * .18), '#fff3cf', taper=(.3, .3), seed=seed + 2)


def uniform(cel, T, B, L, w, seed=0, part='back', sleeves=(1, 1)):
    """B = body design dict.  part 'back' = inside of the collar; 'front' = torso, sleeves, collar front"""
    wo, wi, wf = w; ln = L['cloth_ln']; c = B['collar']; sg = 1 if L['key'][0] < 0 else -1
    top = T.ring(0, c['ty'], c['tz'], c['trx'], c['trz'], -math.pi, math.pi, c['tilt'], 80)
    if part == 'back':
        P = T.P3(top); cel.fill(P, L['cloth_dk']); cel.line(P, wo * .9, ln, taper=(0, 0), seed=seed, closed=False)
        return
    # sleeves + torso flats, outer contour first (covered half by the fills)
    R = [np.array(B['body_r'], float), np.array(B['sleeve_r'], float)]; Lh = [mirror(B.get('body_l', B['body_r'])), mirror(B.get('sleeve_l', B['sleeve_r']))]
    def C3(P): return np.c_[P[:, :2], P[:, 2]] if P.shape[1] == 3 else np.c_[P, np.zeros(len(P))]
    def cv(P, corners=()): return spl(T.P3(C3(P)), n=8, corners=corners)
    br, bl = cv(R[0], B.get('bc', ())), cv(Lh[0], B.get('bc', ())); body = np.vstack([br, bl[::-1]])
    sr = spl(T.P3(C3(R[1])), True, corners=B.get('sc', ()), n=8); sl = spl(T.P3(C3(Lh[1])), True, corners=B.get('sc', ()), n=8)
    far = [(sr, 1), (sl, -1)] if T.yaw > 0 else [(sl, -1), (sr, 1)]
    def sleeve(S, q):
        if not sleeves[(q + 1) // 2]: return
        outline(cel, S, wo * 1.7, ln, seed + 5 + q, tremor=.2); cel.fill(S, L['cloth']); cel.clip(S)
        if q == sg: cel.fill(T.P3(C3(np.array([(x * q, y, z) for x, y, z in B['sleeve_sh']]))), L['cloth_dk'])
        for h in B['sleeve_hi']: cel.fill(T.P3(np.array([(x * q, y, z) for x, y, z in h])), L['cloth_hi'] if q != sg else mixh(L['cloth'], L['cloth_hi'], .45))
        for h in B['sleeve_ln']: fold(cel, spl(T.P3(np.array([(x * q, y, z) for x, y, z in h])), n=8), .085 * T.s, L, wi, seed + int(abs(h[0][1]) * 50), 1. if q != sg else .4)
        cel.unclip()
    sleeve(*far[0])
    outline(cel, body, wo * 1.6, ln, seed + 9, tremor=.2); cel.fill(body, L['cloth'])
    cel.clip(body)
    cel.fill(T.S(np.array([(x * sg, y) for x, y in B['body_sh']]), B), L['cloth_dk'])
    for p in B['dark']: cel.fill(T.S(p, B), L['cloth_dk'])
    for p in B.get('mid', ()): cel.fill(T.S(np.array(p, float), B), mixh(L['cloth'], L['cloth_hi'], .42))
    for p in B['hi']:
        pp = np.array(p, float); side = np.sign(pp[:, 0].mean()) or 1
        cel.fill(T.S(pp, B), L['cloth_hi'] if side != sg else mixh(L['cloth'], L['cloth_hi'], .55))
    for p in B['lines']:
        c_ = spl(T.S(np.array(p, float), B), n=8); sd_ = np.sign(np.mean(np.array(p)[:, 0])) or 1; fold(cel, c_, .10 * T.s * B.get('fw', 1.), L, wi, seed + int(abs(p[0][1]) * 90), 1. if sd_ != sg else .5)
    for p in B.get('seams', ()): cel.line(spl(T.S(np.array(p, float), B), n=8), wi, ln, taper=(.08, .3), press=.3, seed=seed + 77)
    cel.unclip()
    sleeve(*far[1])
    for p in B['over']: cel.line(spl(T.S(np.array(p, float), B), n=8), wi * 1.1, ln, taper=(.1, .5), press=.3, seed=seed + 41)
    # collar front: the half of the frustum that faces the camera
    a0 = T.yaw; t0, t1 = -math.pi / 2 - a0 - .10, math.pi / 2 - a0 + .10
    tp = T.P3(T.ring(0, c['ty'], c['tz'], c['trx'], c['trz'], t0, t1, c['tilt'])); bt = T.P3(T.ring(0, c['by'], c['bz'], c['brx'], c['brz'], t0, t1, c['btilt']))
    bow = np.sin(np.linspace(0, math.pi, 12))[:, None]
    e1 = np.linspace(tp[-1], bt[-1], 12) + bow * np.array([c['bow'] * T.s, 0]); e0 = np.linspace(bt[0], tp[0], 12) - bow * np.array([c['bow'] * T.s, 0])
    G = np.vstack([tp, e1, bt[::-1], e0]); outline(cel, G, wo * 1.6, ln, seed + 21, tremor=.2); cel.fill(G, L['cloth'])
    cel.clip(G)
    def K(a, v, dr=0.):                                  # collar coords: a angle, v 0 top .. 1 bottom
        a = np.asarray(a, float); v = np.asarray(v, float)
        rx = c['trx'] + (c['brx'] - c['trx']) * v + dr; rz = c['trz'] + (c['brz'] - c['trz']) * v + dr
        y = (c['ty'] - np.cos(a) * c['tilt']) * (1 - v) + (c['by'] - np.cos(a) * c['btilt']) * v
        return T.P3(np.stack([np.sin(a) * rx, y, (c['tz'] + (c['bz'] - c['tz']) * v) + np.cos(a) * rz], -1))
    h = -a0 + sg * np.linspace(.35, 1.9, 10)
    cel.fill(np.vstack([K(h[:1].repeat(6) + sg * np.array([0, .12, .02, .2, .1, .3]), np.linspace(-.05, 1.05, 6)), K(h[-1:].repeat(2), [1.05, -.05])]), L['cloth_dk'])
    for q in c['hi']: cel.fill(K([x for x, y in q], [y for x, y in q]), L['cloth_hi'])
    for q in c['ln']: cel.line(spl(K([x for x, y in q], [y for x, y in q]), n=8), wi, ln, taper=(.1, .5), press=.3, seed=seed + 51)
    cel.unclip()
    cel.line(tp, wo * .9, ln, taper=(.02, .02), seed=seed + 60); cel.line(e0, wo * .9, ln, taper=(.1, .1), seed=seed + 61); cel.line(e1, wo * .9, ln, taper=(.1, .1), seed=seed + 62); cel.line(bt, wi * 1.2, ln, taper=(.2, .2), seed=seed + 63)
    if B.get('button'):
        a, v = B['button']; vis = math.cos(a + a0)
        if vis > .1: swirl(cel, K(np.array([a]), np.array([v]), .02)[0], B.get('br', .075) * T.s, L, w, seed, max(.35, vis))
    if B.get('cbutton'):
        q = T.S(np.array([B['cbutton']]), B, .03)[0]; swirl(cel, q, B.get('br', .075) * T.s, L, w, seed, max(.35, math.cos(T.yaw)))


def mixh(a, b, t): return mix(rgb(a), rgb(b), t)


def torso_under(H, yaw=None, pitch=0., roll=None, drop=1.12, s=None):
    """torso frame hanging under a posed head"""
    o = H.P3(np.array([(0, -drop, -.06)]))[0]
    return Torso(o[0], o[1], s or H.s, H.yaw * .6 if yaw is None else yaw, pitch, H.roll * .3 if roll is None else roll)


def grade(layer, L, box, rimw=None, para_amt=.9):
    """airbrush gradient + rim pass on a finished character layer.  box = (x0,y0,x1,y1) of the figure"""
    x0, y0, x1, y1 = box; kx, ky = L['key']; n = math.hypot(kx, ky); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; r = math.hypot(x1 - x0, y1 - y0) * .42
    layer = para(layer, (cx + kx / n * r, cy + ky / n * r), (cx - kx / n * r, cy - ky / n * r), L['para'], 'mul', para_amt)
    if L['rim']:
        c, dx, dy, a = L['rim']; k = (rimw or 1.)
        layer = rim(layer, dx * k, dy * k, c, a, .5)
    return layer


def sleeve2(cel, path, width, L, w, seed=0, bumps=.10, hem=True, folds=7, hi=None, bunch=(), close=False, rnd=(0, 0), phase=None, openend=False, bold=1.):
    """a sleeve (or trouser leg) along a 2-D path, wrist/ankle first: pulled folds with a notch in the contour, a lit sliver above and a dark one under each fold,
    one jagged shadow side.  bunch = [u ...] places with two or three crowded folds (elbow, hem)"""
    wo, wi, wf = w; r = np.random.default_rng(seed); ln = L['cloth_ln']; hi = hi or L['cloth_hi']; dk = L['cloth_dk']
    C = resample(spl(np.asarray(path, float), n=14), 3.0); n = len(C); T = np.gradient(C, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-6; N = np.c_[-T[:, 1], T[:, 0]]
    tw = np.array(width, float); t = np.linspace(0, 1, n); W_ = np.interp(t, tw[:, 0], tw[:, 1]); asp = np.hypot(*np.diff(C, axis=0).T).sum() / W_.mean()
    kx, ky = L['key']; m = n // 2; lit = 1 if N[m] @ np.array([kx, ky]) > 0 else -1
    # folds: (side, u0, length in widths, reach, weight)
    F = []
    for i, u in enumerate(np.linspace(.10, .92, folds) + r.normal(0, .035, folds)): F.append((1 if i % 2 else -1, float(u), r.uniform(.35, 1.1) * r.choice([1, 1, -1]), r.uniform(-.1, .75), r.uniform(.6, 1.1)))
    for u in bunch:
        for j in range(3): F.append((1 if j % 2 else -1, u + (j - 1) * .06 + r.normal(0, .006), r.uniform(.18, .34), r.uniform(.55, .85), r.uniform(.7, 1.0)))
    def edge(sd):
        k = np.zeros(n)
        for s_, u0, ln_, rc, wt in F:
            if s_ != sd: continue
            e = .16 / asp * 2; x = (t - u0) / e; k += wt * bumps * (.55 * np.exp(-((x + 1.1) / .9) ** 2) - .75 * np.exp(-(x / .35) ** 2) + .25 * np.exp(-((x - 1.0) / .9) ** 2))
        return C + N * (sd * W_ * (1 + k))[:, None]
    A, B = edge(1), edge(-1)
    def cap(i, sg, k):                                     # rounded end
        u = np.linspace(0, math.pi, 12)[1:-1]; return np.array([C[i] + N[i] * W_[i] * math.cos(q) + sg * T[i] * W_[i] * k * math.sin(q) for q in u])
    P = np.vstack([A] + ([cap(-1, 1, rnd[1])] if rnd[1] else []) + [B[::-1]] + ([cap(0, -1, rnd[0])[::-1]] if rnd[0] else []))
    def rim_():
        if not openend: outline(cel, P, wo * 1.7, ln, seed, tremor=.2); return
        m_ = int(n * .90)                                  # the shoulder end melts into the body: side lines lift before they get there
        for q, (E_, k_) in enumerate(((A, m_), (B, m_))): cel.line(E_[:k_], wo * 1.7, ln, taper=(.02, .16), press=.3, seed=seed + q)
        cel.line(np.vstack([A[:1], C[:1] - T[0] * (W_[0] * (.9 if rnd[0] else .05)), B[:1]]), wo * 1.7, ln, taper=(.02, .02), seed=seed + 3)
    if phase == 'o': rim_(); return P
    X = lambda tt, v: np.array([np.interp(tt, t, C[:, i]) + np.interp(tt, t, N[:, i]) * v * np.interp(tt, t, W_) for i in (0, 1)])
    if phase != 'f': rim_()
    cel.fill(P, L['cloth']); cel.clip(P)
    pts = [X(0, -lit * .30)]; u = .0; z = 0
    while u < 1: u += r.uniform(.06, .13); z ^= 1; pts.append(X(min(u, 1), -lit * (r.uniform(.08, .30) if z else r.uniform(.42, .66))))
    cel.fill(np.vstack([pts, [X(1, -lit * 1.5), X(0, -lit * 1.5)]]), dk)
    for q, (s_, u0, ln_, rc, wt) in enumerate(F):
        du = ln_ / asp * 1.6; v1 = s_ * (1 - 1.15 * (1 - rc) - .25); c = spl([X(u0, s_ * 1.06), X(u0 + du * .30, s_ * .62 + v1 * .05), X(u0 + du * .72, v1 * .70 + s_ * .12), X(u0 + du, v1)], n=7); k = len(c); tt = np.linspace(0, 1, k)
        th = (np.clip(tt / .16, 0, 1) ** .9 * (1 - tt) ** .85 * wt * .24 * bold)[:, None] * np.interp(u0, t, W_); d = np.array([np.interp(u0, t, T[:, 0]), np.interp(u0, t, T[:, 1])])
        up = -d * (1 if True else 0)
        if s_ == lit: cel.fill(np.vstack([c, (c + up * th * 1.25)[::-1]]), hi)
        elif wt > .85: cel.fill(np.vstack([c, (c + up * th * .7)[::-1]]), mixh(L['cloth'], hi, .30))
        cel.fill(np.vstack([c, (c - up * th * .65)[::-1]]), dk)
        cel.line(c, wi * (.75 + .4 * wt), ln, taper=(.04, .7), press=.3, seed=seed + q)
    # long light on the lit edge, interrupted by the folds
    us = sorted([u0 for s_, u0, *_ in F if s_ == lit] + [0.02, 1.0])
    for u0, u1 in zip(us[:-1], us[1:]):
        if u1 - u0 < .05 or r.random() < .25: continue
        a, b = u0 + (u1 - u0) * .12, u0 + (u1 - u0) * r.uniform(.62, .9); v0 = r.uniform(.80, .90)
        cel.fill(np.vstack([spl([X(a, lit * v0), X((a + b) / 2, lit * (v0 + .03)), X(b, lit * v0)], n=5), spl([X(b - (b - a) * .25, lit * (v0 - .16)), X(a + (b - a) * .35, lit * (v0 - .22))], n=3)]), hi)
    cel.unclip()
    if hem:
        e = spl([A[0], C[0] - T[0] * W_[0] * .16, B[0]], n=8); cel.fill(np.vstack([e, [C[0] + T[0] * W_[0] * .1]]), '#08070c'); cel.line(e, wo * .9, ln, taper=(.05, .05), seed=seed + 60); cel.line(spl([X(.05, 1.), X(.035, 0.), X(.05, -1.)], n=8), wi * .8, ln, taper=(.1, .1), seed=seed + 61)
    return P


def arm(cel, sh, el, wr, wd, L, w, seed=0, order='fu', bow=0., folds=5, bumps=.07, hem=True, hi=None, openend=False, bold=1., bunch=True):
    """a bent sleeved arm (shoulder, elbow, wrist in px) as one silhouette: round shoulder, round elbow.  wd = half widths (wrist, forearm, elbow, upper, shoulder).
    order 'fu' draws the upper arm over the forearm, 'uf' the forearm over the upper arm"""
    sh, el, wr = [np.asarray(v, float) for v in (sh, el, wr)]; d1, d2 = el - sh, wr - el; n1 = np.array([-d1[1], d1[0]]) / (np.hypot(*d1) + 1e-6); n2 = np.array([-d2[1], d2[0]]) / (np.hypot(*d2) + 1e-6)
    up = dict(path=[el, (el + sh) / 2 + n1 * bow, sh], width=[(0, wd[2]), (.5, wd[3]), (.85, wd[4]), (1, wd[4] * .94)], seed=seed + 1, hem=False, rnd=(.95, 1.0), bunch=(.10,) if bunch else (), folds=folds, bumps=bumps, openend=openend, bold=bold); fo = dict(path=[wr, (wr + el) / 2 + n2 * bow * .6, el], width=[(0, wd[0]), (.5, wd[1]), (1, wd[2])], seed=seed + 2, hem=hem, rnd=(0, .95), bunch=(.86, .12) if bunch else (.86,), folds=folds, bumps=bumps, bold=bold)
    Q = (fo, up) if order == 'fu' else (up, fo)
    for q in Q: sleeve2(cel, L=L, w=w, hi=hi, phase='o', **q)
    for q in Q: sleeve2(cel, L=L, w=w, hi=hi, phase='f', **q)
    # elbow: two crossing pull folds
    b = (el + sh * .25 + wr * .25) / 1.5
    for k, e in ((-.5, .55), (.45, .4)): c = spl(np.array([el + (el - b) * .25 + n1 * k * wd[2] * .5, el + (b - el) * .5, el + (b - el) * 1.6 + n2 * k * wd[2] * .6]), n=8); cel.line(c, w[1], L['cloth_ln'], taper=(.1, .6), press=.3, seed=seed + 5 + int(k * 10))
