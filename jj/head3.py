"""3-D head construction for drawn faces.  You author the face in FRONT coordinates (x right, y UP, eye line y=0, chin y=-1,
skull top y=+1.0) and the head projects it to any yaw/pitch/roll with correct foreshortening; contours come from a head mesh.
    H = Head(props).pose(cx, cy, s, yaw, pitch, roll)     s = px per unit (chin-to-eye distance = 1 unit)
    H.P([(x,y),...], dz=0)   face-surface points  -> screen px        H.P3([(x,y,z),...]) free 3-D points -> screen px
    H.face()  -> closed face contour (cheek, jaw, chin, nose if it breaks the profile)     H.skull() -> whole bald head contour
    H.jaw(side)  -> jaw line ear->chin      H.ear(side) -> dict of ear curves     H.facing(x,y) -> cos between surface normal and view
    yaw > 0 turns the head to the viewer's right (shows more of the character's right cheek on the left side of the picture)
"""
import numpy as np, math, cv2
from .ink import spl, resample
from .kit import contours

DEF = dict(a=[(-1.0, .13), (-.96, .22), (-.86, .37), (-.72, .50), (-.56, .60), (-.36, .665), (-.12, .70), (.12, .715), (.35, .72), (.6, .69), (.8, .55), (.93, .34), (1.0, .05)],
           f=[(-1.0, .64), (-.92, .72), (-.8, .74), (-.64, .765), (-.46, .77), (-.28, .755), (-.06, .735), (.06, .745), (.2, .80), (.4, .80), (.6, .73), (.8, .55), (.93, .32), (1.0, .05)],
           b=[(-1.0, .05), (-.9, .10), (-.72, .22), (-.5, .48), (-.2, .74), (.1, .88), (.4, .92), (.65, .84), (.85, .58), (.95, .32), (1.0, .05)],
           p=2.25, nose=(.10, -.06, -.41, -.47, .165), ear=(.715, .10, -.40, -.10), cheek=.030, socket=.035)


def _it(tab, y):
    t = np.asarray(tab, float)
    # monotone-ish smooth interpolation
    return np.interp(y, t[:, 0], t[:, 1])


class Head:
    def __init__(self, **kw):
        self.k = dict(DEF); self.k.update(kw)
        n = 161
        yy = np.linspace(-1, 1, n)
        ker = np.hanning(13); ker /= ker.sum()
        def sm(tab):
            v = _it(tab, yy); pad = np.r_[np.full(6, v[0]), v, np.full(6, v[-1])]
            return np.convolve(pad, ker, 'valid')
        self.yy = yy; self.A = sm(self.k['a']); self.F = sm(self.k['f']); self.B = sm(self.k['b'])
        self.pose(0, 0, 1)

    def pose(self, cx, cy, s, yaw=0., pitch=0., roll=0., persp=.10):
        """pitch > 0 looks up, roll > 0 tilts clockwise on screen"""
        self.cx, self.cy, self.s, self.yaw, self.pitch, self.roll, self.persp = cx, cy, s, yaw, pitch, roll, persp
        a, b = yaw, pitch
        Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
        Rx = np.array([[1, 0, 0], [0, math.cos(b), math.sin(b)], [0, -math.sin(b), math.cos(b)]])
        c, sn = math.cos(-roll), math.sin(-roll)
        Rz = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
        self.R = Rz @ Rx @ Ry
        return self

    # ---- projection
    def P3(self, Q):
        q = np.asarray(Q, float) @ self.R.T
        f = 1 / (1 - np.clip(q[..., 2] * self.persp, -.5, .5))
        return np.stack([self.cx + q[..., 0] * f * self.s, self.cy - q[..., 1] * f * self.s], -1)

    def depth(self, Q):
        return (np.asarray(Q, float) @ self.R.T)[..., 2]

    def z(self, x, y):
        """face-surface depth at front coordinates"""
        x = np.asarray(x, float); y = np.asarray(y, float)
        a = np.interp(y, self.yy, self.A); f = np.interp(y, self.yy, self.F); p = self.k['p']
        u = np.clip(np.abs(x) / a, 0, .999)
        z = f * (1 - u ** p) ** (1 / p)
        z = z + self.k['cheek'] * np.exp(-((y + .22) / .13) ** 2) * np.exp(-((np.abs(x) - .44) / .2) ** 2) * 1.4
        z = z - self.k['socket'] * np.exp(-((y - .02) / .10) ** 2) * np.exp(-((np.abs(x) - .40) / .24) ** 2) * 1.3
        return z + self._mid(x, y)

    def _mid(self, x, y):
        """lips, chin and brow relief near the centre line (shapes the profile)"""
        g = lambda y0, sy, sx, k: k * np.exp(-((y - y0) / sy) ** 2) * np.exp(-(x / sx) ** 2)
        return g(-.575, .050, .17, .040) + g(-.705, .042, .15, .036) - g(-.640, .020, .16, .012) - g(-.795, .045, .2, .030) + g(-.915, .07, .2, .030) + g(.21, .08, .42, .020) - g(.075, .07, .16, .035)

    def P(self, xy, dz=0.):
        xy = np.asarray(xy, float)
        return self.P3(np.concatenate([xy, (self.z(xy[..., 0], xy[..., 1]) + dz)[..., None]], -1))

    def facing(self, x, y):
        e = 1e-2
        n = np.array([-(self.z(x + e, y) - self.z(x - e, y)) / (2 * e), -(self.z(x, y + e) - self.z(x, y - e)) / (2 * e), 1.])
        n /= np.linalg.norm(n); return float((self.R @ n)[2])

    # ---- mesh silhouettes
    def _mesh(self, y0, y1, half, ny=90, nf=90):
        y = np.linspace(y0, y1, ny)
        zj = np.interp(y, [-1.0, -.995, -.96, -.90, -.82, -.72, -.60, -.46, -.30, -.2], [.64, .64, .58, .48, .32, .16, .04, -.04, -.10, -.10])
        A, F, B = (np.interp(y, self.yy, t) for t in (self.A, self.F, self.B)); p = self.k['p']
        ph = np.linspace(-math.pi, math.pi, nf) if not half else np.linspace(-math.pi * .5 * half, math.pi * .5 * half, nf)
        PH = np.repeat(ph[None], ny, 0)
        if half:                                   # cut the shell along the jaw line so the jaw is an edge, not a tube
            c = np.clip(zj / np.maximum(np.where(zj >= 0, F, B), 1e-3), -1, .98)
            t = np.where(c >= 0, np.arctan2((1 - np.abs(c) ** p) ** (1 / p), np.abs(c)), np.pi / 2 + .0 * c + np.arcsin(np.clip(-c, 0, 1)))
            PH = np.linspace(-1, 1, nf)[None] * np.minimum(math.pi * .5 * half, t)[:, None]
        sx, cz = np.sin(PH), np.cos(PH)
        r = (np.abs(sx) ** p + np.abs(cz) ** p) ** (-1 / p)
        X = A[:, None] * sx * r; Z = np.where(cz >= 0, F[:, None], B[:, None]) * cz * r
        Y = np.repeat(y[:, None], nf, 1)
        fr = np.clip(cz, 0, 1)
        Z = Z + fr * (self._mid(X, Y) + self.k['cheek'] * np.exp(-((Y + .22) / .13) ** 2) * np.exp(-((np.abs(X) - .44) / .2) ** 2) * 1.4
                      - self.k['socket'] * np.exp(-((Y - .02) / .10) ** 2) * np.exp(-((np.abs(X) - .40) / .24) ** 2) * 1.3)
        return np.stack([X, Y, Z], -1)

    def _sil(self, meshes, extra=()):
        k = max(.5, min(1., 900. / (self.s * 3.2)))
        Ps = [self.P3(m) for m in meshes] + [np.asarray(e)[None] for e in extra]
        allp = np.vstack([p.reshape(-1, 2) for p in Ps]); lo = allp.min(0) - 8; hi = allp.max(0) + 8
        w, h = (np.ceil((hi - lo) * k)).astype(int) + 2
        m = np.zeros((h, w), np.uint8)
        for p in Ps:
            q = (p - lo) * k
            if p.shape[0] == 1: cv2.fillPoly(m, [np.round(q[0] * 16).astype(np.int32)], 255, cv2.LINE_8, 4); continue
            quad = np.stack([q[:-1, :-1], q[:-1, 1:], q[1:, 1:], q[1:, :-1]], 2).reshape(-1, 4, 2)
            for c in np.round(quad * 16).astype(np.int32): cv2.fillConvexPoly(m, c, 255, cv2.LINE_8, 4)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        c = max(contours(m / 255.), key=len)
        return resample(c / k + lo, 3.0)

    def face(self, top=.75):
        """closed contour of the face mask (front shell from chin up to y=top), nose included when it breaks the silhouette"""
        return self._sil([self._mesh(-1, top, 1.12)], [self.P3(self.nose_profile())] if abs(self.yaw) > .75 else [])

    def skull(self):
        return self._sil([self._mesh(-.42, 1, 0)])

    def nose_profile(self):
        w, yt, ytip, yb, d = self.k['nose']; z = lambda y: float(self.z(0, y))
        return np.array([(0, .10, z(.10) - .01), (0, yt, z(yt) + d * .18), (0, (yt + ytip) / 2, z((yt + ytip) / 2) + d * .55), (0, ytip + .03, z(ytip) + d * .93),
                         (0, ytip, z(ytip) + d), (0, ytip - .03, z(ytip) + d * .82), (0, yb, z(yb) + d * .30), (0, yb - .05, z(yb - .05))])

    def jaw(self, side=1):
        s = side; y = np.array([-.30, -.46, -.60, -.72, -.82, -.90, -.96, -.995])
        a = np.interp(y, self.yy, self.A)
        z = np.array([-.10, -.04, .04, .16, .32, .48, .58, .64]); x = a * np.array([1.0, 1.0, 1.0, .985, .95, .85, .62, .0]) * s
        return self.P3(np.stack([x, y, z], -1))

    def ear(self, side=1):
        ex, t, b, z = self.k['ear']; s = side
        out = [(.0, b), (.02, b - .03), (.07, b + .03), (.115, b + .16), (.15, t - .14), (.155, t - .04), (.12, t + .03), (.06, t + .02), (.015, t - .05), (.0, t - .12)]
        inn = [(.02, t - .07), (.06, t - .02), (.10, t - .03), (.115, t - .10), (.09, b + .20), (.05, b + .08)]
        in2 = [(.02, t - .20), (.065, t - .16), (.07, t - .26), (.035, b + .15), (.05, b + .10)]
        # ear leans back and flares from the head
        def Q(L): return self.P3(np.array([(s * (ex + u * .95), v, z - u * 1.55) for u, v in L]))
        return dict(outer=spl(Q(out), n=6), inner=spl(Q(inn), n=6), tragus=spl(Q(in2), n=6), vis=float((self.R @ np.array([s * .84, 0, -.35 + .62]))[2]))

    def under(self):
        """underside of the jaw (fill with shadow before the face when the head looks up): closed screen polygon"""
        a, b = self.jaw(-1), self.jaw(1)
        back = self.P3(np.array([(.42, -.60, -.18), (0, -.68, -.12), (-.42, -.60, -.18)]))
        return np.vstack([a, b[::-1][1:], back])

    def neck(self, w=.40, back=-.30, front=.28, length=.9, lean=0.):
        """left and right neck lines (screen) from under the jaw down"""
        L = []
        for s in (-1, 1):
            L.append(self.P3(np.array([(s * (w + .04), -.62, back * .2 + .02), (s * w, -.85, -.05), (s * (w - .01), -1.1, -.05), (s * (w + .05) + lean, -1 - length * .7, -.08), (s * (w + .22) + lean, -1 - length, -.10)])))
        return L

    # ---- hair: one clump in 3-D
    def clump(self, theta, phi, length, width, dirv, curl=0., lift=.02):
        """root on the skull at azimuth theta (0 front, +right, radians) and height phi (y of the root, -0.6..0.98).
        dirv = 3-D growth direction (x,y,z), width = root width in units, curl bends the tip (3-D vector or scalar toward -y).
        returns (root_a, tip, root_b, depth) in screen px for kit.spike"""
        y = phi; A = np.interp(y, self.yy, self.A); B = np.interp(y, self.yy, self.F if math.cos(theta) >= 0 else self.B)
        p = self.k['p']; sx, cz = math.sin(theta), math.cos(theta); r = (abs(sx) ** p + abs(cz) ** p) ** (-1 / p)
        root = np.array([A * sx * r, y, B * cz * r]); n = np.array([sx / max(A, .1), (y - .1) * .9, cz / max(B, .1)]); n /= np.linalg.norm(n)
        root = root + n * lift; d = np.asarray(dirv, float); d /= np.linalg.norm(d) + 1e-9
        tip = root + d * length + (np.asarray(curl, float) if np.ndim(curl) else np.array([0, -curl, 0]))
        # root edge perpendicular to growth on the skull
        e = np.cross(d, n); e = e / (np.linalg.norm(e) + 1e-9) if np.linalg.norm(e) > 1e-3 else np.array([1., 0, 0])
        q = self.P3(np.stack([root - e * width / 2, tip, root + e * width / 2]))
        return q[0], q[1], q[2], float(self.depth((root + tip) / 2))
