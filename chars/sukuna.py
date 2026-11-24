"""SUKUNA, king of curses: one 3-D-constructed drawing, two finishes (brush ink / fire cel).
    k = Suk(cx, cy, s, yaw, pitch, roll, expr='smile', seed=3, ...)     s = px from eye line to chin
    k.ink()  -> premultiplied RGBA (head + neck)      k.fire(light=...) -> same drawing as fire-lit cel
See the bottom of the file for bodies (robe, bare) and sukuna_eyes.py for the four ink eyes."""
import numpy as np, cv2, math
from jj import W, H as HH
from jj.paint import rgb, f32, fbm, smooth, mix, grid, lum
from jj.ink import Cel, spl, resample, noise1, xf
from jj.kit import contours, mask_of, rim, para, tint, svg, s1
from jj.head3 import Head
from jj.comp import over, screen, blur
from chars.sukuna_ink import *

SKIN = '#ecdcc3'; SKIN_S = '#b9a596'; SKIN_D = '#7d6a66'
HAIR = '#cb888c'; HAIR_D = '#93505a'; HAIR_L = '#ecc4b8'

ACCENT = dict(blood=dict(mark='#8e101b', scratch='#6d0c16', scratch2='#5a0a13', ear='#8f1520', ear2='#6d0c16', earfill='#6b1119', hair='#7a2833'),
              ice=dict(mark='#8fd0f4', scratch='#3f7fb0', scratch2='#2c5f8c', ear='#a9defa', ear2='#4f8fbd', earfill='#1c3c58', hair='#33506e'))

# ------------------------------------------------------------------ expressions
EXPR = {
    'prayer':   dict(eo=(0., 0.), e2=(0., 0.), brow=(-.010, -.035), mouth='flat', jaw=0., look=(0, 0), low=0.),
    'cold':     dict(eo=(.82, .82), e2=(.9, .9), brow=(-.012, -.02), mouth='flat', jaw=0., look=(0, 0), low=.1),
    'smile':    dict(eo=(.78, .72), e2=(.95, .9), brow=(-.030, .01), mouth='smile', jaw=.02, look=(0, 0), low=.35),
    'laugh':    dict(eo=(.62, .55), e2=(1., 1.), brow=(.004, .035), mouth='laugh', jaw=.17, look=(0, .25), low=.6),
    'contempt': dict(eo=(.52, .46), e2=(.6, .6), brow=(-.004, .0), mouth='smirk', jaw=0., look=(.25, -.7), low=.15),
    'strain':   dict(eo=(.74, .70), e2=(.9, .9), brow=(-.034, .02), mouth='grit', jaw=.03, look=(.3, 0), low=.30),
    'calm':     dict(eo=(.6, .6), e2=(.5, .5), brow=(-.005, -.03), mouth='flat', jaw=0., look=(0, 0), low=0.),
}

# hair: front row on the hairline (x, y, length, width, dir, bend) hand set, then scalp rows (theta, y, ...)
# kind of a lock: (x, y, length, width, dir, bend, asym, split)   split = (t of the notch, side) or 0
FRONT = [
    # widow's peak mass, leaning to picture left
    (-.010, 0.535, .44, .150, (-.16, .90, -.30), .16, .25, 0), (-.085, 0.565, .62, .185, (-.26, .90, -.30), .13, .30, (.52, 1)), (.075, 0.560, .50, .150, (.10, .93, -.28), .10, .20, 0),
    (-.165, 0.590, .40, .135, (-.40, .84, -.24), .20, .25, 0),
    # right rising mass
    (.150, 0.595, .70, .200, (.20, .93, -.28), -.10, -.30, (.48, -1)), (.235, 0.622, .46, .145, (.34, .88, -.22), -.18, -.2, 0), (.315, 0.635, .60, .185, (.48, .82, -.22), -.14, -.3, (.55, -1)),
    (.395, 0.622, .36, .130, (.66, .70, -.16), -.22, -.2, 0), (.455, 0.610, .54, .185, (.74, .64, -.16), -.14, -.3, (.5, 1)),
    # left mass, bigger and lower
    (-.255, 0.622, .66, .205, (-.40, .86, -.26), .14, .3, (.50, 1)), (-.335, 0.636, .40, .140, (-.56, .78, -.20), .22, .2, 0), (-.415, 0.618, .58, .190, (-.66, .70, -.20), .14, .3, (.46, -1)),
    (-.480, 0.605, .34, .130, (-.80, .56, -.14), .20, .2, 0),
    # temples
    (.535, 0.555, .46, .160, (.88, .44, -.12), -.16, -.3, 0), (-.540, 0.555, .50, .175, (-.90, .40, -.14), .14, .3, (.5, 1)), (.590, 0.475, .30, .120, (.95, .26, -.12), -.2, -.2, 0), (-.600, 0.455, .28, .115, (-.97, .20, -.14), .2, .2, 0),
    (.640, 0.385, .36, .140, (.97, .22, -.16), -.14, -.3, (.5, -1)), (-.645, 0.375, .32, .130, (-.97, .12, -.18), .16, .3, 0), (.665, 0.285, .22, .100, (.98, .02, -.2), -.2, -.2, 0), (-.668, 0.285, .24, .105, (-.98, -.04, -.2), .2, .2, 0),
]
ROWS = [  # theta, y, L, w, dir, bend, asym, split
    (-0.97, 0.67, 0.51, 0.18, (-0.28, 0.42, -0.88), 0.07, 0.28, 0), (0.94, 0.67, 0.46, 0.18, (0.31, 0.39, -0.88), -0.14, -0.28, 0), (-1.17, 0.65, 0.55, 0.19, (-0.24, 0.39, -0.88), 0.16, 0.28, (0.5, 1)), (1.12, 0.6, 0.5, 0.18, (0.23, 0.51, -0.88), -0.14, -0.28, (0.5, -1)), (-1.37, 0.73, 0.57, 0.18, (-0.25, 0.52, -0.88), 0.07, 0.28, 0), (1.34, 0.73, 0.58, 0.2, (0.26, 0.54, -0.88), -0.09, -0.28, 0), (-1.52, 0.62, 0.53, 0.2, (-0.36, 0.48, -0.88), 0.13, 0.28, (0.5, 1)), (1.5, 0.62, 0.55, 0.18, (0.27, 0.54, -0.88), -0.11, -0.28, (0.5, -1)), (-1.63, 0.46, 0.52, 0.17, (-0.23, 0.39, -0.88), 0.13, 0.28, 0), (1.59, 0.48, 0.53, 0.18, (0.33, 0.37, -0.88), -0.13, -0.28, 0), (-1.72, 0.83, 0.65, 0.2, (-0.23, 0.58, -0.88), 0.1, 0.28, (0.5, 1)), (1.7, 0.78, 0.56, 0.19, (0.34, 0.44, -0.88), -0.14, -0.28, (0.5, -1)), (-1.74, 0.34, 0.39, 0.17, (-0.24, 0.23, -0.88), 0.11, 0.28, 0), (1.79, 0.31, 0.42, 0.17, (0.38, 0.34, -0.88), -0.15, -0.28, 0), (-1.56, 0.18, 0.29, 0.14, (-0.39, 0.14, -0.88), 0.12, 0.28, 0), (1.59, 0.19, 0.31, 0.15, (0.27, 0.11, -0.88), -0.13, -0.28, 0), (-1.97, 0.67, 0.64, 0.22, (-0.31, 0.51, -0.88), 0.15, 0.28, (0.5, 1)), (1.94, 0.68, 0.6, 0.19, (0.36, 0.46, -0.88), -0.07, -0.28, (0.5, -1)), (-1.98, 0.49, 0.47, 0.19, (-0.26, 0.32, -0.88), 0.07, 0.28, 0), (2.04, 0.53, 0.48, 0.19, (0.29, 0.29, -0.88), -0.09, -0.28, 0), (-2.09, 0.33, 0.42, 0.18, (-0.29, 0.19, -0.88), 0.07, 0.28, 0), (2.02, 0.32, 0.39, 0.16, (0.24, 0.23, -0.88), -0.16, -0.28, 0), (-2.11, 0.07, 0.35, 0.18, (-0.32, 0.08, -0.88), 0.15, 0.28, 0), (2.08, 0.08, 0.36, 0.16, (0.35, 0.11, -0.88), -0.14, -0.28, 0), (-2.24, 0.89, 0.65, 0.22, (-0.28, 0.52, -0.88), 0.14, 0.28, 0), (2.2, 0.85, 0.54, 0.19, (0.35, 0.52, -0.88), -0.09, -0.28, 0), (-1.8, 0.06, 0.27, 0.13, (-0.27, 0.17, -0.88), 0.16, 0.28, 0), (1.72, 0.04, 0.23, 0.12, (0.29, 0.08, -0.88), -0.15, -0.28, 0), (-1.92, -0.06, 0.22, 0.13, (-0.37, 0.05, -0.88), 0.15, 0.28, 0), (1.9, -0.1, 0.26, 0.13, (0.36, 0.11, -0.88), -0.14, -0.28, 0),
    (1.22, .52, .46, .17, (.52, .36, -.78), -.10, -.3, (.5, -1)), (-1.22, .52, .48, .17, (-.52, .34, -.78), .10, .3, (.5, 1)), (1.38, .40, .44, .16, (.44, .22, -.88), -.10, -.3, 0), (-1.38, .40, .42, .16, (-.44, .24, -.88), .10, .3, 0),
    (1.30, .28, .36, .14, (.46, .10, -.88), -.10, -.3, 0), (-1.30, .28, .34, .14, (-.46, .08, -.88), .10, .3, 0), (1.42, .18, .28, .12, (.40, -.02, -.9), -.10, -.2, 0), (-1.42, .18, .26, .12, (-.40, -.04, -.9), .10, .2, 0),
    (-.10, .78, .58, .19, (-.14, .92, -.36), .12, .3, (.55, 1)), (.14, .80, .48, .16, (.06, .94, -.34), .08, .2, 0), (.36, .80, .72, .21, (.26, .92, -.32), -.10, -.3, (.5, -1)), (.62, .79, .50, .17, (.46, .84, -.28), -.16, -.2, 0),
    (-.36, .80, .70, .21, (-.30, .90, -.34), .10, .3, (.5, 1)), (-.62, .79, .46, .16, (-.48, .82, -.28), .18, .2, 0), (-.86, .78, .66, .21, (-.58, .76, -.30), .12, .3, (.48, -1)), (.88, .78, .64, .20, (.62, .74, -.28), -.12, -.3, (.5, 1)),
    (1.12, .76, .44, .16, (.76, .60, -.24), -.18, -.2, 0), (-1.12, .76, .44, .16, (-.74, .62, -.24), .18, .2, 0), (-1.32, .70, .60, .20, (-.82, .50, -.26), .12, .3, (.5, 1)), (1.34, .70, .56, .19, (.84, .48, -.26), -.12, -.3, (.46, -1)),
    (-1.46, .58, .42, .16, (-.92, .34, -.22), .16, .2, 0), (1.46, .58, .44, .16, (.92, .36, -.22), -.16, -.2, 0), (-1.42, .46, .52, .17, (-.95, .22, -.24), .10, .3, 0), (1.42, .46, .46, .16, (.96, .18, -.24), -.12, -.3, 0),
    (-1.45, .34, .34, .14, (-.97, .04, -.26), .14, .2, 0), (1.45, .34, .38, .15, (.97, .06, -.26), -.14, -.2, 0), (1.5, .22, .26, .12, (.92, -.12, -.36), -.14, -.2, 0), (-1.5, .22, .28, .12, (-.92, -.10, -.36), .14, .2, 0),
    (-.4, .92, .66, .20, (-.18, .90, -.40), .10, .3, (.5, 1)), (.5, .92, .62, .20, (.16, .90, -.42), -.10, -.3, 0), (-1.2, .90, .60, .20, (-.40, .82, -.42), .10, .2, (.55, -1)), (1.3, .90, .66, .21, (.44, .80, -.42), -.10, -.3, (.5, 1)),
    (.0, .97, .56, .19, (.02, .88, -.48), .06, .2, 0), (2.2, .95, .58, .21, (.20, .80, -.58), -.08, -.2, 0), (-2.3, .95, .62, .22, (-.24, .78, -.58), .08, .2, (.5, 1)), (3.1, .97, .60, .22, (.0, .76, -.66), .05, .2, 0),
    (1.75, .82, .60, .20, (.66, .58, -.48), -.12, -.3, (.5, -1)), (-1.8, .82, .56, .20, (-.68, .58, -.46), .12, .3, 0), (2.25, .82, .56, .21, (.46, .58, -.66), -.08, -.2, 0), (-2.25, .82, .60, .21, (-.48, .56, -.68), .08, .2, (.5, 1)), (2.8, .84, .58, .22, (.2, .58, -.8), -.06, -.2, 0), (-2.8, .84, .54, .22, (-.2, .56, -.8), .06, .2, 0),
    (1.8, .62, .52, .19, (.80, .34, -.50), -.12, -.3, 0), (-1.8, .62, .56, .20, (-.82, .32, -.48), .12, .3, (.5, -1)), (2.2, .60, .50, .20, (.60, .32, -.72), -.08, -.2, 0), (-2.2, .60, .50, .20, (-.60, .30, -.74), .08, .2, 0), (2.7, .60, .54, .22, (.28, .32, -.90), -.06, -.2, 0), (-2.7, .60, .56, .22, (-.28, .30, -.90), .06, .2, 0), (3.14, .62, .52, .22, (.0, .34, -.95), .04, .2, 0),
    (1.75, .40, .40, .17, (.86, .10, -.50), -.12, -.2, 0), (-1.75, .40, .42, .17, (-.86, .08, -.50), .12, .2, 0), (2.15, .38, .42, .19, (.64, .06, -.76), -.08, -.2, 0), (-2.15, .38, .42, .19, (-.64, .04, -.76), .08, .2, 0), (2.65, .38, .44, .21, (.30, .04, -.95), -.05, -.2, 0), (-2.65, .38, .46, .21, (-.30, .02, -.95), .05, .2, 0), (3.14, .38, .44, .21, (0, .06, -1), .03, .2, 0),
    (1.85, .16, .30, .15, (.78, -.16, -.58), -.1, -.2, 0), (-1.85, .16, .30, .15, (-.78, -.18, -.58), .1, .2, 0), (2.3, .14, .34, .18, (.52, -.2, -.82), -.06, -.2, 0), (-2.3, .14, .34, .18, (-.52, -.2, -.82), .06, .2, 0), (2.8, .14, .34, .2, (.2, -.22, -.95), 0, .1, 0), (-2.8, .14, .34, .2, (-.2, -.22, -.95), 0, .1, 0),
    (2.0, -.04, .24, .14, (.66, -.40, -.60), -.06, -.2, 0), (-2.0, -.04, .24, .14, (-.66, -.40, -.60), .06, .2, 0), (2.5, -.06, .26, .17, (.40, -.46, -.80), 0, .1, 0), (-2.5, -.06, .26, .17, (-.40, -.46, -.80), 0, .1, 0), (3.14, -.06, .28, .19, (0, -.46, -.90), 0, .1, 0),
    (2.2, -.22, .18, .13, (.46, -.76, -.46), 0, .1, 0), (-2.2, -.22, .18, .13, (-.46, -.76, -.46), 0, .1, 0), (2.7, -.24, .20, .15, (.24, -.82, -.5), 0, .1, 0), (-2.7, -.24, .20, .15, (-.24, -.82, -.5), 0, .1, 0), (3.14, -.24, .2, .16, (0, -.85, -.5), 0, .1, 0),
]


class SHead(Head):
    """Head with a jaw that opens (3-D, so contour and features follow) and a face projection that never folds over the far side"""
    jopen = 0.

    def P3(self, Q):
        Q = np.asarray(Q, float)
        if self.jopen:
            Q = Q.copy(); w = smooth(-.54, -.92, Q[..., 1]) * smooth(-.3, .35, Q[..., 2])
            Q[..., 1] -= self.jopen * w; Q[..., 2] -= self.jopen * .28 * w
        return Head.P3(self, Q)

    def limits(self):
        ys = np.linspace(-.99, .9, 48); V = self.R[2]; lo, hi = [], []
        for y in ys:
            a = np.interp(y, self.yy, self.A); x = np.linspace(-a * .985, a * .985, 141); e = 4e-3; z0 = self.z(x, y)
            nx = -(self.z(x + e, y) - self.z(x - e, y)) / (2 * e); ny = -(self.z(x, y + e) - self.z(x, y - e)) / (2 * e)
            c = (nx * V[0] + ny * V[1] + V[2]) / np.sqrt(nx * nx + ny * ny + 1); ok = c > .04
            j = int(np.argmax(c)); l = j; r = j
            while l > 0 and ok[l - 1]: l -= 1
            while r < len(x) - 1 and ok[r + 1]: r += 1
            lo.append(x[l] if l > 0 else -9.); hi.append(x[r] if r < len(x) - 1 else 9.)
        self._lim = (ys, np.array(lo), np.array(hi)); return self._lim

    def Pc(self, xy, dz=0., k=.10):
        xy = np.array(xy, float); ys, lo, hi = getattr(self, '_lim', None) or self.limits()
        l = np.interp(xy[..., 1], ys, lo); h = np.interp(xy[..., 1], ys, hi); x = xy[..., 0]
        x = np.where(x > h - k, h - k * np.exp(-np.clip((x - h + k) / k, 0, 20)), x); x = np.where(x < l + k, l + k * np.exp(-np.clip((l + k - x) / k, 0, 20)), x)
        xy[..., 0] = x; return self.P(xy, dz)


class Suk:
    def __init__(self, cx, cy, s, yaw=0., pitch=0., roll=0., expr='cold', seed=3, marks=1., dissolve=0., wind=0., hair=1., persp=.10, lw=None, t=0., noir=0., neck_len=1.75, accent='blood', **over_):
        self.s, self.yaw, self.pitch, self.roll, self.seed = s, yaw, pitch, roll, seed
        e = dict(EXPR[expr] if isinstance(expr, str) else expr); e.update(over_); self.e = e
        self.marks, self.dissolve, self.wind, self.hs, self.t, self.noir, self.neck_len = marks, dissolve, wind, hair, t, noir, neck_len
        self.H = SHead().pose(cx, cy, s, yaw, pitch, roll, persp); self.H.jopen = e['jaw']; self.H.limits()
        self.lw = lw or float(np.clip((s / 260.) ** .6, .30, 2.2))      # line weight unit
        self.ac = ACCENT[accent] if isinstance(accent, str) else dict(ACCENT['blood'], **accent)
        self.bg = float(max(1., (s / 420.) ** .5))
        self.up = self._dir((0, 1., 0)); self.rt = self._dir((1., 0, 0)); self.petals = []

    def _dir(self, v):
        d = self.H.R @ np.array(v, float); return np.array([d[0], -d[1]]) * self.s

    # --------------------------------------------------------- projection
    def P(self, xy, dz=0.):
        q = self.H.Pc(xy, dz)
        return self.mould(q) if q.ndim == 2 else q

    def C(self, xy, n=10, corners=(), closed=False, dz=0.):
        """spline in face coordinates, then project"""
        return self.P(spl(np.asarray(xy, float), closed, corners, n), dz)

    def vis(self, x, y): return self.H.facing(float(np.clip(x, -.66, .66)), y)

    def head_y(self, p):
        """approximate face y of screen points near the front of the face"""
        o = self.H.P3(np.array([0, 0, .55])); d = Head.P3(self.H, np.array([0, -1., .55])) - o
        return -((np.asarray(p) - o) @ d) / (d @ d)

    @classmethod
    def of(cls, H, **k):
        """from a posed jj.head3.Head (for jj.figure head_fn and others)"""
        return cls(H.cx, H.cy, H.s, H.yaw, H.pitch, H.roll, persp=getattr(H, 'persp', .10), **k)

    def box(self, m=1.0):
        """screen ROI of the head with its hair (and neck)"""
        q = self.H.P3(np.array([(x, y, z) for x in (-1.25, 1.25) for y in (-2.6, 1.75) for z in (-1.5, .95)])); lo, hi = q.min(0) - 12, q.max(0) + 12
        return lo[0], lo[1], hi[0], hi[1]

    # --------------------------------------------------------- eyes
    def eye_lines(self, side):
        e = self.e; o = e['eo'][0 if side < 0 else 1]; low = e['low']
        j = lambda k: float(np.random.default_rng(self.seed * 13 + k + (side > 0) * 50).normal(0, .003))
        inn = np.array([.140 + j(1), -.068 + j(2)]); out = np.array([.545 + j(3), .034 + j(4)])
        clo = np.array([inn, (.205, -.090), (.290, -.102), (.380, -.092), (.470, -.058), out + (-.004, -.028)])
        upo = np.array([inn, (.190, -.044), (.265, -.014), (.350, .012), (.455, .034), out])
        lo = np.array([inn, (.205, -.094), (.298, -.110), (.392, -.100), (.475, -.062), out + (-.010, -.008)])
        lo[1:5, 1] += low * .034
        up = clo + (upo - clo) * min(o, 1.3)
        if o < .05: lo = clo
        s = np.array([side, 1.])
        return dict(up=up * s, lo=lo * s, inn=inn * s, out=out * s, o=o, clo=clo * s)

    def eye2_lines(self, side):
        o = self.e['e2'][0 if side < 0 else 1]; s = np.array([side, 1.])
        a = np.array([.268, -.246]); b = np.array([.520, -.160])
        mid = np.array([(.330, -.246), (.400, -.224), (.466, -.190)])
        up = np.vstack([a, mid + np.array([(0, .020), (.0, .036), (0, .028)]) * o, b]); lo = np.vstack([a, mid + np.array([(0, -.014), (0, -.022), (0, -.016)]) * max(o, .0), b])
        return dict(up=up * s, lo=lo * s, o=o, a=a * s, b=b * s, c=np.vstack([a, mid, b]) * s)

    # --------------------------------------------------------- face marks (front coordinates, closed polygons)
    def mark_polys(self):
        M = []
        for s in (-1, 1):
            f = np.array([s, 1.])
            fo = lambda L: spl((np.array(L) - (0, .44)) * (1.08, 1.00) + (0, .392), n=6)
            M.append(('fore', np.vstack([fo([(.150, .570), (.124, .510), (.156, .458), (.210, .408), (.165, .392), (.105, .402), (.056, .430)]),
                                         fo([(.056, .430), (.098, .422), (.128, .426), (.146, .434), (.114, .472), (.092, .520), (.150, .570)])]) * f))
            M.append(('fore', np.vstack([fo([(.096, .448), (.084, .400), (.054, .345), (.034, .296)]), fo([(.034, .296), (.026, .360), (.038, .410), (.056, .430)])]) * f))
            M.append(('nose', np.vstack([spl(np.array([(-.02, -.204), (.0, -.204), (.055, -.224), (.110, -.250), (.165, -.252), (.240, -.206)]), n=7), spl(np.array([(.240, -.206), (.176, -.284), (.112, -.290), (.052, -.256), (.0, -.232), (-.02, -.232)]), n=7)]) * f))
            for dx, dy, k in ((0, 0, 1.), (-.040, -.168, .86)):
                T = lambda L: spl(np.array([(.70 - (.70 - x) * k + dx, -.32 + (y + .32) * k + dy) for x, y in L]), n=6)
                M.append(('cheek', np.vstack([T([(.735, -.338), (.650, -.312), (.588, -.308), (.512, -.262)]), T([(.512, -.262), (.556, -.326), (.592, -.340)]), T([(.592, -.340), (.566, -.366), (.522, -.398)]), T([(.522, -.398), (.596, -.386), (.650, -.374), (.735, -.338)])]) * f))
        a = np.linspace(0, 2 * math.pi, 14, endpoint=False)
        M.append(('fore', np.c_[.0 + .025 * np.cos(a), .452 + .029 * np.sin(a)]))
        M.append(('chin', np.vstack([spl(np.array([(-.014, -.812), (-.007, -.89), (-.012, -.975)]), n=5), spl(np.array([(.010, -.975), (.006, -.89), (.016, -.812), (.0, -.795)]), n=5)])))
        return M

    def marks_screen(self):
        """visible marks in screen px -> list of (name, polygon); dissolve breaks them into drifting petals"""
        out = []; petals = []
        for i, (nm, p) in enumerate(self.mark_polys()):
            p = resample(p, .012); c = p.mean(0)
            if self.vis(c[0], c[1]) < .06: continue
            q = self.P(p, .004)
            if self.dissolve > 0:
                d = float(np.clip(self.dissolve * 1.5 - .5 * ((i * .37) % 1), 0, 1))
                if d > 0:
                    r = np.random.default_rng(self.seed * 7 + i); n = 5 + i % 3
                    bb = np.linspace(0, len(q), n + 1).astype(int); cc = q.mean(0)
                    for k in range(n):
                        if bb[k + 1] - bb[k] < 2: continue
                        g = np.vstack([q[bb[k]:bb[k + 1] + 1], cc[None]]); dd = float(np.clip(d * 1.6 - .6 * r.random(), 0, 1)); m = g.mean(0)
                        th = -math.pi / 2 + r.normal(.5, .6); dist = dd ** 1.6 * self.s * r.uniform(.3, .9); sc = 1 - .75 * dd
                        petals.append((xf(g, sc, sc * (1 - .5 * dd), rot=dd * r.normal(0, 2.4), pivot=m) + np.array([math.cos(th), math.sin(th)]) * dist, dd))
                    continue
            out.append((nm, q))
        return out, petals

    # --------------------------------------------------------- hair geometry
    def _root(self, th, y):
        h = self.H; A = np.interp(y, h.yy, h.A); B = np.interp(y, h.yy, h.F if math.cos(th) >= 0 else h.B); p = h.k['p']
        sx, cz = math.sin(th), math.cos(th); r = (abs(sx) ** p + abs(cz) ** p) ** (-1 / p)
        q = np.array([A * sx * r, y, B * cz * r]); n = np.array([sx / max(A, .1), (y - .1) * .9, cz / max(B, .1)]); return q, n / np.linalg.norm(n)

    def locks(self):
        """every hair clump -> list of dict(c = centre line px, w = half width px along it, depth, tone, kind), far to near"""
        out = []; r = np.random.default_rng(91); wd = self.wind
        def add(q, n, L, w, d, bend, asym, split, kind, k):
            d = np.asarray(d, float); d = d / np.linalg.norm(d); L = L * self.hs * .84; w = w * 1.32
            sw = wd * math.sin(self.t * 2.1 + k * 1.7) * .06 + wd * .08
            d2 = d + np.array([sw + wd * .25, -.02 * abs(wd), 0]); d2 /= np.linalg.norm(d2)
            root = q + n * .012 - d * L * .07; e = np.cross(d, n); e /= np.linalg.norm(e) + 1e-9
            tip = root + d * L * .55 + d2 * L * .54 - n * L * .02 + e * L * bend
            Q = np.stack([root, root + d * L * .22 + n * L * .045 - e * L * bend * .10, root + d * L * .52 + n * L * .06 + e * L * bend * .10, root + d * L * .42 + d2 * L * .42 + n * L * .03 + e * L * bend * .52, tip]); c = spl(self.H.P3(Q), n=7)
            wpx = max(w * self.s * .80, .5)
            out.append(dict(c=c, w=wpx, depth=float(self.H.depth((root * .6 + tip * .4))), kind=kind, k=k, face=float((self.H.R @ n)[2]), asym=asym, split=split, root=root))
        for k, (x, y, L, w, d, b, as_, sp) in enumerate(FRONT):
            xx = np.clip(x, -.66, .66); q = np.array([x, y, float(self.H.z(xx, y))]); e = 1e-2
            n = np.array([-(self.H.z(xx + e, y) - self.H.z(xx - e, y)) / (2 * e), -(self.H.z(xx, y + e) - self.H.z(xx, y - e)) / (2 * e), 1.]); n /= np.linalg.norm(n)
            if abs(x) > .585: q[2] *= .80; n = np.array([math.copysign(.93, x), .1, .3])
            add(q, n, L, w, d, b, as_, sp, 'front', k)
        for k, (th, y, L, w, d, b, as_, sp) in enumerate(ROWS):
            q, n = self._root(th, y); add(q, n, L, w, d, b, as_, sp, 'row', 40 + k)
        out.sort(key=lambda o: o['depth'])
        return out

    @staticmethod
    def lock_poly(c, w, asym=0., split=0):
        """flame shaped clump around its centre line: one flank full, the other lean, optional second tip"""
        n = len(c); t = np.linspace(0, 1, n); T = np.gradient(c, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]
        f = (.46 + .54 * np.clip(t / .30, 0, 1) ** .8) * np.clip(1 - t ** 1.55, 0, 1) ** .92
        hl = .5 * w * f * (1 + asym); hr = .5 * w * f * (1 - asym)
        A = c + N * hl[:, None]; Bk = c - N * hr[:, None]
        if split:
            tn, sd = split; i = int(n * tn); j = min(n - 3, i + max(2, int(n * .16))); P_, h = (A, hl) if sd > 0 else (Bk, hr)
            sg = 1. if sd > 0 else -1.; tip = c[j] + N[j] * sg * h[i] * 1.32; tip = tip + (tip - P_[i - 2]) * .35; inn = c[i] + N[i] * sg * h[i] * .42 + (c[i] - c[i - 1]) * 1.2
            tail = P_[j:] * .75 + c[j:] * .25
            P2 = np.vstack([P_[:i + 1], tip[None], inn[None], tail])
            if sd > 0: A = P2
            else: Bk = P2
        return np.vstack([A, Bk[::-1]]), N, hl, hr

    HL = np.array([(-.662, -.06), (-.626, .14), (-.632, .26), (-.590, .35), (-.555, .45), (-.470, .545), (-.395, .625), (-.270, .640), (-.140, .600), (-.015, .525), (.105, .595), (.255, .648), (.385, .628), (.465, .548), (.556, .45), (.588, .35), (.634, .26), (.628, .14), (.662, -.06)])

    def hairline(self):
        """-> (hairline polyline on screen left..right, closed polygon of everything above it)"""
        hp = self.P(spl(self.HL, n=6)); up = self.up; ext0, ext1 = [], []
        for q, ext in ((-1, ext0), (1, ext1)):
            if (self.H.R @ np.array([q * 1., 0, 0]))[2] > -.10 or abs(self.yaw) < .12:
                ext += [self.H.P3(np.array(v)) for v in ((q * .690, .09, .05), (q * .73, .30, -.2))]
        chain = np.vstack([np.array(ext0[::-1]).reshape(-1, 2), hp, np.array(ext1).reshape(-1, 2)])
        return hp, np.vstack([chain, chain[-1:] + up * 1.5, chain[:1] + up * 1.5])

    # ========================================================= INK
    def _lock_ink(self, sh, o, dark):
        """one inked clump: black offset shape, flat, wash, serrated black root, tip strokes"""
        lw = self.lw; r = np.random.default_rng(1000 + o['k']); c = o['c']; w = o['w']
        poly, N, hl, hr = self.lock_poly(c, w, o['asym'], o['split']); n = len(c); L = length(c)
        lo_, hi_ = poly.min(0), poly.max(0)
        if L < 8 or hi_[0] < sh.ox or lo_[0] > sh.ox + sh.w or hi_[1] < sh.oy or lo_[1] > sh.oy + sh.h: return
        ds = 1. if (N[n // 2] @ np.array([dark, .35])) > 0 else -1.          # which flank is in shadow
        fl = lambda u: c + N * (np.where(u >= 0, hl, hr) * u)[:, None] if np.ndim(u) else c + N * ((hl if u >= 0 else hr) * u)[:, None]
        g = float(np.clip(w * .034, 1.0, 4.2) * (.75 + .5 * lw))
        t = np.linspace(0, 1, n); gl = g * (.55 + .7 * (1 - t) ** .8) * np.clip((1 - t) * 8, 0, 1) ** .7
        # contour: the outline polygon stroked as a black twin, fatter on the shadow flank
        off = np.array([dark * .55, .45]) * g * .9
        tip = c[-1] + (c[-1] - c[-3]) / (np.hypot(*(c[-1] - c[-3])) + 1e-6) * g * 1.3
        ctx = sh.k.ctx
        sh.cut(poly)
        sh.k.fill(np.vstack([poly + off]), BLACK); sh.a.fill(poly + off, HAIR_D)
        ctx.save(); sh.k._path(poly); sh.k._src(BLACK, 1.); ctx.set_line_width(g * 1.25); ctx.set_line_join(cairo.LINE_JOIN_MITER); ctx.set_miter_limit(6); ctx.stroke(); ctx.restore()
        sh.cut(poly, sh.k)
        nd = bool(self.noir) and float((c[n // 2] - self.H.P3(np.array([0, .3, 0.]))) @ self.rt) * dark / self.s ** 2 > .42
        back_row = o['face'] < .2 or nd; base = self.ac['hair'] if nd else HAIR_D if o['face'] < -.05 else (HAIR if not back_row else '#b06a73')
        sh.a.fill(poly, base)
        # flat wash on the shadow flank, with a sawtooth edge
        e = fl(ds * .05) + N * (ds * np.where(ds > 0, hl, hr) * .35 * noise1(n, 4, r))[:, None]
        sh.wash(np.vstack([e, fl(ds * 1.)[::-1]]), '#a05560', .6, within=poly)
        mz = int(n * r.uniform(.34, .50)); zz = np.array([fl(u)[int(mz * (.72 + .5 * r.random()))] for u in np.linspace(1, -1, 7)]); sh.wash(np.vstack([fl(1.)[:int(mz * .7)], zz, fl(-1.)[:int(mz * .7)][::-1]]), '#7c3444', .55, within=poly)
        sh.k.clip(poly); sh.a.clip(poly)
        if r.random() < .34 and not back_row:
            i0, i1 = int(n * r.uniform(.34, .44)), int(n * r.uniform(.78, .92)); q = fl(-ds * .72); q2 = fl(-ds * .36)
            sh.a.fill(np.vstack([q[i0:i1], q2[i0 + 2:i1 - 2][::-1]]), HAIR_L)
        # serrated black root: fat strokes along the flow, nearly touching at the base, each dying at its own height
        k = int(np.clip(w / (5.0 + 2.2 * lw), 3, 16)); ws = w / k * 1.25; deep = r.uniform(.34, .56) if o['kind'] == 'row' else r.uniform(.22, .44)
        for q in range(k):
            u = (q + .5) / k * 2 - 1 + r.normal(0, .05); sh_ = .5 + .5 * u * ds
            t1 = deep * (.45 + .75 * r.random()) + .22 * sh_ * r.random() + (.22 * r.random() if r.random() < .22 else 0); i1 = max(5, int(n * min(t1, .86)))
            hw_ = hl if u >= 0 else hr; ln = c + N * (hw_ * u)[:, None]
            wb = ws * r.uniform(.55, 1.15) * (.55 + .6 * sh_)
            brush(sh.k, ln[:i1], [wb, wb * .95, wb * .6, wb * .25, .1], seed=int(r.integers(1 << 30)), rough=.2, smooth_=False, tremor=.1)
        # flow lines: from the tip inward, and one long S through the body
        for q in range(int(np.clip(w / (16 * lw + 6), 1, 6))):
            u = r.uniform(-.55, .85) * ds; i0 = int(n * r.uniform(.36, .60)); i1 = int(n * r.uniform(.84, .97)); hw_ = hl if u >= 0 else hr; ln = c + N * (hw_ * u)[:, None]
            if i1 - i0 > 4: brush(sh.k, ln[i0:i1], np.array([.15, 1.5, 1.1, .2]) * lw * r.uniform(.8, 1.3), seed=int(r.integers(1 << 30)), rough=.15, smooth_=False, tremor=.1)
        sh.k.unclip(); sh.a.unclip()

    def _contour(self, poly):
        """open run of the face contour, picture left to right: starts at the ear lobe on the near side, at the temple on the far side"""
        y = self.head_y(poly); k = int(np.argmax(y)); q = np.roll(poly, -k, 0); y = np.roll(y, -k); x = (q - q.mean(0)) @ (self.rt / (np.hypot(*self.rt) + 1e-6))
        far = np.sign(self.yaw) if abs(self.yaw) > .22 else 0
        lim = np.where(x * far > 0, .40, -.34) if far else np.full(len(q), -.30)
        i = np.nonzero(y < lim)[0]
        if len(i) < 8: return q
        run = q[i[0]:i[-1] + 1]
        o = self.H.P3(np.array([[0, 0, .5], [1, 0, .5]])); ax = o[1] - o[0]
        if (run[-1] - run[0]) @ ax < 0: run = run[::-1]
        return run

    def ink(self, dark=1, hatch=1., head_only=False, box=None, neck='light'):
        """dark = +1: shadow side is the picture right (head space), -1 left.  Returns premultiplied RGBA (full frame)"""
        sh = Sheet(self.seed, box=box or self.box()); self._ink_into(sh, dark, hatch, head_only, neck); return sh.layer()

    def mould(self, poly):
        """give the far side of a turned face its brow, socket, cheekbone, muzzle and chin"""
        k = float(np.clip((abs(self.yaw) - .18) / .35, 0, 1)) * float(np.clip((1.25 - abs(self.yaw)) / .3, 0, 1))
        if k <= 0: return poly
        y = self.head_y(poly); ax = self.rt / (np.hypot(*self.rt) + 1e-6) * np.sign(self.yaw); c = self.H.P3(np.array([0, -.2, .3])); x = (poly - c) @ ax
        g = lambda y0, w, a: a * np.exp(-((y - y0) / w) ** 2)
        b = g(.17, .075, .022) + g(.015, .075, -.050) + g(-.21, .11, .030) + g(-.50, .13, -.028) + g(-.68, .07, .016) + g(-.80, .05, -.016) + g(-.93, .07, .022)
        w = np.clip(x / (self.s * .18), 0, 1) * (np.abs(y) < 1.15) * (y < .5)
        return poly + ax * (b * w * k * self.s)[:, None]

    def neck_lines(self, r=.43, z=-.20):
        """left and right neck contour: an upright column whatever the yaw"""
        ys = np.array([-.40, -.70, -1.0, -1.3, -1.6, -1.9, -2.5]); zc = z + np.array([.0, .0, .0, -.02, -.05, -.08, -.1]); rr = r * np.array([1., 1., .98, .98, 1.06, 1.36, 2.2])
        q = np.linspace(-.40, -max(.9, self.neck_len), 7); zc, rr = np.interp(-q, -ys, zc), np.interp(-q, -ys, rr); ys = q
        c = Head.P3(self.H, np.stack([0 * ys, ys, zc], -1)); ax = self.rt / (np.hypot(*self.rt) + 1e-6)
        ax = np.array([math.cos(self.roll), math.sin(self.roll)])
        return c - ax * (rr * self.s)[:, None], c + ax * (rr * self.s)[:, None]

    def split_locks(self):
        L_ = self.locks(); back, low, front = [], [], []
        for o in L_:
            (back if o['face'] < (.10 if o['kind'] == 'front' else -.12) else low if (o['kind'] == 'row' and o['root'][1] < .26 and o['root'][2] < .05) else front).append(o)
        return back, low, front

    def side_skin(self, q):
        return self.H.P3(np.array([(q * .69, .10, .20), (q * .712, .10, -.10), (q * .718, -.10, -.14), (q * .714, -.28, -.12), (q * .70, -.36, -.07), (q * .68, -.36, .12)]))

    def _ink_into(self, sh, dark=1, hatch=1., head_only=False, neck='light'):
        s, lw, e = self.s, self.lw, self.e; H = self.H; K = sh.k; sd = self.seed
        back, low, front = self.split_locks(); face = self.mould(H.face()); skull = H.skull(); gap = 6.0 if s > 150 else 4.6; self.hatch = hatch
        B = lambda P, ws, **k: brush(K, P, np.asarray(ws, float) * lw, **k)
        F = self.C
        rear = (self.H.R @ np.array([0, 0, 1.]))[2] < -.12
        if rear:
            sh.put(face, SKIN); sh.wash(face, SKIN_S, .8); B(self._contour(face)[::3], [2., 3.4, 4.2, 3.4, 2.], seed=sd + 22, smooth_=False)
        else:
            for o in back: self._lock_ink(sh, o, dark)
            sh.put(skull, '#3a1820'); K.fill(skull, BLACK)
        jl = np.vstack([H.jaw(-1), H.jaw(1)[::-1][1:]])
        # ---- neck
        if not head_only:
            nl, nr = self.neck_lines(); nk = np.vstack([spl(nl, n=6), spl(nr, n=6)[::-1]]); sh.put(nk, SKIN)
            dn = -self.up; ju = jl[np.linspace(0, len(jl) - 1, 9).astype(int)]; d_ = dark
            pr = np.array([.62, .40, .34, .52, .30, .46, .58, .66, .80]) if d_ > 0 else np.array([.80, .66, .58, .46, .30, .52, .34, .40, .62])
            sha = spl(ju + dn[None] * pr[:, None] * .62, corners=(3, 5), n=6); sh.wash(np.vstack([jl, sha[::-1]]), SKIN_D, .8, within=nk)
            if neck == 'dark': K.fill(nk, BLACK)
            K.clip(nk); bl = spl(ju + dn[None] * pr[:, None] * .30, corners=(3, 5), n=6); K.fill(np.vstack([jl, bl[::-1]]), BLACK)
            feather(K, bl, sha + dn * .06, gap * .75, 3.2 * lw, seed=sd + 1, reach=(.35, 1.1), group=6, hole=.6, back=.1)
            sid = spl(nr, n=6) if d_ > 0 else spl(nl, n=6); mid = H.P3(np.array([(d_ * .16, -.9, .1), (d_ * .14, -1.3, .12), (d_ * .22, -1.8, .1)]))
            feather(K, sid, spl(mid, n=6), gap * .8, 2.6 * lw, seed=sd + 5, reach=(.25, .62), group=7, hole=.5, env=lambda t: .6 + .5 * t)
            K.unclip()
            B(spl(nl, n=5)[::3], [3.0, 3.6, 3.0, 2.0], seed=sd + 2); B(spl(nr, n=5)[::3], [3.4, 4.2, 3.4, 2.2], seed=sd + 3)
            for q in ((-1, 1) if neck != 'dark' and abs(self.yaw) < .9 else ()): B(H.P3(np.array([(q * .34, -.90, .06), (q * .20, -1.25, .12), (q * .07, -1.72, .20)])), [.4, 2.0, 2.6], seed=sd + 40 + q)
        if self.pitch > .02:
            un = H.under(); sh.put(un, SKIN_D); K.clip(un)
            c_ = un.mean(0); feather(K, jl * .25 + c_ * .75, jl, gap * .8, 2.6 * lw, seed=sd + 8, reach=(.5, 1.05), back=.2, hole=.35); K.unclip()
        if rear:
            sh.put(skull, '#3a1820'); K.fill(skull, BLACK)
        for o in (low if not rear else [q for q in low if q['face'] > -.1]): self._lock_ink(sh, o, dark)
        # ---- side of the face in front of the ear, ears
        for q in (-1, 1):
            er = H.ear(q)
            if er['vis'] < .02: continue
            if self.noir and q == dark:
                sh.k.fill(self.side_skin(q), BLACK); o = er['outer']; sh.put(o, self.ac['earfill']); sh.k.fill(o, BLACK); brush(sh.g, o[int(len(o) * .30):int(len(o) * .78)][::3], np.array([.3, 2.6, 3.2, 2., .3]) * lw, self.ac['ear'], sd, .5); brush(sh.g, er['inner'][int(len(er['inner']) * .12):int(len(er['inner']) * .5)][::2], np.array([.3, 1.8, .3]) * lw, self.ac['ear2'], sd + 1, .5); continue
            sk = self.side_skin(q); sh.put(sk, SKIN); sh.wash(sk, SKIN_S, .85 if q == dark else .45)
            o = er['outer']; i = er['inner']; tr = er['tragus']; sh.put(o, SKIN); sh.wash(o, SKIN_S, .35 if q == -dark else .8); sh.wash(i, SKIN_D, .8)
            B(o[::4], [1.4, 3.0, 4.4, 4.0, 2.8, 3.2, 1.6], seed=sd + 10 + q)
            n = len(i); B(i[: int(n * .60)][::3], [.5, 2.6, 2.0, .5], seed=sd + 12 + q); K.fill(tr, BLACK)
            hi = er['outer'].mean(0); K.fill(np.vstack([i[int(n * .10):int(n * .34)], (i[int(n * .10):int(n * .34)] * .72 + hi * .28)[::-1]]), BLACK, .95)
            feather(K, i[int(n * .55):], i[: int(n * .45)][::-1], gap * .8, 2.0 * lw, seed=sd + 14 + q, reach=(.25, .8), hole=.3)
            feather(K, o[int(len(o) * .60):int(len(o) * .86)], i[int(n * .05):int(n * .30)][::-1], gap * .8, 1.6 * lw, seed=sd + 16 + q, reach=(.3, .9), hole=.4)
        if rear:
            for o in front: self._lock_ink(sh, o, dark)
            return sh
        # ---- face
        sh.put(face, SKIN); sh.clip(face)
        self._face_wash(sh, dark); self._face_hatch(sh, dark, gap, hatch)
        hp, hpoly = self.hairline(); K.fill(hpoly, BLACK)
        sh.unclip()
        self._features_ink(sh, dark, gap, face)
        # ---- contour
        run = self._contour(face); n = len(run); y = self.head_y(run); kc = int(np.clip(np.argmin(y), 8, n - 8))
        wl = np.array([1.2, 2.4, 3.0, 4.0, 4.6, 3.6, 3.8, 2.8, 1.2]); wr = np.array([1.2, 3.8, 5.2, 6.4, 6.8, 5.6, 5.4, 4.0, 1.8])
        a, b = (wl, wr) if dark > 0 else (wr, wl)
        B(run[:kc + 3][::3], a, seed=sd + 20, smooth_=False, tremor=.15); B(run[max(0, kc - 3):][::-1][::3], b, seed=sd + 21, smooth_=False, tremor=.15)
        # ---- hair in front
        for o in front: self._lock_ink(sh, o, dark)
        self._hairline_strokes(sh, hp)
        self._petals(sh)
        return sh

    def _hairline_strokes(self, sh, hp):
        """short strokes that start on the forehead and run up into the roots"""
        ok = np.array([self.vis(x, y) > .16 for x, y in spl(self.HL, n=6)]); hp = hp[ok] if ok.sum() > 6 else hp
        r = np.random.default_rng(5); hp2 = resample(hp, 3.0 + 1.2 * self.lw); c0 = self.H.P3(np.array([0, .22, .6])); s = self.s; sway = 0.
        for k in range(len(hp2)):
            p = hp2[k]; d = p - c0; d /= np.hypot(*d) + 1e-9; uu = self.up / np.hypot(*self.up); low_ = float(np.clip((.34 - self.head_y(p[None])[0]) / .25, 0, 1)); d = d * (1 - .75 * low_) + uu * low_ * .9; d /= np.hypot(*d) + 1e-9
            if k % 7 == 0: sway = r.normal(0, .25)
            c, q = math.cos(sway), math.sin(sway); d = np.array([d[0] * c - d[1] * q, d[0] * q + d[1] * c])
            l = s * (.07 + .12 * r.random() ** 1.5); st = p - d * s * (.006 + .05 * r.random() ** 3.2)
            brush(sh.k, np.array([st, p + d * l * .4 + r.normal(0, .8, 2), p + d * l]), np.array([.1, 2.6, 2.6, 1.2]) * self.lw * r.uniform(.6, 1.15), seed=k, rough=.15, tremor=.1)

    def _petals(self, sh):
        for g, d in self.petals: sh.k.fill(g, BLACK if d < .5 else BLOOD_D, 1 - max(0, d - .75) * 4)

    def _face_wash(self, sh, dark):
        """dirty grey wash: one designed shadow mass (split light) + sockets + under planes"""
        F = self.C; d = dark; l = -dark
        sh.wash(F([(d * .80, .90), (d * .30, .72), (d * .36, .46), (d * .30, .30), (d * .46, .22), (d * .56, .06), (d * .50, -.10), (d * .58, -.22), (d * .46, -.33), (d * .37, -.46), (d * .355, -.60), (d * .27, -.70), (d * .25, -.82), (d * .14, -.93), (d * .10, -1.1), (d * .82, -1.1)], closed=True, corners=(0, 1, 3, 4, 7, 11, 14, 15), n=8), SKIN_S, .85)
        sh.wash(F([(d * .80, .5), (d * .62, .30), (d * .645, .05), (d * .60, -.12), (d * .63, -.34), (d * .53, -.50), (d * .46, -.66), (d * .33, -.84), (d * .20, -1.1), (d * .84, -1.1)], closed=True, corners=(0, 8, 9)), SKIN_D, .55)
        sh.wash(F([(l * .80, .6), (l * .60, .52), (l * .625, .20), (l * .67, -.05), (l * .62, -.30), (l * .48, -.52), (l * .39, -.72), (l * .22, -.94), (l * .16, -1.1), (l * .80, -1.1)], closed=True), SKIN_S, .5)
        if (self.H.R @ np.array([d, 0., 0.]))[2] > .35: sh.wash(self.H.P3(np.array([(d * .70, .70, .10), (d * .70, .60, -.20), (d * .74, .0, -.22), (d * .73, -.40, -.16), (d * .66, -.62, .02), (d * .69, -.30, .16), (d * .70, .10, .16)])), SKIN_D, .8)
        for q in (-1, 1):
            if self.vis(q * .34, 0) < .05: continue
            # inner socket wedge running into the side of the nose, and the hollow under the eye
            sh.wash(F([(q * .070, .10), (q * .20, .105), (q * .34, .165), (q * .47, .165), (q * .56, .10), (q * .47, .072), (q * .34, .070), (q * .24, .045), (q * .17, -.02), (q * .13, -.10), (q * .10, -.20), (q * .075, -.06)], closed=True, corners=(0, 4, 8, 10)), SKIN_D, .62)
            sh.wash(F([(q * .125, -.085), (q * .21, -.118), (q * .32, -.130), (q * .42, -.118), (q * .34, -.172), (q * .22, -.196), (q * .13, -.17)], closed=True, corners=(0, 3)), SKIN_S, .8)
        sh.wash(F([(d * .015, .02), (d * .055, -.12), (d * .05, -.34), (d * .012, -.43), (d * .10, -.475), (d * .145, -.43), (d * .12, -.30), (d * .10, -.12), (d * .105, .04)], closed=True, corners=(0, 3, 8)), SKIN_D, .5)
        sh.wash(F([(-.085, -.468), (0, -.452), (.085, -.468), (.04, -.515), (d * .03, -.555), (-.05, -.515)], closed=True, corners=(0, 2, 4)), SKIN_D, .75)
        sh.wash(F([(-.15, -.742), (0, -.728), (.15, -.742), (.06, -.790), (d * .02, -.815), (-.08, -.785)], closed=True, corners=(0, 2, 4)), SKIN_S, .9)

    def _face_hatch(self, sh, dark, gap, hatch):
        F = self.C; K = sh.k; lw = self.lw; sd = self.seed; d = dark; l = -dark
        if hatch <= 0: return
        def fe(A, Bc, w=2., reach=(.5, 1.), seed=0, side=None, **k):
            if side is not None and self.vis(*side) < .06: return
            feather(K, F(A), F(Bc), gap / hatch, w * lw, seed=sd * 3 + seed, reach=reach, **k)
        if self.noir:
            z = self.noir - 1
            T_ = [(d * .44 - z * d * .06, .80), (d * .47, .46), (d * .585, .24), (d * .575, .06), (d * .60, -.08), (d * .57, -.20), (d * .50 - z * d * .1, -.30), (d * .455 - z * d * .12, -.42), (d * .47 - z * d * .12, -.56), (d * .40 - z * d * .1, -.70), (d * .34, -.84), (d * .24, -1.04)]
            Tc = F(T_, n=12, corners=(2, 4, 8)); K.fill(np.vstack([Tc, F([(d * .9, -1.1), (d * .9, .9)])]), BLACK)
            In = F([(x - d * (.16 + .05 * (i % 3)), y - .05 - .03 * (i % 2)) for i, (x, y) in enumerate(T_)], n=12)
            feather(K, Tc, In, gap * .85, 4.2 * lw, seed=sd + 300, reach=(.18, .95), group=8, hole=.7, back=.25, bow=-.05 * d)
            feather(K, Tc, F([(x - d * .07, y - .03) for x, y in T_], n=12), gap * .8, 5. * lw, seed=sd + 301, reach=(.3, 1.), group=5, hole=.4, back=.3)
            feather(sh.g, F([(d * .715, -.30), (d * .64, -.50), (d * .53, -.68), (d * .40, -.82), (d * .26, -.95)]), F([(d * .64, -.34), (d * .575, -.50), (d * .48, -.65), (d * .36, -.78), (d * .23, -.90)]), gap * 1.5, 2.2 * lw, self.ac['scratch'], sd + 302, (.2, 1.), group=7, hole=.75, skew=-.04)
            feather(sh.g, F([(d * .66, -.02), (d * .65, -.12), (d * .62, -.22)]), F([(d * .60, .0), (d * .585, -.14), (d * .55, -.25)]), gap * 1.5, 2. * lw, self.ac['scratch2'], sd + 303, (.2, .9), group=6, hole=.7)
        # temples: strokes drop out of the hair down the side plane; the dark side long and layered
        fe([(d * .36, .68), (d * .48, .63), (d * .575, .52), (d * .62, .38)], [(d * .44, .20), (d * .53, .16), (d * .60, .06), (d * .655, -.06)], 2.4, (.35, 1.), 1, (d * .52, .4), group=7, hole=.55)
        fe([(d * .52, .60), (d * .60, .46), (d * .64, .30)], [(d * .585, .10), (d * .635, .02), (d * .685, -.10)], 2.6, (.6, 1.1), 2, (d * .52, .4), group=5, hole=.3, skew=.04)
        fe([(l * .50, .62), (l * .575, .52), (l * .63, .36)], [(l * .55, .24), (l * .61, .14), (l * .665, .02)], 2.0, (.3, .95), 3, (l * .52, .4), group=6, hole=.6)
        # under the cheekbone: strokes follow the masseter
        fe([(d * .70, -.14), (d * .66, -.32), (d * .60, -.48), (d * .52, -.63), (d * .40, -.78)], [(d * .50, -.26), (d * .42, -.42), (d * .36, -.56), (d * .33, -.68), (d * .24, -.84)], 2.5, (.35, 1.), 4, (d * .5, -.4), group=7, hole=.6, bow=-.06 * d)
        fe([(d * .72, -.10), (d * .68, -.30), (d * .63, -.46), (d * .55, -.62), (d * .43, -.78), (d * .28, -.94)], [(d * .58, -.22), (d * .56, -.38), (d * .51, -.50), (d * .44, -.64), (d * .34, -.78), (d * .20, -.95)], 3.0, (.6, 1.15), 5, (d * .5, -.4), group=5, hole=.35, skew=.05)
        fe([(l * .70, -.42), (l * .63, -.52), (l * .55, -.64), (l * .45, -.76), (l * .32, -.90)], [(l * .53, -.56), (l * .47, -.66), (l * .40, -.76), (l * .33, -.86), (l * .20, -.97)], 2.2, (.35, 1.), 6, (l * .5, -.4), group=5, hole=.6)
        for q in (-1, 1):
            if self.vis(q * .30, 0) < .08: continue
            k = 1. if q == d else .85
            # socket: a black hook from the brow down the nose root into the inner corner, feathered out along the lid
            K.fill(F([(q * .074, .096), (q * .150, .078), (q * .215, .098), (q * .250, .118), (q * .222, .070), (q * .176, .020), (q * .146, -.040), (q * .122, -.075), (q * .104, -.030), (q * .086, .030)], closed=True, corners=(0, 3, 7)), BLACK)
            fe([(q * .168, .092), (q * .182, .040), (q * .150, -.02)], [(q * .52, .135), (q * .46, .078), (q * .33, .020)], 2.7 * k, (.30, 1.), 20 + q, group=6, hole=.55, back=.10, bow=.04 * q)
            fe([(q * .150, .07), (q * .160, .03), (q * .140, -.03)], [(q * .30, .088), (q * .27, .050), (q * .21, .004)], 3.2 * k, (.5, 1.1), 21 + q, group=4, hole=.3, back=.25)
            fe([(q * .124, -.03), (q * .112, -.09), (q * .104, -.14)], [(q * .098, -.16), (q * .11, -.24), (q * .13, -.30)], 2.4 * k, (.35, 1.), 24 + q, group=5, hole=.4)
            # bag under the eye: long sweeps from the inner corner round the orbit
            fe([(q * .130, -.078), (q * .120, -.120), (q * .112, -.160)], [(q * .46, -.118), (q * .40, -.176), (q * .27, -.236)], 2.6 * k, (.30, 1.), 26 + q, group=6, hole=.5, bow=-.10 * q)
            fe([(q * .300, -.128), (q * .40, -.120), (q * .47, -.092)], [(q * .22, -.196), (q * .34, -.196), (q * .44, -.168)], 1.9 * k, (.3, .9), 27 + q, group=5, hole=.5, skew=-.05)
            # crow's feet
            fe([(q * .565, .045), (q * .560, .005), (q * .530, -.060)], [(q * .73, .07), (q * .75, -.06), (q * .70, -.20)], 2.5 * k, (.35, 1.), 28 + q, group=5, hole=.5, bow=.05 * q)
            fe([(q * .30, .150), (q * .44, .182), (q * .58, .190)], [(q * .42, .095), (q * .53, .105), (q * .63, .10)], 1.9 * k, (.2, .9), 30 + q, group=6, hole=.6)
        fe([(l * .70, -.05), (l * .70, -.20), (l * .66, -.34), (l * .60, -.48)], [(l * .60, -.14), (l * .57, -.28), (l * .52, -.42), (l * .47, -.56)], 2.2, (.2, .9), 50, (l * .55, -.3), group=8, hole=.8, skew=.05)
        # nose: dark flank, and the cast shadow below
        fe([(d * .066, .02), (d * .070, -.14), (d * .078, -.28), (d * .112, -.40)], [(d * .14, -.12), (d * .17, -.28), (d * .18, -.38), (d * .20, -.46)], 2.2, (.3, 1.), 40, (d * .12, -.2), group=6, hole=.5, skew=.10)
        fe([(-d * .02, -.486), (d * .03, -.492), (d * .085, -.484)], [(d * .055, -.548), (d * .085, -.556), (d * .125, -.530)], 2.0, (.3, .9), 41, group=4, hole=.5)
        fe([(-d * .03, -.748), (d * .05, -.745), (d * .125, -.750)], [(d * .06, -.800), (d * .12, -.812), (d * .17, -.790)], 2.1, (.25, .9), 42, group=5, hole=.5)

    # --------------------------------------------------------- mouth geometry (front coordinates)
    def mouth(self):
        m = self.e['mouth']; J = self.e['jaw']
        D = dict(
            flat=dict(cl=(-.165, -.652), cr=(.160, -.655), up=[(-.08, -.644), (0, -.652), (.08, -.646)], lo=None),
            hard=dict(cl=(-.186, -.646), cr=(.204, -.672), up=[(-.09, -.650), (.02, -.653), (.12, -.660)], lo=None),
            smirk=dict(cl=(-.150, -.668), cr=(.205, -.612), up=[(-.06, -.656), (.02, -.664), (.11, -.650)], lo=None),
            smile=dict(cl=(-.272, -.552), cr=(.322, -.500), up=[(-.17, -.606), (-.06, -.630), (.04, -.632), (.14, -.612), (.24, -.564)], lo=[(-.20, -.690), (-.08, -.760), (.06, -.772), (.18, -.722), (.270, -.618)]),
            laugh=dict(cl=(-.300, -.535), cr=(.318, -.520), up=[(-.20, -.575), (-.08, -.600), (.02, -.602), (.13, -.592), (.23, -.558)], lo=[(-.24, -.665), (-.12, -.772), (.02, -.800), (.15, -.772), (.262, -.655)]),
            grit=dict(cl=(-.205, -.640), cr=(.285, -.560), up=[(-.13, -.622), (-.04, -.634), (.05, -.626), (.17, -.588)], lo=[(-.14, -.694), (-.02, -.716), (.09, -.702), (.20, -.634)]),
        )
        if not isinstance(m, str):                          # ('mix', t[, drop]): corners run to full width first (t < .42), then the lips part (t > .30).  1 = 'smile', 1.1 overshoots; drop 0..1 lowers the corners
            t = float(m[1]); dr = float(m[2]) if len(m) > 2 else 0.; A_, B_ = D['flat'], D['smile']; kc = float(smooth(0., .42, t)) + max(0., t - 1); kl = float(np.clip((t - .30) / .70, 0, 1.12)); V = lambda u, v, k: np.array(u, float) + (np.array(v, float) - np.array(u, float)) * k
            fl = spl(np.array([A_['cl']] + A_['up'] + [A_['cr']]), n=8); cl = V(A_['cl'], B_['cl'], kc) - (0, .055 * dr); cr = V(A_['cr'], B_['cr'], kc) - (0, .070 * dr); up = [tuple(V(at(fl, (k + 1) / 6.), v, kc)) for k, v in enumerate(B_['up'])]
            if kl < .05: return 'closed', dict(cl=tuple(cl), cr=tuple(cr), up=up, lo=None)
            bu = spl(np.array([B_['cl']] + B_['up'] + [B_['cr']]), n=8); mu = spl(np.array([cl] + up + [cr]), n=8); lo = [tuple(at(mu, (k + 1) / 6.) + (np.array(v) - at(bu, (k + 1) / 6.)) * kl) for k, v in enumerate(B_['lo'])]
            return 'smile', dict(cl=tuple(cl), cr=tuple(cr), up=up, lo=lo)
        return m, D[m]

    def _mouth_ink(self, sh, dark, gap):
        K = sh.k; lw = self.lw; sd = self.seed; F = self.C; s = self.s
        if self.vis(0, -.64) < -.15: return
        B = lambda P, ws, **k: brush(K, P, np.asarray(ws, float) * lw * self.bg, **k)
        m, D = self.mouth(); cl, cr = np.array(D['cl']), np.array(D['cr'])
        if D['lo'] is None:
            P = F([cl] + D['up'] + [cr]); B(P, [3.0, 2.2, 2.6, 2.0, 3.2], seed=sd + 70, smooth_=False)
            B(F([cl + (-.022, .006), cl + (.0, -.006), cl + (.012, -.022)]), [.3, 2.4, .4], seed=sd + 71); B(F([cr + (.022, .006), cr + (.0, -.006), cr + (-.012, -.024)]), [.3, 2.4, .4], seed=sd + 72)
            B(F([(-.07, -.742), (.0, -.750), (.065, -.740)]), [.3, 2.0, .4], seed=sd + 73)
            if m == 'smirk': B(F([(.235, -.545), (.262, -.60), (.250, -.655)]), [.3, 2.0, .3], seed=sd + 74)
            return
        up = spl(np.array([cl] + D['up'] + [cr]), n=8); lo = spl(np.array([cl] + D['lo'] + [cr]), n=8)
        U, Lw = self.P(up), self.P(lo); poly = np.vstack([U, Lw[::-1]])
        sh.cut(poly); sh.a.fill(poly, '#e9e0cf'); K.fill(poly, BLACK)
        # teeth: cut out of black, upper and lower rows with wedge gaps
        def tooth(Q, S, base, opp, A0, B0, e0, e1, q, seed, k):
            far = abs(q - .5) * 2
            sh.wash(S, '#8f7f78', float(np.clip(far * 1.1 - .45, 0, .75)))
            if s > 420:
                ga, gb = self.P(np.array([A0, B0])); ea, eb = self.P(np.array([e0, e1]))
                feather(K, np.array([ga, gb]), np.array([ea, eb]), gap * .8, 1.3 * lw, seed=seed + k, reach=(.08, .34), hole=.5, group=3)
                if far > .35: feather(K, np.array([gb, eb]) if q > .5 else np.array([ga, ea]), np.array([ga, ea]) if q > .5 else np.array([gb, eb]), gap * .8, 1.3 * lw, seed=seed + 50 + k, reach=(.1, .45), hole=.4, group=3)
        snag = []

        def row(base, opp, frac, n, sgn, seed, fang=(), hook=-1):
            r = np.random.default_rng(abs(seed)); T = np.linspace(.035, .965, n + 1) + np.r_[0, r.normal(0, .0125, n - 1), 0]; T = np.clip(T, .02, .98)
            for k in range(n):
                t0, t1 = T[k], T[k + 1]; q = (t0 + t1) / 2; fr = frac * r.uniform(.76, 1.06); tl = r.normal(0, .10) * (t1 - t0); g = min(.16 * (t1 - t0), .0055)
                a, b = at(base, t0 + g), at(base, t1 - g); c, d_ = at(opp, np.clip(t0 + g * 1.5 + tl, .01, .99)), at(opp, np.clip(t1 - g * 1.5 + tl, .01, .99)); mid = at(opp, np.clip(q + tl, 0, 1)); bm = at(base, q)
                ed = .05 + .04 * r.random(); A0 = a + (c - a) * ed; B0 = b + (d_ - b) * ed; e0 = a + (c - a) * fr * r.uniform(.92, 1.06); e1 = b + (d_ - b) * fr * r.uniform(.92, 1.06); em = bm + (mid - bm) * fr * 1.05
                if k in fang:
                    ex = 1.20 if k == hook else min(1.0, fr * 1.38); P_ = [A0, bm + (mid - bm) * .02, B0, b + (d_ - b) * max(fr * .55, .42 if k == hook else 0), bm + (mid - bm) * ex, a + (c - a) * max(fr * .55, .42 if k == hook else 0)]; Q = spl(np.array(P_), True, (0, 2, 4), n=5)
                else: Q = spl(np.array([A0, bm + (mid - bm) * .02, B0, e1, em, e0]), True, (0, 2, 3, 5), n=4)
                S = self.P(Q)
                if k == hook: snag.append((Q, S, A0, B0, e0, e1, q, seed, k)); continue
                sh.cut(S, K); tooth(Q, S, base, opp, A0, B0, e0, e1, q, seed, k)
        if m == 'smile':
            row(lo, up, .33, 10, -1, sd + 2, fang=(1,)); row(up, lo, .55, 11, 1, sd + 1, fang=(2, 8), hook=8 if not isinstance(self.e['mouth'], str) and self.e['mouth'][1] < .7 else 2 if self.yaw > 0 else 8)
        elif m == 'grit':
            row(up, lo, .50, 11, 1, sd + 1, fang=(2, 8)); row(lo, up, .46, 10, -1, sd + 2)
        else:
            row(up, lo, .26, 11, 1, sd + 1, fang=(2, 8)); row(lo, up, .17, 10, -1, sd + 2)
            tg = F([(-.16, -.70), (-.06, -.655), (.04, -.660), (.14, -.700), (.12, -.775), (0, -.80), (-.11, -.775)], closed=True)
            sh.clip(poly); sh.cut(tg, K); sh.a.fill(tg, '#a3202a'); sh.wash(tg, '#4a0c14', .5); B(F([(.0, -.67), (.008, -.73), (.0, -.78)]), [.4, 2.2, .4], seed=sd + 77); feather(K, F([(-.16, -.70), (-.06, -.655), (.04, -.660), (.14, -.70)]), F([(-.12, -.78), (-.04, -.76), (.04, -.76), (.11, -.78)]), gap, 1.8 * lw, seed=sd + 78, reach=(.2, .7))
            sh.unclip()
        # lips: the upper edge is a long brush stroke that overshoots the corners; lower edge broken
        ex = lambda p, v: np.array(p) + v
        B(F([ex(cl, (-.045, .030)), cl] + D['up'] + [cr, ex(cr, (.050, .034))]), [.4, 4.4, 3.6, 2.6, 3.0, 2.6, 3.8, 5.2, .4], seed=sd + 75)
        n_ = len(Lw); B(Lw[int(n_ * .02):int(n_ * .44)], [2.6, 2.6, 2.0, .5], seed=sd + 76, smooth_=False); B(Lw[int(n_ * .50):int(n_ * .99)], [.5, 2.6, 3.0, 3.4], seed=sd + 79, smooth_=False)
        for (Q, S, A0, B0, e0, e1, q, sd_, k) in snag:               # the canine that will not stay behind the lip
            sh.cut(S, K); sh.cut(S, sh.v); sh.a.fill(S, '#efe6d6'); tooth(Q, S, up, lo, A0, B0, e0, e1, q, sd_, k); m_ = len(S); i = int(m_ * .40); B(S[i:int(m_ * .72)], [2.0, 2.6, 2.2], seed=sd + 97, smooth_=False); B(S[int(m_ * .64):int(m_ * .98)], [2.4, 2.0, 1.2], seed=sd + 98, smooth_=False)
        # corner creases and laugh lines: folds with hatching that runs along the cheek.  In an in-between the fold only meets the corner from above: nothing runs under it, so no ring
        ck = 1. if isinstance(self.e['mouth'], str) else float(np.clip((self.e['mouth'][1] - .62) / .30, 0, 1)); cw = .35 + .65 * ck
        for c_, q in ((cl, -1), (cr, 1)):
            if self.vis(c_[0] + q * .05, c_[1]) < .10: continue
            if ck > .5:
                B(F([c_ + (q * .026, .085), c_ + (q * .060, .010), c_ + (q * .042, -.075)]), [.3, 3.6 * cw, .4], seed=sd + 80 + q)
                B(F([c_ + (q * .070, .050), c_ + (q * .104, -.01), c_ + (q * .088, -.085)]), [.3, 2.0 * cw, .3], seed=sd + 82 + q)
                feather(K, F([c_ + (q * .052, -.02), c_ + (q * .020, -.10), c_ + (-q * .05, -.17)]), F([c_ + (q * .20, -.10), c_ + (q * .15, -.20), c_ + (q * .05, -.27)]), gap * .9, 2.2 * lw, seed=sd + 88 + q, reach=(.25, 1.), group=6, hole=.6, skew=.05)
                fold_ = [(q * .112, -.405), (q * .170, -.436), (q * .250, -.492), c_ + (q * .045, .055)]
            else:                                            # the corner is pulled INTO the cheek: one hook outward and up, the fold lands on it
                B(F([c_ + (-q * .012, .004), c_ + (q * .030, .022), c_ + (q * .056, .064)]), [2.6, 3.4, .3], seed=sd + 80 + q)
                fold_ = [(q * .112, -.405), (q * .190, -.428), (c_[0] * .55 + q * .150, c_[1] * .55 - .210), c_ + (q * .062, .060)]
            B(F(fold_), [.4, 3.2 * cw, 2.6 * cw, .4], seed=sd + 84 + q)
            feather(K, F([(q * .150, -.428), (q * .225, -.474), c_ + (q * .050, .060), c_ + (q * .075, -.01)]), F([(q * .36, -.35), (q * .45, -.40), c_ + (q * .22, .10), c_ + (q * .21, .03)]), gap * .9, 2.4 * lw, seed=sd + 86 + q, reach=(.20, .5 + .4 * ck), group=7, hole=.65, skew=-.06, bow=.06 * q)
            # cheekbone pushed up by the grin
            B(F([(q * .30, -.285), (q * .42, -.300), (q * .53, -.262)]), [.3, 2.2, .3], seed=sd + 92 + q)
            feather(K, F([(q * .30, -.290), (q * .42, -.306), (q * .53, -.270)]), F([(q * .36, -.375), (q * .47, -.395), (q * .57, -.350)]), gap, 2.0 * lw, seed=sd + 94 + q, reach=(.2, .5 + .4 * ck), group=6, hole=.6, skew=.04)
        lc = [(-.11, -.775), (0, -.800), (.12, -.785)] if m != 'laugh' else [(-.10, -.83), (.0, -.845), (.11, -.83)]
        B(F(lc), [.3, 2.8, .3], seed=sd + 90); lc = np.array(lc); feather(K, F(lc[1:] if dark > 0 else lc[:2]), F((lc[1:] if dark > 0 else lc[:2]) + (dark * .055, -.060)), gap, 1.9 * lw, seed=sd + 91, reach=(.3, 1.), group=4, hole=.5)

    def _eye_ink(self, sh, q, dark, gap):
        K = sh.k; lw = self.lw; sd = self.seed + q * 7; P = self.P; s = self.s
        if self.vis(q * .34, -.02) < .06: return
        B = lambda Pt, ws, **k: brush(K, Pt, np.asarray(ws, float) * lw * self.bg, **k)
        E = self.eye_lines(q); up = spl(E['up'], n=8); lo = spl(E['lo'], n=8)
        if E['o'] > .05:
            op = np.vstack([up, lo[::-1]]); OP = P(op); sh.cut(OP); sh.a.fill(OP, '#efe6d4')
            lx, ly = self.e['look']; o = E['o']
            ik = self.e.get('iris', (1., 1.))[0 if q < 0 else 1]; c = np.array([q * .348 + lx * .105, -.026 + ly * .05]); rx, ry = .052 * ik, .062 * ik
            a = np.linspace(0, 2 * math.pi, 26, endpoint=False)
            ring = lambda k: P(np.c_[c[0] + np.cos(a) * rx * k, c[1] + np.sin(a) * ry * k])
            sh.a.clip(OP); K.clip(OP); sh.v.clip(OP)
            sh.a.fill(ring(1.), '#a8101b'); sh.a.fill(ring(.80), '#e02a32'); sh.wash(np.vstack([P(up), P(up * [1, 1] - (0, .040))[::-1]]), '#8d7f80', .8)
            K.fill([ring(1.10), ring(.88)], BLACK, evenodd=True)
            r = np.random.default_rng(abs(sd) + 3)
            for k in range(int(np.clip(s / 34, 7, 22))):
                th = 2 * math.pi * (k + r.uniform(-.3, .3)) / int(np.clip(s / 34, 7, 22)); k1 = r.uniform(.58, .76); u = np.array([math.cos(th) * rx, math.sin(th) * ry])
                brush(K, P(np.array([c + u * .92, c + u * k1])), np.array([2.2, 1.2, .1]) * np.clip(s / 420, .5, 2.2), seed=sd + k, rough=.1)
            K.fill(P(spl(np.array([(c[0], c[1] + ry * .95), (c[0] + rx * .30, c[1] + ry * .1), (c[0] + .002, c[1] - ry * .74), (c[0] - rx * .28, c[1] + ry * .08)]), True, (0, 2), n=6)), BLACK)
            if s > 260: sh.g.clip(OP); sh.g.fill(P(spl(np.array([(c[0] - rx * .62, c[1] - ry * .02), (c[0] - rx * .50, c[1] + ry * .14), (c[0] - rx * .38, c[1] + ry * .02), (c[0] - rx * .50, c[1] - ry * .10)]), True, n=5)), '#f7c9bf'); sh.g.unclip()
            K.fill(P(np.vstack([up, (up - (0, .014))[::-1]])), BLACK)
            feather(K, P(up[2:-2]), P(up[2:-2] - (0, .060)), gap * .75, 2.2 * lw, seed=sd + 41, reach=(.15, .8), hole=.6, group=5)
            if s > 350: feather(K, P(lo), P(lo + (0, .03)), gap, 1.2 * lw, seed=sd + 40, reach=(.2, .7), hole=.5)
            sh.v.unclip(); K.unclip(); sh.a.unclip()
        # upper lid: fat, presses down at the inner corner and flicks past the outer one
        fl = np.vstack([E['inn'] + (-q * .030, -.026), E['up'][0:1], E['up'][1:-1], E['out'], E['out'] + (q * .052, .026 if E['o'] > .05 else -.004)])
        th = 6.2 if E['o'] > .05 else 5.0
        B(P(spl(fl, n=8)), [.4, 2.8, th, th * .86, th * 1.05, th * 1.15, 3.4, .3], seed=sd + 50, smooth_=False)
        # lashes ticks on the outer third
        for k, t in enumerate((.74, .84, .93)):
            p = at(up, t); B(P(np.array([p, p + (q * .030, .012 + .010 * k)])), [2.2, .2], seed=sd + 60 + k, smooth_=False)
        if E['o'] > .05:
            l = lo[int(len(lo) * .42):int(len(lo) * .97)]; B(P(l), [.3, 1.8, 2.6, 2.0], seed=sd + 52, smooth_=False); l = lo[2:int(len(lo) * .26)]; B(P(l), [2.2, 1.4, .3], seed=sd + 53, smooth_=False)
            cr = spl(E['up'][1:] + (0, .034), n=8); B(P(cr[int(len(cr) * .04):int(len(cr) * .86)]), [.3, 1.6, 2.0, .3], seed=sd + 54, smooth_=False)
        else:
            cr = spl(E['clo'][1:] + (0, .060), n=8); B(P(cr[int(len(cr) * .05):int(len(cr) * .82)]), [.3, 1.8, 1.8, .3], seed=sd + 54, smooth_=False)
            feather(K, P(up[int(len(up) * .4):]), P(up[int(len(up) * .4):] + (q * .03, -.05)), gap * .8, 1.6 * lw, seed=sd + 55, reach=(.3, 1.), hole=.4)

    def _eye2_ink(self, sh, q, gap):
        K = sh.k; lw = self.lw; sd = self.seed + q * 9 + 200; P = self.P
        if self.vis(q * .40, -.2) < .06 or self.marks <= 0: return
        B = lambda Pt, ws, **k: brush(K, Pt, np.asarray(ws, float) * lw * self.bg, **k)
        E = self.eye2_lines(q); up = spl(E['up'], n=8); lo = spl(E['lo'], n=8)
        if E['o'] > .05:
            OP = P(np.vstack([up, lo[::-1]])); sh.cut(OP); sh.a.fill(OP, '#e6dac6'); sh.a.clip(OP); K.clip(OP)
            lx, ly = self.e['look']; c = np.array([q * .398 + lx * .05, -.212 + ly * .02]); a = np.linspace(0, 6.283, 20, endpoint=False)
            R = lambda k: P(np.c_[c[0] + np.cos(a) * .036 * k, c[1] + np.sin(a) * .042 * k])
            sh.a.fill(R(1.), BLOOD); K.fill([R(1.10), R(.84)], BLACK, evenodd=True); K.fill(P(spl(np.array([(c[0], c[1] + .042), (c[0] + .011, c[1]), (c[0], c[1] - .030), (c[0] - .011, c[1])]), True, (0, 2), n=5)), BLACK); K.fill(P(np.vstack([up, (up - (0, .010))[::-1]])), BLACK); K.unclip(); sh.a.unclip()
            B(P(lo[int(len(lo) * .25):]), [.3, 1.8, 1.8, .6], seed=sd + 3, smooth_=False)
        ov = np.vstack([E['a'] + (-q * .035, -.004), E['up'], E['b'] + (q * .05, .034)])
        B(P(spl(ov, n=8)), [.3, 3.0, 4.2 if E['o'] > .05 else 3.6, 4.0, 2.8, .3], seed=sd, smooth_=False)
        feather(K, P(lo[int(len(lo) * .35):]), P(lo[int(len(lo) * .35):] + (q * .02, -.055)), gap, 1.5 * lw, seed=sd + 5, reach=(.2, .95), group=4, hole=.5)

    def _brow_poly(self, q):
        bi, bo = self.e['brow']; a = np.array([.078, .030 + bi])
        top = spl(np.array([a + (-.004, .046), (.160, .082 + bi), (.290, .128 + bi * .6 + bo * .2), (.420, .168 + bo * .7), (.520, .190 + bo), (.640, .192 + bo * 1.1)]), n=8)
        bot = spl(np.array([(.640, .192 + bo * 1.1), (.510, .164 + bo), (.405, .136 + bo * .7), (.280, .086 + bi * .6 + bo * .2), (.165, .026 + bi), a]), n=8)
        return np.vstack([top, bot]) * [q, 1]

    def _features_ink(self, sh, dark, gap, face):
        K = sh.k; lw = self.lw; sd = self.seed; P = self.P; F = self.C; H = self.H
        B = lambda Pt, ws, **k: brush(K, Pt, np.asarray(ws, float) * lw * self.bg, **k)
        # marks first (tattoo: skin hatch crosses them)
        mk, pet = self.marks_screen()
        if self.marks > 0:
            K.clip(face)
            c0_ = self.H.P3(np.array([0, -.2, .5])); rt_ = self.rt / (np.hypot(*self.rt) + 1e-6)
            for nm, m in mk:
                if self.noir and nm == 'cheek' and (m.mean(0) - c0_) @ rt_ * dark > .1 * self.s: sh.g.clip(face); sh.g.fill(m, self.ac['mark'], self.marks); sh.g.unclip()
                else: K.fill(m, BLACK, self.marks)
            K.unclip()
        self.petals = pet
        # nose
        yw = self.yaw
        if abs(yw) < 1.3:
            if abs(yw) > .16:
                pr = H.P3(H.nose_profile()); far = 1 if yw > 0 else -1
                B(spl(pr[1:6], n=6), [.3, 1.6, 2.6, 3.6, 4.0], seed=sd + 30, smooth_=False, tremor=.1)
                B(spl(pr[4:7], n=6), [3.4, 3.0, .8], seed=sd + 31, smooth_=False)
                q = -far
                B(F([(q * .046, -.458), (q * .088, -.470), (q * .126, -.440), (q * .130, -.395), (q * .112, -.360)], dz=.03), [2.4, 1.4, 2.8, 2.4, .3], seed=sd + 32)
                K.fill(F([(q * .012, -.466), (q * .050, -.452), (q * .086, -.474), (q * .046, -.486)], closed=True, dz=.06), BLACK)
            else:
                B(F([(dark * .030, -.30), (dark * .046, -.36), (dark * .052, -.405)], dz=.04), [.3, 2.2, 1.2], seed=sd + 30)
                B(F([(-.030, -.478), (0, -.490), (.030, -.478)], dz=.06), [.4, 2.4, .4], seed=sd + 31)
                for q in (-1, 1):
                    K.fill(F([(q * .038, -.462), (q * .062, -.452), (q * .088, -.462), (q * .060, -.478)], closed=True, dz=.04), BLACK)
                    B(F([(q * .100, -.470), (q * .128, -.440), (q * .124, -.395)], dz=.02), [.6, 2.4, .3], seed=sd + 32 + q)
        for q in (-1, 1):
            self._eye_ink(sh, q, dark, gap); self._eye2_ink(sh, q, gap)
            if self.vis(q * .34, .22) > .05:
                K.fill(P(self._brow_poly(q)), BLACK)
                bp = self._brow_poly(q); n = len(bp) // 2; feather(K, P(bp[int(n * .02):int(n * .5)]), P(bp[int(n * .02):int(n * .5)] + (q * .05, .045)), gap * .7, 1.8 * lw, seed=sd + 100 + q, reach=(.2, .9), group=4, back=.3)
        # frown creases between the brows
        if self.e['brow'][0] < -.015:
            for q in (-1, 1): B(F([(q * .036, .19), (q * .052, .13), (q * .044, .07)]), [.3, 2.2, .3], seed=sd + 110 + q)
        self._mouth_ink(sh, dark, gap)
