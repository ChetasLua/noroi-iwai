"""Mahoraga construction: skeleton, poses and the sculpt (used by chars/mahoraga.py).  Character space: +X = his LEFT, -Y = the way
he faces, +Z up, metres, 3.2 m tall.  Pose targets are in character space; `place` puts him in the world."""
import numpy as np, math
from jj.ink import spl
from .maho_sdf import Sculpt, march

SKIN, HORN, FEATH, RIDGE, TOOTH, MOUTH, CLOTH, WRAP, BLADE, CORD, WHEEL, NAIL = range(12)
# part codes (per pixel, for ink decisions)
P_TORSO, P_ARML, P_ARMR, P_HEAD, P_HORN, P_WING, P_TEETH, P_LEG, P_FOOT, P_BLADE, P_ORN, P_WHEEL, P_SASH = range(13)


def Rx(a): c, s = math.cos(a), math.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def Ry(a): c, s = math.cos(a), math.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def Rz(a): c, s = math.cos(a), math.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
def ypr(v): return Rz(v[0]) @ Rx(v[1]) @ Ry(v[2])          # yaw (to his left +), pitch (forward +), roll (toward his left +)
def unit(v): v = np.asarray(v, float); return v / (np.linalg.norm(v) + 1e-12)
def V(*a): return np.array(a, float)


def lerp_pose(a, b, t):
    if isinstance(a, dict):
        out = {}
        for k in set(a) | set(b): out[k] = lerp_pose(a[k], b[k], t) if k in a and k in b else a.get(k, b.get(k))
        return out
    if a is None or b is None or isinstance(a, (str, bool)): return b if t >= .5 else a
    return np.asarray(a, float) * (1 - t) + np.asarray(b, float) * t


def merge(a, **kw):
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in a.items()}
    for k, v in kw.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict): out[k] = merge(out[k], **v)
        else: out[k] = v
    return out


REST = dict(
    root=(0, 0, 1.60), pelvis=(0, .04, 0), waist=(0, -.02, 0), chest=(0, .10, 0), neck=(0, .30, 0), head=(0, -.18, 0), shl=(0, 0, 0), shr=(0, 0, 0),
    lh=dict(t=(.80, -.10, 1.24), pole=(.9, .8, 1.7), fwd=(.05, -.15, -1), dor=(1, .1, 0), curl=.35, spread=.25, thumb=.3),
    rh=dict(t=(-.80, -.10, 1.24), pole=(-.9, .8, 1.7), fwd=(-.05, -.15, -1), dor=(-1, .1, 0), curl=.9, spread=.1, thumb=.5),
    lf=dict(t=(.30, .02, .12), pole=(.36, -2, .9), fwd=(.25, -1, 0), pitch=0.), rf=dict(t=(-.30, .02, .12), pole=(-.36, -2, .9), fwd=(-.25, -1, 0), pitch=0.),
    jaw=0.06, wing=0.0, spin=0.0, wtilt=(0, 0, 0), woff=(0, 0, 0), tail=0.0, cut=None, sword=True, breath=0.0, crest=(0, 0, 0), fan=1.0, ruffle=0.0, spread=0.0, flare=(0, 0, 0), sash=(0, 0, 0))


class Maho:
    """Maho(pose, pos=(x,y,z), heading=rad, scale=1).  .S packed sculpt, .J joints (world), .wheel (centre, axes), .curves (named 3-D ink curves)."""
    def __init__(self, pose=None, pos=(0, 0, 0), heading=0., scale=1., wheel=True, lod=0):
        self.pose = p = merge(REST, **(pose or {}))
        self.Rw = Rz(heading) * scale; self.Ro = Rz(heading); self.tw = np.asarray(pos, float); self.sc = scale
        self.s = Sculpt(); self.curves = []; self.J = {}; self.rk = 1.; self.hs = 1.16; self.lod = lod
        self.with_wheel = wheel
        self._spine(); self.rk = 1.09; self._torso(); self.rk = self.rk_arm = 1.14
        for side in (1, -1): self._arm(side)
        self.rk = 1.
        for side in (1, -1): self._leg(side)
        self._hips(); self.rk = self.hs; self._head(); self.rk = 1.; self._ornament()
        if wheel: self.rk = self.hs; self._wheel(); self.rk = 1.
        self.S = self.s.pack(); self.np_ = len(self.S['T'])

    # ---- helpers: everything is authored in char space then placed
    def w(self, p): return np.asarray(p, float) @ self.Rw.T + self.tw
    def wd(self, d): return np.asarray(d, float) @ self.Ro.T
    def wM(self, M): return np.asarray(M, float) @ self.Ro.T
    def cone(self, a, b, r1, r2, **k): z = self.sc * self.rk; k['k'] = k.get('k', .04) * z; return self.s.cone(self.w(a), self.w(b), r1 * z, r2 * z, **k)
    def ell(self, c, r, M=None, **k): z = self.sc * self.rk; k['k'] = k.get('k', .04) * z; return self.s.ell(self.w(c), np.asarray(r, float) * z, self.wM(np.eye(3) if M is None else M), **k)
    def box(self, c, h, M=None, r=.01, **k): z = self.sc * self.rk; k['k'] = k.get('k', 0.) * z; return self.s.box(self.w(c), np.asarray(h, float) * z, self.wM(np.eye(3) if M is None else M), r * z, **k)
    def tor(self, c, R, r, M=None, **k): z = self.sc * self.rk; return self.s.tor(self.w(c), R * z, r * z, self.wM(np.eye(3) if M is None else M), **k)
    def mus(self, a, b, w, t, n, **k): z = self.sc * self.rk; k['k'] = k.get('k', .045) * z; return self.s.muscle(self.w(a), self.w(b), w * z, t * z, self.wd(n), **k)
    def fea(self, c, L, w, t, M, bend=0., **k): z = self.sc * self.rk; return self.s.feather(self.w(c), L * z, w * z, t * z, self.wM(M), bend / z, **k)
    def chain(self, pts, rad, **k):
        pts = np.asarray(pts, float); rad = np.broadcast_to(rad, len(pts))
        for i in range(len(pts) - 1): self.cone(pts[i], pts[i + 1], rad[i], rad[i + 1], **k)
    def curve(self, kind, pts, w=1., nrm=None, **k): self.curves.append(dict(kind=kind, p=self.w(pts), w=w, n=None if nrm is None else self.wd(nrm), **k))

    # ---- skeleton
    def _spine(self):
        p = self.pose
        R0 = ypr(p['pelvis']); o0 = np.asarray(p['root'], float)
        self.fp = (R0, o0)
        Rw = R0 @ ypr(p['waist']); ow = o0 + R0 @ V(0, 0, .16); self.fw = (Rw, ow)
        Rc = Rw @ ypr(p['chest']); oc = ow + Rw @ V(0, .0, .30); self.fc = (Rc, oc)
        Rn = Rc @ ypr(p['neck']); on = oc + Rc @ V(0, .05, .50); self.fn = (Rn, on)
        Rh = Rn @ ypr(p['head']); oh = on + Rn @ V(0, -.115, .165); self.fh = (Rh, oh)
        self.J.update(root=self.w(o0), chest=self.w(oc), neck=self.w(on), head=self.w(oh))

    def C(self, x, y, z): return self.fc[1] + self.fc[0] @ V(x * 1.13, y * 1.08, z)
    def Wp(self, x, y, z): return self.fw[1] + self.fw[0] @ V(x * 1.07, y * 1.05, z)
    def Pp(self, x, y, z): return self.fp[1] + self.fp[0] @ V(x, y, z)
    def Hd(self, x, y, z): return self.fh[1] + self.fh[0] @ V(x, y, z) * self.hs
    def Nk(self, x, y, z): return self.fn[1] + self.fn[0] @ V(x, y, z)

    @staticmethod
    def ik(S, T, l1, l2, pole):
        d = T - S; L = np.linalg.norm(d); Lm = (l1 + l2) * .998
        if L > Lm: T = S + d / L * Lm; L = Lm
        d = (T - S) / L
        a = (l1 * l1 - l2 * l2 + L * L) / (2 * L); h = math.sqrt(max(l1 * l1 - a * a, 0.))
        e = np.asarray(pole, float) - S; e = e - d * (e @ d); e = unit(e)
        return S + d * a + e * h, T, e

    # ---- torso
    def _torso(self):
        C, W = self.C, self.Wp; Rc, Rw = self.fc[0], self.fw[0]; br = self.pose['breath']
        g = dict(g=0, mat=SKIN, part=P_TORSO)
        self.ell(C(0, .0, .12), (.345, .235 + .012 * br, .40), Rc.T, k=.05, **g)
        self.ell(C(0, .06, .31), (.41, .19, .19), Rc.T, k=.06, **g)
        self.ell(W(0, .015, .06), (.232, .165, .32), Rw.T, k=.06, **g)
        self.ell(self.Pp(0, .01, .05), (.27, .19, .2), self.fp[0].T, k=.06, **g)
        # neck
        self.cone(C(0, .07, .46), self.Hd(0, .045, .02), .135, .098, k=.05, **g)
        self.cone(C(0, -.08, .46), self.Hd(0, .012, -.045), .042, .045, k=.03, **g)
        for s in (1, -1):
            sh = self.pose['shl' if s > 0 else 'shr']; so = lambda x, y, z: C(x * s + sh[0] * s, y + sh[1], z + sh[2])
            c = lambda x, y, z: C(x * s, y, z); w = lambda x, y, z: W(x * s, y, z); n = lambda x, y, z: Rc @ V(x * s, y, z)
            # traps + sternomastoid + clavicle
            self.mus(self.Hd(.05 * s, .07, .02), so(.47, .05, .475), .075, .095, n(0, .4, 1), k=.05, ext=.05, **g)
            self.mus(c(.05, .14, .66), so(.30, .16, .38), .13, .10, n(.2, 1, .5), k=.05, **g)
            self.cone(c(.028, -.175, .405), self.Hd(.082 * s, .045, .0), .030, .030, k=.022, **g)
            self.cone(so(.36, .03, .46), self.Hd(.07 * s, .06, -.01), .035, .035, k=.05, **g)
            self.chain([c(.03, -.172, .405), so(.12, -.172, .425), so(.22, -.150, .452), so(.34, -.085, .475), so(.45, -.035, .490)], (.020, .019, .018, .020, .026), k=.02, **g)
            # back
            self.mus(so(.42, .07, .30), w(.10, .13, -.12), .19, .085, n(.75, .6, 0), k=.05, **g)
            self.ell(so(.20, .165, .28), (.17, .07, .19), Rc.T, k=.04, **g)
            self.cone(so(.12, .215, .36), so(.42, .10, .47), .028, .035, k=.03, **g)
            self.cone(c(.065, .18, .25), w(.07, .15, -.05), .06, .06, k=.04, **g)
            # serratus + ribs
            for i in range(4):
                a = math.radians(-38 - 5 * i); z = .06 - .078 * i; rx = .345 * math.sqrt(max(0, 1 - ((z - .12) / .40) ** 2)); ry = .235 * math.sqrt(max(0, 1 - ((z - .12) / .40) ** 2))
                p0 = V((rx + .004) * math.cos(a) * s, (ry + .004) * math.sin(a), z); d = unit(V(-.35 * s, -.75, -.55))
                self.mus(self.fc[1] + Rc @ ((p0 - d * .095) * V(1.11, 1.08, 1)), self.fc[1] + Rc @ ((p0 + d * .040) * V(1.11, 1.08, 1)), .036, .022, Rc @ V(math.cos(a) * s, math.sin(a), .1), k=.012, **g)
            for i in range(5):
                z0 = -.045 - .062 * i; pts = []
                for a in np.linspace(math.radians(12), math.radians(-84 + 9 * i), 7):
                    z = z0 - .075 * math.sin(max(0, -a)) + .05 * math.cos(a) * 0
                    q = math.sqrt(max(.02, 1 - ((z - .12) / .40) ** 2)); pts.append(c(.345 * q * math.cos(a), .237 * q * math.sin(a), z))
                self.chain(pts, np.linspace(.021, .011, 7), k=.018, **g)
            arch = [c(.035, -.222, .045), c(.12, -.212, -.05), c(.20, -.165, -.17), c(.235, -.10, -.27)]
            self.chain(arch, (.024, .026, .022, .016), k=.02, **g)
            # abs, obliques
            for i, (z, hh, hw) in enumerate(((.235, .052, .074), (.125, .056, .077), (.012, .056, .074), (-.125, .082, .066))):
                y = -.168 - .018 * math.sin(math.pi * (z + .2) / .5)
                self.ell(w(.073, y, z), (hw, .042, hh), (Rw @ Ry(.10 * s)).T, k=.012, **g)
            self.mus(w(.265, -.035, .27), w(.215, -.09, -.16), .105, .062, Rw @ V(s, -.35, 0), k=.03, **g)
            self.mus(w(.19, -.125, .20), w(.165, -.135, -.18), .055, .035, Rw @ V(.5 * s, -1, 0), k=.02, **g)
            # pecs (last: their lower edge stays crisp)
            tip = self._shoulder(s)
            self.mus(c(.010, -.218, .140), tip + Rc @ V(-.01 * s, -.075, -.11), .150, .112, n(.22, -1, .05), k=.014, ext=.04, **g)
            self.mus(c(.030, -.196, .285), tip + Rc @ V(0, -.075, -.03), .092, .092, n(.2, -1, .5), k=.022, ext=.02, **g)

    def _shoulder(self, s):
        sh = self.pose['shl' if s > 0 else 'shr']
        return self.C(.50 * s + sh[0] * s, .015 + sh[1], .405 + sh[2])

    # ---- arms
    def _arm(self, s):
        p = self.pose['lh' if s > 0 else 'rh']; Rc = self.fc[0]; S = self._shoulder(s); part = P_ARML if s > 0 else P_ARMR
        l1, l2 = .59, .53; T = np.asarray(p['t'], float)
        E, T, e = self.ik(S, T, l1, l2, p['pole'])
        z = unit(E - S); y = e - z * (e @ z)
        mid = E - (S + T) / 2
        if np.linalg.norm(mid) > .04: y = mid - z * (mid @ z)
        y = unit(y); x = np.cross(y, z) * 1.; x = x if x @ (Rc @ V(s, 0, 0)) > 0 or s == 0 else x
        x = np.cross(y, z) if s < 0 else np.cross(z, y); x = -x        # outward
        U = lambda a, b, c: S + x * a + y * b + z * c
        zf = unit(T - E); yf = y - zf * (y @ zf)
        yf = unit(yf if np.linalg.norm(yf) > .15 else -z - zf * (-z @ zf)); xf = unit(np.cross(zf, yf)); xf = xf if xf @ x > 0 else -xf
        # forearm twist toward the hand's dorsal
        dor = np.asarray(p['dor'], float); fw = unit(p['fwd']); dor = unit(dor - fw * (dor @ fw))
        F = lambda a, b, c: E + xf * a + yf * b + zf * c
        self.J['S%d' % s], self.J['E%d' % s], self.J['W%d' % s] = self.w(S), self.w(E), self.w(T)
        cut = self.pose['cut'] if s < 0 else None             # None or fraction 0..1 along the forearm/upper arm kept
        gt = dict(g=0, mat=SKIN, part=P_TORSO); ga = dict(g=20 if s > 0 else 21, mat=SKIN, part=part)
        # deltoid lives in the torso group
        self.mus(U(.052, 0., -.150), U(.056, .0, .330), .122, .108, x, k=.022, ext=.0, **gt)
        self.mus(S + Rc @ V(-.10 * s, -.065, .055), U(.050, -.040, .29), .086, .090, unit(x - y * .9), k=.016, ext=.04, **gt)
        self.mus(S + Rc @ V(-.10 * s, .10, .055), U(.048, .045, .29), .088, .090, unit(x + y * .9), k=.016, ext=.04, **gt)
        self.ell(S, (.12, .115, .12), k=.05, **gt)
        if cut is not None and cut < .42: 
            self.cutp = (U(0, 0, .30), z)
            self.cone(S, U(0, 0, .30), .085, .078, k=.03, **ga); self.mus(U(-.005, -.052, .05), U(0, -.045, .33), .08, .07, -y, k=.02, **ga); self.mus(U(-.01, .06, .05), U(0, .06, .33), .08, .07, y, k=.02, **ga)
            self.ell(U(0, 0, .325), (.082, .082, .02), np.stack([x, y, z]), k=.0, g=22, mat=MOUTH, part=part)
            return
        self.cone(S, E, .078, .064, k=.03, **ga)
        self.mus(U(-.008, -.058, .07), U(-.006, -.046, .575), .086, .080, -y, k=.02, ext=.0, **ga)
        self.mus(U(.052, -.018, .24), U(.034, -.02, .60), .048, .052, x, k=.02, **ga)
        self.mus(U(-.035, .062, .05), U(-.015, .070, .53), .076, .070, y, k=.02, **ga)
        self.mus(U(.058, .052, .10), U(.026, .072, .50), .062, .062, unit(x + y), k=.02, **ga)
        self.ell(E + y * .008, (.060, .060, .06), k=.03, **ga); self.ell(E + y * .052 - z * .0, (.030, .03, .034), k=.02, **ga)
        # forearm
        self.cone(E, T, .062, .043, k=.03, **ga)
        self.mus(F(.052, -.03, -.10), F(.022, -.034, .32), .060, .058, xf, k=.02, **ga)
        self.mus(F(.052, .035, -.03), F(.012, .03, .38), .052, .054, unit(xf + yf), k=.02, **ga)
        self.mus(F(-.055, -.01, -.04), F(-.016, -.016, .38), .062, .060, -xf, k=.02, **ga)
        self.mus(F(0, -.045, .0), F(.0, -.030, .40), .05, .045, -yf, k=.02, **ga)
        self.cone(F(0, .055, .0), F(.0, .036, .45), .024, .020, k=.02, **ga)
        self.ell(T, (.046, .036, .05), np.stack([np.cross(dor, fw), dor, fw]), k=.03, **ga)
        tside = np.cross(fw, dor) * (1 if s > 0 else -1) * -1
        self._hand(T, fw, dor, tside, p, ga, s)
        if s < 0 and self.pose['sword']:
            ax = zf; nr = yf if True else yf; M = np.stack([ax, np.cross(nr, ax), nr])
            org = F(.012, .085, -.10)
            zz = self.sc * self.rk
            self.s.blade(self.w(org), self.wM(M), 1.50 * zz, .078 * zz, .020 * zz, g=9, mat=BLADE, part=P_BLADE, curve=0.02 / zz, tip=.34)
            self.box(F(.012, .085, -.11), (.05, .03, .03), np.stack([xf, yf, zf]), r=.012, g=9, mat=CORD, part=P_BLADE)
            self.blade_line = (self.w(org), self.wd(ax), 1.50 * zz); self.blade_c = .02 / zz
            self.blade_M = self.wM(M)
            for u, r, t in ((.07, .098, .1), (.12, .100, -.16), (.165, .096, .12), (.30, .080, -.12), (.345, .076, .15), (.39, .072, -.1)):
                M2 = np.stack([xf, yf * math.cos(t) + zf * math.sin(t), zf * math.cos(t) - yf * math.sin(t)])
                self.tor(F(.004, .026, u), r, .020, M2, g=10, mat=WRAP, part=P_BLADE, zs=.72)
            # loose ends of the binding
            self.chain([F(.05, .10, .30), F(.12, .13, .36), F(.15, .12, .50), F(.20, .16, .62)], (.014, .012, .012, .004), g=10, mat=WRAP, part=P_BLADE, k=.01)

    def _hand(self, T, fw, dor, th, p, ga, s):
        M = np.stack([th, dor, fw]); H = lambda a, b, c: T + th * a + dor * b + fw * c
        sz = 1.12
        self.box(H(0, 0, .078 * sz), (.056 * sz, .022 * sz, .066 * sz), M, r=.018, k=.022, **ga)
        self.ell(H(.036 * sz, -.02 * sz, .05 * sz), (.026 * sz, .024 * sz, .048 * sz), M, k=.02, **ga); self.ell(H(-.04 * sz, -.016 * sz, .055 * sz), (.02 * sz, .02 * sz, .05 * sz), M, k=.02, **ga)
        curl = np.broadcast_to(np.asarray(p['curl'], float), (4,)); sp = float(p['spread'])
        for i, (x, L, r) in enumerate(((.044, .135, .0185), (.015, .150, .0190), (-.015, .140, .0180), (-.043, .110, .0160))):
            b = H(x * sz, .004 * sz, (.138 - .010 * abs(i - 1.2)) * sz); a = sp * (1.2 - i) * .34
            d = unit(fw * math.cos(a) + th * math.sin(a)); ax = unit(np.cross(d, dor))         # bend axis
            c = float(curl[i]); ang = 0.; q = b; nd = dor
            self.ell(b, (r * 1.18, r * 1.12, r * 1.25), M, k=.012, **ga)
            for j, (fr, bend, r1, r2) in enumerate(((.44, .95, 1.0, .92), (.32, 1.25, .95, .84), (.24, .85, .84, .62))):
                ang += c * bend; dd = d * math.cos(ang) - dor * math.sin(ang); e = q + dd * L * sz * fr
                self.cone(q, e, r * r1 * sz, r * r2 * sz, k=.006, **ga)
                if j < 2: self.ell(e + (dor * math.cos(ang) + d * math.sin(ang)) * r * .18, (r * 1.04 * sz,) * 3, k=.006, **ga)
                q = e
            n_ = dor * math.cos(ang) + d * math.sin(ang)
            self.cone(q - dd * .012 + n_ * r * .25, q + dd * .030 - n_ * .004, r * .52, r * .10, g=ga['g'] + 10, mat=NAIL, part=ga['part'], k=0.)
        tc = float(p['thumb'])
        b = H(.050 * sz, -.010 * sz, .030 * sz); d1 = unit(th * .75 + fw * .72 - dor * (.15 + .35 * tc)); m = b + d1 * .062 * sz
        d2 = unit(d1 * .75 + fw * .55 - dor * (.35 * tc) - th * .55 * tc); m2 = m + d2 * .058 * sz; d3 = unit(d2 + fw * .1 - th * .5 * tc - dor * .2 * tc); m3 = m2 + d3 * .044 * sz
        self.cone(b, m, .027 * sz, .022 * sz, k=.02, **ga); self.cone(m, m2, .0215 * sz, .0195 * sz, k=.006, **ga); self.ell(m, (.023 * sz,) * 3, k=.008, **ga)
        self.cone(m2, m3, .0195 * sz, .014 * sz, k=.006, **ga); self.cone(m3 - d3 * .01 + dor * .004, m3 + d3 * .03, .010, .002, g=ga['g'] + 10, mat=NAIL, part=ga['part'], k=0.)

    # ---- legs, cloth
    def _leg(self, s):
        p = self.pose['lf' if s > 0 else 'rf']; Hp = self.Pp(.165 * s, .0, -.03); A = np.asarray(p['t'], float)
        l1, l2 = .74, .74
        K, A, e = self.ik(Hp, A, l1, l2, p['pole'])
        z = unit(K - Hp); f = e - z * (e @ z); f = unit(f); x = unit(np.cross(f, z)); x = x if x @ (self.fp[0] @ V(s, 0, 0)) > 0 else -x
        zs = unit(A - K); fs = unit(f - zs * (f @ zs)); xs = unit(np.cross(fs, zs)); xs = xs if xs @ x > 0 else -xs
        self.J['K%d' % s], self.J['A%d' % s] = self.w(K), self.w(A)
        g = dict(g=6, mat=CLOTH, part=P_LEG)
        T = lambda a, b, c: Hp + x * a + f * b + z * c
        S = lambda a, b, c: K + xs * a + fs * b + zs * c
        self.cone(T(.035, 0, -.02), T(.03, .0, .70), .232, .188, k=.06, pleat=(.024, 3., 1.3 * s, 0.), **g)
        self.mus(T(.06, .10, .05), T(.02, .075, .74), .11, .09, f, k=.04, **g)
        self.ell(K + f * .045, (.105, .105, .12), k=.05, **g)
        # gathered lower leg, blousing over the binding
        fl = np.asarray(self.pose['flare'], float)
        S0 = S; S = lambda a, b, c: S0(a, b, c) + fl * max(0., math.sin(math.pi * min(1., c / .56))) if c < .56 else S0(a, b, c)
        self.cone(K, S(0, -.05, .40), .172, .196, k=.05, pleat=(.034, 4., .4 + s, 0.), **g)
        self.cone(S(0, -.045, .40), S(0, -.01, .53), .190, .085, k=.03, pleat=(.026, 5., 2. + s, 0.), **g)
        # bold folds: two pipes off the hip, a zig-zag behind the knee, one drop over the binding
        for (a0, u0, a1, u1, r0) in ((.75, .02, .15, .60, .046), (-.55, .0, -.95, .58, .042), (2.3, .05, 1.5, .55, .044)):
            P_ = [T(math.sin(a0 + (a1 - a0) * t) * (.215 - .045 * t), math.cos(a0 + (a1 - a0) * t) * (.215 - .045 * t), u0 + (u1 - u0) * t) for t in (0, .33, .66, 1.)]; self.chain(P_, (r0 * .35, r0, r0 * .9, r0 * .35), k=.02, **g)
        for (a0, u0, a1, u1, r0) in ((.9, .03, -.5, .20, .042), (-.8, .12, .55, .33, .046), (2.6, .05, 1.2, .38, .042)):
            P_ = [S(math.sin(a0 + (a1 - a0) * t) * .170, math.cos(a0 + (a1 - a0) * t) * .170 - .03, u0 + (u1 - u0) * t) for t in (0, .33, .66, 1.)]; self.chain(P_, (r0 * .35, r0, r0, r0 * .4), k=.02, **g)
        # shin: binding + skin
        gs = dict(g=8, mat=SKIN, part=P_FOOT)
        self.cone(S(0, -.01, .50), A, .062, .048, k=.03, **gs); self.mus(S(0, -.055, .42), S(0, -.035, .70), .055, .05, -fs, k=.03, **gs)
        for i in range(5):
            u = .50 + .036 * i; t = .22 * (-1) ** i; M = np.stack([xs, fs * math.cos(t) + zs * math.sin(t), zs * math.cos(t) - fs * math.sin(t)])
            self.tor(S(0, -.016, u), .066 - .0035 * i, .020, M, g=10, mat=WRAP, part=P_FOOT, zs=.8)
        # foot
        fw = unit(p['fwd']); up = V(0, 0, 1.); up = unit(up - fw * (up @ fw)); pt = p['pitch']; ff = fw * math.cos(pt) - up * math.sin(pt); uu = up * math.cos(pt) + fw * math.sin(pt); rt = np.cross(ff, uu) * (-1 if s > 0 else 1)
        rt = rt if rt @ (self.fp[0] @ V(s, 0, 0)) > 0 else -rt         # outer side
        Ft = lambda a, b, c: A + rt * a + ff * b + uu * c
        self.ell(A, (.055, .06, .06), k=.03, **gs)
        self.ell(Ft(0, -.065, -.055), (.052, .062, .055), np.stack([rt, ff, uu]), k=.04, **gs)
        self.cone(Ft(0, -.02, -.03), Ft(-.012, .18, -.075), .062, .042, k=.03, **gs); self.cone(Ft(.03, -.03, -.065), Ft(.045, .17, -.085), .04, .034, k=.03, **gs)
        self.cone(Ft(-.04, .17, -.085), Ft(.05, .16, -.088), .036, .030, k=.03, **gs)
        bend = max(0., pt) * .8; tf = ff * math.cos(bend) + uu * math.sin(bend)
        for i, (a, b0, L, r) in enumerate(((-.052, .185, .105, .027), (-.018, .195, .092, .0205), (.012, .188, .084, .0195), (.038, .176, .074, .0185), (.062, .160, .064, .0175))):
            q = Ft(a, b0, -.088 + r * .2); m = q + tf * L * .55 + uu * .016; e2 = q + tf * L - uu * .006
            self.cone(q, m, r * 1.05, r, k=.010, **gs); self.cone(m, e2, r, r * .86, k=.006, **gs)
            self.cone(e2 - tf * .01 + uu * r * .4, e2 + tf * .030 - uu * .012, r * .5, r * .1, g=18, mat=NAIL, part=P_FOOT, k=0.)

    def _hips(self):
        P = self.Pp; R = self.fp[0]; g = dict(g=6, mat=CLOTH, part=P_LEG)
        self.ell(P(0, .01, -.02), (.265, .20, .17), R.T, k=.06, **g)
        self.cone(P(0, -.04, -.05), P(0, -.02, -.22), .14, .10, k=.08, **g)
        gs = dict(g=7, mat=WRAP, part=P_SASH)
        for z0, rr, tl in ((.095, .038, .03), (.150, .040, -.03), (.205, .034, .02)):
            a = np.linspace(0, 2 * math.pi, 19)
            pts = [P(.292 * math.cos(t), .01 + .222 * math.sin(t), z0 + tl * math.sin(t + .8) + (.02 * math.cos(t * 2))) for t in a]
            self.chain(pts, rr, k=.008, **gs)
        # knot + hanging tails
        self.ell(P(.10, -.225, .15), (.055, .045, .05), R.T, k=.01, **gs); self.ell(P(.155, -.212, .165), (.05, .035, .04), (R @ Ry(.5)).T, k=.01, **gs); self.ell(P(.055, -.228, .175), (.048, .035, .038), (R @ Ry(-.4)).T, k=.01, **gs)
        sw = float(self.pose.get('sway', 0.))
        for dx, L, q in ((.085, .62, .0), (.135, .50, .5)):
            pts = [P(dx, -.245, .13)]; 
            for j in range(1, 6):
                t = j / 5; pts.append(P(dx, -.245, .13) + V(0, 0, -1) * L * t * (1 - .55 * min(1., np.linalg.norm(self.pose['sash'])) * t) + np.asarray(self.pose['sash'], float) * (t ** 1.3 * L * 1.25) * (1 + .25 * math.sin(6 * t + 4 * q)) + V(0, 0, 1) * (.10 * math.sin(7 * t + 5 * q) * np.linalg.norm(self.pose['sash'])) + R @ V(1, 0, 0) * (.05 * math.sin(2.2 * t + q) + sw * t * t * .35) + R @ V(0, -1, 0) * (.02 + .05 * t))
            self.chain(pts, (.020, .030, .034, .036, .038, .030), k=.01, **gs)

    # ---- head
    def _head(self):
        H = self.Hd; R = self.fh[0]; p = self.pose; g = dict(g=1, mat=SKIN, part=P_HEAD)
        n = lambda *v: R @ V(*v)
        jw = float(p['jaw']); hinge = V(0, .015, .005); Rj = Rx(jw)
        slide = V(0, -.018 * jw, -.030 * jw)
        Jn = lambda x, y, z: self.fh[1] + R @ (hinge + slide + Rj @ (V(x, y, z) - hinge)) * self.hs
        self.jawf = Jn
        self.ell(H(0, .04, .165), (.108, .145, .132), R.T, k=.04, **g)
        self.chain([H(0, -.098, .115), H(0, -.070, .215), H(.01, -.030, .315), H(.055, .060, .42)], (.036, .044, .050, .03), k=.022, **g)
        for s in (1, -1):
            h = lambda x, y, z: H(x * s, y, z); j = lambda x, y, z: Jn(x * s, y, z)
            self.chain([h(.048, -.086, .115), h(.060, -.045, .215), H(.062 * s + .012, -.0, .31), H(.060 * s + .065, .085, .42)], (.022, .030, .035, .02), k=.013, **g)
            self.chain([h(.086, -.045, .12), h(.094, .0, .205), H(.098 * s + .015, .055, .31), H(.098 * s + .075, .13, .42)], (.024, .032, .034, .02), k=.016, **g)
            # brow mass the upper wing grows out of, cheekbone, socket wall
            self.ell(h(.062, -.100, .088), (.042, .036, .026), (R @ Ry(-.25 * s)).T, k=.02, **g)
            self.ell(h(.080, -.078, .028), (.036, .048, .024), (R @ Ry(.3 * s) @ Rz(-.5 * s)).T, k=.02, **g)
            self.ell(h(.040, -.128, .040), (.022, .022, .030), R.T, k=.012, **g)
        # nose: a blunt bony ridge with no nostrils
        self.cone(H(0, -.128, .105), H(0, -.160, .030), .021, .025, k=.016, **g); self.ell(H(0, -.160, .018), (.030, .020, .016), R.T, k=.010, **g)
        # upper jaw
        self.ell(H(0, -.075, .0), (.088, .085, .040), R.T, k=.03, **g)
        self.ell(H(0, -.0, .01), (.095, .10, .06), R.T, k=.04, **g)
        # mouth cavity
        op = 1 + 1.9 * jw
        self.ell(H(0, -.140, -.048 - .050 * jw), (.094, .080, .040 * op), R.T, k=.006, sub=True, g=1, mat=MOUTH, part=P_TEETH)
        # lower jaw
        for s in (1, -1):
            j = lambda x, y, z: Jn(x * s, y, z); h = lambda x, y, z: H(x * s, y, z)
            self.cone(h(.094, .025, .015), j(.090, -.015, -.085), .026, .024, k=.02, **g)
            self.cone(j(.090, -.015, -.085), j(.040, -.120, -.128), .024, .026, k=.02, **g)
            # cheek straps: the skin is stretched between cheekbone and jaw
            self.cone(h(.094, -.045, .022), j(.091, -.040, -.078), .011, .010, k=.010, **g)
            self.cone(h(.096, -.012, .015), j(.094, -.012, -.070), .020, .018, k=.016, **g)
            # nasolabial fold
            self.chain([h(.034, -.150, .022), h(.070, -.126, .006), h(.094, -.088, -.010), h(.099, -.066, -.030)], (.010, .012, .012, .008), k=.010, **g)
        self.ell(Jn(0, -.132, -.128), (.044, .026, .027), R.T, k=.022, **g)
        self.ell(Jn(0, -.055, -.112), (.074, .085, .030), (R @ Rj).T, k=.03, **g)
        # lips: thin, drawn back
        ph = np.radians(np.linspace(-84, 84, 17))
        ax, by, y0 = .0935, .100, -.062
        up = [H(ax * math.sin(a), y0 - by * math.cos(a) * 1.0 + .003, -.006 + .020 * (abs(a) / 1.46) ** 2.2) for a in ph]
        lo = [Jn(ax * math.sin(a), y0 - by * math.cos(a) * .985 + .006, -.098 + .090 * (abs(a) / 1.46) ** 2.6) for a in ph]
        self.chain(up, .0078, k=.008, **g); self.chain(lo, .0080, k=.008, **g)
        self.lips = (self.w(up), self.w(lo))
        # teeth
        def arch(a, r): return V(.0800 * math.sin(a) * r, -.060 - .0925 * math.cos(a) * r, 0)
        wd = np.array([.0250, .0215, .0220, .0185, .0180, .0190, .0185]); rr = np.random.default_rng(11)
        s_ = np.r_[0, np.cumsum(wd)]; tot = s_[-1]
        fine = np.linspace(0, 1.50, 200); ar = np.array([arch(a, 1) for a in fine]); al = np.r_[0, np.cumsum(np.linalg.norm(np.diff(ar, axis=0), axis=1))]
        for s in (1, -1):
            for i in range(7):
                a = float(np.interp((s_[i] + s_[i + 1]) / 2 * .97, al, fine)); c = arch(a, 1.) * V(s, 1, 1); t = unit(V(math.cos(a) * .080 * s, .0925 * math.sin(a), 0)); nn = unit(V(math.sin(a) * .0925 * s, -.08 * math.cos(a), 0))
                for kind in (0, 1):
                    hh = (.0235, .0225, .0250, .0215, .0205, .0195, .018)[i] * (1 if kind == 0 else .88); ww = wd[i] * .47 * (1 if kind == 0 else (.86, .9, .95, .98, 1, 1, 1)[i])
                    zc = -.030 - .0005 * i + .0105 * (a / 1.4) ** 2 * 1.0 if kind == 0 else -.0745 + .013 * (a / 1.4) ** 2
                    jt = rr.normal(0, .05, 3); t2 = unit(t + V(0, 0, jt[0] * .5)); cc = c * V(1 - .02 * kind, 1, 1) + nn * (.0015 if kind == 0 else -.0035) + V(0, 0, zc + jt[1] * .012)
                    up_ = unit(V(0, 0, 1) + nn * (-.12 if kind == 0 else .10)); Mx = np.stack([t2, np.cross(up_, t2), up_]); Mx[1] /= np.linalg.norm(Mx[1])
                    if kind == 0: self.box(self.Hd(*cc), (ww, .0085, hh), Mx @ R.T, r=.0062, g=5, mat=TOOTH, part=P_TEETH)
                    else: self.box(Jn(*cc), (ww, .008, hh), Mx @ Rj.T @ R.T, r=.0060, g=5, mat=TOOTH, part=P_TEETH)
        # gums, tongue
        for s in (1, -1):
            for i in range(6):
                a = .1 + i * .22; c = arch(a, .965) * V(s, 1, 1)
                self.ell(self.Hd(*(c + V(0, 0, -.010 + .006 * (a / 1.4) ** 2))), (.013, .012, .011), R.T, k=.004, g=30, mat=MOUTH, part=P_TEETH)
                cj = arch(a, .955) * V(s, 1, 1) + V(0, 0, -.095 + .012 * (a / 1.4) ** 2)
                self.ell(Jn(*cj), (.013, .012, .011), (R @ Rj).T, k=.004, g=30, mat=MOUTH, part=P_TEETH)
        self.ell(H(0, -.035, -.045 - .055 * jw), (.074, .060, .040 + .055 * jw), R.T, k=.0, g=31, mat=MOUTH, part=P_TEETH)
        self.ell((H(0, -.02, -.02) + Jn(0, -.04, -.10)) / 2, (.080, .075, .062 + .02 * jw), R.T, k=.0, g=31, mat=MOUTH, part=P_TEETH)
        if jw > .15: self.ell(Jn(0, -.082, -.078), (.040, .066, .016), (R @ Rj).T, k=.01, g=32, mat=MOUTH, part=P_TEETH)
        self._horn(); 
        for s in (1, -1):
            for low in (0, 1): self._wing(s, low)

    def horn_pts(self):
        sw = float(self.pose['tail'])
        P = np.array([(0, .055, .20, .118), (.01, .085, .30, .112), (.055, .13, .395, .122), (.165, .175, .475, .134), (.32, .205, .570, .134), (.455, .21, .73, .126), (.515, .21, .94, .116),
                      (.475, .215, 1.16, .106), (.335, .22, 1.335, .097), (.115, .225, 1.425, .089), (-.135, .225, 1.405, .081), (-.345, .215, 1.285, .072), (-.465, .20, 1.095, .062),
                      (-.480, .175, .895, .050), (-.415, .145, .72, .036), (-.31, .115, .595, .020), (-.19, .09, .525, .003)])
        t = np.linspace(0, 1, len(P)); P[:, 0] += sw * .10 * np.sin(t * 5.0) * t; P[:, 2] += sw * .05 * np.cos(t * 4.) * t
        Q = spl(P[:, :2], n=4); Z = spl(P[:, 2:], n=4); out = np.c_[Q, Z]
        if any(self.pose['crest']):
            n = len(out)
            for i in range(n): k = min(1., max(0., (i / (n - 1) - .02) / .16)); k = k * k * (3 - 2 * k); out[i, :3] = self.CR0 + ypr(np.asarray(self.pose['crest'], float) * k) @ (out[i, :3] - self.CR0)
        return out
    CR0 = V(.01, .07, .30)

    def _horn(self):
        q = self.horn_pts(); pts = np.array([self.Hd(*v[:3]) for v in q])
        self.chain(pts, q[:, 3], k=.03, g=1, mat=HORN, part=P_HORN)
        self.horn_w = self.w(pts); self.horn_r = q[:, 3] * self.sc * self.rk
        # sinew strands that run from the skull into the trunk and die out
        T = np.gradient(q[:, :3], axis=0); T /= np.linalg.norm(T, axis=1)[:, None]; n = len(q); fr = self._frames(q[:, :3])
        for ph, r0, u1 in ((-1.57, .30, .40), (-.75, .30, .36), (-2.4, .28, .34), (.1, .26, .30), (3.1, .28, .28), (1.4, .34, .5), (2.3, .3, .42)):
            m = int(u1 * n); P = []; rd = []
            for i in range(2, m):
                t = (i - 2) / max(1, m - 3); rr = q[i, 3]; a = ph + .55 * t
                P.append(self.Hd(*(q[i, :3] + (fr[i, 0] * math.cos(a) + fr[i, 1] * math.sin(a)) * rr * (.80 - .22 * t)))); rd.append(rr * (r0 + .06 * math.sin(6 * t + ph)) * (1 - .5 * t))
            self.chain(P, rd, k=.02, g=1, mat=HORN, part=P_HORN)

    @staticmethod
    def _frames(P):
        """rotation-minimising frames (n,2,3): frame 0 starts with x = his left, y = back"""
        T = np.gradient(P, axis=0); T /= np.linalg.norm(T, axis=1)[:, None]
        x = V(1, 0, 0); out = []
        for t in T:
            x = x - t * (x @ t); x = unit(x); out.append((x, np.cross(t, x)))
        return np.array(out)

    def wing_spine(self, low, side=1):
        wv = float(self.pose['wing'] if side < 0 or self.pose.get('wingL') is None else self.pose['wingL'])
        if not low:
            P = np.array([(.026, -.112, .108), (.080, -.146, .120), (.165, -.168, .116), (.265, -.155, .120), (.375, -.112, .165), (.485, -.050, .255), (.575, .015, .375), (.625, .065, .480)])
        else:
            P = np.array([(.028, -.140, .066), (.078, -.174, .080), (.145, -.192, .078), (.212, -.190, .034), (.268, -.172, .000), (.362, -.128, .020), (.478, -.066, .088), (.600, -.002, .190)])
        u = np.linspace(0, 1, 8)
        P = P.copy(); P[:, 2] += wv * .22 * u ** 1.5; P[:, 1] += wv * .05 * u
        sp = float(self.pose['spread']) * (1. if not low else .55)
        if sp: c, q = math.cos(sp), math.sin(sp); d = P - P[0]; k = np.clip(u / .35, 0, 1)[:, None]; P = P[0] + d * (1 - k) + k * np.stack([d[:, 0] * c + d[:, 2] * q, d[:, 1], d[:, 2] * c - d[:, 0] * q], 1)
        return spl(P[:, :2], n=5), spl(np.c_[P[:, 2], np.zeros(8)], n=5)[:, 0]

    def _wing(self, s, low):
        xy, z = self.wing_spine(low, s); Q = np.c_[xy, z]; Q[:, 0] *= s; n = len(Q); R = self.fh[0]
        u = np.linspace(0, 1, n)
        rad = np.interp(u, [0, .06, .2, .34, .55, .8, 1], [.020, .031, .034, .036, .029, .018, .006])
        g = 3 if not low else 4
        self.chain([self.Hd(*v) for v in Q], rad, k=.012, g=g, mat=RIDGE, part=P_WING)
        T = np.gradient(Q, axis=0); T /= np.linalg.norm(T, axis=1)[:, None]
        rng = np.random.default_rng(5 + low * 3 + (s > 0))
        fan = float(self.pose['fan'])
        rows = [  # count, u0, u1, length0, length1, width0, width1, lift, back, ang0, ang1
            (11, .10, 1.0, .250, .560, .046, .052, .006, .046, 1.05, .10 + .26 * fan),      # primaries (behind): fan out past the ridge
            (11, .08, .98, .200, .400, .044, .048, .010, .024, 1.06, .16 + .24 * fan),
            (12, .05, .93, .140, .240, .036, .038, .012, .002, 1.06, .26 + .16 * fan),
            (12, .03, .85, .088, .120, .031, .029, .012, -.016, 1.08, .34 + .08 * fan)]     # scale coverts (front)
        if self.lod:                                                            # at distance: one row of long blades, one of broad coverts
            rows = [(7, .13, 1.0, .350, .500, .058, .070, .006, .050, 1.22, .62 + .22 * fan), (6, .07, .92, .170, .260, .056, .060, .012, .004, 1.15, .62 + .10 * fan)]
        fwd = V(0, -1, 0); sc = .80 if low else 1.
        for r, (cnt, u0, u1, L0, L1, w0, w1, lift, back, a0, a1) in enumerate(rows):
            for i in range(cnt):
                t = i / (cnt - 1); uu = u0 + (u1 - u0) * t + rng.normal(0, .008); uu = min(max(uu, 0), 1); k = uu * (n - 1); i0 = min(int(k), n - 2); fr = k - i0
                b = Q[i0] * (1 - fr) + Q[i0 + 1] * fr; tg = unit(T[i0] * (1 - fr) + T[i0 + 1] * fr)
                upv = V(.45 * s * float(self.pose['spread']) * uu, .20, 1.); up = unit(upv - tg * (upv @ tg)); nn = unit(np.cross(tg, up)); nn = nn if nn @ fwd > 0 else -nn
                an = a0 + (a1 - a0) * t ** .75 + rng.normal(0, .05)
                ln = (L0 + (L1 - L0) * t ** 1.2) * rng.uniform(.93, 1.14) * sc; wd = (w0 + (w1 - w0) * t) * rng.uniform(.9, 1.12)
                odd = rng.random() < .16; an += rng.normal(0, .12 if odd else .025) + self.pose['ruffle'] * rng.normal(0, .25); d = unit(up * math.sin(an) + tg * math.cos(an) + nn * (-.14 + .045 * r + rng.normal(0, .06 if odd else .015))); yy = unit(np.cross(nn, d)); nz = np.cross(d, yy)
                roll = -.30 * s; yy, nz = yy * math.cos(roll) + nz * math.sin(roll), nz * math.cos(roll) - yy * math.sin(roll)
                c = b + up * lift + nn * (-back + .004 * rng.normal()) + d * ln * .46
                self.fea(self.Hd(*c), ln * .54, wd, .0062, np.stack([d, yy, nz]) @ R.T, bend=-.30 * (1 if nz @ nn > 0 else -1), g=14 + low, mat=FEATH, part=P_WING)

    # ---- chest ornament: black cord, trefoil ring, small rings, tassels (laid on the real chest surface)
    def _ornament(self):
        C = self.C; Rc = self.fc[0]; g = dict(g=11, mat=CORD, part=P_ORN, k=.004); S = self.s.pack(); cache = {}
        def surf(x, z, lift=.012):
            key = (round(x, 3), round(z, 3))
            if key not in cache:
                o = self.w(C(x, -.75, z)); t = march(S0, o, self.wd(Rc @ V(0, 1, 0)), 1.2 * self.sc)
                cache[key] = -.75 + (t if t > 0 else .55 * self.sc) / self.sc / 1.08
            return C(x, cache[key] - lift, z)
        S0 = {k: v[S['G'] == 0] for k, v in S.items()}
        fr = lambda x, z: np.stack([unit(surf(x + .01, z) - surf(x - .01, z)), unit(surf(x, z + .01) - surf(x, z - .01)), unit(np.cross(surf(x + .01, z) - surf(x - .01, z), surf(x, z + .01) - surf(x, z - .01)))])
        tre = [(-.085, .285), (-.112, .268), (-.112, .236), (-.075, .216), (-.055, .205), (-.040, .172), (0, .160), (.040, .172), (.055, .205), (.075, .216), (.112, .236), (.112, .268), (.085, .285), (.04, .280), (0, .272), (-.04, .280)]
        pts = spl(np.array(tre), True, n=3); pts = np.vstack([pts, pts[:1]])
        mid = min(cache.get('m', surf(x, .23)[1]) for x in (-.11, .11)); del mid
        self.chain([surf(a, b, .020 + .022 * max(0, 1 - abs(a) / .07)) for a, b in pts], .0105, **g)
        down = V(0, 0, -1)
        def tassel(p0, L, w):
            for q in range(5):
                a = (q - 2) * .0072; off = Rc @ V(a, -.004 * (1 - abs(q - 2) / 2), 0)
                self.cone(p0 + off * .3, p0 + off * 1.5 + down * (L - .006 * ((q * 7) % 3)) + Rc @ V(0, -.012, 0), w * .8, w, **g)
            self.ell(p0 - down * .004, (.014, .014, .012), **g)
        for s in (1, -1):
            for x, z, r, L in ((.160, .262, .028, .125), (.255, .322, .025, .100)):
                c = surf(x * s, z, .022); self.tor(c, r, .0095, fr(x * s, z), g=11, mat=CORD, part=P_ORN)
                tassel(surf(x * s, z - r - .012, .026), L, .0078)
            self.chain([surf(.184 * s, .277, .02), surf(.232 * s, .306, .02)], .010, **g)
            self.chain([surf(.276 * s, .337, .018), surf(.315 * s, .375, .014), surf(.335 * s, .420, .012), self.C(.33 * s, -.06, .530), self.C(.25 * s, .10, .56), self.C(.10 * s, .20, .58)], .0090, **g)
        tassel(surf(0, .145, .045), .16, .0085)

    # ---- the wheel
    def _wheel(self):
        p = self.pose; o = np.asarray(p['woff'], float); R = self.fh[0]
        q = self.CR0 + ypr(p['crest']) @ (V(-.035, .115, .965) - self.CR0) if any(p['crest']) else V(-.035, .115, .965); c = self.Hd(q[0] + o[0], q[1] + o[1], q[2] + o[2]); tl = p['wtilt']; M = R @ ypr(p['crest']) @ ypr(tl)
        ex, ey, ez = M @ V(1, 0, 0), M @ V(0, 0, 1), M @ V(0, -1, 0); hs = self.hs                  # in-plane right(char left), up, axis toward front
        self.wheel = dict(c=self.w(c), ex=self.wd(ex), ey=self.wd(ey), ez=self.wd(ez), R=.270 * self.sc * self.hs, spin=float(p['spin']))
        if self.with_wheel == 'burn': return
        Mw = np.stack([ex, ey, ez]); g = dict(g=12, mat=WHEEL, part=P_WHEEL)
        self.tor(c, .158, .026, Mw, zs=.8, **g); self.tor(c, .158, .012, Mw, zs=.35, **g); self.tor(c, .060, .018, Mw, **g)
        self.ell(c, (.052, .052, .046), Mw, k=.0, **g)
        for i in range(8):
            a = p['spin'] + i * math.pi / 4; d = (ex * math.cos(a) + ey * math.sin(a)) * hs; t_ = -ex * math.sin(a) + ey * math.cos(a)
            self.cone(c + d * .05, c + d * .145, .0165, .0120, k=.0, **g); self.cone(c + d * .10, c + d * .112, .022, .022, k=.0, **g)
            self.cone(c + d * .165, c + d * .255, .016, .0125, k=.0, **g); self.cone(c + d * .198, c + d * .206, .023, .023, k=.0, **g)
            self.ell(c + d * .285, (.046, .044, .044), np.stack([d / hs, t_, ez]), k=.0, **g); self.cone(c + d * .325, c + d * .35, .012, .003, k=.0, **g)
