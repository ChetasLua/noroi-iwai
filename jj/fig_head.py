"""Small and mid scale heads for jj.figure, constructed on head3.Head so any angle of the 3-D figure works."""
import numpy as np, math, cv2
from . import W, H as HH
from .paint import rgb, f32
from .ink import Cel, spl, resample
from . import kit
from .head3 import Head

PI = math.pi

YU = [(-1.0, .10), (-.96, .185), (-.86, .33), (-.72, .465), (-.56, .575), (-.36, .65), (-.12, .69), (.12, .71), (.35, .72), (.6, .69), (.8, .55), (.93, .34), (1.0, .05)]
PROP = dict(
    yuji=dict(a=YU), megumi=dict(a=YU, nose=(.10, -.06, -.40, -.46, .155)), gojo=dict(a=YU, nose=(.10, -.06, -.42, -.48, .17)), sukuna=dict(a=YU),
    nobara=dict(a=[(-1.0, .11), (-.96, .18), (-.86, .31), (-.72, .45), (-.56, .575), (-.36, .655), (-.12, .70), (.12, .72), (.35, .73), (.6, .70), (.8, .56), (.93, .35), (1.0, .05)], nose=(.10, -.10, -.40, -.45, .12)),
    nanami=dict(a=[(-1.0, .15), (-.96, .25), (-.86, .40), (-.72, .51), (-.56, .585), (-.36, .64), (-.12, .675), (.12, .695), (.35, .70), (.6, .67), (.8, .54), (.93, .34), (1.0, .05)], nose=(.12, -.06, -.43, -.49, .18)),
    todo=dict(a=[(-1.0, .18), (-.96, .30), (-.86, .46), (-.72, .58), (-.56, .655), (-.36, .70), (-.12, .72), (.12, .725), (.35, .72), (.6, .68), (.8, .54), (.93, .34), (1.0, .05)], nose=(.13, -.06, -.44, -.50, .19)))
for q in ('crowd', 'crowdf', 'curse'): PROP[q] = dict(PROP['nobara'] if q == 'crowdf' else {})

HAIR = dict(  # light, base, shadow, line
    yuji=('#f0bfae', '#cf8f86', '#93565c', '#3a1c24'), sukuna=('#f2b8ae', '#cc7f7c', '#8c4a54', '#36161f'), megumi=('#3f5a8c', '#1a2133', '#0c0f19', '#05060a'), nobara=('#f0a860', '#c8742c', '#84441c', '#3c1a0e'), gojo=('#ffffff', '#eceef6', '#a9aed0', '#4c4c6c'),
    nanami=('#f6eaa6', '#d9c274', '#95803f', '#42331a'), todo=('#4a4652', '#1b1a20', '#0d0c10', '#050507'), crowd=('#5a5664', '#25232b', '#131218', '#060609'), crowdf=('#8a6650', '#4a3228', '#291a16', '#120a08'), curse=('#000000', '#000000', '#000000', '#000000'))
IRIS = dict(yuji='#8a5a34', sukuna='#b0222c', megumi='#2f5a6c', nobara='#a8622c', gojo='#6fd0f2', nanami='#5c6a3a', todo='#4a3a30', crowd='#3a3434', crowdf='#4a3a34', curse='#000')

_heads = {}


def head_for(char):
    if char not in _heads: _heads[char] = Head(**PROP.get(char, {}))
    return _heads[char]


def pose_info(cam, sk, char=None):
    """screen pose of the head: x, y, s (px per unit), z, u, R (head -> view), yaw, pitch, roll, and a posed head3.Head"""
    c = sk.head; p, z = cam.project(c); t = cam.eye - c; D = np.linalg.norm(t); t = t / D
    up = -cam.R[1]; r = np.cross(up, t); r /= np.linalg.norm(r) + 1e-9; u = np.cross(t, r)
    M = np.stack([-sk.Rh[:, 0], sk.Rh[:, 2], sk.Rh[:, 1]], 1)
    R = np.stack([r, u, t]) @ M
    pitch = math.asin(float(np.clip(-R[2, 1], -1, 1))); yaw = math.atan2(-R[2, 0], R[2, 2])
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    RxRy = np.array([[cy, 0, sy], [-sp * sy, cp, sp * cy], [-cp * sy, -sp, cp * cy]]); Q = R @ RxRy.T; roll = -math.atan2(Q[1, 0], Q[0, 0])
    s = cam.f / max(float(z), .05) * sk.u
    return dict(x=float(p[0]), y=float(p[1]), s=float(s), z=float(z), u=sk.u, R=R, yaw=yaw, pitch=pitch, roll=roll, persp=float(min(.45, sk.u / D)), view=np.stack([r, u, t]), H=None, world=c)


def posed(char, hi):
    H = head_for(char).pose(hi['x'], hi['y'], hi['s'], hi['yaw'], hi['pitch'], hi['roll'], hi['persp']); H.R = hi['R']; return H


# ------------------------------------------------------------------ caps
def cap(H, line, lift=.035, front=True, ny=26, nf=64, top=1.0):
    """hair cap above hairline(theta) -> list of screen polygons (front-facing quads only, or all)"""
    th = np.linspace(-PI, PI, nf); y0 = np.array([line(t) for t in th])
    v = np.linspace(0, 1, ny)[:, None]; Y = y0[None] * (1 - v) + top * v
    A = np.interp(Y, H.yy, H.A); F = np.interp(Y, H.yy, H.F); Bk = np.interp(Y, H.yy, H.B); p = H.k['p']
    sx, cz = np.sin(th)[None], np.cos(th)[None]; r = (np.abs(sx) ** p + np.abs(cz) ** p) ** (-1 / p)
    X = A * sx * r; Z = np.where(cz >= 0, F, Bk) * cz * r
    Q = np.stack([X * (1 + lift), Y + lift * (Y > .5) * (Y - .5) * 2 + 0 * X, Z * (1 + lift)], -1)
    P = H.P3(Q); q = np.stack([P[:-1, :-1], P[:-1, 1:], P[1:, 1:], P[1:, :-1]], 2).reshape(-1, 4, 2)
    if front:
        a = q[:, 1] - q[:, 0]; b = q[:, 3] - q[:, 0]; cr = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        a2 = q[:, 2] - q[:, 1]; cr2 = a2[:, 0] * (q[:, 3] - q[:, 2])[:, 1] - a2[:, 1] * (q[:, 3] - q[:, 2])[:, 0]
        q = q[(cr + cr2) < 0]
    if len(q) == 0: return []
    k = max(.6, min(3., 160. / max(H.s, 8))); lo = q.reshape(-1, 2).min(0) - 4; hi = q.reshape(-1, 2).max(0) + 4
    w, h = np.ceil((hi - lo) * k).astype(int) + 2; m = np.zeros((h, w), np.uint8)
    for c in np.round((q - lo) * k * 16).astype(np.int32): cv2.fillConvexPoly(m, c, 255, cv2.LINE_8, 4)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return [resample(c / k + lo, max(1., H.s * .02)) for c in kit.contours(m / 255., min_len=12)]


def face_shadow(H, lv, thr=.06, eps=None):
    """cel terminator on the face shell from real surface normals: list of angular screen polygons.  lv = light in view space (x right, y up, z toward viewer)"""
    y = np.linspace(-1, .78, 54); u = np.linspace(-1, 1, 44); a = np.interp(y, H.yy, H.A)[:, None]; X = u[None] * a * .995; Y = np.repeat(y[:, None], 44, 1); Z = H.z(X, Y)
    e = .012; zx = (H.z(np.clip(X + e, -a, a), Y) - H.z(np.clip(X - e, -a, a), Y)) / (2 * e); zy = (H.z(X, Y + e) - H.z(X, Y - e)) / (2 * e)
    n = np.stack([-zx, -zy, np.ones_like(Z)], -1); n /= np.linalg.norm(n, axis=-1, keepdims=True); l = np.asarray(lv, float); l = H.R.T @ (l / np.linalg.norm(l))
    m = (n @ l) < thr; P = H.P3(np.stack([X, Y, Z], -1)); q = np.stack([P[:-1, :-1], P[:-1, 1:], P[1:, 1:], P[1:, :-1]], 2).reshape(-1, 4, 2); mm = (m[:-1, :-1] & m[1:, 1:]).ravel()
    q = q[mm]
    if len(q) < 4: return []
    k = max(.8, min(4., 150. / max(H.s, 6))); lo = q.reshape(-1, 2).min(0) - 4; hi = q.reshape(-1, 2).max(0) + 4; w, h = np.ceil((hi - lo) * k).astype(int) + 2; M = np.zeros((h, w), np.uint8)
    for c in np.round((q - lo) * k * 16).astype(np.int32): cv2.fillConvexPoly(M, c, 255, cv2.LINE_8, 4)
    M = cv2.morphologyEx(cv2.morphologyEx(M, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)); out = []
    for c in kit.contours(M / 255., min_len=20):
        v = cv2.approxPolyDP(c.astype(np.float32)[:, None], 150. * .035, True)[:, 0] / k + lo
        if len(v) >= 3: out.append(v)
    return out


def C(th, y, L, w, d, curl=(0, 0, 0), bulge=.16, bend=0., pr=0):
    return dict(th=th, y=y, L=L, w=w, d=np.asarray(d, float), curl=np.asarray(curl, float), bulge=bulge, bend=bend, pr=pr)


def hair_yuji(r, k=1.0):
    """three tiers of unequal spikes swept up and back; five or six of them carry the silhouette"""
    o = []; cb = (0, .0, -.16)
    for t, y, L, w, dx, dy, dz in ((-1.14, .46, .34, .34, -.86, .40, .10), (-.80, .53, .70, .50, -.70, .80, .18), (-.38, .57, .44, .36, -.24, .95, .30), (.0, .58, .78, .54, .12, .98, .26), (.42, .56, .46, .36, .36, .92, .30), (.76, .53, .74, .50, .74, .74, .18), (1.14, .46, .38, .34, .90, .32, .10)):
        o.append(C(t, y, L * k, w, (dx, dy, dz), (0, 0, -.10), .16, .06 * np.sign(dx), 2))
    for t, y, L, w, dx, dy, dz in ((-1.52, .64, .56, .52, -.88, .60, -.2), (-.58, .82, .56, .50, -.34, 1., -.15), (.24, .86, .70, .58, .18, 1., -.2), (1.0, .78, .48, .46, .56, .84, -.15), (1.56, .62, .62, .50, .92, .50, -.2), (2.3, .72, .56, .52, .58, .62, -.6), (-2.4, .72, .62, .54, -.56, .62, -.6), (PI, .80, .58, .58, .08, .72, -.8)):
        o.append(C(t, y, L * k, w, (dx, dy, dz), cb, .18, .05 * np.sign(dx), 1))
    for t, L, w, dz, drop in ((1.62, .34, .40, .25, .06), (1.95, .50, .46, .60, .20), (2.35, .38, .40, .80, .14), (2.75, .56, .50, .90, .30), (3.14, .44, .46, 1., .22), (3.50, .60, .50, .90, .32), (3.92, .40, .40, .80, .16), (4.30, .52, .46, .60, .22), (4.64, .32, .38, .25, .06)):
        o.append(C(t, .44, L * k, w * 1.1, (math.sin(t) * .60, -.10, -dz), (.16 * math.sin(t * 5), -drop * 1.9 - .10, -.04), .24, .09 * math.sin(t * 7), 0))          # nape side: swept back and down over the undercut, unequal
    return o


def hair_megumi(r, k=1.0):
    o = []
    for i, t in enumerate(np.linspace(0, 2 * PI, 12)[:-1] + .15):
        L = .62 + .26 * r.random(); o.append(C(t, .62, L, .34, (math.sin(t) * 1.0, .42 + .25 * r.random(), math.cos(t) * .9), (0, -.04, 0), .10, r.normal(0, .07), 1))
    for i, t in enumerate(np.linspace(0, 2 * PI, 9)[:-1] + .4): o.append(C(t, .84, .66 + .22 * r.random(), .36, (math.sin(t) * .62, .95, math.cos(t) * .5), (0, 0, 0), .10, r.normal(0, .07), 1))
    for t in (-2., .3, 2.): o.append(C(t, .96, .62 + .15 * r.random(), .32, (math.sin(t) * .3, 1., math.cos(t) * .2), (0, 0, 0), .10, r.normal(0, .07), 1))
    for i, t in enumerate(np.linspace(1.5, 2 * PI - 1.5, 8)): o.append(C(t, .22, .46 + .2 * r.random(), .36, (math.sin(t) * .62, -.55 + .35 * r.random(), math.cos(t) * .75), (0, -.05, 0), .12, r.normal(0, .07), 0))
    for t, L, bx in ((-.95, .62, -.25), (-.55, .60, -.12), (-.18, .66, .02), (.14, .58, .10), (.52, .62, .18), (.92, .60, .25)): o.append(C(t, .60, L, .26, (bx, -.82, .52), (0, -.02, -.12), .12, bx * .3, 3))
    return o


def hair_gojo(r, k=1.0):
    o = []
    for i, t in enumerate(np.linspace(-1.35, 1.35, 8)): o.append(C(t, .44, .62 + .2 * r.random(), .30, (math.sin(t) * .50, .95, math.cos(t) * .35), (0, .04, -.10), .12, r.normal(0, .06), 2))
    for i, t in enumerate(np.linspace(0, 2 * PI, 10)[:-1]): o.append(C(t, .72, .56 + .22 * r.random(), .36, (math.sin(t) * .52, 1., math.cos(t) * .34), (0, 0, 0), .13, r.normal(0, .07), 1))
    for t in (-2.2, -.6, .9, 2.6): o.append(C(t, .94, .50 + .2 * r.random(), .36, (math.sin(t) * .34, 1., math.cos(t) * .24), (0, 0, 0), .13, r.normal(0, .07), 1))
    for i, t in enumerate(np.linspace(1.4, 2 * PI - 1.4, 8)): o.append(C(t, .40, .42 + .16 * r.random(), .34, (math.sin(t) * .85, .72, math.cos(t) * .8), (0, 0, 0), .14, r.normal(0, .07), 1))
    return o


def hair_nobara(r, k=1.0):
    """bob: the body is a shell with a ragged hem (shell()); fringe swept from a part on her left"""
    o = []
    for t, y, L, bx, w in ((-.78, .78, 1.26, -.34, .34), (-.50, .80, .92, -.36, .34), (-.22, .82, .70, -.30, .34), (.02, .82, .58, -.18, .28)): o.append(C(t, y, L, w, (bx, -.92, .30), (-.03, -.02, -.16), .16, bx * .26, 3))
    for t, y, L, bx, w in ((.30, .82, .74, .46, .30), (.58, .80, 1.14, .42, .36), (.84, .76, 1.30, .30, .34)): o.append(C(t, y, L, w, (bx, -.92, .26), (.02, -.02, -.14), .16, bx * .24, 3))
    return o


def hair_nanami(r, k=1.0):
    """one combed mass: see shell(); only the lifted front wave and a few loose ends are locks"""
    o = []
    for t, y, L, w, dx, dy, dz in ((-.95, .50, .34, .40, -.55, .30, -.45), (-.55, .57, .42, .46, -.45, .55, -.35), (.10, .60, .62, .60, .72, .50, -.28), (.46, .58, .58, .50, .80, .34, -.35), (.86, .50, .42, .44, .72, .10, -.50)): o.append(C(t, y, L, w, (dx, dy, dz), (.08 * np.sign(dx), -.12, -.12), .34, .10 * np.sign(dx), 2))
    for t, L in ((1.7, .30), (2.4, .26), (3.2, .32), (4.0, .26), (4.6, .30)): o.append(C(t, -.14, L, .34, (math.sin(t) * .25, -.9, math.cos(t) * .3), (0, 0, 0), .16, 0, 0))
    return o


def hair_todo(r, k=1.0):
    o = []
    for i, t in enumerate(np.linspace(-1.2, 1.2, 7)): o.append(C(t, .62 - .10 * abs(t), .46, .34, (-math.sin(t) * .25, .62, -.75), (0, 0, -.1), .20, 0, 2))
    for i, t in enumerate(np.linspace(0, 2 * PI, 11)): o.append(C(PI + .25 * math.sin(t * 3), .99, .30 + .16 * r.random(), .16, (math.cos(t) * .75, .75 + .3 * r.random(), math.sin(t) * .5 - .25), (0, 0, 0), .14, r.normal(0, .15), 4))
    return o


def hair_crowd(r, k=1.0):
    o = []
    for i, t in enumerate(np.linspace(-1.2, 1.2, 6)): o.append(C(t, .62, .42 + .1 * r.random(), .36, (math.sin(t) * .5, -.35, .7), (0, -.12, -.12), .18, r.normal(0, .06), 3))
    for i, t in enumerate(np.linspace(0, 2 * PI, 9)[:-1]): o.append(C(t, .88, .40, .44, (math.sin(t) * .8, .1, math.cos(t) * .7), (0, -.10, 0), .2, r.normal(0, .05), 1))
    for i, t in enumerate(np.linspace(1.3, 2 * PI - 1.3, 8)): o.append(C(t, .40, .42, .38, (math.sin(t) * .45, -.75, math.cos(t) * .5), (0, 0, 0), .16, r.normal(0, .05), 0))
    return o


def hair_crowdf(r, k=1.0):
    o = []
    for i, t in enumerate(np.linspace(.8, 2 * PI - .8, 10)): o.append(C(t, .66, 1.7 + .2 * r.random(), .46, (math.sin(t) * .30, -1., math.cos(t) * .22 - .05), (0, 0, 0), .14, r.normal(0, .03), 0 if abs(math.atan2(math.sin(t), math.cos(t))) > 1.7 else 2))
    for t, bx in ((-.6, -.25), (-.2, -.1), (.2, .1), (.6, .25)): o.append(C(t, .74, .62, .32, (bx, -.9, .45), (0, 0, -.12), .14, 0, 3))
    return o


GEN = dict(yuji=hair_yuji, sukuna=lambda r: hair_yuji(r, 1.22), megumi=hair_megumi, gojo=hair_gojo, nobara=hair_nobara, nanami=hair_nanami, todo=hair_todo, crowd=hair_crowd, crowdf=hair_crowdf)
_clumps = {}


def clumps(char):
    if char not in _clumps: _clumps[char] = GEN[char](np.random.default_rng(sum(map(ord, char)) + 3))
    return _clumps[char]


def hairline(char):
    def f(t):
        a = abs(math.atan2(math.sin(t), math.cos(t)))
        base = dict(yuji=.60, sukuna=.62, megumi=.60, gojo=.30, nobara=.62, nanami=.60, todo=.60).get(char, .60)
        side = dict(yuji=.30, sukuna=.30, megumi=.0, gojo=.20, nobara=-.1, nanami=.05, todo=.12).get(char, .05)
        back = dict(yuji=.16, sukuna=.16, megumi=-.36, gojo=.0, nobara=-.45, nanami=-.36, todo=-.22).get(char, -.36)
        if a < .95: return base - .05 * math.sin(a * 5) * (a > .3) - (0 if a < .6 else (a - .6) * .35)
        if a < 1.5: u = (a - .95) / .55; return (base - .12) * (1 - u) + side * u
        u = min(1., (a - 1.5) / .9); return side * (1 - u) + back * u
    return f


# ------------------------------------------------------------------ draw
def shell(H, hem, top=1.0, infl=.10, open_=0., front=True, ny=30, nf=72, side_in=.0):
    """hair shell hanging from the crown to hem(theta) (may go below the skull): radius never less than the skull at y=.15.
    open_ = half angle left open round the face.  -> screen polygons"""
    th = np.linspace(-PI + 1e-3, PI - 1e-3, nf); y1 = np.array([hem(t) for t in th]); v = np.linspace(0, 1, ny)[:, None]; Y = top * (1 - v) + y1[None] * v
    yc = np.maximum(Y, .16); A = np.interp(yc, H.yy, H.A); F = np.interp(yc, H.yy, H.F); Bk = np.interp(yc, H.yy, H.B); p = H.k['p']; sx, cz = np.sin(th)[None], np.cos(th)[None]; r = (np.abs(sx) ** p + np.abs(cz) ** p) ** (-1 / p)
    g = 1 + infl * np.clip((1.02 - Y) / .5, .25, 1) - side_in * np.clip((-.25 - Y) / .5, 0, 1)
    Q = np.stack([A * sx * r * g, Y + infl * .55 * np.clip((Y - .4) / .6, 0, 1), np.where(cz >= 0, F, Bk) * cz * r * g], -1); P = H.P3(Q)
    q = np.stack([P[:-1, :-1], P[:-1, 1:], P[1:, 1:], P[1:, :-1]], 2).reshape(-1, 4, 2); keep = np.ones(len(q), bool)
    if open_: keep &= (np.abs(np.repeat(((th[:-1] + th[1:]) / 2)[None], ny - 1, 0)) > open_).ravel() | ((Y[:-1, :-1] > .66).ravel())
    if front:
        d = H.depth(Q); n_ = np.cross(Q[1:, :-1] - Q[:-1, :-1], Q[:-1, 1:] - Q[:-1, :-1]); f_ = (n_ @ H.R.T)[..., 2].ravel(); s_ = np.sign((np.cross(Q[1:2, :-1] - Q[:1, :-1], Q[:1, 1:] - Q[:1, :-1]) * Q[:1, :-1]).sum(-1)); s_ = 1. if np.nanmedian(s_) >= 0 else -1.
        a_ = q[:, 1] - q[:, 0]; b_ = q[:, 3] - q[:, 0]; keep &= (a_[:, 0] * b_[:, 1] - a_[:, 1] * b_[:, 0]) > 0
    q = q[keep]
    if len(q) == 0: return []
    k = max(.6, min(3., 160. / max(H.s, 8))); lo = q.reshape(-1, 2).min(0) - 4; hi = q.reshape(-1, 2).max(0) + 4; w, h = np.ceil((hi - lo) * k).astype(int) + 2; m = np.zeros((h, w), np.uint8)
    for c in np.round((q - lo) * k * 16).astype(np.int32): cv2.fillConvexPoly(m, c, 255, cv2.LINE_8, 4)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return [resample(c / k + lo, max(1., H.s * .02)) for c in kit.contours(m / 255., min_len=12)]


def bob_hem(t):
    a = abs(math.atan2(math.sin(t), math.cos(t))); z = ((t * 2.35 + .3) % 1.); z2 = ((t * 1.1 + .7) % 1.)
    return -.56 - .20 * (1 - abs(z * 2 - 1)) - .10 * (1 - abs(z2 * 2 - 1)) + (.08 if a > 2.2 else 0) - (.06 if a < 1.2 else 0)


def comb_hem(t):
    a = abs(math.atan2(math.sin(t), math.cos(t)))
    if a < .95: return .60 - (0 if a < .55 else (a - .55) * .35)
    if a < 1.35: return .46 - (a - .95) * 1.1
    return float(np.interp(a, [1.35, 1.7, 2.2, PI], [.02, -.02, -.34, -.46]))


def surf(H, th, y, infl):
    """screen point(s) of the hair surface"""
    th, y = np.asarray(th, float), np.asarray(y, float); yc = np.maximum(y, .16); A = np.interp(yc, H.yy, H.A); F = np.interp(yc, H.yy, H.F); Bk = np.interp(yc, H.yy, H.B); p = H.k['p']; sx, cz = np.sin(th), np.cos(th); r = (np.abs(sx) ** p + np.abs(cz) ** p) ** (-1 / p)
    Q = np.stack([A * sx * r * (1 + infl), y + infl * .55 * np.clip((y - .4) / .6, 0, 1), np.where(cz >= 0, F, Bk) * cz * r * (1 + infl)], -1); return H.P3(Q), H.depth(Q)


def draw(char, hi, roi, D):
    """premultiplied RGBA of the roi (x0, y0, x1, y1).  D = the _Draw of figure.py (light, palette, pen weights)"""
    x0, y0, x1, y1 = roi; w, h = x1 - x0, y1 - y0
    H = posed(char, hi); hi['H'] = H; s = hi['s']; pal = D.pal; ink = D.ink
    cel = Cel(w, h, x0, y0, ss=2 if s > 10 else 3)
    if char == 'curse': return np.zeros((h, w, 4), f32)
    f = D.fx['f']; expr = D.expr or {}
    lw = min(D.wo, max(1.0, s * .10)); lo, li, lf = lw * .92, lw * .66, lw * .52
    skin = pal(D.B.sp['skin']); hc = [rgb(c) for c in D.fig.spec.get('haircol', HAIR[char])]; hl, hb, hs, hk = (hc[0] * pal.litmul + pal.add, hc[1] * pal.mul + pal.add, hc[2] * pal.shmul + pal.add * .6, hc[3])
    lc = D.lc; sd = -1 if lc[0] < 0 else 1                       # light side on screen
    fz = hi['R'][2, 2]; yaw = hi['yaw']                          # how much the face looks at us
    ld = np.array([lc[0], -lc[1]]); ld = ld / (np.hypot(*ld) + 1e-6); cx_, cy_ = hi['x'], hi['y'] - .45 * s
    o_ = H.P3(np.array([(0., 0., 0.), (0., -1., 0.), (0., 1., 0.)])); dn = (o_[1] - o_[0]) / (np.hypot(*(o_[1] - o_[0])) + 1e-6); up_ = -dn
    # ---- locks
    lag = hi['view'] @ np.asarray(D.fx.get('hair', np.zeros(3)), float); lagH = hi['R'].T @ lag
    cl = []
    for i, c in enumerate(clumps(char)):
        sway = lagH * c['L'] * .55 + np.array([math.sin(f * .5 + i * 1.3), 0, math.cos(f * .43 + i)]) * np.linalg.norm(lagH) * .05
        dv_, cv_ = c['d'], c['curl']
        if char in ('yuji', 'sukuna') and fz < .25 and c['pr'] in (1, 2):
            q_ = min(1., (.25 - fz) / .5); sd_ = math.sin(c['th'] * 3.7 + i); dv_ = dv_ * [1., 1 - .42 * q_, 1.] + np.array([.28 * sd_, 0, -.45]) * q_; cv_ = cv_ + np.array([.10 * sd_, -.16 - .10 * abs(sd_), -.10]) * q_
        a, t, b, dp = H.clump(c['th'], c['y'], c['L'] * (1 + (.22 * math.sin(i * 2.3) if char in ('yuji', 'sukuna') and fz < .25 else 0)), c['w'], dv_, cv_ + sway)
        n = np.array([math.sin(c['th']), (c['y'] - .1) * .9, math.cos(c['th'])]); n /= np.linalg.norm(n); fac = float((hi['R'] @ n)[2])
        cl.append(dict(P=kit.spike(a, t, b, c['bulge'], c['bend'], n=9), dp=dp, fac=fac, pr=c['pr'], i=i, a=a, t=t, b=b, L=c['L'], w=c['w']))
    back = [c for c in cl if c['fac'] < -.05 and c['pr'] < 3 or (c['pr'] == 0 and c['fac'] < .25)]
    if char in ('yuji', 'sukuna') and fz < .1: back = [c for c in back if not (c['pr'] == 0 and c['fac'] > -.2)]
    back = [c for c in back if c['pr'] != 3]; ids = {c['i'] for c in back}; front = [c for c in cl if c['i'] not in ids and not (c['pr'] == 3 and c['fac'] < .05)]; back.sort(key=lambda c: c['dp'])
    hlf = hairline(char); mass = char in ('nanami', 'nobara', 'crowdf')
    if char == 'nanami': capf, capa = shell(H, comb_hem, infl=.075, front=True), shell(H, comb_hem, infl=.075, front=False)
    elif char in ('nobara', 'crowdf'):
        hem = bob_hem if char == 'nobara' else (lambda t: bob_hem(t) - .55)
        si = -.16 if char == 'nobara' else .06; capf, capa = shell(H, hem, infl=.17, open_=.86, front=True, side_in=si), shell(H, hem, infl=.17, front=False, open_=.80, side_in=si)
    else: capf = cap(H, hlf, .045, True); capa = cap(H, hlf, .045, False)
    if char in ('yuji', 'sukuna'):
        def ul(t):
            a_ = abs(math.atan2(math.sin(t), math.cos(t))); return hlf(t) - float(np.interp(a_, [1.05, 1.16, 1.35, 1.7, 2.3, PI], [0, .30, .38, .22, .46, .66]))
        uc = cap(H, ul, .012, False); ucf = cap(H, ul, .012, True); ucol = (rgb('#4a2e30') if char == 'yuji' else rgb('#7c3c48')) * pal.mul + pal.add
    else: uc = ucf = []; ucol = hs
    # ---- everything behind the face
    for c in back + [c for c in front if c['pr'] != 3]: kit.outline(cel, c['P'], lo, ink, seed=c['i'])
    for P in capa: cel.line(P, lo, ink, closed=True, taper=(0, 0), seed=3)
    for c in back: cel.fill(c['P'], hb)
    for P in uc: cel.line(P, lo * .7, ink, closed=True, taper=(0, 0), seed=13); cel.fill(P, ucol)
    for P in capa: cel.fill(P, hs if char in ('nobara', 'crowdf') else hb)
    if s > 9:
        for c in back:
            m_ = (c['a'] + c['b']) / 2
            cel.fill(np.vstack([c['a'], c['a'] * .5 + c['t'] * .5 - ld * .02 * s, m_ * .3 + c['t'] * .7, c['b'] * .55 + c['t'] * .45, c['b']]), hs)
    # ---- face
    sk_ = H._sil([H._mesh(-.50, .70, 0)])
    kp = ((sk_ - o_[0]) @ dn) < .40 * s
    if kp.any() and not kp.all():
        r_ = int(np.argmin(kp)); q_ = np.roll(sk_, -r_, 0); k_ = np.roll(kp, -r_); i_ = np.nonzero(k_)[0]
        if len(i_) > 6: cel.line(q_[i_[0]:i_[-1] + 1], lo * .85, ink, taper=(.1, .1), seed=21)
    cel.fill(sk_, skin[2] if fz < .2 else skin[1])
    if fz > -.35:
        if hi['pitch'] > -.02: cel.fill(H.under(), skin[2])
        for e in (-1, 1):
            E = H.ear(e)
            if E['vis'] > .10 and char not in ('nobara', 'crowdf'):
                kit.shape(cel, E['outer'], skin[1] if e * -sd < 0 else skin[2], lo * .8, ink, seed=7 + e)
                if s > 26: cel.line(E['inner'], lf, ink, seed=9)
        F = H.face(); cel.fill(F, skin[1])
        # the terminator from the form itself, cut into straight planes
        head_clip = [F, sk_]
        cel.ctx.save(); cel._path(head_clip); cel.ctx.set_fill_rule(0); cel.ctx.clip(); sg = -sd
        for q_ in face_shadow(H, lc, .10): cel.fill(q_, skin[2])
        if s > 13 and abs(yaw) < .9: cel.fill(H.P([(sg * .20, .10), (sg * .30, .03), (sg * .48, .06), (sg * .60, .16), (sg * .60, .02), (sg * .46, -.10), (sg * .66, -.20), (sg * .72, .2)]), skin[2])
        if lc[1] > .2: cel.fill(H.P([(-.66, .80), (-.64, .42), (-.44, .50), (-.30, .36), (-.14, .50), (.02, .38), (.2, .50), (.34, .37), (.48, .48), (.64, .40), (.66, .8)]), skin[2])
        if lc[1] > .2 and s > 18: cel.fill(H.P([(-.16, -.955), (0, -.90), (.16, -.955), (.2, -1.05), (-.2, -1.05)]), skin[2])
        cel.ctx.restore()
        # ink only what is contour: the part of the face edge that the skull does not continue behind, then the jaw to the ear lobe
        poly = sk_.astype(np.float32)[:, None]; inside = np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), True) for x, y in F]) > .045 * s
        vis = ~inside; r_ = int(np.argmin(vis)) if (~vis).any() else 0; Fr = np.roll(F, -r_, 0); vr = np.roll(vis, -r_)
        runs_ = _runs(vr, 5)
        for a_, b_ in runs_:
            seg = Fr[a_:b_]
            if len(seg) > 4: cel.line(spl(seg[::max(1, len(seg) // 40)], n=4), lo * .95, ink, taper=(.14, .14), press=.25, seed=11)
        nearside = [e for e in (-1, 1) if H.ear(e)['vis'] > .10 and abs(yaw) > .18 and (e * yaw < 0 or abs(yaw) < .3)]
        for e in nearside:
            J = H.jaw(e); cel.line(spl(J[2:], n=5), li * .9, ink, taper=(.65, .10), seed=15 + e)
        if abs(yaw) > .62:                                              # the nose breaks the contour of a turned head
            np_ = H.nose_profile(); Pn = H.P3(np_); cel.fill(Pn, skin[2] if (np.sign(yaw) * -sd) > 0 and lc[2] < .6 else skin[1])
            cel.line(H.P3(np_[1:8]), lo * (.95 if abs(yaw) > .95 else .7), ink, taper=(.45, .15), seed=5)
        face_marks(cel, H, char, s, li, lf, ink, skin, hi, pal, expr, sd)
    else:
        for e in (-1, 1):
            E = H.ear(e)
            if E['vis'] > -.2 and abs(yaw) < 2.9 and char not in ('nobara', 'crowdf'): kit.shape(cel, E['outer'], skin[2], lo * .8, ink, seed=7 + e)
    # ---- hair in front: tiers far to near; a nearer tier throws one zig-zag shadow on the roots of the next
    for P in ucf: cel.fill(P, ucol)
    if not (char in ('yuji', 'sukuna') and fz < .45):
        for P in capf: cel.line(P, lo * .8, ink, closed=True, taper=(0, 0), seed=5)
    if char == 'gojo' and not expr.get('bare'): blindfold(cel, H, hi, lo, lf, ink, pal, s)
    for P in capf: cel.fill(P, hb)
    if s > 7 and not mass:
        for P in capf: cel.clip(P); cel.fill(P, hs); cel.fill(P + ld * .30 * s, hb); cel.unclip()
    tiers = {}
    for c in front: tiers.setdefault(c['pr'], []).append(c)
    order = sorted(tiers, key=lambda k: (k == 3, np.mean([c['dp'] for c in tiers[k]]) + (9 if k == 0 and fz < .1 else 0)))
    if mass and s > 8: _mass_detail(cel, H, char, s, capf, hl, hb, hs, hk, lf, ld, sd)
    done = list(capf)
    for k_ in order:
        T = sorted(tiers[k_], key=lambda c: c['L']); Ps = [c['P'] for c in T]
        if done and s > 7:
            off = up_ * .085 * s - ld * .05 * s
            cel.ctx.save(); cel._path(done); cel.ctx.set_fill_rule(0); cel.ctx.clip(); cel.fill([P + off for P in Ps], hs); cel.ctx.restore()
        if k_ == 3:
            for P in Ps: kit.outline(cel, P, lf * 1.25, hk, seed=1)
        for P in Ps: cel.fill(P, hb)
        if s > 8:
            cel.ctx.save(); cel._path(Ps); cel.ctx.set_fill_rule(0); cel.ctx.clip()
            for c in T:                                                   # far side of the head: whole locks in shadow; lit side stays flat
                m_ = (c['a'] + c['b']) / 2; side = float((m_ * .6 + c['t'] * .4 - [cx_, cy_]) @ ld / s)
                dk = c['b'] if (c['b'] - c['a']) @ ld < 0 else c['a']; lt_ = c['a'] if dk is c['b'] else c['b']
                if side < -.34: cel.fill(c['P'], hs)
                elif side < .02 and c['L'] > .5: cel.clip(c['P']); cel.fill(np.vstack([dk, dk * .85 + lt_ * .15, m_ * .30 + dk * .10 + c['t'] * .60, c['t'], dk * .5 + c['t'] * .5 + (dk - lt_) * .3]), hs); cel.unclip()
                elif s > 18 and side > .18 and c['L'] > .6: cel.fill(np.vstack([lt_ * .5 + m_ * .12 + c['t'] * .38, lt_ * .25 + c['t'] * .75, (lt_ * .7 + m_ * .3) * .34 + c['t'] * .66]), hl)
            cel.ctx.restore()
        if s > 10:
            for c in T:
                N_ = len(c['P']) // 2; big = c['L'] > .5; u_ = .30 if big else .58
                for E_ in (c['P'][:N_ + 1], c['P'][N_:][::-1]): cel.line(E_[max(1, int(N_ * u_)):], lf * (1.15 if big else .85), hk, taper=(.75, .03), seed=c['i'], tmin=.06)
        done += Ps
    return cel.rgba()


def _mass_detail(cel, H, char, s, capf, hl, hb, hs, hk, lf, ld, sd):
    """combed sheets: shadow side, comb lines that follow the sweep, one light band"""
    cel.ctx.save(); cel._path(capf); cel.ctx.set_fill_rule(0); cel.ctx.clip()
    for P in capf: cel.fill(P, hs); cel.fill(P + ld * .26 * s, hb)
    r = np.random.default_rng(6)
    if char == 'nanami':
        part = .34
        for sg, n in ((1, 6), (-1, 5)):
            for q in range(n):
                t0 = part + sg * (.02 + .16 * q); t = np.linspace(t0, t0 + sg * (1.5 + .12 * q), 12); y = np.linspace(.64 - .05 * q, .42 - .10 * q, 12) + .24 * np.sin(np.linspace(0, PI, 12)); P, d = surf(H, t, y, .08); k = d > .05
                if k.sum() > 4: cel.line(P[k], lf * (1.1 if q % 2 == 0 else .8), hk, taper=(.15, .7), seed=q, tmin=.08)
        P, d = surf(H, np.full(6, part), np.linspace(.60, .98, 6), .08)
        if (d > 0).all(): cel.line(P, lf * 1.5, hk, taper=(.05, .7), seed=40)
        t = np.linspace(-.9, 1.5, 14); P1, d1 = surf(H, t, .80 + .02 * np.sin(t * 9), .085); P2, _ = surf(H, t, .72 + .03 * np.sin(t * 9 + 1), .085); k = d1 > .1
        if k.sum() > 4 and s > 14: cel.fill(np.vstack([P1[k], P2[k][::-1]]), hl, .55)
    else:
        for q in range(13):
            t0 = -PI + (q + .5 + r.uniform(-.2, .2)) * 2 * PI / 13
            if abs(t0) < .8: continue
            P, d = surf(H, t0 + .10 * np.sin(np.linspace(0, 2, 10)), np.linspace(.80 - .3 * r.random(), -.66 + .2 * r.random(), 10), .165); k = d > .05
            if k.sum() > 4: cel.line(P[k], lf * r.uniform(.75, 1.15), hk, taper=(.6, .15), seed=q, tmin=.08)
        t = np.linspace(-PI, PI, 60); z = .030 * np.abs(((t * 3.1) % 1.) * 2 - 1); P1, d1 = surf(H, t, .52 + z, .17); P2, _ = surf(H, t, .40 - z, .17); k = (d1 > .12) & (np.abs(t) > .5)
        if k.sum() > 6 and s > 12:
            for a_, b_ in _runs(k, 4): cel.fill(np.vstack([P1[a_:b_], P2[a_:b_][::-1]]), hl, .8)
        P1, d1 = surf(H, t, -.36 - z * 2, .17); P2, _ = surf(H, t, np.full_like(t, -1.6), .17); k = d1 > .0
        for a_, b_ in _runs(k, 4): cel.fill(np.vstack([P1[a_:b_], P2[a_:b_][::-1]]), hs)
    cel.ctx.restore()


def _runs(ok, n):
    out = []; a = None
    for i, v in enumerate(list(ok) + [False]):
        if v and a is None: a = i
        if not v and a is not None:
            if i - a >= n: out.append((a, i))
            a = None
    return out


def _half(P, side):
    """the long edge of a spike nearest to `side`"""
    n = len(P) // 2; A, B = P[:n + 1], P[n:][::-1]
    return A if np.hypot(*(A[0] - side)) < np.hypot(*(B[0] - side)) else B


def blindfold(cel, H, hi, lo, lf, ink, pal, s):
    th = np.linspace(-PI * .98, PI * .98, 60)
    def ring(y, lift):
        A = np.interp(y, H.yy, H.A); F = np.interp(y, H.yy, H.F); Bk = np.interp(y, H.yy, H.B); p = H.k['p']; sx, cz = np.sin(th), np.cos(th); r = (np.abs(sx) ** p + np.abs(cz) ** p) ** (-1 / p)
        return np.stack([A * sx * r * (1 + lift), np.full_like(th, y), np.where(cz >= 0, F, Bk) * cz * r * (1 + lift)], -1)
    a, b = ring(.34, .04), ring(-.10, .045); d = H.depth((a + b) / 2) > 0.02
    if d.sum() < 4: return
    i = np.nonzero(d)[0]
    # contiguous run
    k = np.split(i, np.nonzero(np.diff(i) > 1)[0] + 1)
    for q in k:
        if len(q) < 3: continue
        P = np.vstack([H.P3(a[q]), H.P3(b[q])[::-1]])
        kit.shape(cel, P, rgb('#17161c') * pal.mul + pal.add, lo, ink, seed=3)
        if s > 14:
            hl = rgb('#4b4668') * pal.litmul + pal.add
            for t0, t1, y0, y1 in ((.12, .5, .24, .12), (.3, .62, .02, .06), (.6, .9, .22, .10)):
                z = q[int(len(q) * t0):max(int(len(q) * t1), int(len(q) * t0) + 3)]; m = len(z); yy = np.linspace(y0, y1, m); L = np.stack([a[z][:, 0], yy, a[z][:, 2]], -1); cel.line(H.P3(L), lf * 1.4, hl, taper=(.5, .5), seed=int(t0 * 100))


def eye(cel, H, e, s, ink, iris, li, lf, kind='m', look=(0, 0), open_=1.0, white='#f3efec', lash=1.0, shade=None, tilt=0.):
    ex = .325 * e; f = H.facing(ex, 0.)
    if f < .10: return
    w = .150; o = open_; tl = tilt
    if s < 13:
        cel.line(H.P([(ex - e * .13, .005 - tl * .3), (ex, .035), (ex + e * .15, .02 + tl * .3)]), max(1.0, li * 1.3), ink, taper=(.2, .2)); return
    top = [(ex - e * w * .95, -.022 - tl * .35), (ex - e * .075, .026 * o - tl * .1), (ex + e * .02, .046 * o + tl * .1), (ex + e * .095, .040 * o + tl * .35), (ex + e * w * 1.12, .012 + tl * .5)]
    bot = [(ex + e * w * 1.0, -.016 + tl * .35), (ex + e * .075, -.050 * o), (ex - e * .03, -.054 * o), (ex - e * w * .86, -.030 - tl * .3)]
    P = H.P(spl(np.array(top + bot), True, n=5)); cel.fill(P, rgb(white)); cel.clip(P)
    cx = ex + look[0] * .05 - e * .005; cy = -.004 + look[1] * .03; r = .060
    a = np.linspace(0, 2 * PI, 20)[:-1]; cel.fill(H.P(np.c_[cx + r * .88 * np.cos(a), cy + r * 1.08 * np.sin(a)]), rgb(iris) if isinstance(iris, str) else iris)
    cel.fill(H.P(np.c_[cx + r * .42 * np.cos(a), cy + r * .5 * np.sin(a)]), ink)
    if shade is not None: cel.fill(H.P([(ex - .25, .1), (ex - .25, .016 * o), (ex + .25, .022 * o), (ex + .25, .1)]), ink, .38)
    if s > 24: cel.fill(H.P(np.c_[cx - .020 + .013 * np.cos(a), cy + .024 + .014 * np.sin(a)]), '#ffffff')
    cel.unclip()
    cel.line(H.P(spl(np.array(top), n=6)), max(1.2, li * 1.75 * lash), ink, taper=(.4, .08), seed=int(5 + e), press=.2)
    if s > 16: cel.line(H.P(spl(np.array(bot[:3]), n=5)), lf * .9, ink, taper=(.3, .6))
    if s > 30: cel.line(H.P([(ex - e * .10, .058 * o + .02), (ex + e * .02, .084 * o + .02), (ex + e * .12, .070 * o + .03 + tl * .3)]), lf * .8, ink, taper=(.5, .4))


def eye_sharp(cel, H, e, s, ink, iris, lf, skin, look=(0, 0), open_=1.0, a=-.03, hcol=None, lash=1.0, brow=1.0, white='#f5f1ec'):
    """mid-shot eye: brow wedge and lid as two dark shapes that nearly touch, a dash of white, a cut-off iris.  x outward"""
    if H.facing(.325 * e, 0.) < .08: return
    X = lambda L: H.P(spl(np.array([(e * x, y) for x, y in L]), True, n=4), .006); o = open_
    if a < -.04: cel.fill(X([(.105, .085 + a), (.21, .055), (.17, -.005), (.11, .015)]), skin[3], .8)
    Wh = X([(.165, -.012), (.235, .022 * o), (.335, .040 * o), (.425, .026 * o), (.50, .0), (.41, -.036 * o), (.29, -.042 * o)])
    cel.fill(Wh, rgb(white)); cel.clip(Wh); cx = .318 + look[0] * .05 * e; cy = .002 + look[1] * .03; t = np.linspace(0, 2 * PI, 14)[:-1]
    cel.fill(H.P(np.c_[e * cx + .056 * np.cos(t), cy + .062 * np.sin(t)], .006), iris); cel.fill(H.P(np.c_[e * cx + .026 * np.cos(t), cy + .032 * np.sin(t)], .006), ink); cel.fill(X([(.14, .06), (.52, .06), (.52, .008 * o), (.33, .020 * o), (.16, .0)]), ink, .45); cel.unclip()
    th = .022 * lash
    cel.fill(X([(.145, -.014), (.225, .030 * o + th * .4), (.335, .048 * o + th), (.43, .036 * o + th), (.525, .012 + th * .4), (.43, .026 * o), (.335, .036 * o), (.235, .018 * o)]), ink)
    if s > 26: cel.line(H.P(np.array([(e * .46, -.018), (e * .39, -.044 * o), (e * .30, -.050 * o)]), .006), lf * .8, ink, taper=(.2, .7))
    b = .030 * brow
    cel.fill(X([(.105, .085 + a), (.12, .118 + a), (.22, .150 + a * .65 + b * .4), (.37, .186 + a * .05 + b * .2), (.53, .140 - a * .35), (.37, .172 + a * .05 - b * .55), (.23, .124 + a * .7 - b * .3)]), ink if hcol is None else hcol)


def face_marks(cel, H, char, s, li, lf, ink, skin, hi, pal, expr, sd):
    """brows, eyes, nose, mouth, and what makes each of them themselves"""
    mood = expr.get('mood', dict(yuji='firm', megumi='calm', nobara='firm', gojo='smile', nanami='calm', todo='grin', sukuna='grin').get(char, 'calm'))
    look = expr.get('look', (0, 0)); op = expr.get('open', dict(megumi=.82, sukuna=.85, nanami=.8, todo=.8).get(char, 1.))
    iris = rgb(IRIS[char]) * pal.mul + pal.add; lin = ink; hc = rgb(HAIR[char][2])
    shade = skin[2]
    if char == 'gojo' and not expr.get('bare'): pass
    elif char == 'nanami':
        for e in (-1, 1):
            if H.facing(.33 * e, 0) > .08:
                a = np.linspace(0, 2 * PI, 22)[:-1]; P = H.P(np.c_[.31 * e + .20 * np.cos(a), -.005 + .125 * np.sin(a) + .012 * np.cos(2 * a)], .05); kit.shape(cel, P, rgb('#26342c') * pal.mul + pal.add, li, ink, seed=3)
                if s > 18: cel.fill(H.P([(.31 * e - .10, .07), (.31 * e + .02, .085), (.31 * e - .06, -.02), (.31 * e - .13, .0)], .055), rgb('#7d9a86') * pal.mul + pal.add)
        cel.line(H.P([(-.11, .02), (0, .035), (.11, .02)], .06), li * 1.1, ink)
    else:
        if 13 <= s < 115:
            am = dict(firm=-.035, calm=.0, smile=.02, grin=-.05, sad=.045, rage=-.085, shout=-.075).get(mood, 0)
            for e in (-1, 1): eye_sharp(cel, H, e, s, ink, iris, lf, skin, look, op * dict(rage=.95, shout=1.15, grin=.85, sad=.8).get(mood, 1.), am, None if char in ('megumi', 'todo', 'crowd', 'sukuna') else rgb(HAIR[char][3]) * .8, 1.35 if char in ('nobara', 'megumi', 'crowdf') else 1., dict(todo=2.0, yuji=1.3, sukuna=1.3, megumi=.9, nobara=.75, crowdf=.75).get(char, 1.))
        else:
            for e in (-1, 1): eye(cel, H, e, s, ink, iris, li, lf, look=look, open_=op * dict(rage=.95, shout=1.1, grin=.85).get(mood, 1.), lash=1.25 if char in ('nobara', 'megumi', 'crowdf') else 1.0, shade=shade, tilt=dict(firm=.03, rage=.07, grin=.05, shout=.05, sad=-.03).get(mood, .015))
    # brows
    if not (char == 'gojo' and not expr.get('bare')) and s > 8 and (not 13 <= s < 115 or char == 'nanami'):
        for e in (-1, 1):
            if H.facing(.33 * e, .18) < .08: continue
            a = dict(firm=-.035, calm=.0, smile=.02, grin=-.05, sad=.05, rage=-.095, shout=-.085).get(mood, 0)
            B = [(e * .115, .110 + a), (e * .21, .140 + a * .72), (e * .37, .188 + a * .0), (e * .52, .140 - a * .35)]
            cel.line(H.P(spl(np.array(B), n=6), .015), max(1.1, li * dict(todo=2.4, yuji=1.8, sukuna=1.6, nanami=1.1).get(char, 1.4)), ink if char in ('megumi', 'todo', 'crowd') else hc * .45, taper=(.1, .5), press=.15)
    # nose
    np_ = H.nose_profile(); yaw = hi['yaw']
    if s > 10 and abs(yaw) < 1.0:
        k_ = .062 if abs(yaw) < .62 else .03; sg_ = -sd
        cel.fill(H.P3(np.array([np_[3] + [.012 * sg_, .03, .0], np_[4] + [0, .004, 0], np_[5] + [sg_ * .02, -.010, -.01], np_[5] + [sg_ * k_, -.012, -.07], np_[4] + [sg_ * k_ * .9, .02, -.06]])), shade)
        if s > 34: cel.line(H.P3(np.array([np_[5] + [.030 * sg_, -.004, -.03], np_[5] + [.004 * sg_, -.016, 0], np_[5] + [-.030 * sg_, -.006, -.03]])), lf * .9, ink, taper=(.4, .4))
        if abs(yaw) > .22: cel.line(H.P3(np_[2:5]), lf * .9, skin[3], taper=(.8, .15))
    # mouth
    my = -.665
    if H.facing(0, my) > .02 and s > 8:
        if mood in ('grin', 'shout', 'rage'):
            if mood == 'grin': M = [(-.23, my + .055), (-.10, my + .012), (0, my + .008), (.10, my + .012), (.24, my + .065), (.14, my - .045), (0, my - .055), (-.14, my - .040)]; cs = (0, 4)
            elif mood == 'shout': M = [(-.17, my + .045), (-.08, my + .075), (0, my + .080), (.08, my + .075), (.17, my + .045), (.195, my - .10), (.12, my - .165), (0, my - .175), (-.12, my - .165), (-.195, my - .10)]; cs = (0, 4, 5, 9)
            else: M = [(-.20, my - .015), (-.15, my + .045), (0, my + .055), (.15, my + .045), (.20, my - .015), (.19, my - .085), (0, my - .075), (-.19, my - .085)]; cs = (0, 1, 3, 4, 5, 7)
            P = H.P(spl(np.array(M), True, corners=cs, n=5)); tooth = rgb('#f1e9df') * pal.mul + pal.add
            cel.fill(P, tooth if mood != 'shout' else rgb('#4a141c') * pal.mul + pal.add)
            if mood == 'shout' and s > 12:
                cel.clip(P); cel.fill(H.P([(-.2, my + .1), (.2, my + .1), (.17, my + .030), (-.17, my + .030)]), tooth); cel.fill(H.P([(-.12, my - .2), (-.11, my - .11), (0, my - .085), (.11, my - .11), (.12, my - .2)]), rgb('#a03a44') * pal.mul + pal.add); cel.unclip()
            if mood == 'rage' and s > 14: cel.line(H.P([(-.19, my - .02), (0, my - .010), (.19, my - .02)]), lf, ink, taper=(.1, .1))
            kit.outline(cel, P, lf * 1.35, ink, seed=3)
            if s > 16 and mood in ('shout', 'rage') and char in ('nanami', 'todo', 'sukuna'):
                for e in (-1, 1): cel.line(H.P(spl(np.array([(e * .20, my + .33), (e * .27, my + .16), (e * .28, my - .02)]), n=5)), lf, skin[3], taper=(.3, .6))
        elif mood == 'smile': cel.line(H.P(spl(np.array([(-.17, my - .0), (0, my - .03), (.19, my + .045)]), n=6)), li, ink, taper=(.3, .15))
        else: cel.line(H.P(spl(np.array([(-.15, my - .012), (0, my + .006), (.15, my - .016)]), n=6)), li * .95, ink, taper=(.35, .35))
        if s > 26: cel.line(H.P([(-.05, -.765), (.06, -.765)]), lf, ink, taper=(.5, .5))
    # individuals
    if char in ('yuji', 'sukuna') and s > 30:
        for e in (-1, 1):
            if H.facing(.33 * e, -.15) > .15: cel.line(H.P([(e * .22, -.135), (e * .33, -.155), (e * .43, -.125)]), lf, ink if char == 'sukuna' else skin[3], taper=(.4, .4))
    if char == 'sukuna' and s > 8:
        mk = rgb('#1a1216'); wd = max(1.2, li * 1.35)
        for L in ([(0, .62), (0, .47)], [(-.10, .58), (-.05, .46), (0, .40), (.05, .46), (.10, .58)], [(-.20, -.02), (-.06, -.14), (.06, -.14), (.20, -.02)], [(-.06, -.80), (0, -.92), (.06, -.80)]): cel.line(H.P(np.array(L), .012), wd, mk, taper=(.2, .2))
        for e in (-1, 1):
            if H.facing(.5 * e, -.2) > .05:
                cel.line(H.P(spl(np.array([(e * .60, .02), (e * .50, -.10), (e * .36, -.14)]), n=5), .012), wd, mk, taper=(.1, .4)); cel.line(H.P(spl(np.array([(e * .60, -.18), (e * .46, -.30), (e * .52, -.44)]), n=5), .012), wd, mk, taper=(.1, .4))
                if s > 15: cel.line(H.P([(e * .20, -.175), (e * .32, -.155), (e * .42, -.185)], .015), li * 1.4, ink, taper=(.3, .2)); cel.fill(H.P([(e * .25, -.175), (e * .32, -.165), (e * .38, -.18), (e * .31, -.205)], .015), rgb('#b0222c'))
    if char == 'todo' and s > 10:
        cel.line(H.P(spl(np.array([(.22, .42), (.30, .20), (.36, -.16)]), n=6), .012), lf * 1.3, skin[3], taper=(.2, .4))
        if s > 18:
            for e in (-1, 1): cel.line(H.P([(e * .42, -.20), (e * .36, -.42), (e * .30, -.55)]), lf, skin[3], taper=(.3, .6))
    if char == 'nanami' and s > 18:
        for e in (-1, 1): cel.line(H.P([(e * .50, -.24), (e * .40, -.36), (e * .34, -.50)]), lf, skin[3], taper=(.4, .6))
    if expr.get('damage') and s > 12:
        dm = expr['damage']; bl = rgb('#7a1420') * pal.mul + pal.add
        cel.fill(H.P([(.22, -.30), (.46, -.22), (.52, -.36), (.36, -.50), (.26, -.44)], .004), skin[2]); cel.fill(H.P([(-.50, .30), (-.28, .40), (-.20, .30), (-.42, .20)], .004), skin[2])
        for L in ([(-.34, -.20), (-.20, -.30), (-.10, -.34)], [(.30, -.12), (.46, -.20)], [(.10, .40), (.16, .22), (.13, .06)]): cel.line(H.P(np.array(L), .008), lf * 1.3 * dm, bl, taper=(.1, .7), seed=3)
    if expr.get('tears'):
        for e in (-1, 1): cel.line(H.P(spl(np.array([(e * .33, -.08), (e * .37, -.30), (e * .34, -.62), (e * .22, -.9)]), n=6), .01), li * 1.4, '#e8f4ff', taper=(.1, .3))
