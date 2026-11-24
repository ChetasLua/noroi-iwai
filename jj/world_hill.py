"""The hill above the bay: sloping road with guard rail, and the overlook.  golden afternoon / sunset / dusk / spring noon."""
import numpy as np, cv2, math
from .paint import rgb, f32, fbm, smooth, mix, grid, Canvas, arcs, jit_color, lum
from .cam import Camera3D, rot_x, rot_y, rot_z
from .comp import blur, radial, over
from . import world as Wd, world_sky as SK
from .world import C, Scene, Wall, Plate, painter, FAR
from .world_props import *

ROOFS = ['#66728a', '#7a8090', '#94645a', '#5a6a82', '#a87c62', '#747e84', '#62749a', '#b0785c', '#848ea0', '#56627c', '#8a6a5e']
WALLS = ['#e2d8c4', '#d4cab8', '#e8e0d0', '#c8bca8', '#dcd2c8', '#bfc2c0', '#d8c8ae', '#b8a890']
LEAF = [('#4a6a3a', '#7f9a4a', '#2f4a34'), ('#3f5f3c', '#6f8f48', '#284232'), ('#5a7040', '#96a254', '#374a34')]


def house(S, x, y, z, w, d, h, rot, r, det=1, lit=0.):
    """Japanese town house: plaster box, tiled gable or hip roof with eaves, windows, balcony, aerial"""
    ca, sa = math.cos(rot), math.sin(rot); ex, ey = np.array([ca, sa, 0.]), np.array([-sa, ca, 0.]); up = np.array([0, 0, 1.]); o = np.array([x, y, z], float)
    P = lambda a, b, c: o + ex * a + ey * b + up * c
    wc, rc = C(pick(r, WALLS)) * r.uniform(.9, 1.05), C(pick(r, ROOFS)) * r.uniform(.85, 1.15); fl = 2 if h > 4.2 else 1
    W4 = [(-w / 2, -d / 2, w / 2, -d / 2), (w / 2, -d / 2, w / 2, d / 2), (w / 2, d / 2, -w / 2, d / 2), (-w / 2, d / 2, -w / 2, -d / 2)]
    for (a0, b0, a1, b1) in W4:
        A, B = P(a0, b0, 0), P(a1, b1, 0); W = Wall(S, A, B, z - 4, z + h)
        if np.dot(S.eye - W.o, W.n) < 0: continue
        W.rect(0, W.w, 0, 4 + h, 0, wc * .86, ct=wc, tex=14 if det else 0)
        if det:
            n = max(1, int(W.w / 3.0)); bw = W.w / n
            for f in range(fl):
                u = np.arange(n) * bw + bw * .24; k = r.random(n) < .8
                if k.any():
                    v = 4 + f * 2.8 + (.9 if f else .7); L = r.random(k.sum()) < lit; glass(W, u[k], u[k] + bw * .52, np.full(k.sum(), v), np.full(k.sum(), v + 1.25), .03, r, L, 1, dark=('#3a4458', '#8fa2b4'), em=.95)
                    if det > 1: W.slab(u[k] - .06, u[k] + bw * .52 + .06, v - .10, v, 0, .16, wc * .8); W.slab(u[k] - .05, u[k] + bw * .52 + .05, v + 1.25, v + 1.33, 0, .22, rc)
            if fl > 1: W.slab(0, W.w, 4 + 2.65, 4 + 2.8, 0, .06, wc * .8)
    e = .62; rh = min(w, d) * r.uniform(.22, .34); hw, hd = w / 2 + e, d / 2 + e
    kind = r.random()
    if kind < .62:                                                        # gable
        R0, R1 = P(-hw, 0, h + rh), P(hw, 0, h + rh); ev = h - e * rh / (d / 2) * .6
        S.quads(np.stack([P(-hw, -hd, ev), P(hw, -hd, ev), R1, R0])[None], rc, tex=11, d=ex); S.quads(np.stack([P(hw, hd, ev), P(-hw, hd, ev), R0, R1])[None], rc * .92, tex=11, d=ex)
        for s_ in (-1, 1): S.tris(np.array([[P(s_ * w / 2, -d / 2, h), P(s_ * w / 2, d / 2, h), P(s_ * w / 2, 0, h + rh * .93)]]), wc * .95)
        S.boxes([(R0 + R1) / 2 + up * .06], [w + 2 * e, .3, .22], rot, rc * .8)
        S.quads(np.stack([P(-hw, -hd, ev - .16), P(hw, -hd, ev - .16), P(hw, -hd, ev), P(-hw, -hd, ev)])[None], rc * .6)
    elif kind < .86:                                                      # hip
        q = min(hw, hd) * .75; R0, R1 = (P(-hw + q, 0, h + rh), P(hw - q, 0, h + rh)) if hw > hd else (P(0, -hd + q, h + rh), P(0, hd - q, h + rh)); c = [P(-hw, -hd, h - .1), P(hw, -hd, h - .1), P(hw, hd, h - .1), P(-hw, hd, h - .1)]
        if hw > hd: F = [(c[0], c[1], R1, R0), (c[1], c[2], R1, R1), (c[2], c[3], R0, R1), (c[3], c[0], R0, R0)]
        else: F = [(c[0], c[1], R0, R0), (c[1], c[2], R1, R0), (c[2], c[3], R1, R1), (c[3], c[0], R0, R1)]
        for i, f_ in enumerate(F): S.quads(np.array(f_)[None], rc * (1, .93, .9, .95)[i], tex=11)
    else:                                                                  # flat roof with parapet and washing
        S.quads(np.stack([P(-w / 2, -d / 2, h), P(w / 2, -d / 2, h), P(w / 2, d / 2, h), P(-w / 2, d / 2, h)])[None], '#9a9894', tex=1); S.boxes(P(0, 0, h + .3), [w + .1, d + .1, .6], rot, wc * .9, faces='xXyY'); S.boxes(P(w * .2, d * .1, h + 1.), [w * .25, d * .25, 2.], rot, wc * .85)
        if det:
            for k in range(3): S.quads(np.stack([P(-w * .3 + k * .9, -d * .2, h + .7), P(-w * .3 + k * .9 + .7, -d * .2, h + .7), P(-w * .3 + k * .9 + .7, -d * .2, h + 1.9), P(-w * .3 + k * .9, -d * .2, h + 1.9)])[None], pick(r, ['#f0ece4', '#a8c0d0', '#d87a6a', '#f0d890']))
    if det and r.random() < .5: p = P(r.uniform(-.3, .3) * w, r.uniform(-.2, .2) * d, h + rh * .6); S.seg([p], [p + up * 2.4], .05, '#6a6c70'); S.seg([p + up * 2.2 - ex * .6], [p + up * 2.2 + ex * .6], .04, '#6a6c70')


def bush(S, c, R, r, cols=None, n=90, flat=.75, fine=1.):
    """clump of leaf cards"""
    cols = cols or pick(r, LEAF); c = np.asarray(c, float); n = int(n * 3.2 * fine); d = r.normal(0, 1, (n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True); d[:, 2] = np.abs(d[:, 2]) * 1.0 - .15
    P = c + d * (R * r.uniform(.6, 1., (n, 1)) ** .6) * [1, 1, flat]; s = R * r.uniform(.075, .16, n) / fine ** .6; u = np.cross(d, r.normal(0, 1, (n, 3))); u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-9; v = np.cross(d, u); u = u * s[:, None]; v = v * (s * .6)[:, None]
    lev = np.clip(.42 + .5 * d[:, 2] + r.normal(0, .16, n), 0, 1); cc = np.where(lev[:, None] > .70, C(cols[1]), np.where(lev[:, None] > .34, C(cols[0]), C(cols[2]))) * r.uniform(.88, 1.12, (n, 1))
    S.quads(np.stack([P - u, P - v, P + u, P + v], 1), cc.astype(f32), lab=S.newlab(n), d=(0, 0, 1))


def guardrail(S, P, r, h=.82, z=None, col='#d6d8d4', post='#b8bab8', every=2.0):
    """white pipe-and-beam guard rail along a 3-D polyline"""
    P = np.asarray(P, float); L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]; t = np.arange(0, L[-1], every); Q = np.stack([np.interp(t, L, P[:, i]) for i in range(3)], -1)
    for q in Q: S.tube([q, q + [0, 0, h + .02]], .062, post, 7)
    t = np.linspace(0, L[-1], max(3, int(L[-1] / 1.2))); Q = np.stack([np.interp(t, L, P[:, i]) for i in range(3)], -1)
    for zz, rr in ((h * .98, .055), (h * .70, .04), (h * .42, .04)): S.tube(Q + [0, 0, zz], rr, col, 8)


def torii(S, p, w=3.0, h=3.6, rot=0., z=0., col='#b8402c'):
    """shrine gate"""
    ca, sa = math.cos(rot), math.sin(rot); ex = np.array([ca, sa, 0.]); o = np.array([*p, z]); up = np.array([0, 0, 1.])
    for s in (-1, 1): S.tube([o + ex * s * w * .54, o + ex * s * w * .48 + up * h], [.19, .16], col, 10); S.tube([o + ex * s * w * .54, o + ex * s * w * .54 + up * .5], .21, '#2a2624', 10)
    t = np.linspace(-.72, .72, 9); top = o + ex * (t * w)[:, None] + up * (h + .06 + .30 * np.abs(t)[:, None] ** 2.2); S.tube(top + up * .20, .15, '#2c2826', 8); S.tube(top, .17, col, 8); S.tube([o - ex * w * .58 + up * h * .80, o + ex * w * .58 + up * h * .80], .12, col, 8); S.boxes(o + up * h * .90, [.36, .14, h * .18], rot, col)


TIMES = dict(
    golden=dict(sky=lambda h, w, hz, s: SK.golden(h, w, hz, (.16, h * .30), s), sun=((-.70, .55, .36), '#ffc06c', 1.18), amb=('#7688b4', '#5c4a4c'), fills=[((.6, -.6, .2), '#1c2a40')], fog=dict(col='#d8cdc0', col2='#ffe0ae', dir=(-.7, .8, .05), dens=.00105, start=40, hf=0, floor=1., keep=.1, pw=5.), sea=('#9fc0c8', '#ffe9b8')),
    sunset=dict(sky=lambda h, w, hz, s: SK.sunset(h, w, hz, (.60, h * .035), s), sun=((.22, .95, .13), '#ff9c50', 1.0), amb=('#6a6aa4', '#4a3a58'), fills=[((-.7, -.5, .2), '#141c3c'), ((.2, 1, .1), '#5a2c24')], fog=dict(col='#c8889a', col2='#ffb070', dir=(.22, 1, .0), dens=.0021, start=40, hf=0, floor=1., keep=.1, pw=8.), sea=('#b87a94', '#ffcf80')),
    dusk=dict(sky=lambda h, w, hz, s: SK.bluehour(h, w, hz, s, .55), sun=None, amb=('#4a5c96', '#2a2c48'), fills=[((.1, 1, .1), '#3a2430')], fog=dict(col='#6a74ac', col2='#c89a98', dir=(.1, 1, 0), dens=.0026, start=30, hf=0, floor=1., keep=.4, pw=6.), sea=('#5a6ca0', '#e0a890')),
    noon=dict(sky=lambda h, w, hz, s: SK.day(h, w, hz, s, .7), sun=((.45, .25, .85), '#fff4d8', .78), amb=('#9ab8d8', '#6a7068'), fills=[((-.6, -.6, .2), '#182a40')], fog=dict(col='#cfe4ee', dens=.0013, start=40, hf=0, floor=1., keep=.1), sea=('#6cb8d4', '#f4fbff')))


def town(S, r, zf, pts, det_d=140, lit=0.):
    """houses on points, back to front LOD"""
    for (x, y) in pts:
        d = math.hypot(x - S.eye[0], y - S.eye[1]); w, dp = r.uniform(6.5, 12), r.uniform(6, 9.5); h = r.choice([3.2, 5.6, 5.8, 6.2, 8.4], p=[.2, .35, .25, .15, .05]); house(S, x, y, float(zf(x, y)), w, dp, h, r.choice([0, 0, math.pi / 2, .2, -.3, .5]) + r.normal(0, .1), r, 2 if d < 60 else 1 if d < det_d else 0, lit)


def apartments(S, r, zf, pts, f, lit=.0):
    for (x, y) in pts:
        z = float(zf(x, y)); w, d, h = r.uniform(14, 34), r.uniform(10, 14), r.uniform(12, 30); a = r.uniform(-.5, .5); c, s = math.cos(a), math.sin(a); q = np.array([(-w, -d), (w, -d), (w, d), (-w, d)]) / 2 @ np.array([[c, s], [-s, c]]) + [x, y]; building(S, q, h + 6, pick(r, ['resid', 'resid', 'grid', 'office']), r, S.eye, f, lit, pick(r, ['#d4ccbe', '#c4b8a8', '#b8bcc0', '#d8d2c4']), z0=z - 6, detmax=1)


def far_land(S, r, z0, spec):
    """distant ridges as layered flat silhouettes: spec=[(dist, az0, az1, height, col, seed)]"""
    for (d, a0, a1, hh, col, sd) in spec:
        rr = np.random.default_rng(sd); a = np.radians(np.linspace(a0, a1, 60)); t = np.linspace(0, 1, 60); n = sum(np.interp(t, np.linspace(0, 1, m), rr.random(m)) * k for m, k in ((5, .6), (11, .28), (25, .12))); env = np.sin(np.pi * np.clip(t, 0, 1)) ** .45; H_ = hh * n * env
        x, y = np.sin(a) * d, np.cos(a) * d; A = np.c_[x, y, np.full(60, z0 - 3.)]; B = np.c_[x, y, z0 + H_]; S.quads(np.stack([A[:-1], A[1:], B[1:], B[:-1]], 1), np.stack([np.stack([C(col) * 1.06, C(col) * 1.06, C(col) * .96, C(col) * .96])] * 59), snow=0, lab=S.newlab(), d=(1, 0, 0))


def sea_layer(img, cam, zs, res, horizon, cols, sunx, seed, glit=1., mask=None):
    """painted water plane with ripple strokes and the sun's road"""
    h, w = img.shape[:2]; y0 = int(np.clip(horizon, 0, h - 1)); m = mask
    x, y = grid(h, w); q = np.clip((y - horizon) / (h * .16), 0, 1)
    col = C(cols[0]) * (1.06 - .22 * q)[..., None]; col = col + (C(cols[1]) - col) * (np.exp(-q * 9) * .55)[..., None]
    n = fbm(h, w, (120, 40, 14), (1, .6, .4), seed + 2, aspect=9.)
    col = col * (1 + .05 * n)[..., None]
    ys = np.clip(y - horizon, 0, None); half = w * .012 + ys * .42; sr = np.exp(-((x - sunx) / half) ** 2); r = np.random.default_rng(seed); sp = cv2.GaussianBlur(r.standard_normal((h, w)).astype(f32), (0, 0), sigmaX=5, sigmaY=.8); sp /= sp.std(); spk = smooth(.2, 1.6, sp + 1.6 * sr - .6)
    col = col + (C(cols[1]) * 1.15 - col) * (np.clip(sr * .55 + spk * sr, 0, 1) * glit)[..., None] + C('#ffffff') * (smooth(1.2, 2.2, sp + sr) * sr * .5 * glit)[..., None]
    return img * (1 - m[..., None]) + col * m[..., None]


def _finish(S, cam, res, sky, seed, w, h, ao=.36, paint=True, sizes=(17, 8.5, 4.2), soft=True, opk=1., valk=1.):
    geo = (res['z'] < FAR * .5).astype(f32); img = res['img'] * (1 + (Wd.occlusion(res['z'], ao, (3, 9, 26)) - 1) * geo)[..., None] + sky * (1 - res['a'][..., None])
    if paint:
        img = Wd.wobble(img, .5, 30, seed + 12); img = Wd.gouache(img, res['lab'], res['ang'] + fbm(h, w, (140, 50), (1, .5), seed + 8) * .14, seed=seed + 4, sizes=sizes, op=(.6 * opk, .55 * opk, .5 * opk), val=(.044 * valk, .038 * valk, .032 * valk), hue=.016 * valk, thr=(.016, .030, .055), mask=geo); img = Wd.tooth_pass(img, seed + 6)
    return img, geo


@painter('hillroad', 4)
def hillroad(w=2400, h=1350, seed=0, time='golden', eye=(.9, -1.5, 1.45), target=(-1.25, 12., 2.55), focal=1480., paint=True, ss=2):
    """road falling toward us from a bend: guard rail and the drop to the sea of roofs on our left, stone wall, steps and torii on the right, poles and wires.
    road surface z = 0.085*y (P.extra slope), width x in (-3.1, 3.2); project walkers with P.cam.  masks: sky, ground.  layers 'rail' none"""
    T = TIMES[time]; cam = Camera3D(eye, target, focal, w, h); S = Scene(seed + 3); r = np.random.default_rng(seed + 23); S.eye = np.asarray(eye, float); S.groups = 4
    sl = .085; bend = lambda y: .0105 * np.maximum(y - 14, 0) ** 1.55        # road swings right-to-left up the hill
    cx = lambda y: -bend(y); rz = lambda y: sl * np.asarray(y, float)
    Y = np.linspace(-6, 64, 71); xl, xr = cx(Y) - 3.1, cx(Y) + 3.2
    def rib(off0, off1, dz, col, tex=0, **k):
        a = np.c_[cx(Y) + off0, Y, rz(Y) + dz]; b = np.c_[cx(Y) + off1, Y, rz(Y) + dz]; S.quads(np.stack([a[:-1], b[:-1], b[1:], a[1:]], 1), col, tex=tex, d=(0, 1, 0), **k)
    rib(-3.1, 3.2, 0., '#7d7a7a', 2, seed=31); rib(3.2, 3.75, .02, '#a8a49c', 1); rib(-3.65, -3.1, .05, '#b0aaa0', 1); rib(-2.85, -2.72, .012, '#d8d4c8', 8, bias=4e-4); rib(2.85, 2.98, .012, '#d8d4c8', 8, bias=4e-4); rib(3.36, 3.6, .03, '#5a5652', 0, bias=4e-4)
    # anti-slip rings cast in the concrete of the steep part
    for yy in np.arange(-.5, 12., .42):
        for xx in np.arange(-2.3, 2.4, .42):
            o = xx + (.21 if int(yy / .42) % 2 else 0); a = np.linspace(0, 2 * np.pi, 9); q0, q1 = .085, .125; c0 = np.array([cx(yy) + o, yy, rz(yy) + .006]); e = np.stack([np.cos(a), np.sin(a), sl * np.sin(a)], -1); S.quads(np.stack([c0 + e[:-1] * q0, c0 + e[:-1] * q1, c0 + e[1:] * q1, c0 + e[1:] * q0], 1), '#646164', bias=3e-4, lab=1, snow=0)
    # right: stone retaining wall, coping, hedge, steps + torii
    ys = np.linspace(-6, 64, 36)
    for a, b in zip(ys[:-1], ys[1:]):
        if 12.6 < (a + b) / 2 < 16.4: continue
        A, B = np.array([cx(a) + 3.78, a]), np.array([cx(b) + 3.78, b]); z0, z1 = rz(a), rz(b); hw = 2.5 + .5 * math.sin(a * .3); hw2 = 2.5 + .5 * math.sin(b * .3)
        S.quads(np.array([[*B, z1], [*A, z0], [A[0] + .35, A[1], z0 + hw], [B[0] + .35, B[1], z1 + hw2]])[None], '#9a8f82', tex=4, d=(0, 1, 0)); S.quads(np.array([[A[0] + .30, A[1], z0 + hw + .12], [B[0] + .30, B[1], z1 + hw2 + .12], [B[0] + .9, B[1], z1 + hw2 + .12], [A[0] + .9, A[1], z0 + hw + .12]])[None], '#b8b0a4', tex=1); S.quads(np.array([[B[0] + .30, B[1], z1 + hw2 - .06], [A[0] + .30, A[1], z0 + hw - .06], [A[0] + .30, A[1], z0 + hw + .12], [B[0] + .30, B[1], z1 + hw2 + .12]])[None], '#c4bcae')
        for k in range(3): p = (cx(a) + 4.9 + r.uniform(-.4, 1.6), a + r.uniform(0, 2), z0 + hw + r.uniform(.1, .6)); bush(S, p, r.uniform(.6, 1.15), r, n=80)
        if r.random() < .5: bush(S, (cx(a) + 4.0, a, z0 + hw + .05), r.uniform(.3, .55), r, n=50, flat=.6)
    S.ground(3.9, 60, -10, 90, lambda X_, Y_: rz(Y_) + 2.6 + .42 * np.maximum(X_ - cx(Y_) - 4, 0), '#5a6a40', nx=14, ny=30, tex=10)
    for k in range(20): y_ = r.uniform(2, 60); x_ = cx(y_) + r.uniform(6.5, 18); x_ = x_ + 7 if 11 < y_ < 27 else x_; zz = rz(y_) + 2.6 + .42 * (x_ - cx(y_) - 4); tree(S, (x_, y_), r.uniform(5, 9), r, zz, pick(r, LEAF), n=90 if y_ < 24 else 40, leaf=.30)
    for k in range(13): z0 = rz(14.4) + k * .17; S.boxes([cx(14.4) + 3.9 + k * .34, 14.5, z0 + .0], [.5, 3.0, .5], 0, '#a39a8e', top='#c0b8ac', tex=1)
    x14, z14 = cx(14.4), rz(14.4); S.boxes([[x14 + 9.3, 14.5, z14 + 2.2], [x14 + 6.6, 12.45, z14 + 1.5], [x14 + 6.6, 16.55, z14 + 1.6], [x14 + 6.4, 14.5, z14 - .2]], [[1., 5., 6.], [5.4, .4, 3.6], [5.4, .4, 3.8], [4.6, 4., 1.]], 0, '#a0968a', tex=4)
    torii(S, (cx(25) + 6.6, 25.), 3.6, 4.4, .12, rz(25) + 2.5); 
    for s_ in (-1, 1): S.boxes([cx(14.4) + 5.0, 14.5 + s_ * 1.75, rz(14.4) + 1.6], [.42, .42, 1.5], 0, '#8f877c', tex=1); S.boxes([cx(14.4) + 5.0, 14.5 + s_ * 1.75, rz(14.4) + 2.5], [.8, .8, .22], 0, '#7c756a'); S.boxes([cx(14.4) + 5.0, 14.5 + s_ * 1.75, rz(14.4) + 2.2], [.5, .5, .42], 0, '#e8dcc0')
    # left: guard rail, verge grass, the drop, rooftops, bay
    guardrail(S, np.c_[cx(Y) - 3.4, Y, rz(Y) + .04], r)
    zf = lambda x, y: np.maximum(rz(np.clip(y, -20, 90)) * .30 - 11.0 - .046 * np.maximum(-np.asarray(x, float) - 12, 0) - .040 * np.maximum(np.asarray(y, float) - 34, 0), -40.)
    S.ground(-520, -3.6, -40, 520, lambda X_, Y_: np.where(X_ > -4.2, rz(np.clip(Y_, -6, 64)) - .15, zf(X_, Y_) - .3), '#6a6e58', nx=60, ny=40, pw=1.5, tex=10)
    for k in range(90): y_ = r.uniform(-1, 46); x_ = cx(y_) - r.uniform(3.8, 5.6); bush(S, (x_, y_, rz(y_) - .35 - .5 * (-(x_ - cx(y_)) - 3.8)), r.uniform(.25, .7), r, n=46, flat=.8)
    n = 900; th = np.radians(r.uniform(-74, 8, n)); d = 31 + 420 * r.random(n) ** 1.25; px, py = np.sin(th) * d, np.cos(th) * d; k = (px < cx(np.maximum(py, 0)) - 12) & (px > -440); P = np.c_[px, py][k]
    keep = [];
    for p in P:
        if all(np.hypot(*(p - q)) > 8.8 + .010 * np.hypot(*p) for q in keep[-260:]): keep.append(p)
    P = np.array(sorted(keep, key=lambda t: -np.hypot(*t))); town(S, r, zf, P, lit=.05 if time == 'golden' else .25 if time == 'sunset' else .5)
    for k in range(120): th_ = math.radians(r.uniform(-74, 4)); d_ = r.uniform(24, 260); x_, y_ = math.sin(th_) * d_, math.cos(th_) * d_
    for p in P[::3]: x_, y_ = p + r.normal(0, 5, 2); bush(S, (x_, y_, float(zf(x_, y_)) + 1.2), r.uniform(1.6, 3.2), r, n=40, flat=1.)
    apartments(S, r, zf, [(-190, 250), (-105, 330), (-300, 190), (-46, 300)], cam.f, .1)
    far_land(S, r, -40., [(1500, -64, -24, 120, '#7c8fb0', 4), (1250, -22, 12, 110, '#6f86a6', 7), (2400, -90, 10, 210, '#8f9fbe', 9)])
    # poles and wires
    an = []
    for (x_, y_, hgt) in ((3.55, 4.5, 9.4), (-3.75, 13.5, 9.), (3.6, 23., 9.2), (-3.8, 33, 9.), (3.6, 44, 9.), (3.4, 58, 9)): an.append(pole(S, (cx(y_) + x_, y_), hgt, r, rz(y_), ((1, .1),), '#948c84'))
    a0 = [np.array([4.6, -9., 9.2]) + [0, 0, .25 * j] for j in range(8)]
    for i in range(len(an) - 1):
        for j in (0, 1, 2, 3, 5, 6): S.wire(an[i][j], an[i + 1][j], r.uniform(.25, .7), .022, '#33323a', min_px=1.1)
    for j in (0, 1, 2, 3, 5, 6): S.wire(a0[j], an[0][j], .5, .022, '#33323a', min_px=1.2)
    # convex mirror, warning sign plate (blank pictogram), drain covers
    mp = np.array([cx(26) - 3.7, 26., rz(26)]); S.tube([mp, mp + [0, 0, 2.6]], .05, '#d87a2a', 7); Wm = Wall.frame(S, mp + [-.45, -.12, 2.25], (1, 0, 0), (0, 0, 1), .9, .9); Wm.disc(.45, .45, .5, 0, '#e07a28'); Wm.disc(.45, .45, .43, .02, '#b8cad4', em=.2); Wm.poly([(.1, .3), (.8, .3), (.7, .12), (.2, .12)], .03, '#6a7a6a')
    hz = Wd.horizon_y(cam); sky = T['sky'](h, w, hz + 6, seed + 2)
    nk = dict(amb=T['amb'], fills=T['fills'], fog=T['fog'])
    if T['sun'] is not None: nk['sun'] = T['sun']
    res = S.render(cam, ss=ss, smres=4096, shadow_box=(-16, 14, -8, 60, 16), **nk)
    sm = (1 - res['a']); rows = np.arange(h, dtype=f32)[:, None]; seam = sm * (rows > hz + .5)
    sky = sea_layer(sky, cam, -40, res, hz, T['sea'], w * .10, seed + 5, 1., np.clip(seam, 0, 1))
    img, geo = _finish(S, cam, res, sky, seed, w, h, .34, paint, valk=.6)
    img = Wd.air_glow(img, res['z'], .78, (60, 170), .24, (20, 200), '#ffd898')
    return Plate(img, res['z'], cam, dict(sky=1 - res['a']), dict(sky=np.dstack([sky, np.ones((h, w), f32)]).astype(f32)), extra=dict(slope=sl))


@painter('overlook', 3)
def overlook(w=2400, h=1800, seed=0, time='sunset', paint=True, ss=2, hz=1120., focal=1500., sunx=.60):
    """the guard rail above the bay, huge sky.  tall plate for a crane-up.  eye height 1.9 m at (0,0); rail at y=3.2; ground z=0 near.
    layers: 'rail' (rail, verge grass, pole) to put IN FRONT of / behind people; img has no rail.  masks: sky, sea"""
    T = TIMES[time]; cam = Camera3D((0, 0, 1.9), (0, 10, 1.9), focal, w, h, cy=hz); S = Scene(seed + 5); r = np.random.default_rng(seed + 29); S.eye = cam.eye; S.groups = 4
    zf = lambda x, y: np.maximum(-5. - 11. * smooth(5, 26, np.asarray(y, float)) - .082 * np.maximum(np.asarray(y, float) - 20, 0) + .010 * np.abs(np.asarray(x, float) + 40) + 2.4 * np.sin(np.asarray(x, float) * .021 + 1) * np.exp(-np.asarray(y, float) / 260.), -52.)
    S.ground(-700, 700, 3.6, 620, lambda X_, Y_: np.where(Y_ < 4.6, -.35, zf(X_, Y_) - .3), '#5f6650', nx=50, ny=46, pw=1.7, tex=10)
    n = 1500; th = np.radians(r.uniform(-48, 48, n)); d = 34 + 430 * r.random(n) ** 1.2; P = np.c_[np.sin(th) * d, np.cos(th) * d]; shore = lambda x: 400 + 70 * np.sin(x * .008 + .6) + 30 * np.sin(x * .023)
    P = P[P[:, 1] < shore(P[:, 0]) - 8]; keep = []
    for p in P:
        if all(np.hypot(*(p - q)) > 8.6 + .014 * np.hypot(*p) for q in keep): keep.append(p)
    P = np.array(sorted(keep, key=lambda t: -np.hypot(*t))); town(S, r, zf, P, 160, .28)
    for p in P[::2]: x_, y_ = p + r.normal(0, 5, 2); bush(S, (x_, y_, float(zf(x_, y_)) + 1.0), r.uniform(1.6, 3.4), r, n=34, flat=1.)
    apartments(S, r, zf, [(-150, 270), (-128, 300), (208, 290), (250, 330), (-66, 372)], cam.f, .25)
    for k in range(64): y_ = r.uniform(7, 30); x_ = r.uniform(-1.1, 1.1) * (y_ + 4); bush(S, (x_, y_, float(zf(x_, y_)) + r.uniform(.8, 2.6)), r.uniform(1.2, 2.8), r, n=150, flat=.9, fine=2.6)
    # a wooded shrine knoll, harbour cranes, breakwater with a lamp
    for k in range(46): a_, q_ = r.uniform(0, 6.28), r.random() ** .5 * 24; x_, y_ = 92 + math.cos(a_) * q_ * 1.5, 150 + math.sin(a_) * q_; bush(S, (x_, y_, float(zf(x_, y_)) + (1 - q_ / 24) * 11 + 1), r.uniform(3, 5), r, n=70, flat=1.)
    far_land(S, r, -52., [(1500, 16, 60, 250, '#7a6a90', 4), (2300, -62, -14, 260, '#8a7898', 7), (3400, -50, 60, 190, '#a088a0', 12), (900, 34, 64, 150, '#64587e', 15)])
    for (x_, hgt) in ((-180, 34), (-160, 30)): S.seg([[x_, 470, -52], [x_ + 10, 470, -52], [x_ - 14, 470, -52 + hgt]], [[x_ + 2, 470, -52 + hgt], [x_ + 6, 470, -52 + hgt], [x_ + 30, 470, -52 + hgt + 7]], 1.2, '#6a5a70')
    S.boxes([120, 520, -51], [180, 4, 2.4], .12, '#7a6c80'); S.boxes([212, 530, -47], [2.4, 2.4, 9], 0, '#e8ddd0'); S.boxes([212, 530, -42], [1.6, 1.6, 1.6], 0, '#ffe0a0', em=1.5)
    for k in range(9): x_, y_ = r.uniform(-260, 300), r.uniform(470, 900); S.boxes([x_, y_, -51], [r.uniform(8, 22), 3, 2.], r.uniform(-.4, .4), '#5a5070'); S.boxes([x_ + 1, y_, -49], [3, 2, 2.6], 0, '#d8c8c0')
    # poles on the slope with wires crossing low
    an = []
    for (x_, y_, hgt) in ((-44, 34, 12), (-13, 22, 15.), (21, 27, 15.), (58, 40, 14)): an.append(pole(S, (x_, y_), hgt, r, float(zf(x_, y_)) + 1.2, ((1, .25),), '#6a6070'))
    for i in range(3):
        for j in (0, 1, 2, 3, 5, 6): S.wire(an[i][j], an[i + 1][j], r.uniform(.5, 1.1), .025, '#3a3046', min_px=1.1)
    sky = T['sky'](h, w, hz + 4, seed + 2); nk = dict(amb=T['amb'], fills=T['fills'], fog=T['fog'])
    if T['sun'] is not None: nk['sun'] = T['sun']
    res = S.render(cam, ss=ss, smres=4096, shadow_box=(-90, 90, 0, 190, 4), **nk)
    sm = (1 - res['a']); rows = np.arange(h, dtype=f32)[:, None]; seam = np.clip(sm * (rows > hz + .5), 0, 1)
    sky = sea_layer(sky, cam, -52, res, hz, T['sea'], w * sunx, seed + 5, 1., seam); img, geo = _finish(S, cam, res, sky, seed, w, h, .34, paint)
    img = Wd.air_glow(img, res['z'], .80, (70, 190), .22, (40, 300), '#ffd0a0')
    # foreground rail layer
    F = Scene(seed + 9); F.r = np.random.default_rng(seed + 31); rr = F.r; F.eye = cam.eye; x = np.linspace(-9, 9, 19)
    F.ground(-9, 9, 1.0, 3.7, 0., '#8a8078', nx=10, ny=6, tex=1); F.quads(np.array([[-9, 3.0, .0], [9, 3.0, .0], [9, 3.0, .14], [-9, 3.0, .14]])[None], '#a89c90', tex=1); F.quads(np.array([[-9, 3.0, .14], [9, 3.0, .14], [9, 3.5, .14], [-9, 3.5, .14]])[None], '#b8aca0', tex=1)
    guardrail(F, np.c_[x, np.full(19, 3.2), np.full(19, .12)], rr, .86)
    for k in range(420): bx, by = rr.uniform(-8, 8), rr.uniform(3.4, 3.9); b = np.array([bx, by, -.1]); t = b + [rr.normal(0, .14), rr.normal(0, .05), rr.uniform(.2, .62)]; m = (b + t) / 2 + [rr.normal(0, .03), 0, .05]; F.seg([b, m], [m, t], .018, pick(rr, ['#6a7044', '#8a8a4c', '#4a5638', '#a89a58']), min_px=1.6)
    nk2 = dict(nk); res2 = F.render(cam, ss=ss, smres=2048, **nk2); a2 = res2['a']; st = np.where(a2[..., None] > .01, res2['img'] / np.maximum(a2[..., None], .01), 0)
    if paint: st = Wd.gouache(st, res2['lab'], res2['ang'], seed=seed + 24, sizes=(12, 6, 3), op=(.5, .5, .4), val=(.04, .036, .03), hue=.014, mask=(a2 > .5).astype(f32))
    return Plate(img, res['z'], cam, dict(sky=1 - res['a'], sea=seam), dict(rail=np.dstack([st * a2[..., None], a2]).astype(f32)))
