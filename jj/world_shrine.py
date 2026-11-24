"""Malevolent Shrine in ink: built in 3-D (tiers, horns, ribs, mouths, skulls), then inked: blacks, bone, blood, hatching, dry brush."""
import numpy as np, cv2, math
from .paint import rgb, f32, fbm, smooth, mix, grid, Canvas, arcs, jit_color, lum, warp_by, ramp
from .cam import Camera3D, rot_x, rot_y, rot_z
from .comp import blur, radial, over
from . import world as Wd
from .world import C, Scene, Wall, Plate, painter, FAR
from .world_props import *

INK, BONE, BONE2, RED, RED2, RED3, CHAR = [C(c) for c in ('#0a070a', '#d8cbb2', '#8f7f6a', '#a3141e', '#5c0d14', '#d8262c', '#2a2428')]


def tooth(S, p, d, L, w, r, col=BONE, n=6, curve=.15):
    """one fang: tapered bent cone from p along d"""
    d = np.asarray(d, float); d /= np.linalg.norm(d); s = np.cross(d, [.3, .9, .2]); s /= np.linalg.norm(s); t = np.linspace(0, 1, 5); b = r.normal(0, curve) * L
    P = np.asarray(p, float) + d * (t * L)[:, None] + s * (b * t ** 2)[:, None]; S.tube(P, w * np.array([1., .95, .75, .45, .04]), np.stack([col * k for k in (.55, .75, .95, 1.05, 1.1)]), n)


def mouth(S, W, u, v, a, b, r, d=.05, n=9, lip=.12, open_=1., tongue=False):
    """a mouth in a Wall frame: half-axes a (across) b (up); lips, black throat, uneven teeth from both jaws"""
    th = np.linspace(0, 2 * np.pi, 25); ring = W.p(u + np.cos(th) * a, v + np.sin(th) * b, d); W.disc(u, v, a, d * .4, INK, 24, b, snow=0)
    W.disc(u, v - b * .1, a * .72, d * .5, RED2 * .5, 18, b * .62)
    S.tube(ring, lip * (1 + .25 * np.sin(th * 3 + r.uniform(0, 6))), RED * .75, 6)
    if a > b:
        for sg in (1, -1):
            x = np.linspace(-.82, .82, n) + r.normal(0, .03, n)
            for k, q in enumerate(x): L = b * r.uniform(.55, .95) * (1.25 if abs(abs(q) - .5) < .12 else 1.) * (1 - .35 * q * q); tooth(S, W.p(u + q * a, v + sg * b * math.sqrt(max(0, 1 - q * q)), d * 1.2), W.ez * -sg + W.n * .25 + W.ex * r.normal(0, .10), L, a / n * r.uniform(.6, .85), r)
    else:
        for sg in (1, -1):
            y = np.linspace(-.82, .82, n) + r.normal(0, .03, n)
            for q in y: L = a * r.uniform(.55, 1.) * (1 - .35 * q * q); tooth(S, W.p(u + sg * a * math.sqrt(max(0, 1 - q * q)), v + q * b, d * 1.2), W.ex * -sg + W.n * .25 + W.ez * r.normal(0, .12), L, b / n * r.uniform(.6, .85), r)


def horn(S, p, d0, d1, L, w, r, col=BONE, n=9, seg=14, ring=True):
    """a great horn: starts along d0, sweeps into d1"""
    d0, d1 = [np.asarray(q, float) / np.linalg.norm(q) for q in (d0, d1)]; t = np.linspace(0, 1, seg); D = d0 * (1 - t[:, None] ** 1.1) + d1 * t[:, None] ** 1.1; D /= np.linalg.norm(D, axis=1, keepdims=True); P = np.asarray(p, float) + np.cumsum(D * L / seg, 0); rad = w * (1 - t) ** .8 * (1 + (.07 * np.sin(t * 60) if ring else 0)) + .015
    S.tube(np.vstack([[p], P]), np.r_[w * 1.02, rad], np.stack([col * k for k in np.linspace(.42, 1.08, seg + 1)]), n)


def rib(S, base, top, bulge, w, r, col=BONE, seg=18):
    """curved rib from base to top, bulging outward"""
    base, top, bulge = [np.asarray(q, float) for q in (base, top, bulge)]; t = np.linspace(0, 1, seg)[:, None]; P = (1 - t) ** 3 * base + 3 * (1 - t) ** 2 * t * (base * .55 + top * .1 + bulge) + 3 * (1 - t) * t * t * (top * .75 + base * .2 + bulge * .9) + t ** 3 * top
    rad = w * (.75 + .5 * np.sin(np.pi * t[:, 0] ** .8)) * np.where(t[:, 0] > .8, 1 - (t[:, 0] - .8) * 2.6, 1); S.tube(P, rad, np.stack([col * k for k in np.linspace(.40, 1.05, seg)]), 9)
    return P


def skull(S, c, s, yaw, r, col=BONE, pitch=0., roll=0., det=1, jaw=True):
    """human skull: cranium, cheeks, sockets, nose, teeth"""
    R = rot_z(yaw) @ rot_x(pitch) @ rot_y(roll); c = np.asarray(c, float); nu, nv = (9, 6) if det else (6, 4)
    a = np.linspace(0, 2 * np.pi, nu + 1); b = np.linspace(-.5 * np.pi, .5 * np.pi, nv + 1); V = np.stack([np.cos(b)[:, None] * np.cos(a)[None] * .42, np.cos(b)[:, None] * np.sin(a)[None] * .52 + .06, np.broadcast_to(np.sin(b)[:, None], (nv + 1, nu + 1)) * .46 + .12], -1); V = V @ R.T * s + c
    lab = S.newlab(); S.quads(np.stack([V[:-1, :-1], V[:-1, 1:], V[1:, 1:], V[1:, :-1]], 2).reshape(-1, 4, 3), col * r.uniform(.8, 1.05), lab=lab)
    P = lambda x, y, z: np.array([x, y, z]) @ R.T * s + c
    f = -.44
    if jaw: S.quads(np.array([P(-.24, f + .03, -.16), P(.24, f + .03, -.16), P(.20, f + .01, -.46), P(-.20, f + .01, -.46)])[None], col * .85, lab=lab); S.quads(np.array([P(-.24, f + .03, -.16), P(-.20, f + .01, -.46), P(-.24, .1, -.30), P(-.36, .1, .0)])[None], col * .6, lab=lab); S.quads(np.array([P(.24, f + .03, -.16), P(.20, f + .01, -.46), P(.24, .1, -.30), P(.36, .1, .0)])[None], col * .6, lab=lab)
    for sg in (-1, 1): q = np.linspace(0, 2 * np.pi, 9)[:-1]; S.poly(np.array([P(sg * .175 + math.cos(k) * .115, f - .035, .07 + math.sin(k) * .125) for k in q]), INK, bias=2e-3, snow=0)
    S.tris(np.array([[P(-.06, f - .05, -.14), P(.06, f - .05, -.14), P(0, f - .06, .0)]]), INK, bias=2e-3)
    if jaw:
        for k in range(6): x = -.16 + k * .064; S.seg([P(x, f - .03, -.22)], [P(x, f - .03, -.40)], .02 * s, INK, bias=2e-3, min_px=.8)
        S.seg([P(-.18, f - .03, -.31)], [P(.18, f - .03, -.31)], .022 * s, INK, bias=2e-3, min_px=.8)


def beast_skull(S, c, s, r, yaw=0., horns=1.6, col=BONE):
    """horned ox-like skull that crowns gables"""
    R = rot_z(yaw); c = np.asarray(c, float); P = lambda x, y, z: np.array([x, y, z]) @ R.T * s + c; D = lambda x, y, z: np.array([x, y, z]) @ R.T
    prof = [(0., .50, .62, .25), (-.35, .48, .50, .05), (-.7, .34, .36, -.18), (-1.05, .26, .28, -.42), (-1.3, .22, .20, -.60)]; ring = []
    for (y, wx, wz, zc) in prof: a = np.linspace(0, 2 * np.pi, 11)[:-1]; ring.append(np.array([P(math.cos(q) * wx * .62, y, zc + math.sin(q) * wz * .62) for q in a]))
    lab = S.newlab()
    for A, B in zip(ring[:-1], ring[1:]): S.quads(np.stack([A, np.roll(A, -1, 0), np.roll(B, -1, 0), B], 1), col, lab=lab)
    S.poly(ring[0], col * .9, lab=lab); S.poly(ring[-1], col * .6, lab=lab)
    for sg in (-1, 1):
        q = np.linspace(0, 2 * np.pi, 9)[:-1]; S.poly(np.array([P(sg * (.26 + .03 * math.sin(k)), -.48 + math.cos(k) * .13, .18 + math.sin(k) * .12) for k in q]) + D(0, 0, .0), INK, bias=4e-3); S.poly(np.array([P(sg * .10 + math.cos(k) * .05, -1.36, -.56 + math.sin(k) * .07) for k in q]), INK, bias=3e-3)
        horn(S, P(sg * .28, .0, .42), D(sg * 1., -.05, .25), D(sg * -.25, -.15, 1.), horns * s, .17 * s, r, col)
    for k in range(7): x = -.17 + k * .057; tooth(S, P(x, -1.22, -.70), D(0, -.1, -1), .22 * s, .035 * s, r, col)


def roof(S, cx, cy, w, d, z, h, r, flare=1.6, over_=2.4, tile='#1c171a', rib_col=None, ridge=.42, horns=4.5, rafters=True):
    """hipped temple roof with swept eaves: concave tiles, ribs of tile, teeth along the eave, rafters like ribs, horns at the corners"""
    W, D = w / 2 + over_, d / 2 + over_; nt, nu = 9, 22; t = np.linspace(0, 1, nt); tile = C(tile)
    def side(ax, sg):
        L = (W if ax == 0 else D); O = (D if ax == 0 else W); Lr = max(.4, L - (O - .3) * 1.0) if L > O else .3 if L < O else .3
        rg_l = w / 2 * ridge if ax == 0 else .25; rg_o = .25 if ax == 0 else w / 2 * ridge
        u = np.linspace(-1, 1, nu); V = np.zeros((nt, nu, 3))
        for i, q in enumerate(t):
            k = q ** .92; l_ = L * (1 - k) + rg_l * k; o_ = O * (1 - k) + rg_o * k; zz = z + h * (q ** 1.75) + flare * np.abs(u) ** 3.4 * (1 - q) ** 2.2; a = u * l_; b = np.full(nu, o_ * sg) - sg * .5 * np.abs(u) ** 2 * 0
            V[i] = np.c_[cx + a, cy + b, zz] if ax == 0 else np.c_[cx + b, cy + a, zz]
        return V
    for ax, sg in ((0, -1), (0, 1), (1, -1), (1, 1)):
        V = side(ax, sg); S.quads(np.stack([V[:-1, :-1], V[:-1, 1:], V[1:, 1:], V[1:, :-1]], 2).reshape(-1, 4, 3), tile, lab=S.newlab(), d=(0, 0, 1))
        for j in range(0, nu):                                             # tile ribs
            S.tube(V[:, j] + [0, 0, .06], np.linspace(.16, .09, nt), tile * 1.5, 5)
        e = V[0]; lo = e - [0, 0, .42]; S.quads(np.stack([lo[:-1], lo[1:], e[1:], e[:-1]], 1), RED2 * 1.2, lab=S.newlab()); S.tube(e, .17, tile * 2.0, 6)
        n = int((W if ax == 0 else D) * 2 / .62)                           # teeth hang from the eave
        q = np.linspace(.02, .98, n); ii = q * (nu - 1); i0 = np.clip(ii.astype(int), 0, nu - 2); P = e[i0] + (e[i0 + 1] - e[i0]) * (ii - i0)[:, None]
        for p in P: tooth(S, p - [0, 0, .36], [r.normal(0, .06), r.normal(0, .06), -1], r.uniform(.45, .95), .13, r, BONE * r.uniform(.8, 1.05), 5)
        if rafters:                                                         # rafters: bone ribs under the eave
            m = int(n * .7); q = np.linspace(.04, .96, m); ii = q * (nu - 1); i0 = np.clip(ii.astype(int), 0, nu - 2); P = e[i0] + (e[i0 + 1] - e[i0]) * (ii - i0)[:, None]; inn = P.copy(); oi = over_ * 1.05
            if ax == 0: inn[:, 1] -= sg * oi; inn[:, 0] = cx + (inn[:, 0] - cx) * (w / 2) / W
            else: inn[:, 0] -= sg * oi; inn[:, 1] = cy + (inn[:, 1] - cy) * (d / 2) / D
            inn[:, 2] = z - .9
            for a, b in zip(P, inn): S.tube([a - [0, 0, .38], (a + b) / 2 - [0, 0, .25], b], [.09, .13, .10], BONE2 * r.uniform(.5, .8), 5)
            S.quads(np.stack([e[:-1] - [0, 0, .40], e[1:] - [0, 0, .40], np.interp(np.arange(1, nu), np.arange(m) * (nu - 1) / (m - 1), np.arange(m))[:, None] * 0 + inn[np.clip(((np.arange(1, nu)) * (m - 1) / (nu - 1)).astype(int), 0, m - 1)], inn[np.clip((np.arange(nu - 1) * (m - 1) / (nu - 1)).astype(int), 0, m - 1)]], 1), INK * 1.6, lab=S.newlab())
    for sx in (-1, 1):
        for sy in (-1, 1):
            p = np.array([cx + sx * W, cy + sy * D, z + flare]); dv = np.array([sx * W, sy * D, 0.]); dv /= np.linalg.norm(dv); S.tube([[cx + sx * w / 2 * ridge, cy + sy * .25, z + h], [cx + sx * (W * .55), cy + sy * (D * .52), z + h * .40], p], [.22, .24, .26], tile * 1.8, 7)
            horn(S, p - dv * .3, dv + [0, 0, .25], [dv[0] * .15, dv[1] * .15, 1.], horns, .34, r)
    S.tube([[cx - w / 2 * ridge - .6, cy, z + h + .25], [cx, cy, z + h + .12], [cx + w / 2 * ridge + .6, cy, z + h + .25]], .36, tile * 1.7, 8)
    for sx in (-1, 1): horn(S, (cx + sx * (w / 2 * ridge + .4), cy, z + h + .3), (sx, 0, .5), (-sx * .2, 0, 1), horns * .8, .30, r)


def body(S, cx, cy, w, d, z0, z1, r, pil=6, mouths=True, gate=False, eye=None):
    """a storey: black wall, red pillars with mouths, lattice, bands of bone"""
    S.boxes([cx, cy, (z0 + z1) / 2], [w, d, z1 - z0], 0, CHAR * .55, tex=1)
    hh = z1 - z0; xs = np.linspace(-w / 2, w / 2, pil)
    for k, x in enumerate(xs):
        rr = min(.95, w / pil * .24); S.tube([[cx + x, cy - d / 2 - .25, z0 - .2], [cx + x, cy - d / 2 - .25, (z0 + z1) / 2], [cx + x, cy - d / 2 - .25, z1]], [rr * 1.12, rr, rr * .94], RED2 * 1.25, 12); S.cyl((cx + x, cy - d / 2 - .25), rr * 1.35, z0 - .25, z0 + .3, BONE2 * .6, 12)
        if mouths and hh > 3 and not (gate and abs(x) < w * .16):
            W = Wall.frame(S, (cx + x - rr, cy - d / 2 - .25 - rr * .98, z0), (1, 0, 0), (0, 0, 1), 2 * rr, hh); mouth(S, W, rr, hh * r.uniform(.42, .62), rr * .62, hh * r.uniform(.13, .20), r, .06, 7, .10)
            for _ in range(2): v = r.uniform(.12, .9) * hh; S.tube([W.p(rr - rr * .9, v, .0), W.p(rr, v + r.normal(0, .1), .10), W.p(rr + rr * .9, v + r.normal(0, .2), .0)], .05, RED * .6, 5)
    n = int(w / 1.1); u = np.linspace(-w / 2 + .5, w / 2 - .5, n); S.boxes(np.c_[cx + u, np.full(n, cy - d / 2 - .05), np.full(n, (z0 + z1) / 2)], [.16, .2, hh], 0, CHAR * 1.1); S.boxes([[cx, cy - d / 2 - .12, z0 + hh * .80], [cx, cy - d / 2 - .12, z0 + hh * .22]], [w, .34, .30], 0, RED2 * .9)
    for sx in (-1, 1):
        for q in np.linspace(-.42, .42, 4): S.tube([[cx + sx * (w / 2 + .1), cy + q * d, z0], [cx + sx * (w / 2 + .1), cy + q * d, z1]], .5 if w > 14 else .34, RED2 * 1.1, 8)


def skull_pile(S, r, pts, zf, smin=.5, smax=1.1):
    for (x, y) in pts: s = r.uniform(smin, smax); skull(S, (x, y, float(zf(x, y)) + s * .2), s, r.uniform(-1.1, 1.1), r, BONE * r.uniform(.6, 1.), r.normal(0, .35), r.normal(0, .4), det=1 if y < -6 else 0, jaw=r.random() < .6)


def shrine_scene(seed, eye, part='hall', closed=False):
    S = Scene(seed + 3); r = np.random.default_rng(seed + 11); S.groups = 4
    if part == 'hall':
        cx, cy = 0., 12.
        for k, (w, d, z) in enumerate(((40, 30, 1.0), (36, 27, 1.9), (32, 24, 2.8))): S.boxes([cx, cy, z - .9], [w, d, 1.8], 0, CHAR * (.8 + .15 * k), top=CHAR * 1.2, tex=9)
        body(S, cx, cy, 25, 16, 2.8, 11.6, r, 8, not closed, True); roof(S, cx, cy, 25, 16, 11.6, 5.6, r, 2.5, 4.4, horns=7.2)
        body(S, cx, cy, 15.5, 10, 15.6, 20.6, r, 6, not closed); roof(S, cx, cy, 15.5, 10, 20.6, 4.4, r, 2.0, 3.4, horns=5.6)
        body(S, cx, cy, 9., 6.4, 24.0, 28.0, r, 4, not closed); roof(S, cx, cy, 9, 6.4, 28., 4.6, r, 1.8, 2.8, horns=4.8)
        # the gate: a jaw that fills the middle of the hall
        W = Wall.frame(S, (cx - 4.2, cy - 8.4, 2.6), (1, 0, 0), (0, 0, 1), 8.4, 8.2); th = np.linspace(0, np.pi, 19); W.disc(4.2, 3.9, 4.0, .35, INK, 30, 3.9); W.disc(4.2, 3.2, 3.0, .40, RED2 * .55, 26, 2.6); W.disc(4.2, 2.8, 1.9, .44, RED * .5, 22, 1.6)
        a = np.linspace(0, 2 * np.pi, 33); S.tube(W.p(4.2 + np.cos(a) * 4.1, 3.9 + np.sin(a) * 4.0, .55), .42 * (1 + .18 * np.sin(a * 5)), RED * .85, 9); S.tube(W.p(4.2 + np.cos(a) * 3.55, 3.9 + np.sin(a) * 3.45, .75), .24, RED2 * 1.5, 8)
        for sg in (1, -1):
            for q in np.linspace(-.86, .86, 13): L = 2.3 * (1 - .4 * q * q) * r.uniform(.8, 1.1) * (1.45 if abs(abs(q) - .42) < .09 else 1.); tooth(S, W.p(4.2 + q * 3.6, 3.9 + sg * 3.55 * math.sqrt(1 - q * q), .85), W.ez * -sg + W.n * .22 - W.ex * q * .10, L, .30, r, BONE * r.uniform(.9, 1.05), 8)
        # the tongue is the stair
        t = np.linspace(0, 1, 20); ty = cy - 8.2 - 12.5 * t; tz = 4.4 - 4.2 * t ** .8; wd = 1.55 + .6 * np.sin(t * 5) * .4 + .3 * t
        for i in range(19): S.quads(np.array([[cx - wd[i], ty[i], tz[i]], [cx + wd[i], ty[i], tz[i]], [cx + wd[i + 1], ty[i + 1], tz[i + 1]], [cx - wd[i + 1], ty[i + 1], tz[i + 1]]])[None], RED * (.75 + .25 * (i % 2)), lab=S.newlab()); S.quads(np.array([[cx - wd[i + 1], ty[i + 1], tz[i + 1] - .5], [cx + wd[i + 1], ty[i + 1], tz[i + 1] - .5], [cx + wd[i + 1], ty[i + 1], tz[i + 1]], [cx - wd[i + 1], ty[i + 1], tz[i + 1]]])[None], RED2, lab=S.newlab())
        S.tube(np.c_[np.full(20, cx), ty, tz + .05], .10, RED2 * .8, 5)
        for sx in (-1, 1): S.tube(np.c_[cx + sx * wd, ty, tz + .02], .22, RED * .7, 7)
        # mouths on the walls between storeys, jaw balustrades
        for (z_, w_, d_) in ((14.2, 14, 9.6), (22.6, 8.4, 6.2)):
            W = Wall.frame(S, (cx - w_ / 2, cy - d_ / 2 - 1.6, z_), (1, 0, 0), (0, 0, 1), w_, 1.8); W.slab(0, w_, 0, 1.6, -.6, 0, CHAR * .8)
            for u in np.linspace(.12, .88, 3 if w_ > 10 else 2): mouth(S, W, u * w_, .85, w_ * (.12 if w_ > 10 else .17), .55, r, .08, 9, .12)
        beast_skull(S, (cx, cy - 12.6, 15.6), 2.9, r, 0., 2.1); beast_skull(S, (cx, cy - 7.6, 24.4), 2.1, r, 0., 2.0); beast_skull(S, (cx, cy - 4.2, 33.2), 2.3, r, 0., 3.0)
        for sx in (-1, 1): beast_skull(S, (cx + sx * 12.6, cy - 12.2, 12.6), 1.5, r, sx * .35, 1.9)
        # ropes and hanging strips
        for sx in (-1, 1): S.wire((cx, cy - 8.8, 11.6), (cx + sx * 12, cy - 8.9, 11.2), 1.6, .42, BONE2 * .9, 16)
        for k in range(18): x = cx + r.uniform(-12, 12); S.wire((x, cy - 11.5 + r.normal(0, .3), 11.2), (x + r.normal(0, .3), cy - 11.4, 11.2 - r.uniform(1.4, 4.6)), .0, .05, INK * 2, 4)
        # side wings: rib halls
        for sx in (-1, 1):
            for k in range(6): y = cy - 6 + k * 3.; rib(S, (cx + sx * 13.2, y, 2.), (cx + sx * 19.8, y - .5, 7.2 - .2 * k), (sx * 5.2, 0, 6.5), .34, r, BONE * r.uniform(.75, 1.))
            S.tube([[cx + sx * 13, cy - 8, 10.2], [cx + sx * 16.5, cy - 8, 10.8], [cx + sx * 20, cy - 8.2, 7.4]], .3, BONE2, 7)
        m = 190; a = r.uniform(0, 2 * np.pi, m); q = 1 + .30 * r.random(m); pts = np.c_[cx + np.cos(a) * 20.5 * q, cy + np.sin(a) * 15.5 * q]; pts = pts[(pts[:, 1] < cy + 5) & ~((np.abs(pts[:, 0] - cx) < 3) & (pts[:, 1] < cy))]; skull_pile(S, r, pts, lambda x, y: max(-.1, 1.7 - 1.9 * max(0, math.hypot((x - cx) / 20.5, (y - cy) / 15.5) - 1) / .22), .7, 1.5)
        for _ in range(20): x = cx + r.uniform(-22, 22); rib(S, (x, cy - 17 + r.normal(0, 1.5), -.6), (x + r.normal(0, 2), cy - 18 + r.normal(0, 1.5), r.uniform(.8, 3.2)), (r.normal(0, 1.4), -.6, .6), .13, r)
    else:                                                                   # a rib gate: torii of bone.  part = ('gate', y, half width, height)
        _, y, W_, H_ = part
        for sx in (-1, 1):
            P = rib(S, (sx * W_, y, -2.), (sx * W_ * .30, y - .4, H_ * 1.02), (sx * W_ * .42, 0, H_ * .34), W_ * .085, r); rib(S, (sx * W_ * 1.16, y + 1.6, -2.), (sx * W_ * .52, y + 1.2, H_ * .86), (sx * W_ * .42, 0, H_ * .30), W_ * .055, r, BONE2)
            for k in (4, 7, 10, 13): tooth(S, P[k], (-sx * .9, -.1, .35 - .1 * k), W_ * r.uniform(.10, .20), W_ * .03, r, BONE, 6)
            horn(S, (sx * W_ * .40, y, H_ * .93), (sx, 0, .12), (sx * .25, -.05, 1), W_ * 1.05, W_ * .075, r)
        n = 17; tb = np.linspace(-1, 1, n * 8); bm = np.c_[tb * W_ * .78, np.full(len(tb), y), H_ * (.80 + .07 * tb ** 2)]; sp = np.abs(np.sin(tb * math.pi * n / 2)); S.tube(bm, W_ * (.030 + .030 * sp ** 2.), np.stack([BONE * (.45 + .5 * q) for q in sp]), 9)
        for k in range(n):
            q = -1 + (2 * k + 1) / n; x, z_ = q * W_ * .78, H_ * (.80 + .07 * q * q); tooth(S, (x, y - W_ * .03, z_ + W_ * .045), (q * .35 + r.normal(0, .12), -.25, 1), W_ * r.uniform(.11, .20), W_ * .022, r, BONE, 6, .3); tooth(S, (x, y - W_ * .04, z_ - W_ * .04), (r.normal(0, .15), -.2, -1), W_ * r.uniform(.04, .12), W_ * .016, r, BONE, 5, .3)
            if r.random() < .5: rib(S, (x, y - W_ * .05, z_ - W_ * .03), (x + r.normal(0, .04) * W_, y - W_ * .10, z_ - W_ * r.uniform(.08, .17)), (r.normal(0, .04) * W_, -W_ * .06, 0), W_ * .011, r, BONE2, 8)
        beast_skull(S, (0, y - W_ * .08, H_ * .93), W_ * .17, r, 0., 1.9)
        for k in range(5): x = r.uniform(-.7, .7) * W_; S.wire((x, y, H_ * .78), (x + r.normal(0, .2), y, H_ * .78 - r.uniform(.05, .16) * H_), 0, W_ * .007, INK * 2.2, 4)
    return S


def ink_pass(res, w, h, seed, hatch=1., rim=None):
    """turn a lit buffer into an ink drawing: 4 tones (black, shade, body, light) cut with dry-brush edges, pen outlines, hatching.
    returns rgb, alpha"""
    a = res['a']; img = np.where(a[..., None] > .02, res['img'] / np.maximum(a[..., None], .02), 0).astype(f32); z = res['z']; lab = res['lab']; geo = (z < FAR * .5)
    L = lum(img); alb = img / np.maximum(L[..., None], .02)                                             # hue
    nz = fbm(h, w, (26, 9, 4), (1, .7, .5), seed + 1) * .020 + fbm(h, w, (120, 40), (1, .5), seed + 2) * .018
    ang = res['ang']; x, y = grid(h, w)
    br = np.random.default_rng(seed + 3).standard_normal((h, w)).astype(f32); br = cv2.GaussianBlur(br, (0, 0), sigmaX=9, sigmaY=.9); br /= br.std()
    v = L + nz + .012 * br
    t1 = smooth(.052, .060, v); t2 = smooth(.135, .147, v); t3 = smooth(.31, .325, v)
    sat = np.clip((img.max(-1) - img.min(-1)) / (img.max(-1) + 1e-3), 0, 1); red = smooth(.50, .66, sat)[..., None]
    black = C('#0b080b'); sh = C('#33100f') * red + C('#3a3230') * (1 - red); bd = C('#7d1119') * red + C('#8c7d69') * (1 - red); hi = C('#c8202a') * red + C('#e0d3ba') * (1 - red)
    o = black * np.ones((h, w, 1), f32); o = o + (sh - o) * t1[..., None]; o = o + (bd - o) * t2[..., None]; o = o + (hi - o) * t3[..., None]
    # hatching at the tone borders, along and across the form
    if hatch:
        for (da, gap, lo, hi_, am) in ((.5, 7., .060, .16, .85), (-.75, 8.5, .145, .25, .55)):
            c, s = math.cos(da), math.sin(da); u = (x * -s + y * c) + 5 * fbm(h, w, (180, 60), (1, .4), seed + 5); k = np.abs((u / gap) % 1 - .5) * gap; wl = 1.5 * np.clip((hi_ - v) / (hi_ - lo), 0, 1); ln = smooth(wl + .5, wl - .5, k) * smooth(lo - .01, lo + .02, v) * smooth(hi_ + .015, hi_ - .03, v) * smooth(-.9, .0, fbm(h, w, (60, 22), (1, .6), seed + 6)); o = o * (1 - (ln * am * hatch)[..., None]) + black * (ln * am * hatch)[..., None]
    # pen outline: silhouettes heavy, inner label edges thin, broken
    lz = np.where(geo, np.log(np.maximum(z, .1)), 8.).astype(f32); g = np.maximum(np.abs(cv2.Sobel(lz, cv2.CV_32F, 1, 0, ksize=3)), np.abs(cv2.Sobel(lz, cv2.CV_32F, 0, 1, ksize=3))) / 4
    sil = smooth(.018, .05, g); le = np.zeros((h, w), f32)
    for (dx, dy) in ((1, 0), (0, 1)): le = np.maximum(le, (lab != np.roll(lab, (dy, dx), (0, 1))).astype(f32))
    brk = smooth(-1.1, -.3, fbm(h, w, (40, 15), (1, .6), seed + 7)); silw = cv2.GaussianBlur(cv2.dilate(sil, np.ones((3, 3), np.uint8)), (0, 0), .8); thin = cv2.GaussianBlur(le, (0, 0), .7) * 1.15
    ln = np.clip(np.maximum(silw * (.75 + .25 * brk), thin * brk * .8 * smooth(.03, .09, v)), 0, 1)[..., None]; o = o * (1 - ln) + black * ln
    # white-ink nicks on lit edges
    nick = smooth(.012, .05, g) * smooth(.28, .4, np.roll(v, (2, 2), (0, 1))) * smooth(.5, 1.4, fbm(h, w, (34, 12), (1, .6), seed + 9)); o = o + (C('#f2e6cc') - o) * (nick * .5)[..., None]
    if rim is not None: rz = np.clip(geo.astype(f32) - cv2.warpAffine(geo.astype(f32), np.float32([[1, 0, rim[0]], [0, 1, rim[1]]]), (w, h)), 0, 1); o = o + (C(rim[2]) - o) * (rz * .9 * brk)[..., None]
    return o.astype(f32), np.clip(a, 0, 1)


def ink_sky(h, w, hz, seed, cx=.56):
    """charcoal and dried blood: torn vertical dry-brush, a red wound of light behind the roofs"""
    x, y = grid(h, w); n = fbm(h, w, (300, 110, 44, 16), (1, .6, .4, .3), seed + 1, aspect=.7); img = C('#1b1619') * (1 + .22 * n)[..., None]
    d = np.sqrt(((x - w * cx) / (w * .30)) ** 2 + ((y - hz * .52) / (hz * .70)) ** 2) + .10 * n; g = smooth(1.05, .18, d); img = img + (C('#5e0d13') - img) * (g * .9)[..., None]; g2 = smooth(.62, .16, d + .06 * fbm(h, w, (90, 30), (1, .6), seed + 2, .6)); img = img + (C('#a81a1f') - img) * (g2 * .85)[..., None]; g3 = smooth(.30, .10, d + .05 * fbm(h, w, (60, 24), (1, .6), seed + 4, .6)); img = img + (C('#d8443a') - img) * (g3 * .6)[..., None]
    r = np.random.default_rng(seed + 5); cv = Canvas(h, w, img=np.clip(img, 0, 1)); k = 220; px = r.uniform(-.05, 1.05, k) * w; py = r.uniform(-.1, .85, k) * hz; a = np.pi / 2 + (px / w - cx) * 1.1 + r.normal(0, .16, k); L = r.uniform(160, 760, k); wd = r.uniform(3, 34, k) * r.random(k) ** .6 + 1.5; far = np.hypot((px - w * cx) / (w * .34), (py - hz * .5) / (hz * .7)); col = np.where((far > r.uniform(.5, 1.1, k))[:, None], C('#0c090c'), np.where((r.random(k) < .5)[:, None], C('#2a0a0e'), C('#7a1218')))
    cv.strokes(arcs(px, py, a, L, r.normal(0, .25, k), K=9), wd, col.astype(f32), op=r.uniform(.3, .85, k), dry=r.uniform(.45, 1.1, k), seed=seed, shape='flat', bristle=.12)
    k = 12; px = r.uniform(0, 1, k) * w; py = r.uniform(0, .5, k) * hz; cv.strokes(arcs(px, py, np.pi / 2 + (px / w - cx) * 1.1 + r.normal(0, .1, k), r.uniform(200, 600, k), r.normal(0, .2, k), K=9), r.uniform(1, 5, k), np.tile(C('#8f7f6a'), (k, 1)), op=r.uniform(.1, .35, k), dry=r.uniform(.4, .9, k), seed=seed + 1, bristle=.1)
    o = cv.img; sp = np.zeros((h, w), f32); m = 420
    for px, py, q in zip(r.uniform(0, w, m), r.uniform(0, hz, m), r.random(m) ** 4): cv2.circle(sp, (int(px * 8), int(py * 8)), int((.7 + 6.5 * q) * 8), 1., -1, cv2.LINE_AA, 3)
    o = o * (1 - sp * .9)[..., None] + C('#0a070a') * (sp * .9)[..., None]
    vg = smooth(.55, 1.15, np.hypot((x - w * cx) / (w * .62), (y - hz * .5) / (hz * .85))); return (o * (1 - .62 * vg)[..., None]).astype(f32)


@painter('shrine', 3)
def shrine(w=2400, h=1350, seed=0, eye=(-3.6, -41., 3.0), target=(.9, 0., 13.2), focal=1130., roll=-.04, paint=True, ss=2, gates=((-3., 31., 46.), (-20., 18.5, 30.), (-31.5, 9.8, 14.4)), top=0):
    """Malevolent Shrine over blood-black water, ink.  img = finished still.  For animation use ShrineShot(P).frame(...).
    layers: 'sky' 'hall' 'gate0' (far) 'gate1' 'gate2' (near), premult RGBA; extra: hm_<layer> metres above the water per pixel, wl = waterline y px, dist = metres, hz"""
    h = h + top; cam = Camera3D(eye, target, focal, w, h, roll=roll, cy=(h - top) / 2 + top); hz = Wd.horizon_y(cam)
    nk = dict(amb=('#24222a', '#0c0608'), fills=[((-.7, -.6, .45), '#8a8078'), ((.9, .1, .1), '#4a0d10'), ((0, 1, .6), '#2a0708')], sun=((-.45, -.55, .75), '#b8aa96', .55))
    lays, ex = {}, {}; wl, dist = [], []; rays = cam.rays(); yy = grid(h, w)[1]
    for nm, part, yb in [('hall', 'hall', -9.)] + [('gate%d' % i, ('gate', *g), g[0]) for i, g in enumerate(gates)]:
        gi = -1 if nm == 'hall' else int(nm[-1]); S = shrine_scene(seed + (0 if gi < 0 else gi * 7), cam.eye, part); res = S.render(cam, ss=ss, smres=3072, **nk); Z = cam.eye[2] + rays[..., 2] * np.minimum(res['z'], 1e4); tz = 36. if gi < 0 else gates[gi][2]
        ao = Wd.occlusion(res['z'], .66, (4, 12, 34), 9.); k = 1. if gi < 0 else (.82, .60, .40)[gi]; vg = (.50 + .62 * np.clip(Z / tz, 0, 1)) * k
        res['img'] = res['img'] * (ao * vg)[..., None]; rim = (6, 1, '#c8242a') if gi < 0 else ((-5, 2, '#a3161e'), (-6, 2, '#7a1016'), (-8, 3, '#8a1218'))[gi]; o, a = ink_pass(res, w, h, seed + 20 + len(lays), 1., rim)
        if gi >= 0: o = o + (C('#160b0e') - o) * (.04, .10, .16)[gi]
        a = a * smooth(-.02, .06, Z); lays[nm] = np.dstack([o * a[..., None], a]).astype(f32); ex['hm_' + nm] = np.clip(Z, -5, 60).astype(np.float16); p, _ = cam.project(np.array([eye[0], yb, 0.])); wl.append(float(p[1])); dist.append(float(cam.cam(np.array([0, yb, 0.]))[2]))
    sk = ink_sky(h - top, w, (hz - top) * 1.04, seed + 40, .53); sk = np.vstack([np.repeat(sk[:1], top, 0), sk]) if top else sk; lays['sky'] = np.dstack([sk, np.ones((h, w), f32)]).astype(f32)
    P = Plate(np.zeros((h, w, 3), f32), None, cam, {}, lays, dict(ex, wl=np.array(wl), dist=np.array(dist), hz=hz)); P.img = Wd.tooth_pass(ShrineShot(P).frame(), seed + 6, .010, .012); return P


class ShrineShot:
    """cheap compositor for the rising shrine.  frame(sink=(hall, g0, g1, g2) metres still under water, t=ripple time in frames, rings=[(X, Y, radius, amp)])"""
    NAMES = ('hall', 'gate0', 'gate1', 'gate2')

    def __init__(s, P, q=1):
        s.P, s.cam = P, P.cam; s.h, s.w = P.img.shape[:2]; s.u = s.cam.uv((0, 0, 0), (1, 0, 0), (0, 1, 0)); s.gr = (s.u[2] > 0); s.x, s.y = grid(s.h, s.w); s.roll = math.asin(s.cam.R[0][2])
        s.r = np.random.default_rng(5); s.wv = [(s.r.uniform(0, 6.28), s.r.uniform(.18, 1.5), s.r.uniform(.06, .18) * (1 if k % 2 else -1), s.r.uniform(0, 6)) for k in range(6)]
        V = np.maximum(s.u[2], .5); s.amp = np.clip(70 / V, .5, 13.); s.n1 = fbm(s.h, s.w, (90, 30), (1, .5), 9, 7.)

    def lay(s, nm, sink):
        L = s.P.layers[nm]; 
        if sink <= 0: return L
        i = s.NAMES.index(nm); dy = s.cam.f * sink / float(s.P.extra['dist'][i]); m = smooth(sink - .05, sink + .10, s.P.extra['hm_' + nm].astype(f32))[..., None]
        return cv2.warpAffine(L * m, np.float32([[1, 0, 0], [0, 1, dy]]), (s.w, s.h), flags=cv2.INTER_LINEAR)

    def flip(s, L, wl):
        """mirror about the (rolled) waterline"""
        c, sn = math.cos(2 * s.roll), math.sin(2 * s.roll); x0, y0 = s.w / 2, wl; M = np.float32([[c, sn, x0 - c * x0 - sn * y0], [sn, -c, y0 - sn * x0 + c * y0]])
        return cv2.warpAffine(L, M, (s.w, s.h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE if L.shape[2] == 3 else cv2.BORDER_CONSTANT)

    def frame(s, sink=(0, 0, 0, 0), t=0., rings=(), dark=1.):
        P = s.P; sky = P.layers['sky'][..., :3]; img = sky.copy(); hz = float(P.extra['hz']); ref = s.flip(sky, hz); wl = P.extra['wl']; cov = np.zeros((s.h, s.w), f32)
        for i, nm in enumerate(s.NAMES):
            if sink[i] >= 60: continue
            L = s.lay(nm, sink[i]); img = img * (1 - L[..., 3:]) + L[..., :3]; F = s.flip(L, float(wl[i])); ref = ref * (1 - F[..., 3:]) + F[..., :3]
            cov = cov * (1 - L[..., 3]) + L[..., 3]
        U_, V_, z = s.u; rip = np.zeros((s.h, s.w), f32)
        for a, f_, w_, p_ in s.wv: rip += np.sin((U_ * math.cos(a) * .55 + V_ * math.sin(a) * 1.5) * f_ * 2 + t * w_ + p_) / (1 + f_)
        rg = np.zeros((s.h, s.w), f32)
        for (X, Y, R, am) in rings: q = np.hypot(U_ - X, V_ - Y) - R; rg += am * np.exp(-(q / 1.6) ** 2) * np.cos(q * 2.4)
        rp = rip + rg; far = np.clip(20 / np.maximum(z, 3), .12, 1.)
        R_ = cv2.remap(ref, np.clip(s.x + rp * s.amp * 1.1 + s.n1 * s.amp * .5, 0, s.w - 1), np.clip(s.y + rp * s.amp * .8, 0, s.h - 1), cv2.INTER_LINEAR)
        wat = R_ * np.array([.46, .27, .27], f32) * dark + C('#120306') * .7
        # crisp ink shapes on the swell: red lights, black troughs, bone glints on the ring
        sw = rp + .35 * s.n1; lt = smooth(.98, 1.06, sw * (.55 + .45 * far)) * smooth(.02, .10, far); dk = smooth(-.95, -1.05, sw) * smooth(.04, .2, far); wat = wat + (C('#8a1118') - wat) * (lt * .55)[..., None]; wat = wat + (C('#080305') - wat) * (dk * .55)[..., None]
        if len(rings): b = smooth(.45, .55, rg * (.6 + .4 * np.sin(U_ * 2.1 + V_ * .9 + s.n1 * 2))); wat = wat + (C('#c8463e') - wat) * (b * .8)[..., None]
        ln = smooth(.07, .015, np.abs(np.sin(V_ * 1.1 + rp * 1.1 + .5 * np.sin(U_ * .6)))) * smooth(.12, .5, far) * smooth(-.1, .5, s.n1); wat = wat * (1 - .6 * ln)[..., None]
        m = s.gr.astype(f32) * (1 - cov); return (img * (1 - m[..., None]) + wat * m[..., None]).astype(f32)


def pillar_mouths(cam):
    """screen boxes of the pillar mouths: [(tier, x0, y0, x1, y1, cx, cy)]"""
    out = []
    for tier, (w, d, z0, z1, pil) in enumerate(((25, 16, 2.8, 11.6, 8), (15.5, 10, 15.6, 20.6, 6), (9., 6.4, 24., 28., 4))):
        hh = z1 - z0; rr = min(.95, w / pil * .24); yf = 12 - d / 2 - .25 - rr * .98
        for x in np.linspace(-w / 2, w / 2, pil):
            if tier == 0 and abs(x) < w * .16: continue
            q, _ = cam.project(np.array([[x - rr * 1.15, yf, z0 + .16 * hh], [x + rr * 1.15, yf, z0 + .16 * hh], [x - rr * 1.15, yf, z0 + .88 * hh], [x + rr * 1.15, yf, z0 + .88 * hh]])); c = q.mean(0); out.append((tier, *q.min(0), *q.max(0), *c))
    return out


@painter('shrine_closed', 1)
def shrine_closed(w=1920, h=1080, seed=0, eye=(-3.6, -41., 3.0), target=(.9, 0., 13.2), focal=904., roll=-.04, ss=2, top=0):
    """the hall alone with every pillar mouth shut (for f2's openings).  layer 'hall'"""
    h = h + top; cam = Camera3D(eye, target, focal, w, h, roll=roll, cy=(h - top) / 2 + top); nk = dict(amb=('#24222a', '#0c0608'), fills=[((-.7, -.6, .45), '#8a8078'), ((.9, .1, .1), '#4a0d10'), ((0, 1, .6), '#2a0708')], sun=((-.45, -.55, .75), '#b8aa96', .55))
    S = shrine_scene(seed, cam.eye, 'hall', True); res = S.render(cam, ss=ss, smres=3072, **nk); Z = cam.eye[2] + cam.rays()[..., 2] * np.minimum(res['z'], 1e4); res['img'] = res['img'] * (Wd.occlusion(res['z'], .66, (4, 12, 34), 9.) * (.50 + .62 * np.clip(Z / 36., 0, 1)))[..., None]; o, a = ink_pass(res, w, h, seed + 20, 1., (6, 1, '#c8242a')); a = a * smooth(-.02, .06, Z)
    return Plate(np.zeros((4, 4, 3), f32), None, cam, {}, dict(hall=np.dstack([o * a[..., None], a]).astype(f32)))
