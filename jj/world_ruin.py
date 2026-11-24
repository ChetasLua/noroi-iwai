"""Ruined Shibuya: the same place as world_city, broken.  modes: fire | rain | snow | ash | night"""
import numpy as np, cv2, math
from .paint import rgb, f32, fbm, smooth, mix, grid, Canvas, arcs, jit_color, lum
from .cam import Camera3D, rot_x, rot_y, rot_z
from .comp import blur, radial, over
from . import world as Wd, world_sky as SK, clouds
from .world import C, Scene, Wall, Plate, painter, FAR
from .world_props import *
from .world_city import ROADS, U, _edge, row, shibuya, grounds, markings

CON = ['#8c8a88', '#7f8086', '#979088', '#86817e', '#8a8f98', '#78757c']


def transform(S0, S, R=None, piv=(0, 0, 0), t=(0, 0, 0)):
    """append sub-scene S0 to S, rotated about piv and moved"""
    R = np.eye(3) if R is None else np.asarray(R, float); piv = np.asarray(piv, float); t = np.asarray(t, float)
    f = lambda P: ((np.asarray(P, float) - piv) @ R.T + piv + t)
    S0.P = [f(p).astype(f32) for p in S0.P]; S0.Dd = [(np.asarray(d, float) @ R.T).astype(f32) for d in S0.Dd]
    S0.segs = [(f(a), f(b), *rest) for (a, b, *rest) in S0.segs]
    S0.lights = [(*f(np.array(l[:3])), *l[3:]) for l in S0.lights]
    S.merge(S0)


def ragged(a, b, r, n=6, amp=.5, up=(0, 0, 1), out=(0, 0, 0)):
    """broken edge polyline between a and b (bitten inward along `out`)"""
    t = np.sort(np.r_[0, r.uniform(.05, .95, n), 1])[:, None]; P = np.asarray(a, float) + (np.asarray(b, float) - np.asarray(a, float)) * t
    k = np.abs(r.normal(0, amp, n + 2)); k[0] *= .3; k[-1] *= .3
    return P - np.asarray(out, float) * k[:, None]


def tower(S, foot, h, r, eye, dmg=.5, col=None, fh=3.6, bay=5.2, keep=.86, lean=None, soot=.5, style=None, skirt=1., nmax=9, dens=1.):
    """broken concrete frame: bays stand to different heights; slabs, columns, facade panels, hanging pieces, rebar, rubble skirt.
    foot = 4 CCW corners (street edge first)."""
    P = np.asarray(foot, float); ex = P[1] - P[0]; w = float(np.hypot(*ex)); ex = ex / w; ey = P[3] - P[0]; d = float(np.hypot(*ey)); ey = ey / d
    nx, ny = int(np.clip(round(w / bay), 2, nmax)), int(np.clip(round(d / bay), 2, 5)); bx, by = w / nx, d / ny; nf = max(1, int(h / fh)); col = C(pick(r, CON) if col is None else col)
    T = Scene(int(r.integers(1 << 30))); T.r = r
    # standing floors per bay: a diagonal bite plus noise
    a = r.uniform(0, 2 * np.pi); gx, gy = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny)); s = math.cos(a) * gx + math.sin(a) * gy * .7
    H_ = nf * (1 - dmg * (.10 + .62 * np.clip(s * .8 + .45, 0, 1.3)) + r.normal(0, .09 * dmg + .02, (ny, nx))); H_ = np.clip(np.round(H_), 0, nf).astype(int)
    if r.random() < .5 * dmg: H_[int(r.integers(ny)), int(r.integers(nx))] = 0
    def pt(u, v, z): return np.array([P[0][0] + ex[0] * u + ey[0] * v, P[0][1] + ex[1] * u + ey[1] * v, z])
    def hgt(i, j): return H_[j, i] if 0 <= i < nx and 0 <= j < ny else 0
    inner = col * .22; wm = pick(r, ['grid', 'grid', 'ribbon', 'solid', 'slit']); wv0, wv1, wfr = dict(grid=(.26, .80, .62), ribbon=(.30, .78, .90), solid=(.36, .72, .34), slit=(.12, .88, .30))[wm]; E3 = np.r_[ex, 0.]; N3 = np.r_[ey, 0.]; Z3 = np.array([0, 0, 1.])
    ax3 = np.stack([E3, N3, Z3])
    cen, siz, cc = [], [], []
    for j in range(ny):
        for i in range(nx):
            for k in range(hgt(i, j)):
                z = (k + 1) * fh; c = pt((i + .5) * bx, (j + .5) * by, z - .16); cen.append(c); siz.append((bx + .02, by + .02, .34)); cc.append(col * r.uniform(.78, .95))
                for (du, dv) in ((0, 0), (1, 0), (0, 1), (1, 1)):
                    if (du == 1 and hgt(i + 1, j) > k) or (dv == 1 and hgt(i, j + 1) > k): continue
                    cen.append(pt((i + du) * bx, (j + dv) * by, z - fh / 2)); siz.append((.55, .55, fh)); cc.append(col * r.uniform(.7, .9))
                if r.random() < .82:                                     # partitions: rooms are dark, light leaks through only some
                    ic = col * (1 + r.normal(0, .06, 3)).astype(f32) * r.uniform(.26, .62)
                    cen.append(pt((i + .5) * bx, (j + r.uniform(.35, .9)) * by, z - fh / 2)); siz.append((bx * .97, .14, fh)); cc.append(ic)
                    if r.random() < .55: cen.append(pt((i + r.uniform(.1, .9)) * bx, (j + .5) * by, z - fh / 2)); siz.append((.14, by * .96, fh * r.uniform(.6, 1.))); cc.append(ic * 1.2)
                for _ in range(int(r.integers(0, 3) * dens)):            # things left on the floor
                    q = r.uniform(.3, 1.1, 3); cen.append(pt((i + r.uniform(.15, .85)) * bx, (j + r.uniform(.15, .85)) * by, z - fh + q[2] / 2 + .02)); siz.append(q); cc.append(C(pick(r, ['#5a5652', '#6e6258', '#4a5058', '#806a58'])) * .7)
    if cen: T.boxes(np.array(cen), np.array(siz), 0, np.array(cc, f32), axes=ax3, tex=1)
    # facade panels on outer faces
    faces = [(0, -1, 'y'), (0, 1, 'Y'), (-1, 0, 'x'), (1, 0, 'X')]
    for (di, dj, nm) in faces:
        o2 = (P[0] if nm in 'yx' else P[2]); nn = -ey if nm == 'y' else ey if nm == 'Y' else -ex if nm == 'x' else ex
        if np.dot(eye[:2] - o2, nn) < -2: continue
        for j in range(ny):
            for i in range(nx):
                if (dj and j != (0 if dj < 0 else ny - 1)) or (di and i != (0 if di < 0 else nx - 1)): continue
                top = hgt(i, j); sk = r.random() < .12
                for k in range(top):
                    q = k / max(1, top); p = keep * (1 - .55 * q ** 3) * (.4 if sk else 1.)
                    if nm == 'y': A, B = pt(i * bx, 0, k * fh), pt((i + 1) * bx, 0, k * fh)
                    elif nm == 'Y': A, B = pt((i + 1) * bx, d, k * fh), pt(i * bx, d, k * fh)
                    elif nm == 'x': A, B = pt(0, (j + 1) * by, k * fh), pt(0, j * by, k * fh)
                    else: A, B = pt(w, j * by, k * fh), pt(w, (j + 1) * by, k * fh)
                    W = Wall(T, A, B, k * fh, (k + 1) * fh); ww = W.w
                    if r.random() > p:
                        if r.random() < .35: hang(T, W, r, col)
                        continue
                    c0 = col * r.uniform(.86, 1.08); sh = c0 * (1 - .55 * soot * (r.random() < .5))
                    if style == 'glass' and r.random() < .7:
                        m = max(2, int(ww / 1.7)); x = np.arange(m + 1) * ww / m; W.slab(x - .05, x + .05, 0, fh, -.05, .06, '#3c4048'); 
                        kk = r.random(m) < .55
                        if kk.any(): W.rect(x[:-1][kk] + .06, x[1:][kk] - .06, .5, fh - .1, 0., '#1c2636', ct='#4a6080', em=.15, snow=0)
                        W.slab(0, ww, 0, .5, -.1, .08, c0, tex=1); continue
                    v0, v1 = fh * wv0, fh * wv1; m = (2 if ww > 4 else 1) if wm != 'slit' else max(2, int(ww / 1.5)); mg = max(.25, ww * (1 - wfr) / (m + 1)); gap = (ww - mg * (m + 1)) / m
                    W.slab(0, ww, 0, v0, -.22, 0, c0, side=c0 * .6, tex=1); W.slab(0, ww, v1, fh, -.22, 0, c0 * .96 if r.random() < .5 else np.stack([sh]), side=c0 * .5, tex=1)
                    xs = np.arange(m + 1) * (gap + mg); W.slab(xs, xs + mg, v0, v1, -.22, 0, c0, side=c0 * .6, tex=1); W.slab(0, ww, v0 - .09, v0, 0, .10, c0 * 1.1) if r.random() < .5 else None
                    for b in range(m):
                        u0 = xs[b] + mg
                        if r.random() < .35:                             # shard left in the frame
                            t = r.uniform(.2, .8); W.poly([(u0, v0), (u0 + gap * r.uniform(.3, 1), v0), (u0 + gap * t * .4, v0 + (v1 - v0) * r.uniform(.4, 1)), (u0, v0 + (v1 - v0) * r.uniform(.3, .9))], -.15, '#4a5a6e', em=.12, snow=0)
                        if soot > .2 and r.random() < soot * .45:        # soot tongue above the opening
                            n_ = 7; tt = np.linspace(0, 1, n_); hh_ = (.25 + .9 * np.sin(np.pi * np.clip(tt + r.normal(0, .06, n_), 0, 1)) ** .7 * r.uniform(.5, 1.25, n_)) * (fh - v1) * r.uniform(.8, 1.7); xx_ = u0 + gap * (tt * 1.15 - .075)
                            T.quads(np.stack([W.p(xx_[:-1], v1 - .02, .014), W.p(xx_[1:], v1 - .02, .014), W.p(xx_[1:], v1 + hh_[1:], .014), W.p(xx_[:-1], v1 + hh_[:-1], .014)], 1), np.stack([np.stack([c0 * .26, c0 * .26, c0 * .62, c0 * .62])] * (n_ - 1)), snow=0, bias=3e-4, lab=T.newlab())
    # broken rims: every exposed slab edge is bitten, with hanging plates and rods
    for j in range(ny):
        for i in range(nx):
            t = hgt(i, j)
            for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                o = hgt(i + di, j + dj)
                if not (0 <= i + di < nx and 0 <= j + dj < ny): continue
                for k in range(o, t):
                    if r.random() > .8 * dens: continue
                    z = (k + 1) * fh; A = pt((i + (di > 0)) * bx if di else i * bx, (j + (dj > 0)) * by if dj else j * by, z); B = pt((i + (di > 0)) * bx if di else (i + 1) * bx, (j + (dj > 0)) * by if dj else (j + 1) * by, z)
                    out = (E3 * di + N3 * dj)
                    m = 5; tt = np.sort(np.r_[0, r.uniform(.05, .95, m - 1), 1]); L = r.uniform(.0, 1.5, m + 1)
                    e = A + (B - A) * tt[:, None] + out * L[:, None] - Z3 * (L * r.uniform(.05, .5, m + 1))[:, None]; b_ = A + (B - A) * tt[:, None]
                    c0 = col * r.uniform(.8, 1.); T.quads(np.stack([b_[:-1], b_[1:], e[1:], e[:-1]], 1), c0, tex=1); lo = np.array([0, 0, -.33]); T.quads(np.stack([e[:-1], e[1:], e[1:] + lo, e[:-1] + lo], 1), c0 * 1.1, tex=9); T.quads(np.stack([b_[:-1] + lo, b_[1:] + lo, e[1:] + lo, e[:-1] + lo], 1), c0 * .5)
                    if r.random() < .7: rebar(T, e + lo * .5, r, int(r.integers(3, 8)), .35, (.5, 1.7), out * .9 - Z3 * .15)
                    if r.random() < .30:                                  # a plate hanging by its rods
                        q = r.uniform(.2, .7); c_ = A + (B - A) * q; l_ = r.uniform(1.2, 2.8); w_ = r.uniform(.8, 2.); s_ = (B - A) / np.linalg.norm(B - A); dn = out * r.uniform(.1, .5) - Z3; dn /= np.linalg.norm(dn); p0 = c_ + out * .3 - Z3 * r.uniform(.4, .9)
                        pl = np.array([p0 - s_ * w_ / 2, p0 + s_ * w_ / 2 * r.uniform(.6, 1.), p0 + s_ * w_ / 2 * r.uniform(.3, 1.) + dn * l_, p0 - s_ * w_ / 2 * r.uniform(.5, 1.) + dn * l_ * r.uniform(.6, 1.)]); th = np.cross(s_, dn) * .16; T.quads(pl[None], c0 * 1.05, tex=9); T.quads(pl[None] + th, c0 * .9, tex=1); T.quads(np.stack([pl[2], pl[3], pl[3] + th, pl[2] + th])[None], c0 * .7); T.quads(np.stack([pl[1], pl[2], pl[2] + th, pl[1] + th])[None], c0 * .75)
                        T.seg([c_ - s_ * w_ * .3, c_ + s_ * w_ * .25], [pl[0], pl[1]], .035, '#4a2c22')
            if t:                                                        # jagged parapet stubs and column stumps on the roofline
                z = t * fh; m = int(r.integers(2, 6)); 
                for _ in range(m): q = r.uniform(.3, 1.1, 3) * [1.2, 1., r.uniform(.5, 1.6)]; rock(T, pt((i + r.random()) * bx, (j + r.random()) * by, z + q[2] * .25), q, r, col * r.uniform(.8, 1.1), 1)
                if r.random() < .5: rebar(T, pt((i + .02) * bx, (j + .02) * by, z), r, 6, .16, (.8, 2.2), Z3)
                if r.random() < .25: T.boxes(pt(i * bx, j * by, z + .8), [.5, .5, 1.6], 0, col * .85, axes=ax3, tex=9)
    # skirt of rubble
    if skirt:
        m = int(26 * skirt * (w + d) / 30); u = r.uniform(-.14, 1.14, m) * w; v = r.uniform(-.4, .5, m) * d; e = np.abs(r.normal(0, 1, m)); p = np.array([pt(a_, b_, 0)[:2] for a_, b_ in zip(u, v)])
        dist = np.maximum(0, -v) / d; z = np.clip((.26 - dist) * 9 * dmg, 0, 3.2) * r.uniform(.2, 1., m); rubble(T, p, np.clip(.35 + .75 * e, .3, 3.2), r, z, [tuple(col * k_) for k_ in (.8, 1., 1.1)], .4, .2)
    R = np.eye(3)
    if lean is not None: R = rot_z(0) @ (rot_x(lean[0]) @ rot_y(lean[1]))
    c = P.mean(0); transform(T, S, R, (c[0], c[1], 0.), (0, 0, -.6 * abs(lean[0] + lean[1]) * min(w, d) if lean is not None else 0))
    return H_


def hang(T, W, r, col):
    """torn facade remains: a drooping panel, cables"""
    w, h = W.w, W.h
    if r.random() < .5:
        q = r.uniform(.3, .6) * w; u = r.uniform(0, w - q); n = r.uniform(.2, 1.1); T.quads(np.stack([W.p(u, h * .95, 0), W.p(u + q, h * .95, 0), W.p(u + q * r.uniform(.6, 1), h * r.uniform(.2, .5), n), W.p(u, h * r.uniform(.1, .4), n * r.uniform(.5, 1.2))])[None], C(col) * .9, tex=1)
    for _ in range(int(r.integers(1, 4))): u = r.uniform(0, w); L = r.uniform(.6, 2.6); p = W.p(u, h, .05); q = W.p(u + r.normal(0, .3), h - L * .6, .2); T.seg([p, q], [q, W.p(u + r.normal(0, .4), h - L, .1)], .03, '#1c1a1c')


def facade_slab(S, o, w, h, R, r, col='#8a8884', bay=3.2, fh=3.4, th=.4, spike=1.):
    """a whole piece of wall thrown down: window holes, bitten outline, rods at the edge.  o = centre, R = 3x3 orientation (x along, y up the wall, z out)"""
    R = np.asarray(R, float); o = np.asarray(o, float); col = C(col); nx, ny = max(2, int(w / bay)), max(2, int(h / fh)); bx, by = w / nx, h / ny; W = Wall.frame(S, o - R[0] * w / 2 - R[1] * h / 2, R[0], R[1], w, h)
    al = np.ones((ny, nx), bool)
    for (j, i) in ((0, 0), (0, nx - 1), (ny - 1, 0), (ny - 1, nx - 1)): al[j, i] = r.random() < .35
    al &= r.random((ny, nx)) > .10
    for j in range(ny):
        for i in range(nx):
            if not al[j, i]: continue
            u, v = i * bx, j * by; a, b, c_, d_ = bx * .2, bx * .8, by * .28, by * .78; cc = col * r.uniform(.88, 1.08)
            W.slab(u, u + bx, v, v + c_, -th, 0, cc, side=cc * .7, tex=1); W.slab(u, u + bx, v + d_, v + by, -th, 0, cc, side=cc * .7, tex=1); W.slab(np.array([u, u + b]), np.array([u + a, u + bx]), v + c_, v + d_, -th, 0, cc, side=cc * .6, tex=1)
            for (di, dj, p0, p1, dr) in ((-1, 0, (u, v), (u, v + by), -R[0]), (1, 0, (u + bx, v), (u + bx, v + by), R[0]), (0, -1, (u, v), (u + bx, v), -R[1]), (0, 1, (u, v + by), (u + bx, v + by), R[1])):
                ii, jj = i + di, j + dj
                if 0 <= ii < nx and 0 <= jj < ny and al[jj, ii]: continue
                A, B = W.p(*p0, 0), W.p(*p1, 0); t = np.linspace(0, 1, 6)[:, None]; L = r.uniform(0, .9, 6) * spike; e = A + (B - A) * t + dr * L[:, None]; bb = A + (B - A) * t; lo = -R[2] * th; S.quads(np.stack([bb[:-1], bb[1:], e[1:], e[:-1]], 1), cc, tex=1); S.quads(np.stack([e[:-1], e[1:], e[1:] + lo, e[:-1] + lo], 1), cc * .8, tex=9); S.quads(np.stack([bb[:-1] + lo, bb[1:] + lo, e[1:] + lo, e[:-1] + lo], 1), cc * .6)
                if r.random() < .6: rebar(S, e, r, 4, .3, (.4, 1.3), dr)


def drifts(S, pts, r, col, z=None, rmin=.8, rmax=3.4, hmul=.22):
    """soft mounds (snow, ash, earth) as low faceted domes"""
    for (x, y) in pts:
        R_ = r.uniform(rmin, rmax); n = 11; a = np.linspace(0, 2 * np.pi, n, endpoint=False); k = R_ * (1 + .22 * r.normal(0, 1, n)); zz = 0. if z is None else float(z(np.array(x), np.array(y))); h_ = R_ * hmul * r.uniform(.6, 1.3); rot = r.uniform(0, 3.14); ar = r.uniform(1., 2.2)
        def ring(f, zf): q = np.c_[np.cos(a) * k * f * ar, np.sin(a) * k * f]; c, s_ = math.cos(rot), math.sin(rot); q = q @ np.array([[c, s_], [-s_, c]]); return np.c_[q + [x, y], np.full(n, zz + zf)]
        A, B, D = ring(1., -.05), ring(.66, h_ * .62), ring(.28, h_)
        lab = S.newlab(); S.quads(np.stack([A, np.roll(A, -1, 0), np.roll(B, -1, 0), B], 1), col, lab=lab); S.quads(np.stack([B, np.roll(B, -1, 0), np.roll(D, -1, 0), D], 1), col, lab=lab); S.poly(D, col, lab=lab)


def crater(c, R=13., dep=4.2, seed=0):
    """ground height function with a blown crater at c"""
    r = np.random.default_rng(seed); k = r.uniform(0, 6.28, 6); am = r.uniform(.03, .10, 6)
    def z(X, Y):
        dx, dy = X - c[0], Y - c[1]; th = np.arctan2(dy, dx); q = np.hypot(dx, dy) / (R * (1 + sum(a * np.sin((i + 2) * th + p) for i, (a, p) in enumerate(zip(am, k)))))
        bowl = -dep * np.clip(1 - q * q, 0, 1) ** .8; rim = .26 * dep ** .9 * np.exp(-((q - 1.08) / .15) ** 2)
        return bowl + rim + .12 * np.sin(X * .9 + 2 * np.sin(Y * .5)) * np.exp(-((q - .8) / .5) ** 2)
    return z


def plates(S, c, R, r, n=34, col='#4e4f58', z=None, smin=1.2, smax=3.8, paint=True):
    """road surface heaved into tilted plates round the crater"""
    for k in range(n):
        th = r.uniform(0, 2 * np.pi); d = R * r.uniform(.78, 1.35); s = r.uniform(smin, smax); m = int(r.integers(4, 7)); a = np.sort(r.uniform(0, 2 * np.pi, m)); a = np.linspace(0, 2 * np.pi, m, endpoint=False) + r.normal(0, .25, m); pl = np.c_[np.cos(a), np.sin(a)] * s * r.uniform(.6, 1., (m, 1))
        tilt = r.uniform(.12, .75) * np.clip(1.3 - abs(d / R - 1.02) * 2.4, .2, 1.); A = rot_z(th) @ rot_y(tilt * r.choice([1, 1, -1])) @ rot_x(r.normal(0, .16))
        o = np.array([c[0] + math.cos(th) * d, c[1] + math.sin(th) * d, .25 + s * .2 * tilt + (0 if z is None else 0)])
        hi = np.c_[pl, np.zeros(m)] @ A.T + o; lo = np.c_[pl * .96, np.full(m, -r.uniform(.28, .5))] @ A.T + o
        S.poly(hi, C(col) * r.uniform(.8, 1.15), tex=13); 
        for i in range(m): S.quads(np.stack([lo[i], lo[(i + 1) % m], hi[(i + 1) % m], hi[i]])[None], C('#7a6e66') * r.uniform(.7, 1.05), tex=9)
        if paint and r.random() < .4:
            q = np.array([(-.5, -.14), (.5, -.14), (.5, .14), (-.5, .14)]) * s; S.poly(np.c_[q, np.full(4, .015)] @ A.T + o, '#b9b7b0', tex=8, bias=4e-4)
        if r.random() < .25: rebar(S, (hi + lo) / 2, r, 4, .5, (.3, .9), A @ [1, 0, .1])


def smoke(h, w, x, y, size, height, seed=0, lean=.25, tones=('#6a6470', '#4a4654', '#33303e', '#22202c'), light=(-.6, -.6, .3), op=.92, glow=None, n=15, rise=1.0):
    """painted smoke column: a stack of cauliflower heads, thin at the root, torn and lost at the top.  premult RGBA"""
    r = np.random.default_rng(seed * 31 + 5); T = clouds.tones(*tones); S = clouds.Sky(h, w, tones[2], light, layered=True, seed=seed)
    L = []
    for k in range(n):
        q = k / (n - 1.); yy = y - height * q; xx = x + lean * height * q ** 1.4 + math.sin(q * 6 + seed) * size * .22; s = size * (.28 + .95 * q ** .8)
        for j in range(3): L.append((xx + r.normal(0, .38) * s, yy + r.normal(0, .22) * s, r.uniform(-.2, .6) * s, s * r.uniform(.42, .72), s * r.uniform(.36, .6), 0))
    L = np.array(L); out = list(L)
    for lv in range(2):
        nxt = []
        for (cx, cy, zc, rx, ry, _) in L:
            for _k in range(6): a = r.uniform(0, 2 * np.pi); ko = r.uniform(.6, .95); sc = r.uniform(.25, .48); nxt.append((cx + math.cos(a) * rx * ko, cy + math.sin(a) * ry * ko, zc + r.uniform(.0, .8) * rx, rx * sc, ry * sc * 1.05, lv + 1))
        out += nxt; L = np.array([v for v in nxt if v[3] > 5]); 
        if len(L) == 0: break
    L = np.array(out); x0 = int(max(-40, L[:, 0].min() - size)); x1 = int(min(w + 40, L[:, 0].max() + size)); y0 = int(max(-40, L[:, 1].min() - size)); y1 = int(min(h + 40, y + size * .3))
    if x1 - x0 < 8 or y1 - y0 < 8: return np.zeros((h, w, 4), f32)
    res = clouds.shade(L, x0, y0, x1 - x0, y1 - y0, y + height * .2, height, T, T[2], light, seed, 1, .8, 1.6, foot=.0, lit=.18, half=.34, deep=.16, lost=1.0, contrast=1.0)
    if res is None: return np.zeros((h, w, 4), f32)
    c, a, an = res; yy = np.arange(y0, y1, dtype=f32)[:, None]; q = np.clip((y - yy) / height, 0, 1); nz = fbm(*a.shape, (size * .6 + 8, size * .25 + 5), (1, .6), seed + 3)
    a = a * np.clip(1.25 - q * 1.0 + nz * .18 * q, 0, 1) * smooth(-.02, .06, q + .02) * op
    if glow is not None: g = np.exp(-q * 3.2)[..., None] * (.55 + .25 * nz[..., None].clip(-1, 1)); c = c + (C(glow) - c) * np.clip(g, 0, .9)
    out = np.zeros((h, w, 4), f32); X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1); sl = (slice(Y0 - y0, Y1 - y0), slice(X0 - x0, X1 - x0)); a = cv2.GaussianBlur(a, (0, 0), 1.2)[sl]; c = c[sl]
    st = Wd.gouache(c.astype(f32), None, an[sl] if an is not None else None, seed=seed + 3, sizes=(12, 6), op=(.5, .5), val=(.03, .03), hue=.006, thr=(.05, .09), aspect=3., wander=.4)
    out[Y0:Y1, X0:X1] = np.dstack([st * a[..., None], a]); return out


def ruin_lots(r, eye):
    """(foot, h, dmg, kw) for every lot of the square, from world_city.shibuya"""
    out = []
    for (foot, h, st, pal, kw) in shibuya():
        out.append((foot, h, st, pal, kw))
    return out


TINT = dict(snow=np.array([.80, .92, 1.16], f32), fire=np.array([.95, .88, .85], f32), rain=np.array([.86, .92, 1.0], f32), night=np.array([.85, .92, 1.1], f32))


def ruin_scene(seed, cam, mode='snow', crater_c=(0., 4.), crater_r=14., dep=4.4, snow_piles=True, density=1., clear=None, fg=None, pool=True, spike=1.):
    S = Scene(seed + 7); S.groups = 10; r = np.random.default_rng(seed + 71); eye = cam.eye; z = crater(crater_c, crater_r, dep, seed + 3)
    pal_ = [C(c) * TINT.get(mode, 1.) for c in CON]
    S.ground(-460, 460, 60, 1400, 0., '#4c4d56', nx=40, ny=30, pw=1.6, tex=13, seed=33)
    nse = fbm(256, 256, (40, 16, 7), (1, .6, .4), seed + 12)
    def gcol(X, Y):
        d_ = np.hypot(X - crater_c[0], Y - crater_c[1]) / crater_r; th_ = np.arctan2(Y - crater_c[1], X - crater_c[0]); n_ = nse[np.clip(((Y + 110) * 1.5).astype(int), 0, 255), np.clip(((X + 90) * 1.42).astype(int), 0, 255)]
        ray = .5 + .5 * np.sin(th_ * 23 + 3 * np.sin(th_ * 7) + n_) ; c = C('#50515a') * (1 + .10 * n_)[..., None]
        sc = smooth(2.6, 1.0, d_) * (.35 + .65 * ray) * .6; c = c * (1 - .55 * sc)[..., None]
        e_ = smooth(1.06, .86, d_ + .05 * n_)[..., None]; earth = C('#3a302c') * (1 + .28 * n_)[..., None] * (.55 + .5 * np.clip(d_, 0, 1))[..., None]
        return (c * (1 - e_) + earth * e_).astype(f32)
    S.ground(-90, 90, -110, 60, z, gcol, nx=170, ny=170, tex=13, seed=33)
    n0 = len(S.P); markings(S, r, .22)
    for k in range(n0, len(S.P)):                                      # the crossing is torn away where the crater is
        q = S.P[k].mean(1); kk = np.hypot(q[:, 0] - crater_c[0], q[:, 1] - crater_c[1]) > crater_r * 1.16; kk &= (np.sin(q[:, 0] * .55 + 1.3) + np.sin(q[:, 1] * .47 + q[:, 0] * .2)) > -.75
        for nm in ('P', 'Cc', 'Aa', 'Dd', 'Ll', 'B'): getattr(S, nm)[k] = getattr(S, nm)[k][kk]
    plates(S, crater_c, crater_r, r, int(46 * density))
    # crater inside: broken earth, pipes
    for k in range(int(70 * density)):
        th, d = r.uniform(0, 6.28), crater_r * r.random() ** .6 * .98; p = np.array([crater_c[0] + math.cos(th) * d, crater_c[1] + math.sin(th) * d]); s_ = r.uniform(.2, 1.3) * (.5 + d / crater_r)
        if clear is not None and np.hypot(*(p - clear[:2])) < clear[2]: continue
        rock(S, (p[0], p[1], float(z(p[0], p[1])) + s_ * .15), (s_, s_ * r.uniform(.7, 1.2), s_ * .7), r, C(pick(r, ["#6e6258", "#7a7a7e", "#5a5048", "#4a4a50"])), 1)
    for k in range(4): th = r.uniform(0, 6.28); p = np.array([crater_c[0] + math.cos(th) * crater_r * .86, crater_c[1] + math.sin(th) * crater_r * .86]); q = p - np.array([math.cos(th + .4), math.sin(th + .4)]) * r.uniform(2, 4); S.tube([[*p, float(z(*p)) + .3], [*((p + q) / 2), float(z(*p)) + .1], [*q, float(z(*q)) + .7]], .28, '#6a5a52', 8)
    if mode in ('rain', 'night', 'fire') and pool:                        # rain water standing in the bowl
        zw = -dep + 1.15 + .10 * (dep - 4.4); a_ = np.linspace(0, 2 * np.pi, 49)[:-1]; rad = crater_r * .60 * (1 + .10 * np.sin(3 * a_ + 1) + .06 * np.sin(7 * a_)); ring = np.c_[crater_c[0] + np.cos(a_) * rad, crater_c[1] + np.sin(a_) * rad, np.full(48, zw)]; S.poly(ring, '#1a222c', snow=0, lab=S.newlab())
    # scattered rubble on the road, denser near the buildings
    m = int(420 * density); th = r.uniform(0, 2 * np.pi, m); d = crater_r * (1.0 + 3.6 * r.random(m) ** 1.5); P = np.c_[crater_c[0] + np.cos(th) * d, crater_c[1] + np.sin(th) * d]; s_ = np.clip(.18 + .42 * np.abs(r.normal(0, 1, m)), .15, 2.4)
    k = np.hypot(*(P - eye[:2]).T) > 5.; k &= P[:, 1] > eye[1] - 2; s_ = np.where(np.hypot(*(P - eye[:2]).T) < 16, np.minimum(s_, .45), s_)
    if clear is not None: k &= np.hypot(*(P - np.array(clear[:2])).T) > clear[2]
    P, s_ = P[k], s_[k]; rubble(S, P, s_, r, np.array([float(z(a, b)) for a, b in P]), slabs=.3)
    # the broken buildings
    for n_, (foot, h, st, pal, kw) in enumerate(shibuya()):
        c = foot.mean(0); dist = np.hypot(*(c - eye[:2])); rr = np.random.default_rng(int(abs(c[0] * 37 + c[1] * 11)) + seed)
        if np.dot(c - eye[:2], cam.R[2][:2]) < -14: continue
        dm = float(np.clip(1.05 - np.hypot(*c) / 190, .28, .86)) * rr.uniform(.7, 1.15); gl = 'glass' if st in ('curtain', 'fins') else None
        if dist > 150: tower(S, foot, h * 1.05, rr, eye, dm * .8, pick(rr, pal_), bay=7.5, keep=.92, style=gl, skirt=0, nmax=5, dens=.4)
        else: tower(S, foot, h * 1.1, rr, eye, dm, pick(rr, pal_), bay=5.4, keep=.88, style=gl, skirt=1. * density, lean=(rr.normal(0, .02), rr.normal(0, .02)), dens=1. * density)
    # thrown facade pieces: the big diagonals
    up = np.array([0, 0, 1.])
    for (x, y, ww, hh, az_, tilt, roll) in ((-21, 8, 15, 12, 2.6, .95, .25), (20, 27, 17, 13, -.5, .80, -.22), (27, -13, 13, 11, -1.2, 1.1, .3), (-13, 44, 12, 14, .3, .62, .18), (-31, -14, 14, 10, 1.9, 1.2, -.15)):
        B = rot_z(az_) @ rot_x(math.pi / 2 - tilt) @ rot_z(roll); R_ = B.T; hz_ = hh * .5 * abs(R_[1][2]) + ww * .5 * abs(R_[0][2]); facade_slab(S, (x, y, hz_ * .82), ww, hh, R_, r, pick(r, pal_), spike=spike)
    if fg:                                                             # big dark framing pieces close to the lens
        e2 = np.array(eye[:2]); fw = cam.R[2][:2] / np.hypot(*cam.R[2][:2]); rt = np.array([fw[1], -fw[0]])
        for (a_, b_, ww, hh, az_, tilt, roll) in fg:
            p = e2 + fw * a_ + rt * b_; B = rot_z(az_) @ rot_x(math.pi / 2 - tilt) @ rot_z(roll); R_ = B.T; hz_ = hh * .5 * abs(R_[1][2]) + ww * .5 * abs(R_[0][2]); facade_slab(S, (*p, hz_ * .62), ww, hh, R_, r, pal_[1] * .8, 2.4, 2.6, .34, spike)
            q = p + r.normal(0, 1., (16, 2)) * [ww * .35, 1.2]; rubble(S, q, r.uniform(.25, .9, 16), r, 0., [tuple(c) for c in pal_], 1., .4)
    if mode in ('snow', 'ash'):
        n_ = 150; th = r.uniform(0, 2 * np.pi, n_); d_ = crater_r * (1.1 + 3.4 * r.random(n_) ** 1.2); drifts(S, np.c_[crater_c[0] + np.cos(th) * d_, crater_c[1] + np.sin(th) * d_], r, '#dfe6f4' if mode == 'snow' else '#a8a29c')
    # the dead screen on the wedge, hanging off its frame
    L0 = _edge('NL', +1, 42, 46)[0]; R0 = _edge('NR', -1, 42, 46)[0]; aL = _edge('NL', +1, 53.5, 54)[0]; aR = _edge('NR', -1, 53.5, 54)[0]
    tower(S, np.array([L0, R0, aR, aL]), 33, np.random.default_rng(seed + 4), eye, .42, C('#7a7c84') * TINT.get(mode, 1.), bay=4.6, keep=.6, skirt=1.)
    W = Wall(S, L0 + [0, -.8], R0 + [0, -.8], 11, 31); nx, ny = 9, 10
    for i in range(nx):
        for j in range(ny):
            q = r.random(); u0, v0 = i * W.w / nx, j * W.h / ny; du, dv = W.w / nx, W.h / ny
            if q < .34 + .25 * (j > 6) + .2 * (i > 6): continue
            t = r.normal(0, .05); 
            if q > .93: t = r.uniform(.3, .9)
            W.slab(u0 + .04, u0 + du - .04, v0 + .04 - t, v0 + dv - .04 - t, 0 + abs(t) * .4, .22 + abs(t) * .4, mix(C('#16181f'), C('#2c3442'), r.random()), side='#0c0d12', em=.1, snow=.5)
    for i in range(nx + 1): S.seg([W.p(i * W.w / nx, 0, -.1)], [W.p(i * W.w / nx, W.h * r.uniform(.5, 1.), -.1)], .14, '#3a3034')
    for j in range(ny): S.seg([W.p(0, j * W.h / ny, -.1)], [W.p(W.w * r.uniform(.4, 1.), j * W.h / ny, -.1)], .10, '#3a3034')
    for _ in range(14): u = r.uniform(0, W.w); v = r.uniform(4, W.h); S.wire(W.p(u, v, .3), W.p(u + r.normal(0, 2.4), v - r.uniform(3, 9), .5), r.uniform(0, .8), .05, '#141216')
    # drum tower, sheared
    cc = U(-55) * 45; RR = 9.6; a = np.linspace(0, 2 * np.pi, 25)[:-1]
    top = 21 + 9 * np.sin(a * 1. + .6) + r.normal(0, 1.3, 24); ft = np.c_[cc[0] + np.cos(a) * RR, cc[1] + np.sin(a) * RR]; nx_ = np.roll(ft, -1, 0); tn = np.roll(top, -1)
    S.quads(np.stack([np.c_[ft, np.zeros(24)], np.c_[nx_, np.zeros(24)], np.c_[nx_, tn], np.c_[ft, top]], 1), C('#86848a') * TINT.get(mode, 1.), tex=6, lab=S.newlab(24)); fi = np.c_[cc[0] + np.cos(a) * RR * .93, cc[1] + np.sin(a) * RR * .93]; fin = np.roll(fi, -1, 0)
    S.quads(np.stack([np.c_[ft, top], np.c_[nx_, tn], np.c_[fin, tn - .1], np.c_[fi, top - .1]], 1), '#a09a94', tex=9); S.quads(np.stack([np.c_[fi, np.zeros(24)], np.c_[fin, np.zeros(24)], np.c_[fin, tn - .1], np.c_[fi, top - .1]], 1), '#2c2a30', tex=1)
    for k in range(5): zz = 4 + k * 4.4; S.poly(np.c_[fi, np.full(24, zz)], '#4a474c', tex=1)
    for k in range(24): rebar(S, [ft[k][0], ft[k][1], top[k]], r, 3, .3, (.4, 1.5), (0, 0, 1))
    for q in (-.42, .5): ang = math.atan2(eye[1] - cc[1], eye[0] - cc[0]); dv = np.array([math.cos(ang + q), math.sin(ang + q)]); p = cc + dv * (RR + .4); ex = np.array([-math.sin(ang + q), math.cos(ang + q), 0]); F = Wall.frame(S, np.r_[p, 5.] - ex * 3.6, ex, (0, 0, 1), 7.2, 13); F.slab(0, 7.2, 0, 13, -.3, 0, '#20222a'); F.poly([(.4, .4), (6.8, .4), (6.8, 9), (4.4, 12.4), (.4, 8.2)], .03, C('#5a2a30' if q < 0 else '#38304e') * TINT.get(mode, 1.), tex=1, snow=.4)
    # leaning / fallen towers in the second rank
    for (ang_, dist, w, dp, hh, dm, ln) in ((-44, 150, 32, 28, 82, .35, (.05, -.12)), (-28, 205, 28, 26, 58, .5, (0, .05)), (-10, 250, 30, 30, 66, .4, (-.04, .02)), (8, 320, 34, 30, 74, .3, (0, 0)), (20, 235, 26, 26, 60, .5, (.02, .10)), (38, 160, 34, 30, 88, .42, (.03, .07)), (57, 130, 30, 28, 70, .5, (0, -.06)), (-64, 120, 30, 26, 64, .5, (0, .08))):
        p = U(ang_) * dist; rot = math.radians(-ang_ * .6); c_, s2 = math.cos(rot), math.sin(rot); q = np.array([(-w, -dp), (w, -dp), (w, dp), (-w, dp)]) / 2 @ np.array([[c_, s2], [-s2, c_]]) + p; tower(S, q, hh, np.random.default_rng(int(dist) + seed), eye, dm, pick(r, pal_), bay=7.4, keep=.92, style='glass' if dist > 300 else None, skirt=0, lean=ln, nmax=5, dens=.35)
    # far broken skyline (flat masses with notches)
    for k in range(70):
        an = math.radians(r.uniform(-62, 62)); d = r.uniform(340, 950); x, y = math.sin(an) * d, math.cos(an) * d; w = r.uniform(18, 46); h = r.uniform(14, 70); n_ = 6; t = np.linspace(-.5, .5, n_); hz = h * (1 - .5 * np.abs(r.normal(0, .5, n_))).clip(.25, 1); ex = np.array([math.cos(an), -math.sin(an)]); A = np.array([x, y]) + ex * w * t[:, None]
        S.quads(np.stack([np.c_[A[:-1], np.zeros(n_ - 1)], np.c_[A[1:], np.zeros(n_ - 1)], np.c_[A[1:], hz[1:]], np.c_[A[:-1], hz[:-1]]], 1), pick(r, pal_), tex=7)
    # street furniture wrecked
    for (x, y, rx, ry, hh) in ((14.5, -19.5, .35, -.5, 7.), (-15.5, 18.5, -.6, .2, 6.5), (17, 21, .1, .75, 7.5), (-19, -8, .15, -.22, 8.)):
        T = Scene(3); T.r = r; signal(T, (x, y), (-5 if x > 0 else 5, .6), h=hh, col='none', z=0.); transform(T, S, rot_x(rx) @ rot_y(ry), (x, y, 0))
    an = []
    for (x, y, rx, ry) in ((-13.5, -22, .03, .10), (15.5, 2, -.05, -.16), (-17.5, 24, .06, .22), (14, 30, .10, -.08), (-30, 12, 0, .3), (6, 52, 0, .12)):
        T = Scene(4); T.r = r; w_ = pole(T, (x, y), 10.5, r, 0., ((1, 0),)); R = rot_x(rx) @ rot_y(ry); an.append([(q - [x, y, 0]) @ R.T + [x, y, 0] for q in w_]); transform(T, S, R, (x, y, 0))
    for i in range(len(an) - 1):
        for j in (0, 2, 5):
            if r.random() < .7: S.wire(an[i][j], an[i + 1][j], r.uniform(1.2, 3.5), .035, '#17161c')
        q = an[i][3]; S.wire(q, np.r_[q[:2] + r.normal(0, 3, 2), .1], .4, .035, '#17161c')
    for (x, y, a_, kd, tl) in ((-9.5, -13, .6, 'white', (.1, .16)), (19, 8, 2.2, 'taxi', (.0, .5)), (12, 24, 1.1, 'black', (.25, .1)), (-20, 2, 2.7, 'van', (1.55, 0)), (-6, 27, .2, 'red', (3.05, .1)), (3, 38, 1.8, 'bus', (.05, .2)), (9, -24, 2.0, 'blue', (.1, -.2))):
        T = Scene(5); T.r = r; car(T, x, y, a_, r, kd, .0, 0., wreck=1.0); T.Cc = [c_ * np.array([.42, .36, .34], f32) * (1 + .25 * T.r.normal(0, 1, (len(c_), 1, 1))).astype(f32) for c_ in T.Cc]; T.P = [p_ * np.array([1, 1, .78], f32) for p_ in T.P]; transform(T, S, rot_x(tl[0]) @ rot_y(tl[1]), (x, y, .6), (0, 0, .15 if tl[0] < 1 else .8))
    return S, r, z


MODES = dict(
    fire=dict(amb=('#161828', '#140a0c'), fills=[((-.7, -.4, .2), '#0c1424'), ((.8, -.3, .0), '#42180a'), ((0, 1, .3), '#5a2410')], fog=dict(col='#5a2416', col2='#b0481c', dir=(.2, 1, .05), dens=.0052, start=24, hf=.006, floor=.6, keep=.3, pw=3.), street=('#a03c14', 8., (4, 40, 70, .35))),
    rain=dict(amb=('#6580a6', '#10141f'), fills=[((-.8, -.3, .1), '#0a1422'), ((.3, 1, .3), '#33424f')], sun=((.12, 1, .42), '#8fa6ba', .40), fog=dict(col='#6f86a0', col2='#9fb0bc', dir=(.2, 1, .05), dens=.0115, start=26, hf=.006, floor=.7, keep=.2, pw=7., up='#4a5a72')),
    snow=dict(amb=('#9bb0dc', '#1e2c54'), fills=[((-.8, -.4, .1), '#0a1430'), ((.5, .8, .1), '#2a2430')], sun=((.385, .92, .135), '#ffb482', 1.0), fog=dict(col='#8ea8d8', col2='#f6c8c0', dir=(.40, 1, .05), dens=.0082, start=22, hf=.004, floor=.75, keep=.1, pw=70., up='#86a2d2'), snow=(1.0, '#dfe8fa', .22)),
    ash=dict(amb=('#8a8894', '#24222a'), fills=[((-.7, -.4, .2), '#0c0f18'), ((.8, -.2, .1), '#33190c')], sun=((.62, -.30, .55), '#e6b48c', .48), fog=dict(col='#a8a3a6', col2='#e0b890', dir=(.4, 1, .02), dens=.0060, start=20, hf=.004, floor=.8, keep=.2, pw=12., up='#9a969c'), snow=(.60, '#aca7a8', .35)),
    night=dict(amb=('#2c3a60', '#10131f'), fills=[((-.8, -.4, .2), '#0c1830'), ((.8, -.2, .2), '#1c1430')], fog=dict(col='#2a3250', col2='#4a4468', dir=(0, 1, 0), dens=.005, start=24, hf=.004, floor=.8, keep=.2, pw=3.)))

VIEWS = dict(dawn=dict(eye=(-7., -41., 2.3), target=(3.4, 10., 9.2), focal=1260., fg=((8.5, -6.2, 7., 6., .5, 1.15, .35), (11., 7.6, 8., 5., -.7, .75, -.2))), crater=dict(eye=(-6., -38., 19.), target=(1., 10., 2.2), focal=1220., dep=6.6, R=15.5), low=dict(eye=(-3.2, -38., 1.9), target=(1., 8., 5.6), focal=1180.), down=dict(eye=(0., -21., 17.), target=(0., 4., 0.), focal=1150.), wide=dict(eye=(-16., -62., 24.), target=(6., 30., 14.), focal=1120.), up=dict(eye=(3., -30., .9), target=(-2., 10., 11.), focal=1050.))


@painter('ruin', 5)
def ruin(mode='snow', view='crater', w=2400, h=1350, seed=0, eye=None, target=None, focal=None, paint=True, ss=2, density=1., figure=None, dep=4.4, crater_r=14., cr=(0., 4.), spike=1.):
    """ruined Shibuya.  mode fire|rain|snow|ash|night, view crater|low|down|wide|up (or eye/target/focal).  figure=(x,y): keep that spot clear.
    masks sky, ground, ember (where fire burns).  extra['ground_z'] = ground height at figure.  layers: 'smoke' (over sky, behind nothing)"""
    v = VIEWS[view]; dep = v.get('dep', dep); crater_r = v.get('R', crater_r); cam = Camera3D(eye or v['eye'], target or v['target'], focal or v['focal'], w, h)
    S, r, z = ruin_scene(seed, cam, mode, cr, crater_r, dep, density=density, clear=None if figure is None else (*figure, 2.2), fg=v.get('fg'), spike=spike)
    hz = Wd.horizon_y(cam); m = {k: v_ for k, v_ in MODES[mode].items()}
    fires = []
    if mode in ('fire', 'ash', 'rain', 'night'):
        nfr = dict(fire=26, ash=7, rain=3, night=8)[mode]; rr = np.random.default_rng(seed + 5)
        for k in range(nfr):
            th = rr.uniform(-1.1, 1.1); d = rr.uniform(16, 95); p = np.array([math.sin(th) * d, math.cos(th) * d + 4]); zz = rr.uniform(.4, 13) if rr.random() < .5 else .6; s_ = rr.uniform(.8, 2.6); fires.append((*p, zz, s_)); S.light((p[0], p[1] - 1.5, zz + 1.), '#ff7a2a', 22 * s_ ** .4, 5, 2.4 if mode == 'fire' else 1.5, 1)
    sky = dict(fire=lambda: SK.fire(h, w, hz + 8, seed + 2, .56), rain=lambda: SK.storm(h, w, hz + 8, seed + 2), snow=lambda: SK.dawn(h, w, hz + 8 - .15 * h, seed + 2, .60), ash=lambda: SK.storm(h, w, hz + 8, seed + 4, ('#b8b0ae', '#8e888a', '#625e64', '#45424a'), hzc='#e6c8a6', top='#4a484e'), night=lambda: SK.storm(h, w, hz + 8, seed + 6, ('#3c4868', '#283250', '#1a2038', '#101424'), hzc='#4a4a70', top='#0c1020'))[mode]()
    baked = S._bake(cam); res = S.render(cam, ss=ss, baked=baked, smres=4096, shadow_box=(-110, 110, -70, 190, 70) if 'sun' in m else None, **m)
    X = cam.eye + cam.rays() * res['z'][..., None]; geo = (res['z'] < FAR * .5).astype(f32)
    gm = ((np.abs(X[..., 2]) < .35) & (geo > 0)).astype(f32)
    img = res['img'] * (1 + (Wd.occlusion(res['z'], .46, (3, 9, 24), 9.) - 1) * geo)[..., None]
    if mode in ('rain', 'fire', 'night'):
        wl = ((np.abs(X[..., 2] - (-dep + 1.15 + .10 * (dep - 4.4))) < .04) & (geo > 0)).astype(f32); pud = smooth(-.3, .8, fbm(h, w, (150, 54, 20), (1, .6, .3), seed + 41, aspect=3.0)); wm = np.maximum(cv2.GaussianBlur(gm, (0, 0), 1.2) * (.18 + (.80 if mode == 'rain' else .45) * pud), cv2.GaussianBlur(wl, (0, 0), .8) * 1.15); r2 = dict(res); r2['img'] = img; img, _ = Wd.wet_reflect(S, cam, r2, sky, wm, baked=baked, strength=.8, vblur=12, ripple=2.6, seed=seed + 3, **m)
        if mode == 'rain': img = img + (C('#b9cad8') * (.75 + .25 * smooth(-.6, .8, fbm(h, w, (120, 30), (1, .5), seed + 44, aspect=5.)))[..., None] - img) * (cv2.GaussianBlur(wl, (0, 0), 1.) * .50)[..., None]
    # smoke columns between sky and buildings
    sm = np.zeros((h, w, 4), f32)
    if mode != 'snow':
        tn = dict(fire=('#c8552a', '#6e2a1e', '#341618', '#1a0e12'), rain=('#63748a', '#4a5a70', '#354356', '#263040'), ash=('#b0a69e', '#8e8682', '#6e6866', '#54504f'), night=('#39435e', '#283148', '#1b2134', '#121624'))[mode]
        for k, (fx, sz, ht, ln) in enumerate(((.22, .085, .62, .22), (.66, .11, .78, .16), (.86, .06, .46, .30), (.44, .05, .36, .25))):
            L = smoke(h, w, w * fx, hz + 20, w * sz, h * ht, seed + 11 * k + 3, ln, tn, light=(.3, .8, .2) if mode == 'fire' else (-.4, -.7, .3), op=.9, glow='#ff7a30' if mode == 'fire' else None); sky = sky * (1 - L[..., 3:]) + L[..., :3]
    else:
        for k, (fx, sz, ht, ln) in enumerate(((.30, .035, .34, .5), (.60, .026, .26, .6))):
            L = smoke(h, w, w * fx, hz + 10, w * sz, h * ht, seed + 11 * k + 3, ln, ('#b4bdd6', '#92a3c8', '#7a8fbc', '#6480b0'), light=(.6, -.5, .2), op=.55); sky = sky * (1 - L[..., 3:]) + L[..., :3]
    img = img + sky * (1 - res['a'][..., None])
    em = np.zeros((h, w), f32)
    if fires:
        img, em = paint_fires(img, cam, fires, res['z'], seed + 9, 1. if mode == 'fire' else .55)
    if paint:
        img = Wd.wobble(img, .5, 30, seed + 12); ang = res['ang'] + fbm(h, w, (140, 50), (1, .5), seed + 8) * .16
        img = Wd.gouache(img, res['lab'], ang, seed=seed + 4, sizes=(18, 9, 4.5), op=(.62, .58, .52), val=(.046, .042, .036), hue=.016, thr=(.018, .032, .06), mask=geo)
        if mode == 'fire': img = Wd.air_glow(img, res['z'], .5, (40, 130), .35, (10, 120), '#ffb080')
        if mode == 'snow':
            img = Wd.air_glow(img, res['z'], .80, (60, 160), .16, (20, 200), '#ffe0c8'); vp, vz = cam.project(cam.eye + np.array(m['sun'][0]) * 1e4)
            if vz > 0:                                                  # low sun through the street: three soft shafts
                from .comp import shaft
                rr = np.random.default_rng(seed + 77); bm = np.zeros((h, w), f32); vx, vy = vp
                for (dx0, dx1, w0, w1, a_) in ((-.46, -.10, .012, .20, .5), (-.12, .02, .012, .16, .6), (.22, .10, .010, .13, .35)): bx = vx + dx0 * w; bm += a_ * shaft(h, w, [(vx - w0 * w, vy), (vx + w0 * w, vy), (bx + w1 * w, h * 1.05), (bx - w1 * .4 * w, h * 1.05)], 26, int(rr.integers(99)), .5)
                y_ = grid(h, w)[1]; bm = bm * smooth(vy - 30, vy + 120, y_) * (1 - .55 * smooth(h * .62, h, y_)); glow = np.exp(-(np.hypot((grid(h, w)[0] - vx) / (w * .17), (y_ - vy) / (h * .20))) ** 1.5); img = img + C('#ffc9a0') * (np.clip(bm, 0, 1.2) * .13 + glow * .34 * (1 - .4 * geo))[..., None] * (1 - np.clip(img, 0, 1) * .6)
        img = Wd.tooth_pass(img, seed + 6)
    gz = float(z(np.array(figure[0]), np.array(figure[1]))) if figure is not None else float(z(np.array(cr[0]), np.array(cr[1])))
    return Plate(img, res['z'], cam, dict(sky=1 - res['a'], ground=gm, ember=np.clip(em, 0, 1)), extra=dict(ground_z=gz, water_z=-dep + 1.15 + .10 * (dep - 4.4)))


def paint_fires(img, cam, fires, z, seed, k=1.):
    """tongues of painted flame sitting in the ruins (depth tested)"""
    h, w = img.shape[:2]; r = np.random.default_rng(seed); em = np.zeros((h, w), f32); out = img
    for (x, y, zz, s) in sorted(fires, key=lambda t: -t[1]):
        p, d = cam.project(np.array([x, y, zz])); d = float(d)
        if d < 2 or not (-200 < p[0] < w + 200): continue
        q = cam.f / d * s * 1.5; x0, y0 = int(p[0] - q * 2), int(p[1] - q * 3.2); x1, y1 = int(p[0] + q * 2), int(p[1] + q * .7); X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if X1 - X0 < 3 or Y1 - Y0 < 3: continue
        ww, hh = X1 - X0, Y1 - Y0; cv = Canvas(hh, ww, color='#ff7a22', layer=True); n = int(np.clip(q * .9, 10, 46))
        for c_, f_ in (('#b8301a', 1.), ('#ff6a1e', .78), ('#ffb03a', .52), ('#fff0b4', .28)):
            m = max(4, int(n * f_)); xx = p[0] - X0 + r.normal(0, .55, m) * q * f_; L = q * r.uniform(.7, 2.4, m) * f_; yy = p[1] - Y0 + r.normal(0, .12, m) * q - L * .42; P = arcs(xx, yy, -np.pi / 2 + r.normal(0, .22, m), L, r.normal(0, .9, m), K=6); cv.strokes(P, q * r.uniform(.08, .2, m) * f_ ** .5 + .6, np.tile(rgb(c_), (m, 1)), op=.95, dry=.06, seed=int(r.integers(1 << 20)), shape='leaf', bristle=.04)
        vis = (z[Y0:Y1, X0:X1] > d - 1.5 - s).astype(f32); a = cv2.GaussianBlur(cv.a, (0, 0), .7) * vis * min(1., .35 + k)
        out[Y0:Y1, X0:X1] = out[Y0:Y1, X0:X1] * (1 - a[..., None]) + cv.img * a[..., None] * 1.15; em[Y0:Y1, X0:X1] = np.maximum(em[Y0:Y1, X0:X1], a)
    g = blur(em, 14) * .5 + blur(em, 46) * .7
    return out + C('#ff6a22') * (g * .55 * k)[..., None] * (1 - np.clip(out, 0, 1) * .5), em
