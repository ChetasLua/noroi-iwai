"""Painted ruined street in true perspective (gouache over a 3-D block-in) for figure stills and h2."""
import numpy as np, cv2, math
from . import W, H
from .paint import rgb, f32, fbm, smooth, mix, ramp, repaint, Canvas, arcs, grid
from .cam import Camera3D
from . import comp

NIGHT = dict(sky=[(0, '#0b0e1c'), (.45, '#1c2038'), (.75, '#4a3444'), (1, '#b0603c')], far='#4a3440', haze='#6a4238', hazed=95., left='#22304a', right='#171c2e', front='#1b2236', win='#0c1222', winl='#3a5474', lit='#ffb45a', road='#2a2c3a', roadl='#4a4450', stripe='#8a8690', sign=['#7a2430', '#24506a', '#6a5a2a', '#3a2c5a', '#9a9288'], glow='#ff7a34')
ASH = dict(sky=[(0, '#23202a'), (.5, '#4a4048'), (.8, '#8a6a5c'), (1, '#d89a6a')], far='#8a6c62', haze='#a88270', hazed=70., left='#56525c', right='#3a343e', front='#474049', win='#221f28', winl='#78706e', lit='#ff9a4a', road='#4a444a', roadl='#7a6c68', stripe='#a89c94', sign=['#8a3a34', '#3a5a6a', '#7a6a3a', '#54405a', '#b0a494'], glow='#ff8a3c')


def _poly(guide, lab, pts, col, k, z=None):
    q = np.round(np.asarray(pts) * 16).astype(np.int64)
    if np.abs(q).max() > 2 ** 24: return
    q = q.astype(np.int32)
    cv2.fillPoly(guide, [q], tuple(float(c) for c in col), cv2.LINE_AA, 4)
    if k is not None: cv2.fillPoly(lab, [q], int(k), cv2.LINE_8, 4)


def quad(cam, pts):
    """project a 3-D polygon, clipped at the near plane"""
    c = cam.cam(np.asarray(pts, float)); out = []
    n = .08
    for i in range(len(c)):
        a, b = c[i], c[(i + 1) % len(c)]
        if a[2] > n: out.append(a)
        if (a[2] > n) != (b[2] > n): t = (n - a[2]) / (b[2] - a[2]); out.append(a + (b - a) * t)
    if len(out) < 3: return None
    o = np.array(out); return np.c_[o[:, 0] / o[:, 2] * cam.f + cam.cx, o[:, 1] / o[:, 2] * cam.f + cam.cy]


def street(cam, seed=1, pal=NIGHT, width=8.5, depth=(1., 120.), lit=.05, damage=1., road=True, stripes=True, paint=True, sizes=(26, 11, 5), side_fire=1.):
    """-> float RGB frame.  The street runs along +Y, centred on x=0"""
    h, w = cam.h, cam.w; r = np.random.default_rng(seed); g = np.zeros((h, w, 3), f32); lab = np.zeros((h, w), np.int32)
    # sky
    d = cam.rays(); up = d[..., 2] / np.linalg.norm(d, axis=-1)
    g[:] = ramp(1 - np.clip(up / .75, 0, 1), pal['sky'])
    x, y = grid(h, w); n1 = fbm(h, w, (260, 120, 50, 22), (1, .7, .4, .2), seed + 3)
    cl = smooth(-.1, .9, n1)[..., None]; g[:] = g * (1 - .45 * cl) + rgb(pal['haze']) * .55 * cl * .45 + g * .45 * cl * .55
    k = 1
    B = []
    for s in (-1, 1):
        yy = depth[0] + r.uniform(-3, 1)
        while yy < depth[1]:
            ln = r.uniform(7, 15); B.append((s, yy, yy + ln * .96, r.uniform(13, 40) * (1.15 if r.random() < .25 else 1.), width + r.uniform(-.4, 1.1), r.integers(1 << 30))); yy += ln + (r.uniform(2.5, 4.5) if r.random() < .22 else r.uniform(0, .6))
    for q in range(5): B.append((0, depth[1] + 10, depth[1] + 20, r.uniform(25, 70), -22 + q * 10 + r.uniform(-2, 2), r.integers(1 << 30)))
    B.sort(key=lambda b: -b[1])
    hz = rgb(pal['haze'])
    def fog(c, dist): t = 1 - math.exp(-max(dist - 4, 0) / pal['hazed']); return rgb(c) * (1 - t) + hz * t if isinstance(c, str) else c * (1 - t) + hz * t
    def facade(o, e1, L, Hh, base, rs, dist, side):
        """rectangular wall from world corner o along e1 (len L), height Hh"""
        nonlocal k
        e1 = np.asarray(e1, float); up_ = np.array([0, 0, 1.])
        P = lambda u, v: o + e1 * u + up_ * v
        # broken top
        top = []
        m = max(5, int(L / 1.4)); cut = damage * (rs.random() < .6)
        for i in range(m + 1): top.append((L * i / m, Hh - cut * abs(rs.normal(0, 1)) * 2.6 * (rs.random() < .7)))
        W3 = [P(0, 0)] + [P(u, v) for u, v in top] + [P(L, 0)]
        q = quad(cam, W3)
        if q is None: return
        k += 1; kk = k; b = fog(base, dist); _poly(g, lab, q, b, kk)
        # vertical light fall: warmer and paler near the ground fires
        fl = 3.5; cw = rs.uniform(1.6, 2.6); nf = int(Hh / fl); nc = max(1, int(L / cw))
        if dist > 75: nf = min(nf, 8)
        lev = (math.floor(Hh / fl)) * fl
        # floor bands
        for f_ in range(1, nf):
            q = quad(cam, [P(0, f_ * fl - .18), P(L, f_ * fl - .18), P(L, f_ * fl + .10), P(0, f_ * fl + .10)])
            if q is not None: _poly(g, lab, q, b * 1.16 + .010, None)
        mode = rs.integers(3)
        for f_ in range(1, nf):
            for c_ in range(nc):
                if rs.random() < .06 * damage: continue
                u0 = (c_ + .14) * L / nc; u1 = (c_ + (.86 if mode else .96)) * L / nc; v0 = f_ * fl + .55; v1 = f_ * fl + fl - .55
                tmax = np.interp((u0 + u1) / 2, [t[0] for t in top], [t[1] for t in top])
                if v1 > tmax - .2: continue
                q = quad(cam, [P(u0, v0), P(u1, v0), P(u1, v1), P(u0, v1)])
                if q is None: continue
                wk = None
                if dist < 45: k += 1; wk = k
                z = rs.random(); col = fog(pal['win'], dist) * rs.uniform(.8, 1.2)
                if z < lit: col = fog(pal['lit'], dist * .5)
                elif z < .30: col = fog(pal['winl'], dist) * rs.uniform(.7, 1.1)
                _poly(g, lab, q, col, wk)
                if z > .88 and damage:                               # smashed pane
                    ms = P((u0 + u1) / 2, (v0 + v1) / 2); j = [P(u0 + (u1 - u0) * (.5 + .5 * math.cos(a) * rs.uniform(.3, 1)), v0 + (v1 - v0) * (.5 + .5 * math.sin(a) * rs.uniform(.3, 1))) for a in np.linspace(0, 6.28, 9)[:-1]]
                    q = quad(cam, j)
                    if q is not None: k += wk is not None; _poly(g, lab, q, fog('#05060a', dist), k if wk is not None else None)
                elif z > .45:                                        # slanting reflection
                    q = quad(cam, [P(u0, v0 + (v1 - v0) * .3), P(u0 + (u1 - u0) * .5, v1), P(u0 + (u1 - u0) * .8, v1), P(u0, v0)])
                    if q is not None: k += wk is not None; _poly(g, lab, q, col * 1.35 + .012, k if wk is not None else None)
        # street level: shutters and signs
        q = quad(cam, [P(0, 0), P(L, 0), P(L, 3.2), P(0, 3.2)])
        if q is not None: _poly(g, lab, q, b * .55, None)
        for i in range(max(1, int(L / 3.2))):
            u0 = (i + .1) * L / max(1, int(L / 3.2)); u1 = u0 + L / max(1, int(L / 3.2)) * .8
            q = quad(cam, [P(u0, .1), P(u1, .1), P(u1, 2.5), P(u0, 2.5)])
            if q is not None: k += 1; _poly(g, lab, q, fog(pal['win'], dist) * rs.uniform(.6, 1.6), k if dist < 40 else None)
            if rs.random() < .7:
                q = quad(cam, [P(u0, 2.7), P(u1, 2.7), P(u1, 3.5), P(u0, 3.5)])
                if q is not None: k += 1; _poly(g, lab, q, fog(pal['sign'][rs.integers(len(pal['sign']))], dist) * rs.uniform(.5, .9), k if dist < 40 else None)
        # blade signs sticking out
        if side and dist < 60:
            for i in range(rs.integers(1, 4)):
                u = rs.uniform(.5, L - .5); v = rs.uniform(4, max(6, min(Hh - 5, 14))); hh = rs.uniform(2., 5.); n_ = np.array([-side, 0, 0.])
                tl = rs.normal(0, .10)
                q = quad(cam, [P(u, v) + n_ * .1, P(u, v) + n_ * 1.3 + up_ * tl, P(u, v + hh) + n_ * 1.3 + up_ * tl, P(u, v + hh) + n_ * .1])
                if q is not None:
                    k += 1; c_ = fog(pal['sign'][rs.integers(len(pal['sign']))], dist) * rs.uniform(.55, 1.0); _poly(g, lab, q, c_, k)
                    for j in range(3):
                        v0 = v + hh * (.12 + .29 * j); q = quad(cam, [P(u, v0) + n_ * .35, P(u, v0) + n_ * 1.05, P(u, v0 + hh * .18) + n_ * 1.05, P(u, v0 + hh * .18) + n_ * .35])
                        if q is not None: k += 1; _poly(g, lab, q, np.minimum(c_ * 1.9 + .05, .9), k)
    for s, y0, y1, Hh, x0, sd in B:
        rs = np.random.default_rng(sd)
        if s == 0:
            facade(np.array([x0, y0, 0.]), (1, 0, 0), 9., Hh, pal['far'], rs, y0 - cam.eye[1], 0); continue
        dist = y0 - cam.eye[1]
        tint = rs.uniform(.85, 1.18)
        # side wall along the street, then the end wall facing us
        facade(np.array([s * x0, y0, 0.]), (0, 1, 0), y1 - y0, Hh, rgb(pal['left' if s < 0 else 'right']) * tint, rs, max(dist, 1), s)
        a = s * x0 if s > 0 else -x0 - 16.
        facade(np.array([a, y0, 0.]), (1, 0, 0), 16., Hh, rgb(pal['front']) * tint, rs, max(dist, 1), 0)
    # road
    if road:
        u, v, t = cam.uv((0, 0, 0), (1, 0, 0), (0, 1, 0))
        m = t > 0
        m &= (np.abs(u) < width + .2) | (lab == 0)
        bl = 1 - np.exp(-np.maximum(t - 3, 0) / (pal['hazed'] * .8)); c = rgb(pal['road']) * (1 - bl[..., None]) + hz * bl[..., None]
        nn = fbm(h, w, (90, 30, 10), (1, .7, .5), seed + 8)[..., None]; c = c * (1 + .12 * nn)
        c = np.where((np.abs(u) > width - 2.2)[..., None], c * 1.18 + .012, c); c = np.where((np.abs(np.abs(u) - (width - 2.2)) < .10)[..., None], c * .55, c)
        if stripes:
            st = ((np.abs(((u + 20) % .95) - .475) < .235) & (((v > 1.2) & (v < 5.2)) | ((v > 14) & (v < 18))) & (np.abs(u) < width - 2.6))
            worn = smooth(-.6, .5, fbm(h, w, (40, 12, 5), (1, 1, .8), seed + 21))
            sc = rgb(pal['stripe']) * (1 - bl[..., None]) + hz * bl[..., None]; c = np.where(st[..., None], c * (1 - .8 * worn[..., None]) + sc * .8 * worn[..., None], c)
            ln = (np.abs(u) < .09) & ((v % 6) < 3.2) & (v > 6); c = np.where(ln[..., None], c * .4 + sc * .6, c)
        g[m] = c[m]; lab[m] = np.where(np.abs(u)[m] > width - 2.2, 900001, 900000)
        if stripes: lab[m & st & (worn > .45)] = 900002
        # fire-light lying on the asphalt
        if side_fire:
            f_ = np.exp(-((u - width) / 5.) ** 2) * np.exp(-((v - 9) / 12.) ** 2) * m; g[:] = g + f_[..., None] * rgb(pal['glow']) * .30 * side_fire
    if side_fire:
        lg = np.clip(1 - np.abs(up - .02) / .42, 0, 1) ** 2; e = smooth(.35, 1., x / w)
        g[:] = comp.screen(g, (lg * e * .34 * side_fire)[..., None] * rgb(pal['glow']))
    if paint:
        ang = None
        g = repaint(np.clip(g, 0, 1), sizes=sizes, seed=seed + 2, lab=lab, thr=(0, .015, .015), density=1.0, aspect=4., jit=(.010, .02, .035), op=.95, dry=.2)
    return g


def rubble(cam, n=120, seed=4, box=(-7., 7., 1., 30.), size=(.15, 1.4), pal=NIGHT, lw=1.6, light=(-.5, -.75, .45), lit=None):
    """broken concrete on the road (fx.rock solids), heaped along the sides, far to near -> premultiplied layer"""
    from .ink import Cel
    from . import fx
    r = np.random.default_rng(seed); cel = Cel(cam.w, cam.h, 0, 0, ss=2); P = np.c_[r.uniform(box[0], box[1], n), box[2] + (box[3] - box[2]) * r.random(n) ** 1.4]
    P[:, 0] = np.sign(P[:, 0]) * np.abs(P[:, 0] / box[1]) ** .55 * box[1]
    for x, y in P[np.argsort(-P[:, 1])]:
        s = size[0] + (size[1] - size[0]) * r.random() ** 3.2
        p, z = cam.project(np.array([x, y, s * .16]))
        if z < .6: continue
        k = cam.f / z * s * .5
        if k < 2.5 or k > 260: continue
        t = 1 - math.exp(-max(y - cam.eye[1] - 3, 0) / pal['hazed']); f = lambda c, q: rgb(c) * q * (1 - t) + rgb(pal['haze']) * t
        v = r.uniform(.8, 1.15); pl = dict(lit=f(lit or pal['roadl'], v * 1.35), mid=f(pal['roadl'], v * .85), dark=f(pal['road'], v * .62), ink='#0d0b11')
        fx.rock(cel, p[0], p[1], k, tuple(r.uniform(0, 6.28, 3)), seed=int(r.integers(24)), pal=pl, light=light, lw=lw * min(1.4, k / 45 + .45), crack=k > 24, kind='rock')
    return cel.rgba()
