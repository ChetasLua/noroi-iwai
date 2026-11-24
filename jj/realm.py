"""The Prison Realm: a palm-sized cube of grey flesh-like stone with a sleeping eye on every face.

    R = Face(seed, size);  R.tex(open=0..1, gaze=(0,0), pupil=.36) -> RGB (size, size, 3)
    warp_face(img_layer, tex, quad, shade, blur) ;  Cube(h).draw(cam, pos, rot, opens, light) -> premult layer
"""
import numpy as np, cv2, math, os
from . import W, H
from .paint import rgb, f32, fbm, smooth, mix, grid, warp_by
from .ink import Cel, spl, resample, noise1

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = 'r6'


def _ridge(n): return 1 - np.abs(n)


class Face:
    """one face.  The stone is painted once (cel steps, ink in the folds, brush hatching, wet points); the eye is drawn over it at any openness."""

    def __init__(s, seed=0, size=720):
        s.n, s.seed = size, seed; r = np.random.default_rng(seed + 100); n = size
        y, x = np.mgrid[0:n, 0:n].astype(f32); s.x = (x + .5) / n * 2 - 1; s.y = (y + .5) / n * 2 - 1
        s.c = np.array([r.normal(0, .03), .02 + r.normal(0, .03)]); s.rot = r.normal(0, .09); s.a, s.b = .60 + r.normal(0, .02), .40 + r.normal(0, .02)
        p = os.path.join(ROOT, 'cache', f'realm_{VER}_{seed}_{size}.npy')
        if os.path.exists(p): s.base = np.load(p).astype(f32)
        else:
            s.base = s._stone(r); os.makedirs(os.path.dirname(p), exist_ok=True); q = p + f'.{os.getpid()}.npy'; np.save(q, s.base.astype(np.float16)); os.replace(q, p)

    def eyexy(s):
        c, q = math.cos(s.rot), math.sin(s.rot); X, Y = s.x - s.c[0], s.y - s.c[1]; return X * c + Y * q, -X * q + Y * c

    def _stone(s, r):
        from .manga import hatch_tone
        n, sd = s.n, s.seed; u = n / 720.; w1 = fbm(n, n, (210 * u, 100 * u), (1, .35), sd) * .045; w2 = fbm(n, n, (210 * u, 100 * u), (1, .35), sd + 1) * .045
        X, Y = s.eyexy(); X, Y = X + w1, Y + w2; R = np.sqrt((X / s.a) ** 2 + (Y / s.b) ** 2) + 1e-4; th = np.arctan2(Y / s.b, X / s.a)
        f1 = fbm(n, n, (130 * u, 60 * u), (1, .4), sd + 2); f2 = fbm(n, n, (30 * u, 12 * u), (1, .5), sd + 3); h = .014 * f1 + .0016 * f2
        h += .235 * np.sqrt(np.clip(1 - (R / 1.06) ** 2, 0, 1)) ** 1.1                                     # the ball under the lids
        rk = 1.10
        for k in range(6):                                                                                   # folds gathered round the eye, broken
            w_ = .046 + .009 * k; off = .040 * fbm(n, n, (150 * u, 80 * u), (1, .4), sd + 20 + k); br = smooth(-1.1, .3, fbm(n, n, (170 * u, 90 * u), (1, .4), sd + 40 + k)); h += (.062 - .006 * k) * np.exp(-(np.abs((R - rk - off) / w_) ** 1.5)) * br; rk += .150 + .022 * k
        tw = th * 4.5 + R * .7 * (1 if sd % 2 else -1) + f1 * .45; st = _ridge(np.sin(tw + .7 * np.sin(th * 2 + sd))) ** 2.2; st2 = _ridge(np.sin(th * 10 + R * 1.2 + f1 * .9)) ** 6
        far = smooth(1.15, 1.75, R); h += far * (.085 * st + .026 * st2)                                     # sinew running out to the edges
        d = 1 - np.maximum(np.abs(s.x + w1 * .3), np.abs(s.y + w2 * .3)); h += .05 * smooth(.02, .07, d) * (1 - smooth(.12, .17, d)) * (1 + .5 * f1) - .10 * np.exp(-((d - .185) / .016) ** 2) * smooth(-.6, .6, f2 + .4); h -= .30 * smooth(.05, -.02, d)
        cr = np.abs(fbm(n, n, (190 * u, 60 * u, 20 * u), (1, .35, .12), sd + 5)); crk = np.exp(-(cr / .016) ** 2) * smooth(1.25, 1.7, R) * smooth(.1, .9, f1 + .3); h -= .030 * crk   # cracks
        for _ in range(22): cx, cy, q = r.uniform(-1, 1), r.uniform(-1, 1), r.uniform(.008, .018); h -= .012 * np.exp(-(((s.x - cx) ** 2 + (s.y - cy) ** 2) / q ** 2))
        hs = cv2.GaussianBlur(h.astype(f32), (0, 0), 1.8 * u); k = n * .40
        gx = cv2.Sobel(hs, cv2.CV_32F, 1, 0, ksize=3) / 8 * k; gy = cv2.Sobel(hs, cv2.CV_32F, 0, 1, ksize=3) / 8 * k
        N = np.dstack([-gx, -gy, np.ones_like(gx)]); N /= np.linalg.norm(N, axis=-1, keepdims=True); L = np.array([-.48, -.66, .58]); L /= np.linalg.norm(L)
        dif = N @ L.astype(f32); cav = np.clip((cv2.GaussianBlur(h, (0, 0), 9 * u) - h) * 22, -1, 1); cv_ = np.clip((cv2.GaussianBlur(h, (0, 0), 3.4 * u) - cv2.GaussianBlur(h, (0, 0), .8 * u)) * 95, -1, 1); cv_ = np.maximum(cv_, crk * .9)
        lit = dif - .55 * np.clip(cav, 0, 1) + .12 * f1 * .4 + .08
        col = np.ones((n, n, 3), f32) * rgb('#8d908a'); col = mix(col, rgb('#aeb1a6'), smooth(.80, .85, lit)[..., None]); sh = smooth(.56, .52, lit)[..., None]; col = mix(col, rgb('#5d6364'), sh); col = mix(col, rgb('#383d41'), smooth(.28, .24, lit)[..., None])
        col = col * (1 + .05 * f2 + .04 * f1)[..., None]; col = mix(col, rgb('#4c3338'), (np.clip(cav, 0, 1) * .55 * smooth(.2, .7, cav))[..., None]); col = mix(col, rgb('#7a6a62'), (.22 * smooth(.2, 1.4, fbm(n, n, (70 * u, 28 * u), (1, .6), sd + 8)))[..., None])
        dark = np.clip(.86 - 1.10 * lit, 0, 1); Nb = [cv2.GaussianBlur(N[..., i], (0, 0), 3 * u) for i in (0, 1)]
        ink = hatch_tone(dark, Nb[0], Nb[1], sd, int(26 * u) + 6, 2.2 * max(.8, u), .95, .26, .06)
        ink = np.maximum(ink * .78, smooth(.40, .70, cv_) * .96)                                             # pen in the valleys
        col = col * (1 - ink[..., None]) + rgb('#14151a') * ink[..., None]
        Hh = L + np.array([0, 0, 1.]); Hh /= np.linalg.norm(Hh); sp = np.clip(N @ Hh.astype(f32), 0, 1) ** 60; wet = smooth(-.2, 1.0, fbm(n, n, (80 * u, 30 * u), (1, .6), sd + 9) + .5 * (R < 1.5)); gl = smooth(.50, .66, sp * wet) * (1 - ink)
        col = mix(col, rgb('#eef2e8'), (gl * .85)[..., None]); e = smooth(.0, .03, d)[..., None]; col = col * e + rgb('#0e1013') * (1 - e)
        return np.clip(col, 0, 1.2).astype(f32)

    def cached(s, open=0., gaze=(0., 0.), pupil=.36):
        """tex() at 1/12 steps of openness, kept in memory and on disk (uint8)"""
        k = (int(round(open * 12)), round(gaze[0], 1), round(gaze[1], 1), round(pupil, 2))
        if not hasattr(s, '_m'): s._m = {}
        if k not in s._m:
            p = os.path.join(ROOT, 'cache', f'realmtex_{VER}_{s.seed}_{s.n}_{k[0]}_{k[1]}_{k[2]}_{k[3]}.npy')
            if os.path.exists(p): a = np.load(p)
            else:
                a = (np.clip(s.tex(k[0] / 12., gaze, pupil), 0, 1.2) / 1.2 * 255 + .5).astype(np.uint8); q = p + f'.{os.getpid()}.npy'; np.save(q, a); os.replace(q, p)
            if len(s._m) > 10: s._m.pop(next(iter(s._m)))
            s._m[k] = a
        return s._m[k].astype(f32) * (1.2 / 255)

    def tex(s, open=0., gaze=(0., 0.), pupil=.36, glow=0.):
        n = s.n; u = n / 720.; X, Y = s.eyexy(); ea = s.a * .93; t = np.clip(X / ea, -1, 1); ar = 1 - t * t; img = s.base.copy()
        seam = .085 * ar ** .9 - .012 * np.sin(t * 5 + s.seed); up = seam * (1 - open) + (-.285 * np.clip(ar * (1 + .14 * t), 0, 2) ** .72) * open; lo = seam * (1 - open) + (.215 * ar ** .85) * open
        px = 2.0 / n; inside = np.abs(X) < ea
        # swollen lids: soft fold shadow above, lash-line creases
        if open > .02:
            m = smooth(-px, px, Y - up) * smooth(-px, px, lo - Y) * inside
            gx, gy = gaze[0] * .16, gaze[1] * .10 - .02; q2 = (X / (ea * 1.02)) ** 2 + (Y / .44) ** 2; ball = mix(rgb('#d3cfbb'), rgb('#6d6a62'), smooth(.25, 1.05, q2)[..., None])
            th = np.arctan2(Y - gy, X - gx); rr = np.sqrt((X - gx) ** 2 + (Y - gy) ** 2); v = np.abs(np.sin(th * 13 + 2.2 * fbm(n, n, (60 * u, 24 * u), (1, .6), s.seed + 60))); vein = np.exp(-(v / .055) ** 2) * smooth(.24, .50, rr) * smooth(-.4, .9, fbm(n, n, (44 * u, 18 * u), (1, .6), s.seed + 61)); ball = mix(ball, rgb('#8c3a3a'), (vein * .65)[..., None])
            ir = .205; k = rr / ir; sp_ = .5 + .5 * np.sin(th * 46 + 3 * np.sin(th * 7 + s.seed)); iris = mix(rgb('#b9ab5f'), rgb('#5d5430'), smooth(.45, .98, k)[..., None]); iris = mix(iris, rgb('#2b2717'), (smooth(.55, .9, sp_) * smooth(.25, .5, k) * .75)[..., None]); iris = mix(iris, rgb('#e2d791'), (smooth(.7, 1., 1 - sp_) * smooth(.55, .36, np.abs(k - .56)) * .5)[..., None])
            ball = mix(ball, iris, smooth(1.02, .98, k)[..., None]); ball = mix(ball, rgb('#120f0c'), (np.exp(-((k - .99) / .055) ** 2) * .95)[..., None]); ball = mix(ball, rgb('#0a0809'), smooth(pupil + .03, pupil - .03, k)[..., None])
            ball = ball * (1 + glow * smooth(1.1, .2, k))[..., None]; sh = smooth(.115, .035, Y - up); ball = mix(ball, ball * np.array([.30, .31, .36], f32), (sh * .92)[..., None])
            hx, hy = gx - .075, gy - .085; hl = smooth(.052, .040, np.sqrt(((X - hx) / 1.25) ** 2 + (Y - hy) ** 2)) ; hl2 = smooth(.020, .014, np.sqrt((X - gx - .07) ** 2 + (Y - gy - .085) ** 2)); ball = mix(ball, np.array([1.02, 1.04, 1.0], f32), (hl * .96 + hl2 * .8)[..., None].clip(0, 1))
            wl = np.exp(-((lo - Y - .012) / .006) ** 2) * smooth(.05, .25, ar) * smooth(-.5, .5, fbm(n, n, (40 * u, 16 * u), (1, .6), s.seed + 62)); ball = mix(ball, rgb('#f4f6ea'), (wl * .9)[..., None])
            img = img * (1 - m[..., None]) + ball * m[..., None]
            rimu = np.exp(-((Y - up + .016) / .012) ** 2) * inside * smooth(.02, .2, ar); img = mix(img, rgb('#7d5b58'), (np.exp(-((lo - Y + .016) / .013) ** 2) * inside * smooth(.02, .2, ar) * open)[..., None] * .9)
        # ink: the two lid edges
        tl = .030 * (.35 + .65 * ar ** .6) * (1 + .25 * open); il = smooth(px * 1.2, -px * 1.2, np.abs(Y - up + tl * .35) - tl * .5) * (np.abs(X) < ea * 1.06); tl2 = .013 * (.35 + .65 * ar ** .5); il2 = smooth(px * 1.2, -px * 1.2, np.abs(Y - lo) - tl2 * .5) * inside
        img = img * (1 - np.maximum(il, il2 * .95)[..., None]) + rgb('#0b0b0f') * np.maximum(il, il2 * .95)[..., None]
        # lashes of torn skin: short hooked ticks off the seam
        c = Cel(n, n, ss=2); r = np.random.default_rng(s.seed + 7); cc, q = math.cos(s.rot), math.sin(s.rot)
        P = lambda a, b: (np.array([a * cc - b * q + s.c[0], a * q + b * cc + s.c[1]]) + 1) * n / 2
        for i in range(17):
            a = (-.92 + 1.84 * i / 16 + r.normal(0, .03)) * ea; tt = a / ea; arr = max(0, 1 - tt * tt); se = .085 * arr ** .9; uu = se * (1 - open) + (-.285 * arr ** .72) * open; ll = se * (1 - open) + .215 * arr ** .85 * open
            L_ = r.uniform(.04, .10) * (1 - .3 * open); c.line(spl([P(a, ll), P(a + tt * .03, ll + L_ * .6), P(a + tt * .07 + .02, ll + L_)], n=6), 4.6 * u, '#0c0c10', taper=(.02, .9), seed=i)
            if i % 2: L_ = r.uniform(.03, .075); c.line(spl([P(a, uu - .012), P(a + tt * .02, uu - L_ * .7), P(a + tt * .05, uu - L_)], n=6), 3.8 * u, '#0c0c10', taper=(.02, .9), seed=40 + i)
        for k_, (dy, al) in enumerate(((-.075, .9), (-.125, .7), (-.185, .5))):                              # lid creases follow the opening
            pts = [P(a * ea, (.085 * (1 - a * a) ** .9) * (1 - open) + (-.285 * (1 - a * a) ** .72) * open + dy * (.35 + .65 * (1 - a * a))) for a in np.linspace(-.9, .9, 13) + r.normal(0, .01, 13)]; c.line(spl(pts, n=6), (4.2 - .8 * k_) * u, '#0f0f14', taper=(.3, .35), press=.5, seed=70 + k_, alpha=al)
        l = c.rgba(); return img * (1 - l[..., 3:4]) + l[..., :3]


def warp_face(tex, quad, shade=1.0, h=H, w=W, ss=1, light=None, fade=None):
    """texture -> premult RGBA layer in a screen quad (tl, tr, br, bl).  shade scalar or 4 corner values (bilinear in texture space)"""
    n = tex.shape[0]; t = tex
    if np.ndim(shade): a, b, c, d = [np.asarray(v, f32) for v in shade]; v, u = np.mgrid[0:n, 0:n].astype(f32) / (n - 1.); sh = (a * (1 - u) + b * u) * (1 - v) + (d * (1 - u) + c * u) * v; t = t * sh[..., None]
    else: t = t * shade
    if light is not None: t = t * np.asarray(light, f32)
    q = np.asarray(quad, np.float32) * ss; M = cv2.getPerspectiveTransform(np.float32([[0, 0], [n, 0], [n, n], [0, n]]), q)
    src = np.dstack([t, np.ones((n, n), f32)]).astype(f32); big = max(np.ptp(q[:, 0]), np.ptp(q[:, 1])); f = 1
    while n / f > big * 1.6 and f < 8: f *= 2
    if f > 1: src = cv2.resize(src, (n // f, n // f), interpolation=cv2.INTER_AREA); M = cv2.getPerspectiveTransform(np.float32([[0, 0], [n // f, 0], [n // f, n // f], [0, n // f]]), q)
    out = cv2.warpPerspective(src, M, (w * ss, h * ss), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA) if ss > 1 else out


def quat(ax, an):
    ax = np.asarray(ax, float); ax = ax / np.linalg.norm(ax); return np.r_[math.cos(an / 2), ax * math.sin(an / 2)]


def qmul(a, b):
    w0, x0, y0, z0 = a; w1, x1, y1, z1 = b; return np.array([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1, w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1, w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1, w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1])


def qmat(q):
    w, x, y, z = q / np.linalg.norm(q); return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


FACES = [((0, 0, 1), (1, 0, 0), (0, -1, 0)), ((0, 0, -1), (1, 0, 0), (0, 1, 0)), ((0, -1, 0), (1, 0, 0), (0, 0, -1)), ((0, 1, 0), (-1, 0, 0), (0, 0, -1)), ((1, 0, 0), (0, 1, 0), (0, 0, -1)), ((-1, 0, 0), (0, -1, 0), (0, 0, -1))]   # normal, u, v


class Cube:
    def __init__(s, h=.045, seeds=(1, 2, 3, 4, 5, 6), size=460): s.h = h; s.f = [Face(k, size) for k in seeds]

    def draw(s, cam, pos, R, opens, lights, amb=.2, ss=2, mirror=False, off=(0., 0.), gaze=(0, 0), h=H, w=W):
        """opens = 6 openness values; lights = [((x,y,z) dir to light, gain rgb)].  mirror = reflection in the floor z = 0.  -> premult layer, screen centre"""
        out = np.zeros((h, w, 4), f32); pos = np.asarray(pos, float)
        for i, (n, u, v) in enumerate(FACES):
            n, u, v = [R @ np.asarray(k, float) for k in (n, u, v)]; c = pos + n * s.h
            if (cam.eye - (c if not mirror else c * [1, 1, -1])) @ (n if not mirror else n * [1, 1, -1]) <= 0: continue
            P = np.array([c + s.h * (-u - v), c + s.h * (u - v), c + s.h * (u + v), c + s.h * (-u + v)])
            if mirror: P = P * [1, 1, -1]
            q, z = cam.project(P); q = q + np.asarray(off)
            l = np.full(3, amb) if np.ndim(amb) == 0 else np.asarray(amb, float).copy()
            for dv, g in lights: dv = np.asarray(dv, float); dv = dv - c if np.linalg.norm(dv) > 2. else dv; l = l + max(0., float(n @ dv) / np.linalg.norm(dv)) * np.asarray(g, float)
            lay = warp_face(s.f[i].cached(opens[i], gaze), q, 1., h, w, ss, light=l); out = out * (1 - lay[..., 3:4]) + lay
        return out
