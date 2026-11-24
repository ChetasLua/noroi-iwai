"""Quick ruined-street set for Mahoraga's keys, built with the same construction + brush so plate and creature share one hand.
(The film shots may use jj/world.py plates instead; this is the stand-in.)"""
import numpy as np, math, cv2
from jj import W, H
from jj.paint import rgb, f32, fbm, smooth, grid, mix
from jj.comp import blur
from .maho_sdf import Sculpt, BOX
from .maho_body import Rx, Ry, Rz, unit, V

SKIN, CLOTH = 0, 6


class Ruin:
    """street canyon along +Y.  walls: list of (x0, x1, y0, y1, h, lean)"""
    def __init__(self, seed=4, half=6.5, depth=70, near=-6, rubble=150, hmax=26, center=(0, 0), keep=None, eye=None):
        s = self.s = Sculpt(); r = np.random.default_rng(seed); self.lines = []; self.with_wheel = False; self.sc = 1.; self.blocks = []
        s.box((0, depth / 2 - 10, -.6), (60, depth, .6), r=.02, mat=13, part=0)
        for sd in (-1, 1):
            y = near + r.uniform(-2, 2)
            while y < depth:
                d = r.uniform(5, 10); wd = r.uniform(5, 9); hh = r.uniform(.35, 1) ** .8 * hmax * (.75 + .25 * min(1, y / 20)); x0 = sd * (half + r.uniform(0, 1.6)); lean = r.normal(0, .035); ly = r.normal(0, .03)
                M = (Ry(lean) @ Rx(ly)).T; c = V(x0 + sd * wd / 2, y + d / 2, hh / 2 - .3)
                g = 100 + len(self.blocks)
                s.box(c, (wd / 2, d / 2 - .15, hh / 2), M, r=.006, g=g, mat=12, part=0, k=.0)
                # bites out of the top and corners
                for q in range(int(r.integers(2, 5))):
                    cc = V(x0 + sd * r.uniform(-.5, wd * .6), y + r.uniform(0, d), hh * r.uniform(.72, 1.06)); s.box(cc, (r.uniform(1, 3), r.uniform(1., 3), r.uniform(1.2, 4)), (Rz(r.uniform(0, 3)) @ Rx(r.normal(0, .5)) @ Ry(r.normal(0, .5))).T, r=.02, k=.02, g=g, sub=True, mat=12, part=0)
                # floor slabs that stick out of the bites, hanging
                for q in range(int(r.integers(2, 5))):
                    z = hh * r.uniform(.35, .98); s.box(V(x0 - sd * r.uniform(-.8, .35), y + r.uniform(.2, .8) * d, z), (r.uniform(.4, 1.3), r.uniform(.8, 2.4), .13), (Ry(r.normal(0, .16) + sd * .12) @ Rx(r.normal(0, .12))).T, r=.004, g=g + 400, mat=12, part=0)
                self.blocks.append((x0, sd, y, d, hh, wd, M, c)); y += d + r.uniform(-.2, 1.8)
        # end of the street: distant towers
        for q in range(9):
            hh = r.uniform(14, 46); s.box((r.uniform(-34, 34), depth + r.uniform(0, 40), hh / 2), (r.uniform(2.5, 5), 3, hh / 2), Ry(r.normal(0, .02)).T, r=.05, g=900 + q, mat=12, part=0)
        for q in range(rubble):
            y = near + 30 * r.random() ** 1.5; x = r.normal(0, half * .55) if r.random() < .6 else r.choice([-1, 1]) * half * r.uniform(.6, 1.1)
            if keep is not None and abs(x - keep[0]) < keep[2] and abs(y - keep[1]) < keep[3]: continue
            if eye is not None and math.hypot(x - eye[0], y - eye[1]) < 1.9: continue
            sz = .06 + .75 * r.random() ** 3.2 * (1 + .6 * (abs(x) > half * .5)); s.box((x, y, sz * .25), (sz * r.uniform(.6, 1.4), sz * r.uniform(.5, 1.2), sz * r.uniform(.2, .6)), (Rz(r.uniform(0, 3)) @ Rx(r.normal(0, .5)) @ Ry(r.normal(0, .5))).T, r=.004 + .006 * sz, g=1000 + q, mat=12, part=0)
        for q in range(26):                                                     # reinforcing bars and broken poles
            x, y = r.normal(0, half * .6), near + r.uniform(2, 24)
            if keep is not None and abs(x - keep[0]) < keep[2] and abs(y - keep[1]) < keep[3]: continue
            if eye is not None and math.hypot(x - eye[0], y - eye[1]) < 4.2: continue
            a = V(x, y, 0); d = unit(V(r.normal(0, .6), r.normal(0, .4), 1)); L = r.uniform(.5, 1.7); b = a + d * L; e = b + unit(d + V(r.normal(0, .8), 0, -.2)) * L * .5; s.cone(a, b, .022, .020, k=0, g=1300, mat=9, part=0); s.cone(b, e, .020, .014, k=0, g=1300, mat=9, part=0)
        for x, y, L, tl in ((-half * .86, 9, 8.5, .20), (half * .80, 15, 8.5, -.10)):
            a = V(x, y, 0); b = a + V(math.sin(tl) * L, r.normal(0, .6), math.cos(tl) * L); s.cone(a, b, .12, .08, k=0, g=1301, mat=9, part=0); t = unit(V(1, .1, -.2 * np.sign(tl))); s.cone(b - t * .9, b + t * .9, .05, .05, k=0, g=1301, mat=9, part=0)
        self.S = s.pack(); self.J = dict(chest=V(0, 5, 2))

    def windows(self, cam, zf, pen, r, lw=.5, fills=None):
        """ink window bays on the facades, broken rows"""
        for x0, sd, y, d, hh, wd, M, c in self.blocks:
            nb = int(d / 1.5); nf = int(hh / 3.0)
            for f in range(nf):
                for b in range(nb):
                    if r.random() < .12: continue
                    u0 = -d / 2 + .5 + b * 1.5; z0 = -hh / 2 + 1.0 + f * 3.0; q = np.array([(-sd * (wd / 2 + .01), u0, z0), (-sd * (wd / 2 + .01), u0 + 1.0, z0), (-sd * (wd / 2 + .01), u0 + 1.0, z0 + 1.5), (-sd * (wd / 2 + .01), u0, z0 + 1.5)]) @ M + c
                    p, z = cam.project(q)
                    if z.min() < .3: continue
                    m = p.mean(0)
                    if not (0 <= m[0] < W and 0 <= m[1] < H): continue
                    dist = np.linalg.norm(q.mean(0) - cam.eye)
                    if abs(zf[int(m[1]), int(m[0])] - dist) > .5: continue
                    if fills is not None: fills.append(p)


def smoke_field(seed, t=0., scales=(260, 110, 46, 20), w=W, h=H):
    a = fbm(h // 4, (w // 4) * 2, tuple(s / 4 for s in scales), None, seed); dx = int(t) % (w // 4); a = a[:, dx:dx + w // 4]
    return cv2.resize(np.clip(.5 + .27 * a, 0, 1.2), (w, h), interpolation=cv2.INTER_CUBIC)


def add_windows(b, R, cam, seed=0, amt=.62):
    """window bays painted into a drawn Ruin layer (dark panes, a lit sill here and there)"""
    f = []; R.windows(cam, b['z'], None, np.random.default_rng(seed), fills=f); m = np.zeros((H, W), np.uint8); e = np.zeros((H, W), np.uint8); r = np.random.default_rng(seed + 1)
    for p in f:
        q = np.clip(p, -6000, 6000); j = r.normal(0, .5, q.shape); qq = np.round((q + j) * 16).astype(np.int32); cv2.fillPoly(m, [qq], int(255 * r.uniform(.55, 1)), cv2.LINE_AA, 4)
        if r.random() < .3: cv2.line(e, tuple(qq[0]), tuple(qq[1]), 255, 1, cv2.LINE_AA, 4)
        if r.random() < .5: c_ = qq.mean(0).astype(np.int32); a_ = ((qq[0] + qq[3]) / 2).astype(np.int32); b_ = ((qq[1] + qq[2]) / 2).astype(np.int32); cv2.line(m, tuple(a_), tuple(b_), 90, 1, cv2.LINE_AA, 4)
    m = cv2.GaussianBlur(m, (0, 0), .7).astype(f32) / 255 * amt; L = b['layer'].copy(); L[..., :3] = L[..., :3] * (1 - m[..., None]) ; L[..., :3] += (e.astype(f32) / 255)[..., None] * L[..., :3].max(-1, keepdims=True) * .5; b['layer'] = L
    return b
