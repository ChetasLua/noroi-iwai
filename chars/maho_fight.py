"""Sukuna's blood world for f3 / f4: blood-red dry-brush sky with black ink clouds, towers that can be cut, bone-white Mahoraga, black silhouettes."""
import numpy as np, cv2, math
from jj import W, H, comp
from jj.paint import rgb, f32, fbm, smooth, grid, ramp, Canvas, arcs
from jj.comp import over, blur, screen
from jj.ink import Cel, spl
from jj.cam import Camera3D
from . import mahoraga as M
from .maho_body import V, unit, Rx, Ry, Rz
from .maho_ink import Pen, wash
from .maho_sdf import Sculpt

BK, BN, BL, BD = '#0c0a0d', '#efe6d3', '#b3121f', '#4d0810'
DUSTB = dict(lit='#e9cdb4', mid='#a3343a', dark='#350a11', ink='#0b0608'); DUSTK = dict(lit='#d0786a', mid='#a04a44', dark='#702a2a', ink='#4a1618'); ROCKB = dict(lit='#e6cfb8', mid='#8f2a30', dark='#23080d', ink='#080406')
BM = dict(M.BONE, alb={6: .075, 7: .17, 9: .03, 10: .20, 8: .95})              # cloth, wrap, cord, wheel, blade


def key_lights(m, key=1.05, wheel=1.4, d=(.80, .42, -.50), blood=.6, kick=(-.6, -.55, .45), shadow=True):
    L = [dict(kind='dir', dir=d, I=key, shadow=shadow, rim=.5, bounce=.015, wrap=-.02)]
    if m is not None and m.with_wheel and wheel > 0: L.append(dict(kind='point', pos=m.wheel['c'], I=wheel, d0=.95 * m.sc, shadow=True, rim=2.4, bounce=0, wrap=.0, col='#dcefff', mix='over'))
    L.append(dict(kind='dir', dir=kick, I=blood, shadow=False, rim=2.4, bounce=0, wrap=-.12, col='#c4121f', mix='shadow')); return L


def maho(m, cam, seed=3, scale=.5, lw=.8, lights=None, **k):
    """bone-white, black planes, cool rim from the wheel, red kick"""
    a = dict(fill=.016, teeth=.55, blade=.40, hatch=1.25, pal=BM, hard=.9, dry=1.2); a.update(k); return M.draw(m, cam, lights if lights is not None else key_lights(m), seed=seed, scale=scale, lw=lw, **a)


def well(m, cam, seed=0, grow=1.0):
    """black ink scribbled behind the burning wheel so it reads white on black.  -> RGBA"""
    r = np.random.default_rng(seed); wh = m.wheel; t = np.linspace(0, 6.283, 60); pen = Pen(); q = M._wp(wh, cam, np.c_[np.cos(t), np.sin(t)] * 1.12 * grow); cel = Cel(); cel.fill(q, BK)
    for k in range(34):
        R = r.uniform(.75, 1.32) * grow; a0 = r.uniform(0, 6.283); a = np.linspace(a0, a0 + r.uniform(.8, 2.6), 24); p = M._wp(wh, cam, np.c_[np.cos(a) * R, np.sin(a) * R * r.uniform(.94, 1.06)]); px = np.ptp(q[:, 0]) / 420; pen.stroke(p, r.uniform(4, 22) * px, r.integers(1 << 30), taper=(.2, .4), press=.5, tremor=1.5)
    K = pen.raster(H, W); L = cel.layer(); a = np.maximum(L[..., 3], K); return np.dstack([rgb(BK) * a[..., None], a]).astype(f32)


def white_hot(im, li, amt=1.0):
    """light that burns WHITE on red: pushed toward cool white before it adds"""
    k = np.clip(li * .9 * amt, 0, 1)[..., None]; w = np.array([.96, .985, 1.02], f32); im = im * (1 - k) + w * np.maximum(k, 0) + 0; return im + np.clip(li * amt - 1.0, 0, 1.2)[..., None] * .35


def hot(im, lay, amt=1.0):
    """white_hot for a cached mono Lay, inside its own rectangle, in place"""
    h, w = lay.a.shape[:2]
    if h: im[lay.y:lay.y + h, lay.x:lay.x + w] = white_hot(im[lay.y:lay.y + h, lay.x:lay.x + w], lay.roi(), amt)
    return im


def lit(im, lay, k):
    """additive light from a cached mono Lay"""
    h, w = lay.a.shape[:2]
    if h: im[lay.y:lay.y + h, lay.x:lay.x + w] += lay.roi()[..., None] * np.asarray(k, f32)
    return im


# ------------------------------------------------------------------ the sky
_SKY = {}
def sky(seed=1, hz=700, cx=900, w=W, h=H, dark=1.0, swirl=0.):
    k = (seed, hz, cx, w, h, dark, swirl)
    if k in _SKY: return _SKY[k]
    r = np.random.default_rng(seed); x, y = grid(h, w); t = np.clip(y / hz, 0, 1.2); dx = np.abs(x - cx) / w
    base = ramp(np.clip(t - .25 * dx, 0, 1.2), [(0, '#1a0307'), (.3, '#50080f'), (.6, '#941019'), (.85, '#cf2a2a'), (1.0, '#ee7a58'), (1.2, '#b01c22')])
    cv = Canvas(h, w, img=base)
    for wd0, n, jit in ((70, 190, .10), (34, 420, .12), (15, 420, .16)):
        px, py = r.uniform(-100, w + 100, n), hz * 1.05 * r.random(n) ** .8; ix = np.clip(px.astype(int), 0, w - 1); iy = np.clip(py.astype(int), 0, h - 1); col = base[iy, ix] * (1 + r.normal(0, jit, (n, 1))).astype(f32) + r.normal(0, .015, (n, 1)).astype(f32) * np.array([1, .15, .15], f32)
        a = np.arctan2(py - hz * 1.15, px - cx) * .0 + (.10 * np.sign(px - cx) * (1 - py / hz)) + r.normal(0, .16, n) - .12; P = arcs(px, py, a, wd0 * r.uniform(4, 9, n), r.normal(.0, .5, n), K=7)
        cv.strokes(P, wd0 * r.uniform(.5, 1.1, n), np.clip(col, 0, 1), op=r.uniform(.35, .85, n), dry=r.uniform(.2, .55, n), seed=seed + wd0, shape='flat', bristle=.12)
    # black ink clouds: loaded sweeps, ragged undersides, a few hot lips
    CL = []
    for c in range(15):
        yy = hz * (.02 + .62 * (c / 15) ** 1.1 + r.normal(0, .03)); xx = r.uniform(-80, w + 80); s = 1 - .55 * yy / hz; n = int(r.integers(9, 18))
        ox = xx + r.normal(0, 210 * s, n); oy = yy + r.normal(0, 26 * s, n); wd = r.uniform(16, 62, n) * s * dark; P = arcs(ox, oy, r.normal(-.05, .10, n), wd * r.uniform(4, 8, n), r.normal(.25, .35, n), K=8)
        cv.strokes(P + np.array([0, 9 * s], f32), wd * .95, np.repeat(rgb('#d8403a')[None], n, 0), op=r.uniform(.25, .6, n), dry=r.uniform(.45, .75, n), seed=seed * 3 + c, shape='flat', bristle=.1)
        cv.strokes(P, wd, np.repeat(rgb('#0d0609')[None], n, 0), op=r.uniform(.8, 1, n), dry=r.uniform(.18, .48, n), seed=seed * 5 + c, shape='flat', bristle=.05)
    _SKY[k] = np.clip(cv.img, 0, 1.2).astype(f32); return _SKY[k]


def ground_ink(im, cam, z0=0., seed=2, far='#8e1219', near='#16060a', hz_fade=60.):
    """street: blood-wet black, brushed in perspective"""
    pr = cam.rays(); t = cam.plane_t(np.array([0, 0, z0]), np.array([0, 0, 1.]), pr); ok = (t > .05) & (t < 400); P = cam.eye.astype(f32) + pr * t[..., None]; d = t
    k = np.clip(1 - np.exp(-d / hz_fade * 1.8), 0, 1)[..., None]; c = rgb(near) * (1 - k) + rgb(far) * k
    n = fbm(H, W, (120, 40, 14), (1, .6, .4), seed); u = cv2.remap(n, np.mod(P[..., 0] * 46 + 800, W - 2).astype(f32), np.mod(P[..., 1] * 12 + 300, H - 2).astype(f32), cv2.INTER_LINEAR); c = c * (1 + .34 * np.clip(u, -1, 1.5))[..., None]
    wet = smooth(.35, 1.1, u)[..., None] * np.exp(-d / 26)[..., None]; c = c + wet * rgb('#a3161e') * .35
    m = ok.astype(f32)[..., None]; return im * (1 - m) + c * m


# ------------------------------------------------------------------ towers
class Tower:
    """a city block as a construction sculpt; x0..x1 (street axis +Y), front face at y0, height h.  Balconies, stairs, tanks, signs are sculpt; windows, weathering, rim are brush."""
    def __init__(self, x0, x1, y0, y1, h, seed=0, floor=3.2, detail=1.0):
        s = self.s = Sculpt(); r = np.random.default_rng(seed); self.J = dict(chest=V((x0 + x1) / 2, (y0 + y1) / 2, h / 2)); self.sc = 1.; self.with_wheel = False; self.box = (x0, x1, y0, y1, h); self.seed = seed; self.floor = floor; self.signs = []; self.style = int(r.integers(0, 3))
        W2, D2 = (x1 - x0) / 2, (y1 - y0) / 2; c = V((x0 + x1) / 2, (y0 + y1) / 2, h / 2); bx = lambda *a, **k: s.box(*a, **dict(dict(r=.006, mat=12, part=0), **k)); bx(c, (W2, D2, h / 2), g=0)
        sd = 1 if abs(x1) < abs(x0) else -1; xs = x1 if sd > 0 else x0; nf = int(h / floor); st = (1, 2, 3)[self.style]
        for k in range(1, nf + 1, st): bx(V(c[0], c[1], k * floor - .15), (W2 + .14, D2 + .14, .11), g=1)
        bx(V(c[0], c[1], h + .35), (W2 + .10, D2 + .10, .45), g=2); bx(V(c[0], c[1], h * .02), (W2 + .25, D2 + .25, h * .02 + .4), g=2)
        for q in range(int(r.integers(2, 4))): bx(V(r.uniform(x0 + 1, x1 - 1), r.uniform(y0 + 1, y1 - 1), h + 1.0), (r.uniform(.6, 1.6), r.uniform(.5, 1.4), r.uniform(.5, 1.2)), g=3)
        for q in range(int(r.integers(1, 3))):                                  # water tanks on legs
            px, py = r.uniform(x0 + 1.4, x1 - 1.4), r.uniform(y0 + .9, y0 + min(4, 2 * D2 - 1)); R = r.uniform(.7, 1.15); z = h + 1.9; s.cone(V(px, py, z), V(px, py, z + 1.7), R, R, k=0, g=40, mat=12, part=0); s.cone(V(px, py, z + 1.8), V(px, py, z + 2.5), R * 1.05, .1, k=0, g=41, mat=12, part=0)
            for ax, ay in ((-1, -1), (1, -1), (1, 1), (-1, 1)): s.cone(V(px + ax * R * .7, py + ay * R * .7, h), V(px + ax * R * .6, py + ay * R * .6, z), .07, .07, k=0, g=42, mat=12, part=0)
        for q in range(int(r.integers(1, 4))): px, py = r.uniform(x0 + .5, x1 - .5), r.uniform(y0 + .5, y1 - .5); s.cone(V(px, py, h), V(px + r.normal(0, .15), py, h + r.uniform(2, 5)), .06, .02, k=0, g=43, mat=12, part=0)
        for q in range(int(r.integers(2, 4))): xx = r.uniform(x0 + .3, x1 - .3); bx(V(xx, y0 - .12, h / 2), (.16, .14, h / 2), g=5)
        for q in range(int(r.integers(2, 5))):                                  # vertical signs, blank
            zc = r.uniform(.15, .72) * h; hh = r.uniform(1.5, 4.2)
            if q % 2 == 0: ce, ex = V(xs + sd * .75, y0 + r.uniform(.3, 2.2), zc), (.55, .12, hh); qd = np.array([(ce[0] - .5, ce[1] - .13, zc - hh * .95), (ce[0] + .5, ce[1] - .13, zc - hh * .95), (ce[0] + .5, ce[1] - .13, zc + hh * .95), (ce[0] - .5, ce[1] - .13, zc + hh * .95)])
            else: xq = r.uniform(x0 + .8, x1 - .8); ce, ex = V(xq, y0 - .45, zc), (.45, .30, hh); qd = np.array([(xq - .4, y0 - .76, zc - hh * .95), (xq + .4, y0 - .76, zc - hh * .95), (xq + .4, y0 - .76, zc + hh * .95), (xq - .4, y0 - .76, zc + hh * .95)])
            bx(ce, ex, g=44 + q); self.signs.append((qd, int(r.integers(0, 3))))
        if detail > 0:
            for k in range(1, nf):                                              # balconies: front bays and the street side
                z = k * floor + .15
                for i in range(int((2 * W2) / 2.6)):
                    if r.random() < .30 * detail: xx = x0 + 1.3 + i * 2.6; bx(V(xx, y0 - .45, z), (1.0, .45, .07), g=60); bx(V(xx, y0 - .88, z + .5), (1.0, .03, .42), g=60)
                for i in range(int((2 * D2) / 2.8)):
                    if r.random() < .34 * detail: yy = y0 + 1.4 + i * 2.8; bx(V(xs + sd * .45, yy, z), (.45, 1.05, .07), g=61); bx(V(xs + sd * .88, yy, z + .5), (.03, 1.05, .42), g=61)
                if r.random() < .5: bx(V(r.uniform(x0 + .6, x1 - .6), y0 - .22, z + r.uniform(.4, 2)), (.42, .20, .3), g=62)
            if nf > 3:                                                          # iron stair zig-zag on the front
                xq = r.uniform(x0 + 1.6, x1 - 1.6)
                for k in range(0, min(nf - 1, 9)):
                    z = k * floor + .4; d_ = 1 if k % 2 == 0 else -1; s.cone(V(xq - d_ * 1.2, y0 - .7, z), V(xq + d_ * 1.2, y0 - .7, z + floor), .09, .09, k=0, g=63, mat=12, part=0); s.cone(V(xq - d_ * 1.2, y0 - .7, z + .8), V(xq + d_ * 1.2, y0 - .7, z + floor + .8), .035, .035, k=0, g=63, mat=12, part=0); bx(V(xq + d_ * 1.5, y0 - .5, z + floor), (.5, .5, .06), g=63)
            for q in range(3): xx = r.uniform(x0 + .2, x1 - .2); s.cone(V(xx, y0 - .1, 0), V(xx, y0 - .1, h), .07, .07, k=0, g=64, mat=12, part=0)
        self.S = s.pack()

    def bays(self):
        """(quad bl br tr tl, outward normal) of every window"""
        x0, x1, y0, y1, h = self.box; r = np.random.default_rng(self.seed + 9); f = self.floor; out = []; bw, ww, hh = ((1.9, .74, .58), (2.6, .86, .62), (1.5, .66, .70))[self.style]
        for k in range(int(h / f)):
            z0 = k * f + .70; z1 = z0 + f * hh
            for i in range(int((x1 - x0 - .6) / bw)): a = x0 + .4 + i * bw; out.append((np.array([(a, y0, z0), (a + bw * ww, y0, z0), (a + bw * ww, y0, z1), (a, y0, z1)]), V(0, -1, 0)))
            for xs, n in ((x0, V(-1, 0, 0)), (x1, V(1, 0, 0))):
                for i in range(int((y1 - y0 - .6) / bw)): a = y0 + .4 + i * bw; q = np.array([(xs, a, z0), (xs, a + bw * ww, z0), (xs, a + bw * ww, z1), (xs, a, z1)]); out.append((q if n[0] > 0 else q[[1, 0, 3, 2]], n))
        return out

    def draw(self, cam, seed=5, scale=.5, lw=.6, light=(.80, .42, -.50), I=.9, lit=.06, rim=1.0, weather=1.0):
        d = M.draw(self, cam, [dict(kind='dir', dir=light, I=I, shadow=False, rim=.4, bounce=.02, wrap=.0)], seed=seed + self.seed, scale=scale, lw=lw, hatch=.4, fill=.17, fill_dir=(0, -.6, .5), teeth=0, extra_curves=False, dry=.8, straight=True, term=.15, pal=M.CITY, hard=.9)
        L, z = d['layer'], d['z']; ys, xs_ = np.nonzero(L[::4, ::4, 3] > .05)
        if not len(ys): return dict(layer=L, z=z)
        bx0, bx1, by0, by1 = max(0, xs_.min() * 4 - 8), min(W, xs_.max() * 4 + 12), max(0, ys.min() * 4 - 8), min(H, ys.max() * 4 + 12); r = np.random.default_rng(self.seed + 4); cel = Cel(bx1 - bx0, by1 - by0, bx0, by0, ss=2); Ld = -unit(light); wet = []
        for q, n in self.bays():
            ce = q.mean(0)
            if n @ (cam.eye - ce) <= 0: continue
            p, zc = cam.project(q)
            if zc.min() < .4 or p[:, 0].max() < -20 or p[:, 0].min() > W + 20 or p[:, 1].max() < -20 or p[:, 1].min() > H + 20: continue
            cc = p.mean(0); xi, yi = int(np.clip(cc[0], 0, W - 1)), int(np.clip(cc[1], 0, H - 1))
            if z[yi, xi] < np.linalg.norm(ce - cam.eye) - .35: continue
            U = lambda u, v: (p[0] * (1 - u) + p[1] * u) * (1 - v) + (p[3] * (1 - u) + p[2] * u) * v; quad = lambda a, b, c_, e: np.array([U(a, c_), U(b, c_), U(b, e), U(a, e)]); px = np.hypot(*(p[1] - p[0])); lf = n @ Ld > .05; t = r.random()
            cel.fill(quad(-.05, 1.05, -.05, 1.04), '#0a0406'); cel.fill(quad(-.10, 1.10, -.15, -.05), '#d04a45' if lf else '#5d1217')
            if t < lit:                                                        # somebody left the light on
                cel.fill(quad(.05, .95, .05, .97), '#d8957a'); cel.fill(quad(.05, .95, r.uniform(.55, .8), .97), '#a8604e') if r.random() < .6 else None; cel.fill(quad(.05, r.uniform(.18, .4), .05, .97), '#8e4a3e') if r.random() < .5 else None
            elif t < lit + .20:                                                # broken
                cel.fill(quad(.05, .95, .05, .97), '#050203')
                for j in range(int(r.integers(2, 5))): a = r.uniform(0, 1); s_ = r.integers(0, 4); e0, e1 = [((a, .05), (a + .25, .05)), ((.95, a), (.95, a + .25)), ((a, .97), (a + .25, .97)), ((.05, a), (.05, a + .25))][s_]; tip = (r.uniform(.25, .75), r.uniform(.25, .75)); cel.fill(np.array([U(*np.clip(e0, .05, .96)), U(*np.clip(e1, .05, .96)), U(*tip)]), '#6a1b20' if lf else '#35090e')
            else:
                cel.fill(quad(.05, .95, .05, .97), '#2a0a0f' if lf else '#14060a')
                if r.random() < .55: a = r.uniform(.1, .6); cel.fill(np.array([U(a, .05), U(a + .18, .05), U(a + .42, .97), U(a + .24, .97)]), '#8f1f26' if lf else '#3a0b11')
                if r.random() < .25: cel.fill(quad(.05, .95, r.uniform(.4, .75), .97), '#4a1216' if lf else '#1f080c')
            if px > 13 and t >= lit * .5: cel.line(np.array([U(.5, .05), U(.5, .5), U(.5, .97)]), max(1., px * .045), '#0a0406', taper=(0, 0), tremor=.2); cel.line(np.array([U(.05, .62), U(.5, .62), U(.95, .62)]), max(.9, px * .035), '#0a0406', taper=(0, 0), tremor=.2)
            if r.random() < .45: wet.append((U(r.uniform(.1, .9), -.12), px, lf))
        for qd, k in self.signs:
            p, zc = cam.project(qd)
            if zc.min() < .5: continue
            xi, yi = np.clip(p.mean(0), 0, (W - 1, H - 1)).astype(int)
            if z[yi, xi] < np.linalg.norm(qd.mean(0) - cam.eye) - .5: continue
            U = lambda u, v: (p[0] * (1 - u) + p[1] * u) * (1 - v) + (p[3] * (1 - u) + p[2] * u) * v; cel.fill(p, ('#3a0c11', '#2a090d', '#1a070b')[k]); cel.fill(np.array([U(0, 0), U(.22, 0), U(.22, 1), U(0, 1)]), '#0a0406', .35)
            for j in range(3): a = r.uniform(0, 1); cel.line(np.array([U(r.random(), a), U(r.random(), a - .1), U(r.random(), a - .22)]), 1.6, '#0a0406')
        w_ = cel.rgba(); R = L[by0:by1, bx0:bx1]; a = R[..., 3:4]; k = w_[..., 3:4] * a; R = R * (1 - k) + np.dstack([w_[..., :3] * a, k * 0 + w_[..., 3:4] * a]); R[..., 3:4] = a
        if weather > 0:                                                        # dry-brush: rain streaks under the sills, scrapes across the planes
            cv = Canvas(by1 - by0, bx1 - bx0, color='#0a0406', layer=True); h_, w__ = by1 - by0, bx1 - bx0; off = np.array([bx0, by0]); PP, WD = [], []
            for q, px, lf in wet: l = px * r.uniform(1., 4.5); t = np.linspace(0, 1, 7)[:, None]; PP.append((q - off) + np.array([r.normal(0, 2), l]) * t); WD.append(px * r.uniform(.06, .22))
            if PP: cv.strokes(np.array(PP), np.array(WD), rgb('#0a0406'), op=.75, dry=.5, seed=seed, shape='taper', bristle=.0)
            n = int(22 * weather); P2 = arcs(r.uniform(0, w__, n), r.uniform(0, h_, n), r.normal(1.35, .12, n), r.uniform(60, 260, n), r.normal(0, .25, n), K=7); cv.strokes(P2, r.uniform(6, 26, n), np.repeat(rgb('#0a0406')[None], n, 0), op=r.uniform(.3, .7, n), dry=r.uniform(.55, .8, n), seed=seed + 3, shape='flat', bristle=0)
            m = cv.a[..., None] * a; R[..., :3] = R[..., :3] * (1 - m) + rgb('#0a0406') * m
            c2 = Canvas(h_, w__, color='#d23c3a', layer=True); n = int(9 * weather); P3 = arcs(r.uniform(0, w__, n), r.uniform(0, h_, n), r.normal(1.45, .10, n), r.uniform(40, 200, n), r.normal(0, .2, n), K=7); c2.strokes(P3, r.uniform(2, 8, n), np.repeat(rgb('#c8343a')[None], n, 0), op=r.uniform(.15, .4, n), dry=r.uniform(.6, .85, n), seed=seed + 5, shape='flat', bristle=0); m = c2.a[..., None] * a; R[..., :3] = R[..., :3] * (1 - m) + rgb('#b8292f') * m
        if rim > 0:                                                            # red rim on the side that faces the burning horizon
            from jj.kit import shifted
            sg = 1 if (bx0 + bx1) / 2 < W * .52 else -1; al = np.pad(R[..., 3], 6, mode='edge'); m = np.clip(al - shifted(al, -sg * 4, 1.5), 0, 1)[6:-6, 6:-6]; nz = fbm(by1 - by0, bx1 - bx0, (90, 26), (1, .6), self.seed + 2); m = m * smooth(-.9, .1, nz) * rim; R[..., :3] = R[..., :3] * (1 - m[..., None]) + rgb('#ff5a42') * (m * R[..., 3])[..., None]
        L[by0:by1, bx0:bx1] = R; return dict(layer=L, z=z)


def haze(L, z, col='#8d1119', start=14., dens=.018, keep=.0):
    """red air over a layer by distance"""
    k = (1 - np.exp(-np.maximum(np.minimum(z, 400) - start, 0) * dens)) * (1 - keep); o = L.copy(); o[..., :3] = L[..., :3] * (1 - k[..., None]) + rgb(col) * (k * L[..., 3])[..., None]; return o


def calm(im, amt=.45, haze='#a33430', sig=2.0, mask=None):
    """background behind the figures: contrast down, pale red air, soft focus"""
    o = (blur(im, sig) if sig > 0 else im) * (1 - amt) + rgb(haze) * amt
    return o.astype(f32) if mask is None else (im * (1 - mask[..., None]) + o * mask[..., None]).astype(f32)


def halo(im, lay, col='#c9584a', r=26, amt=.6):
    """pale air hugging a figure so its silhouette is clean: draw before the figure.  in place"""
    h, w = lay.a.shape[:2]
    if not h: return im
    p = int(r * 2.4); y0, y1, x0, x1 = max(0, lay.y - p), min(H, lay.y + h + p), max(0, lay.x - p), min(W, lay.x + w + p); a = np.zeros((y1 - y0, x1 - x0), f32); a[lay.y - y0:lay.y - y0 + h, lay.x - x0:lay.x - x0 + w] = lay.roi()[..., 3]; k = (np.clip(blur(a, r) * 1.7, 0, 1) * amt)[..., None]; b = im[y0:y1, x0:x1]; b *= (1 - k); b += rgb(col) * k; return im


# ------------------------------------------------------------------ the cut
class Bit:
    """a moving piece: RGBA float array with its corner"""
    def __init__(s, a, x, y): s.a, s.x, s.y = a, int(x), int(y)
    def over(s, im):
        h, w = s.a.shape[:2]; x0, y0, x1, y1 = max(0, s.x), max(0, s.y), min(W, s.x + w), min(H, s.y + h)
        if x1 <= x0 or y1 <= y0: return im
        q = s.a[y0 - s.y:y1 - s.y, x0 - s.x:x1 - s.x]; b = im[y0:y1, x0:x1]; b *= (1 - q[..., 3:4]); b += q[..., :3]; return im


def section(T, cam, p0, p1, seed=0):
    """what the cut opened: the underside of the upper piece of tower T, as walls, columns, slabs and rebar in perspective.  -> full RGBA or None"""
    x0, x1, y0, y1, h = T.box; p0, p1 = np.asarray(p0, float), np.asarray(p1, float); d = unit(p1 - p0); n = np.array([d[1], -d[0]]); g = lambda X, Y, z: float((cam.project(np.array([[X, Y, z]]))[0][0] - p0) @ n); zs = []
    for X in (x0, x1):
        lo, hi = 0.02, h + 1.; a, b = g(X, y0, lo), g(X, y0, hi)
        if a * b > 0: zs.append(None); continue
        for _ in range(30):
            m = (lo + hi) / 2; c = g(X, y0, m)
            if a * c <= 0: hi = m
            else: lo, a = m, c
        zs.append((lo + hi) / 2)
    if zs[0] is None and zs[1] is None: return None
    zL = zs[0] if zs[0] is not None else (0. if g(x0, y0, 0) > 0 else h); zR = zs[1] if zs[1] is not None else (0. if g(x1, y0, 0) > 0 else h)
    Q = lambda u, v: cam.project(np.array([[x0 + (x1 - x0) * u, y0 + (y1 - y0) * v, zL + (zR - zL) * u]]))[0][0]; quad = lambda a, b, c, e: np.array([Q(a, c), Q(b, c), Q(b, e), Q(a, e)]); r = np.random.default_rng(seed + T.seed); cel = Cel(ss=1)
    cel.fill(quad(0, 1, 0, 1), '#8a4038'); cel.fill(quad(.035, .965, .05, .95), '#1a070b')
    for i in range(1, 4): cel.fill(quad(i / 4 - .012, i / 4 + .012, .05, .95), '#5c1c20')
    for j in range(1, 3): cel.fill(quad(.035, .965, j / 3 - .02, j / 3 + .02), '#5c1c20')
    for i in range(5):
        for j in range(4): cel.fill(quad(max(0, i / 4 - .03), min(1, i / 4 + .03), max(0, j / 3 - .045), min(1, j / 3 + .045)), '#a85f50')
    for k in range(70): u = r.random(); cel.fill(np.array([Q(u, r.random()), Q(u + r.uniform(.01, .05), r.random() * .2 + .4), Q(u + .02, r.random())]), '#0a0406', .25)
    for k in range(46):                                                        # rebar
        u, v = r.random(), (0. if r.random() < .6 else r.random()); q = Q(u, v); l = r.uniform(14, 52); cel.line(np.array([q, q + np.array([r.normal(0, 6), l * .6]), q + np.array([r.normal(0, 18), l])]), r.uniform(1.4, 2.8), '#0a0406' if r.random() < .7 else '#e6b49c', taper=(0, .3), tremor=.8)
    return cel.layer()


def part(lay, p0, p1, slide=0., gap=0., rot=0., fall=0., face=16., side=1, sec=None, seed=0):
    """cut a Lay (maho_shot) or full RGBA along p0->p1 and move the piece on the `side` (1 = above a left-to-right line).
    slide px along the line, gap px away from it, rot radians about its low end, fall px down.  sec = full RGBA / Lay of section().
    -> (under Bit or None, still Bit, moving Bit or None): composite in that order"""
    if hasattr(lay, 'roi'): L, ox, oy = lay.roi(), lay.x, lay.y
    else: L, ox, oy = lay, 0, 0
    h, w = L.shape[:2]
    if not h: return None, Bit(L, ox, oy), None
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float); d = unit(p1 - p0); n = np.array([d[1], -d[0]]) * side; y, x = np.mgrid[0:h, 0:w].astype(f32); x += ox; y += oy; s = (x - p0[0]) * n[0] + (y - p0[1]) * n[1]; k = np.clip(s + .5, 0, 1)
    if k.max() <= 0 or not (slide or gap or rot or fall): return None, Bit(L, ox, oy), None
    A = L * k[..., None]; B = L * (1 - k)[..., None]; a = L[..., 3]
    bm = ((s > 0) & (s < face) & (a > .5)).astype(f32); hh = (np.mod((x * d[0] + y * d[1]) * .22 + np.sin(s * .7) * .6, 1) < .42)
    A[..., :3] = A[..., :3] * (1 - bm[..., None]) + (rgb('#0a0406') * (1 - .55 * hh)[..., None] + rgb('#5c0d13') * (.55 * hh)[..., None]) * (bm * a)[..., None]
    lip = ((s < 0) & (s > -4.2) & (a > .5)).astype(f32); B[..., :3] = B[..., :3] * (1 - lip[..., None]) + rgb('#f4dcc4') * (lip * a)[..., None]
    # floors seen in the cut of the still part + bent bars
    r = np.random.default_rng(seed + int(ox) * 3 + int(oy)); e = (np.abs(s) < 1.2) & (a > .5); ys, xs = np.nonzero(e); pen, wp = Pen(), Pen()
    for j in (r.choice(len(ys), min(len(ys), 46), replace=False) if len(ys) else ()):
        q = np.array([xs[j], ys[j]], float); l = r.uniform(8, 30); (pen if r.random() < .75 else wp).stroke(np.array([q, q + n * l * .6 + d * r.normal(0, 4), q + n * l + d * r.normal(0, 12)]), r.uniform(1.3, 2.6), int(j), taper=(0, .4))
    K, Wm = pen.raster(h, w), wp.raster(h, w); B[..., :3] = B[..., :3] * (1 - K[..., None]) + rgb('#0a0406') * K[..., None]; B[..., 3] = np.maximum(B[..., 3], K); B[..., :3] = B[..., :3] * (1 - Wm[..., None]) + rgb('#f0c4aa') * Wm[..., None]; B[..., 3] = np.maximum(B[..., 3], Wm)
    ys, xs = np.nonzero(bm[::4, ::4]); piv = np.array([xs[np.argmax(ys)] * 4. + ox, ys.max() * 4. + oy]) if len(xs) else (p0 + p1) / 2; c, q = math.cos(rot), math.sin(rot); t = d * slide + n * gap + np.array([0, fall]); Mx = np.array([[c, -q, piv[0] - c * piv[0] + q * piv[1] + t[0]], [q, c, piv[1] - q * piv[0] - c * piv[1] + t[1]]])
    def move(A, ox, oy):
        ys, xs = np.nonzero(A[::4, ::4, 3] > .02)
        if not len(ys): return None
        y0, y1, x0, x1 = max(0, ys.min() * 4 - 4), min(A.shape[0], ys.max() * 4 + 8), max(0, xs.min() * 4 - 4), min(A.shape[1], xs.max() * 4 + 8); A = A[y0:y1, x0:x1]; cor = np.array([(ox + x0, oy + y0), (ox + x1, oy + y0), (ox + x1, oy + y1), (ox + x0, oy + y1)], float) @ Mx[:, :2].T + Mx[:, 2]; lo = np.maximum(np.floor(cor.min(0)) - 2, (0, 0)); hi = np.minimum(np.ceil(cor.max(0)) + 2, (W, H)); sz = (hi - lo).astype(int)
        if sz.min() <= 0: return None
        M2 = Mx.copy(); M2[:, 2] += Mx[:, :2] @ np.array([ox + x0, oy + y0]) - lo; return Bit(cv2.warpAffine(np.ascontiguousarray(A, f32), M2.astype(np.float32), (int(sz[0]), int(sz[1])), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT), lo[0], lo[1])
    u = None
    if sec is not None: u = move(sec.roi(), sec.x, sec.y) if hasattr(sec, 'roi') else move(sec, 0, 0)
    return u, Bit(B, ox, oy), move(A, ox, oy)


def wisps(cx, cy, rx, ry, col='#e2604c', amt=.7, seed=0, ink=26, angle=-.2):
    """pale red haze with black ink smoke dragged through it: separates planes.  -> RGBA"""
    x, y = grid(); d = np.sqrt(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2); n = fbm(H, W, (260, 100, 40, 16), (1, .6, .4, .25), seed, aspect=1.8); a = np.clip(1 - d, 0, 1) ** .9 * smooth(-1.1, .9, n + .4) * amt; L = np.dstack([rgb(col) * a[..., None], a]).astype(f32); r = np.random.default_rng(seed + 1)
    if ink:
        cv = Canvas(H, W, color='#12060a', layer=True); P = arcs(cx + r.normal(0, rx * .6, ink), cy + r.normal(0, ry * .5, ink), angle + r.normal(0, .25, ink), r.uniform(180, 620, ink), r.normal(0, .9, ink), K=9); cv.strokes(P, r.uniform(10, 46, ink), np.repeat(rgb('#12060a')[None], ink, 0), op=r.uniform(.25, .65, ink), dry=r.uniform(.5, .8, ink), seed=seed, shape='taper', bristle=0); m = (cv2.GaussianBlur(cv.a, (0, 0), 1.2) * np.clip(1.25 - d, 0, 1))[..., None]; L[..., :3] = L[..., :3] * (1 - m) + rgb('#12060a') * m; L[..., 3:4] = L[..., 3:4] * (1 - m) + m
    return L


def cracks(im, cam, centre, n=9, reach=7., seed=0, w=.05):
    """cracks in the street from a world point, in perspective: black gap, hot lip"""
    r = np.random.default_rng(seed); pen, lip = Pen(), Pen(); c = np.asarray(centre, float)
    def run(p, a, L, wd, dep):
        pts = [p.copy()]; k = int(L / .35) + 2
        for i in range(k):
            a2 = a + r.normal(0, .5); p = p + np.array([math.cos(a2), math.sin(a2), 0]) * .35; pts.append(p.copy())
            if dep and r.random() < .16: run(p.copy(), a + r.choice([-1, 1]) * r.uniform(.5, 1.1), L * r.uniform(.2, .45), wd * .6, dep - 1)
        pts = np.array(pts); q, z = cam.project(pts); ok = z > .3
        if ok.sum() < 3: return
        q, z = q[ok], z[ok]; hw = np.maximum(.4, wd * cam.f / z * np.linspace(1, .15, len(q)) * .5 * (1 + .4 * r.normal(0, 1, len(q)))); pen.add(q, np.maximum(.3, hw)); lip.add(q + np.array([0, 1.]) * (hw + 1.2)[:, None], np.maximum(.4, hw * .22))
    for i in range(n): run(c + np.array([r.normal(0, .15), r.normal(0, .15), 0]), i / n * 6.283 + r.normal(0, .2), reach * r.uniform(.4, 1.), w * r.uniform(.6, 1.4), 2)
    m, l = pen.raster(H, W)[..., None], lip.raster(H, W)[..., None] * .8; im = im * (1 - l) + rgb('#ff6a4e') * l; return im * (1 - m) + rgb('#070304') * m


class Rubble:
    """broken slabs and lumps on the street (sculpt, CITY palette); spots = [(x, y, spread, n, size)]"""
    def __init__(self, spots, seed=0):
        s = self.s = Sculpt(); r = np.random.default_rng(seed); self.J = dict(chest=V(0, 5, 1)); self.sc = 1.; self.with_wheel = False; g = 0
        for x, y, sp, n, size in spots:
            for k in range(n):
                g += 1; sz = size * (.18 + .9 * r.random() ** 2.4); px, py = x + r.normal(0, sp), y + r.normal(0, sp); slab = r.random() < .35; hx = (sz * r.uniform(.8, 1.6), sz * r.uniform(.6, 1.2), sz * (.12 if slab else r.uniform(.25, .55))); s.box(V(px, py, sz * .22), hx, (Rz(r.uniform(0, 3)) @ Rx(r.normal(0, .5)) @ Ry(r.normal(0, .5))).T, r=.01, g=g, mat=12, part=0)
                if r.random() < .22: a = V(px, py, sz * .3); b = a + V(r.normal(0, .4), r.normal(0, .3), r.uniform(.5, 1.1)) * sz * 1.4; s.cone(a, b, .025, .02, k=0, g=9000, mat=9, part=0); s.cone(b, b + V(r.normal(0, .5), 0, -.1) * sz, .02, .012, k=0, g=9000, mat=9, part=0)
        self.S = s.pack()

    def draw(self, cam, seed=4, scale=.5, lw=.7, light=(.8, -.2, -.45), I=1.2):
        d = M.draw(self, cam, [dict(kind='dir', dir=light, I=I, shadow=False, rim=.5, bounce=.02, wrap=0)], seed=seed, scale=scale, lw=lw, hatch=.8, fill=.04, fill_dir=(0, -.6, .5), teeth=0, extra_curves=False, dry=1., straight=True, term=.3, pal=M.CITY, hard=.9); return d


def cut_line(im, p0, p1, k, w=2.6):
    """the cut itself: k = 0 one hair-thin white line across the frame; k = 1, 2 the silence after it (thread of black)"""
    pen = Pen(); p0, p1 = np.asarray(p0, float), np.asarray(p1, float); t = np.linspace(0, 1, 60)[:, None]; P = p0 * (1 - t) + p1 * t
    if k == 0:
        pen.add(P, np.full(60, w * .5)); m = pen.raster(H, W); g = cv2.GaussianBlur(m, (0, 0), 5) * .9 + cv2.GaussianBlur(m, (0, 0), 22) * .9; return im * (1 - m[..., None]) + m[..., None] * 1.25 + g[..., None] * np.array([.8, .55, .5], f32)
    ww = w * (.9, .7, .5, .35)[min(k - 1, 3)] * np.sin(np.pi * np.clip(t[:, 0] * 1.04 - .02, 0, 1)) ** .3; pen.add(P, np.maximum(.3, ww * .5)); m = pen.raster(H, W)[..., None]; return im * (1 - m) + rgb('#0a0406') * m


def silhouette(L, col=BK, rim_=None, dx=-3, dy=2):
    """flat ink silhouette of any layer, optional hard rim"""
    from jj.kit import shifted
    a = L[..., 3]; o = np.dstack([rgb(col) * a[..., None], a]).astype(f32)
    if rim_ is not None: m = np.clip(a - shifted(a, dx, dy), 0, 1); o[..., :3] = o[..., :3] * (1 - m[..., None]) + rgb(rim_) * (m * a)[..., None]
    return o


def ink_spatter(cel, f, cx, cy, n=60, seed=0, speed=34., ang=None, spread=1.2, grav=1.6, rmax=9., col=BK):
    """thrown ink drops that fly and fall (frames since the hit)"""
    r = np.random.default_rng(seed)
    for k in range(n):
        a = r.uniform(0, 6.283) if ang is None else ang + r.normal(0, spread); v = speed * r.uniform(.15, 1.2); x = cx + math.cos(a) * v * f * (1 - .02 * f); y = cy + math.sin(a) * v * f * (1 - .02 * f) + .5 * grav * f * f; R = rmax * r.random() ** 2.4 + 1; th = math.atan2(math.sin(a) * v + grav * f, math.cos(a) * v); st = 1 + min(3., v * .05); t = np.linspace(0, 6.283, 10, endpoint=False); c, s = math.cos(th), math.sin(th); q = np.c_[np.cos(t) * R * st, np.sin(t) * R * (.8 + .2 * np.cos(t))]; cel.fill(np.c_[x + q[:, 0] * c - q[:, 1] * s, y + q[:, 0] * s + q[:, 1] * c], col)
    return cel
