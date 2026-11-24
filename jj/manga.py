"""Manga page toolkit: ink panels in perspective, screentone, focus lines, explosions, rubble, black fills with flying white,
and six ready panel drawings (explosion, runner, fist, street, eye, fingers).  Every drawing returns a float32 GREY picture
(1 = paper, 0 = ink) at its native size; Sheet.panel() warps it into a tilted frame on the page.
"""
import numpy as np, cv2, math
from numba import njit
from . import W, H
from .paint import rgb, f32, fbm, smooth, lum
from .ink import Cel, spl, resample, ribbon, noise1, boil
from .kit import s1, svg, outline, hatch, focus_lines, speed_lines, splatter, screentone
from .sumi import BLACK, SHEET, tooth, inkify, paper, blob
from .letters import homog, persp, sfx

K, P_ = '#0d0b0e', '#ffffff'


class Art:
    """a drawing board in its own px, 2x supersampled"""

    def __init__(self, w, h, seed=0):
        self.w, self.h = w, h; self.c = Cel(w, h); self.r = np.random.default_rng(seed); self.n = seed * 100
        self.c.fill(np.array([(0, 0), (w, 0), (w, h), (0, h)], float), P_); self.tones = []

    def sd(self): self.n += 1; return self.n

    def pts(self, d, n=14): return s1(d, n) if isinstance(d, str) else np.asarray(d, float)

    def shape(self, d, w=4., fill=P_, **k):
        p = self.pts(d); self.c.fill(p, fill)
        if w: outline(self.c, p, w, K, self.sd(), **{**dict(press=.45, tremor=.35), **k})
        return p

    def line(self, d, w=3., taper=(.18, .35), **k):
        for p in (svg(d) if isinstance(d, str) else [np.asarray(d, float)]): self.c.line(p, w, k.pop('col', K) if 'col' in k else K, taper=taper, press=.4, seed=self.sd(), **k)

    def white(self, d, w=3., **k): self.line(d, w, col=P_, **k)

    def black(self, d): self.c.fill(self.pts(d), K)

    def paper(self, d): self.c.fill(self.pts(d), P_)

    def hatch(self, d, ang, gap=7., w=1.6, **k):
        hatch(self.c, self.pts(d), ang, gap, w, K, self.sd(), **k)

    def tone(self, d, dark=.25, pitch=7.0):
        self.tones.append((self.pts(d), dark, pitch))

    def feather(self, A, B, n=26, w=3.2, reach=(.3, 1.), out=False):
        """spotting: strokes from curve A toward curve B, thick at A, flicked to a point"""
        A = resample(self.pts(A), 3.); B = resample(self.pts(B), 3.); r = self.r
        for i in range(n):
            t = (i + r.uniform(-.35, .35)) / n; t = min(max(t, 0), 1); a = A[int(t * (len(A) - 1))]; b = B[int(min(1, max(0, t + r.normal(0, .02))) * (len(B) - 1))]
            q = r.uniform(*reach); m = (a + b) / 2 + r.normal(0, np.hypot(*(b - a)) * .03, 2)
            self.c.line(spl([a, a + (m - a) * q * 1.0, a + (b - a) * q], n=8), w * r.uniform(.6, 1.2), K, taper=(.02, .9), press=.25, seed=self.sd(), tmin=.06)

    def grey(self):
        g = smooth(.10, .90, lum(self.c.rgba()[..., :3]).astype(f32))
        for p, dark, pitch in self.tones:
            m = np.zeros((self.h * 2, self.w * 2), f32); cv2.fillPoly(m, [np.round(p * 2 * 16).astype(np.int32)], 1., cv2.LINE_AA, 4); m = cv2.resize(m, (self.w, self.h), interpolation=cv2.INTER_AREA)
            g = g * (1 - screentone((m > .02).astype(f32) * m, pitch, .785, dark, self.h, self.w))
        return np.clip(g, 0, 1)


def rock(a, x, y, R, rot=0., w=3., flat=.8, dark=.6, seed=None):
    """a chunk of concrete: ragged outline, a broken ridge, the under-plane black or hatched"""
    r = a.r if seed is None else np.random.default_rng(seed); m = int(r.integers(6, 11)) if R > 9 else int(r.integers(4, 7))
    th = np.linspace(0, 2 * math.pi, m, endpoint=False) + r.normal(0, .30, m) + rot; th = np.sort(th); rd = R * r.uniform(.58, 1.18, m); el = r.uniform(.62, 1.0)
    P = np.c_[np.cos(th) * rd, np.sin(th) * rd * flat * el]; t = r.normal(0, .5); c, s = math.cos(t), math.sin(t); P = P @ np.array([[c, s], [-s, c]]) + [x, y]
    a.shape(np.vstack([P, P[:1]]), w, breaks=0)
    if R > 6:
        sh = P[:, 1] * .9 + P[:, 0] * .35; i = int(np.argmax(sh)); k = max(1, m // 4); i0, i1 = (i - k) % m, (i + k) % m
        mid = np.array([x, y]) + r.normal(0, R * .12, 2) + [-.05 * R, .02 * R]; q0 = P[i0] * .55 + mid * .45 + r.normal(0, R * .07, 2); q1 = P[i1] * .55 + mid * .45 + r.normal(0, R * .07, 2)
        arc = [P[(i0 + j) % m] for j in range((i1 - i0) % m + 1)]; pl = np.array(arc + [q1, mid, q0])
        if r.random() < dark: a.black(pl)
        else: a.hatch(pl, r.uniform(0, 3), max(3.2, R * .085), max(.9, w * .42))
        if R > 12:
            o = P[(i + m // 2) % m]; a.line(spl([mid, (mid + o) / 2 + r.normal(0, R * .10, 2), o * .9 + mid * .1], n=6), w * .55, taper=(.05, .7))
            for _ in range(int(R / 14)): p = mid + r.normal(0, R * .3, 2); a.line([p, p + r.normal(0, R * .16, 2)], w * .4, taper=(.1, .6))
    return P


def crack(a, x, y, ang, L, w=5., depth=3, jag=.5):
    """a fracture: kinked, tapering, forking"""
    r = a.r; n = max(3, int(L / 26)); p = np.array([x, y], float); P = [p]
    for i in range(n): ang2 = ang + r.normal(0, jag); p = p + np.array([math.cos(ang2), math.sin(ang2)]) * L / n; P.append(p)
    P = np.array(P); t = np.linspace(1, 0, len(P)); T = np.gradient(P, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-6; N = np.c_[-T[:, 1], T[:, 0]]
    hw = w * .5 * t ** .8 * r.uniform(.5, 1.4, len(P)) + .3
    a.c.fill(np.vstack([P + N * hw[:, None], (P - N * hw[:, None])[::-1]]), K)
    if depth > 0:
        for _ in range(r.integers(1, 4)):
            k = r.integers(1, len(P) - 1) if len(P) > 2 else 1; crack(a, *P[k], ang + r.choice([-1, 1]) * r.uniform(.5, 1.2), L * r.uniform(.25, .55), w * .55, depth - 1, jag)
    return P


# ---------------------------------------------------------------- inking a relief (normals, depth, ids)
@njit(cache=True, fastmath=True)
def _lic(ns, vx, vy, n, out):
    h, w = ns.shape
    for y in range(h):
        for x in range(w):
            acc = ns[y, x]; cnt = 1.0
            for sg in (-1.0, 1.0):
                px = float(x); py = float(y); ux = vx[y, x] * sg; uy = vy[y, x] * sg
                for k in range(n):
                    ix = int(px); iy = int(py)
                    if ix < 0 or iy < 0 or ix >= w or iy >= h: break
                    dx = vx[iy, ix]; dy = vy[iy, ix]
                    if dx * ux + dy * uy < 0: dx = -dx; dy = -dy
                    ux = dx; uy = dy
                    px += dx; py += dy
                    ix = int(px); iy = int(py)
                    if ix < 0 or iy < 0 or ix >= w or iy >= h: break
                    acc += ns[iy, ix]; cnt += 1
            out[y, x] = acc / cnt


def streaks(vx, vy, L=34, width=2.6, seed=0):
    """brush-hatching field: uniform 0..1 streaks along the direction field; (streaks < dark) is a hatch of coverage `dark`"""
    h, w = vx.shape; r = np.random.default_rng(seed)
    m = np.hypot(vx, vy) + 1e-6; vx = (vx / m).astype(f32); vy = (vy / m).astype(f32)
    ns = cv2.GaussianBlur(r.standard_normal((h, w)).astype(f32), (0, 0), width * .42); ns /= ns.std()
    out = np.zeros((h, w), f32); _lic(ns.astype(f32), vx, vy, int(L), out)
    out = cv2.GaussianBlur(out, (0, 0), .6)
    ix = np.argsort(out.ravel()[::7]); ref = out.ravel()[::7][ix]
    return np.interp(out.ravel(), ref, np.linspace(0, 1, len(ref))).reshape(h, w).astype(f32)


def hatch_tone(dark, vx, vy, seed=0, L=34, width=2.6, black=.82, white=.10, soft=.07):
    """tone (0..1) -> ink alpha: solid black above `black`, clean paper below `white`, brush hatching along (vx, vy) between"""
    t = np.clip((dark - white) / (black - white), 0, 1); s = streaks(vx, vy, L, width, seed)
    return np.where(dark >= black, 1., smooth(-soft, soft, t - s * .98 - .01) * (t > .005)).astype(f32)


def edge_ink(gid, z, jump, w=3.0, seed=0, dark=None, mask=None):
    """contour ink of a relief: lines where ids change or depth jumps, thicker on the dark side, broken where light burns. -> ink alpha"""
    h, w_ = gid.shape
    def e(dy, dx):
        a = gid; b = np.roll(gid, (-dy, -dx), (0, 1)); zz = np.abs(z - np.roll(z, (-dy, -dx), (0, 1)))
        return (a != b) | (zz > jump)
    m = e(0, 1) | e(1, 0); m[-1] = False; m[:, -1] = False
    if mask is not None: m &= mask
    r = np.random.default_rng(seed); out = np.zeros((h * 2, w_ * 2), f32)
    # skeleton pixels -> polylines by contour walking of the dilated edge
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    dk = np.zeros_like(z) if dark is None else cv2.GaussianBlur(dark.astype(f32), (0, 0), 3)
    pn = np.clip(fbm(h, w_, (60, 22), (1, .5), seed) * .30, -.6, .6)
    d = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5)
    half = w * .5 * (.66 + .84 * np.clip(dk, 0, 1) + pn * .7)
    q = cv2.GaussianBlur((half - d).astype(f32), (0, 0), 1.1)
    return np.clip(q * 1.1 + .62, 0, 1).astype(f32)


def puffs(spec, h, w, seed=0, kids=11, gen=3, shrink=.46):
    """cauliflower smoke relief: spec = [(x, y, z, R)] -> z buffer (px), normal (h, w, 3: x right, y down, z to us), id"""
    r = np.random.default_rng(seed); L = [(x, y, z, R, i) for i, (x, y, z, R) in enumerate(spec)]; cur = list(L)
    for g in range(1, gen):
        nx = []
        for x, y, z, R, i in cur:
            for _ in range(kids):
                v = r.normal(0, 1, 3); v[2] = abs(v[2]) * .8 + .05; v /= np.linalg.norm(v); q = R * shrink * r.uniform(.55, 1.25)
                nx.append((x + v[0] * R * .86, y + v[1] * R * .86, z + v[2] * R * .80, q, i))
        L += nx; cur = nx
    L = np.array(L); Z = np.full((h, w), -1e4, f32); ID = np.full((h, w), -1, np.int32)
    _puff(L, Z, ID)
    zz = cv2.GaussianBlur(np.where(Z < -9e3, -50, Z).astype(f32), (0, 0), 1.1)
    zz = zz + fbm(h, w, (16, 7), (1, .6), seed) * 1.2
    gx = cv2.Sobel(zz, cv2.CV_32F, 1, 0, ksize=3) / 8; gy = cv2.Sobel(zz, cv2.CV_32F, 0, 1, ksize=3) / 8
    N = np.dstack([-gx, -gy, np.ones_like(gx)]); N /= np.linalg.norm(N, axis=-1, keepdims=True)
    return Z, N.astype(f32), ID


@njit(cache=True, fastmath=True)
def _puff(L, Z, ID):
    h, w = Z.shape
    for k in range(L.shape[0]):
        x, y, z, R = L[k, 0], L[k, 1], L[k, 2], L[k, 3]
        for j in range(max(0, int(y - R)), min(h, int(y + R) + 2)):
            for i in range(max(0, int(x - R)), min(w, int(x + R) + 2)):
                d2 = (i - x) ** 2 + (j - y) ** 2
                if d2 >= R * R: continue
                q = z + math.sqrt(R * R - d2)
                if q > Z[j, i]: Z[j, i] = q; ID[j, i] = k


# ====================================================================== the six drawings
def explosion(w=1300, h=900, seed=1):
    """ground burst: white-hot heart low in the street, black-bellied smoke, thrown concrete, cracks racing toward us"""
    a = Art(w, h, seed); r = a.r; cx, cy = w * .55, h * .66; hz = h * .70
    focus_lines(a.c, cx, cy - h * .1, h * .44, 1500, 260, 8, K, seed + 3, clump=.9, ry=.95, a0=math.pi * .97, a1=math.pi * 2.03)
    for x0, x1, t0, t1 in ((-10, w * .13, .40, .46), (w * .89, w + 10, .36, .30)):
        a.black([(x0, h * t0), (x1, h * t1), (x1, h * .72), (x0, h * .72)])
        for _ in range(4): x = r.uniform(x0, x1); a.white([(x, h * .46), (x + r.normal(0, 5), h * .70)], r.uniform(1.5, 4))
    a.paper([(0, hz), (w, hz - 12), (w, h), (0, h)])
    for _ in range(24):
        th = r.uniform(.06, math.pi - .06); x, y = cx + math.cos(th) * h * .40, hz + 6 + math.sin(th) * h * .03
        crack(a, x, y, math.atan2(math.sin(th) * .50, math.cos(th) * 1.3), h * r.uniform(.3, .85), r.uniform(5, 15), 2, .30)
    for _ in range(60):
        th = r.uniform(.04, math.pi - .04); d0 = h * r.uniform(.28, .9); L = h * r.uniform(.05, .25); c, s = math.cos(th) * 1.5, math.sin(th) * .42
        a.line([(cx + c * d0, hz + s * d0), (cx + c * (d0 + L), hz + s * (d0 + L))], r.uniform(1., 3.5), taper=(.4, .6))
    g = a.grey()
    # smoke as a relief, inked
    S = []
    for i in range(34):
        t = math.pi * (1.02 + .96 * (i + r.uniform(-.3, .3)) / 34); d = h * (.36 + .15 * r.random() + .11 * abs(math.sin(t))); R = h * r.uniform(.075, .125)
        S.append((cx + math.cos(t) * d * 1.18, cy + math.sin(t) * d, r.uniform(-40, 20), R))
    for i in range(16):
        t = math.pi * (1.0 + 1.0 * i / 15); d = h * r.uniform(.27, .33); S.append((cx + math.cos(t) * d * 1.2, cy + math.sin(t) * d * .95 + 8, r.uniform(30, 60), h * r.uniform(.05, .075)))
    for i in range(22):
        x = cx + h * (-.78 + 1.56 * i / 21) + r.normal(0, 8); S.append((x, hz + h * (.015 + .03 * (1 - abs(x - cx) / (h * .8))), 90, h * r.uniform(.032, .062)))
    Z, N, ID = puffs(S, h, w, seed + 5, kids=7, gen=3, shrink=.50)
    Nb = np.dstack([cv2.GaussianBlur(N[..., i], (0, 0), 5) for i in range(3)])
    m = Z > -9e3
    x, y = np.meshgrid(np.arange(w, dtype=f32), np.arange(h, dtype=f32)); dx, dy = cx - x, (cy - 10) - y; dd = np.sqrt((dx / 1.2) ** 2 + dy ** 2) + 1
    Nm = .45 * N + .55 * Nb; core = (Nm[..., 0] * dx / (dd * 1.2) + Nm[..., 1] * dy / dd) * 1.0 + Nm[..., 2] * .55
    Lt = core * np.clip(1.50 - dd / (h * .62), .40, 1.2) + .16 * (-N[..., 1]) + .10 * fbm(h, w, (90, 34), (1, .6), seed + 9)
    dark = np.clip(.70 - Lt * .95, 0, 1) * m
    ink = hatch_tone(dark, .55 * Nb[..., 0] - .45 * dx / dd, .55 * Nb[..., 1] - .45 * dy / dd, seed + 2, 30, 3.0, .86, .16, .05) * m
    e = edge_ink(np.where(m, 1, 0).astype(np.int32), np.where(m, Z, -400), h * .016, 4.4, seed, dark)
    g = g * (1 - m) + m
    g = np.minimum(g, 1 - np.clip(ink + e, 0, 1))
    # the heart burns out to paper
    th = np.arctan2((y - cy) / .9, (x - cx) / 1.25); q = np.sqrt(((x - cx) / 1.25) ** 2 + ((y - cy + 6) / .9) ** 2); an = noise1(720, 16, r) * .5 + noise1(720, 5, r) * .35; rag = np.interp(th, np.linspace(-math.pi, math.pi, 720), an)
    sk = streaks(dx, dy, 60, 4.5, seed + 8); g = np.maximum(g, (q + rag * h * .04 + (sk - .5) * h * .13 < h * .25 + np.maximum(0, y - hz) * -2.5).astype(f32))
    g = cv2.GaussianBlur(g, (0, 0), .7)
    b = Art(w, h, seed + 1); r = b.r; b.c = Cel(w, h); b.n = 999
    for _ in range(130):
        t = r.uniform(math.pi * .96, math.pi * 2.04); q0 = h * r.uniform(.16, .30); L = h * r.gamma(1.5, .05); c, s = math.cos(t) * 1.25, math.sin(t) * .9
        b.line([(cx + c * q0, cy + s * q0), (cx + c * (q0 + L), cy + s * (q0 + L))], r.uniform(1.2, 6), taper=(.5, .6))
    for i in range(80):
        t = r.uniform(math.pi * 1.0, math.pi * 2.0) if r.random() < .85 else r.uniform(-.2, math.pi + .2); d = h * r.uniform(.30, .95); k = r.random() ** 3
        xx, yy = cx + math.cos(t) * d * 1.25, cy + math.sin(t) * d * .95; R = 3 + 11 * k + (30 * r.random() ** 2 if r.random() < .14 else 0)
        if R > 8:
            u = np.array([math.cos(t) * 1.25, math.sin(t)]); u /= np.hypot(*u); nn = np.array([-u[1], u[0]])
            for v in r.uniform(-.8, .8, 6): b.line([np.array([xx, yy]) - u * R + nn * R * v, np.array([xx, yy]) - u * (R + r.uniform(1, 4.5) * R) + nn * R * v * .6], max(1.2, R * .08), taper=(.05, .8))
            rock(b, xx, yy, R, r.uniform(0, 6), max(1.8, R * .10))
        else: b.c.fill(blob(xx, yy, R * .7, r, t, 1.8, 0, 7), K)
    for xx, yy, R in ((w * .09, h * .92, 66), (w * .27, h * .965, 34), (w * .90, h * .91, 80), (w * .72, h * .97, 28), (w * .18, h * .80, 20), (w * .80, h * .80, 18)): rock(b, xx, yy, R, r.uniform(0, 6), 5.)
    splatter(b.c, cx, cy, h * .36, 120, 8, K, seed)
    q = b.c.rgba(); return np.clip(g * (1 - q[..., 3]) + lum(q[..., :3]), 0, 1)


def _hand_relief(hand, cam, w, h, light=(-.5, .7, .55), clip=None):
    """g-buffer of a jj.hands Hand resampled to the panel: z (px-ish), N (x right, y down, z out), gid, prim, lit, mask"""
    from . import hands as HD
    sc = HD.Scene([hand]); g = HD.gbuffer(sc, cam, light=light, res=1350, clip=(0, 0, w, h))
    s = 1. / (g['px'] * cam.s)
    M = np.float32([[g['px'] * cam.s, 0, cam.cx + g['x0'] * cam.s], [0, g['px'] * cam.s, cam.cy - g['y0'] * cam.s]])
    W_ = lambda a, fl, bv: cv2.warpAffine(a, M, (w, h), flags=fl, borderMode=cv2.BORDER_CONSTANT, borderValue=bv)
    gid = W_(g['gid'].astype(np.float32), cv2.INTER_NEAREST, 0).astype(np.int32); prim = W_(g['prim'].astype(np.float32), cv2.INTER_NEAREST, -1).astype(np.int32)
    z = W_(np.where(g['prim'] >= 0, g['z'], -9).astype(f32), cv2.INTER_NEAREST, -9); lit = W_(g['lit'].astype(f32), cv2.INTER_LINEAR, 1.)
    n = np.dstack([W_(g['N'][..., i].astype(f32), cv2.INTER_LINEAR, 0) for i in range(3)]); n = n * np.array([1, -1, 1], f32)
    return dict(gid=gid, prim=prim, z=z * cam.s, N=n, lit=lit, m=gid > 0)


def ink_hand(g, w, h, seed=0, light=(-.5, -.7, .5), lw=5., black=.86, amb=.10, cast=.55, cuff=None):
    """manga-ink a hand relief: outline weight by shadow, feathered hatching, black undersides"""
    N = g['N']; m = g['m']; l = np.array(light) / np.linalg.norm(light); gid = g['gid'] % 16
    Nb = np.dstack([cv2.GaussianBlur(N[..., i], (0, 0), 6) for i in range(3)])
    dif = N @ l.astype(f32); dark = np.clip(.46 - .72 * dif, 0, 1) + cast * (1 - np.clip(g['lit'], 0, 1)) * .55 + amb * np.clip(1 - N[..., 2], 0, 1)
    dark = cv2.GaussianBlur((dark * m).astype(f32), (0, 0), 1.2); dark = np.clip(dark, 0, 1) * m
    if cuff is not None: dark = np.where(np.isin(gid, (7, 14)), np.clip(.60 + .8 * (dark - .2), 0, 1), dark)
    ink = hatch_tone(dark, Nb[..., 0] + .12, Nb[..., 1] + .05, seed, 30, 2.6, black, .30, .06) * m
    nail = np.isin(gid, (8, 9, 10, 11, 12))
    e1 = edge_ink(np.where(m, 1, 0).astype(np.int32), np.where(m, 0, -99.).astype(f32), 1e9, lw * 1.35, seed + 1, dark)
    e2 = edge_ink(np.where(m, g['gid'], 0).astype(np.int32), np.where(m, g['z'], -999).astype(f32), w * .012, lw * .80, seed + 2, dark, cv2.erode(m.astype(np.uint8), np.ones((3, 3))) > 0)
    e3 = edge_ink(nail.astype(np.int32), np.zeros((h, w), f32), 1e9, lw * .40, seed + 3, dark)
    e2 = np.where(cv2.dilate(nail.astype(np.uint8), np.ones((7, 7))) > 0, np.minimum(e2, e3 * 1.0 + 0), e2)
    return np.clip(np.maximum(np.maximum(ink, e1), np.maximum(e2, e3)), 0, 1), dark


def _sleeve(w, h, seed, top, left, right, whites, cuff=None):
    """black cuff / sleeve with flying white folds -> Art layer"""
    c = Art(w, h, seed); c.c = Cel(w, h); r = c.r
    body = np.vstack([top, right, left]); c.shape(np.vstack([body, body[:1]]), 6, K)
    if cuff is not None:
        q = c.pts(cuff); c.black(q); n = len(q); k = np.argmax(q[:, 0]); lo = q[:k + 1] if np.argmin(q[:, 0]) < k else q
        c.white(lo + [0, 3], 4.5, taper=(.1, .1)); m = q.mean(0)
        for t in np.linspace(.12, .88, 13): x = q[:, 0].min() + t * np.ptp(q[:, 0]); c.white([(x, m[1] + 4), (x + r.normal(0, 3), m[1] + r.uniform(10, 22))], r.uniform(1, 2.2), taper=(.6, .2))
    for d in whites:
        c.white(d, r.uniform(3, 9), taper=(.05, .85))
        p = c.pts(d); 
        for k in range(r.integers(2, 5)): c.white(p + r.normal(0, 7, 2) + [6 * k, 8 * k], r.uniform(1, 2.6), taper=(.1, .9))
    q = c.c.rgba(); return q


def fist(w=1000, h=1000, seed=2):
    """the clenched fist: raised, knuckles lit from the upper left, forearm running into a black sleeve"""
    from . import hands as HD
    a = Art(w, h, seed); r = a.r; cx, cy = w * .50, h * .40
    focus_lines(a.c, cx, cy, h * .46, 1500, 360, 9, K, seed, clump=1.1, ry=1.0)
    g = a.grey()
    hd = HD.Hand('fist', 'R', HD.euler(yaw=-27, pitch=47, roll=13), (0, 0, 0), HD.HEAVY, 1.0)
    cam = HD.Cam(cx - 10, cy, 452, 3.4)
    b = _hand_relief(hd, cam, w, h, (-.60, .62, .50))
    ink, dark = ink_hand(b, w, h, seed, (-.60, -.62, .50), 6.0, black=.84)
    halo = cv2.dilate(b['m'].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))).astype(f32)
    g = np.maximum(g, halo); g = np.minimum(g, 1 - ink)
    q = _sleeve(w, h, seed + 2, spl(np.array([(366, 812), (470, 846), (590, 850), (700, 812)], float)), np.array([(250, 1010), (300, 930), (340, 850)], float), np.array([(742, 900), (780, 1010)], float),
                ('M 400 862 C 390 910 372 960 350 1010', 'M 470 880 C 468 930 452 970 440 1010', 'M 640 866 C 664 910 680 960 690 1010', 'M 560 884 C 566 930 566 970 566 1010'),
                'M 366 812 C 470 846 590 850 700 812 C 690 788 660 776 640 772 C 560 800 480 798 410 776 C 388 786 374 798 366 812')
    g = g * (1 - q[..., 3]) + lum(q[..., :3])
    b = Art(w, h, seed + 1); b.c = Cel(w, h); r = b.r
    for _ in range(40):
        t = r.uniform(0, 2 * math.pi); q0 = h * r.uniform(.36, .52); L = h * r.gamma(1.6, .06); c, s = math.cos(t), math.sin(t)
        b.line([(cx + c * q0, cy + s * q0), (cx + c * (q0 + L), cy + s * (q0 + L))], r.uniform(2, 7), taper=(.12, .7))
    q = b.c.rgba(); return np.clip(g * (1 - q[..., 3]) + lum(q[..., :3]), 0, 1)


def fingers(w=800, h=1100, seed=5):
    """two fingers raised, the rest folded under the thumb: a sign held up against the light"""
    from . import hands as HD
    a = Art(w, h, seed); r = a.r; cx, cy = w * .50, h * .56
    focus_lines(a.c, w * .50, h * .16, h * .24, 1700, 330, 6.5, K, seed, clump=1.2, ry=1.1)
    g = a.grey()
    P = HD.pose((2, 6, 3, 2), (5, 8, 4, 0), (82, 104, 58, 0), (86, 104, 56, 0), thumb=(12, 61, -67, 55, 28))
    hd = HD.Hand(P, 'R', HD.euler(yaw=172, pitch=6, roll=7), (0, 0, 0), HD.SLIM, 1.0)
    cam = HD.Cam(cx - 6, cy + 10, 352, 6.5)
    b = _hand_relief(hd, cam, w, h, (-.65, .5, .5))
    ink, dark = ink_hand(b, w, h, seed, (-.65, -.5, .5), 5.0, black=.84)
    halo = cv2.dilate(b['m'].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))).astype(f32)
    g = np.maximum(g, halo); g = np.minimum(g, 1 - ink)
    q = _sleeve(w, h, seed + 2, spl(np.array([(268, 934), (360, 972), (470, 972), (572, 926)], float)), np.array([(160, 1120), (196, 1040), (232, 970)], float), np.array([(630, 1010), (676, 1120)], float),
                ('M 306 990 C 296 1030 276 1070 262 1110', 'M 380 1004 C 376 1040 366 1076 360 1110', 'M 540 986 C 566 1030 584 1070 590 1110', 'M 466 1004 C 474 1040 476 1074 478 1110'),
                'M 268 934 C 360 972 470 972 572 926 C 562 900 540 888 524 884 C 450 914 380 914 314 894 C 288 904 274 918 268 934')
    return np.clip(g * (1 - q[..., 3]) + lum(q[..., :3]), 0, 1)


def limb(L, p0, r0, p1, r1, pid, n=70, mid=None, sq=1.0):
    """append a chain of balls from p0 to p1 (x, y, z px), radii r0 -> r1, optional mid-radius bulge"""
    t = np.linspace(0, 1, n)[:, None]; P = np.asarray(p0, float) * (1 - t) + np.asarray(p1, float) * t
    R = r0 * (1 - t[:, 0]) + r1 * t[:, 0] + (0 if mid is None else mid * np.sin(t[:, 0] * math.pi))
    for q, rr, u in zip(P, R, t[:, 0]): L.append((q[0], q[1], q[2], rr, pid, u))


def relief(L, w, h, sm=2.2, folds=None, seed=0):
    """balls -> z, N, part id, u (0..1 along the limb)"""
    A = np.array([q[:4] for q in L], float); Z = np.full((h, w), -1e4, f32); ID = np.full((h, w), -1, np.int32); _puff(A, Z, ID)
    m = ID >= 0; pid = np.array([q[4] for q in L] + [-1])[ID]; u = np.array([q[5] for q in L] + [0.])[ID].astype(f32)
    zz = np.where(m, Z, 0).astype(f32); mm = m.astype(f32)
    zb = cv2.GaussianBlur(zz, (0, 0), sm) / np.maximum(cv2.GaussianBlur(mm, (0, 0), sm), 1e-3)
    if folds is not None: zb = zb + folds
    gx = cv2.Sobel(zb, cv2.CV_32F, 1, 0, ksize=3) / 8; gy = cv2.Sobel(zb, cv2.CV_32F, 0, 1, ksize=3) / 8
    # silhouettes: the blur flattens them, so lean the normal outward near the rim
    d = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5); db = cv2.GaussianBlur(d, (0, 0), 3); ex = cv2.Sobel(db, cv2.CV_32F, 1, 0) / 8; ey = cv2.Sobel(db, cv2.CV_32F, 0, 1) / 8; k = np.exp(-d / 9.)
    N = np.dstack([-gx - ex * k * 1.4, -gy - ey * k * 1.4, np.ones_like(gx)]); N /= np.linalg.norm(N, axis=-1, keepdims=True)
    return Z, N.astype(f32), np.where(m, pid, -1), u, m


def runner(w=1000, h=1200, seed=3):
    """back view: a spiky-haired fighter in a dark jacket tearing away from us.  One line of action: trailing foot -> spine -> crown."""
    from .kit import spike
    from .sumi import Page
    a = Art(w, h, seed); r = a.r; vx, vy = w * .70, h * .16
    focus_lines(a.c, vx, vy, h * .30, 2000, 250, 9, K, seed, clump=1.3, ry=1.1)
    for x, y, an, L in ((560, 1130, 2.7, 300), (600, 1120, .25, 330), (560, 1150, 3.5, 240), (620, 1130, -.3, 300), (580, 1170, 1.5, 120)): crack(a, x, y, an, L, 15, 2, .28)
    a.paper('M 330 300 L 470 60 L 620 10 L 800 110 L 840 330 L 930 250 L 985 330 L 900 560 L 790 660 L 800 900 L 720 1140 L 560 1160 L 470 900 L 260 1140 L 60 1180 L 40 1040 L 280 800 L 250 620 L 60 640 L 70 420 L 250 360 Z')
    J = dict(
        armR='M 690 372 C 734 364 776 378 800 416 L 818 426 C 826 450 842 468 864 482 L 890 500 C 894 530 886 548 868 558 L 842 552 C 812 554 790 542 770 526 L 744 508 L 738 486 C 722 456 704 424 690 372 Z',
        foreR='M 842 556 L 872 562 C 896 552 910 530 904 504 L 918 482 L 912 456 C 920 420 930 386 944 354 L 928 342 L 876 330 L 874 352 C 862 392 852 430 838 462 L 824 486 L 832 502 Z',
        thighR='M 500 700 C 560 690 640 690 706 704 C 752 716 778 748 770 786 C 756 822 706 836 660 828 C 596 826 536 806 500 776 Z',
        shinR='M 770 756 C 786 800 776 846 758 886 C 742 930 722 980 706 1030 L 634 1016 C 638 960 640 900 650 850 C 660 800 712 758 770 756 Z',
        thighL='M 360 660 C 340 730 330 800 318 872 C 324 920 372 944 418 918 C 462 872 506 800 540 726 Z',
        shinL='M 318 862 C 262 900 206 952 150 1010 L 206 1076 C 280 1032 360 980 420 916 C 424 872 366 840 318 862 Z',
        torso='M 500 336 C 462 342 420 356 388 384 C 380 440 366 500 352 552 L 338 590 L 350 606 C 338 652 320 694 296 732 L 268 776 C 312 766 340 786 378 780 L 410 806 C 452 792 500 806 540 788 L 566 756 L 612 742 C 608 690 620 640 640 590 C 664 530 692 476 716 416 C 700 384 660 358 620 344 Z',
        armL='M 404 360 C 360 350 316 366 290 398 C 250 440 206 474 160 500 C 132 520 128 552 150 570 C 190 590 244 574 290 546 C 336 516 380 478 410 436 Z',
        foreL='M 158 502 C 122 520 110 560 130 594 C 152 640 182 680 214 716 L 270 676 C 254 634 242 594 236 558 C 222 520 190 496 158 502 Z')
    kw = dict(col=P_, taper=(.1, .1))
    def part(k, w_=5, kl=3.8): a.shape(J[k], w_, K); a.line(J[k], kl, **kw)
    # far hand, far arm
    part('armR'); part('foreR')
    a.shape('M 884 342 C 872 322 874 296 890 282 C 898 268 916 264 928 274 C 942 268 956 276 960 290 C 974 298 976 320 966 336 C 962 350 948 358 936 356 Z', 5, P_)
    a.line('M 896 284 C 902 298 902 312 896 326 M 928 276 C 928 292 926 306 918 320 M 958 292 C 954 308 948 320 940 330', 3, taper=(.05, .8)); a.hatch('M 936 356 C 948 358 962 350 966 336 C 976 320 974 298 960 290 C 962 316 952 338 930 350 Z', .8, 5, 1.9)
    # front leg (driving up), then the trailing leg
    part('shinR'); a.shape('M 634 1014 C 622 1040 622 1066 634 1092 C 664 1122 722 1128 760 1112 C 768 1086 744 1060 708 1034 Z', 5, K); a.white('M 630 1082 C 664 1104 716 1108 758 1098', 4.5); a.white('M 650 1024 C 644 1044 646 1060 652 1076', 3)
    part('thighR'); part('thighL'); part('shinL')
    # trailing shoe: sole to us, big
    a.shape('M 160 1000 C 130 1000 92 1024 66 1062 C 36 1108 30 1160 56 1182 C 90 1200 146 1176 186 1136 C 222 1102 240 1066 226 1040 C 212 1014 190 1000 160 1000 Z', 7, K)
    sole = 'M 158 1018 C 134 1018 102 1040 80 1074 C 56 1112 50 1150 70 1164 C 96 1176 142 1154 174 1122 C 202 1094 216 1066 206 1048 C 196 1030 180 1018 158 1018 Z'; a.paper(sole)
    for t in np.linspace(.04, .96, 9):
        c = np.array([182., 1034.]) * (1 - t) + np.array([76., 1148.]) * t; wd = 40 + 12 * math.sin(t * 3.1); d = np.array([.70, .72]) * wd; n_ = np.array([-.5, .5]) * 12
        a.line([c - d * .92, c + n_, c + d * .92], 8.5, taper=(.06, .06))
    a.hatch('M 206 1048 C 216 1066 202 1094 174 1122 C 142 1154 96 1176 70 1164 L 120 1120 L 190 1060 Z', 1.2, 6, 2.2); a.line(sole, 3)
    part('torso'); part('foreL', kl=4.4); part('armL', kl=4.4)
    # near fist
    a.shape('M 206 712 C 198 732 200 752 212 766 C 214 784 232 798 250 794 C 264 804 284 798 292 784 C 308 778 316 758 308 740 C 306 718 292 694 272 676 C 248 684 224 696 206 712 Z', 5.5, P_)
    a.line('M 212 766 C 220 756 230 748 242 744 M 250 794 C 254 780 260 768 268 760 M 292 784 C 292 772 294 760 298 750', 3.2, taper=(.05, .7)); a.black('M 212 766 C 226 758 238 762 248 770 C 262 764 276 770 286 766 C 298 760 306 752 308 740 C 316 758 308 778 292 784 C 284 798 264 804 250 794 C 232 798 214 784 212 766 Z'); a.hatch('M 210 748 C 214 724 232 706 258 694 L 270 720 L 238 750 Z', 1.0, 6, 1.8)
    # light shards from the tension points: shoulder blades, elbow, belt, knees
    for d in ('M 166 506 L 232 470 L 300 408 L 326 396 L 292 446 L 224 500 Z', 'M 340 372 L 384 362 L 396 374 L 360 408 Z', 'M 150 560 L 214 552 L 250 566 L 200 576 Z', 'M 142 590 L 166 640 L 200 688 L 176 664 Z', 'M 264 520 L 318 468 L 360 436 L 318 498 Z',
              'M 420 366 L 486 372 L 560 360 L 498 396 L 446 398 Z', 'M 400 430 L 440 500 L 452 590 L 412 530 Z', 'M 604 366 L 676 392 L 640 404 L 600 390 Z', 'M 540 400 L 530 500 L 500 620 L 512 500 Z', 'M 676 440 L 620 530 L 572 590 L 632 486 Z', 'M 360 620 L 420 650 L 500 654 L 430 676 L 366 672 Z', 'M 600 620 L 560 676 L 496 704 L 566 650 Z',
              'M 292 760 L 326 700 L 342 676 L 330 736 Z', 'M 592 736 L 566 690 L 560 660 L 586 700 Z', 'M 424 700 L 432 752 L 424 790 L 414 746 Z', 'M 346 720 L 336 800 L 352 862 L 366 786 Z', 'M 470 740 L 440 820 L 402 880 L 424 806 Z', 'M 300 888 L 238 948 L 190 1010 L 252 954 Z', 'M 380 910 L 320 970 L 260 1014 L 330 950 Z',
              'M 540 716 L 630 712 L 700 724 L 620 736 Z', 'M 690 760 L 740 756 L 758 782 L 712 780 Z', 'M 748 800 L 728 880 L 700 960 L 714 876 Z', 'M 664 860 L 656 930 L 648 1000 L 646 930 Z', 'M 716 392 L 772 416 L 820 470 L 764 438 Z', 'M 780 500 L 826 522 L 862 530 L 816 538 Z', 'M 890 396 L 876 450 L 858 506 L 868 446 Z'):
        p = a.pts(d); a.c.fill(p, P_); c = p.mean(0)
        for k in range(4): q0 = p[r.integers(len(p))]; q1 = p[r.integers(len(p))]; o = r.normal(0, 7, 2) + (q0 - c) * .1; a.white([q0 + o, q1 + o], r.uniform(1.2, 2.8), taper=(.1, .8))
    a.white('M 566 350 C 552 410 530 470 506 540 C 488 610 462 690 432 782', 2.6, taper=(.2, .4)); a.white('M 270 768 C 316 756 344 776 382 770 L 412 796 C 452 782 500 796 538 778 L 562 748 L 606 734', 2.4)
    # collar, nape, ears
    a.shape('M 492 372 C 486 326 506 292 530 284 C 560 300 606 306 636 296 C 660 314 666 350 652 386 C 606 408 536 402 492 372 Z', 5, K); a.line('M 492 372 C 536 402 606 408 652 386', 4, **kw); a.paper('M 506 350 L 520 306 L 530 306 L 522 360 Z'); a.paper('M 626 374 L 642 322 L 650 338 L 638 376 Z'); a.white('M 574 310 C 572 336 572 364 568 396', 2.6, taper=(.2, .7))
    a.shape('M 530 300 C 526 276 530 258 540 244 L 612 250 C 616 270 612 290 604 306 C 580 314 552 312 530 300 Z', 0, P_); a.hatch('M 530 300 C 526 276 530 258 540 244 L 612 250 C 616 270 612 290 604 306 C 580 314 552 312 530 300 Z', 1.4, 5, 2.4)
    # kicked-up concrete, rush of air
    for _ in range(22): t = r.uniform(0, 1); d = r.uniform(20, 280); rock(a, 600 + r.choice([-1, 1]) * d, 1120 - t * t * 150 + r.uniform(-8, 40), r.uniform(4, 22) * (1 - .4 * t), r.uniform(0, 6), 3)
    for _ in range(14): t = r.uniform(0, 1); rock(a, 190 + r.normal(0, 70) + 80 * t, 1060 - t * 190 + r.normal(0, 30), r.uniform(5, 20), r.uniform(0, 6), 3)
    g = a.grey()
    # hair by brush: six big unequal clumps torn back by the run, then the small ones that disobey
    b = Art(w, h, seed + 1); b.c = Cel(w, h); r = b.r; pg = Page(h, w, seed=seed + 2)
    cl = [  # root a, tip, root b, bulge, bend, outline weight
        ((498, 250), (330, 320), (520, 290), .10, .05, 9), ((496, 206), (262, 196), (500, 266), .10, .06, 13), ((520, 164), (330, 46), (496, 232), .09, .05, 14), ((566, 146), (470, 40), (516, 190), .12, .02, 9),
        ((624, 150), (556, -40), (536, 172), .08, .07, 15), ((648, 170), (668, 30), (594, 152), .12, -.04, 9), ((664, 210), (800, 66), (616, 160), .10, -.07, 13), ((662, 250), (770, 190), (640, 200), .14, -.03, 8), ((646, 284), (742, 304), (650, 238), .14, .02, 8)]
    skull = spl(np.array([(500, 280), (492, 226), (510, 172), (562, 140), (622, 146), (662, 190), (664, 246), (640, 292), (570, 306)], float), True, n=10); b.c.fill(skull, P_)
    front = [((520, 226), (404, 300), (560, 258), .12, .07, 9), ((552, 196), (466, 342), (604, 246), .11, .05, 12), ((590, 214), (556, 362), (640, 254), .12, .03, 12), ((622, 214), (640, 340), (664, 250), .13, -.04, 9), ((640, 226), (708, 330), (666, 262), .14, -.05, 7)]
    ink = np.zeros((h, w), f32); lay_ = np.zeros((h, w, 4), f32)
    def clump(q, i, inner):
        p0, tp, p1, bl, bd, ow = q; P = spike(p0, tp, p1, bl, bd, n=16); m = (np.array(p0) + p1) / 2; tp = np.array(tp, float); d = tp - m; L = np.hypot(*d); d /= L; sd = np.array([-d[1], d[0]])
        c = Cel(w, h); c.fill(P, P_); pm = c.rgba()[..., 3]
        pg = Page(h, w, seed=seed + 40 + i); n = len(P) // 2
        for sgn, side in ((1, P[:n + 1]), (-1, P[::-1][:n + 1])):             # two brush strokes meet at the tip, heavier on the shadow side
            hv = 1.35 if np.dot(side[n // 2] - m, np.array([.5, .85])) > 0 else .70; k = resample(side, L / 7.); t = np.linspace(0, 1, len(k)); wd = ow * hv * (.55 + .75 * np.sin(np.clip(t * 1.25, 0, 1) * math.pi) ** .8) * (1 - .75 * smooth(.80, 1., t))
            pg.stroke([(x, y, v_, 1.2) for (x, y), v_ in zip(k, wd)], dry=.10, deplete=.25, load=1., split=.6, fork=.5, seed=seed * 7 + i * 3 + sgn, bleed=.8, tufts=8, gain=.9)
        for k in range(inner):                                                # the sweep of the clump, dry brush
            o = r.uniform(-.30, .30); t0, t1 = r.uniform(.16, .40), r.uniform(.60, .92); f_ = lambda t: m + d * L * t + sd * (o * np.hypot(*(np.array(p1) - p0)) * (1 - t) ** .9 + bd * L * .5 * math.sin(t * 3.1)); wv = r.uniform(2.0, 5.5)
            pg.stroke([(*f_(t0), wv * .5), (*f_((t0 + t1) / 2), wv), (*f_(t1), .8)], dry=.25, deplete=.4, load=.8, split=.9, fork=.6, seed=seed * 11 + i * 17 + k, bleed=0, tufts=7, gain=.9)
        return pm, pg.ink()[0]
    lay_[..., 3] = Cel(w, h).fill(skull, P_).rgba()[..., 3]
    for i, q in enumerate(cl + front):
        pm, ik = clump(q, i, 7 if q[5] > 10 else 4); ink = ink * (1 - pm); ink = 1 - (1 - ink) * (1 - ik); lay_[..., 3] = np.maximum(lay_[..., 3], pm)
    # black pooled between the clumps at the nape, dry shadow on the far side of the skull
    sh = Page(h, w, seed=seed + 5); sh.stroke([(668, 218, 6), (668, 252, 20), (654, 284, 16), (630, 304, 3)], dry=.15, split=.6, fork=.4, seed=61); sh.stroke([(500, 270, 6), (512, 290, 20), (536, 306, 12), (560, 312, 3)], dry=.1, split=.5, fork=.3, seed=62)
    for x0, y0, x1, y1, wv, sd_ in ((522, 262, 478, 320, 20, 63), (590, 262, 524, 344, 20, 64), (632, 262, 596, 346, 18, 65), (652, 262, 660, 322, 12, 66)): sh.stroke([(x0, y0, wv), ((x0 * 2 + x1) / 3, (y0 * 2 + y1) / 3, wv * .8), (x1, y1, 1)], dry=.1, deplete=.2, split=.5, fork=.3, seed=sd_, gain=.9)
    ink = 1 - (1 - ink) * (1 - sh.ink()[0] * .95)
    g = g * (1 - lay_[..., 3]) + lay_[..., 3]; g = np.minimum(g, 1 - ink)
    b = Art(w, h, seed + 2); b.c = Cel(w, h); r = b.r
    for _ in range(26):
        x0 = r.uniform(20, 980); y0 = r.uniform(240, 1080); d = np.array([x0 - vx, y0 - vy]); d /= np.hypot(*d); L_ = r.uniform(80, 280)
        if 120 < x0 < 900 and not (x0 < 300 and y0 > 700): continue
        b.line([(x0, y0), (x0 + d[0] * L_, y0 + d[1] * L_)], r.uniform(2, 7), taper=(.7, .1))
    splatter(b.c, 600, 1120, 150, 60, 6, K, seed); q = b.c.rgba()
    return np.clip(g * (1 - q[..., 3]) + lum(q[..., :3]), 0, 1)


def eye(w=1300, h=640, seed=4):
    """close: a man's eye under the fringe, lid like a blade, the iris contracted"""
    from .kit import spike
    a = Art(w, h, seed); r = a.r
    lid = a.pts('M 196 420 C 268 356 392 298 560 276 C 748 256 930 282 1112 380')             # upper edge of the opening
    low = a.pts('M 196 420 C 300 478 470 516 640 518 C 820 516 1000 470 1112 380')            # lower edge
    # tone and hatching of the orbit
    a.tone('M 0 0 L 1300 0 L 1300 240 C 1200 170 1000 128 780 128 C 520 128 260 210 60 380 L 0 420 Z', .24, 6.5)
    a.hatch('M 130 372 C 300 236 520 176 740 178 C 900 182 1040 224 1150 312 L 1190 262 C 1080 170 920 122 740 118 C 520 120 290 190 120 330 Z', -.62, 8, 2.2, fade=.35, length=90)
    a.hatch('M 200 360 C 330 270 530 212 740 214 C 900 216 1030 250 1130 330 C 980 262 800 238 620 250 C 440 262 300 310 220 372 Z', -.35, 7, 2.2, fade=.1)
    a.hatch('M 210 478 C 360 552 520 580 680 576 C 860 570 1000 524 1090 450 L 1100 480 C 1010 560 860 606 690 610 C 500 606 330 560 230 506 Z', .42, 9, 2., fade=.2, length=80)
    op = np.vstack([lid, low[::-1]]); a.paper(op); a.c.clip(op)
    ix, iy, R = 664, 384, 146; th = np.linspace(0, 2 * math.pi, 200)
    P = np.c_[ix + np.cos(th) * R * .95, iy + np.sin(th) * R * 1.04] + np.c_[noise1(200, 16, r), noise1(200, 16, r)] * 1.3
    a.c.fill(P, P_)
    for k in range(210):
        t = r.uniform(0, 2 * math.pi); up = .5 - .5 * math.sin(t); q1 = R * r.uniform(.84, .98); q0 = R * (r.uniform(.46, .60) if r.random() < .5 + .4 * up else r.uniform(.66, .80))
        a.line([(ix + math.cos(t) * q1 * .95, iy + math.sin(t) * q1 * 1.04), (ix + math.cos(t + .02) * q0 * .95, iy + math.sin(t + .02) * q0 * 1.04)], r.uniform(2, 6.5) * (.45 + .7 * up), taper=(.02, .9))
    for k in range(80): t = r.uniform(0, 2 * math.pi); q0 = R * r.uniform(.24, .30); q1 = q0 + R * r.gamma(1.6, .07); a.line([(ix + math.cos(t) * q0, iy + math.sin(t) * q0 * 1.05), (ix + math.cos(t) * q1, iy + math.sin(t) * q1 * 1.05)], r.uniform(2, 5), taper=(.02, .85))
    for rr, ww in ((.985, 9), (.93, 2.6), (.62, 1.6)): a.c.line((P - [ix, iy]) * rr + [ix, iy], ww, K, taper=(0, 0), press=.5, seed=a.sd())
    q = np.linspace(0, 2 * math.pi, 60); a.c.fill(np.c_[ix + np.cos(q) * R * .24, iy + np.sin(q) * R * .27] + r.normal(0, .8, (60, 2)), K)
    a.black(np.vstack([lid + [0, -20], (lid + [0, 30 + 0])[::-1]]))
    sl = lid + [0, 28]; a.feather(sl[4:-4], sl[4:-4] + [0, 44], 120, 6, (.15, 1.0))
    a.paper('M 598 330 L 636 318 L 648 352 L 612 366 Z'); a.c.fill(blob(726, 440, 11, r, .4, 1.7, 0, 14), P_)
    for d in ('M 214 424 C 280 420 340 430 400 446', 'M 250 440 C 310 444 360 454 410 470', 'M 270 408 C 320 402 370 404 410 414', 'M 1080 396 C 1030 400 980 412 930 430', 'M 1070 410 C 1020 424 970 440 920 456', 'M 1040 378 C 1000 378 960 384 920 396', 'M 330 432 C 344 440 352 452 354 466', 'M 980 420 C 968 430 962 444 964 456'): a.line(d, 2.2, taper=(.05, .8))
    a.c.unclip()
    # lid: one brushed blade, split at the ends
    up = a.pts('M 150 452 C 196 396 280 330 390 286 C 540 230 760 222 930 258 C 1040 284 1120 326 1216 366'); lo = a.pts('M 1216 366 C 1160 372 1130 384 1112 396 C 1000 316 850 280 700 284 C 520 288 330 342 226 418 C 204 432 180 446 150 452'); a.black(np.vstack([up, lo]))
    for x0, y0, x1, y1, ww in ((1140, 352, 1262, 372, 9), (1100, 334, 1230, 322, 10), (1050, 306, 1160, 270, 9), (990, 280, 1080, 236, 8), (920, 262, 984, 222, 7), (236, 384, 160, 384, 7), (290, 342, 222, 324, 7)): a.line([(x0, y0), ((x0 + x1) / 2 + 8, (y0 + y1) / 2 + 6), (x1, y1)], ww * 1.6, taper=(.02, .9))
    for k in range(26): t = r.uniform(.1, .95); p = up[int(t * (len(up) - 1))]; a.line([p + [0, 4], p + [r.uniform(8, 40) * (t - .35) * 2, -r.uniform(6, 22)]], r.uniform(2, 5), taper=(.02, .9))
    a.line(low, 5.5, taper=(.30, .12)); a.line(low + [0, 5], 3.2, taper=(.6, .05))
    for t in (.62, .70, .78, .85, .91): p = low[int(t * (len(low) - 1))]; a.line([p, p + [r.uniform(14, 30), r.uniform(22, 38)]], 5, taper=(.02, .85))
    a.line('M 250 504 C 400 584 600 606 800 588', 3.2, taper=(.3, .6)); a.line('M 180 470 C 230 508 280 534 340 556', 3., taper=(.1, .7)); a.line('M 196 420 C 180 432 160 440 136 442', 5, taper=(.1, .7))
    a.line('M 176 330 C 320 232 520 182 740 184 C 900 188 1030 226 1140 300', 6.5, taper=(.5, .4))
    # brow: pressed down, hair by hair over a black core
    bu = a.pts('M 96 300 C 190 190 350 106 560 70 C 760 40 980 52 1260 120'); bl = a.pts('M 1260 120 C 1040 110 820 118 620 148 C 440 176 280 232 120 330'); a.black(np.vstack([bu, bl]))
    for k in range(150): t = r.random() ** .9; p = bu[int(t * (len(bu) - 1))]; q = bl[::-1][int(t * (len(bl) - 1))]; m = p * (u := r.random()) + q * (1 - u); a.line([m, m + [r.uniform(20, 70), -r.uniform(14, 46) * (1 - t * .7)]], r.uniform(3, 8), taper=(.05, .9))
    for k in range(34): t = r.uniform(.05, .95); p = bu[int(t * (len(bu) - 1))]; q = bl[::-1][int(t * (len(bl) - 1))]; m = p * (u := r.random()) + q * (1 - u); a.white([m, m + [r.uniform(16, 44), -r.uniform(8, 24)]], r.uniform(1.4, 3), taper=(.2, .8))
    for d in ('M 40 300 C 70 270 84 236 86 200', 'M 64 340 C 92 310 110 280 116 250', 'M 30 250 C 50 220 58 190 56 160'): a.line(d, 5, taper=(.2, .7))
    # cheek: scratches, dust
    a.feather('M 1240 320 C 1268 420 1290 520 1300 640', 'M 1150 420 C 1190 500 1210 570 1210 640', 36, 4.5, (.2, .9))
    for x0, y0 in ((900, 560), (940, 590), (1000, 560)): a.line([(x0, y0), (x0 + 90, y0 - 60)], 4.5, taper=(.4, .5)); a.line([(x0 + 20, y0 + 6), (x0 + 70, y0 - 40)], 2, taper=(.4, .5))
    splatter(a.c, 300, 560, 130, 34, 7, K, seed); splatter(a.c, 1120, 540, 90, 20, 6, K, seed + 1)
    # fringe across
    for (ax, ay), (tx, ty), (bx, by), bl_, bd in (((-60, -60), (150, 520), (90, -60), .03, .04), ((60, -60), (300, 290), (200, -60), .04, .04), ((470, -60), (452, 190), (600, -60), .05, -.05), ((930, -60), (1030, 300), (1080, -60), .04, .04), ((1100, -60), (1330, 420), (1330, -60), .04, .04)):
        p = spike((ax, ay), (tx, ty), (bx, by), bl_, bd); a.c.fill(p, K); a.line(p, 5, col=P_, taper=(0, 0)); m = np.array([(ax + bx) / 2, ay]); T = np.array([tx, ty], float)
        for k in range(9): o = r.uniform(-.40, .40) * np.array([bx - ax, 0]); t0, t1 = r.uniform(.1, .5), r.uniform(.6, .96); f_ = lambda t: m + (T - m) * t + o * (1 - t) ** .8; a.c.line(spl([f_(t0), f_((t0 + t1) / 2), f_(t1)], n=10), r.uniform(1.6, 4.5), P_, taper=(.3, .6), press=.5, seed=a.sd())
    return a.grey()


def street(w=1400, h=900, seed=6):
    """the shattered street: canted camera, broken towers, heaved slabs, fallen poles"""
    a = Art(w, h, seed); r = a.r; f = 900.; vx, vy = w * .56, h * .40; tl = -.075; ct, st = math.cos(tl), math.sin(tl)
    def P(X, Y, Z):
        X, Y, Z = np.broadcast_arrays(np.asarray(X, float), np.asarray(Y, float), np.asarray(Z, float)); x = f * X / Z; y = f * (1.5 - Y) / Z
        return np.stack([vx + x * ct - y * st, vy + x * st + y * ct], -1)
    focus_lines(a.c, vx, vy, h * .20, 1700, 170, 6, K, seed, clump=1.2, ry=.9, a0=math.pi * 1.02, a1=math.pi * 1.98)
    # far haze block
    for X0, X1, Y1, Z in ((-30, -12, 26, 80), (-14, -5, 16, 90), (4, 15, 20, 95), (13, 34, 30, 85), (-4, 4, 11, 110)):
        top = [(X0 + (X1 - X0) * t, Y1 * r.uniform(.72, 1.04)) for t in np.linspace(0, 1, 7)]; q = P([X0] + [t[0] for t in top] + [X1], [0] + [t[1] for t in top] + [0], Z); a.shape(np.vstack([q, q[:1]]), 2.5); a.hatch(q, 1.3, 6, 1.4, fade=.4)
    def building(sd, X, Z0, Z1, Y, deep, dk, seedb):
        rr = np.random.default_rng(seedb); n = 9; zs = np.linspace(Z0, Z1, n); ys = Y * (1 - np.abs(rr.normal(0, .12, n))); ys[rr.integers(2, n - 1)] *= rr.uniform(.45, .7)
        side = P([X] * n, ys, zs); wall = np.vstack([P(X, 0, Z0)[None], side, P(X, 0, Z1)[None]])
        Xo = X + sd * deep; nf = 7; xs = np.linspace(X, Xo, nf); yf = np.r_[ys[0], Y * (1 - np.abs(rr.normal(0, .14, nf - 1)))]
        front = np.vstack([P(X, 0, Z0)[None], P(xs, yf, [Z0] * nf), P(Xo, 0, Z0)[None]])
        a.shape(np.vstack([front, front[:1]]), 5, K if dk else P_); a.shape(np.vstack([wall, wall[:1]]), 5, P_ if dk else K)
        lit, drk = (a.black, a.paper)
        fl = np.arange(1.4, Y * .98, 2.6)
        for i in range(n - 1):
            for yy in fl:
                z0, z1 = zs[i] + (zs[i + 1] - zs[i]) * .14, zs[i] + (zs[i + 1] - zs[i]) * .84; yt = min(yy + 1.6, min(ys[i], ys[i + 1]) - .35)
                if yt - yy < .5 or rr.random() < .06: continue
                q = P([X, X, X, X], [yy, yy, yt, yt], [z0, z1, z1, z0]); q = q + rr.normal(0, 1.0, q.shape)
                if dk:
                    a.black(q)
                    if rr.random() < .5: c = q.mean(0); a.white([q[0] * .8 + c * .2, q[2] * .7 + c * .3], rr.uniform(1.5, 4), taper=(.2, .6))
                else: a.line(np.vstack([q, q[:1]]), 1.6, col=P_, taper=(0, 0))
        for yy in fl[1:]: a.line(P([X] * n, np.minimum(yy - .5, ys), zs), 2.2, col=K if dk else P_)
        if dk:
            for k in range(3):
                i = rr.integers(1, nf - 1); x0 = xs[i]
                for yy in fl:
                    if yy + 1.6 < yf[i] - .5 and rr.random() < .8: q = P([x0, x0 + sd * deep * .10, x0 + sd * deep * .10, x0], [yy, yy, yy + 1.5, yy + 1.5], Z0); a.line(np.vstack([q, q[:1]]), 2.2, col=P_, taper=(0, 0))
        else: a.hatch(front, 1.2, 8, 2.2, fade=.3)
        # blown-out hole with rebar
        k = rr.integers(2, n - 2); c = P(X, Y * rr.uniform(.35, .6), zs[k]); R = f * 2.0 / zs[k]; th = np.sort(rr.uniform(0, 6.28, 12)); q = np.c_[c[0] + np.cos(th) * R * .6 * rr.uniform(.6, 1.3, 12), c[1] + np.sin(th) * R * rr.uniform(.6, 1.3, 12)]
        a.shape(np.vstack([q, q[:1]]), 4, K if dk else P_)
        for _ in range(7): p = q[rr.integers(12)]; crack(a, p[0], p[1], math.atan2(*(p - c)[::-1]), R * rr.uniform(.5, 1.6), 4, 1, .4) if dk else a.line([p, p + (p - c) * rr.uniform(.2, .6)], 2.5, col=P_)
        for _ in range(5): p = c + rr.normal(0, R * .4, 2); a.line(spl([p, p + [rr.normal(0, R * .3), -R * .3], p + [rr.normal(0, R * .5), rr.uniform(-.8, .2) * R]], n=6), 2.2, col=P_ if dk else K, taper=(.1, .4))
    building(1, 9.5, 30, 60, 22, 14, True, 11); building(-1, -8.5, 34, 66, 18, 12, False, 12); building(-1, -9.5, 15, 30, 19, 12, True, 13); building(1, 8.5, 11, 25, 23, 14, False, 14)
    building(-1, -8.2, 6.0, 12.5, 11, 10, False, 15)
    # ground
    gd = np.vstack([P(-60, 0, 120)[None], P(60, 0, 120)[None], P(8.2, 0, 11)[None], P(16, 0, 2.2)[None], P(-16, 0, 2.2)[None], P(-8.2, 0, 6)[None]])
    a.paper(np.vstack([P([-8.5, 8.8], 0, [70, 70]), P([8.5, 8], 0, [11, 2.]), P([-8, -8.5], 0, [2., 6])]))
    for X in (-6.8, 6.8): a.line(P([X] * 12, 0, np.linspace(2.2, 60, 12)), 4)
    for X, z0, z1 in ((0, 3.0, 4.4), (0, 5.6, 7.6), (0, 9.2, 12), (0, 15, 19), (0, 24, 30)): q = P([X - .13, X + .13, X + .13, X - .13], 0, [z0, z0, z1, z1]); a.shape(np.vstack([q, q[:1]]), 2.2); a.hatch(q, .4, 5, 1.2)
    c0 = P(.6, 0, 7.2)
    for k in range(17):
        t = k / 17 * 6.28 + r.normal(0, .12); X, Z = .6 + math.cos(t) * .9, 7.2 + math.sin(t) * 1.3; pts = [(X, Z)]
        for _ in range(9): t2 = t + r.normal(0, .40); X += math.cos(t2) * r.uniform(.4, .9); Z += math.sin(t2) * r.uniform(.5, 1.3); pts.append((X, max(2.3, Z)))
        q = P([p[0] for p in pts], 0, [p[1] for p in pts]); wd = np.linspace(1, .1, len(q)) * 9 * np.clip(7 / np.array([p[1] for p in pts]), .3, 2); T = np.gradient(q, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-6; N_ = np.c_[-T[:, 1], T[:, 0]] * (wd * r.uniform(.6, 1.3, len(q)))[:, None] * .5
        a.c.fill(np.vstack([q + N_, (q - N_)[::-1]]), K)
        for j in range(2, len(q) - 1): 
            if r.random() < .6: crack(a, *q[j], r.uniform(0, 6.28), r.uniform(20, 90) * np.clip(7 / pts[j][1], .4, 2), 5, 1, .4)
    def slab(X, Z, sx, sz, yaw, lift, thick=.32):
        c, s = math.cos(yaw), math.sin(yaw); R_ = np.array([(-sx, -sz), (sx, -sz), (sx, sz), (-sx, sz)]) * r.uniform(.8, 1.15, (4, 2)); Q = np.c_[X + R_[:, 0] * c - R_[:, 1] * s, Z + R_[:, 0] * s + R_[:, 1] * c]; hi = np.array([0, 0, lift, lift]) if r.random() < .5 else np.array([lift, 0, 0, lift])
        top = P(Q[:, 0], hi + .05, Q[:, 1]); bot = P(Q[:, 0], hi + .05 - thick, Q[:, 1]); g0 = P(Q[:, 0], 0, Q[:, 1])
        a.black(g0 + [10, 10])
        for i in range(4):
            j = (i + 1) % 4; a.shape(np.vstack([top[i], top[j], bot[j], bot[i], top[i]]), 3.5, K if i in (0, 1) else P_)
        a.shape(np.vstack([top, top[:1]]), 4.5); e0, e1 = top[0], top[1]
        a.hatch(np.vstack([top[1], top[2], (top[2] + top[3]) / 2, (top[0] + top[1]) / 2]), r.uniform(-.3, .3), 9, 2., fade=.75, length=60)
        for _ in range(12): u, v = r.random(2); p = (top[0] * (1 - u) + top[1] * u) * (1 - v) + (top[3] * (1 - u) + top[2] * u) * v; a.c.fill(blob(p[0], p[1], r.uniform(1, 3.4), r, 0, 1.8, 0, 8), K)
        for _ in range(3): u = r.random(); p = top[0] * (1 - u) + top[1] * u; crack(a, p[0], p[1], math.atan2(*(top.mean(0) - p)[::-1]), np.hypot(*(top.mean(0) - p)) * 1.2, 4, 1, .3)
        for _ in range(7): u = r.random(); p = e0 * (1 - u) + e1 * u; a.white([p + [0, 3], p + [r.normal(0, 3), r.uniform(8, 24)]], r.uniform(1.5, 4), taper=(.1, .8))
    for q in ((-3.6, 9.5, 1.3, 1.1, .4, 1.5), (3.4, 10.5, 1.2, 1.2, -.3, 1.3), (4.9, 6.2, 1.2, .9, -.6, 1.2), (-2.9, 5.4, 1.1, .8, .5, 1.0), (2.0, 4.2, .8, .6, -.3, .55), (-4.9, 6.8, .9, .9, .2, .9)): slab(*q)
    # fallen poles and wires
    def pole(p0, p1, wd):
        A, B = P(*p0), P(*p1); d = B - A; d /= np.hypot(*d); n_ = np.array([-d[1], d[0]]); w0, w1 = wd * f / p0[2], wd * f / p1[2]
        q = np.vstack([A + n_ * w0, B + n_ * w1, B - n_ * w1, A - n_ * w0]); a.shape(np.vstack([q, q[:1]]), 5); a.black(np.vstack([A + n_ * w0 * .05, B + n_ * w1 * .05, B - n_ * w1, A - n_ * w0]))
        for k, t in ((0, .78), (1, .9)):
            c = A + (B - A) * t; L = 1.2 * f / (p0[2] * (1 - t) + p1[2] * t); u = n_ * .95 + d * .25; a.line([c - u * L, c + u * L], 10 * 6 / p1[2], taper=(.02, .02))
            for v in (-.8, -.4, .4, .8): a.c.fill(blob(*(c + u * L * v - d * 4), 6.5 * 6 / p1[2] + 2, r, 0, 1.2, 0, 10), K)
        return A + (B - A) * .9
    t1 = pole((-6.9, 0, 10.5), (-3.0, 6.6, 8.6), .16); t2 = pole((6.9, 0, 16), (5.0, 8.0, 17), .18); t3 = pole((7.0, 0, 34), (7.6, 8.2, 34), .18)
    for A, B, sag in ((t1, t2, 120), (t1 + [-14, 18], t2 + [12, 16], 170), (t2, t3, 30), (t2 + [20, 4], t3 + [8, 4], 44)): a.line(spl([A, (A + B) / 2 + [0, sag], B], n=20), 3.0, taper=(.02, .02))
    for A, L_ in ((t1 + [20, 0], 260), (t1 + [-30, 6], 180)): a.line(spl([A, A + [r.normal(0, 30), L_ * .5], A + [r.normal(0, 60), L_ * .9], A + [r.normal(40, 30), L_]], n=12), 3.2, taper=(.02, .3))
    # rubble
    for k in range(700): X, Z = r.uniform(-8, 8), 2.4 + 22 * r.random() ** 2; q = P(X, 0, Z); a.line([q, q + [r.normal(0, 60 / Z), r.normal(0, 6 / Z)]], r.uniform(.8, 2.4) * np.clip(8 / Z, .5, 1.6), taper=(.3, .6))
    for k in range(190):
        sd = r.choice([-1, 1]); X = sd * r.uniform(4.2, 8.2) if r.random() < .75 else r.uniform(-5, 5); Z = r.uniform(3.2, 24); q = P(X, r.uniform(0, .25), Z); rock(a, q[0], q[1], r.uniform(.07, .30) * f / Z * (1.5 if r.random() < .1 else 1), r.uniform(0, 6), np.clip(18 / Z, 1.4, 4.5))
    for x, y, R in ((120, 850, 84), (250, 880, 50), (1290, 840, 92), (1170, 882, 46), (40, 770, 52)): rock(a, x, y, R, r.uniform(0, 6), 6)
    splatter(a.c, *c0, 260, 70, 6, K, seed)
    return a.grey()


# ====================================================================== the page
def _fit(src, quad):
    """source rectangle (cover fit, centred) whose aspect matches the quad"""
    h, w = src.shape[:2]; q = np.asarray(quad, float); qw = (np.hypot(*(q[1] - q[0])) + np.hypot(*(q[2] - q[3]))) / 2; qh = (np.hypot(*(q[3] - q[0])) + np.hypot(*(q[2] - q[1]))) / 2
    a = qw / qh; sw, sh = (w, w / a) if w / h < a else (h * a, h)
    return (w - sw) / 2, (h - sh) / 2, sw, sh


class Sheet:
    """a manga page (grey, 1 = paper).  Panels are drawings warped into quads (tl, tr, br, bl), ruled by hand, over gutters."""

    def __init__(self, h=H, w=W, paper_=1.0, seed=0):
        self.h, self.w = h, w; self.g = np.full((h, w), paper_, f32); self.seed = seed; self.n = 0

    def panel(self, art, quad, border=7., at=(0., 0.), zoom=1.0, bleed=()):
        """art = grey drawing; at = pan in fractions of the crop; zoom > 1 crops in; bleed = border indices (0 top .. 3 left) left open"""
        q = np.asarray(quad, np.float32); x, y, sw, sh = _fit(art, q); cx, cy = x + sw / 2 + at[0] * sw, y + sh / 2 + at[1] * sh; sw, sh = sw / zoom, sh / zoom
        ah, aw = art.shape[:2]; cx = np.clip(cx, sw / 2, aw - sw / 2); cy = np.clip(cy, sh / 2, ah - sh / 2)
        S = np.float32([[cx - sw / 2, cy - sh / 2], [cx + sw / 2, cy - sh / 2], [cx + sw / 2, cy + sh / 2], [cx - sw / 2, cy + sh / 2]])
        ss = 2; M = cv2.getPerspectiveTransform(S, q * ss); x0, y0 = np.floor(q.min(0)).astype(int) - 12; x1, y1 = np.ceil(q.max(0)).astype(int) + 12
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(self.w, x1), min(self.h, y1)
        if x1 <= x0 or y1 <= y0: return self
        T = np.float64([[1, 0, -x0 * ss], [0, 1, -y0 * ss], [0, 0, 1]]); wr = cv2.warpPerspective(art, T @ M, ((x1 - x0) * ss, (y1 - y0) * ss), flags=cv2.INTER_AREA if sw > q[:, 0].max() - q[:, 0].min() else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        wr = cv2.resize(wr, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
        c = Cel(x1 - x0, y1 - y0, x0, y0); c.fill(q.astype(float), '#ffffff'); m = c.rgba()[..., 3]
        self.g[y0:y1, x0:x1] = self.g[y0:y1, x0:x1] * (1 - m) + wr * m
        c = Cel(x1 - x0, y1 - y0, x0, y0); r = np.random.default_rng(self.seed * 31 + self.n); self.n += 1
        for i in range(4):
            if i in bleed: continue
            a, b = q[i].astype(float), q[(i + 1) % 4].astype(float); d = (b - a) / np.hypot(*(b - a)); o = r.normal(0, .8, (2, 2))
            c.line(np.array([a - d * r.uniform(0, 5) + o[0], (a + b) / 2 + r.normal(0, .9, 2), b + d * r.uniform(0, 5) + o[1]]), border, K, taper=(.01, .01), press=.10, seed=int(r.integers(1 << 30)), tmin=.8)
        l = c.rgba(); self.g[y0:y1, x0:x1] *= 1 - l[..., 3]
        return self

    def ink(self, cel):
        """composite a Cel (lettering, splatter) on the page"""
        l = cel.layer(self.h, self.w); self.g = self.g * (1 - l[..., 3]) + lum(l[..., :3]); return self

    def rgb(self, paper_='#d9d6d0', ink_=BLACK, grain=.35, seed=0):
        """printed: newsprint paper, slightly uneven black"""
        t = np.tile(tooth(2304, 2304), (1, 1))[:self.h, :self.w]; n = fbm(self.h, self.w, (300, 90), (1, .5), seed)
        k = np.clip((1 - self.g) * (1 - grain * .28 * smooth(.62, 1., t)), 0, 1)[..., None]
        return rgb(paper_) * (1 + .012 * n[..., None]) * (1 - k) + rgb(ink_) * k


_art = {}
VER = 'v2'          # bump when a drawing changes


def art(name):
    """a ready drawing by name (explosion, runner, fist, street, eye, fingers), kept in memory"""
    if name not in _art:
        import os
        d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache'); p = os.path.join(d, f'manga_{name}_{VER}.npy')
        if os.path.exists(p): _art[name] = np.load(p).astype(f32)
        else:
            _art[name] = globals()[name]()
            os.makedirs(d, exist_ok=True); q = p + f'.{os.getpid()}.npy'; np.save(q, _art[name].astype(np.float16)); os.replace(q, p)
    return _art[name]


def page(t=99, seed=0):
    """the six drawings as one collage (b4 may use it or rearrange): panels appear in order, t = number shown"""
    S = Sheet(seed=seed)
    L = [('street', [(-20, -20), (760, -20), (700, 400), (-20, 452)], dict(at=(0, -.05))), ('explosion', [(782, -20), (1940, -20), (1940, 470), (722, 392)], dict(at=(0, .04))),
         ('runner', [(-20, 476), (470, 438), (420, 1100), (-20, 1100)], dict(at=(0, -.10), zoom=1.12)), ('eye', [(492, 436), (1352, 452), (1312, 690), (474, 706)], dict(zoom=1.12, at=(0, .02))),
         ('fist', [(472, 732), (1016, 722), (984, 1100), (446, 1100)], dict(at=(0, -.15), zoom=1.1)), ('fingers', [(1376, 454), (1940, 492), (1940, 1100), (1300, 1100)], dict(at=(0, -.12), zoom=1.05))]
    L.insert(5, ('street', [(1040, 722), (1308, 716), (1276, 1100), (1010, 1100)], dict(at=(.34, -.1), zoom=1.5)))
    for i, (n, q, k) in enumerate(L):
        if i >= t: break
        S.panel(art(n), q, **k)
    return S


if __name__ == '__main__':
    pass
