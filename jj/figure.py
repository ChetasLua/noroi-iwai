"""Full-body figures: a posed 3-D build (fig_pose, fig_body, fig_sdf) re-drawn as a hand-inked cel for wide and mid shots (80..1000 px tall).

    F = Fig('yuji').pose('run_contact', hR=(.2, .8, 0)).place(0, 6, 0, heading=math.pi)     # heading 0 looks along +Y, pi faces a camera at -Y, -pi/2 looks +X
    L = F.draw(cam, f, light=(-.5,.75,.4), rim=dict(dir=(.9,.2,-.4), col='#ff9c48', thr=.5, amt=.8), pal=dict(mul='#cdd2f0'), expr=dict(mood='shout'))
    img = comp.over(img, L)                                  # premultiplied (H, W, 4).  draw(..., layer=False) -> (roi, box) for crowds
characters  yuji megumi nobara gojo nanami todo sukuna crowd crowdf curse.   Fig(char, spec=dict(cloth='wine', haircol=[4 hex]), scale=, body=dict(H=..)) varies them
light/kick  direction TO the light in camera space (x right, y up, z toward us).  light = key (one angular shadow), kick = back light that makes the pale shapes on dark cloth.  key=.1 terminator bias
rim         dict(dir, col, thr, amt) planes of coloured light from the form itself.   pal: mul / shmul / litmul tint lit, shadow, highlight; add; ink; mats={'navy': (light, base, shadow, deep)}
expr        dict(mood= calm firm smile grin shout rage sad, look=(x, y), open=, tears=True, bare=True for Gojo without the blindfold)
head_fn     head_fn(cel, H, info) draws on a Cel covering the roi (e.g. lambda cel, H, i: Yuji.head(cel, H, 'rage', 'fire', i['f'])), or head_fn(H, info) -> layer.
            H = head3.Head posed from the skeleton (any angle), info = x y s(px per head unit) yaw pitch roll R roi hair f light expr
animation   F.anim(lambda f: dict(pose=seq(f, KEYS), x=, y=, z=, heading=)); F.at(f).draw(cam, f).  Hood, hem, sleeves and hair lag behind the motion by themselves; F.blow(vx, vy, vz) wind
            seq(f, [(0,'guard','in'), (6,'punch_wind','out'), (9,'punch_hit','hold'), ...]); sprint(u) walk(u) cycles (u in strides); mix_pose, add_pose, mirror, P(base, **changes)
smears      F.smear(cam, f0, f1, 'armR') swept limb, draw it under the new pose; after_limb(...) ink ghosts; tails(layer, (dx, dy)) dry-brush speed tails under a figure
after draw  F.info: hands wrists elbows feet knees shoulders pelvis chest head (px), hands3 (world), box, px, skel;  F.mask('handR', 'farmR') part mask;  F.depth() z buffer
poses       see fig_pose.POSES (and P() for the pose language)
"""
import numpy as np, cv2, math, zlib
from . import W, H
from .paint import rgb, f32
from .ink import Cel, spl, resample, over_roi, paste
from . import kit
from .fig_pose import *
from .fig_pose import POSES, Skel, CHAR_BODY, P, mix_pose
from .fig_body import Body, MATS, SPEC
from .fig_sdf import wrap
from . import fig_head
from .fig_smear import limb_smear, after_limb, tails

INK = '#17131a'
DARK = {'navy', 'navyb', 'gblack', 'black', 'hakama', 'sock', 'slate', 'tabi', 'obi', 'band', 'tights', 'hairk', 'redshoe', 'red'}


OUTER = {'torso', 'jacket', 'skirt', 'belt', 'collar', 'hood1', 'hood2', 'hoodb', 'scarf1', 'scarf2', 'sleeveL', 'sleeveR'}
INNER = {'thigh', 'shin', 'pelvis', 'torso', 'uarm', 'neck', 'sock', 'skirt'}


# ------------------------------------------------------------------ vector helpers
def _smooth(P, k, closed):
    k = int(k) | 1
    if k < 3 or len(P) < k + 2: return P
    w = np.hanning(k + 2)[1:-1]; w /= w.sum(); h = k // 2
    Q = np.vstack([P[-h:], P, P[:h]]) if closed else np.vstack([2 * P[0] - P[h:0:-1], P, 2 * P[-1] - P[-2:-h - 2:-1]])
    return np.stack([np.convolve(Q[:, i], w, 'valid') for i in (0, 1)], -1)


def _dp(P, eps, closed=False):
    """Douglas-Peucker indices"""
    n = len(P)
    if n < 3: return np.arange(n)
    q = cv2.approxPolyDP(np.round(P * 16).astype(np.int32)[:, None], eps * 16, closed)[:, 0]
    # map back to indices
    d = {}
    R = np.round(P * 16).astype(np.int32)
    idx = []
    j = 0
    key = {}
    for i, p in enumerate(map(tuple, R)): key.setdefault(p, i)
    for p in map(tuple, q):
        if p in key: idx.append(key[p])
    return np.array(sorted(set(idx)))


def draw_contour(P, eps, closed=True, sharp=.75, n=5):
    """straights against curves: simplify, keep corners sharper than `sharp` rad, bend the rest"""
    if len(P) < 8: return P
    idx = _dp(P, eps, closed)
    if len(idx) < 3: return P[[0, len(P) // 2, -1]] if not closed else P
    V = P[idx]
    m = len(V)
    a = np.roll(V, 1, 0) if closed else np.vstack([V[:1], V[:-1]]); b = np.roll(V, -1, 0) if closed else np.vstack([V[1:], V[-1:]])
    t1 = np.arctan2(*(V - a).T[::-1]); t2 = np.arctan2(*(b - V).T[::-1]); turn = np.abs((t2 - t1 + math.pi) % (2 * math.pi) - math.pi)
    c = [i for i in range(m) if turn[i] > sharp and (closed or 0 < i < m - 1)]
    if closed:
        if len(c) == 1: c.append((c[0] + m // 2) % m)
        return spl(V, True, corners=c, n=n)
    return spl(V, False, corners=c, n=n)


def clean_closed(P, eps, sharp=.5, n=6):
    """closed raw contour -> (clean closed polyline, dp index list, n): straights against curves, corners kept.  raw index r maps to clean index via clean_index()"""
    idx = _dp(P, eps, True)
    if len(idx) < 4:
        idx = np.unique(np.linspace(0, len(P) - 1, 5).astype(int)[:-1])
    V = P[idx]; d1 = np.roll(V, -1, 0) - V; L1 = np.hypot(*d1.T) + 1e-6; u1 = d1 / L1[:, None]; u0 = np.roll(u1, 1, 0); L0 = np.roll(L1, 1)
    turn = np.abs((np.arctan2(u1[:, 1], u1[:, 0]) - np.arctan2(u0[:, 1], u0[:, 0]) + math.pi) % (2 * math.pi) - math.pi)
    T = u0 * L1[:, None] + u1 * L0[:, None]; T /= np.hypot(*T.T)[:, None] + 1e-6
    c = (turn > sharp)[:, None]
    to = np.where(c, u1, T); ti = np.where(c, u0, T)
    t = (np.arange(n) / n)[None, :, None]; h00 = 2 * t ** 3 - 3 * t ** 2 + 1; h10 = t ** 3 - 2 * t ** 2 + t; h01 = -2 * t ** 3 + 3 * t ** 2; h11 = t ** 3 - t ** 2
    d = L1[:, None, None] * .92; V1 = np.roll(V, -1, 0)
    Q = h00 * V[:, None] + h10 * to[:, None] * d + h01 * V1[:, None] + h11 * np.roll(ti, -1, 0)[:, None] * d
    return Q.reshape(-1, 2), idx, n


def clean_index(r, idx, n, N):
    """position on the clean contour of raw index r (0..N)"""
    k = np.searchsorted(idx, r, side='right') - 1
    if k < 0: a = idx[-1] - N; b = idx[0]; k = len(idx) - 1; return (k + (r - a) / max(b - a, 1)) * n
    a = idx[k]; b = idx[k + 1] if k + 1 < len(idx) else idx[0] + N
    return (k + (r - a) / max(b - a, 1)) * n


def mask_polys(m, g, box, eps=.6, up=2, min_area=10., smooth=5, sharp=.75, raw=False, sig=.6):
    """uint8/bool mask at G resolution -> list of screen polylines"""
    ys, xs = np.nonzero(m)
    if len(ys) < 3: return []
    y0, y1, x0, x1 = max(0, ys.min() - 3), ys.max() + 4, max(0, xs.min() - 3), xs.max() + 4
    s = m[y0:y1, x0:x1].astype(f32)
    s = cv2.copyMakeBorder(s, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    if sig > 0: s = cv2.GaussianBlur(s, (0, 0), sig)
    s = cv2.resize(s, None, fx=up, fy=up, interpolation=cv2.INTER_LINEAR)
    cs, _ = cv2.findContours((s > .5).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        if len(c) < 8: continue
        c = c[:, 0].astype(float)
        if abs(cv2.contourArea(c.astype(f32))) < min_area * up * up: continue
        c = _smooth(c, smooth * up, True)
        c = ((c + .5) / up - 2 + [x0, y0]) / g + [box[0], box[1]]
        out.append(c if raw else draw_contour(c, eps, True, sharp))
    return out


def _hash(*a):
    return zlib.crc32(repr(a).encode()) & 0x7fffffff


def _mixc(a, b, t):
    a = rgb(a) if isinstance(a, str) else np.asarray(a, f32); b = rgb(b) if isinstance(b, str) else np.asarray(b, f32)
    return a * (1 - t) + b * t


class Pal:
    """colour script: mats override MATS; mul / shmul tint lit and shadow tones; ink colour; black = all-ink mode"""

    def __init__(self, d=None):
        d = dict(d or {}); self.d = d; self.mats = dict(MATS); self.mats.update(d.get('mats', {})); self.ink = d.get('ink', INK)
        v = lambda k, e: rgb(d[k]) if isinstance(d.get(k), str) else np.asarray(d.get(k, e), f32)
        self.mul = v('mul', (1, 1, 1)); self.shmul = v('shmul', (1, 1, 1)); self.add = v('add', (0, 0, 0)); self.litmul = v('litmul', self.mul)
        self.cache = {}

    def __call__(self, mat):
        if mat not in self.cache:
            m = mat
            if m == 'slate_in': m = 'slate'
            c = self.mats.get(m, MATS['grey']); c = [rgb(q) for q in c]
            if mat == 'slate_in': c = [c[1] * 1.25, c[1] * 1.1, c[2], c[3]]
            self.cache[mat] = (c[0] * self.litmul + self.add, c[1] * self.mul + self.add, c[2] * self.shmul + self.add * .6, c[3] * self.shmul + self.add * .4)
        return self.cache[mat]


class Fig:
    def __init__(self, char, spec=None, seed=0, scale=1.0, body=None):
        self.char = char; self.spec = spec or {}; self.seed = seed; self.scale = scale; self.body = dict(CHAR_BODY[char]); self.body.update(body or {})
        self._pose = P(); self.x = self.y = self.z = 0.; self.heading = 0.; self.fn = None; self.wind = (0., 0., 0.); self.info = {}; self._fx = {}; self._cache = {}

    # ---- state
    def pose(self, p=None, **kw):
        q = POSES[p] if isinstance(p, str) else (p if p is not None else P())
        if kw: q = dict(q); q.update(kw)
        self._pose = q; return self

    def place(self, x=0., y=0., z=0., heading=0.):
        self.x, self.y, self.z, self.heading = x, y, z, heading; return self

    def blow(self, vx=0., vy=0., vz=0.):
        """steady wind (world, 1 = strong breeze)"""
        self.wind = (vx, vy, vz); return self

    def anim(self, fn):
        self.fn = fn; return self

    def state(self, f):
        if self.fn is None: return dict(pose=self._pose, x=self.x, y=self.y, z=self.z, heading=self.heading)
        d = dict(x=self.x, y=self.y, z=self.z, heading=self.heading); d.update(self.fn(f))
        if isinstance(d['pose'], str): d['pose'] = POSES[d['pose']]
        return d

    def at(self, f):
        d = self.state(f); self._pose = d['pose']; self.x, self.y, self.z, self.heading = d['x'], d['y'], d['z'], d['heading']; return self

    def skel(self, f=None):
        d = self.state(f) if f is not None else dict(pose=self._pose, x=self.x, y=self.y, z=self.z, heading=self.heading)
        return Skel(self.body, d['pose'], d['x'], d['y'], d['z'], d['heading'], self.scale)

    # ---- secondary motion: where the anchors were a few frames ago (pure function of f)
    def _lag(self, f, key, frames=3):
        if self.fn is None: return np.zeros(3)
        out = np.zeros(3); wt = 0
        p0 = key(self.skel(f))
        for k in range(1, frames + 1):
            w = .62 ** k; out += w * (key(self.skel(f - k)) - p0) / k; wt += w
        return out / wt

    def make_fx(self, f):
        fx = dict(self._fx); k = self.scale; wnd = np.asarray(self.wind, float); r = math.sin(f * .55 + self.seed) * .5 + math.sin(f * .9 + 1.7 * self.seed) * .3
        gust = wnd * (1 + .35 * r)
        cl = lambda v, m: v * min(1., m / (np.linalg.norm(v) + 1e-9))
        hem = cl(self._lag(f, lambda s: s.pel) * 1.2, .09 * k) + gust * .04 * k
        fx.setdefault('hem', hem); fx.setdefault('hood', cl(self._lag(f, lambda s: s.neck) * .5, .05 * k) + gust * .02 * k)
        for s in 'LR': fx.setdefault('sleeve' + s, cl(self._lag(f, lambda q, s=s: q.wr[s]) * 9., 1.4) + gust * .9)
        fx.setdefault('hair', cl(self._lag(f, lambda s: s.head) * 4.5, .8) + gust * .5)
        fx.setdefault('scarf', cl(self._lag(f, lambda s: s.neck) * 6., 1.2) + gust * .8)
        fx.setdefault('down', np.array([0, 0, -1.]) + gust * .35)
        fx['f'] = f
        return fx

    def smear(self, cam, f0, f1, limb='armR', **k):
        """swept-limb smear between two frames of the animation (or two state dicts): put it under draw() of the second"""
        a = self.state(f0) if not isinstance(f0, dict) else f0; b = self.state(f1) if not isinstance(f1, dict) else f1
        return limb_smear(self, cam, a, b, limb, **k)

    # ---- draw
    def draw(self, cam, f=0, light=(-.55, .75, .40), pal=None, head_fn=None, expr=None, weight=1.0, rim=None, kick=(-.80, .50, -.35), key=.10, lod=None, head=True, seed=None, cache=True, layer=True, ink_only=False):
        """light / kick: direction TO the light in camera space (x right, y up, z toward the viewer).  kick = back light that makes the pale
        shapes on dark cloth.  key = terminator bias (higher = more shadow).  rim = dict(col, dx, dy, amt)."""
        pal = pal if isinstance(pal, Pal) else Pal(pal)
        sk = self.skel(); self.sk = sk
        fx = self.make_fx(f)
        px = float(cam.scale(sk.pel)) * sk.H
        if not np.isfinite(px) or cam.cam(sk.pel)[2] < .1: px = 900.
        B = Body(self.char, sk, fx, px if lod is None else (100, 250, 700)[lod], self.spec, self.seed); self.B = B
        self._last = None
        D = _Draw(self, cam, B, px, pal, light, kick, key, weight, expr, head_fn if head else None, head, self.seed if seed is None else seed, fx, ink_only, rim)
        out = D.run(); self._last = D; self._cam = cam
        if out is None: return np.zeros((cam.h, cam.w, 4), f32)
        img, box = out
        if rim and 'dx' in rim: img = kit.rim(img, rim.get('dx', 2), rim.get('dy', 2), rim['col'], rim.get('amt', 1.), rim.get('soft', .5))
        if not layer: return img, box
        L = np.zeros((cam.h, cam.w, 4), f32); paste(L, img, box[0], box[1]); return L


    def mask(self, *labels):
        """full-frame float mask of drawn parts after draw(): label prefixes such as 'handR', 'farmR', 'uarm', 'foot', 'hood', 'torso', 'head'"""
        D, cam = self._last, self._cam; out = np.zeros((cam.h, cam.w, 1), f32)
        if D is None or not hasattr(D, 'lab'): return out[..., 0]
        ids = [i for i, n in enumerate(D.lname) if any(n.startswith(q) for q in labels)]; m = np.isin(D.lab, ids).astype(f32); x0, y0, x1, y1 = D.bodybox
        paste(out, cv2.resize(m, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)[..., None], x0, y0); return out[..., 0]

    def depth(self):
        """full-frame z_cam of the last draw (0 = empty), without the head's hair: for intersecting with sets"""
        D, cam = self._last, self._cam; out = np.zeros((cam.h, cam.w, 1), f32); x0, y0, x1, y1 = D.bodybox
        paste(out, cv2.resize(D.Z, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)[..., None], x0, y0); return out[..., 0]


class _Draw:
    def __init__(s, fig, cam, B, px, pal, light, kick, key, weight, expr, head_fn, head, seed, fx, ink_only, rim=None):
        s.fig, s.cam, s.B, s.px, s.pal, s.key, s.expr, s.head_fn, s.head, s.seed, s.fx = fig, cam, B, px, pal, key, expr, head_fn, head, seed, fx
        c2w = lambda v: cam.R.T @ (np.array([v[0], -v[1], -v[2]], float) / np.linalg.norm(v))
        s.rimd = rim if rim and 'dir' in rim else None; s.RD = c2w(rim['dir']) if s.rimd else None
        s.L = c2w(light); s.K = c2w(kick); s.lc = np.asarray(light, float); s.kc = np.asarray(kick, float)
        h = px
        s.wo = float(np.interp(h, [60, 120, 250, 450, 900, 1800], [1.2, 1.5, 2.0, 2.6, 3.4, 4.8])) * weight
        s.wi = s.wo * .74; s.wf = s.wo * .56; s.ink = pal.ink; s.ink_only = ink_only

    # ------------------------------------------------ buffers
    def run(s):
        cam, B, c = s.cam, s.B, s.B.c
        box = c.bbox(cam)
        x0, y0, x1, y1 = box
        if x1 - x0 < 4 or y1 - y0 < 4: return None
        big = max(x1 - x0, y1 - y0); s.g = g = min(1., 760. / big) if s.px > 130 else 1.5 if s.px < 90 else 1.25
        s.box = box; s.bodybox = box
        Z, Wp, N, T, A, C = s.buf = c.render(cam, box, g)
        s.Z, s.W, s.N, s.T, s.A, s.C = s.buf
        if (Wp >= 0).sum() < 6: return None
        s.hit = Wp >= 0
        h, w = Z.shape
        # labels / materials
        names = sorted(set(c.lab)); s.lname = names; lid = {n: i for i, n in enumerate(names)}
        pl = np.array([lid[l] for l in c.lab]); s.lab = np.where(s.hit, pl[np.maximum(Wp, 0)], -1).astype(np.int16)
        mn = []; mi = {}
        def mid(n):
            if n not in mi: mi[n] = len(mn); mn.append(n)
            return mi[n]
        pm = np.array([mid(m) for m in c.mat]); s.mat = np.where(s.hit, pm[np.maximum(Wp, 0)], -1).astype(np.int16)
        need = [i for i, r in enumerate(c.rule) if r]
        if need:
            jj, ii = np.mgrid[0:h, 0:w]
            for i in need:
                m = Wp == i
                if not m.any(): continue
                y_, x_ = np.nonzero(m)
                X = ((x_ + .5) / g + x0 - cam.cx) / cam.f; Y = ((y_ + .5) / g + y0 - cam.cy) / cam.f
                Q = cam.eye + (np.stack([X, Y, np.ones_like(X)], -1) @ cam.R) * Z[y_, x_][:, None]
                for fn, mt in c.rule[i]:
                    ok = fn(T[y_, x_], A[y_, x_], Q)
                    s.mat[y_[ok], x_[ok]] = mid(mt)
        s.mn = mn
        # heads are drawn separately: remember the proxy, then treat it as a hole that the neck/collar show through under the head layer
        s.hl = lid.get('head', -99)
        if not s.ink_only: s.light_pass()
        s.cel = Cel(x1 - x0, y1 - y0, x0, y0, ss=2)
        s.paint(); s.decals(); s.lines()
        if getattr(B, 'curseface', False) and not s.ink_only: s.curse_face()
        img = s.cel.rgba()
        s.anchors()
        if s.head and 'head' in lid: img = s.head_pass(img)
        return img, s.box

    def light_pass(s):
        g = s.g; N = s.N
        # calm the normals inside each label so the terminator is a drawn edge, not a lumpy one
        sig = max(.8, s.px * g * .0065)
        m = s.hit.astype(f32); w = cv2.GaussianBlur(m, (0, 0), sig) + 1e-4
        n = cv2.GaussianBlur(N, (0, 0), sig) / w[..., None]
        n = n * .75 + N * .25
        # do not smooth across occlusions
        dz = np.abs(cv2.GaussianBlur(s.Z * m, (0, 0), sig) / w - s.Z); k = np.clip(1 - dz / (.05 * s.B.H / 1.73), 0, 1)[..., None]
        n = n * k + N * (1 - k); n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6
        s.n = n
        nd = n @ s.L.astype(f32)
        sh = s.B.c.shadow(s.cam, s.box, g, s.buf, s.L, 1.1 * s.B.H / 1.73, 2 if s.px * g > 260 else 1)
        S = ((nd < s.key) | (sh > 0)) & s.hit
        ks = int(max(3, s.px * g * .012)) | 1
        S = cv2.medianBlur(S.astype(np.uint8), min(ks, 9)) > 0
        s.S = S & s.hit
        nk = n @ s.K.astype(f32)
        s.Hl = (nk > .52) & (~s.S) & s.hit
        s.Rm = ((n @ s.RD.astype(f32)) > s.rimd.get('thr', .42)) & s.hit & (s.C > .25) if s.rimd else None
        s.Dp = (s.C < .34) & s.hit & (s.lab != s.hl)

    # ------------------------------------------------ colour
    def polys(s, m, eps_k=1.0, min_area=8., **k):
        e = max(.45, s.px * .0011) * eps_k
        return mask_polys(m, s.g, s.box, eps=e, min_area=min_area, smooth=max(3, int(3.2 * s.g + 1.5)), **k)

    def eps_of(s, nm):
        return max(.5, s.px * .0011) if nm.startswith(('hand', 'foot', 'neck', 'sock', 'neckl')) else max(.6, s.px * .0021)

    def paint(s):
        cel, pal = s.cel, s.pal
        fig = s.hit.copy()
        if s.hl >= 0 and 'neck' in s.lname: s.mat[s.lab == s.hl] = s.mn.index(s.B.sp['skin']) if s.B.sp['skin'] in s.mn else 0; s.S = s.S | (s.lab == s.hl) if hasattr(s, 'S') else None
        s.figm = fig; g = s.g
        if s.ink_only:
            cel.fill(s.polys(fig), s.ink); return
        a = max(6., (s.px * g * .016) ** 2)
        Sp = [draw_contour(q, max(.9, s.px * .0034), True, .55, n=3) for q in mask_polys(s.S, g, s.box, raw=True, min_area=a, smooth=max(3, int(s.px * g * .008)))]
        Dp = [draw_contour(q, max(.8, s.px * .003), True, .5, n=3) for q in mask_polys(cv2.medianBlur(s.Dp.astype(np.uint8), 3), g, s.box, raw=True, min_area=a * .6, smooth=5)]
        Hp = [draw_contour(q, max(.8, s.px * .003), True, .5, n=3) for q in mask_polys(cv2.medianBlur(s.Hl.astype(np.uint8), 5), g, s.box, raw=True, min_area=a * .8, smooth=5)]
        s.Rp = [draw_contour(q, max(.8, s.px * .0026), True, .5, n=3) for q in mask_polys(cv2.medianBlur(s.Rm.astype(np.uint8), 5), g, s.box, raw=True, min_area=a * .5, smooth=5)] if s.rimd else []
        s.Sp = Sp; s.matpoly = {}; s.lpoly = {}
        # labels far to near; each is filled under its nearer neighbours so no seam opens, and its own edge is exactly the edge that will be inked
        ids = [i for i in range(len(s.lname)) if (s.lab == i).any()]
        zm = {i: float(np.median(s.Z[s.lab == i])) for i in ids}
        ids.sort(key=lambda i: -zm[i] - (100 if i == s.hl else 0))
        sm = max(3, int(3.0 * g + 1.5)); ks = np.ones((7, 7), np.uint8)
        for i in ids:
            nm = s.lname[i]; m = (s.lab == i)
            raw = mask_polys(m, g, s.box, raw=True, up=2, min_area=3., smooth=sm, sig=.6)
            cl = [clean_closed(c, s.eps_of(nm)) for c in raw if len(c) >= 10]; raw = [c for c in raw if len(c) >= 10]
            s.lpoly[i] = (raw, cl)
            if not cl: continue
            polys = [c[0] for c in cl]
            # underlap: grow into nearer labels only
            near = np.isin(s.lab, [q for q in ids if zm[q] < zm[i] and q != i]); grow = (cv2.dilate(m.astype(np.uint8), ks) > 0) & (near | m)
            und = s.polys(grow, min_area=2.) if grow.sum() > m.sum() else []
            mats, cnt = np.unique(s.mat[m], return_counts=True); main = mats[np.argmax(cnt)]
            nmn = s.mn[main]; lit, base, shd, deep = pal(nmn)
            same = [pal(nmn)]
            def flat(clip, col, ps):
                cel._path(ps); cel._src(col, 1.); cel.ctx.set_fill_rule(1); cel.ctx.fill()
            for ps in (und, polys):
                if not ps: continue
                flat(None, base, ps)
            s.tones(polys + und, nmn, Sp, Hp, Dp)
            for q in mats:
                if q == main: continue
                mm = m & (s.mat == q)
                if mm.sum() < 4: continue
                Pm = s.polys(cv2.dilate(mm.astype(np.uint8), np.ones((2, 2), np.uint8)) & m.astype(np.uint8), min_area=2., eps_k=.8)
                if not Pm: continue
                cel.ctx.save(); cel._path(polys); cel.ctx.set_fill_rule(0); cel.ctx.clip()
                flat(None, pal(s.mn[q])[1], Pm); s.tones(Pm, s.mn[q], Sp, Hp, Dp); cel.ctx.restore()
        for i, (raw, cl) in s.lpoly.items():
            mats, cnt = np.unique(s.mat[s.lab == i], return_counts=True)
            for q in mats: s.matpoly.setdefault(s.mn[q], []).extend([c[0] for c in cl])

    def tones(s, clip, nm, Sp, Hp, Dp):
        cel = s.cel; lit, base, shd, deep = s.pal(nm); dark = nm in DARK
        cel.ctx.save(); cel._path(clip); cel.ctx.set_fill_rule(0); cel.ctx.clip()
        if Hp and dark: cel.fill(Hp, lit, evenodd=True)
        elif Hp and s.px > 160: cel.fill(Hp, _mixc(base, lit, .6 if nm.startswith('skin') or nm in ('curse', 'cursed') else .35), evenodd=True)
        if Sp: cel.fill(Sp, shd, evenodd=True)
        if Dp and s.px > 110 and nm != 'inside': cel.fill(Dp, deep if dark else _mixc(shd, deep, .55), evenodd=True)
        if s.Rp and nm != 'inside': cel.fill(s.Rp, _mixc(_mixc(base, lit, .5), rgb(s.rimd['col']), s.rimd.get('amt', .85)), evenodd=True)
        cel.ctx.restore()

    # ------------------------------------------------ surface decals
    def vis(s, Q, prim, tol=.035):
        """screen points + visibility of world points lying on a prim"""
        p, z = s.cam.project(Q); g = s.g; h, w = s.Z.shape
        u = np.floor((p[:, 0] - s.box[0]) * g).astype(int); v = np.floor((p[:, 1] - s.box[1]) * g).astype(int)
        ok = (u >= 0) & (v >= 0) & (u < w) & (v < h) & (z > .06)
        o = np.zeros(len(p), bool); i = np.nonzero(ok)[0]
        if len(i):
            zb = s.Z[v[i], u[i]]; lb = s.lab[v[i], u[i]]
            grp = s.B.c.I[prim][2]; okl = s._glab(grp)
            o[i] = (np.abs(zb - z[i]) < tol * s.B.H / 1.73) & (lb >= 0) & okl[np.maximum(lb, 0)]
        return p, o

    def _glab(s, g):
        """labels that belong to clay group g"""
        if not hasattr(s, '_gl'): s._gl = {}
        if g not in s._gl:
            c = s.B.c; names = {c.lab[i] for i in range(len(c.lab)) if c.I[i][2] == g}; s._gl[g] = np.array([n in names for n in s.lname])
        return s._gl[g]

    def decals(s):
        B, cel, pal = s.B, s.cel, s.pal; lvl = 0 if s.px < 200 else 1 if s.px < 380 else 2
        if s.ink_only: return
        for k, d in enumerate(B.deco):
            if d.get('pri', 0) > lvl or d['kind'] not in ('shard', 'poly', 'cutsh'): continue
            t, p = d['t'], d['p']
            cl = np.vstack([np.c_[t, p], [[t[0], p[0]]]])
            Q = B.c.point(d['prim'], *resample2(cl, .04).T)
            xy, ok = s.vis(Q, d['prim'])
            if ok.mean() < (.5 if d['kind'] != 'poly' else .12): continue
            if d['kind'] == 'shard':
                r = np.random.default_rng(_hash(s.seed, k))
                V = xy[ok]
                if len(V) < 4: continue
                # thin leaf between its two long sides, angular
                mt = B.c.mat[d['prim']]; lit, base, shd, deep = pal(mt)
                cx_, cy_ = V.mean(0); u, v = s.ij(cx_, cy_)
                inS = s.S[v, u] if u is not None else False
                col = (_mixc(base, lit, .9) if not inS else _mixc(shd, lit, .55)) if mt in DARK else (shd if not inS else deep)
                if mt not in DARK: col = _mixc(base, shd, .85) if not inS else _mixc(shd, deep, .5)
                V = draw_contour(V, max(.5, s.px * .0016), True, .6, n=3)
                cel.ctx.save(); cel._path(s.matpoly.get(mt, [V])); cel.ctx.set_fill_rule(0); cel.ctx.clip(); cel.fill(V, col); cel.ctx.restore()
            elif d['kind'] == 'cutsh':
                mt = B.c.mat[d['prim']]; V = xy[ok]
                if len(V) < 4: continue
                cel.ctx.save(); cel._path(s.matpoly.get(mt, [V])); cel.ctx.set_fill_rule(0); cel.ctx.clip(); cel.fill(draw_contour(V, max(.5, s.px * .0016), True, .6, n=3), pal(mt)[2]); cel.ctx.restore()
            else:
                mt = B.c.mat[d['prim']]; lit, base, shd, deep = pal(d['mat'])
                V = xy[ok]
                if len(V) < 4: continue
                cel.ctx.save(); cel._path(s.matpoly.get(mt, [V])); cel.ctx.set_fill_rule(0); cel.ctx.clip()
                cel.fill(V, base); cel.clip(V); cel.fill(s.Sp, shd, evenodd=True); cel.unclip(); cel.ctx.restore()
                kit.outline(cel, V, s.wf, s.ink, seed=k)

    def curse_face(s):
        """our own cursed spirit: three mismatched pits for eyes over a grin that is too wide for it"""
        sk, cel, B = s.B.sk, s.cel, s.B; k = B.H / 2.35; up, fwd, rt = sk.Rc[:, 2], sk.Rc[:, 1], sk.Rc[:, 0]
        o = sk.neck; view = (s.cam.eye - o) / np.linalg.norm(s.cam.eye - o)
        if fwd @ view < -.1: return
        def pt(x, z, bulge=0.):                                      # drop a (x right, z up) face point on the clay along the facing direction
            base = o + rt * x * k + up * z * k; y = .30 - .35 * x * x - .10 * (z + .05) ** 2
            return base + fwd * (y + bulge) * k
        pr = lambda L, b=0.: s.cam.project(np.array([pt(x, z, b) for x, z in L]))[0]
        V, D, T, G = s.pal('void'), s.pal('curse'), s.pal('tooth'), s.pal('gum')
        r = np.random.default_rng(5)
        a = np.linspace(0, 2 * math.pi, 17)[:-1]
        for x, z, rx, rz in ((-.085, .085, .048, .060), (.080, .105, .036, .050), (.015, .170, .020, .026)):
            E = pr(np.c_[x + rx * np.cos(a) * (1 + .12 * np.sin(3 * a)), z + rz * np.sin(a) * (1 + .10 * np.cos(2 * a))]); E = spl(E, True, n=4)
            cel.fill(E, V[1]); kit.outline(cel, E, s.wi, s.ink, seed=int(x * 1000))
            q = pr(np.c_[x + .2 * rx + .16 * rx * np.cos(a), z - .1 * rz + .16 * rz * np.sin(a)]); cel.fill(q, T[0])
            for d in (-.6, 0, .6): cel.line(pr([(x + d * rx, z - rz * 1.02), (x + d * rx * 1.4, z - rz * 1.5), (x + d * rx * 1.3, z - rz * 2.0)]), s.wf, s.ink, taper=(.1, .7))
        n = 15; u = np.linspace(-1, 1, n)
        top = np.c_[u * .23, -.10 + .13 * u ** 2 + .012 * np.cos(u * 9)]; bot = np.c_[u * .23, -.17 - .115 * (1 - u ** 2) + .13 * u ** 2]
        M = pr(np.vstack([top, bot[::-1]]), .01); cel.fill(M, V[1]); cel.clip(M)
        cel.fill(pr([(-.10, -.30), (-.02, -.215), (.08, -.225), (.14, -.31)], .01), G[1])
        for row, (edge, sg) in enumerate(((top, -1), (bot, 1))):
            x = -.215
            while x < .205:
                w = r.uniform(.026, .048); ln = r.uniform(.045, .085) * (1.2 - abs(x) / .3); z0 = np.interp(x, edge[:, 0], edge[:, 1]); z1 = np.interp(x + w, edge[:, 0], edge[:, 1])
                Tq = pr([(x, z0 - sg * .01), (x + .12 * w, z0 + sg * ln * .7), (x + w * r.uniform(.35, .65), (z0 + z1) / 2 + sg * ln), (x + w * .9, z1 + sg * ln * .7), (x + w, z1 - sg * .01)], .012)
                cel.fill(Tq, T[1]); cel.fill(np.vstack([Tq[3:], Tq[2:3] * .6 + Tq[4:5] * .4]), T[2]); kit.outline(cel, Tq, s.wf, s.ink, seed=int(x * 999) + row); x += w * .98
        cel.unclip(); kit.outline(cel, M, s.wi * 1.15, s.ink, seed=3)
        for sg in (-1, 1):
            cel.line(pr([(sg * .225, .03), (sg * .27, .09), (sg * .265, .17)]), s.wi, s.ink, taper=(.1, .7)); cel.line(pr([(sg * .20, -.20), (sg * .25, -.14), (sg * .28, -.05)]), s.wf, s.ink, taper=(.3, .6)); cel.line(pr([(sg * .05, .02), (sg * .10, .035), (sg * .16, .02)]), s.wf, s.ink, taper=(.3, .6))

    def ij(s, x, y):
        u = int((x - s.box[0]) * s.g); v = int((y - s.box[1]) * s.g); h, w = s.Z.shape
        return (u, v) if 0 <= u < w and 0 <= v < h else (None, None)

    # ------------------------------------------------ ink
    def stroke(s, P, w, col=None, seed=0, taper=(.2, .2), split=True, press=.34):
        """one pen gesture, broken where the contour turns a corner"""
        P = np.asarray(P, float)
        if len(P) < 4: return
        Q = resample(P, 1.6)
        if len(Q) < 4: return
        col = s.ink if col is None else col
        if split:
            T = np.diff(Q, axis=0); a = np.unwrap(np.arctan2(T[:, 1], T[:, 0])); k = 3
            tn = np.abs(a[k:] - a[:-k]) if len(a) > k else np.zeros(0); cut = [0]; j = 0
            while j < len(tn):
                th_ = .9 if s.px > 330 else 1.25
                if tn[j] > th_:
                    e = j
                    while e + 1 < len(tn) and tn[e + 1] > th_: e += 1
                    i = int(j + np.argmax(tn[j:e + 1])) + 2
                    if i - cut[-1] > 4: cut.append(i)
                    j = e + k
                j += 1
            cut.append(len(Q) - 1)
        else: cut = [0, len(Q) - 1]
        for n_, (a_, b_) in enumerate(zip(cut[:-1], cut[1:])):
            g = Q[a_:b_ + 1]
            if len(g) < 4: continue
            L = len(g) * 1.6; c_ = .18 if s.px > 330 else .07; t0 = taper[0] if n_ == 0 else min(c_, 10 / L); t1 = taper[1] if n_ == len(cut) - 2 else min(c_, 10 / L)
            s.cel.line(g, w, col, taper=(t0, t1), press=press, seed=seed * 17 + n_, tremor=.22, tmin=.22 if s.px > 330 else .5)

    def lines(s):
        g, Z, lab = s.g, s.Z, s.lab; H_, W_ = Z.shape; B = s.B; sc = B.H / 1.73
        if s.ink_only: return
        mats = s.mat
        big = np.where(s.hit, Z, 1e6).astype(f32)
        up = 2
        hands = [n.startswith('hand') for n in s.lname]
        for li, nm in enumerate(s.lname):
            if li == s.hl: continue
            m = lab == li
            if m.sum() < 5: continue
            raws, cls_ = s.lpoly.get(li, ([], []))
            for ci, c in enumerate(raws):
                n = len(c)
                T = np.roll(c, -2, 0) - np.roll(c, 2, 0); T /= np.hypot(*T.T)[:, None] + 1e-9
                # findContours (y down) external contours run counter-clockwise in image space; resolve the side by test
                Nn = np.c_[T[:, 1], -T[:, 0]]
                def samp(off):
                    q = (c + Nn * off / g - [s.box[0], s.box[1]]) * g; u = np.clip(np.floor(q[:, 0]).astype(int), 0, W_ - 1); v = np.clip(np.floor(q[:, 1]).astype(int), 0, H_ - 1); return u, v
                u1, v1 = samp(1.6); u2, v2 = samp(-1.6)
                if (lab[v1, u1] == li).mean() > (lab[v2, u2] == li).mean(): (u1, v1), (u2, v2) = (u2, v2), (u1, v1); Nn = -Nn
                lo = lab[v1, u1]; zi = big[v2, u2]; zo = big[v1, u1]
                jump = zo - zi
                ishand = hands[li]
                oth_hand = np.array([hands[q] if q >= 0 else False for q in lo])
                layer = nm in OUTER; under = np.array([s.lname[q].rstrip('LR') in INNER if q >= 0 else False for q in lo])
                thr = np.where(oth_hand & ishand, .004, np.where(layer & under, .0035, .012)) * sc
                dm = mats[v1, u1] != mats[v2, u2]
                nd = np.sum(s.N[v1, u1] * s.N[v2, u2], -1)
                cls = np.zeros(n, int)
                cls[(lo < 0) | (lo == s.hl) & (jump > 0)] = 1
                occ = (lo >= 0) & (lo != li) & (jump > thr)
                cls[occ] = 2
                flat = (lo >= 0) & (lo != li) & (np.abs(jump) <= thr) & (lo > li)
                cls[flat & (dm | (nd < .72))] = 3
                cls[flat & ishand & oth_hand & (nd < .93)] = 3
                s.runs(c, cls, li, ci, Nn, cls_[ci])
        # material edges inside one label (cuffs, soles, shirt V)
        for li, nm in enumerate(s.lname):
            if li == s.hl: continue
            ms = np.unique(mats[lab == li])
            if len(ms) < 2: continue
            for q in ms[1:] if len(ms) > 1 else []:
                m = (lab == li) & (mats == q)
                if m.sum() < 8: continue
                for ci, c in enumerate(mask_polys(m, g, s.box, raw=True, up=up, min_area=4., smooth=5)):
                    uu = (c - [s.box[0], s.box[1]]) * g
                    T = np.roll(c, -2, 0) - np.roll(c, 2, 0); T /= np.hypot(*T.T)[:, None] + 1e-9; Nn = np.c_[T[:, 1], -T[:, 0]]
                    def lb(off):
                        z = uu + Nn * off; u = np.clip(np.floor(z[:, 0]).astype(int), 0, W_ - 1); v = np.clip(np.floor(z[:, 1]).astype(int), 0, H_ - 1); return lab[v, u], mats[v, u]
                    l1, m1 = lb(1.6); l2, m2 = lb(-1.6)
                    cls = np.where((l1 == li) & (l2 == li) & (m1 != m2), 3, 0)
                    s.runs(c, cls, 900 + li * 10 + int(q), ci, Nn)
        # authored folds
        lvl = 0 if s.px < 200 else 1 if s.px < 380 else 2
        for k, d in enumerate(B.deco):
            if d['kind'] != 'line' or d.get('pri', 0) > lvl: continue
            c = np.c_[d['t'], d['p']]
            C2 = spl(c, n=7) if not d.get('straight') else resample2(c, .06)
            Q = B.c.point(d['prim'], C2[:, 0], C2[:, 1], .001)
            xy, ok = s.vis(Q, d['prim'], .028)
            mt = B.c.mat[d['prim']]
            col = s.ink if not d.get('mat') else (_mixc(s.pal(mt)[3], s.ink, .35) if d['mat'] == 'skinln' else s.pal('obi')[0] if d['mat'] == 'obiln' else s.ink)
            for a, b in _true_runs(ok, 3):
                seg = xy[a:b]
                if np.hypot(*np.diff(seg, axis=0).T).sum() < max(4., s.px * .012): continue
                if mt in DARK and not d.get('mat'):
                    u, v = s.ij(*seg[len(seg) // 2])
                    if u is not None and s.S[v, u]: col = _mixc(s.pal(mt)[2], s.pal(mt)[0], .42)
                    else: col = s.ink
                s.cel.line(spl(seg, n=4), s.wf * d['w'] * 1.05, col, taper=(.12, .55), press=.3, seed=_hash(s.seed, k) & 0xffff, tmin=.12)
        s.front_decals()

    def front_decals(s):
        """suit front etc.: points given in the chest frame (x right, z up, metres at H=1.73), dropped onto the buffer"""
        B, sk = s.B, s.B.sk; fr = getattr(B, 'front', None)
        if not fr or s.px < 130: return
        k = B.m; facing = (sk.Rc[:, 1] @ (s.cam.eye - sk.chest)) / np.linalg.norm(s.cam.eye - sk.chest)
        if facing < .1: return
        for q, d in enumerate(fr):
            pts = d['pts'] * k
            if len(pts) > 1: pts = resample2(pts, .012)
            # chest above the waist, pelvis frame below
            fzb = lambda x, z: .145 * k if z > -.02 else .125 * k
            Q = np.array([sk.chest + sk.Rc[:, 0] * x + sk.Rc[:, 2] * z + sk.Rc[:, 1] * (.135 * k - 1.1 * x * x / k) if z > -.06 * k else sk.waist + sk.Rp[:, 0] * x + sk.Rp[:, 2] * (z + .145 * k) + sk.Rp[:, 1] * (.125 * k - 1.1 * x * x / k) for x, z in pts])
            xy, ok = s.vis(Q, B.prim['ribs'], .10)
            # only on torso labels
            for i, (x, y) in enumerate(xy):
                u, v = s.ij(x, y); ok[i] = ok[i] and u is not None and s.lname[s.lab[v, u]] in ('torso',)
            if d['kind'] == 'badge':
                if ok.mean() > .6:
                    V = xy; s.cel.fill(V, s.pal('gold')[1]); s.cel.fill(V[:len(V) // 2 + 1], s.pal('gold')[2]); kit.outline(s.cel, V, s.wf, s.ink, seed=3)
                continue
            if d['kind'] == 'dot':
                if ok[0] and s.px > 260:
                    u, v = s.ij(*xy[0])
                    if s.mn[s.mat[v, u]] == 'tie':
                        r = d['w'] * s.px / B.H; a = np.linspace(0, 6.283, 7)[:-1]; s.cel.fill(np.c_[xy[0][0] + r * np.cos(a) * .8, xy[0][1] + r * np.sin(a)], s.pal('tie')[3])
                continue
            if d['w'] <= 0: continue
            for a, b in _true_runs(ok, 3): s.stroke(xy[a:b], s.wf * d['w'] * 1.15, seed=_hash(s.seed, 'fr', q) & 0xffff, taper=(.1, .3))

    def runs(s, c, cls, li, ci, Nn, clean=None):
        """split a closed contour into same-class runs and pen them"""
        n = len(c); g = s.g
        k = 5
        # majority filter (closed)
        for v in (1, 2, 3):
            m = (cls == v).astype(np.uint8)
            mm = np.r_[m[-k:], m, m[:k]][None]; mm = cv2.morphologyEx(mm, cv2.MORPH_CLOSE, np.ones((1, 2 * k + 1), np.uint8))[0, k:-k]
            cls = np.where((cls == 0) & (mm > 0), v, cls)
        ch = np.nonzero(cls != np.roll(cls, 1))[0]
        if len(ch) == 0:
            if cls[0] == 0: return
            segs = [(0, n, cls[0], True)]
        else:
            segs = [(a, b if b > a else b + n, cls[a], False) for a, b in zip(ch, np.r_[ch[1:], ch[0] + n])]
        for a, b, v, closed in segs:
            if v == 0: continue
            idx = np.arange(a, b + (1 if closed else 1)) % n; P = c[idx]
            if clean is not None:
                Q, ix, nn_ = clean; i0 = clean_index(a, ix, nn_, n); M = len(Q)
                i1 = i0 + M if closed else clean_index(b % n, ix, nn_, n)
                if i1 <= i0: i1 += M
                ii = np.arange(math.ceil(i0), math.floor(i1) + 1) % M
                if len(ii) >= 3: P = Q[ii]
            L = np.hypot(*np.diff(P, axis=0).T).sum()
            if L < max(3.5, s.px * .010) and not closed: continue
            w = {1: s.wo, 2: s.wo * .86, 3: s.wf * 1.1}[int(v)]
            nm = s.lname[li] if li < 900 else ''
            if nm.startswith('hand'): w *= .70 if s.px > 250 else .85
            # weight follows the light: heavier on the side that faces away from it
            nn = Nn[idx].mean(0); lit = nn @ np.array([s.lc[0], -s.lc[1]]) / (np.hypot(s.lc[0], s.lc[1]) + 1e-6) if abs(nn).sum() > .3 else 0
            w *= 1 - .16 * lit
            tp = (.02, .02) if closed else ((.12, .12) if v == 1 else (.22, .30))
            s.stroke(P, w, seed=_hash(s.seed, li, ci, int(v)) & 0xffff, taper=tp)
            # dark cloth over dark cloth: a hair of edge light inside the nearer form keeps the overlap readable
            if v == 2 and s.px > 170 and li < 900 and L > s.px * .03:
                q = c[idx] - Nn[idx] * (w * .5 + s.wf * .55); u_ = np.clip(((q[:, 0] - s.box[0]) * g).astype(int), 0, s.Z.shape[1] - 1); v_ = np.clip(((q[:, 1] - s.box[1]) * g).astype(int), 0, s.Z.shape[0] - 1)
                mi = s.mat[v_, u_]; mo = s.mn[int(np.bincount(np.maximum(mi, 0)).argmax())]
                if mo in DARK and (s.lab[v_, u_] == li).mean() > .7:
                    c_ = _mixc(s.pal(mo)[1], s.pal(mo)[0], .55); n_ = len(q); a_ = int(n_ * .12); s.stroke(q[a_:n_ - a_], s.wf * .75, c_, seed=li + 7, taper=(.45, .45), split=False)

    # ------------------------------------------------ anchors + head
    def anchors(s):
        cam, sk = s.cam, s.B.sk; f = s.fig; p = lambda q: tuple(cam.project(np.asarray(q, float))[0])
        hl = s.B.b['hand'] * sk.H
        f.info = dict(px=s.px, box=s.box, hands={q: p(sk.wr[q] + sk.Rhand[q][:, 1] * hl * .55) for q in 'LR'}, hands3={q: sk.wr[q] + sk.Rhand[q][:, 1] * hl * .55 for q in 'LR'}, wrists={q: p(sk.wr[q]) for q in 'LR'},
                      feet={q: p(sk.an[q]) for q in 'LR'}, elbows={q: p(sk.el[q]) for q in 'LR'}, knees={q: p(sk.kn[q]) for q in 'LR'}, shoulders={q: p(sk.sh[q]) for q in 'LR'}, pelvis=p(sk.pel), chest=p(sk.chest), depth=float(cam.cam(sk.pel)[2]), skel=sk)
        f.info['head'] = fig_head.pose_info(cam, sk)

    def head_pass(s, img):
        hi = s.fig.info['head']; x0, y0, x1, y1 = s.box
        # grow the picture so hair fits
        m = int(hi['s'] * 2.4) + 8; nx0, ny0, nx1, ny1 = min(x0, int(hi['x'] - m)), min(y0, int(hi['y'] - m)), max(x1, int(hi['x'] + m)), max(y1, int(hi['y'] + m))
        nx0, ny0, nx1, ny1 = max(nx0, -260), max(ny0, -260), min(nx1, s.cam.w + 260), min(ny1, s.cam.h + 260)
        if nx1 - nx0 < 4 or ny1 - ny0 < 4: return img
        body = np.zeros((ny1 - ny0, nx1 - nx0, 4), f32); paste(body, img, x0 - nx0, y0 - ny0)
        roi = (nx0, ny0, nx1, ny1)
        if s.head_fn is not None:
            import inspect
            hi['H'] = fig_head.posed(s.fig.char, hi); info = dict(hi, f=s.fx['f'], roi=roi, hair=s.fx['hair'], fig=s.fig, pal=s.pal, wo=s.wo, light=s.lc, expr=s.expr)
            hd = np.zeros_like(body); np_ = sum(q.default is inspect._empty and q.kind in (0, 1) for q in inspect.signature(s.head_fn).parameters.values())
            if np_ >= 3:                                  # head_fn(cel, H, info): draws on a Cel that covers the roi
                cel = Cel(nx1 - nx0, ny1 - ny0, nx0, ny0, ss=2); L = s.head_fn(cel, hi['H'], info); L = cel.rgba() if L is None or isinstance(L, Cel) else L
            else: L = s.head_fn(hi['H'], info)
            if L is not None: hd = L if L.shape[:2] == body.shape[:2] else _crop(L, roi)
        else: hd = fig_head.draw(s.fig.char, hi, roi, s)
        if getattr(s.fig, 'head_warp', None) is not None: hd = s.fig.head_warp(hd, roi, hi)                  # squash of the drawn head (premultiplied roi), e.g. a fist in the cheek
        if s.rimd:
            q = np.array(s.rimd['dir'], float); q = q[:2] / (np.hypot(*q[:2]) + 1e-6); d_ = max(1.2, hi['s'] * .085)
            hd = kit.rim(hd, -q[0] * d_, q[1] * d_, s.rimd['col'], s.rimd.get('amt', .85) * .95, .5)
        # what stands in front of the head: anything nearer than the proxy (hands, arms, collar front)
        zref = hi['z']
        near = s.hit & (s.lab != s.hl) & (s.Z < zref - .10 * hi['u'])
        bl = [i for i, n in enumerate(s.lname) if n in ('neck',)]
        for i in bl: near &= s.lab != i
        # inside the proxy silhouette only what really hides it
        fm = cv2.resize(near.astype(f32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        F = np.zeros(body.shape[:2], f32); tmp = np.zeros(body.shape[:2] + (1,), f32); paste(tmp, fm[..., None], x0 - nx0, y0 - ny0); F = np.clip((tmp[..., 0] - .5) * 6 + .5, 0, 1)
        F = np.maximum(F, 0)
        # ink around the nearer parts belongs to them
        r = max(1, int(round(s.wo * .6))); F = cv2.dilate(F, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        front = body * F[..., None]
        out = body * (1 - hd[..., 3:4]) + hd
        out = out * (1 - F[..., None] * body[..., 3:4]) + front
        s.box = roi; s.fig.info['box'] = roi
        return out


def _crop(L, roi):
    x0, y0, x1, y1 = roi; out = np.zeros((y1 - y0, x1 - x0, 4), f32); h, w = L.shape[:2]
    a0, b0, a1, b1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if a1 > a0 and b1 > b0: out[b0 - y0:b1 - y0, a0 - x0:a1 - x0] = L[b0:b1, a0:a1]
    return out


def resample2(P, step):
    P = np.asarray(P, float); d = np.hypot(*np.diff(P, axis=0).T); s_ = np.r_[0, np.cumsum(d)]
    n = max(len(P), int(s_[-1] / step) + 2); t = np.linspace(0, s_[-1], n)
    return np.c_[np.interp(t, s_, P[:, 0]), np.interp(t, s_, P[:, 1])]


def _true_runs(ok, n):
    out = []; a = None
    for i, v in enumerate(list(ok) + [False]):
        if v and a is None: a = i
        if not v and a is not None:
            if i - a >= n: out.append((a, i))
            a = None
    return out
