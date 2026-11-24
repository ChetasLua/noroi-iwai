"""SUKUNA lane: brush-and-ink toolkit (paper, brush, feathering, wash, ink edge, splatter, dry brush)."""
import numpy as np, cv2, math, cairo
from jj import W, H
from jj.paint import rgb, f32, fbm, smooth, mix, grid, lum, Canvas, warp_by
from jj.ink import Cel, spl, resample, noise1, xf, paste
from jj.comp import over, blur

PAPER = '#e6dcc8'; BLACK = '#0d0b0e'; BLOOD = '#b3121f'; BLOOD_D = '#6d0c16'; BLOOD_L = '#e0454a'; DIRT = '#8d8782'
_cache = {}
CANVAS = [W, H]
HI = [1]          # raster scale of Sheets: 1 = frame size, 1.5 / 2 = master for camera moves (same drawing, more pixels)


class hires:
    """with hires(1.5): Sheet.roi()/layer() come out 1.5x bigger"""
    def __init__(self, k): self.k = k
    def __enter__(self): self.o = HI[0]; HI[0] = self.k
    def __exit__(self, *a): HI[0] = self.o


class canvas:
    """with canvas(w, h): ...   drawings bigger than the frame (masters for camera moves)"""
    def __init__(self, w, h): self.n = [int(w), int(h)]
    def __enter__(self): self.o = list(CANVAS); CANVAS[:] = self.n
    def __exit__(self, *a): CANVAS[:] = self.o


def _noise(key, scales, weights=None, h=H, w=W, seed=0, aspect=1.):
    k = (key, h, w, seed, scales)
    if k not in _cache: _cache[k] = fbm(h, w, scales, weights, seed, aspect)
    return _cache[k]


def _nz(key, scales, weights, h, w, seed, ox=0, oy=0, aspect=1.):
    """window of one cached full-frame noise plate (tiles)"""
    n = _noise(key, scales, weights, 1088, 1936, seed, aspect)
    if h <= 1088 and w <= 1936 and ox == 0 and oy == 0: return n[:h, :w]
    return n[np.ix_((np.arange(h) + oy) % 1088, (np.arange(w) + ox) % 1936)]


# ------------------------------------------------------------------ strokes
def prof(n, ws):
    """smooth width profile through control widths"""
    ws = np.asarray(ws, float)
    if ws.ndim == 0: return np.full(n, float(ws))
    if len(ws) == 1: return np.full(n, ws[0])
    x = np.linspace(0, len(ws) - 1, n); i = np.clip(x.astype(int), 0, len(ws) - 2); f = x - i; f = f * f * (3 - 2 * f)
    return ws[i] * (1 - f) + ws[i + 1] * f


def stroke_poly(P, ws, seed=0, rough=.35, step=1.3, tremor=.25):
    """polygon of a brush stroke along P with an explicit width profile ws (full widths px, 0 at a tip is a real point)"""
    P = resample(P, step); n = len(P)
    if n < 4: return None
    r = np.random.default_rng(abs(int(seed)) % (1 << 31))
    T = np.gradient(P, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]
    w = prof(n, ws)
    P = P + N * (noise1(n, 30, r) * tremor)[:, None]
    ra = rough * np.minimum(1, w / 3.)
    l = np.maximum(.16, .5 * w * (1 + .07 * noise1(n, 22, r)) + ra * .45 * noise1(n, 3.2, r))
    q = np.maximum(.16, .5 * w * (1 + .07 * noise1(n, 22, r)) + ra * .45 * noise1(n, 3.2, r))
    return np.vstack([P + N * l[:, None], (P - N * q[:, None])[::-1]])


def brush(c, P, ws, col=BLACK, seed=0, rough=.35, corners=(), smooth_=True, alpha=1., n=10, tremor=.25):
    """one brush stroke: P control points -> spline, ws = list of widths along it"""
    P = np.asarray(P, float)
    if len(P) < 2 or not np.isfinite(P).all(): return c
    lo, hi = P.min(0), P.max(0)
    if hi[0] < c.ox - 60 or lo[0] > c.ox + c.w + 60 or hi[1] < c.oy - 60 or lo[1] > c.oy + c.h + 60: return c
    Q = spl(P, corners=corners, n=n) if smooth_ and len(P) > 2 else P
    R = stroke_poly(Q, ws, seed, rough, tremor=tremor)
    if R is not None: c.fill(R, col, alpha)
    return c


def at(Q, t):
    """points at arclength fractions t of polyline Q"""
    Q = np.asarray(Q, float); s = np.r_[0, np.cumsum(np.hypot(*np.diff(Q, axis=0).T))]; s /= s[-1] + 1e-9
    t = np.asarray(t, float)
    return np.stack([np.interp(t, s, Q[:, 0]), np.interp(t, s, Q[:, 1])], -1)


def length(Q):
    Q = np.asarray(Q, float); return float(np.hypot(*np.diff(Q, axis=0).T).sum())


def feather(c, A, B, gap=6., w=2.2, col=BLACK, seed=0, reach=(.55, 1.), skew=0., bow=.06, group=5, hole=.5, edge=(.0, .0), env=None, alpha=1., back=.0, jit=.28, tip=.0, cross=False):
    """inker's feathering: tapered strokes that start ON curve A (fat) and die toward curve B.
    reach = min,max fraction of A->B; group = strokes per tuft, hole = how much shorter between tufts;
    env(t) scales the reach along the curve (shape of the shadow edge); skew shifts B along t; back = start behind A (fraction)."""
    r = np.random.default_rng(abs(int(seed)) + 17); A = resample(A, 2.); B = resample(B, 2.)
    n = max(3, int(length(A) / gap)); t = (np.arange(n) + .5 + r.normal(0, jit, n)) / n
    ph = r.uniform(0, 6.28)
    for k in range(n):
        tt = float(np.clip(t[k], 0, 1)); a = at(A, tt); b = at(B, np.clip(tt + skew, 0, 1))
        u = reach[0] + (reach[1] - reach[0]) * r.random() ** .8
        u *= 1 - hole * (.5 + .5 * math.sin(2 * math.pi * k / max(group, 1) + ph)) * r.uniform(.5, 1)
        if env is not None: u *= env(tt)
        if edge[0] or edge[1]: u *= min(1, tt / max(edge[0], 1e-3)) if tt < edge[0] else min(1, (1 - tt) / max(edge[1], 1e-3)) if tt > 1 - edge[1] else 1
        if u < .05: continue
        d = b - a; L = np.hypot(*d)
        if L * u < 2.5 or (max(a[0], b[0]) < c.ox - 5 or min(a[0], b[0]) > c.ox + c.w + 5 or max(a[1], b[1]) < c.oy - 5 or min(a[1], b[1]) > c.oy + c.h + 5): continue
        nn = np.array([-d[1], d[0]]) / (L + 1e-9); bw = L * u * (bow + r.normal(0, .025))
        p0 = a - d * back * r.random(); p2 = a + d * u; m = (p0 + p2) / 2 + nn * bw
        ww = w * r.uniform(.75, 1.25)
        Q = spl([p0, m, p2], n=8)
        R = stroke_poly(Q, [ww * .55, ww, ww * .75, ww * .4, tip], int(r.integers(1 << 30)), .2, 1.6, .15)
        if R is not None: c.fill(R, col, alpha)
    return c


def hatchfill(c, A, B, gap=6., w=1.5, seed=0, col=BLACK, taper=(.15, .45), jit=.25, bow=.02, drop=.0, alpha=1.):
    """hatching between two guide curves (strokes run from A(t) to B(t), nearly full length)"""
    r = np.random.default_rng(abs(int(seed)) + 3); A = resample(A, 2.); B = resample(B, 2.)
    n = max(2, int(max(length(A), length(B)) / gap))
    for k in range(n):
        if r.random() < drop: continue
        t = np.clip((k + .5 + r.normal(0, jit)) / n, 0, 1); a = at(A, t); b = at(B, np.clip(t + r.normal(0, .006), 0, 1))
        d = b - a; L = np.hypot(*d)
        if L < 4: continue
        u0, u1 = r.uniform(0, .10), 1 - r.uniform(0, .22); nn = np.array([-d[1], d[0]]) / L
        Q = spl([a + d * u0, a + d * (u0 + u1) / 2 + nn * L * (bow + r.normal(0, .012)), a + d * u1], n=8); ww = w * r.uniform(.7, 1.3)
        R = stroke_poly(Q, [ww * .3, ww, ww, ww * .6, .1], int(r.integers(1 << 30)), .2, 1.6, .15)
        if R is not None: c.fill(R, col, alpha)
    return c


def spot(c, P, edge_from=None, edge_to=None, seed=0, gap=5., w=2.4, col=BLACK, **k):
    """black pool; its edge_from -> edge_to side is feathered instead of cut"""
    c.fill(P, col)
    if edge_from is not None: feather(c, edge_from, edge_to, gap, w, col, seed, back=.12, **k)
    return c


# ------------------------------------------------------------------ surfaces
def paper(h=H, w=W, col=PAPER, seed=4, dirt=.5):
    """bone paper: fibres, stains, dirty wash in the corners"""
    k = ('paper', h, w, col, seed, dirt)
    if k in _cache: return _cache[k].copy()
    b = np.empty((h, w, 3), f32); b[:] = rgb(col)
    n = _noise('pa', (190, 70, 22, 7), (1, .7, .45, .3), h, w, seed); fi = _noise('fi', (2.2, 1.3), (1, .6), h, w, seed + 2, 2.4)
    x, y = grid(h, w); e = np.clip(np.hypot((x - w / 2) / (w * .6), (y - h / 2) / (h * .6)) - .62, 0, 1)
    d = smooth(.35, 1.9, n + e * 2.4) * dirt
    b = mix(b, rgb('#a79c8c') * np.ones_like(b), (d * .55)[..., None])
    b *= 1 + .011 * fi[..., None] + .012 * n[..., None] * np.array([1, .9, .7], f32)
    _cache[k] = b.astype(f32); return _cache[k].copy()


def inkify(L, amp=.3, seed=0, pit=.35, ox=0, oy=0):
    """ink on tooth: roughen every edge of a premultiplied layer by sub-pixel noise and punch a few dry pits into solid areas"""
    h, w = L.shape[:2]; x, y = grid(h, w); ox += seed * 131; oy += seed * 97
    nx = _nz('ikx', (3.4, 1.7), (1, .8), h, w, 11, ox, oy); ny = _nz('iky', (3.4, 1.7), (1, .8), h, w, 12, ox, oy)
    o = cv2.remap(L, x + nx * amp, y + ny * amp, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if pit > 0:
        p = _nz('pit', (4.2, 1.8), (1, .7), h, w, 13, ox, oy, 3.2); q = _nz('pit2', (190, 60), (1, .5), h, w, 14, ox, oy)
        m = smooth(2.5, 3.1, p + q * .62) * pit
        o = o * (1 - m)[..., None]
    return o


def washify(L, gran=.09, edge=.5, seed=0, amp=.9, ox=0, oy=0):
    """turn a flat layer of wash polygons into watercolour: granulated, dried darker at the edge, edge wobble"""
    h, w = L.shape[:2]; x, y = grid(h, w)
    nx = _nz('wx', (34, 13, 5), (1, .6, .35), h, w, 21, ox, oy); ny = _nz('wy', (34, 13, 5), (1, .6, .35), h, w, 22, ox, oy)
    o = cv2.remap(L, x + nx * amp, y + ny * amp, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    a = o[..., 3]; e = np.clip(a - cv2.GaussianBlur(a, (0, 0), 2.2), 0, 1)
    g = _nz('wg', (24, 6, 2.4), (.5, .7, .6), h, w, 23, ox, oy); bl = _nz('wb', (150, 60), (1, .5), h, w, 24, ox, oy)
    k = np.clip(1 + gran * g + edge * e * 1.4 - .16 * np.clip(bl, -.6, 1.4) + .0, .2, 1.6)
    return np.clip(o * k[..., None], 0, 1)


def drop_shape(cx, cy, r, seed=0, stretch=1., ang=0., m=18):
    """one ink drop as a closed curve: blobby, sometimes with a tail"""
    q = np.random.default_rng(abs(int(seed))); a = np.linspace(0, 2 * math.pi, m, endpoint=False)
    rad = r * (1 + .07 * np.sin(2 * a + q.uniform(0, 6)) + .04 * np.sin(3 * a + q.uniform(0, 6)) + .02 * q.normal(0, 1, m))
    X = np.cos(a) * rad; Y = np.sin(a) * rad
    X = np.where(X > 0, X * stretch, X)
    if stretch > 1.3: Y = Y * np.where(X > 0, np.clip(1 - .55 * X / (r * stretch), .12, 1), 1)
    c, s = math.cos(ang), math.sin(ang)
    return spl(np.c_[cx + X * c - Y * s, cy + X * s + Y * c], True, n=4)


def spray(c, p0, ang, n=60, reach=420, spread=.35, rmax=12, col=BLACK, seed=0, stretch=1.8):
    """thrown ink: big drops near, fine mist far, tails along the throw"""
    r = np.random.default_rng(seed)
    for k in range(n):
        d = reach * r.random() ** .8; th = ang + r.normal(0, spread) * (1 - .4 * d / reach)
        R = (rmax * r.random() ** 3.2 * (1 - .6 * d / reach) + .7)
        x, y = p0[0] + math.cos(th) * d, p0[1] + math.sin(th) * d; c.fill(drop_shape(x, y, R, int(r.integers(1 << 30)), 1 + (stretch - 1) * r.random() ** 2 * (R > 1.6), th), col)
        if R > 4.5:                                  # satellites and a thread
            for j in range(int(r.integers(2, 6))): q = r.uniform(1.3, 3.2) * R; t2 = th + r.normal(0, .8); c.fill(drop_shape(x + math.cos(t2) * q, y + math.sin(t2) * q, R * r.uniform(.08, .24), int(r.integers(1 << 30))), col)
            if r.random() < .5: t2 = th + r.normal(0, .25); e = R * r.uniform(1.6, 3.4); brush(c, np.array([(x, y), (x + math.cos(t2) * e * .5, y + math.sin(t2) * e * .5), (x + math.cos(t2) * e, y + math.sin(t2) * e)]), [R * .9, R * .35, .3], col, int(r.integers(1 << 30)), .3)
    return c


def dry(h, w, P, width, col=BLACK, seed=0, dryness=.4, bristle=0., op=1., shape='flat', cv=None):
    """big dry-brush sweep(s) on a transparent canvas; P polyline or list of polylines"""
    Ps = P if isinstance(P, list) else [P]; cv = cv or Canvas(h, w, color=col, layer=True)
    for i, p in enumerate(Ps):
        Q = resample(np.asarray(p, float), max(4., width * .22)); K = min(56, max(6, len(Q)))
        Q = Q[np.linspace(0, len(Q) - 1, K).round().astype(int)]
        cv.strokes(Q[None], width / 2, rgb(col)[None], op=op, dry=dryness, seed=seed + i * 7, shape=shape, bristle=bristle)
    return cv


class Sheet:
    """one inked drawing = flats + colour washes + ink.  a (flat cel), v (wash cel, multiplied), k (ink cel).  box = (x0,y0,x1,y1) ROI"""
    def __init__(self, seed=0, amp=.28, pit=.10, box=None):
        x0, y0, x1, y1 = [int(q) for q in (box or (0, 0, *CANVAS))]; x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(CANVAS[0], x1), min(CANVAS[1], y1)
        w, h = max(8, x1 - x0), max(8, y1 - y0); self.ox, self.oy, self.w, self.h = x0, y0, w, h
        self.hi = HI[0]; ss = 2 if self.hi == 1 else int(round(2 * self.hi)); self.a = Cel(w, h, x0, y0, ss); self.v = Cel(w, h, x0, y0, ss); self.k = Cel(w, h, x0, y0, ss); self.g = Cel(w, h, x0, y0, ss); self.seed, self.amp, self.pit = seed, amp, pit
        self.rw, self.rh = int(w * self.hi), int(h * self.hi)

    def ras(self, c):
        if self.hi == 1: return c.rgba()
        c.surf.flush(); a = np.ndarray((c.h * c.ss, c.surf.get_stride() // 4, 4), np.uint8, c.surf.get_data())[:, :c.w * c.ss]
        return cv2.resize(a, (self.rw, self.rh), interpolation=cv2.INTER_AREA).astype(f32)[..., [2, 1, 0, 3]] / 255.

    def wash(self, P, col, op=1., within=None):
        if within is not None: self.v.clip(within)
        self.v.fill(P, col, op)
        if within is not None: self.v.unclip()

    def cut(self, P, *cels):
        """erase ink and wash under a shape that is about to be drawn in front"""
        for c in (cels or (self.v, self.k, self.g)):
            c.ctx.save(); c.ctx.set_operator(cairo.OPERATOR_DEST_OUT); c.fill(P, '#000000'); c.ctx.restore()

    def put(self, P, col, cut=True):
        if cut: self.cut(P)
        self.a.fill(P, col)

    def pen(self, P, w, col=BLACK, cel=None, miter=6):
        """hard mitred outline (used only under other strokes)"""
        c = cel or self.k; x = c.ctx; x.save(); c._path(P); c._src(col, 1.); x.set_line_width(w); x.set_line_join(cairo.LINE_JOIN_MITER); x.set_miter_limit(miter); x.stroke(); x.restore()

    def clip(self, P):
        for c in (self.a, self.v, self.k, self.g): c.clip(P)

    def unclip(self):
        for c in (self.a, self.v, self.k, self.g): c.unclip()

    def flat(self):
        A = self.ras(self.a); V = washify(self.ras(self.v), ox=self.ox, oy=self.oy); va = V[..., 3:4]; vc = V[..., :3] / np.maximum(va, 1e-4)
        A = A.copy(); A[..., :3] *= 1 - va * (1 - vc)
        fi = _nz('fig', (2.2, 1.3), (1, .6), self.rh, self.rw, 41, self.ox, self.oy, 2.4); mo = _nz('figm', (150, 46), (1, .6), self.rh, self.rw, 42, self.ox, self.oy)
        A[..., :3] *= (1 + .012 * fi - .022 * np.clip(mo, -1, 2))[..., None]
        return A

    def roi(self):
        """-> premultiplied RGBA of the ROI (origin self.ox, self.oy in drawing px; scale self.hi)"""
        A = self.flat(); K = inkify(self.ras(self.k), self.amp * self.hi, self.seed, self.pit, self.ox, self.oy)
        A = A * (1 - K[..., 3:4]) + K; G = inkify(self.ras(self.g), self.amp * self.hi, self.seed + 1, .15, self.ox, self.oy)
        return A * (1 - G[..., 3:4]) + G

    def layer(self):
        out = np.zeros((int(CANVAS[1] * self.hi), int(CANVAS[0] * self.hi), 4), f32); paste(out, self.roi(), int(self.ox * self.hi), int(self.oy * self.hi)); return out


# ------------------------------------------------------------------ cloth marks
def off(P, d):
    """polyline moved sideways by d px (number or per-point array), left of travel positive"""
    P = np.asarray(P, float); T = np.gradient(P, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9
    return P + np.c_[T[:, 1], -T[:, 0]] * (np.broadcast_to(d, len(P)))[:, None]


def fold(sh, P, w=6., reach=40., side=1, seed=0, gap=6.5, hw=2.6, shape=(.2, 1., .8, .45, .08), wash=None, wr=1.3, feath=True, env=None, rng=(.25, 1.)):
    """one cloth fold: fat tapered brush line from its tension point, wash and feathering on its shadow side (side=+1 left of travel)"""
    Q = spl(np.asarray(P, float), n=12) if len(P) > 2 else resample(P, 4.); n = len(Q); t = np.linspace(0, 1, n)
    R_ = reach * np.sin(np.pi * np.clip(.12 + .80 * t, 0, 1)) ** .9
    if wash is not None: sh.wash(np.vstack([Q, off(Q, side * R_ * wr)[::-1]]), wash, .7)
    brush(sh.k, Q[::2], np.array(shape) * w, seed=seed, rough=.5)
    if feath: feather(sh.k, Q[int(n * .04):int(n * .94)], off(Q, side * R_)[int(n * .04):int(n * .94)], gap, hw, seed=seed + 1, reach=rng, skew=.10, group=6, hole=.6, env=env, back=.05)


def pool(sh, P, corners=(), seed=0, fe=None, n=9, hw=2.6, gap=6., reach=(.3, 1.), **k):
    """designed black: closed spline; fe = (A, B) polylines for a feathered side"""
    Q = spl(np.asarray(P, float), True, corners, n); sh.k.fill(Q, BLACK)
    if fe is not None: feather(sh.k, spl(np.asarray(fe[0], float), n=8), spl(np.asarray(fe[1], float), n=8), gap, hw, seed=seed, reach=reach, back=.25, **k)


def edge(sh, P, ws, seed=0, corners=(), lw=1.):
    """cloth contour with authored weight"""
    brush(sh.k, P, np.asarray(ws, float) * lw, seed=seed, rough=.45, corners=corners)


def crease(sh, P, w=12., fan=6, spread=30., side=1, seed=0, t1=(.40, .92), hw=3., wash=None, wr=1.25, tail=.1, cel=None, col=BLACK, start=.0):
    """a fold drawn from its tension point: a black wedge whose shadow side frays into long strokes that diverge with the cloth"""
    Q = spl(np.asarray(P, float), n=14) if len(P) > 2 else resample(P, 3.); n = len(Q); t = np.linspace(0, 1, n); r = np.random.default_rng(abs(int(seed)) + 9)
    T = np.gradient(Q, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; N = np.c_[T[:, 1], -T[:, 0]] * side; k = cel or sh.k
    if wash is not None:
        z = spread * wr * np.sin(np.pi * np.clip(.04 + .86 * t, 0, 1)) ** .75; sh.wash(np.vstack([Q, (Q + N * z[:, None])[::-1]]), wash, .75)
    brush(k, Q[::2], np.array([.35, 1., .8, .5, .22, tail / max(w, 1)]) * w, col, seed, .5)
    for j in range(fan):
        f = (j + .6 + r.normal(0, .18)) / fan; a = start + .16 * r.random() + .05 * f; b = r.uniform(*t1) * (1 - .30 * f) + .06
        if b - a < .12: continue
        i0, i1 = int(n * a), max(int(n * a) + 6, int(n * min(b, .99))); tt = t[i0:i1]; u = (tt - tt[0]) / (tt[-1] - tt[0] + 1e-6)
        o = w * .34 + spread * f * (u ** .85) * (.45 + .55 * tt) / .9; L = Q[i0:i1] + N[i0:i1] * (o + r.normal(0, .8))[:, None]
        ww = hw * r.uniform(.65, 1.35) * (1.25 - .6 * f); R = stroke_poly(L, np.array([.5, 1., .8, .4, .05]) * ww, int(r.integers(1 << 30)), .3, 1.6, .25)
        if R is not None: k.fill(R, col)


def ink_ground(seed=6, keep=None, tone='#161314', paper_col=PAPER):
    """bone sheet swept over with a huge dry brush of black; keep(x, y) -> 0..1 how much raw paper stays (default: a wedge lower left)"""
    k = ('ground', seed, tone, paper_col)
    if k in _cache: return _cache[k].copy()
    r = np.random.default_rng(seed); b = paper(H, W, paper_col, seed, .8)
    keep = keep or (lambda x, y: np.clip((y - 640 - .52 * x) / 260., 0, 1))
    # grey first coat, then charcoal, then black, every pull steep and slightly fanned
    for tone_, n, w0, w1, dr, op in (('#6d6662', 22, 120, 260, .45, .75), ('#3a3433', 44, 110, 250, .35, .9), (tone, 70, 70, 220, .30, .95), ('#0b090b', 36, 40, 150, .45, .9)):
        cv = Canvas(H, W, img=b)
        for j in range(n):
            x = r.uniform(-120, W + 120); y = r.uniform(-260, H * .72); a = 1.16 + .18 * (x / W - .5) + r.normal(0, .07); L = r.uniform(420, 1000); q = np.linspace(0, 1, 9)[:, None]; d = np.array([math.cos(a), math.sin(a)]); P = np.array([x, y]) + d * L * q + np.array([-d[1], d[0]]) * r.normal(0, 26) * (q ** 2)
            e = P[-1]; kk = float(keep(e[0], e[1]))
            while kk > .25 and L > 120: L *= .86; P = np.array([x, y]) + d * L * q; e = P[-1]; kk = float(keep(e[0], e[1]))
            if L <= 130: continue
            cv.strokes(P[None].astype(f32), r.uniform(w0, w1) / 2, rgb(tone_)[None] * r.uniform(.85, 1.15), op=op * r.uniform(.8, 1), dry=dr * r.uniform(.5, 1.6), seed=j + seed * 50, bristle=.10)
        b = cv.img
    t = _noise('ti', (2.6, 1.3), (1, .8), H, W, seed + 3); b = b * (1 + .05 * t)[..., None]
    _cache[k] = np.clip(b, 0, 1).astype(f32); return _cache[k].copy()


def red_rim(L, dx, dy, col=BLOOD, amt=.85, seed=3, lo=-.2):
    """blood-red dry-brushed edge light on a finished layer (light comes from (-dx,-dy))"""
    a = L[..., 3]; h, w = a.shape; sh_ = cv2.warpAffine(a, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE); m = np.clip(a - sh_, 0, 1)
    n = _nz('rr', (40, 14, 5), (1, .7, .5), h, w, 31); m = m * smooth(lo - .25, lo + .25, n) * amt; m = cv2.GaussianBlur(m, (0, 0), .7) * (a > .05)
    lu = (L[..., 0] * .3 + L[..., 1] * .59 + L[..., 2] * .11) / np.maximum(a, 1e-3); m = m * smooth(.10, .30, lu)       # ink stays black
    o = L.copy(); o[..., :3] = L[..., :3] * (1 - m[..., None]) + rgb(col) * (m * a)[..., None]; return o
