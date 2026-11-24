"""Drawing kit on top of ink.Cel: SVG paths, 2.5-D turn, hair spikes, auto ink from flats, layer light (rim, para,
cast, aura), hatching, dry-brush ink, splatter, screentone, speed lines, smears, impact frames."""
import numpy as np, cv2, math, re, cairo
from . import W, H
from .paint import rgb, f32, fbm, lum, smooth, mix, grid, Canvas, arcs
from .ink import Cel, spl, resample, ribbon, noise1, over_roi, boil, xf
from .comp import over, screen, blur, radial

_TOK = re.compile(r'([MmLlHhVvCcSsQqTtZz])|(-?\d*\.?\d+(?:[eE][-+]?\d+)?)')


# ----------------------------------------------------------------- SVG paths
def svg(d, n=14):
    """SVG path string -> list of (k,2) polylines (M L H V C S Q T Z, abs + rel).  One subpath -> still a list."""
    t = [(a, b) for a, b in _TOK.findall(d)]
    i = 0; out = []; cur = []; x = y = 0.; sx = sy = 0.; pc = None; cmd = None
    def num():
        nonlocal i
        v = float(t[i][1]); i += 1; return v
    while i < len(t):
        if t[i][0]: cmd = t[i][0]; i += 1
        c = cmd.upper(); r = cmd.islower()
        if c == 'M':
            if cur: out.append(np.array(cur))
            a, b = num(), num(); x, y = (x + a, y + b) if r else (a, b)
            sx, sy = x, y; cur = [(x, y)]; cmd = 'l' if r else 'L'; pc = None
        elif c == 'L':
            a, b = num(), num(); x, y = (x + a, y + b) if r else (a, b); cur.append((x, y)); pc = None
        elif c == 'H':
            a = num(); x = x + a if r else a; cur.append((x, y)); pc = None
        elif c == 'V':
            a = num(); y = y + a if r else a; cur.append((x, y)); pc = None
        elif c in 'CS':
            if c == 'C':
                x1, y1 = num(), num()
                if r: x1 += x; y1 += y
            else:
                x1, y1 = (2 * x - pc[0], 2 * y - pc[1]) if pc is not None else (x, y)
            x2, y2, x3, y3 = num(), num(), num(), num()
            if r: x2 += x; y2 += y; x3 += x; y3 += y
            u = np.linspace(0, 1, n + 1)[1:, None]
            P = (1 - u) ** 3 * np.array([x, y]) + 3 * (1 - u) ** 2 * u * np.array([x1, y1]) + 3 * (1 - u) * u * u * np.array([x2, y2]) + u ** 3 * np.array([x3, y3])
            cur += [tuple(p) for p in P]; pc = (x2, y2); x, y = x3, y3
        elif c in 'QT':
            if c == 'Q':
                x1, y1 = num(), num()
                if r: x1 += x; y1 += y
            else:
                x1, y1 = (2 * x - pc[0], 2 * y - pc[1]) if pc is not None else (x, y)
            x3, y3 = num(), num()
            if r: x3 += x; y3 += y
            u = np.linspace(0, 1, n + 1)[1:, None]
            P = (1 - u) ** 2 * np.array([x, y]) + 2 * (1 - u) * u * np.array([x1, y1]) + u * u * np.array([x3, y3])
            cur += [tuple(p) for p in P]; pc = (x1, y1); x, y = x3, y3
        elif c == 'Z':
            cur.append((sx, sy)); x, y = sx, sy; out.append(np.array(cur)); cur = [(x, y)]; pc = None
            if i < len(t) and not t[i][0]: cmd = 'L'
    if len(cur) > 1: out.append(np.array(cur))
    return out


def s1(d, n=14):
    """first subpath of an SVG path as one polyline"""
    return svg(d, n)[0]


# --------------------------------------------------------- 2.5-D turn (Live2D style)
class Turn:
    """Head/body turn for an authored drawing.  Local points (x right, y down, z toward viewer, any unit).
    T = Turn(cx, cy, scale, yaw=.1, pitch=-.05, roll=.08);  T(points_xy, z=0.3 or array) -> screen px.
    yaw/pitch in radians, keep |yaw| < .35 for an authored front view.  persp = strength of perspective (0..0.4)."""
    def __init__(self, cx, cy, s, yaw=0., pitch=0., roll=0., sx=1., sy=1., persp=.12, unit=1.):
        self.c = np.array([cx, cy], float); self.s = s; self.sx, self.sy = sx, sy; self.k = persp / unit
        cy_, sy_ = math.cos(yaw), math.sin(yaw); cp, sp = math.cos(pitch), math.sin(pitch); cr, sr = math.cos(roll), math.sin(roll)
        Ry = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]])
        Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
        self.R = Rz @ Rx @ Ry

    def __call__(self, P, z=0.):
        P = np.asarray(P, float)
        if P.ndim == 2 and P.shape[1] == 3: Q = P
        else: Q = np.concatenate([P, np.broadcast_to(np.asarray(z, float), P.shape[:-1])[..., None]], -1)
        q = Q @ self.R.T
        f = 1 / (1 - np.clip(q[..., 2] * self.k, -.6, .6))
        return np.stack([q[..., 0] * f * self.s * self.sx, q[..., 1] * f * self.s * self.sy], -1) + self.c


# ----------------------------------------------------------------- hair
def spike(a, tip, b, bulge=.18, bend=0., n=12):
    """one hair clump: root a -> tip -> root b.  bulge = belly of the sides (fraction of length), bend = sideways S-curve."""
    a, tip, b = [np.asarray(v, float) for v in (a, tip, b)]
    m = (a + b) / 2; ax = tip - m; L = np.hypot(*ax) + 1e-6; nrm = np.array([-ax[1], ax[0]]) / L
    def side(p, sgn):
        u = np.linspace(0, 1, n)[:, None]
        c1 = p + ax * .38 + nrm * L * (bulge * sgn + bend * .6); c2 = p + (tip - p) * .74 + nrm * L * (bulge * .35 * sgn + bend)
        return (1 - u) ** 3 * p + 3 * (1 - u) ** 2 * u * c1 + 3 * (1 - u) * u * u * c2 + u ** 3 * tip
    s = 1 if np.dot(a - m, nrm) >= 0 else -1
    A = side(a, s); B = side(b, -s)
    return np.vstack([A, B[::-1][1:]])


# --------------------------------------------------------- inked flats
def outline(cel, P, w=3.0, col='#17131a', seed=0, closed=True, press=.35, corners=(), breaks=0, tremor=.25, gaps=None):
    """ink a closed contour like a pen: split at sharp corners so every stroke tapers into its corner.
    breaks = number of random lifted-pen gaps; gaps = [(t0,t1)] explicit gaps along 0..1."""
    P = resample(np.asarray(P, float), 2.0)
    if closed: P = np.vstack([P, P[:1]])
    T = np.diff(P, axis=0); ang = np.unwrap(np.arctan2(T[:, 1], T[:, 0]))
    k = 4
    turn = np.abs(ang[k:] - ang[:-k]) if len(ang) > k else np.zeros(0)
    idx = []
    j = 0
    while j < len(turn):
        if turn[j] > .85:
            e = j
            while e + 1 < len(turn) and turn[e + 1] > .85: e += 1
            idx.append(int(j + np.argmax(turn[j:e + 1])) + k // 2 + 1); j = e + k
        j += 1
    cut = [0] + [q for q in idx if 3 < q < len(P) - 3] + [len(P) - 1]
    r = np.random.default_rng(abs(int(seed)) + 5)
    gp = list(gaps or [])
    for _ in range(breaks):
        g = r.uniform(.05, .9); gp.append((g, g + r.uniform(.008, .022)))
    for q, (a, b) in enumerate(zip(cut[:-1], cut[1:])):
        seg = P[a:b + 1]
        if len(seg) < 4: continue
        parts = [seg]
        for g0, g1 in gp:
            i0, i1 = int(g0 * len(P)), int(g1 * len(P))
            np_ = []
            for s in parts:
                off = a if s is seg else None
                if off is None: np_.append(s); continue
                l0, l1 = i0 - a, i1 - a
                if 4 < l0 < len(s) - 4: np_ += [s[:l0], s[min(l1, len(s) - 4):]]
                else: np_.append(s)
            parts = np_
        for s in parts:
            if len(s) >= 4: cel.line(s, w, col, taper=(.16, .16), press=press, seed=seed * 31 + q, tremor=tremor, tmin=.32)
    return cel


def shape(cel, P, fill, w=3.0, ink='#17131a', seed=0, **k):
    """flat + ink outline in one call; P = polyline or list of polylines"""
    Ps = P if isinstance(P, list) else [P]
    cel.fill(Ps, fill)
    if w > 0:
        for i, q in enumerate(Ps): outline(cel, q, w, ink, seed + i, **k)
    return cel


def mask_of(P, h=H, w=W, ss=2):
    """antialiased float mask of polygon(s) in screen px"""
    Ps = P if isinstance(P, list) else [P]
    c = Cel(w, h, ss=ss)
    c.fill(Ps, '#ffffff'); return c.rgba()[..., 3]


def contours(mask, eps=1.0, min_len=24):
    """smooth outer/inner contours of a float mask -> list of polylines"""
    m = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    cs, _ = cv2.findContours((m > 127).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        c = c[:, 0].astype(float)
        if len(c) < min_len: continue
        k = np.hanning(9); k /= k.sum()
        cc = np.vstack([c[-8:], c, c[:8]])
        s = np.stack([np.convolve(cc[:, i], k, 'same') for i in (0, 1)], -1)[8:-8]
        out.append(s)
    return out


# ----------------------------------------------------------------- light on layers (premultiplied RGBA float)
def shifted(a, dx, dy):
    h, w = a.shape[:2]
    return cv2.warpAffine(a, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def rim(layer, dx, dy, col, amt=1.0, soft=.6, where=None):
    """hard rim light: the band of the layer not covered by itself shifted by (dx,dy).  Light comes from (-dx,-dy)."""
    a = layer[..., 3]; m = np.clip(a - shifted(a, dx, dy), 0, 1)
    if soft > 0: m = cv2.GaussianBlur(m, (0, 0), soft) * (a > .02)
    if where is not None: m = m * where
    c = rgb(col) if isinstance(col, str) else np.asarray(col, f32)
    out = layer.copy(); out[..., :3] = out[..., :3] * (1 - m[..., None] * amt) + c * (m * a)[..., None] * amt
    return out


def para(layer, p0, p1, stops, mode='mul', amt=1.0):
    """MAPPA-style gradient pass over a cel: stops=[(t,'#hex')] from p0 to p1; mode mul|screen|over"""
    h, w = layer.shape[:2]; x, y = grid(h, w)
    d = np.array(p1, float) - p0; t = ((x - p0[0]) * d[0] + (y - p0[1]) * d[1]) / (d @ d + 1e-9)
    from .paint import ramp
    g = ramp(np.clip(t, 0, 1), stops); a = layer[..., 3:4]; c = layer[..., :3]
    if mode == 'mul': n = c * g
    elif mode == 'screen': n = a - (a - c) * (1 - g)
    else: n = g * a
    out = layer.copy(); out[..., :3] = c + (n - c) * amt
    return out


def tint(layer, col, amt, mode='mul'):
    c = rgb(col) if isinstance(col, str) else np.asarray(col, f32); out = layer.copy()
    if mode == 'mul': out[..., :3] *= 1 + (c - 1) * amt
    else: out[..., :3] = out[..., :3] * (1 - amt) + c * layer[..., 3:4] * amt
    return out


def aura(base, layer, r, col, amt=1.0, hard=0.0):
    """glow outline behind a cel (composite BEFORE drawing the cel): blurred alpha, optional hard inner band"""
    a = layer[..., 3]; g = blur(a, r)
    if hard > 0: g = np.maximum(g, (cv2.dilate(a, np.ones((3, 3), np.uint8), iterations=int(hard))) * .9)
    c = rgb(col) if isinstance(col, str) else np.asarray(col, f32)
    return screen(base, np.clip(g * 1.6, 0, 1)[..., None] * c * amt)


def cast(base, layer, dx, dy, col='#1a1426', amt=.5, soft=3):
    a = shifted(layer[..., 3], dx, dy)
    if soft: a = blur(a, soft)
    c = rgb(col) if isinstance(col, str) else np.asarray(col, f32)
    return base * (1 - a[..., None] * amt) + c * a[..., None] * amt


# ----------------------------------------------------------------- ink texture
def hatch(cel, region, ang, gap=7., w=1.5, col='#17131a', seed=0, length=None, fade=0., curve=.04, alpha=1.0, jit=.25):
    """pen hatching inside a polygon.  ang radians; length None = full chords, else strokes of that length;
    fade 0..1 drops strokes toward the far end of the region along the hatch normal."""
    P = np.asarray(region, float); r = np.random.default_rng(seed)
    c, s = math.cos(ang), math.sin(ang); d = np.array([c, s]); nrm = np.array([-s, c])
    u = P @ d; v = P @ nrm
    cel.clip(P)
    vs = np.arange(v.min(), v.max(), gap)
    for k, vv in enumerate(vs):
        vv += r.normal(0, gap * jit); tt = (vv - v.min()) / (v.max() - v.min() + 1e-6)
        if fade and r.random() < fade * tt: continue
        a, b = u.min() - 4, u.max() + 4
        spans = [(a, b)] if length is None else [(q, q + length * r.uniform(.5, 1.2)) for q in np.arange(a, b, length * 1.15) + r.uniform(-.3, .3) * length]
        for q0, q1 in spans:
            m = (q0 + q1) / 2; bow = (q1 - q0) * curve * r.normal(0, 1)
            L = spl([q0 * d + vv * nrm, m * d + (vv + bow) * nrm, q1 * d + (vv + r.normal(0, gap * .2)) * nrm], n=8)
            cel.line(L, w * r.uniform(.7, 1.25), col, taper=(.12, .5), press=.35, seed=int(r.integers(1 << 30)), alpha=alpha)
    cel.unclip(); return cel


def drybrush(h, w, P, width, col='#0c0a0d', seed=0, dry=.45, K=None, shape='flat', bristle=.0, op=1.0):
    """big dry-brush ink stroke(s): P = polyline -> premultiplied RGBA layer"""
    cv = Canvas(h, w, color=col, layer=True)
    Q = resample(np.asarray(P, float), max(4., width * .25)); K = K or min(64, max(6, len(Q)))
    idx = np.linspace(0, len(Q) - 1, K).round().astype(int)
    cv.strokes(Q[idx][None], width / 2, rgb(col)[None], op=op, dry=dry, seed=seed, shape=shape, bristle=bristle)
    return cv.layer()


def splatter(cel, cx, cy, spread, n=60, rmax=9, col='#0c0a0d', seed=0, ang=None, aniso=1.0):
    """ink drops; ang+aniso throws them along a direction"""
    r = np.random.default_rng(seed)
    for _ in range(n):
        d = abs(r.normal(0, 1)) * spread; th = r.uniform(0, 2 * math.pi) if ang is None else ang + r.normal(0, .5 / aniso)
        x, y = cx + math.cos(th) * d, cy + math.sin(th) * d
        R = rmax * r.random() ** 3 + .7; m = 14; a = np.linspace(0, 2 * math.pi, m, endpoint=False)
        rad = R * (1 + .22 * r.normal(0, 1, m)); st = 1 + (r.random() * 1.6 if ang is not None else 0)
        cel.fill(spl(np.c_[x + np.cos(a) * rad * st, y + np.sin(a) * rad], True, n=4), col)
    return cel


def screentone(mask, pitch=7.0, ang=.785, dark=.35, h=H, w=W):
    """manga dot tone 0..1 coverage inside a mask (multiply or use as ink alpha)"""
    x, y = grid(h, w); c, s = math.cos(ang), math.sin(ang)
    u = (x * c + y * s) / pitch; v = (-x * s + y * c) / pitch
    d = np.sqrt((u - np.round(u)) ** 2 + (v - np.round(v)) ** 2)
    rad = np.sqrt(np.clip(dark, 0, 1) / math.pi) if np.isscalar(dark) else np.sqrt(np.clip(dark, 0, 1) / math.pi)
    return np.clip((rad - d) * pitch + .5, 0, 1) * mask


def focus_lines(cel, cx, cy, r0, r1=1400, n=150, w=7., col='#0c0a0d', seed=0, clump=.5, ry=.75, a0=0, a1=2 * math.pi):
    """manga concentration lines: wedges from r1 in to a ragged inner ellipse r0"""
    r = np.random.default_rng(seed)
    th = np.sort(r.uniform(a0, a1, n)) + clump * .05 * r.normal(0, 1, n)
    for t in th:
        q = r0 * r.uniform(.85, 1.5); ww = w * r.random() ** 2 * 2 + .8
        c, s = math.cos(t), math.sin(t)
        tip = np.array([cx + c * q, cy + s * q * ry]); far = np.array([cx + c * r1, cy + s * r1])
        nrm = np.array([-s, c]) * ww
        cel.fill(np.array([tip, far + nrm, far - nrm]), col)
    return cel


def speed_lines(cel, ang, n=80, w=5., L=(300, 1400), col='#0c0a0d', seed=0, box=(0, 0, W, H), alpha=1.0):
    """parallel speed lines (tapered both ends)"""
    r = np.random.default_rng(seed); d = np.array([math.cos(ang), math.sin(ang)]); nrm = np.array([-d[1], d[0]])
    x0, y0, x1, y1 = box
    for _ in range(n):
        p = np.array([r.uniform(x0, x1), r.uniform(y0, y1)]); ln = r.uniform(*L); ww = w * r.random() ** 2 + .5
        a, b = p - d * ln / 2, p + d * ln / 2
        cel.fill(np.array([a, p + nrm * ww, b, p - nrm * ww]), col, alpha)
    return cel


# ----------------------------------------------------------------- camera / shock
def shake(f, amp, seed=0, decay=None, t0=0):
    """hand-held / impact shake (px): new offset every frame; decay in frames"""
    r = np.random.default_rng(seed * 977 + int(f)); k = 1. if decay is None else math.exp(-max(0, f - t0) / decay) * (f >= t0)
    return r.normal(0, 1, 2) * amp * k


def smear(img, dx, dy, n=13):
    """directional shutter blur of an RGB/RGBA float image"""
    out = np.zeros_like(img)
    h, w = img.shape[:2]
    for t in np.linspace(-.5, .5, n):
        out += cv2.warpAffine(img, np.float32([[1, 0, dx * t], [0, 1, dy * t]]), (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return out / n


def zoomblur(img, cx, cy, amt=.08, n=13):
    out = np.zeros_like(img); h, w = img.shape[:2]
    for t in np.linspace(0, 1, n):
        s = 1 + amt * t
        out += cv2.warpAffine(img, np.float32([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)]]), (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return out / n


def impact(img, mode='bw', lo=.25, hi=.62, ink='#08070a', paper='#f6f2ea', accent='#d8142a'):
    """impact frame from a finished picture: bw | inv | red (3-tone)"""
    L = cv2.GaussianBlur(lum(np.clip(img, 0, 1)), (0, 0), 1.0); t = smooth(lo + .5 * (hi - lo) - .03, lo + .5 * (hi - lo) + .03, L)[..., None]
    a, b = rgb(ink), rgb(paper)
    if mode == 'inv': a, b = b, a
    o = a * (1 - t) + b * t
    if mode == 'red':
        m = (smooth(lo - .03, lo + .03, L) * (1 - smooth(hi - .03, hi + .03, L)))[..., None]; o = o * (1 - m) + rgb(accent) * m
    return o


def flare(base, cx, cy, r, col, amt=1., streak=0., ang=0.):
    """soft light bloom + optional anamorphic streak"""
    g = radial(cx, cy, r, power=2.4) + .5 * radial(cx, cy, r * .28, power=1.5)
    if streak:
        x, y = grid(); c, s = math.cos(ang), math.sin(ang); u = (x - cx) * c + (y - cy) * s; v = -(x - cx) * s + (y - cy) * c
        g = g + streak * np.exp(-np.abs(v) / (r * .012)) * np.exp(-np.abs(u) / (r * 1.6))
    c = rgb(col) if isinstance(col, str) else np.asarray(col, f32)
    return base + np.clip(g, 0, 2)[..., None] * c * amt
