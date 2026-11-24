"""fx air: rain (can stop and run backwards), splashes, snow, ash, petals, dust motes, mist."""
from .fx_core import *
from .fx_break import _rotm
from .comp import disc_kernel


# ------------------------------------------------------------------ rain
def rain(f=None, tau=None, v=1., n=(340, 150, 34), wind=.10, seed=0, col='#cfdcf2', dark='#3a4668', alpha=1., speed=1500., slope=0., box=None, size=1., near=True, light=None):
    """rain in three depths.  tau = rain time in seconds (default f/24): any function of time, so it can slow, stop, reverse;
    v = d tau / d t now (1 normal, 0 hanging, -0.5 rising): streak length follows |v|, at v ~ 0 each drop is a drawn bead with a highlight.  -> RGBA layer"""
    tau = f / 24. if tau is None else tau; av = abs(v); pad = 260; cel = Cel(); far = np.zeros((H, W), np.uint8); th = math.atan2(1, wind)
    for li, (cnt, sp, ln, wd, al) in enumerate(((n[0], .62, 60, 1.0, .30), (n[1], 1.0, 120, 1.7, .46), (n[2], 1.75, 260, 3.6, .62))):
        for i in range(cnt):
            u = h1(seed, li, i, 1); q = (tau * speed * sp * (.85 + .3 * u) / (H + 2 * pad) + h1(seed, li, i, 2)); cyc = math.floor(q); y = (q - cyc) * (H + 2 * pad) - pad; x0 = h1(seed, li, i, cyc, 3) * (W + 400) - 200; x = x0 + (y - H / 2) * wind; l = ln * size * av * (.6 + .8 * h1(seed, li, i, 4)); w = wd * size * (.7 + .6 * h1(seed, li, i, 5))
            if li == 0 and av > .10: cv2.line(far, (int((x - l * wind) * 16), int((y - l) * 16)), (int(x * 16), int(y * 16)), int(255 * al * (.5 + .5 * h1(seed, i, 6))), 1, cv2.LINE_AA, 4); continue
            if li == 2 and not near: continue
            if av > .10:
                d = np.array([wind, 1.]); d /= np.hypot(*d); m = np.array([-d[1], d[0]]) * w * .5; p = np.array([x, y]); a = p - d * l; b = p + d * w
                cel.fill(np.array([a, p - d * l * .25 + m, b, p - d * l * .25 - m]), col, al)
                if li == 2: cel.fill(np.array([p - d * l * .5 + m * .5, p - d * l * .1 + m * .7, b, p - d * l * .1]), '#ffffff', al * .8)
            else:                                            # hanging bead
                k = 1 - av / .10; r = size * (1.5, 3.6, 9.5)[li] * (.6 + .9 * h1(seed, li, i, 7) ** 1.5); ex = 1 + 1.6 * (1 - k); wob = 1 + .08 * math.sin(i + tau * 9); a = np.linspace(0, 6.283, 18, endpoint=False); t = np.sin(a); P = np.c_[x + np.cos(a) * r * wob * (1 - .18 * np.maximum(0, -t)), y + t * r * ex / wob]
                if li == 0: cel.fill(P, col, .55); continue
                cel.fill(P, dark, .55 * al + .25); cel.fill(np.c_[x + np.cos(a) * r * .80, y + r * .16 + t * r * ex * .74], col, .55); cel.fill(np.c_[x + np.cos(a) * r * .62 + r * .06, y - r * .02 + t * r * ex * .55], dark, .55)
                cel.fill(np.c_[x - r * .34 + np.cos(a) * r * .20, y - r * .38 + t * r * .26], '#ffffff', .95); cel.fill(np.c_[x + r * .22 + np.cos(a) * r * .34, y + r * .52 + t * r * .16], '#ffffff', .75)
    lay = cel.layer()
    if av > .10: a = far.astype(f32)[..., None] / 255.; lay = lay * (1 - a) + np.dstack([a * C(col), a])
    if light is not None: m = light[..., None] if light.ndim == 2 else light; lay = lay * np.clip(m, 0, 2)
    return (lay * alpha).astype(f32)


def splash(cel, x, y, s, k, seed=0, col='#dfe9fa', shade='#6d7ba0', asp=.36):
    """one drop hitting: k frames 0..7.  ring, crown spikes, beads"""
    if k < 0 or k > 7: return cel
    e = 1 - (1 - k / 7.) ** 2; rx = s * (.25 + .95 * e); w = s * .10 * (1 - k / 7.5)
    for g in range(3):
        a0 = h1(seed, g) * 6.283 + g * 2.1; a = np.linspace(a0, a0 + 1.6, 14); hw = w * np.sin(np.linspace(0, np.pi, 14)); cel.fill(np.vstack([np.c_[x + np.cos(a) * (rx + hw), y + np.sin(a) * (rx + hw) * asp], np.c_[x + np.cos(a) * (rx - hw), y + np.sin(a) * (rx - hw) * asp][::-1]]), col, .85)
    if k < 5:
        h = s * (.95, 1.25, 1.1, .75, .35)[k]; n = 6
        for i in range(n):
            a = math.pi + (i + .5) / n * math.pi; bx = x + math.cos(a) * rx * .55; lean = math.cos(a) * h * .42; hh = h * (.5 + .6 * h1(seed, i, 2)); cel.fill(np.array([[bx - s * .07, y], [bx + lean, y - hh], [bx + s * .07, y]]), col, .8)
    for i in range(5):
        a = -math.pi * (.15 + .7 * h1(seed, i, 3)); v = s * (.30 + .35 * h1(seed, i, 4)); px = x + math.cos(a) * v * k; py = y + math.sin(a) * v * k + .045 * s * k * k * 1.3
        if py < y + s * .1: r = s * .055 * (1.2 - k / 9.); cel.fill(np.array([[px, py - r * 1.4], [px + r, py], [px, py + r], [px - r, py]]), col, .95)
    return cel


def splashes(f, n=60, seed=0, place=None, size=(5, 26), rate=1., tau=None, v=1., **k):
    """field of splashes on the ground.  place(u, w) -> (x, y, scale 0..1) maps randoms to the ground (default: lower third, nearer = bigger).  Freezes / reverses with tau, hidden when v ~ 0"""
    tau = f / 24. if tau is None else tau; cel = Cel(); place = place or (lambda u, w: (u * W, H * (.66 + .34 * w ** .7), .15 + .85 * w))
    if abs(v) < .05: return cel.layer()
    for i in range(n):
        per = (.55 + .5 * h1(seed, i, 1)) / rate; q = tau / per + h1(seed, i, 2); cyc = math.floor(q); fr = int((q - cyc) * per * 24)
        x, y, s = place(h1(seed, i, cyc, 3), h1(seed, i, cyc, 4)); splash(cel, x, y, size[0] + (size[1] - size[0]) * s, fr, seed * 3 + i + cyc * 7, **k)
    return cel.layer()


# ------------------------------------------------------------------ snow
def _flake(x, y, r, seed, rot):
    a = lin01(20)[:-1] * 6.283 + rot; q = r * (1 + .10 * np.sin(a * 3 + seed) + .13 * np.sin(a * 2 - seed * 2) + .10 * np.sin(a * 5 + seed * 3)); return c2(x + np.cos(a) * q, y + np.sin(a) * q * .9)


def snow(f, n=(520, 240, 70, 12), wind=-40., fall=95., seed=0, col='#f6f8ff', shade='#b9c3dd', gust=.6, size=1., tau=None, near=True, op=1.):
    """snow in four depths: dust, small flakes, drawn clumps with a shadow side, big soft near ones.  -> RGBA layer"""
    t = f / 24. if tau is None else tau; pad = 80; far = np.zeros((H, W), np.uint8); cel = Cel(); nc = Cel(W // 2, H // 2, ss=2)
    for li, (cnt, sp, r0) in enumerate(((n[0], .40, .9), (n[1], .75, 2.2), (n[2], 1.25, 5.2), (n[3] if near else 0, 2.3, 17.))):
        for i in range(cnt):
            u = h1(seed, li, i, 1); q = t * fall * sp * (.75 + .5 * u) / (H + 2 * pad) + h1(seed, li, i, 2); cyc = math.floor(q); y = (q - cyc) * (H + 2 * pad) - pad; g = 1 + gust * math.sin(t * .9 + y * .002 + li)
            x = (h1(seed, li, i, cyc, 3) * (W + 400) + wind * sp * g * t + 26 * sp * math.sin(t * (1.1 + u) + i) * 1. + 30 * sp * noise(i * 1.7, t * .7, li, seed)) % (W + 400) - 200; r = r0 * size * (.6 + .8 * h1(seed, li, i, 4)); rot = t * (h1(seed, li, i, 5) - .5) * 3 + i
            if li == 0: cv2.circle(far, (int(x * 16), int(y * 16)), int(r * 16), int(120 + 100 * u), -1, cv2.LINE_AA, 4)
            elif li == 3: P = _flake(x / 2, y / 2, r / 2, i, rot); nc.fill(P, col, .55)
            else:
                P = _flake(x, y, r, i, rot); cel.fill(P, col if li == 1 else shade, .92)
                if li == 2: cel.fill(_flake(x - r * .12, y - r * .16, r * .80, i, rot), col)
    lay = cel.layer(); a = far.astype(f32)[..., None] / 255. * .75; lay = lay + np.dstack([a * C(col), a]) * (1 - lay[..., 3:4])
    if near and n[3]: L = cv2.filter2D(nc.layer(H // 2, W // 2), -1, disc_kernel(3.6)); L = cv2.resize(L, (W, H), interpolation=cv2.INTER_LINEAR); lay = lay * (1 - L[..., 3:4]) + L
    return (lay * op).astype(f32)


# ------------------------------------------------------------------ tumbling flat things
def _tumble(shape, x, y, s, rx, ry, rz, bend=.0):
    P = np.c_[shape, bend * (shape[:, 0] ** 2 - .3)]; M = _rotm(rx, ry, rz); Q = P @ M.T; k = s / (1 - .18 * Q[:, 2]); return np.c_[Q[:, 0] * k + x, Q[:, 1] * k + y], M[2, 2]


SAKURA = spl(np.array([[0, .95], [-.34, .62], [-.58, .05], [-.46, -.55], [-.16, -.86], [0, -.66], [.16, -.86], [.46, -.55], [.58, .05], [.34, .62]]), True, corners=(0, 4, 5, 6), n=4)
HIGAN = spl(np.array([[0, 1], [-.12, .6], [-.16, .1], [-.11, -.45], [.02, -1], [.12, -.4], [.17, .1], [.11, .6]]), True, corners=(0, 4), n=4)
ASHF = np.array([[-.9, -.1], [-.5, -.5], [.1, -.35], [.6, -.6], [.95, 0], [.4, .25], [.2, .6], [-.4, .35]])
PINK = dict(lit='#ffe3ea', mid='#f6b3c4', dark='#cf7f9c', ink='#8a4460')
BLACKPETAL = dict(lit='#2a1a20', mid='#0a0608', dark='#000000', ink='#f0142e')
ASHPAL = dict(lit='#b7adab', mid='#6f6668', dark='#38333a', ink='#1e1a20')


def petals(f, n=90, seed=0, pal=PINK, shape=SAKURA, wind=(-120., 70.), size=(8, 30), box=(-100, -100, W + 100, H + 100), spin=2.4, sway=60., step=1, near=5, lw=1.2, origin=None, burst=0., edge=None, tau=None, depth=True, op=1.):
    """drawn petals tumbling in real 3-D (two-tone faces, thin ink), small far ones to a few big defocused near ones.  origin = thrown from a point with `burst` px/s.  edge = coloured rim (black higanbana petals).  -> RGBA layer"""
    t = (int(f) // step * step) / 24. if tau is None else tau; cel = Cel(); nc = Cel(); x0, y0, x1, y1 = box; bw, bh = x1 - x0, y1 - y0; it = []
    for i in range(n):
        z = h1(seed, i, 1); dep = .35 + .65 * z if depth else 1.; s = (size[0] + (size[1] - size[0]) * z ** 1.6); isn = i >= n - near
        if isn: s = size[1] * (2.0 + 1.6 * h1(seed, i, 2)); dep = 1.9
        if origin is None:
            x = x0 + (h1(seed, i, 3) * bw + wind[0] * dep * t + sway * dep * math.sin(t * (.9 + h1(seed, i, 4)) + i)) % bw; y = y0 + (h1(seed, i, 5) * bh + wind[1] * dep * t * (.7 + .6 * h1(seed, i, 6)) + sway * .4 * dep * math.sin(t * 1.7 + i * 2)) % bh; a = 1.
        else:
            th = h1(seed, i, 3) * 6.283; v = burst * (.25 + 1.0 * h1(seed, i, 5)); d = v * (1 - math.exp(-2.2 * t)) / 2.2; x = origin[0] + math.cos(th) * d + wind[0] * dep * t + sway * math.sin(t * 1.3 + i) * min(1, t); y = origin[1] + math.sin(th) * d + wind[1] * dep * t
        w = spin * (.4 + 1.1 * h1(seed, i, 7)); it.append((dep, x, y, s, t * w + i, t * w * .7 + i * 2.2, t * w * .35 * (1 if i % 2 else -1) + i * 1.3, i, isn))
    it.sort(key=lambda q: q[0])
    for dep, x, y, s, rx, ry, rz, i, isn in it:
        if not (-80 < x < W + 80 and -80 < y < H + 80): continue
        c = nc if isn else cel; P, nz = _tumble(shape, x, y, s, rx, ry, rz, .35 * h1(seed, i, 8)); fc = abs(nz)
        if edge: c.fill(P + (P - [x, y]) * (2.2 / max(s, 4)) + [.8, 1.0], edge)
        c.fill(P, pal['lit'] if fc > .55 and nz > 0 else pal['mid'] if fc > .22 else pal['dark'])
        if s > 9 and not isn:
            m = len(P) // 2; c.fill(np.vstack([P[m:], P[:1] * .5 + P[m:m + 1] * .5 + (P[m + m // 2] - [x, y]) * .25]), pal['mid'] if nz > 0 else pal['dark']); thin(c, np.array([P[0], [x, y], P[m]]), lw, pal['ink'], .7, .3, .6, i, 0, 10)
            if lw and not edge: c.line(np.vstack([P, P[:1]]), lw, pal['ink'], taper=(.2, .4), seed=i, alpha=.8)
    lay = cel.layer()
    if near: L = nc.layer(); L = cv2.GaussianBlur(L, (0, 0), 5.5); lay = lay * (1 - L[..., 3:4]) + L
    return (lay * op).astype(f32)


def ash(f, n=90, seed=0, wind=(-60., -26.), size=(2, 11), glow=.25, pal=ASHPAL, **k):
    """floating ash: ragged curled flakes; some still burn at the rim (FX light)"""
    lay = petals(f, n, seed, pal, ASHF, wind, size, spin=1.6, sway=50., lw=0., near=k.pop('near', 4), **k)
    if not glow: return FX(lay, None)
    L = np.zeros((H // 2, W // 2, 3), f32); t = f / 24.
    for i in range(int(n * glow)):
        x = (h1(seed, i, 41) * (W + 200) + wind[0] * t * (.5 + h1(seed, i, 42))) % (W + 200) - 100; y = (h1(seed, i, 43) * (H + 200) + wind[1] * t * (.6 + h1(seed, i, 44))) % (H + 200) - 100; X, Y = int(x / 2), int(y / 2); k2 = 4; fl = .4 + .6 * abs(math.sin(t * 5 + i))
        if k2 < X < W // 2 - k2 and k2 < Y < H // 2 - k2: yy, xx = np.mgrid[-k2:k2 + 1, -k2:k2 + 1]; L[Y - k2:Y + k2 + 1, X - k2:X + k2 + 1] += np.exp(-(xx ** 2 + yy ** 2) / 2.6)[..., None].astype(f32) * C('#ff6a1e') * fl
    return FX(lay, cv2.resize(L, (W, H), interpolation=cv2.INTER_LINEAR))


def motes(f, n=120, seed=0, light=None, col='#fff0cf', drift=(6., -5.), size=1., amt=.8):
    """dust in light: specks that brighten inside `light` (H,W mask) and turn; near ones soft.  -> additive RGB"""
    t = f / 24.; L = np.zeros((H // 2, W // 2), f32)
    for i in range(n):
        z = h1(seed, i, 1); x = (h1(seed, i, 2) * W + drift[0] * t * (.4 + z) + 22 * noise(i * 1.3, t * .25, 0, seed)) % W; y = (h1(seed, i, 3) * H + drift[1] * t * (.4 + z) + 22 * noise(i * 2.7, t * .25, 5, seed)) % H; r = size * (.6 + 2.6 * z ** 5) ; b = (.25 + .75 * abs(math.sin(t * (.8 + z) + i * 2.1))) * (.35 + .65 * z)
        if light is not None: b *= float(light[int(y), int(x)])
        k = int(r * 3 + 2); X, Y = int(x / 2), int(y / 2)
        if b > .02 and k < X < W // 2 - k and k < Y < H // 2 - k: yy, xx = np.mgrid[-k:k + 1, -k:k + 1]; L[Y - k:Y + k + 1, X - k:X + k + 1] += (np.exp(-(xx ** 2 + yy ** 2) / (r * r * .5 + .25)) * b).astype(f32)
    return cv2.resize(L, (W, H), interpolation=cv2.INTER_LINEAR)[..., None] * C(col) * amt


def mist(f, seed=0, y0=560., y1=1080., col='#c9d4ea', amt=.55, speed=26., bands=6, scale=340., hard=.18):
    """drifting mist: layered flat banks with a torn upper edge that softens downward.  -> RGBA layer"""
    t = f / 24.; h, w = H // 2, W // 2; y, x = np.mgrid[0:h, 0:w].astype(f32) * 2; A = np.zeros((h, w), f32); xs = np.arange(w) * 2.
    for b in range(bands):
        k = b / max(1, bands - 1); yb = y0 + (y1 - y0) * (k * .8 - .05); n = noise(xs / scale - t * speed * (.5 + k) / scale, b * 7.7, t * .12, seed) + .45 * noise(xs / scale * 2.7 - t * speed * (.5 + k) / scale * 2.2, b * 3.1, t * .2, seed) + .2 * noise(xs / scale * 7, b, t * .3, seed)
        top = (yb + n * 62 * (.5 + k))[None, :].astype(f32); d = (y - top); bank = smooth(-hard * 120, hard * 120, d) * np.exp(-np.clip(d, 0, 1e4) / (110 + 170 * k)); A = 1 - (1 - A) * (1 - bank * .34)
    A = cv2.resize(A, (W, H), interpolation=cv2.INTER_LINEAR) * amt; return np.dstack([A[..., None] * C(col), A])
