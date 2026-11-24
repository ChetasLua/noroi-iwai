"""fx fire: drawn flat-shape flames (yellow / orange / red, thin dark-red flow lines), streams, arrow, column, embers."""
from .fx_core import *

PAL = dict(rim='#d8290a', body='#f2650e', amber='#fb9c22', hot='#ffe92e', core='#fffab0', ink='#641608', dark='#2a0507', maroon='#5c0c0a')
BLUEPAL = dict(rim='#12209c', body='#1f55e6', amber='#33a4ff', hot='#8ef0ff', core='#f2ffff', ink='#0b1150', dark='#05071c', maroon='#0a0f48')


# ------------------------------------------------------------------ one stream along a path
def flame_stream(cel, P, w, t=0., seed=0, pal=PAL, prof=None, speed=520., pitch=None, lick=.62, fringe=True, islands=1.0, ink=True, slivers=True,
                 lw=1.5, dens=1.0, core=False, undul=.30, smooth=.5):
    """fire running along path P (start -> tip).  w = half width px at the belly, t seconds, speed px/s of the licks.
    Yellow outer shape with swept licks and broad undulation, long orange islands, amber folds, yellow veins, thin broken ink, torn-off needles.
    smooth 0 = nervous spikes, 1 = long flowing S-contours with big orange masses (the Fuga look)."""
    p = Path(P, 5.); L = p.L; pitch = pitch or max(60., w * (1.4 + .9 * smooth)); sh = speed * t; lick = lick * (1 - .40 * smooth); undul = undul * (1 + .55 * smooth)
    prof = prof or (lambda v: leaf(v, .22, .55, .9))
    base = lambda v: w * prof(v)
    und = lambda v, q: 1 + undul * noise(np.asarray(v) * L / (w * 3.4) - sh / (w * 3.4), q * 5.3, t * .8, seed)
    hw = (lambda v: base(v) * und(v, 1), lambda v: base(v) * und(v, 2))
    env = lambda v: .30 + .85 * np.sin(np.pi * np.clip(v, 0, 1)) ** .7
    if fringe:
        F = band(p, (lambda v: hw[0](v) * 1.05 + 4, lambda v: hw[1](v) * 1.05 + 4), seed + 3, sh * .9, pitch * .9, lick * 1.1, 1.8, t, env); cel.fill(F, pal['rim'], 1 - .35 * smooth)
    body = band(p, hw, seed, sh, pitch, lick, 1.7, t, env, wave=.04)
    cel.fill(body, pal['hot'])
    cel.clip(body)
    def island(s0, ln, off, hh, kk, fold=True):
        a0, a1 = max(s0, L * .015), min(s0 + ln, L * .985)
        if a1 - a0 < hh * 2.2 or a1 - a0 < 22: return
        cut0, cut1 = (a0 - s0) / ln, (a1 - s0) / ln; ph = h1(seed, kk, 15) * 6.283; m = L * 1e-9 + 1
        mid = lambda v: base((a0 + (a1 - a0) * np.asarray(v)) / L)
        o = lambda v: off * mid(v) + mid(v) * (.16 + .20 * smooth) * np.sin(np.asarray(v) * (2.4 + ln / w * (.35 + .35 * smooth)) - t * 6 + ph)
        q = p.sub(a0, a1, o, max(18, int((a1 - a0) / 14)))
        wd = lambda v: np.minimum(hh, mid(v) * (.62 + .12 * smooth)) * leaf(cut0 + (cut1 - cut0) * np.asarray(v), .34, .62 - .12 * smooth, .95 - .2 * smooth)
        isl = band(q, (lambda v: wd(v) * (1 + (.22 + .30 * smooth) * noise(np.asarray(v) * ln / w * .75, 1, t, kk)), lambda v: wd(v) * (1 + (.22 + .30 * smooth) * noise(np.asarray(v) * ln / w * .75, 5, t, kk))),
                   seed * 7 + kk, sh * .2, max(40, hh * (1.8 + 1.4 * smooth)), .50 - .26 * smooth, 1.6, t, None, (1, .8), .06)
        c = pal['body'] if h1(seed, kk, 16) < .78 else pal['amber']
        cel.fill(isl, c)
        if ink: ink_broken(cel, isl, lw * .9, pal['ink'], seed * 3 + kk, .72, (60, 260))
        if not fold or hh < 8: return
        pp = Path(q, 5.)
        for j in range(1 + int(ln / (hh * 7))):
            if h1(seed, kk, j, 17) > .85: continue
            b0 = pp.L * (.06 + .8 * (j + .2 * h1(seed, kk, j, 18)) / (1 + int(ln / (hh * 7)))); b1 = min(pp.L * .97, b0 + hh * (2.6 + 2.4 * h1(seed, kk, j, 19)))
            if b1 - b0 < 24: continue
            sg = 1 if h1(seed, kk, j, 20) > .5 else -1
            q2 = pp.sub(b0, b1, lambda v: sg * hh * .18 + hh * .12 * np.sin(v * 5 + ph), 20); h2 = hh * (.30 + .22 * h1(seed, kk, j, 21))
            f2 = band(q2, (lambda v: h2 * leaf(v, .3, .7, 1.), lambda v: h2 * .8 * leaf(v, .5, .7, .8)), seed + kk + j, 0, max(28, h2 * 1.6), .42, 1.5, t, None, (1, 1), .12)
            cel.fill(f2, pal['amber'] if c == pal['body'] else pal['hot'])
            if ink:
                ink_broken(cel, f2, lw * .85, pal['ink'], seed + kk * 5 + j, .85, (70, 240))
                if smooth > .3 and h2 > 9: ink_broken(cel, q2.mean(0) + (f2 - q2.mean(0)) * [.80, .80], lw * .75, pal['ink'], seed + kk + j, .7, (90, 300))
                pq = Path(q2, 5.)
                for m, o2 in enumerate((.34, -.10, -.42)):
                    if h1(seed, kk, j, 22 + m) < .62: cel.line(pq.sub(pq.L * (.06 + .12 * m), pq.L * (.70 + .25 * h1(seed, kk, m)), lambda v: o2 * h2 * (1 - .7 * v) + h2 * .10 * np.sin(v * 5 + ph)), lw * .8, pal['ink'], taper=(.25, .4), seed=kk + m, tremor=.3)
    cm = w * 5.2 / dens                                    # long central islands
    for k in range(int(math.floor(-sh * .95 / cm)) - 2, int((L - sh * .95) / cm) + 2):
        if h1(seed, 1, k) < .93 * islands: island((k + .25 * h1(seed, 2, k)) * cm + sh * .95, cm * (.75 + .5 * h1(seed, 3, k)), (h1(seed, 4, k) - .5) * (.5 + .35 * smooth), w * (.34 + .16 * h1(seed, 5, k) + .13 * smooth), k * 5 + 1)
    cs = w * 2.4 / dens                                    # small side islands
    for k in range(int(math.floor(-sh / cs)) - 2, int((L - sh) / cs) + 2):
        if h1(seed, 6, k) < .55 * islands:
            sg = 1 if h1(seed, 7, k) > .5 else -1
            island((k + .7 * h1(seed, 8, k)) * cs + sh, cs * (.55 + .7 * h1(seed, 9, k)), sg * (.46 + .2 * h1(seed, 10, k)), w * (.10 + .12 * h1(seed, 11, k)), k * 5 + 3, False)
    cv_ = w * 3.6                                          # yellow veins crossing the orange
    for k in range(int(math.floor(-sh / cv_)) - 2, int((L - sh) / cv_) + 2):
        if h1(seed, 12, k) < .7:
            s0 = (k + .6 * h1(seed, 13, k)) * cv_ + sh; s1 = s0 + w * (2.0 + 1.6 * h1(seed, 14, k)); sg = 1 if h1(seed, 15, k) > .5 else -1
            if s0 < L * .03 or s1 > L * .97: continue
            q = p.sub(s0, s1, lambda v: sg * float(base(np.array([s0 / L]))[0]) * (.75 - 1.15 * v ** .8) + w * .06 * np.sin(v * 7 + k), 22)
            hv = w * (.06 + .09 * h1(seed, 16, k)); vein = band(q, lambda v: hv * leaf(v, .12, .4, .9), seed + k, 0, 40, .7, 1.6, t, None, (1, 1), .2)
            cel.fill(vein, pal['hot'])
            if ink and h1(seed, 17, k) < .6: ink_broken(cel, vein, lw * .75, pal['ink'], seed + k * 9, .5, (40, 120))
    cel.unclip()
    if core:
        q = p.sub(L * .04, L * .92, lambda v: w * .10 * np.sin(v * 7 - t * 9))
        cel.fill(band(q, lambda v: w * .20 * prof(v), seed + 41, sh, pitch * .8, .5, 1.5, t, env), pal['core'])
    if slivers:
        cell = w * 2.4
        for k in range(int(math.floor(-sh * 1.15 / cell)) - 1, int((L * 1.12 - sh * 1.15) / cell) + 2):
            for sg in (1, -1):
                if h1(seed, k, sg, 51) > .46: continue
                s0 = (k + h1(seed, k, sg, 52)) * cell + sh * 1.15; z = bms(h1(seed, k, sg, 53)); ln = w * (.35 + 1.7 * z); v = np.clip(s0 / L, 0, 1)
                if s0 < L * .12 or s0 > L * 1.06: continue
                n = sg * (float(base(np.array([min(v, 1.)]))[0]) * 1.12 + w * (.10 + .6 * h1(seed, k, sg, 54)) * (v + .15))
                q = p.sub(s0, s0 + ln, lambda v: n + sg * w * .16 * v + w * .05 * np.sin(v * 4 + k), 14); hs = w * .10 * z + 1.3
                cel.fill(band(q, lambda v: hs * 1.2 * leaf(v, .35) + 1.2 * (v < .98), seed + k, 0, 40, .4), pal['body'])
                cel.fill(band(q, lambda v: hs * leaf(v, .35), seed + k, 0, 40, .4), pal['hot'])
    return cel


# ------------------------------------------------------------------ standing flames
def _tongue_spine(x, y, L, th, t, seed, sway=.16, waves=.95, hz=2.1, n=22):
    """S-curved centre line of a tongue rooted at (x, y), heading th (radians, -pi/2 = up); wave travels toward the tip"""
    v = lin01(n); ph = h1(seed, 3) * 6.283
    off = sway * L * (v ** 1.25) * np.sin(6.283 * (waves * v - hz * t) + ph) + .05 * L * v * v * math.sin(1.7 * t * 6.283 * .5 + ph * 2)
    c, s = math.cos(th), math.sin(th)
    return c2(x + c * v * L - s * off, y + s * v * L + c * off)


def _tongue_shapes(x, y, h, b, th, t, seed, hz=1.5, sway=.16, pinch=True, waves=.95, big=1.):
    """life of one flame tongue -> list of (spine, half-width fn, seed, v0) pieces: the root tongue and its torn-off tip"""
    tau = (t * hz + h1(seed, 9)) % 1.0; cyc = int(math.floor(t * hz + h1(seed, 9))); sd = seed * 31 + cyc % 97
    out = []
    grow = min(1., tau / .55); grow = 1 - (1 - grow) ** 2
    H = h * (.34 + .66 * grow) if tau < .58 else h * (.62 - .28 * (tau - .58) / .42)
    sp = _tongue_spine(x, y, H, th, t, seed, sway, waves)
    asym = 1 if h1(sd, 4) > .5 else -1
    rd = lambda v: np.sqrt(np.clip(.25 + np.asarray(v) / .10, 0, 1)); pl = lambda v, b=b: b * tongue(v, .24, 1.0) * rd(v); pr = lambda v, b=b: b * tongue(v, .34, 1.75) * rd(v)
    out.append((sp, (pl, pr) if asym > 0 else (pr, pl), sd))
    if pinch and tau > .52 and big > .45:
        k = (tau - .52) / .48; full = _tongue_spine(x, y, h * (1.12 + .42 * k), th, t, seed, sway, waves, n=40)
        pf = Path(full, 4.); a = .56 + .22 * k; e = min(.99, a + (.46 - .10 * k) * (1 - k * .55))
        q = pf.sub(pf.L * a, pf.L * e, 0, 20); bb = b * .42 * (1 - k) ** .7
        if bb > 1.2 and len(q) > 4: out.append((q, (lambda v: bb * leaf(v, .30, .55, 1.1), lambda v: bb * .9 * leaf(v, .42, .7, .9)), sd + 5))
    return out


def flames(cel, base, h, t=0., seed=0, n=None, pal=PAL, hot=False, spread=.10, b=.20, hz=1.5, sway=.15, ink=True, lw=1.5, rows=2, lean=0., heights=None, lick=.55,
           up=-math.pi / 2, dens=1.0, waves=.95, core=True):
    """standing fire along a base polyline.  h = tallest tongue px.  hot=False: red rim, orange body, yellow core (classic);
    hot=True: yellow outside, orange islands inside (ref fuga look).  All layers of all tongues union per colour."""
    p = Path(np.asarray(base, float), 6.); L = p.L; n = n or max(3, int(L / (h * .20) * dens))
    tg = []
    for row in range(rows):
        bk = (rows - 1 - row) / max(1, rows - 1) if rows > 1 else 0.
        for i in range(n):
            u = (i + .5 + .55 * (h1(seed, row, i, 1) - .5) + .5 * (row % 2)) / n
            if u < 0 or u > 1: continue
            z = (.22 + .82 * bms(h1(seed, row, i, 2), .24, .36)) * (1 - .22 * (1 - bk))
            if heights is not None: z *= float(heights(u))
            x, y = p.at(u * L); th = up + lean + spread * (2 * u - 1) + .10 * (h1(seed, row, i, 3) - .5)
            bb = h * b * (.42 + .75 * z) * (.8 + .4 * h1(seed, row, i, 4))
            tg.append((bk, x, y, h * z, bb, th, seed * 101 + row * 37 + i))
    tg.sort(key=lambda q: -q[0])
    def layer(col, kw, kh, hzph=0., edge=None, off=0.):
        polys = []
        for bk, x, y, hh, bb, th, sd in tg:
            for j, (sp, (pl, pr), s2) in enumerate(_tongue_shapes(x, y + off * bb, hh * kh, bb * kw, th, t + hzph, sd, hz * (.8 + .5 * h1(sd, 8)), sway, True, waves, hh / h)):
                P = band(sp, (pl, pr), s2 + int(kw * 10), hh * .55 * ((t + hzph) % 4.), max(24., hh * .24), lick if j == 0 else lick * .6, 1.4, t, lambda v: np.sin(np.pi * np.clip(v, 0, 1)) ** .8, (1, 1), .07)
                polys.append(P)
        for P in polys: cel.fill(P, col)
        return polys
    if not hot:
        layer(pal['rim'], 1.16, 1.05, -.02); A = layer(pal['body'], 1.0, .93); B = layer(pal['hot'], .62, .70, .01, off=.12)
        if core: layer(pal['core'], .30, .44, .02, off=.25)
    else:
        layer(pal['rim'], 1.10, 1.04, -.02); A = layer(pal['hot'], 1.0, .97); B = layer(pal['body'], .60, .74, .01, off=.15)
        layer(pal['amber'], .34, .52, .02, off=.2)
    if ink:
        for k, P in enumerate(B):
            if len(P) > 30: ink_broken(cel, P[len(P) // 8:-len(P) // 8], lw, pal['ink'], seed + k * 3, .5, (40, 150))
        for bk, x, y, hh, bb, th, sd in tg:
            for j in range(2):
                if h1(sd, j, 61) > .5: continue
                sp = _tongue_spine(x + (h1(sd, j, 62) - .5) * bb * .9, y, hh * (.35 + .35 * h1(sd, j, 63)), th, t, sd, sway, waves, n=20)
                cel.line(sp[4:], lw * .85, pal['ink'], taper=(.4, .55), seed=sd + j, tremor=.3)
    return cel


def fire_core(poly, t=0., seed=0, pal=PAL, h=H, w=W, soft=3.2, petals=13, size=None):
    """black-red burning core (a fist, a wound, a coal): soft dark maroon heart with a jagged top, red petals with ink around it. -> RGBA layer"""
    P = np.asarray(poly, float); c = P.mean(0); rx, ry = (P.max(0) - P.min(0)) / 2; s = size or max(rx, ry)
    cel = Cel(w, h)
    r = R(seed, 1)
    q = Path(np.vstack([P, P[:1]]), 4.); n = petals
    for i in range(n * 2):
        u = ((i + h1(seed, i, 1) * .8) / (n * 2)) * q.L; pt = q.at(u); d = pt - c; up = -d[1] / s
        ln = s * (.24 + .55 * max(0, up)) * (.5 + bms(h1(seed, i, int(t * 12), 2))); th = math.atan2(d[1], d[0]); th = th + (-math.pi / 2 - th if d[1] > 0 and abs(d[0]) < 0 else 0)
        th = math.atan2(d[1] * .5 - s * .75, d[0] * 1.0)
        sp = _tongue_spine(pt[0] - d[0] * .12, pt[1] - d[1] * .12, ln, th, t, seed * 7 + i, .14)
        P2 = band(sp, lambda v: s * .17 * tongue(v), seed + i, 0, 30, .5); cel.fill(P2, pal['rim'])
        if h1(seed, i, 6) < .55: ink_broken(cel, P2, 1.4, pal['ink'], seed + i, .7, (30, 120))
    o = P + (P - c) * .10; cel.fill(o, pal['rim'])
    lay = cel.layer(h, w)
    k = Cel(w, h)
    inner = c + (P - c) * .84; top = []
    k.fill(inner, pal['maroon'])
    for i in range(11):
        x = c[0] + (i / 10 * 2 - 1) * rx * .78; ln = ry * (.30 + .5 * h1(seed, i, 21, int(t * 12)))
        k.fill(np.array([[x - rx * .16, c[1]], [x + rx * .05 * (h1(i, seed) - .5), c[1] - ry * .55 - ln], [x + rx * .16, c[1]]]), pal['maroon'])
    k2 = c + (P - c) * .62 + [0, ry * .1]
    d = cv2.GaussianBlur(k.layer(h, w), (0, 0), soft)
    k = Cel(w, h); k.fill(k2, pal['dark']); d2 = cv2.GaussianBlur(k.layer(h, w), (0, 0), soft * 2.4)
    lay = lay * (1 - d[..., 3:4]) + d; lay = lay * (1 - d2[..., 3:4]) + d2
    k = Cel(w, h)
    for i in range(3):
        x = c[0] + (h1(seed, i, 31) - .5) * rx * 1.1; y = c[1] + (h1(seed, i, 32) - .6) * ry * .7
        k.fill(band(_tongue_spine(x, y, s * .3, -math.pi / 2, t, seed + i, .1), lambda v: s * .035 * leaf(v), seed, 0, 30, .3), pal['rim'])
    e = k.layer(h, w); return lay * (1 - e[..., 3:4]) + e


# ------------------------------------------------------------------ embers
def embers(f, n=70, seed=0, box=(0, 0, W, H), wind=(26., -150.), pal=PAL, near=.16, size=1.0, origin=None, spread=300., life=2.2, swirl=40., bokeh=True, gain=1.0):
    """sparks + bokeh: tumbling slivers with a hot core, defocused near ones as uneven discs.  -> FX (all additive light + small cores)"""
    cel = Cel(); L = np.zeros((H // 2, W // 2, 3), f32); t = f / 24.
    x0, y0, x1, y1 = box; bw, bh = x1 - x0, y1 - y0
    for i in range(n):
        z = h1(seed, i, 1); dep = z ** 2.2; isn = z > 1 - near
        sp = .55 + 1.1 * dep; lf = life * (.6 + .8 * h1(seed, i, 2)); tau = (t / lf + h1(seed, i, 3)) % 1.; cyc = int(t / lf + h1(seed, i, 3))
        if origin is None: px, py = x0 + h1(seed, i, cyc, 4) * bw, y0 + h1(seed, i, cyc, 5) * bh * 1.15
        else:
            a = h1(seed, i, cyc, 4) * 6.283; q = spread * math.sqrt(h1(seed, i, cyc, 5)); px, py = origin[0] + math.cos(a) * q, origin[1] + math.sin(a) * q * .5
        tt = tau * lf
        x = px + wind[0] * sp * tt + swirl * noise(i * 3.3, tt * .9, 0, seed) * (1 + tt); y = py + wind[1] * sp * tt + swirl * .6 * noise(i * 5.1, tt * 1.1, 4, seed)
        fade = min(1, tau / .12) * min(1, (1 - tau) / .35) * (.62 + .38 * math.sin(f * 1.7 + i * 2.1))
        if fade <= .02 or not (-80 < x < W + 80 and -80 < y < H + 80): continue
        if isn and bokeh:
            rr = size * (10 + 34 * h1(seed, i, 6)); m = 18; a = np.linspace(0, 6.283, m, endpoint=False)
            rad = rr * (1 + .05 * np.cos(a * 5 + i) + .03 * np.sin(a * 3 + i)); asp = .72 + .28 * h1(seed, i, 7); rot = h1(seed, i, 8) * 3.14
            P = np.c_[np.cos(a) * rad, np.sin(a) * rad * asp]; P = np.c_[P[:, 0] * math.cos(rot) - P[:, 1] * math.sin(rot) + x, P[:, 0] * math.sin(rot) + P[:, 1] * math.cos(rot) + y]
            bx = int(max(0, (x - rr * 2.4) / 2)); by = int(max(0, (y - rr * 2.4) / 2)); ex = int(min(W // 2, (x + rr * 2.4) / 2 + 2)); ey = int(min(H // 2, (y + rr * 2.4) / 2 + 2))
            if ex <= bx or ey <= by: continue
            m8 = np.zeros((ey - by, ex - bx), np.uint8); cv2.fillPoly(m8, [np.round((P / 2 - [bx, by]) * 16).astype(np.int32)], 255, cv2.LINE_AA, 4); mm = m8.astype(f32) / 255
            g = cv2.GaussianBlur(mm, (0, 0), rr * .06 + .8); halo = cv2.GaussianBlur(mm, (0, 0), rr * .42)
            yy, xx = np.mgrid[by:ey, bx:ex]; d = np.sqrt((xx * 2 - x) ** 2 + (yy * 2 - y) ** 2) / rr; c0 = np.exp(-(d / .55) ** 2)
            col = C(pal['amber']) * 1.0; L[by:ey, bx:ex] += (g[..., None] * col * (.30 + .10 * h1(i, seed)) + halo[..., None] * C(pal['rim']) * .34 + (g * c0)[..., None] * C(pal['core']) * .55) * fade * gain
        else:
            vx = wind[0] * sp; vy = wind[1] * sp; an = math.atan2(vy, vx) + .7 * noise(i * 2.2, t * 2.0, 8, seed); ln = size * (3 + 11 * dep) * (1 + .5 * h1(seed, i, 9)); ww = size * (1.0 + 2.6 * dep) * (.35 + .65 * abs(math.sin(f * .45 + i)))
            c, s = math.cos(an), math.sin(an); q = np.array([[-ln, 0], [-ln * .1, -ww], [ln, 0], [0, ww]]); P = np.c_[q[:, 0] * c - q[:, 1] * s + x, q[:, 0] * s + q[:, 1] * c + y]
            cel.fill(P, pal['amber'] if fade < .6 else pal['core'], min(1., fade * 1.2))
            if fade > .25:
                k = int(6 + ln * 1.6); X, Y = int(x / 2), int(y / 2)
                if k < X < W // 2 - k and k < Y < H // 2 - k:
                    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]; L[Y - k:Y + k + 1, X - k:X + k + 1] += np.exp(-(xx ** 2 + yy ** 2) / (k * .33) ** 2)[..., None].astype(f32) * C(pal['rim']) * .55 * fade * gain
    return FX(cel.layer(), cv2.resize(L, (W, H), interpolation=cv2.INTER_LINEAR))


# ------------------------------------------------------------------ ground flames, burst ring
def flame_ring(cel, cx, cy, rx, ry, h, t, seed=0, pal=PAL, n=34, front=True, back=True, out=.55, lw=1.4, hot=True, hz=1.6):
    """skirt of flames thrown outward round an ellipse on the ground (perspective): far side first, near side last; big / medium / small tongues with their own lives, torn tips"""
    it = []; rd = lambda v: np.sqrt(np.clip(.02 + np.asarray(v) / .16, 0, 1))
    for i in range(n):
        a = (i + .9 * h1(seed, i, 1)) / n * 6.283; it.append((math.sin(a), a, i))
    it.sort()
    for s, a, i in it:
        if (s < 0 and not back) or (s >= 0 and not front): continue
        k = .72 + .5 * h1(seed, i, 7); x, y = cx + math.cos(a) * rx * k, cy + s * ry * k; per = 1 + .50 * s; z = bms(h1(seed, i, 2), .20, .34); th = math.atan2(min(-.30, -1 + out * s * .9 * h1(seed, i, 8)), math.cos(a) * out * 1.8 + .35 * (h1(seed, i, 3) - .5)); hh = h * per * (.22 + .95 * z); b = hh * (.13 + .10 * h1(seed, i, 4))
        for j, (sp, (pl, pr), sd) in enumerate(_tongue_shapes(x, y + 4, hh, b, th, t, seed * 13 + i, hz * (.8 + .5 * h1(seed, i, 9)), .19, True, 1.0)):
            e = 1.5 + 3.5 * h1(sd, 5); sx = (h1(sd, 6) - .5) * e * 1.6; pit = max(24., hh * .26)
            cel.fill(band(sp, (lambda v: pl(v) * 1.08 + e * rd(v), lambda v: pr(v) * 1.08 + e * rd(v)), sd + 4, t * hh * 2, pit, .75, t=t) + [sx, -e * .4], pal['rim']); P = band(sp, (pl, pr), sd, t * hh * 2, pit, .75, 1.5, t); cel.fill(P, pal['hot'] if hot else pal['body'])
            if j == 0 and b > 4:
                q = Path(sp, 4.); q2 = q.sub(q.L * .04, q.L * (.60 + .18 * h1(sd, 7)), lambda v: b * .25 * (h1(sd, 8) - .5) * 2 * (1 - v)); P2 = band(q2, lambda v: b * .56 * tongue(v) * rd(v), sd + 50, t * hh * 2, max(20., hh * .2), .6, t=t); cel.fill(P2, pal['body'] if hot else pal['hot'])
                if h1(sd, 9) < .6: ink_broken(cel, P2, lw, pal['ink'], sd, .6, (30, 120))
                if b > 12: q3 = Path(q2, 4.).sub(0, poly_len(q2) * .6, 0, 14); P3 = band(q3, lambda v: b * .22 * tongue(v) * rd(v), sd + 70, t * hh, max(18., hh * .18), .5, t=t); cel.fill(P3, pal['amber']); ink_broken(cel, P3, lw * .85, pal['ink'], sd + 1, .7, (30, 100))
    return cel


# ------------------------------------------------------------------ arrow, column
def fire_arrow(cel, tail, tip, w, t, seed=0, pal=PAL, spread=.62, arm=.85, arms=True, shaft=True, head=1.0, lick=.62, lw=1.5, bend=0., speed=760.):
    """Fuga: an arrow of fire.  All streams start at the tip and blow back: two bow-wave arms (angle +-spread, length arm * shaft), the shaft, needles.  w = shaft half width px, bend bows it"""
    tail, tip = np.asarray(tail, float), np.asarray(tip, float); d = tail - tip; L = np.hypot(*d); d = d / L; m = np.array([-d[1], d[0]]); rot = lambda a: d * math.cos(a) + m * math.sin(a)
    if shaft: flame_stream(cel, spl([tip, tip + d * L * .35 + m * L * bend * .6, tip + d * L * .7 + m * L * bend, tail], n=16), w, t, seed + 1, pal, lambda v: leaf(v, .12, .7, .75), speed, None, lick, lw=lw, smooth=1.)
    if arms:
        for sg in (1, -1):
            a = spread * sg; l = L * arm * (1. if sg > 0 else .92); k = rot(a); out = rot(sg * (spread + .75)); q = spl([tip, tip + k * l * .30 - out * l * .02, tip + k * l * .64 + out * l * .05, tip + k * l * .90 + out * l * .20, tip + k * l + out * l * .32], n=16); flame_stream(cel, q, w * .95 * head, t + .31 * sg, seed + 5 + sg, pal, lambda v: leaf(v, .30, .62, .8), speed * 1.1, None, lick * 1.1, lw=lw, smooth=1., dens=1.15)
        hd = np.array([tip - d * w * .9, tip + rot(spread * .72) * w * 3.2, tip + d * w * 2.6, tip + rot(-spread * .72) * w * 3.2]); cel.fill(spl(hd, True, corners=(0, 1, 2, 3), n=4), pal['hot'])
        c = Path(np.array([tip + d * w * .4, tip + d * w * 1.6, tip + d * w * 4.2]), 4.); isl = band(c, lambda v: w * .42 * leaf(v, .30, .7, 1.), seed + 9, speed * t * .3, 110, .28); cel.fill(isl, pal['body']); ink_broken(cel, isl, lw, pal['ink'], seed, .8)
    return cel


def fire_column(f, cx=1080., y0=930., y1=-140., w=190., seed=0, pal=PAL, step=2, grow=1.0, lean=0., tongues=1.0, sweeps=12, ring=True, speed=1300., lw=1.8, smoke=True, light=1.0, holes=True, hsmoke=None):
    """Towering flame column: flared skirt, waist, spreading head.  Big tongues peel off both edges and climb; inside, steep half-turn sweeps of drawn fire wind round it,
    black-red hearts show between them, pale rivers in the middle; hot cumulus smoke behind.  grow 0..1 raises it.  -> (smoke FX or None, fire FX)"""
    from .fx_break import puffs, billow, HOTSMOKE
    hs = hsmoke or HOTSMOKE; k = int(f) // step * step; t = k / 24.; top = y0 + (y1 - y0) * (1 - (1 - min(grow, 1.)) ** 2); Hh = y0 - top; cel = Cel()
    ax = lambda v: cx + lean * Hh * np.asarray(v) + w * .18 * np.sin(np.asarray(v) * 5.2 - t * 4.5) * (.15 + np.asarray(v))
    prof = lambda v: (.80 + .95 * np.exp(-(np.clip(np.asarray(v, float), 0, 2) / .085) ** 1.3) + .95 * np.clip(np.asarray(v, float), 0, 2) ** 2.2) * (1 if grow >= 1 else (1 - np.clip(np.asarray(v, float), 0, 1) ** 6))
    vv = np.linspace(0, 1, 48); spine = np.c_[ax(vv), y0 - Hh * vv]; sm = None
    if smoke:
        sc = Cel(); n = 8
        for i in range(n):                                   # climbing smoke, both sides
            ph = (t * .13 * (.7 + .5 * h1(seed, i, 1)) + (i + .6 * h1(seed, i, 2)) / n) % 1.; sg = 1 if i % 2 else -1; y = y0 - 90 - Hh * 1.10 * ph; x = cx + lean * Hh * ph + sg * w * (1.25 + 1.6 * ph + .4 * h1(seed, i, 3)); sz = w * (1.0 + 1.25 * ph) * (.8 + .4 * h1(seed, i, 4)); puffs(sc, billow(.5 + .25 * math.sin(t * .8 + i), x, y, sz, seed * 3 + i, 7, .2, 1., 4.2, 1.0 + t * .5), hs, (-sg * .90, .44), lw, seed + i, tone=.86)
        for i in range(12):                                  # low smoke rolling out along the ground: big against the column, broken far out
            sg = 1 if i % 2 else -1; q = (i // 2 + .9 * h1(seed, i, 31)) / 6.; sz = w * (.45 + 1.0 * bms(h1(seed, i, 32), .4, .4)) * (1.15 - .6 * q); puffs(sc, billow(.45 + .2 * math.sin(t + i), cx + sg * w * (1.4 + 3.4 * q ** 1.2), y0 - w * .10 - sz * .12 * h1(seed, i, 33), sz, seed * 5 + i, 7, .2, 1.15, 3.4, 1.2 + t * .6), hs, (-sg * .85, -.30), lw, seed + i + 50, tone=1.0 - .45 * q)
        sm = FX(sc.layer(), None)
    r_ = []
    for sg in (1, -1):                                       # peeling tongues
        n = int(8 * tongues)
        for i in range(n):
            q = t * speed / Hh * .50 * (.8 + .4 * h1(seed, i, sg, 1)) + (i + .8 * h1(seed, i, sg, 2)) / n; ph = q % 1.; cyc = int(q); z = bms(h1(seed, i, sg, cyc, 3), .35, .4); v = ph * .98; x = float(ax(v)); hw = float(w * prof(np.array([v]))[0]); L = w * (.9 + 2.3 * z) * min(1., ph * 5 + .2) * min(1., (1.05 - ph) * 3); th = -math.pi / 2 + sg * (.18 + .50 * h1(seed, i, sg, 4))
            if L > 20: r_.append((x + sg * hw * .58, y0 - Hh * v + L * .12, L, th, seed * 17 + i * 3 + sg + cyc, sg))
    rd = lambda v: np.sqrt(np.clip(.02 + np.asarray(v) / .14, 0, 1))
    for kw, kl, col in ((1.22, 1.06, pal['rim']), (1., .96, pal['hot'])):
        for x, y, L, th, sd, sg in r_:
            sp = _tongue_spine(x, y, L * kl, th, t, sd, .20, .85, 1.6, 26); b = L * .155 * kw; cel.fill(band(sp, (lambda v: b * tongue(v, .25, 1.0) * rd(v), lambda v: b * tongue(v, .36, 1.8) * rd(v)), sd, t * 500, max(40., L * .22), .55, 1.6, t), col)
    for x, y, L, th, sd, sg in r_:
        sp = Path(_tongue_spine(x, y, L, th, t, sd, .20, .85, 1.6, 26), 4.); P = band(sp.sub(sp.L * .06, sp.L * .70, 0, 20), lambda v: L * .075 * tongue(v) * rd(v), sd + 3, t * 400, max(36., L * .2), .5, 1.5, t); cel.fill(P, pal['body']); ink_broken(cel, P, lw * .9, pal['ink'], sd, .6, (50, 200))
    env = lambda v: .6 + .4 * np.sin(np.pi * np.clip(v, 0, 1))
    cel.fill(band(spine, lambda v: w * prof(v) * 1.03 + 4, seed + 1, speed * t * .9, w * .70, .62, 1.7, t, env, wave=.28), pal['rim'])
    cel.fill(band(spine, lambda v: w * prof(v), seed + 2, speed * t, w * .62, .52, 1.7, t, env, wave=.26), pal['hot'])
    A = band(spine, lambda v: w * prof(v) * .80, seed + 3, speed * t * .95, w * .60, .40, 1.6, t, env, wave=.26); cel.fill(A, pal['body']); ink_broken(cel, A, lw, pal['ink'], seed + 4, .55, (80, 320))
    q = spine[2:]
    for j, (o, ww) in enumerate(((-.34, .20), (.30, .26), (.0, .16))):   # rivers inside
        P = Path(q, 8.); R0 = P.sub(0, P.L, lambda v: w * (o + .24 * np.sin(v * 6.5 + j * 2.1 - t * 7)) * prof(v), 60); B = band(R0, lambda v: w * ww * prof(v) * (1 + .35 * noise(v * 9 - t * 3, j * 3., 0, seed)), seed + 11 + j, speed * t * (1 + .1 * j), w * .5, .6, 1.7, t, None, wave=.3); cel.fill(B, pal['hot'] if j < 2 else pal['amber']); ink_broken(cel, B, lw * .9, pal['ink'], seed + 30 + j, .5, (60, 240))
    R1 = Path(q, 8.).sub(0, Path(q, 8.).L, lambda v: w * .14 * np.sin(v * 8 - t * 8) * prof(v), 60); B = band(R1, lambda v: w * .20 * prof(v) * np.sqrt(np.clip(np.asarray(v) / .10, 0, 1)) * (1.15 - .7 * v) * (1 + .4 * noise(v * 11 - t * 4, 2., 0, seed)), seed + 21, speed * t * 1.2, w * .4, .8, 1.8, t, None, wave=.3); cel.fill(B, pal['core'])
    if holes:                                                # black-red hearts between the sweeps
        for i in range(4):
            ph = (t * speed / Hh * .7 + (i + .5 * h1(seed, i, 7)) / 4.) % 1.; v = .04 + ph; x = float(ax(v)) + (h1(seed, i, 8) - .5) * w * .95 * float(prof(np.array([v]))[0]); y = y0 - Hh * v; s = w * (.30 + .24 * h1(seed, i, 9)) * min(1., ph * 6); sp = _tongue_spine(x, y + s, s * 3.4, -math.pi / 2 + .25 * (h1(seed, i) - .5), t, seed + i, .12, 1., 2., 22)
            cel.fill(band(sp, lambda v: s * .70 * leaf(v, .28, .5, 1.), seed + i, t * 400, s * .7, .75, 1.5, t), pal['rim']); B = band(sp, lambda v: s * .56 * leaf(v, .28, .5, 1.), seed + i + 2, t * 400, s * .6, .75, 1.5, t); ink_broken(cel, B, lw * .8, pal['ink'], seed + i, .6, (30, 90)); cel.fill(B, '#a3160c'); cel.fill(band(sp, lambda v: s * .40 * leaf(v, .30, .5, 1.), seed + i + 1, t * 400, s * .55, .8, 1.5, t), pal['dark'])
    it = []
    per = w * 5.6                                            # steep half-turn sweeps of fire round the front
    for i in range(sweeps):
        q = (t * speed * .75 / (Hh + per) + (i + .5 * h1(seed, i, 21)) / sweeps) % 1.; s0 = -per * .25 + q * (Hh + per); sg = 1 if h1(seed, i, 22) < .72 else -1; it.append((h1(seed, i, 23), s0, sg, i))
    it.sort()
    for _, s0, sg, i in it:
        l = per * (.62 + .5 * h1(seed, i, 24)); u0 = max(0., -s0 / l)
        if u0 > .6: continue
        u = np.linspace(u0, 1, 30); v = (s0 + u * l) / Hh; hw = w * prof(np.clip(v, 0, 1.3)); X = ax(v) - sg * np.cos(np.pi * (.06 + .88 * u)) * hw * (.92 + .10 * np.sin(u * 6 + i)); Y = y0 - Hh * v; flame_stream(cel, np.c_[X, Y], w * (.22 + .20 * h1(seed, i, 25)) * (.85 + .3 * float(prof(np.array([v[15]]))[0])), t + i * .37, seed * 3 + i, pal, lambda q: leaf(q, .42, .62, .85), speed * .5, None, .60, True, 1.0, True, True, lw * .9, 1.1, False, .30)
    lay = cel.layer()
    if ring:
        c2 = Cel(); flame_ring(c2, cx, y0 - w * .05, w * 1.9, w * .40, w * 1.35, t, seed + 9, pal, 34, True, True, .80, lw); a = c2.layer(); lay = lay * (1 - a[..., 3:4]) + a
    gl = cv2.resize(lay, (W // 4, H // 4), interpolation=cv2.INTER_AREA)[..., 3]; g = (.34 * cv2.GaussianBlur(gl, (0, 0), 5) + .34 * cv2.GaussianBlur(gl, (0, 0), 16) + .42 * cv2.GaussianBlur(gl, (0, 0), 46)); L = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)[..., None] * C('#ff6a1a') * .62 * light; L *= (1 - .85 * lay[..., 3:4])
    return sm, FX(lay, L)
