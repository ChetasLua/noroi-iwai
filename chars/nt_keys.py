"""Nanami / Todo hero keys and the backgrounds they share with the shots."""
from chars.ntkit import *
from chars.nanami import Nanami, PAL, Body, EXPR
from chars.todo import Todo, TBody, hand_placeholder, hand3, PAL as TPAL, EXPR as TEXPR
from jj import clouds as CL
from jj.kit import para, tint, cast, aura, flare, hatch, splatter
from chars.ntkit import rim

_C = {}


_FB = {}


def fb(scales, wts, seed, h=H, w=W):
    k = (scales, wts, seed, h, w)
    if k not in _FB: _FB[k] = fbm(h, w, scales, wts, seed)
    return _FB[k]


def memo(fn):
    def g(*a):
        k = (fn.__name__,) + a
        if k not in _C: _C[k] = fn(*a)
        return _C[k]
    return g


# ================================================================= beach
@memo
def beach_plate(hz=585, seed=4):
    """overexposed noon beach, painted, no figure (float RGB)"""
    x, y = grid()
    r = np.random.default_rng(seed)
    t = np.clip(y / hz, 0, 1)
    sky = ramp(t, [(0, '#7fc4ee'), (.35, '#a9daf4'), (.72, '#dff2fa'), (1, '#fdfaf2')])
    sky = sky + (radial_(x, y, -80, 100, 1200, 900, 2.0) * .18)[..., None] * rgb('#fff3d8')
    S_ = CL.Sky(H, W, np.clip(sky, 0, 1), light=(-.75, -.55, .36), seed=seed)
    T = ('#ffffff', '#edf5fc', '#b9d5ef', '#97bde2')
    m = '#f3f8fb'
    S_.bank(1500, hz + 6, 700, 190, T, m, seed=5, detail=.9, op=.85)
    S_.bank(360, hz + 8, 1250, 360, T, m, seed=9, detail=1.0)
    S_.bank(980, hz + 8, 620, 150, T, m, seed=12, detail=.8, op=.9)
    S_.bank(90, hz + 10, 560, 130, T, m, seed=15, detail=.8)
    img = S_.paint(seed=3, sizes=(16, 8), amt=.7)
    isl = Cel(); isl.fill(spl([(430, hz + 4), (500, hz - 14), (560, hz - 34), (610, hz - 30), (660, hz - 52), (720, hz - 40), (770, hz - 18), (850, hz - 8), (900, hz + 4)], n=8), '#a8cfe6'); isl.fill(spl([(640, hz + 4), (690, hz - 20), (740, hz - 26), (800, hz - 10), (880, hz + 4)], n=8), '#97c3dd')
    img = comp.over(img, isl.layer(), .8)
    # ---- sea
    sea_b = 842
    s_ = np.clip((y - hz) / (sea_b - hz), 0, 1)
    sea = ramp(s_, [(0, '#5fb0d2'), (.10, '#4fb3cf'), (.45, '#6fd0d4'), (.78, '#a8e8df'), (1, '#dcf6ea')])
    nz = fbm(H, W, (60, 22, 8), (1, .7, .5), 7, aspect=7.0)
    sea = sea * (1 + .045 * nz[..., None])
    gl = np.clip(radial_(x, y, 120, hz, 1150, 420, 1.1), 0, 1)
    gn = fbm(H, W, (5, 2.6), (1, .8), 11, aspect=4.0)
    sea = sea + (np.clip(gn - 1.0, 0, 2) * gl * 1.3)[..., None] * rgb('#fffaf0') + (gl * .22)[..., None]
    shore = sea_b + 16 * np.sin(x / 260.) + 9 * np.sin(x / 83. + 1.2) + (x - 960) * .05
    img = np.where((y > hz)[..., None], sea, img)
    # ---- sand
    sv = np.clip((y - shore) / (H - sea_b), 0, 1.4)
    sand = ramp(sv, [(0, '#e9ddc6'), (.10, '#f3e8d2'), (.35, '#fdf5e4'), (1, '#fff9ee')])
    sand = sand * (1 + .018 * fbm(H, W, (40, 14, 5), None, 17, aspect=3.5)[..., None])
    img = np.where((y > shore)[..., None], sand, img)
    cv = Canvas(img=img)
    # foam: three broken lace lines
    for k, (off, wd, n) in enumerate([(-30, 6, 60), (-8, 9, 80), (10, 5, 46)]):
        xs = r.uniform(-60, W + 60, n); ys = np.interp(xs, x[0], shore[0]) + off + r.normal(0, 5, n)
        cv.strokes(arcs(xs, ys, r.normal(.04, .08, n), r.uniform(60, 200, n), r.normal(0, .5, n)), wd * r.uniform(.4, 1.1, n), rgb('#ffffff')[None], op=.9, dry=.25, seed=31 + k, shape='round')
    # swell lines
    n = 220; ys = hz + (sea_b - hz) * r.random(n) ** 1.6; xs = r.uniform(-50, W + 50, n); q = (ys - hz) / (sea_b - hz)
    cv.strokes(arcs(xs, ys, r.normal(0, .02, n), 40 + 200 * q * r.uniform(.5, 1.3, n), r.normal(0, .12, n)), .8 + 3.2 * q, ramp(r.random(n), [(0, '#ffffff'), (.5, '#c6f0ec'), (1, '#4aa9c4')]), op=.55, dry=.2, seed=41, shape='round')
    img = cv.img
    hzb = np.exp(-((y - hz) / 5.) ** 2) * .55
    img = img * (1 - hzb[..., None]) + rgb('#fbfdfb') * hzb[..., None]
    return np.clip(img, 0, 1.2)


def radial_(x, y, cx, cy, rx, ry, p=2.):
    return np.clip(1 - np.sqrt(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2), 0, 1) ** p


def frond(cel, x0, y0, ang, L, n=26, w=46, curl=.6, col='#5f9a64', dark='#3d7350', seed=0, sway=0.):
    """a palm frond: drooping rib with blades"""
    r = np.random.default_rng(seed); t = np.linspace(0, 1, 40)
    a = ang + curl * t ** 1.4 + sway * t ** 2; st = L / 39.
    P = np.array([x0, y0]) + np.c_[np.cumsum(np.cos(a)), np.cumsum(np.sin(a))] * st
    for side in (-1, 1):
        for k in range(n):
            u = .06 + .92 * k / n + r.normal(0, .008); i = min(38, int(u * 39)); p = P[i]; th = a[i] + side * (1.05 - .45 * u + r.normal(0, .06))
            ln = L * .34 * math.sin(math.pi * (.16 + .80 * u)) * r.uniform(.85, 1.12); ww = w * .28 * r.uniform(.8, 1.2)
            d = np.array([math.cos(th), math.sin(th)]); nn = np.array([-d[1], d[0]]); dr = np.array([0, 1.]) * ln * .22
            mid = p + d * ln * .5 + nn * ww * .5 + dr * .25; tip = p + d * ln + dr
            cel.fill(np.vstack([spl([p + nn * ww * .3, mid + nn * ww * .4, tip], n=6), spl([tip, mid - nn * ww * .9, p - nn * ww * .4], n=6)]), dark if side < 0 else col)
    cel.line(P, 7, dark, taper=(0, .6))
    return P


def key_beach(yaw=-.34, f=0, expr='gentle', byaw=.30, cx=1215, cy=450, s=262, roll=-.035, pitch=.02, wind=.055, gaze=(.60, .0), finish=True):
    x, y = grid()
    bg = comp.bokeh(beach_plate().astype(f32), 6.5, 1.0)
    # sun glints in the bokeh
    r = np.random.default_rng(21); sp_ = np.zeros((H, W, 3), f32)
    for _ in range(150):
        px, py = abs(r.normal(0, 520)) + r.uniform(-40, 60), 592 + abs(r.normal(0, 90)); rad = r.uniform(2.0, 7.5) * (1 + (py - 590) / 260)
        if py > 840: continue
        cv2.circle(sp_, (int(px * 8), int(py * 8)), int(rad * 8), tuple(float(v) for v in (rgb('#fffaf0') * r.uniform(.25, .9))), -1, cv2.LINE_AA, 3)
    bg = comp.screen(bg, np.clip(cv2.GaussianBlur(sp_, (0, 0), 1.2), 0, 1))
    bg = bg + (radial_(x, y, -60, 60, 1150, 950, 2.4) * .30)[..., None] * rgb('#fff6e2')
    gu = Cel()
    for gx_, gy_, gs, ga in ((318, 214, 30, .16), (412, 268, 20, -.10), (236, 300, 15, .05), (640, 176, 13, -.2)):
        fl_ = .35 + .25 * math.sin(f * .35 + gx_); wing = lambda q: spl([(gx_, gy_), (gx_ + q * gs * .55, gy_ - gs * fl_), (gx_ + q * gs * 1.25, gy_ - gs * (fl_ * .35) + q * ga * gs)], n=10)
        for q in (-1, 1): gu.line(wing(q), gs * .17, '#8fb4cc', taper=(.15, .75), press=.2)
        gu.line([(gx_ - gs * .18, gy_ + 2), (gx_, gy_), (gx_ + gs * .26, gy_ + 4)], gs * .20, '#7fa6c2', taper=(.5, .5))
    bg = comp.over(bg, cv2.GaussianBlur(gu.layer(), (0, 0), 1.6), .9)
    pal = dict(PAL['beach'])
    e = dict(EXPR[expr]) if isinstance(expr, str) else dict(expr); e['gaze'] = gaze; e['wet'] = 1
    c = Cel(); lw = s / 300. * 5.0
    H_ = Nanami.pose(cx, cy, s, yaw, pitch, roll); b = Body.under(H_, byaw, .03, .02)
    kw = dict(f=f, wind=wind, seed=3, goggles=False, lit=-1, lw=lw, loose=1.5, wdir=(1.0, .10, 0.), eat=.8, hatch_=0.)
    pal['blown'] = 1; e['lockshade'] = 1; e['asym'] = 1.
    Nanami.head(c, H_, e, pal, part='back', **kw)
    Nanami.draw_shirt(c, H_, b, pal, lw, 3, -1, f, wind * 8)
    Nanami.head(c, H_, e, pal, part='front', **kw)
    lay = c.layer()
    # ---- dappled palm shade on the shirt, clipped to the cel
    m = Cel(); frond(m, 1980, 640, 2.75, 900, n=20, w=60, curl=-.35, seed=5, col='#ffffff', dark='#ffffff')
    dm = cv2.GaussianBlur(m.rgba()[..., 3], (0, 0), 1.6) * smooth(640, 820, y)
    sh = rgb('#b9cdee'); lay[..., :3] = lay[..., :3] * (1 - dm[..., None] * .62 * (1 - sh))
    # ---- light on the cel
    lay = rim(lay, 5, 2, '#fff7e6', .62, .5); lay = rim(lay, 2, 4, '#fff3d6', .45, .5)
    lay = para(lay, (cx - 520, 200), (cx + 160, 520), [(0, '#8a7a5c'), (.45, '#33291c'), (1, '#000000')], 'screen', .9)
    lay = para(lay, (cx - 100, 500), (cx + 420, 1080), [(0, '#ffffff'), (1, '#d5dff6')], 'mul', .8)
    bl = qb(lay[..., 3], 26)
    bg = comp.screen(bg, (bl * .10)[..., None] * rgb('#fff4e0'))
    img = comp.over(bg, lay)
    # ---- foreground palm, far out of focus
    g = Cel(); frond(g, 2040, -120, 2.30, 900, n=22, w=90, curl=-.5, seed=2, col='#9cc79a', dark='#5f9c78'); frond(g, 2080, 40, 2.75, 760, n=20, w=80, curl=-.55, seed=8, col='#a9d0a4', dark='#6ba584')
    fl = comp.bokeh(g.layer(), 13, 1.0); img = comp.over(img, fl, .92)
    # ---- wrap of sea light over everything, motes
    img = img + (radial_(x, y, -60, 100, 1200, 1000, 2.4) * .20)[..., None] * rgb('#fff4de')
    mo = np.zeros((H, W, 3), f32); r = np.random.default_rng(8)
    for k in range(46):
        px, py = (r.uniform(-60, W + 60) + f * r.uniform(.6, 1.6)) % (W + 120) - 60, (r.uniform(0, H) - f * r.uniform(.2, .8)) % H; rad = r.uniform(1.0, 3.0) if r.random() < .75 else r.uniform(5, 13)
        a_ = r.uniform(.25, .8) if rad < 4 else r.uniform(.07, .16)
        cv2.circle(mo, (int(px * 8), int(py * 8)), int(rad * 8), tuple(float(v) for v in rgb('#fff6df') * a_), -1, cv2.LINE_AA, 3)
    img = comp.screen(img, cv2.GaussianBlur(mo, (0, 0), 1.0))
    return comp.finish(img, f, bloom=.34, bloom_thr=.84, vig=.05, halation='#fff0dc', soft=.20) if finish else img


# ================================================================= embers
@memo
def ember_plate(seed=6, w=W):
    """burning underground concourse, out of focus: pillars, beams, fire light low left, smoke (w may be wider than the frame: more pillars to the right)"""
    x, y = grid(H, w)
    r = np.random.default_rng(seed)
    W_ = w
    g = ramp(np.clip(y / H, 0, 1), [(0, '#0d0609'), (.45, '#1c0b0d'), (.75, '#3a1410'), (1, '#1a0a0a')])
    g = g + (radial_(x, y, 250, 820, 1150, 760, 1.5) * .95)[..., None] * rgb('#c8481a') + (radial_(x, y, 180, 900, 620, 420, 1.6) * .9)[..., None] * rgb('#ff9b3c')
    g = g + (radial_(x, y, 1850, 900, 620, 520, 2.0) * .35)[..., None] * rgb('#a82a14')
    c = Cel(w, H)
    if w > W: g = g + (radial_(x, y, 2250, 860, 520, 460, 1.7) * .55)[..., None] * rgb('#c8481a')
    # far pillars and a broken ceiling beam in silhouette
    for px, w0, top, col in ((330, 70, 120, '#3b1512'), (560, 90, 90, '#2b0f0f'), (905, 130, 40, '#1b0a0b'), (1360, 190, 0, '#12080a'), (1780, 260, -20, '#0d0608'), (2060, 80, 110, '#2b0f0f'), (2330, 150, 20, '#170909'), (2620, 110, 90, '#260d0d')):
        k = r.normal(0, 6, 8)
        c.fill(np.array([(px - w0 * .5 + k[0], top), (px + w0 * .5 + k[1], top), (px + w0 * .48 + k[2], 460), (px + w0 * .52 + k[3], 900), (px + w0 * .6, 1080), (px - w0 * .6, 1080), (px - w0 * .5 + k[4], 700), (px - w0 * .46 + k[5], 380)]), col)
    c.fill(np.array([(-20, -20), (1940, -20), (1940, 30), (1500, 70), (1180, 40), (960, 96), (700, 84), (520, 130), (300, 120), (140, 168), (-20, 160)]), '#100709')
    c.fill(np.array([(620, 90), (700, 84), (520, 420), (470, 402)]), '#1a0b0c')
    for k in range(14):
        bx, by = r.uniform(-40, 1400), r.uniform(900, 1060); w0 = r.uniform(40, 170); h0 = w0 * r.uniform(.3, .6)
        c.fill(np.array([(bx - w0, by + h0), (bx - w0 * .6, by - h0 * .2), (bx - w0 * .1, by - h0 * .8), (bx + w0 * .4, by - h0 * .4), (bx + w0, by + h0)]), '#170a0a')
    g = comp.over(g, c.layer(H, w))
    cv = Canvas(h=H, w=w, img=g)
    # smoke: long slow strokes
    n = int(170 * w / W); xs, ys = r.uniform(-100, w + 100, n), r.uniform(0, 900, n)
    cv.strokes(arcs(xs, ys, r.normal(-.5, .35, n), r.uniform(160, 520, n), r.normal(0, 1.0, n)), r.uniform(16, 60, n), ramp(r.random(n), [(0, '#14090b'), (.6, '#2c1110'), (1, '#5a2014')]), op=.16, dry=.4, seed=3, shape='round')
    # fire behind the rubble: additive tongues
    fr = np.zeros((H, w, 3), f32); n = 46
    for k in range(n):
        fx, fy = r.normal(215, 250), r.uniform(900, 1080); hh = r.uniform(120, 340) * math.exp(-((fx - 215) / 420) ** 2); ww = r.uniform(26, 70); b_ = r.normal(0, 40)
        P = np.vstack([spl([(fx - ww, fy), (fx - ww * .7 + b_ * .3, fy - hh * .45), (fx + b_, fy - hh)], n=8), spl([(fx + b_, fy - hh), (fx + ww * .6 + b_ * .4, fy - hh * .4), (fx + ww, fy)], n=8)])
        l_ = np.zeros((H, w), f32); cv2.fillPoly(l_, [np.round(P * 16).astype(np.int32)], 1., cv2.LINE_AA, 4); fr += l_[..., None] * rgb(['#ff6a1e', '#ff8a30', '#c8431a'][k % 3]) * .22
    fr = qb(fr, 5)
    cv.img = cv.img + np.minimum(fr, 1.0)
    img = cv.img + (radial_(x, y, 200, 930, 560, 330, 1.5) * .5)[..., None] * rgb('#ffb05a')
    return np.clip(img, 0, 1.4)


def embers(f=0, n=240, seed=3, rmax=3.2, blur_=0., col=('#ff5a14', '#ff9a3a', '#ffe0a0'), drift=(3.2, -2.2), box=None):
    """additive RGB layer of sparks: short hot streaks in the heat wind"""
    r = np.random.default_rng(seed); o = np.zeros((H, W, 3), f32)
    x0, y0, x1, y1 = box or (-80, -80, W + 80, H + 80)
    for k in range(n):
        sx, sy = r.uniform(x0, x1), r.uniform(y0, y1); v = r.uniform(.5, 1.7); ph = r.uniform(0, 6.28)
        px = x0 + (sx - x0 + f * drift[0] * v + 14 * math.sin(f * .11 * v + ph)) % (x1 - x0); py = y0 + (sy - y0 + f * drift[1] * v + 8 * math.sin(f * .17 + ph * 2)) % (y1 - y0)
        rad = .6 + rmax * r.random() ** 3; hot = r.random(); c = rgb(col[0]) * (1 - hot) + rgb(col[2] if hot > .6 else col[1]) * hot
        tw = .55 + .45 * math.sin(f * .6 * v + ph); L = rad * r.uniform(.6, 3.2); d = np.array([drift[0] + 8 * .11 * math.cos(f * .11 * v + ph), drift[1]]); d = d / (np.hypot(*d) + 1e-6)
        a, b = np.array([px, py]) - d * L, np.array([px, py]) + d * L
        cv2.line(o, tuple(int(q * 16) for q in a), tuple(int(q * 16) for q in b), tuple(float(q) for q in c * (.6 + .8 * tw)), max(1, int(rad * 1.3)), cv2.LINE_AA, 4)
    return cv2.GaussianBlur(o, (0, 0), blur_) if blur_ > 0 else o


def char_burn(lay, mask, f=0, seed=2, amt=1.0, keep=.10, cracks=(), amp=.12, sc=(64, 24, 9)):
    """burn part of a cel to charcoal.  lay premult RGBA; mask 1 = burnt.  Ragged glowing edge, no gore: silhouette only."""
    a = lay[..., 3]
    n = fb(sc, (1, .5, .22), seed)
    soft_ = qb(mask.astype(f32), 6) + amp * n * (qb(mask.astype(f32), 26) * (1 - qb(mask.astype(f32), 26)) * 4)
    m = smooth(.48, .53, soft_) * (a > .02)
    live = ((m <= .5) & (a > .5)).astype(np.uint8); live = cv2.erode(live, np.ones((3, 3), np.uint8)); d_in = cv2.distanceTransform(1 - live, cv2.DIST_L2, 5) - 1.5; d_in = np.where(m > .4, np.maximum(d_in, 0), -1.); d_out = cv2.distanceTransform((m <= .5).astype(np.uint8), cv2.DIST_L2, 5)
    L = (lay[..., :3] * np.array([.3, .59, .11], f32)).sum(-1)
    coal = rgb('#090507') + L[..., None] * keep * rgb('#ff6a3a') + np.clip(fb((5, 2.5), (1, .7), seed + 3) - 1.1, 0, 2)[..., None] * .035 * rgb('#ff8a50')
    out = lay.copy(); out[..., :3] = lay[..., :3] * (1 - m[..., None]) + coal * (m * a)[..., None]
    # scorched skin just outside
    sc = np.exp(-d_out / 16.) * (1 - m) * (.75 + .25 * np.clip(n, -1, 1)); out[..., :3] *= (1 - .72 * np.clip(sc, 0, 1))[..., None]
    # ember edge: hot hairline, orange band, dull red fall-off, broken by noise that crawls
    nz = fb((30, 11), (1, .6), seed + 5); cr = np.clip(.55 + .45 * np.sin(nz * 2.6 + f * .22), 0, 1)
    e = (np.exp(-(d_in / 1.5) ** 2) * 1.2 * smooth(.25, .6, cr))[..., None] * rgb('#ffe6ae') + (np.exp(-(d_in / 4.2) ** 2) * smooth(.1, .5, cr))[..., None] * rgb('#ff8126') + (np.exp(-d_in / 13.) * .55 * cr)[..., None] * rgb('#b0300f')
    e = e * (d_in >= 0)[..., None]
    out[..., :3] += e * a[..., None] * amt
    glow_ = qb(e * a[..., None], 14) * .9
    return out, m, glow_ * amt


PROF_PAL = dict(skin='#a4583a', skinS='#4a1f1c', skinD='#220e12', hair='#9c6a38', hairS='#4d2a1e', hairD='#231113', suit='#75473a', suitS='#371d1f', suitD='#190c10', shirt='#2f2940', shirtS='#191526', tie='#8c5a24', tieS='#422619', sclera='#d8a67c', scleraS='#744538')
BURN = [(.34, 1.6), (.30, 1.08), (.315, .92), (.265, .84), (.275, .62), (.215, .50), (.225, .30), (.165, .22), (.175, .02), (.125, -.08), (.135, -.30), (.085, -.38), (.095, -.58), (.045, -.64), (.06, -.86), (.00, -.95), (.03, -1.20), (-.03, -1.30), (.02, -1.62), (-.02, -1.74), (.08, -2.10), (.05, -2.30), (.13, -2.70), (.10, -2.95), (.14, -3.6), (-3, -3.6), (-3, 1.6)]


def prof_figure(cx, cy, s, pitch=-.03, roll=-.05, f=0, pal=None, e=None, arm=0., wind=.03, lagx=0., light=True, loose=.7, seed=3, hat=True):
    """Nanami in profile, facing picture left, down to the chest.  arm = swing of the near sleeve (rad), lagx = lag of tie / lapel (units).
    returns (layer lit by fire, Z = local (z forward, y up) -> px, head)"""
    x, y = grid(); pal = dict(PAL['ember'], **PROF_PAL) if pal is None else pal; e = dict(EXPR['far'], open=.52) if e is None else e
    c = Cel(); lw = s / 300. * 5.0; D = -1.; H_ = Nanami.pose(cx, cy, s, -1.5, pitch, roll); rb = roll * .45; cr, sr = math.cos(rb), math.sin(rb)
    def Z(pts):
        P = np.asarray(pts, float); u, v = D * P[:, 0], -P[:, 1]; piv = 1.2; v = v - piv * 0
        return np.c_[cx + (u * cr - v * sr) * s, cy + (u * sr + v * cr) * s]
    ln = pal['line']
    def shp(pts, col, w=.9, corners=(), sd=0, sh=None):
        P = Z(spl(np.array(pts, float), True, corners=corners, n=9)); c.fill(P, col)
        if sh:
            c.clip(P)
            for q, cl in sh: c.fill(Z(q), cl)
            c.unclip()
        if w: outline(c, P, lw * w, ln, sd, press=.45)
        return P
    kw = dict(f=f, wind=wind, seed=seed, goggles=False, lit=-1, lw=lw, loose=loose, wdir=(1., .2, 0)); L_ = lagx
    Nanami.head(c, H_, e, pal, part='back', **kw)
    shp([(.20, -1.62), (.62, -1.95), (.86 + L_, -2.6), (.92 + L_, -3.4), (.1, -3.4)], pal['suitS'], .8, (0, 3, 4), 1)
    shp([(.40, -.80), (.345, -.96), (.355, -1.08), (.300, -1.20), (.275, -1.40), (.30, -1.75), (-.5, -1.75), (-.78, -1.35), (-.66, -.98), (-.60, -.60), (-.62, -.30), (-.1, -.20), (.1, -.5)], pal['skin'], .85, (0, 5, 6, 10, 11), 2,
        [([(.50, -.75), (.30, -.98), (.18, -1.10), (.24, -1.28), (.05, -1.40), (.10, -1.72), (-1, -1.8), (-1, -.2), (.2, -.2)], pal['skinS']), ([(.2, -.78), (-.06, -.98), (-.16, -1.3), (-.40, -1.7), (-1, -1.8), (-1, -.2)], pal['skinD'])])
    c.line(Z(spl([(-.30, -.66), (-.08, -1.0), (.10, -1.32), (.20, -1.62)], n=9)), lw * .46, pal['sline'], taper=(.4, .5), seed=4)
    shp([(.26, -1.44), (.52, -1.66), (.70, -2.2), (.80, -3.4), (.1, -3.4), (.0, -1.9)], pal['shirt'], .6, (0, 3, 4), 6, [([(.42, -1.5), (.52, -2.2), (.5, -3.4), (-.2, -3.4), (-.2, -1.5)], pal['shirtS'])])
    shp([(.275, -1.46), (.425, -1.50), (.50, -1.64), (.46, -1.80), (.33, -1.74)], pal['tie'], .6, (0, 3), 7, [([(.33, -1.4), (.36, -1.8), (.2, -1.8), (.2, -1.4)], pal['tieS'])])
    tb = shp([(.34, -1.75), (.47, -1.80), (.66 + L_ * .8, -2.25), (.77 + L_ * 1.6, -3.4), (.56 + L_ * 1.6, -3.4), (.50 + L_ * .8, -2.4)], pal['tie'], .6, (0, 1, 3, 4), 8, [([(.36, -1.7), (.54 + L_ * .8, -2.3), (.62 + L_ * 1.6, -3.4), (.4, -3.4), (.3, -2)], pal['tieS'])])
    c.clip(tb); splr = np.random.default_rng(5)
    for k in range(26):
        yy = -1.8 - 1.4 * k / 26; z = np.interp(yy, [-3.2, -2.3, -1.8], [.66 + L_ * 1.5, .60 + L_ * .8, .41]) + ((k % 3) - 1) * .06; rd = .026 * (1 + splr.normal(0, .25, 8)); an = np.linspace(0, 6.283, 8, endpoint=False)
        c.fill(Z(spl(np.c_[z + rd * np.cos(an) * .8, yy + rd * np.sin(an) * 1.2], True, n=4)), pal['spot'])
    c.unclip()
    shp([(.30, -1.36), (.05, -1.14), (-.34, -.96), (-.70, -.94), (-.80, -1.10), (-.52, -1.22), (-.12, -1.46), (.22, -1.98), (.36, -1.72)], pal['shirt'], .75, (0, 3, 7), 9, [([(.1, -1.1), (-.1, -1.32), (.12, -1.75), (.2, -2.1), (-.9, -1.4), (-.9, -.9)], pal['shirtS'])])
    shp([(-.72, -.98), (-.95, -1.12), (-1.22, -1.45), (-1.42, -2.0), (-1.50, -2.7), (-1.45, -3.5), (.86, -3.5), (.74, -2.75), (.46, -2.30), (.16, -1.95), (-.14, -1.60), (-.42, -1.26)], pal['suit'], .95, (0, 5, 6), 10,
        [([(.34, -2.15), (.10, -2.3), (.22, -2.7), (.05, -3.5), (-1.7, -3.5), (-1.7, -1.0), (-.6, -.9), (-.30, -1.5)], pal['suitS'])])
    shp([(-.70, -.97), (-.86, -1.10), (-.62, -1.44), (-.22, -1.82), (-.02, -2.10), (.12, -2.02), (.26, -2.22), (.60 + L_ * .4, -2.86), (.74 + L_ * .6, -3.5), (.86 + L_ * .6, -3.5), (.74 + L_ * .4, -2.70), (.46, -2.24), (.14, -1.90), (-.16, -1.56), (-.44, -1.22)], pal['suit'], .7, (0, 4, 5, 8, 9), 11, [([(-.1, -1.5), (-.02, -1.92), (.34, -2.4), (.5, -3.5), (-1, -3.5), (-1, -.9)], pal['suitS'])])
    piv = np.array([-.42, -1.75]); ca, sa = math.cos(arm), math.sin(arm); rot = lambda P: (np.asarray(P, float) - piv) @ np.array([[ca, sa], [-sa, ca]]) + piv
    k_ = np.clip((-1.7 - np.array([-1.58, -1.66, -1.92, -2.35, -2.9, -3.5, -3.5, -2.6, -2.0, -1.68])) / .6, 0, 1)[:, None]
    base_ = np.array([(-.62, -1.58), (-.30, -1.66), (.02, -1.92), (.20, -2.35), (.24, -2.9), (.22, -3.5), (-1.10, -3.5), (-1.08, -2.6), (-.98, -2.0), (-.84, -1.68)]); slv = base_ * (1 - k_) + rot(base_) * k_
    shp(slv, pal['suit'], .9, (5, 6), 12, [(rot([(.0, -1.95), (-.10, -2.4), (.02, -2.8), (-.12, -3.5), (-1.3, -3.5), (-1.3, -1.5), (-.4, -1.5), (-.16, -1.9)]), pal['suitS'])])
    for pp in ([(.06, -2.0), (-.14, -2.30), (-.20, -2.72)], [(.22, -2.74), (.00, -2.86), (-.26, -2.78)], [(-.60, -1.6), (-.72, -2.0), (-.70, -2.5)]):
        c.line(Z(spl(rot(np.array(pp)), n=8)), lw * .42, ln, taper=(.1, .7), seed=13)
    Nanami.head(c, H_, e, pal, part='front', **kw)
    if hat:
        hatch(c, Z([(.30, -.32), (.385, -.38), (.36, -.48), (.40, -.58), (.38, -.68), (.28, -.70), (.22, -.5)]), .95, 7, 1.3, '#1d0a0b', 3, length=30, curve=.05)
        hatch(c, Z([(.22, -1.05), (.20, -1.3), (.0, -1.35), (-.1, -1.05)]), 1.1, 7, 1.3, '#140709', 4, length=40, curve=.05)
        hatch(c, Z([(.40, -2.3), (.60, -2.9), (.30, -3.2), (.12, -2.5)]), .7, 10, 1.5, '#140709', 6, length=60, curve=.05)
    lay = c.layer()
    if light:
        lay = rim(lay, 7, -4, '#ff9d44', .95, .8); lay = rim(lay, 3, -1, '#ffe0a2', .9, .6)
        lay = para(lay, (cx - 1.0 * s, cy + 1.6 * s), (cx + .2 * s, cy - .3 * s), [(0, '#8a3a12'), (.5, '#3a1508'), (1, '#000000')], 'screen', .95)
        lay = para(lay, (cx - .2 * s, cy), (cx + 1.5 * s, cy + .2 * s), [(0, '#ffffff'), (1, '#7a5a66')], 'mul', .85)
    return lay, Z, H_


def ember_cracks(cel, P, w, seed=0, cover=.55, hot=.35, branch=.5):
    """a crack that follows a drawn contour P (px): broken pieces, dull red, hot in the middle, short side cracks"""
    r = np.random.default_rng(seed); P = resample(np.asarray(P, float), 4.); n = len(P); i = int(r.integers(0, 6))
    while i < n - 5:
        L_ = int(r.integers(7, 24))
        if r.random() < cover:
            q = P[i:i + L_] + r.normal(0, .7, (len(P[i:i + L_]), 2))
            if len(q) > 4:
                cel.line(q, w * 1.9, '#8a2410', taper=(.3, .4), press=.4, seed=i); cel.line(q, w, '#e2561a', taper=(.3, .4), press=.5, seed=i + 1)
                if r.random() < hot: cel.line(q[2:-2], w * .55, '#ffb45c', taper=(.4, .4), seed=i + 2)
                if r.random() < branch:
                    k = int(r.integers(1, len(q) - 1)); t = q[k] - q[k - 1]; a = math.atan2(t[1], t[0]) + r.choice([-1, 1]) * r.uniform(.7, 1.3); p_ = q[k]; br = [p_]
                    for _ in range(int(r.integers(2, 5))): a += r.normal(0, .4); p_ = p_ + np.array([math.cos(a), math.sin(a)]) * r.uniform(5, 13); br.append(p_)
                    cel.line(np.array(br), w * .6, '#c2441a', taper=(.05, .8), seed=i + 3)
        i += L_ + int(r.integers(2, 9))


def char_form(Z, H_, s):
    """what the charcoal is: skull, ear, jaw, neck cord, collar, lapel, shoulder and sleeve as dark planes, plus the cracks that run along them"""
    c = Cel(); k = Cel(); sd = 1 if H_.yaw < 0 else -1; lw = s / 300. * 4.; f_ = lambda P, col, cs=(): c.fill(Z(spl(np.array(P, float), True, corners=cs, n=7)) if len(P) > 3 else Z(P), col)
    A, B_, C_, D_, E_ = '#130a0c', '#1f1011', '#2e1715', '#43211a', '#070405'
    # skull: crown plane catches the far red fire, side plane dark, occiput lighter again
    f_([(.40, 1.12), (.0, 1.20), (-.45, 1.10), (-.84, .80), (-1.02, .42), (-.86, .44), (-.58, .74), (-.16, .92), (.30, .96)], D_, (0, 4, 8)); f_([(.30, .96), (-.16, .92), (-.58, .74), (-.86, .44), (-1.02, .42), (-1.04, .12), (-.80, .18), (-.52, .46), (-.12, .62), (.28, .66)], C_, (0, 4, 5, 9))
    f_([(.28, .66), (-.12, .62), (-.52, .46), (-.56, .10), (-.22, .04), (.0, .16), (.22, .12)], A, (0, 3, 6)); f_([(-.52, .46), (-.80, .18), (-1.04, .12), (-.98, -.12), (-.74, -.38), (-.56, -.20), (-.56, .10)], B_, (0, 2, 4))
    # temple and cheek plane back to the ear
    f_([(.24, .14), (.02, .16), (-.04, -.02), (.0, -.26), (.14, -.34)], B_, (0, 1, 4)); f_([(.14, -.30), (.0, -.26), (-.03, -.46), (.10, -.64), (.30, -.80), (.40, -.86), (.24, -.60)], C_, (0, 1, 5))
    # under the jaw, neck cord, nape
    f_([(.44, -.88), (.20, -.76), (.02, -.58), (-.06, -.46), (-.12, -.62), (.06, -.92), (.28, -1.06)], E_, (0, 3, 6)); f_([(-.12, -.50), (-.02, -.78), (.12, -1.08), (.22, -1.40), (.04, -1.42), (-.14, -1.06), (-.26, -.64)], C_, (0, 3, 4)); f_([(-.26, -.62), (-.14, -1.06), (-.02, -1.32), (-.40, -1.12), (-.60, -.94), (-.58, -.58), (-.46, -.44)], A, (0, 2, 5)); f_([(-.60, -.46), (-.57, -.70), (-.66, -.94), (-.60, -.96), (-.48, -.72), (-.50, -.46)], D_, (0, 2, 3))
    # shirt collar, lapel, jacket
    f_([(.20, -1.30), (.02, -1.14), (-.34, -.97), (-.68, -.95), (-.78, -1.08), (-.52, -1.20), (-.14, -1.44), (.12, -1.78)], C_, (0, 3, 4, 7)); f_([(.02, -1.14), (-.34, -.97), (-.68, -.95), (-.70, -1.0), (-.36, -1.04), (.0, -1.24)], D_, (0, 2, 3))
    f_([(-.72, -.98), (-.88, -1.10), (-.62, -1.44), (-.22, -1.82), (-.02, -2.10), (.12, -2.02), (.16, -1.92), (-.14, -1.58), (-.42, -1.24)], B_, (0, 1, 4, 5, 6)); f_([(-.88, -1.10), (-1.22, -1.45), (-1.42, -2.0), (-1.18, -1.86), (-.94, -1.62), (-.66, -1.50)], D_, (0, 2, 5)); f_([(-.64, -1.46), (-.94, -1.62), (-1.18, -1.86), (-1.42, -2.0), (-1.50, -2.7), (-1.45, -3.5), (-1.05, -3.5), (-1.04, -2.6), (-.90, -1.95), (-.62, -1.62), (-.30, -1.70), (-.22, -1.82)], A, (0, 3, 5, 6))
    f_([(-.62, -1.60), (-.30, -1.68), (.0, -1.94), (-.22, -2.02), (-.52, -1.92), (-.78, -1.82)], D_, (0, 2, 5)); f_([(-.78, -1.82), (-.52, -1.92), (-.22, -2.02), (.0, -1.94), (.18, -2.36), (.22, -2.9), (.20, -3.5), (-.36, -3.5), (-.34, -2.8), (-.46, -2.3)], B_, (0, 3, 6, 7)); f_([(-.78, -1.82), (-.46, -2.3), (-.34, -2.8), (-.36, -3.5), (-1.06, -3.5), (-1.04, -2.6), (-.94, -2.0)], E_, (0, 3, 4))
    # ear
    e = H_.ear(sd); o = e['outer']; c.fill(o, D_); c.clip(o); c.fill(e['inner'], A); c.fill(e['tragus'], C_); c.unclip(); c.line(e['inner'], lw * .6, E_, taper=(.3, .3)); outline(c, o, lw * .8, E_, 3)
    # drawn edges of the planes
    for P, w in (([(.28, .66), (-.12, .62), (-.52, .46), (-.60, .12)], .5), ([(.30, .96), (-.16, .92), (-.58, .74), (-.86, .44)], .4), ([(-.12, -.50), (.0, -.84), (.14, -1.14)], .6), ([(-.36, -1.04), (.0, -1.24), (.14, -1.50)], .6), ([(-.42, -1.24), (-.14, -1.58), (.16, -1.92)], .8), ([(-.62, -1.58), (-.84, -1.80), (-1.0, -2.2)], .6)): c.line(Z(spl(np.array(P), n=8)), lw * w, E_, taper=(.2, .4))
    J = H_.jaw(sd)[:-1]; c.line(spl(J, n=6), lw * 1.1, E_, taper=(.1, .3))
    # ---- cracks along the forms
    cr = lambda P, w, sd_, cv=.55, h=.35: ember_cracks(k, Z(spl(np.array(P, float), n=10)), w, sd_, cv, h)
    ember_cracks(k, spl(J, n=8), 2.4, 3, .80, .6); ember_cracks(k, o, 2.0, 5, .70, .5); ember_cracks(k, e['inner'], 1.5, 7, .6, .4)
    for j, (P, w, cv) in enumerate((([(.30, .80), (-.10, .84), (-.50, .68), (-.80, .36), (-.90, .04)], 1.8, .46), ([(.24, .58), (-.14, .56), (-.46, .38), (-.66, .06), (-.70, -.26)], 1.7, .46), ([(.34, 1.04), (-.12, 1.08), (-.54, .96), (-.88, .64)], 1.4, .30), ([(.18, .40), (-.16, .36), (-.38, .18), (-.46, -.06)], 1.4, .40),
                                    ([(.16, .02), (.04, -.10), (.04, -.26), (.12, -.40)], 1.8, .6), ([(-.14, -.50), (-.02, -.82), (.12, -1.12), (.20, -1.38)], 2.0, .62), ([(-.36, -.52), (-.34, -.80), (-.22, -1.10)], 1.4, .45), ([(.18, -1.32), (.0, -1.16), (-.36, -.98), (-.68, -.96)], 2.0, .66), ([(-.40, -1.22), (-.12, -1.56), (.14, -1.90)], 2.2, .70),
                                    ([(-.86, -1.12), (-.62, -1.46), (-.22, -1.84), (-.02, -2.10)], 2.0, .6), ([(-.64, -1.60), (-.30, -1.70), (.0, -1.96), (.18, -2.38), (.22, -2.9)], 2.0, .52), ([(-.88, -1.14), (-1.20, -1.46), (-1.40, -2.0), (-1.46, -2.6)], 1.6, .34), ([(-.50, -1.94), (-.40, -2.35), (-.30, -2.8), (-.34, -3.3)], 1.6, .42), ([(.10, -2.1), (-.12, -2.35), (-.20, -2.75)], 1.5, .45), ([(-.80, -1.86), (-.96, -2.3), (-1.02, -2.9)], 1.4, .38))): cr(P, w * s / 300., 20 + j, cv)
    return c.rgba(), k.rgba()


def burnt(lay, Z, cx, f=0, shift=0., H_=None, s=318.):
    """prof_figure layer -> half burnt black, its forms still legible; returns layer, mask, additive glow"""
    x, y = grid(); B = np.array(BURN, float); B[:-2, 0] += shift
    lay, m, gl = char_burn(lay, mask_of(Z(B), ss=1), f, keep=.03, amp=.055, sc=(34, 12, 5)); a = (m * lay[..., 3])[..., None]; edge = np.clip(1 - lay[..., :3].max(-1, keepdims=True) * 1.5, 0, 1)
    P_, K_ = char_form(Z, H_, s); pa = P_[..., 3:4] * a * edge; lay = lay.copy(); lay[..., :3] = lay[..., :3] * (1 - pa * .96) + P_[..., :3] * a * edge * .96
    br = .80 + .20 * math.sin(f * .35) + .12 * np.sin(fb((90, 40), (1, .5), 71) * 3 + f * .25)[..., None]; hot = K_[..., :3] * a * br; lay[..., :3] = lay[..., :3] * (1 - K_[..., 3:4] * a * .85) + hot; gl = gl + qb(hot * K_[..., 3:4], 9) * 1.1
    lay = rim(lay, -5, 1, '#7e2212', .95, .8, where=(x > cx).astype(f32)); return lay, m, gl


def ash_flakes(f, cx, cy, s, n=120, seed=31, sp=5.):
    """flakes torn off his back, ember edged (RGBA layer)"""
    fl = Cel(); r = np.random.default_rng(seed)
    for k in range(n):
        t = r.random(); a = r.uniform(.4, 1.2); px = cx + r.uniform(.05, .9) * s + (t ** 1.2) * 780 + f * sp * a; py = cy + r.uniform(-1.2, 2.2) * s - t * 260 - f * sp * .48 * r.uniform(.3, 1.) + 8 * math.sin(f * .3 + k); R = r.uniform(2, 13) * (1 - .5 * t); px = cx + (px - cx) % 1100
        an = np.sort(r.uniform(0, 6.283, 6)) + f * .10 * r.normal(0, 1); P = np.c_[px + np.cos(an) * R * r.uniform(.5, 1.2), py + np.sin(an) * R * r.uniform(.4, 1.)]
        if r.random() < .6: fl.line(np.vstack([P, P[:1]]), r.uniform(1.2, 2.4), '#ff8a30' if r.random() < .6 else '#ffd08a', taper=(.1, .6), seed=k)
        fl.fill(P, '#0a0506')
    return fl.layer()


def key_profile(f=0, finish=True, cx=1065, cy=452, s=318, pitch=-.03, roll=-.05):
    x, y = grid(); bg = comp.bokeh(ember_plate().astype(f32), 9, 1.0)
    bg = bg + qb(embers(f, 90, 5, 9.0), 7) * 1.0 + comp.bokeh(embers(f, 60, 15, 6.0, 0, drift=(2.2, -1.5)), 7, 0) * 1.2
    lay, Z, H_ = prof_figure(cx, cy, s, pitch, roll, f); lay, m, gl = burnt(lay, Z, cx, f, 0., H_, s)
    img = comp.over(comp.over(bg, lay), ash_flakes(f, cx, cy, s)) + gl
    img = img + embers(f, 150, 8, 2.6, .5) * 1.2 + embers(f, 40, 9, 4.2, .6, box=(900, -80, W + 80, H + 80)) * 1.3 + (radial_(x, y, 120, 1000, 900, 620, 2.) * .22)[..., None] * rgb('#ff8a3c')
    return comp.finish(img, f, bloom=.55, bloom_thr=.62, vig=.24, halation='#ff9a5a', soft=.18) if finish else img


# ================================================================= glory
@memo
def glory_plate(cx=1010, cy=330, seed=9):
    """absurd heroic light: night torn open by a golden-rose burst, painted"""
    x, y = grid(); r = np.random.default_rng(seed)
    d = np.hypot(x - cx, (y - cy) * 1.12)
    g = ramp(np.clip(d / 1250, 0, 1), [(0, '#fffbe2'), (.14, '#ffe39a'), (.34, '#ffb36a'), (.55, '#e8688a'), (.78, '#7a3a8e'), (1, '#2a1a54')])
    th = np.arctan2(y - cy, x - cx)
    ray = np.zeros((H, W), f32)
    for k in range(34):
        a0 = 2 * math.pi * k / 34 + r.normal(0, .05); wd = r.uniform(.015, .075); da = np.abs((th - a0 + math.pi) % (2 * math.pi) - math.pi); ray += np.clip(1 - da / wd, 0, 1) ** .6 * r.uniform(.35, 1)
    ray = cv2.GaussianBlur(np.clip(ray, 0, 1), (0, 0), 4) * smooth(120, 520, d)
    g = g + ray[..., None] * (rgb('#fff0b8') * .30) - ((1 - ray) * smooth(260, 900, d))[..., None] * .06
    cv = Canvas(img=np.clip(g, 0, 1.1))
    n = 900; aa = r.uniform(0, 2 * math.pi, n); rr = 230 + 1000 * r.random(n) ** .8; xs, ys = cx + np.cos(aa) * rr, cy + np.sin(aa) * rr / 1.12
    ix, iy = np.clip(xs, 0, W - 1).astype(int), np.clip(ys, 0, H - 1).astype(int); col = np.clip(cv.img[iy, ix] * (1 + r.normal(0, .09, (n, 1))), 0, 1)
    cv.strokes(arcs(xs, ys, np.arctan2(ys - cy, xs - cx), r.uniform(90, 360, n), r.normal(0, .06, n)), r.uniform(5, 26, n), col.astype(f32), op=.75, dry=.35, seed=4, shape='round')
    img = cv.img
    # city lights sunk in the corners
    lo = np.zeros((H, W, 3), f32)
    for k in range(120):
        px, py = (r.uniform(0, 620) if r.random() < .5 else r.uniform(1330, W)), r.uniform(720, 1080); rad = r.uniform(7, 24)
        cv2.circle(lo, (int(px * 8), int(py * 8)), int(rad * 8), tuple(float(v) for v in rgb(['#ffd27a', '#ff8ab0', '#8fd6ff', '#fff3d0'][k % 4]) * r.uniform(.06, .22)), -1, cv2.LINE_AA, 3)
    img = img * (1 - .35 * smooth(620, 1080, y)[..., None] * smooth(300, 900, np.abs(x - cx))[..., None])
    return img + cv2.GaussianBlur(lo, (0, 0), 1.2)


def sparkle(cel, x, y, R, col='#ffffff', rot=0., pinch=.085, alpha=1.):
    """hand-cut four-point star, long vertical arms"""
    a = np.arange(8) * math.pi / 4 + rot; rr = np.where(np.arange(8) % 2 == 0, R, R * pinch * 1.6); rr = rr * np.where(np.arange(8) % 4 == 0, .72, 1)
    P = np.c_[x + np.cos(a) * rr, y + np.sin(a) * rr]; out = []
    for i in range(8):
        p, q = P[i], P[(i + 1) % 8]; m = (p + q) / 2 * .72 + np.array([x, y]) * .28; out.append(spl([p, m, q], n=5)[:-1])
    cel.fill(np.vstack(out), col, alpha)


def para_lit(lay, p0, p1, stops, amt=1., lo=.10, hi=.42):
    """screen gradient that spares the blacks (ink and black hair stay black)"""
    n = para(lay, p0, p1, stops, 'screen', amt); a = np.maximum(lay[..., 3], 1e-4); L = (lay[..., :3] * np.array([.3, .59, .11], f32)).sum(-1) / a
    k = smooth(lo, hi, L)[..., None]; out = lay.copy(); out[..., :3] = lay[..., :3] + (n[..., :3] - lay[..., :3]) * k; return out


def key_todo(f=0, finish=True, cx=1035, cy=292, s=138, yaw=.18, pitch=.20, roll=.09, expr='weep', pal=None, hands=None, lift=0.):
    x, y = grid(); bg = glory_plate().astype(f32).copy(); pal = dict(TPAL['glory']) if pal is None else pal
    k = Cel(); r = np.random.default_rng(17)
    from jj.kit import focus_lines
    focus_lines(k, cx, cy + 30, 560, 1500, 120, 5.0, '#fff6d8', 3, ry=.78); fl = k.layer(); bg = comp.screen(bg, cv2.GaussianBlur(fl, (0, 0), .8)[..., :3] * .55)
    c = Cel(); lw = s / 300. * 9.0; lit = -1
    H_ = Todo.pose(cx, cy, s, yaw, pitch, roll); b = TBody.under(H_, .10, .24, .055, drop=1.28, fwd=.10, persp=.115)
    hp = hands or (lambda q: (lambda cel, w, e_, fz: hand3(cel, w, e_, fz, pal, lw, 'L' if q > 0 else 'R', 'open', 188 if q > 0 else 168, 16 if q > 0 else 24, 8 * q, .74, seed=4 + q)))
    kw = dict(f=f, wind=.10, seed=2, lit=lit, lw=lw)
    Todo.head(c, H_, expr, pal, part='back', **kw)
    Todo.trunk(c, b, pal, lw, lit)
    Todo.arm(c, b, (2.08, -.34 + lift * .5, -.1), (3.58, .10 - lift * .8, .45), (4.62, 1.22 - lift * 1.6, .85), pal, lw, 1, 1, lit, 3, hp(1), bulk=1.2)
    Todo.arm(c, b, (-2.08, -.34 + lift * .5, -.1), (-3.50, -.66 - lift * .8, 1.0), (-4.12, .02 - lift * 1.6, 2.1), pal, lw, -1, -1, lit, 5, hp(-1), bulk=1.2)
    Todo.neck(c, H_, b, pal, lw, lit)
    Todo.head(c, H_, expr, pal, part='front', **kw)
    lay = c.layer()
    lay = rim(lay, 5, 5, '#fff2b0', .95, .6); lay = rim(lay, 2, 2, '#ffffff', .8, .5); lay = rim(lay, -5, 1, '#ff9ac0', .75, .6)
    lay = para_lit(lay, (cx - 420, 0), (cx + 60, 620), [(0, '#6a4a1c'), (.5, '#22100a'), (1, '#000000')], .75)
    lay = para(lay, (cx - 60, 480), (cx + 160, 1080), [(0, '#ffffff'), (.5, '#d8a8b8'), (1, '#8f5f9c')], 'mul', .95)
    bg = aura(bg, lay, 18, '#fff0b0', .75); img = comp.over(bg, lay)
    # ---- sparkles: composed, big to tiny
    g = Cel(); pts = [(560, 120, 92), (1500, 560, 74), (1290, 92, 60), (700, 560, 46), (1660, 60, 52), (330, 640, 66), (1780, 700, 42), (820, 60, 30), (1390, 330, 34), (150, 420, 34), (1240, 640, 26), (470, 330, 26)]
    for i, (px, py, R) in enumerate(pts):
        tw = .72 + .28 * math.sin(f * .7 + i * 2.1); sparkle(g, px, py, R * tw * 1.15, '#fff3c0', .06 * (i % 3 - 1), .10, .9); sparkle(g, px, py, R * tw * .72, '#ffffff', .06 * (i % 3 - 1), .10)
    for i in range(80):
        a_ = r.uniform(0, 6.283); d_ = r.uniform(180, 1050); px, py = cx + math.cos(a_) * d_, cy + 140 + math.sin(a_) * d_ * .62; tw = .5 + .5 * math.sin(f * .9 + i * 1.7); sparkle(g, px, py, r.uniform(4, 13) * (.5 + .5 * tw), '#fffbe6', r.uniform(-.2, .2), .16, .5 + .5 * tw)
    gl = g.layer(); img = comp.over(img, gl); img = img + qb(gl[..., :3], 12) * .55 * rgb('#ffdf90') + cv2.GaussianBlur(gl[..., :3], (0, 0), 3.5) * .4
    img = comp.flare(img, cx - 40, 20, 900, '#ffe6a0', .16) if hasattr(comp, 'flare') else flare(img, cx - 40, 20, 900, '#ffe6a0', .16)
    return comp.finish(img, f, bloom=.34, bloom_thr=.8, vig=.14, halation='#ffd9a8', soft=.15) if finish else img


if __name__ == '__main__':
    cv2.setNumThreads(4)
    which = sys.argv[1]
    im = globals()['key_' + which]()
    comp.save('qa/nanami/key_%s.jpg' % which, im, 94)
