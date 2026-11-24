"""Painted skies.  every function returns (h,w,3) float32; hz = horizon y in px (may lie outside), all seeded."""
import numpy as np, cv2, math
from .paint import rgb, f32, fbm, smooth, mix, grid, Canvas, arcs, jit_color, ramp, warp_by, lum
from .comp import blur, radial, screen
from . import clouds
from .world import C, painter, Plate, gouache, tooth_pass


def gradient(h, w, hz, stops, bow=0., cx=None, seed=0, wob=.012):
    """vertical ramp measured from the horizon up (0) to the zenith (1 = frame top); bow dishes it toward cx"""
    x, y = grid(h, w); cx = w / 2 if cx is None else cx
    t = (hz - y) / max(hz, 1) + bow * ((x - cx) / w) ** 2 + wob * fbm(h, w, (420, 150), (1, .5), seed + 1)
    return ramp(np.clip(t, 0, 1.2), stops)


def brush_sky(img, seed=0, amt=.55, sizes=(44, 20), val=(.018, .016), tilt=0.):
    """broad horizontal sky strokes"""
    h, w = img.shape[:2]
    a = fbm(h, w, (500, 180), (1, .5), seed + 5) * .16 + tilt
    return gouache(img, None, a, seed=seed + 9, sizes=sizes, op=(amt, amt * .8), val=val, hue=.006, dry=.35, thr=(.02, .03), aspect=5.5, wander=.06, bristle=.03)


def wisps(img, n, y0, y1, cols, seed=0, L=(120, 620), th=(2, 14), op=(.06, .24), tilt=-.05, x=None, curve=.12, power=1.):
    """streaky cirrus / stratus strokes, dry brush"""
    h, w = img.shape[:2]; r = np.random.default_rng(seed + 31); cv = Canvas(h, w, img=np.clip(img, 0, 1)); ext = img - np.clip(img, 0, 1)
    x = (-.05 * w, 1.05 * w) if x is None else x
    q = r.random(n) ** power; y = y0 + (y1 - y0) * q
    xx = r.uniform(*x, n); ln = r.uniform(*L, n) * (.35 + .65 * q); c = np.stack([C(cols[k]) for k in r.integers(0, len(cols), n)]) * (1 + r.normal(0, .02, (n, 1)).astype(f32))
    P = arcs(xx, y, tilt + r.normal(0, .035, n), ln, r.normal(0, curve, n), K=9)
    cv.strokes(P, r.uniform(*th, n) * (.4 + .6 * q), np.clip(c, 0, 1), op=r.uniform(*op, n), dry=r.uniform(.1, .75, n), seed=seed, shape='round', bristle=.05)
    return cv.img + ext


def cirrus(img, y0, y1, col, shade=None, amt=.6, ang=-.12, seed=0, aspect=7., size=1., cover=.5, fringe=1.):
    """feathered high cloud: combed filaments in torn patches, cut on the sun side, lost on the other"""
    h, w = img.shape[:2]; x, y = grid(h, w)
    R = int(math.hypot(h, w)) + 8
    n = fbm(R, R, (64 * size, 28 * size, 12 * size, 5 * size), (1, .72, .5, .34), seed + 1, aspect=aspect)
    b = fbm(R, R, (190 * size, 80 * size), (1, .55), seed + 2, aspect=2.6)
    n = warp_by(n, 16 * size * min(1., 5. / aspect) ** 2, 90 * size, seed + 3); f = n * .62 + b * 1.0
    M = cv2.getRotationMatrix2D((R / 2, R / 2), -math.degrees(ang), 1.); M[0, 2] += (w - R) / 2; M[1, 2] += (h - R) / 2
    f = cv2.warpAffine(f.astype(f32), M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    env = smooth(y0 - (y1 - y0) * .25, y0 + (y1 - y0) * .25, y) * (1 - smooth(y1 - (y1 - y0) * .35, y1 + (y1 - y0) * .1, y))
    th = 1.15 - 1.5 * cover
    m = smooth(th, th + 1.15, f) * env
    edge = np.clip((f - np.roll(f, (6, 3), (0, 1))) * 1.1 + .45, 0, 1)[..., None]
    c = C(col) if shade is None else C(shade) + (C(col) - C(shade)) * edge
    return img + (c - img) * (m * amt)[..., None]


def puffs(img, n, box, cols, seed=0, size=(14, 60), flat=.5, op=(.2, .7), soft=0):
    """small torn cloud dabs clustered along bars: scud lit from one side"""
    h, w = img.shape[:2]; r = np.random.default_rng(seed + 77); cv = Canvas(h, w, img=np.clip(img, 0, 1))
    x0, y0, x1, y1 = box; nb = max(3, n // 24)
    bx = r.uniform(x0, x1, nb); by = r.uniform(y0, y1, nb); bl = r.uniform(.04, .16, nb) * w
    k = r.integers(0, nb, n); xx = bx[k] + r.normal(0, 1, n) * bl[k] * .5; yy = by[k] + r.normal(0, 1, n) * bl[k] * .045
    s = r.uniform(*size, n); c = np.stack([C(cols[j]) for j in r.integers(0, len(cols), n)])
    P = arcs(xx, yy, r.normal(0, .12, n), s * r.uniform(1., 2.6, n), r.normal(0, .5, n), K=6)
    cv.strokes(P, s * flat * r.uniform(.3, .8, n), np.clip(c * (1 + r.normal(0, .02, (n, 1)).astype(f32)), 0, 1), op=r.uniform(*op, n), dry=r.uniform(.1, .6, n), seed=seed, shape='round', bristle=.05)
    o = cv.img
    if soft: o = blur(o, soft)
    return o + (img - np.clip(img, 0, 1))


def night(h, w, hz, seed=0, glow='#7a4a7c', glow2='#2f6f8e', top='#0b1026', mid='#1a2654', gx=.5, stars=90, clouds_amt=1.):
    """city night: ink-blue zenith, violet smog glow on the horizon, torn cloud lit from below"""
    x, y = grid(h, w)
    img = gradient(h, w, hz, [(0, mix(C(glow), C('#e0909a'), .22)), (.10, C(glow)), (.26, mix(C(glow), C(mid), .55)), (.50, mix(C(glow), C(mid), .88)), (.8, mix(C(mid), C(top), .6)), (1.1, C(top))], bow=.25, cx=w * gx, seed=seed)
    g2 = np.exp(-((x - w * (gx + .28)) / (w * .25)) ** 2) * np.exp(-np.clip(hz - y, 0, None) / (h * .16))
    img = img + (C(glow2) - img) * (.42 * g2[..., None])
    if clouds_amt > 0:
        n1 = fbm(h, w, (w * .16, w * .06, w * .024, w * .01), (1, .62, .4, .24), seed + 11, aspect=3.4)
        el = np.clip((hz - y) / max(hz, 1), 0, 1)
        n1 = warp_by(n1, 10, 120, seed + 3)
        m = smooth(.0, .8, n1 + .35 - .55 * el) * smooth(0, .10, el) * (1 - .5 * smooth(.5, 1., el))
        n2 = np.roll(n1, (int(h * .018), int(-w * .006)), (0, 1)); under = np.clip((n1 - n2) * 1.5 + .35, 0, 1)
        lit = mix(C(glow), C('#f0b0b0'), .34); dark = mix(C(mid), C(top), .35)
        col = dark + (lit * .9 - dark) * (under * (1 - .8 * smooth(0, .7, el)))[..., None]
        img = img + (col - img) * (m * .72 * clouds_amt)[..., None]
    if stars:
        r = np.random.default_rng(seed + 4); xs = r.uniform(0, w, stars); ys = r.uniform(0, hz * .7, stars)
        for a, b, v in zip(xs, ys, r.random(stars) ** 3):
            i, j = int(b), int(a)
            if i < h - 2 and j < w - 2 and i > 1 and j > 1: img[i, j] += (.18 + .5 * v) * C('#cfe0ff'); img[i, j + 1] += .12 * v
    img = brush_sky(img, seed, .5, val=(.014, .012))
    return np.maximum(img, 0)


def _hmix(a, b, t): return [mix(C(p), C(q), t) for p, q in zip(a, b)]


def sunset(h, w, hz, sun=(.62, .0), seed=0, towers=True, strength=1.):
    """huge evening sky: gold at the horizon through rose to violet-blue, backlit cumulus towers, streak cirrus.  sun=(x frac, px above hz)"""
    sx, sy = w * sun[0], hz - sun[1]
    x, y = grid(h, w)
    img = gradient(h, w, hz, [(0, '#ffd27a'), (.07, '#ffb765'), (.17, '#f7946a'), (.30, '#d97b83'), (.45, '#a1709c'), (.64, '#6a67a3'), (.85, '#46558f'), (1.1, '#344478')], bow=.10, cx=sx, seed=seed)
    d = np.sqrt(((x - sx) / (w * .42)) ** 2 + ((y - sy) / (h * .42)) ** 2)
    img = img + (C('#ffe0a0') - img) * (np.clip(1 - d, 0, 1) ** 1.6 * .72)[..., None]
    img = cirrus(img, -hz * .05, hz * .50, '#f2a59c', '#8d78b2', .62, -.20, seed + 2, 6., 1.25, .46)
    img = cirrus(img, hz * .40, hz * .86, '#ffd09a', '#c9869c', .50, -.05, seed + 21, 9., .7, .40)
    img = brush_sky(img, seed + 4, .4)
    if towers:
        S = clouds.Sky(h, w, img, light=((sx - w * .5) / w * 1.2, -.18, -.22), seed=seed)
        T_far = ('#ffd9a2', '#e8a08e', '#b07f9a', '#7f6f9e'); T_mid = ('#ffe0ac', '#f0a184', '#9c6f94', '#645c92'); T_hi = ('#ffcfa4', '#d98f94', '#8a6d9c', '#56568c')
        r = np.random.default_rng(seed + 8)
        def bk(fx, base, fw, fh, T, mist, sd, prof, **k): S.bank(x=w * fx, base=hz - base * hz, width=w * fw, height=hz * fh, T=T, mist=mist, seed=sd, profile=prof, **k)
        lt = lambda fx: (float(np.clip((sun[0] - fx) * 1.6, -.8, .8)), -.35, -.10)
        bk(.10, .02, .34, .20, T_far, '#f3b184', seed + 1, clouds.peak(.6, .3), light=lt(.10), lit=.16, half=.30, soft=.6, rim='#ffe9b8', rim_amt=.8)
        bk(.93, .01, .40, .26, T_far, '#f0a886', seed + 2, clouds.peak(.35, .25), light=lt(.93), lit=.16, half=.30, soft=.6, rim='#ffe9b8', rim_amt=.8)
        bk(.22, .16, .36, .46, T_mid, '#b883a0', seed + 3, clouds.peak(.42, .22, .30, .78, .55), light=lt(.22), lit=.14, half=.26, n=6, rim='#fff0c2', rim_amt=1.)
        bk(.86, .20, .42, .56, T_hi, '#a87aa4', seed + 4, clouds.peak(.60, .20, .28, .25, .55), light=lt(.86), lit=.13, half=.25, n=6, rim='#fff0c2', rim_amt=1.)
        bk(.50, .62, .30, .10, T_hi, '#8c74a6', seed + 5, clouds.peak(.5, .4), light=lt(.5), lit=.2, half=.3, soft=1.2, rim='#ffd9b0', rim_amt=.7)
        bk(.40, .03, .26, .10, T_far, '#ffc07c', seed + 6, clouds.peak(.4, .3), light=lt(.4), lit=.2, half=.3, soft=.8, rim='#fff0c8', rim_amt=.9)
        S.strata(hz * .80, hz * .985, n=46, cols=['#ffd48c', '#f19c74', '#d68088', '#ffe3ac'], seed=seed, th=(1., 4.2), op=(.25, .7), power=.6)
        img = S.paint(seed=seed + 5)
        d2 = np.clip(1 - d * 1.15, 0, 1) ** 2.2
        img = img + (C('#ffdf9c') - img) * (.5 * d2)[..., None]
    img = cirrus(img, hz * .76, hz * .99, '#fff2c8', '#f0a080', .50, -.01, seed + 12, 16., .5, .40)
    g = np.exp(-d * 4.2)[..., None]; img = img + C('#fff0cc') * g * .55 * strength
    sd = np.sqrt((x - sx) ** 2 + (y - sy) ** 2)
    img = img + (C('#fff8e2') * 1.25 - img) * smooth(h * .030, h * .026, sd)[..., None]
    return np.maximum(img, 0)


def golden(h, w, hz, sun=(.2, 420), seed=0):
    """late afternoon: warm-white haze low, clear blue-green above, tall sunlit cumulus"""
    sx, sy = w * sun[0], hz - sun[1]; x, y = grid(h, w)
    img = gradient(h, w, hz, [(0, '#ffe9c0'), (.08, '#ffe2b4'), (.2, '#f3dcbc'), (.36, '#c6d6cf'), (.58, '#86b9d0'), (.85, '#4f94c2'), (1.1, '#3a7fb4')], bow=.06, cx=sx, seed=seed)
    d = np.sqrt(((x - sx) / (w * .5)) ** 2 + ((y - sy) / (h * .6)) ** 2); img = img + (C('#fff2cf') - img) * (np.clip(1 - d, 0, 1) ** 1.5 * .75)[..., None]
    img = cirrus(img, -hz * .05, hz * .55, '#fff6e2', '#c4dbe0', .62, -.30, seed + 2, 5., 1.5, .42)
    img = brush_sky(img, seed + 4, .35)
    S = clouds.Sky(h, w, img, light=(-.62 if sun[0] < .5 else .62, -.62, .30), seed=seed)
    T = ('#fff6dc', '#ffdcae', '#c9b6bf', '#9aa3c0'); T2 = ('#fff3d4', '#f6cfa6', '#bda9b8', '#8f99ba')
    S.bank(w * .78, hz * .99, w * .62, hz * .66, T, '#f4d9b4', seed=seed + 1, profile=clouds.peak(.55, .20, .3, .2, .5), n=6, lit=.30, half=.28)
    S.bank(w * .16, hz * .995, w * .36, hz * .24, T2, '#f8dfba', seed=seed + 2, profile=clouds.peak(.4, .3), n=6, lit=.30, half=.3, soft=.5)
    S.bank(w * .46, hz * .995, w * .30, hz * .14, T2, '#fbe3bd', seed=seed + 3, profile=clouds.peak(.5, .4), lit=.3, half=.3, soft=.8)
    S.bank(w * .34, hz * .42, w * .20, hz * .07, T2, '#b8cfd4', seed=seed + 4, profile=clouds.peak(.5, .4), lit=.4, half=.3, soft=1.4)
    S.strata(hz * .9, hz * .99, n=26, cols=['#fff0cf', '#f5d3ae', '#e2c3b6'], seed=seed, th=(1, 4), op=(.2, .6), power=.7)
    return np.maximum(S.paint(seed=seed + 5), 0)


def bluehour(h, w, hz, seed=0, gx=.3):
    """blue hour: deep ultramarine, last apricot band, one thin violet bar of cloud"""
    img = gradient(h, w, hz, [(0, '#f2a274'), (.06, '#e59a86'), (.15, '#b78aa2'), (.28, '#7a80b4'), (.48, '#4767a8'), (.72, '#2b4a8a'), (1.1, '#1b3268')], bow=.12, cx=w * gx, seed=seed)
    img = wisps(img, 60, hz * .35, hz * .97, ['#6a6aa6', '#d98f94', '#4c5c9c', '#f0b08c'], seed + 2, L=(260, 1000), th=(2, 12), op=(.10, .42), tilt=-.04, power=.7, curve=.06)
    img = puffs(img, 150, (0, hz * .4, w, hz * .85), ['#3f5296', '#5664a6', '#8a78ac'], seed + 3, size=(16, 70), flat=.4, op=(.12, .4))
    r = np.random.default_rng(seed + 4)
    for a, b, v in zip(r.uniform(0, w, 26), r.uniform(0, hz * .4, 26), r.random(26) ** 2):
        i, j = int(b), int(a)
        if 1 < i < h - 2 and j < w - 2: img[i, j] += .2 + .5 * v; img[i, j + 1] += .15 * v
    return np.maximum(brush_sky(img, seed, .45), 0)


def storm(h, w, hz, seed=0, tonesT=('#59606f', '#3a4252', '#262c3a', '#161a25'), light=(-.2, -.7, .5), hzc='#6b7484', top='#1c212d'):
    """low rolling storm ceiling: shelf cloud bellies in perspective, pale slot at the horizon"""
    x, y = grid(h, w)
    el = np.clip((hz - y) / max(hz, 1), 0, 1)
    img = ramp(el, [(0, hzc), (.10, mix(C(hzc), C(tonesT[1]), .5)), (.35, tonesT[2]), (1, top)])
    # cloud ceiling plane: perspective noise on a plane overhead
    v = 1 / np.maximum(el, .02); u = (x - w * .5) / w * 2 * v
    def samp(sx, sy, sd, oct=4):
        n = fbm(1024, 1024, (130, 60, 27, 12), (1, .6, .38, .22), sd)
        return cv2.remap(n, ((u * sx + 3.3) * 160 % 1023).astype(f32), ((v * sy) * 150 % 1023).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    def samp2(sx, sy, sd, du=0., dv=0.):
        n = fbm(1400, 2600, (190, 84, 36, 15), (1, .6, .40, .24), sd)
        uu = np.clip((u + du) * sx * 150 + 1300, 1, 2598); vv = np.clip((v + dv - 1) * sy * 130 + 20, 1, 1398)
        return cv2.remap(n, uu.astype(f32), vv.astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    n = samp2(.95, 1.25, seed + 1); n2 = samp2(.95, 1.25, seed + 1, .05, .07)
    s = np.clip((n - n2) * 2.0, -1.2, 1.2); f = smooth(.02, .20, el)
    T = [C(c) for c in tonesT]
    t = np.clip(.46 + .30 * s + .16 * n, 0, 1)
    c = ramp(t, [(0, T[3]), (.30, T[2]), (.36, T[2]), (.50, T[1]), (.64, T[1]), (.76, T[0]), (1, mix(T[0], C('#ffffff'), .12))])
    img = img + (c - img) * (f * .92)[..., None]
    img = img * (1 - .35 * smooth(.45, 1., el))[..., None] + C(top) * (.35 * smooth(.45, 1., el))[..., None]
    a = np.arctan2(y - hz, x - w * .5) * .0 + fbm(h, w, (300, 100), (1, .5), seed + 8) * .5
    img = gouache(img, None, a, seed=seed + 3, sizes=(40, 18, 8), op=(.5, .5, .45), val=(.022, .02, .016), hue=.006, thr=(.02, .03, .05), aspect=3.6, wander=.35, bristle=.04)
    return np.maximum(img, 0)


def dawn(h, w, hz, seed=0, gx=.72, snow=True):
    """blue winter dawn: cold cobalt above, thin peach seam at the horizon, soft broken overcast"""
    img = gradient(h, w, hz, [(0, '#f6cdb4'), (.05, '#e9c0bd'), (.13, '#b9b6cf'), (.27, '#8aa2cc'), (.5, '#5c80b6'), (.78, '#3c5f9a'), (1.1, '#2b4a80')], bow=.16, cx=w * gx, seed=seed)
    x, y = grid(h, w)
    S = clouds.Sky(h, w, img, light=(.5, -.2, .15), seed=seed)
    T = ('#f6d6c4', '#bdb8d2', '#8497c6', '#5d7ab0'); T2 = ('#dcd2dd', '#a3aed4', '#728cc0', '#5070a8')
    S.bank(w * .22, hz * .90, w * .95, hz * .13, T2, '#8ea4cf', seed=seed + 1, profile=clouds.peak(.6, .35), lit=.16, half=.3, soft=2.6, tiers=3, n=8)
    S.bank(w * .98, hz * .86, w * .80, hz * .11, T, '#a7aed0', seed=seed + 2, profile=clouds.peak(.35, .3), lit=.2, half=.3, soft=2.2, n=7)
    S.bank(w * .58, hz * .62, w * .90, hz * .08, T2, '#7c94c6', seed=seed + 4, profile=clouds.peak(.4, .5), lit=.14, half=.3, soft=3.4, n=9)
    S.strata(hz * .84, hz * .985, n=44, cols=['#f9d6ba', '#c9bcd2', '#f0c2b2', '#98a8d0'], seed=seed, th=(1, 4.5), op=(.2, .6), power=.7)
    img = S.paint(seed=seed + 5)
    img = cirrus(img, -hz * .05, hz * .62, '#c9cbe2', '#7f9ccd', .55, .10, seed + 6, 5., 1.6, .52)
    img = brush_sky(img, seed + 7, .32, sizes=(22, 10))
    d = np.sqrt(((x - w * gx) / (w * .34)) ** 2 + ((y - hz) / (h * .16)) ** 2); img = img + (C('#ffdcb8') - img) * (np.clip(1 - d, 0, 1) ** 1.7 * .7)[..., None]
    return np.maximum(img, 0)


def fire(h, w, hz, seed=0, fx=.55):
    """burning city sky: smoke ceiling lit orange from below, black above"""
    s = storm(h, w, hz, seed, ('#c4532a', '#7d2c1f', '#3f1718', '#190b0f'), hzc='#ff9a4a', top='#0d070b')
    x, y = grid(h, w); el = np.clip((hz - y) / max(hz, 1), 0, 1)
    g = np.exp(-((x - w * fx) / (w * .36)) ** 2) * np.exp(-el * 2.6)
    s = s + C('#ff6a22') * (.55 * g)[..., None]
    return s * (1 - .42 * smooth(.25, 1., el))[..., None]


def day(h, w, hz, seed=0, sunx=.8):
    """white tropical light"""
    img = gradient(h, w, hz, [(0, '#f4fbff'), (.10, '#dff3fb'), (.3, '#a9dcf2'), (.6, '#62b9e6'), (1.1, '#2f93d6')], bow=.1, cx=w * sunx, seed=seed)
    img = brush_sky(cirrus(img, -hz * .05, hz * .6, '#ffffff', '#d4edf8', .6, -.34, seed + 7, 5., 1.6, .44), seed, .3)
    S = clouds.Sky(h, w, img, light=(.5 if sunx > .5 else -.5, -.75, .4), seed=seed)
    T = ('#ffffff', '#f4fafe', '#c9e2f2', '#a3c9e4')
    S.bank(w * .16, hz * 1.0, w * .40, hz * .50, T, '#e8f6fc', seed=seed + 1, profile=clouds.peak(.45, .22, .3, .8, .5), n=6, lit=.4, half=.3)
    S.bank(w * .55, hz * 1.0, w * .34, hz * .2, T, '#eef9fd', seed=seed + 2, profile=clouds.peak(.5, .3), lit=.4, half=.3, soft=.8)
    S.bank(w * .90, hz * 1.0, w * .36, hz * .3, T, '#eef9fd', seed=seed + 3, profile=clouds.peak(.6, .25), lit=.4, half=.3, soft=.4)
    return np.maximum(S.paint(seed=seed + 5), 0)


KINDS = dict(night=night, sunset=sunset, golden=golden, bluehour=bluehour, storm=storm, dawn=dawn, fire=fire, day=day)


@painter('sky', 3)
def sky_plate(kind='sunset', w=2400, h=1350, hz=None, seed=0):
    """big painted sky: kind night|sunset|golden|bluehour|storm|dawn|fire|day.  hz = horizon y (default below the frame bottom)"""
    hz = h * 1.02 if hz is None else hz
    return Plate(tooth_pass(KINDS[kind](h, w, hz, seed=seed)))
