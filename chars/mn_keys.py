"""Full-finish hero keys of Megumi, Nobara and the dog (also the look targets for shots d1, d2)."""
import numpy as np, cv2, math
from jj import comp, W, H
from jj.paint import rgb, f32, fbm, smooth, ramp, grid
from jj.ink import Cel, spl
from jj.kit import rim, para, splatter, focus_lines, aura, zoomblur, outline, shape, hatch, shifted
from chars import mn, dog


def hpal(L):
    """jj.hands palette from one of our light modes"""
    return dict(skin=L['skin'], shade=L['skin2'], line=L['line'], nail=mn.mixh(L['skin'], L['skinhi'], .45), nail_shade=L['skin2'], light=L['skinhi'], cuff=L['cloth'], cuff_shade=L['cloth_dk'], cuff_light=L['cloth_hi'], cuffin='#08070c', prop='#6f4c34', prop_shade='#3a2734')


def licks(x0, x1, y, h, n, seed=0, lean=.0, hook=.45, f=0, rise=0., base=1200, boil=0., grow=1.):
    """one boiling ink mass: a ground line of hooked flame-like crests, every crest its own size.  returns a closed polygon"""
    r = np.random.default_rng(seed); e = np.sort(np.linspace(x0, x1, n + 1) + r.normal(0, .22 * (x1 - x0) / n, n + 1)); P = []
    bz = lambda p0, p1, p2, p3, m=16: (lambda u: (1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * p1 + 3 * (1 - u) * u * u * p2 + u ** 3 * p3)(np.linspace(0, 1, m)[:, None])
    tr = [np.array([x, y + r.uniform(-.06, .16) * h]) for x in e]
    for i in range(n):
        a, b = tr[i], tr[i + 1]; w = b[0] - a[0]; k = r.choice([.30, .6, 1.0, 1.55], p=[.28, .30, .27, .15]); hh = h * k * r.uniform(.8, 1.2) * (1 + rise * math.sin(f * .55 + i * 2.1)); s = 1 if r.random() < .5 + lean * .5 else -1
        T_ = r.uniform(11, 22); ph = r.uniform(0, 6.28); hk0 = r.uniform(.5, 1.2); sw = 0.
        if boil: hh *= (1 + boil * math.sin(6.283 * f / T_ + ph)); hk0 *= 1 + boil * .8 * math.sin(6.283 * f / (T_ * 1.4) + ph * 2); sw = boil * .22 * hh * math.sin(6.283 * f / (T_ * .9) + ph * 3)
        hh *= grow; hk = hook * hh * s * hk0; tip = np.array([(a[0] + b[0]) / 2 + lean * hh + hk * .8 + sw, min(a[1], b[1]) - hh])
        if s > 0: P += [bz(a, a + [w * .02, -hh * .42], tip + [-hk * 1.0 - w * .10, hh * .30], tip), bz(tip, tip + [-hk * .55, hh * .22], b + [-w * .12 - hk * .25, -hh * .40], b)]
        else: P += [bz(a, a + [w * .12 - hk * .25, -hh * .40], tip + [-hk * .55, hh * .22], tip), bz(tip, tip + [-hk * 1.0 + w * .10, hh * .30], b + [-w * .02, -hh * .42], b)]
    P = np.vstack(P); return np.vstack([P, [(x1 + 200, base), (x0 - 200, base)]])


_NZ = {}


def nz(scales, w, seed):
    k = (scales, w, seed)
    if k not in _NZ: _NZ[k] = fbm(H, W, scales, w, seed)
    return _NZ[k]


def boil_layer(polys, col, hi, line, w=2.6, inner=(16, 18), hi2=None, rimc=None, drops=0, seed=0, box=(0, W, 200, 900)):
    """ink mass with a hard blue light shape along its lit edges; returns a layer"""
    c = Cel(); r = np.random.default_rng(seed)
    polys = list(polys)
    for q in range(drops):                                   # thrown drops: teardrops, hooked
        x, y = r.uniform(box[0], box[1]), r.uniform(box[2], box[3]); sz = 4 + 20 * r.random() ** 2.2; a = r.uniform(-.7, .7) - math.pi / 2; e = np.array([math.cos(a), math.sin(a)]); n = np.array([-e[1], e[0]]) * r.choice([-1, 1]); p0 = np.array([x, y])
        polys.append(spl([p0 - e * sz * .55, p0 - n * sz * .5, p0 + e * sz * .9 - n * sz * .15, p0 + e * sz * 2.1 + n * sz * .5, p0 + e * sz * .5 + n * sz * .40], True, corners=(3,), n=6))
    for i, P in enumerate(polys): outline(c, P, w * 1.6, line, seed + i, tremor=.2)
    for P in polys: c.fill(P, col)
    lay = c.layer(); a = lay[..., 3]
    K = 7; t = np.zeros_like(a)                              # depth from the lit edge, 0 at the edge .. 1 inside
    for k in range(1, K + 1): t += dog_shift(a, inner[0] * 1.7 * k / K, inner[1] * 1.7 * k / K)
    t = t / K; n = nz((130, 60, 26), (1, .55, .25), seed + 3); th = np.clip(.50 + n * .30, .0, .95)
    m1 = smooth(.02, -.02, t - th) * (a > .5); lay[..., :3] = lay[..., :3] * (1 - m1[..., None]) + rgb(hi) * m1[..., None]
    if hi2:
        m2 = smooth(.02, -.02, t - np.clip(.10 + n * .16, 0, .6)) * (a > .5); lay[..., :3] = lay[..., :3] * (1 - m2[..., None]) + rgb(hi2) * m2[..., None]
    return dog.rim2(lay, 4, 5, rimc, .85, .6) if rimc else lay


def nzs(shape, scales, w, seed):
    """cached noise of any size"""
    k = (tuple(shape), scales, w, seed)
    if k not in _NZ: _NZ[k] = fbm(shape[0], shape[1], scales, w, seed)
    return _NZ[k]


def facets(lay, dx, dy, col, th0=.50, amt=.30, seed=3, sc=(130, 60, 26), only=None):
    """hard light shapes along the edges that face (-dx,-dy), their width broken by noise"""
    a = lay[..., 3]; K = 7; t = np.zeros_like(a)
    for k in range(1, K + 1): t += dog_shift(a, dx * 1.7 * k / K, dy * 1.7 * k / K)
    m = smooth(.02, -.02, t / K - np.clip(th0 + nzs(a.shape, sc, (1, .55, .25), seed) * amt, 0, .95)) * (a > .5)
    if only is not None: m = m * only
    out = lay.copy(); out[..., :3] = out[..., :3] * (1 - m[..., None]) + rgb(col) * m[..., None]; return out


def dog_shift(a, dx, dy): return cv2.warpAffine(a, np.float32([[1, 0, dx], [0, 1, dy]]), (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def key1(f=0, seed=3):
    """Megumi: the shadow boils, the sign in front of his face"""
    from jj import hands as HD
    from chars.megumi import Megumi as M, BODY, EXPR
    P = dog.PAL['battle']; x, y = grid(); L = dict(mn.LIGHT['battle'])
    # night: indigo air, a cold light well low behind him
    n = fbm(H, W, (420, 160, 60), seed=8); d = np.sqrt(((x - 1120) / 1150) ** 2 + ((y - 760) / 820) ** 2)
    bg = ramp(np.clip(d + n * .05, 0, 1.4), [(0, '#5f9af0'), (.22, '#2f56c4'), (.5, '#182b78'), (.85, '#0b1136'), (1.4, '#070819')])
    # the wolf in the dark behind him
    c = Cel(); dk = dict(P, base='#0b1030', sh='#070a20', hi='#16245f', hi2='#2c58c0', nose='#0d1338', nosehi='#274ba6', gum='#1b1030', gumhi='#2a1846', gumdk='#100a20', tongue='#1c1436', tonguehi='#2a2050', tooth='#7fa2e6', toothsh='#3f5fb4', toothdk='#263c86', mark='#8fb4f4', lip='#080a1c', line='#040616'); dog.PAL['_far'] = dk
    dog.fur_fill(c, dog.mane(430, 360, 300, 270, 30, 470, 170, 12, swirl=.28), dk, 3.0, 4, wedge=.5)
    D = dog.DogHead().pose(455, 330, 225, .42, .02, .05, jaw=.42, persp=.28); dog.head(c, D, '_far', f, seed, w=2.8, eyes=1)
    lay = dog.rim2(c.layer(), -4, 5, '#3f7be8', .7, .8); bg = comp.over(bg, comp.blur(lay, 1.2))
    for q in (-1, 1):
        e = D.P3(np.array([(q * .44, .40, .44)]))[0]; bg = comp.glow(bg, e[0], e[1], 210, '#5fb0ff', .75, 2.4)
    # far shadow tongues
    bg = comp.over(bg, boil_layer([licks(-80, 2000, 960, 330, 15, 5, lean=.10)], '#0b1136', '#1d378f', '#050819', 2.4, (14, 12), rimc='#4d93f2', drops=14, seed=5, box=(700, 1900, 250, 760)))
    # Megumi
    s = 262; cx, cy = 1190, 440
    cel = Cel(); Hd = M.pose(cx, cy, s, -.30, -.10, -.06); w = mn.lw(s); hair = M._hair(Hd, f, (.12, -.16, .3)); T = mn.torso_under(Hd, yaw=-.22, roll=.05, drop=1.10)
    M.back(cel, Hd, 'battle', f, (0, 0), seed, hair); mn.uniform(cel, T, BODY, L, w, seed, 'back'); mn.neck(cel, Hd, L, w, .335, seed=seed); mn.uniform(cel, T, BODY, L, w, seed, 'front')
    M.head(cel, Hd, dict(EXPR['focus'], gaze=(-.012, .004)), 'battle', f, (0, 0), seed, parts='f', L=hair)
    body = mn.grade(cel.layer(), L, (cx - 420, 60, cx + 420, 1080), 1.0)
    # hands: far palm first, its sleeve, then the near hand
    hc = Cel(); pal = hpal(L)
    cam = HD.Cam(985, 772, 190, 7.5)
    far = HD.Hand('dog_far', 'R', HD.euler(180 - 10, 6, 68), (0.03, .02, -.17), HD.SLIM); near = HD.Hand('dog_near', 'L', HD.euler(-10, 6, 68), (0, 0, .165), HD.SLIM)
    W_ = lambda h, y_: cam.px(h.world(np.array([0, y_, 0])))
    a = W_(far, -.10); mn.sleeve2(hc, [a + [26, 6], (1010, 960), (900, 1110), (800, 1260)], [(0, 66), (.5, 100), (1, 130)], L, w, 21)
    g = HD.render([far, near], cam, light=(-.75, -.25, .55), clip=(500, 250, 1500, 1080)); HD.ink(hc, g, pal, w[0] * .8, seed, inner=.6)
    b = W_(near, -.28); e = W_(near, -1.4); mn.sleeve2(hc, [b, (b + e) / 2 + [6, 10], (1400, 1030), (1520, 1220)], [(0, 64), (.4, 96), (1, 140)], L, w, 33)
    hl = mn.grade(hc.layer(), L, (600, 400, 1500, 1100), .62)
    im = comp.over(bg, body)
    sh = cv2.GaussianBlur(shifted(hl[..., 3], 16, 26), (0, 0), 2.0) * body[..., 3]; im = im * (1 - sh[..., None] * (1 - rgb('#8478b4')))
    im = comp.over(im, hl)
    # near tongues in front, under-light, floating ink
    im = comp.over(im, boil_layer([licks(-90, 860, 960, 300, 8, 14, lean=.28), licks(1320, 2020, 990, 300, 6, 23, lean=-.18), licks(500, 1700, 1105, 110, 11, 31, lean=.05)], P['base'], '#2f4aa8', P['line'], 3.0, (18, 16), hi2='#5d8fe6', rimc='#7cc6ff', drops=16, seed=14, box=(60, 900, 420, 900)))
    im = comp.screen(im, comp.radial(1010, 1120, 1000, 420, 2.2)[..., None] * rgb('#2c62e0') * .42)
    im = comp.over(im, comp.bokeh(boil_layer([licks(-200, 420, 1090, 300, 3, 41, lean=.35), licks(1560, 2150, 1120, 300, 3, 47, lean=-.35)], '#05060f', '#0f1c4a', '#030409', 2, (26, 20), drops=5, seed=44, box=(1500, 1900, 500, 900)), 9, 0))
    return im


def blueflame(layers, seed=0, w=2.4, line='#071040', f=0, boxes=(), cols=('#1a3fc8', '#3f8dff', '#a6e2ff'), core=1.0, rise=.06, hook=.36, boil=0., grow=1.):
    """cel cursed-energy fire: hooked licks (one contour) with two inner tones that follow the same shape lower down, like a hand-separated flame.
    layers = [(x0, x1, y, h, n), ...] far -> near"""
    out = np.zeros((H, W, 4), f32); r = np.random.default_rng(seed)
    for i, (x0, x1, y, h, n) in enumerate(layers):
        c = Cel(); P = [licks(x0, x1, y, h, n, seed * 7 + i, lean=.06, hook=hook, f=f, rise=rise, boil=boil, grow=grow)]
        if i == len(layers) - 1:
            for (bx, by, sz) in boxes:
                a = r.uniform(-.5, .5) - math.pi / 2; e = np.array([math.cos(a), math.sin(a)]); q = np.array([-e[1], e[0]]) * r.choice([-1, 1]); p0 = np.array([bx, by], float)
                P.append(spl([p0 - e * sz * .7, p0 - q * sz * .46, p0 + e * sz * .7 - q * sz * .18, p0 + e * sz * 2.1 + q * sz * .5, p0 + e * sz * .55 + q * sz * .40], True, corners=(3,), n=6))
        for j, q in enumerate(P): outline(c, q, w * 1.5, line, seed + i + j, tremor=.2)
        for q in P: c.fill(q, cols[0])
        lay = c.layer(); a = lay[..., 3]; k = core * h / 420.
        wx, wy = nz((90, 40), (1, .5), seed + 60 + i) * 9 * k, nz((90, 40), (1, .5), seed + 61 + i) * 14 * k
        gx, gy = grid(); wp = lambda m: cv2.remap(m, gx + wx, gy + wy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        for (d, e_), col in zip(((58, 17), (128, 36)), cols[1:]):
            m = np.minimum(np.minimum(dog_shift(a, 0, d * k), dog_shift(a, e_ * k, d * .62 * k)), np.minimum(dog_shift(a, -e_ * k, d * .62 * k), dog_shift(a, e_ * .5 * k, d * .85 * k)))
            m = smooth(.42, .58, cv2.GaussianBlur(wp(m), (0, 0), 1.1)) * a; lay[..., :3] = lay[..., :3] * (1 - m[..., None]) + rgb(col) * m[..., None]
        out = lay + out * (1 - lay[..., 3:4])
    return out


def steel_hammer(c, p0, p1, s, w, flip=1, pal=('#dfe6f4', '#99a3c4', '#525a7c', '#23283e'), shaft=True, shw=.085, shc=('#3a2f3c', '#1e1822', '#6a5a6c')):
    """Nobara's hammer along p0 (butt) -> p1 (head end): dark shaft, plain two-faced iron head.  s = head half length"""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float); d = p1 - p0; L_ = np.hypot(*d); d /= L_; n = np.array([-d[1], d[0]]) * flip; P = lambda u, v: p0 + d * (L_ + u * s) + n * v * s
    if shaft:
        t = shw * s; S_ = np.vstack([spl([p0 + n * t * 1.25, p0 + d * L_ * .12 + n * t * 1.15, p0 + d * L_ * .5 + n * t * .9, p1 + n * t], n=8), spl([p1 - n * t, p0 + d * L_ * .5 - n * t * .9, p0 + d * L_ * .12 - n * t * 1.15, p0 - n * t * 1.25], n=8)])
        outline(c, S_, w * 1.7, '#0b0910', 3); c.fill(S_, shc[0]); c.fill(np.array([p0 - n * t * 2 - d * 4, p0 - n * t * .1, p1 - n * t * .1, p1 - n * t * 2]), shc[1]) if False else None
        c.clip(S_); c.fill(np.vstack([[p0 - n * t * 2], [p0 - n * t * .15], [p1 - n * t * .15], [p1 - n * t * 2]]), shc[1]); c.line(np.array([p0 + d * L_ * .1 + n * t * .62, p0 + d * L_ * .5 + n * t * .52, p0 + d * L_ * .9 + n * t * .6]), t * .28, shc[2], taper=(.3, .4), seed=4); c.unclip()
    B = np.array([P(.30, -.98), P(.33, -.62), P(.24, -.50), P(.27, -.26), P(.27, .26), P(.24, .50), P(.33, .62), P(.30, .98), P(-.30, .98), P(-.33, .62), P(-.24, .50), P(-.27, .26), P(-.27, -.26), P(-.24, -.50), P(-.33, -.62), P(-.30, -.98)])
    outline(c, B, w * 1.8, '#0b0d18', 5); c.fill(B, pal[1]); c.clip(B)
    c.fill(np.array([P(.5, -1.1), P(.10, -1.1), P(.12, -.62), P(.07, -.5), P(.10, .5), P(.14, .62), P(.10, 1.1), P(.5, 1.1)]), pal[0]); c.fill(np.array([P(-.5, -1.1), P(-.14, -1.1), P(-.17, -.62), P(-.11, -.5), P(-.13, .5), P(-.17, .62), P(-.14, 1.1), P(-.5, 1.1)]), pal[2])
    for v in (-1, 1): c.fill(np.array([P(-.4, v * .90), P(.4, v * .90), P(.4, v * 1.0), P(-.4, v * 1.0)]), pal[3] if v > 0 else pal[0])
    c.unclip()
    for v in (-.62, -.50, .50, .62): c.line(np.array([P(-.32, v), P(0, v * 1.01), P(.32, v)]), w * .8, '#0b0d18', taper=(.05, .05), seed=int(abs(v) * 90))
    c.line(np.array([P(.21, -.88), P(.22, -.78), P(.21, -.68)]), w * 1.1, '#ffffff', taper=(.2, .2), seed=3); c.line(np.array([P(.19, -.3), P(.20, 0), P(.19, .34)]), w * 1.1, '#ffffff', taper=(.3, .4), seed=6)
    e = np.array([P(.27, -.10), P(.40, -.08), P(.40, .08), P(.27, .10)]); outline(c, e, w, '#0b0d18', 7); c.fill(e, shc[0])


def key2(f=0, seed=3):
    """Nobara: grin, hammer on the shoulder, blue fire behind"""
    from jj import hands as HD
    from chars.nobara import Nobara as N, BODY, EXPR
    x, y = grid(); L = mn.LIGHT['blue']
    n = fbm(H, W, (420, 160, 60), seed=21); d = np.sqrt(((x - 1080) / 1150) ** 2 + ((y - 760) / 760) ** 2)
    bg = ramp(np.clip(d + n * .05, 0, 1.4), [(0, '#3f6fe0'), (.3, '#2340a8'), (.6, '#141f66'), (.95, '#0a0c30'), (1.4, '#06061a')])
    bg = comp.over(bg, comp.blur(blueflame([(-100, 2020, 860, 520, 13), (-100, 2020, 960, 400, 15)], 4, 2.2, '#080f44', cols=('#121f74', '#1b37ac', '#2a55d8')), 2.4))
    bx = [(260, 250, 30), (420, 120, 18), (1500, 170, 34), (1690, 330, 24), (1800, 120, 16), (120, 470, 22), (1400, 70, 15), (640, 190, 16), (560, 330, 12), (1760, 480, 14), (80, 210, 13), (1330, 260, 12)]
    fl = blueflame([(-100, 2020, 900, 470, 12), (-100, 2020, 985, 390, 15), (-100, 2020, 1070, 250, 19)], 9, 2.6, '#0a1766', boxes=bx)
    g = comp.blur(fl, 34); bg = comp.screen(bg, g[..., :3] * .5); bg = comp.over(bg, fl)
    s = 288; cx, cy = 1020, 428; cel = Cel(); Hd = N.pose(cx, cy, s, .27, .06, -.15); w = mn.lw(s); hair = N._hair(Hd, f, (-.12, -.14, .3)); T = mn.torso_under(Hd, yaw=.14, roll=.10, drop=1.10)
    N.back(cel, Hd, 'blue', f, (0, 0), seed, hair); mn.uniform(cel, T, BODY, L, w, seed, 'back'); mn.neck(cel, Hd, L, w, .285, seed=seed); mn.uniform(cel, T, BODY, L, w, seed, 'front')
    N.head(cel, Hd, dict(EXPR['smug'], gaze=(-.006, .0)), 'blue', f, (0, 0), seed, parts='f', L=hair)
    body = mn.grade(cel.layer(), L, (cx - 420, 100, cx + 420, 1080), 1.0)
    # hammer and the fist that holds it
    hc = Cel(); pal = hpal(L); pal.update(prop='#2d2634', prop_shade='#17131d')
    cam = HD.Cam(800, 985, 168, 7.5); hr = HD.Hand('hammer', 'R', HD.euler(16, -8, -38), (0, 0, 0), HD.FEMALE)
    ax = np.array((.0, .86, -.345)); dr = HD.unit(np.array((1., .10, .03))); q0 = cam.px(hr.world(ax - dr * .62)); q1 = cam.px(hr.world(ax + dr * 1.95))
    steel_hammer(hc, q0, q1, 112, w[0] * .9, flip=-1, shw=.16)
    wr = cam.px(hr.world(np.array([0, -.15, 0]))); el = cam.px(hr.world(np.array([0, -1.6, 0])))
    g = HD.render([hr], cam, light=(.75, .3, .55), clip=(200, 380, 1100, 1080)); HD.ink(hc, g, pal, w[0] * .8, seed, inner=.6)
    mn.sleeve2(hc, [wr, wr + [-40, 60], wr + [-90, 160], wr + [-150, 320]], [(0, 66), (.5, 92), (1, 118)], L, w, 23)
    # nails at the lens
    c2 = Cel(); cam2 = HD.Cam(1600, 850, 225, 4.4); hn = HD.Hand('nails', 'L', HD.euler(164, 40, 8), (0, 0, 0), HD.FEMALE)
    wr2 = cam2.px(hn.world(np.array([0, -.30, 0]))); np_ = HD.nail_props(hn, 3, 1.35, .036); ends = [cam2.px(q['F'].o + q['F'].R[:, 1] * q['L']) for q in np_[1::3]]; mids = [cam2.px(q['F'].o + q['F'].R[:, 1] * q['L'] * .25) for q in np_[0::3]]
    for i, (e, m) in enumerate(zip(ends, mids)):             # cursed energy licking up each nail
        d = (e - m) / np.hypot(*(e - m)); q = np.array([-d[1], d[0]])
        for k, col in ((1., '#1a3fc8'), (.68, '#3f8dff'), (.40, '#c9f2ff')):
            z = 46 * k; P = spl([m - d * 20 * k, m + q * z * .55 + d * 30, m + d * 120 + q * z * (.85 if i % 2 else .6), e + d * 18 * k + q * z * .25, e + d * 128 * k + q * z * (.9 if i % 2 else -.7), e + d * 6 - q * z * .45, m + d * 110 - q * z * (.6 if i % 2 else .9), m - q * z * .55 + d * 34], True, corners=(4,), n=7); outline(c2, P, 2.6, '#0a1766', i); c2.fill(P, col)
    g2 = HD.render([hn], cam2, light=(-.5, .6, .6), props=np_); HD.ink(c2, g2, dict(pal, prop='#565f82', prop_shade='#1d2138'), w[0] * .9, seed, inner=.6)
    mn.sleeve2(c2, [wr2, wr2 + [40, 130], wr2 + [130, 420]], [(0, 92), (.5, 128), (1, 170)], L, w, 29)
    hl = mn.grade(hc.layer(), L, (300, 400, 900, 1100), .7); nl = mn.grade(c2.layer(), L, (1300, 500, 1800, 1100), .8)
    im = comp.over(bg, body)
    sh = cv2.GaussianBlur(shifted(hl[..., 3], -14, 24), (0, 0), 2.0) * body[..., 3]; im = im * (1 - sh[..., None] * (1 - rgb('#8478b4'))); im = comp.over(im, hl)
    im = comp.over(im, nl)
    for e in ends: im = comp.glow(im, e[0], e[1], 150, '#4fa8ff', .5, 2.2)
    im = comp.over(im, comp.bokeh(blueflame([(-200, 560, 1120, 230, 4), (1560, 2100, 1150, 230, 4)], 31, 2, '#0a1766'), 8, 0))
    c = Cel(); r = np.random.default_rng(12)
    for _ in range(70):
        p = np.array([r.uniform(0, W), r.uniform(0, H)]); sz = 1.5 + 6 * r.random() ** 2.5; a = r.uniform(-.6, .6) - math.pi / 2; e = np.array([math.cos(a), math.sin(a)]); q = np.array([-e[1], e[0]])
        c.fill(spl([p - e * sz, p + q * sz * .5, p + e * sz * 2.2, p - q * sz * .5], True, corners=(2,), n=5), '#c9f1ff' if r.random() < .6 else '#4fa8ff')
    e = c.layer(); im = im + e[..., :3] * 1.1
    return im


def key3(f=0, seed=3):
    """the dog's jaws at the lens"""
    P = dog.PAL['battle']; x, y = grid()
    cx, cy = 1010, 420
    d = np.sqrt(((x - cx) / 1250) ** 2 + ((y - cy) / 820) ** 2); n = fbm(H, W, (300, 120, 50), seed=5)
    bg = ramp(np.clip(d + n * .05, 0, 1.3), [(0, '#4f86e8'), (.25, '#2a4fc0'), (.55, '#15246e'), (.9, '#090d2a'), (1.3, '#05060f')])
    fl = Cel(); focus_lines(fl, cx, cy, 620, 1700, 120, 9, '#8fc2ff', seed, ry=.62); focus_lines(fl, cx, cy, 520, 1700, 150, 12, '#05071a', seed + 1, ry=.62)
    bg = comp.over(bg, fl.layer(), .75)
    c = Cel(); far = dict(P, base='#0a0d22', sh='#060816', hi='#182766', hi2='#2d53b0')
    dog.fur_fill(c, dog.mane(1000, 380, 760, 520, 34, 600, 230, 4, swirl=.30), far, 3.2, 1, wedge=.45)
    lay = dog.rim2(c.layer(), 5, 6, '#5aa8ff', .8, .8); bg = comp.over(bg, lay)
    bg = comp.screen(bg, comp.radial(1000, 120, 1100, 520, 2.)[..., None] * rgb('#1b3f9f') * .5)
    c = Cel()
    dog.fur_fill(c, dog.mane(1000, 420, 620, 430, 30, 470, 190, 9, swirl=-.22), P, 3.4, 2, wedge=.6)
    D = dog.DogHead().pose(1010, 382, 345, -.16, -.05, .14, jaw=.90, persp=.445); dog.head(c, D, 'battle', f, seed, w=3.8)
    lay = c.layer()
    lay = para(lay, (500, 0), (1300, 1080), [(0, '#f4f8ff'), (.45, '#ffffff'), (1, '#8a86c8')], 'mul', .8)
    lay = dog.rim2(lay, 6, 5, '#79c4ff', .9, .7)
    im = comp.over(bg, lay)
    for q in (-1, 1):
        e = D.P3(np.array([(q * .44, .40, .44)]))[0]; im = comp.glow(im, e[0], e[1], 250, '#3f8cff', .5, 2.6)
    # saliva, then blurred ink and sparks thrown at the lens
    c = Cel(); r = np.random.default_rng(7)
    up = [D.P3(np.array([p]))[0] for p in ((-.205, -.62, 1.30), (.205, -.60, 1.30), (.115, -.36, 1.40))]; lo = D.P3(D.low(np.array([(-.245, -.10, 1.25), (.245, -.06, 1.25), (.13, -.34, 1.30)])))
    for (a, b, k) in ((up[1], lo[1], 2.2),):
        m = (a + b) / 2 + [r.uniform(-30, 30), 30]; c.line(spl([a, m, b], n=12), k * 2, '#cfe0ff', taper=(.05, .05), wfun=lambda t: .35 + 1.3 * (2 * t - 1) ** 2, alpha=.8, seed=3)
    im = comp.over(im, c.layer())
    c = Cel(); c2 = Cel()
    for _ in range(46):
        a = r.uniform(0, 6.283); d0 = r.uniform(430, 1050); p = np.array([1000 + math.cos(a) * d0, 520 + math.sin(a) * d0 * .62]); e = np.array([math.cos(a), math.sin(a) * .62]); e /= np.hypot(*e); L_ = r.uniform(16, 90); wd = r.uniform(2, 9)
        q = np.array([p, p + e * L_ * .5 + np.array([-e[1], e[0]]) * wd, p + e * L_, p + e * L_ * .5 - np.array([-e[1], e[0]]) * wd]); (c if r.random() < .72 else c2).fill(spl(q, True, corners=(0, 2), n=5), '#04050c' if r.random() < .75 else '#101a44')
    im = comp.over(im, comp.bokeh(c.layer(), 3.2, 0)); lay2 = np.zeros((H, W, 4), f32)
    s_ = Cel()
    for _ in range(40):
        a = r.uniform(0, 6.283); d0 = r.uniform(380, 1100); p = np.array([1000 + math.cos(a) * d0, 500 + math.sin(a) * d0 * .6]); e = np.array([math.cos(a), math.sin(a) * .62]); L_ = r.uniform(10, 46); s_.line(np.array([p, p + e * L_ * .5, p + e * L_]), r.uniform(1.2, 3.4), '#bfe4ff', taper=(.5, .2), seed=int(a * 99))
    im = im + s_.layer()[..., :3] * 1.1
    return im


def sheet():
    """model sheet: Megumi row, Nobara row, dog row"""
    from chars.megumi import Megumi as M; from chars.nobara import Nobara as N
    BG = {'day': '#b9d3e2', 'sunset': '#e08a5c', 'under': '#343a52', 'battle': '#1c2350', 'snow': '#ccd4e6', 'blue': '#2141a8'}
    A = [('deadpan', 'day', 0, 0, 0), ('sigh', 'sunset', .45, -.04, .05), ('focus', 'battle', -.35, -.08, 0), ('alarm', 'under', .3, .05, 0), ('rage', 'blue', -.45, 0, .05), ('smile', 'snow', .2, .02, -.06), ('dots', 'day', 0, 0, 0), ('deadpan', 'sunset', 1.45, 0, 0)]
    B = [('point', 'day', -.3, .04, -.06), ('smug', 'sunset', .35, 0, -.06), ('fierce', 'battle', -.4, -.05, 0), ('alarm', 'under', .3, 0, 0), ('teary', 'snow', -.2, .03, .08), ('calm', 'day', 0, .28, 0), ('dots', 'day', 0, 0, 0), ('calm', 'sunset', -1.4, 0, 0)]
    rows = []
    for CH, Q in ((M, A), (N, B)):
        cells = []
        for k, (e, m, yw, pt, rl) in enumerate(Q):
            im = np.zeros((600, 480, 3), f32) + rgb(BG[m]); kw = dict(patch=True) if (CH is N and e == 'teary') else {}
            l = CH.bust(240, 262 if CH is M else 250, 128, yw, pt, rl, e, m, 0, (0, 0), 3, w=480, h=600, **kw); cells.append(comp.over(im, l))
        rows.append(np.hstack(cells))
    d = []
    for k in ('erupt', 'lunge', 'leap'):
        im = np.zeros((1080, 1920, 3), f32) + rgb('#2b41a0'); im = comp.over(im, dog.draw(dog.SPEC[k])); d.append(cv2.resize(im, (1280, 720), interpolation=cv2.INTER_AREA))
    rows.append(np.hstack(d)); S = np.vstack(rows); comp.save('qa/meguno/sheet.jpg', S, 90)


if __name__ == '__main__':
    import sys
    k = sys.argv[1]
    if k == 'sheet': sheet()
    else: im = globals()['key' + k](); comp.save(f'qa/meguno/key{k}.jpg', comp.to8(comp.finish(im, 0, bloom=.34, vig=.2)), 92)
