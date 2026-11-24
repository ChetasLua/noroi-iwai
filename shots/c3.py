"""c3 HANDS: four 18-frame close-ups.  Megumi's dog sign / Nobara's nails / Yuji's wrapped fist / Gojo's crossed fingers."""
import numpy as np, cv2, math, os, pickle, hashlib
from jj import W, H, comp, kit, hands as HD, hands_sleeve as HS, fx
from jj.paint import rgb, mix, smooth, fbm, f32, grid, ramp, lum
from jj.ink import Cel, spl, noise1
from chars.mn_keys import steel_hammer
E = HD.euler; VER = 6
PW, PH = 2080, 1170                                   # plates are painted a little larger than the frame
GEO = None; HAMFLIP = 1


# ---------------------------------------------------------------- geometry (ray cast once, cached)
def _pose(p, **k): q = dict(HD.POSES[p] if isinstance(p, str) else p); q.update(k); return q


START = HD.pose((8, 14, 7, 1), (4, 8, 5, 1), (6, 10, 5, -2), (10, 14, 8, 2), thumb=(26, 8, 50, 2, 4))


def b_a(gap, t, th, sq=0., tag=0):
    near = HD.blend(START, 'dog_near', t, drag=.3, order=(0, 1, 2, 3)); far = HD.blend(START, 'dog_far', t, drag=.3, order=(1, 0, 2, 3))
    A, B = np.array(START['thumb']), np.array(HD.POSES['dog_near']['thumb']); near['thumb'] = tuple(A + (B - A) * th); B = np.array(HD.POSES['dog_far']['thumb']); far['thumb'] = tuple(A + (B - A) * th)
    near['wrist'] = (-12, 6); far['wrist'] = (-12, 6)
    hs = HD.two('dog', R=E(-27, 13, -7), gap=gap, bld=HD.SLIM, poses=(near, far))
    hs[0].o = hs[0].o + np.array([-.10 * sq, 0, 0]); hs[1].o = hs[1].o + np.array([.55 * gap, .10 * gap, 0])
    cam = HD.Cam(1075, 640, 415, 5.2); L = (-.62, .72, .02)
    g = HD.render(hs, cam, L, res=640, clip=(-60, -60, W + 60, H + 60), thr=.20)
    return HD.lite(g, dict(k=HD.rim_polys(g, (-.75, .62, -.2), .42, .55), b=HD.rim_polys(g, (.7, .2, -.6), .34, .42)), dict(d=HD.depth_polys(g, L, -.30)),
                   dict(sl=[HS.sleeve(g, 0, y0=.20, y1=2.4, seed=4, light=L, seam=-1.0, button=True), HS.sleeve(g, 1, y0=.17, y1=2.4, seed=9, light=L, button=False)]))


def b_b(slide, sp, cl, tag=0):
    q = _pose('nails'); q = {k: (list(v) if not np.isscalar(v) else v) for k, v in q.items()}
    for i, f in enumerate(HD.FING):
        q[f][0] += cl * 4 - 3; q[f][1] += cl * 5 - 4; q[f][3] += (sp[0] * (i == 0) * -5 + sp[0] * (i >= 1) * 2 + sp[1] * (i <= 1) * -4 + sp[1] * (i >= 2) * 4 + sp[2] * (i <= 2) * -2 + sp[2] * (i == 3) * 7)
    q['wrist'] = (-10, 8); q['thumb'] = (16, 46, -38, 26, 50)
    h = HD.Hand(q, 'R', E(192, 43, 15), bld=HD.FEMALE, pivot=(0, .8, 0)); cam = HD.Cam(760, 640, 440, 4.6); L = (.80, .40, .42)
    g = HD.render([h], cam, L, res=640, clip=(-60, -60, W + 60, H + 60), thr=.12, props=HD.nail_props(h, slide=slide, L=1.02, r=.031))
    tips = []
    for k, (f1, f2) in enumerate(zip(HD.FING[:3], HD.FING[1:])):
        A, B = h.sk.fr[f1][0], h.sk.fr[f2][0]; m = (A.pt((0, h.sk.fr[f1 + '_L'][0] * .66, 0)) + B.pt((0, h.sk.fr[f2 + '_L'][0] * .70, 0))) / 2; d = HD.unit(np.array(((.16, .02, -.14)[k], 1., .16))); tips.append(cam.px(h.world(m + d * 1.10)))
    return HD.lite(g, dict(t=HD.big(HD.rim_polys(g, (-.85, .15, -.35), .34, .42), 440, .012), w=HD.big(HD.rim_polys(g, (.75, .6, -.3), .50, .46), 440, .006)), dict(d=HD.depth_polys(g, L, -.34)), dict(tips=tips, sl=[HS.sleeve(g, 0, y0=.16, y1=2.3, seed=7, light=L, seam=1.2)]))


def b_h(tag=0):
    h = HD.Hand(_pose('hammer', wrist=(6, 10)), 'L', E(-118, 6, 16), bld=HD.FEMALE, pivot=(0, .8, 0)); cam = HD.Cam(1500, 650, 300, 6); L = (.80, .40, .42)
    g = HD.render([h], cam, L, res=520, clip=(900, -60, W + 160, H + 160), thr=.12)
    ax, dr = np.array((.0, .86, -.345)), HD.unit(np.array((1., .10, .03))); pt = lambda u: cam.px(h.world(ax + u * dr)); ham = (pt(-.62), pt(1.95), .35 * 1.95 * cam.s * .88)      # the film's one hammer: butt, head, half length of the head
    return HD.lite(g, dict(t=HD.big(HD.rim_polys(g, (-.85, .15, -.35), .34, .42), 300, .02), w=HD.big(HD.rim_polys(g, (.75, .6, -.3), .50, .46), 300, .02)), {}, dict(ham=ham, sl=[HS.sleeve(g, 0, y0=.18, y1=2.3, seed=12, light=L, button=False)]))


def b_c(seed, sq, tag=0):
    q = _pose('fist')
    if seed: q = HD.tremble(q, seed, 2.4)
    q = {k: (list(v) if not np.isscalar(v) else v) for k, v in q.items()}
    for f in HD.FING: q[f][0] += -10 * (1 - sq) + 2 * sq; q[f][1] += -9 * (1 - sq); q[f][2] += -4 * (1 - sq)
    q['thumb'] = list(q['thumb']); q['thumb'][4] += 10 * sq - 5; q['wrist'] = (-14 + 8 * sq, -6)
    h = HD.Hand(q, 'R', E(142, 44, 22), bld=HD.HEAVY, pivot=(0, .95, 0), tape=dict(fingers=HD.FING, fspan=(-.2, .74))); cam = HD.Cam(850, 520, 640, 3.5); L = (-.45, .86, .22)
    g = HD.render([h], cam, L, res=700, clip=(-60, -60, W + 60, H + 60), thr=.16)
    return HD.lite(g, dict(e=HD.rim_polys(g, (.55, -.75, -.1), .22, .70), f=HD.rim_polys(g, (-.85, -.2, -.35), .30, .55), r=HD.rim_polys(g, (.95, .05, -.55), .42, .36), t=HD.rim_polys(g, (-.3, .9, -.3), .50, .5)), dict(d=HD.depth_polys(g, L, -.32)), dict(sl=[HS.sleeve(g, 0, y0=.30, y1=2.3, seed=3, light=L, seam=-.9, button=True)]))


def b_d(t, over=0., br=0., tag=0):
    q = HD.blend('two', 'cross', t, drag=.15, order=(0, 1, 2, 3)) if t >= 0 else HD.lerp_pose('two', HD.pose((-6, 14, 8, -4), (14, 16, 10, 4), (80, 104, 58, 0), (86, 104, 58, 0), thumb=(-5, 48, -75, 20, 14), cup=.1), -t)
    q = {k: (list(v) if not np.isscalar(v) else v) for k, v in q.items()}
    q['middle'][3] -= 5 * over; q['index'][3] += 4 * over; q['middle'][1] += 8 * over; q['index'][0] += 4 * over; q['ring'][0] += 3 * br; q['little'][0] += 3 * br
    q['wrist'] = (-6 - 4 * over, 8)
    h = HD.Hand(q, 'R', E(157, -9, -7), bld=HD.SLIM, pivot=(0, .85, 0)); cam = HD.Cam(830, 470, 316, 5.6); L = (.85, .40, -.05)
    g = HD.render([h], cam, L, res=700, clip=(-60, -60, W + 60, H + 60), thr=.12)
    return HD.lite(g, dict(w=HD.rim_polys(g, (.9, .45, -.35), .24, .60), c=HD.rim_polys(g, (-.95, .1, -.35), .34, .46)), dict(d=HD.depth_polys(g, L, -.42)),
                   dict(sl=[HS.sleeve(g, 0, y0=.17, y1=2.4, seed=6, light=(.42, .25, .86), seam=-.9, flare=.07, ease=1.22, gap=.065, folds=[(.46, 1.3, .14, .050, .060), (.66, -1.4, .10, .045, .05), (.98, 1.0, .24, .055, .08), (1.35, -1.1, .3, .05, .1)])]))


KEYS = {}
for i, (fr, a) in enumerate(((0, (.34, 0, 0)), (2, (.40, 0, 0)), (4, (.14, .10, 0)), (5, (.0, .18, .0, 1.)), (6, (.0, .42, .10, .4)), (8, (.0, .85, .25)), (9, (.0, 1., 1.16)), (10, (.0, 1., 1.06)), (12, (.0, 1., 1.0)))): KEYS['a', fr] = (b_a, a)
_s = lambda f, t0: 1. if f < t0 else {0: .62, 1: .16, 2: -.05, 3: -.02}.get(f - t0, 0.)
for fr in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14):
    KEYS['b', fr] = (b_b, (tuple(_s(fr, t0) for t0 in (1, 5, 9)), tuple(max(0., 1 - abs(fr - t0) / 2.2) for t0 in (0, 4, 8)), smooth(9, 12, float(fr)) * 1.))
KEYS['h', 0] = (b_h, ())
for k in range(4): KEYS['c', k] = (b_c, (31 + k, .15 + .05 * k))
KEYS['c', 8] = (b_c, (0, .0)); KEYS['c', 9] = (b_c, (0, 1.15)); KEYS['c', 10] = (b_c, (0, 1.0))
for fr, a in ((0, (0., 0, 0)), (3, (0., 0, .5)), (6, (-.5, 0)), (8, (-.9, 0)), (9, (.75, .4)), (10, (1., 1.)), (11, (1., .45)), (13, (1., 0.)), (16, (1., 0., .6))): KEYS['d', fr] = (b_d, a)


def _job(k):
    cv2.setNumThreads(1); fn, a = KEYS[k]; return k, fn(*a)


def geometry():
    global GEO
    if GEO is not None: return GEO
    key = hashlib.md5((open(HD.__file__).read() + open(HS.__file__).read() + open(__file__).read().split('# ---------------------------------------------------------------- plates')[0]).encode()).hexdigest()[:10]
    path = f'cache/c3_geo_{key}.pkl'
    if os.path.exists(path): GEO = pickle.load(open(path, 'rb')); return GEO
    import multiprocessing as mp
    try:
        with mp.get_context('fork').Pool(2) as P: GEO = dict(P.map(_job, list(KEYS), chunksize=1))
    except AssertionError: GEO = dict(_job(k) for k in KEYS)
    import glob
    for o in glob.glob('cache/c3_geo_*.pkl'): os.remove(o)
    pickle.dump(GEO, open(path, 'wb')); return GEO


# ---------------------------------------------------------------- plates
def P3(x, y, z, f=1150, vx=PW * .5, vy=PH * .5, cy=1.5): return np.array([vx + f * x / z, vy - f * (y - cy) / z])


def platform(pal, vx, vy, f=1150, side=-1, seed=1, tubes=True, cols=((2.0, (5, 8.5, 12, 15.5, 19)),), cy=1.2):
    """an underground platform in perspective, painted flat in value masses (it lives out of focus)"""
    r = np.random.default_rng(seed); c = Cel(PW, PH, ss=1); p = lambda x, y, z: P3(x, y, z, f, vx, vy, cy); q = lambda pts, col, a=1.: c.fill(np.array([p(*v) for v in pts]), col, a)
    c.fill(np.array([(0, 0), (PW, 0), (PW, PH), (0, PH)]), pal['far']); N, F = 1.2, 46
    q([(-5, 3.1, N), (5, 3.1, N), (5, 3.1, F), (-5, 3.1, F)], pal['ceil']); q([(-5, 0, N), (5, 0, N), (5, 0, F), (-5, 0, F)], pal['floor'])
    for s in (-1, 1):
        q([(5 * s, 0, N), (5 * s, 3.1, N), (5 * s, 3.1, F), (5 * s, 0, F)], pal['wall'])
        q([(5 * s, 1.0, N), (5 * s, 1.18, N), (5 * s, 1.18, F), (5 * s, 1.0, F)], pal['band'])
        for z in np.arange(2, 30, 1.1): q([(5 * s, 0, z), (5 * s, 3.1, z), (5 * s, 3.1, z + .05), (5 * s, 0, z + .05)], pal['grout'], .8)
        for y in np.arange(.25, 3, .38): q([(5 * s, y, N), (5 * s, y + .03, N), (5 * s, y + .03, F), (5 * s, y, F)], pal['grout'], .6)
        for z in (4.5, 11): q([(4.95 * s, 1.5, z), (4.95 * s, 2.5, z), (4.95 * s, 2.5, z + 3.4), (4.95 * s, 1.5, z + 3.4)], pal['poster'], .9)
    for z in np.arange(1.4, 30, 1.2): q([(-5, .001, z), (5, .001, z), (5, .001, z + .04), (-5, .001, z + .04)], pal['grout'], .6)
    for x in np.arange(-4.8, 5, 1.2): q([(x, .001, N), (x + .04, .001, N), (x + .04, .001, F), (x, .001, F)], pal['grout'], .5)
    q([(side * 3.4, .002, N), (side * 3.9, .002, N), (side * 3.9, .002, F), (side * 3.4, .002, F)], pal['tactile'], .85)
    if tubes:
        for x in (-2.6, 2.6):
            for z in np.arange(2.2, 30, 2.6):
                on = r.random() > pal.get('dead', .25)
                q([(x - .13, 3.05, z), (x + .13, 3.05, z), (x + .13, 3.05, z + 1.5), (x - .13, 3.05, z + 1.5)], pal['tube'] if on else pal['grout'])
                if on: q([(x - .6, .003, z), (x + .6, .003, z), (x + .5, .003, z + 1.5), (x - .5, .003, z + 1.5)], pal['tube'], .10)
    for cx_, zs in cols:
        for z in sorted(zs, reverse=True):
            for x in (-cx_, cx_):
                w = .34; q([(x - w, 0, z), (x + w, 0, z), (x + w, 3.1, z), (x - w, 3.1, z)], pal['col'])
                s = 1 if x < 0 else -1; q([(x + s * w, 0, z), (x + s * w, 3.1, z), (x + s * w, 3.1, z + .7), (x + s * w, 0, z + .7)], pal['col_side'])
                q([(x - w, 0, z), (x - w * .45, 0, z), (x - w * .45, 3.1, z), (x - w, 3.1, z)], pal['col_lit'] if pal.get('lit', -1) < 0 else pal['col_side'])
                q([(x + w * .45, 0, z), (x + w, 0, z), (x + w, 3.1, z), (x + w * .45, 3.1, z)], pal['col_lit'] if pal.get('lit', -1) > 0 else pal['col_side'])
                for y in np.arange(.3, 3, .3): q([(x - w, y, z), (x + w, y, z), (x + w, y + .025, z), (x - w, y + .025, z)], pal['grout'], .7)
    im = c.rgba()[..., :3]; im = im * np.clip(1 + .10 * fbm(PH, PW, (120, 40, 12), seed=seed), .75, 1.25)[..., None]
    return im


def dof(im, s):
    out = comp.blur(im, s); return out


def fg_blob(polys, col, blur):
    c = Cel(PW, PH, ss=1)
    for q in polys: c.fill(q, col)
    L = c.layer(PH, PW); return comp.blur(L, blur)


def plates():
    path = lambda k: f'cache/c3_plate_{VER}_{k}.npy'
    if all(os.path.exists(path(k)) for k in 'abcd'): return {k: np.load(path(k), mmap_mode='r') for k in 'abcd'}
    x, y = grid(PH, PW); out = {}
    # a: cold blue-green fluorescents
    pal = dict(far='#0a2428', ceil='#08181c', floor='#0e2b30', wall='#164a4e', band='#2d7f80', grout='#0b2a2e', poster='#1d5c62', tactile='#6a7a3a', tube='#d9fff0', col='#123c42', col_side='#081c21', col_lit='#2f7d7c', dead=.35)
    a = platform(pal, 700, 500, 1250, side=-1, seed=3); a = a + comp.blur(np.clip(a - .55, 0, 1), 40) * 1.6
    out['a'] = dof(a, 9) * np.clip(1.12 - .55 * np.hypot((x - 620) / 1500, (y - 360) / 900), .25, 1.2)[..., None]
    # b: warm lamp against teal
    pal = dict(far='#12343c', ceil='#0d1c24', floor='#1d2a30', wall='#17424a', band='#37767a', grout='#0d242b', poster='#1e5560', tactile='#8a6a2e', tube='#8fe6df', col='#1d3f47', col_side='#0a171d', col_lit='#c9743a', lit=1, dead=.5)
    b = platform(pal, 1250, 560, 1150, side=1, seed=8, cols=((2.0, (4.5, 8, 12, 16)),)); d = np.hypot((x - 1800) / 1.1, y - 330)
    b = b * (1 - .55 * smooth(1300, 200, d)[..., None]) + (smooth(1250, 60, d) ** 1.6)[..., None] * rgb('#ff8f3a') * .80
    lc = Cel(PW, PH, ss=1); lc.fill(np.array([(1690, 90), (2000, 130), (1990, 250), (1680, 200)]), '#fff3d2'); lc.fill(np.array([(1700, 268), (1985, 312), (1975, 420), (1692, 366)]), '#ffe2ac'); lc.fill(np.array([(1650, 60), (1684, 64), (1668, 600), (1640, 600)]), '#2a1a14'); lc.fill(np.array([(1300, 780), (1470, 770), (1500, 1010), (1320, 1030)]), '#7a3f22', .6)
    ll = comp.blur(lc.layer(PH, PW), 7); b = comp.over(b, ll) + comp.blur(ll[..., :3], 60) * .5
    out['b'] = dof(b, 10)
    # c: almost dark tunnel, one red signal
    pal = dict(far='#05070c', ceil='#080a12', floor='#171c2a', wall='#1a2234', band='#324058', grout='#0a0e18', poster='#202b40', tactile='#4a4424', tube='#8ea2cc', col='#1e2740', col_side='#0a0e18', col_lit='#46587e', dead=.55)
    c = platform(pal, 1330, 430, 1200, side=-1, seed=5); d = np.hypot(x - 1405, y - 470); c = c + (smooth(30, 6, d))[..., None] * rgb('#ff3a2a') * 2.2 + (smooth(260, 10, d) ** 2)[..., None] * rgb('#a01410') * .6
    out['c'] = dof(c, 9)
    # d: the stair mouth, hard blue-white light, columns as black slabs
    dx, dy = x - 1120, y - 360; d = np.hypot(dx / 1.1, dy); th = np.arctan2(dy, dx); r = np.random.default_rng(4); ray = np.zeros_like(x)
    for k in range(20):
        a_ = r.uniform(-math.pi, math.pi); wd = r.uniform(.010, .06); ray += r.uniform(.2, 1) * np.exp(-((((th - a_ + math.pi) % (2 * math.pi)) - math.pi) / wd) ** 2)
    im = ramp(np.clip(d / 1250, 0, 1), [(0, '#f2fbff'), (.07, '#cfeaff'), (.17, '#74b2f0'), (.34, '#3a53b4'), (.62, '#1f1f5e'), (1, '#0a0a22')])
    cc = Cel(PW, PH, ss=1)
    for k in range(6):                                        # steps catching the light
        yy = 690 + 34 * k + 11 * k * k; hw = 330 + 62 * k; cc.fill(np.array([(1120 - hw, yy), (1120 + hw, yy), (1120 + hw + 34, yy + 12 + 4 * k), (1120 - hw - 34, yy + 12 + 4 * k)]), '#0b0c26'); 
        cc.fill(np.array([(1120 - hw, yy - 4 - k), (1120 + hw, yy - 4 - k), (1120 + hw, yy), (1120 - hw, yy)]), '#86b4f0', .38)
    cc.fill(np.array([(0, 0), (560, 0), (600, 1170), (0, 1170)]), '#080818'); cc.fill(np.array([(560, 0), (640, 0), (690, 1170), (600, 1170)]), '#20235c'); cc.fill(np.array([(1690, 0), (2080, 0), (2080, 1170), (1640, 1170)]), '#080818'); cc.fill(np.array([(1650, 0), (1690, 0), (1640, 1170), (1580, 1170)]), '#3140a0')
    cc.fill(np.array([(0, 0), (2080, 0), (2080, 70), (1500, 110), (700, 110), (0, 70)]), '#07071a')
    L = comp.blur(cc.layer(PH, PW), 5); im = comp.over(im, L); im = im + (np.clip(ray, 0, 1.4) * smooth(1250, 120, d) * .30)[..., None] * rgb('#a9d0ff'); out['d'] = dof(im, 5)
    for k, v in out.items(): np.save(path(k), v.astype(np.float16))
    out.clear(); return {k: np.load(path(k), mmap_mode='r') for k in 'abcd'}


# ---------------------------------------------------------------- drawing
def pals():
    A = dict(HD.PAL['day'], skin='#aecdc2', shade='#3f6870', deep='#213f4b', line='#0a1418', nail='#cfe6da', nail_shade='#5a7f86', rim='#e9fff6', rim2='#2a8f97',
             cloth='#1b2634', cloth_sh='#0f161f', cloth_lt='#46569a', cuff_in='#05070a', cast='#274550', button='#6b8f7a', button_hi='#d9fff0')
    B = dict(HD.PAL['day'], skin='#f6b384', shade='#70505c', deep='#3a4a58', line='#1a0f12', nail='#ffd3b0', nail_shade='#86606a', rim='#63d6cc', rim2='#fff0cf', steel='#7c7a86', steel_shade='#2c3540', prop='#9a6a40', prop_shade='#4a342c',
             cloth='#191420', cloth_sh='#0b0a12', cloth_lt='#6a4480', cuff_in='#080609', cast='#57404e', button='#c98f3c', button_hi='#fff0c0')
    C = dict(HD.PAL['day'], skin='#b4b4cc', shade='#494b70', deep='#26274a', line='#0b0b14', tape='#e2e0ec', tape_shade='#5a5980', tape_deep='#34335a', nail='#aeb0c4', nail_shade='#5a5c7c', rim='#8fe2ff', rim2='#3f7fff',
             cloth='#11111b', cloth_sh='#08080e', cloth_lt='#463f80', cuff_in='#050509', cast='#34344f', button='#7d7348', button_hi='#d8cfa0')
    C2 = dict(C, nail='#a9d0ff', nail_shade='#4a70cc', skin='#8fb2e8', shade='#2c4aa0', deep='#182a72', tape='#c8deff', tape_shade='#4668c8', tape_deep='#24399a', cloth='#10152c', cloth_lt='#3f4cc0')
    D = dict(HD.PAL['day'], skin='#343879', shade='#1d1c4a', deep='#121136', line='#080820', nail='#4a4f96', nail_shade='#28285e', rim='#f2faff', rim2='#63b8ff',
             cloth='#12112a', cloth_sh='#080716', cloth_lt='#5147ac', cuff_in='#04040c', cast='#15143a', button='#8f86c8', button_hi='#f4f2ff')
    return dict(a=A, b=B, c=C, c2=C2, d=D)


def mixpal(p, q, t): return {k: ('#%02x%02x%02x' % tuple(int(round(255 * v)) for v in (rgb(p[k]) * (1 - t) + rgb(q[k]) * t))) for k in p}


SKIN = HD.FING + ('thumb', 'palm')


def hand(cel, g, pal, w, seed=1, rims=(), deep=True, order=None, sleeves=True, srim=(), holes=False):
    """hands + sleeves, far to near"""
    sl = g.extra['sl']; ids = order or range(len(g.hands))
    for hi in ids:
        if sleeves: HS.draw_back(cel, sl[hi], pal)
    if holes:
        for q in g.mask: cel.fill(q, pal['deep'])
    for hi in ids:
        q = HD.only(g, hi); HD.fill_flat(cel, q, pal); HD.fill_shadow(cel, q, pal)
        if deep: HD.fill_polys(cel, q, g.deeps['d'], pal['deep'], names=SKIN)
        HD.fill_overlays(cel, q, pal, w)
        if 'tape_deep' in pal:
            for t in q.tapes:
                G = [G for G in q.groups if G['id'] == t['gid']]
                if G and g.deeps['d'].get(t['gid']): HD._clipg(cel, G[0]); cel.ctx.save(); cel._path(t['polys']); cel.ctx.set_fill_rule(1); cel.ctx.clip(); cel.fill(g.deeps['d'][t['gid']], pal['tape_deep'], evenodd=True); cel.ctx.restore(); cel.ctx.restore()
        pm = [p_ for G in q.groups if G['name'] == 'palm' for p_ in G['polys']]
        if sleeves and pm: HS.draw_cast(cel, sl[hi], pal, pm)
        for nm, col, al, *only_ in rims: HD.fill_polys(cel, q, g.rims[nm], col, al, names=only_[0] if only_ else SKIN + ('steel', 'prop'))
        HD.ink_lines(cel, q, pal, w, seed)
        if sleeves:
            HS.draw_front(cel, sl[hi], pal, w * .95, seed + hi)
            for rd, rc, rw in srim: srimline(cel, sl[hi], pal, w, seed + hi, rd, rc, rw)
    return cel


def srimline(cel, sl, pal, w, seed, rd, rc, rw):
    d = np.array([rd[0], -rd[1]], float); d /= np.hypot(*d)
    for k, q in enumerate(sl['sil']):
        T = np.gradient(q, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]
        if ((q - q.mean(0)) * N).sum(1).mean() < 0: N = -N
        a = N @ d
        for x, y in HD._runs(HD._close_gaps(a > .20, 3, True), True):
            if y - x < 10: continue
            idx = np.arange(x, y) % len(q); aa = np.clip((a[idx] - .12) * 1.5, .25, 1.2); t0 = np.linspace(0, 1, len(idx))
            cel.line(q[idx] - N[idx] * w * (.5 + rw * .5), w * rw, rc, taper=(.2, .2), press=.35, seed=seed * 3 + k + x, wfun=lambda t, A=t0, B=aa: np.interp(t, A, B), tmin=.1)


def at(d, pre, f):
    k = max(q for (p, q) in d if p == pre and q <= f); return d[pre, k]


def stand(im, zoom, cx=0., cy=0., rot=0.): return comp.cam2d(np.asarray(im, f32), PW / 2 + cx, PH / 2 + cy, zoom, rot, W, H)


def move(L, dx=0., dy=0., z=1., ax=W / 2, ay=H / 2, rot=0.):
    c, s = math.cos(rot) * z, math.sin(rot) * z
    return cv2.warpAffine(np.asarray(L, f32), np.float32([[c, -s, ax - c * ax + s * ay + dx], [s, c, ay - s * ax - c * ay + dy]]), (W, H), flags=cv2.INTER_LINEAR)


WOLF = 'M2100 1200 L2100 520 C1960 470 1850 470 1760 520 L1700 380 L1655 520 L1560 350 L1530 540 C1470 500 1420 470 1380 420 C1300 330 1180 250 1090 214 L1120 40 C1030 70 960 130 930 196 C900 190 880 188 852 192 L846 20 C770 70 724 150 712 236 C640 262 590 306 540 356 C440 384 340 402 250 420 C196 430 150 446 128 470 C120 492 130 512 150 520 C160 540 190 548 230 548 L262 584 L286 546 C340 548 400 552 452 566 L470 604 L494 572 C560 590 610 610 636 640 C560 640 470 652 400 676 L382 640 L362 690 C310 704 270 722 246 744 C300 772 380 792 470 800 C560 808 640 800 700 786 C720 850 700 930 640 1010 L600 1200 Z'


def wolf(f):
    """the shadow that gets up behind the hands: 0..17"""
    k = f // 2 * 2; up = comp.ease_out(np.clip((k - 5) / 9., 0, 1), 2.2); c = Cel()
    r = np.random.default_rng(40 + k)
    # pool swelling along the bottom
    xs = np.linspace(-60, W + 60, 70); top = H + 30 - 330 * comp.ease_out(np.clip((k + 1) / 12., 0, 1)) * (.35 + .65 * np.exp(-((xs - 620) / 620) ** 2)) + 34 * np.sin(xs * .011 + k * .5) + 16 * np.sin(xs * .031 - k * .8) + r.normal(0, 4, 70)
    c.fill(np.vstack([np.c_[xs, top], [(W + 60, H + 60), (-60, H + 60)]]), '#04080b')
    if up > .02:
        P = kit.s1(WOLF, 16); P = P - [1100, 1200]; P = P * [.82, .93 * up] + [905 - 40 * (1 - up), 1180]
        from jj.ink import resample
        Q = resample(P, 9.); T = np.gradient(Q, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-6; N = np.c_[T[:, 1], -T[:, 0]]
        n = noise1(len(Q), 5, r) * 5 + noise1(len(Q), 16, np.random.default_rng(7 + k // 4)) * 9; Q = Q + N * n[:, None]
        c.fill(Q, '#04080b')
        for j in range(26):                                     # smoke-hair torn off the back and ears
            i = int(r.uniform(.02, .40) * len(Q)); b = Q[i]; v = np.array([r.uniform(.2, 1.), -r.uniform(.2, .9)]); L = r.uniform(30, 130) * up
            c.fill(kit.spike(b - T[i] * 13, b + v * L, b + T[i] * 13, .2, .12), '#04080b')
        c.line(Q[int(.10 * len(Q)):int(.60 * len(Q))] + [-2, -3], 3.2, '#2f7d7c', taper=(.3, .3), press=.6, seed=k, alpha=.55)
    L = comp.blur(c.layer(), 2.6)
    if k >= 10:                                                 # its eye
        o = np.clip((k - 9) / 4., 0, 1); e = Cel(); ex, ey = 520, 1180 - 835 * .93 * up - 6
        e.fill(np.array([(ex - 52, ey + 14), (ex - 10, ey - 15 * o), (ex + 46, ey - 20 * o), (ex + 62, ey - 6), (ex + 8, ey + 10 * o + 4)]), '#eafff4'); e.fill(np.array([(ex - 2, ey - 14 * o), (ex + 13, ey - 15 * o), (ex + 11, ey + 7 * o), (ex + 3, ey + 8 * o)]), '#04080b')
        el = e.layer(); L = L * (1 - el[..., 3:4]) + el; return L, comp.blur(el[..., :3], 16) * .8 * o
    return L, None


class Shot:
    id = 'c3'; n = 72
    finish = dict(bloom=.30, vig=.20)
    EVENTS = [(0, 'cut', dict(note='Megumi, underground hum')), (0, 'cloth', dict(size=.4, note='two sleeves slide in')), (5, 'clasp', dict(size=.6, note='palms meet, dry slap')), (6, 'shadow_rise', dict(size=.8, dur=12, note='ink pool swells, low whoosh')),
              (9, 'lock', dict(size=.8, note='thumbs snap up')), (10, 'beast_breath', dict(size=.5, note='shadow dog eye opens, low growl')),
              (18, 'cut', dict(note='Nobara')), (19, 'nail_slide', dict(size=.6, pitch=0, note='iron shhk')), (23, 'nail_slide', dict(size=.65, pitch=2)), (27, 'nail_slide', dict(size=.8, pitch=4, note='third nail on the beat')), (29, 'leather_creak', dict(size=.4, note='fist tightens')), (28, 'hammer_up', dict(size=.4, note='hammer lifts into frame')), (33, 'glint', dict(size=.7, note='light runs over the hammer face')),
              (36, 'cut', dict(note='Yuji')), (36, 'tremble', dict(size=.4, dur=8, note='cloth and knuckle rattle')), (45, 'clench', dict(size=.9, note='wraps creak tight, shaking stops dead')), (45, 'breath_out', dict(size=.5, dur=9)), (46, 'energy_start', dict(size=.8, dur=8, note='blue cursed energy catches')),
              (54, 'cut', dict(note='Gojo, hard light')), (60, 'cloth', dict(size=.2)), (63, 'finger_snap', dict(size=.9, note='fingers cross, crisp')), (63, 'light_burst', dict(size=.9, dur=9, note='bell-like shimmer'))]

    def setup(self):
        self.G = geometry(); self.P = plates(); self.pal = pals()
        x, y = grid(H, W); self.x, self.y = x, y
        c = Cel(); c.fill(np.array([(-40, -40), (150, -40), (90, 1120), (-40, 1120)]), '#020507'); c.fill(np.array([(-40, 930), (700, 1010), (900, 1120), (-40, 1120)]), '#03080b'); self.fga = comp.blur(c.layer(), 24).astype(np.float16)
        c = Cel(); c.fill(np.array([(1960, -40), (1800, -40), (1880, 560), (1960, 1120)]), '#0a0508'); c.fill(np.array([(-40, 1120), (-40, 820), (190, 900), (420, 1120)]), '#03080b'); self.fgb = comp.blur(c.layer(), 26).astype(np.float16)
        c = Cel(); c.fill(np.array([(-40, -40), (250, -40), (130, 300), (110, 700), (-40, 760)]), '#020308'); self.fgc = comp.blur(c.layer(), 30).astype(np.float16)
        c = Cel(); c.fill(np.array([(-40, 1120), (-40, 640), (170, 760), (330, 1120)]), '#03030c'); c.fill(np.array([(1960, 1120), (1960, 700), (1700, 860), (1500, 1120)]), '#03030c'); self.fgd = comp.blur(c.layer(), 28).astype(np.float16)

    # -------------------------------------------------- a: Megumi
    def fa(self, f):
        u = f / 18.; im = stand(self.P['a'], 1.0 + .03 * u, -20 * u, 0) * (1 - .22 * smooth(4, 14, float(f)))
        fl = .88 if f in (2, 3, 7) else 1.0; im = im * fl                                           # a tube stutters
        L, li = wolf(f); im = comp.over(im, L, .96)
        if li is not None: im = im + li
        g = at(self.G, 'a', f); cel = Cel(); hand(cel, g, self.pal['a'], 5.2, 3, (('k', self.pal['a']['rim'], .92), ('b', self.pal['a']['rim2'], .9)), order=(1, 0), srim=(((-.5, .85), '#4fb5ae', .9),))
        lay = cel.layer(); lay = kit.para(lay, (500, 200), (1700, 1080), [(0, '#ffffff'), (.5, '#cfe4e0'), (1, '#4a6a78')], 'mul', .9)
        # shadow climbs the sleeves
        cr = np.clip((f // 2 * 2 - 4) / 12., 0, 1); c = Cel(); r = np.random.default_rng(90 + f // 2)
        for k in range(13):
            q = np.random.default_rng(300 + k); x0 = 1180 + 62 * k + q.uniform(-20, 20); hgt = (q.uniform(260, 620)) * cr * (1 - .30 * k / 13); x1 = x0 - q.uniform(60, 220) * cr
            P = spl([(x0, 1140), (x0 + q.uniform(-30, 30), 1140 - hgt * .4), ((x0 + x1) / 2 + r.normal(0, 8), 1140 - hgt * .75), (x1, 1140 - hgt)], n=14)
            wd = q.uniform(70, 150); c.line(P, wd, '#03070a', taper=(0, .9), press=.5, seed=k, tmin=.02); t_ = np.linspace(0, 1, len(P))[:, None]
            c.line(P[len(P) // 6:] - np.array([1., .25]) * wd * .46 * np.clip((1 - t_[len(P) // 6:]) / .9, 0, 1) ** .7, 3.4, '#2f8f8c', taper=(.3, .4), press=.6, seed=k + 40, alpha=.75)
        lay2 = c.layer(); lay = lay * (1 - lay2[..., 3:4]) + lay2
        hit = kit.shake(f, 5., 3, 2.0, 5) + kit.shake(f, 6., 5, 1.8, 9)
        z = 1 + .035 * u; lay = move(lay, hit[0] - 10 * u, hit[1] + 2 * math.sin(f * .5), z, 1000, 620)
        im = kit.cast(im, lay, 26, 34, '#020608', .55, 14); im = comp.over(im, lay)
        im = im + fx.motes(f, 70, 5, None, '#bff2e4', (10, -8), 1.2, .45)
        im = comp.over(im, move(self.fga, -34 * u, 0, 1 + .05 * u)); return im

    # -------------------------------------------------- b: Nobara
    def fb(self, f):
        u = f / 18.; im = stand(self.P['b'], 1.02 + .03 * u, 26 * u, 0); p = self.pal['b']
        gh = self.G['h', 0]; cel = Cel(); p0_, p1_, hs_ = gh.extra['ham']; steel_hammer(cel, p0_, p1_, hs_, 3.4, flip=HAMFLIP, shw=.16, pal=('#4d4a60', '#8f8592', '#ffcf9c', '#23222f'))      # before the hand: the fingers close over the shaft
        hand(cel, gh, p, 3.6, 5, (('t', p['rim'], .7), ('w', p['rim2'], .8)), deep=False, holes=True, srim=(((.8, .5), '#ff9a55', 1.),)); hl = cel.layer()
        hl = kit.para(hl, (1900, 200), (1200, 1000), [(0, '#ffe9c8'), (.5, '#d9a488'), (1, '#4a3a4a')], 'mul', .9)
        rise = comp.ease_out(np.clip((f - 10) / 6., 0, 1), 2.4); hl = move(hl, 24 - 10 * u, 250 * (1 - rise) + 3 * math.sin(f * .4), 1., 1480, 800, .10 * (1 - rise)); hl = comp.blur(hl, 2.2)
        im = comp.over(im, hl)
        g = at(self.G, 'b', f); cel = Cel(); hand(cel, g, p, 5.0, 7, holes=True, rims=(('t', p['rim'], .85, ('little', 'thumb', 'steel')), ('w', p['rim2'], .95, ('index', 'thumb', 'steel'))), srim=(((.8, .5), '#ff9a55', .9), ((-.9, .1), '#2f8f8c', .8)))
        lay = cel.layer(); lay = kit.para(lay, (1250, 150), (430, 1050), [(0, '#fff2dc'), (.45, '#f0cdb8'), (1, '#4f5a70')], 'mul', .9)
        dy = sum(-7 * math.exp(-(f - t0) / 1.6) for t0 in (2, 6, 10) if f >= t0); sh = kit.shake(f, 2.2, 8, 1.6, 10)
        lay = move(lay, sh[0] + 8 * u, dy + sh[1], 1 + .03 * u, 760, 640)
        im = kit.cast(im, lay, -40, 30, '#06090c', .5, 18); im = comp.over(im, lay)
        for t0, (gx, gy) in zip((2, 6, 10), g.extra['tips']):                                      # each point catches the lamp as it arrives
            k = f - t0
            if 0 <= k < 4: q = np.array([760, 640]) + (np.array([gx, gy]) - [760, 640]) * (1 + .03 * u); im = kit.flare(im, q[0] + sh[0] + 8 * u, q[1] + dy + sh[1], 64, '#ffd9a0', .9 * (1 - k / 4), streak=.8, ang=-.5)
        k = f - 15
        if 0 <= k < 3: d_ = (p1_ - p0_) / np.hypot(*(p1_ - p0_)); n_ = np.array([-d_[1], d_[0]]); n_ = n_ if n_[1] < 0 else -n_; q = p1_ + n_ * hs_ * .95 + d_ * hs_ * (-.30 + .24 * k) + [24 - 10 * u, 0]; im = kit.flare(im, q[0], q[1], 120, '#fff0cf', (1., .8, .35)[k], streak=1.2, ang=.35)
        im = im + fx.motes(f, 90, 9, smooth(1200, 100, np.hypot(self.x - 1700, self.y - 330)), '#ffd9a0', (-14, -6), 1.3, .8)
        return comp.over(im, move(self.fgb, 30 * u, 0, 1 + .05 * u))

    # -------------------------------------------------- c: Yuji
    def fc(self, f):
        u = f / 18.; e = float(np.clip((f - 9.) / 7., 0, 1)); e = e * e * (3 - 2 * e); p = mixpal(self.pal['c'], self.pal['c2'], e)
        im = stand(self.P['c'], 1.02 + .02 * u, 10 * u, 0); fk = 1 + .10 * math.sin(f * 2.3) * (e > 0)
        im = im + (smooth(1500, 100, np.hypot(self.x - 900, self.y - 760)) ** 1.5)[..., None] * rgb('#173fae') * .55 * e * fk
        g = self.G['c', (f % 4) if f < 8 else (8 if f == 8 else 9 if f == 9 else 10)]
        cel = Cel(); rims = [('t', '#a9b6da', .7 * (1 - e)), ('r', '#8f3346', .75 * (1 - e))] + ([('e', p['rim'], min(1., e * 1.3)), ('f', p['rim2'], min(1., e * 1.2))] if e > .02 else [])
        hand(cel, g, p, 6.0, 5, rims, srim=(((.9, .15), '#8f2a35', .9 * (1 - e) + .02), ((-.4, -.8), '#4f8cff', .05 + 1.1 * e))); lay = cel.layer(); lay = kit.para(lay, (800, 40), (1250, 1080), [(0, '#ffffff'), (.5, '#d8d8e8'), (1, '#54547a')], 'mul', .9 - .4 * e)
        r = np.random.default_rng(700 + f); jit = r.normal(0, 1, 2) * (4.6 if f < 8 else 0.); rot = r.normal(0, .004) if f < 8 else 0.
        drop = 10 * math.exp(-(f - 9) / 1.5) if f >= 9 else (6 if f == 8 else 0)
        zz = 1 + .02 * u + (.018 * math.exp(-(f - 9) / 2.) if f >= 9 else 0); lay = move(lay, jit[0], jit[1] + drop + (2.2 * math.sin((f - 9) * .45) if f > 9 else 0), zz, 850, 520, rot)
        back = front = None
        if e > .03: back, front = fx.cursed(lay, f + 100, (.25, -1), size=150 * (.45 + .55 * e), seed=4, amount=.25 + .75 * e, wisps=.4 + .6 * e, light=.9)
        im = kit.cast(im, lay, 30, 40, '#010207', .55, 18)
        if back is not None: im = fx.put(im, back)
        im = comp.over(im, lay)
        if front is not None: im = fx.put(im, front)
        if 9 <= f:                                             # breath out, drawn: three lobed puffs sliding across in front, thinning
            k = f - 9; c = Cel(); q = np.random.default_rng(5)
            for j in range(16):
                t = k / 9.; cx = 1500 - 520 * t * (.6 + .05 * j) - 46 * j + q.normal(0, 30); cy = 60 + 250 * t * (.7 + .02 * j) + 15 * j + q.normal(0, 26); R_ = (60 + 11 * j) * (.45 + .75 * t) * q.uniform(.7, 1.25); th = np.linspace(0, 2 * math.pi, 11)[:-1]
                rr = R_ * (1 + .22 * np.sin(th * 3 + j * 2.1 + k // 2 * .6) + .10 * np.sin(th * 5 + j)); c.fill(spl(np.c_[cx + np.cos(th) * rr * 1.5, cy + np.sin(th) * rr * .62], True, n=5), '#d5e0f6', .16)
            m = comp.blur(c.layer(), 9 + 2.2 * k); im = comp.over(im, m, (1 - k / 9.5) ** .7)
        if f < 9:
            c = Cel(); q = np.random.default_rng(60 + f)                # a little sweat and dust shaken loose
            for k in range(7): pt = np.array([q.uniform(560, 1150), q.uniform(760, 980) + 10 * f]); c.fill(spl(pt + np.array([(0, -4), (2.4, 2), (0, 5), (-2.4, 2)]), True, n=4), '#c9d0e8', .8)
            im = comp.over(im, c.layer())
        return comp.over(im, move(self.fgc, 20 * u, 0, 1 + .04 * u))

    # -------------------------------------------------- d: Gojo
    def fd(self, f):
        u = f / 18.; p = self.pal['d']; b = math.exp(-(f - 9) / 4.) if f >= 9 else 0.; pre = smooth(5, 8.5, float(f)) * (f < 9)
        im = stand(self.P['d'], 1.02 + .03 * u + .02 * b, 14 * u, -6 * u) * (1 - .30 * pre) * (1 + .22 * b)
        cx, cy = 1060, 330
        g = at(self.G, 'd', f); cel = Cel(); hand(cel, g, p, 4.8, 4, (('w', p['rim'], 1.), ('c', p['rim2'], .9)), srim=(((.9, .35), '#cfe6ff', 1.0), ((-.95, .1), '#4f7fe0', .8)))
        lay = cel.layer(); lay = kit.para(lay, (1100, 200), (650, 1080), [(0, '#ffffff'), (.5, '#c8c4ee'), (1, '#4a4688')], 'mul', .75)
        dy = 2.5 * math.sin(f * .33) + (9 * math.exp(-(f - 9) / 1.6) if f >= 9 else 0) - 8 * pre; sh = kit.shake(f, 3.0, 2, 1.6, 9)
        lay = move(lay, sh[0] + 4 * u, dy + sh[1], 1 + .03 * u, 830, 470)
        im = kit.aura(im, lay, 22, '#7fc4ff', .30 + .35 * b); im = comp.over(im, lay)
        im = im + (smooth(900, 60, np.hypot((self.x - cx) / 1.2, self.y - cy)) ** 2.4)[..., None] * rgb('#8fc4ff') * (.10 + .10 * b)
        lm = smooth(1300, 80, np.hypot(self.x - cx, self.y - cy)); im = im + fx.motes(f, 120, 3, lm, '#dff2ff', (5, -9 - 30 * b), 1.4, .9)
        return comp.over(im, move(self.fgd, -20 * u, 0, 1 + .06 * u))

    def frame(self, f):
        im = (self.fa, self.fb, self.fc, self.fd)[min(3, f // 18)](f % 18 if f < 72 else 17)
        return np.ascontiguousarray(im, dtype=f32)
