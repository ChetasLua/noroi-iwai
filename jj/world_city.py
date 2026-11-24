"""Tokyo at night: the scramble crossing, a street, the flying skyline."""
import numpy as np, cv2, math
from .paint import rgb, f32, fbm, smooth, mix, grid, Canvas, arcs, jit_color, lum, warp_by
from .cam import Camera3D
from .comp import blur, radial, over
from . import world as Wd, world_sky as SK
from .world import C, Scene, Wall, Plate, painter, FAR
from .world_props import *

# ------------------------------------------------------------------------------------------------ the place
# O = crossing centre.  +Y is the way we look.  Two streets fan out north (NL, NR) round a wedge block with the big screen.
U = lambda a: np.array([math.sin(math.radians(a)), math.cos(math.radians(a))])
ROADS = dict(S=(U(176), 30.), NL=(U(-27), 13.), NR=(U(24), 21.), W=(U(-84), 24.), E=(U(82), 22.))


def _edge(name, side, r0, r1):
    """points of a road edge at distances r0,r1 from O; side -1 left / +1 right when walking away from O"""
    d, w = ROADS[name]; n = np.array([d[1], -d[0]]) * side * w / 2
    return d * r0 + n, d * r1 + n


def row(name, side, r0, spec):
    """lots along a road edge.  spec = [(width, depth, height, style, pal, kw)]; returns [(foot, h, style, pal, kw)]"""
    d, w = ROADS[name]; n = np.array([d[1], -d[0]]) * side * w / 2; out = []; t = r0
    for (wd, dp, h, st, pal, kw) in spec:
        a, b = d * t + n, d * (t + wd) + n
        foot = lot(a, b, dp) if side < 0 else lot(b, a, dp)      # street must lie on the right of a->b
        out.append((foot, h, st, pal, dict(kw))); t += wd + kw.get('gap', .25)
    return out


P1, P2, P3, P4, P5, P6, P7 = '#6b6f86', '#585c74', '#8a8578', '#7a7f92', '#4d5470', '#8f8a94', '#6e6574'


def shibuya():
    L = []
    T1, T2, T3, T4, T5, T6 = '#8f8678', '#5a5f74', '#a39c92', '#6d7288', '#4a5068', '#7a6a66'
    # wedge rows behind the screen building
    L += row('NL', +1, 54, [(11, 12, 27, 'tenant', T3, dict(shop=4.2, blades=2, g=1)), (10, 12, 33, 'tenant', T6, dict(shop=4, blades=3, g=1)), (15, 14, 24, 'grid', T1, dict(shop=4, blades=2, g=1)), (18, 16, 36, 'office', T4, dict(shop=4, g=2)), (24, 18, 30, 'resid', T3, dict(g=2))])
    L += row('NR', -1, 54, [(13, 12, 26, 'tenant', T2, dict(shop=4.6, blades=2, g=1)), (12, 13, 33, 'tenant', T3, dict(shop=4.4, blades=3, g=1)), (18, 16, 28, 'grid', T6, dict(shop=4.4, blades=1, g=2)), (22, 18, 42, 'curtain', T5, dict(shop=4.4, g=2)), (30, 20, 36, 'office', T2, dict(g=2))])
    # east of NR
    L += row('NR', +1, 30, [(17, 20, 35, 'tenant', T4, dict(shop=5, blades=2, bill=1., g=3)), (13, 18, 27, 'grid', T3, dict(shop=4.2, blades=2, g=3)), (22, 22, 46, 'office', T2, dict(shop=4.2, blades=1, g=3)), (18, 20, 25, 'tenant', T6, dict(shop=4, blades=2, g=3)), (30, 26, 52, 'fins', T4, dict(g=3))])
    L += row('E', -1, 36, [(16, 16, 31, 'tenant', T1, dict(shop=4.2, blades=2, g=3)), (20, 20, 24, 'grid', T6, dict(shop=4, blades=2, g=3)), (30, 24, 62, 'curtain', T5, dict(g=3)), (40, 26, 44, 'office', T4, dict(g=3))])
    L += row('E', +1, 24, [(24, 24, 12.5, 'grid', T5, dict(shop=4.4, blades=1, bill=0., g=4)), (28, 26, 26, 'office', T6, dict(shop=4.4, blades=2, g=4)), (40, 30, 34, 'grid', T3, dict(g=4))])
    # west of NL: low kiosks in front of the drum, then tenant buildings
    L += row('NL', -1, 24, [(8, 5.2, 9.5, 'tenant', T6, dict(shop=4.2, blades=1, g=5)), (9, 5.2, 13, 'tenant', T3, dict(shop=4.2, blades=2, g=5)), (8, 5.2, 10.5, 'grid', T1, dict(shop=4.2, blades=1, g=5)), (12, 14, 29, 'tenant', T3, dict(shop=4.2, blades=3, g=5)), (13, 15, 24, 'grid', T2, dict(shop=4.2, blades=2, g=5)), (16, 18, 37, 'office', T6, dict(shop=4, blades=1, g=5)), (26, 20, 32, 'dept', T4, dict(g=5)), (30, 22, 46, 'fins', T4, dict(g=5))])
    L += row('W', +1, 49, [(16, 16, 30, 'tenant', T2, dict(shop=5, blades=2, bill=1., g=5)), (16, 18, 26, 'grid', T3, dict(shop=4, blades=2, g=5)), (28, 24, 48, 'office', T4, dict(g=5)), (34, 26, 40, 'grid', T1, dict(g=5))])
    L += row('W', -1, 24, [(24, 26, 14.5, 'tenant', T5, dict(shop=4.6, blades=1, bill=0., g=6)), (24, 24, 30, 'curtain', T5, dict(shop=4.4, blades=1, g=6)), (36, 30, 42, 'office', T4, dict(g=6))])
    return L


def far_city(S, r, eye, f, rmin=170, rmax=1500, n=420, lit=.45, span=(-62, 62), grp=2, hgain=1., em=1., det0=0):
    """sea of buildings behind: every block is its own box with its own lights"""
    a = np.radians(r.uniform(*span, n)); d = rmin + (rmax - rmin) * r.random(n) ** 1.4
    for k in np.argsort(-d):
        x, y = math.sin(a[k]) * d[k], math.cos(a[k]) * d[k]; w, dp = r.uniform(12, 38, 2); h = (14 + r.gamma(2., 12)) * hgain
        if r.random() < .10: h *= r.uniform(1.8, 3.2)
        h = min(h, 3.4 * min(w, dp))
        rot = r.uniform(-.5, .5); c, s = math.cos(rot), math.sin(rot); q = np.array([(-w, -dp), (w, -dp), (w, dp), (-w, dp)]) / 2 @ np.array([[c, s], [-s, c]]) + [x, y]
        st = pick(r, ['office', 'grid', 'curtain', 'resid', 'office', 'grid']); building(S, q, h, st, r, eye, f, lit * r.uniform(.5, 1.3), pick(r, [P1, P2, P4, P5, P6, P7]), 0, d[k] < 500, grp, detmax=0 if d[k] > 330 else 1, glow=em)


def grounds(S, r, wet=True):
    """asphalt, pavements, kerbs, markings"""
    S.ground(-420, 420, -120, 900, 0., '#54565f', nx=30, ny=60, pw=1.6, tex=2, seed=11)
    def pave(poly, col='#82838a'): S.prism(poly, 0, .16, '#8a8c94', top=col, tex=3)
    e = lambda n, s, a, b: _edge(n, s, a, b)
    cor = []
    for (A, sa, B, sb, rr, bev) in (('NL', +1, 'NR', -1, 34, 0), ('NR', +1, 'E', -1, 22, 0), ('W', +1, 'NL', -1, 22, 0)):
        a0, a1 = e(A, sa, rr, 300); b0, b1 = e(B, sb, rr, 300); m = (a0 + b0) / 2 * .86
        p = np.array([a1, a0, m * 1.0, b0, b1]) if sa > 0 else np.array([b1, b0, m, a0, a1])
        area = np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]); p = p if area > 0 else p[::-1]
        S.prism(p, 0, .16, '#9a9ca4', top='#83848c', tex=3); cor.append(p)
    for (A, sa, B, sb, rr) in (('E', +1, 'S', -1, 22), ('S', +1, 'W', -1, 22)):
        a0, a1 = e(A, sa, rr, 260); b0, b1 = e(B, sb, rr, 160); m = (a0 + b0) / 2 * .9; p = np.array([a1, a0, m, b0, b1, (a1 + b1)]); area = np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]); p = p if area > 0 else p[::-1]
        S.prism(p, 0, .16, '#9a9ca4', top='#83848c', tex=3); cor.append(p)
    return cor


def markings(S, r, worn=0.):
    A = dict(); 
    def cross(name, r0, wid, **k):
        d, w = ROADS[name]; n = np.array([d[1], -d[0]]); zebra(S, d * r0 - n * (w / 2 - .3), d * r0 + n * (w / 2 - .3), wid, skip=worn, r=r, **k)
    cross('S', 21, 6.5); cross('NR', 25, 6.5); cross('W', 22, 6.); cross('E', 21, 6.); cross('NL', 27, 5.)
    zebra(S, np.array([-14.5, -12.5]), np.array([13.8, 15.5]), 7.0, skip=worn, r=r)                         # the diagonal
    for nm in ('S', 'NR', 'W', 'E'):
        d, w = ROADS[nm]; n = np.array([d[1], -d[0]])
        strip(S, [d * 27.5 - n * (w / 2 - .5) * (1 if nm in ('S', 'NR') else 1), d * 27.5 + n * 0], .45) if worn < .5 else None
        for q in (-.25, 0., .25): strip(S, [d * 30 + n * w * q, d * 150 + n * w * q], .16, dash=(3., 4.) if q else (6., .01), col='#d8d6cc' if q else '#d8b24a')
        for q in (-.46, .46): strip(S, [d * 28 + n * w * q, d * 180 + n * w * q], .15)
    for nm in ('S', 'NR'):
        d, w = ROADS[nm]; n = np.array([d[1], -d[0]]); p = d * 34 + n * w * .125; 
        S.quads(np.array([[*(p - n * .12 - d * 2), .02], [*(p + n * .12 - d * 2), .02], [*(p + n * .12 + d * 1), .02], [*(p - n * .12 + d * 1), .02]])[None], '#d4d2ca', tex=8, bias=4e-4)


def walkers(r, n):
    """people inside the crossing: streams along each zebra plus the scramble, plus crowds waiting on the corners"""
    P = []
    def stream(a, b, m, wid):
        a, b = np.asarray(a, float), np.asarray(b, float); d = b - a; nn = np.array([-d[1], d[0]]) / np.hypot(*d); t = r.random(m) * 1.1 - .05; P.append(a + d * t[:, None] + nn * (r.normal(0, .38, m) * wid)[:, None])
    for nm, r0, w_, m in (('S', 21, 5, .13), ('NR', 25, 5, .13), ('W', 22, 5, .10), ('E', 21, 5, .10), ('NL', 27, 5, .08)):
        d, w = ROADS[nm]; q = np.array([d[1], -d[0]]); stream(d * r0 - q * w / 2, d * r0 + q * w / 2, int(n * m), w_)
    stream((-15, -13), (14, 16), int(n * .22), 6); stream((15, -12), (-13, 15), int(n * .12), 5)
    for nm, sd, r0 in (('NL', 1, 35), ('NR', 1, 24), ('NL', -1, 25), ('E', 1, 24), ('S', 1, 25)):
        a, b = _edge(nm, sd, r0, r0 + 8); m = int(n * .05); q = ROADS[nm][0]; q = np.array([q[1], -q[0]]) * sd; P.append(a + (b - a) * r.random((m, 1)) + q * (r.random((m, 1)) * 6 + .8))
    return np.vstack(P)


def scramble_scene(seed=0, cam=None, crowd_n=420, cars=True, lit=.55, far=True, worn=0.):
    S = Scene(seed + 5); r = np.random.default_rng(seed + 21); eye = cam.eye; f = cam.f; g = None
    S.groups = 10
    grounds(S, r); markings(S, r, worn)
    for (foot, h, st, pal, kw) in shibuya():
        building(S, foot, h, st, np.random.default_rng(int(abs(foot[0, 0] * 71 + foot[0, 1] * 13 + h * 7)) + seed), eye, f, lit, pal, shop=kw.get('shop', 0), grp=kw.get('g', 1), bill=kw.get('bill', .25), blades=kw.get('blades', 0))
    # the screen building: a blunt prow on the wedge, its whole face an LED wall
    aL = _edge('NL', +1, 53.5, 54)[0]; aR = _edge('NR', -1, 53.5, 54)[0]; L0 = _edge('NL', +1, 42, 46)[0]; R0 = _edge('NR', -1, 42, 46)[0]; H_ = 35.
    foot = np.array([aL, L0, R0, aR]); foot = foot if np.sum(foot[:, 0] * np.roll(foot[:, 1], -1) - np.roll(foot[:, 0], -1) * foot[:, 1]) > 0 else foot[::-1]
    for i in range(4):
        W = Wall(S, foot[i], foot[(i + 1) % 4], 0, H_, 1)
        if np.dot(eye - W.o, W.n) <= 0: continue
        facade(W, 'fins', r, lit, '#5a6076', 2, 5.4)
    S.poly(np.c_[foot, np.full(4, H_)], '#4a4e60', tex=1)
    W2 = Wall(S, L0, R0, 0, H_, 1); W2.slab(.5, W2.w - .5, 12.2, H_ - 1.6, 0, .9, '#12141c', side='#2a2e3e'); screen_art(W2, 1.0, W2.w - 1.0, 12.7, H_ - 2.1, .93, r, 'iris', em=1.12)
    W2.slab(.5, W2.w - .5, 5.6, 11.6, 0, .6, '#12141c', side='#2a2e3e'); lightbox(W2, .9, W2.w * .46, 6., 11.2, .63, r, NEON[6], .95, 'wave'); lightbox(W2, W2.w * .5, W2.w - .9, 6., 11.2, .63, r, NEON[0], .95, 'disc')
    shopfront(Wall(S, L0 + [0, -.1], R0 + [0, -.1], .1, 5.2, 1), r, 5.0)
    for u in (0., W2.w): S.cyl(W2.p(u, 0, -.2)[:2], 1.15, 0, H_ + .8, '#7c8298', 14, tex=7)
    rooftop(S, foot, H_, r, 2, True, 1, 0)
    c = (L0 + R0) / 2
    S.light((-3, 2, 26), '#b8c8f0', 60, 26, 1.5, 0); S.light((c[0], c[1] - 13, 22), '#58b8ff', 70, 24, 1.8, 1); S.light((c[0], c[1] - 9, 6), '#ffb07a', 34, 10, 1.6, 1)
    # the drum tower
    cc = U(-55) * 45; RR = 9.6; a = np.linspace(0, 2 * np.pi, 33)[:-1]; ft = np.c_[cc[0] + np.cos(a) * RR, cc[1] + np.sin(a) * RR]; DH = 38.; S.prism(ft, 0, DH, '#7e8398', top='#6a6e80', tex=6)
    for i in range(32):
        W = Wall(S, ft[i], ft[(i + 1) % 32], 0, DH, 5)
        if np.dot(eye - W.o, W.n) > 0: W.slab(0, W.w, np.arange(8, DH - 1, 5.), np.arange(8, DH - 1, 5.) + .35, 0, .25, '#b4bace'); W.slab(0, W.w, DH - 2.2, DH - .6, 0, .3, '#e8dcc8', em=.35, grp=5)
    ang = math.atan2(eye[1] - cc[1], eye[0] - cc[0])
    for q, art in ((-.42, 'koi'), (.50, 'bloom')):
        dv = np.array([math.cos(ang + q), math.sin(ang + q)]); p = cc + dv * (RR + .5); ex = np.array([-math.sin(ang + q), math.cos(ang + q), 0]); W = Wall.frame(S, np.r_[p, 13.] - ex * 3.8, ex, (0, 0, 1), 7.6, 20, 5); W.slab(0, 7.6, 0, 20, -.5, -.02, '#14161f'); screen_art(W, .25, 7.35, .25, 19.75, .03, r, art, em=1.05)
        S.light(np.r_[p + dv * 11, 16.], '#ff6a5a' if art == 'koi' else '#ff7ac0', 50, 16, 1.0, 5)
    rooftop(S, ft, DH, r, 2, True, 5)
    # second rank towers that break the skyline
    for (ang_, dist, w, dp, h, st, g_, pl) in ((-44, 150, 34, 30, 86, 'fins', 5, P5), (-30, 210, 30, 26, 62, 'office', 5, P2), (-13, 260, 30, 30, 70, 'office', 2, P2), (-7, 300, 34, 30, 58, 'fins', 2, P5), (11, 300, 26, 26, 52, 'office', 2, P4), (38, 165, 34, 30, 92, 'office', 3, P5), (28, 260, 30, 30, 66, 'fins', 3, P4), (56, 130, 30, 28, 84, 'curtain', 3, P5), (-64, 125, 30, 26, 78, 'office', 5, P1)):
        p = U(ang_) * dist; rot = math.radians(-ang_ * .6); c_, s_ = math.cos(rot), math.sin(rot); q = np.array([(-w, -dp), (w, -dp), (w, dp), (-w, dp)]) / 2 @ np.array([[c_, s_], [-s_, c_]]) + p; building(S, q, h, st, np.random.default_rng(int(dist) + seed), eye, f, lit * .9, pl, 0, True, g_)
    if far: far_city(S, np.random.default_rng(seed + 9), eye, f, lit=lit * .8)
    # street life
    for (x, y, ar, g_) in ((17, -13, (-2, 1), 4), (-17, -15, (2, 1), 6), (13, 27, (-1.8, -1), 3), (-17, 20, (1.8, -1), 5), (2.5, 31, (0, -1.8), 1), (20, 24, (1.5, 0), 3), (-24, 14, (-1.5, 0), 5)): lamp(S, (x, y), 9., ar, '#ffd9a0', g_, .16, 1.0, 26)
    for nm, sd, g_ in (('NR', -1, 1), ('NR', 1, 3), ('NL', 1, 1), ('NL', -1, 5), ('W', 1, 5), ('E', -1, 3)):
        d, w = ROADS[nm]; n = np.array([d[1], -d[0]]) * sd
        for k in range(5): p = d * (48 + k * 22 + r.uniform(-3, 3)) + n * (w / 2 + .8); lamp(S, p, 7.5, -n * 1.3, '#ffdca8', g_, .16, .9, 20)
        for k in range(4):
            p = d * (42 + k * 17 + r.uniform(-4, 4)) + n * (w / 2 + 1.6)
            if r.random() < .6: tree(S, p, r.uniform(5, 7.5), r, .16, ('#33503c', '#4c7048', '#22362e'), n=14)
    signal(S, (13.4, -20.5), (-6.5, 1.5), h=7.4, col='red', grp=9, z=.16, face=(-.15, -1), heads=2, k=1.7); signal(S, (-15.5, 19.), (6.0, -1.), col='red', grp=8, z=.16, face=(.1, -1), heads=1); signal(S, (15.5, 19.5), (-6, -.6), col='red', grp=8, z=.16, face=(-.1, -1), heads=1); signal(S, (-17.5, -18.5), (4.5, .8), col='green', grp=8, z=.16, face=(0, -1), heads=1)
    for nm in ('S', 'NR', 'E', 'W'):
        d, w = ROADS[nm]; n = np.array([d[1], -d[0]]); m = 5
        if not cars: break
        for k in range(m):
            for ln, dirn in ((.125, 1), (.375, 1), (-.125, -1), (-.375, -1)):
                if r.random() < .72:
                    t = 31 + k * 7.2 + r.uniform(-.8, .8) if dirn > 0 else 40 + k * 11 + r.uniform(-4, 4); p = d * t + n * w * ln * -1 * (1 if True else 1); a = math.atan2(-d[1], -d[0]) if dirn > 0 else math.atan2(d[1], d[0])
                    kind = 'bus' if (k == 1 and ln == .375 and nm == 'NR') else pick(r, ['taxi', 'black', 'white', 'black', 'van', 'blue', 'taxi', 'red', 'white'])
                    if np.hypot(*(p - eye[:2])) > 9: car(S, p[0], p[1], a, r, kind, .0, 1., 7)
    if crowd_n: crowd(S, walkers(r, crowd_n), cam, r, 0., grp=0, umb=.04)
    return S, r


NIGHT = dict(amb=('#1c2244', '#12121f'), street=('#7a5238', 9., (0, 10, 130, .35)), fills=[((-.8, -.5, .1), '#2a1430'), ((.8, -.5, .1), '#0c2438'), ((0, -1, .2), '#0c0e1c')], fog=dict(col='#3c3268', col2='#5a4480', dir=(0, 1, 0), dens=.0030, start=95, hf=0, floor=1., keep=.45, pw=1.5))


def finish_night(S, cam, res, sky, wet, seed, rk, refl=.8, ao=.42):
    h, w = cam.h, cam.w
    img = res['img'] + sky * (1 - res['a'][..., None])
    geo = (res['z'] < FAR * .5).astype(f32)
    img = img * (1 + (Wd.occlusion(res['z'], ao) - 1) * geo)[..., None]
    return img


@painter('scramble', 5)
def scramble(w=2400, h=1350, seed=0, lights=None, eye=(-7., -80., 37.), target=(1.5, 20., 28.), focal=1150., crowd_n=430, paint=True, ss=2, lit=.55, cars=True):
    """Shibuya-like scramble crossing at night from a high angle.  lights = 10 gains for the light groups:
    0 always, 1 wedge screen block, 2 far city, 3 north-east, 4 south-east, 5 north-west, 6 south-west, 7 cars, 8 other signals, 9 the last red signal.
    masks: sky, ground.  project people with P.cam; the crossing centre is (0,0,0), metres."""
    cam = Camera3D(eye, target, focal, w, h)
    S, r = scramble_scene(seed, cam, crowd_n, cars, lit)
    hz = Wd.horizon_y(cam)
    sky = SK.night(h, w, hz + 30, seed + 2, glow='#7c4a80', gx=.54)
    lg = [1.] * 10 if lights is None else list(lights); city = float(np.mean(lg[1:7]))
    sky = sky * (.45 + .55 * city)
    nk = dict(NIGHT); k = .42 + .58 * city
    nk['amb'] = tuple(C(c) * k for c in NIGHT['amb']); nk['fills'] = [(d, C(c) * k) for d, c in NIGHT['fills']]; nk['street'] = (C(NIGHT['street'][0]) * max(0., city * 1.15 - .15), *NIGHT['street'][1:]); nk['fog'] = dict(NIGHT['fog']); nk['fog']['col'] = C(NIGHT['fog']['col']) * k; nk['fog']['col2'] = C(NIGHT['fog']['col2']) * k
    baked = S._bake(cam)
    res = S.render(cam, ss=ss, groups=lg, baked=baked, **nk)
    X = cam.eye + cam.rays() * res['z'][..., None]
    gmask = ((np.abs(X[..., 2]) < .12) & (res['z'] < FAR * .5)).astype(f32); g2 = ((np.abs(X[..., 2] - .16) < .06) & (res['z'] < FAR * .5)).astype(f32)
    pud = smooth(-.3, .9, fbm(h, w, (170, 60, 22), (1, .6, .3), seed + 41, aspect=2.6))
    wet = cv2.GaussianBlur(gmask, (0, 0), 1.0) * (.34 + .50 * pud) + g2 * (.16 + .2 * pud)
    img, _ = Wd.wet_reflect(S, cam, res, sky, wet, baked=baked, strength=.9, seed=seed + 3, groups=lg, **nk)
    geo = (res['z'] < FAR * .5).astype(f32)
    img = img * (1 + (Wd.occlusion(res['z'], .40) - 1) * geo)[..., None]
    img = img + sky * (1 - res['a'][..., None])
    if paint:
        img = Wd.wobble(img, .5, 30, seed + 12)
        ang = res['ang'] + fbm(h, w, (160, 60), (1, .5), seed + 8) * .10
        img = Wd.gouache(img, res['lab'], ang, seed=seed + 4, sizes=(18, 9, 4.5), op=(.6, .55, .5), val=(.050, .042, .034), hue=.016, thr=(.016, .028, .05), mask=geo)
        e = np.maximum(img - .62, 0); img = Wd.lamp_glow(img, e * 1.3, ((5, .26), (20, .22))); img = Wd.air_glow(img, res['z'], .45, (50, 140), .42, (30, 260))
        img = Wd.tooth_pass(img, seed + 6)
    return Plate(img, res['z'], cam, dict(sky=1 - res['a'], ground=np.clip(gmask + g2, 0, 1)))


def veil_field(cam, w, h, n=44, seed=3):
    """static geometry of the ink veil: per-pixel (azimuth, elevation) on the sky dome plus a set of runs.  use with veil_alpha"""
    d = cam.rays(h, w); d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    return np.arctan2(d[..., 0], d[..., 1]).astype(f32), np.arcsin(np.clip(d[..., 2], -1, 1)).astype(f32)


class Veil:
    """black ink running down the dome of the sky.  V = Veil(cam, w, h); rgba = V.layer(t) premultiplied, t 0 (a bead at the zenith) .. 1 (sky gone)"""

    def __init__(s, cam, w, h, seed=3, n=64, q=2, az=None, el=None):
        s.w, s.h, s.q = w, h, q
        if az is None: az, el = veil_field(cam, w, h)
        s.az = cv2.resize(az, (w // q, h // q), interpolation=cv2.INTER_AREA); s.el = cv2.resize(el, (w // q, h // q), interpolation=cv2.INTER_AREA)
        r = np.random.default_rng(seed); s.a0, s.a1 = float(az.min()) - .1, float(az.max()) + .1
        s.n = n; s.ca = np.sort(r.uniform(s.a0, s.a1, n)); sz = r.random(n) ** 2.2
        s.cw = .005 + .034 * sz; s.sp = r.uniform(.55, 1.25, n) * (.7 + .5 * sz); s.dl = r.random(n) * .34; s.cv = r.uniform(.6, 1.6, n)
        s.A = np.linspace(s.a0, s.a1, 4096).astype(f32)
        s.base = np.interp(np.linspace(0, 1, 4096), np.linspace(0, 1, 40), r.standard_normal(40)).astype(f32); s.base = cv2.GaussianBlur(s.base[None], (0, 0), 40)[0]
        s.tex = fbm(h // q, w // q, (70, 26, 10), (1, .6, .4), seed + 2, aspect=.35)
        s.wob = fbm(h // q, w // q, (60, 22), (1, .5), seed + 4)
        s.noise = r.standard_normal((n, 24))

    def front(s, t, lead=0.):
        """elevation of the ink edge along azimuth"""
        t = max(0., t) + lead
        e = .66 - .66 * t ** 1.5 + (.055 * s.base + .012 * np.roll(s.base[::-1], 900)) * min(1., t * 2.2) * (1 - .6 * t)
        E = e.astype(f32)
        for i in range(s.n):
            tt = (t - s.dl[i]) / (1 - .34)
            if tt <= 0: continue
            j = int((s.ca[i] - s.a0) / (s.a1 - s.a0) * 4095); head = e[j] - (.10 + .42 * s.sp[i]) * tt ** .75 * (1 - .45 * t) * (.35 + .65 * s.cw[i] / .036)   # tip elevation
            x = np.abs(s.A - s.ca[i]) / s.cw[i]; k = x < 3.2
            if not k.any(): continue
            xx = x[k]; bead = head + s.cw[i] * (1 - np.sqrt(np.clip(1 - np.minimum(xx, 1) ** 2, 0, 1)))
            neck = bead + np.maximum(0, xx - .86) ** 1.5 * s.cw[i] * 34
            E[k] = np.minimum(E[k], neck.astype(f32))
        return cv2.GaussianBlur(E[None], (0, 0), 1.2)[0]

    def layer(s, t, col='#050308', rim='#4a2a86', film=.55, edge=1., wet=0.):
        az = s.az + .010 * s.wob * .5
        def samp(E): return np.interp(az.ravel(), s.A, E).reshape(az.shape).astype(f32)
        E = samp(s.front(t)); px = float(np.abs(np.gradient(s.el, axis=0)).mean()) + 1e-6
        d = (s.el - E) / px                                                  # px inside the ink
        a = np.clip(d + .5, 0, 1)
        E2 = samp(s.front(t, .05)); a2 = np.clip((s.el - E2) / px + .5, 0, 1)
        body = a; thin = np.clip(a2 - a, 0, 1) * film * (.55 + .25 * np.clip(s.tex, -1, 1))
        r_ = np.exp(-((d - 5 * edge) / (4. * edge)) ** 2) * a * .55 + np.exp(-((d - 16 * edge) / (10. * edge)) ** 2) * a * .16
        streak = np.clip(s.tex - .9, 0, 1) * a * .10
        c = C(col) + (C(rim)) * (r_ + streak)[..., None]
        if wet > 0:                                                      # thick wet lip: a bright line inside the left flanks and the beads
            gx = cv2.Sobel(cv2.GaussianBlur(d, (0, 0), 1.2), cv2.CV_32F, 1, 0) / 8; sd = np.clip(gx * .9 + .25, 0, 1); c = c + C('#a8a0d8') * (wet * np.exp(-((d - 4.5 * edge) / (1.5 * edge)) ** 2) * a * sd * np.clip(.7 + .8 * s.tex, 0, 1))[..., None]
        A = np.clip(body + thin, 0, 1); rgb_ = c * body[..., None] + (C(col) * .6 + C(rim) * .22) * thin[..., None]
        L = np.dstack([rgb_, A]).astype(f32)
        return cv2.resize(L, (s.w, s.h), interpolation=cv2.INTER_LINEAR) if s.q > 1 else L


# ------------------------------------------------------------------------------------------------ street level
def street_scene(seed, cam, wid=11., length=190., lit=.6, ybeg=-14., hs=(13, 30), lamps=True, signs=1., glow=.62, spill=1.1):
    S = Scene(seed + 3); r = np.random.default_rng(seed + 31); eye = cam.eye; f = cam.f; S.groups = 10
    S.ground(-wid / 2, wid / 2, ybeg, length * 3, 0., '#5a5b66', nx=8, ny=50, pw=1.8, tex=2, seed=12)
    for sd in (-1, 1):
        a = np.array([sd * wid / 2, ybeg]); b = np.array([sd * wid / 2, length * 3]); P = np.array([[sd * wid / 2, ybeg], [sd * (wid / 2 + 2.6), ybeg], [sd * (wid / 2 + 2.6), length * 3], [sd * wid / 2, length * 3]]); P = P if sd > 0 else P[::-1]
        S.prism(P, 0, .15, '#8c8e96', top='#7f8088', tex=3)
        strip(S, [[sd * (wid / 2 - .5), ybeg], [sd * (wid / 2 - .5), length * 2]], .15)
    strip(S, [[0, ybeg], [0, length * 2]], .16, dash=(3., 3.))
    for sd, g in ((-1, 5), (1, 3)):
        y = ybeg
        while y < length * 1.6:
            w = r.uniform(6.5, 13.5); dp = r.uniform(12, 20); h = r.uniform(*hs) * (1.25 if r.random() < .2 else 1.); st = pick(r, ['tenant', 'tenant', 'grid', 'office', 'tenant', 'brick', 'resid']); x = sd * (wid / 2 + 2.6)
            a, b = ((x, y), (x, y + w)) if sd < 0 else ((x, y + w), (x, y))
            a, b = (b, a) if sd < 0 else (a, b)
            foot = lot(np.array([x, y + w]) if sd > 0 else np.array([x, y]), np.array([x, y]) if sd > 0 else np.array([x, y + w]), dp)
            if sd > 0: foot = lot(np.array([x, y + w]), np.array([x, y]), dp)
            n_ = np.array([-sd, 0.])
            e0 = foot[1] - foot[0]; nn = np.array([e0[1], -e0[0]])
            if nn @ n_ < 0: foot = lot(foot[1], foot[0], dp)
            building(S, foot, h, st, np.random.default_rng(int(r.integers(1 << 30))), eye, f, lit, pick(r, ['#7d776e', '#5c6076', '#938c84', '#6a5e5a', '#4c526a', '#8a8f9c']), shop=r.uniform(3.8, 4.4), blades=int(r.integers(0, 3)) * signs, bill=.2, grp=g if y < 60 else 2, glow=glow)
            S.light((sd * (wid / 2 + 1.2), y + w / 2, 3.2), pick(r, ['#ffd09a', '#ffe6c4', '#bfe0ff', '#ffa890']), 17, 4.2, spill, g if y < 60 else 2)
            y += w + r.uniform(0, .3)
    # clutter: poles, wires, vending machines, bicycles, banners, A-boards
    an = []
    for k in range(9):
        sd = -1 if k % 2 else 1; y = 6 + k * 17 + r.uniform(-2, 2); an.append(pole(S, (sd * (wid / 2 + .5), y), 10.5, r, .15, ((1, 0),), '#716f74'))
    for i in range(len(an) - 2):
        for j in range(0, 8, 1): S.wire(an[i][j], an[i + 2][j], r.uniform(.4, .9), .03, '#15151d', min_px=1.)
        if i % 2 == 0: S.wire(an[i][3], an[i + 1][2], .7, .03, '#15151d')
    for k in range(7):
        sd = r.choice([-1, 1]); y = r.uniform(4, 70); x = sd * (wid / 2 + 2.15); c = pick(r, ['#c8202e', '#e6e8ec', '#2a5aa8']); S.boxes([x, y, 1.07], [.78, 1.1, 1.84], 0, c); W = Wall.frame(S, (x - sd * .40, y + sd * .5, .25), (0, -sd, 0), (0, 0, 1), 1.0, 1.7, 5 if sd < 0 else 3)
        W.rect(.08, .92, .7, 1.6, .01, '#fff4e0', em=.8); u_ = np.tile(np.arange(5), 3) * .16 + .12; v_ = np.repeat(np.arange(3), 5) * .28 + .78; W.rect(u_, u_ + .10, v_, v_ + .2, .02, np.stack([C(pick(r, ['#c8202e', '#2a7a4a', '#e8a02a', '#2a4a9a', '#f0f0f0'])) for _ in u_]), em=.7); S.light((x - sd * 1.2, y, 1.3), '#fff0dc', 6, 1.6, 1.6, 3)
    for k in range(6): sd = r.choice([-1, 1]); y = r.uniform(8, 60); S.boxes([sd * (wid / 2 + 1.5), y, .55], [.55, .06, .8], 0, pick(r, ['#e8e0d0', '#2a2c38', '#b8242e']), em=.25)
    for k in range(16):
        sd = r.choice([-1, 1]); y = r.uniform(6, 90); x = sd * (wid / 2 + 2.35); z = r.uniform(2.9, 3.3); S.cyl((x - sd * .45, y), .19, z, z + .52, pick(r, ['#e0402a', '#e8a860', '#e0402a']), 9, em=.95, grp=3)
    if lamps:
        for k in range(8): sd = -1 if k % 2 else 1; lamp(S, (sd * (wid / 2 + .4), 14 + k * 20), 6.8, (-sd * 1.2, 0), '#ffe2b4', 0, .15, 1.6, 20)
    return S, r


@painter('street', 3)
def street(w=2400, h=1350, seed=0, eye=(-1.2, -3., 1.55), target=(.5, 30., 4.6), focal=1180., lit=.52, wet=1., people=80, night_glow='#80507c', paint=True, ss=2, car_=True, wid=11.):
    """night shopping street at eye level, wet asphalt, shops both sides.  masks: sky, ground"""
    cam = Camera3D(eye, target, focal, w, h); S, r = street_scene(seed, cam, wid, lit=lit)
    if people:
        y = r.uniform(13, 95, people); sd = r.choice([-1, 1], people); x = sd * (wid / 2 + r.uniform(.3, 2.2, people)); k = r.random(people) < .25; x[k] = r.uniform(-wid / 2 + .5, wid / 2 - .5, k.sum()); crowd(S, np.c_[x, y], cam, r, np.where(np.abs(x) > wid / 2, .15, 0.), umb=.25)
    if car_: car(S, -2.6, 34, -math.pi / 2, r, 'taxi'); car(S, 2.4, 52, math.pi / 2, r, 'black'); car(S, -2.6, 70, -math.pi / 2, r, 'white'); car(S, 2.5, 24, math.pi / 2, r, 'van')
    hz = Wd.horizon_y(cam); sky = SK.night(h, w, hz + 20, seed + 2, glow=night_glow, gx=.5)
    nk = dict(amb=('#1c2240', '#14131f'), street=('#6a4a38', 7.), fills=[((-.9, -.2, .1), '#1c1230'), ((.9, -.2, .1), '#0e2036')], fog=dict(col='#4a3a66', col2='#7a5478', dir=(0, 1, 0), dens=.0085, start=26, hf=.03, floor=.5, keep=.35, pw=6.))
    baked = S._bake(cam); res = S.render(cam, ss=ss, baked=baked, **nk)
    X = cam.eye + cam.rays() * res['z'][..., None]; g = ((np.abs(X[..., 2]) < .20) & (res['z'] < FAR * .5)).astype(f32)
    pud = smooth(-.5, .8, fbm(h, w, (200, 70, 24), (1, .6, .3), seed + 41, aspect=3.2)); wm = cv2.GaussianBlur(g, (0, 0), 1.) * (.25 + .65 * pud) * wet
    img, _ = Wd.wet_reflect(S, cam, res, sky, wm, baked=baked, strength=.80, vblur=20, ripple=3.2, seed=seed + 3, **nk)
    geo = (res['z'] < FAR * .5).astype(f32); img = img * (1 + (Wd.occlusion(res['z'], .42) - 1) * geo)[..., None] + sky * (1 - res['a'][..., None])
    if paint:
        img = Wd.wobble(img, .55, 34, seed + 12); img = Wd.gouache(img, res['lab'], res['ang'] + fbm(h, w, (160, 60), (1, .5), seed + 8) * .10, seed=seed + 4, sizes=(20, 10, 5), op=(.6, .55, .5), val=(.050, .042, .034), hue=.016, thr=(.016, .028, .05), mask=geo)
        e = np.maximum(img - .66, 0); img = Wd.lamp_glow(img, e, ((6, .22), (24, .16))); img = Wd.air_glow(img, res['z'], .5, (60, 150), .26, (8, 90)); img = Wd.tooth_pass(img, seed + 6)
    return Plate(img, res['z'], cam, dict(sky=1 - res['a'], ground=g))


# ------------------------------------------------------------------------------------------------ the city from the air
def air_city(S, r, eye, f, cam, rng=(0, 1500), x_span=1.15, lit=.5, grp0=2, skip=None):
    """irregular street grid with blocks of buildings; streets glow, cars are drops of light"""
    bx, by = 74., 62.; ny = int(rng[1] / by) + 2; far_ = []
    for j in range(ny):
        yc = rng[0] + j * by; half = max(400, (yc + 260) * x_span); nx = int(half / bx) + 1
        for i in range(-nx, nx + 1):
            xc = i * bx + (j % 2) * 13. + 9 * math.sin(j * 1.7 + i); d = math.hypot(xc - eye[0], yc - eye[1])
            hole = skip is not None and skip(xc, yc)
            # glowing street along x and y
            c = C(pick(r, ['#ffb468', '#ffcf94', '#ff9a58'])); q = r.uniform(.28, .6)
            S.quads(np.array([[xc - bx / 2, yc - by / 2, .05], [xc + bx / 2, yc - by / 2, .05], [xc + bx / 2, yc - by / 2 + 9, .05], [xc - bx / 2, yc - by / 2 + 9, .05]])[None], c * q, em=.75, grp=grp0, snow=0)
            S.quads(np.array([[xc - bx / 2, yc - by / 2, .06], [xc - bx / 2 + 8, yc - by / 2, .06], [xc - bx / 2 + 8, yc + by / 2, .06], [xc - bx / 2, yc + by / 2, .06]])[None], c * q * .8, em=.7, grp=grp0, snow=0)
            if d < 700:
                m = int(r.integers(2, 7)); px = xc + r.uniform(-.5, .5, m) * bx; py = yc - by / 2 + r.choice([2.6, 6.2], m); cc = np.stack([C('#fff2d8') if t < 4 else C('#ff3028') for t in (py - (yc - by / 2))]); S.boxes(np.c_[px, py, np.full(m, .9)], [2.6, 1.2, .8], 0, cc, em=1.7, grp=grp0, snow=0)
            x0, y0 = xc - bx / 2 + 8.6, yc - by / 2 + 9.6; x = x0
            while x < xc + bx / 2 - 8:
                w = min(r.uniform(11, 27), xc + bx / 2 - .6 - x); 
                if w < 7: break
                y = y0
                while y < yc + by / 2 - 8:
                    dp = min(r.uniform(13, 30), yc + by / 2 - .6 - y)
                    if dp < 8: break
                    h = min(8 + r.gamma(1.7, 11), 78); h = h * (2.0 if (r.random() < .07 and min(w, dp) > 14) else 1.); h = min(h, 4 * min(w, dp))
                    st = pick(r, ['office', 'grid', 'curtain', 'resid', 'tenant', 'grid', 'office']); dist = math.hypot(x - eye[0], y - eye[1])
                    foot = np.array([(x, y), (x + w - .8, y), (x + w - .8, y + dp - .8), (x, y + dp - .8)])
                    if dist > 620:
                        far_.append((x + (w - .8) / 2, y + (dp - .8) / 2, w - .8, dp - .8, h)); y += dp; continue
                    if hole and (skip(x + w / 2, y + dp / 2, 24) or skip(x, y, 14) or skip(x + w, y + dp, 14) or skip(x + w, y, 14) or skip(x, y + dp, 14)): y += dp; continue
                    building(S, foot, h, st, np.random.default_rng(int(r.integers(1 << 30))), eye, f, lit * r.uniform(.35, 1.25), pick(r, [P1, P2, P4, P5, P6, P7, '#7c766e']), 0, dist < 650, grp0, bill=.12 if dist < 500 else 0., detmax=0 if dist > 420 else 1 if dist > 190 else 2)
                    y += dp
                x += w

    if far_:                                                            # far blocks in one go: box plus its own lit panes
        F = np.array(far_); n = len(F); pal = np.stack([C(pick(r, [P1, P2, P4, P5, P6, P7, '#7c766e'])) for _ in range(n)]); S.boxes(np.c_[F[:, 0], F[:, 1], F[:, 4] / 2], np.c_[F[:, 2], F[:, 3], F[:, 4]], 0, pal, faces='xXyZ', tex=0)
        m = 34; k = np.repeat(np.arange(n), m); f_ = r.integers(0, 1 << 30, len(k)); nf = np.maximum(1, (F[k, 4] / 3.8).astype(int)); fl = f_ % nf; u = r.uniform(-.46, .4, len(k)); wd = r.uniform(.04, .24, len(k)) * F[k, 2]; side = r.random(len(k)) < .3; ok = r.random(len(k)) < lit * 1.1; k, fl, u, wd, side = k[ok], fl[ok], u[ok], wd[ok], side[ok]; z0 = fl * 3.8 + 1.1; sx = np.where(F[k, 0] > eye[0], -1., 1.)
        a = np.where(side[:, None], np.c_[F[k, 0] + sx * (F[k, 2] / 2 + .05), F[k, 1] + u * F[k, 3]], np.c_[F[k, 0] + u * F[k, 2], F[k, 1] - F[k, 3] / 2 - .05]); b = a + np.where(side[:, None], np.c_[np.zeros(len(k)), np.minimum(wd, .4 * F[k, 3])], np.c_[np.minimum(wd, (.5 - u) * F[k, 2]), np.zeros(len(k))]); S.quads(np.stack([np.c_[a, z0], np.c_[b, z0], np.c_[b, z0 + 1.9], np.c_[a, z0 + 1.9]], 1), wcols(r, len(k)) * r.uniform(.5, 1.05, (len(k), 1)).astype(f32), em=.85, grp=grp0, snow=0, lab=S.newlab(len(k)))


@painter('skyline', 4)
def skyline(w=2400, h=1350, seed=0, eye=(0, -260., 150.), target=(0, 300., 96.), focal=1250., lit=.5, paint=True, ss=2, reach=1300, near=True, fog='#3a3a78'):
    """night city from the air.  layers: 'near' (tall towers close to camera, for parallax).  masks: sky"""
    cam = Camera3D(eye, target, focal, w, h); r = np.random.default_rng(seed + 61); S = Scene(seed + 62); S.groups = 10
    S.ground(-16000, 16000, -300, 22000, 0., '#23263a', nx=24, ny=60, pw=2.4, tex=0)
    nf_ = 5200; yy_ = 1150 + 4200 * r.random(nf_) ** 1.6; xx_ = r.uniform(-1, 1, nf_) * (yy_ + 500) * 1.15; hh_ = 6 + r.gamma(1.6, 10, nf_); ww_ = r.uniform(14, 44, nf_); S.boxes(np.c_[xx_, yy_, hh_ / 2], np.c_[ww_, ww_, hh_], 0, '#3a3a5c', faces='yZ', tex=0)
    nl_ = 9000; yy_ = 1050 + 5200 * r.random(nl_) ** 1.5; xx_ = r.uniform(-1, 1, nl_) * (yy_ + 500) * 1.15; S.boxes(np.c_[xx_, yy_, r.uniform(3, 34, nl_)], np.c_[r.uniform(2.5, 7, nl_), np.ones(nl_), r.uniform(1.6, 3.4, nl_)], 0, np.stack([C(pick(r, ['#ffd9a0', '#ffc27a', '#e8f0ff', '#ffb060', '#ff6a4a'])) for _ in range(nl_)]) * r.uniform(.5, 1.3, (nl_, 1)).astype(f32), faces='y', em=1.1, grp=2, snow=0)
    # an expressway swings through, a dark river beside it
    t = np.linspace(0, 1, 90); rx = -260 + 520 * t + 90 * np.sin(t * 5.2); ry = -60 + 1300 * t ** 1.25
    road = np.c_[rx, ry]
    def near_road(x, y, d=75): return np.min(np.hypot(rx - x, ry - y)) < d
    air_city(S, r, eye, cam.f, cam, (-120, reach), 1.15, lit, 2, near_road)
    for a, b in zip(road[:-1], road[1:]):
        d = (b - a) / np.hypot(*(b - a)); n = np.array([-d[1], d[0]])
        for off, wd_, z in ((0, 18, 13.),):
            p = [np.r_[a + n * wd_ / 2, z], np.r_[a - n * wd_ / 2, z], np.r_[b - n * wd_ / 2, z], np.r_[b + n * wd_ / 2, z]]; S.quads(np.array(p)[None], '#4a4658', em=.10, tex=0, d=(*d, 0)); S.quads(np.array([p[1], p[2], p[2] - [0, 0, 2.2], p[1] - [0, 0, 2.2]])[None], '#6a6678'); S.quads(np.array([p[0], p[3], p[3] - [0, 0, 2.2], p[0] - [0, 0, 2.2]])[None], '#6a6678')
        L = np.hypot(*(b - a)); m = int(L / 9)
        for k in range(m):
            q = a + d * (k + r.random()) * 9; s_ = r.choice([-1, 1]); u = q + n * s_ * r.uniform(2.5, 7); S.boxes([*u, 14.], [r.uniform(4, 9), 1.5, .8], math.atan2(d[1], d[0]), '#fff0d0' if s_ > 0 else '#ff2a22', em=1.9, grp=7, snow=0)
        S.boxes(np.array([[*(a + n * 10), 17.], [*(a - n * 10), 17.]]), [.9, .9, .5], 0, '#ffcf8a', em=2.0, grp=2, snow=0)
        S.boxes([*((a + b) / 2), 6.], [3, 3, 12], 0, '#5a5668')
    # landmark lattice tower, warm lit
    c0 = np.array([210., 560.]); Hh = 250.; zz = np.linspace(0, 1, 15); wd_ = lambda t: 40 * (1 - t) ** 1.9 + 2.2
    for k in range(14):
        for cx_, cy_ in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            a = np.array([c0[0] + cx_ * wd_(zz[k]), c0[1] + cy_ * wd_(zz[k]), zz[k] * Hh]); b = np.array([c0[0] + cx_ * wd_(zz[k + 1]), c0[1] + cy_ * wd_(zz[k + 1]), zz[k + 1] * Hh]); g = C(mix(C('#ffb060'), C('#ff5a2a'), zz[k])); S.seg([a], [b], 2.4 * (1 - zz[k]) + .9, g, em=.9, grp=3, min_px=1.4)
            for sx_, sy_ in ((-cx_, cy_), (cx_, -cy_)): S.seg([a], [[c0[0] + sx_ * wd_(zz[k + 1]), c0[1] + sy_ * wd_(zz[k + 1]), zz[k + 1] * Hh]], .8, g * .8, em=.75, grp=3, min_px=1.)
    for zq, sz in ((.36, 1.15), (.72, 1.5)): q = wd_(zq) * 2 * sz; S.boxes([*c0, zq * Hh], [q, q, 9], 0, '#fff0d0', em=1.0, grp=3)
    S.seg([[*c0, Hh]], [[*c0, Hh + 36]], .8, '#ff5a3a', em=1.2, grp=3, min_px=1.4)
    hz = Wd.horizon_y(cam); sky = SK.night(h, w, hz + 14, seed + 7, glow='#8a5a86', glow2='#3f7f9e', gx=.6, clouds_amt=1.1, stars=160)
    nk = dict(amb=('#1a2148', '#10111f'), street=('#b0683a', 14.), fills=[((-.8, -.4, .1), '#24143a'), ((.8, -.4, .1), '#0e2240'), ((0, -1, .1), '#0c0f22')], fog=dict(col=fog, col2='#8a5a86', dir=(.2, 1, 0), dens=.00125, start=200, hf=.004, floor=.8, keep=.5, pw=10.))
    res = S.render(cam, ss=ss, **nk); geo = (res['z'] < FAR * .5).astype(f32)
    img = res['img'] * (1 + (Wd.occlusion(res['z'], .42) - 1) * geo)[..., None] + sky * (1 - res['a'][..., None])
    lay = {}
    if paint:
        img = Wd.wobble(img, .45, 28, seed + 12); img = Wd.gouache(img, res['lab'], res['ang'], seed=seed + 4, sizes=(16, 8, 4), op=(.6, .55, .5), val=(.05, .042, .034), hue=.016, thr=(.016, .03, .05), mask=geo)
        e = np.maximum(img - .58, 0); img = Wd.lamp_glow(img, e * 1.3, ((4, .30), (16, .26))); img = Wd.air_glow(img, res['z'], .40, (40, 120), .55, (150, 900)); img = Wd.tooth_pass(img, seed + 6)
    if near:
        S2 = Scene(seed + 81); r2 = np.random.default_rng(seed + 82); S2.groups = 10
        for (x, y, ww, dd, hh, st, pl) in ((-150, -70, 44, 40, 142, 'office', P5), (-250, 40, 50, 40, 168, 'fins', P2), (178, -60, 46, 40, 126, 'curtain', P5), (268, 60, 52, 44, 176, 'office', P4), (-96, 10, 30, 30, 84, 'grid', P6), (110, 40, 34, 30, 96, 'tenant', P3)):
            q = np.array([(-ww, -dd), (ww, -dd), (ww, dd), (-ww, dd)]) / 2 + [x, y]; building(S2, q, hh, st, np.random.default_rng(int(abs(x) * 3 + hh)), eye, cam.f, lit, pl, 0, True, 3)
        r3 = S2.render(cam, ss=ss, **nk); g3 = (r3['z'] < FAR * .5).astype(f32); im3 = r3['img'] * (1 + (Wd.occlusion(r3['z'], .35) - 1) * g3)[..., None]; a3 = r3['a']
        if paint:
            st_ = np.where(a3[..., None] > .01, im3 / np.maximum(a3[..., None], .01), 0); st_ = Wd.gouache(st_, r3['lab'], r3['ang'], seed=seed + 14, sizes=(20, 10, 5), op=(.6, .55, .5), val=(.05, .042, .034), hue=.016, thr=(.016, .03, .05), mask=g3); e = np.maximum(st_ - .6, 0); st_ = Wd.lamp_glow(st_, e, ((6, .3), (22, .2))); im3 = st_ * a3[..., None]
        lay['near'] = np.dstack([im3, a3]).astype(f32)
    return Plate(img, res['z'], cam, dict(sky=1 - res['a']), lay)
