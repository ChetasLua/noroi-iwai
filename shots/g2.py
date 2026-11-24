"""g2 SNOW: one space.  The hands frame sits D px under the faces frame; arms run from the hands to real shoulders.  A flake melts on Yuji's open hand; Megumi's hand, then Nobara's, take his wrist; tilt up the arms to three tired faces; the sun comes in."""
import numpy as np, cv2, math, os, pickle, hashlib, glob
from jj import W, H, comp, kit, hands as HD, hands_sleeve as HS, fx, world
from jj.paint import rgb, smooth, fbm, f32, grid, ramp, lum
from jj.ink import Cel, spl, noise1
E = HD.euler; GEO = None
CAM = (708, 600, 300, 5.2); L0 = (-.30, .85, .45); D_ = 980
SHO = dict(y=(690., -78.), m=(292., -112.), n=(1506., -118.))                    # shoulders in hands-frame px (they match the busts above)


# ---------------------------------------------------------------- geometry
PALM = HD.pose((8, 13, 7, -3), (10, 15, 8, 0), (13, 17, 9, 3), (15, 19, 10, 8), thumb=(52, 24, 30, 8, 6), cup=.30)


def _yuji(curl=0., sq=0.):
    """palm to us, seen a little from above, fingers soft: the pads show, never a nail"""
    q = {k: (list(v) if not np.isscalar(v) else v) for k, v in PALM.items()}; curl = min(curl, 16.) * .55
    for f, a in zip(HD.FING, (.8, .9, 1., 1.1)): q[f][0] += curl * a * .7; q[f][1] += curl * a * .8; q[f][2] += curl * a * .35
    q['thumb'][1] += curl * .4; q['thumb'][4] += curl * .5; q['wrist'] = (-40 - 5 * sq, -4)
    return HD.Hand(q, 'R', E(174, 8, 170), pos=(0, 0, 0), pivot=(0, .6, 0), bld=HD.HEAVY, nails=None)


def _grip(y, who, t, d, lift):
    """t 0 open .. 1 closed (1.1 squeeze); d = how far back along its arm, lift off the skin"""
    g0 = HD.pose((6, 20, 12, -5), (4, 18, 10, 0), (6, 20, 12, 4), (8, 24, 14, 9), thumb=(58, 34, 10, 4, 4)); p = HD.blend(g0, 'grip', t, drag=.18, over=0.) if t <= 1 else HD.lerp_pose('grip', HD.pose((58, 70, 42, 0), (60, 72, 44, 0), (62, 74, 44, 0), (64, 74, 44, 2), thumb=(38, 66, -42, 24, 36), cup=.18), (t - 1) * 5)
    if who == 'm': h = HD.grip_wrist(y, HD.SLIM, dict(p, wrist=(4, -6)), 'R', at=.30, turn=176, thumb='elbow', skew=26, lift=lift, arm=1.75)
    else: h = HD.grip_wrist(y, HD.FEMALE, dict(p, wrist=(4, -4)), 'L', at=.88, turn=184, thumb='elbow', skew=24, lift=lift, arm=1.75)
    h.o = h.o - (h.A @ np.array([0, 1., 0])) * d; return h


def build(curl, sq, m=None, n=None):
    y = _yuji(curl, sq); hs = [y]; who = ['y']
    if m: hs.append(_grip(y, 'm', *m)); who.append('m')
    if n: hs.append(_grip(y, 'n', *n)); who.append('n')
    cam = HD.Cam(*CAM); g = HD.render(hs, cam, L0, res=640, clip=(-80, -420, W + 80, H + 60), thr=.12, cross=True)
    scr = []
    r = np.random.default_rng(8)
    for (a, b, w) in (((.06, .56), (-.10, .70), 1.), ((.12, .50), (-.02, .63), .8), ((-.16, .40), (-.26, .52), .9), ((.20, .30), (.10, .40), .7), ((-.02, .78), (-.10, .86), .7), ((-.24, .62), (-.31, .70), .6)):
        for P in HD.surface_curve(g, 0, [a, ((a[0] + b[0]) / 2 + r.normal(0, .015), (a[1] + b[1]) / 2 + r.normal(0, .015)), b]): scr.append((P, w))
    soot = [np.vstack(HD.surface_curve(g, 0, q)) for q in ([(-.30, .34), (-.20, .25), (-.06, .22), (.06, .27), (.02, .33), (-.12, .33), (-.22, .42), (-.30, .34)], [(.22, .62), (.28, .52), (.27, .44), (.20, .52), (.22, .62)]) if HD.surface_curve(g, 0, q)]
    spot = HD.surface_curve(g, 0, [(-.03, .50), (-.02, .51), (-.01, .52)]); spot = spot[0][1] if spot else g.pts[0]['palm']
    ring = HD.surface_curve(g, 0, [(-.03 + .10 * math.cos(t), .52 + .10 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 14)])
    ring = np.vstack(ring) if ring else None
    sl = []
    for i, w in enumerate(who):
        if w == 'y': sl.append(HS.sleeve(g, i, y0=.62, y1=2.05, seed=21, light=L0, torn=.13, button=False, ease=1.18, gap=.06, folds=[(.86, 1.3, .12, .07, .06), (1.10, -1.2, .16, .075, .07), (1.42, 1.1, .2, .07, .09), (1.8, -1.2, .3, .06, .1)]))
        else: sl.append(HS.sleeve(g, i, y0=.17, y1=2.05, seed=31 + i, light=L0, seam=.9 if w == 'm' else -.9, button=True, folds=[(.42, -1.4, .14, .052, .05), (.82, 1.6, .20, .068, .07), (1.30, -.8, .26, .055, .09)] if w == 'm' else [(.46, 1.5, .16, .055, .05), (.74, -1.9, .10, .045, .05), (1.22, .6, .26, .070, .09), (1.62, 2.4, .2, .05, .08)]))
        h = hs[i]; Wf = h.sk.fr['wrist']; sl[-1]['elbow'] = [cam.px(h.world(Wf.pt((dx, 1.92, 0)))) for dx in (-.40, .40)], cam.px(h.world(Wf.pt((0, 1.95, 0)))), cam.px(h.world(Wf.pt((0, 1.2, 0))))
    q = HD.lite(g, dict(s=HD.big(HD.rim_polys(g, (.85, .45, -.4), .42, .36), CAM[2], .006), u=HD.big(HD.rim_polys(g, (-.5, .8, -.2), .62, .42), CAM[2], .012)), dict(d=HD.big(HD.depth_polys(g, L0, -.36), CAM[2], .012)), dict(sl=sl, who=who, scr=scr, soot=soot, spot=spot, ring=ring))
    q.creases = [c for c in q.creases if not (c['kind'] in ('flex', 'wrinkle') and np.hypot(*np.diff(c['P'], axis=0).T).sum() < .13 * CAM[2])]; return q


KEYS = {0: (0., 0.), 22: (6., 0.), 23: (13., 0.), 25: (8., 0.), 28: (4., 0.),
        33: (4., 0., (0., 1.5, .42)), 34: (4., 0., (.10, .95, .32)), 35: (3., 0., (.35, .38, .20)), 36: (10., .8, (.92, .0, .03)), 37: (14., 1., (1.12, 0., 0.)), 39: (9., .5, (1.03, 0., 0.)), 41: (8., .4, (1., 0., 0.))}
_m = (1., 0., 0.)
KEYS.update({42: (8., .4, _m, (0., 1.5, .42)), 43: (8., .4, _m, (.10, .95, .32)), 44: (8., .4, _m, (.40, .36, .18)), 45: (12., .9, _m, (.92, 0., .03)), 46: (15., 1., (1.06, 0., 0.), (1.12, 0., 0.)), 48: (11., .6, _m, (1.03, 0., 0.)), 50: (10., .5, _m, (1., 0., 0.))})


def _job(k): cv2.setNumThreads(1); return k, build(*KEYS[k])


def _pool(fn, items, jobs=2):
    import multiprocessing as mp
    try:
        with mp.get_context('fork').Pool(jobs) as P: return P.map(fn, items, chunksize=1)
    except AssertionError: return [fn(i) for i in items]


def geometry():
    global GEO
    if GEO is not None: return GEO
    key = hashlib.md5((open(HD.__file__).read() + open(HS.__file__).read() + open(__file__).read().split('# ---------------------------------------------------------------- people')[0]).encode()).hexdigest()[:10]
    path = f'cache/g2_geo_{key}.pkl'
    if os.path.exists(path): GEO = pickle.load(open(path, 'rb')); return GEO
    GEO = dict(_pool(_job, list(KEYS)))
    for o in glob.glob('cache/g2_geo_*.pkl'): os.remove(o)
    pickle.dump(GEO, open(path, 'wb')); return GEO


# ---------------------------------------------------------------- people
PV = 10
FR = [61, 64, 68, 72, 75, 78, 80, 83, 87, 89, 92, 96, 98, 101, 104]             # redraws: wind, and one beat each at 78 (Nobara), 87 (Megumi), 96 (Yuji)
SUN = (2010, 400)


def lights():
    """dawn behind them, low on the right: faces in soft warm shadow with bounce off the snow, a hard rim on every right edge (registered in this process only)"""
    from chars import mn, megumi as MG, nobara as NB
    from chars.yuji import Yuji
    mn.LIGHT['g2b'] = dict(mn.LIGHT['sunset'], skin='#dca58c', skin2='#b0717a', skin3='#80495c', skinhi='#f0c6a6', blush='#d9786c', white='#f1dccb', white2='#b3909a', line='#2c1420', lip='#a85a5e', cloth='#2b2438', cloth_hi='#6e5570', cloth_dk='#17121e', key=(.62, .55),
                           rim=('#fff0c2', -6, 2, 1.), para=[(0, '#ffeeda'), (.5, '#ffffff'), (1, '#c3abc4')])
    MG.HAIR['g2b'] = dict(MG.HAIR['sunset'], hi='#b9776a'); NB.HAIR['g2b'] = dict(NB.HAIR['sunset'])
    return Yuji.light('sunset', skin=('#e0ab8c', '#b4757a', '#824c5e'), amt=-.55, blush='#dd7c66', rim=[dict(d=(-13, 4), col='#fff2c6', amt=1., over=True)], glint='#fff8e6', white=('#f4dfcc', '#b8949c'),
                      para=[dict(p0=(1.3, -.6), p1=(-1.2, .8), stops=[(0, '#fff0dc'), (.45, '#ffffff'), (1, '#b8a2c2')], amt=.8)])


def person(a):
    cv2.setNumThreads(1); who, f, li = a; path = f'cache/g2_p{PV}_{who}_{f}_{li}.npz'
    if os.path.exists(path): return path
    tall = f == 'T'; ff = 61 if tall else f; YL = lights(); wind = (.22 + .10 * math.sin(ff * .5), .06); br = math.sin((ff - 54) / 24. * 2 * math.pi * .55); h = 1700 if tall else H
    if who == 'y':
        from chars.yuji import Yuji, YHead
        s = float(comp.ease(np.clip((ff - 80) / 10., 0, 1))); e = Yuji.blend(Yuji.expr('tired', slim=1., glint=1.15), Yuji.expr('gentle', glint=1.2), .45 + .55 * s); up = float(comp.ease(np.clip((ff - 94) / 6., 0, 1)))
        e = dict(e, gx=.030 if ff >= 72 else .0)
        if ff == 72: e = Yuji.lids(e, .40)
        if ff == 96: e = Yuji.lids(e, .60)
        if ff >= 98: e = Yuji.lids(Yuji.blend(e, Yuji.expr('smile', slim=1., naso=0., pouch=0., cheek=.2), .40), .97)
        lay = Yuji.bust(YHead().pose(934, 572 - 3 * br, 212, .34 + .05 * up, -.03 + .12 * up, .05 - .03 * up), e, 'snow' if li == 'snow' else YL, ff, wind, 3, damage=.3, breath=.5 + .5 * br, h=h)
    elif who == 'm':
        from chars import megumi as MG
        S_ = MG.EXPR['smile']; D0 = MG.EXPR['deadpan']
        sm = dict(S_, lid=.46, gaze=(.012, -.030), spark=1.35, mouth=dict(S_['mouth'], pts=[(-.108, -.640), (-.052, -.667), (.0, -.673), (.056, -.662), (.114, -.632)], w=1.2))
        e = dict(D0, lid=.70, spark=1.3, gaze=(.010, -.008)) if ff < 87 else dict(D0, lid=.22, gaze=(.012, -.03)) if ff == 87 else sm
        dn = float(comp.ease(np.clip((ff - 85) / 5., 0, 1))); lay = MG.Megumi.bust(572, 452 - 2.5 * br + 5 * dn, 204, .44, -.06 - .13 * dn, .08 + .03 * dn, expr=e, light='snow' if li == 'snow' else 'g2b', f=ff, wind=wind, seed=4, torso=dict(roll=.04), h=h)
    else:
        from chars import nobara as NB
        C_ = NB.EXPR['calm']; T_ = NB.EXPR['teary']; e = dict(C_, lid=.86, spark=1.3, gaze=(.02, -.014)) if ff < 78 else dict(C_, lid=.30, gaze=(-.03, -.01)) if ff == 78 else dict(T_, spark=1.45, gaze=(-.034, -.012))
        tn = float(comp.ease(np.clip((ff - 77) / 4., 0, 1))); lay = NB.Nobara.bust(1346 - 6 * tn, 470 - 2.5 * br, 200, -.34 - .10 * tn, -.02, -.07 - .03 * tn, expr=e, light='snow' if li == 'snow' else 'g2b', f=ff, wind=wind, seed=5, patch=True, torso=dict(roll=-.04), h=h)
    if tall:                                                   # coat and trousers run on out of the bottom
        op = lay[..., 3] > .6; last = np.where(op.any(0), h - 1 - np.argmax(op[::-1], 0), -1); yy = np.arange(h)[:, None]; src = np.clip(last - 26, 0, h - 1); colr = lay[src, np.arange(W)]; colr = cv2.GaussianBlur(colr[None], (0, 0), 9)[0]
        m = (yy > (last - 26)[None]) & (last > 1150)[None] & (colr[None, :, 3] > .5); lay = np.where(m[..., None], (colr / np.maximum(colr[:, 3:4], 1e-3))[None] * np.array([.86, .86, .90, 1.], f32), lay).astype(f32)
    al = lay[..., 3]; ys = np.where(al.max(1) > .004)[0]; xs = np.where(al.max(0) > .004)[0]; y0, y1, x0, x1 = ys[0], ys[-1] + 1, xs[0], xs[-1] + 1
    np.savez(path, a=np.clip(lay[y0:y1, x0:x1] * 255 + .5, 0, 255).astype(np.uint8), o=np.array([x0, y0])); return path


def jobs():
    J = [(w, 'T', 'snow') for w in 'ymn']
    for who in 'ymn':
        for f in FR:
            if f < 90: J.append((who, f, 'snow'))
            if f >= 72: J.append((who, f, 'warm'))
    return J


# --- whole bodies behind the hands: jj.figure stills at the scale of the busts, half size, out of focus
MW = {26: 0, 28: 1, 30: 2, 32: 3, 34: 4, 36: 5, 38: 6}; NW = {35: 0, 37: 1, 39: 2, 41: 3, 43: 4, 45: 5, 47: 6}
STEP = dict(y=[(0, 'y0')], m=[(k, f'm{v}') for k, v in MW.items()], n=[(k, f'n{v}') for k, v in NW.items()])


def fcam(w=W // 2, h=H // 2, f=1000.):
    from jj.cam import Camera3D
    return Camera3D((0., -1.30, 1.50), (0., .05, 1.04), focal=f, w=w, h=h)


def still(k):
    cv2.setNumThreads(1); path = f'cache/g2_f{PV}_{k}.npy'
    if os.path.exists(path): return path
    from jj.figure import Fig, P as FP, walk, mix_pose
    BACK = (-.10, -.42, -.70)
    if k == 'y0': F = Fig('yuji').pose(FP('stand', hR=BACK, cst=(6, 0, -4), hed=(10, 0, 0))).place(0, 0, 0, math.pi)
    else:
        who, n = k[0], int(k[1]); sg = -1 if who == 'm' else 1; x1 = sg * (.235 if who == 'm' else .225); x0 = sg * 1.22
        if n < 5:                                              # two real steps in ten frames: .2 m a drawing, the body drops at each contact
            u = n / 4.; x = x0 + (x1 + sg * .06 - x0) * u; p = walk(u * .98 + (.0 if who == 'm' else .5), 1.2)
        else: x = (x1 - sg * .025) if n == 5 else x1; p = mix_pose(walk(.98 + (.0 if who == 'm' else .5), 1.), FP('contra'), .6 if n == 5 else 1.)
        p = dict(p, **({'hR': BACK} if who == 'm' else {'hL': BACK})); F = Fig('megumi' if who == 'm' else 'nobara').pose(p).place(x, .13, 0, math.pi - sg * .30 * (1 if n < 5 else .6))
    L = F.draw(fcam(), 10, light=(-.3, .85, .45), pal=dict(mul='#c9d0ee')); L = comp.blur(L, 6.0); np.save(path, np.clip(L * 255 + .5, 0, 255).astype(np.uint8)); return path


STILLS = ['y0'] + [f'{w}{n}' for w in 'mn' for n in range(7)]


def plates():
    path = lambda k: f'cache/g2_plate_{PV}_{k}.npy'; KS = ('cool', 'warm', 'ground')
    if all(os.path.exists(path(k)) for k in KS): return {k: np.load(path(k), mmap_mode='r') for k in KS}
    P = world.plate('ruin', mode='snow', view='low', w=2400, h=1350); im = P.img; sky = np.clip(P.mask['sky'], 0, 1)[..., None] if 'sky' in P.mask else np.zeros(im.shape[:2] + (1,), f32)
    im = cv2.resize(im, (2080, 1170), interpolation=cv2.INTER_AREA); sky = cv2.resize(sky, (2080, 1170))[..., None]
    h, w = im.shape[:2]; x, y = grid(h, w)
    cool = comp.blur(im, 4.5) * .96 + .02
    # dawn arrives from the right: warm light on the faces that look right, gold sky low right
    d = np.hypot((x - 2050) / 1.25, y - 470); g_ = (smooth(1500, 40, d) ** 1.3)[..., None]
    warm = im * (np.array([1.12, .96, .86], f32) * (1 - .45 * g_) + np.array([1.45, 1.10, .78], f32) * .45 * g_ * 1.2) + g_ * rgb('#ff9f4e') * .42
    sk = ramp(np.clip(y / 620 + .25 * (2080 - x) / 2080 - .05, 0, 1), [(0, '#7f93c4'), (.35, '#c9b3bf'), (.62, '#ffd2a0'), (1, '#fff0c9')]); warm = warm * (1 - sky * .72) + (sk * .72 + g_ * rgb('#ffcf8a') * .35) * sky
    warm = comp.blur(warm, 4.5) + (smooth(300, 10, d) ** 2)[..., None] * rgb('#fff3d6') * 1.2
    # ground for the hands: soft snow with dark broken slabs
    gx, gy = grid(1170, 2080); gr = ramp(np.clip(gy / 1170, 0, 1), [(0, '#8fa2cf'), (.5, '#b6c4e6'), (1, '#dbe4f6')]) * np.clip(1 + .10 * fbm(1170, 2080, (200, 70, 20), seed=5, aspect=2.2), .8, 1.2)[..., None]
    c = Cel(2080, 1170, ss=1); r = np.random.default_rng(6)
    for k in range(16):
        px, py = r.uniform(-60, 2140), r.uniform(260, 1170); s_ = r.uniform(40, 170) * (.4 + py / 1170); n = 6; a = np.sort(r.uniform(0, 2 * math.pi, n)); rr = s_ * r.uniform(.6, 1.1, n); Q = np.c_[px + np.cos(a) * rr, py + np.sin(a) * rr * .45]
        c.fill(Q, ('#46527c', '#58648c', '#3b4668')[k % 3], .8); c.fill(Q * [1, 1] - [0, s_ * .20], '#e9effa', .9)
    gr = comp.over(gr, comp.blur(c.layer(1170, 2080), 5)); gr = comp.blur(gr, 15)
    for k, v in dict(cool=cool, warm=warm, ground=gr).items(): np.save(path(k), v.astype(np.float16))
    del cool, warm, gr, im; return {k: np.load(path(k), mmap_mode='r') for k in KS}


PALC = dict(HD.PAL['day'], skin='#e0c7c6', shade='#9385b0', deep='#645a90', line='#241f2e', nail='#f3e4e2', nail_shade='#b5a6c4', rim='#fff4ee', rim2='#c9d8ff', cloth='#24283f', cloth_sh='#151728', cloth_lt='#56648f', cuff_in='#0b0c14', cast='#8f82a8', button='#b89a58', button_hi='#fff0c4')
PALM_ = dict(PALC, skin='#e3cfce', shade='#9689b4', cloth='#272a3c', cloth_sh='#171824', cloth_lt='#59628c')
PALN = dict(PALC, skin='#ecd0c8', shade='#a58cae', deep='#75608c', cloth='#272a3c', cloth_sh='#171824', cloth_lt='#59628c')


def upperarm(cel, sl, p, w, who):
    """elbow and upper arm: from the far end of the sleeve up to the shoulder above the frame"""
    (a, b), e, m = sl['elbow']; sh = np.array(SHO[who]); d = (e - m) / (np.hypot(*(e - m)) + 1e-6); n = np.array([-d[1], d[0]]); hw = np.hypot(*(a - b)) * .5
    lo, up = (n, -n) if n[1] > 0 else (-n, n); v = (sh - e) / np.hypot(*(sh - e)); nn = np.array([-v[1], v[0]]); nn = nn if nn @ d > 0 else -nn          # nn points to the outer side (elbow tip)
    P = spl([e + lo * hw * .9 - d * hw * 1.1, e + lo * hw * .95 + d * hw * .15, e + d * hw * .95 + lo * hw * .45, e + nn * hw * 1.12 + v * hw * 1.0, e + nn * hw * 1.05 + v * hw * 3.2, sh + nn * hw * 1.00, sh + v * hw * .45, sh - nn * hw * 1.00, e - nn * hw * 1.02 + v * hw * 2.2, e - nn * hw * .75 + v * hw * 1.25, e + up * hw * .9 - d * hw * .9], True, corners=(10,), n=10)
    cel.fill(P, p['cloth']); cel.clip(P)
    cel.fill(np.array([e + lo * hw * .9 - d * hw * 1.2, e + d * hw * .4 + lo * hw * .1, e + nn * hw * .25 + v * hw * 1.6, e + nn * hw * .5 + v * hw * 3.6, sh + nn * hw * .5, sh + nn * hw * 2, e + nn * hw * 2, e + lo * hw * 2]), p['cloth_sh'])
    for k, (t, l, q) in enumerate(((.35, 1.0, .5), (.75, 1.25, -.3), (1.35, .9, .35), (2.3, 1.1, -.2))):
        o = e - nn * hw * .80 + v * hw * t; cel.line(spl([o, o + nn * hw * l * .5 + v * hw * q * .5, o + nn * hw * l + v * hw * q], n=8), w * .7, p['line'], taper=(.1, .6), press=.3, seed=k)
        cel.fill(np.array([o + v * hw * .12, o + nn * hw * l * .55 + v * hw * (q * .5 + .12), o + nn * hw * l * .8 + v * hw * (q * .8 + .16), o + nn * hw * l * .35 + v * hw * (q * .4 + .42)]), p['cloth_lt'])
    cel.unclip(); n_ = len(P); far = np.hypot(*(P - e).T) / (np.hypot(*(sh - e)) + 1e-6)             # 0 at the elbow .. 1 at the shoulder: the pen lifts before the top
    for x_, y_ in HD._runs(far < .84, True):
        Q_ = P[np.arange(x_, y_) % n_]
        if len(Q_) > 8: cel.line(Q_, w, p['line'], taper=(.10, .10), press=.3, seed=3 + x_)


def drawhands(cel, g, w, pals, warm=0.):
    sl = g.extra['sl']; who = g.extra['who']; order = sorted(range(len(who)), key=lambda i: {'y': 0, 'n': 1, 'm': 2}[who[i]]); SK = HD.FING + ('thumb', 'palm')
    for i in order:
        p = pals[who[i]]
        if 'elbow' in sl[i]: upperarm(cel, sl[i], p, w, who[i])
        HS.draw_back(cel, sl[i], p); q = HD.only(g, i)
        if i == 0: q.core = []
        HD.fill_flat(cel, q, p); HD.fill_shadow(cel, q, p); HD.fill_polys(cel, q, g.deeps['d'], p['deep'], names=SK)
        pm = [P for G in q.groups if G['name'] == 'palm' for P in G['polys']]
        if who[i] == 'y' and pm:
            cel.ctx.save(); cel._path(pm); cel.ctx.set_fill_rule(1); cel.ctx.clip()
            for s in g.extra['soot']: cel.fill(s, '#6d6480', .13)
            for k, (P, wt) in enumerate(g.extra['scr']):
                cel.line(P, w * 1.9 * wt, '#c98088', taper=(.3, .5), press=.5, seed=k, alpha=.75); cel.line(P, w * .75 * wt, '#8a2f3c', taper=(.25, .5), press=.5, seed=k + 9)
            cel.ctx.restore()
        HD.fill_overlays(cel, q, p, w)
        if pm: HS.draw_cast(cel, sl[i], p, pm)
        HD.fill_polys(cel, q, g.rims['s'], p['rim'], .85, names=('palm',) if who[i] == 'y' else SK)
        HD.ink_lines(cel, q, p, w, 3 + i)
        done = [j for j in order[:order.index(i)]]
        if done: HD.cross_lines(cel, g, p, w, [(i, j) for j in done], 9 + i)
        HS.draw_front(cel, sl[i], p, w * .95, 5 + i)
        if who[i] == 'y':                                                      # charred edge, ash-grey burn holes
            n_ = sl[i]['near']
            for x_, y_ in HD._runs(n_, True):
                e = sl[i]['ring'][np.arange(x_, y_) % len(n_)]
                if len(e) > 6: cel.line(e, w * 1.5, '#05060a', taper=(.1, .1), press=.6, seed=4); cel.line(e + [0, -w * 1.8], w * .7, '#8a8fae', taper=(.3, .3), press=.8, seed=7, alpha=.5)
    return cel


def flake(cel, x, y, r, rot, tilt, seed, k=1., col='#ffffff', edge='#7f8fc0'):
    """a drawn snow crystal: six arms, none the same.  k 1 fresh .. 0 melted"""
    q = np.random.default_rng(seed); c, s = math.cos(rot), math.sin(rot)
    def T(P): P = np.asarray(P, float); return np.c_[x + (P[:, 0] * c - P[:, 1] * s), y + (P[:, 0] * s + P[:, 1] * c) * tilt]
    segs = []
    for a in range(6):
        th = a * math.pi / 3 + q.normal(0, .05); L = r * q.uniform(.86, 1.08) * (.25 + .75 * k); d = np.array([math.cos(th), math.sin(th)]); segs.append((d * r * .08, d * L, 1.))
        for u, l in ((.42, .30), (.68, .22)):
            for sg in (-1, 1):
                t2 = th + sg * (1.0 + q.normal(0, .08)); segs.append((d * L * u, d * L * u + np.array([math.cos(t2), math.sin(t2)]) * r * l * k * q.uniform(.7, 1.2), .7))
    for w0, col_ in ((r * .17, edge), (r * .095, col)):
        for i, (a, b, wt) in enumerate(segs): cel.line(T([a, (a + b) / 2, b]), max(1., w0 * wt), col_, taper=(.0, .35), press=.2, seed=seed + i, tmin=.4)
    return cel


class Shot:
    id = 'g2'; n = 108
    finish = dict(bloom=.24, vig=.14)
    EVENTS = [(0, 'ambience', dict(note='snow hush, very quiet, far wind')), (8, 'breath_out', dict(size=.3, dur=14)), (22, 'snow_touch', dict(size=.25, note='flake lands on skin, tiny tick')), (27, 'melt', dict(size=.2, note='flake becomes a drop')),
              (27, 'step_snow', dict(size=.3, note='Megumi walks in')), (31, 'step_snow', dict(size=.45, note='second step, foot into snow')), (33, 'cloth', dict(size=.3, note='arm comes in from the left')), (36, 'hand_grab', dict(size=.7, note='Megumi takes the wrist, skin and cloth')), (37, 'snow_fall_off', dict(size=.3)),
              (36, 'step_snow', dict(size=.3, note='Nobara walks in')), (40, 'step_snow', dict(size=.4)), (42, 'cloth', dict(size=.3, note='arm from the right')), (45, 'hand_grab', dict(size=.6, note='Nobara')), (46, 'snow_fall_off', dict(size=.3)),
              (56, 'camera_tilt', dict(size=.4, dur=6, note='fast, up along the three arms, cloth and air')), (62, 'camera_settle', dict(size=.2)), (72, 'sun', dict(size=.8, dur=24, note='light comes in from behind, right')),
              (78, 'beat', dict(size=.2, note='Nobara looks at Yuji')), (80, 'breath_out', dict(size=.2, dur=10, note='Nobara')), (87, 'beat', dict(size=.2, note='Megumi looks down and smiles')), (89, 'breath_out', dict(size=.2, dur=10, note='Megumi, through the nose')), (96, 'beat', dict(size=.3, note='Yuji closes his eyes on the sun')), (98, 'breath_out', dict(size=.5, dur=10, note='Yuji, long'))]

    def setup(self):
        self.G = geometry(); self.P = plates(); J = jobs()
        if not all(os.path.exists(f'cache/g2_p{PV}_{a}_{b}_{c}.npz') for a, b, c in J): _pool(person, J, 2)
        if not all(os.path.exists(f'cache/g2_f{PV}_{k}.npy') for k in STILLS): _pool(still, STILLS, 2)
        self.people = {}; self.x, self.y = grid(H, W)

    def _p(self, who, f, li):
        k = (who, f, li)
        if k not in self.people:
            z = np.load(f'cache/g2_p{PV}_{who}_{f}_{li}.npz'); self.people[k] = (z['a'], tuple(int(v) for v in z['o']))                  # kept as uint8
            if len(self.people) > 7: self.people.pop(next(iter(self.people)))
        a, o = self.people[k]; return a.astype(f32) / 255., o

    def put(self, who, f, li, dy=0.):
        a, (ox, oy) = self._p(who, f, li); return cv2.warpAffine(a, np.float32([[1, 0, ox], [0, 1, oy + dy]]), (W, H), flags=cv2.INTER_LINEAR)

    def body(self, k, dx=0., dy=0., z=1.):
        a = np.load(f'cache/g2_f{PV}_{k}.npy', mmap_mode='r').astype(f32) / 255.; M = np.float32([[2 * z, 0, W / 2 * (1 - z) + dx], [0, 2 * z, H / 2 * (1 - z) + dy]])
        return cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR)

    # ------------------------------------------------ the hands (0..56)
    def hands(self, f, tall=False):
        k = max(q for q in KEYS if q <= f); g = self.G[k]
        cel = Cel(W, H + (720 if tall else 140), 0, -720 if tall else -140); drawhands(cel, g, 4.2, dict(y=PALC, m=PALM_, n=PALN))
        if f < 40:
            sx, sy = g.extra['spot']
            if 4 <= f < 22:                                                  # drifting down
                u = (f - 4) / 18.; py = -40 + (sy + 40) * (u * .9 + .1 * u * u); px = sx + 74 * math.sin(u * 7.2 + .4) * (1 - u) ** .7; flake(cel, px, py, 52 - 10 * u, u * 2.6, .55 + .4 * abs(math.cos(u * 5)), 5)
            elif f >= 22:
                m = float(np.clip((f - 23) / 8., 0, 1)); kk = 1 - m
                if m > .15:                                                  # water: a small dome on the skin
                    R_ = 17 * min(1., m * 1.5); th = np.linspace(0, 2 * math.pi, 14)[:-1]; D_ = spl(np.c_[sx + np.cos(th) * R_ * 1.2 * (1 + .08 * np.sin(th * 3)), sy + 3 + np.sin(th) * R_ * .85], True, n=5)
                    cel.fill(D_ + [2.5, 4], '#8a7ca6', .55); cel.fill(D_, '#d9e2f6', .85); cel.fill(spl(np.c_[sx - 4 + np.cos(th) * R_ * .42, sy - 2 + np.sin(th) * R_ * .24], True, n=4), '#ffffff'); cel.line(np.vstack([D_, D_[:1]]), 1.5, '#5f5680', taper=(.2, .2), seed=2)
                if kk > .02: flake(cel, sx, sy, 42 * (.35 + .65 * kk), 2.6, .80, 5, kk, '#ffffff', '#7f8cc4')
        return cel.rgba()

    def dip(self, f):
        d = 0.
        for t0, a in ((36, 17.), (45, 13.)):
            if f >= t0: q = f - t0; d += a * math.exp(-q / 3.2) * math.cos(q * .95) + (4 if q > 0 else 0)
            elif t0 - 3 <= f < t0: d += -3 * (f - t0 + 3) / 3.
        if 22 <= f < 30: d += 3.5 * math.exp(-(f - 22) / 2.)
        return d + 2.2 * math.sin(f / 24. * 2 * math.pi * .55)

    def handsframe(self, f, off=0.):
        """the first half; off = camera offset in px (picture moves down)"""
        im = comp.cam2d(np.asarray(self.P['ground'], f32), 1040 + .3 * f, 585 - off * .5, 1.02, 0, W, H); z = 1 + .0006 * f
        for who in 'mny':                                                       # bodies: steps on twos, weight drops at the contact
            ks = [k for k in STEP[who] if k[0] <= f]
            if not ks: continue
            L = self.body(ks[-1][1], 0, off * .85 + 1.4 * math.sin(f * .16 + ord(who)), z); im = kit.cast(im, L, 14, 26, '#5f70a6', .30, 24); im = comp.over(im, L)
        hl = self.hands(f); zz = (1 + .0011 * f) * (1 + .13 * (1 - comp.ease(np.clip((f - 14) / 26., 0, 1)))); M = np.float32([[zz, 0, 800 * (1 - zz)], [0, zz, 40 * (1 - zz) - 140 * zz + self.dip(f) + off]])
        hl = cv2.warpAffine(hl, M, (W, H), flags=cv2.INTER_LINEAR); im = kit.cast(im, hl, 12, 36, '#3a4576', .40, 22); im = comp.over(im, hl)
        for t0, cx_, cy_, sd in ((36, 600, 300, 3), (45, 900, 210, 8)):                  # snow shaken off the sleeves
            q = f - t0
            if 0 <= q < 12:
                r = np.random.default_rng(sd); cel = Cel()
                for j in range(26):
                    px, py = cx_ + r.normal(0, 150), cy_ + r.normal(0, 50); vx, vy = r.normal(0, 4.5), r.uniform(-7, 1); X = px + vx * q; Y = py + vy * q + .62 * q * q; R_ = r.uniform(2, 8.5) * (1 - q / 16.); th = np.linspace(0, 2 * math.pi, 8)[:-1]
                    cel.fill(spl(np.c_[X + np.cos(th) * R_ * r.uniform(.8, 1.2, 7), Y + np.sin(th) * R_ * r.uniform(.8, 1.2, 7)], True, n=4), '#ffffff', .95)
                im = comp.over(im, cel.layer())
        if 8 <= f < 30: im = self.breath(im, f - 8, 22., (1010, -30), (-150, 230), 1.0, 3)
        return im

    def tilt(self, f):
        """six drawn frames up along the arms: one tall space, nothing smeared"""
        c = {56: .055, 57: .21, 58: .48, 59: .76, 60: .945, 61: 1.018}[f]; oy = c * D_; x, y = self.x, self.y
        gr = comp.cam2d(np.asarray(self.P['ground'], f32), 1040 + .3 * 56, 585 - oy * .5, 1.04, 0, W, H); sk = comp.cam2d(np.asarray(self.P['cool'], f32), 1040, 575 - (D_ - oy) * .52, 1.05, 0, W, H)
        e = smooth(-220, 120, y - (oy * 1.12 - 60))[..., None]; im = sk * (1 - e) + gr * e; sg = 7. * (1 - smooth(.15, .7, c)); pl = None
        for who in 'mny': L = self.put(who, 'T', 'snow', oy - D_); pl = L if pl is None else pl * (1 - L[..., 3:4]) + L
        pl = pl * (1 - smooth(oy + 30, oy + 290, y))[..., None]                     # below this the soft bodies of the first half carry on
        if sg > .6: pl = comp.blur(pl, float(sg))
        for who in 'mny': im = comp.over(im, self.body(STEP[who][-1][1], 0, oy, 1.0336))                # blurred legs go on below the coats
        im = kit.cast(im, pl, 16, 14, '#3a4576', .25, 16); im = comp.over(im, pl)
        hl = self.hands(56, True); zz = 1 + .0011 * 56; M = np.float32([[zz, 0, 800 * (1 - zz)], [0, zz, 40 * (1 - zz) - 720 * zz + self.dip(55) + oy]]); hl = cv2.warpAffine(hl, M, (W, H), flags=cv2.INTER_LINEAR)
        hb = 6. * smooth(.45, .95, c)
        if hb > .6: hl = comp.blur(hl, float(hb))
        hl = hl * (1 - smooth(.90, 1.0, c)) if c > .9 else hl
        im = kit.cast(im, hl, 14, 22, '#141a36', .40, 16); return comp.over(im, hl)

    # ------------------------------------------------ the faces (59..108)
    def facesframe(self, f, dy=0.):
        x, y = self.x, self.y; warm = float(comp.ease(np.clip((f - 70) / 16., 0, 1))); z = 1.03 + .00035 * f
        sk = comp.cam2d(np.asarray(self.P['cool'], f32), 1040, 575 - dy * .4, z, 0, W, H)
        if warm > 0:
            wm = comp.cam2d(np.asarray(self.P['warm'], f32), 1040, 575 - dy * .4, z, 0, W, H); fr_ = smooth(-260, 260, x - (W + 320) + (W + 900) * warm + .25 * (y - 540)); sk = sk * (1 - fr_[..., None]) + wm * fr_[..., None]
        im = sk; kf = max(q for q in FR if q <= max(f, 61)); pl = None
        for who in 'mny':
            if warm < 1: L = self.put(who, min(kf, 87), 'snow', dy)
            if warm > 0:
                Lw = self.put(who, max(kf, 72), 'warm', dy)
                if warm < 1:
                    off = dict(y=0, m=90, n=-70)[who]; fr_ = smooth(-34, 34, x - (W + 320) + (W + 900) * warm + .33 * (y - 540) + off + 26 * np.sin(y * .012 + x * .004))[..., None]; L = L * (1 - fr_) + Lw * fr_
                else: L = Lw
            if pl is not None:                                                 # whoever leans in front throws a shadow on the one behind
                sh = comp.blur(kit.shifted(L[..., 3], -14 if who == 'y' else 10, 12), 9)[..., None] * pl[..., 3:4] * .30; pl[..., :3] = pl[..., :3] * (1 - sh) + rgb('#231a30') * sh * pl[..., 3:4]
            pl = L if pl is None else pl * (1 - L[..., 3:4]) + L
        zp = 1 + .035 * float(comp.ease(np.clip((f - 62) / 46., 0, 1))); M = np.float32([[zp, 0, 962 * (1 - zp)], [0, zp, 480 * (1 - zp)]]); pl = cv2.warpAffine(pl, M, (W, H), flags=cv2.INTER_LINEAR)
        im = kit.cast(im, pl, 18, 10, '#1a1c30', .30, 14); im = comp.over(im, pl)
        if warm > 0: im = comp.over(im, pl * 0) ; b_ = smooth(520, 1080, y)[..., None] * pl[..., 3:4] * warm; im = im + b_ * rgb('#ffcfa6') * .10        # bounce off the snow
        for t0, n_, p_, v_, a_, sd in ((64, 14., (1010, 726), (90, 70), .55, 2), (80, 13., (1236, 642), (-46, 74), .60, 5), (89, 13., (662, 636), (40, 78), .60, 8), (98, 10., (1040, 712), (130, 34), .95, 11)):
            if t0 <= f < t0 + n_: im = self.breath(im, f - t0, n_, p_, v_, a_, sd)
        return im, warm

    def frame(self, f):
        x, y = self.x, self.y; warm = 0.; snow_dy = 0.
        if f <= 55: im = self.handsframe(f, {54: -3., 55: -7.}.get(f, 0.))
        elif f <= 61: im = self.tilt(f); snow_dy = {56: 54, 57: 150, 58: 260, 59: 260, 60: 150, 61: 60}[f]
        else: im, warm = self.facesframe(f, {62: 9., 63: 3.}.get(f, 0.))
        col = '#f6f8ff' if warm < .02 else '#%02x%02x%02x' % tuple(int(255 * v) for v in (rgb('#f6f8ff') * (1 - warm) + rgb('#fff0cf') * warm))
        sn = fx.snow(f + 40, n=(260, 130, 40, 7), wind=-46, fall=88, seed=12, col=col, shade='#b9c3dd' if warm < .5 else '#e0c3a4', op=.95)
        im = comp.over(im, kit.shifted(sn, 0, snow_dy * .8) if snow_dy else sn)
        if warm > 0:                                                            # the sun: raked light, haze, flare off the right edge; 30 % less where Nobara stands
            sx, sy = SUN; nb = 1 - .30 * np.exp(-(((x - 1300) / 330) ** 2 + ((y - 520) / 420) ** 2)); d = np.hypot((x - sx) / 1.3, y - sy); im = im + ((smooth(1500, 60, d) ** 2) * warm * nb)[..., None] * rgb('#ffb061') * .24
            th = np.arctan2(y - sy, sx - x); ray = .5 + .5 * np.sin(th * 23 + f * .03) * np.sin(th * 9.3 - f * .021 + 1.3); im = im + ((smooth(1700, 200, d) * smooth(.35, .95, ray)) * warm * nb)[..., None] * rgb('#ffd9a0') * .10
            im = im + fx.motes(f, 70, 11, smooth(1700, 100, d) * warm, '#fff0cf', (-10, -4), 1.5, .9)
        return np.ascontiguousarray(im, dtype=f32)

    def breath(self, im, k, n, p, v, amt, seed):
        c = Cel(); q = np.random.default_rng(seed); t = k / n
        for j in range(13):
            cx = p[0] + v[0] * t * (.5 + .06 * j) + q.normal(0, 22) * (.4 + t); cy = p[1] + v[1] * t * (.5 + .05 * j) + q.normal(0, 14) * (.4 + t); R_ = (16 + 6 * j) * (.5 + 1.5 * t) * q.uniform(.7, 1.2); th = np.linspace(0, 2 * math.pi, 11)[:-1]
            rr = R_ * (1 + .22 * np.sin(th * 3 + j * 2.1 + (k // 2) * .5)); c.fill(spl(np.c_[cx + np.cos(th) * rr * 1.35, cy + np.sin(th) * rr * .72], True, n=5), '#ffffff', .12)
        L = comp.blur(c.layer(), 5 + 9 * t); return comp.over(im, L, amt * math.sin(math.pi * min(1., t * 1.02)) ** .8)
