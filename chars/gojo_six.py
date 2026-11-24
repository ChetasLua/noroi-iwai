"""six_eye(): the extreme close-up of the Six Eyes, drawn at full frame.  Everything is a function of (open, t)."""
import numpy as np, math, cv2, cairo
from jj import comp
from jj.ink import Cel, spl, resample, noise1
from jj.paint import rgb, f32, smooth, fbm, ramp
from jj.kit import outline
from jj import kit
from .gojo_eye import _rad, _lin

PAL6 = dict(limb='#0a1440', ring='#12349a', mid='#1f7be8', body='#4fbcff', lite='#a6ecff', core='#f4feff', pupil='#060f30', pup2='#123f9c', deep='#0c2a86',
            viol='#a98bff', mint='#a8ffea')
SK = dict(base='#4c4f9c', lit='#8fb2ee', sh='#2f3176', dp='#191a4a', ink='#0c0c24', warm='#8a6fb0', rim='#e9fbff', ln='#12123a')
LASH = dict(hi='#ffffff', base='#e6eeff', sh='#93a4e2', dp='#5a67b4', ln='#191b4c')


# ------------------------------------------------------------------------------------------------ the iris, full detail
def big_iris(cel, c, R, t=0., pupil=.27, par=(0., 0.), glow=.35, seed=11, shade=.9, pal=PAL6, ex=None, ey=None, spin=None, refr=0.):
    c = np.asarray(c, float); ex = np.array([R, 0.]) if ex is None else np.asarray(ex, float); ey = np.array([0., R]) if ey is None else np.asarray(ey, float)
    M = cairo.Matrix(ex[0], ex[1], ey[0], ey[1], c[0], c[1]); B = np.stack([ex, ey]); T = lambda P: np.asarray(P, float) @ B + c
    r = np.random.default_rng(seed); par = np.asarray(par, float); spin = t * .12 if spin is None else spin
    q = np.linspace(0, 2 * np.pi, 200, endpoint=False); circ = np.c_[np.cos(q), np.sin(q)]
    pol = lambda a, rr: np.c_[np.cos(a) * rr, np.sin(a) * rr]
    cel.clip(T(circ))
    # the well: ultramarine rim -> cobalt -> cyan -> light
    _rad(cel, M, None, 0, 1, [(0, pal['deep'], 1), (pupil * .9, pal['ring'], 1), (pupil + .06, pal['lite'], 1), (pupil + .17, pal['body'], 1), (.60, pal['mid'], 1), (.78, pal['ring'], 1), (.92, pal['deep'], 1), (.985, pal['limb'], 1), (1, pal['limb'], 1)], centre=tuple(par * .5))
    _rad(cel, M, None, 0, .95, [(0, pal['lite'], 0), (.30, pal['lite'], 0), (.60, pal['lite'], .95), (.82, pal['body'], .75), (1, pal['body'], 0)], centre=(par[0] * .5, .36 + par[1] * .5))
    # --- cut facets: three rings of kites, each facet its own brightness (a brilliant cut that turns very slowly)
    for ring, (r0, r1, n, sgn) in enumerate(((.36, .64, 14, 1), (.52, .84, 18, -1), (.74, 1.0, 24, 1))):
        a0 = spin * sgn * (.5 - ring * .12) + ring * .4
        for k in range(n):
            am = a0 + (k + .5) * 2 * math.pi / n; da = math.pi / n; j = r.uniform(-.02, .02)
            kite = np.array([(math.cos(am) * r0, math.sin(am) * r0), (math.cos(am + da) * (r0 * .45 + r1 * .55 + j), math.sin(am + da) * (r0 * .45 + r1 * .55 + j)), (math.cos(am) * r1, math.sin(am) * r1), (math.cos(am - da) * (r0 * .45 + r1 * .55 - j), math.sin(am - da) * (r0 * .45 + r1 * .55 - j))])
            o = par * (.35 + .2 * ring); v = r.random()
            lit = math.sin(am * 1.0 + .3) * .5 + .5                      # lower facets catch the light
            if v < .42: cel.fill(T(kite + o), pal['core'] if lit > .45 else pal['lite'], (.08 + .30 * r.random()) * (.25 + .75 * lit))
            elif v < .72: cel.fill(T(kite + o), pal['deep'], (.10 + .30 * r.random()) * (1.1 - .7 * lit))
            # half facet: the hard edge down the middle
            if r.random() < .6: cel.fill(T(kite[[0, 1, 2]] + o), pal['lite'] if lit > .4 else pal['mid'], .10 + .22 * r.random())
            if r.random() < .30: cel.fill(T(np.array([kite[0] * .6 + kite[2] * .4, kite[1] * .97, kite[2]]) + o), pal['viol'] if r.random() < .6 else pal['mint'], .28)
    # --- fibres: long fine shards from the pupil (dense, unequal), six of them are longer: the six
    def shard(a, w, r0, r1, col, al, k=.3, bend=0.):
        o = par * k; m = r0 * .45 + r1 * .55
        pts = np.vstack([pol(np.array([a - w * .18]), r0), pol(np.array([a - w * .5 + bend * .4]), m), pol(np.array([a + bend]), r1), pol(np.array([a + w * .5 + bend * .4]), m), pol(np.array([a + w * .18]), r0)])
        cel.fill(T(pts + o), col, al)
    for k in range(110):
        a = r.uniform(0, 6.2832) + spin * .35; lo = math.sin(a) * .5 + .5
        col = [pal['core'], pal['lite'], pal['body'], pal['ring'], pal['deep']][int(np.clip(r.normal(1.9 - 1.5 * lo + .6, 1.0), 0, 4))]
        shard(a, r.uniform(.018, .075), pupil * r.uniform(1.05, 1.5), r.uniform(.50, .97), col, r.uniform(.25, .75), k=.3, bend=r.normal(0, .035))
    for k in range(6):
        a = math.pi / 6 + k * math.pi / 3 + spin * .2
        shard(a, .30, pupil * 1.0, .80, pal['lite'], .38, k=.4); shard(a, .12, pupil * 1.0, .93, pal['core'], .80, k=.4, bend=.01)
    # dark cracks growing in from the rim
    for k in range(30):
        a = r.uniform(0, 6.2832); shard(a, r.uniform(.012, .06), 1.0, r.uniform(.66, .9), pal['limb'], r.uniform(.4, .8), k=.1, bend=r.normal(0, .05))
    # --- broken concentric rings with a thin prism edge
    for j, (rr, th, col, al) in enumerate(((.405, .020, pal['core'], .95), (.455, .006, pal['core'], .8), (.535, .013, pal['lite'], .9), (.585, .005, pal['core'], .8), (.665, .016, pal['core'], .75), (.715, .005, pal['lite'], .8), (.80, .010, pal['lite'], .7), (.865, .006, pal['core'], .7), (.925, .008, pal['body'], .8))):
        a = r.uniform(0, 6.28) + spin * (1.0 if j % 2 else -.7); end = a + 2 * math.pi
        while a < end - .1:
            ln = r.uniform(.10, .95); b = min(a + ln, end - .04); s_ = np.linspace(a, b, 26); jr = rr + r.normal(0, .006)
            tw = np.minimum(1, np.minimum((s_ - a), (b - s_)) / .06 + .12) * th * r.uniform(.6, 1.5) * .5
            lo = .45 + .55 * max(0., math.sin((a + b) / 2) * .8 + .4); o = par * (.35 + .5 * (1 - rr))
            if r.random() < .5: cel.fill(T(np.vstack([pol(s_, jr + tw + .006 + refr * .012), pol(s_[::-1], (jr + tw * .2 + .001)[::-1])]) + o), pal['viol'] if j % 2 else pal['mint'], .55 * lo)
            cel.fill(T(np.vstack([pol(s_, jr + tw), pol(s_[::-1], (jr - tw)[::-1])]) + o), col, al * lo)
            if r.random() < .35: cel.fill(T(spl(pol(np.array([b + .02, b + .035, b + .05, b + .035]), np.array([jr, jr + .012, jr, jr - .012])) + o, True, n=3)), pal['core'], .9 * lo)
            a = b + r.uniform(.05, .38)
    # --- sky: cloud flecks, two tones, flat bottoms toward the pupil
    for k in range(9):
        a = (.18 + .64 * (k + r.uniform(.1, .9)) / 6) * math.pi if k < 6 else r.uniform(1.1, 1.9) * math.pi
        a += spin * .22 + .015 * math.sin(t * 1.3 + k); rr = r.uniform(.56, .80); p = np.array([math.cos(a) * rr, math.sin(a) * rr]) + par * .5; tg = np.array([-math.sin(a), math.cos(a)]); ra = np.array([math.cos(a), math.sin(a)])
        L = r.uniform(.07, .14); nb = int(r.integers(5, 8)); W2 = lambda a_: p + tg * a_[:, :1] + ra * a_[:, 1:2]; th_ = np.linspace(0, 2 * np.pi, 22, endpoint=False); cc = np.c_[np.cos(th_), np.sin(th_)]
        bl = []
        for q_ in range(nb):
            u = -1 + 2 * (q_ + .5) / nb; rad = L * .30 * (1.05 - .70 * u * u) * r.uniform(.8, 1.2); bl.append((u * L * .9, rad * r.uniform(.15, .75) * (1 - u * u), rad))
        base_ = spl(np.array([(-L * 1.25, 0), (-L * .5, -L * .10), (L * .5, -L * .11), (L * 1.25, 0), (L * .5, L * .12), (-L * .5, L * .12)]), True, corners=(0, 3), n=5)
        cel.fill(T(W2(base_)), '#ffffff', .95)
        for bx, by, rad in bl: cel.fill(T(W2(cc * rad + (bx, by))), '#ffffff', .95)
        for bx, by, rad in bl: cel.fill(T(W2(cc * rad * .72 + (bx + rad * .18, by - rad * .28))), '#b4d6fb', .55)
        cel.fill(T(W2(base_ * (.92, .75) - (0, L * .03))), '#8fc0f6', .50)
    # --- corona and pupil
    n = 150; ang = np.linspace(0, 2 * np.pi, n, endpoint=False); nz = noise1(n, 5, r) * .6; nz[-8:] = nz[-8:] * np.linspace(1, 0, 8) + nz[0] * np.linspace(0, 1, 8)
    pc = par * .95
    _rad(cel, M, None, 0, pupil * 1.9, [(0, pal['core'], .9), (.52, pal['core'], .85), (.68, pal['lite'], .55), (1, pal['lite'], 0)], centre=tuple(pc))
    for k in range(120):
        a = r.uniform(0, 6.2832); big = r.random() ** 3; l = pupil * (1.10 + .16 * r.random() + .55 * big); wd = r.uniform(.016, .05) * (1 - .4 * big)
        cel.fill(T(np.vstack([pol(np.array([a - wd]), pupil * .99), pol(np.array([a + r.normal(0, .01)]), l), pol(np.array([a + wd]), pupil * .99)]) + pc), pal['core'], r.uniform(.6, 1.))
    for k in range(6):
        a = math.pi / 6 + k * math.pi / 3 + spin * .2; cel.fill(T(np.vstack([pol(np.array([a - .13]), pupil * .99), pol(np.array([a]), pupil * 2.15), pol(np.array([a + .13]), pupil * .99)]) + pc), '#ffffff', .95)
    cel.fill(T(circ * pupil * 1.045 + pc), pal['core'])
    rr = pupil * (1.0 + .016 * nz); cel.fill(T(pol(ang, rr) + pc), pal['pup2'])
    cel.fill(T(circ * pupil * .88 + pc), pal['pupil'])
    _rad(cel, M, T and (circ * pupil * .88 + pc).tolist(), 0, pupil, [(0, '#2a6fe0', .75), (.5, '#133a94', .5), (1, pal['pupil'], 0)], centre=tuple(pc + np.array([.0, .10])))
    for k in range(26):                                        # a little deep space inside
        a, d = r.uniform(0, 6.28) + spin, pupil * .8 * math.sqrt(r.random()); w = r.uniform(.003, .010)
        p = np.array([math.cos(a) * d, math.sin(a) * d]) + pc * 1.15; cel.fill(T(p + np.array([(0, -w * 1.8), (w * .4, 0), (0, w * 1.8), (-w * .4, 0)])), '#cfeaff', r.uniform(.4, 1.))
    if glow > 0:
        _rad(cel, M, None, 0, 1, [(0, '#ffffff', .55 * glow ** 1.5), (pupil, '#eafdff', .62 * glow), (pupil + .20, '#b4f0ff', .48 * glow), (.75, '#66ccff', .16 * glow), (1, '#66ccff', 0)], centre=tuple(pc))
    if shade > 0: _lin(cel, M, None, (0, -1.02), (0, .02), [(0, '#050c34', .96 * shade), (.40, '#0b2170', .72 * shade), (.75, '#12389c', .30 * shade), (1, '#1f62d4', 0)])
    cel.unclip()
    w = 1 + .5 * np.clip(noise1(200, 14, r), -1, 1) * .4
    P = np.vstack([circ * 1.0, circ[:1]]); cel.line(T(P * .99), max(2., R * .030), pal['limb'], taper=(0, 0), press=.35, seed=seed, tremor=0)


def hi_lights(cel, c, R, al=1., s=-1):
    T = lambda P: np.asarray(P, float) * R + np.asarray(c, float)
    a = spl(np.array([(.27, -.66), (.45, -.60), (.585, -.43), (.61, -.24), (.535, -.12), (.455, -.17), (.37, -.31), (.25, -.45)]) * (s, 1), True, corners=(0, 3, 4, 7), n=8)
    cel.grad(T(a), T(np.array([(0, -.66)]) * (s, 1))[0], T(np.array([(0, -.12)]) * (s, 1))[0], [(0, '#ffffff', .98 * al), (.55, '#f2fbff', .95 * al), (1, '#a9dcff', .80 * al)])
    cel.fill(T(spl(np.array([(.655, -.40), (.725, -.26), (.735, -.06), (.69, .03), (.66, -.14)]) * (s, 1), True, corners=(0, 3), n=6)), '#ffffff', .85 * al)
    cel.fill(T(spl(np.array([(.10, -.74), (.22, -.71), (.18, -.57), (.09, -.60)]) * (s, 1), True, corners=(0, 2), n=6)), '#ffffff', .80 * al)
    cel.fill(T(spl(np.array([(-.40, .37), (-.29, .30), (-.20, .37), (-.25, .49), (-.36, .50)]) * (s, 1), True, corners=(0, 2, 4), n=6)), '#ffffff', .95 * al)
    cel.fill(T(spl(np.array([(-.16, .52), (-.11, .48), (-.07, .53), (-.12, .58)]) * (s, 1), True, corners=(0, 2), n=5)), '#ffffff', .9 * al)
    q = np.linspace(.20 * math.pi, .78 * math.pi, 40); w = np.sin(np.linspace(0, math.pi, 40)) ** .7 * .040
    cel.fill(T(np.vstack([np.c_[np.cos(q), np.sin(q)] * .925, (np.c_[np.cos(q), np.sin(q)] * (.925 - w)[:, None])[::-1]])), '#e4fbff', .85 * al)


# ------------------------------------------------------------------------------------------------ lids
UP1 = np.array([(-648, 104), (-600, 48), (-520, -38), (-400, -138), (-240, -222), (-40, -262), (170, -250), (360, -190), (500, -110), (590, -48), (652, -4)], float)
UP0 = np.array([(-648, 108), (-600, 128), (-520, 156), (-400, 190), (-240, 222), (-40, 238), (170, 228), (360, 186), (500, 126), (590, 70), (652, 16)], float)
LO1 = np.array([(652, -4), (600, 52), (500, 122), (350, 186), (160, 224), (-40, 232), (-240, 214), (-420, 178), (-540, 140), (-610, 118), (-648, 104)], float)


def lids(open_):
    o = float(np.clip(open_, 0, 1.15)); U = UP0 * (1 - o) + UP1 * o
    Lo = LO1 + np.array([0, 1.]) * (10 * (1 - o)) * np.sin(np.linspace(0, np.pi, len(LO1)))[:, None]
    return U, Lo


def lash(cel, p, d, L, w, bow, col, ln, lnw, hi=None, seed=0):
    """one eyelash clump: a curved blade"""
    p, d = np.asarray(p, float), np.asarray(d, float); d = d / (np.hypot(*d) + 1e-6); n = np.array([-d[1], d[0]])
    u = np.linspace(0, 1, 20)[:, None]; C = p + d * L * u + n * bow * L * (u ** 2)
    Tn = np.gradient(C, axis=0); Tn /= np.hypot(Tn[:, 0], Tn[:, 1])[:, None]; N = np.c_[-Tn[:, 1], Tn[:, 0]]
    wd = w * (np.interp(u[:, 0], [0, .12, .4, 1], [.55, 1.0, .72, 0.])) * .5
    A, B = C + N * wd[:, None], C - N * wd[:, None]; P = np.vstack([A, B[::-1]])
    cel.fill(P, col)
    if hi: cel.fill(np.vstack([A[2:], (C + N * wd[:, None] * .2)[2:][::-1]]), hi)
    cel.line(A[1:], lnw, ln, taper=(.4, .1), press=.3, seed=seed, tremor=.1, tmin=.1); cel.line(B[1:], lnw * 1.25, ln, taper=(.4, .1), press=.3, seed=seed + 1, tremor=.1, tmin=.1)


def six_eye(w=1920, h=1080, open=1., t=0., pupil=.27, glow=.35, gaze=(0., 0.), refr=0., side=1, centre=None, scale=1., roll=-.06, seed=5, skin=True, wet=1., lash_sway=0., light=1., ss=2, mid=None):
    """full-frame layer (premultiplied RGBA).  open 0 shut .. 1 open .. 1.1 wide; pupil .12 .. .5; glow 0..1 light in the iris;
    gaze in iris radii; refr 0..1 prism split; side 1 = inner corner on the left of the picture, -1 mirrored."""
    c = Cel(w, h, ss=ss); k = scale * w / 1920.; ctr = np.array([w * .5, h * .545]) if centre is None else np.asarray(centre, float)
    cs, sn = math.cos(roll), math.sin(roll)
    def Q(P):
        P = np.asarray(P, float) * (side, 1); return np.stack([P[..., 0] * cs - P[..., 1] * sn, P[..., 0] * sn + P[..., 1] * cs], -1) * k + ctr
    U_, L_ = lids(open); U, Lo = spl(U_, n=16), spl(L_, n=16)
    r = np.random.default_rng(seed); ink = SK['ink']
    cr = U_ + np.c_[np.zeros(len(U_)), np.interp(np.linspace(0, 1, len(U_)), [0, .2, .5, .8, 1], [-22, -66, -84, -70, -24])] * (1 if open > .2 else .6); CR = spl(cr, n=12)
    if skin:
        full = np.array([(0, 0), (w, 0), (w, h), (0, h)], float)
        c.fill(full, SK['base'])
        c.grad(full, Q((0, -560)), Q((0, 520)), [(0, '#0e0e30', 1), (.28, SK['dp'], 1), (.52, SK['sh'], 1), (.78, '#484a98', 1), (1, '#2c2d70', 1)])
        # planes: socket under the brow (angular edge), nose bridge, cheek
        c.fill(Q(spl(np.array([(-1100, -900), (1400, -900), (1400, -190), (1120, -176), (930, -120), (800, -64), (700, -118), (560, -240), (400, -322), (190, -374), (-60, -384), (-290, -340), (-450, -262), (-590, -140), (-690, 10), (-746, 150), (-800, 330), (-880, 700), (-1300, 700)]), True, corners=(0, 1, 2, 5, 12, 15, 16, 17, 18), n=8)), SK['dp'])
        c.fill(Q(spl(np.array([(-1300, -900), (-690, -900), (-640, -560), (-625, -360), (-650, -200), (-716, -50), (-790, 120), (-845, 310), (-900, 560), (-940, 800), (-1300, 800)]), True, corners=(0, 1, 9, 10), n=8)), '#0c0c2c')
        c.fill(Q(spl(np.array([(-1300, 420), (-900, 300), (-800, 420), (-660, 520), (-400, 600), (0, 640), (400, 600), (760, 480), (1400, 300), (1400, 900), (-1300, 900)]), True, corners=(0, 1, 8, 9, 10), n=8)), SK['sh'])
        # upper lid: plane between lash line and crease, caught by the light of the eye
        c.fill(Q(np.vstack([U, CR[::-1]])), '#5f6fc0')
        c.fill(Q(spl(np.vstack([U_[2:9] + (0, -8), (cr[3:8] + (0, 16))[::-1]]), True, corners=(0, 6, 7, 11), n=8)), '#8fb6f2', .85)
        # lower lid: wet thickness, then the pouch lit from inside, its hard edge, then the plane that turns away
        c.fill(Q(spl(np.array([(-590, 150), (-430, 218), (-190, 268), (60, 286), (290, 266), (480, 200), (610, 110), (530, 208), (400, 284), (220, 326), (0, 340), (-230, 322), (-430, 268)]), True, corners=(0, 6), n=8)), '#5a6cc0')
        c.fill(Q(spl(np.array([(-330, 252), (-160, 284), (60, 296), (240, 280), (390, 236), (300, 290), (120, 312), (-90, 312), (-230, 290)]), True, corners=(0, 4), n=8)), '#8fb8f2', .9)
    if skin:
        # crease, folds at the corners, pen hatching: the hand of the key animator
        c.line(Q(CR[int(len(CR) * .06):int(len(CR) * .97)]), 4.4 * k, SK['ln'], taper=(.35, .22), press=.4, seed=seed + 7, tmin=.06)
        cr2 = CR + np.c_[np.zeros(len(CR)), np.interp(np.linspace(0, 1, len(CR)), [0, .5, 1], [-8, -30, -34])]
        c.line(Q(cr2[int(len(CR) * .48):int(len(CR) * .93)]), 2.6 * k, SK['ln'], taper=(.5, .4), seed=seed + 8, tmin=.05)
        c.line(Q(spl(np.array([(-652, 110), (-690, 132), (-726, 170)]), n=8)), 4 * k, SK['ln'], taper=(.1, .7), seed=seed + 9)
        for q_ in range(13):
            a0 = np.array([-742 + q_ * 9, 178 + q_ * 24.]); c.line(Q(np.array([a0, a0 + (18, 26), a0 + (30 + r.uniform(0, 20), 62)])), 2.0 * k, '#4a4c98', taper=(.2, .7), seed=seed + 20 + q_, alpha=.8)
        for q_ in range(16):
            a0 = np.array([700 + q_ * 20., -150 - q_ * 4.]); c.line(Q(np.array([a0, a0 + (-10, 30), a0 + (-26, 70 + r.uniform(0, 24))])), 2.0 * k, '#3a3c86', taper=(.2, .75), seed=seed + 40 + q_, alpha=.8)
        for q_ in range(9):
            a0 = np.array([-300 + q_ * 70., 300 + 22 * math.sin(q_ * .7)]); c.line(Q(np.array([a0, a0 + (14, 8), a0 + (34, 12)])), 1.8 * k, SK['sh'], taper=(.3, .6), seed=seed + 60 + q_)
    if mid is not None: mid(c, Q, k)
    E = Q(np.vstack([U, Lo[1:]]))
    ir = np.array([2 + gaze[0] * 286, -6 + gaze[1] * 286]); R = 286 * k
    ic = Q(ir)
    c.fill(E, '#c9d6f6')
    c.clip(E)
    c.grad(E, Q((-648, 0)), Q((652, 0)), [(0, '#515aa8', 1), (.16, '#8f9fdc', 1), (.32, '#c9d8f6', 1), (.46, '#dbe7fb', 1), (.60, '#dbe7fb', 1), (.76, '#b9c8f0', 1), (.90, '#7b89cc', 1), (1, '#4b54a4', 1)])
    c.grad(E, Q((0, 262)), Q((0, 20)), [(0, '#6fb8ff', .85), (.45, '#a9dcff', .30), (1, '#ffffff', 0)])
    # caustic slivers the iris throws on the ball, pointing at it
    for a_ in ([(-318, 40), (-470, 66), (-330, 56)], [(-330, 92), (-520, 122), (-480, 132), (-336, 116)], [(-320, 150), (-430, 176), (-330, 168)], [(330, 6), (470, 8), (330, 22)], [(340, 60), (520, 48), (500, 62), (336, 82)], [(326, 120), (450, 112), (330, 136)], [(-322, -20), (-400, -18), (-322, -8)], [(322, -50), (410, -70), (326, -38)]):
        c.fill(Q(np.array(a_, float)), '#f2fbff', .60)
    sg = side
    ex, ey = np.array([cs, sn]) * R, np.array([-sn, cs]) * R
    c.unclip(); c.clip(E)
    big_iris(c, ic, R, t, pupil, (gaze[0] * .25, gaze[1] * .25), glow, shade=.80 * (1 - .4 * glow), ex=ex, ey=ey, refr=refr)
    # --- cast shadow of lid + lash blades on the ball (hard edge, saw-toothed)
    n = len(U); tt = np.linspace(0, 1, n); dep = np.interp(tt, [0, .12, .45, .8, 1], [8, 40, 62, 46, 10]); D = U + np.c_[0 * dep, dep]
    saw = D.copy(); j = np.arange(n); saw[:, 1] += 20 * np.abs(((j * .155) % 1) - .5) * np.sin(np.pi * tt)
    c.fill(Q(np.vstack([U, saw[::-1]])), '#1b2468', .66)
    hi_lights(c, ic, R, wet)
    # tear film along the lower lid, caruncle
    dn = Lo + np.c_[np.zeros(len(Lo)), -np.interp(np.linspace(0, 1, len(Lo)), [0, .1, .5, .9, 1], [2, 10, 13, 10, 3])]
    c.fill(Q(np.vstack([Lo, dn[::-1]])), '#d3b4d2', .9)
    for a, b in ((.18, .33), (.38, .46), (.52, .72), (.80, .86)):
        i0, i1 = int(a * len(Lo)), int(b * len(Lo)); c.line(Q((dn[i0:i1])), 4.5 * k, '#ffffff', taper=(.3, .3), press=.3, seed=seed + i0)
    c.fill(Q(spl(np.array([(-648, 104), (-618, 78), (-582, 84), (-566, 106), (-590, 124), (-625, 120)]), True, corners=(0,), n=6)), '#b98fbf'); c.fill(Q(spl(np.array([(-628, 100), (-606, 88), (-590, 98), (-606, 110)]), True, n=6)), '#efd6ee')
    c.unclip()
    # --- lower lid: rim + ink
    c.line(Q(Lo[:int(len(Lo) * .97)]), 6.5 * k, ink, taper=(.06, .35), press=.35, seed=seed + 1, tmin=.12)
    # lower lashes: fine clusters that meet at their tips
    rl = np.random.default_rng(seed + 9); nl = len(Lo)
    def cluster(root_u, curve, nrm, tipv, L, nb, spread, w_, bow, base, hi, ln, lnw, sway=0., dy=0.):
        i0 = root_u * (len(curve) - 1); p0 = curve[int(np.clip(i0, 0, len(curve) - 1))]; tip = p0 + tipv * L
        for q_ in np.argsort(-np.abs(np.linspace(-1, 1, nb))):
            o = np.linspace(-1, 1, nb)[q_] if nb > 1 else 0.
            ii = int(np.clip(i0 + o * spread * (len(curve) - 1), 0, len(curve) - 1)); p = curve[ii] + nrm[ii] * dy
            tp = tip + np.array([rl.normal(0, .05), rl.normal(0, .05)]) * L + (p - p0) * .30; d = tp - p; Lq = np.hypot(*d) * rl.uniform(.86, 1.04)
            P0, P1 = Q(p), Q(p + d); lash(c, P0, P1 - P0, Lq * k, w_ * rl.uniform(.75, 1.2) * k, bow * side, base, ln, lnw * k, hi, seed=seed + int(rl.integers(1 << 16)))
    nlo = np.gradient(Lo, axis=0); nlo = np.c_[-nlo[:, 1], nlo[:, 0]]; nlo /= np.hypot(nlo[:, 0], nlo[:, 1])[:, None] + 1e-6; nlo *= np.sign(nlo[nl // 2, 1])
    for u, L, a, nb in ((.05, 120, 1.0, 3), (.095, 92, .85, 2), (.14, 128, .78, 3), (.19, 84, .60, 2), (.235, 112, .56, 3), (.285, 74, .42, 2), (.33, 100, .36, 3), (.385, 66, .24, 2), (.43, 88, .20, 3), (.485, 56, .08, 2), (.53, 76, .02, 3), (.585, 50, -.08, 2), (.63, 64, -.16, 2), (.69, 44, -.26, 2), (.74, 52, -.34, 2), (.81, 36, -.44, 2), (.87, 30, -.5, 1), (.07, 60, .9, 2), (.118, 54, .8, 2), (.165, 58, .66, 2), (.212, 50, .56, 2), (.26, 46, .46, 2), (.31, 44, .36, 1), (.358, 40, .28, 2), (.41, 38, .2, 1), (.458, 36, .12, 2), (.508, 32, .04, 1), (.56, 30, -.04, 2), (.61, 28, -.12, 1), (.66, 26, -.2, 1)):
        a += lash_sway * .2; cluster(u, Lo, nlo, np.array([math.sin(a), math.cos(a)]), L, nb, .011, 8.5, -.10 + .10 * u, LASH['base'], LASH['hi'], LASH['ln'], 1.3, dy=-3)
    # --- upper lid line: heavy, alive, with a wing
    nr = np.gradient(U, axis=0); nr = np.c_[nr[:, 1], -nr[:, 0]]; nr /= np.hypot(nr[:, 0], nr[:, 1])[:, None]; nr *= -np.sign(nr[n // 2, 1])
    th = np.interp(tt, [0, .05, .2, .5, .78, .93, 1], [4, 12, 24, 30, 34, 30, 14]) * (.7 + .3 * min(1, open)) * (1 + .06 * np.sin(tt * 37))
    wing = U[-1] + np.array([44, 4 + 22 * (1 - min(1, open))])
    band = np.vstack([U, [wing], (U + nr * th[:, None])[::-1]])
    # back row of upper lashes (shade), then the lid line, then the front row
    kk = .62 + .38 * min(1, max(0, open)); dr = 1 - min(1., max(0., open)); rr = np.random.default_rng(seed + 3); rl = np.random.default_rng(seed + 10)
    def tipdir(u, j):
        a = np.interp(u, [0, .12, .3, .5, .7, .86, 1], [-1.02, -.78, -.44, -.06, .44, .86, 1.20]) + j; i = int(np.clip(u, 0, 1) * (n - 1)); tg = np.array([-nr[i][1], nr[i][0]]); tg = tg * np.sign(tg[0])
        d = nr[i] * math.cos(a) + tg * math.sin(a) + np.array([.25, 1.15]) * dr * 1.15; return d / np.hypot(*d)
    def row(us, scale, base, hi, lnw, off, w_, nbs, bow, spread):
        for q_, u in enumerate(us):
            u = u + rr.normal(0, .007); L = np.interp(u, [0, .12, .3, .5, .7, .88, 1], [80, 142, 196, 232, 268, 300, 284]) * scale * rr.uniform(.74, 1.14) * kk
            cluster(float(np.clip(u, .004, .998)), U, nr, tipdir(u, rr.normal(0, .10) + lash_sway * .06 * math.sin(t * 2 + q_)), L, int(nbs[q_ % len(nbs)]), spread, w_, bow * rr.uniform(.6, 1.2) * (1 - 1.6 * dr), base, hi, LASH['ln'], lnw, dy=off)
    row(np.linspace(.02, 1.0, 27), 1.06, LASH['dp'], None, 1.0, 14, 15, (2, 3, 2), .20, .016)
    row(np.linspace(.035, .985, 24), 1.0, LASH['sh'], None, 1.2, 10, 17, (3, 2, 3, 2), .22, .017)
    # feathered white band the lashes come out of
    be = []
    for q_ in range(int(n)):
        be.append(U[q_] + nr[q_] * (th[q_] + 8 + 12 * np.sin(np.pi * tt[q_]) + 7 * abs(((q_ * .31) % 1) - .5)))
    c.fill(Q(np.vstack([U, np.array(be)[::-1]])), LASH['base'])
    row(np.linspace(.01, 1.0, 25), .94, LASH['base'], LASH['hi'], 1.5, 8, 19, (3, 2, 4, 2, 3), .24, .018)
    row(np.linspace(.06, .97, 13), .60, LASH['hi'], None, 1.1, 6, 12, (2, 1, 2), .2, .014)
    c.fill(Q(band), ink)
    # soft lower edge of the lid line: tiny hairs of ink
    for q_ in range(26):
        i = int(rr.uniform(.06, .96) * (n - 1)); a = U[i]; c.line(Q(np.array([a + nr[i] * 4, a - nr[i] * 3, a - nr[i] * rr.uniform(7, 13) + (rr.normal(0, 4), 0)])), 2.2 * k, ink, taper=(.1, .8), seed=seed + q_)
    return c.layer(h, w)


def aperture(w=1920, h=1080, open=1., centre=None, scale=1., roll=-.06, side=1):
    """float mask of the opening between the lids (same placement arguments as six_eye): where the eye can shine"""
    k = scale * w / 1920.; ctr = np.array([w * .5, h * .545]) if centre is None else np.asarray(centre, float); cs, sn = math.cos(roll), math.sin(roll)
    U_, L_ = lids(open); P = np.vstack([spl(U_, n=16), spl(L_, n=16)[1:]]) * (side, 1)
    Q = np.stack([P[:, 0] * cs - P[:, 1] * sn, P[:, 0] * sn + P[:, 1] * cs], -1) * k + ctr
    return kit.mask_of(Q, h, w, 1)
