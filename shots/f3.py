"""f3  KIRE: hair-thin white cuts; the city leans apart in clean pieces; Mahoraga walks through them, loses the sword arm, the wheel turns, the arm is back."""
import numpy as np, cv2, math
from jj import W, H, comp, kit, fx
from jj.paint import rgb, f32, fbm, smooth, grid
from jj.comp import ease, blur, over
from jj.cam import Camera3D
from jj.ink import Cel
from chars import mahoraga as M, maho_shot as MS, maho_fight as F
from chars.maho_body import V, unit
from chars.maho_ink import Pen

ID = 'f3'; VER = dict(d_bg=4, suk=4, d_tower=8, d_sec=3, d_rub=2, d_hz=2, d_cbg=8, d_mb=12, d_mc=7, d_wellc=6, d_wc=7)
HITS = (0, 18, 36, 45, 54)
CA = Camera3D((-5.2, -27, 2.6), (4.2, 20, 10.2), 1240); CB = Camera3D((-.75, -4.5, .55), (-.95, 8, 3.75), 1180, cy=575); CC = Camera3D((.45, -4.7, 1.25), (-1.02, 0, 2.80), 1600, roll=-.05)
TA = [(-4, 1.5, 62, 76, 46, 14), (-30, -15, 44, 58, 34, 15), (24, 38, 40, 56, 34, 16), (3.0, 15.5, 16, 31, 33, 11), (-17.5, -8.5, 30, 42, 23, 12), (17.5, 28, 20, 34, 24, 13)]
TB = [(-17, -7.2, 40, 52, 30, 1), (7.4, 18, 44, 56, 36, 2), (-20, -7.0, 22, 37, 24, 3), (7.0, 17, 24, 38, 28, 4), (-18, -6.6, 6, 19, 21, 5), (6.8, 16, 7, 21, 25, 6)]
CUT_A = ((-60, 700), (1980, 300)); CUT_B1 = ((-60, 250), (1980, 505)); CUT_B2 = ((1980, 165), (-60, 640)); CUT_C = ((-60, 392), (1980, 560))
WIDE = dict(wing=-.10, spread=.62, fan=1.1)


# ------------------------------------------------------------------ poses
def walk(u):
    """u in steps: contact poses on integers; planted foot stays planted"""
    k = int(math.floor(u)); t = u - k; a, b = (M.KEYS['stride'], M.KEYS['stride2']) if k % 2 == 0 else (M.KEYS['stride2'], M.KEYS['stride']); p = M.lerp_pose(a, b, t); fr, bk = ('rf', 'lf') if k % 2 == 0 else ('lf', 'rf'); s = math.sin(math.pi * t)
    sw = dict(p[bk]); q = list(sw['t']); q[2] = q[2] + .30 * s; sw['t'] = q; sw['pitch'] = float(np.asarray(sw['pitch'])) + .25 * s; p[bk] = sw; r = list(np.asarray(p['root'], float)); r[2] += .07 * s; p['root'] = r; p.update(WIDE); return p


def maho_b(f):
    u = (f - 18) / 9.; y = 8.3 - 1.56 * u; p = walk(u); p = M.merge(p, neck=(p['neck'][0], .40, p['neck'][2]), head=(p['head'][0] * .6, -.64, p['head'][2]), jaw=.10, fan=1.5, spread=.60); return M.Maho(p, pos=(2.45 - .30 * u, y, 0), heading=-.26, scale=1.0, wheel='burn', lod=0)


ARM = dict(M.KEYS['cut'], cut=None, root=(.03, .02, 1.46), chest=(-.26, .04, -.06), shr=(-.02, .0, .10), rh=M._hand((-1.28, -.42, 2.66), (-1.5, .5, 1.9), (-.35, -.30, .9), (-.8, .2, .3), .95, .1, .8),
           lh=M._hand((.92, -.46, 1.62), (1.5, .6, 1.8), (.15, -.7, -.6), (.6, -.2, .7), (.5, .42, .40, .48), .9, .25), lf=M._foot((.36, .30, .13), (.5, -2, .9), (.4, -1, 0)), rf=M._foot((-.36, -.36, .13), (-.4, -2, .9), (-.2, -1, 0)), wing=-.10, spread=.50, fan=1.1)
HEADS = dict(a=dict(neck=(.05, .32, .02), head=(.05, -.30, .05), jaw=.05), b=dict(neck=(-.10, .34, -.04), head=(-.18, -.24, -.06), jaw=.03), c=dict(neck=(-.16, .40, -.08), head=(-.34, -.14, -.12), jaw=.03),
             d=dict(neck=(-.02, .34, .04), head=(-.02, -.36, .12), jaw=.10), e=dict(neck=(.04, .32, .08), head=(.08, -.42, .20), jaw=.24))
HF = [(45, 'a'), (50, 'b'), (52, 'c'), (60, 'b'), (63, 'd'), (68, 'e')]
def head_at(f): return [h for t, h in HF if f >= t][-1]


def maho_c(head, cut):
    return M.Maho(M.merge(ARM, cut=.3 if cut else None, **HEADS[head]), pos=(0, 0, 0), heading=-.10, wheel='burn')


def wheel_c(f):
    m = maho_c(head_at(f), False); wh = dict(m.wheel); wh['spin'] = M.notch(f - 54); le = math.radians(4) * ease((f - 49) / 4.) if f < 54 else 0.
    ca, sa = math.cos(le), math.sin(le); ey, ez = wh['ey'], wh['ez']; wh['ey'] = ey * ca - ez * sa; wh['ez'] = ez * ca + ey * sa; return wh


# ------------------------------------------------------------------ cached drawings
def d_bg(s):
    if s == 'A': c, sd, cx = CA, 7, 1250
    elif s == 'B': c, sd, cx = CB, 3, 1000
    else: c, sd, cx = CC, 5, 700
    hz = float(c.project(np.array([c.eye + np.array([c.R[2][0], c.R[2][1], 0]) * 4000]))[0][0][1]); im = F.sky(sd, hz=hz + 8, cx=cx).copy(); im = F.ground_ink(im, c, seed=sd)
    if s == 'B': im = F.cracks(im, c, (.6, 5.5, 0), n=9, reach=9, seed=3, w=.07); im = F.cracks(im, c, (-1.4, .2, 0), n=7, reach=4, seed=5, w=.04)
    return im


def d_tower(s, i):
    c, T = (CA, TA) if s == 'A' else (CB, TB); t = F.Tower(*T[i]); d = t.draw(c, scale=.55, I=1.0, lit=.035, lw=.66, light=(.80, -.30, -.45) if s == 'A' else (.85, -.12, -.42)); z = d['z']; return F.haze(d['layer'], z, start=20 if s == 'A' else 14, dens=.0105 if s == 'A' else .016)


CUTS = dict(A={3: CUT_A, 5: CUT_A, 0: CUT_A}, B={2: CUT_B1, 4: CUT_B1, 0: CUT_B1, 3: CUT_B2, 5: CUT_B2, 1: CUT_B2})
def d_sec(s, i):
    c, T = (CA, TA) if s == 'A' else (CB, TB); ln = CUTS[s][i]; ln = ln if ln[0][0] < ln[1][0] else ln[::-1]; r = F.section(F.Tower(*T[i]), c, *ln); return r if r is not None else np.zeros((H, W, 4), f32)


def d_rub():
    d = F.Rubble([(-5.8, 5, .9, 18, .85), (6.0, 6.5, .7, 18, .80), (-4.2, 13, 1.6, 10, .7), (3.9, -1.6, .4, 5, .55), (-3.9, -1.8, .4, 5, .5)], 7).draw(CB, scale=.55); return F.haze(d['layer'], d['z'], start=14, dens=.016)


def d_hz(s):
    if s == 'A': return F.wisps(330, 640, 640, 330, '#ea6c54', .80, 5, 30)
    p = CB.project(np.array([[-2.85, 1.2, 1.2]]))[0][0]; return F.wisps(p[0], p[1] + 20, 400, 470, '#ea6c54', .85, 8, 22)


def d_cbg():
    """medium shot plate: hot sky, sliced towers leaning, out of focus"""
    im = d_bg('C'); c = CC
    for i, (T, p0, p1, sl, gp, ro) in enumerate((((-22, -11, 26, 40, 30, 21), (0, 520), (900, 300), 70, 10, .05), ((7.5, 18, 22, 36, 28, 22), (2000, 360), (900, 640), 90, 16, -.07), ((-9, -2.5, 40, 54, 40, 23), (0, 250), (1100, 420), -60, 8, -.04), ((14, 26, 12, 22, 22, 24), (1980, 220), (1100, 500), 60, 12, -.05))):
        t = F.Tower(*T); d = t.draw(c, scale=.5, I=1.0, lit=.03, lw=.6, light=(.80, -.30, -.45)); L = F.haze(d['layer'], d['z'], start=10, dens=.016); sc = F.section(t, c, *((p0, p1) if p0[0] < p1[0] else (p1, p0))); u, B, A = F.part(L, p0, p1, slide=sl, gap=gp, rot=ro, face=14, sec=sc)
        for q in (u, B, A):
            if q is not None: im = q.over(im)
    return F.calm(im, .45, sig=4.5)


def d_mb(f):
    m = maho_b(f); mo = m.w(m.jawf(0, -.16, -.06)); d = F.maho(m, CB, scale=.8, lw=.72, teeth=.42, lights=F.key_lights(m, key=1.0) + [dict(kind='point', pos=mo + np.array([-.35, -1.0, -.25]), I=.16, d0=.40, shadow=False, rim=.2, bounce=0)]); w = F.well(m, CB, 2); li = M.burn_wheel(dict(m.wheel, spin=0.), CB, seed=f)[0][..., 1]; z = np.where(d['alpha'] > .5, d['z'], 0).astype(f32); return dict(l=over_rgba(w, d['layer']), li=li)


def over_rgba(a, b): k = b[..., 3:4]; return a * (1 - k) + b


def d_mc(head, cut, arm):
    m = maho_c(head, cut); d = F.maho(m, CC, scale=.6, lw=1.0, lights=F.key_lights(m, key=1.05, d=(.75, .50, -.55))); r = dict(l=d['layer'])
    if arm: s, tot = M.arm_map(d, m); r['s'] = s; r['tot'] = np.full((2, 2), tot, f32)
    return r


def d_wellc(h): return F.well(maho_c(h, False), CC, 3)
def d_wc(f):
    k = 1 + (1.5 * math.exp(-(f - 54) / 2.6) if f >= 54 else 0); return M.burn_wheel(wheel_c(f), CC, seed=f, core=.55 * k ** .5, width=.95 + .10 * (k - 1))[0][..., 1]


def suk(s, k):
    """Sukuna's back, tiny: k = 0 arm at rest, 1 the finger has moved"""
    from jj.figure import Fig, P
    p = P(root=(0, 0, 0), pel=(0, 0, 4), cst=(-2, 0, 4), nck=(2, 0, 0), hed=(-2, 2, 4), hR=(.30, .42, .06) if k else (.30, .34, -.22), eR=(.8, 0, -1), gR='point' if k else 'relax', wR=(120, 0, 0), hL=(.14, .04, -.95), fL=(.075, .02, .043), fR=(.085, -.01, .043))
    if s == 'A': F_ = Fig('sukuna').pose(p).place(-8.6, -12.5, 2.62, heading=-.25); c = CA
    else: F_ = Fig('sukuna').pose(p).place(-2.85, 1.2, .0, heading=-.62); c = CB
    F_.blow(.9 if s == 'A' else .7, .1, .1); L = F_.draw(c, 3 + 2 * k, light=(-.5, .6, .3)); return F.silhouette(L, '#080406', '#d3262c', -4, 2) if s == 'A' else F.silhouette(L, '#080406', '#c8202a', -5, 2)


JOBS = {f'bg{s}': ('d_bg', (s,)) for s in 'AB'}
JOBS['cbg'] = ('d_cbg', ()); JOBS['rub'] = ('d_rub', ()); JOBS['hzA'] = ('d_hz', ('A',)); JOBS['hzB'] = ('d_hz', ('B',)); JOBS.update({f'S{s_}{i}': ('d_sec', (s_, i)) for s_ in 'AB' for i in CUTS[s_]}); JOBS.update({f'wellc_{h}': ('d_wellc', (h,)) for h in 'abcde'})
JOBS.update({f'T{s}{i}': ('d_tower', (s, i)) for s in 'AB' for i in range(6)})
JOBS.update({f'mb{f}': ('d_mb', (f,)) for f in range(18, 45, 2)})
JOBS.update({f'mc_{h}{int(c)}': ('d_mc', (h, c, not c)) for h, c in (('a', 0), ('a', 1), ('b', 1), ('c', 1), ('b', 0), ('c', 0), ('d', 0), ('e', 0), ('d', 1))})
WC = [45, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 60, 62, 63, 65, 67, 68, 70]
JOBS.update({f'wc{f}': ('d_wc', (f,)) for f in WC})
JOBS.update({f'suk{s}{k}': ('suk', (s, k)) for s in 'AB' for k in (0, 1)})


def stack(im, bits):
    for q in bits:
        if q is not None: q.over(im)
    return im


def weight(t, p=2.1): return float(np.clip(t, 0, 2)) ** p


class Shot:
    id = ID; n = 72
    finish = dict(bloom=.26, vig=.20, grain=.013)
    EVENTS = [(0, 'cut', dict(size=1.0, note='hair-thin white cut, whole frame, then 2 frames of silence')), (3, 'stone_slide', dict(size=.9, dur=14)), (12, 'rubble', dict(size=.6)),
              (18, 'cut', dict(size=1.0)), (21, 'stone_slide', dict(size=1.0, dur=18)), (27, 'step', dict(size=.6, note='Mahoraga, 3 m, bare feet')), (34, 'rubble', dict(size=.9)), (36, 'cut', dict(size=1.0)), (36, 'step', dict(size=.7)), (39, 'stone_slide', dict(size=1.0, dur=8)),
              (45, 'cut', dict(size=1.2, note='the sword arm')), (45, 'step', dict(size=.8)), (48, 'arm_off', dict(note='ink, wet tear, no gore')), (50, 'creak', dict(note='wheel leans back')), (54, 'gakon', dict(size=1.0)), (55, 'knit', dict(dur=11, note='ink threads pull the arm together')), (66, 'snap', dict(size=.6, note='arm whole')), (66, 'grin', {})]

    def setup(self):
        MS.ensure(ID); x, y = grid(); self.x, self.y = x, y; n = fbm(H, W, (160, 60, 22, 9), (1, .6, .35, .2), 21); self.nz = n
        # black mound for the wide shot
        pts = [(-40, 1120), (-40, 800), (60, 768), (150, 742), (236, 716), (318, 704), (400, 706), (470, 730), (520, 768), (590, 800), (660, 872), (760, 905), (860, 968), (980, 1010), (1120, 1120)]; c = Cel(); c.fill(np.array(pts, float), '#080406')
        r = np.random.default_rng(3); pen = Pen()
        for i in range(60): k = r.integers(1, len(pts) - 2); p = np.array(pts[k]) * (1 - .5) + np.array(pts[k + 1]) * .5 + r.normal(0, 16, 2) + np.array([0, 14]); pen.stroke(np.array([p, p + np.array([r.uniform(-60, 90), r.uniform(-8, 46)])]), r.uniform(5, 22), i, taper=(.2, .6), press=.6)
        K = pen.raster(H, W); L = c.layer(); a = np.maximum(L[..., 3], K); self.mound = np.dstack([rgb('#080406') * a[..., None], a]).astype(f32)
        m = np.clip(a - kit.shifted(a, -2, 4), 0, 1) * smooth(-.2, .8, n); self.mound[..., :3] += (m * .75)[..., None] * rgb('#c3222a')
        self.mc = maho_c('a', False); self.tot = float(MS.val(ID, 'mc_a0', 'tot')[0, 0]); self.foot = {}
        for t0 in (27, 36): m_ = maho_b(t0); self.foot[t0] = CB.project(m_.J['A-1' if (t0 - 18) // 9 % 2 == 0 else 'A1'][None])[0][0]
        del self.x, self.y; return self

    # ---------------------------------------------------------- A: the wide cut
    def A(self, f):
        im = MS.val(ID, 'bgA'); t = (f - 3) / 14.; P = {3: (430, 12, .13, 130, 0.), 5: (300, 10, .12, 90, .10), 0: (150, 5, .05, 20, .22)}
        for i in range(6):
            L = MS.get(ID, f'TA{i}')
            if i in P and t > 0:
                sl, gp, ro, fa, dl = P[i]; e = weight(max(0, t - dl)); stack(im, F.part(L, *CUT_A, slide=-sl * e, gap=gp * min(1, t * 4), rot=-ro * e, fall=fa * e, face=11, sec=MS.get(ID, f'SA{i}')))
            else: L.over(im)
        if f >= 6:
            cel = Cel(ss=1)
            for j, (x, y, s) in enumerate(((1010, 490, 46), (1420, 420, 54), (1230, 450, 40))): fx.smoke_burst(cel, f - 6, x, y, s, seed=30 + j, n=3, ang=(.4, 2.7), speed=110, life=1.2, pal=F.DUSTK, lw=1.4) if j < 2 else None
            fx.debris(cel, f - 5, (1240, 470), n=8, seed=4, speed=(0, 60), ang=(.9, 2.2), size=(16, 52), grav=2100, pal=F.ROCKB, spread=(420, 20), lw=1.6, trails=False, drag=.1, spin=2.5); im = over(im, cel.layer())
        im = F.calm(im, .26, sig=1.0); MS.get(ID, 'hzA').over(im, .95); im = over(im, self.mound); sk = MS.get(ID, f'sukA{int(f >= 1)}'); F.halo(im, sk, '#e88470', 54, .55); sk.over(im)
        if f < 4: im = F.cut_line(im, *CUT_A, f)
        return im

    # ---------------------------------------------------------- B: through the falling pieces
    def B(self, f):
        im = MS.val(ID, 'bgB'); t1 = (f - 21) / 20.; t2 = (f - 39) / 13.; late = []
        C1 = {2: (300, 14, .16, 240, 0.), 4: (210, 10, .10, 160, .12), 0: (120, 6, .06, 60, .25)}; C2 = {3: (260, 12, .12, 190, .0), 5: (260, 14, .13, 240, .05), 1: (120, 6, .05, 60, .2)}
        for i in range(6):
            L = MS.get(ID, f'TB{i}'); A_ = u_ = None
            if i in C1 and t1 > 0: sl, gp, ro, fa, dl = C1[i]; e = weight(max(0, t1 - dl), 2.2); u_, B, A_ = F.part(L, *CUT_B1, slide=sl * e, gap=gp * min(1, t1 * 5), rot=ro * e, fall=fa * e, face=14, sec=MS.get(ID, f'SB{i}'))
            elif i in C2 and t2 > 0: sl, gp, ro, fa, dl = C2[i]; e = weight(max(0, t2 - dl), 2.0); u_, B, A_ = F.part(L, *CUT_B2, slide=sl * e, gap=gp * min(1, t2 * 5), rot=-ro * e, fall=fa * e, face=14, side=-1, sec=MS.get(ID, f'SB{i}'))
            else: B = L
            if u_ is not None: late.append(u_) if i >= 4 else u_.over(im)
            B.over(im)
            if A_ is not None: late.append(A_) if i >= 4 else A_.over(im)
        MS.get(ID, 'rub').over(im)
        for A_ in late: A_.over(im)
        cel = Cel(ss=1); any_ = False                                           # few, big, accelerating
        for t0, x, y, s, sd in ((31, 330, 900, 170, 2), (42, 1690, 860, 130, 3)):
            if f >= t0: any_ = True; fx.smoke_burst(cel, f - t0, x, y, s, seed=sd * 9, n=4, ang=(-2.8, -.35), speed=300, life=1.7, pal=F.DUSTK, lw=2.0)
        if f >= 22: any_ = True; fx.debris(cel, f - 22, (430, 330), n=6, seed=11, speed=(0, 60), ang=(.8, 2.0), size=(40, 110), grav=2400, pal=F.ROCKB, spread=(260, 50), lw=2.2, trails=False, drag=.1, spin=2.2)
        if f >= 40: fx.debris(cel, f - 40, (1680, 300), n=5, seed=12, speed=(0, 60), ang=(1.2, 2.4), size=(40, 100), grav=2400, pal=F.ROCKB, spread=(160, 50), lw=2.2, trails=False, drag=.1, spin=2.2)
        if any_: im = over(im, cel.layer())
        im = F.calm(im, .45, sig=2.2); MS.get(ID, 'hzB').over(im, .55); d = MS.get(ID, 'mb%d' % (f // 2 * 2)); F.halo(im, d['l'], amt=.5); d['l'].over(im); im = F.hot(im, d['li'], .94 + .06 * math.sin(f * 2.1))
        st = [t0 for t0 in (27, 36) if 0 <= f - t0 < 12]
        if st:
            cel = Cel(ss=1)
            for t0 in st: p = self.foot[t0]; fx.dust_skirt(cel, f - t0, p[0], p[1] + 10, 100, 22, 40, seed=t0, n=6, pal=F.DUSTK, life=.5, lw=1.5)
            im = over(im, cel.layer())
        sk = MS.get(ID, f'sukB{int(any(h <= f < h + 5 for h in (18, 36)))}'); F.halo(im, sk, '#d9685a', 70, .55); sk.over(im)
        for h, ln in ((18, CUT_B1), (36, CUT_B2)):
            if h <= f < h + 4: im = F.cut_line(im, *ln, f - h)
        return im

    # ---------------------------------------------------------- C: the arm
    def C(self, f):
        im = MS.val(ID, 'cbg'); hd = head_at(f); MS.get(ID, f'wellc_{hd}').over(im); li = MS.get(ID, f'wc{max(w for w in WC if w <= f)}'); m = self.mc; kt = dict(throw=(-520, -260), spin=-1.5)
        g = lambda h, c: MS.get(ID, f'mc_{h}{c}')
        if f < 48: g('a', 0)['l'].over(im)
        elif f < 55:
            fl = g('a', 0); t = (f - 48) / 13.
            for l in M.sever(dict(layer=fl['l'].full()), dict(layer=g(hd, 1)['l'].full()), m, CC, max(.02, t), seed=2, s=(fl['s'].full(), self.tot), **kt): im = over(im, l)
        elif f < 66:
            if f < 60:
                fl = g('a', 0); im = over(im, M.sever(dict(layer=fl['l'].full()), dict(layer=np.zeros((H, W, 4), f32)), m, CC, (f - 48) / 13., seed=2, s=(fl['s'].full(), self.tot), **kt)[1])
            fl = g(hd, 0); im = over(im, M.regrow(dict(layer=fl['l'].full()), dict(layer=g(hd, 1)['l'].full()), m, CC, (f - 54) / 11., seed=f // 2, s=(fl['s'].full(), self.tot)))
        else: g(hd, 0)['l'].over(im)
        im = F.hot(im, li, (1.2 if f in (54, 55) else 1.) * (.94 + .06 * math.sin(f * 2.1)))
        if f >= 54:
            p = CC.project(m.wheel['c'][None])[0][0]; e = math.exp(-(f - 54) / 2.4)
            if e > .05: im = kit.flare(im, p[0], p[1], 640, '#e6eefc', .40 * e, .4)
        if 48 <= f < 72:
            cel = Cel(ss=1); p = CC.project((m.J['S-1'] + unit(m.J['E-1'] - m.J['S-1']) * .3)[None])[0][0]; w = CC.project(m.J['W-1'][None])[0][0]
            if f < 62: F.ink_spatter(cel, f - 48, p[0], p[1], n=80, seed=6, speed=32, ang=math.pi * .95, spread=.7, rmax=12)
            if f >= 66: F.ink_spatter(cel, f - 66, w[0], w[1], n=30, seed=8, speed=24, rmax=7, grav=1.2)
            im = over(im, cel.layer())
        if 45 <= f < 48: im = F.cut_line(im, *CUT_C, f - 45)
        return im

    def frame(self, f):
        im = self.A(f) if f < 18 else self.B(f) if f < 45 else self.C(f)
        z = 1.; dx = dy = 0.; cx, cy = W / 2, H / 2; rot = 0.
        if f < 18: z = 1.02 + .03 * ease(f / 18.); dx, dy = kit.shake(f, 5, 2, 8, 8)
        elif f < 45: z = 1.02 + .05 * (f - 18) / 27.; cx, cy = 960, 560
        else: z = 1.03 + .05 * ease((f - 45) / 27.) + .34 * ease(max(0., (f - 62) / 9.)); cx, cy = 1060 + 260 * ease(max(0., (f - 60) / 8.)), 470 + 190 * ease(max(0., (f - 60) / 8.))
        for h in (27, 36): 
            if h <= f < h + 4: a = kit.shake(f, (9, 6, 3, 1)[f - h], h); dx += a[0]; dy += a[1]
        for h in (3, 21, 39):
            if h <= f < h + 5: a = kit.shake(f, (7, 6, 4, 3, 2)[f - h], h); dx += a[0]; dy += a[1]
        if 54 <= f <= 58: a = kit.shake(f, (24, 18, 11, 5, 2)[f - 54], 9); dx += a[0]; dy += a[1]; z *= (1.035, 1.022, 1.01, 1.0, 1.0)[f - 54]
        if 48 <= f <= 50: a = kit.shake(f, (10, 6, 3)[f - 48], 4); dx += a[0]; dy += a[1]
        c0, c1 = cx + (W / 2 - cx) / z, cy + (H / 2 - cy) / z
        return comp.cam2d(im.astype(f32), c0 + dx, c1 + dy, z, rot).astype(f32)
