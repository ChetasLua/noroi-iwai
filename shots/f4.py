"""f4  MAWARE: Sukuna vs Mahoraga, four cuts of 18.  (1) the sweep and the duck (2) the kick through a building (3) its palm holds a slash, his eye (4) the wheel, both lunge."""
import numpy as np, cv2, math
from jj import W, H, comp, kit, fx
from jj.paint import rgb, f32, fbm, smooth, grid
from jj.comp import ease, blur, over
from jj.cam import Camera3D
from jj.ink import Cel
from jj.figure import Fig, P
from jj.fig_pose import POSES, mix_pose
from chars import mahoraga as M, maho_shot as MS, maho_fight as F
from chars.maho_body import V, unit
from chars.maho_ink import Pen
from jj.ink import spl

ID = 'f4'; VER = dict(d_tr=15, d_m=13, d_suk=11, d_bg=12, d_wall=12, d_wsec=3, d_eye=8, d_w4=6, d_mw=5, d_bgw=5)
SUKPAL = dict(ink='#0b090c', mats=dict(bone=('#fff8ea', '#efe6d3', '#978784', '#33272b'), obi=('#8a1a22', '#18131a', '#0d0a0e', '#060406'), skin=('#ffeed8', '#e6cdb4', '#9c7a70', '#4a3034'), slate=('#5a1a20', '#1a1216', '#0e0a0d', '#060405'), tabi=('#3a2a2e', '#151115', '#0b090b', '#050405')))
WIDE = dict(wing=-.05, spread=.50, fan=1.1)
C1 = Camera3D((-.15, -6.3, .55), (.05, 0, 1.66), 1600, roll=-.04); C2 = Camera3D((-1.3, -6.2, .75), (.75, .3, 1.80), 2050, roll=-.10); C4 = Camera3D((.2, -9.4, 1.0), (-.05, 0, 2.15), 2250)


# ------------------------------------------------------------------ Mahoraga
def kp(*keys):
    """pose through keys [(u, pose)], u 0..1"""
    def at(u):
        if u <= keys[0][0]: return keys[0][1]
        for (a, pa), (b, pb) in zip(keys[:-1], keys[1:]):
            if u < b: return M.lerp_pose(pa, pb, (u - a) / (b - a))
        return keys[-1][1]
    return at


W1 = dict(wing=-.05, spread=.5, fan=1.2)
E0 = dict(root=(0, .38, 1.34), pelvis=(-.30, .02, -.06), waist=(-.18, -.06, -.04), chest=(-.50, -.12, -.12), neck=(.40, .34, 0), head=(.45, -.30, .04), shl=(0, -.14, 0), shr=(.02, .06, .12),
          lh=M._hand((.55, -1.05, 1.90), (1.4, -.3, 1.4), (-.1, -.9, .2), (.3, .2, .9), (.35, .3, .3, .4), 1., .2), rh=M._hand((-.75, .80, 3.0), (-1.1, 1.2, 2.2), (-.1, .7, .7), (-.6, -.2, .6), .95, .1, .8),
          lf=M._foot((.38, -.50, .13), (.6, -2, .9), (.2, -1, 0)), rf=M._foot((-.42, .78, .2), (-.8, -1.4, .7), (-.4, -.9, 0), .35), jaw=.14, flare=(0, -.03, 0), sash=(.1, .25, 0), **W1)
E0b = M.merge(E0, root=(0, .46, 1.26), chest=(-.58, -.18, -.14), jaw=.08)
E1 = dict(E0, root=(0, -.05, 1.28), pelvis=(0, .14, 0), waist=(0, .08, 0), chest=(-.05, .10, 0), neck=(.05, .38, 0), head=(.05, -.42, 0), rh=M._hand((-.50, -.55, 3.12), (-1.0, .4, 2.3), (-.1, -.4, .9), (-.4, .2, .8), .95, .1, .8),
          lh=M._hand((.95, -.30, 1.9), (1.5, .5, 1.5), (.3, -.7, -.3), (.3, .5, .6), .4, .9, .2), jaw=.4, flare=(0, .14, .04), sash=(.1, .8, .1))
E2 = dict(E1, root=(0, -.58, 1.04), pelvis=(.30, .36, .04), waist=(.14, .20, .04), chest=(.38, .34, .08), neck=(-.25, .28, 0), head=(-.40, -.66, -.02), shr=(.04, -.14, -.02),
          rh=M._hand((-.50, -2.5, 1.24), (-.9, .2, 2.0), (-.1, -1, 0), (-.2, .2, .9), .95, .1, .8), lh=M._hand((.98, .62, 1.95), (1.5, 1.2, 1.5), (.5, .6, .2), (.4, .6, .6), .4, .9, .2),
          lf=M._foot((.40, .88, .26), (.7, -1, .6), (.3, -1, 0), .65), rf=M._foot((-.32, -1.0, .13), (-.5, -2.6, .8), (-.1, -1, 0)), jaw=.52, flare=(0, .26, .05), sash=(.1, 1.15, .2))
E3 = dict(E2, root=(.02, -.60, 1.0), pelvis=(.40, .38, .06), waist=(.18, .22, .05), chest=(.48, .38, .10), head=(-.46, -.64, -.03), rh=M._hand((.10, -2.1, .85), (-.6, -.2, 1.6), (.2, -.9, -.4), (-.1, .4, .9), .95, .1, .8), jaw=.30, flare=(0, .06, .02), sash=(.1, .45, .0))
SWEEP = kp((0, E0), (.16, E0b), (.55, E1), (1.0, E2), (1.5, E3))
U1 = {0: 0, 2: .14, 4: .40, 5: .62, 6: .84, 7: 1.0, 11: 1.12, 13: 1.28, 15: 1.42, 17: 1.5}; T1 = list(U1); P1, H1 = (1.45, .25, 0), -1.2
def m1(f): return M.Maho(SWEEP(U1[f]), pos=P1, heading=H1, wheel='burn', lod=0)


GUARD = M.merge(M.KEYS['sweep_f'], **WIDE, rh=dict(t=(-.6, -.9, 2.2)), jaw=.1)
W2 = dict(wing=.5, spread=.2, fan=1.2)
F0 = dict(root=(0, .30, 1.42), pelvis=(.10, .42, 0), waist=(.05, .42, 0), chest=(.10, .34, .05), neck=(0, .25, 0), head=(.05, -.10, .05), crest=(0, -.75, 0), shl=(0, -.14, -.02), shr=(0, -.14, -.02), jaw=.62,
          lh=M._hand((.95, -1.20, 1.55), (1.6, -.2, 1.2), (.4, -.9, 0), (.4, .2, .8), (.3, .25, .25, .3), .9, .2), rh=M._hand((-1.05, -1.10, 1.75), (-1.6, -.2, 1.2), (-.3, -.9, .2), (-.5, .2, .7), .9, .2, .7),
          lf=M._foot((.36, -.62, .42), (.6, -2, .9), (.3, -1, .2), -.3), rf=M._foot((-.34, -.50, .22), (-.5, -2, .9), (-.3, -1, 0), -.2), flare=(0, -.12, .03), sash=(.1, -.9, .1), **W2)
F1 = dict(F0, root=(0, .55, 1.62), pelvis=(.18, .55, .12), waist=(.05, .48, .05), chest=(.10, .38, .10), head=(.1, -.05, .1), crest=(0, -.95, .1),
          lh=M._hand((.80, -1.45, 1.85), (1.6, -.6, 1.4), (.2, -.9, .3), (.3, .2, .9), (.25, .2, .2, .3), 1., .2), rh=M._hand((-.85, -1.40, 2.05), (-1.6, -.6, 1.4), (-.2, -.9, .4), (-.4, .2, .8), .9, .2, .7),
          lf=M._foot((.40, -1.05, 1.05), (.6, -2, 1.6), (.3, -.8, .5), -.6), rf=M._foot((-.36, -.95, .70), (-.5, -2, 1.4), (-.3, -.8, .4), -.5), flare=(0, -.2, .05), sash=(.1, -1.2, .3))
T2 = [18, 21, 24, 27, 29, 30, 31, 32, 34]
def m2(f):
    if f < 27: p, x, y, z, h = (GUARD if f < 24 else M.lerp_pose(GUARD, F0, .12)), 1.30, .50, 0, -.95
    elif f < 29: p, x, y, z, h = F0, 1.18, .55, 0, -1.05
    else: u = (f - 28) / 6.; p = M.lerp_pose(F0, F1, min(1, u * 3)); x, y, z, h = 1.18 + 3.4 * u ** .85, .55 + 2.3 * u, .15 + .75 * math.sin(min(1, u) * 2.), -1.05 - .10 * u
    return M.Maho(p, pos=(x, y, z), heading=h, wheel='burn', lod=0 if f < 31 else 1)


PALM = M.merge(M.KEYS['palm'], **WIDE, pelvis=(-.15, .15, 0), waist=(-.05, .10, 0), chest=(-.30, .18, 0), rh=dict(t=(-.75, .85, 1.45), pole=(-1.2, 1.4, 1.9)), lh=dict(t=(-.10, -1.9, 1.72), pole=(1.3, -.6, 1.1), fwd=(.04, -.14, 1), dor=(0, 1, .15), curl=(.16, .10, .12, .20), spread=1.1, thumb=.1), head=(.10, -.42, -.04), neck=(.05, .36, 0), jaw=.05)
def m3(k):
    c = (0., .55, 1.)[k] if k < 3 else 1.; cu = np.array((.16, .10, .12, .20)) * (1 - c) + np.array((.74, .80, .84, .86)) * c; return M.Maho(M.merge(PALM, lh=dict(curl=tuple(cu), spread=1.1 - .8 * c, thumb=.1 + .7 * c), jaw=.04 if k < 3 else .16), pos=(0, 0, 0), heading=-.38, wheel='burn')


LU0 = M.merge(M.KEYS['lunge'], **WIDE, root=(0, -.1, 1.12), pelvis=(.1, .30, .04), lh=dict(t=(.80, -.9, 1.5)), rh=dict(t=(-1.2, .30, 2.3)), jaw=.2); LU1 = M.merge(M.KEYS['lunge'], **WIDE, rh=dict(t=(-.75, -1.62, 1.75), pole=(-1.7, -.6, 1.0), fwd=(-.1, -1, 0)), lh=dict(t=(1.0, -.55, 1.7), pole=(1.6, .6, 1.5), fwd=(.4, -.6, -.4)), root=(0, -.6, 1.45), lf=dict(t=(.40, .95, .75), pitch=1.0), rf=dict(t=(-.30, -.55, .62), pitch=.3), jaw=.6)
T4 = [59, 61, 63, 65, 67]
def m4(f):
    u = (f - 59) / 8.; e = ease(u); p = M.lerp_pose(LU0, LU1, min(1, u * 1.5)); return M.Maho(p, pos=(2.85 - .95 * e ** 1.3, .25, .55 * math.sin(math.pi * .6 * e)), heading=-1.42 + .40 * e, wheel='burn', lod=1)


# ------------------------------------------------------------------ Sukuna
DUCK = POSES['duck']
KW = P('kick_wind', pel=(10, -10, -40), wst=(10, -4, -24), cst=(12, -4, -24), nck=(4, 2, 30), hed=(0, 2, 26), root=(0, 0, -.10), hL=(.30, .10, .10), hR=(-.10, .40, .05))
KC = P('kick_wind', pel=(-6, -22, 40), wst=(6, 4, 6), cst=(10, 6, 4), fR=(.26, .20, .42))
K1 = P('kick', pel=(-22, -50, 86), wst=(8, 10, -24), cst=(12, 16, -34), nck=(6, 18, -26), hed=(2, 14, -22), fR=(.02, .56, .70), tR=(-60, -70, -10), hL=(.10, .30, .36), hR=(.66, -.56, -.20))
K2 = P(K1, pel=(-28, -40, 104), fR=(.10, .52, .60), hL=(.2, .2, .5), hR=(.7, -.5, .1)); K3 = P('kick_wind', pel=(-2, -14, 70), wst=(6, 0, -10), cst=(10, 4, -20), fR=(.24, .16, .34), hR=(.5, -.3, -.2))
S1 = (0, 2, 3, 4, 6, 8, 10, 12, 14, 16); S2 = (18, 20, 22, 24, 26, 27, 29, 31, 33, 35)
def s1(f):
    pose = POSES['guard'] if f < 3 else mix_pose(POSES['guard'], DUCK, .6) if f < 4 else DUCK if f < 12 else mix_pose(DUCK, KW, (f - 11) / 12.); q = Fig('sukuna').pose(pose).place(-1.66, 0, 0, heading=-math.pi / 2); q.blow(-2.2 if 5 <= f < 12 else -.8, .2, .2); return q
def s2(f):
    if f < 20: pose, x, z, h = KW, -1.25, .0, -math.pi / 2 + .8
    elif f < 22: pose, x, z, h = KW, -1.10, .10, -math.pi / 2 + 2.2
    elif f < 24: pose, x, z, h = KC, -.95, .30, -math.pi / 2 + 4.4
    elif f < 26: pose, x, z, h = KC, -.75, .42, -math.pi / 2 - .9
    elif f < 27: pose, x, z, h = mix_pose(KC, K1, .55), -.58, .44, -math.pi / 2 - .2
    elif f < 29: pose, x, z, h = K1, -.45, .42, -math.pi / 2 + .1
    elif f < 33: pose, x, z, h = mix_pose(K1, K2, min(1, (f - 28) / 3.)), -.42, .40 - .03 * (f - 29), -math.pi / 2 + .12 + .05 * (f - 29)
    else: pose, x, z, h = mix_pose(K2, K3, (f - 32) / 3.), -.42, .24 - .06 * (f - 33), -math.pi / 2 + .5
    q = Fig('sukuna').pose(pose).place(x, .05, z, heading=h); q.blow(-1.6, .2, .4); return q
def s4(f):
    u = (f - 59) / 8.; e = ease(u); pose = mix_pose(POSES['punch_wind'], POSES['dive_punch'], min(1, u * 1.6)); return Fig('sukuna').pose(pose).place(-2.75 + 1.50 * e ** 1.3, .0, .25 + 1.0 * math.sin(math.pi * .55 * e), heading=-math.pi / 2)


def d_suk(n, f):
    g, c = dict(s1=(s1, C1), s2=(s2, C2), s4=(s4, C4))[n]; q = g(f)
    if n == 's4': q.blow(-1.0, .2, .1)
    L = q.draw(c, f, light=(-.4, .7, .5), pal=SUKPAL, rim=dict(dir=(.8, .3, -.5), col='#e6f0ff', thr=.5, amt=.8), kick=(-.8, .4, -.4), expr=dict(mood='grin'), weight=1.25 if n != 's1' else 1.5)
    if n == 's2' and f in (27, 29, 31): L = bare_leg(L, q.info, f)
    return dict(l=L)


def bare_leg(L, i, f):
    """the kimono opens along the kicking leg: thigh and shin, two flaps of cloth torn back by the speed"""
    pe, kn, ft = [np.array(v, float) for v in (i['pelvis'], i['knees']['R'], i['feet']['R'])]; d1 = unit(kn - pe); d2 = unit(ft - kn); n1 = np.array([-d1[1], d1[0]]); n2 = np.array([-d2[1], d2[0]]); w = np.hypot(*(kn - pe)) * .30; an = kn + (ft - kn) * .80; c = Cel(); ink = '#0b090c'; st = pe + d1 * w * .55
    lo = spl(np.array([pe - n1 * -w * .2, st + n1 * w * 1.15, kn + n1 * w * 1.5 - d1 * w * .4, kn + n1 * w * 2.6 - d1 * w * 1.4, pe + n1 * w * 3.2 + d1 * w * .2, pe + n1 * w * 2.2 - d1 * w * .6]), True, (3, 4), n=8); c.fill(lo, '#efe6d3'); c.fill(spl(np.array([st + n1 * w * 1.3, kn + n1 * w * 1.7 - d1 * w * .6, kn + n1 * w * 2.5 - d1 * w * 1.4, pe + n1 * w * 2.4]), True, (2,), n=8), '#978784'); kit.outline(c, lo, 5, ink, 3)
    leg = spl(np.array([st - n1 * w * .95, kn - n1 * w * .72 - d1 * 4, kn - n2 * w * .62 + d2 * w * .2, kn + (an - kn) * .45 - n2 * w * .55, an - n2 * w * .30, an + n2 * w * .30, kn + (an - kn) * .42 + n2 * w * .62, kn + n2 * w * .60 + d2 * w * .1, kn + n1 * w * .66 - d1 * w * .3, st + n1 * w * .98]), True, (0, 4, 5, 9), n=8); c.fill(leg, '#e6cdb4')
    c.fill(spl(np.array([st + n1 * w * .2, kn + n1 * w * .05, kn + (an - kn) * .5 + n2 * w * .02, an + n2 * w * .05, an + n2 * w * .30, kn + (an - kn) * .42 + n2 * w * .62, kn + n2 * w * .60, st + n1 * w * .98]), True, (3, 4), n=8), '#9c7a70'); kit.outline(c, leg, 5, ink, 5)
    for k in range(5): q = kn + d1 * w * (-.9 + .22 * k) + n1 * w * .1; c.line(np.array([q, q + n1 * w * .3 + d1 * 4]), 2.2, ink, taper=(.1, .6))
    up = spl(np.array([pe - n1 * w * .4 - d1 * w * .4, st - n1 * w * .80, kn - n1 * w * 1.00 - d1 * w * .7, kn - n1 * w * 2.3 - d1 * w * 2.2, pe - n1 * w * 1.9 - d1 * w * .9]), True, (2, 3), n=8); c.fill(up, '#efe6d3'); c.fill(spl(np.array([st - n1 * w * .95, kn - n1 * w * 1.05 - d1 * w * .9, kn - n1 * w * 1.8 - d1 * w * 1.7, pe - n1 * w * 1.2]), True, (1, 2), n=8), '#978784'); kit.outline(c, up, 5, ink, 7)
    for k in range(6): a_ = st - n1 * w * (1.0 + .1 * k); c.line(np.array([a_, a_ + (kn - n1 * w * (1.6 + .1 * k) - d1 * w * 1.4 - a_) * (.4 + .1 * k)]), 2.2, ink, taper=(.2, .7), seed=k)
    q = c.layer(); k = q[..., 3:4]; return L * (1 - k) + q


def d_m(n, f):
    g, c = dict(m1=(m1, C1), m2=(m2, C2), m3=(m3, None), m4=(m4, C4))[n]; c = c or c3(); m = g(f); lw = dict(m1=.95, m2=.62, m3=1.15, m4=.62)[n]; d = F.maho(m, c, scale=.6 if n in ('m1', 'm3') else .5, lw=lw, teeth=.7, lights=F.key_lights(m, key=1.05, d=dict(m1=(.7, .55, -.55), m2=(.75, .45, -.5), m3=(.85, .35, -.55), m4=(.5, .6, -.5))[n])); r = dict(l=d['layer'], w=F.well(m, c, 2)); r['li'] = M.burn_wheel(m.wheel, c, seed=f)[0][..., 1]; return r


def d_bg(k):
    c = (C1, C2, None, C4)[k - 1] or c3(); hz = float(c.project(np.array([c.eye + np.array([c.R[2][0], c.R[2][1], 0]) * 4000]))[0][0][1]); im = F.sky(10 + k, hz=hz + 8, cx=(500, 700, 1300, 960)[k - 1]).copy(); im = F.ground_ink(im, c, seed=k)
    Ts = {1: [(-16, -5, 22, 34, 22, 31), (4, 13, 26, 38, 28, 32), (14.5, 25, 18, 30, 20, 33), (-30, -18, 30, 44, 30, 34)], 2: [(-18, -8, 22, 34, 24, 41), (-5, 3, 34, 46, 34, 42), (17, 30, 22, 36, 30, 43)], 3: [(-14, -5, 16, 28, 24, 51), (5, 14, 18, 30, 30, 52)], 4: [(-22, -11, 26, 38, 18, 61), (12, 22, 26, 38, 20, 62), (-38, -26, 30, 44, 28, 63), (26, 38, 30, 44, 26, 64)]}[k]
    for i, T in enumerate(Ts):
        t = F.Tower(*T); d = t.draw(c, scale=.5, I=1.0, lit=.03, lw=.65, light=(.80, -.25, -.45)); L = F.haze(d['layer'], d['z'], start=12, dens=.014); r = np.random.default_rng(i + k * 7); y0 = r.uniform(180, 480); p0, p1 = (-60, y0), (1980, y0 + r.uniform(-260, 260)); sc = F.section(t, c, p0, p1)
        for q in F.part(L, p0, p1, slide=r.choice([-1, 1]) * r.uniform(30, 90), gap=10, rot=r.normal(0, .04), face=10, sec=sc):
            if q is not None: im = q.over(im)
    return F.calm(im, .45, sig={1: 4.0, 2: 3.0, 3: 7., 4: 3.0}[k])


WT = (3.3, 12, 2.6, 12, 14, 71); WL = ((-100, 250), (2000, 420))
def d_wall():
    """the building he goes through (cut 2): whole"""
    t = F.Tower(*WT); return dict(l=t.draw(C2, scale=.6, I=.75, lit=.08, lw=.75, light=(.45, .62, -.45), rim=.6)['layer'])
def d_wall():
    t = F.Tower(*WT); L = t.draw(C2, scale=.6, I=.75, lit=.02, lw=.75, light=(.45, .62, -.45), rim=.3)['layer']; a = L[..., 3:4]; L[..., :3] = L[..., :3] * .62 + rgb('#a33430') * a * .38; return dict(l=L)
def d_wsec(): return F.section(F.Tower(*WT), C2, *WL)


def d_eye(k):
    """his face, close: 0 cold, 1 the eye widens, 2 smile"""
    from chars.sukuna import Suk, EXPR
    e = dict(EXPR['cold']) if k < 2 else dict(EXPR['smile']); e['look'] = (.25, 0)
    if k == 0: e['eo'] = (.62, .60)
    if k == 1: e['eo'] = (1.12, 1.08); e['e2'] = (1.2, 1.15); e['brow'] = (.02, .02)
    from chars.sukuna_ink import Sheet
    from chars.sukuna_body import scarf_back, scarf_front
    T = lambda p: (np.asarray(p, float) - (1330, 640)) * (2.15, 2.0) + (690, 985); sb = Sheet(7); scarf_back(sb, T); L = sb.layer(); s = Suk(700, 400, 490, yaw=.46, pitch=.02, roll=-.10, expr=e, seed=3); q = s.ink(); L = L * (1 - q[..., 3:4]) + q; sf = Sheet(7); scarf_front(sf, T); q = sf.layer(); return L * (1 - q[..., 3:4]) + q


def d_w4(f):
    """close on the wheel over his shoulder: scribble, notch"""
    m = wheelm(); wh = dict(m.wheel); wh['spin'] = M.notch(f - 54 + (0 if f >= 54 else 0)); k = 1 + 1.6 * math.exp(-(f - 54) / 2.4); return M.burn_wheel(wh, CW, seed=f, core=.6 * k ** .5, width=1. + .1 * (k - 1))[0][..., 1]


_c3 = None
def c3():
    global _c3
    if _c3 is None: m = m3(0); hp = m.J['W1'] + V(0, -.05, .16); _c3 = M.aim(hp + np.array([.20, -.75, -.08]), [hp, m.w(m.jawf(0, -.16, -.06))], [(600, 640), (1290, 400)], f=1120, roll0=.04, lock=True)
    return _c3
def wheelm(): return M.Maho(dict(neck=(0, .30, 0), head=(0, -.20, .08), chest=(0, .12, 0), wing=-.2, fan=1.1, jaw=.18), heading=-.25, wheel='burn')
CW = Camera3D((-.55, -2.55, 2.72), (-.34, 0, 3.38), 1240, roll=.06)
def d_mw(): m = wheelm(); d = F.maho(m, CW, scale=.6, lw=1.1, lights=F.key_lights(m, key=1.0, d=(.8, .4, -.5))); return dict(l=d['layer'], w=F.well(m, CW, 4))
def d_bgw(): im = F.sky(17, hz=1500, cx=500).copy(); return F.calm(im, .45, sig=5)


TRU = {4: (.16, .40), 5: (.30, .62), 6: (.42, .84), 7: (.50, 1.0)}
def d_tr(f):
    """the road of the blade, through the drawn poses"""
    bl = []
    for u in np.linspace(*TRU[f], 9): q = M.Maho(SWEEP(u), pos=P1, heading=H1, wheel=None); hl, tp = M.blade_at(q, [0, 1]); bl.append((hl, hl + (tp - hl) * 1.16))
    return M.sweep_trail(bl, C1, seed=3, light='#fff6e6', n=80, w=2.0, fill=1.5, dark=.25, u0=.34)


def last(keys, f): return max(k for k in keys if k <= f)


JOBS = {f'bg{k}': ('d_bg', (k,)) for k in (1, 2, 3, 4)}; JOBS['wall'] = ('d_wall', ()); JOBS['wsec'] = ('d_wsec', ()); JOBS['mw'] = ('d_mw', ()); JOBS['bgw'] = ('d_bgw', ())
JOBS.update({f'm1_{f}': ('d_m', ('m1', f)) for f in T1}); JOBS.update({f'm2_{f}': ('d_m', ('m2', f)) for f in T2}); JOBS.update({f'm3_{k}': ('d_m', ('m3', k)) for k in (0, 1, 2)}); JOBS.update({f'm4_{f}': ('d_m', ('m4', f)) for f in T4})
JOBS.update({f's1_{f}': ('d_suk', ('s1', f)) for f in S1}); JOBS.update({f's2_{f}': ('d_suk', ('s2', f)) for f in S2}); JOBS.update({f's4_{f}': ('d_suk', ('s4', f)) for f in T4})
JOBS.update({f'eye{k}': ('d_eye', (k,)) for k in (0, 1, 2)}); WF = [54, 55, 56, 57, 58]; JOBS.update({f'w4_{f}': ('d_w4', (f,)) for f in WF})


JOBS.update({f'tr{f}': ('d_tr', (f,)) for f in TRU})


class Shot:
    id = ID; n = 72
    finish = dict(bloom=.26, vig=.18, grain=.013)
    EVENTS = [(2, 'whoosh', dict(size=.6, note='sword raised, body coils')), (4, 'sweep', dict(size=1.0, note='3 m blade comes over and down 4-7, stops on his hair')), (7, 'cloth_cut', dict(size=.5, note='sleeve and hair sliced; pose held 7-10')), (7, 'step', dict(size=.6, note='front foot lands')),
              (18, 'whoosh', dict(size=.5)), (22, 'spin', dict(size=.7)), (27, 'kick', dict(size=1.0, note='contact, 2-frame hit-stop')), (29, 'launch', dict(size=.8)), (30, 'wall_crack', dict(size=.8)), (31, 'building', dict(size=1.0, note='concrete explosion')), (33, 'rubble', dict(size=.8, note='floors above come down')),
              (36, 'slash', dict(size=.9, note='one white cut')), (39, 'catch', dict(size=1.0, note='caught in the palm: the cut rings like glass under load')), (41, 'strain', dict(note='light bending 41-44')), (46, 'eye', dict(size=.6, note='his eye widens')), (50, 'smile', {}), (54, 'gakon', dict(size=1.0)), (59, 'lunge', dict(size=.9)), (66, 'silence', dict(note='freeze 66-69')), (70, 'impact', dict(size=1.2)), (71, 'impact', dict(size=1.0))]

    def setup(self):
        MS.ensure(ID); x, y = grid(); self.x, self.y = x, y; self.nz2 = fbm(H, W, (150, 60), (1, .35), 6); self.exit = C2.project(m2(33).J['chest'][None])[0][0]; self.red = F.calm(F.sky(23, hz=1400, cx=300), .45, sig=5); c = c3(); self.hp = []
        for k in range(3): m = m3(k); self.hp.append(c.project((m.J['W1'] + V(0, -.07, .17))[None])[0][0])
        self.wp = CW.project(wheelm().wheel['c'][None])[0][0]; F._SKY.clear(); self.hit = C2.project(np.array([[4.3, 2.6, 2.4]]))[0][0]; q = s2(27); q.draw(C2, 27, pal=SUKPAL); self.con = np.array(q.info['feet']['R'], float)
        self.gline = []
        for j in range(16): a = math.radians(196 + 9.0 * j); X = np.array([P1[0] + 2.9 * math.cos(a), P1[1] + 2.9 * math.sin(a), 0.]); p_, z_ = C1.project(X[None]); self.gline.append((float(p_[0][0]), float(p_[0][1]), float(np.clip(2.2 / max(z_[0], .4), .2, 1.6))))
        a = MS.val(ID, 's1_6', 'l')[..., 3]; xh = int(np.nonzero(a.max(0) > .5)[0].max()); mid = int(C1.project(np.array([[-1.50, 0, 1.0]]))[0][0][0]); ys, xs = np.nonzero(a[:, :mid - 20] > .5); tp = np.array([xs[np.argmin(ys)], ys.min()], float); ys2, xs2 = np.nonzero(a[:, mid - 20:mid + 170] > .5); ht = np.array([xs2[np.argmin(ys2)] + mid - 20, ys2.min()], float)
        box = lambda p, w, h: (int(max(0, p[0] - w)), int(max(0, p[1] - 20)), int(min(W, p[0] + w)), int(min(H, p[1] + h)))
        self.snips = [(box(tp, 130, 130), ((tp[0] + 200, tp[1] + 16), (tp[0] - 200, tp[1] + 40)), (-40, -26), .16), (box(ht, 110, 60), ((ht[0] + 200, ht[1] + 12), (ht[0] - 200, ht[1] + 26)), (-16, -34), -.2)]; return self

    def maho(self, im, d, fl=1., f=0, well=True):
        if well: F.halo(im, d['l'], amt=.45); d['w'].over(im)
        d['l'].over(im); return F.hot(im, d['li'], fl * (.94 + .06 * math.sin(f * 2.1)))

    # 1 ------------------------------------------------ the sweep: two profiles, a gap, one arc
    def cut1(self, f):
        im = MS.val(ID, 'bg1'); d = MS.get(ID, f'm1_{last(T1, f)}'); sk = MS.get(ID, f's1_{last(S1, f)}'); sk = sk['l'] if isinstance(sk, dict) else sk
        if 4 <= f <= 6: MS.get(ID, f'tr{f}').over(im, .95)
        im = self.maho(im, d, 1., f)
        if 7 <= f <= 10: MS.get(ID, 'tr7').over(im, (1., .9, .6, .3)[f - 7])
        F.halo(im, sk, amt=.45); L = sk.full()
        if f >= 7:
            a = f - 7; L6 = MS.val(ID, 's1_6', 'l')
            for k, (box, ln, v, w) in enumerate(self.snips):                    # sleeve and hair, sliced clean
                x0, y0, x1, y1 = box; R = L6[y0:y1, x0:x1]; yy, xx = np.mgrid[y0:y1, x0:x1].astype(f32); (ax, ay), (bx, by) = ln; sd = ((xx - ax) * (by - ay) - (yy - ay) * (bx - ax)) < 0; m = sd[..., None].astype(f32); pc = R * m
                if f < 12: L[y0:y1, x0:x1] = L[y0:y1, x0:x1] * (1 - m)
                th = math.degrees(w * a); pad = 60; q = np.pad(pc, ((pad, pad), (pad, pad), (0, 0))); M_ = cv2.getRotationMatrix2D((q.shape[1] / 2, q.shape[0] / 2), th, 1.0); q = cv2.warpAffine(q, M_, (q.shape[1], q.shape[0])); F.Bit(q, x0 - pad + v[0] * a, y0 - pad + v[1] * a + 1.2 * a * a).over(im)
        k = L[..., 3:4]; im = im * (1 - k) + L[..., :3]; cel = Cel(ss=1); dr = False
        if 7 <= f: p_ = C1.project(np.array([[.55, .2, 0]]))[0][0]; fx.dust_skirt(cel, f - 7, p_[0], p_[1], 190, 26, 46, seed=3, n=7, pal=F.DUSTK, life=.7, lw=1.6); dr = True
        if 5 <= f <= 7: kit.speed_lines(cel, .55, 16, 6, (500, 1500), '#fff3e2', f, box=(500, 0, 1300, 700), alpha=.40); dr = True
        return over(im, cel.layer()) if dr else im

    # 2 ------------------------------------------------ the kick through a building
    def cut2(self, f):
        im = MS.val(ID, 'bg2'); wl = MS.get(ID, 'wall'); wl = wl['l'] if isinstance(wl, dict) else wl; d = MS.get(ID, f'm2_{last(T2, f)}'); sk = MS.get(ID, f's2_{last(S2, f)}'); sk = sk['l'] if isinstance(sk, dict) else sk; hit = self.hit; c = self.con
        if f < 30: wl.over(im); im = self.maho(im, d, 1., f)
        elif f == 30:                                                           # he meets the wall: it stars, then goes
            wl.over(im); pen, wp = Pen(), Pen(); r = np.random.default_rng(8)
            for j in range(34):
                th = j / 34 * 6.283 + r.normal(0, .1); q = np.array(hit, float); P_ = [q]
                for i in range(int(r.integers(4, 9))): th += r.normal(0, .22); q = q + np.array([math.cos(th), math.sin(th)]) * r.uniform(40, 110); P_.append(q)
                pen.stroke(np.array(P_), r.uniform(4, 13), j, taper=(0, .8)); wp.stroke(np.array(P_) + (3, 4), r.uniform(1.5, 3.5), j, taper=(0, .8))
            a_ = wl.full()[..., 3:4]; Wm = wp.raster(H, W)[..., None] * a_; K = pen.raster(H, W)[..., None] * a_; im = im * (1 - Wm) + rgb('#f4d9c0') * Wm; im = im * (1 - K) + rgb('#070304') * K; im = self.maho(im, d, 1., f)
        else:
            t = (f - 30.4) / 6.; L = wl.full(); x, y = self.x, self.y; m = ((x - hit[0]) / 300) ** 2 + ((y - hit[1]) / 345) ** 2 + .13 * self.nz2; ep = self.exit; e2 = ((x - ep[0]) / 185) ** 2 + ((y - ep[1]) / 215) ** 2 + .16 * np.roll(self.nz2, 300, 1); g_ = np.clip(1.25 - e2, 0, 1)[..., None]; im = im * (1 - g_) + (rgb('#ff6a45') * (1 - g_) + rgb('#ffe2bd') * g_) * g_; im = self.maho(im, d, 1., f); hole = smooth(.97, 1.03, m)[..., None]; W0 = L * hole; chunk = L * (1 - hole); aw = L[..., 3]
            ex = smooth(.93, .99, e2); ins = (1 - hole[..., 0]) * ex * aw                # the tunnel he made: torn floors, then the far wall, then sky with him in it
            col = rgb('#12050a') + np.clip(1.9 - np.sqrt(np.maximum(e2, 0)), 0, 1)[..., None] * rgb('#7a1a1e') * .9; im = im * (1 - ins[..., None]) + col * ins[..., None]
            cs = Cel(ss=1); r = np.random.default_rng(12)
            for k, yy in enumerate((-250, -95, 70, 225)):
                e0, e1 = np.array([hit[0] - 300, hit[1] + yy + 34]), np.array([hit[0] + 300, hit[1] + yy - 46]); pts = [e0 + (e1 - e0) * u_ + np.array([0, r.normal(0, 4)]) for u_ in np.linspace(0, 1, 9)]; cs.fill(np.vstack([pts, [p_ + (14, 30 + r.uniform(-8, 12)) for p_ in pts[::-1]]]), '#7d2226'); cs.fill(np.vstack([pts, [p_ + (4, 10) for p_ in pts[::-1]]]), '#e9b9a2')
                for q_ in range(9): p_ = e0 + (e1 - e0) * r.random() + (6, 26); cs.line(np.array([p_, p_ + (r.normal(0, 8), 22), p_ + (r.normal(0, 22), r.uniform(34, 64))]), 2.4, '#050203', taper=(0, .4))
            sl = cs.layer() * ins[..., None]; im = over(im, sl); rimh = (np.clip(1 - np.abs(m - 1.0) / .05, 0, 1) * aw * (m > .97) * smooth(-.3, .7, np.roll(self.nz2, 517, 1)))[..., None]; lipx = (np.clip(1 - np.abs(e2 - .97) / .06, 0, 1) * ins * smooth(-.6, .5, np.roll(self.nz2, 211, 0)))[..., None]; im = im * (1 - lipx) + rgb('#ffd2ac') * lipx
            for q in F.part(W0, *WL, slide=-70 * t ** 1.6, gap=-6, rot=-.16 * t ** 1.7, fall=150 * t ** 2.2, face=20):   # the floors above come down
                if q is not None: q.over(im)
            im = im * (1 - rimh * .8) + rgb('#f6dcc6') * rimh * .8; r = np.random.default_rng(6); bits = []
            for i in range(4):                                                  # few, big, slow to start, then gone
                for j in range(4):
                    x0, y0 = int(hit[0] - 330 + i * 165 + r.normal(0, 16)), int(hit[1] - 370 + j * 186 + r.normal(0, 16)); x0_, y0_ = max(0, x0), max(0, y0); x1_, y1_ = min(W, x0 + 178), min(H, y0 + 198)
                    if x1_ - x0_ < 30 or y1_ - y0_ < 30: continue
                    pc = chunk[y0_:y1_, x0_:x1_]
                    if pc[..., 3].mean() < .15: continue
                    cc = np.array([(x0_ + x1_) / 2, (y0_ + y1_) / 2]); dv = cc - np.array([hit[0], hit[1]]); far = np.hypot(*dv) / 400; k = max(0., f - 30.6); g = k ** 1.75; v = unit(dv + r.normal(0, 30, 2)) * r.uniform(26, 40) * (1.2 - .4 * far) + np.array([0, -r.uniform(0, 6)]); sz = 1 + .025 * g; th = r.normal(0, 3.2) * g; hh, ww = pc.shape[:2]; P_ = np.array([(r.uniform(0, 30), 0), (.55 * ww, r.uniform(0, 22)), (ww, r.uniform(0, 50)), (ww - r.uniform(0, 40), hh), (.35 * ww, hh - r.uniform(0, 34)), (r.uniform(0, 30), .6 * hh)], np.float32); ms = np.zeros((hh, ww), np.uint8); cv2.fillPoly(ms, [np.round(P_).astype(np.int32)], 255, cv2.LINE_AA); pc = pc * (ms.astype(f32) / 255)[..., None]; pad = 70; q = np.pad(np.ascontiguousarray(pc, f32), ((pad, pad), (pad, pad), (0, 0))); M_ = cv2.getRotationMatrix2D((q.shape[1] / 2, q.shape[0] / 2), th, sz); q = cv2.warpAffine(q, M_, (q.shape[1], q.shape[0])); e = np.clip(q[..., 3] - kit.shifted(q[..., 3], 4, 6), 0, 1)[..., None]; q[..., :3] = q[..., :3] * (1 - e) + rgb('#0a0406') * e * q[..., 3:4]; bits.append((far, F.Bit(q, x0_ - pad + v[0] * g, y0_ - pad + v[1] * g + 2.6 * g)))
            for _, q in sorted(bits, key=lambda b: -b[0]): q.over(im)
        cel = Cel(ss=1)
        if f >= 31:
            for j in range(3): th = j / 3 * 6.283 + 1.2; fx.smoke_burst(cel, f - 30, hit[0] + math.cos(th) * 330, hit[1] + math.sin(th) * 370, 110, seed=21 + j, n=3, ang=(th - .6, th + .6), speed=420, life=1.5, pal=F.DUSTK, lw=2.0)
            fx.debris(cel, f - 30, (hit[0], hit[1]), n=7, seed=22, speed=(120, 500), ang=(-3.4, .2), size=(44, 100), grav=2600, pal=F.ROCKB, spread=(120, 160), lw=2.2, trails=False, drag=.2, spin=2.5)
        if f >= 27: fx.dust_skirt(cel, f - 27, *C2.project(np.array([[-.2, .1, 0]]))[0][0], 230, 38, 56, seed=5, n=7, pal=F.DUSTK, life=.7, lw=1.5)
        if f >= 27: im = over(im, cel.layer())
        if 22 <= f <= 25:                                                       # the spin: brush arcs of the hem, round a planted body
            p = C2.project(np.array([[-1.0 + .12 * (f - 22), .05, 1.15]]))[0][0]; pen, wp, gp = Pen(), Pen(), Pen(); r = np.random.default_rng(f); sk.over(im)
            for j in range(64): a0 = r.uniform(-3.6, .6) + (f - 22) * 1.3; a = np.linspace(a0, a0 + r.uniform(.6, 1.7), 20); R = r.uniform(110, 400); q = np.c_[p[0] + np.cos(a) * R * 1.1, p[1] + 40 + np.sin(a) * R * .34 - .12 * R]; (wp if r.random() < .55 else pen if r.random() < .6 else gp).stroke(q, r.choice([2, 5, 11, 24]), j, taper=(.6, .12), press=.6)
            for pn, col in ((gp, '#978784'), (pen, '#0b090c'), (wp, '#f4ead8')): K = pn.raster(H, W)[..., None]; im = im * (1 - K) + rgb(col) * K
        else: sk.over(im)
        if f == 27:                                                             # contact, 2 frames: the graphic, then the held blow
            im = np.zeros((H, W, 3), f32) + rgb('#c4141f'); cl = Cel(ss=1); kit.focus_lines(cl, c[0], c[1], 300, 2400, 170, 13, '#0a0608', 3); r = np.random.default_rng(2); th = np.sort(r.uniform(0, 6.283, 34)); R = np.where(np.arange(34) % 2 == 0, r.uniform(300, 760, 34), r.uniform(90, 170, 34)); cl.fill(np.c_[c[0] + np.cos(th) * R, c[1] + np.sin(th) * R * .8], '#f6ead6'); im = over(im, cl.layer()); im = over(im, F.silhouette(d['l'].full(), '#0a0608', '#f6ead6', 4, 3)); im = F.hot(im, d['li'], .9); im = over(im, F.silhouette(sk.full(), '#0a0608', '#f6ead6', -4, 3)); cl = Cel(ss=1); th = np.sort(r.uniform(0, 6.283, 22)); R = np.where(np.arange(22) % 2 == 0, r.uniform(120, 300, 22), r.uniform(30, 60, 22)); cl.fill(np.c_[c[0] + np.cos(th) * R, c[1] + np.sin(th) * R * .8], '#fffaf0'); im = over(im, cl.layer())
        elif f in (28, 29):
            cl = Cel(ss=1); kit.focus_lines(cl, c[0], c[1], 300 + 50 * (f - 28), 2200, 130, 9, '#fff3e2' if f == 28 else '#0b0608', f); r = np.random.default_rng(3)
            if f == 28: th = np.sort(r.uniform(0, 6.283, 26)); R = np.where(np.arange(26) % 2 == 0, r.uniform(90, 210, 26), r.uniform(22, 44, 26)); cl.fill(np.c_[c[0] + np.cos(th) * R, c[1] + np.sin(th) * R * .8], '#fffaf0')
            im = over(im, cl.layer())
        return im

    # 3 ------------------------------------------------ the palm, the eye
    def cut3(self, f):
        if f >= 45:
            k = 0 if f < 46 else 1 if f < 50 else 2; im = self.red.copy(); MS.get(ID, f'eye{k}').over(im); z = 1 + .05 * ease((f - 45) / 8.); z *= 1.05 if f in (46,) else 1.; cx, cy = 760, 540; return comp.cam2d(im, cx + (W / 2 - cx) / z, cy + (H / 2 - cy) / z, z)
        k = 0 if f < 39 else 1 if f < 41 else 2; im = MS.val(ID, 'bg3'); im = self.maho(im, MS.get(ID, f'm3_{k}'), 1., f); hp = self.hp[k]
        pen = Pen(); r = np.random.default_rng(f); x0 = -80.                 # the slash: a white thread that stops in the hand, bends, splinters
        if f < 39: e = min(1, (f - 35) / 3.); p0 = np.array([x0, hp[1] - 250]); p1 = p0 + (hp - p0) * e; t = np.linspace(0, 1, 50)[:, None]; pen.add(p0 * (1 - t) + p1 * t, np.full(50, 2.2 + (f == 38) * 1.6))
        else:
            w = 1 + .25 * math.sin(f * 3.1)
            for j in range(9): o = r.normal(0, 1) * (6 + 3 * (f - 39)); t = np.linspace(0, 1, 60); P_ = np.c_[x0 + (hp[0] - 40 - x0) * t, hp[1] - 250 * (1 - t) + o * np.sin(np.pi * t) * .5 + 60 * np.sin(np.pi * t * .9) ** 3 * (f - 38) / 6. * (1 if j % 2 else -.6)]; pen.add(P_, np.full(60, (2.6 if j == 0 else .7) * w))
            for j in range(26): a = r.uniform(0, 6.283); q = r.uniform(20, 90 + 26 * (f - 39)); l = r.uniform(20, 120); c_, s_ = math.cos(a), math.sin(a); pen.stroke(np.array([hp + (c_ * q, s_ * q), hp + (c_ * (q + l) + r.normal(0, 10), s_ * (q + l) + r.normal(0, 10))]), r.uniform(1.2, 3.6), j, taper=(.2, .7))
        mk = pen.raster(H, W); gl = cv2.GaussianBlur(mk, (0, 0), 4) + .9 * cv2.GaussianBlur(mk, (0, 0), 16)
        if f >= 39: rr = np.sqrt((self.x - hp[0]) ** 2 + (self.y - hp[1]) ** 2); a = 26 * np.exp(-((rr - 150 - 14 * (f - 39)) / 60.) ** 2) * (1 if f < 43 else .7); im = cv2.remap(im, (self.x + (self.x - hp[0]) / (rr + 1) * a).astype(f32), (self.y + (self.y - hp[1]) / (rr + 1) * a).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        im = F.white_hot(im, np.clip(mk * 1.3 + gl * .5, 0, 2))
        if f == 39: im = kit.impact(im, 'red', .20, .60, '#0a0608', '#f8f0e0', '#c4141f')
        return im

    # 4 ------------------------------------------------ gakon, lunge, freeze, impact
    def cut4(self, f):
        if f < 59:
            im = MS.val(ID, 'bgw'); d = MS.get(ID, 'mw'); d['w'].over(im); d['l'].over(im); im = F.hot(im, MS.get(ID, f'w4_{f}'), 1.25 if f < 56 else 1.); p = self.wp; e = math.exp(-(f - 54) / 2.2); return kit.flare(im, p[0], p[1], 700, '#e6eefc', .42 * e, .4)
        k = last(T4, min(f, 67)); im = MS.val(ID, 'bg4'); d = MS.get(ID, f'm4_{k}'); sk = MS.get(ID, f's4_{k}'); sk = sk['l'] if isinstance(sk, dict) else sk
        if f < 66:
            g = min(1, (f - 59) / 6.); cl = Cel(ss=1); kit.speed_lines(cl, .0, int(30 + 40 * g), 4 + 4 * g, (300, 1500), '#fff0dc', f // 2, alpha=.35 * g); kit.speed_lines(cl, .0, 26, 7, (300, 1200), '#0b0608', f // 2 + 9, alpha=.8 * g); im = over(im, cl.layer()); im = self.maho(im, d, 1., f); sk.over(im); return im
        c = np.array([930., 520.]); hot = np.exp(-(((self.x - c[0]) / 620) ** 2 + ((self.y - c[1]) / 420) ** 2)); im = im * .55 + (hot * .85)[..., None] * rgb('#ff5a3c') + (np.clip(hot - .45, 0, 1) * 1.3)[..., None] * rgb('#fff0d6')      # freeze: two silhouettes on a burning sky
        cl = Cel(ss=1); kit.focus_lines(cl, c[0], c[1], 330, 2300, 130, 9, '#0b0608', 5); im = over(im, cl.layer()); im = over(im, F.silhouette(d['l'].full(), '#080507', '#fff3df', 4, 3)); im = F.hot(im, d['li'], .9); im = over(im, F.silhouette(sk.full(), '#080507', '#ff4a3c', -4, 2))
        if f == 70: im = kit.impact(im, 'bw', .2, .6, '#080507', '#f8f2e6')
        if f == 71: im = kit.impact(im, 'inv', .2, .6, '#080507', '#c4141f')
        return im

    def frame(self, f):
        im = self.cut1(f) if f < 18 else self.cut2(f) if f < 36 else self.cut3(f) if f < 54 else self.cut4(f); z = 1.02; dx = dy = 0.; cx, cy = W / 2, H / 2
        g = lambda t0, amps, s: kit.shake(f, amps[f - t0], s) if 0 <= f - t0 < len(amps) else np.zeros(2)
        a = g(8, (12, 9, 6, 3), 3) + g(27, (30, 22, 14, 8, 5), 5) + g(30, (16, 14, 10, 8, 5, 3), 7) + g(39, (20, 12, 8, 6, 5, 5), 8) + g(54, (24, 16, 9, 4), 9); dx, dy = a
        if f < 18: z = 1.03 + .03 * f / 18.
        elif f < 36: z = 1.03 + (.05 if f >= 27 else 0)
        elif f < 45: z = 1.03 + .04 * (f - 36) / 9.; cx, cy = 880, 520
        elif 59 <= f < 72: z = 1.02 + .22 * ease(min(1, (f - 59) / 7.)) + .010 * max(0, f - 66); cx, cy = 930, 500
        c0, c1 = cx + (W / 2 - cx) / z, cy + (H / 2 - cy) / z; im = comp.cam2d(im.astype(f32), c0 + dx, c1 + dy, z, 0)
        return im.astype(f32)
