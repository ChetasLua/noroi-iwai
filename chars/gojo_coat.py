"""Gojo's black high-collar jacket: collar as three deformed rings (any yaw / pitch), torso as authored front pattern on a
rounded trunk, sleeves from a skeleton.  Light = decisive violet polygons along seams and folds, after the model sheet."""
import numpy as np, math
from jj.ink import spl, resample
from jj.kit import outline, hatch

D2R = math.pi / 180


class Body:
    """origin = pit of the neck; x right, y up, z to the viewer; unit = the head unit"""
    def pose(b, cx, cy, s, yaw=0., pitch=0., roll=0., persp=.06):
        b.cx, b.cy, b.s, b.yaw, b.pitch, b.roll, b.persp = cx, cy, s, yaw, pitch, roll, persp
        a, p = yaw, pitch
        Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
        Rx = np.array([[1, 0, 0], [0, math.cos(p), math.sin(p)], [0, -math.sin(p), math.cos(p)]])
        c, sn = math.cos(-roll), math.sin(-roll); Rz = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
        b.R = Rz @ Rx @ Ry; return b

    def P3(b, Q):
        q = np.asarray(Q, float) @ b.R.T; f = 1 / (1 - np.clip(q[..., 2] * b.persp, -.5, .5))
        return np.stack([b.cx + q[..., 0] * f * b.s, b.cy - q[..., 1] * f * b.s], -1)

    def z(b, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        w = np.interp(y, [-4, -2.6, -1.4, -.5, 0, .3], [1.7, 1.55, 2.0, 2.0, 1.5, 1.1]); d = np.interp(y, [-4, -2.6, -1.2, -.5, 0, .3], [.62, .60, .78, .62, .46, .42])
        return d * (1 - np.clip(np.abs(x) / w, 0, .999) ** 2.3) ** (1 / 2.3)

    def P(b, xy, dz=0.):
        xy = np.asarray(xy, float); return b.P3(np.concatenate([xy, (b.z(xy[..., 0], xy[..., 1]) + dz)[..., None]], -1))


def lw(b, w, lo=.5): return w * max(lo, (b.s / 260.) ** .55)


# ------------------------------------------------------------------------------------------------ collar
PH = np.array([-180, -150, -120, -100, -86, -70, -52, -34, -16, 0, 16, 34, 52, 70, 86, 100, 120, 150, 180.])
YT = np.array([1.00, .98, .90, .80, .70, .60, .545, .50, .47, .462, .475, .51, .56, .62, .73, .83, .93, .99, 1.00])
YM = np.array([.52, .50, .45, .40, .34, .31, .27, .24, .235, .23, .22, .23, .25, .29, .33, .38, .44, .50, .52])
YL = np.array([.30, .26, .20, .14, .07, .03, -.01, -.04, -.06, -.07, -.06, -.04, -.01, .03, .08, .15, .21, .27, .30])
RT, RM, RL = (.90, .84), (.76, .74), (.80, .72)


def col_pt(b, deg, v, off=0.):
    """collar surface: deg azimuth, v 0 top .. 1 bottom"""
    deg = np.asarray(deg, float); v = np.asarray(v, float); d = ((deg + 180) % 360) - 180
    yt, ym, yl = (np.interp(d, PH, t) for t in (YT, YM, YL)); a = d * D2R
    crump = .018 * np.sin(a * 7 + 1.) * np.sin(v * 9) + .012 * np.sin(a * 13 + 2)
    w0, w1, w2 = (1 - v) * (1 - 2 * v), 4 * v * (1 - v), v * (2 * v - 1)
    y = yt * w0 + ym * w1 + yl * w2
    rx = RT[0] * w0 + RM[0] * w1 + RL[0] * w2 + off + crump; rz = RT[1] * w0 + RM[1] * w1 + RL[1] * w2 + off + crump
    return np.stack([rx * np.sin(a), y, rz * np.cos(a) - .02 + 0 * y], -1)


def collar_back(cel, b, pal, seed=0):
    g = np.linspace(-180, 180, 121); R = b.P3(col_pt(b, g, np.zeros_like(g)))
    low = b.P3(col_pt(b, g, np.full_like(g, .55)))
    back = np.abs(((g + math.degrees(b.yaw) + 180) % 360) - 180) > 82
    # outside of the far wall (only its silhouette shows), then the inside
    cel.fill(R, pal['coat_in']); cel.fill(low, pal['coat_in'])
    gg = np.linspace(82, 278, 60) - math.degrees(b.yaw); E = b.P3(col_pt(b, gg, np.zeros_like(gg)))
    cel.line(E, lw(b, 4.2), pal['coat_ln'], taper=(.05, .05), press=.3, seed=seed + 1)
    # a thin lit edge inside the far rim
    for a0, a1 in ((100, 165), (200, 258)):
        q = np.linspace(a0, a1, 24) - math.degrees(b.yaw); w = .045 * np.sin(np.linspace(0, np.pi, 24)) ** .8
        cel.fill(b.P3(np.vstack([col_pt(b, q, .02 + 0 * q), col_pt(b, q[::-1], .02 + w)])), pal['coat_lt2'])


def collar_front(cel, b, pal, seed=0):
    c = -math.degrees(b.yaw); g = np.linspace(c - 92, c + 92, 93)
    T = b.P3(col_pt(b, g, 0 * g)); B = b.P3(col_pt(b, g, 0 * g + 1))
    vv = np.linspace(0, 1, 18); side = lambda a, k: b.P3(col_pt(b, np.full(18, a), vv) + np.array([0, 0, 0]))
    bump = lambda sd: sd + 0
    L, R = side(g[0], 0), side(g[-1], 1)
    # crumpled silhouette: the cloth buckles at two heights
    def buckle(Q, sg):
        d = np.array([sg, 0.]); amp = np.interp(vv, [0, .10, .24, .36, .46, .58, .70, .82, 1], [.0, .035, .05, -.015, -.05, .01, .045, .01, .0]) * b.s
        return Q + d * amp[:, None]
    L, R = buckle(L, -1), buckle(R, 1)
    P = np.vstack([T, R[1:], B[::-1][1:], L[::-1][1:]])
    cel.fill(P, pal['coat'])
    cel.clip(P)
    def poly(pts, col=None, corners=None):
        a = np.asarray(pts, float); n = np.stack([np.sin(a[:, 0] * D2R), 0 * a[:, 0], np.cos(a[:, 0] * D2R)], -1) @ b.R.T
        if n[:, 2].max() < .0: return
        q = b.P3(col_pt(b, a[:, 0], a[:, 1], .004)); cel.fill(q if corners is None else spl(q, True, corners=corners, n=6), col or pal['coat_lt'])
    dp, mid, lt = pal['coat_dp'], pal['coat_md'], pal['coat_lt']
    poly([(-100, .30), (-60, .36), (-20, .46), (-12, .62), (-40, .70), (-80, .62), (-100, .56)], dp); poly([(22, .40), (60, .30), (104, .26), (104, .52), (70, .58), (30, .66)], dp)
    poly([(-100, .80), (-50, .84), (-14, .86), (20, .86), (60, .80), (100, .74), (100, 1.1), (-100, 1.1)], dp)
    # rim of the top edge, broken
    poly([(-96, .035), (-70, .03), (-40, .035), (-14, .045), (-13, .115), (-30, .12), (-52, .145), (-66, .10), (-84, .17)])
    poly([(21, .04), (46, .035), (78, .03), (98, .03), (90, .075), (66, .06), (50, .11), (34, .085)])
    # folds
    poly([(-74, .33), (-50, .375), (-14, .455), (-32, .50), (-48, .52), (-60, .42)])
    poly([(17, .62), (34, .50), (58, .40), (80, .34), (66, .47), (46, .56), (34, .63)])
    poly([(-92, .66), (-66, .70), (-40, .79), (-62, .77), (-84, .75)]); poly([(30, .86), (56, .76), (86, .66), (70, .80), (50, .86)])
    poly([(96, .50), (84, .52), (70, .60), (88, .60)], mid); poly([(-98, .48), (-84, .50), (-70, .56), (-88, .56)], mid)
    # placket: the strip that closes the collar
    poly([(-12, -.02), (20, -.02), (20.5, 1.02), (-12.5, 1.02)], pal['coat'])
    poly([(-12, .0), (-6.5, .01), (-8, .10), (-7.5, .40), (-3.5, .55), (-1, .60), (-6, .59), (-12, .60), (-10.5, .34)])
    poly([(-4, .03), (16, .03), (17, .065), (-3, .085)], mid); poly([(-12, .84), (-7, .80), (-6, .90), (10, .91), (17, .86), (16.5, .96), (2, .98), (-12, .99)])
    poly([(6, .20), (15, .30), (12, .52), (8, .40)], mid)
    for a, w in ((-12.3, 3.4), (20.3, 2.2)):
        n = float((b.R @ np.array([math.sin(a * D2R), 0, math.cos(a * D2R)]))[2])
        if n > .05: cel.line(b.P3(col_pt(b, np.full(16, a), np.linspace(-.01, 1.0, 16), .006)), lw(b, w), pal['coat_ln'], taper=(.1, .3), seed=seed + 20)
    for q in ([(-96, .36), (-70, .33), (-40, .40), (-16, .45)], [(98, .30), (74, .33), (46, .44), (24, .55)], [(-96, .62), (-70, .66), (-46, .74)], [(96, .62), (74, .66), (52, .76)], [(-70, .18), (-58, .26), (-54, .34)], [(70, .16), (60, .24), (58, .32)]):
        q = spl(np.array(q), n=8); n = np.stack([np.sin(q[:, 0] * D2R), 0 * q[:, 0], np.cos(q[:, 0] * D2R)], -1) @ b.R.T
        q = q[n[:, 2] > -.02]
        if len(q) > 4: cel.line(b.P3(col_pt(b, q[:, 0], q[:, 1], .004)), lw(b, 2.0), pal['coat_ln'], taper=(.1, .7), seed=seed + 25, tmin=.05)
    cel.unclip()
    cel.line(T, lw(b, 4.4), pal['coat_ln'], taper=(.06, .06), press=.35, seed=seed + 2)
    for Q in (L, R): cel.line(Q, lw(b, 4.6), pal['coat_ln'], taper=(.06, .1), press=.4, seed=seed + 3)
    return P


# ------------------------------------------------------------------------------------------------ torso
def torso(cel, b, pal, seed=0, arms='down', left=True, right=True, y_end=-3.6):
    """chest, shoulders; arms='down' also draws both upper sleeves as part of the silhouette.  left/right = picture sides."""
    sil_r = [(.60, .10), (.74, .0), (1.02, -.12), (1.40, -.25), (1.60, -.33), (1.82, -.50), (1.94, -.78), (2.00, -1.05), (1.99, -1.40), (2.06, -1.85), (2.10, -2.35), (2.06, -2.75), (2.12, -3.2), (2.10, -3.8)]
    sil_l = [(-.60, .10), (-.74, .0), (-1.00, -.13), (-1.38, -.27), (-1.58, -.34), (-1.80, -.52), (-1.95, -.82), (-1.98, -1.12), (-2.04, -1.5), (-2.02, -1.95), (-2.10, -2.4), (-2.06, -2.9), (-2.1, -3.8)]
    tr_r = [(.60, .10), (.74, .0), (1.02, -.12), (1.40, -.25), (1.58, -.36), (1.60, -.70), (1.50, -1.15), (1.36, -1.45), (1.24, -1.9), (1.16, -2.5), (1.18, -3.1), (1.26, -3.8)]
    tr_l = [(-x, y) for x, y in tr_r]
    def corners(a): return a
    Rr = sil_r if (arms == 'down' and right) else tr_r; Ll = sil_l if (arms == 'down' and left) else tr_l
    R_, L_ = np.array(Rr), np.array(Ll)
    P = np.vstack([b.P(resample(R_, .08), 0), b.P(resample(L_[::-1], .08), 0)])
    cel.fill(P, pal['coat'])
    cel.clip(P)
    lt, md, dp = pal['coat_lt'], pal['coat_md'], pal['coat_dp']
    def poly(pts, col=lt): cel.fill(b.P(np.asarray(pts, float), .004), col)
    # --- blackest blacks first: under the collar, between arm and body
    poly([(-1.0, .05), (-.6, -.20), (0, -.30), (.62, -.20), (1.0, .05)], dp)
    poly([(-1.50, -1.0), (-1.36, -1.40), (-1.18, -2.0), (-1.10, -2.8), (-1.2, -3.8), (-1.5, -3.8), (-1.46, -2.4), (-1.5, -1.6)], dp); poly([(1.52, -1.0), (1.38, -1.42), (1.22, -2.0), (1.14, -2.8), (1.2, -3.8), (1.55, -3.8), (1.5, -2.4), (1.54, -1.6)], dp)
    # --- shoulders: light on the yoke with dark facets left inside
    poly([(-.70, .02), (-1.02, -.11), (-1.42, -.26), (-1.62, -.34), (-1.86, -.54), (-2.0, -.90), (-2.02, -1.3), (-1.90, -1.02), (-1.80, -.78), (-1.60, -.58), (-1.38, -.50), (-1.12, -.40), (-1.30, -.40), (-1.46, -.37), (-1.16, -.26), (-.92, -.20), (-.64, -.20), (-.80, -.13), (-.62, -.10)])
    poly([(.70, .02), (1.02, -.10), (1.42, -.24), (1.62, -.32), (1.84, -.50), (1.98, -.84), (2.0, -1.2), (1.90, -.94), (1.78, -.70), (1.62, -.52), (1.42, -.43), (1.20, -.38), (1.36, -.37), (1.18, -.28), (.90, -.22), (.70, -.23), (.86, -.14), (.64, -.10)])
    # --- chest: the long diagonal from the shoulder to the placket
    poly([(-1.50, -.52), (-1.26, -.63), (-.86, -.85), (-.42, -1.03), (-.175, -1.11), (-.175, -1.46), (-.44, -1.33), (-.78, -1.12), (-1.10, -.90), (-1.36, -.70)])
    poly([(-.98, -.36), (-.66, -.42), (-.40, -.53), (-.62, -.51), (-.84, -.47)])
    poly([(1.60, -.46), (1.28, -.60), (.92, -.78), (.58, -.93), (.36, -1.05), (.66, -1.04), (.88, -1.00), (1.16, -.84), (1.46, -.62)])
    poly([(1.22, -.42), (.92, -.48), (.60, -.64), (.80, -.61), (1.04, -.56)])
    # placket
    poly([(-.165, -.52), (-.10, -.44), (.08, -.46), (.16, -.40), (.15, -.52), (-.04, -.555), (-.12, -.62)])
    poly([(-.165, -.74), (.015, -.76), (-.01, -1.06), (-.05, -1.42), (-.11, -1.90), (-.16, -2.3), (-.165, -1.5)])
    # falls from the armpits
    poly([(-1.18, -1.52), (-.96, -1.80), (-.70, -2.02), (-.40, -2.22), (-.175, -2.36), (-.175, -2.46), (-.46, -2.32), (-.80, -2.12), (-1.02, -1.86)])
    poly([(1.16, -1.60), (1.00, -1.92), (.84, -2.10), (.70, -2.20), (.86, -2.16), (1.04, -2.00), (1.14, -1.80)], md)
    poly([(-.17, -2.46), (.06, -2.56), (.18, -2.62), (-.02, -2.64), (-.11, -2.72), (-.12, -3.1), (-.165, -2.9)])
    poly([(1.04, -2.3), (.92, -2.7), (.88, -3.1), (.98, -2.84), (1.06, -2.55)]); poly([(-1.0, -2.7), (-.72, -2.96), (-.46, -3.06), (-.72, -3.08), (-.92, -2.96)])
    # sleeves hanging
    if arms == 'down' and left:
        poly([(-1.92, -1.62), (-1.78, -1.86), (-1.62, -2.02), (-1.80, -2.02), (-1.95, -1.86)]); poly([(-2.08, -2.0), (-1.96, -2.3), (-1.72, -2.52), (-1.94, -2.5), (-2.1, -2.36)]); poly([(-1.60, -1.5), (-1.52, -1.9), (-1.56, -2.3), (-1.66, -1.9)], md)
    if arms == 'down' and right:
        poly([(1.96, -1.48), (1.84, -1.74), (1.66, -1.96), (1.84, -1.90), (1.98, -1.72)]); poly([(2.1, -2.1), (2.0, -2.34), (1.78, -2.5), (1.98, -2.5), (2.1, -2.42)]); poly([(1.62, -1.45), (1.54, -1.85), (1.58, -2.2), (1.68, -1.8)], md)
    cel.unclip()
    # pen strokes inside the big lights: the hand of the draughtsman
    if b.s > 90:
        for pts in ([(-.70, -.98), (-.64, -1.18)], [(-.62, -1.00), (-.57, -1.20)], [(-.55, -1.03), (-.51, -1.22)], [(-.58, -2.10), (-.55, -2.22)], [(-.52, -2.14), (-.49, -2.25)], [(.78, -.86), (.72, -1.0)], [(.85, -.83), (.79, -.98)], [(.92, -.80), (.87, -.94)], [(-1.62, -.42), (-1.70, -.56)], [(-1.56, -.39), (-1.62, -.50)]):
            cel.line(b.P(np.array([pts[0], (np.array(pts[0]) + pts[1]) / 2 + (.008, 0), pts[1]]), .006), lw(b, 1.5), pal['coat_ln'], taper=(.2, .7), seed=seed + int(abs(pts[0][0]) * 999), tmin=.06)
    ln = pal['coat_ln']
    L1 = lambda pts, w, tp=(.1, .4), sd=0: cel.line(b.P(spl(np.array(pts, float), n=10), .006), lw(b, w), ln, taper=tp, press=.35, seed=seed + sd, tmin=.08)
    L1([(-.17, -.16), (-.172, -.8), (-.168, -1.6), (-.172, -2.5), (-.17, -3.7)], 3.2, (.05, .2), 31)
    L1([(.275, -.18), (.278, -.6), (.282, -1.1)], 2.0, (.1, .6), 32); L1([(.28, -1.3), (.284, -2.0), (.28, -2.9)], 1.6, (.5, .5), 33)
    if arms == 'down':
        if left: L1([(-1.36, -1.30), (-1.42, -1.62), (-1.30, -2.1), (-1.18, -2.7), (-1.22, -3.4)], 3.4, (.3, .2), 34); L1([(-1.58, -.36), (-1.66, -.70), (-1.60, -1.05), (-1.46, -1.36)], 2.0, (.3, .6), 35)
        if right: L1([(1.36, -1.30), (1.44, -1.64), (1.32, -2.1), (1.20, -2.7), (1.24, -3.4)], 3.4, (.3, .2), 36); L1([(1.60, -.35), (1.70, -.68), (1.62, -1.05), (1.47, -1.36)], 2.0, (.3, .6), 37)
    for E, sd in ((Rr, 41), (Ll, 42)): outline(cel, b.P(resample(np.array(E), .03), 0), lw(b, 5.0), ln, seed=seed + sd, closed=False, press=.35)
    return P


# ------------------------------------------------------------------------------------------------ sleeve from a skeleton
def sleeve(cel, S, E, W_, r0, r1, r2, pal, s, seed=0, lit=1, cuff=True, front=True, bend=None, hand_fn=None):
    """screen joints S (shoulder) E (elbow) W (wrist); radii in px.  lit = +1/-1 side that carries the light (left normal of S->W)"""
    S, E, W_ = [np.asarray(v, float) for v in (S, E, W_)]
    a = E - S; c = W_ - E; La, Lc = np.hypot(*a) + 1e-6, np.hypot(*c) + 1e-6; a /= La; c /= Lc
    na, nc = np.array([-a[1], a[0]]), np.array([-c[1], c[0]]); cr = a[0] * c[1] - a[1] * c[0]; out = -1 if cr > 0 else 1     # outer side of the bend
    nb = na + nc; nb /= np.hypot(*nb) + 1e-6
    k = min(1.0, abs(cr))
    def side(sg):
        o = sg == out
        if o:
            P = [S + na * sg * r0, S + a * La * .30 + na * sg * r0 * 1.06, S + a * La * .72 + na * sg * r1 * 1.02, E + nb * sg * r1 * (1.04 + .16 * k) - a * r1 * .10, E + nb * sg * r1 * (1.0 + .10 * k) + c * r1 * .34, E + c * Lc * .42 + nc * sg * r1 * .98, E + c * Lc * .72 + nc * sg * r2 * 1.12, E + c * Lc * .84 + nc * sg * r2 * 1.02, W_ + nc * sg * r2 * 1.08]
        else:
            P = [S + na * sg * r0, S + a * La * .35 + na * sg * r0 * .96, S + a * La * .74 + na * sg * r1 * .96, E + nb * sg * r1 * (.92 - .30 * k) - a * r1 * .30, E + nb * sg * r1 * (.70 - .30 * k), E + nb * sg * r1 * (.95 - .22 * k) + c * r1 * .40, E + c * Lc * .50 + nc * sg * r1 * .92, E + c * Lc * .80 + nc * sg * r2 * .96, W_ + nc * sg * r2 * 1.04]
        return np.array(P)
    Lp, Rp = side(1), side(-1)
    LL, RR = spl(Lp, corners=(3, 4, 6), n=6), spl(Rp, corners=(3, 4, 6), n=6)
    cuffv = c * r2 * .22
    if cuff:                                     # far lip of the cuff, then the hand comes out of it, then the sleeve goes over the forearm
        q0 = spl(np.array([LL[-1], W_ + cuffv * 1.2 + nc * r2 * .55, W_ + cuffv * 1.5, W_ + cuffv * 1.2 - nc * r2 * .55, RR[-1]]), n=6); cel.fill(q0, pal['coat_in']); cel.line(q0, 3.0 * max(.5, (s / 260.) ** .55), pal['coat_ln'], taper=(.05, .05), seed=seed + 7)
    if hand_fn is not None:                      # only what is beyond the cuff
        q1 = W_ - c * r2 * .55; cel.clip(np.array([q1 + nc * 9e3, q1 + nc * 9e3 + c * 9e3, q1 - nc * 9e3 + c * 9e3, q1 - nc * 9e3])); hand_fn(); cel.unclip()
    end_ = spl(np.array([LL[-1], W_ - cuffv * .55 + nc * r2 * .5, W_ - cuffv * .75, W_ - cuffv * .55 - nc * r2 * .5, RR[-1]]), n=6)
    P = np.vstack([LL, end_[1:-1], RR[::-1]]); cel.fill(P, pal['coat'])
    cel.clip(P)
    lt, md, dp, ln = pal['coat_lt'], pal['coat_md'], pal['coat_dp'], pal['coat_ln']
    def at(t, u):                        # t 0..1 upper arm, 1..2 forearm; u -1..1 across (+ = left normal)
        t = np.asarray(t, float); u = np.asarray(u, float)
        p = np.where((t < 1)[..., None], S + a * (La * t)[..., None], E + c * (Lc * (t - 1))[..., None]); rr = np.interp(t, [0, 1, 2], [r0, r1, r2])
        w = np.clip((t - .82) / .36, 0, 1)[..., None]; nn = na * (1 - w) + nc * w; nn /= np.hypot(nn[..., 0], nn[..., 1])[..., None] + 1e-6
        return p + nn * (u * rr)[..., None]
    def poly(tu, col=lt): tu = np.asarray(tu, float); cel.fill(at(tu[:, 0], tu[:, 1]), col)
    g = float(lit)
    poly([(0, -g * .2), (.5, -g * .35), (.86, -g * .25), (1.0, -g * 0.), (1.2, -g * .30), (1.7, -g * .40), (2.1, -g * .3), (2.1, -g * 1.6), (0, -g * 1.6)], dp)
    poly([(.0, g * 1.4), (.0, g * .70), (.22, g * .78), (.42, g * .56), (.50, g * .80), (.72, g * .88), (.80, g * 1.4)])
    poly([(.60, g * .18), (.80, g * .46), (.92, g * .90), (.90, g * 1.4), (1.02, g * 1.4), (1.06, g * .70), (.96, g * .34), (.84, g * .20)])
    poly([(1.04, -g * .10), (1.12, g * .30), (1.30, g * .66), (1.20, g * .22)]); poly([(1.10, g * 1.4), (1.16, g * .82), (1.40, g * .74), (1.62, g * .84), (1.80, g * 1.4)])
    poly([(1.52, g * .10), (1.66, g * .46), (1.84, g * .62), (1.76, g * .24), (1.64, g * .06)]); poly([(1.46, -g * .62), (1.62, -g * .34), (1.80, -g * .22), (1.66, -g * .52)], md)
    if cuff:
        poly([(1.86, g * 1.4), (1.87, g * .30), (1.90, -g * .10), (1.94, g * .34), (2.0, g * .86), (2.0, g * 1.4)])
    cel.unclip()
    inn = -out
    l = lambda tu, w, sd: cel.line(spl(at(np.array(tu)[:, 0], np.array(tu)[:, 1]), n=8), w * max(.5, (s / 260.) ** .55), ln, taper=(.1, .7), press=.3, seed=seed + sd, tmin=.06)
    l([(.98, inn * .70), (1.0, inn * .20), (1.06, -inn * .30)], 2.4, 1); l([(.90, inn * .80), (.86, inn * .30), (.92, -inn * .10)], 2.0, 2); l([(1.10, inn * .80), (1.22, inn * .35), (1.26, -inn * .05)], 2.0, 3)
    if cuff: l([(1.86, -1.02), (1.885, -.3), (1.875, .4), (1.86, 1.02)], 2.2, 4)
    for Q, sd in ((LL, 5), (RR, 6)): outline(cel, Q, 4.6 * max(.5, (s / 260.) ** .55), ln, seed=seed + sd, closed=False, press=.35)
    if cuff:
        near = spl(np.array([LL[-1], W_ - cuffv * .55 + nc * r2 * .5, W_ - cuffv * .75, W_ - cuffv * .55 - nc * r2 * .5, RR[-1]]), n=6)
        if hand_fn is None: cel.fill(np.vstack([near, q0[::-1]]), pal['coat_in'])
        cel.line(near, 3.2 * max(.5, (s / 260.) ** .55), ln, taper=(.05, .05), seed=seed + 8)
    return P
