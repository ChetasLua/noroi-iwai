"""Hands: a posable 3-D construction that comes out as a hand-inked anime cel.  (ray caster: jj/handsdf.py; examples: tools/hands_hero.py, tools/hands_sheet.py)

    from jj import hands as HD
    cel = Cel(...)
    HD.draw(cel, 'fist', HD.Cam(960, 540, 420, dist=3.2), pal='night', light=(-.5, .75, .6), w=5, side='R', R=HD.euler(180, 78, 0))      # one call

    h = HD.Hand('cross', 'R', R=HD.euler(162, -8, 0), pos=(0, 0, 0), bld=HD.SLIM, pivot=(0, .75, 0), cuff=dict(kind='uniform', y=.16))
    g = HD.render([h], HD.Cam(960, 520, 380, 8), light=(-.5, .75, .6), res=560, clip=(0, 0, W, H))      # geometry, ~0.4-0.9 s per hand at res 560
    HD.ink(cel, g, pal, w=5, seed=3)                                                                      # flats + hard shadow + nails/bands/wraps + pen lines

Units: 1 = wrist to middle knuckle; a hand is about 2.1 long, the forearm 2.4.  World: x right, y up, z toward the viewer.  Cam(cx, cy, s, dist): s = px per
unit, dist = eye distance (12 flat, 5 normal, 3 strong, 1.9 fingers huge at the lens).  euler(yaw, pitch, roll) degrees: 0,0,0 = BACK of the hand to camera,
fingers up; yaw 180 = PALM to camera; from palm view pitch > 0 pushes the fingers at the lens (fist at lens = euler(180, 78, 0)); roll > 0 counter-clockwise.
A right hand seen from its back has the thumb on the left.  VIEWS lists good starting angles.  pivot = the hand point that sits on pos ((0,0,0) = wrist).

Poses (POSES): flat relaxed open splay fist point two cross pinch hammer nails rear flick pray sukuna dog_near dog_far grip rest thrust palm_up reach hook
carry slap.  A pose is a dict: finger = (mcp flex, pip flex, dip flex, abduction, twist) degrees, thumb = (radial, palmar, rho, mcp, ip), wrist = (flex, deviation),
cup.  Make your own with pose(...) or change one: dict(POSES['open'], wrist=(-30, 0)).   blend(a, b, t, drag=, over=, order=, curl=) = drawn inbetween (fingers
lead and drag), tremble(pose, seed, deg) = nervous hand.  Builds: MALE SLIM HEAVY FEMALE or build(...).
Two hands: two('pray'|'sukuna'|'dog'|'clap'|'reach', R=, pos=, gap=, bld=, **style) -> [L, R];  grip_wrist(target_hand, at=, turn=) -> the gripping Hand.
Props: hammer_props(h), nail_props(h), rod(p0, p1, r0, r1, g='prop'|'steel', ph=k): pass as render(..., props=[...]); fingers occlude them correctly.
Styles (Hand keywords): nails='short'|'dark'|'claw'|None, bands=((y0,y1),...) black rings down the forearm, tape=dict(fingers=FING, span=, fspan=, pitch=)
bandage wraps, cuff=dict(kind='uniform'|'kimono', y=distance below wrist, flare=), marks=False to drop wrinkles.  Palettes: PAL day night fire pale ink, or a dict
of overrides (skin shade line nail nail_shade nail_dark band tape tape_shade cuff cuff_shade cuff_light kimono... steel prop).

Drawn g (screen px, occlusion already solved): g.groups back to front [{id, hand, name, polys (visible region), depth}], g.mask (whole silhouette), g.shadow {id: polys,
clip to the group}, g.strokes [{kind out|crease, P, nr outward normal, wt weight, z, group, hand, inner}], g.creases [{kind, P, w}], g.nails, g.bands, g.tapes,
g.pts[hand][(finger, 'mcp'|'pip'|'dip'|'tip'|'pad')] + 'palm' 'back' 'wrist', g.wrist[hand] = forearm rings [{y, P ring px, near mask, c}], g.gb buffers (z, N, gid).
Pieces of ink(): fill_flat, fill_shadow, fill_overlays, ink_lines.  More light: rim_polys(g, dir), depth_polys(g, light, thr) + fill_polys(cel, g, polys, col).
Ink passages: brush_lines(cel, g, w), feather(cel, g), hatch(cel, g, polys).   Keep `seed` fixed while a pose holds.
"""
import numpy as np, math, cv2
from . import handsdf as S
from .handsdf import K

FING = ('index', 'middle', 'ring', 'little')
GID = dict(palm=1, index=2, middle=3, ring=4, little=5, thumb=6, cuff=7, nail_index=8, nail_middle=9, nail_ring=10, nail_little=11, nail_thumb=12, prop=13, cuffin=14, steel=15)
GNAME = {v: k for k, v in GID.items()}


# ----------------------------------------------------------------- rotations
def Rx(a): c, s = math.cos(a), math.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def Ry(a): c, s = math.cos(a), math.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def Rz(a): c, s = math.cos(a), math.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
def unit(v): v = np.asarray(v, float); return v / (np.linalg.norm(v) + 1e-12)
def D(a): return math.radians(a)


def euler(yaw=0., pitch=0., roll=0.):
    """hand orientation in degrees.  0,0,0 = back of the hand to camera, fingers up.  yaw 180 = palm to camera;
    pitch > 0 tips the fingers toward the camera; roll > 0 turns it counter-clockwise on screen."""
    return Rz(D(roll)) @ Rx(D(pitch)) @ Ry(D(yaw))


# ----------------------------------------------------------------- builds
# unit = wrist to middle knuckle.  canonical hand = LEFT: x toward the thumb, y to the fingers, z out of the back of the hand
def build(len=1.0, wid=1.0, boxy=.50, tip=.55, knuckle=1.0, palm=1.0, wrist=1.0, meat=1.0):
    """male default.  len finger length, wid finger thickness, boxy plane-ness of fingers, tip squareness, meat pads"""
    return dict(len=len, wid=wid, boxy=boxy, tip=tip, knuckle=knuckle, palm=palm, wrist=wrist, meat=meat)


MALE = build()
SLIM = build(len=1.05, wid=.92, boxy=.46, tip=.45, wrist=.95)            # Gojo, Megumi: long elegant
HEAVY = build(len=.98, wid=1.12, boxy=.58, tip=.65, knuckle=1.12, palm=1.08, wrist=1.16, meat=1.12)   # Todo, Sukuna, Yuji fist
FEMALE = build(len=1.02, wid=.83, boxy=.34, tip=.20, knuckle=.9, palm=.92, wrist=.86, meat=.95)

MCP = dict(index=(.305, .965, .015), middle=(.085, 1.0, .045), ring=(-.135, .945, .030), little=(-.345, .845, -.035))
LEN = dict(index=(.485, .295, .225), middle=(.535, .345, .235), ring=(.490, .320, .232), little=(.385, .235, .205))
WID = dict(index=(.112, .106, .092, .080), middle=(.114, .108, .094, .082), ring=(.108, .102, .089, .077), little=(.098, .090, .079, .068))
FAN = dict(index=-3., middle=0., ring=4., little=9.)         # neutral abduction, degrees, + = away from the thumb
ROLL = dict(index=5., middle=0., ring=-6., little=-14.)      # arch: how the flexion plane tilts
MCB = dict(index=(.135, .09, -.02), middle=(.045, .10, .01), ring=(-.05, .10, .0), little=(-.135, .08, -.045))


def _prof(kn, n=K):
    """smooth profile through knots evenly spaced on 0..1"""
    kn = np.asarray(kn, float); t = np.linspace(0, len(kn) - 1, n)
    i = np.minimum(t.astype(int), len(kn) - 2); f = t - i
    p = np.r_[2 * kn[0] - kn[1], kn, 2 * kn[-1] - kn[-2]]
    p0, p1, p2, p3 = p[i], p[i + 1], p[i + 2], p[i + 3]
    m1, m2 = (p2 - p0) / 2, (p3 - p1) / 2
    cr = (2 * f ** 3 - 3 * f ** 2 + 1) * p1 + (f ** 3 - 2 * f ** 2 + f) * m1 + (-2 * f ** 3 + 3 * f ** 2) * p2 + (f ** 3 - f ** 2) * m2
    ln = p1 * (1 - f) + p2 * f
    return .55 * cr + .45 * ln


class Frame:
    """local -> hand transform"""
    def __init__(self, R=None, o=None): self.R = np.eye(3) if R is None else R; self.o = np.zeros(3) if o is None else np.asarray(o, float)
    def move(self, v): return Frame(self.R, self.o + self.R @ np.asarray(v, float))
    def rot(self, Q): return Frame(self.R @ Q, self.o)
    def pt(self, v): return np.asarray(v, float) @ self.R.T + self.o


def basis(y, zref):
    """frame whose y axis = y and z axis as close as possible to zref; columns = axes"""
    y = unit(y); z = np.asarray(zref, float) - y * np.dot(zref, y); z = unit(z); x = np.cross(y, z)
    return np.stack([x, y, z], -1)


class Skel:
    """posed skeleton of one hand in canonical (left) coordinates + its primitives"""
    def __init__(self, pose, bld, style=None):
        self.pose = pose; self.b = bld; self.style = style or {}; self.fr = {}; self.prims = []; self.chain = {}; self.body = []; self.ext = []
        self._fk(); self._solid(); self._extras()

    # ---- forward kinematics
    def _fk(self):
        p, b = self.pose, self.b
        for f in FING:
            mcp, pip, dip, ab, tw = p[f]
            L = np.array(LEN[f]) * b['len']
            o = np.array(MCP[f]) * [b['palm'], 1, 1]
            # metacarpal cup of ring/little
            cup = p.get('cup', 0.) * dict(index=0, middle=.1, ring=.55, little=1.)[f]
            o = o + np.array([0.02, -.02, -.13]) * cup
            fan = FAN[f] * max(0., math.cos(D(mcp))) + ab
            A = Frame(np.eye(3), o).rot(Ry(D(ROLL[f] - 14 * cup + tw))).rot(Rz(D(fan)))
            P1 = A.rot(Rx(-D(mcp))); P2 = P1.move((0, L[0], 0)).rot(Rx(-D(pip))); P3 = P2.move((0, L[1], 0)).rot(Rx(-D(dip)))
            self.fr[f] = (P1, P2, P3); self.fr[f + '_L'] = L; self.fr[f + '_mc'] = (np.array(MCB[f]) * [b['palm'], 1, 1], o)
        rad, pal, rho, mcp, ip = p['thumb']
        y = np.array([math.sin(D(rad)) * math.cos(D(pal)), math.cos(D(rad)) * math.cos(D(pal)), -math.sin(D(pal))])
        zr = np.array([math.cos(D(rho)), 0, math.sin(D(rho))])
        L = np.array((.47, .375, .285)) * [1, .5 + .5 * b['len'], .5 + .5 * b['len']]
        C = Frame(basis(y, zr), np.array((.185, .10, -.085)) * [b['palm'], 1, 1])
        T1 = C.move((0, L[0], 0)).rot(Rx(-D(mcp))); T2 = T1.move((0, L[1], 0)).rot(Rx(-D(ip)))
        self.fr['thumb'] = (C, T1, T2); self.fr['thumb_L'] = L
        fl, dev = p.get('wrist', (0., 0.))
        W = Frame(Rz(D(dev)) @ Rx(D(fl)), np.array((0., .07, -.02)))
        self.fr['wrist'] = W.rot(np.diag([-1., -1., 1.]))          # y axis runs down the forearm, z still dorsal, x flips

    # ---- primitives
    def _add(self, where, F, L, a, zd, zp, cap=(.08, .08), boxy=.5, g='palm', k=.0, typ=0, ph=-1):
        T = np.stack([_prof(a), _prof(zd), _prof(zp)])
        self.prims.append(dict(T=typ, F=F, L=L, TAB=T, cap=cap, boxy=boxy, g=g, k=k, where=where, ph=ph)); return len(self.prims) - 1

    def _solid(self):
        p, b = self.pose, self.b; kn, meat = b['knuckle'], b['meat']; pw = b['palm']
        add = self._add
        # --- palm body
        for f in FING:
            base, o = self.fr[f + '_mc']; d = o - base; L = np.linalg.norm(d)
            z = np.array([math.sin(D(ROLL[f]) * 1.2), 0, math.cos(D(ROLL[f]))])
            B = Frame(basis(d, z), base); w = WID[f][0]
            fx = max(0., min(1., p[f][0] / 90.))                       # flexed knuckle: the metacarpal head stands out
            add('body', B, L, np.array([.080, .088, .096, .108, .114 * kn]) * (w / .112), np.array([.115, .112, .106, .104, (.108 + .012 * fx) * kn]), np.array([.10, .11, .115, .13, .125 * meat]), (.07, .080 + .018 * fx * kn), .42, 'palm', .10)
        # palmar slab and distal pad
        B = Frame(np.eye(3), np.array((-.01, .10, -.06)))
        add('body', B, .74, np.array([.17, .24, .31, .345, .345]) * pw, [.05] * 5, np.array([.10, .10, .09, .10, .115]) * meat, (.07, .10), .75, 'palm', .12)
        # hypothenar
        hy0 = np.array((-.165 * pw, .10, -.085)); hy1 = np.array((-.325 * pw, .70, -.085)) + (np.array(self.fr['little_mc'][1]) - np.array(MCP['little']) * [pw, 1, 1]) * .8
        add('body', Frame(basis(hy1 - hy0, (-.25, 0, 1)), hy0), np.linalg.norm(hy1 - hy0), [.10, .132, .135, .12, .10], [.08, .10, .10, .09, .08], np.array([.11, .15, .145, .12, .09]) * meat, (.07, .11), .25, 'palm', .12)
        # thumb metacarpal + thenar
        C, T1, T2 = self.fr['thumb']; TL = self.fr['thumb_L']
        add('body', C, TL[0], np.array([.09, .140, .150, .125, .108]) * (.5 + .5 * b['wid']), np.array([.07, .095, .10, .10, .098]), np.array([.10, .19, .20, .145, .10]) * meat, (.06, .09), .30, 'palm', .11, ph=-2)
        # thumb web
        I0, I1 = self.fr['index_mc']; q0 = C.pt((0, TL[0] * .30, -.02)); q1 = C.pt((0, TL[0] * 1.0, -.03)); q2 = I0 + (I1 - I0) * .78 + np.array((.02, 0, -.05))
        opp = max(0., min(1., (p['thumb'][1] - 35) / 30.))
        q1 = q1 * (1 - .28 * opp) + T1.pt((0, TL[1] * .35, -.02)) * .28 * opp; q2 = q2 + np.array((.0, .06, -.05)) * opp
        self._tri(q0, q1, q2, .050 + .02 * opp, 'palm', .16)
        q3 = I0 + (I1 - I0) * .25 + np.array((.0, 0, -.04)); self._tri(q0, q2, q3, .06, 'palm', .14)
        # finger webs
        for f, g in zip(FING[:-1], FING[1:]):
            a = self.fr[f][0].pt((-.02, self.fr[f + '_L'][0] * .25, -.035)); c = self.fr[g][0].pt((.02, self.fr[g + '_L'][0] * .25, -.035))
            d = c - a; L = np.linalg.norm(d)
            add('body', Frame(basis(d, (0, 0, 1)), a), L, [.072] * 5, [.040] * 5, np.array([.070] * 5) * meat, (.05, .05), .3, 'palm', .07)
        # wrist + forearm
        W = self.fr['wrist']; wr = b['wrist']
        arm = self.style.get('arm', 2.4) + .1; self.arm = arm
        _at = lambda v: np.interp(np.linspace(0, arm, 9), np.linspace(0, 2.5, 9), v)
        add('body', W.move((0, -.10, 0)), arm, _at([.262, .262, .262, .285, .318, .350, .375, .39, .40]) * wr, _at([.125, .150, .170, .195, .22, .24, .26, .27, .28]) * wr, _at([.150, .172, .188, .21, .24, .26, .28, .29, .30]) * wr, (.10, .12), .38, 'palm', .10)
        U = W.move((.145 * wr, .13, .085 * wr)); add('body', U, .05, [.078 * wr] * 5, [.078 * wr] * 5, [.078] * 5, (.08, .10), .0, 'palm', .09)          # ulna head
        # --- fingers
        for f in FING:
            F = self.fr[f]; L = self.fr[f + '_L']; w0, w1, w2, w3 = np.array(WID[f]) * b['wid']; bx = b['boxy']
            fx = max(0., min(1., p[f][1] / 100.)); k1 = .5 + .5 * kn; fat = (.5 + .5 * meat) * (1 + .16 * fx)
            c0 = len(self.prims)
            add('chain', F[0], L[0], [w0 * 1.0, w0 * .93, (w0 + w1) * .435, w1 * .89, w1 * .97 * k1], [w0 * .95, w0 * .86, w1 * .81, w1 * .82, w1 * .93 * kn], np.array([w0 * 1.0, w0 * 1.06, w0 * 1.0, w1 * .92, w1 * .76]) * fat, (w0 * .9, w1 * (.82 + .10 * fx)), bx, f, .0, ph=0)
            add('chain', F[1], L[1], [w1 * .97 * k1, w1 * .90, w1 * .86, w2 * .93, w2 * .975], [w1 * .92 * kn, w1 * .82, w1 * .75, w2 * .79, w2 * .85], np.array([w1 * .76, w1 * .97, w1 * .98, w2 * .91, w2 * .76]) * fat, (w1 * (.82 + .06 * fx), w2 * .78), bx, f, .0, ph=1)
            tp = b['tip']; s = L[2] / .23; tc = .078 * s
            add('chain', F[2], L[2] - tc, [w2 * .975, w2 * .95, w2 * .945, w3 * (1.0 + .06 * tp), w3 * (.88 + .12 * tp)], [w2 * .85, w2 * .78, w2 * .70, w3 * .66, w3 * .54], np.array([w2 * .76, w2 * .95, w3 * 1.08, w3 * 1.10, w3 * .90]), (w2 * .78, tc), bx * .45 + .28 * tp, f, .0, ph=2)
            if p[f][1] > 75 and p[f][2] > 35:                          # flesh squeezed inside a closed finger
                q0 = F[0].pt((0, L[0] * .62, -w1 * .9)); q1 = F[2].pt((0, L[2] * .30, -w2 * .3)); d = q1 - q0
                if np.linalg.norm(d) > .02: add('chain', Frame(basis(d, F[0].R[:, 2]), q0), np.linalg.norm(d), [w1 * .80] * 5, [w1 * .75] * 5, [w1 * .75] * 5, (w1 * .6, w1 * .6), .2, f, .0, ph=1)
            self.chain[f] = (c0, len(self.prims))
        # --- thumb
        c0 = len(self.prims); tw = .5 + .5 * b['wid']; tp = b['tip']
        add('chain', T1, TL[1], np.array([.104, .096, .092, .100, .110]) * tw, np.array([.100, .090, .084, .088, .092 * kn]) * tw, np.array([.100, .110, .105, .09, .084]) * tw, (.09, .085), bx, 'thumb', .0, ph=0)
        add('chain', T2, TL[2] - .090, np.array([.110, .110, .114, .112 + .004 * tp, .094 + .012 * tp]) * tw, np.array([.092, .078, .070, .062, .052]) * tw, np.array([.084, .102, .114, .112, .090]) * tw, (.085, .090), bx * .45 + .28 * tp, 'thumb', .0, ph=2)
        self.chain['thumb'] = (c0, len(self.prims))

    def _extras(self):
        st, b = self.style, self.b
        if st.get('nails') == 'claw':                                          # long dark pointed nails
            ex = st.get('claw', .06)
            for f in FING + ('thumb',):
                F = self.fr[f][2]; L = self.fr[f + '_L'][2]; i = self.chain[f][1] - 1
                pr = [q for q in self.prims[self.chain[f][0]:self.chain[f][1]] if q['ph'] == 2][0]; w = pr['TAB'][0, K // 2]; zd = pr['TAB'][1]
                y0 = L * .42; tot = L - y0 + ex
                Q = Frame(F.R, F.pt((0, y0, zd[int(K * .45)] * .70))).rot(Rx(D(-5)))
                self._add('ext', Q, tot, [w * .56, w * .66, w * .60, w * .40, w * .10], [.024, .026, .024, .016, .008], [.024, .024, .024, .016, .008], (.03, .02), .25, 'nail_' + f)
        c = st.get('cuff')
        if c:
            W = self.fr['wrist']; wr = b['wrist']; y0 = c.get('y', .30); fl = c.get('flare', 1.0); n = 9
            t = np.linspace(0, 1, n); base = np.interp(t * 2.4, np.linspace(0, 2.5, 9) - .1, np.array([.262, .262, .262, .285, .318, .350, .375, .39, .40]) * wr)
            if c.get('kind', 'uniform') == 'kimono':
                a = base * 1.5 + .30 * fl * (1 - t) ** .6; bz = base * .95 + .16 * fl * (1 - t) ** .6
            else:
                a = base * 1.06 + .055 + .03 * fl * (1 - t) ** 3; bz = base * .80 + .055 + .03 * fl * (1 - t) ** 3
            T_ = np.stack([_prof(a), _prof(bz), _prof(bz)])
            self.prims.append(dict(T=2, F=W.move((0, y0, 0)), L=2.45 - y0 * .0, TAB=T_, cap=(.0, .028), boxy=.35, g='cuff', k=0., where='ext', ph=-1))

    def _tri(self, a, b, c, th, g, k):
        x = unit(b - a); z = unit(np.cross(b - a, c - a)); y = np.cross(z, x)
        T = np.zeros((3, K)); T[0] = np.dot(c - a, x); T[1] = np.dot(c - a, y); T[2] = th
        self.prims.append(dict(T=1, F=Frame(np.stack([x, y, z], -1), a), L=float(np.linalg.norm(b - a)), TAB=T, cap=(.1, .1), boxy=0., g=g, k=k, where='body', ph=-1))


# ----------------------------------------------------------------- poses
def pose(index=(0, 0, 0, 0, 0), middle=(0, 0, 0, 0, 0), ring=(0, 0, 0, 0, 0), little=(0, 0, 0, 0, 0), thumb=(38, 24, 25, 12, 8), wrist=(0, 0), cup=0.):
    """each finger: (mcp flex, pip flex, dip flex, abduction (+ = away from thumb), twist) degrees.
    thumb: (radial abduction, palmar abduction, rho = where the nail looks: 0 sideways (radial) .. 90 dorsal .. negative = opposed, mcp flex, ip flex)"""
    q = lambda v: tuple(list(v) + [0] * (5 - len(v)))
    return dict(index=q(index), middle=q(middle), ring=q(ring), little=q(little), thumb=tuple(thumb), wrist=tuple(wrist), cup=cup)


POSES = dict(
    flat=pose(),
    relaxed=pose((18, 28, 14, 0), (26, 36, 18, 0), (32, 42, 20, 0), (36, 44, 22, 2), thumb=(34, 30, 18, 14, 18)),
    open=pose((4, 10, 5, -6), (2, 8, 4, 0), (4, 12, 6, 5), (6, 16, 8, 12), thumb=(54, 22, 30, 4, -6)),
    splay=pose((-4, 30, 22, -10), (-6, 32, 24, -1), (-4, 34, 24, 8), (-2, 36, 26, 18), thumb=(60, 26, 24, 6, 24)),           # tense claw
    fist=pose((86, 112, 70, 1), (88, 113, 72, 0), (90, 113, 70, -1), (92, 110, 66, -2), thumb=(19.6, 42, -30, 22, 48), cup=.15),
    point=pose((4, 6, 4, 0), (84, 108, 66, 0), (88, 110, 66, 0), (90, 108, 64, 0), thumb=(10, 45, -52, 20, 34), cup=.1),
    two=pose((3, 6, 4, 3), (3, 6, 4, -3), (82, 106, 60, 0), (88, 106, 60, 0), thumb=(-7, 51.5, -75, 29.4, 18), cup=.1),
    cross=pose((13, 9, 6, 7), (0, 16, 14, -11), (74, 104, 56, 2), (82, 104, 56, 0), thumb=(-10, 40, -75, -2, 36), cup=.2),
    pinch=pose((58, 58, 38, 2), (32, 38, 22, 1), (24, 30, 18, 3), (18, 26, 14, 6), thumb=(27.6, 37.8, -27, 27.8, -12)),
    hammer=pose((70, 96, 50, 0), (74, 96, 52, 0), (78, 96, 52, 0), (82, 94, 50, 0), thumb=(7.4, 41, -75, -2, 24), cup=.1),
    nails=pose((78, 98, 58, -3), (82, 100, 60, 0), (84, 100, 60, 3), (86, 98, 58, 8), thumb=(32, 50, -55, 30, 22), cup=.05),
    rear=pose((66, 84, 40, 2), (72, 86, 42, 0), (84, 100, 56, 0), (88, 102, 58, 0), thumb=(20, 30, -18, 0, 12), cup=.2),
    flick=pose((30, 30, 16, -3), (62, 88, 52, 0), (34, 44, 24, 4), (26, 38, 20, 8), thumb=(17.8, 67, -74, 49, 4)),
    pray=pose((0, 3, 2, -2), (-1, 3, 2, 0), (-2, 3, 2, -1), (-6, 3, 2, -4), thumb=(24, 8, 60, 2, 4)),
    sukuna=pose((30, 5, 4, 3), (28, 4, 4, 0), (80, 90, 46, 0), (84, 88, 44, 3), thumb=(4, 30, 30, 14, 26)),
    dog_near=pose((20, 78, 44, 4), (2, 8, 4, 2), (4, 8, 5, -4), (34, 22, 12, 16), thumb=(82, 2, 58, -6, -8)),
    dog_far=pose((4, 12, 6, 2), (4, 10, 6, 1), (6, 10, 6, -4), (38, 20, 12, 14), thumb=(54, 4, 55, -2, 4)),
    grip=pose((52, 62, 36, 0), (54, 64, 38, 0), (56, 66, 38, 0), (58, 66, 38, 2), thumb=(40, 62, -40, 18, 30), cup=.15),
    rest=pose((26, 22, 12, -3), (28, 22, 12, 0), (30, 24, 12, 3), (32, 26, 14, 8), thumb=(44, 34, 10, 10, 10)),
    thrust=pose((-6, 8, 4, -9), (-4, 6, 4, -1), (10, 22, 12, 6), (18, 34, 18, 14), thumb=(60, 18, 34, 2, -8)),
    palm_up=pose((16, 20, 10, -4), (18, 24, 12, 0), (22, 28, 14, 3), (26, 32, 16, 8), thumb=(48, 26, 26, 8, 8), cup=.35),
    reach=pose((8, 10, 6, -2), (22, 26, 14, 0), (34, 38, 20, 2), (42, 44, 22, 6), thumb=(40, 34, 10, 14, 16)),
    hook=pose((40, 74, 34, 2), (46, 78, 36, 0), (54, 80, 40, 0), (60, 80, 40, 2), thumb=(24, 40, -20, 12, 24), cup=.1),
    carry=pose((40, 88, 46, 0), (44, 90, 48, 0), (48, 90, 48, 0), (52, 88, 46, 0), thumb=(18, 50, -40, 26, 26), cup=.1),
    slap=pose((0, 4, 2, 2), (-1, 4, 2, 0), (-2, 4, 2, -1), (-6, 4, 2, -4), thumb=(26, 10, 55, 2, 4)),
)


def lerp_pose(a, b, t):
    a = POSES[a] if isinstance(a, str) else a; b = POSES[b] if isinstance(b, str) else b
    return {k: (tuple(np.asarray(a[k], float) * (1 - t) + np.asarray(b[k], float) * t) if not np.isscalar(a[k]) else a[k] * (1 - t) + b[k] * t) for k in a}


def blend(a, b, t, drag=.22, over=0., order=(0, 1, 2, 3), curl=0.):
    """a real drawn inbetween of two poses, t 0..1: the fingers do not move together.  order = who leads (0 first), drag = how late the
    last finger is, the tips lag behind their knuckles; over = overshoot past b (0.1 = 10 %); curl = degrees of extra bend on the way
    (an arc instead of a straight mix: fingers that unfold pass through a softer bend)."""
    a = POSES[a] if isinstance(a, str) else a; b = POSES[b] if isinstance(b, str) else b
    def e(x):
        x = min(1., max(0., x)); y = x * x * (3 - 2 * x)
        return y + over * math.sin(math.pi * min(1., max(0., (x - .55) / .45))) * (x < 1) if over else y
    out = {}
    for i, f in enumerate(FING):
        d = drag * order[i] / 3.
        s0 = e((t - d) / (1 - drag + 1e-6)) if drag > 0 else e(t); s1 = e((t - d - .06) / (1 - drag + 1e-6)); s2 = e((t - d - .12) / (1 - drag + 1e-6))
        if t >= 1: s0 = s1 = s2 = 1.
        arc = curl * math.sin(math.pi * min(1., max(0., t)))
        A, B = np.asarray(a[f], float), np.asarray(b[f], float); s = np.array([s0, s1, s2, s0, s0]); v = A * (1 - s) + B * s
        v[1] += arc; v[2] += arc * .6; out[f] = tuple(v)
    s = e((t - .05) / .95) if t < 1 else 1.
    out['thumb'] = tuple(np.asarray(a['thumb'], float) * (1 - s) + np.asarray(b['thumb'], float) * s)
    out['wrist'] = tuple(np.asarray(a['wrist'], float) * (1 - e(t * 1.15)) + np.asarray(b['wrist'], float) * e(t * 1.15)); out['cup'] = a['cup'] * (1 - e(t)) + b['cup'] * e(t)
    return out


def tremble(p, seed, amt=2.0):
    """nervous hand: a few degrees of joint noise (one drawing per seed)"""
    r = np.random.default_rng(seed); q = {}
    for k, v in p.items(): q[k] = tuple(np.asarray(v, float) + r.normal(0, amt, len(v)) * ([.5, 1, 1, .3, 0] if k in FING else [.4] * len(v))) if not np.isscalar(v) else v
    return q


# ----------------------------------------------------------------- scene
class Cam:
    """s = px per hand unit at depth 0, (cx,cy) = screen point of the scene origin; dist = eye distance in hand units
    (12 = long lens, 3 = strong wide-angle foreshortening)"""
    def __init__(self, cx, cy, s, dist=9.0): self.cx, self.cy, self.s, self.dist = cx, cy, s, dist
    def px(self, P):
        P = np.asarray(P, float); f = self.dist / np.maximum(self.dist - P[..., 2], .05)
        return np.stack([self.cx + P[..., 0] * f * self.s, self.cy - P[..., 1] * f * self.s], -1)


class Hand:
    def __init__(self, pose='relaxed', side='R', R=None, pos=(0, 0, 0), bld=MALE, scale=1.0, pivot=(0, .75, 0), **style):
        """pivot = hand point that lands on pos"""
        self.pose = POSES[pose] if isinstance(pose, str) else pose
        self.side = side; self.bld = bld; self.style = style; self.scale = scale
        self.sk = Skel(self.pose, bld, style)
        m = np.diag([1. if side == 'L' else -1., 1., 1.])
        self.A = (np.eye(3) if R is None else np.asarray(R, float)) @ m * scale      # linear part, hand -> world
        self.o = np.asarray(pos, float) - self.A @ np.asarray(pivot, float)

    def world(self, v): return np.asarray(v, float) @ self.A.T + self.o


def rod(p0, p1, r0, r1=None, boxy=0., cap=.012, g='prop', k=0., ph=-1):
    """solid prop (world / hand units): straight rod from p0 to p1, radii r0 -> r1.  pass a list of these as props="""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float); d = p1 - p0; r1 = r0 if r1 is None else r1
    z = (0, 0, 1) if abs(unit(d)[2]) < .9 else (1, 0, 0)
    t = np.linspace(r0, r1, K); T = np.stack([t, t, t])
    return dict(T=0, F=Frame(basis(d, z), p0), L=float(np.linalg.norm(d)), TAB=T, cap=(cap, cap), boxy=boxy, g=g, k=k, where='ext', ph=ph)


class Scene:
    """hands (and prop solids) packed for the ray caster"""
    def __init__(self, hands, props=()):
        self.hands = hands; R = []; O = []; L = []; T = []; TAB = []; CAP = []; BX = []; G = []; KB = []; HB = []; CH = []; CP = []; BC = []; BR = []; XT = []; SA = []; SB = []; SR = []; self.meta = []
        def push(pr, A, o, sc, hi):
            M = A @ pr['F'].R / sc; org = A @ pr['F'].o + o
            R.append(M.T); O.append(org); L.append(pr['L'] * sc); T.append(pr['T']); CAP.append(np.array(pr['cap']) * sc); BX.append(pr['boxy'])
            TAB.append(pr['TAB'] * sc); G.append(hi * 16 + GID[pr['g']]); KB.append(pr['k'] * sc)
            # bounding sphere
            if pr['T'] == 1:
                tri = np.array([[0, 0, 0], [pr['L'], 0, 0], [pr['TAB'][0, 0], pr['TAB'][1, 0], 0]]); c = tri.mean(0); r = np.linalg.norm(tri - c, axis=1).max() + pr['TAB'][2, 0]
            else:
                c = np.array([0, pr['L'] / 2, 0]); w = max(pr['TAB'][0].max(), 0); d = max(pr['TAB'][1].max(), pr['TAB'][2].max()); bx = 1.19 if pr['boxy'] > .05 else 1.0
                r = math.sqrt((pr['L'] / 2 + max(pr['cap'])) ** 2 + (w * bx) ** 2 + (d * bx) ** 2)
            BC.append(M @ c * sc + org); BR.append(r * sc); self.meta.append((hi, pr))
            if pr['T'] == 0:
                SA.append(org); SB.append(M @ np.array([0, pr['L'], 0]) * sc + org); SR.append((max(w, d, max(pr['cap'])) * bx + pr['k'] * .25 + .02) * sc)
            else: SA.append(BC[-1]); SB.append(BC[-1] + 1e-4); SR.append((r + pr['k'] * .25 + .02) * sc)
        for hi, h in enumerate(hands):
            b0 = len(T)
            for pr in h.sk.prims:
                if pr['where'] == 'body': push(pr, h.A, h.o, h.scale, hi)
            HB.append((b0, len(T)))
            for f in FING + ('thumb',):
                a, b = h.sk.chain[f]; c0 = len(T)
                for pr in h.sk.prims[a:b]: push(pr, h.A, h.o, h.scale, hi)
                F0 = h.sk.fr[f][0 if f != 'thumb' else 1]; w = h.sk.prims[a]['TAB'][0, 0]
                mc = h.world(F0.pt((0, .03, -.01)))
                CH.append((c0, len(T), hi)); CP.append((*mc, .12 * h.scale, .09 * h.scale, .30 * h.scale, .035 * h.scale))
            for pr in h.sk.prims:
                if pr['where'] == 'ext': XT.append(len(T)); push(pr, h.A, h.o, h.scale, hi)
        for pr in props:
            XT.append(len(T)); push(pr, np.eye(3), np.zeros(3), 1.0, 0)
        f = lambda v, d=np.float64: np.ascontiguousarray(np.array(v, d))
        self.R, self.O, self.L, self.T, self.TAB, self.CAP, self.BX = f(R), f(O), f(L), f(T, np.int64), f(TAB), f(CAP), f(BX)
        self.BC, self.BR, self.HB, self.KB = f(BC), f(BR), f(HB, np.int64), f(KB)
        self.CH, self.CP = f(CH, np.int64).reshape(-1, 3), f(CP).reshape(-1, 7)
        self.XT = np.array(XT, np.int64); self.G = f(G, np.int32); self.SA, self.SB, self.SR = f(SA), f(SB), f(SR)
        self.args = (self.T, self.R, self.O, self.L, self.TAB, self.CAP, self.BX, self.BC, self.BR, self.HB, self.KB, self.CH, self.CP, self.XT)

    def bounds(self, cam, arm=2.0):
        """screen box of all solids"""
        lo = np.full(2, 1e9); hi = np.full(2, -1e9); zmin, zmax = 1e9, -1e9
        for i in range(len(self.T)):
            c, r = self.BC[i], self.BR[i]
            if self.T[i] == 0:
                ax = self.R[i].T @ np.array([0, self.L[i], 0]); w = self.TAB[i].max() * 1.15 + self.CAP[i].max() * .3
                pts = [(self.O[i], w), (self.O[i] + ax, w)]
            else: pts = [(c, r)]
            for q, w in pts:
                zmin = min(zmin, q[2] - w); zmax = max(zmax, q[2] + w)
                z = min(q[2] + w, cam.dist * .92); f = cam.dist / (cam.dist - z)
                p = np.array([q[0], q[1]]) * f; lo = np.minimum(lo, p - w * f); hi = np.maximum(hi, p + w * f)
        return lo, hi, zmin, zmax


def gbuffer(sc, cam, light=None, res=620, clip=None, shadow=True, lo=None, hi=None):
    """ray cast -> dict(prim, gid, z, N, lit, px, x0, y0): buffer pixel (i,j) sits at plane point (x0+i px, y0 - j px)"""
    a, b, z0, z1 = sc.bounds(cam)
    if clip is not None:                                         # screen clip (x0,y0,x1,y1) px
        a = np.maximum(a, [(clip[0] - cam.cx) / cam.s, (cam.cy - clip[3]) / cam.s]); b = np.minimum(b, [(clip[2] - cam.cx) / cam.s, (cam.cy - clip[1]) / cam.s])
    sz = np.maximum(b - a, 1e-3); px = max(sz.max() / res, .35 / cam.s)
    gw, gh = int(sz[0] / px) + 4, int(sz[1] / px) + 4; x0, y0 = a[0] - 1.5 * px, b[1] + 1.5 * px
    L = unit(light) if light is not None else np.zeros(3)
    P, Z, N, LIT = S.cast(x0, y0, px, gw, gh, cam.dist, z0 - .02, min(z1 + .02, cam.dist * .97), *sc.args, sc.SA, sc.SB, sc.SR, L if shadow else np.zeros(3), 8)
    gid = np.where(P >= 0, sc.G[np.maximum(P, 0)], 0)
    return dict(prim=P, gid=gid, z=Z, N=N, lit=LIT, px=px, x0=x0, y0=y0, w=gw, h=gh)


# ----------------------------------------------------------------- contour tools (buffer px)
def _regions(mask, up=3, sig=.8, min_len=10, eps=0):
    """sub-pixel closed contours of a boolean mask, resampled at ~1 px"""
    ys, xs = np.nonzero(mask)
    if len(xs) < 6: return []
    a, b, c, d = max(0, xs.min() - 3), xs.max() + 4, max(0, ys.min() - 3), ys.max() + 4
    m = mask[c:d, a:b].astype(np.float32)
    m = cv2.copyMakeBorder(m, 3, 3, 3, 3, cv2.BORDER_CONSTANT)
    h, w = m.shape
    m = cv2.resize(cv2.GaussianBlur(m, (0, 0), sig), (w * up, h * up), interpolation=cv2.INTER_CUBIC)
    cs, _ = cv2.findContours((m > .5).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    for q in cs:
        q = (q[:, 0].astype(float) + .5) / up - .5 + np.array([a - 3, c - 3])
        if len(q) < min_len * up: continue
        out.append(_rs(_smooth(q, 5 * up // 2 * 2 + 1, True), 1.0, True))
    return out


def _smooth(P, n, closed):
    if n < 3 or len(P) < n + 2: return P
    k = np.hanning(n + 2)[1:-1]; k /= k.sum(); r = n // 2
    if closed: Q = np.vstack([P[-r:], P, P[:r]])
    else: Q = np.vstack([2 * P[0] - P[1:r + 1][::-1], P, 2 * P[-1] - P[-r - 1:-1][::-1]])
    return np.stack([np.convolve(Q[:, i], k, 'valid') for i in (0, 1)], -1)


def _rs(P, step, closed=False):
    """uniform arc-length resampling (closed: last point not repeated)"""
    Q = np.vstack([P, P[:1]]) if closed else np.asarray(P, float)
    d = np.hypot(*np.diff(Q, axis=0).T); s = np.r_[0, np.cumsum(d)]
    if s[-1] < 1e-6: return Q
    n = max(4, int(round(s[-1] / step)))
    t = np.linspace(0, s[-1], n + 1)[:-1] if closed else np.linspace(0, s[-1], n + 1)
    return np.c_[np.interp(t, s, Q[:, 0]), np.interp(t, s, Q[:, 1])]


def _dp(P, eps):
    """Douglas-Peucker: indices of the kept points of an open polyline"""
    keep = [0, len(P) - 1]; st = [(0, len(P) - 1)]
    while st:
        a, b = st.pop()
        if b - a < 2: continue
        v = P[b] - P[a]; L = np.hypot(*v); q = P[a + 1:b] - P[a]
        d = np.abs(v[0] * q[:, 1] - v[1] * q[:, 0]) / L if L > 1e-6 else np.hypot(q[:, 0], q[:, 1])
        i = int(np.argmax(d))
        if d[i] > eps: keep.append(a + 1 + i); st += [(a, a + 1 + i), (a + 1 + i, b)]
    return sorted(keep)


def angular(P, eps, amt=.5, closed=True, rnd=5):
    """straights against corners: pull a smooth curve toward its own simplified polygon (index preserving)"""
    n = len(P)
    if n < 8: return P
    if closed:
        c = P.mean(0); i0 = int(np.argmax(((P - c) ** 2).sum(1))); Q = np.roll(P, -i0, 0); Q = np.vstack([Q, Q[:1]])
    else: Q = np.asarray(P, float); i0 = 0
    if closed:
        far = int(np.argmax(((Q - Q[0]) ** 2).sum(1)))
        k = sorted(set(_dp(Q[:far + 1], eps) + [far + j for j in _dp(Q[far:], eps)]))
    else: k = _dp(Q, eps)
    R = np.empty_like(Q)
    for a, b in zip(k[:-1], k[1:]):
        s = np.r_[0, np.cumsum(np.hypot(*np.diff(Q[a:b + 1], axis=0).T))]; s = s / (s[-1] + 1e-9)
        R[a:b + 1] = Q[a] + (Q[b] - Q[a]) * s[:, None]
    R = Q * (1 - amt) + R * amt
    if closed: R = np.roll(R[:-1], i0, 0)
    if rnd >= 3: R = _smooth(R, rnd, closed)
    return R


def _nn(arr, P):
    h, w = arr.shape[:2]; x = np.clip(np.rint(P[:, 0]).astype(int), 0, w - 1); y = np.clip(np.rint(P[:, 1]).astype(int), 0, h - 1)
    return arr[y, x]


def _runs(flag, closed):
    """[(a,b)] index runs of True; closed runs may wrap (b may exceed n)"""
    n = len(flag)
    if flag.all(): return [(0, n)]
    if not flag.any(): return []
    s = 0
    if closed:
        s = int(np.argmin(flag)); flag = np.roll(flag, -s)
    d = np.diff(np.r_[0, flag.astype(int), 0]); a = np.nonzero(d == 1)[0]; b = np.nonzero(d == -1)[0]
    return [(int(p) + s, int(q) + s) for p, q in zip(a, b)]


def _close_gaps(flag, n, closed):
    f = flag.copy()
    for a, b in _runs(~flag, closed):
        if b - a <= n and b - a < len(flag):
            f[np.arange(a, b) % len(flag)] = True
    return f


class Drawn:
    """result of render(): every contour, shadow, crease in screen px (see module docstring)"""
    pass


SIL, OCC, HID, CRE, NON, TOU, XTO = range(7)
VIEW = np.array([0., 0., 1.])


def render(hands, cam, light=(-.5, .75, .6), res=560, clip=None, props=(), thr=.16, cast=True, style=.45, shadow_eps=.024, marks=True, cross=False):
    """solve one drawing.  hands = Hand or list of Hand (they may interlock); light = direction TO the light (x right, y up, z to viewer);
    clip = screen box that matters (x0,y0,x1,y1); thr = how much of the form falls in shadow (-1..1); style 0..1 = angularity of contours.
    cross=True: borders between two hands become their own strokes (kind 'cross'), for drawings painted hand by hand with only(): finish with cross_lines()."""
    hands = list(hands) if isinstance(hands, (list, tuple)) else [hands]
    sc = Scene(hands, props); L = unit(light)
    gb = gbuffer(sc, cam, L, res, clip, cast)
    g = Drawn(); g.sc, g.cam, g.gb, g.L, g.hands = sc, cam, gb, L, hands; g.cross = cross
    prim, Z, N, px = gb['prim'], gb['z'], gb['N'], gb['px']
    k = px * cam.s; org = np.array([cam.cx + cam.s * gb['x0'], cam.cy - cam.s * gb['y0']])
    g.b2s = lambda P: np.asarray(P, float) * k + org
    g.s2b = lambda P: (np.asarray(P, float) - org) / k
    g.k = k; u = 1 / px                                                     # buffer px per hand unit
    g.u = u
    ph = np.array([m[1].get('ph', -1) for m in sc.meta] + [-1])             # phalanx index per primitive
    g.PH = np.where(prim >= 0, ph[prim], -1)
    _cuff_inside(g)
    _tidy(g, u)
    gid = gb['gid']
    eps_d = .030
    g.groups = []; g.strokes = []
    ids = [int(i) for i in np.unique(gid) if i > 0]
    P3z = {i: float(np.median(Z[gid == i])) for i in ids}
    for i in sorted(ids, key=lambda i: P3z[i]):
        m = gid == i; nm = GNAME[i % 16]; polys = []
        eps = (.012 if nm != 'palm' else .016) * u
        tiny = m.sum() < .006 * u * u
        for q in _regions(m):
            if len(q) < .10 * u: continue
            q2 = angular(q, eps, style, True, 5)
            polys.append(q2)
            if not tiny: _classify(g, q, q2, m, i, None, u, eps_d)
        g.groups.append(dict(id=i, hand=i // 16, name=nm, polys=[g.b2s(q) for q in polys], depth=P3z[i]))
        if nm in FING or nm in ('thumb', 'palm', 'prop', 'steel'):                             # phalanx over phalanx; thenar over palm
            for s in np.unique(g.PH[m]):
                ms = m & (g.PH == s)
                if ms.sum() < 12 or ms.sum() == m.sum(): continue
                for q in _regions(ms):
                    if len(q) < .10 * u: continue
                    _classify(g, q, angular(q, eps, style, True, 5), ms, i, m, u, eps_d)
    _join(g)
    g.mask = [g.b2s(angular(q, .012 * u, style, True, 5)) for q in _regions(gid > 0)]
    g.core = [g.b2s(q) for q in _regions(cv2.erode((np.isin(gid % 16, (1, 2, 3, 4, 5, 6))).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0)]
    _shadows(g, thr, u, shadow_eps)
    _overlays(g)
    g.creases = []
    if marks: _marks(g)
    _folds(g)
    for t in g.tapes:
        keep = []
        for c in g.creases:
            b = g.s2b(c['P']); keep.append(_nn(t['mask'], b).mean() < .35)
        g.creases = [c for c, k_ in zip(g.creases, keep) if k_]
    _anchors(g)
    for st in g.strokes: st['P'] = g.b2s(st['P'])
    return g


def _hit(g, m):
    """world points of buffer pixels in mask m"""
    gb = g.gb; j, i = np.nonzero(m); z = gb['z'][j, i].astype(float); f = (g.cam.dist - z) / g.cam.dist
    return j, i, np.stack([(gb['x0'] + i * gb['px']) * f, (gb['y0'] - j * gb['px']) * f, z], -1)


def _tidy(g, u):
    """specks: small islands of one group join their neighbour (or vanish into the background) before anything is contoured"""
    gid, Z, prim, PH = g.gb['gid'], g.gb['z'], g.gb['prim'], g.PH
    for _ in range(2):
        for i in [int(v) for v in np.unique(gid) if v > 0]:
            m = (gid == i).astype(np.uint8)
            nl, lab, st, _c = cv2.connectedComponentsWithStats(m, connectivity=8)
            if nl < 2: continue
            for j in range(1, nl):
                a = st[j, 4]
                if a > .020 * u * u: continue
                x, y, w, h = st[j, :4]; y0, y1, x0, x1 = max(0, y - 3), y + h + 3, max(0, x - 3), x + w + 3
                c = (lab[y0:y1, x0:x1] == j).astype(np.uint8); dt = cv2.distanceTransform(np.pad(c, 1), cv2.DIST_L2, 3).max()
                thin = dt < .026 * u
                if a > .0035 * u * u and not thin: continue
                ring = (cv2.dilate(c, np.ones((5, 5), np.uint8)) > 0) & (c == 0)
                nb = gid[y0:y1, x0:x1][ring]
                if len(nb) == 0: continue
                v, n = np.unique(nb, return_counts=True)
                same = [(k_, q) for q, k_ in zip(v, n) if q > 0 and q // 16 == i // 16]
                bg = n[v == 0].sum() if (v == 0).any() else 0
                sub = gid[y0:y1, x0:x1]
                if bg > .42 * len(nb) and a < .012 * u * u: sub[c > 0] = 0; Z[y0:y1, x0:x1][c > 0] = -1e3; prim[y0:y1, x0:x1][c > 0] = -1; PH[y0:y1, x0:x1][c > 0] = -1
                elif same: sub[c > 0] = max(same)[1]; PH[y0:y1, x0:x1][c > 0] = -1


def _cuff_inside(g):
    sc, gb = g.sc, g.gb
    for idx in np.nonzero(sc.T == 2)[0]:
        m = gb['prim'] == idx
        if not m.any(): continue
        j, i, P = _hit(g, m); q = (P - sc.O[idx]) @ sc.R[idx].T; n = gb['N'][j, i] @ sc.R[idx].T
        ins = (n[:, 0] * q[:, 0] + n[:, 2] * q[:, 2]) < -.02
        gb['gid'][j[ins], i[ins]] = (sc.G[idx] // 16) * 16 + GID['cuffin']


def _classify(g, q, q2, m, i, grp, u, eps_d):
    """split the contour q (region m of group i) into drawn runs; q2 = stylised twin of q.  grp: mask of the whole group for inner boundaries"""
    gb = g.gb; gid, Z, N, PH = gb['gid'], gb['z'], gb['N'], g.PH
    n = len(q); T = np.roll(q, -2, 0) - np.roll(q, 2, 0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9
    nr = np.c_[-T[:, 1], T[:, 0]]
    if _nn(m, q + nr * 1.6).mean() > _nn(m, q - nr * 1.6).mean(): nr = -nr       # nr points outward
    Pi, Po = q - nr * 1.7, q + nr * 1.9
    zi = _nn(Z, Pi); go = _nn(gid, Po); zo = _nn(Z, Po); Ni = _nn(N, q - nr * 2.6); No = _nn(N, q + nr * 2.8); pi = _nn(PH, Pi); po = _nn(PH, Po)
    dz = zi - zo
    cos = (Ni * No).sum(1); conc = ((No - Ni)[:, :2] * np.c_[nr[:, 0], -nr[:, 1]]).sum(1) < -.15     # screen y is down, world y up
    nm = GNAME[i % 16]; cl = np.full(n, NON)
    same_hand = (go // 16) == (i // 16); on = go % 16
    fing = lambda v: (v >= 2) & (v <= 6)
    if grp is None:
        cl[go == 0] = SIL
        o = go > 0
        front = o & (dz > eps_d); back = o & (dz < -eps_d); near = o & ~front & ~back
        cl[front] = OCC; cl[back] = HID
        rel = same_hand & ((fing(np.array(i % 16)) & (on == 1) & (pi == 0)) | (fing(on) & (i % 16 == 1) & (po == 0)))
        nail = (i % 16 >= 8) | (on >= 8)
        cl[near & rel & conc & (cos < .55)] = CRE
        c2 = near & ~rel; pick = (i < go) | nail
        cl[c2 & pick] = TOU; cl[c2 & ~pick] = HID
        if g.cross:
            oth = o & ~same_hand; cl[oth & (front | (near & pick))] = XTO
        cl[front & rel & (dz < .13)] = CRE; cl[front & rel & (cos > .72) & (dz < .075)] = NON      # a finger base that leaves its own palm: at most a fold
    else:
        o = (go == i); cl[:] = HID
        front = o & (dz > eps_d); cl[front] = OCC
        near = o & (np.abs(dz) <= eps_d); c1 = near & conc & (cos < (.40 if nm != 'palm' else .1))
        cl[near] = NON; cl[c1] = CRE
        if nm != 'palm': cl[c1 & (pi > po)] = NON
        else: cl[front & (dz < eps_d * 1.8)] = NON
    for kinds, kind in (((SIL, OCC, TOU), 'out'), ((CRE,), 'crease'), ((XTO,), 'cross')):
        f = _close_gaps(np.isin(cl, kinds), max(4, int(.030 * u)) if kind != 'crease' else 4, True)
        for a, b in _runs(f, True):
            ln = b - a
            idx = np.arange(a, b) % n; c = cl[idx]
            if ln < (.06 if kind == 'out' else .05) * u and not (kind == 'out' and (c == SIL).mean() > .5 and ln > .025 * u): continue
            wt = np.where(c == SIL, 1.0, np.where(c == OCC, .74, .60)) if kind == 'out' else np.full(ln, .95 if kind == 'cross' else 1.)
            go_ = go[idx]; other = int(np.bincount(go_[go_ > 0]).argmax()) // 16 if (go_ > 0).any() else -1
            e0 = c[:4]; e1 = c[-4:]
            g.strokes.append(dict(kind=kind, gid=i, hand=i // 16, group=nm, closed=ln >= n, inner=grp is not None, P=q2[idx], nr=nr[idx], wt=wt, z=zi[idx], other=other,
                                  next=[int(cl[(a - 3) % n]), int(cl[(b + 2) % n])], ends=[SIL if (e0 == SIL).any() else OCC, SIL if (e1 == SIL).any() else OCC], joined=[False, False]))


def _rev(st):
    for k in ('P', 'nr', 'wt', 'z'): st[k] = st[k][::-1]
    for k in ('next', 'ends', 'joined'): st[k] = st[k][::-1]
    return st


def _join(g, gap=7.0, ang=.80):
    """silhouette strokes that continue each other across two groups become one stroke"""
    S_ = g.strokes
    def tan(P, e): 
        n = min(10, len(P) - 1); v = P[n] - P[0] if e == 0 else P[-1 - n] - P[-1]
        return v / (np.hypot(*v) + 1e-9)                                     # pointing into the stroke
    while True:
        best = None
        c = [(a, e) for a, st in enumerate(S_) if st['kind'] == 'out' and not st['inner'] and not st['closed'] for e in (0, 1) if st['ends'][e] == SIL]
        for x in range(len(c)):
            for y in range(x + 1, len(c)):
                (a, e), (b, f) = c[x], c[y]
                if a == b or S_[a]['hand'] != S_[b]['hand']: continue
                d = np.hypot(*(S_[a]['P'][0 if e == 0 else -1] - S_[b]['P'][0 if f == 0 else -1]))
                if d > gap: continue
                co = -np.dot(tan(S_[a]['P'], e), tan(S_[b]['P'], f))
                if co < math.cos(ang): continue
                s = d - 4 * co
                if best is None or s < best[0]: best = (s, a, e, b, f)
        if best is None: break
        _, a, e, b, f = best
        A, B = S_[a], S_[b]
        if e == 0: _rev(A)
        if f == 1: _rev(B)
        for k in ('P', 'nr', 'wt', 'z'): A[k] = np.concatenate([A[k][:-2], B[k][2:]]) if len(A[k]) > 4 and len(B[k]) > 4 else np.concatenate([A[k], B[k]])
        n = len(A['P']); 
        for k in ('next', 'ends', 'joined'): A[k][1] = B[k][1]
        A['P'] = _smooth(A['P'], 7, False)
        S_.pop(b)


@S.njit(fastmath=True, cache=True)
def _geo(V, Z, M, r, sig, tol):
    """geodesic separable blur of V (h,w) over the mask, never across depth steps"""
    h, w = V.shape; A = np.zeros_like(V); B = np.zeros_like(V)
    W = np.exp(-.5 * (np.arange(r + 1) / sig) ** 2)
    for j in range(h):
        for i in range(w):
            if not M[j, i]: continue
            s = V[j, i]; n = 1.
            for d in (-1, 1):
                z = Z[j, i]
                for k in range(1, r + 1):
                    x = i + d * k
                    if x < 0 or x >= w or not M[j, x] or abs(Z[j, x] - z) > tol: break
                    z = Z[j, x]; s += V[j, x] * W[k]; n += W[k]
            A[j, i] = s / n
    for j in range(h):
        for i in range(w):
            if not M[j, i]: continue
            s = A[j, i]; n = 1.
            for d in (-1, 1):
                z = Z[j, i]
                for k in range(1, r + 1):
                    y = j + d * k
                    if y < 0 or y >= h or not M[y, i] or abs(Z[y, i] - z) > tol: break
                    z = Z[y, i]; s += A[y, i] * W[k]; n += W[k]
            B[j, i] = s / n
    return B


def _shadows(g, thr, u, seps):
    """one hard angular shadow region per group -> polygons (extended past the group edge; clip to the group when filling)"""
    gb = g.gb; gid, N, lit, Z = gb['gid'], gb['N'], gb['lit'], gb['z']
    M = gid > 0
    sg = max(1.0, .030 * u); r = int(sg * 2.2) + 1; tol = max(.010, 2.2 * gb['px'])
    val = np.where(M, N @ g.L, 0).astype(np.float32)
    v = _geo(_geo(val, Z, M, r, sg, tol), Z, M, r, sg, tol)
    s2 = max(1.0, .012 * u); vf = _geo(val, Z, M, int(s2 * 2.2) + 1, s2, tol)
    fg = np.isin(gid % 16, (2, 3, 4, 5, 6)); wf = _geo(fg.astype(np.float32), Z, M, r, sg, tol); v = v * (1 - wf) + vf * wf
    li = _geo(lit.astype(np.float32), Z, M, r, sg * .7, tol)
    S0 = (np.minimum((v - thr) * 4, (li - .5) * 2) < 0) & M
    g.val = v; g.li = li
    S1 = S0.copy(); g.shadow = {}
    A = max(20, .022 * u * u)
    for val_ in (True, False):
        t = (S1 == val_) & M
        nl, lab, st, _ = cv2.connectedComponentsWithStats(t.astype(np.uint8), connectivity=4)
        for j in range(1, nl):
            if st[j, 4] < A:
                c = lab == j
                if st[j, 4] < A * .25 or fg[c].mean() < .5: S1[c] = not val_
    g.shade_mask = S1
    for G in g.groups: g.shadow[G['id']] = _poly_in(g, gid == G['id'], S1, seps * u)
    # shaded seams between two regions that meet without a line
    b = np.zeros_like(M)
    for ax in (0, 1):
        d = (gid != np.roll(gid, 1, ax)) & M & np.roll(M, 1, ax); b |= d | np.roll(d, -1, ax)
    b = cv2.dilate(b.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ok = S1 & (cv2.erode(S1.astype(np.uint8) | (~M).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & b
    g.seams = [g.b2s(q) for q in _regions(ok, min_len=6)]
    g.seams_by = {h_: [g.b2s(q) for q in _regions(ok & (gid // 16 == h_), min_len=6)] for h_ in range(len(g.hands))}


def _poly_in(g, m, s, eps, amt=.88, rnd=3):
    """polygons of region s inside m, extended past the border of m"""
    ys, xs = np.nonzero(m); pad = 10
    a, b, c, d = max(0, xs.min() - pad), min(m.shape[1], xs.max() + pad + 1), max(0, ys.min() - pad), min(m.shape[0], ys.max() + pad + 1)
    mm = m[c:d, a:b]; t = s[c:d, a:b] & mm
    if not t.any(): return []
    A = max(12, .005 * g.u * g.u)
    for val_ in (True, False):
        nl, lab, st, _ = cv2.connectedComponentsWithStats(((t == val_) & mm).astype(np.uint8), connectivity=4)
        for j in range(1, nl):
            if st[j, 4] < A: t[lab == j] = not val_
    if not t.any(): return []
    _, lab = cv2.distanceTransformWithLabels((~mm).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    zz = np.nonzero(mm.ravel())[0]; lut = np.zeros(lab.max() + 1, bool); lut[lab.ravel()[zz]] = t.ravel()[zz]
    e = cv2.copyMakeBorder(lut[lab].astype(np.uint8), 6, 6, 6, 6, cv2.BORDER_REPLICATE) > 0
    e[[0, 1, -1, -2]] = 0; e[:, [0, 1, -1, -2]] = 0
    out = []
    for q in _regions(e, sig=1.0):
        if len(q) < .10 * g.u: continue
        out.append(g.b2s(angular(q + np.array([a - 6, c - 6]), eps, amt, True, rnd)))
    return out


# ----------------------------------------------------------------- painted overlays: nails, bands, tape
def _prim_of(g, hi, f, ph):
    for i, (h, pr) in enumerate(g.sc.meta):
        if h == hi and pr['g'] == f and pr.get('ph', -1) == ph: return i
    return -1


def _local(g, idx, m):
    j, i, P = _hit(g, m); sc = g.sc
    return j, i, (P - sc.O[idx]) @ sc.R[idx].T


def _overlays(g):
    sc, gb = g.sc, g.gb; g.nails = []; g.bands = []; g.tapes = []
    for hi, h in enumerate(g.hands):
        st = h.style; sc_ = h.scale
        if st.get('nails', 'short') in ('short', 'dark'):
            for f in FING + ('thumb',):
                idx = _prim_of(g, hi, f, 2)
                if idx < 0: continue
                m = gb['prim'] == idx
                if m.sum() < 30: continue
                j, i, q = _local(g, idx, m); L = sc.L[idx]; cap = sc.CAP[idx, 1]; tot = L + cap
                t = np.clip(q[:, 1] / L, 0, 1); a = np.interp(t, np.linspace(0, 1, K), sc.TAB[idx, 0]); zd = np.interp(t, np.linspace(0, 1, K), sc.TAB[idx, 1])
                dk = st.get('nails') == 'dark'; x = q[:, 0] / (a * (.66 if dk else .74)); y = q[:, 1] / tot
                y0 = (.42 if f != 'thumb' else .40) + .06 * dk; r = np.clip(np.abs(x), 0, 1)
                top = .955 - .07 * x * x
                ok = (q[:, 2] > zd * .18) & (np.abs(x) < 1) & (y > y0 + .13 * r ** 3.0) & (y < top) & (gb['N'][j, i, 2] > .20)
                if ok.sum() < max(8, .0022 * g.u ** 2) or (h.pose['thumb' if f == 'thumb' else f][1] > 100 and f != 'thumb' and m.sum() < .030 * g.u ** 2): continue
                if gb['N'][j[ok], i[ok], 2].mean() < (.42 if dk else .25): continue
                mm = np.zeros(m.shape, bool); mm[j[ok], i[ok]] = True
                polys = [g.b2s(angular(p, .006 * g.u, .3, True, 3)) for p in _regions(mm, sig=.9, min_len=8) if len(p) > .06 * g.u]
                if polys: g.nails.append(dict(hand=hi, finger=f, polys=polys, dark=st.get('nails') == 'dark', gid=hi * 16 + GID[f], facing=float(gb['N'][j[ok], i[ok]][:, 2].mean())))
        W = h.sk.fr['wrist']; Aw = h.A / sc_
        def wl(m):
            j, i, P = _hit(g, m); c = ((P - h.o) @ Aw) / sc_                       # canonical hand coordinates
            return j, i, c, (c - W.o) @ W.R
        pm = gb['gid'] == hi * 16 + 1
        for a, b in st.get('bands', ()):
            if not pm.any(): break
            j, i, c, w = wl(pm); ok = (w[:, 1] > a) & (w[:, 1] < b)
            mm = np.zeros(pm.shape, bool); mm[j[ok], i[ok]] = True
            if mm.sum() > 10: g.bands.append(dict(hand=hi, gid=hi * 16 + 1, polys=_poly_in(g, pm, mm, .008 * g.u, .4, 5)))
        tp = st.get('tape')
        if tp:
            # wrap field over palm, wrist and proximal phalanges
            names = [GID['palm']] + [GID[f] for f in tp.get('fingers', ())]
            for gi in names:
                m = gb['gid'] == hi * 16 + gi
                if not m.any(): continue
                j, i, c, w = wl(m)
                if gi == 1:
                    s = c[:, 1] + .16 * c[:, 0] * np.sign(c[:, 2] + .03) ; lo, hi_ = tp.get('span', (-.55, 1.3))
                    ok = (c[:, 1] > lo) & (c[:, 1] < hi_ + .10 * c[:, 0])
                    # keep the thumb and its web out
                    C, T1, _ = h.sk.fr['thumb']; tc = (c - C.o) @ C.R
                    ok &= ~((tc[:, 1] > .30) & (np.abs(tc[:, 0]) < .2) & (c[:, 0] > .30))
                    per = tp.get('pitch', .135)
                else:
                    f = GNAME[gi]; F = h.sk.fr[f]; Lf = h.sk.fr[f + '_L'] if f != 'thumb' else h.sk.fr['thumb_L']
                    q0 = (c - F[0].o) @ F[0].R; s = q0[:, 1] + .12 * q0[:, 0]; lo, hi_ = tp.get('fspan', (-.2, .70))
                    ph_ = g.PH[j, i]; ok = (ph_ == 0) & (q0[:, 1] > Lf[0] * lo) & (q0[:, 1] < Lf[0] * hi_); per = .105
                mm = np.zeros(m.shape, bool); mm[j[ok], i[ok]] = True
                if mm.sum() < 40: continue
                F_ = np.full(m.shape, np.nan, np.float32); F_[j, i] = s
                polys = _poly_in(g, m, mm, .010 * g.u, .4, 5); lines = []
                lv0, lv1 = np.floor(np.nanmin(F_[mm]) / per), np.ceil(np.nanmax(F_[mm]) / per)
                for lv in np.arange(lv0, lv1 + 1):
                    t = mm & (np.nan_to_num(F_, nan=-9) > lv * per)
                    if t.sum() < 10 or t.sum() > mm.sum() - 10: continue
                    er = cv2.erode(mm.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                    for p in _regions(t):
                        fl = _nn(er, p)
                        for a, b in _runs(fl, True):
                            if b - a > .06 * g.u: lines.append(g.b2s(_smooth(p[np.arange(a, b) % len(p)], 9, False)))
                g.tapes.append(dict(hand=hi, gid=hi * 16 + gi, polys=polys, lines=lines, mask=mm))


# ----------------------------------------------------------------- authored marks on the surface
def _sz(sc, idx, x, y, side=1):
    """local z of the primitive's surface above (x,y)"""
    L = sc.L[idx]; t = np.clip(y / L, 0, 1); tt = np.linspace(0, 1, K)
    a = np.interp(t, tt, sc.TAB[idx, 0]); b = np.interp(t, tt, sc.TAB[idx, 1 if side > 0 else 2]); w = np.where(y < 0, -y / sc.CAP[idx, 0], np.where(y > L, (y - L) / sc.CAP[idx, 1], 0))
    u_ = np.abs(x) / a; B = sc.BX[idx]; lo = np.zeros_like(u_); hi = np.ones_like(u_) * 1.2
    for _ in range(18):
        v = (lo + hi) / 2; r = (1 - B) * np.sqrt(u_ ** 2 + v ** 2 + w ** 2) + B * (u_ ** 4 + v ** 4 + w ** 4) ** .25
        hi = np.where(r > 1, v, hi); lo = np.where(r > 1, lo, v)
    return lo * b * side


def _ring(sc, idx, y, phi):
    """local surface points at height y (array) and angle phi (array; 0 dorsal, + toward local x)"""
    y = np.broadcast_to(np.asarray(y, float), np.shape(phi)); t = np.clip(y / sc.L[idx], 0, 1); tt = np.linspace(0, 1, K); c, s = np.cos(phi), np.sin(phi)
    a = np.interp(t, tt, sc.TAB[idx, 0]); b = np.where(c > 0, np.interp(t, tt, sc.TAB[idx, 1]), np.interp(t, tt, sc.TAB[idx, 2]))
    u_, v = s / a, c / b; B = sc.BX[idx]; r = (1 - B) * np.hypot(u_, v) + B * (u_ ** 4 + v ** 4) ** .25
    return np.stack([s / r, y, c / r], -1)


def _emit(g, hi, P, kind, wf, gids, tol=.045, dark=False):
    """visible runs of world polyline P on the groups gids -> g.creases"""
    b = g.s2b(g.cam.px(P)); h, w = g.gb['gid'].shape
    inb = (b[:, 0] > 1) & (b[:, 0] < w - 2) & (b[:, 1] > 1) & (b[:, 1] < h - 2)
    vis = inb & np.isin(_nn(g.gb['gid'], b), gids) & (np.abs(_nn(g.gb['z'], b) - P[:, 2]) < tol)
    for a, e in _runs(_close_gaps(vis, 2, False), False):
        if e - a >= 4 and np.hypot(*(g.cam.px(P[e - 1]) - g.cam.px(P[a]))) > .035 * g.cam.s: g.creases.append(dict(hand=hi, kind=kind, w=wf, P=g.cam.px(P[a:e]), z=float(P[a:e, 2].mean())))


def _marks(g):
    sc = g.sc
    W = lambda idx, q: q @ sc.R[idx] + sc.O[idx]                                   # local -> world (R rows are axes)
    for hi, h in enumerate(g.hands):
        p = h.pose; lim = np.linspace
        gid0 = hi * 16
        if h.style.get('marks', True) is False: continue
        for f in FING + ('thumb',):
            th = f == 'thumb'; i0, i1, i2 = _prim_of(g, hi, f, 0), _prim_of(g, hi, f, 1), _prim_of(g, hi, f, 2)
            gids = [gid0 + GID[f], gid0 + 1]
            pip, dip = (p[f][4], None) if th else (p[f][1], p[f][2])
            jA, jB = (i0, i2) if th else (i0, i1)
            L0 = sc.L[jA]
            e = 1 if h.side == 'L' else -1
            if pip < 45:                                                          # straight: skin folds over the middle joint
                for dy, a0, a1, wf in ((-.028, -.62, .40, .46), (.030, -.35, .55, .40)):
                    ph = lim(a0, a1, 12); y = L0 + dy * h.scale + .010 * h.scale * np.cos(ph * 2.2)
                    _emit(g, hi, W(jA, _ring(sc, jA, np.minimum(y, L0 + sc.CAP[jA, 1] * .6), ph)), 'wrinkle', wf, gids)
            else:                                                                 # bent: corners of the knuckle box
                for sg in (-1, 1):
                    q = np.concatenate([_ring(sc, jA, lim(L0 * .80, L0 * .98, 6), np.full(6, sg * .78)), ])
                    _emit(g, hi, W(jA, q), 'knuckle', .46, gids, .06)
            if not th and dip is not None and dip < 40:
                L1 = sc.L[i1]; ph = lim(-.55, .35, 10)
                _emit(g, hi, W(i1, _ring(sc, i1, np.full(10, L1 * 1.0), ph)), 'wrinkle', .36, gids)
            # palmar flexion lines
            if pip < 60:
                ph = math.pi + lim(-.85, .55, 14); _emit(g, hi, W(jA, _ring(sc, jA, L0 * 1.0 + .006 * np.cos(ph * 3), ph)), 'flex', .50, gids)
                ph = math.pi + lim(-.2, .7, 10); _emit(g, hi, W(jA, _ring(sc, jA, np.full(10, L0 * .93), ph)), 'flex', .36, gids)
            if not th:
                if dip < 50:
                    L1 = sc.L[i1]; ph = math.pi + lim(-.7, .7, 12); _emit(g, hi, W(i1, _ring(sc, i1, np.full(12, L1 * 1.0), ph)), 'flex', .46, gids)
                if p[f][0] < 55:
                    ph = math.pi + lim(-.75, .75, 12); _emit(g, hi, W(i0, _ring(sc, i0, sc.L[i0] * .40 + .012 * np.cos(ph * 2), ph)), 'flex', .44, gids)
        # --- palm and back: probe the body surface
        tri = [(np.array(v, float) @ h.A.T) + h.o for v in ()]
        def skin(xy, side, kind, wf, n=22):
            from .ink import spl
            c = np.asarray(xy, float); c = spl(c, n=8) if len(c) > 2 else c[0] + (c[1] - c[0]) * np.linspace(0, 1, 14)[:, None]
            P0 = np.c_[c, np.full(len(c), .9 * side)]; Pw = np.ascontiguousarray(h.world(P0)); dr = np.ascontiguousarray(np.tile(h.A @ np.array([0, 0, -side], float) / h.scale, (len(c), 1)))
            a = sc.args; b0, b1 = sc.HB[hi]
            HB = np.array([[b0, b1]], np.int64)
            r = S.probe(Pw, dr, 2.0 * h.scale, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], HB, a[10], np.zeros((0, 3), np.int64), a[12], np.zeros(0, np.int64))
            ok = r > 0
            if ok.sum() < 4: return
            Q = Pw[ok] + dr[ok] * r[ok, None]
            _emit(g, hi, Q, kind, wf, [gid0 + 1], .05)
        tr = p['thumb']; fl = np.mean([p[f][0] for f in FING]); op = np.clip(tr[1] / 60, 0, 1)
        skin([(.30, .66), (.20, .63), (.10, .52), (.035, .36), (.02, .20), (.05, .09)], -1, 'palm', .62)                      # life line
        skin([(.27, .67), (.10, .585), (-.08, .50), (-.22, .42)], -1, 'palm', .50)                                            # head line
        skin([(-.37, .64), (-.22, .655), (-.05, .70), (.06, .78), (.11, .88)], -1, 'palm', .50)                               # heart line
        skin([(-.20, .015), (-.02, -.015), (.17, .02)], -1, 'palm', .42); skin([(-.12, -.07), (.02, -.085), (.13, -.065)], -1, 'palm', .30)
        for f, wf in (('index', .45), ('middle', .50), ('ring', .42), ('little', .36)):
            b_, o = h.sk.fr[f + '_mc']; fx = p[f][0]; r_ = WID[f][0]
            d = (o - b_)[:2] / np.linalg.norm((o - b_)[:2]); nn = np.array([d[1], -d[0]])
            if fx < 50:
                if f != 'little': skin([o[:2] + nn * r_ * .62 - d * .07, o[:2] + nn * r_ * .50 - d * .01, o[:2] + nn * r_ * .12 + d * .035], 1, 'knuckle', wf)
                if f in ('index', 'middle'): skin([o[:2] - d * .15 + nn * .03, o[:2] - d * .31 + nn * .02], 1, 'tendon', .38)
            else:
                if f != 'little': skin([o[:2] - nn * r_ * 1.02 - d * .17, o[:2] - nn * r_ * 1.05 - d * .10, o[:2] - nn * r_ * 1.0 - d * .0], 1, 'knuckle', wf * 1.1)     # valley to the next knuckle
                if f in ('index', 'middle'): skin([o[:2] + nn * r_ * .62 - d * .10, o[:2] + nn * r_ * .30 - d * .045, o[:2] - nn * r_ * .05 - d * .04], 1, 'knuckle', wf * .8)
        skin([(-.10, -.04), (-.075, -.085), (-.085, -.14)], 1, 'bone', .40)                                    # ulna head


def surface_curve(g, hi, xy, side=-1, n=8, tol=.06):
    """author a curve on the palm (side -1) or back (+1) in hand coordinates (x to the thumb, y to the fingers) -> list of visible screen polylines
    (full render only, before lite)"""
    from .ink import spl
    h = g.hands[hi]; sc = g.sc; c = np.asarray(xy, float); c = spl(c, n=n) if len(c) > 2 else c[0] + (c[-1] - c[0]) * np.linspace(0, 1, 12)[:, None]
    Pw = np.ascontiguousarray(h.world(np.c_[c, np.full(len(c), .9 * side)])); dr = np.ascontiguousarray(np.tile(h.A @ np.array([0, 0, -side], float) / h.scale, (len(c), 1))); a = sc.args
    r = S.probe(Pw, dr, 2.0 * h.scale, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], np.array([sc.HB[hi]], np.int64), a[10], np.zeros((0, 3), np.int64), a[12], np.zeros(0, np.int64))
    ok = r > 0; Q = Pw + dr * r[:, None]; b = g.s2b(g.cam.px(Q)); H_, W_ = g.gb['gid'].shape
    vis = ok & (b[:, 0] > 1) & (b[:, 0] < W_ - 2) & (b[:, 1] > 1) & (b[:, 1] < H_ - 2); vis &= (_nn(g.gb['gid'], b) == hi * 16 + 1) & (np.abs(_nn(g.gb['z'], b) - Q[:, 2]) < tol)
    return [g.cam.px(Q[x:y]) for x, y in _runs(_close_gaps(vis, 2, False), False) if y - x >= 4]


def _anchors(g):
    """screen points of every joint and tip, wrist rings"""
    g.pts = {}; g.wrist = {}
    for hi, h in enumerate(g.hands):
        d = {}
        for f in FING + ('thumb',):
            F = h.sk.fr[f]; L = h.sk.fr[f + '_L']
            J = [F[0].o, F[1].o, F[2].o, F[2].pt((0, L[2], 0))]
            for nme, q in zip(('mcp', 'pip', 'dip', 'tip') if f != 'thumb' else ('cmc', 'mcp', 'ip', 'tip'), J): d[f, nme] = g.cam.px(h.world(q))
            d[f, 'pad'] = g.cam.px(h.world(F[2].pt((0, L[2] * .65, -.08))))
        d['palm'] = g.cam.px(h.world((0, .5, -.12))); d['back'] = g.cam.px(h.world((0, .5, .12))); d['wrist'] = g.cam.px(h.world((0, 0, 0)))
        g.pts[hi] = d
        W = h.sk.fr['wrist']; idx = next(i for i, (hh, pr) in enumerate(g.sc.meta) if hh == hi and pr['where'] == 'body' and pr['L'] == h.sk.arm)
        rings = []
        for y in (.0, .15, .3, .45, .6, .9, 1.2, 1.6, 2.0, 2.4):
            ph = np.linspace(0, 2 * math.pi, 40, endpoint=False); q = _ring(g.sc, idx, np.full(40, (y + .10) * h.scale), ph); Pw = q @ g.sc.R[idx] + g.sc.O[idx]
            c = Pw.mean(0); vdir = np.array([0, 0, g.cam.dist]) - Pw; near = ((Pw - c) * vdir).sum(1) > 0
            rings.append(dict(y=y, P=g.cam.px(Pw), near=near, c=g.cam.px(c)))
        g.wrist[hi] = rings


# ----------------------------------------------------------------- palettes
PAL = dict(
    day=dict(skin='#f2cdb3', shade='#c48a7c', line='#17131a', nail='#f8dccb', nail_shade='#d09c90', nail_dark='#1a1218', light='#fff0de', band='#131018', tape='#ece6da', tape_shade='#a49aa8',
             cuff='#15141f', cuff_shade='#0c0b14', cuff_light='#4a3f7a', cuffin='#0a0910', prop='#8a6a48', prop_shade='#4a3630', steel='#8f97a6', steel_shade='#3e4354', kimono='#f3f0f2', kimono_shade='#9c94b4', kimono_light='#ffffff'),
    night=dict(skin='#b9a2ae', shade='#5d4d6e', line='#100d16', nail='#cbb4bb', nail_shade='#6d5a78', light='#cfe0ff', tape='#b4b2c4', tape_shade='#5e5a78', cuff='#12111b', cuff_shade='#090810', cuff_light='#3a3568'),
    fire=dict(skin='#f9cf72', shade='#a8602c', line='#3a0f0a', nail='#ffe09a', nail_shade='#b8702f', light='#fff3b8', band='#2a0d08', tape='#fbe2a6', tape_shade='#a86a3a'),
    pale=dict(skin='#f6e4da', shade='#b3a3bd', line='#17131a', nail='#fbeee6', nail_shade='#c2b0c4', light='#ffffff'),
    ink=dict(skin='#e9d6bd', shade='#a98c86', line='#0c0a0d', nail='#1a1218', nail_shade='#0c0a0d', light='#f6ecd8', band='#0c0a0d'),
)


def _corners(P, span, thr=.95):
    """indices where a stroke turns hard"""
    n = len(P); k = max(2, int(span))
    if n < 4 * k + 4: return []
    a = P[k:-k] - P[:-2 * k]; b = P[2 * k:] - P[k:-k]
    th = np.abs((np.arctan2(b[:, 1], b[:, 0]) - np.arctan2(a[:, 1], a[:, 0]) + math.pi) % (2 * math.pi) - math.pi)
    out = []; j = 0
    while j < len(th):
        if th[j] > thr:
            e = j
            while e + 1 < len(th) and th[e + 1] > thr: e += 1
            c = j + int(np.argmax(th[j:e + 1])) + k
            if 2 * k < c < n - 2 * k: out.append(c)
            j = e + k
        j += 1
    return out


def _tail(P, n, step, end=True):
    """extrapolate a stroke end along its own curvature"""
    Q = P if end else P[::-1]
    if len(Q) < 12 or n < 1: return P
    a = Q[-1] - Q[-6]; b = Q[-6] - Q[-11]
    th = math.atan2(a[1], a[0]); dth = np.clip((th - math.atan2(b[1], b[0]) + math.pi) % (2 * math.pi) - math.pi, -.22, .22) / 5
    p = Q[-1].copy(); ext = []
    for j in range(n):
        th += dth * (1 - j / n); p = p + step * np.array([math.cos(th), math.sin(th)]); ext.append(p)
    Q = np.vstack([Q, ext])
    return Q if end else Q[::-1]


def _clipg(cel, G):
    cel.ctx.save(); cel._path(G['polys']); cel.ctx.set_fill_rule(1); cel.ctx.clip()


def _mat(g, G):
    n = G['name']
    if n == 'cuff': return 'kimono' if g.hands[G['hand']].style['cuff'].get('kind') == 'kimono' else 'cuff'
    if n in ('cuffin', 'prop', 'steel'): return n
    return 'nail_dark' if n.startswith('nail') else 'skin'


def fill_flat(cel, g, pal):
    """colour flats of every region"""
    c = cel.ctx
    if g.core: cel.fill(g.core, pal['skin'], evenodd=True)
    for G in g.groups:
        if not G['polys']: continue
        col = pal[_mat(g, G)]
        cel.fill(G['polys'], col, evenodd=True)
        c.set_line_width(max(1.0, 1.8 * g.k))
        for q in G['polys']: cel._path(q); cel._src(col, 1.0); c.stroke()


def fill_shadow(cel, g, pal):
    for G in g.groups:
        S_ = g.shadow.get(G['id']); m = _mat(g, G)
        if not G['polys'] or m in ('cuffin', 'nail_dark'): continue
        _clipg(cel, G)
        if S_: cel.fill(S_, pal['shade' if m == 'skin' else m + '_shade'], evenodd=True)
        if m in ('cuff', 'kimono'):
            for q in g.folds.get(G['hand'], []): cel.fill(q, pal[m + '_light'] if m == 'cuff' else pal['kimono_shade'])
        cel.ctx.restore()
    if g.seams: cel.fill(g.seams, pal['shade'], evenodd=True)


def _folds(g):
    """angular light shapes along the folds of a sleeve cuff"""
    g.folds = {}
    sc = g.sc
    for idx in np.nonzero(sc.T == 2)[0]:
        hi, pr = sc.meta[idx]; r = np.random.default_rng(11 + hi); L = sc.L[idx]; out = []; lines = []
        l = sc.R[idx] @ g.L; ph0 = math.atan2(l[0], l[2])
        for k, (da, y0, ln, sk, wd) in enumerate(((-.75, .10, .62, .95, .16), (.25, .34, .70, -.80, .13), (.75, .02, .34, -.45, .10), (-1.15, .62, .60, .75, .14), (.0, .95, .7, .8, .15))):
            a = ph0 + da; y = np.array([y0, y0 + ln * .40, y0 + ln, y0 + ln * .62, y0]); ph = a + np.array([0, sk * .30 - wd * .6, sk, sk * .70 + wd * .55, 0])
            P = []
            for j in range(4):
                t = np.linspace(0, 1, 10); bow = np.sin(t * math.pi) * .04 * (-1) ** j; P.append(_ring(sc, idx, y[j] + (y[j + 1] - y[j]) * t, ph[j] + (ph[j + 1] - ph[j]) * t + bow))
            q = np.concatenate(P)
            Pw = q @ sc.R[idx] + sc.O[idx]; c = sc.O[idx] + (np.array([0, 1, 0]) * q[:, 1:2]) @ sc.R[idx]
            if (((Pw - c) * (np.array([0, 0, g.cam.dist]) - Pw)).sum(1) > 0).mean() < .80: continue
            out.append(g.cam.px(Pw)); m = len(P[0]); lines.append(g.cam.px(Pw[:2 * m]))
        q = _ring(sc, idx, np.full(60, .085), np.linspace(0, 2 * math.pi, 60)); Pw = q @ sc.R[idx] + sc.O[idx]; c = sc.O[idx] + (np.array([0, 1, 0]) * q[:, 1:2]) @ sc.R[idx]
        nearh = ((Pw - c) * (np.array([0, 0, g.cam.dist]) - Pw)).sum(1) > 0
        for x, y_ in _runs(nearh, True):
            if y_ - x > 6: lines.append(g.cam.px(Pw[np.arange(x, y_) % 60]))
        g.folds.setdefault(hi, []); g.folds[hi] += out
        for q in lines: g.creases.append(dict(hand=hi, kind='fold', w=.5, P=q, z=0., gid=hi * 16 + GID['cuff']))


def fill_overlays(cel, g, pal, w):
    byid = {G['id']: G for G in g.groups}
    for t in g.tapes:
        G = byid.get(t['gid'])
        if G is None: continue
        _clipg(cel, G); cel.fill(t['polys'], pal['tape'], evenodd=True)
        if g.shadow.get(t['gid']):
            cel.ctx.save(); cel._path(t['polys']); cel.ctx.set_fill_rule(1); cel.ctx.clip(); cel.fill(g.shadow[t['gid']], pal['tape_shade'], evenodd=True); cel.ctx.restore()
        cel.ctx.restore()
    for b in g.bands:
        G = byid.get(b['gid'])
        if G is None: continue
        _clipg(cel, G); cel.fill(b['polys'], pal['band'], evenodd=True); cel.ctx.restore()
    for n in g.nails:
        G = byid.get(n['gid'])
        if G is None: continue
        _clipg(cel, G)
        c = pal['nail_dark'] if n['dark'] else pal['nail']
        cel.fill(n['polys'], c, evenodd=True)
        if n['dark']:                                                           # one hard glint
            for q in n['polys']:
                if len(q) < 8: continue
                m_ = q.mean(0); k_ = q[np.argmin((q - m_) @ np.array([.6, .8]))]; v = q[np.argmax(np.abs((q - m_) @ np.array([.8, -.6])))] - m_
                cel.line([m_ + (k_ - m_) * .62 - v * .34, m_ + (k_ - m_) * .70, m_ + (k_ - m_) * .62 + v * .34], max(.8, w * .34), pal.get('nail_glint', '#8d7f86'), taper=(.4, .4), seed=3)
        if g.shadow.get(n['gid']) and not n['dark']:
            cel.ctx.save(); cel._path(n['polys']); cel.ctx.set_fill_rule(1); cel.ctx.clip(); cel.fill(g.shadow[n['gid']], pal['nail_shade'], evenodd=True); cel.ctx.restore()
        cel.ctx.restore()


def rim_polys(g, direction, lo=.30, graze=.72, eps=.016):
    """hard kick-light shapes: {group id: polygons}; direction = TO the second light"""
    d = unit(direction); N, gid = g.gb['N'], g.gb['gid']; M = gid > 0
    r = int(.016 * g.u * 2.2) + 2; tol = max(.010, 2.2 * g.gb['px'])
    a = _geo((N @ d).astype(np.float32), g.gb['z'], M, r, max(1., .012 * g.u), tol); z = _geo(N[..., 2].copy(), g.gb['z'], M, r, max(1., .012 * g.u), tol)
    s_ = (a > lo) & (z < graze) & M
    return {G['id']: _poly_in(g, gid == G['id'], s_, eps * g.u) for G in g.groups}


def depth_polys(g, light, thr=-.3, eps=.022):
    """a second, deeper shadow shape: {group id: polygons} where the form turns further from the light than thr"""
    L = unit(light); gid = g.gb['gid']; M = gid > 0; sg = .022 * g.u; tol = max(.010, 2.2 * g.gb['px']); r = int(sg * 2.2) + 2
    v = _geo(_geo((g.gb['N'] @ L).astype(np.float32), g.gb['z'], M, r, sg, tol), g.gb['z'], M, r, sg, tol)
    return {G['id']: _poly_in(g, gid == G['id'], (v < thr) & M, eps * g.u) for G in g.groups}


def big(polys, s, amin=.015):
    """drop the small pieces of a light-shape dict (area in square hand units; s = px per unit)"""
    ar = lambda q: abs(np.dot(q[:, 0], np.roll(q[:, 1], 1)) - np.dot(q[:, 1], np.roll(q[:, 0], 1))) / 2
    return {k: [q for q in v if ar(q) > amin * s * s] for k, v in polys.items()}


def fill_polys(cel, g, polys, col, alpha=1.0, names=None):
    """fill {group id: polygons} clipped to each visible region"""
    for G in g.groups:
        P = polys.get(G['id'])
        if not P or not G['polys'] or (names and G['name'] not in names): continue
        _clipg(cel, G); cel.fill(P, col, alpha, evenodd=True); cel.ctx.restore()


def _axis2(g, G):
    """screen direction of a group's bone"""
    if hasattr(g, 'axis'): return g.axis[G['id']]
    h = g.hands[G['hand']]; n = G['name']
    if n in FING or n == 'thumb':
        F = h.sk.fr[n]; a, b = h.world(F[0].o), h.world(F[2].pt((0, h.sk.fr[n + '_L'][2], 0)))
        a, b = g.cam.px(a), g.cam.px(b)
    else: a, b = g.cam.px(h.world((0, 0, 0))), g.cam.px(h.world((0, 1, 0)))
    return math.atan2(b[1] - a[1], b[0] - a[0])


def hatch(cel, g, polys=None, gap=.030, w=2.2, col='#0c0a0d', seed=0, ang=1.0, alpha=1.0, names=None, length=.16, fade=.0):
    """pen hatching across the form inside shadow polygons (default: the cel shadow).  gap/length in hand units; ang = angle to the bone"""
    from .kit import hatch as KH
    polys = polys or g.shadow; u = g.cam.s
    for j, G in enumerate(g.groups):
        P = polys.get(G['id'])
        if not P or not G['polys'] or (names and G['name'] not in names) or G['name'] in ('cuff', 'cuffin', 'prop', 'steel') or G['name'].startswith('nail'): continue
        a = _axis2(g, G) + ang
        _clipg(cel, G)
        for k, q in enumerate(P): KH(cel, q, a, gap * u, w, col, seed * 31 + j * 7 + k, length=length * u if length else None, fade=fade, alpha=alpha)
        cel.ctx.restore()


def ink_lines(cel, g, pal, w=4.0, seed=0, inner=.60, tail=.055, lw=.30, tremor=.22, marks=1.0, press=.20, col=None):
    """pen lines: contours, overlaps, creases, marks"""
    u = g.cam.s; col = col or pal['line']
    l2 = np.array([g.L[0], -g.L[1]]); l2 = l2 / (np.hypot(*l2) + 1e-6) * min(1., np.hypot(*g.L[:2]) * 1.5)
    byid = {G['id']: G for G in g.groups}
    pf = lambda z: np.clip((g.cam.dist / np.maximum(g.cam.dist - z, .2)) ** .55, .6, 2.0)
    for j, c in enumerate(g.creases):
        if c['kind'] == 'fold':
            G = byid.get(c['gid'])
            if G is None or not G['polys']: continue
            _clipg(cel, G); cel.line(c['P'], w * inner * .7, col, taper=(.25, .4), press=press, seed=seed * 13 + j, tremor=tremor); cel.ctx.restore(); continue
        cel.line(c['P'], w * inner * c['w'] * 1.55 * marks * pf(c['z']), col, taper=(.30, .55), press=press, seed=seed * 17 + j, tremor=tremor, tmin=.15)
    for t in g.tapes:
        for j, q in enumerate(t['lines']): cel.line(q, w * inner * .75, col, taper=(.25, .25), press=press, seed=seed * 19 + j, tremor=tremor)
        for j, q in enumerate(t['polys']):
            G = byid.get(t['gid'])
            if G: _clipg(cel, G); cel.line(q, w * inner, col, taper=(0, 0), press=press, seed=seed * 23 + j, tremor=tremor, closed=True); cel.ctx.restore()
    for n in g.nails:
        G = byid.get(n['gid'])
        if G is None: continue
        _clipg(cel, G)
        for j, q in enumerate(n['polys']):
            q = np.vstack([q, q[:1]]); cel.line(q, w * inner * .62, col, taper=(.2, .2), press=.3, seed=seed * 29 + j, tremor=tremor * .6)
        cel.ctx.restore()
    for j, st in enumerate(sorted(g.strokes, key=lambda s: (s['kind'] == 'out', float(np.mean(s['z']))))):
        P, wt = st['P'], st['wt'].astype(float)
        wt = wt * (1 - lw * (st['nr'] @ l2)) * pf(st['z'])
        base = w * (1.0 if st['kind'] in ('out', 'cross') else inner * 1.0 if st['kind'] == 'touch' else inner * .85)
        if st['inner']: base *= inner * 1.05
        wt = _smooth(np.c_[wt, wt], 15, False)[:, 0] if len(wt) > 20 else wt
        if st['closed']:
            c = P.mean(0); i0 = int(np.argmax(((P - c) ** 2).sum(1)) + len(P) // 2); P = np.roll(P, -i0, 0); wt = np.roll(wt, -i0); P = np.vstack([P, P[:1]]); wt = np.r_[wt, wt[:1]]
        else:
            n = int(tail * u / max(g.k, .5))
            if st['kind'] == 'out' and len(P) > 14:
                if st['next'][0] == NON and st['ends'][0] == OCC: P = _tail(P, n, g.k, False); wt = np.r_[np.full(len(P) - len(wt), wt[0]), wt]
                if st['next'][1] == NON and st['ends'][1] == OCC: P = _tail(P, n, g.k, True); wt = np.r_[wt, np.full(len(P) - len(wt), wt[-1])]
        cs = [] if st['kind'] != 'out' else _corners(P, max(3, .030 * g.u))
        cut = [0] + cs + [len(P) - 1]
        for q, (x, y) in enumerate(zip(cut[:-1], cut[1:])):
            Q, ww = P[x:y + 1], wt[x:y + 1]
            if len(Q) < 4: continue
            Lp = len(Q) * g.k; tp = []
            for e in (0, 1):
                if (e == 0 and q > 0) or (e == 1 and y < len(P) - 1): tp.append(min(.30, .07 * u / Lp))           # lift at the corner
                elif st['next'][e] in (HID,) or st['closed']: tp.append(min(.10, 4 / max(Lp, 1)))             # butts against a nearer form
                else: tp.append(min(.45, (.16 if st['kind'] == 'out' else .10) * u / Lp))
            t0 = np.linspace(0, 1, len(ww))
            cel.line(Q, base, col, taper=tuple(tp), press=press, seed=seed * 131 + j * 7 + q, tremor=tremor, wfun=lambda t, a=t0, b=ww: np.interp(t, a, b), tmin=.30)
    return cel


def feather(cel, g, gap=.030, length=.15, w=2.4, col='#0c0a0d', seed=0, ang=.75, side=-.10, density=1.0):
    """feathered hatching: strokes that start on the shadow-side contours and flick inward, in clusters (gap/length in hand units)"""
    r = np.random.default_rng(seed + 5); u = g.cam.s; byid = {G['id']: G for G in g.groups}
    l2 = np.array([g.L[0], -g.L[1]]); l2 = l2 / (np.hypot(*l2) + 1e-6)
    from .ink import noise1
    for j, st in enumerate(g.strokes):
        G = byid.get(st['gid']); P = st['P']; n = len(P)
        if st['kind'] != 'out' or G is None or not G['polys'] or n * g.k < .22 * u: continue
        T = np.gradient(P, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9
        cl = noise1(n, max(8, .30 * u / g.k), r); sh = -(st['nr'] @ l2)                       # > 0 on the shadow side
        pf = np.clip((g.cam.dist / np.maximum(g.cam.dist - st['z'], .2)), .6, 2.2)
        sg = 1 if r.random() < .5 else -1
        if (T @ np.array([0, 1.])).mean() < 0: sg = -sg
        _clipg(cel, G); i = float(r.uniform(0, 4))
        while i < n - 1:
            a = int(i); i += gap * u / g.k * r.uniform(.6, 1.4) * pf[a] ** .5
            amt = np.clip((sh[a] - side) * 1.6, 0, 1) * np.clip(.65 + .7 * cl[a], 0, 1) * density
            if r.random() > amt * 1.4: continue
            L = length * u * (.35 + .8 * amt) * r.uniform(.6, 1.15) * pf[a] * (.42 if G['name'] != 'palm' else 1.)
            d = -st['nr'][a]; th = ang * sg + r.normal(0, .10); c, s_ = math.cos(th), math.sin(th); d1 = np.array([d[0] * c - d[1] * s_, d[0] * s_ + d[1] * c]); d2 = d1 * .85 + T[a] * sg * .25
            cel.line([P[a] - d1 * 2, P[a] + d1 * L * .5, P[a] + d1 * L * .5 + d2 * L * .5], w * r.uniform(.7, 1.25) * (.6 + .6 * amt), col, taper=(.04, .85), press=.35, seed=int(r.integers(1 << 30)), tremor=.5, tmin=.08)
        cel.ctx.restore()
    return cel


def brush_lines(cel, g, w=8.0, col='#0c0a0d', seed=0, inner=.55, dry=.7, hairs=2, tremor=.9, press=.45, lw=.45, marks=1.0):
    """brush-ink contours: heavy swelling strokes with dry companion hairs (the ink passages)"""
    pal = dict(line=col); ink_lines(cel, g, pal, w, seed, inner=inner, tail=.09, lw=lw, tremor=tremor, marks=marks, press=press, col=col)
    r = np.random.default_rng(seed + 77); u = g.cam.s
    for j, st in enumerate(g.strokes):
        P = st['P']; n = len(P)
        if st['kind'] != 'out' or n * g.k < .16 * u: continue
        for q in range(hairs + int(n * g.k > .6 * u)):
            if r.random() > dry: continue
            a = r.uniform(0, .6); b = min(1., a + r.uniform(.18, .5)); i, e = int(a * (n - 1)), int(b * (n - 1))
            if e - i < 8: continue
            sg = 1 if r.random() < .72 else -1
            off = -sg * w * r.uniform(.55, 1.35) * np.sin(np.linspace(0, math.pi, e - i))[:, None] ** .5
            cel.line(_smooth(P[i:e] + st['nr'][i:e] * off, 9, False), w * r.uniform(.10, .26), col, taper=(.35, .6), press=.5, seed=seed * 7 + j * 5 + q, tremor=tremor)
    return cel


class _Lite: pass


def lite(g, rims=None, deeps=None, extra=None):
    """picklable drawing for caches: everything the inking needs, no buffers.  rims/deeps = {name: polys} from rim_polys/depth_polys; extra = your dict"""
    q = _Lite()
    for k in ('groups', 'strokes', 'creases', 'shadow', 'nails', 'bands', 'mask', 'core', 'seams', 'seams_by', 'folds', 'pts', 'wrist', 'L', 'k', 'u'): setattr(q, k, getattr(g, k))
    q.tapes = [{k: v for k, v in t.items() if k != 'mask'} for t in g.tapes]
    q.cam = g.cam; q.hands = []
    for h in g.hands:
        z = _Lite(); z.style = {k: v for k, v in h.style.items()}; z.sk = _Lite(); z.sk.fr = {}; q.hands.append(z)
    q.axis = {G['id']: _axis2(g, G) for G in g.groups}
    q.rims = rims or {}; q.deeps = deeps or {}; q.extra = extra or {}
    return q


def only(g, hi):
    """the same drawing restricted to one hand (to slip a sleeve between two hands)"""
    import copy
    q = copy.copy(g); ids = {G['id'] for G in g.groups if G['hand'] == hi}
    q.groups = [G for G in g.groups if G['hand'] == hi]; q.strokes = [s for s in g.strokes if s['hand'] == hi and s['kind'] != 'cross']; q.creases = [c for c in g.creases if c['hand'] == hi]
    q.nails = [n for n in g.nails if n['hand'] == hi]; q.bands = [b for b in g.bands if b['hand'] == hi]; q.tapes = [t for t in g.tapes if t['hand'] == hi]; q.shadow = {k: v for k, v in g.shadow.items() if k in ids}
    q.seams = g.seams_by.get(hi, [])
    if hi > 0: q.core = []
    return q


def cross_lines(cel, g, pal, w, pairs, seed=0, **k):
    """ink the borders between two hands (render(cross=True)) after both have been painted; pairs = [(hand, hand), ...]"""
    import copy
    q = copy.copy(g); ok = {frozenset(p) for p in pairs}; q.strokes = [s for s in g.strokes if s['kind'] == 'cross' and frozenset((s['hand'], s['other'])) in ok]; q.creases = []; q.tapes = []; q.nails = []
    return ink_lines(cel, q, pal, w, seed, **k)


def ink(cel, g, pal=None, w=4.0, seed=0, **k):
    """full cel: flats, one hard shadow, nails/bands/tape, pen lines"""
    pal = dict(PAL['day'], **(PAL[pal] if isinstance(pal, str) else pal or {}))
    fill_flat(cel, g, pal); fill_shadow(cel, g, pal); fill_overlays(cel, g, pal, w)
    return ink_lines(cel, g, pal, w, seed, **k)


def draw(cel, pose, cam, pal=None, light=(-.5, .75, .6), w=4.0, seed=0, side='R', R=None, pos=(0, 0, 0), bld=MALE, pivot=(0, .75, 0), style=None, **k):
    """one-call version: pose + camera -> inked hand on the cel.  returns the Drawn geometry"""
    h = Hand(pose, side, R, pos, bld, pivot=pivot, **(style or {}))
    rk = {q: k.pop(q) for q in ('res', 'clip', 'props', 'thr', 'cast', 'marks') if q in k and q != 'marks'}
    g = render([h], cam, light, **rk); ink(cel, g, pal, w, seed, **k); return g


# ----------------------------------------------------------------- two-hand signs and props
def _pair(poseL, poseR, RL, RR, pL, pR, R, pos, bL, bR, sty, pivot=(0, .75, 0)):
    G = np.eye(3) if R is None else np.asarray(R, float); pos = np.asarray(pos, float)
    return [Hand(poseL, 'L', G @ RL, G @ np.asarray(pL, float) + pos, bL, pivot=pivot, **sty), Hand(poseR, 'R', G @ RR, G @ np.asarray(pR, float) + pos, bR, pivot=pivot, **sty)]


def _w(p, wrist): q = dict(POSES[p] if isinstance(p, str) else p); q['wrist'] = wrist; return q


def two(name, R=None, pos=(0, 0, 0), bld=MALE, gap=0., poses=None, **style):
    """[left Hand, right Hand] of a sign, ready for render().  R = turn of the whole sign, pos = where its centre sits, gap = opening (clap: 1 = wide)
    pray | sukuna | dog | clap | reach (fingertips almost touching; gap in hand units)
    front view = what someone facing the character sees (thumbs toward the character)."""
    if name in ('pray', 'sukuna', 'clap'):
        pn = dict(pray='pray', sukuna='sukuna', clap='slap')[name]; w = dict(pray=-34, sukuna=-34, clap=-38)[name]
        pL, pR = poses or (_w(pn, (w, 0)), _w(pn, (w, 0)))
        tl = dict(pray=3.9, sukuna=4.0, clap=3.4)[name] + 26 * gap; yw = dict(pray=78, sukuna=62, clap=81)[name] - 22 * gap
        off = dict(pray=.178, sukuna=.56, clap=.180)[name] + .62 * gap
        return _pair(pL, pR, Rz(-D(-tl)) @ Ry(D(yw)), Rz(D(-tl)) @ Ry(-D(yw)), (off, 0, 0), (-off, 0, 0), R, pos, bld, bld, style)
    if name == 'dog':
        pL, pR = poses or (_w('dog_near', (-8, 0)), _w('dog_far', (-8, 0)))
        return _pair(pL, pR, euler(0, 0, 90), euler(180, 0, 90), (0, 0, .165 + .4 * gap), (.03, .02, -.165 - .4 * gap), R, pos, bld, bld, style)
    if name == 'reach':
        pL, pR = poses or ('reach', 'reach')
        out = []
        for sd, p, yw, rl, sx in (('R', pL, 60, -78, -1), ('L', pR, -50, 84, 1)):
            h = Hand(p, sd, (np.eye(3) if R is None else R) @ euler(yw, 0, rl), (0, 0, 0), bld, **style)
            F = h.sk.fr['index']; tip = h.world(F[2].pt((0, h.sk.fr['index_L'][2], 0)))
            h.o = h.o - tip + np.asarray(pos, float) + np.array([sx * gap / 2, 0, 0]); out.append(h)
        return out
    raise KeyError(name)


def grip_wrist(target, bld=MALE, pose_='grip', side='L', at=.10, turn=0., slide=0., thumb='elbow', skew=0., lift=0., **style):
    """a second Hand whose fist closes round the forearm of `target` (a Hand); at = distance below the wrist, turn = degrees round the arm (0 = its back rests
    on the back of the target's forearm, 180 = on the palm side), thumb = which way its thumb points along the arm ('elbow' | 'wrist'), skew = degrees the
    gripping hand is turned about its own back (swings its forearm), lift = units it still floats off the arm (for the approach), pose_ = any pose or dict"""
    W = target.sk.fr['wrist']; A = target.A / target.scale
    f = unit(A @ W.R[:, 1]); c = target.world(W.pt((0, at, 0))); n = unit(A @ W.R[:, 2])
    n = n * math.cos(D(turn)) + np.cross(f, n) * math.sin(D(turn))
    sgn = (1 if side == 'L' else -1) * (1 if thumb == 'elbow' else -1)
    x = f * sgn; z = n; y = np.cross(z, x)                                         # columns of the (left-handed-canonical) orientation
    R = np.stack([x, y, z], -1) @ Rz(D(skew if side == 'L' else -skew))
    h = Hand(pose_, side, R, (0, 0, 0), bld, **style)
    h.o = h.o - h.world((0., .74 + slide, -.43 - lift)) + c
    return h


def hammer_props(h, length=2.6, r=.115, up=.55, head=True):
    """rods for a hammer held by Hand h in pose 'hammer' (handle along the fist axis); returns props list"""
    a = np.array((.0, .86, -.345)); d = unit(np.array((1., .10, .03)))
    p0 = a - d * up; p1 = a + d * (length - up)
    P = [rod(h.world(p0), h.world(p1), r, r * .86, .25, ph=0)]
    if head:
        q = a + d * (length - up - .12); n = np.array((0., 1., 0.)); n = unit(n - d * np.dot(n, d))
        P.append(rod(h.world(q - n * .12), h.world(q + n * .30), .155, .15, .5, g='steel', ph=1)); P.append(rod(h.world(q + n * .30), h.world(q + n * .46), .19, .20, .25, g='steel', ph=2))
        P.append(rod(h.world(q - n * .12), h.world(q - n * .50), .15, .03, .6, g='steel', ph=3))
    return P


def nail_props(h, n=3, L=.95, r=.030, slide=None):
    """iron nails held between the knuckles of Hand h (pose 'nails')"""
    out = []
    for k, (f, g_) in enumerate(list(zip(FING[:-1], FING[1:]))[:n]):
        A, B = h.sk.fr[f][0], h.sk.fr[g_][0]; La, Lb = h.sk.fr[f + '_L'][0], h.sk.fr[g_ + '_L'][0]
        m = (A.pt((0, La * .66, 0)) + B.pt((0, Lb * .70, 0))) / 2
        d = unit(np.array(((.16, .02, -.14)[k], 1., .16)))
        sl = 0. if slide is None else slide[k]                                   # slide 1 = still hidden in the fist, 0 = out
        if sl > .96: continue
        a, b = m - d * (.34 + L * sl), m + d * L * (1 - sl)
        out.append(rod(h.world(a), h.world(b), r, r * .9, .5, g='steel', ph=k)); out.append(rod(h.world(b), h.world(b + d * .16), r * .9, .004, .5, g='steel', ph=k))
        out.append(rod(h.world(a - d * .05), h.world(a), r * 2.0, r * 2.0, .2, g='steel', ph=k))
    return out


VIEWS = dict(            # good camera angles to start from: (pose, euler(yaw,pitch,roll), eye distance)
    fist_lens=('fist', (180, 80, 0), 3.4), fist_34=('fist', (148, 58, 0), 4), fist_side=('fist', (90, 0, 90), 8), fist_up=('fist', (180, 0, 0), 8),
    gojo_cross=('cross', (165, -8, 0), 9), gojo_two=('two', (158, -8, 6), 9), point=('point', (150, 20, 0), 7), reach_lens=('thrust', (176, 62, 0), 2.6),
    palm_up=('palm_up', (172, -58, 10), 6), nails=('nails', (178, 58, 0), 5), pinch=('pinch', (84, 0, 0), 9), rest=('rest', (8, 42, 0), 7))
