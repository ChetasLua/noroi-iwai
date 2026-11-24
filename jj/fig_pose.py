"""Skeleton, poses, interpolation and cycles for jj.figure.
Local figure frame: x = character's right, y = forward, z = up, metres.  Side values in poses are written OUTWARD (x out, y fwd, z up) and mirrored.
Pose = dict (all keys optional, defaults in STAND):
  root (right, fwd, up)  pelvis offset, fractions of height H          pel wst cst nck hed  (pitch fwd+, roll right+, yaw left+) degrees
  cL cR (up, fwd) extra clavicle, metres*100        hL hR wrist target from the shoulder, chest frame, arm-length units        eL eR elbow pole
  wL wR (twist, flex, dev) wrist degrees   gL gR hand shape name
  fL fR ankle target (out, fwd, up) from the ground point under root, fractions of H   kL kR knee pole   tL tR (toe-out yaw, toe-up pitch, toe bend)
  sL sR arm stretch (1 = natural)   lead = 'root' puts hands in the root frame instead of the chest frame
"""
import numpy as np, math

D2R = math.pi / 180


def Rx(a):
    c, s = math.cos(a), math.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = math.cos(a), math.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = math.cos(a), math.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler(p):
    """(pitch fwd+, roll right+, yaw left+) degrees -> matrix"""
    return Rz(p[2] * D2R) @ Rx(-p[0] * D2R) @ Ry(p[1] * D2R)


def unit(v):
    v = np.asarray(v, float); n = np.linalg.norm(v); return v / (n + 1e-12)


# ------------------------------------------------------------------ proportions
BODY = dict(H=1.73, heads=7.1, sw=.098, hw=.045, arm=.168, fore=.147, hand=.112, hipz=.525, kneez=.285, ankz=.043,
            waistz=.612, chestz=.695, neckz=.828, shz=.787, headz=.868, mass=1.0, leg=1.0, arml=1.0, limb=1.0, chest=1.0, waist=1.0, neck=1.0, hips=1.0, foot=1.0)
CHAR_BODY = dict(
    yuji=dict(H=1.73, heads=6.95, mass=1.0, sw=.101, limb=1.03, neck=1.08),
    megumi=dict(H=1.75, heads=7.05, mass=.94, sw=.095, limb=.95),
    nobara=dict(H=1.60, heads=6.7, mass=.83, sw=.082, limb=.74, waist=.80, hips=1.0, chest=.86, neck=.72, foot=.88),
    gojo=dict(H=1.91, heads=7.8, mass=.93, sw=.097, limb=.92, hipz=.538, kneez=.294, neck=.98),
    nanami=dict(H=1.84, heads=7.5, mass=.99, sw=.102, limb=.99, neck=1.05),
    todo=dict(H=1.91, heads=7.6, mass=1.16, sw=.122, limb=1.26, chest=1.16, waist=1.0, neck=1.32, hips=1.05),
    sukuna=dict(H=1.76, heads=7.1, mass=1.02, sw=.104, limb=1.05, neck=1.08),
    crowd=dict(H=1.68, heads=6.9, mass=.96, sw=.095, limb=.98),
    crowdf=dict(H=1.58, heads=6.7, mass=.86, sw=.084, limb=.80, waist=.84, chest=.88, neck=.76, foot=.9),
    curse=dict(H=2.35, heads=6.2, mass=1.9, sw=.150, hw=.062, arm=.225, fore=.215, hand=.14, hipz=.40, kneez=.215, waistz=.50, chestz=.60, neckz=.76, shz=.73, headz=.79, limb=1.75, chest=1.35, neck=1.8, foot=1.1),
)

STAND = dict(root=(0, 0, 0), pel=(0, 0, 0), wst=(0, 0, 0), cst=(0, 0, 0), nck=(4, 0, 0), hed=(-2, 0, 0), cL=(0, 0), cR=(0, 0),
             hL=(.13, .08, -.975), hR=(.13, .08, -.975), eL=(.3, -1, -.1), eR=(.3, -1, -.1), wL=(0, 8, 0), wR=(0, 8, 0), gL='relax', gR='relax',
             fL=(.055, 0, .043), fR=(.055, 0, .043), kL=(.12, 1, 0), kR=(.12, 1, 0), tL=(12, 0, 0), tR=(12, 0, 0), sL=1., sR=1., lead='chest')


def P(base=None, **kw):
    """new pose: P(fL=..., pel=...) on top of STAND, or P('crouch', hR=...)"""
    d = dict(STAND if base is None else (POSES[base] if isinstance(base, str) else base)); d.update(kw); return d


def mirror(p):
    q = dict(p)
    for k in list(p):
        if k[-1] in 'LR' and k[:-1] in ('h', 'e', 'w', 'g', 'f', 'k', 't', 's', 'c'):
            q[k[:-1] + ('R' if k[-1] == 'L' else 'L')] = p[k]
    for k in ('pel', 'wst', 'cst', 'nck', 'hed'): a = p.get(k, STAND[k]); q[k] = (a[0], -a[1], -a[2])
    r = p.get('root', (0, 0, 0)); q['root'] = (-r[0], r[1], r[2]); return q


def to_root(p):
    """the same pose with hand targets expressed in the root frame (lead='root')"""
    if p.get('lead', 'chest') == 'root': return p
    g = lambda k: np.asarray(p.get(k, STAND[k]), float); Rc = euler(g('pel')) @ euler(g('wst')) @ euler(g('cst')); q = dict(p); q['lead'] = 'root'
    for s, sg in (('L', -1.), ('R', 1.)):
        m = np.array([sg, 1, 1.])
        for k in ('h', 'e'): q[k + s] = (Rc @ (g(k + s) * m)) * m
    return q


def mix_pose(a, b, t):
    """joint-space in-between.  hands travel on arcs round the shoulder, feet on straight lines, angles linear"""
    if t <= 0: return a
    if t >= 1: return b
    if a.get('lead', 'chest') != b.get('lead', 'chest'): a, b = to_root(a), to_root(b)
    o = {}
    for k, v0 in STAND.items():
        va, vb = a.get(k, v0), b.get(k, v0)
        if isinstance(v0, str): o[k] = va if t < .5 else vb; continue
        x, y = np.asarray(va, float), np.asarray(vb, float)
        if k in ('hL', 'hR'):
            la, lb = np.linalg.norm(x), np.linalg.norm(y); u, w = x / (la + 1e-9), y / (lb + 1e-9)
            om = math.acos(float(np.clip(u @ w, -1, 1)))
            if 1e-3 < om < math.pi - .05: d = (math.sin((1 - t) * om) * u + math.sin(t * om) * w) / math.sin(om)
            else: d = u * (1 - t) + w * t
            o[k] = d * (la * (1 - t) + lb * t)
        elif k in ('eL', 'eR', 'kL', 'kR'): o[k] = unit(unit(x) * (1 - t) + unit(y) * t)
        else: o[k] = x * (1 - t) + y * t
    return o


def add_pose(a, **d):
    """a with numeric offsets added"""
    o = dict(a)
    for k, v in d.items(): o[k] = np.asarray(a.get(k, STAND[k]), float) + np.asarray(v, float)
    return o


# ------------------------------------------------------------------ FK / IK
def two_bone(a, t, pole, l1, l2, stretch=False):
    """-> mid joint, end joint (clamped), e1 = unit direction the mid joint points to"""
    d = t - a; D = np.linalg.norm(d); m = l1 + l2
    if D > m * .998:
        if stretch: k = D / (m * .995); l1 *= k; l2 *= k
        else: d = d / D * m * .998; D = m * .998
    D = max(D, abs(l1 - l2) * 1.02 + 1e-4); d = unit(d)
    x = (l1 * l1 - l2 * l2 + D * D) / (2 * D); h = math.sqrt(max(l1 * l1 - x * x, 0.))
    p = np.asarray(pole, float); p = p - d * (p @ d)
    if np.linalg.norm(p) < 1e-4: p = np.cross(d, [.3, .9, .1])
    p = unit(p)
    return a + d * x + p * h, a + d * D, p


class Skel:
    """joint positions and frames for one pose: attributes pelvis Rp, waist Rw, chest Rc, neckb, Rn, head Rh, sh/el/wr/ha[side], Ra.., hip/kn/an[side], Rf.."""

    def __init__(self, body, pose, x=0., y=0., z=0., heading=0., scale=1.0):
        b = dict(BODY); b.update(body); self.b = b; H = b['H'] * scale; self.H = H; self.sc = scale
        g = lambda k: np.asarray(pose.get(k, STAND[k]), float)
        self.origin = np.array([x, y, z], float); self.Rr = Rz(heading); Rr = self.Rr; self.heading = heading
        r = g('root')
        pel = self.origin + Rr @ np.array([r[0] * H, r[1] * H, (b['hipz'] + .02 + r[2]) * H]); self.pel = pel
        Rp = Rr @ euler(g('pel')); Rw = Rp @ euler(g('wst')); Rc = Rw @ euler(g('cst')); Rn = Rc @ euler(g('nck')); Rh = Rn @ euler(g('hed'))
        self.Rp, self.Rw, self.Rc, self.Rn, self.Rh = Rp, Rw, Rc, Rn, Rh
        z0 = b['hipz'] + .02
        self.waist = pel + Rp @ np.array([0, 0, (b['waistz'] - z0) * H]); self.chest = self.waist + Rw @ np.array([0, .004 * H, (b['chestz'] - b['waistz']) * H])
        self.neck = self.chest + Rc @ np.array([0, -.012 * H, (b['neckz'] - b['chestz']) * H]); self.headb = self.neck + Rn @ np.array([0, .010 * H, (b['headz'] - b['neckz']) * H])
        hh = H / b['heads']; self.hh = hh; self.u = hh / 2                                  # u = Head unit (chin to eye line)
        self.head = self.headb + Rh @ np.array([0, .012 * H, (1 - hh / H / 2 - b['headz']) * H])
        self.sh, self.el, self.wr, self.hip, self.kn, self.an, self.e1a, self.e1l, self.Rhand, self.Rfoot, self.grip, self.clav = ({} for _ in range(12))
        self.bend = {}; self.lift = {}; self.kbend = {}
        for s, sg in (('L', -1.), ('R', 1.)):
            out = lambda v: np.array([sg * v[0], v[1], v[2]])
            la, lf = b['arm'] * H * b['arml'], b['fore'] * H * b['arml']; st = float(pose.get('s' + s, 1.)); A = la + lf
            tgt = out(g('h' + s)) * A; Rs = Rc if pose.get('lead', 'chest') == 'chest' else Rr
            base = self.chest + Rc @ np.array([sg * b['sw'] * H, -.006 * H, (b['shz'] - b['chestz']) * H])
            # shoulder girdle follows the arm: rises and closes in when the hand goes up, slides forward when it reaches across / ahead
            dirc = unit(Rc.T @ (Rs @ tgt)); up = max(0., dirc[2] + .25); fw = dirc[1]; ac = -dirc[0] * sg
            c = g('c' + s)
            base = base + Rc @ np.array([-sg * (.020 * up + .016 * max(0, ac) + .010 * max(0, fw)) * H, (.026 * max(fw, -.5) + .012 * max(0, ac)) * H + c[1] * .01, (.036 * up - .006 * max(0, -dirc[2] - .8)) * H + c[0] * .01])
            self.sh[s] = base; self.clav[s] = self.chest + Rc @ np.array([sg * .02 * H, .02 * H, (b['neckz'] - b['chestz'] - .02) * H])
            pole = Rs @ out(g('e' + s))
            el, wr, e1 = two_bone(base, base + Rs @ tgt * st, pole, la * st, lf * st)
            self.el[s], self.wr[s], self.e1a[s] = el, wr, e1
            self.bend[s] = math.acos(float(np.clip(unit(el - base) @ unit(wr - el), -1, 1))); self.lift[s] = up
            # hand frame: columns = (thumb side, finger direction, back of hand)
            fd = unit(wr - el); tw, fl, dv = g('w' + s) * D2R
            # neutral: thumb forward-ish = flexor side turned a quarter; back of hand outward
            e = unit(e1 - fd * (e1 @ fd)); sidev = np.cross(e, fd) * sg              # sidev: outward (lateral) side of the forearm
            back = unit(sidev * math.cos(tw) + e * math.sin(tw)); thumb = np.cross(back, fd) * sg
            # flex: fingers bend toward palm (-back);  dev: toward thumb
            fd2 = unit(fd * math.cos(fl) - back * math.sin(fl)); back2 = unit(back * math.cos(fl) + fd * math.sin(fl))
            fd3 = unit(fd2 * math.cos(dv) + thumb * math.sin(dv)); th3 = unit(thumb * math.cos(dv) - fd2 * math.sin(dv))
            self.Rhand[s] = np.stack([th3, fd3, back2], 1); self.grip[s] = pose.get('g' + s, 'relax')
            # legs
            lt, ls = (b['hipz'] - b['kneez']) * H * b['leg'], (b['kneez'] - b['ankz']) * H * b['leg']
            hip = pel + Rp @ np.array([sg * b['hw'] * H, 0, -.02 * H]); self.hip[s] = hip
            ft = self.origin + Rr @ (out(g('f' + s)) * H + np.array([0, 0, 0]))
            kn, an, k1 = two_bone(hip, ft, Rr @ out(g('k' + s)), lt, ls)
            self.kn[s], self.an[s], self.e1l[s] = kn, an, k1
            self.kbend[s] = math.acos(float(np.clip(unit(kn - hip) @ unit(an - kn), -1, 1)))
            t = g('t' + s); Rt = Rr @ Rz(-sg * t[0] * D2R) @ Rx(t[1] * D2R)
            self.Rfoot[s] = Rt; self.lift['t' + s] = t[2]

    def pt(self, frame, v):
        o, R = dict(pelvis=(self.pel, self.Rp), waist=(self.waist, self.Rw), chest=(self.chest, self.Rc), neck=(self.neck, self.Rn), head=(self.head, self.Rh), root=(self.origin, self.Rr))[frame]
        return o + R @ np.asarray(v, float)


# ------------------------------------------------------------------ key poses (figure frame)
POSES = {}
_ = P
POSES['stand'] = P()
POSES['contra'] = P(root=(.018, 0, -.004), pel=(2, 5, -6), wst=(0, -5, 4), cst=(-2, -4, 8), nck=(4, 2, -4), hed=(0, 3, -6), hL=(.16, .10, -.96), hR=(.18, .02, -.96), fL=(.075, .03, .043), fR=(.055, -.015, .043), tL=(22, 0, 0), tR=(10, 0, 0), kL=(.25, 1, 0))
POSES['pockets'] = P(root=(-.012, 0, -.004), pel=(-3, -5, 5), wst=(6, 4, -3), cst=(6, 3, -5), nck=(8, -2, 4), hed=(2, -3, 4), cL=(1.5, 1), cR=(1.0, 1), hL=(.15, .10, -.80), hR=(.14, .12, -.80), eL=(1, -.6, 0), eR=(1, -.6, 0), wL=(-20, 10, 0), wR=(-20, 10, 0), fL=(.05, -.01, .043), fR=(.085, .035, .043), tR=(24, 0, 0), kR=(.3, 1, 0))
POSES['headhands'] = P(root=(.012, 0, -.004), pel=(-4, 4, 4), wst=(-4, -3, 0), cst=(-5, -2, -2), nck=(2, 0, 0), hed=(-8, 5, 6), hL=(.18, -.18, .46), hR=(.18, -.18, .46), eL=(1, .25, .2), eR=(1, .25, .2), gL='soft', gR='soft', wL=(40, -10, 0), wR=(40, -10, 0), fL=(.055, .05, .043), fR=(.06, -.03, .043))
# fighting guard, bladed, weight low
POSES['guard'] = P(root=(0, -.01, -.085), pel=(10, 0, -32), wst=(8, 0, -6), cst=(10, 0, -4), nck=(6, 0, 18), hed=(-6, 0, 18), cL=(2, 3), cR=(2, 2), hL=(.02, .64, .10), hR=(.10, .34, .04), eL=(.75, -.1, -1), eR=(1, -.2, -.8), gL='fist', gR='fist', wL=(-10, -6, 0), wR=(10, -6, 0), fL=(.085, .20, .043), fR=(.13, -.17, .052), kL=(.35, 1, 0), kR=(.75, .7, 0), tL=(8, 0, 0), tR=(50, -8, 0))
# punch: wind (right hand back, whole spine coiled), strike (right arm through, hips over), follow
POSES['punch_wind'] = P(lead='root', root=(.02, -.085, -.125), pel=(8, 10, -58), wst=(8, 8, -20), cst=(6, 10, -24), nck=(4, -4, 48), hed=(-4, -4, 34), cR=(3, -3), cL=(2, 3), hL=(-.10, .72, .10), hR=(.10, -.66, .12), eL=(.9, -.2, -.5), eR=(1, -.5, .9), gL='flat', gR='fist', wL=(-30, 0, 0), wR=(-40, -18, 0), fL=(.06, .24, .043), fR=(.15, -.20, .050), kL=(.1, 1, 0), kR=(.9, .5, 0), tL=(0, 6, 0), tR=(68, 0, 0))
POSES['punch_hit'] = P(lead='root', root=(-.03, .13, -.150), pel=(20, -6, 38), wst=(10, -6, 18), cst=(12, -8, 20), nck=(-6, 4, -38), hed=(-8, 4, -34), cR=(4, 7), cL=(2, -2), hR=(-.06, 1.0, .06), hL=(.34, -.42, -.10), eR=(1, 0, -.3), eL=(.5, -1, .2), gR='fist', gL='fist', wR=(-80, -4, 0), fL=(.11, .30, .043), fR=(.06, -.34, .075), kL=(.15, 1, 0), kR=(-.1, 1, -.3), tL=(6, 0, 0), tR=(26, -38, 38))
POSES['punch_follow'] = P(lead='root', root=(-.05, .18, -.185), pel=(26, -14, 58), wst=(14, -8, 24), cst=(16, -10, 26), nck=(-6, 6, -46), hed=(-6, 6, -38), cR=(3, 8), hR=(-.66, .66, -.16), hL=(.50, -.62, .10), eR=(.9, .5, .2), eL=(.4, -1, .5), gR='fist', gL='fist', wR=(-70, -6, 0), fL=(.12, .30, .043), fR=(.05, -.36, .080), kL=(.2, 1, 0), kR=(0, 1, -.3), tL=(6, 0, 0), tR=(30, -44, 42))
POSES['uppercut'] = P(lead='root', root=(-.01, .07, .01), pel=(-6, -8, 30), wst=(-8, -8, 12), cst=(-10, -10, 14), nck=(-4, 4, -24), hed=(-14, 4, -16), cR=(8, 3), hR=(.0, .52, .62), hL=(.40, -.40, -.42), eR=(.8, .6, -1), eL=(.5, -1, -.1), gR='fist', gL='fist', wR=(60, -10, 0), fL=(.06, .10, .060), fR=(.10, -.20, .090), kL=(.1, 1, 0), kR=(.2, 1, 0), tL=(4, -14, 16), tR=(16, -52, 46))
# high roundhouse with the right leg, weight on the left, trunk leaning away
POSES['kick_wind'] = P(root=(-.03, 0, -.03), pel=(-4, -10, 20), wst=(6, -4, 6), cst=(8, -4, 8), nck=(0, 2, -14), hed=(-2, 2, -12), hL=(.10, .50, .14), hR=(.30, .28, -.10), eL=(.8, 0, -1), eR=(1, -.3, -.6), gL='fist', gR='fist', fL=(.02, .02, .043), fR=(.22, .08, .30), kL=(.4, 1, 0), kR=(.8, 1, .4), tL=(40, 0, 0), tR=(20, -50, 0))
POSES['kick'] = P(root=(-.055, -.035, -.010), pel=(-18, -46, 62), wst=(4, 6, -10), cst=(10, 12, -22), nck=(4, 16, -18), hed=(0, 12, -16), cL=(2, 2), cR=(1, -3), hL=(.05, .48, .10), hR=(.72, -.42, -.36), eL=(.6, -.2, -1), eR=(.4, -1, .2), gL='fist', gR='fist', fL=(.02, -.02, .043), fR=(.06, .50, .685), kL=(.55, 1, 0), kR=(.9, .2, .3), tL=(70, 0, 0), tR=(-60, -64, -10))
POSES['leap'] = P(root=(0, 0, .20), pel=(16, -6, 18), wst=(-6, -4, 8), cst=(-14, -6, 18), nck=(-10, 2, -20), hed=(-12, 2, -16), cR=(6, -4), cL=(2, 4), hR=(.34, -.46, .52), hL=(.30, .72, .12), eR=(1, -.5, .4), eL=(.6, -.2, -1), gR='fist', gL='claw', wR=(10, -16, 0), fL=(.07, .20, .40), fR=(.06, -.36, .30), kL=(.2, 1, .25), kR=(.12, 1, -.5), tL=(4, 10, 0), tR=(4, -70, -20))
POSES['land'] = P(root=(.015, -.02, -.300), pel=(34, 4, -14), wst=(20, 2, -6), cst=(16, 2, -8), nck=(-24, 0, 14), hed=(-26, 0, 14), cL=(2, 5), cR=(3, -2), hL=(.10, .74, -.60), hR=(.72, -.34, .26), eL=(.8, -.1, .6), eR=(.5, -1, .2), gL='claw', gR='fist', wL=(-60, -30, 0), fL=(.10, .13, .043), fR=(.22, -.16, .062), kL=(.5, 1, 0), kR=(.9, .8, 0), tL=(16, 0, 0), tR=(50, -18, 22))
POSES['slide'] = P(root=(.03, -.10, -.330), pel=(-12, 12, 30), wst=(18, -4, -8), cst=(16, -6, -14), nck=(-4, 0, -6), hed=(-4, 0, -6), cL=(2, 2), hL=(.66, .48, -.08), hR=(.56, -.66, -.40), eL=(.4, -.6, -1), eR=(.5, -1, .4), gL='splay', gR='claw', wR=(-30, -40, 0), fL=(.12, .36, .043), fR=(.05, .06, .060), kL=(.2, 1, 0), kR=(1, .4, .1), tL=(6, 14, 0), tR=(64, -20, 10))
POSES['kneel'] = P(root=(0, -.04, -.265), pel=(8, 0, 0), wst=(16, 0, 0), cst=(20, 0, 2), nck=(26, 0, 0), hed=(18, 0, 0), hL=(.16, .20, -.84), hR=(.10, .36, -.72), eL=(.6, -1, 0), eR=(.8, -.8, 0), gL='relax', gR='fist', fL=(.085, .18, .043), fR=(.065, -.235, .090), kL=(.25, 1, 0), kR=(.10, 1, -.6), tL=(8, 0, 0), tR=(0, -72, 44))
POSES['knees'] = P(root=(0, -.05, -.380), pel=(2, 0, 0), wst=(20, 0, 0), cst=(24, 3, 2), nck=(30, 2, 0), hed=(18, 3, 0), cL=(-1, 2), cR=(-1, 2), hL=(.18, .20, -.93), hR=(.16, .14, -.96), eL=(.6, -1, 0), eR=(.6, -1, 0), fL=(.11, -.155, .050), fR=(.10, -.17, .050), kL=(.45, 1, -.1), kR=(.35, 1, -.1), tL=(4, -84, 6), tR=(4, -84, 6))
POSES['sit_rail'] = P(root=(0, 0, -.085), pel=(-14, 0, 0), wst=(14, 0, 0), cst=(16, 0, 0), nck=(10, 0, 0), hed=(0, 0, 0), cL=(2, 0), cR=(2, 0), hL=(.24, -.10, -.82), hR=(.24, -.10, -.82), eL=(.7, -1, .1), eR=(.7, -1, .1), gL='hold', gR='hold', wL=(-50, 40, 0), wR=(-50, 40, 0), fL=(.07, .22, .20), fR=(.07, .19, .19), kL=(.2, 1, .2), kR=(.2, 1, .2), tL=(6, -18, 0), tR=(6, -18, 0))
POSES['stagger'] = P(root=(.03, -.04, -.075), pel=(-12, 8, 14), wst=(14, 6, 10), cst=(24, 8, 14), nck=(24, 6, -4), hed=(14, 8, -6), cL=(-1, 3), cR=(2, 1), hL=(.22, .34, -.82), hR=(-.12, .34, -.46), eL=(.5, -1, 0), eR=(1, -.2, -.2), gL='relax', gR='claw', fL=(.12, .10, .043), fR=(.075, -.17, .060), kL=(-.2, 1, 0), kR=(.4, 1, 0), tL=(-8, 0, 0), tR=(34, -20, 24))
POSES['thrown'] = P(root=(0, -.06, .06), pel=(-34, -10, -8), wst=(20, -6, -8), cst=(26, -8, -10), nck=(22, -10, -4), hed=(16, -12, -4), hL=(.26, .86, .10), hR=(.44, .74, .22), eL=(.8, 0, -.6), eR=(.8, 0, -.4), gL='splay', gR='claw', fL=(.12, .40, .20), fR=(.06, .30, .08), kL=(.4, 1, .6), kR=(.3, 1, .4), tL=(10, -30, 0), tR=(10, -40, 0))
POSES['cast'] = P(root=(0, .0, -.02), pel=(0, 0, 8), wst=(2, 0, 4), cst=(-2, 0, 4), nck=(4, 0, -6), hed=(2, 0, -6), hR=(.36, .34, .34), hL=(.18, .06, -.94), eR=(.7, .1, -1), gR='two', wR=(150, 0, 0), fL=(.09, .02, .043), fR=(.085, -.03, .043))
POSES['float'] = P(root=(0, 0, .0), pel=(8, 0, 0), wst=(-2, 0, 0), cst=(-8, 0, 0), nck=(4, 0, 0), hed=(6, 0, 0), hL=(.64, .20, -.42), hR=(.64, .20, -.42), eL=(.6, -1, 0), eR=(.6, -1, 0), gL='splay', gR='splay', fL=(.05, -.02, .03), fR=(.05, -.05, .05), kL=(.2, 1, 0), kR=(.2, 1, 0), tL=(8, -46, -10), tR=(8, -50, -10))
POSES['hammer_up'] = P(root=(0, .0, -.055), pel=(6, -6, -26), wst=(-4, -6, -12), cst=(-8, -8, -22), nck=(0, 4, 34), hed=(-4, 6, 22), cR=(6, -2), hR=(.34, -.22, .72), hL=(.16, .70, -.06), eR=(1, .2, .3), eL=(.7, 0, -1), gR='hold', gL='splay', wR=(10, 0, 20), fL=(.10, .16, .043), fR=(.16, -.14, .050), kL=(.3, 1, 0), kR=(.8, .7, 0), tL=(8, 0, 0), tR=(46, -8, 0))
POSES['clap'] = P(root=(0, 0, -.045), pel=(4, 0, 0), wst=(-4, 0, 0), cst=(-8, 0, 0), nck=(0, 0, 0), hed=(-8, 0, 0), cL=(2, 4), cR=(2, 4), hL=(-.30, .62, .02), hR=(-.30, .62, .02), eL=(1, -.1, -.1), eR=(1, -.1, -.1), gL='flat', gR='flat', wL=(-80, 0, 0), wR=(-80, 0, 0), fL=(.14, .02, .043), fR=(.14, -.02, .043), kL=(.5, 1, 0), kR=(.5, 1, 0), tL=(26, 0, 0), tR=(26, 0, 0))
POSES['hulk'] = P(root=(0, -.02, -.06), pel=(24, 0, 0), wst=(16, 0, 0), cst=(14, 0, 0), nck=(-10, 0, 0), hed=(-10, 0, 0), cL=(2, 4), cR=(2, 4), hL=(.34, .40, -.80), hR=(.34, .40, -.80), eL=(1, -.4, .2), eR=(1, -.4, .2), gL='claw', gR='claw', fL=(.12, .03, .043), fR=(.12, -.05, .043), kL=(.5, 1, 0), kR=(.5, 1, 0), tL=(24, 0, 0), tR=(24, 0, 0))
# sprint keys: right-foot contact, down, push, flight; the left half is the mirror
_SP = dict(nck=(-10, 0, 0), hed=(-10, 0, 0), gL='fist', gR='fist')
POSES['run_contact'] = P(root=(0, 0, -.040), pel=(24, -2, -14), wst=(6, 2, 10), cst=(4, 2, 18), nck=(-12, 0, -8), hed=(-12, 0, -6), cL=(1, 5), cR=(3, -4), hL=(-.02, .62, -.02), hR=(.28, -.60, -.18), eL=(.5, -.3, -1), eR=(.35, -1, .45), gL='fist', gR='fist', fR=(.035, .30, .050), fL=(.05, -.38, .30), kR=(.05, 1, 0), kL=(.1, 1, -.6), tR=(4, 14, 0), tL=(4, -86, 0))
POSES['run_down'] = P(root=(0, 0, -.072), pel=(28, -4, -6), wst=(8, 3, 6), cst=(6, 3, 8), nck=(-14, 0, -4), hed=(-12, 0, -4), cL=(1, 2), cR=(2, -1), hL=(.06, .38, -.40), hR=(.24, -.30, -.50), eL=(.5, -.7, -.7), eR=(.4, -1, .1), gL='fist', gR='fist', fR=(.03, .12, .043), fL=(.05, -.24, .22), kR=(.05, 1, 0), kL=(.1, 1, -.2), tR=(4, 0, 0), tL=(4, -76, 0))
POSES['run_push'] = P(root=(0, 0, .000), pel=(26, 3, 16), wst=(4, -2, -12), cst=(0, -2, -20), nck=(-10, 0, 8), hed=(-12, 0, 8), cL=(3, -4), cR=(1, 5), hL=(.28, -.56, -.16), hR=(-.02, .64, .04), eL=(.35, -1, .4), eR=(.5, -.3, -1), gL='fist', gR='fist', fR=(.03, -.36, .105), fL=(.045, .26, .33), kR=(.08, 1, -.2), kL=(.1, 1, .35), tR=(6, -56, 46), tL=(4, 6, 0))
POSES['run_fly'] = P(root=(0, 0, .030), pel=(22, 1, 18), wst=(4, 0, -12), cst=(0, 0, -22), nck=(-10, 0, 8), hed=(-10, 0, 8), cL=(3, -5), cR=(1, 6), hL=(.32, -.62, -.06), hR=(-.04, .66, .12), eL=(.35, -1, .5), eR=(.5, -.2, -1), gL='fist', gR='fist', fR=(.05, -.46, .26), fL=(.045, .34, .22), kR=(.1, 1, -.5), kL=(.1, 1, .3), tR=(4, -78, 0), tL=(4, 10, 0))
# walk keys (right contact, passing)
POSES['walk_contact'] = P(root=(0, 0, -.012), pel=(2, -1, -7), wst=(0, 1, 4), cst=(-2, 1, 8), nck=(4, 0, -3), hed=(0, 0, -2), hL=(.12, .34, -.91), hR=(.16, -.22, -.95), fR=(.035, .155, .050), fL=(.035, -.165, .075), kR=(.08, 1, 0), kL=(.08, 1, 0), tR=(8, 16, 0), tL=(8, -34, 30))
POSES['walk_pass'] = P(root=(.010, 0, .008), pel=(2, 3, 0), wst=(0, -2, 0), cst=(-2, -2, 0), nck=(4, 0, 0), hed=(0, 0, 0), hL=(.14, .08, -.97), hR=(.14, .04, -.97), fR=(.025, .0, .043), fL=(.035, .035, .105), kR=(.08, 1, 0), kL=(.10, 1, .1), tR=(8, 0, 0), tL=(8, -28, 0))


def cycle(keys, u):
    """keys = list of poses round one full cycle (loops); u in cycles"""
    n = len(keys); x = (u % 1.) * n; i = int(x); return mix_pose(keys[i % n], keys[(i + 1) % n], x - i)


def walk(u, bounce=1.0):
    """walk cycle, u in strides (1 = two steps).  u=0 right heel contact"""
    a, b = POSES['walk_contact'], POSES['walk_pass']; return cycle([a, b, mirror(a), mirror(b)], u)


RUN_KEYS = ['run_contact', 'run_down', 'run_push', 'run_fly']


def sprint(u):
    """sprint cycle, u in strides (1 = two steps, 8 drawn keys).  u=0 right contact"""
    k = [POSES[n] for n in RUN_KEYS]; return cycle(k + [mirror(p) for p in k], u)


def seq(f, keys):
    """held / eased key poses: keys = [(frame, pose or name, ease)] ease in 'hold','lin','in','out','io','snap' describes how we LEAVE the key"""
    g = lambda p: POSES[p] if isinstance(p, str) else p
    if f <= keys[0][0]: return g(keys[0][1])
    for (a, pa, *ea), (b, pb, *_) in zip(keys[:-1], keys[1:]):
        if f < b:
            e = ea[0] if ea else 'io'; t = (f - a) / float(b - a)
            if e == 'hold': return g(pa)
            t = dict(lin=t, io=t * t * (3 - 2 * t), out=1 - (1 - t) ** 2.4, snap=1 - (1 - t) ** 4.)[e] if e != 'in' else t ** 2.4
            return mix_pose(g(pa), g(pb), t)
    return g(keys[-1][1])
# airborne hook coming down on somebody; and the man who takes it on his forearm, sinking back
POSES['dive_punch'] = P(lead='root', root=(0, .06, .22), pel=(40, -10, 34), wst=(10, -6, 18), cst=(8, -8, 22), nck=(-18, 4, -38), hed=(-14, 4, -30), cR=(5, 8), cL=(3, -2), hR=(-.04, .95, -.26), hL=(.50, -.62, .42), eR=(1, 0, .5), eL=(.6, -.6, 1), gR='fist', gL='fist', wR=(-80, -4, 0), fL=(.10, .26, .46), fR=(.05, -.46, .40), kL=(.5, 1, .5), kR=(.2, 1, -.6), tL=(10, -24, 0), tR=(6, -80, -12))
POSES['block_back'] = P(lead='root', root=(.02, -.13, -.165), pel=(-10, 8, -30), wst=(8, 8, -14), cst=(10, 8, -16), nck=(2, -4, 34), hed=(4, -6, 24), cL=(8, 5), cR=(2, -3), hL=(-.02, .56, .40), hR=(.52, -.42, -.46), eL=(.8, .3, -1), eR=(.5, -1, .3), gL='fist', gR='claw', wL=(-60, -8, 0), wR=(-20, -20, 0), fL=(.06, .26, .050), fR=(.16, -.26, .050), kL=(.2, 1, .1), kR=(1, .55, 0), tL=(0, 12, 0), tR=(66, 0, 0))
POSES['duck'] = P(root=(0, .02, -.235), pel=(38, 6, -20), wst=(18, 4, -8), cst=(16, 4, -8), nck=(-20, 0, 14), hed=(-22, 0, 12), cL=(3, 4), cR=(3, 2), hL=(.08, .52, .16), hR=(.20, .20, -.24), eL=(.8, 0, -1), eR=(1, -.6, -.4), gL='fist', gR='fist', fL=(.10, .20, .043), fR=(.14, -.22, .056), kL=(.4, 1, 0), kR=(.8, .7, 0), tL=(10, 0, 0), tR=(50, -14, 16))
POSES['palm'] = P(lead='root', root=(-.02, .12, -.12), pel=(14, -4, 30), wst=(8, -4, 14), cst=(8, -4, 16), nck=(-4, 2, -32), hed=(-4, 2, -26), cR=(2, 8), hR=(-.04, .98, .0), hL=(.30, -.30, -.34), eR=(.7, 0, -.8), eL=(.6, -1, 0), gR='splay', gL='claw', wR=(-85, -62, 0), fL=(.10, .28, .043), fR=(.08, -.30, .065), kL=(.2, 1, 0), kR=(.2, 1, -.2), tL=(8, 0, 0), tR=(34, -26, 28))
POSES['lunge_low'] = P(lead='root', root=(0, .10, -.215), pel=(46, 0, -24), wst=(10, 0, -8), cst=(6, 0, -12), nck=(-26, 0, 20), hed=(-26, 0, 16), hL=(.22, .56, -.52), hR=(.40, -.70, .30), eL=(1, 0, .2), eR=(.5, -.6, 1), gL='claw', gR='fist', fL=(.09, .24, .043), fR=(.07, -.46, .09), kL=(.35, 1, 0), kR=(.2, 1, -.5), tL=(8, 0, 0), tR=(20, -52, 40))


def idle(p, f, amt=1., seed=0):
    """breath and weight for a held pose: call every frame (or on twos); never freezes, never floats"""
    p = POSES[p] if isinstance(p, str) else p; s = math.sin(f / 24. * 2 * math.pi / 2.9 + seed); t = math.sin(f / 24. * 2 * math.pi / 5.3 + 1.7 * seed) * amt; s *= amt
    return add_pose(p, cst=(-1.3 * s, .3 * t, 0), wst=(-.5 * s, 0, .3 * t), cL=(.45 * s, 0), cR=(.45 * s, 0), hed=(.8 * s, .4 * t, .5 * t), root=(.002 * t, 0, .0012 * s))
# ---- traversal
POSES['slide_low'] = P(lead='root', root=(.02, -.06, -.375), pel=(-24, -16, -22), wst=(20, 6, 6), cst=(18, 8, 8), nck=(6, 4, 10), hed=(0, 4, 12), cL=(3, 4), cR=(4, -2), hL=(.30, .62, .22), hR=(.50, -.60, -.52), eL=(.9, 0, -.6), eR=(.6, -.8, .5), gL='claw', gR='splay', wR=(-40, -50, 0), fL=(.05, .44, .055), fR=(.10, .05, .085), kL=(.1, 1, .5), kR=(1, .5, .25), tL=(4, 8, 0), tR=(70, -20, 10))
POSES['vault'] = P(lead='root', root=(.04, .0, .14), pel=(34, 44, -24), wst=(6, 6, -8), cst=(4, 4, -12), nck=(-18, -16, 20), hed=(-14, -14, 18), cL=(2, 5), cR=(8, -3), hL=(.30, .34, -.86), hR=(.66, -.34, .52), eL=(.8, .3, .3), eR=(.4, -.8, .6), gL='splay', gR='claw', wL=(-80, 50, 0), fL=(.24, .06, .58), fR=(.40, -.16, .62), kL=(.5, 1, .6), kR=(.9, .6, .5), tL=(10, -30, 0), tR=(20, -40, 0))
POSES['push_off'] = P(lead='root', root=(0, .05, .06), pel=(26, 4, -10), wst=(-4, 2, -8), cst=(-12, 2, -14), nck=(-12, 0, 12), hed=(-14, 0, 10), cR=(8, 2), cL=(4, 2), hR=(.20, .52, .76), hL=(.36, -.60, -.20), eR=(.9, .2, -.2), eL=(.5, -1, .3), gL='fist', gR='fist', fR=(.05, -.34, .10), fL=(.06, .30, .42), kR=(.1, 1, -.4), kL=(.2, 1, .5), tR=(6, -58, 44), tL=(6, -10, 0))
POSES['twist_tuck'] = P(lead='root', root=(0, 0, .18), pel=(58, 26, -10), wst=(16, 6, -14), cst=(18, 8, -18), nck=(-8, -6, 18), hed=(-4, -6, 18), cL=(2, 6), cR=(2, 6), hL=(-.22, .44, -.22), hR=(-.10, .40, -.40), eL=(1, .2, -.2), eR=(1, .2, -.2), gL='fist', gR='fist', fL=(.10, .24, .52), fR=(.06, .0, .46), kL=(.4, 1, .6), kR=(.3, 1, .2), tL=(8, -30, 0), tR=(8, -60, -10))
POSES['twist_open'] = P(lead='root', root=(0, .04, .20), pel=(22, -24, 30), wst=(-4, -10, 14), cst=(-10, -12, 22), nck=(-10, 8, -26), hed=(-10, 8, -20), cR=(8, -6), cL=(4, 4), hR=(.50, -.64, .42), hL=(.18, .78, -.10), eR=(.8, -.4, 1), eL=(.9, 0, -.3), gL='splay', gR='fist', wR=(-30, -16, 0), fL=(.16, .30, .44), fR=(.06, -.44, .36), kL=(.5, 1, .4), kR=(.2, 1, -.6), tL=(10, -24, 0), tR=(6, -80, -12))
POSES['guard_high'] = P(lead='root', root=(0, -.05, -.10), pel=(6, 0, 20), wst=(10, 0, 8), cst=(12, 0, 8), nck=(6, 0, -20), hed=(0, 0, -14), cL=(8, 5), cR=(8, 5), hL=(-.06, .46, .52), hR=(-.16, .40, .40), eL=(.8, .2, -1), eR=(.8, .2, -1), gL='fist', gR='fist', wL=(-60, 0, 0), wR=(-60, 0, 0), fL=(.13, -.15, .050), fR=(.10, .20, .043), kL=(.8, .7, 0), kR=(.35, 1, 0), tL=(44, -8, 0), tR=(10, 0, 0))
POSES['backfist'] = P(lead='root', root=(0, .06, -.14), pel=(16, -8, 70), wst=(6, -6, 26), cst=(6, -8, 34), nck=(-4, 4, -60), hed=(-4, 4, -40), cR=(3, 4), hR=(.50, .86, .08), hL=(.36, -.44, -.30), eR=(.6, -.6, -.4), eL=(.6, -1, .1), gR='fist', gL='fist', wR=(60, 0, 0), fL=(.16, .20, .043), fR=(.10, -.26, .06), kL=(.4, 1, 0), kR=(.4, 1, 0), tL=(20, 0, 0), tR=(34, -20, 20))
# body thrown flat and spinning round its own length: spin it with the heading
POSES['corkscrew'] = P(lead='root', root=(0, 0, .20), pel=(26, 64, -12), wst=(2, 4, -20), cst=(-4, 6, -26), nck=(-10, -10, 30), hed=(-8, -8, 24), cL=(3, 5), cR=(4, 2), hL=(.10, .56, -.16), hR=(.60, -.34, .42), eL=(.9, .1, -.3), eR=(.5, -.6, .8), gL='fist', gR='claw', fL=(.45, -.10, .52), fR=(-.30, .16, .50), kL=(.4, 1, .6), kR=(.2, 1, .5), tL=(10, -50, 0), tR=(8, -60, -10))
