"""THE DIVINE DOG: a huge wolf of shadow.  A 3-D authored head (any yaw / pitch / roll, jaw angle) plus ink-brush body keys.
    D = DogHead().pose(cx, cy, s, yaw, pitch, roll, jaw=0..1)      s = px per unit (half the skull width)
    head(cel, D, light, f, seed, wind)                              draws fur ruff, ears, mouth, fangs, eyes, mark
    draw(key, ...) -> layer   keys: 'erupt', 'leap', 'jaws'
Coordinates of the head: x right, y up, z toward the viewer, origin between the eyes."""
import numpy as np, cv2, math
from jj.ink import Cel, spl, resample, ribbon, noise1
from jj.kit import outline, shape, contours, rim, para, hatch, splatter
from jj.paint import rgb, mix
from chars import mn
from chars.mn import Lock, hair_group

PAL = {
    'battle': dict(base='#10121f', sh='#07080f', hi='#34479a', hi2='#6b9cf0', line='#030408', gum='#54172b', gumhi='#93334c', gumdk='#270915', throat='#0b0308', tongue='#7d2640', tonguehi='#b34a5f', tooth='#f2f1e8', toothsh='#8d97c4', toothdk='#4a5188', eye='#f4fbff', eyeglow='#7fc0ff', nose='#171a2a', nosehi='#5c74b8', mark='#dfe8ff', lip='#1a0d16'),
}
PAL['under'] = dict(PAL['battle'], base='#12131a', hi='#2a3048', hi2='#5d6a8a')
PAL['day'] = dict(PAL['battle'], base='#191b28', hi='#394064', hi2='#8a94c0')


class DogHead:
    HINGE = np.array([0, -.34, -.20])

    def pose(self, cx, cy, s, yaw=0., pitch=0., roll=0., jaw=.0, persp=.34):
        self.cx, self.cy, self.s, self.yaw, self.pitch, self.roll, self.jaw, self.persp = cx, cy, s, yaw, pitch, roll, jaw, persp
        a, b = yaw, pitch
        Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]); Rx = np.array([[1, 0, 0], [0, math.cos(b), math.sin(b)], [0, -math.sin(b), math.cos(b)]])
        c, n = math.cos(-roll), math.sin(-roll); self.R = np.array([[c, -n, 0], [n, c, 0], [0, 0, 1]]) @ Rx @ Ry
        return self

    def P3(self, Q):
        q = np.asarray(Q, float) @ self.R.T; f = 1 / (1 - np.clip(q[..., 2] * self.persp, -1.5, .78))
        return np.stack([self.cx + q[..., 0] * f * self.s, self.cy - q[..., 1] * f * self.s], -1)

    def depth(self, Q): return (np.asarray(Q, float) @ self.R.T)[..., 2]

    def low(self, Q):
        """rotate lower-jaw points by the jaw angle"""
        Q = np.asarray(Q, float) - self.HINGE; a = self.jaw * 1.12; c, s = math.cos(a), math.sin(a)
        y = Q[..., 1] * c - Q[..., 2] * s; z = Q[..., 1] * s + Q[..., 2] * c
        return np.stack([Q[..., 0], y, z], -1) + self.HINGE

    def k(self, p):
        """perspective scale at a 3-D point (for line weights and tooth sizes)"""
        z = float(self.depth(np.asarray(p, float))); return 1 / (1 - min(max(z * self.persp, -1.5), .78))


def mir(P): P = np.array(P, float); P[:, 0] *= -1; return P


def both(half):
    """half curve from the mouth corner (x>0) to the centre -> full curve right corner -> left corner"""
    h = np.array(half, float); return np.vstack([h, mir(h)[::-1][1:]])


UPLIP = both([(.66, -.40, .05), (.60, -.30, .34), (.50, -.25, .66), (.41, -.255, .96), (.318, -.262, 1.20), (.20, -.205, 1.42), (.08, -.140, 1.53), (0, -.160, 1.545)])
UPGUM = both([(.58, -.42, .05), (.52, -.35, .34), (.43, -.31, .66), (.345, -.285, .96), (.26, -.28, 1.18), (.16, -.25, 1.36), (.07, -.225, 1.42), (0, -.225, 1.43)])
LOLIP = both([(.66, -.42, .05), (.60, -.50, .34), (.53, -.52, .66), (.46, -.53, .94), (.38, -.535, 1.16), (.26, -.53, 1.35), (.12, -.515, 1.43), (0, -.515, 1.445)])
LOGUM = both([(.58, -.42, .05), (.53, -.455, .34), (.46, -.47, .66), (.395, -.475, .92), (.32, -.475, 1.10), (.21, -.465, 1.26), (.10, -.455, 1.325), (0, -.455, 1.335)])
# x, z along the arch, length, half width, forward lean
UPTEETH = [(.045, 1.425, .10, .040, .02), (.115, 1.40, .13, .042, .02), (.205, 1.30, .52, .070, .10), (.292, 1.10, .15, .055, .02), (.340, .95, .21, .070, .0), (.392, .78, .25, .082, .0), (.440, .58, .30, .098, .0), (.500, .36, .20, .085, .0)]
LOTEETH = [(.045, 1.332, .10, .036, .10), (.130, 1.305, .13, .040, .10), (.245, 1.205, .56, .074, .20), (.345, 1.04, .17, .054, .0), (.385, .90, .24, .066, .0), (.430, .74, .28, .080, .0), (.475, .55, .30, .092, .0), (.515, .37, .20, .080, .0)]


def _tooth(cel, D, p, d, ln, hw, side, pal, w, seed, lit, up=None, collar=True):
    """one fang: root p, direction d (3-D), curved, two-tone with a hard terminator"""
    d = np.asarray(d, float); d /= np.linalg.norm(d); k = D.k(p); t = np.linspace(0, 1, 14)[:, None]
    C = p + d * ln * t + np.array([0, 0, -1.]) * (t ** 2) * ln * .22
    Q = D.P3(C); ll = np.hypot(*(Q[-1] - Q[0])); mn_ = hw * D.s * k * 1.9
    if ll < mn_: Q = Q[0] + (Q - Q[0]) * (mn_ / (ll + 1e-3))          # foreshortened fang: keep a readable tip
    if up is not None and (Q[-1] - Q[0]) @ up < mn_ * .75:
        o_ = np.array([up[1], -up[0]]) * side * -.25; Q = Q[0] + (up + o_) * mn_ * 1.15 * t ** .9
    T = np.gradient(Q, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-6; N = np.c_[-T[:, 1], T[:, 0]]
    wd = hw * D.s * k * ((1 - t[:, 0]) ** .86 * (1 + .16 * np.sin(math.pi * np.clip(t[:, 0] * 1.5, 0, 1)))); A, B = Q + N * wd[:, None], Q - N * wd[:, None]
    P = np.vstack([A, B[::-1]]); lwd = max(1.2, w * min(k, 2.2))
    outline(cel, P, lwd * 1.5, pal['line'], seed); cel.fill(P, pal['tooth'])
    c = .12 if lit > 0 else -.12
    M = Q + N * (wd * (c + .25 * np.sin(t[:, 0] * 3)))[:, None]
    cel.fill(np.vstack([M, (B if lit > 0 else A)[::-1]]), pal['toothsh'])
    if wd[0] > 12:
        E = Q + N * (wd * (-.62 if lit > 0 else .62))[:, None]; cel.fill(np.vstack([E, (B if lit > 0 else A)[::-1]]), pal['toothdk'])
        s0 = Q[2:10] + N[2:10] * (wd[2:10] * (.55 if lit > 0 else -.55))[:, None]; cel.line(s0, wd[0] * .16, '#ffffff', taper=(.3, .6), seed=seed)
    cel.line(A, lwd * .7, pal['line'], taper=(.1, .1), seed=seed + 1); cel.line(B, lwd * .7, pal['line'], taper=(.1, .1), seed=seed + 2)
    # gum collar closes over the root
    if not collar or np.hypot(*(Q[-1] - Q[0])) < wd[0] * 1.5: return P
    g0, g1 = A[0] + (A[0] - B[0]) * .16, B[0] + (B[0] - A[0]) * .16; tt = T[0] * min(1., .34 * np.hypot(*(Q[-1] - Q[0])) / (wd[0] + 1e-6))
    col = np.vstack([spl([g0 - tt * wd[0] * .22, g0 + tt * wd[0] * .10, Q[0] + tt * wd[0] * .52, g1 + tt * wd[0] * .10, g1 - tt * wd[0] * .22], n=6)])
    cel.fill(col, pal['gum']); cel.line(spl([g0 + tt * wd[0] * .02, Q[0] + tt * wd[0] * .50, g1 + tt * wd[0] * .02], n=8), lwd * .8, pal['line'], taper=(.3, .3), seed=seed + 3)
    return P


def _arch(teeth, gum, D, low):
    out = []
    g = np.array(gum)
    for x, z, ln, hw, lean in teeth:
        for s in (-1, 1):
            h = g[:len(g) // 2 + 1]; y = float(np.interp(z, h[:, 2], h[:, 1])); p = np.array([s * x, y, z]); d = np.array([s * (-.05 if ln < .4 else .04), 1. if low else -1., lean])
            if low: q = D.low(np.array([p, p + d])); p, d = q[0], q[1] - q[0]
            out.append((float(D.depth(p)), p, d, ln, hw, s))
    return sorted(out, key=lambda t: t[0])


def _fur(D, pts, w, id, kind='spike', tag=''):
    l = Lock.__new__(Lock); P = np.asarray(pts, float); l.p0 = P[0]; l.p1 = P[len(P) // 2]; l.p2 = P[-1]; l.path = P
    n = P[0] - [0, .2, -.6]; l.n = n / np.linalg.norm(n); l.w = w; l.kind = kind; l.asym = 0.; l.flex = 1.; l.id = id; l.w1 = None; l.tag = tag; return l


def fur_locks(D):
    """ruff, crown and cheek fur of the head: big designed flames, not fuzz"""
    r = np.random.default_rng(11); L = []; i = 0
    def add(p, d, ln, w, curl, tag, kind='spike'):
        nonlocal i; i += 1; p = np.array(p, float); d = np.array(d, float); d /= np.linalg.norm(d)
        L.append(_fur(D, [p, p + d * ln * .35 + np.array(curl) * .12, p + d * ln * .72 + np.array(curl) * .5, p + d * ln + np.array(curl)], w, i, kind, tag))
    for s in (-1, 1):
        # outer mane, three depths
        for y, z, dx, dy, dz, ln, w, c in ((1.05, -.70, .55, 1.0, -.55, 1.10, .52, (.15, .05, -.2)), (.80, -.75, .9, .75, -.6, 1.25, .56, (.12, .22, -.2)), (.45, -.80, 1, .35, -.6, 1.35, .58, (.05, .30, -.1)), (.05, -.80, 1, .0, -.6, 1.30, .56, (.0, .30, -.1)), (-.35, -.75, 1, -.35, -.5, 1.25, .54, (-.05, .25, -.05)), (-.75, -.65, .85, -.8, -.4, 1.20, .52, (-.1, .18, 0)), (-1.05, -.50, .5, -1.05, -.2, 1.10, .50, (-.12, .05, 0))):
            j = r.uniform(.85, 1.2); add((s * .72, y * .85, z), (s * dx, dy, dz), ln * j, w, (s * c[0], c[1], c[2]), 'back')
        # cheeks and jowls, pointing out and back
        for y, z, dx, dy, dz, ln, w, c in ((.72, -.15, .8, .55, -.55, .62, .34, (.06, .12, 0)), (.40, -.05, 1, .12, -.6, .72, .38, (.04, .18, 0)), (.08, .0, 1, -.20, -.6, .80, .40, (.02, .18, 0)), (-.22, .0, .95, -.52, -.5, .78, .38, (.0, .15, 0)), (-.50, -.10, .75, -.9, -.3, .74, .36, (-.04, .10, 0)), (.56, -.35, 1, .35, -.4, .92, .40, (.04, .16, 0)), (-.06, -.35, 1, -.15, -.4, 1.0, .42, (.02, .22, 0)), (-.38, -.35, .9, -.6, -.3, .95, .40, (-.02, .16, 0))):
            j = r.uniform(.85, 1.2); add((s * .70, y, z), (s * dx, dy, dz), ln * j, w, (s * c[0], c[1], c[2]), 'cheek')
        # crown
        for x, y, z, dx, dy, dz, ln, w in ((.18, 1.02, .0, .18, 1, -.45, .48, .30), (.46, .95, -.05, .5, 1, -.5, .52, .30), (.30, .86, .18, .28, 1, .1, .34, .20), (.62, .80, .05, .8, .9, -.4, .44, .26)):
            add((s * x, y, z), (s * dx, dy, dz), ln * r.uniform(.8, 1.25), w, (s * .05, .05, -.05), 'crown')
        # chin beard (on the lower jaw)
        for x, z, dx, dy, dz, ln, w in ((.10, 1.05, .05, -1, .15, .40, .20), (.22, .80, .25, -1, .05, .46, .24), (.34, .50, .45, -1, -.1, .52, .28), (.46, .20, .7, -1, -.2, .56, .30)):
            p = D.low(np.array([(s * x, -.62, z), (s * (x + dx * ln), -.62 + dy * ln, z + dz * ln)])); i += 1
            L.append(_fur(D, [p[0], p[0] * .5 + p[1] * .5 + [s * .03, 0, .05], p[1]], w, i, 'spike', 'beard'))
    return L


def _mesh_ell(c, r, n=20):
    u, v = np.meshgrid(np.linspace(0, 2 * math.pi, 2 * n), np.linspace(-math.pi / 2, math.pi / 2, n)); return np.stack([c[0] + r[0] * np.cos(v) * np.sin(u), c[1] + r[1] * np.sin(v), c[2] + r[2] * np.cos(v) * np.cos(u)], -1)


def _loft(rows, n=28):
    """rows: (z, cy, rx, ry)"""
    R = np.array(rows, float); zz = np.linspace(R[0, 0], R[-1, 0], 24); t = np.linspace(0, 2 * math.pi, n); f = lambda j: np.interp(zz, R[:, 0], R[:, j])[:, None]
    return np.stack([np.cos(t)[None] * f(2), f(1) + np.sin(t)[None] * f(3), np.repeat(zz[:, None], n, 1)], -1)


def _shell(rows, n=24):
    """upper muzzle: open shell from the left lip over the top to the right lip.  rows (z, y lip, half width, y top)"""
    R = np.array(rows, float); zz = np.linspace(R[0, 0], R[-1, 0], 36); t = np.linspace(0, math.pi, n); f = lambda j: np.interp(zz, R[:, 0], R[:, j])[:, None]
    c, sn = np.cos(t), np.sin(t); x = np.sign(c) * np.abs(c) ** .8; y = sn ** .8
    return np.stack([x[None] * f(2), f(1) + (f(3) - f(1)) * y[None], np.repeat(zz[:, None], n, 1)], -1)


def sil(D, meshes):
    Ps = [D.P3(m) for m in meshes]; A = np.vstack([p.reshape(-1, 2) for p in Ps]); lo = A.min(0) - 8; hi = A.max(0) + 8; k = min(1., 1400. / max(hi - lo))
    w, h = np.ceil((hi - lo) * k).astype(int) + 2; m = np.zeros((h, w), np.uint8)
    for p in Ps:
        q = (p - lo) * k; quad = np.stack([q[:-1, :-1], q[:-1, 1:], q[1:, 1:], q[1:, :-1]], 2).reshape(-1, 4, 2)
        for c in np.round(quad * 16).astype(np.int32): cv2.fillConvexPoly(m, c, 255, cv2.LINE_8, 4)
    return resample(max(contours(m / 255.), key=len) / k + lo, 3.0)


def head(cel, D, light='battle', f=0, seed=0, wind=(0, 0), w=3.0, key=(-.6, -.8), parts='bf', eyes=1.0):
    pal = PAL[light]; s = D.s; ln = pal['line']; lit = 1 if key[0] < 0 else -1
    L = fur_locks(D)
    class _H: pass
    for l in L: l.build(D, mn.wind_off(D, wind, f, (l.id * .618) % 1))
    cols = dict(base=pal['base'], sh=pal['sh'], hi=pal['hi'], hi2=pal['hi2'], hip=0, wedge=.72, stroke=pal['hi'])
    if 'b' in parts:
        bk = sorted([l for l in L if l.tag == 'back'], key=lambda l: l.depth)
        hair_group(cel, bk, cols, w * 1.3, w * .6, w * .5, seed, key, line=ln)
        for q in (-1, 1):                                  # ears, laid back
            E = [(q * .38, 1.02, -.20), (q * .52, 1.42, -.42), (q * .80, 1.86, -.80), (q * .98, 1.34, -.62), (q * .84, .90, -.40)]
            P = spl(D.P3(np.array(E)), True, corners=(2,), n=8); outline(cel, P, w * 2, ln, seed + 3); cel.fill(P, pal['base'])
            I = spl(D.P3(np.array([(q * .50, 1.06, -.22), (q * .62, 1.36, -.42), (q * .80, 1.72, -.72), (q * .86, 1.32, -.56), (q * .74, .98, -.34)])), True, corners=(2,), n=8); cel.fill(I, pal['sh']); cel.shade(I + np.array([q * .05, -.03]) * s, pal['hi'], within=I)
    if 'f' not in parts: return cel
    skull = _mesh_ell((0, .32, -.28), (.84, .80, .80)); muz = _shell([(.05, -.40, .68, .54), (.34, -.30, .61, .49), (.66, -.25, .515, .37), (.96, -.255, .425, .275), (1.20, -.262, .325, .212), (1.42, -.205, .208, .158), (1.53, -.145, .09, .11), (1.56, -.155, .01, .05)])
    lowm = D.low(_loft([(-.1, -.50, .60, .16), (.3, -.56, .52, .13), (.7, -.585, .42, .11), (1.0, -.59, .43, .10), (1.2, -.585, .35, .09), (1.38, -.575, .24, .07)]))
    ck = sorted([l for l in L if l.tag in ('cheek', 'crown')], key=lambda l: l.depth)
    bd = sorted([l for l in L if l.tag == 'beard'], key=lambda l: l.depth)
    # --- lower jaw
    hair_group(cel, bd, cols, w * 1.2, w * .6, w * .5, seed + 5, key, line=ln)
    lj = sil(D, [lowm]); outline(cel, lj, w * 2, ln, seed + 6); cel.fill(lj, pal['base'])
    lolip, logum = D.P3(D.low(LOLIP)), D.P3(D.low(LOGUM)); uplip, upgum = D.P3(UPLIP), D.P3(UPGUM)
    SL, SG, UL, UG = [spl(q, n=8) for q in (lolip, logum, uplip, upgum)]
    # head mass (skull + muzzle) with cheeks first so the mouth opens in front of it
    hm = sil(D, [skull]); outline(cel, hm, w * 1.8, ln, seed + 7); cel.fill(hm, pal['base'])
    hair_group(cel, ck, cols, w * 1.2, w * .6, w * .5, seed + 8, key, caps=[], line=ln)
    # --- mouth cavity
    cav = np.vstack([UL, SL[::-1]])
    if D.jaw > .04:
        cel.fill(cav, pal['throat']); cel.clip(cav)
        # cheek webs: stretched gum skin between the jaws
        for q in (0, -1):
            a0, b0 = UG[q], SG[q]; sgn = 1 if q == 0 else -1
            U3 = UPGUM[:4] if sgn > 0 else UPGUM[-4:][::-1]; L3 = D.low(LOGUM[:4] if sgn > 0 else LOGUM[-4:][::-1])
            web = np.vstack([spl(D.P3(U3), n=6), spl(D.P3(np.array([U3[-1], (U3[-1] + L3[-1]) / 2 * [.95, 1, .55], L3[-1]])), n=6), spl(D.P3(L3), n=6)[::-1]]); cel.fill(web, pal['gumdk'])
            for v in (.3, .55, .8):
                cel.line(spl(D.P3(np.array([U3[1] * (1 - v) + U3[3] * v, ((U3[1] * (1 - v) + U3[3] * v) + (L3[1] * (1 - v) + L3[3] * v)) / 2 * [.93, 1, .8], L3[1] * (1 - v) + L3[3] * v])), n=8), w * 1.1, pal['gum'], taper=(.2, .2), seed=seed + 9)
        # palate with ridges
        pal3 = np.vstack([UPGUM, [(-.30, -.30, .0), (.30, -.30, .0)]]); cel.fill(D.P3(pal3), pal['gumdk'])
        for z in (.35, .55, .75, .95, 1.12, 1.26):
            x = float(np.interp(z, UPGUM[:8, 2], UPGUM[:8, 0])) * .86; R = spl(D.P3(np.array([(-x, -.30, z - .06), (-x * .4, -.27, z + .01), (0, -.28, z + .03), (x * .4, -.27, z + .01), (x, -.30, z - .06)])), n=8); cel.line(R, w * 1.5 * D.k((0, -.3, z)), pal['gum'], taper=(.3, .3), seed=seed + int(z * 30))
        # tongue: long, curling up at the tip, central groove
        tg = both([(.38, -.46, .15), (.36, -.40, .5), (.32, -.40, .8), (.26, -.41, 1.02), (.15, -.38, 1.20), (0, -.355, 1.26)]); T3 = D.low(tg); TP = spl(D.P3(T3), n=8); TP = np.vstack([TP, D.P3(D.low(np.array([(0, -.30, .0)])))])
        cel.fill(TP, pal['tongue']); cel.clip(TP)
        cel.fill(np.vstack([spl(D.P3(D.low(np.array([(lit * -.30, -.42, .2), (lit * -.22, -.38, .6), (lit * -.16, -.40, .95), (lit * -.05, -.385, 1.14)]))), n=8), spl(D.P3(D.low(np.array([(lit * -.10, -.36, 1.05), (lit * -.13, -.36, .7), (lit * -.12, -.40, .25)]))), n=8)]), pal['tonguehi'])
        for z0, c0 in ((.52, .55), (.30, .0)):               # the throat swallows the back of the tongue in two hard steps
            cc = mn.mixh(pal['throat'], pal['tongue'], c0); cel.fill(np.vstack([spl(D.P3(D.low(np.array([(-.5, -.3, z0 + .04), (-.22, -.3, z0 - .06), (-.06, -.3, z0 + .08), (.10, -.3, z0 - .04), (.24, -.3, z0 + .05), (.5, -.3, z0 - .03)]))), n=6), D.P3(np.array([(.8, .0, -.4), (-.8, .0, -.4)]))]), cc)
        cel.unclip(); cel.line(spl(D.P3(D.low(np.array([(0, -.42, .1), (0.01, -.40, .5), (0, -.40, .85), (0, -.39, 1.08)]))), n=8), w * 1.8, pal['gumdk'], taper=(.4, .5), seed=seed + 20); cel.line(TP[:-1], w * 1.3, ln, taper=(.1, .1), seed=seed + 21)
        # gums
        cel.fill(np.vstack([SL, SG[::-1]]), pal['gum'])
        cel.unclip()
        h = len(UL) // 2
        for i0, i1 in ((0, h + 1), (h, len(UL))):
            u, g = UL[i0:i1], UG[i0:i1]; cel.fill(np.vstack([u, g[::-1]]), pal['gum']); cel.fill(np.vstack([u, (u * .55 + g * .45)[::-1]]), pal['lip']); cel.fill(np.vstack([u * .75 + g * .25, (u * .55 + g * .45)[::-1]]), pal['gumdk']); cel.line(g, w * 1.0, ln, taper=(.1, .1), seed=seed + 23)
    # --- teeth: lower then upper, far to near
    o = D.P3(np.array([(0, 0, .6), (0, 1., .6)])); uu = (o[1] - o[0]) / (np.hypot(*(o[1] - o[0])) + 1e-6)
    for dp, p, d, l_, hw, sd in _arch(LOTEETH, LOGUM, D, True): _tooth(cel, D, p, d, l_, hw, sd, pal, w, seed + int(abs(p[0]) * 100), lit, uu, l_ > .4)
    cel.fill(np.vstack([SL, (SL + (SG - SL) * .55)[::-1]]), pal['lip']); cel.line(SL, w * 1.5, ln, taper=(.05, .05), seed=seed + 25)
    o = D.P3(np.array([(0, 0, .6), (0, 1., .6)])); up = (o[1] - o[0]) / (np.hypot(*(o[1] - o[0])) + 1e-6) * 5 * s
    for dp, p, d, l_, hw, sd in _arch(UPTEETH, UPGUM, D, False): _tooth(cel, D, p, d, l_, hw, sd, pal, w, seed + 40 + int(abs(p[0]) * 100), lit)
    # --- muzzle over the upper gums
    mz = sil(D, [muz]); o = D.P3(np.array([(0, 0, .6), (0, 1., .6)])); up = (o[1] - o[0]) / (np.hypot(*(o[1] - o[0])) + 1e-6) * 5 * s
    outline(cel, mz, w * 1.9, ln, seed + 27); cel.fill(mz, pal['base'])
    face = np.vstack([spl(D.P3(np.array([(-.66, -.40, .05), (-.72, -.05, .10), (-.70, .35, .12), (-.50, .72, .18), (0, .86, .28), (.50, .72, .18), (.70, .35, .12), (.72, -.05, .10), (.66, -.40, .05)])), n=8)]); cel.fill(np.vstack([UL, face]), pal['base'])
    h = len(UL) // 2                                       # upper lip drapes over the roots
    for i0, i1 in ((0, h + 1), (h, len(UL))):
        if abs(D.yaw) > .5 and (i0 == 0) == (D.yaw > 0): continue
        u, g = UL[i0:i1], UG[i0:i1]; e_ = u * .46 + g * .54; cel.fill(np.vstack([u, e_[::-1]]), pal['lip']); cel.fill(np.vstack([u, (u * .80 + g * .20)[::-1]]), pal['base']); cel.line(e_, w * 1.1, ln, taper=(.1, .1), seed=seed + 24)
    ph = D.P3(np.array([(-.10, -.04, 1.54), (.10, -.04, 1.54), (.10, -.15, 1.525), (.03, -.175, 1.55), (-.03, -.175, 1.55), (-.10, -.15, 1.525)])); cel.fill(ph, pal['base'])
    _facework(cel, D, pal, w, seed, key, lit, UL, eyes)
    return cel


def _facework(cel, D, pal, w, seed, key, lit, UL, eyes):
    ln = pal['line']; s = D.s; P = lambda q: D.P3(np.array(q, float)); hi, hi2 = pal['hi'], pal['hi2']
    # muzzle light planes: top plane on the lit side, lip ridge, bridge
    for q in (-1, 1):
        if q * D.yaw > .55: continue
        a = 1 if q == -lit else .6
        c = hi if q == -lit else mn.mixh(pal['base'], hi, .55)
        cel.fill(np.vstack([spl(P([(q * .05, .20, 1.40), (q * .16, .18, 1.18), (q * .24, .24, .86), (q * .30, .33, .55)]), n=6), spl(P([(q * .16, .36, .50), (q * .10, .28, .86), (q * .04, .22, 1.2)]), n=6)]), c)
        cel.fill(np.vstack([spl(P([(q * .24, -.02, 1.36), (q * .36, -.02, 1.08), (q * .46, -.02, .74), (q * .56, -.02, .42)]), n=6), spl(P([(q * .50, -.10, .46), (q * .42, -.11, .78), (q * .33, -.12, 1.08)]), n=6)]), c)
        # whisker pad creases
        for z, y0 in ((1.28, .04), (1.14, .06), (.98, .08), (.80, .11), (.62, .15)):
            x = float(np.interp(z, [.3, .7, 1.05, 1.3, 1.46], [.57, .46, .37, .29, .21]))
            cel.line(spl(P([(q * x * .35, y0 + .17, z - .02), (q * x * .72, y0 + .10, z + .02), (q * x * .98, y0 - .06, z)]), n=8), w * 1.15 * D.k((x, 0, z)) ** .5, ln, taper=(.4, .5), press=.35, seed=seed + int(z * 90))
            cel.line(spl(P([(q * x * .40, y0 + .20, z - .03), (q * x * .72, y0 + .135, z + .01), (q * x * .92, y0 + .02, z)]), n=8), w * .9 * D.k((x, 0, z)) ** .5, hi if q != -lit else hi2, taper=(.5, .5), seed=seed + int(z * 91), alpha=.75)
        # brow ridge: heavy black wedge over the eye + lit top edge
        br = np.vstack([spl(P([(q * .06, .40, .56), (q * .26, .47, .48), (q * .52, .60, .32), (q * .84, .80, .02)]), n=8), spl(P([(q * .80, .66, .06), (q * .52, .47, .36), (q * .28, .36, .52), (q * .10, .30, .60)]), n=8)])
        # eye: slit, pale
        e = np.vstack([spl(P([(q * .22, .335, .55), (q * .36, .42, .47), (q * .52, .475, .37), (q * .66, .52, .26)]), n=8), spl(P([(q * .55, .395, .37), (q * .40, .325, .46), (q * .28, .30, .53)]), n=8)])
        sock = np.vstack([spl(P([(q * .14, .33, .58), (q * .36, .47, .47), (q * .74, .60, .20)]), n=8), spl(P([(q * .64, .34, .34), (q * .44, .22, .46), (q * .24, .23, .56)]), n=8)]); cel.fill(sock, pal['sh'])
        if eyes > 0:
            cel.fill(e, pal['eyeglow']); cel.clip(e); c0 = P([(q * .42, .385, .44)])[0]; r = .115 * s * D.k((q * .42, .385, .44)); t = np.linspace(0, 6.283, 24)
            cel.fill(np.c_[c0[0] + np.cos(t) * r * 1.08, c0[1] + np.sin(t) * r * 1.12], '#3f78d8'); cel.fill(np.c_[c0[0] + np.cos(t) * r, c0[1] + np.sin(t) * r * 1.05], pal['eye']); cel.fill(np.c_[c0[0] + np.cos(t) * r * .17, c0[1] + np.sin(t) * r * .30], '#05060c'); cel.fill(np.vstack([e[:17], (e[:17] + np.array([0, .035]) * s)[::-1]]), '#8fb4ea'); cel.unclip()
            cel.line(np.vstack([e, e[:1]]), w * .9, ln, taper=(.1, .1), seed=seed + 31)
        cel.fill(br, pal['sh']); cel.line(spl(P([(q * .04, .42, .56), (q * .26, .495, .48), (q * .52, .63, .32), (q * .84, .84, .02)]), n=8), w * 2.0, hi2 if q == -lit else hi, taper=(.3, .5), press=.4, seed=seed + 33)
        # under-eye tear line
        cel.line(spl(P([(q * .20, .28, .58), (q * .24, .18, .70), (q * .22, .12, .82)]), n=8), w * 1.2, ln, taper=(.2, .6), seed=seed + 34)
    # forehead furrow + unusual mark
    for x in (-.07, .07): cel.line(spl(P([(x, .62, .42), (x * .8, .50, .52), (x * 1.4, .40, .58)]), n=8), w * 1.2, ln, taper=(.4, .4), seed=seed + 36)
    M = lambda L_: spl(P([(x, y, .42 - .38 * max(0, y - .55) - .25 * abs(x)) for x, y in L_]), True, n=6)
    for shp in ([(0, 1.06), (.050, .88), (.026, .70), (0, .60), (-.026, .70), (-.050, .88)], ):
        cel.fill(M(shp), pal['mark'])
    for q in (-1, 1):
        cel.fill(M([(q * .075, .70), (q * .20, .78), (q * .27, .92), (q * .235, 1.02), (q * .205, .93), (q * .15, .85), (q * .075, .80)]), pal['mark'])
        cel.fill(M([(q * .34, .74), (q * .39, .80), (q * .37, .87), (q * .33, .81)]), pal['mark'])
    # lip line, nose
    cel.line(UL, w * 1.8, ln, taper=(.06, .06), press=.3, seed=seed + 38)
    k = D.k((0, .05, 1.55)); half = [(0, .145, 1.45), (.085, .150, 1.45), (.155, .105, 1.465), (.172, .050, 1.49), (.140, -.005, 1.535), (.085, -.030, 1.555), (.040, -.060, 1.565), (0, -.070, 1.57)]
    NP = np.vstack([spl(P(half), n=8), spl(P(mir(half)), n=8)[::-1]])
    outline(cel, NP, w * 2, ln, seed + 39); cel.fill(NP, pal['nose']); cel.clip(NP)
    cel.fill(np.vstack([spl(P([(-.15, .09, 1.47), (-.10, .135, 1.455), (.0, .128, 1.455), (.085, .138, 1.455), (.14, .10, 1.465)]), n=6), spl(P([(.10, .098, 1.50), (.05, .085, 1.52), (0, .098, 1.515), (-.06, .078, 1.52), (-.11, .085, 1.50)]), n=6)]), pal['nosehi'])
    cel.fill(np.vstack([spl(P([(-.17, .03, 1.50), (-.10, -.02, 1.54), (0, -.03, 1.56), (.10, -.02, 1.54), (.17, .03, 1.50)]), n=6), P([(.2, -.2, 1.5), (-.2, -.2, 1.5)])]), '#06070d')
    r = np.random.default_rng(5)
    for q in range(16):
        x_, y_ = r.uniform(-.13, .13), r.uniform(.02, .13); cel.line(P([(x_, y_, 1.48 + .06 * (1 - y_ / .13)), (x_ + r.uniform(-.02, .02), y_ - r.uniform(.012, .03), 1.49 + .06 * (1 - y_ / .13))]), w * .8, ln if r.random() < .6 else pal['hi'], taper=(.3, .3), seed=q)
    cel.unclip()
    for q in (-1, 1):
        nz = spl(P([(q * .040, .020, 1.565), (q * .085, .052, 1.55), (q * .140, .036, 1.515), (q * .155, .002, 1.515), (q * .115, .004, 1.545), (q * .075, -.012, 1.56)]), True, n=6); cel.fill(nz, '#000004'); cel.line(nz[:len(nz) // 2], w * 1.0, pal['nosehi'], taper=(.3, .4), seed=seed + 45, alpha=.8)
    cel.line(P([(0, -.0, 1.57), (0.004, -.07, 1.57), (0, -.155, 1.545)]), w * 1.5 * k ** .5, ln, taper=(.3, .1), seed=seed + 44)


# ------------------------------------------------------------------ 2-D fur masses (screen px)
def mane(cx, cy, rx, ry, n, ln, w, seed=0, a0=0., a1=2 * math.pi, swirl=.25, jit=.35, inner=.75):
    """ring of big flame locks radiating from an ellipse, three sizes, uneven"""
    r = np.random.default_rng(seed); L = []
    for i, a in enumerate(np.sort(np.linspace(a0, a1, n, endpoint=False) + r.normal(0, .12 * (a1 - a0) / n * 3, n))):
        k = r.choice([.55, .8, 1.0, 1.45], p=[.25, .3, .3, .15]); l_ = ln * k * r.uniform(.8, 1.2); sg = swirl * r.uniform(.2, 1.4) * (1 if r.random() < .8 else -1)
        p0 = np.array([cx + math.cos(a) * rx * inner, cy + math.sin(a) * ry * inner]); e = np.array([math.cos(a), math.sin(a)]); t = np.array([-e[1], e[0]])
        pts = [p0, p0 + e * l_ * .38 + t * l_ * sg * .10, p0 + e * l_ * .72 + t * l_ * sg * .45, p0 + e * l_ + t * l_ * sg]
        L.append(Lock.d2(pts, w * (k ** .6) * r.uniform(.8, 1.25), 'ink', r.uniform(-.25, .25), 1000 + i + seed * 50, depth=k * -1))
        if r.random() < .6:
            o = t * w * r.uniform(-.6, .6); L.append(Lock.d2([p + o * (1 - j * .2) + e * l_ * .10 for j, p in enumerate(pts)], w * .16, 'ink', 0, 3000 + i + seed * 50, depth=0))
    return sorted(L, key=lambda l: l.depth)


def body(path, width, seed=0, dens=1.0, ln=1.15, spread=.55, f=0, sway=0.):
    """furred tube along a screen path (tail -> head): shingled flame locks that stream back toward the tail.  width = [(t, half width px)]"""
    r = np.random.default_rng(seed); C = resample(spl(np.asarray(path, float), n=12), 4.0); T = np.gradient(C, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-6; N = np.c_[-T[:, 1], T[:, 0]]
    n = len(C); tw = np.array(width, float); W_ = np.interp(np.linspace(0, 1, n), tw[:, 0], tw[:, 1]); L = []; i = 0; t = 0.
    while t < n - 1:
        j = int(t); w = W_[j]; rows = 4 + int(r.integers(0, 2))
        for v in np.linspace(-1, 1, rows) + r.normal(0, .13, rows):
            v = float(np.clip(v, -1.05, 1.05)); k = r.choice([.55, .85, 1.2, 1.8], p=[.3, .35, .25, .1]); l_ = w * ln * k * r.uniform(.8, 1.2)
            e = -T[j] * (1 - abs(v) * spread * .55) + N[j] * v * spread * r.uniform(.6, 1.25); e /= np.hypot(*e); q = np.array([-e[1], e[0]]) * (1 if v < 0 else -1)
            b = math.sin(f * .42 + t * .02 + v * 3) * sway * l_
            p0 = C[j] + N[j] * w * v * .84; pts = [p0, p0 + e * l_ * .4 + q * l_ * .02, p0 + e * l_ * .75 + q * (l_ * .12 + b * .5), p0 + e * l_ + q * (l_ * .24 + b)]
            i += 1; L.append(Lock.d2(pts, w * .46 * (k ** .5), 'ink', r.uniform(-.2, .2), seed * 100 + i, depth=t + (1 - abs(v)) * 6))
        t += max(3., w * .46 / dens / 4.)
    edge = np.vstack([C + N * W_[:, None] * .86, (C - N * W_[:, None] * .86)[::-1]])
    return L, edge


def fur_fill(cel, locks, pal, w, seed=0, key=(-.6, -.8), core=None, wedge=.5, hi2=True):
    cols = dict(base=pal['base'], sh=pal['sh'], hi=pal['hi'], wedge=wedge, stroke=pal['hi'])
    if hi2: cols['hi2'] = pal['hi2']; cols['splint'] = 1
    hair_group(cel, locks, cols, w * 1.1, w * .5, w * .45, seed, key, caps=[core] if core is not None else (), line=pal['line']); return cel


def rim2(layer, dx, dy, col, amt=1., soft=.6, where=None):
    """rim light that does not leak along the frame edge"""
    a = layer[..., 3]; h, w = a.shape; sh = cv2.warpAffine(a, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE); m = np.clip(a - sh, 0, 1)
    if soft > 0: m = cv2.GaussianBlur(m, (0, 0), soft) * (a > .02)
    if where is not None: m = m * where
    c = rgb(col); out = layer.copy(); out[..., :3] = out[..., :3] * (1 - m[..., None] * amt) + c * (m * a)[..., None] * amt; return out


def paw(cel, c, s, ang, pal, w, seed=0, spread=1., lit=-1, curl=.0):
    """clawed forepaw, toes pointing along ang (screen radians, 0 = right)"""
    a = ang - math.pi / 2; R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]); P = lambda L: np.asarray(c, float) + (np.asarray(L, float) * s) @ R.T; ln = pal['line']
    base = P(spl([(-.46, -.5), (-.60, .0), (-.52, .40), (0, .52), (.52, .40), (.60, .0), (.46, -.5)], n=8)); outline(cel, base, w * 1.7, ln, seed); cel.fill(base, pal['base'])
    for i, (x, l_) in enumerate(((-.47, .62), (.46, .60), (-.16, .80), (.16, .78))):
        x *= spread; e = np.array([x * .42, 1.]); e /= np.hypot(*e); n = np.array([e[1], -e[0]]); r0 = np.array([x, .12]); tip = r0 + e * l_
        k = P(spl([tip - n * .085 - e * .12, tip + e * (.16 - curl * .1) - n * .03, tip + e * (.50 - curl * .25) + n * (.05 + .10 * np.sign(x)) , tip + e * .10 + n * .09, tip + n * .085 - e * .12], n=6)); outline(cel, k, w * 1.4, ln, seed + i); cel.fill(k, pal['tooth']); cel.shade(k + np.array([lit * -.06, .02]) * s, pal['toothsh'], within=k)
        toe = P(spl([r0 - n * .20, r0 - n * .23 + e * l_ * .55, tip - n * .13, tip + e * .05, tip + n * .13, r0 + n * .23 + e * l_ * .55, r0 + n * .20], n=7)); outline(cel, toe, w * 1.5, ln, seed + 10 + i); cel.fill(toe, pal['base'])
        cel.fill(P(spl([r0 + n * lit * .18 + e * .1, r0 + n * lit * .21 + e * l_ * .55, tip + n * lit * .10, tip + n * lit * .02 - e * .06, r0 + n * lit * .06 + e * l_ * .5], True, n=6)), pal['hi'])
        cel.line(P(np.array([r0 - n * .02, r0 + e * l_ * .4 + n * .03, r0 + e * l_ * .7])), w * .7, ln, taper=(.3, .5), seed=seed + 20 + i)
    r = np.random.default_rng(seed); L = []
    for i, x in enumerate(np.linspace(-.5, .5, 6)): q = P([(x, -.55), (x * 1.1, -.2), (x * 1.15 + r.uniform(-.08, .08), .05 + r.uniform(0, .25))]); L.append(Lock.d2(q, s * .28, 'ink', 0, seed * 9 + i))
    hair_group(cel, L, dict(base=pal['base'], sh=pal['sh'], hi=pal['hi'], wedge=.7), w, w * .5, w * .4, seed, (lit, -.6), line=ln)



def scaled(spec, k, ox, oy, rx=960, ry=500, spread=1.):
    """spec scaled by k about (rx, ry), that point moved to (ox, oy)"""
    f = lambda p: ((p[0] - rx) * k + ox, (p[1] - ry) * k + oy); wd = lambda w: [(t, v * k) for t, v in w]; s = dict(spec)
    if 'path' in s: s['path'] = [f(p) for p in s['path']]; s['width'] = wd(s['width'])
    h = dict(s['head']); h['cx'], h['cy'] = f((h['cx'], h['cy'])); h['s'] *= k; s['head'] = h
    if s.get('mane'): m = dict(s['mane']); m.update(rx=m['rx'] * k, ry=m['ry'] * k, ln=m['ln'] * k, w=m['w'] * k); s['mane'] = m
    s['legs'] = [dict(g, path=[f(((p[0] - rx) * spread + rx, p[1])) for p in g['path']], width=wd(g['width']), paw=(g['paw'][0] * k, g['paw'][1])) for g in s.get('legs', ())]
    return s


def draw(spec, light='battle', f=0, seed=3, w=3.0, key=(-.6, -.8), size=(1920, 1080), rimc='#79c4ff', rimd=(6, 5), paramt=.8, layer=True, cel=None):
    """whole dog from a spec: dict(path, width, head=dict(...), legs=[dict(path, width, paw=(size, angle), front=bool)], mane=dict(...))"""
    pal = PAL[light]; c = cel or Cel(*size); lit = -1 if key[0] < 0 else 1
    def leg(g, sd):
        L, e = body(g['path'], g['width'], sd, g.get('dens', .9), g.get('ln', .95), g.get('spread', .42), f, g.get('sway', .0)); fur_fill(c, L, pal, w * .9, sd, key, e, .55)
        if 'paw' in g: paw(c, g['path'][-1], g['paw'][0], g['paw'][1], pal, w, sd, g.get('toes', 1.), lit)
    for i, g in enumerate(spec.get('legs', ())):
        if not g.get('front'): leg(g, seed + 20 + i)
    if 'path' in spec:
        L, e = body(spec['path'], spec['width'], seed, spec.get('dens', 1.), spec.get('ln', 1.15), spec.get('spread', .6), f, spec.get('sway', .0)); fur_fill(c, L, pal, w, seed, key, e, .55)
    h = dict(spec['head']); hf = h.pop('parts', 'bf'); sp = h.pop('seed', seed)
    D = DogHead().pose(**h); m = spec.get('mane')
    if m: fur_fill(c, mane(D.cx, D.cy, m['rx'], m['ry'], m['n'], m['ln'], m['w'], seed + 5, m.get('a0', 0.), m.get('a1', 2 * math.pi), swirl=m.get('swirl', .25)), pal, w, seed + 6, key, None, .55)
    head(c, D, light, f, sp, w=w, key=key, parts=hf)
    for i, g in enumerate(spec.get('legs', ())):
        if g.get('front'): leg(g, seed + 40 + i)
    if not layer: return D
    lay = c.layer()
    lay = para(lay, (D.cx - 500 * lit * -1, D.cy - 400), (D.cx + 300 * lit * -1, D.cy + 800), [(0, '#f4f8ff'), (.45, '#ffffff'), (1, '#8a86c8')], 'mul', paramt)
    return rim2(lay, rimd[0], rimd[1], rimc, .9, .7)


SPEC = {
    # bursting up out of the pool, neck arched, head whipped round at us
    'erupt': dict(path=[(860, 1130), (930, 980), (960, 830), (900, 690), (860, 560), (930, 430)], width=[(0, 250), (.2, 205), (.45, 190), (.75, 165), (1, 120)], sway=.10, ln=1.35, spread=.75,
                  head=dict(cx=1000, cy=350, s=130, yaw=.60, pitch=.20, roll=-.16, jaw=.60, persp=.22), mane=dict(rx=150, ry=140, n=16, ln=250, w=80, swirl=.4, a0=1.5, a1=5.2)),
    # full lunge out of the pool
    'lunge': dict(path=[(420, 1160), (380, 980), (450, 800), (640, 660), (860, 600), (1010, 560)], width=[(0, 300), (.25, 225), (.5, 200), (.75, 185), (1, 140)], sway=.10, ln=1.3, spread=.7,
                  head=dict(cx=1110, cy=470, s=150, yaw=.52, pitch=.06, roll=.06, jaw=.75, persp=.25), mane=dict(rx=170, ry=160, n=18, ln=280, w=92, swirl=.4, a0=1.4, a1=5.3),
                  legs=[dict(path=[(960, 640), (1140, 690), (1300, 650), (1420, 700)], width=[(0, 110), (.5, 78), (1, 54)], paw=(98, .55)), dict(path=[(860, 720), (1010, 850), (1190, 860), (1330, 960)], width=[(0, 140), (.5, 95), (1, 66)], paw=(130, .75))]),
    # landing / gathering under itself: forepaws planted, haunches up behind, head low
    'gather': dict(path=[(1010, 190), (985, 260), (960, 340), (960, 430)], width=[(0, 90), (.35, 190), (.7, 215), (1, 170)], ln=1.2,
                   head=dict(cx=955, cy=500, s=150, yaw=-.10, pitch=-.14, roll=.08, jaw=.40, persp=.34), mane=dict(rx=190, ry=170, n=24, ln=300, w=100, swirl=.3),
                   legs=[dict(path=[(1120, 300), (1290, 400), (1240, 560), (1290, 680)], width=[(0, 110), (.4, 92), (1, 50)], paw=(86, 1.35)), dict(path=[(830, 300), (650, 400), (700, 560), (640, 680)], width=[(0, 110), (.4, 92), (1, 50)], paw=(86, 1.8)),
                         dict(path=[(1060, 520), (1130, 650), (1090, 770), (1120, 850)], width=[(0, 110), (.5, 92), (1, 80)], paw=(135, 1.45)), dict(path=[(850, 520), (770, 650), (800, 770), (770, 850)], width=[(0, 115), (.5, 96), (1, 84)], paw=(145, 1.72))]),
    # straight at the lens: paws huge and early, body tiny behind
    'leap': dict(path=[(1560, 150), (1430, 190), (1270, 260), (1080, 380)], width=[(0, 60), (.35, 120), (.7, 170), (1, 170)], ln=1.2,
                 head=dict(cx=960, cy=420, s=172, yaw=-.12, pitch=.02, roll=-.10, jaw=.85, persp=.36), mane=dict(rx=210, ry=190, n=26, ln=340, w=110, swirl=.3),
                 legs=[dict(path=[(1470, 200), (1560, 290), (1650, 270), (1730, 360)], width=[(0, 80), (.5, 46), (1, 30)], paw=(52, .8)), dict(path=[(1400, 260), (1440, 380), (1540, 420), (1600, 520)], width=[(0, 80), (.5, 52), (1, 34)], paw=(60, 1.0)),
                       dict(path=[(1060, 480), (1270, 520), (1430, 640), (1500, 800)], width=[(0, 120), (.5, 118), (1, 130)], paw=(220, 1.30)), dict(path=[(860, 480), (640, 490), (460, 600), (370, 770)], width=[(0, 120), (.5, 130), (1, 148)], paw=(255, 1.90))]),
}
