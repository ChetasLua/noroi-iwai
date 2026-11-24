"""SUKUNA lane: the Fuga key drawings (fire cel, torn black shirt, extreme low lens).  key(n) -> dict of RGBA layers back to front."""
import numpy as np, cv2, math
from jj import W, H, hands as HD
from jj.paint import rgb, f32, smooth
from jj.ink import Cel, spl, resample, noise1
from jj.comp import over
from jj.kit import para
from chars.sukuna import Suk, EXPR
from chars.sukuna_fire import FIRE, fire_hands, pen, rim
from chars.sukuna_hands import BANDS
from chars.sukuna_ink import stroke_poly, at, off

A = lambda *p: np.array(p, float)


def limb(c, P, W_, C, lit, line=3.2, seed=0, marks=(), split=.42, deep=False, lines=(), closed_end=False):
    """one muscle chain: centre line P (px points), half widths W_ (list of (left, right) along it, left of travel first).
    lit = unit screen vector TO the flame; the flank that faces it takes the light shape, broken by the muscle."""
    Q = spl(np.asarray(P, float), n=14) if len(P) > 2 else resample(np.asarray(P, float), 6.); n = len(Q); t = np.linspace(0, 1, n); k = np.linspace(0, 1, len(W_)); wl = np.interp(t, k, [w[0] for w in W_]); wr = np.interp(t, k, [w[1] for w in W_])
    ker = np.hanning(15); ker /= ker.sum(); sm = lambda v: np.convolve(np.pad(v, 7, 'edge'), ker, 'valid'); wl, wr = sm(wl), sm(wr)
    T = np.gradient(Q, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; N = np.c_[T[:, 1], -T[:, 0]]; L, R = Q + N * wl[:, None], Q - N * wr[:, None]; poly = np.vstack([L, R[::-1]])
    c.fill(poly, C['deep'] if deep else C['skin']); s = 1 if float(np.mean(N @ lit)) > 0 else -1; E, F = (L, R) if s > 0 else (R, L)
    c.clip(poly); r = np.random.default_rng(seed); fr = split + .16 * np.sin(t * math.pi * 2.3 + seed) - .10 * np.sin(t * math.pi * 5.1 + 1.3 * seed); z = E + (F - E) * np.clip(fr, .06, .8)[:, None]
    # angular terminator: straights with corners
    idx = np.unique(np.linspace(0, n - 1, 9 + seed % 3).astype(int)); c.fill(np.vstack([E, z[idx][::-1]]), C['skin'] if deep else C['light'])
    for (a, b, u0, u1) in lines:                           # muscle / tendon lines: along t a..b, across u (0 = lit edge .. 1)
        i0, i1 = int(a * (n - 1)), int(b * (n - 1)); m = (E + (F - E) * np.linspace(u0, u1, n)[:, None])[min(i0, i1):max(i0, i1) + 1]; pen(c, m[::max(1, len(m) // 10)], line * .55, C['line'], seed + i0, (.35, .5))
    for a, b in marks:                                      # tattoo rings
        i0, i1 = int(a * (n - 1)), int(b * (n - 1)); bul = T[(i0 + i1) // 2] * (wl[i0] * .45); e0 = spl(np.array([L[i0], Q[i0] + bul, R[i0]]), n=8); e1 = spl(np.array([L[i1], Q[i1] + bul, R[i1]]), n=8); c.fill(np.vstack([e0, e1[::-1]]), C['mark'])
    c.unclip()
    c.line(L, line, C['line'], taper=(.06, .06), press=.35, seed=seed, tremor=.25, tmin=.3); c.line(R, line, C['line'], taper=(.06, .06), press=.35, seed=seed + 1, tremor=.25, tmin=.3)
    return L, R


def tatter(c, P, C, seed=0, line=3., rag=12., step=24., lights=(), wind=(0, 0), ragged=None):
    """torn black cloth: authored polygon, the edges listed in `ragged` (index pairs into P) are ripped"""
    P = [np.asarray(p, float) for p in P]; r = np.random.default_rng(seed); out = []; m = len(P)
    for i in range(m):
        a, b = P[i], P[(i + 1) % m]
        if ragged and i in ragged:
            L = np.hypot(*(b - a)); k = max(3, int(L / step)); d = (b - a) / L; nn = A(-d[1], d[0]); out.append(a)
            for j in range(1, k): out.append(a + (b - a) * (j + r.normal(0, .2)) / k + nn * (rag * r.uniform(.2, 1.5) * (1 if j % 2 else -.35)) + A(*wind) * (.5 if j % 2 else 0))
        else: out.append(a)
    Q = np.array(out); c.fill(Q, C['cloth']); c.clip(Q)
    for L in lights: c.fill(spl(np.array(L, float), True, (0, len(L) // 2), n=6), C['cloth_l'])
    c.unclip(); c.line(np.vstack([Q, Q[:1]]), line, '#070504', taper=(.02, .02), press=.4, seed=seed, tremor=.2, tmin=.4); return Q


def hand(pose, cx, cy, s, side, R, dist, light, style=None, res=None, wrist=None, claw=.05, **k):
    p = dict(HD.POSES[pose]) if isinstance(pose, str) else dict(pose)
    if wrist is not None: p['wrist'] = wrist
    return HD.render([HD.Hand(p, side, R, (0, 0, 0), HD.HEAVY, nails='claw', claw=claw, bands=BANDS, **(style or {}))], HD.Cam(cx, cy, s, dist), light=light, res=res or int(np.clip(s * 2.4, 420, 980)), clip=(-200, -200, W + 200, H + 200), **k)


def hand_layer(g, lw, pal=None, lit=None):
    c = Cel(); fire_hands(c, g, lw, C=pal); L = c.layer()
    if lit is not None: L = rim(L, -lit[0] * 5 * lw, -lit[1] * 5 * lw, '#fff0a0', .9, .5)
    return L


D = math.radians
LIT = A(-.62, .78)
MIRR = np.diag([-1., 1., 1.])


def mir(R): return MIRR @ R @ MIRR                          # the same hand seen in a mirror (left <-> right)


class Form:
    """one continuous limb: centre line and the two half widths at each station; everything on it is authored in (t, u): t along, u across (+1 upper edge, -1 lower)"""
    def __init__(self, pts, up, dn):
        P = np.asarray(pts, float); self.Q = spl(P, n=16); n = len(self.Q); t = np.linspace(0, 1, n); k = np.linspace(0, 1, len(up)); ker = np.hanning(17); ker /= ker.sum(); sm = lambda v: np.convolve(np.pad(np.interp(t, k, v), 8, 'edge'), ker, 'valid'); self.up, self.dn = sm(up), sm(dn)
        T = np.gradient(self.Q, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; N = np.c_[T[:, 1], -T[:, 0]]; self.N = N * (1 if np.mean(N[:, 1]) < 0 else -1); self.T = T; self.n = n

    def at(self, t, u):
        t, u = np.broadcast_arrays(np.asarray(t, float), np.asarray(u, float)); x = np.clip(t, 0, 1) * (self.n - 1); i = np.clip(x.astype(int), 0, self.n - 2); f = (x - i)[..., None]; L = lambda v: v[i] * (1 - f) + v[i + 1] * f
        w = np.where(u[..., None] >= 0, L(self.up[:, None]), L(self.dn[:, None])); return L(self.Q) + L(self.N) * w * u[..., None]

    def edge(self, u, t0=0., t1=1., k=70): return self.at(np.linspace(t0, t1, k), u)
    def poly(self): return np.vstack([self.edge(1), self.edge(-1)[::-1]])
    def plane(self, L, n=5): q = spl(np.array(L, float), True, tuple(range(len(L))), n); return self.at(q[:, 0], q[:, 1])
    def curve(self, L, n=10): q = spl(np.array(L, float), n=n); return self.at(q[:, 0], q[:, 1])

    def draw(self, c, C, light, deep=(), kick=(), lines=(), rings=(), lw=4., seed=0, base='skin', open0=True, open1=True):
        P = self.poly(); c.fill(P, C[base]); c.clip(P)
        for L in deep: c.fill(self.plane(L), C['deep'])
        for L in light: c.fill(self.plane(L), C['light'])
        for L in kick: c.fill(self.plane(L), C['hot'])
        for a, b_ in rings:
            bow = .030; tt = np.linspace(-1, 1, 15); e0 = self.at(a + bow * (1 - tt ** 2), tt); e1 = self.at(b_ + bow * (1 - tt ** 2), tt); c.fill(np.vstack([e0, e1[::-1]]), C['mark'])
        c.unclip()
        for j, (L, w) in enumerate(lines): c.line(self.curve(L), w, C['line'], taper=(.28, .55), press=.3, seed=seed + j, tremor=.2, tmin=.12)
        for j, u in enumerate((1, -1)): c.line(self.edge(u), lw * (1. if u > 0 else 1.25), C['line'], taper=(.05, .03), press=.35, seed=seed + 50 + j, tremor=.25, tmin=.35)


def cap(c, F, C, bulge, lw, t_in=(.18, .97), light=.5, seed=3):
    """the inner end of a limb: a deltoid head that rolls into the body.  Its groove is drawn thin and only in part, so it joins instead of ending"""
    p1, p2 = F.at(0., .985), F.at(0., -.985); d = -F.T[0]; m = lambda k, b: p1 * (1 - k) + p2 * k + d * bulge * b; Q = spl(np.array([p1, m(.22, .72), m(.5, 1.), m(.80, .66), p2]), n=10); c.fill(np.vstack([Q, F.at(.03, np.linspace(-.985, .985, 12))]), C['skin']); n = len(Q)
    if light > 0: c.fill(np.vstack([Q[:int(n * light)], F.at(.06, np.linspace(1 - 2 * light, .985, 8))]), C['light'])
    i0, i1 = int(n * t_in[0]), int(n * t_in[1]); c.line(Q[i0:i1], lw, C['line'], taper=(.55, .12), press=.3, seed=seed, tremor=.2, tmin=.10); return Q


def rings_edges(g, y, hand_=0):
    """upper and lower contour points of the drawn forearm at distance y (hand units) below the wrist"""
    r = min(g.wrist[hand_], key=lambda q: abs(q['y'] - y)); w0 = g.wrist[hand_][0]['c']; ax = r['c'] - w0; ax = ax / (np.hypot(*ax) + 1e-6); n = A(-ax[1], ax[0]); n = n if n[1] < 0 else -n; d = (r['P'] - r['c']) @ n; return r['P'][int(np.argmax(d))], r['P'][int(np.argmin(d))], r['c'], ax


def cut_past(L, c, ax, h=H, w=W, soft=2.):
    """hand a drawn forearm over to the arm: erase the layer beyond a line through c (direction ax = away from the hand), over `soft` px"""
    y, x = np.mgrid[0:h, 0:w].astype(f32); m = np.clip(-((x - c[0]) * ax[0] + (y - c[1]) * ax[1]) / soft + .5, 0, 1); return L * m[..., None]


def torn_cloth(c, T, body, folds, lights, tongues, C, lw=3.6, seed=5):
    """heavy torn shirt: swept outline with a few long torn tongues, each with a dark underside and one lit lip; folds run from their tension points"""
    S = lambda L, cs=(), n=9: spl(T(np.array(L, float)), True, cs, n); P = S(*body); c.fill(P, C['cloth']); c.clip(P)
    for L, cs in lights: c.fill(S(L, cs, 7), C['cloth_l'])
    c.unclip()
    for j, (L, w) in enumerate(folds): c.line(spl(T(np.array(L, float)), n=12), w, '#050403', taper=(.10, .75), press=.45, seed=seed + j, tremor=.2, tmin=.06)
    for j, (r0, r1, tip, bend) in enumerate(tongues):
        a, b_, t = T(A(*r0)), T(A(*r1)), T(A(*tip)); m = (a + b_) / 2; d = t - m; n = A(-d[1], d[0]) * bend; L1 = spl(np.array([a, (a + t) / 2 + n, t]), n=10); L2 = spl(np.array([b_, (b_ + t) / 2 + n * 1.25 + d * .04, t]), n=10); c.fill(np.vstack([L1, L2[::-1]]), C['cloth']); c.fill(np.vstack([L1[3:], (L1 * .72 + L2 * .28)[4:][::-1]]), C['cloth_l'])
        for q in (L1, L2): c.line(q, lw * .8, '#050403', taper=(.02, .25), press=.4, seed=seed + 30 + j, tremor=.2, tmin=.2)
    c.line(np.vstack([P, P[:1]]), lw, '#050403', taper=(.02, .02), press=.45, seed=seed, tremor=.2, tmin=.35)


# the pose of key 4 is authored in its own tilt (about 17 degrees); keys 2 and 3 wind the same body back
SP = dict(index=(-4, 10, 6, -9, 0), middle=(-2, 8, 4, -1, 0), ring=(-12, 16, 10, 9, 0), little=(-14, 20, 14, 20, 0), thumb=(60, 18, 34, 2, -8))
KEYS = {
    2: dict(head=(1176, 388, 196, -.16, -.22, -.12), expr=dict(eo=(.80, .78), look=(-.5, -.45), mouth='grit', jaw=.02), rot=.17, sh=(-70, 40),
            lens=(600, 800, 200, 2.2, -2), rear=(1000, 700, 138, 4.4, -62), shL=(1050, 672), shD=(1520, 610), elD=(1470, 930), wD=((100, 92), (84, 78), (58, 58)), mode='under'),
    3: dict(head=(1204, 348, 202, -.26, -.17, -.22), expr=dict(eo=(.92, .88), look=(-.4, -.2), jaw=.03), rot=.08, sh=(-34, 18),
            lens=(510, 780, 275, 2.0, -5), rear=(1196, 600, 146, 4.4, -50), shL=(1072, 640), shD=(1600, 520), elD=(1640, 560), wD=((104, 96), (88, 82), (62, 62)), mode='under'),
    4: dict(head=(1226, 312, 208, -.32, -.15, -.31), expr=dict(eo=(1.0, .94), look=(-.35, -.15), jaw=.05), rot=0., sh=(0, 0),
            lens=(424, 766, 352, 1.82, -8), rear=(1400, 552, 152, 4.4, 4), shL=(1096, 606), shD=(1656, 566), elD=None, wD=((108, 100), (92, 86), (66, 66)), mode='half'),
}


def key(n, seed=0, j=0., C=None):
    """one of the three wide draw keys -> ordered dict of RGBA layers.  j = tremble px (seed changes the hand)"""
    K = KEYS[n]; C = dict(FIRE, **(C or {})); r = np.random.default_rng(41 + seed); J = lambda p: np.asarray(p, float) + (r.normal(0, 1, np.shape(p)) * j if j else 0); lit = LIT / np.hypot(*LIT); out = {}
    piv = A(1560, 1000); cr, sr = math.cos(K['rot']), math.sin(K['rot'])
    def T(p): q = np.asarray(p, float) - piv; return np.stack([q[..., 0] * cr - q[..., 1] * sr, q[..., 0] * sr + q[..., 1] * cr], -1) + piv + A(*K['sh'])
    S = lambda L, cs=(), m=8: spl(T(np.array(L, float)), True, cs, m); ink = lambda c, L, w, s_=0, tp=(.25, .5): c.line(spl(T(np.array(L, float)), n=10), w, C['line'], taper=tp, press=.35, seed=s_, tremor=.2, tmin=.12)
    shL, shD = J(K['shL']), J(K['shD'])
    # ------------------------------------------------ hands first: their forearms tell the arms where to go
    lx, ly, ls_, ld, la = K['lens']; RL = HD.Rz(D(180 - la)) @ mir(HD.Rz(D(58)) @ HD.euler(166, 66, 26)); gl = hand(SP, lx + r.normal(0, j * 1.6), ly + r.normal(0, j * 1.6), ls_, 'R', RL, ld, (-.30, .62, .30), wrist=(34, -22), claw=.03, thr=.30)
    rx, ry, rs_, rd, ra = K['rear']; RR = HD.Rz(D(ra)) @ mir(HD.Rz(D(-44)) @ HD.euler(58, -4, -80)); gr = hand('rear', rx + r.normal(0, j), ry + r.normal(0, j), rs_, 'L', RR, rd, (-.62, -.30, .55), claw=.04, thr=.22)
    # ------------------------------------------------ drawing arm: shoulder -> raised elbow -> forearm back to the cheek
    el = gr.wrist[0][-2]['c'] if K['elD'] is None or True else A(*K['elD']); eu, ed, ec, eax = rings_edges(gr, 2.0); c = Cel(); (a0, a1), (b0, b1), (c0, c1) = K['wD']; mid = (shD + el) / 2 + A(46, 10)
    inw = A(1430, 640) - shD; inw = inw / np.hypot(*inw)
    FD = Form([shD + inw * 62, shD + inw * 8, shD * .64 + el * .36 + A(30, 8), mid * .35 + el * .65 + A(26, 4), el + eax * 10], (a0 * .80, a0 * 1.08, b0, c0 + 8, c0), (a1 * .80, a1 * 1.10, b1, c1 + 6, c1))
    cap(c, FD, C, 44, 3.2, seed=12, light=.42)
    FD.draw(c, C, [[(.0, .97), (.30, .98), (.40, .50), (.30, .10), (.10, .16)], [(.46, .97), (.74, .97), (.78, .40), (.60, .16), (.50, .24)]], deep=[[(0, -.98), (1, -.98), (1, -.46), (.76, -.54), (.54, -.34), (.34, -.56), (.12, -.40)]], lines=[([(.38, .92), (.46, .36), (.42, -.20), (.32, -.50)], 3.6), ([(.46, -.04), (.62, -.12), (.82, .04)], 2.8), ([(.12, .62), (.22, .44), (.32, .52)], 2.4)], rings=((.52, .575), (.63, .685)), lw=4.4, seed=11)
    # torn armhole: the shirt lies on the root of the deltoid
    Qh = spl(np.array([FD.at(.02, 1.10), FD.at(-.0, .5) + inw * 30, FD.at(0., 0.) + inw * 46, FD.at(0., -.5) + inw * 30, FD.at(.03, -1.12)]), n=14); rg = np.random.default_rng(31); n_ = len(Qh); jag = (10 + 9 * np.abs(np.sin(np.arange(n_) * .55 + 1.)) ** 2 + 5 * noise1(n_, 4, rg)); c.fill(np.vstack([Qh + inw * 40, (Qh - inw * jag[:, None])[::-1]]), C['cloth']); c.line(Qh - inw * jag[:, None], 3.2, '#050403', taper=(.05, .05), press=.4, seed=6); c.line(Qh + inw * 4, 3.4, C['cloth_l'], taper=(.3, .4), press=.5, seed=7)
    out['darm'] = rim(c.layer(), 5, 4, '#fff0a0', .8, .5)
    # ------------------------------------------------ torso: chest toward us, twisted; lens-side pectoral, serratus, ribs and lat are drawn planes
    c = Cel(); P = S([(1030, 600), (1110, 500), (1230, 470), (1400, 450), (1560, 400), (1660, 470), (1730, 600), (1800, 760), (1860, 930), (1900, 1130), (1340, 1130), (1300, 1040), (1228, 920), (1150, 800), (1080, 700)], (9, 10)); c.fill(P, C['skin']); c.clip(P)
    for L, cs, k in (([(1046, 640), (1120, 720), (1200, 772), (1300, 770), (1398, 716), (1380, 640), (1290, 580), (1190, 560), (1100, 580)], (0, 3, 4), 'light'), ([(1400, 716), (1480, 742), (1570, 716), (1640, 640), (1600, 560), (1500, 560), (1420, 610)], (0, 2, 3), 'light'),      # pectorals
                     ([(1098, 722), (1150, 800), (1196, 876), (1222, 860), (1190, 800), (1140, 742)], (0, 2), 'light'),                                                                                                                                 # lat edge catching the arrow
                     ([(1212, 800), (1262, 828), (1300, 872), (1254, 862)], (0, 2), 'light'), ([(1250, 890), (1306, 916), (1340, 966), (1290, 950)], (0, 2), 'light'), ([(1300, 978), (1350, 1000), (1376, 1050), (1330, 1040)], (0, 2), 'light'),                    # serratus steps
                     ([(1372, 800), (1440, 820), (1452, 900), (1400, 910)], (0, 2), 'light'), ([(1404, 930), (1466, 940), (1480, 1020), (1426, 1026)], (0, 2), 'light'),                                                                            # abdominals
                     ([(1060, 690), (1140, 800), (1222, 930), (1290, 1040), (1330, 1130), (1240, 1130), (1140, 900)], (0, 4), 'deep'), ([(1200, 790), (1310, 790), (1400, 730), (1394, 780), (1320, 826), (1240, 820)], (0, 2), 'deep'), ([(1180, 470), (1290, 560), (1400, 590), (1500, 540), (1560, 430), (1400, 470)], (0, 4), 'deep')):
        c.fill(S(L, cs, 6), C[k])
    c.unclip()
    for L, w in (([(1046, 648), (1126, 728), (1208, 780), (1310, 776), (1400, 716)], 3.6), ([(1400, 716), (1480, 748), (1576, 722), (1650, 640)], 3.2), ([(1404, 600), (1402, 680), (1416, 800), (1452, 960)], 2.6), ([(1150, 800), (1206, 812), (1262, 828)], 2.6), ([(1204, 890), (1256, 898), (1306, 916)], 2.6), ([(1262, 978), (1310, 984), (1352, 1002)], 2.4), ([(1372, 800), (1430, 812), (1450, 806)], 2.2), ([(1400, 918), (1456, 928), (1478, 922)], 2.2), ([(1150, 540), (1240, 590), (1340, 604), (1398, 596)], 3.0), ([(1400, 594), (1480, 580), (1560, 520)], 2.8)): ink(c, L, w, int(w * 10))
    ink(c, [(1030, 600), (1080, 700), (1150, 800), (1228, 920), (1300, 1040), (1340, 1130)], 4.6, 3, (.06, .06)); out['skin'] = rim(c.layer(), -lit[0] * 7, -lit[1] * 7, '#ffe98a', .9, .5)
    c = Cel(); c.fill(S([(1150, 500), (1230, 440), (1400, 430), (1440, 520), (1420, 600), (1300, 640), (1180, 600)], ()), C['deep']); ink(c, [(1200, 520), (1230, 570), (1290, 600)], 2.6, 2); ink(c, [(1400, 520), (1380, 560), (1330, 600)], 2.6, 4); out['neck'] = c.layer()
    # ------------------------------------------------ trapezius masses rise beside the jaw, head sunk between them
    c = Cel()
    for L, li in (([(1096, 560), (1140, 506), (1200, 506), (1256, 540), (1290, 596), (1230, 610), (1150, 600)], [(1100, 560), (1146, 520), (1200, 520), (1240, 548), (1190, 566), (1130, 572)]), ([(1370, 560), (1420, 500), (1490, 440), (1570, 410), (1600, 470), (1530, 540), (1440, 590)], [(1400, 552), (1450, 500), (1516, 456), (1500, 510), (1440, 560)])): c.fill(S(L, (0, 4), 7), C['skin']); c.fill(S(li, (0, 3), 6), C['light']); ink(c, L[:4] if L[0][0] < 1200 else L[:4], 3.6, 5, (.15, .3))
    out['traps'] = c.layer()
    hx, hy, hs, yw, pt, rl = K['head']; e = dict(EXPR['strain'], e2=(1., 1.), **K['expr']); k = Suk(hx + r.normal(0, j * .5), hy + r.normal(0, j * .5), hs, yw, pt, rl, e, seed=4, neck_len=1.5)
    if K['mode'] == 'half': from chars.sukuna_fire import BURN; FL = k.fire(ls=-1, light_vec=lit, mode='half', pal=BURN, rim_col='#ffffff', head_only=True); out['head'] = para(FL, (hx + 170, hy - 260), (hx - 170, hy + 230), [(0, '#8a5a44'), (.5, '#fff2e0'), (1, '#ffffff')])
    else: FL = k.fire(ls=-1, light_vec=lit, head_only=True); out['head'] = para(FL, (hx + 130, hy - 220), (hx - 130, hy + 200), [(0, '#a08877'), (.55, '#f6ece0'), (1, '#ffffff')])
    # ------------------------------------------------ what is left of the shirt: hangs from the drawing shoulder, wrenched across the chest
    c = Cel(); torn_cloth(c, T, ([(1500, 470), (1576, 404), (1664, 470), (1734, 600), (1806, 760), (1866, 930), (1906, 1140), (1440, 1140), (1470, 1040), (1452, 950), (1486, 860), (1456, 770), (1500, 690), (1470, 600), (1488, 530)], (1, 6, 7)),
        [([(1580, 420), (1600, 560), (1580, 720), (1556, 860)], 7.), ([(1640, 470), (1680, 640), (1680, 820), (1650, 1010)], 8.), ([(1720, 600), (1760, 780), (1766, 960), (1740, 1130)], 7.), ([(1500, 690), (1560, 720), (1610, 790)], 5.), ([(1486, 860), (1560, 880), (1620, 940), (1650, 1030)], 6.), ([(1470, 1040), (1540, 1050), (1600, 1110)], 5.)],
        [([(1596, 440), (1640, 560), (1650, 700), (1610, 820), (1604, 680), (1590, 560)], (0, 3)), ([(1690, 600), (1724, 740), (1726, 900), (1690, 1060), (1690, 880), (1684, 740)], (0, 3)), ([(1506, 560), (1550, 620), (1560, 700), (1520, 660)], (0, 2)), ([(1510, 800), (1570, 830), (1590, 900), (1530, 880)], (0, 2)), ([(1780, 800), (1830, 920), (1850, 1060), (1800, 960)], (0, 2))],
        [((1470, 600), (1500, 690), (1398, 640), -.10), ((1456, 770), (1486, 850), (1352, 846), .12), ((1452, 950), (1470, 1040), (1370, 1040), -.10), ((1664, 470), (1720, 570), (1846, 452), .16), ((1780, 700), (1830, 820), (1990, 700), .14), ((1500, 476), (1560, 420), (1470, 420), -.12)], C, 3.8, 9)
    out['shirt'] = para(rim(c.layer(), -lit[0] * 8, -lit[1] * 8, '#ffd45a', .9, .5), (1400, 700), (1880, 900), [(0, '#ffffff'), (1, '#a8826a')])
    RL_ = hand_layer(gr, 1.35, lit=lit); c = Cel(); u0, d0, c0_, ax_ = rings_edges(gr, 1.6); tip = ec + eax * (np.hypot(*(eu - ed)) * .56); nrm = (eu - ed) / (np.hypot(*(eu - ed)) + 1e-6); w_ = np.hypot(*(eu - ed)) / 2
    c.clip(spl(np.array([u0, eu, tip + nrm * w_ * .55, tip, tip - nrm * w_ * .62, ed, d0]), True, n=6)); c.fill(spl(np.array([d0, c0_ * .55 + d0 * .45, ec * .72 + ed * .28, tip + nrm * w_ * .30, tip + eax * 30, ed + eax * 40, d0 - nrm * 20]), True, (0, 2, 3), n=6), C['skin']); c.unclip()
    c.line(spl(np.array([ec + nrm * w_ * .55 - eax * w_ * .5, ec + nrm * w_ * .20, ec + eax * w_ * .10 - nrm * w_ * .25]), n=8), 3.0, C['line'], taper=(.3, .5), press=.3, seed=4); c.line(spl(np.array([ec + eax * w_ * .42 + nrm * w_ * .40, ec + eax * w_ * .52, ec + eax * w_ * .40 - nrm * w_ * .35]), n=8), 2.6, C['line'], taper=(.3, .5), press=.3, seed=5); q = c.layer(); RL_ = RL_ * (1 - q[..., 3:4]) + q * (RL_[..., 3:4] > .5)
    out['rear'] = RL_; out['g_rear'] = gr
    # ------------------------------------------------ lens arm: deltoid, biceps plane, elbow, forearm, then the hand: one form that takes over the drawn forearm half way down
    fu, fd, fc, fax = rings_edges(gl, 1.2); LH = cut_past(hand_layer(gl, .9 + ls_ / 270., lit=A(.1, -1.)), fc, fax, soft=11.); hu, hd = np.hypot(*(fu - fc)), np.hypot(*(fd - fc)); hw = (hu + hd) / 2; dl = shL - fc; Ln = np.hypot(*dl); bw = float(np.clip(Ln * .155, 62, 118))
    F = Form([shL + A(104, 12), shL + A(60, 4), shL + A(0, -2), fc + dl * .66 + A(0, -10), fc + dl * .44 + A(0, 2), fc + dl * .20 + A(0, 8), fc, fc - fax * 16], (bw * .84, bw * 1.10, bw * 1.18, bw * .98, bw * .86, (bw * .70 + hu) / 2, hu, hu), (bw * .80, bw * 1.04, bw * 1.10, bw * 1.00, bw * .84, (bw * .86 + hd) / 2, hd, hd))
    c = Cel(); arm = F.at(.04, -.96)
    c.fill(spl(np.array([F.at(.30, -.90), F.at(.10, -.92), arm + A(10, 6), T(A(1190, 770)), T(A(1130, 760)), T(A(1086, 706))]), True, (0, 3), n=7), C['deep'])          # the hollow of the armpit
    cap(c, F, C, 46, 3.4, t_in=(.34, .98), light=.46, seed=15)
    F.draw(c, C, [[(.00, .98), (.40, .99), (.43, .46), (.35, .10), (.22, .02), (.10, .10), (.0, .0)], [(.44, .99), (.64, .99), (.67, .44), (.58, .14), (.49, .20)], [(.70, .99), (1., .99), (1., .02), (.90, -.04), (.77, .16)]], deep=[[(.04, -.99), (.90, -.99), (.83, -.62), (.71, -.64), (.60, -.40), (.46, -.62), (.30, -.44), (.12, -.50)]], kick=[[(.16, .99), (.33, .99), (.29, .80), (.19, .78)], [(.48, .99), (.60, .99), (.57, .82), (.51, .80)]],
           lines=[([(.39, .92), (.44, .40), (.41, -.10), (.34, -.34)], 4.2), ([(.43, -.12), (.55, -.20), (.66, -.08)], 3.4), ([(.665, .50), (.695, .16), (.685, -.14)], 3.2), ([(.73, .30), (.83, .16), (.94, .20)], 2.8), ([(.16, .46), (.25, .32), (.33, .38)], 2.4), ([(.74, -.34), (.84, -.28), (.95, -.36)], 2.4)], rings=((.49, .535), (.575, .62)), lw=5.0, seed=8)
    # trapezius and collar bone run down INTO the deltoid, the pectoral folds into the armpit
    acr = F.at(.20, .97); n0, n1 = T(A(1252, 560)), T(A(1300, 612)); tm = (n0 + acr) / 2 + A(4, -16); TZ = spl(np.array([n0, tm, acr]), n=10); c.fill(np.vstack([TZ, F.at(.10, np.linspace(.96, .30, 6)), spl(np.array([F.at(.02, .22), T(A(1216, 606)), n1]), n=8)]), C['skin']); c.fill(np.vstack([TZ, F.at(.12, np.linspace(.96, .66, 4)), spl(np.array([F.at(.06, .62), (n0 + F.at(.06, .5)) / 2 + A(0, 12), n0 + A(4, 14)]), n=6)]), C['light'])
    c.line(TZ, 4.2, C['line'], taper=(.12, .30), press=.3, seed=21, tremor=.2, tmin=.2); c.line(spl(np.array([n1, T(A(1230, 600)), F.at(.04, .34)]), n=10), 3.2, C['line'], taper=(.15, .55), press=.35, seed=22); c.line(spl(np.array([n1 + A(-6, 12), T(A(1236, 618)), F.at(.06, .16)]), n=10), 1.9, C['line'], taper=(.3, .6), seed=23)
    c.line(spl(np.array([F.at(.16, -.80), arm + A(16, 2), T(A(1150, 742)), T(A(1226, 782))]), n=12), 3.8, C['line'], taper=(.30, .45), press=.35, seed=24, tremor=.2, tmin=.12)
    out['narm'] = rim(c.layer(), 0, 6, '#fff6d0', .7, .5); out['lens'] = LH; out['g_lens'] = gl
    return out


def stack(out, names=None):
    L = np.zeros((H, W, 4), f32)
    for n in names or [k for k in out if not k.startswith('g_')]: q = out[n]; L = L * (1 - q[..., 3:4]) + q
    return L


def k2(): return key(2)
def k3(): return key(3)
def k4(): return key(4)
