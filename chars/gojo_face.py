"""Gojo's bare eyes, lashes, brows and face shadows, authored in head-front coordinates (x right, y up, unit = chin to eye)."""
import numpy as np, math, cv2
from jj.ink import spl, resample, noise1
from jj.kit import outline
from . import gojo_eye as GE

ES, EY = 1.225, -.030
EC = .310                                             # eye centre x


def _lw(H, w, lo=.55):
    return w * max(lo, (H.s / 260.) ** .55)


# lid curves relative to the eye centre, for the eye on +x (inner corner at negative u).  6 upper + corners, 4 lower
LIDS = {
    'open':  dict(i=(-.168, -.040), up=[(-.135, .018), (-.070, .070), (.005, .089), (.085, .076), (.150, .040)], o=(.196, .004), lo=[(.150, -.036), (.080, -.062), (.0, -.070), (-.090, -.062)]),
    'cold':  dict(i=(-.170, -.050), up=[(-.135, -.004), (-.075, .036), (.0, .058), (.085, .062), (.155, .042)], o=(.200, .014), lo=[(.152, -.030), (.080, -.060), (.0, -.072), (-.095, -.070)]),
    'calm':  dict(i=(-.168, -.044), up=[(-.135, .008), (-.072, .054), (.005, .074), (.085, .068), (.152, .040)], o=(.197, .008), lo=[(.150, -.034), (.080, -.061), (.0, -.071), (-.092, -.065)]),
    'sad':   dict(i=(-.165, -.036), up=[(-.130, .012), (-.065, .050), (.005, .058), (.085, .040), (.148, .004)], o=(.188, -.026), lo=[(.145, -.050), (.075, -.066), (.0, -.070), (-.090, -.060)]),
    'wide':  dict(i=(-.172, -.048), up=[(-.145, .036), (-.080, .098), (.0, .118), (.085, .106), (.155, .060)], o=(.202, .014), lo=[(.158, -.042), (.085, -.078), (.0, -.090), (-.095, -.078)]),
    'laugh': dict(i=(-.165, -.020), up=[(-.120, .020), (-.060, .052), (.005, .064), (.085, .052), (.145, .020)], o=(.190, -.014), lo=[(.145, .006), (.075, .030), (.0, .038), (-.085, .020)]),
    'shut':  dict(i=(-.165, -.044), up=[(-.130, -.058), (-.065, -.072), (.005, -.074), (.085, -.062), (.148, -.036)], o=(.192, -.016), lo=[(.148, -.037), (.085, -.063), (.005, -.075), (-.085, -.064)]),
}
BROWS = {
    'open': [(-.190, .078), (-.085, .112), (.03, .138), (.125, .140), (.222, .098)],
    'calm': [(-.192, .064), (-.085, .096), (.03, .124), (.125, .132), (.224, .098)],
    'cold': [(-.200, .024), (-.090, .062), (.025, .104), (.125, .126), (.230, .104)],
    'sad':  [(-.178, .118), (-.085, .130), (.03, .134), (.125, .114), (.214, .068)],
    'wide': [(-.200, .070), (-.090, .120), (.025, .168), (.125, .182), (.228, .138)],
    'laugh': [(-.182, .092), (-.085, .138), (.03, .162), (.125, .154), (.214, .104)],
}


def blend_lids(a, b, t):
    A, B = LIDS[a] if isinstance(a, str) else a, LIDS[b] if isinstance(b, str) else b
    return {k: (np.array(A[k], float) * (1 - t) + np.array(B[k], float) * t) for k in A}


def eye(cel, H, side, pal, lid='open', brow=None, openk=1., gaze=(0., 0.), iris=.083, pupil=.30, glow=0., seed=0, spin=0., detail=None, shade=True, look=None):
    """one eye.  lid = LIDS key or dict; openk 1 open .. 0 shut; gaze in head units (picture right, up)."""
    sg = side; L = blend_lids('shut', lid, openk) if openk < 1 else {k: np.array(v, float) for k, v in (LIDS[lid] if isinstance(lid, str) else lid).items()}
    def Q(a, dz=.0):
        a = np.atleast_2d(np.asarray(a, float)); return H.P(np.c_[sg * (EC + a[:, 0] * ES), a[:, 1] * ES + EY], dz=dz)
    if H.facing(sg * EC, 0.) < .04: return
    s = H.s; det = np.clip((s - 60) / 420., .10, 1.) if detail is None else detail
    cor = (0, len(L['up']) + 1)
    U = spl(np.vstack([[L['i']], L['up'], [L['o']]]), n=12); LO = spl(np.vstack([[L['o']], L['lo'], [L['i']]]), n=12)
    shape_ = np.vstack([U, LO[1:]])
    E = Q(shape_)
    ln = pal['line']
    # --- socket shadow: the plane between brow and lid, a hard wedge at the inner corner
    if openk > .05 and shade:
        w = np.array([L['i'] + (-.035, .012), L['up'][0] + (-.05, .055), L['up'][1] + (-.04, .075), L['up'][2] + (-.02, .052), L['up'][2] + (.0, .020), L['up'][1] + (.0, .010), L['up'][0] + (.0, .008), L['i']])
        cel.fill(Q(spl(w, True, corners=(0, 2, 3, 7), n=5)), pal['skin_sh'])
    if openk > .06:
        cel.fill(E, pal['white'])
        cel.clip(E)
        # iris on the eyeball
        g = np.asarray(gaze, float); cx, cy = sg * EC + g[0] - sg * .012, .006 + g[1] + EY; iris = iris * ES
        c = H.P([(cx, cy)])[0]; e = .02
        ex = (H.P([(cx + e, cy)])[0] - H.P([(cx - e, cy)])[0]) / (2 * e) * iris; ey = (H.P([(cx, cy - e)])[0] - H.P([(cx, cy + e)])[0]) / (2 * e) * iris * 1.06
        if look is not None: ex = ex * look
        GE.iris(cel, c, ex, ey, pupil=pupil, seed=7 if sg > 0 else 12, detail=det, par=(g[0] * 1.2, -g[1] * 1.2), glow=glow, spin=spin, pal=pal.get('iris_pal', GE.IR), shade_top=.8)
        # cast shadow of lid and lashes on the ball (hard)
        dn = U.copy(); dn[:, 1] -= .030 * np.sin(np.linspace(.15, np.pi - .15, len(U))) + .004
        cel.fill(Q(np.vstack([U, dn[::-1]])), pal['white_sh'], .78)
        # wet corner
        cel.fill(Q(spl([L['i'], L['i'] + (.030, .018), L['i'] + (.034, -.006), L['i'] + (.012, -.014)], True, n=4)), pal['lip'], .55)
        GE.iris_hi(cel, c, ex, ey, side=-1, detail=det)
        cel.unclip()
    # --- lower lid: broken thin line + its lash spikes
    lo_s = LO[:int(len(LO) * .74)]
    cel.line(Q(lo_s), _lw(H, 1.7), pal['lid'], taper=(.1, .7), seed=seed + 41, tmin=.05, alpha=.9)
    n = len(LO); r = np.random.default_rng(seed * 7 + 91 + (sg > 0))
    if openk > .25 or lid == 'laugh':
        for k, (u, ll, aa) in enumerate(((.07, .040, .75), (.19, .036, .50), (.31, .028, .32), (.45, .020, .15))):
            p = LO[int(u * (n - 1))]; tip = p + np.array([math.sin(aa), -math.cos(aa)]) * ll; wd = .010
            t_ = LO[min(n - 1, int(u * (n - 1)) + 2)] - p; t_ = t_ / (np.hypot(*t_) + 1e-6)
            q = np.array([p - t_ * wd, (p + tip) / 2 - t_ * wd * .25, tip, (p + tip) / 2 + t_ * wd * .35, p + t_ * wd])
            cel.fill(Q(spl(q, True, corners=(0, 2, 4), n=4)), pal['lash'])
            cel.line(Q(spl(q[:3], n=5)), _lw(H, 1.0), pal['lash_ln'], taper=(.5, .1), seed=seed + 60 + k); cel.line(Q(spl(q[2:], n=5)), _lw(H, 1.0), pal['lash_ln'], taper=(.1, .5), seed=seed + 70 + k)
    # --- upper lid: dark line with weight, winged outside
    m = len(U); t = np.linspace(0, 1, m)
    th = np.interp(t, [0, .10, .35, .65, .88, 1], [.004, .014, .023, .027, .030, .012]) * (.62 + .38 * openk)
    nr = np.gradient(U, axis=0); nr = np.c_[-nr[:, 1], nr[:, 0]]; nr /= np.hypot(nr[:, 0], nr[:, 1])[:, None]; nr *= np.sign(nr[m // 2, 1])
    wing = L['o'] + np.array([.030, .012 if lid != 'sad' else -.012])
    lidp = np.vstack([U, [wing], (U + nr * th[:, None])[::-1]])
    # --- white lashes: one feathered shape riding the lid (big clumps outside, almost nothing inside), a shaded one behind it
    k_ = .50 + .50 * openk
    tgt = lambda i: np.array([nr[i][1], -nr[i][0]])
    def at(u, h):
        uu = min(max(u, 0.), 1.); i = min(m - 1, int(uu * (m - 1))); hh = h * k_ + .006
        if openk < .45: hh = hh * np.interp(openk, [0, .2, .45], [-1.15, -.6, 1.])        # the lashes of a shut eye hang down
        p = U[i] + nr[i] * (th[i] * .55 + hh)
        if u > 1: p = p + tgt(m - 1) * (u - 1) * .36
        return p
    def feather(specs, fill, lnw, sd, hk=1., du=0.):
        pts, cor = [], []
        for q, (uv, hv, ut, ht) in enumerate(specs):
            nx = specs[q + 1] if q + 1 < len(specs) else (1.0, .004, 0, 0)
            v0, tp, v1 = at(uv + du, hv * hk), at(ut + du, ht * hk), at(nx[0] + du, nx[1] * hk)
            pts.append(v0); cor.append(len(pts) - 1)
            pts.append(v0 * .45 + tp * .55 + nr[min(m - 1, int(min(ut, 1) * (m - 1)))] * ht * .07 * hk)
            pts.append(tp); cor.append(len(pts) - 1)
            pts.append(v1 * .58 + tp * .42 - nr[min(m - 1, int(min(ut, 1) * (m - 1)))] * ht * .10 * hk)
        pts.append(wing + np.array([-.012, -.006])); cor.append(len(pts) - 1)
        edge = spl(np.array(pts), corners=cor, n=5)
        P = Q(np.vstack([edge, U[::-1]]), .012); cel.fill(P, fill)
        if lnw: outline(cel, Q(edge, .012), _lw(H, lnw), pal['lash_ln'], seed=sd, closed=False, press=.3, tremor=.1)
    FE = [(.02, .006, .22, .018), (.20, .016, .44, .026), (.40, .021, .64, .032), (.60, .025, .84, .044), (.77, .028, 1.03, .066), (.91, .028, 1.22, .080)]
    BK = [(.10, .006, .36, .026), (.34, .020, .56, .036), (.52, .024, .76, .044), (.70, .028, .95, .064), (.85, .030, 1.12, .088)]
    feather(BK, pal['lash_sh'], .7, seed + 100)
    feather(FE, pal['lash'], 1.25, seed + 130)
    cel.fill(Q(lidp), pal['lid'])
    # a few long single hairs
    for k, (u, ht) in enumerate(((.80, .070), (.92, .092))):
        a, b = at(u, .012), at(u + .13, ht); cel.line(Q(np.array([a, (a + b) / 2 + nr[min(m - 1, int(u * m))] * .010, b]), .012), _lw(H, 1.2), pal['lash'], taper=(.1, .8), seed=seed + 160 + k, tmin=.05)
    # crease
    if openk > .3:
        cr = U[int(m * .18):int(m * .92)] + nr[int(m * .18):int(m * .92)] * np.interp(np.linspace(0, 1, int(m * .92) - int(m * .18)), [0, .5, 1], [.040, .064, .050])[:, None]
        cel.line(Q(cr), _lw(H, 1.5), pal['skin_dp'], taper=(.5, .35), seed=seed + 50, tmin=.05)
    # --- brow
    if brow is not None:
        b = np.array(BROWS[brow] if isinstance(brow, str) else brow, float); C = spl(b, n=10); tt = np.linspace(0, 1, len(C))
        N = np.gradient(C, axis=0); N = np.c_[-N[:, 1], N[:, 0]]; N /= np.hypot(N[:, 0], N[:, 1])[:, None]; N *= np.sign(N[len(N) // 2, 1])
        wd = np.interp(tt, [0, .08, .35, .7, 1], [.014, .032, .036, .024, .0])
        up, dn = C + N * (wd * .5)[:, None], C - N * (wd * .5)[:, None]
        # ragged root end: two hairs
        P = np.vstack([up, dn[::-1]])
        cel.fill(Q(P), pal['brow'])
        cel.line(Q(up), _lw(H, 1.4), pal['brow_ln'], taper=(.2, .15), seed=seed + 51); cel.line(Q(dn), _lw(H, 1.9), pal['brow_ln'], taper=(.12, .12), seed=seed + 52)
        for k, (a, bq) in enumerate(((.10, .26),)):
            i0, i1 = int(a * len(C)), int(bq * len(C)); cel.line(Q(np.array([dn[i0], C[(i0 + i1) // 2], up[i1]])), _lw(H, .9), pal['brow_ln'], taper=(.4, .6), seed=seed + 53 + k, alpha=.7)


def face_shade(cel, H, pal, key='L', lit=None):
    """one hard cel shadow on the face, inside the face clip.  key: 'L' / 'R' side light, 'BL' / 'BR' orb from below at that
    side, 'T' top light.  lit = colour of the small reflected facet on the dark side (None = none)."""
    if not key: return
    m = -1. if key in ('R', 'BR') else 1.; sh = pal['skin_sh']
    def poly(pts, col, corners=None):
        a = np.asarray(pts, float) * (m, 1); a = a if corners is None else spl(a, True, corners=corners, n=6); q = H.P(a)
        far = np.abs(a[:, 0]) >= .59                       # the outer boundary runs past the silhouette (the face clip trims it)
        q[far] += np.sign(a[far, 0])[:, None] * np.array([math.cos(H.roll), math.sin(H.roll)]) * H.s * .9
        cel.fill(q, col)
    if key in ('L', 'R'):
        poly([(.40, .74), (.47, .44), (.50, .20), (.575, .12), (.53, .03), (.555, -.10), (.50, -.22), (.41, -.31), (.375, -.46), (.285, -.60), (.245, -.70), (.205, -.79), (.10, -.90), (.06, -1.02), (.6, -1.02), (.78, -.4), (.78, .74)], sh)
        poly([(.020, .07), (.040, -.10), (.066, -.24), (.094, -.35), (.084, -.398), (.038, -.410), (.012, -.38), (.020, -.26), (.008, -.12)], sh)
    elif key in ('BL', 'BR'):
        # everything that faces up and away from the orb is dark; the terminator runs brow - nose bridge - mouth - chin
        poly([(-.70, 1.0), (-.64, .34), (-.52, .22), (-.36, .205), (-.20, .155), (-.115, .085), (-.055, .02), (-.020, -.12), (.014, -.26), (.056, -.335), (.092, -.378), (.068, -.418), (.030, -.46), (.052, -.60), (.135, -.655), (.185, -.71), (.125, -.79), (.145, -.88), (.10, -.95), (.07, -1.02), (.8, -1.02), (.8, 1.0)], sh)
        poly([(.205, -.155), (.30, -.175), (.43, -.150), (.385, -.26), (.315, -.40), (.255, -.30)], lit or pal['skin'])
        poly([(-.50, .165), (-.36, .178), (-.22, .125), (-.18, .075), (-.24, .105), (-.36, .140), (-.47, .115)], sh)
    elif key == 'B':
        # orb straight below: forehead, temples, the top of the nose and the lids' upper planes fall dark
        poly([(-.80, 1.0), (-.80, .10), (-.63, .02), (-.56, .12), (-.46, .185), (-.30, .175), (-.16, .115), (-.085, .075), (-.045, .00), (-.030, -.14), (-.052, -.30), (-.030, -.345), (.0, -.335), (.030, -.345), (.052, -.30), (.030, -.14), (.045, .00), (.085, .075), (.16, .115), (.30, .175), (.46, .185), (.56, .12), (.63, .02), (.80, .10), (.80, 1.0)], sh)
        for q in (-1, 1): poly([(q * .80, .08), (q * .60, .0), (q * .585, -.14), (q * .63, -.30), (q * .555, -.44), (q * .80, -.40)], sh)
    elif key == 'T':
        poly([(-.64, .22), (-.48, .17), (-.33, .175), (-.16, .13), (-.06, .06), (.06, .06), (.16, .13), (.33, .175), (.48, .17), (.64, .22), (.66, -.08), (.53, -.12), (.40, -.10), (.24, -.07), (.14, -.02), (.09, .0), (.05, -.14), (-.05, -.14), (-.09, .0), (-.14, -.02), (-.24, -.07), (-.40, -.10), (-.53, -.12), (-.66, -.08)], sh)
        poly([(-.040, -.40), (0, -.41), (.040, -.40), (.0, -.452)], sh)
        poly([(.66, -.1), (.60, -.30), (.44, -.46), (.34, -.66), (.24, -.86), (.2, -1.0), (.8, -1), (.8, -.1)], sh); poly([(-.66, -.1), (-.60, -.30), (-.44, -.46), (-.34, -.66), (-.24, -.86), (-.2, -1.0), (-.8, -1), (-.8, -.1)], sh)
