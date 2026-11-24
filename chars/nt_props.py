"""Insert drawings for Nanami: wrist watch, goggles (3-D, any angle, cracked), sandwich roll in a paper bag, the bakery bag."""
from chars.ntkit import *
from chars.nanami import PAL as NPAL


# ------------------------------------------------------------------ goggles, 3-D
def _lens_outline(s, k=1.0, n=10):
    P = [(.086, .060), (.145, .118), (.30, .136), (.45, .112), (.528, .040), (.524, -.060), (.456, -.142), (.31, -.172), (.175, -.154), (.102, -.085)]
    q = spl(np.array(P), True, n=n); c = np.array([.305, -.018]); q = c + (q - c) * k
    return np.c_[s * q[:, 0], q[:, 1]]


def _wrapz(x): return -.42 * np.abs(x) ** 2.2                             # the lens plane curves back round the face


def goggles3d(cel, cx, cy, s, R=None, yaw=0., pitch=0., roll=0., pal=None, lw=4., crack=(False, True), lens_fill=None, seed=0, persp=.25, glass=.62, lit=(-.5, -.8)):
    """Nanami's goggles as an object.  s px per head unit.  crack = (picture-left lens, picture-right lens).
    lens_fill(cel, side, M) is called inside the clipped lens (M maps lens plane (x,y) -> px): draw a reflection there."""
    pal = pal or NPAL['day']; c, sn = math.cos, math.sin
    if R is None:
        Ry = np.array([[c(yaw), 0, sn(yaw)], [0, 1, 0], [-sn(yaw), 0, c(yaw)]]); Rx_ = np.array([[1, 0, 0], [0, c(pitch), sn(pitch)], [0, -sn(pitch), c(pitch)]]); Rz = np.array([[c(-roll), -sn(-roll), 0], [sn(-roll), c(-roll), 0], [0, 0, 1]]); R = Rz @ Rx_ @ Ry
    def M(P, dz=0.):
        P = np.asarray(P, float); z = _wrapz(P[:, 0]) + dz; q = np.c_[P, z] @ R.T; f = 1 / (1 - np.clip(q[:, 2] * persp, -.6, .6)); return np.c_[cx + q[:, 0] * f * s, cy - q[:, 1] * f * s]
    front = R[2, 2] >= 0; T = .075                                           # rim depth
    order = sorted((-1, 1), key=lambda q: (np.array([q * .3, 0, 0]) @ R.T)[2])
    B = np.array([(-.11, .062), (-.05, .046), (0, .054), (.05, .046), (.11, .062), (.11, .006), (.05, -.010), (0, .0), (-.05, -.010), (-.11, .006)])
    def bridge():
        for dz, col in ((-T * .6, pal['frame']), (T * .4, pal['frame'])): P = M(spl(B, True, n=6), dz); cel.fill(P, col); outline(cel, P, lw * .6, pal['line'], seed)
        P = M(spl(B[:5] * [1, .9], n=6), T * .4); cel.line(P, lw * .5, pal['lensH'], taper=(.4, .4), alpha=.7)
    if not front: bridge()
    for q in order:
        o, i = _lens_outline(q), _lens_outline(q, .84)
        back, fr = M(o, -T), M(o, T * .4)
        # extruded side wall: union of the two outlines + quads
        n = len(o); quads = np.stack([back, fr, np.roll(fr, -1, 0), np.roll(back, -1, 0)], 1)
        wall = quads_contour(quads, k=2.0, close=3)
        for w_ in wall[:1]: cel.fill(w_, pal['frame']); outline(cel, w_, lw * .95, pal['line'], seed + q)
        near_, far_ = (fr, back) if front else (back, fr)
        cel.fill(near_, hexmix(pal['frame'], '#ffffff', .10)); outline(cel, near_, lw * .55, pal['line'], seed + 2 + q)
        I = M(i, T * .25 if front else -T * .7); cel.fill(I, pal['lens']); cel.clip(I); Lm = lambda P: M(np.asarray(P, float), T * .25 if front else -T * .7)
        S_ = lambda pts, col, al=1.: cel.fill(Lm(resample(np.vstack([pts, pts[:1]]), .03)), col, al)
        S_([(q * .0, .20), (q * .60, .20), (q * .56, .02), (q * .40, .05), (q * .25, .03), (q * .08, .06)], pal['lensD'])
        if lens_fill is not None: lens_fill(cel, q, Lm)
        S_([(.305 * q - .16, .16), (.305 * q - .05, .16), (.305 * q - .17, -.2), (.305 * q - .28, -.2)], pal['lensH'], glass * .45); S_([(.305 * q + .00, .16), (.305 * q + .03, .16), (.305 * q - .09, -.2), (.305 * q - .12, -.2)], pal['lensH'], glass * .4)
        S_([(q * .12, -.12), (q * .30, -.12), (q * .48, -.08), (q * .44, -.2), (q * .2, -.2)], pal['lensH'], glass * .3)
        if crack[0 if q < 0 else 1]:
            r = np.random.default_rng(seed + 14); c0 = np.array([q * .36, .035])
            for k in range(11):
                a = 2 * math.pi * k / 11 + r.normal(0, .16); p = c0.copy(); pts = [p.copy()]; L_ = r.uniform(.10, .26)
                for j in range(5): a += r.normal(0, .22); p = p + np.array([math.cos(a), math.sin(a)]) * L_ / 5; pts.append(p.copy())
                pts = Lm(np.array(pts)); cel.line(pts + [1.2, 1.2], lw * .42, pal['lensD'], taper=(.0, .6), seed=k, press=.5); cel.line(pts, lw * .34, '#f4fff8', taper=(.0, .6), seed=k, press=.5)
            for rad in (.035, .085, .14):
                a0 = r.uniform(0, 6.283)
                for k in range(4):
                    aa = np.linspace(a0 + k * 1.57 + .1, a0 + k * 1.57 + r.uniform(.6, 1.25), 5); rr = rad * (1 + r.normal(0, .12, 5)); cel.line(Lm(c0 + np.c_[np.cos(aa) * rr, np.sin(aa) * rr]), lw * .28, '#eafcf1', taper=(.2, .4), seed=k, alpha=.9)
            sh = c0 + np.array([(0, 0), (.05, .03), (.035, -.02)]) * [q, 1]; S_(sh, '#ffffff', .55)
        cel.unclip(); outline(cel, I, lw * .42, pal['line'], seed + 4 + q)
        a = len(i); cel.line(M(o[int(a * .10):int(a * .42)], T * .4 if front else -T), lw * .5, pal['lensH'], taper=(.4, .4), alpha=.85)
    if front: bridge()
    return R


# ------------------------------------------------------------------ wrist watch insert
def watch(cel, cx, cy, s, ang=-.12, hm=(17, 59, 48.), pal=None, lw=4.5, sleeve=True, hand=True, light=None, seed=0):
    """his left wrist seen from above, hand to the right.  s = px of the wrist width.  hm = (hour, minute, second)"""
    pal = pal or NPAL['blue']; ln, skl = pal['line'], pal['sline']; M = lambda P: xf(np.asarray(P, float) * s, rot=ang, tx=cx, ty=cy)
    if hand:
        try:
            from jj import hands as HD
            hp = dict(skin=pal['skin'], shade=pal['skinS'], line=ln, nail=hexmix(pal['skin'], '#ffffff', .22), nail_shade=hexmix(pal['skinS'], '#ffffff', .15), light=pal['skinH'])
            w = M([(.34, 0)])[0]; cam = HD.Cam(w[0], w[1], s * 1.72, 8.0)
            cel.clip(M([(.30, -6), (9, -6), (9, 6), (.30, 6)])); HD.draw(cel, 'relaxed', cam, hp, (-.5, .7, .5), lw * .95, seed, side='L', R=HD.euler(6, 14, -90 - math.degrees(ang)), bld=HD.SLIM, pivot=(0, 0, 0), res=700); cel.unclip()
        except Exception as ex: print('hands', ex)
    arm = [(-1.2, -.60), (-.4, -.56), (.2, -.50), (.46, -.47), (.46, .47), (.2, .50), (-.4, .57), (-1.2, .62)]; P = M(arm); cel.fill(P, pal['skin']); cel.clip(P); cel.fill(M([(-1.3, .18), (-.5, .26), (-.1, .12), (.2, .24), (.6, .16), (.6, .8), (-1.3, .8)]), pal['skinS']); cel.unclip()
    cel.line(M(spl([(-1.2, -.60), (-.4, -.56), (.2, -.50), (.42, -.475)], n=8)), lw, ln, taper=(.02, .25)); cel.line(M(spl([(-1.2, .62), (-.4, .57), (.2, .50), (.42, .475)], n=8)), lw, ln, taper=(.02, .25))
    for y in (-.2, .05): cel.line(M(spl([(.10, y), (.30, y + .02), (.46, y - .01)], n=6)), lw * .34, skl, taper=(.5, .5))
    # strap
    for sg in (-1, 1):
        st = [(-.235, sg * .30), (.235, sg * .30), (.215, sg * .49), (.20, sg * .60), (-.20, sg * .62), (-.22, sg * .5)]; P = M(st); cel.fill(P, '#4a2f25' if pal is NPAL['day'] else hexmix('#4a2f25', pal['skinD'], .35)); cel.clip(P)
        cel.fill(M([(.02, sg * .2), (.30, sg * .2), (.30, sg * .8), (.06, sg * .8)]), '#2a1915'); cel.unclip(); outline(cel, P, lw * .8, ln, 3)
        for x in (-.19, .18):
            for k in range(4): cel.line(M([(x, sg * (.34 + .065 * k)), (x, sg * (.37 + .065 * k))]), lw * .28, '#a8805c', taper=(0, 0))
        cel.line(M(spl([(-.2, sg * .55), (0, sg * .57), (.2, sg * .54)], n=6)), lw * .34, ln, taper=(.3, .3))
    # lugs, case, bezel
    for sx in (-1, 1):
        for sg in (-1, 1): P = M([(sx * .16, sg * .24), (sx * .25, sg * .24), (sx * .245, sg * .42), (sx * .185, sg * .42)]); cel.fill(P, pal['whiteS']); outline(cel, P, lw * .6, ln, 4)
    ring = lambda r, sx=1., n=60: np.c_[np.cos(np.linspace(0, 2 * math.pi, n)[:-1]) * r * sx, np.sin(np.linspace(0, 2 * math.pi, n)[:-1]) * r]
    P = M([(.34, -.05), (.415, -.06), (.415, .06), (.34, .05)]); cel.fill(P, pal['whiteS']); outline(cel, P, lw * .6, ln, 5)
    c = M(boil(ring(.375), .0015, 3)); cel.fill(c, pal['whiteS']); cel.clip(c)
    cel.fill(M(np.vstack([ring(.40)[8:36], [(0, 0)]])), pal['whiteD']); cel.fill(M([(-.34, -.22), (-.18, -.34), (.02, -.38), (-.10, -.30), (-.26, -.18)]), pal['whiteH']); cel.unclip(); outline(cel, c, lw * 1.0, ln, 6, press=.5)
    d = M(boil(ring(.318), .001, 4)); cel.fill(d, hexmix(pal['white'], '#fff6e0', .35)); cel.clip(d)
    cel.fill(M([(-.5, -.5), (.5, -.5), (.5, -.16), (.16, -.20), (-.12, -.12), (-.5, -.16)]), pal['whiteS'], .55); cel.unclip(); outline(cel, d, lw * .55, ln, 7)
    for k in range(60):
        a = k * math.pi / 30 - math.pi / 2; r0 = .262 if k % 5 == 0 else .288; cel.line(M([(math.cos(a) * r0, math.sin(a) * r0), (math.cos(a) * .302, math.sin(a) * .302)]), lw * (.62 if k % 15 == 0 else .44 if k % 5 == 0 else .16), ln, taper=(0, 0), press=0)
    h, m, sec = hm; am = (m + sec / 60.) * math.pi / 30 - math.pi / 2; ah = ((h % 12) + m / 60.) * math.pi / 6 - math.pi / 2; as_ = sec * math.pi / 30 - math.pi / 2
    def hnd(a, L_, w_, col, tail=.03):
        u, n = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)]); P_ = np.array([-u * tail + n * w_ * .7, u * L_ * .75 + n * w_, u * L_, u * L_ * .75 - n * w_, -u * tail - n * w_ * .7])
        cel.fill(M(P_ + [.008, .012]), pal['whiteS'], .8); cel.fill(M(P_), col)
    hnd(ah, .165, .017, '#20212c'); hnd(am, .255, .012, '#20212c'); hnd(as_, .275, .0035, '#b03028', .07); cel.fill(M(ring(.020, n=20)), '#20212c'); cel.fill(M(ring(.009, n=14)), pal['whiteH'])
    # glass: hard window reflection
    cel.clip(d); cel.fill(M([(-.30, -.06), (-.12, -.30), (.00, -.31), (-.22, .02), (-.30, .10)]), '#ffffff', .34); cel.fill(M([(.06, -.31), (.10, -.30), (-.14, .10), (-.18, .08)]), '#ffffff', .24); cel.unclip()
    if sleeve:
        sh = [(-5, -.86), (-1.0, -.80), (-.62, -.74), (-.585, -.2), (-.60, .25), (-.64, .76), (-1.1, .84), (-5, .95)]; P = M(sh); cel.fill(P, pal['shirt']); cel.clip(P); cel.fill(M([(-5, .30), (-1, .34), (-.5, .22), (-.5, 1.2), (-5, 1.2)]), pal['shirtS']); cel.unclip(); outline(cel, P, lw * .9, ln, 8)
        cel.fill(M([(-.585, -.72), (-.50, -.50), (-.50, .52), (-.62, .74)]), pal['skinD'], .55)
        P = M(spl([(-.80, -.20), (-.77, -.17), (-.80, -.14), (-.83, -.17)], True, n=5)); cel.fill(P, pal['white']); outline(cel, P, lw * .34, ln, 3)
        sl = [(-5, -1.10), (-2.0, -1.04), (-1.10, -.98), (-.98, -.3), (-1.0, .3), (-1.06, .98), (-2.0, 1.12), (-5, 1.25)]; P = M(spl(sl, corners=(2, 5), n=8)); cel.fill(M(np.array(sl) + [.05, .03]), pal['shirtS'], .6); cel.fill(P, pal['suit']); cel.clip(P)
        cel.fill(M([(-5, .34), (-2.6, .22), (-2.1, .40), (-1.6, .18), (-1.25, .36), (-.9, .26), (-.9, 1.4), (-5, 1.4)]), pal['suitS']); cel.fill(M([(-5, .80), (-2.4, .70), (-1.4, .78), (-.9, .70), (-.9, 1.4), (-5, 1.4)]), pal['suitD'], .7)
        for pp in ([(-1.5, -1.0), (-1.75, -.55), (-1.62, -.05), (-1.95, -.5)], [(-2.3, -1.05), (-2.5, -.3), (-2.2, .35), (-2.75, -.3)], [(-1.30, .95), (-1.55, .55), (-1.95, .42), (-1.6, .72)]): cel.fill(M(pp), pal['suitS'])
        cel.unclip(); outline(cel, P, lw, ln, 9, press=.4)
        for pp in ([(-1.48, -1.0), (-1.74, -.55), (-1.60, .0)], [(-2.3, -1.05), (-2.5, -.3), (-2.2, .35)], [(-1.28, .95), (-1.55, .55), (-2.0, .40)]): cel.line(M(spl(pp, n=8)), lw * .45, ln, taper=(.1, .7))
        for k, x in enumerate((-1.20, -1.36, -1.52)):
            P = M(spl([(x - .055, .66), (x, .61), (x + .055, .66), (x, .715)], True, n=6)); cel.fill(P, pal['suitD']); outline(cel, P, lw * .42, ln, 10 + k); cel.fill(M([(x - .03, .645), (x + .01, .625), (x, .65)]), pal['suitH'])


# ------------------------------------------------------------------ sandwich roll
def sandwich(cel, cx, cy, s, ang=-.5, pal='warm', lw=4., bite=0., seed=0, bag=True, lift=0.):
    """a crusty roll (ham, lettuce, egg) standing in a crumpled paper bag.  s = px of roll width"""
    ln = '#2a1714'; r = np.random.default_rng(seed + 3); off = [np.array([0., -lift])]; M = lambda P: xf((np.asarray(P, float) + off[0]) * s, rot=ang, tx=cx, ty=cy)
    def shp(P, col, w=1., sd=0, sh=(), hi=(), **k):
        q = M(P); cel.fill(q, col); cel.clip(q)
        for p_, c_ in sh: cel.fill(M(p_), c_)
        for p_, c_ in hi: cel.fill(M(p_), c_)
        cel.unclip(); outline(cel, q, lw * w, ln, sd, press=.45, **k); return q
    # far half of the roll
    shp(spl([(-.42, 1.6), (-.46, .6), (-.44, -.4), (-.36, -1.10), (-.16, -1.42), (.0, -1.44), (.02, -.4), (.0, 1.6)], True, corners=(0, 7), n=8), '#d99a4e', .9, 1, sh=[([(-.5, 1.8), (-.30, 1.0), (-.34, 0), (-.24, -.9), (-.05, -1.5), (.1, -1.5), (.1, 1.8)], '#f0d9a6')])
    # filling
    for k, (x, col, cs, amp) in enumerate([(-.10, '#7fae4b', '#4e7f35', .085), (.02, '#e9a3a0', '#c06a6c', .05), (.10, '#f3d873', '#c99f3c', .05), (.17, '#8fbd55', '#57883a', .075)]):
        yy = np.linspace(1.6, -1.30, 26); wob = amp * np.sin(yy * 9 + k * 2.1) + amp * .5 * np.sin(yy * 17 + k); a_ = np.c_[x - .10 + wob, yy]; b_ = np.c_[x + .10 + wob * .8 + .02, yy]
        a_[-1] = b_[-1] = [(x) * .7, -1.36]; shp(np.vstack([spl(a_, n=4), spl(b_, n=4)[::-1]]), col, .55, 4 + k, sh=[(np.vstack([spl(a_ + [.07, 0], n=4), spl(b_ + [.1, 0], n=4)[::-1]]), cs)])
    # near half with scored crust
    shp(spl([(.16, 1.6), (.16, .4), (.15, -.5), (.12, -1.18), (.10, -1.40), (.30, -1.30), (.50, -.86), (.58, -.1), (.58, .7), (.54, 1.6)], True, corners=(0, 4, 9), n=8), '#e0a052', 1., 10,
        sh=[([(.12, 1.8), (.24, .9), (.22, 0), (.28, -.6), (.18, -1.4), (.0, -1.4), (.0, 1.8)], '#f2dcae'), ([(.70, 1.8), (.46, 1.0), (.50, .3), (.44, -.3), (.46, -.9), (.7, -1)], '#b3702f')], hi=[([(.36, -.95), (.42, -.6), (.40, -.2), (.35, -.55)], '#f8cf86')])
    for k, y in enumerate((-.75, -.25, .25, .75)):
        P = spl([(.27, y + .12), (.40, y - .04), (.54, y - .10), (.44, y + .06), (.32, y + .16)], True, n=6); cel.fill(M(P), '#f6e0b0'); cel.line(M(spl([(.26, y + .12), (.40, y - .05), (.55, y - .11)], n=6)), lw * .5, '#7a4420', taper=(.2, .5)); cel.line(M(spl([(.30, y + .17), (.44, y + .06), (.54, y - .04)], n=6)), lw * .36, '#8a5226', taper=(.4, .5))
    for k in range(60): p = np.array([r.uniform(.2, .56), r.uniform(-1.1, 1.2)]); cel.line(M([p, p + [.012, .02]]), lw * .3, '#8a5226' if r.random() < .5 else '#fbe6b8', taper=(.2, .2))
    off[0] = np.array([0., 0.])
    if bag:
        # kraft bag: torn zig-zag rim, crumple folds
        xs = np.linspace(-.66, .74, 15); ys = .34 + .055 * np.where(np.arange(15) % 2, 1, -1) + .06 * np.sin(xs * 4); rim_ = np.c_[xs, ys]
        back = np.c_[xs[::-1], ys[::-1] - .13]
        cel.fill(M(np.vstack([back, rim_])), '#7d5a3c')
        body = np.vstack([rim_, [(.80, .9), (.72, 1.5), (.78, 2.4), (-.70, 2.4), (-.64, 1.5), (-.74, .9)]])
        shp(body, '#c9a06d', 1., 20, sh=[([(.80, .3), (.52, .5), (.60, .9), (.38, 1.3), (.50, 1.8), (.34, 2.5), (1, 2.5)], '#946b45'), ([(-.2, .5), (-.05, .9), (-.30, 1.3), (-.10, 1.7), (-.28, 2.4), (-.40, 2.4), (-.44, 1.5)], '#a87f53'), ([(-.74, .8), (-.52, 1.1), (-.60, 1.6), (-.48, 2.4), (-.9, 2.4)], '#a87f53')], hi=[([(.02, .48), (.22, .6), (.10, 1.0), (.26, 1.4), (.02, 1.0)], '#e6c391'), ([(-.56, .5), (-.40, .6), (-.50, .95)], '#e6c391')])
        for pp in ([(-.2, .42), (-.04, .9), (-.30, 1.32), (-.10, 1.72)], [(.52, .42), (.60, .9), (.38, 1.32), (.50, 1.8)], [(-.74, .9), (-.50, 1.1), (-.60, 1.6)], [(.05, .9), (.26, 1.4), (.12, 1.9)], [(.60, .9), (.30, 1.02), (.05, .9)]): cel.line(M(spl(pp, n=7)), lw * .42, ln, taper=(.2, .6))
        # bakery stamp: a hand-cut wheat ear, no letters
        c0 = np.array([-.02, 1.30])
        cel.line(M([c0 + [0, .34], c0 + [.01, 0], c0 + [.0, -.22]]), lw * .42, '#6a3f26', taper=(.1, .4))
        for k in range(4):
            for q in (-1, 1): b0 = c0 + [0, -.16 + .10 * k]; P = spl([b0, b0 + [q * .06, -.07], b0 + [q * .085, -.15], b0 + [q * .015, -.09]], True, corners=(0, 2), n=5); cel.fill(M(P), '#6a3f26')
