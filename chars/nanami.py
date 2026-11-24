"""KENTO NANAMI.
H = Nanami.pose(cx, cy, s, yaw, pitch, roll)      posed head (s = px chin-to-eye; picture-right yaw > 0; looks up pitch > 0)
Nanami.head(cel, H, expr, light, f, wind, seed, goggles, lit, lw, part, loose, wdir)
    expr: deadpan look_down corner closed gentle calm far speak_a speak_o grit (EXPR) or a dict (open low gaze brow_in brow_out mouth bags wet)
    light: day blue window ember beach night (PAL) or a palette dict;  lit: -1 key from picture left, +1 right, 0 flat
    part: 'back' (far hair, far ear: draw before the neck) / 'front' / 'all';  loose: 0 combed .. 1 three strands .. >1 sea wind
Nanami.bust(cx, cy, s, yaw, pitch, roll, byaw, bpitch, broll, expr, light, ..., outfit='suit'|'shirt', goggles, loosen) -> layer (about .45 s)
    b = Body.under(H, byaw, ...) then Nanami.draw_suit / draw_shirt(cel, H, b, pal, lw, ...) to build custom pictures; |body yaw| < .7
Keys, beach/ember plates, embers(), char_burn(): chars/nt_keys.py.  Inserts (watch, goggles3d, sandwich): chars/nt_props.py."""
from chars.ntkit import *

# ---------------------------------------------------------------- palettes per light
PAL = dict(
    day=dict(skin='#f7dcc6', skinS='#d8a893', skinD='#a9756c', skinH='#fff0df', hair='#d6c997', hairS='#9c9062', hairD='#696041', hairH='#f2e9c6',
             suit='#d9d0c5', suitS='#a79c98', suitD='#7a7076', suitH='#efeae0', shirt='#4d7bb0', shirtS='#315384', shirtH='#7ca5cf', white='#f4f3f1', whiteS='#bcc3d4', whiteD='#8d97b4', whiteH='#ffffff',
             tie='#d9ba45', tieS='#a6832f', spot='#241b16', sclera='#efe9e6', scleraS='#b9aeb6', iris='#8d7743', irisD='#4d3d22', brow='#6b5333', lens='#2f5142', lensD='#16261f', lensH='#8fc4a0', frame='#1c211f',
             line=INK, sline='#3a211f', hline='#2f2717', lip='#c08478'),
    blue=dict(skin='#c6a59f', skinS='#82708f', skinD='#584d78', skinH='#ffdcb0', hair='#b2a783', hairS='#82797a', hairD='#524e66', hairH='#e8d6ac',
              suit='#aaa7b5', suitS='#6e6e8b', suitD='#494b6c', suitH='#f2dcc0', shirt='#3c6096', shirtS='#294475', shirtH='#7d9cc8', white='#c3c7da', whiteS='#878db0', whiteD='#5d638c', whiteH='#ffe8c8',
              tie='#bba35a', tieS='#7b6d58', spot='#171520', sclera='#d3ccd6', scleraS='#928ba8', iris='#7a6a48', irisD='#33302f', brow='#4f4337', lens='#2c4749', lensD='#131d27', lensH='#ffd9a0', frame='#14161e',
              line='#13121e', sline='#2a1a2c', hline='#262327', lip='#9a6f76'),
    window=dict(skin='#f6cda6', skinS='#8a7192', skinD='#574b78', skinH='#fff0d0', hair='#dcc48c', hairS='#a58d74', hairD='#665a68', hairH='#fff0c8',
              suit='#e2cfb8', suitS='#7a7690', suitD='#4b4c6d', suitH='#fff0d8', shirt='#5079a6', shirtS='#2c4778', shirtH='#9cb8d6', white='#f6e8d4', whiteS='#8f93b4', whiteD='#5d638c', whiteH='#fff6e6',
              tie='#e6c25c', tieS='#7f6f56', spot='#1c1718', sclera='#f4e6d8', scleraS='#a394a8', iris='#8d713c', irisD='#3a3024', brow='#64503a', lens='#3a5a4c', lensD='#17242a', lensH='#ffe2ae', frame='#15161c',
              line='#15121c', sline='#33191f', hline='#2a2422', lip='#b27c72'),
    ember=dict(skin='#7a3f2e', skinS='#3a1a1c', skinD='#1c0d12', skinH='#ffb35c', hair='#7d5430', hairS='#3c221c', hairD='#1d1014', hairH='#ffc56a',
               suit='#5a3a36', suitS='#2b181d', suitD='#150b11', suitH='#ff9d4d', shirt='#2a2336', shirtS='#191524', shirtH='#8a5040', white='#8c5a44', whiteS='#43252a', whiteD='#1e1016', whiteH='#ffc07a',
               tie='#6f4a22', tieS='#34201a', spot='#0d080b', sclera='#b98a6c', scleraS='#5a3430', iris='#5a3a20', irisD='#1c0f10', brow='#2c1a16', lens='#1a1f1a', lensD='#0b0d0c', lensH='#ff9d4d', frame='#0b090b',
               line='#0b0709', sline='#1d0a0c', hline='#1a0d0c', lip='#4a2220'),
    beach=dict(skin='#fff0e0', skinS='#e6b7a4', skinD='#b98a94', skinH='#ffffff', hair='#f5e6b3', hairS='#cfae6c', hairD='#98794a', hairH='#fffbe6',
               suit='#f3ece2', suitS='#c4bcc6', suitD='#9a94ad', suitH='#ffffff', shirt='#7aa6d6', shirtS='#5180b4', shirtH='#b4d2ee', white='#fdfdfd', whiteS='#c3d3ea', whiteD='#93abd4', whiteH='#ffffff',
               tie='#eccb5e', tieS='#c09a44', spot='#4a3a34', sclera='#f7f3f0', scleraS='#c9bccb', iris='#94783f', irisD='#54402a', brow='#8a6a45', lens='#5f9478', lensD='#345a4a', lensH='#d9f5e2', frame='#39433f',
               line='#4a3038', sline='#7a4a48', hline='#7d5f36', lip='#d79a8c'),
    night=dict(skin='#d3b0a3', skinS='#85637a', skinD='#51406a', skinH='#e9f2ff', hair='#b9ad84', hairS='#756b6a', hairD='#474264', hairH='#dfeaff',
               suit='#a9a6b4', suitS='#6b6884', suitD='#454366', suitH='#e2eeff', shirt='#375a90', shirtS='#263c6c', shirtH='#6f95c8', white='#d5d8e6', whiteS='#8b92b5', whiteD='#5c6190', whiteH='#ffffff',
               tie='#b59a49', tieS='#775f44', spot='#15131c', sclera='#dcd6de', scleraS='#8f87a8', iris='#796740', irisD='#352b2c', brow='#51453c', lens='#27443f', lensD='#101a1e', lensH='#8fd8ff', frame='#13161a',
               line='#100f1a', sline='#251929', hline='#241f26', lip='#9a6c78'))
PAL['dawn'] = PAL['day']

EXPR = dict(
    deadpan=dict(open=.70, low=.06, gaze=(0, .02), brow_in=-.004, brow_out=.004, mouth='flat', bags=1.),
    look_down=dict(open=.40, low=.06, gaze=(.02, -.34), brow_in=-.004, brow_out=-.006, mouth='flat', bags=1.),
    corner=dict(open=.66, low=.16, gaze=(.0, -.02), brow_in=.010, brow_out=.0, mouth='corner', bags=.9, crease=.6),
    closed=dict(open=0., low=.05, gaze=(0, 0), brow_in=.020, brow_out=-.012, mouth='rest', bags=.7, closed=1),
    gentle=dict(open=.64, low=.20, gaze=(0, 0), brow_in=.030, brow_out=-.022, mouth='smile', bags=.9, crease=1.),
    calm=dict(open=.68, low=.10, gaze=(.0, .02), brow_in=.020, brow_out=-.014, mouth='rest', bags=1.),
    speak_a=dict(open=.70, low=.18, gaze=(0, 0), brow_in=.028, brow_out=-.020, mouth='a', bags=.9, crease=.7),
    speak_o=dict(open=.70, low=.18, gaze=(0, 0), brow_in=.028, brow_out=-.020, mouth='to', bags=.9, crease=.7),
    far=dict(open=.66, low=.08, gaze=(.05, .04), brow_in=.004, brow_out=-.006, mouth='flat', bags=1.),
    grit=dict(open=.56, low=.20, gaze=(0, 0), brow_in=-.030, brow_out=.02, mouth='grit', bags=1.))

HAIR = Hair(yc=.20, ry=.855, rx=.700, rzf=.82, rzb=.93, zc=-.06, low=.95, box=.74)


def hairline(th):
    """v of Nanami's hairline at azimuth th (deg)"""
    a = abs(th)
    t = [0, 18, 34, 48, 60, 70, 78, 84, 90, 100, 112, 125, 145, 180]
    v = [.265, .285, .330, .285, .18, .03, -.16, -.16, .0, .02, -.08, -.44, -.80, -.84]
    return float(np.interp(a, t, v))


def hairline2(th):
    """fill line of the shell: as hairline, but at the back it stops where the points begin"""
    return max(hairline(th), float(np.interp(abs(th), [100, 112, 124, 140, 180], [-2., -.04, -.16, -.30, -.40])))


class _Nanami:
    KW = dict(a=[(-1.0, .17), (-.965, .235), (-.90, .305), (-.80, .405), (-.68, .512), (-.58, .566), (-.45, .584), (-.31, .606), (-.17, .634), (.12, .640), (.35, .655), (.6, .63), (.8, .50), (.93, .31), (1.0, .05)],
              f=[(-1.0, .655), (-.92, .735), (-.8, .745), (-.64, .755), (-.46, .755), (-.28, .745), (-.06, .73), (.06, .745), (.2, .795), (.4, .79), (.6, .72), (.8, .54), (.93, .32), (1.0, .05)],
              p=2.2, nose=(.090, .0, -.355, -.415, .165), ear=(.650, .10, -.36, -.10), cheek=.040, socket=.045)
    # sagittal profile (z, y): high straight bridge, long upper lip, square chin
    PROF = [(.70, .74), (.775, .58), (.80, .40), (.815, .24), (.822, .15), (.79, .06), (.775, .0), (.815, -.12), (.868, -.27), (.905, -.385), (.912, -.425), (.888, -.465), (.80, -.492), (.782, -.525), (.790, -.60),
            (.812, -.640), (.808, -.662), (.775, -.678), (.800, -.700), (.806, -.725), (.782, -.765), (.768, -.80), (.790, -.86), (.800, -.92), (.772, -.975), (.70, -1.0)]
    ZJ = [.66, .655, .57, .44, .26, .10, .00, -.06, -.10, -.10]

    def make(self): return FHead(zj=self.ZJ, prof=self.PROF, low=(-.42, .90), **self.KW)

    def pose(self, cx, cy, s, yaw=0., pitch=0., roll=0., persp=.10): return self.make().pose(cx, cy, s, yaw, pitch, roll, persp)

    # ------------------------------------------------------------ hair
    def _locks(self, H_, pal, f=0, wind=0., seed=0, wdir=(1., .1, 0.), loose=1.0):
        out = []
        wv = lambda k, a=1.: wind_vec(f, wind * a, k, wdir) if wind else None
        def add(path, w, off=None, wa=.3, k=0, **kw):
            d = lock(H_, HAIR, path, w, kw.pop('prof', None), off, wind=wv(k, wa)); d.update(kw); out.append(d)
        under = dict(col=pal['hairS'], shade=pal['hairD'], sh=.42, lw=.8)
        # back: the combed mass runs over the crown and down, breaks into unequal points and tapers into the nape
        # flow: sweeps over from the part, turns down behind the crown, draws in toward the nape
        inn = lambda th, k: th + (180 - th) * k
        def flow(th, dr, le=0.): return [(th - 52 + dr, .80, .105), (th - 30 + dr * .6, .62, .118), (th - 10, .34, .108), (inn(th, .03), .04, .082 + le), (inn(th, .14), -.22, .056 + le), (inn(th, .30), -.42, .030 + le)]
        BK = [(104, 10), (127, 2), (138, -6), (167, 4), (180, -4), (212, 8), (229, -2), (254, -10)]; TV = [-.52, -.48, -.74, -.60, -.78, -.56, -.50]; TT = [130, 141, 160, 174, 191, 207, 222]
        def bp(e0, e1, **kw):
            d = patch_lock(H_, HAIR, e0, e1, None, wind=wv(kw.get('k', 0), .10)); d.update(kw); out.append(d)
        # fine short hairs on the nape, under the points: unequal, drawn in
        rn = np.random.default_rng(14)
        for i in range(17):
            th = 132 + 96 * (i + .5) / 17 + rn.normal(0, 2); v0 = -.40 - .12 * rn.random(); ln = rn.uniform(.14, .40) * (1.2 - abs(th - 180) / 70.)
            add([(th, v0, .020), (inn(th, .06) + rn.normal(0, 2), v0 - ln * .55, .014), (inn(th, .16) + rn.normal(0, 3), v0 - ln, .004)], rn.uniform(.05, .10), wa=0, k=200 + i, sh=.3, t0=.1, lw=.5, mass=0, strands=0, col=pal['hairS'], shade=pal['hairD'], prof=[(0, .9), (.4, .8), (1, 0)])
        for i in (0, 6, 1, 5, 3, 2, 4):
            (t0_, d0), (t1_, d1) = BK[i], BK[i + 1]; tp = (TT[i], TV[i], .010); far_ = t0_ >= 180
            bp(flow(t0_, d0) + [tp], flow(t1_, d1, .012) + [tp], k=i, sh=(.16, .10, .22, .14, .40, .46, .50)[i], t0=(.34, .46, .22, .52, .28, .40, .34)[i], lw=.85, strands=(2, 0, 3, 1, 2, 1, 1)[i], side=-1, hi=(.12, .30) if i in (2,) else 0)
        # strands the wind has picked out of the crown
        if loose > .5:
            for i, (th, v, L_, up_, w_) in enumerate([(150, .70, .16, .10, .030), (196, .60, .20, .06, .036), (172, .30, .15, .02, .024), (128, .34, .12, .03, .022), (206, -.02, .13, -.02, .020)]):
                wd = wdir[0]; add([(th, v, .10), (th + 4, v - .06, .135), (th + 8, v - .14, .140), (th + 10, v - .22, .115)], w_, [(0, 0, 0), (wd * L_ * .2, up_ * .4, 0), (wd * L_ * .6, up_ * .8, 0), (wd * L_ * 1.1, up_ * .3, 0)], wa=.9, k=230 + i, sh=.2, t0=.2, lw=.55, thin=1, strands=0, prof=[(0, .1), (.35, 1), (.7, .8), (1, 0)])
        # short sides in shadow, combed back
        for sgn in (-1, 1):
            for i, (th0, v0, th1, v1, w) in enumerate([(84, -.06, 118, -.26, .15), (75, .04, 126, -.20, .22), (68, .16, 136, -.10, .27), (70, .27, 140, .06, .28)]):
                add([(sgn * th0, v0, .012), (sgn * (th0 * .6 + th1 * .4), (v0 + v1) / 2 + .03, .04), (sgn * th1, v1, .05)], w, wa=.08, k=20 + i, t0=.35, side=sgn, **under)
            add([(sgn * 74, .02, .006), (sgn * 75.5, -.12, .010), (sgn * 77, -.25, .004)], .12, wa=0, k=26, t0=.35, side=sgn, prof=[(0, 1), (.65, .8), (1, .15)], **under)
        def patch(e0, e1, off=None, wa=.2, k=0, **kw):
            d = patch_lock(H_, HAIR, e0, e1, off, wind=wv(k, wa)); d.update(kw); out.append(d)
        # ---- combed-over mass.  flow lines run from the part over the long side (g) and the short side (h); locks are the bands between them
        g = [[(-27, .275, .010), (-15, .305, .085), (6, .290, .115), (28, .285, .115), (52, .250, .105), (74, .180, .085), (95, .120, .075)],
             [(-28, .365, .045), (-13, .470, .170), (10, .510, .205), (36, .480, .180), (64, .365, .135), (90, .205, .095), (104, .120, .085)],
             [(-29, .455, .065), (-9, .590, .195), (18, .640, .215), (50, .590, .175), (82, .430, .125), (108, .265, .090), (120, .190, .080)],
             [(-30, .555, .075), (-2, .715, .180), (34, .755, .190), (74, .660, .150), (110, .440, .105), (132, .300, .080), (140, .240, .075)],
             [(-32, .680, .080), (14, .870, .150), (78, .870, .145), (126, .640, .115), (148, .420, .085), (158, .300, .075), (160, .240, .070)]]
        tip = lambda e, f_: e[:-1] + [f_]
        def two(a, b, f_, off=None, **kw): patch(tip(a, f_), tip(b, f_), off, **kw)
        two(g[4], g[3], (162, .20, .07), wa=.1, k=40, sh=.30, t0=.30, side=1, strands=3)
        two(g[3], g[2], (146, .16, .075), wa=.15, k=41, sh=.34, t0=.10, side=1, strands=3)
        two(g[2], g[1], (126, .09, .085), [(0, 0, 0), (.03, .0, -.01)], wa=.5, k=42, sh=.38, t0=.03, side=1, hi=(.20, .50), strands=3)
        two(g[1], g[0][:-1] + [(97, .115, .08), (0, 0, 0)], (114, .06, .085), [(0, 0, 0), (.045, .0, -.02)], wa=.6, k=43, sh=.44, t0=.02, side=1, hi=(.20, .46), strands=4, root=.10)
        # a medium lock that leaves the roll at the temple
        patch([(38, .470, .175), (56, .360, .130), (76, .210, .100), (94, .130, .085), (108, .040, .080)], [(38, .470, .175), (57, .315, .112), (76, .185, .090), (92, .090, .080), (108, .040, .080)], [(0, 0, 0), (.03, -.01, -.01)], wa=.35, k=44, sh=.40, t0=.30, side=1, strands=2)
        # short side of the part
        h = [[(-27, .275, .010), (-40, .325, .065), (-54, .270, .090), (-70, .120, .095), (-90, -.020, .085)], [(-29, .400, .050), (-45, .430, .115), (-66, .315, .115), (-90, .120, .090), (-106, .030, .075)],
             [(-31, .560, .070), (-60, .550, .130), (-92, .370, .115), (-120, .190, .080), (-128, .120, .075)], [(-34, .690, .075), (-82, .710, .125), (-128, .490, .095), (-150, .300, .075), (-156, .220, .070)]]
        two(h[3], h[2], (-154, .14, .07), wa=.1, k=46, sh=.30, t0=.25, side=-1, strands=3)
        two(h[2], h[1], (-130, .03, .075), wa=.12, k=47, sh=.34, t0=.06, side=-1, strands=3)
        two(h[1], h[0], (-110, -.08, .07), [(0, 0, 0), (-.02, 0, 0)], wa=.15, k=48, sh=.40, t0=.03, side=-1, strands=3)
        # fine roots along the hairline: hair grows out of skin, not out of an edge
        rr = np.random.default_rng(4)
        for i, th in enumerate([-24, -19, 49, -58, -50]):
            v = hairline(th); g = 1 if th > -30 else -1; ln = rr.uniform(.06, .12)
            add([(th, v - rr.uniform(.0, .035), .006), (th + g * rr.uniform(4, 8), v + ln * .6, .05), (th + g * rr.uniform(12, 20), v + ln, .09)], rr.uniform(.05, .10), wa=0, k=90 + i, sh=0, t0=.0, lw=.55, mass=0, col=pal['hairS'], strands=0, prof=[(0, .25), (.5, 1), (1, .2)], fine=1)
        # the one lock that will not lie down: leaves the roll at the part, falls across the brow, moves in wind
        if loose > 0:
            patch([(-27, .470, .150), (-19, .455, .205), (-10, .370, .225), (-3, .255, .215), (4, .120, .165), (9, .000, .120)], [(-27, .470, .150), (-13, .430, .200), (-3, .345, .222), (3, .240, .208), (7, .115, .162), (9, .000, .120)], [(0, 0, 0), (.0, .0, .01), (.02, -.02, .02), (.075, -.05, .02)], wa=.9, k=70, sh=.34, t0=.22, lw=.85, thin=1, side=1, strands=2)
            if loose > .8: patch([(22, .420, .175), (32, .395, .205), (41, .300, .200), (48, .150, .150)], [(22, .420, .175), (34, .385, .203), (43, .292, .198), (48, .148, .150)], [(0, 0, 0), (.01, -.01, .01), (.06, -.04, .0)], wa=1.2, k=71, sh=0, t0=.15, lw=.6, thin=1, side=1, strands=0)
        if loose > 1:                                                      # sea wind: the ends of the mass lift, two hairs fly
            for i, (th, v, L_, up_, w_) in enumerate([(62, .44, .20, .015, .034), (74, .30, .24, -.03, .046), (80, .22, .16, -.05, .026)]):
                add([(th, v, .11), (th + 14, v + .01, .165), (th + 28, v - .01, .175), (th + 40, v - .05, .15)], w_, [(0, 0, 0), (L_ * .15, up_ * .2, 0), (L_ * .5, up_ * .8, -.02), (L_ * .95, up_ * .3, -.05)], wa=.8 + .4 * i, k=110 + i, sh=.2, t0=.3, lw=.6, thin=1, strands=0, prof=[(0, .1), (.3, .9), (.6, 1), (1, 0)])
        # flicks at temple and behind the ear
        for i, (th, v, dx, dy, w) in enumerate([(94, .10, .075, .015, .11), (-101, .08, -.080, -.01, .10), (112, .20, .08, .03, .12), (-124, -.30, -.07, -.10, .12), (130, -.34, .07, -.10, .12)]):
            g = 1 if th > 0 else -1
            add([(th - 16 * g, v + .03, .05), (th, v + .01, .075), (th + 10 * g, v - .02, .07)], w, [(0, 0, 0), (dx * .3, dy * .3, 0), (dx, dy, -.04)], wa=.5, k=80 + i, sh=.4, t0=.0, lw=.8, side=g, mass=0, prof=[(0, .8), (.4, 1), (1, 0)])
        return out

    @staticmethod
    def _lift(T, V):
        """base volume: flat sides, a wave on the long side of the part"""
        top = smooth(.24, .56, V); wave = np.exp(-((T - 14) / 58.) ** 2)
        return .022 + top * (.085 + .105 * wave * (1 - smooth(.72, .98, V)) + .04 * smooth(.72, .98, V))

    def hair(self, cel, H_, pal, lw, back, f=0, wind=0., seed=0, wdir=(1., .1, 0.), loose=1., lit=-1):
        ls = self._locks(H_, pal, f, wind, seed, wdir, loose)
        fr, bk = HAIR.shell(H_, hairline2, lift=self._lift)
        if back:
            cs = quads_contour(bk)
            for c in cs: cel.fill(c, pal['hairD'])
            sel = [d for d in ls if d['face'] < -.05]
            u = union_contour([d['poly'] for d in sel] + cs) if sel else cs
            for c in u: outline(cel, c, lw * .95, pal['hline'], 5 + seed)
            draw_locks(cel, sel, pal['hairS'], pal['hairD'], pal['hline'], lw * .42, seed=seed, shade=False)
            return
        sel = [q for q in (trim_lock(d, -.06 if not d.get('thin') else -.30) for d in ls if d['face'] >= -.05 or d['fz'].max() > .25) if q is not None]
        lowf, _ = HAIR.shell(H_, hairline2, v1=.40, lift=self._lift)
        cs = quads_contour(fr)
        for c in cs: cel.fill(c, pal['hair'])
        for c in quads_contour(lowf): cel.fill(c, pal['hairS'])
        thin = [d for d in sel if d.get('thin')]; sel = [d for d in sel if not d.get('thin')]
        u = union_contour([d['poly'] for d in sel if d.get('mass', 1)] + cs)
        for d in sel: d['side'] = -lit if lit else d.get('side', 1)
        fine = [d for d in sel if d.get('fine')]; sel = [d for d in sel if not d.get('fine')]
        draw_locks(cel, fine, pal['hairS'], pal['hairD'], pal['hline'], lw * .44, seed=seed, shade=False)
        draw_locks(cel, sel, pal['hair'], pal['hairS'], pal['hline'], lw * .44, seed=seed, hi=pal['hairH'])
        for k, d in enumerate(fine):                                        # roots again as pen flicks over the edge
            m = d['mid']; cel.line(m, lw * .30, pal['hline'], taper=(.25, .6), seed=900 + k, alpha=.8)
        for i, c in enumerate(u): outline(cel, c, lw * .72, pal['hline'], 11 + seed + i, press=.6, breaks=1)
        for k, d in enumerate(thin):
            draw_locks(cel, [d], pal['hair'], pal['hairS'], pal['hline'], lw * .44, seed=300 + k, t0=.1)

    # ------------------------------------------------------------ face
    EX, EW = .283, .140
    gog = {}
    def face(self, cel, H_, pal, lw, e, lit=-1, seed=0, goggles=False, cast=1., hatch_=0., eat=0.):
        """eat 0..1: light dissolves the contour on the lit side"""
        P = H_.parts(); fill = P['fill']; near = H_.near(); skl = pal['sline']
        prof = abs(H_.yaw) > 1.12
        if H_.pitch > .02: cel.fill(H_.under(), pal['skinS'])
        cel.fill(fill, pal['skin']); cel.clip(fill)
        q = lit if lit else -1; s = -q                           # +1 light from the right: shadow side = -q
        z0 = lambda x, y, d: np.array([x, y, float(H_.z(x, y)) + d])
        if lit != 0 and not prof:
            # side plane: temple, socket, cheekbone, HOLLOW, jaw, chin
            S(cel, H_, [(s * .70, .60), (s * .56, .50), (s * .50, .34), (s * .52, .20), (s * .452, .10), (s * .470, .0), (s * .500, -.10), (s * .490, -.19), (s * .425, -.28), (s * .350, -.37), (s * .318, -.47), (s * .332, -.57), (s * .302, -.66), (s * .245, -.755), (s * .195, -.83),
                        (s * .165, -.91), (s * .06, -1.0), (s * .4, -1.02), (s * .72, -.6), (s * .75, 0)], pal['skinS'])
            S(cel, H_, [(s * .70, -.15), (s * .585, -.22), (s * .545, -.31), (s * .455, -.40), (s * .435, -.50), (s * .455, -.60), (s * .40, -.74), (s * .34, -.86), (s * .26, -.97), (s * .5, -1.02), (s * .75, -.6)], pal['skinD'], alpha=.24)
            # hollow of the lit cheek: one thin wedge
            S(cel, H_, [(-s * .550, -.260), (-s * .470, -.340), (-s * .405, -.43), (-s * .375, -.52), (-s * .415, -.450), (-s * .482, -.368)], pal['skinS'], alpha=.8)
            if pal.get('blown'): S(cel, H_, [(-s * .72, .55), (-s * .53, .46), (-s * .485, .30), (-s * .52, .16), (-s * .50, .02), (-s * .535, -.10), (-s * .51, -.20), (-s * .575, -.32), (-s * .52, -.50), (-s * .48, -.64), (-s * .36, -.80), (-s * .25, -.95), (-s * .5, -1.0), (-s * .8, -.4)], pal['skinH'])
        if prof:
            n = near; Z = lambda pts: H_.P3(np.array([(n * .60, y, z) for z, y in pts]))
            cel.fill(Z([(.30, .62), (.34, .40), (.26, .22), (.30, .08), (.24, -.10), (.30, -.24), (.40, -.36), (.36, -.50), (.42, -.62), (.40, -.74), (.48, -.86), (.56, -1.0), (.2, -1.05), (-.4, -.6), (-.4, .6)]), pal['skinS'])
        else:
            for ss in (-1, 1):
                k = 1. if (ss == s or lit == 0) else .5
                S(cel, H_, [(ss * .070, .120), (ss * .16, .150), (ss * .40, .165), (ss * .50, .115), (ss * .440, .070), (ss * .32, .075 - .02 * k), (ss * .20, .062 - .02 * k), (ss * .140, .010 - .03 * k), (ss * .112, -.055 * k), (ss * .085, .02)], pal['skinS'])
                S(cel, H_, [(ss * .080, .070), (ss * .125, .062), (ss * .150, .030), (ss * .108, .018)], pal['skinD'], alpha=.38)
            # nose: a wedge of shadow at the bridge, the underside of the tip, nothing between
            sh = s if lit else (1 if H_.yaw <= 0 else -1); nS = hexmix(pal['skin'], pal['skinS'], pal.get('nose', .62))
            cel.fill(H_.P3(np.array([z0(0, .090, .010), z0(sh * .008, .02, .035), z0(sh * .014, -.07, .066), z0(sh * .024, -.125, .080), z0(sh * .040, -.06, .03), z0(sh * .056, .0, .0), z0(sh * .070, .06, 0), z0(sh * .04, .095, 0)])), nS)
            cel.fill(H_.P3(np.array([z0(sh * .020, -.322, .150), z0(sh * .010, -.362, .163), z0(-sh * .030, -.388, .128), z0(0, -.400, .12), z0(sh * .045, -.395, .09), z0(sh * .084, -.388, .035), z0(sh * .088, -.368, .030), z0(sh * .052, -.358, .085)])), nS)
            cel.fill(H_.P3(np.array([z0(-.066, -.392, .05), z0(0, -.404, .12), z0(.074, -.392, .05), z0(.050, -.418, .015), z0(sh * .030, -.446, .0), z0(-.030, -.418, .02)])), nS)
            cel.fill(H_.P3(np.array([z0(-.044, -.398, .075), z0(0, -.406, .12), z0(.050, -.398, .07), z0(.022, -.414, .06), z0(-.022, -.412, .06)])), pal['skinS'], .75)
        if cast > 0:
            g = [(-.70, .50), (-.61, .18), (-.55, .28), (-.50, .40), (-.385, .505), (-.28, .440), (-.205, .405), (-.10, .455), (.06, .435), (.20, .445), (.32, .405), (.41, .345), (.49, .235), (.55, .12), (.60, .13), (.70, .5), (.4, .9), (-.4, .9)]
            S(cel, H_, g, pal['skinS'])
            if e.get('lockshade', 0): S(cel, H_, [(-.12, .50), (-.02, .36), (.045, .20), (.075, .13), (.085, .20), (.06, .34), (.02, .48)], pal['skinS'])
        cel.unclip()
        if not prof:
            for ss in (-1, 1): self._eye(cel, H_, pal, lw, e, ss, fill, seed, shade=(ss == s))
            self._nose(cel, H_, pal, lw, s, seed, fill)
            self._mouth(cel, H_, pal, lw, e, seed, fill)
            cel.clip(fill)
            for ss in (-1, 1):
                if H_._face_cos(np.array(ss * .42), np.array(-.4)) < .12: continue
                L(cel, H_, [(ss * .545, -.235), (ss * .475, -.325), (ss * .400, -.425), (ss * .365, -.515)], lw * (.34 if ss == s else .28), skl, seed + 3, taper=(.45, .7))
                if ss == s: L(cel, H_, [(ss * .505, -.335), (ss * .440, -.425), (ss * .405, -.51)], lw * .22, skl, seed + 4, taper=(.6, .6))
            if hatch_ > 0:
                for k in range(6):
                    y0 = -.30 - .062 * k; x0 = np.interp(y0, [-.66, -.57, -.47, -.37, -.28], [.302, .332, .318, .350, .425])
                    if H_._face_cos(np.array(s * x0), np.array(y0)) < .1: continue
                    L(cel, H_, [(s * (x0 + .006), y0 + .004), (s * (x0 + .042), y0 - .030), (s * (x0 + .070), y0 - .066)], lw * .22, skl, seed + 80 + k, taper=(.6, .5), alpha=hatch_)
            cel.unclip()
        else:
            self._profile(cel, H_, pal, lw, e, near, seed, fill)
        for nm in ('left', 'right'):
            side = -1 if nm == 'left' else 1; pl = P[nm]
            if eat > 0 and side == lit:                          # the sea light takes the line: thin, warm, broken
                col = hexmix(pal['line'], pal.get('eatc', '#f0c9a4'), .35 + .55 * eat); n_ = len(pl)
                for a_, b_, w_ in ((.0, .30, .62), (.33, .52, .40), (.60, .74, .30), (.86, 1., .30)): cel.line(pl[int(n_ * a_):int(n_ * b_)], lw * w_ * (1 - .45 * eat), col, taper=(.10, .5), press=.4, seed=seed + 5, tmin=.1)
            else: cel.line(pl, lw, pal['line'], taper=(.10, .22), press=.36, seed=seed + (1 if nm == 'left' else 2), tmin=.3)
        return P

    def _eye(self, cel, H_, pal, lw, e, s, fill, seed, shade=False):
        """narrow, heavy: a straight thick upper lid over a small iris cut by both lids, a tired lower lid with its bag"""
        ex, ey, ew, eh = self.EX, -.012, self.EW, .104; tl = .003; as_ = e.get('asym', 1.) * (1. if s > 0 else 0.)
        o = (e.get('open', .62) if 'open2' not in e else e['open2'][0 if s < 0 else 1]) * (1 - .07 * as_)
        lo = e.get('low', .1) + .02 * as_; by = .006 * as_
        E = lambda U: eye_pts(s, ex, ey, ew, eh, tl, U); base = -.18
        if H_._face_cos(np.array(s * .28), np.array(0.)) < .02: return
        up = [(-1.06, -.34), (-.90, -.02), (-.56, .26), (-.06, .36), (.46, .33), (.86, .16), (1.08, -.06)]
        if e.get('closed'): u = sp([(-1., -.30), (-.7, -.48), (-.3, -.60), (.15, -.62), (.6, -.52), (.95, -.32), (1.12, -.16)])
        else: u = sp([(a, base + (b - base) * o * 1.36 if b > base else b) for a, b in up])
        low = sp([(-1.06, -.34), (-.62, -.54 + lo * .5), (-.06, -.62 + lo * .95), (.46, -.56 + lo * .95), (.86, -.34 + lo * .5), (1.08, -.06)])
        cel.clip(fill)
        if not e.get('closed'):
            poly = np.vstack([u, low[::-1]]); pol = H_.Q(E(poly)); big = H_.s > 150
            th = np.interp(u[:, 0], [-1.06, -.6, 0, .6, 1.08], [.08, .22, .28, .30, .15])
            cr = np.c_[u[:, 0], u[:, 1] + th + np.interp(u[:, 0], [-1.06, -.4, .3, 1.08], [.10, .22, .26, .22])]
            cel.fill(H_.Q(E(np.vstack([cr, u[::-1]]))), pal['skinS'], .85)
            cel.fill(pol, pal['sclera']); cel.clip(pol)
            gx, gy = e.get('gaze', (0, 0)); rad = .0640
            c = (s * ex + .006 * -s + gx * .5 * ew, ey + .004 + gy * eh)
            circ = lambda r, oy=0., ox=0., sy=1.: [(c[0] + ox + r * math.cos(a), c[1] + oy + r * sy * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 34)[:-1]]
            cel.fill(H_.Q(E([(-1.1, -.5), (-.80, -.38), (-.76, -.16), (-.86, .1), (-1.1, .0)])), hexmix(pal['skinS'], pal['sclera'], .35))
            q_ = H_.Q(circ(rad)); t_, b_ = H_.Q(np.array([(c[0], c[1] + rad), (c[0], c[1] - rad)]))
            cel.grad(q_, t_, b_, [(0, pal['irisD'], 1), (.40, hexmix(pal['irisD'], pal['iris'], .55), 1), (.70, pal['iris'], 1), (1, hexmix(pal['iris'], '#fff0c0', .30), 1)])
            cel.fill(H_.Q(circ(.0265)), pal['line'])
            cel.line(np.vstack([q_, q_[:1]]), lw * .42, pal['line'], taper=(0, 0), seed=seed, press=.1)
            cel.fill(np.vstack([H_.Q(E(u)), H_.Q(E([(a, b - .20) for a, b in u]))[::-1]]), pal['scleraS'], .62); cel.fill(np.vstack([H_.Q(E(u)), H_.Q(E([(a, b - .11) for a, b in u]))[::-1]]), pal['line'], .34)
            cel.fill(H_.Q(circ(.0145, .012, -.024, .85)), '#ffffff')
            if big: cel.fill(H_.Q(circ(.0060, -.026, .024)), '#ffffff', .75)
            wet = e.get('wet', 0)
            if wet: cel.fill(np.vstack([H_.Q(E(low)), H_.Q(E([(a, b + .09 * wet) for a, b in low]))[::-1]]), '#ffffff', .62)
            cel.unclip()
            lash = np.vstack([u, [(1.22, -.16)], np.c_[u[:, 0], u[:, 1] + th][::-1]]); cel.fill(H_.Q(E(lash)), pal['line'])
            cel.line(H_.Q(E(u)), lw * .40, pal['line'], taper=(.3, .1), seed=seed + 9)
            cel.line(H_.Q(E(cr[int(len(cr) * .06):])), lw * .36, pal['sline'], taper=(.45, .5), seed=seed + 12)
            cel.line(H_.Q(E(low[int(len(low) * .18):])), lw * .50, pal['sline'], taper=(.6, .12), seed=seed + 14)
        else:
            th = np.interp(u[:, 0], [-1., -.5, .2, .8, 1.12], [.05, .14, .19, .18, .06]); cel.fill(H_.Q(E(np.vstack([u, np.c_[u[:, 0], u[:, 1] + th][::-1]]))), pal['line'])
            for a0, b0, a1, b1 in ((.64, -.52, .74, -.68), (.86, -.40, 1.0, -.56)): cel.line(H_.Q(E([(a0, b0), ((a0 + a1) / 2 + .02, (b0 + b1) / 2), (a1, b1)])), lw * .40, pal['line'], taper=(0, .8), seed=seed)
            cel.line(H_.Q(E(sp([(-.80, .02), (-.2, .16), (.4, .12), (.95, -.10)]))), lw * .34, pal['sline'], taper=(.5, .5), seed=seed + 12)
        b = e.get('bags', 1.); cz = e.get('crease', 0)
        if b > 0:
            cel.line(H_.Q(E(sp([(-.98, -.56), (-.55, -.92 + lo * .3), (.0, -1.06 + lo * .3), (.42, -1.00 + lo * .3)]))), lw * .36 * b, pal['sline'], taper=(.25, .75), seed=seed + 20)
            if H_.s > 200: cel.line(H_.Q(E(sp([(-.66, -1.10 + lo * .3), (-.25, -1.26 + lo * .3), (.10, -1.28 + lo * .3)]))), lw * .20 * b, pal['sline'], taper=(.5, .7), seed=seed + 21)
        if cz > 0:                                               # the smile lives here: the cheek comes up under the eye
            cel.line(H_.Q(E(sp([(.30, -1.02 + lo * .25), (.80, -.86 + lo * .25), (1.16, -.52)]))), lw * .34 * cz, pal['sline'], taper=(.6, .4), seed=seed + 22)
            for a0, b0, a1, b1 in ((1.26, -.10, 1.52, .02), (1.28, -.30, 1.54, -.36)): cel.line(H_.Q(E([(a0, b0), ((a0 + a1) / 2, (b0 + b1) / 2 - .02), (a1, b1)])), lw * .22 * cz, pal['sline'], taper=(.2, .7))
        # brow: straight, low, dark; a little uneven
        bi, bo = e.get('brow_in', 0) + by, e.get('brow_out', 0) + by * .5
        bt = [(.080, .104 + bi), (.14, .132 + bi * .8), (.26, .158 + bi * .3), (.39, .180 + bo * .5), (.49, .170 + bo), (.555, .126 + bo)]
        bb = [(.555, .126 + bo), (.48, .150 + bo), (.38, .150 + bo * .5), (.26, .120 + bi * .3), (.15, .088 + bi * .8), (.088, .068 + bi)]
        br = np.array([(s * x, y) for x, y in np.vstack([sp(bt), sp(bb)])]); cel.fill(H_.Q(br), pal['brow'])
        cel.line(H_.Q(np.array([(s * x, y) for x, y in sp(bb)])[::-1]), lw * .40, pal['line'], taper=(.2, .6), seed=seed + 30)
        for k in range(5):
            x = .100 + .050 * k; y = float(np.interp(x, [q[0] for q in bb[::-1]], [q[1] for q in bb[::-1]]))
            cel.line(H_.Q(np.array([(s * x, y + .003), (s * (x + .018), y + .018), (s * (x + .042), y + .024)])), lw * .22, pal['line'], taper=(.1, .7), seed=seed + 40 + k, alpha=.75)
        cel.unclip()

    def _nose(self, cel, H_, pal, lw, s, seed, fill):
        """tip and nostril only; the bridge is a shadow, not a line"""
        z0 = lambda x, y, d: np.array([x, y, float(H_.z(x, y)) + d]); yaw = H_.yaw; far = 1 if yaw > 0 else -1; skl = pal['sline']
        if abs(yaw) > .16:
            k = min(1., max(0., (abs(yaw) - .5) / .4))           # the bridge line only appears when the nose starts to break the far cheek
            rd = np.array([z0(0, -.12 - .12 * (1 - k), .095 + .02 * (1 - k)), z0(0, -.28, .142), z0(0, -.325, .160), z0(0, -.358, .166), z0(0, -.385, .150), z0(-far * .018, -.402, .115)])
            cel.line(spl(H_.P3(rd), n=8), lw * .56, skl, taper=(.65, .25), press=.2, seed=seed + 50)
            n = -far
            cel.line(spl(H_.P3(np.array([z0(n * .076, -.325, .05), z0(n * .098, -.356, .042), z0(n * .094, -.386, .034)])), n=8), lw * .36, skl, taper=(.6, .25), seed=seed + 51)
            cel.line(spl(H_.P3(np.array([z0(n * .012, -.410, .10), z0(n * .042, -.400, .08), z0(n * .068, -.407, .05)])), n=6), lw * .62, pal['line'], taper=(.35, .35), seed=seed + 52)
        else:
            cel.line(spl(H_.P3(np.array([z0(s * .060, .070, 0), z0(s * .054, .01, .02), z0(s * .052, -.05, .03)])), n=6), lw * .36, skl, taper=(.2, .7), seed=seed + 50)
            cel.line(spl(H_.P3(np.array([z0(-.050, -.398, .085), z0(-.024, -.410, .12), z0(0, -.414, .135), z0(.024, -.410, .12), z0(.050, -.398, .085)])), n=6), lw * .46, skl, taper=(.4, .4), seed=seed + 51)
            for q in (-1, 1): cel.line(spl(H_.P3(np.array([z0(q * .086, -.350, .04), z0(q * .096, -.375, .035), z0(q * .084, -.396, .04)])), n=6), lw * .32, skl, taper=(.5, .3), seed=seed + 54)

    MOUTH = dict(                                                # (left corner dy, right corner dy, half width, bow)  units
        flat=(-.004, -.004, .128, 0.), calm=(.002, .002, .122, 0.), corner=(-.002, .016, .130, .0), smile=(.002, .0135, .122, .003), grit=(-.014, -.014, .145, 0.), rest=(.0, .004, .120, .001))

    def _mouth(self, cel, H_, pal, lw, e, seed, fill):
        m = e.get('mouth', 'flat'); cel.clip(fill); y = -.592; near = H_.near() or 1; sk = pal['sline']
        if m in self.MOUTH:
            c0, c1, w, bow = self.MOUTH[m]; cl, cr = (c0, c1) if near > 0 else (c1, c0); dx = .004
            # one relaxed line: thin, it only thickens where the lips press; broken once
            pts = [(-w + dx, y + cl), (-w * .74 + dx, y - .003 + cl * .45 - bow), (-.052 + dx, y - .0045 - bow), (.0 + dx, y - .010 - bow), (.050 + dx, y - .0045 - bow), (w * .74 + dx, y - .003 + cr * .45 - bow), (w + dx, y + cr)]
            q = sp(pts, n=10); n_ = len(q); wf = lambda t: .62 + .50 * np.exp(-((t - .5) / .16) ** 2) + .38 * np.exp(-((t - .06) / .06) ** 2) + .38 * np.exp(-((t - .94) / .06) ** 2)
            cel.line(H_.Q(q), lw * .60, pal['line'], taper=(.10, .10), press=.22, seed=seed + 60, tmin=.25, wfun=wf)
            for k_, cc in ((-1, cl), (1, cr)):
                if H_._face_cos(np.array(k_ * w), np.array(y)) < .05: continue
                if cc > .010: L(cel, H_, [(k_ * (w + .012), y + cc + .004), (k_ * (w + .022), y + cc + .014), (k_ * (w + .024), y + cc + .026)], lw * .26, sk, seed + 61, taper=(.4, .7))
            if e.get('lip', 1): L(cel, H_, [(-.030 + dx, y - .084), (.004 + dx, y - .089), (.036 + dx, y - .084)], lw * .30, sk, seed + 63, taper=(.5, .5))
            L(cel, H_, [(.0 + dx, y + .082), (.002 + dx, y + .054)], lw * .16, sk, seed + 64, taper=(.5, .6), smooth_=False)
        else:
            # small, quiet openings.  a: lips part; to: rounded; ta: wider with the tongue tip; no / mu
            k = dict(speak_a=(.088, .046, .010, .020), speak_o=(.052, .040, .0, .012), a=(.092, .050, .010, .022), to=(.050, .038, -.002, .012), ta=(.094, .036, .008, .016), no=(.054, .030, -.002, .010), mu=(.074, .0, .0, .0))[m]
            w, h, c, u_ = k
            if h <= .001:                                        # 'mu': lips pressed
                L(cel, H_, [(-w, y - .002), (0, y - .008), (w, y - .002)], lw * .70, pal['line'], seed + 60, taper=(.15, .15), press=.2); L(cel, H_, [(-.045, y + .016), (0, y + .020), (.045, y + .016)], lw * .24, sk, seed + 61, taper=(.5, .5)); L(cel, H_, [(-.05, y - .050), (0, y - .058), (.05, y - .050)], lw * .28, sk, seed + 62, taper=(.5, .5))
            else:
                U = [(-w, y + c), (-w * .5, y + u_ * .9), (0, y + u_ * .6), (w * .5, y + u_ * .9), (w, y + c)]; D_ = [(w, y + c), (w * .66, y - h * .72), (0, y - h), (-w * .66, y - h * .72), (-w, y + c)]
                pol = H_.Q(np.vstack([sp(U), sp(D_)])); cel.fill(pol, hexmix('#6a3230', pal['skinD'], .35)); cel.clip(pol)
                if m in ('a', 'ta', 'speak_a'): cel.fill(H_.Q(np.vstack([sp(U), sp([(w, y + c - .012), (0, y + u_ * .6 - .020), (-w, y + c - .012)])])), pal['sclera'])
                if m == 'ta': cel.fill(H_.Q(sp([(-w * .5, y - h), (0, y - h * .35), (w * .5, y - h)], True)), hexmix('#b86a64', pal['skinS'], .3))
                cel.unclip()
                L(cel, H_, U, lw * .58, pal['line'], seed + 60, taper=(.08, .08)); L(cel, H_, D_, lw * .40, sk, seed + 61, taper=(.25, .25))
                L(cel, H_, [(-.034, y - h - .050), (0, y - h - .056), (.034, y - h - .050)], lw * .28, sk, seed + 63, taper=(.45, .45))
        cel.unclip()

    def _profile(self, cel, H_, pal, lw, e, n, seed, fill):
        X = n * .34; Z = lambda pts, x=X: H_.P3(np.array([(x, y, z) for z, y in pts]))
        cel.clip(fill)
        # socket + nose side + lip shadow
        cel.fill(Z([(.80, .15), (.70, .165), (.50, .165), (.40, .10), (.50, .085), (.62, .08), (.665, .02), (.685, -.08), (.72, -.02), (.775, .02)]), pal['skinS'])
        cel.fill(Z([(.90, -.455), (.80, -.475), (.765, -.41), (.715, -.425), (.71, -.465), (.74, -.50), (.79, -.53), (.83, -.50)], n * .05), pal['skinS'])
        cel.fill(Z([(.80, -.735), (.72, -.742), (.70, -.76), (.74, -.80), (.80, -.80)], n * .05), pal['skinS'])
        # eye
        o = e.get('open', .6); lo = e.get('low', .1)
        if e.get('closed'):
            c = sp([(.640, -.055), (.60, -.085), (.53, -.085), (.455, -.055), (.42, -.030)])
            pol = Z(np.vstack([c, (c + [0, .016])[::-1]])); cel.fill(pol, pal['line']); cel.line(Z(c), lw * .8, pal['line'], seed=seed, taper=(.2, .4))
            cel.line(Z([(.60, -.075), (.655, -.105)]), lw * .5, pal['line'], taper=(0, .8)); cel.line(Z(sp([(.645, .035), (.56, .050), (.46, .01)])), lw * .4, pal['sline'], taper=(.4, .6))
        else:
            up = sp([(.645, .034 * o * 1.5 - .004), (.612, .052 * o * 1.55), (.55, .042 * o * 1.55), (.475, .006), (.425, -.022)])
            dn = sp([(.634, -.050 + lo * .05), (.585, -.058 + lo * .06), (.505, -.046 + lo * .04), (.425, -.022)])
            pol = Z(np.vstack([up, dn[::-1]])); cel.fill(pol, pal['sclera']); cel.clip(pol)
            gx, gy = e.get('gaze', (0, 0)); ci = (.608 + gx * .0, .000 + gy * .1)
            it = np.linspace(0, 2 * math.pi, 26)[:-1]
            cel.fill(Z(np.c_[ci[0] + .027 * np.cos(it), ci[1] + .060 * np.sin(it)]), pal['iris'])
            cel.fill(Z(np.c_[ci[0] + .027 * np.cos(it), ci[1] + .018 + .044 * np.sin(it)]), pal['irisD'])
            cel.fill(Z(np.c_[ci[0] + .006 + .012 * np.cos(it), ci[1] + .028 * np.sin(it)]), pal['line'])
            cel.fill(Z(np.vstack([up, (up - [0, .024])[::-1]])), pal['scleraS'], .85)
            cel.fill(Z(np.c_[ci[0] + .010 + .007 * np.cos(it), ci[1] + .018 + .011 * np.sin(it)]), '#ffffff'); cel.unclip()
            th = np.interp(np.linspace(0, 1, len(up)), [0, .3, .7, 1], [.022, .026, .020, .004])
            cel.fill(Z(np.vstack([up, np.c_[up[:, 0], up[:, 1] + th][::-1]])), pal['line']); cel.line(Z(up), lw * .5, pal['line'], taper=(.1, .4), seed=seed)
            cel.line(Z([(.640, .034 * o * 1.5 + .008), (.668, .030 * o * 1.5 - .006)]), lw * .6, pal['line'], taper=(0, .7))
            cel.line(Z(sp([(.650, .046 * o * 1.6 + .030), (.58, .066 * o * 1.6 + .030), (.47, .036)])), lw * .36, pal['sline'], taper=(.4, .6))
            cel.line(Z(dn), lw * .40, pal['sline'], taper=(.1, .5), seed=seed + 1)
        if e.get('bags', 1) > 0: cel.line(Z(sp([(.66, -.10), (.61, -.145), (.54, -.155)])), lw * .34, pal['sline'], taper=(.3, .7), seed=seed + 2)
        # brow
        bi, bo = e.get('brow_in', 0), e.get('brow_out', 0)
        bt = sp([(.790, .118 + bi), (.72, .150 + bi * .7), (.60, .184), (.47, .188 + bo), (.34, .138 + bo)]); bb = sp([(.34, .138 + bo), (.46, .162 + bo), (.60, .146), (.72, .106 + bi * .7), (.785, .080 + bi)])
        cel.fill(Z(np.vstack([bt, bb])), pal['brow']); cel.line(Z(bb[::-1]), lw * .42, pal['line'], taper=(.2, .6), seed=seed + 3)
        # nostril, wing, mouth, hollow
        cel.line(Z(sp([(.735, -.395), (.708, -.425), (.712, -.462), (.742, -.478)]), n * .10), lw * .46, pal['sline'], taper=(.5, .2), seed=seed + 4)
        cel.line(Z(sp([(.80, -.478), (.775, -.468), (.75, -.476)]), n * .06), lw * .7, pal['line'], taper=(.3, .3), seed=seed + 5)
        m = e.get('mouth', 'flat'); up = dict(flat=-.006, calm=.004, corner=.028, smile=.034, grit=-.02).get(m, 0)
        if m.startswith('speak'):
            h = .05 if m == 'speak_a' else .04
            pol = Z(np.vstack([sp([(.80, -.655), (.75, -.65), (.69, -.645)]), sp([(.69, -.645), (.73, -.70), (.795, -.70 - h * .4)])]), n * .05); cel.fill(pol, '#4a2226')
            cel.line(Z(sp([(.80, -.655), (.75, -.65), (.69, -.645)]), n * .05), lw * .6, pal['line'], taper=(.1, .3)); cel.line(Z(sp([(.69, -.645), (.73, -.70), (.795, -.70 - h * .4)]), n * .05), lw * .45, pal['sline'], taper=(.2, .2))
        else:
            cel.line(Z(sp([(.780, -.678), (.74, -.676), (.695, -.672 + up * .4), (.655, -.672 + up)]), n * .06), lw * .72, pal['line'], taper=(.1, .25), seed=seed + 6)
            if up > .015: cel.line(Z(sp([(.645, -.60), (.628, -.645), (.632, -.69)]), n * .1), lw * .28, pal['sline'], taper=(.6, .6))
        cel.line(Z(sp([(.775, -.772), (.745, -.780), (.715, -.775)]), n * .05), lw * .36, pal['sline'], taper=(.3, .6))
        cel.line(Z(sp([(.40, -.30), (.46, -.40), (.50, -.50)]), n * .58), lw * .34, pal['sline'], taper=(.5, .7), seed=seed + 7)
        cel.unclip()

    def ears(self, cel, H_, pal, lw, which, lit=-1, seed=0):
        for s in (-1, 1):
            e = H_.ear(s)
            near = H_.near()
            isnear = (near == 0) or (near == s)
            if e['vis'] <= -.15: continue
            if (which == 'near') != isnear: continue
            o = e['outer']; cel.fill(o, pal['skin']); cel.clip(o)
            cel.fill(e['inner'], pal['skinS']); tr = e['tragus']; n_ = len(tr); tri = tr[[0, int(n_ * .22), int(n_ * .45), int(n_ * .72)]]; cel.fill(np.vstack([tri]), pal['skinD'], .55); cel.unclip()
            outline(cel, o, lw * .8, pal['line'], seed + s, press=.3)
            cel.line(e['inner'], lw * .42, pal['sline'], seed=seed + 2 + s, taper=(.3, .4)); cel.line(e['tragus'][:int(len(e['tragus']) * .5)], lw * .34, pal['sline'], seed=seed + 4 + s, taper=(.4, .5))

    # ------------------------------------------------------------ goggles
    def goggles(self, cel, H_, pal, lw, seed=0, lit=-1, crack=False):
        prof = abs(H_.yaw) > 1.12; near = H_.near()
        fl = []
        for s in (-1, 1):
            if prof and s != near: continue
            if H_._face_cos(np.array(s * .30), np.array(0.)) < .05: continue
            pts = [(.078, .050), (.128, .104), (.28, .122), (.43, .104), (.508, .040), (.506, -.066), (.440, -.146), (.29, -.172), (.155, -.152), (.092, -.085)]
            fr = np.array([(s * x, y) for x, y in sp(pts, True, n=8)]); inn = np.array([(s * (.293 + (x - .293) * .87), -.024 + (y + .024) * .82) for x, y in sp(pts, True, n=8)])
            F = H_.Q(fr, .055); I = H_.Q(inn, .062); fl.append((s, F, I))
        # side caps (they hug the socket)
        for s, F, I in fl:
            cap = np.array([(s * .505, .040), (s * .548, .022), (s * .552, -.035), (s * .505, -.055)])
            cel.fill(H_.Q(cap, .03), pal['frame'])
        # bridge
        B = np.array([(-.105, .060), (-.05, .040), (0, .048), (.05, .040), (.105, .060), (.105, .012), (.05, -.004), (0, .006), (-.05, -.004), (-.105, .012)])
        if not prof:
            z = [float(H_.z(x, y)) + .075 for x, y in B]; cel.fill(H_.P3(np.c_[B, z]), pal['frame'])
        g_ = self.gog; clear, slide, glare = g_.get('clear', 0.), g_.get('slide', -.10 * lit), g_.get('glare', 0.)
        for s, F, I in fl:
            fr_ = [F, I]; cel.fill(fr_, pal['frame'], evenodd=True); cel.fill(I, pal['lens'], 1 - .64 * clear); cel.clip(I)
            S(cel, H_, [(s * .05, .15), (s * .56, .15), (s * .54, .02), (s * .40, .045), (s * .25, .030), (s * .10, .050)], pal['lensD'], dz=.062, alpha=1 - .72 * clear)
            S(cel, H_, [(s * .12, -.12), (s * .30, -.11), (s * .46, -.09), (s * .42, -.18), (s * .2, -.18)], pal['lensH'], alpha=.22 * (1 - .5 * clear), dz=.062)
            for dx, w_, al in ((0., .105, .46), (.155, .040, .38), (-.11, .022, .26)):          # the light of the window, the same slab on both lenses, moving with the head
                x0 = s * .305 + (slide + dx) * .55
                S(cel, H_, [(x0, .16), (x0 + w_, .16), (x0 + w_ - .10, -.19), (x0 - .10, -.19)], pal['lensH'], alpha=al * (1 - .55 * clear), dz=.062)
            if glare > 0:
                S(cel, H_, [(s * .04, .16), (s * .57, .16), (s * .57, -.19), (s * .04, -.19)], pal['lensH'], alpha=.80 * glare, dz=.062)
                x0 = s * .305 + slide * .55; S(cel, H_, [(x0 - .03, .16), (x0 + .22, .16), (x0 + .12, -.19), (x0 - .13, -.19)], '#fffaf0', alpha=.85 * glare, dz=.062)
            cel.unclip()
            outline(cel, F, lw * .62, pal['line'], seed + s); outline(cel, I, lw * .30, pal['line'], seed + 3 + s)
            g = H_.Q(np.array([(s * x, y) for x, y in sp([(.115, .085), (.2, .118), (.34, .118), (.44, .098)])]), .058); cel.line(g, lw * .34, pal['lensH'], taper=(.4, .4), alpha=.8)

    # ------------------------------------------------------------ head
    def head(self, cel, H_, expr='deadpan', light='day', f=0, wind=0., seed=0, goggles=False, lit=-1, lw=None, part='all', loose=1., wdir=(1., .1, 0.), eat=0., hatch_=0.):
        """part: 'all' | 'back' (behind neck/collar) | 'front'"""
        pal = PAL[light] if isinstance(light, str) else light
        e = dict(EXPR[expr]) if isinstance(expr, str) else dict(expr)
        lw = lw or max(1.4, H_.s / 300. * 5.2); self.gog = goggles if isinstance(goggles, dict) else {}
        if part in ('all', 'back'):
            self.hair(cel, H_, pal, lw, True, f, wind, seed, wdir, loose, lit)
            self.ears(cel, H_, pal, lw, 'far', lit, seed)
            if part == 'back': return None
        P = self.face(cel, H_, pal, lw, e, lit, seed, goggles, eat=eat, hatch_=hatch_)
        self.ears(cel, H_, pal, lw, 'near', lit, seed)
        if goggles: self.goggles(cel, H_, pal, lw, seed, lit)
        self.hair(cel, H_, pal, lw, False, f, wind, seed, wdir, loose, lit)
        return P


# ==================================================================== body
class Body:
    """Upper body in the head's unit (chin-to-eye = 1).  Origin = pit of the neck; x right, y up, z to the viewer.
    b(points) projects (x,y) on the chest surface or explicit (x,y,z)."""
    def __init__(self, cx, cy, s, yaw=0., pitch=0., roll=0., persp=.05, hw=1.42, hd=.64):
        self.c = np.array([cx, cy], float); self.s, self.yaw, self.roll, self.k, self.hw, self.hd = s, yaw, roll, persp, hw, hd
        a, b = yaw, pitch; c, sn = math.cos(-roll), math.sin(-roll)
        Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
        Rx = np.array([[1, 0, 0], [0, math.cos(b), math.sin(b)], [0, -math.sin(b), math.cos(b)]])
        Rz = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
        self.R = Rz @ Rx @ Ry

    @classmethod
    def under(cls, H_, yaw=None, pitch=0., roll=None, drop=1.24, fwd=.16, **k):
        """body hung under a posed head (neck pit = head centre + R_body (0,-drop,fwd))"""
        b = cls(0, 0, H_.s, H_.yaw * .5 if yaw is None else yaw, pitch, H_.roll * .35 if roll is None else roll, **k)
        q = b.R @ np.array([0, -drop, fwd]); b.c = np.array([H_.cx + q[0] * H_.s, H_.cy - q[1] * H_.s]); return b

    def z(self, x, y):
        u = np.clip(np.abs(x) / self.hw, 0, .995); d = self.hd * (1 + .12 * np.exp(-((y + .95) / .6) ** 2)) * np.interp(y, [-4, -2.6, -.9, -.1, .3], [.85, .9, 1.0, .55, .25])
        return d * (1 - u ** 2.2) ** .5 - .12

    def P3(self, Q):
        q = np.asarray(Q, float) @ self.R.T; f = 1 / (1 - np.clip(q[..., 2] * self.k, -.5, .5))
        return np.stack([self.c[0] + q[..., 0] * f * self.s, self.c[1] - q[..., 1] * f * self.s], -1)

    def __call__(self, P, dz=0.):
        P = np.asarray(P, float)
        if P.shape[-1] == 3: return self.P3(P)
        return self.P3(np.concatenate([P, (self.z(P[..., 0], P[..., 1]) + dz)[..., None]], -1))

    def S(self, pts, corners=(), n=8, dz=0., closed=True):
        P = np.asarray(pts, float)
        return self(spl(P, closed, corners, n), dz)


def _m(pts, s): return [(p[0] * s,) + tuple(p[1:]) for p in pts]


class _NanamiBody(_Nanami):
    def neck(self, cel, H_, b, pal, lw, seed=0, lit=-1, tie=True, shadow=True):
        """column of the neck from under the jaw into the collar"""
        jl, jr = H_.jaw(-1), H_.jaw(1)
        top = [H_.P3(np.array([(q * .545, -.36, -.16)]))[0] for q in (-1, 1)]
        lo = [b(np.array([(q * .60, .10, -.02)]))[0] for q in (-1, 1)]
        mid = [(t * .45 + l * .55) + np.array([-q * .065 * H_.s, 0]) for t, l, q in zip(top, lo, (-1, 1))]
        Lq = spl([top[0], mid[0], lo[0]], n=10); Rq = spl([top[1], mid[1], lo[1]], n=10)
        pit = b(np.array([(0., -.62, .30)]))[0]
        poly = np.vstack([Lq, [pit], Rq[::-1]])
        cel.fill(poly, pal['skin']); cel.clip(poly)
        if shadow:                                                # cast by the jaw: hard, angular
            chin = H_.P3(np.array([(0, -1.0, .60)]))[0]; s = H_.s
            q = np.vstack([jl[1:-1:1], [chin], jr[1:-1][::-1]]); off = np.array([.05 * s * -lit, .30 * s])
            cel.fill(np.vstack([[top[0] + [-.4 * s, 0]], q + off + np.c_[np.zeros(len(q)), .05 * s * np.sin(np.arange(len(q)) * 1.7)], [top[1] + [.4 * s, 0]], [top[1] + [0, -s]], [top[0] + [0, -s]]]), pal['skinS'])
        q = -lit if lit else 1
        a, c = (Rq, Lq) if q > 0 else (Lq, Rq)
        cel.fill(np.vstack([a, (a + (c - a) * np.linspace(.22, .34, len(a))[:, None])[::-1]]), pal['skinS'])
        cel.unclip()
        cel.line(Lq, lw * .8, pal['line'], seed=seed, taper=(.2, .2)); cel.line(Rq, lw * .8, pal['line'], seed=seed + 1, taper=(.2, .2))
        # sternomastoid
        for q_, side in ((-1, Lq), (1, Rq)):
            p0 = side[4] + (pit - side[4]) * .10; p1 = pit + np.array([q_ * .10 * H_.s, -.10 * H_.s])
            cel.line(spl([p0, (p0 + p1) / 2 + np.array([q_ * .03 * H_.s, 0]), p1], n=8), lw * .40, pal['sline'], taper=(.5, .5), seed=seed + 3)
        # throat
        t0 = H_.P3(np.array([(0., -1.0, .42)]))[0]; m_ = t0 * .42 + pit * .58
        cel.line(spl([m_ + [-.05 * H_.s, -.02 * H_.s], m_ + [0, .05 * H_.s], m_ + [.05 * H_.s, -.02 * H_.s]], n=6), lw * .34, pal['sline'], taper=(.5, .5), seed=seed + 5)
        return pit

    # ------------------------------------------------------------ suit
    FRONT = [(.44, .28, -.04), (.78, .08, -.10), (1.22, -.14, -.12), (1.70, -.36, -.10), (1.80, -.66, .04), (1.69, -1.08, .15), (1.50, -1.6, .12), (1.34, -2.3, .06), (1.27, -2.85, .02), (1.38, -4.2, .02), (1.50, -5.8, .02)]
    SLEEVE = [(1.64, -.34, -.12), (1.86, -.45, -.10), (2.04, -.72, -.04), (2.16, -1.26, -.02), (2.24, -2.05, .0), (2.18, -2.6, .02), (2.22, -3.35, .06), (2.14, -4.9, .1), (1.46, -4.9, .1), (1.44, -3.25, .08), (1.49, -2.45, .06), (1.46, -2.0, .06), (1.48, -1.6, .12), (1.67, -1.08, .16), (1.78, -.66, .05)]

    def collar_back(self, cel, b, pal):
        P = b.S([(-.66, .14, -.12), (-.60, .34, -.22), (-.42, .46, -.32), (0, .52, -.38), (.42, .46, -.32), (.60, .34, -.22), (.66, .14, -.12), (0, -.1, -.1)], corners=(0, 6)); cel.fill(P, pal['suitS']); outline(cel, P, b.s / 300. * 4.2, pal['line'], 3)

    def draw_suit(self, cel, H_, b, pal, lw, seed=0, lit=-1, loosen=0., arms=(1, 1), neck=True):
        A, ln, r = b, pal['line'], np.random.default_rng(seed + 31)
        sd = -lit if lit else 1                                                    # shadow side
        order = (1, -1) if b.yaw >= 0 else (-1, 1)                                 # far side first
        self.collar_back(cel, b, pal)
        cel.fill(b.S([(-.47, .16, -.2), (-.40, .40, -.26), (0, .46, -.30), (.40, .40, -.26), (.47, .16, -.2), (0, -.2, -.1)], corners=(0, 4)), pal['shirtS'])
        if neck: self.neck(cel, H_, b, pal, lw, seed, lit)
        # far sleeve
        def sleeve(q):
            P = A.S(_m(self.SLEEVE, q), corners=(0, 7, 8, 14)); cel.fill(P, pal['suit']); cel.clip(P)
            if q == sd:
                cel.fill(A.P3(np.array(_m([(1.2, -.9, .1), (1.64, -1.02, .15), (1.72, -1.6, .1), (1.60, -2.1, .1), (1.78, -2.42, .1), (1.56, -2.7, .1), (1.62, -3.4, .1), (1.5, -4.5, .1), (1.2, -4.5, .1)], q))), pal['suitS'])
            else:
                cel.fill(A.P3(np.array(_m([(1.2, -1.2, .1), (1.50, -1.35, .15), (1.52, -2.0, .1), (1.48, -2.6, .1), (1.46, -4.5, .1), (1.2, -4.5, .1)], q))), pal['suitS'])
            # elbow: zig-zag fold
            for pts in ([(1.36, -2.20), (1.62, -2.30), (1.82, -2.22), (1.60, -2.40), (1.40, -2.42)], [(2.14, -1.98), (1.92, -2.18), (1.80, -2.36), (1.98, -2.26), (2.12, -2.20)], [(1.36, -2.72), (1.56, -2.62), (1.74, -2.66), (1.56, -2.76), (1.38, -2.84)], [(1.48, -1.22), (1.66, -1.10), (1.80, -.88), (1.72, -1.22), (1.54, -1.40)]):
                cel.fill(A.P3(np.array(_m([p + (.1,) for p in pts], q))), pal['suitS'] if q != sd else pal['suitD'])
            cel.unclip()
            outline(cel, P, lw, ln, seed + 60 + q, press=.4)
            for pts in ([(1.38, -2.22), (1.58, -2.33), (1.84, -2.22)], [(2.12, -1.96), (1.94, -2.16), (1.78, -2.40)], [(1.38, -2.80), (1.58, -2.63), (1.76, -2.68)], [(1.46, -1.30), (1.66, -1.10), (1.82, -.86)]):
                cel.line(A.P3(spl(np.array(_m([p + (.1,) for p in pts], q)), n=8)), lw * .40, ln, taper=(.2, .7), seed=seed + int(10 * abs(pts[0][1])))
        if arms[0 if order[0] < 0 else 1]: sleeve(order[0])
        # ---- shirt
        sv = [(-.56, .26, -.06), (-.30, .02, .16), (0, -.08, .26), (.30, .02, .16), (.56, .26, -.06), (.92, -.3), (.80, -1.2), (.10, -2.7), (-.10, -2.7), (-.80, -1.2), (-.92, -.3)]
        Sv = np.vstack([A.P3(np.array([q for q in sv if len(q) == 3])), A(np.array([q for q in sv if len(q) == 2]))]); cel.fill(Sv, pal['shirt']); cel.clip(Sv)
        cel.fill(A(np.array([(sd * .16, -.3), (sd * .34, -.42), (sd * .30, -1.1), (sd * .12, -2.4), (sd * .7, -2.4), (sd * .9, -.2)])), pal['shirtS'])
        for q in (-1, 1):
            for y0 in (-.62, -1.15, -1.62): cel.fill(A(np.array([(q * .20, y0 - .16), (q * .38, y0 + .02), (q * .60, y0 + .04), (q * .38, y0 - .07)])), pal['shirtS'])
        cel.unclip()
        # ---- tie
        dy = -.22 * loosen; tl = .05 * loosen
        kn = A.S([(-.15 + tl, -.02 + dy), (0, -.06 + dy), (.15 + tl, -.02 + dy), (.118 + tl, -.36 + dy), (0 + tl, -.39 + dy), (-.118 + tl, -.36 + dy)], corners=(0, 2, 3, 5), dz=.05)
        bl = A.S([(-.100 + tl, -.22 + dy), (.100 + tl, -.22 + dy), (.125 + tl, -.46 + dy), (.20 + tl, -1.2), (.245, -2.0), (.255, -2.8), (-.245, -2.8), (-.235, -2.0), (-.185 + tl, -1.2), (-.125 + tl, -.46 + dy)], corners=(0, 1, 5, 6), dz=.04)
        if loosen > .02:
            for q_ in (-1, 1):                                     # the band round the neck, slack
                lp = A.P3(spl(np.array([(q_ * .14 + tl, -.02 + dy, .40), (q_ * .02 + tl, -.05 + dy, .44), (q_ * (.10 + .04 * loosen), .06 + dy * .55, .34), (q_ * .24, .16, .16), (q_ * .40, .26, -.02), (q_ * .47, .20, -.02), (q_ * .34, .06 + dy * .15, .18), (q_ * (.24 + .03 * loosen), -.04 + dy * .6, .34)]), True, corners=(0, 1, 4, 5), n=7)); cel.fill(lp, pal['tie']); cel.clip(lp)
                for k in range(7): an_ = np.linspace(0, 6.283, 8)[:-1]; c_ = A.P3(np.array([(q_ * (.12 + .05 * k), -.02 + dy * (1 - k / 7.) + .035 * k, .40 - .05 * k)]))[0]; cel.fill(c_ + np.c_[np.cos(an_), np.sin(an_) * 1.2] * b.s * .026, pal['spot'])
                if q_ == sd: cel.fill(lp, pal['tieS'], .55)
                cel.unclip(); outline(cel, lp, lw * .55, ln, 77 + q_)
        for P, nm in ((bl, 0), (kn, 1)):
            cel.fill(P, pal['tie']); cel.clip(P)
            rr = np.random.default_rng(77)
            for k in range(70):
                y = .02 - 2.8 * (k + .5) / 70 + rr.normal(0, .02); x = ((k % 4) - 1.5) * .105 + rr.normal(0, .022); rad = rr.uniform(.022, .042); n_ = 9
                an = np.linspace(0, 2 * math.pi, n_, endpoint=False); rd = rad * (1 + rr.normal(0, .22, n_))
                blob = spl(np.c_[x + tl + rd * np.cos(an) * 1.0, y + rd * np.sin(an) * 1.25], True, n=4)
                cel.fill(A(blob, .05), pal['spot'])
            sx = .05 * sd
            cel.fill(A(np.array([(sx, .1), (sx + sd * .02, -.4), (sx + sd * .05, -1.2), (sx + sd * .06, -2.9), (sd * .5, -2.9), (sd * .5, .1)]), .05), pal['tieS'], .62)
            if nm == 0: cel.fill(A(np.array([(-.3, -.36 + dy), (.3, -.36 + dy), (.16, -.50 + dy), (.02, -.44 + dy), (-.14, -.52 + dy)]), .05), pal['spot'], .55)
            cel.unclip(); outline(cel, P, lw * .62, ln, seed + 70 + nm)
        cel.line(A(np.array([(-.05 + tl, -.40 + dy), (-.02 + tl, -.56 + dy), (-.03 + tl, -.74)]), .05), lw * .36, ln, taper=(.1, .8))
        # ---- shirt collar wings
        op = .10 * loosen
        for q in order:
            W_ = [(q * .455, .30, -.06), (q * .30, .14, .16), (q * (.115 + op), -.03, .30), (q * (.20 + op), -.20, .34), (q * (.40 + op), -.42, .36), (q * .56, -.10, .16), (q * .555, .12, .0)]
            P = A.P3(spl(np.array(W_), True, corners=(0, 2, 4), n=8)); cel.fill(P, pal['shirt'])
            cel.clip(P)
            if q == sd: cel.fill(A.P3(np.array([(q * .6, .4, 0), (q * .40, .10, .1), (q * .44, -.16, .3), (q * .38, -.5, .36), (q * .7, -.3, .2)])), pal['shirtS'])
            else: cel.fill(A.P3(np.array([(q * .6, .4, 0), (q * .49, .10, .1), (q * .53, -.12, .25), (q * .42, -.45, .36), (q * .7, -.3, .2)])), pal['shirtS'])
            cel.unclip(); outline(cel, P, lw * .72, ln, seed + 80 + q)
            cel.line(A.P3(spl(np.array([(q * .40, .18, .06), (q * .36, .04, .2), (q * (.30 + op), -.22, .36)]), n=6)), lw * .28, pal['shirtS'] if lit else ln, taper=(.3, .5))
        # ---- jacket fronts (wearer's left, picture right, closes on top)
        for q in (-1, 1):
            f2 = [(q * .70, .0), (q * .72, -.35), (q * .64, -1.1), (q * .46, -1.75), (q * .035, -2.55), (-.05, -3.1), (-.07, -4.4), (-.07, -5.8)]
            o3 = np.array(_m(self.FRONT, q)); inner = A(spl(np.array(f2), n=8))
            P = np.vstack([A.P3(spl(o3, corners=(3, 6), n=8)), inner[::-1]])
            cel.fill(P, pal['suit']); cel.clip(P)
            if q == sd:
                cel.fill(np.vstack([A.P3(np.array(_m([(1.9, -.35, 0), (1.52, -.62, .2), (1.38, -1.05, .3), (1.20, -1.40, .4), (1.24, -1.9, .3), (1.04, -2.5, .35), (1.12, -3.1, .3), (.98, -4.5, .3), (1.6, -4.5, 0), (1.6, -1.5, 0)], q)))]), pal['suitS'])
            else:
                cel.fill(A.P3(np.array(_m([(1.8, -.9, .1), (1.46, -1.15, .22), (1.34, -1.62, .22), (1.26, -2.4, .18), (1.20, -3.2, .18), (1.24, -4.5, .1), (1.6, -4.5, 0), (1.6, -1.2, 0)], q))), pal['suitS'])
            # drag folds toward the button
            for pts in ([(q * .12, -2.60), (q * .52, -2.36), (q * .94, -2.28), (q * .56, -2.52), (q * .18, -2.72)], [(q * .10, -2.70), (q * .42, -2.92), (q * .80, -3.22), (q * .46, -3.06), (q * .12, -2.84)], [(q * 1.30, -1.62), (q * 1.04, -1.78), (q * .80, -2.02), (q * 1.06, -1.94), (q * 1.28, -1.84)]):
                cel.fill(A(np.array(pts)), pal['suitS'] if not (q == sd and abs(pts[0][0]) > 1) else pal['suitD'])
            # compression at the buttoned waist: the cloth buckles once on each side, deeper on the shadow side
            cp_ = [(q * 1.34, -2.52), (q * .96, -2.70), (q * .54, -2.72), (q * .26, -2.66), (q * .56, -2.84), (q * .98, -2.86), (q * 1.32, -2.74)]; cel.fill(A(np.array(cp_)), pal['suitS'] if q != sd else pal['suitD'])
            if q == sd: cel.fill(A(np.array([(q * 1.30, -2.98), (q * .98, -3.04), (q * .66, -3.00), (q * .98, -3.12), (q * 1.30, -3.12)])), pal['suitD'])
            cel.unclip()
            cel.line(A(spl(np.array([(q * 1.33, -2.50), (q * .96, -2.69), (q * .54, -2.71), (q * .24, -2.66)]), n=9)), lw * .55, ln, taper=(.1, .6), press=.4, seed=seed + 17)
            outline(cel, P, lw * .95, ln, seed + 90 + q, press=.4)
            for pts in ([(q * .12, -2.60), (q * .50, -2.36), (q * .92, -2.28)], [(q * .10, -2.72), (q * .44, -2.94), (q * .82, -3.24)], [(q * 1.28, -1.64), (q * 1.02, -1.80), (q * .80, -2.02)]):
                cel.line(A(spl(np.array(pts), n=8)), lw * .40, ln, taper=(.1, .75), seed=seed + 7)
            # dart
            cel.line(A(np.array([(q * .84, -1.78), (q * .80, -2.5), (q * .83, -3.3)])), lw * .32, ln, taper=(.4, .4))
        # breast pocket on the wearer's left
        pk = A(np.array([(.80, -1.16), (1.16, -1.10), (1.16, -1.21), (.80, -1.27)])); cel.fill(pk, pal['suit']); outline(cel, pk, lw * .38, ln, 5)
        # ---- lapels
        for q in (-1, 1):
            Lp = [(q * .44, .30, -.05), (q * .62, .02, .12), (q * .69, -.35, .36), (q * .63, -1.05, .52), (q * .47, -1.70, .57), (q * .26, -2.22, .57), (q * .03, -2.56, .52), (q * .40, -2.20, .56), (q * .60, -1.84, .53), (q * .80, -1.36, .44), (q * 1.07, -.82, .30), (q * .79, -.66, .38), (q * 1.00, -.52, .26), (q * .90, -.16, .12), (q * .66, .18, -.02), (q * .52, .32, -.08)]
            P = A.P3(spl(np.array(Lp), True, corners=(0, 6, 10, 11, 12, 15), n=8))
            # shadow the lapel throws on the jacket
            off = np.array([.035 * sd, .05]) * b.s
            cel.fill(P + off, pal['suitS']) if q == sd else cel.fill(P + off * np.array([-.0, .8]), pal['suitS'])
            cel.fill(P, pal['suit']); cel.clip(P)
            if q == sd: cel.fill(A.P3(np.array([(q * .40, .4, 0), (q * .66, .05, .12), (q * .74, -.35, .34), (q * .70, -.9, .44), (q * .56, -1.5, .5), (q * .2, -2.3, .52), (q * -.1, -2.6, .52), (q * .3, -1.2, .5)])), pal['suitS'])
            cel.unclip(); outline(cel, P, lw * .82, ln, seed + 100 + q, press=.35)
            cel.line(A.P3(spl(np.array([(q * .79, -.66, .38), (q * .72, -.50, .37), (q * .665, -.30, .33)]), n=6)), lw * .5, ln, taper=(.05, .5))
            # stitching breaks
            st = spl(np.array([(q * .98, -.82, .33), (q * .74, -1.36, .46), (q * .40, -2.00, .51)]), n=10)
            for k in range(0, len(st) - 1, 2): cel.line(A.P3(st[k:k + 2]), lw * .22, ln, taper=(.1, .1), alpha=.55)
        # button
        bt = np.array([(.10 + .055 * math.cos(t), -2.66 + .055 * math.sin(t)) for t in np.linspace(0, 6.283, 14)[:-1]]); bp = A(spl(bt, True, n=4), .02); cel.fill(bp, pal['suitD']); outline(cel, bp, lw * .36, ln, 3)
        # near sleeve
        if arms[0 if order[1] < 0 else 1]: sleeve(order[1])
        # shoulder seam
        for q in (-1, 1): cel.line(A.P3(spl(np.array(_m([(1.69, -.36, -.1), (1.80, -.66, .05), (1.70, -1.06, .16)], q)), n=8)), lw * .48, ln, taper=(.1, .5), seed=seed + 8)

    # ------------------------------------------------------------ white shirt, open collar, rolled sleeves
    SSLEEVE = [(1.66, -.18, -.12), (1.92, -.31, -.10), (2.12, -.63, -.04), (2.22, -1.14, .0), (2.28, -1.75, .02), (2.24, -2.36, .04), (2.28, -2.74, .06), (1.48, -2.82, .10), (1.46, -2.36, .08), (1.45, -1.95, .06), (1.48, -1.58, .12), (1.64, -1.05, .16), (1.72, -.57, .05)]

    def draw_shirt(self, cel, H_, b, pal, lw, seed=0, lit=-1, f=0, wind=0., arms=(1, 1), neck=True, open_=1.0):
        A, ln = b, pal['line']; sd = -lit if lit else 1
        order = (1, -1) if b.yaw >= 0 else (-1, 1); lc = pal.get('wline', hexmix(ln, pal['whiteD'], .35))
        g = [wind * (1 + .5 * math.sin(f * .33 + k * 1.9)) for k in range(6)]
        # collar stand behind the neck
        P = A.S([(-.66, .16, -.10), (-.60, .36, -.20), (-.40, .50, -.30), (0, .56, -.36), (.40, .50, -.30), (.60, .36, -.20), (.66, .16, -.10), (0, -.1, -.1)], corners=(0, 6)); cel.fill(P, pal['whiteS']); outline(cel, P, lw * .8, ln, 3)
        # skin of the chest (seen through the open collar)
        sk = np.vstack([A.P3(np.array([(-.62, .22, -.08)])), A(np.array([(-.70, -.4), (-.42, -1.2), (0, -1.75), (.42, -1.2), (.70, -.4)])), A.P3(np.array([(.62, .22, -.08)]))])
        cel.fill(sk, pal['skin'])
        if neck: self.neck(cel, H_, b, pal, lw, seed, lit)
        cel.clip(sk)
        cel.fill(A(np.array([(sd * .05, -.42), (sd * .20, -.60), (sd * .12, -.95), (sd * .03, -1.5), (sd * .5, -1.5), (sd * .9, -.2)])), pal['skinS'])
        cel.unclip()
        for q in (-1, 1):
            cel.line(A(spl(np.array([(q * .07, -.46), (q * .22, -.36), (q * .42, -.30), (q * .60, -.28)]), n=8)), lw * .46, pal['sline'], taper=(.1, .6), seed=seed + 11 + q)
            cel.line(A(spl(np.array([(q * .10, -.36), (q * .08, -.43), (q * .03, -.49)]), n=8)), lw * .40, pal['sline'], taper=(.4, .4), seed=seed + 13 + q)
        cel.line(A(np.array([(0, -.78), (sd * .01, -1.1), (0, -1.5)])), lw * .34, pal['sline'], taper=(.5, .6))

        def sleeve(q):
            k = g[0] * (1 if q > 0 else -1)
            pts = [(p[0] + (.05 * k if 2 <= i <= 6 else 0), p[1], p[2]) for i, p in enumerate(self.SSLEEVE)]
            # forearm first (skin), the rolled cuff sits over it
            fa = [(1.46, -2.5, .06), (1.40, -2.9, .10), (1.46, -3.4, .14), (1.58, -4.2, .2), (2.02, -4.2, .2), (2.10, -3.4, .14), (2.08, -2.9, .1), (2.0, -2.5, .06)]
            P = A.P3(spl(np.array(_m(fa, q)), n=8)); cel.fill(P, pal['skin']); cel.clip(P)
            cel.fill(A.P3(np.array(_m([(1.3, -2.5, .1), (1.62, -2.6, .1), (1.60, -3.2, .1), (1.70, -4.4, .1), (1.3, -4.4, .1)] if q != sd else [(2.3, -2.5, .1), (1.86, -2.6, .1), (1.80, -3.2, .1), (1.72, -4.4, .1), (2.3, -4.4, .1)], q))), pal['skinS']); cel.unclip()
            cel.line(P, lw * .9, ln, taper=(.02, .02), seed=seed + 21 + q)
            P = A.P3(spl(np.array(_m(pts, q)), True, corners=(0, 6, 7, 12), n=8)); cel.fill(P, pal['white']); cel.clip(P)
            if q == sd:
                cel.fill(A.P3(np.array(_m([(1.3, -.7, .1), (1.62, -.84, .15), (1.74, -1.30, .1), (1.60, -1.72, .1), (1.82, -2.05, .1), (1.58, -2.34, .1), (1.70, -2.8, .1), (1.2, -2.9, .1)], q))), pal['whiteS'])
                cel.fill(A.P3(np.array(_m([(1.3, -1.2, .1), (1.50, -1.28, .15), (1.46, -1.9, .1), (1.52, -2.4, .1), (1.2, -2.6, .1)], q))), pal['whiteD'])
            else:
                cel.fill(A.P3(np.array(_m([(1.2, -1.2, .1), (1.52, -1.36, .15), (1.46, -1.8, .1), (1.54, -2.1, .1), (1.44, -2.8, .1), (1.2, -2.8, .1)], q))), pal['whiteS'])
            for pp in ([(1.40, -1.36), (1.62, -1.20), (1.84, -.92), (1.74, -1.26), (1.52, -1.48)], [(1.34, -1.82), (1.60, -1.86), (1.88, -1.74), (1.62, -1.96), (1.36, -1.98)], [(2.12, -1.62), (1.92, -1.86), (1.74, -2.14), (1.96, -1.98), (2.12, -1.80)], [(1.38, -2.20), (1.60, -2.16), (1.80, -2.26), (1.58, -2.28), (1.40, -2.34)], [(1.62, -.30), (1.74, -.62), (1.72, -1.02), (1.82, -.66), (1.74, -.36)]):
                cel.fill(A.P3(np.array(_m([p + (.1,) for p in pp], q))), pal['whiteS'] if q != sd else pal['whiteD'])
            cel.unclip(); outline(cel, P, lw * .80, ln if q == sd else lc, seed + 30 + q, press=.5)
            for pp in ([(1.38, -1.36), (1.62, -1.20), (1.86, -.90)], [(1.36, -1.84), (1.60, -1.88), (1.90, -1.72)], [(2.10, -1.62), (1.92, -1.86), (1.72, -2.16)], [(1.40, -2.22), (1.60, -2.17), (1.82, -2.28)]):
                cel.line(A.P3(spl(np.array(_m([p + (.1,) for p in pp], q)), n=8)), lw * .38, lc, taper=(.15, .75), seed=seed + int(10 * abs(pp[0][1])))
            # rolled cuff: two folded bands
            cf = [(1.32, -2.52, .08), (1.70, -2.60, .12), (2.13, -2.50, .04), (2.16, -2.84, .04), (1.74, -2.95, .14), (1.30, -2.86, .1)]
            P = A.P3(spl(np.array(_m(cf, q)), True, corners=(0, 2, 3, 5), n=8)); cel.fill(P, pal['white']); cel.clip(P)
            cel.fill(A.P3(np.array(_m([(1.2, -2.4, .1), (1.52, -2.5, .1), (1.56, -2.72, .1), (1.48, -3, .1), (1.2, -3, .1)] if q != sd else [(2.3, -2.4, .1), (1.94, -2.5, .1), (1.90, -2.72, .1), (1.96, -3, .1), (2.3, -3, .1)], q))), pal['whiteS']); cel.unclip()
            outline(cel, P, lw * .8, ln, seed + 40 + q)
            cel.line(A.P3(spl(np.array(_m([(1.32, -2.70, .1), (1.70, -2.78, .14), (2.14, -2.68, .06)], q)), n=8)), lw * .42, lc, taper=(.1, .3))
        if arms[0 if order[0] < 0 else 1]: sleeve(order[0])
        # ---- body of the shirt: two fronts
        for q in (-1, 1):
            v = open_
            o3 = [(q * .52, .30, -.08), (q * .88, .10, -.10), (q * 1.28, -.05, -.12), (q * 1.70, -.20, -.11), (q * 1.76, -.57, .05), (q * 1.64, -1.05, .15), (q * 1.48, -1.60, .12), (q * (1.38 + .05 * g[1] * q), -2.3, .05), (q * (1.33 + .07 * g[2] * q), -3.1, .0), (q * (1.43 + .08 * g[3] * q), -4.6, .0)]
            inn = [(q * .56, .10), (q * .50 * v + q * .02, -.45), (q * .30 * v, -1.05), (q * .04, -1.62), (-.06, -2.6), (-.06, -4.4)]
            P = np.vstack([A.P3(spl(np.array(o3), corners=(3, 6), n=8)), A(spl(np.array(inn), n=8))[::-1]]); cel.fill(P, pal['white']); cel.clip(P)
            if q == sd:
                cel.fill(A.P3(np.array(_m([(1.9, -.2, 0), (1.34, -.40, .25), (1.18, -.80, .35), (1.30, -1.15, .3), (1.02, -1.52, .42), (1.10, -2.1, .38), (.84, -2.5, .45), (1.0, -3.2, .4), (.8, -4.5, .4), (1.6, -4.5, 0), (1.6, -1.5, 0)], q))), pal['whiteS'])
                cel.fill(A.P3(np.array(_m([(1.5, -1.25, .1), (1.28, -1.5, .25), (1.18, -2.0, .25), (1.2, -2.6, .2), (1.10, -3.2, .25), (1.2, -4.5, .1), (1.6, -4.5, 0)], q))), pal['whiteD'])
            else:
                cel.fill(A.P3(np.array(_m([(1.8, -1.0, .1), (1.40, -1.2, .22), (1.24, -1.62, .26), (1.18, -2.4, .24), (1.10, -3.0, .24), (1.16, -4.5, .2), (1.6, -4.5, 0), (1.6, -1.2, 0)], q))), pal['whiteS'])
            folds = [[(1.44, -.36), (1.16, -.62), (.84, -.98), (1.10, -.78), (1.36, -.58)], [(1.42, -1.30), (1.10, -1.44), (.72, -1.72), (1.06, -1.62), (1.36, -1.50)], [(.98, -.02), (.86, -.42), (.80, -.92), (.94, -.52), (1.06, -.14)], [(1.26, -2.0), (.96, -2.24), (.62, -2.62), (.94, -2.46), (1.22, -2.24)],
                     [(.18, -1.72), (.34, -2.2), (.36, -2.9), (.26, -2.4)], [(1.20, -2.9), (.90, -3.0), (.56, -3.3), (.9, -3.2), (1.18, -3.1)]]
            folds += [[(1.60, -.22), (1.40, -.40), (1.12, -.50), (1.34, -.52), (1.54, -.46)], [(.60, -1.2), (.72, -1.75), (.66, -2.3), (.80, -1.8), (.74, -1.2)], [(1.02, -1.75), (.82, -2.1), (.50, -2.3), (.80, -2.22), (1.0, -1.98)], [(.28, -1.30), (.46, -1.52), (.80, -1.62), (.50, -1.62)]]
            for k, pp in enumerate(folds):
                cel.fill(A(np.array([(q * x, y) for x, y in pp])), pal['whiteS'] if not (q == sd and k in (1, 3, 6)) else pal['whiteD'])
            cel.unclip(); outline(cel, P, lw * .74, ln if q == sd else lc, seed + 50 + q, press=.5)
            for k, pp in enumerate(folds):
                cel.line(A(spl(np.array([(q * x, y) for x, y in pp[:3]]), n=8)), lw * (.42 if k < 3 else .34), lc, taper=(.1, .75), seed=seed + k)
        # placket and buttons
        cel.line(A(spl(np.array([(.16, -1.70), (.14, -2.6), (.15, -4.4)]), n=8)), lw * .34, lc, taper=(.2, .1))
        for y0 in (-1.78, -2.5, -3.2):
            bp = A(spl(np.array([(.045 + .045 * math.cos(t), y0 + .045 * math.sin(t)) for t in np.linspace(0, 6.283, 12)[:-1]]), True, n=4), .02); cel.fill(bp, pal['whiteS']); outline(cel, bp, lw * .3, lc, 3)
        # ---- open collar wings: the one on the lee side lies flat, the windward one is lifted and shows its underside
        for q in order:
            k = max(0., g[4]) * (1. if q == -1 else .25) if wind else 0.
            edge = [(q * .60, .32, -.10), (q * .46, .10, .10), (q * .42, -.16, .30), (q * .40, -.44, .42)]
            tip = (q * (.64 + .10 * k), -.80 + .34 * k, .50 + .30 * k); outr = [(q * (.80 + .10 * k), -.44 + .20 * k, .32 + .22 * k), (q * .765, -.08, .10), (q * .68, .18, -.06)]
            W_ = edge + [tip] + outr
            P = A.P3(spl(np.array(W_), True, corners=(0, 3, 4), n=8))
            cel.fill(P + np.array([.035 * sd, .07 + .05 * k]) * b.s, pal['whiteS'], 1.)
            if k > .25:                                            # underside
                U_ = A.P3(spl(np.array([edge[3], tip, (q * .66, -.50, .36), (q * .52, -.50, .40)]), True, corners=(0, 1), n=6)); cel.fill(U_, pal['whiteD'])
            cel.fill(P, pal['white']); cel.clip(P)
            if q == sd: cel.fill(A.P3(np.array([(q * .5, .5, -.1), (q * .60, .10, .1), (q * .54, -.20, .3), (q * .60, -.46, .42), (q * .50, -.8, .5), (q * .2, -.8, .46), (q * .2, .2, .1)])), pal['whiteS'])
            else: cel.fill(A.P3(np.array([(q * .56, .5, -.1), (q * .53, .06, .1), (q * .47, -.24, .34), (q * .46, -.8, .5), (q * .2, -.8, .46), (q * .2, .2, .1)])), pal['whiteS'])
            cel.unclip(); outline(cel, P, lw * .78, ln if q == sd else hexmix(ln, lc, .4), seed + 60 + q, press=.5)
            cel.line(A.P3(spl(np.array([(q * .62, .10, .04), (q * .61, -.14, .20), (q * .62, -.40, .40)]), n=8)), lw * .28, lc, taper=(.3, .5))
            st = spl(np.array([(q * .70, .02, .06), (q * (.735 + .08 * k), -.40 + .15 * k, .32), (q * (.63 + .08 * k), -.70 + .30 * k, .48)]), n=10)
            for j in range(0, len(st) - 1, 2): cel.line(A.P3(st[j:j + 2]), lw * .18, lc, taper=(.1, .1), alpha=.7)
        if arms[0 if order[1] < 0 else 1]: sleeve(order[1])
        for q in (-1, 1): cel.line(A.P3(spl(np.array(_m([(1.68, -.20, -.1), (1.76, -.58, .06), (1.66, -1.03, .17)], q)), n=8)), lw * .34, lc, taper=(.1, .5), seed=seed + 8)

    # ------------------------------------------------------------ seen from behind: white shirt, rolled sleeves, sea wind from the left
    def draw_back(self, cel, H_, b, pal, lw, f=0, wind=1., seed=0, lit=-1, turn=0.):
        """b = Body whose +z faces us (his back).  wind 0..1 fills the shirt on the lee (right) side; f drives it (call it on twos).  turn 0..1 pulls the cloth as he turns"""
        A, ln = b, pal['line']; sd = -lit if lit else 1; lc = pal.get('wline', hexmix(ln, pal['whiteD'], .35)); w_ = lambda k, sp=.42: math.sin(f * sp + k * 1.7)
        f3 = lambda pts, col, al=1.: cel.fill(A.P3(np.array([p if len(p) == 3 else (p[0], p[1], float(A.z(p[0], p[1])) + .02) for p in pts])), col, al)
        ll = lambda pts, w, c=lc, tp=(.15, .7), sd_=0: cel.line(A.P3(spl(np.array([p if len(p) == 3 else (p[0], p[1], float(A.z(p[0], p[1])) + .02) for p in pts]), n=9)), lw * w, c, taper=tp, press=.3, seed=seed + sd_)
        # ---- neck
        nr_ = 1 if H_.yaw < 0 else -1; kk_ = float(np.clip((abs(H_.yaw) - .75) / .75, 0, 1))
        top = [H_.P3(np.array([(nr_ * .34, -.46, .06)]))[0], H_.P3(np.array([(nr_ * .34 * (1 - kk_), -.24, -.50 - .13 * kk_)]))[0]] if abs(H_.yaw) < 1.75 else [np.array([H_.cx, H_.cy]) + np.array([(-.40 if q < 0 else .44) + .03 * math.sin(H_.yaw), (.08 if (q < 0) == (H_.yaw < 0) else .26)]) * H_.s for q in (-1, 1)]
        if .6 < abs(H_.yaw) < 1.75:                                # skin behind the ear and under the hair: nothing may be seen through
            cel.fill(H_.P3(np.array([(nr_ * .60, .02, -.06), (nr_ * .60, -.42, -.06), (nr_ * .40, -.66, -.14), (nr_ * .10, -.60, -.50), (0, -.18, -.80), (nr_ * .30, .10, -.60)])), pal['skinS'])
        top = sorted(top, key=lambda p_: p_[0]); lo = [A(np.array([(q * .56, .20, -.02)]))[0] for q in (-1, 1)]
        Lq = spl([top[0], top[0] * .4 + lo[0] * .6 + [.05 * H_.s, 0], lo[0]], n=8); Rq = spl([top[1], top[1] * .4 + lo[1] * .6 + [-.05 * H_.s, 0], lo[1]], n=8); P = np.vstack([Lq, Rq[::-1]]); cel.fill(P, pal['skin']); cel.clip(P)
        mid_ = (Lq + Rq) / 2; n_ = len(Lq); U = H_.s
        cel.fill(np.vstack([Rq + [40, 0], (Rq + (Lq - Rq) * (np.linspace(.50, .30, n_) + .05 * np.sin(np.arange(n_) * 1.1))[:, None])[::-1]]), pal['skinS'])
        cel.fill(np.vstack([mid_[:1] + [-.10 * U, -.5 * U], mid_[4:5] + [-.16 * U, .0], [mid_[6] + [-.10 * U, .04 * U], mid_[8] + [.0, -.06 * U], mid_[6] + [.16 * U, .06 * U]], Rq[5:6], Rq[:1] + [30, -90]]), pal['skinS'])                # hair over the nape
        cel.fill(np.vstack([Lq[-5:], lo[0:1] + np.array([0, 60.]), lo[1:2] + np.array([0, 60.]), Rq[-5:][::-1], [mid_[-4] + [.14 * U, .05 * U], mid_[-3] + [0, -.02 * U], mid_[-4] + [-.16 * U, .06 * U]]]), pal['skinD'], .55)     # collar
        cel.unclip()
        cel.line(spl([mid_[3] + [-.03 * U, 0], mid_[n_ // 2] + [-.015 * U, 0], mid_[-3]], n=7), lw * .30, pal['sline'], taper=(.5, .6))
        for q_, e_ in ((-1, Lq), (1, Rq)):                                                       # the two cords of the trapezius
            cel.line(spl([mid_[2] + [q_ * .13 * U, 0], mid_[n_ // 2] * .62 + e_[n_ // 2] * .38, mid_[-2] * .25 + e_[-2] * .75], n=8), lw * .40, pal['sline'], taper=(.45, .4), seed=q_ + 5)
            cel.line(spl([e_[3] * .75 + mid_[3] * .25, e_[n_ // 2] * .86 + mid_[n_ // 2] * .14, e_[-3] * .96 + mid_[-3] * .04], n=7), lw * .26, pal['sline'], taper=(.5, .6), seed=q_ + 7)
        cel.line(Lq, lw * .6, lc, taper=(.2, .2)); cel.line(Rq, lw * .85, ln, taper=(.2, .2))
        # ---- sleeves
        def sleeve(q):
            wv_ = lambda i: wind * (.10 * math.sin(i * 1.5 - f * .62) if q > 0 else .030 * math.sin(i * 1.7 - f * .62))            # a ripple runs down the lee sleeve, the windward one is pressed to the arm
            pts = [(p[0] + (wv_(i) + (.11 if q > 0 else -.05) * wind) * (2 <= i <= 6), p[1], p[2]) for i, p in enumerate(self.SSLEEVE)]
            fa = [(1.50, -2.5, .06), (1.44, -2.95, .10), (1.50, -3.6, .14), (1.62, -4.5, .2), (1.64, -5.6, .2), (2.04, -5.6, .2), (2.06, -4.5, .2), (2.16, -3.6, .14), (2.14, -2.95, .1), (2.06, -2.5, .06)]
            P = A.P3(spl(np.array(_m(fa, q)), n=8)); cel.fill(P, pal['skin']); cel.clip(P); cel.fill(A.P3(np.array(_m([(2.4, -2.5, .1), (1.86, -2.6, .1), (1.78, -3.2, .1), (1.80, -5.8, .1), (2.4, -5.8, .1)] if q == sd else [(1.3, -2.5, .1), (1.60, -2.6, .1), (1.58, -3.2, .1), (1.72, -5.8, .1), (1.3, -5.8, .1)], q))), pal['skinS']); cel.unclip(); cel.line(P, lw * .85, ln if q == sd else lc, taper=(.02, .02))
            P = A.P3(spl(np.array(_m(pts, q)), True, corners=(0, 6, 7, 12), n=8)); cel.fill(P, pal['white']); cel.clip(P)
            if q == sd:
                cel.fill(A.P3(np.array(_m([(1.3, -.5, .1), (1.72, -.70, .15), (1.86, -1.2, .1), (1.70, -1.6, .1), (1.92, -2.0, .1), (1.66, -2.4, .1), (1.8, -2.9, .1), (1.2, -2.9, .1)], q))), pal['whiteS']); cel.fill(A.P3(np.array(_m([(1.3, -1.2, .1), (1.60, -1.3, .15), (1.56, -1.9, .1), (1.62, -2.4, .1), (1.3, -2.9, .1)], q))), pal['whiteD'])
            else: cel.fill(A.P3(np.array(_m([(1.2, -1.3, .1), (1.66, -1.4, .15), (1.60, -1.9, .1), (1.68, -2.2, .1), (1.56, -2.9, .1), (1.2, -2.9, .1)], q))), pal['whiteS'])
            for pp in ([(1.52, -1.40), (1.76, -1.24), (2.02, -.96), (1.90, -1.32), (1.64, -1.54)], [(2.26, -1.66), (2.04, -1.90), (1.84, -2.20), (2.08, -2.02), (2.26, -1.86)], [(1.50, -2.24), (1.74, -2.20), (1.96, -2.30), (1.72, -2.34), (1.52, -2.40)]): cel.fill(A.P3(np.array(_m([p + (.1,) for p in pp], q))), pal['whiteS'] if q != sd else pal['whiteD'])
            cel.unclip(); outline(cel, P, lw * .80, ln if q == sd else lc, seed + 30 + q, press=.5)
            for pp in ([(1.50, -1.40), (1.76, -1.24), (2.04, -.94)], [(2.24, -1.66), (2.04, -1.90), (1.82, -2.22)], [(1.50, -2.26), (1.74, -2.21), (1.98, -2.32)]): cel.line(A.P3(spl(np.array(_m([p + (.1,) for p in pp], q)), n=8)), lw * .36, lc, taper=(.15, .75))
            cf = [(1.44, -2.60, .08), (1.84, -2.68, .12), (2.29, -2.58, .04), (2.32, -2.92, .04), (1.88, -3.03, .14), (1.42, -2.94, .1)]; P = A.P3(spl(np.array(_m(cf, q)), True, corners=(0, 2, 3, 5), n=8)); cel.fill(P, pal['white']); cel.clip(P)
            cel.fill(A.P3(np.array(_m([(2.5, -2.4, .1), (2.08, -2.5, .1), (2.04, -2.8, .1), (2.1, -3.1, .1), (2.5, -3.1, .1)] if q == sd else [(1.3, -2.5, .1), (1.62, -2.6, .1), (1.66, -2.8, .1), (1.58, -3.1, .1), (1.3, -3.1, .1)], q))), pal['whiteS']); cel.unclip(); outline(cel, P, lw * .7, ln if q == sd else lc, seed + 40 + q); cel.line(A.P3(spl(np.array(_m([(1.44, -2.78, .1), (1.84, -2.86, .14), (2.30, -2.76, .06)], q)), n=8)), lw * .36, lc, taper=(.1, .3))
        sleeve(1); sleeve(-1)
        # ---- back of the shirt; the lee side billows in two travelling scallops
        w3 = lambda k: math.sin(f * .62 - k * 1.25); B1, B2, B3 = (.14 + .10 * w3(0)) * wind, (.24 + .17 * w3(1)) * wind, (.34 + .24 * w3(2)) * wind
        tr = [(-1.30, -3.6, .0), (-1.36, -4.6, .0), (-1.42, -6.4, .1), (-.10, -6.4, .2), (0, -5.2, .3), (.10, -6.4, .2), (1.46, -6.4, .1), (1.40, -4.6, .0), (1.32, -3.6, .0)]; P = A.P3(np.array(tr)); cel.fill(P, pal['suit']); cel.clip(P)
        cel.fill(A.P3(np.array([(sd * .3, -3.6, .2), (sd * .4, -4.8, .2), (sd * .62, -5.4, .2), (sd * .5, -6.6, .2), (sd * 2, -6.6, 0), (sd * 2, -3.6, 0)])), pal['suitS']); cel.unclip(); outline(cel, P, lw * .7, ln if sd > 0 else lc, 71); cel.line(A.P3(np.array([(0, -4.4, .3), (.02, -5.3, .3)])), lw * .4, lc, taper=(.1, .4))
        left = [(-.50, .34, -.10), (-.92, .16, -.10), (-1.32, -.02, -.12), (-1.72, -.20, -.11), (-1.78, -.57, .05), (-1.66, -1.05, .15), (-1.48, -1.60, .12), (-1.35, -2.3, .05), (-1.30, -3.1, .0), (-1.34, -3.95 + .04 * w3(0), .0)]
        right = [(.50, .34, -.10), (.92, .16, -.10), (1.32, -.02, -.12), (1.72, -.20, -.11), (1.78, -.57, .05), (1.66, -1.05, .15), (1.52 + B1 * .5, -1.60, .12), (1.44 + B1, -2.05, .05), (1.40 + B1 * .5 + B2 * .3, -2.5, .0), (1.46 + B2, -3.0, .0), (1.42 + B2 * .7 + B3 * .2, -3.45, .0), (1.56 + B3, -3.80 + .16 * wind * max(0, w3(2)), .0)]
        hem = [(1.30 + B3 * .8, -4.02 + .18 * wind * w3(2.4), .15), (.98 + B3 * .4, -4.30 + .12 * wind * w3(3.2), .35), (.55 + B3 * .2, -4.40 + .08 * wind * w3(4.0), .55), (.05, -4.46 + .05 * wind * w3(4.6), .62), (-.50, -4.40 + .04 * w3(5.2), .55), (-1.0, -4.20, .3)]
        if wind > .3:                                              # the lee tail turns over and shows its inside
            fl = [right[-1], (1.30 + B3 * 1.3, -4.16 + .22 * wind * w3(2.2), -.1), (.92 + B3 * .7, -4.44 + .14 * wind * w3(3.), .0), hem[1]]; U_ = A.P3(spl(np.array(fl + [hem[0]]), True, corners=(0, 3), n=7)); cel.fill(U_, pal['whiteD']); outline(cel, U_, lw * .6, ln if sd > 0 else lc, 73)
        P = np.vstack([A.P3(spl(np.array(left), corners=(3, 6), n=8)), A.P3(spl(np.array(hem[::-1]), n=8)), A.P3(spl(np.array(right), corners=(3, 6), n=8))[::-1]]); cel.fill(P, pal['white']); cel.clip(P)
        # planes: shoulder blades catch the light, the spine and the lee half fall into shade
        f3([(sd * 1.95, -.2, .0), (sd * 1.36, -.34), (sd * 1.16, -.74), (sd * 1.28, -1.10), (sd * .96, -1.44), (sd * 1.06, -1.9), (sd * .74, -2.4), (sd * .92, -2.9), (sd * .66, -3.5), (sd * .84, -5.0), (sd * 2.6, -5., 0.), (sd * 2.4, -1.5, 0.)], pal['whiteS'])
        f3([(sd * 1.72, -1.05, .1), (sd * 1.36, -1.30), (sd * 1.22, -1.75), (sd * 1.44, -2.1, .05), (sd * 1.16, -2.5), (sd * 1.40, -2.9, .05), (sd * 1.12, -3.5), (sd * 1.36, -4.0, 0.), (sd * 1.2, -5.), (sd * 2.6, -5., 0.), (sd * 2.2, -1.2, 0.)], pal['whiteD'])
        f3([(-.10, -.62), (.06, -.60), (.10 + .04 * turn, -1.4), (.02 + .18 * turn, -2.4), (.12 + .30 * turn, -3.4), (-.02 + .30 * turn, -5.), (-.22 + .16 * turn, -5.), (-.10 + .14 * turn, -3.4), (-.18 + .06 * turn, -2.4), (-.08, -1.4)], pal['whiteS'])
        f3([(-sd * 1.80, -1.0, .1), (-sd * 1.46, -1.2), (-sd * 1.34, -1.7), (-sd * 1.22, -2.4), (-sd * 1.16, -3.2), (-sd * 1.22, -5.), (-sd * 1.8, -5., 0.), (-sd * 1.8, -1.2, 0.)], pal['whiteS'])
        # folds: everything is dragged from the windward shoulder toward the lee hip
        sh_ = .05 * w_(3., .3) * wind; T_ = turn
        folds = [[(-1.42, -1.00), (-.92 + .10 * T_, -1.26 - .06 * T_), (-.42 + .42 * T_, -1.70 + sh_ - .22 * T_), (-.90, -1.46), (-1.38, -1.22)], [(-1.30, -1.62), (-.55 + .14 * T_, -2.10 - .10 * T_), (.34 + .50 * T_, -2.62 + sh_ - .26 * T_), (-.50 + .12 * T_, -2.38), (-1.26, -1.98)], [(-1.22, -2.34), (-.86, -2.48), (-.56, -2.80), (-.88, -2.66), (-1.18, -2.52)], [(-1.22, -2.86), (-.38, -3.32), (.62, -4.2 - sh_), (-.30, -3.72), (-1.16, -3.28)], [(-1.18, -3.9), (-.6, -4.2), (.1, -4.8), (-.55, -4.5), (-1.14, -4.2)],
                 [(1.60, -.95, .12), (1.20, -1.20), (.70, -1.62), (1.12, -1.44), (1.54, -1.20, .1)], [(1.50, -2.0, .05), (1.06, -2.24), (.56, -2.70), (1.0, -2.50), (1.46, -2.24, .05)], [(1.50 + B2, -3.0, .0), (1.06, -3.14), (.60, -3.55), (1.02, -3.40), (1.46, -3.26, .0)], [(-.62, -.10), (-.70, -.5), (-.62, -1.0), (-.54, -.56)], [(.74, -.16), (.82, -.6), (.70, -1.1), (.66, -.6)], [(-.72, -4.5), (-.80, -4.0), (-.62, -3.5), (-.66, -4.0), (-.54, -4.5)], [(.30 + B3 * .3, -4.6), (.16, -4.0), (.34, -3.4), (.34, -4.0), (.58 + B3 * .3, -4.6)], [(1.0 + B3 * .5, -4.4), (.86, -3.9), (1.10 + B2 * .5, -3.4), (1.04, -3.9), (1.24 + B3 * .6, -4.2)]]
        for k, pp in enumerate(folds): f3(pp, pal['whiteD'] if (k in (5, 6, 7) and sd > 0) else pal['whiteS'])
        cel.unclip(); outline(cel, P, lw * .78, ln if sd > 0 else lc, seed + 50, press=.5)
        for k, pp in enumerate(folds): ll(pp[:3], (.46, .40, .28, .44, .34)[k] if k < 5 else .36, sd_=k)
        ll([(-1.70, -.46, .06), (-.95, -.60), (0, -.66), (.95, -.60), (1.70, -.46, .06)], .40, tp=(.1, .1)); ll([(-.07, -.66), (-.08 + .05 * T_, -1.6), (-.05 + .22 * T_, -2.8)], .34, tp=(.05, .6)); ll([(.07, -.66), (.07 + .05 * T_, -1.6), (.10 + .22 * T_, -2.8)], .34, tp=(.05, .6))
        if T_ > .05:                                             # the twist: cloth is pulled from the near shoulder across the spine to the far hip
            for a0, b0, c0 in (((-1.10, -.40), (-.50, -.86), (.10, -1.60)), ((-.70, -.30), (-.18, -.62), (.30, -1.16)), ((.50, -2.2), (.86, -2.7), (1.16, -3.4))): ll([a0, b0, (c0[0] * (.5 + .5 * T_), c0[1])], .42 * T_, tp=(.3, .7))
        for q in (-1, 1): ll([(q * 1.70, -.20, -.1), (q * 1.78, -.58, .06), (q * 1.67, -1.03, .16)], .36, tp=(.1, .5))
        # ---- collar: a folded band, its lee corner flutters
        k = wind * (.5 + .5 * w_(4., .8)); C = [(-.60, .30, -.10), (-.42, .44, .02), (0, .50, .08), (.42, .44, .02), (.60, .30, -.10), (.74 + .06 * k, -.02 + .10 * k, .05 + .1 * k), (.36, .04, .20), (0, .06, .24), (-.36, .04, .20), (-.72, -.04, .05)]
        P = A.P3(spl(np.array(C), True, corners=(0, 4, 5, 9), n=8)); cel.fill(P + np.array([.02, .07]) * b.s, pal['whiteS']); cel.fill(P, pal['white']); cel.clip(P); cel.fill(A.P3(np.array([(sd * .20, .6, .1), (sd * .30, .2, .2), (sd * .24, -.1, .2), (sd * .9, -.1, 0.), (sd * .9, .6, -.1)])), pal['whiteS']); cel.unclip(); outline(cel, P, lw * .78, ln if sd > 0 else lc, seed + 60, press=.5)
        ll([(-.50, .34, .0), (0, .40, .12), (.50, .34, .0)], .28, tp=(.3, .3))

    def head_back(self, cel, H_, pal, lw, f=0, wind=0., seed=0, wdir=(1., .1, 0.), loose=1., lit=-1, part='all'):
        """the head turned away (|yaw| > 1.75): sliver of cheek, ear, hair"""
        if part == 'hair':
            self.hair(cel, H_, pal, lw, False, f, wind, seed, wdir, loose, lit); self._ear_back(cel, H_, pal, lw, seed); return
        fc = H_.face(top=.60); cel.fill(fc, pal['skin']); cel.clip(fc); cel.fill(fc + [(.06 if H_.yaw < 0 else -.06) * H_.s, .02 * H_.s], pal['skinS'], .55); cel.unclip()
        outline(cel, fc, lw * .7, hexmix(pal['line'], pal['skinS'], .25), seed + 3, press=.4)
        if part == 'face': return
        self.hair(cel, H_, pal, lw, False, f, wind, seed, wdir, loose, lit); self._ear_back(cel, H_, pal, lw, seed)

    def _ear_back(self, cel, H_, pal, lw, seed=0):
        for s in (-1, 1):
            e = H_.ear(s); o = e['outer']; vis = (H_.R @ np.array([s * .9, 0, -.35]))[2]; wd = np.ptp(o[:, 0]) / H_.s
            if vis > .30 and wd > .10:
                cel.fill(o, pal['skin']); cel.clip(o); cel.fill(e['inner'], pal['skinS']); cel.unclip(); outline(cel, o, lw * .72, pal['line'], seed + s, press=.3); cel.line(e['inner'], lw * .34, pal['sline'], taper=(.3, .4))

    # ------------------------------------------------------------ one suit sleeve with elbow, cuff and hand
    def suit_arm(self, cel, b, S_, E_, W_, pal, lw, up=1, lit=-1, seed=0, hand=None, watch=False, root=True, bulk=1., cuff=.16):
        """S_,E_,W_ shoulder / elbow / wrist in body units (3-D); up as in Todo.arm (+1 for the picture-right arm).  hand(cel, wrist_px, elbow_px, px_per_unit)"""
        ln = pal['line']; Q3 = np.array([S_, E_, W_], float); p = b.P3(Q3); dz = (Q3 @ b.R.T)[:, 2]; fz = b.s / (1 - np.clip(dz * b.k, -.5, .5)) * bulk; S2, E2, W2 = p
        def seg(a, c, fa, fc, P):
            u = c - a; L_ = np.hypot(*u); u = u / (L_ + 1e-6); n = np.array([u[1], -u[0]]) * up; P = np.asarray(P, float); r = fa + (fc - fa) * np.clip(P[:, 0:1], 0, 1); return a + u * L_ * P[:, 0:1] + n * r * P[:, 1:2]
        U_ = lambda P: seg(S2, E2, fz[0], fz[1], P); F_ = lambda P: seg(E2, W2, fz[1], fz[2], P); Ld = np.array([lit * .8 if lit else 0, -.6])
        side = lambda a, c: np.array([(c - a)[1], -(c - a)[0]]) / (np.hypot(*(c - a)) + 1e-6) * up @ Ld
        UO, UI = [(-.125, -.12), (-.135, .10), (-.075, .33), (.05, .45), (.28, .475), (.62, .43), (.90, .42), (1.02, .40)], [(-.02, -.30), (.12, -.40), (.34, -.41), (.62, -.375), (.88, -.36), (1.02, -.33)]
        FO, FI = [(-.06, .38), (.10, .43), (.34, .40), (.66, .345), (.90, .31), (1.0, .305)], [(-.06, -.32), (.08, -.385), (.36, -.36), (.66, -.325), (.90, -.295), (1.0, -.29)]
        Pu = U_(np.vstack([spl(np.array(UO), n=7), spl(np.array(UI), n=7)[::-1]])); Pf = F_(np.vstack([spl(np.array(FO), n=7), spl(np.array(FI), n=7)[::-1]])); an = np.linspace(0, 6.283, 22); disc = E2 + np.c_[np.cos(an), np.sin(an)] * fz[1] * .375
        if hand is not None: hand(cel, W2, E2, fz[2])
        # bare wrist, watch, shirt cuff come out of the sleeve before the sleeve is laid over them
        if watch:
            w0 = F_([(1.06, .255), (1.30, .235), (1.30, -.235), (1.06, -.25)]); cel.fill(w0, pal['skin']); cel.clip(w0); cel.fill(F_([(1.0, -.05), (1.4, -.02), (1.4, -.4), (1.0, -.4)]) if side(E2, W2) >= 0 else F_([(1.0, .05), (1.4, .02), (1.4, .4), (1.0, .4)]), pal['skinS']); cel.unclip(); cel.line(F_([(1.02, .262), (1.30, .240)]), lw * .85, ln, taper=(0, .1)); cel.line(F_([(1.02, -.255), (1.30, -.24)]), lw * .85, ln, taper=(0, .1))
            st = F_([(1.145, .262), (1.265, .252), (1.265, -.252), (1.145, -.262)]); cel.fill(st, hexmix('#4a2f25', pal['skinD'], .25)); outline(cel, st, lw * .6, ln, 3)
            c0 = F_([(1.205, .02)])[0]; u = (W2 - E2) / np.hypot(*(W2 - E2)); n = np.array([u[1], -u[0]]) * up; R = fz[2] * .185; ring = lambda k: np.array([c0 + u * R * k * .92 * math.cos(t) + n * R * k * math.sin(t) for t in np.linspace(0, 6.283, 26)[:-1]])
            cel.fill(ring(1.), pal['whiteS']); outline(cel, ring(1.), lw * .7, ln, 4); cel.fill(ring(.80), hexmix(pal['white'], '#fff6e0', .35)); cel.line(np.vstack([ring(.80), ring(.80)[:1]]), lw * .35, ln, taper=(0, 0))
            for k in range(12): t = k * math.pi / 6; cel.line(np.array([c0 + (u * .92 * math.cos(t) + n * math.sin(t)) * R * .62, c0 + (u * .92 * math.cos(t) + n * math.sin(t)) * R * .74]), lw * .24, ln, taper=(0, 0))
            cel.line(np.array([c0, c0 + n * R * .58]), lw * .42, ln, taper=(0, .2)); cel.line(np.array([c0, c0 - n * R * .40 + u * R * .12]), lw * .5, ln, taper=(0, .2)); cel.fill(np.array([c0 - u * R * .5 + n * R * .2, c0 - u * R * .1 + n * R * .55, c0 + u * R * .1 + n * R * .4, c0 - u * R * .4 - n * R * .1]), '#ffffff', .5)
        if cuff > 0:
            cf = F_([(.95, .292), (1.0 + cuff * .5, .285), (1.0 + cuff * .5, -.275), (.95, -.285)]); cel.fill(cf, pal['shirt']); cel.clip(cf); cel.fill(F_([(.9, -.05), (1.2, -.05), (1.2, -.4), (.9, -.4)]) if side(E2, W2) >= 0 else F_([(.9, .05), (1.2, .05), (1.2, .4), (.9, .4)]), pal['shirtS']); cel.unclip(); outline(cel, cf, lw * .72, ln, seed + 4)
        cel.fill(disc, pal['suit']); cel.fill(Pu, pal['suit']); cel.clip(Pu)
        sg = 1 if side(S2, E2) >= 0 else -1
        cel.fill(U_([(-.3, -.9 * sg), (-.05, -.04 * sg), (.14, -.14 * sg), (.34, -.02 * sg), (.52, -.16 * sg), (.70, -.02 * sg), (.86, -.14 * sg), (1.1, .02 * sg), (1.1, -.9 * sg)]), pal['suitS']); cel.fill(U_([(.1, -.9 * sg), (.24, -.30 * sg), (.5, -.32 * sg), (.75, -.25 * sg), (1.1, -.2 * sg), (1.1, -.9 * sg)]), pal['suitD'], .55)
        for pp in ([(.70, -.40), (.80, -.10), (.98, .18), (.90, -.12), (.84, -.38)], [(.84, -.40), (.94, -.06), (1.04, .10), (1.02, -.2)], [(.04, -.30), (.14, .0), (.30, .20), (.22, -.06), (.16, -.34)]): cel.fill(U_(pp), pal['suitS' if sg > 0 else 'suitD'])
        cel.unclip(); cel.fill(Pf, pal['suit']); cel.clip(Pf); s2 = 1 if side(E2, W2) >= 0 else -1
        cel.fill(F_([(-.2, -.9 * s2), (-.05, .0), (.12, -.10 * s2), (.30, .02 * s2), (.46, -.10 * s2), (.66, .0), (.84, -.08 * s2), (1.1, .0), (1.1, -.9 * s2)]), pal['suitS']); cel.fill(F_([(.0, -.9 * s2), (.12, -.28 * s2), (.4, -.28 * s2), (.7, -.2 * s2), (1.1, -.18 * s2), (1.1, -.9 * s2)]), pal['suitD'], .55)
        for pp in ([(-.10, -.34), (.06, -.10), (.28, .04), (.14, -.16), (.02, -.38)], [(-.04, -.40), (.16, -.24), (.42, -.18), (.20, -.30)], [(-.06, .36), (.08, .18), (.26, .12), (.10, .30)]): cel.fill(F_(pp), pal['suitS' if s2 > 0 else 'suitD'])
        cel.unclip()
        big = max(union_contour([Pu, disc, Pf], close=7), key=len); uu = E2 - S2; L_ = np.hypot(*uu); uu = uu / L_; tt = (big - S2) @ uu / L_; nn = np.array([uu[1], -uu[0]]) * up; vv = (big - S2) @ nn / fz[0]
        skip = ((tt < -.115) | ((tt < .10) & (vv < .08))) if root else np.zeros(len(big), bool); k0 = int(np.argmax(skip)) if skip.any() else 0; big = np.roll(big, -k0, 0); skip = np.roll(skip, -k0); i = 0
        while i < len(big):
            if skip[i]: i += 1; continue
            j = i
            while j < len(big) and not skip[j]: j += 1
            if j - i > 8: cel.line(big[i:j], lw, ln, taper=(.06, .05), press=.4, seed=seed + i % 7, tmin=.2)
            i = j
        for pts, w in (([(.70, -.38), (.82, -.08), (1.0, .18)], .46), ([(.86, -.38), (.96, -.06), (1.06, .08)], .36), ([(.02, -.30), (.14, .0), (.32, .20)], .40)): cel.line(U_(spl(np.array(pts, float), n=8)), lw * w, ln, taper=(.1, .75), press=.3, seed=seed + int(w * 100))
        for pts, w in (([(-.10, -.32), (.08, -.10), (.30, .04)], .46), ([(-.02, -.38), (.18, -.24), (.44, -.18)], .36), ([(.0, .34), (.12, .18), (.28, .12)], .34)): cel.line(F_(spl(np.array(pts, float), n=8)), lw * w, ln, taper=(.1, .75), press=.3, seed=seed + 5 + int(w * 100))
        for k in range(3):
            q = F_([(.94 - .10 * k, -.22 * s2)])[0]; R = fz[2] * .042; o = np.array([q + R * np.array([math.cos(t), math.sin(t)]) for t in np.linspace(0, 6.283, 12)[:-1]]); cel.fill(o, pal['suitD']); outline(cel, o, lw * .34, ln, 8 + k)
        return p, fz

    def bust(self, cx, cy, s, yaw=0., pitch=0., roll=0., byaw=None, bpitch=0., broll=None, expr='deadpan', light='day', f=0, wind=0., seed=0, lit=-1, goggles=True, outfit='suit', loosen=0., loose=.4, lw=None, w=W, h=H, wdir=(1., .1, 0.), head=True, ss=2):
        pal = PAL[light] if isinstance(light, str) else light
        lw = lw or max(1.4, s / 300. * 5.2)
        c = Cel(w, h, ss=ss); H_ = self.pose(cx, cy, s, yaw, pitch, roll)
        b = Body.under(H_, byaw, bpitch, broll)
        if head: self.head(c, H_, expr, light, f, wind, seed, goggles, lit, lw, 'back', loose, wdir)
        if outfit == 'suit': self.draw_suit(c, H_, b, pal, lw, seed, lit, loosen)
        else: self.draw_shirt(c, H_, b, pal, lw, seed, lit, f, wind)
        if head: self.head(c, H_, expr, light, f, wind, seed, goggles, lit, lw, 'front', loose, wdir)
        return c.layer(h, w)


Nanami = _NanamiBody()
