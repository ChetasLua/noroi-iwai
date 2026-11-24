"""AOI TODO.
H = Todo.pose(cx, cy, s, yaw, pitch, roll);  Todo.head(cel, H, expr, light, f, wind, seed, lit, lw, part)
    expr: smug focus warcry weep joy grin;  light: day night glory ember (blue = night, beach = day);  tears run with f
b = TBody.under(H, byaw, bpitch, broll, drop=1.28, fwd=.10, persp=.05..0.12)   (persp + bpitch = low angle)
Todo.torso(cel, b, pal, lw, lit, sash) bare trunk, sash, trousers;  Todo.neck;  Todo.arm(cel, b, shoulder, elbow, wrist, pal, lw, side, up, lit, hand=fn)
Todo.figure(cel, H, b, pose, expr, light, ...)  pose: spread windup near clap crossed flex hang (POSES) or a dict; Todo.mix_pose(a, b, t) inbetweens
Todo.bust(cx, cy, s, yaw, pitch, roll, byaw, bpitch, pose=..., expr=...) -> layer (about .9 s).  hand3 = jj/hands.py seated on a wrist."""
from chars.ntkit import *
from chars.nanami import Body, _m

PAL = dict(
    day=dict(skin='#e2ba8c', skinS='#a9785a', skinD='#6f4a40', skinH='#f6dcb4', hair='#26242b', hairS='#131218', hairH='#5d5a6c', brow='#1a171b', white='#eceaf0', whiteS='#a9a8bd', pants='#2f3347', pantsS='#1b1d2c', pantsH='#4a5070',
             sclera='#f2ece6', scleraS='#b9a8a2', iris='#3b2c26', scar='#b4705c', line=INK, sline='#3a1d16', mouth='#5c1f22', tongue='#c0605c', teeth='#f4f0ea', tear='#e9f8ff', tearS='#9fd2ee', cord='#17161a', lip='#b5795f'),
    night=dict(skin='#c79b80', skinS='#7c5462', skinD='#482f4e', skinH='#f5e0cf', hair='#1c1a24', hairS='#0d0c13', hairH='#5a6288', brow='#141218', white='#c9cde0', whiteS='#7a80a6', pants='#252a42', pantsS='#131628', pantsH='#4a588c',
             sclera='#e2dce2', scleraS='#8f86a6', iris='#2e2230', scar='#9c5a5c', line='#100e16', sline='#2c1626', mouth='#3c1320', tongue='#9c4a56', teeth='#e8e4ec', tear='#dff4ff', tearS='#7cb8e6', cord='#0e0d14', lip='#93605c'),
    glory=dict(skin='#f0bd86', skinS='#b0684f', skinD='#6c3847', skinH='#fff0c4', hair='#2b2230', hairS='#150f1a', hairH='#8d5f86', brow='#1c1418', white='#fff3e2', whiteS='#d3a8b4', pants='#35304e', pantsS='#1d1830', pantsH='#6b5a92',
               sclera='#fff4e4', scleraS='#d0a79a', iris='#402a24', scar='#b8624e', line='#1d0f16', sline='#4a1c1a', mouth='#611c24', tongue='#cf625c', teeth='#fff8ec', tear='#f2fbff', tearS='#a8dcf6', cord='#1a1218', lip='#c07a5c'),
    ember=dict(skin='#8f4c30', skinS='#42201e', skinD='#1e0e13', skinH='#ffb765', hair='#150d10', hairS='#0a0608', hairH='#8a3e1e', brow='#0c0709', white='#8c5a44', whiteS='#43252a', pants='#241a22', pantsS='#120c12', pantsH='#6a3324',
               sclera='#c99a78', scleraS='#5a3430', iris='#1c0f10', scar='#6a2a20', line='#0b0709', sline='#1d0a0c', mouth='#2a0c10', tongue='#7a3028', teeth='#d8b090', tear='#ffd9a0', tearS='#c46a30', cord='#080507', lip='#5a2a20'))
PAL['blue'] = PAL['night']; PAL['beach'] = PAL['day']

EXPR = dict(
    smug=dict(eye='closed', brow_in=-.030, brow_out=.045, mouth='smirk', tears=0),
    focus=dict(eye='open', open=.62, gaze=(0, 0), brow_in=-.050, brow_out=.030, mouth='shut', tears=0),
    warcry=dict(eye='open', open=1.25, gaze=(0, 0), iris=.72, brow_in=-.075, brow_out=.060, mouth='roar', tears=0),
    weep=dict(eye='weep', open=.5, gaze=(0, .1), brow_in=.062, brow_out=-.030, mouth='cry', tears=1),
    joy=dict(eye='squeeze', brow_in=.050, brow_out=-.020, mouth='cry', tears=1),
    grin=dict(eye='open', open=.75, gaze=(0, 0), brow_in=-.045, brow_out=.050, mouth='grin', tears=0))

HAIR = Hair(yc=.24, ry=.80, rx=.715, rzf=.83, rzb=.97, zc=-.06, low=.95, box=.80)


def hairline(th):
    a = abs(th)
    t = [0, 12, 34, 44, 52, 62, 72, 79, 83, 90, 104, 116, 130, 150, 180]
    v = [.34, .365, .385, .30, .27, .20, .02, -.20, -.20, .0, .05, -.04, -.24, -.46, -.50]
    return float(np.interp(a, t, v))


class _Todo:
    KW = dict(a=[(-1.0, .235), (-.968, .330), (-.90, .420), (-.80, .520), (-.69, .605), (-.58, .645), (-.44, .652), (-.28, .655), (-.12, .660), (.12, .662), (.35, .665), (.6, .635), (.8, .50), (.93, .31), (1.0, .05)],
              f=[(-1.0, .69), (-.92, .78), (-.8, .77), (-.64, .765), (-.46, .76), (-.28, .745), (-.06, .735), (.06, .76), (.2, .83), (.4, .80), (.6, .72), (.8, .54), (.93, .32), (1.0, .05)],
              p=2.35, nose=(.125, .0, -.40, -.47, .185), ear=(.668, .10, -.36, -.10), cheek=.038, socket=.050)
    PROF = [(.69, .74), (.77, .58), (.80, .40), (.835, .25), (.850, .16), (.79, .06), (.775, .0), (.81, -.10), (.865, -.25), (.905, -.36), (.915, -.40), (.895, -.45), (.80, -.48), (.782, -.515), (.795, -.60),
            (.822, -.640), (.815, -.665), (.780, -.680), (.812, -.705), (.815, -.735), (.785, -.775), (.775, -.81), (.815, -.87), (.835, -.93), (.80, -.985), (.72, -1.0)]
    ZJ = [.70, .69, .61, .46, .26, .08, -.04, -.09, -.11, -.10]

    def make(self): return FHead(zj=self.ZJ, prof=self.PROF, **self.KW)
    def pose(self, cx, cy, s, yaw=0., pitch=0., roll=0., persp=.12): return self.make().pose(cx, cy, s, yaw, pitch, roll, persp)

    # ------------------------------------------------------------ hair
    def _locks(self, f=0, wind=0., seed=0, wdir=(1., .2, 0.)):
        out = []; r = np.random.default_rng(50 + seed)
        def add(path, w, off=None, wa=.0, k=0, **kw):
            wv = wind_vec(f, wind * wa, k, wdir) if wind and wa else None
            d = lock(None_H[0], HAIR, path, w, kw.pop('prof', None), off, wind=wv); d.update(kw); out.append(d)
        # everything is pulled back, tight, to the knot at the crown (theta 180, v .78)
        for i, th in enumerate([-30, -18, -6, 6, 18, 30]):
            add([(th, hairline(th) - .01, .01), (th * .95, .56, .035), (th * .8, .78, .04), (th * .4 + 180 * .0, .97, .05)], .24, k=i, sh=.2, t0=.0, lw=.55, strands=3, prof=[(0, .7), (.3, 1), (.8, .7), (1, .4)], hi=(.22, .52) if abs(th) < 22 else 0)
        for g in (-1, 1):
            for i, (th, v) in enumerate([(44, .31), (56, .24), (66, .12), (74, -.04), (98, .02), (112, -.06)]):
                add([(g * th, v, .006), (g * (th + 16), v + .20, .03), (g * (th + 40), v + (.62 - v) * .55 + .08, .035), (g * 165, .80, .04)], .24, k=10 + i, sh=.25, t0=.0, lw=.55, strands=3, side=g, prof=[(0, .6), (.3, 1), (.8, .7), (1, .4)], hi=(.15, .40) if i in (1, 2) else 0)
            add([(g * 76, .04, .004), (g * 78, -.08, .008), (g * 79.5, -.19, .003)], .10, k=20, sh=0, t0=.5, lw=.6, strands=0, prof=[(0, 1), (.65, .8), (1, .2)])
            for i, (th, v) in enumerate([(126, -.18), (142, -.36), (162, -.46), (179, -.48)]):
                add([(g * th, v, .006), (g * (th + 6), v + .32, .03), (g * (th + 12), v + .75, .035), (g * 176, .72, .04)], .30, k=30 + i, sh=.25, t0=.0, lw=.55, strands=3, side=g, prof=[(0, .5), (.3, 1), (.8, .7), (1, .4)])
        # wisps that slipped the knot: temples and nape, short
        for i, (th, v, dx, dy) in enumerate([(-66, .20, -.05, -.16), (70, .10, .05, -.20), (-120, -.06, -.09, -.12), (124, -.10, .09, -.10), (168, -.44, .03, -.14), (-160, -.42, -.05, -.12)]):
            add([(th, v + .05, .01), (th * 1.02, v, .035), (th * 1.03, v - .06, .05)], .035, [(0, 0, 0), (dx * .3, dy * .4, 0), (dx, dy, .02)], wa=.8, k=50 + i, sh=0, t0=0, lw=.6, strands=0, thin=1, prof=[(0, .4), (.4, 1), (1, 0)])
        return out

    def _knot(self, H_, f=0, wind=0., seed=0, wdir=(1., .2, 0)):
        """topknot: a bound tail that bursts into curls, in 3-D around the crown"""
        r = np.random.default_rng(5 + seed); C0 = np.array([0., 1.17, -.10]); out = []
        wv = wind_vec(f, wind, 3, wdir) if wind else np.zeros(3)
        for k in range(17):
            a = 2 * math.pi * k / 17 + r.normal(0, .12); e = [.15, .55, .95][k % 3] + r.normal(0, .08); R = r.uniform(.30, .46)
            d = np.array([math.cos(a) * math.cos(e), math.sin(e) + .12, math.sin(a) * math.cos(e) * .8]); side = np.cross(d, [0, 1, 0]); side /= np.linalg.norm(side) + 1e-6
            t = np.linspace(0, 1, 22)[:, None]; curl = r.uniform(.05, .10) * r.choice([-1, 1]); tw = r.uniform(5, 8)
            P = C0 + d * R * t + np.array([0, -.16, 0]) * t ** 3 + side * curl * np.sin(t * tw) * t + np.cross(d, side) * curl * (1 - np.cos(t * tw)) * t + wv * t ** 2
            out.append((float(H_.depth(P).mean()), P, r.uniform(.035, .06), k))
        return sorted(out, key=lambda q: q[0]), C0

    def knot(self, cel, H_, pal, lw, f=0, wind=0., seed=0, wdir=(1., .2, 0)):
        curls, C0 = self._knot(H_, f, wind, seed, wdir)
        band = H_.P3(np.array([(.15 * math.cos(a), .98, -.10 + .15 * math.sin(a)) for a in np.linspace(0, 6.283, 18)[:-1]] + [(.12 * math.cos(a), 1.17, -.10 + .12 * math.sin(a)) for a in np.linspace(0, 6.283, 18)[:-1]]))
        hull = cv2.convexHull(band.astype(np.float32))[:, 0].astype(float)
        for dp, P, w, k in curls:
            q = H_.P3(P); s_ = H_.s; T = np.gradient(q, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-6; N = np.c_[-T[:, 1], T[:, 0]]; hw = (w * s_ * np.interp(np.linspace(0, 1, len(q)), [0, .3, .75, 1], [.9, 1.1, .9, 0.05]))[:, None]
            poly = np.vstack([q + N * hw, (q - N * hw)[::-1]]); cel.fill(poly, pal['hair'])
            cel.fill(np.vstack([q + N * hw * -.1, (q - N * hw)[::-1]]), pal['hairS'])
            if k % 2 == 0:
                i0, i1 = 5, 15; g = np.sin(np.linspace(0, math.pi, i1 - i0))[:, None]; cel.fill(np.vstack([q[i0:i1] + N[i0:i1] * hw[i0:i1] * .62, (q[i0:i1] + N[i0:i1] * hw[i0:i1] * (.62 - .42 * g))[::-1]]), pal['hairH'])
            outline(cel, poly, lw * .55, pal['line'], seed + k, press=.5)
        cel.fill(hull, pal['hairS']); outline(cel, hull, lw * .6, pal['line'], seed + 77)
        for k in range(4):
            a = H_.P3(np.array([(-.12, 1.02 + .035 * k, -.02), (0, 1.00 + .035 * k, .05), (.12, 1.02 + .035 * k, -.02)])); cel.line(spl(a, n=6), lw * .3, pal['hairH'], taper=(.3, .3), alpha=.7)

    def hair(self, cel, H_, pal, lw, back, f=0, wind=0., seed=0, wdir=(1., .2, 0.), lit=-1):
        None_H[0] = H_; ls = self._locks(f, wind, seed, wdir)
        fr, bk = HAIR.shell(H_, hairline, lift=.03)
        knot_front = H_.pitch < .12
        if back:
            for c in quads_contour(bk): cel.fill(c, pal['hairS'])
            self.knot(cel, H_, pal, lw, f, wind, seed, wdir)
            cs = quads_contour(bk)
            for c in cs: outline(cel, c, lw * .9, pal['line'], 5)
            return
        sel = [d for d in ls if d['face'] >= -.03]; thin = [d for d in sel if d.get('thin')]; sel = [d for d in sel if not d.get('thin')]
        cs = quads_contour(fr)
        for c in cs: cel.fill(c, pal['hair'])
        u = union_contour([d['poly'] for d in sel] + cs)
        draw_locks(cel, sel, pal['hair'], pal['hairS'], pal['hairS'], lw * .44, seed=seed, hi=pal['hairH'], strands=.7)
        for i, c in enumerate(u): outline(cel, c, lw * .85, pal['line'], 11 + seed + i, press=.5)
        for k, d in enumerate(thin): cel.fill(d['poly'], pal['hair']); outline(cel, d['poly'], lw * .3, pal['line'], 200 + k)
        if H_.pitch < -.05:
            self.knot(cel, H_, pal, lw, f, wind, seed, wdir)

    # ------------------------------------------------------------ face
    def face(self, cel, H_, pal, lw, e, lit=-1, seed=0, f=0):
        P = H_.parts(hair_y=.60); fill = P['fill']; near = H_.near(); skl = pal['sline']; prof = abs(H_.yaw) > 1.12
        if H_.pitch > .02:
            un = H_.under(); cel.fill(un, pal['skinS']); cel.clip(un); hatch(cel, un, .5, 8, lw * .26, skl, 3, length=H_.s * .18); cel.unclip()
        cel.fill(fill, pal['skin']); cel.clip(fill); s = -lit if lit else 1
        roar = e['mouth'] in ('roar', 'cry')
        z0 = lambda x, y, d: np.array([x, y, float(H_.z(x, y)) + d])
        if not prof:
            S(cel, H_, [(s * .72, .64), (s * .58, .50), (s * .50, .36), (s * .545, .22), (s * .455, .10), (s * .485, -.04), (s * .525, -.14), (s * .475, -.25), (s * .385, -.36), (s * .345, -.46), (s * .385, -.56), (s * .335, -.68), (s * .275, -.80), (s * .235, -.90), (s * .12, -1.02), (s * .5, -1.04), (s * .75, -.6), (s * .8, 0)], pal['skinS'])
            S(cel, H_, [(-s * .70, -.30), (-s * .585, -.40), (-s * .50, -.56), (-s * .47, -.70), (-s * .40, -.84), (-s * .5, -1.0), (-s * .75, -.6)], pal['skinS'])
            for ss in (-1, 1):
                k = 1. if ss == s else .6
                S(cel, H_, [(ss * .075, .145), (ss * .18, .175), (ss * .40, .205), (ss * .50, .15), (ss * .44, .085), (ss * .32, .105 - .03 * k), (ss * .20, .07 - .03 * k), (ss * .145, -.01 - .03 * k), (ss * .115, -.08), (ss * .09, .02)], pal['skinS'])
                S(cel, H_, [(ss * .095, .085), (ss * .145, .075), (ss * .175, .035), (ss * .125, .0)], pal['skinD'], alpha=.6)
                # naso-labial and cheek planes
                if roar: S(cel, H_, [(ss * .13, -.42), (ss * .20, -.46), (ss * .30, -.58), (ss * .335, -.74), (ss * .30, -.72), (ss * .24, -.58), (ss * .16, -.50)], pal['skinS'])
            sh = s
            nose = np.array([z0(0, .085, .012), z0(0, -.06, .06), z0(0, -.22, .12), z0(0, -.34, .172), z0(0, -.405, .185), z0(0, -.45, .15), z0(sh * .06, -.465, .10), z0(sh * .128, -.440, .035), z0(sh * .132, -.385, .02), z0(sh * .095, -.31, .03), z0(sh * .085, -.2, .01), z0(sh * .09, -.06, 0), z0(sh * .10, .04, 0), z0(sh * .055, .10, 0)])
            cel.fill(H_.P3(nose), pal['skinS'])
            cel.fill(H_.P3(np.array([z0(-.122, -.438, .03), z0(-.05, -.448, .13), z0(0, -.442, .165), z0(.05, -.448, .13), z0(.122, -.438, .03), z0(.09, -.492, .02), z0(sh * .02, -.545, .0), z0(-.09, -.492, .02)])), pal['skinS'])
            cel.fill(H_.P3(np.array([z0(-.085, -.448, .05), z0(0, -.458, .12), z0(.085, -.448, .05), z0(.04, -.48, .05), z0(-.04, -.48, .05)])), pal['skinD'], .6)
            S(cel, H_, [(-.11, -.80), (0, -.815), (.12, -.80), (.08, -.855), (0, -.87), (-.07, -.855)], pal['skinS'])
            # cleft chin
            S(cel, H_, [(-.02, -.90), (.02, -.895), (.012, -.985), (-.012, -.985)], pal['skinS'])
        else:
            n = near; Z = lambda pts: H_.P3(np.array([(n * .62, y, z) for z, y in pts]))
            cel.fill(Z([(.30, .62), (.36, .40), (.24, .20), (.32, .06), (.22, -.12), (.32, -.26), (.42, -.38), (.36, -.50), (.46, -.62), (.40, -.76), (.50, -.88), (.60, -1.0), (.2, -1.05), (-.4, -.6), (-.4, .6)]), pal['skinS'])
        # hair cast
        g = [(-.70, .56), (-.60, .20), (-.555, .34), (-.50, .44), (-.43, .50), (-.36, .555), (-.2, .525), (-.04, .555), (.10, .52), (.26, .55), (.40, .52), (.47, .46), (.53, .36), (.59, .20), (.70, .56), (.4, .9), (-.4, .9)]
        S(cel, H_, g, pal['skinS'])
        cel.unclip()
        if not prof:
            cel.clip(fill)
            # scar: over his left brow and down the cheek (picture right in front view)
            sc = [(.275, .44), (.300, .30), (.325, .16), (.345, .02), (.375, -.14), (.415, -.30), (.400, -.14), (.372, .02), (.360, .16), (.335, .30), (.300, .42)]
            S(cel, H_, sc, pal['scar']); L(cel, H_, sc[:6], lw * .50, skl, 4, taper=(.4, .5)); L(cel, H_, sc[5:][::-1], lw * .34, skl, 5, taper=(.5, .5))
            for k, (x, y) in enumerate([(.318, .24), (.36, -.06), (.392, -.2)]): L(cel, H_, [(x - .035, y + .012), (x + .045, y - .010)], lw * .26, skl, 6 + k, taper=(.4, .4), smooth_=False)
            cel.unclip()
            for ss in (-1, 1): self._eye(cel, H_, pal, lw, e, ss, fill, seed)
            self._nose(cel, H_, pal, lw, s, seed)
            self._mouth(cel, H_, pal, lw, e, seed, fill)
            cel.clip(fill)
            for ss in (-1, 1):
                if H_._face_cos(np.array(ss * .42), np.array(-.4)) < .10: continue
                L(cel, H_, [(ss * .555, -.24), (ss * .490, -.37), (ss * .400, -.47)], lw * .42, skl, seed + 3, taper=(.4, .7))
                L(cel, H_, [(ss * .525, -.34), (ss * .455, -.46), (ss * .39, -.53)], lw * .28, skl, seed + 4, taper=(.6, .6))
                L(cel, H_, [(ss * .135, -.415), (ss * .185, -.47), (ss * .225, -.56)] if not roar else [(ss * .135, -.40), (ss * .225, -.46), (ss * .315, -.58), (ss * .335, -.74)], lw * (.42 if roar else .30), skl, seed + 5, taper=(.2, .6))
                for k in range(4):                                        # jaw hatching
                    y0 = -.50 - .075 * k; x0 = np.interp(y0, [-.8, -.6, -.45], [.43, .50, .56])
                    L(cel, H_, [(ss * x0, y0), (ss * (x0 + .05), y0 - .05), (ss * (x0 + .075), y0 - .10)], lw * .24, skl, seed + 80 + k, taper=(.5, .5), alpha=.8)
            # brow knit
            L(cel, H_, [(-.045, .20), (-.052, .14), (-.040, .08)], lw * .34, skl, seed + 7, taper=(.4, .6)); L(cel, H_, [(.050, .19), (.055, .13), (.045, .09)], lw * .30, skl, seed + 8, taper=(.4, .6))
            L(cel, H_, [(0, -.90), (.006, -.95), (0, -.985)], lw * .34, skl, seed + 9, taper=(.4, .4))
            cel.unclip()
        else: self._profile(cel, H_, pal, lw, e, near, seed, fill)
        for nm in ('left', 'right'):
            cel.line(P[nm], lw * 1.05, pal['line'], taper=(.10, .22), press=.34, seed=seed + (1 if nm == 'left' else 2), tmin=.3)
        if e.get('tears') and not prof: self._tears(cel, H_, pal, lw, e, fill, f, seed)
        return P

    def _eye(self, cel, H_, pal, lw, e, s, fill, seed):
        ex, ey, ew, eh = .285, -.005, .125, .088; o = e.get('open', .7); E = lambda U: eye_pts(s, ex, ey, ew, eh, .012, U)
        if H_._face_cos(np.array(s * .28), np.array(0.)) < .02: return
        cel.clip(fill); k = e['eye']
        if k == 'closed':
            u = sp([(-1.05, .10), (-.6, -.22), (0, -.36), (.6, -.24), (1.1, .16)]); th = np.interp(u[:, 0], [-1, -.4, .3, 1.1], [.06, .24, .26, .10])
            cel.fill(H_.Q(E(np.vstack([u, np.c_[u[:, 0], u[:, 1] + th][::-1]]))), pal['line'])
            for a in (.45, .75, 1.0): cel.line(H_.Q(E([(a, -.30 + .4 * max(0, a - .5)), (a + .14, -.52 + .45 * max(0, a - .5))])), lw * .4, pal['line'], taper=(0, .8))
            cel.line(H_.Q(E(sp([(-.8, .55), (0, .62), (.8, .60)]))), lw * .34, pal['sline'], taper=(.4, .5))
        elif k == 'squeeze':
            u = sp([(-1.1, -.35), (-.5, .30), (.1, .50), (.6, .28), (1.15, -.42)]); th = np.interp(u[:, 0], [-1.1, -.4, .3, 1.15], [.08, .34, .36, .10])
            cel.fill(H_.Q(E(np.vstack([u, np.c_[u[:, 0], u[:, 1] + th][::-1]]))), pal['line'])
            for a, b in (((1.05, -.25), (1.45, -.05)), ((1.1, -.40), (1.5, -.48)), ((-1.0, -.2), (-1.3, .05))): cel.line(H_.Q(E([a, b])), lw * .36, pal['sline'], taper=(.1, .7))
            cel.line(H_.Q(E(sp([(-.8, -.75), (0, -.62), (.9, -.80)]))), lw * .4, pal['sline'], taper=(.4, .4))
        else:
            b0 = -.2; U0 = [(-1.06, -.30), (-.80, .18), (-.30, .52), (.25, .62), (.75, .44), (1.10, .02)]
            if k == 'weep': U0 = [(-1.06, -.28), (-.78, .30), (-.30, .50), (.25, .42), (.75, .16), (1.10, -.24)]
            u = sp([(a, b0 + (b - b0) * o * 1.35 if b > b0 else b) for a, b in U0]); low = sp([(-1.06, -.30), (-.5, -.60), (.15, -.66), (.75, -.44), (1.10, U0[-1][1])])
            pol = H_.Q(E(np.vstack([u, low[::-1]]))); cel.fill(pol, pal['sclera']); cel.clip(pol)
            rad = .054 * e.get('iris', 1.); gx, gy = e.get('gaze', (0, 0)); c = (s * ex - .010 * s + gx * .5 * ew, ey + .010 + gy * eh)
            circ = lambda r, oy=0., ox=0.: [(c[0] + ox + r * math.cos(a), c[1] + oy + r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 24)[:-1]]
            cel.fill(H_.Q(circ(rad)), pal['iris']); cel.fill(H_.Q(circ(rad * .46)), pal['line']); cel.line(np.vstack([H_.Q(circ(rad)), H_.Q(circ(rad))[:1]]), lw * .36, pal['line'], taper=(0, 0))
            cel.fill(np.vstack([H_.Q(E(u)), H_.Q(E([(a, b - .24) for a, b in u]))[::-1]]), pal['scleraS'], .8)
            cel.fill(H_.Q(circ(.014, .020, -.018)), '#ffffff')
            if k == 'weep':
                cel.fill(H_.Q(circ(.018, -.030, .024)), '#ffffff', .9); cel.fill(np.vstack([H_.Q(E(low)), H_.Q(E([(a, b + .42) for a, b in low]))[::-1]]), pal['tear'], .85)
            cel.unclip()
            th = np.interp(u[:, 0], [-1.06, -.5, .2, 1.1], [.14, .34, .36, .16]); cel.fill(H_.Q(E(np.vstack([u, np.c_[u[:, 0], u[:, 1] + th][::-1]]))), pal['line'])
            cel.line(H_.Q(E(low[int(len(low) * .15):])), lw * .46, pal['sline'], taper=(.5, .15)); cel.line(H_.Q(E(sp([(-.9, -.82), (-.2, -1.05), (.5, -.95)]))), lw * .34, pal['sline'], taper=(.4, .6))
        bi, bo = e.get('brow_in', 0), e.get('brow_out', 0)
        bt = [(.075, .120 + bi), (.15, .165 + bi * .8), (.27, .205 + bi * .3), (.39, .235 + bo * .5), (.47, .225 + bo), (.545, .165 + bo)]
        bb = [(.545, .165 + bo), (.46, .180 + bo), (.38, .180 + bo * .5), (.27, .146 + bi * .3), (.15, .096 + bi * .8), (.085, .064 + bi)]
        br = np.array([(s * x, y) for x, y in np.vstack([sp(bt), sp(bb)])]); cel.fill(H_.Q(br), pal['brow'])
        for k_ in range(5):
            x = .09 + .085 * k_; y = float(np.interp(x, [q[0] for q in bt], [q[1] for q in bt])) - .02
            cel.line(H_.Q(np.array([(s * x, y), (s * (x + .030), y + .026), (s * (x + .060), y + .030)])), lw * .6, pal['brow'], taper=(.1, .7), seed=k_)
        cel.unclip()

    def _nose(self, cel, H_, pal, lw, s, seed):
        z0 = lambda x, y, d: np.array([x, y, float(H_.z(x, y)) + d]); yaw = H_.yaw; far = 1 if yaw > 0 else -1; skl = pal['sline']
        if abs(yaw) > .16:
            cel.line(spl(H_.P3(np.array([z0(0, -.10, .08), z0(0, -.22, .122), z0(0, -.32, .162), z0(0, -.385, .184), z0(0, -.415, .187), z0(0, -.448, .16), z0(-far * .03, -.465, .115)])), n=8), lw * .70, skl, taper=(.5, .25), press=.2, seed=seed + 50)
            n = -far
            cel.line(spl(H_.P3(np.array([z0(n * .10, -.345, .06), z0(n * .138, -.39, .048), z0(n * .134, -.44, .035), z0(n * .09, -.462, .045)])), n=8), lw * .52, skl, taper=(.4, .2), seed=seed + 51)
            cel.line(spl(H_.P3(np.array([z0(n * .015, -.468, .10), z0(n * .055, -.455, .085), z0(n * .09, -.465, .05)])), n=6), lw * .75, pal['line'], taper=(.35, .35), seed=seed + 52)
        else:
            for q in (-1, 1):
                cel.line(spl(H_.P3(np.array([z0(q * .12, -.375, .04), z0(q * .136, -.42, .035), z0(q * .110, -.452, .04)])), n=6), lw * .5, skl, taper=(.4, .3), seed=seed + 54)
                cel.line(spl(H_.P3(np.array([z0(q * .085, -.455, .065), z0(q * .055, -.447, .10), z0(q * .03, -.462, .12)])), n=6), lw * .7, pal['line'], taper=(.35, .35), seed=seed + 55)
            cel.line(spl(H_.P3(np.array([z0(-.03, -.470, .12), z0(0, -.478, .135), z0(.03, -.470, .12)])), n=6), lw * .5, skl, taper=(.4, .4), seed=seed + 51)
            cel.line(spl(H_.P3(np.array([z0(s * .075, .075, 0), z0(s * .064, .0, .02), z0(s * .066, -.08, .03)])), n=6), lw * .46, skl, taper=(.2, .7), seed=seed + 50)

    def _mouth(self, cel, H_, pal, lw, e, seed, fill):
        m = e['mouth']; cel.clip(fill); y = -.645; near = H_.near() or 1
        if m in ('shut', 'smirk'):
            cl, cr = (0., 0.) if m == 'shut' else ((.052, .006) if near <= 0 else (.006, .052)); w = .19
            L(cel, H_, [(-w, y + cl - .004), (-w * .6, y + .004 + cl * .35), (-.05, y + .002), (0, y - .010), (.05, y + .002), (w * .6, y + .004 + cr * .35), (w, y + cr - .004)], lw * .80, pal['line'], seed + 60, taper=(.08, .08), press=.35, tmin=.4)
            for q, cc in ((-1, cl), (1, cr)):
                if cc > .02: L(cel, H_, [(q * (w - .01), y + cc - .01), (q * (w + .03), y + cc + .03), (q * (w + .035), y + cc + .065)], lw * .5, pal['sline'], seed + 61, taper=(.1, .7))
                else: L(cel, H_, [(q * (w - .015), y + cc), (q * (w + .02), y + cc - .022)], lw * .55, pal['line'], seed + 61, taper=(.1, .6), smooth_=False)
            L(cel, H_, [(-.085, y - .105), (0, y - .116), (.085, y - .105)], lw * .46, pal['sline'], seed + 63, taper=(.4, .4))
        else:
            if m == 'grin': U = [(-.235, y + .045), (-.12, y + .022), (0, y + .012), (.12, y + .022), (.235, y + .045)]; D_ = [(.235, y + .045), (.17, y - .060), (0, y - .105), (-.17, y - .060), (-.235, y + .045)]
            elif m == 'roar': U = [(-.205, y + .040), (-.13, y + .115), (0, y + .105), (.13, y + .115), (.205, y + .040)]; D_ = [(.205, y + .040), (.215, y - .10), (.12, y - .225), (0, y - .240), (-.12, y - .225), (-.215, y - .10), (-.205, y + .040)]
            else: U = [(-.250, y + .025), (-.15, y + .085), (0, y + .070), (.15, y + .085), (.250, y + .025)]; D_ = [(.250, y + .025), (.215, y - .105), (.10, y - .185), (0, y - .195), (-.10, y - .185), (-.215, y - .105), (-.250, y + .025)]
            u, d = sp(U), sp(D_); pol = H_.Q(np.vstack([u, d[1:]])); cel.fill(pol, pal['mouth']); cel.clip(pol)
            lo = d[:, 1].min(); wd = abs(U[0][0])
            cel.fill(H_.Q(sp([(-wd * .62, lo - .02), (-wd * .4, lo + .075), (0, lo + .095), (wd * .4, lo + .075), (wd * .62, lo - .02)])), pal['tongue'])
            if m == 'grin': cel.fill(H_.Q(np.vstack([u, d[1:]])), pal['teeth'])
            # upper row: one slab, hinted divisions only at the edges
            tp = sp([(x, yy - .062 + .020 * (abs(x) / wd) ** 2) for x, yy in U]); cel.fill(H_.Q(np.vstack([u, tp[::-1]])), pal['teeth'])
            cel.line(H_.Q(tp), lw * .36, pal['sline'], taper=(.2, .2))
            if m != 'grin':
                lt = sp([(-wd * .66, lo + .030), (0, lo + .052), (wd * .66, lo + .030)]); cel.fill(H_.Q(np.vstack([lt, [(wd * .6, lo - .05), (-wd * .6, lo - .05)]])), pal['teeth']); cel.line(H_.Q(lt), lw * .3, pal['sline'], taper=(.2, .2))
                for x in (-.62, -.36, .36, .62): cel.line(H_.Q(np.array([(x * wd, y + .10), (x * wd, y + .012)])), lw * .22, pal['sline'], taper=(.2, .5))
            else:
                cel.line(H_.Q(sp([(-wd, y + .035), (-.12, y - .020), (0, y - .030), (.12, y - .020), (wd, y + .035)])), lw * .42, pal['sline'], taper=(.1, .1))
                for x in (-.75, -.5, -.25, 0, .25, .5, .75): cel.line(H_.Q(np.array([(x * wd, y + .04), (x * wd * .95, y - .075)])), lw * .2, pal['sline'], taper=(.2, .4))
            cel.unclip()
            L(cel, H_, U, lw * .86, pal['line'], seed + 60, taper=(.05, .05), press=.3); L(cel, H_, D_, lw * .62, pal['line'], seed + 61, taper=(.1, .1), press=.35)
            for q in (-1, 1): L(cel, H_, [(q * (wd - .004), U[0][1] + .01), (q * (wd + .035), U[0][1] + .035), (q * (wd + .05), U[0][1] + .0)], lw * .45, pal['sline'], seed + 62, taper=(.1, .7))
            L(cel, H_, [(-.09, lo - .045), (0, lo - .056), (.09, lo - .045)], lw * .46, pal['sline'], seed + 63, taper=(.4, .4))
        cel.unclip()

    def _tears(self, cel, H_, pal, lw, e, fill, f, seed):
        """manly tears: two flat ribbons that run over the cheekbones and tear off at the jaw"""
        for s in (-1, 1):
            if H_._face_cos(np.array(s * .30), np.array(-.1)) < .08: continue
            w = .060; g = lambda k: .010 * math.sin(f * .9 + k * 2.1 + s)
            C = [(.285, -.050), (.300, -.16), (.335 + g(1), -.30), (.330 + g(2), -.46), (.345 + g(3), -.62), (.320 + g(4), -.80), (.300, -.95)]
            Wd = [.085, .060, .050, .056, .050, .060, .045]
            A = np.array([(s * (x - q), y) for (x, y), q in zip(C, Wd)]); B = np.array([(s * (x + q), y) for (x, y), q in zip(C, Wd)])
            a, b = H_.Q(spl(A, n=8), .012), H_.Q(spl(B, n=8), .012); P = np.vstack([a, b[::-1]])
            cel.clip(fill); cel.fill(P, pal['tear'], .93); n = len(a); t = np.linspace(0, 1, n)[:, None]
            m = a * .5 + b * .5; q = .62 + .16 * np.sin(t * 9 + f * 1.3 + s); cel.fill(np.vstack([a * (1 - q) + b * q, b[::-1]]), pal['tearS'], .9)
            cel.fill(np.vstack([a + (b - a) * .16, (a + (b - a) * (.16 + .14 * np.abs(np.sin(t * 7 - f * 1.1))))[::-1]]), '#ffffff')
            cel.line(a, lw * .46, pal['sline'], taper=(.1, .1), seed=seed); cel.line(b, lw * .46, pal['sline'], taper=(.1, .1), seed=seed + 1); cel.unclip()
        # drops thrown off the jaw and lashes
        r = np.random.default_rng(40 + f // 2)
        for k in range(9):
            s = -1 if k % 2 else 1; t = ((f * .11 + k * .23) % 1.); p = H_.P3(np.array([(s * (.40 + .9 * t * (1 + .3 * (k % 3))), -.05 + .55 * t - 1.35 * t * t, .62 + .1 * t)]))[0]; R = H_.s * (.045 - .025 * t) * (1 + .3 * (k % 2))
            an = math.atan2(-(.55 - 2.7 * t), s * .9); d = np.array([math.cos(an), math.sin(an)]); nn = np.array([-d[1], d[0]])
            P = spl(np.array([p + d * R, p + nn * R * .75, p - d * R * 1.9, p - nn * R * .75]), True, corners=(2,), n=8); cel.fill(P, pal['tear'], .95); cel.fill(spl(np.array([p + d * R * .9, p + nn * R * .7, p - d * R * .2, p + nn * R * .05]), True, n=6), pal['tearS'], .9); cel.line(np.vstack([P, P[:1]]), lw * .42, pal['sline'], taper=(0, 0)); cel.fill(spl(np.array([p + d * R * .4 - nn * R * .5, p + d * R * .1 - nn * R * .25, p - d * R * .5 - nn * R * .3]), True, n=5), '#ffffff')

    def _profile(self, cel, H_, pal, lw, e, n, seed, fill):
        X = n * .34; Z = lambda pts, x=X: H_.P3(np.array([(x, y, z) for z, y in pts])); skl = pal['sline']; cel.clip(fill)
        cel.fill(Z([(.83, .16), (.72, .175), (.50, .19), (.40, .12), (.50, .09), (.60, .08), (.65, .02), (.67, -.08), (.72, -.02), (.775, .02)]), pal['skinS'])
        cel.fill(Z([(.90, -.43), (.80, -.455), (.765, -.38), (.70, -.395), (.695, -.45), (.74, -.49), (.79, -.52), (.83, -.49)], n * .05), pal['skinS'])
        cel.fill(Z([(.80, -.79), (.72, -.80), (.70, -.82), (.74, -.86), (.80, -.86)], n * .05), pal['skinS'])
        o = e.get('open', .7)
        if e['eye'] in ('closed', 'squeeze'):
            c = sp([(.640, -.01), (.60, -.05), (.53, -.055), (.455, -.02), (.42, .02)]); cel.fill(Z(np.vstack([c, (c + [0, .022])[::-1]])), pal['line'])
        else:
            up = sp([(.645, .030 * o * 1.4), (.612, .052 * o * 1.4), (.54, .046 * o * 1.4), (.46, .012)]); dn = sp([(.636, -.052), (.58, -.058), (.50, -.036), (.46, .012)])
            pol = Z(np.vstack([up, dn[::-1]])); cel.fill(pol, pal['sclera']); cel.clip(pol); it = np.linspace(0, 6.283, 22)[:-1]; k = e.get('iris', 1.)
            cel.fill(Z(np.c_[.610 + .023 * k * np.cos(it), .002 + .048 * k * np.sin(it)]), pal['iris']); cel.fill(Z(np.c_[.616 + .011 * np.cos(it), .002 + .022 * np.sin(it)]), pal['line']); cel.fill(Z(np.c_[.619 + .007 * np.cos(it), .018 + .010 * np.sin(it)]), '#ffffff'); cel.unclip()
            cel.fill(Z(np.vstack([up, (up + [0, .026])[::-1]])), pal['line']); cel.line(Z(dn), lw * .42, skl, taper=(.1, .5))
        bi, bo = e.get('brow_in', 0), e.get('brow_out', 0)
        bt = sp([(.815, .130 + bi), (.72, .180 + bi * .6), (.58, .232 + bo * .5), (.46, .222 + bo), (.37, .160 + bo)]); bb = sp([(.37, .160 + bo), (.46, .176 + bo), (.58, .165 + bo * .5), (.72, .105 + bi * .6), (.805, .074 + bi)])
        cel.fill(Z(np.vstack([bt, bb])), pal['brow'])
        cel.line(Z(sp([(.735, -.36), (.70, -.40), (.705, -.45), (.745, -.465)]), n * .12), lw * .55, skl, taper=(.4, .2)); cel.line(Z(sp([(.80, -.462), (.77, -.452), (.74, -.462)]), n * .07), lw * .8, pal['line'], taper=(.3, .3))
        m = e['mouth']
        if m in ('roar', 'cry', 'grin'):
            h = dict(roar=.20, cry=.15, grin=.08)[m]; up = sp([(.80, -.645), (.74, -.60), (.66, -.60), (.60, -.64)]); dn = sp([(.60, -.64), (.63, -.70 - h * .4), (.70, -.70 - h), (.78, -.70 - h)]); pol = Z(np.vstack([up, dn]), n * .05)
            cel.fill(pol, pal['mouth']); cel.clip(pol); cel.fill(Z(np.vstack([up, (up - [0, .055])[::-1]]), n * .05), pal['teeth']); cel.unclip(); cel.line(Z(up, n * .05), lw * .8, pal['line']); cel.line(Z(dn, n * .05), lw * .6, pal['line'])
        else: cel.line(Z(sp([(.788, -.680), (.74, -.672), (.68, -.668), (.615, -.66 + (.04 if m == 'smirk' else 0))]), n * .06), lw * .85, pal['line'], taper=(.1, .25))
        cel.line(Z(sp([(.40, -.28), (.46, -.40), (.50, -.52)]), n * .60), lw * .42, skl, taper=(.4, .7))
        cel.unclip()

    def ears(self, cel, H_, pal, lw, which, seed=0):
        for s in (-1, 1):
            e = H_.ear(s); near = H_.near(); isnear = (near == 0) or (near == s)
            if e['vis'] <= -.15 or (which == 'near') != isnear: continue
            o = e['outer']; cel.fill(o, pal['skin']); cel.clip(o); cel.fill(e['inner'], pal['skinS']); cel.fill(e['tragus'], pal['skinD'], .8); cel.unclip()
            outline(cel, o, lw * .85, pal['line'], seed + s, press=.3); cel.line(e['inner'], lw * .45, pal['sline'], taper=(.3, .4)); cel.line(e['tragus'], lw * .4, pal['sline'], taper=(.4, .4))

    def head(self, cel, H_, expr='focus', light='day', f=0, wind=0., seed=0, lit=-1, lw=None, part='all', wdir=(1., .2, 0.)):
        pal = PAL[light] if isinstance(light, str) else light; e = dict(EXPR[expr]) if isinstance(expr, str) else dict(expr); lw = lw or max(1.4, H_.s / 300. * 5.4)
        if part in ('all', 'back'):
            self.hair(cel, H_, pal, lw, True, f, wind, seed, wdir, lit); self.ears(cel, H_, pal, lw, 'far', seed)
            if part == 'back': return
        P = self.face(cel, H_, pal, lw, e, lit, seed, f); self.ears(cel, H_, pal, lw, 'near', seed)
        self.hair(cel, H_, pal, lw, False, f, wind, seed, wdir, lit)
        return P



# ==================================================================== body
class TBody(Body):
    """Todo's trunk: deep chest, unit = head unit, origin = pit of the neck"""
    def __init__(self, *a, **k):
        k.setdefault('hw', 2.0); k.setdefault('hd', 1.0); super().__init__(*a, **k)

    def z(self, x, y):
        hw = np.interp(y, [-4.6, -3.7, -2.6, -1.6, -.6, .3], [1.36, 1.24, 1.72, 2.1, 2.05, 1.6])
        u = np.clip(np.abs(x) / hw, 0, .995); d = self.hd * np.interp(y, [-4.6, -3.7, -2.6, -1.7, -1.15, -.5, -.05, .3], [.74, .72, .84, .90, 1.10, .86, .46, .22])
        return d * (1 - u ** 2.3) ** .5 - .18


def hand_placeholder(cel, w, d, s, pal, lw, flip=1, spread=1., seed=0):
    """open palm to the viewer.  w wrist px, d pointing angle (rad), s = px of palm length; flip = thumb side"""
    c_, s_ = math.cos(d), math.sin(d); M = np.array([[c_, s_], [-s_, c_]]); ln = pal['line']
    Q = lambda P: w + (np.asarray(P, float) * [1, flip]) @ M * s
    fingers = [((.92, .34), .30 * spread, .78, .125), ((1.0, .10), .06 * spread, .88, .130), ((.96, -.15), -.16 * spread, .80, .122), ((.84, -.36), -.42 * spread, .62, .105)]
    for (x, y), a, L_, r in fingers[::-1]:
        u = np.array([math.cos(a), math.sin(a)]); n = np.array([-u[1], u[0]]); b = np.array([x, y]) - u * .12
        pts = [b + n * r * .95, b + u * L_ * .32 + n * r * 1.0, b + u * L_ * .55 + n * r * .88, b + u * L_ * .85 + n * r * .86, b + u * (L_ + .10) + n * r * .45, b + u * (L_ + .10) - n * r * .45, b + u * L_ * .85 - n * r * .86, b + u * L_ * .55 - n * r * .90, b + u * L_ * .32 - n * r * 1.02, b - n * r * .95]
        P = Q(spl(np.array(pts), True, corners=(0, 9), n=6)); cel.fill(P, pal['skin']); cel.clip(P); cel.fill(Q(np.array([b - n * r * 2, b + u * (L_ + .3) - n * r * 2, b + u * (L_ + .2) - n * r * .25, b + u * L_ * .5 - n * r * .42, b + u * L_ * .25 - n * r * .2])), pal['skinS']); cel.unclip(); outline(cel, P, lw * .8, ln, seed)
        for t in (.40, .72): cel.line(Q(np.array([b + u * L_ * t + n * r * .6, b + u * (L_ * t + .015), b + u * L_ * t - n * r * .6])), lw * .36, pal['sline'], taper=(.3, .3))
    th = [(.22, .34), (.40, .52), (.62, .70), (.80, .82), (.90, .80), (.88, .70), (.72, .56), (.58, .40), (.50, .22)]
    palm = [(-.05, .26), (.18, .40), (.55, .46), (.92, .48), (1.02, .22), (1.06, .0), (1.0, -.26), (.82, -.48), (.50, -.46), (.18, -.40), (-.05, -.28)]
    P = Q(spl(np.array(palm), True, corners=(0, 10), n=7)); cel.fill(P, pal['skin']); cel.clip(P); cel.fill(Q(np.array([(-.1, -.1), (.25, -.18), (.55, -.12), (.86, -.30), (1.1, -.2), (1.1, -.6), (-.1, -.6)])), pal['skinS']); cel.unclip(); outline(cel, P, lw * .9, ln, seed + 1)
    P = Q(spl(np.array(th), True, corners=(0, 8), n=7)); cel.fill(P, pal['skin']); outline(cel, P, lw * .85, ln, seed + 2)
    for L_ in ([(.30, .30), (.50, .16), (.52, -.10), (.34, -.30)], [(.62, .34), (.70, .10), (.80, -.12), (.82, -.34)], [(.56, .30), (.60, .06), (.66, -.16)]): cel.line(Q(spl(np.array(L_), n=7)), lw * .42, pal['sline'], taper=(.3, .5))


def hand3(cel, w, e, fz, pal, lw, side='R', pose_='open', yaw=180., pitch=0., roll=0., scale=.70, light=(-.6, .7, .5), seed=0, dist=7.0, bld=None, cut=.10):
    """real hand (jj/hands.py) seated on a wrist.  w, e wrist / elbow px, fz px per body unit; yaw 180 = palm to us; roll is added to the forearm direction.
    side = the character's hand ('R' is on the picture left when he faces us)"""
    try: from jj import hands as HD
    except Exception: return hand_placeholder(cel, w, math.atan2(*(w - e)[::-1]), fz * .6, pal, lw, 1 if side == 'L' else -1)
    d = (np.asarray(w, float) - e); d = d / (np.hypot(*d) + 1e-6); n = np.array([-d[1], d[0]])
    rl = math.degrees(math.atan2(-d[0], -d[1])) + roll
    hp = dict(skin=pal['skin'], shade=pal['skinS'], line=pal['line'], nail=hexmix(pal['skin'], '#ffffff', .22), nail_shade=hexmix(pal['skinS'], '#ffffff', .15), light=pal['skinH'])
    s = fz * scale; cam = HD.Cam(w[0], w[1], s, dist)
    a = np.asarray(w, float) - d * fz * cut; R = 5000.
    cel.clip(np.array([a - n * R, a + n * R, a + n * R + d * R, a - n * R + d * R]))
    HD.draw(cel, pose_, cam, hp, light, max(1.2, lw * .85), seed, side=side, R=HD.euler(yaw, pitch, rl), bld=bld or HD.HEAVY, pivot=(0, 0, 0), res=480)
    cel.unclip()


class _TodoBody(_Todo):
    # ------------------------------------------------------------ neck and trapezius
    def neck(self, cel, H_, b, pal, lw, lit=-1, seed=0):
        top = [H_.P3(np.array([(q * .60, -.34, -.18)]))[0] for q in (-1, 1)]; lo = [b(np.array([(q * .80, .30, -.12)]))[0] for q in (-1, 1)]
        mid = [(t * .45 + l * .55) + np.array([q * .015 * H_.s, 0]) for t, l, q in zip(top, lo, (-1, 1))]
        Lq = spl([top[0], mid[0], lo[0]], n=10); Rq = spl([top[1], mid[1], lo[1]], n=10); pit = b(np.array([(0., -.22)]))[0]
        poly = np.vstack([Lq, [pit], Rq[::-1]]); cel.fill(poly, pal['skin']); cel.clip(poly)
        jl, jr = H_.jaw(-1), H_.jaw(1); chin = H_.P3(np.array([(0, -1.0, .66)]))[0]; s = H_.s; sd = -lit if lit else 1
        q = np.vstack([jl[1:-1], [chin], jr[1:-1][::-1]]); off = np.array([.05 * s * -lit, .20 * s])
        cel.fill(np.vstack([[top[0] + [-.5 * s, 0]], q + off + np.c_[np.zeros(len(q)), .04 * s * np.sin(np.arange(len(q)) * 1.9)], [top[1] + [.5 * s, 0]], [top[1] + [0, -s]], [top[0] + [0, -s]]]), pal['skinS'])
        a, c = (Rq, Lq) if sd > 0 else (Lq, Rq); cel.fill(np.vstack([a, (a + (c - a) * np.linspace(.18, .30, len(a))[:, None])[::-1]]), pal['skinS']); cel.unclip()
        cel.line(Lq, lw * .85, pal['line'], seed=seed, taper=(.2, .2)); cel.line(Rq, lw * .85, pal['line'], seed=seed + 1, taper=(.2, .2))
        for q_, side in ((-1, Lq), (1, Rq)):
            p0 = side[3] + (pit - side[3]) * .06; p1 = b(np.array([(q_ * .12, -.16)]))[0]
            cel.line(spl([p0, (p0 * .4 + p1 * .6) + np.array([q_ * .05 * s, 0]), p1], n=8), lw * .55, pal['sline'], taper=(.4, .3), seed=seed + 3)
            cel.line(spl([p0 + (pit - side[3]) * .10, (p0 * .5 + p1 * .5) + np.array([-q_ * .02 * s, 0]), p1 + [-q_ * .01 * s, -.02 * s]], n=8), lw * .34, pal['sline'], taper=(.5, .5), seed=seed + 4)
            # veins of effort
            m0 = side[6] + (pit - side[6]) * .22; cel.line(spl([m0, m0 + np.array([.03 * s * q_, .10 * s]), m0 + np.array([-.01 * s, .20 * s])], n=6), lw * .30, pal['sline'], taper=(.4, .6), seed=seed + 5)
        t0 = H_.P3(np.array([(0., -1.0, .44)]))[0]; m_ = t0 * .45 + pit * .55
        cel.line(spl([m_ + [-.07 * s, -.04 * s], m_ + [0, .06 * s], m_ + [.07 * s, -.04 * s]], n=6), lw * .46, pal['sline'], taper=(.5, .5), seed=seed + 6)

    # ------------------------------------------------------------ trunk
    def trunk(self, cel, b, pal, lw, lit=-1, seed=0, sash=True, flex=1.0):
        """bare torso from the neck to the sash.  flex 0..1.2 deepens the cuts"""
        A, ln, skl = b, pal['line'], pal['sline']; sd = -lit if lit else 1
        sil = [(.66, .78, -.18), (.86, .52, -.18), (1.30, .28, -.16), (1.95, .10, -.12), (2.10, -.45, -.05), (1.98, -.95, .0), (1.96, -1.25, .0), (2.10, -1.62, -.05), (1.98, -2.2, -.05), (1.60, -2.95, .0), (1.24, -3.65, .0), (1.34, -4.5, .0), (1.40, -5.2, .0)]
        P = A.P3(np.vstack([spl(np.array(sil), n=8), spl(np.array(_m(sil, -1)), n=8)[::-1]])); cel.fill(P, pal['skin']); cel.clip(P)
        f = lambda pts, c, al=1.: cel.fill(A(resample(np.vstack([np.asarray(pts, float), pts[:1]]), .06)), c, al)
        def both(pts, c=None, hidden=1.):
            for q in (-1, 1): f([(q * x, y) for x, y in pts], c or pal['skinS'])
        # side plane away from the light
        f([(sd * 2.2, -.9), (sd * 1.70, -1.15), (sd * 1.46, -1.62), (sd * 1.30, -1.95), (sd * 1.52, -2.15), (sd * 1.16, -2.40), (sd * 1.32, -2.62), (sd * 1.00, -2.86), (sd * 1.08, -3.2), (sd * .78, -3.6), (sd * .92, -4.6), (sd * 1.8, -4.6), (sd * 2.2, -2)], pal['skinS'])
        f([(sd * 2.3, -1.3), (sd * 1.80, -1.55), (sd * 1.66, -2.1), (sd * 1.72, -2.5), (sd * 1.36, -3.1), (sd * 1.24, -3.7), (sd * 1.3, -4.6), (sd * 2.0, -4.6)], pal['skinD'], .55)
        f([(-sd * 2.2, -1.1), (-sd * 1.82, -1.35), (-sd * 1.70, -1.9), (-sd * 1.52, -2.55), (-sd * 1.22, -3.1), (-sd * 1.14, -3.7), (-sd * 1.24, -4.6), (-sd * 1.8, -4.6)], pal['skinS'])
        # under the pecs, sternum
        both([(1.84, -1.0), (1.50, -1.36), (1.12, -1.50), (.62, -1.50), (.22, -1.40), (.06, -1.16), (.0, -1.60), (.20, -1.64), (.64, -1.74), (1.16, -1.78), (1.56, -1.62), (1.86, -1.36)])
        f([(-.055, -.30), (.055, -.30), (.085, -.9), (.13, -1.40), (.0, -1.75), (-.13, -1.40), (-.085, -.9)], pal['skinS'])
        both([(1.60, -1.50), (1.10, -1.66), (.64, -1.64), (.30, -1.54), (.64, -1.72), (1.14, -1.76), (1.54, -1.64)], pal['skinD'])
        # delto-pectoral groove and clavicle pits
        both([(1.02, -.10), (1.50, -.22), (1.80, -.62), (1.90, -.95), (1.70, -.66), (1.42, -.36)])
        both([(.16, .02), (.55, .10), (1.10, .20), (1.62, .14), (1.12, .06), (.56, -.02)])
        f([(sd * .10, -.9), (sd * .16, -1.22), (sd * .44, -1.10), (sd * .78, -1.22), (sd * 1.10, -1.02), (sd * 1.36, -1.12), (sd * 1.44, -.72), (sd * 1.80, -.55), (sd * 1.95, -1.0), (sd * 1.5, -1.5), (sd * .6, -1.62), (sd * .1, -1.45)], pal['skinS'])
        f([(-sd * .12, -1.18), (-sd * .40, -1.36), (-sd * .86, -1.40), (-sd * 1.30, -1.24), (-sd * 1.56, -.98), (-sd * 1.80, -1.0), (-sd * 1.5, -1.5), (-sd * .6, -1.62), (-sd * .1, -1.5)], pal['skinS'])
        # pec fibres on the shadow side
        for (x0, y0, x1, y1) in ((1.66, -.80, 1.02, -.62), (1.60, -1.02, .86, -.98), (1.42, -1.22, .74, -1.26)): f([(sd * x0, y0), (sd * x1, y1), (sd * (x0 - .05), y0 - .10)], pal['skinS'])
        # ribcage arch, serratus fingers
        both([(.62, -1.86), (.80, -2.10), (.96, -2.50), (1.06, -2.96), (1.16, -2.56), (1.02, -2.12), (.78, -1.84)])
        def finger(q, k):
            y0 = (-1.70, -2.02, -2.34, -2.64)[k]; x0 = (1.80, 1.74, 1.62, 1.46)[k]; L_ = (.50, .56, .50, .40)[k]; h_ = (.22, .24, .21, .17)[k]
            top = spl([(x0, y0 + .04), (x0 - L_ * .45, y0 + .03), (x0 - L_, y0 - h_ * .55)], n=8); bot = spl([(x0 - L_, y0 - h_ * .55), (x0 - L_ * .55, y0 - h_ * .95), (x0 - .04, y0 - h_ * 1.15)], n=8)
            return top * [q, 1], bot * [q, 1]
        for q in (-1, 1):
            for k in range(4):
                t_, b_ = finger(q, k); cut = b_ * .5 + (t_[::-1] * .5); f(np.vstack([b_, cut[::-1]]), pal['skinS'] if q != sd else pal['skinD'])
        # abs: each block has its own cut (no two alike)
        def blk(q, k, y0, y1, xo):
            j = np.random.default_rng(70 + k * 2 + (q > 0)).normal(0, .018, 8); sl = (.05, .02, -.02)[min(k, 2)]
            return spl(np.array([(q * (.05 + j[0] * .3), y0 + j[1]), (q * (xo * .55), y0 + .05 + sl + j[2]), (q * (xo - .03 + j[3]), y0 + sl), (q * (xo + .03 + j[4]), (y0 + y1) / 2), (q * (xo - .05 + j[5]), y1 + .03 + sl), (q * xo * .5, y1 - .03 + j[6]), (q * .05, y1 + j[7])]), True, corners=(0, 6), n=7)
        rows = [(-1.88, -2.30, .62), (-2.42, -2.84, .60), (-2.95, -3.42, .54)]
        f([(sd * .03, -1.75), (sd * .02, -4.4), (sd * .50, -4.4), (sd * .56, -3.5), (sd * .74, -3.4), (sd * .70, -2.9), (sd * .78, -2.4), (sd * .70, -1.95)], pal['skinS'])
        both([(.60, -1.96), (.74, -2.2), (.66, -2.36), (.72, -2.7), (.62, -2.88), (.64, -3.3), (.50, -3.5), (.56, -4.4), (.40, -4.4), (.44, -3.5), (.56, -3.2), (.50, -2.9), (.60, -2.6), (.56, -2.36)])
        for k, (y0, y1, xo) in enumerate(rows):
            for q in (-1, 1):
                B_ = blk(q, k, y0, y1, xo); f(B_, pal['skin']); pb = A(B_); cel.clip(pb)
                e = xo * (1 if q == sd else .86)
                f([(q * .0, y1 + .10), (q * .12, y1 + .075), (q * xo * .5, y1 + .10), (q * (xo * .78), y1 + .16), (q * (e - .05), (y0 + y1) / 2 + .02), (q * (e - .02), y0 + .1), (q * (xo + .2), y0 + .1), (q * (xo + .2), y1 - .2), (q * 0, y1 - .2)], pal['skinS'])
                if q == sd: f([(q * xo * .62, y1 + .02), (q * xo * .80, y1 + .20), (q * (xo * .92), y0 - .02), (q * (xo + .2), y0), (q * (xo + .2), y1 - .1)], pal['skinS'])
                f([(q * .0, y1 + .025), (q * xo * .5, y1 + .035), (q * xo * .9, y1 + .09), (q * xo, y1 - .1), (q * 0, y1 - .1)], pal['skinD'], .65)
                cel.unclip()
        for q in (-1, 1):
            f([(q * .05, -3.52), (q * .10, -3.9), (q * .06, -4.4), (q * .0, -4.4), (q * .0, -3.5)], pal['skinS'])
        f([(-.04, -3.40), (.04, -3.40), (.03, -3.56), (-.03, -3.56)], pal['skinD'])
        both([(1.36, -3.0), (1.06, -3.35), (.70, -3.95), (.58, -4.4), (.78, -4.0), (1.16, -3.5)])
        # highlights on the lit tops
        for q in (-1, 1):
            if q == sd: continue
            f([(q * .30, -.44), (q * .62, -.34), (q * .98, -.36), (q * .70, -.46), (q * .66, -.60), (q * .42, -.56)], pal['skinH']); f([(q * 1.06, -.40), (q * 1.28, -.48), (q * 1.22, -.58), (q * 1.04, -.48)], pal['skinH']); f([(q * .22, -2.02), (q * .44, -2.0), (q * .40, -2.14), (q * .24, -2.16)], pal['skinH']); f([(q * .22, -2.52), (q * .40, -2.50), (q * .37, -2.62), (q * .24, -2.64)], pal['skinH'])
        cel.unclip()
        # ---- line
        l = lambda pts, w, c=skl, tp=(.25, .45), n=9, sd_=0: cel.line(A(spl(np.array(pts, float), n=n)), lw * w, c, taper=tp, press=.35, seed=seed + sd_)
        for q in (-1, 1):
            e = spl(np.array(_m(sil, q)), n=8); cel.line(A.P3(e[10:26]), lw * .95, ln, taper=(.4, .3), press=.35, seed=seed + q); cel.line(A.P3(e[46:]), lw * .95, ln, taper=(.2, .05), press=.35, seed=seed + 2 + q)
            l([(q * .08, -.30), (q * .07, -.8), (q * .10, -1.26), (q * .24, -1.46), (q * .62, -1.60), (q * 1.12, -1.62), (q * 1.52, -1.46), (q * 1.80, -1.14), (q * 1.90, -.86)], .78, ln, (.3, .25), 10, 10 + q)
            l([(q * 1.02, -.06), (q * 1.46, -.24), (q * 1.72, -.56), (q * 1.88, -.92)], .55, skl, (.5, .2), 8, 12 + q)
            l([(q * .10, -.10), (q * .50, .02), (q * 1.0, .10), (q * 1.55, .10), (q * 1.9, .0)], .58, skl, (.1, .5), 8, 14 + q); l([(q * .30, .06), (q * .70, .16), (q * 1.15, .22)], .34, skl, (.4, .6), 8, 15)
            l([(q * .14, -.18), (q * .08, -.26), (q * .02, -.30)], .5, skl, (.3, .3), 6, 16)
            for (x0, y0, x1, y1) in ((1.70, -.86, 1.20, -.72), (1.62, -1.06, 1.05, -1.02), (1.44, -1.26, .96, -1.30)): l([(q * x0, y0), (q * (x0 + x1) / 2, (y0 + y1) / 2 + .03), (q * x1, y1)], .30, skl, (.1, .8), 6, 18)
            l([(q * .64, -1.88), (q * .84, -2.12), (q * .99, -2.50), (q * 1.07, -2.94)], .50, skl, (.4, .6), 8, 20 + q)
            for k in range(4):
                t_, b_ = finger(q, k); cel.line(A(b_), lw * .46, skl, taper=(.25, .4), press=.4, seed=seed + 22 + k); cel.line(A(t_[6:]), lw * .26, skl, taper=(.6, .2), seed=seed + 26 + k)
            for k, (y0, y1, xo) in enumerate(rows):
                B_ = blk(q, k, y0, y1, xo); n_ = len(B_); cel.line(A(B_[int(n_ * .30):int(n_ * .86)]), lw * .46, skl, taper=(.35, .4), press=.4, seed=seed + 30 + k)
                cel.line(A(B_[int(n_ * .10):int(n_ * .30)]), lw * .24, skl, taper=(.5, .5), seed=seed + 33 + k)
            l([(q * 1.34, -3.04), (q * 1.04, -3.40), (q * .70, -3.95), (q * .60, -4.4)], .48, skl, (.2, .5), 8, 40)
            l([(q * 1.58, -2.4), (q * 1.36, -2.95), (q * 1.16, -3.3)], .34, skl, (.4, .7), 8, 41)
        l([(0, -1.74), (.02, -2.3), (-.01, -2.85), (.01, -3.36)], .50, skl, (.2, .3), 8, 42); l([(0, -3.56), (.01, -4.0), (0, -4.4)], .38, skl, (.2, .6), 8, 43)
        l([(-.05, -3.40), (0, -3.46), (.05, -3.40)], .6, ln, (.3, .3), 6, 44)
        # hatching in the deep shade
        rg = A(np.array([(sd * 1.82, -1.4), (sd * 1.36, -1.66), (sd * 1.30, -2.3), (sd * 1.10, -2.7), (sd * 1.36, -3.0), (sd * 1.84, -2.2)])); hatch(cel, rg, .9 if sd > 0 else 2.2, b.s * .055, lw * .24, skl, 5, length=b.s * .20, curve=.06, fade=.3)
        for q in (-1, 1):
            rg = A(np.array([(q * .20, -1.50), (q * .62, -1.64), (q * 1.12, -1.66), (q * 1.5, -1.52), (q * 1.46, -1.64), (q * 1.12, -1.80), (q * .62, -1.78)])); hatch(cel, rg, 1.35, b.s * .05, lw * .24, skl, 7, length=b.s * .16, curve=.05)
        # cord
        cd = A.P3(spl(np.array([(-.72, .46, .0), (-.52, .02, .42), (-.22, -.30, .58), (0, -.40, .62), (.22, -.30, .58), (.52, .02, .42), (.72, .46, .0)]), n=8)); cel.line(cd, lw * .95, pal['cord'], taper=(.05, .05), press=.1)
        if sash:
            ring = lambda y, r_, d_, sg=1., n=24: np.array([(r_ * math.sin(t), y - .10 * math.cos(t) * sg, d_ * math.cos(t) - .12) for t in np.linspace(-math.pi * .5, math.pi * .5, n)])
            # trousers: wide, gathered, dark
            tr = np.vstack([ring(-4.95, 1.40, .78), [(1.75, -5.8, .1), (2.10, -7.2, .1), (2.35, -9.0, .1), (.12, -9.0, .5), (0, -6.6, .7), (-.12, -9.0, .5), (-2.35, -9.0, .1), (-2.10, -7.2, .1), (-1.75, -5.8, .1)]]); P = A.P3(tr)
            cel.fill(P, pal['pants']); cel.clip(P)
            for q in (-1, 1):
                for pp in ([(.25, -5.1), (.55, -5.9), (.50, -7.4), (.70, -9.2), (.20, -9.2), (.18, -6.6)], [(.95, -5.1), (1.30, -5.8), (1.45, -7.0), (1.70, -9.2), (1.35, -9.2), (1.15, -6.8)]): cel.fill(A.P3(np.array([(q * x, y, .5) for x, y in pp])), pal['pantsS'])
                cel.fill(A.P3(np.array([(q * x, y, .4) for x, y in ((.70, -5.2), (.86, -5.9), (.92, -7.4), (1.0, -7.6), (.98, -5.9))])), pal['pantsH'], .8)
            cel.unclip(); outline(cel, P, lw * .95, ln, 21)
            for q in (-1, 1):
                for pp in ([(.30, -5.1), (.60, -6.0), (.62, -7.6)], [(1.0, -5.1), (1.36, -6.0), (1.55, -7.6)], [(.05, -5.2), (.10, -6.0), (.04, -6.6)]): cel.line(A.P3(spl(np.array([(q * x, y, .5) for x, y in pp]), n=8)), lw * .45, ln, taper=(.2, .7))
            top, bot = ring(-4.02, 1.40, .80), ring(-4.98, 1.47, .84); P = A.P3(np.vstack([top, bot[::-1]]))
            cel.fill(P, pal['white']); cel.clip(P); cel.fill(A.P3(np.array([(sd * .35, -3.8, .8), (sd * .25, -4.3, .8), (sd * .50, -4.6, .8), (sd * .40, -5.3, .8), (sd * 2.0, -5.3, 0), (sd * 2.0, -3.8, 0)])), pal['whiteS'])
            for q in (-1, 1): cel.fill(A.P3(np.array([(q * 1.45, -4.30, .1), (q * .7, -4.36, .6), (q * .2, -4.50, .7), (q * .7, -4.44, .6), (q * 1.45, -4.42, .1)])), pal['whiteS'])
            cel.unclip(); outline(cel, P, lw * .95, ln, 9, press=.4)
            for y_, a0, a1 in ((-4.26, .02, .62), (-4.44, .30, .98), (-4.62, .0, .48), (-4.80, .42, .95)):
                R_ = ring(y_, 1.44, .82, n=40); cel.line(A.P3(R_[int(a0 * 40):int(a1 * 40)] + np.c_[np.zeros(int(a1 * 40) - int(a0 * 40)), .03 * np.sin(np.linspace(0, 4, int(a1 * 40) - int(a0 * 40))), np.zeros(int(a1 * 40) - int(a0 * 40))]), lw * .4, ln, taper=(.4, .5))

    # ------------------------------------------------------------ arm
    UP = [(-.40, .10), (-.26, .36), (-.08, .58), (.08, .68), (.22, .66), (.34, .55), (.42, .48), (.54, .54), (.68, .60), (.80, .54), (.92, .42), (1.02, .37)]   # deltoid, biceps (top edge)
    LO = [(-.22, -.20), (.0, -.38), (.16, -.46), (.34, -.56), (.50, -.64), (.66, -.58), (.82, -.45), (.94, -.37), (1.02, -.35)]                               # triceps (bottom edge)
    FT = [(-.10, .34), (.06, .48), (.20, .54), (.38, .48), (.56, .37), (.74, .28), (.90, .235), (1.02, .235)]
    FB = [(-.10, -.33), (.06, -.40), (.22, -.47), (.40, -.44), (.58, -.35), (.76, -.26), (.90, -.22), (1.02, -.22)]

    def arm(self, cel, b, S_, E_, W_, pal, lw, side=1, up=1, lit=-1, seed=0, hand=None, bulk=1.15, root=True):
        """one bare arm.  S_,E_,W_ shoulder / elbow / wrist in body units (3-D).  up = +1 when the biceps is on the counter-clockwise side of the
        bone seen on screen, -1 otherwise.  hand(cel, wrist_px, elbow_px, px_per_unit) is called at the right depth."""
        ln, skl = pal['line'], pal['sline']; p = b.P3(np.array([S_, E_, W_], float)); dz = (np.array([S_, E_, W_], float) @ b.R.T)[:, 2]
        fz = b.s / (1 - np.clip(dz * b.k, -.5, .5)) * bulk; S2, E2, W2 = p
        def seg(a, c, fa, fc, P):
            u = (c - a); L_ = np.hypot(*u); u = u / (L_ + 1e-6); n = np.array([u[1], -u[0]]) * up
            P = np.asarray(P, float); r = fa + (fc - fa) * np.clip(P[:, 0:1], 0, 1); return a + u * L_ * P[:, 0:1] + n * r * P[:, 1:2]
        U_ = lambda P: seg(S2, E2, fz[0], fz[1], P); F_ = lambda P: seg(E2, W2, fz[1], fz[2], P)
        Ld = np.array([lit * .75 if lit else 0, -.66])
        def litside(a, c): u = (c - a) / (np.hypot(*(c - a)) + 1e-6); return np.array([u[1], -u[0]]) * up @ Ld
        dn = float(np.clip((E2 - S2)[1] / (np.hypot(*(E2 - S2)) + 1e-6) * 1.25, 0, 1)); UPk = np.array(self.UP, float); UPk[:4] = UPk[:4] * (1 - dn) + np.array([(-.245, -.16), (-.235, .10), (-.15, .38), (.0, .60)]) * dn
        ut, ul = spl(UPk, n=7), spl(np.array(self.LO), n=7); Pu = U_(np.vstack([ut, ul[::-1]]))
        Pf = F_(np.vstack([spl(np.array(self.FT), n=7), spl(np.array(self.FB), n=7)[::-1]])); disc = E2 + np.c_[np.cos(np.linspace(0, 6.283, 24)), np.sin(np.linspace(0, 6.283, 24))] * fz[1] * .345
        hand_first = hand is not None and fz[2] < fz[1] * .97
        if hand_first: hand(cel, W2, E2, fz[2])
        # flats
        cel.fill(disc, pal['skin']); cel.fill(Pu, pal['skin']); cel.clip(Pu); t = litside(S2, E2)
        if t >= 0:
            cel.fill(U_([(-.1, -.9), (.0, -.10), (.14, -.20), (.30, -.06), (.44, -.24), (.60, -.10), (.76, -.22), (.92, -.06), (1.1, -.10), (1.1, -.9)]), pal['skinS']); cel.fill(U_([(.04, -.8), (.18, -.36), (.40, -.42), (.60, -.48), (.80, -.38), (1.1, -.28), (1.1, -.9)]), pal['skinD'], .6)
            cel.fill(U_([(.28, .74), (.34, .40), (.40, .18), (.47, .36), (.46, .7)]), pal['skinS']); cel.fill(U_([(.84, .6), (.88, .24), (.98, .05), (1.1, .1), (1.1, .6)]), pal['skinS'])
            cel.fill(U_([(.0, .42), (.12, .54), (.24, .50), (.14, .38)]), pal['skinH']); cel.fill(U_([(.56, .40), (.68, .47), (.78, .42), (.66, .34)]), pal['skinH'])
        else:
            cel.fill(U_([(-.3, .9), (-.02, .30), (.14, .40), (.30, .22), (.40, .04), (.50, .26), (.66, .22), (.82, .06), (.96, .12), (1.1, .9)]), pal['skinS']); cel.fill(U_([(-.1, .8), (.08, .54), (.28, .48), (.42, .34), (.66, .44), (.9, .30), (1.1, .9)]), pal['skinD'], .55)
            cel.fill(U_([(.34, -.22), (.52, -.06), (.70, -.14), (.60, -.26)]), pal['skinS']); cel.fill(U_([(.30, -.42), (.50, -.50), (.66, -.44), (.48, -.36)]), pal['skinH'])
        cel.unclip(); cel.fill(Pf, pal['skin']); cel.clip(Pf); t2 = litside(E2, W2)
        if t2 >= 0:
            cel.fill(F_([(-.2, -.8), (-.04, -.06), (.16, -.16), (.34, -.02), (.50, -.12), (.70, -.02), (.88, -.08), (1.1, -.02), (1.1, -.8)]), pal['skinS']); cel.fill(F_([(0, -.8), (.14, -.32), (.40, -.32), (.62, -.24), (.9, -.15), (1.1, -.8)]), pal['skinD'], .55); cel.fill(F_([(.10, .30), (.22, .40), (.40, .34), (.24, .26)]), pal['skinH'])
        else:
            cel.fill(F_([(-.2, .8), (-.04, .08), (.18, .20), (.36, .06), (.54, .12), (.74, .02), (.9, .08), (1.1, .02), (1.1, .8)]), pal['skinS']); cel.fill(F_([(0, .8), (.12, .38), (.36, .34), (.62, .22), (.9, .16), (1.1, .8)]), pal['skinD'], .55); cel.fill(F_([(.12, -.28), (.26, -.35), (.42, -.31), (.26, -.22)]), pal['skinH'])
        cel.unclip()
        # shadow the biceps throws into the elbow
        cel.clip(Pf); cel.fill(F_([(-.12, .60), (.05, .46), (.11, .22), (.05, .02), (-.02, .10), (-.12, .20)]), pal['skinS']); cel.unclip()
        # ---- web of the armpit: arm grows out of the trunk
        if root:
            q_ = 1. if S_[0] > 0 else -1.; lt = b.P3(np.array([(q_ * 2.04, -1.50, -.05), (q_ * 1.86, -1.02, .1), (q_ * 1.80, -.55, .2)])); wb = U_([(.24, -.50), (.10, -.42), (.02, -.10)])
            web = np.vstack([spl([wb[0], wb[1] * .5 + lt[0] * .5 + (wb[0] - wb[2]) * .10, lt[0]], n=8), lt[1:], wb[2:]]); cel.fill(web, pal['skinS'])
            cel.line(spl([wb[0], wb[1] * .5 + lt[0] * .5 + (wb[0] - wb[2]) * .10, lt[0]], n=8), lw * .95, ln, taper=(.2, .2), seed=seed + 60)
            cel.line(spl([wb[1], (wb[1] + lt[1]) / 2, lt[1]], n=8), lw * .4, skl, taper=(.3, .6), seed=seed + 61)
        # ---- one silhouette for the whole limb, open at the root
        big = max(union_contour([Pu, disc, Pf], close=7), key=len); uu = (E2 - S2); L_ = np.hypot(*uu); uu = uu / L_; tt = (big - S2) @ uu / L_; nn = np.array([uu[1], -uu[0]]) * up; vv = (big - S2) @ nn / fz[0]
        wr = np.hypot(*(big - W2).T) < fz[2] * .20; skip = ((tt < -.34) | ((tt < .14) & (vv < .10))) if root else np.zeros(len(big), bool)
        skip = skip | (((big - W2) @ ((W2 - E2) / np.hypot(*(W2 - E2)))) > fz[2] * .015)
        k0 = int(np.argmax(skip)) if skip.any() else 0; big = np.roll(big, -k0, 0); skip = np.roll(skip, -k0); i = 0
        while i < len(big):
            if skip[i]: i += 1; continue
            j = i
            while j < len(big) and not skip[j]: j += 1
            if j - i > 8: cel.line(big[i:j], lw * 1.0, ln, taper=(.06, .05), press=.45, seed=seed + i % 7, tmin=.2)
            i = j
        # ---- inner lines
        for pts, w in (([(-.20, .30), (-.10, .02), (.04, -.10), (.18, .04), (.28, .26), (.38, .46)], .60), ([(.36, .46), (.50, .20), (.62, .0), (.80, -.08), (.97, -.04)], .52), ([(.24, -.12), (.44, -.06), (.60, -.12)], .34), ([(.40, -.34), (.58, -.28), (.76, -.31)], .36), ([(.04, .36), (.14, .42), (.24, .38)], .26), ([(.02, .18), (.14, .22), (.24, .18)], .22), ([(.82, .36), (.91, .22), (1.0, .16)], .40), ([(.58, .40), (.68, .36), (.76, .28)], .26)):
            cel.line(U_(spl(np.array(pts, float), n=8)), lw * w, skl, taper=(.35, .55), press=.3, seed=seed + int(w * 100))
        for pts, w in (([(-.02, .12), (.14, .18), (.34, .10), (.56, .12), (.78, .10)], .46), ([(.02, -.14), (.22, -.06), (.44, -.12), (.70, -.06)], .36), ([(.50, .02), (.70, .0), (.92, .03)], .30), ([(.62, -.10), (.80, -.06), (.96, -.08)], .28), ([(-.10, .36), (.0, .24), (.0, .02)], .46), ([(.16, .34), (.30, .30), (.44, .24)], .24)):
            cel.line(F_(spl(np.array(pts, float), n=8)), lw * w, skl, taper=(.3, .6), press=.3, seed=seed + 5 + int(w * 100))
        if hand is not None and not hand_first: hand(cel, W2, E2, fz[2])
        return p, fz


    # ------------------------------------------------------------ poses
    # shoulder / elbow / wrist for both arms (picture right = q +1), up = biceps side; hands = (pose, yaw, pitch, roll) in hands.py terms
    POSES = dict(
        spread=dict(R=((2.08, -.34, -.1), (3.58, .10, .45), (4.62, 1.22, .85), 1), L=((-2.08, -.34, -.1), (-3.50, -.66, 1.0), (-4.12, .02, 2.1), -1), hR=('open', 188, 16, 8), hL=('open', 168, 24, -8)),
        windup=dict(R=((2.05, -.38, -.05), (3.05, -1.45, .55), (2.35, -.75, 1.75), 1), L=((-2.05, -.38, -.05), (-3.05, -1.45, .55), (-2.35, -.75, 1.75), -1), hR=('slap', 112, 10, 0), hL=('slap', 68, 10, 0)),
        near=dict(R=((2.0, -.40, -.0), (2.52, -2.10, .65), (1.22, -.98, 1.85), 1), L=((-2.0, -.40, -.0), (-2.52, -2.10, .65), (-1.22, -.98, 1.85), -1), hR=('slap', 104, 10, 0), hL=('slap', 76, 10, 0)),
        clap=dict(R=((1.98, -.42, .0), (2.10, -2.25, .75), (.44, -1.72, 1.85), 1), L=((-1.98, -.42, .0), (-2.10, -2.25, .75), (-.44, -1.72, 1.85), -1), both='clap'),
        crossed=dict(R=((2.06, -.40, -.05), (2.25, -2.05, .55), (.15, -1.62, 1.35), 1), L=((-2.06, -.40, -.05), (-2.25, -2.15, .55), (.55, -1.98, 1.60), -1), hR=None, hL=('rest', 18, 8, -18)),
        flex=dict(R=((2.10, -.30, -.1), (3.35, -.50, .35), (2.95, 1.05, .70), 1), L=((-2.05, -.40, -.05), (-2.75, -1.95, .20), (-1.75, -3.05, .90), -1), hR=('fist', 150, 10, 0), hL=('fist', 30, 20, 0)),
        hang=dict(R=((2.08, -.42, -.1), (2.70, -2.15, -.05), (2.62, -3.75, .45), 1), L=((-2.08, -.42, -.1), (-2.70, -2.15, -.05), (-2.62, -3.75, .45), -1), hR=('fist', 80, 0, 0), hL=('fist', 100, 0, 0)))

    @staticmethod
    def mix_pose(a, b, t):
        """inbetween of two arm poses (positions only; hands switch at t .5)"""
        A, B = (_TodoBody.POSES[q] if isinstance(q, str) else q for q in (a, b)); o = dict(A if t < .5 else B)
        for k in ('R', 'L'): o[k] = tuple(tuple(np.asarray(A[k][i]) * (1 - t) + np.asarray(B[k][i]) * t) for i in range(3)) + ((A if t < .5 else B)[k][3],)
        return o

    def figure(self, cel, H_, b, pose='hang', expr='focus', light='day', lit=-1, lw=None, f=0, wind=0., seed=0, hands=True, bulk=1.2, front=('R', 'L'), head=True, sash=True):
        """whole upper figure on a cel.  front = order in which arms are drawn (last on top)"""
        pal = PAL[light] if isinstance(light, str) else light; lw = lw or max(1.4, H_.s / 300. * 8.0); po = self.POSES[pose] if isinstance(pose, str) else pose
        kw = dict(f=f, wind=wind, seed=seed, lit=lit, lw=lw)
        if head: self.head(cel, H_, expr, pal, part='back', **kw)
        low = {k: (np.asarray(po[k][1]) @ b.R.T)[2] < -.1 for k in ('R', 'L')}
        def arm(k):
            q = 1 if k == 'R' else -1; S_, E_, W_, up = po[k]; h = po.get('h' + k)
            hd = (lambda c, w, e, fz: hand3(c, w, e, fz, pal, lw, 'L' if q > 0 else 'R', h[0], h[1], h[2], h[3], .72, seed=seed + q)) if (hands and h) else None
            self.arm(cel, b, S_, E_, W_, pal, lw, q, up, lit, seed + 3 + q, hd, bulk)
        for k in ('R', 'L'):
            if low[k]: arm(k)
        self.trunk(cel, b, pal, lw, lit, seed, sash)
        self.neck(cel, H_, b, pal, lw, lit, seed)
        if head: self.head(cel, H_, expr, pal, part='front', **kw)
        for k in front:
            if not low[k]: arm(k)
        if hands and po.get('both') == 'clap': self.clap_hands(cel, b, po, pal, lw, bulk, seed)
        return po

    def clap_hands(self, cel, b, po, pal, lw, bulk=1.2, seed=0, gap=0.):
        """the two palms together (hands.two('clap')), seated on both wrists"""
        try: from jj import hands as HD
        except Exception: return
        wR, wL = b.P3(np.array(po['R'][2]))[None][0], b.P3(np.array(po['L'][2]))[None][0]; eR, eL = b.P3(np.array(po['R'][1])), b.P3(np.array(po['L'][1]))
        z = (np.array(po['R'][2]) @ b.R.T)[2]; fz = b.s / (1 - min(.5, z * b.k)) * bulk; m = (wR + wL) / 2; s = fz * .72
        hp = dict(skin=pal['skin'], shade=pal['skinS'], line=pal['line'], nail=hexmix(pal['skin'], '#ffffff', .22), nail_shade=hexmix(pal['skinS'], '#ffffff', .15), light=pal['skinH'])
        cam = HD.Cam(m[0], m[1] - s * .62, s, 6.0); hs = HD.two('clap', R=HD.Rx(HD.D(22)), bld=HD.HEAVY, gap=gap)
        y0 = m[1] + s * .10; cel.clip(np.array([(m[0] - 4000, y0), (m[0] + 4000, y0), (m[0] + 4000, y0 - 4000), (m[0] - 4000, y0 - 4000)]))
        g = HD.render(hs, cam, (-.6, .7, .5), res=480); HD.ink(cel, g, hp, max(1.2, lw * .85), seed); cel.unclip()

    def bust(self, cx, cy, s, yaw=0., pitch=0., roll=0., byaw=None, bpitch=.0, broll=None, pose='hang', expr='focus', light='day', f=0, wind=0., seed=0, lit=-1, lw=None, persp=.08, hands=True, w=W, h=H, ss=2, front=('R', 'L')):
        c = Cel(w, h, ss=ss); H_ = self.pose(cx, cy, s, yaw, pitch, roll)
        b = TBody.under(H_, yaw * .45 if byaw is None else byaw, bpitch, roll * .5 if broll is None else broll, drop=1.28, fwd=.10, persp=persp)
        self.figure(c, H_, b, pose, expr, light, lit, lw, f, wind, seed, hands, front=front)
        return c.layer(h, w)


_TodoBody.torso = _TodoBody.trunk


None_H = [None]
Todo = _TodoBody()
