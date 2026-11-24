"""G3  Curse and blessing, out of the same mouth.  The hill of b1 in spring and evening gold: pull back from Gojo to everybody.  72 f."""
import numpy as np, cv2, math
from jj import comp, kit, W, H
from jj.paint import rgb, f32, smooth
from jj.grass import Hill, Foreground, motes, PW, PH, Q, GW, GH
from jj.ink import spl
from chars import chibi as C
from shots.b1 import torii, e, out

NOD, PAGE, LAND, PETAL, FLOP, LOWER = 54, 36, 63, 27, 27, 45
K0 = .748                                                     # b1's last framing, as seen by this shot's first


def mp(p): return ((p[0] - 1154) * K0 + 662, (p[1] - 644) * K0 + 716)


# who, pose, x, y, S, rot, kwargs, breath period, breath phase (painting space, back to front)
CAST = [('nanami', 'sit', 1738, 662, 84, .05, dict(seed=4, py=.40, lL=[(-.30, .90), (-.54, 1.16), (-.62, 1.36)], lR=[(.30, .90), (.54, 1.16), (.62, 1.36)]), 50, .3),
        ('panda', 'starfish', 1424, 712, 110, -.05, dict(seed=5, aL=[(-.44, .2), (-.92, .16), (-1.20, .34)]), 46, 2.0),
        ('todo', 'pillow', 1672, 806, 84, -.98, dict(seed=3), 35, 4.1),
        ('megumi', 'curl', 1752, 1012, 84, .06, dict(seed=8), 41, 3.0),
        ('nobara', 'curl', 1602, 985, 82, .10, dict(seed=7), 32, 1.1),
        ('yuji', 'pillow', 1462, 918, 84, .46, dict(seed=6), 29, 5.2),
        ('gojo', 'starfish', 1006, 790, 86, .58, dict(seed=1), 36, -1.2),
        ('sukuna', 'crossed', 1430, 474, 44, .02, dict(seed=10), 46, .9)]


def hill():
    far = [(262, 300, 30), (318, 282, 44), (388, 296, 36), (452, 300, 24), (214, 306, 20), (1672, 318, 30), (1728, 300, 46), (1800, 322, 36), (1034, 196, 17), (1076, 190, 22), (1120, 197, 16), (604, 204, 13), (640, 200, 17), (1560, 214, 15), (1598, 208, 19)]
    tr = [(*mp((x, y)), r * .8) for x, y, r in far] + [(1640, 360, 24), (1690, 346, 34), (1752, 358, 28), (1990, 372, 40), (2060, 360, 30), (2110, 378, 22), (1400, 250, 15), (1440, 246, 19), (1960, 238, 16), (2004, 232, 22), (860, 236, 14), (900, 232, 18)]
    st = [([mp(q) for q in P], w * .8, v) for P, w, v in [([(-40, 196), (200, 176), (470, 196), (760, 168)], 11, -.20), ([(-40, 214), (220, 196), (500, 214), (720, 192)], 10, .18), ([(520, 318), (700, 286), (900, 268), (1120, 262)], 10, .16)]]
    st += [([(1180, 330), (1420, 318), (1700, 326), (2260, 296)], 12, -.20), ([(1180, 352), (1420, 340), (1700, 350), (2260, 322)], 11, .18), ([(1500, 420), (1760, 404), (2000, 410), (2260, 396)], 12, .16), ([(-40, 250), (400, 226), (900, 240), (1300, 214)], 10, -.16), ([(-40, 268), (400, 246), (900, 258), (1300, 234)], 9, .14)]
    G = Hill('gold', seed=11, crest=[(-400, 660), mp((-60, 540)), mp((360, 430)), mp((800, 322)), mp((1240, 316)), (1060, 512), (1400, 520), (1800, 492), (2320, 540)], crest2=[(-60, 400), mp((420, 292)), mp((900, 212)), (960, 408), (1400, 372), (1900, 396), (2320, 350)],
             blobs=[(1500, 900, 700, 330, -4, .06, .8), (300, 1060, 520, 260, 14, -.15, .6), (640, 740, 460, 90, -9, -.06, .7), (2050, 1020, 380, 220, -20, -.10, .6), (1400, 1200, 900, 110, 0, -.08, .8)],
             quiet=[(1020, 770, 235), (1560, 860, 360)], trees=tr, strips=st,
             far_blobs=[(700, 440, 420, 30, -4, .10, .6), (1700, 440, 520, 30, 3, .08, .5), (1200, 300, 520, 30, 2, .06, .6)],
             carpets=[(470, 700, 250, 70, -14, 90, 'pink'), (250, 930, 220, 70, 10, 70, 'daisy'), (560, 1040, 220, 60, -6, 60, 'butter'), (2030, 760, 220, 70, 18, 80, 'daisy'), (2060, 980, 200, 80, 30, 80, 'pink'), (1200, 620, 200, 34, -4, 60, 'violet'), (1890, 1150, 260, 60, -8, 70, 'butter'), (900, 1140, 320, 60, 6, 80, 'pink'), (1430, 1180, 300, 44, -4, 70, 'daisy'), (150, 660, 150, 34, -8, 50, 'violet'), (2000, 600, 170, 26, 4, 50, 'pink'), (1120, 900, 110, 60, 20, 30, 'daisy')],
             drifts=[(720, 600, 200, 30, 12, 'daisy'), (700, 900, 160, 50, 10, 'butter'), (1180, 1040, 200, 40, 10, 'violet')],
             cloud_seed=3, clouds=.45, lam=1080., period=18., weak=.72, wave=1.5, gk=1.75, level=-.045, wind=14., gust_ref=(2, 330, 760), sunny=[(1006, 780, 420), (1560, 820, 560)], haze=.55, far_air=(.24, .16))
    G.setup()
    G.add_path([mp(q) for q in [(-40, 330), (150, 318), (300, 346), (470, 336), (640, 290), (800, 250), (940, 236)]], 3.5, 5.5, col='#fff0cc', amt=.75)
    G.add_path([(506, 408), (640, 424), (800, 446), (960, 452), (1160, 440), (1400, 430), (1640, 440), (1900, 420), (2260, 430)], 4.5, 8., col='#fff0cc', amt=.7)
    return G


class Shot:
    id = 'g3'; n = 72
    finish = dict(bloom=.30, vig=.14, halation='#ffd9a8', grade=('#3a2f4a', (1.05, .995, .92), '#ffe9bd', .34))
    EVENTS = [(0, 'wind_bed', dict(dur=72, size=.35, note='evening hill air, warm; a distant bell at the gate would fit on 0')),
              (0, 'gust', dict(size=1.0, dur=30, pan=(-1, 1), note='grass wave on the downbeat, as in b1')), (18, 'gust', dict(size=.7, dur=30, pan=(-1, 1))), (36, 'gust', dict(size=1.0, dur=30, pan=(-1, 1))), (54, 'gust', dict(size=.7, dur=18, pan=(-1, .4))),
              (0, 'snore', dict(size=.4, dur=18, pan=.2, note='Gojo')), (8, 'breaths', dict(size=.35, dur=64, pan=(0, .2), note='seven sleepers, deep, all out of phase, fade in as the frame widens')),
              (0, 'butterfly_flutter', dict(size=.2, dur=62, pan=(.1, .1))),
              (17, 'ear_flick', dict(size=.2, pan=.5, note='Panda, two quick flicks 17 and 19')), (40, 'ear_flick', dict(size=.2, pan=.15, note='40 and 42')), (65, 'ear_flick', dict(size=.15, pan=.2)),
              (22, 'foot_kick', dict(size=.3, pan=.35, note='Yuji, dreaming; cloth and grass, 22 and 24')),
              (24, 'arm_swing', dict(size=.3, pan=.5, note="Nobara's arm comes up")), (FLOP, 'slap', dict(size=.7, pan=.45, note="her hand flops onto Megumi's face: soft comic smack")), (29, 'grumble', dict(size=.3, dur=8, pan=.45, note="Megumi's brow, through his nose")), (33, 'lip_smack', dict(size=.25, pan=.4, note='Nobara')),
              (PETAL, 'petal_land', dict(size=.2, pan=-.2, note='a petal on the nose')), (45, 'snort', dict(size=.3, pan=-.35, note='Gojo: puff lifts the petal, it settles again at 51')),
              *[(f_, 'scratch', dict(size=.35, pan=.45, note='Todo, belly')) for f_ in (31, 33, 35, 37, 60, 62, 64)],
              (34, 'paper', dict(size=.3, pan=.55, note='Nanami lifts the page')), (PAGE, 'page_turn', dict(size=.6, dur=8, pan=.55, note='page comes over, lands 44')), (LOWER, 'paper_lower', dict(size=.4, dur=6, pan=.5, note='he lowers the paper and looks at them')), (54, 'breath_out', dict(size=.25, pan=.5, note='Nanami, almost a laugh')),
              (30, 'sukuna_droop', dict(size=.2, pan=.2, note='eyelids')), (39, 'sukuna_jerk', dict(size=.3, pan=.2, note='head dips and snaps back')), (NOD, 'sukuna_nod', dict(size=.5, pan=.2, note='nods off, tiny thump')),
              (50, 'foot_kick', dict(size=.3, pan=.15, note='Yuji again, 50 and 52')), (58, 'grumble', dict(size=.2, pan=.4, note='Megumi')), (LAND, 'butterfly_land', dict(size=.3, pan=.2, note='settles on his hair, almost a chime'))]

    # -------------------------------------------------- setup
    def setup(self):
        self.G = hill()
        masks = []; self.beds = []
        for who, pose, x, y, S, rot, kw, P, ph in CAST:
            if who == 'sukuna': continue
            rgba, x0, y0, _ = C.solo(who, pose, x, y, S, rot=rot, t=0, breath=0., **kw); a = rgba[..., 3]; h, w = a.shape; lay = np.zeros((PH, PW), f32)
            X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(PW, x0 + w), min(PH, y0 + h); R = a[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0].copy()
            if who == 'nanami': yy = np.arange(Y0, Y1)[:, None]; R = R * smooth(y + 88, y + 100, yy.astype(f32))
            lay[Y0:Y1, X0:X1] = R; masks.append(lay); self.beds.append(who)
        ex = np.zeros((GH, GW), f32); X, Y = np.meshgrid(np.arange(GW) * Q, np.arange(GH) * Q)
        ex = np.clip(1 - np.sqrt(((X - 1828) / 165.) ** 2 + ((Y - 796) / 46.) ** 2), 0, 1).astype(f32) ** .6 * .75
        self.seat = self.G.seat(masks, sun=(21, 15), extra=ex); del masks
        one = np.ones((H, W, 4), f32); t = C.daylight(one, 120, 1000, bounce='#e6e8b4', amt=.5); t = kit.para(t, (0, 0), (W, H * .5), [(0, '#fff3d6'), (.6, '#fff0dc'), (1, '#f2dcd8')], 'mul', .8); self.tone = t[..., :3].astype(f32)
        blk = np.zeros((H, W, 3), f32); g = comp.glow(blk, -20, -160, 1560, '#ffab55', .52, power=1.7); g = comp.glow(g, W * .5, 60, 1300, '#ffe3b0', .10, power=1.5); self.sun = (1 - g).astype(f32)
        self.fg = Foreground([(-60, 1230, 380, 30, .15, 'pink'), (190, 1240, 260, 26, .3, None), (1960, 1230, 360, 26, -.3, 'pink'), (1740, 1220, 220, 14, -.1, 'daisy')], season='gold', seed=14, blur_px=7)

    def cam(self, f):
        t = np.clip((f - 2) / 41., 0, 1); t = t * t * t * (t * (6 * t - 15) + 10); s = max(0., f - 40) / 32.
        return (700 + 562 * t - 4 * s, 702 - 12 * t, math.exp(math.log(1.6) * (1 - t)) * (1 + .012 * s * s), .004 * (1 - t))

    def _g(self, f, x, y):
        return float(self.G.fields(f)[0][int(np.clip(y / Q, 0, GH - 1)), int(np.clip(x / Q, 0, GW - 1))])

    # -------------------------------------------------- acting
    def sukuna(self, f):
        """(expr, lid, hy, hroll, pitch)"""
        if f < 22: return 'grump', .25, 0., 0., 0.
        if f < 37: q = e((f - 22) / 14.); return 'grump', .25 + .55 * q, .04 * q, .03 * q, .06 * q
        if f < 40: q = e((f - 36) / 3.); return 'grump', .80 + .15 * q, .04 + .12 * q, .03 + .10 * q, .06 + .16 * q       # dips
        if f < 43: q = (f - 39) / 3.; return 'grump', [.2, .12, .2][f - 40], [-.02, -.05, -.01][f - 40], [-.04, -.02, 0][f - 40], 0.   # snaps back
        if f < NOD: q = e((f - 42) / 12.); return 'grump', .2 + .72 * q, .05 * q, .05 * q + .02 * math.sin(f * .5) * q, .08 * q
        j = f - NOD; d = [.30, .74, 1.10, 1.06, .96, .98, 1.0][min(j, 6)]; b = .02 * math.sin(2 * math.pi * (f - 60) / 30.) if f > 60 else 0.
        return 'sleep', 1., .23 * d + b, .22 * d, .28 * d

    @staticmethod
    def br(f, P, ph):
        """a sleeper's breath: quick in, long out"""
        u = (f / P + ph / (2 * math.pi)) % 1.; return -math.cos(math.pi * u / .38) if u < .38 else math.cos(math.pi * (u - .38) / .62)

    def act(self, who, f, b, g):
        """what this one does in its sleep -> kwargs"""
        k = dict(deep=2.8, mb=b, hem=.05 + .22 * g)
        if who == 'panda':                                            # ear flicks (2 f each), belly breath
            fl = lambda f0: {0: .0, 1: .42, 2: .05, 3: .34, 4: .0}.get(f - f0, 0.)
            k['ear'] = (fl(40) + fl(64), fl(16)); k['deep'] = 2.4
        elif who == 'todo':                                           # scratch, three strokes, rest, three more
            q = {30: .4, 31: 1., 32: 0., 33: -1., 34: 0., 35: 1., 36: 0., 37: -1., 38: -.4, 59: .4, 60: 1., 61: 0., 62: -1., 63: 0., 64: 1., 65: .3}.get(f, 0.)
            k['aL'] = [(-.44, .22), (-.84 - .02 * abs(q), .52 + .05 * q), (-.30 + .10 * q, .70 + .14 * q)]; k['hroll'] = .03 * math.sin(f * .16)
            if 38 <= f < 58: k['mouth'] = {39: 'wave', 40: 'wave', 42: 'flat', 43: 'flat'}.get(f, None)
        elif who == 'yuji':                                           # dream kicks, like a puppy
            r0 = np.array([(.30, .95), (.42, 1.32), (.50, 1.64)]); up = np.array([(.30, .95), (.66, 1.12), (.46, 1.36)]); out_ = np.array([(.30, .95), (.60, 1.26), (.96, 1.50)])
            T = lambda f0: {-1: (.5, 0.), 0: (1., 0.), 1: (.2, .8), 2: (0., 1.), 3: (.8, .0), 4: (.1, .9), 5: (0., 1.), 6: (0., .6), 7: (0., .25), 8: (0., .08)}.get(f - f0, (0., 0.))
            u1, u2 = [x + y for x, y in zip(T(21), T(49))]; L = r0 * (1 - u1 - u2) + up * u1 + out_ * u2
            k['lR'] = [tuple(v) for v in L]; k['toe'] = (-.5, .55 + .5 * u2 - .4 * u1); k['hroll'] = .04 * u2; k['mouth'] = 'laugh' if 52 <= f < 60 else None
        elif who == 'nobara':                                         # the hand
            c0 = [(.62, .30), (.54, .64), (.14, .62)]; K = {23: [(.62, .30), (.78, .58), (.44, .66)], 24: [(.62, .28), (.96, .06), (1.02, -.34)], 25: [(.62, .27), (1.06, -.14), (1.42, -.52)], 26: [(.62, .28), (1.16, -.06), (1.72, -.26)],
                 27: [(.62, .31), (1.18, .20), (1.62, .15)], 28: [(.62, .31), (1.18, .23), (1.60, .19)], 29: [(.62, .31), (1.18, .17), (1.62, .10)], 30: [(.62, .31), (1.18, .20), (1.62, .15)]}
            if f >= 23:
                k['armU'] = K.get(f, [(.62, .31), (1.18, .20 - .012 * b), (1.62 + .01 * b, .15 - .015 * b)]); k['hkU'] = 'open' if f < 27 else 'flat'; k['hrU'] = {24: -.5, 25: .2, 26: .5}.get(f, .10 + (.12 if 47 <= f < 49 or 51 <= f < 53 else 0.))
            k['hroll'] = {24: -.03, 25: -.05, 26: -.02, 27: .04, 28: .05, 29: .03}.get(f, 0.); k['mouth'] = {33: 'open', 34: 'open', 35: 'flat', 36: 'open', 37: 'open', 38: 'flat'}.get(f, 'smile' if f > 38 else None)
        elif who == 'megumi':                                         # takes it
            j = f - FLOP; k['hy'] = {0: .03, 1: .07, 2: .03, 3: -.02, 4: 0.}.get(j, 0.); k['hx'] = {0: .02, 1: .05, 2: .02}.get(j, 0.); k['hroll'] = {0: .04, 1: .08, 2: .05, 3: .0}.get(j, 0.)
            if j >= 1:
                tw = {1: .3, 2: 1., 3: .55, 4: 1., 5: 1., 6: .45, 7: .9, 8: .9, 9: .8}.get(j, .72 if j < 24 else .55); tw = tw + (.35 if f in (45, 46, 49, 50, 58, 59, 62, 63) else 0.); k['brow'] = min(1., tw); k['mouth'] = 'frown'
                k['vein'] = {2: .5, 3: 1.25, 4: 1.}.get(j, 1. if 4 < j < 17 else (.92 if f in (58, 59, 60, 61, 62, 63) else 0.))
        elif who == 'gojo':
            k['mouth'] = {45: 'open', 46: 'gape', 47: 'open', 48: 'wave', 49: 'wave', 50: 'cat', 51: 'cat'}.get(f, None); k['hy'] = {45: -.03, 46: -.06, 47: -.03, 48: .01}.get(f, 0.)
            w_ = {18: .06, 19: .10, 20: .02, 21: .08, 22: 0., 57: .07, 58: .0, 59: .08}.get(f, 0.); k['aL'] = [(-.44, .18), (-.94, .28 - w_ * .5), (-1.38, .04 - w_)]; k['hroll'] = -.05 * e((f - 46) / 8.)
            k['toe'] = (-.75 + (.22 if f in (30, 31, 34, 35) else 0.), .60 + (.2 if f in (32, 33, 36, 37) else 0.))
        elif who == 'nanami':                                         # a page, then them
            u = (f - 34) / 10.; k['turn'] = {.5: .395}.get(round(u, 2), e(u)) if 0 < u < 1 else None; q = e((f - 43) / 6.); ov = .05 * math.sin(math.pi * min(1, max(0, (f - 48) / 5.)))
            k['py'] = .40 - .03 * math.sin(math.pi * min(1, max(0, (f - 41) / 3.))) + (.50 * q + ov); k['flut'] = .018 + .05 * g; k['expr'] = 'soft'; k['deep'] = 1.6
            lk = e((f - 45) / 6.); k['yaw'] = -.24 * lk; k['pitch'] = .14 * lk; k['hroll'] = -.07 * lk + .03 * (e((f - 30) / 5.) - e((f - 44) / 6.)); k['hy'] = .02 * lk - .02 * math.sin(math.pi * min(1, max(0, (f - 53) / 6.)))
        return k

    def petal(self, f, nose):
        """painting position, angle, flatness"""
        if f <= PETAL:
            u = f / PETAL; p = np.array([nose[0] - 330, nose[1] - 300]) * (1 - u) + nose * u + [46 * math.sin(u * 9.5) * (1 - u), 20 * math.cos(u * 12) * (1 - u)]
            return p, u * 7.0, .35 + .6 * abs(math.sin(u * 8))
        br = math.sin(2 * math.pi * f / 36. - 1.2)
        if 45 <= f < 51: v = (f - 45) / 6.; return nose + [10 * math.sin(v * 6), -30 * math.sin(math.pi * v)], 7 + 5 * v, .5 + .4 * abs(math.sin(v * 8))
        return nose + [0, -1.2 * br], 7.0 + (5 if f >= 51 else 0) + .12 * br, .78 + .1 * br

    def butterfly(self, f, head):
        K = [(0, (870, 600)), (8, (960, 520)), (16, (1080, 580)), (24, (1170, 480)), (32, (1230, 540)), (40, (1290, 420)), (48, (1330, 480)), (56, (1392, 352)), (LAND, tuple(head))]
        if f < LAND:
            ts = np.array([k for k, _ in K], float); S = spl(np.array([p for _, p in K], float), n=24); i = min(len(S) - 1, int(round(np.interp(f, ts, np.arange(len(ts))) * 24)))
            d = S[min(len(S) - 1, i + 6)] - S[max(0, i - 6)]; return S[i] + [5 * math.sin(f * .9), 8 * math.sin(f * 1.35)] * (np.array(1.) if f < LAND - 3 else np.array(.25)), math.atan2(d[1], d[0]) * .6, None
        return np.array(head), .2, .25 + .6 * math.sin((f - LAND) * .5)

    # -------------------------------------------------- frame
    def frame(self, f):
        cam = self.cam(f); M = Hill.matrix(cam); z = math.hypot(M[0, 0], M[1, 0]); fs = f - f % 2
        items = []; info = {}; brs = {}; pb = self.br(f, CAST[1][7], CAST[1][8])
        for who, pose, x, y, S, rot, kw, P, ph in CAST:
            sx, sy, SS, rr = C.through(M, x, y, S, rot); k = dict(kw)
            if sx < -3.4 * SS or sx > W + 3.4 * SS or sy < -3 * SS or sy > H + 3.4 * SS:
                continue
            g = self._g(f, x, y - 60); b = self.br(f, P, ph); k.update(self.act(who, f, b, g))
            if who == 'sukuna': ex, lid, hy, hr, pt = self.sukuna(f); k.update(expr=ex, lid=lid, hy=hy, hroll=hr, pitch=pt); sy -= .0288 * pb * 110 * z     # rides the Panda's head
            if who == 'panda': pb = b
            if who == 'yuji': sy -= 3.4 * pb * z; sx += 1.2 * pb * z; rr += .012 * pb               # head on a breathing belly
            if who == 'todo': sy -= 1.6 * pb * z
            brs[who] = b
            it = C.solo(who, pose, sx, sy, SS, rot=rr, t=fs + x * .1, wind=(1, .3), gust=.05 + .36 * g, wamp=.05 + .12 * g, breath=b, **k); items.append(it); info[who] = it[3]
        L = C.stack(items, sun=(10 * z, 8 * z), dark=(.66, .54, .66))
        L[..., :3] *= self.tone
        L = kit.rim(L, 2.1 * z, 2.3 * z, '#ffe6a0', .62, soft=.7)
        im = self.G.frame(f, layer=L, cam=cam, seat=self.seat, breathe=[2.2 * brs.get(w_, 0.) for w_ in self.beds])
        im = torii(im, M, 506., 408., 33., air='#f0dfbd')
        # butterfly, petal
        n0 = info['sukuna']['hn'] if 'sukuna' in info else None
        p, ha, fl = self.butterfly(f, (1443., 399.)); x, y, S, _ = C.through(M, p[0], p[1], 25.)
        d = C.D(int(x - 90), int(y - 90), 180, 180, 2); C.butterfly(d, x, y, S, f, head=ha, flap=fl, col='#fffdf0', col2='#e6cfc0')
        if f < 20: im = kit.cast(im, d.layer(outer=None), 60 * z, 70 * z, '#31584f', .20, 5)
        im = C.stamp(im, d, outer=None)
        if 'gojo' in info:
            nose = info['gojo']['head'](np.array([(.02, .40)]), 'd')[0]; Mi = cv2.invertAffineTransform(M); np_ = Mi[:, :2] @ nose + Mi[:, 2]
            pp, an, fl_ = self.petal(f, np_); x, y, S, _ = C.through(M, pp[0], pp[1], 13.); d = C.D(int(x - 60), int(y - 60), 120, 120, 2); d.lw = 1.5; c, s = math.cos(an), math.sin(an)
            Pt = spl([(-1, 0), (-.4, .52 * fl_), (.5, .62 * fl_), (1.05, .16 * fl_), (.86, 0), (1.05, -.16 * fl_), (.5, -.62 * fl_), (-.4, -.52 * fl_)], True, n=6, corners=(0, 4)); Pt = np.c_[Pt[:, 0] * c - Pt[:, 1] * s, Pt[:, 0] * s + Pt[:, 1] * c] * S + [x, y]
            d.part(Pt, '#ffc3d4', '#ee93b0', shade=[Pt * .5 + np.array([x + S * .5, y + S * .5]) * .5], w=.8, ink='#a24f6c', tip=False); im = C.stamp(im, d, outer=None)
        if 'yuji' in info and f >= 16:
            o = info['yuji']['head'](np.array([(.02, .36)]), 'd')[0]; R = (10 + 9 * (brs['yuji'] + 1) / 2 * (1 + .25 * math.sin(f * .9) * 0)) * z * e((f - 16) / 6.); c_ = o + np.array([-.95, -.95]) * R; im = C.bubble(im, c_[0], c_[1], R, f, o[0], o[1])
        a = e((f - 60) / 6.)
        if a > .02 and 'sukuna' in info:
            x, y, _, _ = C.through(M, 1484, 404, 1.); d = C.D(int(x - 60), int(y - 160), 240, 220, 2); C.zzz(d, x, y, 19 * z, f, drift=(.5, -.8)); im = C.stamp(im, d, a, outer=None)
        cx, cy = cam[0], cam[1]
        im = comp.over(im, self.fg.layer(f, gust=.3 + .3 * self._g(f, cx - 400, cy + 400), z=1 + (z - 1) * 1.25, dx=-(cx - 1262) * 1.25, dy=-(cy - 690) * 1.2 + 60 * (z - 1)))
        im = 1 - (1 - im) * self.sun
        return motes(im, f, n=54, col='#fff0c4', amt=.6, wind=(2.7, .55), petals=[('#f49ab8', '#ffd6e2'), ('#fbe9ee', '#fff8f4')])
