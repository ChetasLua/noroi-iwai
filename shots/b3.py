"""b3 COMEDY (72 f).  0-17 Yuji smiling; 18 a toothy mouth opens on his cheek; 27 it sticks its tongue out; 36 white-eyed; 45 SLAP; 54-62 the two stare; 63 the mouth is on the back of his hand, cackling."""
import numpy as np, math, cv2
from jj import comp, W, H, hands as HD
from jj.ink import Cel, spl
from jj.kit import focus_lines, screentone, speed_lines, shake
from jj.paint import Canvas, rgb, fbm, smooth, ramp, grid, arcs, f32
from chars import yuji as Y
from chars.yuji import Yuji, YHead, mixc
from chars import yuji_gag as G
A = np.array


def warm_bg(w=2200, h=1240):
    x, y = grid(h, w); t = np.clip(y / h * .92 + x / w * .18, 0, 1)
    im = ramp(t, [(0, '#8fc6de'), (.22, '#c5dfd8'), (.42, '#f6e3b4'), (.58, '#f0bf86'), (.78, '#b5946c'), (1, '#6f7257')])
    cv = Canvas(h, w, img=im); r = np.random.default_rng(14)
    n = 46; xs = r.uniform(0, w, n); ys = r.uniform(70, 430, n)        # cloud streaks
    cv.strokes(arcs(xs, ys, r.normal(-.05, .06, n), r.uniform(160, 460, n), r.normal(0, .3, n)), r.uniform(14, 44, n), rgb('#fff3d8')[None] + r.normal(0, .02, (n, 1)), op=.55, dry=.55, seed=3)
    n = 520; xs = r.uniform(-100, w + 100, n); base = 700 + 60 * np.sin(xs / 340.) + 24 * np.sin(xs / 91. + 1); ys = base + np.abs(r.normal(0, 1, n)) * 270; d = np.clip((ys - base) / 430, 0, 1)[:, None]
    cols = (1 - d) * rgb('#c9b070') + d * rgb('#4f6246') + r.normal(0, .05, (n, 1)); wm = r.random(n) < .20; cols[wm] = rgb('#f4d486') + r.normal(0, .04, (wm.sum(), 1))
    cv.strokes(arcs(xs, ys, r.uniform(-.7, .7, n), r.uniform(80, 230, n), r.normal(0, .6, n)), r.uniform(20, 56, n), cols, op=.9, dry=.2, seed=2)
    im = cv2.GaussianBlur(cv.img, (0, 0), 12)
    lay = np.zeros((h, w, 3), f32)
    for k in range(30):
        cx, cy, R = r.uniform(0, w), r.uniform(380, 860), r.uniform(14, 44); cv2.circle(lay, (int(cx * 8), int(cy * 8)), int(R * 8), tuple(float(v) for v in rgb(['#fff0c0', '#ffdf9a', '#f4f0c8'][k % 3]) * r.uniform(.06, .22)), -1, cv2.LINE_AA, 3)
    im = comp.screen(im, cv2.GaussianBlur(lay, (0, 0), 1.6))
    return (im + comp.radial(260, 130, 1500, power=2.2, h=h, w=w)[..., None] * rgb('#ffe9b4') * .34).astype(f32)


class Shot:
    id = 'b3'; n = 72
    finish = dict(bloom=.26, vig=.12)
    EVENTS = [(0, 'birds', dict(note='warm afternoon, hair in the wind, hold')), (18, 'mouth_pop', dict(note='toothy mouth opens on his cheek')), (27, 'tongue_lick', dict(note='tongue rolls out')),
              (36, 'freeze_crack', dict(note='he goes white-eyed')), (40, 'cloth_bags', dict(size=.4, note='arm wind-up whoosh, off screen right')), (45, 'slap_wobble', dict(size=1.0, note='SLAP: contact on 45, cheek wobbles to 52')),
              (45, 'star_burst', dict(note='burst and little stars')), (54, 'blink_pips', dict(note='Nobara and Megumi stare; blink 57-60')), (63, 'cackle', dict(note='mouth on the back of his hand, cackling 63-71'))]

    def setup(self):
        self.bgA = warm_bg(); self._c = {}
        x, y = grid()
        # shock: pale flat with drawn gloom lines
        b = np.empty((H, W, 3), f32); b[:] = rgb('#dfe6ee'); dk = np.clip(.02 + .42 * smooth(.45, .0, y / H), 0, .42); b = b * (1 - screentone(np.ones((H, W), f32), 15., .785, dk)[..., None]) + rgb('#8fa0bd') * screentone(np.ones((H, W), f32), 15., .785, dk)[..., None]
        c = Cel(); r = np.random.default_rng(5)
        for i in range(64):
            xx = r.uniform(-20, W + 20); l = r.uniform(120, 560) * (1. if abs(xx - 960) > 420 else .45); c.line([(xx, -10), (xx + r.normal(0, 2), l * .5), (xx + r.normal(0, 3), l)], r.uniform(1.5, 5), '#55607e', taper=(.02, .8), seed=i)
        self.bgB = comp.over(b, c.layer())
        self.bgS = []
        for k in range(2):                                                # slap: hot flat, focus lines (two drawings)
            b = np.empty((H, W, 3), f32); b[:] = rgb('#ffd23e'); d = np.hypot((x - 1180) / W, (y - 640) / H); tn = screentone(np.ones((H, W), f32), 22., .5, np.clip((d - .18) * .9, 0, .42)); b = b * (1 - tn[..., None]) + rgb('#ff8f2e') * tn[..., None]
            c = Cel(); focus_lines(c, 1190, 640, 480, 1700, 130, 10., '#17131a', 11 + k, ry=.8); self.bgS.append(comp.over(b, c.layer()))
        b = np.ones((H, W, 3), f32); c = Cel(); focus_lines(c, 1190, 640, 420, 1700, 210, 13., '#17131a', 77, ry=.8); self.bgHit = comp.over(b, c.layer())
        self.bgC = []
        for k in range(2):
            b = np.empty((H, W, 3), f32); b[:] = rgb('#f0b7c4'); tn = screentone(np.ones((H, W), f32), 20., .35, np.clip(.05 + .4 * smooth(.2, 1.05, x / W * .7 + y / H * .5), 0, .42)); b = b * (1 - tn[..., None]) + rgb('#c45a86') * tn[..., None]
            c = Cel(); focus_lines(c, 1180, 620, 560, 1700, 64, 7., '#b04a7c', 21 + k, ry=.8); self.bgC.append(comp.over(b, c.layer()))

        q8 = lambda a: np.clip(a * 255 + .5, 0, 255).astype(np.uint8)          # flat comic grounds live as 8 bit
        self.bgA = q8(self.bgA); self.bgB = q8(self.bgB); self.bgHit = q8(self.bgHit); self.bgS = [q8(a) for a in self.bgS]; self.bgC = [q8(a) for a in self.bgC]

    
    def F8(s, a): return a.astype(f32) / 255.

    # ---- pieces
    POSE = (905, 575, 372, -.42, .02, -.05)

    def mouth_map(self, H_, roll=-.30, cx=.455, cy=-.36, hw=.175):
        ca, sa = math.cos(roll), math.sin(roll)
        return lambda L: H_.P(np.c_[cx + (np.asarray(L, float)[:, 0] * ca - np.asarray(L, float)[:, 1] * sa) * hw, cy + (np.asarray(L, float)[:, 0] * sa + np.asarray(L, float)[:, 1] * ca) * hw], .03)

    def hand(self, key, pose, R, pos, s, dist, light, pal, w, clip=True, cuffy=.34):
        k = ('hand',) + key
        if k not in self._c:
            h = HD.Hand(pose, 'L', R=HD.euler(*R), pos=(0, 0, 0), pivot=(0, .62, 0), cuff=dict(kind='uniform', y=cuffy)); g = HD.render([h], HD.Cam(pos[0], pos[1], s, dist), light=light, res=460, clip=(-100, -100, W + 100, H + 100))
            c = Cel(); HD.ink(c, g, pal, w=w, seed=4); self._c[k] = (c.layer().astype(np.float16), dict(g.pts[0]))
        return self._c[k][0].astype(f32), self._c[k][1]

    PAL = dict(HD.PAL['day'], skin='#f6cfb0', shade='#dc9c80', cuff='#20213a', cuff_shade='#0f0e1a', cuff_light='#424a80')

    def frame(self, f):
        if 54 <= f < 63:
            from chars.chibi import stare_insert
            return stare_insert(f)
        if f < 36: return self.calm(f)
        if f < 54: return self.slap(f)
        return self.cackle(f)

    def calm(self, f):
        d = f // 2 * 2
        px = f * .9
        im = comp.cam2d(self.F8(self.bgA), 1100 - 40 + px, 620 - f * .15, 1.0 + f * .0006)
        X0, Y0, S0, yw, pt, rl = self.POSE; br = math.sin(d / 24 * 2 * math.pi * .55)
        # acting: smile, eyes slide to the cheek, brows twitch
        e = Yuji.expr('smile')
        if d in (4, 6): e = Yuji.lids(e, 1.0 if d == 4 else .50)
        nod = 0.
        if 12 <= f < 18:                                                  # something moved in there: the eyes go first, then the near brow and the smile
            e = Yuji.expr(Yuji.blend('smile', 'neutral', (0., .12, .30, .34, .42, .50)[f - 12]), BR=[(.134, .160), (.262, .210 + .004 * (f - 12)), (.400, .236 + .006 * (f - 12)), (.534, .200 + .005 * (f - 12))]); nod = (.0, .004, .009, .006, .010, .012)[f - 12]
        if 18 <= f < 27:
            k = f - 18; e = Yuji.expr('smile', gx=0., B=[(.134, .160), (.262, .208), (.400, .232), (.534, .196)])
            nod = [-.018, -.018, .010, .010, 0, 0, 0, 0, 0][k]
        if f >= 22: e = Yuji.expr(e, hw=.18, tU=.02)
        if f >= 27:
            e = Yuji.expr(Yuji.blend('smile', 'neutral', .62), BR=[(.134, .168), (.262, .222), (.400, .250), (.534, .216)], BL=[(.134, .136), (.262, .170), (.400, .190), (.534, .160)], mU=[.012, .0, .004, -.004, .020], mL=[.012, .0, .0, -.010, .020], tU=.0, sweat=.0 if f < 31 else .8)
        hd = YHead().pose(X0 + br * 1.2, Y0 + br * 2.4, S0, yw + nod, pt + br * .006, rl + nod * .5)
        # gaze: both eyes slide toward the near cheek from 23
        gz = 0. if f < 12 else .020 if f < 13 else .044 if f < 18 else .030
        e = dict(e, gx=gz, gy=-.016 if 12 <= f < 18 else e['gy'])
        L = Yuji.light('sunset', side=-1, amt=-.10)
        # mouth opening (overshoot, settle, chatter)
        o = 0.; tg = 0.
        if f >= 18:
            k = f - 18; o = [.45, 1.28, 1.28, .78, .78, 1.04, 1.04, .94, .94][k] if k < 9 else (.92 + .13 * ((d // 3) % 2))
        if f >= 27:
            k = f - 27; tg = [.55, 1.24, 1.24, .90, .90, 1.05, 1.05, 1.0, 1.0][min(8, k)]
        if 18 <= f < 27 or f >= 27:                                      # the skin jumps when it opens
            c0 = hd.P([(.455, -.36)])[0]; a = 0.
            if 18 <= f < 26: a = 14 * math.exp(-(f - 18) / 3.2) * math.cos((f - 18) * 1.5)
            elif 27 <= f < 34: a = 8 * math.exp(-(f - 27) / 2.6) * math.cos((f - 27) * 1.6)
            if abs(a) > .4: hd.warp = G.jelly(c0[0], c0[1], 190, a, 0., .0, 1., -.2)
        tw = (0., 0.) if not 9 <= f < 18 else ((6., .30), (8., .42), (-4., .12), (0., 0.), (11., .66), (14., .82), (-5., .34), (16., .92), (22., 1.02))[f - 9]         # bulge px, seam length
        if tw[0]: c0 = hd.P([(.455, -.36)])[0]; hd.warp = G.jelly(c0[0], c0[1], 120, tw[0], 0., .012, 1., -.5)
        bd = self.gaze_bust(hd, e, L, d, gz, 5)
        if o > 0:
            c = Cel(); G.skin_mouth(c, self.mouth_map(YHead.of(hd)), o, tg, .5 * math.sin(d * .9) if f >= 32 else .25, hd.s * .175, 3, (L['skin'][1], L['skin'][2]), '#22101a', w=5.2)
            bd = bd * (1 - c.layer()[..., 3:4]) + c.layer()
        if tw[1] > .05: T_ = self.twitch(hd, f, tw, L); bd = bd * (1 - T_[..., 3:4]) + T_
        im = comp.over(im, bd)
        inner = np.clip(comp.blur(1 - bd[..., 3], 18) * bd[..., 3], 0, 1)
        im = comp.screen(im, inner[..., None] * rgb('#ffd79a') * .28 * comp.radial(100, 200, 1500, power=1.)[..., None])
        return comp.screen(im, comp.radial(120, 60, 1250, power=2.4)[..., None] * rgb('#ffdf9e') * .26)

    def twitch(self, hd, f, tw, L):
        """the closed mouth moving under the skin: a lump with its own little shadow, a seam that comes and goes, ticks"""
        M = self.mouth_map(YHead.of(hd)); c = Cel(); b, s = tw; a = np.linspace(-.20, -2.94, 26); R = .52 + .34 * s; w = np.sin(np.linspace(0, math.pi, 26)) ** .8
        o = np.c_[np.cos(a) * R * 1.15, np.sin(a) * R * .72]; i = np.c_[np.cos(a) * R * 1.12, np.sin(a) * R * .72 + w * (.20 + .12 * s)]; c.fill(M(np.vstack([o, i[::-1]])), L['skin'][2], .50); c.line(M(o[3:-3]), 3.2, '#6d3a30', taper=(.4, .4), seed=3, alpha=.85)
        a = np.linspace(.5, 2.64, 18); c.line(M(np.c_[np.cos(a) * R * 1.10, .06 + np.sin(a) * R * .66]), 2.8, '#6d3a30', taper=(.5, .5), seed=4, alpha=.7); c.fill(M(np.c_[np.r_[np.cos(a) * R * .86, np.cos(a[::-1]) * R * .70], .05 + np.r_[np.sin(a) * R * .52, np.sin(a[::-1]) * R * .38]]), mixc(L['skin'][0], '#fff3dc', .6), .8)
        x = np.linspace(-s, s, 30); c.line(M(np.c_[x, .20 * np.abs(x) ** 2.2 + .035 * np.sin(x * 9 + f * 2.1)]), 5.2 * min(1., .4 + s), '#22101a', taper=(.35, .35), seed=5)
        return c.layer()

    def gaze_bust(self, hd, e, L, f, gz, seed, **k):
        return Yuji.bust(hd, e, L, f=f, wind=(.42, .08), seed=seed, byaw=-.26, broll=.02, breath=1., **k)

    def slap(self, f):
        X0, Y0, S0, yw, pt, rl = self.POSE
        im = self.F8(self.bgB if f < 45 else self.bgHit if f == 45 else self.bgS[(f // 2) % 2])
        L = Yuji.light('day', side=-1)
        e = Yuji.expr('panic', sweat=0.)
        jx = jy = jr = 0.; warp = None; tg = 1.; o = 1.
        if f < 45:                                                        # 36: the jolt, then frozen trembling (on ones)
            k = f - 36; up = [-46, -30, -8, 6, 2, 0, 0, 0, 0][k]; jy = up; jx = [-10, -6, 2, 0, 0, 0, 0, 0, 0][k]; jr = [-.06, -.04, .01, 0, 0, 0, 0, .015, .03][k]
            if k >= 3: jx += 2.2 * (-1) ** f
            sc = [1.05, 1.03, .99, 1.0, 1, 1, 1, 1, 1][k]
        else:
            k = f - 45; jx = [-62, -92, -84, -70, -64, -60, -58, -57, -57][k]; jy = [10, 22, 18, 12, 9, 8, 8, 8, 8][k]; jr = [-.14, -.22, -.19, -.15, -.135, -.13, -.13, -.13, -.13][k]; sc = 1.0
            e = Yuji.expr('panic', mx=-.07, mrot=.16, hw=.20, sweat=0., jx=-.05)
        hd = YHead().pose(X0 + jx, Y0 + jy, S0 * sc, yw + (0 if f < 45 else -.05), pt, rl + jr)
        if f >= 45:
            c0 = hd.P([(.50, -.36)])[0]; k = f - 45; dep = [78, 70, 62, 60, 58, 57, 56, 56, 56][k]; a = 40 * math.exp(-k / 2.6)
            hd.warp = G.chain(G.press(c0[0] + 46, c0[1], 175, dep, -1., .04), G.jelly(c0[0] - 20, c0[1] + 30, 250, a, k * 2.05, .022, -1., .35))
            if k < 2: e = Yuji.expr(e, white=0., U=[-.04, -.050, -.052, -.050, -.04], Lo=[-.05, -.058, -.060, -.058, -.05])
        bd = Yuji.bust(hd, e, L, f=f, wind=(.1 if f < 45 else -.9, .1), seed=5 + (f >= 45), byaw=-.26, broll=.02 + jr * .4, raise_=.10)
        im = comp.over(im, bd) if f < 45 else im
        if f < 45:
            c = Cel(); q = YHead.of(hd); c.clip(q.face())
            for j in range(13):
                x0 = -.50 + j * .082; l = (.20, .34, .26, .40)[j % 4]; c.line(q.Pc([(x0, .42), (x0, .42 - l * .6), (x0 + .004, .42 - l)]), 3.6, '#7f8fc0', taper=(.05, .8), seed=j)
            c.unclip(); dp = spl(A([(0, -58), (25, 0), (0, 30), (-25, 0)]), True, corners=(0,)) + q.P([(.50, .30)])[0] + A([0, 3 * (f - 36)]); c.fill(dp, '#bfe4ff'); Y.outline(c, dp, 4, '#17131a', 3)
            G.skin_mouth(c, self.mouth_map(YHead.of(hd)), 1.05 + .1 * (f % 2), 1.05, .6 * math.sin(f * 1.1), hd.s * .175, 3, (L['skin'][0], L['skin'][1]), w=5.2); im = comp.over(im, c.layer())
            if f >= 42:                                                   # the arm comes in: two smear drawings on ones
                k = f - 42; c = Cel(); lay = None
                if k == 0: lay, _ = self.hand(('wind',), 'open', (-14, 0, -16), (1762, 262), 232, 6, (-.5, .75, .6), self.PAL, 6.); self.arm(c, (1370, 1120), (1566, 948), None); im = comp.over(im, c.layer()); im = comp.over(im, lay)
                else:
                    p0, p1 = (A([1830., 260.]), A([1500., 430.])) if k == 1 else (A([1640., 330.]), A([1290., 600.]))
                    self.smear(c, p0, p1, k, (1390, 1130), ((1706, 968), (1664, 1000))[k - 1]); im = comp.over(im, c.layer())
            return im
        k = f - 45; ct = hd.P([(.56, -.30)])[0] + A([40, 0])
        c = Cel()
        if k < 8:
            rr = [1.15, 1.0, .92, .90, .86, .80, .70, .55][k]; G.star(c, ct[0] + 20, ct[1], 300 * rr, 860 * rr, 19, 31 + k // 2, '#ffffff' if k else '#ffd23e', w=8, rot=k * .02, asp=.9); G.star(c, ct[0] + 20, ct[1], 240 * rr, 560 * rr, 15, 41 + k // 2, '#fff2a0', w=0, asp=.9)
        im = comp.over(im, c.layer()); im = comp.over(im, bd)
        sway = [0, 6, 3, 0, -1, 0, 0, 0, 0][k]
        lay, g = self.hand(('hit', min(k, 3)), 'slap', (-26, 6, -10 + [4, 0, 1, 2][min(k, 3)]), (ct[0] + 26 + [10, -8, -2, 0][min(k, 3)], ct[1] + 6), 262, 6, (-.5, .75, .6), self.PAL, 6.4)
        im = comp.over(im, lay)
        c = Cel(); r = np.random.default_rng(8)
        for i in range(11):                                               # little stars and drops fly off in a ring and fall
            th = -3.5 + i * .355 + r.normal(0, .06); R0 = r.uniform(.8, 1.15) * (250 + 480 * (1 - math.exp(-k * .42))); x, y = ct[0] - 40 + math.cos(th) * R0 * 1.15, ct[1] - 60 + math.sin(th) * R0 * .92 + 2.2 * k * k
            if k >= 1:
                if i % 3: G.little_star(c, x, y, r.uniform(26, 50), k * .35 + i, ['#ffd84a', '#ff7a5a', '#ffffff'][i % 3], w=4.5)
                else: ca, sa = math.cos(th + 1.57), math.sin(th + 1.57); P = spl(A([(0, -34), (15, 0), (0, 18), (-15, 0)]), True, corners=(0,)); P = np.c_[P[:, 0] * ca - P[:, 1] * sa + x, P[:, 0] * sa + P[:, 1] * ca + y]; c.fill(P, '#bfe4ff'); Y.outline(c, P, 4., '#17131a', i)
        im = comp.over(im, c.layer())
        s = shake(f, 16, 3, 2.4, 45); return comp.cam2d(im, 960 + s[0], 540 + s[1], 1.03)

    def arm(self, c, S, E, W_, r=(170., 124., 112., 78.)):
        """his left arm: shoulder S -> elbow E -> wrist W_ (None: the upper arm and elbow only).  one outline round the union, hard shadow, elbow folds, cuff"""
        S, E = A(S, float), A(E, float); ink = '#17131a'; u1 = (E - S) / np.hypot(*(E - S)); n1 = A([-u1[1], u1[0]]); shapes = []
        up = spl([S - u1 * 120 + n1 * r[0], S + n1 * r[0], (S + E) / 2 + n1 * (r[0] + r[1]) * .52, E + n1 * r[1], E + u1 * r[1] * .92, E - n1 * r[1], (S + E) / 2 - n1 * (r[0] + r[1]) * .50, S - n1 * r[0], S - u1 * 120 - n1 * r[0]], True, n=8); shapes.append(up)
        if W_ is not None:
            W_ = A(W_, float); u2 = (W_ - E) / np.hypot(*(W_ - E)); n2 = A([-u2[1], u2[0]]); D = W_ - E
            fo = np.vstack([spl([E - u2 * 30 + n2 * r[1], E + D * .28 + n2 * r[2] * 1.08, E + D * .65 + n2 * r[2] * .88, W_ + n2 * r[3]], n=8), spl([W_ - n2 * r[3], E + D * .65 - n2 * r[2] * .84, E + D * .28 - n2 * r[2] * 1.02, E - u2 * 30 - n2 * r[1]], n=8)]); shapes.append(fo)
        for P in shapes: c.line(np.vstack([P, P[:2]]), 13., ink, taper=(.01, .01), press=.1, seed=2, tremor=.2)
        for P in shapes: c.fill(P, '#20213a')
        c.clip(up); c.fill(up + A([34, 74]), '#0f0e1a'); c.unclip()
        for q, o in ((-.30, 18), (.12, 40), (.52, 10)): c.line(spl([E - u1 * 60 + n1 * r[1] * q, E - u1 * 10 + n1 * r[1] * (q * .8) + A([o * .3, -o]), E + u1 * 46 + n1 * r[1] * q * .5]), 5.5, ink, taper=(.2, .6), seed=int(30 + o))
        c.fill(spl([S + n1 * -r[0] * .78, (S + E) / 2 - n1 * r[1] * .86, E - n1 * r[1] * .66, (S + E) / 2 - n1 * r[1] * .52], True, corners=(0, 2), n=6), '#424a80')
        if W_ is not None:
            c.clip(fo); sh = fo + n2 * (r[2] * .95 if n2[0] > 0 else -r[2] * .95); c.fill(sh, '#0f0e1a'); s_ = -1 if n2[0] > 0 else 1
            c.fill(spl([E + D * .16 + n2 * s_ * r[2] * .86, E + D * .50 + n2 * s_ * r[2] * .74, E + D * .86 + n2 * s_ * r[3] * .70, E + D * .48 + n2 * s_ * r[2] * .42], True, corners=(0, 2), n=6), '#424a80'); c.unclip()
            for t_, l, q in ((.06, .9, 1), (.14, .62, 1), (.22, .40, -1), (.30, .5, 1), (.74, .5, -1), (.82, .62, 1)): b_ = E + D * t_; c.line(spl([b_ + n2 * q * r[2] * .98, b_ + n2 * q * r[2] * (.98 - l * .5) + D * .035, b_ + n2 * q * r[2] * (.98 - l) + D * .03]), 5.5, ink, taper=(.1, .65), seed=int(t_ * 90))
            # cuff: a turned edge and the dark opening
            q = np.linspace(0, 2 * math.pi, 34)[:-1]; el = W_ + np.cos(q)[:, None] * n2 * r[3] * 1.04 + np.sin(q)[:, None] * u2 * 22; band = np.vstack([W_ + n2 * r[3] * 1.06 + u2 * 0, W_ - n2 * r[3] * 1.06, W_ - n2 * r[3] * 1.08 - u2 * 44, W_ + n2 * r[3] * 1.08 - u2 * 44]); c.fill(band, '#20213a'); Y.outline(c, band, 5., ink, 8); c.fill(el, '#0a0910'); Y.outline(c, el, 5.5, ink, 9)

    def smear(self, c, p0, p1, k, S, E):
        """hand smear: a stretched, multi-fingered drawing between two positions"""
        d = p1 - p0; L = np.hypot(*d); u = d / L; n = A([-u[1], u[0]]); sk, sh, ink = '#f6cfb0', '#dc9c80', '#17131a'
        arc = lambda t, o: p0 + d * t + n * (o + 90 * math.sin(math.pi * t) * .6)
        for j, (o, w0, t0, t1) in enumerate(((-120, 46, -.55, .96), (-58, 56, -.45, 1.06), (6, 60, -.42, 1.10), (70, 58, -.48, 1.04), (126, 40, -.2, .78))):
            t = np.linspace(t0, t1, 26); mid = A([arc(v, o * (.75 + .35 * v)) for v in t]); ww = w0 * np.clip((t - t0) / (t1 - t0) * 1.25, 0, 1) ** .7 * np.clip((t1 - t) * 7, 0, 1) ** .5
            T = np.gradient(mid, axis=0); T /= np.hypot(*T.T)[:, None]; N = np.c_[-T[:, 1], T[:, 0]]; P = np.vstack([mid + N * ww[:, None], (mid - N * ww[:, None])[::-1]]); c.fill(P, sk)
            c.fill(np.vstack([mid + N * (ww * .15)[:, None], (mid + N * ww[:, None])[::-1]]), sh); c.line((mid + N * ww[:, None])[4:], 5, ink, taper=(.6, .05), seed=j); c.line((mid - N * ww[:, None])[4:], 5, ink, taper=(.6, .05), seed=j + 9)
        for j in range(16):
            o = -190 + j * 25; t = np.linspace(-1.2, .1 + .25 * math.sin(j * 2.3), 8); c.line(A([arc(v, o) for v in t]), 3 + 3 * (j % 3 == 0), ink if j % 2 else '#ffffff', taper=(.7, .3), seed=j)
        # the arm that throws it: shoulder at the lower right, elbow out, forearm, cuff, and a wrist joining it to the stretched hand
        W_ = arc(.02, 0) + A([96., 118.]); v = W_ - A(E, float); v = v / np.hypot(*v); m = A([-v[1], v[0]]); self.arm(c, S, E, W_ - v * 34)
        b = spl([W_ - v * 40 - m * 54, W_ + v * 40 - m * 60, arc(.22, 118), arc(.30, 20), arc(.22, -96), W_ + v * 70 + m * 58, W_ - v * 40 + m * 56], False, n=8); c.fill(b, sk); c.fill(np.vstack([spl([W_ - v * 40 + m * 10, W_ + v * 40 + m * 4, arc(.22, 60), arc(.22, 118)], n=8), spl([arc(.22, 118), W_ + v * 40 - m * 60, W_ - v * 40 - m * 54], n=8)]), sh)
        c.line(spl([W_ - v * 40 - m * 54, W_ + v * 40 - m * 60, arc(.20, 118)], n=8), 6, ink, taper=(.05, .4), seed=41); c.line(spl([W_ - v * 40 + m * 56, W_ + v * 70 + m * 58, arc(.20, -96)], n=8), 6, ink, taper=(.05, .4), seed=42)
        for j in range(9): o = (j - 4) * 30.; q0 = A(E, float) * .45 + W_ * .55 + m * o + A([36., -10.]); c.line(spl([q0 + A([70., -60.]) + m * o * .2, q0 + A([24., -20.]), q0], n=6), 4, '#ffffff' if j % 2 else ink, taper=(.6, .3), seed=j)

    def cackle(self, f):
        k = f - 63; d = f // 2 * 2; X0, Y0, S0, yw, pt, rl = self.POSE
        im = self.F8(self.bgC[(f // 3) % 2]); L = Yuji.light('day', side=-1)
        hd = YHead().pose(X0 - 57, Y0 + 8, S0, yw - .05, pt, rl - .13)
        c0 = hd.P([(.50, -.36)])[0]; hd.warp = G.press(c0[0] + 46, c0[1], 175, 56, -1., .04)
        e = Yuji.expr('panic', mx=-.07, mrot=.16, hw=.17, jx=-.05, mU=[.02, .03, .03, .03, .01], mL=[-.02, -.05, -.06, -.05, -.02], tU=.03, tL=.03, sweat=0., white=1., BL=[(.130, .25), (.262, .25), (.405, .23), (.575, .17)], BR=[(.130, .25), (.262, .25), (.405, .23), (.575, .17)])
        bd = Yuji.bust(hd, e, L, f=d, wind=(.1, .05), seed=7, byaw=-.26, broll=-.03, raise_=.10); im = comp.over(im, bd)
        ct = hd.P([(.56, -.30)])[0] + A([40, 0]); hop = [0, 0, -26, -26, -10, -10, -16, -16, -12][min(k, 8)] * (k > 1)
        lay, g = self.hand(('hit', 3), 'slap', (-26, 6, -8), (ct[0] + 26, ct[1] + 6), 262, 6, (-.5, .75, .6), self.PAL, 6.4)
        im = comp.over(im, lay)
        pt_ = g; wr, mk = A(pt_['wrist'], float), A(pt_[('middle', 'mcp')], float); ik, lk = A(pt_[('index', 'mcp')], float), A(pt_[('little', 'mcp')], float)
        ax = (ik - lk); ax = ax / np.hypot(*ax); ay = (mk - wr); ay = ay / np.hypot(*ay); cen = wr * .40 + mk * .60
        o = [.15, 1.3, 1.3, .8, .8, 1.15, 1.15, .7, 1.1][min(k, 8)]; sc = 92 * [.55, 1.22, 1.15, .96, .96, 1.04, 1.04, .98, 1.02][min(k, 8)]
        jig = 3.0 * (-1) ** (f // 1) * (k > 0)
        M = lambda Q: cen + np.asarray(Q, float)[:, :1] * ax * sc * 1.05 + np.asarray(Q, float)[:, 1:] * ay * sc + A([jig, 0])
        c = Cel(); G.skin_mouth(c, M, o, 0., 0., sc, 13, ('#f6cfb0', '#dc9c80'), w=5.2, grin=.34); im = comp.over(im, c.layer())
        c = Cel(); r = np.random.default_rng(2)
        for i in range(6):                                                # cackle ticks beside the hand
            th = -.62 + i * .30; R0 = 235 + 14 * (f % 2); p = cen + A([math.cos(th), math.sin(th)]) * R0; q = cen + A([math.cos(th), math.sin(th)]) * (R0 + (70, 46)[i % 2]); c.line([p, (p + q) / 2, q], 9, '#17131a', taper=(.1, .6), seed=i)
        P = spl(A([(520, 250), (548, 318), (540, 366), (500, 366), (492, 318)]), True, corners=(0,)); c.fill(P + A([0, 3 * k]), '#bfe4ff'); Y.outline(c, P + A([0, 3 * k]), 4, '#17131a', 3)
        im = comp.over(im, c.layer())
        return comp.cam2d(im, 960, 540, 1.0 + .012 * (k >= 1))
