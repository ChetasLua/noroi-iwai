"""e5 RAIN (144 f).  0-71 wide crater, Yuji on his knees, tiny, slow push; 72 hard cut: close-up, the mouth opens, he screams (three drawings on twos); 90 the rain stops in the air, 104 rises; 126 white."""
import numpy as np, math, cv2, os
from jj import comp, W, H, world, fx
from jj.ink import Cel, spl
from jj.kit import shake
from jj.paint import rgb, fbm, smooth, ramp, grid, f32
from chars import yuji as Y
from chars.yuji import Yuji, YHead, mixc, rad
A = np.array


def _v(f):
    """speed of rain time"""
    return float(np.interp(f, [0, 84, 91, 104, 112, 126, 144], [1., 1., 0., 0., -.35, -1.25, -1.6]))


TAU = np.cumsum([_v(f) for f in range(145)]) / 24.


class Shot:
    id = 'e5'; n = 144
    finish = dict(bloom=.34, vig=.22)
    EVENTS = [(0, 'rain', dict(note='rain in three distances over the crater; 0-71 wide')), (0, 'kneel_rubble', dict(size=.3, note='far, already kneeling')), (72, 'rain', dict(size=.8, note='hard cut close: rain on skin and cloth')), (72, 'tear', dict(size=.6)),
              (76, 'breath_fear', dict(note='he fills his lungs')), (80, 'scream_in', dict(size=1., note='the mouth opens fully: no voice, only the note; camera trembles 80-90')), (84, 'rain_stop', dict(note='drops slow 84-91, hang from 91')),
              (90, 'silence', dict(note='everything stops')), (104, 'rain_reverse', dict(note='drops and tears begin to rise; fast by 120')), (122, 'light_rise', dict(note='light eats the picture')), (126, 'white', dict(note='white, nothing, to 144')), (128, 'ears_ring', dict(size=.35))]

    EYE, TGT, FOC = (-3.4, -9.6, 1.75), (.35, 9.5, -.15), 1290.

    def setup(self):
        self._c = {}; kw, kb, kp, kf, km = 'cache/e5_v14_wide.npy', 'cache/e5_v14_bg.npy', 'cache/e5_v14_meta.npy', 'cache/e5_v15_fig.npy', 'cache/e5_v15_meta.npy'
        if not all(os.path.exists(k) for k in (kw, kb, kp)):
            P = world.plate('ruin', mode='rain', view='crater', eye=self.EYE, target=self.TGT, focal=self.FOC, w=2400, h=1350, figure=(0., 4.))
            b = cv2.resize(P.img[150:1100, 300:1989], (W, H), interpolation=cv2.INTER_AREA).astype(f32); b = cv2.GaussianBlur(b, (0, 0), 26) * .58; yy = grid()[1]
            b = b * (1.0 - .35 * smooth(300, 1080, yy))[..., None] * rgb('#c3cfe8') + comp.radial(400, -20, 1150, power=2.)[..., None] * rgb('#a9c2f0') * .55; np.save(kb, b.astype(np.float16)); del b
            wz = P.extra['water_z']; cam = P.cam; a_ = np.linspace(0, 2 * np.pi, 60, endpoint=False); c0 = cam.project(A([[0., 4., wz]]))[0][0]
            hull = cam.project(np.c_[6.9 * np.cos(a_), 4. + 6.9 * np.sin(a_), np.full(60, wz)])[0]          # the pale disc of water he kneels in
            sc = 1.5; hw, hh = 3600, 2025; im = cv2.resize(P.img.astype(f32), (hw, hh), interpolation=cv2.INTER_CUBIC); im = im + (im - cv2.GaussianBlur(im, (0, 0), 1.6)) * .55; sk = cv2.resize(P.mask['sky'].astype(f32), (hw, hh), interpolation=cv2.INTER_LINEAR); fg = cv2.resize(smooth(24, 85, np.minimum(P.z, 200.)).astype(f32), (hw, hh), interpolation=cv2.INTER_LINEAR) * (1 - sk); del P
            im = im * (1 - .80 * fg[..., None]) + rgb('#9db3cf') * (.80 * fg)[..., None] * .95             # rain haze: the far blocks go pale, the near ones stay dark
            # ---- the pool: one pale wet disc.  a hand-cut edge, the real reflection of everything behind it, sky light on it
            r = np.random.default_rng(3); H_ = spl(hull, True, n=8) * sc; cen = H_.mean(0); th = np.arctan2(H_[:, 1] - cen[1], H_[:, 0] - cen[0]); o_ = np.argsort(th); nz = np.zeros(len(th)); nz[o_] = np.convolve(np.tile(r.normal(0, 1, len(th)), 3), np.hanning(9) / 4.5, 'same')[len(th):2 * len(th)]; nz2 = np.zeros(len(th)); nz2[o_] = np.convolve(np.tile(r.normal(0, 1, len(th)), 3), np.hanning(61) / 30., 'same')[len(th):2 * len(th)]; wob = 1 + .012 * nz + .075 * nz2
            H_ = cen + (H_ - cen) * wob[:, None]; mk = np.zeros((hh, hw), np.uint8); cv2.fillPoly(mk, [np.round(H_ * 16).astype(np.int32)], 255, cv2.LINE_AA, 4); m = cv2.GaussianBlur(mk.astype(f32) / 255., (0, 0), 1.2)
            x0, x1, y0, y1 = int(H_[:, 0].min()) - 4, int(H_[:, 0].max()) + 4, max(0, int(H_[:, 1].min()) - 4), min(hh, int(H_[:, 1].max()) + 4)
            far = y0 + 6; src = np.clip(2 * far - np.arange(y0, y1), 0, hh - 1); rf = im[src][:, x0:x1].copy(); rf = cv2.GaussianBlur(rf, (0, 0), sigmaX=2.5, sigmaY=9.)
            rip = fbm(y1 - y0, x1 - x0, (9, 4), (1, .6), 5, aspect=9.); yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0].astype(f32); dep = yy / (y1 - y0); rf = cv2.remap(rf, xx + rip * (2 + 9 * dep), yy + rip * (2 + 5 * dep), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            sky = ramp(np.clip(dep + .05 * rip, 0, 1), [(0, '#dfe8ee'), (.35, '#b7c6d6'), (.75, '#8196b0'), (1, '#5a6f8e')]); bw = np.exp(-((xx - (c0[0] * sc - x0) + 60) / (.30 * (x1 - x0))) ** 2)[..., None]; pool = rf * (1.0 - .55 * bw) * .72 + sky * (.14 + .58 * bw); pool = pool + (np.clip(rip - 1.0, 0, 1.5) * .10 * (1 - dep))[..., None]
            mm = m[y0:y1, x0:x1][..., None]; wet = cv2.GaussianBlur(m, (0, 0), 46); im = im * (1 - .50 * np.clip(wet - m, 0, 1)[..., None]); im[y0:y1, x0:x1] = im[y0:y1, x0:x1] * (1 - mm) + pool * mm; del rf, pool, sky, rip, wet
            edge = np.clip(m - cv2.warpAffine(m, np.float32([[1, 0, 0], [0, 1, -3]]), (hw, hh)), 0, 1); im = im + edge[..., None] * .35
            # ---- values: dark ring of ruins, light only in the disc and where the cloud is broken
            c1 = A([c0[0], c0[1]]) * sc; x, y = grid(hh, hw); d = np.hypot((x - cen[0]) / (x1 - x0), (y - cen[1]) / ((y1 - y0) * 1.05)); ring = smooth(.50, .80, d) * .92 + .08 * smooth(.8, 1.5, d)
            ring = ring * (1 - sk) * (1 - .82 * fg); im = im * (1.0 - .80 * ring)[..., None] * (1 - ring[..., None] * (1 - rgb('#8094c0'))); brk = comp.radial(cen[0] - 210, -60, 900, 620, power=1.5, h=hh, w=hw); im = im + (brk * (.08 + .90 * sk + .40 * fg))[..., None] * rgb('#c6d6e8') * .95
            # stones along the water's edge, half sunk, so that the disc has a shore and a size
            cl = Cel(hw, hh, ss=1); q = np.random.default_rng(8); idx = np.sort(q.choice(len(H_), 150)); pal = dict(lit='#3f4b62', mid='#1d2433', dark='#10141f', ink='#05060b'); near_ = lambda y: .35 + .65 * np.clip((y - y0) / (y1 - y0), 0, 1) ** 1.2
            for i in sorted(idx, key=lambda i: H_[i, 1]):
                p = cen + (H_[i] - cen) * q.uniform(.985, 1.07); s_ = (8 + 46 * q.random() ** 2.4) * near_(p[1]) * 1.15; cl.fill(spl(p + A([(-s_ * 1.1, s_ * .25), (0, s_ * .95), (s_ * 1.1, s_ * .25), (0, -.1 * s_)]), True, n=4), '#0c1220', .55); fx.rock(cl, p[0], p[1], s_, tuple(q.uniform(-3, 3, 3)), int(i), pal, light=(-.45, -.85, .3), lw=2.0, crack=False, kind='rock' if q.random() < .6 else 'slab')
            rk = cl.rgba(); im = im * (1 - rk[..., 3:4]) + rk[..., :3]; del rk, cl
            q = 4; sh = comp.shaft(hh // q, hw // q, A([(cen[0] - 520, -30), (cen[0] - 30, -30), (x1 - 60, cen[1] + 20), (x0 + 60, cen[1] + 20)]) / q, 16, 5, .5); sh = cv2.resize(sh, (hw, hh), interpolation=cv2.INTER_LINEAR) * smooth(cen[1] + (y1 - y0) * .35, cen[1] - 120, y); im = im + sh[..., None] * rgb('#c3d6ea') * .16
            np.save(kw, im.astype(np.float16)); del im, sh, brk, d, ring, x, y, sk, fg
            np.save(kp, A([0, 0, 1, 0, 0, c0[0], c0[1], cen[0] / sc, cen[1] / sc, (x1 - x0) / sc, (y1 - y0) / sc, sc]))
        if not all(os.path.exists(k) for k in (kf, km)):
            # ---- the figure: one hand-drawn cel (two breaths) set on the water at his spot in the plate; he is 81 plate px tall
            from chars import yuji_kneel as K
            g = [K.draw(0.), K.draw(1.)]; rows_ = np.where((g[0][..., 3] > .4).any(1))[0]; hh = K.GROUND - rows_[0]; k = 4.2; q = 79. * k / hh; n = int(round(1000 * q)); arr = np.stack([cv2.resize(i, (n, n), interpolation=cv2.INTER_AREA) for i in g]).astype(np.float16)
            p = np.load(kp); rows_ = np.where((arr[..., 3] > .4).any((0, 2)))[0]; bot, top = int(rows_.max()), int(rows_.min()); p[:5] = [bot, top, k, p[5] * k - 500 * q, (p[6] + 10) * k - bot]; np.save(kf, arr); np.save(km, p)
        self.wide = np.load(kw, mmap_mode='r'); self.bg = np.array(np.load(kb, mmap_mode='r')); self.figs = np.load(kf); self.meta = np.load(km)

    # ---------------------------------------------------------------- wide
    def wide_frame(self, f):
        bot, top, k, X0, Y0, cx0, cy0, px, py, pw, ph, sc = [float(v) for v in self.meta]; fh = (bot - top) / k          # figure height in plate px
        t = comp.ease(f / 74.) * .5 + f / 74. * .5; z0, z1 = 90. / fh, 262. / fh; z0 = max(z0, .802); z = z0 * (z1 / z0) ** t
        sx, sy = 960 + (70 - 110 * t), 770 - 150 * t                                                          # where he sits in the frame
        fx0, fy0 = (X0 + self.figs.shape[2] * .5) / k, (Y0 + self.figs.shape[1] * .62) / k; cx, cy = fx0 - (sx - 960) / z, fy0 - (sy - 540) / z
        cx = float(np.clip(cx, 961 / z, 2400 - 961 / z)); cy = float(np.clip(cy, 541 / z, 1350 - 541 / z)); sx, sy = 960 + (fx0 - cx) * z, 540 + (fy0 - cy) * z
        hw, hh = 961 / z, 541 / z; a0, b0, a1, b1 = [int(v) for v in (max(0, (cx - hw) * sc - 3), max(0, (cy - hh) * sc - 3), min(3600, (cx + hw) * sc + 4), min(2025, (cy + hh) * sc + 4))]
        im = comp.cam2d(np.asarray(self.wide[b0:b1, a0:a1], f32), cx * sc - a0, cy * sc - b0, z / sc)
        P_ = lambda x, y: A([960 + (x - cx) * z, 540 + (y - cy) * z]); C_ = P_(px, py); rx, ry = pw * z * .5, ph * z * .5
        # he and his reflection
        g = self.figs[(f // 14) % 2].astype(f32); s = z / k; fo = P_(X0 / k, Y0 / k); M = np.float32([[s, 0, fo[0]], [0, s, fo[1]]]); L = cv2.warpAffine(g, M, (W, H), flags=cv2.INTER_AREA if s < .7 else cv2.INTER_LINEAR); L[..., :3] *= rgb('#c9d3e6'); keep = 1 - .97 * L[..., 3:4]
        yb = fo[1] + bot * s - 5 * s; rows = np.arange(H, dtype=f32); r = np.random.default_rng(f // 2); n = np.interp(rows, np.arange(0, H + 40, 5), r.normal(0, 1, (H + 40) // 5 + 1)[:len(np.arange(0, H + 40, 5))]); yy, xx = np.mgrid[0:H, 0:W].astype(f32)
        Rf = cv2.remap(L, (xx + (n * (1.2 + 2.4 * s))[:, None]).astype(f32), (2 * yb - yy).astype(f32), cv2.INTER_LINEAR, borderValue=0); Rf = cv2.GaussianBlur(Rf, (0, 0), sigmaX=.8 + 1.2 * s, sigmaY=1.5 + 3 * s) * np.clip(1 - (yy - yb) / (g.shape[0] * s * 1.1 + 1), 0, 1)[..., None] * (yy > yb)[..., None]
        im = comp.over(im, Rf, .62); im = im * (1 - .35 * comp.radial(sx, yb - 2, 60 * z + 26, 16 * z + 7, power=.9))[..., None]
        c = Cel()
        for j in range(5):                                                                                     # rings where the rain hits the water round him, perspective ellipses
            u = ((f + j * 7) % 22) / 22.; q = C_ + A([math.cos(j * 2.4 + (f + j * 7) // 22 * 1.9) * rx * .62, math.sin(j * 1.7 + (f + j * 7) // 22 * 2.3) * ry * .6]); a = np.linspace(.2, 5.6, 40); R_ = (8 + 46 * u) * z * .8
            c.line(np.c_[q[0] + np.cos(a) * R_, q[1] + np.sin(a) * R_ * .30], max(1., 2.6 * (1 - u)), '#f2f8ff', taper=(.3, .3), seed=j, alpha=.55 * (1 - u))
        im = comp.over(im, c.layer()); im = comp.over(im, L)
        # splash crowns on the disc and over the rubble
        def place(u, w): th = u * 6.283; rr = math.sqrt(w); return C_[0] + math.cos(th) * rx * 1.25 * rr, C_[1] + math.sin(th) * ry * 1.25 * rr, .25 + .75 * (.5 + .5 * math.sin(th) * rr)
        im = comp.over(im, fx.splashes(f, n=64, seed=4, size=(3 * z, 15 * z), place=place, col='#f4f9ff') * keep, .85)
        im = comp.over(im, fx.splashes(f, n=46, seed=9, size=(3, 13 * z), place=lambda u, w: (u * W, H * (.40 + .60 * w ** .8), .1 + .9 * w ** 1.5)) * keep, .40)
        # the water line round him: a broken bright edge and the two slow rings his breathing makes
        c = Cel(); xa, xb = fo[0] + 112 * s * g.shape[1] / 1000., fo[0] + 918 * s * g.shape[1] / 1000.; yb2 = fo[1] + bot * s; r = np.random.default_rng(f // 3); xs_ = np.linspace(xa - 3 * s, xb + 4 * s, 26); ys_ = yb2 + r.normal(0, .5, 26) + .6 * s
        for i in range(0, 25, 1):
            if r.random() < .72: c.line([(xs_[i], ys_[i]), (xs_[i + 1], ys_[i + 1])], max(1., 2.2 * s) * r.uniform(.6, 1.3), '#f1f7ff', taper=(.2, .2), seed=i, alpha=.8)
        for j in range(2): u = ((f + 14 * j) % 28) / 28.; th_ = np.linspace(.1, math.pi - .1, 50); c.line(np.c_[(xa + xb) / 2 + np.cos(th_) * ((xb - xa) * .52 + 60 * s * u), yb2 + np.sin(th_) * (4 + 26 * u) * s], max(.9, 1.8 * s * (1 - u)), '#e8f1ff', taper=(.3, .3), seed=70 + j, alpha=.55 * (1 - u))
        im = comp.over(im, c.layer())
        im = self.on_him(im, L, f, s)
        # cold light coming down, then rain in three depths (the nearest out of focus)
        im = im + comp.radial(sx - 60, sy - 420, 760, 720, power=2.0)[..., None] * rgb('#a8c0e0') * .10
        im = comp.over(im, fx.rain(f, wind=-.10, seed=2, n=(460, 0, 0), size=1.), .75); im = comp.over(im, fx.rain(f, wind=-.10, seed=5, n=(0, 190, 0), size=1.), .85)
        nr = fx.rain(f, wind=-.13, seed=8, n=(0, 0, 34), size=1.4); im = comp.over(im, cv2.GaussianBlur(nr, (0, 0), sigmaX=2.6, sigmaY=5.), .75)
        v = np.hypot((xx - sx) / 1150, (yy - sy) / 720); im = im * (1 - .42 * smooth(.55, 1.35, v))[..., None]
        return im * (.45 + .55 * smooth(-2, 5, f))

    def on_him(self, im, L, f, s):
        """rain breaking on his head, hood and shoulders"""
        a = L[..., 3]; ys, xs = np.where((a > .6) & (cv2.warpAffine(a, np.float32([[1, 0, 0], [0, 1, 4]]), (W, H)) < .4))
        if len(xs) < 8: return im
        c = Cel(); q0 = np.random.default_rng(12); ytop = ys.min(); hgt = ys.max() - ytop + 1; ok = np.where(ys < ytop + hgt * .55)[0]; xs, ys = xs[ok], ys[ok]
        for i in range(15):
            per = 9; k = (f + int(q0.integers(0, per))) % per; rr = np.random.default_rng(i * 13 + (f + i) // per); j = int(rr.integers(0, len(xs))); x, y = xs[j], ys[j]
            if k > 3: continue
            for m_ in range(3): th = -math.pi * (.18 + .30 * m_) + rr.normal(0, .2); l0, l1 = (2 + 5 * k) * s, ((2 + 5 * k) + (9, 11, 7, 3)[k]) * s; d = A([math.cos(th), math.sin(th)]); c.line([A([x, y]) + d * l0, A([x, y]) + d * (l0 + l1) / 2, A([x, y]) + d * l1], max(.9, (2.6 - .4 * k) * s), '#f4f9ff', taper=(.2, .7), seed=m_)
        for i in range(9):                                          # drips off hair and hood
            j = int((i * 37) % len(xs)); u = ((f * 1.4 + i * 5) % 16) / 16.; c.line([(xs[j], ys[j] + u * u * 70 * s), (xs[j], ys[j] + u * u * 70 * s + 9 * s)], max(.8, 1.6 * s), '#dcebff', taper=(.5, .1), seed=i, alpha=.8)
        return comp.over(im, c.layer(), .9)

    # ---------------------------------------------------------------- close-up
    # one way only: bowed -> he fills his lungs -> it cracks -> it opens -> full.  after that the same drawing, boiling on threes, then frozen with the rain
    KEYS = {'g': ('grief', (944, 372, 440, .10, -.12, .02)), 'g2': (('grief', 'scream_a', .35), (946, 360, 441, .11, -.04, .025)), 'in': (('grief', 'scream_a', .75), (950, 344, 446, .13, .04, .035)),
            'a': ('scream_a', (956, 330, 452, .14, .10, .05)), 'b': (('scream_a', 'scream_b', .7), (963, 320, 462, .16, .14, .065)), 'b2': (('scream_b', 'scream_c', .5), (969, 312, 472, .17, .17, .08)),
            'c': ('scream_c', (973, 306, 482, .18, .19, .09)), 'd': ('scream_d', (973, 306, 482, .18, .19, .09))}
    ORDER = [(72, 'g'), (76, 'g2'), (78, 'in'), (80, 'a'), (82, 'b'), (83, 'b2'), (84, 'c'), (87, 'd'), (90, 'c')]

    def key(self, name, lift=0., tears=None):
        k = (name, round(lift, 2), tears)
        if k not in self._c:
            e, p = self.KEYS[name]; X = Yuji.expr(e) if not isinstance(e, tuple) else Yuji.blend(*e)
            if name in ('g', 'g2', 'in'): X = Yuji.expr(X, tears=.75, well=1., hatch=.5, black=.35 if name != 'in' else .6, gy=-.020 if name == 'g' else -.004)
            if name == 'g': X = Yuji.expr(X, U=[-.030, .014, .022, .006, -.036], Lo=[-.056, -.078, -.082, -.072, -.046], iris=.056, pupil=.018, glint=.35, pouch=.5)
            if name == 'g2': X = Yuji.expr(X, pouch=.7)
            if tears is not None: X = Yuji.expr(X, tears=tears)
            L = Yuji.light('rain', side=-1, amt=.46, skin=('#c2b2ba', '#6a6488', '#383555'), rim=[dict(d=(10, 8), col='#d6e4ff', amt=1.)])
            h = YHead().pose(*p); bd = Yuji.bust(h, X, L, f=0, wind=(0., lift) if lift else 0., seed=dict(g=3, g2=4, a=5, b=6, b2=7).get(name, 8), wet=max(.35, 1. - .75 * lift), byaw=.06, raise_=dict(g=.04, g2=.10, a=.16).get(name, .18))
            # shadow side sinks, light side lifts
            x = grid()[0]; g = smooth(560, 1500, x); bd = bd.copy(); bd[..., :3] = bd[..., :3] * (1.08 - .34 * g)[..., None] * (1 - g[..., None] * (1 - rgb('#b9bce6')))
            if len(self._c) > 4: self._c.clear()
            self._c[k] = bd.astype(np.float16)
        return self._c[k].astype(f32)

    def close(self, f):
        v = _v(f); tau = TAU[f]
        name = [n for t, n in self.ORDER if f >= t][-1]; lift = 0.
        if f >= 108: lift = .20 * ((f - 108) // 3 + 1) if f < 120 else .95                        # hair begins to float, on threes
        tears = None
        if f >= 112: tears = [.85, .6, .35, .1][min(3, (f - 112) // 4)]
        z = float(np.interp(f, [72, 80, 84, 90, 130], [1.0, 1.010, 1.030, 1.040, 1.085]))                          # it only ever comes closer
        im = self.bg.astype(f32)
        if f < 90: im = comp.over(im, fx.rain(f, tau=tau, v=v, wind=.07, seed=11, n=(300, 130, 0), size=1.25), .8)
        else: im = comp.over(im, self.beads(f, 0)); im = comp.over(im, self.beads(f, 1))
        bd = self.key(name, lift, tears); im = comp.over(im, bd)
        inner = np.clip(comp.blur(1 - bd[..., 3], 26) * bd[..., 3], 0, 1)
        im = comp.screen(im, inner[..., None] * rgb('#9fb9ec') * .42 * comp.radial(300, 60, 1500, power=1.)[..., None])
        im = comp.screen(im, comp.radial(380, -40, 1150, power=2.2)[..., None] * rgb('#8fa9df') * .22)
        im = self.breaking(im, f, name) if f < 91 else im
        im = comp.over(im, fx.rain(f, tau=tau, v=v, wind=.07, seed=23, n=(0, 26, 12), size=1.5), .9) if f < 90 else comp.over(im, self.beads(f, 2))
        if f >= 103: im = self.rising(im, f)
        s = np.zeros(2)
        if 80 <= f < 91: s = shake(f, 9. * (1 - .06 * (f - 80)), 9)
        elif 72 <= f < 80: s = shake(f // 2, 1.2, 5)
        im = comp.cam2d(im, 960 + s[0] + 10, 540 + s[1] - 14 * (z - 1) * 10, z, rot=.004 * math.sin(f * 2.1) * (80 <= f < 91))
        # time stops: colour drains, then light comes
        if f >= 90:
            g = smooth(89, 97, f) * .55; l = comp.lum(im)[..., None]; im = im * (1 - g) + l * rgb('#d9e2f4') * 1.12 * g
            w = smooth(106, 127, f) ** 2.2; yy = grid()[1]; up = np.clip(1.25 - yy / H * .8, 0, 1.3)[..., None]
            im = im * (1 + .9 * w) + w * 1.05 * up * rgb('#f4f8ff') + smooth(116, 126.5, f) ** 2 * 1.5
        return im

    def breaking(self, im, f, name):
        """rain breaking on the planes of the face: short crowns on forehead, nose, cheek, collar"""
        c = Cel(); p = self.KEYS[name][1]; h = YHead().pose(*p); r = np.random.default_rng(50)
        for i in range(0, 16, 2):
            x, y = [(-.30, .40), (.12, .36), (.34, .33), (-.50, .22), (0, -.12), (.02, -.30), (-.48, -.22), (-.58, -.05), (.40, -.22), (-.38, -.40), (.30, .12), (-.12, .42), (-.60, -.38), (.50, .18), (-.20, .16), (.10, .02)][i]
            per = 11; k = (f * 1 + i * 7) % per
            if k < 4 and h.facing(x, y) > .15:                      # a drop bursting on skin: three ticks and two beads, gone in four frames
                q = h.P([(x, y)])[0]; rr = np.random.default_rng(i * 5 + (f + i * 7) // per)
                for j in range(3):
                    th = -math.pi * (.20 + .28 * j) + rr.normal(0, .22); a0, a1 = (4 + 8 * k), (4 + 8 * k) + (24, 26, 16, 7)[k] * rr.uniform(.6, 1.1); d = A([math.cos(th), math.sin(th) * .9]); c.line([q + d * a0, q + d * (a0 + a1) / 2, q + d * a1], 3.4 - .6 * k, '#f2f7ff', taper=(.2, .7), seed=j)
                    if j % 2: p_ = q + d * (a1 + 8 + 7 * k) + A([0, 1.8 * k * k]); r_ = 3.4 - .5 * k; c.fill(p_ + r_ * A([(-1, 0), (0, -1.2), (1, 0), (0, 1)]), '#ffffff', .9)
        return comp.over(im, c.layer(), .60)

    def beads(self, f, d):
        """the rain of one depth once time has stopped (90): round drops hanging in the air; from 104 they go up, accelerating, each with its tail BELOW it.  d 0 far, 1 middle, 2 near (out of focus)"""
        n, R0, sp, a_ = ((760, 3.0, .50, .60), (360, 6.2, 1.0, .85), (84, 15., 2.0, .95))[d]; r = np.random.default_rng(300 + d); c = Cel(ss=1 if d == 2 else 2)
        t = max(0., f - 104 - 1.5 * (2 - d)); y = r.uniform(-60, H + 60, n); x = r.uniform(-30, W + 30, n); R = R0 * r.uniform(.6, 1.45, n); go = r.uniform(0, 5, n); ph = r.uniform(0, 6.28, n)
        for i in range(n):
            u = max(0., t - go[i]); a = 1.9 * sp * (.8 + .4 * (R[i] / R0 - .6)); v = a * u; yy = (y[i] - .5 * a * u * u + 60) % (H + 120 + 90 * sp) - 60; xx = x[i] + (1.2 * math.sin(f * .25 + ph[i]) if u == 0 else 0.); rr = R[i]
            if d < 2 and u == 0: rr *= 1.0
            ln = min(v * 1.5, 130 * sp); w = rr * (1 - .25 * min(1, v / 40)); th = np.linspace(math.pi, 2 * math.pi, 9)                            # round head on top, tapering tail under it
            if ln < rr * 1.1: q = np.linspace(0, 2 * math.pi, 16, endpoint=False); P = np.c_[xx + np.cos(q) * rr, yy + np.sin(q) * rr]
            else: P = np.vstack([np.c_[xx + np.cos(th) * w, yy + np.sin(th) * w], [(xx + w * .62, yy + ln * .35), (xx + w * .18, yy + ln * .75), (xx, yy + ln), (xx - w * .18, yy + ln * .75), (xx - w * .62, yy + ln * .35)]])
            if d == 0: c.fill(P, '#dfe9fb', a_); continue
            c.fill(P, '#53638f', .85 * a_); q = np.linspace(0, 2 * math.pi, 14, endpoint=False)
            c.fill(np.c_[xx + np.cos(q) * w * .80, yy + rr * .10 + np.sin(q) * rr * .76], '#c6d8fb', .9 * a_); c.fill(np.c_[xx + w * .10 + np.cos(q) * w * .55, yy - rr * .02 + np.sin(q) * rr * .50], '#7488b6', .55 * a_)
            c.fill(np.c_[xx - w * .34 + np.cos(q) * w * .24, yy - rr * .40 + np.sin(q) * rr * .22], '#ffffff', a_)
            if ln > rr * 1.1: c.line([(xx - w * .1, yy + rr * .7), (xx - w * .05, yy + ln * .55), (xx, yy + ln * .92)], max(1., w * .40), '#e9f1ff', taper=(.1, .8), seed=i, alpha=.8 * a_)
        L = c.layer()
        return cv2.GaussianBlur(L, (0, 0), 3.4) * .9 if d == 2 else L

    def rising(self, im, f):
        """tears leave the face and go up"""
        c = Cel(); r = np.random.default_rng(77); p = self.KEYS['c'][1]; h = YHead().pose(*p)
        src = [(-.25, -.20), (-.44, -.22), (.25, -.22), (.42, -.26), (-.22, -.42), (.28, -.45), (-.30, -.66), (.30, -.70), (-.05, -.95), (.10, -.97), (-.55, -.42), (.0, -.12)]
        for i in range(30):
            t0 = 104 + i * .62 + r.uniform(-1, 2); t = f - t0; x0, y0 = src[i % 12]; R = r.choice([7, 11, 16, 24], p=[.35, .3, .22, .13])
            if t < 0: continue
            q = h.P([(x0 + r.normal(0, .05), y0 + r.normal(0, .03))])[0]; y = q[1] - .55 * t * t - 1.0 * t; x = q[0] + 5 * math.sin(t * .35 + i); sx = 1 - .10 * min(1, t / 12); sy = 1 + .22 * min(1, t / 12)
            a = np.linspace(0, 6.283, 20, endpoint=False); P = np.c_[x + np.cos(a) * R * sx, y + np.sin(a) * R * sy]
            c.fill(P, '#46557e', .55); c.fill(np.c_[x + np.cos(a) * R * .82 * sx, y + R * .14 + np.sin(a) * R * .78 * sy], '#c9dcff', .75); c.fill(np.c_[x + np.cos(a) * R * .58, y - R * .04 + np.sin(a) * R * .5], '#5f6f9c', .5); c.fill(np.c_[x - R * .32 + np.cos(a) * R * .2, y - R * .4 + np.sin(a) * R * .24], '#ffffff', .95); c.fill(np.c_[x + R * .25 + np.cos(a) * R * .34, y + R * .5 + np.sin(a) * R * .16], '#ffffff', .75)
        return comp.over(im, c.layer())

    def frame(self, f):
        k = float(smooth(110, 126, f)); self.finish = dict(bloom=.34 * (1 - k), vig=.22 * (1 - k), soft=.16 * (1 - k), ca=.9 * (1 - k), grain=.013 * (1 - .7 * k), shoulder=f < 124)        # the optics let go as the white comes
        if f < 72: return self.wide_frame(f)
        if f >= 128: return np.full((H, W, 3), .990, f32)
        return self.close(f)
