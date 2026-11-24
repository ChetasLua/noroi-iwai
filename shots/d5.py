"""d5: Hollow Purple.  0 Blue (everything leans in), 9 Red (everything leans out), 27-45 the hands bring them together and the colours
fight on his face, 45 purple and silence, 54 release past the lens: the erased tunnel."""
import numpy as np, cv2, math, os
from jj import W, H as HH, comp, kit, fx, world, hands as HD
from jj.paint import rgb, f32, smooth, fbm, grid, lum, Canvas, arcs
from jj.ink import Cel, spl, resample, paste
from jj.cam import Camera3D
from chars.gojo import Gojo, gojo_head, PAL, GC, GF, rim, eye_glow
from chars import gojo_keys as GK
from chars import gojo_hover as GV
from chars.gojo_cache import Plates
from jj.fx_orb import star_core as SC

VER = 'd5_v6'
PX, PY = 860., 420.                 # where he hovers in the wide
PVIO = dict(fx.VIOLET) if isinstance(fx.VIOLET, dict) else fx.VIOLET


class Shot:
    id = 'd5'; n = 72
    finish = dict(bloom=.28, vig=.18, halation='#dcd4ff', soft=.12)
    EVENTS = [(0, 'orb_blue', dict(size=.9, note='Blue ignites in his right hand: implosion, air sucked in')), (0, 'blue_loop', dict(end=44)), (9, 'orb_red', dict(size=1.0, note='Red ignites: shock outward, windows rattle')), (9, 'red_loop', dict(end=44)),
              (18, 'surge', dict(size=.5, end=26)), (27, 'cut', dict(size=.4, note='close: both orbs roar left / right')), (27, 'merge_rise', dict(end=45, note='two pitches bending toward each other')),
              (45, 'purple_ignite', dict(size=1.0, note='one hard crack, then nothing')), (46, 'silence', dict(end=52)), (52, 'inhale', dict(size=.3)), (54, 'release', dict(size=1.0, note='Hollow Purple leaves the hand')),
              (56, 'bore', dict(size=.3, note='touches tower 1')), (57, 'bite', dict(size=.35, note='clean hole, no crash: a swallowed thump, glass')), (58, 'bore', dict(size=.45)), (59, 'bite', dict(size=.5)), (60, 'bore', dict(size=.65)), (61, 'bite', dict(size=.7)), (62, 'bore', dict(size=.9, note='fills the last tower')),
              (63, 'pass_lens', dict(size=1.0, note='it goes past the camera, Doppler, sub drop')), (64, 'collapse', dict(size=.6, note='severed floors let go, tower 3')), (65, 'collapse', dict(size=1.0, note='severed floors let go, the near tower; masonry falling to the end')), (66, 'collapse', dict(size=.4))]

    # ---------------------------------------------------------------- plates
    def setup(self):
        self._memo = {}
        def build():
            P = world.plate('skyline'); A = {'city': P.img.astype(np.float16), 'near': P.layers['near'].astype(np.float16)}; A.update(self._figs()); A['clouds'] = self._clouds(); return A
        self.A = Plates(VER, build)
        self._bust_cache = {}

    def _figs(self):
        """hover drawings on twos (0..26), release drawings on twos (56..71), each cropped with its origin and the two hands in full-frame px"""
        out = {}; cam = Camera3D((0., -5.0, .25), (0., 0., 1.05), 1900., W, HH)
        def put(k, L, I):
            b = np.argwhere(L[..., 3] > .02); (y0, x0), (y1, x1) = b.min(0) - 12, b.max(0) + 12; y0, x0 = max(0, y0), max(0, x0); hs = I['hands']
            out['fig_' + k] = L[y0:y1, x0:x1]; out['fig_' + k + '_h'] = np.array([hs['R'][0], hs['R'][1], hs['L'][0], hs['L'][1], x0, y0], f32)
        for f in range(0, 27, 2): put('hv%02d' % f, *GV.drawing(float(f), cam))
        for k in range(8): put('rs%d' % k, *GV.shot(2. * k, cam, 1.25 if k < 4 else 1.))
        return out

    def _clouds(self):
        """a low deck of torn night cloud over the city, lit from within later (premult layer, 2400x1350)"""
        from numpy import clip
        h, w = 1350, 2400; n = fbm(h, w, (520, 210, 90, 40, 18), None, 7, aspect=2.6); y = np.linspace(0, 1, h)[:, None]; band = np.exp(-((y - .27) / .15) ** 2); m = smooth(.15, 1.25, n + band * 1.5 - 1.0)
        c = np.zeros((h, w, 3), f32) + rgb('#2b2458'); lit = smooth(-1.5, 1.8, n - cv2.GaussianBlur(np.roll(n, -14, 0), (0, 0), 8) * .8); c = c + lit[..., None] * rgb('#5d3f8c') * .7
        cv = Canvas(h, w, img=c); r = np.random.default_rng(3); ys, xs = np.nonzero(m > .25); k = r.integers(0, len(xs), 2600); px, py = xs[k], ys[k]; cv.strokes(arcs(px, py, r.normal(0, .25, 2600), r.uniform(60, 240, 2600), r.normal(0, .5, 2600)), r.uniform(6, 22, 2600), np.clip(c[py, px] * r.uniform(.75, 1.3, (2600, 1)), 0, 1).astype(f32), op=.6, dry=.5, seed=2)
        return np.dstack([cv.img * m[..., None], m]).astype(f32)

    # ---------------------------------------------------------------- wide
    def city(self, cx, cy, z, near=True, par=1.0):
        mv = lambda a, k: cv2.warpAffine(a, np.float32([[z * k, 0, W / 2 - cx * z * k], [0, z * k, HH / 2 - cy * z * k]]), (W, HH), flags=cv2.INTER_AREA if z * k < .98 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 if a.shape[2] == 3 else cv2.BORDER_CONSTANT)
        img = mv(self.A['city'], 1.0) * .78
        return img, (mv(self.A['near'], 1.0 * par) if near else None)

    REF = (960., 560.)                                                # fixed point of the hover drawings (native px)

    def _rock(self, c, p, s, seed, rot, lit, dark='#0b0a18', mid='#25234a', vel=None, stretch=1.):
        """a torn piece of building: three flat facets, one lit, one dark, an ink edge on the big ones"""
        r = np.random.default_rng(seed); n = int(r.integers(5, 8)); a = np.sort(r.uniform(0, 6.283, n)) + rot; Q = np.c_[np.cos(a), np.sin(a)] * s * r.uniform(.62, 1.1, (n, 1)) * (1, .8)
        if vel is not None and stretch > 1.: u = vel / (np.hypot(*vel) + 1e-6); al = Q @ u; Q = Q + np.outer(al * (stretch - 1), u)
        c.fill(Q + p, mid); ctr = np.array([r.uniform(-.2, .2), r.uniform(-.2, .1)]) * s; k = n // 2; c.fill(np.vstack([Q[:k + 1], [ctr]]) + p, dark); c.fill(np.vstack([Q[k:k + 2], [ctr]]) + p, lit)
        if s > 12:
            kit.outline(c, Q + p, 1.6 + s * .025, '#04040c', seed)
            if s > 20: c.line(np.array([ctr + p, (ctr + Q[k - 1]) / 2 + p + (2, 2), Q[k - 1] + p]), 1.6, '#04040c', taper=(.1, .6), seed=seed); q = Q.mean(0) * .5 + p; c.fill(q + np.array([(-.22, -.12), (.02, -.16), (.04, .04), (-.2, .08)]) * s, '#e8c684')

    def wide(self, f):
        x, y = grid(); u = f / 26.; ue = u * u * (3 - 2 * u) * .4 + u * .6; cx, cy, z = 1140 + 96 * ue, 604 - 22 * u, .86 + .05 * u
        img, _ = self.city(cx, cy, z, near=False); pulse = 1 + .05 * math.sin(f * 1.3)
        zn = z * 1.16; cn = 1190 + 250 * (ue - .5); near = cv2.warpAffine(self.A['near'], np.float32([[zn, 0, W / 2 - cn * zn], [0, zn, HH / 2 - (cy - 34) * zn]]), (W, HH), flags=cv2.INTER_LINEAR)
        cl = cv2.warpAffine(self.A['clouds'], np.float32([[.86, 0, -10 - f * 5.2], [0, .86, -30]]), (W, HH), flags=cv2.INTER_AREA); img = comp.over(img, cl, .75)
        k0 = 10 if f == 9 else f // 2 * 2; fig = self.A['fig_hv%02d' % k0]; hh = self.A['fig_hv%02d_h' % k0]; h1_ = self.A['fig_hv%02d_h' % min(26, k0 + 2)]; m_ = 0. if f in (8, 9) else (f - k0) / 2.; sc = .72 + .14 * u
        px, py = 1046 - 196 * ue, 470 - 16 * u + 7 * math.sin(f * .30); P = lambda q: np.array([px, py]) + (np.asarray(q, float) - self.REF) * sc
        hb, hr = P(hh[:2] * (1 - m_) + h1_[:2] * m_), P(hh[2:4] * (1 - m_) + h1_[2:4] * m_); gap = 92
        ob, orr = hb + (-34, -gap * .52), hr + (34, -gap * .52)
        kb = float(comp.lerp_keys(f, [(0, .35), (2, 1.25), (5, .95), (9, 1.0), (26, 1.12)])); kr = 0. if f < 9 else float(comp.lerp_keys(f, [(9, .4), (11, 1.35), (14, .95), (26, 1.12)]))
        bl = comp.radial(ob[0], ob[1], 900, 640, power=1.8)[..., None] * rgb('#2a6cff') * .75 * kb; rl = comp.radial(orr[0], orr[1], 900, 640, power=1.8)[..., None] * rgb('#ff2a3a') * .70 * kr
        img = img * (1 + 1.6 * (comp.radial(ob[0], HH, 1500, 900, power=1.6)[..., None] * rgb('#4a86ff') * kb + comp.radial(orr[0], HH, 1500, 900, power=1.6)[..., None] * rgb('#ff5a50') * kr)) * (1 - .12 * min(1, kb)) + bl * .6 + rl * .6
        img = comp.over(img, near)
        if kb > 0: img = fx.pinch(img, ob[0], ob[1], 480 * (1 if f > 4 else .5 + .13 * f), .32 * kb * (1.35 if f < 5 else 1.), .6)
        if kr > 0: img = fx.pinch(img, orr[0], orr[1], 460, -.28 * kr * (1.5 if f < 13 else 1.), 0.)
        # light torn out of the windows and drawn into Blue
        if 'lights' not in self._memo:
            c_ = self.A['city']; g = lum(c_[640:1250]); ys, xs = np.nonzero(g > .60); r = np.random.default_rng(12); k = r.integers(0, len(xs), 150); self._memo['lights'] = (np.c_[xs[k], ys[k] + 640.], np.clip(np.asarray(c_[ys[k] + 640, xs[k]], f32) * 1.15, 0, 1), r.random((150, 4)))
        LP, LC, LR = self._memo['lights']; c = Cel(); r = np.random.default_rng(5)
        for k in range(150):
            p0 = (LP[k] - (cx, cy)) * z + (W / 2, HH / 2); ph = (LR[k, 0] + f / 15. * (.8 + .6 * LR[k, 1])) % 1.; red = kr > 0 and LR[k, 3] < .34
            if not (-60 < p0[0] < W + 60 and p0[1] < HH + 60): continue
            if red:                                                    # near Red the lights are blown outward and die
                d = p0 - orr; d /= np.hypot(*d); t0 = ph ** .7; a_ = p0 + d * (30 + 260 * t0); c.line(np.array([a_, a_ + d * 20, a_ + d * (26 + 60 * (1 - t0))]), 1.2 + 2.2 * (1 - t0), LC[k] * .5 + rgb('#ff7a60') * .5, taper=(.5, .4), seed=k, alpha=.8 * (1 - t0)); continue
            v = ob - p0; L_ = np.hypot(*v); n_ = np.array([-v[1], v[0]]) / L_; c1, c2 = p0 + (0, -.28 * L_) + n_ * L_ * .16, ob - v * .22 + n_ * L_ * .26; t0 = ph ** 1.5; t1 = min(1., t0 + .07 + .20 * t0); q = np.linspace(t0, t1, 9)[:, None]; B = (1 - q) ** 3 * p0 + 3 * (1 - q) ** 2 * q * c1 + 3 * (1 - q) * q * q * c2 + q ** 3 * ob
            c.line(B, 1.3 + 3.2 * LR[k, 2] * (.4 + t0), LC[k] * (1 - t0) + rgb('#a8e4ff') * t0, taper=(.75, .12), seed=k, alpha=.30 + .65 * min(1, ph * 3))
        # torn masonry: spirals into Blue, is thrown out of Red.  a third of it passes in front of him
        fr = Cel()
        for k in range(46):
            a = r.uniform(.15, 2.95) if k % 4 else r.uniform(3.2, 6.1); d0 = r.uniform(620, 1180); ph = (r.random() + f / 24. * r.uniform(.7, 1.25)) % 1.; e = ph ** 1.7; d = d0 * (1 - e) + 46; th = a + 2.1 * e; p = ob + np.array([math.cos(th) * 1.2, math.sin(th) * .86]) * d; s = (5 + 34 * r.random() ** 2.2) * (.22 + .78 * (d / d0)) * (.8 if k % 3 else 1.35); v = np.array([-math.sin(th) * 1.2, math.cos(th) * .86]) * 2.1 - np.array([math.cos(th) * 1.2, math.sin(th) * .86]) * 1.4
            if p[1] > HH + 60 or np.hypot(*(p - ob)) < 50: continue
            C_ = fr if k % 3 == 0 else c
            if s > 5: C_.line(np.array([p - v / np.hypot(*v) * (26 + 60 * e), p - v / np.hypot(*v) * 10, p]), s * .34, '#5aa8ff', taper=(.8, .1), seed=k, alpha=.5)
            self._rock(C_, p, s, 100 + k, f * (.10 + .10 * r.random()) * (1 if k % 2 else -1), '#8fc4ff', vel=v, stretch=1 + 1.6 * max(0, 1 - d / 260.))
        if kr > 0:
            for k in range(44):
                burst = k < 26; ph = (f - 9) / r.uniform(9., 15.) if burst else ((f - 9) / 17. * r.uniform(.8, 1.3) + r.random()) % 1.; a = r.uniform(0, 6.283)
                if ph > 1.1 or (f < 12 and not burst): continue
                w_ = np.array([math.cos(a) * 1.25, math.sin(a) * .9]); d = 70 + 1250 * max(0, ph) ** .72; p = orr + w_ * d; s = (5 + 30 * r.random() ** 2) * (.5 + .9 * ph)
                if not (-80 < p[0] < W + 80 and -80 < p[1] < HH + 80): continue
                C_ = fr if k % 3 == 0 else c; C_.line(np.array([p - w_ * (50 + 150 * (1 - min(1, ph))), p - w_ * 16, p]), s * .42, '#ff6a4a' if k % 2 else '#ffd0b8', taper=(.85, .1), seed=k + 300, alpha=.75); self._rock(C_, p, s, 400 + k, f * .3 * (1 if k % 2 else -1), '#ffb09a')
        img = comp.over(img, c.layer())
        if 9 <= f < 12: S = Cel(); fx.shockwave(S, orr[0], orr[1], (f - 9) / 8., 60, 1500, .80, 15, 4, '#ffe9e0', '#c03040'); sl = S.layer(); img = img + comp.blur(sl, 9)[..., :3] * .5; img = comp.over(img, sl, .9)
        # him
        sw, sh_ = int(fig.shape[1] * sc), int(fig.shape[0] * sc); tl = P(hh[4:6]); lay = np.zeros((HH, W, 4), f32); paste(lay, cv2.resize(fig, (sw, sh_), interpolation=cv2.INTER_AREA), int(round(tl[0])), int(round(tl[1])))
        xs = smooth(px - 70, px + 70, x); lay = kit.tint(lay, '#8f86d8', .50)
        lay = rim(lay, 4, 1, '#4aa0ff', .95 * min(1, kb), soft=.5, where=1 - xs); lay = rim(lay, -4, 1, '#ff5a4a', .95 * min(1, kr), soft=.5, where=xs)
        img = fx.put(img, fx.orb_blue(ob[0], ob[1], 68 * kb * pulse, f, 1., 3, 1.))
        if kr > 0: img = fx.put(img, fx.orb_red(orr[0], orr[1], 66 * kr * pulse, f, 1., 5, 1.))
        img = comp.over(img, lay); img = comp.over(img, fr.layer())
        # what the lens passes on its way: torn cloud
        wz = 2.1; wisp = cv2.warpAffine(self.A['clouds'], np.float32([[wz, 0, -300 - f * 44], [0, wz, 260]]), (W, HH), flags=cv2.INTER_LINEAR); wisp = comp.blur(wisp, 9); img = comp.over(img, wisp * (smooth(640, 1080, y) * .85 + smooth(170, 0, y) * .6)[..., None])
        a = {0: 8, 1: 6, 2: 4, 9: 18, 10: 13, 11: 9, 12: 6}.get(f, 1.5 if f < 9 else 2.4); d = kit.shake(f, a, 3)
        if f < 2: img = comp.screen(img, (comp.radial(ob[0], ob[1], 1500, 1100, power=.9)[..., None] * 1.3) * rgb('#e6f4ff') * (.95 if f == 0 else .25))
        if f == 9: img = kit.impact(img, 'red', .30, .62, ink='#120008', paper='#fff2ea', accent='#e01828')
        ro = math.radians(-1.4 + 2.8 * ue); M_ = cv2.getRotationMatrix2D((W / 2, HH / 2), math.degrees(ro), 1.045); M_[:, 2] += d; return cv2.warpAffine(img, M_, (W, HH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # ---------------------------------------------------------------- close: the colours fight
    def close(self, f):
        x, y = grid(); t = (f - 27) / 18.; e = float(comp.ease(min(1, t))) if t < 1 else 1.
        cx, cy, s = 960, 400, 246; gap = 640 * (1 - e ** 1.3) + 0; ox, oy = 960, 830 - 40 * e; r0 = 118
        fd = f // 2 * 2
        if f < 45: key = ('c', fd)
        else: key = ('p', 45 if f < 48 else 48 if f < 52 else 52)
        if key not in self._bust_cache:
            wx = .10 * math.sin(fd * .8); ex = dict(band=False, mouth='half', lid='cold', brow='cold', iris=.085, pupil=.24 if f < 45 else .17, glow=.12, gaze=(0, -.004), flip=False)
            hgap = max(250., gap * .62 + 190)
            def over(c, H, B, pal):
                for sg in (-1, 1):
                    hp = np.array([ox + sg * hgap, oy + 26 + 40 * (1 - e)]); w_ = hp + np.array([sg * 120, 96]); el = w_ + np.array([sg * 190, 330])
                    GC.sleeve(c, el + (-sg * 60, 420), el, w_, 130, 118, 78, pal, s, seed=20 + sg, lit=-sg, hand_fn=lambda sg=sg, hp=hp: Gojo.hand(c, 'open' if e < .55 else 'splay', hp[0], hp[1], 150, 96 * sg if False else (112 if sg > 0 else -112) + 180 * 0, 8, 38 * sg, pal, 'L' if sg > 0 else 'R', 4 + sg, w=4.6, lit=(-sg * .9, .2, .4)))
            if f < 45:
                Ls = [Gojo.bust(cx, cy, s, .0, .12, .0, 'wind', ex, li, key=k, f=fd, wind=(wx, .14), seed=4, body=(0, .10, 0), left=False, right=False, arms=None, over=over) for li, k in (('blue', 'BL'), ('red', 'BR'))]
            else: Ls = [Gojo.bust(cx, cy, s, .0, .12, .0, 'wind', ex, 'violet', key='B', f=0, wind=(0, .10), seed=4, body=(0, .10, 0), left=False, right=False, arms=None, over=over)]
            self._bust_cache.clear()
            self._bust_cache[key] = Ls
        Ls = self._bust_cache[key]
        # background: the sky he stands in, split by the two lights
        bg = cv2.warpAffine(self.A['city'], np.float32([[1.15, 0, -420], [0, 1.15, 40]]), (W, HH), borderMode=cv2.BORDER_REFLECT_101); bg = comp.blur(bg, 9) * .35
        if f < 45:
            mt_ = min(.795, .795 * (t ** 1.15)); ee = mt_ / .80; gp = max(40., gap * 1.15 + 180); dd_ = gp * (1 - ee ** 1.7) * .5; an = 3.6 * ee ** 3 * 1.6; ca, sa = math.cos(an), math.sin(an)      # where purple_merge puts Blue (-) and Red (+)
            bnd = 960.; side = (x - 960.) * ca + (y - 560.) * sa * .55; zig = 34 * (np.abs(((y * .011) % 1) - .5) * 2 - .5) + 18 * np.sin(y * .03); m = smooth(-6, 6, side - zig)[..., None]
            bg = bg + (1 - m) * (comp.radial(ox - ca * dd_ * 1.1, oy - sa * dd_, 1300, 1000, power=1.5)[..., None] * rgb('#1f5cff') * .85) + m * (comp.radial(ox + ca * dd_ * 1.1, oy + sa * dd_, 1300, 1000, power=1.5)[..., None] * rgb('#e0182c') * .80)
            L = Ls[0] * (1 - m) + Ls[1] * m
            L = rim(L, 5, -3, '#bfe8ff', .9, soft=.7, where=1 - m[..., 0]); L = rim(L, -5, -3, '#ffd0b0', .9, soft=.7, where=m[..., 0])
            c = Cel(); r = np.random.default_rng(fd // 4)
            for i in range(46): a = r.uniform(0, 6.283); d0 = r.uniform(500, 900); u = np.array([math.cos(a), math.sin(a) * .8]); p = np.array([960, 520]) + u * d0; c.line(np.array([p, p + u * 160, p + u * r.uniform(260, 600)]), r.uniform(2, 9), '#9fd0ff' if (p[0] - 960) * ca + (p[1] - 560) * sa * .55 < 0 else '#ff9a86', taper=(.6, .1), seed=i, alpha=.55)
            bg = comp.over(bg, c.layer())
            img = comp.over(bg, L)
            img = eye_glow(img, gojo_head(True).pose(cx, cy, s, 0, .12, 0), .35)
            mt = min(.795, .795 * (t ** 1.15))
            img = fx.put(img, fx.purple_merge(ox, oy, r0, mt, f, gap=max(40., gap * 1.15 + 180), seed=3, ang=math.pi if False else 0.))
            # bolts between the two, drawn over his coat
            a = 2 + 7 * e; d = kit.shake(f, a, 5); img = comp.cam2d(img, W / 2 + d[0], HH / 2 + d[1], 1.03 + .05 * e)
            return img
        u = f - 45
        if u == 0:
            img = np.ones((HH, W, 3), f32) * rgb('#fbf4ff'); L = Ls[0]; a = L[..., 3:4]; g = lum(L[..., :3]) / np.maximum(L[..., 3], 1e-3); ink = smooth(.34, .38, g)[..., None]
            c = Cel(); kit.focus_lines(c, ox, oy, 300, 1800, 220, 12, '#12042a', 7, ry=.85); img = comp.over(img, c.layer()); img = img * (1 - a) + a * (rgb('#12042a') * (1 - ink) + rgb('#b06cff') * ink * .9)
            q = np.linspace(0, 6.283, 80); c = Cel(); c.fill(np.c_[ox + np.cos(q) * r0 * 1.25, oy + np.sin(q) * r0 * 1.25], '#12042a'); c.fill(np.c_[ox + np.cos(q) * r0 * 1.12, oy + np.sin(q) * r0 * 1.12], '#ffffff'); return comp.over(img, c.layer())
        # silence: the world goes to violet-black, only the sphere and his face
        bg = bg * .30 + comp.radial(ox, oy, 1200, 900, power=2)[..., None] * rgb('#4a14b8') * .55
        L = Ls[0].copy(); L = kit.para(L, (ox, oy), (960, -80), [(0, '#ffffff'), (.35, '#d9c2f8'), (.8, '#6a55a8'), (1, '#3a2c70')], 'mul'); L = rim(L, 0, -5, '#f0dcff', .8, soft=.8)
        img = comp.over(bg, L); img = eye_glow(img, gojo_head(True).pose(cx, cy, s, 0, .12, 0), .45, '#9fe0ff')
        rr = r0 * (1.16 if u == 1 else 1.0 + .015 * math.sin(u * 2)); rr = rr * (.86 if f >= 52 else 1)
        img = fx.put(img, fx.orb_purple(ox, oy, rr, 46 if u < 8 else f, .8, 3, 1.0 if u < 3 else .35))
        if u == 1: img = img * 1.15
        return comp.cam2d(img, 960, 540 + (6 if f >= 52 else 0), 1.08 + .004 * u - (.015 if f >= 52 else 0))

    # ---------------------------------------------------------------- 54.. release and the tunnel
    PATH = [(836., 352., 17.), (900, 400, 34), (985, 462, 75), (1090, 548, 163), (1230, 668, 323), (1440, 838, 612), (1740, 1080, 1122), (2200, 1440, 2125)]   # x, y, radius in frame px, far -> lens

    def _tunnel(self):
        P = np.array(self.PATH); C = spl(P[:, :2], n=40); t = np.linspace(0, 1, len(C)); return C, np.exp(np.interp(t, np.linspace(0, 1, len(P)), np.log(P[:, 2])))

    # radius where the sphere meets it, left / right / top extents in radii from the centre of the sphere, face, side, wall steps, share of lit windows
    TW = ((84., 1.15, 1.2, 1.55, '#262a62', '#171945', 8, .46), (124., -.30, 1.75, .58, '#23275e', '#151742', 9, .44), (183., 1.05, -.28, .10, '#20245a', '#13153c', 10, .42), (270., -.20, .84, 1.85, '#1c2052', '#101236', 12, .40),
          (395., 1.12, -.50, .20, '#181c4c', '#0e1030', 14, .36), (575., -.42, 1.02, 1.28, '#141840', '#0b0d2a', 15, .33), (830., 2.05, -.66, .05, '#101336', '#080922', 16, .30))
    HIT = (56, 57, 58, 59, 60, 61, 62)                                # the frame the sphere touches each tower (its bite is seen one frame later)
    RING0 = (900., 505., 4.0, 220., 900.)                             # ground point under the hand: screen x, horizon, depth, height * focal, focal

    def _stage(self):
        """the stage of the release: sky and city (closed / hole in the cloud / lights out), seven towers intact and bitten, each on its own cel"""
        x, y = grid(); cx, cy, z = 1290, 596, .90; HZ = 505.; D = {}
        img, near = self.city(cx, cy, z); cl = cv2.warpAffine(self.A['clouds'], np.float32([[.86, 0, -60 - 26 * 1.2], [0, .86, -30]]), (W, HH), flags=cv2.INTER_AREA); C, R = self._tunnel()
        hx, hy = 930., 300.; dh = np.sqrt(((x - hx) / 1.75) ** 2 + (y - hy) ** 2); rag = fbm(HH, W, (90, 34, 13), None, 8) * 9 + fbm(HH, W, (9, 4), None, 9) * 2.; hole = smooth(184, 200, dh + rag); sky = smooth(HZ + 4, HZ - 8, y); ang = np.arctan2(y - hy, x - hx)
        dens = np.clip(cl[..., 3] + .55 * sky * np.exp(-((y - 300) / 190.) ** 2), 0, 1); deck = rgb('#4a3a7c') * (.75 + .25 * smooth(150, 480, y))[..., None]; deck = deck * (1 - cl[..., 3:4] * .6) + cl[..., :3] * .6 * 1.25
        D['base0'] = img * (1 - (dens * .85)[..., None]) + deck * (dens * .85)[..., None]
        im = img * (1 - (dens * hole * .85)[..., None]) + deck * (dens * hole * .85)[..., None]; r = np.random.default_rng(2); stars = np.zeros((HH, W), f32)
        for _ in range(1500): px, py = int(hx + r.normal(0, 210)), int(hy + r.normal(0, 110)); stars[max(0, min(HH - 1, py)), max(0, min(W - 1, px))] = r.random() ** 4 * 2.2
        for _ in range(700): u_ = r.normal(0, 200); px, py = int(hx + u_), int(hy - u_ * .35 + r.normal(0, 22)); stars[max(0, min(HH - 1, py)), max(0, min(W - 1, px))] += r.random() ** 2 * .5
        ins = (1 - hole) * sky; skyc = rgb('#0c1748') * (1 - smooth(110, 480, y)[..., None]) + rgb('#3a56a8') * smooth(110, 480, y)[..., None]; milky = np.exp(-np.abs(y - hy + (x - hx) * .35) / 46.) * .22
        im = im * (1 - ins * .92)[..., None] + (skyc + milky[..., None] * rgb('#8fa8f0')) * (ins * .92)[..., None] + (cv2.GaussianBlur(stars, (0, 0), .8) * 3.4 * ins)[..., None] * rgb('#eef2ff')
        rimc = np.exp(-np.abs(dh - 196 + rag) / 4.) * sky * np.clip(.5 + .5 * np.sin(ang * 3 + 1) + .25 * fbm(HH, W, (60, 20), None, 12), 0, 1); lit = np.exp(-np.clip(dh + rag - 194, 0, 999) / 52.) * hole * sky * dens
        D['base1'] = im + rimc[..., None] * rgb('#d9c2ff') * .30 + lit[..., None] * rgb('#9a6cf0') * .55; D['C'] = C; D['R'] = R
        g = lum(img); hot = smooth(.02, .10, g - cv2.GaussianBlur(g, (0, 0), 22) * .85)[..., None]; mass = cv2.GaussianBlur(img, (0, 0), 1.2); md = mass.mean(2, keepdims=True) * rgb('#5a52b8') * .62 + rgb('#07071a') * .8
        fog = np.exp(-np.clip(y - 505, 0, 999) / 120.)[..., None]; body = np.minimum(img * (1 - hot) + cv2.GaussianBlur(img, (0, 0), 4) * hot * .45, .30); gl_ = lum(body)[..., None]; D['dark'] = np.clip((body * .55 + gl_ * rgb('#7a74d0') * .9) * 1.15 - .012, 0, 1) * .80 + fog * rgb('#4a3c96') * .13      # the same city with every light out
        t_ = np.linspace(0, 1, 160); circ = np.c_[np.cos(t_ * 2 * np.pi), np.sin(t_ * 2 * np.pi)]
        for ti, (R_, wl, wr, up, face, side, thick, lit_) in enumerate(self.TW):
            i0 = int(np.argmin(np.abs(R - R_))); ctr, r_ = C[i0], R[i0]; ib = max(0, i0 - thick); cb, rb = C[ib], R[ib]; x0, x1, y0, y1 = ctr[0] - wl * r_, ctr[0] + wr * r_, ctr[1] - up * r_, HH + 60; dv = (np.array([940., 470.]) - np.array([x0, y0])) * .10
            for cut in (0, 1):
                rr = np.random.default_rng(60 + ti); c = Cel(); fc, sd = (face, side) if not cut else (np.clip(rgb(face) * .92, 0, 1), np.clip(rgb(side) * .85, 0, 1))
                if dv[0] < 0: c.fill(np.array([(x0, y0), (x0, y1), (x0 + dv[0], y1), (x0 + dv[0], y0 + dv[1])]), sd)
                else: c.fill(np.array([(x1, y0), (x1, y1), (x1 + dv[0] * .5, y1), (x1 + dv[0] * .5, y0 + dv[1]), (x1, y0)]), sd)
                c.fill(np.array([(x0, y0), (x1, y0), (x1 + dv[0], y0 + dv[1]), (x0 + dv[0], y0 + dv[1])]), '#2c2a66' if not cut else '#1e1b4a')
                if cut:
                    K = 120; a = np.linspace(0, 2 * np.pi, K + 1); fl = max(5., r_ / 9.); c.clip(np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
                    for k in range(K):
                        P = np.array([ctr + r_ * np.array([math.cos(a[k]), math.sin(a[k])]), ctr + r_ * np.array([math.cos(a[k + 1]), math.sin(a[k + 1])]), cb + rb * np.array([math.cos(a[k + 1]), math.sin(a[k + 1])]), cb + rb * np.array([math.cos(a[k]), math.sin(a[k])])])
                        lev = int((ctr[1] + r_ * math.sin(a[k])) / fl); h_ = ((lev * 7919) % 97) / 97.; low = math.sin(a[k]); col = rgb('#4a34a0') * (.80 - .45 * low)
                        if lev % 4 == 0: col = rgb('#0a0620')
                        elif h_ < .20: col = rgb('#8a5cf0') * (.55 + .5 * h_)
                        elif h_ < .55: col = col * .55
                        c.fill(P, np.clip(col, 0, 1))
                    for k in range(0, K, 5): c.line(np.array([ctr + r_ * np.array([math.cos(a[k]), math.sin(a[k])]), (ctr + cb) / 2 + (r_ + rb) / 2 * np.array([math.cos(a[k]), math.sin(a[k])]), cb + rb * np.array([math.cos(a[k]), math.sin(a[k])])]), max(.7, r_ * .006), '#07031a', taper=(.05, .2), seed=k)
                    c.unclip()
                c.ctx.save(); c.ctx.new_path(); c.ctx.rectangle(x0, y0, x1 - x0, y1 - y0)
                if cut: c.ctx.new_sub_path(); c.ctx.arc(ctr[0], ctr[1], r_, 0, 2 * math.pi)
                c.ctx.set_fill_rule(1); c.ctx.clip(); c.grad(np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]), (0, y0), (0, HH), [(0, fc, 1), (.6, fc, 1), (1, sd, 1)]); ww, wh = 2.6 + r_ * .022, 3.0 + r_ * .026
                for yy in np.arange(y0 + wh * .8, HH + 40, wh * 1.55):
                    row = rr.random()
                    for xx in np.arange(x0 + ww * .7, x1 - ww, ww * 1.42):
                        q_ = rr.random(); cq = int(rr.integers(4)); bq = rr.uniform(.35, .95)
                        if q_ < lit_ * (.35 + row) and not cut: c.fill(np.array([(xx, yy), (xx + ww, yy), (xx + ww, yy + wh), (xx, yy + wh)]), np.clip(rgb(['#f2e2b4', '#e9edff', '#ffd58a', '#9fb4ff'][cq]) * bq, 0, 1), .95)
                        else: c.fill(np.array([(xx, yy), (xx + ww, yy), (xx + ww, yy + wh), (xx, yy + wh)]), '#05061a' if cut else '#0a0c26', .6)
                c.ctx.restore(); c.line(np.array([(x0, y0 + 1), ((x0 + x1) / 2, y0), (x1, y0 + 1)]), max(1.2, r_ * .009), '#8f7fe0', taper=(.05, .05), press=.4, seed=3); c.line(np.array([(x0 + 1, y0), (x0, (y0 + HH) / 2), (x0 + 1, HH)]), max(1., r_ * .006), '#5a4cc0', taper=(.02, .2), press=.4, seed=4)
                for k in range(3 if ti < 4 else 0): ax = x0 + (x1 - x0) * rr.uniform(.1, .6); c.line(np.array([(ax, y0 + 2), (ax, y0 - r_ * .09), (ax, y0 - r_ * rr.uniform(.12, .30))]), max(.9, r_ * .007), fc, taper=(0, .5), seed=k)
                L = c.layer(); d_ = np.sqrt((x - ctr[0]) ** 2 + (y - ctr[1]) ** 2)
                if cut: out_ = np.maximum(smooth(r_ - 1.2, r_ + 1.2, d_), ((x > x0) & (x < x1) & (y > y0)).astype(f32) * smooth(rb - 1., rb + 1., np.sqrt((x - cb[0]) ** 2 + (y - cb[1]) ** 2))); L = L * out_[..., None]
                ox = int(max(0, min(x0, x0 + dv[0]) - 12)); oy = int(max(0, y0 - r_ * .5 - 6)); ex = int(min(W, max(x1, x1 + dv[0]) + 16))
                if cut:
                    hc = Cel(); hc.line(np.vstack([circ, circ[:1]])[:161] * r_ + ctr, max(1.4, 2.2 + r_ * .006), '#ffffff', taper=(0, 0), press=.35, seed=int(R_)); hc.line(circ[8:120] * rb + cb, max(.8, r_ * .003), '#e3ccff', taper=(.2, .2), press=.4, seed=int(R_) + 1); D['h%d' % ti] = (hc.layer()[..., 3] * ((x > x0 - 2) & (x < x1 + 2) & (y > y0 - 2)))[oy:, ox:ex]
                D['t%d%s' % (ti, 'c' if cut else 'f')] = L[oy:, ox:ex]; D['o%d' % ti] = np.array([ox, oy, ctr[0], ctr[1], r_, x0, x1, y0], f32)
        return D

    SR = {56: 84., 57: 124., 58: 183., 59: 270., 60: 395., 61: 575., 62: 830., 63: 1800.}

    def stage(self, f):
        """56..71: the sphere comes toward the lens through seven planes of buildings; what it touched is gone one frame later, and the lights go with it"""
        if 'st' not in self._memo: self._memo['st'] = Plates(VER + 'stageC', self._stage)
        P = self._memo['st']; C, R = P['C'], P['R']; x, y = grid(); n = len(C)
        sR = self.SR.get(f, 0.); si = int(np.argmin(np.abs(R - sR))) if sR else n - 1; sp, sr = C[si], R[si]; fly = 56 <= f <= 62
        drift = max(0., f - 55); dx, dy = -1.6 * drift - .25 * max(0, f - 63) ** 1.5, -.5 * drift
        base = P['base0']; rad = {55: 60., 56: 140., 57: 230.}.get(f, 400.); dh = np.sqrt(((x - 930) / 1.75) ** 2 + (y - 300) ** 2); m = smooth(rad + 14, rad - 14, dh)[..., None]; base = base * (1 - m) + P['base1'] * m if f < 58 else P['base1']
        # lights out behind the shock ring (true circle on the ground plane)
        gx0, hz, Z0, HF, FF = self.RING0; rho = .28 * 1.64 ** (f - 55); gd = y > hz + 1.5; Zg = HF / np.maximum(y - hz, 1.5); Xg = (x - gx0) * Zg / FF; dg = np.sqrt(Xg ** 2 + (Zg - Z0) ** 2); out = smooth(rho, rho * .90 - .02, dg) * gd
        if f >= 64: out = np.maximum(out, smooth(hz - 70, hz - 20, y))
        base = base * (1 - out[..., None]) + P['dark'] * out[..., None]
        band = np.exp(-((dg - rho) / (.045 * rho + .03)) ** 2) * gd; base = base + band[..., None] * rgb('#b890ff') * (.8 if f < 63 else .4) * np.clip(Zg / Z0, .30, 1.)[..., None] + (np.exp(-np.clip(dg - rho, 0, 99) / (.30 * rho + .1)) * (dg > rho) * gd)[..., None] * base * rgb('#c8a8ff') * .9
        if fly:                                                        # the sky and the cloud are dragged after it
            w_ = smooth(hz + 60, hz - 120, y)[..., None]; pu = fx.pinch(base, sp[0], sp[1], 900 + sr * 1.2, .55, .5); pu = kit.zoomblur(pu, sp[0], sp[1], .10 + .04 * (f - 56), 15); base = base * (1 - w_ * .85) + pu * w_ * .85
        elif f >= 64: w_ = (smooth(hz + 20, hz - 160, y) * max(0., 1 - (f - 64) / 7.) * .6)[..., None]; base = base * (1 - w_) + kit.zoomblur(base, 1900, 1240, .07 * max(.2, 1 - (f - 64) / 7.), 13) * w_
        img = cv2.warpAffine(base, np.float32([[1, 0, dx * .25], [0, 1, dy * .25]]), (W, HH), borderMode=cv2.BORDER_REFLECT_101)
        if sR and f < 64: img = img * (.66 + .34 * (f >= 63)) + comp.radial(sp[0], sp[1], 760 + sr * 2.4, power=1.6)[..., None] * rgb('#7a2cf0') * (.75 if f < 62 else .95) + img * (comp.radial(sp[0], max(hz + 60, sp[1] + sr), 900 + sr * 2, 340 + sr, power=1.4) * gd)[..., None] * rgb('#c8a0ff') * 1.5
        img = self._ringwall(img, f, rho, dx, dy)
        img = self._him(img, f, 'rs%d' % min(7, max(0, (f - 56) // 2)), 262, (C[0][0] - 14 + dx * .3, C[0][1] - 16 + dy * .3), 1.2)

        def trail(img, a, b):                                         # the scar in the air, fat behind the sphere, closing with age
            ia, ib = int(np.argmin(np.abs(R - a))), min(si, int(np.argmin(np.abs(R - b))))
            if ib - ia < 4 or f > 68: return img
            age = max(0., f - 62) / 7.; Q = C[ia:ib] + (dx * .5, dy * .5); T = np.gradient(Q, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-6; N = np.c_[-T[:, 1], T[:, 0]]; k = np.arange(ia, ib); wb = (R[ia:ib] * (.14 * max(0., 1 - age) + (.62 * np.exp(-(si - k) / 17.) if fly else 0.)) + .8); r_ = np.random.default_rng(f); z1, z2 = (1 + .22 * np.sin(k * .55 + f * 2.1) + .12 * r_.normal(0, 1, len(k))), (1 + .22 * np.sin(k * .49 - f * 1.7) + .12 * r_.normal(0, 1, len(k)))
            c = Cel(); sh = lambda v, w: np.vstack([Q + N * (wb * v * (z1 if w else 1))[:, None], (Q - N * (wb * v * (z2 if w else 1))[:, None])[::-1]]); al = max(0., 1 - age); c.fill(sh(1.25, 1), '#3a0fa0', .55 * al); c.fill(sh(.9, 1), '#7a2cf0', .85 * al); c.fill(sh(.50, 1), '#c9a0ff', al); c.fill(sh(.24, 0), '#ffffff', al); L = c.layer(); return comp.over(img + comp.blur(L, 14)[..., :3] * .9, L)

        def suction(img, lo, hi, seed, nn=44):                         # air: lines bent in from the sides and pulled along the scar
            if f > 63 or f < 56: return img
            c = Cel(); r2 = np.random.default_rng(seed * 31 + f); ks = [i for i in range(n) if lo <= R[i] < hi and i <= si]
            if len(ks) < 3: return img
            fade = 1. if f < 64 else max(0., 1 - (f - 63) / 4.5)
            for j in range(nn if f < 64 else nn // 2):
                i = ks[int(r2.integers(len(ks)))]; i2 = min(n - 1, i + int(r2.uniform(14, 34))); a = r2.uniform(0, 6.283); v = np.array([math.cos(a), math.sin(a)]); r0, r1 = R[i] * r2.uniform(2.2, 5.2) + 20, R[i2] * r2.uniform(.5, .95); p0, p2 = C[i] + v * r0, C[i2] + v * r1; p1 = C[i] * .55 + C[i2] * .45 + v * r1 * 1.15; q = np.linspace(0, 1, 10)[:, None]; B = (1 - q) ** 2 * p0 + 2 * (1 - q) * q * p1 + q * q * p2; fat = j % 5 == 0
                c.line(B + (dx * .5, dy * .5), (R[i2] * .16 + 4) if fat else 1.2 + R[i2] * .022 * r2.random(), ('#4c3a94' if j % 2 else '#6a4cc0') if fat else ['#f2e8ff', '#b78cff', '#8fa4ff'][j % 3], taper=(.75, .25), press=.3, seed=j, alpha=(.28 if fat else .85) * fade)
            return comp.over(img, c.layer())

        def ball(img):
            c = Cel(); r2 = np.random.default_rng(f)
            for j in range(110): a = r2.uniform(0, 6.283); d0 = sr * r2.uniform(1.08, 1.45); v = np.array([math.cos(a), math.sin(a)]); q = sp + v * d0; Ls = sr * r2.uniform(.4, 2.6) ** 1. + 16; c.line(np.array([q, q + v * Ls * .5, q + v * Ls]), 1.2 + sr * .03 * r2.random(), ['#f6eaff', '#b47cff', '#6a2cf0'][j % 3], taper=(.2, .8), seed=j, alpha=.9)
            img = comp.over(img, c.layer())
            for k_, (rr_, al) in enumerate(((1.42, .85), (2.0, .5))):   # pressure shells
                S = Cel(); fx.ragged_ring(S, sp[0], sp[1], sr * rr_, sr * rr_, sr * (.030, .016)[k_] + 1.2, 11 + k_ + f, '#e6d2ff', gaps=.30, n=9); L = S.layer() * al; img = img + comp.blur(L, 8)[..., :3] * .5; img = comp.over(img, L)
            return fx.put(img, fx.orb_purple(sp[0], sp[1], sr * .98, f, 1.35, 3, 1.2))
        prev = 14.; done = not fly
        for ti, hit in enumerate(self.HIT):
            R_ = self.TW[ti][0]; img = trail(img, prev, R_); img = suction(img, prev, R_, ti)
            if not done and sr <= R_ * 1.01 and not (f == 56 and ti == 0): img = ball(img); done = True
            ox, oy, tx, ty, tr, x0, x1, y0 = P['o%d' % ti]; ox, oy = int(ox), int(oy); gone = f > hit; L = P['t%d%s' % (ti, 'c' if gone else 'f')]
            full = np.zeros((HH, W, 4), f32); full[oy:oy + L.shape[0], ox:ox + L.shape[1]] = L; par = .35 + .9 * R_ / 830.; tm = np.float32([[1, 0, dx * par], [0, 1, dy * par]])
            if not gone and f == hit:                                  # contact: every window on it blazes
                g = comp.radial(tx, ty, tr * 2.1, power=1.3); w_ = smooth(.16, .5, lum(full[..., :3])); full[..., :3] = full[..., :3] * (1 + 1.2 * g[..., None]) + (g * w_ * full[..., 3])[..., None] * rgb('#f0dcff') * 1.3 + (comp.radial(tx, ty, tr * 1.12, power=.8) * full[..., 3])[..., None] * rgb('#b06cff') * .55
            elif not gone and 0 < hit - f <= 3 and sR: full[..., :3] = full[..., :3] * .85 + (comp.radial(sp[0], sp[1], sr * 3.2 + 200, power=1.5) * full[..., 3])[..., None] * rgb('#8a3cf0') * .30
            ht = np.zeros((HH, W), f32)
            if gone:
                h_ = P['h%d' % ti]; ht[oy:oy + h_.shape[0], ox:ox + h_.shape[1]] = h_; age = (f - hit - 1) / 10.; d_ = np.sqrt((x - tx) ** 2 + (y - ty) ** 2); heat = np.exp(-np.clip(d_ - tr, 0, 1e4) / (tr * .09 + 5)) * (d_ > tr - 2); full[..., :3] = full[..., :3] + (heat * full[..., 3])[..., None] * rgb('#b06cff') * (.75 + 1.5 * max(0, 1 - age)) + (np.exp(-np.clip(d_ - tr, 0, 1e4) / (tr * .5)) * (d_ > tr) * full[..., 3])[..., None] * rgb('#3a1c90') * .35
                st = {1: 66, 3: 61, 5: 65}.get(ti)
                if st is not None and f >= st:                        # the floors above the bite let go
                    u = f - st + 1; k_ = tr / 830.
                    if ti == 3: cutl = np.full((HH, W), ty, f32); pm = smooth(1.5, -1.5, y - cutl); an = .0045 * u ** 1.5; mx, my = 1.1 * u ** 1.3, 1.15 * u ** 2.05; piv = ((x0 + x1) / 2, ty - tr * .8)
                    else: cutl = ty - .02 * tr - .80 * (x - tx - tr); pm = smooth(1.5, -1.5, y - cutl) * (x > x0 + min(0., (940. - x0) * .10) - 8); an = -.010 * u ** 1.5; mx, my = -7.5 * u ** 1.55 * k_, 9. * u ** 1.7 * k_; piv = (tx + tr, ty - .02 * tr)
                    pc = full * pm[..., None]; hp = ht * pm; full = full * (1 - pm)[..., None]; ht = ht * (1 - pm)
                    ca, sa = math.cos(an), math.sin(an); pm2 = np.float32([[ca, -sa, piv[0] - ca * piv[0] + sa * piv[1] + mx + dx * par], [sa, ca, piv[1] - sa * piv[0] - ca * piv[1] + my + dy * par]]); pc = cv2.warpAffine(pc, pm2, (W, HH), flags=cv2.INTER_LINEAR); hp = cv2.warpAffine(hp, pm2, (W, HH))
                else: pc = hp = None
            else: pc = hp = None
            if pc is not None and ti == 3: img = self._piece(img, pc, hp)  # falls behind what is left of its own stump
            full = cv2.warpAffine(full, tm, (W, HH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE); img = comp.over(img, full)
            if gone:
                ht = cv2.warpAffine(ht, tm, (W, HH)); hot = max(0., 1 - age); ht = np.clip(ht, 0, 1); img = img * (1 - ht[..., None]) + ht[..., None] * (rgb('#ffffff') * (.90 + .35 * hot)) + comp.blur(ht, 6)[..., None] * rgb('#b06cff') * (.8 + 2.0 * hot)
                if pc is not None and ti != 3: img = self._piece(img, pc, hp)
                img = self._crumbs(img, f - hit - 1, ti, tx + dx * par, ty + dy * par, tr)
            if f == 56 and ti == 0: img = ball(img); done = True
            prev = R_
        img = trail(img, prev, 2100.); img = suction(img, prev * .6, 2100., 9, 40)
        if not done: img = ball(img)
        if fly or f in (64, 65, 66): img = self._chase(img, f, C, R, si, dx, dy)
        if f == 63:                                                    # past the lens
            p = np.array([1900., 1240.]); d = np.sqrt((x - p[0]) ** 2 + (y - p[1]) ** 2); e_ = d / 1430.; inn = smooth(1.015, .985, e_)[..., None]; img = img * (1 - inn) + (rgb('#12032e') * (1 - smooth(.45, 1., e_)[..., None]) + rgb('#7a2cf0') * smooth(.5, .95, e_)[..., None] * .95) * inn + (np.exp(-np.abs(e_ - 1) * 46) * 1.25 + np.exp(-np.abs(e_ - 1.07) * 60) * .5)[..., None] * rgb('#f6eaff'); img = kit.zoomblur(img, p[0], p[1], .035, 41)
        return img

    def _piece(self, img, pc, hp):
        img = comp.over(img, pc); hp = np.clip(hp, 0, 1); return img * (1 - hp[..., None]) + hp[..., None] * .95 + comp.blur(hp, 6)[..., None] * rgb('#b06cff') * .8

    def _ringwall(self, img, f, rho, dx, dy):
        """the shock ring over the city: a wall of dust and broken glass standing on the true ground circle"""
        gx0, hz, Z0, HF, FF = self.RING0
        if f > 66: return img
        a = np.linspace(0, 2 * np.pi, 1081); Z = Z0 + rho * np.sin(a); X = rho * np.cos(a); ok = Z > .34; r = np.random.default_rng(40); n1 = r.normal(0, 1, 1081); n2 = np.random.default_rng(41 + f).normal(0, 1, 1081); c = Cel(); k_ = 1. if f < 63 else max(0., 1 - (f - 62) / 5.); run = np.flatnonzero(ok)
        if len(run) < 6: return img
        for seg in np.split(run, np.flatnonzero(np.diff(run) > 1) + 1):
            if len(seg) < 4: continue
            z_ = Z[seg]; b = np.c_[gx0 + X[seg] / z_ * FF + dx * .25, hz + HF / z_ + dy * .25]; h = (46. / z_ * 2.1 + 5) * (.75 + .20 * n1[seg] + .18 * n2[seg] + .3 * np.sin(a[seg] * 23 + f)) * k_; h = np.clip(h, 2, 130) * (.12 + .88 * smooth(.9, 2.2, z_)) + 4 * np.abs(n2[seg])
            top = b - np.c_[0 * h, h]; c.fill(np.vstack([b + (0, 3), top[::-1]]), '#8a6cd8', .42 * k_); tp2 = b - np.c_[0 * h, h * .55]; c.fill(np.vstack([b + (0, 2), tp2[::-1]]), '#d6c0ff', .55 * k_); c.line(b, 5., '#ffffff', taper=(.01, .01), press=.25, seed=f, alpha=.95 * k_)
            for i in seg[::7]: z = Z[i]; q = np.array([gx0 + X[i] / z * FF + dx * .25, hz + HF / z + dy * .25]); v = np.array([math.cos(a[i]) * 1.2, -.9 - .5 * r.random()]); s_ = (26 + 60 * r.random()) / z ** .8 * k_; c.line(np.array([q, q + v * s_ * .5, q + v * s_]), 1.2 + 2.2 / z, '#e8dcff', taper=(.1, .8), seed=int(i), alpha=.8 * k_)
        L = c.layer(); return comp.over(img + comp.blur(L, 12)[..., :3] * .45, L)

    def _chase(self, img, f, C, R, si, dx, dy):
        """glass and masonry sucked along after the sphere, over everything"""
        c = Cel(); r = np.random.default_rng(77); n = len(C); fade = 1. if f < 63 else max(.0, 1 - (f - 63) / 4.)
        for k in range(90):
            lead = int(r.uniform(2, 60)); i = si - lead if f < 63 else int(n - 1 - lead * (1.0 + .0 * f)); i = int(np.clip(i, 0, n - 1)); a = r.uniform(0, 6.283) + f * .25 * r.uniform(.5, 1.5); v = np.array([math.cos(a), math.sin(a)]); rr = R[i] * r.uniform(1.15, 2.3); p = C[i] + v * rr + (dx * .5, dy * .5); j = int(max(0, i - 9)); p0 = C[j] + v * R[j] * rr / R[i] * 1.25 + (dx * .5, dy * .5); s = R[i] * .05 * r.random() ** 1.6 + 1.2
            if not (-60 < p[0] < W + 60 and -60 < p[1] < HH + 60): continue
            c.line(np.array([p0, (p0 + p) / 2, p]), 1. + s * .55, '#b78cff' if k % 3 else '#f2e8ff', taper=(.85, .1), seed=k, alpha=.7 * fade)
            if s > 2.5: self._rock(c, p, s * (1. if f < 63 else fade), 800 + k, f * .5, '#e0ccff', vel=p - p0, stretch=1.5)
        return comp.over(img, c.layer())

    # ---------------------------------------------------------------- 54-55: the impact, two drawings
    def impact_draw(self, f):
        """54 white paper, black sphere, violet wedges; 55 the same blow turned inside out, black, the sphere white and cracking"""
        INK, VI, PA = '#0a0314', '#7a2cf0', '#fbf6ff'; w = f == 54; c = Cel(); r = np.random.default_rng(f); sx, sy, R0 = (1130., 520., 372.) if w else (1170., 530., 520.)
        c.fill(np.array([(-40, -40), (W + 40, -40), (W + 40, HH + 40), (-40, HH + 40)], float), PA if w else INK)
        def wedges(nn, r0, r1, col, wd, sd, gap=(0, 0)):
            q = np.random.default_rng(sd); a0 = q.uniform(0, 6.283, 9)
            for i in range(nn):
                a = a0[i % 9] + q.normal(0, .17); hw = wd * q.random() ** 2.2 + .0015; s_ = r0 * q.uniform(.95, 1.5); c.fill(np.array([(sx + math.cos(a) * s_, sy + math.sin(a) * s_ * .92), (sx + math.cos(a - hw) * r1, sy + math.sin(a - hw) * r1), (sx + math.cos(a + hw) * r1, sy + math.sin(a + hw) * r1)]), col)
        if w: wedges(90, R0 * 1.12, 1900, VI, .050, 3); wedges(150, R0 * 1.2, 1900, INK, .030, 5); wedges(40, R0 * 1.5, 1900, PA, .016, 8)
        else: wedges(120, R0 * 1.05, 1900, VI, .060, 13); wedges(90, R0 * 1.16, 1900, '#3a0fa0', .05, 15); wedges(80, R0 * 1.1, 1900, PA, .022, 18); wedges(40, R0 * 1.4, 1900, INK, .03, 19)
        # city: a black comb along the bottom, bent away from the blow
        q = np.random.default_rng(4); xx = -60.
        while xx < W + 60:
            bw = q.uniform(40, 130); hh_ = q.uniform(60, 250) * (1 - .55 * math.exp(-((xx - sx) / 420) ** 2)); ln = (xx - sx) * .10; c.fill(np.array([(xx, HH + 40), (xx + ln, HH - hh_), (xx + bw + ln, HH - hh_ * q.uniform(.9, 1.)), (xx + bw, HH + 40)]), INK if w else '#1c0a4a')
            for k in range(int(q.integers(2, 7))): px, py = xx + q.uniform(.15, .8) * bw + ln * .5, HH - hh_ * q.uniform(.15, .85); c.fill(np.array([(px, py), (px + 9, py), (px + 9, py + 13), (px, py + 13)]), VI if w else PA)
            xx += bw + q.uniform(0, 14)
        L = c.layer(); img = np.zeros((HH, W, 3), f32); img = comp.over(img, L)
        # the sphere
        def ragn(sd):                                                  # torn edge: long swells and a few sharp spurs
            q = np.random.default_rng(int(sd * 10)); v = sum(q.normal(0, 1) * np.sin(t * k + q.uniform(0, 6.28)) / k ** .5 for k in (2, 3, 5)); 
            return v * .9 + .35 * np.sin(t * 13 + sd) * np.sin(t * 2 + sd)
        c = Cel(); t = np.linspace(0, 2 * np.pi, 420, endpoint=False); rg = lambda R_, am, sd: np.c_[sx + np.cos(t) * R_ * (1 + am * ragn(sd)), sy + np.sin(t) * R_ * (1 + am * ragn(sd))]
        def swirl(R_, a0, span, wd, col): a = np.linspace(a0, a0 + span, 60); v = np.linspace(0, 1, 60); rr = R_ * (1 - .20 * v); hw = wd * np.sin(np.pi * v) ** .6; o = np.c_[sx + np.cos(a) * (rr + hw), sy + np.sin(a) * (rr + hw)]; i_ = np.c_[sx + np.cos(a) * (rr - hw * .4), sy + np.sin(a) * (rr - hw * .4)]; c.fill(np.vstack([o, i_[::-1]]), col)
        if w:
            fx.ragged_ring(c, sx, sy, R0 * 1.30, R0 * 1.30, R0 * .035, 5, VI, gaps=.4, n=8); fx.ragged_ring(c, sx, sy, R0 * 1.12, R0 * 1.12, R0 * .085, 3, INK, gaps=.06, n=6); c.fill(rg(R0 * 1.06, .006, 2.2), PA); c.fill(rg(R0, .005, 3.), INK); c.clip(rg(R0, .005, 3.))
            for (rr_, a0, sp_, wd, col) in ((.93, 2.9, 2.5, .085, VI), (.90, -.2, 1.5, .050, '#3a0fa0'), (.74, 1.4, 2.1, .075, '#3a0fa0'), (.76, 4.2, 1.9, .055, VI), (.56, 3.2, 2.4, .060, VI), (.52, .2, 1.3, .035, '#b78cff'), (.36, 1.9, 2.2, .030, '#b78cff')): swirl(R0 * rr_, a0, sp_, R0 * wd, col)
            swirl(R0 * .90, 3.5, 1.5, R0 * .040, PA); swirl(R0 * .60, 3.9, .9, R0 * .018, PA)
            for i in range(7):
                a = i / 7. * 6.283 + r.normal(0, .3); b = a + r.uniform(.9, 2.2); p0, p1 = np.array([sx, sy]) + R0 * .98 * np.array([math.cos(a), math.sin(a)]), np.array([sx, sy]) + R0 * r.uniform(.3, .8) * np.array([math.cos(b), math.sin(b)]); v = np.linspace(0, 1, 7)[:, None]; nn = np.array([-(p1 - p0)[1], (p1 - p0)[0]]); c.line(p0 * (1 - v) + p1 * v + nn * (r.normal(0, .09, (7, 1)) * (v > 0) * (v < 1)), r.uniform(5, 13), PA, taper=(.05, .7), seed=i)
            c.unclip()
        else:
            fx.ragged_ring(c, sx, sy, R0 * 1.30, R0 * 1.30, R0 * .05, 15, PA, gaps=.4, n=9); c.fill(rg(R0 * 1.10, .010, 4.), VI); fx.ragged_ring(c, sx, sy, R0 * 1.14, R0 * 1.14, R0 * .07, 13, VI, gaps=.1, n=7); fx.ragged_ring(c, sx, sy, R0 * 1.035, R0 * 1.035, R0 * .05, 14, INK, gaps=.05, n=6); c.fill(rg(R0, .006, 6.), PA); c.clip(rg(R0, .006, 6.))
            for i in range(7): swirl(R0 * (.97 - .10 * i), 2.2 + i * 2.1, 1.9, R0 * .045, ['#d9c2ff', VI, '#b78cff'][i % 3])
            c.fill(SC(sx, sy, R0 * .20, R0 * .52, 15, 7), INK); c.fill(SC(sx, sy, R0 * .10, R0 * .26, 11, 8), VI); c.fill(SC(sx, sy, R0 * .05, R0 * .12, 9, 9), PA); c.unclip()
            for i in range(13):                                        # it cracks the paper
                a = i / 13. * 6.283 + r.normal(0, .12); p = np.array([sx, sy]) + np.array([math.cos(a), math.sin(a)]) * R0 * .28; Q = [p]; d = a
                for k in range(int(r.integers(4, 8))): d = a + r.normal(0, .42); p = p + np.array([math.cos(d), math.sin(d)]) * r.uniform(50, 130); Q.append(p)
                c.line(np.array(Q), r.uniform(6, 17), INK, taper=(.02, .8), seed=i)
        # ink thrown off it
        for i in range(70):
            a = r.uniform(0, 6.283); d0 = R0 * r.uniform(1.12, 1.9); s_ = 3 + 26 * r.random() ** 3; p = np.array([sx + math.cos(a) * d0, sy + math.sin(a) * d0]); q = np.sort(r.uniform(0, 6.283, 7)); u = np.array([math.cos(a), math.sin(a)]); B = np.c_[np.cos(q), np.sin(q)] * s_ * r.uniform(.6, 1.1, (7, 1)); B = B + np.outer(B @ u, u) * r.uniform(.4, 2.4); c.fill(B + p, (INK if i % 3 else VI) if w else (PA if i % 3 else VI))
        img = comp.over(img, c.layer())
        # him: pure shapes
        fig = self.A['fig_rs0' if w else 'fig_rs1']; hh = self.A['fig_rs0_h' if w else 'fig_rs1_h']; sc = 1.16 if w else 1.06; small = cv2.resize(fig, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC); a = np.clip(small[..., 3], 0, 1); g = lum(small[..., :3]) / np.maximum(a, 1e-3); g = cv2.GaussianBlur(g, (0, 0), 1.2); hi = smooth(.30, .36, g)[..., None]; md = smooth(.150, .175, g)[..., None]
        body = (rgb(INK) * (1 - md) + rgb(VI) * md) if w else (rgb(INK) * (1 - md) + rgb('#b78cff') * md); body = body * (1 - hi) + rgb(PA) * hi; hx, hy = (hh[2] - hh[4]) * sc, (hh[3] - hh[5]) * sc; tgt = (sx - R0 * 1.12, sy - 40) if w else (sx - R0 * 1.10, sy - 60); lay = np.zeros((HH, W, 4), f32); paste(lay, np.dstack([body * a[..., None], a]).astype(f32), int(tgt[0] - hx), int(tgt[1] - hy))
        al = (cv2.dilate(lay[..., 3], np.ones((9, 9), np.uint8)) > .4).astype(f32); al = cv2.GaussianBlur(al, (0, 0), 1.)[..., None]; img = img * (1 - al) + rgb(PA if not w else INK) * al if not w else img * (1 - cv2.GaussianBlur((cv2.dilate(lay[..., 3], np.ones((13, 13), np.uint8)) > .4).astype(f32), (0, 0), 1.)[..., None]) + rgb(PA) * cv2.GaussianBlur((cv2.dilate(lay[..., 3], np.ones((13, 13), np.uint8)) > .4).astype(f32), (0, 0), 1.)[..., None]
        return comp.over(img, lay)

    def _crumbs(self, img, t, ti, tx, ty, tr):
        """masonry and glass thrown from the edge of a bite, then falling"""
        c = Cel(); r = np.random.default_rng(900 + ti); k = max(.30, tr / 440.); face = self.TW[ti][4]
        for i in range(int(60 + 60 * k)):
            a = r.uniform(0, 6.283); v0 = r.uniform(6, 34) * (k + .12); u = np.array([math.cos(a), math.sin(a)]); p = np.array([tx, ty]) + u * tr * r.uniform(.98, 1.03) + u * v0 * t * .5 + np.array([r.normal(0, 3) * t * k, 2.1 * (k + .10) * t * t * r.uniform(.7, 1.3)]); s = (max(tr, 130.) * .046 * r.random() ** 2.6 + 1.3)
            if not (-40 < p[0] < W + 40 and -40 < p[1] < HH + 40): continue
            th = np.sort(r.uniform(0, 6.283, 5)) + r.normal(0, .25) * t; Q = p + np.c_[np.cos(th), np.sin(th) * .85] * s * r.uniform(.6, 1.1, (5, 1)); q_ = r.random()
            c.fill(Q, '#efe4ff' if q_ < .22 and s < 5 else ('#0b0722' if q_ < .55 else face))
            if s > 6: c.fill(Q[:3] * .7 + Q.mean(0) * .3, '#5a4cb0'); kit.outline(c, Q, 1.4, '#05030f', i)
            if s > 9 and q_ > .7: c.fill(Q.mean(0) + np.array([(-.3, -.2), (.1, -.25), (.12, .1), (-.28, .12)]) * s, '#f2d59a')
        img = comp.over(img, c.layer())
        if t >= 1 and tr > 60:                                        # dust breathing out of the cut
            d2 = Cel(W // 2, HH // 2)
            for i in range(22): a = r.uniform(.1, 3.0); p = (np.array([tx, ty]) + np.array([math.cos(a), math.sin(a)]) * tr * 1.0) / 2 + (r.normal(0, 8) * k, 3.5 * t * k * r.uniform(.5, 1.5)); rad = tr * .05 * (.4 + .16 * t) * r.uniform(.6, 1.5); qq = np.linspace(0, 6.283, 22); d2.fill(np.c_[p[0] + np.cos(qq) * rad * (1 + .2 * np.sin(qq * 4 + i)), p[1] + np.sin(qq) * rad * (.8 + .2 * np.sin(qq * 3 + i))], '#6a5498', .20)
            dl = cv2.resize(cv2.GaussianBlur(d2.layer(HH // 2, W // 2), (0, 0), 5), (W, HH)); img = comp.over(img, dl, .9 * min(1., t / 3.))
        return img

    def _him(self, img, f, name, size, at, glow=1.):
        fig = self.A['fig_' + name]; hh = self.A['fig_' + name + '_h']; sc = size / 790.; fw, fh = int(fig.shape[1] * sc), int(fig.shape[0] * sc); small = cv2.resize(fig, (fw, fh), interpolation=cv2.INTER_AREA); lay = np.zeros((HH, W, 4), f32); ox, oy = at[0] - (hh[2] - hh[4]) * sc, at[1] - (hh[3] - hh[5]) * sc; paste(lay, small, int(ox), int(oy))
        lay = kit.tint(lay, '#8f78d8', .5); lay = rim(lay, -3, -2, '#f0dcff', .95 * glow, soft=.5); lay = rim(lay, 2, 0, '#7a4cf0', .7, soft=.6); return comp.over(img, lay)

    def frame(self, f):
        if f < 27: return self.wide(f)
        if f < 54: return self.close(f)
        if f < 56: d = kit.shake(f, 16, 8); return comp.cam2d(self.impact_draw(f), W / 2 + d[0], HH / 2 + d[1], 1.06 if f == 54 else 1.10)
        img = self.stage(f)
        t = f - 63; a = {56: 10, 57: 10, 58: 11, 59: 13, 60: 16, 61: 19, 62: 25, 63: 32, 64: 16, 65: 10, 66: 7, 67: 5}.get(f, 1.4); d = kit.shake(f, a, 8); img = comp.cam2d(img, W / 2 + d[0] + (2.4 * max(0, t)), HH / 2 + d[1] + 6, 1.04 + .0035 * max(0, t) + max(0., 2.2 * (abs(d[1]) - 12) / HH))
        if f in (63, 64, 65): img = comp.screen(img, np.ones((1, 1, 3), f32) * rgb('#f3e4ff') * (.08, .10, .03)[f - 63])
        return img
