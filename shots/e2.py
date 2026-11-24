"""e2  NANAMI walks through embers, half burnt; every ember turns into a glint of the sea; he turns his head."""
import numpy as np, cv2, math, os
from jj import W, H, comp
from jj.paint import rgb, f32, smooth, ramp, grid, fbm
from jj.ink import Cel, spl
from jj.comp import lerp_keys, keyed
from jj.kit import shake, para, smear
from chars.ntkit import rim, hexmix, qb
from chars.nanami import Nanami, Body, PAL, EXPR
from chars import nt_keys as K

VER = 5


class Shot:
    id = 'e2'; n = 72
    finish = dict(bloom=.46, bloom_thr=.70, vig=.12, halation='#ffcf9c', soft=.18)
    EVENTS = [(0, 'fire_bed', dict(size=.7, until=40, note='embers crackling, low burning hall, air roar; ducks under the light at 30')),
              (9, 'step_heavy', dict(size=1.0, note='dragging leather step on rubble, cloth, one breath')), (12, 'whoosh', dict(size=.35, note='dark pillar passes behind him left to right, 12-20')),
              (27, 'step_heavy', dict(size=1.15, note='second, slower, heavier step; he stops')), (30, 'light_open', dict(size=1.0, until=42, note='light eats outward from the burn line: fire sound thins into surf, rising airy swell, hundreds of tiny glass glints')),
              (42, 'sea', dict(size=.9, until=72, note='white opens on the beach: one wave drawing back over sand, warm wind, shirt flapping softly, far gulls')), (54, 'cloth', dict(size=.3, note='he turns his head, collar and sleeve rustle (54-64)')), (66, 'hush', dict(size=.3, note='wave recedes, almost silence'))]

    def setup(self):
        pf = lambda k: f'cache/e2_v{VER}_{k}.npy'
        if not all(os.path.exists(pf(k)) for k in ('em', 'beb', 'close')):
            np.save(pf('em'), comp.bokeh(K.ember_plate(6, 2560).astype(f32), 9, 1.0).astype(np.float16)); be = K.beach_plate().astype(f32); np.save(pf('beb'), comp.bokeh(be, 4.2, 1.0).astype(np.float16))
            q = cv2.resize(be[600:840, 60:1100], (W, H), interpolation=cv2.INTER_CUBIC); np.save(pf('close'), np.clip(comp.bokeh(cv2.GaussianBlur(q, (0, 0), 5), 12, 0) * .62 + .40, 0, 1.15).astype(np.float16)); K._C.clear(); del be, q
        self.em, self.beb, self.close = [np.load(pf(k), mmap_mode='r') for k in ('em', 'beb', 'close')]; self.x, self.y = grid()
        r = np.random.default_rng(77); self.P = [dict(x=r.uniform(-80, W + 80), y=r.uniform(-60, H + 60), v=r.uniform(.5, 1.6), ph=r.uniform(0, 6.28), R=.8 + 3.6 * r.random() ** 2.4, hot=r.random()) for _ in range(300)]
        self.nz = K.fb((150, 52, 18), (1, .55, .25), 40); self.vg = 1 - .30 * np.clip(np.sqrt(((self.x - W / 2) / (W * .62)) ** 2 + ((self.y - H / 2) / (H * .68)) ** 2) - .55, 0, 1)[..., None] ** 1.3

    # ------------------------------------------------------------------ particles: one list, two lives
    def parts(self, f, inside=None, drift=1.0):
        em = np.zeros((H, W, 3), f32); g = Cel(); n_g = 0
        for k, q in enumerate(self.P):
            px = (q['x'] + 80 + f * (4.4 * q['v']) * drift + 14 * math.sin(f * .11 * q['v'] + q['ph'])) % (W + 160) - 80; py = (q['y'] + 60 - f * 2.3 * q['v'] + 8 * math.sin(f * .17 + q['ph'] * 2)) % (H + 120) - 60
            ix, iy = int(np.clip(px, 0, W - 1)), int(np.clip(py, 0, H - 1)); tw = .55 + .45 * math.sin(f * .6 * q['v'] + q['ph'])
            if inside is not None and inside[iy, ix] > .5:
                R = (3.2 + q['R'] * 3.4) * (.55 + .6 * tw); K.sparkle(g, px, py, R * (1.5 if q['hot'] > .8 else 1.), '#ffffff', 0., .15, .55 + .45 * tw); n_g += 1
                if q['hot'] > .55: g.line([(px - R * 2.4, py), (px, py + .4), (px + R * 2.4, py)], 1.6, '#fffbe8', taper=(.5, .5), alpha=.7 * tw)
            else:
                c = rgb('#ff5a14') * (1 - q['hot']) + rgb('#ffe0a0' if q['hot'] > .6 else '#ff9a3a') * q['hot']; L = q['R'] * (.8 + 2.2 * ((k * 37) % 10) / 10); d = np.array([4.4 * q['v'], -2.3]); d /= np.hypot(*d)
                a, b = np.array([px, py]) - d * L, np.array([px, py]) + d * L; cv2.line(em, tuple(int(v * 16) for v in a), tuple(int(v * 16) for v in b), tuple(float(v) for v in c * (.6 + .8 * tw)), max(1, int(q['R'] * 1.3)), cv2.LINE_AA, 4)
        return em, (g.layer() if n_g else None)

    # ------------------------------------------------------------------ 0-42
    WALK = [(0, (1180, 446, -.050, .03)), (5, (1166, 430, -.030, .06)), (9, (1146, 476, -.098, -.10)), (13, (1134, 462, -.078, -.06)), (19, (1112, 428, -.036, .08)), (24, (1094, 444, -.060, .0)), (27, (1074, 486, -.110, -.11)), (31, (1066, 470, -.086, -.05)), (36, (1062, 456, -.052, .0)), (42, (1060, 450, -.030, .0))]

    def walk(self, f):
        d = f // 2 * 2; v = lerp_keys(d, self.WALK); v0 = lerp_keys(max(0, d - 3), self.WALK)
        lag = float(np.clip((v[1] - v0[1]) * .0022, -.10, .10)); return v, lag, d

    def ember_frame(self, f):
        v, lag, d = self.walk(f); cx, cy, rl, arm = v; s = 318
        pan = lerp_keys(f, [(0, 330), (9, 286), (12, 274), (24, 226), (28, 210), (42, 192)]); sx, sy = shake(f, 5, 3, 3.0, 9) + shake(f, 7, 5, 3.5, 27)
        fl = 1 + .05 * math.sin(f * 1.7) + .04 * math.sin(f * 2.9 + 1)
        bg = comp.cam2d(np.array(self.em, f32), W / 2 + pan - 40 + sx * .5, H / 2 + sy * .5 + (cy - 450) * .10, 1.015, border=cv2.BORDER_REFLECT_101) * fl
        r_ = K.embers(f, 70, 5, 8.0, drift=(5.4, -1.6)); bg = bg + qb(r_, 7) + comp.bokeh(K.embers(f, 50, 15, 5.0, 0, drift=(4.6, -1.8)), 7, 0) * 1.1
        if 11 <= f <= 21:                                        # a pillar goes by between him and the fire
            px = -330 + (f - 10.5) * 250; c = Cel(); c.fill(np.array([(px - 120, -20), (px + 140, -20), (px + 124, 420), (px + 148, 1100), (px - 138, 1100), (px - 112, 500)]), '#0c0507'); bg = comp.over(bg, smear(cv2.GaussianBlur(c.layer(), (0, 0), 9), 90, 0, 7), .96)
        e = dict(EXPR['far'], open=.50 if f < 30 else float(lerp_keys(d, [(30, .50), (36, .34)])))
        pit = float(lerp_keys(d, [(0, -.06), (9, -.10), (19, -.04), (27, -.12), (34, -.05), (42, .03)]))
        lay, Z, H_ = K.prof_figure(cx + sx, cy + sy, s, pit, rl, d, arm=arm, lagx=lag, wind=.035, loose=.7)
        lay, m, gl = K.burnt(lay, Z, cx, d, 0., H_, s)
        img = comp.over(comp.over(bg, lay), K.ash_flakes(d, cx, cy, s, sp=7.)) + gl
        # smoke leaving the shoulder
        sm = np.zeros((H, W), f32); r = np.random.default_rng(4)
        for k in range(9):
            t = ((f * .022 + k / 9.) % 1.); px = cx + 150 + 60 * k % 260 + t * 380; py = cy + 420 - t * 520 - 22 * math.sin(k * 2.2 + f * .12); cv2.ellipse(sm, (int(px), int(py)), (int(40 + 90 * t), int(20 + 46 * t)), -32, 0, 360, float(.18 * math.sin(math.pi * t)), -1, cv2.LINE_AA)
        sm = qb(sm, 22)[..., None]; img = img * (1 - sm) + sm * rgb('#3a1a14')
        return img, Z, H_, (cx, cy, s)

    def frame(self, f):
        x, y = self.x, self.y
        if f < 42:
            img, Z, H_, (cx, cy, s) = self.ember_frame(f); inside = None; R = 0
            if f >= 30:
                R = float(lerp_keys(f, [(30, 0), (31, 16), (32, 42), (33, 88), (34, 165), (35, 290), (36, 490), (37, 820), (38, 1320), (39, 2300)], lambda t: t))
                B = np.array(K.BURN[:-2], float); ln = np.ones((H, W), np.uint8); cv2.polylines(ln, [np.round(Z(spl(B, n=5)) * 8).astype(np.int32)], False, 0, 3, cv2.LINE_8, 3)
                D = cv2.distanceTransform(ln, cv2.DIST_L2, 5) + self.nz * min(90., 12 + R * .22) - R
                inside = smooth(5, -5, D); k = float(lerp_keys(f, [(30, .10), (35, .28), (38, .70), (40, 1.)]))
                sea = np.array(self.close, f32) * (1 - k) + rgb('#fffdf6') * k * 1.06
                img = img * (1 - inside[..., None]) + sea * inside[..., None]
                out = np.clip(D, 0, None); e1 = np.exp(-(D / 5.) ** 2)[..., None] * rgb('#fffdf2') * 1.1 + (np.exp(-out / 26.) * (D > -4))[..., None] * rgb('#ffdf9c') * .85 + (np.exp(-out / 150.) * (D > 0))[..., None] * rgb('#ffcf86') * .30 * min(1., R / 120.)
                img = img + e1
            if f >= 40: return (np.ones((H, W, 3), f32) * rgb('#fffdf6') * 1.06).astype(f32)                       # two frames of nothing but light
            em, gl = self.parts(f, inside)
            img = img + em * 1.15 + K.embers(f, 30, 9, 4.2, .6, drift=(6.2, -2.0), box=(900, -80, W + 80, H + 80)) * 1.2 * (0 if f > 37 else 1)
            if gl is not None: img = comp.over(img, gl) + qb(gl[..., :3], 6) * .5
            if f < 30: img = img + (K.radial_(x, y, 120, 1000, 900, 620, 2.) * .22)[..., None] * rgb('#ff8a3c')
            return (img * (self.vg if f < 34 else 1 - (1 - self.vg) * max(0., 1 - (f - 34) / 5.))).astype(f32)
        return self.beach_frame(f)

    # ------------------------------------------------------------------ 42-72
    def chair(self, f, z):
        """foreground, out of focus: a low table edge with the books he never had time for, one open with a lifting page; palm fronds above"""
        d3 = f // 3 * 3; c = Cel(); X, Y = -1.0 * (f - 42), .45 * (f - 42); P = lambda L_: np.array([(X + u, Y + v) for u, v in L_])
        c.fill(P([(-40, 1012), (520, 984), (566, 1000), (548, 1016), (-40, 1052)]), '#c79a6c'); c.fill(P([(-40, 1052), (548, 1016), (540, 1034), (-40, 1076)]), '#8c6246'); c.fill(P([(380, 1030), (424, 1027), (404, 1120), (360, 1120)]), '#7c563e'); c.fill(P([(-40, 1020), (300, 1003), (520, 992), (470, 1002), (-40, 1030)]), '#e9c79a')
        for j, (col, sd, x0, w_, h_) in enumerate((('#b4523f', '#7d342c', 60, 236, 34), ('#34607f', '#213f58', 74, 214, 30), ('#d9b25e', '#a37e3a', 66, 222, 32))):
            y0 = 1010 - 32 * j - 4 * j; c.fill(P([(x0, y0 - h_), (x0 + w_, y0 - h_ - 12), (x0 + w_ + 4, y0 - 12), (x0 + 2, y0)]), col); c.fill(P([(x0 + w_ * .62, y0 - h_ - 7), (x0 + w_, y0 - h_ - 12), (x0 + w_ + 4, y0 - 12), (x0 + w_ * .60, y0 - 7)]), sd); c.fill(P([(x0 + 8, y0 - h_ + 7), (x0 + w_ * .58, y0 - h_), (x0 + w_ * .58, y0 - 12), (x0 + 10, y0 - 6)]), '#fbf4e2', .75)
        # the open one
        fl = math.sin(d3 * .62); c.fill(P(spl([(318, 988), (368, 958), (430, 972), (428, 996), (372, 990), (322, 1010)], True, corners=(0, 2, 3, 5), n=6)), '#fffaf0'); c.fill(P(spl([(430, 972), (486, 944), (540, 962), (538, 984), (490, 974), (428, 996)], True, corners=(0, 2, 3, 5), n=6)), '#efe6d2'); c.fill(P(spl([(430, 972), (444 - 10 * fl, 918 - 10 * fl), (492 + 16 * fl, 884 - 18 * fl), (500 + 14 * fl, 898 - 14 * fl), (462, 938), (452, 968)], True, corners=(0, 2, 3), n=7)), '#ffffff')
        sw = .05 * math.sin(d3 * .20); g2 = Cel(); K.frond(g2, 2010, -150, 2.36, 820, n=20, w=86, curl=-.5, seed=2, col='#9cc79a', dark='#5f9c78', sway=sw); K.frond(g2, 2050, -40, 2.80, 640, n=18, w=74, curl=-.55, seed=8, col='#a9d0a4', dark='#6ba584', sway=sw * 1.3)
        a_ = comp.bokeh(c.layer(), 7.5, 1.0); b_ = comp.bokeh(g2.layer(), 12, 1.0); return a_ * (1 - b_[..., 3:4]) + b_

    YAW = [(42, -2.66), (54, -2.66), (56, -2.74), (58, -2.20), (60, -1.60), (62, -1.24), (64, -.94), (66, -.78), (68, -.80), (72, -.81)]

    def beach_frame(self, f):
        x, y = self.x, self.y; d = f // 2 * 2; z = 1.07 + .0016 * (f - 42)
        bg = comp.cam2d(np.array(self.beb, f32), W / 2 - 30 + (f - 42) * .5, H / 2 - 22, z)
        g = Cel(); r = np.random.default_rng(50 + d)                                                     # the sea goes on glittering: new glints every drawing
        for _ in range(70):
            px = abs(r.normal(0, 560)) + r.uniform(-60, 120); py = 616 + abs(r.normal(0, 88))
            if py > 836: continue
            sc = .4 + 1.2 * (py - 600) / 240.; tw = r.random(); K.sparkle(g, px, py, (4 + 13 * tw ** 3) * sc, '#ffffff', 0, .13, .4 + .6 * tw); g.line([(px - 16 * sc * tw, py), (px, py), (px + 16 * sc * tw, py)], 1.6, '#ffffff', taper=(.5, .5), alpha=.5)
        for gx_, gy_, gs, ga in ((330, 236, 26, .16), (470, 300, 17, -.10), (236, 330, 12, .05)):
            fl_ = .35 + .28 * math.sin(d * .55 + gx_); gx2 = gx_ + (f - 42) * 1.3
            for q in (-1, 1): g.line(spl([(gx2, gy_), (gx2 + q * gs * .55, gy_ - gs * fl_), (gx2 + q * gs * 1.25, gy_ - gs * fl_ * .35 + q * ga * gs)], n=10), gs * .17, '#86aecb', taper=(.15, .75), press=.2)
        gl = g.layer(); bg = comp.over(bg, gl) + qb(gl[..., :3] * (gl[..., 3:4] > 0), 7) * .30
        bg = bg + (K.radial_(x, y, -60, 60, 1250, 1000, 2.2) * .34)[..., None] * rgb('#fff6e2')
        # ---- him, from behind
        s = 116 * z / 1.07; cx, cy = 1224, 404 + 3 * math.sin(d * .20); yaw = float(lerp_keys(d, self.YAW, lambda t: t)); tn = float(np.clip((yaw + 2.66) / 1.8, 0, 1))
        pal = dict(PAL['beach'], blown=1, suit='#efe6d4', suitS='#cfc6c0', suitD='#b0a8b4'); lw = s / 300. * 7.0; d3 = f // 3 * 3; gust = .90 + .25 * math.sin((d3 - 42) * .20 + .4)
        c = Cel(); rl = -.02 - .05 * tn; H_ = Nanami.pose(cx - 10 * tn, cy + 6 * tn, s, yaw, .02 + .015 * tn, rl); yl = float(lerp_keys(max(42, d - 4), self.YAW, lambda t: t)); tb = float(comp.ease(np.clip((yl + 2.66) / 1.75, 0, 1))); yl2 = float(lerp_keys(max(42, d - 7), self.YAW, lambda t: t)); tc = float(comp.ease(np.clip((yl2 + 2.66) / 1.75, 0, 1)))                # shoulders follow the head by four frames, cloth by seven
        b = Body(cx + 4 - 7 * tb, cy + 1.27 * s + 3 * tb, s, yaw=.08 + .25 * tb, roll=.018 - .060 * tb, persp=.15)
        e = dict(EXPR['calm' if f < 66 else 'gentle'], gaze=(.75, .0), wet=1, asym=1.); e['open'] = .68; kw = dict(f=d, wind=.055 * gust, seed=3, lit=-1, lw=lw, loose=1.0, wdir=(1., .12, 0.))
        if abs(yaw) < 1.75: Nanami.head(c, H_, e, pal, part='back', **kw)
        else: Nanami.head_back(c, H_, pal, lw, d, .055 * gust, 3, (1., .12, 0.), 1.0, -1, part='face')
        Nanami.draw_back(c, H_, b, pal, lw, d3, gust, 3, -1, tc)
        if abs(yaw) < 1.75: Nanami.head(c, H_, e, pal, part='front', eat=.5, **kw)
        else: Nanami.head_back(c, H_, pal, lw, d, .055 * gust, 3, (1., .12, 0.), 1.0, -1, part='hair')
        lay = c.layer(); lay = rim(lay, 4, 1, '#fffaf0', .75, .5)
        lay = para(lay, (cx - 420, 260), (cx + 120, 640), [(0, '#6a5c44'), (.5, '#241d14'), (1, '#000000')], 'screen', .9); lay = para(lay, (cx - 60, 520), (cx + 320, 1080), [(0, '#ffffff'), (1, '#cfdcf6')], 'mul', .9)
        if f in (58, 60): lay = smear(lay, 16 if f == 58 else 10, 0, 7) * .35 + lay * .65
        img = comp.over(bg + (qb(lay[..., 3], 24) * .08)[..., None] * rgb('#fff4e0'), lay)
        img = comp.over(img, self.chair(f, z), .97)
        img = img + (K.radial_(x, y, -60, 100, 1200, 1000, 2.4) * .20)[..., None] * rgb('#fff4de')
        ex = float(lerp_keys(f, [(42, 1.), (44, .78), (48, .34), (53, .0)]))                              # the white settles into daylight
        if ex > 0:
            em, gp = self.parts(f, np.ones((H, W), f32), .35); img = img * (1 - ex * .80) + rgb('#fffdf6') * ex * .86
            if gp is not None: img = comp.over(img, gp, ex)
        return img.astype(f32)
