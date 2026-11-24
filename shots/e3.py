"""e3  He says two words we do not hear, smiles, the light takes him.  Cut: his goggles fall through black water; in the cracked lens Yuji's eyes overflow."""
import numpy as np, cv2, math, os
from jj import W, H, comp
from jj.paint import rgb, f32, smooth, ramp, grid, fbm
from jj.ink import Cel, spl
from jj.comp import lerp_keys, keyed
from jj.kit import para, smear, flare
from chars.ntkit import rim, hexmix, qb
from chars.nanami import Nanami, Body, PAL, EXPR
from chars.nt_props import goggles3d
from chars.yuji import Yuji, YHead
from chars import nt_keys as K

VER = 4
MOUTH = [(0, 'rest'), (6, 'a'), (9, 'to'), (12, 'rest'), (15, 'ta'), (18, 'no'), (21, 'mu'), (24, 'rest'), (27, 'smile')]


class Shot:
    id = 'e3'; n = 72
    finish = dict(bloom=.40, bloom_thr=.78, vig=.10, halation='#fff0dc', soft=.20)
    EVENTS = [(0, 'sea', dict(size=.6, until=36, note='wind and one slow wave, getting quieter; all ambience is gone by 34')), (6, 'silence_word', dict(size=.2, note='a-to: lips only, frames 6-12, no voice; the sea ducks under it')), (15, 'silence_word', dict(size=.2, note='ta-no-mu: frames 15-24, no voice')),
              (27, 'smile', dict(size=.4, note='he smiles; a single held high tone')), (30, 'light_open', dict(size=.6, until=36, note='light swallowing him, airy, no hit')), (36, 'water_cut', dict(size=1.0, note='HARD cut to black water: everything muffled, deep pressure, sub drone')),
              (37, 'bubbles', dict(size=.5, until=56, note='small bubbles rising off the goggles')), (44, 'glint', dict(size=.5, note='cracked lens catches the light: thin glass ting under water')), (46, 'push', dict(size=.4, until=58, note='camera sinks toward the lens, slow heartbeat on 54 and 63; his eyes show in the lens 43, clear by 48')),
              (58, 'tear', dict(size=.6, note='the tear lets go of the eye in the reflection, falls 58-70')), (66, 'drop', dict(size=.7, note='it crosses the rim and leaves the reflection: one clear drop note with long tail'))]

    def setup(self):
        p = f'cache/e3_v{VER}_be.npy'
        if not os.path.exists(p): np.save(p, comp.bokeh(K.beach_plate().astype(f32), 8.5, 1.0).astype(np.float16)); K._C.clear()
        self.be = np.load(p, mmap_mode='r')
        self.x, self.y = grid(); self.nz = K.fb((210, 80, 30), (1, .5, .2), 23); self.nz2 = K.fb((260, 90), (1, .5), 31)

    def frame(self, f): return self.face(f) if f < 36 else self.water(f)

    # ------------------------------------------------------------------ 0-36
    def face(self, f):
        x, y = self.x, self.y; d = f // 2 * 2; t3 = f // 3 * 3; z = 1 + .0011 * f
        s, cx, cy = 372 * z, 1148 - .5 * f, 486 + .2 * f
        bg = comp.cam2d(np.array(self.be, f32), W / 2 + 40 - .4 * f, H / 2 - 20, 1.16 * z)
        g = Cel(); r = np.random.default_rng(90 + d)
        for _ in range(54):
            px = abs(r.normal(0, 470)) + r.uniform(-60, 100); py = 640 + abs(r.normal(0, 100))
            if py > 900: continue
            sc = .6 + 1.5 * (py - 620) / 260.; cv_ = r.random(); R = (5 + 13 * cv_ ** 2) * sc; g.fill(spl([(px + R * math.cos(a), py + R * .92 * math.sin(a)) for a in np.linspace(0, 6.283, 9)[:-1]], True, n=4), '#ffffff', .10 + .34 * cv_)
        bg = comp.over(bg, g.layer()); bg = bg + (K.radial_(x, y, -80, 120, 1300, 1150, 2.2) * .28)[..., None] * rgb('#fff6e2')
        # ---- acting
        sm = float(lerp_keys(t3, [(21, 0.), (24, .15), (27, .55), (30, .85), (33, 1.)])); mouth = keyed(f, MOUTH)
        bl = float(keyed(f, [(0, 1.), (12, .62), (13, .30), (14, .55), (15, .95), (16, 1.)]))
        a_, b_ = dict(EXPR['calm']), dict(EXPR['gentle']); e = {k: (a_[k] * (1 - sm) + b_[k] * sm if isinstance(a_.get(k), (int, float)) else b_.get(k, a_.get(k))) for k in b_ if k not in ('gaze', 'mouth')}
        e.update(open=(.69 * (1 - sm) + .60 * sm) * bl, low=.10 + .13 * sm, crease=sm, brow_in=.022 + .014 * sm + (.008 if mouth in ('ta', 'a') else 0), gaze=(.56, -.02), mouth=mouth, wet=.8 + .7 * sm, asym=1., lockshade=1, bags=1.)
        yaw = float(lerp_keys(d, [(0, -.35), (14, -.33), (20, -.30), (36, -.285)])); pit = float(lerp_keys(d, [(0, .030), (6, .020), (10, .030), (15, .016), (21, -.012), (26, .014), (34, .026)])); rl = float(lerp_keys(d, [(0, -.030), (15, -.036), (21, -.050), (30, -.062), (36, -.066)]))
        eat = float(lerp_keys(f, [(0, .70), (14, .82), (34, 1.)])); gust = .05 + .018 * math.sin(d * .23)
        pal = dict(PAL['beach'], blown=1); lw = s / 300 * 4.8; c = Cel()
        H_ = Nanami.pose(cx, cy + 3 * math.sin(d * .16), s, yaw, pit, rl); b = Body.under(H_, .28, .03, .02); kw = dict(f=d, wind=gust, seed=3, lit=-1, lw=lw, loose=1.2, wdir=(1., .10, 0.))
        Nanami.head(c, H_, e, pal, part='back', **kw); Nanami.draw_shirt(c, H_, b, pal, lw, 3, -1, d, .42); Nanami.head(c, H_, e, pal, part='front', eat=eat, **kw)
        lay = c.layer(); lay = rim(lay, 5, 2, '#fff7e6', .70, .5); lay = rim(lay, 2, 4, '#fff3d6', .45, .5)
        lay = para(lay, (cx - 640, 200), (cx + 190, 560), [(0, '#9a8a6c'), (.45, '#3a2f21'), (1, '#000000')], 'screen', .95); lay = para(lay, (cx - 100, 560), (cx + 520, 1080), [(0, '#ffffff'), (1, '#d2ddf6')], 'mul', .8)
        # bounce of the white shirt under the jaw
        lay = K.para_lit(lay, (cx - 60, cy + 1.5 * s), (cx - 10, cy + .62 * s), [(0, '#5a4c40'), (1, '#000000')], .8, .25, .6)
        img = comp.over(bg + (qb(lay[..., 3], 26) * .10)[..., None] * rgb('#fff4e0'), lay)
        # ---- the light comes round him: a broad soft front with a warm lip, led by the sea side
        k = float(lerp_keys(f, [(0, -1.), (16, -.45), (24, .10), (30, .55), (35, 1.02)], lambda t: t)); X = 340 + 660 * k
        D = (x - X) + (y - 520) * .16 + self.nz * 46; wh = smooth(230, -260, D)[..., None]; lip = np.exp(-((D - 90) / 120.) ** 2)[..., None]
        img = img + lip * rgb('#ffe9c2') * (.10 + .10 * max(0, k)); img = img * (1 - wh * .97) + rgb('#fffdf7') * wh * 1.04
        img = img + (K.radial_(x, y, -60, 160, 1250, 1100, 2.2) * (.10 + .16 * max(0, k)))[..., None] * rgb('#fff4de')
        mo = np.zeros((H, W, 3), f32); r = np.random.default_rng(8)
        for j in range(40):
            px, py = (r.uniform(-60, W + 60) + f * r.uniform(.8, 2.0)) % (W + 120) - 60, (r.uniform(0, H) - f * r.uniform(.3, .9)) % H; R = r.uniform(1.2, 3.2) if r.random() < .75 else r.uniform(6, 15); a = r.uniform(.25, .8) if R < 4 else r.uniform(.06, .13); cv2.circle(mo, (int(px * 8), int(py * 8)), int(R * 8), tuple(float(v) for v in rgb('#fff6df') * a), -1, cv2.LINE_AA, 3)
        img = comp.screen(img, cv2.GaussianBlur(mo, (0, 0), 1.0))
        return img.astype(f32)

    # ------------------------------------------------------------------ 36-72
    def water(self, f):
        x, y = self.x, self.y; t = f - 36; d = f // 2 * 2
        u = np.clip(y / H, 0, 1); img = ramp(u, [(0, '#0c3440'), (.22, '#072029'), (.55, '#030e15'), (1, '#01040a')])
        # shafts from a surface far above, swaying
        for k, (x0, w0, a, sp, am) in enumerate([(420, 70, .22, .021, .34), (760, 150, .16, .013, .26), (1120, 60, .20, .027, .30), (1500, 190, .12, .010, .20)]):
            sh = (x - (x0 + 26 * math.sin(f * sp * 3 + k) + (y + 60) * (a + .03 * math.sin(f * sp * 2 + k * 2)))) / (w0 * (.6 + .9 * u)); m = np.exp(-sh ** 2) * np.clip(1 - u * 1.15, 0, 1) ** 1.4 * am * (1 + .22 * self.nz2); img = img + m[..., None] * rgb('#3fb3b8')
        # the fall, then the lens
        ez = lambda a, b: float(comp.ease(np.clip((f - a) / float(b - a), 0, 1))); p = ez(46, 58); q = ez(36, 62)
        s = 820 * (1 - p) + 2080 * p + (5 * min(t, 22) + .9 * max(0, t - 22)) * p; yaw = float(lerp_keys(f, [(36, .62), (45, .0), (55, -.20), (72, -.15)])); pitch = float(lerp_keys(f, [(36, .66), (46, .16), (57, -.02), (72, .02)])); roll = float(lerp_keys(f, [(36, -.50), (44, -.18), (54, .10), (62, .06), (72, .085)]))
        gx, gy = 905 + 44 * math.sin(t * .16), 250 + 13.5 * t + 12 * math.sin(t * .32)
        # with p = 1 the cracked lens (x = +.305) sits in the middle of the frame
        c_, s_ = math.cos(roll), math.sin(roll); lx, ly = .305 * math.cos(yaw) * s, 0.; tx, ty = 1010 - (lx * c_), 436 - (lx * s_); cx, cy = gx * (1 - p) + tx * p, min(gy, 600) * (1 - p) + (ty + 6 * math.sin(t * .2)) * p
        pal = dict(PAL['night'], lens='#0e3b3a', lensD='#03100f', lensH='#8fe6dc', frame='#0b1116', line='#01040a', sline='#01040a'); lw = 4.6 + 6.5 * p
        rf = float(lerp_keys(f, [(42, 0.), (43, .22), (44, .36), (45, .52), (46, .70), (47, .88), (48, 1.)], lambda v: v)); cry = float(lerp_keys(f, [(46, .0), (52, .35), (58, .8), (64, 1.)]))
        cen = {}
        def fill(cel, side, Lm):
            c0 = Lm([(side * .305, -.018)])[0]; cen[side] = (c0, Lm)
            if side < 0 or rf <= 0: return
            ax = Lm([(side * .305 - .1, -.018), (side * .305 + .1, -.018)]); sx_ = np.hypot(*(ax[1] - ax[0])) / .2
            yh = YHead().pose(c0[0] + 6, c0[1] - .030 * s, sx_ * .365, .06, -.03, roll + .02); cen['y'] = yh; X = Yuji.expr('grief', well=.5 + .6 * cry, tears=max(0., cry * 1.25 - .05), under=.7, glint=1.2)
            cel.group(); Yuji.head(cel, yh, X, 'cursed', f=d, wind=.1, seed=2, wet=.1); cel.ungroup(rf * .92)
            cel.fill(Lm([(side * .05, .25), (side * .62, .25), (side * .62, -.25), (side * .05, -.25)]), '#0a4a46', .24 * rf); cel.fill(Lm([(side * .05, .25), (side * .62, .25), (side * .60, .070), (side * .42, .088), (side * .22, .078), (side * .06, .100)]), '#020c0c', .38 * rf)
            # the overflow itself: bright ribbons that find their way down the cheek
            for q_, x0, dl in ((-1, -.300, 0.), (-1, -.392, .12), (1, .300, .05), (1, .200, .22)):
                g_ = np.clip((cry - .18 - dl) / .6, 0, 1)
                if g_ <= 0: continue
                ys = np.linspace(-.088, -.088 - .62 * g_, 12); wv = .010 * np.sin(ys * 16 + q_ + d * .25) + (x0 + .035 * np.sign(x0) * np.clip((-.1 - ys) / .5, 0, 1)); wd = (.018 + .010 * np.sin(ys * 11 + x0 * 9)) * (1. if dl < .1 else .6) * np.interp(np.linspace(0, 1, 12), [0, .12, .85, 1], [.5, 1, 1, .75])
                A_, B_ = yh.Pm(np.c_[wv - wd, ys]), yh.Pm(np.c_[wv + wd, ys]); cel.fill(np.vstack([A_, B_[::-1]]), '#8fe0dc', .62 * rf); cel.fill(np.vstack([A_, (A_ * .62 + B_ * .38)[::-1]]), '#eafffb', .80 * rf); cel.line(B_, 2.2, '#0b3a44', taper=(.2, .2), alpha=.7 * rf)
                p_ = (A_[-1] + B_[-1]) / 2; cel.fill(spl([p_ + [-.020 * yh.s, -.01 * yh.s], p_ + [0, -.03 * yh.s], p_ + [.020 * yh.s, -.01 * yh.s], p_ + [.012 * yh.s, .022 * yh.s], p_ + [-.012 * yh.s, .022 * yh.s]], True, n=5), '#dffff9', .9 * rf)
        c = Cel(); goggles3d(c, cx, cy, s, yaw=yaw, pitch=pitch, roll=roll, pal=pal, lw=lw, crack=(False, True), lens_fill=fill, seed=0, glass=.50 * (1 - .55 * rf))
        lay = c.layer(); lay = rim(lay, 3 + 4 * p, 6 + 6 * p, '#8fe8ee', .85, .7); lay = rim(lay, -3 - 2 * p, -2, '#1c5f74', .6, .8)
        lay = para(lay, (cx, cy - .5 * s), (cx + .1 * s, cy + .4 * s), [(0, '#ffffff'), (.5, '#b8d4d8'), (1, '#3c5a68')], 'mul', .9)
        # bubbles behind and in front
        def bub(n, seed, amp, far):
            bc = Cel(); r = np.random.default_rng(seed)
            for k in range(n):
                ph = r.uniform(0, 6.283); v = r.uniform(5, 13) * (1.4 if not far else .8); born = r.uniform(34, 64); tt = f - born
                if tt < 0 or tt > 30: continue
                bx = cx + r.normal(0, .30) * s * (1 - .5 * p) + 10 * math.sin(tt * .5 + ph) * amp; by = cy + r.normal(0, .06) * s - v * tt - .10 * tt * tt; R = r.uniform(2.2, 9) * (1 + .7 * p) * min(1, tt / 3 + .3)
                an = np.linspace(0, 6.283, 11)[:-1]; w_ = 1 + .10 * np.sin(an * 2 + tt * .8 + ph); P = spl(np.c_[bx + R * 1.06 * w_ * np.cos(an), by + R * .92 * w_ * np.sin(an)], True, n=5)
                bc.fill(P, '#2a8f98', .22); bc.line(np.vstack([P, P[:1]]), max(1.2, R * .16), '#9fe8ea', taper=(.2, .2), press=.5, seed=k); bc.fill(spl([(bx - R * .5, by - R * .25), (bx - R * .2, by - R * .55), (bx + R * .05, by - R * .6), (bx - R * .25, by - R * .25)], True, n=4), '#eafffb', .9)
            return bc.layer()
        img = comp.over(img, cv2.GaussianBlur(bub(22, 5, 1., True), (0, 0), 2.2), .7)
        img = img * (1 - .55 * qb(lay[..., 3], 30)[..., None]); img = comp.over(img, lay)
        img = comp.over(img, bub(16, 9, 1.5, False))
        # marine snow
        sn = np.zeros((H, W, 3), f32); r = np.random.default_rng(12)
        for k in range(120):
            dz = r.random(); px = (r.uniform(0, W) + math.sin(f * .05 + k) * 12 - 2 * p * t * (dz + .2)) % W; py = (r.uniform(0, H) - t * (1.2 + 5 * dz)) % H; R = .7 + 2.6 * dz ** 3; cv2.circle(sn, (int(px * 8), int(py * 8)), int(R * 8), tuple(float(v) for v in rgb('#8fd8d4') * (.10 + .35 * r.random()) * (1 - .6 * py / H)), -1, cv2.LINE_AA, 3)
        img = img + cv2.GaussianBlur(sn, (0, 0), .9)
        # the crack catches the light on its way down
        if 1 in cen:
            c0, Lm = cen[1]; k0 = Lm([(.36, .035)])[0]; a = .85 * math.exp(-((f - 44) / 1.5) ** 2) + .25 * math.exp(-((f - 55) / 2.5) ** 2)
            if a > .03:
                g = Cel(); K.sparkle(g, k0[0], k0[1], (70 + 90 * p) * a + 14, '#dffff8', .12, .07, .95); gl = g.layer(); img = comp.over(img, gl, min(1, a * 1.2)) + qb(gl[..., :3], 14) * a * .8; img = flare(img, k0[0], k0[1], 300 + 200 * p, '#7fe8e0', .28 * a, .35, roll)
            # ---- the tear: slides down the reflected cheek, crosses the rim, hangs in the water as a drop of light
            if f >= 58:
                yh = cen['y']; e0 = yh.Pm(np.array([(.30, -.088)]))[0]; e1 = yh.Pm(np.array([(.315, -.40)]))[0]; rimp = Lm([(.40, -.205)])[0]; out = rimp + np.array([26., 128.]); tt = f - 58
                P_ = [(58, e0), (61, e0 + (e1 - e0) * .26), (64, e1), (66, rimp), (68, rimp + (out - rimp) * .52), (70, rimp + (out - rimp) * .84), (72, rimp + (out - rimp) * .88)]; pos = lerp_keys(f, P_, lambda v: v); R = float(lerp_keys(f, [(58, 4), (61, 11), (65, 17), (68, 22), (72, 24)])); st = float(lerp_keys(f, [(58, .9), (63, 1.25), (66, 1.55), (68, 1.15), (72, 1.0)]))
                wob = 1 + .06 * math.sin(tt * 1.3); an = np.linspace(0, 6.283, 15)[:-1]; tail = 1 + (st - 1) * 1.5 * np.clip(-np.sin(an), 0, 1) ** 2; Q = np.c_[pos[0] + R * wob * np.cos(an), pos[1] + R / wob * np.sin(an) * tail]; P = spl(Q, True, n=6)
                tc = Cel(); tc.grad(P, pos + [0, -R * 1.4], pos + [0, R], [(0, '#4fb7c0', .55), (.45, '#a8f2ee', .80), (1, '#f4fffd', 1)]); tc.line(np.vstack([P, P[:1]]), 2.2, '#eafffb', taper=(.1, .1), press=.5); tc.fill(spl([(pos[0] - R * .55, pos[1] - R * .1), (pos[0] - R * .30, pos[1] - R * .55), (pos[0] - R * .05, pos[1] - R * .60), (pos[0] - R * .28, pos[1] - R * .12)], True, n=5), '#ffffff'); tl = tc.layer(); g_ = min(1., tt / 5.)
                img = comp.over(img, tl) + qb(tl[..., :3], 10) * .9 * g_; img = flare(img, pos[0], pos[1], 190, '#7ff0e6', .22 * g_)
        v = 1 - .42 * np.clip(np.sqrt(((x - W / 2) / (W * .60)) ** 2 + ((y - H / 2) / (H * .64)) ** 2) - .42, 0, 1)[..., None] ** 1.2
        return (img * v).astype(f32)
