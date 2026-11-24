"""h3 four BLACK FLASHES (0, 9, 18, 27), colour draining each time; 36-71 the black lightning opens like spider lilies."""
import numpy as np, cv2, math, os
from jj import W, H, comp, kit, world, fig_street as FS
from jj.fx_act import *
from jj.fx_flash import _opening, _arc_path, _ease
from jj.figure import Fig, POSES
from jj.fig_pose import add_pose, P as POSE, mix_pose, mirror
from jj.paint import ramp, repaint
from chars.yuji import Yuji, YHead
from chars.sukuna import Suk
from chars import sukuna_fire as SF

CX, CY = 960., 552.
FPAL = dict(mul='#e6bfae', shmul='#9c7482', litmul='#ffdcb8')
FRIM = dict(dir=(.25, .5, -.82), col='#ff9a3e', thr=.40, amt=.88)
BR = '#ff2438'


def ez(t, p=2.): t = min(max(t, 0.), 1.); return 1 - (1 - t) ** p


def hot(a, b, t): t = min(1., max(0., t)); return np.asarray(C(a)) * (1 - t) + np.asarray(C(b)) * t


def drawn_smear(img, dx, dy, seed, n=420):
    """a drawn smear: the picture pulled into long tapered strokes of its own colours, ink and light lines over it (no ghost steps)"""
    q = cv2.resize(img.astype(f32), (W // 4, H // 4), interpolation=cv2.INTER_AREA); l = np.hypot(dx, dy); L = int(max(9, l / 4)) | 1; k = np.zeros((L, L), np.uint8); u = np.array([dx, dy]) / (l + 1e-6); cv2.line(k, tuple(np.round(L // 2 - u * L / 2).astype(int).tolist()), tuple(np.round(L // 2 + u * L / 2).astype(int).tolist()), 255, 1, cv2.LINE_AA); k = k.astype(f32); k /= k.sum(); b = cv2.resize(cv2.filter2D(q, -1, k, borderType=cv2.BORDER_REPLICATE), (W, H), interpolation=cv2.INTER_LINEAR); c = Cel(); r = R(seed, 9); m = np.array([-u[1], u[0]])
    for i in range(n):
        p = np.array([r.uniform(0, W), r.uniform(0, H)]); col = img[int(p[1]), int(p[0]), :3]; w = 2.5 + 24 * bms(r.random(), .10, .30); ln = l * r.uniform(.4, 1.25) + 60; a = p - u * ln * .35; e = p + u * ln * .65; c.fill(np.array([a, p + m * w * .5, e, p - m * w * .5]), np.clip(col * r.uniform(.8, 1.12), 0, 1), .95)
    for i in range(46): p = np.array([r.uniform(0, W), r.uniform(0, H)]); ln = l * r.uniform(.6, 1.6); w = 1. + 5 * bms(r.random()); c.fill(np.array([p - u * ln / 2, p + m * w, p + u * ln / 2, p - m * w]), '#0a0508' if i % 3 else '#fff2e4', .9)
    return put(b, c.layer())


class Shot:
    id = 'h3'; n = 72
    finish = dict(bloom=.30, vig=.18)
    EVENTS = [(0, 'impact', dict(size=.85, note='black flash 1, left straight on the guard')), (1, 'bolt', {}), (3, 'bolt', dict(size=.6)), (7, 'whoosh', dict(note='whip to the second')), (9, 'impact', dict(size=.9, note='black flash 2, right hook in the ribs; picture goes red/black/white')), (10, 'bolt', {}), (12, 'bolt', dict(size=.6)), (16, 'whoosh', {}),
              (18, 'impact', dict(size=.95, note='black flash 3, uppercut; black on white; his head snaps back')), (19, 'bolt', {}), (20, 'crack', dict(note='the smile breaks')), (25, 'whoosh', dict(note='one fist drawn back')), (27, 'impact', dict(size=1.0, note='black flash 4, one fist into the lens from the right, he folds at the waist round it; black, red, white')), (28, 'bolt', {}), (30, 'bolt', {}), (33, 'bolt', dict(size=.6)),
              (36, 'silence', dict(dur=36, note='slow: lightning relaxes into flowers')), (38, 'bloom', dict(size=.5, note='first wave of flowers opens')), (42, 'bloom', dict(size=.8, note='hero lily rises from the heart toward the lens')), (52, 'bloom', dict(size=.9, note='second wave runs out through the field')), (60, 'bloom', dict(size=1.0, note='third wave; the hero lily is wide open')), (64, 'petals', dict(note='petals sweep past the lens')), (71, 'cut_black', dict(note='petals shut the lens: black'))]

    def setup(self):
        k = 'cache/h3_v2_'
        def plate(name, fn):
            p = f'{k}{name}.npy'
            if os.path.exists(p): return np.load(p, mmap_mode="r")
            np.save(p, fn().astype(np.float16)); return np.load(p, mmap_mode="r")
        self.bg2 = plate('2', lambda: FS.street(Camera3D((1.5, -3., .5), (-1.6, 4, 3.4), focal=1000., roll=-.12), seed=27, pal=FS.ASH, lit=.02, stripes=False))
        self.field = plate('f', self.paint_field); self.lil = self.layout(); self.fire = plate('fire', lambda: np.clip(np.nan_to_num(world.plate('ruin', mode='fire', view='low', w=2400, h=1350, seed=3).img), 0, 3)); [self.pair(k * 3) for k in range(4)]


    # ---------------------------------------------------------------- hits
    def h1(self, k):
        x, y = grid(); cm = Camera3D((.15, -3.5, .62), (.22, 0, 1.08), focal=1350. + 26 * k, roll=-.05); jx, jy = kit.shake(k, 22 * math.exp(-k / 2.2), 3); cm.cx += jx; cm.cy += jy; e = ez(k / 5.)
        Y = Fig('yuji').pose(mirror(mix_pose(POSES['punch_hit'], POSES['punch_follow'], .5 * e))).place(-.58 + .10 * e, 0, 0, -math.pi / 2 + .05); Yl = Y.draw(cm, k, light=(.8, .3, .4), expr=dict(mood='shout'), pal=FPAL, rim=FRIM, kick=(.9, .2, -.2)); hx, hy = Y.info['hands']['L']
        Sk = Fig('sukuna').pose(mix_pose(POSES['block_back'], add_pose(POSES['block_back'], root=(0, -.05, -.03), pel=(-6, 0, 0)), e)).place(.66 + .16 * e, 0, 0, math.pi / 2 + .1); Sl = Sk.draw(cm, k, light=(-.8, .3, .4), head_fn=SF.head_fn('cel', SF.ASH, 'smile'), pal=FPAL, rim=dict(FRIM, dir=(-.3, .4, -.8)), kick=(-.9, .2, -.2)); Yl = kit.rim(Yl, -3, 1, '#ff5a4a', .7, .5); Sl = kit.rim(Sl, 3, 1, '#ff5a4a', .7, .5)
        cx, cy = hx + 34, hy; dd = np.hypot((x - cx) / 1.25, y - cy); bg = comp.blur(comp.cam2d(f16(self.fire), 1180 - jx, 650 - jy, 1.14 + .008 * k, -.05, W, H), 1.5); im = bg * (1.12 - .40 * math.exp(-k / 4.)) + ramp(np.clip(dd / 1100, 0, 1), [(0, '#fff2e0'), (.09, '#ffb08a'), (.22, '#e83a2a'), (.5, '#6a1218'), (1, '#180608')]) * .85 * math.exp(-k / 2.6)
        c = Cel(); ground_crack(c, ground_proj(cm, centre=(.1, .1)), 2.4, min(1, (k + 1) / 3.), 41, 13, col='#140806', lip='#ffb070', side='#3a1c14', top='#6a4434', topdark='#4a2c24', lw=2.2); debris(c, k, (cx, cy + 420), 34, 4, (500, 1900), (-3., -.14), (5, 56), 1900, pal=dict(lit='#ffb070', mid='#5c3426', dark='#1c0e0c', ink='#0a0504'), spread=(260, 20)); im = put(im, c.layer()); im = put(im, embers(k, 80, 6, wind=(260., -260.), near=.14))
        if 0 < k < 5: c = Cel(); radial_wedges(c, cx, cy, 70, 330, 2400, 5 + k, .022, BONE); im = put(im, c.layer(), (0, .62, .38, .20, .08)[k])
        im = put(im, black_flash_layers(5 + k, cx, cy, .95, 3, 0., 16, 980, .30, part='back', shard=.6)); im = put(im, Sl); im = put(im, Yl); fl = black_flash_layers(5 + k, cx, cy, .95, 3, 0., 16, 980, .30, part='front', star=False, shard=0.); far = smooth(330, 640, np.hypot(x - cx, y - cy))[..., None]; im = put(im, FX(fl.rgba * far, fl.light * far if fl.light is not None else None)); c = Cel(); c.fill(star_poly(cx, cy, 26, 120 - 10 * k, 11, 4 + k), '#fff6ea'); im = put(im, c.layer())
        if k == 0: c = Cel(); radial_wedges(c, cx, cy, 70, 330, 2400, 5, .022, BONE); [c.fill(K, BONE) for K, _ in kite_star(cx, cy, 60, 620, 22, 9, (0., .9), .10, 0)]; im = put(im, c.layer())
        return im if k < 7 else drawn_smear(im, 420 + 260 * (k - 7), -210 - 130 * (k - 7), 31 + k)

    def h2(self, k):
        cm = Camera3D((.70, -1.75, .50), (-.02, .05, .98), focal=1120. + 22 * k, roll=.12); jx, jy = kit.shake(k + 20, 24 * math.exp(-k / 2.2), 3); cm.cx += jx; cm.cy += jy; e = ez(k / 5.)
        hook = add_pose(POSES['punch_hit'], root=(0, 0, -.04)); Y = Fig('yuji').pose(mix_pose(hook, POSES['punch_follow'], .5 * e)).place(-.44 + .08 * e, -.12, 0, -math.pi / 2 - .45); Yl = Y.draw(cm, k, light=(.7, .3, .5), expr=dict(mood='rage'), kick=(.9, .2, -.2), weight=1.15); hx, hy = Y.info['hands']['R']
        Sk = Fig('sukuna').pose(add_pose(mix_pose(POSES['block_back'], POSES['stagger'], .12 + .2 * e), hed=(-6, 0, 0))).place(.50 + .14 * e, .10, 0, math.pi / 2 + .55); Sl = Sk.draw(cm, k, light=(-.7, .3, .5), head_fn=SF.head_fn('cel', SF.ASH, 'strain'), kick=(-.9, .2, -.2), weight=1.15); cx, cy = hx + 20, hy
        c = Cel(); c.fill(np.array([[0, 0], [W, 0], [W, H], [0, H]]), RED); radial_wedges(c, cx, cy, int(110 - 7 * k), 330 + 24 * k, 2400, 14 + (k // 2 if k > 3 else k), .030, BK); radial_wedges(c, cx, cy, 14, 430, 2400, 24 + k // 2, .010, BONE); im = c.layer()[..., :3]
        G = kit.hatch if False else None; M = lambda L, lo, hi: toned(L, 'red', lo=lo, hi=hi)
        Yt = M(Yl, .24, .60); Yt = kit.rim(Yt, -4, 2, BONE, 1., .4); Yt = kit.rim(Yt, 5, 1, BONE, 1., .4); ov = Yt + M(Sl, .30, .62) * (1 - Yl[..., 3:4]); fl = black_flash_layers(3 + k if k > 1 else 2 + k, cx, cy, 1.05, 7, .2, 16, 1050, .8, part='back'); im = put(im, FX(fl.rgba, None)); im = put(im, ov); im = put(im, FX(black_flash_layers(3 + k if k > 1 else 2 + k, cx, cy, 1.05, 7, .2, 16, 1050, .8, part='front').rgba, None))
        return im if k < 7 else drawn_smear(im, -460 - 280 * (k - 7), 260 + 150 * (k - 7), 51 + k)

    def h3(self, k):
        """black on white: the uppercut lands, the smile breaks"""
        e = ez(k / 4., 2.6); s = 330 + 10 * k; roll = -.10 - .42 * e; hx, hy = 1060 + 70 * e, 420 - 60 * e; jx, jy = kit.shake(k + 40, 26 * math.exp(-k / 2.), 3); hx += jx; hy += jy
        c = Cel(); c.fill(np.array([[0, 0], [W, 0], [W, H], [0, H]]), BONE); px, py = 880 + jx, 720 + jy; radial_wedges(c, px, py, 120 - 6 * k, 380 + 30 * k, 2400, 34 + (k // 2 if k > 3 else k), .028, BK); im = c.layer()[..., :3]
        back = black_flash_layers(3 + k if k > 1 else 2 + k, px, py, 1.1, 13, -1.2, 17, 1150, .9, part='back').rgba; im = put(im, self.bw(back))
        S = Suk(hx, hy, s, -.38 + .16 * e, .10 + .36 * e, roll, 'smile' if k < 1 else dict(eo=(.95 - .2 * e, .60), e2=(1., 1.), brow=(.01, .05), mouth='grit', jaw=.05, look=(.1, .6), low=.3), seed=4, wind=.5, noir=.35);
        nb = np.asarray(S.H.P3(np.array([0, -1.22, -.05])), float)[:2]; dn = -np.asarray(S.up, float) / s; rt = np.asarray(S.rt, float) / s; T = lambda u, v: nb + rt * u * s + dn * v * s; c = Cel(); body = spl(np.array([T(-.62, -.12), T(-.30, .06), T(.10, .10), T(.50, -.02), T(.74, -.20), T(1.5, .35), T(2.9, 1.2), T(3.4, 4.), T(-3.2, 4.), T(-2.4, 1.1), T(-1.2, .35)]), True, (4, 7, 8), n=8); c.fill(body, BK)
        for (u0, v0, u1, v1, w_) in ((-.5, .08, -1.7, 1.0, 12), (-.2, .25, -.6, 1.6, 10), (.2, .28, .5, 1.5, 12), (.55, .12, 1.8, .95, 14), (-.95, .45, -1.4, 1.9, 8), (1.0, .5, 1.3, 1.8, 9)): c.fill(band(np.array([T(u0, v0), T((u0 + u1) / 2 + .08, (v0 + v1) / 2), T(u1, v1)]), lambda v, w_=w_: w_ * (1 - v) ** .8, 3, 0, 60, .1), BONE)
        for (u0, u1, v, b_) in ((-.66, .78, -.10, .20), (-.58, .70, .08, .22), (-.48, .62, .26, .2)): P_ = spl(np.array([T(u0, v), T(u0 * .4 + u1 * .2, v + b_ * .9), T((u0 + u1) / 2 + .1, v + b_), T(u1, v - .06)]), n=8); c.line(P_, 5, BONE, taper=(.3, .4), press=.5, seed=int(v * 50) + 9)
        im = put(im, c.layer()); L = S.ink(dark=-1, neck='dark'); g = cv2.cvtColor(L[..., :3].astype(f32), cv2.COLOR_RGB2GRAY)[..., None]; a = L[..., 3:4]; t = smooth(.34, .40, g / np.maximum(a, .02)); im = im * (1 - a) + (C(BK) * (1 - t) + C(BONE) * t) * a
        c = Cel()
        for i in range(30):                                  # spit and sweat thrown off as ink
            an = -2.2 + 1.9 * h1(i, 1); d = 120 + (260 + 520 * h1(i, 2)) * ez((k + 1) / 5.); r_ = 2 + 12 * h1(i, 3) ** 3; X, Y = px + 60 + math.cos(an) * d, py - 120 + math.sin(an) * d; th = lin01(12)[:-1] * 6.283; q = c2(np.cos(th) * r_ * (1.2 + 1.6 * h1(i, 4)), np.sin(th) * r_); ca, sa = math.cos(an), math.sin(an); c.fill(c2(q[:, 0] * ca - q[:, 1] * sa + X, q[:, 0] * sa + q[:, 1] * ca + Y), BK)
        im = put(im, c.layer()); pp = dict(HD.PAL['night'], skin='#fff6f0', shade='#060308', tape='#fff6f0', tape_shade='#060308', line='#060308', nail='#fff6f0', nail_shade='#060308'); F, g_ = fist(px - 10 + 30 * e, py + 20 - 60 * e, 330 + 12 * k, E3(150, -20, -24), 'R', 3.0, 'fist', pp, (-.6, .6, .5), 7.5, 7, 460, True, True, dict(cloth='#060308', cloth_sh='#060308', cloth_lt='#fff6f0', line='#060308', cast='#060308')); im = put(im, F)
        front = black_flash_layers(3 + k if k > 1 else 2 + k, px, py, 1.1, 13, -1.2, 17, 1150, .9, part='front').rgba; im = put(im, self.bw(front)); c = Cel(); ks = kite_star(px - 10, py - 20, 40, (430 if k < 1 else 250) - 22 * k, 15, 6 + k, (2.5, 1.0), .06, 5); ks = [q for q in ks if (q[0].mean(0) - [px, py]) @ np.array([.8, -.6]) < 40]
        for K, K2 in ks: c.fill(K2, BK)
        for K, K2 in ks: c.fill(K, BONE)
        return put(im, c.layer()) if k < 6 else put(im, c.layer(), .0)

    def bw(self, L):
        """black flash layer without its red: edges turn white"""
        a = L[..., 3:4]; col = np.where(a > .02, L[..., :3] / np.maximum(a, .02), 0); r = np.clip((col[..., :1] - col[..., 1:2]) * 1.6, 0, 1); col = col * (1 - r) + C(BONE) * r; return np.dstack([col * a, a])

    def h4(self, k):
        """black / red / white: Yuji's fist huge at the lens on the right, Sukuna in profile folded at the waist round it, every limb streaming to the left edge"""
        e = ez(k / 6., 2.2); jx, jy = kit.shake(k + 60, 26 * math.exp(-k / 2.), 3); x, y = grid(); J = np.array([jx, jy]); KX_, KY_ = 1000 - 44 * e + jx, 690 + jy; d = np.hypot((x - KX_) / 1.25, y - KY_)
        im = ramp(np.clip(d / 1200, 0, 1), [(0, '#ff5a4c'), (.16, RED), (.6, '#c80f26'), (1, '#8a0a1a')]).astype(f32); dk = smooth(1040, 1560, x + .25 * (y - 540)); im = im * (1 - dk)[..., None] + C(BK) * dk[..., None]; c = Cel(); radial_wedges(c, KX_, KY_, 110 - 6 * k, 420 + 24 * k, 2400, 54 + (k // 2 if k > 3 else k), .026, BK); im = put(im, c.layer()); c = Cel(); r = R(54, 3 + k // 2)
        for i in range(70):                                  # the arm's speed, behind it
            Y = r.uniform(-20, 1100); X0 = r.uniform(1100, 1900); l = r.uniform(160, 620); c.fill(kite(np.array([X0 + l, Y + .10 * l]), np.array([X0, Y]), 2 + 7 * bms(r.random()), r, pk=.4), RED if i % 4 else BONE)
        im = put(im, c.layer()); im = put(im, black_flash_layers(2 + k, KX_, KY_, 1.0, 23, 0., 15, 1150, .6, part='back', star=False, shard=.35).rgba)
        wv = lambda a, s_: a * math.sin(k * .55 + s_) * (.4 + .6 * (1 - e)); M = lambda P, w=1., dy=0.: np.array(P, float) * 1. + [-52 * e * w, dy] + J
        c = Cel()
        def blob(P, cr=(), w=1., dy=0., fl=BONE, lw=7.): q = spl(M(P, w, dy), True, cr, n=9); c.fill(q, fl); kit.outline(c, q, lw, BK, 7); return q
        def fold(P, w=1., dy=0., hw=13): c.fill(band(spl(M(P, w, dy), n=10), lambda v: hw * np.clip(np.asarray(v) / .16, 0, 1) ** .8 * (1 - np.asarray(v)) ** .7, 3, 0, 60, .1), BK)
        def boot(ox, oy, w, dy): q = blob([(ox + 40, oy - 70), (ox - 40, oy - 78), (ox - 110, oy - 50), (ox - 200, oy - 6), (ox - 226, oy + 44), (ox - 150, oy + 70), (ox - 30, oy + 52), (ox + 46, oy + 50)], (), w, dy, BK, 5.); c.line(spl(M([(ox - 204, oy + 34), (ox - 120, oy + 44), (ox - 10, oy + 26)], w, dy), n=8), 5, BONE, taper=(.2, .3), press=.5); return q
        # far arm, far leg (behind)
        dyA, dyB, dyC, dyD = wv(26, .0), wv(22, 1.1), wv(20, 2.), wv(24, 2.9)
        blob([(770, 372), (640, 262), (500, 196), (388, 150), (352, 232), (270, 322), (150, 372), (300, 396), (480, 384), (660, 452)], (3, 6), 1.35, dyA); fold([(690, 380), (520, 290), (330, 300)], 1.35, dyA); fold([(640, 330), (500, 250), (400, 200)], 1.35, dyA, 10)
        boot(186, 796, .8, dyC); blob([(720, 770), (560, 730), (400, 738), (226, 700), (150, 730), (136, 846), (206, 892), (380, 868), (560, 880), (720, 900)], (4, 5), .75, dyC); fold([(640, 800), (460, 790), (250, 770)], .75, dyC); fold([(600, 850), (430, 835), (260, 840)], .75, dyC, 10)
        # the body: one C, hollow side to the fist
        T = blob([(716, 352), (612, 428), (532, 548), (508, 684), (548, 806), (650, 890), (772, 946), (900, 966), (1000, 926), (1046, 846), (972, 776), (944, 700), (962, 634), (1020, 572), (970, 514), (880, 462), (800, 396)]); c.clip(T)
        bk = spl(M([(716, 352), (612, 428), (532, 548), (508, 684), (548, 806), (650, 890), (772, 946)]), n=12); c.fill(band(bk, lambda v: 30 + 26 * np.sin(np.pi * np.clip(v, 0, 1)), 5, 0, 120, .22, wave=.10), BK)
        for j in range(19):                                  # cloth pulled into the blow
            an = math.pi + (j - 9) * .128; r0 = 66 + 34 * h1(j, 1); r1 = r0 + 130 + 210 * h1(j, 2); p0 = np.array([KX_ - 22 + math.cos(an) * r0, KY_ + math.sin(an) * r0 * 1.15]); p1 = np.array([KX_ - 22 + math.cos(an) * r1 * 1.15, KY_ + math.sin(an) * r1 * 1.15]); c.fill(kite(p0, p1, 9 + 11 * h1(j, 3), R(j, 4), pk=.18), BK)
        c.unclip(); ob = blob([(512, 664), (660, 640), (800, 640), (950, 668), (948, 730), (966, 772), (800, 770), (660, 782), (545, 800), (516, 730)], (0, 3, 5, 8), 1., 0., BK, 4.)
        for o_ in (0, 34, 66): c.line(spl(M([(522 + .3 * o_, 684 + 1.1 * o_), (680, 664 + o_), (810, 662 + o_), (940, 688 + .9 * o_)]), n=9), 4.5, BONE, taper=(.1, .2), seed=o_, press=.4)
        blob([(560, 742), (500, 706), (420, 700), (330, 664), (350, 724), (420, 760), (330, 790), (440, 806), (540, 800)], (3, 6), 1.5, dyB, BK, 4.)                                      # sash ends stream left
        # near leg
        boot(258, 1004, .7, dyD); blob([(900, 900), (760, 872), (600, 876), (440, 884), (300, 900), (216, 918), (190, 1046), (286, 1086), (440, 1048), (620, 1040), (800, 1010), (930, 968)], (5, 6), .65, dyD); fold([(820, 930), (620, 946), (360, 962)], .65, dyD, 15); fold([(760, 980), (560, 1000), (340, 1030)], .65, dyD, 11); fold([(700, 906), (520, 912), (330, 926)], .65, dyD, 9)
        # neck, collar
        nk = blob([(716, 366), (768, 296), (838, 236), (900, 300), (940, 400), (900, 480), (840, 446), (770, 396)], (0,)); c.clip(nk); c.fill(spl(M([(700, 380), (770, 300), (826, 250), (850, 330), (868, 400), (850, 470), (770, 420)]), True, n=8), BK); c.unclip(); c.line(spl(M([(868, 330), (884, 390), (876, 446)]), n=8), 5, BK, taper=(.2, .4)); co = spl(M([(690, 376), (770, 400), (850, 446), (930, 500), (960, 470), (880, 408), (790, 356), (716, 330)]), True, n=8); c.fill(co, BK); im = put(im, c.layer())
        hx, hy, hr = 1000 - 48 * e + jx, 232 + 14 * e + jy, .12 + .16 * e; S = Suk(hx, hy, 178, .98, .06, hr, dict(mouth='laugh', jaw=.28 + .12 * min(1., k / 2.), eo=(1.35, 1.42), iris=(.55, .5), e2=(1., 1.), brow=(.03, .07), look=(.2, .1), low=.2), seed=4, wind=.9, noir=.3); L = S.ink(dark=-1, head_only=True); a = L[..., 3:4]; q = np.where(a > .02, L[..., :3] / np.maximum(a, .02), 0).astype(f32); g = cv2.cvtColor(q, cv2.COLOR_RGB2GRAY)[..., None]; t = smooth(.32, .37, g); rt = q[..., 1:2] / np.maximum(q[..., :1], .03); rd_ = (1 - smooth(.28, .40, rt)) * smooth(.14, .22, q[..., :1]); im = im * (1 - a) + (C(BK) * (1 - t) * (1 - rd_) + C(BONE) * t * (1 - rd_) + C(RED) * rd_) * a
        pb = dict(HD.PAL['night'], skin='#fff6f0', shade='#060308', line='#060308', band='#060308', nail='#060308', nail_dark='#060308', nail_shade='#060308'); H_, _ = fist(300 - 70 * e + jx, 128 + dyA + jy, 92, E3(200, 24, 66), 'R', 3.0, 'splay', pb, (0, .7, .5), 4.6, 5, 200, False, False, nails='claw', bands=((.04, .12), (.18, .26))); im = put(im, H_)
        c = Cel(); sv = blob([(748, 392), (640, 388), (500, 412), (372, 400), (262, 386), (226, 470), (150, 566), (60, 610), (220, 622), (400, 580), (560, 566), (690, 520)], (4, 7), 1.35, dyB); fold([(690, 450), (500, 462), (280, 470)], 1.35, dyB, 14); fold([(640, 500), (460, 520), (240, 560)], 1.35, dyB, 11); fold([(600, 410), (450, 430), (310, 420)], 1.35, dyB, 9); im = put(im, c.layer()); H_, _ = fist(196 - 70 * e + jx, 410 + dyB + jy, 112, E3(196, 30, 88), 'L', 3.0, 'splay', pb, (0, .7, .5), 4.6, 5, 200, False, False, nails='claw', bands=((.04, .12), (.18, .26))); im = put(im, H_)
        c = Cel(); ks = kite_star(KX_ - 6, KY_, 60, (330 if k < 1 else 250 - 16 * k), 17, 8 + k // 2, (math.pi / 2, .8), .06, 8); ks = [q for q in ks if q[0].mean(0)[1] > KY_ - 70]                                                                # the hit, behind the knuckles
        for K_, K2 in ks: c.fill(K2, BK)
        for K_, K2 in ks: c.fill(K_, BONE if k < 1 else RED)
        im = put(im, c.layer()); pp = dict(HD.PAL['night'], skin='#f0142e', shade='#060308', tape='#ff3442', tape_shade='#060308', line='#060308', nail='#ff3442', nail_shade='#060308'); s = 540 + 60 * e; F, _ = fist(W - 1190 + 20 * e, 738, s, E3(134 + 6 * e, 28, -80), 'R', 2.05, 'fist', pp, (.55, .5, .6), 9, 7, 460, True, True, dict(cloth='#060308', cloth_sh='#060308', cloth_lt='#f0142e', line='#060308', cast='#060308'), thr=.10); F = np.ascontiguousarray(F[:, ::-1]); dxf = KX_ - 34 - float(np.nonzero((F[600:780:6, :, 3] > .5).any(0))[0].min()); F = kit.shifted(F, dxf, jy); xc = int(min(W, W + dxf)) - 12; cam = HD.Cam(W - 1190 + 20 * e, 738, s, 2.05); h = HD.Hand('fist', 'R', R=E3(134 + 6 * e, 28, -80), pivot=(0, .8, 0), bld=HD.HEAVY); E_ = cam.px(h.world(np.array([0, -1.9, 0.]))); c = Cel(); p0 = np.array([W - E_[0] - 120., E_[1] + jy]); P = np.array([p0 + [0, -190], [1990, p0[1] - 150], [1990, p0[1] + 330], p0 + [0, 210]]); c.fill(P, BK); c.line(P[:2], 7, RED, taper=(.1, .1), press=.4); c.line(P[2:][::-1], 7, RED, taper=(.1, .1), press=.4)
        for j in range(4): q_ = p0 + [90 + 120 * j, -120 + 30 * j]; c.fill(kite(q_, q_ + [150, 230], 18, R(j, 6), pk=.4), RED)
        A = c.layer(); F = F + A * (1 - F[..., 3:4]); col = F[:, xc - 3, 3:4].copy(); F[:, xc:, 3:4] = col[:, None, :]; F[:, xc:, :3] = C(BK) * col[:, None, :]; hl = cv2.dilate(cv2.resize(F[..., 3], (W // 2, H // 2), interpolation=cv2.INTER_AREA), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))); hl = cv2.resize(hl, (W, H))[..., None]; im = im * (1 - hl) + C(BK) * hl; F = kit.rim(F, 5, 5, BONE, 1., .3, where=((x < xc - 16) & (y < H - 30)).astype(f32)); return put(im, F)

    # ---------------------------------------------------------------- the field
    def paint_field(self):
        x, y = grid(); d = np.hypot((x - CX) / 1.15, (y - CY) / .95); n = fbm(H, W, (240, 90, 36), None, 7); g = ramp(np.clip(d / 1120 + .03 * n, 0, 1), [(0, '#ffe8d4'), (.045, '#ffb890'), (.10, '#ff5a48'), (.22, '#ec1a30'), (.46, '#a80c22'), (.74, '#5a0616'), (1, '#1c0208')]); return repaint(g, sizes=(44, 20, 9), seed=4, angle=np.arctan2(y - CY, x - CX) + .45, density=.9, aspect=8, jit=(.008, .03, .035))

    def layout(self):
        """(x, y, size, outward angle, birth frame, seed, florets, detail, tilt): bilateral, alive"""
        L = []; r = R(3)
        def ring(n, rx, ry, size, t0, t1, off, fl, det, jit):
            for i in range(n):
                a = off + i / n * 6.283; L.append([CX + math.cos(a) * rx * (1 + jit * r.normal()), CY + math.sin(a) * ry * (1 + jit * r.normal()), size * r.uniform(.9, 1.12), a, t0 + (t1 - t0) * r.random(), len(L) + 3, fl, det, r.uniform(.70, .86)])
        ring(10, 985, 640, 345, 43, 49, .314, 5, .35, .02); ring(6, 1010, 760, 320, 45, 50, .785 - .262, 5, .2, .03); ring(8, 640, 430, 262, 40, 46, 0.39, 6, .6, .02); ring(6, 385, 268, 205, 36, 41, math.pi / 6, 6, 1., .015); return L

    def wave(self, dist, f):
        """bloom waves that run outward through the field"""
        return sum(math.exp(-((dist - 34. * (f - f0)) / 120.) ** 2) for f0 in (41, 53, 61))
    def lilies(self, f, zoom):
        cel = Cel(); ctx = cel.ctx; c0 = np.array([CX, CY]); ctx.save(); ctx.translate(CX, CY); ctx.scale(zoom, zoom); ctx.translate(-CX, -CY); order = [q for q in self.lil]
        for X, Y, size, a, t0, sd, fl, det, tl in order:     # stems: bolts that relax, then keep swaying
            k = _ease((f - 36) / max(8., t0 - 30)); p1 = np.array([X, Y]); bolt = bolt_line(c0 + (p1 - c0) * .03, p1, sd * 7, .20, 6, .1); d = p1 - c0; m = np.array([-d[1], d[0]]) * (.16 * math.sin(sd * 2.4) + .030 * math.sin(f * .22 + sd * 1.9)); curve = spl(np.array([c0 + d * .03, c0 + d * .42 + m, c0 + d * .74 + m * .9, p1]), n=10); curve = resample(curve, np.hypot(*np.diff(curve, axis=0).T).sum() / 30.)
            g = min(1., (f - 34.5) / 3.); P = _opening(bolt, curve, k); n = max(3, int(len(P) * g)); w = size * .013; wv = self.wave(np.hypot(*d) * .5, f)
            for e, col in ((3.2 + 1.6 * min(1., wv), hot(BR, '#ff6a50', wv)), (0, BK)): cel.fill(band(P[:n], lambda v, e=e: (w * (1 - .45 * v) + e) * np.clip(np.asarray(v) / .32, .04, 1) ** 1.1 * (1 + .5 * (1 - k) * np.sin(v * 23 + sd)), sd, 0, 60, .0, wave=.05, sides=(0, 0)), col)
        for X, Y, size, a, t0, sd, fl, det, tl in order:
            wv = self.wave(math.hypot(X - CX, Y - CY), f); t = (f - t0) / 20.
            if t < -.15: continue
            t = min(1., t) * (.90 + .05 * math.sin(f * .21 + sd)) + .12 * min(1., wv); grow = _ease((f - t0 + 4) / 9.) * (1 + .11 * min(1., wv)); br = 1 + .022 * math.sin(f * .19 + sd); ctx.save(); ctx.translate(X, Y); ctx.rotate(a + math.pi / 2 + .05 * math.sin(f * .15 + sd * 1.7)); ctx.scale(grow * br, grow * br); higanbana(cel, 0., 0., size, max(.02, t), sd, tl, fl, (BK, hot(BR, '#ff5c48', wv) if size < 300 else hot('#c01228', '#f02a3c', wv), DRED, BONE), f * .12, sd * 1.1, max(1.6, size * (.010 if size < 300 else .0065) * (1 + .25 * min(1., wv))), None, det); ctx.restore()
        ctx.restore(); return cel.layer()
    def hero(self, f):
        """one lily in real depth: a bud low in the field, its axis tipped up; it opens, turns its face to us and comes at the lens"""
        q = f - 39.
        if q < 0: return None
        op = _ease(q / 24.); D = 13.5 * math.exp(-.058 * q) if q < 24 else 3.36 * math.exp(-.28 * (q - 24)); F_ = 900.; tau = 1.12 * (1 - _ease((q - 9) / 19.)); cx, cy = CX + 10 * math.sin(f * .2), 905 - 300 * _ease((q - 15) / 13.); n = 24; v = lin01(n); it = []; ct, st_ = math.cos(tau), math.sin(tau)
        def d3(r, z, ps): y_ = r * np.sin(ps); return r * np.cos(ps), y_ * ct - z * st_, y_ * st_ + z * ct
        def pj(r, z, ps): X, Y, Z = d3(r, z, ps); k = F_ / np.maximum(D - Z, .10); return c2(cx + X * k, cy + Y * k)
        for i in range(6):
            ph = i * 1.0472 + .26 + .05 * math.sin(f * .17 + i); o2 = _ease(q / 24. * 1.35 - .30 * h1(i, 1))
            for j in range(6):
                ps0 = ph + (j - 2.5) * (.10 + .24 * o2) + .12 * (h1(i, j, 2) - .5); Lp = 1.0 + .34 * h1(i, j, 3); th = .07 + .52 * o2 + .12 * (j % 2) + (.16 + 2.75 * o2 * (.70 + .45 * h1(i, j, 4))) * v ** 1.6; st = Lp / (n - 1); r = .09 * o2 + np.r_[0, np.cumsum(np.sin(th[:-1]) * st)]; z = np.r_[0, np.cumsum(np.cos(th[:-1]) * st)] * .95; ps = ps0 + .42 * (h1(i, j, 5) - .5) * v ** 1.4 + .05 * np.sin(v * 9 + f * .2 + j) * v; w = Lp * (.032 + .066 * o2) * np.sin(np.pi * np.minimum(v, .999) ** .78) ** .9 * (1 + .18 * np.sin(v * 15 + j) * o2); it.append((d3(r, z, ps)[2], r, z, ps, w, i * 6 + j))
        it.sort(key=lambda a_: a_[0].mean()); c = Cel(); zs = [a_[0].mean() for a_ in it]; z0, z1 = min(zs), max(zs) + 1e-3
        for Zc, r, z, ps, w, sd in it:
            ok = D - Zc > .13
            if ok.sum() < 8: continue
            r, z, ps, w = r[ok], z[ok], ps[ok], w[ok]; zm = Zc[ok].mean(); rr = np.maximum(r, .035); e = .013 + .010 * h1(sd, 7); sg = 1 if h1(sd, 8) > .5 else -1; L_ = lambda g: pj(r, z, ps + g / rr); kO = L_(w + e * (1.5 if sg > 0 else .6)); kI = L_(-(w + e * (.6 if sg > 0 else 1.5))); lit = float(np.clip((zm - z0) / (z1 - z0), 0, 1)); c.fill(np.vstack([kO, kI[::-1]]), np.asarray(C(BR)) * (.55 + .45 * lit)); pet = np.vstack([L_(w), L_(-w)[::-1]]); c.fill(pet, np.asarray(C('#34060f')) * (1 - lit) + np.asarray(C(BK)) * lit); pxw = F_ / max(D - zm, .12) * .012
            if pxw > .5: thin(c, L_(w * .05)[1:-3], max(1.2, pxw), DRED, 1., .2, .6, sd); thin(c, L_(w * .55 * sg)[4:-2], max(1., pxw * .7), BR, .9, .3, .5, sd); thin(c, L_(-w * .40 * sg)[6:-5], max(1., pxw * .5), DRED, .9, .3, .5, sd + 3)
        for i in range(6):                                   # stamens thrown past the petals, each its own curve
            for j in range(4):
                ps0 = i * 1.0472 + .26 + (j - 1.5) * (.08 + .20 * op); Lp = 1.45 + .65 * h1(i, j, 11); th = .04 + .34 * op + (.10 + 1.05 * op * (.5 + .8 * h1(i, j, 12))) * v ** 1.5 - .5 * op * v ** 4; st = Lp / (n - 1); r = .04 + np.r_[0, np.cumsum(np.sin(th[:-1]) * st)]; z = np.r_[0, np.cumsum(np.cos(th[:-1]) * st)] * .85; ps = ps0 + .75 * (h1(i, j, 13) - .5) * v ** 1.3 + .07 * np.sin(v * 5 + f * .25 + j); Zc = d3(r, z, ps)[2]; ok = D - Zc > .15
                if ok.sum() < 7: continue
                P = pj(r, z, ps)[ok]; k_ = F_ / max(D - Zc[ok][-1], .15); thin(c, P, max(1.3, min(9., k_ * .011)), BR if j % 2 else DRED, 1., .05, .35, j); thin(c, P, max(.8, min(4., k_ * .0045)), BK if j % 2 else BR, 1., .05, .35, j)
                if op > .35 and ok.all(): u = P[-1] - P[-3]; u = u / (np.hypot(*u) + 1e-6); m = np.array([-u[1], u[0]]); l = min(15., k_ * .030); c.fill(spl(np.array([P[-1] - m * l, P[-1] + u * l * .4, P[-1] + m * l, P[-1] - u * l * .4]), True, n=4), '#ffb49a' if j % 2 else BR)
        return c.layer()
    def lens(self, f):
        """65-71: petals slide across the lens until there is nothing but black"""
        if f < 65: return None
        g = (f - 64.) / 6.6; c = Cel(ss=1 if f > 68 else 2)
        if f >= 71: c.fill(np.array([[-9, -9], [W + 9, -9], [W + 9, H + 9], [-9, H + 9]]), BK); return c.layer()
        for i in range(11):
            a = i * 2.399 + .3; gi = min(1.25, max(0., g * (1.15 + .35 * h1(i, 1)) - .20 * h1(i, 2))); d = np.array([math.cos(a), math.sin(a)]); tip = 1250 * max(0., 1 - gi) ** 1.1 - 330 * max(0., gi - .55); p1 = np.array([CX, 540.]) + d * tip; p0 = np.array([CX, 540.]) + d * 1700; m = np.array([-d[1], d[0]]); bow = m * 190 * (1 if i % 2 else -1); P = spl(np.array([p0, p0 * .55 + p1 * .45 + bow, p1 + bow * .12]), n=14); wd = lambda v, i=i: (330 + 130 * h1(i, 3)) * np.sin(np.pi * np.clip(.16 + .84 * np.asarray(v), 0, 1) ** .9) ** .8 + 8
            c.fill(band(P, lambda v: wd(v) + 9, i, 0, 400, .05, wave=.05, sides=(0, 0)), BR); c.fill(band(P, wd, i, 0, 400, .05, wave=.05, sides=(0, 0)), BK); pt = Path(P, 8.); thin(c, pt.sub(pt.L * .2, pt.L * .96, lambda v: wd(.2 + .76 * v) * .05), 5, DRED, 1., .2, .5, i); thin(c, pt.sub(pt.L * .35, pt.L * .9, lambda v: wd(.35 + .55 * v) * .55), 4, DRED, .9, .3, .6, i + 5)
        return c.layer()
    def pair(self, f):
        """the two fighters at the heart: four silhouettes, hair and cloth moving"""
        k = (f // 3) % 4; p = f'cache/h3_pair5_{k}.npy'
        if not os.path.exists(p):
            cm = Camera3D((0, -9.6, 1.05), (0, 0, 1.0), focal=1500.); Y = Fig('yuji').pose(mirror(POSES['punch_hit'])).blow(-1.6, 0, .25).place(-.98, 0, 0, -math.pi / 2 + .05).draw(cm, 5 + k * 7); S = Fig('sukuna').pose(POSES['palm']).blow(1.6, 0, .25).place(.98, 0, 0, math.pi / 2 + .05).draw(cm, 9 + k * 7); np.save(p, (np.clip(np.maximum(Y[..., 3], S[..., 3]), 0, 1) * 255).astype(np.uint8))
        return np.asarray(np.load(p, mmap_mode='r'), f32) / 255.
    def bloom(self, f):
        q = f - 36; zoom = 1.46 - .46 * ez(q / 35., 1.7); im = comp.cam2d(f16(self.field), CX, CY - 12, zoom, 0., W, H); x, y = grid()
        g = np.exp(-(np.hypot((x - CX) / 1.2, y - CY) / (200 * zoom)) ** 2)[..., None]; im = im + g * C('#ffb08a') * (.55 + .06 * math.sin(f * .7) + .35 * math.exp(-q / 6.)); ray = Cel(ss=1); radial_wedges(ray, CX, CY, 60, 300, 2400, 3 + f // 4, .012, '#ff6a58'); im = put(im, ray.layer(), .18); L = self.lilies(f, zoom); sh = cv2.GaussianBlur(kit.shifted(L, 8, 14)[..., 3], (0, 0), 9)[..., None]; im = im * (1 - .45 * sh); im = put(im, L); dd = np.hypot((x - CX) / 1.15, y - CY); im = im * (1 - .62 * smooth(430 * zoom, 1080 * zoom, dd))[..., None]
        s = .66 * zoom; P = cv2.warpAffine(self.pair(f), np.float32([[s, 0, CX - s * W / 2], [0, s, CY + 14 - s * 620]]), (W, H), flags=cv2.INTER_AREA)[..., None]; halo = cv2.GaussianBlur(cv2.resize(P[..., 0] if P.ndim == 3 else P, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 5); im = im + cv2.resize(halo, (W, H))[..., None] * C('#ffd8b0') * .35; im = im * (1 - P) + C('#0a0205') * P
        c = Cel(); c.fill(star_poly(CX, CY - 22 * zoom, 5 * zoom, (40 + 12 * math.sin(f * .5)) * zoom, 7, f // 3, rot=f // 3), '#fffaf0')
        if q < 7: B = black_flash_layers(7 + q, CX, CY - 14, .9, 31, 0., 15, 900, .5, part='back', star=False).rgba; im = put(im, B, max(0., 1 - q / 7.))
        im = put(im, c.layer()); H_ = self.hero(f)
        if H_ is not None: sh = cv2.GaussianBlur(cv2.resize(H_[..., 3], (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 9); im = im * (1 - .80 * cv2.resize(sh, (W, H))[..., None]); im = put(im, H_)
        im = put(im, petals(f, 80, 5, BLACKPETAL, HIGAN, (-60., 60.), (9, 32), edge=BR, spin=1.4, sway=40., near=5, tau=f / 24. * 1.3 + 3)); Z = self.lens(f); return im if Z is None else put(im, Z)

    LIFT = {15: .05, 16: .13, 17: .17, 27: .34, 28: .22, 29: .12, 30: .05}         # red grounds lifted toward bone
    SINK = {18: .45, 19: .58, 20: .76, 21: .90, 24: .93, 25: .80, 26: .60}         # bone grounds pulled toward red
    def ease(self, f, im):
        """no hard value flip between the four patterns: every change of ground is spread over at least four frames"""
        R_, B_ = C(RED), C(BONE); a = self.LIFT.get(f, 0.)
        if a: im = im + a * (B_ - R_) * np.clip(im[..., :1] - im[..., 1:2], 0, 1)
        if f in self.SINK: im = im * (R_ + (B_ - R_) * self.SINK[f]) / B_
        if f in (7, 8): w = .28 * (f - 6); g = cv2.cvtColor(np.clip(im, 0, 1).astype(f32), cv2.COLOR_RGB2GRAY)[..., None]; im = im * (1 - w) + (R_ * .22 + R_ * 1.6 * g) * w
        return im * {9: .86, 10: .93, 34: .95, 35: .88}.get(f, 1.)
    def frame(self, f):
        f = int(f); return self.ease(f, self.h1(f) if f < 9 else self.h2(f - 9) if f < 18 else self.h3(f - 18) if f < 27 else self.h4(f - 27)) if f < 36 else self.bloom(f)
