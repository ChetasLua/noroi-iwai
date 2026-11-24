"""h4 two fists.  0-17 Yuji sets his jaw / 18-35 Sukuna's laugh dies / 36-44 both coil away / 45-53 the thrust, one shared diagonal, on twos / 54 contact drawing on white /
55-57 the jolt / 57-71 the picture cracks from that touch, falls into white, the marks blow off a pale face."""
import numpy as np, cv2, math, os
from jj import W, H, comp, kit, world
from jj.cam import Camera3D
from jj.fx_act import *
from jj.paint import ramp, repaint
from jj.figure import Fig, POSES
from jj.fig_pose import add_pose, mix_pose, mirror
from chars.yuji import Yuji, YHead, getx, blend
from chars.sukuna import Suk
from chars import sukuna_fire as SF

KX, KY = 960., 640.                                         # where the fists meet
PHI = .40; U = np.array([math.cos(PHI), -math.sin(PHI)]); M = np.array([-U[1], U[0]]); K = np.array([KX, KY])
FPAL = dict(mul='#e6bfae', shmul='#9c7482', litmul='#ffdcb8'); FRIM = dict(dir=(.25, .5, -.82), col='#ff9a3e', thr=.40, amt=.88)      # h2's firelight
GAP = (560, 360, 185, 66, 0)                                # the thrust: five drawings on twos


def fx_tumble(p, s, rx, ry, rz):
    from jj.fx_air import _tumble, HIGAN
    Q, nz = _tumble(HIGAN * [1.7, 1.], p[0], p[1], s, rx, ry, rz, .4); return Q, nz


def ez(t, p=2.): t = min(max(t, 0.), 1.); return 1 - (1 - t) ** p


def u8(p, fn):
    """cached premultiplied layer, uint8, memory-mapped"""
    if not os.path.exists(p): L = fn(); np.save(p, (np.clip(L, 0, 1) * 255 + .5).astype(np.uint8)); del L
    return np.load(p, mmap_mode='r')


def f8(a): return np.asarray(a, f32) / 255.


def aim(side, yaw, pitch, dist, ang):
    """hand rotation whose wrist->knuckle axis points along screen angle ang (radians, counter-clockwise)"""
    roll = -88. if side == 'R' else 88.
    for _ in range(5):
        h = HD.Hand('fist', side, R=E3(yaw, pitch, roll), pivot=(0, .8, 0), bld=HD.HEAVY); cam = HD.Cam(0., 0., 100., dist); p0, p1 = cam.px(h.world(np.array([0, .1, 0.]))), cam.px(h.world(np.array([0, 1.02, 0.]))); d = p1 - p0; e = ang - math.atan2(-d[1], d[0]); roll += math.degrees((e + math.pi) % (2 * math.pi) - math.pi)
    return E3(yaw, pitch, roll)


class Shot:
    id = 'h4'; n = 72
    finish = dict(bloom=.30, vig=0.)                        # the vignette is applied in frame() so that the white frames stay white to the corners
    EVENTS = [(0, 'wind', dict(dur=36, note='ash wind, city holding its breath')), (7, 'jaw', dict(size=.3, note='Yuji sets his jaw, brow drops')), (9, 'flare', dict(size=.6, note='eyes flare')), (11, 'tear', dict(size=.3, note='one tear torn backward')), (18, 'laugh', dict(dur=9, note='Sukuna laughs, cut short at 27')), (27, 'stop', dict(note='the laugh fails')), (32, 'eye', dict(size=.5, note='near eye wide, silence')),
              (36, 'fire', dict(dur=9, note='fire roar, both wind away')), (38, 'cloth', dict(note='bodies coil, cloth cracks')), (42, 'creak', dict(note='held at full stretch; stones lift')), (45, 'whoosh', dict(size=.6, note='thrust begins')), (45, 'riser', dict(dur=9)), (47, 'whoosh', dict(size=.7)), (49, 'whoosh', dict(size=.8)), (51, 'whoosh', dict(size=.9)), (51, 'spark', dict(note='black hairline between the knuckles')), (53, 'touch', dict(size=.2, note='knuckles touch; silence')),
              (54, 'impact', dict(size=1.0, note='contact drawing on pure white, black star')), (55, 'boom', dict(size=1.0, note='jolt, ripple runs up both arms')), (56, 'boom', dict(size=.7)), (57, 'glasscrack', dict(size=1.0, note='cracks race out from the touch')), (58, 'glasscrack', dict(size=.8)), (59, 'shatter', dict(size=1.0, note='centre panes fly at the lens')), (61, 'shatter', dict(size=.8)), (62, 'face', dict(size=.2, note='pale face seen through the hole')), (63, 'shatter', dict(size=.7)), (63, 'petals', dict(dur=8, note='marks peel off like petals, wind to the left')), (65, 'shatter', dict(size=.5, note='outer panes tip over')), (68, 'shards', dict(size=.4, note='last pieces, glass tinkle')), (71, 'white', dict(note='pure white, ring out'))]

    def setup(self):
        k = 'cache/h4_v4_'
        def plate(name, fn):
            p = f'{k}{name}.npy'
            if os.path.exists(p): return np.load(p, mmap_mode="r")
            np.save(p, fn().astype(np.float16)); return np.load(p, mmap_mode="r")
        self.fire = plate('fire', self.paint_fire); self.bgY = plate('y', lambda: self.paint_bg('#ff6a3a', '#8a1c14', '#1a0a10', (1500, 620), 3)); self.bgS = plate('s', lambda: self.paint_bg('#5fb4ff', '#1c3c9a', '#080c1c', (420, 640), 5)); self.cells = glass_cells(KX, KY, 5, 17, (70, 165, 300, 480, 720, 1060, 1700)); self._cheek = None; self._pic = None
        [self.suk_key(n) for n in self.SUKD]; [(self.arm('Y', i), self.arm('S', i), self.body(i)) for i in range(5)]
        x, y = grid(); d = np.sqrt(((x - W / 2) / (W * .60)) ** 2 + ((y - H / 2) / (H * .70)) ** 2); self.V = (1 - .24 * np.clip(d - .5, 0, 1) ** 1.5)[..., None].astype(f32)

    def paint_fire(self):
        """h2's burning street, seen sideways between the two"""
        P = world.plate('ruin', mode='fire', view='low', w=2400, h=1350, seed=3); return np.clip(np.nan_to_num(P.img), 0, 3)

    def paint_bg(self, a, b, c, p, sd):
        """the ruined city behind a face, thrown out of focus and lit by one fist (colour a) at p"""
        x, y = grid(); P = world.plate('ruin', mode='ash', view='low', w=2400, h=1350, seed=3); im = comp.cam2d(np.clip(P.img, 0, 2).astype(f32), 620 if p[0] > 960 else 1780, 620, 1.30, 0., W, H); im = comp.blur(im, 7); g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)[..., None]; d = np.hypot((x - p[0]) / 1.2, y - p[1]); n = fbm(H, W, (300, 120, 50), None, sd)
        L = ramp(np.clip(d / 1500 + .05 * n, 0, 1), [(0, a), (.16, a), (.40, b), (.80, c), (1, c)]); out = L * (.30 + 1.05 * g) + np.exp(-(d / 330) ** 2)[..., None] * C(a) * .35; out = np.clip(out, 0, 1.1)
        return repaint(np.clip(out, 0, 1), sizes=(30, 14, 7), seed=sd, angle=np.where(y > 700, 0., np.pi / 2) + .25 * n, density=.9, aspect=5, jit=(.006, .02, .022))


    # ---------------------------------------------------------------- air
    def ashfall(self, im, f, n=60, wind=(-520., 60.), seed=3, col=('#d8cfc8', '#8a807c')):
        c = Cel(); t = f / 24.
        for i in range(n):
            z = h1(seed, i, 1); X = (h1(seed, i, 2) * (W + 300) + wind[0] * (.4 + z) * t) % (W + 300) - 150; Y = (h1(seed, i, 3) * (H + 200) + wind[1] * (.4 + z) * t + 30 * noise(i, t * 2, 0, seed)) % (H + 200) - 100; s = 1.5 + 8 * z ** 3; a = h1(seed, i, 4) * 6.28 + t * 5; c.fill(np.array([[X + math.cos(a) * s * 1.5, Y + math.sin(a) * s], [X + math.cos(a + 1.9) * s, Y + math.sin(a + 1.9) * s * .7], [X + math.cos(a + 3.3) * s * 1.3, Y + math.sin(a + 3.3) * s], [X + math.cos(a + 4.9) * s * .7, Y + math.sin(a + 4.9) * s]]), col[i % 2], .85)
        return put(im, c.layer())
    def torn(self, c, p0, d, n, L, w, f, seed, col, dark, ink='#0d0b12', spread=.5):
        """strips of cloth dragged straight back by the pressure: tapered, forked ends, a slow shiver on threes"""
        k = f // 3
        for i in range(n):
            a = math.atan2(d[1], d[0]) + spread * (i / max(1, n - 1) - .5) + .05 * math.sin(k * 1.3 + i * 2); l = L * (.55 + .6 * h1(seed, i, 1)); m = np.array([-math.sin(a), math.cos(a)]); dd = np.array([math.cos(a), math.sin(a)]); s0 = np.asarray(p0, float) + m * (i - (n - 1) / 2.) * w * .9; bow = l * .12 * (1 if i % 2 else -1); P = spl(np.array([s0, s0 + dd * l * .35 + m * bow, s0 + dd * l * .7 + m * bow * .4, s0 + dd * l]), n=10); ww = w * (.7 + .6 * h1(seed, i, 2)); pr = lambda v, ww=ww: ww * (1 - v) ** .55
            c.fill(band(P, lambda v: pr(v) + 2.6, seed + i, 0, 60, .35, sides=(1, 1)), ink); c.fill(band(P, pr, seed + i, 0, 60, .35, sides=(1, 1)), col); pth = Path(P, 4.); c.fill(band(pth.sub(0, pth.L * .8, lambda v: ww * .3), lambda v: ww * .32 * (1 - v) ** .6, seed + i + 9, 0, 60, .2), dark)
        return c
    # ---------------------------------------------------------------- 0-17: four key drawings, then the flare
    def face(self, f):
        sn = getx('snarl'); gr = getx('grief'); set_ = dict(sn, dark=.15, tears=1., well=1., glint=.5, hatch=1., hw=.190, my=-.655, mU=[-.004, .020, .024, .020, .000], mL=[-.020, -.030, -.030, -.030, -.014], tU=.020, tL=.016, jaw=.0, B=[(.094, .004), (.246, .080), (.402, .152), (.558, .186)], U=[-.080, -.030, .010, .028, .006], Lo=[-.066, -.070, -.070, -.062, -.018])
        fl = dict(set_, U=[-.060, .018, .062, .076, .030], Lo=[-.074, -.090, -.096, -.082, -.024], iris=.041, pupil=.011, glint=1., B=[(.092, -.004), (.246, .074), (.404, .154), (.560, .196)], hw=.232, mU=[.000, .056, .052, .060, .016], mL=[-.032, -.086, -.094, -.082, -.020], tU=.060, tL=.052, jaw=.05, wrink=1.)
        if f < 5: X = dict(blend(gr, sn, .42), tears=1., well=1., glint=.8)
        elif f < 7: X = dict(blend(gr, set_, .75), tears=1., well=1.)
        elif f < 9: X = set_
        else: X = dict(fl); X['hw'] = .232 + .008 * ((f // 3) % 2); X['jaw'] = .05 + .012 * ((f // 3) % 2)
        return X
    def yuji(self, f):
        t = f / 17.; x, y = grid(); hit = max(0., 1 - abs(f - 9.5) / 2.5); jx, jy = kit.shake(f, 1.4 + 1.2 * t + 5 * hit, 5); im = comp.cam2d(f16(self.bgY), W / 2 + 40 * t, H / 2, 1.05 + .03 * t + .03 * (f >= 9)); im = put(im, radial_streaks(1500, 620, 44, 520, 1800, 9 + f, '#ffc8a0', .005, .35 + .25 * hit)); s = 372 + 22 * t + 22 * (f >= 9)
        nod = (0, 0, 10, 16, 8)[0 if f < 5 else 1 if f < 7 else 3 if f < 9 else 4]; Hd = YHead().pose(820 - 14 * t + jx, 470 + nod + jy, s, .42, .16 - .05 * (5 <= f < 9), -.13 - .02 * (f >= 9)); L = Yuji.bust(Hd, self.face(f), Yuji.light('ash', side=1, amt=.10), f // 2 * 2, (-.85, .35), 4, byaw=.62, raise_=.2 + .12 * (f >= 7), damage=.6, breath=.5 + .5 * math.sin(f * .5))
        L = kit.rim(L, -13, 3, '#ff6a40', .9, .7, where=(smooth(700, 1100, x) * (y < 1070)).astype(f32)); L = kit.rim(L, -5, 1, '#ffd8a8', .9, .5, where=(smooth(760, 1150, x) * (y < 1070)).astype(f32)); L = kit.para(L, (1300, 500), (300, 620), [(0, '#ffd8c0'), (.4, '#f0c4b8'), (1, '#7a6078')], 'mul', .75 - .25 * hit); im = put(im, L)
        if f >= 8:                                           # one tear: it swells on the lid, is dragged back along the cheek, tears free
            e = np.asarray(Hd.P(np.array([[-.335, -.115]])))[0]; c = Cel(); u = (f - 8) / 9.; d = np.array([-.985, -.17]); n_ = np.array([.17, -.985])
            if f < 11: r_ = 7 + 3.2 * (f - 8); a = lin01(20)[:-1] * 6.283; Q = e + c2(np.cos(a) * r_ * 1.05, np.sin(a) * r_ * .9 + r_ * .35); c.fill(Q, '#ffe6da', .95); c.fill(e + (Q - e) * .55 + [-2, -2], '#ffffff', .8)
            else:
                g = (f - 11) / 6.; h_ = e + d * (40 + 520 * g ** 1.5) + n_ * 26 * math.sin(g * 3.1); ln = 70 + 150 * math.sin(min(1., g * 1.4) * 3.1416 * .62); P = spl(np.array([h_ - d * ln, h_ - d * ln * .45 + n_ * 7, h_]), n=10); w_ = 15 * (1 - .25 * g); pr = lambda v: w_ * (np.asarray(v) ** 1.5) * np.sqrt(np.clip(1.06 - np.asarray(v), 0, 1) / .2).clip(0, 1); c.fill(band(P, lambda v: pr(v) + 2.2, 3, 0, 60, 0., sides=(0, 0), wave=.05), '#8a3a34', .75); c.fill(band(P, pr, 3, 0, 60, 0., sides=(0, 0), wave=.05), '#ffe9de'); c.fill(band(Path(P, 3.).sub(poly_len(P) * .62, poly_len(P) * .96, lambda v: 2.5), lambda v: w_ * .3 * leaf(v, .6), 3, 0, 60, 0., sides=(0, 0)), '#ffffff')
            im = put(im, c.layer())
        im = put(im, embers(f, 46, 6, wind=(-620., -60.), near=.10)); F = np.exp(-(np.hypot(x - 1780, (y - 700) / 1.2) / 520) ** 2)[..., None]; return im + F * C('#ff5a30') * (.30 + .06 * math.sin(f * 1.3) + .20 * hit)
    # frame -> drawing: the laugh on threes (further off), a cut in at 27, then three drawings: grin / half covered / shut with one eye wide
    BROW = (-.030, .01)                                   # lane sukuna's smile key
    SUKD = {'l0': (985, 452, 500, .13, 'laugh', dict(brow=BROW)), 'l1': (990, 444, 500, .17, 'laugh', dict(brow=BROW)), 'l2': (982, 458, 500, .11, 'smile', {}), 'A': (988, 450, 500, .14, 'smile', {}), 'B': (988, 452, 500, .135, 'smile', dict(mouth=('mix', .58, .45), jaw=0., low=.2, eo=(.75, .95), iris=(1., .88))), 'C': (988, 455, 500, .125, 'smile', dict(mouth='hard', jaw=0., low=.1, eo=(.72, 1.3), iris=(1., .62)))}
    def suk_key(self, n):
        """one cached ink drawing"""
        def fn():
            from chars.sukuna import EXPR; from chars.sukuna_ink import red_rim
            cx, cy, s, ro, e, o = self.SUKD[n]; k = Suk(cx, cy, s, -.34, .06, ro, dict(EXPR[e], **o), noir=1., seed=4, accent='ice'); L = k.ink(dark=1, neck='dark', box=(0, 0, W, H)); L = red_rim(L, 6, 1, col='#9fd8ff'); L = kit.rim(L, 4, 1, '#9fd8ff', .95, .5, where=((grid()[0] < 1100) & (grid()[1] < 1040)).astype(f32)); return kit.para(L, (520, 500), (1500, 760), [(0, '#eef8ff'), (.45, '#dfe8f6'), (1, '#b4bede')], 'mul', .75)
        return u8(f'cache/h4_suk7_{n}.npy', fn)
    def suk(self, f):
        q = f - 18; x, y = grid(); laugh = q < 9; t = q / 8. if laugh else (q - 9) / 8.; n = ('l0', 'l1', 'l2')[q // 3] if laugh else 'A' if f < 30 else 'B' if f < 32 else 'C'
        jx, jy = (kit.shake(f, 2.4, 6) + [0, 7 * math.sin(q * 2.1)]) if laugh else kit.shake(f, .8, 6); im = comp.cam2d(f16(self.bgS), W / 2 - 3. * q, H / 2, 1.06 + .004 * q); im = put(im, radial_streaks(300, 640, 46, 460, 1800, 9 + f, '#b8d8ff', .005, .40))
        L = f8(self.suk_key(n)); z = 1.022 + .0020 * q; L = scale_layer(L, z, 985, 600, jx, jy - 1); im = put(im, L)
        F = np.exp(-(np.hypot(x - 60, (y - 720) / 1.2) / 560) ** 2)[..., None]; im = im + F * C('#3f8cff') * (.26 + .06 * math.sin(f * 1.3)) * (1 - .55 * np.clip(L[..., 3:4], 0, 1)); return self.ashfall(im, f, 60, (460., 60.) if laugh else (120., 20.), 8, ('#e6f0ff', '#8a98bc'))
    # ---------------------------------------------------------------- 36-44: both wind away from each other
    def coil(self, f):
        q = f - 36; x, y = grid(); k = [0., .28, .62, .92, 1.10, 1.06, 1.0, 1.02, 1.04][q]; jx, jy = kit.shake(f, 1.5 + (2.6 if q >= 6 else 0.), 8); cm = Camera3D((.0, -2.75, .34), (.05, 0, 1.18), focal=1120. + 9 * q, roll=.17); cm.cx += jx; cm.cy += jy
        im = comp.cam2d(f16(self.fire), 1200 + jx, 640 + jy, 1.10 + .008 * q, -.15, W, H); im = comp.blur(im, 1.6) * .95
        wy = add_pose(POSES['punch_wind'], root=(0, -.03, -.03), pel=(0, 0, -14), cst=(4, 0, -12)); Y = Fig('yuji').pose(mix_pose(POSES['guard'], wy, min(1.15, k))).place(-.74 - .15 * k, -.12, 0, -math.pi / 2 - .42); S = Fig('sukuna').pose(mirror(mix_pose(POSES['guard'], wy, min(1.15, k)))).place(.80 + .15 * k, .12, .46 + .10 * k, math.pi / 2 + .42)
        Sl = S.draw(cm, f, light=(-.5, .5, .55), kick=(.8, .3, -.4), rim=dict(FRIM, dir=(-.4, .4, -.8)), pal=FPAL, head_fn=SF.head_fn('cel', SF.ASH, 'cold')); cs = np.nonzero(Sl[::4, ::4, 3].max(0) > .5)[0] * 4; Sl = kit.para(Sl, (cs.min() + .20 * np.ptp(cs), 640), (cs.max(), 760), [(0, '#fff6ea'), (.42, '#ecc8b0'), (.75, '#a8706a'), (1, '#6e4450')], 'mul', .85); mS = S.mask('handL', 'farmL'); Yl = Y.draw(cm, f, light=(.5, .5, .55), kick=(-.8, .3, -.4), rim=FRIM, pal=FPAL, expr=dict(mood='rage', damage=1.)); mY = Y.mask('handR', 'farmR'); g = .4 + .6 * min(1., k)
        hp_ = S.info['hands']['L']; th_ = lin01(26)[:-1] * 6.283; mS = [c2(hp_[0] + np.cos(th_) * 30, hp_[1] + np.sin(th_) * 30)]; bS, fS = cursed(mS, f, (.8, -1.4), 60 + 130 * g, dict(BLOOD, out='#7a0818', body='#e0162c', hot='#ff5a50', core='#ffd8c8'), 8, 1., .8, light=.6); bY, fY = cursed(mY, f, (-.8, -1.4), 40 + 120 * g, CURSE, 9, 1., .8, light=.8)
        c = Cel(); ground_crack(c, ground_proj(cm, centre=(-1.15, -.1)), 1.5, min(1., q / 5.), 21, 12, rings=3, col='#140806', lip='#ffb070', side='#3a1c14', top='#6a4434', topdark='#4a2c24', lift=.30, lw=2.); im = put(im, c.layer()); im = put(im, Sl); im = put(im, bS, .92); im = put(im, fS, light=.5); im = put(im, bY); im = put(im, Yl); im = put(im, fY, light=.5)
        c = Cel()
        for i in range(40):                                  # stones lift out of the road and shiver
            X = h1(i, 1) * W; Y0 = 1010 - (40 + 480 * h1(i, 3)) * ez((q - 2 * h1(i, 4)) / 7., 2.); s_ = 5 + 30 * h1(i, 5) ** 2.4; rock(c, X + 3 * noise(f * .4, i), Y0, s_, (q * .05 + i, q * .07 + i * 2, i), i, dict(lit='#ffb070', mid='#5c3426', dark='#1c0e0c', ink='#0a0504'), (.2 if X > W / 2 else -.2, .6, .3), lw=1.8)
        im = put(im, c.layer()); return put(im, embers(f, 80, 6, wind=(40., -260.), near=.14))

    # ---------------------------------------------------------------- 45-53: one shared diagonal
    def arm(self, who, i):
        """Yuji (from lower left) or Sukuna (from upper right): fist, forearm, sleeve and upper arm to a shoulder at the frame edge, all on the axis; drawing i of five, foreshortening grows"""
        def fn():
            sg = 1. if who == 'Y' else -1.; sc = (236, 290, 350, 420, 486)[i] * (1. if who == 'Y' else .95); dist = (4.4, 3.4, 2.75, 2.25, 1.9)[i]; yaw = (99, 110, 120, 129, 136)[i]; pit = (6, 12, 18, 24, 30)[i]; side = 'R' if who == 'Y' else 'L'; yaw = yaw if who == 'Y' else 360 - yaw; R_ = aim(side, yaw, pit, dist, PHI if who == 'Y' else PHI + math.pi)
            py = dict(HD.PAL['day'], skin='#e2b89e', shade='#6b4a68', tape='#e6ded8', tape_shade='#78689a', line='#100d16', nail='#ecc8b4', nail_shade='#94708a', light='#fff0e6'); ps = dict(HD.PAL['day'], skin='#e0cbbd', shade='#664a66', line='#100d16', band='#100d16', nail='#16121a', nail_dark='#16121a', nail_shade='#16121a'); sy = dict(cloth='#1b1a2b', cloth_sh='#0c0b15', cloth_lt='#5a4a82', line='#0a090e', cast='#6b4a68'); ss = dict(cloth='#e6dcd6', cloth_sh='#94809a', cloth_lt='#fff4e8', line='#100d16', cuff_in='#3a2838', cast='#664a66', button='#e6dcd6')
            def draw(cx, cy, res, sleeve): return fist(cx, cy, sc, R_, side, dist, 'fist', py if who == 'Y' else ps, (sg * .55, .25 * sg, .6), 6.6, 4 if who == 'Y' else 9, res, who == 'Y', sleeve, sy if who == 'Y' else ss, rimdir=(.1, -1.), rimcol='#ff9a3e', deep='#3d2a4a', bands=None if who == 'Y' else ((.04, .12), (.18, .26)), y0=.30 if who == 'Y' else .44, nails='short' if who == 'Y' else 'claw')
            c0 = K - sg * U * (GAP[i] / 2 + sc * .3); L, _ = draw(c0[0], c0[1], 150, False); a = L[::3, ::3, 3] > .5; ys, xs = np.nonzero(a); al = ((xs * 3 - KX) * U[0] + (ys * 3 - KY) * U[1]) * sg; e = np.percentile(al, 99.95); fr = al > e - .30 * sc; lt = ((xs * 3 - KX) * M[0] + (ys * 3 - KY) * M[1])[fr].mean(); c1 = c0 + sg * U * (-GAP[i] / 2 - e) - M * lt; L, g = draw(c1[0], c1[1], 420, True)
            cam = HD.Cam(c1[0], c1[1], sc, dist); h = HD.Hand('fist', side, R=R_, pivot=(0, .8, 0), bld=HD.HEAVY); E = cam.px(h.world(np.array([0, -2.0, 0.]))); wE = np.hypot(*(cam.px(h.world(np.array([.46, -1.9, 0.]))) - cam.px(h.world(np.array([-.46, -1.9, 0.]))))) / 2; aE = float((E - K) @ U) * sg; aS = min(-1230., aE - 260.); E = K + sg * U * aE + M * float((E - K) @ M); Sh = K + sg * U * aS + M * (18 if who == 'Y' else -10); mid = (E + Sh) / 2; wS, wM = wE * 1.26, wE * 1.12; c = Cel()
            P = spl(np.array([E + M * wE * 1.05, mid + M * max(wM, wE * 1.15), Sh + M * wS, Sh - sg * U * 120, Sh - M * wS, mid - M * max(wM * .9, wE * 1.1), E - M * wE * 1.05]), True, (2, 4), n=9); base, shd, hi = (('#1b1a2b', '#0c0b15', '#5a4a82') if who == 'Y' else ('#e6dcd6', '#94809a', '#fff4e8'))
            if who == 'S':                                   # hanging kimono sleeve, thrown back by the thrust
                dr = spl(np.array([E - sg * U * 10 + M * wE, mid + M * (wM + 60), mid + M * (wM + 190) - sg * U * 60, Sh + M * (wS + 170), Sh + M * wS]), True, (0, 2, 3), n=8); c.fill(dr, shd); kit.outline(c, dr, 5.5, '#100d16', 3)
            c.fill(P, base); c.clip(P); c.fill(spl(np.array([E + M * wE * .25, mid + M * wM * .30, Sh + M * wS * .2, Sh + M * wS * 1.4, mid + M * wM * 1.6, E + M * wE * 1.5]), True, n=8), shd)
            r = R(i + 3, 5 if who == 'Y' else 6)
            for j in range(5): v = .12 + .17 * j; p = E + (Sh - E) * v; w_ = wE + (wS - wE) * v; q = spl(np.array([p - M * w_ * 1.0, p - M * w_ * .35 + sg * U * 26, p + M * w_ * .30 + sg * U * 14, p + M * w_ * .95 - sg * U * 30]), n=8); c.line(q[: int(len(q) * r.uniform(.5, .95))], 5.2, '#0a090e', taper=(.1, .6), seed=j, press=.5); c.fill(np.vstack([q[2:int(len(q) * .6)], (q[2:int(len(q) * .6)] - sg * U * r.uniform(14, 34))[::-1]]), hi, .85 if who == 'Y' else .7)
            c.unclip(); kit.outline(c, P, 6.4, '#100d16', 7); Bk = c.layer(); Bk = kit.rim(Bk, 0, -7, '#ff9a3e', .85, .5, where=(grid()[1] < H - 12).astype(f32)); return L + Bk * (1 - L[..., 3:4])
        return u8(f'cache/h4_arm7_{who}{i}.npy', fn)
    def body(self, i):
        """both men behind their arms: Yuji's head over his own shoulder lower left, Sukuna's over his upper right"""
        def fn():
            xx, yy = grid(); Hd = YHead().pose(318 + 9 * i, 520 - 5 * i, 184 + 8 * i, .54, .12, -.20 - .015 * i); X = dict(getx('snarl'), dark=.12, tears=.7, glint=1., hatch=1., U=[-.060, .012, .056, .070, .028], jaw=.04 + .02 * i, hw=.225 + .008 * i); B = Yuji.bust(Hd, X, Yuji.light('fire', side=1), 40 + i * 2, (-.95, .30), 4, byaw=.85, broll=-.22, raise_=.4, damage=.6, base=(190 + 8 * i, 800 - 4 * i)); B = kit.rim(B, -7, 3, '#ff5a50', .85, .6, where=(smooth(330, 600, xx) * (yy < 1066)).astype(f32)); B = cv2.GaussianBlur(B.astype(f32), (0, 0), 1.3)
            s = 150 + 7 * i; hx, hy = 1622 - 9 * i, 150 + 6 * i; S = Suk(hx, hy, s, -.50, .06, .20 + .01 * i, dict(mouth='hard', jaw=0., eo=(.80, 1.12 + .04 * i), iris=(1., .72 - .02 * i), e2=(.9, .95), brow=(-.02, .0), look=(-.25, .15), low=.1), noir=.8, seed=4, wind=.9, accent='ice'); nb = np.asarray(S.H.P3(np.array([0, -.98, -.10])), float)[:2]; T = lambda u, v: nb + np.asarray(S.rt, float) * u - np.asarray(S.up, float) * v
            c = Cel(); rb = spl(np.array([T(-1.1, .10), T(-1.7, .75), (1480, 700), (1500, 1140), (2010, 1140), (2010, 40), T(1.5, -.3), T(.9, -.2)]), True, (2, 3, 4, 5), n=8); c.fill(rb, '#b7a6b4'); c.clip(rb)
            for P in ([(1620, 560), (1700, 760), (1660, 1140), (1500, 1140), (1520, 800)], [(1800, 420), (1760, 640), (1830, 900), (1780, 1140), (2010, 1140), (2010, 300)]): c.fill(spl(np.array(P, float), True, n=6), '#5f4a68')
            for P in ([(1700, 520), (1740, 700), (1720, 860), (1680, 700)], [(1900, 360), (1880, 560), (1930, 760), (1960, 520)]): c.fill(spl(np.array(P, float), True, n=6), '#ffe0c0')
            for j, P in enumerate(([(1600, 430), (1640, 640), (1600, 900)], [(1730, 400), (1700, 620), (1750, 900), (1720, 1090)], [(1850, 330), (1820, 600), (1880, 900)], [(1940, 260), (1910, 500), (1960, 800)])): c.line(spl(np.array(P, float), n=8), 5, '#100d16', taper=(.1, .5), seed=j, press=.5)
            c.unclip(); kit.outline(c, rb, 5.5, '#100d16', 5); self_t = c.layer(); hd = S.ink(dark=1, neck='dark', head_only=True); L = hd + self_t * (1 - hd[..., 3:4]); c = Cel(); self.torn(c, T(.9, .1), np.array([.95, -.30]), 5, 420, 24, 45 + i * 3, 31, '#2b3148', '#161a2a', spread=.8)
            cowl = spl(np.array([T(-1.0, -.36), T(-.5, -.10), T(.2, -.04), T(.8, -.22), T(1.2, -.50), T(1.35, .1), T(.8, .52), T(0, .60), T(-.8, .42), T(-1.25, .05)]), True, (0, 4), n=8); c.fill(cowl, '#2a2438'); c.clip(cowl); c.fill(spl(np.array([T(-1., -.4), T(-.5, .0), T(.1, .1), T(-.2, .4), T(-.9, .3)]), True, n=6), '#6a4a5c'); c.unclip(); kit.outline(c, cowl, 5, '#100d16', 7); [c.line(spl(np.array([T(a, v), T((a + b) / 2, v + .20), T(b, v - .12)]), n=8), 4, '#100d16', taper=(.2, .5), seed=j) for j, (a, b, v) in enumerate(((-.9, .9, -.02), (-.8, 1.1, .18), (-.6, .9, .34)))]
            Cw = c.layer(); L = Cw + L * (1 - Cw[..., 3:4]); L = kit.para(L, (1400, 300), (1960, 700), [(0, '#fff0dc'), (.4, '#ecc6a8'), (1, '#7a5050')], 'mul', .85); L = kit.rim(L, 8, -3, '#ff9a3e', .9, .6, where=(yy > 14).astype(f32)); L = cv2.GaussianBlur(L.astype(f32), (0, 0), 1.3); return B + L * (1 - B[..., 3:4])
        return u8(f'cache/h4_body6_{i}.npy', fn)
    def bgaxis(self, f, t, dr):
        im = comp.cam2d(f16(self.fire), 1200, 600, 1.30 + .10 * t, -.22, W, H); im = pinch(im, KX, KY, 1400, -.30 - .22 * t, .06 * t); im = comp.blur(im, 3.0); x, y = grid(); d = np.hypot(x - KX, y - KY); im = mono(im, dr, '#b09aa0', .92, 1.12); return im * np.clip(1.12 - .5 * (d / 1250) ** 1.4, .3, 1.)[..., None] + (np.exp(-(d / 360) ** 2) * (.15 + .45 * t))[..., None] * C('#fff0dc')
    def thrust(self, f):
        i = (f - 45) // 2; q = f - 45; t = q / 8.; jx, jy = kit.shake(f, 2 + 5 * t, 8); im = self.bgaxis(f, t, .15 + .40 * t); c = Cel(ss=1); kit.speed_lines(c, -PHI, 70 + 10 * q, 6, (300, 1500), '#ffe6cc', 20 + i, alpha=.32); kit.speed_lines(c, -PHI, 40, 9, (300, 1300), '#120808', 50 + i, alpha=.55); im = put(im, c.layer()); im = put(im, f8(self.body(i))); AS, AY = f8(self.arm('S', i)), f8(self.arm('Y', i)); k_ = f // 2
        bS, fS = cursed(AS, k_ * 2, tuple(U * 3.), 92, dict(BLOOD, out='#7a0818', body='#e0162c', hot='#16040a', core='#ffb0a0'), 8, 1., .55, light=.5, speed=900); im = put(im, bS, light=.8); im = put(im, AS); del AS; bY, fY = cursed(AY, k_ * 2, tuple(-U * 3.), 104, CURSE, 5, 1., .55, light=.6, speed=900); im = put(im, bY, light=.8); im = put(im, AY); im = put(im, fS, light=.3); im = put(im, fY, light=.3); del AY
        c = Cel()
        for j in range(26): a = h1(j, 1) * 2 - 1; s_ = 4 + 26 * h1(j, 2) ** 2.4; p = K + U * (a * 1000) + M * ((h1(j, 3) - .5) * 900) + (K - (K + U * (a * 1000))) * .03 * q; rock(c, p[0], p[1], s_, (j + .2 * q, j * 2 + .15 * q, j), j, dict(lit='#ffb070', mid='#4a2c26', dark='#170c0c', ink='#0a0504'), (0, .6, .3), lw=1.8)
        im = put(im, c.layer()); im = distort_ring(im, KX, KY, 230 - 110 * t, 80, 6 + 20 * t, 3, ca=1. + 2. * t)
        c = Cel(); gp = GAP[i]
        for j in range(5): o = (j - 2) * (8 + gp * .15); hh = 190 * (.6 + .4 * t); v = np.linspace(-1, 1, 24); P = K + U * o + np.outer(v * hh, M) + np.outer((1 - v ** 2) * (-16 if o < 0 else 16) * t, U); thin(c, P, 1.4 + 1.6 * t, '#fff6ee', .25 + .4 * t, .45, .45, j, 0, 24)
        if f >= 50: e = f - 49; bolts_draw(c, bolt_poly(bolt_line(K - M * (170 + 40 * e), K + M * (180 + 44 * e), f * 3, .22, 7, 0.), 5.0 + 1.8 * e, f, edge=5.0 + 1.4 * e)); r_ = R(f, 4); [bolts_draw(c, bolt_poly(bolt_line(K, K + np.array([math.cos(a), math.sin(a)]) * l, f * 9 + int(a * 99), .5, 4), 2.4 + e, f, edge=2.8)) for a, l in zip(r_.uniform(0, 6.28, e), r_.uniform(60, 120 + 50 * e, e))]
        im = put(im, c.layer()); return comp.cam2d(im, W / 2 - jx, H / 2 - jy, 1.02 + .010 * q + .01 * (q % 2 == 0))

    # ---------------------------------------------------------------- the touch
    def jolt(self, L, sg, ph, amp, sink=46.):
        """arm layer pressed past the contact plane, cut flat on it, with the ripple of the blow running back along the limb"""
        x, y = grid(); a = (x - KX) * U[0] + (y - KY) * U[1]; b = (x - KX) * M[0] + (y - KY) * M[1]; k = 1 - amp * np.exp(-np.abs(a) / 480) * np.sin(np.abs(a) / 40 - ph); k = k / (1 + .13 * np.exp(-(a / 60) ** 2)); a2, b2 = a - sg * sink * np.clip(1 - np.abs(a) / 900, 0, 1), b * k; out = cv2.remap(L, (KX + U[0] * a2 + M[0] * b2).astype(f32), (KY + U[1] * a2 + M[1] * b2).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE); wob = 3 * np.sin(b / 23.) + 2 * np.sin(b / 9. + 1); return out * np.clip(.5 - sg * (a - wob) / 1.6, 0, 1)[..., None].astype(f32)
    def ripples(self, c, col, al, ph, seed=0):
        for sg in (-1, 1):
            for j in range(6): a = sg * (42 + 46 * j + 7 * math.sin(ph + j)); hh = (150 - 14 * j) * (.8 + .4 * h1(seed, j, sg)); v = np.linspace(-1, 1, 22); P = K + U * (a + 0) + np.outer(v * hh + 16 * (h1(seed, j, 3) - .5), M) + np.outer((1 - v ** 2) * sg * (20 + 5 * j), U); thin(c, P[int(h1(seed, j) * 6):], 4.6 - .5 * j, col, al, .3, .5, j, .3, 22)
    def contact(self):
        """f54: black on pure white. knuckles pressed flat, skin rippling, the star exactly where they touch"""
        im = np.ones((H, W, 3), f32); c = Cel(); r = R(54, 1)
        for i in range(120): a = r.uniform(0, 6.283); q0 = r.uniform(300, 470); q1 = q0 + r.uniform(80, 420) * (1 if abs(math.sin(a + PHI)) > .3 else .4); hw = .006 + .016 * r.random() ** 2; qm = q0 + (q1 - q0) * .62; c.fill(np.array([[KX + math.cos(a) * q0, KY + math.sin(a) * q0 * .8], [KX + math.cos(a - hw) * qm, KY + math.sin(a - hw) * qm * .8], [KX + math.cos(a) * q1 * 1.08, KY + math.sin(a) * q1 * .86], [KX + math.cos(a + hw) * qm, KY + math.sin(a + hw) * qm * .8]]), BK)
        im = put(im, c.layer())
        for who, sg in (('S', -1), ('Y', 1)):
            L = self.jolt(f8(self.arm(who, 4)), sg, 0., .11); a = L[..., 3:4]; g = cv2.cvtColor((L[..., :3] / np.maximum(a, .02)).astype(f32), cv2.COLOR_RGB2GRAY)[..., None]; t1 = smooth(.065, .085, g); t2 = smooth(.33, .37, g); col = (1 - t1) * 1. + t1 * (C(RED) * (1 - t2) + C(BK) * t2); im = im * (1 - a) + col * a
        c = Cel(); self.ripples(c, '#ffffff', 1., 0., 3); v = np.linspace(-1, 1, 30); c.fill(np.vstack([K + np.outer(v * 250, M) + np.outer(5 * (1 - v ** 2) + .6, U), (K + np.outer(v * 250, M) - np.outer(5 * (1 - v ** 2) + .6, U))[::-1]]), '#ffffff')
        c.fill(star_poly(KX, KY, 96, 330, 17, 4, rot=.2, skew=(PHI + math.pi / 2, .7)), '#ffffff'); ks = kite_star(KX, KY, 44, 300, 18, 8, (math.pi / 2 - PHI, 1.0), .085, 8)
        for Kk, K2 in ks: c.fill(K2, RED)
        for Kk, K2 in ks: c.fill(Kk, BK)
        c.fill(star_poly(KX, KY, 46, 104, 11, 6), BK); c.fill(star_poly(KX, KY, 7, 24, 7, 2), '#ffffff'); return put(im, c.layer())
    def hitpic(self, k):
        """k = 0, 1, 2 (f55-57): colour again, ripple travelling back along both arms, black lightning from the touch"""
        x, y = grid(); d = np.hypot(x - KX, y - KY); im = self.bgaxis(53, 1., .55) * .55 + ramp(np.clip(d / 1200, 0, 1), [(0, '#fff2e4'), (.10, '#ffb090'), (.24, '#e8283a'), (.55, '#5a0c18'), (1, '#16060a')]) * (.95 - .22 * k); im = put(im, f8(self.body(4))); im = put(im, black_flash_layers(5 + k, KX, KY, .9, 17, math.pi / 2 - PHI, 16, 1150, .3, part='back', star=k < 2))
        for who, sg in (('S', -1), ('Y', 1)): im = put(im, self.jolt(f8(self.arm(who, 4)), sg, 1.4 + 2.1 * k, .12 - .03 * k))
        c = Cel(); self.ripples(c, '#2a1420', .8, 1.4 + 2.1 * k, 3); ks = kite_star(KX, KY, 34, 330 - 60 * k, 20, 11 + k, (math.pi / 2 - PHI, .9), .07, 7)
        for Kk, K2 in ks: c.fill(K2, RED)
        for Kk, K2 in ks: c.fill(Kk, BONE if k < 1 else BK)
        c.fill(star_poly(KX, KY, 22, 60, 9, 6 + k), BONE); im = put(im, c.layer()); return np.clip(put(im, black_flash_layers(5 + k, KX, KY, .9, 17, math.pi / 2 - PHI, 16, 1150, .3, part='front', star=False)), 0, 1.2)
    def cheek(self, f):
        """white on white: the line of a cheek; the marks lift off as black petals"""
        q = f - 57; t = min(1., max(0., q / 13.5)); im = np.ones((H, W, 3), f32) * C('#fffdfb'); x, y = grid()
        if q < 0: return im
        s = 400; ex = dict(eo=(.10, .08), e2=(0., 0.), brow=(.0, -.03), mouth='flat', jaw=0., look=(0., 0.), low=0.); S = Suk(1010, 430, s, -.40, .06, .06, ex, seed=4)
        if self._cheek is None:                              # the face under the marks: pale, eyes shut, nothing written on it
            pl = dict(SF.DAY, line='#6d6074', skin='#eadfe4', deep='#dccdd6', light='#fff8f4', hair='#efd5d8', hair_d='#dcb9c2', hair_l='#fff0ee', white='#f4eef0', white_l='#fffdfb', iris='#c9a8ac', mark='#eadfe4', mark_l='#fff8f4', teeth='#f4eef0', teeth_l='#fff'); self._cheek = Suk(1010, 430, s, -.40, .06, .06, ex, seed=4, marks=0.).cel(pal=pl, ls=-1).astype(np.float16)
        fade = 1 - ez(max(0., t - .66) / .34, 1.5); im = put(im, np.asarray(self._cheek, f32), fade)
        c = Cel(); k = 0; wind_ = np.array([-1., -.22])
        for name, P in S.marks_screen()[0]:
            P = np.asarray(P, float)
            if len(P) < 4: continue
            k += 1; cen = P.mean(0); t0 = .33 + .20 * h1(k, 3) + .10 * float(np.clip((cen[0] - 700) / 700, 0, 1)); u = max(0., (t - t0) / .24); v = P @ wind_; lo, hi = v.min(), v.max(); cut = hi - (hi - lo) * min(1., u); keep = P.copy()
            if u < 1:                                        # what is still on the skin: eaten back from the windward end, with a torn edge
                over_ = v > cut; keep[over_] -= np.outer(v[over_] - cut + 5 * np.abs(noise(P[over_] @ np.array([.22, -.97]) / 9., k, 0, 3)), wind_ / (wind_ @ wind_)); c.fill(keep, '#15101a')
            m = 5 + int(5 * h1(k, 5))
            for j in range(m):                               # each bite becomes a petal: curls, tumbles, goes with the wind
                w = (j + .5) / m; tj = t0 + .24 * w; uu = (t - tj) / .46
                if uu <= 0 or uu > 1.6: continue
                i = int(np.argmin(np.abs(v - (hi - (hi - lo) * w)))); p0 = P[i] * .6 + cen * .4; sz = np.clip((hi - lo) / m * 1.9, 24, 70) * (.7 + .6 * h1(k, j, 4)) * (1 - .25 * uu); mv = np.array([-(160 + 760 * uu) * uu * (.6 + .8 * h1(k, j, 8)), (-210 * uu + 120 * uu * uu) * (.4 + 1.1 * h1(k, j, 9)) + 26 * math.sin(uu * 6 + j)]); a0 = h1(k, j, 10) * 6.28; Q, nz = fx_tumble(p0 + mv, sz, a0 + uu * 4, a0 * 1.7 + uu * 3, a0 + uu * 2.2); c.fill(Q + (Q - Q.mean(0)) * .10 + [.8, 1.2], '#c81428'); c.fill(Q, '#15101a' if nz > 0 else '#2c1a24')
        im = put(im, c.layer()); w = smooth(.90, 1., t); return im * (1 - w) + w

    def glass(self, f):
        """the picture of f57 cracks from the touch and falls"""
        if self._pic is None: self._pic = self.hitpic(2).astype(np.float16)
        t = (f - 56) / 24.; a = kit.shake(f, 20 * math.exp(-(f - 57) / 2.), 3); im = comp.cam2d(np.asarray(self._pic, f32), W / 2 - a[0], H / 2 - a[1], 1.05); out = shatter(im, t, KX, KY, self.cheek(f - 1), 5, 17, hold=.105, cells=self.cells, grav=3200., burst=960., spin=6.8, depth=1600., refr=22., delay=.23, crackw=2.2, dark=.5); w = smooth(69, 71, f); return out * (1 - w) + w

    def frame(self, f):
        f = int(f)
        if f < 18: im = self.yuji(f)
        elif f < 36: im = self.suk(f)
        elif f < 45: im = self.coil(f)
        elif f < 54: im = self.thrust(f)
        elif f == 54: return self.contact() * 1.14
        elif f < 57: a = kit.shake(f, 30 if f == 55 else 22, 17); im = comp.cam2d(self.hitpic(f - 55), W / 2 - a[0], H / 2 - a[1], 1.07, .012 * (1 if f % 2 else -1))
        else: im = self.glass(f)
        w = float(smooth(64, 71, f)); return im * (self.V * (1 - w) + w) * (1. + .35 * max(0., f - 69) / 2.)
