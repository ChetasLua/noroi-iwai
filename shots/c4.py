"""c4: stop-time.  Four hits (0 fist, 9 Yuji's eye, 18 hammer, 27 the dog's eye), then the thumb lifts the blindfold (36-54)
and the Six Eyes open (54), constrict (61), burn to white (70-72)."""
import numpy as np, cv2, math, os
from jj import W, H as HH, comp, kit, fx, hands as HD
from jj.paint import rgb, f32, smooth, fbm, grid, lum
from jj.ink import Cel, spl, resample, noise1
from chars.gojo import Gojo, gojo_head, PAL, GC, GF, GH, rim, eye_glow, blindfold
from chars import gojo_keys as GK
from chars.gojo_cache import Plates

VER = 'c4_v9'
HOOK = HD.pose((62, 70, 34, 2), (66, 76, 38, 0), (70, 78, 40, 0), (72, 76, 38, 2), thumb=(62, 6, 30, 2, -8), cup=.1)
STEEL = dict(skin='#a9b6d8', shade='#2a3058', line='#05060e', nail='#c4d0ec', nail_shade='#39406a', light='#eef6ff', tape='#c9d2ea', tape_shade='#3a4274', band='#0a0b16')


class Shot:
    id = 'c4'; n = 72
    finish = dict(bloom=.30, vig=.20, halation='#cfe4ff', soft=.12)
    EVENTS = [(0, 'hit', dict(size=1.0, note='an open hand')), (4, 'fist_lock', dict(size=.8, note='fist locks, thumb over the fingers: knuckle crack, bandage creak')), (9, 'hit', dict(size=.9, note="Yuji's eye: heartbeat, pupil snaps")),
              (18, 'hit', dict(size=1.0, note='hammer face strikes the lens: iron clang, blue sparks')), (27, 'hit', dict(size=.9, note="the dog's eye opens in black: wet click, low growl")),
              (36, 'silence', dict(note='everything stops; cloth only')), (39, 'cloth_tension', dict(size=.3, note='thumb hooks the blindfold')), (42, 'cloth_drag', dict(size=.5, end=49, note='blindfold dragged up')),
              (48, 'cloth_whip', dict(size=.6, note='blindfold off, hair falls')), (52, 'breath', dict(size=.3)), (54, 'eye_open', dict(size=1.0, note='lashes part, glass chime, rush of light')), (58, 'crystal', dict(end=69, note='crystal hum, rings turning')),
              (61, 'iris_snap', dict(size=.8, note='pupil constricts, glass tick')), (64, 'light_rise', dict(end=72, note='build to white')), (70, 'white', dict(size=1.0))]

    def setup(self):
        os.makedirs('cache', exist_ok=True); self._memo = {}
        x, y = grid(); self.x, self.y = x, y
        self.P = Plates(VER, lambda: dict(fist0=self._fist(0), fist1=self._fist(1), fist2=self._fist(2), fist3=self._fist(3), eye0=self._yeye(0., press=.7), eye1=self._yeye(1., pr=.24), eye2=self._yeye(0., ov=.6, pr=.28), eye3=self._yeye(0., press=1.8, pr=.17), ham0=self._hammer(0), ham1=self._hammer(1), ham2=self._hammer(2), dog0=self._dog(.18, 0), dog1=self._dog(1., 0), dog2=self._dog(1., 1)))

    # ---------------------------------------------------------------- 1 the fist
    def _fist(self, k):
        """0 open hand, 1 half, 2 locked (thumb over the fingers), 3 squeezed harder"""
        pose = (HD.POSES['open'], HD.blend('open', 'fist', .55, drag=.12), HD.POSES['fist'], HD.tremble(HD.POSES['fist'], 4, 2.0))[k]
        x, y = self.x, self.y
        img = np.zeros((HH, W, 3), f32) + rgb('#04050d'); img = img + np.clip(1 - np.sqrt(((x - 1120) / 900) ** 2 + ((y - 440) / 640) ** 2), 0, 1)[..., None] ** 1.5 * rgb('#24418f') * .9
        c = Cel(); kit.focus_lines(c, 960, 520, 520, 1500, 120, 12, '#020309', 3 + k, ry=.8); kit.focus_lines(c, 960, 520, 640, 1500, 46, 5, '#8fc0ff', 13 + k, ry=.8); img = comp.over(img, c.layer())
        c = Cel(); sc = (372, 400, 452, 460)[k]; cy = (640, 640, 600, 598)[k]
        HD.draw(c, pose, HD.Cam(940, cy, sc, 4.0), pal=STEEL, light=(.80, .45, .30), w=10.5, seed=5, side='R', R=HD.euler(180, 12, -3), bld=HD.HEAVY, style=dict(tape=dict(span=(-.55, .45), pitch=.2)), res=620)
        L = c.layer(); L = rim(L, -9, 5, '#dff2ff', .95, soft=.7); L = rim(L, 5, 3, '#3a78ff', .7, soft=.9); L = kit.para(L, (300, 1000), (1350, 250), [(0, '#4a55a0'), (.5, '#c9d4f4'), (1, '#ffffff')], 'mul')
        img = img * (1 - .6 * comp.blur(np.roll(L[..., 3], 30, 1), 30))[..., None]
        return comp.over(img, L)

    # ---------------------------------------------------------------- 2 Yuji's eye
    def _yeye(self, sq=0., ov=0., pr=.20, press=0.):
        """Yuji's eye, resolved.  sq 0 glare .. 1 squint, ov overshoot (lids a touch wider), press = late tightening, pr pupil / iris"""
        c = Cel(); full = np.array([(-40, -40), (W + 40, -40), (W + 40, HH + 40), (-40, HH + 40)], float); ink = '#120406'
        S1, S0, S2, SK, HOT = '#54190f', '#2a0b0d', '#6c2517', '#c56a36', '#ffc078'
        ck = 18 * sq + 5 * press                      # the cheek is pushed up with the lower lid
        c.fill(full, S1)
        # planes inside the shadow: forehead catching a little bounce, nose side falling away
        c.fill(np.array([(700, 250), (980, 150), (1330, 96), (1960, 60), (1960, 300), (1700, 330), (1480, 300), (1180, 318), (900, 372), (640, 470)], float), S2)
        c.fill(np.array([(-40, 480), (150, 560), (250, 690), (206, 800), (330, 930), (250, 1130), (-40, 1130)], float), S0)
        c.fill(np.array([(250, 690), (430, 700), (600, 790), (520, 860), (380, 900), (330, 930), (206, 800)], float), '#431310')
        # the lit half: one hard terminator, temple -> cheekbone -> under the eye
        LT = np.array([(1960, 352), (1800, 392), (1726, 470), (1640, 520), (1596, 620), (1500, 690 - ck * .3), (1404, 716 - ck), (1250, 776 - ck), (1060, 806 - ck), (930, 850 - ck), (872, 832 - ck), (770, 930), (690, 944), (640, 1040), (560, 1130), (1960, 1130)], float)
        c.fill(LT, SK)
        c.clip(LT)
        c.fill(np.array([(560, 1130), (760, 990), (960, 1004), (1150, 1060), (1330, 1042), (1480, 1130)], float), '#a24f28')
        c.fill(np.array([(1960, 352), (1800, 392), (1740, 480), (1800, 560), (1960, 600)], float), '#a95429')
        # ONE hard light on the cheekbone
        HL = np.array([(1004, 884 - ck), (1130, 852 - ck), (1290, 822 - ck), (1440, 760 - ck), (1560, 694 - ck), (1660, 600), (1790, 556), (1960, 500), (1960, 742), (1820, 772), (1720, 858), (1600, 872), (1500, 946), (1340, 972), (1250, 940), (1120, 948)], float)
        c.fill(HL, HOT); c.fill(np.array([(1300, 842 - ck), (1450, 782 - ck), (1580, 708 - ck), (1680, 640), (1780, 606), (1730, 690), (1600, 764), (1460, 832), (1360, 862)], float), '#ffe6b8')
        for a_, b_, w_ in (((1560, 800), (1690, 716), 5), ((1610, 812), (1668, 776), 4), ((1830, 640), (1900, 596), 4)): c.line(np.array([a_, (np.array(a_) + b_) / 2 + (4, 5), b_], float), w_, '#8a2c1c', taper=(.3, .6), seed=int(a_[0]))      # scratches
        c.unclip()
        # ---- the eye
        t = np.linspace(0, 1, 9); bump = np.sin(np.pi * np.linspace(0, 1, 9)) ** .7
        du = 10 + 46 * sq - 12 * ov + 5 * press; dl = -38 * sq + 6 * ov - 7 * press; db = 26 * sq - 12 * ov + 6 * press
        I, O = np.array([432., 690.]), np.array([1484., 640.])
        U = np.array([I, (520, 628), (660, 588), (820, 560), (980, 538), (1140, 524), (1290, 528), (1404, 572), O], float) + np.c_[0 * t, bump * du]; U = spl(U, n=18)
        Lo = np.array([O, (1390, 684), (1230, 708), (1040, 716), (840, 712), (700, 706), (600, 706), (510, 704), I], float) + np.c_[0 * t, bump * dl]; Lo = spl(Lo, n=18); E = np.vstack([U, Lo[1:]])
        # socket: the lid crushed under the brow
        SO = np.array([(340, 712), (366, 620), (520, 500), (760, 430), (1040, 380), (1290, 384), (1470, 440), (1560, 520), (1500, 640), (1400, 580), (1100, 530), (760, 560), (560, 620), (430, 700)], float) + (0, db * .6); c.fill(SO, S0)
        c.fill(np.array([(1150, 470 + db), (1330, 452 + db), (1456, 500 + db * .6), (1520, 556), (1494, 610), (1400, 540 + du * .5), (1280, 490 + du)], float), '#4a160f')
        c.fill(E, '#c79c8a'); c.clip(E)
        c.fill(np.array([(400, 560), (640, 600), (700, 720), (560, 760), (400, 760)], float), '#8c5a50')
        c.fill(np.array([(1010, 740), (1086, 664), (1180, 636), (1330, 600), (1460, 590), (1530, 640), (1400, 740)], float), '#ffe9cc')      # the lit corner of the white
        ic = np.array([946., 596. + 4 * sq]); R = 143.; q = np.linspace(0, 2 * np.pi, 120, endpoint=False); cc = np.c_[np.cos(q), np.sin(q)]; r = np.random.default_rng(5)
        for rr_, col in ((1.0, '#1d0706'), (.92, '#5a220c'), (.80, '#8e3d12'), (.60, '#b4571a')): c.fill(cc * R * rr_ * (1, 1.01) + ic, col)
        c.clip(cc * R * .93 + ic); c.fill(cc * R * .86 + ic + (R * .16, R * .50), '#d8781f'); c.fill(cc * R * .60 + ic + (R * .24, R * .66), '#ffb84a')
        for i in range(64): a = r.uniform(0, 6.283); r0, r1 = R * r.uniform(.30, .5), R * r.uniform(.66, .96); wd = r.uniform(.012, .045); c.fill(np.array([(math.cos(a - wd) * r0, math.sin(a - wd) * r0), (math.cos(a) * r1, math.sin(a) * r1), (math.cos(a + wd) * r0, math.sin(a + wd) * r0)]) + ic, ['#3a1408', '#e08a30', '#6e2c10'][int(r.integers(3))], .62)
        c.unclip()
        c.fill(cc * pr * R + ic, '#0a0202'); c.line(np.vstack([cc, cc[:1]]) * R * .985 + ic, 11, '#140405', taper=(0, 0), press=.45, seed=3)
        dn = U + np.c_[np.zeros(len(U)), np.interp(np.linspace(0, 1, len(U)), [0, .2, .5, .8, 1], [10, 40, 52, 34, 8]) * (1 - .35 * sq)]; c.fill(np.vstack([U, dn[::-1]]), '#2a0808', .80)      # cast shadow of the lid
        c.fill(np.array([(1004, 626), (1050, 606), (1056, 628), (1016, 650)], float) + (0, 8 * sq), '#fffaf0')
        c.unclip()
        # upper lid: heavy, pressed, thickest over the iris and the outer third
        tt = np.linspace(0, 1, len(U)); nr = np.gradient(U, axis=0); nr = np.c_[nr[:, 1], -nr[:, 0]]; nr /= np.hypot(*nr.T)[:, None]; nr *= -np.sign(nr[len(nr) // 2, 1]); th = np.interp(tt, [0, .05, .25, .5, .72, .9, 1], [8, 18, 42, 46, 48, 34, 14]); wing = O + (34, 2)
        c.fill(np.vstack([U, [O + (12, 10)], [wing], (U + nr * th[:, None])[::-1]]), ink)
        c.line(np.array([I + (10, -4), I + (-26, 14), I + (-58, 40)]), 14, ink, taper=(.05, .8), seed=15)      # tear duct pulled to the nose
        c.line(Lo[:int(len(Lo) * .93)], 13, ink, taper=(.04, .3), press=.5, seed=5); c.line(Lo[int(len(Lo) * .10):int(len(Lo) * .42)] + (0, 13), 5, '#ffd39a', taper=(.3, .5), press=.4, seed=6)      # wet edge of the raised lid
        # the raised lower lid: its pouch, two creases
        c.line(spl(np.array([(560, 772), (720, 792), (900, 800), (1090, 790), (1270, 752), (1400, 700)], float) + (0, dl * .7), n=10), 10 + 4 * sq, '#2a0b0d', taper=(.35, .55), press=.5, seed=7)
        c.line(spl(np.array([(1290, 800), (1420, 752), (1520, 690)], float) + (0, dl * .6), n=10), 6 + 3 * sq, '#7c3018', taper=(.3, .6), press=.4, seed=17)
        # lid fold, squeezed flat, swallowed by the brow at the nose end
        fo = U + nr * (th + np.interp(tt, [0, .3, .6, 1], [4, 18, 34, 24]))[:, None]; c.line(fo[int(len(U) * .34):int(len(U) * .95)], 7, '#7a2a1a', taper=(.5, .4), press=.5, seed=16)
        # brow: one blunt mass driven down to the nose
        B = spl(np.array([(268, 610), (400, 566), (560, 520), (780, 462), (1040, 408), (1320, 372), (1500, 378), (1640, 440)], float) + np.c_[0 * t[:8], np.interp(t[:8], [0, .4, 1], [1.15, 1., .5]) * db], n=14); tb = np.linspace(0, 1, len(B)); nb = np.gradient(B, axis=0); nb = np.c_[-nb[:, 1], nb[:, 0]]; nb /= np.hypot(*nb.T)[:, None]; wb = np.interp(tb, [0, .04, .16, .4, .78, 1], [26, 76, 100, 98, 62, 4]); r = np.random.default_rng(17)
        e0 = np.array([noise1(a * 9, 3) if False else math.sin(a * 23) * 3 + math.sin(a * 57 + 1) * 3 for a in tb]); c.fill(np.vstack([B + nb * (wb * .5 + e0)[:, None], (B - nb * (wb * .5 + e0[::-1])[:, None])[::-1]]), '#1f0809')
        for i in range(96):
            jx = int(r.uniform(0, .94) * len(B)); d = B[min(len(B) - 1, jx + 5)] - B[max(0, jx - 5)]; d = d / np.hypot(*d); p_ = B[jx] + nb[jx] * r.uniform(-.55, .50) * wb[jx]; L_ = r.uniform(60, 150); u_ = d + nb[jx] * r.uniform(-.34, .06); c.line(np.array([p_, p_ + u_ * L_ * .5, p_ + u_ * L_ + (0, 6)]), r.uniform(4, 12), ['#120406', '#1f0809', '#1f0809', '#5e1f1a', '#7e3226'][int(r.integers(5))], taper=(.1, .85), press=.4, seed=i)
        for i in range(10):                          # tail hairs catching the warm light
            jx = int(r.uniform(.74, .97) * len(B)); d = B[min(len(B) - 1, jx + 5)] - B[max(0, jx - 5)]; d = d / np.hypot(*d); p_ = B[jx] + nb[jx] * r.uniform(.1, .5) * wb[jx]; c.line(np.array([p_, p_ + d * 40 + (0, 4), p_ + d * 80 + (0, 14)]), 5, '#d8793c', taper=(.2, .85), seed=400 + i)
        # corrugator: the skin bunched between brow and nose
        kb = db * .8 * np.array([0, 1.])
        c.fill(np.array([(232, 318), (262, 452), (300, 590), (250, 690), (150, 560), (176, 430)], float) + kb * np.array([[.4], [.8], [1], [0], [0], [.3]]), '#3a100e')
        for P_, w0 in (([(230, 300), (250, 400), (268, 500), (306, 600)], 13), ([(352, 408), (344, 452), (352, 500), (380, 552)], 10)):      # frown furrows: wedges that open into the brow
            F = spl(np.array(P_, float) + kb * np.linspace(.4, 1, 4)[:, None], n=10); nF = np.gradient(F, axis=0); nF = np.c_[-nF[:, 1], nF[:, 0]]; nF /= np.hypot(*nF.T)[:, None]; wF = np.linspace(.5, w0 * (1 + .4 * sq), len(F))[:, None]; c.fill(np.vstack([F + nF * wF * .5, (F - nF * wF * .5)[::-1]]), '#160506'); c.line(F[6:-4] - nF[6:-4] * (wF[6:-4] * .5 + 5), 4, '#7e2c1b', taper=(.5, .4), seed=int(P_[0][0]))
        for P_, w_ in (([(412, 706), (334, 730), (286, 776), (262, 846)], 14), ([(446, 752), (392, 776), (352, 820), (338, 876)], 9)): c.line(spl(np.array(P_, float), n=10), w_ * (1 + .25 * sq), '#160506', taper=(.02, .8), press=.5, seed=int(P_[0][0]))
        # the mark on the cheek
        M = spl(np.array([(596, 846), (770, 888), (980, 892), (1190, 858), (1400, 786)], float) + (0, -ck * .8), n=12); tm = np.linspace(0, 1, len(M)); wm = np.interp(tm, [0, .06, .3, .6, 1], [10, 24, 30, 22, 2]); c.fill(np.vstack([M + np.c_[0 * wm, -wm * .5], (M + np.c_[0 * wm, wm * .5])[::-1]]), '#2c090b'); k_ = len(M) // 2; c.line(M[k_:] + (2, 21), 5, '#ffe0ac', taper=(.4, .5), seed=10); c.line(M[:k_] + (0, 21), 4, '#7a2a18', taper=(.5, .3), seed=12)
        # fringe: straight dark spikes in the upper corners
        for b0, b1, tip, col in (((300, -60), (640, -60), (392, 120), '#260a0f'), ((1640, -60), (1990, -60), (1700, 150), '#260a0f'), ((-80, -60), (250, -60), (40, 300), '#14050a'), ((120, -60), (470, -60), (214, 196), '#190609'), ((1800, -60), (1990, -60), (1884, 236), '#14050a')):
            b0, b1, tip = (np.array(v, float) for v in (b0, b1, tip)); m0, m1 = b0 * .5 + tip * .5 + (10, 0), b1 * .42 + tip * .58 + (-16, 0); c.fill(spl(np.array([b0, m0, tip, m1, b1]), True, corners=(0, 2, 4), n=8), col); c.fill(np.array([tip, m1 * .9 + tip * .1 - (4, 0), b1 * .7 + m1 * .3 - (30, 0), m1 - (18, 0)]), '#5c2022')
        img = c.rgba()[..., :3].astype(f32); x, y = self.x, self.y
        g = np.clip((x * .8 + y * .6) / 2000., 0, 1)[..., None]; return img * (.80 + .34 * g)                      # airbrush pass over the whole cel

    # ---------------------------------------------------------------- 3 the hammer, face on
    def _hammer(self, k):
        x, y = self.x, self.y; cx, cy, R = ((760, 610, 440), (770, 610, 370), (820, 600, 270))[k]
        img = np.zeros((HH, W, 3), f32) + rgb('#070713'); d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2); img = img + np.clip(1 - d / (R * 2.6), 0, 1)[..., None] ** 1.6 * rgb('#1f4fd8') * .8
        c = Cel(); kit.focus_lines(c, cx, cy, R * 1.15, 1600, 170, 13, '#020308', 21 + k, ry=.9); kit.focus_lines(c, cx, cy, R * 1.25, 1600, 60, 7, '#7fc8ff', 31 + k, ry=.9); img = comp.over(img, c.layer())
        # shaft and fist far behind (perspective), upper right
        c = Cel(); q = np.linspace(0, 2 * np.pi, 90, endpoint=False); cc = np.c_[np.cos(q), np.sin(q)]
        sh = np.array([(cx + R * .35, cy - R * .20), (cx + R * 1.55, cy + R * .60), (cx + R * 1.42, cy + R * .82), (cx + R * .10, cy + R * .12)]); 
        # the head in perspective: striking face, neck, cheek with the shaft, peen; stacked from far to near
        v = np.array([R * 1.75, -R * .80]); C0 = np.array([cx, cy]); seg = [(0., 1.0), (.20, .95), (.25, .78), (.34, .74), (.48, .80), (.84, .76), (.96, .70), (1.04, .80), (1.26, .74)]
        hx = C0 + v * .66; far = hx + np.array([R * .55, R * 2.7]); s0, s1 = hx + np.array([-R * .20, R * .30]), hx + np.array([R * .22, R * .24])
        def hull(a, b):
            Pq = np.vstack([C0 + v * a[0] + cc * R * a[1] * (.96, 1), C0 + v * b[0] + cc * R * b[1] * (.96, 1)]); return cv2.convexHull(Pq.astype(np.float32))[:, 0].astype(float)
        for a, b, col, hi in list(zip(seg[:-1], seg[1:], ('#33417e', '#141a3a', '#1c244c', '#1c244c', '#2a3670', '#1c244c', '#141a3a', '#33417e'), (1, 0, 1, 0, 1, 0, 0, 1)))[::-1]:
            if a[0] == .48:                          # the shaft leaves the cheek, the fist is on it
                pass
            h_ = hull(a, b); c.fill(h_, col); c.clip(h_)
            if hi: c.fill(h_ * 1.0 + np.array([-R * .05, -R * 1.18 * a[1]]) , '#8ea6e6', .9); c.fill(h_ + np.array([R * .02, R * .95 * a[1]]), '#080b1e', .85)
            c.unclip(); kit.outline(c, h_, 7, '#04050c', int(a[0] * 100), press=.35)
            if a[0] == .48:
                c.fill(np.array([s0, s1, far + (R * .10, 0), far - (R * .12, 0)]), '#3a2f3c'); c.fill(np.array([s0 * .42 + s1 * .58, s1, far + (R * .10, 0), far + (R * .01, 0)]), '#1e1822'); c.line(np.array([s0 * .82 + s1 * .18, (s0 * .82 + s1 * .18) * .5 + (far - (R * .085, 0)) * .5, far - (R * .085, 0)]), R * .022, '#6a5a6c', taper=(.1, .3), seed=4); c.line(np.array([s0, far - (R * .12, 0)]), 8, '#05060e', taper=(.02, .1)); c.line(np.array([s1, far + (R * .10, 0)]), 8, '#05060e', taper=(.02, .1))
                HD.draw(c, 'hammer', HD.Cam(far[0] - R * .20, far[1] - R * 1.22, R * .66, 5), pal=STEEL, light=(-.6, .6, .5), w=6, seed=4, side='R', R=HD.euler(20, 10, 74), bld=HD.FEMALE)
        # the face: chamfer ring, plane, wear
        w = 1 + .012 * np.sin(q * 5 + 1) + .008 * np.sin(q * 11)
        c.fill(cc * (R * w)[:, None] + (cx, cy), '#8fa4d8'); c.fill(cc * (R * .995 * w)[:, None] + (cx - R * .02, cy + R * .02), '#3c4a86'); c.fill(cc * R * .90 + (cx, cy), '#5d70b4'); c.clip(cc * R * .90 + (cx, cy))
        c.fill(np.array([(cx - R, cy - R), (cx - R * .1, cy - R), (cx - R * .30, cy - R * .52), (cx - R * .52, cy - R * .46), (cx - R * .56, cy - R * .12), (cx - R * .80, cy + R * .06), (cx - R, cy + R * .4)]), '#a9bfee'); c.fill(np.array([(cx - R * .86, cy - R * .50), (cx - R * .44, cy - R * .86), (cx - R * .46, cy - R * .62), (cx - R * .70, cy - R * .40)]), '#e6f2ff')
        c.fill(np.array([(cx + R * .1, cy + R), (cx + R, cy - R * .1), (cx + R, cy + R)]), '#26305f'); r = np.random.default_rng(4 + k)
        for _ in range(46): a = r.uniform(0, 6.28); p = np.array([cx, cy]) + r.normal(0, R * .40, 2); L_ = r.uniform(10, 90); c.line(np.array([p, p + (math.cos(a) * L_ * .5, math.sin(a) * L_ * .5 + 2), p + (math.cos(a) * L_, math.sin(a) * L_)]), r.uniform(1.2, 4), ['#1c254e', '#dfeaff', '#2a3568'][int(r.integers(3))], taper=(.3, .6), seed=int(r.integers(1 << 20)))
        for _ in range(12): p = np.array([cx, cy]) + r.normal(0, R * .45, 2); s = r.uniform(5, 26); a_ = np.sort(r.uniform(0, 6.28, 7)); c.fill(p + np.c_[np.cos(a_), np.sin(a_)] * s * r.uniform(.5, 1, (7, 1)), '#1f2955')
        c.unclip(); c.line(np.vstack([cc, cc[:1]]) * R * 1.0 + (cx, cy), 11, '#04050c', taper=(0, 0), press=.5, seed=3); c.line(np.vstack([cc, cc[:1]])[5:80] * R * .90 + (cx, cy), 5, '#0b0f26', taper=(.2, .2), press=.5, seed=4); c.line(cc[38:66] * R * .945 + (cx, cy), 9, '#f4fbff', taper=(.4, .4), press=.4, seed=5)
        img = comp.over(img, c.layer())
        # blue sparks off the contact
        c = Cel()
        for _ in range(120 if k < 2 else 70):
            a = r.uniform(0, 6.283); d0 = R * r.uniform(.96, 1.25 + .25 * k); L_ = r.uniform(30, 260) * (1 if k else 1.5); u = np.array([math.cos(a), math.sin(a)]); n_ = np.array([-u[1], u[0]]); p = np.array([cx, cy]) + u * d0; q_ = p + u * L_ + n_ * L_ * r.normal(0, .12)
            c.line(np.array([p, (p + q_) / 2 + n_ * r.normal(0, 5), q_]), r.uniform(2, 9) * (1.2 - .3 * k), ['#ffffff', '#bfeaff', '#4aa0ff'][int(r.integers(3))], taper=(.2, .8), seed=int(r.integers(1 << 20)))
        L = c.layer(); img = img + comp.blur(L, 10)[..., :3] * rgb('#4a9bff') * 1.3; return comp.over(img, L)

    # ---------------------------------------------------------------- 4 the dog's eye
    def _dog(self, o, k):
        x, y = self.x, self.y; cx, cy = 900, 545; img = np.zeros((HH, W, 3), f32) + rgb('#020207')
        r = np.random.default_rng(9); c = Cel(); rot = -.20; cs, sn = math.cos(rot), math.sin(rot); Q = lambda P: np.stack([(p := np.asarray(P, float))[..., 0] * cs - p[..., 1] * sn, p[..., 0] * sn + p[..., 1] * cs], -1) + (cx, cy)
        # fur: strokes that flow round the eye and back over the cheek, lit only near the eye
        for i in range(1500):
            a = r.uniform(0, 6.283); d = 150 + 880 * r.random() ** .8; p = np.array([math.cos(a) * d * 1.55, math.sin(a) * d * .80]); fl = np.array([.95, -.22 + .55 * p[1] / 500.]); fl = fl / np.hypot(*fl); t_ = np.array([-p[1], p[0]]); t_ = t_ / (np.hypot(*t_) + 1e-6) * np.sign(p[1] + 1e-6) * -1
            u = fl * .8 + t_ * .5 * math.exp(-d / 400.); u /= np.hypot(*u); L_ = r.uniform(40, 170) * (.6 + d / 800.); n_ = np.array([-u[1], u[0]]); q_ = p + u * L_ + n_ * L_ * r.normal(0, .10); near = math.exp(-((d - 150) / 330.) ** 2); b = near * r.random() ** 1.5
            col = rgb('#0b0e26') * (1 - b) + rgb('#5a86e8') * b if r.random() < .8 else rgb('#000003')
            c.line(Q(np.array([p, (p + q_) / 2 + n_ * 4, q_])), r.uniform(3, 12) * (.5 + d / 900.), col, taper=(.15, .85), press=.4, seed=int(r.integers(1 << 20)))
        img = comp.over(img, c.layer())
        c = Cel()
        I, O = np.array([-330, 92]), np.array([350, -74]); up = lambda o_: spl(np.array([I, (-230, 30 - 36 * o_), (-90, 60 - 190 * o_), (80, 50 - 226 * o_), (230, -2 - 130 * o_), O]), n=14); lo = lambda o_: spl(np.array([O, (240, 24 + 56 * o_), (90, 58 + 92 * o_), (-70, 82 + 84 * o_), (-220, 94 + 40 * o_), I]), n=14)
        U, Lo = up(o), lo(o); E = np.vstack([U, Lo[1:]])
        soc = np.vstack([U + (0, -34), (Lo + (0, 40))]); c.fill(Q(spl(soc[::6], True, n=5)), '#000002')
        c.fill(Q(E), '#10214f'); c.clip(Q(E))
        ic = np.array([12, -6]); R = 172; q = np.linspace(0, 2 * np.pi, 100, endpoint=False); cc = np.c_[np.cos(q), np.sin(q)]
        for rr_, col in ((1.12, '#071029'), (1.0, '#2a62c8'), (.92, '#5fa8f2'), (.74, '#a9dcff'), (.52, '#e9f9ff')): c.fill(Q(cc * R * rr_ + ic), col)
        for i in range(90): a = r.uniform(0, 6.283); r0, r1 = R * r.uniform(.3, .6), R * r.uniform(.7, 1.0); wd = r.uniform(.01, .04); c.fill(Q(np.array([(math.cos(a - wd) * r0, math.sin(a - wd) * r0), (math.cos(a) * r1, math.sin(a) * r1), (math.cos(a + wd) * r0, math.sin(a + wd) * r0)]) + ic), ['#1c4fb4', '#ffffff', '#0d2a70'][int(r.integers(3))], .55)
        pw = (.30 if k == 0 else .13) * R; c.fill(Q(spl(np.array([(0, -R * .80), (pw * .8, -R * .3), (pw, 0), (pw * .8, R * .3), (0, R * .80), (-pw * .8, R * .3), (-pw, 0), (-pw * .8, -R * .3)]), True, corners=(0, 4), n=8) + ic), '#01020a')
        dn = U + np.c_[np.zeros(len(U)), np.interp(np.linspace(0, 1, len(U)), [0, .4, .7, 1], [4, 46, 40, 6]) * o]; c.fill(Q(np.vstack([U, dn[::-1]])), '#040a24', .75)
        c.fill(Q(spl(np.array([(-96, -96), (-40, -110), (-28, -66), (-76, -44)]), True, corners=(0, 1, 2, 3), n=4)), '#ffffff'); c.fill(Q(spl(np.array([(70, 62), (104, 52), (112, 78), (80, 88)]), True, corners=(0, 2), n=4)), '#ffffff', .9)
        c.unclip(); c.line(Q(U), 26, '#000002', taper=(.12, .18), press=.4, seed=2); c.line(Q(Lo), 16, '#000002', taper=(.1, .3), press=.4, seed=3); c.line(Q(U + (0, -24)), 5, '#7fb4ff', taper=(.4, .3), press=.5, seed=4); c.line(Q(Lo[20:56] + (0, 18)), 4, '#8fc4ff', taper=(.4, .4), press=.5, seed=5)
        # tear line forward
        c.line(Q(np.array([I, I + (-60, 34), I + (-130, 96)])), 20, '#000002', taper=(.05, .7), seed=6)
        L = c.layer(); img = comp.over(img, L); g = kit.mask_of(Q(E), ss=1); img = img + comp.blur(g, 70)[..., None] * rgb('#3a86ff') * .5 * o + comp.blur(g, 16)[..., None] * rgb('#bfe4ff') * .16 * o
        v = np.clip(1.12 - np.sqrt(((x - cx) / 1150) ** 2 + ((y - cy) / 620) ** 2), 0, 1)[..., None]; return img * v

    # ---------------------------------------------------------------- the lift
    def lift(self, f):
        """36-53: close on the face"""
        k = f
        # lift of his left side, then the whole band; pulled off at 48
        lf = {36: 0, 37: 0, 38: 0, 39: -.03, 40: -.05, 41: .06, 42: .34, 43: .66, 44: .92, 45: 1.06, 46: 1.0, 47: 1.0}.get(k, 1.0); rs = {46: .18, 47: .52}.get(k, 0.)
        cx, cy, s = 905, 560, 420; push = 1 + .0035 * (k - 36)
        key = ('lift', k if k >= 39 else 36)
        if key not in self._memo:
            pal = dict(PAL['blue'], skin='#bcd2f4', skin_sh='#33508f', hair='#e6f2ff', white='#d6e8fb')
            yaw, pitch, roll = -.12, -.04, -.05
            hp = {47: (0, 0, 0), 48: (150, -170, 10), 49: (420, -400, 22), 50: (800, -700, 34), 51: (1250, -1050, 44)}.get(k)                # the hand, gone with the cloth
            rs2 = {47: .95}.get(k, 0.); hl = {47: .80, 48: .46, 49: .16, 50: -.04, 51: .02}.get(k, 0.)
            if k < 47:
                ex = dict(band=True, mouth='flat', lift=lf, lside=1, raise_=rs, lid='calm', open=0., brow='calm', flip=False); st = dict(state='up')
            elif k == 47:
                ex = dict(band=True, mouth='flat', lift=.5, lside=1, raise_=rs2, lid='calm', open=0., brow='calm', flip=False); st = dict(state='down', hair_lift=hl, wind=(0, .10))
            else:
                ex = dict(band=False, mouth='flat', lid='calm', brow='calm', open=0., flip=False); st = dict(state='down', hair_lift=hl, wind=(.05 * max(hl, 0), .04 * hl))
            def over(c, H, B, pal_):
                if hp is None and k >= 47: return
                u = .62 * s
                if k < 47: hy = -.36 * max(lf, 0) - .42 * rs + (.03 if lf < 0 else 0); tip = np.array([cx + (.50 + .03 * max(lf, 0)) * s, cy + (-.06 + hy) * s]); rr_ = 80; po = HOOK
                else: tip = np.array([cx + .56 * s + hp[0], cy - .78 * s + hp[1]]); rr_ = 80 - hp[2]; po = HOOK if k == 47 else 'grip'
                hc = tip + np.array([-.28, 1.06]) * u if po is HOOK else tip + np.array([.0, .3]) * u
                dv = np.array([math.sin(math.radians(rr_)), math.cos(math.radians(rr_))]); wp = hc + dv * 1.30 * u; el = wp + dv * 1.7 * s + (0, .25 * s)
                if k < 48:
                    bp = blindfold(Cel(8, 8), H, pal_, 3, lift=max(lf, 0.) if k < 47 else .5, lside=1, raise_=rs if k < 47 else rs2, poly_only=True)
                    c.ctx.save(); c.ctx.new_path(); c.ctx.rectangle(0, 0, W, HH); c.ctx.move_to(*bp[0]); [c.ctx.line_to(*q) for q in bp[1:]]; c.ctx.close_path(); c.ctx.set_fill_rule(1); c.ctx.clip()
                else:                                # the band streams from his fist, its free end still leaving the head
                    t = k - 48; P0 = hc + np.array([-.1, -.25]) * u; end = np.array([cx - .55 * s + 260 * t * t, cy - .62 * s - 170 * t * t]); P = spl(np.array([P0, P0 * .72 + end * .28 + (0, 60 + 30 * t), P0 * .4 + end * .6 + (0, -70), end + (0, 24)]), n=20); T = np.gradient(P, axis=0); T /= np.hypot(*T.T)[:, None]; N = np.c_[-T[:, 1], T[:, 0]]; v = np.linspace(0, 1, len(P)); hw = (46 + 60 * v) * (.75 + .25 * np.cos(v * 7 + t * 2))
                    c.fill(np.vstack([P + N * hw[:, None], (P - N * hw[:, None])[::-1]]), '#06070f'); c.fill(np.vstack([P + N * hw[:, None] * .9, (P + N * hw[:, None] * .35)[::-1]]), '#1b2a68'); c.fill(np.vstack([(P - N * hw[:, None] * .1)[10:50], (P - N * hw[:, None] * .45)[10:50][::-1]]), '#2b469c'); c.line(P + N * hw[:, None], 7, '#02030a', taper=(.02, .1), seed=t); c.line(P - N * hw[:, None], 6, '#5f9cff', taper=(.3, .3), press=.5, seed=t + 5)
                GC.sleeve(c, el + (.1 * s, 2.4 * s), el, wp, .55 * s, .48 * s, .34 * s, pal_, s, seed=31, lit=1, hand_fn=lambda: Gojo.hand(c, po, hc[0], hc[1], u, 0, 0, rr_, dict(pal_, skin='#a9c2ea', skin_sh='#2b4584'), 'L', 3, w=6.2, lit=(-.7, .5, .4)))
                if k < 48: c.ctx.restore()
            L = Gojo.bust(cx, cy, s, yaw, pitch, roll + (.006 * max(0, k - 47) if k < 50 else .012), expr=ex, light=pal, key='L', body=(-.10, 0, 0), seed=3, f=0, over=over, **st)
            self._memo = {q: v for q, v in self._memo.items() if q[0] != 'lift'}; self._memo[key] = L
        L = self._memo[key].copy()
        x, y = self.x, self.y
        img = np.zeros((HH, W, 3), f32) + rgb('#03040e'); img = img + np.clip(1 - np.sqrt(((x - 620) / 1000) ** 2 + ((y - 380) / 760) ** 2), 0, 1)[..., None] ** 1.6 * rgb('#12286a') * .9
        L = rim(L, 6, 2, '#cfeaff', .8, soft=.8); L = kit.para(L, (300, 200), (1500, 900), [(0, '#ffffff'), (.45, '#aebfe8'), (1, '#39457e')], 'mul')
        img = comp.over(img, L)
        # a breath of light leaking from under the lids just before they open
        if k >= 50:
            u = (k - 50) / 4.; H = gojo_head(True).pose(cx, cy, s, -.12, -.04, -.04)
            for sd in (-1, 1):
                q = H.P([(sd * GF.EC, GF.EY - .075)])[0]; img = img + comp.radial(q[0], q[1], 190, 60, power=2)[..., None] * rgb('#8fe0ff') * .55 * u
        if k in (48, 49): m_ = np.clip((x - 1050) / 260., 0, 1)[..., None]; img = img * (1 - m_) + kit.smear(img, 60 if k == 48 else 110, -56 if k == 48 else -96, 21) * m_
        z = push
        return comp.cam2d(img, 960 + (k - 36) * 1.0, 540, z)

    # ---------------------------------------------------------------- frame
    def cut(self, f):
        i, t = divmod(f, 9); P = self.P
        nm = ('fist', 'eye', 'ham', 'dog')[i]
        d = np.array(kit.shake(f, 20 * math.exp(-t / 1.6), 3 + i)); z = 1.04 + .012 * t
        if nm == 'fist':
            img = P['fist0' if t < 2 else 'fist1' if t < 4 else 'fist2' if t < 6 or t % 2 else 'fist3']
            if t in (2, 4): img = kit.zoomblur(img, 950, 560, .05, 15)
            if t == 4: d = d + np.array(kit.shake(f, 16, 9)); img = img * 1.25
        elif nm == 'eye':
            img = P['eye1' if t in (1, 2) else 'eye2' if t == 3 else 'eye0' if t < 6 else 'eye3']; z = 1.03 + .012 * t + {1: .03, 2: .045, 3: .01}.get(t, 0)      # squint on the beat (2 frames), opens past the glare, settles, tightens
            if t == 1: img = kit.zoomblur(img, 950, 590, .08, 15)
        elif nm == 'ham':
            img = P['ham0' if t < 3 else 'ham1' if t < 6 else 'ham2']
            if t in (1, 3, 6): img = kit.zoomblur(img, 960, 540, .10 if t == 1 else -.0 + .06, 9)
            z = 1.06 - .008 * t
        else:
            img = P['dog0' if t == 1 else 'dog1' if t < 5 else 'dog2']; z = 1.0 + .010 * t
        img = comp.cam2d(img, W / 2 + d[0], HH / 2 + d[1], z)
        if t == 0:                                  # one impact frame each, every one a different inversion
            src = P[{'fist': 'fist0', 'eye': 'eye0', 'ham': 'ham0', 'dog': 'dog1'}[nm]]; src = comp.cam2d(src, W / 2, HH / 2, 1.10)
            img = kit.impact(src, {'fist': 'inv', 'eye': 'red', 'ham': 'bw', 'dog': 'inv'}[nm], *{'fist': (.12, .34), 'eye': (.22, .60), 'ham': (.18, .60), 'dog': (.08, .30)}[nm], ink='#05040a', paper='#f4f6ff', accent='#c0121e')
            c = Cel(); kit.focus_lines(c, 960, 540, 620, 1500, 90, 9, '#05040a' if nm in ('fist', 'dog') else '#f4f6ff', 50 + i, ry=.8); img = comp.over(img, c.layer())
        elif t == 8 and i < 3: img = img * .35      # a blink of dark before the next hit
        return img

    def frame(self, f):
        if f < 36: return self.cut(f)
        if f < 54: return self.lift(f)
        t = f - 54
        op = {0: .0, 1: .0, 2: .58, 3: 1.06, 4: 1.10}.get(t, 1.0 if t > 5 else 1.03)
        if t in (0, 1): op = .0
        pu = float(comp.lerp_keys(t, [(0, .36), (3, .36), (5, .33), (6, .31), (7, .17), (8, .13), (9, .155), (12, .15), (18, .10)])); gl = float(comp.lerp_keys(t, [(0, 0.), (4, .06), (9, .14), (12, .30), (15, .62), (17, 1.)])); burn = float(comp.lerp_keys(t, [(0, 0.), (10, 0.), (13, .12), (15, .42), (16, .85), (17, 1.35), (18, 1.6)]))
        e = t / 18.; sc = 1.10 * (1 + .50 * e ** 1.6); cen = (900 - 40 * e, 620 - 30 * e)
        rf = {7: .9, 8: .6, 9: .35}.get(t, .0) + .25 * max(0, t - 11) / 6.
        img = GK.key2(f - 54, op, 1.0 + t / 24. * 2.0 + (1.2 if t >= 7 else 0), pu, gl, cen=cen, sc=sc, roll=-.075 + .012 * e, refr=min(1., rf), burn=burn, bare=True, sway=math.sin(t * .7) * .6 + (1.6 if t in (3, 4) else 0), cloth_dy=-10 - 30 * e, shards=0. if t < 3 else min(1., (t - 2) / 4.))
        img = GK.front_locks(img, t / 6., .80 + .35 * min(1., op)) if burn < .4 else img
        if t in (3, 7): img = kit.zoomblur(img, cen[0], cen[1], .05, 9)
        if t == 7: img = img * 1.12
        d = kit.shake(f, {2: 4, 3: 7, 4: 3, 7: 6, 8: 3}.get(t, .6 + 2.5 * max(0, t - 11) / 6.), 9); img = comp.cam2d(img, W / 2 + d[0], HH / 2 + d[1], 1.012)
        if t >= 16: img = np.maximum(img, {16: .80, 17: 1.1}[t])
        return img
