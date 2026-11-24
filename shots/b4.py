"""b4: Gojo over a manga page.  Panels slam on 0 / 9 / 18, the page tears open on 27 and the colour Gojo rises through it,
the half smile comes at 45, at 54 he lifts a bag of mochi and the aura pops into sparkles."""
import numpy as np, cv2, math, os
from jj import W, H as HH, comp, kit, fx, manga as MG, letters as LT
from jj.paint import rgb, f32, smooth, fbm, grid, lum
from jj.ink import Cel, spl, resample
from chars.gojo import Gojo, gojo_head, PAL, GC, rim
from chars import gojo_keys as GK
from chars.gojo_cache import Plates

VER = 'b4_v6'
INK = '#100d12'
PAPER = '#d9d6d0'
MG_ = 150                 # margin above and below the figure drawing (he overshoots)

# page coordinates.  (name, quad, frame it slams, options, sfx)
PANELS = [
    ('explosion', [(762, -34), (1954, -34), (1954, 524), (672, 452)], 0, dict(at=(0, .02)), ('ドン', [(1440, 26), (1770, -14), (1744, 418), (1398, 446)], False)),
    ('profile', [(-34, -34), (738, -34), (620, 640), (-34, 588)], 9, dict(at=(.02, -.04)), ('ゴゴゴ', [(40, 60), (214, 40), (190, 520), (24, 548)], True)),
    ('street', [(640, 476), (1268, 510), (1190, 1114), (556, 1114)], 18, dict(at=(0, -.05)), None),
    ('hand', [(1292, 512), (1954, 548), (1954, 1114), (1212, 1114)], 18, {}, ('ガッ', [(1640, 566), (1880, 606), (1800, 866), (1590, 826)], False)),
    ('eye', [(-34, 612), (610, 662), (572, 860), (-34, 838)], 20, dict(zoom=1.12), None),
    ('fist', [(-34, 862), (566, 886), (520, 1114), (-34, 1114)], 22, dict(at=(0, -.15), zoom=1.1), None),
]
# camera states of the first 24 frames: every slam pulls back.  (first frame, zoom, page point at frame centre, rotation)
CAMS = dict(A=(0, 2.46, (1392., 214.), -.075), B=(9, 1.70, (628., 306.), .035), C=(18, 1.02, (960., 540.), 0.))


def cam_m(z, c, r):
    a, b = math.cos(r) * z, math.sin(r) * z; return np.array([[a, -b, W / 2 - a * c[0] + b * c[1]], [b, a, HH / 2 - b * c[0] - a * c[1]], [0, 0, 1.]])


def _art(name):
    if name == 'profile': return lum(GK.art_profile())[0:700, 0:960]
    if name == 'hand': return lum(GK.art_hand())[300:1080, 0:760]
    return MG.art(name)


def _tr(img, M):
    return cv2.warpAffine(img, np.float32(M), (W, HH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def scale_about(c, s, rot=0., d=(0, 0)):
    a, b = math.cos(rot) * s, math.sin(rot) * s
    return [[a, -b, c[0] + d[0] - a * c[0] + b * c[1]], [b, a, c[1] + d[1] - b * c[0] - a * c[1]]]


class Shot:
    id = 'b4'; n = 72
    finish = dict(bloom=.24, vig=.14, soft=.10, halation='#d8ecff')
    EVENTS = [(0, 'panel_slam', dict(size=1.0, word='DON', note='manga panel slams, paper thud, ink dust')), (9, 'panel_slam', dict(size=.8, word='GOGOGO', note='second panel, low rumble lettering')),
              (18, 'panel_slam', dict(size=1.0, word='GA', note='two panels')), (20, 'panel_slam', dict(size=.35, note='small panel')), (22, 'panel_slam', dict(size=.35, note='small panel')),
              (25, 'paper_strain', dict(note='page bulges')), (27, 'paper_rip', dict(size=1.0, note='page tears open, aura whoosh, he rises')), (33, 'cloth_settle', dict(size=.4)),
              (27, 'aura', dict(end=53, note='cursed energy loop, licking flames')), (45, 'smile', dict(size=.3, note='half smile, tiny tick')),
              (52, 'anticipation', dict(size=.3, note='he dips, aura sucks in')), (54, 'pop', dict(size=.8, note='aura pops into sparkles, comedy')), (52, 'paper_bag', dict(size=.6, end=56, note='bag of mochi rises from behind his back, rustle')), (58, 'sparkle', dict(size=.5)), (63, 'sparkle', dict(size=.3)), (60, 'boing', dict(size=.3, note='bag settles, one mochi wobbles'))]

    # ---------------------------------------------------------------- plates
    def setup(self):
        self.L = Plates(VER, self._build)
        self._memo = {}
        self._memo = {}
        # tear line, top to bottom: x(y)
        r = np.random.default_rng(8); y = np.linspace(-60, HH + 60, 64); x = 1110 - .12 * (y - 500) + np.cumsum(r.normal(0, 13, 64)) * .9 + 22 * r.standard_normal(64) * (r.random(64) < .55) + 60 * np.sin(y / 190. + 1.) * .5
        self.tear = np.c_[x, y]

    def _build(self):
        d = {}; tp = lambda M, q: (np.c_[np.asarray(q, float), np.ones(len(q))] @ M.T)[:, :2]
        for st, (f0_, z, c_, r_) in CAMS.items():
            M = cam_m(z, c_, r_); last = {'A': 8, 'B': 17, 'C': 60}[st]
            for i, (name, q, f0, k, s) in enumerate(PANELS):
                if f0 > last: continue
                Q = tp(M, q); S = MG.Sheet(seed=i + 3); S.panel(_art(name), Q, border=7. * z ** .7, **k); g = S.rgb(PAPER); m = cv2.dilate(kit.mask_of(Q), np.ones((9, 9))); d[f'{st}_p{i}'] = np.dstack([g * m[..., None], m]).astype(np.float16)
                if s is not None: c = Cel(); LT.sfx(c, s[0], tp(M, s[1]), seed=11 + i, vertical=s[2], halo='#e6e3dc', halo_w=9 * z ** .6, rag=1.2); d[f'{st}_s{i}'] = c.layer().astype(np.float16)
            # ink thrown across the gutters by the slams
            c = Cel(); r = np.random.default_rng(int(z * 100))
            for (gx, gy) in {'A': [(700, 470), (1960, 500)], 'B': [(700, 470), (640, 640), (1000, 480)], 'C': [(700, 470), (636, 640), (1270, 506), (560, 872), (1210, 1090)]}[st]:
                p = tp(M, [(gx, gy)])[0]; kit.splatter(c, p[0], p[1], 120 * z ** .6, 46, 15 * z ** .5, INK, int(gx + gy), ang=r.uniform(-.6, .6), aniso=1.2); kit.splatter(c, p[0], p[1], 46 * z ** .6, 16, 30 * z ** .5, INK, int(gx))
                for k in range(5): a = r.uniform(-.5, .5) + (math.pi if r.random() < .5 else 0); L = r.uniform(60, 220) * z ** .6; c.line(np.array([p, p + (math.cos(a) * L * .5, math.sin(a) * L * .5 + r.normal(0, 8)), p + (math.cos(a) * L, math.sin(a) * L)]), r.uniform(4, 17) * z ** .5, INK, taper=(.1, .9), press=.5, seed=k)
            d[st + '_ink'] = c.layer().astype(np.float16)
        # the page before, showing through the paper (mirrored, pale)
        S = MG.Sheet(seed=9)
        for nm, q, k in (('fist', [(-20, -20), (560, -20), (610, 520), (-20, 470)], {}), ('eye', [(600, -20), (1400, -20), (1380, 360), (650, 410)], {}), ('explosion', [(1430, -20), (1940, -20), (1940, 700), (1400, 640)], {}), ('street', [(-20, 500), (900, 560), (860, 1100), (-20, 1100)], {}), ('fingers', [(930, 460), (1360, 420), (1330, 1100), (890, 1100)], {}), ('eye', [(1390, 690), (1940, 730), (1940, 1100), (1350, 1100)], {})): S.panel(MG.art(nm), q, **k)
        g = cv2.GaussianBlur(S.g[:, ::-1], (0, 0), 1.6)[..., None]; n = fbm(HH, W, (300, 90), (1, .5), 2); pap = rgb('#dedbd4') * (1 + .012 * n[..., None]); d['ghost'] = (pap * (1 - (1 - g) * .30) + rgb('#8a8794') * (1 - g) * .0).astype(np.float16)
        d['burst'] = self._burst().astype(np.float16); d['cream'] = self._cream().astype(np.float16); return d

    def _burst(self):
        x, y = grid(); cx, cy = 1180, 520; d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        img = np.zeros((HH, W, 3), f32) + rgb('#0e0a2e'); img = img + np.clip(1 - d / 1250, 0, 1)[..., None] ** 1.5 * rgb('#2a1a8c') + np.clip(1 - d / 560, 0, 1)[..., None] ** 2 * rgb('#2b7dff') * .9
        c = Cel(); kit.focus_lines(c, cx, cy, 330, 1500, 180, 10, '#070518', 4, ry=.85); kit.focus_lines(c, cx, cy, 420, 1500, 60, 6, '#7fd8ff', 9, ry=.85)
        return comp.over(img, c.layer())

    def _cream(self):
        x, y = grid(); img = np.zeros((HH, W, 3), f32) + rgb('#ffe9b8'); img = img * (1 - smooth(200, 1300, np.sqrt((x - 1080) ** 2 + (y - 430) ** 2))[..., None] * .10)
        t = kit.screentone(np.ones((HH, W), f32), 22, .6, np.clip(.05 + .22 * smooth(250, 1000, y), 0, .4)); return img * (1 - t[..., None] * .5) + rgb('#ffc98a') * t[..., None] * .5

    # ---------------------------------------------------------------- the page at frame f
    def page(self, f):
        st = 'A' if f < 9 else 'B' if f < 18 else 'C'; f0_, z, c_, r_ = CAMS[st]; Mst = cam_m(z, c_, r_); ti = f - f0_; pk = {'B': 'A', 'C': 'B'}.get(st)
        u = {0: .45, 1: .12, 2: .015}.get(ti, 0.) if pk else 0.                     # every slam knocks the camera back from the previous framing
        if u: z0, c0, r0 = CAMS[pk][1:]; Mf = cam_m(math.exp(math.log(z0) * u + math.log(z) * (1 - u)), np.array(c0) * u + np.array(c_) * (1 - u), r0 * u + r_ * (1 - u))
        else: Mf = Mst
        warp = lambda a, M: cv2.warpAffine(a, np.asarray(M)[:2].astype(np.float32), (W, HH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT); V3 = lambda m: np.vstack([np.asarray(m, float)[:2], [0, 0, 1]])
        img = cv2.warpAffine(self.L['ghost'], Mf[:2].astype(np.float32), (W, HH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101); dust = []
        if u:                                                                        # what the previous framing held, under the new drawing of the same page
            Dp = Mf @ np.linalg.inv(cam_m(*CAMS[pk][1:]))
            for i, (name, q, f0, k, s) in enumerate(PANELS):
                if f0 < f0_:
                    if f'{pk}_p{i}' in self.L: img = comp.over(img, warp(self.L[f'{pk}_p{i}'], Dp))
            for i, (name, q, f0, k, s) in enumerate(PANELS):
                if f0 < f0_ and f'{pk}_s{i}' in self.L: img = comp.over(img, warp(self.L[f'{pk}_s{i}'], Dp))
        D = Mf @ np.linalg.inv(Mst) if u else np.eye(3); tp = lambda q, M=Mst: (np.c_[np.asarray(q, float), np.ones(len(q))] @ M.T)[:, :2]
        for i, (name, q, f0, k, s) in enumerate(PANELS):
            if f < f0: continue
            t = f - f0; L = self.L[f'{st}_p{i}']; q_ = tp(q, Mf); c = tp(q).clip([0, 0], [W, HH]).mean(0); m = D
            if t < 4:
                sc = (1.36, 1.12, .970, 1.010)[t]; rot = (.045, .012, -.004, 0)[t] * (1 if i % 2 else -1); m = D @ V3(scale_about(c, sc, rot))
            if m is not D or u: L = warp(L, m)
            if t < 4:
                if t == 0: L = kit.smear(L, 0, 26, 7) * .96
                sh = comp.blur(_tr(L[..., 3], [[1, 0, 18 * (1.6 - min(t, 1.5))], [0, 1, 26 * (1.6 - min(t, 1.5))]]), 9); img = img * (1 - .5 * sh[..., None])
            img = comp.over(img, L)
            if t < 9: dust.append((i, t, q_.clip([0, 0], [W, HH]).mean(0), q_))
            if f'{st}_s{i}' in self.L and t >= 1:
                S = self.L[f'{st}_s{i}']; sc = {1: 1.5, 2: 1.12, 3: .95, 4: 1.02}.get(t, 1.0); cs = tp(s[1]).mean(0)
                if sc != 1.0 or u: S = warp(S, D @ V3(scale_about(cs, sc)))
                img = comp.over(img, S)
        if ti >= 1: img = comp.over(img, warp(self.L[st + '_ink'], D) if u else self.L[st + '_ink'])
        if dust:
            cel = Cel()
            for i, t, c, q in dust:
                r = np.random.default_rng(i * 91 + 5); e = 1 - (1 - min(1, (t + 1) / 7)) ** 2
                for _ in range(int(70 if PANELS[i][2] in (0, 9, 18) else 30)):
                    k = r.integers(4); v = r.random(); p = q[k] * (1 - v) + q[(k + 1) % 4] * v; p = np.clip(p, [-10, -10], [W + 10, HH + 10]); d = p - c; d = d / (np.hypot(*d) + 1e-6); L = r.uniform(20, 170) * e; pp = p + d * L + (0, 40 * (t / 9.) ** 2 * r.random())
                    R = (1 - t / 9.) * r.uniform(1.2, 8) * r.random() + .6; a = np.linspace(0, 6.283, 9, endpoint=False); cel.fill(np.c_[pp[0] + np.cos(a) * R * r.uniform(.8, 1.5), pp[1] + np.sin(a) * R], INK)
            img = comp.over(img, cel.layer())
        if pk and ti == 0: img = kit.zoomblur(img, W / 2, HH / 2, .05, 7)
        return img

    def _page_full(self):
        if 'full' not in self._memo: self._memo['full'] = self.page(60)
        return self._memo['full']

    # ---------------------------------------------------------------- the two halves of the torn page
    def halves(self, f):
        """0..: opening.  returns list of (layer) premult RGBA for left and right half, moved"""
        k = {25: .012, 26: .03, 27: .25, 28: .62, 29: .95, 30: 1.12, 31: 1.10, 32: 1.04, 33: 1.0}.get(f, 1.0 if f > 33 else 0.)
        k *= (1 + .010 * (f - 33) ** .9 + .012 * math.sin((f - 33) * .16)) if f > 33 else 1                # the halves keep drifting apart
        pg = self._page_full() * .80
        out = []
        for sg in (-1, 1):
            T = self.tear; far = np.array([(-400, HH + 60), (-400, -60)]) if sg < 0 else np.array([(W + 400, HH + 60), (W + 400, -60)])
            poly = np.vstack([T, far]); m = kit.mask_of(poly, ss=1)
            lay = np.dstack([pg * m[..., None], m])
            c = Cel(); n = np.c_[np.ones(len(T)), np.zeros(len(T))] * sg * -1; r = np.random.default_rng(5 + sg); w = 9 + 13 * r.random(len(T)); E = T + n * w[:, None]      # white torn fibre
            c.fill(np.vstack([T, E[::-1]]), '#f7f4ee'); c.line(T, 2.2, '#8a8680', taper=(0, 0), seed=4); fib = c.layer(); lay = lay * (1 - fib[..., 3:4]) + fib
            dx = sg * 330 * k; piv = (1030 - 60 * sg, HH * 1.15 if True else 0); M = scale_about((1110 + sg * 200, 1250), 1 + .03 * k, sg * .17 * k, (dx * .55, 8 * k)); lay = _tr(lay, M)
            # curl: the edge near the tear is lit then shaded
            out.append((sg, lay))
        return out, k

    # ---------------------------------------------------------------- Gojo
    def gojo(self, f):
        """drawing on twos; returns (layer, shape key)"""
        rise = {27: 620, 28: 300, 29: 60, 30: -34, 31: -22, 32: -4, 33: 6, 34: 3}.get(f, 0.)
        fd = f // 2 * 2 if f > 33 else f
        com = f >= 54
        ph = {52: 14, 53: 22, 54: -26, 55: -14, 56: 8, 57: 4}.get(f, 0.)
        key = (fd if not 52 <= f <= 58 else f, com, f in (52, 53), f >= 56, f >= 45, f >= 48); pos = (0., rise + ph)
        if key not in self._memo or key[0] < 34:
            t = (f - 45) / 6.
            mouth = 'flat' if f < 45 else ('sad' if f < 47 else 'half')
            expr = dict(band=True, mouth=mouth if not com else ('smile' if f < 56 else 'laugh'), flip=False)
            roll = -.07 - (.02 if 45 <= f < 54 else 0); yaw = -.10
            if com: roll = {54: -.02, 55: .10, 56: .20, 57: .18}.get(f, .17); yaw = -.04
            if f in (52, 53): roll = -.10
            wind = (.16 * math.sin(fd * .42) + .05 * math.sin(fd * 1.1), .16 + .10 * math.sin(fd * .8)) if 27 <= f < 54 else (.05 + .07 * math.sin(fd * .36), .05 + .04 * math.sin(fd * .55))
            if f in (27, 28, 29): wind = (0., -.30)
            lay = self._figure(fd, expr, yaw, roll, wind, com, f)
            for q in [q for q in self._memo if isinstance(q, tuple)]: self._memo.pop(q)
            self._memo[key] = lay
        return self._memo[key], pos

    def _figure(self, f, expr, yaw, roll, wind, com, fr):
        cx, cy, s = 1226, 452 + MG_, 288
        bagk = {52: .16, 53: .34, 54: .56, 55: .84, 56: 1.12, 57: 1.05, 58: .98}.get(fr, 1.0) if fr >= 52 else 0.; behind = fr < 55
        bob = 5 * math.sin((fr - 58) * .8) * math.exp(-(fr - 58) / 5.) if com and fr >= 58 else 0.
        def put_bag(c, B, pal):
            bx, by = cx + (1.06 if behind else 1.42) * s, cy + 1.95 * s - 1.12 * s * bagk + bob - (.30 * s if behind else 0); w2 = np.array([bx + .16 * s, by + .74 * s])
            bag(c, bx, by, .60 * s, math.radians(6 - 18 * (1 - bagk) + 3 * math.sin(fr * .5) * (fr > 58)), seed=2, wob=bob + (2. * math.sin(fr * .45) if fr > 58 else 0))
            GC.sleeve(c, B.P3((1.30, -.66, .10)), B.P3((1.95, -2.3, .7)), w2, .50 * s, .44 * s, .30 * s, pal, s, seed=31, lit=1, hand_fn=lambda: Gojo.hand(c, 'grip', bx + .10 * s, by + .34 * s, .58 * s, 20, 6, -8, pal, 'L', 5, w=4.8))
        def under(c, H, B, pal):
            if bagk and behind: put_bag(c, B, pal)                    # still behind his shoulder
        def over(c, H, B, pal):
            # crossed fingers, picture left (hand high beside the face)
            wr = np.array([cx - 1.62 * s, cy + 1.02 * s + (0 if not com else 70)]); up = (0, 0)
            GC.sleeve(c, B.P3((-1.40, -.66, .22)), B.P3((-1.92, -2.15, .85)), wr, .50 * s, .44 * s, .30 * s, pal, s, seed=21, lit=-1,
                      hand_fn=lambda: Gojo.hand(c, 'cross', wr[0] + .06 * s, wr[1] - .50 * s, .64 * s, 160, 4, 4, pal, 'R', 3, w=5.0))
            if bagk and not behind: put_bag(c, B, pal)
        pal = dict(PAL['day'])
        if com: pal.update(skin='#fbe2cf', coat_lt='#5d4a94')
        L = Gojo.bust(cx, cy, s, yaw=yaw, pitch=.08 if not com else .0, roll=roll, state='up', expr=expr, light=pal, key='L' if not com else None, f=f, wind=wind, body=(-.70, .08, .05), shift=(.14, .0), seed=5, left=False, right=not (bagk > 0), over=over, under=under, h=HH + 2 * MG_)
        return L

    # ---------------------------------------------------------------- frame
    def frame(self, f):
        x, y = grid()
        if f < 25:
            img = self.page(f)
            a = 0.
            for (_, _, f0, _, _), amp in zip(PANELS, (15, 12, 17, 0, 7, 7)):
                if f >= f0: a = max(a, amp * math.exp(-(f - f0) / 2.2))
            d = kit.shake(f, a, 3)
            z = 1.015 + .0035 * (f - (0 if f < 9 else 9 if f < 18 else 18))
            img = comp.cam2d(img, W / 2 + d[0], HH / 2 + d[1], z, d[0] * .0006)
            u = float(smooth(21, 25, f)); return img * (1 - u) * .93 + img * rgb('#b7b2c8') * .80 * u
        com = f >= 54
        base = self.L['cream' if com else 'burst']
        if com: base = comp.cam2d(base, 960 + 2.2 * (f - 54), 540 - 1.4 * (f - 54), 1.04)
        if not com:
            base = comp.cam2d(base, 1180 + 3 * math.sin(f * 2.1), 520 + 3 * math.cos(f * 1.7), 1.03 + .012 * (f % 2) + .004 * max(0, f - 30), .004 * max(0, f - 30))
        hv, k = self.halves(f)
        for sg, lay in hv:
            sh = comp.blur(_tr(lay[..., 3], [[1, 0, -sg * 26], [0, 1, 20]]), 14); base = base * (1 - .55 * sh[..., None])
            tint = rgb('#b7b2c8') if not com else rgb('#f2e6d2')
            l = lay.copy(); l[..., :3] *= tint
            # light from the opening on the paper near the tear
            base = comp.over(base, l)
        if not com:
            g = np.exp(-(np.abs(x - (1150 - .10 * (y - 500))) / 430.) ** 2) * smooth(.1, 1., k); base = comp.screen(base, g[..., None] * rgb('#3a2bd0') * .42)
        L, pos = self.gojo(f) if f >= 27 else (None, None)
        if L is not None:
            L = cv2.warpAffine(L, np.float32([[1, 0, pos[0]], [0, 1, pos[1] - MG_]]), (W, HH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            al = L[..., 3]
            if not com:
                pw = smooth(27, 31, f) * (1 - smooth(51, 54, f) * .75)
                base = comp.screen(base, comp.blur(al, 80)[..., None] * rgb('#4a1cff') * .75 * pw)
                bk, fr = fx.cursed(L, f, size=150 * pw + 10, seed=3, pal=fx.VIOLET, amount=.95, wisps=1.2)
                for q in (bk, fr):
                    for a_ in q:
                        if a_ is not None: a_[1052:] *= np.linspace(1, 0, 28)[:, None, None].astype(f32)
                base = fx.put(base, bk); base = fx.put(base, fx.aura(L, r=58 * pw + 4, inner='#5ff2ff', mid='#1a8cff', outer='#7d4dff', hard=6, f=f // 2 * 2, seed=2))
                L = rim(L, 5, 1, '#37d6ff', .8, soft=.9); L = rim(L, -4, 1, '#7fe9ff', .7, soft=.8)
                L = kit.para(L, (1100, 1080), (1260, 400), [(0, '#a79be8'), (.45, '#eae2ff'), (1, '#ffffff')], 'mul'); L = kit.para(L, (620, 720), (1260, 540), [(0, '#3f62ff'), (.5, '#000000'), (1, '#000000')], 'screen', .5)
                img = comp.over(base, L); keep = (1 - comp.radial(1226, 520, 330, 380, power=.7)) * (1 - comp.radial(770, 520, 210, 260, power=.7)); fr = fx.FX(None if fr.rgba is None else fr.rgba * keep[..., None], None if fr.light is None else fr.light * keep[..., None]); img = fx.put(img, fr, .8)
                if f in (27, 28): img = kit.smear(img, 0, 60 if f == 27 else 26, 9)
            else:
                # white sticker edge, the comedy cut-out
                w = cv2.dilate(al, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))); w = comp.blur(w, 1.2); base = kit.cast(base, np.dstack([np.zeros((HH, W, 3), f32), w]), 10, 14, '#b98a52', .35, 6)
                base = base * (1 - w[..., None]) + w[..., None] * rgb('#fffaf2'); img = comp.over(base, L)
                img = self.sparkles(img, f)
        else: img = base
        a = {25: 3, 26: 6, 27: 16, 28: 11, 29: 7, 30: 4, 54: 7, 55: 4}.get(f, 0.); d = kit.shake(f, a, 7) if a else (0., 0.)
        u = max(0., f - 33) / 20. if f < 54 else (f - 54) / 18.; z = 1.02 + (.060 * u if f < 54 else .012 + .050 * u) + (.03 if f in (54,) else 0.)
        img = comp.cam2d(img, W / 2 + d[0] + 46 * u * (z - 1) / .06, HH / 2 + d[1] - 30 * u * (z - 1) / .06, z, -.004 * u if f >= 33 else 0.)
        if f == 27: img = img * .55 + .55
        return img

    def sparkles(self, img, f):
        t = f - 54; c = Cel(); r = np.random.default_rng(12)
        for k in range(26):
            a = r.uniform(0, 6.283); d0 = r.uniform(300, 900); e = 1 - (1 - min(1, (t + 1) / 8)) ** 2.4; p = np.array([1226 + math.cos(a) * d0 * 1.2 * (.45 + .55 * e), 480 + math.sin(a) * d0 * .70 * (.45 + .55 * e) + 10 * math.sin(t * .4 + k)])
            ph = r.uniform(0, 14); tw = abs(math.sin((t + ph) * .26)) ** 1.5; s = r.uniform(16, 52) * (.15 + .85 * tw) * (1.35 if t < 4 else 1.)
            if 900 < p[0] < 1560 and 60 < p[1] < 800: continue
            if s > 6: star(c, p, s, r.uniform(-.2, .2), ['#ffffff', '#fff3c4', '#ffd0e0', '#bfe8ff'][k % 4])
        if t < 8:                                   # the pop: short hand dashes flying out, little loops left behind
            for k in range(30):
                a = k / 30 * 6.283 + .1 * r.normal(); d0 = 470 + 60 * t * (1.2 - t / 12.) + 40 * r.normal(); u = np.array([math.cos(a) * 1.22, math.sin(a) * .82]); p = np.array([1226, 500]) + u * d0; L_ = (70 - 8 * t) * r.uniform(.5, 1.2)
                c.line(np.array([p, p + u * L_ * .5, p + u * L_]), 9 - t, ['#ffffff', '#9fe6ff', '#b9a0ff'][k % 3], taper=(.15, .6), press=.4, seed=k)
        L = c.layer(); img = comp.screen(img, comp.blur(L, 8)[..., :3] * .35); return comp.over(img, L)


# ------------------------------------------------------------------------------------------------ props
def star(c, p, s, rot=0., col='#ffffff'):
    """four-point hand sparkle"""
    a = np.arange(8) / 8 * 6.283 + rot; rad = np.where(np.arange(8) % 2 == 0, s, s * .20) * np.array([1, 1, .62, 1, 1.1, 1, .58, 1]); P = spl(np.c_[p[0] + np.cos(a) * rad, p[1] + np.sin(a) * rad], True, corners=range(0, 8, 2), n=4)
    c.fill(P, col); kit.outline(c, P, max(1.2, s * .07), '#7a6aa8', int(s * 10), press=.3)


def bag(c, x, y, s, rot=0., seed=0, wob=0.):
    """kraft paper bag, hand-folded top, three mochi looking out; (x, y) = centre of the bag, s = half height px"""
    cs, sn = math.cos(rot), math.sin(rot); Q = lambda P: np.stack([(p := np.asarray(P, float))[..., 0] * cs - p[..., 1] * sn, p[..., 0] * sn + p[..., 1] * cs], -1) * s + (x, y)
    ink = '#2a1712'
    # inside of the bag, then mochi, then the front
    c.fill(Q([(-.62, -.82), (.60, -.90), (.66, -.60), (-.66, -.55)]), '#7a4a2a')
    def mochi(px, py, r_, col, sh, k):
        q = np.linspace(0, 6.283, 40, endpoint=False); py = py - .02 * wob * (1 if k == 1 else .4); B = np.c_[px + np.cos(q) * r_ * 1.10 * (1 + .03 * np.sin(q * 3 + k)), py + np.sin(q) * r_ * .92 + .06 * r_ * np.cos(q * 2)]
        c.fill(Q(B), col); S = B.copy(); S[:, 0] += .18 * r_; S[:, 1] += .30 * r_; c.clip(Q(B)); c.fill(Q(np.c_[B[:, 0] * 1.0 + .10 * r_, B[:, 1] + 1.02 * r_]), sh); c.fill(Q(spl([(px - .5 * r_, py - .42 * r_), (px - .15 * r_, py - .62 * r_), (px + .05 * r_, py - .50 * r_), (px - .30 * r_, py - .30 * r_)], True, n=5)), '#ffffff')
        rr = np.random.default_rng(k)
        for _ in range(16): a, d = rr.uniform(0, 6.28), rr.uniform(0, .9) * r_; p0 = np.array([px + math.cos(a) * d, py + math.sin(a) * d * .8]); c.fill(Q(p0 + np.array([(-.008, 0), (0, -.010), (.010, 0), (0, .010)])), '#ffffff', .8)
        c.unclip(); kit.outline(c, Q(B), 3.2, ink, k + seed, press=.35)
    mochi(-.30, -.96, .30, '#f7f4ec', '#d9cfc2', 1); mochi(.32, -1.00, .29, '#bfe0a4', '#8fbf78', 2); mochi(.02, -1.10, .31, '#ffc9d6', '#e598ad', 3)
    top = [(-.70, -.66), (-.60, -.60), (-.50, -.70), (-.38, -.61), (-.26, -.70), (-.14, -.60), (-.02, -.69), (.10, -.60), (.22, -.70), (.34, -.61), (.46, -.71), (.58, -.62), (.70, -.70)]
    F = np.array(top + [(.66, -.2), (.70, .35), (.64, .95), (.1, 1.0), (-.62, .96), (-.68, .4), (-.64, -.1)])
    c.fill(Q(F), '#d9a066'); c.clip(Q(F))
    c.fill(Q([(.16, -.8), (.80, -.8), (.80, 1.1), (.30, 1.1), (.22, .4), (.30, -.1)]), '#b87f48'); c.fill(Q([(-.8, .62), (-.2, .70), (.3, .60), (.8, .70), (.8, 1.2), (-.8, 1.2)]), '#c48c54')
    c.fill(Q([(-.60, -.52), (-.36, -.30), (-.44, .30), (-.52, .02)]), '#ecbd86'); c.fill(Q([(-.1, -.5), (.06, -.22), (.0, .12), (-.08, -.16)]), '#ecbd86')
    # drawn label: a fat round mochi with a pressed flower, no lettering
    q = np.linspace(0, 6.283, 40, endpoint=False); c.fill(Q(np.c_[-.06 + np.cos(q) * .30, .28 + np.sin(q) * .29]), '#fff6e6'); c.line(Q(np.c_[-.06 + np.cos(q) * .30, .28 + np.sin(q) * .29]), 2.6, '#c23b3b', closed=True, taper=(0, 0), seed=3)
    for k in range(5): a = k / 5 * 6.283 - 1.57; c.fill(Q(spl([(-.06, .28), (-.06 + math.cos(a - .42) * .13, .28 + math.sin(a - .42) * .13), (-.06 + math.cos(a) * .19, .28 + math.sin(a) * .19), (-.06 + math.cos(a + .42) * .13, .28 + math.sin(a + .42) * .13)], True, corners=(0,), n=5)), '#e06a7a')
    c.unclip()
    for P, w in (([(-.52, -.5), (-.40, -.1), (-.46, .3)], 2.2), ([(.28, -.5), (.24, -.05), (.30, .5), (.26, .9)], 2.6), ([(-.60, .62), (-.2, .70), (.24, .62)], 2.0), ([(.02, -.5), (.06, -.3), (.02, -.1)], 1.8)): c.line(Q(spl(np.array(P), n=8)), w, '#6a3c20', taper=(.2, .6), seed=int(w * 10))
    kit.outline(c, Q(F), 4.2, ink, seed + 9, press=.35)
