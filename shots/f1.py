"""F1  The prayer.  Hands surface out of black ink, the lens climbs to the closed face, eyes tear open behind; 36 the fingers lock into the sign;
45 red floods the ground; 54 his lids rise; 60-72 the smile, all the way in.  72 f."""
import numpy as np, cv2, math, os
from jj import comp, W, H, hands as HD
from jj.paint import rgb, f32, smooth, fbm
from jj.ink import Cel, spl, resample, over_roi, noise1
from jj.kit import shake, impact
from jj.comp import over, lerp_keys, ease
from chars.sukuna_ink import *
from chars.sukuna import Suk, EXPR
from chars.sukuna_body import Prayer, scarf_back, scarf_front
from chars.sukuna_eyes import InkEye, bristle
from chars.sukuna_hands import sign, ink_hands
from chars.sukuna_keys import prayer_eyes, red_ground, ov

VER = 'v1'; HI_ = 1.5; MZ = (70, 60)                     # margin round the close-up masters
BOX = (850, 40, 1900, 1118)                             # design rect of the figure master
EYE_AT = [18, 27, 22, 31]
SHUT = [62, 63, 60, 61]                                 # the ink eyes clamp shut as his own open
ESTATE = dict(s0=(0., .55), s1=(0., 1.), t1=(.20, 1.), t2=(.62, 1.), over=(1.12, 1.), set=(.92, 1.), A=(1., 1.), B=(.965, 1.), glare=(.70, 1.))
ESEQ = ['s0', 's1', 't1', 't2', 'over', 'over', 'set', 'set']
HEADK = dict(cx=1318, cy=398, s=205, yaw=-.40, pitch=.14, roll=.05)
LIDS = [.14, .34, .58, .80]


def rip(t):
    """shock ripple travelling out of the hands (t = drawing number, 0 = none)"""
    c = np.array([1196., 720.])
    def T(p):
        p = np.asarray(p, float)
        if t <= 0: return p
        d = p - c; r = np.hypot(d[..., 0], d[..., 1]) + 1e-6; a = 15. * math.exp(-.38 * (t - 1)); w = 90 + 34 * t; k = np.exp(-((r - 150 - 150 * t) / w) ** 2) * np.sin((r - 150 * t) / 34.); return p + d / r[..., None] * (a * k)[..., None] * np.array([1., .8])
    return T


class Shot:
    id = 'f1'; n = 72
    finish = dict(vig=.14, bloom=.14, grain=.011)
    EVENTS = [(0, 'ink_surface', dict(dur=16, size=.8, note='hands rise out of black ink')), (2, 'drip', dict(size=.5)), (5, 'drip', dict(size=.4, pan=-.2)), (8, 'ink_run', dict(dur=14, note='ink runs off the hands')), (9, 'drip', dict(size=.4, pan=.2)), (12, 'ink_run', dict(dur=18, size=.9, note='the black slides off the figure as the lens climbs')), (15, 'drip', dict(size=.4)),
              (18, 'brush_slash', dict(size=1., pan=-.5)), (20, 'tear', dict(size=1., pan=-.5, note='giant eye left')), (22, 'brush_slash', dict(size=.6, pan=-.7)), (24, 'tear', dict(size=.6, pan=-.7)), (27, 'brush_slash', dict(size=.8, pan=.6)), (29, 'tear', dict(size=.8, pan=.6, note='second big eye right')), (31, 'brush_slash', dict(size=.4, pan=.8)), (33, 'tear', dict(size=.4, pan=.8)),
              (30, 'anticipation', dict(dur=4, note='palms press, shoulders sink')), (34, 'whoosh', dict(dur=2, size=1., note='fingers smear')), (36, 'impact', dict(size=1., note='domain sign locks: bone crack, taiko')), (36, 'cloth_ripple', dict(dur=10, size=.8)), (37, 'ink_splatter', dict(size=1., dur=8)), (36, 'eye_dart', dict(size=.6, note='all irises snap')),
              (45, 'flood', dict(dur=7, size=1., note='red floods the ground')), (45, 'drone', dict(dur=27, note='low choir bed opens')), (54, 'lid', dict(size=.5, dur=6, note='his lids rise, four drawings')), (59, 'impact_low', dict(size=.5, note='eyes open')), (60, 'lid', dict(size=.4, pan=-.7, note='ink eye shuts')), (61, 'lid', dict(size=.3, pan=.8, note='ink eye shuts')), (62, 'lid', dict(size=.6, pan=-.5, note='ink eye shuts')), (63, 'lid', dict(size=.5, pan=.6, note='ink eye shuts')), (60, 'push_in', dict(dur=12, size=1.)), (64, 'teeth', dict(size=.6, note='lips part')), (68, 'breath', dict(size=.5, note='a laugh held in the throat'))]

    # ------------------------------------------------------------------ caches
    def _npy(self, name, fn):
        p = f'cache/f1_{VER}_{name}.npy'
        if not os.path.exists(p): tmp = p + f'.{os.getpid()}.npy'; np.save(tmp, np.clip(fn() * 255 + .5, 0, 255).astype(np.uint8)); os.replace(tmp, p)
        return p

    def _sheet(self, draw):
        with canvas(1920, 1130), hires(HI_):
            sh = Sheet(7, box=BOX); draw(sh); return sh.roi()

    def _robe(self, v): pr = Prayer(rip(v)); return self._sheet(lambda sh: (scarf_back(sh, pr.T), pr.robe(sh), pr.insides(sh)))
    def _slv(self, v): pr = Prayer(rip(v)); return self._sheet(lambda sh: (pr.sleeves(sh), pr.scarf_shadow(sh), scarf_front(sh, pr.T)))

    def _head(self, **k):
        e = dict(EXPR['prayer']); e.update(k.pop('e', {})); h = dict(HEADK); h.update(k.pop('h', {}))
        q = Suk(h['cx'], h['cy'], h['s'], h['yaw'], h['pitch'], h['roll'], e, seed=3, **k); return self._sheet(lambda sh: q._ink_into(sh, 1, 1., False, 'dark'))

    def _body(self, v, head): return red_rim(ov(ov(self._robe(v), head), self._slv(v)), 12 * HI_, 2 * HI_)

    def _hands(self, kind):
        pr = Prayer()
        def base(g, lw=1.05):
            def d(sh):
                ink_hands(sh, g, lw)
                for q in pr.behind_lips(): sh.cut(q, sh.a, sh.v, sh.k, sh.g)
                sh.cut(FARCUT, sh.a, sh.v, sh.k, sh.g)
            return red_rim(self._sheet(d), 10 * HI_, 2 * HI_)
        P_ = dict(HD.POSES['pray'], wrist=(-34, 0)); S_ = dict(HD.POSES['sukuna'], wrist=(-34, 0))
        if kind == 'pray': return base(sign('pray', 1190, 700, 172))
        if kind == 'press': return base(sign('pray', 1190, 716, 172, poses=(dict(P_, **{q: tuple(np.asarray(P_[q], float) + np.r_[7., 8., 6., np.zeros(len(P_[q]) - 3)]) for q in ('index', 'middle', 'ring', 'little')}),) * 2))
        if kind == 'lock': return base(sign('sukuna', 1190, 690, 180, yaw=-30, pitch=-8, roll=-3))
        if kind == 'sign': return base(sign('sukuna', 1190, 704, 172, yaw=-30, pitch=-8, roll=-3))
        # smear drawings: every hand becomes one stretched mass between the two signs, a few long lines of travel, no fingers
        from jj.kit import mask_of, contours
        t0, t1 = dict(smA=(.0, .55), smB=(.40, 1.0))[kind]; steps = np.linspace(t0, t1, 5); Gs = []
        for t in steps: pose = dict(HD.blend(P_, S_, t, drag=.1), wrist=(-34, 0)); Gs.append(sign('sukuna', 1190, 700 - 14 * math.sin(math.pi * t), 174, yaw=-38 + 8 * t, pitch=-10, roll=-3, poses=(pose, pose), gap=.16 * math.sin(math.pi * t), bands=True, res=420))
        head = Gs[-1]; pr_ = pr
        def d(sh):
            r = np.random.default_rng(5 if kind == 'smA' else 9)
            for g in Gs[:-1]: sh.a.fill([np.asarray(q) for q in g.mask], SKIN_)
            m = mask_of([np.asarray(q) for g in Gs[:-1] for q in g.mask], 1130, 1920, 1)
            for q in contours(m, min_len=60): sh.wash(q, '#b9a596', .55); n_ = len(q); brush(sh.k, q[:n_ // 2:3], np.array([.4, 4.4, 2.6, 4.2, .4]), seed=3, smooth_=False); brush(sh.k, q[n_ // 2::3], np.array([.4, 3.4, 4.6, 2.2, .4]), seed=4, smooth_=False)
            ink_hands(sh, head, .95)
            for h in (0, 1):
                for fn, part in (('index', 'tip'), ('middle', 'tip'), ('ring', 'pip'), ('little', 'pip'), ('index', 'pip')):
                    L = spl(np.array([g.pts[h][fn, part] for g in Gs]), n=8)
                    if length(L) < 24: continue
                    for o in (-7, 2, 9): brush(sh.k, off(L, o + r.normal(0, 2))[: int(len(L) * r.uniform(.6, .95))], np.array([.2, 2.2, 1.2, .2]), seed=int(r.integers(99)))
            for q in pr_.behind_lips(): sh.cut(q, sh.a, sh.v, sh.k, sh.g)
            sh.cut(FARCUT, sh.a, sh.v, sh.k, sh.g)
        return self._sheet(d)

    def setup(self):
        os.makedirs('cache', exist_ok=True); self.A = {}; A = self.A
        mk = lambda n, fn: A.__setitem__(n, self._npy(n, fn))
        c0 = None
        def head0():
            nonlocal c0
            if c0 is None: c0 = self._head()
            return c0
        mk('B0', lambda: self._body(0, head0()))
        for v in (1, 2, 3, 4): mk(f'B{v}', lambda v=v: self._body(v, head0()))
        mk('B5', lambda: self._body(0, self._head(noir=1.)))
        for i, o in enumerate(LIDS): mk(f'B{6 + i}', lambda i=i, o=o: self._body(0, self._head(noir=1., e=dict(eo=(o, o), e2=(o * .6, o * .6), brow=(-.012 - .006 * i, -.03 + .008 * i), look=(0, .15)), h=dict(pitch=.14 - .015 * i))))
        mk('robe', lambda: self._robe(0)); mk('slv', lambda: self._slv(0))
        for k in ('pray', 'press', 'smA', 'smB', 'lock', 'sign'): mk('H3_' + k, lambda k=k: self._hands(k))
        self.eyes = prayer_eyes(); self.E = {}
        for i, e in enumerate(self.eyes):
            for nm, (o, sl) in ESTATE.items():
                for q in 'ul': self.E[i, nm, q] = f'cache/f1_{VER}_e{i}_{nm}_{q}'
                if not all(os.path.exists(self.E[i, nm, q] + '_xy.npy') for q in 'ul'):
                    d = e.state(o, sl, hold=list(ESTATE).index(nm) if nm not in ('A', 'B') else 6)
                    for q in 'ul': x, y, a = d[q]; np.save(self.E[i, nm, q] + '.npy', a); np.save(self.E[i, nm, q] + '_xy.npy', [x, y])
        self.zoom_heads = {f: self._npy(f'Zn{f}', lambda f=f: self._zhead(f)) for f in range(60, 72, 2)}
        g = f'cache/f1_{VER}_ground.npy'; r_ = f'cache/f1_{VER}_red.npy'
        if not os.path.exists(g): np.save(g, np.clip(ink_ground() * 255 + .5, 0, 255).astype(np.uint8))
        if not os.path.exists(r_): np.save(r_, np.clip(red_ground() * 255 + .5, 0, 255).astype(np.uint8))
        self._g, self._r = g, r_                                # uint8 plates, memory-mapped on use
        x = np.arange(W + 400); r = np.random.default_rng(3); n = len(x); n1 = noise1(n, 46, r); n2 = noise1(n, 13, r); n3 = noise1(n, 130, r)
        self.dr = 70 * np.clip(n1, -.2, 2) ** 2 + 26 * np.clip(n2, 0, 2) ** 2 + 40 * n3            # drip profile of a running edge
        self.flood = fbm(H // 2, W // 2, (150, 60, 22, 8), (1, .7, .45, .25), 8); self.flood = cv2.resize(self.flood, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float16)
        self.dust = np.c_[r.uniform(-60, W + 60, 70), r.uniform(-60, H + 60, 70), r.uniform(-1, 1, 70), r.uniform(.15, .9, 70), r.uniform(.6, 2.4, 70)]

    # ------------------------------------------------------------------ the close-up: redrawn on twos at its own size
    def zpar(self, f):
        """head parameters of the push (screen space) for f >= 59"""
        k = np.clip((f - 59) / 12.5, 0, 1); e = k * k * (3 - 2 * k) * .55 + k ** 2.2 * .45; z0 = self.cam(59); a = (np.array([HEADK['cx'], HEADK['cy']]) - z0[:2]) * z0[2] + (W / 2, H / 2); s0 = HEADK['s'] * z0[2]
        s = s0 * (900 / s0) ** e; q = (s - s0) / (900 - s0); c = a * (1 - q) + np.array([1150, 205]) * q; return c[0], c[1], s, -.40 + .06 * q, .08 - .02 * q, .05 + .12 * q

    def _zhead(self, f):
        cx, cy, s, y, p, r = self.zpar(f + .5); t = {60: .06, 62: .20, 64: .40, 66: .70, 68: 1.0, 70: 1.08}[f]; e = dict(EXPR['smile'], mouth=('mix', t), eo=(.80 + .02 * t, .80 - .06 * t), e2=(.5 + .45 * t, .5 + .42 * t), low=.05 + .30 * t, brow=(-.030, -.004 + .014 * t), jaw=.02 * t, look=(0, .15 * (1 - t)))
        q = Suk(cx + MZ[0], cy + MZ[1], s, y, p, r, e, seed=3 if f < 66 else 4, noir=1.)
        with canvas(W + 2 * MZ[0], H + 2 * MZ[1]): sh = Sheet(4, box=(0, 0, W + 2 * MZ[0], H + 2 * MZ[1])); q._ink_into(sh, 1, 1., False, 'dark'); return sh.layer()

    # ------------------------------------------------------------------ camera
    def cam(self, f):
        K = [(0, (1196, 655, 1.92)), (9, (1200, 632, 1.84)), (18, (1182, 560, 1.50)), (28, (1010, 545, 1.07)), (33, (968, 541, 1.015)), (36, (960, 540, 1.0)), (45, (966, 538, 1.012)), (53, (992, 530, 1.06)), (59, (1200, 455, 1.36))]
        return lerp_keys(min(f, 59), K)

    def warp(self, L, dx, dy, hi, cam, shift=(0, 0), sc=1., piv=None):
        z = cam[2] * sc / hi; tx = (dx + shift[0] - cam[0]) * cam[2] + W / 2; ty = (dy + shift[1] - cam[1]) * cam[2] + H / 2
        return cv2.warpAffine(L, np.float32([[z, 0, tx], [0, z, ty]]), (W, H), flags=cv2.INTER_AREA if z < .9 else cv2.INTER_LINEAR if z < 1.15 else cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)

    def lay(self, name): return np.load(self.A[name], mmap_mode='r').astype(f32) / 255.

    def look(self, i, f):
        if f >= 36:
            b = np.array([(.0, -.02), (.0, .0), (.02, 0.), (0., 0.)][i]); a = np.array([(.30, .0), (-.35, .10), (.45, -.1), (-.2, 0)][i])
            if f == 36: return a + (b - a) * .6, float(np.sign((b - a)[0])) * 1.1, (1.1, 1.2, .7)
            return b, 0., (1.08, 1.15, 1.) if f < 40 else (.92, .55, 1.)
        K = [[(0, (.45, .1)), (26, (.10, .25)), (31, (.30, .0))], [(0, (-.2, .1)), (33, (-.35, .10))], [(0, (.2, 0)), (29, (.45, -.1))], [(0, (-.2, 0))]][i]; j = max(q for q, (a, _) in enumerate(K) if a <= f); v = np.array(K[j][1], float)
        if j and f == K[j][0]: p = np.array(K[j - 1][1], float); return p + (v - p) * .55, float(np.sign(v[0] - p[0])) * .9, (1., 1., 0.)
        return v, 0., (1., 1., 0.)

    @property
    def G(self): return np.load(self._g, mmap_mode='r').astype(f32) / 255.

    @property
    def R(self): return np.load(self._r, mmap_mode='r').astype(f32) / 255.

    def back(self, f, flood):
        """ground + eyes in design space (1x)"""
        im = self.R if flood >= 1 else self.G
        if 0 < flood < 1:
            y, x = np.mgrid[0:H, 0:W].astype(f32); d = np.hypot((x - 1190) * .8, (y - 980)) - self.flood.astype(f32) * 120 - 30 * np.sin(np.arctan2(y - 980, x - 1190) * 23); R_ = 90 + 1650 * flood; m = np.clip((R_ - d) / 3. + .5, 0, 1)[..., None]; rim_ = (np.clip((R_ - d) / 26., 0, 1) * np.clip(1 - (R_ - d) / 60., 0, 1))[..., None]
            im = im * (1 - m) + self.R * (1 - .55 * rim_) * m
        live = []
        for i, s in enumerate(EYE_AT):
            k = f - s
            if k < 0: continue
            nm = ESEQ[k] if k < len(ESEQ) else ('A' if ((k - 8) // 5 + i) % 2 == 0 else 'B')
            if f >= 45: nm = 'glare'
            z_ = f - SHUT[i]
            if z_ >= 0: nm = 't1' if z_ < 2 else 's1'
            live.append((i, nm, k))
        dim = .62 if f >= 45 else 1.
        def put(i, nm, q, im):
            x, y = np.load(self.E[i, nm, q] + '_xy.npy'); a = np.asarray(np.load(self.E[i, nm, q] + '.npy', mmap_mode='r'), f32) / 255.
            if q == 'u' and dim < 1: a = a * np.array([dim, dim, dim, 1.], f32)
            return over_roi(im, a, int(x), int(y))
        im = np.ascontiguousarray(im)
        c = Cel(); gr = Cel()
        for i, nm, k in live: self.eyes[i].throw(c, k, n=(50, 40, 34, 24)[i], ground=gr)
        if live: im = np.ascontiguousarray(over(im, gr.rgba()))
        for i, nm, k in live: im = put(i, nm, 'u', im)
        for i, nm, k in live:
            o = ESTATE[nm][0]
            if o > .15: l, sm, (rk, pk, up) = self.look(i, f); x0, y0, a = self.eyes[i].iris_roi(o, l, sm, 0, rk, pk, up); im = over_roi(im, a, x0, y0)
        for i, nm, k in live: im = put(i, nm, 'l', im)
        for i, nm, k in live:
            if ESTATE[nm][0] > .5: self.eyes[i].runs(c, ESTATE[nm][0], k)
        if live: im = over(im, c.rgba())
        return im

    def edge(self, y0, seed=0):
        """screen-free running edge: design x -> y of the ink boundary"""
        return lambda x: y0 + np.interp(np.asarray(x) + 37 * seed, np.arange(len(self.dr)) - 200., self.dr)

    def coat(self, im, alpha, f, yb, cam, bx, band=None):
        """black ink still lying on a layer below the running edge; wet highlights on the left of every run"""
        ys = (np.arange(H) - H / 2) / cam[2] + cam[1]; xs = (np.arange(W) - W / 2) / cam[2] + cam[0]; b = self.edge(yb, bx)(xs); m = np.clip((ys[:, None] - b[None]) * cam[2] / 2 + .5, 0, 1).astype(f32) * alpha
        bk = smooth(-.1, .7, self.flood.astype(f32))             # wet light comes and goes along an edge
        hl = np.clip(m - cv2.warpAffine(m, np.float32([[1, 0, 4], [0, 1, 3]]), (W, H), borderMode=cv2.BORDER_REPLICATE), 0, 1) * (cv2.GaussianBlur(m, (0, 0), 6) > .04) * bk
        hl2 = np.clip(m - cv2.warpAffine(m, np.float32([[1, 0, -3], [0, 1, 2]]), (W, H), borderMode=cv2.BORDER_REPLICATE), 0, 1) * (cv2.GaussianBlur(m, (0, 0), 6) > .04) * bk
        if band: q = np.exp(-(((ys[:, None] - b[None]) - band * .3) / band) ** 2).astype(f32); hl, hl2 = hl * q, hl2 * q      # only where the ink is actually running
        im = im * (1 - m[..., None]) + rgb(BLACK) * m[..., None]; return im + hl[..., None] * np.array([.62, .56, .58], f32) * .75 + hl2[..., None] * np.array([.50, .05, .08], f32) * .45

    def frame(self, f):
        if f >= 60: return self.close(f)
        cam = np.array(self.cam(f), float); k2 = f // 2 * 2
        if f >= 36: s = shake(f, 13, 4, 2.2, 36); cam[0] += s[0]; cam[1] += s[1]; cam[2] *= 1 + .05 * math.exp(-(f - 36) / 2.2)
        for s_, a in ((20, 6), (29, 6), (45, 8)):
            if s_ <= f < s_ + 6: q = shake(f, a, 5, 1.8, s_); cam[0] += q[0]; cam[1] += q[1]
        flood = 0. if f < 45 else float(np.interp(f, [45, 46, 47, 48, 49, 50, 51], [.10, .28, .48, .66, .82, .95, 1.]))
        dark = float(np.interp(f, [0, 16, 20], [0., .0, 1.]))
        # ---- ground and eyes
        im = self.back(f, flood) if f >= 16 else np.zeros((H, W, 3), f32) + rgb('#0b090b')
        if f < 20: im = im * dark + rgb('#0a080a') * (1 - dark) if f >= 16 else im
        im = cv2.warpAffine(im, np.float32([[cam[2], 0, W / 2 - cam[0] * cam[2]], [0, cam[2], H / 2 - cam[1] * cam[2]]]), (W, H), flags=cv2.INTER_LINEAR if cam[2] < 1.1 else cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
        # ---- body
        br = 1.6 * math.sin(f / 24. * 2 * math.pi * .55)
        if f >= 8:
            nm = 'B0' if f < 36 else f'B{min(4, (f - 36) // 2 + 1)}' if f < 44 else 'B0' if f < 45 else 'B5' if f < 54 else f'B{6 + min(3, (f - 54) // 2 + (1 if f >= 59 else 0))}'
            sink = float(np.interp(f, [29, 33, 35, 36, 38, 42], [0, 7, 8, -5, 2, 0])); L = self.warp(self.lay(nm), BOX[0], BOX[1], HI_, cam, (0, sink + br)); im = over(im, L)
            if f < 31:
                yb = float(np.interp(k2, [8, 30], [-40, 1180])); im = self.coat(im, L[..., 3], f, yb, cam, 1, 70)
        # ---- hands
        rise = float(np.interp(k2, [0, 2, 4, 6, 8, 10, 12, 14], [268, 206, 140, 84, 44, 18, 5, 0])); hn = 'pray' if f < 30 else 'press' if f < 34 else 'smA' if f == 34 else 'smB' if f == 35 else 'lock' if f < 38 else 'sign'
        low = float(np.interp(f, [55, 60], [0, 120])) if f >= 55 else 0.
        sk = float(np.interp(f, [30, 33, 34, 36, 37, 39, 42], [0, 10, 4, -10, -4, 3, 0])); HL = self.warp(self.lay('H3_' + hn), BOX[0], BOX[1], HI_, cam, (0, rise + sk + low + br * .6))
        if f < 20:                                              # under the surface of the ink nothing exists
            ys = (np.arange(H) - H / 2) / cam[2] + cam[1]; sf = 838 + float(np.interp(f, [8, 20], [0, 420])); xs = (np.arange(W) - W / 2) / cam[2] + cam[0]; wl = sf + 5 * np.sin(xs / 23. + f * .9) + 3 * np.sin(xs / 9. - f * 1.3); HL = HL * np.clip((wl[None] - ys[:, None]) * cam[2] / 2 + .5, 0, 1)[..., None].astype(f32)
        im = over(im, HL)
        if f < 22:
            yb = float(np.interp(k2, [0, 2, 4, 6, 10, 14, 20], [466 + 250, 466 + 205, 640, 620, 700, 830, 1060])); im = self.coat(im, HL[..., 3], f, yb, cam, 0)
        im = self.live(im, f, cam, rise)
        if f == 36: im = impact(im, 'red', .16, .50)
        return im.astype(f32)

    def live(self, im, f, cam, rise):
        """everything that never stops: rings on the ink, drops, dust, splatter of the lock"""
        S = lambda p: (np.asarray(p, float) - cam[:2]) * cam[2] + (W / 2, H / 2); z = cam[2]; c = Cel(); r = np.random.default_rng(12)
        if f < 22:
            rg = np.random.default_rng(40 + f // 3); py = S((0, 842))[1]
            for j in range(34):                                 # short glints lying on the black surface, denser near the hands, redrawn on threes
                dy = rg.normal(0, 1) * 62 * z; dx = rg.normal(0, 1) * 420 * z; L_ = rg.uniform(20, 120) * z * (1 - min(.8, abs(dx) / (900 * z))); x0_, y0_ = S((1190, 0))[0] + dx, py + 20 * z + dy; brush(c, np.array([(x0_ - L_, y0_), (x0_, y0_ + rg.normal(0, 2)), (x0_ + L_, y0_ + rg.normal(0, 3))]), np.array([.2, rg.uniform(1.2, 3.4), .2]) * z * (1 - f / 24.), '#6d6667' if j % 3 else '#6d0c16', j, .5)
            for j, t0 in enumerate((-9, -4, 0, 4, 8, 12)):
                k = f - t0
                if 0 <= k < 16:
                    R_ = (70 + 46 * k) * z; a = 1 - k / 16.; th = np.linspace(0, 2 * math.pi, 70); ctr = S((1185, 842)); q = np.c_[ctr[0] + np.cos(th) * R_ * (1 + .02 * np.sin(5 * th + j)), ctr[1] + np.sin(th) * R_ * .16]
                    for a0, a1 in ((2, 22), (26, 44), (48, 66)): brush(c, q[a0 + j:a1], np.array([.3, 3.8, 2.4, .3]) * a * z, '#8f8687' if j % 2 else '#8a101b', j, .4)
        for j in range(46):                                     # drops falling off the hands and sleeves
            t0 = r.uniform(0, 64); x, y = (r.uniform(1120, 1290), r.uniform(560, 860)) if j % 3 else (r.uniform(930, 1700), r.uniform(780, 1000)); R_ = r.uniform(1.6, 4.2); k = f - t0
            if 0 <= k < 16 and not (f < 8 and j % 3 == 0):
                p = S((x, y + rise * .6 + .55 * k * k)); c.fill(drop_shape(p[0], p[1], R_ * z, j, 1. + .12 * k, math.pi / 2), BLACK if f > 22 else '#1a1618'); c.fill(drop_shape(p[0] - R_ * z * .3, p[1] - R_ * z * .3, R_ * z * .28, j), '#8a8082')
        for a, b, vx, vy, R_ in self.dust:
            px, py = (a + vx * 3.6 * f + 8 * math.sin(f * .21 + a)) % (W + 120) - 60, (b - vy * 3.2 * f) % (H + 120) - 60; c.fill(drop_shape(px, py, R_, int(a), 1.), '#6a6060' if vx > .2 else '#7d0f1a' if f >= 45 and vx < -.5 else '#191416', .85)
        if f >= 36:
            k = f - 36; o = np.array([1196., 700.]); rr = np.random.default_rng(77)
            for j in range(120):                                # lock: ink and blood thrown out of the hands
                th = rr.uniform(0, 6.283); v = rr.uniform(160, 900) * rr.random() ** .4; R_ = 1.5 + 11 * rr.random() ** 3.2; land = 3 + 3 * rr.random(); kk = min(k + .7, land); d = np.array([math.cos(th), math.sin(th) * .8]); pos = lambda t: o + d * (110 + v * (1 - math.exp(-t * .6))); col = ['#b3121f', '#0d0b0e', '#7d0f1a', '#d9cdb8'][int(rr.integers(4))]
                if k < land: p, q = S(pos(kk)), S(pos(max(0, kk - 1.2))); brush(c, np.array([q, (p + q) / 2, p]), np.array([.4, R_, R_ * 1.5, R_ * .8]) * z, col, j, .4)
                elif j % 2 == 0 and not (1110 < pos(land)[0] < 1570 and pos(land)[1] < 700):      # never on or above the face
                    st_ = rr.random(); g = max(0., k - (land + 2 + 9 * ((j * .37) % 1))); p = S(pos(land) + (0, .55 * g * g))
                    if p[1] < H + 40: c.fill(drop_shape(p[0], p[1], R_ * z, j, 1 + st_ + .05 * g, math.pi / 2 if g else th), col)
            if 0 < k < 6:                                       # torn dashes blown off the hands
                rb = np.random.default_rng(9)
                for j in range(54):
                    th = 2 * math.pi * (j + rb.normal(0, .35)) / 54; d = np.array([math.cos(th), math.sin(th) * .72]); R0 = (110 + 120 * k + rb.uniform(-30, 60)); L_ = rb.uniform(60, 230) * math.exp(-.28 * k); w = rb.uniform(3, 15) * math.exp(-.30 * k) * z; p, q = S(o + d * R0), S(o + d * (R0 + L_)); brush(c, np.array([p, (p + q) / 2, q]), np.array([.5, 1, .7, .1]) * w, ['#d9cdb8', '#b3121f', '#0d0b0e', '#7d0f1a'][j % 4], j, .9)
        return over(im, c.rgba())

    def close(self, f):
        k2 = f // 2 * 2; cx, cy, s, *_ = self.zpar(f); cx2, cy2, s2, *_ = self.zpar(k2 + .5); L = np.load(self.zoom_heads[k2], mmap_mode='r').astype(f32) / 255.; q = s / s2; L = cv2.warpAffine(L, np.float32([[q, 0, cx - (cx2 + MZ[0]) * q], [0, q, cy - (cy2 + MZ[1]) * q]]), (W, H), flags=cv2.INTER_LINEAR)
        z0 = self.cam(59); sc = s / (HEADK['s'] * z0[2]); cam = np.array([HEADK['cx'] - (cx - W / 2) / (z0[2] * sc), HEADK['cy'] - (cy - H / 2) / (z0[2] * sc), z0[2] * sc])
        # the background goes on from the camera of frame 59 as one push, slower than the head because it is far, and softens as the head comes near
        q_ = (sc - 1) / 2.22; zb = z0[2] * sc ** .18; bx, by = z0[0] - 390 * q_ ** .85, z0[1] + 10 * q_
        im = cv2.warpAffine(self.back(f, 1.), np.float32([[zb, 0, W / 2 - bx * zb], [0, zb, H / 2 - by * zb]]), (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
        if q_ > .04: im = cv2.GaussianBlur(im, (0, 0), .4 + 1.3 * q_)
        if sc < 2.6: im = over(im, self.warp(self.lay('robe'), BOX[0], BOX[1], HI_, cam))
        im = over(im, L)
        if sc < 2.6:
            im = over(im, self.warp(self.lay('slv'), BOX[0], BOX[1], HI_, cam)); im = over(im, self.warp(self.lay('H3_sign'), BOX[0], BOX[1], HI_, cam, (0, 120 + 60 * (sc - 1))))
        c = Cel(); r = np.random.default_rng(31)
        for j in range(60):                                     # specks of ink and blood drift across the lens
            z = r.uniform(.5, 2.6); x, y = (r.uniform(0, W) + (r.uniform(0, W) - 960) * .02 * z * (f - 60) * 2) % W, (r.uniform(0, H) - 3.5 * z * (f - 60)) % H; c.fill(drop_shape(x, y, r.uniform(.8, 3.2) * z * .7, j, 1.), BLACK if j % 3 else '#5a0a13')
        im = over(im, c.rgba())
        a = min(1., (f - 63) / 6.) if f > 63 else 0
        if a > 0:
            c = Cel(); spray(c, (470, 600), -2.7, int(60 * a), 420, .45, 15, BLACK, 5); spray(c, (560, 860), 2.5, int(40 * a), 300, .4, 12, BLACK, 6); im = over(im, inkify(c.rgba(), .3))
        return im.astype(f32)


SKIN_ = '#ecdcc3'
FARCUT = np.array([(1014, 950), (1036, 990), (1076, 1026), (1112, 1056), (1122, 1200), (600, 1200), (600, 800), (940, 846), (986, 904)], float)
