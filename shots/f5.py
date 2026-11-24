"""F5  FUGA.  One spark in the dark; the whisper; fire takes his fingers; four keys into the draw; 54 loose; a column of fire eats the city; white.  72 f."""
import numpy as np, cv2, math, os
from jj import comp, W, H, hands as HD, fx_fire as FF
from jj.paint import rgb, f32, smooth, fbm
from jj.ink import Cel, spl, resample
from jj.kit import shake, impact, smear, zoomblur, focus_lines, speed_lines, para
from jj.comp import over, screen, blur, radial
from jj.fx_core import put, hot, FX
from chars.sukuna_ink import *
from chars.sukuna import Suk, EXPR
from chars.sukuna_fire import FIRE, fire_hands, fire_ground, rim
from chars import sukuna_fuga as SF

VER = 'v8'
PALF = dict(FF.PAL, hot='#ffe04a', core='#fff8c0')
A = lambda *p: np.array(p, float)
KEYAT = [(36, 2), (40, 3), (45, 4)]                      # first frame of each key; the two frames before a key are its smear


def flames_put(im, l, amt=.6, wts=(.35, .35, .35)):
    return put(im, FX(l, hot(l, .5, (6, 26, 80), wts, '#ff7a30', amt) * (1 - .8 * l[..., 3:4])))


def drag(L, v, seed=0, n=150, wid=(3, 24), stretch=1.0):
    """a smear DRAWING: tapered strokes start inside the form, carry its flat colour (ink stripped) and shoot off along v to a point, a third of them inked on one side;
    the crisp drawing, stretched along the move, stays on top.  Hard edged, no blur, no bands."""
    v = np.asarray(v, float); M = float(np.hypot(*v)); u = v / (M + 1e-6); nn = A(-u[1], u[0]); r = np.random.default_rng(seed); h, w = L.shape[:2]
    f8 = cv2.resize(np.clip(L * 255, 0, 255).astype(np.uint8), (w // 2, h // 2), interpolation=cv2.INTER_AREA); f8 = cv2.medianBlur(cv2.medianBlur(f8, 19), 13).astype(f32) / 255.; al = f8[..., 3] > .95
    k = int(max(6, min(46, M * .10))); sh_ = np.roll(np.roll(al, -int(round(u[0] * k)), 1), -int(round(u[1] * k)), 0); edge = al & ~sh_; ys, xs = np.nonzero(edge); yi, xi = np.nonzero(al); cel = Cel(); S = []
    if len(xs) == 0: return L
    for q in range(n):
        if r.random() < .78: m = int(r.integers(len(xs))); px, py = xs[m], ys[m]
        else: m = int(r.integers(len(xi))); px, py = xi[m], yi[m]
        c = f8[py, px]; c = c[:3] / max(float(c[3]), 1e-3); big = r.random(); l = M * (r.uniform(.55, 1.) if big < .22 else r.uniform(.25, .6) if big < .6 else r.uniform(.10, .30)); hw = wid[0] + (wid[1] - wid[0]) * (r.random() ** 1.6); S.append((l, A(px * 2., py * 2.), hw, c, r.normal(0, .035), r.random()))
    S.sort(key=lambda q: -q[0])
    for l, p, hw, c, b, ink_ in S:
        a0 = p - u * hw * 2.2; t = np.linspace(0, 1, 12)[:, None]; mid = a0 + u * (l + hw * 2.2) * t + nn * (b * l * np.sin(t * math.pi * .9)); wd = hw * np.clip(1 - t ** 1.25, 0, 1) * np.clip(t * 6 + .25, 0, 1); P = np.vstack([mid + nn * wd, (mid - nn * wd)[::-1]]); cel.fill(P, np.clip(c, 0, 1))
        if ink_ < .34: cel.line((mid + nn * wd)[:10], float(np.clip(hw * .22, 1.4, 4.5)), '#1b0b06', taper=(.15, .7), press=.35, seed=int(l))
    z = cel.rgba(); top = L
    if stretch != 1.0:
        c0 = A(xi.mean() * 2, yi.mean() * 2); R_ = np.stack([u, nn], 1); Ms = R_ @ np.diag([stretch, 1 / stretch ** .35]) @ R_.T; top = cv2.warpAffine(L, np.float32([[Ms[0, 0], Ms[0, 1], c0[0] - Ms[0] @ c0], [Ms[1, 0], Ms[1, 1], c0[1] - Ms[1] @ c0]]), (w, h), flags=cv2.INTER_LINEAR)
    return z * (1 - top[..., 3:4]) + top


class Shot:
    id = 'f5'; n = 72
    finish = dict(vig=.18, bloom=.20, bloom_thr=.84, soft=.10, grain=.012)
    EVENTS = [(0, 'silence', dict(note='black, one spark')), (2, 'spark', dict(size=.2)), (7, 'spark', dict(size=.2)), (13, 'spark', dict(size=.3)), (18, 'ember_pulse', dict(size=.5, note='whisper hi')), (22, 'ember_pulse', dict(size=.6, note='whisper ra')), (27, 'ember_pulse', dict(size=.8, note='whisper ke')),
              (31, 'ignite', dict(size=.9, dur=5, note='fire finds the fingers')), (34, 'whoosh', dict(size=.8, dur=2)), (36, 'cloth_snap', dict(size=.7, note='key 2: left hand punched at the lens')), (38, 'whoosh', dict(size=.9, dur=2)), (40, 'sinew_strain', dict(size=.8, note='key 3')), (43, 'whoosh', dict(size=1., dur=2)), (45, 'bow_lock', dict(size=1., note='key 4: full draw, everything locks')), (45, 'fire_roar', dict(size=.9, dur=9, rise=1)), (46, 'tremble', dict(dur=8, size=.6)), (53, 'cut_to_silence', dict(dur=1, note='breath held')),
              (54, 'release', dict(size=1., note='loose: impact drawing')), (54, 'impact', dict(size=1.)), (56, 'arrow_pass', dict(size=1., note='arrow tears past the lens, right to left')), (57, 'explosion', dict(size=1., note='column of fire')), (57, 'fire_roar', dict(size=1., dur=15)), (59, 'shockwave', dict(size=1.)), (60, 'debris', dict(size=.9, dur=8)), (66, 'overexpose', dict(dur=6, size=1., note='to white'))]

    # ------------------------------------------------------------------ cache
    def _npy(self, name, fn):
        p = f'cache/f5_{VER}_{name}.npy'
        if not os.path.exists(p): tmp = p + f'.{os.getpid()}.npy'; np.save(tmp, np.clip(fn() * 255 + .5, 0, 255).astype(np.uint8)); os.replace(tmp, p)
        return p

    def lay(self, n): return np.load(self.A[n], mmap_mode='r').astype(f32) / 255.

    # ---- close scene: the face and the hand with the spark
    P0 = A(866, 536)                                       # the spark

    def _face(self, st):
        e = dict(EXPR['cold'], **dict(a=dict(eo=(.42, .42), e2=(.2, .2), look=(-.75, -.55), brow=(-.016, -.01)), b=dict(eo=(.52, .52), e2=(.7, .7), look=(-.75, -.50), brow=(-.020, .0), mouth='smirk'), m=dict(eo=(.72, .70), e2=(.86, .86), look=(-.58, -.30), brow=(-.030, .01), mouth=('mix', .52), jaw=.01, low=.10), c=dict(eo=(.92, .86), e2=(1., 1.), look=(-.35, -.10), brow=(-.040, .02), mouth=('mix', .95), jaw=.03, low=.2))[st])
        if st != 'c' or True: e.setdefault('mouth', 'flat')
        k = Suk(1266, 372, 415, -.46, -.15, .07, e, seed=4, neck_len=2.5); L = k.fire(ls=-1, light_vec=(-.85, .5)); return para(L, (1700, 60), (1000, 800), [(0, '#6d5a50'), (.55, '#efe4d6'), (1, '#ffffff')])

    def _hand0(self, st):
        pose, R_ = dict(a=('pinch', (64, 4, -18)), b=('pinch', (58, 2, -16)), m=(HD.blend('pinch', 'splay', .5), (55, -2, -18)), c=('splay', (52, -6, -20)))[st]; g = SF.hand(pose, 918, 690, 214, 'R', HD.euler(*R_), 4.5, (.15, .75, .45), claw=.06); np.save(f'cache/f5_{VER}_g_{st}.npy', np.array([g.pts[0][f, 'tip'] for f in ('thumb', 'index', 'middle', 'ring', 'little')] + [g.pts[0]['wrist'], g.pts[0]['palm']])); return SF.hand_layer(g, 1.45, lit=A(.2, -.98))

    def _key(self, n, seed=0, j=0., part=None):
        o = SF.key(n, seed, j); np.save(f'cache/f5_{VER}_gk{n}_{seed}.npy', np.array([o['g_rear'].pts[0]['index', 'tip'], o['g_rear'].pts[0]['thumb', 'tip'], o['g_rear'].pts[0]['wrist'], o['g_lens'].pts[0]['index', 'tip'], o['g_lens'].pts[0]['middle', 'tip'], o['g_lens'].pts[0]['palm'], o['g_lens'].pts[0]['wrist'], (o['g_lens'].pts[0]['index', 'mcp'] + o['g_lens'].pts[0]['middle', 'mcp']) / 2]))
        self._parts = (SF.stack(o, ['skin', 'neck', 'traps', 'head', 'shirt']), SF.stack(o, ['darm', 'rear']), SF.stack(o, ['narm', 'lens'])); return self._parts[0]

    def setup(self):
        os.makedirs('cache', exist_ok=True); self.A = {}; mk = lambda n, fn: self.A.__setitem__(n, self._npy(n, fn))
        for s in 'abmc': mk('face_' + s, lambda s=s: self._face(s)); mk('hand_' + s, lambda s=s: self._hand0(s))
        self.J = {s: np.load(f'cache/f5_{VER}_g_{s}.npy') for s in 'abmc'}; self.P0 = (self.J['a'][0] + self.J['a'][1]) / 2 + A(6, -10)
        for n, seeds in ((2, (0,)), (3, (0,)), (4, (0, 1, 2))):
            for q in seeds:
                nm = f'k{n}_{q}'
                if not all(os.path.exists(f'cache/f5_{VER}_{nm}{x}.npy') for x in ('A', 'B', 'C')):
                    self._key(n, q, 0. if q == 0 else 2.2); P = self._parts
                    for x, L in zip('ABC', P): self._npy(nm + x, lambda L=L: L)
                for x in 'ABC': self.A[nm + x] = f'cache/f5_{VER}_{nm}{x}.npy'
        self.K = {(n, q): np.load(f'cache/f5_{VER}_gk{n}_{q}.npy') for n, qs in ((2, (0,)), (3, (0,)), (4, (0, 1, 2))) for q in qs}
        b = 'cache/f5_bg_v1.npy'
        if not os.path.exists(b): np.save(b, fire_ground(3, ((620, 700, 1150, 1.0), (1300, 1200, 700, .5))).astype(np.float16))
        self._b = b
        self.flick = fbm(H // 4, W // 4, (40, 16), (1, .6), 3); self.flick = cv2.resize(self.flick, (W, H), interpolation=cv2.INTER_CUBIC)
        self.ruin = np.load('cache/f5_ruin_img.npy', mmap_mode='r') if os.path.exists('cache/f5_ruin_img.npy') else None
        x, y = np.meshgrid(np.arange(W, dtype=f32), np.arange(H, dtype=f32)); self.d0 = np.hypot(x - self.P0[0], (y - self.P0[1]) * 1.05)

    @property
    def BG(self): return cv2.resize(np.load(self._b, mmap_mode='r').astype(f32), None, fx=1.12, fy=1.12, interpolation=cv2.INTER_LINEAR)

    # ------------------------------------------------------------------ 0-31  the spark
    def spark(self, f):
        k = f // 2; r = np.random.default_rng(50 + k); pulse = sum(math.exp(-max(0, f - t) / d) * a * (f >= t) for t, a, d in ((18, .55, 2.2), (22, .75, 2.4), (27, 1.15, 4.0))); st = 'a' if f < 22 else 'b'
        base = float(np.interp(f, [0, 2, 5, 12, 18, 27, 31], [0., .20, .42, .58, .62, .72, .95])) * (1 + .22 * r.normal()) + pulse
        # light is a set of hard pools: the spark draws what it touches
        d = self.d0 * (1 + .07 * self.flick + .05 * math.sin(k * 2.1)); R_ = 110 + 560 * base; m = smooth(R_ * 1.25, R_ * .72, d) * .55 + smooth(R_ * .66, R_ * .50, d) * .45
        lit = np.zeros((H, W, 3), f32) + rgb('#050304'); F = self.lay('face_' + st); HL = self.lay('hand_' + st); z = 1.0 + .0016 * f; sway = (3 * math.sin(f * .11), 2 * math.sin(f * .17))
        M = np.float32([[z, 0, (1 - z) * self.P0[0] + sway[0]], [0, z, (1 - z) * self.P0[1] + sway[1]]]); F = cv2.warpAffine(F, M, (W, H), flags=cv2.INTER_LINEAR); HL = cv2.warpAffine(HL, np.float32([[1, 0, sway[0] * .4], [0, 1, sway[1] * .4]]), (W, H), flags=cv2.INTER_LINEAR)
        im = over(lit, F); im = over(im, HL); im = im * m[..., None] * np.array([1.06, .96, .86], f32) + rgb('#050304') * (1 - m[..., None])
        # irises answer the spark before the face does
        im = im + (smooth(.35, .6, F[..., 0] - F[..., 1]) * (F[..., 3] > .5) * .30 * min(1, base * 1.4))[..., None] * np.array([1., .10, .04], f32)
        im = self.sparkle(im, f, base); return im

    def sparkle(self, im, f, base):
        c = Cel(); r = np.random.default_rng(90 + f // 2); p = self.P0 + r.normal(0, 1.2, 2); s = 5 + 15 * min(1.4, base)
        for j in range(5 + int(7 * min(base, 1.5))):                              # needle star, redrawn on twos
            a = r.uniform(0, 6.283); L = s * r.uniform(.8, 3.4); w = s * .16; d = A(math.cos(a), math.sin(a)); n = A(-d[1], d[0]); c.fill(np.array([p + n * w, p + d * L, p - n * w, p - d * L * .3]), '#fff3b8')
        c.fill(drop_shape(p[0], p[1], s * .42, f, 1.), '#ffffff')
        if base > .62:                                                            # it starts to be a flame
            FF.flames(c, np.array([p + (-s * .5, s * .4), p + (s * .5, s * .4)]), s * 4.2, f / 24., 3, 3, PALF, hot=True, ink=False)
        for j in range(16):                                                       # sparks falling away, catching gravity
            t0 = j * 2.3; k = f - t0
            if 0 <= k < 13: a = -1.4 + 1.5 * math.sin(j * 5.1); q = self.P0 + A(math.cos(a) * 7.5 * k, math.sin(a) * 7.5 * k * .6 + .75 * k * k); v = A(math.cos(a) * 7.5, math.sin(a) * 4.5 + 1.5 * k); v = v / np.hypot(*v); L = 9 + k; c.fill(np.array([q + A(-v[1], v[0]) * 1.5, q + v * L, q - A(-v[1], v[0]) * 1.5, q - v * 3]), '#ffd45a' if k < 7 else '#e0571a')
        l = c.layer(); g = radial(self.P0[0], self.P0[1], 90 + 560 * base, power=2.6) * .50 + radial(self.P0[0], self.P0[1], 40 + 90 * base, power=1.6) * .7
        im = im + g[..., None] * np.array([1., .46, .12], f32) * min(1.2, .35 + base * .6); return put(im, FX(l, hot(l, .5, (4, 16, 60), (.5, .4, .4), '#ffb060', .9)))

    # ------------------------------------------------------------------ 31-35  fire takes the hand
    def whip(self, f):
        """34-35: two smear drawings.  34 the close drawing is torn off to the upper right; 35 key 2 comes in, still dragging behind itself"""
        d = A(.93, -.36); bg = cv2.resize(self.BG, (W, H), interpolation=cv2.INTER_AREA); c = Cel(); r = np.random.default_rng(f); nn = A(-d[1], d[0])
        for j in range(70):
            p = A(r.uniform(-200, W + 200), r.uniform(-100, H + 100)); l = r.uniform(300, 1300); w = r.uniform(2, 34) * r.random(); c.fill(np.array([p - d * l, p + nn * w, p + d * l, p - nn * w]), ['#ff9a2a', '#e0571a', '#8a220a', '#ffd04a', '#4a0f08'][int(r.integers(5))])
        im = over(bg, c.rgba())
        if f == 34:
            L = self.lay('face_c'); H_ = self.lay('hand_c'); L = L * (1 - H_[..., 3:4]) + H_; L = cv2.warpAffine(L, np.float32([[1, 0, 150], [0, 1, -58]]), (W, H)); L = drag(L, -d * 760, 3, 260, (3, 40), 1.16)
        else:
            L = np.zeros((H, W, 4), f32)
            for x in 'ABC': q = self.lay('k2_0' + x); L = L * (1 - q[..., 3:4]) + q
            L = drag(L, d * 640, 7, 230, (3, 30), 1.07); L = cv2.warpAffine(L, np.float32([[1.05, 0, -90], [0, 1.05, 6]]), (W, H))
        im = over(im, L); c = Cel()
        for j in range(16):
            p = A(r.uniform(0, W), r.uniform(0, H)); l = r.uniform(200, 700); w = r.uniform(1.5, 8); c.fill(np.array([p - d * l, p + nn * w, p + d * l, p - nn * w]), ['#ffe9a0', '#1b0b06'][j % 2])
        s = shake(f, 12, 3); return comp.cam2d(over(im, c.rgba()), W / 2 + s[0], H / 2 + s[1], 1.05)

    def ignite(self, f):
        k = f - 31; t = f / 24.; st = 'bmc'[min(k, 2)]; z = 1.03 + .035 * k; s = shake(f, 4 + 2 * k, 3)
        g_ = float(np.interp(k, [0, 1, 2, 4], [.25, .5, .9, 1.])); im = cv2.resize(self.BG, (W, H), interpolation=cv2.INTER_AREA) * np.array([g_, g_ * (.45 + .55 * g_), g_ * (.25 + .75 * g_)], f32)
        im = over(im, self.lay('face_' + st)); J = self.J[st]
        c = Cel(); w0 = J[5]; pl = J[6]                                            # tongues climb out of the palm, round the forearm
        for j, (a, b, w) in enumerate([(pl + (0, 60), pl + (-250, 420), 40 + 26 * k), (w0 + (30, 40), w0 + (40, 330), 34 + 20 * k), (pl, pl + (-30, -330 - 60 * k), 46 + 22 * k)]):
            m = (a + b) / 2 + A(60 * math.sin(j * 2 + t * 9), -20); FF.flame_stream(c, spl(np.array([a, m, b]), n=14), w, t + j * .4, 20 + j, PALF, lambda v: FF.leaf(v, .2, .7, .9), smooth=.6)
        im = flames_put(im, c.layer()); im = over(im, self.lay('hand_' + st))
        c = Cel()
        for j in range(5):                                                        # every finger tip lights in turn
            if k * 1.6 + .6 > j: p = J[j]; h = (90 + 60 * k) * (1.1 - .12 * j); FF.flames(c, np.array([p + (-16, 14), p + (16, 14)]), h, t + j * .21, 30 + j, 3, PALF, hot=True, sway=.22)
        FF.flame_stream(c, spl(np.array([w0 + (-70, 120), pl + (-90, 30), J[1] + (-40, 60), J[1] + (30, -220)]), n=14), 26 + 12 * k, t, 41, PALF, lambda v: FF.leaf(v, .3, .4, .9), smooth=.7); l = c.layer(); im = flames_put(im, l, .8)
        im = im * (1 + .30 * radial(self.P0[0], self.P0[1], 900, power=1.2))[..., None]; im = put(im, FF.embers(f, 50 + 20 * k, 4, (300, 0, W, H), near=.10))
        im = comp.cam2d(im, W / 2 + s[0] - 20 * k, H / 2 + s[1], z)
        return im

    # ------------------------------------------------------------------ 36-53  the draw
    def kstate(self, f):
        """-> key number, variant seed, frames since the key, smear (dx, dy)"""
        n, a = [(n, a) for a, n in KEYAT if a <= f][-1]; k = f - a; nxt = [(a2, n2) for a2, n2 in KEYAT if a2 > f]; sm = None
        if nxt and nxt[0][0] - f <= 2: sm = (nxt[0][1], 3 - (nxt[0][0] - f))      # 1, 2 = first, second smear drawing toward the next key
        q = 0 if n < 4 or k < 2 else (f // 2) % 3; return n, q, k, sm

    def draw(self, f):
        n, q, k, sm = self.kstate(f); n_from = n; i_sm = 0
        if sm is not None:
            i_sm = sm[1]
            if i_sm == 2: n, q, k = sm[0], 0, 0                                   # second smear already stands in the next key
        t = (f // 2 * 2) / 24.; fl = 1 + .07 * math.sin(f // 2 * 2.7) + .05 * np.random.default_rng(f // 2).normal()
        K0 = self.K[n, q]; pin = (K0[0] + K0[1]) / 2; gap = K0[7]; palm, lw = K0[5], K0[6]
        st = .0 if n < 4 else min(1., k / 8.); punch = 1.07 * math.exp(-k / 1.6) if k < 6 else 0; z = 1.09 + .05 * punch + .04 * st; s = shake(f, (5, 7, 10)[n - 2] * (.35 + math.exp(-k / 2.) + .55 * st), 7)
        # ---- ground: hot air dragged toward the lens
        dr = -16. * (f - 36); im = cv2.warpAffine(self.BG, np.float32([[1.02, 0, -110 + dr * .25], [0, 1.02, -60]]), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101) * (.92 + .10 * (n - 2)) * fl
        # sky darkens at the corner opposite to the fire: value contrast for the figure
        im = im * (1 - .45 * np.clip(radial(1800, 60, 1000, power=1.0), 0, 1))[..., None]
        c = Cel(); r = np.random.default_rng(300 + f // 2); vp = gap + A(-260, 160)
        for j in range(46):                                                       # streaks of fire air converge on where he aims
            a = r.uniform(0, 6.283); R0, R1 = r.uniform(520, 900), r.uniform(950, 1900); d = A(math.cos(a), math.sin(a) * .7); w = r.uniform(1.5, 8); nn = A(-d[1], d[0]); c.fill(np.array([vp + d * R0, vp + d * R1 + nn * w, vp + d * R1 - nn * w]), ['#ff9a2a', '#5a160a', '#ffd04a', '#8a220a'][int(r.integers(4))], .75)
        if st > 0: focus_lines(c, vp[0], vp[1], 1100 - 160 * st, 2100, 80, 13, '#2a0806', 400 + f // 2, ry=.62)
        im = over(im, c.rgba()); im = put(im, FF.embers(f, 70, 4, (0, 0, W, H), wind=(-220, -90), near=.12))
        B, Rr, Lz = (self.lay(f'k{n}_{q}{x}') for x in 'ABC')
        if sm is not None:
            D_ = self.K[sm[0], 0][5] - self.K[n_from, 0][5]; v = D_ * (.62 if i_sm == 1 else -.55); Lz = drag(Lz, v * 1.5, f, 70, (2, 20));  v2 = (np.mean(self.K[sm[0], 0][:2], 0) - np.mean(self.K[n_from, 0][:2], 0)) * (.5 if i_sm == 1 else -.4); Rr = drag(Rr, v2 * 1.6, f + 50, 34, (2, 14)); B = cv2.warpAffine(B, np.float32([[1, 0, -10 if i_sm == 1 else 12], [0, 1, 4]]), (W, H))
        # ---- fire behind the body
        c = Cel(); amp = (.55, .85, 1.25)[n - 2] * (1 + .25 * st)
        FF.flame_stream(c, spl(np.array([pin + (-20, 10), (pin + lw) / 2 + (0, -90 * amp), lw + (-40, -150 * amp), palm + (-240, -230 * amp), palm + (-620, -320 * amp)]), n=14), 70 * amp, t, 61, PALF, lambda v: .25 + .75 * np.sin(np.pi * np.clip(v * .9 + .05, 0, 1)), smooth=.8); im = flames_put(im, c.layer())
        fig = B * (1 - Rr[..., 3:4]) + Rr; fig = self.flicker(fig, pin, fl); im = over(im, fig)
        # ---- the near arm, then the one straight shaft: from the pinch, through the fork of the fingers, at the lens
        Lz = self.flicker(Lz, gap, fl); im = over(im, Lz); im = self.arrow(im, f, n, k, pin, gap, palm, lw, amp, t, st)
        im = comp.cam2d(im, W / 2 + s[0], H / 2 + s[1], z, .006 * math.sin(f * .8) * (1 + st))
        if sm is not None:                                                        # lines of travel: straight, tapered, on the path of the knuckles only
            c = Cel(); rr = np.random.default_rng(f); Ka, Kb = self.K[n_from, 0], self.K[sm[0], 0]
            for a, b, w in ((Ka[5], Kb[5], 190), (Ka[3], Kb[3], 70), (Ka[6], Kb[6], 120)):
                u = b - a; L = np.hypot(*u) + 1e-6; u = u / L; nn = A(-u[1], u[0])
                for j in range(7):
                    o = rr.uniform(-.5, .5) * w; l0, l1 = rr.uniform(-.8, .0) * L, rr.uniform(.5, 1.2) * L; p, q_ = a + nn * o + u * l0, a + nn * o + u * l1; ww = rr.uniform(2, 11); c.fill(np.array([p, (p + q_) / 2 + nn * ww, q_, (p + q_) / 2 - nn * ww]), ['#fdeb86', '#1b0b06', '#f6c963', '#7f672c'][int(rr.integers(4))])
            im = over(im, c.rgba())
        return im

    def flicker(self, L, p, fl):
        """fire light moves on twos: warm gain near the flame, never a flat global flash"""
        g = radial(float(p[0]), float(p[1]), 1150, power=1.2)[..., None]; return np.concatenate([L[..., :3] * (1 + (fl - 1) * 1.5 * g + .06 * g), L[..., 3:4]], -1)

    def arrow(self, im, f, n, k, pin, gap, palm, lw, amp, t, st):
        d = gap - pin; L = np.hypot(*d); d = d / L; nn = A(-d[1], d[0]); far = gap + d * (260 + 900 * min(1., (n - 2) / 2. + .4 * st)); bend = (1 - .5 * (n - 2)) * 60                     # slack in key 2, dead straight at full draw
        c = Cel(); FF.flames(c, np.array([lw + (-40, -60), lw + (40, -86), lw + (190, -74), lw + (330, -80)]), 120 * amp, t, 71, 8, PALF, hot=True, lean=.5, sway=.2, lw=1.8); im = flames_put(im, c.layer(), .5)
        c = Cel(); P = spl(np.array([pin - d * 26, pin + d * L * .35 + nn * bend, gap - d * L * .18 + nn * bend * .4, gap + d * 120, far]), n=18); w = (30, 46, 64)[n - 2] + 22 * st; ln = length(P); kk = L / ln
        prof = lambda v: np.interp(np.asarray(v, float), [0, .06, kk * .5, kk, 1.], [.10, .32, .36, .50, 1.0])
        FF.flame_stream(c, P, w, t, 73, PALF, prof, speed=1150, smooth=1., lw=2.0)
        for q, (o, w2, a0) in enumerate(((.9, .34, .2), (-.8, .30, .45))):        # two threads of fire wound round the shaft
            v = np.linspace(a0, 1, 22); Q = at(P, v) + nn * (np.sin(v * 13 + q * 3.1 + t * 11) * w * (.5 + .9 * v) * o)[:, None]; FF.flame_stream(c, Q, w * w2, t + .3 * q, 81 + q, PALF, lambda u: FF.leaf(u, .4, .5, .9), smooth=.9)
        # the string: what he is holding back
        for q, (b_, w3) in enumerate(((lw + (120, -150), 9), (lw + (60, 170), 8))):
            m = (pin + b_) / 2 + nn * (10 * math.sin(t * 40 + q)); c.line(spl(np.array([pin, m, b_]), n=10), w3 + 5 * st, '#ffd04a', taper=(.1, .5), press=.5, seed=f // 2 + q); c.line(spl(np.array([pin, m, b_]), n=10), w3 * .4, '#fff8c0', taper=(.1, .5), seed=q)
        core = [at(P, np.linspace(.01, .995, 40))]; c.line(core[0], w * .34, '#fffbe0', taper=(.30, .02), press=.3, seed=f // 2, tmin=.25)
        l = c.layer(); im = flames_put(im, l, .9)
        g = radial(float(gap[0]), float(gap[1]), 420 + 260 * st, power=2.0) * (.16 + .30 * st) + radial(float(pin[0]), float(pin[1]), 150 + 60 * st, power=2.2) * (.35 + .35 * st); return im + g[..., None] * np.array([1., .62, .22], f32)

    # ------------------------------------------------------------------ 54-56 loose
    def loose(self, f):
        im = self.draw(53); K0 = self.K[4, 0]; pin = (K0[0] + K0[1]) / 2; gap = K0[7]
        if f == 54:                                                               # impact drawing: black, bone, red
            o = impact(im, 'red', .30, .66, '#070506', '#fbf6ea', '#d8142a'); c = Cel(); focus_lines(c, gap[0] - 60, gap[1] + 40, 360, 2100, 120, 22, '#070506', 9, ry=.7); o = over(o, c.rgba()); c = Cel()
            for j in range(34):
                a = 2 * math.pi * (j + .5 * math.sin(j * 7)) / 34; R_ = 110 + 150 * abs(math.sin(j * 3.3)); d = A(math.cos(a), math.sin(a)); nn = A(-d[1], d[0]); c.fill(np.array([gap + nn * 34, gap + d * R_ * 1.5, gap - nn * 34]), '#fbf6ea')
            return over(o, c.rgba())
        if f == 55:
            o = impact(im, 'inv', .40, .62, '#070506', '#fbf6ea'); c = Cel(); r = np.random.default_rng(4); d = A(-.93, .36); nn = A(-d[1], d[0])
            for j in range(150): u = r.normal(0, 1) * 330; L = r.uniform(700, 2400); p = gap + nn * u - d * r.uniform(200, 1400); w = r.uniform(2, 26) * math.exp(-(u / 520) ** 2); c.fill(np.array([p, p + d * L * .5 + nn * w, p + d * L, p + d * L * .5 - nn * w]), '#d8142a' if j % 3 else '#070506')
            c.fill(np.array([gap - d * 1400 + nn * 8, gap + d * 200 + nn * 120, gap + d * 1700, gap + d * 200 - nn * 120, gap - d * 1400 - nn * 8]), '#070506'); return over(o, c.rgba())
        # 56: the arrow is past us: fire fills the lens, torn right to left
        im = smear(im, -420, 150, 13); c = Cel(); d = A(-.93, .36); nn = A(-d[1], d[0])
        for j, (u, w, t) in enumerate(((0, 330, .1), (-260, 130, .5), (260, 150, .8))): FF.flame_stream(c, spl(np.array([gap - d * 2700 + nn * u * .3, gap - d * 900 + nn * u * .7, gap + d * 300 + nn * u, gap + d * 1500 + nn * u * 1.6]), n=14), w, t, 90 + j, PALF, lambda v: .25 + .75 * v, smooth=1., speed=1500, lw=2.6, core=j == 0)
        im = flames_put(smear(im, -200, 70, 5), c.layer(), 1.); return im * 1.15 + .06

    # ------------------------------------------------------------------ 57-71 the column
    def column(self, f):
        k = f - 57; t = f / 24.; grow = float(np.interp(k, [0, 1, 2, 3, 5], [.12, .42, .72, .92, 1.])); s = shake(f, 20 * math.exp(-k / 5.) + 8, 11); z = 1.16 - .012 * k
        if self.ruin is not None: bg = cv2.resize(np.asarray(self.ruin, f32), (W, H), interpolation=cv2.INTER_AREA)
        else: bg = cv2.resize(self.BG, (W, H)) * .45
        heat = min(1.6, .55 + .13 * k); g = radial(1040, 720, 1500, power=1.0)[..., None]; L_ = (bg[..., :1] * .3 + bg[..., 1:2] * .6 + bg[..., 2:] * .1); im = np.clip(bg * 1.25 - .10, 0, 2) * (.30 + heat * g * np.array([1.5, .72, .34], f32)) + g * g * heat * np.array([.20, .05, .0], f32) * np.clip(L_ * 2.2, 0, 1)
        w = float(np.interp(k, [0, 2, 5, 9, 14], [90, 190, 250, 420, 900])); sm, fx = FF.fire_column(f, 1040., 880., -220., w, 7, PALF, step=2, grow=grow, sweeps=10 if k < 8 else 7, tongues=1.1)
        if sm is not None: im = put(im, sm)
        im = put(im, fx)
        if k < 7:                                                                 # ground shock: a ring of dust and stone
            c = Cel(); r = np.random.default_rng(5)
            for j in range(40):                                 # the ground tears outward: ragged tongues of dust and fire skimming the street
                a = r.uniform(0, 6.283); R0 = 120 + 300 * k * r.uniform(.6, 1.1); L_ = r.uniform(60, 260) * math.exp(-k / 6.); d = A(math.cos(a) * 1.5, math.sin(a) * .24); p = A(1040, 886) + d * R0; q = A(1040, 886) + d * (R0 + L_); w = r.uniform(4, 20) * math.exp(-k / 5.); nn = A(0, 1) * w; c.fill(np.array([p, (p + q) / 2 + nn, q, (p + q) / 2 - nn * .6]), ['#ffe9a0', '#ff9a2a', '#2a0806', '#5a160a'][j % 4])
            for j in range(60): a = r.uniform(0, 6.283); v = r.uniform(120, 420); p = A(1040 + math.cos(a) * (v * k * .8 + 80) * 1.4, 880 + math.sin(a) * (v * k * .16) - r.uniform(40, 210) * k + 17 * k * k); R2 = r.uniform(3, 15); m = int(r.integers(5, 8)); th2 = np.sort(r.uniform(0, 6.283, m)) + k * r.normal(0, .4); c.fill(np.c_[p[0] + np.cos(th2) * R2, p[1] + np.sin(th2) * R2 * .8], '#1c0a06')
            im = over(im, c.rgba())
        im = put(im, FF.embers(f, 110, 14, (0, 0, W, H), wind=(60, -420), near=.14))
        im = comp.cam2d(im, W / 2 + s[0], H / 2 + s[1], z, .004 * math.sin(f * 1.7))
        if f == 57: return np.clip(.86 + .30 * im, 0, 1.1)                          # the flash of the hit: almost paper
        e = float(np.interp(f, [64, 66, 68, 70, 71], [.0, .16, .52, .90, 1.]))     # overexposure, not a white veil: the blacks burn through last
        if e > 0: im = im * (1 + 2.4 * e) + (e ** 1.7) * np.array([.95, .80, .60], f32); w_ = float(np.clip((e - .80) / .2, 0, 1)); im = im * (1 - w_) + np.array([1.02, 1.0, .97], f32) * w_
        return im

    def frame(self, f):
        im = self.spark(f) if f < 31 else self.ignite(f) if f < 34 else self.whip(f) if f < 36 else self.draw(f) if f < 54 else self.loose(f) if f < 57 else self.column(f)
        return np.ascontiguousarray(im.astype(f32))
