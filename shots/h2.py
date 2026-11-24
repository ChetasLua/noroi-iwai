"""h2  きみがくれた いたみごと
0-33 side-tracking run right to left through fire and slashes: sprint, slide under the cut that halves the street, vault, airborne twist.
33-72 four exchanged blows with Sukuna on 36, 45, 54, 63: over his shoulder, ground level, from above, tight side."""
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT)
import numpy as np, cv2, math
from jj import W, H, comp, kit, fx, world
from jj.paint import rgb, f32, fbm, smooth, mix, grid, ramp, lum
from jj.ink import Cel, spl
from jj.cam import Camera3D
from jj.figure import Skel, Fig, P, POSES, sprint, add_pose, mix_pose, mirror, idle, seq, limb_smear, tails, to_root
from jj import fx_break as FB, fx_flash as FF
from chars.yuji import Yuji
from chars import sukuna_fire as SF

VER = 'h2_5'
HALF = math.pi / 2
PAL = dict(mul='#e6bfae', shmul='#9c7482', litmul='#ffdcb8')
RIM = dict(dir=(.25, .5, -.82), col='#ff9a3e', thr=.40, amt=.88)
DUSTP = dict(lit='#f0a468', mid='#9c5338', dark='#54251f', ink='#1c0a0a')
FPAL = dict(fx.PAL, rim='#8a1c0c', body='#e2541a', hot='#ffa030', core='#ffe08a')
RK = dict(lit='#e8965a', mid='#5a3432', dark='#2a161c', ink='#10080c')
CUTF = 11                                       # the frame the street is cut
HITS = (36, 45, 54, 63)


def C(c): return rgb(c) if isinstance(c, str) else np.asarray(c, f32)
def st(p, x, y, z, h): return dict(pose=POSES[p] if isinstance(p, str) else p, x=x, y=y, z=z, heading=h)
def rnd(*s): return np.random.default_rng(abs(hash(s)) % (1 << 31))
def lerp(f, K, e=comp.ease): return float(comp.lerp_keys(f, K, e))


def yhead(expr='snarl', light='sunset', wind=(-.6, .1), **k):
    def fn(cel, H_, i): Yuji.head(cel, H_, expr, light, i['f'], wind, damage=.6, **dict(dict(dark=.07, under=.0, hatch=.0, wrink=0., naso=.12, pouch=0., flare=.5), **k))
    return fn


def run_head(H_, i):
    """the small head of the run: real brow, eye and mouth, warmed by the fire, hair combed back by the speed"""
    x0, y0, x1, y1 = i['roi']; cel = Cel(x1 - x0, y1 - y0, x0, y0, ss=3); Yuji.head(cel, H_, 'snarl', 'sunset', i['f'], (1., .10), damage=.6, lod=1, dark=.10, under=.0, hatch=.0, wrink=.2, naso=.12, pouch=0., flare=.5); L = cel.rgba(); L = L * np.array([.965, .875, .80, 1.], f32); s = i['s']
    o = H_.P(np.array([(0., 0.), (0., .8)])); up = (o[1] - o[0]) / (np.hypot(*(o[1] - o[0])) + 1e-6); yy, xx = np.mgrid[y0:y1, x0:x1].astype(f32); v = ((xx - o[0][0]) * up[0] + (yy - o[0][1]) * up[1]) / s; d = .62 * s * smooth(.30, 1.7, v) ** 1.25 * (1 + .12 * np.sin(v * 5 + i['f'] * 1.3)); return cv2.remap(L.astype(f32), (xx - x0 - d).astype(f32), (yy - y0 - .10 * d).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def fist_at(sk, s): return sk.wr[s] + sk.Rhand[s][:, 1] * .080


def aim(F, q_, s, tgt, n=14):
    """bend the arm until the fist sits on a world point"""
    q = np.array(q_['pose']['h' + s], float); g = lambda v: fist_at(Skel(F.body, dict(q_['pose'], **{'h' + s: tuple(v)}), q_['x'], q_['y'], q_['z'], q_['heading'], F.scale), s)
    for _ in range(n):
        p = g(q); e = tgt - p
        if np.linalg.norm(e) < .002: break
        J = np.stack([(g(q + np.eye(3)[i] * .01) - p) / .01 for i in range(3)], 1); q = q + np.clip(np.linalg.solve(J.T @ J + 1e-3 * np.eye(3), J.T @ e), -.2, .2)
    return dict(q_, pose=dict(q_['pose'], **{'h' + s: tuple(q)})), float(np.linalg.norm(tgt - g(q)))


def squash(c, d, s, amt=.30):
    """head_warp: the flesh runs away from the knuckles: pushed along d, squeezed out to the sides"""
    c, d = np.asarray(c, float), np.asarray(d, float); d = d / (np.hypot(*d) + 1e-6); n = np.array([-d[1], d[0]])
    def fn(hd, roi, hi):
        x0, y0, x1, y1 = roi; yy, xx = np.mgrid[y0:y1, x0:x1].astype(f32); rx, ry = xx - c[0], yy - c[1]; a, b = (rx * d[0] + ry * d[1]) / s, (rx * n[0] + ry * n[1]) / s; g = np.exp(-(np.maximum(a, -.2) / .75) ** 2 - (b / .55) ** 2); push = amt * g; side = .42 * amt * b * np.exp(-((a - .25) / .45) ** 2 - (b / .70) ** 2); return cv2.remap(hd.astype(f32), xx - x0 - ((push * d[0] + side * n[0]) * s).astype(f32), yy - y0 - ((push * d[1] + side * n[1]) * s).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return fn


class Shot:
    id = 'h2'; n = 72
    finish = dict(bloom=.36, vig=.20, grain=.015, halation='#ff9a50', bloom_thr=.80)
    EVENTS = [(0, 'run', dict(size=.8, dur=9, note='sprint on broken asphalt: steps at 0, 5')), (0, 'fire', dict(size=.8, dur=33, note='street on fire, passing left to right')), (2, 'slash', dict(size=.4, note='hair-thin cut, behind him')), (5, 'slash', dict(size=.5, note='cut strikes the road')),
              (9, 'slide', dict(size=.7, dur=8, note='drops and slides, grit and sparks')), (11, 'slash', dict(size=1.0, note='THE cut: whole street behind is halved, high thin ring')), (13, 'rumble', dict(size=.9, dur=18, note='upper storeys slide off along the cut')), (15, 'slash', dict(size=.5)), (18, 'step', dict(size=.5)), (20, 'slash', dict(size=.5)), (21, 'plant', dict(size=.5, note='hand on the slab')), (23, 'leap', dict(size=.6, note='kicks off')),
              (24, 'slash', dict(size=.6, note='cuts the slab he left')), (26, 'whoosh', dict(size=.7, dur=7, note='airborne twist, cloth')), (30, 'crash', dict(size=.7, note='cut storeys hit the street')), (33, 'cut', dict(note='layout 1, over Sukuna\'s shoulder')), (34, 'whoosh', dict(size=.8, note='fist')), (36, 'impact', dict(size=1.0, note='blow 1 on his forearm; 2 f hit-stop, shock ring, black sparks')), (39, 'cloth', dict(size=.6, note='kimono sleeve whips')),
              (42, 'cut', dict(note='layout 2, ground level')), (43, 'whoosh', dict(size=.9, note='roundhouse')), (45, 'impact', dict(size=1.0, note='blow 2: kick on crossed arms; ground bursts')), (47, 'skid', dict(size=.6, dur=4)), (51, 'cut', dict(note='layout 3, from above')), (52, 'whoosh', dict(size=.8)), (54, 'impact', dict(size=1.1, note='blow 3: backfist against palm, road cracks in a ring')), (60, 'cut', dict(note='layout 4, tight side')), (61, 'whoosh', dict(size=.9)), (63, 'impact', dict(size=1.2, note='blow 4: both land; hit-stop, black sparks crawl')), (66, 'sparks', dict(size=.7, dur=6, note='black lightning begins to crackle, into h3'))]

    # ======================================================================= setup
    def setup(self):
        os.makedirs('cache', exist_ok=True)
        self.pl = {}
        for nm, kw, w, h in (('A', dict(eye=(0, -16, 1.5), target=(0, 30, 5.5), focal=1500, cr=(0, 200.), seed=3), 2400, 1350), ('B', dict(eye=(-10, -10, .5), target=(14, 26, 6.5), focal=1100, cr=(60, 160.), seed=5), 2400, 1350)):
            k = f'cache/h2_plate_{nm}.npy'
            if not os.path.exists(k): np.save(k, world.plate('ruin', w=w, h=h, mode='fire', **kw).img.astype(np.float16))
            self.pl[nm] = np.load(k, mmap_mode='r')
        self.track_layers(); self.ground_tex(); self.sprites()

    def track_layers(self):
        ks = {q: f'cache/h2_6_track_{q}.npy' for q in ('far', 'mid', 'near')}
        if not all(os.path.exists(k) for k in ks.values()):
            p = world.plate('ruin', w=3400, h=1300, mode='fire', eye=(0, -16, .9), target=(0, 30, 4.2), focal=1500, cr=(0, 220.), seed=11); img, z, g = np.nan_to_num(p.img).astype(f32), p.z, p.mask['ground'] > .5; del p
            low = np.arange(1300)[:, None] * np.ones((1, 3400)) > 786
            near = (z < 82) & ~g & ~low; mid = (z >= 82) & (z < 130) & ~g & ~low; far = ~near & ~mid; del z
            def fill(m):
                out = None
                for s in (220, 70, 22, 7):
                    w = comp.blur(m.astype(f32), s); c = comp.blur(img * m[..., None], s) / (w[..., None] + 1e-4); out = c if out is None else out + (c - out) * np.clip(w * 4, 0, 1)[..., None]
                return out + (img - out) * m[..., None]
            al = lambda m: cv2.GaussianBlur(m.astype(f32), (0, 0), .9)
            np.save(ks['far'], fill(far & ~low & ~g).astype(np.float16))
            for q, m in (('mid', mid), ('near', near)): a = al(m); np.save(ks[q], np.dstack([fill(m) * a[..., None], a]).astype(np.float16))
        self.far, self.mid, self.near = (np.load(ks[q], mmap_mode='r') for q in ('far', 'mid', 'near'))

    @staticmethod
    def window(layer, cx, cy, zoom, rot, T=None):
        """stand camera over a big float16 memmap: (cx, cy) lands at frame centre.  T = 2x3 motion of the layer itself (px of the layer)"""
        c, s = math.cos(rot) / zoom, math.sin(rot) / zoom; M = np.array([[c, -s, cx - c * W / 2 + s * H / 2], [s, c, cy - s * W / 2 - c * H / 2], [0, 0, 1.]])
        if T is not None: M = np.linalg.inv(np.vstack([T, [0, 0, 1.]])) @ M
        q = (M @ np.array([[0, W, W, 0], [0, 0, H, H], [1, 1, 1, 1.]]))[:2]; h, w = layer.shape[:2]
        x0, x1 = int(np.clip(q[0].min() - 3, 0, w - 2)), int(np.clip(q[0].max() + 4, 2, w)); y0, y1 = int(np.clip(q[1].min() - 3, 0, h - 2)), int(np.clip(q[1].max() + 4, 2, h))
        M = M[:2].copy(); M[:, 2] -= (x0, y0)
        return cv2.warpAffine(np.asarray(layer[y0:y1, x0:x1], f32), M.astype(f32), (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT_101 if layer.shape[2] == 3 else cv2.BORDER_CONSTANT)

    def ground_tex(self):
        k = f'cache/h2_4_ground.npy'
        if not os.path.exists(k):
            p = world.plate('asphalt', w=2400, h=1350, wet=0., amt=0., ppm=165., lamp=(1200., -5000.), seed=9); b = np.nan_to_num(p.img / (C('#10162e') * 2.1)[None, None]); L = np.clip(lum(b) * 1.7, 0, 1.3)
            L = cv2.resize(L, (1305, 734), interpolation=cv2.INTER_AREA); L = np.hstack([L, L[:, ::-1]]); L = np.vstack([L, L[::-1]])[:1170]             # 45 px/m: 58 m x 26 m
            L = cv2.blur(L, (9, 1)); v = np.linspace(0, 1, 1170)[:, None]                                                                    # v 0 far .. 1 near
            t = np.clip(L * (.62 + .9 * (1 - v) ** 2.2), 0, 1.2); im = ramp(t, [(0, '#0e0507'), (.25, '#34120f'), (.5, '#722618'), (.75, '#c2561f'), (1., '#ff9a3e'), (1.2, '#ffd080')])
            n = fbm(1170, 2610, (160, 50), (1, .6), 4); n = np.hstack([n[:, :1305], n[:, :1305][:, ::-1]]); gl = smooth(.5, 1.7, n) * (1 - v) ** 1.2; im = im + gl[..., None] * C('#ff8a34') * .38
            np.save(k, im.astype(np.float16))
        self.gtex = np.load(k, mmap_mode='r')

    def sprites(self):
        k = f'cache/{VER}_flames.npy'
        if not os.path.exists(k):
            S = []
            for v in range(3):
                for ph in range(8):
                    c = Cel(820, 620, 0, 0, ss=2); r = np.array([[210, 592], [330, 602], [480, 598], [610, 592]], float); fx.flames(c, r, dict({0: 360, 1: 470, 2: 300})[v], ph / 12., seed=11 + v * 7, b=.17, lw=1.2, spread=.06, sway=.10, n=7, pal=FPAL); S.append((c.rgba() * 255).astype(np.uint8))
            np.save(k, np.stack(S))
        self.fl = np.load(k, mmap_mode='r')
        # burning beams that whip past the lens, out of focus
        self.fg = []
        for i, (w, h, x0, y0, x1, y1, th) in enumerate(((1300, 520, 120, 440, 1180, 150, 96), (1200, 420, 120, 110, 1080, 300, 78), (430, 1250, 160, -30, 270, 1280, 92), (1500, 400, 120, 300, 1380, 230, 120))):
            c = Cel(w + 240, h, 0, 0, ss=1); r = rnd('beam', i); a_, b_ = np.array([x0 + 120, y0], float), np.array([x1 + 120, y1], float); d = (b_ - a_) / np.hypot(*(b_ - a_)); n_ = np.array([-d[1], d[0]]); L_ = np.hypot(*(b_ - a_)); u = np.linspace(0, 1, 15)
            up = a_ + np.outer(u, b_ - a_) + n_ * (th * .5 * (1 + r.normal(0, .10, 15)))[:, None]; dn = a_ + np.outer(u, b_ - a_) - n_ * (th * .5 * (1 + r.normal(0, .10, 15)))[:, None]; c.fill(np.vstack([up, dn[::-1]]), '#140608')
            for e, q in ((dn, 1), (up, -1)): c.fill(np.vstack([e, (e + n_ * q * 15)[::-1]]), '#ff7a24'); c.fill(np.vstack([e, (e + n_ * q * 6)[::-1]]), '#ffd27a')
            for j in range(16): v = r.uniform(.04, .96); p0 = a_ + (b_ - a_) * v + n_ * r.uniform(-.35, .35) * th; c.line(np.array([p0, p0 + d * r.uniform(10, 30) + n_ * 6, p0 + d * r.uniform(30, 70)]), r.uniform(3, 8), '#ff9a3a', taper=(.3, .5), seed=j)
            for j in range(9): v = r.uniform(.05, .95); p0 = a_ + (b_ - a_) * v - n_ * th * .4; hh = r.uniform(40, 120); fw = np.array([.75, -.66]); c.fill(spl(np.array([p0 - d * 26, p0 + fw * hh * .6 - d * 12, p0 + fw * hh * 1.3, p0 + fw * hh * .5 + d * 14, p0 + d * 26]), True, corners=(0, 2, 4), n=4), '#ff8a2c', .95); c.fill(spl(np.array([p0 - d * 14, p0 + fw * hh * .4, p0 + fw * hh * .8, p0 + d * 14]), True, corners=(0, 2), n=4), '#ffd880', .9)
            L = c.rgba(); L = cv2.GaussianBlur(cv2.blur(L, (171, 3)), (0, 0), 7); self.fg.append((np.clip(L, 0, 1) * 255).astype(np.uint8))
        # set dressing along the run: (kind, x, y, size, seed) metres
        r = rnd('dress'); D = []
        for x in np.arange(-17, 6, .95):
            y = r.uniform(1.4, 8.5); D.append(('rock', x + r.uniform(-.3, .3), y, r.uniform(.06, .20) * (1 + .12 * y), int(r.integers(24))))
        for x in np.arange(-17, 5, 2.1): D.append(('fire', x + r.uniform(-.8, .8), r.uniform(5.0, 11.), r.uniform(1.0, 2.1), int(r.integers(3))))
        for x in (-2.6, -8.4, -13.6): D.append(('pole', x, r.uniform(4.5, 7.), r.uniform(4.5, 6.5), int(r.integers(9))))
        D.sort(key=lambda d: -d[2]); self.dress = D

    # ======================================================================= A. the run
    XK = [(0, 0.), (9, -3.06), (17, -5.66), (20, -6.38), (24, -7.42), (33, -10.1), (40, -12.2)]

    def X(self, f): k = np.array(self.XK); return float(np.interp(f, k[:, 0], k[:, 1]))

    def ystate(self, f):
        """Yuji on the run: pure function of the frame"""
        x = self.X(f); z = 0.; h = HALF + .22
        if f < 9: p = sprint(.12 + f / 10.6)
        else:
            K = [(8.9, sprint(.12 + 9 / 10.6), 'in'), (10.6, 'slide_low', 'lin'), (15.5, add_pose(POSES['slide_low'], pel=(-5, 3, 4), hL=(.02, .02, .06)), 'in'), (17.4, 'land', 'out'), (19, 'run_push', 'in'), (20.6, add_pose(POSES['vault'], root=(0, 0, -.05)), 'out'), (22.6, 'vault', 'in'), (24.2, 'push_off', 'out'), (25.6, 'push_off', 'in'), (27.0, 'corkscrew', 'lin'), (29.8, add_pose(POSES['corkscrew'], pel=(4, -8, 6), hR=(.06, .1, .08)), 'out'), (31.6, 'twist_open', 'out'), (33.6, 'dive_punch', 'lin'), (36, 'dive_punch', 'lin')]
            p = seq(f, [(a, to_root(POSES[q]) if isinstance(q, str) else to_root(q), e) for a, q, e in K])
        k = 1. if f < 24.5 else max(0., 1 - (f - 24.5) / 2.); p = add_pose(p, nck=(0, 0, 10 * k), hed=(0, 0, 18 * k))                                            # he shows us his face
        z = float(np.interp(f, [19, 20.5, 22.5, 24, 26, 28, 30, 33], [0, .18, .42, .62, 1.25, 1.62, 1.72, 1.55]))
        if f > 24.5: t = min(1., (f - 24.5) / 6.8); h += 2 * math.pi * (t * t * (3 - 2 * t))
        return dict(pose=p, x=x, y=0., z=z, heading=h)

    def zc(self, f):
        """height of his hips, smoothed: the operator follows a little late"""
        F_ = self.yuji(); return float(np.mean([Skel(F_.body, q['pose'], q['x'], q['y'], q['z'], q['heading']).pel[2] for q in (self.ystate(g) for g in (f - 2.5, f - 1.2, f, f + .8))]))

    def cam_track(self, f):
        off = lerp(f, [(0, -.62), (8, -.30), (11, -.16), (16, .0), (22, .08), (33, .18)]); cx = self.X(f) + off
        self.D = lerp(f, [(0, 3.85), (8, 3.55), (11, 2.75), (15, 2.70), (19, 3.30), (23, 3.15), (27, 2.65), (33, 2.30)]); z = self.zc(f)
        ez = max(lerp(f, [(0, .50), (8, .40), (11, .19), (16, .20), (20, .42), (24, .70), (33, .95)]), z - .30 * self.D); tz = z + lerp(f, [(0, .02), (9, .02), (11, .26), (16, .28), (20, .04), (33, .0)]); fo = lerp(f, [(0, 1440), (9, 1480), (12, 1500), (18, 1460), (25, 1480), (33, 1560)])
        ro = lerp(f, [(0, .16), (8, .14), (10.5, -.20), (15, -.17), (18.5, .14), (24, .16), (28, .22), (31, .20), (33, .12)])
        s = kit.shake(f, 4., 4) + kit.shake(f, 22, 9, 3., CUTF) + kit.shake(f, 10, 11, 2.5, 5) + kit.shake(f, 17, 12, 3., 30) + kit.shake(f, 10, 13, 2., 23)
        eye = np.array([cx, -self.D, ez]); return Camera3D(eye, (cx - .08, 0, tz), focal=fo, roll=ro, cx=W / 2 + s[0], cy=H / 2 + s[1]), cx, ez, tz, fo, ro, s

    def bg(self, cx, ez, tz, fo, ro, s, f):
        z = fo / 1500.; dp = ((tz - ez) / self.D - .0717) * 1500
        def put(layer, d, T=None):
            sh = -cx * 1500. / d * .62; return self.window(layer, 1700 - sh + 260 * (24. / d) - s[0] / z, 650 - dp - (ez - .9) * 1500. / d - s[1] / z, z, -ro, T)
        hb = lambda L, n: cv2.blur(L, (int(n) | 1, 1))
        out = put(self.far, 170.); out = comp.over(out, hb(put(self.mid, 90.), 5)); n = put(self.near, 30.); k = f - CUTF
        if k < 0: return comp.over(out, hb(n, 13))
        # the cut in plate px is y = 520 - x / 34; carry it to the screen
        sh = -cx * 1500. / 30. * .62; c0 = (1700 - sh + 260 * .8 - s[0] / z, 650 - dp - (ez - .9) * 50. - s[1] / z); cr, sr = math.cos(-ro) / z, math.sin(-ro) / z
        x, y = grid(); px = cr * (x - W / 2) - sr * (y - H / 2) + c0[0]; py = sr * (x - W / 2) + cr * (y - H / 2) + c0[1]; d = py - (520 - px / 34.); bot = np.clip(d + .5, 0, 1)
        out = comp.over(out, hb(n * bot[..., None], 13)); lip = np.exp(-(d / 2.4) ** 2) * n[..., 3] * max(0., 1 - k / 16.)
        t = max(0., k - 1.0); T = cv2.getRotationMatrix2D((2300, 480), -.20 * t, 1.); T[:, 2] += (15. * t + .85 * t * t, 1.6 * t + .62 * t * t)
        self.top_mask(); tp = self.window(self.top, c0[0] + 900, c0[1], z, -ro, T); a = tp[..., 3]; Ti = np.linalg.inv(np.vstack([T, [0, 0, 1.]])); X0 = Ti[0, 0] * (px + 900) + Ti[0, 1] * py + Ti[0, 2]; Y0_ = Ti[1, 0] * (px + 900) + Ti[1, 1] * py + Ti[1, 2]; dT = Y0_ - (520 - (X0 - 900) / 34.); under = np.exp(-((dT + 4) / 5.) ** 2) * cv2.dilate(a, np.ones((9, 3), np.uint8)) * max(0., 1 - k / 20.); del px, py, X0, Y0_, dT; tp = hb(tp, 11 + t)
        shd = kit.shifted(tp[..., 3], 4, 34 + 2 * t); out = out * (1 - .45 * np.clip(shd - tp[..., 3], 0, 1))[..., None]                       # its shadow on what is left standing
        out = comp.over(out, tp); out = out + comp.blur(under, 6)[..., None] * C('#ff8a3a') * 1.3 + under[..., None] * C('#ffd9a0') * 1.2
        c = Cel(); r = rnd('cutstones'); kk = k - 1
        for _ in range(54):                                                              # grit spills along the whole cut
            x0 = r.uniform(-100, W + 300); v = r.uniform(-60, 120); g_ = r.uniform(16, 30); sz = r.uniform(3, 15); d0 = r.integers(0, 12); u = kk - d0; w_ = r.uniform(0, 6.28); dl = r.uniform(-1, 1, 5)
            if u <= 0: continue
            px_ = x0 + (26 + v * .05) * u; py_ = 455 - 60 * (x0 / W) + x0 * 0 + v * .04 * u + g_ * .09 * u * u; th = np.linspace(0, 6.283, 6)[:-1] + w_ + u * .4; c.fill(np.c_[px_ + np.cos(th) * sz * (1 + .25 * dl), py_ + np.sin(th) * sz * (1 - .2 * dl)], '#1c0a0c'); c.fill(np.c_[px_ + np.cos(th) * sz * .5 + 1, py_ + np.sin(th) * sz * .4 - sz * .25], '#b8552a')
        out = comp.over(out, cv2.blur(c.layer(), (7, 3)))
        return out + (comp.blur(lip, 9)[..., None] * C('#ffb070') * 1.6 + lip[..., None] * C('#fff0d8') * 1.8)

    def top_mask(self):
        """upper half of the near row, cut along the slash: built once, on disk"""
        if hasattr(self, 'top'): return
        k = f'cache/h2_7_track_top.npy'
        if not os.path.exists(k):
            x = np.arange(3400, dtype=f32)[None]; y = np.arange(540, dtype=f32)[:, None]; t_ = np.asarray(self.near[:540], f32) * np.clip(520 - x / 34. - y + .5, 0, 1)[..., None]; np.save(k, np.pad(t_, ((0, 0), (900, 900), (0, 0)), mode='reflect').astype(np.float16))
        self.top = np.load(k, mmap_mode='r')

    def dressing(self, img, cam, f, y0, y1):
        """rocks, poles, fires between y0 and y1 (far to near)"""
        cel = None; ph = (int(f) // 2)
        for kind, x, y, s, sd in self.dress:
            if not (y0 <= y < y1): continue
            p, z = cam.project(np.array([x, y, 0.])); k = cam.f / z
            if not -700 < p[0] < W + 700: continue
            if kind == 'fire':
                if cel is not None: img = comp.over(img, cv2.blur(cel.layer(), (9, 1))); cel = None
                h = s * k * 1.45; spx = self.fl[sd * 8 + (ph + sd * 3 + int(x * 7)) % 8]; sc = h / 560.; w_ = int(820 * sc); h_ = int(620 * sc)
                if w_ < 8 or h_ < 8: continue
                q = cv2.resize(np.asarray(spx), (w_, h_), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR); q = cv2.GaussianBlur(q, (0, 0), .6 + .42 * max(0., y - 2.)); blit(img, q, int(p[0] - w_ / 2), int(p[1] - h_ * .965)); gl = int(h * .9)
                add_glow(img, p[0], p[1] - h * .35, gl, C('#ff7a26') * .34)
            elif kind == 'pole':
                cel = cel or Cel(); t_, _ = cam.project(np.array([x + (sd - 4) * .10, y, s])); n_ = np.array([.055 * k, 0]); b_ = np.array(p); cel.fill(np.array([b_ - n_ * 1.2, b_ + n_ * 1.2, t_ + n_ * .8, t_ - n_ * .8]), '#1c0a0d'); cel.fill(np.array([b_ + n_ * .4, b_ + n_ * 1.2, t_ + n_ * .8, t_ + n_ * .3]), '#b8501e'); a_ = t_ * .86 + b_ * .14; cel.fill(np.array([a_ + [-.7 * k, -.04 * k], a_ + [.6 * k, .05 * k], a_ + [.6 * k, .13 * k], a_ + [-.7 * k, .05 * k]]), '#1c0a0d')
            else:
                cel = cel or Cel(); cel.fill(np.c_[p[0] + np.cos(np.linspace(0, 6.28, 14)) * s * k * .95, p[1] + np.sin(np.linspace(0, 6.28, 14)) * s * k * .16], '#0a0406', .6); fx.rock(cel, p[0], p[1] - s * k * .30, s * k * .8, (sd * .7, sd * 1.3, sd * .4), seed=sd, pal=RK, light=(.3, -.8, -.2), lw=1.6, crack=s * k > 40)
        if cel is not None: img = comp.over(img, cv2.blur(cel.layer(), (9, 1)))
        return img

    def slab(self, cam, f):
        """the tilted slab he vaults, and its two halves after frame 24"""
        x0 = -7.62; c = Cel(); k = max(0., f - 24.5)
        def box(xa, xb, ha, hb, dz=0., rot=0., dx=0.):
            V = []
            for y in (-.75, .85):
                for (x, z) in ((xa, 0.), (xb, 0.), (xb, hb), (xa, ha)):
                    cxm = (xa + xb) / 2; X_, Z_ = x - cxm, z; X2 = X_ * math.cos(rot) - Z_ * math.sin(rot); Z2 = X_ * math.sin(rot) + Z_ * math.cos(rot); V.append((cxm + X2 + dx, y, Z2 + dz))
            q = cam.project(np.array(V))[0]; n, b = q[:4], q[4:]
            top = np.array([n[3], n[2], b[2], b[3]]); side = np.array([n[1], n[2], b[2], b[1]]); left = np.array([n[0], n[3], b[3], b[0]])
            c.fill(left, '#33191c'); c.fill(side, '#4a2420'); c.fill(top, '#c47244'); c.fill(n, '#54302f')
            for A in (n, top): kit.outline(c, A, 3.0, '#12080b', seed=int(xa * 9))
            m = n.mean(0); c.line(np.array([n[3] * .7 + n[2] * .3, m * .6 + n[2] * .4, n[1] * .5 + m * .5]), 2., '#1c0c10', taper=(.1, .6), seed=3); c.line(np.array([n[0] * .6 + n[3] * .4, m, n[1] * .4 + n[0] * .6]), 1.7, '#1c0c10', taper=(.2, .6), seed=4)
            c.fill(np.array([n[3] * .95 + n[0] * .05, n[2] * .95 + n[1] * .05, n[2] * .86 + n[1] * .14, n[3] * .90 + n[0] * .10]), '#7a4638')
        if k <= 0: box(x0 - .85, x0 + .95, .46, .62)
        else: box(x0 - .85, x0 + .02, .46, .535, 0, .02 * k, -.02 * k); box(x0 + .02, x0 + .95, .535, .62, max(-.2, -.015 * k * k), -.035 * k, .035 * k)
        return c

    SL = [(2, (1500, 250, 2300, 560)), (5, (-100, 760, 900, 1010)), (7, (300, 120, 1250, 420)), (15, (900, 180, 2050, 330)), (20, (100, 420, 1100, 180)), (24, (500, 720, 1500, 880)), (27, (1200, 200, 2100, 640)), (29, (-60, 300, 700, 60))]

    SLF = [(3, (-300, 980, 760, 520)), (8, (900, 60, 1800, 420)), (14, (-250, 300, 900, 640)), (18, (700, 900, 1700, 560)), (22, (-200, 120, 800, 480)), (26, (600, 980, 1500, 360)), (29, (-250, 700, 850, 330))]

    def track(self, f):
        cam, cx, ez, tz, fo, ro, s = self.cam_track(f)
        img = self.bg(cx, ez, tz, fo, ro, s, f)
        # heat haze: fire glow sits behind the figure so that he reads dark on light
        hp = cam.project(np.array([cx, 14., .6]))[0]; x, y = grid(); yy = (y - hp[1]) + (x - W / 2) * math.tan(-ro) * 0
        img = comp.screen(img, (np.exp(-(yy / 210) ** 2) * .30)[..., None] * C('#ff7a2c')); img = comp.screen(img, (np.exp(-((yy + 40) / 120) ** 2) * np.exp(-((x - 860) / 620) ** 2) * .34)[..., None] * C('#ffb24a'))
        # road
        xa = 29. * math.floor((cx - 14.5) / 29.)
        c0 = int(np.clip((cx - 18 - xa) * 45, 0, 1300)); g = cam.warp_plane(np.asarray(self.gtex[:, c0:c0 + 1620], f32), (xa + c0 / 45., 21.5, 0.), (1, 0, 0), (0, -1, 0), 36., 26.); t = cam.plane_t((0, 0, 0), (0, 0, 1.)); a = g[..., 3] * (1 - smooth(13., 24., t)); img = img * (1 - a[..., None]) + g[..., :3] * a[..., None]
        img = self.dressing(img, cam, f, 2.0, 12.)
        # cuts in the air, behind him
        if f >= CUTF and f < CUTF + 12:
            k = f - CUTF; L = [(-80, 486 + 2 * k, W + 80, 404 + 2 * k)]; img = fx.put(img, fx.slash_fx(L, k, w=5.5, seed=3, glowcol='#ffd0a0', amt=.8, gap=12., life=11))
        for k0, ln in self.SL:
            k = f - k0
            if 0 <= k <= 8: q = np.array(ln, float); q[[0, 2]] += 46 * k; img = fx.put(img, fx.slash_fx([q], k, w=3.6, seed=k0, glowcol='#ffc890', amt=.55, gap=7., life=9))
        # dust and grit
        img = self.dust(img, cam, f, False)
        # shadow
        kk = 5.2 / self.D; sx, sy = cam.project(np.array([self.X(f) + .05, .0, 0.]))[0]; hgt = self.ystate(f)['z']; sh = np.zeros((H // 2, W // 2), f32); cv2.ellipse(sh, (int(sx / 2), int(sy / 2)), (int(170 * kk / (1 + hgt)), int(17 * kk / (1 + hgt * .5))), -math.degrees(ro), 0, 360, 1., -1); sh = cv2.resize(cv2.GaussianBlur(sh, (0, 0), 6 + 8 * hgt), (W, H)); img = img * (1 - sh[..., None] * .72 / (1 + hgt))
        if 14 < f < 33: img = comp.over(img, self.slab(cam, f).layer())
        # Yuji
        Y = self.yuji().anim(self.ystate); Y.at(f); Y.blow(1.0, 0, .25)
        L = Y.draw(cam, f, light=(-.55, .55, .5), kick=(-.8, .3, -.4), rim=RIM, pal=PAL, expr=dict(mood='rage', damage=1.), seed=int(f), head_fn=run_head)
        v = cam.project(np.array([self.X(f), 0, 1.]))[0] - cam.project(np.array([self.X(f - 1) + (cx - self.cam_track(f - 1)[1]), 0, 1.]))[0]
        tl = tails(L, (-250 if f < 24 else -160, -14 if f < 24 else 60 if f < 29 else -50), n=64, seed=f, width=(1.6, 6.5), length=(.3, 1.0), dry=.6)
        img = comp.over(img, tl, .62)
        if f in (20, 21): img = comp.over(img, Y.smear(cam, f - 1.6, f, 'armL', pal=PAL, seed=f, lines=6), .9)
        if 26 <= f <= 30:
            img = comp.over(img, Y.smear(cam, f - .8, f, 'legR', pal=PAL, seed=f, lines=5, tail=.7), .55); img = comp.over(img, Y.smear(cam, f - .8, f, 'armR', pal=PAL, seed=f + 40, lines=5, tail=.7), .55)
        img = comp.over(img, L)
        img = self.dust(img, cam, f, True)
        img = self.dressing(img, cam, f, -1., 2.0)
        for k0, ln in self.SLF:
            k = f - k0
            if 0 <= k <= 6: q = np.array(ln, float); q[[0, 2]] += 78 * k; img = fx.put(img, fx.slash_fx([q], k, w=6.5, seed=k0 + 50, glowcol='#ffd8a8', amt=.7, gap=13., life=7))
        # something burning whips past the lens
        for f0, i, yv in ((2.5, 1, 820), (6.5, 0, -230), (13.5, 3, 850), (17, 0, -260), (20.5, 1, 840), (26.5, 3, 880), (30.5, 0, 800)):
            k = f - f0
            if abs(k) <= 4.2: q = self.fg[i]; blit(img, q, int(W * .5 - q.shape[1] / 2 + k * 520), yv)
        # streaks of hot air
        c = Cel(ss=1); r = rnd('sp', f)
        for _ in range(26):
            yv = r.uniform(80, H); x0 = r.uniform(-200, W); ln = r.uniform(160, 620); c.fill(np.array([(x0, yv), (x0 + ln * .5, yv - 1.2), (x0 + ln, yv + 2), (x0 + ln * .5, yv + r.uniform(1.2, 3.2))]), '#ffc070' if r.random() < .6 else '#20080a', r.uniform(.25, .7))
        img = comp.over(img, c.layer(), .75)
        img = fx.put(img, fx.embers(f, 60, 5, wind=(900., -120.), size=1.2, near=.14, life=1.2))
        if f in (0,): pass
        if f == CUTF:
            img = self.struck([L], (W * .5, 440), 21, inv=True, lines=False); c = Cel(); c.fill(np.array([(-20, 487), (W * .55, 440), (W + 20, 400), (W * .45, 452)]), '#fff6e6'); img = comp.over(img, c.layer())
        if f == 32: img = streak(img, -60, 130)
        return img

    def dust(self, img, cam, f, front):
        """slide plume, pushes, sparks, slash bursts: emitted in the world, left behind by the tracking camera"""
        c = Cel(); any_ = False; kk = 5.2 / self.D
        for e in range(9, 19):
            t = (f - e) / 24.
            if t < 0 or t > .62 or (e % 2 == 0) == front: continue
            p, z = cam.project(np.array([self.X(e) + .34, -.10 if front else .35, .10])); L_ = FB.billow(t + .16, p[0], p[1] - 8, (150 if front else 190) * kk * (1.05 - .05 * (e - 9)), e * 3, 6, .75, 1.1, .9, sub=3)
            if L_: FB.puffs(c, L_, DUSTP, (.2, -.9), 1.7, e); any_ = True
        for e, sz in ((0, 62), (5, 66), (18.5, 110), (23.5, 110)):
            t = (f - e) / 24.
            if 0 <= t <= .5 and front:
                p, z = cam.project(np.array([self.X(e) + .25, .1, .08])); FB.puffs(c, FB.billow(t + .12, p[0], p[1], sz, int(e) + 50, 6, .7, 1., .7, sub=2), DUSTP, (.2, -.9), 1.6, int(e)); any_ = True
        if front:
            r = rnd('spark', f)
            if 10 <= f <= 17:                                                # heel and hand throw sparks
                for px in (cam.project(np.array([self.X(f) - .62, .0, .04]))[0], cam.project(np.array([self.X(f) + .42, -.2, .04]))[0]):
                    for _ in range(13): a = r.uniform(-.75, -.02); d0 = r.uniform(0, 90); ln = r.uniform(40, 190); u = np.array([math.cos(a), math.sin(a)]); q = px + u * d0; c.line(np.array([q, q + u * ln * .5 + [0, 3], q + u * ln + [0, 10]]), r.uniform(1.4, 4.), '#fff0c0' if r.random() < .5 else '#ffa53a', taper=(.15, .7), seed=int(r.integers(99))); any_ = True
            for k0, gx in ((5, -1.2), (24, 1.0)):
                k = f - k0
                if 0 <= k <= 7: p, z = cam.project(np.array([self.X(k0) + gx, .6, 0.])); fx.debris(c, k + 1, (p[0], p[1]), n=14, seed=k0, speed=(300, 900), size=(5, 34), spread=(120, 10), pal=RK, light=(.3, -.8, -.2), lw=1.6); any_ = True
            k = f - 30
            if 0 <= k <= 4: p, z = cam.project(np.array([self.X(f) + 1.6, 9., .2])); fx.smoke_burst(c, k + 2, p[0], p[1], 150, seed=3, n=7, pal=DUSTP, light=(.2, -.9), speed=520, ground=True, lobes=5); any_ = True
        return comp.over(img, c.layer()) if any_ else img

    # ======================================================================= B. the blows
    def yuji(self): return Fig('yuji', spec=dict(burnt=1))
    def suk(self): return Fig('sukuna', seed=3)

    def blow_bg(self, nm, cx, cy, z, rot, f, dx=0., dy=0., dof=2.0):
        s = kit.shake(f, 2.4, 21); im = self.window(self.pl[nm], cx + s[0], cy + s[1], z, rot)
        if dof: im = comp.blur(im, dof)
        if abs(dx) + abs(dy) > 2: im = streak(im, dx, dy)
        return im

    def impact_fx(self, img, hit, f, h, scale=1., ang=0., black=1., stones=False, ring=True):
        """what a hit throws: h struck drawing, h+1 held, then ring, sparks, black lightning"""
        k = f - h
        if k < 1 or k > 8: return img
        cx, cy = hit
        img = fx.distort_ring(img, cx, cy, (110 + 170 * k) * scale, 80 * scale, 22 * max(0, 1 - k / 7.), seed=h, asp=.9)
        img = img + kit.radial(cx, cy, 460 * scale, power=2.4)[..., None] * C('#ff7a30') * max(0, .62 - k * .13)
        c = Cel(); fx.ragged_ring(c, cx, cy, (120 + 180 * k) * scale, (96 + 140 * k) * scale, 15 * max(.1, 1 - k / 7.) * scale, h, '#fff0d8', gaps=.5, rot=ang, spurs=.25) if ring else None; img = comp.over(img, c.layer(), max(0, .85 - k * .14))
        c = Cel(); r = rnd('bk', h, f)
        if k <= 2: kit.focus_lines(c, cx, cy, (200 + 120 * k) * scale, 1500, 70, 4.5, '#fff0d8', 7 + h + k, .5, .9)
        for j in range(int(max(0, 26 - 5 * k))):                                     # white-hot grains
            a = r.uniform(0, 6.283); d0 = r.uniform(30, 120 + 120 * k) * scale; l = r.uniform(26, 130) * scale * (1 - k * .08); u = np.array([math.cos(a), math.sin(a) * .86]); p0 = np.array([cx, cy]) + u * d0; c.line(np.array([p0, p0 + u * l * .5 + [0, 2 * k], p0 + u * l + [0, 5 * k]]), r.uniform(1.6, 4.6) * scale, '#fff2cc' if r.random() < .55 else '#ffa844', taper=(.15, .7), seed=j)
        if k <= 2: S = FF.star_poly(cx, cy, (34 - 6 * k) * scale, (270 - 70 * k) * scale, 13, 5 + h, asp=.8, rot=.3 + ang); kit.shape(c, S, '#fff8ea', 2.6, '#140a0c', 3); c.fill(FF.star_poly(cx, cy, 18 * scale, (130 - 34 * k) * scale, 9, 9 + h, asp=.8), '#ffffff')
        for j in range(int((12 - k) * black)):                               # sparks of black: kinked, red-edged
            a = r.uniform(0, 6.283); d0 = r.uniform(20, 80 + 52 * k) * scale; L_ = r.uniform(80, 270) * scale * (1 - k * .07); p0 = np.array([cx + math.cos(a) * d0, cy + math.sin(a) * d0 * .85]); P_ = FF.bolt_line(p0, p0 + np.array([math.cos(a + r.normal(0, .4)), math.sin(a + r.normal(0, .4))]) * L_, int(r.integers(9999)), .34, 5); q = r.uniform(.7, 1.3) * scale
            for w_, col in ((13., '#e81430'), (7.5, '#08040a')): c.line(P_, w_ * q, col, taper=(.10, .75), seed=j, press=.1, tmin=.05, tremor=0)
        img = comp.over(img, c.layer())
        if stones: c = Cel(); fx.debris(c, k + 1, stones, n=20, seed=h, speed=(400, 1500), ang=(-2.8, -.35), size=(5, 40), spread=(220, 24), pal=RK, light=(.3, -.8, -.2), lw=1.6); img = comp.over(img, c.layer())
        return img

    def struck(self, layers, hit, seed, inv=False, red=True, lines=True):
        """the struck drawing: paper, focus lines, the figures cut to black, paper and one red; the hit burns through"""
        pa, ik = (C('#fff3e2'), C('#0a0408')) if not inv else (C('#0a0408'), C('#fff3e2')); im = np.empty((H, W, 3), f32); im[:] = pa
        c = Cel()
        if lines: kit.focus_lines(c, hit[0], hit[1], 300, 1500, 130, 8, '#0a0408' if not inv else '#fff3e2', seed, .6, .85); kit.focus_lines(c, hit[0], hit[1], 520, 1500, 60, 22, '#0a0408' if not inv else '#fff3e2', seed + 3, .8, .85)
        else:
            r = rnd('st', seed)
            for _ in range(60): yv = r.uniform(0, H); x0 = r.uniform(-200, W); ln = r.uniform(300, 1300); c.fill(np.array([(x0, yv), (x0 + ln * .5, yv - 2.5), (x0 + ln, yv), (x0 + ln * .5, yv + r.uniform(1.5, 6))]), '#fff3e2' if inv else '#0a0408')
        im = comp.over(im, c.layer())
        for L in layers:
            a = L[..., 3]; q = L[..., :3] / np.maximum(a, 1e-3)[..., None]; l = lum(q); t = smooth(.34, .42, l)[..., None]; col = C('#0a0408') * (1 - t) + C('#fff3e2') * t
            if red: m = ((q[..., 0] - q[..., 2] > .38) & (q[..., 0] > .55) & (q[..., 1] < .62 * q[..., 0] + .05)).astype(f32); m = cv2.GaussianBlur(m, (0, 0), .8)[..., None]; col = col * (1 - m) + C('#d8142a') * m
            e = np.clip(cv2.dilate(a, np.ones((5, 5), np.uint8)) - a, 0, 1)[..., None]; im = im * (1 - e) + ik * e; im = im * (1 - a[..., None]) + col * a[..., None]
        if lines:
            c = Cel(); S = FF.star_poly(hit[0], hit[1], 40, 330, 15, 5 + seed, asp=.8, rot=.3); kit.shape(c, S, '#ffffff', 4.5, '#0a0408', 3); c.fill(FF.star_poly(hit[0], hit[1], 24, 170, 9, 9 + seed, asp=.8), '#d8142a'); c.fill(FF.star_poly(hit[0], hit[1], 18, 120, 9, 12 + seed, asp=.8), '#ffffff'); im = comp.over(im, c.layer())
        return im

    def toned(self, im, L, ink, paper, red=True, edge=None, thr=.38):
        a = L[..., 3]; q = L[..., :3] / np.maximum(a, 1e-3)[..., None]; l = lum(q); t = smooth(thr - .04, thr + .04, l)[..., None]; col = C(ink) * (1 - t) + C(paper) * t
        if red: m = ((q[..., 0] - q[..., 2] > .38) & (q[..., 0] > .55) & (q[..., 1] < .62 * q[..., 0] + .05)).astype(f32); m = cv2.GaussianBlur(m, (0, 0), .8)[..., None]; col = col * (1 - m) + C('#d8142a') * m
        if edge is not None: e = np.clip(cv2.dilate(a, np.ones((7, 7), np.uint8)) - a, 0, 1)[..., None]; im = im * (1 - e) + C(edge) * e
        return im * (1 - a[..., None]) + col * a[..., None]

    def design45(self, i, layers, hit):
        """the kick as one brush stroke: it enters low left, goes through the guard and leaves the frame"""
        hx, hy = hit; P_ = spl(np.array([(-260, hy + 560), (hx * .45, hy + 250), (hx, hy), (hx + 520, hy - 120), (W + 320, hy - 120)]), n=14); pa, ik = ('#fff3e2', '#0a0408') if i == 0 else ('#0a0408', '#fff3e2')
        im = np.empty((H, W, 3), f32); im[:] = C(ik); c = Cel(); kit.focus_lines(c, hx, hy, 420, 1500, 60, 7, pa, 12 + i, .7, .9, -1.1, 1.1); im = comp.over(im, c.layer())
        im = comp.over(im, kit.drybrush(H, W, P_, 470, pa, seed=4 + i, dry=.42)); im = comp.over(im, kit.drybrush(H, W, P_ + [0, 70], 120, '#d8142a', seed=8 + i, dry=.55)); im = comp.over(im, kit.drybrush(H, W, P_ - [0, 190], 60, pa, seed=14 + i, dry=.6))
        for L in layers: im = self.toned(im, L, '#0a0408', '#fff3e2', True, '#0a0408' if i == 0 else '#fff3e2')
        c = Cel(); S = FF.star_poly(hx, hy, 46, 380, 13, 31 + i, asp=.8, rot=.2, skew=(-.12, .9)); kit.shape(c, S, '#ffffff' if i == 0 else '#d8142a', 5., '#0a0408', 3); c.fill(FF.star_poly(hx, hy, 26, 170, 9, 33 + i, asp=.8), '#d8142a' if i == 0 else '#ffffff')
        kit.splatter(c, hx + 120, hy - 20, 330, 120, 22, '#d8142a', 5 + i, -.10, 2.6); kit.splatter(c, hx + 60, hy, 420, 90, 16, pa if i else '#0a0408', 9 + i, -.05, 2.2); return comp.over(im, c.layer())

    def design63(self, i, layers, hit, pts=()):
        """two fists, and the first black lightning: black, edged red, out to every side of the frame"""
        hx, hy = hit; im = np.empty((H, W, 3), f32); im[:] = C('#fff3e2' if i == 0 else '#0a0408'); c = Cel(); kit.focus_lines(c, hx, hy, 360, 1500, 120, 8, '#0a0408' if i == 0 else '#8e0c20', 41 + i, .6, .85); kit.focus_lines(c, hx, hy, 600, 1500, 46, 26, '#0a0408' if i == 0 else '#d8142a', 47 + i, .8, .85); im = comp.over(im, c.layer())
        for L in layers: im = self.toned(im, L, '#0a0408', '#fff3e2', True, '#fff3e2' if i else '#0a0408')
        c = Cel(); r = rnd('bf63', i)
        for j in range(11):
            o = np.array(pts[j % 2] if len(pts) else (hx, hy), float); a = j / 11. * 6.283 + r.normal(0, .18); ln = r.uniform(620, 1250); q, lines = FF.bolt_tree(o, o + np.array([math.cos(a), math.sin(a) * .8]) * ln, r.uniform(20, 34), 300 + j + 40 * i, .28, .5, 2, .8, .42, 13.); FF.bolts_draw(c, q, j, '#060308', '#e8142e')
        for n_, (px, py) in enumerate(pts if len(pts) else [(hx, hy)]):
            S = FF.star_poly(px, py, 24, 150, 13, 61 + i + 5 * n_, asp=.85, rot=.4 + n_); kit.shape(c, S, '#060308', 6., '#e8142e', 3); c.fill(FF.star_poly(px, py, 15, 86, 11, 63 + i + 5 * n_, asp=.85), '#ffffff' if i == 0 else '#e8142e'); c.fill(FF.star_poly(px, py, 8, 38, 9, 66 + i + 5 * n_, asp=.85), '#e8142e' if i == 0 else '#ffffff'); kit.splatter(c, px, py, 220, 50, 11, '#e8142e', 7 + i + n_)
        return comp.over(im, c.layer())

    # -- 1: over Sukuna's shoulder, wide lens, Yuji falls on us
    def blow1(self, f):
        h = 36; k = f - h; push = lerp(f, [(33, 0.), (36, 1.), (41, 1.25)]); sh = kit.shake(f, 26, 31, 2.4, h)
        cam = Camera3D((-.14, -1.02 + .10 * push, 1.28), (-.10, 1.2, 1.56), focal=900 + 60 * push, roll=math.radians(-12 + 4 * push), cx=W / 2 + sh[0], cy=H / 2 + sh[1])
        a = st(P('twist_open', hed=(-18, 0, -10), hR=(.30, -.30, .80)), -.62, 2.7, .95, math.pi + .10); b = st(P(POSES['dive_punch'], hR=(.10, .99, -.34), hed=(-26, 4, -26), nck=(-22, 4, -30)), -.40, 1.02, .02, math.pi + .10); c_ = st(P(POSES['punch_follow']), -.95, .70, -.10, math.pi + .45)
        K_ = [(33, a, 'in'), (36, b, 'hold'), (38, b, 'out'), (42, c_, 'lin')]
        ks = self.KS1(f); Y = self.yuji().anim(lambda q: self.lerp_state(q, K_)); Y.at(f); Y._fx = dict(hood=np.array([.02, .14, .10]), hair=np.array([0, .7, .35]), hem=np.array([0, .09, .05]), handscaleR=1.25 if k < 2 else 1.1)
        K = self.suk().pose(ks['pose']).place(ks['x'], ks['y'], 0, ks['heading']); w = 1. if k < 0 else max(0., 1.5 - .12 * k); K._fx = dict(hem=np.array([0, -.05, 0]), sleeveL=np.array([.3 + .9 * (k > 1), -1.4 * w, .5 * math.sin(f * .9)]), sleeveR=np.array([.9, -.8 * w, .4 + .4 * math.sin(f * 1.1)]), hair=np.array([.1, -.9, .25]), scarf=np.array([.75 + .3 * math.sin(f * 1.1), -1.0, .55]))
        img = self.blow_bg('A', 1200 - 30 * push, 590, 1.06 + .06 * push, math.radians(12 - 4 * push), f, dof=3.5)
        LY = Y.draw(cam, f, light=(-.35, .6, .6), rim=RIM, pal=PAL, expr=dict(mood='rage', damage=1.), head_fn=yhead()); hit = np.array(Y.info['hands']['R']) + [4, 6]
        LK = K.draw(cam, f, light=(.1, .8, -.2), rim=dict(dir=(.0, .55, -.7), col='#ffb060', thr=.34, amt=.85), pal=dict(PAL, mul='#cfa39a', shmul='#7d6274'), expr=dict(mood='grin'))
        if k == 0: return self.struck([LY, LK], hit, 3)
        if k < 0: img = comp.over(img, tails(LY, (40, 220), 70, f, (2, 8)), .75); img = comp.over(img, limb_smear(Y, cam, Y.state(f - 1), Y.state(f), 'armR', pal=PAL, seed=f), .95)
        img = comp.over(img, LY); img = self.impact_fx(img, hit, f, h, 1.0, .2, ring=False); img = comp.over(img, LK)
        return fx.put(img, fx.embers(f, 46, 7, wind=(160., -260.), size=1.3))

    def KS1(self, f):
        g = P(lead='root', root=(0, -.04, -.10), pel=(-6, 4, 14), wst=(4, 4, 6), cst=(4, 4, 8), nck=(-6, 0, -4), hed=(-10, 0, 4), cL=(8, 4), cR=(2, -2), hL=(.34, .62, .22), hR=(.52, -.18, -.40), eL=(.9, .2, -.8), eR=(.6, -1, .2), gL='fist', gR='claw', wL=(-60, 0, 0), fL=(.12, .18, .043), fR=(.14, -.18, .05), kL=(.4, 1, 0), kR=(.7, .8, 0), tL=(14, 0, 0), tR=(40, 0, 0))
        b = P(g, hL=(.26, .50, .24), pel=(-12, 6, 22), root=(.01, -.07, -.15), cst=(0, 6, 12)); c_ = P(g, hL=(.62, .50, .30), hR=(.20, .30, -.20), pel=(2, -4, -26), cst=(4, -4, -18), root=(0, -.02, -.12))
        s = lambda p: st(p, .40, .02, 0., .22)
        return self.lerp_state(f, [(33, s(P(g, hL=(.10, .40, .05))), 'out'), (35, s(g), 'in'), (36, s(b), 'hold'), (38, s(b), 'out'), (42, s(c_), 'lin')])

    def lerp_state(self, f, K):
        """eased between state dicts: (frame, state, how we leave)"""
        if f <= K[0][0]: return dict(K[0][1])
        for (a, sa, e), (b, sb, _) in zip(K[:-1], K[1:]):
            if f < b:
                t = (f - a) / float(b - a)
                if e == 'hold': return dict(sa)
                t = dict(lin=t, io=t * t * (3 - 2 * t), out=1 - (1 - t) ** 2.6)[e] if e != 'in' else t ** 2.2
                q = lambda n: sa[n] * (1 - t) + sb[n] * t
                return dict(pose=mix_pose(to_root(sa['pose']), to_root(sb['pose']), t), x=q('x'), y=q('y'), z=q('z'), heading=q('heading'))
        return dict(K[-1][1])

    # -- 2: lens on the road, roundhouse into crossed arms
    def blow2(self, f):
        h = 45; k = f - h; sh = kit.shake(f, 26, 41, 2.4, h); pu = lerp(f, [(42, 0.), (45, 1.), (51, 1.3)])
        cam = Camera3D((-.20 + .30 * pu, -1.62, .11), (.03 + .10 * pu, 0, .98), focal=900, roll=math.radians(13 - 5 * pu), cx=W / 2 + sh[0], cy=H / 2 + sh[1])
        hk = -HALF - .12
        kw, kh = P(to_root(POSES['kick']), fR=(.26, .20, .20), pel=(8, -8, -8), wst=(6, 4, -18), cst=(10, 4, -30), root=(0, -.02, -.07), hL=(-.10, .52, .05), hR=(.60, .30, .30)), P(to_root(POSES['kick'])); kf = P(to_root(POSES['kick']), fR=(-.22, .44, .56), pel=(-12, -38, 84))
        KS = [(42, st(kw, -.74, .0, 0, hk - .5), 'in'), (45, st(kh, -.60, .0, 0, hk), 'hold'), (47, st(kh, -.60, .0, 0, hk), 'out'), (51, st(kf, -.56, 0, 0, hk + .5), 'lin')]
        gh = mirror(POSES['guard_high']); gb = P(gh, root=(0, -.10, -.14), pel=(-8, 0, -20), hL=(-.16, .34, .30), hR=(-.10, .38, .40)); gs = P(gh, root=(0, -.12, -.20), pel=(22, 0, -14), cst=(20, 0, -8))
        YS = [(42, st(gh, .44, .04, 0, HALF + .2), 'in'), (45, st(gb, .50, .04, 0, HALF + .2), 'hold'), (47, st(gb, .52, .04, 0, HALF + .2), 'out'), (51, st(gs, 1.05, .10, 0, HALF + .25), 'lin')]
        K = self.suk().anim(lambda q: self.lerp_state(q, KS)); K.at(f); w = max(0., 1.6 - .16 * max(k, 0)); K._fx = dict(sleeveL=np.array([-1.3 * w, .2, .5 + .3 * math.sin(f)]), sleeveR=np.array([-1.1 * w, -.4, .6]), hair=np.array([-.7, 0, .25]), hem=np.array([-.07, 0, .02]))
        Y = self.yuji().anim(lambda q: self.lerp_state(q, YS)); Y.at(f); Y._fx = dict(hood=np.array([.12, 0, .06]), hair=np.array([.8, 0, .2]), hem=np.array([.06, 0, .02]))
        img = self.blow_bg('B', 1200 + 40 * pu, 560, 1.04, math.radians(-(11 - 4 * pu)), f, dof=2.0)
        g = fx.ground_proj(cam, centre=(.42, .0)); c = Cel()
        if k >= 0: fx.ground_crack(c, g, 1.25, min(1., (k + 1) / 5.), seed=8, n=12, col='#12060a', lip='#e08a50', side='#3a1a1a', top='#6a3a30', topdark='#4a2624', glow='#ff8a3a', lift=.16); img = comp.over(img, c.layer())
        fy = cam.project(np.array([[.5, .05, 0], [-.6, 0, 0]]))[0]; sm = np.zeros((H // 2, W // 2), f32)
        for p in fy: cv2.ellipse(sm, (int(p[0] / 2), int(p[1] / 2)), (150, 18), 5, 0, 360, 1., -1)
        img = img * (1 - cv2.resize(cv2.GaussianBlur(sm, (0, 0), 7), (W, H))[..., None] * .7)
        LY = Y.draw(cam, f, light=(-.4, .6, .5), rim=RIM, pal=PAL, expr=dict(mood='rage', damage=1.)); LK = K.draw(cam, f, light=(.5, .6, .45), rim=RIM, pal=PAL, expr=dict(mood='grin'), head_fn=self.skh)
        hit = (np.array(Y.info['hands']['L']) + np.array(Y.info['hands']['R'])) / 2 + [-20, 20]
        if f in (43, 44, 45): img = comp.over(img, limb_smear(K, cam, K.state(f - 1.1), K.state(f), 'legR', pal=PAL, seed=f, tail=.9, lines=12), .95)
        img = comp.over(img, LY)
        if f in (48, 49, 50): img = comp.over(img, tails(LY, (110, 0), 46, f, (2, 6)), .6)
        img = comp.over(img, LK)
        if k >= 1: c = Cel(); p = cam.project(np.array([.55 + .1 * k, -.15, 0.]))[0]; fx.dust_skirt(c, k * 1., p[0], p[1], 300, 50, 120, seed=6, n=11, pal=DUSTP, light=(.2, -.9)); img = comp.over(img, c.layer())
        if k in (0, 1): return self.design45(k, [LY, LK], hit)
        img = self.impact_fx(img, hit, f, h, 1.0, -.2, stones=(float(fy[0][0]), float(fy[0][1])))
        return fx.put(img, fx.embers(f, 46, 9, wind=(220., -240.), size=1.4))

    # -- 3: from above, they turn round each other
    def blow3(self, f):
        h = 54; k = f - h; sh = kit.shake(f, 24, 51, 2.4, h); tw = lerp(f, [(51, 0.), (54, 1.), (60, 1.28)], lambda t: t)
        cam = Camera3D((.10, -.62, 3.35), (.02, .12, .65), focal=1120 + 60 * tw, roll=.55 - .50 * tw, cx=W / 2 + sh[0], cy=H / 2 + sh[1])
        sw = P('punch_wind', pel=(10, 8, -90), cst=(6, 8, -40)); bf = POSES['backfist']; bo = P(POSES['backfist'], pel=(16, -8, 96), hR=(.70, .60, -.1))
        YS = [(51, st(sw, .62, .22, 0, HALF - 2.6), 'in'), (54, st(bf, .50, .0, 0, HALF), 'hold'), (56, st(bf, .50, .0, 0, HALF), 'out'), (60, st(bo, .66, -.10, 0, HALF + .5), 'lin')]
        pw = P('punch_wind', gR='claw'); pm = POSES['palm']; po = P(POSES['palm'], pel=(14, -4, 60), hR=(-.5, .8, 0))
        KS = [(51, st(pw, -.78, -.1, 0, -HALF - .7), 'in'), (54, st(pm, -.62, .10, 0, -HALF), 'hold'), (56, st(pm, -.62, .10, 0, -HALF), 'out'), (60, st(po, -.74, .18, 0, -HALF + .5), 'lin')]
        Y = self.yuji().anim(lambda q: self.lerp_state(q, YS)); Y.at(f); Y._fx = dict(hood=np.array([.14, .1, .04]), hair=np.array([.6, -.4, .2])); K = self.suk().anim(lambda q: self.lerp_state(q, KS)); K.at(f); w = max(0., 1.6 - .16 * max(k, 0)); K._fx = dict(sleeveL=np.array([-1.0 * w, .9, .2]), sleeveR=np.array([-1.2 * w, -.6, .3]), hair=np.array([-.6, .2, .2]), hem=np.array([-.04, .04, 0]))
        g = cam.warp_plane(np.asarray(self.gtex[300:900, 1080:1530], f32), (-5., 13. - 300 / 45., 0.), (1, 0, 0), (0, -1, 0), 10., 600 / 45.); img = g[..., :3] * .92; del g
        x, y = grid(); img = img * (.55 + .75 * np.exp(-(((x - 960) / 900) ** 2 + ((y - 520) / 560) ** 2)))[..., None] + C('#ff7a2c') * (.10 * np.exp(-((x - 200) / 500) ** 2))[..., None]
        c = Cel(); gp = fx.ground_proj(cam, centre=(-.06, .05))
        if k >= 0: fx.ground_crack(c, gp, 1.9, min(1., (k + 1) / 5.), seed=17, n=15, col='#0c0508', lip='#f0a060', side='#3a1a1a', top='#7a4434', topdark='#4e2a26', glow='#ff9040', lift=.14, rings=4)
        img = comp.over(img, c.layer())
        sm = np.zeros((H // 2, W // 2), f32)
        for F_ in (Y, K): p = cam.project(np.array([F_.x + .10, F_.y - .06, 0.]))[0]; cv2.ellipse(sm, (int(p[0] / 2), int(p[1] / 2)), (120, 105), 0, 0, 360, 1., -1)
        img = img * (1 - cv2.resize(cv2.GaussianBlur(sm, (0, 0), 10), (W, H))[..., None] * .62)
        LK = K.draw(cam, f, light=(-.5, .3, .7), rim=dict(RIM, dir=(-.85, .3, -.3), thr=.80), pal=PAL, expr=dict(mood='grin')); LY = Y.draw(cam, f, light=(-.5, .3, .7), rim=dict(RIM, dir=(-.85, .3, -.3), thr=.80), pal=PAL, expr=dict(mood='rage'))
        hit = (np.array(K.info['hands']['R']) + np.array(Y.info['hands']['R'])) / 2
        if f in (52, 53, 54): img = comp.over(img, limb_smear(Y, cam, Y.state(f - 1.0), Y.state(f), 'armR', pal=PAL, seed=f, tail=.9), .95); img = comp.over(img, limb_smear(K, cam, K.state(f - 1.0), K.state(f), 'armR', pal=PAL, seed=f + 9, tail=.9), .9)
        if k >= 1: c = Cel(); p = cam.project(np.array([0, 0, 0.]))[0]; fx.dust_skirt(c, k * 1., p[0], p[1], 560, 500, 120, seed=8, n=22, pal=DUSTP, light=(-.5, -.8), life=.8); img = comp.over(img, c.layer(), .85)
        img = comp.over(comp.over(img, LK), LY)
        img = self.impact_fx(img, hit, f, h, 1.05, .4)
        if k == 0: return self.struck([LK, LY], hit, 7)
        return fx.put(img, fx.embers(f, 40, 12, wind=(90., -60.), size=1.6))

    skh = staticmethod(SF.head_fn('cel', pal=dict(SF.DAY, skin='#b07a64', deep='#7d4f48', light='#f0b888', hair='#a0505a', hair_d='#70343f', hair_l='#e0907c'), expr='cold'))
    skg = staticmethod(SF.head_fn('cel', pal=dict(SF.DAY, skin='#b07a64', deep='#7d4f48', light='#f0b888', hair='#a0505a', hair_d='#70343f', hair_l='#e0907c'), expr='strain'))

    # -- 4: tight side, both land
    def contact4(self):
        """both arms through, each fist in the other's face: the distance is solved, not guessed"""
        if hasattr(self, 'c4'): return self.c4
        Y, K = self.yuji(), self.suk(); py = P('punch_hit', hR=(.06, .90, .30), pel=(16, -6, 34), nck=(-8, 6, -22), hed=(-14, 10, -10), root=(-.02, .10, -.10), eR=(.8, 0, .5)); pk = mirror(P('punch_hit', hR=(.02, .86, .24), pel=(10, -4, 32), nck=(-8, 6, -24), hed=(-12, 10, -12), root=(-.02, .08, -.08), eR=(.8, 0, -.7))); best = None
        for xb in np.arange(.74, .36, -.015):
            a, b = st(py, xb, .04, 0, HALF + .10), st(pk, -xb, -.04, 0, -HALF - .10); sy, sk = (Skel(F.body, q['pose'], q['x'], q['y'], q['z'], q['heading'], F.scale) for F, q in ((Y, a), (K, b)))
            ty, tk = sk.head + np.array([.100, -.060, -.040]), sy.head + np.array([-.100, -.060, -.052]); a, e0 = aim(Y, a, 'R', ty); b, e1 = aim(K, b, 'L', tk); best = (a, b, xb)
            if max(e0, e1) < .012: break
        # the drive: heads give a little, the fists follow them
        a, b, xb = best; a2, b2 = dict(a, x=a['x'] - .016, pose=add_pose(a['pose'], hed=(-5, 5, 2), nck=(-3, 2, 1))), dict(b, x=b['x'] + .016, pose=add_pose(b['pose'], hed=(-5, -5, -2), nck=(-3, -2, -1))); sy, sk = (Skel(F.body, q['pose'], q['x'], q['y'], q['z'], q['heading'], F.scale) for F, q in ((Y, a2), (K, b2))); a2, _ = aim(Y, a2, 'R', sk.head + np.array([.086, -.066, -.044])); b2, _ = aim(K, b2, 'L', sy.head + np.array([-.086, -.066, -.056]))
        self.c4 = (a, b, xb, a2, b2); return self.c4

    def blow4(self, f):
        h = 63; k = f - h; A, B, xb, A2, B2 = self.contact4(); sh = kit.shake(f, 24, 61, 2.6, h) + kit.shake(f, 13, 64, 2.2, 65) + kit.shake(f, 3. * min(1., max(0, k) / 4.), 63); pu = lerp(f, [(60, 0.), (63, .6), (67, .66), (72, 1.15)], lambda t: t)
        cam = Camera3D((.0, -1.86 + .42 * pu, 1.43), (.0, 0, 1.43), focal=1500, roll=math.radians(-6 + 3.0 * pu), cx=W / 2 + sh[0], cy=H / 2 + sh[1])
        yw, kw = P('punch_wind', hL=(.04, .46, .06), gL='fist'), mirror(P('punch_wind', hL=(.04, .46, .06), gL='fist'))
        YS = [(60, st(yw, xb + .42, .04, 0, HALF + .10), 'in'), (63, A, 'hold'), (63.001, A, 'out'), (69, A2, 'lin')]; KS = [(60, st(kw, -xb - .40, -.04, 0, -HALF - .10), 'in'), (63, B, 'hold'), (63.001, B, 'out'), (69, B2, 'lin')]
        fy = (lambda q: self.lerp_state(q, YS[:2]) if q < 63 else A) if f < 67 else (lambda q: self.lerp_state(63.001 + min(5.9, max(0., q - 66.)), YS[2:])); fk = (lambda q: self.lerp_state(q, KS[:2]) if q < 63 else B) if f < 67 else (lambda q: self.lerp_state(63.001 + min(5.9, max(0., q - 66.)), KS[2:]))
        Y = self.yuji().anim(fy); Y.at(f); g = math.sin(f * 1.1); Y._fx = dict(hood=np.array([.14, 0, .06 + .02 * g]), hair=np.array([.9 + .1 * g, 0, .25]), hem=np.array([.08, 0, .03])); K = self.suk().anim(fk); K.at(f); K._fx = dict(sleeveL=np.array([-1.5, .1, .5 + .3 * g]), sleeveR=np.array([-1.6, -.2, .4 - .3 * g]), hair=np.array([-.9, 0, .3 + .1 * g]), scarf=np.array([-1.0, 0, .30 + .2 * g]))
        img = self.blow_bg('A', 1120 - 120 * (f < 63) * (63 - f) / 3., 600, 1.22 + .06 * pu, math.radians(6 - 3.0 * pu), f, 90 if f < 63 else 0, 0, 4.0)
        sy, sk = Y.skel(f), K.skel(f); cy_, ck = (cam.project(np.array([fist_at(q, s_)]))[0][0] for q, s_ in ((sy, 'R'), (sk, 'L'))); dy, dk = (cam.project(np.array([q.wr[s_]]))[0][0] for q, s_ in ((sy, 'R'), (sk, 'L'))); dy, dk = (cy_ - dy) / np.hypot(*(cy_ - dy)), (ck - dk) / np.hypot(*(ck - dk)); pm = 1500 / 1.5; cy_, ck = cy_ + dy * .030 * pm, ck + dk * .030 * pm            # where knuckles meet the face
        if k >= 0: amt = (.44, .44, .46, .46, .48, .46, .45, .44, .44)[min(8, int(k))]; s0 = .123 * pm; K.head_warp = squash(cy_, dy, s0, amt); Y.head_warp = squash(ck, dk, s0, amt)
        LK = K.draw(cam, f, light=(.55, .55, .45), rim=RIM, pal=PAL, expr=dict(mood='grin', look=(.3, .1)), head_fn=self.skh if k < 0 else self.skg); LY = Y.draw(cam, f, light=(-.55, .55, .45), rim=RIM, pal=PAL, expr=dict(mood='rage', damage=1.), head_fn=yhead('snarl' if k < 0 else 'rage', wind=(.9, .1)))
        # his forearm and fist lie over Yuji's face
        m = cv2.dilate(K.mask('handL', 'farmL', 'fist'), np.ones((5, 5), np.uint8)) if k >= 0 else None; ov = LK * np.clip(cv2.GaussianBlur(m, (0, 0), .7), 0, 1)[..., None] if m is not None else None
        hit = (cy_ + ck) / 2
        if f in (61, 62): img = comp.over(img, limb_smear(K, cam, K.state(f - 1.0), K.state(f), 'armL', pal=PAL, seed=f, tail=.9), .9); img = comp.over(img, limb_smear(Y, cam, Y.state(f - 1.0), Y.state(f), 'armR', pal=PAL, seed=f + 5, tail=.9), .95); img = comp.over(img, tails(LK, (190, 0), 50, f, (2, 7)), .6); img = comp.over(img, tails(LY, (-190, 0), 50, f + 3, (2, 7)), .6)
        img = comp.over(comp.over(img, LK), LY)
        if ov is not None: img = comp.over(img, ov)
        if k in (0, 1): return self.design63(k, [LK, LY] + ([ov] if ov is not None else []), hit, (cy_, ck))
        if k >= 2:
            # knuckle to bone: creases driven ahead of the fist, spit and sparks off the far side
            c = Cel(); r = rnd('cr', f // 2)
            for (q, d_, sd) in ((cy_, dy, 1), (ck, dk, 2)):
                n_ = np.array([-d_[1], d_[0]])
                for j in range(3): t = np.linspace(-1, 1, 9); R_ = (.26 + .13 * j) * 123; P_ = q + d_ * 8 + np.outer(np.cos(t * .9) * R_, d_) * .62 + np.outer(np.sin(t * .9) * R_, n_); c.line(P_, 3.0 - .6 * j, '#4a1414', taper=(.5, .5), seed=j + sd, alpha=.75 - .18 * j)
                for j in range(7): a = r.normal(0, .75); u = d_ * math.cos(a) + n_ * math.sin(a); d0 = r.uniform(70, 110) + 16 * (k - 2); l = r.uniform(30, 90) * max(.2, 1 - .12 * k); c.line(np.array([q + u * d0, q + u * (d0 + l * .5) + [0, 2], q + u * (d0 + l)]), r.uniform(2., 5.), '#fff4e0' if j % 2 else '#ffc890', taper=(.1, .7), seed=j)
            img = comp.over(img, c.layer())
            for q, d_ in ((cy_, dy), (ck, dk)): img = self.impact_fx(img, q - d_ * 34, f, h - 1, .62, .1, black=.7, ring=False)
            # it does not fade: the black lightning starts to crawl over both fists
            c = Cel(); r = rnd('crawl', f // 1)
            for j in range(6 + 2 * k):
                o, d_ = ((ck, dk), (cy_, dy))[j % 2]; o = o - d_ * r.uniform(30, 150); a = r.uniform(0, 6.283); L_ = r.uniform(50, 120 + 24 * k); P_ = FF.bolt_line(o + r.normal(0, 26, 2), o + np.array([math.cos(a), math.sin(a) * .8]) * L_, int(r.integers(9999)), .36, 6)
                q_ = r.uniform(.7, 1.35)
                for w_, col in ((14, '#e8162e'), (8.4, '#070308')): c.line(P_, w_ * q_, col, taper=(.15, .7), seed=j, press=.1, tmin=.05, tremor=0)
            img = comp.over(img, c.layer())
            for q in (cy_, ck): img = comp.screen(img, kit.radial(q[0], q[1], 330 + 24 * k, power=2.)[..., None] * C('#ff2a3a') * (.10 + .02 * k))
        return fx.put(img, fx.embers(f, 50, 15, wind=(120., -240.), size=1.5))

    def frame(self, f):
        if f < 33: im = self.track(f)
        elif f < 42: im = self.blow1(f)
        elif f < 51: im = self.blow2(f)
        elif f < 60: im = self.blow3(f)
        else: im = self.blow4(f)
        if f in (33, 42, 51, 60): im = streak(im, *{33: (0, 120), 42: (150, 0), 51: (0, -140), 60: (-160, 0)}[f])                # whip into each layout
        return im.astype(f32)


def streak(img, dx, dy):
    """clean shutter streak (a line kernel, no stacked copies)"""
    if abs(dy) < 1: return cv2.blur(img, (int(abs(dx)) | 1, 1))
    if abs(dx) < 1: return cv2.blur(img, (1, int(abs(dy)) | 1))
    q = cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA); L = math.hypot(dx, dy) / 2; n = int(L) | 1; k = np.zeros((n, n), np.uint8); c = n / 2.; u = np.array([dx, dy]) / math.hypot(dx, dy) * (L / 2 - 1)
    cv2.line(k, tuple(int(v * 16) for v in (c - u[0], c - u[1])), tuple(int(v * 16) for v in (c + u[0], c + u[1])), 255, 1, cv2.LINE_AA, 4); k = k.astype(f32); return cv2.resize(cv2.filter2D(q, -1, k / k.sum()), (W, H), interpolation=cv2.INTER_LINEAR)


def blit(img, q, x, y):
    """premultiplied RGBA sprite over RGB in place"""
    h, w = q.shape[:2]; x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0: return
    s = q[y0 - y:y1 - y, x0 - x:x1 - x]; s = s.astype(f32) / 255. if s.dtype == np.uint8 else s; img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - s[..., 3:4]) + s[..., :3]


def add_glow(img, cx, cy, r, col):
    x0, y0, x1, y1 = int(max(0, cx - r)), int(max(0, cy - r)), int(min(W, cx + r)), int(min(H, cy + r))
    if x1 <= x0 or y1 <= y0: return
    y, x = np.mgrid[y0:y1:4, x0:x1:4].astype(f32); d = np.clip(1 - np.hypot(x - cx, y - cy) / r, 0, 1) ** 2; d = cv2.resize(d, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)[..., None]; img[y0:y1, x0:x1] = 1 - (1 - np.clip(img[y0:y1, x0:x1], 0, None)) * (1 - np.clip(d * col, 0, .95))
