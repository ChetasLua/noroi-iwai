"""e4  THREE SHADOWS.  みっつのかげが ひとつになる
Low over wet asphalt, looking toward one street lamp.  Three pairs of shoes in its glow, three long shadows running at us.  Two steps, then stillness.
One shadow comes apart in petals and her shoes are gone; one is drunk by a black puddle and his shoes are gone; one bows its head.  Rain."""
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT)
import numpy as np, cv2, math
from jj import W, H, comp, kit
from jj.paint import rgb, f32, fbm, smooth, mix, grid
from jj.ink import Cel, spl, paste
from jj.cam import Camera3D
from jj.figure import Fig, P, POSES, walk, add_pose, mix_pose, idle

VER = 'e4_24'
HOR = 64.                                         # horizon row: nearly all of the frame is road
CZ = .80
CAM = Camera3D((0, 0, CZ), (0, 10., CZ - 10. * (540 - HOR) / 1500.), focal=1500)
LAMP = np.array([.10, 9.87, 6.0])                 # metres: x right, y away from us, z up
FEET, STEP = 3.80, .36
WHO = dict(megumi=-.52, yuji=.0, nobara=.50)
PG = 240.; GX, GY, GW, GH = -2.5, .45, 5.0, 6.2     # ground raster: origin, size (metres), px per metre
TW, TH = int(GW * PG), int(GH * PG)
HGS = CAM.H_plane((GX, GY, 0.), (1, 0, 0), (0, 1, 0), TW, TH, GW, GH)
HINV = np.linalg.inv(HGS)
MC = np.array([WHO['megumi'] * 1.12, FEET - .40])                       # where the black puddle opens, on the road


def C(c): return rgb(c) if isinstance(c, str) else np.asarray(c, f32)
def rnd(*s): return np.random.default_rng(abs(hash(s)) % (1 << 31))
def gp(Q): Q = np.atleast_2d(np.asarray(Q, float)); return np.c_[(Q[:, 0] - GX) * PG, (Q[:, 1] - GY) * PG]          # road metres -> ground raster px
def s2g(Q): q = np.c_[np.atleast_2d(Q), np.ones(len(np.atleast_2d(Q)))] @ HINV.T; return q[:, :2] / q[:, 2:3]                 # screen px -> ground raster px
def sp(Q): Q = np.atleast_2d(np.asarray(Q, float)); return CAM.project(np.c_[Q, np.zeros(len(Q))] if Q.shape[1] == 2 else Q)[0]


class Shot:
    id = 'e4'; n = 72
    finish = dict(bloom=.34, vig=.24, grain=.014, halation='#ffcf96')
    EVENTS = [(0, 'step', dict(size=.4, note='three pairs of shoes on wet asphalt, coming toward us')), (9, 'step', dict(size=.4, note='second step')), (18, 'step', dict(size=.25, note='they stop; cloth settles')), (0, 'hum', dict(size=.3, dur=72, note='street lamp hum')),
              (27, 'tremble', dict(size=.3, dur=5, note="Nobara's shadow shivers")), (31, 'petals', dict(size=.7, dur=24, note='it comes apart in petals that blow past the lens: dry rustle, close')), (44, 'gone', dict(size=.2, note='her shoes are not there any more: a small silence')),
              (45, 'raindrop', dict(size=.3, note='first drop')), (48, 'raindrop', dict(size=.25)), (50, 'raindrop', dict(size=.25)), (53, 'raindrop', dict(size=.3)), (54, 'sink', dict(size=.8, dur=10, note="a black puddle drinks Megumi's shadow: low wet gulp, sub")), (61, 'gone', dict(size=.2, note='his shoes go under')), (64, 'ripple', dict(size=.4, note='it closes; one ripple')), (58, 'rain', dict(size=.5, dur=14, note='rain thickens')),
              (63, 'fist', dict(size=.35, note="Yuji's shadow lowers its head, its fists close: cloth, knuckles, a breath out")), (69, 'bubble', dict(size=.2, note='last bubble breaks'))]

    # ------------------------------------------------------------ setup
    def setup(self):
        os.makedirs('cache', exist_ok=True); d = CAM.rays(); t = CAM.plane_t((0, 0, 0), (0, 0, 1.), d); self.ok = (t > .05) & (t < 400); self.X = np.where(self.ok, d[..., 0] * t, 0).astype(f32); self.Y = np.where(self.ok, d[..., 1] * t, 1e3).astype(f32); self.t = t
        ks = {n: f'cache/{VER}_{n}.npy' for n in ('amb', 'lampl', 'pud')}
        if not all(os.path.exists(k) for k in ks.values()):
            self.paint_road(d)
            for n, k in ks.items(): np.save(k, getattr(self, n).astype(np.float16))
        self.amb, self.lampl, self.pud = (np.load(ks[n], mmap_mode='r').astype(f32) for n in ('amb', 'lampl', 'pud'))
        self.sils = {}; self.legs = {}
        for who in WHO:
            k = f'cache/{VER}_sil_{who}.npy'; n = dict(nobara=16, megumi=28, yuji=36)[who]
            if not os.path.exists(k): np.save(k, np.stack([(np.clip(self.sil(who, i * 2), 0, 1) * 255).astype(np.uint8)[::2, ::2] for i in range(n)]))
            self.sils[who] = np.load(k, mmap_mode='r'); k = f'cache/{VER}_legs_{who}.npz'; n = dict(nobara=24, megumi=33, yuji=36)[who]
            if not os.path.exists(k):
                o = {}
                for i in range(n): roi, box = self.leg_draw(who, i * 2 if i * 2 < 24 else 24 + (i * 2 - 24) // 6 * 6); o['a%d' % i] = (np.clip(roi, 0, 1) * 255).astype(np.uint8); o['b%d' % i] = np.array(box)
                np.savez_compressed(k, **o)
            z = np.load(k); self.legs[who] = [(z['a%d' % i], z['b%d' % i]) for i in range(n)]
        self.petal_field()
        r = rnd('drops'); T = np.sort(np.r_[45, 48, 50, 53, 54.5, 56 + 16 * r.random(120) ** .75]); self.drops = []
        for i, tt in enumerate(T): y = 1.05 + 5.2 * r.random() ** 1.7; x = r.uniform(-1, 1) * (.62 * y + .25); self.drops.append((float(tt), x, y, float(r.uniform(.010, .020)), i))
        for i, (x, y) in enumerate(((-.50, 1.7), (.62, 1.5), (.30, 2.6), (-1.0, 2.2), (1.15, 2.4))): self.drops[i] = (self.drops[i][0], x, y, .02, i)

    # ------------------------------------------------------------ the road
    def paint_road(self, d):
        X, Y, t = self.X, self.Y, np.maximum(self.t, .05); S = 1024
        def tile(scales, wt, seed): return fbm(S, S, scales, wt, seed)
        def smp(T_, mpp, ox=0., oy=0.): tri = lambda q: (S - 1) - np.abs((q % (2 * S - 2)) - (S - 1)); return cv2.remap(T_, tri(X / mpp + ox + 400).astype(f32), tri(Y / mpp + oy).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        ag = np.random.default_rng(3).standard_normal((S, S)).astype(f32); ag = cv2.GaussianBlur(ag, (0, 0), 1.3); ag /= ag.std(); fine = np.exp(-(t / 3.2) ** 2)
        wide = smp(tile((240, 90, 34), (1, .6, .4), 4), .0060); mid = smp(tile((60, 22), (1, .6), 5), .0034, 300, 120); g_ = smp(ag, .0016)
        alb = np.clip(.30 + .045 * wide + .025 * mid + .050 * g_ * fine, .06, .8)
        # a patch, a tar seam, one crack: in road metres
        c = Cel(); c.fill(sp([(.95, 1.2), (2.6, 1.35), (2.9, 3.7), (1.12, 3.5)]), '#ffffff', .30); seam = [(.95, 1.2), (1.0, 1.9), (1.05, 2.7), (1.12, 3.5), (1.9, 3.62), (2.9, 3.7)]
        q = np.array(seam); n_ = 26; s_ = np.r_[0, np.cumsum(np.hypot(*np.diff(q, axis=0).T))]; u = np.linspace(0, s_[-1], n_); q = np.c_[np.interp(u, s_, q[:, 0]), np.interp(u, s_, q[:, 1])]; r = np.random.default_rng(6); q = q + r.normal(0, .012, q.shape); w = .016 * r.uniform(.4, 1.3, n_); c.fill(sp(np.vstack([q + np.c_[w, w * 0], (q - np.c_[w, w * 0])[::-1]])), '#ffffff', .8)
        P_ = [np.array([-2.1, 1.25])]; h = .30
        for i in range(34): h = .34 + .5 * math.sin(i * .45) + r.normal(0, .30); P_.append(P_[-1] + np.array([math.cos(h), math.sin(h) * .8]) * r.uniform(.03, .075))
        P_ = np.array(P_); w = .016 * (1 - np.linspace(0, 1, len(P_))) ** .6 * r.uniform(.5, 1.5, len(P_)) + .0012; c.fill(sp(np.vstack([P_ + np.c_[w * 0, w], (P_ - np.c_[w * 0, w])[::-1]])), '#ffffff', 1.); lip = Cel(); lip.fill(sp(np.vstack([P_ + np.c_[w * 0, w * 1.0 + .008], (P_ + np.c_[w * 0, w])[::-1]])), '#ffffff')
        m = c.rgba()[..., 3]; lp = lip.rgba()[..., 3]; alb = alb * (1 - .82 * m)
        # two puddles: hand-drawn, still
        rr_ = np.random.default_rng(12)
        def ragged(cx, cy, rx, ry, seed):
            th = np.linspace(0, 2 * math.pi, 150, endpoint=False); q_ = np.random.default_rng(seed); k = 1 + sum(q_.uniform(.03, .13) / (1 + .25 * j) * np.sin(j * th + q_.uniform(0, 6.28)) for j in range(2, 13)); return np.c_[cx + np.cos(th) * rx * k, cy + np.sin(th) * ry * k]
        pc = Cel(); pc.fill([sp(ragged(-1.62, 2.70, .50, .30, 5)), sp(ragged(-1.02, 1.36, .16, .09, 3)), sp(ragged(1.42, 2.45, .42, .42, 8)), sp(ragged(1.06, 1.62, .12, .10, 4))], '#ffffff'); pud = pc.rgba()[..., 3]; damp = cv2.GaussianBlur(pud, (0, 0), 16)
        # normals: aggregate near us, long swells everywhere, almost none in the puddles
        bx = smp(cv2.Sobel(ag, cv2.CV_32F, 1, 0, ksize=3), .0014) * .0135 * fine + smp(cv2.Sobel(tile((34, 13), (1, .7), 8), cv2.CV_32F, 1, 0), .0030) * .085; by = smp(cv2.Sobel(ag, cv2.CV_32F, 0, 1, ksize=3), .0014) * .0135 * fine + smp(cv2.Sobel(tile((34, 13), (1, .7), 8), cv2.CV_32F, 0, 1), .0030) * .085
        k = 1 - .90 * pud; N = np.stack([-bx * k, -by * k, np.ones_like(bx)], -1); N /= np.linalg.norm(N, axis=-1, keepdims=True)
        V = d / np.linalg.norm(d, axis=-1, keepdims=True); R = V - 2 * np.sum(V * N, -1, keepdims=True) * N; PT = np.stack([X, Y, np.zeros_like(X)], -1); LV = LAMP.astype(f32) - PT; r_ = np.linalg.norm(LV, axis=-1); LV /= r_[..., None]; cs = np.clip(np.sum(R * LV, -1), 0, 1)
        spec = np.exp((cs - 1) / .010) * .62 + np.exp((cs - 1) / .060) * .34 + np.exp((cs - 1) / .22) * .12; sp_p = np.exp((cs - 1) / .0032) * 1.5 + np.exp((cs - 1) / .030) * .45 + np.exp((cs - 1) / .16) * .16
        spec = spec * (1 - pud) * (.72 + .28 * smooth(-.6, .8, wide)) + sp_p * pud
        r0 = np.linalg.norm(LAMP - np.array([0, FEET, 0])); E = (r0 / np.maximum(r_, 1.)) ** 2.1 * np.clip(LV[..., 2] / (LAMP[2] / r0), 0, 2.4)
        wedge = np.exp(-((X - .02) / (.66 + .30 * np.clip(Y, 0, 12))) ** 2 * .85) * smooth(22, 8.0, Y) ; wedge = wedge * (1 + .06 * wide)
        lampc = C('#ffa94e') * 2.35; specc = C('#ffd49a')
        self.lampl = (alb[..., None] * lampc * (E * wedge * (1 - .72 * pud) * (1 - .22 * damp))[..., None] + specc * (spec * (.30 + .75 * wedge))[..., None] + C('#ffe0b0') * (lp * wedge * .25)[..., None]) * self.ok[..., None]
        night = C('#221b48') * 1.15; fres = np.clip(1 - np.abs(V[..., 2]), 0, 1) ** 3.2; amb = alb[..., None] * night + C('#3a2c62') * (.10 * fres + .05)[..., None]; xs_ = np.arange(W, dtype=f32)[None] - 975; sky = C('#1c1840') * (.30 + .50 * fres)[..., None] * (1 + .35 * np.sin(np.arange(H, dtype=f32)[:, None] * .16 + wide * 1.2) * smooth(240, 520, np.arange(H, dtype=f32)[:, None]))[..., None] + C('#ffb05c') * (np.exp(-(xs_ / 520) ** 2) * np.exp(-((np.arange(H, dtype=f32)[:, None] - HOR) / 420)) * .55)[..., None]; amb = amb + (sky - amb) * (pud * .86)[..., None]
        yy = np.arange(H, dtype=f32)[:, None]; amb = amb * (1 - .30 * smooth(700, 1080, yy))[..., None]; e0 = cv2.GaussianBlur(pud, (0, 0), 1.4); ed = np.clip(e0 * (1 - e0) * 4, 0, 1); amb = amb * (1 - .2 * ed[..., None]); self.lampl = self.lampl * (1 - .30 * ed[..., None]) + C('#ffdca4') * (np.clip(kit.shifted(e0, 0, 2.0) - e0, 0, 1) * wedge * .55)[..., None]
        # above the horizon: dark street, a few lights out of focus
        x, y = grid(); bg = C('#171230')[None, None] * (1 + .5 * np.exp(-((x - 960) / 620) ** 2))[..., None]; amb = np.where(self.ok[..., None], amb, bg); amb = amb.astype(f32)
        for lx, ly, lr, col, ai in ((90, 40, 13, '#9a78ff', .20), (214, 22, 17, '#b06cff', .18), (330, 48, 9, '#ffb060', .40), (392, 34, 12, '#ff8a5c', .22), (1452, 50, 8, '#ffcf8c', .38), (1530, 30, 14, '#8f6cff', .20), (1668, 44, 11, '#ffb866', .30), (1800, 20, 17, '#ff7a6a', .18), (1870, 50, 8, '#b090ff', .24)):
            disc = smooth(lr + 4, lr - 4, np.hypot(x - lx, y - ly)); amb += (disc * ai)[..., None] * C(col); st = np.exp(-((x - lx) / (lr * .9)) ** 2) * np.where(y > HOR, np.exp(-np.maximum(y - HOR, 0) / 110.), 0) * (.70 + .5 * wide); amb += (np.clip(st, 0, 1) * ai * .60)[..., None] * C(col)
        self.amb = amb; self.lampl = np.nan_to_num(np.minimum(self.lampl, 3.2)).astype(f32); self.pud = pud.astype(f32)

    # ------------------------------------------------------------ the people
    def state(self, who, f):
        ph = dict(megumi=.06, yuji=0., nobara=-.05)[who]
        if f < 18: d = 2 * STEP * f / 18.; p = walk(f / 18. + ph)
        else:
            t = min(1., (f - 18) / 5.); d = 2 * STEP + .02 * (1 - (1 - t) ** 2); p = mix_pose(walk(1. + ph), self.stand(who), t * t * (3 - 2 * t)); p = idle(p, f, 1.5, dict(megumi=1.2, yuji=0., nobara=2.4)[who])
        if who == 'yuji' and f >= 62:
            k = min(1., (f - 62) / 8.); k = k * k * (3 - 2 * k); p = dict(p, gL='fist' if f >= 63 else 'soft', gR='fist' if f >= 63 else 'soft'); p = add_pose(p, nck=(30 * k, 12 * k, 0), hed=(22 * k, 14 * k, 0), cst=(9 * k, 0, 0), wst=(4 * k, 0, 0), cL=(-2.2 * k, 1.5 * k), cR=(-2.2 * k, 1.5 * k))
        return dict(pose=p, x=WHO[who], y=FEET + 2 * STEP - d, z=0., heading=math.pi + dict(megumi=.06, yuji=.0, nobara=-.06)[who])

    def stand(self, who):
        if who == 'nobara': return P('contra', hR=(.25, .05, -.92), gR='fist', hL=(.20, .03, -.95), gL='soft', fL=(.085, .02, .043), fR=(.080, -.02, .043), tL=(16, 0, 0), tR=(14, 0, 0))
        if who == 'megumi': return P('pockets', hL=(.0, .10, -.70), hR=(.0, .10, -.70), eL=(.7, -.6, .1), eR=(.7, -.6, .1), gL='fist', gR='fist')
        return P('contra', hL=(.19, .04, -.94), hR=(.19, .04, -.94), gL='soft', gR='soft', wL=(-80, 0, 0), wR=(-80, 0, 0), fL=(.10, .03, .043), fR=(.10, -.01, .043), tL=(18, 0, 0), tR=(16, 0, 0))

    def fig(self, who, f):
        s = self.state(who, f); F = (Fig(who, seed=2, spec=dict(shoe='ankle'), body=dict(limb=.90, foot=.94)) if who == 'nobara' else Fig(who, seed=2)).pose(s['pose']).place(s['x'], s['y'], 0, s['heading']); g = .5 + .5 * math.sin(f * .31 + WHO[who]); F.blow(.28 + .2 * g, 0, 0); return F, s

    def leg_draw(self, who, f):
        """what we see of them: shoes and legs, back-lit"""
        F, s = self.fig(who, f); sk = F.skel(); R_ = F.draw(CAM, f, light=(.15, .45, -.9), kick=(-.5, .4, -.75), key=.0, rim=dict(dir=(.08, .42, -.90), col='#ffbe6e', thr=.36, amt=.9), pal=dict(mul='#7d6870', shmul='#5a4a62', litmul='#ffb878', ink='#0c0912'), head=False, layer=False, weight=1.15)
        return R_

    @staticmethod
    def drop(Q):
        """world point(s) -> road metres of the shadow the lamp throws of them"""
        Q = np.atleast_2d(np.asarray(Q, float)); k = LAMP[2] / (LAMP[2] - Q[:, 2:3]); return LAMP[:2] + (Q[:, :2] - LAMP[:2]) * k

    HEADS = dict(      # authored as the eye sees them: x across, y toward us; 1 = half a head; chin -1, crown +1
        nobara=[(-.34, -1.55), (-.35, -1.00), (-.52, -.92), (-.80, -.92), (-.98, -.80), (-1.05, -.50), (-1.08, -.10), (-1.04, .30), (-.90, .66), (-.62, .94), (-.24, 1.08), (.10, 1.09), (.48, 1.00), (.80, .76), (1.00, .40), (1.08, .0), (1.06, -.42), (1.00, -.78), (.82, -.92), (.54, -.92), (.35, -1.00), (.34, -1.55)],
        megumi=[(-.36, -1.45), (-.37, -.98), (-.68, -.70), (-.86, -.28), (-.96, .02), (-1.30, .22, 1), (-1.00, .36), (-1.48, .98, 1), (-.80, .80), (-.78, 1.00), (-.86, 1.66, 1), (-.28, 1.02), (-.02, 1.10), (.30, 1.72, 1), (.44, 1.00), (.64, .92), (1.36, 1.24, 1), (.92, .52), (.98, .34), (1.36, .30, 1), (.96, -.02), (.84, -.32), (.68, -.70), (.38, -.98), (.36, -1.45)],
        yuji=[(-1.02, -1.62), (-1.10, -1.36), (-.96, -1.12), (-.58, -.98), (-.66, -.68), (-.84, -.34), (-.92, -.02), (-1.06, .28, 1), (-.94, .40, 1), (-1.04, .72, 1), (-.74, .72, 1), (-.66, 1.10, 1), (-.36, .94, 1), (-.12, 1.28, 1), (.10, .98, 1), (.46, 1.20, 1), (.56, .88, 1), (.96, .94, 1), (.88, .58, 1), (1.12, .34, 1), (.94, .06), (.84, -.36), (.64, -.72), (.60, -.98), (.98, -1.12), (1.12, -1.38), (1.02, -1.62)])
    HS = dict(nobara=112., megumi=108., yuji=114.)

    def sil(self, who, f):
        """shadow on the road (ground raster): the lamp's view of the body, plus a head, skirt and hammer drawn for the shadow"""
        F, s = self.fig(who, f); cam = Camera3D(LAMP, (s['x'], s['y'], .80), focal=2300, w=900, h=1200); F._fx['handscaleL'] = F._fx['handscaleR'] = 1.25 if who == 'yuji' else 1.12
        F.draw(cam, f, ink_only=True, head=False); D = F._last; a = np.clip((cv2.GaussianBlur(F.mask(*[n for n in D.lname if n != 'head']), (0, 0), .8) - .5) * 2 + .5, 0, 1)
        m = cv2.warpPerspective(a, cam.H_plane((GX, GY, 0.), (1, 0, 0), (0, 1, 0), TW, TH, GW, GH), (TW, TH), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
        sk = F.skel(); pt = lambda Q: gp(self.drop(Q)); fo = pt(np.array([s['x'], s['y'], 0.]))[0]; to = pt(sk.head)[0]; vv, uu = np.mgrid[0:TH, 0:TW].astype(f32); axx = fo[0] + (to[0] - fo[0]) * np.clip((vv - fo[1]) / (to[1] - fo[1]), 0, 1.1); nar = .90 if who != 'nobara' else .94; m = cv2.remap(m, (axx + (uu - axx) / nar).astype(f32), vv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT); c = Cel(TW, TH, 0, 0, ss=1)
        k = 0.
        if who == 'yuji' and f >= 62: k = min(1., (f - 62) / 8.); k = k * k * (3 - 2 * k)
        st_ = lambda Q: sp(self.drop(Q))[0]; N, C_, B_ = st_(sk.neck), st_(sk.head), st_(sk.chest + sk.Rc[:, 2] * .05); dep = float(self.drop(sk.head)[0, 1]); hs = self.HS[who] * 1.60 / max(dep, .6); ax = np.array([s['x'] * .25, 1.]); ax = (C_ - N) / (np.hypot(*(C_ - N)) + 1e-6) * .84 + np.array([0, 1.]) * .16; ax /= np.hypot(*ax); rt = np.array([ax[1], -ax[0]]); C_ = N + ax * hs * 1.36
        q = np.array([(v[0], v[1]) for v in self.HEADS[who]], float); cor = [i for i, v in enumerate(self.HEADS[who]) if len(v) > 2]; hd = spl(q, True, corners=[0, len(q) - 1] + cor, n=5); w_ = smooth(-1.2, -.7, hd[:, 1])[:, None]; tip = .40 * k; piv = np.array([0., -.95]); R = np.array([[math.cos(tip), math.sin(tip)], [-math.sin(tip), math.cos(tip)]])
        hd = hd * (1 - w_) + ((hd - piv) @ R.T * np.array([1., 1 - .22 * k]) + piv + [.10 * k, -.52 * k]) * w_; hd[:, 1] += np.sin(f * .5 + hd[:, 0] * 3) * .012 * w_[:, 0]; toG = lambda Q: s2g(np.asarray(Q))
        c.fill(toG(C_ + rt * (hd[:, 0:1] * hs) + ax * (hd[:, 1:2] * hs * 1.10)), '#ffffff'); wd = hs * (.42 if who == 'nobara' else .44); tp = C_ - ax * hs * .6; c.fill(toG(np.array([tp + rt * wd, N + rt * wd, B_ + rt * wd * 1.9, B_ - rt * wd * 1.9, N - rt * wd, tp - rt * wd])), '#ffffff')
        if who == 'nobara':
            ls = np.array([math.cos(s['heading']), math.sin(s['heading']), 0.]); o = np.array([s['x'], s['y'], 0.]); wv = .02 * math.sin(f * .4); sk_ = [(-.125, 1.0), (-.17, .92), (-.235 + wv, .74), (-.30 + wv, .56), (-.21, .515), (-.12, .55), (0., .505), (.11, .55), (.21, .515), (.31 + wv, .56), (.24, .75), (.17, .92), (.125, 1.0)]; c.fill(spl(pt([o + ls * u + [0, 0, z] for u, z in sk_]), True, corners=(0, 3, 9, 12), n=4), '#ffffff')
            h = sk.wr['L'] + sk.Rhand['L'][:, 1] * .07; e = np.array([ls[0] * .55, ls[1] * .55, .84]); e /= np.linalg.norm(e); n_ = np.array([-e[2] * ls[0], -e[2] * ls[1], e @ np.array([ls[0], ls[1], 0])]); q_ = lambda u, v: h + e * u + n_ * v
            e = np.array([-ls[0] * .62, -ls[1] * .62, -.78]); e /= np.linalg.norm(e); n_ = np.array([-ls[0] * .78, -ls[1] * .78, .62]); c.fill(pt([q_(-.06, -.014), q_(.25, -.012), q_(.25, .012), q_(-.06, .014)]), '#ffffff'); c.fill(pt([q_(.22, -.070), q_(.29, -.070), q_(.285, .03), q_(.32, .080), q_(.29, .11), q_(.235, .055), q_(.225, .0)]), '#ffffff')
        return np.maximum(m, c.rgba()[..., 3])

    def ground(self, who, f):
        a = self.sils[who]; i = min(int(f) // 2, len(a) - 1); m = cv2.resize(np.asarray(a[i]), (TW, TH), interpolation=cv2.INTER_CUBIC).astype(f32) / 255.; return np.clip((np.clip(m, 0, 1) - .5) * 1.9 + .5, 0, 1)

    # ------------------------------------------------------------ Nobara
    def petal_field(self):
        m = self.ground('nobara', 30); r = rnd('pet'); pts = []; occ = {}; cell = .060; ys, xs = np.nonzero(m[::3, ::3] > .5)
        for j in r.permutation(len(ys)):
            x, y = GX + (xs[j] * 3 + r.uniform(0, 3)) / PG, GY + (ys[j] * 3 + r.uniform(0, 3)) / PG; kx, ky = int(x / cell), int(y / (cell * 1.7))
            if (kx, ky) in occ: continue
            occ[kx, ky] = 1; pts.append((x, y))
        self.pts = np.array(pts); n = len(pts); y = self.pts[:, 1]; u = (y - y.min()) / (np.ptp(y) + 1e-6); self.rel = 31.5 + 11. * np.clip(u * .85 + r.uniform(0, .18, n), 0, 1)          # the head, nearest us, goes first
        from scipy.spatial import cKDTree
        vv, uu = np.mgrid[0:TH:3, 0:TW:3]; G_ = np.c_[GX + uu.ravel() / PG, (GY + vv.ravel() / PG) / 1.7]; _, q = cKDTree(np.c_[self.pts[:, 0], self.pts[:, 1] / 1.7]).query(G_); self.near = q.reshape(vv.shape).astype(np.int32)
        self.prop = np.c_[r.uniform(.75, 1.3, n), r.uniform(0, 6.28, n), r.uniform(-1, 1, n), r.uniform(0, 1, n), r.uniform(0, 1, n), r.uniform(0, 6.28, n)]
        # her legs: cells of the drawing itself
        a, b = self.legs['nobara'][20]; al = a[..., 3] > 120; ys, xs = np.nonzero(al); on = (ys + b[1] >= -4); ys, xs = ys[on], xs[on]; pk = r.permutation(len(ys))[:130]; L_ = np.c_[xs[pk], ys[pk]].astype(float); self.lpts = L_; self.lrel = 39.5 + 5. * np.clip((L_[:, 1].max() - L_[:, 1]) / (np.ptp(L_[:, 1]) + 1e-6) * .8 + r.uniform(0, .25, len(L_)), 0, 1); vv, uu = np.mgrid[0:a.shape[0], 0:a.shape[1]]; _, q = cKDTree(L_).query(np.c_[uu.ravel(), vv.ravel()]); self.lnear = q.reshape(vv.shape).astype(np.int32); self.lbox = b; self.lprop = np.c_[r.uniform(.8, 1.4, len(L_)), r.uniform(0, 6.28, len(L_)), r.uniform(-1, 1, len(L_)), r.uniform(0, 1, len(L_)), r.uniform(0, 1, len(L_)), r.uniform(0, 6.28, len(L_))]

    def nobara(self, f):
        """ground mask and petals [(world xyz, ...)]"""
        if f < 27: return self.ground('nobara', f), []
        m = self.ground('nobara', min(f, 30)); amp = 7. * min(1., max(0., (f - 27) / 5.))
        vv, uu = np.mgrid[0:TH, 0:TW].astype(f32); n = cv2.resize(fbm(TH // 6, TW // 6, (12, 5), (1, .6), 900 + f), (TW, TH)); n2 = cv2.resize(fbm(TH // 6, TW // 6, (12, 5), (1, .6), 1900 + f), (TW, TH)); m = cv2.remap(m, uu + n * amp, vv + n2 * amp * 1.4, cv2.INTER_LINEAR)
        al = cv2.resize(((self.rel > f)[self.near]).astype(np.uint8) * 255, (TW, TH), interpolation=cv2.INTER_LINEAR).astype(f32) / 255.; m = m * smooth(.35, .65, al); out = []
        def fly(p0, rel, pr, i):
            t = (f - rel) / 14.; s, a0, sw, r1, z, b0 = pr; e = t * t / (t + .22); go = (p0[1] - .05) * (.72 + .55 * z); tx, tz = (r1 - .5) * 1.5 + .12 * sw, .42 + .80 * math.modf(r1 * 7.3 + z)[0]; q = min(1.4, e); x = p0[0] + (tx - p0[0]) * .62 * q + .10 * sw * t + .06 * math.sin(4 * t + b0) * t; y = p0[1] - go * e + .04 * math.cos(3 * t + b0) * t; zz = p0[2] + (tz - p0[2] * .5) * q * (1.15 - .25 * q) + .05 * math.sin(5 * t + a0) * t
            if y > .16: out.append((x, y, max(zz, .003), .036 * s, a0 + (3 + 5 * z) * t * (1 if sw > 0 else -1), b0 + (3.2 + 3 * z) * t, t, i))
        for i in np.nonzero(self.rel <= f)[0]: fly((self.pts[i, 0], self.pts[i, 1], 0.), self.rel[i], self.prop[i], i)
        for i in np.nonzero(self.lrel <= f)[0]:
            px, py = self.lpts[i, 0] + self.lbox[0], self.lpts[i, 1] + self.lbox[1]; d_ = CAM.R.T @ np.array([(px - CAM.cx) / CAM.f, (py - CAM.cy) / CAM.f, 1.]); k = FEET / d_[1]; fly((d_[0] * k, FEET, CZ + d_[2] * k), self.lrel[i], self.lprop[i], 5000 + i)
        out.sort(key=lambda q: -q[1]); return m, out

    def petal_draw(self, img, petals):
        base = np.array([[0, 1.], [-.40, .62], [-.62, .0], [-.42, -.62], [-.12, -.92], [0, -.74], [.12, -.92], [.42, -.62], [.62, .0], [.40, .62]]); shp = spl(base, True, corners=(0, 4, 5, 6), n=3); m_ = len(shp) // 2; lay = [Cel(), Cel(ss=1), Cel(ss=1)]; shc = Cel(ss=1); used = [0, 0, 0]
        for x, y, z, s, rz, rx, t, i in petals:
            (px, py), d = (v[0] for v in CAM.project(np.array([[x, y, z]])))
            if d < .14 or not (-260 < px < W + 260 and -260 < py < H + 260): continue
            sc = s * CAM.f / d; c1, c2, s2 = math.cos(rx), math.cos(rz), math.sin(rz); Q = shp.copy(); Q[:, 0] *= 1 + .12 * math.sin(rx); Q[:, 1] *= max(abs(c1), .16) if t > .06 else .42; Q = np.c_[Q[:, 0] * c2 - Q[:, 1] * s2, Q[:, 0] * s2 + Q[:, 1] * c2] * sc + [px, py]
            lt = math.exp(-((x - .1) / (.6 + .3 * y)) ** 2) * min(1., .35 + y / 7.); k = min(1., t / .30); face = math.sin(rx * .5 + i) > 0; col = (C('#f2637e') * (.30 + .85 * lt) if face else C('#661a4c') * (.45 + .75 * lt)); col = C('#140e26') * (1 - k) + col * k; j = 0 if d > .95 else 1 if d > .50 else 2; c = lay[j]; used[j] = 1
            al_ = 1. if d > .42 else max(0., (d - .14) / .28); c.fill(Q, col, al_)
            if j == 0:
                if sc > 9 and k > .35: c.fill(np.vstack([Q[m_:], Q[:1]]), col * .84); c.line(np.array([Q[0] * .9 + Q[m_] * .1, (Q[0] * .6 + Q[m_] * .4), Q[m_] * .55 + Q[0] * .45]), max(.8, sc * .035), col * .55, taper=(.1, .8), seed=i)
                if sc > 7: c.line(np.vstack([Q, Q[:1]]), max(1., min(2., sc * .05)), '#1d0a1c', taper=(.2, .4), seed=i, alpha=.8 * k)
            else: c.fill(np.vstack([Q[m_:], Q[:1]]), col * .84, al_)
            # lamp is behind them: a lit edge
            if sc > 12 and k > .5: c.line(np.array([Q[m_ // 2], Q[0], Q[m_ + m_ // 2]]), max(1.2, sc * .045), '#ffcf98', taper=(.3, .3), seed=i, alpha=.7 * lt)
        for j, b in ((0, 0), (1, 5.0), (2, 15.)):
            if used[j]: L = lay[j].layer(); img = comp.over(img, cv2.GaussianBlur(L, (0, 0), b) if b else L)
        return img

    # ------------------------------------------------------------ Megumi
    def megumi(self, f):
        """ground mask, pool radius 0..1, ripple age"""
        m = self.ground('megumi', min(f, 54))
        if f < 54: return m, 0., None
        k = f - 54; s = 1 - min(1., max(0., (k - 1.) / 7.5)) ** 1.5
        if s < .03: m = np.zeros_like(m)
        else:
            vv, uu = np.mgrid[0:TH, 0:TW].astype(f32); cx, cy = gp(MC)[0]; dx, dy = uu - cx, vv - cy; tw = (1 - s) * .55; c_, sn = math.cos(tw), math.sin(tw); m = cv2.remap(m, (cx + (dx * c_ - dy * sn * .25) / (s * .72 + .28)).astype(f32), (cy + (dy * c_ + dx * sn * .3) / max(s, .04)).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        g = 1 - (1 - min(1., (k + .6) / 5.)) ** 2.4; q = min(1., max(0., (k - 8.) / 4.)); back = q * q * (3 - 2 * q); return m, g * (1 - back), (k - 11.5 if k >= 11.5 else None)

    def pool(self, q, k):
        """a still black pool lying on the road, its rim breathing"""
        th = np.linspace(0, 2 * math.pi, 120, endpoint=False); w = 1 + .050 * np.sin(2 * th + 1. + k * .25) + .038 * np.sin(3 * th + 2. - k * .3) + .022 * np.sin(5 * th + .4 + k * .4); return np.c_[MC[0] + np.cos(th) * w * .50 * q, MC[1] + np.sin(th) * w * .60 * q]

    def ring_(self, img, R, w, amp, fade, wav=.034):
        """one travelling swell on the wet film: bends what lies under it, catches the lamp on its far side"""
        d = np.hypot(self.X - MC[0], self.Y - MC[1]); e = np.exp(-((d - R) / w) ** 2); e *= (d > 0) * self.ok
        if fade <= 0: return img
        cx, cy = sp(MC)[0]; x, y = grid(); ux, uy = x - cx, y - cy; L = np.hypot(ux, uy) + 1e-3; ph = np.sin((d - R) / wav); img = cv2.remap(img.astype(f32), x + (ux / L * ph * amp * fade * e).astype(f32), y + (uy / L * ph * amp * .6 * fade * e).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        sl = np.cos((d - R) / wav) * e * fade * np.clip((self.Y - MC[1]) / .12, -1, 1); lm = np.clip(self.lampl.mean(-1), 0, .9); return img * (1 - .34 * np.clip(-sl, 0, 1))[..., None] + C('#ffdba4') * (np.clip(sl, 0, 1) * (.06 + .60 * lm))[..., None]

    def stain_draw(self, img, q, rip, f, refl):
        cx, cy = sp(MC)[0]; k = f - 54
        if rip is not None:
            for j, d0 in enumerate((0., 2.5)):
                t = (rip - d0) / 8.
                if 0 <= t < 1.2: img = self.ring_(img, .08 + 1.25 * (1 - (1 - min(t, 1.)) ** 1.6), .13, 8. - 3 * j, max(0., 1 - t) * (1 - .35 * j))
        if q > .02:
            Q = self.pool(q, k); P = sp(Q); c = Cel(); c.fill(P, '#ffffff'); a = c.rgba()[..., 3]; soak = np.clip(cv2.GaussianBlur(a, (0, 0), 13) - a, 0, 1); img = img * (1 - .40 * soak[..., None]); x, y = grid()
            # a mirror: black, his legs upside-down in it, the lamp's column, rings round his ankles
            col = np.zeros((H, W, 3), f32) + C('#04030a'); col = col * (1 - refl[..., 3:4] * .8) + refl[..., :3] * .8; d = np.hypot((self.X - MC[0]) / .50, (self.Y - MC[1]) / .60); col += C('#7a4a24') * (np.exp(-((x - cx - 60) / 150) ** 2) * smooth(-.2, 1., (self.Y - MC[1]) / .6) * (.8 + .2 * np.sin(y * .5 + 2 * np.sin(x * .013 + k * .6))) * .34)[..., None]
            for sx in (-.085, .085):
                da = np.hypot((self.X - WHO['megumi'] - sx) / 1., (self.Y - FEET + .03) / 1.6)
                for j in range(2): r0 = ((k * .16 + j * .5) % 1.); e = np.exp(-((da - .05 - r0 * .20) / .010) ** 2) * (1 - r0) * min(1., k / 3.); col += C('#ffcf90') * (e * .34)[..., None]
            img = img * (1 - a[..., None]) + col * a[..., None]; b = cv2.GaussianBlur(a, (0, 0), 1.6); img = img + C('#ffd8a0') * (np.clip(b - kit.shifted(b, 0, 2.2), 0, 1) * .75)[..., None] + C('#8a6ad0') * (np.clip(b - kit.shifted(b, 0, -2.0), 0, 1) * .35)[..., None]
        k = f - 67
        if 0 <= k <= 4:
            c = Cel(); rr = (9, 16, 22, 23, 30)[k]; a = np.linspace(0, 2 * math.pi, 40)
            if k < 3: c.fill(np.c_[cx + np.cos(a) * rr, cy - rr * .30 + np.sin(a) * rr * .72], '#0a0814', .7); c.line(np.c_[cx + np.cos(a[22:38]) * rr, cy - rr * .30 + np.sin(a[22:38]) * rr * .72], 2., '#ffdca8', taper=(.4, .4), seed=2); c.line(np.c_[cx + np.cos(a[3:17]) * rr, cy - rr * .30 + np.sin(a[3:17]) * rr * .72], 1.8, '#7a60b8', taper=(.4, .4), seed=1)
            else:
                for j in range(9): th = math.pi + j * .39; d0, d1 = rr * .8, rr * (1.1 + .25 * (j % 3)); c.line(np.array([(cx + math.cos(th) * d0, cy + math.sin(th) * d0 * .6), (cx + math.cos(th) * (d0 + d1) / 2, cy + math.sin(th) * (d0 + d1) * .34 - 4), (cx + math.cos(th) * d1, cy + math.sin(th) * d1 * .6)]), 1.8, '#ffe0b4', taper=(.2, .7), seed=j, alpha=.85)
            img = comp.over(img, c.layer())
        return img

    # ------------------------------------------------------------ rain
    def rain(self, img, f):
        if f < 44: return img
        cel = Cel(); near = Cel(ss=1); r = rnd('streak', f); n = int(3 + max(0, f - 45) * 4.2)
        for j in range(n):
            x = r.uniform(-60, W + 60); y = r.uniform(-120, H); z = r.random(); ln = (40 + 120 * z); wd = .8 + 1.5 * z; lt = math.exp(-((x - 980) / 560) ** 2); cel.line(np.array([(x, y), (x - ln * .06, y + ln * .5), (x - ln * .12, y + ln)]), wd, C('#ffe2b4') * (.35 + .65 * lt) + C('#8a80c0') * .25 * (1 - lt), taper=(.6, .25), seed=j, alpha=.20 + .34 * z)
        for j in range(max(0, (f - 52) // 4)): x = r.uniform(0, W); y = r.uniform(-200, H); near.line(np.array([(x, y), (x - 20, y + 170), (x - 40, y + 340)]), 7., '#c8bce8', taper=(.4, .3), seed=j, alpha=.16)
        for t, X, Y, rr, i in self.drops:
            k = f - t
            if k < 0 or k > 12: continue
            p = sp([(X, Y)])[0]; sc = CAM.f / Y; asp = CZ / Y / math.sqrt(1 + (CZ / Y) ** 2) * 1.1; q = rnd('d', i); a = np.linspace(0, 2 * math.pi, 36); lt = .25 + .75 * float(np.clip(self.lampl[int(np.clip(p[1], 0, H - 1)), int(np.clip(p[0], 0, W - 1))].mean(), 0, 1))
            if k < 10:                                                                   # ring spreading on the wet film
                R = rr * sc * (.8 + .55 * k); ring = np.c_[p[0] + np.cos(a) * R, p[1] + np.sin(a) * R * asp]; cel.line(ring[19:35], max(.8, 2.0 * (1 - k / 10.)), '#ffdfae', taper=(.4, .4), seed=i, alpha=.8 * lt * (1 - k / 10.)); cel.line(ring[2:16], max(.8, 1.8 * (1 - k / 10.)), '#120c1e', taper=(.4, .4), seed=i + 1, alpha=.55 * (1 - k / 10.))
            if k < 4:                                                                    # crown
                u = k / 4.; h = rr * sc * 1.6 * math.sin(math.pi * (.15 + .8 * u))
                for j in range(5): th = math.pi * (1.0 + j / 4.) + q.uniform(-.2, .2); b = np.array([p[0] + math.cos(th) * rr * sc * .8, p[1] + math.sin(th) * rr * sc * .5 * asp]); e = b + [math.cos(th) * h * .5, -h * q.uniform(.5, 1.)]; cel.line(np.array([b, (b + e) / 2 + [0, -h * .08], e]), max(1., rr * sc * .10), '#fff0d0', taper=(.1, .7), seed=j, alpha=.85 * lt * (1 - u * .6))
        img = comp.over(img, cv2.GaussianBlur(near.layer(), (0, 0), 3.)); return comp.over(img, cel.layer())

    # ------------------------------------------------------------ frame
    def leg_layer(self, who, f, cells=False):
        """premultiplied drawing and its box: dark against the glow, warm at the edges, shoes keep their colour"""
        L_ = self.legs[who]; a, b = L_[20] if cells else L_[min(int(f) // 2, len(L_) - 1)]; q = a.astype(f32) / 255.; y_ = self.state(who, min(f, 24))['y']; ank = CAM.project(np.array([[0, y_, .135], [0, y_, .075]]))[0][:, 1] - b[1]; h = np.arange(q.shape[0], dtype=f32)[:, None]; shoe = smooth(ank[0], ank[1], h); lum = q[..., :3].max(-1); keep = smooth(.34, .62, lum); rgb_ = q[..., :3]
        g = .40 + .60 * keep; leg = rgb_ * g[..., None]; tint = dict(yuji=np.array([1.75, .86, .80], f32), nobara=np.array([1.35, .88, .62], f32), megumi=np.array([.52, .46, .50], f32))[who]; sh = rgb_ * tint; o = leg * (1 - shoe[..., None]) + sh * shoe[..., None]; q = np.dstack([np.minimum(o, q[..., 3:4] * 1.1), q[..., 3]])
        if cells: q = q * smooth(.35, .65, cv2.GaussianBlur(((self.lrel > f)[self.lnear]).astype(f32), (0, 0), 1.))[..., None]
        return q, b

    @staticmethod
    def put(dst, q, x0, y0, cut=None):
        h, w = q.shape[:2]; xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(cut if cut is not None else H, y0 + h)
        if xb <= xa or yb <= ya: return
        s = q[ya - y0:yb - y0, xa - x0:xb - x0]; d = dst[ya:yb, xa:xb]
        if dst.shape[2] == 4: dst[ya:yb, xa:xb] = d * (1 - s[..., 3:4]) + s
        else: dst[ya:yb, xa:xb] = d * (1 - s[..., 3:4]) + s[..., :3]

    def frame(self, f):
        my = self.ground('yuji', f); mn, petals = self.nobara(f); mm, pq, rip = self.megumi(f); M = np.maximum(my, np.maximum(mn, mm)); v = np.linspace(GY, GY + GH, TH, dtype=f32)[:, None]; u = np.clip((FEET - v) / 2.9, 0, 1)           # penumbra opens away from the shoes
        M = cv2.GaussianBlur(M, (0, 0), .8) * np.clip(1 - 2 * u, 0, 1) + cv2.GaussianBlur(M, (0, 0), 1.5) * (1 - np.abs(2 * u - 1)) + cv2.GaussianBlur(M, (0, 0), 2.5) * np.clip(2 * u - 1, 0, 1)
        S = cv2.warpPerspective(M.astype(f32), HGS, (W, H), flags=cv2.INTER_LINEAR) * self.ok; fl = 1 + .010 * math.sin(f * 1.9) - (.06 if f == 40 else .03 if f == 41 else 0)
        img = self.amb * np.array([1, 1, 1], f32) + self.lampl * (fl * (1 - .985 * S))[..., None]; img = img * (1 - .30 * S[..., None] * (1 - self.pud * .5)[..., None])
        edge = np.clip(cv2.GaussianBlur(S, (0, 0), 6) - S, 0, 1); img = img + C('#7a3f7c') * (edge * np.clip(self.lampl.mean(-1), 0, .7) * .16)[..., None]
        # who is still here
        L_ = []; fy = int(round(sp([(0, FEET)])[0][1]))
        if f < 62: k = max(0., f - 57.); dy = int(6 * k + 7 * k * k); L_.append(('megumi', self.leg_layer('megumi', f), dy, fy + 10 if k > 0 else None))
        if f < 46: L_.append(('nobara', self.leg_layer('nobara', f, f >= 39), 0, None))
        L_.append(('yuji', self.leg_layer('yuji', f), 0, None))
        # the wet road gives them back, upside down
        refl = np.zeros((H, W, 4), f32)
        for who, (q, b), dy, cut in L_:
            gy = int(round(sp([(0, self.state(who, min(f, 24))['y'])])[0][1])) + 4; flip = q[::-1]; x0, y0 = int(b[0]), int(b[1]) + dy; self.put(refl, flip, x0, 2 * gy - (y0 + q.shape[0]))
        yy = np.arange(H, dtype=f32)[:, None]; fade = np.exp(-np.maximum(yy - 380, 0) / 230.); rb = cv2.GaussianBlur(refl, (0, 0), 3.5, sigmaY=13.); k_ = (.42 * fade * (1 - self.pud))[..., None]; img = img * (1 - rb[..., 3:4] * k_) + rb[..., :3] * k_ * .7; rp = cv2.GaussianBlur(refl, (0, 0), .9, sigmaY=2.); k_ = (self.pud * .85)[..., None]; img = img * (1 - rp[..., 3:4] * k_) + rp[..., :3] * k_ * .85
        if pq > .02 or rip is not None or 67 <= f <= 71: img = self.stain_draw(img, pq, rip, f, rp)
        # the lamp: its post, its breath of mist behind the legs
        x, y = grid(); t = f / 24.; bp = CAM.project(np.array([[1.75, 10.4, 0.], [1.75, 10.4, 6.]]))[0]; c = Cel(); c.fill(np.array([bp[0] + [-9, 2], bp[0] + [9, 2], bp[1] + [7, 0], bp[1] + [-7, 0]]), '#0d0a18'); c.fill(np.array([bp[0] + [-9, 2], bp[0] + [-6, 2], bp[1] + [-4, 0], bp[1] + [-7, 0]]), '#c8884a'); img = comp.over(img, c.layer())
        if not hasattr(self, 'mist'): self.mist = cv2.resize(fbm(90, 160, (34, 14), (1, .6), 55), (W + 240, H))
        n = self.mist[:, int(40 + 34 * t):int(40 + 34 * t) + W]; gl = np.exp(-((x - 985) / 560) ** 2) * np.exp(-((y - 60) / 190) ** 2); img = comp.screen(img, (gl * (.46 + .12 * n))[..., None] * C('#ffb866')); img = img + (np.exp(-((x - 985) / 260) ** 2) * np.exp(-((y - 50) / 80) ** 2) * .50)[..., None] * C('#ffd9a0')
        for who, (q, b), dy, cut in L_: self.put(img, q, int(b[0]), int(b[1]) + dy, cut)
        if petals: img = self.petal_draw(img, petals)
        img = self.rain(img, f)
        z = 1 + .040 * f / 72.; return comp.cam2d(img.astype(f32), W / 2 + 3 * math.sin(f * .05), H / 2 - 8 * f / 72., z, .003 * math.sin(f * .04), border=cv2.BORDER_REFLECT_101).astype(f32)
