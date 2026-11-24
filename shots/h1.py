"""h1 ASH (72 f).  White becomes falling ash.  0-29 kneeling, camera on the ground, tilted; 18 his hand takes the knee; 30 close, 36 his eyes open;
44-51 he rises in four keys (hand pushes the knee, back, shoulders, head last); 52 low lens, the red shoe over us; 54 it comes down and the ground breaks."""
import numpy as np, math, cv2, os
from jj import comp, W, H, world, fx
from jj.cam import Camera3D
from jj.ink import Cel, spl
from jj.kit import shake, impact, outline
from jj.paint import rgb, fbm, smooth, ramp, grid, f32
from jj.fig_pose import P, mix_pose
from chars import yuji as Y
from chars.yuji import Yuji, YHead, mixc, rad
A = np.array
PW, PH = 2400, 1760
EYE, TGT, FOC = (-.95, -3.25, .30), (-.20, 2.0, .98), 1350.
ROT = .31                                            # dutch tilt
REF = dict(a=(1.40, .56, (640., 655.)), b=(1.06, .84, (690., 560.)))          # framings of the set: zoom, height on him (m), where that point sits
EDGE = dict(out='#04050c', body='#0a1136', hot='#1a3cc8', core='#8fd2ff', ink='#020208', glow='#1f49e0')          # blue that has gone black at its edges
DUSTP = dict(fx.DUST); DUSTP.update(lit='#8f837d', mid='#6a5f5e', dark='#3d373d', ink='#171318')

# ---------------------------------------------------------------- acting
K0 = P('kneel', cst=(24, 0, 2), nck=(30, 0, 0), hed=(22, 0, 0), hR=(.06, .34, -.88), hL=(.14, .30, -.72))
K1 = P('kneel', cst=(22, 0, 2), nck=(28, 0, 0), hed=(20, 0, 0), hR=(.06, .34, -.88), hL=(.06, .62, -.22), gL='hold')
K2 = P(K1, root=(0, .0, -.252), pel=(20, 0, -4), wst=(22, 0, 0), cst=(22, 0, 4), nck=(28, 0, 0), hed=(22, 0, 0))
# 1 the hand pushes the knee, hips leave the heel      2 the back unrolls      3 the shoulders open      4 the head, last
R1 = P(K2, root=(0, .06, -.150), pel=(40, 0, -6), wst=(20, 0, 2), cst=(18, 0, 6), nck=(26, 0, 0), hed=(24, 0, 0), cL=(4, 2), hL=(.05, .66, -.34), hR=(.30, .0, -.92), gR='fist', fL=(.085, .18, .043), fR=(.075, -.215, .085), tR=(4, -58, 44), kR=(.14, 1, -.2), kL=(.3, 1, 0))
R2 = P(R1, root=(0, .03, -.045), pel=(18, 0, -4), wst=(18, 0, 0), cst=(22, 0, 2), nck=(32, 0, 0), hed=(24, 0, 0), cL=(2, 4), cR=(2, 4), hL=(.12, .36, -.86), hR=(.16, .10, -.96), fL=(.08, .15, .043), fR=(.075, -.16, .062), tR=(8, -30, 24), kR=(.12, 1, 0))
R3 = P(R2, root=(0, -.004, -.008), pel=(0, 0, 0), wst=(0, 0, 0), cst=(-3, 0, 0), nck=(34, 0, 0), hed=(22, 0, 0), cL=(-2, -4), cR=(-2, -5), hL=(.26, -.10, -.95), hR=(.28, -.14, -.94), gL='fist', gR='fist', fL=(.09, .10, .043), fR=(.085, -.10, .048), tR=(14, -6, 0))
R4 = P(R3, root=(0, -.004, -.004), pel=(2, 0, 0), cst=(-2, 0, 0), nck=(10, 0, 0), hed=(10, 0, 0))
S0 = P(R3, root=(.012, -.02, -.030), pel=(-8, 0, 8), wst=(2, 0, 0), cst=(0, 0, -8), nck=(8, 0, 2), hed=(6, 0, 2), hL=(.30, .30, -.86), hR=(.34, -.34, -.82), fR=(.075, .40, .235), tR=(6, 36, 0), kR=(.12, 1, .4), gL='fist', gR='fist')
# the stride at the lens, one drawing a frame: knee high, the leg reaching, the sole coming down
SA = P(R3, root=(.010, -.01, -.016), pel=(-4, 0, 10), wst=(4, 0, 0), cst=(2, 0, -10), nck=(8, 0, 2), hed=(8, 0, 2), hL=(.30, .42, -.74), hR=(.34, -.40, -.78), fR=(.080, .19, .245), tR=(6, 4, 0), kR=(.06, 1, .55), gL='fist', gR='fist')
SB = P(SA, root=(.012, .02, -.034), pel=(2, 0, 10), fR=(.090, .335, .165), tR=(6, 22, 0), hL=(.32, .34, -.80), hR=(.36, -.36, -.78))
SC = P(SB, root=(.008, .06, -.080), pel=(12, 0, 4), cst=(4, 0, -6), fR=(.100, .345, .085), tR=(8, 10, 0), hL=(.42, .12, -.80), hR=(.44, -.16, -.76), fL=(.095, -.10, .052), tL=(18, -16, 12))
S1 = P(S0, root=(0, .10, -.150), pel=(24, 0, -10), wst=(10, 0, -2), cst=(8, 0, -2), nck=(6, 0, 6), hed=(8, 0, 4), hL=(.56, -.10, -.66), hR=(.50, .10, -.72), gL='claw', gR='fist', fR=(.10, .32, .043), fL=(.10, -.18, .07), tL=(24, -36, 26), tR=(8, 0, 0), kR=(.30, 1, 0), kL=(.35, .8, 0))


def pose_at(f):
    d = f // 2 * 2
    if f < 18: return mix_pose(K0, P(K0, cst=(20, 0, 2), root=(0, -.04, -.262)), .5 + .5 * math.sin(d / 24 * 2 * math.pi * .6))
    if f < 24: return mix_pose(K0, K1, [.35, .80, 1.06][(d - 18) // 2])
    if f < 44: return mix_pose(K1, K2, min(1., (d - 24) / 5.))
    if f < 46: return P(K2, root=(0, .02, -.262), cst=(28, 0, 6))                                      # the weight goes down first
    if f < 51: return [mix_pose(R1, K2, .18), R1, mix_pose(R1, R2, .85), R3, R4][f - 46]
    if f < 54: return [SA, SB, SC][f - 51]
    return [S1, P(S1, root=(0, .10, -.172)), P(S1, root=(0, .10, -.164))][min(2, f - 54)] if f < 58 else P(S1, root=(0, .10, -.140))


# low lens of the step
LE, LT, LF = (.03, -1.04, .115), (-.02, .12, .865), 545.
FT = (-.175, -.555)                                  # where the red shoe lands (metres)


class Shot:
    id = 'h1'; n = 72
    finish = dict(bloom=.34, vig=.24)
    EVENTS = [(0, 'ash_air', dict(note='white becomes falling ash, 0-14')), (18, 'cloth_knee', dict(size=.5, note='his hand takes the knee')), (36, 'eyes_open', dict(note='close: eyes open, iris burning in the sockets')), (38, 'cursed_energy', dict(size=.5, note='blue licks, black edges, begin')),
              (46, 'stand_up', dict(note='hand pushes the knee 44-46, back 48, shoulders 50, head 51')), (51, 'foot_lift', dict(size=.4, note='cut to the low lens: knee high, red shoe over us')), (54, 'step_break', dict(size=1.0, note='the shoe lands at the lens; impact frame 54; slabs tilt up 55-59')), (54, 'cursed_energy', dict(size=1.0, note='black-blue energy explodes backward, holds to 71')), (56, 'shock_ring', dict(size=.8))]

    def setup(self):
        kp, km = 'cache/h1_v7_ref.npy', 'cache/h1_v7_meta.npy'
        if not (os.path.exists(kp) and os.path.exists(km)):
            Pl = world.plate('ruin', mode='ash', view='low', eye=EYE, target=TGT, focal=FOC, w=PW, h=PH, figure=(0., 0.), cr=(28., 75.)); g0 = Pl.extra['ground_z']; pcam = Pl.cam
            im = Pl.img.astype(f32); nr = smooth(9., 2.6, Pl.z) * (Pl.z < 500); fire = smooth(.22, .5, im[..., 0] - im[..., 2]) * smooth(.45, .75, im[..., 0]); fire = cv2.GaussianBlur(cv2.dilate(fire.astype(f32), np.ones((13, 13), np.uint8)), (0, 0), 8); im = im * (1 - .80 * fire[..., None])       # the painted fires go out: drawn ones burn in their place
            far = smooth(8., 60., np.minimum(Pl.z, 200.)); im = cv2.GaussianBlur(im, (0, 0), 1.1) * far[..., None] + im * (1 - far[..., None]); im = im * (1 - .42 * nr[..., None]); del Pl
            meta = [g0]; refs = []; x, y = grid(); c_, s_ = math.cos(ROT), math.sin(ROT)
            for k_ in ('a', 'b'):
                z_, hgt, dst = REF[k_]; pf = pcam.project(A([[0., 0., g0 + hgt]]))[0][0]; dv = (A(dst) - A([960., 540.])) / z_; cc = pf - A([c_ * dv[0] - s_ * dv[1], s_ * dv[0] + c_ * dv[1]]); ref = comp.cam2d(im, cc[0], cc[1], z_, rot=ROT); meta += [cc[0], cc[1]]
                refs.append(self.grade(ref, x, y).astype(np.float16))
            del im, nr, fire, far; np.save(kp, np.stack(refs)); np.save(km, A(meta))
        self.refs = np.load(kp, mmap_mode='r'); mt = [float(v) for v in np.load(km)]; self.gz = mt[0]; self._c = {}; self.ref = self.refs[0]; self.cams = {}
        c_, s_ = math.cos(ROT), math.sin(ROT)
        for j, k_ in enumerate(('a', 'b')):
            z_ = REF[k_][0]; q = (A([PW / 2., PH / 2.]) - A(mt[1 + 2 * j:3 + 2 * j])) * z_; self.cams[k_] = Camera3D(EYE, TGT, FOC * z_, W, H, cx=960 + c_ * q[0] + s_ * q[1], cy=540 - s_ * q[0] + c_ * q[1], roll=-ROT)
        self.low = Camera3D(LE, LT, LF, W, H, roll=.16)
        b = cv2.GaussianBlur(np.asarray(self.ref, f32), (0, 0), 26); self.cubg = (b * .72).astype(np.float16)
        self.n1 = fbm(H, W, (110, 40, 12), (1, .5, .3), 11).astype(np.float16); q = fbm(H, W // 2 + 200, (320, 120, 46, 18), (1, .6, .4, .25), 21, aspect=1.6).astype(np.float16); self.sm = np.hstack([q, q[:, ::-1]])          # periodic, no seam
        # out-of-focus rubble at the lens
        c = Cel(ss=1); r = np.random.default_rng(6); pal = dict(lit='#4a4244', mid='#26222a', dark='#100e14', ink='#08070a')
        for x, y, s in ((60, 1060, 250), (420, 1120, 150), (1180, 1130, 170), (1560, 1085, 260), (1860, 1010, 300), (880, 1140, 110)): fx.rock(c, x, y, s, tuple(r.uniform(-3, 3, 3)), int(x), pal, light=(.7, -.6, .2), lw=4, crack=False)
        L = c.rgba(); L = Y.rim_in(L, -7, 5, '#ff9a50', .9, 0, 1.5, over=True); self.fgr = cv2.GaussianBlur(L, (0, 0), 11).astype(np.float16)

    def grade(self, ref, x, y):
        ref = ref * (.50 + .50 * smooth(1500, 300, np.hypot(x - 1330, (y - 470) * 1.2)))[..., None] * rgb('#e2dcdc'); return ref + comp.radial(1500, 470, 900, 640, power=1.8)[..., None] * rgb('#ff7a36') * .55 + comp.radial(1180, 120, 900, 500, power=2.)[..., None] * rgb('#d9cfc6') * .22

    # ---------------------------------------------------------------- the figure
    def head_fn(self, expr, f, side=1):
        L = Yuji.light('ash', side=side, amt=.10)
        def fn(H_, info):
            x0, y0, x1, y1 = info['roi']; c = Cel(x1 - x0, y1 - y0, x0, y0); Yuji.head(c, H_, expr, L, f // 2 * 2, (.20, .14 + .55 * (f >= 51)), 4, damage=.6, lod=1 if H_.s < 120 else 2); return c.rgba()
        return fn

    def fig(self, f, low=False):
        rk = 'a' if f < 44 else 'b'; d = f if 44 <= f < 60 else f // 2 * 2; key = (low, rk, d if d < 60 else 60 + (d // 2 % 2) * 2)
        if key not in self._c:
            from jj.figure import Fig
            F = Fig('yuji', seed=3).place(0., 0., 0. if low else self.gz, heading=math.pi * (1.0 if low else 1.20 - .05 * float(smooth(45, 50, f)))).anim(lambda q: dict(pose=pose_at(int(max(0, q))))); F.at(d); F.blow(-.5, .6 if low else .2, .6 if f >= 51 else .1)
            e = 'shut' if f < 36 else Yuji.expr('rage', gy=-.01)
            L = F.draw(self.low if low else self.cams[rk], d, light=(.15, .95, -.25) if low else (.75, .55, -.10), pal=dict(mul=(.80, .77, .80)), kick=(-.6, .7, -.4) if low else (.85, .35, -.55), key=.22, seed=3, head_fn=self.head_fn(e, f)).astype(f32)
            lum_ = L[..., :3].max(-1) / np.maximum(L[..., 3], 1e-3); cloth = smooth(.42, .22, lum_) * L[..., 3]; n = self.n1.astype(f32)
            burn = smooth(.7, 1.0, n) * cloth; L[..., :3] *= (1 - .55 * burn)[..., None]
            if low: L = Y.rim_in(L, 4, 8, '#5fa8ff', .95, 2.2); L = Y.para_fast(L, (960, 1080), (960, 60), [(0, '#ffffff'), (.5, '#c9c6d4'), (1, '#7d86b4')])
            else: L = Y.rim_in(L, -8, 1, '#ffb060', 1., 2.2); L = Y.rim_in(L, 2, 4, '#dfe6f0', .35, 2.0); L = Y.para_fast(L, (900, 500), (420, 760), [(0, '#ffe9d4'), (.45, '#d9d0d2'), (1, '#8b8494')])
            hd = F.info['head']
            if len(self._c) > 5: self._c.pop(next(iter(self._c)))
            self._c[key] = (L.astype(np.float16), dict(x=hd['x'], y=hd['y'], s=hd['s']))
        L, info = self._c[key]; return L.astype(f32), info

    def smoke(self, im, f, y0, y1, amt, col, sp, seed=0, hard=.35):
        n = np.roll(self.sm[::-1] if seed % 2 else self.sm, int(-f * sp) + seed * 211, 1)[:, :W].astype(f32); yy = grid()[1]; m = smooth(hard - .5, hard + .9, n) * smooth(y0, (y0 + y1) / 2, yy) * smooth(y1 + 260, y1, yy) * amt
        return im * (1 - m[..., None]) + rgb(col) * m[..., None]

    def fires(self, im, f):
        c = Cel(ss=1); k = f // 2
        for (x0, y0, x1, y1), h, sd in (((1400, 560, 1640, 486), 250, 3), ((1080, 604, 1180, 574), 110, 8), ((1780, 470, 1900, 436), 150, 5)) if f < 44 else (((1350, 566, 1540, 508), 205, 3), ((1100, 600, 1180, 576), 90, 8), ((1680, 476, 1790, 444), 130, 5)): fx.flames(c, A([(x0, y0), ((x0 + x1) / 2, (y0 + y1) / 2 + 4), (x1, y1)]), h, t=k * 2 / 24., seed=sd, lean=.18, rows=2, lw=1.5)
        L = cv2.GaussianBlur(c.rgba(), (0, 0), 3.2); im = im * (1 - L[..., 3:4] * .96) + L[..., :3] * .96; g = cv2.GaussianBlur(cv2.resize(L[..., :3], (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 16); return im + cv2.resize(g, (W, H)) * .75

    def energy(self, im, L, f, en, st, only_blue=False):
        fr = None; sz = 40 + 95 * en
        for pal, sc_, sd in ((EDGE, 1.25, 7), (fx.CURSE, .70, 3))[1 if only_blue else 0:]:
            bk, fr = fx.cursed(L, f, stream=st, size=sz * sc_, pal=pal, seed=sd, amount=min(1., .40 + .6 * en), wisps=.6 + .6 * en, light=.5 + .4 * en); im = im * (1 - bk.rgba[..., 3:4]) + bk.rgba[..., :3]; im = im + bk.light * (.5 if pal is EDGE else .75)
        return im, fr

    # ---------------------------------------------------------------- 0-29 and 44-51
    def stage(self, f):
        rk = 'a' if f < 44 else 'b'; im = np.asarray(self.refs[0 if rk == 'a' else 1], f32).copy(); L, info = self.fig(f); cam = self.cams[rk]
        im = self.fires(im, f); im = self.smoke(im, f, 40, 620, .55, '#5b5352', 5.0, 0, .30); im = self.smoke(im, f, 330, 800, .40, '#8a6a58', 8.0, 3, .55)
        g = lambda p: cam.project(A([p], float))[0][0]; fc = g((0., 0., self.gz)); x, y = grid(); ca, sa = math.cos(ROT), math.sin(ROT); u = (x - fc[0]) * ca - (y - fc[1]) * sa; v = (x - fc[0]) * sa + (y - fc[1]) * ca
        im = im * (1 - .62 * np.clip(1 - np.hypot((u + 150) / 520, (v - 16) / 70), 0, 1) ** .8)[..., None]                       # long shadow thrown toward us by the fire
        en = float(np.clip((f - 25) / 12., 0, 1)) * .22 if f < 44 else .26 + .34 * smooth(44, 51, f); fr = None
        if en > .05: im, fr = self.energy(im, L, f, en, (-.35, -1.))
        im = comp.over(im, L)
        if fr is not None and f >= 44: im = im * (1 - fr.rgba[..., 3:4] * .7) + fr.rgba[..., :3] * .7
        im = self.smoke(im, f, 640, 1010, .50, '#6f5f5a', 12.0, 5, .40)
        G = cv2.warpAffine(self.fgr.astype(f32), np.float32([[1, 0, -f * .7], [0, 1, 0]]), (W, H), borderMode=cv2.BORDER_REPLICATE); im = comp.over(im, G)
        return im, info

    # ---------------------------------------------------------------- 30-43
    def close(self, f):
        im = self.cubg.astype(f32); d = f // 2 * 2; im = self.smoke(im, f, 0, 900, .45, '#4a4243', 7., 2, .4)
        if f < 36: e = Yuji.expr('shut', furrow=.6, dark=.32, B=[(.112, .060), (.256, .112), (.404, .160), (.546, .168)], mU=[-.002, .001, .002, .001, -.002], mL=[-.002, .001, .002, .001, -.002], hw=.16, lip=1.)
        elif f == 36: e = Yuji.expr('rage', U=[-.054, -.030, -.016, -.010, -.004], Lo=[-.066, -.084, -.090, -.080, -.030])
        elif f == 37: e = Yuji.expr('rage', U=[-.050, .014, .052, .062, .022], Lo=[-.068, -.094, -.104, -.094, -.034])
        else: e = Yuji.expr('rage')
        br = math.sin(d / 24. * 2 * math.pi * .7) * 1.6; lift = comp.ease(np.clip((f - 35) / 7., 0, 1))
        hd = YHead().pose(880, 575 + br - 18 * lift + (6 if f == 36 else 0), 480, .30, -.18 + .12 * lift, -.10 + .02 * lift); L = Yuji.light('ash', side=1, amt=.46, rim=[dict(d=(-9, 0), col='#ffb060', amt=1.), dict(d=(6, 4), col='#cfd8e6', amt=.4)])
        bd = Yuji.bust(hd, e, L, f=d, wind=(-.22, .10 + .22 * smooth(36, 44, f)), seed=6, byaw=.20, bpitch=-.04, broll=-.06, damage=1., raise_=.06, breath=br)
        x = grid()[0]; bd[..., :3] *= (.62 + .50 * smooth(420, 1350, x))[..., None]
        if f >= 37:
            en = float(smooth(36, 44, f)); im, _ = self.energy(im, bd, f, .15 + .75 * en, (-.35, -1.1))
        im = comp.over(im, bd)
        inner = np.clip(comp.blur(1 - bd[..., 3], 22) * bd[..., 3], 0, 1); im = comp.screen(im, inner[..., None] * (rgb('#ff8a48') * comp.radial(1700, 560, 1100, power=1.)[..., None] * .55 + rgb('#4f8fff') * (.35 * float(smooth(36, 43, f)))))
        if 36 <= f < 40:
            for q in (-.334, .334): e_ = hd.P([(q, 0)])[0]; im = im + comp.radial(e_[0], e_[1], 190, 105, power=2.)[..., None] * rgb('#ffa23c') * (.45, .55, .28, .12)[f - 36]
        return im

    # ---------------------------------------------------------------- 52-71 the step
    def slabs(self, t, near):
        """road plates tearing up round the foot: 3-D, hinged on the cracked ring, sorted by depth"""
        c = Cel(); ft = A(FT); r = np.random.default_rng(4); it = []
        for i in range(13):
            th = i / 13. * 2 * math.pi + r.normal(0, .12); d = A([math.cos(th), math.sin(th)]); n = A([-d[1], d[0]]); side_ = abs(math.sin(th - (-.5 * math.pi) if False else 0.)); fr_ = math.cos(th + math.pi / 2); k_ = .42 if fr_ > .55 else .75 if fr_ > 0 else 1.15; r0, r1 = r.uniform(.15, .22), .18 + k_ * r.uniform(.22, .52); w0, w1 = r.uniform(.05, .09), k_ * r.uniform(.12, .22); ti = float(np.clip(t * 1.25 - r.uniform(0, .25), 0, 1)); e = 1 - (1 - ti) ** 3; tilt = math.radians(r.uniform(26, 58)) * (e + .16 * math.sin(math.pi * min(1, ti * 1.0)) * (1 - ti)); h = .055 * (.6 + .5 * k_)
            if fr_ > .62: continue
            it.append((ft + d * (r0 + r1) / 2, d, n, r0, r1, w0, w1, tilt, h, i))
        it.sort(key=lambda q: -np.hypot(*(q[0] - A(LE[:2]))))
        for cen, d, n, r0, r1, w0, w1, tilt, h, i in it:
            isnear = cen[1] < ft[1] - .06
            if isnear != near: continue
            def V(rr, ww, up): rd = r0 + (rr - r0) * math.cos(tilt) - up * math.sin(tilt); z = (rr - r0) * math.sin(tilt) + up * math.cos(tilt) - .03; p = ft + d * rd + n * ww; return (p[0], p[1], z)
            q = np.random.default_rng(40 + i); prof = [(r0, -w0), (r0 + (r1 - r0) * .5, -w1 * q.uniform(.85, 1.1)), (r1 * q.uniform(.94, 1.06), -w1 * q.uniform(.5, .9)), (r1 * q.uniform(1., 1.1), w1 * q.uniform(-.1, .4)), (r1 * .92, w1 * q.uniform(.8, 1.)), (r0 + (r1 - r0) * .45, w1 * q.uniform(.8, 1.05)), (r0, w0)]
            top = self.low.project(A([V(a, b, h) for a, b in prof]))[0]; bot = self.low.project(A([V(a, b, 0.) for a, b in prof]))[0]
            if not np.isfinite(top).all() or np.abs(top).max() > 9000: continue
            away = (self.low.cam(A([V((r0 + r1) / 2, 0, h)]))[0] @ self.low.R @ np.zeros(3)) if False else None
            nt = np.cross(A(V(r1, 0, h)) - A(V(r0, 0, h)), A(V(r0, w1, h)) - A(V(r0, 0, h))); nt = nt / (np.linalg.norm(nt) + 1e-9); nt = nt if nt[2] > 0 else -nt; vis = float(nt @ (A(LE) - A(V((r0 + r1) / 2, 0, h)))) > 0
            fill_t, fill_b = '#6f6768', '#1d1a21'
            order = [(bot, fill_b), (top, fill_t)] if vis else [(top, fill_t), (bot, fill_b)]
            c.fill(order[0][0], order[0][1])
            for a in range(len(prof)):
                b = (a + 1) % len(prof); quad = A([top[a], top[b], bot[b], bot[a]]); c.fill(quad, '#3b3539' if a % 2 else '#4a4347'); c.line(A([top[a], (top[a] + bot[a]) / 2, bot[a]]), 2.4, '#0b090d', taper=(.1, .1), seed=a)
            c.fill(order[1][0], order[1][1]); outline(c, order[1][0], 4.2, '#0b090d', i, press=.5)
            if vis:
                for k in range(3): j0, j1 = int(q.integers(0, 7)), int(q.integers(0, 7)); m = top.mean(0); c.line(spl(A([top[j0], m + q.normal(0, 14, 2), m * .4 + top[j1] * .6]), n=6), 2.2, '#2a252a', taper=(.1, .6), seed=k)
            else:
                for k in range(4): m = bot.mean(0) + q.normal(0, 26, 2); c.line(A([m - (30, 6), m, m + (34, 8)]), 2., '#3a3640', taper=(.4, .5), seed=k)
                e0, e1 = bot[2], bot[3]; c.line(A([e0, (e0 + e1) / 2, e1]), 4.6, '#5fa0ff', taper=(.2, .2), seed=2, alpha=.8)       # energy lights the torn edge
        return c.layer()

    def step(self, f):
        x, y = grid(); L, info = self.fig(f, True); hit = f >= 54; t = (f - 54) / 5.; ft = self.low.project(A([[FT[0], FT[1], .02]]))[0][0]; hp = A([info['x'], info['y']])
        im = cv2.flip(cv2.GaussianBlur(np.asarray(self.ref, f32), (0, 0), 14), 1) * .50; im = im * rgb('#b8bccb'); hz = self.low.project(A([[0, 60., 0.]]))[0][0][1]
        gr = smooth(hz - 30, hz + 60, y + (x - 960) * math.tan(.16)); im = im * (1 - gr[..., None]) + (ramp(np.clip((y - hz) / (H - hz), 0, 1), [(0, '#6a605f'), (.5, '#463f43'), (1, '#1f1b20')]) * (1 + .10 * np.roll(self.n1, 300, 1).astype(f32))[..., None]) * gr[..., None]
        im = self.smoke(im, f, -100, 900, .34, '#3d3a44', 14., 1, .35)
        en = .42 if not hit else 1.0
        if hit and f < 69:                                              # the explosion goes backward, behind him
            k = f - 54; c = Cel(); R = [620, 1180, 1420, 1500, 1460, 1400][min(k, 5)] * (1 - .012 * max(0, k - 5)); r = np.random.default_rng(90 + (k // 2)); o = hp * .45 + ft * .55 + A([0, -60])
            for col, sc_, n_, sd in (('#04050c', 1.0, 34, 3), ('#0a1136', .86, 30, 5), ('#1a3cc8', .66, 26, 7), ('#4f9dff', .44, 22, 9), ('#dff2ff', .24, 16, 11)):
                q = np.random.default_rng(sd * 10 + k // 2); a = np.sort(q.uniform(0, 2 * math.pi, n_)); pts = []
                for i, th in enumerate(a):
                    up = .62 + .55 * max(0., -math.sin(th)); rr = R * sc_ * up * (.55 + .75 * q.random() ** 1.4); dth = 2 * math.pi / n_ * .42; b = R * sc_ * .34 * (.8 + .4 * q.random()); bend = .10 * math.sin(i * 2.3 + k)
                    pts += [(o[0] + math.cos(th - dth) * b * 1.25, o[1] + math.sin(th - dth) * b), (o[0] + math.cos(th + bend) * rr * 1.25, o[1] + math.sin(th + bend) * rr), (o[0] + math.cos(th + dth) * b * 1.25, o[1] + math.sin(th + dth) * b)]
                c.fill(spl(A(pts), True, corners=range(1, len(pts), 3), n=5), col, 1. if k < 6 else max(.0, 1 - (k - 6) / 9.))
            E_ = c.layer(); im = comp.over(im, E_); im = im + comp.blur(E_[..., :3], 40) * .35
        im, fr = self.energy(im, L, f, en, (0., -1.2) if not hit else (.0, -1.5), only_blue=hit)
        if hit:
            c = Cel(); g = lambda Q: self.low.project(np.c_[np.asarray(Q, float)[:, 0] + FT[0], np.maximum(np.asarray(Q, float)[:, 1] + FT[1], LE[1] + .16), np.asarray(Q, float)[:, 2] if np.shape(Q)[1] > 2 else np.zeros(len(Q))])[0]          # nothing behind the lens
            fx.ground_crack(c, g, 2.6, float(np.clip((f - 53) / 5., 0, 1)), seed=5, n=11, col='#08070b', lip='#a89e98', side='#2a252b', w=.10, lift=0., plates=False, lw=2.6, rings=2); im = comp.over(im, c.layer())
            c = Cel(); fx.dust_skirt(c, f, ft[0] + 160, ft[1] - 40, 640, 80, 84, seed=14, n=18, life=1.0, t0=54 / 24., pal=DUSTP, front=False); im = comp.over(im, c.layer())
            im = im + comp.radial(ft[0], ft[1], 760, 250, power=1.6)[..., None] * rgb('#3f86ff') * (.8 if f < 58 else .45); im = comp.over(im, self.slabs(np.clip(t, 0, 1), False))
        im = comp.over(im, L)
        if fr is not None: keep = (1 - .95 * np.clip(1 - np.hypot(x - hp[0], y - hp[1]) / (info['s'] * 2.4), 0, 1) ** .5)[..., None]; im = im * (1 - fr.rgba[..., 3:4] * .75 * keep) + fr.rgba[..., :3] * .75 * keep
        for q in (-.30, .30): im = im + comp.radial(hp[0] + q * info['s'], hp[1], info['s'] * .60, power=1.6)[..., None] * rgb('#ffa640') * .50
        if hit:
            t0 = 54 / 24.; c = Cel(); fx.dust_skirt(c, f, ft[0] + 60, ft[1] + 60, 620, 60, 50, seed=4, n=12, life=.8, t0=t0, pal=DUSTP, back=False); fx.debris(c, f, (ft[0] + 120, ft[1] - 30), n=44, seed=9, speed=(600, 2300), ang=(-2.9, -.25), size=(6, 70), t0=t0, spread=(300, 30), light=(-.2, -.9, .4)); im = comp.over(im, c.layer())
            im = comp.over(im, self.slabs(np.clip(t, 0, 1), True))
            if f < 61: c = Cel(); fx.shockwave(c, ft[0], ft[1], (f - 54) / 7., 90., 1700., .16, 30., 3, '#eaf2ff', '#6f86c8'); im = comp.over(im, c.layer() * (1 - (f - 54) / 8.))
        return im, ft

    def ash(self, im, f, seed=1, n=90, near=8, size=(2, 11)):
        im = fx.put(im, fx.ash(f, n=n, seed=seed, wind=(-120., 60.), size=size, glow=.10, near=near), .92, .5); return im

    def frame(self, f):
        if 30 <= f < 44:
            im = self.close(f); z = 1 + .05 * (f - 30) / 14.; im = self.ash(im, f, 21, 70, 9); return comp.cam2d(im, 960, 545, z)
        if f >= 51:
            im, ft = self.step(f); s = shake(f, 26, 6, 4.0, 54) if f >= 54 else A([0., 2.5 * (-1) ** f]); z = 1.02 + (.05 if f == 54 else 0.) + .030 * comp.ease(np.clip((f - 56) / 16., 0, 1))
            out = comp.cam2d(im, 960 + s[0] * .7, 540 + s[1] * .7, z, rot=.007 * math.sin(f * 2.3) * math.exp(-(f - 54) / 4.) if f >= 54 else 0.); out = self.ash(out, f, 4, 70, 9)
            if f == 54: l = comp.lum(out)[..., None]; out = impact(out, 'bw', .14, .54); out = out * .80 + rgb('#3f7dff') * .20 * l
            return out
        im, info = self.stage(f)
        z = 1.0 + .05 * comp.ease(f / 30.) if f < 30 else 1.0 + .020 * (f - 44) / 8.; dy = 0. if f < 44 else -115 * comp.ease(np.clip((f - 45) / 6., 0, 1))
        s = shake(f // 2, 1.6, 3); out = comp.cam2d(im, 960 - (z - 1) * 300 + s[0], 540 + dy * .0 + s[1], z) if f < 44 else comp.cam2d(im, 960 - 10 + s[0], 540 + 10 + dy * .12 + s[1], 1.035)
        out = self.ash(out, f, 4, 110, 12)
        if f < 16:                                                      # white becomes ash: grey flakes on paper white, then the picture comes up through them
            w = 1 - smooth(3, 15, f); lf = fx.ash(f + 6, n=170, seed=33, wind=(-60., 60.), size=(3, 15), glow=0., near=9); wh = np.full((H, W, 3), .995, f32); k = .15 + .85 * float(smooth(0, 5, f)); wh = wh * (1 - lf.rgba[..., 3:4] * k) + lf.rgba[..., :3] * k; m = float(np.clip(w * 1.35 - .0, 0, 1)); out = out * (1 - m) + wh * m
            self.finish = dict(bloom=.34 * (1 - m), vig=.24 * (1 - m))
        else: self.finish = dict(bloom=.34, vig=.24)
        return out
