"""d4 BLACK FLASH.  0-17 sprint at the lens / 18-29 plant and coil / 30-35 eye / 36-53 the last metre / 54-56 impact drawings / 57-66 burst / 66-71 recede"""
import numpy as np, cv2, math, os
from jj import W, H, comp, kit, fig_street as FS
from jj.fx_act import *
from jj.figure import Fig, POSES
from jj.fig_pose import sprint, seq, add_pose, P as POSE, mix_pose, mirror
from jj.paint import ramp, repaint
from chars.yuji import Yuji, YHead

NT = dict(FS.NIGHT, sky=[(0, '#070a18'), (.45, '#121a34'), (.75, '#263252'), (1, '#4a5878')], far='#2c3858', haze='#3a4a70', hazed=70., left='#1c2a46', right='#141c30', front='#18213a', win='#080d1a', winl='#3f5c82', road='#20263a', roadl='#39435e', stripe='#6f7c98', lit='#ffb45a')
CON = (0, 5, 9, 14, 18)                                   # footfalls
PALF = dict(mul='#aab6de', shmul='#7c86c4')


CM = dict(curse=('#b8a0b4', '#63526c', '#30263a', '#161220'), cursed=('#a08ca4', '#54465e', '#2a2232', '#14101c'))


def ez(t, p=2.): t = min(max(t, 0.), 1.); return 1 - (1 - t) ** p


def x_(): return grid()[0]
def edge_l(L, y0=552, y1=606): return float(np.nonzero((L[y0:y1:4, :, 3] > .5).all(0))[0].min())          # first column where the whole band is flesh
def edge_r(L, y0=552, y1=606): return float(np.nonzero((L[y0:y1:4, :, 3] > .5).any(0))[0].max())


class Shot:
    id = 'd4'; n = 72
    finish = dict(bloom=.34, vig=.20)
    EVENTS = [(0, 'step', dict(side='R', note='sprint footfall, road cracks')), (5, 'step', dict(side='L')), (9, 'step', dict(side='R')), (14, 'step', dict(side='L')), (0, 'energy', dict(dur=30, note='cursed energy roar, rising')),
              (18, 'stomp', dict(size=.8, note='foot plants, ground spiders')), (19, 'crack', dict(note='road web opens')), (21, 'whoosh', dict(note='cloth: shoulder turns away')), (30, 'glint', dict(note='light crosses the eye')), (35, 'inhale', dict(note='air sucked in')),
              (36, 'silence', dict(dur=12, note='time dies; only a low hum, dust; the fist crawls')), (48, 'whoosh', dict(size=.9, note='the arm goes, four frames')), (50, 'spark', dict(note='black sparks between knuckle and cheek')), (52, 'touch', dict(note='knuckles sink into the cheek, soft and dead')), (54, 'impact', dict(size=1.0, note='BLACK FLASH: white frame, two black silhouettes')), (55, 'impact2', dict(size=1.0, note='black/red drawing')), (56, 'impact3', dict(size=.9, note='second drawing, knuckles at the lens')),
              (57, 'bolt', dict(note='black lightning; the face folds round the fist')), (58, 'bolt', {}), (59, 'crunch', dict(note='skull gives')), (60, 'bolt', {}), (58, 'shockring', dict(note='distortion ring passes the lens')), (61, 'whoosh', dict(size=1.0, note='the curse is thrown out right')), (61, 'debris', dict(note='road plates lift')), (63, 'bolt', dict(size=.5)), (66, 'boom', dict(note='it is gone; blast recedes, low tail')), (68, 'stones', dict(note='stones falling')), (70, 'stones', {})]

    # ------------------------------------------------------------------ plates
    def setup(self):
        k = 'cache/d4_v3_'
        def plate(name, fn):
            p = f'{k}{name}.npy'
            if os.path.exists(p): return np.load(p, mmap_mode="r")
            np.save(p, fn().astype(np.float16)); return np.load(p, mmap_mode="r")
        self.W2, self.H2 = 2208, 1242
        self.camA = lambda d=0., w=W, h=H, f=860.: Camera3D((.25, d, .26), (.10, d + 10, 2.9), focal=f * w / W, w=w, h=h, roll=-.085)
        self.cP = Camera3D((.25, 0, .26), (.10, 10, 2.9), focal=800. * self.W2 / W, w=self.W2, h=self.H2, roll=-.085)
        self.bgA = plate('A', lambda: FS.street(self.cP, seed=3, pal=NT, lit=.03, stripes=False))
        self.camB = Camera3D((-1.15, -1.95, .20), (.16, .2, .98), focal=980., roll=.08); self.bgB = plate('B', lambda: FS.street(self.camB, seed=8, pal=NT, lit=.03, stripes=False))
        self.camC = Camera3D((-.6, -6., 1.5), (.8, 8, 2.1), focal=1500.); self.bgC = plate('C', lambda: FS.street(self.camC, seed=5, pal=NT, lit=.05, stripes=False))
        self.camE = Camera3D((.2, -3.9, .55), (.35, 0, 1.02), focal=1350., roll=-.03); self.bgE = plate('E', lambda: FS.street(Camera3D((.2, -5.6, .62), (-3.4, 4, 2.1), focal=1150.), seed=12, pal=NT, lit=.03, stripes=False))
        self.camF = Camera3D((-.4, -17., 1.9), (.6, 0, 2.2), focal=1500., roll=.02)
        self.fig = Fig('yuji').anim(self.run); self.figB = Fig('yuji').anim(self.wind); self.rays = None
        [self.curse(n) for n in ('grin', 'shock', 'h1', 'h2')]; self.curse('h3', 1400, 560, 470); self.curse('h5', 1240, 545, 560); [self.curse(*q) for q in self.TIGHT]; [self.bust(d) for d in range(4)]; self.thrown()

    # ------------------------------------------------------------------ A: sprint
    def rel(self, f): return 2.18 - .62 * ((f / 17.) ** 1.15 if f > 0 else f / 17.)
    def run(self, f):
        q = sprint(f / 9.); q = add_pose(q, pel=(10, 0, 0), cst=(6, 0, 0), nck=(-8, 0, 0)); return dict(pose=q, heading=math.pi + .22 - .02 * f / 9., x=.40 - .012 * f, y=self.rel(f), z=0.)
    def A(self, f):
        v = 5.6 / 24.; d = v * (17 - f); cP, camF = self.cP, self.camA(d); jx, jy = kit.shake(f, 3.6, 3) + sum(kit.shake(f, 15 * math.exp(-(f - c) / 1.6), 7) for c in CON if f >= c)
        mx, my, t = box_map(cP, camF); bg = cv2.remap(f16(self.bgA), (mx + jx * .9).astype(f32), (my + jy * .9).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101); hx, hy = camF.project(np.array([.05, d + 60, .30]))[0]; bg = kit.zoomblur(bg, hx, hy, .045, 7) * .92
        fl = Camera3D((.25, 0, .26), (.10, 10, 2.9), focal=800., roll=-.085); fl.cx += jx; fl.cy += jy; im = bg
        for c in CON:                                        # cracks stay in the road behind him
            if f < c: continue
            yr = self.rel(c) - .25 + v * (c - f) + v * (f - c) * 2 * 0 + (self.camA(v * (17 - c)).eye[1] - d); fx_ = .40 - .012 * c + (.10 if CON.index(c) % 2 else -.10)
            if yr < .5: continue
            g = ground_proj(fl, centre=(fx_, yr)); k = fl.f / yr; cel = Cel(); ground_crack(cel, g, 1.05, min(1, (f - c + 1.2) / 4.), seed=11 + c, n=10, rings=2, col='#05070f', lip='#8fa6d8', side='#141a2c', top='#36425e', topdark='#28324a', lw=max(1., min(3., k / 160)), lift=.34)
            p = fl.project(np.array([fx_, yr, 0.]))[0]; debris(cel, f - c, p, 16, c + 2, (k * .5, k * 2.4), (-2.7, -.45), (k * .012, k * .085), k * 5.5, pal=NIGHTROCK, spread=(k * .2, k * .03), lw=1.8, streak='#8fa6d8')
            if f - c < 14: smoke_burst(cel, f - c, p[0], p[1], k * .26, c + 5, 5, (-2.9, -.2), k * .9, .6, dict(lit='#7c88ac', mid='#4a5578', dark='#2a3350', ink='#0c101e'), lw=1.6)
            im = put(im, cel.layer())
        self.fig.at(f); L = self.fig.draw(fl, f, light=(-.45, .5, .6), expr=dict(mood='rage'), pal=PALF, kick=(.85, .35, -.4)); hm = self.fig.mask('hand', 'farm'); hd = self.fig.mask('head')
        k = fl.f / self.rel(f); st = max(34., k * .34); b, fr = cursed(hm, f, (0, -2.4), st, CURSE, 4, 1., .7, speed=900, light=.8); b0, _ = cursed(hd, f, (.2, -1.6), st * .55, CURSE, 14, .8, .5, speed=900, front=False, light=.4); im = put(im, b0); im = put(im, b); im = put(im, L); im = put(im, fr, light=.6)
        glow = cv2.GaussianBlur(cv2.resize(hm, (W // 8, H // 8), interpolation=cv2.INTER_AREA), (0, 0), 14); im = im + cv2.resize(glow, (W, H))[..., None] * C('#2f6cff') * .5
        im = put(im, radial_streaks(hx, hy, 70, 420, 1700, 31 + f, '#a8c4ff', .006, .42))
        c = Cel(); debris(c, f % 9, (960 + 500 * math.sin(f // 9 * 4 + 1.), 1000), 5, 40 + f // 9, (900, 2400), (-2.6, -.5), (40, 120), 2500, pal=dict(lit='#5a74b0', mid='#1a2238', dark='#0a0e1a', ink='#05070e'), lw=3); im = put(im, cv2.GaussianBlur(c.layer(), (0, 0), 7))
        return im

    # ------------------------------------------------------------------ B: plant and coil
    def wind(self, f):
        a = add_pose(POSES['run_contact'], pel=(6, 0, 0)); w = POSES['punch_wind']; ov = add_pose(w, root=(0, -.03, -.03), pel=(0, 0, -10), cst=(4, 0, -10)); return dict(pose=seq(f, [(18, a, 'snap'), (23, ov, 'io'), (27, w, 'lin'), (36, add_pose(w, root=(0, -.02, -.012)), 'hold')]), heading=-math.pi / 2 - .75, x=.10 + .14 * ez((f - 18) / 5.), y=.15, z=0.)
    def B(self, f):
        q = f - 18; jx, jy = kit.shake(f, 18 * math.exp(-q / 2.2), 5); cam = Camera3D((-1.15 + .02 * q, -1.95 + .03 * q, .20), (.16, .2, .98), focal=980. + 5 * q, roll=.08); cam.cx += jx; cam.cy += jy
        im = comp.cam2d(f16(self.bgB), W / 2 - jx, H / 2 - jy, 1.02 + .006 * q) * .62; x, y = grid()
        fp = cam.project(np.array([.28, -.35, 0.]))[0]; g = ground_proj(cam, centre=(.28, -.30)); c = Cel(); ground_crack(c, g, 1.9, min(1, (q + 1.) / 5.5), 21, 15, rings=4, col='#04060e', lip='#9db4e8', side='#121a30', top='#33405f', topdark='#232d46', lift=.28, glow='#2f6cff' if q > 5 else None, lw=2.6); im = put(im, c.layer())
        if q > 5: im = im + (np.exp(-((x - fp[0]) ** 2 / 560 ** 2 + (y - fp[1]) ** 2 / 170 ** 2)) * min(1, (q - 5) / 5.))[..., None] * C('#1c4cff') * .42
        c = Cel(); dust_skirt(c, q, fp[0], fp[1], 620, 150, 130, 4, 14, dict(lit='#8490b6', mid='#4c5880', dark='#2a3454', ink='#0a0e1c'), 1.0, back=True, front=False); im = put(im, c.layer())
        self.figB.at(f // 2 * 2 if q > 5 else f); L = self.figB.draw(cam, f, light=(.35, .55, .6), expr=dict(mood='rage'), pal=PALF, kick=(-.8, .4, -.4), weight=1.1)
        hm = self.figB.mask('handR', 'farmR'); grow = .35 + .65 * ez(q / 9., 2.); b, fr = cursed(hm, f, (-.35, -1.5), 60 + 190 * grow, CURSE, 9, 1., .9, light=1.1, speed=700); b2, _ = cursed(self.figB.mask('torso', 'uarm', 'thigh', 'shin', 'head', 'hood'), f, (0, -1.1), 70 * grow, dict(CURSE, hot='#2a6bff'), 5, .8, .5, front=False, light=.4)
        im = put(im, b2); im = put(im, b); im = put(im, L); im = put(im, fr)
        hc = np.argwhere(hm[::8, ::8] > .5); 
        if len(hc): cy_, cx_ = hc.mean(0) * 8; im = im + (np.exp(-((x - cx_) ** 2 + (y - cy_) ** 2) / 380. ** 2) * grow)[..., None] * C('#2f6cff') * .30
        c = Cel()
        for i in range(34):                                  # stones lifted by the pressure hang and shiver
            a = h1(i, 1) * 6.283; rr = 160 + 900 * h1(i, 2); up = (40 + 420 * h1(i, 3)) * ez((q - 1.5 * h1(i, 4)) / 9., 2.2); sx, sy = fp[0] + math.cos(a) * rr, fp[1] + math.sin(a) * rr * .26 - up; s = 5 + 34 * h1(i, 5) ** 2.4; rock(c, sx + noise(f * .35, i) * 3, sy, s, (q * .05 + i, q * .07 + i * 2, i), i, dict(lit='#86a2e8', mid='#2e3a58', dark='#10162a', ink='#060912'), (.3 if sx < 900 else -.3, -.8, .2), lw=1.8)
        dust_skirt(c, q, fp[0], fp[1] + 30, 900, 170, 120, 4, 9, dict(lit='#6a789e', mid='#3a4668', dark='#1e2844', ink='#080c18'), .9, back=False, front=True); im = put(im, c.layer())
        return put(im, radial_streaks(cx_ if len(hc) else 960, cy_ if len(hc) else 400, 46, 300, 1500, 70 + f, '#9ec0ff', .004, .30)) if q > 4 else im

    # ------------------------------------------------------------------ B2: the eye
    def Eye(self, f):
        q = f - 30; s = 900 + 22 * q; x, y = grid(); d = np.hypot(x - 1500, y - 300); im = ramp(np.clip(d / 1700, 0, 1), [(0, '#3f7dff'), (.18, '#1c3a9c'), (.5, '#0c1538'), (1, '#04060f')]); im = put(im, radial_streaks(1500, 280, 60, 500, 1900, 90 + f, '#bcd6ff', .006, .5))
        sc = 1350 + 34 * q; Hd = YHead().pose(960, 540, sc, -.34, -.08, -.20); e0 = np.asarray(Hd.P(np.array([[-.335, -.02]])))[0]; Hd = YHead().pose(960 + 900 - e0[0] - 6 * q, 540 + 500 - e0[1], sc, -.34, -.08, -.20); c = Cel(); Yuji.head(c, Hd, Yuji.expr('snarl', dark=.15, glint=.2 + .8 * (q >= 2), gx=-.012, hatch=1., irish=1.1), Yuji.light('cursed', side=1, amt=.1), f, (.6, .5), 5); L = c.layer(); im = put(im, L)
        sh = Cel(); sh.fill(s1_(q), '#07080f'); sh.line(s1_(q)[6:-20], 7, '#5fb8ff', taper=(.2, .3), press=.6, seed=4); im = put(im, cv2.GaussianBlur(sh.layer(), (0, 0), 5))
        sw = -300 + 520 * q; band_ = np.exp(-(((x - sw) * .9 + (y - 540) * .45) / 70) ** 2); im = im * (1 - .25 * np.clip(L[..., 3:4], 0, 1) * smooth(700, 1900, x)[..., None]) + band_[..., None] * C('#4f9cff') * .22 * np.clip(L[..., 3:4], 0, 1)
        return im

    # ------------------------------------------------------------------ C: the last metre
    TIGHT = (('h3', 1190, 560, 560), ('h4', 1218, 554, 570), ('h5', 1250, 548, 582), ('h6', 1286, 542, 594))
    DENT = {'grin': ('grin', None, .0), 'shock': ('shock', None, .0), 'h1': ('hit', .20, .03), 'h2': ('hit', .44, .07), 'h3': ('hit', .58, .12), 'h4': ('hit', .74, .20), 'h5': ('hit', .88, .28), 'h6': ('hit', .96, .36)}
    def curse(self, n, cx=1430, cy=560, s=430):
        """lane nanami's curse (shots/d3.curse_head), cold night grade; n = mood + depth of the dent.  cached uint8"""
        p = f'cache/d4_curse4_{n}_{cx}_{cy}_{s}.npy'
        if not os.path.exists(p):
            from shots.d3 import curse_head
            mood, dp, ro = self.DENT[n]; c = Cel(); curse_head(c, cx, cy, s, ro, mood, None if dp is None else (np.array([-1.0, .04]), np.array([dp, -.06 * dp]), .60 + .16 * dp), 7.4, -1, 36, True); L = c.layer(); yy = grid()[1]; L = kit.rim(L, 8, 1, '#a8d0ff', .95, .6, where=(yy > 14).astype(f32)); L = kit.para(L, (cx - 500, 420), (cx + 520, 820), [(0, '#e6eeff'), (.4, '#9aa6cc'), (1, '#3f4468')], 'mul', .9); np.save(p, (np.clip(L, 0, 1) * 255 + .5).astype(np.uint8)); del L
        return np.asarray(np.load(p, mmap_mode='r'), f32) / 255.
    def arm_to(self, c, E, wE, Sh, wS, base='#161b2c', sh='#0a0d18', hi='#4c5f94', ink='#05070e'):
        """upper arm and sleeve from a shoulder Sh to the end E of the drawn forearm"""
        d = E - Sh; L = np.hypot(*d); d = d / L; m = np.array([-d[1], d[0]]); mid = (E + Sh) / 2 + m * 10; P = spl(np.array([E + m * wE, mid + m * (wE + wS) * .54, Sh + m * wS, Sh - d * 90, Sh - m * wS, mid - m * (wE + wS) * .46, E - m * wE]), True, (2, 4), n=9); c.fill(P, base); c.clip(P); c.fill(spl(np.array([E + m * wE * .2, mid + m * wS * .3, Sh + m * wS * .1, Sh + m * wS * 1.5, mid + m * wS * 1.5, E + m * wE * 1.6]), True, n=8), sh)
        for j in range(4): v = .14 + .20 * j; p = E + (Sh - E) * v; w_ = wE + (wS - wE) * v; q = spl(np.array([p - m * w_, p - m * w_ * .3 - d * 24, p + m * w_ * .35 - d * 8, p + m * w_ * .95 + d * 26]), n=8); c.line(q[:int(len(q) * (.6 + .3 * h1(j, 3)))], 5, ink, taper=(.1, .6), seed=j, press=.5); a = q[2:int(len(q) * .55)]; c.fill(np.vstack([a, (a + d * (14 + 20 * h1(j, 4)))[::-1]]), hi, .85)
        c.unclip(); kit.outline(c, P, 6, ink, 7); return c
    def bust(self, d):
        """Yuji over his own shoulder, four drawings (cached)"""
        p = f'cache/d4_bust4_{d}.npy'
        if not os.path.exists(p):
            t = d / 3.; Hd = YHead().pose(318 + 34 * t, 300 + 6 * t, 190 + 20 * t, .48, -.02, -.14); B = Yuji.bust(Hd, Yuji.expr('snarl', glint=.9, dark=.08, jaw=.03 + .05 * t), Yuji.light('cursed', amt=-.1), 36 + d * 5, (.35, .5), 4, base=(212 + 26 * t, 660), byaw=.80, raise_=.15); B = cv2.GaussianBlur(B.astype(f32), (0, 0), 1.3); np.save(p, (np.clip(B, 0, 1) * 255 + .5).astype(np.uint8)); del B
        return np.asarray(np.load(p, mmap_mode='r'), f32) / 255.
    def Cc(self, f):
        q = f - 36; x, y = grid(); dr = min(1., (q + 1) / 5.); gap = {48: 212., 49: 188., 50: 130., 51: -20., 52: -150., 53: -250.}.get(f, 300. - 7.2 * q); run = max(0., (f - 47) / 6.); t = q / 17.; n = 'grin' if f < 47 else 'shock' if f < 52 else 'h1' if f == 52 else 'h2'; go = dr * (1 - .55 * run)
        bg = comp.cam2d(f16(self.bgC), W / 2 + 8 * t, H / 2, 1.06 + .05 * t); bg = comp.blur(bg, 9); im = mono(bg, .92 * go, '#7f96c4', 1.05) * .8; im = put(im, radial_streaks(960, 575, 60 + int(60 * run), 360, 1500, 5 if f < 48 else f, '#c8d8f4', .004, .20 + .25 * run))
        if f >= 48: c = Cel(ss=1); kit.speed_lines(c, -.04, 90, 8, (300, 1500), '#dfeaff', f, alpha=.35 * run); kit.speed_lines(c, -.04, 60, 12, (400, 1500), '#04060c', f + 9, alpha=.6 * run); im = put(im, c.layer())
        B = self.bust(min(3, q // 5)); B = np.dstack([mono(B[..., :3], .86 * go, '#93a8d4', .96), B[..., 3]]); im = put(im, B)
        s = 300 + 40 * t + 70 * run; cx, cy = 962 - gap - s * .44, 590 - 8 * t - 10 * run; pl = dict(NIGHTF, nail='#bfc9de', nail_shade='#5c688c', skin='#aab4cc', shade='#4c587c', tape='#c6cde0', tape_shade='#667298', line='#070912'); R_ = E3(136 + 8 * run, 30 + 6 * run, -78); dist = 2.3 - .35 * run
        L, g = fist(cx, cy, s, R_, 'R', dist, 'fist', pl, (.35, .55, .65), 5.8, 7, 400, True, True, dict(cloth='#161b2c', cloth_sh='#0a0d18', cloth_lt='#4c5f94', line='#05070e', cast='#4c587c'), rimdir=(.8, .5), rimcol='#a8c8ff')
        F_ = self.curse(n); F_ = np.dstack([mono(F_[..., :3], .55 * go, '#8fa4cc', 1.0), F_[..., 3]]); F_ = scale_layer(F_, 1 + .012 * t, 1430, 560, -5 * t + (0 if f < 52 else 8 if f == 52 else 22), 0); ck = edge_l(F_)
        if f >= 52: dx = ck + (46 if f == 52 else 80) - edge_r(L); L = kit.shifted(L, dx, 0); cx += dx
        cam = HD.Cam(cx, cy, s, dist); h = HD.Hand('fist', 'R', R=R_, pivot=(0, .8, 0), bld=HD.HEAVY); E = cam.px(h.world(np.array([0, -1.9, 0.]))); wE = np.hypot(*(cam.px(h.world(np.array([.46, -1.8, 0.]))) - cam.px(h.world(np.array([-.46, -1.8, 0.]))))) / 2; c = Cel(); self.arm_to(c, E, wE, np.array([96. + 30 * t, 770.]), 140); A = c.layer(); A = kit.rim(A, 0, -5, '#6f9cff', .7, .5, where=(y < H - 10).astype(f32)); L = L + A * (1 - L[..., 3:4])
        b, fr = cursed(L, 36 + (q // 3 * 3 if f < 48 else q), (-1.6 - 1.4 * run, -.5), 110 + 40 * run, dict(CURSE, out='#16224e', body='#3f62b0', hot='#9cc4f0', core='#f2f8ff'), 6, .8, .6, light=.35 + .3 * run, step=3 if f < 48 else 1); im = put(im, b, light=.6)
        if f < 52: im = put(im, F_); im = put(im, L)
        else:
            im = put(im, L); im = put(im, F_); fm = (L[..., 3] * (1 - smooth(ck + 12, ck + 34, x + 10 * np.sin(y / 31.)))).astype(f32)[..., None]; im = im * (1 - fm) + (L[..., :3] / np.maximum(L[..., 3:4], .02)) * fm; c = Cel()
            for j in range(4 + 2 * (f - 52)): a = np.linspace(-1.2, 1.2, 24); rr = 70 + 54 * j; thin(c, c2(ck + 10 + np.cos(a) * rr, 577 + np.sin(a) * rr * 1.25), 5.4 - .6 * j, '#15101c', .75, .3, .45, j, .3, 24)
            im = put(im, c.layer())
        im = put(im, fr, light=.25)
        for k_ in range(3):                                   # hanging dust: almost still, then swept
            c = Cel(ss=1)
            for i in range((60, 34, 12)[k_]): X, Y = h1(k_, i, 1) * W + 120 * run * run * (k_ + 1), h1(k_, i, 2) * H; s_ = (1.2, 2.8, 12.)[k_] * (.5 + h1(k_, i, 3)); X += 6 * (k_ + 1) * t * (h1(i) - .3); Y += 4 * t; a = h1(k_, i, 4) * 6.28 + t * .4; st = 1 + 7 * run * run; c.fill(np.array([[X - math.cos(a) * s_ * st, Y], [X + s_ * .3, Y - s_ * .7], [X + s_ * st, Y + .5], [X, Y + s_ * .7]]), '#dbe6fa' if k_ < 2 else '#5a6c94', .75 if k_ < 2 else .6)
            l = c.layer(); im = put(im, l if k_ < 2 else cv2.GaussianBlur(l, (0, 0), 6))
        kp = np.array([(910 - gap + ck) / 2 if f < 52 else ck + 8., 575.])
        if f < 52: im = distort_ring(im, kp[0], kp[1], 120 + 60 * (gap / 300.), 60, 6 + 12 * run, 3, ca=1.5 + 1.5 * run)
        if f >= 50:
            c = Cel(); r_ = R(f, 4); e = f - 50
            for i in range(3 + 2 * e): a = r_.uniform(0, 6.28); l = r_.uniform(40, 90 + 46 * e); bolts_draw(c, bolt_poly(bolt_line(kp, kp + np.array([math.cos(a), math.sin(a)]) * l, f * 9 + i, .5, 4), 3.4 + e, f + i, edge=3.2))
            c.fill(star_poly(kp[0], kp[1], 3, 18 + 10 * e, 6, f, rot=f), '#ffffff'); im = put(im, FX(c.layer(), glow_poly([star_poly(kp[0], kp[1], 10, 40 + 20 * e, 7, 3)], RED, (14, 50), amt=.5)))
        v = np.clip(1 - .5 * ((x - 980) / 1300) ** 2 - .4 * ((y - 540) / 780) ** 2, .28, 1)[..., None]; im = im * (1 - dr * .9) + im * v * dr * .9
        if f >= 48: jx, jy = kit.shake(f, 3 + 9 * run, 3); im = comp.cam2d(im, W / 2 - jx, H / 2 - jy, 1.02 + .035 * run)
        return im

    # ------------------------------------------------------------------ D: impact drawings
    def white(self):
        """f54: pure white, two black silhouettes locked together, the crack of the hit between them"""
        im = np.ones((H, W, 3), f32); F_ = self.curse('h3', 1400, 560, 470); ck = edge_l(F_); kx = ck + 6.; F_ = F_[..., 3:4]; s = 440; L, g = fist(900, 580, s, E3(144, 36, -78), 'R', 1.95, 'fist', NIGHTF, (0, 1, 1), 2, 7, 300, True, True); dx = ck + 60 - edge_r(L); L = kit.shifted(L, dx, 0); cam = HD.Cam(900 + dx, 580, s, 1.95); h = HD.Hand('fist', 'R', R=E3(144, 36, -78), pivot=(0, .8, 0), bld=HD.HEAVY); E = cam.px(h.world(np.array([0, -1.9, 0.]))); c = Cel(); self.arm_to(c, E, 150., np.array([60., 800.]), 170); A = c.layer()[..., 3:4]; a = np.clip(L[..., 3:4] + A, 0, 1); Y = self.bust(3)[..., 3:4]; k = np.clip(np.maximum(np.maximum(F_, a), Y), 0, 1); im = im * (1 - k) + C(BK) * k; c = Cel()
        for K_, K2 in kite_star(kx, 575, 34, 420, 18, 4, (1.5, .9), .06, 0): c.fill(K_, '#ffffff')
        for p0, p1, sd in (((kx, 575), (1960, 330), 5), ((kx, 575), (1620, 1120), 8), ((kx, 575), (kx + 120, -40), 9), ((kx, 575), (-40, 700), 12), ((kx, 575), (700, 1120), 14)): P_ = bolt_line(p0, p1, sd, .10, 6, 0.); thin(c, P_, 9, '#ffffff', 1., .01, .7, sd, .4, 60)
        c.fill(star_poly(kx, 575, 34, 110, 9, 3), '#ffffff'); return put(im, c.layer()) * 1.14
    def imp1(self):
        """side: knuckles buried, the face wrapped round them, three tones"""
        F_ = self.curse('h5', 1240, 545, 560); ck = edge_l(F_); c = Cel(); c.fill(np.array([[0, 0], [W, 0], [W, H], [0, H]]), RED); cx, cy = ck, 556; radial_wedges(c, cx, cy, 120, 330, 2400, 4, .036, BK); radial_wedges(c, cx, cy, 16, 430, 2400, 8, .012, BONE); im = c.layer()[..., :3]
        im = put(im, toned(F_, 'red', lo=.20, hi=.40)); L, g = fist(960, 560, 430, E3(112, 16, -84), 'R', 3.0, 'fist', dict(HD.PAL['night'], skin='#fff6f0', shade='#060308', tape='#fff6f0', tape_shade='#060308', line='#060308', nail='#fff6f0', nail_shade='#060308'), (-.2, .8, .4), 8, 7, 420, True, True, dict(cloth='#060308', cloth_sh='#060308', cloth_lt='#f0142e', line='#060308', cast='#060308')); L = kit.shifted(L, ck + 84 - edge_r(L), 0); fm = (L[..., 3] * (1 - smooth(ck + 18, ck + 22, x_()))).astype(f32)[..., None]; im = im * (1 - fm) + (L[..., :3] / np.maximum(L[..., 3:4], .02)) * fm
        c = Cel(); S_ = kite_star(cx + 30, cy, 50, 470, 18, 6, (math.pi / 2, .9), .055, 6)
        for K, K2 in S_: c.fill(K2, BK)
        for K, K2 in S_: c.fill(K, BONE)
        c.fill(star_poly(cx + 30, cy, 40, 104, 11, 4), BONE); return put(im, c.layer())
    def imp2(self):
        """from inside the blow: four knuckles at the lens, the world reduced to cracks"""
        c = Cel(); c.fill(np.array([[0, 0], [W, 0], [W, H], [0, H]]), BK); cx, cy = 900, 560; radial_wedges(c, cx, cy, 70, 520, 2400, 14, .020, BONE); radial_wedges(c, cx, cy, 40, 480, 2400, 15, .05, RED); im = c.layer()[..., :3]
        L, g = fist(cx, cy + 20, 560, E3(186, 74, 9), 'R', 2.0, 'fist', dict(HD.PAL['night'], skin='#fff6f0', shade='#f0142e', tape='#fff6f0', tape_shade='#f0142e', line='#060308'), (-.75, .5, .35), 12, 7, 520, True, False, deep='#060308', thr=.10); im = put(im, L)
        c = Cel(); allp = []
        for i in range(13):
            a = i / 13 * 6.283 + .4 * h1(i, 3); p0 = np.array([cx + math.cos(a) * 330, cy + math.sin(a) * 290]); p1 = np.array([cx + math.cos(a) * (760 + 700 * h1(i, 4)), cy + math.sin(a) * (540 + 500 * h1(i, 4))]); pl, ln = bolt_tree(p0, p1, 16 + 34 * h1(i, 5), 77 + i, .36, .5, 1, edge=14); allp += pl
        for K, K2 in allp: c.fill(K2, BONE)
        for K, K2 in allp: c.fill(K, BK)
        return put(im, c.layer())

    # ------------------------------------------------------------------ E: tight on the folding face (57-60), then one camera that pulls back to the end (61-71)
    def tight(self, f):
        q = f - 57; x, y = grid(); jx, jy = kit.shake(f, 26 * math.exp(-q / 2.4), 9); F_ = self.curse(*self.TIGHT[q]); ck = edge_l(F_); cx, cy = ck + 24., 572.; d = np.hypot((x - cx) / 1.2, y - cy); im = ramp(np.clip(d / 1250, 0, 1), [(0, '#fff0e4'), (.08, '#ff9a8c'), (.2, '#e81c34'), (.48, '#7a0a1c'), (1, '#16040c')]) * (1.1 - .14 * q); im = put(im, black_flash_layers(5 + q, cx, cy, 1.05, 11, 0., 14, 1100, .8, part='back', shard=.5))
        F_ = np.dstack([mono(F_[..., :3], .68, '#ff8a80', 1.75, 1.15), F_[..., 3]]); F_ = kit.para(F_, (cx, cy), (1920, 720), [(0, '#fff2e8'), (.30, '#f0b0a4'), (.65, '#a05264'), (1, '#54284c')], 'mul', .85); F_ = F_ * np.array([1., 1., 1., 1.], f32); F_ = kit.rim(F_, 9, 1, '#ff5a5e', .95, .5, where=(y > 14).astype(f32)); im = put(im, F_)
        s = 430; R_ = E3(128, 26, -80); L, g = fist(960, 585, s, R_, 'R', 2.4, 'fist', dict(NIGHTF, skin='#e8b8a8', shade='#7a4a64', tape='#f0dcd8', tape_shade='#9a6a84', line='#0a0508', nail='#f0c8bc', nail_shade='#a06a80'), (.8, .4, .5), 6.4, 7, 400, True, True, dict(cloth='#1a1526', cloth_sh='#0c0912', cloth_lt='#7a3f5c', line='#07050a', cast='#7a4a64'), rimdir=(.9, .4), rimcol='#ff8a80'); dx = ck + 84 + 8 * q - edge_r(L); L = kit.shifted(L, dx, 0); cam = HD.Cam(960 + dx, 585, s, 2.4); h = HD.Hand('fist', 'R', R=R_, pivot=(0, .8, 0), bld=HD.HEAVY); E = cam.px(h.world(np.array([0, -1.9, 0.]))); c = Cel(); self.arm_to(c, E, 150., np.array([40., 790.]), 176, '#1a1526', '#0c0912', '#7a3f5c'); A = c.layer(); L = L + A * (1 - L[..., 3:4]); fm = (L[..., 3] * (1 - smooth(ck + 26, ck + 48, x + 10 * np.sin(y / 31.)))).astype(f32)[..., None]; im = im * (1 - fm) + (L[..., :3] / np.maximum(L[..., 3:4], .02)) * fm
        c = Cel()                                            # the cheek piles up round the knuckles and ripples away
        for j in range(7): a = np.linspace(-1.25, 1.25, 26); rr = 66 + 62 * j + 22 * q; thin(c, c2(ck + 20 + np.cos(a) * rr * 1.1, cy + np.sin(a) * rr * 1.25), 6.4 - .6 * j, '#1c0810', .8, .3, .45, j, .3, 26)
        for K_, K2 in kite_star(cx, cy, 34, 270 - 46 * q, 14, 11 + q, (1.5, .9), .06, 6): c.fill(K2, RED); c.fill(K_, BONE if q < 1 else BK)
        im = put(im, c.layer()); im = distort_ring(im, cx, cy, 260 + 210 * q, 110, 46 * math.exp(-q / 2.), 3, ca=3.5); fl = black_flash_layers(5 + q, cx, cy, 1.05, 11, 0., 14, 1100, .8, part='front', star=False, shard=0.); far = smooth(330, 620, d)[..., None]; im = put(im, FX(fl.rgba * far, fl.light * far if fl.light is not None else None)); return comp.cam2d(im, W / 2 - jx, H / 2 - jy, 1.06 - .008 * q)
    def thrown(self):
        """the whole curse (lane nanami's design), bent by the blow, arms trailing; cached upright, turned at draw time"""
        p = 'cache/d4_thrown3.npy'
        if not os.path.exists(p):
            from shots.d3 import curse_head, CP
            c = Cel()
            curse_head(c, 960, 360, 92, .30, 'hit', (np.array([-1.0, .04]), np.array([.9, -.05]), .75), 3.4, -1, 61, True, legs=True); L = c.layer(); np.save(p, (np.clip(L, 0, 1) * 255 + .5).astype(np.uint8)); del L
        return np.asarray(np.load(p, mmap_mode='r'), f32) / 255.
    def E(self, f):
        """61-71: one continuous camera.  the curse is gone right by 66, the blast recedes, stones fall"""
        q = f - 61; x, y = grid(); e = q / 10.; e = e * e * (3 - 2 * e); jx, jy = kit.shake(f, 12 * math.exp(-q / 3.), 9); cm = Camera3D((.15 - .2 * e, -3.6 - 4.6 * e, .55 + .5 * e), (.30 + .25 * e, 0, 1.06 + .34 * e), focal=1350. + 60 * e, roll=-.03 + .04 * e); cm.cx += jx; cm.cy += jy; im = comp.cam2d(f16(self.bgE), W / 2 + 60 * e - jx, H / 2 - 30 * e - jy, 1.26 - .18 * e) * .92; hp = cm.project(np.array([.45, 0, 1.15]))[0]; d = np.hypot((x - hp[0]) / 1.2, y - hp[1]); fade = math.exp(-q / 3.4); im = im * (1 - .35 * fade) + ramp(np.clip(d / 1150, 0, 1), [(0, '#ffd8cc'), (.10, '#ff6a60'), (.25, '#c01a30'), (.5, '#5a0a18'), (1, '#16040c')]) * .8 * fade
        Y = Fig('yuji').pose(mirror(mix_pose(POSES['punch_hit'], POSES['punch_follow'], min(1., .55 + .12 * q)))).place(-.45 + .02 * min(q, 4), 0, 0, -math.pi / 2 + .05); Yl = Y.draw(cm, f, light=(.85, .3, .4), expr=dict(mood='shout' if q < 5 else 'rage'), pal=dict(mul='#f0c4c8', shmul='#8a6a92'), kick=(.9, .2, -.2)); Yl = kit.rim(Yl, -4, 1, '#ff5a64', .95 - .05 * q, .5); hx, hy = Y.info['hands']['L']
        g = ground_proj(cm, centre=(.35, .1)); gp = cm.project(np.array([.6, 0, 0.]))[0]; c = Cel(); ground_crack(c, g, 3.6, 1., 31, 15, rings=4, col='#04060e', lip='#b86a78', side='#1a1422', top='#3a3a52', topdark='#26283c', lift=.26, lw=2.2); ks = 1350 / (3.6 + 4.6 * e)
        puffs(c, billow(.30 + q * .075, gp[0] + 40 + 12 * q, gp[1] - 30, ks * .62, 3, 9, 1.2, 1.2, 1.7, 1.4), dict(lit='#b09aa8', mid='#5c4e66', dark='#2a2438', ink='#0c0a14'), (-.7, -.6), 2., 3); im = put(im, c.layer())
        im = put(im, black_flash_layers(9 + q, hx + 40, hy, 1.0 - .3 * e, 11, 0., 14, 1000, .9, part='back', shard=.5))
        if q < 6:                                            # thrown out of frame, drawn with its own speed
            xx = 1.50 + .50 * q + .24 * q * q; pc = cm.project(np.array([xx, .25 * q, 1.12 + .10 * q]))[0]; sc = 2.15 * cm.f / (3.6 + 4.6 * e + .25 * q) / 700.; Cl = scale_layer(self.thrown(), sc, 960, 560, pc[0] - 960, pc[1] - 560, 1.30 + .13 * q); Cl = np.dstack([mono(Cl[..., :3], .5, '#d88a94', 1.25), Cl[..., 3]]); Cl = kit.para(Cl, (pc[0] - 420 * sc, pc[1]), (pc[0] + 460 * sc, pc[1] + 60), [(0, '#ffe0d4'), (.4, '#d89a9c'), (1, '#4c3458')], 'mul', .9); Cl = kit.rim(Cl, 6, 1, '#ff5a60', .95, .5); im = put(im, drawn_tail(Cl, -(140 + 80 * q), -20, 3 + q)); im = put(im, Cl)
        im = put(im, Yl); c = Cel(); debris(c, 3 + q, (gp[0] + 60, gp[1]), 38, 14, (500, 1900), (-2.5, -.15), (4, 46 * (1 - .4 * e)), 1800, pal=dict(lit='#d08a98', mid='#4a3c54', dark='#1a1626', ink='#0a080e'), spread=(200, 20), lw=1.8, floor=gp[1] + 90, streak='#c08a98')
        for k_ in range(2): rr = 300 + 92 * q + 60 * k_; ragged_ring(c, hp[0], hp[1], rr, rr * .84, (6.5 - 3 * k_) * (1 - q / 11.) + .5, 5 + k_, ('#d8aab4', '#8a3a4a')[k_], gaps=.5, n=13, rag=.02, spurs=.6)
        m = np.clip(1 - Yl[..., 3:4] * 1.0, 0, 1); l = c.layer(); im = im * (1 - l[..., 3:4] * m) + l[..., :3] * m; return im

    def frame(self, f):
        f = int(f)
        if f < 18: return self.A(f)
        if f < 30: return self.B(f)
        if f < 36: return self.Eye(f)
        if f < 54: return self.Cc(f)
        if f == 54: return self.white()
        if f == 55: return self.imp1()
        if f == 56: return self.imp2()
        if f < 61: return self.tight(f)
        return self.E(f)


def drawn_tail(L, dx, dy, seed, n=46):
    """speed of a thrown body drawn as tapered strokes in its own colours, streaming from its trailing edge"""
    a = L[..., 3]; ys, xs = np.nonzero(a[::6, ::6] > .6); c = Cel()
    if len(xs) < 20: return c.layer()
    r = R(seed, 5); k = r.integers(0, len(xs), n); d = np.array([dx, dy], float); l = np.hypot(*d); u = d / l; m = np.array([-u[1], u[0]])
    for i in k:
        p = np.array([xs[i] * 6, ys[i] * 6], float); col = L[int(p[1]), int(p[0]), :3]; w = 2 + 15 * bms(r.random()); e = p + d * r.uniform(.5, 1.5); c.fill(np.array([p - u * 20, p + m * w, e, p - m * w]), col * r.uniform(.55, 1.05), .9)
    return c.layer()


def s1_(q): return kit.s1(f'M -60 {760-6*q} C 160 {700-6*q} 420 {760-6*q} 640 {930-4*q} C 760 1010 900 1080 980 1180 L -60 1180 Z')
