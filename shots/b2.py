"""b2 THE THREE walking downhill toward us: a step on every beat, Nobara points (18), Yuji cracks up (36), Megumi sighs (54)."""
import numpy as np, cv2, math
from jj import comp, W, H
from jj.paint import rgb, f32, fbm, smooth, ramp, grid
from jj.ink import Cel, spl, resample
from jj.kit import outline, shifted, smear, para
from chars import mn, dog
from chars.mn_keys import hpal, nz, facets
from chars.megumi import Megumi as M, BODY as MB, EXPR as ME
from chars.nobara import Nobara as N, BODY as NB, EXPR as NE

CW, CH = 2240, 1260                                     # character canvas; frame centre = (CW/2, CH/2)
E3 = lambda t, p=3: 1 - (1 - np.clip(t, 0, 1)) ** p


def dtime(f):
    """figures on twos inside every 9-frame step, a new drawing exactly on each beat"""
    return f // 9 * 9 + min(f % 9 // 2, 4) * 2


def step(t, amp=1., lag=0):
    """walk state at frame t: body drop px (14 on the contact, 3 more as overshoot), sway px, shoulder swing -1..1"""
    t = t - lag; p = t % 9; c = math.cos(math.pi * t / 9.); return float(np.interp(p, [0, 2, 4, 6, 8, 9], [14, 17, 6, -3, 0, 14])) * amp, 8. * c * amp, float(c)


def vel(t, amp=1., lag=0):
    """how fast the body moves up (+), for hair and cloth that arrive late"""
    return (step(t - 2, amp, lag)[0] - step(t, amp, lag)[0]) / 17.


def dog_shift_(a, dx, dy): return cv2.warpAffine(a, np.float32([[1, 0, dx], [0, 1, dy]]), (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def key(f, *ks):
    """linear keys (frame, value)"""
    return float(np.interp(f, [k[0] for k in ks], [k[1] for k in ks]))


class Shot:
    id = 'b2'; n = 72
    finish = dict(bloom=.36, vig=.12, halation='#ffdcae')
    EVENTS = [(k * 9, 'footsteps3', dict(size=.35, note='three pairs of shoes on asphalt, slightly apart')) for k in range(8)] + [(k * 9 + 3, 'bag_rustle', dict(size=.25)) for k in range(8)] + [(0, 'cicada_bed', dict(len=72, size=.25, note='late afternoon, far crossing, wind in leaves')), (16, 'sleeve_snap', dict(size=.4, note='Nobara throws her arm out')), (18, 'nobara_call', dict(size=.6, note='short bright shout, no words needed')), (36, 'yuji_laugh', dict(len=14, size=.6)), (54, 'megumi_sigh', dict(len=8, size=.4, note='small puff of breath'))]

    def setup(self):
        from jj import world
        p = world.plate('hillroad', time='golden'); self.bg = np.asarray(p.img, np.float16); self.skym = np.asarray(p.mask['sky'], np.float16); self.sky = np.asarray(p.layers['sky'][..., :3] if 'sky' in p.layers else p.img, np.float16); self.C = {}; del p
        # leaf shadow texture: clusters of leaf-shaped holes
        r = np.random.default_rng(14); t = np.zeros((1700, 2800), np.uint8)
        for _ in range(46):
            cx, cy = r.uniform(0, 2800), r.uniform(0, 1700)
            for _ in range(int(r.integers(14, 46))):
                x, y = cx + r.normal(0, 95), cy + r.normal(0, 70); a = r.uniform(0, 180); l = r.uniform(18, 44); cv2.ellipse(t, (int(x), int(y)), (int(l), int(l * r.uniform(.32, .5))), a, 0, 360, 255, -1, cv2.LINE_AA)
            cv2.line(t, (int(cx - 160), int(cy + r.normal(0, 60))), (int(cx + 170), int(cy + r.normal(0, 60))), 255, int(r.integers(5, 12)), cv2.LINE_AA)
        self.leaf = cv2.GaussianBlur(t.astype(f32) / 255., (0, 0), 2.6)
        self.near = self.branch(5, 1); self.near2 = self.branch(9, -1)

    def branch(self, seed, flip):
        """big out-of-focus leaves hanging into frame"""
        c = Cel(1200, 900); r = np.random.default_rng(seed)
        for i in range(120):
            u = r.random(); x, y = 40 + u * 760 + r.normal(0, 70), 60 + 330 * u * r.uniform(.2, 1.1) + r.normal(0, 60); a = math.pi * .30 + r.normal(0, .7); l = r.uniform(70, 150); e = np.array([math.cos(a), math.sin(a)]); q = np.array([-e[1], e[0]]); p = np.array([x, y]); P = spl([p, p + e * l * .4 + q * l * .22, p + e * l, p + e * l * .4 - q * l * .22], True, corners=(0, 2), n=6)
            c.fill(P, ['#1c3320', '#2a4a26', '#3a5c2a', '#14261a'][int(r.integers(4))]); c.fill(np.vstack([P[:7], [p + e * l]]), '#b8a63e' if r.random() < .30 else '#35562a')
        return comp.bokeh(c.layer(900, 1200), 12, 0)

    def cached(self, k, fn):
        if k not in self.C:
            if len(self.C) > 2: self.C.pop(next(iter(self.C)))
            self.C[k] = fn()
        return self.C[k]

    # ------------------------------------------------------------ camera: zoom, centre (canvas px)
    def cam(self, f):
        z = comp.lerp_keys(f, [(0, 1.03), (18, 1.0), (30, .985), (44, .97), (54, .985), (71, 1.03)]); x = comp.lerp_keys(f, [(0, 1080), (18, 1066), (34, 1092), (46, 1120), (58, 1170), (71, 1182)]); y = comp.lerp_keys(f, [(0, 626), (18, 634), (40, 612), (58, 606), (71, 600)])
        return float(z), float(x) + 6.5 * math.sin(f * .21 + .6) + 3.2 * math.sin(f * .53) + 1.2 * math.sin(f * 1.31), float(y) + 4.5 * math.sin(f * .27 + 2.) + 2.4 * math.sin(f * .61 + 1.) + 1.0 * math.sin(f * 1.17)

    @staticmethod
    def hips(body, dx):
        """shift the lower body against the shoulders: returns (right, left) outlines"""
        k = lambda y: dx * float(smooth(1.0, 3.8, -y)); return [(x + k(y), y, z) for x, y, z in body], [(x - k(y), y, z) for x, y, z in body]

    # ------------------------------------------------------------ MEGUMI: far right, turned a little away, slouched, hand in pocket
    def megumi(self, d):
        from jj import hands as HD
        f = d; L = mn.LIGHT['gold']; s = 127; k = s / 127.; w = mn.lw(s * 1.04); lag = 2; bob, sw, q_ = step(d, .90, lag); sg = key(f, (49, 0), (52, -.30), (54, -.34), (56, .55), (58, 1.08), (60, 1.), (64, 1), (71, .45)); bx = 1716 + sw * 1.2; by = 520 + bob + 20 * sg
        look = float(smooth(18, 24, f) * (1 - smooth(31, 37, f))); yl = float(smooth(38, 44, f) * (1 - smooth(49, 55, f))); hb = step(d - 2, .85, lag)[0]; lidk = key(f, (0, .66), (34, .66), (36, .56), (48, .54), (51, .62), (53, .62), (54, .40), (56, .12), (58, .06), (60, .10), (62, .34), (65, .52), (72, .54))
        if 55 <= f < 63: ex = dict(ME['sigh'], lid=lidk, gaze=(-.006, -.03))
        elif f >= 63: ex = dict(ME['smile'] if f >= 65 else ME['deadpan'], lid=lidk, gaze=(-.026, -.01), brow=(.010, -.08, 0))
        elif 38 <= f < 55: ex = dict(ME['deadpan'], lid=lidk, brow=(-.006, .05, 0), gaze=(-.036 * yl - .012, -.004))
        else: ex = dict(ME['deadpan'], lid=lidk, gaze=(-.034 * look - .034 * yl - .016, -.004))
        yaw = .20 - .40 * look - .34 * yl - .14 * float(smooth(62, 66, f)); pitch = .02 - .24 * sg + .02 * (hb - bob) / 10.; roll = .05 - .04 * look + .06 * sg + .02 * q_
        cel = Cel(CW, CH); Hd = M.pose(bx - 4 * q_, by + (hb - bob), s, yaw, pitch, roll); hair = M._hair(Hd, f, (-.05, -.03 + .13 * vel(d, .85, lag), .5)); o = np.array([bx + 3 * q_, by + 1.04 * s + 7 * sg]); T = mn.Torso(o[0], o[1], s, .30 + .13 * q_, -.05, .115 * q_ - .03)
        base = [(.40, -.02, .20), (.60, -.16, 0), (.88, -.36, -.04), (1.14, -.56, -.02), (1.30, -.78, .05), (1.30, -1.1, .05), (1.24, -1.5, 0), (1.16, -2.0, 0), (1.10, -2.6, 0), (1.13, -3.2, 0), (1.22, -3.9, 0), (1.34, -4.58, 0)]; br, bl = self.hips(base, -.16 * q_)
        B = dict(MB, body_r=br, body_l=bl, bc=(), fw=2.0, mid=[[(-.95, -.62), (-.40, -.66), (-.26, -1.30), (-.48, -2.10), (-.78, -1.55)], [(-1.06, -2.7), (-.52, -2.9), (-.30, -3.6), (-.70, -3.9), (-1.08, -3.5)]], hi=[[(-.62, -.22), (-.92, -.38), (-1.22, -.62), (-1.14, -.98), (-.90, -.74), (-.74, -.44)]],
                 lines=[[(-1.30, -1.30), (-.86, -1.36), (-.36, -1.86)], [(-.46, -.42), (-.62, -.95), (-.40, -1.72)], [(1.24, -1.25), (.86, -1.32), (.48, -1.80)], [(-1.10, -2.35), (-.50, -2.62), (.14, -2.50)], [(1.10, -2.9), (.60, -3.18), (.10, -3.50)]])
        M.back(cel, Hd, 'gold', f, (0, 0), 3, hair); mn.uniform(cel, T, B, L, w, 3, 'back'); mn.neck(cel, Hd, L, w, .335, seed=3); tl = -.16 * q_
        legs = np.vstack([spl(T.P3(np.array([(-1.10 - tl * 0 + tl, -4.3, 0), (-1.24 + tl, -5.2, 0), (-1.30 + tl, -6.4, 0)])), n=8), spl(T.P3(np.array([(1.10 + tl, -4.3, 0), (1.24 + tl, -5.2, 0), (1.30 + tl, -6.4, 0)])), n=8)[::-1]]); outline(cel, legs, w[0] * 1.6, L['cloth_ln'], 4); cel.fill(legs, L['cloth_dk']); cel.line(T.P3(np.array([(tl, -4.5, .4), (tl + .02, -5.2, .4), (tl, -6.3, .4)])), w[1], L['cloth_ln'], taper=(.1, .1), seed=8)
        # far arm (screen right) goes down behind the hip; near arm: elbow out and back, hand in the jacket pocket
        j = .24 * step(d - 2, 1., lag)[2]; a_ = T.P3(np.array([(1.18, -.98, -.1), (1.62, -2.55 - j, -.45 - j), (1.20, -4.0 - j, -.1)])); mn.arm(cel, a_[0], a_[1], a_[2], [v * k for v in (34, 40, 45, 50, 52)], L, w, 60, 'fu', bow=-4, folds=3, bold=1.6, bunch=False, hem=False)
        mn.uniform(cel, T, B, L, w, 3, 'front', sleeves=(0, 0))
        hem = spl(T.P3(np.array([(-1.34 + tl, -4.58, 0), (-.6 + tl, -4.70, .5), (tl, -4.72, .55), (.6 + tl, -4.70, .5), (1.34 + tl, -4.58, 0)])), n=8); cel.line(hem, w[0], L['cloth_ln'], taper=(.05, .05), seed=9)
        sh_, el, wr = T.P3(np.array([(-1.14, -1.02, .12), (-1.82, -2.55 + j, -.35), (-.92 + tl * .6, -3.55 + j * .6, .62)])); hc = HD.Cam(wr[0] + 30 * k, wr[1] + 34 * k, 84 * k, 8); h = HD.Hand('relaxed', 'R', HD.euler(40, 10, 220), (0, 0, 0), HD.SLIM); g = HD.render([h], hc, light=(-.7, .5, .5), clip=(wr[0] - 34, wr[1] - 30, wr[0] + 120, wr[1] + 120)); HD.ink(cel, g, hpal(L), w[0] * .8, 3, inner=.6)
        pk0, pk1, pk2, pk3 = T.P3(np.array([(-1.02 + tl, -3.58, .55), (-.46 + tl, -3.74, .66), (-.48 + tl, -4.62, .66), (-1.06 + tl, -4.56, .45)])); pk = np.vstack([spl([pk0, (pk0 + pk1) / 2 + [2, 5], pk1], n=6), [pk2, pk3]]); cel.fill(pk, L['cloth']); cel.fill(np.vstack([pk[-1:], pk[:1], [pk[0] * .7 + pk1 * .3], [pk3 * .7 + pk2 * .3]]), mn.mixh(L['cloth'], L['cloth_hi'], .42)); cel.line(spl([pk0, (pk0 + pk1) / 2 + [2, 5], pk1], n=6), w[0], L['cloth_ln'], taper=(.05, .05), seed=11); cel.line(spl([pk1, pk2], n=6), w[1], L['cloth_ln'], taper=(.1, .4), seed=12)
        mn.arm(cel, sh_, el, wr, [v * k for v in (33, 41, 47, 52, 54)], L, w, 50, 'uf', bow=-5, folds=3, bold=1.7, bunch=False, openend=True)
        M.head(cel, Hd, ex, 'gold', f, (0, 0), 3, parts='f', L=hair)
        return mn.grade(cel.layer(CH, CW), L, (bx - 260, by - 200, bx + 260, by + 760), 1.)

    # ------------------------------------------------------------ YUJI: centre, hands behind his head, elbows uneven
    def yuji(self, d):
        from chars.yuji import Yuji, YHead
        f = d; s = 174; k = s / 127.; lag = 0; bob, sw, q_ = step(d, 1.10, lag); hb = step(d - 2, 1.10, lag)[0]; bx = 1128 + sw * 1.4; by = 480 + bob; la = key(f, (31, 0), (33, -.12), (35, .45), (36, .85), (38, 1.08), (40, 1.), (52, 1.), (54, .85), (56, .6), (58, .42), (62, .36)); jig = (5 * (-1) ** (d // 2)) * (38 <= f < 54)
        L = mn.LIGHT['gold']; w = mn.lw(s * .9); 
        ex = ('smile' if f < 17 else ('smile', 'surprised', .35) if f < 18 else ('smile', 'surprised', .75) if f < 20 else 'surprised' if f < 22 else ('surprised', 'smile', .35) if f < 24 else ('surprised', 'smile', .72) if f < 26 else 'smile' if f < 31 else ('smile', 'neutral', .6) if f < 33
              else ('neutral', 'laughshut', .28) if f < 35 else ('neutral', 'laughshut', .66) if f < 36 else 'laughshut' if f < 44 else ('laughshut', 'laugh', .4) if f < 45 else ('laughshut', 'laugh', .75) if f < 47 else 'laugh' if f < 54 else ('laugh', 'smile', .30) if f < 56 else ('laugh', 'smile', .66) if f < 58 else 'smile')
        yaw = .06 - .26 * float(smooth(15, 22, f) * (1 - smooth(25, 33, f))) - .05 * la + .30 * float(smooth(52, 60, f)); pitch = .05 + .13 * la + .02 * (hb - bob) / 10.; roll = -.05 + .20 * la + .022 * q_
        Hd = YHead().pose(bx - 5 * q_ + 22 * la, by + jig - 16 * la + (hb - bob), s, yaw, pitch, roll); br_ = .115 * q_; byw = .10 - .15 * q_; b0 = np.array([bx + 4 * q_, by + 1.14 * s + jig * .5])
        lay = Yuji.bust(Hd, ex, Yuji.light('sunset', side=-1, amt=.70, skin=('#ffdcae', '#dd9c7f', '#a8605a')), f, (.16, .06 + .14 * vel(d)), 3, byaw=byw, broll=br_, base=tuple(b0), breath=.5 - .5 * math.cos(math.pi * 2 * d / 9.), w=CW, h=CH, rims=False, slim=1.0)
        c_, s_ = math.cos(br_), math.sin(br_); X = lambda u, v: b0 + np.array([u * c_ - v * s_, u * s_ + v * c_]) * s          # body frame: u right, v down, units of s, origin at the neck base
        u, o = Cel(CW, CH), Cel(CW, CH); YL = dict(L, cloth='#2a2140', cloth_hi='#6b4b7c', cloth_dk='#150e1e')
        for q in (-1, 1):
            hi_ = (q > 0); sh_ = X(q * 1.30, .78 + (-.05 if hi_ else .05)); el = np.array([bx + q * (2.18 if hi_ else 2.40) * s - 12 * q_, by + ((-.34 if hi_ else .42) - .24 * la + .10 * q * q_) * s + jig * (1.4 if hi_ else .8) + 1.5 * (step(d - 4, 1.10)[0] - bob)]); wr = np.array([bx + q * .34 * s, by + (.30 if hi_ else .42) * s + jig - 10 * la])
            fo = dict(path=[wr, wr * .5 + el * .5 + [0, -10], el], width=[(0, 36 * k), (.5, 44 * k), (1, 50 * k)], seed=70 + q, hem=False, rnd=(0, .95), bunch=(.86,), folds=3, bumps=.07, bold=1.6); up = dict(path=[el, el * .5 + sh_ * .5 + [q * 9, 8], sh_], width=[(0, 50 * k), (.45, 57 * k), (.8, 64 * k), (1, 60 * k)], seed=80 + q, hem=False, rnd=(.95, .9), bunch=(.12,), folds=4, bumps=.07, openend=True, bold=1.7)
            mn.sleeve2(u, L=YL, w=w, phase='o', **up); mn.sleeve2(u, L=YL, w=w, **fo); mn.sleeve2(o, L=YL, w=w, **up)
        x = np.arange(CW)[None].astype(f32); y = np.arange(CH)[:, None].astype(f32); ux = ((x - b0[0]) * c_ + (y - b0[1]) * s_) / s; vy = (-(x - b0[0]) * s_ + (y - b0[1]) * c_) / s; hp = -.14 * q_
        HF = lambda v: 1.60 - .10 * np.sin(np.clip((v - .6) / 2.2, 0, 1) * math.pi * .5) - .12 * np.clip((v - 1.7) / 1.2, 0, 1); CX = lambda v: hp * np.clip((v - .6) / 2.4, 0, 1)
        cut = smooth(1.6, -1.6, (np.abs(ux - CX(vy)) - HF(vy)) * s) * (vy > .22) + (vy <= .22); tor = lay * cut[..., None].astype(f32); e = Cel(CW, CH)
        for q in (-1, 1): vs = np.linspace(1.15, 3.1, 30); e.line(np.array([X(CX(v) + q * HF(v), v) for v in vs]), w[0] * 1.15, L['cloth_ln'], taper=(.02, .02), seed=5 + q)
        ov = lambda a, b: b + a * (1 - b[..., 3:4]); out = u.layer(CH, CW); out = ov(out, tor); ol = o.layer(CH, CW); out = ov(out, e.layer(CH, CW))
        hm = ((np.abs(x - bx) < 1.02 * s) & (y < by + 1.1 * s)).astype(f32); return ov(out, ol * (1 - hm[..., None]))

    # ------------------------------------------------------------ NOBARA: nearest, cropped left, turned inward; bags over the shoulder, the other arm thrown at the lens
    def bag(self, c, p, w0, h0, th, col, w, kind, seed, strap=None):
        """paper shopping bag hung from p, swung by th"""
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]); X = lambda q: p + np.asarray(q, float) @ R.T; ln = '#2a1618'; a, b, k = w0 / 2, h0, w0 * .22; y0 = h0 * .17; top = np.asarray(strap if strap is not None else p, float)
        for dx in (-1, 1): c.line(spl([top, (top + X((dx * a * .40, y0))) / 2 + [dx * 5, 0], X((dx * a * .42, y0 + 4))], n=8), w[1] * 2.0, '#7a5a3a', taper=(.02, .02), seed=seed + dx)
        F = np.array([X((-a, y0)), X((a, y0)), X((a * .96, y0 + b)), X((-a * .96, y0 + b))]); S_ = np.array([X((a, y0)), X((a + k, y0 - k * .35)), X((a * .96 + k, y0 + b - k * .5)), X((a * .96, y0 + b))]); outline(c, np.vstack([F[0], F[1], S_[1], S_[2], S_[3], F[3]]), w[0] * 1.6, ln, seed); c.fill(F, col[0]); c.fill(S_, col[1]); c.clip(F)
        c.fill(np.array([X((-a, y0)), X((-a * .55, y0)), X((-a * .72, y0 + b)), X((-a, y0 + b))]), col[2]); c.fill(np.array([X((a * .30, y0 + b)), X((a * .62, y0 + b * .35)), X((a, y0 + b * .1)), X((a, y0 + b))]), col[1])
        if kind == 0: c.fill(np.array([X((-a, y0 + b * .42)), X((a, y0 + b * .40)), X((a, y0 + b * .58)), X((-a, y0 + b * .60))]), col[3])
        elif kind == 1: t = np.linspace(0, 6.283, 26)[:-1]; c.fill(X(np.c_[np.cos(t) * a * .42, y0 + b * .46 + np.sin(t) * a * .42]), col[3]); c.fill(X(spl([(-a * .12, y0 + b * .36), (a * .18, y0 + b * .42), (a * .02, y0 + b * .58), (-a * .2, y0 + b * .5)], True, n=5)), col[0])
        else:
            for q in range(5): c.fill(np.array([X((-a + q * a * .42, y0)), X((-a + q * a * .42 + a * .18, y0)), X((-a + q * a * .42 + a * .18, y0 + b)), X((-a + q * a * .42, y0 + b))]), col[3])
        c.unclip(); c.line(np.array([F[1], F[2]]), w[1] * 1.1, ln, taper=(.05, .05), seed=seed + 4); c.line(np.array([X((-a * .5, y0 + 6)), X((-a * .35, y0 + b * .5)), X((-a * .6, y0 + b - 4))]), w[2] * 1.1, ln, taper=(.3, .6), seed=seed + 5, alpha=.6)

    def nobara(self, d):
        from jj import hands as HD
        f = d; L = mn.LIGHT['gold']; s = 210; k_ = s / 127.; w = mn.lw(s * .86); lag = 0; bob, sw, q_ = step(d, 1.35, lag); hb = step(d - 2, 1.35, lag)[0]; bx = 500 + sw * 1.6; by = 640 + bob
        up = key(f, (13, 0), (15, .36), (17, .76), (18, 1.10), (20, .96), (22, 1.), (31, 1.), (33, .74), (35, .34), (36, .0)) if 14 < f < 36 else 0.; ant = key(f, (9, 0), (11, .55), (13, 1), (15, .6), (17, 0)); pta = min(1., up); pop = key(f, (15, 0), (17, .45), (18, 1), (20, .5), (24, 0)); turn = float(smooth(31, 39, f)); pt = up > 0
        closed = dict(NE['calm'], lid=1.0, gaze=(.004, 0), mouth=dict(kind='line', pts=[(-.095, -.640), (-.04, -.664), (.02, -.668), (.105, -.632)], lip=[(-.035, -.735), (0, -.742), (.035, -.735)]))
        if f < 11: ex = dict(closed, brow=(.030, -.34, 0))
        elif f < 13: ex = dict(closed, brow=(.034, -.16, 0), lid=1.08)
        elif f < 17: ex = dict(closed, brow=(.034, .02, 0), lid=1.16)
        elif f < 18: ex = dict(NE['half'])
        elif f < 33: ex = dict(NE['point'])
        elif f < 35: ex = dict(NE['half'], lid=1.0)
        elif f < 54: ex = dict(NE['smug'], gaze=(.022, .006), lid=(.90, .92) if f < 36 else (.76, .82))
        else: ex = dict(NE['smug'], gaze=(.034, .004), lid=(.70, .74) if f < 56 else (.64, .68))
        ex['spark'] = 1.2
        yaw = .24 - (.20 * pta - .06 * ant) * (1 - turn) + .22 * turn + .10 * float(smooth(52, 60, f)); pitch = .04 + .08 * pta * (1 - turn) - .08 * ant + .02 * (hb - bob) / 10. - .03 * turn; roll = -.03 - .09 * pop + .06 * turn + .022 * q_
        ly = -14 * pop + 14 * ant; cel = Cel(CW, CH); Hd = N.pose(bx - 5 * q_ + 10 * pop, by + ly + (hb - bob), s, yaw, pitch, roll); hw = key(f, (15, 0), (17, .14), (18, .30), (20, .22), (22, -.02), (24, -.16), (27, -.06), (31, .04), (33, .10), (35, .16), (38, -.10), (42, -.04), (45, .02), (72, 0)); hair = N._hair(Hd, f, (-.10 + hw + .05 * math.sin(f * .5), -.02 + .07 * float(np.clip(vel(d, 1.35), -1, 1)) + .06 * ant, .8)); o = np.array([bx + 4 * q_, by + 1.08 * s + ly * .5]); T = mn.Torso(o[0], o[1], s, .46 - .30 * pta * (1 - turn) + .12 * ant + .13 * q_, .02, .115 * q_ - .05 * pop)
        base = [(.34, .04, .18), (.50, -.08, 0), (.76, -.22, -.04), (.98, -.36, -.05), (1.12, -.54, -.05), (1.11, -.82, 0), (1.05, -1.22, 0), (1.00, -1.60, 0), (.92, -2.05, 0), (.86, -2.60, 0)]; br, bl = self.hips(base, -.14 * q_)
        B = dict(NB, body_r=br, body_l=bl, bc=(), fw=1.9, mid=[[(-.78, -.56), (-.34, -.58), (-.22, -1.05), (-.40, -1.62), (-.66, -1.22)]], hi=[[(-.52, -.15), (-.80, -.28), (-1.06, -.48), (-1.00, -.80), (-.78, -.62), (-.62, -.34)], [(.50, -.18), (.80, -.30), (1.02, -.46), (.82, -.46)]],
                 lines=[[(-1.06, -1.04), (-.74, -1.06), (-.30, -1.42)], [(-.34, -.32), (-.50, -.70), (-.40, -1.20)], [(1.06, -1.02), (.74, -1.06), (.36, -1.40)], [(-.90, -1.72), (-.46, -1.92), (.06, -1.80)], [(.92, -1.80), (.56, -2.0), (.20, -1.94)]])
        th = .13 * math.sin(math.pi * (f - 4.5) / 9.) + .03 * (hb - bob) / 10.; th2 = .16 * math.sin(math.pi * (f - 6.5) / 9.); bb = 1.7 * (step(d - 3, 1.35)[0] - bob)
        # bags behind the shoulder, hanging from the fist that holds their cords in front of it
        fs = T.P3(np.array([(.90, -.62, .62)]))[0]; an = T.P3(np.array([(1.12, -.28, -.15)]))[0]
        self.bag(cel, an + [132, 8 + bb * 1.2], 176, 236, th2 - .05, ('#2c3f78', '#1c2750', '#3c55a0', '#c9d4f0'), w, 2, 91, an + [10, 6]); self.bag(cel, an + [4, 30 + bb * .8], 186, 226, th * 1.1, ('#fbf1e2', '#d3b9a6', '#fffaf0', '#2fa39a'), w, 1, 93, an + [0, 6]); self.bag(cel, an + [74, 44 + bb], 206, 262, th, ('#f29aa8', '#c4647c', '#ffc4c8', '#fff2ea'), w, 0, 95, an + [6, 4])
        N.back(cel, Hd, 'gold', f, (0, 0), 3, hair); mn.uniform(cel, T, B, L, w, 3, 'back'); mn.neck(cel, Hd, L, w, .285, seed=3)
        if not pt:                                           # her right arm (screen left) hangs and swings, or is cocked back
            j = .30 * step(d - 2, 1.)[2]; a_ = T.P3(np.array([(-1.00, -.72, -.05), (-1.46 - .25 * ant - .10 * q_, -1.96 + j + .25 * ant, -.35 - .5 * ant), (-1.28, -3.2 + 1.5 * ant, .1)])); mn.arm(cel, a_[0], a_[1], a_[2], [v * k_ * .80 for v in (36, 42, 47, 52, 55)], L, w, 57, 'fu', bow=6, folds=3, bold=1.7, bunch=False)
        mn.uniform(cel, T, B, L, w, 3, 'front', sleeves=(0, 0))
        for q in (-5, 4): cel.line(spl(np.array([an + [q, 4], (an + fs) / 2 + [q + 4, -14], fs + [q, 4]]), n=8), w[1] * 2.0, '#7a5a3a', taper=(.02, .02), seed=30 + q)
        sh_, el, wr = T.P3(np.array([(.98, -.80, .10), (1.16, -2.10 + .10 * q_, .62), (.94, -.90 + .05 * q_, .95)])); hcam = HD.Cam(fs[0] + 4, fs[1] + 12, 118, 7); h = HD.Hand('carry', 'L', HD.euler(20, 14, -14), (0, 0, 0), HD.FEMALE); wp = hcam.px(h.world(np.array([0, -.10, 0]))); g = HD.render([h], hcam, light=(-.7, .5, .5), clip=(fs[0] - 140, fs[1] - 130, fs[0] + 140, fs[1] + 84)); HD.ink(cel, g, hpal(L), w[0] * .85, 3, inner=.6); mn.arm(cel, sh_, el, wp + (el - wp) * .10, [v * k_ * .78 for v in (36, 42, 47, 52, 55)], L, w, 52, 'uf', bow=-5, folds=3, bold=1.7, bunch=False, openend=True)
        N.head(cel, Hd, ex, 'gold', f, (0, 0), 3, parts='f', L=hair)
        lay = cel.layer(CH, CW)
        if pt:                                               # foreshortened: shoulder small, hand big; rises from below with the fist opening on the way, sinks again the same way
            ac = Cel(CW, CH); e_ = pta; dy = 4 * math.sin(d * .95); sh_ = T.P3(np.array([(-.98, -.70, .10)]))[0]; F = np.array([sh_[0] - 40 * max(up, .9), sh_[1] + 66 - 160 * (up - 1) * (up > 1) + dy + bob * .4]); lo = np.array([sh_[0] - 20, CH + 60.]); P_ = lo * (1 - e_) + F * e_ + np.array([-96, 0]) * math.sin(math.pi * e_); hs = 150 + 46 * min(1.12, up)
            hcam = HD.Cam(P_[0], P_[1], hs, 3.0); po = HD.POSES['point'] if up >= .95 else HD.blend('fist', 'point', float(np.clip((up - .20) / .70, 0, 1))); h = HD.Hand(po, 'R', HD.euler(150 + 56 * e_, 14 + 40 * e_, 12 + 24 * e_), (0, 0, 0), HD.FEMALE); wr = hcam.px(h.world(np.array([0, -.16, 0]))); g = HD.render([h], hcam, light=(-.45, .55, .75), clip=(P_[0] - 420, P_[1] - 400, P_[0] + 330, P_[1] + 360)); m = wr * .5 + sh_ * .5 + [10, 20]; wk = .62 + .38 * e_; mn.sleeve2(ac, [wr + (m - wr) * .02, wr * .75 + m * .25 + [0, 6], m, sh_], [(0, 92 * wk), (.4, 86 * wk), (.75, 72), (1, 56)], L, w, 55, bunch=(.16,), folds=4, rnd=(0, .9), bumps=.07, openend=True, bold=1.6); HD.ink(ac, g, hpal(L), w[0] * 1.1, 3, inner=.6); al = ac.layer(CH, CW)
            if 15 <= f < 18: al = smear(al, -24, 120, 7) * .5 + al * .6
            elif 33 <= f < 36: al = smear(al, 10, -90, 7) * .45 + al * .62
            lay = al + lay * (1 - np.clip(al[..., 3:4], 0, 1))
        return mn.grade(lay, L, (bx - 320, by - 260, bx + 320, by + 620), 1.)

    def posts(self, im, f, front):
        """railing posts: they come in from the edge close to the lens, then fall behind the three and shrink toward the road's end.  never faded"""
        for (t0, side) in ((4, -1), (25, 1), (46, -1)):
            t = f - t0
            if t < 0 or (t < 4) != front: continue
            e_ = 1 - (1 - min(1., t / 30.)) ** 3.6 + .0012 * max(0, t - 24); xx = (-170 + 900 * e_) if side < 0 else (W + 190 - 560 * e_); wd = 160 - 128 * e_; top = 452 + 250 * e_; bot = H + 60 if t < 12 else H + 60 - min(150, (t - 12) * 10); c = Cel(W, H, ss=1); c.fill(np.array([(xx - wd / 2, top), (xx + wd / 2, top - wd * .06), (xx + wd / 2, bot), (xx - wd / 2, bot)]), '#6d6784'); c.fill(np.array([(xx - wd / 2, top), (xx - wd * .18, top - wd * .03), (xx - wd * .18, bot), (xx - wd / 2, bot)]), '#ffcf8c'); c.fill(np.array([(xx + wd * .24, top), (xx + wd / 2, top), (xx + wd / 2, bot), (xx + wd * .24, bot)]), '#3f3a58')
            for k, (yy, hh) in enumerate(((top + 64 - 50 * e_, 62 - 50 * e_), (top + 260 - 200 * e_, 58 - 46 * e_))): x2 = -300 if side < 0 else W + 300; c.fill(np.array([(x2, yy - 60 + 90 * e_), (xx, yy), (xx, yy + hh), (x2, yy + 62 + 60 * e_)]), '#7b7592'); c.fill(np.array([(x2, yy - 60 + 90 * e_), (xx, yy), (xx, yy + hh * .3), (x2, yy - 24 + 80 * e_)]), '#ffd596')
            im = comp.over(im, comp.blur(smear(c.layer(), max(4., 70 - 110 * e_) * -side, 0, 7), max(2.2, 10 - 11 * e_)))
        return im

    def chars(self, d):
        ov = lambda a, b: b + a * (1 - b[..., 3:4]); y_ = self.yuji(d); m = self.megumi(d); n = self.nobara(d)
        # contact darkening where one stands in front of another
        x, y = grid(CH, CW)
        def sun(q, k, hy):                                  # low sun behind-left: broken lit band, hard white-gold edge; blue sky along the hair on the other side
            hm = smooth(hy + 40, hy - 60, y); q = dog.rim2(q, -3.2 * k, .4 * k, '#6fb8ff', .55, .6, where=hm); q = facets(q, 11 * k, 6 * k, '#ffa23f', .26, .26, 11, (150, 70, 30)); q = dog.rim2(q, 4.4 * k, 2.4 * k, '#ffe49a', .97, .6); return q
        m = sun(m, .85, 560); y_ = sun(y_, 1.1, 520); n = sun(n, 1.3, 640); a = m
        for q, dx, r in ((y_, 22, 12), (n, 30, 16)): s_ = cv2.GaussianBlur(shifted(q[..., 3], dx, 12), (0, 0), r) * (a[..., 3]); a = a.copy(); a[..., :3] *= 1 - .42 * s_[..., None]; a = ov(a, q)
        return a

    def frame(self, f):
        d = dtime(f); x, y = grid(); z, cx, cy = self.cam(f)
        # background recedes about the vanishing point; clouds drift
        zb = 1.150 - .150 * (f / 71.); ox, oy = (cx - 1120) * .34, (cy - 620) * .30; bgc = (1200 + ox, 690 + oy - 118)
        g = comp.cam2d(self.bg.astype(f32), bgc[0], bgc[1], zb, 0); sk = comp.cam2d(self.sky.astype(f32), bgc[0] - f * .35, bgc[1], zb, 0); m = comp.cam2d(self.skym[..., None].astype(f32), bgc[0], bgc[1], zb, 0)
        m = m[..., None] if m.ndim == 2 else m; im = g * (1 - m) + sk * m
        im = im * .80 + comp.blur(im, 2.2) * .20                 # a little depth of field
        im = im * (1 - .30 * (smooth(300, 1900, x) * .6 + .2)[..., None] * (1 - rgb('#b9a6e0'))); g_ = comp.radial(120, 450, 1700, 1000, 1.1)[..., None]; im = im * (1 - g_ * .85 * (1 - rgb('#ffc67e'))); im = comp.screen(im, comp.radial(150, 430, 1300, 820, 1.6)[..., None] * rgb('#ff9a2e') * .50); im = im + comp.radial(60, 450, 480, 400, 2.0)[..., None] * rgb('#ffd98a') * .42
        im = self.posts(im, f, False)
        lay = self.cached(('c', d), lambda: self.chars(d))
        # canvas -> frame
        M_ = np.float32([[z, 0, W / 2 - cx * z], [0, z, H / 2 - cy * z + 4]]); L_ = cv2.warpAffine(lay, M_, (W, H), flags=cv2.INTER_AREA if z < 1 else cv2.INTER_LINEAR)
        a = L_[..., 3:4]
        # shadow of the trees sliding over them: moves with the world, bent a little by the forms
        lum_ = cv2.GaussianBlur(L_[..., 1], (0, 0), 10); tx = x * .92 + 460 - f * 7.5 + lum_ * 60; ty = y * .92 + 420 - f * 13.5 + lum_ * 40; ls = cv2.remap(self.leaf, tx.astype(f32), ty.astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP); ls = smooth(.30, .70, ls) * (.25 + .75 * smooth(-.9, .4, nz((700, 400), (1, .5), 33))) * .40
        sk_ = cv2.GaussianBlur(smooth(.10, .24, L_[..., 0] - L_[..., 2]) * smooth(.50, .66, L_[..., 1]), (0, 0), 5); ls = ls * (1 - .92 * sk_)
        L_ = L_.copy(); L_[..., :3] *= 1 - ls[..., None] * (1 - rgb('#8f7fb0')) * .95
        sh = cv2.GaussianBlur(shifted(a[..., 0], 60, 40), (0, 0), 30); im = im * (1 - .16 * (sh * smooth(520, 760, y))[..., None])
        im = comp.screen(im, np.clip(comp.blur(a[..., 0], 18) - a[..., 0], 0, 1)[..., None] * rgb('#ffb04e') * .55); im = comp.over(im, L_); hb_ = np.clip(shifted(comp.blur(a[..., 0], 5), 7, 4) * 0 + comp.blur(np.clip(a[..., 0] - dog_shift_(a[..., 0], 9, 5), 0, 1), 9), 0, 1); im = im + hb_[..., None] * rgb('#ffab4a') * .42
        # Megumi's sigh: one drawn puff
        if 55 <= f < 68:
            t = (f - 55) // 2 * 2; p = np.array([1716 - 46 - t * 6.0, 520 + 118 + t * 1.0]); p = np.array([p[0] * z + M_[0, 2], p[1] * z + M_[1, 2]]); c = Cel(); r0 = 13 + t * 1.5; k = t / 12.
            lob = []
            for j, (dx_, dy_, rr) in enumerate(((0, 0, 1.), (-1.15, .12, .74), (-.55, .52, .62), (.45, -.42, .52), (-.65, -.50, .60), (1.15 - .4 * k, -.34, .30))):
                u = np.linspace(0, 6.283, 34)[:-1]; R_ = r0 * rr * (1 - .30 * k) * (1 + .05 * np.sin(u * 3 + j)); lob.append(np.c_[p[0] + dx_ * r0 * (1 + k * .5) + np.cos(u) * R_, p[1] + dy_ * r0 * (1 + k * .4) + np.sin(u) * R_ * .88])
            for q in lob: outline(c, q, 5.0 * (1 - .45 * k), '#7a5f66', 3)
            for q in lob: c.fill(q, '#fff8ec')
            c.fill(lob[1] * .7 + np.array([lob[1].mean(0)]) * .3 + [2, 6], '#ead8d2')
            im = comp.over(im, c.layer(), 1 - smooth(8, 13, f - 55) * .9)
        # foreground leaves pass the lens as the camera backs away, dust in the sun
        for (B_, x0, y0, t0, sx, v, fl) in ((self.near, -640, -520, -8, 1.5, 3.0, 1), (self.near2, 1640, -700, 22, 1.7, 3.6, -1)):
            t = f - t0
            if -6 < t < 44: s_ = sx * (1 - .009 * t); bx_ = x0 + t * v * 2.2 * fl; by_ = y0 + t * v * .6; Mb = np.float32([[s_ * fl, 0, bx_ + (1200 * s_ if fl < 0 else 0) - (400 if fl < 0 else 0)], [0, s_, by_]]); lf = cv2.warpAffine(B_, Mb, (W, H), flags=cv2.INTER_LINEAR); im = comp.over(im, lf, float(smooth(-6, 2, t) * (1 - smooth(34, 44, t))))
        im = self.posts(im, f, True)
        from jj import fx
        im = im + fx.motes(f, 70, 5, light=np.clip(comp.radial(480, 420, 1200, 800, 1.2), 0, 1), col='#ffe6b0', drift=(10, -4), amt=.55)
        im = comp.screen(im, (comp.radial(60, 300, 1300, 800, 2.2))[..., None] * rgb('#ff9428') * .30)
        return im
