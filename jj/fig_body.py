"""Costumed bodies for jj.figure: builds Clay primitives, fold strokes and material rules from a posed skeleton."""
import numpy as np, math
from .fig_sdf import Clay, table, ridge, bump, wrap, prof, TT, PH, NT, NP
from .fig_pose import Skel, BODY, CHAR_BODY, unit, D2R

PI = math.pi

# light, base, shadow, deep (blackest accents)
MATS = dict(
    skin=('#ffe9d2', '#f2c8a6', '#b97f6a', '#7d4a45'), skinf=('#fff0e2', '#f7d6bd', '#c99283', '#8a5550'), skint=('#f8d6ae', '#e2b488', '#a9764f', '#6d4530'), skinp=('#fff3ea', '#f4dccb', '#b99a9a', '#6f5560'),
    navy=('#4c5496', '#22253f', '#111222', '#07070d'), navyb=('#4f5c9e', '#242b46', '#121525', '#07080f'), gblack=('#5f4f9c', '#1f1c2e', '#0f0d18', '#06050a'),
    red=('#e8484a', '#c02a34', '#7a1626', '#450a18'), redshoe=('#d8474a', '#b02a34', '#6a1524', '#3c0a16'), sole=('#5a3038', '#3a1c24', '#22101a', '#120810'), solew=('#f4efe6', '#d8d0c6', '#9a9094', '#5c525a'),
    sock=('#34343e', '#1f1f28', '#121218', '#08080c'), ochre=('#b8934e', '#8d6b30', '#55401f', '#30230f'), brown=('#9a5a34', '#6f3a22', '#411f16', '#24100c'), loafer=('#c07038', '#9a5026', '#5e2c18', '#34170e'),
    tights=('#6a5850', '#43362f', '#271e1b', '#140e0d'), blouse=('#e2e6f2', '#aeb6cc', '#6f7893', '#474e66'), belt=('#6a4a30', '#3a261c', '#1f130f', '#100908'),
    cream=('#f3ece2', '#d8cfc6', '#9c9092', '#5f565e'), shirtb=('#5f96c4', '#2f6a9a', '#1b3f66', '#0f2440'), tie=('#dcc45a', '#b79a3c', '#6f5a24', '#3f3214'), shoeb=('#a85c3c', '#7a3a28', '#481f18', '#28100d'),
    black=('#3c3c48', '#17171d', '#0b0b0f', '#050507'), white=('#ffffff', '#efece6', '#b4b4c2', '#6f6f82'), bone=('#fffaf0', '#f1ece0', '#a9adba', '#646a7c'), slate=('#5c7696', '#34506e', '#1d2f46', '#101c2c'), band=('#3a4270', '#272c4e', '#15182e', '#0b0d1a'),
    obi=('#516b86', '#2c4560', '#182a3e', '#0d1826'), tabi=('#42464e', '#23262c', '#121418', '#08090b'), wrap=('#ffffff', '#ebe8e6', '#aaa8b4', '#6c6a78'), hakama=('#4a557e', '#2d3148', '#181a2a', '#0c0d16'),
    curse=('#6fc4b4', '#2f8a80', '#175049', '#0b2c2a'), cursed=('#8fb0c8', '#5a7a98', '#31465f', '#18253a'), gum=('#c84a5a', '#8a2438', '#4f0f22', '#2a0612'), tooth=('#fff6e0', '#ece0c4', '#b0a088', '#6a5c4c'), void=('#1a1016', '#0c070b', '#050305', '#020102'),
    grey=('#a8acb8', '#757a88', '#474b58', '#262933'), olive=('#8a8a66', '#5f6046', '#383a2b', '#1c1d15'), wine=('#a04a58', '#74303e', '#441a26', '#240c14'), denim=('#6f8cb8', '#456290', '#283c5c', '#142036'), camel=('#d8b080', '#b08654', '#6f5232', '#3c2b19'), hairk=('#4a4a60', '#1a1a24', '#0c0c12', '#060609'),
    char=('#2a2126', '#0f0b0f', '#080608', '#040304'), inside=('#1a1822', '#100f16', '#0a090e', '#050407'), metal=('#f4f6f8', '#b6bcc4', '#6a707a', '#383c44'), neckl=('#4a4046', '#2a2228', '#171216', '#0a080a'), gold=('#ffe890', '#d8a83a', '#8a6420', '#4c3610'))

SPEC = dict(
    yuji=dict(top='jacket', hem=.475, collar='hood', bottom='baggy', legcut=.105, shoe='hightop', skin='skin', cloth='navy', cuff='red', hair='yuji'),
    megumi=dict(top='jacket', hem=.455, collar='stand', bottom='straight', legcut=.10, shoe='ankle', skin='skinf', cloth='navyb', cuff=None, hair='megumi'),
    nobara=dict(top='bolero', hem=.50, collar='low', bottom='skirt', legcut=.30, shoe='loafer', skin='skinf', cloth='navyb', cuff=None, hair='nobara'),
    gojo=dict(top='jacket', hem=.475, collar='high', bottom='slim', legcut=.055, shoe='dress', skin='skinf', cloth='gblack', cuff=None, hair='gojo'),
    nanami=dict(top='suit', hem=.455, collar='lapel', bottom='suit', legcut=.05, shoe='dress', skin='skinf', cloth='cream', cuff='shirtb', hair='nanami'),
    todo=dict(top='bare', hem=.0, collar='none', bottom='balloon', legcut=.13, shoe='slipper', skin='skint', cloth='hakama', cuff=None, hair='todo'),
    sukuna=dict(top='kimono', hem=.0, collar='scarf', bottom='kimono', legcut=.07, shoe='zori', skin='skin', cloth='bone', cuff=None, hair='sukuna'),
    crowd=dict(top='jacket', hem=.46, collar='low', bottom='straight', legcut=.05, shoe='ankle', skin='skinf', cloth='grey', cuff=None, hair='crowd'),
    crowdf=dict(top='jacket', hem=.50, collar='low', bottom='skirt', legcut=.28, shoe='loafer', skin='skinf', cloth='wine', cuff=None, hair='crowdf'),
    curse=dict(top='flesh', hem=.0, collar='none', bottom='flesh', legcut=.0, shoe='bare', skin='curse', cloth='curse', cuff=None, hair='curse'),
)


def facet(seed, amp, nt=5, np_=7, fade=(0., 0.)):
    """piecewise-linear cloth irregularity: gives contours straights and corners instead of hose curves"""
    r = np.random.default_rng(seed); g = r.normal(0, 1, (nt, np_)); g = np.hstack([g, g[:, :1]])
    u = TT * (nt - 1); v = PH / (2 * PI) * np_; i = np.minimum(u.astype(int), nt - 2); j = np.minimum(v.astype(int), np_ - 1); a, b = u - i, v - j
    z = (g[i, j] * (1 - b) + g[i, j + 1] * b) * (1 - a) + (g[i + 1, j] * (1 - b) + g[i + 1, j + 1] * b) * a
    w = np.ones_like(TT)
    if fade[0] > 0: w = w * np.clip(TT / fade[0], 0, 1)
    if fade[1] > 0: w = w * np.clip((1 - TT) / fade[1], 0, 1)
    return amp * z * w


def tear(p, seed, n=7):
    """ragged 0..1 edge over phi (irregular triangles)"""
    r = np.random.default_rng(seed); k = np.sort(r.uniform(-PI, PI, n)); v = r.uniform(0, 1, n); v[::2] *= .25; kk = np.r_[k - 2 * PI, k, k + 2 * PI]; vv = np.r_[v, v, v]; return np.interp(wrap(p), kk, vv)


def drape(seed, amp, n=5, t=(.1, .9), skew=(.06, .16), w=(.045, .075), ln=(.7, 1.3), sign=None, p=None):
    """a family of angular drapery ridges that lean the same way (tension), sizes and spacing irregular"""
    r = np.random.default_rng(seed); z = 0; sg = sign if sign is not None else (1 if r.random() < .5 else -1)
    for i in range(n):
        t0 = t[0] + (t[1] - t[0]) * (i + r.uniform(.15, .85)) / n; p0 = r.uniform(-PI, PI) if p is None else r.uniform(*p)
        z = z + ridge(TT, PH, t0, p0, sg * r.uniform(*skew), r.uniform(*w), r.uniform(*ln), amp * r.uniform(.55, 1.15))
    return z


class Body:
    """posed, dressed clay + decals.  deco = list of dict(kind line|shard|poly, prim, t, p, w, mat)"""

    def __init__(self, char, sk, fx, px, spec=None, seed=0):
        self.c = Clay(); self.sk = sk; self.b = sk.b; self.fx = fx; self.px = px; self.deco = []; self.prim = {}; self.char = char
        self.sp = dict(SPEC[char]); self.sp.update(spec or {}); self.seed = seed
        self.H = sk.H; self.k = sk.H / 1.73; self.m = self.k * self.b['mass']
        self.lod = 0 if px < 150 else 1 if px < 330 else 2
        self.rng = np.random.default_rng(seed + 11)
        self.build()
        self.c.pack()

    # ---- helpers
    def line(self, prim, t, p, w=1., **k): self.deco.append(dict(kind='line', prim=prim, t=np.asarray(t, float), p=np.asarray(p, float), w=w, **k))

    def shard(self, prim, t, p, w=0., **k): self.deco.append(dict(kind='shard', prim=prim, t=np.asarray(t, float), p=np.asarray(p, float), w=w, **k))

    def blade(self, prim, a, b, w=.05, kink=.35, bow=0., pri=0, line=True, lw=1.0):
        """house-style light plane on dark cloth: an angular blade from a=(t, phi) (blunt, on the tension point) to b (needle).  w = width in t"""
        a, b = np.asarray(a, float), np.asarray(b, float); d = b - a; L = math.hypot(d[0] * 3.0, d[1]) + 1e-6; n = np.array([d[1] / 3.0, -d[0] * 3.0]) / L          # metric: 1 rad ~ 1/3 of t
        n = n / (abs(n[0]) + abs(n[1]) / 3 + 1e-6); m = a + d * kink + n * np.array([w, w * 3.0]) * .5 + n * np.array([bow, bow * 3]); q = a + d * .62 - n * np.array([w, w * 3.0]) * .34 + n * np.array([bow, bow * 3]) * .6
        c = a + d * .10 - n * np.array([w, w * 3.0]) * .42
        P = np.array([a, m, b, q, c]); self.shard(prim, P[:, 0], P[:, 1], pri=pri)
        if line: self.line(prim, [c[0], q[0], b[0]], [c[1], q[1], b[1]], lw, pri=pri)

    def build(self):
        sp = self.sp
        if sp['bottom'] == 'flesh': self.hulk(); return
        self.legs(); self.torso()
        for s in 'LR': self.arm(s); self.hand(s); self.foot(s)
        self.neck_(); self.collar()
        if sp['cloth'] in ('navy', 'navyb') and sp['top'] in ('jacket', 'bolero') and self.char in ('yuji', 'megumi', 'nobara'):
            x, z = dict(yuji=(.030, .060), megumi=(.030, .075), nobara=(.036, .030))[self.char]; a = np.linspace(0, 2 * PI, 11)[:-1]
            self.front = getattr(self, 'front', []) + [dict(pts=np.c_[x + .0105 * np.cos(a), z + .0105 * np.sin(a)], w=1., kind='badge', mat='gold')]
        sk = self.sk; u = sk.u; R = sk.Rh
        self.c.ball(sk.head + R @ np.array([0, -.03 * u, .02 * u]), (.70 * u, .99 * u, .80 * u), 'head', 'head', 'skin', ax=R[:, 2], e1=R[:, 0], k=.0)

    # -------------------------------------------------------------- torso
    def torso(self):
        sk, c, sp = self.sk, self.c, self.sp; H, m = self.H, self.m; b = self.b; top = sp['top']
        cw, ww, hp = b['chest'], b['waist'], b['hips']; mat = sp['cloth'] if top not in ('bare',) else sp['skin']
        fwd, rt, up = sk.Rc[:, 1], sk.Rc[:, 0], sk.Rc[:, 2]
        ease = dict(jacket=1.08, bolero=1.04, suit=1.10, bare=1.0, kimono=1.10)[top]
        n = dict(bare=2.35, jacket=2.6, suit=2.7, bolero=2.4, kimono=2.5)[top]
        a = sk.chest - up * .075 * H; e = sk.neck - up * .004 * H
        fem = b['sw'] < .095
        S = [(0, .098 * ww, .124 * ww, .088 * ww, .124 * ww), (.25, .108 * cw, .138 * cw, .100 * cw, .138 * cw), (.5, .121 * cw, .156 * cw, .110 * cw, .156 * cw), (.70, .112 * cw, .158 * cw, .110 * cw, .158 * cw), (.86, .080, .112, .092, .112), (1, .056, .068, .064, .068)]
        Rt = table(S, n) * m * ease + ((facet(21, .0045 * m, 5, 7, (.1, .3)) + (drape(121, .0075 * m, 5, (.05, .75), sign=1 if self.twist() > 0 else -1) if self.lod > 1 else 0)) if top != 'bare' else 0)
        tw = abs(wrap((self.twist()) * D2R))
        rt_i = c.loft(a, e, fwd, Rt, 'torso', 'torso', mat, cap=(1.4, 1.3), k=.07, side=-1, name='ribs')
        self.prim['ribs'] = rt_i
        # belly
        a2 = sk.pel + sk.Rp[:, 2] * (-.005 if top != 'bare' else .030) * H; e2 = sk.chest + up * .02 * H
        fb = unit(sk.Rp[:, 1] + fwd)
        tb = table([(0, .102 * hp, .140 * hp, .108 * hp, .140 * hp), (.3, .097, .130 * ww, .094, .130 * ww), (.55, .096, .124 * ww, .088, .124 * ww), (1, .110, .143 * cw, .100, .143 * cw)], n) * m * ease + ((facet(22, .004 * m, 4, 7) + (drape(122, .0085 * m, 4, (.15, .85), sign=1 if self.twist() > 0 else -1) if self.lod > 1 else 0)) if top != 'bare' else 0)
        if top in ('jacket', 'suit', 'kimono') and self.lod:                       # side-bend and bow compress cloth at the waist
            pit = math.degrees(math.acos(float(np.clip(sk.Rc[:, 2] @ sk.Rp[:, 2], -1, 1))))
            am = .016 * m * min(1, pit / 35)
            if am > .002:
                d = sk.Rp[:, 2] - sk.Rc[:, 2] * (sk.Rp[:, 2] @ sk.Rc[:, 2]); ph0 = math.atan2(-(sk.Rc[:, 2] @ sk.Rp[:, 0]), sk.Rc[:, 2] @ sk.Rp[:, 1])
                for q, t0 in enumerate((.42, .58, .74)): tb = tb + ridge(TT, PH, t0, ph0 + .25 * (-1) ** q, .05 * (-1) ** q, .07, 1.2, am)
        self.prim['belly'] = c.loft(a2, e2, fb, tb, 'torso', 'torso', mat, cap=(1.2, 1.2), k=.09, side=-1, name='belly')
        # shoulder girdle: trapezius slopes
        for s, sg in (('L', -1), ('R', 1)):
            q0 = sk.neck + up * .004 * H - fwd * .012 * H + rt * sg * .018 * H; q1 = sk.sh[s] + up * .004 * H - rt * sg * .006 * H
            r0, r1 = (.046, .040) if top != 'bare' else (.058, .052)
            c.loft(q0, q1, up, table([(0, r0 * .8, r0, r0 * 1.2, r0), (1, r1, r1 * 1.15, r1, r1 * 1.15)], 2) * m, 'torso', 'torso', mat, cap=(1, 1), k=.06, name='trap' + s)
            # lat / pec sheet from the rib cage to the arm: keeps the armpit full when the arm is raised
            q2 = sk.sh[s] + (sk.el[s] - sk.sh[s]) * .18; q3 = sk.chest + rt * sg * .115 * m * cw + up * .012 * H
            c.loft(q3, q2, fwd, table([(0, .075, .05, .075, .05), (1, .048, .045, .048, .045)], 2) * m, 'torso', 'torso', mat, cap=(1, 1), k=.06, name='lat' + s)
        if fem:
            for sg in (-1, 1): c.ball(sk.chest + rt * sg * .056 * m + up * .035 * H + fwd * .082 * m, (.060 * m, .056 * m, .052 * m), 'torso', 'torso', mat, ax=up, e1=rt, k=.06)
        if top == 'bare': self.muscle()
        if top == 'bolero':
            H_ = H; w0 = sk.waist
            zone = lambda t, p, q: ((q - w0) @ sk.Rc[:, 2]) < .030 * H_
            bz = lambda t, p, q: ((q - w0) @ sk.Rc[:, 2]) < .012 * H_
            for nm in ('ribs', 'belly'): self.c.rule[self.prim[nm]] = [(zone, 'blouse')]
        if top in ('jacket', 'suit', 'bolero'): self.jacket_skirt(mat, ease)
        if top == 'kimono': self.kimono_top()
        if self.lod: self.torso_folds()

    def twist(self):
        sk = self.sk; a = sk.Rp[:, 1]; b = sk.Rc[:, 1]; u = sk.Rp[:, 2]
        a = unit(a - u * (a @ u)); b = unit(b - u * (b @ u)); return math.degrees(math.atan2(np.cross(a, b) @ u, a @ b))

    def hull(self, o, ax, e1, e2, L, segs, base, pad=.0, balls=(), n=16, web=.5, smooth=0):
        """cloth tube round the body: R(phi) on every ring must enclose the given tapered segments [(p0, p1, r0, r1)] and balls [(c, r)].
        web 0..1 = how far cloth bridges the gap between two legs (support hull vs star hull)"""
        C, r = [], []
        for p0, p1, r0, r1 in segs:
            u = np.linspace(0, 1, n)[:, None]; C.append(p0 * (1 - u) + p1 * u); r.append((r0 * (1 - u) + r1 * u)[:, 0])
        for c_, r_ in balls: C.append(np.asarray(c_, float)[None]); r.append(np.array([r_]))
        C = np.vstack(C) - o; r = np.concatenate(r) + pad
        y = C @ ax; x, z = C @ e1, C @ e2; rad = np.hypot(x, z); ph = np.arctan2(z, x)
        h = y[None] - (L * TT[:, 0])[:, None]                                     # (NT, S)
        q2 = r[None] ** 2 - h ** 2; q = np.sqrt(np.maximum(q2, 0)) * (q2 > 0)     # circle radius on each ring
        dl = wrap(PH[0][None] - ph[:, None])                                      # (S, NP)
        c_, s_ = np.cos(dl), np.sin(dl)
        rc = rad[:, None] * c_; disc = q[:, :, None] ** 2 - (rad[:, None] * s_)[None] ** 2
        star = np.where((disc > 0) & (q[:, :, None] > 0), rc[None] + np.sqrt(np.maximum(disc, 0)), 0.)
        sup = np.where((q[:, :, None] > 0) & (c_[None] > 0), rc[None] + q[:, :, None], 0.)
        out = np.maximum(base, np.maximum(star.max(1), web * sup.max(1) + (1 - web) * star.max(1)))
        for _ in range(smooth):                                                     # cloth bridges bumps: relax, never below the body
            q_ = np.vstack([out[:1], out, out[-1:]]); q_ = .25 * q_[:-2] + .5 * q_[1:-1] + .25 * q_[2:]; q_ = .25 * np.roll(q_, 1, 1) + .5 * q_ + .25 * np.roll(q_, -1, 1)
            out = np.maximum(q_, star.max(1))
        return out

    def legsegs(self, s, f=1.0):
        sk, m = self.sk, self.m * self.b['limb']
        return [(sk.hip[s], sk.kn[s], .086 * m * f, .072 * m * f), (sk.kn[s], sk.an[s], .072 * m * f, .060 * m * f)]

    def jacket_skirt(self, mat, ease):
        sk, c, sp, H, m = self.sk, self.c, self.sp, self.H, self.m; hp, ww = self.b['hips'], self.b['waist']
        up, fwd, rt = sk.Rp[:, 2], sk.Rp[:, 1], sk.Rp[:, 0]
        if sp['top'] == 'bolero':
            a = sk.waist + sk.Rc[:, 2] * .075 * H; e = a - sk.Rc[:, 2] * .048 * H
            R = table([(0, .112, .134, .098, .134), (.5, .126, .142, .108, .142), (1, .138, .148, .118, .148)], 2.3) * m + .014 * m
            self.prim['hem'] = c.loft(a, e, sk.Rc[:, 1], R, 'jskirt', 'jacket', mat, cap=(0, 0), k=.02, name='hem', rule=self.r_cap('inside'))
            # blouse under it, belt on the waist
            a = sk.waist + sk.Rw[:, 2] * .02 * H; e = sk.waist - sk.Rp[:, 2] * .018 * H
            self.prim['belt'] = c.loft(a, e, sk.Rw[:, 1], table([(0, .104, .134 * ww, .098, .134 * ww), (1, .108, .142 * ww, .104, .142 * ww)], 2.3) * m * 1.06 + .020 * m, 'belt', 'belt', 'belt', cap=(0, 0), k=0, name='belt',
                                       rule=[(lambda t, p, q: np.abs(wrap(p)) < .22, 'gold')])
            return
        a = sk.waist + sk.Rw[:, 2] * .012 * H; hemz = sp['hem'] * H; L = (self.b['waistz'] * H - hemz)
        lag = self.fx.get('hem', np.zeros(3))
        mt = unit(unit(sk.kn['L'] - sk.hip['L']) + unit(sk.kn['R'] - sk.hip['R']) + up * -.4); e = a + unit(-up + .38 * mt) * L + lag
        ax = unit(e - a); e1 = unit(fwd - ax * (fwd @ ax)); e2 = -np.cross(ax, e1)
        LL = np.linalg.norm(e - a)
        k = dict(jacket=1.0, suit=1.02)[sp['top']]
        base = table([(0, .090 * ww, .120 * ww, .084 * ww, .120 * ww), (.25, .098, .136, .100, .136), (.6, .112 * hp, .157 * hp, .120 * hp, .157 * hp), (1, .120 * hp, .164 * hp, .126 * hp, .164 * hp)], 2.4) * m * ease * k
        gl = [(sk.hip[q] - fwd * .038 * m + up * .004 * H, .080 * m * hp) for q in 'LR']
        R = np.minimum(self.hull(a, ax, e1, e2, LL, self.legsegs('L')[:1] + self.legsegs('R')[:1], base, .014, gl), base * (1.06 + .20 * TT) + .006)
        # hem ripple
        rng = np.random.default_rng(5); w = sum(rng.uniform(.4, 1) * np.cos(q * PH + rng.uniform(0, 6)) for q in (3, 5, 7))
        R = R + w * .0055 * m * np.clip((TT - .45) / .55, 0, 1) + facet(23, .005 * m, 3, 7, (.4, 0))
        self.prim['hem'] = c.loft(a, e, e1, R, 'torso', 'torso', mat, cap=(1.4, 0), k=.035, side=-1, name='hem', rule=self.r_cap('inside') + ([(lambda t, p, q: (t > .74 + .26 * tear(p, 87, 15)) & (t < 1.0005), 'char')] if sp.get('burnt') else []))

    def r_cap(self, mat):
        return [(lambda t, p, q: t > 1.0005, mat)]

    def torso_folds(self):
        """authored folds on jacket torso: armpit pulls, waist compression, twist diagonals, hem"""
        top = self.sp['top']; sk = self.sk
        if top == 'bare': return
        rb, bl = self.prim['ribs'], self.prim['belly']
        tw = self.twist(); B = self.blade
        for s, sg in (('L', -1), ('R', 1)):
            up = min(1.2, sk.lift[s]); k = .75 + .55 * up
            if top in ('jacket', 'bolero'):                                   # yoke: the plane that catches light on top of the chest, broken in two
                if self.sp['collar'] in ('high', 'stand'): B(rb, (.885, sg * 1.02), (.665, sg * .14), .085, .30, .0, 0, False); B(rb, (.80, sg * 1.18), (.69, sg * .62), .05, .4, 0, 1, False)
                else: B(rb, (.875, sg * 1.06), (.70, sg * .50), .075, .3, 0., 0, False)
            # pulls from the shoulder / armpit: they lengthen and steepen as the arm goes up
            p0 = sg * 1.24; t0 = .70 + .05 * up
            B(rb, (t0, p0), (t0 - .17 * k, p0 - sg * .95 * k), .060, .28, 0., 0, True, 1.1)
            B(rb, (t0 - .10, p0 + sg * .02), (t0 - .40 * k, p0 - sg * .52 * k), .055, .30, 0., 0, True, 1.0)
            B(rb, (t0 + .06, p0 - sg * .04), (t0 + .02, p0 - sg * .80 * k), .040, .35, 0., 1, True, .9)
            # back: shoulder blade pulls toward the spine
            p1 = sg * 1.95
            B(rb, (t0 + .03, p1), (t0 - .20, p1 + sg * .80), .060, .28, 0., 0, True, 1.0); B(rb, (t0 - .08, p1 + sg * .05), (t0 - .36, p1 + sg * .42), .05, .3, 0., 1, True, .9); B(rb, (.90, sg * 2.2), (.72, sg * 2.55), .07, .35, 0., 1, False)
        # waist: compression folds that wedge into each other from both sides, biggest on the concave side
        d = sk.Rc[:, 2]; hx, hy = d @ sk.Rp[:, 0], d @ sk.Rp[:, 1]; bk = math.hypot(hx, hy); ph0 = math.atan2(hx, hy) if bk > .14 else 1.0; g = .8 + .8 * min(1., bk / .5)
        for q, (t0, side, dt, ln, w) in enumerate(((.90, 1, -.06, .95, .075), (.74, -1, -.10, 1.15, .085), (.57, 1, -.10, 1.25, .090), (.42, -1, -.06, 1.0, .075), (.28, 1, .06, .85, .065), (.66, 1, .10, .6, .05))):
            p0 = ph0 + side * (.95 if bk > .14 else 1.05); ln *= g
            B(bl, (t0, p0), (t0 + dt, p0 - side * ln), w * (.8 + .4 * g), .30, 0., 0 if q < 4 else 1, True, 1.1 if q < 3 else .9)
            B(bl, (t0 - .03, p0 + PI), (t0 + dt * .6, p0 + PI + side * ln * .8), w * .8, .30, 0., 1, True, .9)
        if abs(tw) > 10:                                                      # twist: long diagonals from hip to opposite shoulder
            sgn = 1 if tw > 0 else -1; kk = min(1.3, abs(tw) / 38.)
            for q, (t0, p0) in enumerate(((.06, -.75), (.26, -.25), (.04, .28), (.40, .55))): B(rb, (t0, p0 * sgn), (t0 + .30 * kk + .06, (p0 + .78 * kk) * sgn), .060, .34, 0., 0 if q < 3 else 1, True, 1.05)
            for q, (t0, p0) in enumerate(((.10, 2.0), (.30, 2.6))): B(rb, (t0, p0 * sgn), (t0 + .32, (p0 + .7) * sgn), .055, .34, 0., 1, True, .9)
        if 'hem' in self.prim:
            h = self.prim['hem']
            for q, (p0, t0) in enumerate(((.62, .55), (-.8, .45), (1.7, .5), (2.6, .55), (-2.3, .5), (-1.55, .6))):
                self.line(h, [t0, .8, .99], [p0 + .10, p0 + .02, p0], .85, pri=1 + (q > 2))
                self.shard(h, [t0 + .05, .85, .985, .985, .82], [p0 + .10, p0 + .06, p0 + .02, p0 + .13, p0 + .14], pri=1 + (q > 1))
        if top == 'jacket' and self.sp['collar'] in ('high', 'stand'):
            off = 0. if self.sp['collar'] == 'high' else .28
            self.line(rb, [.99, .6, .2, .0], [off, off * .9, off * .8, off * .8], 1.0, pri=0, straight=True)
            self.line(bl, [.55, .2, .0], [off * .8, off * .8, off * .8], 1.0, pri=0, straight=True)
            self.line(self.prim['hem'], [.1, .5, 1.], [off * .75] * 3, 1.0, pri=0, straight=True)

    def muscle(self):
        """bare chest: pecs, abs block, serratus, delts come from the arm"""
        sk, c, m = self.sk, self.c, self.m; H = self.H; up, fwd, rt = sk.Rc[:, 2], sk.Rc[:, 1], sk.Rc[:, 0]; mat = self.sp['skin']
        for s, sg in (('L', -1), ('R', 1)):
            pc = sk.chest + rt * sg * .078 * m + up * .052 * H + fwd * .080 * m
            c.ball(pc, (.092 * m, .064 * m, .050 * m), 'torso', 'torso', mat, ax=unit(up + rt * sg * .35), e1=unit(rt * sg - up * .25), k=.022, name='pec')
            for q, (z, x) in enumerate(((.005, .046), (-.034, .045), (-.072, .040))):
                c.ball(sk.chest + sk.Rw[:, 2] * z * H + sk.Rw[:, 0] * sg * x * m + sk.Rw[:, 1] * (.088 - .004 * q) * m, (.040 * m, .034 * m, .026 * m), 'torso', 'torso', mat, ax=sk.Rw[:, 2], e1=sk.Rw[:, 0], k=.016)
            c.ball(sk.chest - up * .02 * H + rt * sg * .13 * m - fwd * .01, (.045 * m, .085 * m, .07 * m), 'torso', 'torso', mat, ax=up, e1=rt, k=.03)
        rb, bl = self.prim['ribs'], self.prim['belly']
        if self.lod:
            for sg in (-1, 1):
                self.line(rb, [.78, .62, .52, .50], [sg * .08, sg * .06, sg * .10, sg * .34], 1.1, pri=0, mat='skinln')
                self.line(rb, [.50, .50, .57], [sg * .30, sg * .62, sg * 1.02], 1.0, pri=0, mat='skinln')
                self.line(rb, [.36, .28, .22], [sg * 1.0, sg * 1.15, sg * 1.05], .8, pri=1, mat='skinln')
                self.line(rb, [.28, .20, .14], [sg * 1.05, sg * 1.2, sg * 1.1], .8, pri=1, mat='skinln')
                self.line(bl, [.8, .6, .4, .2], [sg * .52, sg * .55, sg * .56, sg * .45], .8, pri=1, mat='skinln')
            self.line(bl, [.95, .6, .3, .1], [0, .01, 0, 0], .9, pri=0, mat='skinln')
            for t0 in (.78, .58, .38): self.line(bl, [t0, t0 - .02, t0], [-.42, 0, .42], .75, pri=1, mat='skinln')

    # -------------------------------------------------------------- arms
    def arm(self, s):
        sk, c, sp, m = self.sk, self.c, self.sp, self.m * self.b['limb']; sg = -1 if s == 'L' else 1
        bare = sp['top'] == 'bare'; mat = sp['skin'] if bare else sp['cloth']
        sh, el, wr = sk.sh[s], sk.el[s], sk.wr[s]; e1 = sk.e1a[s]; be = sk.bend[s]; f = min(1., be / 1.6)
        ua = unit(el - sh); fa = unit(wr - el); G = 'arm' + s
        c.group(G, sh, .22 * self.m, .07 * self.m)
        if sp['top'] == 'kimono': bare = True; mat = sp['skin']
        if bare:
            tu = table([(0, .050, .052, .046, .046), (.2, .054, .054, .048, .046), (.45, .054, .047, .054 + .012 * f, .045), (.7, .046, .042, .050 + .010 * f, .041), (1, .041, .040, .041, .039)], 2.1) * m
            tf = table([(0, .046, .046, .046, .043), (.22, .048, .052, .050, .046), (.5, .040, .044, .043, .040), (.8, .029, .035, .031, .033), (1, .026, .035, .028, .033)], 2.1) * m
            if sp['top'] == 'kimono': tu *= .78; tf *= .96
            else:
                tu = table([(0, .050, .054, .046, .046), (.18, .058, .060, .048, .046), (.38, .064, .050, .056 + .010 * f, .047), (.58, .056, .044, .062 + .014 * f, .045), (.80, .044, .041, .050, .041), (1, .041, .042, .042, .042)], 2.15) * m
                tf = table([(0, .044, .046, .046, .044), (.18, .050, .060, .054, .052), (.42, .044, .052, .046, .046), (.75, .030, .036, .032, .034), (1, .026, .036, .028, .034)], 2.2) * m
        else:
            lo = 1.0 if sp['top'] != 'suit' else 1.03
            tu = table([(0, .052, .054, .052, .050), (.3, .054, .054, .054, .050), (.7, .049, .048, .052, .046), (1, .046, .045, .047, .044)], 2.2) * m * lo
            tf = table([(0, .047, .046, .048, .044), (.22, .050, .049, .052, .047), (.7, .045, .043, .047, .042), (1, .043, .041, .046, .041)], 2.2) * m * lo
            if self.lod:
                q = .015 * m * f + .004 * m; fs = 31 + (s == 'R') * 7
                tu = tu + facet(fs, .0042 * m, 5, 6, (.25, .0)); tf = tf + facet(fs + 1, .0042 * m, 5, 6, (0, .12))
                if self.lod > 1: tu = tu + drape(fs + 100, .0055 * m, 4, (.15, .75), sign=sg); tf = tf + drape(fs + 101, .006 * m, 4, (.35, .97), sign=-sg)
                # inner-elbow compression: angular ridges that fan out from the crease
                for t0, p0, sk_, a in ((.93, PI, .05, 1.), (.80, PI + .5, -.10, .8), (.68, PI - .5, .12, .55)): tu = tu + ridge(TT, PH, t0, p0, sk_, .075, 1.5, q * a)
                for t0, p0, sk_, a in ((.08, PI, -.05, 1.), (.22, PI - .5, .10, .75)): tf = tf + ridge(TT, PH, t0, p0, sk_, .08, 1.5, q * a)
                # sleeve bunching above the cuff + shoulder cap folds
                for t0, p0, sk_ in ((.80, .6, .05), (.90, -1.4, -.05)): tf = tf + ridge(TT, PH, t0, p0, sk_, .06, 1.9, .0075 * m)
                l = sk.lift[s]
                tu = tu + ridge(TT, PH, .10, PI, .10, .10, 1.2, .010 * m * min(1, l)) + ridge(TT, PH, .24, PI + .4, -.12, .09, 1.2, .006 * m)
        k = .022 if not bare else .03
        self.prim['ua' + s] = c.loft(sh, el, e1, tu, G, 'uarm' + s, mat, cap=(1.0, 1.15), k=k, side=sg, name='ua' + s)
        if sp['top'] != 'kimono':
            fr = [] if bare else [(lambda t, p, q: t > 1.0005, 'inside')]
            if sp['cuff'] and not bare: fr.append((lambda t, p, q: (t > .945) & (t <= 1.0005), sp['cuff']))
        else: fr = [(lambda t, p, q: ((t > .80) & (t < .84)) | ((t > .88) & (t < .95)), 'black')]
        bt = float(sp.get('burnt', 0))
        if bt and not bare:
            c0, am = (.46, .26) if s == 'L' else (.76, .18); sd_ = 81 + (s == 'R')
            cut = lambda p: c0 + am * tear(p, sd_, 9)
            bt_ = table([(0, .044, .044, .046, .042), (.22, .046, .050, .048, .044), (.5, .038, .042, .041, .038), (.8, .028, .034, .030, .032), (1, .025, .034, .027, .032)], 2.1) * m
            tf = np.where(TT > cut(PH), bt_, tf)
            fr = [(lambda t, p, q: (t > cut(p) - .07) & (t <= cut(p)), 'char'), (lambda t, p, q: t > cut(p), sp['skin'])]
        wend = wr - fa * (.0 if bare else .004)
        self.prim['fa' + s] = c.loft(el, wend, e1, tf, G, 'farm' + s, mat, cap=(1.05, 0 if not bare and not bt else 1.6), k=k, side=sg, name='fa' + s, rule=fr)
        if bt and not bare:
            hole = lambda t, p, q, o=1.: ((t - .52) / (.13 * o)) ** 2 + (wrap(p - (1.1 if s == 'L' else -1.3)) / (.55 * o)) ** 2 + .55 * (tear(p * 3 + t * 9, 91) - .4)
            c.rule[self.prim['ua' + s]] = [(lambda t, p, q: hole(t, p, q) < 1.15, 'char'), (lambda t, p, q: hole(t, p, q) < .62, sp['skin'])]
        # deltoid: rides half the arm's swing, gives the shoulder its corner
        dm = 1.0 if not bare else 1.10
        if sp['top'] != 'kimono': c.ball(sh + ua * .034 * m + e1 * .004, (.053 * m * dm, .078 * m * dm, .053 * m * dm), G, 'uarm' + s, mat, ax=ua, e1=e1, k=.03, name='delt' + s)
        # olecranon: the elbow is a corner, not a hose
        if sp['top'] != 'kimono': c.ball(el + e1 * .016 * m - fa * .006 * m, np.array([.036, .036, .034]) * m * (1. if not bare else .92), G, 'farm' + s, mat, ax=fa, e1=e1, k=.028, name='elb' + s)
        if bare and self.lod and sp['top'] == 'bare':
            self.line(self.prim['ua' + s], [.14, .34, .46], [1.5, 1.9, 2.4], .9, pri=0, mat='skinln'); self.line(self.prim['ua' + s], [.55, .75, .9], [1.9, 1.75, 1.6], .8, pri=1, mat='skinln')
            self.line(self.prim['fa' + s], [.08, .3, .5], [1.2, 1.6, 1.5], .8, pri=1, mat='skinln'); self.line(self.prim['ua' + s], [.12, .3, .42], [-1.4, -1.8, -2.3], .8, pri=1, mat='skinln')
        if not bare and self.lod: self.sleeve_folds(s, f)

    def sleeve_folds(self, s, f):
        u, a = self.prim['ua' + s], self.prim['fa' + s]; L = self.line; Sh = self.shard
        # creases leave the elbow pit and wrap round the form, lengths differ
        k = .45 + .55 * f
        L(u, [.97, .90, .80], [PI - .3, PI - 1.05 * k, PI - 1.75 * k], 1.0, pri=0); L(u, [.93, .80, .70], [PI + .35, PI + .9 * k, PI + 1.35 * k], .85, pri=1)
        L(u, [.80, .70, .55], [PI - .1, PI - .6, PI - .95], .75, pri=2)
        L(a, [.05, .14, .28], [PI + .2, PI + .9 * k, PI + 1.6 * k], 1.0, pri=0); L(a, [.10, .22, .34], [PI - .3, PI - .8 * k, PI - 1.2 * k], .8, pri=1)
        Sh(u, [.965, .88, .79, .86], [PI - .55, PI - 1.15 * k, PI - 1.85 * k, PI - .95 * k], pri=0); Sh(u, [.90, .76, .67, .80], [PI + .5, PI + .95 * k, PI + 1.4 * k, PI + .75 * k], pri=1)
        Sh(a, [.06, .16, .30, .12], [PI + .45, PI + 1.0 * k, PI + 1.7 * k, PI + .85 * k], pri=0); Sh(a, [.14, .26, .37, .20], [PI - .5, PI - .9 * k, PI - 1.25 * k, PI - .7 * k], pri=1)
        B = self.blade; k2 = .6 + .5 * f
        B(u, (.20, 1.35), (.78, 1.15), .10, .35, 0., 0, False); B(u, (.96, PI - .5), (.66, PI - 1.9 * k2), .11, .30, 0., 0, False); B(u, (.90, PI + .4), (.60, PI + 1.5 * k2), .09, .30, 0., 1, False)
        B(a, (.06, PI + .4), (.42, PI + 1.9 * k2), .11, .30, 0., 0, False); B(a, (.10, 1.2), (.70, 1.0), .09, .40, 0., 1, False); B(a, (.95, -.5), (.72, -1.7), .08, .3, 0., 1, False)
        # cuff bunching
        L(a, [.80, .77, .80], [.2, .9, 1.6], .75, pri=1); L(a, [.90, .87, .89], [-.6, -1.2, -1.9], .75, pri=2); Sh(a, [.81, .785, .80, .83], [.35, .9, 1.5, .95], pri=2)
        # shoulder: two drops from the cap
        l = min(1., self.sk.lift[s])
        L(u, [.10, .22, .36], [PI + .5, PI + .95, PI + 1.2], .85, pri=1); L(u, [.06, .16, .22], [PI - .4, PI - .9, PI - 1.3], .85, pri=1)
        Sh(u, [.12, .24, .40, .26], [PI + .65, PI + 1.1, PI + 1.35, PI + 1.0], pri=1)
        # back of the elbow: pull ticks when bent
        if f > .4: L(u, [.98, .93, .86], [.5, .95, 1.3], .75, pri=2); L(a, [.03, .10, .18], [-.5, -.9, -1.2], .75, pri=2)

    # -------------------------------------------------------------- hands
    HAND = dict(  # per finger (mcp, pip, dip) index..pinky, spread; thumb joints p1 p2 p3 (thumb side, finger dir, back) in hand lengths
        relax=([(20, 26, 14), (28, 36, 18), (34, 42, 20), (38, 46, 22)], 2, ((.25, .30, -.13), (.27, .45, -.20), (.24, .57, -.24))), fist=([(84, 100, 62), (87, 100, 62), (89, 100, 62), (92, 98, 60)], 0, ((.24, .30, -.15), (.155, .405, -.262), (.0, .405, -.268))),
        open=([(2, 8, 4), (0, 6, 4), (2, 8, 4), (6, 10, 6)], 8, ((.33, .27, -.07), (.46, .39, -.08), (.56, .49, -.07))), flat=([(2, 2, 2)] * 4, 1, ((.25, .30, -.09), (.275, .45, -.10), (.265, .57, -.09))),
        claw=([(-12, 62, 40), (-8, 60, 40), (-8, 62, 40), (-4, 62, 42)], 10, ((.32, .26, -.14), (.42, .39, -.22), (.38, .51, -.29))), point=([(0, 4, 2), (84, 98, 58), (86, 100, 58), (88, 100, 58)], 1, ((.24, .30, -.15), (.14, .40, -.262), (-.02, .39, -.268))),
        two=([(2, 5, 2), (-4, 4, 2), (84, 98, 58), (86, 100, 58)], -2, ((.23, .29, -.15), (.10, .38, -.262), (-.06, .37, -.268))), hold=([(58, 70, 36), (60, 72, 36), (62, 74, 36), (64, 76, 36)], 0, ((.25, .30, -.16), (.20, .42, -.30), (.07, .45, -.36))),
        splay=([(-8, 4, 2), (-10, 2, 2), (-8, 4, 2), (-4, 8, 4)], 14, ((.36, .24, -.05), (.52, .33, -.05), (.64, .41, -.03))), soft=([(8, 18, 10), (14, 22, 12), (18, 26, 12), (20, 30, 14)], 4, ((.29, .29, -.11), (.36, .43, -.16), (.37, .55, -.18))))

    def hand(self, s):
        sk, c = self.sk, self.c; hl = self.b['hand'] * self.H * float(self.fx.get('handscale' + s, 1.)); R = sk.Rhand[s]; th, fd, bk = R[:, 0], R[:, 1], R[:, 2]; o = sk.wr[s]; skin = self.sp['skin']
        g = sk.grip[s]; sh = self.HAND.get(g, self.HAND['relax']); G = 'hand' + s; pxh = self.px / self.H * hl * .9
        pal = table([(0, .12, .155, .12, .155), (.35, .11, .20, .12, .19), (.8, .085, .225, .10, .215), (1, .07, .215, .085, .205)], 2.7) * hl * .98
        e = o + fd * hl * .50
        # wrist stub joins palm and forearm
        c.loft(o - fd * hl * .12, e, bk, pal, G, G, skin, cap=(1.2, 2.6), k=.012, side=1., name='palm' + s, cast=pxh > 26)
        simple = pxh < 15
        xs = (.165, .055, -.057, -.160); ln = (.40, .44, .41, .33); rr = (.056, .058, .055, .048)
        fing, spr, thb = sh
        for q in range(4):
            a1, a2, a3 = (v * D2R for v in fing[q]); sp_ = (1.5 - q) * spr * D2R / 1.5
            if simple and q in (1, 2): pass
            b0 = o + fd * hl * (.485 - .03 * abs(q - 1.2)) + th * xs[q] * hl + bk * .015 * hl
            fdir = unit(fd * math.cos(sp_) + th * math.sin(sp_)); p = b0; A = 0.
            lab = G if pxh < 34 else G + 'f%d' % q
            for j, (an, lf) in enumerate(zip((a1, a2, a3), (.44, .31, .25))):
                A += an; d = fdir * math.cos(A) - bk * math.sin(A); e_ = p + d * ln[q] * hl * lf
                r0 = rr[q] * hl * (1 - .10 * j); r1 = r0 * .88
                nrm = fdir * math.sin(A) + bk * math.cos(A)
                c.loft(p, e_, nrm, table([(0, r0, r0 * 1.02, r0, r0 * 1.02), (.5, r0 * .92, r0 * .95, r0, r0 * .95), (1, r1 * 1.02, r1 * 1.02, r1, r1 * 1.02)], 2.1), lab if not simple else G, lab, skin, cap=(1, 1), k=.006, name=G + 'f%d%d' % (q, j), cast=False)
                p = e_
        # thumb
        b0 = o + fd * hl * .12 + th * .13 * hl - bk * .05 * hl
        p1, p2, p3 = (o + th * v[0] * hl + fd * v[1] * hl + bk * v[2] * hl for v in thb)
        lab = G if pxh < 34 else G + 'th'
        for a, e_, r0, r1 in ((b0, p1, .088, .064), (p1, p2, .064, .058), (p2, p3, .058, .042)):
            c.loft(a, e_, bk, table([(0, r0, r0, r0, r0), (1, r1, r1, r1, r1)], 2) * hl, lab if not simple else G, lab, skin, cap=(1, 1), k=.014, name=G + 'th', cast=False)

        if self.sp.get('prop' + s) == 'hammer' or (self.char == 'nobara' and g == 'hold' and s == 'R'): self.hammer(o, th, fd, bk, hl, s)

    def hammer(self, o, th, fd, bk, hl, s):
        c = self.c; a0 = o + fd * hl * .50 - bk * hl * .20 - th * hl * .42; ax = unit(th + fd * .18); L = hl * 2.0; e = a0 + ax * L; r = .075 * hl
        c.loft(a0, e, fd, table([(0, r * 1.15, r * 1.15, r * 1.15, r * 1.15), (.35, r, r, r, r), (.4, r * .7, r * .7, r * .7, r * .7), (1, r * .75, r * .75, r * .75, r * .75)], 2.2), 'ham' + s, 'hamh' + s, 'black', cap=(1, 1), k=0)
        h0 = e - fd * hl * .28; h1 = e + fd * hl * .24; q = hl * .15
        c.loft(h0, h1, ax, table([(0, q, q, q, q), (.22, q * 1.02, q * 1.02, q * 1.02, q * 1.02), (.30, q * .72, q * .72, q * .72, q * .72), (1, q * .82, q * .82, q * .82, q * .82)], 3.), 'ham' + s, 'ham' + s, 'metal', cap=(0, 0), k=.01)
        c.loft(h1 - fd * hl * .02, h1 + fd * hl * .42 - ax * hl * .20, ax, table([(0, q * .8, q * .7, q * .8, q * .7), (1, q * .10, q * .45, q * .10, q * .45)], 2.4), 'ham' + s, 'ham' + s, 'metal', cap=(1, 1), k=.02)

    # -------------------------------------------------------------- legs
    def legs(self):
        sk, c, sp, H, m = self.sk, self.c, self.sp, self.H, self.m; b = self.b; st = sp['bottom']; lm = b['limb'] * m; hp = b['hips']
        fwd, up = sk.Rp[:, 1], sk.Rp[:, 2]
        skirt = st in ('skirt', 'kimono')
        mat = sp['cloth'] if not skirt else ('tights' if sp['cloth'] != 'bone' else sp['skin'])
        if st == 'balloon': mat = sp['cloth']
        a = sk.pel + up * .03 * H; e = sk.pel - up * .045 * H
        self.prim['pelv'] = c.loft(a, e, fwd, table([(0, .090 * hp, .126 * hp, .092 * hp, .126 * hp), (.6, .090 * hp, .134 * hp, .106 * hp, .134 * hp), (1, .076 * hp, .112 * hp, .095 * hp, .112 * hp)], 2.4) * m * (1.05 if not skirt else .97), 'pelvis', 'pelvis', mat if not skirt else 'navy', cap=(1.3, 1.5), k=.05, side=-1, name='pelv')
        for s, sg in (('L', -1), ('R', 1)):
            hip, kn, an = sk.hip[s], sk.kn[s], sk.an[s]; e1 = sk.e1l[s]; kb = min(1., sk.kbend[s] / 1.8); G = 'leg' + s
            c.group(G, hip, .26 * m, .09 * m); sa = unit(an - kn); ta = unit(kn - hip)
            if st == 'slim':
                tt = [(0, .092, .088, .100, .078), (.3, .088, .082, .092, .076), (.7, .072, .068, .072, .064), (1, .062, .059, .062, .056)]
                ts = [(0, .060, .058, .062, .055), (.25, .058, .062, .076, .058), (.6, .054, .056, .066, .052), (.9, .054, .052, .056, .050), (1, .058, .055, .060, .052)]
            elif st == 'baggy':
                tt = [(0, .092, .086, .100, .078), (.3, .094, .088, .098, .082), (.7, .088, .084, .086, .078), (1, .082, .078, .080, .073)]
                ts = [(0, .080, .077, .082, .073), (.3, .078, .080, .092, .076), (.7, .078, .080, .090, .076), (.92, .073, .074, .082, .070), (1, .062, .062, .068, .060)]
            elif st == 'balloon':
                tt = [(0, .092, .086, .098, .078), (.3, .108, .108, .110, .088), (.7, .124, .132, .124, .098), (1, .130, .140, .132, .102)]
                ts = [(0, .130, .140, .132, .102), (.35, .132, .142, .136, .108), (.66, .124, .132, .130, .104), (.86, .088, .092, .094, .080), (1, .052, .054, .058, .050)]
                tt = [tuple([v[0]] + [x / self.b['limb'] * 1.02 for x in v[1:]]) for v in tt]; ts = [tuple([v[0]] + [x / self.b['limb'] * 1.02 for x in v[1:]]) for v in ts]
            elif st == 'suit':
                tt = [(0, .090, .086, .097, .078), (.3, .088, .084, .091, .077), (.7, .077, .073, .077, .070), (1, .071, .068, .071, .064)]
                ts = [(0, .070, .067, .071, .064), (.3, .066, .067, .076, .063), (.7, .063, .064, .070, .061), (.95, .065, .065, .068, .063), (1, .067, .067, .070, .065)]
            elif st == 'straight':
                tt = [(0, .088, .084, .096, .076), (.3, .088, .084, .092, .077), (.7, .077, .073, .077, .070), (1, .070, .067, .070, .063)]
                ts = [(0, .069, .066, .070, .063), (.3, .067, .068, .078, .063), (.7, .066, .067, .074, .063), (1, .068, .067, .072, .064)]
            else:                                                  # skirted: bare or tights
                tt = [(0, .092, .086, .098, .078), (.3, .086, .080, .090, .074), (.7, .064, .060, .064, .057), (1, .053, .050, .053, .048)]
                ts = [(0, .050, .048, .052, .046), (.25, .046, .052, .062, .050), (.55, .040, .042, .050, .040), (.85, .032, .031, .036, .030), (1, .033, .031, .038, .030)]
            T1, T2 = table(tt, 2.25) * lm, table(ts, 2.25) * lm
            if not skirt and self.lod:
                q = (.019 * kb + .004) * lm * (1.3 if st in ('baggy', 'balloon') else 1.); fs = 51 + (s == 'R') * 7; fa_ = dict(slim=.0038, suit=.004, straight=.005, baggy=.0065, balloon=.008).get(st, .004) * lm
                T1 = T1 + facet(fs, fa_, 5, 6, (.2, 0)); T2 = T2 + facet(fs + 1, fa_, 6, 6)
                if self.lod > 1 and st != 'balloon': T1 = T1 + drape(fs + 100, fa_ * 1.7, 5, (.06, .9), sign=sg); T2 = T2 + drape(fs + 101, fa_ * 1.7, 5, (.2, .97), sign=-sg)
                for t0, p0, k_, w in ((.94, PI, .05, 1.), (.83, PI - .6, .10, .7)): T1 = T1 + ridge(TT, PH, t0, p0, k_, .07, 1.4, q * w)
                for t0, p0, k_, w in ((.08, PI, -.06, 1.), (.20, PI + .6, .10, .75)): T2 = T2 + ridge(TT, PH, t0, p0, k_, .07, 1.5, q * w)
                if st in ('slim', 'suit', 'straight'):                                  # trouser break over the shoe
                    for t0, p0, k_ in ((.93, .3, .03), (.84, -1.6, -.04), (.76, 1.7, .04)): T2 = T2 + ridge(TT, PH, t0, p0, k_, .05, 1.8, (.009 if st != 'straight' else .006) * lm)
                else:                                                                   # blouson bag above the cuff
                    T2 = T2 * (1 + .08 * np.exp(-((TT - .82) / .12) ** 2) * (1 + .35 * np.cos(3 * PH + sg)))
            an2 = kn + (an - kn) * (1 - (sp['legcut'] * H - b['ankz'] * H) / max(.05, (b['kneez'] - b['ankz']) * H)) if st not in ('slim', 'suit', 'skirt', 'kimono') else an + sa * (.010 * H if st != 'suit' else .016 * H)
            self.prim['th' + s] = c.loft(hip, kn, e1, T1, G, 'thigh' + s, mat, cap=(1., 1.05), k=.04, side=sg, name='th' + s)
            self.prim['sn' + s] = c.loft(kn, an2, e1, T2, G, 'shin' + s, mat, cap=(1., 0.), k=.04, side=sg, name='sn' + s, rule=self.r_cap('inside') if not skirt else None)
            if sp.get('burnt') and not skirt:
                sd_ = 85 + (s == 'R'); k0 = .84; i_ = self.prim['sn' + s]; kh = (.16, .2 * sg)
                c.rule[i_] = (c.rule[i_] or []) + [(lambda t, p, q, sd_=sd_: (t > .86 + .14 * tear(p, sd_, 9)) & (t < 1.0005), 'char')]
                if s == 'L': c.rule[i_] += [(lambda t, p, q: ((t - .13) / .085) ** 2 + (wrap(p - .15) / .42) ** 2 + .5 * (tear(p * 4 + t * 11, 93) - .4) < 1.2, 'char'), (lambda t, p, q: ((t - .13) / .085) ** 2 + (wrap(p - .15) / .42) ** 2 + .5 * (tear(p * 4 + t * 11, 93) - .4) < .62, sp['skin'])]
                else: c.rule[self.prim['th' + s]] = [(lambda t, p, q: ((t - .66) / .10) ** 2 + (wrap(p - .9) / .34) ** 2 + .5 * (tear(p * 4 + t * 11, 95) - .4) < 1., 'char')]
            kr = .052 if not skirt else .040
            if st != 'balloon': c.ball(kn + e1 * (.030 if not skirt else .016) * lm + ta * .004, np.array([kr, kr * 1.15, kr]) * lm * (1.18 if st == 'baggy' else 1.), G, 'shin' + s, mat, ax=unit(ta + sa), e1=e1, k=.045, name='knee' + s)
            # glute
            c.ball(sk.hip[s] - fwd * .038 * m + up * .004 * H + sk.Rp[:, 0] * sg * .004, np.array([.070, .080, .068]) * m * hp, 'pelvis', 'pelvis', mat if not skirt else 'navy', ax=up, e1=fwd, k=.05)
            if not skirt:                                          # sock / ankle visible under the hem
                sm = 'sock' if st != 'balloon' else sp['skin']
                c.loft(kn + (an - kn) * .55, an + sa * .02, e1, table([(0, .044, .046, .052, .044), (.6, .038, .038, .042, .036), (1, .036, .035, .040, .034)], 2.2) * lm * (1. if st != 'balloon' else .95), 'sock' + s, 'sock' + s, sm, cap=(1, 1), k=.02, name='sock' + s)
                if self.lod: self.trouser_folds(s, sg, kb, st)
        if st == 'skirt': self.skirt()
        if st == 'kimono': self.kimono_skirt()
        if sp['top'] == 'bare' and st == 'balloon':                # white haramaki
            a = sk.waist + sk.Rw[:, 2] * .035 * H; e = sk.pel + up * .012 * H
            self.prim['belt'] = c.loft(a, e, unit(fwd + sk.Rw[:, 1]), table([(0, .106, .134, .098, .134), (.5, .110, .140, .106, .140), (1, .116, .152, .118, .152)], 2.4) * m * 1.07 + .014 * m, 'belt', 'belt', 'wrap', cap=(0, 0), k=0, side=-1, name='belt')
            if self.lod:
                for t0, p0, p1 in ((.3, -1.1, .2), (.55, -.3, 1.2), (.72, -1.4, -.2), (.45, 1.9, 3.1)): self.line(self.prim['belt'], [t0, t0 + .06, t0], [p0, (p0 + p1) / 2, p1], .8, pri=1)

    def trouser_folds(self, s, sg, kb, st):
        u, a = self.prim['th' + s], self.prim['sn' + s]; L, Sh = self.line, self.shard; k = .4 + .6 * kb
        B = self.blade
        B(u, (.05, -1.2), (.42, .45), .10, .28, 0., 0, False); B(u, (.10, -1.55), (.50, -.45), .08, .28, 0., 0, False); B(u, (.12, 1.5), (.82, 1.2), .085, .35, 0., 0, False)
        B(u, (.98, .2), (.70, .75), .08, .3, 0., 0, False); B(u, (.97, -.2), (.74, -.95), .07, .3, 0., 1, False)
        B(a, (.03, .25), (.34, .9), .09, .3, 0., 0, False); B(a, (.10, 1.35), (.86, 1.05), .085, .35, 0., 0, False); B(a, (.97, .3), (.72, -.5), .07, .28, 0., 0, False); B(a, (.95, 1.7), (.70, 2.4), .07, .28, 0., 1, False)
        # crotch fan
        L(u, [.04, .10, .20], [-1.5, -.9, -.2], 1.0, pri=0); L(u, [.06, .18, .30], [-1.7, -1.15, -.75], .85, pri=1); Sh(u, [.05, .12, .22, .15], [-1.35, -.8, -.1, -.75], pri=0)
        L(u, [.05, .14, .24], [-2.0, -2.6, -3.1], .85, pri=1)
        # knee: pit creases wrap forward; pull from the kneecap
        L(u, [.96, .88, .80], [PI - .4, PI - 1.1 * k, PI - 1.8 * k], 1.0, pri=0); L(u, [.92, .82, .74], [PI + .4, PI + .9 * k, PI + 1.4 * k], .85, pri=1)
        L(a, [.05, .13, .24], [PI + .3, PI + 1.0 * k, PI + 1.7 * k], 1.0, pri=0); L(a, [.08, .17, .28], [PI - .3, PI - .9 * k, PI - 1.35 * k], .85, pri=1)
        Sh(u, [.95, .86, .78, .84], [PI - .6, PI - 1.2 * k, PI - 1.95 * k, PI - 1.05 * k], pri=0); Sh(a, [.06, .16, .28, .12], [PI + .5, PI + 1.15 * k, PI + 1.85 * k, PI + .95 * k], pri=0)
        Sh(u, [.45, .70, .93, .72], [1.15, 1.0, .75, .85], pri=1)
        # shin drape
        L(a, [.35, .55, .78], [.9, .6, .75], .85, pri=1); Sh(a, [.22, .50, .85, .52], [1.2, .95, 1.05, 1.25], pri=0)
        if st in ('baggy', 'balloon'):
            L(a, [.80, .88, .97], [.1, .25, .3], .9, pri=0); L(a, [.78, .88, .97], [1.5, 1.35, 1.3], .85, pri=1); L(a, [.80, .9, .97], [-1.3, -1.1, -1.05], .85, pri=1); L(a, [.82, .9, .97], [2.6, 2.7, 2.75], .8, pri=2)
            Sh(a, [.74, .86, .96, .87], [.75, .6, .55, .8], pri=1)
        else:
            L(a, [.90, .93, .90], [-.9, -.1, .7], .85, pri=0); L(a, [.80, .84, .80], [.2, .9, 1.7], .8, pri=1); L(a, [.95, .975, .95], [.4, 1.1, 1.8], .8, pri=2); Sh(a, [.885, .915, .89, .87], [-.8, 0, .75, 0], pri=1)
        if st == 'suit': L(u, [.05, .5, .97], [0, 0, 0], .7, pri=1, straight=True); L(a, [.06, .5, .97], [0, 0, 0], .7, pri=1, straight=True)
        if st == 'balloon':
            for q, (p0, w) in enumerate(((.35, 1.), (-.5, .9), (1.3, .9), (2.4, .8))): L(u, [.15 + .05 * q, .5, .97], [p0, p0 * .95, p0 * .9], w, pri=1); L(a, [.05, .4, .80], [p0 * .9, p0 * .92, p0], w, pri=1)

    def skirt(self):
        """flared pleated skirt: hull of the thighs, hem lags"""
        sk, c, sp, H, m = self.sk, self.c, self.sp, self.H, self.m; up, fwd = sk.Rp[:, 2], sk.Rp[:, 1]; ww = self.b['waist']
        lag = self.fx.get('hem', np.zeros(3)); a = sk.waist - up * .008 * H; e = sk.pel - up * .205 * H + lag * 2.2
        ax = unit(e - a); L = np.linalg.norm(e - a); e1 = unit(fwd - ax * (fwd @ ax)); e2 = -np.cross(ax, e1)
        fl = 1 + float(self.fx.get('flare', 0.))
        base = table([(0, .098 * ww, .128 * ww, .092 * ww, .128 * ww), (.2, .128, .155, .132, .155), (.6, .185 * fl, .215 * fl, .185 * fl, .215 * fl), (1, .24 * fl, .275 * fl, .235 * fl, .275 * fl)], 2.1) * m
        gl = [(sk.hip[q] - fwd * .038 * m + up * .004 * H, .080 * m * self.b['hips']) for q in 'LR']
        R = self.hull(a, ax, e1, e2, L, self.legsegs('L', 1.25) + self.legsegs('R', 1.25), base, .026 * m, gl, web=.75, smooth=2)
        pl = np.abs(((PH * 5 / PI + .25) % 1) - .5) * 2 - .5                            # five-pleat angular wave each half
        rng = np.random.default_rng(3); pl = pl + .5 * np.cos(3 * PH + 1.1) * rng.uniform(.5, .9)
        R = R * (1 + .060 * pl * np.clip((TT - .12) / .88, 0, 1) ** 1.2)
        self.prim['skirt'] = c.loft(a, e, e1, R, 'skirt', 'skirt', sp['cloth'], cap=(0, 0), k=0, side=-1, name='skirt', rule=self.r_cap('inside'))
        if self.lod:
            for q, p0 in enumerate(np.arange(10) * PI / 5 + .16 + .05 * np.sin(np.arange(10) * 2.3)):
                t0 = .12 + .16 * ((q * 7) % 3) / 2
                self.line(self.prim['skirt'], [t0, .55, 1.], [p0 * .99 + .02, p0 + .01, p0], 1.0 if q % 2 else .85, pri=q % 2, straight=True)
                self.shard(self.prim['skirt'], [t0 + .2, .7, .99, .99, .6], [p0 + .05, p0 + .05, p0 + .04, p0 + .20 + .05 * (q % 3), p0 + .10], pri=q % 3)

    def kimono_top(self):
        sk, c, sp, H, m = self.sk, self.c, self.sp, self.H, self.m; rb = self.prim['ribs']; up, fwd, rt = sk.Rc[:, 2], sk.Rc[:, 1], sk.Rc[:, 0]; g = self.fx.get('down', np.array([0, 0, -1.]))
        for s, sg in (('L', -1), ('R', 1)):
            G = 'sleeve' + s; sh, el, wr = sk.sh[s], sk.el[s], sk.wr[s]; ua, fa = unit(el - sh), unit(wr - el); c.group(G, sh, .24 * m, .06 * m)
            lag = self.fx.get('sleeve' + s, np.zeros(3)); d = unit(g + lag)
            def wing(a, e, r0, D0, D1, n_=2.0):
                ax = unit(e - a); L = np.linalg.norm(e - a); h = d - ax * (d @ ax); hn = np.linalg.norm(h); e1 = h / (hn + 1e-6) if hn > .05 else unit(np.cross(ax, [1, 0, 0])); R = np.zeros((NT, NP))
                for i in range(NT):
                    t = TT[i, 0]; D = (D0 + (D1 - D0) * t) * min(1., hn * 1.3 + .1); r = r0[0] + (r0[1] - r0[0]) * t
                    th = r * .9
                    c_, s_ = np.cos(PH[i]), np.sin(PH[i])
                    rr = np.full(NP, r * 1.0)
                    # tall wedge: half-width th, depth D along phi 0
                    b_ = np.where(c_ > 0, ((np.abs(c_) / (r + D)) ** 2.6 + (np.abs(s_) / th) ** 2.6) ** (-1 / 2.6), ((np.abs(c_) / r) ** 2.2 + (np.abs(s_) / th) ** 2.2) ** (-1 / 2.2))
                    R[i] = b_
                return e1, R
            sh2 = sh - up * .010 * H + (el - sh) * .06
            e1, R = wing(sh2, el, (.058 * m, .072 * m), .03 * m, .14 * m)
            self.prim['sl0' + s] = c.loft(sh2, el, e1, R, G, 'sleeve' + s, 'bone', cap=(1, 1), k=.05, name='sl0' + s)
            e1, R = wing(el, el + fa * np.linalg.norm(wr - el) * .80, (.074 * m, .064 * m), .15 * m, .30 * m)
            R = R * (1 + .04 * np.cos(6 * PH) * TT) + facet(71 + (s == 'R') * 5, .006 * m, 4, 6, (.2, 0))
            self.prim['sl1' + s] = c.loft(el, el + fa * np.linalg.norm(wr - el) * .80, e1, R, G, 'sleeve' + s, 'bone', cap=(1, 0), k=.05, name='sl1' + s, rule=self.r_cap('slate_in'))
            if self.lod:
                i = self.prim['sl1' + s]; self.line(i, [.05, .4, .8], [.5, .35, .18], 1., pri=0); self.line(i, [.15, .5, .9], [-.6, -.4, -.22], .9, pri=1); self.line(i, [.3, .6, .97], [1.1, .8, .5], .8, pri=2)
                i = self.prim['sl0' + s]; self.line(i, [.3, .6, .95], [1.2, .8, .4], .9, pri=1); self.line(i, [.4, .7, .97], [-1.3, -.9, -.5], .9, pri=1)
        # crossed collar band: left over right
        self.deco.append(dict(kind='poly', prim=rb, t=[1.0, .82, .50, .0, .0, .46, .78, .98], p=[-.66, -.46, -.08, .46, .70, .20, -.20, -.42], mat='band', pri=0))
        self.deco.append(dict(kind='poly', prim=rb, t=[1.0, .82, .58, .50, .78, .98], p=[.66, .46, .16, .04, .22, .42], mat='band', pri=0, under=True))

    def kimono_skirt(self):
        sk, c, sp, H, m = self.sk, self.c, self.sp, self.H, self.m; up, fwd = sk.Rp[:, 2], sk.Rp[:, 1]
        lag = self.fx.get('hem', np.zeros(3)); a = sk.waist + sk.Rw[:, 2] * .01 * H
        dn = unit(-up + .8 * self.fx.get('down', np.array([0, 0, -1.]))); L = (self.b['waistz'] - .085) * H
        # follow the legs: aim the axis at the mean ankle
        ma = (sk.an['L'] + sk.an['R']) / 2; dn = unit(dn + 1.2 * unit(ma - a)); e = a + dn * L + lag * 3
        ax = unit(e - a); L = np.linalg.norm(e - a); e1 = unit(fwd - ax * (fwd @ ax)); e2 = -np.cross(ax, e1)
        base = table([(0, .104, .140, .100, .140), (.18, .124, .172, .128, .172), (.6, .145, .190, .150, .190), (1, .165, .215, .175, .215)], 2.3) * m
        spread = np.linalg.norm(sk.an['L'] - sk.an['R']) / (.5 * H); ride = float(np.clip(1.25 - .75 * spread, .45, 1.))
        segs = []
        for q in 'LR': t_, s_ = self.legsegs(q, 1.0); segs += [t_, (s_[0], s_[0] + (s_[1] - s_[0]) * ride, s_[2], s_[3])]
        R = self.hull(a, ax, e1, e2, L, segs, base, .026 * m, web=.62, smooth=5)
        rng = np.random.default_rng(9); w = sum(rng.uniform(.3, 1) * (np.abs(((q * PH / (2 * PI) + rng.uniform(0, 1)) % 1) - .5) * 4 - 1) for q in (3, 5, 7))
        R = R * (1 + .030 * w * np.clip((TT - .2) / .8, 0, 1))
        self.prim['skirt'] = i = c.loft(a, e, e1, R, 'skirt', 'skirt', 'bone', cap=(1.2, 0), k=0, side=-1, name='skirt', rule=self.r_cap('slate_in'))
        self.prim['belt'] = c.loft(sk.waist + sk.Rw[:, 2] * .030 * H, sk.waist - up * .026 * H, unit(fwd + sk.Rw[:, 1]), table([(0, .118, .152, .114, .152), (.5, .122, .160, .120, .160), (1, .126, .168, .126, .168)], 2.4) * m * 1.10 + .012 * m, 'belt', 'belt', 'obi', cap=(0, 0), k=0, side=-1, name='belt')
        if self.lod:
            self.line(i, [.06, .4, .75, 1.], [-.28, -.22, -.1, -.02], 1.15, pri=0, straight=True)      # overlap edge
            for q, (p0, t0, p1) in enumerate(((.5, .12, .75), (-.75, .1, -.95), (1.2, .2, 1.25), (-1.4, .25, -1.5), (2.2, .15, 2.1), (-2.4, .12, -2.5), (3.0, .2, 3.1))): self.line(i, [t0, .5, .78 + .07 * (q % 3)], [p0, (p0 + p1) / 2, p1], .9 if q % 2 else 1., pri=q % 3)
            self.deco.append(dict(kind='cutsh', prim=i, t=[.30, .7, 1., 1., .7], p=[-.2, -.1, -.02, .25, .12], pri=1))
            self.deco.append(dict(kind='cutsh', prim=i, t=[.10, .4, .34], p=[.45, .78, .55], pri=0)); self.deco.append(dict(kind='cutsh', prim=i, t=[.5, .95, .9], p=[.95, 1.1, 1.25], pri=1))
            for q in range(5): self.line(self.prim['belt'], [.25, .6, .75, .25], [-.9 + q * .42, -.75 + q * .42, -.6 + q * .42, -.48 + q * .42], .7, pri=2, mat='obiln')

    # -------------------------------------------------------------- feet
    def foot(self, s):
        sk, c, sp = self.sk, self.c, self.sp; k = self.k * self.b['foot'] * (self.b['mass'] ** .3); R = sk.Rfoot[s]; x, y, z = R[:, 0], R[:, 1], R[:, 2]; sg = -1 if s == 'L' else 1
        an = sk.an[s]; G = 'foot' + s; st = sp['shoe']; tb = sk.lift['t' + s] * D2R
        mat, sole = dict(hightop=('redshoe', 'sole'), ankle=('ochre', 'sole'), loafer=('loafer', 'brown'), dress=('shoeb' if sp['cloth'] == 'cream' else 'black', 'black'), slipper=('black', 'solew'), zori=('tabi', 'black'), bare=(sp['skin'], sp['skin']))[st]
        h = self.b['ankz'] * self.H
        ax0 = -h + .036 * k
        heel = an + R @ np.array([0, -.046 * k, ax0]); ball = an + R @ np.array([sg * .004 * k, .112 * k, ax0 - .002 * k])
        dn = .036 * k
        def ft(st_, flat=dn):
            T = table(st_, 2.3) * k; cp = np.cos(PH); lim = np.where(cp < -.05, flat / np.maximum(-cp, .05), 9.); return np.minimum(T, lim)
        hi = 1. if st in ('hightop', 'ankle') else .86
        rear = ft([(0, .052 * hi, .036, .05, .036), (.3, .060 * hi, .040, .05, .038), (.6, .048 * hi, .046, .05, .042), (1, .034 * hi, .052, .05, .046)])
        shz = dict(hightop=.026, ankle=.020, loafer=.016, dress=.016, slipper=.012, zori=.0, bare=.0)[st] * k
        low = lambda t, p, q: ((q - an) @ z) < -h + shz + .001
        rl = [(low, sole)]
        if st == 'slipper': rl = [(low, sole), (lambda t, p, q: (((q - an) @ z) > -h + .050 * k) | ((((q - an) @ z) > -h + .034 * k) & (((q - an) @ y) > -.03 * k) & (((q - an) @ y) < .095 * k)), sp['skin'])]
        self.prim['ft' + s] = c.loft(heel, ball, z, rear, G, G, mat, cap=(1.5, 1.2), k=.02, side=sg, name=G + 'r', rule=rl)
        td = unit(y * math.cos(tb) + z * math.sin(tb)); tz = unit(z * math.cos(tb) - y * math.sin(tb)); tip = ball + td * .064 * k * (1.12 if st in ('dress', 'loafer') else 1.)
        toe = ft([(0, .034 * hi, .052, .05, .046), (.5, .030, .048, .05, .043), (1, .024, .036, .05, .034)])
        c.loft(ball, tip, tz, toe, G, G, mat, cap=(1., 1.15 if st != 'hightop' else 1.35), k=.02, side=sg, name=G + 't', rule=[(low, sole)])
        if st == 'bare':
            for q in range(4):
                o_ = ball + R @ np.array([(q - 1.5) * .026 * k, .012 * k, -.006 * k]); d_ = unit(td + x * (q - 1.5) * .12 - z * .10)
                c.loft(o_, o_ + d_ * (.075 - .006 * abs(q - 1)) * k, tz, table([(0, .020, .019, .022, .019), (.6, .018, .016, .02, .016), (1, .006, .005, .006, .005)], 2) * k, G, G, mat if self.char != 'curse' else 'tooth', cap=(1, 1), k=.008, name=G + 'c')
        # shaft
        up = unit(sk.kn[s] - an)
        ht = dict(hightop=.085, ankle=.062, loafer=.0, dress=.0, slipper=.0, zori=.10, bare=.0)[st] * k
        if ht > 0:
            q = .046 if st != 'zori' else .040
            self.prim['fs' + s] = c.loft(an - z * .030 * k - y * .006 * k, an + up * ht, y, table([(0, .050, .042, .050, .042), (1, q, q * .95, q * 1.1, q * .95)], 2.2) * k, G, G, mat, cap=(1.2, 0 if st != 'zori' else 1), k=.035, name=G + 's', rule=self.r_cap('inside') if st != 'zori' else None)
        else:
            c.loft(an - z * .030 * k, an + up * .035 * k, y, table([(0, .040, .036, .044, .036), (1, .034, .032, .038, .032)], 2.2) * k, G if st in ('slipper', 'bare') else 'sock' + s, G if st in ('slipper', 'bare') else 'sock' + s, sp['skin'] if st in ('slipper', 'bare') else 'tights' if st == 'loafer' else 'sock', cap=(1, 1), k=.03, name=G + 'an')
        if st in ('zori',):
            sl0 = an + R @ np.array([0, -.062 * k, -h + .004 * k]); sl1 = an + R @ np.array([0, .185 * k, -h + .004 * k])
            c.loft(sl0, sl1, z, np.minimum(table([(0, .02, .040, .02, .040), (.6, .02, .050, .02, .05), (1, .02, .040, .02, .04)], 2.3) * k, .012 * k / np.maximum(np.abs(np.cos(PH)), .1)), 'geta' + s, 'geta' + s, 'black', cap=(1.2, 1.2), k=0, name='geta' + s)
        if self.lod:
            i = self.prim['ft' + s]
            if st == 'hightop':
                for t0, w_ in ((.50, .42), (.62, .46), (.74, .50), (.86, .52)): self.line(i, [t0 + .05, t0, t0 + .05], [-w_, 0, w_], 1.0, pri=1)
                self.line(i, [.40, .75, 1.], [.70, .66, .60], .9, pri=0); self.line(i, [.40, .75, 1.], [-.70, -.66, -.60], .9, pri=0)
                if 'fs' + s in self.prim:
                    j = self.prim['fs' + s]
                    for t0 in (.30, .55, .80): self.line(j, [t0 + .06, t0, t0 + .06], [-.55, 0, .55], .9, pri=1)
                    self.line(j, [.05, .5, 1.], [.72, .66, .62], .85, pri=1); self.line(j, [.05, .5, 1.], [-.72, -.66, -.62], .85, pri=1)
            elif st == 'ankle':
                self.line(i, [.66, .56, .66], [-1.1, 0, 1.1], .9, pri=0)
                if 'fs' + s in self.prim: self.line(self.prim['fs' + s], [0., .5, 1.], [1.45, 1.4, 1.35], .9, pri=1); self.line(self.prim['fs' + s], [0., .5, 1.], [-1.45, -1.4, -1.35], .9, pri=1)
            elif st in ('dress', 'loafer'): self.line(i, [.74, .62, .74], [-1.0, 0, 1.0], .9, pri=0); self.line(i, [.40, .34, .40], [-.8, 0, .8], .8, pri=1)

    # -------------------------------------------------------------- neck + collars
    def neck_(self):
        sk, c, m = self.sk, self.c, self.m; r = .052 * m * self.b['neck']; bare = self.sp['top'] == 'bare'
        a = sk.neck - sk.Rc[:, 2] * .03 * self.H; e = sk.headb + sk.Rn[:, 2] * .035 * self.H
        G = 'neck'; c.group(G, sk.neck, .16 * m, .07 * m if bare else 0.)
        c.loft(a, e, sk.Rn[:, 1], table([(0, r * 1.05, r * 1.15, r * 1.1, r * 1.15), (.5, r * .92, r * .98, r, r * .98), (1, r * .92, r, r * 1.05, r)], 2.1), G, 'neck', self.sp['skin'], cap=(1, 1), k=.02, side=-1, name='neck')

    def collar(self):
        sk, c, m, H = self.sk, self.c, self.m, self.H; st = self.sp['collar']; up = unit(sk.Rc[:, 2] + sk.Rn[:, 2] * .5); fwd = sk.Rc[:, 1]; rt = sk.Rc[:, 0]; n0 = sk.neck
        rn = .052 * m * self.b['neck']; ph = np.arange(NP) * 2 * PI / NP                          # ring frame: e1 = fwd, so phi 0 = front; e2 = cross(up, fwd) = left
        mat = self.sp['cloth']
        if st == 'hood':
            # thick folded cowl: two lapping rolls, the front dips to a V, the hood bag hangs behind
            lag = self.fx.get('hood', np.zeros(3))
            c.ring(n0 + up * .000 * H, unit(up + fwd * .22), fwd, .080 * m * (1 + .10 * np.cos(ph)), .036 * m * (1 + .12 * np.cos(2 * ph + .6)), .012 * m * np.cos(2 * ph) - .020 * m * np.maximum(np.cos(ph), 0) ** 3, 1.0, 'hood', 'hood1', 'red', k=.01, name='hood1')
            c.ring(n0 + up * .024 * H + fwd * .004, unit(up + fwd * .10 + rt * .04), fwd, .070 * m * (1 + .10 * np.cos(ph)), .027 * m * (1 + .16 * np.cos(2 * ph + 2.)), -.014 * m * np.maximum(np.cos(ph), 0) ** 4 + .006 * m * np.cos(3 * ph), .85, 'hood', 'hood2', 'red', k=.004, name='hood2')
            b0 = n0 - fwd * .052 * m + up * .012 * H; b1 = n0 - fwd * (.112 * m) - up * .092 * H + lag * 1.4
            hb = table([(0, .030, .100, .046, .100), (.25, .034, .128, .052, .128), (.55, .036, .122, .050, .122), (.82, .030, .080, .040, .080), (1, .020, .030, .026, .030)], 2.3) * m
            hb = hb + ridge(TT, PH, .35, PI, .10, .10, 1.2, .008 * m) + ridge(TT, PH, .60, PI + .3, -.12, .10, 1.1, .006 * m)
            self.prim['hoodb'] = i = c.loft(b0, b1, fwd, hb, 'hood', 'hoodb', 'red', cap=(1.5, 1.), k=.03, side=-1, name='hoodb')
            if self.sp.get('burnt'): c.rule[i] = [(lambda t, p, q: t > .62 + .34 * tear(p, 89, 11), 'char')]
            if self.lod:
                self.line(i, [.10, .30, .46, .30, .10], [PI - 1.25, PI - .8, PI, PI + .8, PI + 1.25], 1.1, pri=0); self.line(i, [.50, .66, .92], [PI + .1, PI + .25, PI + .2], .9, pri=1); self.line(i, [.42, .55, .70], [PI - .5, PI - .7, PI - .6], .8, pri=1)
                self.shard(i, [.14, .34, .50, .38], [PI - 1.0, PI - .6, PI, PI - .6], pri=0)
        elif st in ('high', 'stand', 'low'):
            h, r0, r1 = dict(high=(.066, 1.28, 1.52), stand=(.040, 1.22, 1.36), low=(.018, 1.18, 1.22))[st]
            a = n0 - up * .012 * H; e = n0 + up * h * H + fwd * .004 * H
            T = table([(0, rn * r0 * 1.25, rn * r0 * 1.3, rn * r0 * 1.2, rn * r0 * 1.3), (.35, rn * r0 * 1.02, rn * r0 * 1.06, rn * r0, rn * r0 * 1.06), (1, rn * r1 * 1.06, rn * r1 * 1.02, rn * r1, rn * r1 * 1.02)], 2.1)
            if st == 'high': T = T + ridge(TT, PH, .55, .3, .10, .12, 1.4, .006 * m) + ridge(TT, PH, .35, -.9, -.12, .12, 1.4, .006 * m)
            self.prim['col'] = c.loft(a, e, fwd, T, 'collar', 'collar', mat, cap=(1.5, 0), k=0, side=-1, name='collar', rule=self.r_cap('inside'))
            if self.lod:
                i = self.prim['col']
                self.line(i, [.1, .6, 1.], [0 if st == 'high' else .28] * 3, 1.0, pri=0, straight=True)
                if st == 'high': self.line(i, [.30, .42, .36], [-.9, -.3, .3], .9, pri=1); self.line(i, [.62, .55, .62], [.2, .8, 1.4], .85, pri=1); self.shard(i, [.38, .50, .44, .42], [-.85, -.3, .3, -.3], pri=0); self.shard(i, [.70, .64, .70, .78], [.2, .8, 1.4, .8], pri=1)
        elif st == 'lapel': self.lapel()
        elif st == 'scarf':
            c.ring(n0 + up * .006 * H, unit(up + fwd * .25), fwd, .082 * m * (1 + .12 * np.cos(ph)), .036 * m * (1 + .15 * np.cos(2 * ph + .4)), -.022 * m * np.maximum(np.cos(ph), 0) ** 2, .9, 'scarf', 'scarf1', 'slate', k=.006, name='scarf1')
            c.ring(n0 + up * .030 * H, unit(up + fwd * .05 - rt * .10), fwd, .074 * m * (1 + .10 * np.cos(ph)), .030 * m * (1 + .18 * np.cos(2 * ph + 2.4)), -.010 * m * np.maximum(np.cos(ph), 0) ** 3, .8, 'scarf', 'scarf2', 'slate', k=.004, name='scarf2')
            fl = np.asarray(self.fx.get('scarf', self.fx.get('hair', np.zeros(3))), float); t_ = self.fx.get('f', 0)
            for q, (sx, ln, w0) in enumerate(((-.5, .62, .060), (.75, .50, .050))):                     # two loose ends behind the neck, whipping with the motion
                p0 = n0 - fwd * .085 * m + rt * sx * .06 * m + up * .004 * H; d_ = unit(-up * .75 - fwd * .5 + fl * 1.25); sd = unit(np.cross(d_, fwd) + rt * .4)
                for i in range(6):
                    u_ = i / 6.; wv = math.sin(t_ * .95 - i * .95 + q * 2.1) * (.18 + .5 * u_) * min(1.4, .25 + np.linalg.norm(fl)); dd = unit(d_ + np.cross(d_, sd) * wv + sd * wv * .5 * math.cos(q + i) - up * .05 * i); p1 = p0 + dd * ln * m / 6. * 1.25
                    a0, a1 = w0 * (1 - .10 * i), w0 * (1 - .10 * (i + 1)) if i < 5 else w0 * .30; th = .014
                    c.loft(p0, p1, sd, table([(0, a0, th, a0, th), (1, a1, th, a1, th)], 2.6) * m, 'scarft', 'scarf3', 'slate', cap=(1, 1), k=.012, name='scarf3'); p0 = p1
        elif st == 'none' and self.char == 'todo':
            c.ring(n0 + up * -.012 * H + fwd * .02, unit(up + fwd * .75), fwd, rn * 1.5, .006 * m, 0., 1., 'neckl', 'neckl', 'neckl', k=0, name='neckl')

    def lapel(self):
        """suit: shirt V, tie, lapels as material zones on the chest; collar ring of shirt"""
        sk, c, m, H = self.sk, self.c, self.m, self.H; up, fwd, rt = sk.Rc[:, 2], sk.Rc[:, 1], sk.Rc[:, 0]; rn = .052 * m * self.b['neck']; n0 = sk.neck
        c.loft(n0 - up * .02 * H, n0 + unit(sk.Rn[:, 2]) * .028 * H, fwd, table([(0, rn * 1.35, rn * 1.35, rn * 1.2, rn * 1.35), (1, rn * 1.14, rn * 1.12, rn * 1.1, rn * 1.12)], 2.1), 'collar', 'collar', 'shirtb', cap=(1, 0), k=0, side=-1, name='scol', rule=self.r_cap('inside'))
        o = sk.chest; W = .052 * m; zt = .120 * H; zb = -.030 * H
        def loc(q): d = q - o; return d @ rt, d @ up, d @ fwd
        def vw(zz): return np.clip((zz - zb) / (zt - zb), 0, 1.1) * .070 * m
        def shirt(t, p, q): x, zz, y = loc(q); return (y > .02) & (np.abs(x) < vw(zz)) & (zz > zb)
        def tie(t, p, q):
            x, zz, y = loc(q); w = np.interp(zz, [zb, zt * .55, zt * .80, zt * .90, zt], np.array([.030, .030, .020, .017, .024]) * m); return (y > .02) & (np.abs(x) < np.minimum(w, vw(zz))) & (zz > zb)
        for i, nm in enumerate(c.name):
            if nm in ('ribs', 'belly', 'trapL', 'trapR', 'latL', 'latR') or nm == 'hem': c.rule[i] = (c.rule[i] or []) + [(shirt, 'shirtb'), (tie, 'tie')]
        rb, bl = self.prim['ribs'], self.prim['belly']
        # lapel notch, edges, buttons, pockets, dart: authored on the chest lofts
        self.front = []
        F = lambda pts, w=1., closed=False, kind='line', mat=None: self.front.append(dict(pts=np.array(pts, float), w=w, kind=kind, mat=mat))
        for sg in (-1, 1):
            F([(sg * .072, .105), (sg * .118, .060), (sg * .100, .046), (sg * .112, .036), (sg * .062, -.02), (sg * .004, -.052)], 1.0)
            F([(sg * .085, -.095), (sg * .080, -.03)], .7); F([(sg * .070, -.125), (sg * .115, -.120)], .85); F([(sg * .070, -.137), (sg * .115, -.132)], .85)
        F([(.004, -.052), (.012, -.10), (.004, -.16), (-.03, -.235)], 1.0); F([(.004, -.052), (.006, -.14), (.03, -.235)], .0)
        F([(.075, .030), (.115, .038)], .9)
        for zz in (-.065, -.105): F([(.010, zz), (.017, zz - .004), (.010, zz - .008)], 1.2)
        # tie spots
        rng = np.random.default_rng(4)
        for q in range(26):
            zz = rng.uniform(-.026, .10); x = rng.uniform(-.022, .022)
            F([(x, zz)], .0055 * rng.uniform(.7, 1.3), kind='dot', mat='tiedot')

    # -------------------------------------------------------------- curse hulk
    def hulk(self):
        """generic cursed spirit: a hunched slab with a mouth for a chest; arms longer than legs.  Built for silhouette."""
        sk, c, H = self.sk, self.c, self.H; k = H / 2.35; L_ = self.sp['skin']; up, fwd, rt = sk.Rc[:, 2], sk.Rc[:, 1], sk.Rc[:, 0]; pu, pf = sk.Rp[:, 2], sk.Rp[:, 1]
        # trunk: one leaning wedge, hump higher than the head
        self.prim['ribs'] = c.loft(sk.pel - pu * .02 * H, sk.neck + up * .06 * H - fwd * .08 * k, unit(fwd + pf), table([(0, .20, .24, .22, .24), (.3, .26, .28, .24, .28), (.6, .30, .36, .30, .36), (.85, .25, .34, .32, .34), (1, .16, .22, .24, .22)], 2.2) * k, 'torso', 'torso', L_, cap=(1.2, 1.2), k=.12, side=-1, name='ribs')
        c.ball(sk.neck - fwd * .16 * k + up * .02 * k, np.array([.30, .22, .20]) * k, 'torso', 'torso', L_, ax=up, e1=rt, k=.14, name='hump')
        for q, (x, z, r) in enumerate(((.10, .05, .10), (-.13, -.02, .12), (.02, -.16, .10), (-.06, .13, .08))): c.ball(sk.neck - fwd * .30 * k + rt * x * k + up * z * k, r * k, 'torso', 'torso', L_, k=.06)
        # spines down the hump
        for q in range(6):
            u_ = q / 5.; o_ = sk.neck * (1 - u_) + (sk.pel + pu * .10 * H) * u_ - unit(fwd + pf) * (.34 - .05 * u_ + .06 * math.sin(u_ * 3)) * k + up * (.10 - .10 * u_) * k
            d_ = unit(-unit(fwd + pf) * .9 + up * (.6 - .5 * u_) + rt * .25 * (-1) ** q)
            c.loft(o_, o_ + d_ * (.20 - .10 * u_) * k, rt, table([(0, .050, .05, .05, .05), (1, .006, .006, .006, .006)], 2) * k, 'torso', 'torso', L_, cap=(1, 1), k=.03, name='spine')
        self.curseface = True
        c.group('arm' + 'L', sk.sh['L'], .42 * k, .14 * k); c.group('armR', sk.sh['R'], .42 * k, .14 * k)
        for s, sg in (('L', -1), ('R', 1)):
            sh, el, wr = sk.sh[s], sk.el[s], sk.wr[s]; e1 = sk.e1a[s]; ua, fa = unit(el - sh), unit(wr - el); G = 'arm' + s
            self.prim['ua' + s] = c.loft(sh, el, e1, table([(0, .15, .15, .14, .13), (.35, .15, .13, .14, .12), (.75, .10, .10, .12, .09), (1, .085, .085, .085, .08)], 2.1) * k, G, 'uarm' + s, L_, cap=(1, 1), k=.05, side=sg)
            c.ball(sh + ua * .06 * k + up * .02 * k, np.array([.19, .20, .18]) * k, G, 'uarm' + s, L_, ax=ua, e1=e1, k=.08)
            self.prim['fa' + s] = c.loft(el, wr, e1, table([(0, .10, .10, .10, .09), (.3, .125, .13, .12, .11), (.7, .095, .10, .095, .09), (1, .075, .085, .075, .08)], 2.1) * k, G, 'farm' + s, L_, cap=(1.1, 1.3), k=.06, side=sg)
            c.ball(el + e1 * .04 * k, .075 * k, G, 'farm' + s, L_, k=.05)
            self.hand(s)
            hip, kn, an = sk.hip[s], sk.kn[s], sk.an[s]; e1 = sk.e1l[s]; G = 'leg' + s; c.group(G, hip, .4 * k, .14 * k)
            self.prim['th' + s] = c.loft(hip, kn, e1, table([(0, .17, .16, .18, .13), (.4, .165, .15, .16, .13), (1, .105, .10, .10, .095)], 2.1) * k, G, 'thigh' + s, L_, cap=(1, 1), k=.06, side=sg)
            self.prim['sn' + s] = c.loft(kn, an, e1, table([(0, .10, .10, .11, .09), (.3, .095, .105, .14, .10), (.8, .065, .065, .075, .06), (1, .065, .07, .08, .065)], 2.1) * k, G, 'shin' + s, L_, cap=(1, 1), k=.06, side=sg)
            c.ball(kn + e1 * .05 * k, np.array([.075, .09, .075]) * k, G, 'shin' + s, L_, ax=unit(kn - hip), e1=e1, k=.06)
            self.foot(s)
            if self.lod:
                for pr, ts, ps in (('ua', (.1, .35, .55), (1.4, 1.9, 2.5)), ('ua', (.5, .7, .9), (-1.2, -1.6, -1.5)), ('fa', (.08, .3, .6), (1.3, 1.7, 1.4)), ('fa', (.4, .6, .9), (-1., -.8, -.5)), ('th', (.2, .45, .75), (.9, .8, .4)), ('sn', (.1, .35, .7), (2., 2.3, 2.4))): self.line(self.prim[pr + s], ts, ps, 1., pri=0)
        c.ball(sk.neck + fwd * .20 * k + up * .02 * k, np.array([.17, .16, .16]) * k, 'torso', 'torso', L_, ax=up, e1=rt, k=.08, name='brow')
        c.ball(sk.neck + fwd * .24 * k - up * .20 * k, np.array([.24, .13, .13]) * k, 'torso', 'torso', L_, ax=up, e1=rt, k=.08, name='jaw')
        self.neckless = True
