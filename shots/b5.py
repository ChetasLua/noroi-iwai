"""b5  NANAMI after hours: blue street, warm bakery window; the watch, the tie, the roll, the three in the glass, the corner of his mouth."""
import numpy as np, cv2, math, os
from jj import W, H, comp, world
from jj.paint import rgb, f32, smooth, ramp, grid, fbm
from jj.ink import Cel, spl
from jj.comp import lerp_keys, keyed
from jj.kit import para, smear, flare, shake
from chars.ntkit import rim, hexmix, qb
from chars.nanami import Nanami, Body, PAL, EXPR
from chars.nt_props import watch, sandwich
from chars.todo import hand3
from chars import nt_keys as K
from jj import hands as HD

VER = 8
YAW, LW = .46, 5.6                                   # ONE head for the whole shot: the same three-quarter, the same line, in every cut
WP = PAL['window']


def seq(f, *k): return float(lerp_keys(f, list(k)))


class Shot:
    id = 'b5'; n = 72
    finish = dict(bloom=.44, bloom_thr=.70, vig=.15, halation='#ffcf96', soft=.18)
    EVENTS = [(0, 'street_bed', dict(size=.5, until=72, note='blue-hour shopping street, far traffic, shop bell somewhere, oven fan')), (3, 'cloth', dict(size=.3, note='sleeve as he raises the watch')), (4, 'bike', dict(size=.3, until=17, note='bicycle passes behind, freewheel ticking, dynamo light')),
              (18, 'tick', dict(size=.7, note='watch, very close: second hand jumps')), (27, 'tick', dict(size=.6, note='second jump, under the cut')), (29, 'cloth', dict(size=.5, note='tie knot tugged')), (33, 'cloth', dict(size=.6, note='knot slides, silk')), (36, 'exhale', dict(size=.4, note='he breathes out')),
              (40, 'paper', dict(size=.6, note='paper bag opens')), (42, 'paper', dict(size=.5, note='crusty roll drawn out, crackle')), (45, 'kids_pass', dict(size=.6, until=62, note='three kids laughing behind him, heard through glass, left to right, walking off')), (54, 'mouth', dict(size=.4, note='the glint slides off his lenses (54-58), his eyes soften, the corner of his mouth gives in: small warm note'))]

    # ------------------------------------------------------------------ plates
    def setup(self):
        self.x, self.y = grid(); pf = lambda k: f'cache/b5_v{VER}_{k}.npy'; ks = ('A', 'C', 'E', 'G', 'RF', 'RO', 'PF', 'PO')
        if not all(os.path.exists(pf(k)) for k in ks[:4]):
            P = world.plate('bakery'); im = P.img.astype(f32)
            A = comp.bokeh(comp.cam2d(im, 1290, 722, 1.22, w=2080, h=1170), 3.2, 1.0); np.save(pf('A'), A.astype(np.float16)); del A
            C = comp.bokeh(cv2.GaussianBlur(comp.cam2d(im, 1215, 700, 1.80, w=2080, h=1170), (0, 0), 2), 10, 1.0); np.save(pf('C'), C.astype(np.float16)); del C
            E = comp.bokeh(cv2.GaussianBlur(comp.cam2d(im, 1290, 716, 2.9, w=2080, h=1170, interp=cv2.INTER_CUBIC), (0, 0), 3), 8, 1.0); np.save(pf('E'), E.astype(np.float16)); del E
            np.save(pf('G'), cv2.GaussianBlur(comp.cam2d(P.mask['glass'].astype(f32), 1290, 716, 2.9, w=2080, h=1170), (0, 0), 3).astype(np.float16)); del P, im
        if not all(os.path.exists(pf(k)) for k in ks[4:]):
            from jj.figure import Fig
            from jj.fig_pose import POSES, mirror
            from jj.cam import Camera3D
            from jj.fig_pose import walk
            st = [walk(k / 8.) for k in range(8)]; arms = ('hL', 'hR', 'eL', 'eR', 'wL', 'wR', 'gL', 'gR', 'lead')
            def sprite(L, hh, ww):
                ys, xs = np.nonzero(L[..., 3] > .02); y0, x0 = max(0, ys.min() - 4), max(0, xs.min() - 4); cr = np.zeros((hh, ww, 4), np.float16); q = L[y0:y0 + hh, x0:x0 + ww]; cr[:q.shape[0], :q.shape[1]] = q; return cr, (x0, y0)
            # the three, walking right to left: four drawings of a step each
            from chars.yuji import Yuji
            from chars import nobara as NB
            from chars.nobara import Nobara
            from chars.megumi import Megumi
            from chars.yuji import YHead
            def hf(fn): return lambda cel, Hh, i: (fn(cel, Hh), None)[1]
            HF = dict(nobara=hf(lambda cel, h: Nobara.head(cel, Nobara.pose(h.cx, h.cy, h.s, h.yaw, h.pitch, h.roll), 'point', 'blue', 0, (0, 0), 1)), yuji=hf(lambda cel, h: Yuji.head(cel, YHead().pose(h.cx, h.cy, h.s, h.yaw, h.pitch, h.roll), 'laugh', 'under', 0, 0., 1)), megumi=hf(lambda cel, h: Megumi.head(cel, Megumi.pose(h.cx, h.cy, h.s, h.yaw, h.pitch, h.roll), 'deadpan', 'blue', 0, (0, 0), 1)))
            cam = Camera3D((0, -7.2, 1.35), (0, 0, 1.06), 3050); RF, RO = [], []
            for n, up, x0 in (('nobara', None, -.80), ('yuji', 'headhands', .05), ('megumi', 'pockets', .84)):
                for k, w_ in enumerate(st):
                    q = dict(w_)
                    if up: q.update({m: POSES[up][m] for m in arms if m in POSES[up]})
                    if n == 'nobara': q.update(hR=(.30, .70, .05), eR=(.3, -.6, -.6))
                    if n == 'yuji': q.update(hed=(-10, 4, 8))
                    cr, o = sprite(Fig(n).pose(q).place(x0, 0, 0, heading=math.pi / 2 + (.25 if n != 'megumi' else .1)).draw(cam, k * 2, head_fn=HF[n]), 1000, 620); RF.append(cr); RO.append(o)
            np.save(pf('RF'), np.stack(RF)); np.save(pf('RO'), np.array(RO)); del RF
            cam = Camera3D((0, -8., 1.45), (0, 0, .95), 1750); PF, PO = [], []
            for n, hd in (('crowd', math.pi / 2 + .2), ('crowdf', -math.pi / 2 - .25)):
                for k, w_ in enumerate(st): cr, o = sprite(Fig(n).pose(w_).place(0, 0, 0, heading=hd).draw(cam, k * 2), 480, 320); PF.append(cr); PO.append(o)
            np.save(pf('PF'), np.stack(PF)); np.save(pf('PO'), np.array(PO))
        for k in ks: setattr(self, k, np.load(pf(k), mmap_mode='r'))

    def plate(self, k, x0, y0, z):
        """stand camera over a stored plate; only the needed window is read"""
        P = getattr(self, k); h, w = P.shape[:2]; m = 8; xa, xb = int(max(0, x0 - W / 2 / z - m)), int(min(w, x0 + W / 2 / z + m)); ya, yb = int(max(0, y0 - H / 2 / z - m)), int(min(h, y0 + H / 2 / z + m))
        return comp.cam2d(np.ascontiguousarray(P[ya:yb, xa:xb]).astype(f32), x0 - xa, y0 - ya, z)

    def frame(self, f):
        if f < 18: im = self.wide(f)
        elif f < 27: im = self.insert(f)
        elif f < 40: im = self.tie(f)
        elif f < 45: im = self.roll(f)
        else: im = self.glass(f)
        return im.astype(f32)

    def gog(self, f):
        """one pair of goggles: one tint, one slab of window light sliding on through the cuts; it whitens and leaves once, at the glass"""
        return dict(slide=seq(f, (0, .30), (9, .26), (14, .20), (27, .20), (33, .28), (39, .18), (45, .18), (52, .02), (57, -.40), (72, -.50)), glare=seq(f, (0, 0.), (46, 0.), (49, .40), (51, .80), (53, .92), (54, .78), (55, .48), (56, .20), (57, .05), (58, 0.), (72, 0.)), clear=seq(f, (0, .12), (54, .12), (57, .66), (72, .70)))

    def his(self, f, cx, cy, s, pitch=0., roll=0., turn=0.): return Nanami.pose(cx, cy, s, YAW + turn, pitch, roll), s / 300 * LW, dict(f=f // 3 * 3, wind=.015, seed=3, lit=1, lw=s / 300 * LW, loose=0., goggles=self.gog(f))

    # ------------------------------------------------------------------ light on his cels: window right, sky left
    def lit(self, lay, cx, cy, s, r=1.):
        lay = rim(lay, -5 * r, 2 * r, '#ffdca0', .92, .6); lay = rim(lay, -2 * r, 1, '#fff6dc', .8, .5); lay = rim(lay, 4 * r, 2, '#6f86d8', .55, .7)
        lay = K.para_lit(lay, (cx + 1.5 * s, cy + .4 * s), (cx - .4 * s, cy + .2 * s), [(0, '#5a3a14'), (.6, '#180e06'), (1, '#000000')], .9, .2, .55)
        return para(lay, (cx - .2 * s, cy), (cx - 2.0 * s, cy + 1.2 * s), [(0, '#ffffff'), (.45, '#b9bde6'), (1, '#6f78b8')], 'mul', .9)

    def steam(self, f, px, py, n=7, w=60, h=380, amt=.34, seed=1):
        m = np.zeros((H // 2, W // 2), f32); r = np.random.default_rng(seed)
        for k in range(n):
            t = (f * .030 * r.uniform(.8, 1.25) + k / n) % 1.; yy = np.linspace(0, 1, 14); ph = r.uniform(0, 6.28); X = px + (k - n / 2) * w / n + (14 + 46 * t) * np.sin(yy * 4.2 + ph + f * .10) * yy; Y = py - h * (t * .55 + yy * .45); a = amt * math.sin(math.pi * t) ** 1.3
            cv2.polylines(m, [np.round(np.c_[X, Y] / 2 * 8).astype(np.int32)], False, float(a), int(4 + 12 * t), cv2.LINE_AA, 3)
        return cv2.resize(cv2.GaussianBlur(m, (0, 0), 5), (W, H), interpolation=cv2.INTER_LINEAR)[..., None]

    def passer(self, px, py, s, col='#0a0c1c', lean=0., step=0.):
        """a dark soft passer-by (close to the lens): head, coat, bag"""
        c = Cel(); M = lambda P: np.array([(px + (u + lean * (1 - v) * .2) * s, py + v * s) for u, v in P])
        c.fill(M(spl([(-.16, -1.02), (-.02, -1.18), (.15, -1.06), (.17, -.86), (.06, -.72), (-.10, -.74), (-.18, -.88)], True, n=7)), col)
        c.fill(M(spl([(-.10, -.76), (.10, -.76), (.36, -.58), (.46, -.1), (.50, .9), (.42 + .06 * step, 2.2), (-.40 + .06 * step, 2.2), (-.46, .9), (-.44, -.1), (-.34, -.58)], True, n=7)), col); c.fill(M([(.36, .3), (.66, .34), (.62, .86), (.34, .84)]), col)
        return c.layer()

    # ------------------------------------------------------------------ A  0-18
    def wide(self, f):
        d = f // 3 * 3; bg = self.plate('A', 1050, 585, 1.014 + .0006 * f)                                     # locked street: he does not slide on it
        # the bicycle light far down the street, a couple of soft walkers
        bx, by = 250 + 13.5 * f, 566 + 2.0 * f + 2 * math.sin(f * .9); g = np.zeros((H, W, 3), f32); cv2.circle(g, (int(bx * 8), int(by * 8)), int((5 + .28 * f) * 8), (1.2, 1.15, .95), -1, cv2.LINE_AA, 3)
        bk = Cel(); q = 1 + .05 * f; bk.fill(np.array([(bx - 13 * q, by - 92 * q), (bx + 10 * q, by - 94 * q), (bx + 26 * q, by - 60 * q), (bx + 24 * q, by - 8 * q), (bx + 10 * q, by + 46 * q), (bx - 14 * q, by + 46 * q), (bx - 26 * q, by - 8 * q), (bx - 24 * q, by - 60 * q)]), '#14172e'); bk.fill(spl([(bx - 12 * q, by - 100 * q), (bx, by - 122 * q), (bx + 13 * q, by - 100 * q), (bx, by - 86 * q)], True, n=5), '#14172e')
        for s_ in (-1, 1): bk.line([(bx + s_ * 2, by + 8 * q), (bx + s_ * 3, by + 74 * q)], 5 * q, '#14172e', taper=(0, 0))
        bg = comp.over(bg, cv2.GaussianBlur(bk.layer(), (0, 0), 3.2), .92); bg = bg + cv2.GaussianBlur(g, (0, 0), 1.5) + qb(g, 16) * 1.6 + qb(g, 60) * 1.4
        for i, (x0, v, y0, o) in enumerate(((1600, -10.5, 548, .90), (1170, 9.5, 556, .86))):                     # two soft walkers between him and the lit window, heads on the eye line
            spr = np.array(self.PF[i * 8 + (f // 3 + 3 * i) % 8], f32); h_, w_ = spr.shape[:2]; px, py = int(x0 + v * f), y0; L = np.zeros((H, W, 4), f32); xa, xb, ya, yb = max(0, px), min(W, px + w_), max(0, py), min(H, py + h_)
            if xb > xa and yb > ya: L[ya:yb, xa:xb] = spr[ya - py:yb - py, xa - px:xb - px]
            L[..., :3] = L[..., :3] * .10 + rgb('#16152e') * L[..., 3:4] * .9; L = rim(L, 4 if i else -4, 1, '#8a5a3a', .5, 1.); bg = comp.over(bg, cv2.GaussianBlur(L, (0, 0), 3.4), o)
        # ---- him: standing.  Weight on his right leg; at 9 he lets it go over to the left, once; then only breath
        w = float(lerp_keys(f, [(0, -1.), (8, -1.), (9, -1.08), (10, -.62), (11, .05), (12, .62), (13, .96), (14, 1.10), (15, 1.06), (16, 1.0), (18, 1.0)], lambda t: t)); dip = 6. * math.exp(-((f - 11.2) / 1.3) ** 2) - 2.5 * math.exp(-((f - 14.3) / 1.2) ** 2); br = math.sin((f - 3) / 24. * 2 * math.pi / 2.6)
        s, cx, cy = 128, 742 + 8. * w, 340 + dip; ups = [(0, 0.), (3, .34), (6, .88), (9, 1.06), (12, 1.0), (18, 1.0)]; k = float(lerp_keys(d, ups, lambda t: t)); lo = [(1.62, -.54, 0.), (2.00, -2.24, .35), (1.55, -3.55, 1.15)]; hi = [(1.62, -.54, 0.), (1.98, -2.14, .75), (.66, -1.60, 1.70)]
        S_, E_, W_ = [tuple(np.array(a) * (1 - k) + np.array(b) * k) for a, b in zip(lo, hi)]; E_ = (E_[0] + .16 * math.sin(math.pi * min(k, 1)), E_[1], E_[2]); dn = float(lerp_keys(f, [(0, .25), (3, .35), (8, 1.), (18, 1.)]))
        H_, lw, kw = self.his(f, cx, cy, s, -.085 * dn, -.010 + .020 * w); b = Body.under(H_, .30, 0, -.046 * w); c = Cel(); e = dict(EXPR['deadpan'], open=.62 - .14 * dn, gaze=(-.25 * dn, -.55 * dn))
        hA = lambda cel, w_, e_, fz: hand3(cel, w_, e_, fz, WP, lw, 'L', 'relaxed', 8, 20, 0, .66, bld=HD.SLIM, cut=-.22, light=(.8, .3, .5))
        Nanami.head(c, H_, e, WP, part='back', **kw); Nanami.draw_suit(c, H_, b, WP, lw, 0, 1, 0., arms=(1, 0)); Nanami.head(c, H_, e, WP, part='front', **kw); Nanami.suit_arm(c, b, S_, E_, W_, WP, lw, 1, 1, 3, hand=hA, watch=True); lay = c.layer()
        # the pelvis carries the weight: it goes over the standing leg, the hem follows two frames later, the ribs open and close
        w2 = float(lerp_keys(f - 2, [(0, -1.), (9, -1.), (10, -.6), (11, .1), (12, .7), (13, 1.05), (14, 1.12), (15, 1.0), (18, 1.)], lambda t: t)); x, y = self.x, self.y; u = smooth(cy + 1.9 * s, cy + 5.0 * s, y); v = smooth(cy + 4.4 * s, cy + 6.2 * s, y)
        mx = x - (30. * w * u + 11. * (w2 - w) * v); my = y + 1.9 * br * smooth(cy + 5.2 * s, cy + 2.2 * s, y) + 7. * w * u * np.clip((x - cx) / (2. * s), -1, 1)
        lay = cv2.remap(lay, mx.astype(f32), my.astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE); lay = self.lit(lay, cx, cy, s * 1.6); img = comp.over(bg, lay)
        # breath in the cool air
        if f >= 9: img = comp.screen(img, self.steam(f - 9, cx + 62, cy + 70, 4, 40, 110, .16 * min(1, (f - 9) / 4.), 5) * rgb('#cfd4f2'))
        px = 1960 + 46 * f                                                                                # a shoulder leaves the frame close to the lens
        if f < 6: img = comp.over(img, smear(qb(self.passer(px - 60, 560, 560, '#080a18', .3, math.sin(f)), 20), 40, 0, 7), .95)
        return img

    # ------------------------------------------------------------------ B  18-27  the watch
    def insert(self, f):
        t = f - 18; bg = self.plate('C', 1010 + 1.2 * t, 610, 1.10); bg = qb(bg, 9) * 1.02
        sec = keyed(f, [(18, 47.), (19, 48.35), (20, 47.85), (21, 48.)]); s = 400 + 2.2 * t; dy = 2.2 * math.sin(f * .5)
        c = Cel(); pal = dict(WP); watch(c, 1010, 548 + dy, s, ang=-.15 + .002 * t, hm=(17, 59, sec), pal=pal, lw=6.2, seed=2)
        lay = c.layer(); lay = rim(lay, -4, 6, '#ffdca0', .9, .7); lay = rim(lay, 3, -4, '#5f74c8', .6, .8)
        lay = K.para_lit(lay, (1700, 100), (700, 620), [(0, '#5a3a14'), (.6, '#180e06'), (1, '#000000')], .9, .2, .55); lay = para(lay, (900, 500), (200, 1080), [(0, '#ffffff'), (1, '#8088c4')], 'mul', .85)
        img = comp.over(bg, lay); a = math.exp(-((f - 21.5) / 2.2) ** 2); img = flare(img, 925, 470, 240, '#ffe9c0', .34 * a, .5, -.5)
        return img

    # ------------------------------------------------------------------ C  27-40  the tie
    def tie(self, f):
        d = f // 2 * 2 if f > 28 else f; t = f - 27; bg = self.plate('C', 1040 + .8 * t, 590, 1.0 + .002 * t)
        lo = float(lerp_keys(d, [(27, 0.), (29, .10), (31, .42), (33, .46), (35, .92), (37, 1.06), (39, 1.0)], lambda v: v)); wig = keyed(d, [(27, 0), (29, -.02), (31, .06), (33, -.05), (35, .04), (37, 0)])
        s, cx, cy = 206 + .5 * t, 812, 356; pit = float(lerp_keys(d, [(27, -.04), (31, .015), (33, .0), (35, .035), (38, .0)])); rl = .6 * keyed(d, [(27, 0.), (29, .015), (31, -.03), (33, .012), (35, -.04), (37, -.012)])
        H_, lw, kw = self.his(f, cx + 60 * wig, cy + 1.6 * math.sin((f - 3) / 24. * 2 * math.pi / 2.6), s, pit, rl); b = Body.under(H_, .30, 0, .0); c = Cel(); e = dict(EXPR['deadpan'], open=seq(d, (27, .62), (31, .56), (35, .46), (38, .62)), gaze=(0., -.1))
        lv = max(0., (f - 38) / 2.); W_ = (-.56 + wig * .5 - .25 * lv, -.96 - .24 * lo - .10 * lv, 1.75); E_ = (-2.06 + wig * .3, -2.26 - .05 * lo, .9)
        hC = lambda cel, w, e_, fz: hand3(cel, w, e_, fz, WP, lw, 'R', 'hook', 16 + 60 * wig, 10 + 6 * math.sin(d), 8 + 90 * wig, .66, bld=HD.SLIM, cut=.1, light=(.8, .3, .5))
        Nanami.head(c, H_, e, WP, part='back', **kw); Nanami.draw_suit(c, H_, b, WP, lw, 0, 1, lo, arms=(0, 1)); Nanami.head(c, H_, e, WP, part='front', **kw); Nanami.suit_arm(c, b, (-1.62, -.54, 0.), E_, W_, WP, lw, -1, 1, 3, hand=hC)
        # drag folds of the shirt toward the knot
        kn = b(np.array([(.05 * lo, -.22 - .22 * lo)]), .05)[0]
        for k, (u, v) in enumerate(((-.62, .12), (-.70, -.20), (.60, .10), (.66, -.22), (-.55, -.62), (.58, -.66))):
            p0 = b(np.array([(u * .78, v)]), .05)[0]; p1 = kn * .72 + p0 * .28; c.line(spl([p0, (p0 + p1) / 2 + [0, 6 + 8 * lo], p1], n=7), lw * .42, WP['shirtS'], taper=(.5, .2), alpha=.3 + .6 * min(1, lo), seed=k)
        lay = self.lit(c.layer(), cx, cy, s * 1.3); return comp.over(bg, lay)

    # ------------------------------------------------------------------ D  40-45  the roll
    def roll(self, f):
        t = f - 40; bg = self.plate('C', 1180, 560, 1.35); bg = qb(bg, 5); s = 470; H_ = Nanami.pose(700, -560, s, .40, 0, 0); b = Body.under(H_, .28, 0, 0); c = Cel(); Nanami.draw_suit(c, H_, b, WP, 6.4, 0, 1, 1., neck=True); lay = self.lit(c.layer(), 700, 300, 700); img = comp.over(bg, cv2.GaussianBlur(lay, (0, 0), 2.6))
        lift = keyed(f, [(40, .02), (41, -.06), (42, .34), (43, .74), (44, .66)]); cx, cy, ss = 1010, 640 + 4 * math.sin(f), 300; ang = -.20 + keyed(f, [(40, .0), (41, .03), (42, -.05), (43, -.08), (44, -.06)])
        c = Cel(); sandwich(c, cx, cy, ss, ang, lw=5.4, lift=lift)
        # his hand round the bag, thumb in front
        hp = dict(skin=WP['skin'], shade=WP['skinS'], line=WP['line'], nail=hexmix(WP['skin'], '#ffffff', .22), nail_shade=hexmix(WP['skinS'], '#ffffff', .15), light=WP['skinH'], cuff=WP['suit'], cuff_shade=WP['suitS'], cuff_light=WP['suitH'], cuffin=WP['shirt'])
        cam = HD.Cam(cx + 130, cy + 300, 250, 7.0); HD.draw(c, 'carry', cam, hp, (.8, .3, .5), 6.0, 4, side='L', R=HD.euler(-30, 12, 62), bld=HD.SLIM, style=dict(cuff=dict(y=.36, flare=.9)), res=560)
        lay = c.layer(); lay = rim(lay, -6, 3, '#ffe2ac', .95, .7); lay = para(lay, (cx + 80, 400), (cx - 360, 1000), [(0, '#ffffff'), (.5, '#e6d2d6'), (1, '#8a86b4')], 'mul', .8); img = comp.over(img, lay)
        if f in (42,): img = img * .6 + smear(img, 0, 60, 7) * .4
        return comp.screen(img, self.steam(f + 6, cx - 60 + 60 * lift, cy - 330 - 220 * lift, 8, 170, 520, .40, 2) * rgb('#fff1dc'))

    # ------------------------------------------------------------------ E  45-72  the three in the glass
    def glass(self, f):
        t = f - 45; d = f // 2 * 2; pan = 1120 - .9 * t; z = 1.0 + .0012 * t; bg = self.plate('E', pan, 575, z); gm = np.clip(self.plate('G', pan, 575, z), 0, 1)
        # ---- reflections: only on the glass, never on the painted frame; mirrored, large, soft, a third opaque; they come out from behind his head and walk off
        x, y = self.x, self.y; Lb = (bg @ np.array([.3, .59, .11], f32)); gm = cv2.erode(gm, np.ones((9, 9), np.uint8), iterations=2) * smooth(.13, .26, cv2.GaussianBlur(Lb, (0, 0), 2)); gm = cv2.GaussianBlur(gm, (0, 0), 3); bg = bg + (gm * smooth(900, 1900, x) * .045)[..., None] * rgb('#7f92e0')
        R = np.zeros((H, W, 4), f32); sc = 1.42
        if f < 70:
            for i in range(3):
                j = i * 8 + (f // 2 + 3 * i) % 8; spr = cv2.resize(np.array(self.RF[j], f32)[:, ::-1], None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR); h_, w_ = spr.shape[:2]; px = int(620 - 400 * i + 45. * t - (w_ - 620 * sc) + (self.RO[j][0] - self.RO[i * 8][0]) * -sc - 120); py = int(176 + (self.RO[j][1] - self.RO[8][1]) * sc); x0, y0, x1, y1 = max(0, px), max(0, py), min(W, px + w_), min(H, py + h_)
                if x1 > x0 and y1 > y0: q = spr[y0 - py:y1 - py, x0 - px:x1 - px]; R[y0:y1, x0:x1] = R[y0:y1, x0:x1] * (1 - q[..., 3:4]) + q
            R = cv2.GaussianBlur(R, (0, 0), 2.6); gh = np.zeros_like(R); gh[:, 9:] = R[:, :-9] * .30; R = R + gh * (1 - R[..., 3:4]); op = .35
            bg = bg * (1 - (R[..., 3] * op * gm)[..., None]) + (R[..., :3] * rgb('#c6ceff') * 1.25 + R[..., 3:4] * rgb('#24306a') * .25) * (op * gm)[..., None]
            rl = R[..., 3]; ed = np.clip(rl - cv2.warpAffine(rl, np.float32([[1, 0, -6], [0, 1, 3]]), (W, H)), 0, 1) * gm; bg = bg + ed[..., None] * rgb('#dfe6ff') * .30
        for a, w_, o in ((1250, 120, .06), (1430, 46, .05), (1760, 170, .045)): bg = bg + (np.exp(-(((x - a) + (y - 540) * .55) / w_) ** 4) * gm * o)[..., None] * rgb('#dfe6ff')
        # ---- him
        s, cx, cy = 286 + .40 * t, 560 + .3 * t, 486; e = dict(EXPR['deadpan']); g = keyed(f, [(45, (-.05, -.20)), (47, (.05, -.12)), (48, (.10, .02)), (52, (.22, .02)), (56, (.36, .0)), (60, (.48, .0)), (64, (.30, -.10))]); sm = float(lerp_keys(f // 2 * 2, [(52, 0.), (54, .35), (56, 1.), (58, 1.12), (60, 1.)], lambda v: v)); bl = keyed(f, [(45, 1.), (46, .55), (47, .3), (48, 1.), (61, 1.), (62, .5), (63, .2), (64, .6), (65, 1.)])
        e.update(gaze=g, open=(.62 - .03 * sm) * bl, low=e.get('low', .1) + .08 * sm, crease=.8 * max(0, sm), brow_in=e.get('brow_in', 0) + .014 * sm, mouth='flat')
        if sm > 0: Nanami.MOUTH['corner_'] = (-.004 + .002 * sm, -.004 + .022 * sm, .128, 0.); e['mouth'] = 'corner_'
        H_, lw, kw = self.his(f, cx, cy + 1.6 * math.sin((f - 3) / 24. * 2 * math.pi / 2.6), s, float(lerp_keys(d, [(45, -.07), (48, -.01), (60, .0), (64, -.03), (72, -.03)])), float(lerp_keys(d, [(45, 0), (54, .010), (58, .022)])), .038 * smooth(46, 51, float(f))); b = Body.under(H_, .34, 0, 0); c = Cel()
        Nanami.head(c, H_, e, WP, part='back', **kw); Nanami.draw_suit(c, H_, b, WP, lw, 0, 1, 1.); Nanami.head(c, H_, e, WP, part='front', **kw)
        lay = self.lit(c.layer(), cx, cy, s * 1.15); img = comp.over(bg, lay)
        c = Cel(); sx, sy = 985 + .3 * t, 958 + 5 * math.sin(f * .22); sandwich(c, sx, sy, 128, -.30, lw=4.6, lift=.45); l2 = c.layer(); l2 = rim(l2, -5, 3, '#ffe2ac', .95, .7); l2 = para(l2, (sx + 60, 700), (sx - 200, 1060), [(0, '#ffffff'), (1, '#9a94c0')], 'mul', .7); img = comp.over(img, l2)
        img = comp.screen(img, self.steam(f, sx - 80, sy - 200, 7, 90, 360, .26, 2) * rgb('#fff1dc'))
        return img
