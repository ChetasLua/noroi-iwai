"""SUKUNA lane: the fire-lit TV cel of the same drawing.  Suk.fire() and the fire inking of jj.hands geometry live here."""
import numpy as np, cv2, math, cairo
from jj import W, H
from jj.paint import rgb, f32, smooth, mix
from jj.ink import Cel, spl, resample, noise1, paste
from jj.kit import contours, outline, para
from jj.comp import over, screen, blur
from chars.sukuna import Suk
from chars.sukuna_ink import brush, stroke_poly, at, length

FIRE = dict(line='#1b0b06', skin='#7f672c', deep='#564628', light='#f6c963', hot='#fdeb86', hair='#a35a50', hair_d='#703838', hair_l='#f8d6a2', mark='#24100a', mark_l='#7c1a0c', white='#80814a', white_l='#f7e7a0', iris='#9c1d10', mouth='#3c0d08', teeth='#b09a52', teeth_l='#fbe9a4', cloth='#10130c', cloth_l='#4a2a12')


def rim(L, dx, dy, col, amt=1., soft=.5):
    """kit.rim that does not light the edge of the frame"""
    a = L[..., 3]; h, w = a.shape; m = np.clip(a - cv2.warpAffine(a, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), 0, 1)
    if soft > 0: m = cv2.GaussianBlur(m, (0, 0), soft) * (a > .02)
    o = L.copy(); o[..., :3] = o[..., :3] * (1 - m[..., None] * amt) + rgb(col) * (m * a)[..., None] * amt; return o


def pen(c, P, w, col, seed=0, taper=(.18, .3), wfun=None, smooth_=True):
    P = np.asarray(P, float)
    if len(P) < 2: return
    c.line(spl(P, n=8) if smooth_ and 2 < len(P) < 30 else P, w, col, taper=taper, press=.25, seed=seed, tremor=.2, tmin=.2, wfun=wfun)


DAY = dict(FIRE, line='#17131a', skin='#c08b7d', deep='#9a675f', light='#f1cdb2', hair='#a85f6b', hair_d='#7d3f4f', hair_l='#d98f93', mark='#17131a', mark_l='#17131a', white='#cfc4c4', white_l='#f4efe8', iris='#8e1420', mouth='#3c0d12', teeth='#bdb0aa', teeth_l='#f2ece2', cloth='#15141f', cloth_l='#4a3f7a')
ASH = dict(DAY, skin='#8f8386', deep='#6b6068', light='#d8cdc4', hair='#8a5b66', hair_d='#624250', hair_l='#b98a8c', white='#a9a3a6', white_l='#e6e2dc')
DAWN = dict(DAY, skin='#7f86a8', deep='#5d6488', light='#dfe0ea', hair='#7c6488', hair_d='#54476a', hair_l='#c39fae', white='#9aa2bd', white_l='#eef0f6')
BURN = dict(FIRE, line='#120805', skin='#1c0e08', deep='#0d0705', light='#fff6d6', hot='#ffffff', hair='#6a2f38', hair_d='#3c1a22', hair_l='#fff0cc', mark='#050303', mark_l='#3a0a06', white='#3a120c', white_l='#ffffff', iris='#c8200f', mouth='#0a0403', teeth='#3a1c10', teeth_l='#fffbe8')
BLOODLIT = dict(FIRE, skin='#4a1418', deep='#2a0a0f', light='#c22a2c', hot='#f0645a', hair='#4e1620', hair_d='#2d0c14', hair_l='#b8343c', white='#6d2226', white_l='#f0a090', mark='#12060a', mark_l='#1a080c', teeth='#7a3a38', teeth_l='#f2c2b2')


# light polygons of an under-light on the face, for a light on the side q=+1 (picture right in head space); mirrored by q
def face_lights(q, size=1.):
    f = np.array([q, 1.])
    L = [
        ([(-.16, -1.03), (-.09, -.955), (-.02, -.90), (.06, -.872), (.14, -.842), (.20, -.770), (.222, -.700), (.262, -.610), (.232, -.520), (.168, -.452), (.232, -.386), (.300, -.300), (.400, -.232), (.520, -.196), (.600, -.140), (.640, -.02), (.665, .12), (.82, .14), (.84, -1.06)], (0, 3, 5, 7, 9, 12, 15, 16, 17, 18)),
        ([(-.040, -.480), (-.062, -.452), (-.026, -.412), (.026, -.372), (.040, -.260), (.058, -.120), (.078, -.02), (.118, -.10), (.146, -.26), (.156, -.40), (.128, -.470), (.050, -.492)], (0, 2, 6, 10)),
        ([(.016, -.560), (.084, -.540), (.150, -.545), (.112, -.606), (.030, -.616)], (0, 2, 3)),
        ([(.150, -.010), (.240, .018), (.350, .044), (.460, .060), (.552, .066), (.600, .112), (.520, .120), (.400, .098), (.270, .058), (.190, .030)], (0, 4, 5)),
        ([(.330, .210), (.480, .232), (.600, .222), (.650, .290), (.640, .420), (.600, .520), (.560, .440), (.500, .350), (.410, .270)], (0, 3, 5)),
        ([(-.090, -.720), (.02, -.706), (.110, -.716), (.070, -.752), (-.02, -.764)], (0, 2)),
        ([(-.86, -1.10), (-.82, -.40), (-.690, -.420), (-.610, -.470), (-.530, -.560), (-.455, -.680), (-.350, -.790), (-.215, -.900), (-.10, -.960), (-.08, -1.08)], (0, 1, 2, 3, 8)),
        ([(-.300, -.262), (-.400, -.250), (-.500, -.206), (-.450, -.262), (-.360, -.296)], (0, 2)),
        ([(-.170, -.004), (-.260, .014), (-.350, .030), (-.430, .030), (-.350, .046), (-.240, .034)], (0, 3)),
    ]
    return [(np.array(p) * f, c) for p, c in L]


def _locks_fire(c, k, locks, C, lit, lw, box):
    """flat clumps; the flank that looks at the flame takes a hard light shape near the tip"""
    for o in locks:
        cc = o['c']; poly, N, hl, hr = k.lock_poly(cc, o['w'], o['asym'], o['split']); lo, hi = poly.min(0), poly.max(0); n = len(cc); r = np.random.default_rng(500 + o['k'])
        if hi[0] < box[0] or lo[0] > box[2] or hi[1] < box[1] or lo[1] > box[3]: continue
        base = C['hair_d'] if o['face'] < .0 else C['hair']
        c.fill(poly, base); c.ctx.save(); c._path(poly); c._src(base, 1.); c.ctx.set_line_width(1.4); c.ctx.stroke(); c.ctx.restore()
        d = N[int(n * .6)] @ lit; tipd = (cc[-1] - cc[n // 3]); tipd = tipd / (np.hypot(*tipd) + 1e-6) @ lit
        s = 1. if d > 0 else -1.; amt = max(abs(d), .0) * .55 + max(tipd, 0) * .75
        if amt > .34 and o['face'] > -.6:
            hw = hl if s > 0 else hr; i0 = int(n * float(np.clip(.62 - .40 * amt + r.normal(0, .06), .12, .8))); e = cc + N * (s * hw * (.45 - .55 * (amt - .34) + .12 * np.sin(np.arange(n) * 1.3 + o['k'])))[:, None]; ed = cc + N * (s * hw * 1.02)[:, None]
            c.fill(np.vstack([ed[i0:], e[i0 + 2:][::-1]]), C['hair_l'])
        # line work: both flanks of the free part, thin, lifted near the root
        w = 1.9 * lw; m_ = len(poly); ia = int(n * r.uniform(.34, .52)); ib = m_ - int(n * r.uniform(.36, .56)); run = poly[ia:ib]; T = np.diff(run, axis=0); an = np.unwrap(np.arctan2(T[:, 1], T[:, 0])); dv = np.abs(np.diff(an)); cut = [0] + [int(v) + 1 for v in np.nonzero(dv > .8)[0]] + [len(run) - 1]
        for q, (i, j) in enumerate(zip(cut[:-1], cut[1:])):
            if j - i >= 1: c.line(run[i:j + 1], w, C['line'], taper=(.45 if q == 0 else .04, .45 if q == len(cut) - 2 else .04), press=.2, seed=o['k'] + q * 31, tremor=.12, tmin=.12 if q in (0, len(cut) - 2) else .5)
        if r.random() < .22 and o['face'] > .1:
            q = cc + N * (hl * r.uniform(-.3, .3))[:, None]; a = int(n * r.uniform(.28, .5)); c.line(q[a:a + int(n * .22)], w * .8, C['line'], taper=(.4, .4), press=.2, seed=o['k'] + 5, tremor=.15)


def fire(self, ls=1, pal=None, light_vec=None, box=None, head_only=False, rimlight=True, mode='under', rim_col='#ffe98a'):
    """fire-lit cel.  ls = side of the flame in head space (+1 = picture right at yaw 0); light_vec = screen direction TO the flame"""
    C = dict(FIRE, **(pal or {})); lw = self.lw; s = self.s; H_ = self.H; F = self.C; P = self.P; sd = self.seed
    b = box or self.box(); x0, y0, x1, y1 = [int(v) for v in b]; x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1); c = Cel(x1 - x0, y1 - y0, x0, y0); bx = (x0, y0, x1, y1)
    lit = np.asarray(light_vec if light_vec is not None else (ls * .75, .66), float); lit = lit / np.hypot(*lit)
    back, low, front = self.split_locks(); face = self.mould(H_.face()); skull = H_.skull()
    _locks_fire(c, self, back, C, lit, lw, bx)
    c.fill(skull, C['hair_d'])
    jl = np.vstack([H_.jaw(-1), H_.jaw(1)[::-1][1:]])
    if not head_only:
        nl, nr = self.neck_lines(.45); NL, NR = spl(nl, n=6), spl(nr, n=6); nk = np.vstack([NL, NR[::-1]]); c.fill(nk, C['skin'])
        E, G = (NR, NL) if lit[0] > 0 else (NL, NR); m = len(E); ju = jl[np.linspace(0, len(jl) - 1, 11).astype(int)]; dn = -self.up
        c.clip(nk); c.fill(np.vstack([E[int(m * .08):], (E * .46 + G * .54)[int(m * .30):][::-1]]), C['light'])
        c.fill(np.vstack([jl, spl(ju + dn * np.array([.5, .34, .28, .40, .24, .30, .22, .34, .28, .20, .3])[:, None] * .75, corners=(3, 7), n=5)[::-1]]), C['deep']); c.unclip()
        pen(c, NL, 3.0 * lw, C['line'], sd); pen(c, NR, 3.0 * lw, C['line'], sd + 1)
        for q in (-1, 1): pen(c, H_.P3(np.array([(q * .36, -.92, .06), (q * .20, -1.28, .12), (q * .07, -1.72, .20)])), 1.8 * lw, C['line'], sd + 3 + q, (.5, .3))
    if self.pitch > .02: c.fill(H_.under(), C['deep'])
    _locks_fire(c, self, low, C, lit, lw, bx)
    for q in (-1, 1):
        er = H_.ear(q)
        if er['vis'] < .02: continue
        c.fill(self.side_skin(q), C['skin']); o, i, tr = er['outer'], er['inner'], er['tragus']; c.fill(o, C['skin']); c.fill(i, C['deep']); n = len(o)
        c.clip(o); c.fill(np.vstack([o[int(n * .02):int(n * .42)], (o * .72 + o.mean(0) * .28)[int(n * .02):int(n * .42)][::-1]]), C['light'] if q * ls > 0 else C['skin']); c.unclip()
        pen(c, o, 2.8 * lw, C['line'], sd + 6 + q); pen(c, i[:int(len(i) * .62)], 1.7 * lw, C['line'], sd + 8 + q, (.3, .4)); pen(c, tr[:int(len(tr) * .6)], 1.6 * lw, C['line'], sd + 10 + q, (.3, .4))
    if mode == 'half':                                     # the arrow carves him: one side burnt white, the other black
        q = ls; c.fill(face, C['skin']); c.clip(face)
        half = [(q * .90, .95), (q * .10, .95), (q * .16, .62), (q * .06, .40), (q * .13, .24), (q * .040, .10), (-q * .006, .02), (q * .030, -.10), (q * .022, -.26), (q * .048, -.40), (-q * .020, -.455), (q * .026, -.500), (q * .010, -.585), (q * .050, -.640), (-q * .012, -.700), (q * .036, -.770), (-q * .03, -.860), (q * .03, -.940), (-q * .02, -1.10), (q * .90, -1.10)]
        c.fill(F(half, closed=True, corners=tuple(range(20)), n=3), C['light'])
        # what the light does not reach on the burnt side: socket, under the cheekbone, corner of the mouth, under the lip
        for p_, cs in (([(q * .060, .10), (q * .20, .098), (q * .40, .160), (q * .60, .186), (q * .54, .10), (q * .40, .070), (q * .24, .026), (q * .150, -.045), (q * .112, -.09)], (0, 3, 8)), ([(q * .68, -.16), (q * .58, -.22), (q * .46, -.33), (q * .37, -.46), (q * .345, -.58), (q * .42, -.53), (q * .53, -.42), (q * .64, -.32), (q * .72, -.24)], (0, 4, 8)), ([(q * .02, -.735), (q * .12, -.742), (q * .07, -.790), (q * .02, -.80)], (0, 1)), ([(q * .62, -.44), (q * .54, -.60), (q * .42, -.78), (q * .27, -.94), (q * .60, -1.0), (q * .80, -.6)], (0, 3, 4))): c.fill(F(p_, closed=True, corners=cs), C['skin'])
        c.fill(F([(-q * .30, -.262), (-q * .41, -.250), (-q * .52, -.204), (-q * .45, -.256), (-q * .36, -.290)], closed=True, corners=(0, 2)), C['light'])      # one splinter on the black side
        self.lit_face = [F(half, closed=True, corners=tuple(range(20)), n=3)]
    elif mode == 'top':                                      # key light from above: light base, designed shadows
        d = -ls; c.fill(face, C['light']); c.clip(face)
        c.fill(F([(d * .84, .95), (d * .26, .74), (d * .34, .46), (d * .27, .28), (d * .46, .22), (d * .56, .06), (d * .50, -.10), (d * .58, -.22), (d * .46, -.33), (d * .37, -.46), (d * .355, -.60), (d * .27, -.70), (d * .25, -.82), (d * .14, -.93), (d * .10, -1.1), (d * .84, -1.1)], closed=True, corners=(0, 1, 3, 4, 7, 11, 14, 15), n=8), C['skin'])
        if (H_.R @ np.array([d, 0., 0.]))[2] > .35: c.fill(H_.P3(np.array([(d * .70, .70, .12), (d * .70, .60, -.20), (d * .74, .0, -.22), (d * .73, -.40, -.16), (d * .66, -.62, .02), (d * .69, -.30, .18), (d * .70, .10, .18)])), C['skin'])
        c.fill(F([(-.9, .95), (.9, .95), (.9, .42), (.60, .46), (.48, .50), (.38, .58), (.24, .575), (.10, .53), (.0, .47), (-.14, .54), (-.28, .575), (-.40, .55), (-.50, .46), (-.62, .40), (-.9, .40)], closed=True, corners=(0, 1, 2, 6, 8, 10, 14)), C['skin'])
        for q in (-1, 1):
            if self.vis(q * .34, 0) > .05: c.fill(F([(q * .066, .10), (q * .20, .10), (q * .40, .160), (q * .58, .176), (q * .55, .08), (q * .40, .060), (q * .24, .018), (q * .146, -.05), (q * .128, -.14), (q * .098, -.20), (q * .080, -.05)], closed=True, corners=(0, 3, 9)), C['skin'] if q == ls else C['deep'])
        c.fill(F([(d * .015, .02), (d * .058, -.12), (d * .05, -.34), (d * .012, -.43), (d * .10, -.475), (d * .145, -.43), (d * .12, -.30), (d * .10, -.12), (d * .105, .04)], closed=True, corners=(0, 3, 8)), C['skin'] if d * self.yaw <= 0.2 else C['deep'])
        c.fill(F([(-.085, -.468), (0, -.452), (.085, -.468), (.04, -.515), (d * .03, -.555), (-.05, -.515)], closed=True, corners=(0, 2, 4)), C['skin']); c.fill(F([(-.15, -.742), (0, -.728), (.15, -.742), (.06, -.790), (d * .02, -.815), (-.08, -.785)], closed=True, corners=(0, 2, 4)), C['skin'])
        self.lit_face = []
    else:
        c.fill(face, C['skin']); c.clip(face)
        # top plane sinks: forehead, upper cheek hollow
        c.fill(F([(-.9, .95), (.9, .95), (.9, .50), (.40, .40), (.0, .46), (-.42, .34), (-.9, .30)], closed=True, corners=(0, 1, 2, 6)), C['deep'])
        for q in (-1, 1):
            if self.vis(q * .34, 0) > .05: c.fill(F([(q * .066, .10), (q * .20, .10), (q * .40, .168), (q * .60, .200), (q * .56, .09), (q * .40, .066), (q * .24, .022), (q * .146, -.05), (q * .120, -.10), (q * .085, -.02)], closed=True, corners=(0, 3, 8)), C['deep'])
        c.fill(F([(-.13, -.46), (0, -.45), (.13, -.46), (.10, -.49), (.0, -.50), (-.10, -.49)], closed=True), C['deep'])
        for q in (-1, 1):                                  # hollow under the cheekbone
            if self.vis(q * .5, -.3) > .05: c.fill(F([(q * .66, -.12), (q * .56, -.20), (q * .44, -.315), (q * .345, -.440), (q * .315, -.560), (q * .39, -.520), (q * .50, -.415), (q * .62, -.31), (q * .70, -.22)], closed=True, corners=(0, 4, 8)), C['deep'])
        for p, cs in face_lights(ls): c.fill(F(p, closed=True, corners=cs, n=7), C['light'])
        self.lit_face = [F(p, closed=True, corners=cs, n=7) for p, cs in face_lights(ls)]
    hp, hpoly = self.hairline(); c.fill(hpoly, C['hair_d']); c.unclip()
    # ------------------------------------------------ features
    L = lambda Q, w, k=0, tp=(.2, .3), sm=True: pen(c, Q, w * lw * self.bg ** .6, C['line'], sd + 20 + k, tp, smooth_=sm)
    mk, pet = self.marks_screen(); self.petals = pet
    if self.marks > 0:
        c.clip(face)
        for nm, m in mk: c.fill(m, C['mark'], self.marks)
        for lp in self.lit_face:
            c.clip(lp)
            for nm, m in mk: c.fill(m, C['mark_l'], self.marks)
            c.unclip()
        c.unclip()
    pr = H_.P3(H_.nose_profile()); far = 1 if self.yaw > 0 else -1
    if abs(self.yaw) < 1.3:
        if abs(self.yaw) > .16:
            L(spl(pr[2:6], n=6), 2.2, 1, (.6, .05), False); L(spl(pr[4:7], n=6), 2.6, 2, (.05, .5), False); q = -far
            L(F([(q * .052, -.462), (q * .092, -.470), (q * .126, -.436), (q * .122, -.385)], dz=.03), 2.0, 3, (.3, .6)); c.fill(F([(q * .012, -.468), (q * .050, -.455), (q * .084, -.474), (q * .046, -.486)], closed=True, dz=.06), C['line'])
        else:
            L(F([(-.030, -.482), (0, -.492), (.030, -.482)], dz=.06), 2.2, 1)
            for q in (-1, 1): L(F([(q * .100, -.472), (q * .128, -.440), (q * .122, -.395)], dz=.02), 2.0, 2 + q, (.3, .6)); c.fill(F([(q * .040, -.466), (q * .062, -.457), (q * .084, -.466), (q * .060, -.478)], closed=True, dz=.04), C['line'])
    for q in (-1, 1):
        if self.vis(q * .34, -.02) < .06: continue
        E = self.eye_lines(q); up = spl(E['up'], n=8); lo = spl(E['lo'], n=8); isl = q == ls or mode == 'top'
        if E['o'] > .05:
            OP = P(np.vstack([up, lo[::-1]])); c.fill(OP, C['white_l'] if isl else C['white']); c.clip(OP); lx, ly = self.e['look']
            cc = np.array([q * .348 + lx * .105, -.018 + ly * .05]); rx, ry = .046, .058; a = np.linspace(0, 6.283, 26, endpoint=False); ring = lambda k_: P(np.c_[cc[0] + np.cos(a) * rx * k_, cc[1] + np.sin(a) * ry * k_])
            c.fill(P(np.vstack([up, (up - (0, .030))[::-1]])), C['deep'] if not isl else '#c79a4a')
            c.fill(ring(1.), C['iris']); c.fill(ring(.72), '#d8391a' if isl else '#b3280f'); c.fill([ring(1.06), ring(.90)], C['line'], evenodd=True)
            c.fill(P(spl(np.array([(cc[0], cc[1] + ry * .9), (cc[0] + rx * .30, cc[1]), (cc[0], cc[1] - ry * .62), (cc[0] - rx * .28, cc[1])]), True, (0, 2), n=6)), C['line'])
            c.fill(P(spl(np.array([(cc[0] + ls * rx * .40, cc[1] - ry * .40), (cc[0] + ls * rx * .56, cc[1] - ry * .28), (cc[0] + ls * rx * .44, cc[1] - ry * .14), (cc[0] + ls * rx * .30, cc[1] - ry * .28)]), True, n=4)), '#fff3b8')
            c.unclip()
        fl = np.vstack([E['inn'] + (-q * .026, -.024), E['up'], E['out'] + (q * .045, .022)]); t_ = 5.6 if E['o'] > .05 else 4.2
        c.line(P(spl(fl, n=8)), t_ * lw * self.bg ** .6, C['line'], taper=(.22, .16), press=.15, seed=sd + 40 + q, tremor=.15, tmin=.10)
        L(P(lo[int(len(lo) * .40):int(len(lo) * .98)]), 1.7, 44 + q, (.6, .1), False); cr = spl(E['up'][1:] + (0, .036), n=8); L(P(cr[:int(len(cr) * .86)]), 1.5, 46 + q, (.3, .5), False)
        # second eye
        if self.marks > 0 and self.vis(q * .40, -.2) > .06:
            E2 = self.eye2_lines(q); u2 = spl(E2['up'], n=8); l2 = spl(E2['lo'], n=8)
            if E2['o'] > .05:
                O2 = P(np.vstack([u2, l2[::-1]])); c.fill(O2, C['white_l'] if isl else C['white']); c.clip(O2); lx, ly = self.e['look']; cc = np.array([q * .398 + lx * .05, -.212 + ly * .02]); R = P(np.c_[cc[0] + np.cos(a) * .034, cc[1] + np.sin(a) * .040]); c.fill(R, C['iris']); c.fill(P(np.c_[cc[0] + np.cos(a) * .011, cc[1] + np.sin(a) * .026]), C['line']); c.unclip()
            c.line(P(spl(np.vstack([E2['a'] + (-q * .03, -.004), E2['up'], E2['b'] + (q * .04, .028)]), n=8)), 3.6 * lw * self.bg ** .6, C['line'], taper=(.3, .3), press=.15, seed=sd + 50 + q, tremor=.15, tmin=.08); L(P(l2[int(len(l2) * .3):]), 1.4, 52 + q, (.5, .3), False)
        # bag line and crow's foot
        L(F([(q * .150, -.130), (q * .240, -.170), (q * .330, -.176)]), 1.5, 56 + q, (.3, .5)); L(F([(q * .58, -.02), (q * .63, -.05), (q * .66, -.10)]), 1.4, 58 + q, (.3, .5))
        if self.vis(q * .34, .20) > .05:
            bp = P(self._brow_poly(q)); c.fill(bp, C['mark'])
    if self.e['brow'][0] < -.015:
        for q in (-1, 1): L(F([(q * .036, .17), (q * .052, .11), (q * .044, .05)]), 1.8, 60 + q, (.3, .4))
    self._mouth_fire(c, C, ls, L)
    # ------------------------------------------------ contour, hair in front, hairline zigzag
    run = self._contour(face); n = len(run); y = self.head_y(run); kc = int(np.clip(np.argmin(y), 8, n - 8))
    c.line(run[:kc + 2], 3.8 * lw, C['line'], taper=(.10, .05), press=.3, seed=sd + 70, tremor=.2, tmin=.25); c.line(run[kc - 2:][::-1], 3.8 * lw, C['line'], taper=(.10, .05), press=.3, seed=sd + 71, tremor=.2, tmin=.25)
    _locks_fire(c, self, front, C, lit, lw, bx)
    # hairline: broken into clumps of different width, some combed up, some hanging, with skin showing between
    r = np.random.default_rng(8); ok = np.array([self.vis(x, y_) > .14 for x, y_ in spl(self.HL, n=6)]); h2 = hp[ok] if ok.sum() > 6 else hp; h2 = resample(h2, 2.); c0 = H_.P3(np.array([0, .22, .6])); z = [h2[0]]; i = 0; lines = []; cur = [h2[0]]
    while i < len(h2) - 4:
        big = r.random(); wd = s * (.115 if big < .22 else .060 if big < .55 else .026) * r.uniform(.75, 1.25); j = min(len(h2) - 1, i + max(3, int(wd / 2))); d = h2[(i + j) // 2] - c0; d /= np.hypot(*d) + 1e-6
        hang = wd * r.uniform(.45, .95) * (1 if r.random() < .72 else -.35); sk = r.uniform(.25, .75); tip = h2[i] * (1 - sk) + h2[j] * sk - d * hang; back_ = h2[j] + d * wd * r.uniform(.05, .35)
        sub = spl(np.array([cur[-1], (cur[-1] + tip) / 2 + d * wd * .10, tip]), n=4); sub2 = spl(np.array([tip, (tip + back_) / 2 + d * wd * .06, back_]), n=4); z += list(sub[1:]) + list(sub2[1:]); cur = [back_]
        if big < .55 or r.random() < .6: lines.append(np.vstack([sub, sub2[1:]]))
        if big < .30: m_ = (h2[i] + h2[j]) / 2 + d * wd * .1; lines.append(np.array([m_, m_ + d * wd * r.uniform(.6, 1.2) + r.normal(0, wd * .12, 2)]))
        i = j + (int(wd * r.uniform(.1, .5) / 2) if r.random() < .35 else 0)
    if len(z) > 4:
        z = np.array(z); c.fill(np.vstack([z, (z + (z - c0) / np.hypot(*(z - c0).T)[:, None] * s * .12)[::-1]]), C['hair_d'])
        for k_, l in enumerate(lines):
            m_ = len(l) // 2
            if len(l) > 4: c.line(l[:m_ + 1], 1.9 * lw, C['line'], taper=(.4, .05), press=.25, seed=sd + k_, tremor=.12, tmin=.15); c.line(l[m_ - 1:], 1.7 * lw, C['line'], taper=(.05, .5), press=.25, seed=sd + k_ + 400, tremor=.12, tmin=.15)
            else: c.line(l, 1.4 * lw, C['line'], taper=(.3, .6), press=.2, seed=sd + k_, tremor=.1)
    for g, dd in self.petals: c.fill(g, C['mark'], 1 - max(0, dd - .75) * 4)
    out = np.zeros((H, W, 4), f32); paste(out, c.rgba(), x0, y0)
    if rimlight: out = rim(out, -lit[0] * 5.5 * lw, -lit[1] * 5.5 * lw, rim_col, .95, .5)
    return out


def _mouth_fire(self, c, C, ls, L):
    if self.vis(0, -.64) < -.15: return
    F = self.C; P = self.P; lw = self.lw; m, D = self.mouth(); cl, cr = np.array(D['cl']), np.array(D['cr'])
    if D['lo'] is None:
        L(F([cl] + D['up'] + [cr]), 2.6, 80, (.12, .12)); L(F([(-.06, -.742), (0, -.750), (.06, -.742)]), 1.6, 81, (.4, .4)); L(F([cr + (.02, .03), cr + (.034, -.01), cr + (.02, -.05)]), 1.8, 82, (.4, .4)); return
    up = spl(np.array([cl] + D['up'] + [cr]), n=8); lo = spl(np.array([cl] + D['lo'] + [cr]), n=8); U, Lo = P(up), P(lo); poly = np.vstack([U, Lo[::-1]]); c.fill(poly, C['mouth']); c.clip(poly)
    gap = {'smile': (.56, .64), 'grit': (.50, .54), 'laugh': (.26, .84)}[m]; T = np.linspace(0, 1, 40); a, b = at(up, T), at(lo, T)
    for k, (g0, g1) in enumerate(((.0, gap[0]), (gap[1], 1.))):
        q = P(np.vstack([a + (b - a) * g0, (a + (b - a) * g1)[::-1]])); c.fill(q, C['teeth']); i = slice(int(40 * .42), 40) if ls > 0 else slice(0, int(40 * .58)); q = P(np.vstack([(a + (b - a) * g0)[i], (a + (b - a) * g1)[i][::-1]])); c.fill(q, C['teeth_l'])
    if m == 'laugh': c.fill(F([(-.16, -.70), (-.06, -.655), (.04, -.660), (.14, -.700), (.12, -.775), (0, -.80), (-.11, -.775)], closed=True), '#a8281a')
    r = np.random.default_rng(self.seed + 8)
    for k, t in enumerate(np.linspace(.07, .93, 11) + r.normal(0, .008, 11)):
        p, q = at(up, t), at(lo, t); e = gap[0] * (1. if abs(t - .5) > .30 else r.uniform(.45, .8)); L(P(np.array([p, p + (q - p) * e * .5, p + (q - p) * e])), 1.4, 90 + k, (.05, .7))
    for k, t in enumerate(np.linspace(.11, .89, 9) + r.normal(0, .01, 9)):
        p, q = at(up, t), at(lo, t); e = (1 - gap[1]) * (1. if abs(t - .5) > .30 else r.uniform(.3, .6)); L(P(np.array([q, q + (p - q) * e * .5, q + (p - q) * e])), 1.2, 110 + k, (.05, .7))
    for g in gap: L(P(a + (b - a) * g), 1.6, 130 + int(g * 10), (.1, .1), False)
    c.unclip()
    L(np.vstack([F([cl + (-.04, .03)]), U, F([cr + (.045, .034)])]), 3.4, 84, (.10, .10), False); n = len(Lo); L(Lo[:int(n * .42)], 2.2, 85, (.1, .5), False); L(Lo[int(n * .52):], 2.4, 86, (.5, .1), False)
    for c_, q in ((cl, -1), (cr, 1)):
        L(F([c_ + (q * .030, .085), c_ + (q * .062, .010), c_ + (q * .044, -.070)]), 2.2, 88 + q, (.4, .4)); L(F([(q * .115, -.405), (q * .175, -.440), (q * .250, -.495), c_ + (q * .046, .055)]), 2.2, 92 + q, (.4, .4))
    L(F([(-.09, -.800), (0, -.815), (.09, -.806)]), 1.9, 96, (.4, .4))


def cel(self, pal=None, ls=-1, **k):
    """ordinary TV cel (key light from above on side ls): pal = DAY (default) | ASH | DAWN | BLOODLIT or a dict"""
    k.setdefault('light_vec', (ls * .8, -.6)); k.setdefault('rimlight', False)
    return fire(self, ls=ls, pal=pal or DAY, mode='top', **k)


Suk.fire = fire; Suk.cel = cel; Suk._mouth_fire = _mouth_fire


# ------------------------------------------------------------------ hands
def fire_hands(c, g, lw=1., seed=3, C=None, hot=True):
    """thin-line fire cel of a jj.hands Drawn: olive base (shadow), cream light where the form is lit"""
    C = dict(FIRE, **(C or {}))
    for G in g.groups:
        if not G['polys']: continue
        nail = G['name'].startswith('nail'); col = '#160a08' if nail else C['light']; c.fill(G['polys'], col, evenodd=True); c.ctx.set_line_width(1.8); c._path(G['polys']); c._src(col, 1.); c.ctx.stroke()
        S_ = g.shadow.get(G['id'])
        if nail or not S_: continue
        c.ctx.save(); c._path(G['polys']); c.ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD); c.ctx.clip(); c.fill(S_, C['skin'], evenodd=True); c.ctx.restore()
    if getattr(g, 'seams', None): c.fill(g.seams, C['skin'], evenodd=True)
    byid = {G['id']: G for G in g.groups}
    for b in g.bands + g.nails:
        G = byid.get(b['gid'])
        if G is None: continue
        c.ctx.save(); c._path(G['polys']); c.ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD); c.ctx.clip(); c.fill(b['polys'], '#3a2410' if 'finger' not in b else '#160a08', evenodd=True); c.ctx.restore()
    for j, k in enumerate(g.creases):
        if k['kind'] != 'fold' and len(k['P']) > 3: c.line(k['P'], 1.9 * lw * k['w'] + .4, C['line'], taper=(.3, .5), press=.2, seed=seed * 17 + j, tremor=.2, tmin=.12)
    for j, st in enumerate(sorted(g.strokes, key=lambda s: (s['kind'] == 'out', float(np.mean(s['z']))))):
        P = st['P']; n = len(P)
        if n < 6: continue
        wt = np.asarray(st['wt'], float); wt = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(wt)), wt); w = (3.2 if st['kind'] == 'out' else 2.2) * lw
        Q = np.vstack([P, P[:1]]) if st['closed'] else P; wt = np.r_[wt, wt[:1]] if st['closed'] else wt
        c.line(Q, w, C['line'], taper=(0, 0) if st['closed'] else (.10, .12), press=.2, seed=seed * 131 + j, tremor=.2, wfun=lambda t, b=wt: np.interp(t, np.linspace(0, 1, len(b)), b), tmin=.25)
    return c


def fire_bare(c, C=None, T=None, lw=1.2, seed=5):
    """bare shoulders with a torn black shirt, bust framing (design px of the fire key), flame on the lower right"""
    C = dict(FIRE, **(C or {})); T = T or (lambda p: np.asarray(p, float)); S = lambda L, cs=(), n=9: T(spl(np.array(L, float), True, cs, n)); O = lambda L, n=9: T(spl(np.array(L, float), n=n))
    pl = lambda L, w, k=0, tp=(.15, .3): pen(c, O(L), w * lw, C['line'], seed + k, tp, smooth_=False)
    body = [(560, 800), (520, 880), (430, 926), (318, 942), (214, 972), (120, 1030), (60, 1110), (1180, 1110), (1140, 1050), (1060, 1004), (960, 980), (880, 962), (836, 900)]
    P = S(body, (6, 7)); c.fill(P, C['skin']); c.clip(P)
    # light: chest and far shoulder, broken by the collar bones and the neck's shadow
    c.fill(S([(836, 900), (880, 962), (960, 980), (1060, 1004), (1140, 1050), (1200, 1120), (820, 1120), (850, 1060), (800, 1010), (760, 990), (790, 950)], (0, 5, 6, 8)), C['light'])
    c.fill(S([(560, 930), (640, 985), (720, 1004), (760, 1030), (700, 1040), (600, 1010), (520, 966)], (0, 3)), C['light']); c.fill(S([(318, 944), (400, 952), (470, 980), (420, 990), (340, 980)], (0, 2)), C['light'])
    c.fill(S([(60, 1110), (120, 1030), (214, 972), (290, 950), (250, 1000), (190, 1050), (170, 1110)], (0, 3, 6)), C['deep'])
    c.unclip()
    pl([(500, 898), (560, 960), (650, 1010), (760, 1032)], 2.6, 1); pl([(836, 960), (800, 1000), (770, 1030)], 2.2, 2); pl([(430, 940), (480, 990), (500, 1050)], 1.8, 3, (.4, .5))
    # torn shirt: two straps with ripped edges
    r = np.random.default_rng(seed)
    def torn(A, B):
        z = resample(O(A), 26.); z = z + r.normal(0, 7, z.shape); q = resample(O(B), 30.); q = q + r.normal(0, 8, q.shape); return np.vstack([z, q[::-1]])
    for A, B in (([(228, 968), (300, 1000), (360, 1060), (380, 1120)], [(100, 1040), (90, 1080), (60, 1120)]), ([(900, 968), (930, 1020), (930, 1120)], [(1080, 1010), (1130, 1050), (1180, 1120)])):
        K = torn(A, B); c.fill(K, C['cloth']); c.line(resample(O(A), 8.), 3 * lw, '#050403', press=.5, seed=seed); c.fill(S([(A[1][0] - 8, A[1][1] + 14), (A[2][0] - 10, A[2][1]), (A[2][0] - 14, A[2][1] + 44), (A[1][0] - 40, A[1][1] + 50)], (0, 2)), C['cloth_l'])
    pl([(560, 800), (520, 880), (430, 926), (318, 942), (214, 972), (120, 1030), (60, 1110)], 3.6, 6, (.1, .1)); pl([(836, 900), (880, 962), (960, 980), (1060, 1004), (1140, 1050), (1180, 1110)], 3.0, 7, (.1, .1))
    return c


def fire_ground(seed=3, glow=((1500, 860, 900, 1.0),), warm=1.0):
    """orange-red air: dark corner, soft far flames, smoke"""
    from jj.comp import radial
    from jj.paint import ramp, fbm, grid
    x, y = grid(); n = fbm(H, W, (320, 120, 46), (1, .6, .35), seed); g = sum(radial(a, b, r_, power=1.3) * k for a, b, r_, k in glow)
    t = np.clip(.10 + .78 * g + .07 * n, 0, 1); im = ramp(t, [(0, '#2a0a08'), (.25, '#6b1a0b'), (.48, '#b5330a'), (.72, '#e2560c'), (1, '#f6a02a')])
    from jj import fx_fire as FF
    r = np.random.default_rng(seed); cel = Cel()
    for k in range(7):
        a, b, r_, _ = glow[k % len(glow)]; px, py = a + r.normal(0, r_ * .5), b + r.uniform(.0, .5) * r_; L = r.uniform(260, 620); th = -1.35 + r.normal(0, .35); d = np.array([math.cos(th), math.sin(th)]); nn = np.array([-d[1], d[0]])
        FF.flame_stream(cel, spl(np.array([(px, py), (px, py) + d * L * .35 + nn * L * .10, (px, py) + d * L * .7 - nn * L * .06, (px, py) + d * L]), n=8), r.uniform(26, 60), .3 * k, 60 + k, ink=False)
    from jj.comp import bokeh
    L_ = cv2.resize(bokeh(cv2.resize(cel.layer(), (W // 2, H // 2), interpolation=cv2.INTER_AREA), 8.), (W, H), interpolation=cv2.INTER_LINEAR); return (im * (1 - L_[..., 3:4] * .8) + L_[..., :3] * .8).astype(f32)


def head_fn(style='cel', pal=None, expr='cold', **k):
    """adapter for jj.figure.Fig.draw(head_fn=...): style 'ink' | 'fire' | 'cel'.  Returns fn(Head, info) -> full-frame RGBA"""
    def fn(H_, info):
        q = Suk.of(H_, expr=expr, **k)
        if style == 'ink': return q.ink(head_only=True)
        if style == 'fire': return q.fire(head_only=True)
        return q.cel(pal=pal, head_only=True)
    return fn
