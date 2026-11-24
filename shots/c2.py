"""c2 EYES IN THE DARK (72 f).  Yuji close on the dead platform; the eyes of curses open behind him on every beat; 54 a stitched hand settles on his shoulder; 60 his pupils shrink."""
import numpy as np, math, cv2, os
from jj import comp, W, H, hands as HD, world
from jj.ink import Cel, spl
from jj.kit import shake
from jj.paint import rgb, fbm, smooth, ramp, grid, f32
from chars import yuji as Y
from chars.yuji import Yuji, YHead, mixc
from chars import yuji_eyes as E
A = np.array

# kind, x, y, s, frame it opens, rotation, depth (0 far behind the columns, 1 mid, 2 huge near), hue of its light, way its body runs (dx, dy)
EYES = [('points',  452,  402,  30,  0,  .0, 0, '#ffe0a0', (0, 1)),
        ('slit',    748,  300,  46,  9, -.12, 0, '#d8c23a', (.3, 1)), ('human',   262,  236,  62,  9,  .10, 0, '#b8d0d8', (-.4, 1)),
        ('goat',    214,  560, 122, 18,  .08, 1, '#e8ac3c', (-1, .5)), ('spider',  566,  132,  58, 18, .0, 1, '#c01822', (0, -1)), ('fish',   432,  690,  44, 18, .0, 0, '#c9d6d8', (.2, 1)),
        ('ring',    640,  520,  84, 27,  .0, 1, '#e0b8a8', (0, .2)), ('weeping',  64,  300, 150, 27, -.14, 2, '#cfd8e4', (-1, 0)), ('double',  388,   84,  58, 27, .20, 0, '#e8b0a0', (0, -1)),
        ('compound',  96,  898, 250, 36,  .16, 2, '#5fe0aa', (-.6, 1)), ('crescent', 336, 440,  54, 36, -.25, 0, '#f0dcb4', (0, 1)), ('stitched',748, 640,  44, 36, -.08, 0, '#b0bccc', (0, 1)),
        ('toothed', 1858, 250, 250, 45, -.16, 2, '#ff6a3c', (1, -.3)), ('milky',   300,  812,  96, 45,  .24, 1, '#cfe0e0', (-.4, 1)), ('tri',     664,  226,  34, 45, .0, 0, '#ff3a40', (.4, -1)), ('worm',  556, 800, 66, 45, -.3, 1, '#f080b0', (0, 1)),
        ('human2',  1900,  760, 170, 48, 1.45, 2, '#b0ccd8', (1, .3)), ('spiral',   40,  650, 104, 50,  .0, 1, '#b088e0', (-1, 0)), ('star',    500,  300,  36, 50, .2, 0, '#8fb4f0', (0, 1)), ('slit2',  140,   70,  56, 50, 1.2, 0, '#d8c23a', (-1, -1))]


def flesh(c, kind, x, y, s, o, rot, seed, hue, run, k):
    """the piece of body an eye is buried in: brush-ink mass, lid folds, wet edge, bone or hair.  k 0..1 how much it has come forward"""
    r = np.random.default_rng(seed * 11 + 2); hu = rgb(hue); dk = mixc('#06080d', hu * .5, .10); ed = mixc('#1a2634', hu * .6, .30); ca, sa = math.cos(rot), math.sin(rot)
    T = lambda P: np.c_[x + (np.asarray(P, float)[:, 0] * ca + np.asarray(P, float)[:, 1] * sa) * s, y + (np.asarray(P, float)[:, 0] * sa - np.asarray(P, float)[:, 1] * ca) * s]
    m = 22; a = np.linspace(0, 2 * np.pi, m, endpoint=False); ru = A(run, float); ru = ru / (np.hypot(*ru) + 1e-6); stretch = 1 + 1.5 * np.clip(np.cos(a) * ru[0] + np.sin(a) * ru[1], 0, 1) ** 2
    rad_ = s * (1.55 + .34 * r.normal(0, 1, m)) * stretch; B = np.c_[x + np.cos(a) * rad_ * 1.15, y + np.sin(a) * rad_]
    c.fill(spl(B, True, corners=range(0, m, 2), n=5), '#04060a', .94 * k)
    body = np.c_[x + np.cos(a) * rad_ * .90, y + np.sin(a) * rad_ * .86]; c.fill(spl(body, True, n=5), dk, .95 * k)
    for j in range(m):                                                # broken edge where the cold light rakes it
        if r.random() < .55 and np.sin(a[j]) < .4: q = spl(B[[j, (j + 1) % m, (j + 2) % m]] * 1., n=6); c.line(q, max(1.2, s * .030), ed, taper=(.3, .6), press=.6, seed=seed + j, alpha=.8 * k)
    xx = np.linspace(-1, 1, 33); ev = np.clip(1 - np.abs(xx) ** 2.2, 0, 1) ** .8
    for j in range(1, 6):                                             # lid folds, tighter toward the opening
        for sg in (1, -1):
            off = .56 * max(.35, o) * ev * (1 if sg > 0 else .86) + .10 + .135 * j ** 1.15 * ev; w = 1 + .07 * j; P = np.c_[xx * w, sg * off + .02 * np.sin(xx * 5 + j + seed)]; i0 = int(r.integers(0, 7)); i1 = 33 - int(r.integers(0, 7))
            c.line(T(P[i0:i1]), max(1., s * (.050 - .007 * j)), mixc('#020305', ed, .16 * j) if sg > 0 else mixc('#020305', ed, .08 * j), taper=(.35, .45), press=.5, seed=seed * 3 + j, alpha=k)
            if sg > 0 and j in (1, 3): c.line(T(np.c_[P[:, 0], P[:, 1] + .035][i0 + 4:i1 - 8]), max(.8, s * .018), ed, taper=(.5, .5), seed=j, alpha=.75 * k)
    L = np.c_[xx, -.52 * o * ev - .035]                               # wet lower rim
    for u0, u1 in ((.12, .36), (.46, .58), (.70, .80)): c.line(T(L[int(u0 * 33):int(u1 * 33) + 2]), max(1., s * .030), mixc('#e8f2ff', hu, .25), taper=(.4, .5), seed=seed, alpha=.80 * k * min(1, o))
    for j in range(26):                                               # pores, warts, hatching
        th = r.uniform(0, 6.28); R = r.uniform(1.15, 1.75) * s; p = A([x + math.cos(th) * R * 1.1, y + math.sin(th) * R * .85]); d = A([math.cos(th + .5), math.sin(th + .5)]) * s * r.uniform(.05, .16); c.line([p - d, p, p + d * .8], max(.9, s * .016), ed if j % 3 else '#020305', taper=(.3, .6), seed=j, alpha=.7 * k)
    if seed % 4 == 0:                                                 # bone
        for j in range(4): q = -.9 + j * .55; c.fill(T(spl(A([(q - .14, 1.05), (q + .02, 1.62 + .2 * (j % 2)), (q + .16, 1.04), (q, .98)]), True, corners=(1,), n=5)), mixc('#8f8a80', dk, .55), k); c.line(T(A([(q + .16, 1.04), (q + .06, 1.4), (q + .02, 1.62)])), max(1., s * .02), '#020305', seed=j)
    if seed % 4 == 1:                                                 # hair
        for j in range(18): q = -1.3 + j * .15; c.line(T(spl(A([(q, .95 + .1 * math.sin(j)), (q * 1.1 + .1, 1.4), (q * 1.2 + .3 * math.sin(j * 2.), 2.0)]), n=6)), max(1., s * .026), '#030408' if j % 3 else ed, taper=(.02, .8), seed=j, alpha=k)


class Shot:
    id = 'c2'; n = 72
    finish = dict(bloom=.32, vig=.30)
    EVENTS = [(0, 'subway_air', dict(note='dead platform, breath mist')), (0, 'curse_eye', dict(n=1, pan=-.4)), (9, 'curse_eye', dict(n=2)), (18, 'curse_eye', dict(n=3)), (27, 'curse_eye', dict(n=3)), (36, 'curse_eye', dict(n=3, note='one is huge, lower left')), (45, 'curse_eye', dict(n=4, note='one is huge, right edge')),
              (50, 'curse_eye', dict(n=3, size=.7)), (4, 'breath_fear', dict(size=.6)), (22, 'breath_fear', dict(size=.8)), (40, 'breath_fear', dict(size=1.)), (54, 'hand_settle', dict(note='stitched hand presses his near shoulder; its fingers crept over the edge at 48 49 51 52', pan=-.3)),
              (60, 'pupil_shrink', dict(note='pupils to points, one red frame')), (61, 'heartbeat', dict(note='one beat after the red frame'))]

    def setup(self):
        key = 'cache/c2_v4_bg.npy'
        if not os.path.exists(key):
            P = world.plate('platform', w=2400, h=1350, lit=.12); z = 1.13; M = np.float32([[z, 0, 1000 - 1010 * z], [0, z, 562 - 640 * z]]); im = cv2.warpAffine(P.img.astype(f32), M, (2000, 1125), flags=cv2.INTER_AREA)
            col = np.zeros((1350, 2400), f32); cv2.fillPoly(col, [A([(596, 208), (740, 208), (828, 258), (828, 1090), (596, 1090)], np.int32)], 1.); col = cv2.GaussianBlur(cv2.warpAffine(col, M, (2000, 1125)), (0, 0), 5); del P
            im = comp.bokeh(np.dstack([im, np.ones(im.shape[:2], f32)]).astype(f32), 8.5, 0)[..., :3]; im = cv2.GaussianBlur(im, (0, 0), 2.0); x, y = grid(1125, 2000); d = np.clip(1 - np.hypot((x - 760) / 900, (y - 520) / 620), 0, 1)
            im = im * (.36 + .62 * smooth(0, .9, d))[..., None] * rgb('#c4d8e8'); np.save(key, np.dstack([im, col]).astype(np.float16)); del im, col
        q = np.load(key, mmap_mode='r'); self.bg = np.array(q[..., :3]); self.col = np.array(q[..., 3]); self._c = {}

    # ---- character timeline
    def state(self, f):
        d = f // 2 * 2; br = math.sin((f - 2) / 18. * 2 * math.pi)
        if f < 9: e = Yuji.blend('neutral', 'fear', .15); gx, gy = .020, .0
        elif f < 18: e = Yuji.blend('neutral', 'fear', .32); gx, gy = (-.030, .002) if f >= 10 else (.0, 0)
        elif f < 27: e = Yuji.blend('neutral', 'fear', .46); gx, gy = (.032, -.004) if f >= 20 else (.0, 0)
        elif f < 36: e = Yuji.blend('neutral', 'fear', .62); gx, gy = (-.036, .004) if f >= 28 else (.0, 0)
        elif f < 54: e = Yuji.blend('neutral', 'fear', .82); gx, gy = (.016, -.004) if f < 46 else (-.034, -.010)
        else: e = Yuji.expr('fear'); gx, gy = (-.042, -.016) if f < 60 else (-.040, -.014)
        if f in (13, 14, 32, 33): e = Yuji.lids(e, .95)
        if f in (15, 34): e = Yuji.lids(e, .5)
        e = dict(e, gx=gx, gy=gy, sweat=0., glint=.8, under=.6, hatch=.25 + .35 * smooth(30, 54, f), furrow=max(.2, e['furrow']))
        if f >= 60: k = min(1., (f - 59) / 2.); e.update(iris=.046 * (1 - k) + .025 * k, pupil=.012 * (1 - k) + .005 * k, U=list(A(e['U'], float) + A([0, .012, .014, .012, .0]) * k), Lo=list(A(e['Lo'], float) - A([0, .008, .010, .008, 0]) * k), glint=.4, mL=list(A(e['mL'], float) - .012 * k))
        dip = 0. if f < 54 else [20, 34, 30, 33, 38, 37, 35, 34][min(f - 54, 7)]
        jl = 0. if f < 54 else [-.015, -.035, -.030, -.026, -.030, -.026][min(f - 54, 5)]
        tr = (np.random.default_rng(700 + f // 2).normal(0, 1, 2) * (.6 + 1.2 * smooth(34, 60, f) + 1.6 * (f >= 60))) if f >= 30 else np.zeros(2)
        z = 1 + .045 * comp.ease(f / 71.) + .022 * comp.ease(np.clip((f - 58) / 13., 0, 1))
        hd = YHead().pose(1262 + tr[0], 262 + br * 2.6 + dip * .30 + tr[1], 516, .33 - (.03 if f >= 46 else 0.) + jl * .5, -.03, -.015 + jl * .4)
        return e, hd, dip, z, br, d

    def opening(self, f, i):
        kind, x, y, s, t0 = EYES[i][:5]; k = f - t0
        if k < 0: return 0.
        o = [.22, .62, 1.16, 1.08, .97, 1.0][min(k, 5)]; r = np.random.default_rng(i * 31 + 5); per = int(r.integers(19, 37)); b = (f - t0 - 8 + int(r.integers(0, per))) % per if k > 8 else 99
        if b < 4 and kind not in ('points', 'fish', 'spider', 'tri', 'crescent'): o *= [.55, .08, .12, .7][b]
        return o * (1.10 if f >= 60 else 1.)

    def eyes(self, im, f, hd):
        tgt = hd.P([(0, -.1)])[0]
        for li, bl in ((0, 2.6), (1, 1.7), (2, 9.)):
            c = Cel(); g = np.zeros((H // 4, W // 4, 3), f32); yy, xx = np.mgrid[0:H // 4, 0:W // 4].astype(f32) * 4
            for i, (kind, x, y, s, t0, rot, l, hue, run) in enumerate(EYES):
                if l != li: continue
                near = float(np.clip((f - t0 + 9) / 12., 0, 1)); o = self.opening(f, i)
                if near <= 0: continue
                dr = A([math.sin(f / 24 * 1.3 + i), math.cos(f / 24 * 1.1 + i * 2)]) * (1.2 + li * 1.6); x, y = x + dr[0], y + dr[1] + (1 - near) * s * .22
                flesh(c, kind, x, y, s, o, rot, i + 3, hue, run, .30 * near + .70 * min(1., o))
                v = tgt - A([x, y]); v = v / (np.hypot(*v) + 1e-6); look = A([v[0], -v[1]])
                if kind != 'goat': look = look * .75 + A([math.sin((f // 6) * 2.1 + i * 1.7), math.cos((f // 6) * 1.6 + i)]) * .28 * (f < 54)
                cr, sr = math.cos(-rot), math.sin(-rot); E.eye(c, kind.rstrip('2'), x, y, s, o, A([look[0] * cr - look[1] * sr, look[0] * sr + look[1] * cr]), i + 3, rot, f // 2 * 2)
                if o > 0: g += (np.clip(1 - np.hypot(xx - x, yy - y) / (s * 2.3), 0, 1) ** 2 * .30 * min(1, o))[..., None] * rgb(hue)
            L = cv2.GaussianBlur(c.layer(), (0, 0), bl); g = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)
            if li == 0: q = 1 - .97 * cv2.resize(self.col[22:1102, 40:1960].astype(f32), (W, H)); L = L * q[..., None]; g = g * q[..., None]
            L[..., :3] *= rgb('#c4d0de') * (.74, .86, .80)[li]; im = comp.over(im, L); im = im + g * L[..., 3:4] * (.5, .55, .4)[li] + cv2.GaussianBlur(g, (0, 0), 30) * .10
        return im

    def cheek_light(self, f):
        """how much light the open eyes throw on his near cheek, and its colour"""
        c = np.zeros(3); t = 0.
        for i, q in enumerate(EYES):
            o = min(1., self.opening(f, i)); w = o * (q[3] / 120.) ** 1.1 * (1. if q[1] < 900 else .0); c += rgb(q[7]) * w; t += w
        return (c / (t + 1e-6), 1 - math.exp(-t * .28))

    # ---- the hand
    FG = ('little', 'ring', 'middle', 'index')

    # the fingers come over the edge of the shoulder from behind, one after the other; the back of the hand rises after them and presses on 54
    CREEP = dict(little=(48, (.34, .74, 1.)), ring=(49, (.30, .70, 1.)), middle=(51, (.46, .88, 1.)), index=(53, (1.,)))

    def prog(self, f, nm):
        t0, st = self.CREEP[nm]; return 0. if f < t0 else st[min(f - t0, len(st) - 1)]

    def hand(self):
        """(layer, {finger: mask}, {finger: tip px}): one drawing of the resting hand"""
        key = 'hand'
        if key not in self._c:
            po = dict(HD.POSES['rest']); off = A([-2., -72.])
            h = HD.Hand(po, 'R', R=HD.euler(184, 113, 10), pos=(0, 0, 0), pivot=(0, .75, 0), bld=HD.SLIM, nails='dark')
            g = HD.render([h], HD.Cam(632 + off[0], 800 + off[1], 278, 4.4), light=(-.35, .85, -.10), res=470, clip=(0, -60, 1250, H + 120)); c = Cel()
            pal = dict(HD.PAL['day'], skin='#8e8796', shade='#3b354c', line='#0b0910', nail_dark='#120d17'); p2 = dict(index='#7d8a78', middle='#a08079', ring='#757b90', little='#85796a')
            HD.fill_flat(c, g, pal); pt = g.pts[0]; P_ = lambda a, b: A(pt[(a, b)], float)
            cut = {}
            for G in g.groups:                                           # patched skins
                nm = G['name']
                if nm not in p2 or not G['polys']: continue
                a_, b_ = (P_(nm, 'mcp'), P_(nm, 'pip')) if nm != 'middle' else (P_(nm, 'pip'), P_(nm, 'dip')); u = (b_ - a_) / (np.hypot(*(b_ - a_)) + 1e-6); n = A([-u[1], u[0]]); m = a_ * .48 + b_ * .52; cut[nm] = (m, u, n)
                c.clip(G['polys']); c.fill(A([m - n * 300, m + n * 300, m + n * 300 + u * 900, m - n * 300 + u * 900]), p2[nm]); c.unclip()
            wr, mk, ik, lk = A(pt['wrist'], float), P_('middle', 'mcp'), P_('index', 'mcp'), P_('little', 'mcp'); pk = [G for G in g.groups if G['name'] == 'palm' and G['polys']]
            if pk: c.clip(pk[0]['polys']); c.fill(spl(A([wr * .62 + ik * .38, ik * .82 + mk * .18, mk * .55 + lk * .45, lk * .7 + wr * .3, wr * .6 + lk * .40]), True, n=6), '#9a9084'); c.unclip()
            HD.fill_shadow(c, g, pal); HD.fill_overlays(c, g, pal, 6.); HD.ink_lines(c, g, pal, 7.2, 4)
            def seam(a, b, n=7, bow=0., sz=1.):
                a, b = A(a, float), A(b, float); u = b - a; Ls = np.hypot(*u) + 1e-6; u = u / Ls; nn = A([-u[1], u[0]]); q = np.linspace(0, 1, 12); c.line(A([a + u * Ls * v + nn * bow * math.sin(math.pi * v) for v in q]), 4.0, '#0f0911', taper=(.1, .1), seed=3)
                for j in range(n):
                    v = (j + .5) / n; m = a + u * Ls * v + nn * bow * math.sin(math.pi * v); sk = u * 7 * (-1) ** j; c.line([m - nn * 16 * sz - sk, m, m + nn * 16 * sz + sk], 4.6, '#0f0911', taper=(.1, .1), seed=j); c.line([m - nn * 14 * sz - sk + A([2, 3]), m + A([2, 3])], 1.8, '#a89aa6', taper=(.1, .3), seed=j + 9)
                    for e in (-1, 1): c.fill(spl(m + nn * 17 * sz * e + sk * e + A([(-4, 0), (0, -4), (4, 0), (0, 4)]), True, n=4), '#0f0911')
            for nm, (m, u, n) in cut.items(): w_ = 36 if nm != 'little' else 30; seam(m - n * w_, m + n * w_, 4, 8)
            seam(ik * .82 + mk * .18 + (wr - mk) * .05, lk * .62 + wr * .38, 9, 22, 1.1); seam(wr * .84 + mk * .16 + A([-52, 4]), wr * .84 + mk * .16 + A([56, -2]), 7, 10, 1.1)
            L = c.layer(); L = Y.rim_in(L, 5, 11, '#a8d2e6', .95, 4.); cy = 800 + off[1]; L = Y.para_fast(L, (632, cy + 200), (632, cy - 230), [(0, '#d8d4de'), (.40, '#b0aac0'), (.78, '#37344a'), (1, '#0a0a12')]); yy = grid()[1]; L[..., :3] *= (.10 + .90 * smooth(cy - 190, cy + 20, yy))[..., None]; L = L * smooth(cy - 172, cy - 100, yy)[..., None]          # it sinks into black, it never turns transparent
            mk = {}
            for nm in self.FG:
                m_ = np.zeros((H, W), np.uint8); dl = A([0., 0.])
                for G in g.groups:
                    if G['polys'] and (G['name'] == nm or G['name'] == 'nail_' + nm): cv2.fillPoly(m_, [np.round(np.asarray(q_)).astype(np.int32) for q_ in G['polys']], 255)
                mk[nm] = cv2.GaussianBlur(cv2.dilate(m_, np.ones((11, 11), np.uint8)), (0, 0), 1.2)
            self._c[key] = (L.astype(np.float16), mk, {nm: P_(nm, 'tip') for nm in self.FG})
        return self._c[key]

    def dent(self, bd, f):
        """the cloth gives under the hand"""
        if f < 48: return bd
        a = (1.5, 2.5, 4., 6., 8., 9.)[f - 48] if f < 54 else (8., 14., 11., 12., 14., 13.)[min(f - 54, 5)]; x0, x1, y0, y1 = 300, 1000, 520, H; yy, xx = np.mgrid[y0:y1, x0:x1].astype(f32)
        D = a * np.exp(-((xx - 590) / 150) ** 2) * np.exp(-((yy - 925) / 150) ** 2); bd = bd.copy(); bd[y0:y1, x0:x1] = cv2.remap(bd, xx, yy - D, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE); return bd

    def hand_over(self, im, bd, f):
        L, mk, tips = self.hand(); L = L.astype(f32); ba = bd[..., 3:4]; mv = lambda a, dx, dy: cv2.warpAffine(a, np.float32([[1, 0, dx], [0, 1, dy]]), (W, H))
        sh = lambda a: 1 - .66 * (cv2.GaussianBlur(mv(a, 14, 24), (0, 0), 8) * bd[..., 3])[..., None]
        if f < 54:
            r = 1 - comp.ease(np.clip((f - 48) / 6., 0, 1)); B = mv(L, -8 * r, -8 + 210 * r); im = comp.over(im, B * (1 - ba) * A([1., 1., 1., 1.], f32))                                       # the rest of it, still behind him
            for nm in self.FG if f < 53 else ():
                p = self.prog(f, nm)
                if p <= 0: continue
                F_ = mv(L * (mk[nm].astype(f32) / 255.)[..., None], 0, -8 - (1 - p) * 125) * ba; F_[:868] = 0; im = im * sh(F_[..., 3]); im = comp.over(im, F_)
            if f < 53: return im
            Q = mv(L, 0, -16); im = im * sh(Q[..., 3]); return comp.over(im, Q)                                                    # the back of the hand comes over, still in the air
        dy = [12, 30, 25, 27, 32, 31, 30][min(f - 54, 6)]; Q = mv(L, 0, dy); im = im * sh(Q[..., 3]); c = Cel(ss=1 if f < 54 else 2); k = min(1., (f - 53) / 2.)
        for j, nm in enumerate(self.FG):                              # creases driven out from under each finger
            p = A(tips[nm], float) + A([0, dy])
            for s_, (a, b, w_) in ((-1, (26, 60, 6.)), (1, (24, 54, 5.))):
                q = [p + A([s_ * 24, -26]), p + A([s_ * (24 + a * .5 * k), -14 + 5 * j % 3]), p + A([s_ * (24 + b * k), 4 + 8 * ((j + s_) % 2)])]; c.line(spl(A(q), n=8), w_, '#04050a', taper=(.1, .7), seed=j * 2 + s_); c.line(spl(A(q), n=8) + A([0, 6]), 2.6, '#55618c' if nm != 'index' else '#8c4350', taper=(.2, .7), seed=j, alpha=.8)
        for j, (x0_, y0_, x1_, y1_, x2_, y2_, red) in enumerate(((452, 905, 414, 930, 372, 978, 0), (446, 940, 420, 972, 404, 1020, 0), (470, 1000, 446, 1030, 436, 1070, 0), (818, 930, 862, 948, 916, 1000, 1), (800, 978, 834, 1006, 862, 1050, 1), (836, 886, 880, 894, 936, 930, 1))):          # the cloth pulled toward the hand
            q = spl(A([(x0_, y0_ + dy), ((x0_ * (1 - k) + x1_ * k), y0_ * (1 - k) + y1_ * k + dy), (x0_ * (1 - k) + x2_ * k, y0_ * (1 - k) + y2_ * k + dy)]), n=10); c.line(q, 8., '#04040a' if not red else '#15040a', taper=(.06, .75), seed=60 + j); c.line(q + A([2, 7]), 3.4, '#56639a' if not red else '#8f3f4d', taper=(.1, .7), seed=70 + j, alpha=.9)
        im = comp.over(im, c.layer() * ba); return comp.over(im, Q)

    def frame(self, f):
        e, hd, dip, z, br, d = self.state(f)
        q = .3 * f; im = comp.cam2d(self.bg.astype(f32), 1000 - q, 562, 1.0)
        im = self.eyes(im, f, hd); im = self.breath(im, f)
        Lg = Yuji.light('under', side=1, amt=.60, skin=('#7f7688', '#463f5c', '#272338'), white=('#e4eaee', '#8b90a8'), hair=('#7b5766', '#4a3245', '#a87c8a'), hood=('#5f1b29', '#32101f', '#8a3540'), cloth=('#12131f', '#2a3150', '#07070d'), socket=.9, nose=1., hcut=.22,
                        rim=[dict(d=(9, 13), col='#b4dcf0', amt=1.)], para=[dict(p0=(0, -1.0), p1=(0, 1.9), stops=[(0, '#f2f6ff'), (.35, '#d4d4e6'), (1, '#565578')], amt=1.)])
        bd = Yuji.bust(hd, e, Lg, f=d, wind=(.04, .02), seed=9, byaw=.21, broll=0. if f < 54 else -.034 * min(1, dip / 30.), base=(1240, 922 + dip * .95 + br * 2.2), raise_=.27 + .10 * smooth(20, 50, f) - .13 * (f >= 54), breath=br)
        bd = self.dent(bd, f); col, amt = self.cheek_light(f); x, y = grid(); m = np.clip(1 - np.hypot((x - 900) / 420, (y - 470) / 420), 0, 1) ** 1.2 * bd[..., 3]; inner = np.clip(comp.blur(1 - cv2.warpAffine(bd[..., 3], np.float32([[1, 0, 60], [0, 1, 0]]), (W, H), borderMode=cv2.BORDER_REPLICATE), 34), 0, 1)
        bd[..., :3] += (bd[..., :3] * 1.6 + .03 * bd[..., 3:4]) * col * (amt * .55 * m * (.35 + .65 * inner))[..., None]
        im = comp.over(im, bd)
        c = Cel(); self.bead(c, f, e, hd, Lg); im = comp.over(im, c.layer())
        if f >= 48: im = self.hand_over(im, bd, f)
        im = im * (1 - .34 * smooth(.40, 1.2, np.hypot((x - 1080) / W, (y - 470) / H) * 1.6))[..., None]
        if f == 60: l = comp.lum(np.clip(im, 0, 1)); l = cv2.GaussianBlur(l, (0, 0), .8); t = smooth(.085, .20, l)[..., None]; t2 = smooth(.40, .52, l)[..., None]; im = rgb('#090205') * (1 - t) + rgb('#d80e1c') * t; im = im * (1 - t2) + rgb('#fff4ea') * t2
        elif f == 61: im = im * rgb('#ffb0a8') * 1.05
        s = (0, 0) if f < 60 else shake(f, 6.0, 4, 3.5, 60)
        return comp.cam2d(im, 960 + (z - 1) * 260 + s[0], 540 - (z - 1) * 260 + s[1], z)

    def breath(self, im, f):
        x, y = grid()
        for t0, amp in ((4, .7), (22, .85), (40, 1.0)):
            t = (f - t0) / 16.
            if not 0 < t < 1.25: continue
            cx, cy = 1560 + 240 * (1 - math.exp(-1.6 * t)), 590 - 30 * t; R = 60 + 190 * t
            if not hasattr(self, '_n'): self._n = fbm(H, W, (110, 42, 16), (1, .6, .35), 4)
            m = np.clip(1 - np.hypot((x - cx) / (R * 1.5), (y - cy) / R), 0, 1) ** 1.3 * np.clip(.55 + .45 * np.roll(self._n, (int(f * 2), int(f * 5)), (0, 1)), 0, 1.4) * amp * .30 * np.sin(np.pi * min(1, t / 1.25)) ** .8
            im = im * (1 - m[..., None] * .6) + rgb('#8fa2b6') * m[..., None] * .7
        return im

    def bead(self, c, f, e, hd, Lg):
        """a drop of sweat from the temple, past the ear, along the jaw"""
        if f < 8: return
        q = YHead.of(hd); F = Y._Face(c, q, dict(Yuji.expr(e)), Lg, 3)
        path = A([(-.545, .30), (-.575, .16), (-.60, -.02), (-.60, -.20), (-.585, -.36), (-.555, -.52), (-.50, -.66), (-.42, -.78), (-.33, -.86)]); S_ = spl(path, n=12)
        u = float(np.interp(f // 2 * 2, [8, 12, 18, 26, 32, 40, 45, 52, 58, 66, 71], [0, .05, .07, .30, .34, .55, .58, .76, .80, .97, .98])); i = u * (len(S_) - 1); p = A([np.interp(i, np.arange(len(S_)), S_[:, j]) for j in (0, 1)]); i2 = max(0, i - 2); p0 = A([np.interp(i2, np.arange(len(S_)), S_[:, j]) for j in (0, 1)]); s0, s1 = q.Pc([p0])[0], q.Pc([p])[0]; dv = s1 - s0; ang = math.atan2(dv[0], dv[1]) + q.roll if np.hypot(*dv) > .5 else 0.
        tail = S_[max(0, int(i) - 14):int(i) + 1]
        if len(tail) > 3: c.line(q.Pc(tail, squeeze=False), 5.0, '#a8bccc', taper=(.85, .05), seed=1, alpha=.45)
        F.bead(p[0], p[1], .024 + .008 * smooth(8, 18, f), ang)
