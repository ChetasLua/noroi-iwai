"""SUKUNA lane: finished keys and the model sheet.   python chars/sukuna_keys.py prayer|smile|fuga|sheet|all  -> qa/sukuna/*.jpg"""
import sys, os
if __name__ == '__main__': sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np, cv2, math
from jj import W, H, comp, hands as HD, fx_fire as FF
from jj.paint import rgb, f32, Canvas
from jj.ink import Cel, spl
from jj.kit import para
from jj.comp import over, screen, blur
from jj.fx_core import put, hot, FX
from chars.sukuna_ink import *
from chars.sukuna_ink import _nz
from chars.sukuna import Suk
from chars.sukuna_body import Prayer, scarf_back, scarf_front
from chars.sukuna_eyes import InkEye, draw_eyes, charcoal
from chars.sukuna_hands import sign, one, ink_hands, BANDS
from chars.sukuna_fire import *


def ov(A, B): return np.dstack([over(A[..., :3], B), 1 - (1 - A[..., 3]) * (1 - B[..., 3])])


def prayer_eyes(): return [InkEye(520, 270, 520, 190, .36, 3, 1, .10), InkEye(1700, 150, 400, 130, -.42, 8, 1, -.12), InkEye(330, 690, 360, 128, .20, 13, 1, .08), InkEye(1790, 520, 230, 78, -.66, 21, 1, -.10)]


def prayer_figure(T=None, head=None, expr='prayer', hands=None, dark=1, rimred=12., **hk):
    """the kneeling figure of key 1 as one RGBA layer.  hands = Drawn geometry (default: praying sign); head = dict(cx, cy, s, yaw, pitch, roll)"""
    h = dict(cx=1318, cy=398, s=205, yaw=-.40, pitch=.14, roll=.05); h.update(head or {}); pr = Prayer(T)
    sh = Sheet(7); scarf_back(sh, pr.T); pr.robe(sh); pr.insides(sh); L = sh.layer()
    k = Suk(h['cx'], h['cy'], h['s'], h['yaw'], h['pitch'], h['roll'], expr, **hk); L = ov(L, k.ink(dark=dark, neck='dark'))
    sh = Sheet(7); pr.sleeves(sh); pr.scarf_shadow(sh); scarf_front(sh, pr.T); L = ov(L, sh.layer())
    if hands is not False:
        g = hands or sign('pray', 1190, 700, 172); sh = Sheet(7); ink_hands(sh, g, 1.05)
        for q in pr.behind_lips(): sh.cut(q, sh.a, sh.v, sh.k, sh.g)
        L = ov(L, sh.layer())
    return red_rim(L, rimred, 2) if rimred else L


def key_prayer():
    im = ink_ground(); im = draw_eyes(im, prayer_eyes(), (1., .9, .6, .25), [(.25, -.2), (-.4, .1), (.3, 0), (0, 0)])
    im = over(im, prayer_figure()); c = Cel(); red = Cel()
    spray(c, (250, 860), .5, 60, 420, .45, 15, BLACK, 5); spray(c, (700, 940), 2.6, 40, 360, .5, 11, BLACK, 15); spray(red, (120, 930), -.2, 50, 520, .35, 17, '#a3131e', 6); spray(c, (1010, 900), -.4, 16, 160, .8, 6, BLACK, 9); spray(red, (960, 900), .2, 24, 200, .6, 8, '#b3121f', 10); spray(c, (1740, 880), 2.4, 22, 200, .7, 7, BLACK, 11)
    return over(over(im, inkify(red.rgba(), .3)), inkify(c.rgba(), .3))


def red_ground(seed=3):
    """blood field with black coming in from the right, all pulled with one slanted brush"""
    k = ('redg', seed)
    from chars import sukuna_ink as SI
    if k in SI._cache: return SI._cache[k].copy()
    cv = Canvas(H, W, color='#9c0f1a'); r = np.random.default_rng(seed)
    for (cols, n, w0, w1, dr, op) in ((('#b3121f', '#7d0b15', '#8e0e19'), 60, 90, 240, .25, .9), (('#c81e2a', '#6a0912', '#a5121d'), 80, 40, 150, .5, .85), (('#d8363d', '#4a070e'), 24, 14, 50, .72, .75)):
        for j in range(n):
            a = 1.12 + r.normal(0, .10); L = r.uniform(300, 900); x, y = r.uniform(-100, W), r.uniform(-300, H); q = np.linspace(0, 1, 8)[:, None]; d = np.array([math.cos(a), math.sin(a)]); P = np.array([x, y]) + d * L * q + np.array([-d[1], d[0]]) * r.normal(0, 24) * q ** 2
            cv.strokes(P[None].astype(f32), r.uniform(w0, w1) / 2, rgb(cols[int(r.integers(len(cols)))])[None], op=op, dry=dr * r.uniform(.5, 1.5), seed=j, bristle=.12)
    for (col, n, x0, x1) in (('#2a070b', 26, 430, 1250), ('#140709', 34, 520, 1500), ('#0d0b0e', 50, 700, 1950)):
        for j in range(n):
            a = 1.12 + r.normal(0, .09); L = r.uniform(380, 900); x, y = r.uniform(x0, x1), r.uniform(-420, H * .7); q = np.linspace(0, 1, 8)[:, None]; d = np.array([math.cos(a), math.sin(a)])
            cv.strokes((np.array([x, y]) + d * L * q)[None].astype(f32), r.uniform(50, 200) / 2, rgb(col)[None], op=.95, dry=r.uniform(.15, .55), seed=j + 77, bristle=.1)
    SI._cache[k] = cv.img * (1 + .04 * _nz('ti2', (2.6, 1.3), (1, .8), H, W, 9))[..., None]; return SI._cache[k].copy()


def key_smile():
    im = red_ground(); k = Suk(1150, 205, 900, -.34, .06, .17, 'smile', noir=1., seed=4); im = over(im, k.ink(dark=1, neck='dark', box=(0, 0, W, H)))
    c = Cel(); spray(c, (470, 600), -2.7, 60, 420, .45, 15, BLACK, 5); spray(c, (560, 860), 2.5, 40, 300, .4, 12, BLACK, 6); spray(c, (760, 470), -.3, 22, 240, .8, 4.5, BLACK, 7); spray(c, (620, 980), 2.9, 30, 400, .3, 9, BLACK, 8)
    return over(im, inkify(c.rgba(), .3))


PALF = dict(FF.PAL, hot='#ffdf4a', core='#fff6b4')


def flame(im, *streams):
    """drawn fire streams (path, half width, time s, seed, profile) with their own light"""
    c = Cel()
    for (P, w, tt, sd, pr) in streams: FF.flame_stream(c, spl(np.array(P, float), n=16), w, tt, sd, PALF, **({'prof': pr} if pr else {}))
    l = c.layer(); return put(im, FX(l, hot(l, .5, (6, 26, 80), (.35, .35, .35), '#ff7a30', .6) * (1 - .8 * l[..., 3:4])))


def key_fuga(t=.7):
    g = one('rear', 1140, 880, 205, 'R', HD.euler(58, -4, -62), 4.2, light=(.75, -.15, .45)); px = g.pts[0]; p0 = (px['index', 'tip'] + px['thumb', 'tip']) / 2
    im = fire_ground(3, ((1560, 820, 1150, 1.0), (760, 1200, 760, .45))); im = flame(im, ([(1330, -160), (1560, 90), (1700, 420), (1700, 760), (1600, 1240)], 74, t + .6, 5, None)); im = put(im, FF.embers(20, 60, 4, (300, 0, W, H), near=.08))
    T = lambda p: (np.asarray(p, float) - (640, 960)) * 1.22 + (560, 1010)
    c = Cel(); fire_bare(c, T=T, lw=1.35); im = over(im, para(c.layer(), (100, 760), (1000, 1080), [(0, '#8d7464'), (1, '#f4e9da')]))
    k = Suk(700, 430, 420, .50, -.07, .24, 'strain', seed=4, look=(.45, -.1), neck_len=2.5); F_ = k.fire(ls=1, light_vec=(.8, .6)); im = over(im, para(F_, (360, 60), (900, 860), [(0, '#97806f'), (.55, '#efe6da'), (1, '#ffffff')]))
    pr = lambda v: .10 + .90 * np.clip(v, 0, 1) ** 1.1
    im = flame(im, ([p0 + (-20, 0), p0 + (150, 30), p0 + (360, 90), (1720, 1000), (2140, 1190)], 138, t, 9, pr), ([p0, p0 + (120, -70), p0 + (250, -90), p0 + (420, -180)], 30, t - .4, 12, None), ([p0 + (10, 10), p0 + (80, 90), p0 + (210, 170), p0 + (380, 220)], 26, t + .2, 13, None))
    c = Cel(); fire_hands(c, g, 1.4); im = over(im, rim(c.layer(), -5, 3, '#fff0a0', .9, .5))
    return put(im, FF.embers(31, 14, 9, (900, 300, W, H), near=.2))


def neck_fade(L, k):
    """cut a head layer softly below the neck (model sheets)"""
    y0 = float(k.H.P3(np.array([0, -1.36, 0.]))[1]); yy = np.arange(H)[:, None]; return L * np.clip((y0 - yy) / 26. + 1, 0, 1)[..., None].astype(f32)


def sheet():
    """2880 x 1620 model sheet"""
    out = np.zeros((1620, 2880, 3), f32); cw, ch = 480, 540
    def cell(i, j, im): out[j * ch:(j + 1) * ch, i * cw:(i + 1) * cw] = im[:ch, :cw]
    ink = [('prayer', 0, 0, 0), ('cold', .6, 0, .04), ('smile', -.55, .22, -.08), ('contempt', .30, -.28, .14), ('laugh', .05, .24, -.10), ('strain', -.45, .04, .26)]
    for i, (e, y, p, r) in enumerate(ink):
        k = Suk(240, 300, 172, y, p, r, e, seed=4); b = paper(); cell(i, 0, over(b, neck_fade(k.ink(dark=1 if y <= .1 else 1), k)))
    row = [('fire', 'cold', .45, .02, 0), ('fire', 'smile', -.5, .16, .10), ('fire', 'strain', 1.2, 0, .12), ('ink', 'cold', -1.5, 0, 0), ('ink', 'smile', .45, .20, 2.75), ('cel', 'smile', .1, .05, 0)]
    for i, (st, e, y, p, r) in enumerate(row):
        k = Suk(240, 300 if abs(r) < 1 else 270, 172, y, p, r, e, seed=4, dissolve=.5 if st == 'cel' else 0.)
        if st == 'fire': b = np.broadcast_to(rgb('#a8320c') * 1.0, (H, W, 3)).copy(); L = k.fire(ls=1 if y >= 0 else -1)
        elif st == 'ink': b = paper(); L = k.ink(dark=1)
        else: b = np.broadcast_to(rgb('#aeb6c4'), (H, W, 3)).copy(); L = k.cel(ls=-1)
        cell(i, 1, over(b, neck_fade(L, k) if abs(r) < 1 else L))
    # row 3: hands in both inkings, the eye tearing open
    for i, (nm, kw) in enumerate([('pray', {}), ('sukuna', dict(yaw=-14, pitch=6, roll=0))]):
        sh = Sheet(5); ink_hands(sh, sign(nm, 240, 250, 140, **kw), .95); cell(i, 2, over(paper(), sh.layer()))
    c = Cel(); fire_hands(c, HD.render([HD.Hand(dict(HD.POSES['thrust'], wrist=(34, -22)), 'L', HD.euler(166, 66, 26), bld=HD.HEAVY, nails='claw', claw=.03, bands=BANDS)], HD.Cam(250, 300, 120, 1.8), light=(-.3, -.75, .5)), 1.1); cell(2, 2, over(np.broadcast_to(rgb('#c2410e'), (H, W, 3)).copy(), c.layer()))
    g = charcoal()
    im = g[:, :].copy()
    for j, (o, s_) in enumerate([(0., .7), (.16, 1.), (.5, 1.), (1., 1.)]):
        im = draw_eyes(im, [InkEye(390 + 700 * (j % 2), 140 + 270 * (j // 2), 250, 92, .10, 3 + j, 1, .10)], [o], [(.0, 0) if j < 3 else (.3, -.1)], [s_])
    out[2 * ch:, 3 * cw:] = im[:ch, :3 * cw]
    return out


if __name__ == '__main__':
    cv2.setNumThreads(4); os.makedirs('qa/sukuna', exist_ok=True); w = sys.argv[1:] or ['all']; al = 'all' in w
    for nm, fn, fin in (('prayer', key_prayer, dict(vig=.14)), ('smile', key_smile, dict(vig=.12)), ('fuga', key_fuga, dict(vig=.18))):
        if al or nm in w: comp.save(f'qa/sukuna/key_{nm}.jpg', comp.finish(fn(), 0, **fin), 92); print(nm)
    if al or 'sheet' in w: comp.save('qa/sukuna/sheet.jpg', sheet(), 90); print('sheet')
