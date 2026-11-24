"""Letters: kanji as brush skeletons (written with jj.sumi), katakana sound lettering, pen Latin, the red seal.

KANJI['呪'] -> list of strokes in stroke order, on a 1000 x 1000 em.  A stroke is dict(k=[(x, y, width[, speed])...],
c=(corner key indices), + any jj.sumi.drag keyword).   write(page, '呪', box=(x, y, size), t0=0, times=[...])
"""
import numpy as np, cv2, math
from . import W, H
from .paint import rgb, f32, fbm, smooth
from .ink import Cel, spl, resample, ribbon, noise1
from .sumi import Page, Mark, drag, paper, lay, tooth, BLACK, BONE, BLOOD, BLOOD_D, SHEET


# ------------------------------------------------------------ stroke makers (em units)
def S_(k, c=(), **o): return dict(k=[tuple(q) for q in k], c=tuple(c), **o)


def yoko(x0, y0, x1, y1, w, head=1.18, waist=.74, tail=1.16, bow=10, end='tome', **o):
    """horizontal: slanted landing, lean body that arches a little, pressed stop (or harai end)"""
    dx, dy = x1 - x0, y1 - y0
    P = lambda t, b=0.: (x0 + dx * t, y0 + dy * t - bow * 4 * t * (1 - t) * (1 + b))
    k = [(x0 - .34 * w, y0 - .46 * w, .16 * w, .8), (x0 - .02 * w, y0 - .02 * w, head * w, .7),
         (*P(min(.28, .62 * w / abs(dx) + .06)), (waist + .12) * w, 1.2), (*P(.55), waist * w, 1.5), (*P(.84), (waist + .12) * w, 1.1)]
    if end == 'tome': k += [(x1 - .16 * w, y1 - .07 * w, tail * .96 * w, .55), (x1 + .04 * w, y1 + .10 * w, tail * .90 * w, .4)]; c = ()
    elif end == 'thin': k += [(x1, y1, (waist + .05) * w, 1.2), (x1 + .16 * w, y1 + .02 * w, .3 * w, 1.2)]; c = ()
    else: k += [(x1, y1, .12 * w, 1.6)]; c = ()
    return S_(k, c, **o)


def tate(x0, y0, x1, y1, w, head=1.18, waist=.84, end='tome', bow=0, tail=1.0, hook=1.0, tip=1.0, **o):
    """vertical: slanted landing, straight body, then tome | hari (needle) | hane (hook to the upper left)"""
    dx, dy = x1 - x0, y1 - y0
    P = lambda t: (x0 + dx * t + bow * 4 * t * (1 - t), y0 + dy * t)
    k = [(x0 - tip * .36 * w, y0 - tip * .40 * w, .16 * w, .8), (x0, y0, head * w, .7), (*P(min(.3, .75 * w / abs(dy) + .05)), (waist + .10) * w, 1.1), (*P(.58), waist * w, 1.4)]
    c = ()
    if end == 'tome': k += [(*P(.86), (waist + .08) * w, 1.), (x1, y1 - .05 * w, tail * w, .6), (x1 - .03 * w, y1 + .12 * w, .55 * w, .6)]
    elif end == 'thin': k += [(*P(.9), (waist - .02) * w, 1.1), (x1, y1, .6 * w, 1.)]
    elif end == 'hari': k += [(*P(.80), waist * w, 1.3), (x1, y1, .05 * w, 1.7)]
    else:
        k += [(*P(.86), (waist + .05) * w, 1.), (x1 - .02 * w, y1 - .10 * w, 1.05 * w, .6), (x1 - .10 * w, y1 + .10 * w, 1.12 * w, .45),
              (x1 - (.10 + .55 * hook) * w, y1 - .22 * hook * w, .62 * w, 1.2), (x1 - (.10 + 1.15 * hook) * w, y1 - .72 * hook * w, .06 * w, 1.8)]
        c = (len(k) - 3,)
    return S_(k, c, **o)


def ore(x0, y0, xc, yc, x1, y1, w, head=.80, waist=.70, shoulder=1.22, body=.92, end='tome', bow=8, w2=None, hook=1.0, **o):
    """horizontal, pressed shoulder, then down (box corner).  end: tome | hane"""
    w2 = w2 or w
    dx, dy = xc - x0, yc - y0; ex, ey = x1 - xc, y1 - yc; dp = (shoulder * w2 - waist * w) * .5
    k = [(x0 - .10 * w, y0 - .10 * w, .22 * w, .9), (x0 + .10 * w, y0, head * w, 1.0), (x0 + dx * .36, y0 + dy * .36 - bow, (waist + .02) * w, 1.3),
         (x0 + dx * .70, y0 + dy * .70 - bow * .7, waist * w, 1.3), (xc - .30 * w, yc + .02 * dp, (waist + .10) * w, .9),
         (xc + .06 * w, yc + dp * .92, shoulder * w2, .42), (xc + ex * .20 + .02 * w, yc + ey * .20 + dp * .9, (body + .14) * w2, .8),
         (xc + ex * .58, yc + ey * .58, body * w2, 1.25)]
    c = (5,)
    if end == 'tome': k += [(x1, y1 - .12 * w2, (body + .06) * w2, .8), (x1 - .04 * w, y1 + .06 * w2, .62 * w2, .6)]
    else:
        k += [(x1 + .03 * w2, y1 - .12 * w2, (body + .10) * w2, .7), (x1 - .06 * w2, y1 + .10 * w2, 1.12 * w2, .45),
              (x1 - (.06 + .5 * hook) * w2, y1 - .16 * hook * w2, .6 * w2, 1.2), (x1 - (.06 + 1.05 * hook) * w2, y1 - .62 * hook * w2, .06 * w2, 1.8)]
        c = (5, len(k) - 3)
    return S_(k, c, **o)


def ten(x0, y0, x1, y1, w, **o):
    """dot: tip in, belly pressed, small turn back"""
    dx, dy = x1 - x0, y1 - y0
    return S_([(x0, y0, .10 * w, 1.1), (x0 + dx * .45, y0 + dy * .40, .66 * w, 1.), (x0 + dx * .86, y0 + dy * .84, 1.10 * w, .6),
               (x1, y1, .96 * w, .5), (x1 - .16 * w, y1 + .02 * w, .4 * w, .7)], **o)


def hidari(P, w, head=1.16, end=.05, **o):
    """left-falling sweep through P: pressed landing, flying tail"""
    P = [np.array(p, float) for p in P]; x0, y0 = P[0]
    k = [(x0 - .34 * w, y0 - .38 * w, .16 * w, .8), (x0, y0, head * w, .7)]
    n = len(P) - 1
    for i, p in enumerate(P[1:], 1):
        t = i / n; k.append((p[0], p[1], (w * (.90 - .22 * t) if i < n else end * w), 1.1 + .9 * t))
    o.setdefault('fork', .85)
    return S_(k, **o)


def migi(P, w, foot=1.45, **o):
    """right-falling sweep: thin entry, swelling, pressed heel, flat torn exit"""
    P = [np.array(p, float) for p in P]; n = len(P)
    k = []
    for i, p in enumerate(P):
        t = i / (n - 1)
        if i == n - 1: k.append((p[0], p[1], .06 * w, 1.9))
        elif i == n - 2: k.append((p[0], p[1], foot * w, .6))
        else: k.append((p[0], p[1], w * (.28 + (.62 * t / ((n - 2) / (n - 1)))), 1.2))
    o.setdefault('fork', .75)
    return S_(k, **o)


def kuchi(x0, y0, x1, y1, w, lean=.10, rise=.10, **o):
    """box 口 in three strokes inside (x0,y0)-(x1,y1): sides lean in, horizontals rise"""
    dx, dy = x1 - x0, y1 - y0; r = rise * dx; l = lean * dx
    return [tate(x0, y0 + r * .55, x0 + l, y1, w, head=1.12, waist=.84, end='thin', **o),
            ore(x0 + .20 * w, y0 + r * .9, x1, y0, x1 - l, y1 - r * .35, w, end='tome', **o),
            yoko(x0 + l + .30 * w, y1 - .06 * w, x1 - l - .20 * w, y1 - r * .62 - .06 * w, w * .88, head=.82, waist=.80, tail=.86, bow=2, end='thin', **o)]


# ------------------------------------------------------------ 兄, shared by 呪 and 祝
def ani(w=104):
    """the five strokes of 兄 in the right two thirds of the em"""
    return [
        tate(492, 160, 532, 396, w * .96, head=1.14, waist=.84, end='tome', tail=.86, tip=.62, dry=.60, seed=41),
        ore(512, 160, 868, 96, 816, 370, w * .98, waist=.64, shoulder=1.36, body=1.00, bow=12, dry=.58, deplete=.8, split=.7, seed=42),
        yoko(556, 378, 800, 346, w * .80, head=.78, waist=.82, tail=.92, bow=3, end='thin', dry=.62, seed=43),
        hidari([(626, 424), (606, 556), (532, 726), (408, 872), (244, 982)], w * 1.28, head=1.14, dry=.58, deplete=1.15, load=.60, split=.85, fork=1., speed=1.2, seed=44),
        S_([(752 - 40, 414 - 46, 18, .8, 0), (752, 420, 148, .7, -.1), (746, 528, 124, 1.1, .0), (730, 690, 112, 1.3, .10), (748, 816, 114, 1.0, .16), (834, 892, 116, 1.0, .22),
            (948, 908, 112, 1.35, .22), (1036, 892, 118, .9, .0), (1104, 852, 140, .40, -.45), (1122, 780, 86, 1.1, -.40), (1120, 676, 38, 1.4, -.30), (1092, 506, 5, 1.6, -.1)], (8,),
           dry=.46, deplete=.55, load=.72, split=.85, fork=.5, speed=1.1, seed=45),
    ]


KANJI = {}
KANJI['呪'] = [
    tate(58, 346, 100, 618, 98, head=1.12, waist=.84, end='tome', tail=.86, tip=.62, dry=.60, seed=31),
    ore(80, 346, 352, 290, 314, 584, 98, waist=.66, shoulder=1.32, body=.98, bow=9, dry=.58, split=.7, seed=32),
    yoko(122, 596, 306, 566, 82, head=.76, waist=.82, tail=.90, bow=3, end='thin', dry=.60, seed=33),
] + ani()
KANJI['祝'] = [
    ten(168, 66, 262, 178, 112, dry=.30, seed=51),
    S_([(36, 282, 14, .8), (58, 304, 108, .7), (146, 288, 78, 1.2), (262, 258, 80, 1.0), (330, 262, 124, .45), (298, 344, 92, 1.2), (214, 466, 80, 1.4), (112, 576, 66, 1.6), (6, 662, 5, 1.9)], (4,),
       dry=.52, deplete=1.0, load=.7, split=.8, fork=.9, seed=52),
    tate(214, 486, 214, 944, 104, head=.84, waist=.90, tip=.7, end='hari', dry=.50, deplete=1.1, load=.62, split=.8, fork=1., seed=53),
    ten(266, 470, 384, 606, 98, dry=.30, seed=54),
] + ani()
KANJI['生'] = [
    hidari([(318, 84), (292, 196), (222, 340), (104, 470)], 100, dry=.50, deplete=.9, split=.8, fork=.9, seed=61),
    yoko(262, 338, 800, 274, 94, dry=.56, bow=16, waist=.70, seed=62),
    tate(520, 40, 520, 872, 116, waist=.88, end='thin', dry=.56, deplete=.9, load=.7, split=.75, seed=63),
    yoko(286, 610, 762, 562, 90, dry=.50, bow=12, waist=.72, seed=64),
    yoko(60, 894, 944, 838, 120, dry=.60, deplete=.9, load=.6, bow=20, tail=1.22, waist=.70, split=.8, seed=65),
]
_d = dict(dry=.5, deplete=.9, split=.75)
KANJI['領'] = [
    hidari([(252, 70), (196, 206), (42, 378)], 70, **_d), ten(250, 130, 452, 326, 74, **_d), yoko(186, 380, 318, 366, 56, bow=3, **_d),
    S_([(84, 494, 10, .8), (104, 512, 66, .7), (210, 494, 50, 1.2), (312, 478, 56, .9), (346, 486, 78, .45), (316, 556, 58, 1.1), (262, 630, 44, 1.4), (196, 696, 6, 1.8)], (4,), fork=.6, **_d),
    ten(150, 654, 316, 862, 78, **_d),
    yoko(486, 112, 956, 70, 62, bow=12, **_d), hidari([(730, 122), (704, 186), (640, 250)], 54, head=.95, **_d),
    tate(546, 262, 560, 760, 62, tip=.6, end='thin', **_d), ore(562, 268, 892, 236, 884, 748, 62, waist=.64, shoulder=1.22, body=.94, bow=6, **_d),
    yoko(592, 410, 856, 394, 48, head=.8, bow=2, end='thin', **_d), yoko(592, 566, 856, 552, 48, head=.8, bow=2, end='thin', **_d), yoko(580, 732, 874, 718, 52, head=.8, bow=2, end='thin', **_d),
    hidari([(662, 772), (592, 862), (462, 948)], 62, **_d), ten(772, 778, 962, 944, 78, **_d)]
KANJI['域'] = [
    yoko(52, 398, 356, 352, 70, bow=6, **_d), tate(206, 150, 204, 712, 78, end='thin', **_d),
    S_([(26, 772, 12, .8), (52, 800, 82, .6), (150, 772, 66, 1.1), (270, 724, 46, 1.5), (404, 652, 5, 1.9)], fork=.6, **_d),
    yoko(408, 322, 948, 252, 70, bow=12, **_d)] + kuchi(462, 438, 640, 596, 56, dry=.45) + [
    S_([(392, 770, 10, .8), (414, 794, 72, .6), (500, 766, 56, 1.1), (590, 724, 40, 1.5), (694, 664, 5, 1.9)], fork=.6, **_d),
    S_([(660, 26, 14, .8), (690, 64, 88, .7), (700, 220, 72, 1.2), (734, 450, 72, 1.3), (802, 680, 78, 1.2), (878, 830, 86, 1.0), (934, 900, 92, .45), (950, 840, 62, 1.2), (958, 704, 5, 1.9)], (6,), fork=.5, load=.8, **_d),
    hidari([(862, 420), (800, 596), (700, 752), (560, 900)], 66, **_d), ten(800, 92, 912, 196, 74, **_d)]
KANJI['展'] = [
    ore(238, 122, 800, 78, 780, 262, 66, waist=.70, shoulder=1.2, body=.92, bow=8, **_d), yoko(262, 278, 776, 252, 54, head=.85, bow=4, end='thin', **_d),
    hidari([(216, 110), (214, 360), (178, 620), (56, 904)], 74, **_d),
    yoko(316, 432, 822, 396, 54, bow=8, **_d), tate(444, 336, 456, 556, 56, end='thin', **_d), tate(652, 322, 640, 544, 56, end='thin', **_d),
    yoko(236, 590, 908, 534, 64, bow=14, **_d),
    S_([(322, 606, 14, .8), (346, 634, 72, .7), (342, 740, 56, 1.2), (334, 862, 62, .9), (336, 900, 82, .4), (404, 868, 54, 1.3), (516, 786, 5, 1.9)], (4,), fork=.5, **_d),
    hidari([(778, 626), (714, 686), (604, 752)], 56, head=1., **_d), migi([(474, 600), (566, 716), (680, 830), (800, 908), (952, 922)], 74, **_d)]
KANJI['開'] = [
    tate(112, 92, 118, 930, 72, end='thin', **_d), ore(126, 96, 408, 72, 398, 380, 60, waist=.66, shoulder=1.2, body=.9, bow=5, **_d),
    yoko(148, 238, 384, 224, 46, head=.8, bow=2, end='thin', **_d), yoko(148, 384, 388, 368, 48, head=.8, bow=2, end='thin', **_d),
    tate(584, 78, 594, 376, 58, tip=.6, end='thin', **_d), ore(596, 84, 896, 54, 898, 886, 68, waist=.62, shoulder=1.24, body=.94, bow=6, end='hane', hook=1.1, **_d),
    yoko(612, 224, 872, 208, 46, head=.8, bow=2, end='thin', **_d), yoko(612, 368, 872, 352, 48, head=.8, bow=2, end='thin', **_d),
    yoko(288, 524, 720, 494, 54, bow=8, **_d), yoko(232, 690, 776, 650, 60, bow=10, **_d),
    hidari([(414, 530), (402, 690), (272, 900)], 62, head=1., **_d), tate(604, 506, 606, 900, 66, end='hari', fork=.8, **_d)]
KANJI['五'] = [
    yoko(168, 136, 850, 92, 92, bow=14, **_d), tate(486, 150, 384, 838, 92, end='thin', **_d),
    ore(236, 452, 744, 410, 690, 820, 92, waist=.70, shoulder=1.28, body=.96, bow=10, **_d), yoko(56, 896, 952, 858, 110, bow=18, waist=.68, **_d)]
KANJI['条'] = [
    hidari([(372, 40), (296, 208), (112, 374)], 80, **_d),
    S_([(322, 168, 12, .8), (344, 190, 82, .7), (460, 176, 62, 1.2), (612, 154, 68, .9), (676, 160, 100, .45), (626, 250, 74, 1.1), (476, 400, 62, 1.4), (300, 506, 50, 1.6), (114, 572, 5, 1.9)], (4,), fork=.8, **_d),
    migi([(322, 246), (450, 378), (610, 490), (760, 560), (962, 580)], 86, **_d),
    yoko(152, 690, 858, 640, 76, bow=14, **_d), tate(510, 566, 508, 936, 88, end='thin', **_d),
    hidari([(450, 716), (350, 828), (176, 918)], 72, head=1., **_d), ten(572, 722, 832, 896, 88, **_d)]
KANJI['悟'] = [
    S_([(124, 330, 8, 1.), (106, 400, 44, 1.), (80, 480, 74, .6), (66, 540, 64, .5)], **_d), ten(286, 292, 356, 424, 72, **_d), tate(206, 62, 208, 938, 80, end='thin', **_d),
    yoko(444, 120, 900, 84, 62, bow=8, **_d), tate(632, 130, 566, 440, 60, end='thin', **_d), ore(464, 284, 838, 254, 800, 438, 60, waist=.70, shoulder=1.22, body=.92, bow=6, **_d),
    yoko(368, 492, 968, 452, 72, bow=12, waist=.70, **_d)] + kuchi(484, 604, 858, 892, 66, lean=.08, rise=.09, dry=.45)


def write(pg, ch, box, t0=0., times=None, wt=1.0, seed=0, strokes=None, rot=0., **over):
    """write a kanji on a Page.  box=(x, y, size).  times = [(start, end), ...] per stroke (else 1 unit each, back to back).
    Returns the list of Marks."""
    x, y, s = box; k = s / 1000.; cr, sr = math.cos(rot), math.sin(rot)
    T = lambda a, b: (x + (500 + (a - 500) * cr - (b - 500) * sr) * k, y + (500 + (a - 500) * sr + (b - 500) * cr) * k)
    st = strokes or KANJI[ch]; out = []
    for i, q in enumerate(st):
        q = dict(q); keys = [T(a[0], a[1]) + (a[2] * k * wt,) + tuple(a[3:]) for a in q.pop('k')]; m = max(len(a) for a in keys); keys = [a + (1., 0.)[len(a) - 3:m - 3] for a in keys]; c = q.pop('c')
        a, b = times[i] if times else (t0 + i, t0 + i + 1)
        q['seed'] = q.get('seed', 7) + seed * 1009; q.update(over)
        out.append(pg.stroke(keys, c, t0=a, t1=b, **q))
    return out


# ============================================================ katakana sound lettering (polygons with attitude)
# strokes on a 1000 em: (points, widths, start cut, end cut) ; cut = shear of the end edge (in widths), None = point
KANA = {
    'ド': [([(392, 20), (404, 420), (424, 990)], [150, 148, 92], -.55, .9), ([(470, 372), (640, 470), (838, 640)], [70, 128, 108], .3, -.5),
           ([(618, 40), (668, 250)], [96, 84], -.5, .4), ([(792, 4), (846, 214)], [96, 80], -.5, .4)],
    'ン': [([(110, 160), (236, 236), (372, 356)], [96, 128, 120], .5, -.6), ([(96, 908), (330, 800), (640, 560), (930, 180)], [150, 140, 104, 26], -.8, None)],
    'ゴ': [([(120, 206), (460, 188), (792, 150)], [104, 112, 128], .7, 0), ([(776, 110), (772, 480), (748, 830)], [132, 136, 124], .2, -.2),
           ([(70, 872), (460, 850), (884, 812)], [124, 132, 120], .8, -.8), ([(838, -20), (884, 120)], [82, 72], -.5, .4), ([(968, -52), (1014, 88)], [82, 70], -.5, .4)],
    'コ': [([(120, 206), (460, 188), (812, 150)], [104, 112, 128], .7, 0), ([(796, 110), (792, 480), (768, 830)], [132, 136, 124], .2, -.2), ([(70, 872), (460, 850), (904, 812)], [124, 132, 120], .8, -.8)],
    'ガ': [([(70, 342), (420, 300), (760, 262)], [108, 116, 128], .8, 0), ([(744, 222), (738, 520), (690, 800), (640, 900)], [130, 132, 122, 110], .2, -.2), ([(668, 914), (600, 880), (520, 790)], [104, 90, 20], 0, None),
           ([(404, 40), (396, 330), (306, 640), (90, 930)], [122, 124, 108, 30], -.5, None), ([(806, 20), (862, 180)], [88, 76], -.5, .4), ([(948, -30), (1004, 132)], [88, 74], -.5, .4)],
    'バ': [([(352, 250), (270, 530), (90, 880)], [150, 132, 44], -.6, None), ([(520, 190), (690, 480), (930, 900)], [64, 132, 168], .6, -.7), ([(668, 20), (728, 190)], [94, 80], -.5, .4), ([(822, -26), (880, 146)], [94, 78], -.5, .4)],
    'ッ': [([(150, 400), (222, 620)], [104, 84], -.5, .5), ([(398, 344), (470, 560)], [106, 84], -.5, .5), ([(838, 300), (760, 560), (600, 790), (398, 940)], [132, 120, 96, 24], -.5, None)],
    'ー': [([(60, 520), (480, 500), (940, 470)], [120, 130, 150], .8, -.9)],
}


def _slab(P, Wd, c0, c1, r, rag=1.0):
    """one lettering stroke -> outline polygon with hacked edges and torn ends"""
    P = np.asarray(P, float); Wd = np.asarray(Wd, float)
    C = spl(P, n=14) if len(P) > 2 else np.linspace(P[0], P[-1], 16)
    n = len(C); t = np.linspace(0, 1, n); wd = np.interp(t, np.linspace(0, 1, len(Wd)), Wd)
    T = np.gradient(C, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]
    def side(sg):
        hw = wd * .5 * (1 + rag * .030 * noise1(n, 5, r) + rag * .020 * noise1(n, 2, r))
        for _ in range(r.integers(0, 3)):                       # a hack out of the edge
            k = int(r.integers(2, max(3, n - 2))); hw[k] *= r.uniform(.84, .95)
        return C + sg * N * hw[:, None]
    A, B = side(1), side(-1)
    def end(a, b, T_, cut):                                     # torn edge from a to b
        if cut is None: return np.zeros((0, 2))
        m = int(r.integers(4, 8)); u = np.sort(r.random(m)); L = np.hypot(*(b - a))
        off = (cut * (u - .5) + rag * r.uniform(-.10, .13, m) * np.where(r.random(m) < .4, 1.8, 1)) * L
        return a + (b - a) * u[:, None] + T_ * off[:, None]
    if c0 is not None:
        A = A.copy(); B = B.copy(); A[0] += T[0] * (-c0 * .5) * wd[0]; B[0] += T[0] * (c0 * .5) * wd[0]
    if c1 is not None: A[-1] += T[-1] * (-c1 * .5) * wd[-1]; B[-1] += T[-1] * (c1 * .5) * wd[-1]
    else: A[-1] = B[-1] = C[-1] + T[-1] * wd[-1] * .6
    e1 = end(A[-1], B[-1], T[-1], 0 if c1 is not None else None); e0 = end(B[0], A[0], T[0], 0 if c0 is not None else None)
    poly = np.vstack([A, e1, B[::-1], e0])
    sl = []                                                     # flying-white slivers
    for _ in range(int(r.integers(1, 4))):
        k0 = r.uniform(.0, .55) if r.random() < .5 else r.uniform(.55, .95); ln = r.uniform(.12, .38); q = r.uniform(-.40, .40); tt = np.linspace(k0, min(1.04, k0 + ln), 7)
        I = np.clip((tt * (n - 1)).astype(int), 0, n - 1); c = C[I] + N[I] * (q * wd[I])[:, None]; th = wd[I] * r.uniform(.008, .026) * np.sin(np.linspace(.2, 2.9, 7))
        sl.append(np.vstack([c + N[I] * th[:, None], (c - N[I] * th[:, None])[::-1]]))
    return poly, sl


def homog(quad):
    """unit square -> quad (tl, tr, br, bl): 3x3"""
    return cv2.getPerspectiveTransform(np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), np.float32(quad))


def persp(P, M):
    P = np.asarray(P, float); q = np.c_[P, np.ones(len(P))] @ M.T; return q[:, :2] / q[:, 2:3]


def sfx_polys(word, quad, seed=0, vertical=False, adv=1.0, grow=0., jit=.04, rag=1.0, heavy=1.28):
    """[(outline, [slivers])] in screen px for a sound word laid out in a perspective quad (tl, tr, br, bl)"""
    r = np.random.default_rng(seed); M = homog(quad); n = len(word); out = []
    sz = np.array([1 + grow * i for i in range(n)]); off = np.r_[0, np.cumsum(sz * adv)]; tot = off[-1]
    for i, ch in enumerate(word):
        k = sz[i]; j = r.normal(0, jit, 2); rot = r.normal(0, jit * 1.2); small = ch == 'ッ'
        for P, Wd, c0, c1 in KANA[ch]:
            if small: P = (np.asarray(P, float) - [520, 900]) * .74 + [460, 900]; Wd = np.asarray(Wd, float) * .82
            poly, sl = _slab(P, np.asarray(Wd, float) * heavy, c0, c1, r, rag)
            def f(q):
                a = q / 1000. - .5; c, s = math.cos(rot), math.sin(rot); a = a @ np.array([[c, s], [-s, c]]); a = (a + .5 + j) * k
                if vertical: uv = np.c_[a[:, 0] / sz.max() + (1 - k / sz.max()) * .5, (a[:, 1] + off[i]) / tot]
                else: uv = np.c_[(a[:, 0] + off[i]) / tot, a[:, 1] / sz.max() + (1 - k / sz.max()) * .5]
                return persp(uv, M)
            out.append((f(poly), [f(q) for q in sl]))
    return out


def sfx(cel, word, quad, seed=0, ink=BLACK, halo='#f4f1ea', halo_w=7., slivers=True, hatch=0, hatch_dir=None, **k):
    """draw sound lettering on a Cel: white keyline, black slabs, flying-white slivers, optional speed-line hatching behind (hatch = count)"""
    S = sfx_polys(word, quad, seed, **k); r = np.random.default_rng(seed + 77)
    if hatch:
        q = np.asarray(quad, float); d = np.asarray(hatch_dir if hatch_dir is not None else q[3] - q[0], float); d /= np.hypot(*d)
        for poly, _ in S:
            for _ in range(hatch):
                p = poly[r.integers(len(poly))] + r.normal(0, 5, 2); L = r.uniform(30, 150)
                cel.line(np.array([p, p + d * L]), r.uniform(1., 2.6), ink, taper=(.05, .8), press=.3, seed=int(r.integers(1 << 30)))
    if halo and halo_w:
        import cairo
        c = cel.ctx
        for poly, _ in S:
            cel._path(poly); cel._src(halo, 1.0); c.set_line_width(halo_w * 2); c.set_line_join(cairo.LINE_JOIN_MITER); c.set_miter_limit(3); c.stroke()
    for poly, sl in S: cel.fill(poly, ink)
    if slivers:
        for poly, sl in S:
            for q in sl: cel.fill(q, halo or '#f4f1ea', .9)
    return cel


# ============================================================ pen Latin (x-height 100, baseline 0, y down)
LATIN = {
    'a': (78, 'M62 -90C44 -106 10 -84 6 -40C4 -8 22 6 40 -10C54 -24 62 -54 67 -96C64 -70 58 -40 58 -22C58 4 70 4 86 -12'),
    'b': (74, 'M20 -188C14 -130 8 -70 6 -32C6 2 30 8 48 -16C68 -44 66 -98 40 -99C24 -99 13 -78 8 -50'),
    'c': (62, 'M56 -84C46 -108 10 -94 6 -42C3 4 40 8 64 -20'),
    'd': (84, 'M62 -90C44 -106 10 -84 6 -40C4 -8 22 6 40 -10C56 -26 66 -80 80 -186C72 -120 62 -50 62 -24C62 4 74 4 90 -12'),
    'e': (64, 'M8 -46C40 -48 60 -66 55 -86C48 -108 10 -94 6 -42C3 6 44 8 68 -20'),
    'f': (54, 'M76 -166C64 -196 44 -188 36 -142C28 -90 22 -14 12 38C6 74 -14 86 -26 68M2 -94C20 -98 44 -98 62 -101'),
    'g': (74, 'M62 -90C44 -106 10 -84 6 -42C4 -12 24 0 42 -16C56 -32 62 -60 67 -96C64 -40 60 4 50 44C40 84 4 88 -10 64C-14 46 14 34 44 28'),
    'h': (80, 'M24 -188C18 -124 10 -60 6 0M9 -48C20 -84 46 -110 58 -94C68 -78 54 -40 52 -18C52 4 66 4 82 -14'),
    'i': (44, 'M8 -76C18 -94 29 -100 27 -84C24 -58 14 -32 15 -14C16 4 30 2 44 -14M34 -142C36 -142 38 -138 37 -136'),
    'j': (36, 'M8 -76C18 -94 29 -100 26 -80C20 -30 14 20 8 48C0 80 -22 84 -32 66M34 -142C36 -142 38 -138 37 -136'),
    'k': (72, 'M24 -188C18 -124 10 -60 6 0M58 -98C44 -78 26 -58 11 -50C28 -46 36 -22 46 -6C54 6 66 2 78 -12'),
    'l': (42, 'M32 -188C26 -130 16 -64 14 -24C14 4 28 4 44 -14'),
    'm': (122, 'M4 -78C14 -96 22 -104 22 -88C20 -60 14 -28 10 0M13 -46C24 -84 46 -108 56 -92C64 -76 52 -30 48 0M51 -46C62 -84 86 -110 98 -94C106 -78 94 -40 92 -18C92 4 106 4 122 -14'),
    'n': (84, 'M4 -78C14 -96 22 -104 22 -88C20 -60 14 -28 10 0M13 -46C24 -84 48 -110 60 -94C68 -78 56 -40 54 -18C54 4 68 4 84 -14'),
    'o': (72, 'M42 -98C18 -100 4 -62 6 -34C8 4 44 4 56 -34C66 -66 56 -98 40 -98C48 -84 62 -84 76 -92'),
    'p': (76, 'M4 -78C14 -96 22 -104 21 -86C16 -30 8 30 2 88M14 -46C26 -84 50 -110 62 -84C70 -50 46 4 12 -10'),
    'q': (74, 'M62 -90C44 -106 10 -84 6 -42C4 -12 24 0 42 -16C56 -32 62 -60 67 -96C62 -30 56 30 50 88'),
    'r': (56, 'M4 -78C14 -96 22 -104 22 -88C20 -60 14 -28 10 0M14 -50C24 -84 42 -108 60 -92'),
    's': (56, 'M54 -86C42 -106 14 -100 16 -76C18 -54 48 -50 46 -24C42 6 10 6 2 -14'),
    't': (50, 'M30 -146C26 -100 16 -50 16 -22C18 4 34 4 50 -14M2 -94C18 -98 36 -98 52 -101'),
    'u': (84, 'M8 -78C18 -96 28 -102 26 -84C20 -54 8 -20 18 -4C32 8 54 -40 66 -96C62 -64 58 -38 58 -20C58 4 70 4 86 -12'),
    'v': (68, 'M6 -84C16 -100 26 -96 26 -78C24 -50 20 -20 22 -2C40 -20 64 -56 58 -98C62 -92 70 -90 78 -94'),
    'w': (104, 'M6 -84C16 -100 26 -96 24 -76C18 -44 12 -6 24 -2C40 -4 54 -50 60 -94C54 -50 50 -8 62 -2C82 -8 98 -60 92 -98C96 -92 104 -90 112 -94'),
    'x': (72, 'M4 -84C18 -104 26 -92 32 -60C36 -24 44 4 58 0C66 -4 72 -10 76 -16M66 -96C48 -84 28 -30 4 -4'),
    'y': (78, 'M8 -78C18 -96 28 -102 26 -84C20 -54 8 -20 18 -4C32 8 54 -40 66 -96C60 -40 54 10 44 46C34 84 4 88 -10 66'),
    'z': (66, 'M8 -84C20 -102 36 -88 60 -96C46 -70 24 -34 4 -4C20 -18 38 2 64 -14'),
    'A': (126, 'M-6 -2C10 8 26 -8 40 -40C58 -84 80 -140 100 -186C98 -130 96 -60 98 -18C100 4 112 4 126 -12M34 -62C58 -72 88 -74 116 -70'),
    'G': (118, 'M106 -150C100 -186 70 -194 42 -158C14 -120 0 -60 12 -22C24 10 64 4 84 -30C94 -48 98 -64 100 -78C98 -50 92 -20 92 4C88 40 60 64 34 58M62 -76C80 -82 104 -82 124 -80'),
    'N': (130, 'M-4 -4C10 8 18 -20 26 -60C34 -110 40 -160 46 -184C62 -130 84 -60 96 -4C104 -60 112 -130 124 -176C128 -190 140 -190 146 -180'),
    'I': (58, 'M16 -180C34 -188 48 -186 62 -184M40 -184C32 -120 24 -60 18 -4M-2 -2C14 -6 28 -4 40 -4'),
    '/': (60, 'M56 -190C40 -120 20 -40 0 30'), ',': (30, 'M14 -8C20 -4 16 14 2 26'), '.': (28, 'M12 -6C15 -6 16 -3 15 -1'), '-': (56, 'M8 -48C22 -52 38 -50 48 -50'),
}


def latin(text, x, y, size=34., seed=0, pen=.085, slant=.20, nib=math.radians(38), track=.06, align='left', uneven=1.0):
    """hand-lettered line -> [(stroke polygon, order 0..1)].  x-height = size * .5; baseline at y.  Order lets it be written over time."""
    from .kit import svg
    r = np.random.default_rng(seed); k = size / 200.; strokes = []; cx = 0.; n = len(text)
    for ch in text:
        if ch == ' ': cx += 58 + r.normal(0, 4); continue
        adv, d = LATIN[ch]; sc = 1 + .025 * uneven * r.normal(); dy = 2.6 * uneven * r.normal(); rot = .016 * uneven * r.normal()
        for q in svg(d, 12):
            q = q * sc; q = np.c_[q[:, 0] + q[:, 1] * (-slant), q[:, 1]]; c, s = math.cos(rot), math.sin(rot)
            q = q @ np.array([[c, s], [-s, c]]); strokes.append(q + [cx, dy])
        cx += adv * (1 + track) + r.normal(0, 2.0 * uneven)
    ox = {'left': 0, 'center': -cx / 2, 'right': -cx}[align]
    L = np.cumsum([0] + [len(q) for q in strokes]); out = []
    for i, q in enumerate(strokes):
        P = resample((q + [ox, 0]) * k + [x, y], .5); m = len(P)
        if m < 4: P = np.vstack([P, P + [[size * .02, size * .01]]]); P = resample(P, .3); m = len(P)
        T = np.gradient(P, axis=0); ang = np.arctan2(T[:, 1], T[:, 0]); w = size * pen * (.34 + .66 * np.abs(np.sin(ang + nib)) ** 1.3)
        t = np.linspace(0, 1, m); w = w * (1 + .16 * noise1(m, max(6, size * .6), r)) * (.45 + .55 * np.clip(np.minimum(t * m, (1 - t) * m) / (size * .16), 0, 1) ** .6)
        # ink gathers where the pen stops
        w = w + size * pen * .28 * np.exp(-((1 - t) * m * .5 / (size * .05)) ** 2)
        T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]] * w[:, None] * .5
        out.append((P, N, L[i] / L[-1], L[i + 1] / L[-1]))
    return out


def latin_ink(lines, t=1.0, h=H, w=W, ss=4, seed=0):
    """rasterise latin() strokes written up to t (0..1) -> ink thickness HxW.  lines = list of latin() results (written in order)"""
    al = [s for ln in lines for s in ln]
    P = np.vstack([s[0] for s in al]); x0, y0 = np.floor(P.min(0) - 8).astype(int); x1, y1 = np.ceil(P.max(0) + 8).astype(int)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    b = np.zeros(((y1 - y0) * ss, (x1 - x0) * ss), f32); nl = len(lines)
    for li, ln in enumerate(lines):
        for C, N, a0, a1 in ln:
            a, e = (li + a0) / nl, (li + a1) / nl
            u = np.clip((t - a) / max(1e-6, e - a), 0, 1)
            m = int(u * (len(C) - 1)) + 1
            if u <= 0 or m < 2: continue
            for i in range(0, m - 1):
                q = np.array([C[i] + N[i], C[i + 1] + N[i + 1], C[i + 1] - N[i + 1], C[i] - N[i]])
                cv2.fillConvexPoly(b, np.round((q - [x0, y0]) * ss * 16).astype(np.int32), 1., cv2.LINE_AA, 4)
                cv2.circle(b, tuple(np.round((C[i] - [x0, y0]) * ss * 16).astype(int)), max(1, int(np.hypot(*N[i]) * ss * 16)), 1., -1, cv2.LINE_AA, 4)
    b = cv2.GaussianBlur(b, (0, 0), ss * .22)
    f = cv2.resize(tooth(b.shape[0], b.shape[1], 5), (b.shape[1], b.shape[0])) if b.shape[0] <= 2304 and b.shape[1] <= 2304 else np.tile(tooth(2304, 2304), (1 + b.shape[0] // 2304, 1 + b.shape[1] // 2304))[:b.shape[0], :b.shape[1]]
    b = smooth(.28, .72, b + (f - .5) * .30 * (b > .02) * (1 - b * .4)) * (.80 + .20 * cv2.GaussianBlur(f, (0, 0), 6))
    out = np.zeros((h, w), f32); out[y0:y1, x0:x1] = cv2.resize(b.astype(f32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    return out


# ============================================================ the seal
def seal(size=150, seed=0, ss=4):
    """red seal with 生 in seal script, carved white (hakubun): (ink thickness size x size).  Uneven pressure, chipped rim, dry paste."""
    r = np.random.default_rng(seed); n = int(size * ss)
    def rough(P, amp, sc=14.):
        Q = resample(np.vstack([P, P[:1]]), 3.0); m = len(Q); T = np.gradient(Q, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9
        return Q + np.c_[-T[:, 1], T[:, 0]] * (amp * (noise1(m, sc, r) + .6 * noise1(m, 4, r)))[:, None]
    u = n / 200.
    b = np.zeros((n, n), f32)
    sq = spl(np.array([(8, 14), (14, 7), (100, 9), (187, 6), (194, 14), (192, 100), (195, 186), (188, 194), (100, 192), (13, 195), (6, 187), (9, 100)], float), True, n=6) * u
    cv2.fillPoly(b, [np.round(rough(sq, 1.3 * u) * 16).astype(np.int32)], 1., cv2.LINE_AA, 4)
    def cut(P, w):
        P = resample(np.asarray(P, float) * u, 2.); m = len(P); T = np.gradient(P, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9
        hw = w * u * .5 * (1 + .10 * noise1(m, 26, r) + .05 * noise1(m, 7, r)); N = np.c_[-T[:, 1], T[:, 0]] * hw[:, None]
        cv2.fillPoly(b, [np.round(rough(np.vstack([P + N, (P - N)[::-1]]), .35 * u, 8) * 16).astype(np.int32)], 0., cv2.LINE_AA, 4)
    cut([(101, 24), (99, 90), (101, 164)], 15)                                                   # stem
    cut(spl(np.array([(37, 34), (36, 66), (46, 86), (72, 93), (100, 94), (128, 93), (154, 86), (164, 66), (163, 34)], float), n=8), 14)   # rising arms
    cut([(52, 128), (100, 126), (149, 129)], 14); cut([(26, 165), (100, 163), (175, 166)], 15)   # 土
    for _ in range(9):                                                                           # chips
        t = r.uniform(0, 4); e = r.choice([8, 192]) + r.normal(0, 3); p = (r.uniform(10, 190), e) if r.random() < .5 else (e, r.uniform(10, 190))
        R = r.uniform(1.5, 5.5) * u; cv2.fillPoly(b, [np.round(blob_(p[0] * u, p[1] * u, R, r) * 16).astype(np.int32)], 0., cv2.LINE_AA, 4)
    f = np.tile(tooth(2304, 2304), (1 + n // 2304, 1 + n // 2304))[:n, :n]
    y, x = np.mgrid[0:n, 0:n] / n
    press = .82 + .26 * np.cos((x - .30) * 1.9) * np.cos((y - .64) * 1.7) + .085 * fbm(n, n, (n / 5., n / 12., n / 30.), (1, .7, .5), seed) - .10
    ink = b * smooth(.0, .30, press - .62 * smooth(.45, 1.0, cv2.GaussianBlur(f, (0, 0), 1.2)) * 1.0 + .10)
    ink = ink * (.74 + .26 * smooth(.6, 1.0, press + .12 * fbm(n, n, (n / 40., n / 90.), (1, .6), seed + 2)))
    ed = cv2.GaussianBlur(b, (0, 0), 2.2 * u); ink = ink * (1 + .10 * (b - ed) * 4)               # paste squeezes to the edges
    return np.clip(cv2.resize(ink.astype(f32), (size, size), interpolation=cv2.INTER_AREA), 0, 1)


def blob_(x, y, R, r):
    a = np.linspace(0, 2 * math.pi, 12, endpoint=False); q = R * (1 + .3 * r.normal(0, 1, 12)); return np.c_[x + np.cos(a) * q, y + np.sin(a) * q]


def stamp(base, x, y, size=150, rot=-.04, seed=0, col='#b4161c', amt=1.0):
    """press the seal on an RGB picture, centre (x, y)"""
    s = seal(size, seed); h, w = base.shape[:2]
    M = cv2.getRotationMatrix2D((size / 2, size / 2), math.degrees(-rot), 1.); M[:, 2] += [x - size / 2, y - size / 2]
    a = cv2.warpAffine(s, M, (w, h), flags=cv2.INTER_LINEAR)[..., None] * amt
    return base * (1 - a) + (rgb(col) * (.62 + .38 * np.clip(base, 0, 1))) * a
