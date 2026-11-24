"""two full-finish fx test paintings (QA only): black flash at impact, flame column"""
from .fx import *
from .paint import ramp, repaint, Canvas, arcs
from .kit import svg, s1, outline, hatch, rim, para, drybrush
from . import comp
import os


def _xf(P, s, rot, tx, ty, flip=1):
    P = np.asarray(P, float) * [s * flip, s]; c, q = math.cos(rot), math.sin(rot); return np.c_[P[:, 0] * c - P[:, 1] * q + tx, P[:, 0] * q + P[:, 1] * c + ty]


def fist(cel, s, rot, tx, ty, lw=4., skin='#e7a685', sh='#a8566a', dk='#5c2340', ink='#17131a', lit='#fff0e2'):
    """thumb-side fist flying to the right, forearm + rolled dark sleeve behind it (local units ~ 340 wide)"""
    T = lambda d: _xf(s1(d), s, rot, tx, ty)
    arm = T('M -20 -78 C -160 -104 -300 -112 -420 -128 L -1100 -250 L -1100 330 L -430 168 C -300 150 -150 122 -10 104 Z'); cel.fill(arm, skin)
    cel.shade(T('M -10 40 C -160 46 -320 40 -520 20 L -1100 60 L -1100 340 L -430 170 C -300 152 -150 124 -10 106 Z'), sh, arm); cel.shade(T('M -10 84 C -160 96 -320 104 -520 112 L -1100 220 L -1100 340 L -430 170 C -300 152 -150 124 -10 106 Z'), dk, arm)
    for d in ('M -60 -40 C -140 -36 -230 -44 -330 -70', 'M -120 -10 C -200 -2 -290 -10 -380 -22', 'M -70 24 C -150 34 -220 30 -300 22'): cel.line(T(d), lw * .5, ink, taper=(.3, .6))
    outline(cel, arm, lw, ink, 3)
    slv = T('M -372 -146 C -396 -60 -398 60 -366 186 C -520 214 -800 290 -1200 420 L -1200 -420 C -800 -250 -520 -180 -372 -146 Z'); cel.fill(slv, '#15141f'); cel.shade(T('M -384 -128 C -520 -160 -780 -226 -1200 -370 L -1200 -250 C -900 -180 -700 -120 -520 -92 C -470 -86 -420 -84 -392 -86 Z'), '#4a3f7a', slv)
    for d in ('M -386 -70 L -470 -48 L -560 -88 L -700 -60 L -540 -22 L -440 -30 Z', 'M -384 30 L -452 70 L -610 40 L -760 96 L -560 92 L -430 110 Z', 'M -640 -130 L -760 -120 L -980 -190 L -840 -96 L -700 -92 Z', 'M -700 130 L -860 150 L -1060 260 L -880 210 Z'): cel.shade(T(d), '#4a3f7a', slv)
    outline(cel, slv, lw * 1.1, ink, 5); cel.line(T('M -372 -146 C -340 -60 -342 70 -366 186'), lw * 1.6, ink)
    cuff = T('M -372 -148 C -348 -152 -326 -146 -318 -136 C -296 -50 -298 80 -318 170 C -330 184 -350 190 -366 188 C -342 70 -342 -50 -372 -148 Z'); cel.fill(cuff, '#1e1c2c'); cel.shade(T('M -360 -148 C -336 -150 -324 -144 -318 -136 C -304 -80 -300 -30 -302 10 C -322 -50 -342 -100 -360 -148 Z'), '#5a4f90', cuff); outline(cel, cuff, lw, ink, 7)
    # lower fingers (pinky .. middle) as steps below the index
    for d in ('M 210 40 C 250 60 262 92 250 120 C 238 142 206 148 176 138 C 140 128 120 100 130 70 Z', 'M 226 20 C 276 38 292 70 282 96 C 270 120 232 126 200 112 C 168 96 160 70 170 46 Z', 'M 240 0 C 298 14 314 42 306 68 C 296 92 256 98 224 84 C 190 66 186 46 196 24 Z'):
        P = T(d); cel.fill(P, sh); cel.shade(P + [6 * s, -12 * s], skin, P); outline(cel, P, lw * .8, ink, 11)
    hand = T('M -20 -80 C 40 -104 120 -128 196 -136 C 232 -140 270 -132 292 -116 C 316 -98 330 -70 332 -40 C 334 -12 326 14 310 26 C 292 40 256 42 224 38 C 196 60 150 76 120 78 C 130 100 120 124 96 130 C 60 136 10 120 -20 106 C -60 60 -60 -40 -20 -80 Z'); cel.fill(hand, skin)
    cel.shade(T('M -30 20 C 30 40 80 30 120 6 C 170 -20 230 -6 270 16 C 300 24 320 10 334 -20 L 340 60 L 200 160 L -60 140 Z'), sh, hand); cel.shade(T('M -30 84 C 20 92 70 96 100 78 C 130 60 170 52 224 40 L 240 120 L -60 150 Z'), dk, hand)
    cel.shade(T('M 252 -128 C 290 -116 316 -86 322 -50 C 326 -30 322 -14 316 -4 C 312 -40 296 -80 262 -104 Z'), lit, hand); cel.shade(T('M 60 -110 C 120 -126 180 -132 214 -128 C 180 -116 120 -108 70 -96 Z'), lit, hand)
    outline(cel, hand, lw, ink, 13)
    thumb = T('M 40 60 C 80 34 130 20 176 22 C 214 22 246 28 270 42 C 290 54 290 72 270 78 C 240 84 206 78 176 80 C 130 100 80 110 44 100 C 24 90 24 74 40 60 Z'); cel.fill(thumb, skin); cel.shade(T('M 40 78 C 90 74 140 62 180 58 C 220 56 256 60 284 64 L 270 100 L 40 120 Z'), sh, thumb); cel.shade(T('M 236 32 C 262 36 284 50 284 62 C 276 56 254 48 236 46 Z'), lit, thumb); outline(cel, thumb, lw, ink, 17)
    for d in ('M 196 -134 C 206 -112 210 -96 206 -82', 'M 252 -128 C 262 -108 262 -94 256 -80', 'M 132 -120 C 144 -100 148 -88 144 -76', 'M 292 -20 C 300 -10 306 0 306 12', 'M 224 38 C 236 22 246 8 250 -8', 'M 20 -70 C 60 -60 100 -62 130 -70', 'M 10 -40 C 50 -30 90 -32 120 -40', 'M 150 40 C 160 50 162 60 158 70', 'M 262 46 C 272 48 280 54 282 60'): cel.line(T(d), lw * .55, ink, taper=(.15, .6))
    return hand


def curse(cel, lw=4.):
    """dark mass of the curse caving in round the blow (contact ~ (1085, 545)): only shoulder, ribs and folds, no face"""
    ink = '#07050a'; a, b, c, hi = '#4a2433', '#2a1220', '#14080f', '#c8485a'
    back = s1('M 1300 -40 C 1230 60 1170 200 1150 320 C 1136 400 1112 452 1090 486 C 1074 510 1068 530 1072 552 C 1078 580 1100 600 1112 640 C 1130 700 1120 800 1150 880 C 1176 950 1220 1040 1290 1130 L 1990 1130 L 1990 -40 Z'); cel.fill(back, a)
    cel.shade(s1('M 1560 -60 C 1480 160 1440 320 1380 440 C 1340 520 1360 600 1400 700 C 1450 820 1500 960 1560 1140 L 2000 1140 L 2000 -60 Z'), b, back); cel.shade(s1('M 1800 -60 C 1720 180 1640 360 1640 540 C 1650 720 1720 920 1800 1140 L 2000 1140 L 2000 -60 Z'), c, back)
    for d in ('M 1084 520 C 1150 450 1230 330 1262 150 C 1282 330 1220 480 1096 548 Z', 'M 1088 532 C 1200 470 1330 390 1460 220 C 1390 420 1260 540 1092 556 Z', 'M 1086 552 C 1240 530 1420 500 1640 420 C 1460 560 1260 600 1094 566 Z', 'M 1092 570 C 1240 600 1420 640 1660 640 C 1440 710 1240 680 1104 596 Z', 'M 1102 590 C 1220 670 1340 780 1470 940 C 1310 860 1180 760 1112 640 Z', 'M 1112 620 C 1160 720 1200 830 1230 980 C 1160 880 1120 780 1118 690 Z'): cel.shade(s1(d), c, back)
    for d in ('M 1090 506 C 1140 440 1190 350 1220 220 C 1216 340 1170 440 1104 500 Z', 'M 1150 500 C 1250 440 1330 370 1400 280 C 1350 390 1260 470 1160 520 Z', 'M 1160 556 C 1300 540 1420 510 1560 450 C 1430 530 1300 560 1160 566 Z', 'M 1140 620 C 1220 680 1290 740 1350 820 C 1270 770 1190 700 1140 640 Z', 'M 1180 590 C 1300 620 1420 640 1540 644 C 1410 660 1290 640 1180 606 Z'): cel.shade(s1(d), hi, back)
    outline(cel, back, lw * 1.3, ink, 21)
    r = R(12)
    for i in range(34):
        an = r.uniform(-1.25, 1.25); q0 = r.uniform(60, 260); q1 = q0 + r.uniform(120, 620); bow = r.normal(0, .10); v = np.linspace(0, 1, 14); aa = an + bow * v; rr = q0 + (q1 - q0) * v; cel.line(np.c_[1085 + np.cos(aa) * rr, 548 + np.sin(aa) * rr], lw * r.uniform(.3, .75), ink, taper=(.1, .7), seed=i, press=.5)


def hero_flash(f=6, seed=11):
    x, y = grid(); cx, cy = 1088, 548
    d = np.hypot((x - cx) / 1.3, (y - cy) / .9); g = ramp(np.clip(d / 1250, 0, 1), [(0, '#fff2e4'), (.07, '#ffd0b8'), (.13, '#ff5a50'), (.24, '#d8142c'), (.42, '#86081e'), (.66, '#3a0614'), (1, '#100309')]); n = fbm(H, W, (160, 60, 24), None, 4); g = np.clip(g * (1 + .10 * n[..., None]), 0, 1)
    g = repaint(g, sizes=(34, 16, 8), seed=3, angle=np.arctan2(y - cy, x - cx), density=.8, aspect=8, jit=(.01, .04, .05))
    cel = Cel(); radial_wedges(cel, cx, cy, 90, 430, 2400, 5, .030, '#0a0309'); radial_wedges(cel, cx, cy, 30, 520, 2400, 6, .007, '#ff5a58'); radial_wedges(cel, cx, cy, 12, 560, 2400, 8, .0045, '#fff4ec'); im = put(g, cel.layer())
    cel = Cel(); curse(cel, 4.2); L = cel.layer(); L = para(L, (cx, cy), (1900, 300), [(0, '#ffe0d8'), (.22, '#f09aa0'), (.5, '#8a5068'), (1, '#3a2038')], 'mul'); L = rim(L, 10, 2, '#ff2a3a', 1., .5, where=(y > 16).astype(f32)); L = rim(L, 4, 1, '#fff0e4', .95, .4, where=(y > 16).astype(f32)); im = put(im, L)
    c = Cel(); dust = []; puffs(c, billow(.5, 1120, 380, 120, 3, 8), dict(lit='#ffb09a', mid='#b02a38', dark='#4a0a1a', ink='#12040a'), (-.6, .8), 2.4, 3); puffs(c, billow(.45, 1140, 720, 130, 5, 8), dict(lit='#ffb09a', mid='#b02a38', dark='#4a0a1a', ink='#12040a'), (-.6, -.8), 2.4, 4); im = put(im, c.layer())
    im = distort_ring(im, cx, cy, 520, 120, 40, 3, ca=4)
    c = Cel(); ragged_ring(c, cx, cy, 640, 590, 13, 5, '#ffd8c8', gaps=.5, n=11, rag=.018, spurs=.8); ragged_ring(c, cx, cy, 662, 612, 5, 6, RED, gaps=.5, n=11, rag=.018); im = put(im, c.layer(), op=.9)
    im = put(im, black_flash_layers(f, cx, cy, 1.05, seed, 0., 17, 980, .8, part='back', core=False))
    S = 1.22; tx, ty, ro = 690, 566, -.035; hs = _xf(s1('M -20 -80 C 120 -128 270 -140 320 -80 C 336 -20 330 60 250 130 C 120 150 -10 110 -300 160 L -900 280 L -900 -230 L -300 -120 Z'), S, ro, tx, ty)
    back, fr = cursed(hs, f, (-2.6, -.4), 150, dict(CURSE, out='#0a0836', body='#1b38f4', hot='#5cc4ff'), 3, 1.1, .9, sheath=.15); im = put(im, back)
    cel = Cel(); fist(cel, S, ro, tx, ty, 4.8); L = cel.layer(); L = para(L, (cx - 20, 520), (120, 760), [(0, '#fff6ee'), (.22, '#f6c4c0'), (.5, '#a86a90'), (.78, '#5a3f72'), (1, '#2c2348')], 'mul'); L = rim(L, -7, 4, '#ff3a48', .95, .5, where=smooth(500, 1000, x)); L = rim(L, -3, 1, '#fff4ea', .95, .4, where=smooth(760, 1020, x)); L = rim(L, 0, 5, '#4a7dff', .8, .5, where=1 - smooth(300, 760, x)); im = put(im, L); im = put(im, fr, light=.35)
    c = Cel(); kk = kite_star(cx + 6, cy, 34, 330, 13, 5, (math.pi / 2, .5), .050, 5)
    for K, K2 in kk: c.fill(K2, RED)
    for K, K2 in kk: c.fill(K, BONE)
    c.fill(star_poly(cx + 6, cy, 30, 78, 10, 6), BONE); im = put(im, c.layer())
    c = Cel(); debris(c, 8, (cx, cy), 40, 5, (500, 2300), (-math.pi, math.pi), (4, 44), 500., pal=dict(lit='#ff8a78', mid='#8a1a2c', dark='#22060f', ink='#0c0408'), light=(0, 0, 1), cubes=.2, streak='#ff6a60'); im = put(im, c.layer())
    im = put(im, black_flash_layers(f, cx, cy, 1.05, seed, 0., 17, 980, .8, part='front')); return im + (np.exp(-(np.hypot(x - cx, y - cy) / 300.) ** 2) * .35)[..., None] * C('#ff5a50')


def hero_column(f=6, seed=3):
    """low angle over a dead street: the column, walls of lit smoke, dark towers with burning edges"""
    x, y = grid(); cx, yb = 1130, 850
    d = np.hypot((x - cx) / 1.0, (y - 520) * .62); n = fbm(H, W, (320, 130, 50), None, 8); g = ramp(np.clip(d / 1050 + .07 * n, 0, 1), [(0, '#ff9a34'), (.14, '#ea5c14'), (.28, '#b0300c'), (.46, '#6a150b'), (.7, '#30090a'), (1, '#120406')])
    im = repaint(g, sizes=(40, 18, 9), seed=5, angle=-np.pi / 2 + (x - cx) / 900. + .5 * n, density=.9, aspect=5, jit=(.01, .03, .04))
    far = dict(lit='#e2661c', mid='#7a1c0c', dark='#320a0a', ink='#1a0506'); near = dict(lit='#ffa03a', mid='#8e220c', dark='#2a0709', ink='#120304')
    cel = Cel()
    for i in range(16):                                      # far wall of smoke
        sg = 1 if i % 2 else -1; q = (i // 2 + h1(seed, i)) / 8.; puffs(cel, billow(.7, cx + sg * (250 + 820 * q), 560 - 520 * h1(seed, i, 2) + 80 * q, 210 + 130 * h1(seed, i, 3), 40 + i, 9, .2, 1., 4.5, 1.), far, (-sg * .9, .42), 1.8, i, tone=.80, curls=False)
    im = put(im, cel.layer(), op=.92)
    cel = Cel(); r = R(seed, 5)
    for lay, (c0, c1) in enumerate((('#7c220c', '#ffb04a'), ('#48110a', '#ff8a2a'), ('#1f0708', '#e2601a'))):
        u = -60
        while u < W + 60:
            w = r.uniform(46, 130) * (1 + .5 * lay); k = min(1., abs(u + w / 2 - cx) / 800.); h = (r.uniform(50, 150) + 330 * k ** 1.6) * (.55 + .28 * lay)
            if lay == 2 and k < .42: u += w; continue
            tp = yb - 10 - h; bite = r.random(); top = [[u, yb + 30], [u, tp + 10 * r.random()], [u + w * .28, tp], [u + w * .30, tp - 16 * r.random()], [u + w * (.5 + .2 * bite), tp - 12], [u + w * (.56 + .2 * bite), tp + 30 * bite], [u + w * .86, tp + 18], [u + w, tp + 30 * r.random()], [u + w, yb + 30]]; cel.fill(np.array(top), c0)
            e, s = (u + w, -1) if u + w / 2 < cx else (u, 1); fw = w * (.12 + .16 * r.random()); cel.fill(np.array([[e, yb + 30], [e, tp + 24], [e + s * fw, tp + 18], [e + s * fw, yb + 30]]), c1, .95 - .25 * lay)
            for j in range(int(h / (16 + 8 * lay))):
                for i in range(int((w - fw) / (14 + 8 * lay))):
                    if h1(int(u), i, j, lay) < .5: continue
                    wx = (u + 6 + i * (14 + 8 * lay)) if s < 0 else (u + fw + 6 + i * (14 + 8 * lay)); wy = tp + 34 + j * (16 + 8 * lay); cel.fill(np.array([[wx, wy], [wx + 6 + 4 * lay, wy], [wx + 6 + 4 * lay, wy + 6 + 4 * lay], [wx, wy + 6 + 4 * lay]]), '#ffc45a' if h1(int(u), i, j, 9) > .93 else '#0e0304', .85)
            u += w + r.uniform(-10, 34)
    im = put(im, cel.layer()); air = np.exp(-((y - yb) / 190.) ** 2) * np.exp(-((x - cx) / 620.) ** 2); im = im + (air * .6)[..., None] * C('#ff7a24') * .7
    sm, fi = fire_column(f, cx, yb + 10, -190, 150, seed, lean=.05, sweeps=14, tongues=1.4, hsmoke=near); im = put(im, sm)
    cel = Cel(); gnd = np.array([[0, yb + 8], [W, yb + 8], [W, H], [0, H]]); cel.grad(gnd, (cx, yb), (cx, H + 30), [(0, '#ffb04a', 1), (.14, '#d8541a', 1), (.42, '#651609', 1), (1, '#150405', 1)]); im = put(im, cel.layer())
    side = np.clip(np.abs(x - cx) / 1000., 0, 1) * smooth(yb + 4, yb + 12, y); im = im * (1 - .72 * side[..., None] ** 1.2)
    cel = Cel(); gp = ground_proj(hy=yb - 60, cx=cx, cy=yb + 60)
    for k in range(7):                                       # street marks in perspective
        for sg in (-1, 1): P = gp(np.array([[sg * 2.6, -2.6 + k * 1.2], [sg * 2.6 + .22, -2.6 + k * 1.2], [sg * 2.6 + .22, -2.0 + k * 1.2], [sg * 2.6, -2.0 + k * 1.2]])); cel.fill(P, '#ffb45a', .35)
    ground_crack(cel, gp, 4.4, 1., 4, 15, col='#150304', lip='#ffb85a', side='#240607', top='#8a3014', topdark='#43100a', lift=.22, glow='#ffe06a', w=.055, lw=2.2); im = put(im, cel.layer()); im = put(im, fi)
    back = dict(lit='#c24a16', mid='#4a110a', dark='#160405', ink='#090203'); cel = Cel(); debris(cel, 11, (cx, yb), 150, 8, (500, 2500), (-2.98, -.16), (2, 20), 700., pal=back, light=(0, .6, .25), spread=(230, 30), streak='#ffb04a'); im = put(im, cel.layer())
    fg_ = Cel()                                              # foreground: broken road edge in silhouette, rim-lit
    rub = np.array([[-20, 1100], [-20, 948]] + [[q, 948 + 112 * math.sin(math.pi * min(1., q / 900.) * .5) ** 1.2 + 26 * noise(q / 70., 3.) + 12 * noise(q / 22., 5.)] for q in range(0, 840, 26)] + [[860, 1100], [1380, 1100]] + [[q, 1066 - 130 * ((q - 1400) / 540.) ** 1.3 + 24 * noise(q / 60., 7.) + 10 * noise(q / 20., 9.)] for q in range(1400, 1945, 26)] + [[1945, 1100]]); fg_.fill(rub, '#0a0304')
    for i in range(130):
        k = int(h1(i, 1) * (len(rub) - 6)) + 2; s = 6 + 34 * h1(i, 3) ** 2.4; rock(fg_, rub[k, 0] + 40 * (h1(i, 2) - .5), min(1076, rub[k, 1] + 8 + 46 * h1(i, 5) ** 2), s, (i, i * 2.1, i * .7), i, dict(lit='#ee7a26', mid='#34090a', dark='#0f0304', ink='#050102'), (.55 if rub[k, 0] < cx else -.55, -.95, -.30), 'slab' if i % 6 == 0 else 'rock', lw=1.6)
    im = put(im, fg_.layer()); im = put(im, embers(f, 190, 4, near=.10, size=1.1)); return comp.cam2d(im, 960, 545, 1.06, -.035)
