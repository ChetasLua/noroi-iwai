"""Blue-hour shopping street with a warm bakery window."""
import numpy as np, cv2, math
from .paint import rgb, f32, fbm, smooth, mix, grid, Canvas, arcs, jit_color, lum
from .cam import Camera3D, rot_x, rot_y, rot_z
from .comp import blur, radial, over
from . import world as Wd, world_sky as SK
from .world import C, Scene, Wall, Plate, painter, FAR
from .world_props import *

CRUST = ['#c8843a', '#b86a2a', '#d89a4c', '#a85a24', '#e0ac60', '#8f4a20']


def loaf(S, c, r, kind=None, k=1., grp=1):
    """one bun / tin loaf / croissant / melon bread: a faceted dome with a lit crown"""
    kind = kind or pick(r, ['bun', 'tin', 'bun', 'cross', 'round']); col = C(pick(r, CRUST)) * r.uniform(.85, 1.12); c = np.asarray(c, float); a0 = r.uniform(0, 6.28)
    if kind == 'tin':
        S.boxes(c + [0, 0, .05 * k], [.20 * k, .11 * k, .10 * k], a0, col * .9, top=col * 1.25, grp=grp, snow=0); S.cyl(c[:2], .07 * k, c[2] + .09 * k, c[2] + .13 * k, col * 1.15, 8, .03 * k, grp=grp, snow=0)
    elif kind == 'cross':
        t = np.linspace(-1, 1, 7); p = c + np.c_[np.cos(a0) * t * .10 - np.sin(a0) * (t * t) * .05, np.sin(a0) * t * .10 + np.cos(a0) * (t * t) * .05, np.full(7, .035)] * [k, k, k]; S.tube(p, np.array([.008, .028, .042, .05, .042, .028, .008]) * k, np.stack([col * (1.2 if i % 2 else .85) for i in range(7)]), 7, grp=grp, snow=0)
    else:
        R = (.075 if kind == 'bun' else .10) * k; S.cyl(c[:2], R, c[2], c[2] + R * .45, col * .85, 9, R * .96, cap=False, grp=grp, snow=0); S.cyl(c[:2], R * .96, c[2] + R * .45, c[2] + R * .85, col * 1.1, 9, R * .5, grp=grp, snow=0)


@painter('bakery', 3)
def bakery(w=2400, h=1350, seed=0, eye=(-3.6, -4.6, 1.52), target=(2.0, 3.2, 1.92), focal=1380., paint=True, ss=2, warm=1., figures=True):
    """blue-hour shopping street, bakery on the right, street falling away to the left.  pavement z=.15 for x>1.6 (stand people at x 1.7..2.7), road z=0.
    shop front plane x=3.0 (y -2.4..7.2).  masks: glass (the pane: put reflections there), sky, ground, warm (window light falling on the pavement).
    extra: glass_quad (4 plate-px corners, pane bottom-left near, bottom-right far...), layers: none"""
    cam = Camera3D(eye, target, focal, w, h); S = Scene(seed + 3); r = np.random.default_rng(seed + 37); S.groups = 4; X = 3.0; Y0, Y1 = -2.4, 7.2; f = cam.f
    S.ground(-3.0, 1.6, -9, 160, 0., '#5c5e6a', nx=6, ny=50, pw=1.8, tex=2, seed=9); strip(S, [[-2.6, -9], [-2.6, 120]], .12); strip(S, [[1.2, -9], [1.2, 120]], .12)
    for sd in (1, -1):
        P = np.array([[1.6, -9], [X + .1, -9], [X + .1, 160], [1.6, 160]]) if sd > 0 else np.array([[-5., -9], [-3., -9], [-3., 160], [-5., 160]]); S.prism(P, 0, .15, '#9a9898', top='#8c8a8c', tex=3)
    # ---- bakery
    W = Wall.frame(S, (X, Y1, .15), (0, -1, 0), (0, 0, 1), Y1 - Y0, 3.6, 1); wood = C('#3f5a4c'); cream = C('#e2d2b4')
    gu0, gu1 = 1.0, 6.4; gv0, gv1 = .72, 2.55                              # the big pane (u from the far end)
    W.slab(0, gu0, 0, 3.4, -.3, 0, wood, tex=12); W.slab(gu1, gu1 + .35, 0, 3.4, -.3, 0, wood, tex=12); W.slab(gu0, gu1, 0, gv0, -.3, 0, wood * .95, tex=12); W.slab(gu0, gu1, gv1, 3.4, -.3, 0, wood, tex=12); W.slab(8.2, W.w, 0, 3.4, -.3, 0, wood, tex=12); W.slab(gu1 + .35, 8.2, 2.5, 3.4, -.3, 0, wood, tex=12)
    for u0, u1 in ((gu0 + .25, gu0 + 1.6), (gu0 + 1.9, gu0 + 3.3), (gu0 + 3.6, gu1 - .2)): W.slab(u0, u1, .16, gv0 - .14, 0, .03, wood * 1.18); W.slab(u0 + .09, u1 - .09, .24, gv0 - .22, .02, .035, wood * .9)
    W.slab(gu0 - .05, gu1 + .05, gv0 - .07, gv0, 0, .16, cream * .8); ms = gu0 + (gu1 - gu0) * np.array([0, .34, .67, 1.]); W.slab(ms - .045, ms + .045, gv0, gv1, -.12, .05, wood * 1.1); W.slab(gu0, gu1, gv1 - .52, gv1 - .46, -.1, .04, wood * 1.1); W.slab(gu0, gu1, gv1, gv1 + .10, 0, .10, cream * .85)
    W.slab(0, W.w, 3.4, 3.62, 0, .34, wood * 1.25); W.slab(0, W.w, 2.78, 3.4, 0, .10, '#2c4238'); sg = Wall.frame(S, W.p(1.1, 2.83, .12), W.ex, W.ez, 5.0, .52, 1)
    # sign: loaf, wheat ears and a ribbon, no letters
    sg.poly([(2.0, .10), (3.0, .10), (3.08, .30), (2.86, .44), (2.5, .48), (2.14, .44), (1.92, .30)], 0, '#f0c878', em=.35)
    for q in (2.2, 2.45, 2.7): sg.poly([(q, .16), (q + .05, .15), (q + .18, .40), (q + .12, .42)], .01, '#b8743a', em=.3)
    for s_ in (-1, 1):
        for k in range(6): x0 = 2.5 + s_ * (.95 + k * .17); sg.poly([(x0, .26), (x0 + s_ * .13, .37), (x0 + s_ * .20, .28), (x0 + s_ * .12, .18)], 0, '#e8c070', em=.3)
        sg.rect(2.5 + s_ * 2.2 - .25, 2.5 + s_ * 2.2 + .25, .24, .28, 0, '#e8c070', em=.3)
    for u in np.linspace(1.4, 6.0, 4): S.tube([W.p(u, 3.62, .2), W.p(u, 3.86, .32), W.p(u, 3.72, .62)], .022, '#23262a', 6); S.cyl(W.p(u, 3.7, .62)[:2], .13, W.p(u, 3.62, .62)[2] - .02, W.p(u, 3.74, .62)[2], '#23302a', 10, .05); S.light(W.p(u, 3.5, .62), '#ffd9a0', 3.6, 1.0, 1.6, 1)
    # awning: striped canvas with a scalloped valance
    n = 22; us = np.linspace(.5, 6.9, n + 1); o = 1.15
    for i in range(n):
        c = C('#b8402e') if i % 2 else C('#efe2c8'); a, b = us[i], us[i + 1]; S.quads(np.stack([W.p(a, 2.74, .06), W.p(b, 2.74, .06), W.p(b, 2.42, o), W.p(a, 2.42, o)])[None], np.stack([c * .9, c * .9, c * 1.02, c * 1.02])[None], grp=1, d=W.n)
        t = np.linspace(0, 1, 5); e = .12 + .09 * np.sin(np.pi * t); S.quads(np.stack([W.p(a + (b - a) * t[:-1], 2.42, o), W.p(a + (b - a) * t[1:], 2.42, o), W.p(a + (b - a) * t[1:], 2.42 - e[1:], o + .01), W.p(a + (b - a) * t[:-1], 2.42 - e[:-1], o + .01)], 1), c, grp=1)
    # door: half glass, brass bar, bell, hours plate as little icons
    W.slab(gu1 + .35, gu1 + .45, 0, 2.5, -.2, .02, wood * 1.1); W.slab(8.1, 8.2, 0, 2.5, -.2, .02, wood * 1.1); D = (gu1 + .47, 8.08); W.slab(D[0], D[1], .02, .55, -.14, -.06, wood * 1.05); W.slab(np.array([D[0], D[1] - .16]), np.array([D[0] + .16, D[1]]), .02, 2.45, -.14, -.06, wood * 1.05); W.slab(D[0], D[1], 2.28, 2.45, -.14, -.06, wood * 1.05); S.tube([W.p(D[0] + .3, .95, .0), W.p(D[0] + .3, 1.5, .0)], .022, '#d8a850', 7); W.disc(D[1] - .34, 1.9, .13, -.02, '#e8dcc0', em=.4)
    # ---- the room behind the glass (real depth)
    R0 = X + .3; dp = 3.4; S.quads(np.array([[R0, Y0 + .2, .18], [R0 + dp, Y0 + .2, .18], [R0 + dp, Y1 - .2, .18], [R0, Y1 - .2, .18]])[None], '#a8764a', tex=12, d=(1, 0, 0), grp=1); S.quads(np.array([[R0 + dp, Y0, .18], [R0 + dp, Y1, .18], [R0 + dp, Y1, 3.1], [R0 + dp, Y0, 3.1]])[None], np.array([[C('#c89a6a'), C('#c89a6a'), C('#e6c08a'), C('#e6c08a')]]), tex=14, em=.12, grp=1)
    for y in (Y0 + .2, Y1 - .25): S.quads(np.array([[R0, y, .18], [R0 + dp, y, .18], [R0 + dp, y, 3.1], [R0, y, 3.1]])[None], '#cfa878', tex=14, em=.12, grp=1)
    S.quads(np.array([[R0, Y0, 3.0], [R0 + dp, Y0, 3.0], [R0 + dp, Y1, 3.0], [R0, Y1, 3.0]])[None], '#e2c49a', em=.16, grp=1)
    for k, zz in enumerate((.95, 1.42, 1.89, 2.36)):                       # back shelves full of bread
        S.boxes([R0 + dp - .22, (Y0 + Y1) / 2, zz], [.42, Y1 - Y0 - .8, .04], 0, '#7a4e2c', tex=12, grp=1); y = Y0 + .7
        while y < Y1 - .7:
            if r.random() < .85:
                for j in range(int(r.integers(1, 3))): loaf(S, (R0 + dp - .32 + j * .14, y + r.normal(0, .02), zz + .03), r, k=r.uniform(1., 1.5))
            y += r.uniform(.15, .30)
    for y in (-.6, 1.6, 4.2): S.boxes([R0 + dp - .22, y, 1.6], [.44, .05, 1.8], 0, '#6a4426', grp=1)
    S.boxes([R0 + 1.75, 2.2, .68], [.65, 5.4, 1.0], 0, '#8a5a34', top='#c89462', tex=12, grp=1); S.boxes([R0 + 1.75, 2.2, 1.4], [.5, 4.6, .02], 0, '#e8d8b8', grp=1, em=.2)
    for y in np.arange(-.2, 4.6, .2): loaf(S, (R0 + 1.75 + r.normal(0, .1), y, 1.2), r, k=1.2); loaf(S, (R0 + 1.75 + r.normal(0, .1), y, 1.42), r, k=1.1)
    for k in range(2):                                                      # window display: two steps, cloth, baskets, baguettes
        S.boxes([R0 + .2 + k * .34, (gu0 + gu1) / 2 * -1 + Y1, .74 + k * .20], [.34, gu1 - gu0 - .2, .06], 0, '#efe2cc', grp=1, em=.1); yy = np.arange(Y1 - gu1 + .2, Y1 - gu0 - .1, .205)
        for y in yy:
            if r.random() < .88: loaf(S, (R0 + .2 + k * .34 + r.normal(0, .05), y + r.normal(0, .03), .78 + k * .2), r, k=r.uniform(1.25, 1.8))
    for yb in (1.3, 4.9):
        S.cyl((R0 + .85, yb), .22, .9, 1.22, '#a8743e', 12, .27, cap=False, grp=1)
        for k in range(8): a = r.uniform(0, 6.28); S.tube([[R0 + .85 + math.cos(a) * .08, yb + math.sin(a) * .08, 1.0], [R0 + .85 + math.cos(a) * .25, yb + math.sin(a) * .25, 1.72 + r.uniform(-.1, .1)]], [.03, .035], C(pick(r, CRUST)) * 1.1, 7, grp=1)
    for yb in np.linspace(-.8, 5.8, 5): S.seg([[R0 + .9, yb, 3.0]], [[R0 + .9, yb, 2.32]], .015, '#2a2018'); S.cyl((R0 + .9, yb), .16, 2.14, 2.34, '#e8b060', 12, .05, em=.8, grp=1); S.boxes([R0 + .9, yb, 2.13], [.14, .14, .08], 0, '#fff2cc', em=1.9 * warm, grp=1); S.light((R0 + .9, yb, 1.95), '#ffc070', 7.5, 2.0, 1.55 * warm, 1); S.light((R0 - .05, yb, 1.75), '#ffb866', 7.5, 1.8, .95 * warm, 1, dir=(-1, 0, -.42), cos=.1)
    if figures:                                                              # the baker behind the counter, a customer
        for (yy, xx, hh, c, ap) in ((3.6, R0 + 2.4, 1.66, '#f0e6d6', 1),):
            F = Wall.frame(S, (xx, yy + .3, .18), (.35, -.94, 0), (0, 0, 1), .6, hh, 1); F.poly([(.06, 0), (.54, 0), (.56, hh * .5), (.60, hh * .78), (.42, hh * .85), (.18, hh * .85), (0, hh * .78), (.04, hh * .5)], 0, c); F.disc(.3, hh * .92, hh * .062, .01, '#d8a880', 14, hh * .075); F.poly([(.17, hh * .93), (.43, hh * .93), (.40, hh * 1.0), (.3, hh * 1.03), (.2, hh * 1.0)], .02, '#e8dcc8' if ap else '#2c1c1c'); 
            if ap: F.rect(.14, .46, .1, hh * .62, .015, '#b85a3a')
    # ---- pavement dressing
    for (yb, xb) in ((-1.9, 2.6), (5.9, 2.65), (6.6, 2.6)):
        S.prism(np.array([(-.22, -.22), (.22, -.22), (.22, .22), (-.22, .22)]) * .95 + [xb, yb], .15, .55, '#9a5a3c', top='#3a2c24')
        for k in range(1): 
            from .world_hill import bush; bush(S, (xb, yb, .72), .36, r, n=110, flat=1.15, fine=1.6)
    bp = np.array([1.95, 4.9, .15]); 
    for s_ in (-1, 1): S.quads(np.array([bp + [s_ * .22, -.28, 0], bp + [s_ * .22, .28, 0], bp + [s_ * .04, .28, .9], bp + [s_ * .04, -.28, .9]])[None], '#2c2a2c'); 
    Wb = Wall.frame(S, bp + [-.24, .26, .05], (0, -1, 0), (.21, 0, .9) / np.linalg.norm([.21, 0, .9]), .52, .84); Wb.slab(0, .52, 0, .84, -.02, 0, '#7a5634'); Wb.rect(.04, .48, .04, .80, .006, '#26302c'); graphic(Wb, .08, .44, .40, .76, .01, r, ('#26302c', '#f0e6cc', '#e8b060'), 'cup', em=.15, bg=False); Wb.rect(np.array([.10, .10, .10]), np.array([.40, .34, .42]), np.array([.30, .22, .14]), np.array([.325, .245, .165]), .012, '#e8e0cc', em=.1)
    # a bicycle against the wall
    bc = np.array([2.62, 8.3, .15])
    for dy in (-.5, .5): a = np.linspace(0, 2 * np.pi, 22); S.tube(np.c_[np.full(22, bc[0]), bc[1] + dy + np.cos(a) * .32, bc[2] + .33 + np.sin(a) * .32], .018, '#1c1c20', 5)
    P_ = lambda y, z: bc + [0, y, z]; S.seg([P_(-.5, .33), P_(-.12, .36), P_(-.12, .36), P_(.35, .78), P_(-.2, .74), P_(.5, .33)], [P_(-.2, .74), P_(.35, .78), P_(-.2, .74), P_(-.12, .36), P_(.35, .78), P_(.38, .86)], .035, '#a83a34'); S.seg([P_(.36, .95)], [P_(.36, .95) + [-.25, 0, .02]], .03, '#2a2a2e'); S.boxes(P_(-.24, .84), [.16, .26, .07], 0, '#2a2420'); S.boxes(P_(.60, .72), [.30, .30, .24], 0, '#8a6a44')
    # ---- neighbours and the far street
    def side(sd, y, spec):
        for (wd, hh, st, pal, shop, bl) in spec:
            x = sd * 1.0; xf = X if sd > 0 else -5.0; a, b = (np.array([xf, y + wd]), np.array([xf, y])) if sd > 0 else (np.array([xf, y]), np.array([xf, y + wd])); foot = lot(a, b, 9.)
            building(S, foot, hh, st, np.random.default_rng(int(abs(y * 31 + wd * 7) + seed)), cam.eye, f, .42, pal, shop=shop, blades=bl, grp=2, glow=.72); S.light((xf - sd * 1.2, y + wd / 2, 2.7), pick(r, ['#ffcf94', '#ffe0b8', '#ffb684']), 9, 2.4, .9 if shop else .2, 2); y += wd + .12
    up_ = np.array([[X, Y1, 3.77], [X, Y0, 3.77]]); Wu = Wall.frame(S, (X, Y1, 3.77), (0, -1, 0), (0, 0, 1), Y1 - Y0, 5.4, 2); facade(Wu, 'brick', np.random.default_rng(5), .4, '#8a7a72', 2); S.quads(np.array([[X, Y1, 3.77], [X + 8, Y1, 3.77], [X + 8, Y1, 9.2], [X, Y1, 9.2]])[None], '#6a605e', tex=1)
    side(1, 9.4, [(6.2, 7.4, 'grid', '#a89a88', 3.4, 1), (5.4, 9.6, 'tenant', '#7a7274', 3.4, 2), (7.2, 6.8, 'resid', '#b8a892', 3.4, 1), (6., 11, 'grid', '#8a8690', 3.4, 1), (8, 8.2, 'tenant', '#948a80', 3.4, 2), (9, 10, 'resid', '#a8a098', 3.4, 1), (12, 13, 'grid', '#7c7a86', 3.4, 1), (20, 11, 'office', '#8a8a94', 3.4, 1), (40, 12, 'grid', '#7a7a88', 0, 0)])
    side(-1, 4.5, [(5.8, 7.2, 'resid', '#9c9288', 3.4, 1), (6.4, 9.2, 'grid', '#8a7c74', 3.4, 2), (5.2, 6.6, 'brick', '#98705e', 3.4, 1), (7.5, 11.5, 'tenant', '#7c7a84', 3.4, 2), (6.5, 8, 'grid', '#a09686', 3.4, 1), (9, 10.5, 'resid', '#8e8a90', 3.4, 1), (12, 9, 'tenant', '#8a8078', 3.4, 2), (18, 13, 'office', '#7e8290', 3.4, 0), (40, 12, 'grid', '#7a7a88', 0, 0)])
    side(1, -16.6, [(6., 8.2, 'brick', '#8a6a5c', 3.4, 1), (7.9, 9.4, 'grid', '#8c8a92', 3.4, 1)])
    an = []
    for (x, y) in ((1.75, 10.5), (-3.2, 17.), (1.75, 25.), (-3.2, 36), (1.75, 48)): an.append(pole(S, (x, y), 9.2, r, .15, ((1, .05),), '#6e6c72'))
    for i in range(len(an) - 1):
        for j in (0, 1, 2, 3, 5, 6): S.wire(an[i][j], an[i + 1][j], r.uniform(.3, .7), .022, '#141622', min_px=1.1)
    for j in (0, 1, 2, 5, 6): S.wire(np.array([2.6, -8., 9.4 + .1 * j]), an[0][j], .5, .022, '#141622', min_px=1.1)
    for (x, y, sd) in ((1.85, 8.6, -1), (-3.2, 12.5, 1), (1.85, 19, -1), (-3.2, 27, 1), (1.85, 38, -1)): 
        S.tube([[x, y, .15], [x, y, 3.5]], [.07, .05], '#2c3438', 8); S.boxes([x, y, 3.72], [.34, .34, .42], 0, '#ffe4b4', em=1.35, grp=2); S.boxes([x, y, 3.97], [.46, .46, .08], 0, '#2c3438'); S.light((x, y, 3.6), '#ffd49a', 12, 3., 1.7, 2)
    if figures: 
        m = 16; y = r.uniform(12, 60, m); sd = r.choice([-1, 1], m); crowd(S, np.c_[np.where(sd > 0, r.uniform(1.8, 2.7, m), r.uniform(-4.6, -3.2, m)), y], cam, r, .15, grp=0)
    hz = Wd.horizon_y(cam); sky = SK.bluehour(h, w, hz + 20, seed + 2, .12)
    nk = dict(amb=('#39519a', '#20223a'), street=('#4a3424', 4.), fills=[((.9, .3, .1), '#0a1024'), ((-.9, -.2, .2), '#101c40')], fog=dict(col='#4a5a98', col2='#c0888e', dir=(-.35, 1, .02), dens=.0105, start=14, hf=.02, floor=.6, keep=.4, pw=14.))
    baked = S._bake(cam); res = S.render(cam, ss=ss, baked=baked, **nk); Xp = cam.eye + cam.rays() * res['z'][..., None]; geo = (res['z'] < FAR * .5).astype(f32); g = ((Xp[..., 2] < .17) & (geo > 0) & (Xp[..., 0] < X)).astype(f32)
    img = res['img'] * (1 + (Wd.occlusion(res['z'], .38, (3, 10, 28)) - 1) * geo)[..., None]; r2 = dict(res); r2['img'] = img; pud = smooth(-.4, .9, fbm(h, w, (160, 60, 22), (1, .6, .3), seed + 41, aspect=3.)); img, _ = Wd.wet_reflect(S, cam, r2, sky, cv2.GaussianBlur(g, (0, 0), 1.) * (.10 + .26 * pud), baked=baked, strength=.8, vblur=18, ripple=2., seed=seed + 3, **nk)
    img = img + sky * (1 - res['a'][..., None])
    q, _ = cam.project(np.array([W.p(gu0, gv0, 0), W.p(gu1, gv0, 0), W.p(gu1, gv1, 0), W.p(gu0, gv1, 0)])); gm = np.zeros((h, w), f32); cv2.fillPoly(gm, [np.round(q * 8).astype(np.int32)], 1., cv2.LINE_AA, 3); gm = gm * ((Xp[..., 0] > X + .02) | (res['z'] > FAR * .5)); gm = cv2.GaussianBlur(gm.astype(f32), (0, 0), .8)
    if paint:
        img = Wd.wobble(img, .5, 32, seed + 12); img = Wd.gouache(img, res['lab'], res['ang'] + fbm(h, w, (140, 50), (1, .5), seed + 8) * .10, seed=seed + 4, sizes=(16, 8, 4), op=(.55, .5, .46), val=(.044, .038, .032), hue=.016, thr=(.014, .026, .05), mask=geo)
    # old glass: two pale diagonal reflections of the sky
    x, y = grid(h, w); u = (x * .80 + y * .6) / w; st = smooth(.0, .03, np.sin(u * 34 + .6) - .72 + .05 * fbm(h, w, (300, 90), (1, .4), seed + 50)) * .5 + smooth(.0, .05, np.sin(u * 21 + 2.) - .82) * .5
    img = img + (C('#8fa8e0') - img) * (gm * st * .16)[..., None]
    if paint: e = np.maximum(img - .62, 0); img = Wd.lamp_glow(img, e, ((6, .24), (26, .20))); img = Wd.air_glow(img, res['z'], .5, (60, 160), .34, (4, 60), '#ffcf98'); img = Wd.tooth_pass(img, seed + 6)
    wm = np.clip((img[..., 0] - img[..., 2]) * 2, 0, 1) * g
    return Plate(img, res['z'], cam, dict(sky=1 - res['a'], ground=g, glass=gm, warm=wm), extra=dict(glass_quad=q))
