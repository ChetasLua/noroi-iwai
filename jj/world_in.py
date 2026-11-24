"""Interiors and ground studies: subway platform, the long tiled hall, asphalt under a street lamp."""
import numpy as np, cv2, math
from .paint import rgb, f32, fbm, smooth, mix, grid, Canvas, arcs, jit_color, lum, warp_by
from .cam import Camera3D, rot_x, rot_y, rot_z, voronoi_crevice
from .comp import blur, radial, over
from . import world as Wd
from .world import C, Scene, Wall, Plate, painter, FAR
from .world_props import *


def _paint(img, res, seed, w, h, val=(.040, .034, .028), sizes=(13, 6.5, 3.2), mask=None):
    ang = res['ang'] + fbm(h, w, (160, 60), (1, .5), seed + 8) * .08
    return Wd.gouache(Wd.wobble(img, .45, 34, seed + 12), res['lab'], ang, seed=seed + 4, sizes=sizes, op=(.42, .40, .36), val=val, hue=.012, thr=(.012, .020, .030), mask=mask, aspect=3.2)


def column(S, x, y, r_, z1, r, tile='#c9cfc6', base='#4a4f58'):
    """square tiled pillar with chamfered corners, kick plate and a number-less enamel band"""
    q = r_; c = q * .22; p = np.array([(-q, -q + c), (-q + c, -q), (q - c, -q), (q, -q + c), (q, q - c), (q - c, q), (-q + c, q), (-q, q - c)]) + [x, y]
    S.prism(p, .0, z1, tile, tex=3); S.prism((p - [x, y]) * 1.02 + [x, y], 0, .22, base, tex=1); S.prism((p - [x, y]) * 1.015 + [x, y], 1.9, 2.18, '#2c4f6a', tex=0); S.prism((p - [x, y]) * 1.03 + [x, y], z1 - .25, z1, '#8a8e90', tex=1)


@painter('platform', 3)
def platform(w=2400, h=1350, seed=0, lit=1., eye=(1.5, -2.2, 1.58), target=(-.25, 9., 1.48), focal=1300., paint=True, ss=2, grime=1., flicker=None):
    """subway platform: tiled columns, tubes, tactile strip, track pit and the tunnel mouth.  lit=1 tubes on, .12 blackout (only emergency green and a far tube).
    masks: dark (how unlit every pixel is: put eyes there), tube, floor"""
    cam = Camera3D(eye, target, focal, w, h); S = Scene(seed + 2); r = np.random.default_rng(seed + 17); S.groups = 10
    X0, X1 = -3.1, 3.3; Zc = 3.25; Y0, Y1 = -9, 64
    tiles = '#8d948f'
    S.ground(X0, X1, Y0, Y1, 0., '#767a7c', nx=6, ny=40, tex=7, seed=5, d=(0, 1, 0))
    # edge: yellow tactile strip with raised dots, white line, coping
    S.quads(np.array([[X0 + .62, Y0, .012], [X0 + .95, Y0, .012], [X0 + .95, Y1, .012], [X0 + .62, Y1, .012]])[None], '#c99a2a', tex=8, bias=4e-4); yy = np.arange(Y0 + 4, 30, .075)
    k = yy[yy < 14]; 
    for u in (.68, .755, .83, .905): S.boxes(np.c_[np.full(len(k), X0 + u), k, np.full(len(k), .018)], [.036, .036, .014], 0, '#e0b23c')
    S.quads(np.array([[X0 + .12, Y0, .012], [X0 + .24, Y0, .012], [X0 + .24, Y1, .012], [X0 + .12, Y1, .012]])[None], '#c4c8c4', tex=8, bias=4e-4); S.boxes([X0 - .02, (Y0 + Y1) / 2, -.09], [.22, Y1 - Y0, .2], 0, '#9a9a94', tex=1)
    S.quads(np.array([[X0, Y0, -1.15], [X0, Y1, -1.15], [X0, Y1, -.2], [X0, Y0, -.2]])[None], '#34363a', tex=1)
    # track bed
    S.ground(X0 - 4.4, X0, Y0, Y1, -1.15, '#2c2a2a', nx=6, ny=40, tex=9, d=(0, 1, 0)); yy = np.arange(Y0, 50, .62); S.boxes(np.c_[np.full(len(yy), X0 - 2.1), yy, np.full(len(yy), -1.08)], [2.5, .24, .12], 0, '#4a4440', tex=12)
    for u in (-1.35, -2.85): S.boxes([X0 + u, (Y0 + Y1) / 2, -.93], [.09, Y1 - Y0, .16], 0, '#6a625c'); S.boxes([X0 + u, (Y0 + Y1) / 2, -.845], [.06, Y1 - Y0, .02], 0, '#c4ccd2')
    # far wall of the track: tiles, advertising frames with dead light, cables
    Wt = Wall(S, (X0 - 4.4, Y0), (X0 - 4.4, Y1), -1.15, Zc); 
    Wt = Wall.frame(S, (X0 - 4.4, Y1, -1.15), (0, -1, 0), (0, 0, 1), Y1 - Y0, Zc + 1.15); Wt = Wall.frame(S, (X0 - 4.4, Y0, -1.15), (0, 1, 0), (0, 0, 1), Y1 - Y0, Zc + 1.15)
    Wt.n = np.array([1., 0, 0]); Wt.rect(0, Wt.w, 0, 1.15, 0, '#3c3e40', tex=1); Wt.rect(0, Wt.w, 1.15, Wt.h, 0, tiles, tex=3); Wt.rect(0, Wt.w, 2.0, 2.16, .01, '#2c4f6a'); 
    for v in (3.4, 3.55, 3.66): S.seg([Wt.p(0, v, .08)], [Wt.p(Wt.w, v, .08)], .05, '#1c1c20')
    u = 6.
    while u < Wt.w - 4: 
        q = r.uniform(2.2, 3.4)
        if r.random() < .7: Wt.slab(u, u + q, 2.35, 3.75, 0, .09, '#30343a'); Wt.rect(u + .08, u + q - .08, 2.43, 3.67, .095, pick(r, ['#8a3a34', '#3a5a6a', '#c8b48a', '#4a5a48', '#6a4a5a']), ct=pick(r, ['#b8a070', '#2a3a4a', '#5a2a2a']), em=.12 * lit); graphic(Wt, u + .2, u + q - .2, 2.55, 3.55, .10, r, pick(r, NEON), em=.10 + .35 * lit, bg=False)
        u += q + r.uniform(.8, 2.5)
    # right wall: tiles, benches, vending, notice frames, pipes
    Wr = Wall.frame(S, (X1, Y1, 0), (0, -1, 0), (0, 0, 1), Y1 - Y0, Zc); Wr.rect(0, Wr.w, 0, Zc, 0, tiles, tex=3); Wr.rect(0, Wr.w, 0, .2, .02, '#3c4046'); Wr.rect(0, Wr.w, 1.02, 1.16, .01, '#2c4f6a')
    for yb in (6.2, 16.5, 29.):
        uu = Y1 - yb; Wr.slab(uu - 1.5, uu + 1.5, .40, .46, .05, .5, '#8a5a3a', tex=12); Wr.slab(uu - 1.5, uu + 1.5, .62, .86, .05, .10, '#8a5a3a', tex=12); Wr.slab(np.array([uu - 1.3, uu + 1.2]), np.array([uu - 1.2, uu + 1.3]), 0, .42, .1, .45, '#5a6066')
    for yb in (10.6, 12.0, 22.4): uu = Y1 - yb; c = pick(r, ['#8a2028', '#c8ccd0', '#24487a']); Wr.slab(uu - .5, uu + .5, 0, 1.85, 0, .75, c); Wr.rect(uu - .42, uu + .42, .75, 1.7, .76, '#f4ead0', em=.10 + 1.0 * lit, grp=1); u_ = np.tile(np.arange(5), 3) * .16 + uu - .38; v_ = np.repeat(np.arange(3), 5) * .3 + .8; Wr.rect(u_, u_ + .10, v_, v_ + .2, .77, np.stack([C(pick(r, ['#c8202e', '#2a7a4a', '#e8a02a', '#2a4a9a'])) for _ in u_]), em=.08 + .6 * lit, grp=1)
    for yb, q in ((3.2, 1.1), (14.2, 1.6), (19.6, 1.2), (26, 1.8)): uu = Y1 - yb; Wr.slab(uu - q / 2, uu + q / 2, 1.35, 2.25, 0, .05, '#2c3036'); lightb = pick(r, NEON); Wr.rect(uu - q / 2 + .05, uu + q / 2 - .05, 1.40, 2.2, .055, C(lightb[0]) * .8, ct=lightb[0], em=.06 + .4 * lit); graphic(Wr, uu - q / 2 + .12, uu + q / 2 - .12, 1.5, 2.1, .06, r, lightb, em=.06 + .4 * lit, bg=False)
    # ceiling: panels, cable trays, sprinkler pipes, hanging signs, tubes
    S.quads(np.array([[X0 - 4.4, Y0, Zc + .02], [X1, Y0, Zc + .02], [X1, Y1, Zc + .02], [X0 - 4.4, Y1, Zc + .02]])[None], '#6a6e70', tex=7, d=(0, 1, 0)); yy = np.arange(Y0, Y1, 3.)
    S.boxes(np.c_[np.full(len(yy), (X0 + X1) / 2 - 2.2), yy, np.full(len(yy), Zc - .12)], [X1 - X0 + 4.4, .22, .28], 0, '#5c6064', tex=1)
    S.tube([[1.9, Y0, 2.98], [1.9, Y1, 2.98]], .09, '#7a3a32', 7); S.tube([[2.2, Y0, 3.02], [2.2, Y1, 3.02]], .06, '#8a8e90', 6); S.boxes([-1.1, (Y0 + Y1) / 2, 3.0], [.4, Y1 - Y0, .10], 0, '#4a4e52')
    ty = np.arange(-3.2, 60, 3.2); on = np.ones(len(ty)); 
    if lit < .6: on[:] = 0; on[int(len(ty) * .62)] = .5
    if flicker is not None:
        for i in flicker: on[i] = .25
    for rowx in (-1.9, .5):
        for i, y in enumerate(ty):
            o = on[i] if rowx < 0 or lit >= .6 else 0; S.boxes([rowx, y, Zc - .28], [.22, 1.5, .07], 0, '#c8ccc8'); S.boxes([rowx, y, Zc - .335], [.10, 1.4, .05], 0, '#f2fbf0' if o else '#59605c', em=1.7 * o + .05, grp=0); 
            if o and y < 40: S.light((rowx, y, Zc - .5), '#e4f4e4', 7.5, 2.0, 1.25 * o, 0)
    for y in (4.6, 17., 33.): S.boxes([.4, y, 2.62], [1.9, .12, .42], 0, '#16304a'); W = Wall.frame(S, (-.5, y - .07, 2.43), (1, 0, 0), (0, 0, 1), 1.8, .38); graphic(W, .05, 1.75, .04, .34, .0, r, ('#16304a', '#f0f2ea', '#ffcf5a'), 'chev', em=.06 + .75 * lit); S.seg([[-.3, y, 2.83], [1.1, y, 2.83]], [[-.3, y, Zc], [1.1, y, Zc]], .03, '#30343a')
    for i, y in enumerate(np.arange(2.6, 60, 4.6)): column(S, -1.2, y, .34, Zc, r, '#aab0aa')
    # emergency exit lamp, tunnel mouth, far light
    if lit < .6: S.light((-.2, 27., 2.6), '#9fc4d8', 26, 6, 1.5, 0); S.light((X0 - 2., 46, 1.), '#5a7aa8', 30, 8, 1.2, 0); S.light((2.4, 1.0, 2.8), '#3a5a86', 7, 2.4, .9, 0)
    S.boxes([1.6, 24., 2.55], [.7, .1, .3], 0, '#1fd08a', em=1.1, grp=3); S.light((1.6, 23.4, 2.4), '#2adf98', 9, 2., 1.5 if lit < .6 else .5, 3)
    S.quads(np.array([[X0 - 4.4, Y1, -1.15], [X1, Y1, -1.15], [X1, Y1, Zc], [X0 - 4.4, Y1, Zc]])[None], '#08090c'); Yt = 40.; S.quads(np.array([[X0 - 4.4, Yt, 2.2], [X0, Yt, 2.2], [X0, Yt, Zc], [X0 - 4.4, Yt, Zc]])[None], '#24282c', tex=1)
    S.boxes([X0 - 3.9, 52, .9], [.16, .16, .22], 0, '#ff3a2a', em=2., grp=3); S.light((X0 - 3.9, 50, .9), '#ff3a2a', 8, 2, 1., 3)
    # litter, stains: leaflets, a dropped umbrella
    for _ in range(26): x, y = r.uniform(X0 + 1.2, X1 - .3), r.uniform(.5, 16); a = r.uniform(0, 3.14); c_, s_ = math.cos(a), math.sin(a); q = np.array([(-.11, -.15), (.11, -.15), (.11, .15), (-.11, .15)]) * r.uniform(.6, 1.2) @ np.array([[c_, s_], [-s_, c_]]) + [x, y]; S.quads(np.c_[q, np.full(4, .014) + r.uniform(0, .02, 4)][None], pick(r, ['#c8c4b8', '#b0b4b8', '#a89a84']), bias=5e-4)
    L = lit
    amb = (mix(C('#101828'), C('#59645f'), L), mix(C('#080b14'), C('#3e4644'), L))
    nk = dict(amb=amb, fills=[((.9, -.2, .1), mix(C('#0c1a2e'), C('#1c2426'), L)), ((-.9, .3, 0), mix(C('#0a0c1c'), C('#161c22'), L)), ((0, -1, .1), mix(C('#0a0e1a'), C('#101614'), L))], fog=dict(col=mix(C('#060a12'), C('#39433f'), L), dens=.022, start=10, keep=.3))
    baked = S._bake(cam); res = S.render(cam, ss=ss, baked=baked, **nk); X = cam.eye + cam.rays() * res['z'][..., None]; fl = ((np.abs(X[..., 2]) < .05) & (X[..., 0] > X0) & (res['z'] < FAR * .5)).astype(f32)
    sky = np.zeros((h, w, 3), f32) + C('#05060a')
    img = res['img'] * (1 + (Wd.occlusion(res['z'], .45, (4, 12, 36)) - 1) * res['a'])[..., None]; r2 = dict(res); r2['img'] = img
    img, _ = Wd.wet_reflect(S, cam, r2, sky, fl * (.30 + .18 * smooth(-.6, .9, fbm(h, w, (190, 60), (1, .5), seed + 4, aspect=3.))), baked=baked, strength=.8, vblur=18, ripple=.5, seed=seed, **nk)
    g = np.clip(fbm(h, w, (240, 90, 34), (1, .6, .35), seed + 33) * .5 + .2, 0, 1); img = img * (1 - .16 * grime * g)[..., None]
    if paint:
        img = _paint(img, res, seed, w, h); e = np.maximum(img - .75, 0); img = Wd.lamp_glow(img, e, ((7, .3), (28, .22), (90, .16))); img = Wd.tooth_pass(img, seed + 6)
    dark = smooth(.10, .02, cv2.GaussianBlur(lum(img), (0, 0), 6))
    return Plate(img, res['z'], cam, dict(dark=dark, tube=np.clip(lum(img) - .9, 0, 1), floor=fl))


@painter('hall', 3)
def hall(w=2400, h=1350, seed=0, tube=1., eye=(-1.35, -3.5, .52), target=(.5, 12., 1.15), focal=1350., paint=True, ss=2):
    """long empty tiled passage, one tube alive.  tube 0..1 = that tube's light (paint 1., .25 and 0. for a flicker).  masks floor.  the tube sits at (0, 8.2, 3.0)"""
    cam = Camera3D(eye, target, focal, w, h); S = Scene(seed + 4); r = np.random.default_rng(seed + 19); S.groups = 4; X = 2.4; Z1 = 3.05; Y0, Y1 = -8, 70
    S.ground(-X, X, Y0, Y1, 0., '#5f676a', nx=6, ny=50, tex=3, seed=4, d=(0, 1, 0))
    for sd in (-1, 1):
        W = Wall.frame(S, (sd * X, Y1 if sd > 0 else Y0, 0), (0, -sd, 0), (0, 0, 1), Y1 - Y0, Z1); W.rect(0, W.w, 0, 1.9, 0, '#9aa5a4', tex=3); W.rect(0, W.w, 1.9, Z1, 0, '#7e878a', tex=14); W.rect(0, W.w, 1.82, 1.94, .012, '#2d5566'); W.slab(0, W.w, 0, .16, 0, .03, '#363c42')
        yy = np.arange(2.0, 64, 6.4) + (0 if sd < 0 else 2.1)
        for y in yy:
            uu = (Y1 - y) if sd > 0 else (y - Y0)
            if r.random() < .55: q = r.uniform(1.0, 1.7); W.slab(uu - q / 2, uu + q / 2, .75, 1.7, 0, .05, '#2a2e34'); W.rect(uu - q / 2 + .05, uu + q / 2 - .05, .8, 1.65, .055, C(pick(r, ['#6a4a44', '#44525a', '#8a8068'])), ct=C(pick(r, ['#8a7a5a', '#3a4a52'])), em=.02)
            else: W.slab(uu - .5, uu + .5, 0, 2.15, -.02, .06, '#4c5a60'); W.slab(uu - .42, uu + .42, .02, 2.05, .0, .075, '#39464c'); W.slab(uu + .28, uu + .36, 1.0, 1.1, .07, .16, '#9aa0a4')
    S.quads(np.array([[-X, Y0, Z1], [X, Y0, Z1], [X, Y1, Z1], [-X, Y1, Z1]])[None], '#535b60', tex=7, d=(0, 1, 0)); yy = np.arange(Y0, Y1, 2.6); S.boxes(np.c_[np.zeros(len(yy)), yy, np.full(len(yy), Z1 - .06)], [2 * X, .14, .16], 0, '#485056')
    S.tube([[-2.05, Y0, 2.78], [-2.05, Y1, 2.78]], .07, '#5c4a44', 7); S.tube([[-1.85, Y0, 2.86], [-1.85, Y1, 2.86]], .045, '#7a8084', 6); S.boxes([1.9, (Y0 + Y1) / 2, 2.86], [.36, Y1 - Y0, .12], 0, '#41484e')
    ty = np.arange(-4.6, 66, 6.4)
    for i, y in enumerate(ty):
        a = abs(y - 8.2) < .1; S.boxes([0, y, Z1 - .12], [.24, 1.7, .08], 0, '#9aa0a0' if not a else '#d8dcd8'); S.boxes([0, y, Z1 - .18], [.10, 1.6, .05], 0, '#f0fff4' if a else '#4a5254', em=1.9 * tube if a else .04, grp=0)
        if a: S.light((0, y, Z1 - .4), '#d4f2ea', 22., 4.2, 3.0 * tube, 0)
    # a hanging broken fitting, fallen tiles, a leak
    S.seg([[-.6, 14.4, Z1], [.5, 14.5, Z1]], [[-.75, 14.4, 2.2], [.55, 14.5, 2.72]], .025, '#1c1e22'); S.boxes([-.1, 14.45, 2.45], [1.5, .22, .08], .06, '#8a9090', axes=(rot_y(.36)).T[None]); rubble(S, np.c_[r.uniform(-2.2, -1.3, 14), r.uniform(5, 12, 14)], r.uniform(.04, .14, 14), r, 0., ('#8a9494', '#7a8488'))
    S.boxes([0., Y1 - .2, 1.5], [2 * X, .2, 3.2], 0, '#06080a'); S.boxes([1.55, 34., 2.35], [.6, .1, .26], 0, '#2adf98', em=.9, grp=3); S.light((1.55, 33.3, 2.2), '#2adf98', 7, 1.6, 1.0, 3)
    nk = dict(amb=('#162128', '#0e1419'), fills=[((0, -1, .1), '#0c1318'), ((.8, 0, 0), '#0f1a20')], fog=dict(col='#0b1216', dens=.030, start=8, keep=.4)); baked = S._bake(cam); res = S.render(cam, ss=ss, baked=baked, **nk); Xp = cam.eye + cam.rays() * res['z'][..., None]; fl = ((np.abs(Xp[..., 2]) < .03) & (res['z'] < FAR * .5)).astype(f32)
    img = res['img'] * (1 + (Wd.occlusion(res['z'], .45, (4, 12, 40)) - 1) * res['a'])[..., None]; r2 = dict(res); r2['img'] = img; img, _ = Wd.wet_reflect(S, cam, r2, np.zeros((h, w, 3), f32), fl * .55, baked=baked, strength=.85, vblur=14, ripple=.4, seed=seed, **nk)
    if paint: img = _paint(img, res, seed, w, h); e = np.maximum(img - .75, 0); img = Wd.lamp_glow(img, e, ((8, .35), (34, .28), (110, .24))); img = Wd.tooth_pass(img, seed + 6)
    return Plate(img, res['z'], cam, dict(floor=fl))


@painter('asphalt', 3)
def asphalt(w=2400, h=1350, seed=0, lamp=(1180., 520.), amt=1., ppm=165., wet=0., tilt=.0):
    """cracked road from straight above, in a street lamp's pool.  ppm = plate px per metre, lamp = centre of the pool in px, wet 0..1.
    masks: light (0..1 pool of light), puddle; extra ppm"""
    x, y = grid(h, w); u, v = x / ppm, y / ppm; r = np.random.default_rng(seed + 3)
    q = 2 if w > 1600 else 1
    uu, vv = cv2.resize(u, (w // q, h // q)), cv2.resize(v, (w // q, h // q)); hh, ww = uu.shape
    n = lambda sc, sd, wt=None: fbm(h, w, sc, wt, sd)
    base = np.zeros((h, w, 3), f32) + C('#4b4d57'); tone = .10 * n((420, 160, 60), seed + 1) + .05 * n((24, 9), seed + 2); base = base * (1 + tone)[..., None]
    # repair patches with tarred seams
    lab = np.zeros((h, w), np.int32); seam = np.zeros((h, w), f32)
    for k, (cx, cy, pw, ph, a) in enumerate(((.18, .30, 2.6, 1.7, .06), (.80, .72, 3.4, 1.3, .06), (.55, .10, 1.5, 2.2, .06), (.36, .86, 2.2, 1.1, -.2))):
        c, s = math.cos(a), math.sin(a); X = (x - cx * w) * c + (y - cy * h) * s; Y = -(x - cx * w) * s + (y - cy * h) * c; wob = 4 * n((60, 20), seed + 20 + k); d = np.maximum(np.abs(X) - pw * ppm / 2, np.abs(Y) - ph * ppm / 2) + wob; m = d < 0; lab[m] = k + 1; base[m] *= (.80, 1.14, .90, .84)[k]; seam = np.maximum(seam, np.exp(-(d / 5.) ** 2))
    base = base * (1 - .55 * seam)[..., None]
    # body colour brushed on before any line work
    base = Wd.gouache(base, lab, fbm(h, w, (300, 100), (1, .5), seed + 60) * 1.2, seed=seed + 4, sizes=(26, 12, 6), op=(.5, .45, .4), val=(.05, .045, .04), hue=.014, thr=(.03, .05, .08), aspect=3.2, wander=.6)
    # white stop bar, worn zebra bars, a manhole
    worn = smooth(-.75, .25, n((46, 16, 6), seed + 5, (1, .8, .6)) + .25)
    bar = np.zeros((h, w), f32); ca, sa = math.cos(.06), math.sin(.06); X = (x - w * .5) * ca + (y - h * .5) * sa; Y = -(x - w * .5) * sa + (y - h * .5) * ca
    for i in range(4): xx = (-5.9 + i * .92) * ppm; bar = np.maximum(bar, ((X > xx) & (X < xx + .46 * ppm) & (Y > -9 * ppm) & (Y < -.9 * ppm)).astype(f32))
    bar = np.maximum(bar, ((X > 4.9 * ppm) & (X < 5.06 * ppm)).astype(f32)); bar = cv2.GaussianBlur(bar, (0, 0), 1.) * worn; base = base + (C('#b9b8b0') - base) * (bar * .85)[..., None]
    mx, my, mr = w * .735, h * .30, .33 * ppm; d = np.hypot(x - mx, y - my); th = np.arctan2(y - my, x - mx); mh = smooth(mr + 2, mr - 1, d); rib = (np.sin(th * 14) * np.sin(d / ppm * 60) > .1).astype(f32) * ((d > mr * .22) & (d < mr * .84)); ring = (np.abs(d - mr * .9) < 4).astype(f32); base = base + (C('#3a3838') * (1 + .35 * rib)[..., None] - base) * mh[..., None]; base = base * (1 - .5 * np.maximum(ring * mh, np.exp(-((d - mr) / 4) ** 2)))[..., None]
    # cracks: warped voronoi ridges, three scales, broken, with lit lower lips
    wx, wy = n((150, 50), seed + 7) * .10, n((150, 50), seed + 8) * .10; ht = np.zeros((h, w), f32)
    for sc, wd, lo, sd in ((2.9, .034, -.9, 1), (1.05, .016, .15, 2), (.40, .009, .80, 3)):
        e, idd = voronoi_crevice(u + wx * sc * .5, v + wy * sc * .5, sc, seed + sd * 7, .95); e = e * sc; m = smooth(wd * 1.0, wd * .25, e) * smooth(lo - .1, lo + .35, n((380, 130), seed + 30 + sd) * .8 + (.9 if sd == 1 else 0)); ht = np.maximum(ht, m * (1. if sd < 3 else .6)); base = base * (1 + .07 * (idd - .5) * (sc > .5))[..., None]
    lip = np.clip(np.roll(ht, (-2, -1), (0, 1)) - ht, 0, 1); base = base * (1 - .80 * ht)[..., None] + (lip * .12)[..., None]
    # aggregate
    g1 = np.random.default_rng(seed + 50).standard_normal((h, w)).astype(f32); g = cv2.GaussianBlur(g1, (0, 0), 1.1) * 2.2; base = base * (1 + .07 * g)[..., None]; sp = (cv2.GaussianBlur(g1, (0, 0), .8) > .72).astype(f32); base = base + (sp * .06)[..., None]
    # litter: leaves, gravel, gum
    cv = Canvas(h, w, img=np.clip(base, 0, 1)); k = 260; px, py = r.uniform(0, w, k), r.uniform(0, h, k); L = r.uniform(3, 8, k); cv.strokes(arcs(px + 2, py + 3, r.uniform(0, 3.1, k), L, 0), L * .34, np.tile(rgb('#1c1c24'), (k, 1)), op=.6, seed=seed, shape='round'); cv.strokes(arcs(px, py, r.uniform(0, 3.1, k), L, 0), L * .33, np.stack([rgb(pick(r, ['#7a7a80', '#8a8078', '#60626a'])) for _ in range(k)]), op=.9, seed=seed + 1, shape='round')
    k = 26; px, py = r.uniform(0, w, k), r.uniform(0, h, k); L = r.uniform(12, 26, k); an = r.uniform(0, 6.28, k); cv.strokes(arcs(px + 3, py + 4, an, L, .5), L * .26, np.tile(rgb('#17171e'), (k, 1)), op=.55, seed=seed + 2, shape='leaf'); cv.strokes(arcs(px, py, an, L, .5), L * .25, np.stack([rgb(pick(r, ['#7a5a34', '#8a6a3a', '#6a4430'])) for _ in range(k)]), op=.95, seed=seed + 3, shape='leaf')
    base = cv.img
    # light
    lx, ly = lamp; rr = np.sqrt(((x - lx) / (w * .34)) ** 2 + ((y - ly) / (h * .52)) ** 2) + .025 * n((260, 90), seed + 70); pool = np.exp(-(rr * 1.05) ** 2.4) * .86 + .14 * np.exp(-rr * 1.6); core = np.exp(-(rr * 2.1) ** 2)
    night = C('#10162e') * 1.0; lit = C('#ffc478') * 1.75
    img = base * (night * 2.1 + lit * (pool * amt)[..., None] + C('#fff0cc') * (core * .35 * amt)[..., None])
    pud = smooth(.35, .6, n((300, 110, 40), seed + 80, (1, .6, .35)) - .25 + .0 * x)
    if wet > 0: img = img * (1 - .35 * wet * (1 - pool))[..., None]; sk = C('#22305a') * .8 + lit * (.30 * np.exp(-(rr * 1.4) ** 2))[..., None]; img = img + (sk - img) * (pud * wet * .75)[..., None]; ed = np.clip(pud - np.roll(pud, (3, 2), (0, 1)), 0, 1); img = img + C('#ffd9a0') * (ed * pool * .8 * wet)[..., None]
    img = Wd.tooth_pass(img, seed + 6)
    return Plate(img, None, None, dict(light=np.clip(pool, 0, 1), puddle=pud * max(wet, 0)), extra=dict(ppm=ppm))
