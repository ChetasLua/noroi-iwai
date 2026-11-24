"""White tropical beach, overexposed: palm shade, shallow sea, sandbar, far island."""
import numpy as np, cv2, math
from .paint import rgb, f32, fbm, smooth, mix, grid, Canvas, arcs, jit_color, lum, ramp, warp_by
from .cam import Camera3D, voronoi_crevice
from .comp import blur, radial, over
from . import world as Wd, world_sky as SK
from .world import C, Plate, painter, FAR


def frond(p0, ang, L, droop, r, n=26, wid=.30):
    """palm frond in a 2-D plane: returns (midrib polyline, [leaflet triangles as 3-pt polylines])"""
    t = np.linspace(0, 1, 18); a = ang + droop * t ** 1.4; d = np.c_[np.cos(a), np.sin(a)] * L / 17; mid = np.vstack([p0, np.asarray(p0) + np.cumsum(d, 0)])[:18]
    leaf = []
    for k in range(n):
        q = .10 + .9 * k / (n - 1.); i = q * 17; i0 = min(16, int(i)); p = mid[i0] + (mid[i0 + 1] - mid[i0]) * (i - i0); th = a[i0]
        for s in (-1, 1):
            if r.random() < .06: continue
            ln = L * wid * (math.sin(math.pi * min(1, q * .86 + .12)) ** .7) * r.uniform(.8, 1.1); sw = s * r.uniform(.62, .86) * (1.15 - .55 * q); m = p + np.array([math.cos(th + sw), math.sin(th + sw)]) * ln * .55; e = m + np.array([math.cos(th + sw * .75), math.sin(th + sw * .75) + .22]) * ln * .5; leaf.append(np.array([p, m, e]))
    return mid, leaf


@painter('beach', 3)
def beach(w=2400, h=1350, seed=0, eye=(0., 0., 1.35), target=(1.2, 10., 1.12), focal=1500., paint=True, sun=(.72, .05), glare=1.):
    """white beach at noon.  sand z=0 for y < shore(x) ~ 13 m; project people with P.cam.  masks: sky, sea, sand, shade (palm shadow on the sand), wet.
    layers: 'fronds' (palm leaves overhead, sway it), 'trunk'"""
    cam = Camera3D(eye, target, focal, w, h); r = np.random.default_rng(seed + 3); hz = Wd.horizon_y(cam); u, v, z = cam.uv((0, 0, 0), (1, 0, 0), (0, 1, 0)); gr = z > 0
    x, y = grid(h, w); sky = SK.day(h, w, hz + 2, seed + 4, sun[0]); n = lambda sc, sd, wt=None, asp=1.: fbm(h, w, sc, wt, sd, asp)
    v = np.where(gr, v, 1e5); u = np.where(gr, u, 0)
    shore = 12.5 + 1.6 * np.sin(u * .17 + .4) + .7 * np.sin(u * .43 + 1.7) + .06 * u
    wob = .25 * n((90, 30), seed + 5, (1, .5), 4.)
    d = v - shore + wob * np.clip(v * .06, .2, 1.5)
    # --- water: clear over white sand, then turquoise, then a cobalt line
    q = np.clip(d, 0, 4000); lq = np.log1p(q)
    sea = ramp(lq, [(0, '#eefaf4'), (.9, '#cdf3ec'), (1.7, '#96e4de'), (2.6, '#5fcfd8'), (3.6, '#37b2cf'), (4.6, '#2a93c4'), (5.8, '#3f8fc6'), (7.2, '#8cc4e4')])
    patch = smooth(.2, 1.0, n((360, 120, 40), seed + 7, (1, .6, .3), 6.)); sea = sea * (1 - .14 * (patch * smooth(6, 30, q) * (1 - smooth(120, 600, q)))[..., None])
    ca, _ = voronoi_crevice(u + .25 * np.sin(v * 2.1), v * .62 + .2 * np.sin(u * 1.7), .85, seed + 8, .9); caus = smooth(.16, .03, ca) * smooth(.4, 2.5, q) * (1 - smooth(10, 30, q)); sea = sea + (caus * .20)[..., None]
    rp = np.sin((v + .3 * np.sin(u * .5)) * np.clip(34 / np.maximum(v, 1), .15, 2.6) * 2.2 + n((140, 50), seed + 9, (1, .5), 5.) * 1.4); sea = sea * (1 + .035 * rp * smooth(3, 20, q))[..., None]
    sp = cv2.GaussianBlur(np.random.default_rng(seed + 10).standard_normal((h, w)).astype(f32), (0, 0), sigmaX=4, sigmaY=.7); sp /= sp.std(); glit = smooth(1.5, 2.6, sp + 1.7 * np.exp(-((x - w * sun[0]) / (w * .16)) ** 2) - .2) * smooth(8, 60, q); sea = sea + (glit * .7)[..., None]
    # foam: lace at the lip and two long breakers
    lace, _ = voronoi_crevice(u * 1.0 + .3 * n((60, 24), seed + 11, (1, .5), 3.), v * 1.0, .62, seed + 12, .95); lc = smooth(.30, .10, lace)
    foam = smooth(-.10, .12, d) * (1 - smooth(.25, 1.5, d)) * np.clip(.30 + .9 * lc, 0, 1)
    for (dd, wd, am) in ((4.2, .55, .9), (9.5, .75, .85), (21, 1.2, .8), (44, 1.6, .7)):
        dw = d - dd - .9 * np.sin(u * .12 + dd) - .5 * n((220, 80), seed + int(dd), (1, .5), 5.); brk = smooth(-.2, .7, n((260, 90), seed + 60 + int(dd), (1, .5), 3.) + .25); foam = np.maximum(foam, smooth(wd, wd * .15, np.abs(dw)) * brk * am * np.clip(.5 + lc, 0, 1)); foam = np.maximum(foam, smooth(0, wd * 2.6, dw) * (1 - smooth(wd * 2.6, wd * 6, dw)) * 0 + .35 * lc * smooth(wd * 3.4, 0, np.abs(dw + wd * 2)) * brk)
    sea = sea + (C('#ffffff') * 1.03 - sea) * np.clip(foam, 0, 1)[..., None]; hzb = smooth(hz + 26, hz, y) * gr; sea = sea + (C('#eefaff') - sea) * (hzb * .75)[..., None]
    # --- sand
    nn = n((300, 110, 40, 14), seed + 20, (1, .6, .4, .25), 2.2); drift = np.sin(v * 1.6 + u * .25 + 1.2 * n((260, 80), seed + 21, (1, .4), 2.)) * smooth(26, 3, v); sand = C('#fdf3e0') * (1 + .020 * nn + .016 * drift)[..., None]; sand = sand + (C('#efdcc4') - sand) * (smooth(.2, 1.4, nn + drift * .5) * .5)[..., None]
    g1 = np.random.default_rng(seed + 22).standard_normal((h, w)).astype(f32); sand = sand * (1 + .018 * cv2.GaussianBlur(g1, (0, 0), 1.2) * 2)[..., None]
    wet = smooth(-2.1, -.15, d); sand = sand + (C('#d9cdb8') - sand) * (wet * .75)[..., None]; refl = SK.day(h, w, hz + 2, seed + 4, sun[0])[::-1] if False else np.zeros_like(sand) + C('#d4f0fa'); sand = sand + (refl - sand) * (smooth(-1.1, -.05, d) * .55)[..., None]
    # footprints walking to the water
    fp = np.zeros((h, w), f32)
    for k in range(15):
        fu, fv = .9 + .10 * k + .12 * (-1) ** k, 4.2 + .62 * k; e = ((u - fu) / .07) ** 2 + ((v - fv) / .15) ** 2; fp = np.maximum(fp, np.exp(-e ** 2)); e2 = ((u - fu + .02) / .08) ** 2 + ((v - fv - .05) / .17) ** 2; sand = sand + (.05 * np.exp(-e2 ** 2) * (1 - np.exp(-e ** 2)))[..., None]
    sand = sand * (1 - .16 * fp)[..., None]
    m = smooth(-.06, .05, d)[..., None]; img = np.where(gr[..., None], sand * (1 - m) + sea * m, sky)
    # --- islands on the horizon
    for (cx, wd, hh, col, sd) in ((.20, .16, 52, '#8fb9cc', 1), (.29, .06, 26, '#a9cddb', 2), (.60, .05, 18, '#b4d6e2', 3)):
        rr = np.random.default_rng(seed + 40 + sd); t = np.linspace(0, 1, 60); prof = np.interp(t, np.linspace(0, 1, 9), rr.random(9)) * .55 + np.interp(t, np.linspace(0, 1, 23), rr.random(23)) * .25 + .4; prof *= np.sin(np.pi * t) ** .6; px = (cx + (t - .5) * wd) * w; py = hz + 1 - prof * hh; poly = np.vstack([np.c_[px, py], [[px[-1], hz + 1.5], [px[0], hz + 1.5]]]); mk = np.zeros((h, w), f32); cv2.fillPoly(mk, [np.round(poly * 8).astype(np.int32)], 1., cv2.LINE_AA, 3); sh = smooth(hz - hh * .1, hz - hh * .7, y + 5 * n((30, 12), seed + 41)); c = C(col) + (C('#d8ecef') - C(col)) * (.35 * sh)[..., None]; img = img + (c - img) * (mk * .85)[..., None]
    # --- palm shade on the sand (real projection of fronds from above)
    sh = np.zeros((h, w), f32); r3 = np.random.default_rng(seed + 51)
    def drop(P):
        q, zz = cam.project(np.c_[P, np.zeros(len(P))]); return q if (zz > .4).all() else None
    for (bx, by, L, k0) in ((-3.2, 6.6, 3.0, 9), (-5.4, 9.6, 3.0, 8)):
        for k in range(k0):
            mid, lf = frond((bx, by), k * 2 * np.pi / k0 + r3.uniform(-.2, .2), L * r3.uniform(.85, 1.1), r3.uniform(-.4, .4), r3, 30, .26)
            for tri in lf:
                nrm = np.array([-(tri[2] - tri[0])[1], (tri[2] - tri[0])[0]]); nrm = nrm / (np.hypot(*nrm) + 1e-6) * .035; P = np.array([tri[0] - nrm * .6, tri[1] - nrm, tri[2], tri[1] + nrm, tri[0] + nrm * .6]); q = drop(P)
                if q is not None: cv2.fillPoly(sh, [np.round(q * 8).astype(np.int32)], 1., cv2.LINE_AA, 3)
            q = drop(mid)
            if q is not None: cv2.polylines(sh, [np.round(q * 8).astype(np.int32)], False, 1., 3, cv2.LINE_AA, 3)
    q = drop(np.array([[-3.2, 6.6], [-4.6, 5.0], [-6.4, 3.6], [-6.1, 3.3], [-4.3, 4.8], [-3.0, 6.5]])); 
    if q is not None: cv2.fillPoly(sh, [np.round(q * 8).astype(np.int32)], 1., cv2.LINE_AA, 3)
    sh = cv2.GaussianBlur(sh, (0, 0), 1.6); sh = np.clip(sh + .35 * cv2.GaussianBlur(sh, (0, 0), 16), 0, 1) * gr * (1 - m[..., 0]); shade = img * np.array([.66, .70, .86], f32); img = img + (shade - img) * sh[..., None]
    # --- near beach grass, shells, driftwood
    cv = Canvas(h, w, img=np.clip(img, 0, 1)); ext = img - np.clip(img, 0, 1); k = 420; pu, pv = r.uniform(-4, 5, k), r.uniform(2.6, 11, k); q, zz = cam.project(np.c_[pu, pv, np.zeros(k)]); s = cam.f / zz * r.uniform(.006, .03, k); cv.strokes(arcs(q[:, 0] + 1, q[:, 1] + s * .4, r.normal(0, .2, k), s * 2.2, 0), s * .35 + .4, np.tile(rgb('#b9a894'), (k, 1)), op=.55, seed=seed, shape='round'); cv.strokes(arcs(q[:, 0], q[:, 1], r.uniform(0, 3.1, k), s * 2, .4), s * .5 + .4, np.stack([rgb(c) for c in r.choice(['#fffaf0', '#f0d8c8', '#e8c8b0', '#c8b8a8'], k)]), op=.9, seed=seed + 1, shape='round')
    img = cv.img + ext; lab = (np.where(gr, np.where(d > 0, 2, 1), 0)).astype(np.int32)
    if paint:
        a = fbm(h, w, (300, 100), (1, .5), seed + 30) * .10; img = Wd.gouache(img, lab, a, seed=seed + 31, sizes=(26, 12, 5), op=(.45, .42, .36), val=(.016, .016, .014), hue=.006, thr=(.012, .022, .036), aspect=5., wander=.08, mask=gr.astype(f32))
    # --- palm itself: trunk and fronds as layers (top left)
    def palm(cxy, base, scale, sd, dark):
        rr = np.random.default_rng(sd); cv = Canvas(h, w, color='#2a4a3a', layer=True); tr = Canvas(h, w, color='#8a7664', layer=True); top = np.array(cxy, float); b = np.array(base, float); t = np.linspace(0, 1, 24)[:, None]; c1 = (top + b) / 2 + np.array([-.06 * w, .02 * h]); P = (1 - t) ** 2 * b + 2 * t * (1 - t) * c1 + t * t * top
        tr.strokes(P[None], .0115 * w * scale, rgb('#9a8672')[None], op=1., shape='flat', seed=sd, bristle=.10); tr.strokes((P + [.004 * w * scale, 0])[None], .006 * w * scale, rgb('#d6c4aa')[None], op=.8, dry=.2, shape='flat', seed=sd + 1, bristle=.12); tr.strokes((P - [.006 * w * scale, 0])[None], .005 * w * scale, rgb('#5f5a60')[None], op=.7, dry=.2, shape='flat', seed=sd + 2, bristle=.1)
        kk = 34; i = np.linspace(1, 22, kk).astype(int); tr.strokes(arcs(P[i, 0], P[i, 1], .25 + rr.normal(0, .06, kk), .021 * w * scale, .3), 1.6 * scale, np.tile(rgb('#5a4e4c'), (kk, 1)), op=.75, seed=sd + 3, shape='round')
        fr = []
        for k in range(13): a = rr.uniform(-.5, np.pi + .5) + (0 if k % 2 else np.pi * .04); fr.append((a, rr.uniform(.24, .36) * w * scale))
        fr.sort(key=lambda q_: math.sin(q_[0]))
        for j, (a, L) in enumerate(fr):
            mid, lf = frond(top, -a * .9 + .3, L, (1.5 if math.cos(a) > 0 else -1.5) * rr.uniform(.6, 1.), rr, 46, .30); lt = j / 12.; base_c = mix(C('#2f5f4a'), C('#86b45a'), lt * .8 + .1)
            K = len(lf)
            if K: cv.strokes(np.array(lf), .0030 * w * scale * rr.uniform(.8, 1.2, K), jit_color(np.tile(base_c, (K, 1)), rr, .02, .03, .06), op=1., shape='leaf', seed=sd + 10 + j); hi = rr.random(K) < .4; LF = np.array(lf)[hi]; cv.strokes(LF - [0, 1.2], .0012 * w * scale, np.tile(mix(base_c, C('#f0ffc0'), .55), (len(LF), 1)), op=.8, shape='leaf', seed=sd + 30 + j)
            cv.strokes(mid[None], .0020 * w * scale, mix(base_c, C('#d8d8a0'), .4)[None], op=1., shape='taper', seed=sd + 50 + j)
        for k in range(6): p = top + rr.normal(0, .012, 2) * w * scale + [0, .012 * w * scale]; cv.strokes(arcs(p[0], p[1], 1.4, .016 * w * scale, .2), .008 * w * scale, rgb('#7a9a4a' if k % 2 else '#a8843a')[None], op=1., shape='round', seed=sd + 70 + k)
        return cv.layer().astype(f32), tr.layer().astype(f32)
    F1, T1 = palm((w * .085, h * .10), (-.05 * w, h * .93), 1.15, seed + 81, 0)
    img = Wd.tooth_pass(img, seed + 6, .006, .008)
    img = img + (C('#fff8e8') - img) * (.18 * glare * np.exp(-(((x - w * sun[0]) / (w * .42)) ** 2 + ((y - h * sun[1]) / (h * .6)) ** 2)))[..., None]
    return Plate(img * (1 + .04 * glare), np.where(gr, z, FAR), cam, dict(sky=1 - gr.astype(f32), sea=m[..., 0] * gr, sand=(1 - m[..., 0]) * gr, shade=sh, wet=np.clip(wet * (1 - m[..., 0]) * gr, 0, 1)), dict(fronds=F1, trunk=T1))
