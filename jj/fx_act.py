"""fx act: helpers shared by the action shots (d4, h3, h4): corridor travel, drawn fists with sleeves, colour-script tones."""
from .fx import *
from . import hands as HD, hands_sleeve as HS, kit, comp
from .cam import Camera3D

E3 = HD.euler


def f16(a):
    """memory-mapped float16 plate -> float32 for this frame only"""
    return np.asarray(a, f32)


NIGHTF = dict(HD.PAL['night'], skin='#c9aeb4', shade='#6a5478', tape='#cfcbd6', tape_shade='#7c7696', light='#e6ecff')
FIREF = dict(HD.PAL['fire'], tape='#f4dcae', tape_shade='#a4643c')


def fist(cx, cy, s, R, side='R', dist=2.6, pose='fist', pal=NIGHTF, light=(-.5, .75, .6), w=5., seed=3, res=480, tape=True, sleeve=True, spal=None, y0=.30, rimdir=None, rimcol=None, rim2=None, deep=None, bld=None, pivot=(0, .80, 0), cel=None, clip=(-200, -200, W + 200, H + 200), bands=None, nails='short', thr=0., layer=True):
    """one drawn fist + forearm + sleeve -> (premult layer or cel, geometry).  cx, cy = screen point of the knuckles, s px per hand unit (1 = wrist to knuckle)"""
    st = dict(tape=dict(fingers=HD.FING)) if tape else {}
    if bands: st['bands'] = bands
    h = HD.Hand(pose, side, R=R, pivot=pivot, bld=bld or HD.HEAVY, nails=nails, **st); g = HD.render([h], HD.Cam(cx, cy, s, dist), light=light, res=res, clip=clip, thr=thr); c = cel or Cel()
    sl = None
    if sleeve:
        sl = HS.sleeve(g, 0, y0=y0, y1=2.3, seed=seed, light=light); HS.draw_back(c, sl, spal or {})
    HD.fill_flat(c, g, pal); HD.fill_shadow(c, g, pal)
    if deep: HD.fill_polys(c, g, HD.depth_polys(g, light, -.28), deep)
    if rim2: HD.fill_polys(c, g, HD.rim_polys(g, rim2[0]), rim2[1])
    HD.fill_overlays(c, g, pal, w); HD.ink_lines(c, g, pal, w, seed)
    if sl is not None: HS.draw_cast(c, sl, spal or {}); HS.draw_front(c, sl, spal or {}, w, seed, rimdir, rimcol)
    return (c.layer() if layer and cel is None else c), g


def box_map(camP, camF, x0=-8.5, x1=8.5, far=118., rays=None):
    """remap fields that show a corridor plate (ground z=0, walls x0/x1, painted from camP) from camF: exact forward travel for a box street"""
    d = (camF.rays() if rays is None else rays); ex, ey, ez = camF.eye
    with np.errstate(divide='ignore', invalid='ignore'):
        tg = np.where(d[..., 2] < -1e-5, -ez / d[..., 2], 1e6); tl = np.where(d[..., 0] < -1e-6, (x0 - ex) / d[..., 0], 1e6); tr = np.where(d[..., 0] > 1e-6, (x1 - ex) / d[..., 0], 1e6); tf = np.where(d[..., 1] > 1e-6, (far - ey) / d[..., 1], 1e6)
    t = np.minimum(np.minimum(tg, tl), np.minimum(tr, tf)).astype(f32); X = camF.eye.astype(f32) + d * t[..., None]; c = (X - camP.eye.astype(f32)) @ camP.R.T.astype(f32); z = np.maximum(c[..., 2], .05)
    return (c[..., 0] / z * camP.f + camP.cx).astype(f32), (c[..., 1] / z * camP.f + camP.cy).astype(f32), t


def mono(img, amt, tint='#8ea2c8', gain=1., con=1.):
    """drain colour to a tinted grey"""
    g = cv2.cvtColor(np.clip(img, 0, 4).astype(f32), cv2.COLOR_RGB2GRAY); g = np.clip((g - .5) * con + .5, 0, 4) * gain; return img * (1 - amt) + g[..., None] * (C(tint) * 1.25) * amt


def radial_streaks(cx, cy, n=90, r0=420, r1=1700, seed=0, col='#cfe0ff', w=.006, al=.5, asp=.7):
    c = Cel(ss=1); r = R(seed, 3)
    for i in range(n):
        a = r.uniform(0, 6.283); q = r0 * r.uniform(.8, 2.2); l = r.uniform(120, 700); hw = (w * bms(r.random()) + .0006); p0 = np.array([cx + math.cos(a) * q, cy + math.sin(a) * q * asp]); p1 = np.array([cx + math.cos(a) * (q + l), cy + math.sin(a) * (q + l) * asp]); d = p1 - p0; m = np.array([-d[1], d[0]]) / (np.hypot(*d) + 1e-6) * hw * (q + l * .6); c.fill(np.array([p0, p0 + d * .65 + m, p1, p0 + d * .65 - m]), col, al * r.uniform(.4, 1))
    return c.layer()


def shade_by(img, cx, cy, r, inner=1.0, outer=.35):
    x, y = grid(); k = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (r * r))[..., None]; return img * (outer + (inner - outer) * k)


def toned(layer, mode, accent=RED, lo=.16, hi=.46):
    """premult layer -> its 2/3-tone version, still premultiplied"""
    a = layer[..., 3:4]; col = np.where(a > .02, layer[..., :3] / np.maximum(a, .02), 0).astype(f32); return np.dstack([impact_tone(col, mode, lo, hi, accent) * a, a])


def scale_layer(L, s, cx, cy, dx=0., dy=0., rot=0.):
    c, q = math.cos(rot) * s, math.sin(rot) * s; return cv2.warpAffine(L, np.float32([[c, -q, cx - c * cx + q * cy + dx], [q, c, cy - q * cx - c * cy + dy]]), (L.shape[1], L.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
