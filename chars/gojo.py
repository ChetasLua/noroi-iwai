"""SATORU GOJO.  Built on jj.head3.Head.      sheet: qa/gojo/sheet.jpg      keys: chars/gojo_keys.py

    H = gojo_head().pose(cx, cy, s, yaw, pitch, roll)            s = px from eye line to chin
    Gojo.head(cel, H, state, expr, light, f, wind, seed)          inks the head on a Cel (parts= 'back','face','front' to slip things between)
        state  'up' blindfold hair | 'down' fallen | 'wind' blown up and to the right
        expr   EXPR key or dict(band=, mouth=MOUTH|OPEN key, lid=LIDS key, brow=BROWS key, open 0..1, gaze=(x,y), iris, pupil, glow, lift 0..1, lside)
        light  day night violet blue red void under manga, or a palette dict.   key= 'L','R','BL','BR','T' or None overrides the cel-shadow direction
        wind   (x, y) tip push in lengths;  f drives the breath of the locks;  same seed = same drawing
    Gojo.bust(cx, cy, s, yaw, pitch, roll, state, expr, light, body=(yaw,pitch,roll), over=fn, under=fn, ...) -> premult layer
    Gojo.arm('cross'|'lift'|'open', light) -> hook for over= / under=        Gojo.hand(cel, pose, x, y, u, yaw, pitch, roll, light, side)
    rim(layer, dx, dy, col, amt, soft, where)  hard rim that ignores the frame edge      eye_glow(img, H, amt)  cold light of the irises
    six_eye(w, h, open, t, pupil, glow, gaze, refr, centre, scale, roll, mid=fn(cel,Q,k)) -> full-frame layer of the extreme close-up
"""
import numpy as np, math, cv2
from jj import W, H as HH, comp
from jj.paint import rgb, f32, smooth, fbm, grid
from jj.ink import Cel, spl, resample, ribbon, noise1
from jj.head3 import Head
from jj import kit
from jj.kit import outline, shape, hatch
from . import gojo_hair as GH
from . import gojo_face as GF
from . import gojo_eye as GE
from . import gojo_coat as GC
from .gojo_hair import surf, D2R
from .gojo_six import six_eye

S = spl


def gojo_head(bare=False):
    n = 1. if bare else 0.                        # the bare face: slimmer under the cheekbones, shorter nose
    return Head(a=[(-1.0, .045), (-.985, .10 - .006 * n), (-.95, .175 - .014 * n), (-.88, .285 - .024 * n), (-.74, .425 - .034 * n), (-.56, .538 - .036 * n), (-.36, .622 - .020 * n), (-.12, .670), (.12, .688), (.35, .694), (.6, .665), (.8, .535), (.93, .33), (1.0, .05)],
                f=[(-1.0, .66), (-.92, .735), (-.8, .745), (-.64, .765), (-.46, .77), (-.28, .755), (-.06, .735), (.06, .745), (.2, .80), (.4, .80), (.6, .73), (.8, .55), (.93, .32), (1.0, .05)],
                nose=(.085, -.04, -.415 + .050 * n, -.458 + .050 * n, .155 - .015 * n), ear=(.690, .10, -.40, -.12), cheek=.026)


# ------------------------------------------------------------------------------------------------ palettes
P0 = dict(line='#17131a', skin='#f9dccb', skin_sh='#dcaeaa', skin_dp='#8f626f', skin_lt='#fff1e6', lip='#c98684',
          hair='#f6f4fc', hair_bk='#d8d0ec', hair_sh='#c3b9e2', hair_dp='#8f84bd', hair_root='#b4a8dc', hair_root2='#cfc6ea', hair_bk2='#e6e0f4', hair_hi='#ffffff', hair_line='#2d2742', hair_in='#6a5f8c',
          band='#100d14', band_lt='#4d4956', band_mid='#211d28', band_edge='#5d5868',
          brow='#e9e6f6', brow_ln='#5a5478', white='#f2f5fb', white_sh='#b9c4de', iris='#5cc4f4', iris_dk='#1f62b4', iris_lt='#c8f6ff', pupil='#123a78', lash='#f7f8ff', lash_sh='#c6cbe6', lash_ln='#333252', lid='#1b1830',
          coat='#13101d', coat_lt='#54428c', coat_lt2='#3a2d66', coat_md='#2b2348', coat_dp='#07060c', coat_in='#0a0810', coat_ln='#050409')
PAL = dict(day=P0)
def _pal(**k): q = dict(P0, **k); q['skin_cool'] = k.get('skin_cool', q['skin_sh']); return q
P0.update(key=None, lit2=None, skin_cool='#c0a3c6')
PAL['night'] = _pal(skin='#a9a8d6', skin_sh='#615f9e', skin_dp='#34316a', skin_lt='#e8e8ff', lip='#7a6aa0', hair='#d6dbf6', hair_bk='#8481c0', hair_bk2='#a2a2d8', hair_sh='#8a89ca', hair_dp='#57549a', hair_root='#726fb8', hair_root2='#9c9cd6', hair_line='#1c1838', hair_in='#4c4880',
                    coat='#0d0b18', coat_lt='#36326e', coat_lt2='#26224c', coat_md='#1c1a38', white='#dfe6fb', white_sh='#8d98cc', brow='#c9c9ec', brow_ln='#4a4680', lash='#e9edff', lash_sh='#9a9fd0', key='L')
PAL['violet'] = _pal(line='#160c26', skin='#f0c8f2', skin_sh='#6a4aa2', skin_dp='#33205f', skin_lt='#fff0ff', lip='#a0509a', hair='#f7ecff', hair_bk='#6d56b4', hair_bk2='#8a70cc', hair_sh='#8b6bd0', hair_dp='#4a348e', hair_root='#7050bc', hair_root2='#b493ea', hair_line='#1f1238', hair_in='#5b3f9a',
                     coat='#0e0819', coat_lt='#7c4bd8', coat_lt2='#4a2a92', coat_md='#2c1a5c', coat_dp='#050309', white='#f0e2ff', white_sh='#9a7fd8', brow='#e6d2ff', brow_ln='#53378c', lash='#fbf2ff', lash_sh='#b79ae6', lash_ln='#2c1a52', lid='#150a2c', band='#0b0714', band_lt='#6a4ab8', band_mid='#1f1438', key='BL', lit2='#8f63cc')
PAL['blue'] = _pal(line='#0c1228', skin='#cfe9ff', skin_sh='#476aa8', skin_dp='#1f2f66', skin_lt='#f4fdff', lip='#5a78b4', hair='#f0fbff', hair_bk='#4f74b8', hair_bk2='#6c93d2', hair_sh='#6f9ad8', hair_dp='#2d4d94', hair_root='#4f78c4', hair_root2='#9cc4f2', hair_line='#101a3c', hair_in='#3a5a9c',
                   coat='#080c18', coat_lt='#2f7fe0', coat_lt2='#1a4a90', coat_md='#122a58', coat_dp='#030509', white='#e2f4ff', white_sh='#7ea4dc', brow='#d8ecff', brow_ln='#34528c', lash='#f4fcff', lash_sh='#9ec2ee', lash_ln='#182a56', lid='#081030', band='#080a14', band_lt='#3a76c8', band_mid='#101c38', key='BL', lit2='#6f9ee0')
PAL['red'] = _pal(line='#24080e', skin='#ffc6ac', skin_sh='#8a2f44', skin_dp='#4a1226', skin_lt='#fff0e2', lip='#b43c4a', hair='#fff0ea', hair_bk='#8e4a60', hair_bk2='#b4687a', hair_sh='#c87a86', hair_dp='#6a2438', hair_root='#a4505e', hair_root2='#eab0ae', hair_line='#340c18', hair_in='#8a3a4c',
                  coat='#150609', coat_lt='#d8342e', coat_lt2='#8a1c22', coat_md='#4a1018', coat_dp='#080203', white='#ffe9e0', white_sh='#c88a90', brow='#ffe0da', brow_ln='#7c2a3c', lash='#fff4ee', lash_sh='#e0a6a6', lash_ln='#4a1424', lid='#26060e', band='#120508', band_lt='#b03a3a', band_mid='#34101a', key='BL', lit2='#c86a64')
PAL['void'] = _pal(line='#f4f8ff', skin='#0a0a12', skin_sh='#0a0a12', skin_dp='#8ea0d8', skin_lt='#20243a', lip='#20243a', hair='#0d0e18', hair_bk='#08080f', hair_bk2='#08080f', hair_sh='#08080f', hair_dp='#000', hair_root='#08080f', hair_root2='#0c0c16', hair_hi='#9fc4ff', hair_line='#f4f8ff', hair_in='#7f95d0',
                   coat='#06060c', coat_lt='#1a1e36', coat_lt2='#12152a', coat_md='#0e1020', coat_dp='#020205', coat_in='#030308', coat_ln='#e8f0ff', white='#1b2340', white_sh='#0a0c18', brow='#0d0e18', brow_ln='#cfe0ff', lash='#eaf6ff', lash_sh='#31406e', lash_ln='#eaf6ff', lid='#dff2ff', band='#050509', band_lt='#2a3050', band_mid='#0e1020', band_edge='#fff')
PAL['under'] = _pal(line='#101412', skin='#b7c0a8', skin_sh='#647468', skin_dp='#38463f', skin_lt='#dfe6cf', lip='#84806a', hair='#dfe6da', hair_bk='#84938a', hair_bk2='#a0aea2', hair_sh='#8fa094', hair_dp='#55645c', hair_root='#7c8c82', hair_root2='#b4c0b4', hair_line='#1c2622', hair_in='#506058',
                    coat='#0c100f', coat_lt='#36473f', coat_lt2='#26332e', coat_md='#1b2622', coat_dp='#040605', white='#dbe4d8', white_sh='#8ea298', brow='#d2dccf', brow_ln='#46554c', lash='#e9f0e4', lash_sh='#a3b2a4', lash_ln='#26332c', lid='#101a16', iris='#5cc4f4', band='#0a0d0c', band_lt='#3c4a44', band_mid='#1a211e', key='T')


PAL['manga'] = _pal(line='#0b0a0c', skin='#f3f1ee', skin_sh='#b4b1b4', skin_dp='#6f6c72', skin_lt='#ffffff', lip='#8a868c', hair='#fbfafa', hair_bk='#d2d0d4', hair_bk2='#e6e4e6', hair_sh='#c2c0c6', hair_dp='#8c8a92', hair_root='#c6c4ca', hair_root2='#e2e0e4', hair_hi='#ffffff', hair_line='#0b0a0c', hair_in='#3a383e',
                    band='#0b0a0c', band_lt='#eceaea', band_mid='#1c1a1e', brow='#f6f5f5', brow_ln='#2a282e', white='#fbfafa', white_sh='#b9b7bd', lash='#ffffff', lash_sh='#c9c7cc', lash_ln='#1c1a20', lid='#0b0a0c',
                    coat='#0c0b0e', coat_lt='#f0eeec', coat_lt2='#8a888c', coat_md='#3a383c', coat_dp='#000000', coat_in='#050506', coat_ln='#000000', mouth='#0b0a0c',
                    iris_pal=dict(limb='#0b0a0c', ring='#2a2a2e', mid='#77757c', body='#c9c7cc', lite='#f4f3f4', core='#ffffff', pupil='#0b0a0c', pup2='#3a383e', cloud='#ffffff', cloud_sh='#cccccc', deep='#2a2a2e'))


# ------------------------------------------------------------------------------------------------ helpers
def rim(layer, dx, dy, col, amt=1.0, soft=.6, where=None):
    """kit.rim without a false edge where the figure leaves the frame"""
    a = layer[..., 3]; h, w = a.shape
    sh = cv2.warpAffine(a, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    m = np.clip(a - sh, 0, 1)
    if soft > 0: m = cv2.GaussianBlur(m, (0, 0), soft) * (a > .02)
    if where is not None: m = m * where
    c = rgb(col) if isinstance(col, str) else np.asarray(col, f32); out = layer.copy()
    out[..., :3] = out[..., :3] * (1 - m[..., None] * amt) + c * (m * a)[..., None] * amt
    return out


def eye_glow(img, H, amt=1., col='#7fe0ff', r=.30, sides=(-1, 1), gaze=(0., 0.)):
    """the irises are lamps: screen a small cold light back over the finished picture"""
    for sd in sides:
        if H.facing(sd * GF.EC, 0.) < .12: continue
        c = H.P([(sd * GF.EC + gaze[0], GF.EY + gaze[1])])[0]; R = r * H.s
        img = comp.screen(img, (comp.radial(c[0], c[1], R * 1.5, R, power=2.2)[..., None] * .55 + comp.radial(c[0], c[1], R * .34, power=1.2)[..., None] * .55) * rgb(col) * amt)
    return img


def _lw(H, w, lo=.55):
    """line weight for head scale: w is the width at s = 260"""
    return w * max(lo, (H.s / 260.) ** .55)


def _band_tabs():
    th = np.array([-180, -140, -112, -92, -78, -62, -46, -34, -25, -15, -7, 0, 7, 15, 25, 34, 46, 62, 78, 92, 112, 140, 180.])
    top = np.array([.40, .40, .405, .41, .405, .39, .375, .362, .356, .35, .347, .346, .347, .352, .360, .368, .380, .395, .407, .41, .405, .40, .40])
    # a flat wrap: the lower edge runs almost straight under the eyes, a hair higher over the nose
    bot = np.array([.04, .045, .05, .045, -.030, -.140, -.206, -.224, -.228, -.224, -.219, -.214, -.218, -.222, -.226, -.220, -.200, -.132, -.026, .045, .05, .045, .04])
    return th, top, bot


def band_pts(H, deg, y, off=.030):
    P, n = surf(H, np.asarray(deg, float) * D2R, np.asarray(y, float), off)
    return H.P3(P)


# ------------------------------------------------------------------------------------------------ the blindfold
def blindfold(cel, H, pal, seed=0, lift=0., lside=1, raise_=0., poly_only=False):
    th, top, bot = _band_tabs(); s = H.s
    c = -math.degrees(H.yaw); g = np.linspace(c - 90, c + 90, 181)
    if lift:                                    # one side pushed up off the eye (by a thumb hooked at about 46 deg)
        u = np.interp(th * lside, [-30, 2, 14, 30, 46, 62, 80, 96], [0, 0, .42, .92, 1.0, .80, .34, 0])
        bot = bot + lift * u * np.interp(th * lside, [0, 30, 46, 90], [.30, .46, .50, .22]); top = top + lift * u * .10
    if raise_: front = np.interp(np.abs(th), [0, 70, 110, 180], [1, .9, .5, .2]); bot = bot + raise_ * front * (.50 + .06 * np.sin(th * .13)); top = top + raise_ * front * .36
    w = lambda tab: np.interp(((g + 180) % 360) - 180, th, tab)
    # visible range per edge = normals that face us
    def edge(y, off=.030):
        P, n = surf(H, g * D2R, y, off)
        nz = np.stack([np.sin(g * D2R) / .69, np.zeros_like(g), np.cos(g * D2R) / .78], -1) @ H.R.T
        vis = nz[:, 2] > -.02
        return H.P3(P)[vis], g[vis]
    T, gt = edge(w(top) + .010 * np.sin(g * .05 + 1.), .026); B, gb = edge(w(bot), .034)
    def flank(ga, gb_, yt, yb, rs):
        r = np.random.default_rng(rs); k = 9; u = np.linspace(0, 1, k)[1:-1]
        off = .030 + np.array([.010, .030, .012, .034, .016, .004, .020])[:k - 2] * r.uniform(.6, 1.2, k - 2)
        ang = ga + (gb_ - ga) * u; y = yt + (yb - yt) * u
        P_, _ = surf(H, ang * D2R, y, 0.); out = H.P3(P_)
        ctr = H.P3(np.stack([0 * y, y, 0 * y], -1)); d = out - ctr; d /= np.hypot(d[:, 0], d[:, 1])[:, None] + 1e-6
        return out + d * (off * H.s)[:, None]
    wt = lambda a, tab: float(np.interp(((a + 180) % 360) - 180, th, tab))
    FR = flank(gt[-1], gb[-1], wt(gt[-1], top), wt(gb[-1], bot), seed + 3); FL = flank(gt[0], gb[0], wt(gt[0], top), wt(gb[0], bot), seed + 4)
    SR = spl(np.vstack([T[-1:], FR, B[-1:]]), n=4); SL = spl(np.vstack([T[:1], FL, B[:1]]), n=4)
    P = np.vstack([T, SR[1:-1], B[::-1], SL[::-1][1:-1]])
    if poly_only: return P
    cel.fill(P, pal['band'])
    cel.clip(P)
    hi, mid = pal['band_lt'], pal['band_mid']
    def fold(pts, col=hi, closed=True, corners=(0,)):
        a = np.asarray(pts, float); q = band_pts(H, a[:, 0], a[:, 1], .032)
        n = np.stack([np.sin(a[:, 0] * D2R), 0 * a[:, 0], np.cos(a[:, 0] * D2R)], -1) @ H.R.T
        if n[:, 2].max() < .05: return
        cel.fill(spl(q, True, corners=corners, n=8), col)
    # one soft turn of the cloth, then two folds that pull across it.  nothing that could read as eyes
    fold([(-92, .20), (-50, .13), (-8, .065), (30, .02), (60, .0), (30, -.035), (-12, .0), (-56, .06), (-92, .12)], mid, corners=(0, 4, 8))
    fold([(-96, .392), (-78, .368), (-54, .318), (-30, .296), (-52, .334), (-78, .382)], corners=(0, 3))
    fold([(-90, .285), (-62, .228), (-32, .158), (4, .092), (-28, .130), (-58, .188)], corners=(0, 3))
    fold([(92, .165), (66, .090), (36, .012), (-2, -.060), (32, -.020), (62, .050)], corners=(0, 3))
    fold([(94, .392), (74, .392), (54, .352), (70, .360), (86, .374)], corners=(0, 2))
    fold([(112, .36), (135, .33), (160, .345), (138, .30)], corners=(0, 2)); fold([(-112, .33), (-135, .30), (-160, .32), (-138, .27)], corners=(0, 2))
    # pulled threads: thin bright hairs of cloth near the bridge and at the ears
    for a, b, k in (((-86, .255), (-60, .195), 4), ((84, .25), (62, .165), 5), ((86, .205), (70, .14), 6)):
        u = np.linspace(0, 1, 7)[:, None]; q = np.array(a) * (1 - u) + np.array(b) * u; q[:, 1] -= .016 * np.sin(np.pi * u[:, 0])
        n = np.stack([np.sin(q[:, 0] * D2R), 0 * q[:, 0], np.cos(q[:, 0] * D2R)], -1) @ H.R.T
        if n[:, 2].min() > .1: cel.line(band_pts(H, q[:, 0], q[:, 1], .033), _lw(H, 1.5), hi, taper=(.4, .6), seed=seed + k, tmin=.05)
    if lift > .05:
        for a, b, k in (((46, .24), (8, .06), 31), ((46, .26), (14, .16), 32), ((46, .28), (18, .28), 33), ((46, .24), (84, .12), 34), ((46, .27), (90, .24), 35), ((44, .30), (40, .42), 36), ((46, .25), (70, .18), 37)):
            u = np.linspace(0, 1, 8)[:, None]; q = np.array(a) * (1 - u) + np.array(b) * u; q[:, 0] *= lside; q[:, 1] += (lift - 1) * .2 + .02 * np.sin(np.pi * u[:, 0])
            wd = .012 * lift; q2 = q.copy(); q2[:, 1] += wd * 2 * np.sin(np.pi * u[:, 0] ** .6)
            n = np.stack([np.sin(q[:, 0] * D2R), 0 * q[:, 0], np.cos(q[:, 0] * D2R)], -1) @ H.R.T
            if n[:, 2].min() > .05: cel.fill(np.vstack([band_pts(H, q[:, 0], q[:, 1], .033), band_pts(H, q2[:, 0], q2[:, 1], .033)[::-1]]), hi)
    cel.unclip()
    cel.line(T, _lw(H, 3.0), pal['line'], taper=(.08, .08), seed=seed + 11)
    cel.line(B, _lw(H, 3.2), pal['line'], taper=(.08, .08), seed=seed + 12)
    for q in (SL, SR): cel.line(q, _lw(H, 3.4), pal['line'], taper=(.1, .1), seed=seed + 13)
    return P


L3 = {'L': (-.75, .45, .45), 'R': (.75, .45, .45), 'BL': (-.80, -.35, .50), 'BR': (.80, -.35, .50), 'T': (.0, .95, .25), 'B': (.0, -.6, .6), None: None}


# ------------------------------------------------------------------------------------------------ face
def ear(cel, H, side, pal, seed=0, dark=False):
    """ear as a thin cup: authored in its own plane (u back, v up), hinged on the side of the head, opened 35 deg"""
    ex, t, b, z = H.k['ear']; sg = side; h = t - b
    if dark: pal = dict(pal, skin=pal['skin_sh'], skin_sh=pal['skin_dp'])
    e1 = np.array([sg * .58, 0., -.82]); e2 = np.array([sg * .10, 1., -.16]); o = np.array([sg * (ex - .035), b, z + .04]); nn = np.cross(e1, e2); nn *= np.sign((nn @ np.array([sg, 0, 0.2])))
    def Q(L, up=0.): a = np.asarray(L, float); return H.P3(o + a[:, :1] * h * e1 + a[:, 1:2] * h * e2 + nn * up)
    vis = float((H.R @ np.array([sg * .92, 0, .28]))[2])
    if vis <= -.16: return None
    out = spl([(-.02, -.02), (.06, -.085), (.16, -.04), (.235, .12), (.30, .30), (.40, .50), (.475, .72), (.47, .90), (.385, 1.03), (.24, 1.07), (.10, 1.01), (.0, .86)], n=8)
    P = Q(out)
    cel.fill(np.vstack([Q([(-.03, -.02), (.02, .5), (.02, 1.0)]), H.P3(np.array([(sg * .56, t + .02, .16), (sg * .56, (t + b) / 2, .16), (sg * .54, b - .10, .16), (sg * .60, b - .08, .02)]))]), pal['skin'])
    cel.fill(P, pal['skin'])
    # inner shadow: the hollow, with a hard zig-zag edge
    hol = spl([(.04, .86), (.14, .93), (.27, .93), (.345, .84), (.335, .68), (.27, .50), (.245, .34), (.20, .20), (.12, .18), (.075, .28), (.13, .40), (.15, .52), (.06, .62)], True, corners=(0, 6, 10), n=7)
    cel.fill(Q(hol), pal['skin_sh'])
    dk = spl([(.06, .62), (.15, .54), (.135, .40), (.08, .30), (.20, .26), (.225, .42), (.20, .60), (.13, .70)], True, corners=(0, 3), n=6)
    cel.fill(Q(dk), pal['skin_dp'], .8)
    cel.line(P, _lw(H, 3.6), pal['line'], taper=(.12, .12), press=.3, seed=seed + 3)
    cel.line(Q(spl([(.03, .84), (.13, .925), (.27, .93), (.36, .86), (.375, .72), (.315, .55), (.28, .38)], n=8)), _lw(H, 2.0), pal['line'], taper=(.2, .5), seed=seed + 4)
    cel.line(Q(spl([(.06, .62), (.145, .55), (.135, .41), (.07, .29)], n=8)), _lw(H, 2.0), pal['line'], taper=(.25, .35), seed=seed + 5)
    cel.line(Q(spl([(.10, .74), (.19, .72), (.25, .60), (.235, .44)], n=8)), _lw(H, 1.6), pal['line'], taper=(.4, .5), seed=seed + 6)
    return P


def _vis(H, x, y, t=.1): return H.facing(x, y) > t


MOUTH = {
    'half': dict(line=[(-.160, -.676), (-.11, -.689), (-.03, -.695), (.05, -.690), (.11, -.672), (.160, -.640)], end=[(.148, -.658), (.168, -.637), (.175, -.620)], low=[(-.05, -.772), (.005, -.779), (.06, -.770)], w=2.5),
    'smile': dict(line=[(-.180, -.662), (-.12, -.690), (-.03, -.706), (.06, -.703), (.13, -.684), (.182, -.650)], end=[(.170, -.664), (.190, -.636)], low=[(-.05, -.780), (.01, -.787), (.07, -.778)], w=2.5),
    'flat': dict(line=[(-.120, -.694), (-.05, -.688), (.02, -.690), (.08, -.687), (.130, -.694)], end=None, low=[(-.045, -.768), (.0, -.774), (.05, -.768)], w=2.4),
    'sad': dict(line=[(-.112, -.690), (-.06, -.697), (.0, -.697), (.065, -.693), (.118, -.678)], end=[(.112, -.682), (.130, -.670)], low=[(-.045, -.768), (.0, -.774), (.05, -.768)], w=2.3),
}


def mouth(cel, H, kind, pal, seed=0, flip=None):
    ln = pal['line']
    if flip is None: flip = H.yaw > .25
    mx = np.array([-1., 1.]) if flip else np.ones(2)
    if kind in OPEN: return open_mouth(cel, H, kind, pal, seed, mx)
    if kind in MOUTH:
        m = MOUTH[kind]
        cel.line(H.P(spl(m['line'], n=10) * mx), _lw(H, m['w']), ln, taper=(.35, .18), press=.3, seed=seed + 20, tremor=.2)
        if m['end'] is not None: cel.line(H.P(spl(m['end'], n=8) * mx), _lw(H, 2.2), ln, taper=(.4, .4), seed=seed + 21)
        low = np.array(m['low'], float) * mx
        lo = spl(low, n=8); sh = np.vstack([lo, (lo + (0, .006) * np.sin(np.linspace(0, np.pi, len(lo)))[:, None])[::-1]])
        cel.fill(H.P(sh), pal['skin_sh'])
        cel.line(H.P(lo), _lw(H, 1.7), pal['skin_dp'], taper=(.4, .5), seed=seed + 22)


OPEN = {
    'grin': dict(up=[(-.245, -.628), (-.15, -.664), (-.02, -.678), (.12, -.664), (.215, -.632), (.262, -.596)], lo=[(.225, -.690), (.13, -.752), (.0, -.772), (-.12, -.758), (-.205, -.700)], teeth=1),
    'laugh': dict(up=[(-.205, -.634), (-.12, -.652), (0, -.656), (.12, -.648), (.215, -.618)], lo=[(.185, -.715), (.105, -.795), (.0, -.815), (-.10, -.798), (-.175, -.725)], teeth=.35),
    'yell': dict(up=[(-.150, -.640), (-.08, -.660), (0, -.664), (.08, -.658), (.155, -.636)], lo=[(.165, -.740), (.10, -.850), (.0, -.875), (-.10, -.852), (-.160, -.745)], teeth=.25),
}


def open_mouth(cel, H, kind, pal, seed=0, mx=np.ones(2)):
    m = OPEN[kind]; ln = pal['line']
    U = spl(np.array(m['up'], float) * mx, n=10); Lo = spl(np.vstack([m['up'][-1], m['lo'], m['up'][0]]) * mx, n=10)
    P = H.P(np.vstack([U, Lo[1:]])); cel.fill(P, pal.get('mouth', '#3a0f1b')); cel.clip(P)
    xs = U[:, 0]; mid = lambda x: np.interp(x, np.sort(U[:, 0]), U[np.argsort(U[:, 0]), 1]); lw_ = lambda x: np.interp(x, np.sort(Lo[:, 0]), Lo[np.argsort(Lo[:, 0]), 1])
    if m['teeth'] >= 1:
        x = np.linspace(min(xs) + .012, max(xs) - .012, 40); yu, yl = mid(x), lw_(x)
        a = np.c_[x, yu - .004]; b = np.c_[x, yl + .016 * (1 - (np.abs(x) / .26) ** 2).clip(0)]
        cel.fill(H.P(np.vstack([a, b[::-1]])), pal.get('teeth', pal['white']))
        cel.fill(H.P(np.vstack([a, (a - (0, .020))[::-1]])), pal['white_sh'], .9)
        c = np.c_[x, yu * .46 + yl * .54]; cel.line(H.P(spl(c[::6], n=5)), _lw(H, 1.7), ln, taper=(.2, .2), seed=seed + 5)
        for k, xx in enumerate((-.165, -.105, -.052, .0, .052, .105, .165)):
            yu_, yl_ = float(mid(xx)), float(lw_(xx)); ym = yu_ * .46 + yl_ * .54
            cel.line(H.P(np.array([(xx, yu_), (xx * 1.01, (yu_ + ym) / 2), (xx, ym)])), _lw(H, .9), pal['skin_dp'], taper=(.1, .5), seed=seed + 8 + k, alpha=.75)
            if k in (1, 3, 4): cel.line(H.P(np.array([(xx + .02, ym), (xx + .02, (yl_ + ym) / 2), (xx + .02, yl_ + .02)])), _lw(H, .8), pal['skin_dp'], taper=(.1, .6), seed=seed + 18 + k, alpha=.6)
    else:
        x = np.linspace(min(xs) + .02, max(xs) - .02, 30); yu = mid(x)
        cel.fill(H.P(np.vstack([np.c_[x, yu], np.c_[x, yu - .055 * m['teeth'] / .35 * (1 - (np.abs(x) / .24) ** 4).clip(0)][::-1]])), pal.get('teeth', pal['white']))
        yb = min(p[1] for p in m['lo']); cel.fill(H.P(spl([(-.10, yb - .02), (-.085, yb + .055), (0, yb + .075), (.085, yb + .055), (.10, yb - .02)], True, n=8) * mx), pal.get('tongue', '#b4525e'))
    cel.unclip()
    cel.line(H.P(U), _lw(H, 3.0), ln, taper=(.16, .12), press=.3, seed=seed + 1); cel.line(H.P(Lo), _lw(H, 2.3), ln, taper=(.2, .2), press=.3, seed=seed + 2)
    for e, d in ((U[-1], (.028, .034)), (U[0], (-.024, .026))):
        e = np.array(e); d = np.array(d) * (np.sign(e[0]) * np.sign(d[0]), 1); cel.line(H.P(np.array([e - d * .4, e + d * .4, e + d])), _lw(H, 2.0), ln, taper=(.3, .5), seed=seed + 3)
    yb = min(p[1] for p in m['lo']); lo = spl([(-.06, yb - .035), (0, yb - .042), (.06, yb - .034)], n=8) * mx
    cel.line(H.P(lo), _lw(H, 1.8), pal['skin_dp'], taper=(.4, .5), seed=seed + 4)


def nose(cel, H, pal, light='L', seed=0, band=True, key=None):
    ln = pal['line']; yw = abs(H.yaw); sg = 1 if H.yaw >= 0 else -1; up = 0. if band else .052; U = np.array([0., up])
    if yw < .32:
        dz = .05
        if band:
            top = -.06
            tri = [(-.015, top), (-.012, -.20), (.000, -.30), (-.035, -.27), (-.085, -.205), (-.13, -.15), (-.11, top)]
            if not key: cel.fill(H.P(spl(tri, True, corners=(0, 2, 5, 6), n=6)), pal['skin_sh'])
            cel.line(H.P([(-.012, top), (-.004, -.15), (.004, -.255)], dz=.02), _lw(H, 1.9), ln, taper=(.1, .75), seed=seed + 30, tmin=.06)
        # only the tip: the small shadow under it and two ticks
        q = np.array([(-.040, -.437), (0, -.434), (.040, -.437), (.0, -.456)]) + U
        cel.fill(H.P(spl(q, True, corners=(0, 2), n=5), dz=dz), pal['skin_sh'])
        cel.line(H.P(np.array([(-.084, -.418), (-.066, -.433), (-.038, -.445)]) + U, dz=dz), _lw(H, 2.2), ln, taper=(.5, .25), seed=seed + 31)
        cel.line(H.P(np.array([(.038, -.445), (.066, -.433), (.084, -.415)]) + U, dz=dz), _lw(H, 2.2), ln, taper=(.25, .5), seed=seed + 32)
    else:
        N3 = H.nose_profile(); z = lambda y: float(H.z(0, y)); d = 1. if band else .92
        Y = lambda y: y + up
        if yw < 1.05:
            ridge = H.P3(np.array([(0, Y(-.10), z(Y(-.10)) + .035 * d), (0, Y(-.22), z(Y(-.22)) + .075 * d), (0, Y(-.33), z(Y(-.33)) + .125 * d), (0, Y(-.395), z(Y(-.395)) + .155 * d), (0, Y(-.425), z(Y(-.42)) + .150 * d)]))
            R_ = spl(ridge, n=10)
            if band:
                sh = np.vstack([R_[4:], H.P3(np.array([(sg * .055, -.44, z(-.44) + .05), (sg * .075, -.38, z(-.38) + .02), (sg * .055, -.26, z(-.26) + .015), (sg * .04, -.14, z(-.14) + .012)]))])
                cel.fill(sh, pal['skin_sh']); cel.line(R_, _lw(H, 2.5), ln, taper=(.55, .10), press=.25, seed=seed + 30, tmin=.06, tremor=.15)
            else: cel.line(R_[int(len(R_) * .58):], _lw(H, 2.4), ln, taper=(.75, .10), press=.25, seed=seed + 30, tmin=.05, tremor=.15)
        base = H.P3(np.array([(0, Y(-.425), z(Y(-.42)) + .148 * d), (-sg * .018, Y(-.452), z(Y(-.45)) + .095 * d), (-sg * .050, Y(-.458), z(Y(-.455)) + .055 * d)]))
        cel.line(spl(base, n=8), _lw(H, 2.3), ln, taper=(.15, .5), seed=seed + 31)
        if band: cel.line(spl(H.P3(np.array([(-sg * .098, -.405, z(-.4) + .012), (-sg * .104, -.432, z(-.43) + .004), (-sg * .080, -.446, z(-.44) + .02)])), n=8), _lw(H, 1.9), ln, taper=(.5, .3), seed=seed + 33)
        u = H.P3(np.array([(0, Y(-.43), z(Y(-.43)) + .14 * d), (-sg * .03, Y(-.455), z(Y(-.45)) + .07 * d), (-sg * .058, Y(-.452), z(Y(-.45)) + .03), (-sg * .030, Y(-.470), z(Y(-.47)) + .02)]))
        cel.fill(u, pal['skin_sh'])


def hair_shadow(cel, H, hair, pal, F_clip, dy=.10, dx=.03):
    """hard cool shadow of the fringe on the forehead"""
    cel.clip(F_clip)
    for it in hair.items:
        if it['dp'] > .35 and it['q'].tip[1] > -.3: cel.fill(it['P'] + np.array([dx, dy]) * H.s, pal.get('skin_cool', pal['skin_sh']))
    cel.unclip()


def face_line(cel, H, F, pal, seed=0, near=0, ycut=None):
    """ink the face contour; on a turned head the piece hidden by the near ear and the side of the skull is not drawn"""
    if not near:
        if ycut is None: outline(cel, F, _lw(H, 4.4), pal['line'], seed=seed + 7, press=.25); return
        o = H.P3(np.array([0, ycut, .1])); up_ = H.P3(np.array([0, ycut + .3, .1])) - o; up_ /= np.hypot(*up_) + 1e-6; hide = (F - o) @ up_ > 0      # the temples are under the hair
        n = len(F); st = int(np.argmax(hide & ~np.roll(hide, -1))) if hide.any() else 0; q = F[[(st + 1 + k) % n for k in range(n) if not hide[(st + 1 + k) % n]]]
        outline(cel, q, _lw(H, 4.4), pal['line'], seed=seed + 7, press=.25, closed=False); return
    ex, t, b, z = H.k['ear']
    e = H.P3(np.array([[near * .66, b + .03, -.03], [near * .66, .5, -.03], [0, b, .3]]))
    up = (e[1] - e[0]) / np.hypot(*(e[1] - e[0])); out = e[0] - e[2]; out = out - up * (out @ up); out /= np.hypot(*out) + 1e-6
    d = F - e[0]; v = d @ up; u = d @ out
    hide = (v > .06 * H.s) & (u > -.30 * H.s)
    n = len(F); st = int(np.argmax(hide & ~np.roll(hide, -1))) if hide.any() else 0
    idx = [(st + 1 + k) % n for k in range(n)]; keep = [k for k in idx if not hide[k]]
    q = F[keep]
    br = np.where(np.hypot(*np.diff(q, axis=0).T) > 14)[0]
    for a, c in zip(np.r_[0, br + 1], np.r_[br, len(q) - 1]):
        if c - a > 4: outline(cel, q[a:c + 1], _lw(H, 4.4), pal['line'], seed=seed + 7, press=.25, closed=False)


def undercut(cel, H, near, pal, seed=0):
    """short clipped hair round and behind the ear, under the band"""
    g = np.array([near * a for a in (74, 80, 92, 104, 120, 140, 160, 180, 200)])
    def pt(a, y, off=.02): P, n = surf(H, np.asarray(a, float) * D2R, np.asarray(y, float), off); return H.P3(P)
    up = pt(g, np.full(len(g), .20)); lo = [(74, .0), (78, -.16), (82, -.02), (92, .06), (104, .04), (112, -.10), (120, -.22), (126, -.18), (136, -.32), (144, -.27), (156, -.40), (164, -.36), (180, -.44), (200, -.42)]
    lo = np.array([(near * a, y) for a, y in lo]); L = pt(lo[:, 0], lo[:, 1])
    P = np.vstack([up, L[::-1]])
    cel.fill(P, pal['hair_bk'])
    r = np.random.default_rng(seed + 77)
    for a, y in lo[1:]:
        for k in range(2):
            aa = a + near * r.uniform(-7, 7); y0 = y + r.uniform(.06, .16)
            q = np.array([(aa, y0), (aa + near * 2, (y0 + y) / 2), (aa + near * 5, y + .02)])
            cel.line(spl(pt(q[:, 0], q[:, 1], .022), n=6), _lw(H, 1.5), pal['hair_in'], taper=(.5, .4), seed=int(r.integers(1 << 20)), tmin=.05)
    cel.line(L, _lw(H, 2.0), pal['hair_line'], taper=(.2, .2), seed=seed + 78)


def band_shadow(cel, H, pal, d=.045, lift=0., lside=1, raise_=0.):
    th, top, bot = _band_tabs()
    if raise_: bot = bot + raise_ * np.interp(np.abs(th), [0, 70, 110, 180], [1, .9, .5, .2]) * (.50 + .06 * np.sin(th * .13))
    if lift: bot = bot + lift * np.interp(th * lside, [-30, 2, 14, 30, 46, 62, 80, 96], [0, 0, .42, .92, 1.0, .80, .34, 0]) * np.interp(th * lside, [0, 30, 46, 90], [.30, .46, .50, .22])
    g = np.linspace(-62, 62, 80); b = np.interp(g, th, bot)
    P, n = surf(H, g * D2R, b - d - .02 * np.sin(g * .11 + .6), .004); Q, _ = surf(H, g * D2R, b + .03, .004)
    cel.fill(H.P3(np.vstack([P, Q[::-1]])), pal['skin_sh'])


def draw_neck(cel, H, pal, seed=0, w=.36, length=.95):
    """neck as a 3-D column under the jaw, silhouette from the rings"""
    length = max(length, .02); yy = np.array([-.05, -.25, -.45, -.62, -.8, -1.0, -1.2, -1.2 - length * .5]); rx = np.array([.5, .48, w + .05, w + .02, w, w - .005, w + .02, w + .18 * min(1, length / .9)]); rz = np.array([.60, .54, .44, .39, .36, .35, .37, .46]); cz = np.array([-.24, -.20, -.12, -.09, -.08, -.10, -.13, -.16])
    L, R = [], []
    ph = np.linspace(0, 2 * math.pi, 72, endpoint=False)
    for y, a, b, c in zip(yy, rx, rz, cz):
        Q = H.P3(np.stack([a * np.cos(ph), y + 0 * ph, c + b * np.sin(ph)], -1)); L.append(Q[Q[:, 0].argmin()]); R.append(Q[Q[:, 0].argmax()])
    L, R = spl(L, n=8), spl(R, n=8)
    cel.fill(np.vstack([L, R[::-1]]), pal['skin_sh'])
    cel.line(L, _lw(H, 3.6), pal['line'], taper=(.1, .1), seed=seed + 1); cel.line(R, _lw(H, 3.6), pal['line'], taper=(.1, .1), seed=seed + 2)
    # the two tendons
    for sg in (-1, 1):
        q = H.P3(np.array([(sg * .30, -.78, .12), (sg * .22, -1.05, .20), (sg * .10, -1.35, .26)]))
        cel.line(spl(q, n=8), _lw(H, 2.0), pal['skin_dp'], taper=(.5, .6), seed=seed + 5 + sg)


# ------------------------------------------------------------------------------------------------ the character
EXPR = {
    'smirk': dict(band=True, mouth='half'), 'smile': dict(band=True, mouth='smile'), 'plain': dict(band=True, mouth='flat'), 'laugh': dict(band=True, mouth='laugh'), 'lift': dict(band=True, mouth='half', lift=1., lside=1, lid='cold', glow=.2),
    'bare': dict(band=False, mouth='half', lid='calm', brow='calm'), 'cold': dict(band=False, mouth='flat', lid='cold', brow='cold'),
    'sad': dict(band=False, mouth='sad', lid='sad', brow='sad'), 'shut': dict(band=False, mouth='sad', lid='calm', brow='sad', open=0.), 'joy': dict(band=False, mouth='laugh', lid='laugh', brow='laugh', open=.35), 'mania': dict(band=False, mouth='grin', lid='wide', brow='wide', iris=.072, pupil=.22),
}


class Gojo:
    @staticmethod
    def head(cel, H, state='up', expr='smirk', light='day', f=0, wind=(0., 0.), seed=0, neck=True, parts=('back', 'face', 'front'), key='pal', neck_len=.95, sway=.012, hair_lift=0.):
        pal = PAL[light] if isinstance(light, str) else light
        e = EXPR[expr] if isinstance(expr, str) else expr
        s = H.s; lw = max(.5, (s / 260.) ** .5)
        key = pal.get('key') if key == 'pal' else key
        sway = Gojo.sway(f, wind, seed, sway)
        def capm3(y0, half, off):
            M = H._mesh(y0, 1, half); n = M / (np.linalg.norm(M, axis=-1, keepdims=True) + 1e-6); return M + n * off
        def capm(y0, half, off):
            M = H._mesh(y0, 1, half); n = M / (np.linalg.norm(M, axis=-1, keepdims=True) + 1e-6)
            q = H._sil([M + n * (off * np.clip((M[..., 1:2] - y0) / .22, .12, 1)) + np.array([0, .10, 0]) * np.clip((M[..., 1:2] - .7) / .3, 0, 1) ** 2 * (.3 if state == 'up' else 1)])
            k = 41; g = np.hanning(k); g /= g.sum(); qq = np.vstack([q[-k:], q, q[:k]]); return np.stack([np.convolve(qq[:, i], g, 'same') for i in (0, 1)], -1)[k:-k]
        if state == 'up':
            hair = GH.Hair(H, GH.UP, pal, cap=capm(.27, 1.0, .05), cap_back=capm(.30, 0, .05), seed=seed, sway=sway, lw=lw, light3=L3[key], thr=-.05, capM=capm3(.27, 1.0, .05) if key else None, gdk=((0, 'hair_root'), (.45, 'hair_sh'), (1, 'hair_sh'))); ST, HI = GH.UP_STRANDS, GH.UP_HI
        elif state == 'wind':
            hair = GH.Hair(H, GH.WIND, pal, cap=capm(.58, 1.0, .12), cap_back=capm(.0, 0, .12), seed=seed, sway=sway, lw=lw, light3=L3[key], thr=-.06, capM=capm3(.58, 1.0, .12) if key else None, fade_y=.72, grad=((0, .2, .6), (0, 1.0, .1)),
                           gst=((0, 'hair_root2'), (.5, 'hair'), (1, 'hair')), gbk=((0, 'hair_sh'), (1, 'hair_bk'))); ST, HI = GH.WIND_STRANDS, GH.WIND_HI
        else:
            hair = GH.Hair(H, GH.DOWN, pal, lift=hair_lift, cap=capm(.58, 1.0, .12), cap_back=capm(.0, 0, .12), seed=seed, sway=sway, lw=lw, light3=L3[key], thr=-.06, capM=capm3(.58, 1.0, .12) if key else None, fade_y=.72, grad=((0, 1.2, .3), (0, -.5, .6)),
                           gst=((0, 'hair'), (.5, 'hair'), (1, 'hair_root2')), gbk=((0, 'hair_bk'), (1, 'hair_sh'))); ST, HI = GH.DOWN_STRANDS, GH.DOWN_HI
        if 'back' in parts:
            hair.paint(cel, back=True, front=False); hair.ink_cap(cel, front=False); hair.ink(cel, back=True, front=False)
            if neck: draw_neck(cel, H, pal, seed, length=neck_len)
        if 'face' in parts:
            dk = lambda sd: (key in ('L', 'BL') and sd > 0) or (key in ('R', 'BR') and sd < 0)
            near = -1 if H.yaw > 0 else 1; turned = abs(H.yaw) > .22
            if H.pitch > 0: cel.fill(H.under(), pal['skin_sh'])
            F = H.face(top=(.33 + .30 * e.get('raise_', 0.)) if e['band'] else .74)
            if turned and abs(H.yaw) > .6:      # side of the head behind the cheek: short undercut, nape
                cel.fill(H._sil([H._mesh(-.42, .22, 0)]), pal['skin_sh'] if dk(near) else pal['skin'])
                undercut(cel, H, near, pal, seed)
            for sd in (-1, 1):
                if not (turned and sd == near): ear(cel, H, sd, pal, seed + sd, dk(sd))
            cel.fill(F, pal['skin'])
            cel.clip(F)
            GF.face_shade(cel, H, pal, key, pal.get('lit2'))
            nose(cel, H, pal, 'L', seed, band=e['band'], key=key)
            if e.get('lift') or e.get('raise_'):
                for sd_ in ((-1, 1) if e.get('raise_') else (e.get('lside', 1),)): GF.eye(cel, H, sd_, pal, lid=e.get('lid', 'cold'), brow=e.get('brow') if e.get('raise_', 0) > .6 else None, openk=e.get('open', 1.), gaze=e.get('gaze', (0, 0)), iris=e.get('iris', .083), pupil=e.get('pupil', .28), glow=e.get('glow', 0.), seed=seed, spin=e.get('spin', 0.), shade=False)
            if e['band']: band_shadow(cel, H, pal, lift=e.get('lift', 0.), lside=e.get('lside', 1), raise_=e.get('raise_', 0.))
            else:
                for sd in (-1, 1): GF.eye(cel, H, sd, pal, lid=e['lid'], brow=e['brow'], openk=e.get('open', 1.), gaze=e.get('gaze', (0, 0)), iris=e.get('iris', .083), pupil=e.get('pupil', .30), glow=e.get('glow', 0.), seed=seed, spin=e.get('spin', 0.))
            mouth(cel, H, e['mouth'], pal, seed, e.get('flip'))
            cel.unclip()
            face_line(cel, H, F, pal, seed, near if turned else 0, None if e['band'] else .10)
            if turned: ear(cel, H, near, pal, seed + near, dk(near))
        if 'front' in parts:
            if state != 'up' and not e['band']: hair_shadow(cel, H, hair, pal, F_clip=H.face(top=.74))
            hair.paint(cel, back=False, front=True, cap=True)
            if state != 'up' and hair_lift > .1 and not e['band']:          # fringe flung off the forehead: a ragged hairline, not the rim of a cap
                g_ = np.linspace(-62, 62, 25); yz = np.where(np.arange(25) % 2, .60, .515) + .012 * np.sin(g_ * .4)
                for a_, b_, col_ in ((-62, 13 - math.degrees(H.yaw), pal['hair']), (13 - math.degrees(H.yaw), 62, pal['hair_sh'] if key in ('L', 'BL') else pal['hair'])):
                    m_ = (g_ >= a_ - 3) & (g_ <= b_ + 3); up_, _ = surf(H, g_[m_] * D2R, np.full(m_.sum(), .74), .075); lo_, _ = surf(H, g_[m_] * D2R, yz[m_], .012); cel.fill(H.P3(np.vstack([up_, lo_[::-1]])), col_)
                for a_, y_ in zip(g_[1::2], yz[1::2]): q_, _ = surf(H, np.array([a_, a_ * .97, a_ * .9]) * D2R, np.array([y_ - .02, y_ + .08, y_ + .20]), .03); cel.line(H.P3(q_), _lw(H, 1.5), pal['hair_in'], taper=(.3, .7), seed=int(a_ + 90), tmin=.05)
            hair.ink_cap(cel, back=False); hair.strands(cel, ST); hair.hilite(cel, HI)
            if e['band']: blindfold(cel, H, pal, seed, lift=e.get('lift', 0.), lside=e.get('lside', 1), raise_=e.get('raise_', 0.))
            hair.ink(cel, back=False, front=True)
        return cel

    @staticmethod
    def bust(cx, cy, s, yaw=0., pitch=-.08, roll=0., state='up', expr='smirk', light='day', f=0, wind=(0., 0.), seed=0, hair_lift=0., body=(0., -.05, 0.), shift=(0., 0.), persp=.10, arms='down',
             left=True, right=True, over=None, under=None, key='pal', neck_len=.15, cel=None, h=HH, w=W):
        """head + collar + chest -> premultiplied layer.  (cx, cy) = centre of the head at eye level, s = px per head unit,
        body = (yaw, pitch, roll) of the trunk, shift = neck offset in head units, over / under = hooks fn(cel, H, B, pal) for arms, hands, props"""
        c = cel or Cel(w, h); pal = PAL[light] if isinstance(light, str) else light
        e_ = EXPR[expr] if isinstance(expr, str) else expr
        H = gojo_head(not e_['band']).pose(cx, cy, s, yaw, pitch, roll, persp); B = GC.Body().pose(cx + shift[0] * s, cy + (1.42 + shift[1]) * s, s, *body)
        kw = dict(state=state, expr=expr, light=pal, f=f, wind=wind, seed=seed, key=key, hair_lift=hair_lift)
        if under: under(c, H, B, pal)
        GC.collar_back(c, B, pal, seed)
        Gojo.head(c, H, parts=('back',), neck_len=neck_len, **kw)
        GC.torso(c, B, pal, seed, arms=arms, left=left, right=right)
        Gojo.head(c, H, parts=('face', 'front'), **kw)
        GC.collar_front(c, B, pal, seed)
        if over: over(c, H, B, pal)
        return c if cel is not None else c.layer(h, w)

    # ---- arms and hands (jj.hands)
    @staticmethod
    def hand(cel, pose, x, y, u, yaw=0., pitch=0., roll=0., light='day', side='R', seed=0, w=None, lit=(-.5, .75, .6), bld=None, **k):
        """his long hand: (x, y) = middle of the palm on screen, u = px from wrist to middle knuckle, angles in degrees (jj.hands.euler)"""
        from jj import hands as HD
        p = PAL[light] if isinstance(light, str) else light
        hp = dict(skin=p['skin'], shade=p['skin_sh'], line=p['line'], nail=p['skin_lt'], nail_shade=p['skin_sh'], light=p['skin_lt'])
        return HD.draw(cel, pose, HD.Cam(x, y, u), pal=hp, side=side, R=HD.euler(yaw, pitch, roll), bld=bld or HD.SLIM, w=w or max(2.2, 4.6 * (u / 170.) ** .55), light=lit, seed=seed, **k)

    @staticmethod
    def arm(pose, light='day', seed=0, hand=True, f=0, d=(0., 0.)):
        """hook for bust(over=...).  pose: 'cross' (two crossed fingers by the face, picture left) | 'lift' (thumb under the blindfold, picture right)
        | 'open' (both arms out, palms up).  d = extra hand offset in head units.  Wrist positions are stored on the hook: fn.wrists"""
        pal = PAL[light] if isinstance(light, str) else light
        def fn(c, H, B, pal_):
            s = H.s; o = np.array([H.cx, H.cy]); dd = np.array([d[0], -d[1]]) * s; fn.wrists = []
            def sleeve(S_, E_, Wp, r, sd, lit, hf):
                fn.wrists.append(Wp)
                GC.sleeve(c, B.P3(S_), B.P3(E_), Wp, r[0] * s, r[1] * s, r[2] * s, pal_, s, seed=seed + sd, lit=lit, hand_fn=hf if hand else None)
            if pose == 'cross':
                wr = o + np.array([-1.62, 1.18]) * s + dd
                sleeve((-1.46, -.62, .12), (-2.02, -2.25, .5), wr, (.50, .44, .30), 1, -1, lambda: Gojo.hand(c, 'cross', wr[0] + .05 * s, wr[1] - .50 * s, .64 * s, 158, 4, 2, pal_, 'R', seed))
            elif pose == 'lift':
                wr = o + np.array([1.42, .52]) * s + dd
                sleeve((1.46, -.60, .12), (2.28, -1.95, .5), wr, (.50, .44, .30), 2, 1, lambda: Gojo.hand(c, 'rest', wr[0] - .34 * s, wr[1] - .40 * s, .62 * s, 24, 10, 40, pal_, 'L', seed))
            elif pose == 'open':
                for sg, sd in ((-1, 3), (1, 4)):
                    wr = o + np.array([sg * 3.55, 1.45]) * s + dd * (sg, 1)
                    sleeve((sg * 1.22, -.56, .1), (sg * 2.62, -1.20, .35), wr, (.46, .42, .28), sd, -sg, lambda wr=wr, sg=sg, sd=sd: Gojo.hand(c, 'palm_up', wr[0] + sg * .28 * s, wr[1] - .34 * s, .58 * s, 200 if sg < 0 else 160, 52, -sg * 34, pal_, 'R' if sg < 0 else 'L', seed + sd, lit=(0, .9, .4)))
        fn.wrists = []
        return fn

    @staticmethod
    def sway(f, wind, seed=0, amp=.012):
        wx, wy = wind
        def fn(k, q):
            ph = k * 2.399 + seed; t = f / 24.
            a = amp * math.sin(t * 2.1 + ph) + wx * .35 * (1 + .35 * math.sin(t * 7 + ph))
            b = amp * .5 * math.sin(t * 1.7 + ph * 1.3) + wy * .35 * (1 + .35 * math.sin(t * 6 + ph * 1.7))
            return (a, b, 0.)
        return fn
