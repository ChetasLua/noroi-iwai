"""b6: the evening on the hill.  Five backs, one sky; then a drop of ink falls into it."""
import os, math, numpy as np, cv2
from jj import W, H, comp, kit, world, world_city
from jj import fx
from jj.paint import rgb, f32, fbm, smooth, mix, grid, lum
from jj.cam import Camera3D
from jj.ink import Cel, spl, over_roi
from jj.comp import ease, over, blur
from jj import world_backs as WB
from jj.figure import Fig, P as PP, POSES, mix_pose

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
KEY = 'b6_v7'; BGKEY = 'b6_v3'
F0 = 1200.                                     # focal px of the frame (= plate focal 1500 * .8)
RW, RH = 2560, 1180
REF = Camera3D((0, .5, 1.25), (0, 10.5, 1.25), F0, RW, RH, cy=720)     # figures are drawn here, then carried by the moving camera
RAILZ = .98
HIT = (1226., 292.)


def camera(f):
    t = float(ease(np.clip((f - 2) / 66., 0, 1)))
    return dict(ey=.42 - 2.15 * t, ez=1.27 + .72 * t, hy=722 - 30 * t)


RAILK, RAILW, RAILHY, RAIL0, RAILH = 430., 3900, 0., 380., 620                 # px per metre, width, horizon row, first row, rows (the rail seen from (0, 0, 1.9))


def make_rail(shrubs=False):
    from jj.world_hill import guardrail, TIMES, bush
    from jj.world import Scene, gouache
    f = RAILK * 3.2; cam = Camera3D((0, 0, 1.9), (0, 10, 1.9), f, RAILW, RAILH, cy=RAILHY - RAIL0); S = Scene(9); S.r = np.random.default_rng(31); S.eye = cam.eye; r = S.r; x = np.linspace(-9, 9, 37)
    S.quads(np.array([[-9, 3.0, 0], [9, 3.0, 0], [9, 3.0, .14], [-9, 3.0, .14]])[None], '#a89c90', tex=1); S.quads(np.array([[-9, 3.0, .14], [9, 3.0, .14], [9, 3.6, .14], [-9, 3.6, .14]])[None], '#b8aca0', tex=1)
    guardrail(S, np.c_[x, np.full(37, 3.2), np.full(37, .12)], r, .86)
    if shrubs:
        S.P, S.Cc, S.Aa, S.Dd, S.Ll, S.B = [], [], [], [], [], []
        for x_ in np.arange(-8.6, 8.8, .42): hh = .18 + .16 * math.sin(x_ * 1.3) + .10 * math.sin(x_ * 3.1 + 1); bush(S, (x_ + r.normal(0, .12), 4.3 + r.uniform(0, .7), hh - .30), r.uniform(.42, .70), r, [('#3a4638', '#7a7448', '#24303a'), ('#344238', '#6e7048', '#20303a')][int(r.integers(2))], n=150, flat=.75, fine=2.2)
    T = TIMES['sunset']; res = S.render(cam, ss=2, smres=2048, amb=T['amb'], fills=T['fills'], sun=T['sun'], fog=T['fog']); a = res['a']; st = np.where(a[..., None] > .01, res['img'] / np.maximum(a[..., None], .01), 0); st = gouache(st, res['lab'], res['ang'], seed=24, sizes=(10, 5, 2.5), op=(.5, .5, .4), val=(.04, .036, .03), hue=.014, mask=(a > .5).astype(f32)); L = np.dstack([st * a[..., None], a]).astype(f32)
    return L if shrubs else kit.rim(L, -1.6, 2.0, '#ffc890', .85, .6)


def sit(H_):
    return 1.075 - .44 * H_


# ------------------------------------------------------------------ acting (f on twos)
SIT = dict(POSES['sit_rail'], cL=(0, -1), cR=(0, -1), hL=(.10, -.10, -.96), hR=(.10, -.10, -.96), eL=(.3, -1, .0), eR=(.3, -1, .0), wst=(8, 0, 0), cst=(8, 0, 0), nck=(6, 0, 0), gL='relax', gR='relax', wL=(0, 30, 0), wR=(0, 30, 0))


def lerp(a, b, t): return tuple(x + (y - x) * t for x, y in zip(a, b)) if isinstance(a, tuple) else a + (b - a) * t
def beat(f, a, b): return float(ease(np.clip((f - a) / float(b - a), 0, 1)))


def nobara(f):
    s = math.sin(f * 2 * math.pi / 18.); p = dict(SIT); lean = beat(f, 6, 18)
    p.update(pel=(-12, 2 * s, 0), wst=(8, 6 * lean + 1.5 * s, 0), cst=(6, 4 * lean, -4 * lean), nck=(4, 4 * lean, 10 * lean), hed=(-6 - 4 * lean, 5 * lean + 2 * s, -22 * lean), fL=(.06, .24 + .07 * s, .20 + .03 * s), fR=(.06, .20 - .07 * s, .20 - .03 * s), hL=(.14, -.08, -.95), hR=(.12, -.12, -.95)); return dict(pose=p, x=-1.80, y=3.2, z=sit(1.60), heading=0.)


def yuji(f):
    b = beat(f, 14, 30) - .55 * beat(f, 54, 62); p = dict(SIT); br = .6 * math.sin(f * .26)
    p.update(pel=(-12 - 6 * b, 0, 0), wst=lerp((8, 0, 0), (-4, 0, 0), b), cst=lerp((8, -2, 0), (-8, -3, 0), b), nck=lerp((6, 0, 0), (-8, 0, 0), b), hed=lerp((2, 2, 0), (-14, 6, 8), b), hL=lerp((.10, -.10, -.96), (.30, -.52, -.78), b), hR=lerp((.10, -.10, -.96), (.26, -.50, -.80), b), cL=(0 + 2.5 * b, -1 - 2 * b), cR=(2.5 * b, -1 - 2 * b)); p['cst'] = (p['cst'][0] + br, p['cst'][1], p['cst'][2]); return dict(pose=p, x=-.98, y=3.2, z=sit(1.73), heading=0.)


def megumi(f):
    b = beat(f, 20, 32); r = np.clip((f - 38) / 4., 0, 1) * (1 - beat(f, 52, 60)); w = math.sin(f * 2.1) * r; p = dict(SIT)
    p.update(pel=(-14, 0, 0), wst=(14, -2, 0), root=(0, 0, -.085 - .008 * r), cst=(14 + 6 * r, -2, 0), nck=(12 + 12 * r, -4 * b, 0), hed=(6 + 20 * r, -11 * b + 7 * w, 8 * b + 3 * w), hL=(.10, -.06, -.95), hR=(-.06, .34, -.74), eR=(.8, -.6, 0)); return dict(pose=p, x=-.17, y=3.2, z=sit(1.75), heading=0.)


def gojo(f):
    up = beat(f, 30, 38) * (1 - beat(f, 53, 62)); rf = np.clip((f - 37) / 3., 0, 1) * (1 - beat(f, 50, 54)); w = math.sin(f * 2.1) * rf; br = .5 * math.sin(f * .24 + 1)
    p = PP('pockets', cst=(2 + br, 2 - 5 * up, -6 + 10 * up), nck=(4, 0, 4), hed=(0, 3 + 6 * up, 2 + 22 * up), hL=lerp((.16, .06, -.95), (.20 + .05 * w, .74, .50), up), eL=lerp((.6, -1, 0), (.9, -.2, -.7), up), gL='splay' if up > .4 else 'relax', wL=lerp((0, 10, 0), (-60, 30 + 14 * w, 0), up), cL=(0 + 5 * up, 1 + 3 * up)); step_ = beat(f, 17, 30); p['pel'] = (p['pel'][0], p['pel'][1] - 5 * math.sin(math.pi * step_), p['pel'][2]); return dict(pose=p, x=.70 - .43 * step_, y=2.86, z=.022 * abs(math.sin(math.pi * 2 * step_)), heading=.10 * math.sin(math.pi * step_))


def nanami(f):
    b = beat(f, 16, 28); br = .5 * math.sin(f * .22 + 2); p = PP('contra', hL=(-.42, .42, -.44), hR=(-.42, .40, -.40), eL=(1, -.1, -.3), eR=(1, -.1, -.3), gL='relax', gR='relax', cst=(2 + br, -2, 4), nck=(4, 0, 4 * b), hed=(2, 2, 20 * b)); return dict(pose=p, x=1.44, y=3.0, z=0., heading=0.12)


CAST = [('nobara', nobara, 1.), ('yuji', yuji, 1.), ('megumi', megumi, 1.), ('nanami', nanami, 0), ('gojo', gojo, 0)]
N2 = 36


def dress(L, box, side=1):
    """backlight on a cel: sun rim on every top and right edge, translucent hair, violet fall-off"""
    x0, y0, x1, y1 = box; R = L[y0:y1, x0:x1].copy(); a = R[..., 3]
    h, w = a.shape; yy = np.linspace(0, 1, h, dtype=f32)[:, None]
    R[..., :3] *= (1.06 - .30 * yy)[..., None]
    for (dx, dy, c, am) in ((-3.2 * side, 2.0, '#ffb766', .95), (1.4 * side, 2.4, '#ff9c72', .45), (-1.3 * side, 1.0, '#fff0c0', .9)): R = kit.rim(R, dx, dy, c, am, .5)
    g = cv2.GaussianBlur(a, (0, 0), 9); R[..., :3] += (rgb('#ff9a5a') * .16) * ((1 - g) * a)[..., None] * 2
    return R


class Shot:
    id = 'b6'; n = 72
    finish = dict(bloom=.36, vig=.14, soft=.20)
    EVENTS = [(0, 'ambience', dict(note='evening hill: warm wind in grass and wires, far town, one crow, distant train')), (0, 'cloth', dict(size=.4, note='coats and hair in the wind, soft, through f54')),
              (4, 'creak', dict(size=.25, note='guard rail ticks under swinging feet, again 13, 22, 31')), (14, 'laugh', dict(size=.3, note='Yuji, small breath of a laugh as he leans back')), (18, 'step', dict(size=.25, note='Gojo steps in beside Megumi, 18-30')), (27, 'birds', dict(size=.6, note='flock crosses right to left until 50, wings')),
              (38, 'cloth', dict(size=.5, note='Gojo ruffles hair 38-50, rub rub')), (50, 'sub', dict(size=.4, note='everything warm is sucked away, wind stops')), (51, 'whoosh', dict(size=.2, note='one drop falling, tiny')),
              (54, 'impact', dict(size=1.0, note='ONE ink drop hits the sky like water: deep wet plink, long low tail')), (56, 'ripple', dict(size=.7, note='rings travel outward until 68, pitch bending down')), (60, 'ink', dict(size=.8, note='curtains start to run down the sky')), (62, 'ambience', dict(note='all evening sound gone, cold low air'))]

    def setup(s):
        cv2.setNumThreads(2); c = os.path.join(ROOT, 'cache', KEY); b0 = os.path.join(ROOT, 'cache', BGKEY)
        if not os.path.exists(b0 + '_bg.npy'):
            L = world.lazy('overlook'); im = L.f('img'); bg = cv2.resize(im, (1920, 1440), interpolation=cv2.INTER_AREA).astype(np.float16); del im
            m = cv2.resize(L.f('m_sky'), (1920, 1440), interpolation=cv2.INTER_AREA).astype(np.float16); se = cv2.resize(L.f('m_sea'), (1920, 1440), interpolation=cv2.INTER_AREA).astype(np.float16); del L
            for k, v in (('sky', m), ('sea', se), ('bg', bg)): np.save(b0 + '_%s.tmp.npy' % k, v); os.replace(b0 + '_%s.tmp.npy' % k, b0 + '_%s.npy' % k)
        c4 = c.replace('b6_v7', 'b6_v6')
        if not os.path.exists(c4 + '_rail.npy'): [(np.save(c4 + '_%s.tmp.npy' % q, make_rail(q == 'shrub').astype(np.float16)), os.replace(c4 + '_%s.tmp.npy' % q, c4 + '_%s.npy' % q)) for q in ('shrub', 'rail')]
        s.bg, s.sky, s.sea = [np.load(b0 + '_%s.npy' % k, mmap_mode='r') for k in ('bg', 'sky', 'sea')]; s.rail = np.load(c4 + '_rail.npy', mmap_mode='r'); s.shrub = np.load(c4 + '_shrub.npy', mmap_mode='r')
        s.pk, s.mt = {}, {}
        for ch, fn, _ in CAST:
            p = f'{c}_{ch}' if ch in ('gojo', 'nanami') else f'{c}_{ch}'.replace('b6_v7', 'b6_v6')
            if not os.path.exists(p + '.idx.npy'):
                F = Fig(ch).anim(fn).blow(.75, .1, .05); it, meta = [], []
                for k in range(N2):
                    L = F.at(k * 2).draw(REF, k * 2, light=(.25, .55, -.8), kick=(.35, .75, -.55), rim=dict(dir=(.35, .6, -.72), col='#ffb070', thr=.62, amt=.75), pal=dict(mul='#a08fb8'), expr=dict(mood='calm'), head_fn=WB.HEADS.get(ch))
                    if ch in ('nobara', 'yuji', 'megumi'): hm = cv2.dilate(F.mask('handL', 'handR'), np.ones((7, 7), np.uint8)); L = L * (1 - hm[..., None])
                    if ch == 'gojo' and 15 <= k <= 28: hm = cv2.dilate(F.mask('handL'), np.ones((7, 7), np.uint8)); L = L * (1 - hm[..., None])
                    ys, xs = np.nonzero(L[..., 3] > .01); bx = (max(0, xs.min() - 6), max(0, ys.min() - 6), min(RW, xs.max() + 7), min(RH, ys.max() + 7)); it.append((dress(L, bx, -1 if ch in ('nanami',) else 1), bx[0], bx[1])); h_ = F.info['head']; wr = F.info['wrists']; meta.append((h_['x'], h_['y'], h_['s'], *wr['L'], *wr['R'])); del L
                world.save_pack(p, it); np.save(p + '.meta.npy', np.array(meta, float)); del it, F
            s.pk[ch] = world.Pack(p); s.mt[ch] = np.load(p + '.meta.npy')
        s.veil = None; s.x, s.y = grid(H, W)
        s.nz = fbm(H, W, (220, 80, 30), (1, .6, .4), 5).astype(np.float16); s.hf = fbm(H, W, (14, 5, 2.4), (1, .8, .6), 6).astype(np.float16)

    # ------------------------------------------------------------ pieces
    def place(s, ch, k2, c, yd, sc=1.):
        """a reference drawing carried to the current camera: (roi, matrix) -> full layer"""
        R, ox, oy = s.pk[ch].get(k2); a = (F0 / (yd - c['ey'])) / (F0 / (yd - .5)); X0 = RW / 2; y_ref = 720.; z0 = 1.25
        # screen = hy + (ez - z) f / d ;  ref = 720 + (1.25 - z) f / dref
        ty = c['hy'] + (c['ez'] - z0) * F0 / (yd - c['ey']); M = np.float32([[a, 0, W / 2 + (ox - X0) * a], [0, a, ty + (oy - y_ref) * a]])
        if a < .72: R = cv2.GaussianBlur(R, (0, 0), .42 / a * .6)
        return cv2.warpAffine(R, M, (W, H), flags=cv2.INTER_LINEAR)

    def railing(s, c, which='rail'):
        a = (F0 / (3.2 - c['ey'])) / RAILK; ty = c['hy'] + (c['ez'] - 1.9) * F0 / (3.2 - c['ey']); R = (s.rail if which == 'rail' else s.shrub).astype(f32)
        if a < .8: R = cv2.GaussianBlur(R, (0, 0), .25 / a)
        return cv2.warpAffine(R, np.float32([[a, 0, W / 2 - RAILW / 2 * a], [0, a, ty + (RAIL0 - RAILHY) * a]]), (W, H), flags=cv2.INTER_LINEAR)

    def to(s, p, c, yd):
        a = (F0 / (yd - c['ey'])) / (F0 / (yd - .5)); ty = c['hy'] + (c['ez'] - 1.25) * F0 / (yd - c['ey']); return W / 2 + (p[0] - RW / 2) * a, ty + (p[1] - 720.) * a, a

    def hands(s, k, c, ry):
        cel = Cel(W, 260, ox=0, oy=int(ry) - 110)
        for ch in ('nobara', 'yuji', 'megumi'):
            m = s.mt[ch][k]
            for j, q in enumerate(((m[3], m[4]), (m[5], m[6]))):
                x, y, a = s.to(q, c, 3.2)
                if ch == 'megumi' and j == 1: continue                       # that hand lies in his lap
                WB.mitt(cel, x, ry - 1.5, 40 * a * (.9 if ch == 'nobara' else 1.), 1 if j == 0 else -1, seed=j + len(ch) * 7)
        return cel.layer()

    def ruffle(s, k, c):
        f = k * 2; up = beat(f, 30, 38) * (1 - beat(f, 53, 62))
        if up < .12: return np.zeros((H, W, 4), f32)
        m = s.mt['megumi'][k]; hx, hy, a = s.to((m[0], m[1]), c, 3.2); g = s.mt['gojo'][k]; gx, gy, a2 = s.to((g[3], g[4]), c, 2.86); on = float(np.clip((up - .72) / .28, 0, 1)); sway = .26 * math.sin(f * 2.1) * (36 < f < 51)
        x = gx + (hx + m[2] * a * (.18 + .16 * math.sin(f * 2.1) * (36 < f < 51)) - gx) * on * .9; y = gy + (hy - m[2] * a * .86 - gy) * on * .9; cel = Cel(W, H, ss=2)
        n = np.array([gx - x, gy - y]); L = np.hypot(*n) + 1e-6; n = np.array([-n[1], n[0]]) / L * 17 * a2 * .55
        if L > 4: cel.fill(np.array([(x + n[0], y + n[1]), (x - n[0], y - n[1]), (gx - n[0] * 1.15, gy - n[1] * 1.15), (gx + n[0] * 1.15, gy + n[1] * 1.15)]), '#14101c')
        WB.ruffle(cel, x, y, 42 * a2, sway + .25, 5); return cel.layer()

    def birds(s, f):
        cel = Cel(); t = (f - 25) / 24.
        if t < 0: return None
        r = np.random.default_rng(4)
        for i in range(13):
            lag = r.uniform(0, .34); v = r.uniform(.94, 1.08); x = 2010 - (t - lag * .5) * 1180 * v - i * 6; yb = 330 + 70 * math.sin(i * 1.9) + i * 9 - t * 120 + 16 * math.sin(t * 3 + i); sz = r.uniform(10, 17) * (1 + .25 * math.sin(i)); ph = (f // 2) * 1.25 + i * 2.1; fl = math.sin(ph) if (f // 2 + i) % 9 < 6 else .15
            if not -60 < x < W + 60: continue
            for sg in (-1, 1): cel.fill(spl(np.array([(x, yb), (x + sg * sz * .5, yb - sz * (.36 * fl + .12)), (x + sg * sz * 1.05, yb - sz * (.30 * fl - .16)), (x + sg * sz * .5, yb - sz * (.28 * fl - .06)), (x, yb + sz * .13)]), True, corners=(0, 2), n=4), '#3a2440')
            cel.fill(spl(np.array([(x - sz * .34, yb - 1), (x, yb - sz * .09), (x + sz * .30, yb + sz * .04), (x, yb + sz * .14)]), True, n=4), '#3a2440')
        return cel.layer()

    def frame(s, f):
        c = camera(f); k2 = f // 2
        oy = 1120 * .8 - c['hy']; M = np.float32([[1, 0, 0], [0, 1, -oy]]); y0 = int(oy) - 2; y1 = min(1440, y0 + H + 6)
        sub = np.float32([[1, 0, 0], [0, 1, -(oy - y0)]]); img = cv2.warpAffine(s.bg[y0:y1].astype(f32), sub, (W, H), borderMode=cv2.BORDER_REPLICATE); sky = cv2.warpAffine(s.sky[y0:y1].astype(f32), sub, (W, H), borderMode=cv2.BORDER_REPLICATE); sea = cv2.warpAffine(s.sea[y0:y1].astype(f32), sub, (W, H))
        sx, sy = 1152., c['hy'] - 30
        # sparkle on the bay (redrawn on twos)
        r = np.random.default_rng(900 + k2); tw = np.zeros((H // 2, W // 2), f32); n = 240; yy = c['hy'] + 14 + 150 * r.random(n) ** 1.5; xx = sx + r.normal(0, 1, n) * (14 + (yy - c['hy']) * .42)
        for a, b, q in zip(xx, yy, r.random(n)): cv2.ellipse(tw, (int(a * 4), int(b * 4)), (int((2 + 7 * q) * 4), 5), 0, 0, 360, float(.4 + .6 * q), -1, cv2.LINE_AA, 3)
        img = img + rgb('#fff0c4') * (cv2.resize(tw, (W, H)) * sea * .8)[..., None]
        if f >= 25:
            b = s.birds(f)
            if b is not None: img = over(img, b)
        dr = float(ease(np.clip((f - 54) / 14., 0, 1)))                     # cold, 0..1
        ink = None
        if f >= 54: img, ink = s.ink_sky(img, f, sky * (1 - sea), c); img = img * (1 - ink[..., 3:]) + ink[..., :3]
        img = over(img, s.railing(c, 'shrub'))
        sit_ = [s.place(ch, min(k2, N2 - 1), c, 3.2) for ch, fn, st in CAST if st]
        for L in sit_: img = over(img, L)
        rail = s.railing(c); img = over(img, rail); ry = c['hy'] + (c['ez'] - RAILZ) * F0 / (3.2 - c['ey']); q = F0 / (3.2 - c['ey']); top = smooth(ry + .050 * q + 1, ry + .050 * q - 1, s.y)[..., None]
        for L in sit_: img = over(img, L * top)
        img = over(img, s.hands(min(k2, N2 - 1), c, ry))
        img = over(img, s.place('nanami', min(k2, N2 - 1), c, 3.0)); img = over(img, s.place('gojo', min(k2, N2 - 1), c, 2.86)); img = over(img, s.ruffle(min(k2, N2 - 1), c))
        warm = 1 - dr
        # light through air: bloom round the sun, low veil of gold over everyone, two flares
        if warm > .02:
            d = np.hypot((s.x - sx) / 620, (s.y - sy) / 430); img = img + rgb('#ffbe6a') * (np.exp(-d * 2.6) * .17 * warm * (.94 + .06 * math.sin(f * .5)))[..., None]
            for (qx, qy, rr, col, am) in ((-.34, -.62, 60, '#ff8a6a', .07), (-.62, -1.0, 26, '#ffd0a0', .10), (.55, -.5, 96, '#ff7a6a', .05)):
                fx_, fy = sx + (W * .5 - sx) * qx * 2, sy + (H * .62 - sy) * qy * 1.6; dd = np.hypot(s.x - fx_, s.y - fy); img = img + rgb(col) * (smooth(rr, rr * .82, dd) * am * warm * .6)[..., None]
        if f >= 54:
            front = -300 + 1900 * ((f - 54) / 16.) ** 1.2; D = smooth(front + 120, front - 460, s.y)[..., None]; l = lum(np.clip(img, 0, 1.2))[..., None]
            lc = smooth(.02, 1.0, l); cold = (lc * np.array([.44, .56, .76], f32) * .92 + img * .08 + np.array([.004, .008, .022], f32)) * (.95 - .12 * dr); img = img * (1 - D) + cold * D
        elif f >= 51: img = over(img, s.drop(f))
        return img.astype(f32)

    def drop(s, f):
        cel = Cel(600, 520, ox=int(HIT[0]) - 300, oy=-60); q = (f - 50) / 4.; y = -70 + (HIT[1] + 70) * q ** 1.6; v = 18 + 46 * q; x = HIT[0]
        cel.fill(spl(np.array([(x, y - v * 1.5), (x + 4, y - v * .5), (x + 11, y + 4), (x + 8, y + 14), (x, y + 18), (x - 8, y + 14), (x - 11, y + 4), (x - 4, y - v * .5)]), True, corners=(0,), n=6), '#050307'); cel.fill(spl(np.array([(x - 6, y + 2), (x - 4, y - 8), (x - 3, y + 4), (x - 5, y + 8)]), True, n=4), '#7a5a8a')
        if f < 53: cel.fill(np.array([(x - 1.2, y - v * 4), (x + 1.2, y - v * 4), (x + .6, y - v), (x - .6, y - v)]), '#050307', .6)
        return cel.layer()

    def ink_sky(s, img, f, sky, c):
        t = f - 54; x, y = s.x - HIT[0], s.y - HIT[1]; rad = np.hypot(x, y * 1.22); ang = np.arctan2(y, x); nz = s.nz.astype(f32)
        wob = 1 + .022 * np.sin(ang * 5 + 1.3) + .014 * np.sin(ang * 11 - t * .3) + .018 * nz
        R = rad / wob; disp = np.zeros_like(R); A = np.zeros_like(R)
        for j, b in enumerate((0, 1.5, 3.5, 6, 9)):
            if t < b: continue
            u = (t - b); rr = 46 + 150 * u ** .74 + 8 * u; w = (25 - 3 * j) * math.exp(-u * .07) + 3; amp = math.exp(-u * .075); d = R - rr; disp += 20 * amp * np.exp(-(d / (w * 1.5)) ** 2) * np.sin(d / w * 2.0)
            continue
            th = w * (.34 + .30 * amp) * (.72 + .28 * np.sin(ang * 2 + j * 2 + .6)) + s.hf.astype(f32) * w * .16; A = np.maximum(A, smooth(1.3, -1.3, np.abs(d) - th) * smooth(-.06, .06, nz + .30 * np.sin(ang * 3 + j) + s.hf.astype(f32) * .30 + 1.15 * amp) * min(1., amp * 1.6))
        m = (disp * sky).astype(f32); nx, ny = x / (rad + 1), y / (rad + 1); img = cv2.remap(img, s.x + nx * m, s.y + ny * m, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        cel = Cel(W, int(c['hy']) + 20, ss=2); r = np.random.default_rng(3); k = '#050307'
        for j, b in enumerate((0, 1.5, 3.5, 6, 9)):                          # ragged brush rings: tapered arcs with gaps
            if t < b: continue
            u = t - b; rr = 46 + 150 * u ** .74 + 8 * u; amp = math.exp(-u * .085); wd = (21 - 3 * j) * amp + 1.4
            if amp < .18: continue
            n = 5 + j % 3; a0 = r.uniform(0, 6.28); cuts = np.sort(r.uniform(0, 1, n)) * 6.283 + a0; cuts = np.r_[cuts, cuts[0] + 6.283]
            for q in range(n):
                g0, g1 = cuts[q] + (.03 + .34 * (1 - amp)) * r.random(), cuts[q + 1] - (.03 + .34 * (1 - amp)) * r.random()
                if g1 - g0 < .10: continue
                th = np.linspace(g0, g1, max(8, int((g1 - g0) * rr / 26))); rad_ = rr * (1 + .016 * np.sin(th * 5 + j) + .010 * np.sin(th * 11 - t * .25) + .006 * r.normal(0, 1, len(th)))
                cel.line(spl(np.c_[HIT[0] + np.cos(th) * rad_, HIT[1] + np.sin(th) * rad_ / 1.22], n=4), wd * r.uniform(.7, 1.25), k, taper=(.16, .22), press=.55, tremor=1.6, seed=int(r.integers(1 << 20)), tmin=.10)
            for q in range(int(16 * amp)): th = r.uniform(0, 6.28); o = rr * r.uniform(1.0, 1.12); z_ = r.random() ** 2 * 5 * amp + .8; cel.fill(spl(np.c_[HIT[0] + math.cos(th) * o + np.cos(np.linspace(0, 6.28, 9)[:-1]) * z_ * 1.3, HIT[1] + math.sin(th) * o / 1.22 + np.sin(np.linspace(0, 6.28, 9)[:-1]) * z_], True, n=3), k)
        fx.ink_burst(cel, HIT[0], HIT[1], 64 + 150 * (1 - math.exp(-t * .30)), min(1., .30 + t * .12), 11, k, 24, 46)
        th = np.linspace(0, 6.283, 60)[:-1]; rb = (18 + 168 * (1 - math.exp(-t * .13))) * (1 + .10 * np.sin(th * 3 + .8) + .06 * np.sin(th * 7 - t * .1) + .03 * np.sin(th * 13)); bl = np.c_[HIT[0] + np.cos(th) * rb * 1.08, HIT[1] + np.sin(th) * rb * .86]; cel.fill(spl(bl, True, n=3), k)
        if t > 3: low = [tuple(p) for p in bl[(np.sin(th) > .15)][::2]]; L_ = [60 + 300 * ((i_ * 37) % 11) / 11. for i_ in range(len(low))]; fx.ink_runs(cel, [(p[0], p[1] - 8) for p in low], L_, min(1., (t - 3) / 13.), 5, k, (5, 17), 5., '#a8a0d8', .35)
        B = cel.layer(); A = np.clip(B[..., 3], 0, 1) * sky; Bc = B[..., :3] * sky[..., None]
        if f >= 59:
            if s.veil is None: cam = Camera3D((0, 0, 2), (0, 10, 2), F0, W, H, cy=692); s.veil = world_city.Veil(cam, W, H, seed=8, n=46, q=2)
            V = s.veil.layer(.10 + .46 * ((f - 58) / 14.) ** 1.1, rim='#4a4480', edge=1.9, wet=.75); va = V[..., 3] * sky; col = V[..., :3] * sky[..., None]; A2 = 1 - (1 - A) * (1 - va); return img, np.dstack([Bc * (1 - va[..., None]) + col, A2]).astype(f32)
        return img, np.dstack([Bc, A]).astype(f32)
