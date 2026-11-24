"""g1: snow on the ruins, blue dawn.  Very wide, very quiet: one small figure, a torn red strip, breath; his head lifts; the first pink touches one roof."""
import os, math, numpy as np, cv2
from jj import W, H, comp, kit, world
from jj import fx
from jj.paint import rgb, f32, fbm, smooth, mix, grid, lum
from jj.cam import Camera3D
from jj.ink import Cel, spl, over_roi
from jj.comp import ease, over, blur
from jj.figure import Fig, P as PP, POSES

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
KEY = 'g1_v3'
YX, YY, YZ = -.9, -23., .10
SUN = np.array([.385, .92, .135]); SUN /= np.linalg.norm(SUN)
N3 = 36


def cam_at(f):
    t = f / 71.; e = .35 * t + .65 * float(ease(t)); return dict(cx=1184 + 34 * e, cy=688 + 12 * e, zoom=.804 + .096 * e, dx=-17 + 34 * e, dy=-9 + 24 * e)


HEAD = -1.02; FWD = np.array([math.sin(1.02), math.cos(1.02)]); STEP = .46
def stride(f):
    """two slow steps: distance walked (m) and walk phase (strides)"""
    u1, u2 = float(ease(np.clip((f - 9) / 12., 0, 1))), float(ease(np.clip((f - 23) / 12., 0, 1))); return STEP * (u1 + u2), .5 * (u1 + u2)


PLANT = [(21, 1, STEP), (35, -1, 2 * STEP)]                                    # frame the foot lands, side, metres from the start


def gust(f): return math.exp(-((f - 41) / 6.5) ** 2) if f > 24 else 0.


def yuji(f):
    from jj.figure import walk, mix_pose
    u = float(ease(np.clip((f - 44) / 10., 0, 1))); u = u + .16 * math.sin(np.clip((f - 54) / 8., 0, 1) * math.pi); br = math.sin(f * .21); d, ph = stride(f); g = gust(f)
    st = PP('stand', fL=(.075, .04, .043), fR=(.07, -.05, .043), tL=(16, 0, 0), tR=(20, 0, 0), hL=(.15, .10, -.96), hR=(.17, .05, -.96)); w = float(np.clip((f - 7) / 5., 0, 1) * np.clip((39 - f) / 5., 0, 1)); p = dict(mix_pose(st, walk(ph + .25), .85 * w))
    p.update(wst=(9 - 8 * u + 4 * g, -2, 2), cst=(12 - 13 * u + .8 * br + 3 * g, -2, 3), nck=(22 - 25 * u + 6 * g, 0, 2), hed=(26 - 36 * u, 2, 4), cL=(-1 + .4 * br, 2 - 2 * u), cR=(-1 + .4 * br, 2 - 2 * u)); return dict(pose=p, x=YX + FWD[0] * (d - .5), y=YY + FWD[1] * (d - .5), z=YZ, heading=HEAD)


class Shot:
    id = 'g1'; n = 72
    finish = dict(bloom=.30, vig=.15, soft=.18, grain=.014)
    EVENTS = [(0, 'ambience', dict(note='winter dawn after everything: almost nothing. Snow hush, one far metal creak')), (6, 'wind', dict(size=.2, note='one slow breath of wind through broken floors, until 40')), (6, 'breath', dict(size=.35, note='his breath, shaky, visible cloud, also 27, 52')),
              (14, 'cloth', dict(size=.2, note='torn red strip flutters, light, through 72')), (21, 'step', dict(size=.35, note='foot into deep snow, dry crunch')), (35, 'step', dict(size=.35, note='second step, then stillness')), (36, 'wind', dict(size=.6, note='gust crosses the frame 34-50, snow hisses, cloth snaps')), (26, 'creak', dict(size=.2, note='hanging signal sways, rusty')), (45, 'cloth', dict(size=.3, note='head lifts; snow slides off a shoulder')), (45, 'light', dict(size=.5, note='first sun slides down one broken facade: one high glassy tone swelling to 72'))]

    def setup(s):
        cv2.setNumThreads(2); s.L = world.lazy('ruin', mode='snow', view='dawn', figure=(-.9, -23.), spike=.32); s.cam = s.L.cam; c = os.path.join(ROOT, 'cache', KEY)
        if not os.path.exists(c + '_z.npy'):
            z = np.minimum(s.L.f('z'), 4e3); rays = s.cam.rays(); Zw = s.cam.eye[2] + rays[..., 2] * z; Xw = s.cam.eye[0] + rays[..., 0] * z; Yw = s.cam.eye[1] + rays[..., 1] * z
            m = ((Zw > 14) & (z < 3e3)).astype(f32); tw = np.where((Yw > 30) & (Yw < 62) & (np.abs(Xw) < 18), Zw, 0); np.save(c + '_tw.npy', tw.astype(np.float16)); e = cv2.GaussianBlur(cv2.erode(z, np.ones((7, 7), np.uint8)), (0, 0), 2.2); np.save(c + '_z.tmp.npy', e.astype(np.float16)); os.replace(c + '_z.tmp.npy', c + '_z.npy')
        s.z = np.load(c + '_z.npy', mmap_mode='r'); s.tw = np.load(c + '_tw.npy', mmap_mode='r'); s.ttop = float(np.percentile(np.asarray(s.tw)[np.asarray(s.tw) > 10], 99.5))
        p = c + '_yuji'
        if not os.path.exists(p + '.meta.npy'):
            F = Fig('yuji').anim(yuji).blow(.5, -.1, 0); it, meta = [], []; k = 1.5; cam = Camera3D(s.cam.eye, s.cam.eye + s.cam.R[2], s.cam.f * k, 900, 900, cx=0, cy=0); q, _ = cam.project(np.array([[YX, YY, YZ + .9]])); cam.cx = 450 - q[0, 0]; cam.cy = 450 - q[0, 1]; cam.K = np.array([[cam.f, 0, cam.cx], [0, cam.f, cam.cy], [0, 0, 1.]])
            for j in range(N3):
                f = j * 2; L = F.at(f).draw(cam, f, light=(.72, .25, -.6), kick=(.8, .4, -.45), rim=dict(dir=(.8, .3, -.5), col='#ffc4a0', thr=.35, amt=.9), pal=dict(mul='#8fa0c8'), expr=dict(mood='sad')); ys, xs = np.nonzero(L[..., 3] > .01); b = (xs.min() - 4, ys.min() - 4, xs.max() + 5, ys.max() + 5); R = L[b[1]:b[3], b[0]:b[2]]; R = kit.rim(R, -2.4, 1.2, '#ffd0ae', .9, .5); it.append((R, b[0] - 450, b[1] - 450)); h = F.info['head']; meta.append((h['x'] - 450, h['y'] - 450, h['s']))
            world.save_pack(p, it); np.save(p + '.meta.npy', np.array(meta, float))
        s.pk = world.Pack(p); s.mt = np.load(p + '.meta.npy'); s.k = 1.5; s.x, s.y = grid(H, W); s.zref = 18.5
        vp, _ = s.cam.project(s.cam.eye + SUN * 1e4); s.vp = np.ravel(vp)

    def scr(s, P, c):
        """plate px + depth -> frame px"""
        (px, py), z = P; k = s.zref / max(z, 2.); return (px + c['dx'] * k - c['cx']) * c['zoom'] + W / 2, (py + c['dy'] * k - c['cy']) * c['zoom'] + H / 2

    def frame(s, f):
        c = cam_at(f); X = (s.x - W / 2) / c['zoom'] + c['cx']; Y = (s.y - H / 2) / c['zoom'] + c['cy']; Z = cv2.remap(s.z[:].astype(f32), X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE); k = s.zref / np.maximum(Z, 3.); mx, my = X - c['dx'] * k, Y - c['dy'] * k
        im = s.L.f('img'); img = cv2.remap(im, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101); del im
        # the first pink: a hard edge of light coming down one broken roof
        q = float(np.clip((f - 45) / 26., 0, 1)) ** .85
        if q > 0:
            tw = cv2.remap(s.tw[:].astype(f32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT); geo = (tw > 10).astype(f32); lim = s.ttop + 3.5 - 17. * q + .010 * (s.x - 820) + .8 * np.sin(s.x * .021 + 1) * np.sin(s.y * .017); m = smooth(lim - 4.2, lim + 4.2, tw) * cv2.GaussianBlur(geo, (0, 0), 1.2) * geo; l = lum(img)[..., None]
            warm = img * np.array([1.42, 1.10, .98], f32) * (.80 + .55 * smooth(.18, .62, l)) + np.array([.030, .010, .006], f32); img = img + (np.minimum(warm, 1.05) - img) * (m * (.30 + .62 * min(1, q * 1.6)))[..., None]
            halo = cv2.GaussianBlur(cv2.resize(m, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 14); img = img + rgb('#ffb08a') * (cv2.resize(halo, (W, H)) * .10 * q)[..., None]
        vx, vy = (s.vp[0] - c['cx']) * c['zoom'] + W / 2, (s.vp[1] - c['cy']) * c['zoom'] + H / 2; d = np.hypot((s.x - vx) / 300, (s.y - vy) / 210); img = img + rgb('#ffc6a6') * (np.exp(-d * 2.2) * (.04 + .22 * q + .012 * math.sin(f * .35)))[..., None]
        # slow drifting snow dust low over the ground
        # behind him
        ln_ = []
        img = over(img, s.blown(fx.snow(f, (560, 150, 0, 0), wind=-20., fall=58., seed=4, col='#eef3ff', shade='#b4c4e6', near=False, op=.85), f, .45))
        j = min(N3 - 1, f // 2); p, zz = s.cam.project(np.array([[YX, YY, YZ + .9]])); fx0, fy0 = s.scr((p[0], float(zz[0])), c); sc = c['zoom'] / s.k
        # footprints: an old trail leading up to him, and the two he makes now
        fp = [(-.5 - .46 * k - .02, 1 if k % 2 else -1, 1.) for k in range(1, 12)] + [(-.5, 1, 1.), (-.46, -1, 1.)] + [(d_ - .5, sd, 1.) for (fr, sd, d_) in PLANT if f >= fr]
        for (dd, sd, k_) in fp:
            c0 = np.array([YX, YY]) + FWD * dd + np.array([FWD[1], -FWD[0]]) * .10 * sd + (np.array([FWD[1], -FWD[0]]) * .05 * math.sin(dd * 2)); th = np.linspace(0, 6.283, 14)[:-1]; E = c0 + np.outer(np.cos(th) * .17, FWD) + np.outer(np.sin(th) * .075, [FWD[1], -FWD[0]]); pp, z_ = s.cam.project(np.c_[E, np.full(13, YZ - .01)]); Q = np.array([s.scr((pp[i], float(z_[i])), c) for i in range(13)]); qm = Q.mean(0); Q = (Q - qm) * [1.25, 2.6] + qm; ln_.append(Q)
        cel = Cel(W, H, ss=1)
        for Q in ln_: cel.fill(Q, '#4f6aa8', .72); cel.fill(Q * [1, .55] + [0, Q[:, 1].max() * .45], '#eef3ff', .55)
        img = over(img, cel.layer())
        # long shadow toward us
        g = []
        for (dx, L) in ((-.22, 0), (.22, 0), (.14, 9.), (-.14, 9.)):
            d_ = stride(f)[0] - .5; q_ = np.array([YX + FWD[0] * d_ + dx - SUN[0] * L, YY + FWD[1] * d_ - SUN[1] * L, YZ - .02 * L]); pp, z_ = s.cam.project(q_[None]); g.append(s.scr((pp[0], float(z_[0])), c))
        sh = np.zeros((H, W), f32); cv2.fillPoly(sh, [np.round(np.array(g) * 8).astype(np.int32)], 1., cv2.LINE_AA, 3); sh = cv2.GaussianBlur(sh, (0, 0), 2.2) * smooth(H * .99, fy0 + 30, s.y) * smooth(fy0 + 20, fy0 + 44, s.y); img = img * (1 - sh[..., None] * np.array([.30, .24, .13], f32))
        R, ox, oy = s.pk.get(j); wob = .0; M = np.float32([[sc, 0, fx0 + ox * sc], [0, sc, fy0 + oy * sc]]); Rb = cv2.GaussianBlur(R, (0, 0), .45 / sc * .6); L = cv2.warpAffine(Rb, M, (W, H), flags=cv2.INTER_LINEAR); img = over(img, L)
        img = over(img, s.strip(f, fx0 + s.mt[j][0] * sc, fy0 + s.mt[j][1] * sc, s.mt[j][2] * sc))
        img = s.breath(img, f, fx0 + (s.mt[j][0] + s.mt[j][2] * .85) * sc, fy0 + (s.mt[j][1] + s.mt[j][2] * .35) * sc)
        img = s.powder(img, f); img = over(img, s.blown(fx.snow(f, (0, 110, 46, 7), wind=-34., fall=74., seed=9, col='#f6f8ff', shade='#c2cfec', near=True, op=.92), f, 1.))
        return img.astype(f32)

    def blown(s, L, f, k):
        """the gust carries the whole fall sideways (integrated, so nothing jumps) and smears it"""
        D = -sum(gust(q) for q in range(f + 1)) * 34. * k; n = int(round(D)) % W; L = np.roll(L, n, 1) if n else L; g = gust(f) * k
        return kit.smear(L, -46 * g, 12 * g, 7) if g > .08 else L

    def powder(s, img, f):
        """snow dust torn off the drifts: long low wisps running right to left"""
        g = gust(f)
        if g < .05: return img
        r = np.random.default_rng(17); m = np.zeros((H // 2, W // 2), f32); t = (f - 26) / 24.
        for i in range(34):
            y = r.uniform(.60, .97) * H / 2; k = (y / (H / 2) - .55) / .42; x = (1.25 - ((t * (1.0 + 1.1 * k) + r.random()) % 1.5)) * W / 2; L = r.uniform(60, 230) * (.4 + k); P = np.array([(x + u * L, y - 5 * k * math.sin(u * 3 + i) + 3 * math.sin(u * 7 + i * 2)) for u in np.linspace(0, 1, 9)]); cv2.polylines(m, [np.round(P * 8).astype(np.int32)], False, float(r.uniform(.3, .9)), int(2 + 5 * k), cv2.LINE_AA, 3)
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=16, sigmaY=2.4); m = cv2.resize(m, (W, H)) * g; return img + (rgb('#f2f6ff') - img) * np.clip(m * 1.1, 0, .6)[..., None]

    def strip(s, f, hx, hy, r):
        """the torn end of his red hood"""
        cel = Cel(220, 160, ox=int(hx) - 120, oy=int(hy) - 70, ss=3); t = np.linspace(0, 1, 12); n = r * 3.6; a = np.array([hx - r * .55, hy + r * .95]); gust = .6 + .4 * math.sin(f * .09 + 1) + .5 * globals()['gust'](f); ph = f * (.36) + 2.2 * globals()['gust'](f)
        x = a[0] - n * t * (.92 * gust + .2); y = a[1] + n * t * (.30 - .34 * gust) + r * .55 * t * np.sin(t * 5.5 - ph) + r * .25 * t * np.sin(t * 10 - ph * 1.7); w = r * (.30 - .14 * t); Nn = np.c_[np.gradient(y), -np.gradient(x)]; Nn /= np.hypot(*Nn.T)[:, None]; P = np.c_[x, y]
        A, B = P + Nn * w[:, None], P - Nn * w[:, None]; tail = np.array([P[-1] + [-r * .30, r * .16], P[-1] + [-r * .05, 0], P[-1] + [-r * .36, -r * .14]]); cel.fill(np.vstack([A, tail, B[::-1]]), '#a3182a'); cel.fill(np.vstack([A[3:], P[3:][::-1]]), '#6e1226'); cel.line(A, 1.0, '#2a0c18', seed=f // 3); cel.line(B, 1.0, '#ffc8b0', seed=f // 3 + 1, alpha=.7)
        return cel.layer()

    def breath(s, img, f, x, y):
        for b in (6, 27, 52):
            u = (f - b) / 20.
            if not 0 <= u <= 1: continue
            a = 1.05 * min(1, u * 6) * (1 - u) ** 1.1; m = np.zeros((150, 240), f32); g = gust(f)
            for i in range(7): q = i / 6.; cx = 60 + (14 + 110 * u) * (q * .85 + .12) - 60 * g * q; cy = 92 - 34 * u * q + 6 * math.sin(i * 1.9 + b); cv2.ellipse(m, (int(cx * 8), int(cy * 8)), (int((5 + 22 * u * (1 - .45 * abs(q - .4))) * 8), int((4 + 14 * u * (1 - .4 * q)) * 8)), 0, 0, 360, 1., -1, cv2.LINE_AA, 3)
            m = cv2.GaussianBlur(m, (0, 0), 3.0); m = m * (.72 + .28 * fbm(150, 240, (22, 8), (1, .5), b + f // 3)) * a; x0, y0 = int(x) - 54, int(y) - 92
            if 0 < x0 < W - 240 and 0 < y0 < H - 150: img[y0:y0 + 150, x0:x0 + 240] += (rgb('#f0f5ff') - img[y0:y0 + 150, x0:x0 + 240]) * np.clip(m, 0, 1)[..., None]
        return img
