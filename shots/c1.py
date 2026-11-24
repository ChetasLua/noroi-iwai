"""c1: the veil comes down over the crossing.  Ink runs down the dome of the sky, heads turn in a wave, the blocks go out one after another; one red signal stays."""
import os, math, numpy as np, cv2
from jj import W, H, comp, kit, world, world_city
from jj.paint import rgb, f32, fbm, smooth, mix, grid, lum
from jj.cam import Camera3D
from jj.ink import Cel, spl, over_roi
from jj.comp import ease, over, blur

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
G = [[1.] * 10 for _ in range(5)]
for k in (1, 2, 3, 4): G[k][2] = 0.; G[k][3] = 0.
for k in (2, 3, 4): G[k][5] = 0.
for k in (3, 4): G[k][1] = 0.; G[k][8] = 0.
G[4] = [0.] * 9 + [1.]
CUTS = (36, 45, 54, 63)
RED = np.array([13.4 - 6.5, -20.5 + 1.5, 7.0])


def cam_at(f):
    t = float(ease(np.clip(f / 70., 0, 1))); z = .868 + .132 * t
    return dict(cx=1196 + 70 * t, cy=636 + 112 * t, zoom=z, dx=-16 + 32 * t, dy=-6 + 12 * t)


class Shot:
    id = 'c1'; n = 72
    finish = dict(bloom=.38, vig=.16, soft=.16)
    EVENTS = [(0, 'ambience', dict(note='Shibuya crossing at night: crowd, signals chirping, screens, traffic')), (0, 'ink', dict(size=.6, note='thick wet ink running down glass, from above, swelling until 54')), (10, 'sub', dict(size=.4, note='pressure rising 10-54')),
              (16, 'crowd', dict(size=.5, note='voices stop in a wave across the crossing 16-32, feet stop')), (28, 'phones', dict(size=.2, note='a few shutter clicks 28, 31, 33, 40')),
              (36, 'power', dict(size=.8, note='block of lights dies: relay thunk + 1-frame flicker, hum drops')), (45, 'power', dict(size=.85, note='second block dies, 1-frame flicker')), (54, 'power', dict(size=.95, note='big screen dies, music cut dead, 1-frame flicker')), (54, 'ink', dict(size=.9, note='the veil closes on the horizon: seal')),
              (63, 'power', dict(size=1., note='everything dies; only one signal relay ticking')), (66, 'tick', dict(size=.3, note='red signal relay tick, again 70')), (63, 'silence', dict(note='crowd holds its breath'))]

    def setup(s):
        cv2.setNumThreads(2); s.L = [world.lazy('scramble', lights=tuple(g), crowd_n=0) for g in G]; s.cam = s.L[0].cam; c = os.path.join(ROOT, 'cache', 'c1_v3')
        if not os.path.exists(c + '_z.npy'):
            z = s.L[0].f('z'); z = np.minimum(z, 4e3); z = cv2.GaussianBlur(cv2.erode(z, np.ones((7, 7), np.uint8)), (0, 0), 2.2).astype(np.float16); np.save(c + '_z.tmp.npy', z); os.replace(c + '_z.tmp.npy', c + '_z.npy')
            lm = np.stack([cv2.resize(cv2.GaussianBlur(L.f('img'), (0, 0), 14), (300, 169), interpolation=cv2.INTER_AREA) for L in s.L]).astype(np.float16); np.save(c + '_lm.tmp.npy', lm); os.replace(c + '_lm.tmp.npy', c + '_lm.npy')
        s.z = np.load(c + '_z.npy', mmap_mode='r'); s.lm = np.load(c + '_lm.npy').astype(f32); s.V = world_city.Veil(s.cam, 2400, 1350, seed=3, n=72, q=2); s.V.w, s.V.h = W, H
        r = np.random.default_rng(8); P = world_city.walkers(r, 470); k = np.hypot(*(P - s.cam.eye[:2]).T) > 30; P = P[k]; pr, zc = s.cam.project(np.c_[P, np.full(len(P), .9)]); zz = np.asarray(s.z)[np.clip(pr[:, 1].astype(int), 0, 1349), np.clip(pr[:, 0].astype(int), 0, 2399)].astype(f32); P = P[zz > zc - 2.2]; s.P = P; n = len(P)
        s.o = np.argsort(-P[:, 1]); d = np.hypot(P[:, 0] + 2, P[:, 1] - 6); s.tw = 15 + d * .30 + r.normal(0, 2.2, n); s.dir = r.uniform(0, 6.28, n); s.sp = r.uniform(.035, .075, n); s.ph = r.uniform(0, 6.28, n); s.hh = r.normal(1.68, .09, n)
        s.col = r.integers(0, 8, n); s.col2 = r.integers(0, 6, n); s.umb = r.random(n) < .075; s.phone = (r.random(n) < .17) & ~s.umb; s.tp = s.tw + r.uniform(6, 20, n); s.back = r.random(n) < .25; s.off = 60 + r.uniform(0, 4.5, n); s.bag = r.random(n) < .35
        s.x, s.y = grid(H, W); s.hx, s.hy = grid(H // 2, W // 2); s.vt = cv2.resize(s.V.tex, (W // 2, H // 2)); s.vw = cv2.resize(s.V.wob, (W // 2, H // 2))

    def veil(s, f, c):
        t = .24 + .78 * np.clip((f - 0) / 54., 0, 1) ** 1.22; V = s.V; q = 2; X = (s.hx * q + .5 - W / 2) / c['zoom'] + c['cx']; Y = (s.hy * q + .5 - H / 2) / c['zoom'] + c['cy']; cam = s.cam
        d = np.stack([(X - cam.cx) / cam.f, (Y - cam.cy) / cam.f, np.ones_like(X)], -1) @ cam.R.astype(f32); d /= np.linalg.norm(d, axis=-1, keepdims=True); V.az = np.arctan2(d[..., 0], d[..., 1]).astype(f32); V.el = np.arcsin(np.clip(d[..., 2], -1, 1)).astype(f32); V.tex, V.wob = s.vt, s.vw
        V.q = 2; L = V.layer(t, rim='#4a2a86', film=.5); return L

    def light_state(s, f):
        k = sum(f >= q for q in CUTS); fl = [q for q in CUTS if f == q + 1]
        return (k, None, 0.) if not fl else (k, k - 1, .62)                # one frame later it comes back, weaker, then is gone

    def people(s, f, c, k):
        cam = s.cam; cel = Cel(W, H, ss=2); g = Cel(W, H, ss=1); f2 = f // 2 * 2; tw = np.minimum(f2, s.tw); stop = f2 >= s.tw
        pos = s.P + np.c_[np.cos(s.dir), np.sin(s.dir)] * (s.sp * (tw - s.tw))[:, None] * 1.0
        pr, zc = cam.project(np.c_[pos, np.full(len(pos), .0)]); kz = s.zref / np.maximum(zc, 1.); X = (pr[:, 0] + c['dx'] * kz - c['cx']) * c['zoom'] + W / 2; Y = (pr[:, 1] + c['dy'] * kz - c['cy']) * c['zoom'] + H / 2; S = cam.f / zc * c['zoom']
        lm = s.lm[k]; ix = np.clip((pr[:, 0] / 8).astype(int), 0, 299); iy = np.clip((pr[:, 1] / 8).astype(int), 0, 168); amb = lm[iy, ix]
        pal = [rgb(q) for q in ('#2a2c3c', '#3a3440', '#1e2230', '#4a3a3c', '#2c3a4a', '#6a5a52', '#8f8478', '#5a2c38')]; pal2 = [rgb(q) for q in ('#20222e', '#2c2c36', '#1a1c26', '#34303a', '#24303c', '#3c3430')]
        for i in s.o:
            x, y, h = X[i], Y[i], S[i] * s.hh[i]
            if not (-20 < x < W + 20 and -30 < y < H + 40) or zc[i] < 5: continue
            a = amb[i] * 1.75 + np.array([.03, .03, .06], f32)
            if k == 4: dr_ = math.hypot(pos[i, 0] - RED[0], pos[i, 1] - RED[1]); a = a * .45 + np.array([.75, .05, .04], f32) * math.exp(-dr_ / 9.)
            elif k == 3: a = a * .8
            a = a.astype(f32); ci = pal[s.col[i]] * a * 1.6; c2_ = pal2[s.col2[i]] * a * 1.5; up = float(np.clip((f2 - s.tw[i]) / 4., 0, 1)); st = 0. if stop[i] else math.sin(f2 * .55 + s.ph[i]); w = h * .155
            cel.fill(np.array([(x - w * 1.0, y + h * .02), (x, y - h * .01), (x + w * 1.1, y + h * .02), (x, y + h * .055)]), '#05050b', .55)                                     # contact
            cel.fill(np.array([(x - w * .55, y - h * .46), (x + w * .1, y - h * .46), (x + w * (.05 - .55 * st), y), (x - w * (.35 + .55 * st), y)]), c2_); cel.fill(np.array([(x - w * .1, y - h * .46), (x + w * .55, y - h * .46), (x + w * (.35 + .55 * st), y), (x - w * (.05 - .55 * st), y)]), c2_ * .85)
            cel.fill(np.array([(x - w * .85, y - h * .40), (x - w * .95, y - h * .76), (x - w * .35, y - h * .86), (x + w * .35, y - h * .86), (x + w * .95, y - h * .76), (x + w * .85, y - h * .40)]), ci); cel.fill(np.array([(x - w * .95, y - h * .76), (x - w * .35, y - h * .86), (x + w * .35, y - h * .86), (x + w * .95, y - h * .76), (x, y - h * .78)]), np.minimum(ci * 1.55, 1))
            hr = h * .075; hx, hy = x, y - h * (.925 + .0 * up); hair = np.array([.07, .06, .08], f32) * (.5 + 1.6 * a); skin = np.array([.84, .66, .56], f32) * np.minimum(a * 1.5 + .02, 1.1)
            th = np.linspace(0, 6.283, 9)[:-1]; cel.fill(np.c_[hx + np.cos(th) * hr, hy + np.sin(th) * hr * 1.18], hair)
            if up > 0 and not s.back[i]: cel.fill(np.c_[hx + np.cos(th) * hr * .80, hy + hr * (.35 - .42 * up) + np.sin(th) * hr * .92 * (.35 + .65 * up)], skin)
            if s.bag[i]: cel.fill(np.array([(x + w * .9, y - h * .42), (x + w * 1.35, y - h * .42), (x + w * 1.4, y - h * .22), (x + w * .9, y - h * .22)]), c2_ * 1.3)
            if s.umb[i]:
                ub = [np.array([.62, .68, .78], f32), np.array([.16, .16, .22], f32), np.array([.55, .16, .2], f32)][i % 3] * a * 1.7; ux, uy = x, y - h * 1.12; q = h * .36; m = np.linspace(math.pi, 2 * math.pi, 9); tilt = -.35 * up
                dome = np.c_[ux + np.cos(m) * q, uy + np.sin(m) * q * .5 + (np.cos(m)) * q * tilt * .4]; cel.fill(np.vstack([dome, [[ux + q * .5, uy + q * .10], [ux, uy + q * .04], [ux - q * .5, uy + q * .10]]]), ub); cel.fill(np.array([dome[2], dome[4], [ux, uy + q * .04]]), np.minimum(ub * 1.25, 1))
            if s.phone[i] and s.tp[i] < f2 < s.off[i]:
                px, py = x + w * .75, y - h * 1.02; cel.fill(np.array([(x + w * .6, y - h * .74), (x + w * .95, y - h * .74), (px + w * .2, py + h * .02), (px - w * .15, py + h * .02)]), ci * .9); cel.fill(np.array([(px - w * .34, py - h * .075), (px + w * .34, py - h * .075), (px + w * .34, py + h * .035), (px - w * .34, py + h * .035)]), '#e8f6ff'); cv_ = np.c_[px + np.cos(th) * h * .13, py + np.sin(th) * h * .13]; g.fill(cv_, '#8fd0ff', .55)
        return cel.layer(), g.rgba()[..., 3]

    def frame(s, f):
        c = cam_at(f); cam = s.cam; zn = s.z; s.zref = 95.
        X = (s.x - W / 2) / c['zoom'] + c['cx']; Y = (s.y - H / 2) / c['zoom'] + c['cy']
        Z = cv2.remap(zn.astype(f32), X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE); k = s.zref / np.maximum(Z, 8.); mx, my = X - c['dx'] * k, Y - c['dy'] * k
        st, pv, am = s.light_state(f); im = s.L[st].f('img')
        if pv is not None: im = im * (1 - am) + s.L[pv].f('img') * am
        img = cv2.remap(im, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101); del im
        sky = cv2.remap(s.L[0].f('m_sky'), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        if st == 4:                                                            # nothing but the signal: crush the rest, keep the red
            red = smooth(.06, .22, img[..., 0] - np.maximum(img[..., 1], img[..., 2]) * 1.15)[..., None]; q = ease((f - 63) / 4.); img = img * (1 - (1 - red) * (.20 + .14 * q)); img[..., 2] *= 1.06
        vt = np.clip(f / 54., 0, 1); img = img * (1 - .16 * vt)
        V = s.veil(f, c); m = sky[..., None]; img = img * (1 - m * V[..., 3:]) + V[..., :3] * m
        L, gl = s.people(f, c, st); img = over(img, L)
        if gl.any(): img = img + rgb('#9fd6ff') * (cv2.GaussianBlur(gl, (0, 0), 5) * .9)[..., None]
        # the fall of the ink shows on the wet street: a dark film creeping toward us, violet at its lip
        if 20 <= f:
            yy = 560 + 560 * np.clip((f - 20) / 44., 0, 1) ** 1.4; gm = smooth(yy + 60, yy - 160, s.y) * .20 * (1 - sky); img = img * (1 - gm)[..., None] + rgb('#120a26') * (gm * .35)[..., None]
        if f in CUTS: img = img * .90
        # red signal: breathing halo and its streak on the road
        p, z = cam.project(RED[None]); kz = s.zref / float(z[0]); rx = (p[0, 0] + c['dx'] * kz - c['cx']) * c['zoom'] + W / 2; ry = (p[0, 1] + c['dy'] * kz - c['cy']) * c['zoom'] + H / 2
        if f >= 54:
            a = ease((f - 60) / 6.) * (1. if (f % 18) < 17 else .55); d = np.hypot(s.x - rx, s.y - ry); img = img + rgb('#ff2a1e') * ((np.exp(-d / 26.) * .60 + np.exp(-d / 150.) * .26) * a)[..., None]; st_ = np.exp(-((s.x - rx) / 13.) ** 2) * smooth(ry + 60, ry + 90, s.y) * np.exp(-np.maximum(0, s.y - ry - 70) / 90.) * (.6 + .4 * np.sin(s.y * .31 + 1.2 * np.sin(s.x * .05 + f * .12))); img = img + rgb('#e01c1c') * (st_ * .42 * a)[..., None]
        return img.astype(f32)
