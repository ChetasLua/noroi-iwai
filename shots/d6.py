"""d6: Unlimited Void.  0-17 the fingers cross; 18-19 two impact frames; 20-53 falling past him toward the black sun
(geodesics of a real Schwarzschild hole, drawn as broken engraved lines); 54-71 everything stops, only the ring breathes."""
import numpy as np, cv2, math, os
from numba import njit, prange
from jj import W, H as HH, comp, kit, fx, hands as HD
from jj.paint import rgb, f32, smooth, fbm, grid, lum, Canvas, arcs
from jj.ink import Cel, spl, resample, noise1
from jj.cam import Camera3D
from chars.gojo import Gojo, gojo_head, PAL, GC, GF, rim, eye_glow
from chars import gojo_keys as GK
from chars.gojo_cache import Plates

VER = 'd7_v2'
MW, MH = 2688, 1512                 # the sun is computed on a larger field (the camera falls in: zoom .72 -> 1)
SUN = (1236., 432.)                 # sun centre in the frame
MC = (MW / 2 + (SUN[0] - W / 2), MH / 2 + (SUN[1] - HH / 2))
LEV = dict(A=78., B=34.)            # two fields of the same hole (px per Schwarzschild radius); the frame samples the fields, never an image, so the lines stay sharp at any size
PXU0, PXU1 = 31., 146.              # on screen: the shadow grows from 15 % to 70 % of the frame height


# ------------------------------------------------------------------------------------------------ geodesics
@njit(parallel=True, cache=True, fastmath=True)
def _trace(h, w, cx, cy, pxu, dist, inc, rot, r_in, r_out, R1, P1, R2, P2, SH):
    ci, si = math.cos(inc), math.sin(inc); cr, sr = math.cos(rot), math.sin(rot)
    for j in prange(h):
        for i in range(w):
            bx = (i + .5 - cx) / pxu; by = -(j + .5 - cy) / pxu
            ux = bx * cr - by * sr; uy = bx * sr + by * cr
            # camera at distance dist on the -y' axis, orthographic-ish bundle: start far away with impact (ux, uy)
            # world: disc in the xy plane, z up.  view direction d, right e, up u
            dx, dy, dz = 0., ci, -si
            ex, ey, ez = 1., 0., 0.
            upx, upy, upz = 0., si, ci
            px = -dx * dist + ex * ux + upx * uy; py = -dy * dist + ey * ux + upy * uy; pz = -dz * dist + ez * ux + upz * uy
            vx, vy, vz = dx, dy, dz
            hx = py * vz - pz * vy; hy = pz * vx - px * vz; hz = px * vy - py * vx; h2 = hx * hx + hy * hy + hz * hz
            hit = 0; R1[j, i] = 0; R2[j, i] = 0; SH[j, i] = 0
            for s in range(420):
                r2 = px * px + py * py + pz * pz; r = math.sqrt(r2)
                if r < 1.0: SH[j, i] = 1; break
                if r > dist * 1.15 and px * vx + py * vy + pz * vz > 0: break
                dt = .035 * r if r > 6 else (.06 if r > 2.2 else .03)
                if dt > .7: dt = .7
                a = -1.5 * h2 / (r2 * r2 * r)
                vx += a * px * dt * .5; vy += a * py * dt * .5; vz += a * pz * dt * .5
                nx, ny, nz = px + vx * dt, py + vy * dt, pz + vz * dt
                if (pz > 0) != (nz > 0):
                    t = pz / (pz - nz + 1e-12); qx = px + (nx - px) * t; qy = py + (ny - py) * t; q = math.sqrt(qx * qx + qy * qy)
                    if r_in < q < r_out:
                        if hit == 0: R1[j, i] = q; P1[j, i] = math.atan2(qy, qx); hit = 1
                        elif hit == 1: R2[j, i] = q; P2[j, i] = math.atan2(qy, qx); hit = 2
                px, py, pz = nx, ny, nz
                r2 = px * px + py * py + pz * pz; r = math.sqrt(r2); a = -1.5 * h2 / (r2 * r2 * r)
                vx += a * px * dt * .5; vy += a * py * dt * .5; vz += a * pz * dt * .5


@njit(cache=True, fastmath=True)
def _h(i, s):
    n = (int(i) * 374761393 + int(s) * 668265263) & 0xffffffff; n = ((n ^ (n >> 13)) * 1274126177) & 0xffffffff
    return ((n ^ (n >> 16)) & 0xffff) / 65535.0


@njit(parallel=True, cache=True, fastmath=True)
def _rings(R, P, G, t, nl, r_in, r_out, gain, out, breath):
    h, w = R.shape
    for j in prange(h):
        for i in range(w):
            r = R[j, i]
            if r <= 0: continue
            u = ((r - r_in) / (r_out - r_in)) ** .72; x = u * nl; k = int(x); f = x - k - .5
            dx = G[j, i]
            if dx > .9: 
                # lines too dense to resolve: their mean tone
                out[j, i] += gain * .30 * (1 - u); continue
            jit = (_h(k, 3) - .5) * .5; wd = .9 + 4.6 * _h(k, 5) ** 3.2
            cov = wd * .5 - abs(f - jit) / dx + .5
            dop = 1 + .58 * math.cos(P[j, i] + .4)
            out[j, i] += gain * .030 * dop * dop * (1.1 - u)
            if cov <= 0: continue
            if cov > 1: cov = 1.
            om = .55 * (r_in / r) ** 1.5
            ph = P[j, i] + om * t + 6.2832 * _h(k, 7)
            m = 1 + int(_h(k, 9) * 4); s = ph * m / 6.2831853; s = s - math.floor(s); duty = .55 + .43 * _h(k, 11)
            if s > duty: continue
            tp = min(1., s / .10) * min(1., (duty - s) / .16)
            cov *= .25 + .75 * tp
            b = (.30 + .85 * _h(k, 13) ** 1.8) * dop * dop * (1.12 - .86 * u) * .95
            out[j, i] += gain * cov * b * breath
            if out[j, i] > 2.2: out[j, i] = 2.2


class Shot:
    id = 'd6'; n = 72
    finish = dict(bloom=.22, vig=.14, halation='#d6e6ff', soft=.08, grain=.011)
    EVENTS = [(0, 'hand_raise', dict(size=.3, note='fingers up, cloth')), (6, 'finger_slide', dict(end=15, note='skin on skin, two fingers cross')), (15, 'lock', dict(size=.7, note='fingers lock: a single low bell')),
              (18, 'impact', dict(size=1.0, note='domain: world inverts')), (19, 'impact', dict(size=.8, note='second inversion')), (20, 'void_open', dict(end=53, note='vast low drone, wind of falling, thousands of thin glass threads')),
              (20, 'fall', dict(end=53, note='falling, rises to full')), (54, 'stop', dict(size=1.0, note='everything stops dead: cut all sound')), (54, 'silence', dict(end=71, note='only a breath of the ring: near-inaudible sine, 1 cycle'))]

    # ---------------------------------------------------------------- plates
    def setup(self):
        self._memo = {}; self.A = Plates(VER, self._build)
        self._streams()
        self._hands_cache()

    def _build(self):
        A = {}
        for lv, pxu in LEV.items():
            R1, P1, R2, P2 = [np.zeros((MH, MW), f32) for _ in range(4)]; SH = np.zeros((MH, MW), np.uint8)
            _trace(MH, MW, MC[0], MC[1], pxu, 26., .165, -.30, 2.75, 13.5, R1, P1, R2, P2, SH)
            for k, R, P in (('1', R1, P1), ('2', R2, P2)):
                v = (R > 0).astype(f32); A[lv + 'R' + k] = R.view(np.int32); A[lv + 'c' + k] = (np.cos(P) * v).view(np.int32) if False else np.ascontiguousarray((np.cos(P) * v).astype(f32)).view(np.int32); A[lv + 's' + k] = np.ascontiguousarray((np.sin(P) * v).astype(f32)).view(np.int32)
            A[lv + 'S'] = SH
        A['bg'] = self._bg().astype(np.float16); A['close'] = self._close_bg().astype(np.float16)
        for k, v in self._figures().items(): A[k] = v.astype(np.float16)
        return A

    def _bg(self):
        """painted nothing: almost black, strokes turning round the sun, one cold rift"""
        r = np.random.default_rng(5); x, y = grid(MH, MW); cx, cy = MC; d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2); n1 = fbm(MH, MW, (620, 260, 100), None, 11); n2 = fbm(MH, MW, (460, 190), None, 12)
        base = np.zeros((MH, MW, 3), f32) + rgb('#02030a'); base = base + (np.exp(-(d / 780.) ** 1.3) * .9)[..., None] * rgb('#0b1440') + np.clip(n1 - .4, 0, 2)[..., None] * rgb('#0a0c2e') * .35 + np.clip(n2 - .9, 0, 2)[..., None] * rgb('#1a0f3c') * .35
        cv = Canvas(MH, MW, img=base)
        for hw, n in ((40, 900), (18, 2000), (8, 3600)):
            dd = 260 + 1700 * r.random(n) ** .8; a = r.uniform(0, 6.283, n); px, py = cx + np.cos(a) * dd, cy + np.sin(a) * dd * .92; ok = (px > 0) & (px < MW) & (py > 0) & (py < MH); px, py, a, dd = px[ok], py[ok], a[ok], dd[ok]; col = base[py.astype(int), px.astype(int)] * r.uniform(.55, 1.5, (len(px), 1))
            cv.strokes(arcs(px, py, a + np.pi / 2 + .4 + r.normal(0, .12, len(px)), hw * r.uniform(5, 11, len(px)), hw * 7 / np.maximum(dd, 90)), hw * r.uniform(.5, 1.1, len(px)), np.clip(col, 0, 1).astype(f32), op=.5, dry=.4, seed=hw)
        return np.clip(cv.img, 0, 1)

    def _close_bg(self):
        """behind the fingers: his bare face, far out of focus, the two irises the only colour"""
        pal = dict(PAL['blue'], skin='#31427e', skin_sh='#141c44', hair='#6f86c8', hair_sh='#2c3a78', hair_bk='#1d2858', hair_root='#26346c', hair_root2='#4a5ca0', white='#8fa6e0', coat_lt='#16224e', lit2='#1b2654')
        L = Gojo.bust(1330, 470, 430, -.12, -.06, -.05, 'wind', dict(band=False, mouth='flat', lid='cold', brow='cold', glow=.5, pupil=.22), pal, key='BL', body=(-.2, 0, 0), seed=3)
        img = np.zeros((HH, W, 3), f32) + rgb('#03040f'); x, y = grid(); img = img + np.clip(1 - np.sqrt(((x - 1330) / 1100) ** 2 + ((y - 440) / 760) ** 2), 0, 1)[..., None] ** 1.5 * rgb('#0e1c54')
        img = comp.over(img, L); H = gojo_head(True).pose(1330, 470, 430, -.12, -.06, -.05)
        for sd in (-1, 1): q = H.P([(sd * GF.EC, GF.EY)])[0]; img = img + comp.radial(q[0], q[1], 46, 40, power=1.2)[..., None] * rgb('#7fe4ff') * .75
        return comp.bokeh(comp.blur(img, 2), 10, 0)

    def _figures(self):
        """the man in the void, falling: ten drawings of one slow turn, coat and hair streaming up"""
        out = {}
        from jj.figure import Fig
        def head_fn(cel, H, info):
            Hh = gojo_head(True).pose(H.cx, H.cy, H.s * 1.08, H.yaw, H.pitch, H.roll, H.persp); Gojo.head(cel, Hh, 'wind', dict(band=False, mouth='flat', lid='calm', brow='calm', pupil=.24, glow=.4), 'night', 0, (0., .10), 3, neck=False, key='L'); return cel
        cam = Camera3D((0., -6.2, 1.0), (0., 0., .98), 1700., W, HH)
        for k in range(10):
            u = k / 9.
            kw = dict(root=(0, 0, .02), pel=(-10 + 8 * math.sin(k), 8 * math.sin(k * .9), 10 * math.sin(k * .7)), wst=(4, 5 * math.sin(k * 1.1), 0), cst=(-8, -6 * math.sin(k * .8), 6 * math.sin(k * .6)), nck=(-6, 0, 0), hed=(-8, 4 * math.sin(k), 0),
                      hL=(.84 - .1 * u, .10 + .2 * math.sin(k * .6), -.06 + .16 * math.sin(k * .9)), hR=(.80, .16 - .2 * u, .10 + .18 * math.sin(k * .5 + 1)), eL=(.3, -.6, -.6), eR=(.3, -.6, -.6), gL='open', gR='open', wL=(20, -10, 0), wR=(20, -10, 0),
                      fL=(.05, -.10 - .06 * math.sin(k * .5), .10), fR=(.09, .10 + .06 * math.sin(k * .7), .22 + .06 * math.sin(k)), kL=(.15, 1, 0), kR=(.4, 1, .1), tL=(6, -58, -16), tR=(10, -52, -16))
            F = Fig('gojo').pose('float', **kw).place(0, 0, 0, heading=math.pi - .35 + 2.5 * u).blow(.5, 0, 3.6); L = F.draw(cam, k * 3, light=(.6, .6, .5), head_fn=head_fn)
            hi = F.info['head']; pv = F.info['pelvis']; S_ = 760; cx_, cy_ = int(pv[0]), int(pv[1]); lay = np.zeros((S_, S_, 4), f32); x0, y0 = cx_ - S_ // 2, cy_ - S_ // 2; xs, ys = max(0, x0), max(0, y0); xe, ye = min(W, x0 + S_), min(HH, y0 + S_); lay[ys - y0:ye - y0, xs - x0:xe - x0] = L[ys:ye, xs:xe]
            out['fig%d' % k] = lay; out['fh%d' % k] = np.array([hi['x'] - x0, hi['y'] - y0], f32)
        return out

    def _streams(self):
        """rivers of light: bundles of hair-thin threads in 3-D, from beside the camera into the sun"""
        self.riv = []
        for k, (P, w0, w1, n, seed) in enumerate((
            ([(-420, 1180), (120, 1010), (520, 900), (860, 800), (1040, 690), (1060, 560), (1010, 470)], 560, 5, 620, 31),
            ([(-300, 90), (60, 170), (330, 260), (620, 330), (850, 372), (980, 404), (1050, 440)], 230, 4, 300, 32),
            ([(2320, 1240), (2060, 1010), (1840, 840), (1640, 720), (1510, 620), (1440, 520), (1390, 440)], 420, 5, 430, 33),
            ([(2300, -160), (2020, 40), (1800, 150), (1620, 240), (1500, 310), (1420, 366), (1370, 404)], 160, 3, 200, 34))):
            C = spl(np.array(P, float), n=14); C = resample(C, 14.); T = np.gradient(C, axis=0); T /= np.hypot(*T.T)[:, None]; N = np.c_[-T[:, 1], T[:, 0]]; t = np.linspace(0, 1, len(C)); W_ = w0 * (1 - t) ** 1.5 + w1
            r = np.random.default_rng(seed); th = []
            for i in range(n):
                q = r.normal(0, .42); q = np.clip(q, -1.1, 1.1); cur = q + r.normal(0, .10) * np.sin(t * r.uniform(2, 7) + r.uniform(0, 6.28)) + r.normal(0, .25) * t
                th.append((C + N * (cur * W_)[:, None], r.random() ** 2.4, r.random(), r.uniform(.6, 1.5)))
            self.riv.append(th)
        r = np.random.default_rng(22); self.stars = np.c_[r.uniform(-1, 1, 900), r.uniform(-1, 1, 900), r.uniform(.15, 1, 900), r.random(900)]
        a = r.uniform(0, 6.283, 1500); d = .30 + 2.6 * r.random(1500) ** .8; self.dust = np.c_[np.cos(a) * d * 1.25, np.sin(a) * d * .85, r.uniform(0, 1, 1500), r.random(1500)]
        a = r.uniform(0, 6.283, 110); d = .45 + 2.0 * r.random(110); self.gly = np.c_[np.cos(a) * d * 1.3, np.sin(a) * d * .85, r.uniform(0, 1, 110), r.random(110), r.uniform(0, 6.283, 110), r.uniform(-1, 1, 110)]

    def _hands_cache(self): pass

    # ---------------------------------------------------------------- 0-17 the fingers
    def fingers(self, f):
        x, y = grid(); u = float(comp.lerp_keys(f, [(0, 0.), (4, .04), (12, .88), (14, 1.06), (16, 1.), (18, 1.)]))
        rise = float(comp.lerp_keys(f, [(0, 150.), (4, 0.), (18, -14.)])); fd = f // 2 * 2 if f not in (13, 15) else f
        key = ('h', round(u, 2))
        if key not in self._memo:
            A, B = HD.POSES['two'], dict(HD.POSES['cross']); pose = HD.lerp_pose(A, B, u)
            c = Cel(); p = PAL['blue']; hp = dict(skin='#a9c4f0', shade='#263a78', line='#05070f', nail='#cfe4ff', nail_shade='#3d548f', light='#f4fdff')
            GC.sleeve(c, (560, 2300), (640, 1960), (700, 1290), 420, 360, 250, dict(p), 520, seed=3, lit=1, hand_fn=lambda: HD.draw(c, pose, HD.Cam(742, 800, 500, 4.6), pal=hp, light=(.75, .5, .35), w=9.5, seed=4, side='R', R=HD.euler(158, 4, 2), bld=HD.SLIM, res=640))
            L = c.layer(); L = rim(L, -10, 3, '#e9f8ff', .95, soft=.7); L = rim(L, 7, 2, '#2f6cff', .75, soft=1.); L = kit.para(L, (300, 1080), (1000, 100), [(0, '#2a3470'), (.5, '#b9c8f0'), (1, '#ffffff')], 'mul')
            [self._memo.pop(q) for q in [q for q in self._memo if isinstance(q, tuple)]]
            self._memo[key] = L
        L = self._memo[key]
        L = cv2.warpAffine(L, np.float32([[1, 0, 0], [0, 1, rise]]), (W, HH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE if rise < 0 else cv2.BORDER_CONSTANT)
        img = self.A['close']; img = comp.cam2d(img, 960 + .9 * f, 540, 1.05 + .004 * f)
        img = img * (1 - .55 * comp.blur(np.roll(L[..., 3], 40, 1), 34))[..., None]
        img = comp.over(img, L)
        if f >= 13:                              # where the two fingers touch, a hair of white light
            t = f - 13; px, py = 792, 250 + rise; k = min(1., t / 3.)
            gl = np.exp(-np.abs(y - py - (x - px) * .04) / (1.6 + .3 * t)) * np.exp(-np.abs(x - px) / (90 + 260 * t)); img = img + gl[..., None] * rgb('#e9f8ff') * 1.3 * k + comp.radial(px, py, 60 + 40 * t, power=2.4)[..., None] * rgb('#bfe8ff') * .9 * k
        d = kit.shake(f, .5 if f < 13 else 2.2, 4); return comp.cam2d(img, W / 2 + d[0], HH / 2 + d[1], 1.01 + (.02 * (f - 13) if f > 13 else 0))

    # ---------------------------------------------------------------- the sun
    def sun(self, pxu, t, breath=1.):
        lv = 'A' if pxu > 64 else 'B'; k = pxu / LEV[lv]; M = np.float32([[k, 0, SUN[0] - k * MC[0]], [0, k, SUN[1] - k * MC[1]]]); A = self.A
        wp = lambda a: cv2.warpAffine(a, M, (W, HH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT); fl = lambda n: A.m[n].view(f32) if hasattr(A.m[n], 'view') else A.m[n]
        out = np.zeros((HH, W), f32); near = None
        for q, nl_gain in (('2', .9), ('1', 1.1)):
            R0 = np.asarray(fl(lv + 'R' + q)); v0 = (R0 > 0).astype(f32); v = wp(v0); ok = v > .5; dv = np.maximum(v, 1e-3); R = np.where(ok, wp(R0) / dv, 0).astype(f32); P = np.arctan2(wp(np.asarray(fl(lv + 's' + q))), wp(np.asarray(fl(lv + 'c' + q)))).astype(f32)
            u = np.where(R > 0, ((np.maximum(R, 2.75) - 2.75) / 10.75) ** .72 * 46, 0).astype(f32); gx = cv2.Sobel(u, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8.); gy = cv2.Sobel(u, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8.); G = np.maximum(np.hypot(gx, gy), 1e-3).astype(f32)
            o = np.zeros((HH, W), f32); _rings(R, P, G, t, 46, 2.75, 13.5, nl_gain, o, breath); out = o if q == '2' else np.where(R > 0, o + out * .35, out)
        return out, wp(A['%sS' % lv].astype(f32))

    def void(self, f, br=1.):
        """20..71; time stops at 54.  We fall: the sun opens from 15 % to 70 % of the height, the streams tear past the lens"""
        ff = min(f, 54.); e = (ff - 20) / 34.; ac = e ** 1.5; tt = (ff - 20) / 24.; pxu = math.exp(math.log(PXU0) * (1 - ac) + math.log(PXU1) * ac); grow = pxu / PXU0; moving = f < 54
        cx, cy = SUN; Rs = 2.6 * pxu; x, y = grid(); dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        ring, shadow = self.sun(pxu, tt * 1.5, br)
        z_ = .74 + .30 * ac; bg = self.A['bg']; img = cv2.warpAffine(bg, np.float32([[z_, 0, SUN[0] - z_ * MC[0]], [0, z_, SUN[1] - z_ * MC[1]]]), (W, HH), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT_101)
        add = np.zeros((HH * 2, W * 2), np.uint8); bright = np.zeros((HH * 2, W * 2), np.uint8); pul = np.zeros((HH * 2, W * 2), np.uint8)
        for sx, sy, z, q in self.stars:
            px, py = sx * 1500, sy * 950; d = math.hypot(px, py) + 1e-3; d2 = d + (Rs * 1.15) ** 2 / d * .9; X, Y = cx + px / d * d2, cy + py / d * d2
            if 2 < X < W - 2 and 2 < Y < HH - 2: cv2.circle(add, (int(X * 32), int(Y * 32)), int((.6 + 1.5 * q ** 6) * 16), int(60 + 170 * q ** 3), -1, cv2.LINE_AA, 4)
        # rivers: they widen past us as we fall; pulses run in toward the sun
        lay = [np.zeros((HH * 2, W * 2), np.uint8) for _ in range(4)]; zc = np.array([cx, cy]); Sc = .92 * grow ** .46; sh_ = .46 * tt * (1 + e)
        for k, th in enumerate(self.riv):
            for P, g, ph, sp in th:
                Q = zc + (P - zc) * Sc; q8 = np.round(Q * 32).astype(np.int32); cv2.polylines(lay[k], [q8], False, int(9 + 46 * g), 1, cv2.LINE_AA, 4); n = len(q8); p0 = (ph + sh_ * sp) % 1.25; L_ = .05 + .10 * g + (.10 * e if moving else 0); i0, i1 = max(0, int((p0 - L_) * n)), min(n, int(p0 * n))
                if i1 - i0 > 1: cv2.polylines(pul, [q8[i0:i1]], False, int(50 + 190 * g), 1, cv2.LINE_AA, 4)
        # the fall itself: dust of light in depth, streaked by the shutter, dead points when time stops
        near = np.zeros((HH * 2, W * 2), np.uint8); sp_ = .55 * tt + 1.25 * tt * e; dz = (.020 + .075 * e) if moving else .0
        for X, Y, z0, q in self.dust:
            z = ((z0 - sp_ * (.5 + .5 * q)) % 1.) * 3.2 + .10; x1, y1 = cx + X / z * 620, cy + Y / z * 620; z2 = z * (1 + dz * 9) + dz; x0_, y0_ = cx + X / z2 * 620, cy + Y / z2 * 620
            if math.hypot(x1 - cx, y1 - cy) < Rs * 1.06: continue
            if not (-400 < x1 < W + 400 and -400 < y1 < HH + 400): continue
            w_ = 1 if z > .8 else 2 if z > .34 else 4; b_ = int(min(255, (34 + 190 * q ** 2) * min(1., 1.5 / z) ** .5)); tgt = near if z < .34 else bright
            if moving: cv2.line(tgt, (int(x0_ * 32), int(y0_ * 32)), (int(x1 * 32), int(y1 * 32)), b_, w_, cv2.LINE_AA, 4)
            else: cv2.circle(tgt, (int(x1 * 32), int(y1 * 32)), int((.7 + .5 * w_) * 16), b_, -1, cv2.LINE_AA, 4)
        dn = lambda a: cv2.resize(a, (W, HH), interpolation=cv2.INTER_AREA).astype(f32) / 255.
        img = img + dn(add)[..., None] * rgb('#cfe0ff') * .8
        lines = sum(dn(l)[..., None] * rgb(c_) for l, c_ in zip(lay, ('#bcd8ff', '#d9cfff', '#e6f2ff', '#a9c4ff'))); clear = np.minimum(smooth(Rs * 1.03, Rs * 1.03 + 160, dist), smooth(205. * Sc, 205. * Sc * 2.3, dist)); behind = 1 - np.clip(ring * 1.2, 0, .85)
        img = img + ((lines * 1.05 + dn(pul)[..., None] * rgb('#f4f9ff') * .95) * (clear * behind)[..., None] + dn(bright)[..., None] * rgb('#f4f9ff') * (.95 * smooth(Rs * 1.01, Rs * 1.12, dist) * behind)[..., None])
        # --- the hole, exactly as drawn before
        img = img * (1 - cv2.GaussianBlur(shadow, (0, 0), .7) * .995)[..., None]; haze = cv2.GaussianBlur(ring, (0, 0), 22); img = img + haze[..., None] * rgb('#4a78e0') * .16; img = img * (1 - np.clip(ring * .6, 0, .6))[..., None] + ring[..., None] * rgb('#eef5ff') * 1.05; img = img * (1 - (shadow * (1 - np.clip(ring * 3, 0, 1)))[..., None])
        c = Cel(); r = np.random.default_rng(11); kw = float(np.clip(Rs / 203., .42, 1.7))
        for R_, wd, n, gap, col in ((Rs * 1.008, 5.2, 5, .10, '#ffffff'), (Rs * 1.030, 2.2, 9, .16, '#e6f2ff'), (Rs * 1.058, 1.3, 13, .22, '#b9d2ff'), (Rs * 1.10, .9, 7, .5, '#8fb4ff')):
            a = r.uniform(0, 6.28)
            for i in range(n):
                sp = 6.2832 / n * r.uniform(.55, 1.25); g_ = gap * r.uniform(.4, 1.6); q = np.linspace(a, a + sp, max(12, int(sp * 40))); rr = R_ * (1 + .0025 * np.sin(q * 3 + i))
                c.line(np.c_[cx + np.cos(q) * rr, cy + np.sin(q) * rr], wd * kw * (.45 + .75 * r.random()) * (1.15 if math.cos(a + sp / 2 - math.pi) > 0 else .8), col, taper=(.22, .22), press=.25, seed=i + n * 31, tremor=.0, tmin=.06); a += sp + g_
        rl = c.layer(); img = comp.over(img, rl, min(1., .92 * br)); img = img + comp.blur(rl, 5)[..., :3] * .16 * br
        # fine splinters of unreadable writing, carried in the fall
        hp, px_, py_ = self._where(e); c = Cel()
        for X, Y, z0, q, a, w_ in self.gly:
            z = ((z0 - sp_ * (.4 + .4 * q)) % 1.) * 2.6 + .16; gx, gy = cx + X / z * 620, cy + Y / z * 620; s_ = (5 + 15 * q) / z ** .8
            if not (-60 < gx < W + 60 and -60 < gy < HH + 60) or s_ < 3 or s_ > 46 or math.hypot(gx - cx, gy - cy) < Rs * 1.25 or (abs(gx - px_) < 50 + hp * .5 and abs(gy - py_) < 50 + hp * .55): continue
            if moving: z2 = z * (1 + dz * 5); c.line(np.array([(gx, gy), (cx + X / ((z + z2) / 2) * 620, cy + Y / ((z + z2) / 2) * 620), (cx + X / z2 * 620, cy + Y / z2 * 620)]), max(.7, s_ * .06), '#9fc0ff', taper=(.05, .9), seed=int(q * 999), alpha=.5)
            fx.glyph(c, gx, gy, s_, a + w_ * tt, int(q * 99991), '#eaf3ff' if q > .4 else '#b9a6ff', (.30 + .6 * q) * min(1., (z - .12) * 4), w=max(.7, s_ * .07))
        L = c.layer(); img = comp.over(img, L)
        img = self._figure(img, f, e)
        nl = dn(near); nl = cv2.GaussianBlur(nl, (0, 0), 2.2 if moving else 1.4); img = img + (nl * smooth(Rs * 1.0, Rs * 1.1, dist))[..., None] * rgb('#dfeeff') * .9      # what passes closest is out of focus
        return img

    def _where(self, e):
        k = 1 - (1 - min(1., e * 1.04)) ** 2.2
        return 690 * (1 - k) ** 1.6 + 146, 430 + (560 - 430) * k + 60 * math.sin(k * 3.14), 640 * (1 - k) + 596 * k - 70 * math.sin(k * 3.14)

    def _figure(self, img, f, e):
        hpx, px, py = self._where(e); ff = min(f, 54); u = (ff - 20) / 34.; k = min(9, int(u * 9.99)); F = self.A['fig%d' % k]; S_ = F.shape[0]; sc = hpx / 600.; an = math.radians(-14 + 152 * u ** 1.15)
        ca, sa = math.cos(an) * sc, math.sin(an) * sc; M = np.float32([[ca, -sa, px - ca * S_ / 2 + sa * S_ / 2], [sa, ca, py - sa * S_ / 2 - ca * S_ / 2]])
        src = F if sc > .5 else cv2.GaussianBlur(F, (0, 0), .42 / max(sc, .05)); lay = cv2.warpAffine(src, M, (W, HH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        dark = lay.copy(); dark[..., :3] *= rgb('#39457e') * (1.0 - .55 * u); a = lay[..., 3]; w = max(1.5, 4.6 * sc); dark = rim(dark, -w, w * .7, '#ffffff', 1.0, soft=.35); dark = rim(dark, w * .7, w * .3, '#86aef8', .85, soft=.4); dark = rim(dark, 0, -w * .5, '#86aef8', .5, soft=.4)
        ha = comp.blur(a, 6); img = img * (1 - .55 * comp.blur(a, 18 * sc + 4))[..., None] + (ha * (1 - a))[..., None] * rgb('#3f74e8') * (.22 + .18 * u)
        return comp.over(img, dark)

    # ---------------------------------------------------------------- frame
    def frame(self, f):
        if f < 18: return self.fingers(f)
        if f == 18:
            self.fingers(17); L = self._memo[[q for q in self._memo if isinstance(q, tuple)][-1]]; a = L[..., 3:4]; g = lum(L[..., :3]) / np.maximum(L[..., 3], 1e-3); ink = smooth(.20, .24, g)[..., None] * 1.
            img = np.ones((HH, W, 3), f32) * rgb('#f8fbff'); c = Cel(); kit.focus_lines(c, 792, 250, 330, 1700, 190, 11, '#04050c', 3, ry=.9); img = comp.over(img, c.layer())
            img = img * (1 - a) + a * (rgb('#04050c') * ink + rgb('#f8fbff') * (1 - ink) * .0 + rgb('#8f9ad0') * (1 - ink) * 1.); return img
        if f == 19:                                # the iris of the world: black, white broken rings round the place the sun will be
            img = np.zeros((HH, W, 3), f32) + rgb('#020208'); c = Cel(); r = np.random.default_rng(4)
            for k in range(26):
                R = 150 + 46 * k * (1 + .02 * k); fx.ragged_ring(c, SUN[0], SUN[1], R, R, 2 + 13 * r.random() ** 3, 40 + k, '#f8fbff', gaps=.25 + .5 * r.random(), n=6 + k // 3, rag=.004, spurs=.0)
            kit.focus_lines(c, SUN[0], SUN[1], 620, 1800, 70, 5, '#f8fbff', 5, ry=1.); L = c.layer(); img = comp.over(img, L); cv2.circle(img, (int(SUN[0] * 16), int(SUN[1] * 16)), 150 * 16, (0, 0, 0), -1, cv2.LINE_AA, 4); return img
        if f > 54:                                 # time has stopped: only the ring breathes
            if 'stop' not in self._memo: self._memo['stop'] = (self.void(54), self.void(54, 1.42))
            A, B = self._memo['stop']; br = .5 - .5 * math.cos((f - 54) / 18. * 2 * math.pi); return A + (B - A) * br
        img = self.void(f)
        if f in (20, 21): img = img * (1.7 if f == 20 else 1.2)
        if f < 54: d = kit.shake(f, 1.2 * min(1, (f - 20) / 10.), 6); img = comp.cam2d(img, W / 2 + d[0], HH / 2 + d[1], 1.004)
        return img
