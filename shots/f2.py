"""f2: the Malevolent Shrine rises out of blood-black water; the ring; the mouths open; the air is cut."""
import os, math, numpy as np, cv2
from jj import W, H, comp, kit, world, world_shrine as WS
from jj.paint import rgb, f32, fbm, smooth, mix, grid, lum, Canvas, arcs
from jj.ink import Cel, spl
from jj.comp import ease, over, blur

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
KEY = 'f2_v2'; TOP = 640
NAMES = ('hall', 'gate0', 'gate1', 'gate2')
BONE, RED, BLK = '#e8dcc2', '#b81c24', '#070406'
KW = dict(w=1920, h=1080, focal=904., top=640)


def rise(f):
    """metres still under water: hall, far gate, mid gate, near gate.  heavy start, long push, stops dead at 27"""
    def e(a, b, d, p=1.35): u = np.clip((f - a) / float(b - a), 0, 1); return d * (1 - u ** p)
    return [e(-1, 27, 36.5, 1.25), e(2, 26, 50., 1.2), e(6, 27, 33., 1.3), e(10, 27, 17., 1.4)]


# the twelve cuts: (x0, y0, x1, y1, bow px, width px, drawing 0..2).  CA and CB are the ones that part the tiers, CH takes the horn
CA, CB, CH = (455, 476, 1435, 356), (500, 503, 1530, 606), (1168, 366, 1352, 668)
CUTS = [(*CA, -16, 6.5, 0), (140, 132, 1840, 700, 46, 5.0, 0), (1700, 70, 600, 1010, -60, 5.6, 0),
        (*CB, 14, 6.0, 1), (*CH, 9, 4.6, 1), (742, 24, 830, 1048, 34, 3.4, 1), (44, 560, 1010, 70, -50, 4.4, 1), (940, 1052, 1890, 452, 40, 5.2, 1),
        (1585, 1030, 1400, 40, -44, 3.6, 2), (150, 742, 1300, 930, -30, 4.2, 2), (1880, 884, 1120, 560, 26, 3.0, 2), (70, 40, 520, 700, 30, 3.8, 2)]
POLT = [(738, 120), (1098, 120), (1098, 397.5), (738, 441.4)]                  # top tier, down to CA
POLM = [(636, 280), (1206, 280), (1206, 470), (1152, 512), (1112, 565), (764, 530), (700, 512), (636, 470)]     # middle tier, down to CB
POLH = [(1254, 506), (1296, 452), (1352, 392), (1388, 388), (1388, 444), (1336, 506), (1286, 544), (1264, 530)]


class Shot:
    id = 'f2'; n = 72
    finish = dict(bloom=.16, vig=.20, soft=.08, grain=.016)
    EVENTS = [(0, 'sub', dict(size=.8, note='something enormous moving under water, rising 0-27')), (2, 'water', dict(size=.9, note='water breaking, sheeting off roofs, until 34')), (6, 'bone', dict(size=.5, note='bones knock, skulls surface 6-26, scattered')),
              (27, 'impact', dict(size=1., note='it stops dead: temple-drum THOOM, whole frame jolts')), (27, 'ripple', dict(size=.8, note='one ring of water crosses the frame 27-46')), (36, 'mouth', dict(size=.6, note='six pillar mouths tear open left to right, one per frame 36-41, wet')), (45, 'mouth', dict(size=.7, note='upper mouths open 45-52, ten of them; choir-like breath under')),
              (54, 'slash', dict(size=1., note='hair-thin cuts in rings: three drawings 54, 55, 56, silence between them is total')), (57, 'glass', dict(size=.4, note='the sky slides apart in slabs, no debris')), (58, 'stone', dict(size=.8, note='two roof tiers slide along their cuts, low grind accelerating, stop dead at 67')), (59, 'crack', dict(size=.4, note='a horn comes free')), (68, 'splash', dict(size=.8, note='the horn hits the water, one heavy splash, rings')), (70, 'silence', dict(note='terrible stillness to the cut'))]

    def setup(s):
        cv2.setNumThreads(2); c = os.path.join(ROOT, 'cache', KEY); s.c = c
        if not os.path.exists(c + '_ok.npy'):
            L = world.lazy('shrine', **KW); C2 = world.lazy('shrine_closed', top=640); cam = L.cam
            for nm in NAMES + ('closed',):
                R = (L if nm != 'closed' else C2).f('l_' + ('hall' if nm == 'closed' else nm))
                if nm in ('hall', 'closed'): R = kit.rim(R, 2.6, 1.0, '#a8cbe0', .85, .4)                       # the one cold rim
                else: R = kit.rim(R, 1.6, .8, '#7f9fb6', .30, .4)
                np.save(c + '_%s.npy' % nm, R.astype(np.float16)); del R
            np.save(c + '_sky.npy', L.f('l_sky')[TOP:, :, :3].astype(np.float16))
            for nm in NAMES: np.save(c + '_hm_%s.npy' % nm, np.asarray(L.a('x_hm_' + nm)))
            np.save(c + '_ok.npy', np.array(list(L.a('x_wl')) + list(L.a('x_dist')) + [float(L.a('x_hz'))]))
        s.cam = world.lazy('shrine', **KW).cam; s.cam.cy -= TOP; s.cam.h = H; m = np.load(c + '_ok.npy'); s.wl, s.dist, s.hz = m[:4] - TOP, m[4:8], float(m[8]) - TOP; s.lay = {n: np.load(c + '_%s.npy' % n, mmap_mode='r') for n in NAMES + ('closed', 'sky')}; s.hm = {n: np.load(c + '_hm_%s.npy' % n, mmap_mode='r') for n in NAMES}
        cam = s.cam; u = cam.uv((0, 0, 0), (1, 0, 0), (0, 1, 0)); s.U, s.V, s.Z = [a.astype(f32) for a in u]; s.gr = s.Z > 0; s.x, s.y = grid(H, W); s.roll = math.asin(cam.R[0][2])
        r = np.random.default_rng(5); s.wv = [(r.uniform(0, 6.28), r.uniform(.18, 1.5), r.uniform(.06, .18) * (1 if k % 2 else -1), r.uniform(0, 6)) for k in range(6)]; s.amp = np.clip(56 / np.maximum(s.Z, .5), .4, 10.5); s.n1 = fbm(H, W, (72, 24), (1, .5), 9, 7.); s.far = np.clip(20 / np.maximum(s.Z, 3), .12, 1.)
        s.mo = sorted(WS.pillar_mouths(cam), key=lambda q: (q[0], q[5])); t0 = [q for q in s.mo if q[0] == 0]; up = [q for q in s.mo if q[0] > 0]; s.open = [(q, 36 + i) for i, q in enumerate(t0)] + [(q, 45 + i * .75) for i, q in enumerate(sorted(up, key=lambda q: q[5]))]
        s.mT, s.mM, s.mH = [kit.mask_of(np.array(p, float)) for p in (POLT, POLM, POLH)]; s.mM = s.mM * (1 - s.mT); s.mH = np.clip(s.mH, 0, 1)
        # sky slabs: straight cuts through one far point, each piece slides along its cut
        r = np.random.default_rng(12); cen = np.array([1010., 360.]); s.cut = []
        for k in range(4): a = (-1.18, -.34, .42, 1.05)[k] + r.normal(0, .05); off = (-120, 60, -90, 150)[k]; d = np.array([math.cos(a), math.sin(a)]); n = np.array([-d[1], d[0]]); s.cut.append((d, n, float(n @ cen + off)))
        idm = np.zeros((H, W), np.int32)
        for k, (d, n, q) in enumerate(s.cut): idm += ((s.x * n[0] + s.y * n[1] - q) > 0).astype(np.int32) << k
        ids = np.unique(idm); s.idm = np.searchsorted(ids, idm).astype(np.int32); s.off = np.zeros((len(ids), 2), f32)
        for j, i in enumerate(ids):
            ys, xs = np.nonzero(s.idm == j); p = np.array([xs.mean(), ys.mean()]); best = min(range(4), key=lambda k: abs(p @ s.cut[k][1] - s.cut[k][2])); d, n, q = s.cut[best]; s.off[j] = d * (1 if (p @ n) > q else -1) * r.uniform(20, 48)
        del idm

    # ---------------------------------------------------------------- layers
    def hall(s, f):
        """closed hall with the mouths that have opened so far"""
        if f >= 56: return s.lay['hall'][:].astype(f32)
        B = s.lay['closed'][:].astype(f32); A = s.lay['hall']
        for (t, x0, y0, x1, y1, cx, cy), b in s.open:
            if f < b: continue
            k = (.16, .62, 1.14, 1.)[min(3, int(f - b))]; x0, y0, x1, y1 = [int(round(v)) for v in (x0 - 2, y0 + TOP, x1 + 3, y1 + TOP)]; R = A[y0:y1, x0:x1].astype(f32); w = R.shape[1]
            if k != 1.: R = cv2.warpAffine(R, np.float32([[k, 0, w * .5 * (1 - k)], [0, 1, 0]]), (w, R.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            xx = np.abs(np.arange(w) - w * .5) / (w * .5); m = smooth(min(1., k) + .02, min(1., k) - .10, xx.astype(f32))[None, :, None]; gl = 1 + (.9, .6, .25, 0.)[min(3, int(f - b))]; R[..., :3] = np.minimum(R[..., :3] * gl, R[..., 3:]); B[y0:y1, x0:x1] = B[y0:y1, x0:x1] * (1 - m) + R * m
        return B

    def sunk(s, nm, L, sink):
        if sink <= .02: return L[TOP:], 0.
        dy = s.cam.f * sink / float(s.dist[NAMES.index(nm)])
        if dy > H + TOP: return np.zeros((H, W, 4), f32), dy
        m = smooth(sink - .10, sink + .22, s.hm[nm][:].astype(f32))[..., None]; return cv2.warpAffine(L * m, np.float32([[1, 0, 0], [0, 1, dy - TOP]]), (W, H)), dy

    def flip(s, L, wl):
        c, n = math.cos(2 * s.roll), math.sin(2 * s.roll); x0, y0 = W / 2, wl; M = np.float32([[c, n, x0 - c * x0 - n * y0], [n, -c, y0 - n * x0 + c * y0]]); return cv2.warpAffine(L, M, (W, H), borderMode=cv2.BORDER_REPLICATE if L.shape[2] == 3 else cv2.BORDER_CONSTANT)

    def sky(s, f):
        K = s.lay['sky'][:].astype(f32)
        if f < 57: return K
        t = float(ease(np.clip((f - 57) / 9., 0, 1))); o = s.off[s.idm] * t; mx, my = s.x - o[..., 0], s.y - o[..., 1]; src = cv2.remap(s.idm.astype(np.float32), mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE); gap = (np.abs(src - s.idm) > .5).astype(f32)
        K = cv2.remap(K, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101); sh = 1 + (((s.idm * 37) % 7) / 6. - .5) * 1.25 * t; K = K * sh[..., None]
        gap = cv2.GaussianBlur(gap, (0, 0), .7); K = K * (1 - gap[..., None]) + rgb('#050204') * gap[..., None]; ed = np.zeros((H, W), f32)
        for dx, dy in ((1, 0), (0, 1)): ed = np.maximum(ed, (s.idm != np.roll(s.idm, (dy, dx), (0, 1))).astype(f32))
        ed = cv2.GaussianBlur(ed, (0, 0), .8) * 1.5; return K + (rgb('#f2e6d6') - K) * np.clip(ed * (.10 + .40 * t), 0, 1)[..., None]

    def frame(s, f):
        sk = rise(f); sky = s.sky(f); img = sky.copy(); ref = s.flip(sky, s.hz); cov = np.zeros((H, W), f32); tops = []
        for i, nm in enumerate(NAMES):
            if nm == 'hall': L = s.hall(f)
            else: L = s.lay[nm][:].astype(f32)
            L, dy = s.sunk(nm, L, sk[i])
            if nm == 'hall' and f >= 57: L = s.split(f, L)
            img = img * (1 - L[..., 3:]) + L[..., :3]; F = s.flip(L, float(s.wl[i])); ref = ref * (1 - F[..., 3:]) + F[..., :3]; cov = cov * (1 - L[..., 3]) + L[..., 3]
            tops.append(L[..., 3] > .5 if (f < 36 and sk[i] < 40) else None); del F, L
        # water
        t = f * 1.; rip = np.zeros((H, W), f32); boil = float(np.clip(1.25 - f / 32., 0, 1)) if f < 30 else 0.
        for a, f_, w_, p_ in s.wv: rip += np.sin((s.U * math.cos(a) * .55 + s.V * math.sin(a) * 1.5) * f_ * 2 + t * w_ * (1 + 1.6 * boil) + p_) / (1 + f_)
        rip *= 1 + 1.1 * boil; rg = np.zeros((H, W), f32)
        if f < 30:
            for k in range(4): R = (f * .9 + k * 7.) % 30; q = np.hypot(s.U, (s.V - 4) * 1.0) - 20 - R; rg += .8 * boil * np.exp(-(q / 1.3) ** 2) * np.cos(q * 2.6)
        if f >= 27:
            R = 15 + 2.9 * (f - 27) ** 1.06; q = np.hypot(s.U, s.V - 4) - R; am = 2.6 * math.exp(-(f - 27) / 26.); rg += am * np.exp(-(q / 2.2) ** 2) * np.cos(q * 1.7)
        rp = rip + rg; R_ = cv2.remap(ref, np.clip(s.x + rp * s.amp * 1.1 + s.n1 * s.amp * .5, 0, W - 1), np.clip(s.y + rp * s.amp * .8, 0, H - 1), cv2.INTER_LINEAR); del ref
        wat = R_ * np.array([.46, .27, .27], f32) + rgb('#120306') * .7; sw = rp + .35 * s.n1; lt = smooth(.98, 1.06, sw * (.55 + .45 * s.far)) * smooth(.02, .10, s.far); dk = smooth(-.95, -1.05, sw) * smooth(.04, .2, s.far); wat = wat + (rgb('#8a1118') - wat) * (lt * .55)[..., None]; wat = wat + (rgb('#080305') - wat) * (dk * .55)[..., None]
        b = smooth(.46, .58, rg * (.86 + .16 * s.n1)); wat = wat + (rgb('#d8564a') - wat) * (b * .85)[..., None]; bw = smooth(1.05, 1.25, rg * (.82 + .20 * s.n1 + .10 * np.sin(s.U * 1.7))); wat = wat + (rgb('#f0dcc8') - wat) * (bw * .9)[..., None]
        ln = smooth(.07, .015, np.abs(np.sin(s.V * 1.1 + rp * 1.1 + .5 * np.sin(s.U * .6)))) * smooth(.12, .5, s.far) * smooth(-.1, .5, s.n1); wat = wat * (1 - .6 * ln)[..., None]
        m = s.gr.astype(f32) * (1 - cov); img = img * (1 - m[..., None]) + wat * m[..., None]; del wat, R_
        if f < 40: img = over(img, s.sheets(f, tops, sk))
        # dry-brush ink still settling over everything while it rises
        if f >= 67: img = over(img, s.plunge(f))
        if 54 <= f < 64: img = s.slashes(img, f)
        # camera: heavy creep, tremble while it rises, jolt at 27
        q = float(ease(np.clip((f - 29) / 24., 0, 1))) if f < 54 else 1 - float(ease(np.clip((f - 53) / 4., 0, 1))); z = 1.012 + .016 * np.clip(f / 27., 0, 1) + .27 * q; sh = kit.shake(f, 2.6 * boil + (0 if f < 27 else 13 * math.exp(-(f - 27) / 3.2)) + (7 * math.exp(-(f - 54) / 1.6) if f >= 54 else 0) + (6 * math.exp(-(f - 68) / 1.6) if f >= 68 else 0), 4); img = comp.cam2d(img, W / 2 + sh[0] - 14 * q, H / 2 + sh[1] + 16 * q, z)
        if f == 27: img = kit.impact(img, 'red', .05, .34, BLK, '#efe4d0', '#8a0f18')
        return img.astype(f32)

    def sheets(s, f, tops, sk):
        """water falling off every edge that has just come up, boiling foam round the waterlines"""
        cel = Cel(W, H, ss=1); r = np.random.default_rng(500 + f); k = float(np.clip((f - 1) / 5., 0, 1) * np.clip((40 - f) / 12., 0, 1)); sp = float(np.clip((30 - f) / 22., .10, 1))
        for i, T in enumerate(tops):
            if T is None: continue
            xs = np.nonzero(T.any(0))[0]
            if len(xs) < 10: continue
            n = int((150, 60, 50, 34)[i] * k); px = r.choice(xs, n); g = (1., 1., 1.5, 2.4)[i]
            for x in px:
                col = np.nonzero(T[:, x])[0]
                if len(col) < 8: continue
                # under-edges: where the column of the silhouette or an inner eave ends
                y = int(col[int(len(col) * r.random() ** 1.5)]); L = r.uniform(26, 130) * g * (.5 + .7 * sp); w = r.uniform(.7, 2.4) * g; q = r.random(); c = BONE if q < .30 else '#8a1119' if q < .72 else '#12060a'; x_ = x + r.normal(0, 1.5)
                cel.fill(np.array([(x_ - w, y), (x_ + w, y), (x_ + w * .5 + r.normal(0, 1), y + L * .7), (x_ + r.normal(0, 1), y + L)]), c, float(r.uniform(.5, .95)))
                if q < .25: cel.fill(np.array([(x_ - 1.4 * g, y + L * 1.12), (x_, y + L * 1.05), (x_ + 1.4 * g, y + L * 1.14), (x_, y + L * 1.2)]), BONE, .9)
            # foam at the waterline
            ys = s.y[:, 0]; wl = s.wl[i]; m = int((90, 46, 40, 26)[i] * sp * (f > 1))
            for _ in range(m):
                x = float(r.choice(xs)) + r.normal(0, 16 * g); yb = wl + (x - W / 2) * math.tan(s.roll) * -1 + r.normal(0, 5 * g); wd = r.uniform(10, 46) * g; hh = r.uniform(3, 20) * g * sp; P = [(x - wd, yb + 2 * g), (x - wd * .5, yb - hh * r.uniform(.3, .9)), (x - wd * .1, yb - hh * r.uniform(.5, 1.2)), (x + wd * .3, yb - hh * r.uniform(.4, 1.)), (x + wd, yb + 2 * g), (x, yb + 4 * g)]; cel.fill(spl(np.array(P), True, corners=(0, 4), n=4), BONE if r.random() < .34 else '#a3161e' if r.random() < .6 else '#12060a', float(r.uniform(.6, .95)))
                if r.random() < .5:
                    for _q in range(3): dx, dy = r.normal(0, wd * .6), -abs(r.normal(0, 46 * g * sp)); z = r.uniform(.8, 2.6) * g; cel.fill(np.array([(x + dx - z, yb + dy), (x + dx, yb + dy - z * 1.5), (x + dx + z, yb + dy), (x + dx, yb + dy + z * 1.1)]), BONE if r.random() < .5 else '#b81c24')
        return cel.layer()

    def slashes(s, img, f):
        """twelve tapered cuts, three drawings on ones, then they thin and go"""
        cel = Cel(W, H, ss=2); dk = Cel(W, H, ss=1); j = min(2, f - 54)
        for (x0, y0, x1, y1, bow, w, k) in CUTS:
            if k > f - 54: continue
            age = f - 54 - k; a = (1., .62, .42, .30, .22, .16, .11, .08, .05, .03)[min(9, age)]; t = np.linspace(0, 1, 26); d = np.array([x1 - x0, y1 - y0], float); n = np.array([-d[1], d[0]]) / np.hypot(*d); e = 1 if age == 0 else 1 - .04 * age
            tt = .5 + (t - .5) * e; P = np.array([x0, y0]) + d * tt[:, None] + n * (bow * 4 * tt * (1 - tt))[:, None]; pr = np.sin(np.pi * np.clip(t, 0, 1)) ** 1.3 * (.35 + .65 * np.exp(-((t - .42) / .30) ** 2)); hw = (w * a * pr * .62)[:, None]
            cel.fill(np.vstack([P + n * hw, (P - n * hw)[::-1]]), '#ffffff' if age < 2 else '#efe2cc'); hb = ((w * a * .8 + 1.6) * pr)[:, None]
            if age < 3: dk.fill(np.vstack([P + n * (hb + 1.5), (P + n * .5)[::-1]]), BLK, .9)
        if f < 57: img = img * (.82, .70, .62)[j]
        L = cel.layer(); img = over(img, dk.layer()); img = img + cv2.GaussianBlur(L[..., 3], (0, 0), 6)[..., None] * rgb('#cfe4f4') * (.45 if f < 57 else .12); return over(img, L)

    def split(s, f, L):
        """tiers slide apart along their cuts; the horn comes off"""
        u = float(np.clip((f - 58) / 9., 0, 1)) ** 2.1; u = u if f < 67 else 1 + .012 * math.exp(-(f - 67) / 1.5) * math.cos((f - 67) * 2.2); hc = float(np.clip((f - 57) / 2., 0, 1)) * 2.0
        da = np.array([CA[0] - CA[2], CA[1] - CA[3]], float); da /= np.hypot(*da); db = np.array([CB[2] - CB[0], CB[3] - CB[1]], float); db /= np.hypot(*db); om = db * (48 * u + hc); ot = om + da * (60 * u + hc)
        mT, mM, mH = s.mT[..., None], s.mM[..., None], s.mH[..., None]; T, M, Hn = L * mT, L * mM, L * mH; B = L * (1 - mT) * (1 - mM) * (1 - mH)
        def mv(R, o, ang, pv): c, n = math.cos(ang), math.sin(ang); return cv2.warpAffine(R, np.float32([[c, -n, pv[0] - c * pv[0] + n * pv[1] + o[0]], [n, c, pv[1] - n * pv[0] - c * pv[1] + o[1]]]), (W, H))
        def face(P0, P1, o, th):                                           # the cut surface: black stone, one lit lip
            cel = Cel(W, 400, ox=0, oy=300, ss=2); a, b = np.array(P0, float) + o, np.array(P1, float) + o; cel.fill(np.array([a + [0, -th], b + [0, -th], b + [0, 4], a + [0, 4]]), '#2a0a0e'); cel.fill(np.array([a + [0, -th], b + [0, -th], b + [0, -th + 2.2], a + [0, -th + 2.2]]), '#e8dcc2'); cel.fill(np.array([a + [0, -th * .5], b + [0, -th * .42], b + [0, -th * .2], a + [0, -th * .24]]), '#7d1119'); return cel.layer()
        out = B
        def ov(o, l): return o * (1 - l[..., 3:4]) + l
        if u > .02: out = ov(out, face((800, 533.5), (1080, 562), (0, 0), 8 * min(1, u * 4)))
        out = ov(out, mv(M, om, .010 * u, (1110, 560)))
        if u > .02: out = ov(out, face((770, 437.5), (1066, 401), om, 7 * min(1, u * 4)))
        out = ov(out, mv(T, ot, -.020 * u, (770, 438)))
        # horn: free at 59, in the water at 68
        g = max(0., f - 59.); q = g / 9.
        if f < 69: ang = 1.9 * q ** 1.5; o = np.array([6 + 92 * q, 392 * q ** 2]); pv = (1300, 470); H2 = mv(Hn, o, ang, pv); wl = 862.; H2 = H2 * smooth(wl + 8, wl - 2, s.y)[..., None]; out = ov(out, H2)
        return out

    def plunge(s, f):
        """the splash the horn makes: crown, column, rings (drawn on ones)"""
        k = f - 68; cel = Cel(W, H, ss=2); cx, cy = 1392., 862.; r = np.random.default_rng(60)
        if k < 0: return cel.layer()
        g = (.55, 1.0, 1.18, 1.05)[min(3, k)]
        for j in range(26):
            a = -math.pi * (.06 + .88 * (j + r.uniform(-.3, .3)) / 25.); L = (70 + 120 * r.random() ** 1.4) * g * (1.25 if abs(a + math.pi / 2) < .5 else .8); w = r.uniform(3, 9) * (1 - .10 * k); b = np.array([cx + math.cos(a) * 34 * g, cy + r.uniform(-2, 3)]); tip = b + np.array([math.cos(a) * L * .62, math.sin(a) * L]); n = np.array([-math.sin(a), math.cos(a)]) * w; m = (b + tip) / 2 + np.array([math.cos(a) * 6, 10 * k * .3]); cel.fill(spl(np.array([b - n, m - n * .6, tip, m + n * .6, b + n]), True, corners=(2,), n=5), ('#12060a', '#a3161e', BONE, '#7d1119')[j % 4])
            z = r.uniform(1.4, 4.2); p = tip + np.array([math.cos(a) * 14 * (1 + k), -16 + 9 * k * k * .5]) * r.uniform(.5, 1.4); cel.fill(spl(p + np.array([[-z, 0], [0, -z * 1.4], [z, 0], [0, z]]), True, n=4), BONE if j % 3 else '#b81c24')
        for j, (rr, w) in enumerate(((60 + 46 * k, 7 - k), (30 + 34 * k, 5 - k))):
            th = np.linspace(0, 2 * math.pi, 60); P = np.c_[cx + np.cos(th) * rr * (1 + .02 * np.sin(th * 5 + k)), cy + 4 + np.sin(th) * rr * .17]; cel.line(P, max(1.2, w), '#e8dcc2' if j == 0 else '#c8353a', taper=(0, 0), press=.7, seed=70 + j + k)
        return cel.layer()
