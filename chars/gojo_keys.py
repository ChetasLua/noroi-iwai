"""Full-finish hero keys of Gojo (1920x1080).   python -m chars.gojo_keys 1 2 3 sheet"""
import sys, os, math, numpy as np, cv2
from jj import W, H as HH, comp
from jj.paint import rgb, f32, smooth, fbm, grid, ramp, Canvas, arcs, warp_by
from jj.ink import Cel, spl, resample
from jj import kit
from chars.gojo import Gojo, gojo_head, PAL, GC, GE, GH, GF, rim, eye_glow


def _save(name, img, f=0, **fin):
    os.makedirs('qa/gojo', exist_ok=True)
    comp.save('qa/gojo/%s.jpg' % name, comp.to8(comp.finish(img, f, **fin), f), 93)


# ------------------------------------------------------------------------------------------------ flame aura
def aura_tongues(h, w, layer_a, seed=0, f=0, up=(0, -1), reach=70, n=90, col=('#25c8ff', '#7a5cff'), base=None, step=1):
    """cursed-energy aura: flame tongues that grow from the silhouette outward / upward, drawn (hard inner band, soft outer)"""
    r = np.random.default_rng(seed)
    cs = kit.contours(layer_a, min_len=60)
    c = Cel(w, h)
    c2 = Cel(w, h)
    for C in cs:
        C = resample(C, 4.0); T = np.gradient(C, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-6; N = np.c_[T[:, 1], -T[:, 0]]
        ar = .5 * np.sum(C[:-1, 0] * C[1:, 1] - C[1:, 0] * C[:-1, 1]);
        if ar < 0: N = -N
        k = int(len(C) / 1300. * n)
        for i in np.sort(r.integers(0, len(C), k)):
            p, nn = C[i], N[i]
            if not (-40 < p[0] < w + 40 and -40 < p[1] < h + 40): continue
            L = reach * r.uniform(.35, 1.25) * (1 + .25 * math.sin(f * .9 + i)); wd = r.uniform(10, 26)
            d = nn * .55 + np.array(up) * 1.0; d /= np.hypot(*d) + 1e-6
            if nn @ np.array(up) < -.35: L *= .4
            a, b = C[max(0, i - int(wd / 4))], C[min(len(C) - 1, i + int(wd / 4))]
            tip = p + nn * L * .55 + d * L * .75
            S = kit.spike(a - nn * 8, tip, b - nn * 8, .25, r.normal(0, .12))
            c.fill(S, col[0] if r.random() < .6 else col[1], r.uniform(.45, .9))
            if r.random() < .4: c2.line(np.array([p + nn * 4, p + nn * L * .4 + d * L * .3, tip + d * L * .3]), r.uniform(1.4, 3.2), '#e6fbff', taper=(.2, .8), seed=int(r.integers(1 << 20)))
    return c.layer(h, w), c2.layer(h, w)


# ------------------------------------------------------------------------------------------------ our own manga page
INK_, PAPER_ = '#0b0a0c', '#efece8'


def panel(page, poly, art, border=6.):
    """paste a picture into a panel polygon and ink its frame"""
    P = np.asarray(poly, float); m = kit.mask_of(P)[..., None]
    page = page * (1 - m) + art * m; c = Cel()
    for a, b in zip(P, np.roll(P, -1, 0)): c.line(np.array([a, (a + b) / 2, b]), border, INK_, taper=(0, 0), press=.15, seed=int(abs(a[0])), tremor=.3)
    return comp.over(page, c.layer())


def flat(col=PAPER_): return np.ones((HH, W, 3), f32) * rgb(col)


def hatch_bg(seed, ang, gap=9, col=INK_, w=2.2, n=420, box=(0, 0, W, HH)):
    c = Cel(); kit.speed_lines(c, ang, n, w, (200, 1200), col, seed, box); return c.layer()


def art_profile(f=0):
    """he turns his head: 3/4 from behind the collar, speed lines"""
    img = flat(); c = Cel(); kit.focus_lines(c, 560, 330, 330, 1500, 210, 8, INK_, 12, ry=.8); img = comp.over(img, c.layer())
    tn = kit.screentone(smooth(760, 260, grid()[0]), 6.5, .7, .30); img = img * (1 - tn[..., None]) + rgb(INK_) * tn[..., None]
    L = Gojo.bust(540, 330, 168, yaw=.62, pitch=-.06, roll=.05, state='up', expr='plain', light='manga', body=(.5, 0, 0), shift=(-.10, 0), seed=6)
    img = img * (1 - cv2.dilate(L[..., 3], np.ones((9, 9)))[..., None]) + cv2.dilate(L[..., 3], np.ones((9, 9)))[..., None]
    return comp.over(img, L)


def art_hand(f=0):
    from jj import hands as HD
    img = flat('#0b0a0c'); c = Cel(); kit.focus_lines(c, 320, 700, 250, 1400, 260, 7, '#f4f2ef', 4, ry=.9); img = comp.over(img, c.layer())
    c = Cel(); GC.sleeve(c, (120, 1300), (230, 1130), (318, 960), 150, 130, 78, PAL['manga'], 250, seed=3, lit=-1)
    HD.draw(c, 'cross', HD.Cam(330, 770, 165), pal=dict(skin='#f3f1ee', shade='#b4b1b4', line=INK_, nail='#ffffff', nail_shade='#c6c4c8', light='#ffffff'), side='R', R=HD.euler(160, 6, 6), bld=HD.SLIM, w=6, light=(-.6, .7, .4), seed=2)
    return comp.over(img, c.layer())


def art_eye(f=0):
    from chars.gojo_six import six_eye
    L = six_eye(W, HH, .86, .5, .24, 0., centre=(1490, 330), scale=.40, roll=.10, side=1)
    g = L[..., :3] @ np.array([.3, .5, .2], f32); g = np.clip((g - .10) * 1.55, 0, 1); g = np.where(g > .62, 1., np.where(g > .30, .70, .04)) * .75 + g * .25
    return np.dstack([g, g, g]).astype(f32) * rgb('#f6f4f2')


def art_blast(f=0):
    from jj import fx
    img = flat(); c = Cel(); kit.focus_lines(c, 1560, 860, 150, 1200, 240, 8, INK_, 21, ry=.8); img = comp.over(img, c.layer())
    c = Cel(); fx.smoke_burst(c, 12, 1560, 900, 330, seed=4, n=11, lw=4.0); fx.debris(c, 9, (1560, 900), n=26, seed=5, size=(10, 70), lw=4.0)
    L = c.layer(); g = (L[..., :3] @ np.array([.3, .5, .2], f32)) / np.maximum(L[..., 3], 1e-3); g = np.where(g > .60, .97, np.where(g > .36, .70, .05)).astype(f32)
    L = np.dstack([g * L[..., 3]] * 3 + [L[..., 3]]); img = comp.over(img, L)
    c = Cel(); kit.focus_lines(c, 1560, 880, 520, 1300, 60, 16, INK_, 31, ry=.85); return comp.over(img, c.layer())


PAGE = dict(profile=[(-30, -30), (905, -30), (700, 506), (-30, 452)], hand=[(-30, 486), (690, 540), (520, 1110), (-30, 1110)], eye=[(1010, -30), (1950, -30), (1950, 520), (905, 430)], blast=[(1180, 490), (1950, 556), (1950, 1110), (1010, 1110)])


def manga_page(parts=('profile', 'hand', 'eye', 'blast'), sfx=True, key=2):
    from jj import letters as LT
    os.makedirs('cache', exist_ok=True); fn = 'cache/gojo_page_v%d_%s%d.npy' % (key, ''.join(p[0] for p in parts), sfx)
    if os.path.exists(fn): return np.load(fn).astype(f32)
    pg = flat('#f7f5f2')
    A = dict(profile=art_profile, hand=art_hand, eye=art_eye, blast=art_blast)
    for p in parts: pg = panel(pg, PAGE[p], A[p]())
    if sfx:
        c = Cel()
        LT.sfx(c, 'ドン', [(30, 60), (250, 30), (300, 330), (60, 400)], seed=2)
        LT.sfx(c, 'ゴゴゴ', [(1700, 620), (1890, 600), (1900, 1060), (1690, 1060)], seed=3, vertical=True)
        pg = comp.over(pg, c.layer())
    np.save(fn, pg.astype(np.float16)); return pg


# ------------------------------------------------------------------------------------------------ KEY 1: blindfold, half smile
def key1(f=0):
    from jj import fx
    x, y = grid(); cx, cy, s = 1000, 468, 258
    page = manga_page()
    base = comp.blur(page, 1.6) * rgb('#b9b4c4') * .74 + rgb('#1a1430') * .05
    base = base * (1 - .35 * smooth(300, 1150, np.sqrt((x - cx) ** 2 + (y - cy) ** 2)))[..., None]
    L = Gojo.bust(cx, cy, s, yaw=-.22, pitch=-.12, roll=-.08, state='up', expr='smirk', light='day', key='L', body=(-.40, -.02, .05), shift=(.05, -.02), seed=2)
    al = L[..., 3]
    base = comp.screen(base, comp.blur(al, 90)[..., None] * rgb('#4a1cff') * .85)
    bk, fr = fx.cursed(L, f, size=120, seed=3, pal=dict(fx.VIOLET) if isinstance(fx.VIOLET, dict) else fx.VIOLET, amount=.9, wisps=1.1)
    for q in (bk, fr):                          # no sheath along the frame edge
        for a_ in q:
            if a_ is not None: a_[1060:] *= np.linspace(1, 0, 20)[:, None, None].astype(f32)
    base = fx.put(base, bk)
    base = fx.put(base, fx.aura(L, r=58, inner='#5ff2ff', mid='#1a8cff', outer='#7d4dff', hard=6, f=f, seed=2))
    L = rim(L, 5, 1, '#37d6ff', .80, soft=.9); L = rim(L, -4, 1, '#7fe9ff', .7, soft=.8)
    L = kit.para(L, (cx - 100, 1080), (cx + 60, 400), [(0, '#a79be8'), (.45, '#eae2ff'), (1, '#ffffff')], 'mul')
    L = kit.para(L, (cx - 620, 700), (cx + 40, 520), [(0, '#3f62ff'), (.5, '#000000'), (1, '#000000')], 'screen', .5)
    img = comp.over(base, L)
    img = fx.put(img, fr, .8)
    return img


# ------------------------------------------------------------------------------------------------ KEY 2: the Six Eyes
def god_rays(img, cx, cy, thr=.55, amt=.9, n=26, it=2, q=4, col='#9fe6ff', mask=None):
    """light pouring out of a bright place: radial smear of the thresholded picture, additive"""
    L = comp.lum(img) if hasattr(comp, 'lum') else img.mean(-1)
    src = np.clip((L - thr) / (1 - thr), 0, 1)
    if mask is not None: src = src * mask
    s = cv2.resize(src, (W // q, HH // q), interpolation=cv2.INTER_AREA); h, w = s.shape; x0, y0 = cx / q, cy / q
    for k in range(it):
        out = np.zeros_like(s); tot = 0.
        for t in np.linspace(0, 1, n):
            z = 1 + amt * t * (1 if k == 0 else .35); wt = (1 - t) ** 1.2 + .03
            out += wt * cv2.warpAffine(s, np.float32([[z, 0, x0 * (1 - z)], [0, z, y0 * (1 - z)]]), (w, h), flags=cv2.INTER_LINEAR); tot += wt
        s = out / tot
    return cv2.resize(s, (W, HH), interpolation=cv2.INTER_CUBIC)[..., None] * rgb(col)


def radial_streaks(cx, cy, seed=0, n=900, r0=160, r1=1300, sharp=6., rot=0.):
    x, y = grid(); dx, dy = x - cx, y - cy; d = np.sqrt(dx * dx + dy * dy); a = (np.arctan2(dy, dx) + rot) % (2 * np.pi)
    r = np.random.default_rng(seed); g = np.zeros(n)
    for sc in (9, 23, 61): g += np.interp(np.arange(n), np.linspace(0, n, sc * 3 + 1), np.r_[(v := r.random(sc * 3)), v[0]])
    g = (g - g.min()) / (g.max() - g.min()); g = np.clip((g - .48) / .52, 0, 1) ** 1.4
    ang = np.interp(a.ravel(), np.linspace(0, 2 * np.pi, n), g).reshape(a.shape).astype(f32)
    return ang * smooth(r0 * .5, r0 * 1.6, d) * np.clip(1 - d / r1, 0, 1) ** 1.5


def prism(img, cx, cy, r0, r1, amt=.012):
    """chromatic split that lives on a ring (refraction at the rim of the iris)"""
    x, y = grid(); d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2); m = (smooth(r0 * .55, r0, d) * (1 - smooth(r1, r1 * 1.35, d)))[..., None]
    out = img.copy()
    for c, s_ in ((0, 1 + amt), (2, 1 - amt)):
        out[..., c] = cv2.warpAffine(img[..., c], np.float32([[s_, 0, cx * (1 - s_)], [0, s_, cy * (1 - s_)]]), (W, HH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return img * (1 - m) + out * m


def crystals(seed, cx, cy, n, rmin, rmax, smin, smax, f=0., blur=0., drift=0.):
    """splinters of cut light floating out of the eye"""
    r = np.random.default_rng(seed); c = Cel()
    for k in range(n):
        a = r.uniform(0, 6.28); d = (r.uniform(rmin, rmax) + drift * f * r.uniform(.5, 1.5)); p = np.array([cx + math.cos(a) * d * 1.25, cy + math.sin(a) * d * .8]); s = r.uniform(smin, smax) * (r.random() ** 1.5 + .25)
        th = a + r.normal(0, .5) + f * r.normal(0, .02); ax = np.array([math.cos(th), math.sin(th)]); nx = np.array([-ax[1], ax[0]]); q = r.uniform(.18, .42)
        A, B, C_, D = p + ax * s, p + nx * s * q, p - ax * s * r.uniform(.6, 1.), p - nx * s * q
        c.fill(np.array([A, B, C_, D]), '#9fe4ff', .85); c.fill(np.array([A, B, p]), '#ffffff', .95); c.fill(np.array([C_, D, p]), '#3f8cf0', .8)
        if r.random() < .3: c.fill(np.array([B, C_, p]), '#b79bff', .8)
    L = c.layer(); return comp.bokeh(L, blur, 0) if blur > 1 else L


def key2_cloth(c, Q, k, dy=0.):
    """the edge of the blindfold, pushed up: black cloth, strain folds running to the thumb (upper left), hem lit by the eye"""
    o = np.array([0., dy])
    low = spl(np.array([(-1000, -104), (-860, -190), (-700, -262), (-560, -300), (-400, -352), (-220, -398), (-20, -418), (200, -410), (420, -376), (620, -330), (800, -270), (1060, -190), (1300, -120)]), n=10) + o
    top = np.array([(1500, -1100), (-1300, -1100)]) + o
    c.fill(Q(np.vstack([low + (6, 52), top])), '#07071c', .80)              # its shadow on the brow
    P = np.vstack([low, top]); c.fill(Q(P), '#06050d'); c.clip(Q(P))
    hem = low + np.c_[np.zeros(len(low)), -30 - 5 * np.sin(np.arange(len(low)) * .21)]
    c.fill(Q(np.vstack([low, hem[::-1]])), '#0f1330')
    for a, col in (([(-980, -190), (-700, -330), (-520, -380), (-330, -440), (-560, -420), (-760, -350)], '#2c4aa8'), ([(-300, -452), (-120, -488), (80, -500), (-90, -510)], '#24408e'),
                   ([(-860, -430), (-620, -520), (-300, -560), (-120, -560), (-340, -600), (-640, -580)], '#141a44'), ([(150, -470), (380, -450), (640, -392), (800, -340), (560, -420), (330, -478)], '#2746a0'),
                   ([(880, -320), (1060, -262), (1240, -190), (1080, -290)], '#213b86'), ([(60, -560), (320, -550), (620, -490), (860, -420), (620, -530), (330, -590)], '#131840'), ([(420, -650), (760, -600), (1060, -500), (760, -640)], '#10143a'),
                   ([(-420, -640), (-160, -650), (60, -630), (-150, -690)], '#0f1334')):
        c.fill(Q(spl(np.array(a, float) + o, True, corners=(0, len(a) // 2), n=7)), col)
    for a in ([(-990, -240), (-760, -372), (-520, -452), (-330, -500)], [(-900, -340), (-640, -470), (-380, -540)], [(1240, -230), (920, -370), (620, -460), (400, -500)], [(1180, -360), (860, -480), (560, -540)], [(-300, -600), (-60, -570), (180, -560)], [(-120, -700), (200, -680), (500, -640)], [(-560, -330), (-600, -420), (-660, -520)], [(620, -360), (700, -450), (820, -560)]):
        c.line(Q(spl(np.array(a, float) + o, n=8)), 4.5 * k, '#000000', taper=(.1, .7), press=.4, seed=int(abs(a[0][0])))
    for q_ in range(64):                                                  # hem stitches
        i = int(q_ / 64 * (len(low) - 1)); p = low[i] + (0, -19); c.line(Q(np.array([p, p + (7, -1.5), p + (14, -3.5)])), 2.4 * k, '#3656a8', taper=(.3, .3), seed=q_)
    c.unclip()
    c.line(Q(low), 8 * k, '#03030a', taper=(.02, .02), press=.4, seed=3)
    for a_, b_, w_ in ((.10, .34, 3.2), (.36, .58, 4.4), (.60, .70, 2.6), (.72, .92, 3.4)):
        c.line(Q(low[int(a_ * len(low)):int(b_ * len(low))] + (0, -7)), w_ * k, '#8fd0ff', taper=(.35, .35), press=.6, seed=4 + int(a_ * 50))


def key2_hair(c, Q, k, dy=0., sway=0.):
    """bare: brow and the ends of the fringe above the eye, white hair in the eye's own light"""
    o = np.array([0., dy]); r = np.random.default_rng(6)
    B = spl(np.array([(-540, -372), (-380, -452), (-160, -505), (80, -520), (320, -500), (540, -440), (700, -372)]), n=12) + o; t = np.linspace(0, 1, len(B)); N = np.gradient(B, axis=0); N = np.c_[-N[:, 1], N[:, 0]]; N /= np.hypot(*N.T)[:, None]; w = np.interp(t, [0, .06, .3, .7, 1], [16, 44, 52, 34, 0])
    c.fill(Q(np.vstack([B + N * w[:, None] * .5 + (4, 22), (B - N * w[:, None] * .5 + (4, 22))[::-1]])), '#0e1038', .7)
    c.fill(Q(np.vstack([B + N * w[:, None] * .5, (B - N * w[:, None] * .5)[::-1]])), '#7284cc')
    for i in range(64):
        j = int(r.uniform(0, .93) * len(B)); L = r.uniform(50, 130); d = (B[min(len(B) - 1, j + 6)] - B[max(0, j - 6)]); d = d / (np.hypot(*d) + 1e-6); p = B[j] + N[j] * r.uniform(-.5, .5) * w[j]; u = d * .9 - N[j] * np.sign(N[len(N) // 2, 1]) * r.uniform(.05, .35)
        c.line(Q(np.array([p, p + u * L * .5, p + u * L + (0, 6)])), r.uniform(3, 8) * k, ['#a9bdf6', '#8094dc', '#4c5cae', '#c6d6ff'][int(r.integers(4))], taper=(.2, .85), press=.4, seed=i)


def front_locks(img, t=0., dim=1.):
    """a few ends of his fringe hanging between us and the eye, out of focus, rimmed by its light"""
    c = Cel()
    for k, (tip, root, wd, bow) in enumerate((((210, 330), (-60, -420), 330, .10), ((470, 150), (300, -420), 210, -.10), ((760, 70), (700, -420), 150, .08), ((1420, 110), (1500, -420), 220, .10), ((1770, 330), (1960, -420), 360, -.10))):
        tip = np.array(tip, float) + (9 * math.sin(t * 1.6 + k), 5 * math.sin(t * 1.1 + k * 2)); root = np.array(root, float); u = np.linspace(0, 1, 26)[:, None]; d = tip - root; n_ = np.array([-d[1], d[0]]); C = root + d * u + n_ * bow * np.sin(np.pi * u); T = np.gradient(C, axis=0); T /= np.hypot(*T.T)[:, None]; M = np.c_[-T[:, 1], T[:, 0]]; ww = wd * .5 * (1 - u[:, 0] ** 1.6)
        A, D = C + M * ww[:, None], C - M * ww[:, None]; P = np.vstack([A, D[::-1]])
        c.grad(P, root, tip, [(0, '#2b3472', 1), (.55, '#6f82cc', 1), (1, '#c9dcff', 1)]); c.fill(np.vstack([A[6:], (A + (C - A) * .8)[6:][::-1]]), '#47549e', .8); c.line((D + M * 8)[10:], 9, '#dff4ff', taper=(.5, .1), press=.4, seed=k)
    L = c.layer(); L = cv2.resize(cv2.GaussianBlur(cv2.resize(L, (W // 2, HH // 2), interpolation=cv2.INTER_AREA), (0, 0), 6.5), (W, HH)); L[..., :3] *= dim; return comp.over(img, L, .96)


def key2(f=0, open_=1.0, t=1.2, pupil=.27, glow=.14, cen=(842, 612), sc=1.22, roll=-.075, refr=0., burn=0., cloth_dy=0., gaze=(0., 0.), shards=1., bare=False, sway=0.):
    from chars.gojo_six import six_eye, aperture
    L = six_eye(W, HH, open_, t, pupil, glow, gaze=gaze, refr=refr, centre=cen, scale=sc, roll=roll, mid=lambda c, Q, k: key2_hair(c, Q, k, cloth_dy, sway) if bare else key2_cloth(c, Q, k, cloth_dy))
    img = L[..., :3]; A = aperture(W, HH, open_, cen, sc, roll); lamp = float(smooth(.0, .9, open_))
    ic = (cen[0] + 2 + gaze[0] * 286 * sc, cen[1] - 8 + gaze[1] * 286 * sc); R = 286 * sc
    x, y = grid(); d = np.sqrt(((x - ic[0]) / 1.35) ** 2 + (y - ic[1]) ** 2); dr = np.sqrt((x - ic[0]) ** 2 + (y - ic[1]) ** 2)
    # the eye is the only lamp: fall off to night at the edges, cool
    fall = 1 - .80 * smooth(380, 1000, d); img = img * fall[..., None] + rgb('#04051a') * (1 - fall[..., None]) * .6
    img = prism(img, ic[0], ic[1], R * .80, R * 1.04, .006 + .016 * refr)
    m = (1 - smooth(R * .96, R * 1.04, dr)) * A
    rays = god_rays(img, ic[0], ic[1] + 20, .46, 1.5, mask=m)
    out = smooth(R * .86, R * 1.25, dr)[..., None]
    img = img + rays * (1.25 * out + .10) * (1 - .72 * A[..., None])
    st = radial_streaks(ic[0], ic[1], 4, r0=R * 1.0, r1=1250); img = img + st[..., None] * rgb('#5fb6ff') * .20 * out * lamp
    img = comp.screen(img, comp.radial(ic[0], ic[1] + 60, 1000, 700, power=2.4)[..., None] * rgb('#1f5cff') * (.10 + .20 * lamp))
    img = img + comp.radial(ic[0], ic[1] + R * .35, R * .9, R * .7, power=2.0)[..., None] * rgb('#8fe6ff') * .14 * lamp
    # anamorphic glint across the iris
    gl = np.exp(-np.abs(y - ic[1] + (x - ic[0]) * .075) / 3.2) * np.exp(-np.abs(x - ic[0]) / 520); img = img + gl[..., None] * rgb('#bfeaff') * .22 * lamp
    # bring the iris back: saturation and depth inside the ring
    mi = ((1 - smooth(R * .97, R * 1.01, dr)) * A)[..., None]; lo = img.mean(-1, keepdims=True); pop = lo + (img - lo) * 1.30; pop = np.clip(pop, 0, 2); pop = pop * (1 - .30 * (1 - smooth(.0, .62, lo)))
    ey_ = (L[..., 3:4] * 0 + 1); img = img * (1 - mi) + pop * mi
    for sd, n, rm, rx, s0, s1, bl in ((21, 34, 380, 1150, 10, 34, 0.), (22, 16, 420, 1200, 26, 64, 6.), (23, 7, 560, 1250, 60, 130, 15.)):
        if shards <= 0: break
        cl = crystals(sd, ic[0], ic[1], int(n * min(1., shards)), rm, rx, s0, s1, f, bl, drift=2.4); img = img + comp.blur(cl, 9)[..., :3] * .9; img = comp.over(img, cl, .92 if bl < 10 else .55)
    if burn > 0:                                   # the light eats the picture from the pupil outward
        g = comp.radial(ic[0], ic[1], R * (.8 + 5.5 * burn), power=1.2)[..., None]; img = img + g * rgb('#dff6ff') * 2.6 * burn + burn ** 3 * 1.2
    return img


# ------------------------------------------------------------------------------------------------ KEY 3: bare eyes from below, violet orb
def tower_sky(seed=3, vx=1180, vy=-900):
    """night city seen from the street, looking up: towers lean to a vanishing point above the frame, violet storm in the gap"""
    x, y = grid(); n = fbm(HH, W, (420, 170, 70, 26), None, seed, aspect=1.8)
    d = np.sqrt(((x - 960) / 1100) ** 2 + ((y - 420) / 760) ** 2)
    sky = ramp(np.clip(d * .9 + n * .10, 0, 1), [(0, '#8a5fe0'), (.22, '#5a35b4'), (.5, '#2c1a68'), (.8, '#130c34'), (1, '#0a0720')])
    cv = Canvas(HH, W, img=sky); r = np.random.default_rng(seed)
    gx, gy = r.uniform(0, W, 1500), r.uniform(0, HH, 1500); ii = np.clip(gy.astype(int), 0, HH - 1), np.clip(gx.astype(int), 0, W - 1)
    col = sky[ii] * (1 + .35 * (n[ii] > .5)[:, None] - .25 * (n[ii] < -.5)[:, None]); ang = np.arctan2(gy - 380, gx - 960) + math.pi / 2 + r.normal(0, .25, 1500)
    cv.strokes(arcs(gx, gy, ang, r.uniform(90, 260, 1500), r.normal(0, .5, 1500)), r.uniform(8, 26, 1500), col, op=.55, dry=.4, seed=seed)
    img = cv.img
    c = Cel()
    def tower(x0, x1, ybot, ytop, col, side, win, seed_):
        # a slab whose verticals converge to (vx, vy)
        f = lambda X, t: np.array([X + (vx - X) * t, ybot + (vy - ybot) * t]); t = (ybot - ytop) / (ybot - vy)
        a, b, cc, dd = f(x0, 0), f(x1, 0), f(x1, t), f(x0, t)
        c.fill(np.array([a, b, cc, dd]), col); rr = np.random.default_rng(seed_)
        w = (x1 - x0); s0 = x1 if side > 0 else x0; s1 = s0 + side * w * .34
        c.fill(np.array([f(s0, 0), f(s1, 0), f(s1, t * .97), f(s0, t)]), '#0b0720' if side > 0 else '#24124e')
        for k in range(int(win)):
            for q in range(int(max(4, w / 24))):
                if rr.random() < .24:
                    u0 = (q + .25) / max(4, int(w / 24)); u1 = u0 + .5 / max(4, int(w / 24)); t0 = t * (k + .25) / win; t1 = t0 + t * .40 / win
                    X0, X1 = x0 + w * u0, x0 + w * u1
                    c.fill(np.array([f(X0, t0), f(X1, t0), f(X1, t1), f(X0, t1)]), ['#e8c27a', '#b9a6ff', '#8fd0ff', '#f2e2b4'][int(rr.integers(4))], rr.uniform(.25, .8))
        c.line(np.array([a, dd]), 3, '#07041a'); c.line(np.array([b, cc]), 3, '#07041a'); c.line(np.array([dd, cc]), 4, '#b48cff', taper=(.2, .2))
    tower(-520, 120, 1500, 120, '#1a1040', 1, 40, 1); tower(60, 360, 1500, 420, '#22154e', 1, 30, 2); tower(1560, 1860, 1500, 330, '#1e1248', -1, 34, 3); tower(1760, 2500, 1500, 40, '#150c38', -1, 44, 4); tower(330, 520, 1500, 610, '#2a1a5c', 1, 24, 5); tower(1420, 1600, 1500, 560, '#281858', -1, 24, 6)
    L = c.layer(); return comp.over(img, comp.blur(L, 2.0))


def violet_orb(img, x, y, r, f=0, seed=2):
    """Hollow Purple in the hand: hard white heart, violet shell, black-violet lensing ring, torn arcs"""
    img = img + comp.radial(x, y, r * 5.2, power=2.6)[..., None] * rgb('#8a3cff') * .85
    img = img + comp.radial(x, y, r * 2.4, power=1.8)[..., None] * rgb('#c88cff') * .6
    c = Cel(); rr = np.random.default_rng(seed * 31 + f // 2); q = np.linspace(0, 2 * np.pi, 90, endpoint=False); cc = np.c_[np.cos(q), np.sin(q)]
    nz = lambda k, a: 1 + a * (np.sin(q * k + f * .7 + seed) * .6 + np.sin(q * (k * 2 + 1) - f * .9) * .4)
    c.fill(cc * (r * 1.12 * nz(5, .03))[:, None] + (x, y), '#2a0a66', .9); c.fill(cc * (r * nz(4, .025))[:, None] + (x, y), '#8a35f0'); c.fill(cc * (r * .84 * nz(6, .03))[:, None] + (x + r * .05, y - r * .04), '#c58bff'); c.fill(cc * (r * .62 * nz(5, .04))[:, None] + (x + r * .07, y - r * .05), '#f3e2ff'); c.fill(cc * (r * .46)[None, None] + (x + r * .08, y - r * .06), '#ffffff') if False else c.fill(cc * r * .46 + (x + r * .08, y - r * .06), '#ffffff')
    for k in range(22):
        a = rr.uniform(0, 6.28); sp = rr.uniform(.5, 1.5); R0 = r * rr.uniform(.9, 1.9); th = np.linspace(a, a + sp, 20); P = np.c_[np.cos(th), np.sin(th) * .92] * (R0 * np.linspace(1, rr.uniform(.8, 1.25), 20))[:, None] + (x, y) + rr.normal(0, r * .02, (20, 2))
        c.line(P, rr.uniform(2, 7) * r / 150, ['#f6e6ff', '#c79bff', '#7a2cf0'][int(rr.integers(3))], taper=(.5, .5), press=.5, seed=k)
    for k in range(9):
        a = rr.uniform(0, 6.28); P = [np.array([x, y]) + r * .9 * np.array([math.cos(a), math.sin(a)])]
        for j in range(6): a += rr.normal(0, .5); P.append(P[-1] + r * rr.uniform(.16, .40) * np.array([math.cos(a), math.sin(a)]))
        c.line(np.array(P), 4.5 * r / 150, '#f0dcff', taper=(.05, .7), press=.5, seed=k + 40)
    L = c.layer(); img = comp.screen(img, comp.blur(L, 14)[..., :3] * .9); return comp.over(img, L)


HANDV = dict(skin='#efc6f4', shade='#6a479e', line='#160c26', nail='#f8dcff', nail_shade='#8a5cc0', light='#fff0ff')


def key3(f=0):
    from jj import fx, hands as HD
    img = tower_sky(); x, y = grid()
    cx, cy, s = 1120, 462, 282; P = dict(yaw=.22, pitch=.24, roll=.06, persp=.06)
    L = Gojo.bust(cx, cy, s, state='wind', expr=dict(band=False, mouth='half', lid='cold', brow='cold', iris=.086, pupil=.24, glow=.12, gaze=(-.020, -.006)), light='violet', body=(.28, .16, .04), shift=(-.10, .03), seed=4, **P); L[:, :, :3] *= 1.0
    ox, oy, r = 452, 690, 128
    img = img * .80 + comp.radial(ox, oy, 1300, 1000, power=2)[..., None] * rgb('#7a2cf0') * .34
    st = radial_streaks(ox, oy, 8, r0=180, r1=1700); img = img + st[..., None] * rgb('#a86cff') * .14
    al = L[..., 3]; img = comp.screen(img, comp.blur(al, 46)[..., None] * rgb('#b06cff') * .5)
    L = rim(L, -5, 3, '#9fdcff', .85, soft=.7, where=smooth(cx - 60, cx + 120, x))
    L = rim(L, 6, -4, '#f6dcff', .80, soft=.8, where=1 - smooth(cx - 200, cx + 40, x))
    L = kit.para(L, (ox + 40, oy + 60), (cx + 420, 120), [(0, '#ffffff'), (.42, '#f4e8ff'), (.75, '#b7a2e8'), (1, '#8a78c6')], 'mul')
    L = kit.para(L, (ox, oy), (ox + 640, oy - 360), [(0, '#9a4cff'), (.6, '#1c0838'), (1, '#000000')], 'screen', .6)
    img = comp.over(img, L)
    img = eye_glow(img, gojo_head().pose(cx, cy, s, P['yaw'], P['pitch'], P['roll'], P['persp']), .42)
    # the hand that holds it: palm up, fingers spread round the weight
    c = Cel()
    GC.sleeve(c, (120, 1500), (330, 1330), (432, 1010), 190, 150, 92, PAL['violet'], s, seed=3, lit=-1,
              hand_fn=lambda: HD.draw(c, 'palm_up', HD.Cam(452, 872, 165), pal=HANDV, side='R', R=HD.euler(196, 50, 4), bld=HD.SLIM, w=5.2, light=(.1, .95, .4), seed=2))
    hl = c.layer(); hl = rim(hl, 3, 6, '#ffe8ff', .9, soft=.8); hl = kit.para(hl, (ox, oy), (ox - 40, 1080), [(0, '#ffffff'), (.5, '#c9a6f2'), (1, '#6a4aa8')], 'mul')
    img = comp.over(img, hl)
    img = fx.put(img, fx.orb_purple(ox, oy, r, f, seed=2))
    rr = np.random.default_rng(9); c = Cel()
    for k in range(90):
        a = rr.uniform(0, 6.28); d = rr.uniform(260, 1100); p = np.array([ox + math.cos(a) * d * 1.3, oy + math.sin(a) * d * .85])
        if 860 < p[0] < 1420 and 180 < p[1] < 720: continue
        t = np.array([ox, oy]) - p; t /= np.hypot(*t); n_ = np.array([-t[1], t[0]]); ln = rr.uniform(10, 70) * (1.2 - d / 1400)
        q = p + t * ln + n_ * ln * .22; c.line(np.array([p, (p + q) / 2 + n_ * ln * .06, q]), rr.uniform(1.2, 4.2), ['#f3e0ff', '#c49bff', '#8ec8ff'][int(rr.integers(3))], taper=(.6, .3), seed=k, alpha=.9)
    S_ = c.layer(); img = comp.screen(img, comp.blur(S_, 5)[..., :3] * .9); img = comp.over(img, S_, .9)
    img = img + comp.radial(ox, oy, 560, power=2.4)[..., None] * rgb('#a45cff') * .22
    return img


# ------------------------------------------------------------------------------------------------ model sheet
def sheet():
    CW, CH = 640, 600; out = np.zeros((CH * 5, CW * 4, 3), f32); lab = []
    def cell(k, name, bg, fn):
        r, c = divmod(k, 4); base = np.ones((CH, CW, 3), f32) * rgb(bg); L = fn(); im = comp.over(base, L) if L.shape[2] == 4 else L
        out[r * CH:(r + 1) * CH, c * CW:(c + 1) * CW] = im; lab.append((c * CW + 10, (r + 1) * CH - 12, name))
    b = lambda yaw, pitch, roll, st, ex, li, **k: (lambda: Gojo.bust(k.pop('cx', 320), k.pop('cy', 300), k.pop('s', 138), yaw, pitch, roll, st, ex, li, seed=k.pop('seed', 3), w=CW, h=CH, **k))
    n = 0
    for name, bg, fn in (
        ('blindfold, half smile', '#7fa9c8', b(0, -.10, -.03, 'up', 'smirk', 'day')), ('3/4', '#7fa9c8', b(.50, -.04, .03, 'up', 'smirk', 'day', body=(.35, 0, 0))),
        ('profile', '#7fa9c8', b(-1.35, 0, 0, 'up', 'smile', 'day', body=(-.5, 0, 0))), ('from below, looking down', '#3a4468', b(-.25, .34, .06, 'up', 'smirk', 'night', cy=320, body=(-.3, .2, 0))),
        ('laugh', '#c8a987', b(.2, .06, .10, 'up', 'laugh', 'day')), ('lifting the blindfold', '#27305c', b(-.12, -.05, -.05, 'up', 'lift', 'blue', key='L')),
        ('bare, cold', '#27305c', b(.10, -.14, .0, 'down', 'cold', 'night')), ('sealed: sad almost-smile', '#1d2420', b(-.10, -.02, -.06, 'down', 'sad', 'under')),
        ('mania, Hollow Purple', '#1a0f3c', b(-.18, -.16, .10, 'wind', 'mania', 'violet')), ('bare smirk, wind', '#1a0f3c', b(.32, .10, .05, 'wind', 'bare', 'violet')),
        ('red key', '#2a0a12', b(.25, .06, .0, 'up', 'smirk', 'red')), ('blue key', '#0a1430', b(-.25, .06, .0, 'up', 'smirk', 'blue')),
        ('void', '#05050a', b(.0, -.05, .0, 'up', 'smirk', 'void')), ('void, bare', '#05050a', b(.3, -.05, .0, 'wind', 'cold', 'void')),
        ('crossed fingers', '#7fa9c8', b(.06, -.06, .03, 'up', 'smirk', 'day', cx=360, cy=250, s=112, left=False, over=Gojo.arm('cross'))), ('arms open (orbs)', '#1a1440', b(0, .08, 0, 'wind', 'bare', 'night', cx=320, cy=230, s=72, left=False, right=False, arms=None, under=Gojo.arm('open', 'night'))),
        ('eyes shut', '#1d2420', b(.0, -.12, .05, 'down', 'shut', 'under')), ('joy', '#c8a987', b(-.2, .04, -.08, 'down', 'joy', 'day')), ('manga ink', '#efece8', b(-.45, -.05, -.03, 'up', 'plain', 'manga', body=(-.3, 0, 0))), ('day, bare', '#7fa9c8', b(.0, -.04, .02, 'down', 'bare', 'day'))):
        cell(n, name, bg, fn); n += 1
    a = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    for x, y, t in lab: cv2.putText(a, t, (x, y), 0, .55, (0, 0, 0), 4, cv2.LINE_AA); cv2.putText(a, t, (x, y), 0, .55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite('qa/gojo/sheet.jpg', a[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])


if __name__ == '__main__':
    for a in sys.argv[1:]:
        if a == '2': _save('key2', key2(), bloom=.30, vig=.20, halation='#cfe4ff')
        if a == '3': _save('key3', key3(), bloom=.32, vig=.2, halation='#e6d0ff')
        if a == 'sheet': sheet()
        if a == '1': _save('key1', key1(), bloom=.28, vig=.18)
