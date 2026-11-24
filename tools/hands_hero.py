"""python tools/hands_hero.py pray|cross|fire  -> qa/hands/hero_*.jpg   full-finish stills that show how to use jj/hands.py"""
import sys, os; ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, cv2, math, time
cv2.setNumThreads(4)
from jj import W, H, hands as HD, comp, kit
from jj.paint import rgb, mix, smooth, fbm, f32, grid, ramp, Canvas
from jj.ink import Cel, spl
E = HD.euler


def brushmass(im, paths, cols, wid, seed, dry=(.35, .7), op=.92):
    """a mass painted with many dry strokes that follow a family of curves"""
    r = np.random.default_rng(seed)
    for p in paths:
        p = spl(np.array(p, float), n=18); n = len(p); t = np.linspace(0, 1, n)
        T = np.gradient(p, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-6; N = np.c_[-T[:, 1], T[:, 0]]
        for k in range(len(cols)):
            for q in range(5):
                o = r.uniform(-1, 1) * (1 - k / len(cols) * .75); i, e = sorted(r.integers(0, n, 2))
                if e - i < n * .35: i, e = 0, n
                w = wid * r.uniform(.35, .8) * (1 - .12 * k)
                P = p[i:e] + N[i:e] * (o * wid * np.sin(np.clip(t[i:e], .05, .97) * math.pi) ** .7)[:, None]
                im = comp.over(im, kit.drybrush(H, W, P, w, cols[k], seed=int(r.integers(1 << 30)), dry=r.uniform(*dry), bristle=.15), op)
    return im


def pray():
    x, y = grid(H, W); r = np.random.default_rng(5)
    im = np.ones((H, W, 3), f32) * rgb('#1b171b')
    for k in range(46):                                                   # charcoal ground laid with a wide dry brush
        a = r.uniform(-.5, .5) - .9; c_ = np.array([r.uniform(-100, W + 100), r.uniform(-100, H + 100)]); L_ = r.uniform(300, 900); d = np.array([math.cos(a), math.sin(a)])
        im = comp.over(im, kit.drybrush(H, W, [c_ - d * L_ / 2, c_ + [r.normal(0, 30), r.normal(0, 30)], c_ + d * L_ / 2], r.uniform(60, 190), ('#2b262b', '#39333a', '#121014', '#48424a', '#221d22')[k % 5], seed=k + 50, dry=r.uniform(.4, .8), bristle=.2), .75)
    reds = ['#3d0810', '#6f0e18', '#a3141f', '#c8202c']
    im = brushmass(im, [[(2000, 120), (1650, 200), (1380, 380), (1180, 640), (1080, 1000)], [(2000, 420), (1700, 470), (1450, 620), (1260, 900)]], reds, 150, 3)
    im = brushmass(im, [[(-80, 60), (220, 170), (470, 360), (660, 620), (720, 900)]], reds[:3], 150, 8)
    im = brushmass(im, [[(1960, 190), (1660, 290), (1430, 460), (1250, 690)]], ['#e24a4e', '#f07a74'], 46, 12, (.5, .8), .85)
    for k in range(30):                                                   # ink slashes out of the hands
        a = r.uniform(-math.pi, math.pi); cx, cy = 1010, 420; r0, r1 = r.uniform(420, 620), r.uniform(800, 1500)
        p = [(cx + math.cos(a + s_ * .07) * q, cy + math.sin(a + s_ * .07) * q * .9) for s_, q in ((0, r0), (r.normal(0, 1), (r0 + r1) / 2), (r.normal(0, 1), r1))]
        im = comp.over(im, kit.drybrush(H, W, spl(p, n=16), r.uniform(8, 58), '#09070a', seed=k, dry=r.uniform(.4, .75), bristle=.2), .95)
    c = Cel(); kit.splatter(c, 760, 330, 520, 170, 10, '#09070a', 3); kit.splatter(c, 1560, 300, 320, 50, 7, '#a3141f', 4); kit.splatter(c, 420, 820, 260, 40, 8, '#cfc4b4', 6); im = comp.over(im, c.layer())
    tone = kit.screentone(np.ones((H, W), f32), 8, .785, np.clip(.16 - .0002 * y, 0, .2)); im = im * (1 - .30 * tone[..., None])
    # the hands
    light = (-.70, .25, .50)
    cam = HD.Cam(985, 500, 372, 4.6)
    hs = HD.two('pray', R=E(-24, -17, -7), bld=HD.SLIM, nails='dark', bands=((.17, .28), (.44, .55)))
    g = HD.render(hs, cam, light=light, res=900, clip=(0, 0, W, H), thr=.16)
    cel = Cel()
    pal = dict(HD.PAL['day'], **HD.PAL['ink']); pal.update(skin='#ecd7ba', shade='#b08f7e', nail='#1c1216', nail_dark='#120e11')
    HD.fill_flat(cel, g, pal); HD.fill_shadow(cel, g, pal)
    deep = HD.depth_polys(g, light, -.22); HD.fill_polys(cel, g, deep, '#7d5f5c')
    HD.fill_overlays(cel, g, pal, 8)
    HD.hatch(cel, g, deep, .024, 2.2, '#0c0a0d', 6, ang=.62, length=.22, fade=.35)
    HD.feather(cel, g, .022, .20, 2.4, seed=2, ang=.55, side=.15)
    HD.brush_lines(cel, g, 8.6, seed=3, hairs=3, marks=.8)
    L = cel.layer()
    L = kit.para(L, (960, 40), (1010, 1080), [(0, '#ffffff'), (.45, '#fff8ee'), (1, '#b08f92')], 'mul', .8)
    st = fbm(H, W, (60, 18, 5), seed=12, aspect=.16); L[..., :3] *= np.clip(.97 + .05 * st, .9, 1.03)[..., None]
    sh = kit.shifted(L, 26, 18)[..., 3:4]; im = im * (1 - sh * .75)
    im = comp.over(im, L)
    c = Cel(); kit.splatter(c, 1080, 420, 420, 34, 7, '#0a080b', 14); kit.splatter(c, 1290, 860, 150, 18, 8, '#7a0f1a', 15); im = comp.over(im, c.layer())
    return im.astype(f32), dict(bloom=.10, vig=.22)


def cross():
    x, y = grid(H, W)
    cx, cy = 1120, 300
    d = np.hypot((x - cx) / 1.25, y - cy)
    im = ramp(np.clip(d / 1250, 0, 1), [(0, '#f4fbff'), (.10, '#bfe6ff'), (.24, '#5fa6ee'), (.45, '#3b4fb4'), (.72, '#221d5c'), (1, '#0b0a20')])
    th = np.arctan2(y - cy, x - cx); r = np.random.default_rng(2)
    ray = np.zeros_like(x)
    for k in range(26):
        a = r.uniform(-math.pi, math.pi); wd = r.uniform(.006, .05); ray += r.uniform(.2, 1) * np.exp(-((((th - a + math.pi) % (2 * math.pi)) - math.pi) / wd) ** 2)
    im = im + (np.clip(ray, 0, 1.4) * smooth(1200, 180, d) * .30)[..., None] * rgb('#bfe0ff')
    im = comp.blur(im, 3)
    cam = HD.Cam(985, 520, 395, 6.5)
    hs = [HD.Hand('cross', 'R', E(158, -10, -5), bld=HD.SLIM, pivot=(0, .95, 0), cuff=dict(kind='uniform', y=.16, flare=1.2))]
    g = HD.render(hs, cam, light=(.85, .35, .05), res=820, clip=(0, 0, W, H), thr=.10)
    pal = dict(HD.PAL['day']); pal.update(skin='#cfd8f6', shade='#7a78b8', line='#141026', nail='#e2e9ff', nail_shade='#9391c8', cuff='#16142a', cuff_shade='#0b0a18', cuff_light='#4b41a0')
    cel = Cel()
    HD.fill_flat(cel, g, pal); HD.fill_shadow(cel, g, pal); HD.fill_overlays(cel, g, pal, 7)
    HD.fill_polys(cel, g, HD.depth_polys(g, (.85, .35, .05), -.38), '#544f94', names=HD.FING + ('thumb', 'palm'))
    SK = HD.FING + ('thumb', 'palm'); rim = HD.rim_polys(g, (.9, .45, -.35), .18, .62); HD.fill_polys(cel, g, rim, '#f4fbff', names=SK)
    rim2 = HD.rim_polys(g, (-.95, .1, -.35), .30, .50); HD.fill_polys(cel, g, rim2, '#7fd6ff', names=HD.FING); HD.fill_polys(cel, g, HD.rim_polys(g, (.9, .3, -.5), .62, .34, .008), '#7f78d8', names=('cuff',))
    HD.ink_lines(cel, g, pal, 6.6, seed=4)
    L = cel.layer()
    L = kit.para(L, (1300, 120), (760, 1060), [(0, '#ffffff'), (.4, '#e6e4ff'), (1, '#6d63b0')], 'mul', .85)
    im = kit.aura(im, L, 34, '#9ad8ff', .55)
    im = comp.over(im, L)
    im = kit.flare(im, 1120, 300, 620, '#9fd4ff', .32, streak=.5, ang=-.12)
    c = Cel()                                                             # drifting motes
    for k in range(70):
        p = np.array([r.uniform(0, W), r.uniform(0, H)]); q = r.uniform(.8, 3.2); c.fill(spl(p + q * np.c_[np.cos(np.linspace(0, 5.6, 7)), np.sin(np.linspace(0, 5.6, 7))] * r.uniform(.7, 1.2, (7, 1)), True, n=3), '#dff2ff', r.uniform(.3, .9))
    im = comp.over(im, c.layer())
    return im.astype(f32), dict(bloom=.42, vig=.20)


def ribbon_flame(cel, path, wid, col, seed, licks=14, lean=1.5, size=1.0, n=120):
    """one hard-edged sheet of fire: a swelling ribbon whose edges tear into long tongues"""
    from jj.ink import noise1
    r = np.random.default_rng(seed); p = spl(np.array(path, float), n=24); s = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]; t = np.linspace(0, s[-1], n)
    p = np.c_[np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])]; u = t / s[-1]
    T = np.gradient(p, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None]; N = np.c_[-T[:, 1], T[:, 0]]
    w = wid * np.sin(np.clip(u * .96 + .04, 0, 1) * math.pi) ** .55 * (1 + .32 * noise1(n, 16, r)); w2 = wid * np.sin(np.clip(u * .96 + .04, 0, 1) * math.pi) ** .55 * (1 + .32 * noise1(n, 16, r))
    A, B = p + N * w[:, None], p - N * w2[:, None]
    def teeth(E, sg):
        out = [E[0]]; i = 4
        while i < n - 6:
            L = wid * size * r.choice([.35, .7, 1.2, 2.2], p=[.3, .35, .25, .1]) * (.3 + math.sin(u[i] * math.pi)); m = int(r.integers(4, 11)); e = min(n - 2, i + m)
            tip = E[e] + (T[e] * lean + N[e] * sg * r.uniform(.15, .75)) * L
            a, b = E[i], E[min(n - 1, e + 1)]
            c1 = a + (tip - a) * .45 + N[i] * sg * L * .10; c2 = b + (tip - b) * .5 - T[e] * L * .10 + N[e] * sg * L * .16
            q = np.linspace(0, 1, 9)[:, None]
            out += list((1 - q) ** 2 * a + 2 * (1 - q) * q * c1 + q * q * tip) + list((1 - q) ** 2 * tip + 2 * (1 - q) * q * c2 + q * q * b); i = e + int(r.integers(1, 6))
        out.append(E[-1]); return np.array(out)
    cel.fill(np.vstack([teeth(A, 1), teeth(B, -1)[::-1]]), col)
    for k in range(licks // 3):                                             # torn-off pieces that fly along
        i = int(r.uniform(.2, .95) * n); sg = 1 if r.random() < .5 else -1; c_ = p[i] + N[i] * sg * wid * r.uniform(1.3, 2.4) + T[i] * wid; L = wid * r.uniform(.3, .9)
        cel.fill(kit.spike(c_ - N[i] * L * .13, c_ + T[i] * L * 1.2 + N[i] * sg * L * .3, c_ + N[i] * L * .13, .2, .1), col)


def fire():
    x, y = grid(H, W); r = np.random.default_rng(8)
    d = np.hypot((x - 1330) / 1.5, y - 760)
    im = ramp(np.clip(d / 1050 + .06 * fbm(H, W, (300, 120), seed=2), 0, 1), [(0, '#ffb347'), (.18, '#f0722a'), (.42, '#bf2f1c'), (.72, '#5c1216'), (1, '#1a080c')])
    im = comp.blur(im, 6)
    def sheet(paths, wid, seed, blur_, k=1.0):
        out = None
        for (c1, f, e) in (('#c8371a', 1.0, 0), ('#f58a2b', .66, 1), ('#ffd45f', .36, 2), ('#fff3bd', .15, 3)):
            c = Cel()
            for q, pth in enumerate(paths): ribbon_flame(c, pth, wid * f, c1, seed + q * 10 + e * 3, licks=int(13 * k), size=1.0)
            L = c.layer(); L = comp.blur(L, blur_) if blur_ else L
            out = L if out is None else out * (1 - L[..., 3:4] * (out[..., 3:4] > .5)) + L * (out[..., 3:4] > .5)
        return out
    back = sheet([[(2050, 1000), (1500, 900), (1050, 640), (760, 250), (640, -120)], [(2050, 560), (1650, 520), (1340, 330), (1240, -80)]], 110, 4, 4)
    im = comp.over(im, back, .92); im = im + comp.blur(back[..., :3] * back[..., 3:4], 30) * .22
    light = (.55, -.55, .40)
    cam = HD.Cam(990, 640, 372, 1.82)
    hs = [HD.Hand('thrust', 'L', E(166, 66, 26), pos=(.05, 0, .02), bld=HD.HEAVY, nails='dark', bands=((.14, .25), (.36, .47)), pivot=(0, .9, 0))]
    hs[0] = HD.Hand(dict(HD.POSES['thrust'], wrist=(34, -22)), 'L', E(166, 66, 26), pos=(.05, 0, .02), bld=HD.HEAVY, nails='dark', bands=((.14, .25), (.36, .47)), pivot=(0, .9, 0))
    g = HD.render(hs, cam, light=light, res=1000, clip=(0, 0, W, H), thr=.20)
    pal = dict(HD.PAL['day'], **HD.PAL['fire']); pal.update(skin='#ffc766', shade='#8f3f24', nail_dark='#1f070a', line='#250709', band='#1f070a')
    cel = Cel(); HD.fill_flat(cel, g, pal); HD.fill_shadow(cel, g, pal)
    HD.fill_polys(cel, g, HD.depth_polys(g, light, -.28), '#4f1a17')
    HD.fill_overlays(cel, g, pal, 7)
    HD.fill_polys(cel, g, HD.depth_polys(g, tuple(-np.array(light)), -.80), '#fff0b4')
    HD.fill_polys(cel, g, HD.rim_polys(g, (-.9, .35, -.35), .36, .46), '#ff6a2a')
    HD.ink_lines(cel, g, pal, 6.6, seed=5)
    L = cel.layer(); L = kit.para(L, (1400, 1060), (640, 40), [(0, '#ffffff'), (.5, '#ffe0c0'), (1, '#a8483a')], 'mul', .8)
    im = kit.aura(im, L, 22, '#ff8a3a', .30); im = comp.over(im, L)
    fr = sheet([[(2100, 1130), (1650, 1010), (1300, 1040), (900, 1150)], [(1980, 1150), (1880, 900), (1930, 640), (1820, 380)]], 70, 40, 0, 1.2)
    im = comp.over(im, fr); im = im + comp.blur(fr[..., :3] * fr[..., 3:4], 22) * .45
    c = Cel()
    for k in range(150):
        p = np.array([r.uniform(0, W), r.uniform(0, H)]); q = r.uniform(.8, 3.2) * (1 + 2.2 * (r.random() < .08)); a = r.normal(-2.3, .35); e = np.array([math.cos(a), math.sin(a)])
        c.fill(np.array([p - e * q * 2.4, p + [e[1] * q * .5, -e[0] * q * .5], p + e * q * 1.6, p - [e[1] * q * .5, -e[0] * q * .5]]), '#ffd98a' if r.random() < .6 else '#ff8a3a')
    f = c.layer(); im = comp.over(im, f); im = im + comp.blur(f[..., :3], 8) * .5
    return im.astype(f32), dict(bloom=.36, vig=.24)


if __name__ == '__main__':
    os.makedirs('qa/hands', exist_ok=True)
    for k in sys.argv[1:] or ['pray', 'cross', 'fire']:
        t = time.time(); im, fin = globals()[k](); im = comp.finish(im, 3, **fin); comp.save(f'qa/hands/hero_{k}.jpg', comp.to8(np.clip(im, 0, 1), 3), 92); print(k, round(time.time() - t, 1))
