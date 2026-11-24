"""python tools/fx_qa.py [sheet|strips|hero|cmp|time|all]   effect-kit QA pictures -> qa/fx/"""
import sys, os, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, cv2
cv2.setNumThreads(4)
from jj import fx, comp
from jj.fx import *
from jj.paint import ramp
x, y = grid(); os.makedirs('qa/fx', exist_ok=True)
def G(*st, d=(0., 1.)): return ramp(np.clip((x / W * d[0] + y / H * d[1]), 0, 1), list(st))
NIGHT = G((0, '#0d1226'), (.6, '#262c4c'), (1, '#10101c')); DAY = G((0, '#9cb4d4'), (.39, '#e6d8c4'), (.4, '#8d8078'), (1, '#5d5058')); EMB = G((0, '#c8420f'), (.4, '#d23a0a'), (.75, '#8e1e08'), (1, '#2a0605'), d=(.23, .77)); PAPER = np.ones((H, W, 3), f32) * C('#f3ecdf'); WET = G((0, '#28304a'), (.6, '#46506e'), (.64, '#2c3144'), (1, '#1c1f2c')); SNOWY = G((0, '#7d8fb8'), (.7, '#c9cfe0'), (1, '#e6e8f0')); SPRING = G((0, '#a8cbe8'), (.7, '#e4eadf'), (1, '#a8c48a')); RUBY = G((0, '#3a0710'), (.5, '#7e0d1c'), (1, '#20040a'))
_c = Cel(); _c.fill(np.array([[0, 600], [700, 480], [980, 470], [1000, 560], [960, 640], [700, 650], [0, 900]]), '#c98f70'); ARM = _c.layer(); _c = Cel(); _c.fill(spl(np.array([[1010, 200], [1500, 150], [1800, 400], [1700, 900], [1200, 950], [1040, 700], [1080, 520]]), True), '#4a4064'); VIC = _c.layer(); DUEL = put(put(NIGHT, VIC), ARM); SUB = ARM + VIC * (1 - ARM[..., 3:4])
FIG = limb_poly(spl([(700, 330), (690, 600), (680, 900)], n=12), 90, 110); LIMB = limb_poly(spl([(1150, 800), (1400, 660), (1620, 480)], n=12), 74, 50)
GP = ground_proj(hy=420, cx=1000, cy=760); CITY = put(NIGHT, (lambda c, r: [c.fill(np.array([[X, H], [X, H - h_], [X + w_, H - h_], [X + w_, H]]), '#0c0e1c') for X, w_, h_ in zip(r.uniform(0, W, 40), r.uniform(40, 120, 40), r.uniform(100, 460, 40))] and c.layer())(Cel(), np.random.default_rng(2)))
def stream(f):
    c = Cel(); flame_stream(c, spl([(-80, 180), (500, 320), (1050, 560), (1700, 1000)], n=20), 105, f / 24., 3); flame_stream(c, spl([(1900, 80), (1760, 380), (1640, 700), (1600, 980)], n=20), 70, f / 24., 4); l = c.layer(); return put(EMB, FX(l, hot(l, .5, (6, 26, 80), (.3, .3, .3), '#ff7a30', .5) * (1 - .8 * l[..., 3:4])))
def stand(f):
    c = Cel(); flames(c, [(60, 1040), (960, 1050)], 560, f / 24., 2, 9); flames(c, [(1080, 1040), (1860, 1050)], 520, f / 24., 5, 8, hot=True); l = c.layer(); return put(NIGHT, FX(l, hot(l, .5, (6, 26, 80), (.3, .3, .3), '#ff7a30', .6) * (1 - .8 * l[..., 3:4])))
def arrow(f):
    c = Cel(); fire_arrow(c, (1000, -260), (930, 1010), 78, f / 24., 2); l = c.layer(); return put(put(EMB, FX(l, hot(l, .5, (6, 26, 80), (.3, .3, .3), '#ff7a30', .5) * (1 - .8 * l[..., 3:4]))), embers(f, 60, 2))
def column(f): a, b = fire_column(f, 1000, 930, -140, 170, 3); return put(put(put(G((0, '#4a0f0a'), (.6, '#7a1c0a'), (1, '#180404')), a), b), embers(f, 80, 4))
def curse(f):
    c = Cel(); c.fill(FIG, '#15141f'); c.fill(LIMB, '#d8a27e'); L = c.layer(); a, b = cursed([FIG], f, (0, -1), 150, CURSE, 4); a2, b2 = cursed([LIMB], f, (-1.7, .4), 120, BLACKBLUE, 2); return put(put(put(put(put(NIGHT, a), a2), L), b), b2)
def flash(f): return black_flash(DUEL, f, 1005, 545, 1., 3, 0., subject=SUB, over=ARM)
def lily(f): c = Cel(); t = f / 36.; higanbana(c, 960, 620, 620, min(1, t), 3, stem=[(960, 1200), (930, 900), (960, 620)]); higanbana(c, 360, 340, 300, min(1, t * 1.1), 7, tilt=.8, rot=.4); higanbana(c, 1580, 330, 340, min(1, t * .95), 9, tilt=.6, rot=.8); return put(RUBY, c.layer())
def slash(f):
    LN = [(100, 900, 1800, 180), (300, 100, 1500, 1000), (0, 560, 1920, 640)]; im = CITY if f < 5 else cut_plate(CITY, LN, min(1, (f - 5) / 8.), 80, .05, 3); return put(im, slash_fx(LN, f, 4, 1, stagger=1))
def brk(f):
    c = Cel(); ground_crack(c, GP, 4.2, min(1, f / 9.), 5); smoke_burst(c, f, 1000, 740, 150, 3, 9, (-math.pi + .2, -.2), 620); debris(c, f, (1000, 740), 40, 3, (300, 1500), (-2.7, -.4), (4, 50), cubes=.15)
    if f < 14: shockwave(c, 1000, 760, f / 14., 60, 1200, .28, 30, 1)
    return put(DAY, c.layer())
def smoke(f): c = Cel(); smoke_burst(c, f, 700, 820, 230, 4, 7, (-2.7, -.5), 760, 2.2); dust_skirt(c, f, 1400, 850, 330, 90, 120, 3); return put(DAY, c.layer())
def glass(f): return shatter(BR, f / 24., 1150, 480, None, 3)
def orbs(f): return put(put(pinch(CITY, 480, 480, 420, .5, .8), orb_blue(480, 480, 110, f, 1, 2)), orb_red(1400, 480, 105, f, 1, 3))
def merge_(f): return put(CITY, purple_merge(960, 520, 170, f / 22., f, 700, 1))
def purple(f): P = [(1600, 60), (1200, 380), (700, 660)]; return put(purple_trail(CITY, P, 180, f, 1, persp=lambda v: .35 + .65 * v), orb_purple(700, 660, 170, f, 1, 2))
def void_(f): return void(f, seed=3)
def rainy(f): return put(put(WET, splashes(f, 80, 1)), rain(f))
def rainstop(f): tau = 1. + (min(f, 6) / 24. if f < 6 else 6 / 24. - max(0, f - 14) / 24. * .5); v = 1. if f < 4 else .35 if f < 6 else .02 if f < 15 else -.5; return put(WET, rain(tau=tau, v=v))
def snowy(f): return put(SNOWY, snow(f))
def petal(f): return put(SPRING, petals(f))
def bpetal(f): return put(RUBY, petals(f, 90, 3, BLACKPETAL, HIGAN, size=(12, 44), edge='#f0142e'))
def ashy(f): return put(G((0, '#4a4650'), (1, '#8a7f7c')), ash(f, 130))
def misty(f): return put(WET, mist(f))
def inked(f): c = Cel(); t = min(1, f / 20.); ink_burst(c, 560, 430, 330, t, 3); ink_runs(c, [(1050 + i * 40, 120) for i in range(20)], 620, t, 2); ink_pool(c, 1400, 920, 420, 110, t, 1, wet='#5a5060'); return put(put(PAPER, c.layer()), ink_wash(P=spl([(60, 960), (330, 820), (560, 900), (860, 780)], n=16), w=90, seed=3))
def curtain(f): c = Cel(); ink_curtain(c, f / 36., 3); return put(SPRING, c.layer())
def burn(f): return paper_burn(SPRING, f / 40., 2, under='#08060a', f=f)
BR = None
SHEET = [('fire stream', stream, 8), ('flames classic / hot', stand, 10), ('fire arrow', arrow, 6), ('fire column', column, 6), ('cursed energy', curse, 6), ('black flash f2', flash, 2), ('black flash f3', flash, 3), ('black flash f4', flash, 4), ('black flash f6', flash, 6), ('black flash f10', flash, 10), ('higanbana .45', lily, 16), ('higanbana 1', lily, 36), ('slashes', slash, 1), ('cut plate', slash, 12),
         ('crack dust rock', brk, 8), ('smoke', smoke, 12), ('shatter', glass, 11), ('blue / red', orbs, 5), ('merge', merge_, 13), ('hollow purple', purple, 6), ('unlimited void', void_, 20), ('rain', rainy, 10), ('rain stopped', rainstop, 10), ('snow', snowy, 30), ('petals', petal, 30), ('black petals', bpetal, 30), ('ash', ashy, 30), ('mist', misty, 30), ('ink', inked, 20), ('ink curtain', curtain, 18), ('paper burn', burn, 16)] + [(f'impact {k}', (lambda k: lambda f: impact_frame(k, 1100, 480, 5))(k), 0) for k in ('white', 'black', 'red', 'crack', 'burst')]
def lab(im, t, w=640): s = np.ascontiguousarray(comp.to8(np.clip(cv2.resize(im, (w, w * 9 // 16), interpolation=cv2.INTER_AREA), 0, 1))).copy(); cv2.putText(s, t, (8, w * 9 // 16 - 10), 0, .55, (255, 255, 0), 2); return s
def grid_(ims, n): ims = ims + [np.zeros_like(ims[0])] * (-len(ims) % n); return np.vstack([np.hstack(ims[i:i + n]) for i in range(0, len(ims), n)])
def strip(name, fn, fr, n=4): comp.save(f'qa/fx/strip_{name}.jpg', grid_([lab(comp.finish(fn(f), int(f)), f'{f}') for f in fr], n), 86); print('strip', name)
def cpu(fn, f):
    t = time.process_time(); w = time.time(); r = fn(f); return r, time.process_time() - t, time.time() - w
mode = sys.argv[1] if len(sys.argv) > 1 else 'all'; only = sys.argv[2:]
BR = comp.finish(brk(10), 0)
if mode in ('sheet', 'all'):
    ims = []
    for n, fn, f in SHEET:
        fn(f); r, c, w = cpu(fn, f); ims.append(lab(comp.finish(r, 0), '%s  %.2fs' % (n, c), 600)); print('%-22s cpu %.2f wall %.2f' % (n, c, w), flush=True)
    comp.save('qa/fx/sheet.jpg', grid_(ims, 6), 84)
ST = dict(fire=(stand, range(0, 24, 2)), column=(column, range(0, 12, 2)), stream=(stream, range(0, 12, 2)), flash=(flash, range(0, 20)), smoke=(smoke, range(0, 48, 4)), shatter=(glass, range(0, 24, 2)), rain=(rainstop, range(0, 24, 2)), lily=(lily, range(0, 48, 4)), curse=(curse, range(0, 16, 2)), merge=(merge_, range(0, 24, 2)), crack=(brk, range(0, 24, 2)))
if mode in ('strips', 'all'):
    for k, (fn, fr) in ST.items():
        if not only or k in only: strip(k, fn, list(fr))
if mode in ('hero', 'all'):
    from jj.fx_hero import hero_flash, hero_column
    comp.save('qa/fx/hero_flash.jpg', comp.finish(hero_flash(), 0, bloom=.34, vig=.2), 92); comp.save('qa/fx/hero_column.jpg', comp.finish(hero_column(), 0, bloom=.34, vig=.2), 92)
if mode in ('cmp', 'all'):
    comp.save('qa/fx/_stream.jpg', comp.finish(arrow(6), 0), 94); os.system(f'{sys.executable} tools/cmp.py qa/fx/cmp_fire.jpg ref/1.jpg 60,700,560,420 qa/fx/_stream.jpg 330,330,1150,740')
