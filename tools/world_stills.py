"""python tools/world_stills.py [name ...]   finished still per set -> qa/world/<name>.jpg and qa/world/sheet.jpg (painting is cached)"""
import sys, os, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); os.chdir(ROOT); os.environ['WORLD_VERBOSE'] = '1'
import numpy as np, cv2
cv2.setNumThreads(4)
from jj import comp, world
from jj.comp import over
os.makedirs('qa/world', exist_ok=True)


def base(P, cy=None):
    h, w = P.img.shape[:2]; return lambda im: comp.cam2d(im, w / 2, h / 2 if cy is None else cy, 1920. / w)


def lay(*names):
    def f(P):
        im = P.img
        for n in names: im = over(im, P.layers[n])
        return base(P)(im)
    return f


def hero1(P):
    from jj.world_city import Veil
    h, w = P.img.shape[:2]; L = Veil(P.cam, w, h).layer(.52); m = P.mask['sky'][..., None]; return base(P)(P.img * (1 - m * L[..., 3:]) + L[..., :3] * m)


def figure(im, P, xy, z, H=1.72, col='#141826', rim='#ffcf9a', scarf=None):
    """tiny standing silhouette with wind-blown coat, drawn in metres"""
    from jj.ink import Cel, spl
    p, d = P.cam.project(np.array([[xy[0], xy[1], z]])); k = float(P.cam.scale(np.array([xy[0], xy[1], z]))) * H; x, y = p[0]; h, w = im.shape[:2]
    def S(pts): return spl(np.array([(x + a * k, y - b * k) for a, b in pts]), True, n=6)
    c = Cel(w, h, ss=2) if w < 2000 else Cel(400, 400, ox=int(x - 200), oy=int(y - 300), ss=3)
    c.fill(S([(-.075, 0), (-.03, 0), (-.005, .30), (.0, .46), (-.10, .46), (-.095, .25)]), col); c.fill(S([(.02, 0), (.07, .0), (.085, .28), (.09, .46), (-.01, .46), (.0, .28)]), col)
    c.fill(S([(-.135, .40), (-.13, .62), (-.145, .78), (-.06, .835), (.06, .835), (.135, .78), (.125, .60), (.14, .42), (.21, .36), (.10, .40), (0, .39)]), col); c.fill(S([(-.15, .78), (-.185, .60), (-.16, .42), (-.125, .43), (-.13, .62)]), col); c.fill(S([(.14, .78), (.18, .60), (.165, .43), (.13, .44), (.125, .62)]), col); c.fill(S([(-.06, .84), (-.075, .92), (-.04, 1.0), (.04, 1.0), (.072, .92), (.055, .84)]), col)
    for a in ((-.07, .95, -.10, 1.03), (-.02, .99, -.03, 1.07), (.03, .99, .06, 1.06), (.06, .95, .11, 1.0)): c.fill(np.array([(x + (a[0] - .03) * k, y - a[1] * k), (x + a[2] * k, y - a[3] * k), (x + (a[0] + .03) * k, y - a[1] * k)]), col)
    L = c.rgba(); from jj import kit; L2 = kit.rim(np.pad(L, ((4, 4), (4, 4), (0, 0))), -2.2, .6, rim, .9, .4)[4:-4, 4:-4]; from jj.ink import over_roi; return over_roi(im.copy(), L2, c.ox, c.oy)


def hero2(P): return base(P)(figure(P.img, P, (-.9, -23.), .10))


SETS = [('scramble', 'scramble', {}, hero1), ('street', 'street', {}, base), ('skyline', 'skyline', {}, lay('near')), ('ruin_snow', 'ruin', dict(mode='snow', view='dawn', figure=(-.9, -23.)), hero2), ('ruin_fire', 'ruin', dict(mode='fire', view='low'), base), ('ruin_rain', 'ruin', dict(mode='rain', view='crater'), base),
        ('ruin_ash', 'ruin', dict(mode='ash', view='wide'), base), ('platform', 'platform', {}, base), ('platform_dark', 'platform', dict(lit=.1), base), ('hall', 'hall', {}, base), ('asphalt', 'asphalt', dict(wet=.35), base), ('hillroad', 'hillroad', {}, base), ('hillroad_sunset', 'hillroad', dict(time='sunset'), base),
        ('overlook', 'overlook', {}, None), ('bakery', 'bakery', {}, base), ('beach', 'beach', {}, lay('trunk', 'fronds')), ('shrine', 'shrine', {}, base)]
FIN = dict(scramble=dict(bloom=.34, vig=.10), ruin_snow=dict(bloom=.26, vig=.12), shrine=dict(bloom=.16, vig=.14, soft=.08), beach=dict(bloom=.4, vig=.04))
want = sys.argv[1:]
for (n, st, kw, f) in SETS:
    if want and n not in want: continue
    t = time.time(); P = world.plate(st, **kw)
    if f is None: im = P.img; L = P.layers['rail']; im = comp.cam2d(over(im, L), 1200, 1120 - 60, .8)
    elif f is base: im = base(P)(P.img)
    else: im = f(P)
    comp.save(f'qa/world/{n}.jpg', comp.to8(comp.finish(im, 3, **FIN.get(n, {}))), 92); print(n, '%.1fs' % (time.time() - t), flush=True)
ims = [cv2.resize(cv2.imread(f'qa/world/{n}.jpg'), (800, 450), interpolation=cv2.INTER_AREA) for (n, *_) in SETS if os.path.exists(f'qa/world/{n}.jpg')]
while len(ims) % 3: ims.append(ims[0] * 0)
cv2.imwrite('qa/world/sheet.jpg', np.vstack([np.hstack(ims[i:i + 3]) for i in range(0, len(ims), 3)]), [1, 86])
