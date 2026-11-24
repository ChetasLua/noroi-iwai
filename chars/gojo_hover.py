"""the hover of d5 as drawings on twos: body roll, thrusts of the two hands, streaming coat tails, sleeves and hair"""
import numpy as np, math, cv2
from jj import W, H as HH
from jj.paint import f32, rgb
from jj.ink import Cel, spl
from jj.cam import Camera3D
from jj.figure import Fig
from jj.fig_pose import mix_pose, POSES
from jj import comp
from chars.gojo import Gojo, gojo_head

EX = dict(band=False, mouth='half', lid='cold', brow='cold', pupil=.24, glow=.3)


def _pose(f):
    """blue hand (his right) is up from the start; the red hand snaps up on 9 and throws the body back; a slow roll underneath"""
    a = 2 * math.pi * f / 34.; sn = math.sin
    K = lambda ks: float(comp.lerp_keys(f, ks))
    up = K([(0, 0.), (7, -.06), (8, -.10), (9, .55), (10, 1.16), (12, 1.06), (15, .97), (20, 1.0), (26, 1.02)]); rec = K([(0, .9), (2, .5), (5, 0.), (8, 0.), (9, .3), (10, 1.), (12, .75), (16, .15), (20, 0.), (26, 0.)])
    d = dict(POSES['float'])
    d.update(root=(0, 0, .05), pel=(-10 - 7 * rec + 3 * sn(a), 5 + 9 * sn(a - .5), 4 + 8 * sn(a * .8 + .6)), wst=(3, -3 + 3 * sn(a + .4), -3 + 2 * sn(a)), cst=(-5 - 5 * rec, -6 + 6 * sn(a + .9), -5 + 9 * sn(a * .8 + 1.1) - 8 * rec * (f > 8)),
             nck=(7 + 3 * rec, 2 * sn(a), 0), hed=(8 - 2 * rec, 3 + 4 * sn(a + 1.6), -4 - 4 * sn(a * .8 + 1.4)), cL=(-1.4 + 1.2 * max(0, up), 0), cR=(-1.4, 0),
             hR=(.74 + .04 * sn(a), .26, -.10 + .07 * sn(a + .3) + .12 * rec * (f < 8)), eR=(.3, -.5, -.8), gR='open', wR=(24, -14 - 8 * rec, 0),
             hL=tuple(np.array([.46, .18, -.74]) * (1 - up) + np.array([.78, .30, .00 + .04 * sn(a + .8)]) * up), eL=(.3, -.5, -.8), gL='open' if up > .4 else 'relax', wL=(24 * max(0, up), -14 * max(0, up), 0),
             fL=(.035 + .02 * sn(a), -.04 + .05 * sn(a + .2) - .05 * rec, .02), fR=(.085 + .015 * sn(a + 1), .06 - .07 * rec, .27 + .035 * sn(a + .5)), kL=(.1, 1, 0), kR=(.55, 1, .15), tL=(4, -62 + 6 * sn(a), -16), tR=(8, -58 - 8 * sn(a + .6), -16))
    return d


def _panel(c, A, B, L0, L1, f, ang, seed, n=9, amp=.5, per=8.5, tones=('#11101b', '#1c1a30', '#453a78', '#7a6cc8'), dark='#05050c', curl=0., lw=3., fan=.0, taper=1.):
    """one streaming panel hung from A..B (px); ang(f) = angle of the wind on screen, read later toward the tip.  shaded in flat facets by the travelling wave"""
    m = 7; T = np.linspace(0, 1, n); r = np.random.default_rng(seed); ph = r.uniform(0, 6.28); X = np.zeros((m, n, 2)); Sh = np.zeros((m, n))
    for i in range(m):
        v = i / (m - 1.); p = np.array(A, float) * (1 - v) + np.array(B, float) * v; L = (L0 * (1 - v) + L1 * v) / (n - 1) * (1 + .10 * math.sin(v * 9 + seed)); X[i, 0] = p
        for j in range(1, n):
            w = 2 * math.pi * (f / per - j / (n - 1.) * 1.15) + ph + v * 1.5; th = ang(f - .75 * j) + fan * (1 - 2 * v) + amp * (j / (n - 1.)) ** .85 * math.sin(w) + curl * (j / (n - 1.)) ** 2; p = p + L * np.array([math.cos(th), math.sin(th)]); X[i, j] = p; Sh[i, j] = math.cos(w + v * .6)
    Sh[:, 0] = Sh[:, 1]
    edge = np.vstack([X[0], X[1:-1, -1] + (X[1:-1, -1] - X[1:-1, -2]) * r.uniform(-.5, .9, (m - 2, 1)), X[-1][::-1], X[1:-1, 0][::-1]]); O = np.vstack([spl(X[0], n=5), edge[n:n + m - 2], spl(X[-1][::-1], n=5)])
    c.fill(O, tones[0]); c.clip(O)
    for i in range(m - 1):                                              # facets: lit where the wave turns to the light
        for j in range(n - 1):
            s = Sh[i, j] * .6 + Sh[i + 1, j + 1] * .4; q = np.array([X[i, j], X[i, j + 1], X[i + 1, j + 1], X[i + 1, j]])
            if s > .05: c.fill(q + 0., tones[1])
    for i in range(1, m - 1):                                           # long fold lights
        for j in range(1, n - 2):
            if Sh[i, j] > .62 and (i * 3 + j * 5 + seed) % 3: a_, b_, e_ = X[i, j], X[i, j + 1], X[i, min(n - 1, j + 2)]; d = e_ - a_; nn = np.array([-d[1], d[0]]) / (np.hypot(*d) + 1e-6); w_ = np.hypot(*(X[i + 1, j] - X[i, j])) * .36; c.fill(np.array([a_, b_ + nn * w_, e_, b_ - nn * w_ * .25]), tones[2]); (Sh[i, j] > .86) and c.fill(np.array([a_ * .5 + b_ * .5, b_ + nn * w_ * .55, e_ * .6 + b_ * .4, b_]), tones[3])
    c.unclip()
    for i in (2, 4): c.line(X[i, :n - 1], lw * .7, dark, taper=(.1, .5), press=.4, seed=seed + i)
    for E in (X[0], X[-1]): c.line(spl(E, n=4), lw, dark, taper=(.02, .25), press=.4, seed=seed + 9)
    return X


def wind(f):
    """angle of the wind on screen (0 = right, pi/2 = down): drawn up toward Blue (screen left), whipped down-left by Red on 9"""
    return math.radians(float(comp.lerp_keys(f, [(-12, 150), (0, 158), (4, 146), (8, 140), (9, 132), (10, 112), (12, 132), (14, 178), (16, 186), (19, 160), (22, 144), (26, 158), (30, 162)])))


def drawing(f, cam):
    gust = float(comp.lerp_keys(f, [(0, 1.5), (5, 1.), (9, 1.), (10, 2.2), (14, 1.4), (20, 1.), (26, 1.1)]))
    def head_fn(cel, H, info):
        Hh = gojo_head(True).pose(H.cx, H.cy, H.s * 1.08, H.yaw, H.pitch, H.roll, H.persp); t = f / 24.
        Gojo.head(cel, Hh, 'wind', EX, 'night', f * 2.2, (-.52 * gust - .16 * math.sin(f * .85), .07 + .12 * math.sin(f * .85 + 1.2)), 3, neck=False, key='L', sway=.045); return cel
    F = Fig('gojo').anim(lambda q: dict(pose=_pose(q), x=0., y=0., z=0., heading=math.pi + .48 - .80 * max(0., q) / 26.)).blow(-.5 * gust, -.3, .4)
    F.at(f); body = F.draw(cam, f, light=(-.3, -.3, .6), head_fn=head_fn); I = F.info
    g = lambda k, s=None: np.array(I[k][s] if s else I[k], float)[:2]
    pv, ch = g('pelvis'), g('chest'); up = (ch - pv) / (np.hypot(*(ch - pv)) + 1e-6); rt = np.array([-up[1], up[0]]); u = np.hypot(*(ch - pv)) / .30   # px per metre, roughly
    back = Cel(); hem = pv - up * .02 * u
    wa = up * .10 * u                                                   # the coat is caught at the waist and flies out behind him
    _panel(back, hem + rt * .16 * u, hem - rt * .01 * u, .50 * u, .58 * u, f + 2., lambda q: math.pi - wind(q) + .36, 12, n=9, amp=.38 * gust ** .5, fan=-.26, lw=3.)
    _panel(back, hem + rt * .12 * u + wa, hem - rt * .13 * u, .98 * u, .80 * u, f, wind, 3, n=11, amp=.40 * gust ** .5, fan=.33, lw=3.4)
    _panel(back, hem + rt * .05 * u, hem - rt * .15 * u - up * .03 * u, .66 * u, .50 * u, f + 3., lambda q: wind(q) - .50, 8, n=9, amp=.44 * gust ** .5, fan=.26, lw=3.)
    for s in 'LR':                                                      # sleeves: cloth blown off the forearm
        el, wr, sh_ = g('elbows', s), g('wrists', s), g('shoulders', s); d = wr - el
        _panel(back, el - (el - sh_) * .30, wr - d * .06, .12 * u, .30 * u, f + (2. if s == 'L' else 0.), lambda q: wind(q) - .25, 20 + ord(s), n=6, amp=.5, per=7., lw=2.6, fan=-.12)
    return _ov(back.layer(), body), I


def _ov(a, b): return a * (1 - b[..., 3:4]) + b


SHOT = dict(POSES['float'])
SHOT.update(root=(0, 0, .05), fL=(.035, -.03, .02), fR=(.085, .06, .27), kL=(.1, 1, 0), kR=(.55, 1, .15), tL=(4, -62, -16), tR=(8, -58, -16), pel=(-8, -8, 5), wst=(2, -4, -2), cst=(-2, -10, 2), nck=(5, 0, 0), hed=(4, 4, -3), cR=(-1.4, 0), hL=(.66, .62, .06), eL=(.5, -.5, -.5), gL='open', hR=(.22, -.10, -.90), eR=(.6, -.8, 0), gR='relax')


def shot(f, cam, pull=1.):
    """the arm that let it go; everything loose on him is dragged after the sphere (screen right)"""
    w_ = lambda q: math.radians(18 + 10 * math.sin(q * .5))
    def head_fn(cel, H, info):
        Hh = gojo_head(True).pose(H.cx, H.cy, H.s * 1.08, H.yaw, H.pitch, H.roll, H.persp); Gojo.head(cel, Hh, 'wind', EX, 'night', f * 2.2, (.55 * pull + .12 * math.sin(f * .9), .02 + .1 * math.sin(f * .9)), 3, neck=False, key='R', sway=.045); return cel
    F = Fig('gojo').pose(dict(SHOT, pel=(-8 - 3 * pull, -8 + 2 * math.sin(f * .4), 5), cst=(-2 - 3 * pull, -10, 2))).place(0, 0, 0, heading=math.pi + .10).blow(.6 * pull, -.3, .2); body = F.draw(cam, f, light=(.6, .1, .6), head_fn=head_fn); I = F.info
    g = lambda k, s=None: np.array(I[k][s] if s else I[k], float)[:2]
    pv, ch = g('pelvis'), g('chest'); up = (ch - pv) / (np.hypot(*(ch - pv)) + 1e-6); rt = np.array([-up[1], up[0]]); u = np.hypot(*(ch - pv)) / .30; hem = pv - up * .02 * u; back = Cel()
    _panel(back, hem - rt * .12 * u + up * .1 * u, hem + rt * .13 * u, .95 * u, .80 * u, f, w_, 3, n=11, amp=.36 * pull, fan=-.30, lw=3.4); _panel(back, hem - rt * .05 * u, hem + rt * .15 * u, .62 * u, .5 * u, f + 3., lambda q: w_(q) + .5, 8, n=9, amp=.42, fan=-.24, lw=3.)
    for s in 'LR': el, wr, sh_ = g('elbows', s), g('wrists', s), g('shoulders', s); _panel(back, el - (el - sh_) * .3, wr - (wr - el) * .06, .12 * u, .28 * u, f + (2. if s == 'L' else 0.), lambda q: w_(q) + .35, 20 + ord(s), n=6, amp=.5, per=7., lw=2.6)
    return _ov(back.layer(), body), I
