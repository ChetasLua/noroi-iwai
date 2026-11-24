"""Smear drawings for jj.figure: swept-limb arcs with dry-brush tails, after-image limbs, whole-figure speed tails."""
import numpy as np, math, cv2
from . import W, H
from .paint import rgb, f32, Canvas
from .ink import Cel, spl, resample
from . import kit
from .fig_pose import Skel, mix_pose


def _skel(fig, a, b, t):
    """skeleton in between two states (dicts: pose, x, y, z, heading)"""
    k = lambda n: a.get(n, 0.) * (1 - t) + b.get(n, 0.) * t
    return Skel(fig.body, mix_pose(a['pose'], b['pose'], t), k('x'), k('y'), k('z'), k('heading'), fig.scale)


def limb_points(sk, limb):
    """chain of (world point, radius, material key) from root to tip"""
    s = limb[-1]; m = sk.sc * sk.b['mass'] * sk.b['limb'] * sk.H / sk.sc / 1.73; hl = sk.b['hand'] * sk.H
    if limb.startswith('arm'):
        tip = sk.wr[s] + sk.Rhand[s][:, 1] * hl * .62
        return [(sk.sh[s], .050 * m, 'cloth'), (sk.el[s], .050 * m, 'cloth'), (sk.wr[s], .044 * m, 'cloth'), (sk.wr[s] + sk.Rhand[s][:, 1] * hl * .05, .05 * m, 'skin'), (tip, .055 * m, 'skin')]
    f = sk.Rfoot[s]
    return [(sk.hip[s], .09 * m, 'cloth'), (sk.kn[s], .085 * m, 'cloth'), (sk.an[s], .072 * m, 'cloth'), (sk.an[s] + f[:, 1] * .03, .06 * m, 'shoe'), (sk.an[s] + f[:, 1] * .16 * m, .055 * m, 'shoe')]


def limb_smear(fig, cam, a, b, limb='armR', cols=None, pal=None, n=22, seed=0, tail=.85, edge=True, lines=10, shade=True):
    """premultiplied RGBA layer: the limb swept from state a to state b (a = where it was, b = where it is now).
    Draw it UNDER the figure posed at b (or on its own for a one-frame smear)."""
    from .figure import Pal
    pal = pal if isinstance(pal, Pal) else Pal(pal); sp = dict(__import__('jj.fig_body', fromlist=['SPEC']).SPEC[fig.char]); sp.update(fig.spec)
    shoe = dict(hightop='redshoe', ankle='ochre', loafer='loafer', dress='black', slipper='black', zori='tabi', bare=sp['skin'])[sp['shoe']]
    key = dict(cloth=sp['cloth'] if sp['top'] != 'bare' or limb.startswith('leg') else sp['skin'], skin=sp['skin'], shoe=shoe)
    if limb.startswith('leg') and sp['bottom'] in ('skirt',): key['cloth'] = 'tights'
    r = np.random.default_rng(seed + 17)
    ts = np.linspace(0, 1, n); ch = [limb_points(_skel(fig, a, b, t), limb) for t in ts]
    P = np.array([[q[0] for q in c] for c in ch]); R = np.array([[q[1] for q in c] for c in ch])              # (n, 5, 3)
    xy, z = cam.project(P); sc = cam.f / np.maximum(z, .05)
    L = np.zeros((cam.h, cam.w, 4), f32)
    if (z < .08).any(): return L
    travel = np.hypot(*np.diff(xy[:, -1], axis=0).T).sum()
    if travel < 6: return L
    cv = Canvas(cam.h, cam.w, color='#000000', layer=True)
    # strips from the elbow / knee outward: each bristle starts at its own moment, the tip ones earliest
    k = 1                                                                # chain index the sweep begins at
    segs = [(1, 2), (2, 3), (3, 4)]
    S = []
    for (i, j) in segs:
        w = np.hypot(*(xy[-1, j] - xy[-1, i])); m = max(2, int(w / 5.))
        for u in np.linspace(0, 1, m, endpoint=False): S.append((i, j, u + r.uniform(0, 1. / m)))
    tipw = float(R[-1, 4] * sc[-1, 4])
    for e in np.linspace(-1., 1.15, max(4, int(tipw / 3.0))): S.append((3, 4, 1 + e * .55))
    ink = rgb(pal.ink); Pl, Wd, Col, Op, Dr = [], [], [], [], []
    K = 14
    order = r.permutation(len(S))
    for q in order:
        i, j, u = S[q]
        rad = (i + u) / 4.
        t0 = np.clip((1 - tail) + tail * (1 - rad) * .75 + abs(r.normal(0, .14)), 0, .82) if rad < .95 else np.clip((1 - tail) + abs(r.normal(0, .10)), 0, .6)
        tt = np.linspace(t0, 1., K); line = np.stack([np.interp(tt, ts, xy[:, i, c] * (1 - u) + xy[:, j, c] * u) for c in (0, 1)], -1)
        mt = key['skin' if limb.startswith('arm') else 'shoe'] if i >= 2 else key['cloth']
        lit, base, shd, deep = pal(mt)
        c = base if r.random() < .62 else (shd if r.random() < .6 else lit)
        Pl.append(line); Wd.append(max(1.2, r.uniform(2.2, 4.6) * max(.5, sc[-1, j] / 420.))); Col.append(c); Op.append(1.); Dr.append(r.uniform(.15, .55))
    Pl = np.array(Pl)[:, ::-1]                                           # brush starts loaded at the leading edge and dries toward the tail
    cv.strokes(Pl, np.array(Wd), np.array(Col), op=np.array(Op), dry=np.array(Dr), seed=seed, shape='taper', bristle=.10)
    L = cv.layer()
    cel = Cel(cam.w, cam.h, 0, 0, ss=1)
    if edge:
        out = spl(xy[int(n * .25):, 4] + 0, n=4); cel.line(out, max(1.4, tipw * .10), pal.ink, taper=(.85, .05), seed=seed, press=.4)
        mid = spl(xy[int(n * .5):, 2], n=4); cel.line(mid, max(1.2, tipw * .07), pal.ink, taper=(.9, .05), seed=seed + 1, press=.4)
    for q in range(lines):
        u = r.uniform(.35, 1.12); i = min(3, int(u * 4)); v = u * 4 - i; t0 = r.uniform(.0, .5); t1 = r.uniform(.75, 1.); tt = np.linspace(t0, t1, 12)
        ln = np.stack([np.interp(tt, ts, xy[:, i, c] * (1 - v) + xy[:, i + 1, c] * v) for c in (0, 1)], -1)
        cel.line(ln, r.uniform(.8, 1.8) * max(.7, sc[-1, 4] / 500.), pal.ink if r.random() < .65 else '#f4efe8', taper=(.8, .2), seed=seed + q, press=.4)
    e = cel.rgba()
    return L * (1 - e[..., 3:4]) + e


def after_limb(fig, cam, a, b, limb='armR', ts=(.35, .65), pal=None, seed=0, w=2.2, fill=.0):
    """ink-only after-image limbs at fractions ts between two states"""
    from .figure import Pal
    pal = pal if isinstance(pal, Pal) else Pal(pal); cel = Cel(cam.w, cam.h, 0, 0, ss=2)
    for k, t in enumerate(ts):
        ch = limb_points(_skel(fig, a, b, t), limb); P, z = cam.project(np.array([c[0] for c in ch])); r = np.array([c[1] for c in ch]) * cam.f / np.maximum(z, .06)
        T = np.gradient(P, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-6; N = np.c_[-T[:, 1], T[:, 0]]
        A, B = P + N * r[:, None], P - N * r[:, None]
        for S in (A, B): cel.line(spl(S, n=6), w * (.6 + .5 * t), pal.ink, taper=(.5, .2), seed=seed + k, press=.5)
        cel.line(spl(np.vstack([A[-1], P[-1] + T[-1] * r[-1] * 1.1, B[-1]]), n=6), w * (.6 + .5 * t), pal.ink, taper=(.3, .3), seed=seed + 9 + k)
    return cel.rgba()


def tails(layer, v, n=60, seed=0, width=(2., 9.), length=(.35, 1.0), dark=.75, dry=.5, ink=True, where=None):
    """dry-brush speed tails off the trailing edge of a figure layer moving by v (px, the motion of one drawing).  Returns a layer to put UNDER it"""
    h, w = layer.shape[:2]; a = layer[..., 3]; v = np.asarray(v, float); L = np.hypot(*v)
    if L < 3: return np.zeros_like(layer)
    d = v / L; k = max(6., L * .12)
    sh = kit.shifted(a, d[0] * k, d[1] * k); m = (a > .6) & (sh < .4)
    if where is not None: m &= where
    ys, xs = np.nonzero(m)
    if len(xs) == 0: return np.zeros_like(layer)
    r = np.random.default_rng(seed + 5); q = r.choice(len(xs), min(n, len(xs)), replace=False)
    cv = Canvas(h, w, color='#000000', layer=True); Pl, Wd, Cl = [], [], []
    nrm = np.array([-d[1], d[0]])
    for i in q:
        p = np.array([xs[i], ys[i]], float) + d * 6; ln = L * r.uniform(*length); bow = r.normal(0, .05) * ln
        u = np.linspace(0, 1, 9)[:, None]; Pl.append(p - d * ln * u + nrm * bow * np.sin(u * math.pi)); c = layer[ys[i], xs[i], :3] / max(layer[ys[i], xs[i], 3], .2)
        Cl.append(c * r.uniform(dark, 1.)); Wd.append(r.uniform(*width))
    cv.strokes(np.array(Pl), np.array(Wd), np.array(Cl), dry=dry, seed=seed, shape='taper', bristle=.1)
    out = cv.layer()
    if ink:
        cel = Cel(w, h, 0, 0, ss=1)
        for i in q[::4]:
            p = np.array([xs[i], ys[i]], float); ln = L * r.uniform(.4, 1.1); cel.line(np.array([p - d * ln, p - d * ln * .5, p]), r.uniform(.8, 1.9), '#17131a', taper=(.9, .1), seed=int(i) & 255)
        e = cel.rgba(); out = out * (1 - e[..., 3:4]) + e
    return out
