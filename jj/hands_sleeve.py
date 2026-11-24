"""Drawn sleeves for jj/hands.py: a displaced tube round the forearm -> authored folds, cuff thickness, hem, seam, button, angular light shapes.
    sl = sleeve(g, hand_index, y0=.18, y1=2.2, kind='uniform', seed=1, light=g.L)       (call while g is a full render, before HD.lite)
    draw_back(cel, sl, pal)  -> the far inside of the cuff, BEFORE the hand
    draw_front(cel, sl, pal, w, seed)  -> sleeve over the wrist, AFTER the hand (cast shadow on the wrist included via draw_cast BEFORE it)
pal keys: cloth cloth_sh cloth_lt cloth_rim cuff_in line button button_hi skin_cast
"""
import numpy as np, math, cv2
from .ink import noise1, spl
from .kit import contours, outline
from . import hands as HD

UNI = dict(cloth='#15141f', cloth_sh='#0b0a12', cloth_lt='#4a3f7a', cloth_rim='#8d83c8', cuff_in='#07060b', line='#0a090e', button='#b08a3e', button_hi='#ffe2a0', cast='#8a5560')


def _arm(y):
    return np.interp(y, np.linspace(0, 2.5, 9) - .1, [.262, .262, .262, .285, .318, .350, .375, .39, .40]), np.interp(y, np.linspace(0, 2.5, 9) - .1, [.15, .165, .18, .205, .23, .25, .27, .28, .29])


FOLDS = [  # y, side angle (rad, 0 = back of the arm, + toward thumb), tilt, depth, width
    (.34, 1.35, .10, .075, .050), (.50, -1.2, .13, .085, .055), (.70, 1.1, .16, .080, .062), (.90, -1.5, .20, .075, .07), (1.20, 1.5, .26, .07, .085), (1.55, -1.0, .30, .06, .10)]


def sleeve(g, hi, y0=.18, y1=2.35, ease=1.16, gap=.050, flare=.04, thick=.045, seed=1, light=None, folds=None, torn=0., ny=110, nph=73, kind='uniform', seam=.9, button=True, seams=True):
    h = g.hands[hi]; W = h.sk.fr['wrist']; r = np.random.default_rng(seed); wr = h.bld['wrist']
    F = [list(f) for f in (folds or FOLDS)]
    for f in F: f[0] += r.normal(0, .02); f[1] += r.normal(0, .18); f[2] *= r.uniform(.8, 1.25); f[3] *= r.uniform(.85, 1.2)
    ph = np.linspace(-math.pi, math.pi, nph); Y = np.linspace(0, 1, ny) ** 1.15 * (y1 - y0) + y0
    edge = np.zeros(nph)
    if torn > 0:
        k = r.integers(9, 13); kn = r.uniform(0, 1, k) * torn; edge = np.interp((ph + math.pi) / (2 * math.pi) * k, np.arange(k + 1), np.r_[kn, kn[0]]) + torn * .25 * np.abs(noise1(nph, 3, r)); edge[-1] = edge[0]
    PH, YY = np.meshgrid(ph, Y)
    YY = YY + edge[None] * np.clip(1 - (YY - y0) / .5, 0, 1)
    a, b = _arm(YY); a = a * wr * ease + gap; b = b * wr * ease + gap
    fl = flare * np.clip(1 - (YY - y0) / .22, 0, 1) ** 2; a, b = a + fl, b + fl
    dr = np.zeros_like(YY); 
    for (yf, pf, tl, d, w) in F:
        dp = (PH - pf + math.pi) % (2 * math.pi) - math.pi; s = np.cos(np.clip(dp / 2, -math.pi / 2, math.pi / 2)) ** 2; yv = yf + tl * np.cos(dp) - tl * .2
        q = (YY - yv) / w; dr += d * s * (-1.05 * np.exp(-q * q) + .75 * np.exp(-((q - 1.75) / 1.1) ** 2) + .45 * np.exp(-((q + 1.75) / 1.0) ** 2))
    c, s = np.cos(PH), np.sin(PH); p = 2.4
    rad = 1 / ((np.abs(s / (a + dr)) ** p + np.abs(c / (b + dr)) ** p) ** (1 / p))
    Pl = np.stack([rad * s * (1 if True else 1), YY, rad * c], -1)                   # wrist frame: x flips with the frame itself
    Pw = h.world(Pl.reshape(-1, 3) @ W.R.T + W.o).reshape(ny, nph, 3)
    cen = h.world((np.array([0, 1, 0]) * YY[..., None]).reshape(-1, 3) @ W.R.T + W.o).reshape(ny, nph, 3)
    # normals
    du = np.gradient(Pw, axis=1); dv = np.gradient(Pw, axis=0); n = np.cross(du, dv); n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    if ((n * (Pw - cen)).sum(-1)).mean() < 0: n = -n
    eye = np.array([0, 0, g.cam.dist]); v = eye - Pw; v /= np.linalg.norm(v, axis=-1, keepdims=True); face = (n * v).sum(-1)
    rd = Pw - cen; rd /= np.linalg.norm(rd, axis=-1, keepdims=True) + 1e-9; rface = (rd * v).sum(-1)
    ax_ = h.A @ W.R[:, 1] / h.scale
    if -(ax_ @ HD.unit(eye - cen[0, 0])) > .0: face = np.minimum(face, rface + .02)          # the opening looks at us: the far wall is never outside
    L = HD.unit(light if light is not None else g.L); ndl = n @ L
    px = g.cam.px(Pw)
    sl = dict(px=px, face=face, ndl=ndl, kind=kind, s=g.cam.s)
    # silhouette from quads
    x0, y0_ = np.floor(px.reshape(-1, 2).min(0)).astype(int) - 8; x1, y1_ = np.ceil(px.reshape(-1, 2).max(0)).astype(int) + 8
    x0, y0_ = max(x0, -400), max(y0_, -400); x1, y1_ = min(x1, 2400), min(y1_, 1500)
    k = 1.0; m = np.zeros((max(4, y1_ - y0_), max(4, x1 - x0)), np.uint8)
    q = np.stack([px[:-1, :-1], px[1:, :-1], px[1:, 1:], px[:-1, 1:]], 2).reshape(-1, 4, 2) - [x0, y0_]
    f4 = np.minimum(np.minimum(face[:-1, :-1], face[1:, 1:]), np.minimum(face[1:, :-1], face[:-1, 1:])).ravel() > -.06
    cv2.fillPoly(m, list(np.round(q[f4] * 16).astype(np.int32)), 255, cv2.LINE_AA, 4)
    cs = contours(cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 1.2), min_len=60)
    sl['sil'] = [HD.angular(HD._rs(c_, 2.0, True), .010 * g.cam.s, .45, True, 5) + [x0, y0_] for c_ in cs]
    # cuff opening
    ring = px[0]; inner = g.cam.px(cen[0] + (Pw[0] - cen[0]) * (1 - thick / np.maximum(np.linalg.norm(Pw[0] - cen[0], axis=1), .1))[:, None])
    ax = ax_; sl['opens'] = float(-(ax @ HD.unit(eye - cen[0, 0]))) ; sl['ring'] = ring; sl['inner'] = inner
    sl['near'] = face[0] > 0
    # cast shadow on the wrist: near half of a ring pushed toward the hand
    a0, b0 = _arm(y0 - .15); sh = []
    for dy, sc in ((-.085, .99), (.10, 1.06)):
        aa, bb = _arm(np.full(nph, y0 + dy)); rr = 1 / ((np.abs(np.sin(ph) / (aa * wr * sc)) ** 2.4 + np.abs(np.cos(ph) / (bb * wr * sc)) ** 2.4) ** (1 / 2.4))
        yy = np.full(nph, y0 + dy) - .05 * np.sin(ph + seed) * (dy < 0)
        sh.append(g.cam.px(h.world(np.stack([rr * np.sin(ph), yy, rr * np.cos(ph)], -1) @ W.R.T + W.o)))
    ok = face[0] > -.15; sl['cast'] = _between(sh[0], sh[1], ok)
    # authored marks in (y, phi): folds, light planes, hem, seam
    def at(y, f):
        y = np.asarray(y, float); f = (np.asarray(f, float) + math.pi) % (2 * math.pi) - math.pi
        i = np.interp(y, Y, np.arange(ny)); j = (f + math.pi) / (2 * math.pi) * (nph - 1); i0 = np.clip(i.astype(int), 0, ny - 2); j0 = np.clip(j.astype(int), 0, nph - 2); u, w = (i - i0)[..., None], (j - j0)[..., None]
        bl = lambda A: (A[i0, j0] * (1 - u) * (1 - w) + A[i0 + 1, j0] * u * (1 - w) + A[i0, j0 + 1] * (1 - u) * w + A[i0 + 1, j0 + 1] * u * w)
        return bl(px), bl(face[..., None])[..., 0], bl(ndl[..., None])[..., 0]
    sl['lines'] = []; sl['lights'] = []; sl['darks'] = []
    def vis_runs(y, f, thr=.04):
        P, fc, nl = at(y, f); return [(P[a_:b_], nl[a_:b_]) for a_, b_ in HD._runs(fc > thr, False) if b_ - a_ > 5]
    for (yf, pf, tl, d, w) in F:
        f = pf + np.linspace(-1.45, 1.45, 48); dp = f - pf; yv = yf + tl * np.cos(dp) - tl * .2
        for P, nl in vis_runs(yv, f): sl['lines'].append((P, .62, 'fold'))
        # the lit lip of the fold: a pointed sliver on the arm side of the valley
        f = pf + np.linspace(-1.15, 1.15, 40); dp = f - pf; yv = yf + tl * np.cos(dp) - tl * .2; e = np.sin(np.linspace(0, math.pi, 40)) ** .8
        A_, fa, nla = at(yv + w * .35, f); B_, fb, _ = at(yv + w * (.35 + 2.0 * e), f)
        ok = (fa > .05) & (fb > .02)
        for a_, b_ in HD._runs(ok, False):
            if b_ - a_ < 8: continue
            t = np.linspace(0, 1, b_ - a_); tp = np.clip(np.minimum(t / .25, (1 - t) / .4), 0, 1)[:, None]
            q_ = np.vstack([A_[a_:b_], (A_[a_:b_] + (B_[a_:b_] - A_[a_:b_]) * tp)[::-1]]); lit = nla[a_:b_].mean()
            if lit > .12: sl['lights'].append(q_)
            elif lit < -.05: sl['darks'].append(q_)
    # broken light planes down the lit side of the arm
    j = int(np.argmax(np.where(face[ny // 2] > .35, ndl[ny // 2], -9))); pl = ph[j]; ys = sorted([f[0] for f in F if f[0] < y1]) + [y1]
    for k_, (ya, yb) in enumerate(zip([y0 + .10] + ys[:-1], ys)):
        if yb - ya < .13 or r.random() < .18: continue
        m_ = (yb - ya); c0 = pl + r.normal(0, .15); w0, w1 = r.uniform(.30, .55), r.uniform(.12, .28); sk = r.uniform(.25, .6) * (1 if k_ % 2 else -1)
        pts = [(ya + .20 * m_, c0 - w0), (ya + .12 * m_, c0 + w0 * .7), (ya + .50 * m_, c0 + w1 + sk * .45), (yb + .02 * m_, c0 + sk), (ya + .58 * m_, c0 - w1 + sk * .40)]
        yy, ff = [], []
        for (p0, p1) in zip(pts, pts[1:] + pts[:1]): t = np.linspace(0, 1, 9)[:-1]; yy += list(p0[0] + (p1[0] - p0[0]) * t); ff += list(p0[1] + (p1[1] - p0[1]) * t)
        P, fc, nl = at(yy, ff)
        if fc.min() > -.02 and nl.mean() > .1: sl['lights'].append(P)
    # shade plane: the side turned from the light
    sh_ = ndl < -.02
    f = np.linspace(-math.pi, math.pi, 120)
    th = []
    for i in range(0, ny, 3):
        row = sh_[i] & (face[i] > -.05)
        for a_, b_ in HD._runs(row, False):
            if b_ - a_ > 2: th.append((i, a_, b_ - 1))
    m2 = np.zeros_like(m); qs = (ndl[:-1, :-1] + ndl[1:, 1:]).ravel() < -.04
    cv2.fillPoly(m2, list(np.round(q[f4 & qs] * 16).astype(np.int32)), 255, cv2.LINE_AA, 4)
    mm = cv2.GaussianBlur(m2.astype(np.float32) / 255, (0, 0), max(1.5, .018 * g.cam.s))
    sl['shade'] = [HD.angular(HD._rs(c_, 2.0, True), .030 * g.cam.s, .85, True, 3) + [x0, y0_] for c_ in contours(mm, min_len=int(.5 * g.cam.s))]
    if seams and torn <= 0:
        for dy, wt in ((.095, .5), (.125, .32)):
            P, fc, _ = at(np.full(80, y0 + dy) , np.linspace(-math.pi, math.pi, 80))
            for a_, b_ in HD._runs(fc > .03, False):
                if b_ - a_ > 5: sl['lines'].append((P[a_:b_], wt, 'hem'))
        for P, nl in vis_runs(np.linspace(y0 + .02, y1, 90), seam + .05 * np.sin(np.linspace(0, 14, 90))): sl['lines'].append((P, .40, 'seam'))
    sl['button'] = None
    if button:
        P, fc, _ = at(np.array([y0 + .24]), np.array([seam - .20])); 
        if fc[0] > .15:
            t = np.linspace(0, 2 * math.pi, 18, endpoint=False); Q, _, _ = at(y0 + .24 + .042 * np.sin(t), seam - .20 + .14 * np.cos(t)); sl['button'] = Q
    sl['rim'] = px, face
    return sl


def _between(A, B, ok):
    out = []
    for a, b in HD._runs(ok, True):
        idx = np.arange(a, b) % len(ok)
        if len(idx) > 4: out.append(np.vstack([A[idx], B[idx][::-1]]))
    return out


def draw_back(cel, sl, pal):
    """the inside of the cuff that shows behind the wrist"""
    p = dict(UNI, **pal)
    if sl['opens'] > -.35:
        cel.fill(sl['ring'], p['cloth_sh']); cel.fill(sl['inner'], p['cuff_in']); cel.line(np.vstack([sl['ring'], sl['ring'][:1]]), 2.2, p['line'], taper=(0, 0), seed=2)
    return cel


def draw_cast(cel, sl, pal, polys=None):
    """shadow the cuff throws on the wrist (clip to the hand's palm polygons when given)"""
    p = dict(UNI, **pal)
    if polys: cel.ctx.save(); cel._path(polys); cel.ctx.set_fill_rule(1); cel.ctx.clip()
    for q in sl['cast']: cel.fill(q, p['cast'])
    if polys: cel.ctx.restore()
    return cel


def draw_front(cel, sl, pal, w=5., seed=0, rimdir=None, rimcol=None, rimw=1.1, lights=True):
    p = dict(UNI, **pal); S_ = sl['sil']
    cel.fill(S_, p['cloth'], evenodd=False)
    cel.clip(S_)
    for q in sl['shade']: cel.fill(q, p['cloth_sh'])
    if lights:
        for q in sl['lights']: cel.fill(q, p['cloth_lt'])
    for q in sl['darks']: cel.fill(q, p['cloth_sh'])
    for k, (P, wt, kind) in enumerate(sl['lines']):
        cel.line(P, w * wt * (1. if kind == 'fold' else .8), p['line'] if kind != 'seam' else p['line'], taper=(.12, .5) if kind == 'fold' else (.08, .12), press=.3, seed=seed * 7 + k, tmin=.15)
    cel.unclip()
    if sl['button'] is not None:
        b = sl['button']; cel.fill(b, p['button']); c = b.mean(0); cel.fill(c + (b - c) * .55 + (b[12] - c) * .22, p['button_hi']); cel.line(np.vstack([b, b[:1]]), max(1., w * .4), p['line'], taper=(0, 0), seed=seed)
    # cuff lip
    n = sl['near']; lip = _between(sl['ring'], sl['inner'], n)
    if sl['opens'] > .05:
        for q in lip: cel.fill(q, p['cloth_lt'])
    for a, b in HD._runs(n, True):
        idx = np.arange(a, b) % len(n); cel.line(sl['ring'][idx], w * .85, p['line'], taper=(.1, .1), seed=seed + 3)
        if sl['opens'] > .05: cel.line(sl['inner'][idx], w * .5, p['line'], taper=(.1, .1), seed=seed + 4)
    for k, q in enumerate(S_): outline(cel, q, w, p['line'], seed * 5 + k)
    if rimdir is not None:                                                   # hard kick along the lit silhouette: broken, tapered, thicker on the bulges
        d = np.array([rimdir[0], -rimdir[1]], float); d /= np.hypot(*d) + 1e-9
        for k, q in enumerate(S_):
            T = np.gradient(q, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]; c = q.mean(0)
            if ((q - c) * N).sum(1).mean() < 0: N = -N
            a = N @ d
            for x, y in HD._runs(HD._close_gaps(a > .22, 3, True), True):
                if y - x < 10: continue
                idx = np.arange(x, y) % len(q); aa = np.clip((a[idx] - .15) * 1.5, .25, 1.2); t0 = np.linspace(0, 1, len(idx))
                cel.line(q[idx] - N[idx] * w * (.55 + rimw * .45), w * rimw, rimcol or p['cloth_rim'], taper=(.2, .2), press=.35, seed=seed * 3 + k + x, wfun=lambda t, A=t0, B=aa: np.interp(t, A, B), tmin=.1)
    return cel
