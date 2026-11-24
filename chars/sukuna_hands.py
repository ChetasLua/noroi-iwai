"""SUKUNA lane: two inkings (brush ink / fire cel) of the geometry that jj.hands.render() returns."""
import numpy as np, cv2, math, cairo
from jj import W, H
from jj.paint import rgb, f32, smooth
from jj.ink import Cel, spl, resample, noise1
from jj.kit import mask_of
from jj import hands as HD
from chars.sukuna_ink import *

SKIN = '#ecdcc3'; SKIN_S = '#b9a596'; SKIN_D = '#7d6a66'
BANDS = ((.15, .27), (.42, .55))


def sign(name, cx, cy, s, yaw=-38, pitch=-12, roll=-3, dist=5., light=(-.72, .45, .5), nails='claw', claw=.07, bands=True, poses=None, gap=0., res=None, bld=HD.HEAVY, **k):
    """two-hand sign -> Drawn geometry (screen px)"""
    st = dict(nails=nails, claw=claw);
    if bands: st['bands'] = BANDS
    hs = HD.two(name, R=HD.euler(yaw, pitch, roll), gap=gap, poses=poses, bld=bld, **st)
    k.setdefault('thr', .26)
    return HD.render(hs, HD.Cam(cx, cy, s, dist), light=light, res=res or int(np.clip(s * 2.6, 420, 900)), **k)


def one(pose, cx, cy, s, side='R', R=None, dist=5., light=(-.6, .55, .55), nails='claw', claw=.07, bands=True, res=None, scale=1., pos=(0, 0, 0), style=None, bld=HD.HEAVY, **k):
    st = dict(nails=nails, claw=claw); st.update(style or {})
    if bands: st['bands'] = BANDS
    return HD.render([HD.Hand(pose, side, R, pos, bld, scale=scale, **st)], HD.Cam(cx, cy, s, dist), light=light, res=res or int(np.clip(s * 2.6, 420, 1000)), **k)


def hatch_mask(c, mask, ox, oy, ang, gap=6., w=1.8, seed=0, over=.18, flip=False, group=5, hole=.45, minlen=7., col=BLACK, bow=.03):
    """parallel pen strokes inside a soft mask; each stroke starts on one side of the mask and dies past the other"""
    r = np.random.default_rng(abs(int(seed)) + 2); h, wd = mask.shape; ys, xs = np.nonzero(mask > .4)
    if len(xs) < 20: return c
    d = np.array([math.cos(ang), math.sin(ang)]); n = np.array([-d[1], d[0]]); P = np.c_[xs, ys]; v = P @ n; u = P @ d; u0, u1 = u.min() - 2, u.max() + 2; step = 2.0; m = int((u1 - u0) / step) + 2
    ph = r.uniform(0, 6)
    for k, vv in enumerate(np.arange(v.min(), v.max(), gap)):
        vv += r.normal(0, gap * .22); a = ang + r.normal(0, .03); dd = np.array([math.cos(a), math.sin(a)]); q = (u0 + np.arange(m) * step)[:, None] * dd + vv * n
        xi = np.rint(q[:, 0]).astype(int); yi = np.rint(q[:, 1]).astype(int); ok = (xi >= 0) & (yi >= 0) & (xi < wd) & (yi < h); on = np.zeros(m, bool); on[ok] = mask[yi[ok], xi[ok]] > .5
        e = np.diff(np.r_[0, on.astype(int), 0]); A = np.nonzero(e == 1)[0]; B_ = np.nonzero(e == -1)[0] - 1
        for i, j in zip(A, B_):
            if (j - i) * step < minlen: continue
            L = (j - i) * step; k_ = 1 - hole * (.5 + .5 * math.sin(2 * math.pi * k / group + ph)) * r.random(); p0, p1 = q[i], q[j]
            if flip: p0, p1 = p1, p0
            dv = p1 - p0; e1 = p0 + dv * (k_ * r.uniform(.8, 1.0) + over * r.random()); e0 = p0 - dv * .02; mid = (e0 + e1) / 2 + np.array([-dv[1], dv[0]]) * (bow + r.normal(0, .015))
            R = stroke_poly(spl([e0, mid, e1], n=6) + (ox, oy), np.array([.5, 1, .8, .4, .05]) * w * r.uniform(.7, 1.3), int(r.integers(1 << 30)), .2, 1.8, .15)
            if R is not None: c.fill(R, col)
    return c


def _clip(cels, polys):
    for c in cels: c.ctx.save(); c._path(polys); c.ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD); c.ctx.clip()


def _unclip(cels):
    for c in cels: c.ctx.restore()


def ink_hands(sh, g, lw=1., seed=3, hatch=1., skin=SKIN, gap=5.6, dark_hand=None):
    """brush-ink a jj.hands Drawn into a Sheet.  dark_hand = index of a hand that sits in shadow"""
    ms = [c for c in (sh.a, sh.v, sh.k, sh.g)]
    for G in g.groups:
        if not G['polys']: continue
        for c in (sh.v, sh.k, sh.g):
            c.ctx.save(); c.ctx.set_operator(cairo.OPERATOR_DEST_OUT); c.fill(G['polys'], '#000', evenodd=True) if False else c.fill(G['polys'], '#000000', evenodd=True); c.ctx.restore()
        nail = G['name'].startswith('nail'); sh.a.fill(G['polys'], BLACK if nail else skin, evenodd=True); sh.a.ctx.set_line_width(2.); sh.a._path(G['polys']); sh.a._src(BLACK if nail else skin, 1.); sh.a.ctx.stroke()
        if nail: sh.k.fill(G['polys'], BLACK, evenodd=True); continue
        S_ = g.shadow.get(G['id']); _clip(ms, G['polys'])
        if dark_hand is not None and G['hand'] == dark_hand: sh.v.fill(G['polys'], SKIN_S, .75, evenodd=True)
        if S_:
            sh.v.fill(S_, SKIN_S, .95, evenodd=True)
            P = np.vstack([np.asarray(q) for q in S_]); lo = np.floor(P.min(0)).astype(int) - 3; hi = np.ceil(P.max(0)).astype(int) + 3; lo = np.maximum(lo, (sh.ox, sh.oy)); hi = np.minimum(hi, (sh.ox + sh.w, sh.oy + sh.h))
            if (hi - lo).min() > 6:
                m = Cel(hi[0] - lo[0], hi[1] - lo[1], lo[0], lo[1], ss=1); m.fill(G['polys'], '#ffffff', evenodd=True); A = m.rgba()[..., 3]; m2 = Cel(hi[0] - lo[0], hi[1] - lo[1], lo[0], lo[1], ss=1); m2.fill(S_, '#ffffff', evenodd=True); M = m2.rgba()[..., 3] * A
                ys, xs = np.nonzero(M > .5)
                if len(xs) > 60:
                    GP = np.vstack([np.asarray(q) for q in G['polys']]); cc = GP - GP.mean(0); ev, V = np.linalg.eigh(cc.T @ cc); ax = V[:, 1]; ang = math.atan2(ax[1], ax[0]) + (.55 if G['name'] != 'palm' else .25)
                    if hatch: hatch_mask(sh.k, M, lo[0], lo[1], ang, gap / hatch, 2.0 * lw, seed + G['id'])
                    # deeper core: second, sparser layer near the far edge
                    er = cv2.erode(M, np.ones((3, 3)), iterations=int(4 + 3 * lw)); sh.v.fill(S_, SKIN_D, .25, evenodd=True)
                    if hatch and G['name'] == 'palm': hatch_mask(sh.k, er, lo[0], lo[1], ang - .42, gap * 1.25, 1.8 * lw, seed + G['id'] + 50)
        _unclip(ms)
    if getattr(g, 'seams', None): sh.a.fill(g.seams, SKIN_S, evenodd=True); sh.v.fill(g.seams, SKIN_S, .6, evenodd=True)
    byid = {G['id']: G for G in g.groups}
    for b in g.bands:
        G = byid.get(b['gid'])
        if G is None: continue
        _clip([sh.k], G['polys']); sh.k.fill(b['polys'], BLACK, evenodd=True); _unclip([sh.k])
        for q in b['polys']:
            q = np.asarray(q); c = q.mean(0); d = q - c; i = int(np.argmax(d[:, 0])); L = np.linalg.eigh(d.T @ d)[1]; ax, nn = L[:, 1], L[:, 0]; ext = np.abs(d @ ax).max(); th = np.abs(d @ nn).max()
            _clip([sh.g], G['polys']); _clip([sh.g], b['polys'])
            for k in range(6): p = c + ax * ext * (.25 + .10 * k) * (1 if ax[0] > 0 else -1); brush(sh.g, np.array([p - nn * th * .7, p + nn * th * .5]), [.2, 1.8 * lw, .2], '#cfc3b0', seed + k, .3)
            _unclip([sh.g]); _unclip([sh.g])
    for n in g.nails:
        G = byid.get(n['gid'])
        if G is not None: _clip([sh.k], G['polys']); sh.k.fill(n['polys'], BLACK, evenodd=True); _unclip([sh.k])
    for j, c in enumerate(g.creases):
        if c['kind'] == 'fold' or len(c['P']) < 4: continue
        brush(sh.k, c['P'][::2] if len(c['P']) > 8 else c['P'], np.array([.4, 1., .7, .15]) * 3.0 * c['w'] * lw + .2, seed=seed * 17 + j, smooth_=False)
    l2 = np.array([g.L[0], -g.L[1]]); l2 = l2 / (np.hypot(*l2) + 1e-6)
    for j, st in enumerate(sorted(g.strokes, key=lambda s: (s['kind'] == 'out', float(np.mean(s['z']))))):
        P = st['P']; n = len(P)
        if n < 6: continue
        wt = np.asarray(st['wt'], float) * (1 - .42 * (st['nr'] @ l2)); base = (5.0 if st['kind'] == 'out' else 3.2 if st['kind'] == 'touch' else 2.9) * lw * (.75 if st['inner'] else 1.)
        if len(wt) != n: wt = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(wt)), wt)
        ker = np.hanning(11); ker /= ker.sum(); wt = np.convolve(np.pad(wt, 5, 'edge'), ker, 'valid'); t = np.linspace(0, 1, n); tp = min(.22, 16. / max(n * 1.5, 1))
        if not st['closed']: wt = wt * np.clip(.22 + .78 * np.minimum(t, 1 - t) / tp, .22, 1)
        Q = np.vstack([P, P[:1]]) if st['closed'] else P; w_ = np.r_[wt, wt[:1]] if st['closed'] else wt
        k = np.linspace(0, len(w_) - 1, max(6, min(60, len(w_) // 3))).astype(int)
        R = stroke_poly(Q, w_[k] * base, seed * 131 + j, .4, 1.4, .25)
        if R is not None: sh.k.fill(R, BLACK)
    return sh
