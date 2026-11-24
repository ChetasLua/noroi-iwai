"""fx curse: cursed energy, drawn blue (or black-blue) flame that clings to a contour and streams with motion."""
from .fx_core import *
from .kit import contours
from .fx_fire import _tongue_spine

CURSE = dict(out='#1b22b8', body='#2a6bff', hot='#5fd0ff', core='#f0feff', ink='#080a3a', glow='#2f6cff')
BLACKBLUE = dict(out='#1d46ff', body='#0a0b22', hot='#050512', core='#8fd2ff', ink='#020208', glow='#2350ff')
VIOLET = dict(out='#5a1fb0', body='#8a3cf0', hot='#c08bff', core='#ffffff', ink='#1a0536', glow='#7d3cf0')
BLOOD = dict(out='#5a0512', body='#b80a22', hot='#0a0306', core='#ff5a5a', ink='#050103', glow='#d0122c')


def limb_poly(P, r0, r1=None, n=40):
    """capsule polygon round a limb poly-line, radius r0 -> r1"""
    p = Path(P, 4.); r1 = r0 if r1 is None else r1; s = np.linspace(0, p.L, n); rr = r0 + (r1 - r0) * s / p.L
    a = np.linspace(np.pi / 2, -np.pi / 2, 12); e = p.at(p.L); T = p.T[-1]; N = p.N[-1]; cap = e + (np.cos(a)[:, None] * T + np.sin(a)[:, None] * N) * r1
    e0 = p.at(0.); T0 = p.T[0]; N0 = p.N[0]; cap0 = e0 + (-np.cos(a)[:, None] * T0 - np.sin(a)[:, None] * N0) * r0
    return np.vstack([p.at(s, rr), cap, p.at(s, -rr)[::-1], cap0])


def _runs(P, stream):
    """closed contour -> list of open runs, each oriented along the stream, outward normal known; [(poly, sign)]"""
    P = np.asarray(P, float)
    if np.hypot(*(P[0] - P[-1])) < 1e-6: P = P[:-1]
    P = resample(np.vstack([P, P[:1]]), 6.)[:-1]; n = len(P)
    area = .5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    T = np.roll(P, -2, 0) - np.roll(P, 2, 0); T /= np.hypot(*T.T)[:, None] + 1e-9; dt = T @ stream; k = np.hanning(15); k /= k.sum(); dt = np.convolve(np.r_[dt[-7:], dt, dt[:7]], k, 'same')[7:-7]
    sg = np.where(dt >= 0, 1, -1); cut = [i for i in range(n) if sg[i] != sg[i - 1]]
    if not cut: return [(np.vstack([P, P[:1]]), 1 if area > 0 else -1)]
    out = []
    for a, b in zip(cut, cut[1:] + [cut[0] + n]):
        q = P[[i % n for i in range(a, b + 1)]]
        if len(q) < 4: continue
        o = 1 if area > 0 else -1                        # y-down: area>0 = clockwise on screen, Path.N (T[1],-T[0]) points outward
        if sg[a % n] < 0: q = q[::-1]; o = -o
        out.append((q, o))
    return out


def cursed(shape, f, stream=(0, -1), size=70., pal=CURSE, seed=0, amount=1., wisps=1., step=2, speed=520., sheath=.19, lw=1.6, light=.9, w=W, h=H, style='blue', front=True, wispw=.23):
    """Cursed energy round a shape.  shape = premult RGBA layer / float mask / polygon / list of polygons.  f frame (drawn on `step`s).
    stream = direction the energy flows (up; use minus the velocity on motion, its length 0..3 stretches wisps).  size = tallest wisp px.
    -> (back FX, front FX): put back, then the cel, then front (thin licks + hairline core over the edge)"""
    k = int(f) // step; t = k * step / 24.; st = np.asarray(stream, float); m0 = np.hypot(*st) + 1e-6; sd = st / m0; stretch = .55 + .45 * min(m0, 3.)
    if isinstance(shape, np.ndarray) and shape.ndim == 3: shape = shape[..., 3]
    if isinstance(shape, np.ndarray) and shape.shape[1] > 2 and shape.ndim == 2 and shape.shape[0] > 64:
        m = cv2.resize(shape, (shape.shape[1] // 2, shape.shape[0] // 2), interpolation=cv2.INTER_AREA); polys = [c * 2 for c in contours(m, min_len=40)]
    else: polys = shape if isinstance(shape, list) else [shape]
    A = [[], [], [], []]; fr = []
    for pi, P in enumerate(polys):
        for ri, (q, o) in enumerate(_runs(P, sd)):
            p = Path(q, 6.); L = p.L; sdk = seed * 71 + pi * 13 + ri
            for li, (kw, ph) in enumerate(((1.30, 0.), (.80, 3.), (.34, 6.))):      # ragged sheath: three nested lick edges
                base = size * sheath * kw * amount; pit = size * (.95 - .2 * li)
                wf = lambda s: base * (.55 + .45 * noise(np.asarray(s) / (size * .9) - t * speed / size * .8, ri + ph, t * 2, seed)) + 1.2
                tt = [(a, b, base * z * 2.3 * min(1., max(.05, b / (pit * 1.6))), kd) for a, b, z, kd in teeth(L, pit, sdk + li * 5, t * speed, 1., 1.6, t, .4) if a > -pit * .2]
                E = lick_edge(L, wf, tt, s_end=L); edge = p.at(E[:, 0], E[:, 1] * o); inner = p.at(np.linspace(L, 0, max(4, int(L / 14))), -size * .05 * o); A[min(li, 2) + (0 if li < 2 else 0)].append(np.vstack([edge, inner]))
            cell = size * .30 / max(wisps, .05)                                     # wisps: tongues torn off the contour
            for i in range(int(L / cell) + 1):
                u = (i + h1(sdk, i, 1)) * cell
                if u > L: continue
                N = np.array([np.interp(u, p.s, p.N[:, 0]), np.interp(u, p.s, p.N[:, 1])]) * o; nd = N @ sd; hz = 2.2 + 1.6 * h1(sdk, i, 2); tau = (t * hz + h1(sdk, i, 3)) % 1.; cyc = int(t * hz + h1(sdk, i, 3)); z = bms(h1(sdk, i, cyc, 4), .2, .35)
                ln = size * 1.35 * amount * stretch * (.18 + .82 * z) * (.42 + .58 * min(1., max(0., nd + .35))) * math.sin(math.pi * min(1., tau * 1.15 + .12)) ** .6
                if ln < 8: continue
                d = N * (.75 if nd > 0 else .45) + sd * (.55 + .5 * stretch); th = math.atan2(d[1], d[0]); r0 = p.at(u) - N * size * .05 + sd * ln * .25 * tau
                sp = _tongue_spine(r0[0], r0[1], ln, th, t, sdk + i * 3 + cyc, .13, .9, 2.6, 18); b = ln * wispw * (1.15 - .25 * z) + 2
                side = (1., .55) if h1(sdk, i, 6) > .5 else (.55, 1.)
                mk = lambda kw, kl: band(Path(sp, 3.).sub(0, Path(sp, 3.).L * kl, 0, 18), (lambda v: b * kw * side[0] * tongue(v), lambda v: b * kw * side[1] * tongue(v, .34, 1.7)), sdk + i, t * ln * 3, max(20., ln * .3), .5, 1.5, t)
                A[0].append(mk(1.35, 1.06)); A[1].append(mk(.95, .92))
                if z > .4: A[2].append(mk(.45, .66))
                if tau > .55 and h1(sdk, i, 7) < .7:                                # torn tip
                    e = Path(_tongue_spine(r0[0], r0[1], ln * 1.55, th, t, sdk + i * 3 + cyc, .13, .9, 2.6, 24), 3.); a = .62 + .3 * (tau - .55)
                    q2 = e.sub(e.L * a, e.L * min(1., a + .32), 0, 12); bb = b * .34 * (1 - (tau - .55) / .5); A[0].append(band(q2, lambda v: bb * 1.4 * leaf(v), sdk, 0, 30, .3)); A[1].append(band(q2, lambda v: bb * leaf(v), sdk, 0, 30, .3))
                if front and h1(sdk, i, cyc, 8) < .30:                              # licks crossing in front of the limb
                    s2 = p.at(u) - N * size * (.10 + .25 * h1(sdk, i, 9)); fr.append(_tongue_spine(s2[0], s2[1], ln * .8, math.atan2(sd[1], sd[0]) + .3 * (h1(sdk, i) - .5), t, sdk + i, .14, 1., 2.6, 16))
    cel = Cel(w, h)
    for P in A[0]: cel.fill(P, pal['out'])
    for P in A[1]: cel.fill(P, pal['body'])
    for P in A[2]: cel.fill(P, pal['hot'])
    for j, P in enumerate(A[1][::3]): ink_broken(cel, P, lw, pal['ink'], seed + j, .35, (30, 110))
    back = cel.layer(h, w); gl = glow_poly(A[0], pal['glow'], (12, size * .55), (.45, .45), light * amount, h, w); gl *= (1 - .55 * back[..., 3:4])
    fx = None
    if front:
        cel = Cel(w, h)
        for j, sp in enumerate(fr):
            l = poly_len(sp); b = l * .055 + 1; cel.fill(band(sp, lambda v: b * 1.6 * leaf(v, .35, .8, 1.1), seed + j, 0, 30, .4), pal['body'], .9); cel.fill(band(sp, lambda v: b * .65 * leaf(v, .4, .8, 1.1), seed + j, 0, 30, .4), pal['core'])
        for pi, P in enumerate(polys):                                              # hard hairline core broken along the contour
            q = Path(np.vstack([P, P[:1]]), 5.)
            for i in range(int(q.L / (size * .8)) + 1):
                if h1(seed, pi, i, k, 12) > .55 * amount: continue
                a = (i + h1(seed, i, k)) * size * .8; b = min(q.L, a + size * (.25 + .6 * h1(seed, i, k, 1)))
                if b - a > 10: cel.line(q.sub(a, b, 0, 14), lw * (1.0 + 1.2 * h1(seed, i, k, 2)), pal['core'], taper=(.4, .5), seed=i + k, press=.4)
        fl = cel.layer(h, w); fx = FX(fl, hot(fl, .3, (4, 14, 40), (.5, .3, .15), pal['glow'], .8))
    return FX(back, gl), fx


def aura(mask, r=70., inner='#19e6ff', mid='#1565ff', outer='#7d4dff', hard=5, amt=1., f=0, rag=.35, seed=0):
    """ref-3 glow aura behind a cel: thin hard cyan band on the contour, blue, violet falling off; ragged by slow noise. -> additive-over FX (put before the cel)"""
    a = mask[..., 3] if mask.ndim == 3 else mask; h, w = a.shape; q = cv2.resize(a, (w // 4, h // 4), interpolation=cv2.INTER_AREA); n = noise_img(h // 4, w // 4, r * .30, f / 24. * 1.2, seed, 2, 2)
    d = cv2.distanceTransform((q < .5).astype(np.uint8), cv2.DIST_L2, 5) * 4; d = cv2.GaussianBlur(d, (0, 0), 1.2) * (1 + rag * n); e = lambda s: np.exp(-(np.clip(d, 0, 1e4) / s) ** 1.25)
    c = e(r * .22)[..., None] * C(inner) + (e(r * .62) - e(r * .22) * .6)[..., None] * C(mid) + (e(r * 1.25) - e(r * .62) * .75)[..., None] * C(outer) * .7; al = np.clip(e(r * 1.1) * 1.15, 0, 1)
    c, al = cv2.resize(np.clip(c, 0, 1.2).astype(f32), (w, h)), cv2.resize(al.astype(f32), (w, h)); o = np.dstack([c * al[..., None], al * .9]) * amt
    if hard: b = cv2.dilate(a, np.ones((3, 3), np.uint8), iterations=int(hard)); b = cv2.GaussianBlur(b, (0, 0), .8)[..., None]; o = o * (1 - b) + np.dstack([np.ones_like(a)[..., None] * (C(inner) * .8 + .2), a * 0 + 1])* b * amt
    return FX(o.astype(f32), None)
