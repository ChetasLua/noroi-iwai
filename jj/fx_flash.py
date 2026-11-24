"""fx flash: Black Flash (kokusen), angular bolts, shards, distortion ring, higanbana bolts, slashes, plate cutting, impact frames."""
from .fx_core import *
from .comp import blur

BK = '#060308'; RED = '#f0142e'; DRED = '#8e0618'; BONE = '#fff6f0'


# ------------------------------------------------------------------ bolts
def bolt_line(p0, p1, seed, jag=.30, n=None, back=.22):
    """angular lightning poly-line: few segments, unequal lengths, hard elbows, sometimes doubling back (Z)"""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float); d = p1 - p0; L = np.hypot(*d) + 1e-6; d = d / L; nrm = np.array([-d[1], d[0]])
    r = R(seed, 3); n = n or max(3, int(3 + L / 190. + r.integers(0, 2)))
    seg = r.choice([.45, .7, 1.0, 1.0, 1.7], n) * r.uniform(.8, 1.2, n); u = np.r_[0, np.cumsum(seg)]; u /= u[-1]
    sg = 1 if r.random() < .5 else -1; P = [p0]
    for i in range(1, n):
        a = u[i] * L; o = sg * jag * L / n * r.uniform(.35, 1.5) * (1.15 - .35 * u[i])
        if r.random() < back: a -= L / n * r.uniform(.45, 1.0); o *= 1.35
        P.append(p0 + d * a + nrm * o)
        if r.random() < .84: sg = -sg
    P.append(p1); return np.array(P)


def kite(p0, p1, w, r, over=(0., 0.), pk=None, bow=0.):
    """one hard shard from p0 to p1 (needle ends, fattest at pk), asymmetric sides"""
    d = p1 - p0; L = np.hypot(*d) + 1e-6; d = d / L; m = np.array([-d[1], d[0]]); a = p0 - d * over[0]; e = p1 + d * over[1]; pk = r.uniform(.18, .55) if pk is None else pk
    q0, q1 = a + (e - a) * pk, a + (e - a) * min(.92, pk + r.uniform(.04, .22)); k = r.uniform(.15, .85)
    return np.array([a, q0 + m * w * 2 * k + m * bow, q1 + m * w * 2 * k * r.uniform(.45, .9), e, q1 - m * w * 2 * (1 - k) * r.uniform(.4, .8), q0 - m * w * 2 * (1 - k)])


def bolt_poly(P, w, seed, spur=.6, thin=.36, edge=0.):
    """a bolt as a chain of kites: thick-thin-thick like brush cuts, every elbow overshoots into a spur.  -> [(black, red)] polygon pairs"""
    r = R(seed, 9); n = len(P) - 1; out = []; s = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]; s = s / (s[-1] + 1e-6); tk = r.random() < .5
    for i in range(n):
        l = np.hypot(*(P[i + 1] - P[i])); fat = (1. if (i % 2 == 0) == tk else thin) * r.uniform(.7, 1.3); ww = max(.7, w * fat * (1 - s[i]) ** .55); ww = min(ww, l * (.20 if i == 0 else .115) + 1.0)
        ov0 = 0. if i == 0 else min(l * .5, ww * 2.2 + 6) * (r.uniform(.3, 1.5) if r.random() < spur else .25); ov1 = (min(l * .5, ww * 2.0 + 5) * (r.uniform(.3, 1.3) if r.random() < spur else .22)) if i < n - 1 else 0.
        rs = int(r.integers(1 << 30)); K = kite(P[i], P[i + 1], ww, R(rs), (ov0, ov1)); e0 = edge * (.45 + .55 * min(1., ww / 10.))
        K2 = kite(P[i] + r.normal(0, e0 * .45, 2), P[i + 1] + r.normal(0, e0 * .45, 2), ww + e0 * r.uniform(.35, 1.2), R(rs), (ov0 * 1.2 + e0 * 2, ov1 * 1.2 + e0 * (2 if i < n - 1 else 5)))
        out.append((K, K2))
    return out


def bolt_tree(p0, p1, w, seed, jag=.30, branch=.55, depth=2, spread=.75, blen=.5, edge=6.):
    """main bolt + angular branches -> (list of (black, red) polygon pairs, list of centre polylines)"""
    P = bolt_line(p0, p1, seed, jag); polys = bolt_poly(P, w, seed, edge=edge); lines = [P]; r = R(seed, 21)
    if depth > 0:
        L = np.hypot(*(np.asarray(p1, float) - p0))
        for i in range(1, len(P) - 1):
            if r.random() > branch: continue
            t = P[i + 1] - P[i]; th = math.atan2(*t[::-1]) + r.choice([-1, 1]) * r.uniform(.35, 1.) * spread; l = L * blen * r.uniform(.4, 1.) * (1 - i / len(P) * .6)
            q, ln = bolt_tree(P[i], P[i] + np.array([math.cos(th), math.sin(th)]) * l, w * r.uniform(.25, .5), seed * 5 + i + 1, jag * 1.05, branch * .45, depth - 1, spread, blen, edge * .6)
            polys += q; lines += ln
    return polys, lines


def star_poly(cx, cy, r0, r1, n, seed, asp=1.0, rot=0., skew=None, sharp=.5):
    """angular burst: unequal spikes (big / medium / small), straight sides.  skew = (angle, gain) stretches spikes along a direction"""
    r = R(seed, 4); P = []
    for i in range(n):
        a = (i + .5 + sharp * (r.random() - .5)) / n * 6.283 + rot; z = bms(r.random(), .24, .34)
        ln = r0 + (r1 - r0) * z
        if skew is not None: ln *= 1 + skew[1] * abs(math.cos(a - skew[0])) ** 3
        P.append((cx + math.cos(a) * ln, cy + math.sin(a) * ln * asp))
        b = (i + 1 + .35 * (r.random() - .5)) / n * 6.283 + rot; q = r0 * r.uniform(.72, 1.1); P.append((cx + math.cos(b) * q, cy + math.sin(b) * q * asp))
    return np.array(P)


def _stroke(cel, polys, w, col, alpha=1.0, miter=6.):
    """hard-joined outline stroke (mitre corners stay needle sharp)"""
    import cairo
    c = cel.ctx; c.save(); c.set_line_join(cairo.LINE_JOIN_MITER); c.set_miter_limit(miter); c.set_line_width(w); cel._src(col, alpha)
    for P in polys:
        cel._path(np.asarray(P)); c.stroke()
    c.restore()


def bolts_draw(cel, polys, seed=0, black=BK, red=RED, dred=DRED, lines=None, white=.5, hw=1.6, glowpass=True):
    """kokusen bolts: uneven red edge shapes, black kites, white hairline running off-centre"""
    for K, K2 in polys: cel.fill(K2, red)
    for K, K2 in polys: cel.fill(K, black)
    if lines is not None: hairlines(cel, lines, seed, white, hw)
    return cel


def hairlines(cel, lines, seed=0, prob=.5, hw=1.8, col=BONE, off=2.5):
    r = R(seed, 33)
    for k, P in enumerate(lines):
        if r.random() > prob: continue
        a = int(r.integers(0, max(1, len(P) - 2))); q = P[a:a + int(r.integers(2, 4))]
        if len(q) < 2: continue
        Q = np.vstack([np.linspace(q[i], q[i + 1], 10) for i in range(len(q) - 1)]); cel.line(Q + r.normal(0, off, 2), hw * r.uniform(.7, 1.5), col, taper=(.1, .5), press=.5, seed=seed + k, tremor=.2)
    return cel


def kite_star(cx, cy, r0, r1, n, seed, skew=None, w=.10, edge=8., asp=1.):
    """burst of kites from a point: unequal, clumped, some detached from the centre -> [(black, red)]"""
    r = R(seed, 41); out = []; cl = r.uniform(0, 6.283, max(3, n // 5))
    for i in range(n):
        a = r.choice(cl) + r.normal(0, .22) if r.random() < .45 else (i + r.random()) / n * 6.283
        z = bms(r.random(), .25, .35); ln = r0 + (r1 - r0) * z
        if skew is not None: ln *= 1 + skew[1] * abs(math.cos(a - skew[0])) ** 3
        d = np.array([math.cos(a), math.sin(a) * asp]); st = 0. if r.random() < .75 else r0 * r.uniform(.4, 1.3); c = np.array([cx, cy]); ww = ln * w * r.uniform(.35, 1.) * (.5 + .5 * z); rs = int(r.integers(1 << 30)); pk = r.uniform(.10, .42)
        out.append((kite(c + d * st, c + d * ln, ww, R(rs), (r0 * .4 if st == 0 else 0, 0), pk), kite(c + d * st + r.normal(0, edge * .3, 2), c + d * (ln + edge * 3), ww + edge * r.uniform(.3, 1.1), R(rs), (r0 * .4 if st == 0 else edge * 2, 0), pk)))
    return out


def shards(cel, cx, cy, t, seed=0, n=40, speed=900., spin=6., size=26., dirn=None, aniso=0., r0=40., grav=0., edge=2.2, black=BK, red=RED, drag=2.2):
    """angular fragments thrown outward; t seconds since the hit"""
    k = (1 - math.exp(-drag * t)) / drag
    for i in range(n):
        a = h1(seed, i, 1) * 6.283; z = bms(h1(seed, i, 2), .2, .35); v = speed * (.25 + 1.0 * h1(seed, i, 3))
        if dirn is not None: v *= 1 + aniso * max(-.6, math.cos(a - dirn))
        x = cx + math.cos(a) * (r0 + v * k); y = cy + math.sin(a) * (r0 + v * k) + .5 * grav * t * t
        s = size * z * max(0., 1 - t * (.55 + .9 * h1(seed, i, 4))); 
        if s < 1.2: continue
        rot = h1(seed, i, 5) * 6.283 + spin * t * (h1(seed, i, 6) - .5) * 4; m = 3 + int(h1(seed, i, 7) * 1.3); th = np.sort(np.array([h1(seed, i, 10 + j) for j in range(m)])) * 6.283 + rot
        st = 1 + 2.2 * h1(seed, i, 8); P = np.c_[np.cos(th) * s * st, np.sin(th) * s]; c, q = math.cos(a), math.sin(a); P = np.c_[P[:, 0] * c - P[:, 1] * q + x, P[:, 0] * q + P[:, 1] * c + y]
        if edge > 0: cel.fill(P + (P - P.mean(0)) * (edge * 1.1 / max(s, 1.)) + np.array([h1(seed, i, 20) - .5, h1(seed, i, 21) - .5]) * edge * 2.4, red)
        cel.fill(P, black)
    return cel


def radial_wedges(cel, cx, cy, n, r0, r1, seed, w=.02, col=BK, asp=.8, clump=.6, a0=0., a1=6.283):
    """kanada-style radial wedges: clumped, unequal, straight-sided, tips at different radii"""
    r = R(seed, 6); c = r.uniform(a0, a1, max(3, n // 6))
    for i in range(n):
        a = r.choice(c) + r.normal(0, .16) if r.random() < clump else r.uniform(a0, a1)
        z = bms(r.random(), .12, .3); q = r0 * r.uniform(.8, 1.9) * (1.15 - .3 * z); hw = w * z * r.uniform(.6, 1.5) + .0014
        far = r1 * 1.1; cel.fill(np.array([[cx + math.cos(a + r.normal(0, hw * .3)) * q, cy + math.sin(a) * q * asp], [cx + math.cos(a - hw) * far, cy + math.sin(a - hw) * far], [cx + math.cos(a + hw) * far, cy + math.sin(a + hw) * far]]), col)
    return cel


# ------------------------------------------------------------------ distortion
def distort_ring(img, cx, cy, rad, width, amp, seed=0, asp=1.0, ca=3.0, rag=.12, t=0.):
    """space ripples in a ragged ring: radial refraction + colour split at the ring"""
    img = img.astype(f32); cx, cy, rad, width, amp = float(cx), float(cy), float(rad), float(width), float(amp); h, w = img.shape[:2]; x, y = grid(h, w); dx, dy = x - cx, (y - cy) / asp; d = np.sqrt(dx * dx + dy * dy) + 1e-4; a = np.arctan2(dy, dx)
    tb = np.linspace(-np.pi, np.pi, 721); nz = noise(np.cos(tb) * 3.2, np.sin(tb) * 3.2, t, seed) + .5 * noise(np.cos(tb) * 9, np.sin(tb) * 9, t, seed + 2)
    nz[-1] = nz[0]; rr = rad * (1 + rag * np.interp(a.ravel(), tb, nz).reshape(a.shape).astype(f32))
    u = (d - rr) / width; k = (np.exp(-u * u) * np.sin(u * 2.6) * amp).astype(f32); ux, uy = dx / d * k, dy * asp / d * k
    out = cv2.remap(img, x + ux, y + uy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    if ca > 0:
        e = np.exp(-u * u).astype(f32) * ca
        for c, s in ((0, 1), (2, -1)): out[..., c] = cv2.remap(img[..., c], x + ux + dx / d * e * s, y + uy + dy / d * e * s, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return out


# ------------------------------------------------------------------ the sequence
def _flash_bolts(f, cx, cy, scale, seed, dirn, n, reach, aniso):
    """bolts alive at frame f (2..): [(polys, lines, age, tear)]; redrawn on ones for the first four frames, then on twos"""
    out = []; step = 1 if f < 6 else 2; k = f // step * step
    for i in range(n):
        big = i < 4; b = 2 + int(h1(seed, i, 1) * 5.5) * (0 if i < n * .55 else 1); life = 7 + int(5 * h1(seed, i, 2)) + (6 if big else 0); age = k - b
        if age < 0 or age > life: continue
        a = (i * 2.399 + .9 * h1(seed, i, 3)) + seed; z = 1.1 if big else bms(h1(seed, i, 4), .20, .45)
        if dirn is not None and big: a = dirn + (0, math.pi)[i % 2] + (h1(seed, i, 5) - .5) * 1.1
        g = 1 + aniso * abs(math.cos(a - dirn)) ** 2 if dirn is not None else 1.
        ln = reach * scale * (.34 + .72 * z) * g; grow = min(1., (age + 1.2) / 2.2); tear = max(0., (age - 2.5) / max(1., life - 2.5))
        d = np.array([math.cos(a), math.sin(a)]); c = np.array([cx, cy]); p0 = c + d * ln * .72 * tear ** 1.6; p1 = c + d * ln * (grow + .08 * tear)
        w = scale * (44. if big else 5 + 17. * z) * (1 - .85 * tear) * (1 if age > 0 else .75)
        polys, lines = bolt_tree(p0, p1, w, seed * 991 + i * 17 + k * (1 if age < 4 else 0) + 5, .42, .45 if tear < .4 else .2, 2 if big else 1, edge=scale * (11 if big else 6) * (1 - .6 * tear))
        out.append((polys, lines, age, tear))
    return out


def black_flash_layers(f, cx, cy, scale=1., seed=0, dirn=0., n=15, reach=900., aniso=.7, star=True, part='all', core=True, shard=1.):
    """vector part of the Black Flash at local frame f (0 = spark, 2 = burst ... 20 = last after-image) -> FX.
    part: 'all' | 'back' (star + most bolts: put it between victim and fist) | 'front' (a few bolts, hairlines, shards over the fist)"""
    cel = Cel(); s = scale; step = 1 if f < 6 else 2; k = f // step * step
    if f < 2:                                              # converging hairlines and a glint
        if part == 'back': return FX(None, None)
        r = R(seed, 100 + f)
        for i in range(9):
            a = r.uniform(0, 6.283); l0 = s * r.uniform(120, 420) * (1 - .3 * f); P = bolt_line((cx + math.cos(a) * l0, cy + math.sin(a) * l0), (cx + math.cos(a) * 8, cy + math.sin(a) * 8), seed * 7 + i + f * 31, .5)
            bolts_draw(cel, bolt_poly(P, 3.0 * s, i + f, .7, edge=2.4))
        cel.fill(star_poly(cx, cy, 5 * s, s * (70 + 90 * f), 4, seed, rot=.4, sharp=.1), BONE)
        return FX(cel.layer(), glow_poly([star_poly(cx, cy, 8 * s, 60 * s, 6, seed)], RED, (20, 70), amt=.6))
    B = _flash_bolts(f, cx, cy, s, seed, dirn, n, reach, aniso)
    if part != 'all': B = [q for j, q in enumerate(B) if (j % 5 == 4) == (part == 'front')]
    if f < 14 and part != 'front':                         # far thin red echoes behind
        r = R(seed, 300 + k)
        for i in range(12 - f // 2):
            a = r.uniform(0, 6.283); l1 = reach * s * r.uniform(.5, 1.3); l0 = l1 * r.uniform(.2, .6); c = np.array([cx, cy]); d = np.array([math.cos(a), math.sin(a)])
            for K, _ in bolt_poly(bolt_line(c + d * l0, c + d * l1, seed + i * 3 + k, .5), 2.6 * s, i + k, .6): cel.fill(K, RED if i % 3 else DRED)
    allp = [q for b in B for q in b[0]]; glow = [K for K, _ in allp]
    S = []; S2 = []; R0 = 1.
    if star and f < 14 and part != 'front':
        q = (f - 2) / 12.; R1 = s * (300 + 260 * min(1, (f - 1) / 3.)) * (1 - q * .6); R0 = s * (70 + 34 * min(1, (f - 2) / 2.)) * (1 - .55 * q)
        S = kite_star(cx, cy, R0, R1, int(26 - 12 * q), seed + k * 3, (dirn or 0., .9), .13 * (1 - .45 * q), 10 * s); cr = star_poly(cx, cy, R0 * .8, R0 * 1.6, 11, seed + k * 3, rot=.3 * k)
        S.append((cr, cr + (cr - [cx, cy]) * .10))
        if f < 8 and core: S2 = kite_star(cx, cy, R0 * .5, R1 * .62, 15, seed + k * 5 + 1, (dirn or 0., .6), .055, 0); S2.append((star_poly(cx, cy, R0 * .55, R0 * 1.15, 9, seed + k, rot=k),) * 2)
    for K, K2 in allp + S: cel.fill(K2 + (K2 - K2.mean(0)) * .05, DRED)
    for K, K2 in allp + S: cel.fill(K2, RED)
    for K, K2 in S + allp: cel.fill(K, BK)
    for K, _ in (S2 if f < 6 else S2[:-1]): cel.fill(K, BONE if f < 6 else RED)
    if S2 and f < 6: cel.fill(star_poly(cx, cy, R0 * .22, R0 * .62, 8, seed + k + 7, rot=k), BK)
    for j, (polys, lines, age, tear) in enumerate(B):
        hairlines(cel, lines[:4], seed + j * 5 + k, .62 - .3 * tear, 2.3 * s)
        if tear > .35:                                     # after-image: red hairline where the bolt was
            P = lines[0]; Q = np.vstack([np.linspace(P[i], P[i + 1], 8) for i in range(len(P) - 1)]); cel.line(Q, 1.4 * s, RED, taper=(.3, .5), seed=j, alpha=max(0., 1.1 - tear))
    if f > 5 and part != 'back':                            # after-image: a cracked net of red hairlines that hangs, drifts out and thins
        g = (f - 5) / 15.; kk = 4
        for i in range(13):
            a = i * 2.399 + seed; l = reach * s * (.25 + .75 * h1(seed, i, 61)) * (1 + .06 * g); st = .10 + .5 * g * h1(seed, i, 62); c = np.array([cx, cy]); d = np.array([math.cos(a), math.sin(a)])
            for K, _ in bolt_poly(bolt_line(c + d * l * st, c + d * l, seed * 5 + i, .42), (3.6 * (1 - g) ** 1.2 + .5) * s, seed + i + kk, .5): cel.fill(K, RED if i % 3 else BK, max(0., 1 - g * .7))
    if shard > 0 and part != ('back' if shard >= 1. else 'front'): shards(cel, cx, cy, (f - 2) / 24. + .02, seed + 2, int(52 * min(1, s + .3) * shard), 1300 * s, 7., 34 * s, dirn, .4, 170 * s, drag=1.3, grav=500 * s)
    L = glow_poly(glow + [K for K, _ in S], RED, (9 * s, 40 * s), (.30, .34), amt=max(.0, 1. - (f - 2) / 20.) * .9) if glow or S else None; lay = cel.layer()
    if L is not None: L *= (1 - .9 * lay[..., 3:4])
    return FX(lay, L)


def impact_tone(base, mode, lo=.16, hi=.46, accent=RED):
    """2/3-tone drawing of a finished picture, hard cut: 'bw' paper, 'inv' black, 'red' red field with black, 'blood' black with red lights"""
    L = cv2.GaussianBlur(cv2.cvtColor(np.clip(base, 0, 1).astype(f32), cv2.COLOR_RGB2GRAY), (0, 0), 1.2)
    a = smooth(lo - .02, lo + .02, L)[..., None]; b = smooth(hi - .02, hi + .02, L)[..., None]
    k, w, r = C(BK), C(BONE), C(accent)
    if mode == 'bw': return k * (1 - a) + w * a
    if mode == 'inv': return w * (1 - a) + k * a
    if mode == 'red': return k * (1 - a) + (r * (1 - b) + w * b) * a
    return k * (1 - b) + r * b


def black_flash(base, f, cx, cy, scale=1., seed=0, dirn=0., reach=900., aniso=.7, n=15, dark=.72, subject=None, tone=True, ring=True, over=None):
    """Full Black Flash over a finished RGB picture.  f: 0-1 spark (world dims), 2 white drawing, 3 black/white inversion, 4 red drawing, 5-11 bolts burst,
    12-20 after-image.  dirn = direction of the punch (radians).  subject = optional premult layer (attacker + victim) kept readable in the tone frames.
    over = optional premult layer (the fist and arm) that stays in front of the star and most bolts.
    Outside 0..20 returns base."""
    if f < 0 or f > 21: return base
    s = scale; step = 1 if f < 6 else 2; k = f // step * step
    if f < 2:
        g = cv2.cvtColor(base.astype(f32), cv2.COLOR_RGB2GRAY); im = (base * .25 + g[..., None] * .75) * (.62 - .22 * f)
        return put(im, black_flash_layers(f, cx, cy, s, seed, dirn))
    if tone and f in (2, 3, 4):
        mode = {2: 'bw', 3: 'inv', 4: 'red'}[f]; cel = Cel()
        if f == 2: im = np.empty_like(base); im[:] = C(BONE); radial_wedges(cel, cx, cy, 74, 250 * s, 2400, seed + 1, .030, BK)
        elif f == 3: im = np.empty_like(base); im[:] = C(BK); radial_wedges(cel, cx, cy, 60, 300 * s, 2400, seed + 2, .024, BONE); radial_wedges(cel, cx, cy, 26, 260 * s, 2400, seed + 3, .05, RED)
        else: im = np.empty_like(base); im[:] = C(RED); radial_wedges(cel, cx, cy, 90, 280 * s, 2400, seed + 4, .034, BK); radial_wedges(cel, cx, cy, 14, 320 * s, 2400, seed + 5, .012, BONE)
        im = put(im, cel.layer())
        if subject is not None:
            a = subject[..., 3:4]; col = np.where(a > .02, subject[..., :3] / np.maximum(a, .02), 0); tn = impact_tone(col, mode); im = im * (1 - a) + tn * a
        else: im = im * .0 + np.where(impact_tone(base, 'bw') < .5, impact_tone(base, mode), im) if f == 2 else im
        return _bf_put(im, f, cx, cy, s * 1.12, seed, dirn, n, reach, aniso, over, mode)
    q = min(1., max(0., (f - 4) / 14.)); e = math.exp(-(f - 5) / 5.) if f >= 5 else 1.
    g = cv2.cvtColor(base.astype(f32), cv2.COLOR_RGB2GRAY)[..., None]
    im = base * (1 - .55 * e) + g * .55 * e; im = im * (1 - dark * e); x, y = grid(); d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    im = im + (np.exp(-(d / (520 * s)) ** 2) * e * .55)[..., None] * C(RED) * (.25 + im)
    if ring and f >= 4:
        u = (f - 3) / 12.; rad = 1250 * s * (1 - math.exp(-2.3 * u)); amp = 70 * s * math.exp(-1.9 * u)
        if amp > 1.5: im = distort_ring(im, cx, cy, rad, 90 * s + 90 * s * u, amp, seed, t=k * .05)
    cel = Cel()
    if f < 9: radial_wedges(cel, cx, cy, int(46 * (1 - (f - 4) / 6.)) + 6, 520 * s, 2400, seed + 20 + k, .022, BK)
    im = put(im, cel.layer())
    if ring:                                               # drawn shock ring: two broken crescents
        u = (f - 4) / 12.; rad = 1300 * s * (1 - math.exp(-2.1 * max(u, .05))); c2 = Cel(); ragged_ring(c2, cx, cy, rad, rad * .86, 26 * s * math.exp(-2.4 * u) + 1, seed + k, BK, gaps=.35); ragged_ring(c2, cx, cy, rad * 1.015, rad * .87, 7 * s * math.exp(-2.2 * u) + .6, seed + k + 1, RED, gaps=.45); im = put(im, c2.layer(), op=max(0., 1 - u))
    return _bf_put(im, f, cx, cy, s, seed, dirn, n, reach, aniso, over, None)


def _bf_put(im, f, cx, cy, s, seed, dirn, n, reach, aniso, over, mode):
    if over is None: return put(im, black_flash_layers(f, cx, cy, s, seed, dirn, n, reach, aniso))
    im = put(im, black_flash_layers(f, cx, cy, s, seed, dirn, n, reach, aniso, part='back')); a = over[..., 3:4]
    if mode: col = np.where(a > .02, over[..., :3] / np.maximum(a, .02), 0); im = im * (1 - a) + impact_tone(col, mode) * a
    else: im = im * (1 - a) + over[..., :3]
    return put(im, black_flash_layers(f, cx, cy, s, seed, dirn, n, reach, aniso, part='front'))


def ragged_ring(cel, cx, cy, rx, ry, w, seed, col, gaps=.3, rot=0., rag=.05, n=7, spurs=.5, inner=None):
    """drawn shock ring: broken thick-thin crescents of a wobbly ellipse"""
    r = R(seed, 8); a0 = r.uniform(0, 6.283)
    for i in range(n):
        if r.random() < gaps * .8: continue
        s0 = a0 + i / n * 6.283; s1 = s0 + 6.283 / n * r.uniform(.55, 1.15); a = np.linspace(s0, s1, 40); v = np.linspace(0, 1, 40)
        wob = 1 + rag * noise(np.cos(a) * 2.4, np.sin(a) * 2.4, 0, seed) + rag * .5 * noise(np.cos(a) * 8, np.sin(a) * 8, 3, seed)
        hw = w * np.sin(np.pi * v) ** .6 * (1 + .5 * noise(v * 5, i * 3., 0, seed)); pk = r.uniform(.2, .8)
        E = lambda k: np.c_[np.cos(a) * (rx * wob + k), np.sin(a) * (ry * wob + k * ry / max(rx, 1e-3))]
        P = np.vstack([E(hw), E(-hw * .55)[::-1]])
        if rot: c, q = math.cos(rot), math.sin(rot); P = np.c_[P[:, 0] * c - P[:, 1] * q, P[:, 0] * q + P[:, 1] * c]
        cel.fill(P + [cx, cy], col)
        for j in range(int(spurs * 5)):
            m = int(r.integers(4, 36)); p = E(hw)[m]; th = a[m] + r.normal(0, .3); l = w * r.uniform(.6, 3.2); nn = np.array([-math.sin(th), math.cos(th)]) * w * r.uniform(.08, .3)
            q = np.array([p + nn, p + np.array([math.cos(th) * l, math.sin(th) * l * ry / max(rx, 1e-3)]), p - nn])
            if rot: c, z = math.cos(rot), math.sin(rot); q = np.c_[q[:, 0] * c - q[:, 1] * z, q[:, 0] * z + q[:, 1] * c]
            cel.fill(q + [cx, cy], col)
    return cel


# ------------------------------------------------------------------ higanbana (the slow Black Flash)
def _arc_path(p, th, L, curl, n=26, pw=1.4, wob=0.):
    v = np.linspace(0, 1, n); h = th + curl * v ** pw + wob * np.sin(v * 7); st = L / (n - 1)
    x = p[0] + np.r_[0, np.cumsum(np.cos(h[:-1]) * st)]; y = p[1] + np.r_[0, np.cumsum(np.sin(h[:-1]) * st)]
    return np.c_[x, y]


def _ease(t):
    t = min(max(t, 0.), 1.); return t * t * (3 - 2 * t)


def _opening(straight, curve, k):
    """bolt -> petal: jagged poly-line relaxes into the curve, root first"""
    n = len(curve); S = resample(straight, np.hypot(*np.diff(straight, axis=0).T).sum() / (n * 1.5)); S = np.c_[[np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(S)), S[:, i]) for i in (0, 1)]].T
    v = np.linspace(0, 1, n); m = np.clip(k * 1.9 - v * .9, 0, 1); m = (m * m * (3 - 2 * m))[:, None]
    return S * (1 - m) + curve * m


def higanbana(cel, cx, cy, size, t, seed=0, tilt=.72, florets=6, pal=(BK, RED, DRED, BONE), sway=0., rot=0., edge=None, stem=None, detail=1.0):
    """one red-spider-lily drawn out of black lightning.  t 0 = bolts only, 1 = fully open (petals curled back, stamens thrown wide).
    black petals edged red, red stamens with anthers.  tilt squashes the umbel (view angle).  -> polygons drawn far to near into cel"""
    bk, rd, dr, bo = pal; edge = edge or max(1.6, size * .012); it = []
    for i in range(florets):
        a = (i + .5 * h1(seed, i, 1)) / florets * 6.283 + rot; it.append((math.sin(a), a, i))
    it.sort(key=lambda q: q[0]); c = np.array([cx, cy], float)
    if stem is not None:
        P = np.asarray(stem, float); cel.fill(band(P, lambda v: size * .030 * (1 - .4 * v) + edge, seed, 0, 60, .1), rd); cel.fill(band(P, lambda v: size * .030 * (1 - .4 * v), seed, 0, 60, .1), bk)
    def proj(P):                                        # squash about the umbel centre, lift
        return c + (P - c) * [1, tilt]
    for pas in (0, 1):
        for s, a, i in it:
            near = s > 0; k0 = _ease(t * 1.55 - .42 * h1(seed, i, 2)); d = np.array([math.cos(a), math.sin(a)]); fc = c + d * size * .20 * (.3 + .7 * k0)
            if pas == 0:                                   # petals
                ps = []
                for j in range(6):
                    off = (j - 2.5) / 2.5; th = a + off * (.35 + .85 * k0) + (h1(seed, i, j, 3) - .5) * .25; L = size * (.54 + .26 * h1(seed, i, j, 4)) * (.45 + .55 * k0)
                    curl = (1 if off > 0 else -1) * (.4 + 2.0 * k0 * (.55 + .75 * h1(seed, i, j, 5))) * (.35 + .65 * abs(off)) * k0
                    crv = _arc_path(fc, th, L, curl, 24, 1.8, .10 * k0 * math.sin(j + sway)); st = bolt_line(fc, fc + np.array([math.cos(th), math.sin(th)]) * L * 1.15, seed * 31 + i * 7 + j, .5, 5)
                    ps.append((abs(off), proj(_opening(st, crv, k0)), L, j))
                ps.sort(key=lambda q: -q[0])
                for ao, P, L, j in ps:
                    w = L * (.028 + .062 * k0) * (1 - .25 * ao); wl = lambda v: w * (.18 + .82 * np.sin(np.pi * np.clip(v, 0, 1) ** .8) ** .9) * (1 - np.clip(v, 0, 1) ** 7) * (1 + .22 * np.sin(v * 17 + j) * k0)
                    pth = Path(P, 3.); sg = 1 if h1(seed, i, j, 6) > .5 else -1
                    pd = lambda e0, e1: band(pth, (lambda v: wl(v) + e0 * (1 - .6 * v), lambda v: wl(v) + e1 * (1 - .6 * v)), seed, 0, 60, 0., sides=(0, 0), wave=.03)
                    cel.fill(pd(edge * (2.6 if sg > 0 else .9), edge * (.9 if sg > 0 else 2.6)), rd); cel.fill(pd(0, 0), bk)
                    if detail and w > 3:
                        cel.line(pth.sub(pth.L * .08, pth.L * (.55 + .3 * h1(i, j)), lambda v: wl(v) * .12 * np.sin(v * 3)), max(1., w * .10), dr, taper=(.2, .6), seed=j)
                        cel.line(pth.sub(pth.L * .30, pth.L * .93, lambda v: sg * wl(.3 + .63 * v) * .62), max(.9, w * .07), rd, taper=(.4, .5), seed=j + 9)
            else:                                          # stamens over everything
                for j in range(int(5 * detail) + 1):
                    off = (j - 2.5) / 2.5; th = a + off * .55 * (.4 + .6 * k0); L = size * (.95 + .35 * h1(seed, i, j, 7)) * (.5 + .5 * k0); up = -math.pi / 2; dl = math.atan2(math.sin(up - th), math.cos(up - th))
                    crv = _arc_path(fc, th, L, dl * (.25 + .30 * h1(seed, i, j, 8)) * k0 + off * .25, 26, 1.6); st = bolt_line(fc, fc + np.array([math.cos(th), math.sin(th)]) * L, seed * 57 + i * 7 + j, .4, 5)
                    P = proj(_opening(st, crv, k0)); w = max(1.1, size * .0085); cel.line(P, w * 2.3, rd if j % 2 else dr, taper=(.05, .55), seed=j, tmin=.3, press=.1); cel.line(P, w, bk if j % 2 else rd, taper=(.05, .5), seed=j, press=.1)
                    if k0 > .35:
                        e = P[-1]; q = P[-1] - P[-4]; an = math.atan2(q[1], q[0]) + 1.2; l = size * .030 * k0; u = np.array([math.cos(an), math.sin(an)]); m = np.array([-u[1], u[0]])
                        cel.fill(spl(np.array([e - u * l, e + m * l * .42, e + u * l, e - m * l * .42]), True, n=5), rd); cel.line(np.array([e - u * l * .5 + m * l * .12, e + m * l * .16, e + u * l * .5 + m * l * .12]), max(.8, l * .18), bo, seed=j)
        if pas == 0:
            for _, a, i in it:                              # throat
                d = np.array([math.cos(a), math.sin(a) * tilt]); cel.fill(band(np.vstack([c, c + d * size * .10, c + d * size * .20]), lambda v: size * .012 * (1 - .4 * v), seed, 0, 40, 0.), bk)
    return cel


# ------------------------------------------------------------------ slashes
def slash_marks(cel, lines, f, w=3.2, seed=0, col='#ffffff', gap=9., gapcol='#07040a', hold=4, life=12, edge=None, stagger=0):
    """hair-thin cuts.  f frames since the cut: 0 drawn two thirds, 1 full and brightest, then the dark gap opens and closes.
    lines = [(x0,y0,x1,y1) or (p0,p1)]; stagger frames between lines"""
    for i, ln in enumerate(lines):
        q = np.asarray(ln, float).reshape(2, 2); k = f - i * stagger
        if k < 0 or k > life: continue
        p0, p1 = q; d = p1 - p0; L = np.hypot(*d); d = d / L; m = np.array([-d[1], d[0]]); r = R(seed, i)
        if k == 0: a, b = 0., .66
        else: a, b = min(1., max(0., (k - hold) / (life - hold)) * .5), 1. - min(1., max(0., (k - hold) / (life - hold)) * .45)
        A, B = p0 + d * L * a, p0 + d * L * b; mid = r.uniform(.35, .65); op = 1 - max(0., (k - hold) / float(life - hold))
        if k >= 2 and gap > 0:
            g = gap * min(1., (k - 1) / 2.) * op * r.uniform(.6, 1.3); M = A + (B - A) * mid; n = 24; v = np.linspace(0, 1, n); pr = np.sin(np.pi * v) ** .8
            up = A + (B - A) * v[:, None] + m * (pr * g * (1 + .25 * noise(v * 9, i, 0, seed)))[:, None]; dn = A + (B - A) * v[:, None] - m * (pr * g * .45)[:, None]
            cel.fill(np.vstack([up, dn[::-1]]), gapcol)
            if edge is not None: cel.fill(np.vstack([up, (up + m * (pr * 1.8)[:, None])[::-1]]), edge)
        ww = w * (1.7 if k == 1 else 1.15 if k == 0 else 1.) * (.25 + .75 * op) * r.uniform(.7, 1.2)
        cel.fill(np.array([A, A + (B - A) * mid + m * ww * .5, B, A + (B - A) * mid - m * ww * .5]), col)
        for j in range(2):                                 # torn sub-strokes
            u = r.uniform(.05, .8); l = r.uniform(.05, .18); o = r.normal(0, 5 + w); s = A + (B - A) * u + m * o; cel.fill(np.array([s, s + d * L * l * .5 + m * .7, s + d * L * l, s + d * L * l * .5 - m * .7]), col, .8)
    return cel


def slash_fx(lines, f, w=3.2, seed=0, glowcol='#ffffff', amt=.55, **k):
    cel = Cel(); slash_marks(cel, lines, f, w, seed, **k)
    lay = cel.layer(); L = hot(lay, .7, (3, 14, 40), (.5, .25, .15), glowcol, amt * max(0., 1 - f / 10.))
    return FX(lay, L)


def cut_plate(img, lines, t=1., drift=40., rot=.03, seed=0, under=None, gap='#050308', edge='#fff4ea', order=None, grav=0., offsets=None, edgew=1.6):
    """Cut a picture along straight lines and let the pieces part.  t 0 = whole, 1 = fully parted.  lines [(x0,y0,x1,y1)].
    Each side slides along its cut (clean leaning pieces) + rotates slightly; under = picture seen in the gaps (default dark).  offsets = optional [(dx,dy,rot)] per piece id."""
    h, w = img.shape[:2]; out = np.empty_like(img); out[:] = C(gap) if under is None else 0
    if under is not None: out[:] = under
    if t <= 0: return img.copy()
    y, x = np.mgrid[0:h, 0:w].astype(f32); lab = np.zeros((h, w), np.int32); P = []
    for i, ln in enumerate(lines):
        (x0, y0), (x1, y1) = np.asarray(ln, float).reshape(2, 2); d = np.array([x1 - x0, y1 - y0]); d /= np.hypot(*d); m = np.array([-d[1], d[0]]); P.append((np.array([x0, y0]), d, m))
        lab |= (((x - x0) * m[0] + (y - y0) * m[1]) > 0).astype(np.int32) << i
    ids = np.unique(lab); e = t * t * (3 - 2 * t); cell = None; hl = Cel(w, h)
    for j, q in enumerate(ids if order is None else order):
        msk = (lab == q); dx = dy = 0.; r = R(seed, int(q))
        for i, (p0, d, m) in enumerate(P):
            sg = 1 if (q >> i) & 1 else -1; dl = h1(seed, i, 3) * .25; ee = min(1., max(0., (e - dl) / (1 - dl)))
            s = sg * drift * ee * .5; dx += m[0] * s * .55 + d[0] * s * (1 if h1(seed, i) > .5 else -1); dy += m[1] * s * .55 + d[1] * s * (1 if h1(seed, i) > .5 else -1)
        ro = (r.random() - .5) * 2 * rot * e
        if offsets is not None: dx, dy, ro = offsets[j]
        dy += grav * e * e * r.random(); ys, xs = np.nonzero(msk[::4, ::4]); cx, cy = xs.mean() * 4, ys.mean() * 4; c, s = math.cos(ro), math.sin(ro); pad = int(abs(dx) + abs(dy) + abs(ro) * 1100 + 8); bx0, by0, bx1, by1 = max(0, xs.min() * 4 - pad), max(0, ys.min() * 4 - pad), min(w, xs.max() * 4 + pad + 4), min(h, ys.max() * 4 + pad + 4)
        M = np.float32([[c, -s, cx - c * cx + s * cy + dx - bx0], [s, c, cy - s * cx - c * cy + dy - by0]]); sz = (int(bx1 - bx0), int(by1 - by0))
        A = cv2.warpAffine(msk.astype(np.uint8) * 255, M, sz, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(f32) / 255.; Im = cv2.warpAffine(img, M, sz, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101); B = out[by0:by1, bx0:bx1]; a = A[..., None]; B[:] = B * (1 - a) + Im * a
        if edge and edgew > 0:                             # lit lip of the cut
            sh = cv2.warpAffine(A, np.float32([[1, 0, edgew], [0, 1, edgew * 1.4]]), sz, borderMode=cv2.BORDER_REPLICATE); li = np.clip(A - sh, 0, 1) * (A > .5); B += li[..., None] * C(edge) * .55 * min(1., t * 3)
    return out


# ------------------------------------------------------------------ impact frames (full-frame drawings)
def impact_frame(kind, cx=W / 2, cy=H / 2, seed=0, subject=None, accent=RED, lean=0., dens=1.0):
    """full-frame 1-frame drawing.  kind: 'white' paper + black wedges + black burst | 'black' black + white wedges + red | 'red' red + black |
    'crack' black field split by white/red cracked plates | 'burst' bone paper, brush burst | 'negative' black, white hairline star.  subject = premult layer toned into it"""
    cel = Cel(); bg = dict(white=BONE, black=BK, red=accent, crack=BK, burst=BONE, negative=BK)[kind]; fg = BK if kind in ('white', 'red', 'burst') else BONE
    cel.fill(np.array([[-9, -9], [W + 9, -9], [W + 9, H + 9], [-9, H + 9]]), bg); tone = dict(white='bw', black='inv', red='red', crack='blood', burst='bw', negative='inv')[kind]
    if kind in ('white', 'black', 'red', 'negative'):
        radial_wedges(cel, cx, cy, int(90 * dens), 230, 2500, seed + 1, .030, fg, clump=.7)
        if kind != 'negative': radial_wedges(cel, cx, cy, int(16 * dens), 330, 2500, seed + 2, .018, accent if kind != 'red' else BONE, clump=.4)
    if kind == 'crack':
        r = R(seed, 2)
        for i in range(int(17 * dens)):
            a = (i + r.random()) / 17 * 6.283; q0 = r.uniform(60, 180); q1 = r.uniform(700, 1500); P = bolt_line((cx + math.cos(a) * q0, cy + math.sin(a) * q0), (cx + math.cos(a) * q1, cy + math.sin(a) * q1), seed + i, .34, 5)
            for K, K2 in bolt_poly(P, r.uniform(8, 34), seed + i, .4, edge=9): cel.fill(K2, accent); cel.fill(K, BONE if r.random() < .75 else accent)
        for i in range(26):
            a = r.uniform(0, 6.283); q = r.uniform(180, 900); s = r.uniform(12, 70); th = np.sort(r.uniform(0, 6.283, 4)); P = np.c_[np.cos(th) * s * 1.6, np.sin(th) * s]; c, z = math.cos(a), math.sin(a)
            cel.fill(np.c_[P[:, 0] * c - P[:, 1] * z, P[:, 0] * z + P[:, 1] * c] + [cx + c * q, cy + z * q], BONE if r.random() < .3 else accent if r.random() < .5 else '#2a0408')
    lay = None
    if subject is not None:
        img = cel.layer()[..., :3]; a = subject[..., 3:4]; col = np.where(a > .02, subject[..., :3] / np.maximum(a, .02), 0); img = img * (1 - a) + impact_tone(col, tone) * a; cel = Cel()
    else: img = None
    S = kite_star(cx, cy, 70, 520, 24, seed + 5, (lean, .8), .13, 10)
    for K, K2 in S: cel.fill(K2, accent if kind != 'red' else BONE)
    for K, _ in S: cel.fill(K, BK if kind in ('white', 'red', 'burst') else BONE)
    for K, _ in kite_star(cx, cy, 30, 250, 15, seed + 6, (lean, .5), .07, 0): cel.fill(K, BONE if kind in ('white', 'red', 'burst') else BK)
    cel.fill(star_poly(cx, cy, 34, 84, 10, seed + 7), accent)
    if kind == 'burst':
        r = R(seed, 8)
        for i in range(70):
            a = r.uniform(0, 6.283); q = 130 + abs(r.normal(0, 300)); s = 1.5 + 22 * r.random() ** 4; th = np.linspace(0, 6.283, 12, endpoint=False); rad = s * (1 + .2 * r.normal(0, 1, 12)); st = 1 + 2.2 * r.random(); P = np.c_[np.cos(th) * rad * st, np.sin(th) * rad]
            c, z = math.cos(a), math.sin(a); cel.fill(spl(np.c_[P[:, 0] * c - P[:, 1] * z, P[:, 0] * z + P[:, 1] * c] + [cx + c * q, cy + z * q], True, n=4), BK)
    L = cel.layer(); return L[..., :3].copy() if img is None else img * (1 - L[..., 3:4]) + L[..., :3]
