"""fx break: rocks, debris, yutapon cubes, ground cracks, drawn smoke and dust, shockwaves, glass shatter of the whole picture."""
from .fx_core import *
from .fx_flash import ragged_ring, bolt_line, bolt_poly, _stroke
from .kit import outline
from scipy.spatial import ConvexHull

ROCK = dict(lit='#b9a99a', mid='#7d6f6c', dark='#3c3440', ink='#17131a')
NIGHTROCK = dict(lit='#8f95b8', mid='#565a7c', dark='#252539', ink='#0e0c14')


def _rotm(a, b, c):
    ca, sa, cb, sb, cc, sc = math.cos(a), math.sin(a), math.cos(b), math.sin(b), math.cos(c), math.sin(c)
    return np.array([[cb * cc, sa * sb * cc - ca * sc, ca * sb * cc + sa * sc], [cb * sc, sa * sb * sc + ca * cc, ca * sb * sc - sa * cc], [-sb, sa * cb, ca * cb]])


_rocks = {}


def rock_mesh(seed, kind='rock'):
    """convex faceted chunk: vertices, triangle list merged into planar faces"""
    key = (seed, kind)
    if key in _rocks: return _rocks[key]
    r = R(seed, 17)
    if kind == 'cube':
        s = np.array([1, r.uniform(.7, 1.1), r.uniform(.6, 1.1)]); V = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * s * .6
    elif kind == 'slab': V = np.c_[r.uniform(-1, 1, 12), r.uniform(-.75, .75, 12), np.r_[np.full(6, .16), np.full(6, -.16)] + r.normal(0, .03, 12)]
    else:
        V = r.normal(0, 1, (17, 3)); V /= np.linalg.norm(V, axis=1)[:, None]; V *= r.uniform(.62, 1.10, (17, 1)) * np.array([1.25, .85, .95]) * r.uniform(.75, 1.15, 3)
    Hh = ConvexHull(V); V = V[Hh.vertices]; Hh = ConvexHull(V); faces = {}
    for tri, eq in zip(Hh.simplices, Hh.equations):
        k = tuple(np.round(eq[:3], 1 if kind == 'rock' else 2)); faces.setdefault(k, [eq[:3], set()]); faces[k][1].update(tri.tolist())
    _rocks[key] = (V, [(n, np.array(sorted(ix))) for n, ix in faces.values()]); return _rocks[key]


def _hull2(P):
    h = ConvexHull(P); return P[h.vertices]


def rock(cel, x, y, size, rot, seed=0, pal=ROCK, light=(-.5, -.75, .45), kind='rock', lw=2.2, crack=True, persp=.25, rim=None):
    """one tumbling faceted chunk: real 3-D facets, three flat tones, silhouette ink, ridge ticks.  rot = (rx, ry, rz)"""
    V, F = rock_mesh(seed % 24, kind); M = _rotm(*rot); W3 = V @ M.T; z = W3[:, 2]; s = size / (1 - persp * .5 * z); P = np.c_[W3[:, 0] * s + x, W3[:, 1] * s + y]; l = np.asarray(light, float); l /= np.linalg.norm(l)
    vis = []
    for n, ix in F:
        nn = M @ n
        if nn[2] <= .02 or len(ix) < 3: continue
        try: q = _hull2(P[ix])
        except Exception: continue
        vis.append((W3[ix][:, 2].mean(), nn, q))
    vis.sort(key=lambda v: v[0]); sil = _hull2(P); cel.fill(sil, pal['mid'])
    for _, nn, q in vis:
        d = float(nn @ l); cel.fill(q, pal['lit'] if d > .42 else pal['mid'] if d > -.05 else pal['dark'])
    if size > 7:
        for j, (_, nn, q) in enumerate(vis):
            for a, b in zip(q, np.roll(q, -1, 0)):
                if h1(seed, j, int(a[0])) < .12 and size > 14: cel.line(np.linspace(a, b, 6), lw * .5, pal['ink'], taper=(.1, .6), seed=seed + j)
            if crack and size > 30 and h1(seed, j, 3) < .35:
                c = q.mean(0); e = q[int(h1(seed, j) * len(q))]; cel.line(np.vstack([e, e + (c - e) * .4 + (h1(seed, j, 4) - .5) * size * .12, e + (c - e) * .8]), lw * .45, pal['ink'], taper=(.1, .7), seed=j)
        if rim is not None:
            o = sil + np.array([light[0], light[1]]) * -1.6; cel.clip(sil); _stroke(cel, [sil], lw * 2.4, rim); cel.fill(sil + np.array([-light[0], -light[1]]) * lw * 2.0, pal['mid'], 0.); cel.unclip()
        outline(cel, sil, lw, pal['ink'], seed, press=.4, tremor=.15)
    return cel


def debris(cel, f, origin, n=30, seed=0, speed=(260, 1100), ang=(-math.pi, 0.), size=(5, 60), grav=1500., step=2, pal=ROCK, kind='rock', spin=7., spread=(30, 10), light=(-.5, -.75, .45), floor=None, drag=.5, trails=True, t0=0., lw=2.2, streak='#cfc4b8', cubes=0.):
    """thrown chunks (big few, small many), ballistic, tumbling, redrawn on twos; optional bounce on floor y and speed trails"""
    t = (int(f) // step * step) / 24. - t0
    if t < 0: return cel
    it = []
    for i in range(n):
        z = bms(h1(seed, i, 1), .12, .30); a = ang[0] + (ang[1] - ang[0]) * h1(seed, i, 2); v = (speed[0] + (speed[1] - speed[0]) * h1(seed, i, 3)) * (1.25 - .45 * z); dk = (1 - math.exp(-drag * t)) / drag if drag > 0 else t
        x = origin[0] + (h1(seed, i, 4) - .5) * 2 * spread[0] + math.cos(a) * v * dk; vy = math.sin(a) * v; y = origin[1] + (h1(seed, i, 5) - .5) * 2 * spread[1] + vy * dk + .5 * grav * t * t; s = size[0] + (size[1] - size[0]) * z * z; vis = 1.
        if floor is not None and y > floor - s * .4:
            y = floor - s * .4 - abs(math.sin(t * 9 + i)) * s * .5 * math.exp(-3 * t)
        if not (-200 < x < W + 200 and -300 < y < H + 200): continue
        w = spin * (.3 + 1.2 * h1(seed, i, 6)) * (1.3 - z); it.append((z, x, y, s, (w * t * (h1(seed, i, 7) - .5) * 2 + i, w * t + i * 2.1, w * t * (h1(seed, i, 8) - .5) + i * .7), i, (math.cos(a) * v * math.exp(-drag * t), vy * math.exp(-drag * t) + grav * t)))
    it.sort(key=lambda q: q[0])
    for z, x, y, s, rot, i, vel in it:
        sp = math.hypot(*vel)
        if trails and sp > 350 and s < size[1] * .5:
            l = min(sp * .05, 110); d = np.array(vel) / sp; m = np.array([-d[1], d[0]]); q = np.array([x, y]); cel.fill(np.array([q + m * s * .5, q - d * l, q - m * s * .5]), streak, .45)
        rock(cel, x, y, s, rot, seed * 7 + i, pal, light, 'cube' if h1(seed, i, 9) < cubes else kind, lw if s > 12 else lw * .6)
    return cel


# ------------------------------------------------------------------ ground cracks
def ground_proj(cam=None, centre=(0., 0.), hy=430., cx=W / 2, cy=800., persp=1.):
    """-> function (x, y metres on the ground, y forward) -> px.  Either a cam.Camera3D or a quick horizon model: hy horizon y, (cx, cy) = screen point of (0, 0)"""
    if cam is not None:
        def g(P):
            P = np.asarray(P, float); Q = np.c_[P[:, 0] + centre[0], P[:, 1] + centre[1], np.zeros(len(P)) if P.shape[1] < 3 else P[:, 2]]; return cam.project(Q)[0]
        return g
    def g(P):
        P = np.asarray(P, float); d = 1 / np.maximum(1 + P[:, 1] * .16 * persp, .12); z = P[:, 2] if P.shape[1] > 2 else 0.
        return np.c_[cx + P[:, 0] * 150 * d + centre[0], hy + (cy - hy) * d - z * 150 * d + centre[1]]
    return g


def ground_crack(cel, g, rad, t, seed=0, n=11, col='#0c090e', lip='#d9cdbf', side='#3a3038', glow=None, w=.10, lift=.22, plates=True, top='#7c7278', topdark='#5a5058', lw=2.2, rings=3):
    """impact crater on a ground plane: angular radial cracks that race outward with t (0..1), branches, broken concentric arcs, tilted slabs with thickness near the centre.
    g = ground_proj(...) (metres -> px), rad metres"""
    e = 1 - (1 - min(max(t, 0.), 1.)) ** 2.4; spokes = []
    for i in range(n):
        a = (i + .7 * (h1(seed, i, 1) - .5)) / n * 6.283; l = rad * (.45 + .75 * bms(h1(seed, i, 2), .35, .4)); P = bolt_line((0, 0), (math.cos(a) * l, math.sin(a) * l), seed * 3 + i, .30, 6, 0.); spokes.append((a, l, P))
    if plates and e > .15:                                  # tilted slabs between spokes
        k = min(1., (e - .15) / .5); sl = []
        for i in range(n):
            a0, l0, P0 = spokes[i]; a1, l1, P1 = spokes[(i + 1) % n]
            if i == n - 1: a1 += 6.283
            for rr in range(2):
                if h1(seed, i, rr, 5) < .22: continue
                q0, q1 = rad * (.10 + .20 * rr) * (1 + .3 * h1(seed, i, rr, 6)), rad * (.28 + .24 * rr) * (1 + .3 * h1(seed, i, rr, 7)); da = (a1 - a0); am = [a0 + da * .07, a1 - da * .07]
                C4 = np.array([[math.cos(am[0]) * q0, math.sin(am[0]) * q0], [math.cos(am[0] + da * .04) * q1 * 1.05, math.sin(am[0] + da * .04) * q1 * 1.05], [math.cos((am[0] + am[1]) / 2) * q1 * 1.14, math.sin((am[0] + am[1]) / 2) * q1 * 1.14], [math.cos(am[1]) * q1 * .96, math.sin(am[1]) * q1 * .96], [math.cos(am[1]) * q0 * 1.1, math.sin(am[1]) * q0 * 1.1]])
                hh = lift * rad * .5 * k * (.3 + .9 * h1(seed, i, rr, 8)) * (1.15 - rr * .45); z = np.array([-.25, 1., 1.15, 1., -.25]) * hh; sl.append((C4[:, 1].mean(), C4, z, i * 3 + rr))
        sl.sort(key=lambda q: -q[0]); th = rad * .055
        for _, C4, z, sd in sl:
            T = g(np.c_[C4, z]); B = g(np.c_[C4, z - th]); gd = g(np.c_[C4, z * 0])
            cel.fill(gd, col, .85)
            for j in range(len(T)):
                a, b = j, (j + 1) % len(T); Q = np.array([T[a], T[b], B[b], B[a]]); nrm = T[b] - T[a]
                if nrm[0] > 0: continue
                cel.fill(Q, side)
            for j in range(len(T)):
                a, b = j, (j + 1) % len(T); Q = np.array([T[a], T[b], B[b], B[a]])
                if T[b][0] - T[a][0] < 0 and (T[a][1] + T[b][1]) / 2 > T[:, 1].mean(): cel.fill(Q, side); cel.line(np.linspace(B[a], B[b], 6), lw * .6, '#17131a', seed=j)
            cel.fill(T, top if h1(sd, seed) > .4 else topdark); outline(cel, T, lw * .8, '#17131a', sd, press=.4); cel.line(np.linspace(T[1], T[3], 12) + [0, 1.5], lw * .9, lip, taper=(.3, .3), seed=sd)
            c = T.mean(0); cel.line(np.vstack([T[0] * .6 + T[4] * .4, c + (h1(sd) - .5) * 14, T[2] * .8 + c * .2]), lw * .5, '#17131a', taper=(.1, .6), seed=sd)
    polys = []
    for i, (a, l, P) in enumerate(spokes):
        pth = Path(P, rad * .02); grow = min(1., e * (1.1 + .5 * h1(seed, i, 3)));
        if grow < .05: continue
        q = pth.sub(0, pth.L * grow, 0, 40); ww = w * rad * .34 * (.5 + .8 * h1(seed, i, 4))
        K = Path(q, rad * .02); s = np.linspace(0, K.L, 30); v = s / K.L; hw = ww * (1 - v) ** .8 * (1 + .45 * noise(v * 13, i, 0, seed)) + .002
        polys.append(g(np.vstack([K.at(s, hw), K.at(s, -hw * .8)[::-1]]))); br = int(2 + 2 * h1(seed, i, 9))
        for j in range(br):
            u = .25 + .6 * h1(seed, i, j, 10)
            if u > grow: continue
            b0 = pth.at(u * pth.L); T = np.array([np.interp(u * pth.L, pth.s, pth.T[:, 0]), np.interp(u * pth.L, pth.s, pth.T[:, 1])]); th = math.atan2(T[1], T[0]) + (1 if h1(seed, i, j, 11) > .5 else -1) * (.4 + .6 * h1(seed, i, j, 12)); bl = l * (.15 + .25 * h1(seed, i, j, 13)) * min(1., (grow - u) * 3)
            if bl < rad * .03: continue
            q2 = Path(bolt_line(b0, b0 + np.array([math.cos(th), math.sin(th)]) * bl, seed + i * 7 + j, .35, 4, 0), rad * .02); s = np.linspace(0, q2.L, 14); hw = ww * .42 * (1 - s / q2.L) + .002; polys.append(g(np.vstack([q2.at(s, hw), q2.at(s, -hw)[::-1]])))
    for k in range(rings):                                  # broken concentric cracks
        rr = rad * (.30 + .24 * k) * (1 + .15 * h1(seed, k, 20))
        if e < .25 + .2 * k: continue
        for j in range(n):
            if h1(seed, k, j, 21) < .48: continue
            a0 = spokes[j][0]; a = np.linspace(a0, a0 + 6.283 / n * (.6 + .5 * h1(seed, k, j, 22)), 12); wob = rr * (1 + .05 * noise(a * 6, k, 0, seed)); hw = rad * .012 * np.sin(np.linspace(0, np.pi, 12)) + .002
            polys.append(g(np.vstack([np.c_[np.cos(a) * (wob + hw), np.sin(a) * (wob + hw)], np.c_[np.cos(a) * (wob - hw), np.sin(a) * (wob - hw)][::-1]])))
    for P in polys: cel.fill(P + [0, -2.0], lip, .55)
    for P in polys: cel.fill(P, col)
    if glow is not None:
        for P in polys[:n]: cel.fill(P.mean(0) + (P - P.mean(0)) * [.96, .45], glow, .9)
    return polys


# ------------------------------------------------------------------ smoke and dust
DUST = dict(lit='#e6dccd', mid='#b7a692', dark='#7c6a62', ink='#3a2e30')
SMOKE = dict(lit='#6d6472', mid='#463f50', dark='#2a2534', ink='#15111a')
HOTSMOKE = dict(lit='#ff9a2e', mid='#a3270d', dark='#3a0a0a', ink='#1c0406')
INKSMOKE = dict(lit='#3a3036', mid='#1c171c', dark='#0c0a0d', ink='#050406')


def _puff(x, y, r, seed, rot=0., bumps=7, amp=.12, n=34):
    """one cauliflower lobe: scalloped, never a circle"""
    a = np.linspace(0, 6.283, n, endpoint=False); ph = h1(seed, 1) * 6.283
    q = np.abs(np.sin((a + rot) * bumps * .5 + ph)) ** .65; rad = r * (1 - amp * .7 + amp * q * 1.25 + .06 * np.sin(a * 2 + rot + ph * 2) + .03 * np.sin(a * 3 - rot))
    return np.c_[x + np.cos(a) * rad, y + np.sin(a) * rad]


def puffs(cel, P, pal=DUST, light=(-.55, -.83), lw=2.0, seed=0, ink=True, curls=True, lit=.86, group=True, tone=1.0, dark=1.0):
    """draw a cloud of lobes [(x, y, r, seed, rot, erode 0..1)], back to front, the way it is drawn by hand: silhouette with ink, ONE scalloped lit mass, ONE scalloped shadow mass,
    then a few front lobes pulled out with an overlap line and their own small shadow, curl hooks; eroding lobes open into thinning hooks.  tone > 1 = more light"""
    L = np.asarray(light, float); L /= np.hypot(*L); P = [q for q in P if q[2] >= 2.5]
    if not P: return cel
    A0 = np.array([[q[0], q[1], q[2]] for q in P]); wt = A0[:, 2] ** 2; c0 = (A0[:, :2] * wt[:, None]).sum(0) / wt.sum(); ext = np.percentile(np.abs((A0[:, :2] - c0) @ L) + A0[:, 2] * .5, 92) + 1e-6; live = []; hooks = []
    for x, y, r, sd, rot, er in P:
        ex = float(np.clip(.50 * tone + .52 * ((np.array([x, y]) - c0) @ L) / ext + .08 * (h1(sd, 8) - .5), 0, 1)) if group else .6 * tone
        (hooks if er > .42 else live).append((x, y, r, sd, rot, er, ex, _puff(x, y, r, sd, rot, 6 + int(h1(sd, 2) * 4))))
    if live:
        U = [q[7] for q in live]
        if ink:
            for x, y, r, sd, rot, er, ex, A in live: cel.fill(np.array([x, y]) + (A - [x, y]) * (1 + lw * (.55 + .5 * h1(sd, 3)) / max(r, 8)) + L * -lw * .35, pal['ink'])
        for A in U: cel.fill(A, pal['mid'])
        cel.clip(U)
        for x, y, r, sd, rot, er, ex, A in live:             # shadow mass
            if ex < .58 * dark: cel.fill(_puff(x - L[0] * r * (.05 + .9 * ex), y - L[1] * r * (.05 + .9 * ex), r * (1.06 - .55 * ex), sd + 1, rot + .4, 6, .15), pal['dark'])
        for x, y, r, sd, rot, er, ex, A in live:             # lit mass
            if ex > .44: k = lit * (1 - .6 * er); cel.fill(_puff(x + L[0] * r * (.72 - .60 * ex), y + L[1] * r * (.72 - .60 * ex), r * (.34 + .68 * ex) * k, sd + 3, rot + .9, 7, .15), pal['lit'])
        n = len(live)
        for j, (x, y, r, sd, rot, er, ex, A) in enumerate(live):   # front lobes pulled out of the masses
            if j < n * .45 or r < 13 or h1(sd, 4) > .62: continue
            cel.clip(A)
            sh = pal['dark'] if ex < .52 else pal['mid']; lt = pal['lit'] if ex > .50 else pal['mid']
            if ex > .24:
                cel.fill(A, sh); cel.fill(_puff(x + L[0] * r * .34, y + L[1] * r * .34, r * (.78 + .16 * ex), sd + 5, rot + .3, 7, .14), lt)
            cel.unclip()
            if ink and ex > .30:
                K = np.roll(A, -int((math.atan2(L[1], L[0]) / 6.283) * len(A)), 0); m = len(A); n0 = int(m * (.14 + .18 * ex)); cel.line(np.vstack([K[-n0:], K[:n0]]), lw * .85, pal['ink'], taper=(.35, .35), press=.5, seed=sd, tremor=.2)
        cel.unclip()
        for x, y, r, sd, rot, er, ex, A in live:
            if curls and r > 24 and ex > .46 and h1(sd, 5) < .35:
                a0 = rot * 1.2 + h1(sd, 6) * 6.283; v = np.linspace(0, 1, 18); rr = r * (.60 - .42 * v); a = a0 + v * 3.4; cel.line(np.c_[x + L[0] * r * .15 + np.cos(a) * rr, y + L[1] * r * .15 + np.sin(a) * rr], lw * .8, pal['ink'], taper=(.3, .5), seed=sd, press=.4)
    for x, y, r, sd, rot, er, ex, A in hooks:                # break-up: torn wisps, thin commas that stretch along the roll and go; most lobes just shrink away
        k = (er - .42) / .58; z = h1(sd, 11)
        if z > .62 or r < 8:
            rr = r * (1 - k) ** 1.5 * .9
            if rr > 3.5: B = _puff(x, y, rr, sd, rot, 6, .16); cel.fill(np.array([x, y]) + (B - [x, y]) * (1 + lw * .6 / rr), pal['ink']) if ink and k < .5 else None; cel.fill(B, pal['mid'] if ex > .4 else pal['dark'])
            continue
        th = rot * .6 + h1(sd, 12) * 6.283; sg = 1 if h1(sd, 13) > .5 else -1; l = r * (1.5 + .9 * k); hw = r * .46 * (1 - k) ** 1.3 + .4; n = 22; v = lin01(n); hd = th + sg * (1.1 + .5 * k) * (v - .3); st = l / (n - 1); px = x - math.cos(th) * l * .5 + np.r_[0, np.cumsum(np.cos(hd[:-1]) * st)]; py = y - math.sin(th) * l * .4 + np.r_[0, np.cumsum(np.sin(hd[:-1]) * st)] * .8; W_ = Path(c2(px, py), 3.); s_ = v * W_.L; pr = np.sin(np.pi * v ** .55) ** .9 * (1 + .25 * noise(v * 6, sd * .7, 0, 1)); up = W_.at(s_, hw * pr); dn = W_.at(s_, -hw * .55 * pr)
        if ink and k < .45: cel.fill(np.vstack([W_.at(s_, (hw + lw * .7) * pr), W_.at(s_, -(hw * .55 + lw * .6) * pr)[::-1]]), pal['ink'])
        cel.fill(np.vstack([up, dn[::-1]]), pal['mid'] if ex > .32 else pal['dark'])
        if ex > .34: m = up if up[n // 2] @ L > dn[n // 2] @ L else dn; cel.fill(np.vstack([m, (m * .45 + (up + dn) * .275)[::-1]]), pal['lit'])
    return cel


def billow(t, x, y, size, seed=0, n=9, rise=.9, spread=1., life=2.0, roll=1.2, drift=0., sub=5):
    """lobes of one billow at age t seconds: parent swells (fast then slow), children ride its rim and roll over outward, then shrink and separate. -> list for puffs()"""
    if t < 0 or t > life: return []
    g = 1 - math.exp(-4.2 * t / life * 1.2); q = t / life; out = []; cx = x + drift * t; cy = y - rise * size * (1 - math.exp(-1.6 * q)) * 1.2
    shr = 1 - .55 * max(0., (q - .45) / .55) ** 1.3
    for i in range(n):
        a = (i * 2.399 + h1(seed, i, 1)) ; z = bms(h1(seed, i, 2), .26, .34); ro = size * (.12 + .58 * g * (.45 + .7 * h1(seed, i, 3)) * (1.15 - .45 * z)) * spread; sg = 1 if math.cos(a) > 0 else -1; a2 = a + sg * roll * q * (1 - .3 * z)
        px = cx + math.cos(a2) * ro * 1.15; py = cy + math.sin(a2) * ro * .72 - size * .10 * q * z; rr = size * (.12 + .50 * z) * (.35 + .65 * g) * shr
        out.append((px, py, rr, seed * 91 + i, sg * roll * 2.2 * q + i, max(0., (q - .35) / .65)))
        for j in range(sub if z > .7 else (sub // 2 if z > .4 and h1(seed, i, 12) < .6 else 0)):
            b = a2 + (j - sub / 2.) * .55 + sg * roll * q * 1.5; s2 = rr * (.12 + .42 * h1(seed, i, j, 5) ** 1.6); out.append((px + math.cos(b) * rr * .88, py + math.sin(b) * rr * .88, s2 * shr, seed * 91 + i * 13 + j + 7, sg * roll * 3 * q + j, max(0., (q - .3) / .7)))
    out.sort(key=lambda p: p[1] - p[2] * .4 + 40 * h1(int(p[3]), 9)); return out


def smoke_burst(cel, f, cx, cy, size, seed=0, n=8, ang=(-math.pi, 0.), speed=420., life=1.8, pal=DUST, light=(-.55, -.83), step=2, lw=2.0, ground=False, squash=.55, lobes=8, drag=2.6, rise=.5, t0=0.):
    """drawn cumulus explosion: billows thrown out along a fan, near ones last; on twos"""
    t = (int(f) // step * step) / 24. - t0
    if t < 0: return cel
    it = []
    for i in range(n):
        a = ang[0] + (ang[1] - ang[0]) * (i + .5 + .6 * (h1(seed, i, 1) - .5)) / n; z = bms(h1(seed, i, 2), .35, .4); d = speed * (.35 + .8 * h1(seed, i, 3)) * (1 - math.exp(-drag * t)) / drag; x = cx + math.cos(a) * d; y = cy + math.sin(a) * d * (squash if ground else 1.)
        it.append((math.sin(a), x, y, size * (.35 + .75 * z), seed * 7 + i, life * (.7 + .5 * h1(seed, i, 4))))
    it.sort(key=lambda q: q[0])
    for s, x, y, sz, sd, lf in it: puffs(cel, billow(t, x, y, sz, sd, lobes, rise, 1., lf), pal, light, lw, sd)
    return cel


def dust_skirt(cel, f, cx, cy, rx, ry, size, seed=0, n=16, pal=DUST, life=1.1, step=2, lw=1.8, light=(-.55, -.83), grow=1.0, front=True, back=True, t0=0.):
    """ring of ground dust running outward round an ellipse (landing, shock): far side first"""
    t = (int(f) // step * step) / 24. - t0
    if t < 0 or t > life: return cel
    e = 1 - math.exp(-4.5 * t); it = []
    for i in range(n):
        a = (i + .8 * h1(seed, i, 1)) / n * 6.283; s = math.sin(a); k = (.25 + .85 * e) * grow * (.85 + .3 * h1(seed, i, 2)); z = bms(h1(seed, i, 3), .3, .45); it.append((s, cx + math.cos(a) * rx * k, cy + s * ry * k, size * (1 + .45 * s) * (.4 + .7 * z), seed * 11 + i))
    it.sort(key=lambda q: q[0])
    for s, x, y, sz, sd in it:
        if (s < 0 and back) or (s >= 0 and front): puffs(cel, billow(t, x, y, sz, sd, 5, .55, .9, life * (.8 + .4 * h1(sd, 1)), sub=3), pal, light, lw, sd)
    return cel


def smoke_trail(cel, P, f, w0=10., w1=60., seed=0, pal=SMOKE, step=2, lw=1.8, light=(-.55, -.83), cell=.7, erode=.0):
    """smoke along a path (missile / thrown body): lobes grow toward the tail"""
    t = (int(f) // step * step) / 24.; p = Path(P, 5.); s = p.L; i = 0; L = []
    while s > 0:
        v = 1 - s / p.L; r = w0 + (w1 - w0) * v ** .8; x, y = p.at(s, (h1(seed, i) - .5) * r * .9); L.append((x, y, r * (.75 + .5 * h1(seed, i, 1)), seed * 3 + i, t * 2 * (1 if h1(seed, i, 2) > .5 else -1) + i, min(1., erode + v * .6))); s -= r * cell; i += 1
    puffs(cel, L[::-1], pal, light, lw, seed); return cel


def shockwave(cel, cx, cy, t, r0=60., r1=1200., asp=.35, w=26., seed=0, col='#ffffff', shade='#8a86a8', rot=0., n=8):
    """drawn shock ring in perspective, t 0..1: white thick-thin crescents with a shadow lip, breaking up"""
    e = 1 - (1 - t) ** 2.6; r = r0 + (r1 - r0) * e; k = (1 - t) ** 1.3
    if k < .03: return cel
    ragged_ring(cel, cx, cy, r * .985, r * asp * .97, w * k * 1.25, seed, shade, gaps=.2 + .5 * t, rot=rot, n=n); ragged_ring(cel, cx, cy, r, r * asp, w * k, seed, col, gaps=.2 + .5 * t, rot=rot, n=n)
    ragged_ring(cel, cx, cy, r * 1.09, r * asp * 1.09, w * k * .18 + .5, seed + 5, col, gaps=.5 + .3 * t, rot=rot, n=n + 3, spurs=.2); return cel


# ------------------------------------------------------------------ the picture breaks like glass
def glass_cells(cx, cy, seed=0, n=15, rings=(90, 210, 400, 680, 1050, 1700), w=W, h=H):
    """radial + concentric fracture cells (convex-ish quads / triangles) round an impact point -> list of polygons, clipped to the frame"""
    r = R(seed, 31); th = np.sort((np.arange(n) + r.uniform(-.33, .33, n)) / n * 6.283); m = len(rings); V = np.zeros((m + 1, n, 2)); V[0] = [cx, cy]
    for k, q in enumerate(rings):
        for i in range(n):
            a = th[i] + r.normal(0, .035 + .05 * k / m); rr = q * r.uniform(.72, 1.26); V[k + 1, i] = (cx + math.cos(a) * rr, cy + math.sin(a) * rr * .92)
    P = []
    for i in range(n):
        j = (i + 1) % n; P.append(np.array([V[0, 0], V[1, i], V[1, j]]))
        for k in range(1, m):
            q = [V[k, i], V[k + 1, i], V[k + 1, j], V[k, j]]; u = r.random()
            if u < .42: P += [np.array([q[0], q[1], q[2]]), np.array([q[0], q[2], q[3]])]
            elif u < .70: c = (q[1] + q[2]) / 2 + r.normal(0, 12, 2); P += [np.array([q[0], q[1], c]), np.array([q[0], c, q[3]]), np.array([c, q[2], q[3]])]
            else: P.append(np.array(q))
    out = []; fr = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    for q in P:
        if q[:, 0].max() < 0 or q[:, 0].min() > w or q[:, 1].max() < 0 or q[:, 1].min() > h: continue
        ar, c = cv2.intersectConvexConvex(cv2.convexHull(q.astype(np.float32)), fr)
        if ar > 60 and c is not None and len(c) >= 3: out.append(c[:, 0].astype(float))
    return out


def shatter(img, t, cx, cy, under=None, seed=0, n=15, hold=.22, cells=None, grav=2300., burst=520., spin=5.5, depth=1300., edge='#f4fbff', ink='#1a1622', lines=True, crackcol='#ffffff', refr=9., persp=1.0, delay=.55, dark=.45, crackw=1.0):
    """The whole picture breaks like glass at (cx, cy).  t seconds: 0..hold = cracks race out and every pane slips a little (refraction), then panes
    come loose from the centre outward, rotate in real 3-D, fall and leave `under` (RGB frame or colour, default white).  -> RGB frame"""
    h, w = img.shape[:2]; cells = glass_cells(cx, cy, seed, n, w=w, h=h) if cells is None else cells; out = np.empty_like(img)
    if under is None: out[:] = 1.
    elif isinstance(under, str): out[:] = C(under)
    else: out[:] = under
    rmax = math.hypot(max(cx, w - cx), max(cy, h - cy)); foc = 1500.; items = []
    for i, P in enumerate(cells):
        c = P.mean(0); d = math.hypot(*(c - [cx, cy])); u = d / rmax; t0 = hold + delay * u ** 1.2 * (.75 + .5 * h1(seed, i, 1)); crack = t * 5.5 > u * 1.0; tau = t - t0; items.append((tau, u, i, P, c, crack))
    items.sort(key=lambda q: -q[1] if q[0] <= 0 else 10 - q[0]); cracks = Cel(w, h, ss=1) if lines else None
    still = [q for q in items if q[0] <= 0]
    if still:                                               # the picture stays behind unshifted panes so slips never show the background
        ms = np.zeros((h, w), np.uint8); cv2.fillPoly(ms, [np.round(q[3] * 16).astype(np.int32) for q in still], 255, cv2.LINE_8, 4); out[ms > 0] = img[ms > 0] * .92
    mov = [q for q in items if q[0] > 0]; mov.sort(key=lambda q: -q[0])
    for tau, u, i, P, c, crack in still + mov[::-1][::-1]:
        x0, y0 = np.floor(P.min(0)).astype(int) - 1; x1, y1 = np.ceil(P.max(0)).astype(int) + 1; x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 - x0 < 2 or y1 - y0 < 2: continue
        src = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float); ax = h1(seed, i, 2) * 6.283; sh = 0.
        if tau <= 0:
            k = 0. if not crack else refr * min(1., t / max(hold, 1e-3)) * (.4 + h1(seed, i, 3)); a = (h1(seed, i, 4) - .5) * .03 * (k > 0); rx, ry, rz = 0., 0., a; off = np.array([math.cos(ax) * k, math.sin(ax) * k, 0.]); sh = (h1(seed, i, 5) - .5) * .16 * (k > 0)
        else:
            sp = spin * (.35 + .9 * h1(seed, i, 6)); e = tau; rx, ry = math.cos(ax) * sp * e, math.sin(ax) * sp * e; rz = (h1(seed, i, 7) - .5) * 3 * e; v = (c - [cx, cy]) / (np.hypot(*(c - [cx, cy])) + 1) * burst * (1.2 - .7 * u) * (.5 + .7 * h1(seed, i, 8))
            off = np.array([v[0] * e, v[1] * e - 60 * e + .5 * grav * e * e, -depth * (h1(seed, i, 9) - .25) * e]); sh = 0.
        M = _rotm(rx, ry, rz); Q = np.c_[src - c, np.zeros(4)] @ M.T + off; nz = M[2, 2]; z = foc / np.maximum(foc + Q[:, 2], 120.); D = np.c_[c[0] + Q[:, 0] * z + (c[0] - w / 2) * (z - 1) * persp, c[1] + Q[:, 1] * z + (c[1] - h / 2) * (z - 1) * persp]
        PP = np.c_[P - c, np.zeros(len(P))] @ M.T + off; zz = foc / np.maximum(foc + PP[:, 2], 120.); DP = np.c_[c[0] + PP[:, 0] * zz + (c[0] - w / 2) * (zz - 1) * persp, c[1] + PP[:, 1] * zz + (c[1] - h / 2) * (zz - 1) * persp]
        bx0, by0 = np.floor(DP.min(0)).astype(int) - 2; bx1, by1 = np.ceil(DP.max(0)).astype(int) + 2; bx0, by0, bx1, by1 = max(0, bx0), max(0, by0), min(w, bx1), min(h, by1)
        if bx1 - bx0 < 2 or by1 - by0 < 2: continue
        Hm = cv2.getPerspectiveTransform((src - [x0, y0]).astype(np.float32), (D - [bx0, by0]).astype(np.float32))
        ms = np.zeros((by1 - by0, bx1 - bx0), np.uint8); cv2.fillPoly(ms, [np.round((DP - [bx0, by0]) * 16).astype(np.int32)], 255, cv2.LINE_AA, 4); m = (ms.astype(f32) / 255.)[..., None]
        patch = cv2.warpPerspective(img[y0:y1, x0:x1], Hm, (bx1 - bx0, by1 - by0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        gl = 0.
        if tau > 0: f_ = abs(nz); patch = patch * (1 - dark * (1 - f_)) if nz > 0 else patch * .30 + C('#20243a') * .25; gl = .60 * max(0., math.sin(rx * 1.7 + ry * 2.3 + i)) ** 22
        patch = patch * (1 + sh) + gl
        B = out[by0:by1, bx0:bx1]; B[:] = B * (1 - m) + patch * m
        if crack and lines:
            E = DP; q = np.vstack([E, E[:1]]); lit = (E - E.mean(0)) @ np.array([-.6, -.8]); wd = crackw if tau <= 0 else 2.2
            for a, b in zip(q[:-1], q[1:]):
                nrm = np.array([b[1] - a[1], a[0] - b[0]]); s = nrm @ np.array([-.5, -.85]) / (np.hypot(*nrm) + 1e-6)
                cv2.line(out, tuple(np.round(a * 16).astype(int)), tuple(np.round(b * 16).astype(int)), tuple(float(v) for v in (C(edge) * (1.0 if s > 0 else .75) if tau <= 0 or s > 0 else C(ink))), int(max(1, round(wd * (1.4 if s > 0 else 1.)))), cv2.LINE_AA, 4) if (tau > 0 or h1(seed, i, int(a[0])) < .9) else None
    if lines and t < hold * 3:                             # white fracture star at the hit
        cel = Cel(w, h); r = R(seed, 77); k = min(1., t * 7)
        for i in range(n + 9):
            a = r.uniform(0, 6.283); l = r.uniform(90, 620) * k; P = bolt_line((cx, cy), (cx + math.cos(a) * l, cy + math.sin(a) * l), seed + i, .10, 4, 0)
            for K, _ in bolt_poly(P, r.uniform(1.2, 4.0) * max(0., 1 - t / (hold * 3)), seed + i, .3): cel.fill(K, crackcol)
        lay = cel.layer(h, w); out = out * (1 - lay[..., 3:4]) + lay[..., :3]
    return out
