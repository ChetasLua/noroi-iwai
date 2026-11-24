"""3-D core of jj/hands.py: distance field of posed hands + ray caster -> G-buffer (numba, one thread, no imports from the rest of jj)."""
import numpy as np, math
from numba import njit

K = 33          # samples per profile table


@njit(fastmath=True, cache=True)
def _tab(t, TAB, i, k):
    x = t * (K - 1)
    if x <= 0: return TAB[i, k, 0]
    if x >= K - 1: return TAB[i, k, K - 1]
    j = int(x); f = x - j
    return TAB[i, k, j] * (1 - f) + TAB[i, k, j + 1] * f


@njit(fastmath=True, cache=True)
def _prim(i, px, py, pz, T, R, O, L, TAB, CAP, BX):
    dx = px - O[i, 0]; dy = py - O[i, 1]; dz = pz - O[i, 2]
    x = R[i, 0, 0] * dx + R[i, 0, 1] * dy + R[i, 0, 2] * dz
    y = R[i, 1, 0] * dx + R[i, 1, 1] * dy + R[i, 1, 2] * dz
    z = R[i, 2, 0] * dx + R[i, 2, 1] * dy + R[i, 2, 2] * dz
    if T[i] == 0:                                           # tapered super-ellipsoid capsule with profile tables
        t = y / L[i]
        if t < 0: w = -y / CAP[i, 0]; t = 0.
        elif t > 1: w = (y - L[i]) / CAP[i, 1]; t = 1.
        else: w = 0.
        a = _tab(t, TAB, i, 0)
        b = _tab(t, TAB, i, 1) if z > 0 else _tab(t, TAB, i, 2)
        u = x / a; v = z / b
        u2 = u * u; v2 = v * v; w2 = w * w
        r = (1 - BX[i]) * math.sqrt(u2 + v2 + w2) + BX[i] * math.sqrt(math.sqrt(u2 * u2 + v2 * v2 + w2 * w2))
        m = a if a < b else b
        if w > 0:
            c = CAP[i, 0] if y < 0 else CAP[i, 1]
            if c < m: m = c
        return (r - 1) * m
    if T[i] == 1:                                           # rounded triangle plate: local verts (0,0) (L,0) (TAB0,TAB1) in x,y ; thickness TAB2
        x1 = L[i]; x2 = TAB[i, 0, 0]; y2 = TAB[i, 1, 0]; th = TAB[i, 2, 0]
        # signed distance to the 2-D triangle (0,0),(x1,0),(x2,y2)
        best = 1e9
        s = 1. if y2 * x1 >= 0 else -1.
        inside = True
        ax = (0., x1, x2); ay = (0., 0., y2)
        for e in range(3):
            f = (e + 1) % 3
            ex = ax[f] - ax[e]; ey = ay[f] - ay[e]; qx = x - ax[e]; qy = y - ay[e]
            h = (qx * ex + qy * ey) / (ex * ex + ey * ey + 1e-12)
            h = 0. if h < 0 else 1. if h > 1 else h
            dd = (qx - ex * h) ** 2 + (qy - ey * h) ** 2
            if dd < best: best = dd
            if s * (ex * qy - ey * qx) < 0: inside = False
        d2 = 0. if inside else best
        return math.sqrt(d2 + z * z) - th
    if T[i] == 2:                                           # hollow cuff: tapered tube with a wall
        t = y / L[i]
        a = _tab(t, TAB, i, 0); b = _tab(t, TAB, i, 1) if z > 0 else _tab(t, TAB, i, 2)
        u = x / a; v = z / b
        r = (1 - BX[i]) * math.sqrt(u * u + v * v) + BX[i] * math.sqrt(math.sqrt(u ** 4 + v ** 4))
        m = a if a < b else b
        d = (r - 1) * m
        wall = abs(d + CAP[i, 1]) - CAP[i, 1]
        e0 = -y - CAP[i, 0]; e1 = y - L[i]
        ee = e0 if e0 > e1 else e1
        return wall if wall > ee else ee
    return 1e9


@njit(fastmath=True, cache=True)
def _smin(a, b, k):
    if k <= 1e-6: return a if a < b else b
    h = k - abs(a - b)
    if h <= 0: return a if a < b else b
    return (a if a < b else b) - h * h / (4 * k)


@njit(fastmath=True, cache=True)
def field(px, py, pz, T, R, O, L, TAB, CAP, BX, BC, BR, HB, KB, CH, CP, XT, out):
    """HB[h]=(b0,b1) body prims of hand h (KB = blend radius of each); CH[c]=(c0,c1,h) finger chains, CP[c]=(mcp xyz,k0,r0,r1,kf);
    XT = indices of hard extras.  out[0]=prim index."""
    best = 1e9; bi = -1
    for h in range(HB.shape[0]):
        d = 1e9; di = -1; dmin = 1e9
        for i in range(HB[h, 0], HB[h, 1]):
            lb = math.sqrt((px - BC[i, 0]) ** 2 + (py - BC[i, 1]) ** 2 + (pz - BC[i, 2]) ** 2) - BR[i]
            if lb > d + KB[i] or lb > best + .3:
                if lb < d: d = lb
                continue
            q = _prim(i, px, py, pz, T, R, O, L, TAB, CAP, BX)
            if q < dmin: dmin = q; di = i
            d = _smin(d, q, KB[i])
        hb = d; hi = di
        for c in range(CH.shape[0]):
            if CH[c, 2] != h: continue
            k0, r0, r1, kf = CP[c, 3], CP[c, 4], CP[c, 5], CP[c, 6]
            f = 1e9; fi = -1; fm = 1e9
            for i in range(CH[c, 0], CH[c, 1]):
                lb = math.sqrt((px - BC[i, 0]) ** 2 + (py - BC[i, 1]) ** 2 + (pz - BC[i, 2]) ** 2) - BR[i]
                if lb > f + kf or lb > hb + k0:
                    if lb < f: f = lb
                    continue
                q = _prim(i, px, py, pz, T, R, O, L, TAB, CAP, BX)
                if q < fm: fm = q; fi = i
                f = _smin(f, q, kf)
            if f < d + k0 or f < hb:
                r = math.sqrt((px - CP[c, 0]) ** 2 + (py - CP[c, 1]) ** 2 + (pz - CP[c, 2]) ** 2)
                k = k0 * (1 - (r - r0) / (r1 - r0))
                if k > k0: k = k0
                if k < 0: k = 0.
                g = _smin(d, f, k)
                if g < hb: hb = g
                if fm < dmin: dmin = fm; hi = fi
        if hb < best: best = hb; bi = hi
    for j in range(XT.shape[0]):
        i = XT[j]
        lb = math.sqrt((px - BC[i, 0]) ** 2 + (py - BC[i, 1]) ** 2 + (pz - BC[i, 2]) ** 2) - BR[i]
        if lb > best: continue
        q = _prim(i, px, py, pz, T, R, O, L, TAB, CAP, BX)
        if q < best: best = q; bi = i
    out[0] = bi
    return best


@njit(fastmath=True, cache=True)
def fieldA(act, px, py, pz, T, R, O, L, TAB, CAP, BX, HB, KB, CH, CP, XT, out):
    """field restricted to the active primitives of a screen tile"""
    best = 1e9; bi = -1
    for h in range(HB.shape[0]):
        d = 1e9; di = -1; dmin = 1e9
        for i in range(HB[h, 0], HB[h, 1]):
            if act[i] == 0: continue
            q = _prim(i, px, py, pz, T, R, O, L, TAB, CAP, BX)
            if q < dmin: dmin = q; di = i
            d = _smin(d, q, KB[i])
        hb = d; hi = di
        for c in range(CH.shape[0]):
            if CH[c, 2] != h: continue
            f = 1e9; fi = -1; fm = 1e9
            for i in range(CH[c, 0], CH[c, 1]):
                if act[i] == 0: continue
                q = _prim(i, px, py, pz, T, R, O, L, TAB, CAP, BX)
                if q < fm: fm = q; fi = i
                f = _smin(f, q, CP[c, 6])
            if fi < 0: continue
            k0, r0, r1 = CP[c, 3], CP[c, 4], CP[c, 5]
            if f < d + k0 or f < hb:
                r = math.sqrt((px - CP[c, 0]) ** 2 + (py - CP[c, 1]) ** 2 + (pz - CP[c, 2]) ** 2)
                k = k0 * (1 - (r - r0) / (r1 - r0))
                if k > k0: k = k0
                if k < 0: k = 0.
                g = _smin(d, f, k)
                if g < hb: hb = g
                if fm < dmin: dmin = fm; hi = fi
        if hb < best: best = hb; bi = hi
    for j in range(XT.shape[0]):
        i = XT[j]
        if act[i] == 0: continue
        q = _prim(i, px, py, pz, T, R, O, L, TAB, CAP, BX)
        if q < best: best = q; bi = i
    out[0] = bi
    return best


@njit(fastmath=True, cache=True)
def _active(act, ux, uy, uz, D, tile, SA, SB, SR):
    """mark capsules (SA-SB, radius SR) that meet the cone around the ray from (0,0,D) along u; tile = cone slope"""
    any_ = 0
    for i in range(SA.shape[0]):
        # closest approach ray / segment
        ex = SB[i, 0] - SA[i, 0]; ey = SB[i, 1] - SA[i, 1]; ez = SB[i, 2] - SA[i, 2]
        wx = -SA[i, 0]; wy = -SA[i, 1]; wz = D - SA[i, 2]
        b = ux * ex + uy * ey + uz * ez; c = ex * ex + ey * ey + ez * ez; d = ux * wx + uy * wy + uz * wz; e = ex * wx + ey * wy + ez * wz
        den = c - b * b
        s = (e - b * d) / den if den > 1e-9 else 0.
        s = 0. if s < 0 else 1. if s > 1 else s
        qx = SA[i, 0] + ex * s; qy = SA[i, 1] + ey * s; qz = SA[i, 2] + ez * s - D
        t = qx * ux + qy * uy + qz * uz
        if t < 0: t = 0.
        dx = qx - ux * t; dy = qy - uy * t; dz = qz - uz * t
        rr = SR[i] + tile * t
        if dx * dx + dy * dy + dz * dz < rr * rr: act[i] = 1; any_ = 1
        else: act[i] = 0
    return any_


@njit(fastmath=True, cache=True)
def cast(x0, y0, px, gw, gh, D, z0, z1, T, R, O, L, TAB, CAP, BX, BC, BR, HB, KB, CH, CP, XT, SA, SB, SR, sh, q):
    """Pinhole at (0,0,D) looking down -z through the plane z=0.  Pixel (i,j) = plane point (x0+i*px, y0-j*px).
    z1 > z0 = depth slab of the scene.  sh = unit vector to the light for cast shadows (or zeros).
    returns prim index (-1 bg), depth z, normals, lit (1 lit, 0 cast shadow)"""
    P = np.full((gh, gw), -1, np.int32); Z = np.full((gh, gw), -1e3, np.float32); N = np.zeros((gh, gw, 3), np.float32)
    LIT = np.ones((gh, gw), np.float32)
    out = np.zeros(1, np.int32); act = np.zeros(T.shape[0], np.uint8)
    doS = (sh[0] * sh[0] + sh[1] * sh[1] + sh[2] * sh[2]) > 1e-9
    nw = (gw + q - 1) // q; nh_ = (gh + q - 1) // q
    for J in range(nh_):
        for I in range(nw):
            cx = x0 + (I * q + (q - 1) * .5) * px; cy = y0 - (J * q + (q - 1) * .5) * px
            ln = math.sqrt(cx * cx + cy * cy + D * D)
            dx = cx / ln; dy = cy / ln; dz = -D / ln
            rad = px * q * .72 / ln
            if _active(act, dx, dy, dz, D, rad * 1.1, SA, SB, SR) == 0: continue
            t = (D - z1) / -dz; tm = (D - z0) / -dz
            if t < .02: t = .02
            hit = False
            for _ in range(90):
                d = fieldA(act, dx * t, dy * t, D + dz * t, T, R, O, L, TAB, CAP, BX, HB, KB, CH, CP, XT, out)
                if d < rad * t * 1.05: hit = True; break
                t += d - rad * t * .95 if d - rad * t * .95 > 1e-3 else 1e-3
                if t > tm: break
            if not hit: continue
            tc = t - rad * t * 1.6
            for j in range(J * q, min(gh, J * q + q)):
                for i in range(I * q, min(gw, I * q + q)):
                    X = x0 + i * px; Y = y0 - j * px
                    l2 = math.sqrt(X * X + Y * Y + D * D); ux = X / l2; uy = Y / l2; uz = -D / l2
                    t = tc * l2 / ln if l2 < ln else tc
                    ok = False
                    eps = px * .25
                    for _ in range(70):
                        d = fieldA(act, ux * t, uy * t, D + uz * t, T, R, O, L, TAB, CAP, BX, HB, KB, CH, CP, XT, out)
                        if d < eps: ok = True; break
                        t += d * .92
                        if t > tm: break
                    if not ok: continue
                    hx = ux * t; hy = uy * t; hz = D + uz * t
                    P[j, i] = out[0]; Z[j, i] = hz
                    e = .006
                    a = fieldA(act, hx + e, hy - e, hz - e, T, R, O, L, TAB, CAP, BX, HB, KB, CH, CP, XT, out)
                    b = fieldA(act, hx - e, hy - e, hz + e, T, R, O, L, TAB, CAP, BX, HB, KB, CH, CP, XT, out)
                    c = fieldA(act, hx - e, hy + e, hz - e, T, R, O, L, TAB, CAP, BX, HB, KB, CH, CP, XT, out)
                    dd = fieldA(act, hx + e, hy + e, hz + e, T, R, O, L, TAB, CAP, BX, HB, KB, CH, CP, XT, out)
                    nx = a - b - c + dd; ny = -a - b + c + dd; nz = -a + b - c + dd
                    m = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
                    N[j, i, 0] = nx / m; N[j, i, 1] = ny / m; N[j, i, 2] = nz / m
    if doS:                                                   # cast shadows on a half-resolution lattice
        for j in range(0, gh, 2):
            for i in range(0, gw, 2):
                if P[j, i] < 0: continue
                nx = N[j, i, 0]; ny = N[j, i, 1]; nz = N[j, i, 2]
                lit = 1.
                if nx * sh[0] + ny * sh[1] + nz * sh[2] > 0.02:
                    X = x0 + i * px; Y = y0 - j * px; hz = Z[j, i]; f = (D - hz) / D
                    hx = X * f + nx * .012; hy = Y * f + ny * .012; hz += nz * .012
                    s = .04
                    for _ in range(22):
                        d = field(hx + sh[0] * s, hy + sh[1] * s, hz + sh[2] * s, T, R, O, L, TAB, CAP, BX, BC, BR, HB, KB, CH, CP, XT, out)
                        if d < .004: lit = 0.; break
                        s += d if d > .015 else .015
                        if s > 2.0: break
                for jj in range(j, min(gh, j + 2)):
                    for ii in range(i, min(gw, i + 2)): LIT[jj, ii] = lit
    return P, Z, N, LIT


@njit(fastmath=True, cache=True)
def probe(P0, DIR, smax, T, R, O, L, TAB, CAP, BX, BC, BR, HB, KB, CH, CP, XT):
    """sphere-trace n rays from P0 along DIR (world); returns distance (or -1)"""
    n = P0.shape[0]; r = np.full(n, -1.); out = np.zeros(1, np.int32)
    for k in range(n):
        s = 0.
        for _ in range(80):
            d = field(P0[k, 0] + DIR[k, 0] * s, P0[k, 1] + DIR[k, 1] * s, P0[k, 2] + DIR[k, 2] * s, T, R, O, L, TAB, CAP, BX, BC, BR, HB, KB, CH, CP, XT, out)
            if d < .002: r[k] = s; break
            s += d
            if s > smax: break
    return r


@njit(fastmath=True, cache=True)
def dist(P0, T, R, O, L, TAB, CAP, BX, BC, BR, HB, KB, CH, CP, XT):
    n = P0.shape[0]; r = np.zeros(n); out = np.zeros(1, np.int32)
    for k in range(n): r[k] = field(P0[k, 0], P0[k, 1], P0[k, 2], T, R, O, L, TAB, CAP, BX, BC, BR, HB, KB, CH, CP, XT, out)
    return r
