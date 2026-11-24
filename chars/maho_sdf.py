"""Construction sculpt for Mahoraga: a small numba SDF tracer that gives the DRAWING its perspective (depth, normals, part ids,
cast shadow).  Nothing here is ever shown: chars/mahoraga.py paints planes, brush and ink over it.
World = Camera3D convention: +X right, +Y away from camera, +Z up, metres."""
import numpy as np, math
from numba import njit, prange

RC, ELL, BOX, TOR, FEA, BLD, PLEAT = range(7)


class Sculpt:
    """list of primitives.  grp: primitives of one group melt together (smooth min k); groups meet as hard overlaps."""
    def __init__(self):
        self.T, self.A, self.M, self.P, self.K, self.G, self.MAT, self.PART, self.BC, self.BR, self.SUB, self.DIR = ([] for _ in range(12))

    def _add(self, t, a, m, p, k, g, mat, part, bc, br, sub, d):
        p = list(p) + [0.] * (12 - len(p))
        self.T.append(t); self.A.append(a); self.M.append(m); self.P.append(p); self.K.append(k); self.G.append(g); self.MAT.append(mat)
        self.PART.append(part); self.BC.append(bc); self.BR.append(br + abs(k) * .35 + .004); self.SUB.append(1 if sub else 0); self.DIR.append(d)
        return len(self.T) - 1

    def cone(self, a, b, r1, r2, k=.04, g=0, mat=0, part=0, sub=False, pleat=None):
        a, b = np.asarray(a, float), np.asarray(b, float); ba = b - a; l2 = ba @ ba + 1e-12; rr = r1 - r2
        a2 = max(l2 - rr * rr, 1e-9); n = ba / math.sqrt(l2)
        p = [*ba, l2, rr, a2, 1 / l2, r1, r2]
        c = (a + b) / 2; br = math.sqrt(l2) / 2 + max(r1, r2)
        if pleat is None: return self._add(RC, a, np.eye(3), p, k, g, mat, part, c, br, sub, n)
        amp, cnt, ph, tw = pleat
        # frame: z along the axis
        up = np.array([0, 1., 0]) if abs(n[1]) < .9 else np.array([1., 0, 0])
        x = np.cross(up, n); x /= np.linalg.norm(x); y = np.cross(n, x)
        return self._add(PLEAT, a, np.stack([x, y, n]), p[:9] + [amp, cnt, ph], k, g, mat, part, c, br + amp, sub, n)

    def ell(self, c, r, M=None, k=.04, g=0, mat=0, part=0, sub=False, d=None):
        """M rows = local axes in world"""
        M = np.eye(3) if M is None else np.asarray(M, float)
        return self._add(ELL, np.asarray(c, float), M, list(r), k, g, mat, part, np.asarray(c, float), max(r), sub, M[2] if d is None else d)

    def box(self, c, h, M=None, r=.01, k=0., g=0, mat=0, part=0, sub=False):
        M = np.eye(3) if M is None else np.asarray(M, float)
        return self._add(BOX, np.asarray(c, float), M, [*h, r], k, g, mat, part, np.asarray(c, float), float(np.linalg.norm(h)) + r, sub, M[2])

    def tor(self, c, R, r, M=None, k=0., g=0, mat=0, part=0, zs=1.):
        M = np.eye(3) if M is None else np.asarray(M, float)
        return self._add(TOR, np.asarray(c, float), M, [R, r, zs], k, g, mat, part, np.asarray(c, float), R + r * 1.5, False, M[2])

    def feather(self, c, L, w, t, M, bend=0., k=0., g=0, mat=0, part=0):
        """flat leaf: x = length axis (half length L), y = width, z = thickness"""
        return self._add(FEA, np.asarray(c, float), np.asarray(M, float), [L, w, t, bend], k, g, mat, part, np.asarray(c, float), L * 1.12 + abs(bend) * L * L, False, M[0])

    def blade(self, a, M, L, w, t, g=0, mat=0, part=0, curve=0., tip=.30):
        """M rows: x along the blade, y toward the edge, z flat normal"""
        c = np.asarray(a, float) + np.asarray(M[0]) * L / 2
        return self._add(BLD, np.asarray(a, float), np.asarray(M, float), [L, w, t, curve, tip], 0., g, mat, part, c, L / 2 + w * 1.6 + abs(curve) * L * L, False, np.asarray(M[0]))

    def muscle(self, a, b, w, t, nrm, k=.045, g=0, mat=0, part=0, ext=.12):
        """spindle between a and b, width w (half), thickness t (half) along nrm"""
        a, b = np.asarray(a, float), np.asarray(b, float); z = b - a; L = np.linalg.norm(z) + 1e-9; z = z / L
        n = np.asarray(nrm, float); n = n - z * (n @ z); n /= np.linalg.norm(n) + 1e-9
        x = np.cross(n, z)
        return self.ell((a + b) / 2, (w, t, L / 2 * (1 + ext)), np.stack([x, n, z]), k, g, mat, part)

    def pack(self):
        o = np.argsort(np.asarray(self.G), kind='stable')
        g = lambda L, dt=float: np.asarray(L, dt)[o]
        S = dict(T=g(self.T, np.int64), A=g(self.A), M=g(self.M), P=g(self.P), K=g(self.K), G=g(self.G, np.int64), MAT=g(self.MAT, np.int64),
                 PART=g(self.PART, np.int64), BC=g(self.BC), BR=g(self.BR), SUB=g(self.SUB, np.int64), DIR=g(self.DIR))
        return S


@njit(fastmath=True, cache=True)
def _prim(x, y, z, i, T, A, M, P):
    dx = x - A[i, 0]; dy = y - A[i, 1]; dz = z - A[i, 2]
    t = T[i]
    if t == 0 or t == 6:
        bx = P[i, 0]; by = P[i, 1]; bz = P[i, 2]; l2 = P[i, 3]; rr = P[i, 4]; a2 = P[i, 5]; il2 = P[i, 6]
        yy = dx * bx + dy * by + dz * bz
        zz = yy - l2
        qx = dx * l2 - bx * yy; qy = dy * l2 - by * yy; qz = dz * l2 - bz * yy
        x2 = qx * qx + qy * qy + qz * qz
        y2 = yy * yy * l2; z2 = zz * zz * l2
        k = rr * rr * x2
        if rr < 0: k = -k
        if (z2 if zz > 0 else -z2) * a2 > k: d = math.sqrt(x2 + z2) * il2 - P[i, 8]
        elif (y2 if yy > 0 else -y2) * a2 < k: d = math.sqrt(x2 + y2) * il2 - P[i, 7]
        else: d = (math.sqrt(x2 * a2 * il2) + yy * rr) * il2 - P[i, 7]
        if t == 6:
            u = dx * M[i, 0, 0] + dy * M[i, 0, 1] + dz * M[i, 0, 2]
            v = dx * M[i, 1, 0] + dy * M[i, 1, 1] + dz * M[i, 1, 2]
            th = math.atan2(v, u); tt = yy * il2
            if tt < 0: tt = 0.
            if tt > 1: tt = 1.
            s = math.sin(th * P[i, 10] + P[i, 11] + 1.7 * math.sin(tt * 2.6 + P[i, 11] * 1.7) + .6 * math.sin(th * 2 + P[i, 11]))
            s = 1 - 2 * abs(s) ** 1.0 if s > 0 else 1 - 2 * abs(s) ** .6
            d = (d + P[i, 9] * s * (.25 + .75 * math.sin(3.14159 * min(1., .12 + .88 * tt)))) * .72
        return d
    u = dx * M[i, 0, 0] + dy * M[i, 0, 1] + dz * M[i, 0, 2]
    v = dx * M[i, 1, 0] + dy * M[i, 1, 1] + dz * M[i, 1, 2]
    w = dx * M[i, 2, 0] + dy * M[i, 2, 1] + dz * M[i, 2, 2]
    if t == 1:
        a = u / P[i, 0]; b = v / P[i, 1]; c = w / P[i, 2]
        k0 = math.sqrt(a * a + b * b + c * c) + 1e-9
        a /= P[i, 0]; b /= P[i, 1]; c /= P[i, 2]
        k1 = math.sqrt(a * a + b * b + c * c) + 1e-9
        return k0 * (k0 - 1) / k1
    if t == 2:
        r = P[i, 3]
        a = abs(u) - P[i, 0] + r; b = abs(v) - P[i, 1] + r; c = abs(w) - P[i, 2] + r
        m = max(a, max(b, c))
        a = max(a, 0.); b = max(b, 0.); c = max(c, 0.)
        return math.sqrt(a * a + b * b + c * c) + min(m, 0.) - r
    if t == 3:
        q = math.sqrt(u * u + v * v) - P[i, 0]
        w *= P[i, 2]
        return (math.sqrt(q * q + w * w) - P[i, 1]) / max(1., P[i, 2])
    if t == 4:
        w = w - P[i, 3] * u * u
        s = u / P[i, 0]                                # feather: full in the middle, tapering to a blunt point
        e = s if s > 0 else 0.
        if e > 1.2: e = 1.2
        wd = P[i, 1] * (1. - .42 * e * e)
        a = abs(s); b = v / wd; c = w / P[i, 2]
        k0 = math.sqrt(a * a * a + b * b + c * c) + 1e-9
        a = a * a / P[i, 0]; b /= wd; c /= P[i, 2]
        k1 = math.sqrt(a * a + b * b + c * c) + 1e-9
        return k0 * (k0 - 1) / k1 * .62
    # blade
    L = P[i, 0]; wd = P[i, 1]; th = P[i, 2]
    v = v - P[i, 3] * u * u
    tt = (L - u) / (L * P[i, 4])
    ww = wd
    if tt < 1:
        ww = wd * max(tt, 0.) ** .74
        th = th * (.10 + .90 * max(tt, 0.))
    e = abs(v) / (wd + 1e-9)
    d = max(abs(v) - ww, (abs(w) + min(e, 1.4) * th * .92 - th) * .8)
    d = max(d, max(u - L, -u))
    return d * .8


@njit(fastmath=True, cache=True)
def _map(x, y, z, lst, n0, n1, T, A, M, P, K, G, SUB):
    best = 1e9; bid = -1
    cur = 1e9; curid = -1; curmin = 1e9; g = -999
    for j in range(n0, n1):
        i = lst[j]
        if G[i] != g:
            if cur < best: best = cur; bid = curid
            cur = 1e9; curid = -1; curmin = 1e9; g = G[i]
        d = _prim(x, y, z, i, T, A, M, P)
        k = K[i]
        if SUB[i]:
            dd = -d
            if k > 0:
                h = max(k - abs(cur - dd), 0.) / k
                if dd > cur: curid = i
                cur = max(cur, dd) + h * h * k * .25
            else:
                if dd > cur: cur = dd; curid = i
        else:
            if d < curmin: curmin = d; curid = i
            if k > 0 and cur < 1e8:
                h = max(k - abs(cur - d), 0.) / k
                cur = min(cur, d) - h * h * k * .25
            elif d < cur: cur = d
    if cur < best: best = cur; bid = curid
    return best, bid


@njit(cache=True)
def _cull(C, R, f, cx, cy, w, h, ts, near):
    """C cam-space centres.  -> tile start offsets, flat prim list, per-tile near/far"""
    n = C.shape[0]; tw = (w + ts - 1) // ts; th = (h + ts - 1) // ts
    box = np.zeros((n, 4), np.int64); cnt = np.zeros(tw * th + 1, np.int64)
    for i in range(n):
        z = C[i, 2]; r = R[i]
        if z + r < near: box[i, 0] = 1; box[i, 1] = 0; continue
        if z - r < near * 2:
            x0 = 0; x1 = tw - 1; y0 = 0; y1 = th - 1
        else:
            rr = f * r / math.sqrt(z * z - r * r) * 1.04 + 2
            sx = C[i, 0] / z * f + cx; sy = C[i, 1] / z * f + cy
            ex = rr * (1 + abs(C[i, 0] / z) * .6); ey = rr * (1 + abs(C[i, 1] / z) * .6)
            x0 = max(0, int(math.floor((sx - ex) / ts))); x1 = min(tw - 1, int(math.floor((sx + ex) / ts)))
            y0 = max(0, int(math.floor((sy - ey) / ts))); y1 = min(th - 1, int(math.floor((sy + ey) / ts)))
        box[i, 0] = x0; box[i, 1] = x1; box[i, 2] = y0; box[i, 3] = y1
        for ty in range(y0, y1 + 1):
            for tx in range(x0, x1 + 1): cnt[ty * tw + tx + 1] += 1
    for q in range(1, tw * th + 1): cnt[q] += cnt[q - 1]
    lst = np.zeros(max(1, cnt[tw * th]), np.int64); pos = cnt[:-1].copy()
    nr = np.full(tw * th, 1e9); fr = np.zeros(tw * th)
    for i in range(n):
        x0, x1, y0, y1 = box[i, 0], box[i, 1], box[i, 2], box[i, 3]
        if x1 < x0: continue
        for ty in range(y0, y1 + 1):
            for tx in range(x0, x1 + 1):
                q = ty * tw + tx; lst[pos[q]] = i; pos[q] += 1
                nr[q] = min(nr[q], max(near, C[i, 2] - R[i])); fr[q] = max(fr[q], C[i, 2] + R[i])
    return cnt, lst, nr, fr, tw, th


@njit(parallel=True, fastmath=True, cache=True)
def _trace(w, h, ts, tw, th, cnt, lst, nr, fr, E, Rm, f, cx, cy, T, A, M, P, K, G, SUB, BC, BR, full, hm):
    Z = np.zeros((h, w), np.float32); N = np.zeros((h, w, 3), np.float32); ID = np.full((h, w), -1, np.int32); AO = np.ones((h, w), np.float32)
    for q in prange(tw * th):
        n0 = cnt[q]; n1 = cnt[q + 1]
        if n1 <= n0: continue
        tx = q % tw; ty = q // tw
        loc = np.empty(n1 - n0, np.int64); lo2 = np.empty(n1 - n0, np.int64)
        for y in range(ty * ts, min(h, (ty + 1) * ts)):
            for x in range(tx * ts, min(w, (tx + 1) * ts)):
                a = (x + .5 - cx) / f; b = (y + .5 - cy) / f
                il = math.sqrt(a * a + b * b + 1)
                rx = (a * Rm[0, 0] + b * Rm[1, 0] + Rm[2, 0]) / il; ry = (a * Rm[0, 1] + b * Rm[1, 1] + Rm[2, 1]) / il; rz = (a * Rm[0, 2] + b * Rm[1, 2] + Rm[2, 2]) / il
                m = 0; m2 = 0; t = 1e9; tf = 0.
                for j in range(n0, n1):
                    i = lst[j]
                    ex = BC[i, 0] - E[0]; ey = BC[i, 1] - E[1]; ez = BC[i, 2] - E[2]
                    tc = ex * rx + ey * ry + ez * rz
                    d2 = ex * ex + ey * ey + ez * ez - tc * tc
                    r = BR[i]; s2 = r * r - d2
                    if s2 > 0 and tc + r > .03:
                        s2 = math.sqrt(s2); loc[m] = i; m += 1
                        if tc - s2 < t: t = tc - s2
                        if tc + s2 > tf: tf = tc + s2
                    if full:
                        r2 = r + .045
                        if d2 < r2 * r2: lo2[m2] = i; m2 += 1
                if m == 0: continue
                if t < .03: t = .03
                hit = -1; eps = .55 / f
                for s in range(96):
                    d, i = _map(E[0] + rx * t, E[1] + ry * t, E[2] + rz * t, loc, 0, m, T, A, M, P, K, G, SUB)
                    if d < eps * t: hit = i; break
                    t += d * .88
                    if t > tf: break
                if hit < 0: continue
                Z[y, x] = t; ID[y, x] = hit
                if full:
                    px = E[0] + rx * t; py = E[1] + ry * t; pz = E[2] + rz * t
                    e = max(hm, .0007 * t)
                    d1, _ = _map(px + e, py - e, pz - e, lo2, 0, m2, T, A, M, P, K, G, SUB)
                    d2, _ = _map(px - e, py - e, pz + e, lo2, 0, m2, T, A, M, P, K, G, SUB)
                    d3, _ = _map(px - e, py + e, pz - e, lo2, 0, m2, T, A, M, P, K, G, SUB)
                    d4, _ = _map(px + e, py + e, pz + e, lo2, 0, m2, T, A, M, P, K, G, SUB)
                    nx = d1 - d2 - d3 + d4; ny = -d1 - d2 + d3 + d4; nz = -d1 + d2 - d3 + d4
                    l = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
                    nx /= l; ny /= l; nz /= l
                    N[y, x, 0] = nx; N[y, x, 1] = ny; N[y, x, 2] = nz
                    oc = 0.; sc = 1.
                    for s in range(1, 4):
                        hh = .006 * s * s + .004
                        d, _ = _map(px + nx * hh, py + ny * hh, pz + nz * hh, lo2, 0, m2, T, A, M, P, K, G, SUB)
                        oc += max(hh - d, 0.) / hh * sc * .42; sc *= .8
                    AO[y, x] = max(0., 1 - oc)
    return Z, N, ID, AO


def _call(S, E, Rm, f, cx, cy, w, h, ts=20, near=.05, full=True, hm=.0012):
    C = (S['BC'] - E) @ Rm.T
    cnt, lst, nr, fr, tw, th = _cull(C, S['BR'] + (.045 if full else 0.), float(f), float(cx), float(cy), w, h, ts, near)
    return _trace(w, h, ts, tw, th, cnt, lst, nr, fr, np.asarray(E, float), np.asarray(Rm, float), float(f), float(cx), float(cy),
                  S['T'], S['A'], S['M'], S['P'], S['K'], S['G'], S['SUB'], S['BC'], S['BR'], full, hm)


def gbuffer(S, cam, scale=1.0, full=True):
    """dict(z=ray distance, n=world normal, id, ao, p=world position, w, h, scale)"""
    w, h = int(round(cam.w * scale)), int(round(cam.h * scale))
    f, cx, cy = cam.f * scale, cam.cx * scale, cam.cy * scale
    Z, N, ID, AO = _call(S, cam.eye, cam.R, f, cx, cy, w, h, full=full)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.stack([(x + .5 - cx) / f, (y + .5 - cy) / f, np.ones_like(x)], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    V = (d @ cam.R.astype(np.float32)).astype(np.float32)
    return dict(z=Z, n=N, id=ID, ao=AO, v=V, p=cam.eye.astype(np.float32) + V * Z[..., None], w=w, h=h, scale=scale)


_FACES = [((1, 0, 0), (0, 0, 1)), ((-1, 0, 0), (0, 0, 1)), ((0, 1, 0), (0, 0, 1)), ((0, -1, 0), (0, 0, 1)), ((0, 0, 1), (0, 1, 0)), ((0, 0, -1), (0, 1, 0))]


def cube_shadow(S, L, res=288):
    """distance cube map from point light L.  skip: prims listed in S['NOSH'] mask are removed"""
    L = np.asarray(L, float); out = np.full((6, res, res), 1e9, np.float32); Rs = np.zeros((6, 3, 3))
    for q, (fw, up) in enumerate(_FACES):
        fw = np.array(fw, float); up = np.array(up, float); rt = np.cross(fw, up)
        Rm = np.stack([rt, -up, fw]); Rs[q] = Rm
        Z, _, ID, _ = _call(S, L, Rm, res / 2, res / 2, res / 2, res, res, ts=16, near=.02, full=False)
        out[q] = np.where(ID >= 0, Z, 1e9)
    return out, Rs


@njit(parallel=True, fastmath=True, cache=True)
def _pcf(PX, N, ID, L, cube, Rs, res, bias, soft, seed):
    h, w = ID.shape; out = np.ones((h, w), np.float32)
    for y in prange(h):
        for x in range(w):
            if ID[y, x] < 0: continue
            dx = PX[y, x, 0] + N[y, x, 0] * bias - L[0]; dy = PX[y, x, 1] + N[y, x, 1] * bias - L[1]; dz = PX[y, x, 2] + N[y, x, 2] * bias - L[2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            s = 0.
            for k in range(8):
                an = k * 2.399963 + seed; rr = soft * math.sqrt((k + .5) / 8) * res * .5
                ax = abs(dx); ay = abs(dy); az = abs(dz)
                if ax >= ay and ax >= az: q = 0 if dx > 0 else 1; ma = ax
                elif ay >= az: q = 2 if dy > 0 else 3; ma = ay
                else: q = 4 if dz > 0 else 5; ma = az
                u = (dx * Rs[q, 0, 0] + dy * Rs[q, 0, 1] + dz * Rs[q, 0, 2]) / ma; v = (dx * Rs[q, 1, 0] + dy * Rs[q, 1, 1] + dz * Rs[q, 1, 2]) / ma
                iu = int((u + 1) * res * .5 + math.cos(an) * rr); iv = int((v + 1) * res * .5 + math.sin(an) * rr)
                if iu < 0: iu = 0
                if iv < 0: iv = 0
                if iu > res - 1: iu = res - 1
                if iv > res - 1: iv = res - 1
                if cube[q, iv, iu] > dist - bias * 2: s += 1
            out[y, x] = s / 8
    return out


def shadow(g, S, L, res=288, bias=.016, soft=.03):
    cube, Rs = cube_shadow(S, L, res)
    return _pcf(g['p'], g['n'], g['id'], np.asarray(L, float), cube, Rs, res, bias, soft, .3)


def ortho_shadow(g, S, dirv, center, rad, res=640, bias=.02):
    """directional light shadow: dirv = direction the light TRAVELS.  (long-lens perspective stands in for ortho)"""
    dv = np.asarray(dirv, float); dv /= np.linalg.norm(dv); D = rad * 24
    E = np.asarray(center, float) - dv * D
    up = np.array([0, 0, 1.]) if abs(dv[2]) < .95 else np.array([0, 1., 0])
    rt = np.cross(dv, up); rt /= np.linalg.norm(rt); u = np.cross(rt, dv); Rm = np.stack([rt, -u, dv]); f = res / 2 * D / rad
    Z, _, ID, _ = _call(S, E, Rm, f, res / 2, res / 2, res, res, ts=16, near=.02, full=False)
    Z = np.where(ID >= 0, Z, 1e9).astype(np.float32)
    q = (g['p'] + g['n'] * bias - E) @ Rm.T; dist = np.linalg.norm(q, axis=-1)
    mx = (q[..., 0] / q[..., 2] * f + res / 2).astype(np.float32); my = (q[..., 1] / q[..., 2] * f + res / 2).astype(np.float32)
    import cv2
    s = 0
    for ox, oy in ((0, 0), (1.2, .5), (-.6, 1.1), (-1., -.8), (.7, -1.2)):
        s = s + (cv2.remap(Z, mx + ox, my + oy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=1e9) > dist - bias * 2)
    return (s / 5.).astype(np.float32)


@njit(fastmath=True, cache=True)
def _march(px, py, pz, dx, dy, dz, tmax, n, T, A, M, P, K, G, SUB):
    lst = np.arange(n); t = 0.
    for s in range(160):
        d, i = _map(px + dx * t, py + dy * t, pz + dz * t, lst, 0, n, T, A, M, P, K, G, SUB)
        if d < .0006: return t
        t += d * .8
        if t > tmax: break
    return -1.


def march(S, p, d, tmax=2.):
    """distance along a ray to the sculpt, -1 if missed"""
    d = np.asarray(d, float); d = d / np.linalg.norm(d)
    return _march(*map(float, p), *map(float, d), float(tmax), len(S['T']), S['T'], S['A'], S['M'], S['P'], S['K'], S['G'], S['SUB'])
