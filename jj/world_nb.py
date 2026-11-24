"""Compiled kernels of world.py (kept apart so that editing world.py never triggers a recompile for the other lanes)."""
import numpy as np, math
from numba import njit, prange
from .paint import _vn, _h2


# ------------------------------------------------------------------------------------------------ rasteriser
@njit(parallel=True, cache=True, fastmath=True)
def _raster(PX, PY, IZ, w, y0, y1, ID, ZB, B1, B2, nb):
    n = PX.shape[0]
    HH = y1 - y0
    for b in prange(nb):
        r0 = y0 + HH * b // nb
        r1 = y0 + HH * (b + 1) // nb
        for i in range(n):
            ay = PY[i, 0]; by = PY[i, 1]; cy = PY[i, 2]
            mn = min(ay, min(by, cy)); mx = max(ay, max(by, cy))
            if mx < r0 or mn > r1: continue
            ax = PX[i, 0]; bx = PX[i, 1]; cx = PX[i, 2]
            if max(ax, max(bx, cx)) < 0 or min(ax, min(bx, cx)) > w: continue
            area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
            if abs(area) < 1e-7: continue
            inv = 1.0 / area
            # w1 = a1*x + b1*y + c1 , w2 likewise
            a1 = (cy - ay) * inv; b1 = -(cx - ax) * inv; c1 = -(ax * a1 + ay * b1)
            a2 = -(by - ay) * inv; b2 = (bx - ax) * inv; c2 = -(ax * a2 + ay * b2)
            za = IZ[i, 0]; zb = IZ[i, 1]; zc = IZ[i, 2]
            ya = max(r0, int(math.floor(mn - .5))); yb = min(r1 - 1, int(mx + .5))
            for y in range(ya, yb + 1):
                py = y + 0.5
                lo = 0.0; hi = float(w)
                k1 = b1 * py + c1; k2 = b2 * py + c2
                ok = True
                # w1 >= 0
                if a1 > 1e-12: lo = max(lo, -k1 / a1)
                elif a1 < -1e-12: hi = min(hi, -k1 / a1)
                elif k1 < 0: ok = False
                if a2 > 1e-12: lo = max(lo, -k2 / a2)
                elif a2 < -1e-12: hi = min(hi, -k2 / a2)
                elif k2 < 0: ok = False
                a0 = -(a1 + a2); k0 = 1 - k1 - k2
                if a0 > 1e-12: lo = max(lo, -k0 / a0)
                elif a0 < -1e-12: hi = min(hi, -k0 / a0)
                elif k0 < 0: ok = False
                if not ok or lo > hi: continue
                xa = int(math.ceil(lo - .5)); xb = int(math.floor(hi - .5))
                if xa < 0: xa = 0
                if xb > w - 1: xb = w - 1
                yy = y - y0
                for x in range(xa, xb + 1):
                    px = x + .5
                    w1 = a1 * px + k1; w2 = a2 * px + k2; w0 = 1 - w1 - w2
                    iz = w0 * za + w1 * zb + w2 * zc
                    if iz > ZB[yy, x]:
                        ZB[yy, x] = iz; ID[yy, x] = i
                        B1[yy, x] = w1 * zb / iz; B2[yy, x] = w2 * zc / iz


@njit(cache=True, fastmath=True)
def _cellnoise(u, v, s):
    """voronoi crevice distance (d2-d1), id"""
    ix = math.floor(u); iy = math.floor(v)
    d1 = 9.; d2 = 9.; idd = 0.
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            jx = int(ix) + dx; jy = int(iy) + dy
            px = jx + .5 + (_h2(jx, jy, s) - .5) * .85; py = jy + .5 + (_h2(jx, jy, s + 7) - .5) * .85
            d = math.sqrt((u - px) ** 2 + (v - py) ** 2)
            if d < d1:
                d2 = d1; d1 = d; idd = _h2(jx, jy, s + 13)
            elif d < d2: d2 = d
    return d2 - d1, idd


@njit(cache=True, fastmath=True)
def _tex(kind, u, v, mpp, s, X, Y, Z):
    """material modulation: returns multiplier, warm shift, snowmask-ish roughness"""
    m = 1.0; wm = 0.0
    if kind == 1:      # concrete: patches, formwork joints, run-off streaks
        m += .16 * (_vn(u * .23, v * .31, s) - .5) + .10 * (_vn(u * .9, v * 1.1, s + 1) - .5)
        st = _vn(u * 1.7, v * .10 + 3, s + 2); st = max(0.0, st - .58) * 2.2
        m -= .30 * st * _vn(u * .4, v * .4, s + 4)
        if mpp < .06: m += .08 * (_vn(u * 9, v * 9, s + 3) - .5)
        wm = .03 * (_vn(u * .3, v * .16, s + 5) - .5)
    elif kind == 2:    # asphalt: resurfaced patches, wheel polish, aggregate
        c, i = _cellnoise(u * .085, v * .06, s)
        m += .16 * (i - .5) + .12 * (_vn(u * .35, v * .35, s) - .5)
        if mpp < .10: m += .16 * (_vn(u * 3.1, v * 3.1, s + 1) - .5)
        if mpp < .03: m += .30 * (_vn(u * 17, v * 17, s + 2) - .5)
        wm = .02 * (i - .5)
    elif kind == 3:    # square tiles (0.3 m) with grout
        fu = u / .3; fv = v / .3
        k = _h2(int(math.floor(fu)), int(math.floor(fv)), s)
        m += .10 * (k - .5) + .06 * (_vn(u * .5, v * .5, s) - .5)
        if mpp < .10:
            q = min(min(fu - math.floor(fu), math.ceil(fu) - fu), min(fv - math.floor(fv), math.ceil(fv) - fv)) * .3
            g = 1 - q / max(.011, mpp * 1.2)
            if g > 0: m *= 1 - .60 * g * min(1., .04 / mpp)
    elif kind == 4:    # brick / stone courses
        row = math.floor(v / .12); fu = u / .28 + .5 * (int(row) & 1); fv = v / .12
        k = _h2(int(math.floor(fu)), int(row), s)
        m += .30 * (k - .5) + .10 * (_vn(u, v, s) - .5)
        if mpp < .05:
            q = min(min(fu - math.floor(fu), math.ceil(fu) - fu) * .28, min(fv - math.floor(fv), math.ceil(fv) - fv) * .12)
            g = 1 - q / max(.010, mpp * .9)
            if g > 0: m *= 1 - .4 * g * min(1., .025 / mpp)
    elif kind == 5:    # foliage dabs
        a = _vn(X * 1.3 + Z * .7, Y * 1.3 + Z * 1.1, s); b = _vn(X * 4.1 + Z * 2.2, Y * 4.1 - Z * 2.9, s + 1)
        t = a * .6 + b * .4
        m = .62 if t < .42 else (.95 if t < .56 else 1.32)
        wm = .05 * (t - .5)
    elif kind == 6:    # standing-seam / corrugated metal
        fu = u / .33; q = fu - math.floor(fu)
        a = min(1., .10 / mpp)
        m += .22 * a * (1. if q < .14 else (-.5 if q < .3 else 0.)) + .10 * (_vn(u * .4, v * .25, s) - .5)
        st = max(0.0, _vn(u * 2.1, v * .15, s + 2) - .62) * 2; m -= .2 * st; wm = .10 * st
    elif kind == 7:    # large panels (1.2 x 0.6) joint lines
        fu = u / 1.2; fv = v / .6
        k = _h2(int(math.floor(fu)), int(math.floor(fv)), s)
        m += .09 * (k - .5) + .08 * (_vn(u * .3, v * .4, s) - .5)
        if mpp < .12:
            q = min(min(fu - math.floor(fu), math.ceil(fu) - fu) * 1.2, min(fv - math.floor(fv), math.ceil(fv) - fv) * .6)
            g = 1 - q / max(.012, mpp * .9)
            if g > 0: m *= 1 - .35 * g * min(1., .06 / mpp)
    elif kind == 8:    # worn road paint
        n = _vn(u * 1.1, v * 1.1, s) * .5 + _vn(u * 4.7, v * 4.7, s + 1) * .5
        m += .10 * (n - .5)
        if n < .30: m *= .55
    elif kind == 9:    # broken concrete face
        a = _vn(u * .9, v * .9, s) * .55 + _vn(u * 2.9, v * 2.9, s + 1) * .45
        m = .80 if a < .42 else (1.0 if a < .58 else 1.16)
        if mpp < .03: m += .16 * (_vn(u * 13, v * 13, s + 3) - .5)
    elif kind == 10:   # grass
        a = _vn(X * .55, Y * .55, s); b = _vn(X * 2.6, Y * 1.1, s + 1)
        t = a * .6 + b * .4
        m = .78 if t < .40 else (1.0 if t < .58 else 1.20)
        wm = .05 * (a - .5)
    elif kind == 11:   # roof tiles: rows across v
        fv = v / .30; q = fv - math.floor(fv); fu = u / .28 + .5 * (int(math.floor(fv)) & 1)
        k = _h2(int(math.floor(fu)), int(math.floor(fv)), s)
        a = min(1., .22 / mpp)
        m += a * (.20 * (k - .5) - (.26 if q < .22 else 0.)) + .14 * (_vn(u * .3, v * .3, s) - .5)
    elif kind == 12:   # wood planks
        fu = u / .16; k = _h2(int(math.floor(fu)), 3, s); q = fu - math.floor(fu)
        m += .24 * (k - .5) + .16 * (_vn(u * 14, v * .8, s) - .5)
        if q < .08: m *= 1 - .35 * min(1., .03 / mpp)
    elif kind == 13:   # cracked asphalt
        c, i = _cellnoise(u * .16 + .5 * _vn(u * .3, v * .3, s + 20), v * .16 + .5 * _vn(u * .3 + 9, v * .3, s + 21), s)
        c2, i2 = _cellnoise(u * .55 + .4 * _vn(u, v, s + 22), v * .55 + .4 * _vn(u + 9, v, s + 23), s + 3)
        m += .22 * (i - .5) + .12 * (i2 - .5) + .12 * (_vn(u * .35, v * .35, s) - .5)
        e = c / .16; e2 = c2 / .55
        wd = max(.06, mpp * .8); wd2 = max(.025, mpp * .7)
        if e < wd * (.5 + _vn(u * .8, v * .8, s + 5)): m *= .34
        elif e < wd * 2.2: m *= 1.10
        if e2 < wd2 * 1.2 and _vn(u * .25, v * .25, s + 6) > .45: m *= .58
        if mpp < .08: m += .18 * (_vn(u * 3.3, v * 3.3, s + 1) - .5)
        if mpp < .03: m += .30 * (_vn(u * 15, v * 15, s + 2) - .5)
    elif kind == 14:   # plaster with damp, mould at the foot
        m += .10 * (_vn(u * .5, v * .5, s) - .5) + .07 * (_vn(u * 2.1, v * 2.3, s + 1) - .5)
        dmp = (max(0., 1 - Z / 1.1) if Z > 0 else 0.) * _vn(u * .9, 1., s + 2); m -= .30 * dmp
    return m, wm


@njit(parallel=True, cache=True, fastmath=True)
def _shade(ID, ZB, B1, B2, y0, ch, w, ss, eye, R, f, cx, cy, TC, TQ, QN, QD, QA, lab,
           amb, fills, sun, sunM, SM, smres, LP, LC, LD, fog, fogc, snow, G, groups,
           out, cov, zout, lout, aout):
    """QA columns: 0 em, 1 group, 2 tex, 3 snowable, 4 seed, 5 wet/gloss, 6 nofog, 7 uv mode"""
    nl = LP.shape[0]; nf = fills.shape[0]
    k2 = 1.0 / (ss * ss)
    for yy in prange(ch):
        for xx in range(w):
            rr = 0.; gg = 0.; bb = 0.; cc = 0.
            for sy in range(ss):
                for sx in range(ss):
                    py = yy * ss + sy; px = xx * ss + sx
                    i = ID[py, px]
                    if i < 0: continue
                    z = 1.0 / ZB[py, px]
                    q = TQ[i]
                    dx = (px + .5) / ss - cx; dy = (y0 + (py + .5) / ss) - cy
                    dx /= f; dy /= f
                    wx = R[0, 0] * dx + R[1, 0] * dy + R[2, 0]; wy = R[0, 1] * dx + R[1, 1] * dy + R[2, 1]; wz = R[0, 2] * dx + R[1, 2] * dy + R[2, 2]
                    X = eye[0] + wx * z; Y = eye[1] + wy * z; Z = eye[2] + wz * z
                    rl = math.sqrt(wx * wx + wy * wy + wz * wz)
                    nx = QN[q, 0]; ny = QN[q, 1]; nz = QN[q, 2]
                    if nx * wx + ny * wy + nz * wz > 0:
                        nx = -nx; ny = -ny; nz = -nz
                    b1 = B1[py, px]; b2 = B2[py, px]; b0 = 1 - b1 - b2
                    cr = b0 * TC[i, 0, 0] + b1 * TC[i, 1, 0] + b2 * TC[i, 2, 0]
                    cg = b0 * TC[i, 0, 1] + b1 * TC[i, 1, 1] + b2 * TC[i, 2, 1]
                    cb = b0 * TC[i, 0, 2] + b1 * TC[i, 1, 2] + b2 * TC[i, 2, 2]
                    em = QA[q, 0]; grp = int(QA[q, 1]); kind = int(QA[q, 2])
                    mpp = z / f
                    an = abs(nz)
                    if an > .7:
                        u = X; v = Y
                    else:
                        hl = math.sqrt(nx * nx + ny * ny) + 1e-9
                        u = (-ny * X + nx * Y) / hl; v = Z
                    if kind > 0:
                        m, wm = _tex(kind, u, v, mpp, int(QA[q, 4]), X, Y, Z)
                        cr = cr * m * (1 + wm * 2.2); cg = cg * m; cb = cb * m * (1 - wm * 2.2)
                    em_ = em * groups[grp] if grp < groups.shape[0] else em
                    # snow cover on top faces
                    if snow[0] > 0 and QA[q, 3] > 0:
                        s = (nz - snow[4]) / .35
                        if s > 1: s = 1.
                        n1 = _vn(X * .8 + 3, Y * .8 + Z * .3, 5) * .55 + _vn(X * 3.1, Y * 3.1 + Z, 6) * .45
                        s = s * snow[0] * QA[q, 3] + (n1 - .5) * .62
                        s = (s - .34) / .10
                        if s > 0:
                            if s > 1: s = 1.
                            cr += (snow[1] - cr) * s; cg += (snow[2] - cg) * s; cb += (snow[3] - cb) * s; em_ *= 1 - s
                    # ambient
                    up = .5 + .5 * nz
                    lr = amb[0, 0] * up + amb[1, 0] * (1 - up); lg = amb[0, 1] * up + amb[1, 1] * (1 - up); lb = amb[0, 2] * up + amb[1, 2] * (1 - up)
                    if amb[3, 0] > 0:
                        gk = math.exp(-max(0.0, Z) / amb[3, 0]) * (.55 + .45 * math.exp(-max(0.0, Z) / (amb[3, 0] * .25)))
                        if amb[3, 1] > 0: gk *= math.exp(-((X - amb[4, 0]) ** 2 + (Y - amb[4, 1]) ** 2) / (amb[3, 1] ** 2)) * (1 - amb[3, 2]) + amb[3, 2]
                        lr += amb[2, 0] * gk; lg += amb[2, 1] * gk; lb += amb[2, 2] * gk
                    for j in range(nf):
                        d = nx * fills[j, 0] + ny * fills[j, 1] + nz * fills[j, 2]
                        d = d * .65 + .35
                        if d > 0:
                            lr += fills[j, 3] * d; lg += fills[j, 4] * d; lb += fills[j, 5] * d
                    if sun[3] + sun[4] + sun[5] > 0:
                        d = nx * sun[0] + ny * sun[1] + nz * sun[2]
                        if d > 0:
                            sh = 1.0
                            if smres > 0:
                                sh = 0.0
                                bx_ = X + nx * sun[6]; by_ = Y + ny * sun[6]; bz_ = Z + nz * sun[6]
                                su = sunM[0, 0] * bx_ + sunM[0, 1] * by_ + sunM[0, 2] * bz_ + sunM[0, 3]
                                sv = sunM[1, 0] * bx_ + sunM[1, 1] * by_ + sunM[1, 2] * bz_ + sunM[1, 3]
                                sd = sunM[2, 0] * bx_ + sunM[2, 1] * by_ + sunM[2, 2] * bz_ + sunM[2, 3]
                                if su < 1 or sv < 1 or su >= smres - 2 or sv >= smres - 2: sh = 1.0
                                else:
                                    iu = int(su - .5); iv = int(sv - .5); fu = su - .5 - iu; fv = sv - .5 - iv
                                    a00 = 1. if SM[iv, iu] < sd + sun[7] else 0.; a01 = 1. if SM[iv, iu + 1] < sd + sun[7] else 0.
                                    a10 = 1. if SM[iv + 1, iu] < sd + sun[7] else 0.; a11 = 1. if SM[iv + 1, iu + 1] < sd + sun[7] else 0.
                                    sh = (a00 * (1 - fu) + a01 * fu) * (1 - fv) + (a10 * (1 - fu) + a11 * fu) * fv
                            d = d ** .6 * sh
                            lr += sun[3] * d; lg += sun[4] * d; lb += sun[5] * d
                    for j in range(nl):
                        ex = LP[j, 0] - X; ey = LP[j, 1] - Y; ez = LP[j, 2] - Z
                        d2 = ex * ex + ey * ey + ez * ez
                        rad = LP[j, 3]
                        if d2 > rad * rad: continue
                        gl = groups[int(LC[j, 3])]
                        if gl <= 0: continue
                        d = math.sqrt(d2) + 1e-6
                        ex /= d; ey /= d; ez /= d
                        nd = (nx * ex + ny * ey + nz * ez)
                        nd = (nd + .25) / 1.25
                        if nd <= 0: continue
                        wn = 1 - d / rad
                        at = wn * wn / (1 + (d / LD[j, 4]) ** 2)
                        if LD[j, 3] > -1:
                            cs = -(ex * LD[j, 0] + ey * LD[j, 1] + ez * LD[j, 2])
                            t = (cs - LD[j, 3]) / (1e-3 + (1 - LD[j, 3]) * LD[j, 5])
                            if t <= 0: continue
                            if t > 1: t = 1.
                            at *= t
                        at *= nd * gl
                        lr += LC[j, 0] * at; lg += LC[j, 1] * at; lb += LC[j, 2] * at
                    r_ = cr * lr + cr * em_; g_ = cg * lg + cg * em_; b_ = cb * lb + cb * em_
                    if fog[0] > 0 and QA[q, 6] == 0:
                        dist = z * rl
                        hh = 1.0
                        if fog[2] > 0: hh = math.exp(-max(0.0, Z - fog[4]) * fog[2] * .5)
                        k = 1 - math.exp(-max(0.0, dist - fog[1]) * fog[0] * (fog[3] + (1 - fog[3]) * hh))
                        dd = (wx * fogc[2, 0] + wy * fogc[2, 1] + wz * fogc[2, 2]) / rl
                        dd = max(0.0, dd) ** fog[5]
                        vy = min(1., max(0., (wz / rl) / .35)) if fog[6] > 0 else 0.
                        fr = fogc[0, 0] * (1 - dd) + fogc[1, 0] * dd; fg = fogc[0, 1] * (1 - dd) + fogc[1, 1] * dd; fb = fogc[0, 2] * (1 - dd) + fogc[1, 2] * dd
                        fr += (fogc[3, 0] - fr) * vy; fg += (fogc[3, 1] - fg) * vy; fb += (fogc[3, 2] - fb) * vy
                        emk = k * (1 - fog[7] * min(1., em_))
                        r_ += (fr - r_) * emk; g_ += (fg - g_) * emk; b_ += (fb - b_) * emk
                    rr += r_; gg += g_; bb += b_; cc += 1
            out[yy, xx, 0] = rr * k2; out[yy, xx, 1] = gg * k2; out[yy, xx, 2] = bb * k2; cov[yy, xx] = cc * k2
            # 1x auxiliaries from the centre sample
            py = yy * ss + ss // 2; px = xx * ss + ss // 2
            i = ID[py, px]
            if i >= 0:
                z = 1.0 / ZB[py, px]; q = TQ[i]
                zout[yy, xx] = z; lout[yy, xx] = lab[q]
                dx = (xx + .5 - cx) / f * z; dy = (y0 + yy + .5 - cy) / f * z
                ddx = R[0, 0] * QD[q, 0] + R[0, 1] * QD[q, 1] + R[0, 2] * QD[q, 2]
                ddy = R[1, 0] * QD[q, 0] + R[1, 1] * QD[q, 1] + R[1, 2] * QD[q, 2]
                ddz = R[2, 0] * QD[q, 0] + R[2, 1] * QD[q, 1] + R[2, 2] * QD[q, 2]
                aout[yy, xx] = math.atan2(ddy * z - dy * ddz, ddx * z - dx * ddz)
