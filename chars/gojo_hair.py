"""Gojo's hair: 3-D locks (a centre line rooted on the skull + a flame profile) painted back to front, inked by
union boundary so the silhouette is one decisive contour and the inside has only a few lines that start at a tip."""
import numpy as np, math, cv2, cairo
from jj.ink import spl, resample, ribbon
from jj.paint import rgb

D2R = math.pi / 180


def surf(H, th, y, off=0.):
    """point on the head shell at azimuth th (0 front, + to +x) and height y, pushed out by off; plus its normal"""
    th = np.asarray(th, float); y = np.asarray(y, float)
    A = np.interp(y, H.yy, H.A); F = np.interp(y, H.yy, H.F); B = np.interp(y, H.yy, H.B); p = H.k['p']
    sx, cz = np.sin(th), np.cos(th); r = (np.abs(sx) ** p + np.abs(cz) ** p) ** (-1 / p)
    D = np.where(cz >= 0, F, B)
    P = np.stack([(A + off) * sx * r, y * np.ones_like(th), (D + off) * cz * r], -1)
    n = np.stack([sx / np.maximum(A, .15), (y - .12) * .9 * np.ones_like(th), cz / np.maximum(D, .15)], -1)
    return P, n / np.linalg.norm(n, axis=-1, keepdims=True)


def bez3(p0, p1, p2, p3, n=26):
    u = np.linspace(0, 1, n)[:, None]
    return (1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * p1 + 3 * (1 - u) * u * u * p2 + u ** 3 * p3


class Lock:
    """one clump.  th deg + y0 = root on the skull; tip = 3-D; d = direction the tip points; w = root width (units);
    pw = taper power (1.6 convex leaf, 1 straight, .8 concave flame); asym = which side carries the belly (-1..1);
    k1,k2 handle lengths; out = how much it first grows along the skull normal; ln = length of the inner lines from the tip;
    cs = cast a hard shadow on the hair behind it; sh = shade wedge strength"""
    def __init__(s, th, y0, tip, w, d=None, pw=1.15, belly=.06, asym=0., k1=.36, k2=.34, bias=0., ln=0., out=.55, inset=.06, cs=0., sh=0., root=None, ol=1., thick=.58, r0=.78, rf=.24):
        s.th, s.y0, s.tip, s.w, s.d, s.pw, s.belly, s.asym, s.k1, s.k2, s.bias, s.ln, s.out, s.inset, s.cs, s.sh, s.root, s.ol, s.thick, s.r0, s.rf = \
            th, y0, np.asarray(tip, float), w, d, pw, belly, asym, k1, k2, bias, ln, out, inset, cs, sh, root, ol, thick, r0, rf

    def centre(s, H, sway=None, lift=0.):
        if s.root is not None:
            root = np.asarray(s.root, float); n = root / (np.linalg.norm(root) + 1e-6)
        else: root, n = surf(H, s.th * D2R, s.y0, -s.inset)
        tip = s.tip.copy(); L = np.linalg.norm(tip - root) + 1e-6
        if lift and not (abs(s.th) < 64 and s.tip[1] > -.1): lift = lift * (.45 if lift > .6 else .22)
        if lift:                                    # flung up (1) .. fallen (0), a little past (< 0): for the moment the band lets go
            up = n * np.array([.9, .4, .5]) + np.array([.0, 1.0, .0]); up = up / np.linalg.norm(up); k = min(1., max(0., lift)); v = (tip - root) / L * (1 - lift) + up * lift + n * .38 * math.sin(math.pi * k); tip = root + v / (np.linalg.norm(v) + 1e-6) * L * (1 - .08 * k)
        u = (tip - root) / L
        g = n * s.out + u * (1 - s.out); g /= np.linalg.norm(g)
        d = u if (s.d is None or lift > .5) else np.asarray(s.d, float) * (1 - max(lift, 0)) + u * max(lift, 0)
        d = d / (np.linalg.norm(d) + 1e-9)
        c1 = root + g * L * s.k1; c2 = tip - d * L * s.k2
        if sway is not None:
            sw = np.asarray(sway, float) * L; tip = tip + sw; c2 = c2 + sw * .45
        return bez3(root, c1, c2, tip), L


def lock_sides(C, w, belly, asym, pw, r0=.78):
    """C = projected centre line; returns left side, right side (root->tip)"""
    t = np.linspace(0, 1, len(C)); T = np.gradient(C, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9
    N = np.c_[-T[:, 1], T[:, 0]]
    base = (1 - t ** pw) * .5 if pw >= 1 else ((1 - t) ** (1 / pw)) * .5
    base = base * np.interp(t, [0, .16, 1], [r0, 1, 1])
    bl = np.sin(np.pi * t ** .8) * belly
    a = w * (base + bl * (.5 + .5 * asym)); b = w * (base + bl * (.5 - .5 * asym))
    return C + N * a[:, None], C - N * b[:, None]


def _seg_runs(m):
    out = []; i = 0; n = len(m)
    while i < n:
        if m[i]:
            j = i
            while j + 1 < n and m[j + 1]: j += 1
            out.append((i, j)); i = j + 1
        else: i += 1
    return out


def _fade_fill(cel, P, g0, g1, stops, a0, a1):
    """gradient fill of P whose root end melts into the hair under it (so no lock shows where it was planted)"""
    c = cel.ctx; cel.clip(P); c.push_group(); cel.grad(P, g0, g1, stops); c.pop_group_to_source()
    m = cairo.LinearGradient(a0[0], a0[1], a1[0], a1[1]); m.add_color_stop_rgba(0, 0, 0, 0, 0); m.add_color_stop_rgba(.35, 0, 0, 0, .25); m.add_color_stop_rgba(1, 0, 0, 0, 1)
    c.mask(m); cel.unclip()


class Hair:
    """projected locks of one drawing; paint in two passes (behind the face / in front of it)"""
    def __init__(s, H, locks, pal, cap=None, cap_back=None, seed=0, sway=None, lw=1.0, grad=None, light=(-.6, -.8), strands=(), split=0., gst=None, gbk=None, light3=None, thr=-.02, capM=None, fade_y=None, gdk=None, lift=0.):
        s.fade_y = fade_y
        s.darkM = None if (light3 is None or capM is None) else (capM, light3, thr)
        s.H, s.pal, s.seed, s.lw, s.split = H, pal, seed, lw, split
        s.caps = [] if cap is None else (cap if isinstance(cap, list) else [cap])
        s.cback = [] if cap_back is None else [cap_back]
        s.L2 = np.asarray(light, float) / (np.hypot(*light) + 1e-6)
        s.items = []
        for k, q in enumerate(locks):
            sw = None if sway is None else sway(k, q)
            C3, L = q.centre(H, sw, lift); C = H.P3(C3); dp = float(H.depth(C3[4:]).mean()) + q.bias
            # a lock is a flat ribbon lying on the skull: foreshorten its width, keep some thickness
            nn = C3[4] / (np.linalg.norm(C3[4]) + 1e-6); tg = C3[12] - C3[4]; e3 = np.cross(tg, nn); e3 = H.R @ (e3 / (np.linalg.norm(e3) + 1e-6))
            fw = max(q.thick, min(1., float(np.hypot(e3[0], e3[1]))))
            A, B = lock_sides(C, q.w * H.s * fw, q.belly, q.asym, q.pw, q.r0)
            dark = False
            if light3 is not None and abs(light3[0]) > .3:
                sg = -np.sign(light3[0]); az = ((q.th + math.degrees(H.yaw) + 180) % 360) - 180
                dark = (az * sg > 13 or float((H.R @ C3[-1])[0]) * sg > .95) if abs(az) < 100 else (float((H.R @ C3[-1])[0]) * sg > .12)
            s.items.append(dict(Y=C3[:, 1], dark=dark, k=k, q=q, C=C, A=A, B=B, P=np.vstack([A, B[::-1]]), dp=dp))
        s.items.sort(key=lambda i: i['dp'])
        polys = [i['P'] for i in s.items] + s.caps + s.cback
        allp = np.vstack(polys); s.lo = np.floor(allp.min(0)) - 8; hi = np.ceil(allp.max(0)) + 8
        s.kx = min(1.0, 1500. / max(hi - s.lo)); s.w_, s.h_ = np.ceil((hi - s.lo) * s.kx).astype(int) + 2
        m = np.zeros((s.h_, s.w_), np.uint8)
        for p in polys: cv2.fillPoly(m, [np.round((p - s.lo) * s.kx * 16).astype(np.int32)], 255, cv2.LINE_8, 4)
        s.m = m; s.polys = polys
        # root airbrush: one gradient for the whole mass, along the head's up axis (screen)
        if grad is None: grad = ((0, .38, .2), (0, 1.12, 0.))
        a, b = H.P3(np.array(grad, float))
        s.g0, s.g1 = a, b
        s.gst = [(0, pal['hair_root'], 1), (.30, pal['hair_root2'], 1), (.80, pal['hair'], 1), (1, pal['hair'], 1)]
        if gst is not None: s.gst = [(t, pal[k], 1) for t, k in gst]
        s.gdk = [(t, pal[k], 1) for t, k in (gdk or ((0, 'hair_dp'), (.30, 'hair_bk'), (1, 'hair_sh')))]
        s.gbk = [(t, pal[k], 1) for t, k in gbk] if gbk is not None else [(0, pal['hair_sh'], 1), (.5, pal['hair_bk'], 1), (1, pal['hair_bk2'], 1)]

    def inside(s, Q):
        q = np.round((Q - s.lo) * s.kx).astype(int); q[:, 0] = np.clip(q[:, 0], 0, s.w_ - 1); q[:, 1] = np.clip(q[:, 1], 0, s.h_ - 1)
        return s.m[q[:, 1], q[:, 0]] > 0

    def dark_cap(s, M, light3, thr):
        """the part of the cap that turns from the light, as a hard shape with a ragged (strand) edge"""
        if abs(light3[0]) < .3: return []
        q_ = M @ np.array([[math.cos(s.H.yaw), 0, -math.sin(s.H.yaw)], [0, 1, 0], [math.sin(s.H.yaw), 0, math.cos(s.H.yaw)]]); az = np.degrees(np.arctan2(q_[..., 0], q_[..., 2])) * -np.sign(light3[0])
        d = 13 - az + 3 * np.sin(M[..., 1] * 11.); thr = 0.; P = s.H.P3(M); m = np.zeros((s.h_, s.w_), np.uint8); q = np.round((P - s.lo) * s.kx * 16).astype(np.int32)
        dk = (d[:-1, :-1] + d[1:, 1:] + d[:-1, 1:] + d[1:, :-1]) / 4 < thr
        fc = (s.H.depth(M)[:-1, :-1] + s.H.depth(M)[1:, 1:]) / 2 > -.5
        for i, j in zip(*np.nonzero(dk & fc)): cv2.fillConvexPoly(m, np.array([q[i, j], q[i, j + 1], q[i + 1, j + 1], q[i + 1, j]]), 255, cv2.LINE_8, 4)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); out = []
        for c_ in cs:
            c_ = c_[:, 0].astype(float)
            if len(c_) < 40: continue
            kk = max(9, int(len(c_) * .03) | 1); k = np.hanning(kk); k /= k.sum(); cc = np.vstack([c_[-kk:], c_, c_[:kk]]); out.append(np.stack([np.convolve(cc[:, i], k, 'same') for i in (0, 1)], -1)[kk:-kk] / s.kx + s.lo)
        return out

    def paint(s, cel, back=True, front=True, cap=True):
        pal, H = s.pal, s.H; done = not (s.caps and cap and front)
        def capf():
            for p in s.caps:
                cel.grad(p, s.g0, s.g1, s.gst)
                if s.darkM is not None:
                    cel.clip(p)
                    for q_ in s.dark_cap(*s.darkM): cel.grad(q_, s.g0, s.g1, s.gdk)
                    cel.unclip()
        for it in s.items:
            q = it['q']; bk = it['dp'] < s.split
            if bk and not back: continue
            if not bk and not done: capf(); done = True
            if not bk and not front: continue
            C, A, B, P = it['C'], it['A'], it['B'], it['P']
            if q.cs:                                # hard shadow thrown on the hair behind: a wedge outside the far side, zero at root and tip
                tt = np.linspace(0, 1, len(C)); nA_ = (A - C); sd_ = A if float((nA_[8:18].mean(0) @ s.L2)) < 0 else B
                on = (sd_ - C); on /= np.hypot(on[:, 0], on[:, 1])[:, None] + 1e-6
                cut = int(len(C) * .42); wd = np.interp(tt, [0, .42, .62, .84, 1], [0, 0, .8, 1.15, .25]) * q.cs * H.s
                cel.fill(np.vstack([sd_[cut:], (C[-1] - s.L2 * q.cs * H.s * .5)[None], (sd_ + on * wd[:, None])[cut:][::-1]]), pal['hair_sh'])
            st_ = s.gdk if it['dark'] else s.gbk if (bk and it['dp'] < s.split - .25) else s.gst
            k1 = int(len(C) * q.rf)
            if s.fade_y is not None and it['Y'][-1] < s.fade_y: k1 = min(k1, int(np.argmax(it['Y'] < s.fade_y)) - 1)
            if bk or k1 < 3: cel.grad(P, s.g0, s.g1, st_)
            else:
                ax = C[-1] - C[0]; ax = ax / (np.hypot(*ax) + 1e-6); e1 = float((C[k1] - C[0]) @ ax)
                if e1 < 6: cel.grad(P, s.g0, s.g1, st_)
                else: _fade_fill(cel, P, s.g0, s.g1, st_, C[0] + ax * e1 * .1, C[0] + ax * e1)
            t = np.linspace(0, 1, len(C)); nA = (A - C)
            if q.sh:
                side = A if float((nA[6:14].mean(0) @ s.L2)) < 0 else B
                wd = np.interp(t, [0, .20, .52, .74, .95, 1], [0, 0, .62, .66, .16, 0]) * q.sh
                cel.fill(np.vstack([side, (side + (C - side) * wd[:, None])[::-1]]), pal['hair_bk'] if it['dark'] else pal['hair_sh'])
            if q.ln:
                for sd, e in ((0, A), (1, B)):
                    f0 = 1 - q.ln * (1.0 if sd == (it['k'] % 2) else .62); idx = np.where(t >= f0)[0]
                    if len(idx) > 3: cel.line(e[idx], 2.1 * s.lw, pal['hair_in'], taper=(.8, .10), press=.2, seed=s.seed * 13 + it['k'] * 7 + sd, tremor=.2, tmin=.05)
        if not done and front: capf()
        if back:
            for p in s.cback: cel.grad(p, s.g0, s.g1, s.gst)

    def ink_cap(s, cel, back=True, front=True):
        for j, p in enumerate((s.caps if front else []) + (s.cback if back else [])):
            E = resample(np.vstack([p, p[:1]]), 2.0); T = np.gradient(E, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9
            N = np.c_[-T[:, 1], T[:, 0]]; ar = .5 * np.sum(E[:-1, 0] * E[1:, 1] - E[1:, 0] * E[:-1, 1]); N = -N * np.sign(ar) if True else N
            bd = (~s.inside(E + N * 2.4)) & (~s.inside(E - N * 2.4) == False)
            bd = np.convolve(bd.astype(float), np.ones(7) / 7, 'same') > .6
            for a, b in _seg_runs(bd):
                if b - a < 14: continue
                cel.line(E[a:b + 1], 3.3 * s.lw, s.pal['hair_line'], taper=(.3, .3), press=.35, seed=s.seed * 19 + j, tremor=.15, tmin=.1)

    def ink(s, cel, back=True, front=True):
        for it in s.items:
            bk = it['dp'] < s.split
            if (bk and not back) or (not bk and not front) or it['q'].ol <= 0: continue
            for sd, e in ((0, it['A']), (1, it['B'])):
                E = resample(e, 2.0); T = np.gradient(E, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9
                N = np.c_[-T[:, 1], T[:, 0]] * (1 if sd == 0 else -1)
                bd = (~s.inside(E + N * 2.2)).astype(float)
                bd = np.convolve(bd, np.ones(5) / 5, 'same') > .5
                for a, b in _seg_runs(bd):
                    if b - a < 6: continue
                    e0 = max(0, a - 2); e1 = min(len(E) - 1, b + 2)
                    tp = e1 >= len(E) - 3
                    cel.line(E[e0:e1 + 1], 3.5 * s.lw * it['q'].ol, s.pal['hair_line'], taper=(.28, .10 if tp else .25), press=.35, seed=s.seed * 17 + it['k'] * 5 + sd, tremor=.15, tmin=.22)

    def strands(s, cel, S, w=1.8, dz=0.):
        """thin interior lines: list of [(deg, y0, (tip3), bend_dir, width)] growing on the same field as the locks"""
        for k, (th, y0, tip, d, ww) in enumerate(S):
            q = Lock(th, y0, tip, .1, d, inset=-.02, out=.5); C3, _ = q.centre(s.H)
            n = s.H.R @ (C3[10] / (np.linalg.norm(C3[10]) + 1e-6))
            if n[2] < .12: continue
            C = s.H.P3(C3)
            if not s.inside(C).all() or np.hypot(*(C[-1] - C[0])) < .24 * s.H.s: continue
            cel.line(C, w * ww * s.lw, s.pal['hair_in'], taper=(.35, .7), press=.25, seed=s.seed * 29 + k, tremor=.15, tmin=.04)

    def hilite(s, cel, S, w=2.2):
        for k, (th, y0, tip, d, ww) in enumerate(S):
            q = Lock(th, y0, tip, .1, d, inset=-.03, out=.5); C3, _ = q.centre(s.H)
            n = s.H.R @ (C3[10] / (np.linalg.norm(C3[10]) + 1e-6))
            if n[2] < .2: continue
            if not s.inside(s.H.P3(C3)).all() or np.hypot(*(s.H.P3(C3[-1:])[0] - s.H.P3(C3[:1])[0])) < .24 * s.H.s: continue
            cel.line(s.H.P3(C3), w * ww * s.lw, s.pal['hair_hi'], taper=(.5, .5), press=.3, seed=s.seed * 31 + k, tremor=.1, tmin=.04)


# ---------------------------------------------------------------------------------------------- pushed up
# authored in head units: x right, y up, z toward the viewer.  Sizes are picked by eye, never regular.
UP = [
    # rows behind (valleys from the front, silhouette from the side)
    Lock(180, .36, (.04, .52, -1.30), .34, (0, -.25, -1)), Lock(162, .44, (.40, .74, -1.30), .44, (.2, .25, -1)), Lock(-160, .42, (-.46, .80, -1.28), .40, (-.2, .3, -1)),
    Lock(180, .60, (-.02, 1.00, -1.44), .60, (0, .55, -.9)), Lock(168, .76, (.20, 1.38, -1.26), .60, (.1, .8, -.6)),
    Lock(-166, .78, (-.24, 1.50, -1.14), .58, (-.1, .9, -.5)), Lock(180, .92, (.0, 1.74, -.90), .64, (.1, 1, -.35)),
    Lock(128, .44, (.82, .82, -.92), .36, (.7, .5, -.4)), Lock(-130, .48, (-.84, 1.00, -.92), .42, (-.6, .8, -.4)),
    Lock(120, .60, (.92, 1.14, -.72), .46, (.4, 1, -.3)), Lock(-112, .68, (-.90, 1.32, -.62), .44, (-.3, 1, -.2)),
    Lock(110, .76, (.74, 1.56, -.50), .46, (.2, 1, -.1)), Lock(-100, .84, (-.66, 1.60, -.44), .40, (-.1, 1, -.1)),
    Lock(98, .40, (1.00, .60, -.42), .24, (1, .0, -.2)), Lock(-99, .42, (-1.05, .68, -.42), .28, (-1, .1, -.2)),
    Lock(88, .56, (1.14, .98, -.26), .36, (.5, .8, 0)), Lock(-88, .58, (-1.12, 1.00, -.24), .34, (-.5, .9, 0)),
    Lock(150, .95, (.28, 1.82, -.56), .42, (.4, 1, -.1)), Lock(-140, .95, (-.20, 1.80, -.50), .40, (.3, 1, -.1)),
    Lock(100, .95, (.34, 1.78, -.22), .42, (.3, 1, 0)), Lock(-95, .93, (-.30, 1.82, -.18), .44, (.2, 1, 0)),
    # the crown
    Lock(-74, .46, (-1.02, .44, .14), .24, (-1, .05, 0), out=.75),
    Lock(-72, .52, (-1.22, .78, .04), .40, (-.75, .75, 0), pw=.85, asym=.4),
    Lock(-62, .62, (-1.14, 1.12, .12), .52, (-.15, 1, 0), pw=.86, out=.8, k1=.42),
    Lock(-46, .70, (-.98, 1.46, .14), .62, (-.30, 1, 0), pw=.85, out=.72, k1=.42, sh=.35),
    Lock(-28, .78, (-.44, 1.70, .16), .62, (.55, 1, 0), pw=.85, out=.60, k1=.42, k2=.40),
    Lock(2, .80, (.14, 1.98, .02), .82, (.75, 1, -.1), pw=.84, out=.25, k2=.40, sh=.34),
    Lock(24, .78, (.50, 1.74, .14), .56, (-.10, 1, 0), pw=.85, out=.58),
    Lock(42, .70, (.94, 1.56, .12), .66, (.45, 1, 0), pw=.84, out=.66, k2=.4, sh=.40),
    Lock(58, .62, (1.06, 1.18, .16), .48, (-.10, 1, 0), pw=.86, out=.85, k1=.42),
    Lock(68, .54, (1.30, .94, .04), .42, (1, .62, 0), pw=.85),
    Lock(74, .45, (1.12, .50, .16), .24, (1, -.15, 0), out=.75),
    # the few tips that end inside the mass
    Lock(-30, .46, (-.50, 1.24, .44), .52, (.10, 1, -.1), pw=.82, ln=.50, cs=.030, out=.22),
    Lock(12, .46, (.26, 1.40, .42), .60, (.55, 1, -.1), pw=.82, ln=.58, cs=.034, out=.15),
    Lock(48, .46, (.76, 1.06, .34), .46, (.50, 1, 0), pw=.82, ln=.46, cs=.030, out=.35),
]
UP_STRANDS = [(-52, .44, (-.86, .96, .40), (-.3, 1, 0), 1.0), (-40, .44, (-.70, 1.10, .52), (-.3, 1, 0), .8), (-14, .42, (-.16, 1.06, .78), (-.2, 1, 0), 1.0), (-6, .42, (-.04, .84, .84), (.1, 1, 0), .7),
              (2, .42, (.12, 1.02, .80), (.4, 1, 0), .9), (26, .42, (.46, .92, .70), (.4, 1, 0), 1.0), (38, .43, (.60, .80, .66), (.5, 1, 0), .7), (60, .44, (.94, .80, .34), (.7, .8, 0), .9),
              (-64, .44, (-.98, .74, .30), (-.7, .7, 0), .8), (-24, .42, (-.36, .78, .74), (-.2, 1, 0), .7), (16, .95, (.36, 1.50, .40), (.3, 1, 0), .8), (-10, .9, (-.10, 1.46, .40), (.3, 1, 0), .8)]
UP_HI = [(-38, .60, (-.62, 1.18, .56), (-.2, 1, 0), 1.2), (30, .62, (.62, 1.22, .50), (.4, 1, 0), 1.0), (34, .60, (.60, 1.06, .55), (.4, 1, 0), .7), (-30, .62, (-.46, 1.06, .62), (-.1, 1, 0), .8), (6, .66, (.18, 1.36, .6), (.4, 1, 0), 1.)]


# ---------------------------------------------------------------------------------------------- fallen down, soft, parted
DOWN = [
    # behind the neck
    Lock(180, .30, (.06, -.82, -.92), .60, (.1, -1, -.1), belly=.1), Lock(158, .30, (.42, -.78, -.84), .52, (.3, -1, 0)), Lock(-160, .30, (-.40, -.84, -.84), .52, (-.25, -1, 0)),
    Lock(136, .30, (.74, -.70, -.62), .46, (.5, -1, 0)), Lock(-138, .30, (-.72, -.66, -.64), .48, (-.55, -1, 0)), Lock(170, .55, (.20, -.50, -1.16), .5, (.1, -1, -.2)), Lock(-170, .55, (-.2, -.44, -1.18), .5, (-.1, -1, -.2)),
    Lock(150, .60, (.62, -.34, -1.0), .5, (.3, -1, -.3)), Lock(-150, .60, (-.62, -.40, -1.0), .5, (-.3, -1, -.3)),
    # side silhouette flicks
    Lock(112, .20, (.90, -.60, -.28), .36, (.45, -1, 0)), Lock(-112, .22, (-.88, -.54, -.30), .38, (-.5, -1, 0)),
    Lock(100, .48, (1.02, -.22, -.18), .42, (.85, -.75, 0), out=.75), Lock(-102, .46, (-.98, -.16, -.20), .40, (-.9, -.5, 0), out=.75),
    Lock(94, .72, (1.00, .16, -.10), .44, (.9, -.55, 0), out=.75), Lock(-96, .74, (-1.03, .28, -.12), .46, (-.95, -.15, 0), out=.75),
    Lock(120, .84, (.94, .58, -.36), .36, (.95, .1, 0), out=.7), Lock(-122, .82, (-.92, .62, -.36), .34, (-.9, .3, 0), out=.7),
    # crown: soft, only a few ends that leave the dome
    Lock(140, .96, (.84, .90, -.50), .46, (.8, -.7, -.2), out=.4), Lock(-142, .96, (-.80, .98, -.48), .46, (-.8, -.5, -.2), out=.4),
    Lock(120, .97, (.60, 1.15, -.20), .34, (.95, -.1, 0), out=.5), Lock(-118, .97, (-.66, 1.12, -.14), .36, (-.95, -.2, 0), out=.5),
    Lock(80, .93, (.86, .74, .06), .50, (.7, -.8, .1), out=.45), Lock(-80, .93, (-.88, .82, .04), .50, (-.8, -.6, .1), out=.45),
    # cheek pieces
    Lock(74, .72, (.68, -.58, .24), .36, (-.10, -1, -.1), belly=.14, sh=.5, ln=.35), Lock(-74, .72, (-.69, -.50, .26), .38, (.08, -1, -.1), belly=.14, ln=.35),
    Lock(62, .84, (.70, -.24, .44), .44, (.10, -1, -.1), belly=.14, sh=.45, ln=.4), Lock(-62, .84, (-.73, -.16, .44), .46, (-.16, -1, -.1), belly=.14, sh=.2, ln=.4),
    # the fringe, from a part a little right of centre
    Lock(46, .90, (.66, .06, .66), .46, (.34, -1, 0), belly=.18, asym=-.3, sh=.45, ln=.45), Lock(-44, .90, (-.64, .12, .66), .48, (-.4, -1, 0), belly=.18, asym=.3, sh=.25, ln=.45),
    Lock(30, .92, (.50, .26, .80), .22, (.30, -1, 0), belly=.1, ln=.3), Lock(-30, .92, (-.46, .24, .82), .24, (-.30, -1, 0), belly=.1, ln=.3),
    Lock(28, .93, (.40, -.02, .80), .40, (.18, -1, .05), belly=.2, asym=-.3, sh=.4, ln=.5),
    Lock(-24, .93, (-.36, .10, .80), .44, (-.34, -1, .05), belly=.2, asym=.35, sh=.2, ln=.5),
    Lock(-6, .94, (-.17, -.08, .88), .40, (-.22, -1, .05), belly=.22, asym=.3, sh=.3, ln=.5),
    Lock(14, .94, (.14, .12, .90), .28, (.30, -1, .05), belly=.16, asym=-.2, sh=.3, ln=.45),
    Lock(4, .95, (-.02, .22, .92), .16, (-.2, -1, 0), belly=.1, ln=.3),
]
DOWN_STRANDS = [(-30, .92, (-.52, .42, .74), (-.3, -1, 0), 1.), (-14, .93, (-.26, .46, .84), (-.2, -1, 0), .8), (36, .92, (.54, .44, .72), (.3, -1, 0), 1.), (20, .93, (.30, .50, .84), (.2, -1, 0), .8),
                (-52, .9, (-.74, .42, .50), (-.2, -1, 0), .9), (54, .9, (.74, .40, .50), (.2, -1, 0), .9), (-70, .86, (-.84, .36, .22), (-.2, -1, 0), .8), (72, .86, (.84, .30, .22), (.2, -1, 0), .8),
                (-58, .93, (-.66, .72, .48), (-.5, -1, .1), .8), (62, .93, (.68, .66, .46), (.5, -1, .1), .8), (-2, .95, (-.08, .56, .86), (-.2, -1, 0), .7)]
DOWN_HI = [(-36, .94, (-.50, .70, .70), (-.4, -1, 0), 1.3), (-30, .94, (-.40, .74, .76), (-.4, -1, 0), .8), (40, .94, (.54, .66, .68), (.4, -1, 0), 1.1), (46, .94, (.62, .66, .6), (.4, -1, 0), .7)]


# ---------------------------------------------------------------------------------------------- bare face, blown up and to the right
WIND = [
    Lock(180, .30, (.40, .10, -1.34), .50, (.7, -.4, -.6)), Lock(160, .40, (.96, .36, -1.10), .50, (1, -.1, -.3)), Lock(-160, .40, (.02, .46, -1.36), .50, (.6, .2, -.8)),
    Lock(180, .62, (.46, .86, -1.32), .60, (.8, .3, -.6)), Lock(168, .78, (.92, 1.10, -1.04), .58, (.95, .3, -.4)), Lock(-166, .78, (.10, 1.34, -1.10), .58, (.6, .7, -.4)),
    Lock(180, .92, (.52, 1.50, -.76), .60, (.9, .6, -.2)), Lock(140, .94, (1.06, 1.34, -.52), .48, (1, .3, -.1)), Lock(-140, .94, (-.14, 1.60, -.50), .48, (.5, .9, -.1)),
    Lock(-104, .86, (-.60, 1.50, -.36), .44, (.2, 1, 0)), Lock(-98, .70, (-.94, 1.22, -.30), .42, (-.25, 1, 0)), Lock(-94, .50, (-1.12, .86, -.22), .40, (-.6, .7, 0)), Lock(-98, .28, (-1.06, .44, -.30), .34, (-.8, .4, 0)), Lock(-104, .05, (-.98, .02, -.34), .30, (-.8, -.2, 0)), Lock(-112, -.15, (-.84, -.40, -.36), .28, (-.5, -.8, 0)),
    Lock(104, .86, (1.34, 1.06, -.30), .50, (1, .12, 0)), Lock(100, .62, (1.56, .66, -.24), .46, (1, -.02, 0)), Lock(100, .38, (1.50, .22, -.30), .42, (1, -.15, 0)), Lock(106, .12, (1.30, -.18, -.36), .38, (1, -.3, 0)), Lock(116, -.10, (1.04, -.54, -.42), .32, (.85, -.6, 0)),
    # crown: swept over to the right
    Lock(-74, .70, (-1.02, .92, .10), .44, (-.6, .7, 0), pw=.86, out=.8, ln=.3),
    Lock(-60, .84, (-.82, 1.30, .12), .50, (-.1, 1, 0), pw=.85, out=.8, ln=.35, sh=.2),
    Lock(-42, .92, (-.24, 1.50, .12), .58, (.75, .8, 0), pw=.84, out=.8, sh=.3, ln=.4),
    Lock(-14, .95, (.42, 1.58, .14), .62, (1, .45, 0), pw=.84, out=.8, sh=.3, ln=.4),
    Lock(16, .95, (1.06, 1.36, .14), .60, (1, .12, 0), pw=.84, out=.8, ln=.45),
    Lock(44, .88, (1.42, 1.02, .12), .56, (1, .0, 0), pw=.84, out=.7, sh=.38, ln=.45),
    Lock(60, .78, (1.50, .78, .08), .48, (1, -.04, 0), pw=.85, out=.6),
    Lock(72, .66, (1.44, .40, .04), .42, (1, -.15, 0), pw=.85, out=.6, sh=.35),
    Lock(82, .46, (1.26, .06, .0), .36, (1, -.28, 0), pw=.86), Lock(88, .28, (1.06, -.26, .02), .30, (.9, -.55, 0), pw=.86),
    Lock(-84, .50, (-.96, .46, .06), .34, (-.75, .5, 0), pw=.86), Lock(-86, .30, (-.86, .0, .08), .28, (-.6, -.5, 0), pw=.86),
    Lock(-74, .60, (-.76, -.34, .30), .30, (-.1, -1, 0), belly=.14, ln=.35),
    # fringe: lifted off the forehead, every lock on its own curve
    Lock(-58, .84, (-.82, .10, .52), .40, (-.45, -.85, .1), belly=.16, asym=.3, ln=.45, k1=.46, out=.75),
    Lock(-42, .90, (-.56, .16, .74), .40, (-.10, -1, .1), belly=.18, asym=.4, ln=.5, k1=.46, out=.8, sh=.2),
    Lock(54, .86, (.92, .30, .46), .42, (.9, -.5, .1), belly=.16, asym=-.3, ln=.45, k1=.46, out=.75, sh=.4),
    Lock(38, .90, (.72, .28, .68), .40, (.8, -.6, .1), belly=.16, asym=-.3, ln=.5, k1=.46, out=.8, sh=.3),
    Lock(-26, .92, (-.30, .06, .86), .40, (.15, -1, .1), belly=.18, asym=.4, ln=.5, k1=.46, out=.8, sh=.2),
    Lock(22, .92, (.46, .22, .82), .40, (.7, -.7, .1), belly=.18, asym=.3, ln=.5, k1=.46, out=.8, sh=.3),
    Lock(-10, .94, (-.04, .02, .92), .36, (.45, -.9, .1), belly=.20, asym=.4, ln=.5, k1=.46, out=.8, sh=.25),
    Lock(6, .94, (.22, .30, .92), .28, (.8, -.55, .1), belly=.16, asym=.3, ln=.5, k1=.46, out=.8),
    Lock(-34, .91, (-.46, .34, .82), .18, (-.3, -1, 0), belly=.1, ln=.3), Lock(30, .91, (.64, .46, .74), .18, (.95, -.3, 0), belly=.1, ln=.3),
]
WIND_STRANDS = [(-64, .90, (-.80, .70, .40), (-.5, -1, 0), 1.), (-44, .94, (-.60, .56, .68), (-.2, -1, 0), .9), (-24, .95, (-.28, .60, .84), (.1, -1, 0), 1.1), (-6, .95, (.04, .52, .90), (.3, -1, 0), .9), (-56, .93, (-.58, 1.16, .30), (.2, 1, 0), 1.), (-36, .95, (-.16, 1.34, .34), (.7, .8, 0), .9), (-70, .80, (-.92, .62, .28), (-.6, -.3, 0), .8), (-50, .88, (-.74, .50, .56), (-.2, -1, 0), 1.), (-18, .93, (-.16, .46, .86), (.2, -1, 0), 1.), (30, .92, (.58, .50, .76), (.7, -.7, 0), .9), (14, .93, (.32, .56, .86), (.6, -.8, 0), .8), (-30, .96, (.10, 1.36, .44), (.8, .8, 0), 1.), (0, .97, (.62, 1.34, .40), (1, .4, 0), 1.), (34, .94, (1.04, 1.06, .34), (1, .1, 0), .9), (56, .84, (1.18, .66, .28), (1, -.1, 0), .8), (-52, .92, (-.36, 1.30, .36), (.5, 1, 0), .8)]
WIND_HI = [(-38, .94, (-.46, .72, .74), (-.2, -1, 0), 1.2), (-20, .95, (.0, 1.30, .52), (.7, .8, 0), 1.), (24, .95, (.78, 1.20, .42), (1, .4, 0), 1.), (34, .93, (.60, .66, .74), (.6, -.8, 0), .8)]
