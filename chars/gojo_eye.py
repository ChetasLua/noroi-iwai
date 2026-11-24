"""The Six Eyes.  iris() paints a layered crystalline iris as vector art in unit coordinates (any size, any foreshortening)."""
import numpy as np, math, cairo
from jj.ink import spl, resample
from jj.paint import rgb

IR = dict(limb='#0d1f55', ring='#184a9e', mid='#2f8fe6', body='#63c6f8', lite='#b4f0ff', core='#f2fdff', pupil='#0a1e4a', pup2='#154a96', cloud='#ffffff', cloud_sh='#a9cdf2', deep='#123a86')


def _rad(cel, M, pts, r0, r1, stops, alpha=1., centre=(0., 0.)):
    c = cel.ctx; c.save(); c.transform(M)
    g = cairo.RadialGradient(centre[0], centre[1], r0, centre[0], centre[1], r1)
    for t, col, a in stops:
        k = rgb(col); g.add_color_stop_rgba(t, float(k[0]), float(k[1]), float(k[2]), a * alpha)
    c.new_path()
    if pts is None: c.arc(0, 0, 1, 0, 2 * math.pi)
    else:
        c.move_to(*pts[0])
        for p in pts[1:]: c.line_to(*p)
        c.close_path()
    c.set_source(g); c.fill(); c.restore()


def _lin(cel, M, pts, p0, p1, stops):
    c = cel.ctx; c.save(); c.transform(M); g = cairo.LinearGradient(*p0, *p1)
    for t, col, a in stops:
        k = rgb(col); g.add_color_stop_rgba(t, float(k[0]), float(k[1]), float(k[2]), a)
    c.new_path()
    if pts is None: c.arc(0, 0, 1, 0, 2 * math.pi)
    else:
        c.move_to(*pts[0])
        for p in pts[1:]: c.line_to(*p)
        c.close_path()
    c.set_source(g); c.fill(); c.restore()


def iris(cel, c, ex, ey, t=0., pupil=.30, seed=7, detail=1., par=(0., 0.), glow=.0, spin=0., pal=IR, shade_top=1.):
    """c = centre, ex, ey = screen vectors of unit x, unit y (y down).  detail 0..1.  par = parallax of the inner layers
    (fraction of radius), glow 0..1 = light pouring out of the core, spin = slow drift of the broken rings."""
    c = np.asarray(c, float); ex = np.asarray(ex, float); ey = np.asarray(ey, float)
    M = cairo.Matrix(ex[0], ex[1], ey[0], ey[1], c[0], c[1])
    T = lambda P: np.asarray(P, float) @ np.stack([ex, ey]) + c
    R = float(np.hypot(*ex)); r = np.random.default_rng(seed); px = 1. / max(R, 1.)
    par = np.asarray(par, float)
    # --- depth 0: the well
    circ = np.stack([np.cos(np.linspace(0, 2 * np.pi, 120, endpoint=False)), np.sin(np.linspace(0, 2 * np.pi, 120, endpoint=False))], -1)
    cel.clip(T(circ))
    _rad(cel, M, None, 0, 1, [(0, pal['deep'], 1), (pupil * .9, pal['deep'], 1), (pupil + .10, pal['body'], 1), (.52, pal['body'], 1), (.70, pal['mid'], 1), (.86, pal['ring'], 1), (.955, pal['limb'], 1), (1, pal['limb'], 1)], centre=tuple(par * .5))
    # light sinks to the bottom of the iris (transmitted light), the top is in the lid's shade
    _rad(cel, M, None, 0, .98, [(0, pal['lite'], 0), (.35, pal['lite'], .0), (.62, pal['lite'], .85), (.86, pal['body'], .55), (1, pal['body'], 0)], centre=(0 + par[0] * .6, .30 + par[1] * .6))
    # --- depth 1: long fibres, two generations of shards (hard edged)
    nsh = int(16 + 26 * detail)
    def shard(a, w, r0, r1, col, al, k=0., bend=0., notch=False):
        o = par * k
        pts = [(math.cos(a - w * .2) * r0, math.sin(a - w * .2) * r0), (math.cos(a - w * .5 + bend * .4) * (r0 * .45 + r1 * .55), math.sin(a - w * .5 + bend * .4) * (r0 * .45 + r1 * .55)), (math.cos(a + bend) * r1, math.sin(a + bend) * r1),
               (math.cos(a + w * .5 + bend * .4) * (r0 * .55 + r1 * .45), math.sin(a + w * .5 + bend * .4) * (r0 * .55 + r1 * .45)), (math.cos(a + w * .2) * r0, math.sin(a + w * .2) * r0)]
        cel.fill(T(np.array(pts) + o), col, al)
    for gen in range(2):
        a0 = r.uniform(0, 6.28)
        for k in range(nsh):
            a = a0 + (k + r.uniform(-.35, .35)) * 2 * math.pi / nsh + spin * (.3 - gen * .5)
            up = math.sin(a) < 0
            w = r.uniform(.08, .24) * (1.3 if gen == 0 else .8) * 22 / nsh
            r1 = r.uniform(.66, .95); r0 = pupil * r.uniform(1.02, 1.25)
            col, al = ((pal['mid'], .65) if gen == 0 else (pal['lite'], .55)) if not up else ((pal['ring'], .70) if gen == 0 else (pal['body'], .60))
            if r.random() < .25: col = pal['core']; al = .55
            shard(a, w, r0, r1, col, al, k=.25 + .3 * gen, bend=r.normal(0, .05))
    # dark cracks from the limbal ring inward
    for k in range(int(10 + 16 * detail)):
        a = r.uniform(0, 6.28); shard(a, r.uniform(.03, .09), 1.0, r.uniform(.58, .85), pal['limb'], .55, k=.1, bend=r.normal(0, .06))
    # --- depth 2: broken concentric rings
    rings = [(.44, .020, pal['core'], .9), (.555, .014, pal['lite'], .8), (.60, .007, pal['core'], .7), (.70, .016, pal['lite'], .65), (.775, .008, pal['core'], .55), (.865, .012, pal['body'], .7)]
    if detail < .35: rings = [(.50, .035, pal['core'], .8), (.74, .03, pal['lite'], .6)]
    for j, (rr, th, col, al) in enumerate(rings):
        a = r.uniform(0, 6.28) + spin * (1.2 if j % 2 else -.8) * (1 - rr * .5)
        end = a + 2 * math.pi
        while a < end - .12:
            ln = r.uniform(.18, 1.05); gap = r.uniform(.06, .42)
            b = min(a + ln, end - .05)
            q = np.linspace(a, b, 18); jr = rr + r.normal(0, .010); tw = np.sin(np.linspace(0, np.pi, 18)) ** .45 * th * r.uniform(.6, 1.4) * .5
            P = np.vstack([np.c_[np.cos(q) * (jr + tw), np.sin(q) * (jr + tw)], np.c_[np.cos(q) * (jr - tw), np.sin(q) * (jr - tw)][::-1]])
            cel.fill(T(P + par * (.35 + .5 * (1 - rr))), col, al * (.55 + .45 * max(0, math.sin((a + b) / 2) + .5)))
            a = b + gap
    # --- depth 3: cloud flecks drifting in the lower half, like a sky
    if detail > .3:
        for k in range(int(5 + 6 * detail)):
            a = r.uniform(.15, math.pi - .15) if k % 4 else r.uniform(math.pi, 2 * math.pi); rr = r.uniform(.50, .80); a += spin * .25
            p = np.array([math.cos(a) * rr, math.sin(a) * rr]) + par * .45; tang = np.array([-math.sin(a), math.cos(a)]); rad = np.array([math.cos(a), math.sin(a)])
            L = r.uniform(.07, .17); n = r.integers(4, 7); blobs = []
            for q in range(n):
                u = (q / (n - 1) - .5) * 2; h = L * .42 * (1 - u * u * .7) * r.uniform(.8, 1.2); cc = p + tang * u * L + rad * -h * .25
                th_ = np.linspace(0, 2 * np.pi, 18, endpoint=False); blobs.append(cc + np.c_[np.cos(th_), np.sin(th_)] * h)
            for b in blobs: cel.fill(T(b + rad * L * .10), pal['cloud_sh'], .75)
            for b in blobs: cel.fill(T((b - p) * .86 + p - rad * L * .05), pal['cloud'], .88)
    # --- pupil: a small dark star with a lit rim
    n = 72; th_ = np.linspace(0, 2 * np.pi, n, endpoint=False)
    jag = 1 + (.07 * detail + .02) * np.abs(np.sin(th_ * 7 + 1.3)) ** 3 + .03 * np.sin(th_ * 3 + seed)
    pc = par * .9
    cel.fill(T(np.c_[np.cos(th_), np.sin(th_)] * (pupil * 1.20 * jag)[:, None] + pc), pal['core'], .85)
    cel.fill(T(np.c_[np.cos(th_), np.sin(th_)] * (pupil * 1.06 * jag)[:, None] + pc), pal['pup2'])
    cel.fill(T(np.c_[np.cos(th_), np.sin(th_)] * (pupil * .84) + pc), pal['pupil'])
    if detail > .3:
        for k in range(9):
            a = r.uniform(0, 6.28); shard(a, r.uniform(.12, .3), pupil * .30, pupil * r.uniform(.7, 1.0), pal['mid'], .45, k=.9)
        cel.fill(T(np.c_[np.cos(th_), np.sin(th_)] * (pupil * .34) + pc), pal['lite'], .35 + .5 * glow)
    if glow > 0:
        _rad(cel, M, None, 0, 1, [(0, '#ffffff', .95 * glow), (pupil * .7, '#e6fbff', .75 * glow), (pupil + .25, '#b8f0ff', .45 * glow), (.8, '#6fd0ff', .15 * glow), (1, '#6fd0ff', 0)], centre=tuple(pc))
    # --- lid shade + limbal edge
    if shade_top > 0: _lin(cel, M, None, (0, -1), (0, .20), [(0, '#0a1a50', .92 * shade_top), (.45, '#123a8a', .55 * shade_top), (1, '#1f62b4', 0)])
    cel.unclip()
    q = np.linspace(0, 2 * np.pi, 160)
    cel.line(T(np.c_[np.cos(q), np.sin(q)] * .985), max(1.2, R * .045), pal['limb'], taper=(0, 0), press=.5, seed=seed, tremor=.0)


def iris_hi(cel, c, ex, ey, seed=3, big=1., side=-1, col='#ffffff', detail=1.):
    """hard wet highlights on top of everything inside the eye"""
    c = np.asarray(c, float); T = lambda P: np.asarray(P, float) @ np.stack([np.asarray(ex, float), np.asarray(ey, float)]) + c
    sx = side
    a = np.array([(.30, -.62), (.52, -.50), (.60, -.22), (.50, -.10), (.36, -.20), (.24, -.40)]) * (sx, 1)
    cel.fill(T(spl(a, True, corners=(0, 2, 3), n=6)), col)
    if detail > .3:
        cel.fill(T(spl(np.array([(.66, -.36), (.72, -.22), (.70, -.06), (.64, -.18)]) * (sx, 1), True, corners=(0, 2), n=5)), col, .85)
        cel.fill(T(spl(np.array([(.16, -.70), (.26, -.66), (.20, -.52), (.12, -.60)]) * (sx, 1), True, corners=(0,), n=5)), col, .8)
    b = np.array([(-.34, .36), (-.22, .30), (-.14, .38), (-.22, .50), (-.33, .48)]) * (sx, 1)
    cel.fill(T(spl(b, True, corners=(0, 2), n=5)), col, .92)
    q = np.linspace(.18 * math.pi, .70 * math.pi, 24); w = np.sin(np.linspace(0, math.pi, 24)) * .05
    cr = np.vstack([np.c_[np.cos(q), np.sin(q)] * .86, (np.c_[np.cos(q), np.sin(q)] * (.86 - w)[:, None])[::-1]])
    cel.fill(T(cr), col, .55)
