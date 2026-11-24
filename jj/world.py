"""Painted anime backgrounds.  A set is BUILT in metres (Scene: walls, slabs, rubble, signs, figures), seen through
cam.Camera3D with a real z-buffer, lit as designed planes, then gone over with gouache strokes masked to each face.

USE A SET (all painters live in world_*.py and are reachable here):
    from jj import world
    P = world.plate('scramble', w=2400, h=1350, time='night')    # paints once (<=25 s), then loads from cache/
    P.img    float32 (h,w,3), may exceed 1 where lamps burn      P.z     metres from camera (sky = 1e6)
    P.mask   dict of float masks ('sky', 'ground', ...)           P.layers dict of premult RGBA foregrounds
    P.cam    Camera3D of the plate (project your characters with it so feet stand on the ground)
    world.move(P.img, cx, cy, zoom)                               stand camera -> (1080,1920,3)
    world.parallax(P, dx, dy, zoom)                               cheap 2.5-D move from P.z (3 depth slabs)
`world.SETS` lists every set with its parameters.
"""
import numpy as np, cv2, math, os, hashlib, time as _time
import numba
from . import W, H
from .paint import rgb, f32, fbm, lum, smooth, mix, grid, Canvas, arcs, jit_color, repaint, mottle, _vn, _h2, ramp, warp_by
from .cam import Camera3D, NEAR
from .comp import blur, screen, over, cam2d

try: numba.set_num_threads(min(4, numba.get_num_threads()))
except Exception: pass
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FAR = 1e6


def C(c):
    return rgb(c) if isinstance(c, str) else np.asarray(c, f32)


def hexes(*c):
    return np.stack([C(v) for v in c])


from .world_nb import _raster, _shade, _tex, _cellnoise


# ------------------------------------------------------------------------------------------------ scene
class Scene:
    """geometry list.  every add-method returns the label(s) used."""

    def __init__(s, seed=0):
        s.r = np.random.default_rng(seed)
        s.P, s.Cc, s.Aa, s.Dd, s.Ll, s.B = [], [], [], [], [], []
        s.lights, s.ldir = [], []
        s.segs = []
        s.nlab = 1
        s.groups = 8

    # -- labels
    def newlab(s, n=1):
        a = s.nlab; s.nlab += n
        return a if n == 1 else np.arange(a, a + n)

    # -- core: P (n,4,3)
    def quads(s, P, col, em=0., grp=0, tex=0, lab=None, d=None, snow=1., bias=0., nofog=0., seed=None):
        P = np.asarray(P, f32).reshape(-1, 4, 3); n = len(P)
        if n == 0: return 0
        c = np.asarray(col, f32) if not isinstance(col, str) else rgb(col)
        if c.ndim == 1: c = np.broadcast_to(c, (n, 4, 3))
        elif c.ndim == 2: c = np.broadcast_to(c[:, None] if c.shape[0] == n else c[None], (n, 4, 3))
        else: c = np.broadcast_to(c, (n, 4, 3))
        A = np.zeros((n, 8), f32)
        A[:, 0] = em; A[:, 1] = grp; A[:, 2] = tex; A[:, 3] = snow; A[:, 4] = s.r.integers(1, 9999) if seed is None else seed; A[:, 6] = nofog
        if d is None:
            e1 = P[:, 1] - P[:, 0]; e2 = P[:, 3] - P[:, 0]
            l1 = np.linalg.norm(e1, axis=1, keepdims=True); l2 = np.linalg.norm(e2, axis=1, keepdims=True)
            D = np.where(l1 > l2, e1 / (l1 + 1e-9), e2 / (l2 + 1e-9))
        else: D = np.broadcast_to(np.asarray(d, f32), (n, 3))
        if lab is None: lab = s.newlab()
        s.P.append(P); s.Cc.append(np.array(c, f32)); s.Aa.append(A); s.Dd.append(np.array(D, f32))
        s.Ll.append(np.broadcast_to(np.asarray(lab, np.int32), (n,)).copy()); s.B.append(np.broadcast_to(np.asarray(bias, f32), (n,)).copy())
        return lab

    def poly(s, P, col, **k):
        """convex polygon fan"""
        P = np.asarray(P, float); n = len(P)
        Q = [(P[0], P[i], P[i + 1], P[i + 1]) for i in range(1, n - 1)]
        return s.quads(np.array(Q), col, **k)

    def tris(s, T, col, **k):
        T = np.asarray(T, float).reshape(-1, 3, 3)
        c = col
        if not isinstance(col, str):
            c = np.asarray(col, f32)
            if c.ndim == 3: c = np.concatenate([c, c[:, 2:3]], 1)
        return s.quads(np.concatenate([T, T[:, 2:3]], 1), c, **k)

    def ground(s, x0, x1, y0, y1, z, col, nx=12, ny=40, pw=1.0, **k):
        """subdivided ground sheet; col may be f(x,y)->(...,3)"""
        xs = np.linspace(x0, x1, nx + 1); t = np.linspace(0, 1, ny + 1) ** pw; ys = y0 + (y1 - y0) * t
        Xg, Yg = np.meshgrid(xs, ys); Zg = z(Xg, Yg) if callable(z) else np.full_like(Xg, z)
        V = np.stack([Xg, Yg, Zg], -1)
        P = np.stack([V[:-1, :-1], V[:-1, 1:], V[1:, 1:], V[1:, :-1]], 2).reshape(-1, 4, 3)
        if callable(col):
            cc = col(Xg, Yg); c = np.stack([cc[:-1, :-1], cc[:-1, 1:], cc[1:, 1:], cc[1:, :-1]], 2).reshape(-1, 4, 3)
        else: c = col
        k.setdefault('d', (1, 0, 0))
        return s.quads(P, c, **k)

    def boxes(s, c, size, rot=0., col='#888888', top=None, side=None, faces='xXyYzZ', axes=None, lab=None, **k):
        """n boxes: centres (n,3), full sizes (n,3), z-rotation (n,) or axes (n,3,3) rows = local axes."""
        c = np.atleast_2d(np.asarray(c, float)); n = len(c)
        size = np.broadcast_to(np.asarray(size, float), (n, 3)); rot = np.broadcast_to(np.asarray(rot, float), (n,))
        if axes is None:
            co, si = np.cos(rot), np.sin(rot); z0 = np.zeros(n)
            axes = np.stack([np.stack([co, si, z0], -1), np.stack([-si, co, z0], -1), np.stack([z0, z0, z0 + 1], -1)], 1)
        axes = np.broadcast_to(np.asarray(axes, float), (n, 3, 3))
        h = axes * (size / 2)[:, :, None]
        ex, ey, ez = h[:, 0], h[:, 1], h[:, 2]
        F = {'x': (-ex, -ey, ez, (0, 1)), 'X': (ex, ey, ez, (0, 1)), 'y': (-ey, ex, ez, 0), 'Y': (ey, -ex, ez, 0), 'z': (-ez, ex, -ey, 0), 'Z': (ez, ex, ey, 0)}
        cc = np.asarray(rgb(col) if isinstance(col, str) else col, f32); cc = np.broadcast_to(cc if cc.ndim == 2 else cc[None], (n, 3))
        tc = cc if top is None else np.broadcast_to(np.atleast_2d(C(top)), (n, 3))
        L = s.newlab(n * 6) if lab is None else lab
        for j, ch in enumerate(faces):
            o, a, b, _ = F[ch]
            P = np.stack([c + o - a - b, c + o + a - b, c + o + a + b, c + o - a + b], 1)
            ll = (np.arange(n) * 6 + j + (L[0] if np.ndim(L) else 0)) if lab is None else lab
            s.quads(P, tc if ch == 'Z' else cc, lab=ll, **k)
        return L

    def prism(s, poly, z0, z1, col, top=None, base=None, **k):
        """extrude a CCW footprint (k,2)"""
        P = np.asarray(poly, float); n = len(P); Q = np.roll(P, -1, 0)
        z0a = np.broadcast_to(z0, (n,)); z1a = np.broadcast_to(z1, (n,))
        q = np.stack([np.c_[P, z0a], np.c_[Q, np.roll(z0a, -1)], np.c_[Q, np.roll(z1a, -1)], np.c_[P, z1a]], 1)
        L = s.newlab(n)
        s.quads(q, col, lab=L, **k)
        s.poly(np.c_[P, z1a], col if top is None else top, **k)

    def cyl(s, c, r, z0, z1, col, n=10, r1=None, cap=True, **k):
        a = np.linspace(0, 2 * np.pi, n, endpoint=False) + .3; r1 = r if r1 is None else r1
        A = np.c_[c[0] + np.cos(a) * r, c[1] + np.sin(a) * r, np.full(n, z0)]; B = np.c_[c[0] + np.cos(a) * r1, c[1] + np.sin(a) * r1, np.full(n, z1)]
        s.quads(np.stack([A, np.roll(A, -1, 0), np.roll(B, -1, 0), B], 1), col, d=(0, 0, 1), **k)
        if cap: s.poly(B, col, **k)

    def tube(s, path, rad, col, n=6, **k):
        """tapered tube along a 3-D polyline; rad scalar or per point"""
        P = np.asarray(path, float); m = len(P); rad = np.broadcast_to(rad, (m,))
        T = np.gradient(P, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
        up = np.tile([[.13, .31, .94]], (m, 1)); U = np.cross(T, up); U /= np.linalg.norm(U, axis=1, keepdims=True) + 1e-9; V = np.cross(U, T)
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        R = P[:, None] + (U[:, None] * np.cos(a)[None, :, None] + V[:, None] * np.sin(a)[None, :, None]) * rad[:, None, None]
        q = np.stack([R[:-1], np.roll(R[:-1], -1, 1), np.roll(R[1:], -1, 1), R[1:]], 2).reshape(-1, 4, 3)
        if not isinstance(col, str) and np.ndim(col) == 2:     # per point colour
            cc = np.asarray(col, f32); c4 = np.stack([np.repeat(cc[:-1, None], n, 1), np.repeat(cc[:-1, None], n, 1), np.repeat(cc[1:, None], n, 1), np.repeat(cc[1:, None], n, 1)], 2).reshape(-1, 4, 3)
            col = c4
        return s.quads(q, col, **k)

    def seg(s, A, B, width, col, em=0., grp=0, min_px=.9, lab=None, bias=4e-4, nofog=0.):
        """camera-facing lines (wires, edges, rebar, rails): width metres, never thinner than min_px"""
        A = np.atleast_2d(np.asarray(A, float)); B = np.atleast_2d(np.asarray(B, float)); n = len(A)
        c = np.broadcast_to(np.atleast_2d(C(col)), (n, 3))
        s.segs.append((A, B, np.broadcast_to(width, (n,)), c, np.broadcast_to(em, (n,)), grp, min_px, lab, bias, nofog))

    def pline(s, P, width, col, **k):
        P = np.asarray(P, float); s.seg(P[:-1], P[1:], width, col, **k)

    def wire(s, a, b, sag, width, col, n=12, **k):
        a, b = np.asarray(a, float), np.asarray(b, float); t = np.linspace(0, 1, n + 1)[:, None]
        P = a + (b - a) * t; P[:, 2] -= sag * 4 * (t * (1 - t))[:, 0]
        s.pline(P, width, col, **k)

    def light(s, p, col, rad, r0=None, amt=1., grp=0, dir=None, cos=.5, soft=.6):
        c = C(col) * amt
        s.lights.append((*p, rad, *c, grp)); dd = (0, 0, 0, -2.) if dir is None else (*(np.asarray(dir, float) / np.linalg.norm(dir)), cos)
        s.ldir.append((*dd, rad * .3 if r0 is None else r0, soft))

    def merge(s, o, at=None):
        """append another scene's geometry"""
        for a, b in (('P', 'P'), ('Cc', 'Cc'), ('Aa', 'Aa'), ('Dd', 'Dd'), ('B', 'B')): getattr(s, a).extend(getattr(o, b))
        s.Ll.extend([l + s.nlab for l in o.Ll]); s.nlab += o.nlab; s.lights += o.lights; s.ldir += o.ldir; s.segs += o.segs

    # ------------------------------------------------------------------------------------ bake + render
    def _bake(s, cam):
        P, Cc, A, D, L, B = [list(v) for v in (s.P, s.Cc, s.Aa, s.Dd, s.Ll, s.B)]
        nl = s.nlab
        for (a, b, wd, c, em, grp, mpx, lab, bias, nofog) in s.segs:
            za = cam.cam(a)[:, 2]; zb = cam.cam(b)[:, 2]
            keep = (za > NEAR) | (zb > NEAR)
            if not keep.any(): continue
            a, b, wd, c, em, za, zb = a[keep], b[keep], wd[keep], c[keep], em[keep], za[keep], zb[keep]
            t = b - a
            va = a - cam.eye; vb = b - cam.eye
            na = np.cross(t, va); na /= np.linalg.norm(na, axis=1, keepdims=True) + 1e-9
            nb = np.cross(t, vb); nb /= np.linalg.norm(nb, axis=1, keepdims=True) + 1e-9
            wa = np.maximum(wd, mpx * np.maximum(za, .1) / cam.f) / 2; wb = np.maximum(wd, mpx * np.maximum(zb, .1) / cam.f) / 2
            q = np.stack([a - na * wa[:, None], b - nb * wb[:, None], b + nb * wb[:, None], a + na * wa[:, None]], 1)
            n = len(q); P.append(q.astype(f32)); Cc.append(np.repeat(c[:, None], 4, 1).astype(f32))
            AA = np.zeros((n, 8), f32); AA[:, 0] = em; AA[:, 1] = grp; AA[:, 6] = nofog; A.append(AA)
            D.append((t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)).astype(f32))
            if lab is None: L.append(np.full(n, nl, np.int32)); nl += 1
            else: L.append(np.full(n, lab, np.int32))
            B.append(np.full(n, bias, f32))
        P, Cc, A, D, L, B = [np.concatenate(v) for v in (P, Cc, A, D, L, B)]
        e1 = P[:, 1] - P[:, 0]; e2 = P[:, 2] - P[:, 0]; N = np.cross(e1, e2); ln = np.linalg.norm(N, axis=1)
        bad = ln < 1e-10
        N[bad] = np.cross(P[bad, 2] - P[bad, 0], P[bad, 3] - P[bad, 0])
        N = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)
        return P, Cc, A, D, L, B, N.astype(f32)

    def render(s, cam, ss=2, amb=('#404050', '#202028'), fills=(), sun=None, fog=None, snow=None, groups=None, reflect=False,
               chunk=300, smres=3072, shadow_box=None, baked=None, street=None):
        """amb=(sky, ground) light; fills=[(dir, col)] soft directional; sun=(dir_to_sun, col) with hard cast shadows;
        fog=dict(col, col2, dir, dens, start, hf, floor, pw, up, keep); snow=(amount, col); groups = light-group gains;
        street=(col, height[, (x, y, radius, floor)]) light that pools low in the streets.
        -> dict(img premult, a, z, lab, ang)"""
        w, h = cam.w, cam.h
        P, Cc, A, D, L, B, N = s._bake(cam) if baked is None else baked
        eye = np.array(cam.eye, float); R = np.array(cam.R, float)
        if reflect:
            F = np.diag([1., 1., -1.]); eye = F @ eye; R = R @ F
            up_ = P[:, :, 2].max(1) > .30; P, Cc, A, D, L, B, N = [v[up_] for v in (P, Cc, A, D, L, B, N)]
        # triangles
        n = len(P)
        T = np.concatenate([P[:, [0, 1, 2]], P[:, [0, 2, 3]]]); TCc = np.concatenate([Cc[:, [0, 1, 2]], Cc[:, [0, 2, 3]]]); TQ = np.concatenate([np.arange(n), np.arange(n)])
        e = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
        k = e > 1e-9; T, TCc, TQ = T[k], TCc[k], TQ[k]
        c = ((T.astype(np.float64) - eye) @ R.T)
        zc = c[..., 2]
        near = .08
        ins = (zc > near).sum(1)
        ok = ins == 3; cr = (ins > 0) & (ins < 3)
        ce, cc_, cq = [], [], []
        for i in np.nonzero(cr)[0]:
            V = c[i]; K = TCc[i]; poly, pc = [], []
            for a in range(3):
                b = (a + 1) % 3
                if V[a, 2] > near: poly.append(V[a]); pc.append(K[a])
                if (V[a, 2] > near) != (V[b, 2] > near):
                    t = (near - V[a, 2]) / (V[b, 2] - V[a, 2]); poly.append(V[a] + (V[b] - V[a]) * t); pc.append(K[a] + (K[b] - K[a]) * t)
            for a in range(1, len(poly) - 1):
                ce.append((poly[0], poly[a], poly[a + 1])); cc_.append((pc[0], pc[a], pc[a + 1])); cq.append(TQ[i])
        c, TCc, TQ = c[ok], TCc[ok], TQ[ok]
        if ce:
            c = np.concatenate([c, np.array(ce)]); TCc = np.concatenate([TCc, np.array(cc_, f32)]); TQ = np.concatenate([TQ, np.array(cq)])
        iz = 1.0 / np.maximum(c[..., 2], near)
        sx = (c[..., 0] * iz * cam.f + cam.cx) * ss; sy = (c[..., 1] * iz * cam.f + cam.cy) * ss
        iz = iz * (1 + B[TQ][:, None])
        vis = (sx.max(1) > 0) & (sx.min(1) < w * ss) & (sy.max(1) > 0) & (sy.min(1) < h * ss)
        sx, sy, iz, TCc, TQ = sx[vis], sy[vis], iz[vis], TCc[vis], TQ[vis]
        # lights
        LP = np.array(s.lights, np.float64).reshape(-1, 8) if s.lights else np.zeros((0, 8)); LDs = np.array(s.ldir, np.float64).reshape(-1, 6) if s.ldir else np.zeros((0, 6))
        LPp = LP[:, :4].copy(); LC = LP[:, 4:8].copy()
        ambm = np.zeros((5, 3)); ambm[0] = C(amb[0]); ambm[1] = C(amb[1])
        if street: ambm[2] = C(street[0]); ambm[3, 0] = street[1]; ambm[3, 1:] = (street[2][2], street[2][3]) if len(street) > 2 else (0, 0); ambm[4, :2] = street[2][:2] if len(street) > 2 else 0
        fl = np.array([[*(np.asarray(d, float) / np.linalg.norm(d)), *C(cl)] for d, cl in fills], np.float64).reshape(-1, 6)
        sunv = np.zeros(8); SM = np.zeros((1, 1), f32); SMm = np.zeros((3, 4)); sr = 0
        if sun is not None:
            sd = np.asarray(sun[0], float); sd = sd / np.linalg.norm(sd); sunv[:3] = sd; sunv[3:6] = C(sun[1]) * (sun[2] if len(sun) > 2 else 1.)
            if smres:
                SM, SMm, span = s._shadow(P, B, sd, smres, shadow_box, cam); sr = smres
                sunv[6] = span / smres * 1.6; sunv[7] = span / smres * 1.2
        fg = np.zeros(8); fc = np.zeros((4, 3))
        if fog:
            fg[:] = [fog.get('dens', .004), fog.get('start', 0.), fog.get('hf', 0.), fog.get('floor', .0), fog.get('base', 0.), fog.get('pw', 2.), 1. if 'up' in fog else 0., fog.get('keep', .0)]
            fc[0] = C(fog['col']); fc[1] = C(fog.get('col2', fog['col'])); dv = np.asarray(fog.get('dir', (0, 1, 0)), float); fc[2] = dv / np.linalg.norm(dv); fc[3] = C(fog.get('up', fog['col']))
        sn = np.zeros(5)
        if snow: sn[0] = snow[0]; sn[1:4] = C(snow[1]); sn[4] = snow[2] if len(snow) > 2 else .25
        gr = np.ones(max(s.groups, 1)) if groups is None else np.asarray(list(groups) + [1.] * max(0, s.groups - len(groups)), float)
        out = np.zeros((h, w, 3), f32); cov = np.zeros((h, w), f32); zo = np.full((h, w), FAR, f32); lo = np.zeros((h, w), np.int32); ao = np.zeros((h, w), f32)
        for y0 in range(0, h, chunk):
            ch = min(chunk, h - y0)
            ID = np.full((ch * ss, w * ss), -1, np.int32); ZB = np.zeros((ch * ss, w * ss), f32); B1 = np.zeros((ch * ss, w * ss), f32); B2 = np.zeros((ch * ss, w * ss), f32)
            mk = (sy.max(1) >= y0 * ss) & (sy.min(1) <= (y0 + ch) * ss); ix = np.nonzero(mk)[0]
            if len(ix):
                _raster(sx[ix], sy[ix], iz[ix], w * ss, y0 * ss, (y0 + ch) * ss, ID, ZB, B1, B2, 16)
                _shade(ID, ZB, B1, B2, y0, ch, w, ss, eye, R, cam.f, cam.cx, cam.cy, TCc[ix].astype(f32), TQ[ix].astype(np.int64), N, D, A, L, ambm, fl, sunv, SMm, SM, sr,
                       LPp, LC, LDs, fg, fc, sn, 0, gr, out[y0:y0 + ch], cov[y0:y0 + ch], zo[y0:y0 + ch], lo[y0:y0 + ch], ao[y0:y0 + ch])
        return dict(img=out, a=cov, z=zo, lab=lo, ang=ao)

    def _shadow(s, P, B, sd, res, box, cam):
        up = np.array([0, 0, 1.]) if abs(sd[2]) < .95 else np.array([0, 1., 0])
        u = np.cross(up, sd); u /= np.linalg.norm(u); v = np.cross(sd, u)
        Q = P.reshape(-1, 3).astype(float)
        if box is not None:
            x0, x1, y0, y1, z1 = box; bb = np.array([(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (0, z1)], float)
        else: bb = Q
        uu = bb @ u; vv = bb @ v; span = max(uu.max() - uu.min(), vv.max() - vv.min()) * 1.02
        cu, cv_ = (uu.max() + uu.min()) / 2, (vv.max() + vv.min()) / 2
        k = res / span
        M = np.zeros((3, 4)); M[0, :3] = u * k; M[0, 3] = res / 2 - cu * k; M[1, :3] = v * k; M[1, 3] = res / 2 - cv_ * k; M[2, :3] = sd
        dep = Q @ sd; off = -dep.min() + 10; M[2, 3] = off
        q = P[:, [0, 1, 2, 0, 2, 3]].reshape(-1, 3, 3).astype(float)
        X = q @ M[0, :3] + M[0, 3]; Y = q @ M[1, :3] + M[1, 3]; Z = q @ M[2, :3] + M[2, 3]
        vis = (X.max(1) > 0) & (X.min(1) < res) & (Y.max(1) > 0) & (Y.min(1) < res)
        ID = np.full((res, res), -1, np.int32); ZB = np.zeros((res, res), f32); b1 = np.zeros((res, res), f32); b2 = np.zeros((res, res), f32)
        _raster(X[vis], Y[vis], Z[vis], res, 0, res, ID, ZB, b1, b2, 16)
        return ZB, M, span


class Wall:
    """local frame on a facade.  o = bottom-left seen from outside, ex = unit along the wall to the right, outward n = ex x ez"""

    def __init__(s, S, a, b, z0, z1, grp=0):
        a, b = np.asarray(a, float), np.asarray(b, float)
        s.S = S; d = b[:2] - a[:2]; s.w = float(np.hypot(*d)); s.h = float(z1 - z0); s.grp = grp
        s.o = np.array([a[0], a[1], z0]); s.ex = np.array([d[0] / s.w, d[1] / s.w, 0.]); s.ez = np.array([0, 0, 1.]); s.n = np.cross(s.ex, s.ez)

    @staticmethod
    def frame(S, o, ex, ez, w, h, grp=0):
        q = Wall.__new__(Wall); q.S = S; q.o = np.asarray(o, float); q.ex = np.asarray(ex, float) / np.linalg.norm(ex); q.ez = np.asarray(ez, float) / np.linalg.norm(ez); q.n = np.cross(q.ex, q.ez); q.w, q.h, q.grp = w, h, grp
        return q

    def p(s, u, v, d=0.):
        u, v, d = np.broadcast_arrays(*[np.asarray(t, float) for t in (u, v, d)])
        return s.o + u[..., None] * s.ex + v[..., None] * s.ez + d[..., None] * s.n

    def rect(s, u0, u1, v0, v1, d=0., col='#888888', ct=None, own=False, **k):
        """rectangles (broadcast arrays).  ct = colour at the top edge (gradient)"""
        u0, u1, v0, v1, d = [np.atleast_1d(t) for t in np.broadcast_arrays(*[np.asarray(t, float) for t in (u0, u1, v0, v1, d)])]
        n = len(u0)
        Q = np.stack([s.p(u0, v0, d), s.p(u1, v0, d), s.p(u1, v1, d), s.p(u0, v1, d)], 1)
        c = np.broadcast_to(np.atleast_2d(C(col)), (n, 3))
        if ct is not None:
            t = np.broadcast_to(np.atleast_2d(C(ct)), (n, 3)); c = np.stack([c, c, t, t], 1)
        if own: k['lab'] = np.atleast_1d(s.S.newlab(n))
        k.setdefault('grp', s.grp)
        return s.S.quads(Q, c if c.ndim == 3 else np.repeat(c[:, None], 4, 1), **k)

    def slab(s, u0, u1, v0, v1, d0, d1, col='#888888', side=None, top=None, under=None, **k):
        """boxes standing off the wall from depth d0 to d1"""
        a = [np.atleast_1d(t) for t in np.broadcast_arrays(*[np.asarray(t, float) for t in (u0, u1, v0, v1, d0, d1)])]
        u0, u1, v0, v1, d0, d1 = a; n = len(u0)
        cf = np.broadcast_to(np.atleast_2d(C(col)), (n, 3)); cs = cf * .72 if side is None else np.broadcast_to(np.atleast_2d(C(side)), (n, 3))
        ct = cs if top is None else np.broadcast_to(np.atleast_2d(C(top)), (n, 3)); cu = cs * .8 if under is None else np.broadcast_to(np.atleast_2d(C(under)), (n, 3))
        k.setdefault('grp', s.grp)
        L = s.S.newlab(max(2, n * 5)); em = k.pop('em', 0.); se = k.pop('side_em', em)
        def add(a_, b_, c_, d_, col_, j, e_): s.S.quads(np.stack([a_, b_, c_, d_], 1), col_, lab=L[0] + np.arange(n) * 5 + j, em=e_, **k)
        p = s.p
        add(p(u0, v0, d1), p(u1, v0, d1), p(u1, v1, d1), p(u0, v1, d1), cf, 0, em)
        add(p(u0, v0, d0), p(u0, v0, d1), p(u0, v1, d1), p(u0, v1, d0), cs, 1, se)
        add(p(u1, v0, d1), p(u1, v0, d0), p(u1, v1, d0), p(u1, v1, d1), cs, 2, se)
        add(p(u0, v1, d1), p(u1, v1, d1), p(u1, v1, d0), p(u0, v1, d0), ct, 3, se)
        add(p(u0, v0, d0), p(u1, v0, d0), p(u1, v0, d1), p(u0, v0, d1), cu, 4, se)
        return L

    def pocket(s, u0, u1, v0, v1, dep, col, **k):
        """reveal of a recessed opening"""
        a = [np.atleast_1d(t) for t in np.broadcast_arrays(*[np.asarray(t, float) for t in (u0, u1, v0, v1, dep)])]
        u0, u1, v0, v1, dep = a; p = s.p; c = np.broadcast_to(np.atleast_2d(C(col)), (len(u0), 3)); k.setdefault('grp', s.grp); z = np.zeros(len(u0))
        for q, kk in (((p(u0, v0, z), p(u0, v0, -dep), p(u0, v1, -dep), p(u0, v1, z)), .8), ((p(u1, v0, -dep), p(u1, v0, z), p(u1, v1, z), p(u1, v1, -dep)), 1.), ((p(u0, v1, -dep), p(u1, v1, -dep), p(u1, v1, z), p(u0, v1, z)), .6), ((p(u0, v0, z), p(u1, v0, z), p(u1, v0, -dep), p(u0, v0, -dep)), 1.15)):
            s.S.quads(np.stack(q, 1), c * kk, **k)

    def poly(s, uv, d, col, **k):
        uv = np.asarray(uv, float); k.setdefault('grp', s.grp)
        return s.S.poly(s.p(uv[:, 0], uv[:, 1], d), col, **k)

    def disc(s, u, v, r, d, col, n=20, ry=None, **k):
        a = np.linspace(0, 2 * np.pi, n, endpoint=False); return s.poly(np.c_[u + np.cos(a) * r, v + np.sin(a) * (r if ry is None else ry)], d, col, **k)

    def ring(s, u, v, r0, r1, d, col, n=28, a0=0., a1=2 * np.pi, **k):
        a = np.linspace(a0, a1, n + 1); k.setdefault('grp', s.grp)
        Q = np.stack([s.p(u + np.cos(a[:-1]) * r0, v + np.sin(a[:-1]) * r0, d), s.p(u + np.cos(a[:-1]) * r1, v + np.sin(a[:-1]) * r1, d), s.p(u + np.cos(a[1:]) * r1, v + np.sin(a[1:]) * r1, d), s.p(u + np.cos(a[1:]) * r0, v + np.sin(a[1:]) * r0, d)], 1)
        return s.S.quads(Q, col, **k)


# ------------------------------------------------------------------------------------------------ painterly passes
def occlusion(z, amt=.35, scales=(5, 18, 60), k=9.):
    """painted contact shade and depth separation from the depth map -> multiplier"""
    m = (z < FAR * .5).astype(f32); lz = np.where(m > 0, np.log(np.maximum(z, .05)), 0).astype(f32)
    ao = np.zeros_like(lz)
    for s in scales:
        q = max(1, int(s // 4)); hh, ww = z.shape
        a = cv2.resize(lz * m, (ww // q, hh // q), interpolation=cv2.INTER_AREA); b = cv2.resize(m, (ww // q, hh // q), interpolation=cv2.INTER_AREA)
        a = cv2.GaussianBlur(a, (0, 0), s / q) / (cv2.GaussianBlur(b, (0, 0), s / q) + 1e-3); a = cv2.resize(a, (ww, hh), interpolation=cv2.INTER_LINEAR)
        ao += np.clip((lz - a) * k / (1 + s * .05), 0, 1) * m
    return 1 - amt * np.clip(ao / len(scales) * 1.4, 0, 1)


def gouache(img, lab=None, ang=None, seed=1, sizes=(24, 11, 5), op=(.55, .55, .5), val=(.034, .028, .022), hue=.011, dry=.3, thr=(.018, .032, .055),
            density=1.0, mask=None, amt=1.0, aspect=4.2, wander=.14, bristle=.05):
    """brush the guide over with masked gouache strokes: big strokes only where the picture is quiet, small ones nearer detail"""
    img = np.asarray(img, f32); g = np.clip(img, 0, 1); ext = img - g
    h, w = g.shape[:2]
    if ang is None: ang = fbm(h, w, (220, 80), (1, .5), seed + 3) * .25
    L = lum(g)
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0) / 8; gy = cv2.Sobel(L, cv2.CV_32F, 0, 1) / 8; mag = np.abs(gx) + np.abs(gy)
    cv = Canvas(h, w, img=g); r = np.random.default_rng(seed + 100)
    for li, hw in enumerate(sizes):
        en = cv2.boxFilter(mag, -1, (int(hw * 2) | 1, int(hw * 2) | 1))
        step = hw * 1.55 / math.sqrt(density)
        gy_, gx_ = np.mgrid[step * .5:h:step, step * .5:w:step].astype(f32)
        x = (gx_ + r.uniform(-.6, .6, gx_.shape) * step).ravel().clip(0, w - 1); y = (gy_ + r.uniform(-.6, .6, gx_.shape) * step).ravel().clip(0, h - 1)
        ix, iy = x.astype(int), y.astype(int)
        keep = en[iy, ix] < thr[li] * r.uniform(.5, 1.5, x.size)
        if mask is not None: keep &= mask[iy, ix] > r.uniform(.05, .95, x.size)
        ix, iy, x, y = ix[keep], iy[keep], x[keep], y[keep]
        if x.size == 0: continue
        o = r.permutation(x.size); ix, iy, x, y = ix[o], iy[o], x[o], y[o]
        ref = cv2.GaussianBlur(g, (0, 0), max(.6, hw * .15))
        base = g[iy, ix] * .7 + ref[iy, ix] * .3
        col = jit_color(base, r, hue, .03, val[li])
        lm = lum(base); col = base + (col - base) * np.clip(.35 + lm * 1.6, .35, 1)[:, None]     # quieter in the darks
        P = arcs(x, y, ang[iy, ix] + r.normal(0, wander, x.size), hw * aspect * r.uniform(.55, 1.3, x.size), r.normal(0, .35, x.size))
        cv.strokes(P, hw * r.uniform(.55, 1.15, x.size), col, op=op[li] * r.uniform(.6, 1.1, x.size) * amt, dry=dry * r.uniform(0, 2.2, x.size), seed=seed + li * 31, lab=lab, LAB=None if lab is None else lab[iy, ix], bristle=bristle)
    return cv.img + ext


def lamp_glow(img, emis, amts=((4, .35), (16, .28), (60, .22)), tintc=None):
    """painted light bleed of an emission map"""
    o = img
    for s, a in amts:
        g = blur(np.minimum(emis, 2.2), s)
        if tintc is not None: g = g * C(tintc)
        o = o + g * a * (1 - np.clip(o, 0, 1) * .6)
    return o


def air_glow(img, z, thr=.5, sig=(60, 150), amt=.35, zr=(20, 160), tintc=None):
    """light scattered in damp air: wide bleed that grows with distance"""
    e = np.clip(img - thr, 0, 1.2); d = smooth(zr[0], zr[1], np.minimum(z, 1e4))[..., None]
    g = sum(blur(e, s) for s in sig) / len(sig) * 2.2
    if tintc is not None: g = g * C(tintc)
    return img + g * amt * (.35 + .65 * d) * (1 - np.clip(img, 0, 1) * .5)


def wobble(img, amp=.7, scale=34, seed=0):
    """hand-cut edges"""
    return warp_by(img, amp, scale, seed)


def tooth_pass(img, seed=5, amt=.010, mot=.014):
    h, w = img.shape[:2]
    n = fbm(h, w, (2.6, 1.4), (1, .6), seed); m = fbm(h, w, (90, 34, 12), (1, .7, .4), seed + 1)
    return img * (1 + amt * n + mot * m)[..., None]


def comp_sky(res, sky):
    return res['img'] + sky * (1 - res['a'][..., None])


def wet_reflect(scene, cam, res, sky, wetmask, baked=None, strength=.8, vblur=16, ripple=2.4, seed=0, **rk):
    """mirror the scene in z=0 and lay it into the ground where wetmask>0, smeared like wet asphalt.  returns reflection rgb"""
    rr = scene.render(cam, ss=1, reflect=True, baked=baked, **rk)
    R = rr['img'] + sky_mirror(sky, cam) * (1 - rr['a'][..., None])
    h, w = R.shape[:2]
    q = cv2.resize(R, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    a = cv2.GaussianBlur(q, (0, 0), sigmaX=.9, sigmaY=vblur * .35 / 2); b = cv2.GaussianBlur(q, (0, 0), sigmaX=2.6, sigmaY=vblur * 1.6)
    R2 = cv2.resize(a * .62 + b * .5, (w, h), interpolation=cv2.INTER_LINEAR)
    x, y = grid(h, w); nx = fbm(h, w, (14, 5), (1, .6), seed + 2, aspect=3.5) * ripple
    R2 = cv2.remap(R2, x + nx, y + np.abs(nx) * .6, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    k = (wetmask * strength)[..., None]
    return res['img'] * (1 - k * .55) + R2 * k, R2


def sky_mirror(sky, cam):
    """approximate mirrored sky: flip about the horizon line"""
    h, w = sky.shape[:2]
    fwd = cam.R[2]; hy = cam.cy + cam.f * math.tan(math.asin(np.clip(fwd[2], -1, 1))) if abs(fwd[2]) < .99 else cam.cy
    x, y = grid(h, w)
    return cv2.remap(sky.astype(f32), x, np.clip(2 * hy - y, 0, h - 1), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def horizon_y(cam):
    fwd = cam.R[2]
    return float(cam.cy + cam.f * fwd[2] / math.sqrt(max(1e-6, 1 - fwd[2] ** 2)))


# ------------------------------------------------------------------------------------------------ plates + cache
class Plate:
    def __init__(s, img, z=None, cam=None, mask=None, layers=None, extra=None):
        s.img = img; s.z = z; s.cam = cam; s.mask = mask or {}; s.layers = layers or {}; s.extra = extra or {}

    def save(s, path):
        d = {'img': s.img.astype(np.float16)}
        if s.z is not None: d['z'] = np.minimum(s.z, 6e4).astype(np.float16)
        for k, v in s.mask.items(): d['m_' + k] = (np.clip(v, 0, 1) * 255).astype(np.uint8)
        for k, v in s.layers.items(): d['l_' + k] = v.astype(np.float16)
        for k, v in s.extra.items(): d['x_' + k] = np.asarray(v)
        if s.cam is not None: d['cam'] = np.r_[s.cam.eye, s.cam.R.ravel(), s.cam.f, s.cam.cx, s.cam.cy, s.cam.w, s.cam.h]
        tmp = path + '.%d.tmp.npz' % os.getpid(); np.savez(tmp, **d); os.replace(tmp, path)

    @staticmethod
    def load(path):
        d = np.load(path, allow_pickle=False); p = Plate(d['img'].astype(f32))
        for k in d.files:
            if k == 'z': z = d['z'].astype(f32); p.z = np.where(z > 5.5e4, FAR, z)
            elif k.startswith('m_'): p.mask[k[2:]] = d[k].astype(f32) / 255
            elif k.startswith('l_'): p.layers[k[2:]] = d[k].astype(f32)
            elif k.startswith('x_'): p.extra[k[2:]] = d[k]
            elif k == 'cam':
                c = d['cam']; cam = Camera3D.__new__(Camera3D); cam.eye = c[:3].copy(); cam.R = c[3:12].reshape(3, 3).copy(); cam.f, cam.cx, cam.cy = c[12:15]; cam.w, cam.h = int(c[15]), int(c[16]); cam.K = np.array([[cam.f, 0, cam.cx], [0, cam.f, cam.cy], [0, 0, 1.]]); p.cam = cam
        return p


SETS = {}


def painter(name, ver=1):
    """decorator: register a set painter"""
    def deco(fn):
        SETS[name] = (fn, ver); return fn
    return deco


def _load_all():
    import importlib
    for m in ('world_sky', 'world_city', 'world_ruin', 'world_in', 'world_hill', 'world_town', 'world_beach', 'world_shrine'):
        if os.path.exists(os.path.join(os.path.dirname(__file__), m + '.py')): importlib.import_module('jj.' + m)


def plate(name, fresh=False, **kw):
    """paint (or load) a set.  See SETS / the set's docstring for parameters."""
    if name not in SETS: _load_all()
    fn, ver = SETS[name]
    key = hashlib.md5(repr((ver, sorted(kw.items(), key=lambda t: t[0]))).encode()).hexdigest()[:10]
    path = os.path.join(ROOT, 'cache', f'world_{name}_{key}.npz')
    if os.path.exists(path) and not fresh:
        try: return Plate.load(path)
        except Exception: pass
    t = _time.time(); p = fn(**kw)
    if os.environ.get('WORLD_VERBOSE'): print('painted %s %.1fs' % (name, _time.time() - t))
    p.save(path)
    return p


class Lazy:
    """low-memory view of a plate: float16 .npy files opened with mmap.  L.a('img'), L.a('z'), L.a('l_rail'), L.a('m_sky'), L.a('x_wl') -> mmapped arrays, L.f(key, slice...) -> float32 copy, L.cam"""

    def __init__(s, d):
        s.d = d; s.keys = sorted(f[:-4] for f in os.listdir(d) if f.endswith('.npy')); c = np.load(os.path.join(d, 'cam.npy')) if 'cam' in s.keys else None
        if c is not None: cam = Camera3D.__new__(Camera3D); cam.eye = c[:3].copy(); cam.R = c[3:12].reshape(3, 3).copy(); cam.f, cam.cx, cam.cy = c[12:15]; cam.w, cam.h = int(c[15]), int(c[16]); cam.K = np.array([[cam.f, 0, cam.cx], [0, cam.f, cam.cy], [0, 0, 1.]]); s.cam = cam

    def a(s, k): return np.load(os.path.join(s.d, k + '.npy'), mmap_mode='r')

    def f(s, k, sl=Ellipsis):
        a = np.asarray(s.a(k)[sl]); return a.astype(f32) / 255 if a.dtype == np.uint8 else a.astype(f32)


def lazy(name, drop=True, **kw):
    """like plate(), but returns a Lazy (mmapped) plate.  drop=True removes the .npz after unpacking it"""
    if name not in SETS: _load_all()
    fn, ver = SETS[name]; key = hashlib.md5(repr((ver, sorted(kw.items(), key=lambda t: t[0]))).encode()).hexdigest()[:10]; base = os.path.join(ROOT, 'cache', f'world_{name}_{key}'); d = base + '_d'
    if not os.path.exists(os.path.join(d, 'ok')):
        if not os.path.exists(base + '.npz'): fn_p = fn(**kw); fn_p.save(base + '.npz'); del fn_p
        tmp = d + '.%d' % os.getpid(); os.makedirs(tmp, exist_ok=True); z = np.load(base + '.npz', allow_pickle=False)
        for k in z.files: np.save(os.path.join(tmp, k + '.npy'), z[k])
        z.close(); open(os.path.join(tmp, 'ok'), 'w').close()
        try: os.rename(tmp, d)
        except OSError: pass
        if drop and os.path.exists(base + '.npz'): os.remove(base + '.npz')
    return Lazy(d)


def save_pack(path, items):
    """items = [(roi float RGBA, ox, oy) | None] -> one flat float16 npy + index, mmapped by Pack"""
    idx, data, n = [], [], 0
    for it in items:
        if it is None: idx.append((0, 0, 0, 0, 0)); continue
        r, ox, oy = it; r = np.asarray(r, np.float16); idx.append((n, r.shape[0], r.shape[1], int(ox), int(oy))); data.append(r.ravel()); n += r.size
    np.save(path + '.tmp.npy', np.concatenate(data) if data else np.zeros(1, np.float16)); os.replace(path + '.tmp.npy', path + '.npy'); np.save(path + '.idx.npy', np.array(idx, np.int64))


class Pack:
    def __init__(s, path): s.p = np.load(path + '.npy', mmap_mode='r'); s.i = np.load(path + '.idx.npy')

    def __len__(s): return len(s.i)

    def get(s, k):
        """(roi float32 RGBA, ox, oy) or None"""
        n, h, w, ox, oy = [int(v) for v in s.i[k]]
        return None if h == 0 else (np.asarray(s.p[n:n + h * w * 4]).astype(f32).reshape(h, w, 4), ox, oy)


def move(img, cx, cy, zoom=1., rot=0., w=W, h=H):
    """stand camera over a plate (cx,cy in plate px at frame centre; zoom = screen px per plate px)"""
    return cam2d(img, cx, cy, zoom, rot, w, h)


def parallax(p, dx=0., dy=0., zoom=1., cx=None, cy=None, zref=None, w=W, h=H):
    """2.5-D move: every pixel shifts by (dx,dy)*zref/z px, zoom about (cx,cy).  Smooth, one remap; good for moves < ~60 px."""
    H_, W_ = p.img.shape[:2]; cx = W_ / 2 if cx is None else cx; cy = H_ / 2 if cy is None else cy
    z = np.minimum(p.z, 5e3); zref = float(np.median(z)) if zref is None else zref
    if 'pz' not in p.extra: p.extra['pz'] = cv2.GaussianBlur(cv2.erode(z, np.ones((9, 9), np.uint8)), (0, 0), 3)
    y, x = np.mgrid[0:h, 0:w].astype(f32)
    X = (x - w / 2) / zoom + cx; Y = (y - h / 2) / zoom + cy
    Z = cv2.remap(p.extra['pz'], X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    k = zref / np.maximum(Z, 1e-3)
    return cv2.remap(p.img, X - dx * k, Y - dy * k, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
