"""MAHORAGA: the divine general with the eight-handled wheel.

    from chars.mahoraga import Maho, KEYS, draw, burn_wheel, notch, plate_night, comp_keys
    m = Maho(KEYS['stride'], pos=(0,6,0), heading=.3)          construction in metres (Camera3D world), 3.2 m tall, faces -Y at heading 0
    d = draw(m, cam, lights=[wheel_light(m)], seed=3)          -> dict(layer premult RGBA, z, alpha, V)   the finished drawing
    add, veil = burn_wheel(m.wheel, cam, seed=k, prog=..., ignite=..., I=1.)   -> additive light (H,W,3), soft veil mask
The sculpt (maho_body, maho_sdf) only supplies perspective, planes and cast shadow; everything on screen is dab-textured value planes,
broken brush ink, hatching and scribbled light made here and in maho_ink."""
import numpy as np, cv2, math
from jj import W, H
from jj.paint import rgb, f32, fbm, lum, smooth, ramp, grid, Canvas, arcs
from jj.comp import over, screen, blur, radial, ease, ease_in, ease_out
from jj.cam import Camera3D
from . import maho_body as B
from .maho_body import Maho, merge, lerp_pose, unit, V, SKIN, HORN, FEATH, RIDGE, TOOTH, MOUTH, CLOTH, WRAP, BLADE, CORD, WHEEL, NAIL
from .maho_body import P_TORSO, P_ARML, P_ARMR, P_HEAD, P_HORN, P_WING, P_TEETH, P_LEG, P_FOOT, P_BLADE, P_ORN, P_WHEEL, P_SASH
from .maho_sdf import gbuffer, shadow, ortho_shadow
from .maho_ink import Pen, trace, confident, curve, wash, dabs, _res

INK = '#040506'
#              skin horn  feath ridge tooth mouth cloth wrap blade cord wheel nail
ALB = np.array([.92, .95, 1.00, .95, 1.25, .10, .19, .52, .80, .07, .34, .10, .30, .17, .42], f32)          # .. concrete, asphalt, smoke
SLATE = [(0, '#030406'), (.02, '#06080b'), (.06, '#0d1117'), (.12, '#161c25'), (.22, '#232c38'), (.36, '#374353'), (.52, '#566577'), (.70, '#8c9bab'), (.86, '#c9d4de'), (1., '#f4f9ff')]
CHAR = [(0, '#040405'), (.02, '#080809'), (.06, '#101113'), (.12, '#1b1c1f'), (.22, '#2a2b30'), (.36, '#404248'), (.52, '#606369'), (.70, '#95989c'), (.86, '#d0d3d6'), (1., '#fbfcff')]
TH = np.array([.018, .042, .078, .125, .19, .28, .41, .58, .78]); LV = np.array([.006, .026, .056, .096, .152, .23, .34, .49, .69, .95])


# ------------------------------------------------------------------ light
def wheel_light(m, I=2.8, d0=.85, shadow=True, off=(0, 0, 0)):
    return dict(kind='point', pos=m.wheel['c'] + np.asarray(off, float), I=I, d0=d0 * m.sc, shadow=shadow)


def _light(g, m, lights, seed):
    """raw key light (float map), using plane-quantised normals so planes turn in steps, as painted"""
    n = g['n']; ok = g['id'] >= 0; s = g['scale']; h, w = g['h'], g['w']
    a = ok.astype(f32); sg = 2.4 * s
    nb = cv2.GaussianBlur(n * a[..., None], (0, 0), sg) / (cv2.GaussianBlur(a, (0, 0), sg)[..., None] + 1e-4)
    ws = wash(h, w, seed); w2 = np.roll(ws, (311, 127), (0, 1)); q = 2.3
    nq = np.round(nb * q + np.stack([ws, w2, np.roll(ws, 77, 0)], -1) * .20) / q
    nq /= np.linalg.norm(nq, axis=-1, keepdims=True) + 1e-6
    ndv = np.clip(-(n * g['v']).sum(-1), 0, 1); out = np.zeros((h, w), f32); glow = np.zeros((h, w), f32); cols = []
    for L in lights:
        if L['kind'] == 'point':
            d = np.asarray(L['pos'], f32) - g['p']; r = np.linalg.norm(d, axis=-1) + 1e-5; d = d / r[..., None]; fall = 1 / (1 + (r / L['d0']) ** 2)
            sh = shadow(g, m.S if 'S' not in L else L['S'], L['pos'], soft=L.get('soft', .035)) if L.get('shadow', True) else 1.
            if 'gobo' in L:                                                     # spokes and orbs of the wheel throw turning beams and shadows
                wh, amt = L['gobo']; dp = g['p'] - np.asarray(wh['c'], f32); ph = np.arctan2(dp @ wh['ey'].astype(f32), dp @ wh['ex'].astype(f32)) - wh['spin']; mk = L.get('mark'); b = .5 + .5 * np.cos(8 * ph); bm = 1 - amt * smooth(.35, .75, b)
                if mk is not None: q = (ph - mk * math.pi / 4 + math.pi) % (2 * math.pi) - math.pi; bm = bm * (1 - .45 * np.exp(-(q / .30) ** 2))
                sh = sh * bm.astype(f32)
        else:
            d = np.broadcast_to(-unit(L['dir']).astype(f32), n.shape); fall = 1.
            sh = ortho_shadow(g, m.S, L['dir'], m.J['chest'], 2.6 * m.sc) if L.get('shadow', True) else 1.
        if not np.isscalar(sh): sh = cv2.GaussianBlur(sh, (0, 0), 1.2 * s)
        nl = np.clip((n * d).sum(-1), -1, 1); nlq = np.clip((nq * d).sum(-1), -1, 1); wr = L.get('wrap', .22)
        lam = .42 * np.clip((nl + wr) / (1 + wr), 0, 1) + .58 * np.clip((nlq + wr) / (1 + wr), 0, 1)
        k = lam * (1 + L.get('rim', 1.8) * (1 - ndv) ** 2.2) * fall * sh * L['I']
        k = k + L.get('bounce', .085) * np.clip(.55 + .45 * (.5 * nl + .5 * nlq), 0, 1) ** 1.5 * fall * L['I'] * g['ao']
        k = k * (.40 + .60 * g['ao'])
        if 'col' in L: cols.append((k * ok, rgb(L['col']) if isinstance(L['col'], str) else np.asarray(L['col'], f32), L.get('mix', 'add')))
        else: out += k
    return out * ok, cols, ws, ndv


BONE = dict(levels=[(.075, .30), (.20, .64), (.40, 1.0)], ramp=[(0, '#0c0a0d'), (.30, '#584c4a'), (.64, '#bfb1a0'), (1., '#f1e8d5')], ink='#0b090c', step=.92)
CITY = dict(levels=[(.05, .26), (.14, .52), (.30, .78), (.55, 1.0)], ramp=[(0, '#0b0608'), (.26, '#2c080d'), (.52, '#6b1017'), (.78, '#b02228'), (1., '#e3a48c')], ink='#080406', step=.85)


def _tone(x, ws, seed, h, w, hard=.5, levels=None, step=.7):
    """value design: 1-exp tone curve, stepped with brush-edged, lost-and-found boundaries, over a faint airbrush"""
    v = 1 - np.exp(-1.35 * x)
    vv = v * (1 + .145 * ws) + .004 * ws
    lf = np.clip(.5 + .55 * fbm(h // 4 + 1, w // 4 + 1, (40, 14), (1, .5), seed + 3), 0, 1); lf = cv2.resize(lf, (w, h))
    if levels is not None:
        out = np.zeros((h, w), f32); p = 0.
        for t, q in levels: wd = t * (.05 + .45 * lf ** 2); out += (q - p) * smooth(t - wd, t + wd, vv); p = q
        return np.clip(step * out + (1 - step) * v, 0, 1) * (1 + .030 * ws), v
    out = np.full((h, w), LV[0], f32)
    for i, t in enumerate(TH):
        wd = t * (.05 + .55 * lf ** 2 * (1 - hard * .4)); out += (LV[i + 1] - LV[i]) * smooth(t - wd, t + wd, vv)
    return np.clip(.70 * out + .30 * v, 0, 1) * (1 + .055 * ws), v


# ------------------------------------------------------------------ edges
def _edges(g, S):
    ID = g['id']; h, w = ID.shape; s = g['scale']; P = g['p']; N = g['n']; Z = g['z']; ok = ID >= 0
    mat = np.r_[S['MAT'], -1][ID]; grp = np.r_[S['G'], -1][ID]; part = np.r_[S['PART'], -1][ID]
    out = np.zeros((h, w), np.uint8); inn = out.copy(); hard = out.copy(); depth = np.zeros((h, w), f32)
    for dx, dy in ((1, 0), (0, 1), (1, 1), (-1, 1)):
        def sh(a): return np.roll(a, (-dy, -dx), (0, 1))
        ok2 = sh(ok); Z2 = sh(Z); dP = sh(P) - P
        both = ok & ok2; pd1 = np.abs((dP * N).sum(-1)); pd2 = np.abs((dP * sh(N)).sum(-1)); pd = np.minimum(pd1, pd2)
        near1 = Z <= Z2
        sil = ok ^ ok2
        out |= (sil & ok).astype(np.uint8); out |= np.roll((sil & ok2).astype(np.uint8), (dy, dx), (0, 1))
        big = both & (pd > .055); mid = both & (pd > .0075 + .0016 * Z / 3) & ~big
        for m_, tgt in ((big, out), (mid, inn)):
            tgt |= (m_ & near1).astype(np.uint8); tgt |= np.roll((m_ & ~near1).astype(np.uint8), (dy, dx), (0, 1))
        id2 = sh(ID); m2 = sh(mat); g2 = sh(grp)
        hd = both & (ID != id2) & (np.isin(mat, (FEATH, TOOTH, NAIL)) & (mat == m2) | (grp != g2))
        hard |= (hd & near1).astype(np.uint8); hard |= np.roll((hd & ~near1).astype(np.uint8), (dy, dx), (0, 1))
    out[:2] = out[-2:] = 0; out[:, :2] = out[:, -2:] = 0
    inn &= ~out & 1; hard &= ~(out | inn) & 1
    # creases: concave turn of the normal over a few px
    a = ok.astype(f32); nb = cv2.GaussianBlur(N * a[..., None], (0, 0), 1.3 * s + .3) / (cv2.GaussianBlur(a, (0, 0), 1.3 * s + .3)[..., None] + 1e-4)
    k = max(1, int(round(2.5 * s))); cv_ = np.zeros((h, w), f32)
    for dx, dy in ((k, 0), (0, k), (k, k), (-k, k)):
        n1 = np.roll(nb, (dy, dx), (0, 1)); n2 = np.roll(nb, (-dy, -dx), (0, 1)); p1 = np.roll(P, (dy, dx), (0, 1)); p2 = np.roll(P, (-dy, -dx), (0, 1))
        dp = p2 - p1; L = np.linalg.norm(dp, axis=-1) + 1e-6; c = -((n2 - n1) * dp).sum(-1) / L; c = np.where(L < .16 * np.maximum(Z, .5) * 1., c, 0)
        cv_ = np.maximum(cv_, c)
    fld = cv2.dilate(out | inn | hard, np.ones((3, 3), np.uint8), iterations=max(1, int(2 * s)))
    cv_ = np.where(fld | ~ok, 0, cv_); cv_ = cv2.GaussianBlur(cv_, (0, 0), 1.0 * s + .2)
    return out, inn, hard, cv_, mat, part, grp


def _ridge(c, thr, s):
    """non-max suppression of a strength map -> thin binary"""
    gx = cv2.Sobel(c, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(c, cv2.CV_32F, 0, 1, ksize=3)
    r = np.zeros(c.shape, np.uint8); k = max(1, int(round(1.5 * s)))
    for dx, dy in ((k, 0), (0, k), (k, k), (-k, k)):
        a = np.roll(c, (dy, dx), (0, 1)); b = np.roll(c, (-dy, -dx), (0, 1)); r |= ((c > a) & (c >= b) & (c > thr) & ((c - .5 * (a + b)) > thr * .18)).astype(np.uint8)
    r = cv2.morphologyEx(r, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return r


# ------------------------------------------------------------------ pen work
def _samp(a, P):
    h, w = a.shape[:2]; x = np.clip(P[:, 0].round().astype(int), 0, w - 1); y = np.clip(P[:, 1].round().astype(int), 0, h - 1); return a[y, x]


def _pen_edges(pen, lines, w, r, scale, V_, lw, cls, brk=.5, over_=.45, dbl=.3, lo=.55, hi=1.35, seg=(70, 240), straight=False):
    """hand-ink a set of pixel chains: simplified, broken, overlapped, heavier in shadow"""
    cnt = 0
    for P in lines:
        for A in confident(P, eps=(2.2 if straight else 1.3) * max(scale, .6), sig=(1.0 if straight else 2.2) * max(scale, .6)):
            C = _res(A / scale, 3.) if straight else curve(A / scale, 3.)
            n = len(C)
            if n < 4: continue
            i = 0
            while i < n - 3:
                m = int(r.uniform(*seg) / 3.); j = min(n, i + m)
                if n - j < 8: j = n
                q = C[i:j]
                if len(q) >= 4:
                    v = float(np.mean(_samp(V_, q[::4]))); ww = w * lw * (hi + (lo - hi) * min(1., v * 1.6)) * r.uniform(.75, 1.25)
                    ext = r.uniform(0, 9) if r.random() < over_ else 0
                    if ext > 1 and len(q) > 4:
                        d = q[-1] - q[-4]; d = d / (np.hypot(*d) + 1e-6); q = np.vstack([q, q[-1] + d * ext])
                    pen.stroke(q, ww, r.integers(1 << 30), taper=(r.uniform(.08, .3), r.uniform(.15, .5)), press=.45, tremor=.55)
                    if r.random() < dbl and len(q) > 12:
                        a = int(r.uniform(0, .4) * len(q)); b = int(r.uniform(.55, 1) * len(q)); o = r.normal(0, 1) * 2.6 * lw + np.sign(r.normal()) * 1.4
                        T = np.gradient(q, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]
                        pen.stroke((q + N * o * np.linspace(.3, 1.2, len(q))[:, None])[a:b], ww * r.uniform(.35, .6), r.integers(1 << 30), taper=(.4, .5), tremor=.8)
                    cnt += 1
                i = j + (int(r.uniform(1, 5)) if r.random() < brk else -int(r.uniform(0, 3)))
                i = max(i, 1)
                if j >= n: break
    return cnt


def _cluster(pen, p, d, n, gap, L, w, r, ok=None, fan=.10):
    """hatching cluster: n strokes of length ~L along d, stacked along the normal from p"""
    nrm = np.array([-d[1], d[0]])
    for k in range(n):
        o = p + nrm * gap * (k - (n - 1) / 2) * r.uniform(.8, 1.2) + d * r.normal(0, L * .08); l = L * r.uniform(.5, 1.15) * (1 - .35 * abs(k - (n - 1) / 2) / max(1, n / 2)); a = fan * r.normal(); dd = np.array([d[0] * math.cos(a) - d[1] * math.sin(a), d[0] * math.sin(a) + d[1] * math.cos(a)])
        q = o + dd * l; mid = (o + q) / 2 + nrm * r.normal(0, l * .04)
        if ok is not None and (not ok(o) or not ok(q)): 
            q = o + dd * l * .5; mid = (o + q) / 2
            if not ok(o) or not ok(q): continue
        t = np.linspace(0, 1, 7)[:, None]; pen.stroke((1 - t) ** 2 * o + 2 * t * (1 - t) * mid + t * t * q, w * r.uniform(.7, 1.3), r.integers(1 << 30), taper=(.08, .65), press=.3, tremor=.3)


def draw(m, cam, lights=None, seed=0, scale=1.0, lw=1.0, hatch=1.0, fill=.012, fill_dir=(0, -.4, .6), hard=.6, teeth=.10, ink=True, dry=1.0, expo=1.0, extra_curves=True, straight=False, term=1.0, blade=.12, crease_thr=.25, pal=None, white=True):
    """the finished drawing of the creature.  lw line weight (1 = bust close-up at 1080p, ~.6 full body), fill = flat ambient so the dark body
    keeps a breath of form, teeth = extra value on teeth so the grin reads, hard = how hard the plane edges are."""
    lights = [wheel_light(m)] if lights is None else lights
    S = m.S; g = gbuffer(S, cam, scale); h, w = g['h'], g['w']; r = np.random.default_rng(seed + 17); ok = g['id'] >= 0
    x, cols, ws, ndv = _light(g, m, lights, seed)
    mat = np.r_[S['MAT'], -1][g['id']]; ab = ALB.copy()
    for k_, v_ in ((pal or {}).get('alb') or {}).items(): ab[k_] = v_
    alb = np.r_[ab, 0.][mat]
    fd = unit(fill_dir).astype(f32); amb = fill * (.35 + .65 * np.clip((g['n'] * fd).sum(-1), 0, 1)) * (.25 + .75 * g['ao'])
    part0 = np.r_[S['PART'], 0][g['id']]; pw = np.array([.50, .55, .55, 1., .8, .8, 1., .6, .6, 1., .8, 1., .8], f32)[part0]
    x = (x * expo + amb * pw) * alb
    x = np.where(mat == TOOTH, x + teeth * (.35 + .65 * np.clip((g['n'] * fd).sum(-1) * .5 + .5, 0, 1)) * g['ao'] ** 1.5, x)
    for L in lights:
        if L['kind'] == 'point' and teeth > 0:                                 # wet glint on the teeth
            d = np.asarray(L['pos'], f32) - g['p']; d /= np.linalg.norm(d, axis=-1, keepdims=True) + 1e-6; hv = d - g['v']; hv /= np.linalg.norm(hv, axis=-1, keepdims=True) + 1e-6
            x = np.where(mat == TOOTH, x + min(1., L['I']) * .9 * np.clip((g['n'] * hv).sum(-1), 0, 1) ** 24 * g['ao'], x)
    x = np.where(mat == BLADE, x * 1.2 + .5 * np.clip(x, 0, 1) ** 2 + blade * (.35 + .65 * np.clip((g['n'] * fd).sum(-1), 0, 1)) + blade * .8 * (1 - ndv) ** 2, x)
    Vt, Vs = _tone(x, np.where(mat == CLOTH, ws * .12, ws) if pal else ws, seed, h, w, hard, *((pal['levels'], pal['step']) if pal else ()))
    ink_c = rgb(pal['ink'] if pal else INK)
    # colour: slate against charcoal, drifting
    hue = np.clip(.5 + .45 * fbm(h // 6 + 1, w // 6 + 1, (30, 11), (1, .6), seed + 1), 0, 1); hue = cv2.resize(hue, (w, h))[..., None]
    col = ramp(Vt, pal['ramp']) * (.97 + .06 * hue) if pal else ramp(Vt, SLATE) * hue + ramp(Vt, CHAR) * (1 - hue)
    if True:
        wm = (mat == WHEEL)[..., None]; col = np.where(wm, col * np.array([1.12, 1.0, .80], f32), col)
        wm = (mat == WRAP)[..., None]; col = np.where(wm, col * np.array([1.06, 1.0, .93], f32), col)
    for k, c, mx in cols:
        v, _ = _tone(k * np.maximum(alb, .12), ws, seed + 5, h, w, hard); v = np.maximum(v - LV[0] * 1.2, 0)[..., None]
        if mx == 'shadow': col = col + v * c * (1 - smooth(.15, .8, Vt))[..., None]
        elif mx == 'over': v = np.clip(v * 1.25, 0, 1); col = col * (1 - v) + c * v
        else: col = col + v * c
    A = ok.astype(f32)
    # ---- to full resolution
    if scale != 1:
        a = A[..., None]; cb = cv2.GaussianBlur(col * a, (0, 0), 1.5) / (cv2.GaussianBlur(A, (0, 0), 1.5)[..., None] + 1e-4); col = np.where(a > 0, col, cb)
        col = cv2.resize(col.astype(f32), (W, H), interpolation=cv2.INTER_LINEAR); Vf = cv2.resize(Vt.astype(f32), (W, H)); aa = cv2.resize(cv2.GaussianBlur(A, (0, 0), .9), (W, H), interpolation=cv2.INTER_CUBIC)
    else:
        cb = cv2.GaussianBlur(col * A[..., None], (0, 0), 1.5) / (cv2.GaussianBlur(A, (0, 0), 1.5)[..., None] + 1e-4); col = np.where(A[..., None] > 0, col, cb); Vf = Vt; aa = cv2.GaussianBlur(A, (0, 0), 1.0)
    aa = smooth(.32, .68, aa).astype(f32)
    zf = cv2.resize(np.where(ok, g['z'], 1e3), (W, H), interpolation=cv2.INTER_NEAREST)
    if not ink: return dict(layer=np.dstack([col * aa[..., None], aa]), z=zf, alpha=aa, V=Vf, g=g)
    # ---- ink
    out, inn, hd, crease, mat, part, grp = _edges(g, S)
    crease = np.where(part == P_TEETH, 0, crease); pen = Pen(); V_ = cv2.GaussianBlur(Vf, (0, 0), 3); idf = cv2.resize(g['id'].astype(f32), (W, H), interpolation=cv2.INTER_NEAREST).astype(int)
    sc = scale
    _pen_edges(pen, trace(out, int(8 * sc) + 3), 4.6, r, sc, V_, lw, 'out', brk=.35, over_=.6, dbl=.55, lo=.55, hi=1.25, seg=(90, 320), straight=straight)
    L_in = trace(inn, int(7 * sc) + 3); _pen_edges(pen, L_in, 3.3, r, sc, V_, lw, 'in', brk=.45, dbl=.35, lo=.5, hi=1.3, seg=(60, 220), straight=straight)
    tooth = np.isin(mat, (TOOTH,)); L_h = trace(hd & ~tooth, int(6 * sc) + 3); _pen_edges(pen, L_h, 2.5, r, sc, V_, lw, 'hard', brk=.55, dbl=.28, lo=.6, hi=1.25, seg=(40, 160), straight=straight)
    _pen_edges(pen, trace(hd & tooth, 4), 2.1, r, sc, V_, lw, 'tooth', brk=.3, dbl=.10, lo=.8, hi=1.2, seg=(40, 120))
    rd = _ridge(crease, crease_thr, sc); L_c = trace(rd, int(11 * sc) + 4)
    L_c = [p for p in L_c if len(p) > 10]
    _pen_edges(pen, L_c, 2.5, r, sc, V_, lw, 'crease', brk=.6, dbl=.40, lo=.45, hi=1.25, seg=(50, 180), straight=straight)
    # ---- hatching clusters hanging off creases and overlaps, into the darker side
    if hatch > 0:
        pid = cv2.resize(np.where(ok, part, -1).astype(f32), (W, H), interpolation=cv2.INTER_NEAREST).astype(int); gxv = cv2.Sobel(V_, cv2.CV_32F, 1, 0, ksize=5); gyv = cv2.Sobel(V_, cv2.CV_32F, 0, 1, ksize=5)
        for lines, pr in ((L_c, .50), (L_in, .30), (L_h, .035)):
            for P in lines:
                P = P / sc; i = int(r.uniform(4, 20))
                while i < len(P) - 4:
                    if r.random() < pr * hatch:
                        t = P[min(i + 3, len(P) - 1)] - P[max(i - 3, 0)]; t = t / (np.hypot(*t) + 1e-6); nn = np.array([-t[1], t[0]]); p = P[i]
                        va, vb = float(_samp(V_, (p + nn * 13 * lw)[None])[0]), float(_samp(V_, (p - nn * 13 * lw)[None])[0]); sd = 1 if va < vb else -1
                        if max(va, vb) > .018:
                            pq = int(_samp(pid, (p + nn * sd * 6)[None])[0]); an = r.uniform(.75, 1.15) * (1 if r.random() < .8 else -1); d = nn * sd * math.sin(an) + t * math.cos(an)
                            okf = (lambda q, pq=pq: 0 <= q[0] < W and 0 <= q[1] < H and pid[int(q[1]), int(q[0])] == pq) if pq >= 0 else None
                            if pq != P_TEETH: _cluster(pen, p + t * r.normal(0, 4), d, int(r.integers(3, 8)), r.uniform(4., 6.5) * lw ** .7, r.uniform(13, 42) * lw, 1.9 * lw, r, okf)
                    i += int(r.uniform(16, 60) * lw ** .5 / 1.0)
        # terminator hatching: where a plane turns from light to dark
        gm = np.hypot(gxv, gyv); cand = (aa > .9) & (V_ > .035) & (V_ < .40) & (gm > np.percentile(gm[aa > .5], 72) if (aa > .5).any() else False)
        ys, xs = np.nonzero(cand[::3, ::3]); nc = int(len(ys) * .0045 * hatch * term / max(lw, .4))
        for k in (r.choice(len(ys), min(nc, len(ys)), replace=False) if len(ys) else ()):
            p = np.array([xs[k], ys[k]], float) * 3; y, x_ = int(p[1]), int(p[0]); d = -np.array([gxv[y, x_], gyv[y, x_]]); d = d / (np.hypot(*d) + 1e-6); pq = pid[y, x_]; a = r.normal(.35, .2); d = np.array([d[0] * math.cos(a) - d[1] * math.sin(a), d[0] * math.sin(a) + d[1] * math.cos(a)])
            if pq in (P_TEETH, P_ORN, P_WHEEL): continue
            _cluster(pen, p, d, int(r.integers(4, 9)), r.uniform(4.2, 6.8) * lw ** .7, r.uniform(18, 52) * lw, 1.6 * lw, r, lambda q, pq=pq: 0 <= q[0] < W and 0 <= q[1] < H and pid[int(q[1]), int(q[0])] == pq)
    wpen = Pen()
    if extra_curves: _form_lines(m, cam, pen, wpen, zf, V_, r, lw)
    K = pen.raster(H, W)
    # ---- dry brush: loaded black dragged along the deepest overlaps
    dl = np.zeros((H, W), f32)
    if dry > 0:
        cv = Canvas(H, W, color=INK, layer=True); PP, WD = [], []
        for P in L_in + L_c + trace(out, 30):
            P = P / sc
            if len(P) * 1. < 26 * lw: continue
            if r.random() > .55 * dry: continue
            a = int(r.uniform(0, .5) * len(P)); b = min(len(P) - 1, a + int(r.uniform(26, 120) * lw)); q = P[a:b:2]
            if len(q) < 6: continue
            v = float(np.mean(_samp(V_, q)))
            if v > .34: continue
            T = np.gradient(q, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]
            va = float(np.mean(_samp(V_, q + N * 10))); vb = float(np.mean(_samp(V_, q - N * 10))); sd = 1 if va < vb else -1; wd = r.uniform(3.5, 10) * lw
            idx = np.linspace(0, len(q) - 1, 9).round().astype(int); k = cv2.GaussianBlur((q + N * sd * wd * .8).astype(np.float32), (1, 5), 1.2) if len(q) > 5 else q; PP.append(k[idx]); WD.append(wd)
        if PP: cv.strokes(np.array(PP), np.array(WD), rgb(INK), op=.92, dry=.42, seed=seed, shape='flat', bristle=0.); dl = cv.a * cv2.dilate(aa, np.ones((5, 5), np.uint8)); dl = dl * (1 - cv2.resize((part == P_TEETH).astype(f32), (W, H), interpolation=cv2.INTER_LINEAR))
    ik = np.maximum(K, dl * .85); wk = wpen.raster(H, W) * float(white)
    al = np.maximum(aa, K * .98)
    c = col * aa[..., None]; c = c * (1 - ik[..., None]) + ink_c * ik[..., None]; c = c * (1 - wk[..., None]) + (np.array([1.02, 1.0, .96], f32) if pal else np.array([1.12, 1.15, 1.18], f32)) * wk[..., None]
    al = np.maximum(al, wk)
    return dict(layer=np.dstack([c, al]).astype(f32), z=zf, alpha=aa, V=Vf, g=g)


def _form_lines(m, cam, pen, wpen, zf, V_, r, lw):
    """authored 3-D ink: wrap lines round the horn-tail, scribbled light along its lit edge"""
    pts, rad = m.horn_w, m.horn_r; n = len(pts); fr = m._frames(pts); C = m.wheel['c'] if m.with_wheel else pts.mean(0)
    def vis(P3, tol=.035):
        p, z = cam.project(P3); d = np.linalg.norm(P3 - cam.eye, axis=-1); zz = _samp(zf, p); return p, (d < zz + tol) & (p[:, 0] > -20) & (p[:, 0] < W + 20) & (p[:, 1] > -20) & (p[:, 1] < H + 20)
    def put(P3, w, pn, taper=(.12, .55), tol=.035):
        p, v = vis(P3, tol); i = 0
        while i < len(p):
            if not v[i]: i += 1; continue
            j = i
            while j < len(p) and v[j]: j += 1
            if j - i > 4: pn.stroke(p[i:j], w, r.integers(1 << 30), taper=taper, press=.4, tremor=.4)
            i = j
    u = .12
    while u < .95:
        k = u * (n - 1); i = int(k); c = pts[i]; R = rad[i] * 1.01; T = pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]; T = T / (np.linalg.norm(T) + 1e-9); toC = C - c; toC = toC - T * (toC @ T); toC = unit(toC); bi = np.cross(T, toC)
        for sgn in ((1,) if r.random() < .6 else (1, -1)):
            a0 = r.uniform(2.4, 3.3) * sgn; a1 = r.uniform(-.5, 1.5) * sgn; a = np.linspace(a0, a1, 26); sh = r.normal(0, .5); tt = np.linspace(0, 1, 26)
            P3 = c + R * (np.cos(a)[:, None] * toC + np.sin(a)[:, None] * bi) + T * (R * (sh * (tt - .3) ** 2 + .25 * np.sin(tt * 3 + sh)))[:, None]
            put(P3, r.uniform(1.6, 2.6) * lw, pen)
            if r.random() < .3: put(P3 + T * R * r.uniform(.08, .16), r.uniform(1., 1.5) * lw, pen)
        u += r.uniform(.016, .05)
    # the blade: light along both edges, a black ridge, nicks
    if getattr(m, 'blade_M', None) is not None:
        org, ax, Lb = m.blade_line; by = m.blade_M[1]; wd0 = .078 * m.sc * m.rk_arm
        for sg in (-1, 1):
            for q in range(3):
                u0 = r.uniform(.12, .5) if q else .12; u1 = r.uniform(.6, 1.) if q else 1.; u = np.linspace(u0, u1, 40); tt = np.clip((1 - u) / .34, 0, 1) ** .74; e = np.where(u > .66, tt, 1.) * (1 - .05 * q)
                put(org + ax * (u * Lb)[:, None] + by * (sg * wd0 * e + m.blade_c * (u * Lb) ** 2)[:, None], (2.4 - .6 * q) * lw, wpen if (q < 2 and sg > 0) or q == 0 else pen, (.1, .3), .06)
        for u0 in (.40, .455, .66):                                         # gleam dragged across the flat
            t_ = np.linspace(-.9, .9, 12); put(org + ax * ((u0 + .05 * t_) * Lb)[:, None] + by * (wd0 * t_ + m.blade_c * (u0 * Lb) ** 2)[:, None], r.uniform(2, 7) * lw, wpen, (.4, .4), .06)
        u = np.linspace(.15, .93, 40); put(org + ax * (u * Lb)[:, None] + by * (m.blade_c * (u * Lb) ** 2)[:, None], 2.2 * lw, pen, (.1, .5), .06)
    # muscle fibres: short strokes that run with each muscle
    S = m.S; mi = np.nonzero((S['T'] == 1) & (S['MAT'] == SKIN) & np.isin(S['PART'], (P_TORSO, P_ARML, P_ARMR)) & (S['SUB'] == 0))[0]
    for i in mi:
        rx, ry, rz = S['P'][i, :3]; ce = S['A'][i]; M_ = S['M'][i]; zc = cam.cam(ce)[2]
        if zc < .1 or max(rx, rz) < .05: continue
        pxr = rx * cam.f / zc
        if pxr < 16: continue
        nl = int(min(7, 1 + pxr / 26)); a0 = r.uniform(0, 6.28)
        for q in range(nl):
            if r.random() < .25: continue
            a = a0 + r.normal(0, .9); t0 = r.uniform(-.85, .35); t1 = t0 + r.uniform(.15, .55); t = np.linspace(t0, min(t1, .92), 14); a_ = a + np.linspace(0, r.normal(0, .25), 14); q_ = np.sqrt(np.maximum(0, 1 - t * t))
            P3 = ce + M_[0] * (rx * np.cos(a_) * q_)[:, None] + M_[1] * (ry * np.sin(a_) * q_)[:, None] + M_[2] * (rz * t)[:, None]
            out_ = P3 - ce; out_ /= np.linalg.norm(out_, axis=1)[:, None]
            if ((P3[7] - cam.eye) @ out_[7]) > 0: continue
            put(P3 + out_ * .004, r.uniform(1.1, 2.1) * lw, pen, (.2, .65), .022)
            if r.random() < .4: put(P3 + out_ * .004 + M_[0] * rx * .10, r.uniform(.9, 1.4) * lw, pen, (.3, .65), .022)
    # feathers: shaft, a few barbs, a split or two
    S = m.S; fi = np.nonzero(S['MAT'] == FEATH)[0]; fi = fi[S['T'][fi] == 4]
    for i in fi:
        Lh, wd, th, bd = S['P'][i, :4]; M_ = S['M'][i]; ce = S['A'][i]; zc = cam.cam(ce)[2]
        if zc < .05: continue
        px = wd * cam.f / zc
        if px < 5: continue
        side = 1 if (cam.eye - ce) @ M_[2] > 0 else -1; on = lambda u, v: ce + M_[0] * u * Lh + M_[1] * v * wd + M_[2] * (bd * (u * Lh) ** 2 + side * th * .9 * max(0., 1 - u ** 4 - v * v) ** .5)
        if r.random() < .72:
            u0, u1 = r.uniform(-.35, .1), r.uniform(.55, .93); b = r.normal(0, .10); put(np.array([on(u, b * (1 - u)) for u in np.linspace(u0, u1, 14)]), r.uniform(1.1, 1.9) * lw, pen, (.2, .6), .02)
        for q in range(int(r.integers(0, 4))):
            u = r.uniform(-.1, .75); sg = r.choice([-1, 1]); l = r.uniform(.3, .75); put(np.array([on(u + .26 * l * t, sg * (.10 + l * t * .85)) for t in np.linspace(0, 1, 9)]), r.uniform(.9, 1.5) * lw, pen, (.15, .7), .02)
        if r.random() < .22:
            u = r.uniform(.55, .8); v = r.uniform(-.5, .5); put(np.array([on(u + t * .3, v * (1 - .3 * t)) for t in np.linspace(0, 1, 7)]), 2.2 * lw, pen, (.6, .05), .03)
    # long sinew lines down the trunk
    for q in range(22):
        a0 = r.uniform(1.3, 5.0); i0 = int(r.uniform(.02, .3) * n); i1 = min(n - 2, i0 + int(r.uniform(.1, .4) * n)); P3 = []
        for i in range(i0, i1):
            T = pts[i + 1] - pts[i - 1] if i else pts[1] - pts[0]; T = unit(T); t_ = unit((C - pts[i]) - T * ((C - pts[i]) @ T)); b_ = np.cross(T, t_); a = a0 + .5 * (i - i0) / n * 6 * .3; P3.append(pts[i] + rad[i] * 1.0 * (math.cos(a) * t_ + math.sin(a) * b_))
        if len(P3) > 5: put(np.array(P3), r.uniform(1.6, 2.8) * lw, pen, (.3, .6))
    # scribbled light on the edge that faces the wheel
    for q in range(26):
        i0 = int(r.uniform(.08, .9) * n); i1 = min(n - 1, i0 + int(r.uniform(.05, .18) * n)); a0 = r.normal(0, .55); P3 = []
        for i in range(i0, i1):
            T = unit(pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]); t_ = unit((C - pts[i]) - T * ((C - pts[i]) @ T)); b_ = np.cross(T, t_); P3.append(pts[i] + rad[i] * 1.01 * (math.cos(a0) * t_ + math.sin(a0) * b_))
        if len(P3) < 5: continue
        P3 = np.array(P3); p, v = vis(P3)
        if v.mean() > .5 and float(np.mean(_samp(V_, p[v]))) > .55: put(P3, r.uniform(1.2, 3.2) * lw, wpen, (.3, .5))


# ------------------------------------------------------------------ the wheel burning
def notch(t, n0=0., dur=3.):
    """wheel angle for one GAKON: t in frames from the hit.  -8..0 anticipation (back 8 deg), 0..3 snap one eighth, overshoot 12 deg, settle by ~12"""
    st = math.pi / 4
    if t <= -9: return n0
    if t < 0: u = (t + 9) / 9; return n0 - math.radians(8) * ease(u) * (1 - .10 * max(0, u - .8))
    if t < dur: u = (t + 1) / dur; u = min(1, u); return n0 - math.radians(8) + (st + math.radians(8 + 12)) * (u ** .6)
    u = t - dur; return n0 + st + math.radians(12) * math.exp(-u / 2.4) * math.cos(u * 1.15)


def _wp(wh, cam, pts):
    """wheel-plane units (1 = orb distance) -> screen"""
    P = wh['c'] + wh['R'] * (pts[:, :1] * wh['ex'] + pts[:, 1:2] * wh['ey']); return cam.project(P)[0]


def burn_wheel(wh, cam, seed=0, prog=1.0, ignite=1.0, rim=1.0, core=1.0, I=1.0, width=1.0, mark=None):
    """scribbled light.  prog / ignite: scalar or 8 values (spoke drawn 0..1 outward, orb ignition 0..1), rim 0..1 drawn round.
    -> (light HxWx3 additive, may exceed 1;  ink mask HxW of the scribble)"""
    r = np.random.default_rng(seed * 101 + 7); prog = np.broadcast_to(np.asarray(prog, float), (8,)); ign = np.broadcast_to(np.asarray(ignite, float), (8,))
    cpx, z = cam.project(wh['c'][None]); px = wh['R'] * cam.f / max(z[0], .1); u = px / 210. * width                    # stroke scale
    hot, thin = Pen(), Pen(); sp = wh['spin']
    def line(P2, w, pen=hot, tp=(.05, .12), tr=.6, al=1., seedk=None): pen.stroke(_wp(wh, cam, np.asarray(P2, float)), max(1., w * u), r.integers(1 << 30), taper=tp, press=.55, tremor=tr * u, al=al, tmin=.25, step=2.5)
    for i in range(8):
        a = sp + i * math.pi / 4; d = np.array([math.cos(a), math.sin(a)]); n = np.array([-d[1], d[0]]); L = .80 * prog[i]
        if prog[i] <= .01: continue
        for k in range(int(3 + r.integers(0, 3))):
            s0 = r.uniform(.0, .30) if k else .02; s1 = min(L, r.uniform(.5, .86) if k else L); o0, o1 = r.normal(0, .016, 2)
            if s1 > s0 + .05: line([d * s0 + n * o0, d * (s0 + s1) / 2 + n * (o0 + o1) / 2 + n * r.normal(0, .006), d * s1 + n * o1], (9 - 1.3 * k) * r.uniform(.5, 1.1))
        for k in range(int(r.integers(1, 4))):                                 # chisel blocks along the spoke
            s0 = r.uniform(.15, .7) * L; s1 = min(L, s0 + r.uniform(.06, .17))
            if s1 > s0 + .03: line([d * s0 + n * r.normal(0, .014), d * s1 + n * r.normal(0, .014)], r.uniform(9, 17), tp=(.0, .0))
        if r.random() < .5 and L > .5: line([d * .2 + n * r.normal(0, .05), d * L + n * r.normal(0, .05)], 1.6, thin, (.3, .4))
    if rim > .01:
        for k in range(15):
            a0 = r.uniform(0, 2 * math.pi); a1 = a0 + r.uniform(.5, 2.6) * rim; R0 = .56 * (1 + r.normal(0, .022)); R1 = R0 * (1 + r.normal(0, .03)); t = np.linspace(a0, a1, 40)
            if rim < 1: t = t[((t - sp) % (2 * math.pi)) < rim * 2 * math.pi]
            if len(t) < 5: continue
            ox, oy = r.normal(0, .010, 2); R = np.linspace(R0, R1, len(t)); line(np.c_[ox + R * np.cos(t), oy + R * np.sin(t)], r.uniform(3.5, 9), tp=(.10, .2))
        for k in range(8):
            a0 = r.uniform(0, 2 * math.pi); t = np.linspace(a0, a0 + r.uniform(.4, 1.5) * rim, 30); R = .56 * (1 + r.choice([-1, 1]) * r.uniform(.05, .17)) + np.linspace(0, r.normal(0, .06), 30); line(np.c_[R * np.cos(t), R * np.sin(t)], 1.7, thin, (.35, .5))
    glow_o = []
    for i in range(8):
        if ign[i] <= .01: continue
        a = sp + i * math.pi / 4; c = np.array([math.cos(a), math.sin(a)]) * 1.0; R = .165 * (.55 + .45 * min(1, ign[i])) * r.uniform(.9, 1.1) * (1 + .5 * max(0, ign[i] - 1)); e = ign[i]
        if i == mark:                                                           # the cracked one: hollow, two torn arcs, a jagged black split; it lets the eye follow the turn
            for k in range(6): rr = R * r.uniform(.70, 1.0); a0 = a + (.75 if k % 2 else 3.6) + r.normal(0, .12); t = np.linspace(a0, a0 + r.uniform(1.3, 2.0), 26); line(np.c_[c[0] + rr * np.cos(t), c[1] + rr * np.sin(t)], r.uniform(2.0, 4.5), tp=(.2, .3))
            for k in range(3): o = c + r.normal(0, R * .10, 2) - np.array([math.cos(a + .9), math.sin(a + .9)]) * R * .25; rr = R * r.uniform(.10, .26); t = np.linspace(0, r.uniform(2, 4), 16) + r.uniform(0, 6); line(np.c_[o[0] + rr * np.cos(t), o[1] + rr * np.sin(t)], R * 24, al=.75)
            glow_o.append((_wp(wh, cam, c[None])[0], R * px, e * .30)); continue
        for k in range(7):                                                     # scribbled core
            o = c + r.normal(0, R * .12, 2); rr = R * r.uniform(.08, .5); t = np.linspace(0, r.uniform(3, 7), 24) + r.uniform(0, 6); line(np.c_[o[0] + rr * np.cos(t), o[1] + rr * np.sin(t)], R * 62 * r.uniform(.7, 1.1), al=min(1, e))
        for k in range(int(9 * min(1, e)) + 2):                                # loops round it
            o = c + r.normal(0, R * .10, 2); rr = R * r.uniform(.62, 1.0); el = r.uniform(.88, 1.12); t = np.linspace(0, r.uniform(1.0, 4.4), 30) + r.uniform(0, 6.28); line(np.c_[o[0] + rr * np.cos(t), o[1] + rr * el * np.sin(t)], r.uniform(1.8, 5.5), tp=(.18, .3))
        for k in range(int(4 * min(1, e))):
            o = c + r.normal(0, R * .22, 2); rr = R * r.uniform(1.05, 1.42); t = np.linspace(0, r.uniform(1.2, 4.2), 34) + r.uniform(0, 6.28); g_ = np.linspace(1, r.uniform(.85, 1.25), 34); line(np.c_[o[0] + rr * g_ * np.cos(t), o[1] + rr * g_ * np.sin(t)], 1.5, thin, (.35, .5))
        glow_o.append((_wp(wh, cam, c[None])[0], R * px, e))
    Mh = hot.raster(H, W); Mt = thin.raster(H, W); M = np.maximum(Mh, Mt * .85)
    cx, cy = cpx[0]; xx, yy = grid(H, W)
    # foreshortened radial distance in the wheel plane
    e1 = (_wp(wh, cam, np.array([[.5, 0.]]))[0] - cpx[0]) * 2; e2 = (_wp(wh, cam, np.array([[0., .5]]))[0] - cpx[0]) * 2; Mi = np.linalg.inv(np.stack([e1, e2], 1)); ux = (xx - cx) * Mi[0, 0] + (yy - cy) * Mi[0, 1]; uy = (xx - cx) * Mi[1, 0] + (yy - cy) * Mi[1, 1]; rr = np.sqrt(ux * ux + uy * uy); rs = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / px; rd = .5 * rr + .5 * rs
    veil = (.62 * np.exp(-(rd / .42) ** 1.5) + .26 * np.exp(-(rd / .95) ** 1.6) + .085 * np.exp(-(rd / 2.0) ** 1.3)) * core
    b = .50 * blur(M, 5 * u) + .42 * blur(M, 17 * u) + .34 * blur(M, 46 * u) + .22 * blur(M, 110 * u)
    for p, R, e in glow_o: b = b + .30 * e * np.exp(-(((xx - p[0]) ** 2 + (yy - p[1]) ** 2) / (R * 1.5) ** 2))
    cr = 1.5 * np.exp(-(rr / .155) ** 2.2) * core + .5 * np.exp(-(rr / .30) ** 2) * core
    lt = M * 1.25 + b * .62 + cr
    cool = np.array([.965, .985, 1.02], f32); light = lt[..., None] * cool + (veil * .62)[..., None] * np.array([.93, .97, 1.0], f32)
    return (light * I).astype(f32), M


COOL = np.array([.965, .985, 1.02], f32)


def aim(eye, pts, px, f=None, roll0=.2, f0=1200., lock=False):
    """camera at `eye` that puts world points at screen px.  two points: solves pan, tilt, roll and lens; one point or f given: fewer unknowns"""
    from scipy.optimize import least_squares
    pts = np.atleast_2d(np.asarray(pts, float)); px = np.atleast_2d(np.asarray(px, float)); eye = np.asarray(eye, float); d = pts.mean(0) - eye
    az, el = math.atan2(d[0], d[1]), math.asin(d[2] / np.linalg.norm(d)); n = 2 * len(pts)
    def mk(x):
        a, e = x[:2]; r = x[2] if len(x) > 2 else roll0; ff = math.exp(x[3]) if len(x) > 3 else (f or f0)
        return Camera3D(eye, eye + np.array([math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)]), ff, roll=r)
    x0 = [az, el] + ([roll0] if n >= 4 and not lock else []) + ([math.log(f0)] if n >= 4 and f is None else [])
    r = least_squares(lambda x: (mk(x).project(pts)[0] - px).ravel(), x0); return mk(r.x)


def reproject(cam, P, px, depth=0.):
    """move world point P so that it projects to px (same distance + depth metres)"""
    z = cam.cam(P)[2] + depth; q = np.array([(px[0] - cam.cx) / cam.f * z, (px[1] - cam.cy) / cam.f * z, z]); return cam.eye + q @ cam.R


# ------------------------------------------------------------------ key poses (character space: faces -Y, +X his left)
def _hand(t, pole, fwd, dor, curl=.35, spread=.3, thumb=.3): return dict(t=t, pole=pole, fwd=fwd, dor=dor, curl=curl, spread=spread, thumb=thumb)
def _foot(t, pole, fwd, pitch=0.): return dict(t=t, pole=pole, fwd=fwd, pitch=pitch)

KEYS = dict(
    stand=dict(),
    cock=dict(neck=(0, .30, .25), head=(0, -.38, .50), chest=(0, .12, .05)),
    stride=dict(root=(-.04, -.10, 1.36), pelvis=(.20, .16, -.06), waist=(0, .14, 0), chest=(-.36, .30, .08), neck=(.10, .62, -.05), head=(.12, -.66, -.06), shl=(0, -.06, -.02), shr=(0, .05, .05),
                lh=_hand((.66, -1.00, 1.50), (1.6, .3, 1.5), (-.05, -.85, -.45), (.5, -.15, .85), (.42, .34, .36, .46), 1.0, .2),
                rh=_hand((-1.16, .16, 1.66), (-1.1, 1.4, 2.2), (-.70, .05, -.70), (-.6, .5, .6), .95, .1, .8),
                lf=_foot((.30, .74, .30), (.6, -2, .6), (.22, -1, 0), .80), rf=_foot((-.26, -.82, .14), (-.40, -2.6, .8), (-.18, -1, 0), -.20), sway=.9),
    stride2=dict(root=(.05, -.10, 1.50), pelvis=(-.18, .10, .05), waist=(0, .10, 0), chest=(.30, .26, -.06), neck=(-.10, .52, .05), head=(-.10, -.52, .06), shl=(0, .03, .03), shr=(0, -.05, -.02),
                 lh=_hand((.92, .18, 1.34), (1.4, 1.0, 1.8), (.15, -.30, -1), (1, .2, 0), (.5, .45, .4, .35), .4, .4),
                 rh=_hand((-.66, -.60, 1.36), (-1.6, .3, 1.7), (-.10, -.70, -.75), (-1, -.15, .2), .95, .1, .8),
                 lf=_foot((.24, -.60, .14), (.35, -2.4, .8), (.18, -1, 0), -.22), rf=_foot((-.27, .58, .25), (-.5, -2, .7), (-.22, -1, 0), .65), sway=-.4),
    sweep_a=dict(root=(.02, .12, 1.40), pelvis=(-.45, .05, 0), waist=(-.20, .05, 0), chest=(-.62, .10, -.12), neck=(.35, .30, 0), head=(.55, -.28, .10), shl=(0, -.10, .02), shr=(.02, .10, .08),
                 lh=_hand((.10, -.86, 2.05), (1.2, -.6, 1.7), (-.3, -.8, .4), (.2, .3, 1), (.3, .25, .25, .3), .8, .2),
                 rh=_hand((-1.10, .55, 2.62), (-1.5, 1.0, 2.0), (-.2, .55, .8), (-.7, -.4, .4), .95, .1, .8),
                 lf=_foot((.42, -.52, .13), (.6, -2, .9), (.25, -1, 0)), rf=_foot((-.42, .50, .16), (-1., -1.5, .8), (-.75, -.7, 0), .25), jaw=.16),
    sweep=dict(root=(0, -.30, 1.36), pelvis=(.35, .22, .08), waist=(.18, .14, .05), chest=(.55, .22, .10), neck=(-.30, .40, -.05), head=(-.62, -.52, -.10), shl=(0, .10, .04), shr=(.06, -.16, -.02),
               lh=_hand((1.06, .20, 2.10), (1.4, .9, 1.7), (.8, .1, .5), (.3, .9, .2), (.45, .35, .3, .3), .9, .2),
               rh=_hand((-.30, -1.30, 1.86), (-1.6, -.5, 1.5), (.7, -.7, -.15), (-.5, -.6, .5), .95, .1, .8),
               lf=_foot((.44, .42, .18), (.9, -1.5, .7), (.5, -1, 0), .4), rf=_foot((-.36, -.72, .13), (-.5, -2.6, .8), (-.1, -1, 0)), jaw=.46, sway=.8),
    sweep_f=dict(root=(.06, -.30, 1.30), pelvis=(.60, .25, .10), waist=(.28, .18, .08), chest=(.72, .26, .14), neck=(-.45, .40, -.05), head=(-.70, -.45, -.10), shl=(0, .12, .0), shr=(.08, -.14, -.04),
                 lh=_hand((.82, .42, 1.85), (1.4, 1.0, 1.7), (.6, .4, -.5), (.6, .7, .2), (.5, .4, .35, .3), .7, .3),
                 rh=_hand((.62, -.92, 1.62), (-.8, -1.4, 1.6), (.95, -.05, -.25), (-.1, -.8, .6), .95, .1, .8),
                 lf=_foot((.46, .40, .16), (.9, -1.5, .7), (.5, -1, 0), .35), rf=_foot((-.34, -.74, .13), (-.5, -2.6, .8), (-.1, -1, 0)), jaw=.3, sway=1.2),
    palm=dict(root=(0, -.25, 1.40), pelvis=(-.35, .15, 0), waist=(-.15, .10, 0), chest=(-.50, .18, 0), neck=(.25, .40, 0), head=(.45, -.45, -.08), shl=(.05, -.16, .02), shr=(0, .10, .02),
              lh=_hand((.12, -1.50, 2.18), (1.2, -.7, 1.5), (.10, -.25, 1), (0, 1, .25), (.28, .16, .18, .30), 1.0, .1),
              rh=_hand((-.95, .42, 1.50), (-1.3, 1.1, 1.9), (-.2, -.1, -1), (-1, .3, 0), .95, .1, .8),
              lf=_foot((.34, -.70, .13), (.5, -2.4, .9), (.1, -1, 0)), rf=_foot((-.42, .55, .18), (-.9, -1.5, .7), (-.6, -.8, 0), .4), jaw=.08),
    cut=dict(root=(.03, .10, 1.46), pelvis=(-.1, -.05, -.06), waist=(-.05, -.06, -.04), chest=(-.22, -.06, -.14), neck=(.1, .25, .1), head=(.15, -.30, .22), shr=(0, .06, .06), shl=(0, 0, -.02),
             lh=_hand((.86, -.30, 1.45), (1.4, .6, 1.8), (.2, -.5, -.8), (.8, -.2, .3), (.7, .65, .6, .55), .3, .5),
             rh=_hand((-.90, -.20, 1.90), (-1.4, .6, 2.2), (-.5, -.4, -.4), (-.6, .4, .6), .95, .1, .8), lf=_foot((.36, .25, .13), (.5, -2, .9), (.4, -1, 0)), rf=_foot((-.36, -.30, .13), (-.4, -2, .9), (-.2, -1, 0)), cut=.3, jaw=.04),
    lunge=dict(root=(0, -.55, 1.18), pelvis=(.10, .45, .04), waist=(.05, .25, 0), chest=(-.10, .30, .04), neck=(0, .55, 0), head=(0, -.95, -.10), shl=(.02, -.18, .04), shr=(.02, -.10, .06),
               lh=_hand((.40, -1.90, 1.62), (1.4, -1.0, 1.3), (-.15, -.9, .35), (.15, .35, .9), (.42, .32, .34, .42), 1.1, .15),
               rh=_hand((-1.10, -1.10, 2.20), (-1.7, -.4, 1.9), (.25, -.95, .1), (-.4, -.1, .9), .95, .1, .8),
               lf=_foot((.40, .85, .30), (.7, -1, .4), (.3, -1, 0), .9), rf=_foot((-.34, -.60, .13), (-.5, -2.6, .7), (-.1, -1, 0)), jaw=.55, sway=1.4),
    roar=dict(root=(0, -.05, 1.48), pelvis=(0, .05, 0), chest=(0, -.10, 0), neck=(0, .30, 0), head=(0, -.72, 0), jaw=.62, shl=(0, .06, .04), shr=(0, .06, .04),
              lh=_hand((1.05, -.35, 1.62), (1.5, .8, 1.8), (.3, -.7, -.5), (.7, -.2, .6), (.55, .45, .42, .48), 1., .2), rh=_hand((-1.05, -.35, 1.62), (-1.5, .8, 1.8), (-.3, -.7, -.5), (-.7, -.2, .6), .9, .2, .7),
              lf=_foot((.42, -.05, .13), (.6, -2, .9), (.4, -1, 0)), rf=_foot((-.42, -.05, .13), (-.6, -2, .9), (-.4, -1, 0))),
)


# ------------------------------------------------------------------ air: painted smoke that the wheel lights, in depth
_SMK = {}
def smoke(img, z, cam, light, layers, f=0., seed=0, key='#c9d6e4', dark='#06080c', d0=1.6, I=1.0, steps=True):
    """layers = [(distance m, density, top height m, falloff m, feature px at 1080, drift px/frame), ...] far -> near, z-tested against z (ray distance).
    Lit by the wheel at `light` (world point): toned in brush-edged steps, so it reads as painted smoke."""
    q = 2; h, w = H // q, W // q
    ry = cam.rays()[::q, ::q]; ry = ry / np.linalg.norm(ry, axis=-1, keepdims=True); zq = z[::q, ::q]; E = cam.eye.astype(f32); Lp = np.asarray(light, f32)
    base = np.zeros((h, w, 3), f32); al = np.zeros((h, w), f32); ws = wash(h, w, seed + 5); K, D = rgb(key), rgb(dark)
    for i, (dist, den, top, fo, px, dr) in enumerate(layers):
        kk = (seed, i, px)
        if kk not in _SMK: _SMK[kk] = fbm(h, w * 2, (px / q, px / q * .42, px / q * .18, px / q * .08), (1, .55, .33, .2), seed * 17 + i, aspect=1.7)
        n = _SMK[kk]; off = (f * dr / q + 137 * i) % w; M2 = np.float32([[1, 0, -off], [0, 1, 0]]); n = cv2.warpAffine(n, M2, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        wx = cv2.warpAffine(_SMK[kk], np.float32([[1, 0, -(off * .6 + 311) % w], [0, 1, 40]]), (w, h), borderMode=cv2.BORDER_REFLECT); yy, xx = np.mgrid[0:h, 0:w].astype(f32); n = cv2.remap(n, xx + wx * px * .05, yy + n * px * .05, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        P = E + ry * dist; hm = np.exp(-np.maximum(P[..., 2] - top, 0) / fo); a = smooth(-.35, 1.05, n + 1.1 * (hm - .72)) * den * hm ** .5
        a = a * smooth(dist - .5, dist + .7, zq)
        r = np.linalg.norm(P - Lp, axis=-1); lit = I / (1 + (r / d0) ** 2)
        v = lit * (.35 + .65 * np.clip(.5 + .5 * n, 0, 1.2)) * (1 + .12 * ws)
        if steps:
            vv = np.zeros_like(v)
            for t, s_ in ((.05, .05), (.12, .09), (.24, .16), (.42, .25)): vv += s_ * smooth(t * .72, t * 1.28, v)
            v = .6 * vv + .4 * v * .8
        c = D + (K - D) * np.clip(v, 0, 1)[..., None]; a = np.clip(a, 0, 1)[..., None]
        base = base * (1 - a) + c * a; al = al * (1 - a[..., 0]) + a[..., 0]
    base = cv2.resize(base, (W, H), interpolation=cv2.INTER_LINEAR); al = cv2.resize(al, (W, H), interpolation=cv2.INTER_LINEAR)
    return img * (1 - al[..., None]) + base


def rays(cam, light, seed=0, n=70, f=0):
    """radial light shafts mask (H,W) around the projected light"""
    p = cam.project(np.asarray(light, float)[None])[0][0]; x, y = grid(H // 2, W // 2); dx, dy = x * 2 - p[0], y * 2 - p[1]; a = np.arctan2(dy, dx); r = np.hypot(dx, dy); g = np.random.default_rng(seed).random(n + 1); g[-1] = g[0]
    g = np.convolve(np.r_[g[-3:-1], g, g[1:3]], [.2, .6, .2], 'same')[2:-2]
    u = (a + math.pi) / (2 * math.pi) * n; st = np.interp(u.ravel(), np.arange(n + 1), g).reshape(a.shape).astype(f32)
    return cv2.resize(np.clip(st - .45, 0, 1) * np.exp(-r / 520) * smooth(60, 260, r), (W, H)).astype(f32)


def motes(img, cam, seed, f, n=90, box=((-3, 3), (-3, 4), (0, 5)), light=None, d0=1.3, size=2.2, col='#e8f0fa', vel=(.004, 0, .010)):
    """dust and ash in the air, lit by the wheel (additive)"""
    r = np.random.default_rng(seed); B3 = np.array(box, float); P = B3[:, 0] + (B3[:, 1] - B3[:, 0]) * r.random((n, 3)); ph = r.uniform(0, 6.28, (n, 3)); V_ = np.asarray(vel, float)
    P = P + V_ * f + .05 * np.sin(ph + f * .07); P[:, 2] = B3[2, 0] + (P[:, 2] - B3[2, 0]) % (B3[2, 1] - B3[2, 0]); p, z = cam.project(P); pen = Pen(); L = np.ones(n) if light is None else 1 / (1 + (np.linalg.norm(P - light, axis=1) / d0) ** 2)
    v2 = cam.project(P + V_ * 3 + .01 * np.cos(ph))[0]
    for i in range(n):
        if z[i] < .4: continue
        rr = size * r.uniform(.3, 1.2) * min(3., 3. / z[i]); dv = v2[i] - p[i]; dv = dv * min(1., rr * 1.6 / (np.hypot(*dv) + 1e-6)); pen.add(np.array([p[i], p[i] + dv]), rr * .5, float(np.clip(L[i] * r.uniform(.4, 1.2), 0, 1)))
    return img + pen.raster(H, W)[..., None] * rgb(col) * .9


# ------------------------------------------------------------------ smoke with volume: lobes in the same sculpt, toned in painted steps
class Billow:
    """banks = [(x, y, z, radius, n lobes)]; lobes boil with t (drawn on twos: pass t = f // 2).  .S like a creature"""
    def __init__(self, banks, seed=0, t=0., rise=0.):
        from .maho_sdf import Sculpt
        s = self.s = Sculpt(); r = np.random.default_rng(seed); self.J = {}; self.sc = 1.; self.with_wheel = False
        for b, (x, y, z, R, n) in enumerate(banks):
            c = V(x, y, z); s.ell(c, (R, R * .9, R * .72), np.eye(3), k=.0, g=b, mat=14, part=P_ORN)
            for i in range(n):
                a = r.uniform(0, 6.28); e = r.uniform(-.15, 1.25); q = r.uniform(.30, .62) * R; ph = r.uniform(0, 6.28); wob = .07 * R * np.array([math.sin(t * .22 + ph), math.cos(t * .19 + ph), .6 * math.sin(t * .25 + ph * 2)])
                d = V(math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)); p = c + d * R * r.uniform(.75, 1.02) * V(1, .9, .78) + wob + V(0, 0, rise * t * .01); s.ell(p, (q, q, q * .9), np.eye(3), k=.16 * R, g=b, mat=14, part=P_ORN)
                for j in range(3):
                    d2 = unit(d + r.normal(0, .55, 3)); q2 = q * r.uniform(.35, .55); s.ell(p + d2 * q * .88, (q2, q2, q2 * .92), np.eye(3), k=.07 * R, g=b, mat=14, part=P_ORN)
        self.S = s.pack()


def draw_billow(b, cam, lights, seed=0, scale=.5, soft=.5, fill=.01, ink=.5, lw=.6, col=None, op=.95, warp=11.):
    """-> dict(layer, z).  soft 0..1 = how wispy the rims are"""
    g = gbuffer(b.S, cam, scale); h, w = g['h'], g['w']; ok = g['id'] >= 0; x, cols, ws, ndv = _light(g, b, lights, seed)
    x = (x + fill * (.4 + .6 * g['ao'])) * ALB[14]; Vt, _ = _tone(x, ws, seed, h, w, .35); c = ramp(Vt, SLATE) if col is None else ramp(Vt, col)
    for k, cc, _m in cols: v, _ = _tone(k * ALB[14], ws, seed + 5, h, w, .3); c = c + np.maximum(v - LV[0] * 1.2, 0)[..., None] * cc
    a = ok * smooth(0, .12 + .62 * soft, ndv + .10 * ws * soft) * op
    P = cv2.GaussianBlur(a.astype(f32), (0, 0), 1 + 5 * soft * scale); a = np.minimum(a, P * 1.15); A = ok.astype(f32); cb = cv2.GaussianBlur(c * A[..., None], (0, 0), 2.) / (cv2.GaussianBlur(A, (0, 0), 2.)[..., None] + 1e-4); c = np.where(ok[..., None], c, cb)
    nx = fbm(h, w, (70 * scale, 26 * scale, 11 * scale), (1, .6, .35), seed + 31); ny = fbm(h, w, (70 * scale, 26 * scale, 11 * scale), (1, .6, .35), seed + 37); yy, xx = np.mgrid[0:h, 0:w].astype(f32); mx, my = xx + nx * warp * scale, yy + ny * warp * scale
    c = cv2.remap(c.astype(f32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE); a = cv2.remap(a.astype(f32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE); a = a * (1 - soft * .35 * smooth(.3, 1.6, np.roll(nx, 40, 1)))
    c = cv2.resize(c, (W, H)); a = cv2.resize(a, (W, H)); z = cv2.resize(cv2.remap(np.where(ok, g['z'], 1e3).astype(f32), mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE), (W, H), interpolation=cv2.INTER_NEAREST)
    if ink > 0:
        out, inn, hd, cr, *_ = _edges(g, b.S); r = np.random.default_rng(seed); pen = Pen(); V_ = cv2.resize(Vt.astype(f32), (W, H)); _pen_edges(pen, [p for p in trace(inn | hd, int(14 * scale) + 4)], 2.4, r, scale, V_, lw, 's', brk=.8, dbl=.15, lo=.6, hi=1.1, seg=(30, 110)); K = pen.raster(H, W) * ink * smooth(.3, .8, a); c = c * (1 - K[..., None]) + rgb(INK) * K[..., None]
    return dict(layer=np.dstack([c * a[..., None], a]).astype(f32), z=z)


def zover(img, d, z, fade=.35):
    """composite a drawn layer with depth against z (ray distance), soft where they meet"""
    k = smooth(-fade, fade, z - d['z'])[..., None] * 1.; a = d['layer'][..., 3:4] * k
    return img * (1 - a) + d['layer'][..., :3] * k


def blade_at(m, u):
    """points along the sword, u = 0 (heel) .. 1 (tip), world"""
    o, a, L = m.blade_line; return o + a * (np.atleast_1d(u) * L)[:, None]


def arc_blades(m, frm, n=9, pivot=None):
    """earlier positions of the sword: swung on a great arc from direction `frm` (unit, world) to where it is now, about the shoulder"""
    S0 = m.J['S-1'] if pivot is None else np.asarray(pivot, float); hl, tp = blade_at(m, [0, 1]); e = tp - S0; Lr = np.linalg.norm(e); e = e / Lr; f_ = unit(frm); ax = np.cross(f_, e); an = math.atan2(np.linalg.norm(ax), f_ @ e); ax = unit(ax); out = []
    rot = lambda v, a: v * math.cos(a) + np.cross(ax, v) * math.sin(a) + ax * (ax @ v) * (1 - math.cos(a))
    for k in range(n): a = -an * (1 - k / (n - 1)); out.append((S0 + rot(hl - S0, a), S0 + rot(tp - S0, a)))
    return out


def sweep_trail(blades, cam, seed=0, light='#dfe9f6', u0=.35, n=60, w=1.0, fill=1.0, age=1.0, dark=.35):
    """brush smear of a blade.  blades = [(heel xyz, tip xyz), ...] old -> new.  Loaded dry-brush drags along the arc, thin scribbled lines, black drag.  -> premult RGBA layer"""
    from scipy.interpolate import CubicSpline
    r = np.random.default_rng(seed); A = np.array([b[0] for b in blades], float); Bq = np.array([b[1] for b in blades], float); ts = np.linspace(0, 1, len(A)); fa, fb = CubicSpline(ts, A), CubicSpline(ts, Bq); P = lambda u, t: fa(t) * (1 - u) + fb(t) * u
    cw, ck = Canvas(H, W, color=light, layer=True), Canvas(H, W, color=INK, layer=True); wp, bp = Pen(), Pen()
    for k, (cv, cnt) in enumerate(((ck, 9), (cw, 24))):
        PP, WD, OP, DR = [], [], [], []
        for q in range(cnt):
            u = u0 + (1.0 - u0) * r.random() ** .5; t0 = r.uniform(0, .5) * age; t1 = 1. if r.random() < .8 else r.uniform(.8, 1); t = np.linspace(t0, t1, 16); p, z = cam.project(P(u, t)); ok = z > .12
            if ok.sum() < 6: continue
            p = p[ok]; idx = np.linspace(0, len(p) - 1, 14).round().astype(int); PP.append(p[idx][::-1]); WD.append(float(np.clip(cam.f / np.median(z[ok]) * r.uniform(.006, .034), 2, 70)) * w); OP.append(r.uniform(.25, .75)); DR.append(r.uniform(.42, .68) if r.random() < .5 else r.uniform(.7, .95))
        for sh, ix in (('flat', slice(0, None, 2)), ('taper', slice(1, None, 2))):
            if len(PP[ix]): cv.strokes(np.array(PP[ix]), np.array(WD[ix]) * (1.5 if sh == 'taper' else 1), rgb(light if k else INK), op=np.array(OP[ix]), dry=np.array(DR[ix]) * (.8 if sh == 'taper' else 1), seed=seed + k, shape=sh, bristle=.08)
    for q in range(n):
        u = u0 + (1 - u0) * r.random() ** .4; t1 = 1. if r.random() < .5 else r.uniform(.5, 1.); t0 = max(0, t1 - r.uniform(.15, .55) * age); t = np.linspace(t0, t1, 26); p, z = cam.project(P(u, t))
        if z.min() < .12: continue
        sc = float(np.clip(2.2 / np.median(z), .5, 3)); (wp if r.random() < .7 else bp).stroke(p, w * sc * r.choice([.9, 1.4, 2.4, 5.], p=[.4, .35, .18, .07]), r.integers(1 << 30), taper=(.6, .12), press=.4, tremor=.8)
    W_, K_ = wp.raster(H, W), np.maximum(bp.raster(H, W), ck.a * .9) * dark * 2; K_ = np.clip(K_, 0, .9); Lw = np.clip(cw.a * fill, 0, 1); L = rgb(light)
    c = L * (Lw * .62)[..., None]; a = Lw * .62
    c = c * (1 - K_[..., None]) + rgb(INK) * K_[..., None]; a = a * (1 - K_) + K_; c = c * (1 - W_[..., None]) + np.array([1.08, 1.11, 1.16], f32) * W_[..., None]; a = a * (1 - W_) + W_
    return np.dstack([c, a]).astype(f32)


# ------------------------------------------------------------------ the arm: cut, ink, knitting back
CUT_AT = .30
def arm_map(d, m):
    """(H,W) metres along the sword arm from the shoulder (elbow, wrist, then the blade), -1 elsewhere"""
    g = d['g']; part = np.r_[m.S['PART'], -1][g['id']]; ok = np.isin(part, (P_ARMR, P_BLADE)); P = g['p'][ok].astype(float); J = [m.J['S-1'], m.J['E-1'], m.J['W-1']]
    if getattr(m, 'blade_line', None) is not None: J.append(J[2] + unit(J[2] - J[1]) * 1.1)
    best = np.full(len(P), 1e9); s = np.zeros(len(P)); acc = 0.
    for a, b in zip(J[:-1], J[1:]):
        L = np.linalg.norm(b - a); z = (b - a) / L; t = np.clip((P - a) @ z, 0, L); dd = np.linalg.norm(P - (a + t[:, None] * z), axis=1); k = dd < best; best[k] = dd[k]; s[k] = acc + t[k]; acc += L
    out = np.full(g['id'].shape, -1., f32); out[ok] = s; return cv2.resize(out, (W, H), interpolation=cv2.INTER_NEAREST), acc


def sever(full, cut, m, cam, t, seed=0, throw=(-620, -160), spin=1.1, s=None):
    """arm coming off.  t 0..1 after the cut: ink opens along the line, the arm leans away and falls.  -> layers [(premult RGBA)] back to front"""
    s, tot = s if s is not None else arm_map(full, m); gone = (s > CUT_AT * m.sc).astype(f32); gone = cv2.GaussianBlur(cv2.dilate(gone, np.ones((5, 5), np.uint8)), (0, 0), .8); pc = full['layer'] * gone[..., None]
    p = cam.project(np.stack([m.J['S-1'] + unit(m.J['E-1'] - m.J['S-1']) * CUT_AT * m.sc, m.J['E-1']]))[0]; c0 = p[0]; a = spin * t ** 1.3; e = t ** .8
    M = cv2.getRotationMatrix2D((float(p[1][0]), float(p[1][1])), math.degrees(-a), 1.0); M[:, 2] += np.array(throw) * e + np.array([0, 760]) * t * t; pc = cv2.warpAffine(pc, M, (W, H), flags=cv2.INTER_LINEAR)
    r = np.random.default_rng(seed); pen, wp = Pen(), Pen(); ax = unit(p[1] - p[0]); n = np.array([-ax[1], ax[0]]); px = cam.f / cam.cam(m.J['S-1'])[2]; rad = .10 * px * m.sc
    for k in range(46):                                                        # the ink break: strands pulled between stump and arm, then spray
        o = r.uniform(-1, 1); q0 = c0 + n * rad * o; L = rad * r.uniform(.6, 3.4) * (.3 + e * 1.4) * (1 - .4 * abs(o)); dv = ax * L + n * r.normal(0, .25) * L + np.array([0, .4]) * L * t; ctrl = q0 + dv * .5 + n * r.normal(0, .12) * L
        tt = np.linspace(0, 1, 14)[:, None]; C = (1 - tt) ** 2 * q0 + 2 * tt * (1 - tt) * ctrl + tt * tt * (q0 + dv); (pen if r.random() < .8 else wp).stroke(C, rad * .10 * r.uniform(.3, 1.5) * (1 - .55 * t), r.integers(1 << 30), taper=(.02, .85), press=.5, tremor=1.)
    for k in range(70):
        d_ = ax * abs(r.normal(0, 1)) * rad * 3.2 * e + n * r.normal(0, 1) * rad * 1.1 * (.4 + e) + np.array([0, 300]) * t * t; q0 = c0 + d_; rr = rad * .055 * r.random() ** 2 + 1; pen.add(np.array([q0, q0 + unit(d_ + 1e-6) * rr * r.uniform(1, 4)]), rr)
    K, Wm = pen.raster(H, W), wp.raster(H, W) * .9; ink = np.zeros((H, W, 4), f32); ink[..., :3] = rgb(INK) * K[..., None]; ink[..., 3] = K; ink[..., :3] = ink[..., :3] * (1 - Wm[..., None]) + Wm[..., None]; ink[..., 3] = np.maximum(ink[..., 3], Wm)
    return [cut['layer'], pc.astype(f32), ink]


def regrow(full, cut, m, cam, t, seed=0, s=None):
    """ink threads knit the arm back.  t 0..1.  -> premult RGBA of the whole creature"""
    s, tot = s if s is not None else arm_map(full, m); f0 = CUT_AT * m.sc; front = f0 + (tot - f0) * min(1.2, t * 1.25); ws = cv2.resize(wash(540, 960, seed), (W, H)); rev = ((s > 0) & (s < front - .10 * m.sc * (1 + .5 * ws))).astype(f32) if t < 1 else (s > 0).astype(f32); rev = cv2.GaussianBlur(rev, (0, 0), .9)
    if t >= 1: return full['layer']
    base = cut['layer'] * (1 - rev[..., None]) + full['layer'] * rev[..., None]
    r = np.random.default_rng(seed); pen, wp = Pen(), Pen(); J = [m.J['S-1'], m.J['E-1'], m.J['W-1']]; J.append(J[2] + unit(J[2] - J[1]) * 1.15 * m.sc); J = np.array(J); cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(J, axis=0), axis=1))]
    at = lambda q: np.stack([np.interp(q, cum, J[:, k]) for k in range(3)], -1); e1 = unit(np.cross(J[1] - J[0], cam.R[2])); e2 = unit(np.cross(J[1] - J[0], e1))
    for k in range(64):
        q0 = max(f0 * .9, front - r.uniform(.15, .75) * m.sc); q1 = min(tot, front + r.uniform(-.05, .26) * m.sc)
        if q1 < q0 + .05: continue
        q = np.linspace(q0, q1, 40); ph = r.uniform(0, 6.28); tw = r.uniform(5, 16) * r.choice([-1, 1]); R0 = np.interp(q, [0, .3, .6, .75, 1.1, tot], [.12, .12, .085, .10, .07, .04]) * m.sc * r.uniform(.6, 1.25); lead = np.clip((q - front) / .25, 0, 1); R0 = R0 * (1 + 1.2 * lead * r.random()); a = ph + tw * q + r.normal(0, .3) * np.sin(q * 9)
        P3 = at(q) + R0[:, None] * (np.cos(a)[:, None] * e1 + np.sin(a)[:, None] * e2); p = cam.project(P3)[0]; (pen if r.random() < .68 else wp).stroke(p, r.choice([1.2, 2., 3.4, 6.]) * cam.f / cam.cam(J[0])[2] / 420, r.integers(1 << 30), taper=(.25, .6), press=.6, tremor=1.2)
    K, Wm = pen.raster(H, W), wp.raster(H, W) * .9; base[..., :3] = base[..., :3] * (1 - K[..., None]) + rgb(INK) * K[..., None]; base[..., 3] = np.maximum(base[..., 3], K); base[..., :3] = base[..., :3] * (1 - Wm[..., None]) + Wm[..., None] * np.array([1.0, 1.04, 1.1], f32); base[..., 3] = np.maximum(base[..., 3], Wm); return base
