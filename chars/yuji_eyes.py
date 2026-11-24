"""Eyes of curses for c2: eighteen different drawings.  eye(cel, kind, x, y, s, o, look, seed, rot)   s = half width px, o = opening 0..1.15, look = (dx,dy) -1..1"""
import numpy as np, math
from jj.ink import Cel, spl
from jj.kit import outline
from chars.yuji import mixc, rad
A = np.array
K = '#040307'


def _lids(o, up=.62, dn=.50, n=41, skew=0., pw=1.6):
    x = np.linspace(-1, 1, n); e = np.clip(1 - np.abs(x) ** pw, 0, 1) ** .85
    return np.c_[x, (up * o + .0) * e + skew * x * .25], np.c_[x, -dn * o * e + skew * x * .25]


class _T:
    def __init__(s, c, x, y, r, rot, seed): s.c, s.x, s.y, s.r, s.rot, s.seed = c, x, y, r, rot, seed; s.w = max(1.2, r * .045)
    def __call__(s, P):
        P = np.atleast_2d(np.asarray(P, float)); ca, sa = math.cos(s.rot), math.sin(s.rot); return np.c_[s.x + (P[:, 0] * ca + P[:, 1] * sa) * s.r, s.y + (P[:, 0] * sa - P[:, 1] * ca) * s.r]
    def disc(s, cx, cy, rx, ry=None, n=40): a = np.linspace(0, 2 * np.pi, n, endpoint=False); return s(np.c_[cx + np.cos(a) * rx, cy + np.sin(a) * (ry or rx)])
    def fill(s, P, col, a=1.): s.c.fill(s(P), col, a)
    def ln(s, P, w, col, **k): P = s(P); s.c.line(spl(P, n=8) if 2 < len(P) < 8 else P, max(.8, s.w * w), col, **dict(dict(taper=(.2, .4), press=.3, seed=s.seed), **k))


def _frame(T, U, L, skin, white, w=2.2):
    """lids: dark flesh round the opening"""
    ev = (1 - np.abs(U[:, 0]) ** 4)[:, None]; big = np.vstack([U * A([1.22, 1.]) + A([0, .17]) * ev, (L * A([1.22, 1.]) - A([0, .13]) * ev)[::-1]]); T.fill(big, skin, .85)
    P = np.vstack([U, L[::-1]]); T.fill(P, white); return P


def _edge(T, U, L, w=2.4, col=K):
    T.ln(U, w, col, taper=(.08, .08)); T.ln(L, w * .62, col, taper=(.2, .2))


def eye(c, kind, x, y, s, o, look=(0, 0), seed=1, rot=0., f=0):
    if o < .03: return
    T = _T(c, x, y, s, rot, seed); r = np.random.default_rng(seed * 7 + 1); lx, ly = look[0] * .34, look[1] * .20; n = 41
    if kind == 'slit':
        U, L = _lids(o, .58, .46, skew=.25); P = _frame(T, U, L, '#1b2016', '#d8c23a'); c.clip(T(P)); c.fill(T.disc(lx, ly, .50), '#b48a1c'); c.fill(T.disc(lx, ly, .40), '#e8d050')
        c.fill(T(A([(lx, ly + .52), (lx + .075, ly), (lx, ly - .52), (lx - .075, ly)])), K); c.unclip(); _edge(T, U, L)
        for k in range(5): T.ln([(-.7 + k * .32, .74 * o), (-.66 + k * .32, .90 * o), (-.58 + k * .32, 1.02 * o)], 1., '#3a4028')
    elif kind == 'compound':
        U, L = _lids(o, .70, .70, pw=2.); P = _frame(T, U, L, '#0f1a18', '#16352c'); c.clip(T(P))
        for i in range(-6, 7):
            for j in range(-5, 6):
                hx, hy = i * .165 + (j % 2) * .082, j * .142; d = math.hypot(hx - lx, hy - ly); a = np.linspace(0, 2 * np.pi, 6, endpoint=False) + .52
                T.fill(np.c_[hx + np.cos(a) * .082, hy + np.sin(a) * .082], mixc('#9dffcf' if (i + 2 * j) % 5 else '#f2fff0', '#0b4a3a', min(1, d * 1.05)))
        c.unclip(); _edge(T, U, L, 2.6)
    elif kind == 'stitched':
        U, L = _lids(o, .50, .42); P = _frame(T, U, L, '#2a2028', '#e6dfd2'); c.clip(T(P)); c.fill(T.disc(lx, ly, .30), '#50607a'); c.fill(T.disc(lx, ly, .13), K); c.unclip(); _edge(T, U, L)
        for k in range(6):
            q = -.72 + k * .29; i = int((q + 1) / 2 * (n - 1)); a, b = U[i], L[i]; br = (k * 7 + seed) % 3 == 0
            if br: T.ln([a + A([0, .14]), a + A([.03, -.02]), a + A([.05, -.16])], 1.6, '#0a0608'); T.ln([b - A([0, .12]), b + A([-.04, .10])], 1.6, '#0a0608')
            else: T.ln([a + A([-.03, .16]), (a + b) / 2, b + A([.03, -.14])], 1.7, '#0a0608', taper=(.05, .05))
            T.fill(A([a + A([-.05, .18]), a + A([.01, .20]), a + A([.0, .13])]), '#0a0608')
    elif kind == 'human':
        U, L = _lids(o, .56, .46); P = _frame(T, U, L, '#2c1f20', '#efe8e0'); c.clip(T(P))
        for k in range(12): a = r.uniform(0, 6.28); T.ln([(math.cos(a) * 1.0, math.sin(a) * .6), (math.cos(a + .1) * .7, math.sin(a + .1) * .42), (math.cos(a) * .46, math.sin(a) * .3)], .7, '#b5303a', taper=(.1, .8))
        c.fill(T.disc(lx, ly, .34), '#2d4a5c'); c.fill(T.disc(lx, ly, .29), '#7fa6b4'); c.fill(T.disc(lx, ly, .085), K); c.unclip(); _edge(T, U, L)
        for k in range(9): q = -.8 + k * .2; i = int((q + 1) / 2 * 40); T.ln([U[i], U[i] + A([q * .10, .14]), U[i] + A([q * .22, .24])], 1.1, K, taper=(.02, .8))
    elif kind == 'goat':
        U, L = _lids(o, .60, .48, skew=-.2); P = _frame(T, U, L, '#241a12', '#c98f34'); c.clip(T(P)); c.fill(T.disc(lx, ly, .56, .48), '#e0b250')
        c.fill(T(spl([(lx - .40, ly + .10), (lx, ly + .13), (lx + .40, ly + .10), (lx + .42, ly - .08), (lx, ly - .11), (lx - .42, ly - .08)], True, n=5)), K); c.unclip(); _edge(T, U, L)
    elif kind == 'ring':
        for k in range(7):
            a = k / 7 * 2 * math.pi + .3; q = _T(c, *T(A([[math.cos(a) * .72, math.sin(a) * .62]]))[0], s * .24, rot + a, seed + k); oo = min(1., max(0., o * 1.5 - k * .08)); U, L = _lids(oo, .62, .55); P = np.vstack([U, L[::-1]]); q.fill(P, '#e9d6c8'); c.fill(q.disc(lx, ly, .30), '#8c1e28'); c.fill(q.disc(lx, ly, .13), K); q.ln(U, 2.4, K); q.ln(L, 1.6, K)
    elif kind == 'weeping':
        U, L = _lids(o, .48, .62, skew=.1); P = _frame(T, U, L, '#16161c', '#cfd4d8'); c.clip(T(P)); c.fill(T.disc(lx, ly, .36), '#0c0c12'); c.fill(T.disc(lx - .1, ly + .1, .07), '#ffffff'); c.unclip(); _edge(T, U, L)
        for k, q in enumerate((-.35, .05, .42)):
            l = min(1., o) * (.9 + .5 * ((k * 3 + seed) % 3) + .05 * math.sin(f * .2 + k)); b = L[int((q + 1) / 2 * 40)]; T.fill(spl([b + A([-.06, .04]), b + A([-.045, -l * .5]), b + A([-.07, -l * .92]), b + A([0, -l]), b + A([.07, -l * .92]), b + A([.04, -l * .5]), b + A([.06, .04])], True, n=5), '#08060a')
    elif kind == 'fish':
        rr = .70 * min(1., o); c.fill(T.disc(0, 0, rr * 1.15), '#20262c'); c.fill(T.disc(0, 0, rr), '#c9d6d8'); c.fill(T.disc(lx * .6, ly * .6, rr * .74), '#e8eeea'); c.fill(T.disc(lx, ly, rr * .42), K); c.fill(T.disc(lx - .12, ly + .14, rr * .10), '#ffffff'); outline(c, T.disc(0, 0, rr * 1.02), T.w * 2., K, seed)
    elif kind == 'spiral':
        U, L = _lids(o, .58, .58, pw=2.2); P = _frame(T, U, L, '#1e1426', '#d9c9ec'); c.clip(T(P)); a = np.linspace(0, 5 * 2 * math.pi, 160); rr = a / a[-1] * .62; T.ln(np.c_[lx + np.cos(a + f * .05) * rr, ly + np.sin(a + f * .05) * rr], 1.9, '#4a1a6e', taper=(.05, .05)); c.unclip(); _edge(T, U, L)
    elif kind == 'double':
        U, L = _lids(o, .50, .44); P = _frame(T, U, L, '#22161a', '#eadfcf'); c.clip(T(P))
        for q in (-.34, .34): c.fill(T.disc(q + lx * .5, ly, .25), '#a8302a'); c.fill(T.disc(q + lx * .5, ly, .10), K)
        c.unclip(); _edge(T, U, L)
    elif kind == 'spider':
        for k, (a, b, q) in enumerate(((-.30, .10, .22), (.30, .10, .22), (-.66, .20, .13), (.66, .20, .13), (-.22, .50, .10), (.22, .50, .10), (-.52, -.16, .08), (.52, -.16, .08))):
            q *= min(1., max(.0, o * 1.4 - k * .06)); c.fill(T.disc(a, b, q * 1.2), '#180a0c'); c.fill(T.disc(a, b, q), '#7a0f16'); c.fill(T.disc(a - q * .3, b + q * .35, q * .34), '#ffd0c0')
    elif kind == 'crescent':
        x_ = np.linspace(-1, 1, 41); e = np.clip(1 - np.abs(x_) ** 2, 0, 1) ** .7; U = np.c_[x_, .34 * e * min(1, o) - .0]; L = np.c_[x_, .34 * e - .22 * e * o]; T.fill(np.vstack([U * A([1.15, 1.25]), (L * A([1.15, .6]))[::-1]]), '#1a1016'); T.fill(np.vstack([U, L[::-1]]), '#f6e6c8'); T.ln(U, 2.4, K)
    elif kind == 'toothed':
        U, L = _lids(o, .56, .50); P = _frame(T, U, L, '#3a1016', '#c9b7a4'); c.clip(T(P)); c.fill(T.disc(lx, ly, .34), '#40100c'); c.fill(T.disc(lx, ly, .22), '#e24a22'); c.fill(T.disc(lx, ly, .09), K); c.unclip()
        for sg, E in ((1, U), (-1, L)):
            for k in range(8): i = 3 + k * 5; a = E[min(40, i)]; b = E[min(40, i + 4)]; m = (a + b) / 2; T.fill(A([a, b, m - A([0, sg * .24 * min(1, o)])]), '#efe6d2'); T.ln([a, m - A([0, sg * .24 * min(1, o)]), b], .8, K)
        _edge(T, U, L)
    elif kind == 'milky':
        U, L = _lids(o, .46, .40, skew=-.3); P = _frame(T, U, L, '#1c2024', '#b9c2c4'); c.clip(T(P)); c.fill(T.disc(lx, ly, .36), '#dfe8e8'); c.fill(T.disc(lx + .04, ly - .03, .24), '#f4f8f6'); c.unclip(); _edge(T, U, L)
    elif kind == 'tri':
        k = min(1.1, o); P = A([(-.9, -.40 * k), (0, .78 * k), (.9, -.40 * k), (0, -.22 * k)]); T.fill(P * 1.18, '#1c070a'); T.fill(P, '#d81e2a'); c.clip(T(P)); c.fill(T.disc(lx * .6, .06 + ly, .24), '#ffcf9a'); c.fill(T(A([(lx * .6, .28 + ly), (lx * .6 + .06, .05 + ly), (lx * .6, -.2 + ly), (lx * .6 - .06, .05 + ly)])), K); c.unclip(); outline(c, T(P), T.w * 1.6, K, seed)
    elif kind == 'star':
        U, L = _lids(o, .60, .60, pw=2.); P = _frame(T, U, L, '#101a2a', '#bfd4ee'); c.clip(T(P)); c.fill(T.disc(lx, ly, .42), '#2f5a9e'); a = np.arange(10) * math.pi / 5 + .3; R = np.where(np.arange(10) % 2 == 0, .34, .12); T.fill(np.c_[lx + np.cos(a) * R, ly + np.sin(a) * R], K); c.unclip(); _edge(T, U, L)
    elif kind == 'points':
        for q in (-.4, .4): rad(c, None, T(A([[q, 0]]))[0], s * .42, '#ffe8b0', .55 * min(1, o)); c.fill(T.disc(q, 0, .13 * min(1.1, o), .10 * min(1.1, o)), '#fff6dc')
    elif kind == 'worm':
        U, L = _lids(o, .40, .40, pw=1.2); P = _frame(T, U, L, '#2a0f1e', '#f0c9d4'); c.clip(T(P)); c.fill(T.disc(lx, ly, .22, .34), '#c23a78'); c.fill(T.disc(lx, ly, .10, .22), K); c.unclip(); _edge(T, U, L)
        for k in range(7): q = -.8 + k * .27; i = int((q + 1) / 2 * 40); T.ln([U[i], U[i] + A([.05 * math.sin(f * .3 + k), .22]), U[i] + A([.14 * math.sin(f * .3 + k + 1), .44])], 1.5, '#5a1836', taper=(.02, .7)); T.ln([L[i], L[i] - A([.05 * math.sin(f * .3 + k + 2), .20]), L[i] - A([.14 * math.sin(f * .3 + k), .36])], 1.3, '#5a1836', taper=(.02, .7))
