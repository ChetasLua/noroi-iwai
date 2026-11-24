"""d3  TODO: tears of joy; he claps, and everyone has changed places.  Four claps, four jokes, then back to back."""
import numpy as np, cv2, math, os
from jj import W, H, comp, world
from jj.paint import rgb, f32, smooth, ramp, grid, Canvas, arcs
from jj.ink import Cel, spl, resample
from jj.comp import lerp_keys, keyed
from jj.kit import para, smear, flare, shake, zoomblur, speed_lines, focus_lines, splatter, outline, hatch
from jj.cam import Camera3D
from jj import hands as HD, fx
from jj.fx_flash import radial_wedges, ragged_ring
from chars.ntkit import rim, hexmix, qb
from chars.todo import Todo, TBody, hand3, PAL as TP, EXPR as TE
from chars.yuji import Yuji, YHead
from chars import nt_keys as K

VER = 7
U, DY = .86, 60                                          # the locked ground-level frame: pulled back so there is wet street under all three
M_, N_, N3_, NC_, F_, H_, JL_, JR_ = (-1.5, 1.6), (.80, -.40), (.80, -.45), (.60, -.02), (-6.6, 9.), (-1.4, 1.5), (-2.9, 1.35), (.25, 1.25)
LAMP = (1258, 150)
CLAP = (27, 36, 45, 54)
CP = dict(skin='#7d7e66', skinS='#594d6b', skinD='#2d2440', skinH='#b3af8c', line='#0c090f', sline='#291b31', hair='#15101a', hairS='#0a080d', hairH='#6a6480', white='#e6d8b0', whiteS='#9c8a68', tear='#dff4ff', tearS='#8fc8e8')
BONE, BONES, GUM, MAW = '#e3d3a6', '#94815c', '#7a2f48', '#1b0913'


def warp(P, hit):
    """push a contour in where a fist lands: hit = (centre, direction*amount, radius)"""
    P = np.asarray(P, float)
    if hit is None: return P
    c, d, r = hit; k = np.exp(-(np.hypot(*(P - c).T) / r) ** 2); return P + np.asarray(d, float)[None] * k[:, None]


def curse_teeth(cel, M, up, lo, lw, seed=0, n=15, big=1., gap=()):
    """a mouth between two edge polylines (local): gums, a back row, a front row of wrong teeth (molars, fangs, pegs, gaps)"""
    r = np.random.default_rng(seed); ln = CP['line']; mo = M(np.vstack([up, lo[::-1]])); cel.fill(mo, MAW); cel.clip(mo)
    for back in (1, 0):
        for row, ed, sg in ((0, up, 1.), (1, lo, -1.)):
            m = n + (4 if back else 0)
            if not back: cel.fill(M(np.vstack([ed - [0, sg * .2], (ed + [0, sg * (.045 + .02 * math.sin(row + 1))])[::-1]])), GUM)
            w_ = r.uniform(.5, 1.6, m); w_ = np.r_[0, np.cumsum(w_)]; w_ /= w_[-1]
            for j in range(m):
                if not back and j in gap and row == 0: continue
                i0, i1 = int((w_[j] + .004) * (len(ed) - 1)), int((w_[j + 1] - .004) * (len(ed) - 1))
                if i1 - i0 < 2: continue
                A_, B_ = ed[i0], ed[i1]; d = B_ - A_; L_ = np.hypot(*d); nn = np.array([-d[1], d[0]]) / (L_ + 1e-6); nn = nn * np.sign(nn[1] * sg + 1e-9); h = r.uniform(.09, .20) * big * (1.1 if row == 0 else 1.) * (1.25 if back else 1.); kind = r.random(); lean = r.normal(0, .035)
                if kind < .40: T = [A_ - nn * .05, A_ + nn * h * .72 + [lean, 0], A_ + d * .5 + nn * h * .82 + [lean, 0], B_ + nn * h * .70 + [lean, 0], B_ - nn * .05]; cr = (0, 1, 3, 4)
                elif kind < .80: T = [A_ - nn * .05, A_ + d * .12 + nn * h * .6, A_ + d * (.5 + r.normal(0, .18)) + nn * h * 1.45 + [lean * 1.6, 0], B_ - d * .12 + nn * h * .55, B_ - nn * .05]; cr = (0, 2, 4)
                else: T = [A_ - nn * .05, A_ + d * .2 + nn * h * .95 + [lean, 0], A_ + d * .62 + nn * h * 1.05 + [lean, 0], B_ - d * .05 + nn * h * .4, B_ - nn * .05]; cr = (0, 4)
                q = M(spl(np.array(T), True, corners=cr, n=5)); cel.fill(q, hexmix(BONE, BONES, .55) if back else hexmix(BONE, '#b89a5a', .5 * r.random())); cel.clip(q); cel.fill(M([A_ + d * .62, B_ + d * .1, B_ + d * .1 + nn * h * 2, A_ + d * .55 + nn * h * 2]), BONES if not back else '#5e4f42'); cel.unclip(); outline(cel, q, lw * (.36 if not back else .24), ln, j)
    cel.unclip(); cel.line(M(up), lw * .95, ln, taper=(.06, .06), press=.4); cel.line(M(lo), lw * .85, ln, taper=(.06, .06), press=.4)
    for k in range(9):                                           # lips crease around it
        t = r.uniform(.05, .95); e, sg = (up, -1) if k % 2 else (lo, 1); i = int(t * (len(e) - 1)); p_ = e[i]; cel.line(M(spl(np.array([p_ + [0, sg * .03], p_ + [r.normal(0, .03), sg * .09], p_ + [r.normal(0, .05), sg * r.uniform(.12, .20)]]), n=5)), lw * .32, CP['sline'], taper=(.1, .75))


def curse_eye(cel, M, x, y, R, lw, mood='grin', look=(0., 0.), tilt=0., seed=0):
    """a small staring human eye in folds of skin"""
    r = np.random.default_rng(seed + 60); ct, st = math.cos(tilt), math.sin(tilt); E = lambda P: M(np.array([(x + (u * ct - v * st) * R, y + (u * st + v * ct) * R) for u, v in P])); op = dict(grin=.66, shock=1.18, hit=1.05)[mood] * (1 + .12 * r.normal()); ln = CP['line']
    cel.fill(E(spl(np.array([(-1.75, .05), (-1.0, -1.05), (.1, -1.35), (1.2, -.95), (1.75, .1), (1.0, 1.15), (0, 1.40), (-1.1, 1.05)]), True, n=6)), CP['skinS'], .9); cel.fill(E(spl(np.array([(-1.35, .1), (-.7, -.82), (.2, -.98), (1.0, -.62), (1.40, .05), (.7, .78), (-.2, .92), (-.9, .62)]), True, n=6)), CP['skinD'], .85)
    up = spl(np.array([(-1.12, .10), (-.55, -.72 * op), (.10, -.88 * op), (.72, -.62 * op), (1.12, -.02)]), n=8); dn = spl(np.array([(-1.12, .10), (-.5, .60 * op), (.12, .70 * op), (.70, .50 * op), (1.12, -.02)]), n=8); pol = E(np.vstack([up, dn[::-1]])); cel.fill(pol, '#efe8d8'); cel.clip(pol)
    for k in range(6): a0 = r.uniform(0, 6.283); cel.line(E(np.array([(1.1 * math.cos(a0), .9 * math.sin(a0)), (.75 * math.cos(a0 + .12), .6 * math.sin(a0 + .12)), (.5 * math.cos(a0), .42 * math.sin(a0))])), lw * .20, '#b4484a', taper=(0, .7))
    an = np.linspace(0, 6.283, 22)[:-1]; ir = .40 if mood == 'grin' else .30; lx, ly = look; cel.fill(E(np.c_[lx + ir * np.cos(an), ly + ir * np.sin(an)]), '#5c3a26'); cel.fill(E(np.c_[lx + ir * .55 * np.cos(an), ly + .12 + ir * .45 * np.sin(an)]), '#94653a'); cel.fill(E(np.c_[lx + ir * .45 * np.cos(an), ly + ir * .45 * np.sin(an)]), '#080508'); cel.line(E(np.c_[lx + ir * np.cos(an), ly + ir * np.sin(an)]), lw * .28, ln, taper=(0, 0)); cel.fill(E(np.vstack([up, up[::-1] + [0, .26]])), '#6a5a70', .5); cel.fill(E(np.c_[lx - .12 + .09 * np.cos(an), ly - .14 + .09 * np.sin(an)]), '#ffffff')
    cel.unclip(); cel.line(E(up), lw * .72, ln, taper=(.12, .12), press=.4); cel.line(E(dn), lw * .42, ln, taper=(.2, .2)); cel.line(E(up * [1.12, 1.22] + [0, -.24]), lw * .34, CP['sline'], taper=(.3, .3)); cel.line(E(dn * [1.10, 1.25] + [0, .22]), lw * .30, CP['sline'], taper=(.35, .35)); cel.line(E(dn * [.95, 1.35] + [0, .42]), lw * .22, CP['sline'], taper=(.4, .5))


def warts(cel, M, pts, lw, seed=0):
    r = np.random.default_rng(seed)
    for x, y, R in pts:
        an = np.linspace(0, 6.283, 11)[:-1]; q = M(spl(np.c_[x + R * np.cos(an) * (1 + r.normal(0, .10, 10)), y + R * .9 * np.sin(an) * (1 + r.normal(0, .10, 10))], True, n=4)); cel.fill(q, hexmix(CP['skin'], '#8a6a66', .55)); cel.clip(q); cel.fill(M([(x + R * .1, y - R * 1.2), (x + R * 1.3, y - R * 1.2), (x + R * 1.3, y + R * 1.3), (x - R * .6, y + R * 1.3), (x + R * .35, y + R * .25)]), CP['skinS']); cel.unclip(); cel.fill(M(np.c_[x - R * .35 + R * .18 * np.cos(an), y - R * .35 + R * .15 * np.sin(an)]), CP['skinH']); outline(cel, q, lw * .34, CP['line'], int(x * 100))


def wet_hair(cel, M, roots, lw, f=0, seed=0):
    """lank tufts: several strands stuck together, hanging, dripping"""
    r = np.random.default_rng(seed + 3)
    for (x, y, dx, L_, n) in roots:
        for k in range(n):
            x0, y0 = x + r.normal(0, .05), y + r.normal(0, .03); l = L_ * r.uniform(.55, 1.15); sw = .04 * math.sin(f * .25 + k + x * 5); P = spl(np.array([(x0, y0), (x0 + dx * .30 + r.normal(0, .02), y0 - .06 + l * .10), (x0 + dx * .75 + sw, y0 + l * .48), (x0 + dx * .9 + sw * 1.6 + r.normal(0, .04), y0 + l)]), n=9); w_ = lw * r.uniform(.7, 1.9)
            cel.line(M(P), w_, CP['hair'], taper=(.08, .75), press=.3, seed=k); cel.line(M(P[3:-6]), w_ * .25, CP['hairH'], taper=(.5, .5), seed=k)
            if k % 4 == 0: d = M(P[-1:])[0]; R = lw * 1.1; cel.fill(spl(np.array([d + [0, -R * .6], d + [R * .7, R * .9], d + [0, R * 1.7], d + [-R * .7, R * .9]]), True, corners=(0,), n=5), '#a9c0cc', .9)


def curse_head(cel, cx, cy, s, rot=0., mood='grin', hit=None, lw=6., lit=-1, f=0, body=True, seed=0, legs=False, look=None, lean=0.):
    """the hulking curse, close.  Nothing on it matches: a lumpy head sunk between one low shoulder and one hump, human eyes at every height, two mouths full of the wrong teeth"""
    cr, sr = math.cos(rot), math.sin(rot); ln = CP['line']; r = np.random.default_rng(seed + 9)
    M = lambda P: (lambda q: np.c_[cx + (q[:, 0] * cr - q[:, 1] * sr) * s, cy + (q[:, 0] * sr + q[:, 1] * cr) * s])(warp(P, hit)); S_ = lambda P, c=(), n=9: spl(np.array(P, float), True, corners=c, n=n)
    M0 = lambda P: (lambda q: np.c_[cx + (q[:, 0] - lean * np.clip(q[:, 1] - .4, 0, 9)) * s, cy + q[:, 1] * s])(np.asarray(P, float))                                     # the body does not turn with the head
    if legs:
        for L_, big in (([(-1.75, 3.6), (-1.95, 4.4), (-1.62, 5.05), (-1.70, 5.55), (-2.15, 5.85), (-1.95, 6.02), (-.75, 6.02), (-.62, 5.75), (-.86, 5.3), (-.72, 4.5), (-.35, 3.7)], 0), ([(.05, 3.6), (.30, 4.5), (.62, 5.1), (.52, 5.6), (.25, 5.9), (.45, 6.10), (2.45, 6.10), (2.72, 5.85), (2.30, 5.45), (2.42, 4.6), (2.35, 3.5)], 1)):
            q = M0(S_(L_, (0, 10), 7)); cel.fill(q, CP['skin']); cel.clip(q); x0 = L_[0][0]; x1 = L_[-1][0]; cel.fill(M0([(x0 + (x1 - x0) * .55, 3.4), (x0 + (x1 - x0) * .42, 4.4), (x0 + (x1 - x0) * .60, 5.0), (x0 + (x1 - x0) * .45, 6.3), (x1 + 1., 6.3), (x1 + 1., 3.4)] if lit < 0 else [(x0 + (x1 - x0) * .45, 3.4), (x0 + (x1 - x0) * .55, 6.3), (x0 - 1., 6.3), (x0 - 1., 3.4)]), CP['skinS']); cel.fill(M0([(x0 - .6, 3.4), (x1 + .6, 3.4), (x1 + .4, 3.95), ((x0 + x1) / 2, 4.15), (x0 - .4, 3.95)]), CP['skinD'], .9); cel.unclip(); outline(cel, q, lw * .9, ln, 5 + big, press=.5)
            for k in range(4):
                tx = L_[5][0] + .16 + k * (.28 if not big else .50); T = M0(S_([(tx, 5.80), (tx + .10, 5.72), (tx + .22, 5.80), (tx + .20, 6.02), (tx + .02, 6.02)], (3, 4), 5)); cel.fill(T, '#cbb98c'); outline(cel, T, lw * .34, ln, k)
            hatch(cel, M0([((x0 + x1) / 2, 4.0), (x1, 4.0), (x1 - .1, 5.3), ((x0 + x1) / 2 + .1, 5.2)]), 1.0, s * .09, lw * .26, ln, 4, length=s * .22)
    if body:
        B = S_([(-.70, .55), (-1.25, .98), (-1.70, 1.02), (-2.15, 1.34), (-2.28, 1.80), (-2.02, 2.30), (-2.22, 2.90), (-2.05, 4.0), (-2.4, 6.2), (2.6, 6.2), (2.75, 4.4), (3.05, 3.2), (2.90, 2.2), (3.22, 1.30), (3.05, .42), (2.52, -.06), (1.92, -.02), (1.62, .28), (1.18, .20), (.80, .52)]) if not legs else S_([(-.70, .55), (-1.25, .98), (-1.70, 1.02), (-2.15, 1.34), (-2.28, 1.80), (-2.02, 2.30), (-2.22, 2.90), (-2.05, 4.0), (-1.2, 4.35), (.2, 4.25), (1.4, 4.45), (2.5, 4.2), (2.85, 3.6), (3.05, 3.0), (2.90, 2.2), (3.22, 1.30), (3.05, .42), (2.52, -.06), (1.92, -.02), (1.62, .28), (1.18, .20), (.80, .52)]); q = M0(B); cel.fill(q, CP['skin']); cel.clip(q)
        cel.fill(M0([(lit * -.1, .9), (lit * -.6, 1.4), (lit * -.35, 2.0), (lit * -.95, 2.6), (lit * -.6, 3.4), (lit * -1.1, 4.4), (lit * -.8, 6.4), (lit * -4, 6.4), (lit * -4, -.4), (lit * -2.2, -.3), (lit * -1.7, .6)]), CP['skinS']); cel.fill(M0([(lit * -1.7, 1.2), (lit * -2.2, 1.9), (lit * -1.8, 2.7), (lit * -2.3, 3.6), (lit * -2.0, 6.4), (lit * -4, 6.4), (lit * -4, .6)]), CP['skinD'], .8)
        for pp, al in (([(-1.2, 2.6), (-.7, 2.4), (-.4, 2.9), (-.8, 3.3), (-1.3, 3.1)], .5), ([(1.9, .5), (2.4, .35), (2.7, .8), (2.2, 1.1)], .45), ([(.2, 4.0), (.9, 3.8), (1.1, 4.4), (.4, 4.7)], .5)): cel.fill(M0(S_(pp)), '#7a5a74', al)
        # neck rolls: three folds of skin under the jaw
        for k in range(4): y0 = .92 + .20 * k; cel.fill(M0(S_([(-1.2 - .12 * k, y0 - .10), (-.5, y0 + .30), (.2, y0 + .26), (1.0 + .10 * k, y0 - .05), (.3, y0 + .42), (-.5, y0 + .46)], (0, 3))), CP['skinD' if k % 2 else 'skinS'], .85); cel.line(M0(spl(np.array([(-1.25 - .12 * k, y0 - .12), (-.5, y0 + .30), (.2, y0 + .26), (1.05 + .10 * k, y0 - .07)]), n=9)), lw * .62, ln, taper=(.15, .2), press=.5, seed=k)
        cel.unclip(); outline(cel, q, lw, ln, 3, press=.55)
        # the second mouth, across the chest, crooked
        op = dict(grin=.10, shock=.26, hit=.20)[mood]; u2 = spl(np.array([(-1.35, 2.30), (-.80, 2.16 - op * .4), (-.1, 2.36 - op * .6), (.6, 2.18 - op * .5), (1.25, 2.50)]), n=10); l2 = spl(np.array([(-1.35, 2.30), (-.75, 2.58 + op * .8), (0, 2.74 + op), (.7, 2.62 + op * .7), (1.25, 2.50)]), n=10); curse_teeth(cel, M0, u2, l2, lw * .9, seed + 31, 17, .85, (4, 9, 10))
        for (x, y, R, tl) in ((1.95, 1.05, .115, .3), (2.42, 1.55, .07, -.5), (-1.62, 1.62, .085, -.2)): curse_eye(cel, M0, x, y, R, lw * .8, mood, (.2 * math.sin(f * .3 + x), .05), tl, int(x * 10) + 50)
        warts(cel, M0, [(2.55, .62, .075), (2.75, .86, .05), (2.35, .52, .04), (-1.9, 1.4, .06), (-2.05, 1.62, .04), (1.4, 3.2, .07), (1.62, 3.38, .045), (-.9, 3.5, .06), (2.6, 2.1, .06), (2.72, 2.3, .04)], lw, 3)
        wet_hair(cel, M0, [(2.35, .0, .10, .75, 7), (1.85, .02, -.06, .5, 5), (-1.85, 1.16, -.10, .62, 6), (.5, 3.1, .02, .5, 4)], lw * .7, f, 5)
        for pp in ([(-1.7, 1.15), (-1.45, 1.75), (-1.6, 2.2)], [(1.5, .5), (1.35, 1.3), (1.75, 2.0)], [(2.9, 1.4), (2.4, 1.95), (2.0, 2.05)], [(-1.8, 3.2), (-.9, 3.5), (.2, 3.35)], [(-1.6, 3.9), (-.4, 4.2), (1.2, 3.95)], [(2.6, 3.0), (2.0, 3.4), (1.3, 3.3)]): cel.line(M0(spl(np.array(pp), n=8)), lw * .48, CP['sline'], taper=(.15, .65), press=.4)
        for rg, an_, gp in (([(1.2, .7), (1.75, .4), (1.9, 1.4), (1.5, 2.2), (1.15, 1.6)], 1.1, .075), ([(2.3, 1.9), (2.9, 1.6), (2.9, 3.2), (2.3, 3.3)], .8, .085), ([(-2.1, 2.2), (-1.5, 2.0), (-1.6, 3.2), (-2.1, 3.4)], 2.1, .08), ([(-.8, 2.85), (.6, 2.95), (.4, 3.2), (-.6, 3.15)], 1.5, .07)): hatch(cel, M0(rg), an_, s * gp, lw * .26, CP['line'], int(an_ * 10), length=s * .20, curve=.06)
    D = S_([(-.78, .90), (-1.08, .60), (-1.16, .18), (-1.00, -.16), (-1.22, -.50), (-1.06, -.98), (-.64, -1.18), (-.30, -1.00), (-.04, -1.36), (.42, -1.46), (.80, -1.18), (.84, -.82), (1.12, -.58), (1.02, -.16), (1.22, .24), (1.02, .68), (.74, .88), (.54, 1.24), (.16, 1.10), (-.14, 1.36), (-.46, 1.08)]); q = M(D); cel.fill(q, CP['skin']); cel.clip(q)
    cel.fill(M([(lit * -.05, -1.6), (lit * -.30, -1.0), (lit * -.12, -.62), (lit * -.46, -.34), (lit * -.28, .0), (lit * -.60, .30), (lit * -.40, .62), (lit * -.62, .90), (lit * -.2, 1.5), (lit * -1.8, 1.5), (lit * -1.8, -1.6)]), CP['skinS']); cel.fill(M([(lit * -.62, -1.2), (lit * -.80, -.7), (lit * -.70, -.2), (lit * -.94, .2), (lit * -.78, .7), (lit * -.95, 1.4), (lit * -1.8, 1.4), (lit * -1.8, -1.2)]), CP['skinD'], .8)
    for pp, c_, al in (([(-.95, -.75), (-.6, -.92), (-.42, -.62), (-.72, -.42)], '#7a5a74', .55), ([(.15, -1.2), (.5, -1.28), (.62, -.98), (.28, -.9)], '#8a6a66', .5), ([(-.5, .0), (-.2, -.06), (-.15, .12), (-.45, .16)], '#7a5a74', .4)): cel.fill(M(S_(pp)), c_, al)
    cel.fill(M([(lit * .74, -1.02), (lit * .50, -1.08), (lit * .40, -.94), (lit * .62, -.84)]), CP['skinH']); cel.fill(M([(lit * 1.02, .05), (lit * .90, .22), (lit * .98, .46), (lit * 1.08, .3)]), CP['skinH'])
    cel.unclip()
    # ---- the long mouth, slanted, folded at one end
    k = dict(grin=0., shock=.42, hit=.55)[mood]; u = spl(np.array([(-1.00, .16 - .04 * k), (-.62, .22 - .30 * k), (-.16, .42 - .42 * k), (.34, .30 - .34 * k), (.72, .42 - .20 * k), (1.06, .22)]), n=10); l = spl(np.array([(-1.00, .16 - .04 * k), (-.70, .52 + .22 * k), (-.22, .74 + .36 * k), (.28, .68 + .40 * k), (.74, .66 + .16 * k), (1.06, .22)]), n=10); curse_teeth(cel, M, u, l, lw, seed + 5, 16, 1.0, (5, 6, 11) if mood == 'hit' else (6,))
    look = look or dict(grin=(.06, .04), shock=(0., 0.), hit=(-.16, .14))[mood]
    for j, (x, y, R, tl) in enumerate(((-.50, -.36, .165, .12), (.60, -.30, .215, -.10), (.26, -.74, .115, .25), (-.06, -.10, .080, -.3), (-.86, -.04, .065, .5), (.04, -1.06, .062, -.15), (-.70, -.82, .085, -.35), (.92, .02, .055, .4))): curse_eye(cel, M, x, y, R, lw * (1. if R > .1 else .7), mood, (look[0] + .10 * math.sin(j * 2.1), look[1] + .06 * math.cos(j * 1.7)), tl, j)
    outline(cel, q, lw * 1.05, ln, 7, press=.6)
    warts(cel, M, [(-.95, -.62, .055), (-.84, -.52, .035), (-1.0, .42, .05), (.82, .52, .045), (.92, -.44, .06), (.80, -.58, .035), (-.30, -.86, .045), (.46, -1.10, .04), (-.30, 1.02, .05), (.34, 1.0, .04)], lw, 7)
    for pp in ([(-.30, -1.02), (-.34, -.74), (-.22, -.54)], [(.84, -.84), (.66, -.66), (.62, -.52)], [(-1.02, -.16), (-.82, -.22), (-.66, -.16)], [(1.02, -.16), (.88, -.06), (.80, .08)], [(-.46, 1.06), (-.36, .92), (-.40, .80)], [(.16, 1.10), (.10, .96), (.16, .84)], [(-.22, -.30), (-.02, -.36), (.18, -.28)], [(-.30, .10), (-.04, .16), (.2, .08)]): cel.line(M(spl(np.array(pp), n=7)), lw * .46, CP['sline'], taper=(.15, .65), press=.4)
    for rg, an_ in (([(lit * -.40, -1.1), (lit * -.72, -1.0), (lit * -.60, -.55), (lit * -.36, -.5)], 1.0), ([(lit * -.50, .0), (lit * -.86, -.1), (lit * -.80, .25), (lit * -.56, .25)], .9), ([(lit * -.62, .75), (lit * -.92, .55), (lit * -.76, 1.0), (lit * -.5, 1.1)], .7), ([(-.6, .78), (-.1, .98), (-.2, 1.2), (-.5, 1.0)], 1.9), ([(-.25, -.55), (.05, -.62), (.0, -.35), (-.2, -.3)], 1.3)): hatch(cel, M(rg), an_, s * .055, lw * .26, CP['line'], int(an_ * 20), length=s * .16, curve=.05)
    wet_hair(cel, M, [(-.62, -1.14, -.16, .62, 9), (-.38, -1.02, .04, .40, 5), (.36, -1.42, .10, .72, 9), (.66, -1.28, .20, .80, 8), (.98, -.66, .10, .50, 4), (-.96, -.92, -.14, .52, 5)], lw * .75, f, seed)
    return M


def curse_arm_skin(cel, p, fz, lw, seed=0, n=9):
    """warts, sores, folds and hairs along a drawn arm (screen space)"""
    r = np.random.default_rng(seed + 40); S2, E2, W2 = p
    for k in range(n):
        a, c, z = (S2, E2, fz[0]) if k % 2 else (E2, W2, fz[1]); t = .12 + .76 * ((k * .618) % 1.); u = (c - a) / (np.hypot(*(c - a)) + 1e-6); nn = np.array([-u[1], u[0]]); o = a + (c - a) * t + nn * (.34 * math.sin(k * 2.4) + r.normal(0, .04)) * z; R = r.uniform(.03, .075) * z; warts(cel, lambda P: o + np.asarray(P, float) * R, [(0., 0., 1.)], lw, k)
        if k % 3 == 0:
            for j in range(4): h0 = o + nn * r.uniform(-.14, .14) * z + u * r.uniform(-.12, .12) * z; cel.line(spl([h0, h0 + [r.normal(0, 5), .06 * z], h0 + [r.normal(0, 10), .15 * z]], n=6), lw * .38, CP['hair'], taper=(.1, .75), seed=j)
    for t in (.18, .34, .52): u = (W2 - E2) / (np.hypot(*(W2 - E2)) + 1e-6); nn = np.array([-u[1], u[0]]); o = E2 + (W2 - E2) * t; cel.line(spl([o - nn * .30 * fz[1], o + u * .06 * fz[1], o + nn * .22 * fz[1]], n=7), lw * .42, CP['sline'], taper=(.2, .6))


class Shot:
    id = 'd3'; n = 72
    finish = dict(bloom=.36, bloom_thr=.78, vig=.13, halation='#ffdcb0', soft=.14)
    EVENTS = [(0, 'glory', dict(size=.9, until=18, note='absurd heroic shimmer, choir-like sparkle, tears streaming')), (9, 'sob', dict(size=.8, note='one huge manly sob, 9-12; small one on 14')), (18, 'whoosh', dict(size=.5, note='arms rise, air drawn in 18-22')), (21, 'glint', dict(size=.3, note='his eye')), (23, 'whoosh', dict(size=.7, note='the curse fist flying at him, 23-26')),
              (27, 'clap', dict(size=1.1, note='THE clap: one dry enormous hand clap, black/white frames 27-28')), (28, 'swap', dict(size=.8, note='space flips: reversed whip')), (29, 'punch', dict(size=1.0, note='curse fist meets Yuji fist, cursed energy, ring')),
              (36, 'clap', dict(size=.9, note='clap, 1-frame white flash')), (37, 'swap', dict(size=.7, note='reversed whip')), (38, 'punch', dict(size=1.0, note='the curse hits its own face, teeth fly')), (41, 'rattle', dict(size=.3, note='teeth on asphalt')),
              (45, 'clap', dict(size=.9, note='clap, 1-frame white flash')), (46, 'swap', dict(size=.7, note='reversed whip')), (46, 'boing', dict(size=.5, note='Yuji lands on its head, rubbery')),
              (54, 'clap', dict(size=.95, note='clap, 1-frame white flash')), (55, 'punch', dict(size=1.3, note='both fists on the jaw at once; 57-60 the head snaps back')), (63, 'pose', dict(size=1.0, note='back to back slam, sunburst')), (64, 'sparkle', dict(size=.7, until=72, note='sparkle cascade'))]

    # ------------------------------------------------------------------ plates and figure sprites
    def setup(self):
        self.x, self.y = grid(); pf = lambda k: f'cache/d3_v{VER}_{k}.npy'; self.spr = {}; self.hand = {}
        if not os.path.exists(pf('glory')): np.save(pf('glory'), K.glory_plate().astype(np.float16)); K._C.clear()
        self.glory = np.load(pf('glory'), mmap_mode='r')
        pi = math.pi; self.cam = cC = Camera3D((-.6, -3., .60), (-.2, 30., 3.2), 1250. * U, cx=960, cy=540 + U * 115 - DY); HZ = cC.project(np.array([[0, 4000., .60]]))[0][0, 1]; self.hz = float(HZ)
        yh = lambda ex: (lambda cel, h, i: (Yuji.head(cel, YHead().pose(h.cx, h.cy, h.s, h.yaw, h.pitch, h.roll), ex, 'cursed', 0, .1, 1), None)[1]); th = lambda ex: (lambda cel, h, i: (Todo.head(cel, Todo.pose(h.cx, h.cy, h.s, h.yaw, h.pitch, h.roll), ex, 'night', 0, .05, 1, lw=max(1.6, h.s / 300 * 8.)), None)[1])
        from jj.fig_pose import POSES as PO, mix_pose as mp
        def ab(n, dz=0., **kw):                                  # a pose taken into the legs: the pelvis sinks, the back bends over it, the feet stay where they are
            q = dict(PO[n] if isinstance(n, str) else n); r = q.get('root', (0, 0, 0)); q['root'] = (r[0], r[1], r[2] - dz); pl = q.get('pel', (0, 0, 0)); q['pel'] = (pl[0] + 90 * dz, pl[1], pl[2]); cs_ = q.get('cst', (0, 0, 0)); q['cst'] = (cs_[0] + 60 * dz, cs_[1], cs_[2]); q.update(kw); return q
        op = lambda k, dz: ab('clap', dz, hL=(-.30 + .74 * k, .62 - .12 * k, .02 - .26 * k), hR=(-.30 + .74 * k, .62 - .12 * k, .02 - .26 * k), cst=(-8 + 4 * k, 0, 0))
        # key, who, pose, place, heading, head
        L = [('tO1', 'todo', op(.35, .015), N_, pi + .55, th('focus')), ('tO2', 'todo', op(1., .045), N_, pi + .55, th('focus')), ('tNa', 'todo', ab('clap', .055), N_, pi + .55, th('focus')), ('tN', 'todo', 'clap', N_, pi + .55, th('smug')),
             ('yF', 'yuji', 'run_contact', F_, -pi / 2 - .7, yh('snarl')), ('yF2', 'yuji', 'run_push', F_, -pi / 2 - .7, yh('snarl')), ('tF', 'todo', 'clap', F_, pi - .5, th('smug')),
             ('yNa', 'yuji', ab('punch_hit', .055), N3_, pi / 2 + .30, yh('snarl')), ('yN', 'yuji', 'punch_hit', N3_, pi / 2 + .30, yh('snarl')),
             ('yMa', 'yuji', ab('punch_follow', .05), M_, pi / 2 + .5, yh('snarl')), ('yM', 'yuji', mp(PO['punch_follow'], PO['guard'], .38), M_, pi / 2 + .5, yh('surprised')),
             ('yHa', 'yuji', ab('land', .05), H_, pi - .25, yh('grin')), ('yH', 'yuji', 'land', H_, pi - .25, yh('grin')),
             ('yJa', 'yuji', ab('uppercut', .085, hR=(.0, .74, .22)), JL_, -pi / 2 - .15, yh('snarl')), ('yJ', 'yuji', 'uppercut', JL_, -pi / 2 - .15, yh('snarl')), ('tJa', 'todo', ab('punch_hit', -.085, hR=(-.06, .90, .34)), JR_, pi / 2 + .15, th('warcry')), ('tJ', 'todo', ab(mp(PO['punch_hit'], PO['punch_follow'], .40), -.07, hR=(-.34, .80, .34)), JR_, pi / 2 + .15, th('warcry'))]
        for key, who, pose, (px, py), hd, hf in L:
            p = pf('s_' + key)
            if not os.path.exists(p):
                from jj.figure import Fig
                F = Fig(who).pose(pose).place(px, py, 0, heading=hd); A = F.draw(cC, 3, head_fn=hf); I = F.info; ys, xs = np.nonzero(A[..., 3] > .02); y0, y1, x0, x1 = max(0, ys.min() - 3), ys.max() + 4, max(0, xs.min() - 3), xs.max() + 4; np.save(p, A[y0:y1, x0:x1].astype(np.float16)); h_ = I['hands']; np.save(p[:-4] + '_xy.npy', np.array([x0, y0, *h_['L'], *h_['R']], float)); del A
            q = np.load(p[:-4] + '_xy.npy'); self.spr[key] = (np.load(p, mmap_mode='r'), q[:2].astype(int)); self.hand[key] = dict(L=q[2:4], R=q[4:6])
        # ground-level street, painted once; then the locked frame of it, with the mist that stands between it and the curse
        if not os.path.exists(pf('low')):
            P = world.plate('street', eye=(-.6, -3., .60), target=(-.2, 30., 3.2), focal=1250., people=26, car_=False); np.save(pf('low'), P.img.astype(np.float16)); np.save(pf('lowhz'), np.array([world.horizon_y(P.cam), P.cam.cx, P.cam.cy], float)); del P
        if not os.path.exists(pf('set')):
            low = np.load(pf('low'), mmap_mode='r'); hz, pcx, pcy = np.load(pf('lowhz')); x, y = self.x, self.y
            im = comp.cam2d(np.array(low, f32), pcx, hz - (753.4776 - H / 2) + DY / U, U) * rgb('#b9b2ee') * .90; b_ = qb(im, 60); m = (smooth(HZ - 520, HZ - 40, y) * (1 - smooth(HZ + 6, HZ + 150, y)))[..., None]; lane = np.exp(-((x - 760) / 620.) ** 2)[..., None]
            im = im * (1 - .30 * m) + (b_ * .95 + rgb('#6f78d8') * .20 + rgb('#ffd9b0') * .03) * .42 * m * (.55 + .55 * lane); np.save(pf('set'), im.astype(np.float16)); del low, im, b_
        self.set = np.load(pf('set'), mmap_mode='r'); x, y = self.x, self.y; d = np.clip(y - HZ, 1, None); r = np.random.default_rng(8); nz = cv2.resize(r.normal(0, 1, (18, 26)).astype(f32), (512, 512), interpolation=cv2.INTER_CUBIC); gx_, gy_ = (x - 960) / d * 1.1, 260. / d
        self.wet = (smooth(2, 26, d) * (.66 + .34 * smooth(-1., .9, cv2.remap(nz, ((gx_ * 24 + 4000) % 511).astype(f32), ((gy_ * 60) % 511).astype(f32), cv2.INTER_LINEAR)))).astype(f32)

    def put(self, img, key, dx=0, dy=0, sc=1., tone=None, rim_=None, blur_=0., op=1., flip=False):
        S, (x0, y0) = self.spr[key]; a = np.array(S, f32)
        if flip: a = a[:, ::-1]
        if np.ndim(sc): h0, w0 = a.shape[:2]; a = cv2.resize(a, None, fx=sc[0], fy=sc[1], interpolation=cv2.INTER_LINEAR); dx, dy = dx + (w0 - a.shape[1]) / 2, dy + (h0 - a.shape[0])
        elif sc != 1: a = cv2.resize(a, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
        L = np.zeros((H, W, 4), f32); px, py = int(round(x0 + dx)), int(round(y0 + dy)); h, w = a.shape[:2]; xa, xb, ya, yb = max(0, px), min(W, px + w), max(0, py), min(H, py + h)
        if xb <= xa or yb <= ya: return img, L
        L[ya:yb, xa:xb] = a[ya - py:yb - py, xa - px:xb - px]
        if key[:2] == 'hc':                                      # the far curse: same bruise ramp, lank hair on its back
            a_ = np.maximum(L[..., 3:4], 1e-4); lu = (L[..., :3] / a_) @ np.array([.3, .59, .11], f32); L[..., :3] = ramp(lu, [(0, '#0c090f'), (.16, '#2d2440'), (.30, '#5a4f6c'), (.46, '#7d7e66'), (.62, '#b3af8c'), (1, '#e3d3a6')]) * L[..., 3:4]
        if tone is not None: L[..., :3] *= rgb(tone)
        for r_ in (rim_ or ()): L = rim(L, r_[0], r_[1], r_[2], r_[3], .6)
        if blur_ > 0: L = cv2.GaussianBlur(L, (0, 0), blur_)
        return comp.over(img, L, op), L

    def frame(self, f):
        if f < 18: im = self.spread(f)
        elif f < 23: im = self.wind(f)
        elif f in (27, 28): im = self.clap(f, 27)
        elif f in (36, 45, 54): im = self.flash(f)
        elif f < 63: im = self.three(f)
        else: im = self.pose(f)
        return im.astype(f32)

    # ------------------------------------------------------------------ 0-18  arms spread, tears
    def spread(self, f):
        d = f // 2 * 2 if not 8 <= f <= 12 else f; z = 1 + .0026 * f; K._C[('glory_plate',)] = np.array(self.glory, f32)
        sob = keyed(f, [(0, 0.), (7, -.25), (8, -.5), (9, 1.), (10, .8), (11, .35), (12, .1), (13, -.15), (14, .55), (15, .35), (16, .1), (17, 0.)])      # in-breath, the heave, shudder, a smaller one
        im = K.key_todo(f=d, finish=False, cx=1042 + (5 * math.sin(f * 2.6) if 9 <= f <= 11 else 0), cy=312 + 22 * sob - .4 * f, s=131 * z, yaw=.18 - .10 * max(0, sob), pitch=.20 - .17 * sob, roll=.09 - .05 * sob + .02 * math.sin(d * .27), lift=.20 * sob); K._C.clear(); return im

    # ------------------------------------------------------------------ 18-27  the hands part
    def wind(self, f):
        x, y = self.x, self.y; z = 1.02 + .010 * (f - 18); bg = comp.cam2d(np.array(self.glory, f32), W / 2 + 50, H / 2 - 60, 1.12 * z); dk = smooth(18, 22, float(f)); bg = bg * (1 - .62 * dk) + rgb('#2a1450') * .34 * dk
        k = Cel(); focus_lines(k, 960, 520, 470 - 14 * (f - 18), 1500, 150, 6.5, '#12061f', f, ry=.72); bg = comp.over(bg, k.layer(), .25 + .55 * dk)
        wide = dict(Todo.POSES['windup'], R=((2.05, -.38, -.05), (3.35, -1.25, .10), (3.25, -.10, 1.10), 1), L=((-2.05, -.38, -.05), (-3.35, -1.25, .10), (-3.25, -.10, 1.10), -1))
        up_ = dict(Todo.POSES['spread'], R=((2.08, -.30, -.1), (3.45, .30, .45), (4.05, 1.60, .9), 1), L=((-2.08, -.30, -.1), (-3.45, .30, .45), (-4.05, 1.60, .9), -1), hR=('open', 188, 10, 0), hL=('open', 168, 10, 0)); po = {18: Todo.mix_pose('spread', up_, .55), 19: up_, 20: Todo.mix_pose(up_, wide, .45), 21: Todo.mix_pose(up_, wide, .85), 22: wide}[f]
        e = dict(TE['focus'], tears=1) if f >= 20 else dict(TE['weep']); s = 124 * z; cx, cy = 960, 404 + {18: -10, 19: -22, 20: -6, 21: 8, 22: 12}[f]; pal = TP['glory']; c = Cel(); lw = s / 300 * 8.6
        H_ = Todo.pose(cx, cy, s, 0., {18: .14, 19: .18, 20: .08, 21: -.02, 22: -.05}[f], 0.); b = TBody.under(H_, 0., .06, 0., drop=1.28, fwd=.10, persp=.10); Todo.figure(c, H_, b, po, e, pal, -1, lw, f, .10, 3); lay = c.layer()
        if f >= 25:                                              # only the hands smear, along their path
            m_ = np.exp(-(((np.abs(x - 960) - (330 if f == 25 else 150)) / 190.) ** 2 + ((y - 430) / 260.) ** 2))[..., None].astype(f32); lay = lay * (1 - m_ * .55) + smear(lay, 70, 0, 17) * m_ * .55
        lay = rim(lay, 5, 4, '#ffe9a8', .95, .6); lay = rim(lay, -5, 2, '#ff8ac8', .8, .6); lay = para(lay, (cx, 420), (cx, 1080), [(0, '#ffffff'), (1, '#6a4690')], 'mul', .9); img = comp.over(bg, lay)
        if 21 <= f <= 22: p = H_.P3(np.array([(.285, .0, .72)]))[0]; g = Cel(); K.sparkle(g, p[0], p[1], 120 if f == 21 else 70, '#ffffff', .1, .06); gl = g.layer(); img = comp.over(img, gl) + qb(gl[..., :3], 10) * .6
        return img

    # ------------------------------------------------------------------ the clap: two ink drawings
    def clap(self, f, c0):
        k = CLAP.index(c0); roll = (0, 24, -20, 8)[k]; sc = (1., .88, 1.12, 1.28)[k]; acc = ('#e0142c', '#14b4dc', '#f4b420', '#e0142c')[k]; cx, cy = ((960, 600), (1010, 590), (900, 610), (960, 640))[k]; first = f == c0; c = Cel(); c.fill(np.array([(0, 0), (W, 0), (W, H), (0, H)]), '#fbf8f1')
        radial_wedges(c, cx, cy - 90, 120, 300 if first else 430, 1500, 11 + k * 3 + first, w=.030, col='#0b090c')
        rg = Cel(); ragged_ring(rg, cx, cy - 90, 430 if first else 640, 350 if first else 520, 34 if first else 20, 5 + k, acc, gaps=.35, rag=.08)
        cam = HD.Cam(cx, cy, 400 * sc * (1.06 if first else 1.), 5.0); hs = HD.two('clap', R=HD.Rz(HD.D(roll)) @ HD.Rx(HD.D(20)), bld=HD.HEAVY, gap=0. if first else .02); hp = dict(skin='#fffdf8', shade='#0b090c', line='#0b090c', nail='#fffdf8', nail_shade='#0b090c', light='#ffffff')
        g = HD.render(hs, cam, (-.7, .6, .35), res=420, thr=.30); 
        h = Cel(); HD.ink(h, g, hp, 11, 3 + k, inner=.8); L = h.layer(); img = c.rgba()[..., :3]; img = comp.over(img, rg.layer()); bl = cv2.dilate((L[..., 3] > .5).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=9).astype(f32); img = img * (1 - bl[..., None]) + rgb('#fbf8f1') * bl[..., None]; img = comp.over(img, L)
        sp = Cel(); splatter(sp, cx, cy - 120, 330, 90, 15, '#0b090c', 7 + k); img = comp.over(img, sp.layer())
        if not first:
            lum_ = img.mean(-1, keepdims=True); red = (np.abs(img - rgb(acc)).sum(-1, keepdims=True) < .5); img = np.where(red, img, 1 - img); img = np.where(red, img, img * rgb('#f6ecdf') + (1 - img) * rgb('#0a0710') * .6)
            sx, sy = shake(f, 14, k); img = comp.cam2d(img.astype(f32), W / 2 + sx, H / 2 + sy, 1.04)
        return img

    # ------------------------------------------------------------------ one frame of flash (36, 45, 54)
    def flash(self, f):
        x, y = self.x, self.y; k = (36, 45, 54).index(f); c = Cel(); radial_wedges(c, 960, 520, 70, 520, 1500, 5 + k, w=.010, col=('#ffd9a0', '#c8e6ff', '#ffc9e4')[k]); im = np.ones((H, W, 3), f32) * rgb('#fffdf8') * 1.05; return comp.over(im, c.layer(), .9)

    def star(self, c, q, R, seed, col='#4fb8ff', n=16, a=.95):
        an = np.linspace(0, 6.283, 2 * n + 1)[:-1]; rr = np.where(np.arange(2 * n) % 2 == 0, 1., .36) * np.random.default_rng(seed).uniform(.6, 1.25, 2 * n) * R; c.fill(np.c_[q[0] + np.cos(an) * rr, q[1] + np.sin(an) * rr * .9], '#fbfdff', a); c.fill(np.c_[q[0] + np.cos(an) * rr * .5, q[1] + np.sin(an) * rr * .46], col, a)

    def curse_at(self, img, hx, hy, s, rot, mood, hit, f, arm, lw, lit=-1, look=None, lean=0., small=None, arm_front=True, tone=1.):
        """curse in a slot.  arm = (elbow px, wrist px, depth z, hand yaw, hand roll) of the huge arm in screen px"""
        c = Cel(); b = TBody(hx, hy + 1.15 * s, s * .86, yaw=0., persp=.07); cp = dict(CP); u = s * .86
        inv = lambda q, z: ((q[0] - b.c[0]) / u * (1 - z * .07), -(q[1] - b.c[1]) / u * (1 - z * .07), z)
        curse_head(c, hx, hy, s, rot, mood, hit, lw, lit, f, True, 2, True, look, lean)
        sh = small or ((-2.72, -1.85, .2), (-2.15, -2.55 + .10 * math.sin(f * 1.1), .6)); Todo.arm(c, b, (-2.05, -1.0, .1), sh[0], sh[1], cp, lw * .75, -1, -1, lit, 5, (lambda cel, w, e, fz: hand3(cel, w, e, fz, cp, lw * .75, 'R', 'splay', 30, 20, 0, .80, seed=2)), bulk=.60)
        E_, W_, z, yw, rl = arm; p_, z_ = Todo.arm(c, b, (2.45, .15, .1), inv(E_, z * .5), inv(W_, z), cp, lw, 1, 1, lit, 3, (lambda cel, w, e, fz: hand3(cel, w, e, fz, dict(CP, skin='#8d8c75'), lw, 'L', 'fist', yw, 8, rl, 1.0, seed=3, light=(-.5, .7, .4))), bulk=1.62); curse_arm_skin(c, p_, z_, lw, 2, 7); lay = c.layer()
        lay = para(lay, (hx, hy - s), (hx + 60, hy + 5.5 * s), [(0, '#ffffff'), (.5, '#cfc6ea'), (1, '#8e84c4')], 'mul', .85); return lay, p_

    def ground_shadow(self, img, x, y, rx, ry, a=.55):
        m = np.zeros((H // 2, W // 2), f32); cv2.ellipse(m, (int(x / 2), int(y / 2)), (int(rx / 2), int(ry / 2)), 0, 0, 360, a, -1, cv2.LINE_AA); m = cv2.resize(cv2.GaussianBlur(m, (0, 0), 7), (W, H))[..., None]; return img * (1 - m) + rgb('#0a0818') * m * .6

    # ------------------------------------------------------------------ weight, water, light: what stands on the street belongs to it
    def aff(self, g, sx=1., sy=1., lean=0.): return np.float32([[sx, -lean, g[0] * (1 - sx) + lean * g[1]], [0, sy, g[1] * (1 - sy)]])

    @staticmethod
    def pt(M, p): return M[:, :2] @ np.asarray(p, float) + M[:, 2]

    @staticmethod
    def ip(M, p): return np.linalg.solve(M[:, :2].astype(float), np.asarray(p, float) - M[:, 2])

    def fig(self, key, dx=0., dy=0.):
        """sprite as a full layer, with the point its weight stands on"""
        S, (x0, y0) = self.spr[key]; h, w = S.shape[:2]; px, py = int(round(x0 + dx)), int(round(y0 + dy)); L = np.zeros((H, W, 4), f32); xa, xb, ya, yb = max(0, px), min(W, px + w), max(0, py), min(H, py + h); L[ya:yb, xa:xb] = S[ya - py:yb - py, xa - px:xb - px]
        a = np.array(S[int(h * .90):, :, 3], f32).sum(0); return L, np.array([px + float((a * np.arange(w)).sum() / (a.sum() + 1e-6)), py + h - 5.]), float(h)

    def lit(self, L, g, h, nr):
        """the lamp over the street, the shop fronts on the other side, dark below the knee, light coming back up off the water"""
        sd = 1. if g[0] < LAMP[0] else -1.; k = float(np.clip(nr, .5, 1.6)); ft = smooth(g[1] - .30 * h, g[1], self.y)[..., None]; L = para(L, (g[0] + sd * .34 * h, g[1] - h), (g[0] - sd * .36 * h, g[1]), [(0, '#ffffff'), (.42, '#ddd5f4'), (1, '#7d72b4')], 'mul', .78); L = L.copy(); L[..., :3] *= 1 - ft * (1 - rgb('#8f86c8')) * .42
        up = 1 - smooth(g[1] - .46 * h, g[1] - .10 * h, self.y); up = up * (.55 + .45 * smooth(-.35, .25, np.sin(self.y * .031 + g[0]) + .6 * np.sin(self.y * .083 + self.x * .02)))
        L = rim(L, 0, -2.2 * k, '#8d86ff', .30, .8); L = rim(L, sd * 2.4 * k, .4, '#8fc0ff' if sd > 0 else '#ff8fd0', .55, .7, where=.25 + .75 * up); return rim(L, -sd * 4.4 * k, 3.4 * k, '#ffe6bd', .95, .7, where=up)

    def soles(self, a):
        """where this body touches the street: the flat bottoms of its silhouette, foot by foot"""
        gy, rx = a['g'][1], a['rx']; bd = int(a.get('band', .30) * rx + 8); y0, y1 = max(0, int(gy) - bd), min(H, int(gy) + 8); q = a['L'][y0:y1, :, 3] > .5; c = np.where(q.any(0), y1 - 1 - np.argmax(q[::-1], 0), -1).astype(f32); w = max(3, int(.16 * rx)); mx_ = cv2.dilate(c[None], np.ones((1, 2 * w + 1), np.uint8))[0]; xs = np.nonzero((c >= 0) & (c >= mx_ - .035 * rx - 2))[0]; return xs, c[xs]

    def reflect(self, img, L, so, f, amt=1.):
        """what stands on the street, upside down in the wet of it: each foot mirrors at its own depth, torn sideways by the ripples, softer the further it runs"""
        xs, ys = so
        if not len(xs): return img
        b = np.interp(np.arange(W // 2) * 2., xs, ys).astype(f32) / 2; b = cv2.blur(b[None], (9, 1), borderType=cv2.BORDER_REPLICATE)[0]; y0 = int(b.min()); n = H // 2 - y0
        if n < 4: return img
        a = cv2.resize(L, (W // 2, H // 2), interpolation=cv2.INTER_AREA); yy, xx = np.mgrid[y0:H // 2, 0:W // 2].astype(f32); d = yy - b[None]; R = cv2.remap(a, xx + (.7 + .026 * np.clip(d, 0, None)) * (np.sin(d * .62 + f * .9) + .8 * np.sin(d * .23 - f * .5 + xx * .013)), 2 * b[None] - yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) * (d >= -.5)[..., None]
        al = R[..., 3]; ed = np.clip(al - cv2.warpAffine(al, np.float32([[1, 0, 3], [0, 1, -1]]), (al.shape[1], al.shape[0])), 0, 1) + np.clip(al - cv2.warpAffine(al, np.float32([[1, 0, -3], [0, 1, -1]]), (al.shape[1], al.shape[0])), 0, 1); R[..., :3] += ed[..., None] * rgb('#aeb4ff') * .22
        m = np.clip(d / 90., 0, 1)[..., None]; R = cv2.GaussianBlur(R, (0, 0), sigmaX=.8, sigmaY=1.5) * (1 - m) + cv2.GaussianBlur(R, (0, 0), sigmaX=2.2, sigmaY=8) * m; y1 = y0 * 2; R = cv2.resize(R, (W, H - y1), interpolation=cv2.INTER_LINEAR); dd = cv2.resize(d, (W, H - y1), interpolation=cv2.INTER_LINEAR)
        w = self.wet[y1:] * amt * np.exp(-np.clip(dd, 0, None) / 190.); img[y1:] = img[y1:] * (1 - (R[..., 3] * w * .90)[..., None]) + R[..., :3] * rgb('#dcd5ff') * 1.45 * w[..., None]; return img

    def contact(self, img, so, rx):
        """a hard dark line where each sole touches"""
        xs, ys = so
        if not len(xs): return img
        m = np.zeros((H, W), f32); m[np.clip(ys.astype(int) + 1, 0, H - 1), xs] = 1.; k = max(.6, rx / 90.); ya, yb = max(0, int(ys.min()) - 40), min(H, int(ys.max()) + 70); q = np.clip(cv2.GaussianBlur(m[ya:yb], (0, 0), sigmaX=4.5 * k, sigmaY=1.7 * k) * 4.4 * k + cv2.GaussianBlur(m[ya:yb], (0, 0), sigmaX=14 * k, sigmaY=4.2 * k) * 5. * k, 0, 1)[..., None]; img[ya:yb] = img[ya:yb] * (1 - .88 * q) + rgb('#070414') * .65 * q; return img

    def splash(self, gb, gf_, g, R, t, seed):
        """landing: a torn sheet of water (three drawings), drops on their arcs, rings running out over the street"""
        gx, gy = g; q = max(.05, (gy - self.hz) / 1075.); r = np.random.default_rng(seed); sc = R / 150.; E = lambda an, k, up=0.: np.c_[gx + np.cos(an) * k, gy + 2 + np.sin(an) * k * q - up]
        for j, (t0, w) in enumerate(((0, 1.), (2, .7))):
            tt = t - t0
            if 0 <= tt <= 6: rx = R * (.78 + .24 * tt); ragged_ring(gb, gx, gy + 2, rx, rx * q, max(1.2, (7 - .9 * tt) * w * sc), seed + 5 * j + tt // 2, hexmix('#dfe5ff', '#55498a', min(.85, .15 * tt)), gaps=.24 + .08 * tt, rag=.08, spurs=.1)
        if t <= 2:
            for sd, c in ((-1, gb), (1, gf_)):
                an = np.linspace(.04, math.pi - .04, 110); hg = np.full(110, .05)
                for _ in range(9): hg = np.maximum(hg, r.uniform(.14, .62) * np.exp(-np.abs((an - r.uniform(.1, 3.04)) / r.uniform(.035, .10)) ** 1.35))
                hg = hg * R * (.50, .82, .74)[t] * (.45 + .55 * np.sin(an)); lo = (0., .0, .45)[t]; kt = R * (.58 + .07 * t); a_ = an * sd; tip = np.gradient(hg) * .5
                base, top = E(a_, kt, hg * lo), E(a_, kt + hg * .30, hg); top[:, 0] += np.cos(a_) * hg * .25; c.fill(np.vstack([base, top[::-1]]), '#c3ceff', (.80, .72, .50)[t]); c.fill(np.vstack([base, E(a_, kt, hg * (lo + (1 - lo) * .40))[::-1]]), '#4f4fa8', (.55, .45, .3)[t]); c.line(top, 1.7 * sc, '#f6f8ff', taper=(.1, .1), press=.6, alpha=(.95, .85, .6)[t])
        if t <= 5:
            for i in range(30):
                an = 6.283 * (i + r.uniform(-.4, .4)) / 30; v = r.uniform(.18, 1.) ** 1.3 * R * .40; o = r.uniform(.04, .16) * R; P = lambda u: np.array([gx + math.cos(an) * (R * .6 + o * u), gy + math.sin(an) * (R * .6 + o * u) * q - v * u + .23 * v * u * u]); u = t + .7; p1, p0 = P(u), P(u - .5)
                if p1[1] > gy + math.sin(an) * (R * .6 + o * u) * q - 1: continue
                c = gf_ if math.sin(an) > 0 else gb; w = max(1.3, R * r.uniform(.010, .026) * (1 - .06 * t)); d = p1 - p0; d /= np.hypot(*d) + 1e-6; n = np.array([-d[1], d[0]]); c.fill(spl(np.array([p1 + d * w, p1 + n * w, p0 * .3 + p1 * .7 - d * w * 2.2, p1 - n * w]), True, corners=(2,), n=6), '#eef2ff', .95)

    def trail(self, c, p0, p1, h0, h1, cols, t, seed, up=1.):
        """the swap itself: dry streaks of the body's own colours from where it was to where it is"""
        r = np.random.default_rng(seed); p0, p1 = np.array(p0, float), np.array(p1, float); md = (p0 + p1) / 2 + [0, -up * .13 * abs(p1[0] - p0[0])]
        for i in range(18):
            v = (i + .5) / 18 - .5 + r.normal(0, .02); s = np.linspace(r.uniform(.12, .62) if t == 0 else r.uniform(.62, .90), r.uniform(.90, 1.), 10)[:, None]; c.line((1 - s) ** 2 * (p0 + [0, v * h0]) + 2 * s * (1 - s) * (md + [0, v * (h0 + h1) / 2]) + s * s * (p1 + [0, v * h1]), max(4, h1 * r.uniform(.012, .042)), cols[i % len(cols)], taper=(.95, .2), alpha=.9 if t == 0 else .5, seed=i)

    def stage(self, img, A, T, f):
        for a in A:
            a['so'] = self.soles(a)
            if not a.get('late'): img = self.ground_shadow(img, a['g'][0], a['g'][1] - .02 * a['rx'], a['rx'] * 1.2, max(4, a['rx'] * 1.2 * (a['g'][1] - self.hz) / 1075.), .30)
        for a in A: img = self.reflect(img, a['L'], a['so'] if not a.get('late') else (np.array([0, W]), np.array([a['gr']] * 2)), f)
        gb, gf_ = Cel(), Cel()
        for a in A:
            if not a.get('late'): img = self.contact(img, a['so'], a['rx'])
            if a.get('land') is not None and 0 <= a['land'] <= 7 and not a.get('late'):
                for j, (x_, R_) in enumerate(a.get('feet') or [(a['g'][0], a['rx'])]): self.splash(gb, gf_, (x_, a['g'][1]), R_, a['land'], a.get('seed', 3) + 31 * j)
        img = comp.over(comp.over(img, T.layer()), gb.layer())
        for a in A:
            if a.get('late'): img = self.contact(img, a['so'], a['rx'] * .6)
            if a.get('pre') is not None: img = comp.over(img, a['pre'].layer())
            if a.get('ghost'):
                dx, dy, k = a['ghost']; q = cv2.warpAffine(a['L'], np.float32([[1, 0, -dx * .55], [0, 1, -dy * .55]]), (W, H)); kx, ky = int(abs(dx)) | 1, int(abs(dy)) | 1
                for z in (.62, .62): q = cv2.blur(q, (max(1, int(kx * z) | 1), max(1, int(ky * z) | 1)), borderType=cv2.BORDER_CONSTANT)
                img = comp.over(img, q, k)
            img = comp.over(img, a['L'])
        return comp.over(img, gf_.layer())

    # ------------------------------------------------------------------ 23-63  ONE locked camera, ground level; only the bodies change slots
    def three(self, f):
        fs = lambda p, z=0.: self.cam.project(np.array([[p[0], p[1], z]]))[0][0]; img = np.array(self.set, f32); gm, gn, gf = fs(M_), fs(N_), fs(F_); cs = 86. * U; mx, my = gm[0] + 10 * U, gm[1] - 6.02 * cs; lw = 4.5
        if f < 27: seg, t = 0, f - 23
        elif f < 36: seg, t = 1, f - 29
        elif f < 45: seg, t = 2, f - 37
        elif f < 54: seg, t = 3, f - 46
        else: seg, t = 4, f - 55
        amp = (4, 14, 16, 9, 20)[seg]; shx, shy = shake(f, amp * U * math.exp(-max(0, t) / 1.6) * (seg > 0), 3 + seg); g, T, h_ = Cel(), Cel(), Cel(); br = 1 + .005 * math.sin(f * .6); pre = f in (35, 44, 53)
        LAND = {0: (1.07, .91), 1: (1.045, .945), 2: (.985, 1.022), 3: (.996, 1.006)}; sq = LAND.get(t, (1., br)) if seg else (1., br); STRETCH = (.94, 1.03) if seg == 3 else (.93, 1.045); ppm = lambda gy: (gy - self.hz) / .6; still = (1., br)
        TC, YC, KC = ['#e6ae7c', '#34384c', '#e8dcc8', '#b4744e'], ['#262a52', '#c22f3a', '#e58fa8', '#3c4684'], ['#8b8c70', '#594d6b', '#2d2440', '#b3af8c']; slot = dict(near=(gn[0], gn[1] - .50 * 1.75 * ppm(gn[1])), mid=(mx, gm[1] - .5 * 1.8 * ppm(gm[1])), far=(gf[0], gf[1] - .5 * 1.8 * ppm(gf[1])), top=(mx, my - 1.9 * cs))

        def act(key, sq_, tone='#ffffff', dx=0., dy=0., lean=0., land=None, **k):
            L, gg, h = self.fig(key, dx, dy); pm = ppm(k.get('gr', gg[1])); M = self.aff(gg, sq_[0], sq_[1], lean); L = self.lit(cv2.warpAffine(L, M, (W, H)) * np.r_[rgb(tone), 1.].astype(f32), gg, h, pm / 260.); return dict(L=L, g=gg, h=h, M=M, rx=.40 * pm, land=land, **dict(dict(band=.66 if key[0] == 'y' or key[:2] == 'tJ' else .34), **k))

        def curse(hx, hy, s, rot, mood, hit, arm, lw_, gy, sq_, lean=0., land=None, small=None, look=None, fl=0., **k):
            gg = np.array([hx + .25 * s, gy]); M = self.aff(gg, sq_[0], sq_[1], lean); E_, W_ = self.ip(M, arm[0]), self.ip(M, arm[1]); lay, _ = self.curse_at(None, hx, hy, s, rot, mood, hit, f, (E_, W_) + tuple(arm[2:]), lw_, look=look, small=small, lean=fl); L = self.lit(cv2.warpAffine(lay, M, (W, H)), gg, 7.5 * s, s / 80.); k.setdefault('feet', [(hx - 1.35 * s, .98 * s), (hx + 1.45 * s, 1.15 * s)]); return dict(L=L, g=gg, M=M, h=7.5 * s, rx=2.35 * s, land=land, seed=17, **k)
        A = []
        if seg == 0:                                             # the bait: its fist is on the way to Todo, Yuji is nowhere near
            hT = np.array([1462., 236.]); k = keyed(f, [(23, .10), (24, .38), (25, .66), (26, .88)]); W0, W1 = np.array([mx + 232, my + 34]), hT + [-54, 2]; Wp = W0 * (1 - k) + W1 * k; tp = keyed(f, [(23, 'tO1'), (24, 'tO2'), (26, 'tO1')])
            A.append(act('yF' if t < 2 else 'yF2', still, '#cfd0ee', 6 * t)); A.append(curse(mx, my, cs, .06 + .10 * k, 'grin', None, (np.array([mx + (250 + 130 * k) * U, my + (80 - 10 * k) * U]), Wp, .6 + 1.2 * k, 96, 4), lw, gm[1], still, lean=.055 * k, look=(.25, .06), land=t - 1 if t >= 1 else None, feet=[(mx + 1.45 * cs, .85 * cs)]))
            if t >= 1: g.line(np.array([W0 + [34, -26], (W0 + Wp) / 2 + [0, -52], Wp + [-52, -43]]), 14, '#eaf0ff', taper=(.8, .1), alpha=.5); g.line(np.array([W0 + [34, 34], (W0 + Wp) / 2 + [0, 43], Wp + [-52, 52]]), 9, '#eaf0ff', taper=(.8, .1), alpha=.4)
            A.append(act(tp, keyed(f, [(23, (1., 1.)), (24, (.99, 1.012)), (25, (1.01, .99)), (26, (1.02, .975))]), '#ffe9d4', lean=keyed(f, [(23, 0.), (24, .012), (25, .016), (26, -.010)]), pre=g))
        elif seg == 1:                                           # swap: Yuji stands where Todo stood, fist to fist
            y_ = act('yNa' if t < 2 else 'yN', STRETCH if pre else sq, '#eceaff', lean=0. if pre else -.05 * math.exp(-t / 2.5) * math.cos(t * .9), land=t, seed=5, ghost=(-300, 36, .75) if t == 0 else (-260, 0, .6) if pre else None); q = self.pt(y_['M'], min((self.hand['yNa' if t < 2 else 'yN'][k_] for k_ in 'LR'), key=lambda v: v[0])) + [-10, 0]
            k = keyed(f, [(29, 0.), (30, 10), (31, 4), (32, 0), (33, -6), (34, -10), (35, -12)]); Wp = q + np.array([-118 - k * 2.2, -14]) * U; buck = keyed(f, [(29, 0), (30, 26), (31, 40), (33, 30)]) * U
            A.append(act('tF', STRETCH if pre else sq, '#ffdcc0', land=t, seed=9, ghost=(120, 6, .7) if t == 0 else None)); A.append(curse(mx - buck * .6, my, cs, .14 - .004 * buck, 'grin' if t < 2 else 'shock', None, (np.array([mx + 284 - buck, my + 52 - buck * 1.2]), Wp, 1.7, 96, 4), lw, gm[1], STRETCH if pre else (1.02, .975) if t in (1, 2) else still, lean=-.0012 * buck, look=(.30, .10) if t >= 2 else (.25, .06), land=t - 1 if t >= 1 else None, feet=[(mx + 1.45 * cs, .85 * cs)]))
            if t < 2: self.trail(T, slot['far'], slot['near'], 120, 560, YC, t, 4, 1.); self.trail(T, slot['near'], slot['far'], 520, 130, TC, t, 8, -.5)
            if t < 5: self.star(g, q + [-30, 0], (300 - 46 * t) * U, 3 + t, '#4fb8ff', 16, .95 if t < 2 else .7)
            ragged_ring(g, q[0] - 26, q[1], (130 + 100 * t) * U, (120 + 92 * t) * U, max(3, 21 - 3.4 * t), 5, '#cfeaff', gaps=.35, rag=.08); y_['pre'] = g; A.append(y_); fl = (q[0] - 26, q[1], 420, '#7fd0ff', .42 * math.exp(-t / 2.4), gm[1] + 60)
        elif seg == 2:                                           # swap: the curse stands where Yuji stood, and its own punch arrives
            A.append(act('tF', still, '#ffdcc0')); A.append(act('yMa' if t < 2 else 'yM', STRETCH if pre else sq, '#e6e6ff', lean=-.06 * math.exp(-t / 2.) if not pre else 0., land=t, seed=13, ghost=(-200, 10, .7) if t == 0 else (120, -90, .6) if pre else None)); gc = fs(NC_); s = 109.; nx, ny = gc[0], gc[1] - 6.06 * s
            am = keyed(f, [(37, .40), (38, .86), (39, .92), (40, .80), (41, .70), (42, .64), (43, .60), (44, .58)]); wob = keyed(f, [(37, .02), (38, -.06), (39, -.11), (40, -.15), (41, -.17), (43, -.18)]); hit = (np.array([.98, .10]), np.array([-.62 * am, -.06 * am]), .58); M = self.aff((nx + .25 * s, gc[1]), *(STRETCH if pre else sq)); arm = (self.pt(M, (nx + 3.35 * s, ny + .10 * s)), self.pt(M, (nx + (1.62 - .42 * am) * s, ny + .16 * s)), 1.25, 84, -8)
            A.append(curse(nx, ny, s, wob, 'hit', hit, arm, 5.4, gc[1], STRETCH if pre else sq, lean=0. if pre else -.035 * (1 - math.exp(-t / 2.)), land=t, small=((-2.75, -1.55, .2), (-2.5, -.80 + .14 * math.sin(f * 1.3), .7)), ghost=(330, 60, .7) if t == 0 else (-260, 0, .6) if pre else None)); p_ = self.pt(A[-1]['M'], (nx + (1.0 - .46 * am) * s, ny - .42 * s))
            if t < 2: self.trail(T, slot['mid'], (nx, ny + 2.6 * s), 420, 640, KC, t, 14, .6); self.trail(T, slot['near'], slot['mid'], 520, 320, YC, t, 18, -.6)
            if t < 4: self.star(g, p_, 124 - 29 * t, 13 + t, '#ffb42c', 14)
            r = np.random.default_rng(21); c0 = self.pt(A[-1]['M'], (nx - .75 * s, ny + .25 * s))
            for i in range(15):
                a_ = math.pi + r.normal(.25, .75); v = r.uniform(30, 100) * .68; q = c0 + np.array([math.cos(a_), math.sin(a_) - .35]) * (34 + v * (t + .6)) + [0, 4. * t * t]; R = r.uniform(12, 30) * .68; th_ = r.uniform(0, 6.28) + t * r.normal(0, .6); Tn = np.array([(-.5, -.5), (.5, -.5), (.34, .3), (0, 1.1), (-.34, .3)]) @ np.array([[math.cos(th_), math.sin(th_)], [-math.sin(th_), math.cos(th_)]]) * R + q; Tn = spl(Tn, True, corners=(0, 1, 3), n=4); g.fill(Tn, BONE); outline(g, Tn, 2.2, '#14101c', i)
            fl = (p_[0], p_[1], 280, '#ffd27a', .36 * math.exp(-t / 2.4), gc[1] + 20)
        elif seg == 3:                                           # swap: Yuji is on its head, Todo in the near slot
            sc_ = STRETCH if pre else {0: (1.10, .86), 1: (1.06, .92), 2: (.99, 1.012), 3: (.998, 1.004)}.get(t, still); c_ = curse(mx, my, cs, -.02, 'shock', (np.array([.1, -1.35]), np.array([0, .24 * (t < 2) + .06]), .7), (np.array([mx + 318 * U, my + 200 * U]), np.array([mx + 280 * U, my + 375 * U]), .9, 70, 0), lw, gm[1], sc_, look=(.04, -.34), land=t); A.append(c_)
            kk = 'yHa' if t < 2 else 'yH'; S0, (x0, y0) = self.spr[kk]; top = self.pt(c_['M'], (mx, my - 1.28 * cs))[1]; y_ = act(kk, {0: (1.05, .93), 1: (1.03, .96), 2: (.995, 1.008)}.get(t, (1., 1 + .008 * math.sin(f * .6))), '#eeeaff', mx - 16 - (x0 + S0.shape[1] / 2), top - (y0 + S0.shape[0]) + 46, late=True, gr=gm[1], ghost=(60, 260, .6) if t == 0 else (-120, 40, .5) if pre else None); A.append(y_)
            for i in range(5): a_ = -2.6 + i * .52; d_ = np.array([math.cos(a_), math.sin(a_)]); o_ = np.array([mx - 4, top - 60]); g.line([o_ + d_ * (165 + 7 * t), o_ + d_ * (215 + 12 * t)], 7, '#ffffff', taper=(.2, .6), alpha=max(0, .9 - .16 * t))
            if t < 2: self.trail(T, slot['mid'], slot['top'], 300, 240, YC, t, 24, 0.); self.trail(T, (fs(NC_)[0], slot['near'][1]), slot['mid'], 560, 430, KC, t, 26, -.5); self.trail(T, slot['far'], slot['near'], 120, 640, TC, t, 28, 1.)
            A.append(act('tNa' if t < 2 else 'tN', STRETCH if pre else sq, '#ffe9d4', lean=0. if pre else .03 * math.exp(-t / 2.5) * math.cos(t), land=t, seed=23, pre=g, ghost=(300, 60, .75) if t == 0 else (-260, -10, .6) if pre else None)); fl = None
        else:                                                    # swap: one on each side of the jaw, feet in the street, both fists home
            rc = keyed(f, [(55, 0.), (56, .06), (57, .16), (58, .55), (59, .80), (60, .95), (61, 1.02), (62, 1.05)]); hx, hy = mx + (30 - 16 * rc) * U, my - 40 * U * rc; rot = -.02 - .60 * rc; sq_ = keyed(f, [(55, .30), (56, .40), (57, .36), (58, .22), (59, .12), (60, .06)]); sc_ = {0: (1.04, .955), 1: (1.06, .93), 2: (1.04, .95), 3: (.99, 1.02), 4: (.985, 1.03)}.get(t, (1., 1.01))
            def two(P):                                          # both fists press the jaw at once
                P = np.asarray(P, float); kl, kr = np.exp(-(np.hypot(*(P - [-.92, .72]).T) / .62) ** 2), np.exp(-(np.hypot(*(P - [.92, .70]).T) / .62) ** 2); return P + np.c_[sq_ * 1.5 * kl - sq_ * 1.5 * kr, -sq_ * .55 * (kl + kr)]
            global warp; w0 = warp; warp = lambda P, hit: two(P)
            try: c_ = curse(hx, hy, cs, rot, 'hit', True, (np.array([hx + 318 * U, hy + (120 - 60 * rc) * U]), np.array([hx + 420 * U, hy + (265 - 150 * rc) * U]), .7, 80, 0), lw, gm[1], sc_, lean=-.02 * rc, look=(0., -.30), small=((-2.9, -1.3, .2), (-3.2, -.45, .5)), fl=.06 * rc, land=t - 3 if t > 2 else None)
            finally: warp = w0
            A.append(c_); Mc = self.aff((mx + .25 * cs, gm[1]), *sc_); x0_ = mx + 30 * U; cL, cR = self.pt(Mc, (x0_ - .80 * cs + sq_ * 52, my + .62 * cs)), self.pt(Mc, (x0_ + .80 * cs - sq_ * 52, my + .60 * cs)); ky, kt = ('yJa', 'tJa') if t < 3 else ('yJ', 'tJ'); hy_ = max((self.hand[ky][k_] for k_ in 'LR'), key=lambda v: v[0]); ht_ = min((self.hand[kt][k_] for k_ in 'LR'), key=lambda v: v[0]); gyY, gyT = self.spr[ky][1][1] + self.spr[ky][0].shape[0] - 5., self.spr[kt][1][1] + self.spr[kt][0].shape[0] - 5.
            fitY, fitT = (float(np.clip((gyY - cL[1]) / (gyY - hy_[1]), .92, 1.08)), float(np.clip((gyT - cR[1]) / (gyT - ht_[1]), .92, 1.08))) if t < 3 else (br, br); st = {0: 26, 1: 0, 2: -3, 3: 10, 4: 18, 5: 20}.get(t, 20)          # two frames of weight, then the follow-through takes them past it
            yo = act(ky, (1 + (1 - fitY) * .5, fitY), '#eceaff', cL[0] - hy_[0] - 16 + (st if t > 2 else -st * .5) * .8, 0, land=t, seed=43, ghost=(0, 240, .6) if t == 0 else None); to = act(kt, (1 + (1 - fitT) * .5, fitT), '#ffe9d4', cR[0] - ht_[0] + 18 - (st if t > 2 else -st * .5) * .8, 0, land=t, seed=47, ghost=(-300, 0, .75) if t == 0 else None)
            if t < 2: self.trail(T, slot['near'], (cR[0] + 210, gm[1] - 220), 560, 360, TC, t, 34, .8); self.trail(T, slot['top'], (cL[0] - 110, gm[1] - 200), 220, 360, YC, t, 36, 0.)
            if t < 4: self.star(g, cL + [-12, 5], (128 - 30 * t) * U, 41 + t, '#4fb8ff', 12); self.star(g, cR + [12, 5], (138 - 32 * t) * U, 61 + t, '#ffb04a', 12)
            ragged_ring(g, x0_, my + .6 * cs, (150 + 130 * t) * U, (120 + 100 * t) * U, max(3, 26 - 3.4 * t), 9, '#fff0d8', gaps=.3, rag=.08); r = np.random.default_rng(33)
            for i in range(20): a_ = -1.57 + r.normal(0, .8); v = r.uniform(30, 120) * U; q = np.array([x0_, my + .3 * cs]) + np.array([math.cos(a_), math.sin(a_)]) * (34 + v * (t + .5)) + [0, 4. * t * t]; R = r.uniform(7, 19); an = np.sort(r.uniform(0, 6.283, 5)) + t * r.normal(0, .5); P = np.c_[q[0] + np.cos(an) * R, q[1] + np.sin(an) * R]; g.fill(P, BONE if i % 2 else '#8e9a6c'); outline(g, P, 2., '#14101c', i)
            yo['pre'] = g; A += [yo, to]; fl = (x0_, my + .6 * cs, 450, '#fff0d0', .34 * math.exp(-t / 2.2), gm[1] + 10)
        img = self.stage(img, A, T, f)
        if seg == 1: bk_, fr_ = fx.cursed(fx.limb_poly(np.array([q + [0, 2], q + [104, 9], q + [198, 34]]), 45, 40), f, stream=(1, -.25), size=60, amount=.8, seed=3); img = fx.put(img, fr_, .9)
        if seg == 2: img = comp.over(img, g.layer())
        if seg and fl:                                           # the blow lights the air, and the street under it
            img = flare(img, fl[0], fl[1], fl[2], fl[3], fl[4]); x, y = self.x, self.y; img = img + (np.exp(-((x - fl[0]) / (.42 * fl[2])) ** 2) * smooth(fl[5] - 30, fl[5] + 40, y) * np.exp(-np.clip(y - fl[5], 0, None) / 160.) * self.wet * fl[4] * .9)[..., None] * rgb(fl[3])
        if seg in (1, 2) and t < 2: q = self.hand['tF']['R']; K.sparkle(h_, q[0], q[1], 54 - 18 * t, '#ffffff', 0, .10, .95); img = comp.over(img, h_.layer())                     # his hands, still ringing, far off
        if seg == 3 and t < 2: q = self.hand['tNa']['R']; K.sparkle(h_, q[0], q[1], 130 - 44 * t, '#ffffff', 0, .10, .95); img = comp.over(img, h_.layer())
        return comp.cam2d(img.astype(f32), W / 2 + shx, H / 2 + shy, 1.015) if seg else img

    # ------------------------------------------------------------------ 63-72  back to back
    def pose(self, f):
        t = f - 63; x, y = self.x, self.y; z = 1.16 - .05 * comp.ease_out(np.clip(t / 3., 0, 1)) + .003 * t; bg = comp.cam2d(np.array(self.glory, f32), 1010, 470, z, .015 * t)
        k = Cel()
        for i in range(26): a0 = 2 * math.pi * i / 26 + .012 * t; wd = .05 + .03 * ((i * 7) % 3); k.fill(np.array([(960 + math.cos(a0) * 120, 420 + math.sin(a0) * 120), (960 + math.cos(a0 - wd) * 1500, 420 + math.sin(a0 - wd) * 1500), (960 + math.cos(a0 + wd) * 1500, 420 + math.sin(a0 + wd) * 1500)]), '#fff4c0', .30)
        bg = comp.screen(bg, k.layer()[..., :3] * .8); bg = flare(bg, 960, 400, 720, '#fff0b0', .55)
        off = keyed(f, [(63, 150), (64, -18), (65, 8), (66, 0)]); d = f // 2 * 2; br = 2 * math.sin(d * .4)
        yl = Yuji.bust(YHead().pose(1222 + off, 428 + br, 160, .30, .06, .06), 'grin', 'sunset', d, (.5, .25), 2, byaw=.62, bpitch=.06); yl = rim(yl, -7, 3, '#fff6cc', .95, .7); yl = para(yl, (1300, 500), (1300, 1080), [(0, '#ffffff'), (1, '#8a5aa8')], 'mul', .7)
        s = 150; c = Cel(); lw = s / 300 * 8.6; pal = TP['glory']; H_ = Todo.pose(690 - off, 336 + br, s, -.30, .06, -.05); b = TBody.under(H_, -.52, .10, -.03, drop=1.28, fwd=.10, persp=.09); po = dict(Todo.POSES['crossed'], hR=('rest', 2, -10, -30), hL=None); po['R'], po['L'] = ((2.06, -.40, -.05), (2.22, -2.22, .60), (-.52, -1.66, 1.62), 1), ((-2.06, -.40, -.05), (-2.28, -2.12, .50), (1.10, -2.02, 1.20), -1)
        Todo.figure(c, H_, b, po, dict(TE['smug']), pal, 1, lw, d, .12, 3, front=('L', 'R')); tl = c.layer(); tl = rim(tl, 7, 3, '#fff6cc', .95, .7); tl = rim(tl, -4, 2, '#ff9ac0', .6, .6); tl = para(tl, (640, 480), (640, 1080), [(0, '#ffffff'), (1, '#8a5aa8')], 'mul', .7)
        fc = Cel(); HD.draw(fc, 'fist', HD.Cam(1404 + off, 716 + 3 * br, 168, 5.0), dict(skin='#f6c79c', shade='#a8685c', line='#22101a', light='#fff0cc', nail='#fbe0c8', nail_shade='#c58a78', cuff='#2a2140', cuff_shade='#171024', cuff_light='#7a5a9a', cuffin='#150e1e'), (-.7, .5, .4), 6.2, 4, side='L', R=HD.euler(160, 18, 14), bld=HD.HEAVY, style=dict(cuff=dict(y=.42, flare=1.)), res=360); fl_ = fc.layer(); fl_ = rim(fl_, 5, 3, '#fff6cc', .95, .6); fl_ = para(fl_, (1560, 600), (1560, 1080), [(0, '#ffffff'), (1, '#8a5aa8')], 'mul', .7); yl = yl * (1 - fl_[..., 3:4]) + fl_
        if t == 0: tl = smear(tl, 70, 0, 29); yl = smear(yl, -70, 0, 29)
        bg = comp.screen(bg, (qb(np.maximum(tl[..., 3], yl[..., 3]), 20) * .7)[..., None] * rgb('#fff0b0')); img = comp.over(comp.over(bg, yl), tl)
        g = Cel(); pts = [(960, 120, 120), (250, 180, 84), (1690, 160, 96), (1640, 760, 70), (180, 720, 64), (980, 610, 46), (470, 70, 44), (1440, 70, 40), (1800, 470, 50), (90, 440, 44)]
        for i, (px, py, R) in enumerate(pts):
            tt = t - 1 - i * .35
            if tt < 0: continue
            q = min(1.15, tt / 2.) if tt < 2.3 else 1 + .10 * math.sin(f * .9 + i); K.sparkle(g, px, py, R * q * 1.15, '#fff3c0', .05 * (i % 3 - 1), .10, .9); K.sparkle(g, px, py, R * q * .7, '#ffffff', .05 * (i % 3 - 1), .10)
        r = np.random.default_rng(6)
        for i in range(70): a = r.uniform(0, 6.283); dd = r.uniform(120, 1050) + 4 * t; K.sparkle(g, 960 + math.cos(a) * dd, 480 + math.sin(a) * dd * .6, r.uniform(4, 13) * (.6 + .4 * math.sin(f * .9 + i)), '#fffbe6', 0, .16, .8)
        gl = g.layer(); img = comp.over(img, gl) + qb(gl[..., :3], 12) * .5 * rgb('#ffdf90')
        if t < 2: img = img + (1 - t * .5) * .18
        return img
