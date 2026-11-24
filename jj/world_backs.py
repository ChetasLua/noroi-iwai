"""Heads seen from behind, drawn by hand for b6 (head_fn hooks of jj.figure).  Units: s = skull radius px, origin at ear level, y down."""
import numpy as np, math
from .paint import rgb, f32
from .ink import spl
from . import kit

MUL = np.array([.70, .60, .74], f32); INK = '#1a1220'


def c(h, k=1.): return np.clip(rgb(h) * MUL * k, 0, 1)


class T:
    """local -> screen; z = how far toward us a point sits on the skull (slides when the head turns)"""

    def __init__(s, i):
        s.x, s.y, s.s = i['x'], i['y'], i['s']; r = i['roll']; s.c, s.n = math.cos(r), math.sin(r); s.dy = (i['yaw'] - math.pi + math.pi) % (2 * math.pi) - math.pi; s.p = i['pitch']; s.f = i.get('f', 0)

    def __call__(s, P, z=None):
        P = np.atleast_2d(np.asarray(P, float)); u, v = P[:, 0].copy(), P[:, 1].copy()
        if z is not None: zz = np.sqrt(np.clip(.82 ** 2 - u ** 2 * .9, 0, 1)) * z; u = u * math.cos(s.dy) + zz * math.sin(s.dy); v = v - zz * .35 * math.sin(s.p)
        return np.c_[s.x + s.s * (u * s.c - v * s.n), s.y + s.s * (u * s.n + v * s.c)]


def wind(f, i, a=.045): return a * .6 + a * math.sin(f * .34 + i * 1.3)


def clump(cel, t, a, tip, b, fill, w, i, bulge=.14, bend=0., z=None, f=0, sh=None, line=True, amp=.05):
    tip = (tip[0] + wind(f, i, amp), tip[1] + .3 * wind(f, i + 2, amp)); P = kit.spike(a, tip, b, bulge, bend, 9); Q = t(P, z); cel.fill(Q, fill)
    if sh is not None: m = np.array([(a[0] * .25 + b[0] * .75), (a[1] * .25 + b[1] * .75)]); R = kit.spike(m, tip, b, bulge * .5, bend, 8); cel.fill(t(R, z), sh)
    if line: kit.outline(cel, Q, w, INK, i * 7 + 3, closed=False, press=.3)
    return Q


def neck(cel, t, w=.36, y0=.30, y1=1.05, col='#d99a82'):
    cel.fill(t([(-w, y0), (w, y0), (w * 1.06, y1), (-w * 1.06, y1)]), c(col, .82)); cel.fill(t([(-w, y0), (w, y0), (w, y0 + .34), (.06, y0 + .46), (-w, y0 + .30)]), c(col, .58))


def ear(cel, t, sx, y=.10, z=0., k=1.):
    P = np.array([(sx * .76, y - .20), (sx * .90, y - .22), (sx * .95, y - .06), (sx * .90, y + .14), (sx * .80, y + .24), (sx * .75, y + .12)]); cel.fill(spl(t(P), True, n=4), c('#e0a288', .9)); cel.line(spl(t(P[[1, 2, 3, 4]]), n=5), 1.6, INK, seed=3); cel.line(spl(t([(sx * .84, y - .10), (sx * .87, y), (sx * .83, y + .10)]), n=4), 1.1, '#5a2c34', seed=4)


def yuji(cel, H, i):
    t = T(i); f = i['f']; w = max(1.3, t.s * .030); base, lit, sh = c('#c9707f'), c('#eb9a9e', 1.06), c('#8a4462')
    neck(cel, t, .40)
    for sx in (-1, 1): ear(cel, t, sx)
    under = [(-.77, -.30), (-.79, .08), (-.68, .40), (-.46, .60), (-.24, .70), (-.10, .62), (.06, .74), (.22, .63), (.46, .60), (.68, .40), (.79, .08), (.77, -.30), (0, -.5)]
    cel.fill(spl(t(under), True, corners=(4, 5, 6, 7), n=5), c('#5a3038')); cel.fill(spl(t([(-.77, -.2), (-.79, .08), (-.68, .40), (-.5, .57), (-.42, .2), (-.5, -.2)]), True, n=5), c('#42222c'))
    for q in range(7): x = -.5 + q * .165; cel.line(t([(x, .16 + .1 * math.sin(q)), (x * .94, .46)], 1.), 1.0, '#33161f', seed=q, alpha=.8)
    # outer crown clumps, swept back and up
    outer = [((-.66, .02), (-1.00, .20), (-.74, -.20)), ((-.72, -.08), (-1.16, -.18), (-.72, -.40)), ((-.74, -.30), (-1.04, -.70), (-.56, -.74)), ((-.64, -.56), (-.66, -1.16), (-.24, -.90)), ((-.42, -.78), (-.10, -1.42), (.08, -.94)), ((-.04, -.90), (.36, -1.22), (.38, -.86)), ((.20, -.88), (.86, -1.28), (.60, -.70)), ((.46, -.76), (1.06, -.86), (.72, -.46)), ((.62, -.54), (1.30, -.50), (.76, -.18)), ((.70, -.26), (1.06, -.02), (.70, .04)), ((.66, -.04), (.90, .28), (.58, .14))]
    cel.fill(spl(t([(-.80, -.1), (-.78, -.55), (-.45, -.92), (0, -1.02), (.45, -.92), (.78, -.55), (.80, -.1), (.5, .06), (0, .12), (-.5, .06)]), True, n=6), base)
    for j, (a, tip, b) in enumerate(outer): clump(cel, t, a, tip, b, base if j not in (5, 7, 8) else lit, w, j, .15, .09, f=f, sh=sh if tip[0] < .2 else None)
    # whorl and clumps falling back over the undercut
    fall = [((-.64, -.46), (-.74, .30), (-.40, -.40)), ((.62, -.46), (.74, .28), (.38, -.40)), ((-.46, -.52), (-.44, .36), (-.14, -.44)), ((.46, -.54), (.50, .40), (.16, -.46)), ((-.22, -.60), (-.10, .26), (.10, -.52)), ((-.02, -.60), (.22, .44), (.30, -.52))]
    for j, (a, tip, b) in enumerate(fall): clump(cel, t, a, tip, b, (sh, base, base, lit, base, lit)[j], w * .85, 20 + j, .13, .04 * (-1) ** j, z=1., f=f, sh=sh if j in (2, 4) else None, amp=.03)
    for j, (x, y) in enumerate(((-.5, -.62), (-.2, -.76), (.12, -.80), (.42, -.70), (.64, -.5), (-.66, -.4))): cel.line(spl(t([(.04, -.58), ((x + .04) / 2, (y - .62) / 2 - .05), (x, y)], 1.), n=5), w * .55, '#5a2438', seed=40 + j)
    cel.fill(spl(t([(.18, -.90), (.50, -.78), (.66, -.56), (.50, -.66), (.24, -.78)]), True, n=4), lit)


def megumi(cel, H, i):
    t = T(i); f = i['f']; w = max(1.3, t.s * .028); base, lit, sh = c('#232a44'), c('#4a5a8c', 1.1), c('#0e1122')
    neck(cel, t, .37)
    for sx in (-1, 1): ear(cel, t, sx, .12)
    cel.fill(spl(t([(-.82, .16), (-.86, -.4), (-.5, -.96), (0, -1.06), (.5, -.96), (.86, -.4), (.82, .16), (.5, .58), (0, .70), (-.5, .58)]), True, n=6), base)
    spk = [((-.66, .22), (-1.08, .72), (-.74, -.06), sh), ((-.70, .04), (-1.52, .34), (-.78, -.24), base), ((-.74, -.14), (-1.78, -.24), (-.74, -.46), base), ((-.72, -.36), (-1.66, -.86), (-.58, -.68), lit), ((-.60, -.58), (-1.26, -1.42), (-.36, -.84), base), ((-.42, -.74), (-.74, -1.82), (-.10, -.92), base), ((-.16, -.84), (-.14, -2.02), (.16, -.90), lit), ((.10, -.84), (.52, -1.90), (.38, -.82), base), ((.34, -.76), (1.10, -1.62), (.60, -.64), lit), ((.54, -.60), (1.56, -1.06), (.72, -.40), base), ((.66, -.40), (1.82, -.44), (.76, -.14), lit), ((.72, -.16), (1.62, .20), (.76, .10), base), ((.70, .04), (1.16, .66), (.62, .30), sh)]
    ln = (.92, .80, .96, .74, .98, .82, .94, .80, .98, .78, .92, .70, .95)
    for j, (a, tip, b, col) in enumerate(spk): tip = (tip[0] * ln[j] * .94 + .06, tip[1] * ln[j] * .92); clump(cel, t, a, tip, b, col, w, j, .12, .045 * (-1) ** j, f=f, sh=sh if col is not sh and tip[0] < .3 else None, amp=.07)
    nape = [((-.62, .16), (-.62, .92), (-.30, .34)), ((-.40, .30), (-.26, 1.02), (-.04, .40)), ((-.12, .36), (.10, .92), (.20, .38)), ((.14, .34), (.42, 1.04), (.44, .30)), ((.38, .30), (.72, .84), (.64, .12))]
    for j, (a, tip, b) in enumerate(nape): clump(cel, t, a, tip, b, base if j % 2 else sh, w * .9, 30 + j, .12, .03, z=.6, f=f, amp=.03)
    crown = [((-.40, -.50), (-.86, .30), (-.10, -.40), base), ((.02, -.50), (-.28, .50), (.24, -.42), base), ((.16, -.52), (.40, .46), (.50, -.46), lit), ((.40, -.50), (.92, .22), (.62, -.50), base)]
    for j, (a, tip, b, col) in enumerate(crown): clump(cel, t, a, tip, b, col, w * .8, 40 + j, .12, .05 * (-1) ** j, z=.9, f=f, sh=sh, amp=.03)
    for j, (x, y) in enumerate(((-.8, -.7), (-.4, -1.1), (.1, -1.2), (.5, -1.05), (.9, -.6))): cel.line(spl(t([(x * .25, y * .5), (x * .62, y * .80), (x, y)]), n=5), w * .55, '#6a7ab0', seed=50 + j, alpha=.8)


def nobara(cel, H, i):
    t = T(i); f = i['f']; w = max(1.3, t.s * .030); base, lit, sh = c('#c4713f'), c('#f0a468', 1.04), c('#84412c'); g = lambda k, a=.035: wind(f, k, a)
    neck(cel, t, .31, .30, 1.1)
    # back panel under the hem, then the bob as five clean locks with gaps
    cel.fill(spl(t([(-.82, -.2), (-.88, .40), (-.70, .62), (0, .56), (.70, .62), (.88, .40), (.82, -.2)]), True, n=5), sh)
    cap = [(0, -1.10), (.46, -1.00), (.84, -.62), (.96, -.10), (.98, .22), (.5, .30), (0, .26), (-.5, .30), (-.98, .22), (-.96, -.10), (-.84, -.62), (-.46, -1.00)]
    cel.fill(spl(t(cap), True, n=6), base)
    locks = [[(-.98, -.10), (-1.00, .26), (-1.10 + g(1), .46), (-1.30 + g(1), .56), (-.92 + g(1), .70), (-.64, .46), (-.60, .06)], [(-.70, -.10), (-.72, .34), (-.74 + g(2), .56), (-.86 + g(2), .78), (-.52 + g(2), .64), (-.36, .34), (-.32, -.10)], [(-.40, -.10), (-.38, .38), (-.34 + g(3), .60), (-.36 + g(3), .80), (-.08 + g(3), .64), (.06, .34), (.04, -.10)], [(-.02, -.10), (.02, .38), (.16 + g(4), .60), (.28 + g(4), .80), (.40 + g(4), .60), (.44, .32), (.40, -.10)], [(.34, -.10), (.42, .36), (.60 + g(5), .58), (.82 + g(5), .76), (.80 + g(5), .52), (.80, .28), (.70, -.10)], [(.64, -.10), (.78, .32), (.96 + g(6), .50), (1.34 + g(6), .58), (1.12 + g(6), .38), (1.02, .20), (.98, -.10)]]
    for j, L in enumerate(locks):
        Q = spl(t(L, .55), True, corners=(3,), n=5); cel.fill(Q, (base, sh, base, base, sh, base)[j]); kit.outline(cel, Q[: int(len(Q) * .74)], w * .9, INK, 60 + j, closed=False)
    cel.fill(spl(t(cap), True, n=6), base); cel.fill(spl(t([(-.96, -.10), (-.84, -.62), (-.56, -.92), (-.60, -.4), (-.62, .26), (-.98, .22)]), True, n=5), sh)
    cel.fill(spl(t([(.10, -.86), (.46, -.78), (.70, -.46), (.60, -.36), (.36, -.56), (.10, -.64)]), True, corners=(0, 3), n=5), lit); cel.fill(spl(t([(-.48, -.74), (-.16, -.84), (-.18, -.64), (-.46, -.50)]), True, corners=(0, 2), n=4), lit)
    for j, x in enumerate((-.78, -.52, -.26, .0, .26, .52, .78)): cel.line(spl(t([(x * .12, -1.0), (x * .72, -.55), (x, -.05), (x * 1.02, .40)], .8), n=6), w * (.5 + .25 * (j % 2)), '#5a2a22', seed=70 + j, taper=(.1, .4))
    out = spl(t([(-1.02, .26), (-.97, -.10), (-.85, -.62), (-.47, -1.01), (0, -1.11), (.47, -1.01), (.85, -.62), (.97, -.10), (1.02, .26)]), n=6); cel.line(out, w * 1.1, INK, seed=81, taper=(.1, .1))
    cel.line(spl(t([(-.02, -1.10), (.02, -.92), (.0, -.74)], .7), n=4), w * .8, '#5a2a22', seed=90)


def nanami(cel, H, i):
    t = T(i); f = i['f']; w = max(1.3, t.s * .028); base, lit, sh = c('#c9a860'), c('#efd28c', 1.05), c('#8a6a42')
    neck(cel, t, .40, .25)
    # a slice of cheek and jaw on the side he looks to, temple of the glasses
    sd = -1 if t.dy > 0 else 1; k = min(1., abs(t.dy) / .6)
    if k > .15:
        F = np.array([(.50, -.46), (.80, -.36), (.88, -.10), (.93, .06), (.86, .22), (.86, .40), (.74, .74), (.44, .88), (.20, .60)]) * [sd, 1]; cel.fill(spl(t(F), True, corners=(3, 6), n=5), c('#e2a88c', .92)); cel.line(spl(t(F[1:8]), n=6, corners=(2, 5)), w, INK, seed=2); cel.fill(spl(t(np.array([(.30, .30), (.52, .34), (.60, .70), (.42, .86), (.20, .62)]) * [sd, 1]), True, n=4), c('#e2a88c', .64)); cel.line(t(np.array([(.36, -.02), (.90, -.06)]) * [sd, 1]), w * 1.2, '#1c1a20', seed=3)
    hair = [(-.80, .02), (-.86, -.42), (-.62, -.94), (-.12, -1.14), (.44, -1.08), (.80, -.64), (.85, -.06), (.74, .24), (.56, .56), (.22, .72), (-.22, .72), (-.56, .56), (-.74, .24)]
    cel.fill(spl(t(hair), True, n=6), base); cel.fill(spl(t([(-.74, .10), (-.60, .02), (-.2, .10), (.2, .10), (.6, .02), (.76, .10), (.56, .56), (.22, .72), (-.22, .72), (-.56, .56)]), True, n=5), sh)
    cel.fill(spl(t([(.06, -1.02), (.46, -.94), (.70, -.56), (.56, -.40), (.28, -.66), (.02, -.80)], .6), True, corners=(0, 3), n=5), lit); cel.fill(spl(t([(-.60, -.70), (-.30, -.90), (-.22, -.70), (-.52, -.44)], .6), True, corners=(0, 2), n=4), lit)
    # combed lines from the parting to the nape
    for j, (x0, y0, x1, y1, x2, y2) in enumerate(((-.30, -1.08, -.66, -.60, -.70, .04), (-.28, -1.08, -.40, -.50, -.46, .30), (-.24, -1.10, -.10, -.50, -.18, .40), (-.20, -1.10, .22, -.60, .12, .44), (-.14, -1.12, .48, -.70, .40, .32), (-.06, -1.14, .70, -.74, .64, .10), (.1, -1.14, .82, -.6, .78, -.06))): cel.line(spl(t([(x0, y0), (x1, y1), (x2, y2)], .7), n=7), w * .7, '#6a4c30', seed=10 + j, taper=(.08, .5))
    for j in range(6): x = -.44 + j * .18; cel.line(t([(x, .40), (x * .9, .66)], .6), w * .55, '#5a4030', seed=30 + j)
    kit.outline(cel, spl(t(hair), True, n=6), w, INK, 50)
    ex = .10 * -sd + sd * .52 if k > .15 else .84
    E = np.array([(ex - .10, -.10), (ex + .04, -.16), (ex + .12, -.02), (ex + .06, .18), (ex - .04, .28), (ex - .10, .12)]) if k > .15 else None
    if E is not None: cel.fill(spl(t(E), True, n=4), c('#e2a88c')); kit.outline(cel, spl(t(E), True, n=4), w * .9, INK, 60); cel.line(spl(t(E[[0, 2, 3]] * .5 + E.mean(0) * .5), n=4), w * .6, '#5a2c34', seed=61)
    else:
        for sx in (-1, 1): ear(cel, t, sx)
    # shirt collar showing blue over the jacket
    cel.fill(spl(t([(-.46, .80), (-.2, .74), (.2, .74), (.46, .80), (.50, 1.0), (0, .96), (-.50, 1.0)]), True, n=4), c('#4a6a9a')); cel.line(spl(t([(-.46, .80), (-.2, .74), (.2, .74), (.46, .80)]), n=4), w * .8, INK, seed=70)


HEADS = dict(yuji=yuji, megumi=megumi, nobara=nobara, nanami=nanami)


def mitt(cel, x, y, k, flip=1, rim='#ffc27a', seed=0):
    """a hand wrapped over the rail: x, y = rail top under the hand, k = hand width px"""
    P = np.array([(-.46, -.34), (.10, -.46), (.50, -.30), (.58, .10), (.50, .58), (.26, .72), (.02, .70), (-.24, .74), (-.46, .56), (-.54, .10)]) * [flip * k, k] + [x, y]; Q = spl(P, True, n=5)
    cel.fill(Q, c('#e6a98e', .98)); cel.fill(spl(np.array([(-.50, .26), (.0, .34), (.54, .22), (.50, .58), (.26, .72), (.02, .70), (-.24, .74), (-.46, .56)]) * [flip * k, k] + [x, y], True, n=4), c('#b9766a', .92)); kit.outline(cel, Q, max(1.1, k * .055), INK, seed)
    for u in (-.14, .20): cel.line(spl(np.array([(u, .10), (u + .03, .40), (u + .01, .68)]) * [flip * k, k] + [x, y], n=4), max(.9, k * .040), '#4a2030', seed=seed + int(u * 100) + 50, taper=(.5, .1))
    T_ = np.array([(-.52, -.10), (-.74, .10), (-.66, .34), (-.46, .22)]) * [flip * k, k] + [x, y]; cel.fill(spl(T_, True, n=4), c('#e6a98e', .9)); kit.outline(cel, spl(T_, True, n=4), max(.9, k * .04), INK, seed + 9)
    cel.line(spl(np.array([(-.40, -.34), (.10, -.45), (.46, -.30)]) * [flip * k, k] + [x, y], n=4), max(1., k * .06), rim, seed=seed + 3, alpha=.95)


def ruffle(cel, x, y, k, sway, seed=0):
    """Gojo's hand buried in hair, seen from behind: x, y = knuckles, fingers go down and left"""
    R = lambda P: np.array(P, float) @ np.array([[math.cos(sway), math.sin(sway)], [-math.sin(sway), math.cos(sway)]]) * k + [x, y]
    for j, (u, L, a) in enumerate(((-.52, .86, -.30), (-.22, 1.0, -.10), (.10, .98, .10), (.40, .80, .30))):
        d = np.array([math.sin(-a), math.cos(a)]); p = np.array([u, .05]); n = np.array([d[1], -d[0]]) * .15; P = R([p - n, p + n, p + d * L * .7 + n * .9, p + d * L, p + d * L * .7 - n * .9]); cel.fill(spl(P, True, corners=(0, 1), n=4), c('#eab096')); kit.outline(cel, spl(P, True, corners=(0, 1), n=4), max(1., k * .05), INK, seed + j, closed=True)
    P = R([(-.70, .16), (-.60, -.36), (-.10, -.56), (.46, -.46), (.66, .0), (.50, .24), (0, .30)]); cel.fill(spl(P, True, n=5), c('#eab096')); kit.outline(cel, spl(P, True, n=5)[:26], max(1., k * .05), INK, seed + 8, closed=False); cel.line(spl(R([(-.56, -.36), (-.10, -.54), (.46, -.44)]), n=4), max(1., k * .07), '#ffd59a', seed=seed + 11)
    P = R([(.50, -.30), (.86, -.04), (.96, .34), (.76, .40), (.56, .10)]); cel.fill(spl(P, True, n=4), c('#eab096', .9)); kit.outline(cel, spl(P, True, n=4), max(1., k * .05), INK, seed + 12)
    for j, (u, v, tx, ty) in enumerate(((-.5, .95, -.85, .30), (-.1, 1.1, -.2, .35), (.3, 1.05, .46, .34), (.62, .9, 1.0, .25))): cel.fill(kit.spike(R([(u - .14, v)])[0], R([(tx, ty)])[0], R([(u + .14, v)])[0], .14, .05), c('#232a44'))
