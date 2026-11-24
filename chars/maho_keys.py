"""python -m chars.maho_keys 1 2 3 sheet [--scale 1.0]   -> qa/mahoraga/key1..3.jpg, sheet.jpg"""
import sys, os, time, math, numpy as np, cv2
from jj import W, H, comp
from jj.paint import rgb, f32, smooth, grid
from jj.cam import Camera3D
from . import mahoraga as M
from . import maho_ruin as RU
from .maho_body import V, merge, lerp_pose

OUT = 'qa/mahoraga'
KICK = lambda I=.03: dict(kind='dir', dir=(.80, .50, .12), I=I, shadow=False, rim=1.2, bounce=0, wrap=.0)         # cool bounce from low left: finds the grin
BACK = lambda I=.22: dict(kind='dir', dir=(-.35, -.75, -.45), I=I, shadow=False, rim=3.0, bounce=0, wrap=-.25)     # thin back rim


def a1_setup():
    m = M.Maho(M.KEYS['cock'], heading=-.65, wheel='burn'); mo = m.w(m.Hd(0, -.16, -.05))
    c = M.aim(mo + np.array((-.27, -.86, -.27)), [mo, m.wheel['c']], [(600, 745), (1325, 375)]); m.wheel['c'] = M.reproject(c, m.wheel['c'], (1292, 445)); m.wheel['R'] *= .95
    return m, c


def key1(scale=1.):
    m, c = a1_setup(); W0 = m.wheel['c']
    d = M.draw(m, c, [M.wheel_light(m), dict(kind='dir', dir=(.55, .75, .25), I=.040, shadow=False, rim=.3, bounce=0)], seed=3, scale=scale, fill=.018, teeth=.27)
    li, _ = M.burn_wheel(m.wheel, c, seed=1)
    im = np.zeros((H, W, 3), f32) + rgb('#020203')
    im = M.smoke(im, d['z'] * 0 + 60, c, W0, [(3.0, .55, 4, 2, 520, 0)], seed=4, d0=.55, I=.55, dark='#020203')
    im = im + M.rays(c, W0, 3, n=90)[..., None] * rgb('#c8d4e0') * .10
    im = comp.over(im, d['layer']) + li
    im = M.motes(im, c, 5, 0, n=200, box=((-1.2, 1.4), (-1.0, 1.), (2.2, 4.4)), light=W0, d0=.42, size=1.5)
    return comp.finish(im, 0, bloom=.30, vig=.18, grain=.008)


def air(W0, c, sky='#06080b', glow='#3a4a5c', dy=140, rx=1250, ry=980, p=1.6):
    q = c.project(W0[None])[0][0]; g = M.radial(q[0], q[1] + dy, rx, ry, power=p); return np.zeros((H, W, 3), f32) + rgb(sky) + g[..., None] * rgb(glow), g


def key2(scale=1.):
    m = M.Maho(M.KEYS['stride'], heading=-.62, wheel='burn'); c = M.aim(np.array((-1.2, -4.1, .25)), [m.wheel['c'], m.J['A-1']], [(1150, 185), (1000, 1010)]); W0 = m.wheel['c']
    R = RU.Ruin(seed=4, keep=(0, 0, 1.1, 1.6), eye=c.eye)
    b = M.draw(R, c, [dict(kind='point', pos=W0, I=4.2, d0=3.4, shadow=True, rim=.6, bounce=.02)], seed=9, scale=scale, lw=.6, hatch=.35, fill=.012, fill_dir=(0, -.3, 1), teeth=0, extra_curves=False, dry=.8, straight=True, term=.15); RU.add_windows(b, R, c, 3)
    d = M.draw(m, c, [M.wheel_light(m, I=3.8, d0=1.1), KICK(), BACK(.26)], seed=3, scale=scale, lw=.66, fill=.018, teeth=.34, blade=.7)
    li, _ = M.burn_wheel(m.wheel, c, seed=1); sky, g = air(W0, c); bz = b['z']
    im = comp.over(sky, b['layer']); fog = 1 - np.exp(-np.maximum(bz - 5, 0) * .042); fc = rgb('#080b10') + g[..., None] * rgb('#3a4a5c'); im = im * (1 - fog[..., None]) + fc * fog[..., None]
    im = M.smoke(im, bz, c, W0, [(26, .75, 6, 4, 420, 0), (15, .55, 3.2, 2.5, 380, 0)], seed=2, d0=3.0, I=1.1)
    back = [(x, y, z, R_, 9) for x, y, z, R_ in ((-3.6, 4.2, .2, 1.3), (-2.0, 3.4, .0, 1.0), (-.6, 4.8, .3, 1.5), (1.4, 3.6, .2, 1.2), (3.0, 3.0, .3, 1.5), (4.6, 2.2, .4, 1.5), (2.2, 1.9, -.1, .8), (-1.5, 2.0, -.2, .6), (5.2, 5, 1.2, 2.0), (-5.2, 6, .6, 1.9), (2.6, 6.5, 1.4, 2.0))]
    bl = M.draw_billow(M.Billow(back, seed=3), c, [dict(kind='point', pos=W0, I=6.0, d0=2.0, shadow=False, rim=1.4, bounce=.03, wrap=.0)], seed=4, scale=.5, soft=.85, fill=.004)
    im = M.zover(im, bl, bz); im = im + M.rays(c, W0, 3)[..., None] * rgb('#b8c8da') * .16
    im = comp.over(im, d['layer']); z2 = np.minimum(bz, d['z'])
    front = [(x, y, z, R_, 8) for x, y, z, R_ in ((1.3, -.7, -.12, .42), (.7, -.1, -.15, .40), (1.9, -.2, -.05, .55), (-.75, .5, -.12, .40), (2.6, -1.0, -.05, .60), (.35, .55, -.1, .45), (-1.6, .2, -.2, .45), (1.5, -1.7, -.2, .45))]
    fl = M.draw_billow(M.Billow(front, seed=8), c, [dict(kind='point', pos=W0, I=3.0, d0=1.6, shadow=False, rim=1.2, bounce=.03, wrap=.0)], seed=6, scale=.5, soft=.8, fill=.004)
    im = M.zover(im, fl, z2) + li
    im = M.smoke(im, z2, c, W0, [(4.1, .40, .40, .45, 330, 0), (2.2, .40, .12, .28, 420, 0)], seed=7, d0=2.0, I=.6)
    im = M.motes(im, c, 5, 0, n=170, box=((-3, 3), (-3.2, 3), (0, 4.6)), light=W0, d0=1.6, size=2.0)
    return comp.finish(im * 1.06, 0, bloom=.32, vig=.22, grain=.008)


def key3(scale=1.):
    K = merge(M.KEYS['sweep'], head=(-.62, -.62, -.10), neck=(-.30, .30, -.05), jaw=.5); m = M.Maho(K, heading=-.2, wheel='burn'); mo = m.w(m.jawf(0, -.16, -.08)); eye = M.blade_at(m, 1.)[0] + np.array((.70, -.5, .10))
    c = M.aim(eye, [mo, M.blade_at(m, .95)[0]], [(1215, 640), (225, 990)], f=860, roll0=.22); W0 = m.wheel['c']; m.wheel['c'] = W0 = M.reproject(c, W0, (c.project(W0[None])[0][0] + np.array((0, 12)))); m.wheel['R'] *= .93
    R = RU.Ruin(seed=11, keep=(0, 0, 1.4, 2.2), eye=c.eye, rubble=110)
    b = M.draw(R, c, [dict(kind='point', pos=W0, I=4.0, d0=3.2, shadow=True, rim=.6, bounce=.02)], seed=9, scale=scale, lw=.6, hatch=.35, fill=.010, fill_dir=(0, -.3, 1), teeth=0, extra_curves=False, dry=.8, straight=True, term=.15); RU.add_windows(b, R, c, 3)
    d = M.draw(m, c, [M.wheel_light(m, I=3.6, d0=1.05), KICK(.05), BACK(.20)], seed=3, scale=scale, lw=.92, fill=.018, teeth=.32, blade=.18)
    li, _ = M.burn_wheel(m.wheel, c, seed=1); sky, g = air(W0, c, glow='#33424f'); bz = b['z']
    im = comp.over(sky, b['layer']); fog = 1 - np.exp(-np.maximum(bz - 5, 0) * .045); fc = rgb('#080b10') + g[..., None] * rgb('#33424f'); im = im * (1 - fog[..., None]) + fc * fog[..., None]
    im = M.smoke(im, bz, c, W0, [(20, .7, 6, 4, 420, 0), (9, .5, 3.0, 2.0, 360, 0)], seed=2, d0=2.6, I=1.0)
    back = [(x, y, z, R_, 9) for x, y, z, R_ in ((-3.6, 4.2, .2, 1.3), (-2.6, 2.4, .0, .9), (1.6, 2.8, .2, 1.2), (3.0, 2.2, .3, 1.5), (4.6, 1.2, .4, 1.5), (2.2, 1.4, -.1, .8), (5.2, 4, 1.2, 2.0), (-5.2, 5, .6, 1.9))]
    bl = M.draw_billow(M.Billow(back, seed=5), c, [dict(kind='point', pos=W0, I=2.8, d0=1.9, shadow=False, rim=1.4, bounce=.03, wrap=.0)], seed=4, scale=.5, soft=.85, fill=.003); im = M.zover(im, bl, bz)
    im = im + M.rays(c, W0, 3)[..., None] * rgb('#b8c8da') * .16
    im = comp.over(im, M.sweep_trail(M.arc_blades(m, (-.92, .05, .75)), c, seed=4), .9)
    im = comp.over(im, d['layer']) + li
    im = M.smoke(im, np.minimum(bz, d['z']), c, W0, [(2.4, .45, 1.0, .35, 420, 0)], seed=7, d0=2.0, I=.5)
    im = M.motes(im, c, 5, 0, n=170, box=((-3, 3), (-3.2, 3), (0, 4.6)), light=W0, d0=1.6, size=2.0)
    return comp.finish(im * 1.04, 0, bloom=.32, vig=.2, grain=.008)


def sheet(scale=.45):
    """poses, angles, wheel, arm: 4 x 5 cells"""
    from jj.ink import spl
    cells = []; T = lambda im: cv2.resize(im[:, 420:1500], (480, 480), interpolation=cv2.INTER_AREA)
    def lab(im, t): a = (np.clip(im, 0, 1) * 255).astype(np.uint8).copy(); cv2.putText(a, t, (10, 470), 0, .5, (150, 165, 180), 1, cv2.LINE_AA); cells.append(a)
    def cam_(az, el, dist, tgt, f): a, e = math.radians(az), math.radians(el); tgt = np.array(tgt, float); return Camera3D(tgt + dist * np.array([math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)]), tgt, f)
    BG = np.zeros((H, W, 3), f32) + rgb('#1a2029')
    def shot(m, c, t, burn=True, lw=.8, bgc=BG, extra=(), **k):
        lt = [M.wheel_light(m, I=3.2 if burn else 0, d0=1.0)] if burn else [dict(kind='dir', dir=(.5, .7, -.6), I=.9, shadow=True, rim=1.5, bounce=.04)]
        d = M.draw(m, c, lt + [KICK(.06), BACK(.2)] + list(extra), seed=3, scale=scale, lw=lw, fill=.03, teeth=.3, blade=.5, **k); im = comp.over(bgc, d['layer'])
        if burn: im = im + M.burn_wheel(m.wheel, c, seed=1)[0]
        lab(T(comp.finish(im, 0, grain=.005)), t); return d
    A = dict(neck=(0, .2, 0), head=(0, -.2, 0))
    m = M.Maho(dict(wtilt=(.0, .1, 0), spin=.2)); c = cam_(-28, 8, 2.2, m.wheel['c'], 2100); shot(m, c, 'wheel, cold', burn=False, lw=.9)
    m = M.Maho(dict(wtilt=(.0, .1, 0), spin=.2), wheel='burn'); shot(m, cam_(-28, 8, 2.2, m.wheel['c'], 2100), 'wheel, burning', lw=.9)
    # notch strip
    strip = np.zeros((480, 480, 3), np.uint8)
    for i, f in enumerate((-9, -4, 0, 1, 2, 3, 5, 8, 12)):
        m.wheel['spin'] = M.notch(f); im = M.burn_wheel(m.wheel, cam_(0, 0, 2.2, m.wheel['c'], 1500), seed=i)[0] + 0; im = comp.finish(im, 0, grain=0); x, y = i % 3, i // 3; s_ = cv2.resize((np.clip(im[200:880, 620:1300], 0, 1) * 255).astype(np.uint8), (132, 132), interpolation=cv2.INTER_AREA); strip[12 + y * 150:12 + y * 150 + 132, 20 + x * 150:20 + x * 150 + 132] = s_; cv2.putText(strip, 'f%+d' % f, (24 + x * 150, 24 + y * 150), 0, .4, (150, 165, 180), 1, cv2.LINE_AA)
    cv2.putText(strip, 'notch(t): gakon', (10, 472), 0, .5, (150, 165, 180), 1, cv2.LINE_AA); cells.append(strip)
    m = M.Maho(dict(A, jaw=0.), wheel='burn'); shot(m, cam_(0, -4, 1.9, m.w(m.Hd(0, 0, .12)), 1500), 'head front, grin')
    m = M.Maho(dict(A, jaw=0.), wheel='burn'); shot(m, cam_(-42, -20, 1.8, m.w(m.Hd(0, 0, .10)), 1500), 'head low 3/4, grin')
    m = M.Maho(dict(A, jaw=0.), wheel='burn'); shot(m, cam_(-88, 0, 2.0, m.w(m.Hd(0, 0, .12)), 1500), 'head side, grin')
    R_ = dict(neck=(0, .30, 0), head=(0, -.60, 0), jaw=.6)
    m = M.Maho(R_, wheel='burn'); shot(m, cam_(0, -4, 1.9, m.w(m.Hd(0, 0, .10)), 1500), 'head front, roar')
    m = M.Maho(R_, wheel='burn'); shot(m, cam_(38, -18, 1.8, m.w(m.Hd(0, 0, .08)), 1500), 'head low 3/4, roar')
    m = M.Maho(R_, wheel='burn'); shot(m, cam_(86, 0, 2.0, m.w(m.Hd(0, 0, .08)), 1500), 'head side, roar')
    for k, t, az, el in (('stand', 'model, front', 0, -3), ('stand', 'model, 3/4 back', 150, 4)):
        m = M.Maho(M.KEYS[k]); shot(m, cam_(az, el, 6.6, (0, 0, 2.25), 1500), t, burn=False, lw=.55)
    m = M.Maho(M.KEYS['stride'], heading=-.6, wheel='burn'); shot(m, cam_(0, -5, 6.0, (0, 0, 2.1), 1450), 'stride A', lw=.55)
    m = M.Maho(M.KEYS['stride2'], heading=-.6, wheel='burn'); shot(m, cam_(0, -5, 6.0, (0, 0, 2.1), 1450), 'stride B', lw=.55)
    m = M.Maho(M.KEYS['sweep_a'], heading=-.1, wheel='burn'); shot(m, cam_(6, -6, 5.6, (0, 0, 2.0), 1450), 'sweep: anticipation', lw=.6)
    K = M.KEYS['sweep']; m = M.Maho(K, heading=-.1, wheel='burn'); c = cam_(6, -6, 5.6, (0, 0, 2.0), 1450); shot(m, c, 'sweep: smear', lw=.6, bgc=comp.over(BG, M.sweep_trail(M.arc_blades(m, (-.92, .05, .75)), c, seed=4), .85))
    m = M.Maho(M.KEYS['sweep_f'], heading=-.1, wheel='burn'); shot(m, cam_(6, -6, 5.6, (0, 0, 2.0), 1450), 'sweep: follow-through', lw=.6)
    m = M.Maho(M.KEYS['palm'], wheel='burn'); shot(m, cam_(-38, -8, 4.6, (.1, -.6, 2.2), 1250), 'palm: catches the slash', lw=.75)
    mf = M.Maho(merge(M.KEYS['cut'], cut=None), wheel='burn'); mc = M.Maho(M.KEYS['cut'], wheel='burn'); c = cam_(-8, -4, 4.6, (-.35, 0, 2.5), 1500); lt = [M.wheel_light(mf, I=3.2, d0=1.0), KICK(.07), BACK(.25)]
    df = M.draw(mf, c, lt, seed=3, scale=scale, lw=.7, fill=.03, teeth=.3, blade=.5); dc = M.draw(mc, c, lt, seed=3, scale=scale, lw=.7, fill=.03, teeth=.3); bw = M.burn_wheel(mf.wheel, c, seed=1)[0]
    im = BG
    for l in M.sever(df, dc, mf, c, .35, seed=2): im = comp.over(im, l)
    lab(T(comp.finish(im + bw, 0, grain=.005)), 'arm off: ink break')
    for t in (.35, .72): lab(T(comp.finish(comp.over(BG, M.regrow(df, dc, mf, c, t, seed=5)) + bw, 0, grain=.005)), 'arm knitting back %.2f' % t)
    m = M.Maho(M.KEYS['lunge'], wheel='burn'); shot(m, cam_(10, -10, 4.4, (0, -.8, 1.75), 1050), 'lunge at the lens', lw=.8)
    m = M.Maho(M.KEYS['roar'], wheel='burn'); shot(m, cam_(-12, -8, 5.2, (0, 0, 2.3), 1500), 'roar, full', lw=.65)
    m = M.Maho(M.KEYS['stand']); shot(m, cam_(-90, 0, 6.6, (0, 0, 2.25), 1500), 'model, side', burn=False, lw=.55)
    m = M.Maho(M.KEYS['lunge'], wheel='burn'); shot(m, cam_(-62, -4, 6.0, (0, -.5, 1.8), 1500), 'lunge, side', lw=.6)
    while len(cells) % 6: cells.append(np.zeros((480, 480, 3), np.uint8))
    return np.vstack([np.hstack(cells[i:i + 6]) for i in range(0, len(cells), 6)]).astype(f32) / 255


if __name__ == '__main__':
    cv2.setNumThreads(2); os.makedirs(OUT, exist_ok=True); a = sys.argv[1:]; sc = 1.0
    if '--scale' in a: i = a.index('--scale'); sc = float(a[i + 1]); del a[i:i + 2]
    for k in a:
        t = time.time(); im = globals()['key' + k](sc) if k in '123' else globals()[k](); comp.save(f'{OUT}/key{k}.jpg' if k in '123' else f'{OUT}/{k}.jpg', im, 92); print(k, '%.1fs' % (time.time() - t))
