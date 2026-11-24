"""a1  MAHORAGA'S WHEEL: black, breath; scribbled light writes the eight-handled wheel; what hangs under it grins; gakon; into the teeth."""
import numpy as np, cv2, math
from jj import W, H, comp
from jj.paint import rgb, f32, fbm, smooth, grid
from jj.comp import ease, blur
from jj.cam import Camera3D
from jj import kit
from chars import mahoraga as M, maho_shot as MS, maho_fight as F
from chars.maho_body import V, unit, Rz

ID = 'a1'
IGN = [9, 18, 23, 27, 30, 33, 36, 39]
ORDER = [4, 3, 2, 1, 0, 7, 6, 5]                 # from the top, clockwise on screen (the wheel frame is rolled; set_order checks it)
BASE = dict(neck=(0, .22, .08), head=(.02, -.14, .10), chest=(0, .10, 0), crest=(.1, -.22, .80), wing=.05, wingL=-.16, jaw=.07)


def creature(**kw):
    m = M.Maho(M.merge(BASE, **kw), heading=-.45, wheel='burn'); return m


_set = {}
def setup_():
    if _set: return _set
    m = creature(); mo = m.w(m.Hd(0, -.16, -.05)); c = M.aim(mo + np.array((-.12, -.97, -.30)), [mo, m.wheel['c']], [(590, 745), (1290, 335)], roll0=0)
    wh = dict(m.wheel); wh['c'] = M.reproject(c, m.wheel['c'], (1188, 366), -.12); wh['R'] *= .88
    q = M._wp(dict(wh, spin=0.), c, np.array([[math.cos(i * math.pi / 4), math.sin(i * math.pi / 4)] for i in range(8)])) - c.project(wh['c'][None])[0]; an = np.mod(np.arctan2(q[:, 0], -q[:, 1]) + .35, 2 * math.pi); ORDER[:] = [int(i) for i in np.argsort(an)]
    _set.update(m=m, c=c, wh=wh, mouth=mo, head=m.fh, hs=m.hs); return _set


def wheel(spin=0., lean=0.):
    """wheel dict; lean = radians tipped back about its horizontal axis"""
    wh = dict(_set['wh']); wh['spin'] = spin; ca, sa = math.cos(lean), math.sin(lean); ey, ez = wh['ey'], wh['ez']; wh['ey'] = ey * ca - ez * sa; wh['ez'] = ez * ca + ey * sa; wh['c'] = wh['c'] - ez * sa * wh['R'] * .5; return wh


MARK = 7                                         # ORDER slot of the cracked orb


def lights(level, flare=1., spin=0., amt=.62):
    """level 0..8 orbs lit, 9 = full blaze"""
    s = setup_(); wh, c = s['wh'], s['c']; k = min(level, 8) / 8.
    pts = [wh['c'] + wh['R'] * (math.cos(i * math.pi / 4) * wh['ex'] + math.sin(i * math.pi / 4) * wh['ey']) for i in ORDER[:max(1, min(8, level))]]
    P = np.mean(pts, 0) * (1 - k) + wh['c'] * k; b = (.10 + .62 * k ** 1.5 if level < 9 else 1.) * flare
    Lf = P - c.R[2] * .75 - c.R[0] * .10; gw = dict(wheel(spin)); g = dict(gobo=(gw, amt * min(1., k * 1.1)), mark=ORDER[MARK])
    return [dict(M.wheel_light(s['m'], I=2.35 * b, d0=.46 + .38 * b, off=P - s['m'].wheel['c']), **g), dict(g, kind='point', pos=Lf, I=.56 * b ** 1.35, d0=.55 + .45 * b, shadow=True, rim=.5, bounce=.06, wrap=.08, soft=.06),
            dict(kind='dir', dir=(.55, .75, .25), I=.022 * b ** 2, shadow=False, rim=.3, bounce=0)], b


def d_level(level):
    s = setup_(); L, b = lights(level); return M.draw(s['m'], s['c'], L, seed=3, scale=.62, fill=.013 * b ** 2, teeth=.62 * b ** 1.6, hard=.6)['layer']


def d_turn(f):
    """50..59: the same creature while the wheel turns: beams and spoke shadows sweep across horn, wings, teeth"""
    s = setup_(); sp, le = spin_lean(f); u = f - 54
    if f < 54: kw = {}
    elif f < 57: kw = dict(head=(.02, -.06, .13), neck=(0, .27, .08), wing=-.02, wingL=-.22, jaw=.10, ruffle=.5)
    else: kw = dict(head=(.02, -.16, .09), wing=.07, wingL=-.14, jaw=.12, ruffle=.2)
    m = creature(**kw) if kw else s['m']; L, b = lights(9, 1. if f < 54 else 1 + .38 * math.exp(-u / 3.5), spin=sp); return M.draw(m, s['c'], L, seed=3, scale=.62, fill=.014, teeth=.64)['layer']


def d_pose(name):
    s = setup_(); kw = dict(jolt=dict(head=(.02, -.06, .13), neck=(0, .27, .08), wing=-.02, wingL=-.22, jaw=.10, ruffle=.5), settle=dict(head=(.02, -.16, .09), wing=.07, wingL=-.14, jaw=.12, ruffle=.2))[name]
    m = creature(**kw); L, b = lights(9, 1.35 if name == 'jolt' else 1.1); return M.draw(m, s['c'], L, seed=3, scale=.62, fill=.014, teeth=.64)['layer']


def push_cam(f):
    s = setup_(); c = s['c']; t = min(1., max(0., (f - 60) / 11.)); k = .03 * t + .97 * t ** 2.4; R, o = s['head']; hs = s['hs']; fw = R @ V(0, -1, 0); fw = s['m'].wd(fw)
    tgt = s['m'].w(s['m'].Hd(0, -.10, -.075 - .045 * t)); inside = tgt - fw * .03
    eye = c.eye + (inside + fw * .02 - c.eye) * k; look0 = c.eye + c.R[2] * 2.; d0 = unit(c.R[2]); d1 = unit(tgt - fw * .3 - eye); q = ease(min(1, t * 1.15)); d = unit(d0 * (1 - q) + d1 * q)
    up0 = -c.R[1]; n = Camera3D(eye, eye + d, c.f * (1 + .0 * t), cx=c.cx, cy=c.cy); rt = unit(np.cross(d, up0)); u2 = np.cross(rt, d); n.R = np.stack([rt, -u2, d]); return n, t


def d_push(f):
    s = setup_(); c, t = push_cam(f); e = ease(t); m = creature(jaw=.12 + .46 * e, neck=(0, .22 - .22 * e, .08), head=(.02, -.16 - .20 * e, .09), wing=.07, wingL=-.14); L, b = lights(9, 1.1, spin=math.pi / 4); L[0]['I'] *= 1 - .6 * t; L[1]['I'] *= 1 - .92 * t; L = L[:2]
    d = M.draw(m, c, L, seed=3, scale=.58, fill=.014 * (1 - .7 * t), teeth=.62, lw=1. + 1.2 * t, hatch=1 - .6 * t)
    li = M.burn_wheel(wheel(math.pi / 4), c, seed=100 + f // 2, mark=ORDER[MARK])[0][..., 1] if f < 69 else np.zeros((H, W), f32)
    return dict(l=d['layer'], li=li)


def spin_lean(f):
    sp = M.notch(f - 54) if f >= 45 else 0.; le = 0.
    if 46 <= f < 54: le = math.radians(5) * ease(min(1, (f - 49) / 4.)) if f > 49 else 0.
    if f >= 54: le = math.radians(4) * max(0, 1 - (f - 53) / 2.) - math.radians(2.2) * math.exp(-(f - 55) / 2.2) * math.cos((f - 55) * 1.1) * (f >= 55)
    return sp, le


def d_wheel(f):
    """scribbled light at drawing f (one per 2 frames): mono additive"""
    s = setup_(); prog = np.zeros(8); ign = np.zeros(8)
    for k, (t0, i) in enumerate(zip(IGN, ORDER)):
        lead = 5 if k < 2 else 3
        prog[i] = np.clip((f - t0 + lead) / lead, 0, 1) if k else np.clip((f - 2) / 5., 0, 1); ign[i] = 0 if f < t0 else 1 + .75 * math.exp(-(f - t0) / 2.2)
    sp, le = spin_lean(f); rim = np.clip((f - 14) / 26., 0, 1); core = np.clip((f - 8) / 36., 0, 1) ** 1.3 * .62 ; fl = 1 + (.9 * math.exp(-(f - 45) / 2.4) if f >= 45 else 0) + (1.1 * math.exp(-(f - 54) / 2.6) if f >= 54 else 0)
    if f < 9: core = .10 + .02 * (f - 4)
    li = M.burn_wheel(wheel(sp, le), s['c'], seed=10 + f, prog=prog, ignite=ign * (1 if f < 45 else 1), rim=rim, core=core * fl ** .6, width=.95 + .12 * (fl - 1), mark=ORDER[MARK] if f >= IGN[MARK] else None)[0][..., 1]
    return li


VER = 7
JOBS = {f'lv{k}': ('d_level', (k,)) for k in range(1, 10)}
TF = [50, 52, 53, 54, 55, 56, 57, 58, 59]
JOBS.update({f'tu{f}': ('d_turn', (f,)) for f in TF})
WF = sorted(set(range(4, 50, 2)) | set(IGN) | {45, 50, 52, 53} | set(range(54, 60)))
JOBS.update({f'wh{f:02d}': ('d_wheel', (f,)) for f in WF})
JOBS.update({f'pu{f:02d}': ('d_push', (f,)) for f in range(60, 72)})


class Shot:
    id = ID; n = 72
    finish = dict(bloom=.30, vig=.18, grain=.009)
    EVENTS = [(0, 'breath', dict(note='one slow exhale out of the black, 0-8')),
              (4, 'scratch', dict(note='first light: it scribbles out along the first spoke, 4-9'))] + \
             [(t, 'ignite', dict(n=k, size=.5 + .06 * k, note='orb %d catches: white-hot hiss and struck bell, each higher' % (k + 1))) for k, t in enumerate(IGN)] + \
             [(45, 'blaze', dict(size=1.0, note='wheel at full burn, the thing under it is grinning')), (50, 'creak', dict(note='wheel leans back 50-53: wood and iron strain')),
              (54, 'gakon', dict(size=1.0, note='one notch (8 back, 45, 12 over), 3 frames, the heaviest hit of the opening; beams sweep the horn and wings')), (56, 'dust', dict(size=.5)), (57, 'bounce', dict(size=.35, note='overshoot settles')),
              (60, 'push', dict(note='accelerating rush 60-72')), (60, 'jaw', dict(note='jaw creaks open')), (71, 'cut_black', {})]

    def setup(self):
        MS.ensure(ID); setup_(); s = _set
        self.c, self.W0 = s['c'], s['wh']['c']; z = np.full((H, W), 60., f32); bg = np.zeros((H, W, 3), f32) + rgb('#020203')
        self.smk = M.smoke(bg * 0, z, self.c, self.W0, [(3.0, .55, 4, 2, 520, 0)], seed=4, d0=.55, I=.6, dark='#000000'); self.ry = M.rays(self.c, self.W0, 3, n=90)
        x, y = grid(); n = fbm(H, W, (170, 64, 24, 10), (1, .6, .35, .2), 12); self.n1 = n; self.n2 = fbm(H, W, (240, 90, 30), (1, .5, .3), 13); self.x, self.y = x, y; self.smk = self.smk.astype(np.float16); self.ry = self.ry.astype(np.float16); self.n1 = self.n1.astype(np.float16); self.n2 = self.n2.astype(np.float16)
        r = np.random.default_rng(8); q = self.c.project(self.W0[None])[0][0]; N = 340
        th = r.uniform(0, 6.28, N); rr = r.uniform(150, 470, N); hp = self.c.project(s['m'].horn_w)[0]; ix = r.integers(4, len(hp) - 2, N); on = r.random(N) < .62; bx = np.where(on, hp[ix, 0] + r.normal(0, 26, N), q[0] + np.cos(th) * rr); by = np.where(on, hp[ix, 1] + r.uniform(10, 60, N), q[1] + np.sin(th) * rr * .75); self.dust = np.c_[bx, by, r.normal(0, 26, N), r.uniform(-30, 60, N), r.uniform(.6, 2.2, N), r.integers(0, 5, N)]
        return self

    def mist(self, f):
        """breath: rolls out of the mouth to the lower left"""
        t = (f + 2) / 12.; cx, cy = 560 - 150 * t, 770 + 40 * t; R = 120 + 300 * t; d = np.sqrt(((self.x - cx) / (R * 1.5)) ** 2 + ((self.y - cy) / R) ** 2)
        n = np.roll(self.n1, (int(-6 * f), int(-14 * f)), (0, 1)).astype(f32); a = np.clip(1 - d, 0, 1) ** 1.3 * smooth(-.5, 1.5, n + .5 * np.roll(self.n2, int(9 * f), 1).astype(f32)); env = math.sin(math.pi * min(1, (f + 1) / 13.)) ** .8
        return (a * env * .16)[..., None] * rgb('#9fb0c2')

    def creature(self, f):
        if f < 4: return None
        if f < IGN[0]: a = MS.get(ID, 'lv1').full(); a[..., :3] *= (.30, .42, .50, .55, .60)[f - 4]; return a
        if f >= 50: return MS.get(ID, 'tu%d' % max(t for t in TF if t <= f))
        k = sum(f >= t for t in IGN) if f < 45 else 9; t0 = (IGN + [45])[k - 1] if k < 9 else 45; u = f - t0
        s = (.55, 1.18, 1.10, 1.03)[u] if u < 4 else 1.
        a = MS.get(ID, f'lv{k}'); b = MS.get(ID, f'lv{k - 1}') if k > 1 else None
        if abs(s - 1) < 1e-3: return a
        a = a.full(); b = b.full() if b is not None else None; out = a.copy(); lo = b[..., :3] if b is not None else np.zeros_like(a[..., :3]); out[..., :3] = np.maximum(lo + (a[..., :3] - lo) * s, 0); return out

    def frame(self, f):
        im = np.zeros((H, W, 3), f32) + rgb('#020203')
        if f >= 60:
            d = MS.get(ID, f'pu{f:02d}'); t = (f - 60) / 11.; im = im + self.smk.astype(f32) * (1 - t); d['l'].over(im); F.lit(im, d['li'], M.COOL * (1 - .6 * t))
            z = 1.045; im = comp.cam2d(im, 700 + (W / 2 - 700) / z, 600 + (H / 2 - 600) / z, z)
            if f >= 64: im = kit.zoomblur(im, 640, 640, .02 + .08 * (t - .35), 9)
            if f >= 69: im = im * (1., .55, .12)[f - 69]
            return im.astype(f32)
        wf = max([w for w in WF if w <= f], default=None); li = MS.get(ID, f'wh{wf:02d}') if wf is not None else None
        b = 0. if f < 4 else min(1., (f - 3) / 42.) ** 1.3
        im = im + self.smk.astype(f32) * b + self.ry[..., None].astype(f32) * rgb('#c8d4e0') * .14 * b
        L = self.creature(f)
        if L is not None: im = L.over(im) if hasattr(L, 'over') else comp.over(im, L)
        if li is not None:
            fl = 1.; fk = .94 + .06 * math.sin(f * 2.3) if f >= 9 else (.75 if f == 4 else 1.)
            if f in (54, 55): fl = 1.15
            F.lit(im, li, M.COOL * fl * fk)
        if f < 14: im = im + self.mist(f)
        if f >= 45:
            q = self.c.project(self.W0[None])[0][0]; e = math.exp(-(f - 45) / 2.2) if f < 54 else math.exp(-(f - 54) / 2.0) * .9
            if e > .04: im = kit.flare(im, q[0], q[1], 620, '#dfeaf8', .32 * e, .55, 0.02)
        if f >= 54:                                                             # dust shaken loose, falling through the light
            pen = M.Pen(); u = f - 54
            for x, y, vx, vy, s, d in self.dust:
                k = u - d * .4
                if k < 0: continue
                p = np.array([x + vx * k * .25, y + vy * k * .10 + 1.25 * k * k]); pen.add(np.array([p, p - np.array([vx * .2, 2.2 * k + 2])]), s * .55, .75 * max(0, 1 - k / 14))
            im = im + pen.raster(H, W)[..., None] * rgb('#dfe8f2') * .8
        z = 1 + .045 * ease(max(0., (f - 9) / 51.)); dx = dy = 0.
        if 54 <= f <= 58: dx, dy = kit.shake(f, (26, 20, 12, 5, 2)[f - 54], 5)
        if 54 <= f <= 56: z *= (1.030, 1.020, 1.008)[f - 54]
        if z > 1.0005 or dx or dy: cx, cy = 700 + (W / 2 - 700) / z, 600 + (H / 2 - 600) / z; im = comp.cam2d(im, cx + dx, cy + dy, z, .004 * np.sign(dx) if dx else 0)
        if f in (54,): im = kit.smear(im, 10, 16, 7)
        return im.astype(f32)
