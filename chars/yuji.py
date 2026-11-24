"""YUJI ITADORI: drawn on jj.head3.Head (default proportions).

    from chars.yuji import Yuji, YHead
    H = YHead().pose(cx, cy, s, yaw, pitch, roll)          # cx,cy = centre of the head on the eye line; s = px from eye line to chin (a plain jj Head works too)
    layer = Yuji.bust(H, 'laugh', 'sunset', f, wind, seed)  # premultiplied RGBA 1080x1920 (w=, h= for other sizes), head + neck + hood + jacket, rim light and gradients included
    Yuji.head(cel, H, expr, light, f, wind, seed)           # head only, drawn on your own Cel (back=False / front=False draw one half, to put a neck or hood between)
expr   name | dict | (a, b, t) blend.  names: neutral smile grin laugh laughshut surprised fear panic grief scream(_a _b _c) rage snarl tired gentle shut;  slim=0..1 narrows jaw and chin (an expression key, 1 in gentle)
       Yuji.expr('rage', gx=.02, gy=-.02, sweat=1)  overrides;  Yuji.blend(a, b, t);  Yuji.lids(x, t) closes the eyes (blink), keeps the rest
light  name | dict.  names: day sunset under rain fire snow cursed ash;  Yuji.light('sunset', side=-1, amt=.3, ...) mirrors / overrides (amt: -1 no side shadow, 0 side plane, 1 half face)
wind   scalar or (x, y) in screen space (y up), 0..1; f drives it.   wet 0..1 soaked hair + drops on skin.   damage 0..1 cuts, soot, rips, scorched hood
bust   byaw (default .55 yaw) bpitch broll, base=(x,y) neck base in px, raise_ shoulders up, breath 0..1, rims=False, head=False, hood=False
seed   hold it while the drawing holds
"""
import numpy as np, cv2, math, cairo
from jj import W, H as HH
from jj.head3 import Head
from jj.ink import Cel, spl, resample, ribbon, noise1
from jj.kit import outline, shape, spike, contours, svg, s1, rim, para, tint, aura, hatch, splatter
from jj.paint import rgb, f32, smooth, mix
from jj import comp

INK = '#17131a'
A = np.array


# ------------------------------------------------------------------ head with a jaw that opens and a clamped projection
class YHead(Head):
    """Head + jaw opening (units) + Pc: face projection that never folds over the far silhouette"""
    jopen = 0.; jx = 0.; slim = 0.; warp = None          # warp(P screen px) -> P: jelly / squash applied to everything drawn on the head

    @classmethod
    def of(cls, H, jaw=0., jx=0., slim=0.):
        q = cls.__new__(cls); q.__dict__.update(H.__dict__)          # always a copy: the caller's head is never changed
        q.jopen = jaw; q.jx = jx; q.slim = slim; q._xs = None; q._fc = None
        q.k = dict(q.k); w_, yt, ytip, yb, d_ = H.k['nose']
        if abs(yb + .47) < .004: q.k['nose'] = (w_, yt, -.06 + (ytip + .06) * .86, -.06 + (yb + .06) * .86, d_ * .90)      # Yuji's nose: 14 % shorter than the master head
        q.k['cheek'] = max(q.k['cheek'], .052); q.k['socket'] = max(q.k['socket'], .062)
        return q

    def _mesh(self, y0, y1, half, ny=90, nf=90):
        """as Head._mesh with a firmer jaw angle (the shell is cut further back down to the corner of the jaw)"""
        if not half: return Head._mesh(self, y0, y1, half, ny, nf)
        y = np.linspace(y0, y1, ny)
        zj = np.interp(y, [-1.0, -.995, -.96, -.91, -.85, -.78, -.70, -.62, -.54, -.42, -.30, -.2], [.64, .64, .59, .51, .385, .235, .085, -.015, -.055, -.085, -.10, -.10])
        A, F, B = (np.interp(y, self.yy, t) for t in (self.A, self.F, self.B)); p = self.k['p']
        c = np.clip(zj / np.maximum(np.where(zj >= 0, F, B), 1e-3), -1, .98)
        t = np.where(c >= 0, np.arctan2((1 - np.abs(c) ** p) ** (1 / p), np.abs(c)), np.pi / 2 + np.arcsin(np.clip(-c, 0, 1)))
        PH = np.linspace(-1, 1, nf)[None] * np.minimum(math.pi * .5 * half, t)[:, None]
        sx, cz = np.sin(PH), np.cos(PH); r = (np.abs(sx) ** p + np.abs(cz) ** p) ** (-1 / p)
        X = A[:, None] * sx * r; Z = np.where(cz >= 0, F[:, None], B[:, None]) * cz * r; Y = np.repeat(y[:, None], nf, 1); fr = np.clip(cz, 0, 1)
        Z = Z + fr * (self._mid(X, Y) + self.k['cheek'] * np.exp(-((Y + .22) / .13) ** 2) * np.exp(-((np.abs(X) - .44) / .2) ** 2) * 1.4
                      - self.k['socket'] * np.exp(-((Y - .02) / .10) ** 2) * np.exp(-((np.abs(X) - .40) / .24) ** 2) * 1.3)
        return np.stack([X, Y, Z], -1)

    def P3(self, Q):
        Q = np.array(Q, float)
        y = Q[..., 1]; Q[..., 1] = np.where(y < -.44, -.44 + (y + .44) * .925, y)           # lower face a touch shorter than the master head
        if self.slim: w = smooth(-.16, -.78, Q[..., 1]) * smooth(-.45, .10, Q[..., 2]); Q[..., 0] = Q[..., 0] * (1 - .125 * self.slim * w); Q[..., 1] = Q[..., 1] + .030 * self.slim * smooth(-.66, -.95, Q[..., 1]) * smooth(-.2, .3, Q[..., 2])
        if self.jopen or self.jx:
            w = smooth(-.56, -.92, Q[..., 1]) * smooth(-.25, .40, Q[..., 2])
            Q[..., 0] = Q[..., 0] + self.jx * w * smooth(-.50, -.80, Q[..., 1])
            Q[..., 1] = Q[..., 1] - self.jopen * w
            Q[..., 2] = Q[..., 2] - self.jopen * .25 * w
        P = Head.P3(self, Q)
        return P if self.warp is None else self.warp(P)

    def face(self, top=.75):
        F = Head.face(self, top); n = max(3, int(.075 * self.s / 3.0)) * 2 + 1
        k = np.hanning(n + 2)[1:-1]; k /= k.sum(); m = n // 2
        G = np.stack([np.convolve(np.r_[F[-m:, i], F[:, i], F[:m, i]], k, 'valid') for i in (0, 1)], -1)
        # only round the chin and cheek; leave a profile nose sharp
        return G if abs(self.yaw) <= .75 else F

    def _xlim(self):
        """for a grid of y: front x range still facing the camera"""
        if getattr(self, '_xs', None) is not None and self._xs[0] == (self.yaw, self.pitch, self.roll): return self._xs
        ys = np.linspace(-.99, .9, 40); lo, hi = [], []
        V = self.R[2]
        for y in ys:
            a = np.interp(y, self.yy, self.A); x = np.linspace(-a * .985, a * .985, 121); e = 4e-3
            z0 = self.z(x, y); nx = -(self.z(x + e, y) - self.z(x - e, y)) / (2 * e); ny = -(self.z(x, y + e) - self.z(x, y - e)) / (2 * e)
            c = (nx * V[0] + ny * V[1] + V[2]) / np.sqrt(nx * nx + ny * ny + 1)
            ok = np.where(c > .03)[0]
            if len(ok) == 0: lo.append(0); hi.append(0)
            else:
                # run containing the most facing point
                j = int(np.argmax(c)); l = j
                while l > 0 and c[l - 1] > .03: l -= 1
                r = j
                while r < len(x) - 1 and c[r + 1] > .03: r += 1
                lo.append(x[l] if l > 0 else -9.); hi.append(x[r] if r < len(x) - 1 else 9.)
        self._xs = ((self.yaw, self.pitch, self.roll), ys, A(lo), A(hi)); return self._xs

    def Pc(self, xy, dz=0., squeeze=True):
        xy = np.array(xy, float); _, ys, lo, hi = self._xlim()
        l = np.interp(xy[..., 1], ys, lo); h = np.interp(xy[..., 1], ys, hi); x = xy[..., 0]
        if squeeze:                      # features are squeezed softly toward the far silhouette instead of running over it
            k = .10
            lm = np.maximum(h * .955 - k, .02); d = np.maximum(x - k, 0); x = np.where(x > k, k + lm * np.tanh((d / lm) ** 1.25) ** .8, x)
            lm = np.maximum(-l * .955 - k, .02); d = np.maximum(-x - k, 0); x = np.where(x < -k, -k - lm * np.tanh((d / lm) ** 1.25) ** .8, x)
        else: x = np.clip(x, l, h)
        return self.P(np.stack([x, xy[..., 1]], -1), dz)

    def vis(self, x, y):
        _, ys, lo, hi = self._xlim()
        return np.interp(y, ys, lo) < x < np.interp(y, ys, hi)

    def rest_y(self, y):
        """rest y of a point that should END at y after the jaw opened (on the face centre line)"""
        g = np.linspace(-1.02, .5, 220); e = np.where(g < -.44, -.44 + (g + .44) * .925, g); e = e - self.jopen * smooth(-.56, -.92, e)
        return np.interp(y, e, g)

    def Pm(self, xy, dz=0.):
        """mouth points given in FINAL front coordinates (jaw already open)"""
        xy = np.array(xy, float); xy[..., 1] = self.rest_y(xy[..., 1])
        dz = dz - 2.2 * xy[..., 0] ** 2 * float(smooth(.55, 1.25, abs(self.yaw)))
        return self.Pc(xy, dz)


def lw(s, close=6.6):
    """outer contour width for a head of s px per unit"""
    return float(np.interp(math.log(max(s, 8)), [math.log(v) for v in (16, 45, 110, 330, 700, 1200)], [1.15, 1.9, 3.5, close, 9.0, 11.]))


# ------------------------------------------------------------------ expressions
T5 = A([0, .22, .50, .80, 1.])
_E0 = dict(
    U=[-.020, .040, .057, .058, .022], Lo=[-.056, -.086, -.094, -.084, -.030],      # lids: heights inner->outer
    xi=.166, xo=.506, iris=.066, irish=1.20, pupil=.023, gx=0., gy=.0, lidw=1.0, crease=1.0,
    B=[(.132, .142), (.262, .180), (.400, .202), (.532, .172)], bw=1.0, furrow=0.,
    mx=0., my=-.655, hw=.155, mU=[0, .002, .005, .002, 0], mL=[0, .002, .005, .002, 0], tU=0., tL=0., tongue=0., jaw=0., lip=1.0, wob=0.,
    naso=0., blush=0., sweat=0., tears=0., well=0., vein=0., white=0., dark=0., noseline=0., under=0., closed=0., glint=.85, wrink=0., flare=0.,
    asym=0., hatch=0., gum=0., cheek=0., es=1.0, jx=0., mrot=0., black=0., burn=0., pouch=0., squeeze=0., slim=.46,          # slim: he is fifteen
    UL=None, UR=None, LoL=None, LoR=None, BL=None, BR=None)


def _e(**k):
    d = dict(_E0); d.update(k); return d


EXPR = dict(
    neutral=_e(),
    smile=_e(U=[-.030, .030, .046, .046, .008], Lo=[-.056, -.070, -.074, -.068, -.028], B=[(.134, .160), (.262, .206), (.400, .230), (.534, .194)],
             hw=.205, my=-.640, mU=[.044, .004, -.006, .004, .050], mL=[.044, -.034, -.054, -.040, .050], tU=.030, blush=.25, asym=.4, cheek=.5),
    grin=_e(U=[-.030, .030, .046, .044, .006], Lo=[-.058, -.056, -.054, -.056, -.034], B=[(.132, .156), (.264, .204), (.404, .228), (.538, .188)],
            hw=.270, my=-.600, mU=[.056, .026, .012, .026, .062], mL=[.040, -.185, -.232, -.190, .046], tU=.080, tongue=1., jaw=.085, blush=.3, cheek=.9),
    laugh=_e(U=[-.072, -.014, .020, .012, -.044], Lo=[-.074, -.020, .012, .002, -.048], B=[(.136, .176), (.268, .236), (.404, .252), (.540, .192)], crease=0.,
             hw=.312, my=-.580, mU=[.072, .040, .022, .044, .084], mL=[.020, -.290, -.350, -.298, .026], tU=.092, tongue=1.1, jaw=.165, blush=.3, closed=1., cheek=1.2, pouch=.4, lip=0.),
    laughshut=_e(U=[-.072, -.014, .020, .012, -.044], Lo=[-.074, -.020, .012, .002, -.048], B=[(.136, .180), (.268, .240), (.404, .256), (.540, .196)], crease=0.,
                 hw=.270, my=-.610, mU=[.060, .026, .012, .026, .064], mL=[.044, -.170, -.216, -.175, .048], tU=.080, tongue=1., jaw=.08, blush=.3, closed=1., cheek=1., pouch=.4),
    surprised=_e(U=[-.018, .062, .086, .080, .022], Lo=[-.062, -.102, -.114, -.102, -.044], iris=.054, pupil=.018, B=[(.130, .222), (.262, .276), (.400, .290), (.540, .236)],
                 hw=.085, my=-.680, mU=[-.02, .035, .045, .035, -.02], mL=[-.02, -.055, -.07, -.055, -.02], tU=.016, jaw=.05),
    fear=_e(U=[-.016, .062, .084, .076, .016], Lo=[-.062, -.104, -.114, -.102, -.044], iris=.048, pupil=.012, B=[(.122, .246), (.258, .246), (.400, .234), (.540, .168)],
            hw=.130, my=-.672, mU=[-.024, .006, .014, .008, -.022], mL=[-.024, -.032, -.038, -.028, -.022], tU=.034, sweat=1., furrow=.6, under=.6, gx=.02, glint=.6),
    panic=_e(U=[.010, .120, .140, .125, .030], Lo=[-.040, -.130, -.145, -.130, -.030], white=1., B=[(.130, .280), (.262, .300), (.405, .300), (.575, .240)], es=1.1,
             hw=.250, my=-.645, mU=[.050, .100, .085, .100, .050], mL=[-.07, -.150, -.170, -.150, -.07], tU=.07, tL=.05, jaw=.12, wob=1., sweat=1.),
    grief=_e(U=[-.006, .046, .052, .030, -.028], Lo=[-.052, -.076, -.080, -.070, -.042], iris=.060, pupil=.021, gy=-.004, B=[(.116, .246), (.252, .222), (.396, .190), (.540, .120)], furrow=.9,
             hw=.145, my=-.664, mU=[-.036, .004, .018, .008, -.030], mL=[-.036, -.042, -.050, -.044, -.030], tU=.028, well=1., tears=.6, under=.5, naso=.25, glint=.9),
    scream=_e(),
    rage=_e(U=[-.052, .000, .036, .048, .016], Lo=[-.066, -.090, -.098, -.088, -.030], iris=.050, irish=1.15, pupil=.012, gy=.004, B=[(.104, .030), (.252, .096), (.404, .160), (.552, .178)], bw=1.1, furrow=.9,
            hw=.160, my=-.655, mU=[-.002, .001, .002, .001, -.002], mL=[-.002, .001, .002, .001, -.002], dark=1., burn=1., glint=0., under=.5, wrink=.15, hatch=.0),
    snarl=_e(U=[-.074, -.010, .034, .050, .016], Lo=[-.068, -.074, -.076, -.066, -.018], iris=.046, pupil=.013, B=[(.096, .030), (.246, .106), (.400, .176), (.556, .202)], bw=1.15, furrow=1., glint=.5,
             hw=.225, my=-.650, mU=[.000, .050, .046, .056, .014], mL=[-.030, -.074, -.082, -.070, -.018], tU=.055, tL=.045, jaw=.03, dark=.35, wrink=1., naso=.9, flare=1., hatch=.7, pouch=.5, black=.4),
    tired=_e(U=[-.046, .004, .018, .010, -.018], Lo=[-.058, -.080, -.086, -.076, -.032], iris=.062, gy=-.016, B=[(.134, .176), (.262, .192), (.400, .192), (.532, .136)],
             hw=.165, my=-.650, mU=[.026, -.002, -.008, -.002, .030], mL=[.026, -.002, -.010, -.004, .030], under=.8, glint=.8),
    gentle=_e(U=[-.042, .016, .032, .028, -.004], Lo=[-.056, -.074, -.080, -.072, -.030], iris=.065, gy=-.010, B=[(.138, .180), (.262, .204), (.396, .206), (.524, .154)], bw=.92, glint=.9,
              hw=.158, my=-.648, mU=[.010, -.004, -.006, .002, .030], mL=[.010, -.006, -.010, -.002, .030], cheek=.25, slim=1., blush=.12),
    shut=_e(U=[-.046, -.066, -.074, -.066, -.036], Lo=[-.050, -.070, -.078, -.070, -.040], closed=1., B=[(.132, .140), (.262, .172), (.400, .188), (.532, .150)]),
)
# ---- the scream of e5, one way: a = the crack, b = growing, c = full (c and d are the boil of the same drawing).  both eyes squeezed, inner brows up
def _scr(k, jig=0.):
    """k 0..1 how far the scream has come"""
    m = lambda a, b: a + (b - a) * k
    u = [-.034, -.030 + .003 * jig, m(-.034, -.042), m(-.052, -.062), m(-.076, -.086)]
    return _e(U=u, Lo=[v - .012 for v in u], iris=.040, pupil=.011, glint=0., lidw=1.1, crease=0., squeeze=1.,
              B=[(.084, m(.186, .206) + .003 * jig), (.205, m(.176, .184)), (.365, m(.132, .122)), (.525, m(.066, .040))], bw=1.08, furrow=1.,
              mx=.003 * jig, my=m(-.610, -.600), hw=m(.175, .284) + .003 * jig, mrot=.010 * jig, jaw=m(.09, .285) + .005 * jig, jx=.006 * jig,
              mU=[m(-.034, -.066), m(.012, .020), m(.026, .042), m(.012, .022), m(-.030, -.060)], mL=[m(-.085, -.265), m(-.135, -.425), m(-.150, -.462), m(-.135, -.420), m(-.082, -.256)],
              tU=m(.044, .078), tL=m(.026, .062), gum=m(.1, .7), tongue=.9, lip=0., naso=m(.3, .42), well=.9, tears=1., vein=m(.4, 1.), wrink=m(.4, .7), flare=m(.4, 1.), hatch=m(.2, .35), pouch=0., black=m(.40, .58), cheek=.4)


EXPR['scream_a'] = _scr(.15); EXPR['scream_b'] = _scr(.65); EXPR['scream_c'] = _scr(1.); EXPR['scream_d'] = _scr(1., 1.)
EXPR['scream'] = EXPR['scream_c']


def blend(a, b, t):
    """blend two expressions (names or dicts)"""
    a, b = getx(a), getx(b); o = {}
    for k in _E0:
        va, vb = a[k], b[k]
        if va is None and vb is None: o[k] = None; continue
        if va is None: va = a[k[:-1]]
        if vb is None: vb = b[k[:-1]]
        o[k] = (A(va, float) * (1 - t) + A(vb, float) * t) if np.ndim(va) else va * (1 - t) + vb * t
    return o


def getx(e, **over):
    if isinstance(e, str): d = dict(EXPR[e])
    elif isinstance(e, (tuple, list)) and len(e) == 3: d = dict(blend(*e))
    else: d = _e(**e) if not set(_E0) <= set(e) else dict(e)
    d.update(over); return d


def lids(x, t):
    """blink: move only the lids (and a little of the brows) toward closed"""
    X = getx(x); S = EXPR['shut']; o = dict(X); t = float(t)
    for k, a in (('U', 1.), ('Lo', 1.), ('B', .25)): o[k] = A(X[k], float) * (1 - a * t) + A(S[k], float) * a * t
    return o


# ------------------------------------------------------------------ light
# side +1 = key from the viewer's right; amt: -1 no side shadow, 0 side plane, 1 half the face
def _l(**k): return k


# p0/p1 of a para pass are in head units from the head centre (x right, y DOWN)
LIGHT = dict(
    day=_l(skin=('#f6cfb0', '#dc9c80', '#b36f5e'), blush='#e8836e', white=('#fdf8f2', '#c9c0cc'), iris=('#6b4327', '#b98148', '#e3b777'), pupil='#20130f',
           hair=('#e59a9a', '#b5656f', '#f7c7c0'), hline='#4a1d2b', brow='#4d272e', under=('#4a3130', '#2c1d21'), hood=('#c22a35', '#861827', '#e4585a'),
           cloth=('#20213a', '#424a80', '#0f0e1a'), gold=('#e2b04a', '#8a5a22', '#fff0b8'), line=INK, mouth=('#6c1f28', '#c9696c', '#3a0d15'), teeth=('#fdf6ee', '#d6c5c0'),
           side=1, amt=-.45, style='day', rim=None, nose=.5, socket=0., para=[dict(p0=(0, -1.6), p1=(0, 1.9), stops=[(0, '#fff6ea'), (.45, '#ffffff'), (1, '#cfc6e0')], amt=.6)]),
    sunset=_l(skin=('#ffd0a0', '#cf7f66', '#93504f'), blush='#f0785a', white=('#fff2de', '#d8aea4'), iris=('#6a3a1e', '#c98542', '#f6c877'), pupil='#1f100c',
              hair=('#f2a391', '#b35c68', '#ffd8b4'), hline='#4b1a26', brow='#4a2127', under=('#533230', '#2e1a20'), hood=('#d23a2e', '#861f30', '#ff7c50'),
              cloth=('#2a2140', '#6b4b7c', '#150e1e'), gold=('#f2b84a', '#9a5a22', '#fff3bc'), line='#22101a', mouth=('#702028', '#d06a62', '#3c0c14'), teeth=('#fff4e2', '#dcbcae'),
              side=1, amt=.05, style='day', nose=.5, socket=0., ldir=(.85, .42, .30), glint='#fff6e0', rim=[dict(d=(-10, 4), col='#ffe6a6', amt=1.)],
              para=[dict(p0=(1.4, -1.2), p1=(-1.3, 1.5), stops=[(0, '#fff2d6'), (.4, '#ffffff'), (1, '#b49ac6')], amt=.8)]),
    under=_l(skin=('#ad9a98', '#6c6078', '#433a56'), blush='#8f5a66', white=('#d9d8de', '#8d8aa2'), iris=('#3d2a26', '#8a6646', '#b89468'), pupil='#120c10',
             hair=('#a9757f', '#6b475c', '#c69da2'), hline='#251325', brow='#2d1a26', under=('#33262f', '#1b141d'), hood=('#82232f', '#481527', '#a83c46'),
             cloth=('#181a2e', '#333b61', '#0a0a14'), gold=('#a58446', '#56422a', '#e6dcb0'), line='#130f1a', mouth=('#42161f', '#8f4f5a', '#21090f'), teeth=('#d8d4d6', '#9d97a6'),
             side=-1, amt=.28, style='day', nose=1., socket=.55, ldir=(-.45, .80, .25), glint='#e8f2ff', rim=[dict(d=(7, 6), col='#9fc2c8', amt=.55)],
             para=[dict(p0=(0, -1.7), p1=(0, 1.6), stops=[(0, '#f2fff6'), (.45, '#d9dcec'), (1, '#6d6a92')], amt=1.)]),
    rain=_l(skin=('#bcaeb4', '#777292', '#4a4769'), blush='#a8626e', white=('#e6e8f0', '#9593b2'), iris=('#3f2b2a', '#8d6a4c', '#c09c72'), pupil='#110c12',
            hair=('#ac7283', '#694560', '#dcb0bc'), hline='#26132a', brow='#2c1826', under=('#352833', '#1a141f'), hood=('#8c2336', '#4b142b', '#c05862'),
            cloth=('#191b31', '#42507f', '#0a0a16'), gold=('#a88a52', '#54432f', '#f2ecd0'), line='#120f1c', mouth=('#4a1626', '#9c5464', '#230812'), teeth=('#e2e0ea', '#a7a3b8'),
            side=-1, amt=.10, style='day', nose=.9, socket=.8, ldir=(-.40, .85, .15), glint='#f2f8ff', rim=[dict(d=(8, 7), col='#c4d8ff', amt=.85)],
            para=[dict(p0=(-.6, -1.6), p1=(.5, 1.8), stops=[(0, '#f4f8ff'), (.4, '#e4e6f6'), (1, '#7f80ae')], amt=1.)]),
    fire=_l(skin=('#ffb469', '#9a4438', '#5c252c'), blush='#ff7a3c', white=('#fff0cf', '#c98a72'), iris=('#5a2416', '#c8742e', '#ffd070'), pupil='#180a08',
            hair=('#e58a6c', '#8c3b41', '#ffd08c'), hline='#34101a', brow='#33131a', under=('#3c1f20', '#200f14'), hood=('#c8321f', '#701520', '#ff7a35'),
            cloth=('#26182a', '#8e4738', '#12090f'), gold=('#f2a53a', '#8a4420', '#fff0b0'), line='#1e0b10', mouth=('#5c131a', '#d0603c', '#2c0608'), teeth=('#ffe9c4', '#c88c6c'),
            side=1, amt=-.2, style='below', nose=.0, socket=0., ldir=(.35, -.75, .35), hcut=-.35, glint='#ffe9b0', rim=[dict(d=(0, -9), col='#ffd27a', amt=.95)],
            para=[dict(p0=(0, 1.7), p1=(0, -1.6), stops=[(0, '#ffe6b8'), (.4, '#f2b894'), (1, '#6e3448')], amt=.85)]),
    snow=_l(skin=('#e9d8d8', '#a79fc0', '#716c98'), blush='#d98a92', white=('#f8f8fb', '#b4b4cf'), iris=('#4e3a36', '#a27d58', '#d4b48c'), pupil='#19141a',
            hair=('#d49ea9', '#8d698a', '#f4d5d6'), hline='#3a2038', brow='#402734', under=('#46363f', '#2a212c'), hood=('#a93540', '#602242', '#d4757a'),
            cloth=('#24283f', '#56648f', '#121422'), gold=('#c9a860', '#6f5a44', '#fff6d8'), line='#1a1728', mouth=('#57202c', '#b06a78', '#2c0e18'), teeth=('#f2f0f4', '#bcb8cc'),
            side=1, amt=-.10, style='day', nose=.6, socket=.15, ldir=(.60, .62, .35), glint='#ffffff', rim=[dict(d=(-9, 4), col='#fff0e4', amt=.95)],
            para=[dict(p0=(1.2, -1.4), p1=(-1., 1.7), stops=[(0, '#fff4ee'), (.45, '#f2f2fb'), (1, '#8e98c6')], amt=.9)]),
    cursed=_l(skin=('#a6a2bd', '#54507f', '#2d2a55'), blush='#7a5a8e', white=('#e8f2ff', '#8d98cc'), iris=('#3a2a34', '#7d6a6a', '#c0b49c'), pupil='#0c0a16',
              hair=('#a0739a', '#553e6f', '#cdb3e0'), hline='#1e1233', brow='#251a35', under=('#2b2338', '#151126'), hood=('#80264a', '#431a42', '#b0568a'),
              cloth=('#15172f', '#36559e', '#080916'), gold=('#93a0b0', '#465070', '#eaf6ff'), line='#0d0c1e', mouth=('#3c1430', '#7a4a7a', '#1c0818'), teeth=('#dfe8f6', '#98a0cc'),
              side=1, amt=.30, style='day', nose=1., socket=.7, ldir=(.75, .15, .3), glint='#dff6ff', rim=[dict(d=(-11, 1), col='#8fe6ff', amt=1., over=True), dict(d=(7, 2), col='#3f7dff', amt=.85)],
              para=[dict(p0=(1.3, 0), p1=(-1.2, .3), stops=[(0, '#e6f6ff'), (.45, '#d9d6f2'), (1, '#6a5fa8')], amt=.9)]),
    ash=_l(skin=('#cbbcb2', '#776c72', '#403a45'), blush='#a3786e', white=('#e4e0da', '#8c8688'), iris=('#3c261c', '#96683c', '#d2a264'), pupil='#130d0c',
           hair=('#b98484', '#6e4a55', '#dfbab3'), hline='#2a171f', brow='#2c1a1e', under=('#352a2c', '#1c1619'), hood=('#85282f', '#491723', '#ad4b4a'),
           cloth=('#1f1f2a', '#565c76', '#0c0b10'), gold=('#a98a50', '#574429', '#f2e2b0'), line='#15111a', mouth=('#4a1a1e', '#95585a', '#240a0c'), teeth=('#e2dad2', '#a39a98'),
           side=-1, amt=.42, style='day', nose=1., socket=.7, ldir=(-.60, .62, .25), glint='#ffe8c8', rim=[dict(d=(-9, -3), col='#ff9a55', amt=.9), dict(d=(7, 4), col='#eef3f8', amt=.6)],
           para=[dict(p0=(-1., -1.5), p1=(.8, 1.6), stops=[(0, '#ffffff'), (.5, '#e2dcdc'), (1, '#8d7f88')], amt=.9)]),
)


def light(l, side=None, **over):
    """a light mode (name or dict); side=-1/+1 mirrors the whole set-up (key side, hair light, rims, gradients)"""
    L = dict(LIGHT[l] if isinstance(l, str) else l)
    if side is not None and side != L['side']:
        L['side'] = side
        if 'ldir' in L: L['ldir'] = (-L['ldir'][0],) + tuple(L['ldir'][1:])
        L['rim'] = [dict(r, d=(-r['d'][0], r['d'][1])) for r in (L['rim'] or [])]
        L['para'] = [dict(q, p0=(-q['p0'][0], q['p0'][1]), p1=(-q['p1'][0], q['p1'][1])) for q in L.get('para', [])]
    L.update(over); return L


# ------------------------------------------------------------------ small drawing helpers
def _crv(xs, ys, n=48, straight=0.):
    """smooth curve through (xs, ys) (cubic, parametrised by index)"""
    m = max(4, n // (len(xs) - 1)); S = spl(np.c_[xs, ys], n=m)
    if straight:
        u = np.arange(len(S)) / m; P = np.c_[np.interp(u, np.arange(len(xs)), xs), np.interp(u, np.arange(len(xs)), ys)]
        S = S * (1 - straight) + P * straight
        k = np.hanning(7); k /= k.sum(); q = np.vstack([S[:1]] * 3 + [S] + [S[-1:]] * 3); q = np.stack([np.convolve(q[:, i], k, 'valid') for i in (0, 1)], -1); q[0], q[-1] = S[0], S[-1]; S = q
    return S


def _prof(t, pts):
    p = A(pts); return np.interp(t, p[:, 0], p[:, 1])


def _c(c): return rgb(c) if isinstance(c, str) else np.asarray(c, f32)


def mixc(a, b, t): return _c(a) * (1 - t) + _c(b) * t


def rad(cel, clip, c, r, col, a=1., ry=None, rot=0.):
    """airbrush dab (radial gradient) clipped to a polygon"""
    x = cel.ctx; x.save()
    if clip is not None: cel._path(clip); x.clip()
    x.translate(*c); x.rotate(rot); x.scale(1, (ry or r) / r)
    g = cairo.RadialGradient(0, 0, 0, 0, 0, r); q = _c(col)
    for t, v in ((0, 1.), (.35, .78), (.7, .28), (1, 0)): g.add_color_stop_rgba(t, float(q[0]), float(q[1]), float(q[2]), a * v)
    x.set_source(g); x.arc(0, 0, r, 0, 2 * math.pi); x.fill(); x.restore()


def lgrad(cel, clip, p0, p1, stops):
    """airbrush linear gradient clipped to a polygon: stops [(t,col,alpha)]"""
    x = cel.ctx; x.save()
    if clip is not None: cel._path(clip); x.clip()
    g = cairo.LinearGradient(*[float(v) for v in (*p0, *p1)])
    for t, col, a in stops:
        q = _c(col); g.add_color_stop_rgba(t, float(q[0]), float(q[1]), float(q[2]), a)
    x.set_source(g); x.paint(); x.restore()


def L_amt(L): return L['amt'] if L['style'] == 'day' else -1.


# ------------------------------------------------------------------ face
class _Face:
    """one drawing of the face: holds cel, head, expression, light"""

    def __init__(s, cel, H, X, L, seed, damage=0., lines=1.0, wet=0.):
        s.c, s.H, s.X, s.L, s.seed, s.dmg, s.wet = cel, H, X, L, int(seed), damage, wet
        s.X = X = dict(X, my=X['my'] + .034)                               # the mouth follows the shorter nose up
        s.u = H.s; s.w = lw(H.s) * lines; s.ink = L['line']; s.F = H.face()
        s.dark = float(X['dark'])

    # -- colours
    def skin(s, i): return s.L['skin'][i]

    def ln(s, P, w, col=None, taper=(.3, .4), seed=0, **k):
        P = np.asarray(P, float)
        if 2 < len(P) <= 7: P = spl(P, n=8)
        s.c.line(P, max(.7, w * s.w * .80), s.ink if col is None else col, taper=taper, seed=s.seed * 7 + seed, press=k.pop('press', .30), tremor=k.pop('tremor', .18) * min(1, s.u / 200), **k)

    # -- side plane shadow line in front coordinates
    def side_line(s, sg, amt):
        y = A([.80, .60, .36, .18, .06, -.04, -.13, -.24, -.40, -.56, -.64, -.74, -.86, -1.0])
        x0 = A([.62, .62, .645, .635, .585, .575, .600, .555, .470, .425, .385, .345, .250, .100])
        x1 = A([.06, .08, .10, .12, .05, .030, .02, .02, .02, .00, .02, .00, .02, .00])
        xm = A([.80, .80, .78, .74, .72, .72, .73, .70, .640, .575, .520, .440, .330, .150])
        t = float(np.clip(amt, -1, 1.15))
        x = x0 * (1 - t) + x1 * t if t >= 0 else x0 * (1 + t) + xm * (-t)
        return np.c_[sg * x, y]

    def side_poly(s, sg, amt):
        Q = s.H.Pc(s.side_line(sg, amt), squeeze=False); d = np.array([sg * 3 * s.u, 0])
        return np.vstack([Q, (Q + d)[::-1]])

    def base(s):
        c, H, L, X = s.c, s.H, s.L, s.X
        if H.pitch > .05: c.fill(H.under(), s.skin(1))
        c.fill(s.F, s.skin(0))
        s.shadows()

    def shadows(s):
        c, H, L, X = s.c, s.H, s.L, s.X
        sg = -L['side']; amt = L['amt']; st = L['style']; sh = s.skin(1)
        c.clip(s.F)
        # airbrush: cooler toward the jaw, warm cheeks
        cen = H.P([(0, -.35)])[0]
        lgrad(c, s.F, H.P([(0, -.45)])[0], H.P([(0, -1.02)])[0], [(0, sh, 0.), (1, sh, .34)])
        # cheeks / nose blush
        b = X['blush'] * .55 + .10
        for q in (-1, 1):
            if H.vis(.36 * q, -.22): rad(c, s.F, H.Pc([(.36 * q, -.25)])[0], .25 * s.u, L['blush'], .42 * b, ry=.16 * s.u, rot=H.roll)
        rad(c, s.F, H.P3([(0, -.38, float(H.z(0, -.38)) + .12)])[0], .12 * s.u, L['blush'], .30 * b + .05)
        if st == 'below': s.shadow_below(sh)
        else:
            c.fill(s.side_poly(sg, amt), sh)
            s.hair_cast()
            for q in (-1, 1): s.socket(q, q == sg)
            s.nose_shadow(sg)
            s.lip_shadow(sg)
        if s.dark > .02: s.shadow_dark()
        if s.X['naso'] > .5 and st != 'below': s.cheek_fold(sg)
        c.unclip()

    def shadow_below(s, sh):
        """lit from under: forehead, cheek fronts, nose ridge, upper lip and chin front go dark"""
        H = s.H; f = lambda P: s.c.fill(H.Pc(A(P, float)), sh)
        f([(-.82, .9), (-.82, .04), (-.64, .14), (-.57, .245), (-.40, .262), (-.16, .20), (-.075, .13), (0, .17), (.075, .13), (.16, .20), (.40, .262), (.57, .245), (.64, .14), (.82, .04), (.82, .9)])
        for q in (-1, 1): f([(q * .74, .05), (q * .60, -.06), (q * .40, -.135), (q * .22, -.125), (q * .115, -.16), (q * .10, -.30), (q * .13, -.39), (q * .24, -.33), (q * .40, -.30), (q * .52, -.36), (q * .60, -.30), (q * .74, -.22)])
        s.c.fill(s.N([(-.030, .10, 1), (.030, .10, 1), (.034, -.20, .95), (.045, -.36, .9), (0, -.405, 1), (-.045, -.36, .9), (-.034, -.20, .95)]), sh)
        f([(-.13, -.505), (-.04, -.50), (0, -.53), (.04, -.50), (.13, -.505), (.10, -.585), (.04, -.625), (0, -.605), (-.04, -.625), (-.10, -.585)])
        yb = s.X['my'] + min(s.X['mL'])
        P = [(-.14, yb - .10), (0, yb - .075), (.14, yb - .10), (.10, yb - .17), (0, yb - .20), (-.10, yb - .17)]; s.c.fill(H.Pm(P), sh)

    def shadow_dark(s):
        """cold rage: the brows throw their shadow into the sockets.  Irregular lower edge broken by hatching, black only in the depth of the socket, skin between the eyes and on the forehead"""
        H = s.H; d = min(1., s.dark * 1.15); r = np.random.default_rng(s.seed + 66); X = s.X
        c0, c1, c2 = s.skin(1), mixc(s.skin(1), mixc(s.skin(2), '#0d0a11', .10), .25 + .75 * s.dark), mixc(s.skin(2), '#0b0910', -.05 + .62 * s.dark); sg = -s.L['side']
        for q in (-1, 1):
            if not H.vis(q * .25, .02): continue
            sh = q == sg; sfx = 'R' if q > 0 else 'L'; B = A(X.get('B' + sfx) if X.get('B' + sfx) is not None else X['B'], float); by = lambda x: np.interp(x, B[:, 0], B[:, 1]); e = 1. if sh else .0
            Q = [(.082, by(.10) + .004), (.16, by(.16) + .008), (.30, by(.30) + .010), (.44, by(.44) + .006), (.540, by(.54) - .004), (.585 + .03 * e, .060), (.560 + .03 * e, .010), (.590 + .05 * e, -.035), (.520, -.060), (.530 + .03 * e, -.096), (.460, -.090), (.440, -.122), (.375, -.104), (.325, -.124), (.270, -.100), (.215, -.112), (.182, -.074), (.146, -.062), (.130, -.008), (.106, .030)]
            s.c.fill(H.Pc(spl(A([(q * a, b) for a, b in Q]), True, corners=(0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19), n=4)), c1, d)
            core = [(.104, by(.12) - .010), (.22, by(.22) - .004), (.38, by(.38) - .006), (.515, by(.51) - .014), (.530, .010), (.490, -.050), (.400, -.092), (.290, -.094), (.195, -.066), (.146, -.012)]
            s.c.fill(H.Pc(spl(A([(q * a, b) for a, b in core]), True, corners=(0, 3), n=6)), c2, d)
            if s.u > 60:                                            # the pen takes the edge apart
                lw_ = .40 if s.u > 150 else .30
                for x0, y0, x1, y1, n, l, dx, dy in ((.475, -.105, .61, -.010, 9, .125, .62, -.78), (.26, -.112, .42, -.118, 5, .060, .52, -.85), (.100, .020, .150, -.080, 6, .075, -.15, -.98)):
                    for j in range(n):
                        if r.random() < .18: continue
                        u = (j + r.uniform(-.3, .3)) / max(1, n - 1); x = x0 + (x1 - x0) * u; y = (by(x) + .026) if y0 is None else y0 + (y1 - y0) * u; q_ = l * r.uniform(.45, 1.15) * (1. if sh else .75)
                        if H.vis(q * x * 1.04, y): s.ln(H.Pc([(q * x, y), (q * (x + dx * q_ * .5), y + dy * q_ * .5), (q * (x + dx * q_) + r.normal(0, .003), y + dy * q_)]), lw_, col=c1, seed=600 + j, taper=(.02, .85), alpha=d)

    def cheek_fold(s, sg):
        """a lifted cheek throws a small hard shadow along the smile fold"""
        H, X = s.H, s.X; hw = X['hw']; y = X['my'] + X['mU'][0]
        for q in (-1, 1):
            if q != sg and abs(H.yaw) < .9: continue
            P = [(q * .118, -.436), (q * (.13 + hw * .30), -.498), (q * (hw + .040), y + .038), (q * (hw + .056), y - .015), (q * (hw + .050), y - .085), (q * (hw + .078), y - .010), (q * (hw + .076), y + .040), (q * (hw + .040), y + .066), (q * (.16 + hw * .30), -.480)]
            s.c.fill(H.Pm(spl(P, True, corners=(0, 4), n=8)), s.skin(1), .95)

    def hair_cast(s):
        """the hair's own shadow on the forehead: the fringe roots and tips pushed down along the light"""
        H = s.H; hr = getattr(s, 'hair', None)
        if hr is None: return
        sd = -s.L['side']; k = s.L.get('cast', .085)
        ax = A([math.cos(H.roll), math.sin(H.roll)]); dn = A([-math.sin(H.roll), math.cos(H.roll)])
        off = (ax * sd * .045 + dn * k) * H.s
        for q in hr.cl:
            if q.kind in ('front', 'side', 'tooth') and q.face > -.2 and (q.big or q.kind != 'side'):
                g = q.C[:8].mean(0); s.c.fill(g + (q.poly - g) * 1.04 + off, s.skin(1))
        for p in hr.cap: s.c.fill(p + off * .8, s.skin(1))

    def socket(s, q, shade):
        k = s.L.get('socket', .0)
        if k <= 0: return
        if s.H.vis(.2 * q, .05):
            B = A(s.X['B'], float); U = A(s.X['U'], float); by = lambda x: np.interp(x, B[:, 0], B[:, 1]) - .030
            P = [(.082, min(.13, by(.12))), (.125, by(.125)), (.20, by(.20)), (.30, by(.30) - .004), (.40, by(.40) - .012), (.46, by(.46) - .03), (.40, (by(.40) + U[3]) / 2 + .012), (.30, (by(.30) + U[2]) / 2), (.215, U[1] + .020), (.172, U[0] + .016), (.150, -.060), (.112, -.105), (.100, -.02)]
            s.c.fill(s.H.Pc(spl([(q * a, b) for a, b in P], True, corners=(0, 1, 5, 9, 11), n=6)), s.skin(1), min(1, k * (1.2 if shade else .8)))

    def _socket_old(s, q, shade):
        """brow-to-lid shadow: inner corner wedge beside the nose bridge"""
        if not s.H.vis(.2 * q, .05): return
        P = [(.085, .115), (.17, .105), (.245, .098), (.205, .060), (.165, .005), (.150, -.060), (.112, -.10), (.105, -.02), (.098, .05)]
        s.c.fill(s.H.Pc([(q * a, b) for a, b in P]), s.skin(1), .9 if shade else .55)

    def nose3(s, y):
        w, yt, ytip, yb, d = s.H.k['nose']
        g = np.interp(y, [yb - .05, yb, ytip - .03, ytip, ytip + .03, (yt + ytip) / 2, yt, .10], [0, .30 * d, .82 * d, d, .93 * d, .55 * d, .18 * d, -.01])
        return g

    def N(s, pts):
        """nose points (x, y, frac of ridge height)"""
        ny = lambda y: -.06 + (y + .06) * .86 if y < -.06 else y          # nose points are authored on the long nose
        P = A([(x, ny(y), float(s.H.z(x, ny(y))) + s.nose3(ny(y)) * k) for x, y, k in pts]); return s.H.P3(P)

    def nose_shadow(s, sg):
        H = s.H
        # triangular cast of the nose on the shadow side + under plane
        k = s.L.get('nose', .6)
        ytop = -.34 + .20 * k
        side = s.N([(sg * .006, ytop, 1), (sg * .012, -.30, 1), (sg * .014, -.395, 1), (sg * -.004, -.432, .92), (sg * .030, -.462, .55), (sg * .078, -.468, .25),
                    (sg * (.050 + .030 * k), -.425, .3), (sg * (.030 + .022 * k), -.36, .5), (sg * (.016 + .018 * k), -.26, .7), (sg * .014, (ytop - .26) / 2, .8)])
        if abs(H.yaw) < 1.25 and sg * H.yaw < .45 and L_amt(s.L) > .22: s.c.fill(side, s.skin(1))
        und = s.N([(-.070, -.456, .35), (-.02, -.442, .95), (.02, -.442, .95), (.070, -.456, .35), (.050, -.476, .25), (sg * .030, -.506, .05), (-.050, -.476, .25)])
        s.c.fill(und, s.skin(1))

    def lip_shadow(s, sg):
        X, H = s.X, s.H; yb = X['my'] + min(X['mL']) - .055
        sg2 = sg * .012; P = [(-.062 + sg2, yb + .010), (0, yb + .016), (.062 + sg2, yb + .010), (.040 + sg2, yb - .014), (.0, yb - .022), (-.040 + sg2, yb - .014)]
        s.c.fill(H.Pm([(X['mx'] + a * X['hw'] / .17, b) for a, b in P]), s.skin(1), .95)

    # -- ears
    def ears_far(s):
        ay = abs(s.H.yaw); s.ears([q for q in (-1, 1) if ay < .12 or q == (1 if s.H.yaw > 0 else -1)])

    def ears_near(s):
        if abs(s.H.yaw) >= .12: s.ears([-1 if s.H.yaw > 0 else 1])

    def ears(s, which=(-1, 1)):
        c, H, L = s.c, s.H, s.L
        for q in which:
            if H.ear(q)['vis'] <= 0: continue
            ex, _, _, z = H.k['ear']; sd = (q == -L['side'])
            def Q(L_, n=8): return spl(H.P3(A([(q * (ex - .012 + u * .62), v, z + .02 - u * 1.35) for u, v in L_])), n=n)
            o = Q([(-.02, .030), (.012, .085), (.060, .122), (.112, .118), (.150, .070), (.162, -.005), (.150, -.085), (.120, -.160), (.098, -.225), (.082, -.290), (.052, -.335), (.018, -.335), (-.012, -.295), (-.02, -.24)])
            cl = np.vstack([o, o[:1]])
            c.fill(cl, s.skin(0)); c.clip(cl)
            if sd: c.fill(Q([(-.05, .16), (.10, .13), (.118, .06), (.105, -.02), (.118, -.10), (.09, -.20), (.05, -.26), (.06, -.36), (-.05, -.36)]), s.skin(1))
            c.fill(Q([(.020, .030), (.055, .078), (.102, .078), (.120, .030), (.112, -.04), (.088, -.030), (.080, .022), (.052, .030), (.044, -.020), (.056, -.095), (.070, -.160), (.045, -.205), (.018, -.180), (.030, -.12), (.012, -.06)]), s.skin(2) if sd else s.skin(1))
            c.unclip()
            s.c.line(o, s.w * .80, s.ink, taper=(.10, .12), press=.25, seed=s.seed + 3 + q, tremor=.15)
            s.ln(Q([(.022, .010), (.034, .055), (.070, .082), (.108, .068), (.122, .015), (.112, -.055), (.094, -.115)]), .38, seed=11 + q, taper=(.3, .5))
            s.ln(Q([(.050, .030), (.066, -.010), (.052, -.070), (.066, -.130), (.056, -.185), (.028, -.196)]), .38, seed=13 + q, taper=(.5, .4))
            s.ln(Q([(.010, -.075), (.028, -.105), (.022, -.150)]), .32, seed=15 + q, taper=(.3, .5))

    # -- eyes
    def eye(s, q):
        c, H, L, X = s.c, s.H, s.L, s.X
        if not H.vis(.27 * q, -.02): return
        w = s.w; far = q * H.yaw > .25; es = X['es']
        sfx = 'R' if q > 0 else 'L'                                  # optional per-eye lids: UL UR LoL LoR (viewer's left / right)
        Uv, Lv = A(X.get('U' + sfx) if X.get('U' + sfx) is not None else X['U'], float), A(X.get('Lo' + sfx) if X.get('Lo' + sfx) is not None else X['Lo'], float)
        xs = .334 + (X['xi'] + (X['xo'] - X['xi']) * T5 - .334) * es; k_ = 0. if X['white'] > .5 else 1.
        up = _crv(xs, -.018 + (Uv + .018) * es, straight=.60 * k_); lo = _crv(xs, -.018 + (Lv + .018) * es, straight=.40 * k_); n = len(up); t = np.linspace(0, 1, n)
        Mx = lambda P: np.c_[q * P[:, 0], P[:, 1]]
        Pu = H.Pc(Mx(up)); Pl = H.Pc(Mx(lo)); poly = np.vstack([Pu, Pl[::-1]])
        dk = s.dark; gap = float(np.max(up[:, 1] - lo[:, 1])); closed = gap < .032; arc = closed and Uv[2] > Uv[0] + .04
        wh, wh2 = L['white']
        if dk: wh, wh2 = mixc(wh, mixc(s.skin(2), '#141019', .62), .80 * dk), mixc(wh2, mixc(s.skin(2), '#0d0a12', .8), .90 * dk)
        if X['white'] > .5:
            c.fill(poly, '#ffffff')
            outline(c, poly, w * 1.05, s.ink, seed=s.seed + 31 + q); s.ln(Pu, 1.9, seed=33 + q, taper=(.12, .12)); return
        if not closed:
            c.fill(poly, wh)
            c.clip([poly])
            ix = X['xi'] + (X['xo'] - X['xi']) * .485 + q * X['gx']; iy = .004 + X['gy']; rx = X['iris'] * es; ry = rx * X['irish']
            a = np.linspace(0, 2 * np.pi, 41)[:-1]
            E = lambda r1, r2, dx=0, dy=0: H.Pc(np.c_[q * ix + dx + np.cos(a) * r1, iy + dy + np.sin(a) * r2])
            d0, d1, d2 = L['iris']
            if dk: d0, d1, d2 = mixc(d0, '#0d0a10', .6 * dk), mixc(d1, '#150f16', .55 * dk), mixc(d2, s.skin(2), .35 * dk)
            bn = X['burn']
            if bn > 0: bc = [_c(v) for v in L.get('burn', ('#b84a12', '#ffa53c', '#ffe6a0'))]; d0, d1, d2 = mixc(d0, bc[0], bn), mixc(d1, bc[1], bn), mixc(d2, bc[2], bn)
            c.fill(E(rx, ry), mixc(d0, s.ink, .78 if bn < .5 else .55))
            c.fill(E(rx * .86, ry * .90), d1)
            c.clip([E(rx * .86, ry * .90)])
            # two tones: dark flat top, light foot
            c.fill(H.Pc(np.c_[q * ix + A([-1.2, -.9, -.4, .1, .6, 1.0, 1.2, 1.2, -1.2]) * rx, iy + A([.16, .12, .22, .14, .22, .12, .16, 1.4, 1.4]) * ry]), d0)
            c.fill(E(rx * .72, ry * .40, 0, -ry * .62), d2)
            c.unclip()
            if s.u > 260 and bn < .3:
                r = np.random.default_rng(s.seed + 91 + q)
                for j in range(6):
                    th = -np.pi * (.12 + .76 * (j + .5) / 6) + r.normal(0, .05)
                    c.line(H.Pc([(q * ix + np.cos(th) * rx * .50, iy + np.sin(th) * ry * .50), (q * ix + np.cos(th) * rx * .80, iy + np.sin(th) * ry * .80)]), max(.7, .007 * s.u), d0, taper=(.1, .7), seed=j, alpha=.45)
            c.fill(E(X['pupil'] * es, X['pupil'] * es * 1.2, 0, .004), L['pupil'] if bn < .5 else mixc(L['pupil'], '#3a0c04', .4))
            # lid shadow over white + iris: hard, deep, flattens the top of the iris
            th = _prof(t, [(0, .012), (.3, .028), (.6, .034), (.9, .030), (1, .014)]) * (.62 + .38 * min(1.2, gap / .16))
            c.fill(H.Pc(Mx(np.vstack([up, np.c_[up[:, 0], up[:, 1] - th][::-1]]))), wh2, .95)
            c.ctx.save(); c._path(E(rx, ry)); c.ctx.clip(); c.fill(H.Pc(Mx(np.vstack([up, np.c_[up[:, 0], up[:, 1] - th * 1.1][::-1]]))), mixc(d0, s.ink, .70), .96); c.ctx.restore()
            if X['well'] > .02:
                hgt = .026 * X['well'] + .009 * X['well'] * np.sin(np.pi * t * 3) ** 2
                top = np.c_[lo[:, 0], lo[:, 1] + hgt * np.sin(np.pi * np.clip(t, .03, .97)) ** .5]
                c.fill(H.Pc(Mx(np.vstack([lo, top[::-1]]))), mixc(L['white'][0], '#bfe0f5', .55), .42)
                c.line(H.Pc(Mx(top[6:-6])), max(.9, .008 * s.u), '#ffffff', taper=(.4, .4), seed=5, alpha=.9)
            g = X['glint']
            if g > 0:
                ls = L['side']; gc = L.get('glint', '#ffffff'); gx0, gy0 = q * ix + ls * rx * .36, iy + ry * .30; k = rx / .085 * g
                c.fill(H.Pc(spl([(gx0 - .019 * k, gy0 - .004 * k), (gx0 - .004 * k, gy0 + .019 * k), (gx0 + .020 * k, gy0 + .008 * k), (gx0 + .010 * k, gy0 - .016 * k)], True, n=5)), gc)
                if g > .8: c.fill(E(rx * .11, ry * .08, -ls * rx * .40, -ry * .50), gc, .7)
                if X['well'] > .3: c.fill(E(rx * .16, ry * .07, -ls * rx * .30, ry * .24), '#ffffff', .8); c.fill(E(rx * .09, ry * .05, ls * rx * .55, -ry * .45), '#ffffff', .7)
            c.unclip()
            if bn > 0:                                              # burning through the dark: a little light spills over the lids
                cc = H.Pc([(q * ix, iy - .01)])[0]; c.ctx.save(); c.ctx.set_operator(cairo.OPERATOR_SCREEN); rad(c, None, cc, rx * 2.6 * s.u, bc[1], .42 * bn); rad(c, None, cc, rx * 1.1 * s.u, bc[2], .35 * bn); c.ctx.restore()
        if X['squeeze'] > .5 and closed:
            M_ = lambda P: H.Pc(Mx(spl(A(P, float), n=8))); u0 = lambda x: float(np.interp(x, up[:, 0], up[:, 1]))
            ln_ = spl(A([(.146, u0(.17) - .006), (.205, u0(.205) + .004), (.285, u0(.285)), (.375, u0(.375)), (.455, u0(.455)), (.520, u0(.506) - .006)]), n=10); tq = np.linspace(0, 1, len(ln_)); w_ = _prof(tq, [(0, .003), (.16, .016), (.45, .023), (.78, .028), (.93, .020), (1, .002)]) * X['lidw']
            c.fill(H.Pc(Mx(np.vstack([ln_ + np.c_[0 * w_, w_ * .62], (ln_ - np.c_[0 * w_, w_ * .50])[::-1]]))), s.ink)
            c.fill(M_([(.180, u0(.18) + .030), (.270, u0(.27) + .042), (.370, u0(.37) + .040), (.470, u0(.47) + .030), (.400, u0(.40) + .016), (.290, u0(.29) + .018)]), s.skin(2), .75)          # the lid folded over
            c.fill(M_([(.178, u0(.18) - .030), (.270, u0(.27) - .036), (.360, u0(.36) - .038), (.430, u0(.43) - .036), (.370, u0(.37) - .060), (.270, u0(.27) - .058)]), s.skin(1) if q != -s.L['side'] else s.skin(2), .8)
            for j, (P_, k_w) in enumerate([([(.185, u0(.18) + .040), (.270, u0(.27) + .052), (.360, u0(.36) + .050), (.440, u0(.44) + .040)], .46), ([(.190, u0(.19) - .052), (.265, u0(.26) - .062), (.345, u0(.34) - .066), (.415, u0(.41) - .060)], .46),
                                           ([(.516, u0(.5) + .010), (.556, u0(.5) + .026), (.600, u0(.5) + .030)], .46), ([(.524, u0(.5) - .010), (.566, u0(.5) - .014), (.612, u0(.5) - .026)], .50), ([(.510, u0(.5) - .030), (.548, u0(.5) - .050), (.584, u0(.5) - .078)], .42),
                                           ([(.146, u0(.17)), (.120, u0(.17) + .016), (.104, u0(.17) + .044)], .42), ([(.142, u0(.17) - .016), (.116, u0(.17) - .030), (.104, u0(.17) - .052)], .40)]):
                if H.vis(q * P_[-1][0], P_[-1][1]): s.ln(M_(P_), k_w, seed=140 + j + 9 * q, taper=(.25, .65))
            return s.scars(q)
        # upper lid: thin inside, heavy and lifting at the outer corner
        if arc: th = _prof(t, [(0, .004), (.12, .016), (.45, .034), (.75, .036), (.93, .022), (1, .006)]) * X['lidw']
        else: th = _prof(t, [(0, .003), (.12, .012), (.36, .020), (.62, .029), (.82, .042), (.95, .050), (1, .046)]) * X['lidw'] * (1. if not closed else .85)
        B = np.c_[up[:, 0], up[:, 1] + th]; ex, ey = up[-1]
        if arc or far: lid = np.vstack([Mx(B[:-1]), Mx(up)[::-1]])
        elif closed: lid = np.vstack([Mx(B), Mx(A([(ex + .030, ey - .004)])), Mx(up)[::-1]])
        else: lid = np.vstack([Mx(B), Mx(A([(ex + .020, ey + th[-1] + .012), (ex + .044, ey + th[-1] + .020), (ex + .026, ey + .010), (ex + .010, ey - .012)])), Mx(up)[::-1]])
        c.fill(H.Pc(lid), s.ink)
        if s.wet > .4 and s.u > 200 and not closed:
            for j, (tt, k) in enumerate(((.22, .007), (.47, .010), (.63, .007), (.86, .009))):
                i = int(tt * (n - 1)); a = np.linspace(0, 2 * np.pi, 11)[:-1]; e = H.Pc([(q * up[i, 0], up[i, 1] + th[i] * .25)])[0]; kk = k * s.u
                c.fill(np.c_[e[0] + np.cos(a) * kk * .75, e[1] + np.sin(a) * kk], mixc(L['white'][0], '#a9c6ee', .25), .95); c.fill(np.c_[e[0] - kk * .2 + np.cos(a) * kk * .28, e[1] - kk * .3 + np.sin(a) * kk * .3], '#ffffff')
        if not closed:
            # inner corner: a tick, no loop.  lower lid: a broken half line on the outer side only
            ye = max(lo[0, 1] + .004, up[0, 1] - .022); s.ln(H.Pc([(q * (up[0, 0] - .006), ye), (q * (up[0, 0] - .002), (ye + up[0, 1]) / 2), (q * up[2, 0], up[2, 1] + .003)]), .40, seed=43 + q, taper=(.7, .1))
            m = len(lo); cl_ = mixc(s.ink, s.skin(2), .10)
            s.ln(H.Pc(Mx(lo[int(m * .46):int(m * .70)])), .46, seed=45 + q, taper=(.6, .4), col=cl_)
            if not far: s.ln(H.Pc(Mx(lo[int(m * .76):int(m * .97)])), .54, seed=46 + q, taper=(.35, .2), col=cl_)
        if X['crease'] > 0 and not arc:
            off = .030 + .012 * np.clip(gap / .18, 0, 1)
            cr = np.c_[up[:, 0] + .004, up[:, 1] + th + off * _prof(t, [(0, .5), (.25, .8), (.6, 1.), (1, 1.1)])]
            s.ln(H.Pc(Mx(cr[int(.30 * n):int(.86 * n)])), .38, seed=47 + q, taper=(.5, .35), col=mixc(s.ink, s.skin(2), .20))
        if X['under'] > 0:
            P = np.c_[lo[:, 0] - .01, lo[:, 1] - .046 - .012 * np.sin(np.pi * t)]
            s.ln(H.Pc(Mx(P[int(.05 * n):int(.50 * n)])), .34, seed=48 + q, taper=(.3, .6), col=mixc(s.ink, s.skin(2), .35), alpha=float(X['under']))
        if X['pouch'] > 0:                                          # lower lid pushed up into a roll
            P = np.c_[lo[:, 0], lo[:, 1] - .034 - .030 * np.sin(np.pi * np.clip(t * 1.05, 0, 1))]
            s.ln(H.Pc(Mx(P[int(.10 * n):int(.46 * n)])), .46, seed=49 + q, taper=(.5, .6), alpha=float(X['pouch'])); s.ln(H.Pc(Mx(P[int(.54 * n):int(.88 * n)])), .40, seed=50 + q, taper=(.5, .5), alpha=float(X['pouch']))
            if X['pouch'] > .6: s.ln(H.Pc(Mx(np.c_[P[:, 0], P[:, 1] - .030][int(.18 * n):int(.62 * n)])), .34, seed=52 + q, taper=(.5, .6), alpha=float(X['pouch']))
        s.scars(q)

    def scars(s, q):
        """the two-notch closed-eye marks under each eye"""
        H = s.H
        if q * H.yaw > .40 or s.X['white'] > .5: return
        lo = s.X['Lo']; d = .30 * (np.mean(A(lo)[1:4]) + .085)
        P = [(.405, -.180), (.450, -.168), (.498, -.140), (.536, -.102)]
        s.c.line(H.Pc([(q * a, b + d) for a, b in P]), max(.8, .019 * s.u), mixc(s.ink, s.skin(2), .10), taper=(.55, .30), seed=s.seed + 51 + q, press=.1, tremor=.1, tmin=.2)
        P = [(.468, -.196), (.502, -.182), (.530, -.158)]
        s.c.line(H.Pc([(q * a, b + d) for a, b in P]), max(.7, .011 * s.u), mixc(s.ink, s.skin(2), .22), taper=(.5, .5), seed=s.seed + 53 + q, press=.1, tremor=.1)

    def brow(s, q):
        c, H, L, X = s.c, s.H, s.L, s.X
        sfx = 'R' if q > 0 else 'L'; B = A(X.get('B' + sfx) if X.get('B' + sfx) is not None else X['B'], float)
        if not H.vis(B[1, 0] * q, B[1, 1] - .04): return
        mid = _crv(B[:, 0], B[:, 1], 48, straight=.55); n = len(mid); t = np.linspace(0, 1, n)
        T = np.gradient(mid, axis=0); T /= np.hypot(*T.T)[:, None]; N = np.c_[-T[:, 1], T[:, 0]]
        th = _prof(t, [(0, .016), (.05, .025), (.30, .029), (.58, .027), (.80, .017), (1, .002)]) * X['bw']
        up = mid + N * th[:, None]; dn = mid - N * (th * .80)[:, None]
        c.fill(H.Pc(np.c_[q * np.r_[up[:, 0], dn[::-1, 0]], np.r_[up[:, 1], dn[::-1, 1]]]), L['brow'])
        if s.u > 140:                                               # hair ends at the root of the brow
            r = np.random.default_rng(s.seed + 61 + q)
            for j in range(3):
                b = mid[0] + A([r.uniform(.0, .02), (j - 1) * .013]); e = b + A([-.014 - .010 * r.random(), .006 + .012 * r.random()])
                s.ln(H.Pc([(q * b[0], b[1]), (q * (b[0] + e[0]) / 2, (b[1] + e[1]) / 2), (q * e[0], e[1])]), .55, col=L['brow'], seed=63 + j, taper=(.1, .7))

    def furrow(s):
        f = s.X['furrow']; H = s.H
        if f <= .05: return
        B = s.X['B'][0]; y0 = B[1] - .01
        ang = np.clip((s.X['B'][1][1] - y0) * 6, -1, 1)
        for q in (-1, 1):
            if not H.vis(.06 * q, .15): continue
            if ang > .1:   # angry: vertical folds bent over the nose root
                s.ln(H.Pc([(q * .066, y0 + .095), (q * .054, y0 + .052), (q * .060, y0 + .010), (q * .080, y0 - .022)]), .62, seed=71 + q, taper=(.5, .2), alpha=min(1, f * 1.2))
                s.ln(H.Pc([(q * .030, y0 - .030), (q * .045, y0 - .050), (q * .070, y0 - .056)]), .44, seed=73 + q, taper=(.5, .5), alpha=f)
            else:          # grief: brow corners pushed up and together
                s.ln(H.Pc([(q * .070, y0 + .060), (q * .050, y0 + .015), (q * .060, y0 - .040)]), .58, seed=75 + q, taper=(.5, .4), alpha=min(1, f * 1.1))
                s.ln(H.Pc([(q * .165, y0 + .075), (q * .100, y0 + .068), (q * .030, y0 + .085)]), .40, seed=77 + q, taper=(.6, .5), alpha=f)
                if f > .7: s.ln(H.Pc([(q * .250, y0 + .115), (q * .150, y0 + .135), (q * .050, y0 + .118), (q * .015, y0 + .130)]), .34, seed=78 + q, taper=(.6, .5), alpha=f * .8)

    def nose(s):
        H, X = s.H, s.X; yw = H.yaw; d = 1 if yw >= 0 else -1; ay = abs(yw)
        ns = X['flare']; below = H.pitch > .16 or ns > .3
        if ay < .16:
            s.ln(s.N([(-.026, -.452, .85), (-.004, -.461, .80), (.022, -.455, .85)]), .64, seed=81, taper=(.5, .5))
        elif ay < .95:              # one short stroke round the tip, nothing else: the shadow says the rest
            pr = spl(s.H.P3(H.nose_profile()[2:7]), n=8); m = len(pr); s.ln(pr[int(m * .40):int(m * .92)], .86, seed=83, taper=(.65, .25))
        else:
            pr = spl(s.H.P3(H.nose_profile()), n=8); m = len(pr); s.ln(pr[int(m * .04):int(m * .93)], 1.0, seed=83, taper=(.4, .3))
            s.ln(s.N([(-d * .070, -.420, .22), (-d * .098, -.440, .14), (-d * .086, -.464, .20)]), .42, seed=84, taper=(.6, .3))
        if below or .30 < ay < .95:                # a nostril dash where the light does not reach
            for q in ((-1, 1) if below else (-d,)):
                if q == d and ay > .34: continue
                s.c.fill(spl(s.N([(q * .022, -.463, .55), (q * .040, -.457 + .010 * ns, .48), (q * .062, -.461 + .006 * ns, .36), (q * .046, -.471, .38)]), True, corners=(0, 2), n=5), mixc(s.skin(2), s.ink, .78))
        if ns > .3:
            for q in (-1, 1):
                if q == d and ay > .30: continue
                s.ln(s.N([(q * .084, -.395, .18), (q * (.112 + .022 * ns), -.424, .10), (q * (.106 + .022 * ns), -.456, .14), (q * .078, -.468, .30)]), .50, seed=85 + q, taper=(.6, .25))
        if X['wrink'] > .05:
            for q in (-1, 1):
                for j, (y, l) in enumerate(((-.055, .080), (-.105, .088), (-.155, .070))):
                    if j == 2 and X['wrink'] < .8: continue
                    if H.vis(q * .08, y): s.ln(H.Pc([(q * .020, y + .026), (q * .052, y - .004), (q * (.050 + l), y + .016)]), .50 - .06 * j, seed=87 + j + q, taper=(.45, .6), alpha=float(min(1, X['wrink'] * 1.2)))

    def mouth(s):
        c, H, L, X = s.c, s.H, s.L, s.X
        t5 = A([-1, -.5, 0, .5, 1.]); mx, my, hw = X['mx'], X['my'], X['hw']
        Uv, Lv = A(X['mU'], float).copy(), A(X['mL'], float).copy()
        if X['asym']: Uv[4] += .018 * X['asym']; Lv[4] += .018 * X['asym']
        rot = X['mrot']; cr, sr = math.cos(rot), math.sin(rot); piv = my + (Uv[2] + Lv[2]) / 2
        def Rt(P): P = np.array(P, float); x, y = P[..., 0] - mx, P[..., 1] - piv; return np.stack([mx + x * cr - y * sr, piv + x * sr + y * cr], -1)
        def cv(v): return np.c_[mx + hw * t5, my + v]
        U = Rt(spl(cv(Uv), n=12)); Lw = Rt(spl(cv(Lv), n=12)); n = len(U); t = np.linspace(0, 1, n)
        if X['wob'] > 0:
            wv = np.sin(t * np.pi * 7) * .016 * X['wob'] * np.sin(np.pi * t); U[:, 1] += wv; Lw[:, 1] += wv * .8 * np.sign(np.sin(t * 20 + 1))
        op = float(np.max(U[:, 1] - Lw[:, 1])); P = H.Pm
        if op > .012:
            poly = np.vstack([P(U), P(Lw)[::-1]]); m0, m1, m2 = L['mouth']; deep = mixc(m0, m2, .55)
            c.fill(poly, deep); c.clip([poly])
            if X['tongue'] > 0 and op > .06:                        # tongue: one flat shape
                a, b = int(n * .10), int(n * .90) + 1; k = X['tongue']; tt = np.linspace(-1, 1, b - a); base = Lw[a:b]
                h = op * (.36 * k) * (1 - np.abs(tt) ** 2.6) ** .6 * (1 + .10 * np.cos(tt * 4.2 + .5)); hump = np.c_[base[:, 0], base[:, 1] + h + .02 * op]
                c.fill(np.vstack([P(base - A([0, .03])), P(hump)[::-1]]), m1)
            tw0, tw1 = L['teeth']; arch = lambda e, k: np.clip(1 - np.abs((2 * t - 1) / e) ** k, 0, 1) ** .5
            if X['tL'] > .004:
                tb = Lw.copy(); tb[:, 1] += X['tL'] * arch(.80, 4.); i0, i1 = int(n * .09), int(n * .91) + 1
                c.fill(np.vstack([P(Lw[i0:i1]), P(tb[i0:i1])[::-1]]), mixc(tw0, tw1, .55))
                if s.u > 120: s.ln(P(tb[i0:i1]), .30, seed=95, taper=(.3, .3), col=mixc(m0, s.ink, .45))
            if X['tU'] > .004:                                      # teeth: one white band
                tb = U.copy(); tb[:, 1] -= X['tU'] * arch(.94 if op > .06 else .98, 5.); i0, i1 = int(n * .03), int(n * .97) + 1
                if X['gum'] > .1:
                    gm = U.copy(); gm[:, 1] -= X['tU'] * .40 * X['gum'] * arch(.86, 3.); c.fill(np.vstack([P(U[i0:i1]), P(tb[i0:i1])[::-1]]), tw0)
                    c.fill(np.vstack([P(U[i0:i1]), P(gm[i0:i1])[::-1]]), mixc(m1, m0, .45))
                else:
                    c.fill(np.vstack([P(U[i0:i1]), P(tb[i0:i1])[::-1]]), tw0)
                    if X['tU'] > .03: sh = U.copy(); sh[:, 1] -= X['tU'] * .22 * arch(.9, 4.); c.fill(np.vstack([P(U[i0:i1]), P(sh[i0:i1])[::-1]]), tw1)
                if X['tU'] > .03 and s.u > 120: s.ln(P(tb[int(n * .06):int(n * .94)]), .34, seed=91, taper=(.3, .3), col=mixc(m0, s.ink, .45))
            if X['gum'] > .5 and s.u > 150:
                for q, x0, x1 in ((0, -.56, -.50), (1, .64, .52)):  # a thread of spit
                    i, j = int((x0 + 1) / 2 * (n - 1)), int((x1 + 1) / 2 * (n - 1)); a_, b_ = U[i] - A([0, X['tU']]), Lw[j] + A([0, X['tL']])
                    c.line(P(spl([a_, (a_ * .6 + b_ * .4) + A([.010, 0]), b_], n=10)), max(.8, s.w * .18), '#ffffff', taper=(.1, .1), seed=q, alpha=.55)
            c.unclip()
            s.ln(P(U), 1.15, seed=101, taper=(.10, .10), wfun=lambda q: 1.15 - .45 * np.sin(np.pi * q))
            i0 = .04 if op > .10 else .22; s.ln(P(Lw[int(n * i0):int(n * (1 - i0))]), .86 if op > .10 else .52, seed=102, taper=(.40, .40))
            for q, i in ((-1, 0), (1, n - 1)):
                if H.vis(U[i, 0], float(H.rest_y(U[i, 1]))): s.ln(P([(U[i, 0] - q * .010, U[i, 1] + .002), (U[i, 0] + q * .008, U[i, 1] + .006), (U[i, 0] + q * .018, U[i, 1] - .012)]), .62, seed=103 + q, taper=(.2, .6))
        else:
            mid = (U + Lw) / 2; br = int(n * .66)
            # a closed JJK mouth: a long stroke, a lift of the pen, a short one
            s.ln(P(mid[:br]), .98, seed=101, taper=(.20, .30), wfun=lambda q: .80 + .40 * np.abs(q - .5) * 2); s.ln(P(mid[br + 3:]), .86, seed=102, taper=(.35, .20))
            for q, i in ((-1, 0), (1, n - 1)):
                if H.vis(mid[i, 0], mid[i, 1]): s.ln(P([(mid[i, 0] - q * .030, mid[i, 1] - .002), (mid[i, 0], mid[i, 1]), (mid[i, 0] + q * .010, mid[i, 1] + (mid[i, 1] - mid[n // 2, 1]) * .3)]), .80, seed=103 + q, taper=(.5, .4))
        yb = piv + (min(Lv) + my - piv) - .046
        if X['black'] > .3 and X['squeeze'] < .3: kb = float(X['black']); c.fill(P(Rt(spl([(mx - hw * .46, yb + .030), (mx, yb + .012), (mx + hw * .46, yb + .030), (mx + hw * .26, yb - .020), (mx, yb - .034), (mx - hw * .26, yb - .020)], True, corners=(0, 2), n=6))), mixc(s.skin(2), s.ink, .35 + .5 * kb), .85)
        if X['lip'] > 0: s.ln(P([(mx - sr * (yb - piv) - .050, yb + .004), (mx - sr * (yb - piv), yb - .004), (mx - sr * (yb - piv) + .046, yb + .004)]), .46, seed=104, taper=(.5, .5), alpha=.9)
        if X['naso'] > .03:
            for q in (-1, 1):
                x1 = hw + .050; y_ = U[0, 1] if q < 0 else U[-1, 1]; xe = abs((U[0, 0] if q < 0 else U[-1, 0])) + .050
                pts = [(q * .118, -.430), (q * (.12 + (xe - .12) * .45), -.496), (q * xe, y_ + .035), (q * (xe + .008), y_ - .05 - op * .25)]
                pr = H.Pm(spl(pts, n=10)); m = len(pr); e = int(m * (.45 + .5 * X['naso']))
                if H.vis(pts[2][0], -.6): s.ln(pr[int(m * .14):e], .52, seed=105 + q, taper=(.5, .55), alpha=min(1, .5 + X['naso'] * .6), col=mixc(s.ink, s.skin(2), .10))

    def contour(s):
        """outer line: jaw and far cheek only (the top is under the hair, the near temple edge carries no line)"""
        H = s.H; F = s.F
        # drop the part above the brow on the near side
        near = -1 if H.yaw > 0 else 1
        top = H.P([(0, .46)])[0]; up = -H.R[:, 1][:2] if False else None
        ax = np.array([math.sin(H.roll), -math.cos(H.roll)])        # up on screen
        cen = H.P3([(0, 0, 0)])[0]
        v = (F - cen) @ ax / s.u
        keep = v < .52 * math.cos(H.pitch) + .35 * math.sin(H.pitch)
        # near vertical edge of the shell in front of the ear: no line above the lobe
        if abs(H.yaw) > .10:
            lob = H.P3([(near * .70, -.30, -.08)])[0]; rt = np.array([math.cos(H.roll), math.sin(H.roll)])
            e = (F - lob) @ rt * near / s.u; keep &= ~((v > -.33) & (e > -.14))
        n = len(F); i0 = int(np.argmin(keep)) if not keep.all() else 0
        K = np.r_[keep[i0:], keep[:i0]]; G = np.r_[F[i0:], F[:i0]]
        runs = []; j = 0
        while j < n:
            if K[j]:
                e = j
                while e + 1 < n and K[e + 1]: e += 1
                if e - j > 6: runs.append(G[j:e + 1])
                j = e + 1
            else: j += 1
        ld = A([s.L['side'] * .6, -.8])
        for r, g in enumerate(runs):
            k = np.hanning(9); k /= k.sum(); pad = np.vstack([g[:1]] * 4 + [g] + [g[-1:]] * 4); g = np.stack([np.convolve(pad[:, i], k, 'valid') for i in (0, 1)], -1)
            g = resample(g, 1.6); T = np.gradient(g, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-9; N = np.c_[T[:, 1], -T[:, 0]]
            if ((g + N * 3 - cen) ** 2).sum() < ((g - cen) ** 2).sum(): N = -N
            hv = .78 + .52 * np.clip(-(N @ ld), -.75, 1); kk = np.hanning(61); kk /= kk.sum()
            hv = np.convolve(np.r_[np.full(30, hv[0]), hv, np.full(30, hv[-1])], kk, 'valid') if len(hv) > 4 else hv
            s.c.line(g, s.w, s.ink, taper=(.12, .12), press=.42, seed=s.seed + 121 + r, tremor=.2, wfun=lambda q, hv=hv: np.interp(q, np.linspace(0, 1, len(hv)), hv))

    def blacks(s):
        """spotted blacks of a key animator: under brow, beside nose, under nose and lip, cheek hollow; edges feathered with pen strokes"""
        X, H, c = s.X, s.H, s.c; k = float(X['black'])
        if k < .05: return
        col = mixc(s.skin(2), s.ink, .35 + .55 * k); sg = -s.L['side']; r = np.random.default_rng(s.seed + 808)
        def feather(a, b, n, l, dx, dy, w=.40):                    # strokes of length l leaving the edge a->b in direction (dx,dy)
            for j in range(n):
                u = (j + r.uniform(-.25, .25)) / max(1, n - 1); x, y = a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u; q = l * r.uniform(.45, 1.15)
                if H.vis(x * 1.04, y): s.ln(H.Pc([(x, y), (x + dx * q * .5, y + dy * q * .5), (x + dx * q + r.normal(0, .004), y + dy * q)]), w, col=col, seed=400 + j, taper=(.03, .8), press=.2)
        for q in (-1, 1):
            if not H.vis(.2 * q, .05): continue
            sfx = 'R' if q > 0 else 'L'; B = A(X.get('B' + sfx) if X.get('B' + sfx) is not None else X['B'], float); U = A(X.get('U' + sfx) if X.get('U' + sfx) is not None else X['U'], float); Lo = A(X.get('Lo' + sfx) if X.get('Lo' + sfx) is not None else X['Lo'], float)
            sh = q == sg; by = lambda x: np.interp(x, B[:, 0], B[:, 1]) - .036
            xe = .325 if sh else .205
            P = [(B[0, 0] + .006, by(B[0, 0]) + .004), (.16, by(.16)), (.22, by(.22)), (xe, by(xe) - .004), (xe - .045, (by(xe) + U[2]) / 2 + .020), (.225, U[1] + .034), (.182, U[0] + .042), (.160, U[0] + .004), (.138, -.105), (.122, -.050), (.112, .030)]
            c.fill(H.Pc(spl([(q * a, b) for a, b in P], True, corners=(0, 3, 8), n=6)), col, .55 + .45 * k if sh else .35 + .45 * k)
            feather((q * .105, -.110), (q * .075, .030), 6, .07, q * -.15, -1.)
            # the outer edge of the black breaks up into hatching that follows the lid
            ya, yb_ = by(xe) - .004, (by(xe) + U[2]) / 2 + .020; feather((q * (xe - .012), ya), (q * (xe - .050), yb_), 7 if sh else 5, .115 if sh else .080, q * .96, -.22, .46); feather((q * (xe + .030), ya - .006), (q * (xe + .004), yb_ - .004), 5 if sh else 3, .085, q * .97, -.20, .34)
            # pouch: a crescent of black under the lower lid, heavier inside
            y0 = Lo[1] - .030
            if X['squeeze'] > .5: continue
            P = [(.150, Lo[0] - .020), (.205, y0 - .020), (.300, y0 - .030), (.395, y0 - .018), (.330, y0 - .052), (.235, y0 - .056), (.165, y0 - .030)]
            c.fill(H.Pc(spl([(q * a, b) for a, b in P], True, corners=(0, 3), n=6)), col, .35 + .4 * k if sh else .22 + .3 * k)
            feather((q * .150, y0 - .035), (q * .235, y0 - .066), 7, .105, q * -.50, -.86, .36)
        # side of the nose and the plane under it
        side = s.N([(sg * .010, -.06, 1), (sg * .016, -.22, 1), (sg * .020, -.395, 1), (sg * .004, -.436, .92), (sg * .05, -.470, .42), (sg * .104, -.470, .16), (sg * .098, -.420, .2), (sg * .060, -.34, .45), (sg * .044, -.20, .6)])
        if sg * H.yaw < .6: c.fill(side, col if X['squeeze'] < .3 else mixc(s.skin(2), col, .35), .50 + .4 * k)
        c.fill(s.N([(-.080, -.455, .32), (-.03, -.445, .9), (.03, -.445, .9), (.080, -.455, .32), (.066, -.480, .2), (sg * .03, -.512, .02), (-.066, -.480, .2)]), col if X['squeeze'] < .3 else s.skin(2), .6 + .4 * k)
        # stress lines over the bunched cheek
        hw = X['hw']
        for q in (-1, 1):
            for (x0, y0), (x1, y1), (x2, y2), w in (((.40, -.225), (.48, -.250), (.545, -.300), .40),):
                if H.vis(q * x1, y1): s.ln(H.Pc([(q * x0, y0), (q * x1, y1), (q * x2, y2)]), w, seed=420 + int(x0 * 90) + q, taper=(.5, .6), alpha=k)
        # cheek hollow on the dark side
        if X['squeeze'] > .3: col = mixc(s.skin(2), s.ink, .25)
        y1 = X['my'] + .02; hw = X['hw']
        P = [(.66, -.30), (.56, -.34), (hw + .135, -.45), (hw + .080, y1 - .02), (hw + .070, y1 - .16), (hw + .12, y1 - .06), (hw + .19, -.50), (.60, -.46), (.68, -.42)]
        c.fill(H.Pm(spl([(sg * a, b) for a, b in P], True, corners=(0, 3, 4, 8), n=6)), col, .45 + .4 * k)
        if X['squeeze'] > .3: return
        feather((sg * .60, -.33), (sg * (hw + .12), -.46), 12, .10, sg * .20, 1., .36); feather((sg * (hw + .12), -.52), (sg * (hw + .075), y1 - .14), 9, .08, sg * .9, -.35, .34)

    def features(s):
        s.c.clip(s.F) if abs(s.H.yaw) < .75 else None
        s.blacks()
        for q in (-1, 1): s.eye(q)
        s.tearlines(); s.mouth()
        if abs(s.H.yaw) < .75: s.c.unclip()
        s.nose()
        for q in (-1, 1): s.brow(q)
        s.furrow()
        s.effects()

    # -- water, sweat, veins, dirt
    def stream(s, pts, w, seed=0, al=.62):
        """a running line of water over the skin: pts in front coordinates"""
        c, H, L = s.c, s.H, s.L
        P = spl(A(pts, float), n=10); n = len(P); t = np.linspace(0, 1, n)
        T = np.gradient(P, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]
        r = np.random.default_rng(seed + s.seed * 3); wd = w * (.55 + .30 * np.sin(t * 9 + r.uniform(0, 6)) ** 2 + .15 * r.random(1)) * np.clip(t * 6, .25, 1); wd[-6:] *= A([1.2, 1.45, 1.55, 1.35, .95, .35])
        Q = H.Pc(np.vstack([P + N * wd[:, None], (P - N * wd[:, None])[::-1]]))
        c.fill(Q, mixc(L['white'][0], '#a9cdf0', .30), al)
        c.line(H.Pc(P + N * (wd * .95)[:, None]), max(.8, s.w * .20), mixc(s.skin(2), s.ink, .35), taper=(.5, .1), seed=seed, alpha=.8)
        c.line(H.Pc((P - N * (wd * .35)[:, None])[int(n * .10):int(n * .55)]), max(.8, s.w * .22), '#ffffff', taper=(.3, .6), seed=seed + 1, alpha=.95)
        e = P[-3]; c.fill(H.Pc(spl([(e[0] - wd[-4] * .5, e[1]), (e[0], e[1] + wd[-4] * .6), (e[0] + wd[-4] * .3, e[1]), (e[0], e[1] - wd[-4] * .5)], True)), '#ffffff', .95)

    def bead(s, x, y, r, ang=0.):
        """a drop of sweat or rain standing on the skin"""
        H = s.H; c = s.c; u = r; q = np.linspace(0, 2 * np.pi, 33)[:-1]; ang = ang - H.roll; ca, sa = math.cos(ang), math.sin(ang); cen = H.Pc([(x, y)])[0]          # drawn in screen space so that it never smears at the edge of the face
        def D(k, ox=0., oy=0., sy=1.):
            X_, Y_ = u * k * 1.02 * np.sin(q) * np.abs(np.sin(q / 2)) ** 1.1 + ox * u, u * k * 1.45 * sy * np.cos(q) + oy * u
            return np.c_[cen[0] + (X_ * ca - Y_ * sa) * H.s, cen[1] - (X_ * sa + Y_ * ca) * H.s]
        c.fill(D(1.05, -s.L['side'] * .35, -.30), mixc(s.skin(2), s.ink, .25), .45)                   # its little cast shadow
        c.fill(D(1.), mixc(s.skin(1), s.skin(2), .35), .85); c.fill(D(.80, s.L['side'] * -.06, -.34, .70), mixc(s.L['white'][0], s.skin(0), .30), .90); c.fill(D(.46, 0., -.10, .7), mixc(s.skin(1), s.skin(0), .3), .8)
        Q = D(1.); c.line(Q[6:27], max(.8, s.w * .30), s.ink, taper=(.35, .35), seed=s.seed + int(abs(x) * 99), alpha=.85)
        e = cen + A([s.L['side'] * .30, -.28]) * u * H.s; rr = u * H.s * .20; c.fill(np.c_[e[0] + np.cos(q) * rr * .7, e[1] + np.sin(q) * rr], '#ffffff', .95)

    def tearlines(s):
        X, H = s.X, s.H
        if X['tears'] > .02:
            k = X['tears']
            for q in (-1, 1):
                if not H.vis(q * .30, -.2): continue
                y0 = float(np.mean(A(X['Lo'], float)[1:4])); e = 1.0
                a = [(q * .250, y0 + .004), (q * .235, y0 - .12), (q * .215, -.36), (q * .255, -.50), (q * .300, -.64), (q * .285, -.80), (q * .215, -.93)]
                b = [(q * .425, y0 + .012), (q * .455, y0 - .10), (q * .470, -.34), (q * .455, -.52), (q * .400, -.70)]
                s.stream(a[:max(3, int(round(len(a) * min(1, k + .15))))], .0165, 3 + q, .52)
                if k > .45: s.stream(b[:max(3, int(round(len(b) * k)))], .011, 5 + q, .48)

    def effects(s):
        X, H, c = s.X, s.H, s.c
        cl = True
        c.clip(s.F)
        if X['hatch'] > .02: s.hatching()
        if s.wet > .05: s.rainskin()
        if s.dmg > .02: s.damage_face()
        if cl: c.unclip()
        if X['sweat'] > .05:
            d = 1 if H.yaw >= 0 else -1
            s.bead(-d * .50, .10, .030 * X['sweat']); s.bead(d * .30 if H.vis(d * .30, .35) else -d * .1, .34, .020 * X['sweat'])
            if X['sweat'] > .7: s.bead(-d * .46, -.48, .024)
        if X['vein'] > .3:
            for q in (-1, 1):
                if H.vis(q * .50, .34):
                    s.ln(H.Pc([(q * .46, .46), (q * .50, .40), (q * .475, .345), (q * .53, .30)]), .40, seed=131 + q, taper=(.3, .5)); s.ln(H.Pc([(q * .50, .40), (q * .55, .385), (q * .58, .40)]), .32, seed=133 + q, taper=(.1, .7))
                    s.ln(H.Pc([(q * .475, .345), (q * .43, .325), (q * .41, .29)]), .30, seed=135 + q, taper=(.1, .7))

    def hatching(s):
        """scratched key-animator hatching: under the eyes, beside the nose, under the cheekbone"""
        H = s.H; r = np.random.default_rng(s.seed + 55); k = s.X['hatch']; col = mixc(s.ink, s.skin(2), .35)
        for q in (-1, 1):
            sh = q == -s.L['side']
            for (x0, y0, x1, y1, n, l, ang) in ((.125, -.075, .205, -.160, 6, .075, -1.25), (.085, .100, .140, .020, 5, .045, -.95), (.43, -.34, .60, -.14, 12 if sh else 0, .13, -1.2), (.34, -.80, .46, -.62, 8 if sh else 0, .09, -1.0)):
                for j in range(n):
                    if r.random() > .55 + .5 * k: continue
                    u = (j + r.uniform(-.3, .3)) / max(1, n - 1); x, y = x0 + (x1 - x0) * u, y0 + (y1 - y0) * u; a = ang + r.normal(0, .08); L_ = l * r.uniform(.55, 1.2)
                    if not H.vis(q * x * 1.05, y): continue
                    s.ln(H.Pc([(q * x, y), (q * (x + math.cos(a) * L_ * .5), y + math.sin(a) * L_ * .5), (q * (x + math.cos(a) * L_), y + math.sin(a) * L_)]), .26, col=col, seed=200 + j, taper=(.1, .7), alpha=min(1., .5 + k * .5))

    def rainskin(s):
        H, c = s.H, s.c; r = np.random.default_rng(s.seed * 5 + 17); n = int(26 * s.wet)
        ls = s.L['side']; g = mixc(s.L['white'][0], s.skin(0), .28)
        for P, al in (([(ls * .012, -.10), (ls * .026, -.24), (ls * .022, -.35), (ls * .006, -.26)], .9), ([(ls * .02, -.888), (ls * .10, -.880), (ls * .17, -.856), (ls * .10, -.892)], .6), ([(ls * .30, .128), (ls * .40, .132), (ls * .47, .105), (ls * .39, .116)], .7)):
            if H.vis(*P[1]): c.fill(H.Pc(spl(P, True, corners=(0, 2), n=6)), g, al * s.wet)
        # hand-placed drops and trails (front coordinates: x, y, length or 0 for a standing drop)
        for j, (x, y, l) in enumerate(((-.04, .42, .11), (.22, .46, .07), (-.30, .40, .05), (.50, .30, .10), (-.56, .16, .12), (.07, .30, 0), (-.18, .33, 0), (.52, -.36, .10), (-.56, -.30, .14), (-.50, -.62, 0), (.46, -.56, 0), (.60, -.20, 0),
                                        (-.60, -.06, 0), (.04, -.885, 0), (-.16, -.905, .06), (.24, -.84, 0), (.10, -.22, 0), (-.08, -.30, 0), (-.37, -.50, 0), (.36, -.36, 0), (.015, -.985, .03))):
            if j / 21. > s.wet or not H.vis(x * 1.05, y): continue
            if l:
                P = spl(H.Pc([(x, y), (x + r.normal(0, .006), y - l * .33), (x + r.normal(0, .005), y - l * .66), (x + r.normal(0, .008), y - l)]), n=8)
                c.line(P, max(.9, .011 * s.u), mixc(s.L['white'][0], s.skin(0), .25), taper=(.6, .08), seed=j, alpha=.8); c.line(P + A([.008 * s.u, 0]), max(.7, .005 * s.u), s.skin(2), taper=(.5, .2), seed=j + 1, alpha=.6)
                e = H.Pc([(x, y - l)])[0]; k = .011 * s.u; a = np.linspace(0, 2 * np.pi, 11)[:-1]; c.fill(np.c_[e[0] + np.cos(a) * k * .7, e[1] + np.sin(a) * k], mixc(s.L['white'][0], s.skin(0), .2), .9)
            else:
                k = r.uniform(.007, .012); a = np.linspace(0, 2 * np.pi, 11)[:-1]
                c.fill(H.Pc(np.c_[x + np.cos(a) * k * .8, y + np.sin(a) * k * 1.2]), mixc(s.L['white'][0], s.skin(0), .35), .85); c.fill(H.Pc(np.c_[x + k * .3 + np.cos(a) * k * .5, y - k * .55 + np.sin(a) * k * .45]), s.skin(2), .55)

    def damage_face(s):
        """cuts, dirt, dried blood: level 0..1"""
        H, c, d = s.H, s.c, s.dmg; r = np.random.default_rng(404); red = mixc('#7c1420', s.skin(2), .35)
        # dirt: a breath of soot, a broken flat smudge, flecks and one or two drags
        dirt = mixc(s.skin(1), '#2f282b', .50)
        for k, (x, y, w, h, a) in enumerate(((-.47, -.36, .13, .075, -.55), (.40, -.47, .10, .055, .6), (.20, .40, .12, .04, .15), (-.12, -.87, .09, .035, -.1), (.54, .12, .045, .08, .2), (-.28, .34, .06, .028, .3))):
            if k / 6 > d: break
            if not H.vis(x, y): continue
            ca, sa = math.cos(a), math.sin(a); R2 = lambda q: np.c_[x + q[:, 0] * ca - q[:, 1] * sa, y + q[:, 0] * sa + q[:, 1] * ca]
            dl = .45 + .55 * d; rad(c, None if abs(H.yaw) > .75 else s.F, H.Pc([(x, y)])[0], w * 1.5 * s.u, dirt, .30 * dl, ry=h * 1.9 * s.u, rot=-a + H.roll)
            for part in range(3):
                m = 9; q = np.linspace(0, 2 * np.pi, m, endpoint=False) + r.uniform(0, 1); rr = r.uniform(.45, 1.15, m); ox, oy, sc = r.normal(0, w * .55), r.normal(0, h * .55), r.uniform(.28, .62)
                c.fill(H.Pc(R2(np.c_[ox + np.cos(q) * w * sc * rr, oy + np.sin(q) * h * sc * rr])), dirt, .42 * dl)
            for j in range(7):
                q = A([[r.normal(0, w * .9), r.normal(0, h * .9)]]); e = r.uniform(.004, .010); z = R2(q + A([[-e, 0], [0, e * .8], [e * 1.3, 0], [0, -e * .6]])); c.fill(H.Pc(z), dirt, .6)
            for j in range(2):
                oy = r.normal(0, h * .4); l = w * r.uniform(.6, 1.2); ox = r.normal(0, w * .3)
                c.line(H.Pc(spl(R2(A([[ox - l, oy], [ox, oy + h * r.normal(0, .25)], [ox + l, oy + h * r.normal(0, .3)]])), n=8)), max(.8, .009 * s.u), dirt, taper=(.5, .6), seed=j + k * 9, alpha=.6)
        cuts = [([(.22, -.235), (.30, -.27), (.39, -.295)], 1.), ([(-.50, .34), (-.46, .27), (-.43, .20)], .9), ([(-.28, -.40), (-.22, -.44), (-.17, -.455)], .8), ([(.10, -.600), (.115, -.655), (.105, -.70)], .7), ([(-.06, -.16), (0, -.19), (.07, -.175)], .75), ([(.47, -.60), (.43, -.66), (.36, -.71)], .7)]
        for k, (P, w) in enumerate(cuts):
            if k / len(cuts) > d: break
            if not H.vis(*P[1]): continue
            Q = spl(H.Pc(A(P)), n=8); c.line(Q, max(1., .020 * s.u * w), red, taper=(.4, .5), seed=k); c.line(Q, max(.8, .008 * s.u * w), s.ink, taper=(.35, .5), seed=k + 3)
        if d > .55:
            s_ = [(-.47, .26), (-.50, .16), (-.545, .02), (-.56, -.14)]
            if H.vis(*s_[2]): c.line(H.Pc(spl(A(s_), n=8)), max(1., .022 * s.u), red, taper=(.1, .6), seed=9)


# ------------------------------------------------------------------ hair
def _surf(H, th, y, infl=0.):
    """skull point + outward normal at azimuth th, height y"""
    th = np.asarray(th, float); y = np.broadcast_to(np.asarray(y, float), th.shape)
    A_ = np.interp(y, H.yy, H.A); D = np.where(np.cos(th) >= 0, np.interp(y, H.yy, H.F), np.interp(y, H.yy, H.B)); p = H.k['p']
    sx, cz = np.sin(th), np.cos(th); r = (np.abs(sx) ** p + np.abs(cz) ** p) ** (-1 / p)
    P = np.stack([A_ * sx * r, y, D * cz * r], -1)
    n = np.stack([np.sign(sx) * np.abs(sx) ** (p - 1) / np.maximum(A_, .12), np.clip(y - .12, -.8, 1.2) * 1.25, np.sign(cz) * np.abs(cz) ** (p - 1) / np.maximum(D, .12)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return P + n * np.asarray(infl)[..., None] if np.ndim(infl) else P + n * infl, n


def _patch(H, th, lo, hi, infl=0., front=True, nv=22, close=3, cond=None):
    """screen contours of the front- (or back-) facing part of a skull band  lo(th) <= y <= hi(th)"""
    v = np.linspace(0, 1, nv)[:, None]; Y = lo[None] * (1 - v) + hi[None] * v; TH = np.repeat(th[None], nv, 0)
    P, n = _surf(H, TH, Y, infl)
    f = n @ H.R[2]; q = H.P3(P)
    lo_, hi_ = q.reshape(-1, 2).min(0) - 6, q.reshape(-1, 2).max(0) + 6
    k = min(1., 700. / max(hi_ - lo_)); k = max(k, .25)
    w, h = np.ceil((hi_ - lo_) * k).astype(int) + 2
    m = np.zeros((h, w), np.uint8); q = (q - lo_) * k
    quad = np.stack([q[:-1, :-1], q[:-1, 1:], q[1:, 1:], q[1:, :-1]], 2).reshape(-1, 4, 2)
    fq = ((f[:-1, :-1] + f[:-1, 1:] + f[1:, 1:] + f[1:, :-1]) / 4).reshape(-1)
    quad = quad[fq > -.02] if front else quad[fq < .08]
    if cond is not None:
        nq = ((n[:-1, :-1] + n[:-1, 1:] + n[1:, 1:] + n[1:, :-1]) / 4).reshape(-1, 3) @ H.R.T; sel = (fq > -.02) if front else (fq < .08); quad = quad[cond(nq[sel])]
    for c in np.round(quad * 16).astype(np.int32): cv2.fillConvexPoly(m, c, 255, cv2.LINE_8, 4)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    return [c / k + lo_ for c in contours(m / 255., min_len=12)]


class _Clump:
    __slots__ = ('poly', 'L', 'R', 'C', 'depth', 'face', 'n', 'kind', 'len', 'wid', 'id', 'out', 'dir', 'big')


def _hairline(th):
    """lower edge of the pink mass (y) by azimuth"""
    a = np.abs(th)
    return np.interp(a, [0, .25, .55, .80, 1.0, 1.15, 1.45, 1.9, 2.5, 3.1416], [.585, .590, .565, .49, .42, .365, .315, .285, .245, .225])


def _zig(th):
    """irregular saw edge of the mass (hair roots never end in a clean line)"""
    u = th * 9.0 + .55 * np.sin(th * 6.1 + .4) + .35 * np.sin(th * 13.7)
    k = np.floor(u); a = np.abs(np.sin(k * 12.9898) * 43758.5453) % 1; f = u - k; pk = .35 + .3 * a
    return -(.018 + .045 * a) * np.where(f < pk, f / pk, (1 - f) / (1 - pk)) ** 1.3


# authored clump table: (kind, theta, y, length, width, up, forward, outward, lean, [children (dtheta_dir, len, width, side)])
_F = [(-1.04, .435, .34, .150, .40, .02, .85, -.05), (-.88, .480, .44, .175, .66, .30, .66, .00), (-.71, .520, .30, .115, .84, .46, .46, .06), (-.55, .540, .56, .205, .92, .46, .34, .10),
      (-.37, .565, .36, .130, .98, .56, .20, .18), (-.21, .572, .62, .215, 1.0, .46, .06, .20), (-.05, .570, .38, .125, 1.0, .58, .0, .34), (.11, .575, .55, .195, 1.0, .50, .06, .36),
      (.28, .568, .34, .120, .96, .56, .20, .30), (.45, .555, .58, .205, .90, .46, .36, .18), (.63, .535, .36, .130, .78, .44, .52, .08), (.80, .505, .48, .185, .66, .30, .68, .04), (.98, .450, .36, .145, .44, .10, .85, .0)]
_M = [(-1.18, .56, .38, .19, .56, -.10, .72, 0), (-.94, .64, .50, .21, .78, .10, .60, .02), (-.66, .70, .40, .17, .94, .22, .40, .08), (-.38, .73, .58, .22, 1., .20, .22, .12), (-.12, .74, .42, .17, 1.05, .24, .04, .22),
      (.16, .74, .56, .21, 1.03, .20, .12, .24), (.42, .72, .40, .17, .96, .22, .32, .18), (.70, .67, .52, .21, .82, .10, .55, .08), (.98, .60, .40, .18, .64, -.05, .70, .02), (1.2, .53, .36, .16, .50, -.15, .78, 0)]
_T = [(-1.45, .76, .42, .20, .78, -.30, .58, 0), (-.95, .84, .50, .22, .94, -.10, .46, .02), (-.50, .88, .40, .18, 1.04, -.05, .28, .08), (-.05, .89, .56, .22, 1.08, -.04, .04, .14), (.42, .88, .44, .19, 1.04, -.05, .26, .12),
      (.88, .85, .52, .22, .92, -.10, .44, .04), (1.4, .77, .42, .20, .80, -.30, .60, 0)]
_K = [(-2.6, .93, .42, .20, .85, -.55, .3, 0), (-1.7, .94, .44, .21, .95, -.40, .38, 0), (-.4, .97, .50, .22, 1.08, -.20, .12, .05), (.5, .965, .42, .18, 1.05, -.22, .2, .08), (1.6, .945, .46, .21, .95, -.40, .40, 0), (2.6, .93, .40, .20, .88, -.55, .3, 0), (3.14, .95, .44, .20, .92, -.55, .0, 0),
      (1.57, .988, .54, .21, 1.1, -.10, .14, .04), (-1.5, .984, .46, .19, 1.1, -.16, .16, 0), (.95, .990, .40, .16, 1.1, .02, .10, .06), (-2.25, .978, .52, .20, 1.05, -.34, .16, 0), (2.35, .982, .46, .19, 1.05, -.36, .16, 0), (2.95, .975, .52, .20, 1., -.48, .06, 0)]
_S = [(1.10, .395, .26, .125, .14, -.20, .74), (1.26, .372, .33, .150, .06, -.42, .70), (1.46, .345, .27, .130, .02, -.60, .64), (1.66, .335, .34, .155, .06, -.72, .60), (1.90, .318, .28, .140, -.04, -.85, .55),
      (1.30, .50, .38, .175, .42, -.42, .66), (1.56, .48, .42, .180, .34, -.60, .62), (1.84, .46, .38, .170, .24, -.78, .56), (1.42, .64, .42, .195, .62, -.40, .62), (1.75, .64, .40, .190, .52, -.62, .56)]
_B = [(2.15, .290, .28, .150, -.08, -.9, .5), (2.45, .270, .25, .140, -.16, -.95, .4), (2.78, .260, .29, .150, -.18, -1., .22), (2.15, .46, .36, .180, .16, -.85, .5), (2.5, .45, .36, .180, .12, -.95, .36), (2.85, .44, .33, .175, .08, -1., .16),
      (2.2, .66, .40, .200, .46, -.75, .48), (2.65, .68, .42, .200, .42, -.88, .28), (2.3, .82, .42, .210, .68, -.62, .36)]


# short points that fall out of the swept mass over the forehead (theta, y, length, width, down, forward, lean)
_TEETH = [(-.92, .430, .15, .060, -.55, .5, -.1), (-.64, .525, .20, .070, -.62, .6, .10), (-.47, .560, .12, .055, -.5, .7, .1), (-.30, .575, .26, .080, -.60, .7, .22), (-.13, .580, .13, .055, -.55, .8, .2), (.03, .575, .19, .070, -.62, .8, .28),
          (.20, .585, .11, .050, -.5, .8, .2), (.36, .570, .24, .078, -.62, .7, .12), (.52, .555, .13, .055, -.55, .6, .05), (.69, .515, .19, .068, -.62, .55, -.05), (.94, .435, .15, .060, -.55, .45, 0.)]


def _hair_spec(seed=11):
    r = np.random.default_rng(seed); out = []
    for th, y, L, w, up, fw, ln in _TEETH: out.append(('tooth', th, y + .055, L * 1.12, w * 1.1, up, fw, .30, ln, False))
    for sg in (-1, 1):                       # the same along the sides and back, over the undercut
        for k, th in enumerate((1.06, 1.17, 1.31, 1.42, 1.57, 1.70, 1.86, 2.02, 2.22, 2.42, 2.66, 2.92)):
            th = sg * (th + r.normal(0, .025)); y = float(_hairline(np.array(th))) + .075; L = (.19, .13, .22, .15)[k % 4] * r.uniform(.9, 1.15)
            out.append(('tooth', th, y, L, L * .42, -.50, -.30 - .25 * abs(math.sin(th)), .42, 0., False))
    def add(kind, th, y, L, w, up, fw, ou, ln=0., big=True): out.append((kind, th, y, L, w, up, fw, ou, ln, big))
    def fam(kind, th, y, L, w, up, fw, ou, ln=0.):
        """one clump and the small points that split off it"""
        add(kind, th, y, L, w, up, fw, ou, ln)
        if w > .16:
            for sg, p in ((-1, .62), (1, .50)):
                if r.random() < p + .18:
                    a = sg * r.uniform(.16, .38); add(kind, th + sg * r.uniform(.05, .12), y + r.uniform(-.02, .06), L * r.uniform(.60, .88), w * r.uniform(.42, .62), up, fw, ou, ln + a, False)
    for k, tab in (('front', _F), ('mid', _M), ('top', _T), ('crown', _K)):
        for v in tab: fam(k, *v)
    for sg in (-1, 1):
        for k, tab in (('side', _S), ('back', _B)):
            for th, y, L, w, up, fw, ou in tab:
                j = r.normal(0, 1, 4); fam(k, sg * (th + .04 * j[0]), y + .014 * j[1], L * (1 + .14 * j[2]), w * (1 + .10 * j[3]), up, fw, ou)
    for v in ((3.14, .25, .27, .15, -.16, -1., 0), (3.14, .45, .36, .18, .10, -1., 0), (3.10, .66, .40, .20, .44, -.9, 0), (3.14, .82, .42, .20, .70, -.65, 0)): fam('back', *v)
    return out


LEN = dict(front=.90, mid=.82, top=.80, crown=.80, side=.92, back=.90)
WID = dict(front=.84, mid=.78, top=.76, crown=.76, side=.80, back=.78)
_SPEC = None


def _clumps(H, f=0, wind=0., wet=0., seed=1):
    global _SPEC
    if _SPEC is None: _SPEC = _hair_spec()
    wv = np.zeros(3)
    if np.ndim(wind) == 0: wv[:2] = (float(wind), .12 * float(wind))
    else: wv[:len(wind)] = wind
    amp = float(np.linalg.norm(wv)); r = np.random.default_rng(101); out = []
    wl = wv @ H.R; V = H.R[2]
    for i, (kind, th, y, L, w, up, fw, ou, ln, big) in enumerate(_SPEC):
        j = r.normal(0, 1, 6)
        root, n = _surf(H, A(th), y, .0); root = root + n * {'front': .035, 'tooth': .02, 'side': .03}.get(kind, .085); L = L * LEN.get(kind, 1.); w = w * WID.get(kind, 1.)
        sx = math.sin(th); cz = math.cos(th)
        d0 = A([sx * ou + ln * .60, up, fw]) if kind not in ('side', 'back') else A([sx * ou + ln * .3, up + .3 * ln, fw])
        d0 = d0 / np.linalg.norm(d0)
        d1 = A([sx * ou * .85 + ln * 1.1, up * 1.02, fw * .25 - .10]) if kind in ('front', 'mid', 'top', 'crown') else A([sx * .25 + ln * 1.2, up * 1.5, .25]) if kind == 'tooth' else A([sx * ou * .75, up + .22, fw * 1.15])
        d1 = d1 / np.linalg.norm(d1)
        d0 = d0 + .05 * j[:3]; d1 = d1 + A([.13 * j[0], .10 * j[1], .10 * j[2]])
        L_ = L
        fall = wet > .3 and kind == 'front' and (not big or min(abs(th - v) for v in (-.71, -.55, -.37, -.05, .28, .45, .63)) < .03)
        if wet > 0 and not fall and kind != 'tooth':             # soaked spikes sag outward and lose volume
            o = A([sx * .95 + .3 * ln, -.10, (.75 * cz if kind in ('front', 'mid', 'top') else .5 * cz)]); w = w * (1 - .12 * wet); L_ = L * (1 - .10 * wet)
            d0 = d0 / np.linalg.norm(d0) * (1 - .30 * wet) + o * .30 * wet; d1 = d1 / np.linalg.norm(d1) * (1 - .62 * wet) + o * .62 * wet
        d0 /= np.linalg.norm(d0); d1 /= np.linalg.norm(d1)
        if amp > 0:
            ph = 2 * math.pi * (f / 22.) - 1.5 * sx - 1.1 * y + .4 * j[4]
            k = .55 + .45 * math.sin(ph) + .16 * math.sin(2.3 * ph + j[5])
            kt = .45 if kind == 'tooth' else 1.; d1 = d1 + wl * (1.05 * k * kt); d0 = d0 + wl * .30 * k * kt
            d1 /= np.linalg.norm(d1); d0 /= np.linalg.norm(d0)
        # clumps that point at the camera are drawn fanned out, as an animator would
        for d in (d0, d1):
            dv = d @ V
            if dv > .42: d -= V * (dv - .42) * .9
        d0 /= np.linalg.norm(d0); d1 /= np.linalg.norm(d1)
        t = np.linspace(0, 1, 19)[:, None]
        p0, p1, p2, p3 = root, root + d0 * L_ * .42, root + d0 * L_ * .62 + d1 * L_ * .20, root + d0 * L_ * .52 + d1 * L_ * .50
        C3 = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t * t * p2 + t ** 3 * p3
        if fall:                             # fringe strands hang over the forehead
            ph0 = math.acos(np.clip(y - .01, -1, .985)); ph = ph0 - .24 + t[:, 0] * (L * .80 / .84 + .24)
            Sf, nf = _surf(H, th + (.10 * j[3] + .16 * ln) * t[:, 0] ** 1.5, np.cos(ph))
            W3 = Sf + nf * (.085 + .035 * np.sin(math.pi * t) + .02 * t) + (wl * (.06 * k) * t ** 2 if amp > 0 else 0)
            C3 = W3; w = w * .86
        C = H.P3(C3); T = np.gradient(C, axis=0); ln_ = np.hypot(*T.T)[:, None]; T = T / (ln_ + 1e-6); Nn = np.c_[-T[:, 1], T[:, 0]]
        tt = t[:, 0]
        rt = .5 if kind == 'tooth' else 0. if fall else .12 if kind in ('front', 'side') else .45
        wp = _prof(tt, [(0, rt), (.07, .30 + rt * .7), (.16, .90), (.26, 1.), (.40, .88), (.55, .64), (.70, .41), (.83, .21), (.93, .075), (1, 0)]) * w * H.s * .5
        as_ = .22 * j[3]
        cl = _Clump(); cl.C = C; cl.L = C + Nn * (wp * (1 + as_))[:, None]; cl.R = C - Nn * (wp * (1 - as_))[:, None]
        cl.poly = np.vstack([cl.L, cl.R[::-1]]); cl.depth = float(H.depth(C3[5])); cl.face = float(n @ V); cl.n = H.R @ n; cl.kind = kind
        cl.len = float(np.hypot(*(C[-1] - C[0]))); cl.wid = w * H.s; cl.id = i; cl.big = big; cl.out = False
        cl.dir = A([C[-1, 0] - C[0, 0], C[0, 1] - C[-1, 1]]) / (cl.len + 1e-6)
        out.append(cl)
    return out


class _Hair:
    def __init__(s, cel, H, L, f, wind, wet, seed, w, lod):
        s.c, s.H, s.L, s.w, s.seed = cel, H, L, w, seed; s.lod = lod; s.wet = wet
        s.cl = _clumps(H, f, wind, wet, seed)
        s.ink = L['hline'] if L['hline'] else L['line']
        ld = A([L['side'] * .55, .70, .45]) if 'ldir' not in L else A(L['ldir'], float)
        s.ld = ld / np.linalg.norm(ld)
        s.th = np.linspace(-math.pi, math.pi, 121); s.tf = np.linspace(-math.pi, math.pi, 521)
        s._outer()
        v = np.linspace(0, 1, 22)[:, None]; s.dome = (.035 + .105 * smooth(0, .45, v)) * np.ones((1, 121)); s.domef = (.035 + .105 * smooth(0, .45, v)) * np.ones((1, 521))
        s.caplo = _hairline(s.tf) + .035 + _zig(s.tf) - .04 * float(wet > .3)
        s.cap = _patch(H, s.tf, s.caplo, np.full(521, .99), s.domef, front=True)

    def _outer(s):
        """which tips break the silhouette (raster test)"""
        allp = np.vstack([q.poly for q in s.cl]); lo = allp.min(0) - 4; hi = allp.max(0) + 4
        k = min(1., 420. / max(hi - lo)); w, h = np.ceil((hi - lo) * k).astype(int) + 2
        m = np.zeros((h, w), np.uint8)
        for q in s.cl: cv2.fillPoly(m, [np.round((q.poly - lo) * k * 16).astype(np.int32)], 255, cv2.LINE_8, 4)
        d = cv2.distanceTransform(cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)), cv2.DIST_L2, 3) / k
        for q in s.cl:
            i = np.round((q.C[-3] - lo) * k).astype(int); q.out = bool(d[np.clip(i[1], 0, h - 1), np.clip(i[0], 0, w - 1)] < .055 * s.H.s)

    def back(s, lim=-9.):
        H, c, L = s.H, s.c, s.L
        lo = np.maximum(lim, np.interp(np.abs(s.th), [0, 1.0, 1.30, 1.62, 1.95, 2.4, 3.1416], [.3, .2, -.05, -.20, -.40, -.50, -.52]))
        for p in _patch(H, s.th, lo, np.full_like(lo, .97), .012, front=True): c.fill(p, L['under'][0])
        for p in _patch(H, s.th, lo, np.full_like(lo, .97), .012, front=False): c.fill(p, L['under'][1])
        for p in _patch(H, s.th, _hairline(s.th), np.full_like(lo, .985), s.dome, front=False):
            outline(c, p, s.w * 2.1, s.ink, seed=s.seed + 41); c.fill(p, L['hair'][1])
        B = sorted([q for q in s.cl if q.face <= -.12], key=lambda q: q.depth)
        c.ctx.push_group()
        for q in B: s._one(q, behind=True)
        c.ctx.pop_group_to_source(); c.ctx.paint()

    def under(s):
        """undercut on the near temple, sideburn"""
        H, c, L = s.H, s.c, s.L
        Tq = [.90, .985, 1.07, 1.16, 1.26, 1.34, 1.42]; Yq = [.44, .36, .25, .13, -.01, -.09, -.130]
        for sg in (-1, 1):
            ths = np.linspace(.90, math.pi, 60)
            lo = np.interp(ths, Tq + [1.55, 1.75, 1.95, 2.4, 3.1416], Yq + [-.02, -.18, -.40, -.50, -.52])
            hi = _hairline(ths) + .12
            shade = not (sg == L['side'] or L['amt'] < -.2)
            col = L['under'][0] if not shade else mixc(L['under'][0], L['under'][1], .7)
            for p in _patch(H, sg * ths, lo, np.maximum(hi, lo + .02), .010, front=True): c.fill(p, col)
            if s.lod > 0:                    # fuzzy front edge and short cropped texture
                r = np.random.default_rng(s.seed + 300 + sg)
                for j in range(44):
                    tq = r.uniform(.93, 1.40); yq = np.interp(tq, Tq, Yq) + r.normal(0, .010)
                    a = r.uniform(.6, 1.3)
                    P, n = _surf(H, sg * A([tq + .06, tq + .010, tq - .022 * a]), A([yq + .050, yq + .012, yq - .026 * a]), .012)
                    if (n @ H.R[2]).min() < .10: continue
                    c.line(H.P3(P), max(.8, .012 * H.s * r.uniform(.6, 1.2)), mixc(col, L['under'][1], r.random() * .6), taper=(.1, .7), seed=j)
                for j in range(90):
                    tq = r.uniform(1.0, 2.8); yq = r.uniform(-.46, .34)
                    if yq < np.interp(tq, Tq + [1.55, 1.75, 1.95, 2.4], Yq + [.05, -.12, -.36, -.46]) + .05: continue
                    P, n = _surf(H, sg * A([tq, tq + .02, tq + .03]), A([yq, yq - .035, yq - .07]), .012)
                    if (n @ H.R[2]).min() < .12: continue
                    c.line(H.P3(P), max(.7, .009 * H.s), L['under'][1] if j % 3 else mixc(col, '#ffffff', .12), taper=(.3, .6), seed=j, alpha=.75)

    def front(s):
        H, c, L = s.H, s.c, s.L
        lo = _hairline(s.th) + .04
        c.ctx.push_group()
        # mass: one shade tone, everything is built up on it
        for p in _patch(H, s.tf, s.caplo - .022 + .4 * _zig(s.tf + .05), s.caplo + .30, .010, front=True): c.fill(p, L.get('hdeep', mixc(L['hair'][1], s.ink, .22)))
        for p in s.cap: c.fill(p, L['hair'][0])
        hc = L.get('hcut', .06)
        v = np.linspace(0, 1, 22)[:, None]
        for p in _patch(H, s.tf, s.caplo, np.full(521, .99), s.domef, front=True, cond=lambda nq: nq @ s.ld < hc): c.fill(p, L['hair'][1])
        s.edge()
        F = sorted([q for q in s.cl if q.face > -.12], key=lambda q: q.depth)
        for q in F: s._one(q)
        s.wisps()
        c.ctx.pop_group_to_source(); c.ctx.paint()

    def atop(s, P, col, a=1.):
        c = s.c; c.ctx.save(); c.ctx.set_operator(cairo.OPERATOR_ATOP); c.fill(P, col, a); c.ctx.restore()

    def _one(s, q, behind=False):
        c, L, H = s.c, s.L, s.H; b, sh, hi = L['hair']; u = H.s; n = len(q.C)
        d = float(q.n @ s.ld); tl = float(q.dir @ s.ld[:2])
        dark = behind or d < L.get('hcut', .06)
        # shadow it throws on the mass under it (stays inside the hair)
        if not behind and s.lod > 0 and q.big and not dark:
            g = q.C.mean(0); off = A([-s.ld[0] * .026, .020]) * u; h0 = int(n * .30)
            s.atop(np.vstack([q.L[h0:], q.R[h0:][::-1]]) * 1. + off, sh)
        c.fill(q.poly, sh if dark else b)
        Lside = (q.L - q.C)[n // 3] @ A([s.ld[0], -s.ld[1]])
        dk, lt = (q.R, q.L) if Lside > 0 else (q.L, q.R)
        t = np.linspace(0, 1, n)[:, None]
        if not dark:
            # own shade: a blade on the side away from the light that opens toward the belly and closes at the tip
            if (q.big or q.wid > 20) and q.kind != 'tooth':
                z0 = .05 if q.kind == 'front' else .0
                k = np.interp(t, [0, .18, .42, .75, .95, 1.], [z0, .16 + z0 * 4, .56, .44, .12, 0.]) * (1.15 if tl < .2 else .85)
                inn = dk + (q.C - dk) * np.clip(k * 1.35, 0, 1.2)
                c.fill(np.vstack([dk, inn[::-1]]), sh)
            if s.lod > 1 and d > .30 and q.big and (q.id * 5) % 4 < 2 and q.wid > 14:
                i0, i1 = int(n * .34), int(n * .84); sn = np.sin(np.linspace(0.1, math.pi - .05, i1 - i0))[:, None]
                a = q.C[i0:i1] + (lt[i0:i1] - q.C[i0:i1]) * .42; z = a + (lt[i0:i1] - q.C[i0:i1]) * .36 * sn
                c.fill(np.vstack([a, z[::-1]]), hi)
        elif s.lod > 1 and q.big and not behind:
            k = np.interp(t, [0, .3, .7, 1.], [.3, .4, .25, 0.]); inn = dk + (q.C - dk) * k
            c.fill(np.vstack([dk, inn[::-1]]), L.get('hdeep', mixc(sh, s.ink, .28)))
        # line
        wl = s.w * (.60 if q.out else .40)
        if q.kind == 'tooth':
            if s.lod > 0: c.line(q.L[int(n * .35):], max(.8, s.w * .34), s.ink, taper=(.5, .03), press=.2, seed=s.seed * 3 + q.id, tremor=.1); c.line(q.R[int(n * .20):], max(.8, s.w * .34), s.ink, taper=(.5, .03), press=.2, seed=s.seed * 5 + q.id, tremor=.1)
            return
        if s.wet > .3 and not dark and s.lod > 1 and q.big and q.face > .1:       # wet streak
            i0, i1 = int(n * .25), int(n * .75); c.line(q.C[i0:i1] + (lt[i0:i1] - q.C[i0:i1]) * .35, max(1., s.w * .50), hi, taper=(.4, .5), seed=q.id)
        if s.wet > .3 and q.kind == 'front':
            k0 = int(n * .25)
            c.line(dk[k0:], max(.8, s.w * .50), s.ink, taper=(.5, .03), press=.2, seed=s.seed * 3 + q.id, tremor=.1); c.line(lt[int(n * .40):], max(.8, s.w * .44), s.ink, taper=(.5, .03), press=.2, seed=s.seed * 5 + q.id, tremor=.1)
        elif q.out or behind:
            k0 = int(n * (.22 if not behind else .1))
            c.line(q.L[k0:], max(.8, wl), s.ink, taper=(.45, .03), press=.2, seed=s.seed * 3 + q.id, tremor=.1)
            c.line(q.R[k0 + 1:], max(.8, wl), s.ink, taper=(.45, .03), press=.2, seed=s.seed * 5 + q.id, tremor=.1)
        elif s.lod > 0 and q.big and (s.lod > 1 or q.id % 2):
            k0 = int(n * .42); e = (q.id * 7) % 3
            c.line(dk[k0:], max(.8, wl), s.ink, taper=(.5, .03), press=.2, seed=s.seed * 3 + q.id, tremor=.1)
            if e: c.line(lt[int(n * .68):], max(.8, wl * .85), s.ink, taper=(.55, .03), press=.2, seed=s.seed * 5 + q.id, tremor=.1)

    def edge(s):
        """thin broken zig-zag pen line along the roots"""
        if s.lod < 1: return
        P, n = _surf(s.H, s.tf, s.caplo, .035); f = n @ s.H.R[2]; Q = s.H.P3(P); r = np.random.default_rng(s.seed + 31); i = 0
        while i < len(Q) - 4:
            m = int(r.integers(9, 26)); seg = Q[i:i + m]
            if len(seg) > 4 and f[i:i + m].min() > .02: s.c.line(seg, max(.7, s.w * .34), s.ink, taper=(.15, .3), press=.35, seed=i, tremor=.05)
            i += m + int(r.integers(1, 7))

    def wisps(s):
        """8-16 thin interior lines along the growth, at the hairline and between clumps"""
        if s.lod < 1: return
        r = np.random.default_rng(s.seed + 900); H = s.H
        for q in s.cl:
            if q.face < .15 or not q.big: continue
            if q.kind in ('front', 'side') and r.random() < .75:
                for j in range(1 + int(r.random() < .6)):
                    u = r.uniform(-.5, .5); i0 = int(r.uniform(.04, .14) * len(q.C)); i1 = i0 + int(r.uniform(.18, .34) * len(q.C))
                    P = q.C[i0:i1 + 1] + (q.L - q.C)[i0:i1 + 1] * u
                    if len(P) > 2: s.c.line(P, max(.7, s.w * .30), s.ink, taper=(.25, .75), seed=q.id + j, tremor=.1)
            elif s.lod > 1 and r.random() < .35:
                u = r.uniform(-.35, .35); i0 = int(r.uniform(.25, .4) * len(q.C)); i1 = i0 + int(r.uniform(.2, .35) * len(q.C))
                P = q.C[i0:i1 + 1] + (q.L - q.C)[i0:i1 + 1] * u
                if len(P) > 2: s.c.line(P, max(.7, s.w * .28), s.ink, taper=(.6, .5), seed=q.id + 5, tremor=.1)


# ------------------------------------------------------------------ public API
class Yuji:
    light = staticmethod(light); expr = staticmethod(getx); blend = staticmethod(blend); lids = staticmethod(lids)

    @staticmethod
    def head(cel, H, expr='neutral', light='day', f=0, wind=0., seed=1, wet=0., damage=0., lod=None, back=True, front=True, nape=-9., **over):
        X = getx(expr, **over); L = LIGHT[light] if isinstance(light, str) else light
        h = YHead.of(H, X['jaw'], X['jx'], X['slim']); h.k['cheek'] += .028 * X['cheek']; fc = _Face(cel, h, X, L, seed, damage, wet=wet)
        lod = lod if lod is not None else (2 if h.s > 150 else 1 if h.s > 70 else 0)
        hr = _Hair(cel, h, L, f, wind, wet, seed, fc.w, lod); fc.hair = hr
        if back: hr.back(nape)
        elif front: hr.back(-.10)            # separate head layer: keep the back of the hair so that its edge lights read as one mass
        if front:
            fc.ears_far(); fc.base(); fc.contour(); hr.under(); fc.ears_near(); fc.features(); hr.front()
        return fc


def _bbox(a, pad, h, w):
    ys = np.where(a.max(1) > .004)[0]; xs = np.where(a.max(0) > .004)[0]
    if len(ys) == 0: return None
    return max(0, ys[0] - pad), min(h, ys[-1] + pad + 1), max(0, xs[0] - pad), min(w, xs[-1] + pad + 1)


def rim_in(layer, dx, dy, col, amt=1., inset=2., soft=.6, over=False, close=0):
    """hard rim light that sits INSIDE the ink line (over=True: light eats the line). Light comes from (-dx,-dy)."""
    h, w = layer.shape[:2]; bb = _bbox(layer[..., 3], int(abs(dx) + abs(dy)) + 6, h, w)
    if bb is None: return layer
    y0, y1, x0, x1 = bb; L = layer[y0:y1, x0:x1]; al = L[..., 3]; a = np.ascontiguousarray((al * 255).astype(np.uint8))
    K = lambda k: np.ones((int(k) * 2 + 1,) * 2, np.uint8)
    if close >= 2:
        q = cv2.resize(a, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA); r_ = max(1, int(close) // 2)
        q = cv2.morphologyEx(cv2.copyMakeBorder(q, r_, r_, r_, r_, cv2.BORDER_REPLICATE), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_ * 2 + 1,) * 2))[r_:-r_, r_:-r_]
        a = np.maximum(a, cv2.resize(q, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR))
    e = cv2.erode(a, K(inset), borderType=cv2.BORDER_REPLICATE) if not over and inset >= 1 else a
    sh = cv2.warpAffine(e, np.float32([[1, 0, dx], [0, 1, dy]]), (e.shape[1], e.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    m = cv2.subtract(e, sh).astype(f32) / 255.
    if soft > 0: m = cv2.GaussianBlur(m, (0, 0), soft)
    m = (m * ((al > .5) if over else e.astype(f32) / 255.) * al)[..., None] * amt
    out = layer.copy(); out[y0:y1, x0:x1, :3] = L[..., :3] * (1 - m) + _c(col) * m
    return out


def para_fast(layer, p0, p1, stops, mode='mul', amt=1.):
    """kit.para on the bounding box with a colour table"""
    h, w = layer.shape[:2]; bb = _bbox(layer[..., 3], 2, h, w)
    if bb is None: return layer
    y0, y1, x0, x1 = bb; d = np.array(p1, float) - p0; k = 1 / (d @ d + 1e-9)
    tt = np.clip(((np.arange(x0, x1, dtype=f32) - p0[0]) * d[0] * k)[None] + ((np.arange(y0, y1, dtype=f32) - p0[1]) * d[1] * k)[:, None], 0, 1)
    from jj.paint import ramp
    lut = ramp(np.linspace(0, 1, 513), stops).astype(f32); g = lut[(tt * 512).astype(np.int16)]; L = layer[y0:y1, x0:x1]; c = L[..., :3]; a = L[..., 3:4]
    n = c * g if mode == 'mul' else a - (a - c) * (1 - g) if mode == 'screen' else g * a
    out = layer.copy(); out[y0:y1, x0:x1, :3] = c + (n - c) * amt; return out


# ------------------------------------------------------------------ body
class _Body:
    """bust in 3-D landmarks (head units); origin = neck base"""

    def __init__(s, cel, H, L, seed, base=None, byaw=None, bpitch=0., broll=0., damage=0., wet=0., raise_=0., breath=0., f=0, wind=0.):
        s.c, s.H, s.L, s.seed, s.u = cel, H, L, int(seed), H.s; s.w = lw(H.s); s.ink = L['line']; s.dmg = damage; s.wet = wet; s.up = raise_; s.br = breath; s.f = f; s.wind = wind
        byaw = .55 * H.yaw if byaw is None else byaw
        if base is None:
            r = H.roll * .62; d = 1.30 - .12 * H.pitch; fw = -.10 * math.sin(H.yaw)
            base = (H.cx - math.sin(r) * d * H.s + fw * H.s, H.cy + math.cos(r) * d * H.s)
        s.base = A(base, float); s.yaw = byaw
        a, b = byaw, bpitch
        Ry = A([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]); Rx = A([[1, 0, 0], [0, math.cos(b), math.sin(b)], [0, -math.sin(b), math.cos(b)]])
        c, sn = math.cos(-broll), math.sin(-broll); Rz = A([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
        s.R = Rz @ Rx @ Ry

    def P(s, Q):
        q = np.asarray(Q, float) @ s.R.T; f = 1 / (1 - np.clip(q[..., 2] * .085, -.5, .5))
        return np.stack([s.base[0] + q[..., 0] * f * s.u, s.base[1] - q[..., 1] * f * s.u], -1)

    def S(s, Q, closed=False, corners=(), n=10): return spl(s.P(Q), closed, corners, n)

    def ln(s, P, w, col=None, taper=(.3, .4), seed=0, **k):
        P = np.asarray(P, float)
        if 2 < len(P) <= 8: P = spl(P, n=8)
        s.c.line(P, max(.7, w * s.w), s.ink if col is None else col, taper=taper, seed=s.seed * 7 + seed, press=k.pop('press', .25), tremor=k.pop('tremor', .18), **k)

    # ---- hood behind the neck
    def hood_back(s):
        c, L = s.c, s.L; b, sh, hi = L['hood']; sway = .04 * math.sin(s.f / 24 * 2 * math.pi * .7) * min(1, abs(float(np.ravel(s.wind)[0])) if np.size(s.wind) else 0)
        o = [(-.90, -.02, -.10), (-1.04, .26, -.30), (-1.03, .56, -.55), (-.80, .80 + sway, -.80), (-.40, .88 + sway, -.95), (.02, .86, -1.0), (.42, .90 + sway, -.95), (.80, .78 + sway, -.80), (1.03, .54, -.55), (1.04, .24, -.30), (.90, -.02, -.10), (.4, -.5, .1), (-.4, -.5, .1)]
        Po = s.S(o, True); c.fill(Po, sh)
        # outer roll catches light
        roll = [(-.92, .00, -.10), (-1.04, .26, -.30), (-1.03, .56, -.55), (-.80, .80 + sway, -.80), (-.40, .88 + sway, -.95), (.02, .86, -1.0), (.42, .90 + sway, -.95), (.80, .78 + sway, -.80), (1.03, .54, -.55), (1.04, .24, -.30), (.92, .00, -.10),
                (.80, .08, -.02), (.86, .30, -.22), (.80, .50, -.40), (.58, .62, -.62), (.20, .68, -.74), (-.22, .66, -.74), (-.58, .60, -.62), (-.80, .48, -.40), (-.86, .28, -.22), (-.80, .06, -.02)]
        c.fill(s.S(roll, True), b)
        sg = -L['side']
        shp = [(sg * .95, .02, -.12), (sg * 1.06, .28, -.32), (sg * 1.05, .58, -.57), (sg * .84, .80, -.82), (sg * .64, .70, -.72), (sg * .86, .46, -.44), (sg * .92, .22, -.22)]
        c.clip([Po]); c.fill(s.S(shp, True), sh); c.unclip()
        c.fill(s.S([(-.62, .62, -.6), (-.2, .68, -.74), (.22, .68, -.74), (.60, .62, -.6), (.74, .40, -.3), (.5, -.2, .1), (-.5, -.2, .1), (-.74, .40, -.3)], True), L.get('hooddeep', mixc(sh, s.ink, .45)))
        P = s.S(o[:11]); s.c.line(P, s.w * .95, s.ink, taper=(.08, .08), press=.25, seed=s.seed + 3)
        s.ln(s.P([(-.80, .10, -.02), (-.86, .30, -.22), (-.78, .50, -.40), (-.56, .62, -.62)]), .50, seed=4, taper=(.4, .4))
        s.ln(s.P([(.80, .10, -.02), (.86, .30, -.22), (.80, .48, -.40), (.62, .60, -.60)]), .50, seed=5, taper=(.4, .4))
        s.ln(s.P([(-.98, .34, -.35), (-.90, .48, -.46), (-.82, .70, -.7)]), .36, seed=6, taper=(.2, .6)); s.ln(s.P([(.99, .30, -.35), (.93, .46, -.46), (.90, .64, -.6)]), .36, seed=7, taper=(.2, .6))
        s.ln(s.P([(-.30, .86, -.95), (-.26, .78, -.85), (-.14, .70, -.76)]), .34, seed=8, taper=(.1, .7))

    # ---- neck
    def neck(s):
        c, H, L = s.c, s.H, s.L; b, sh, dp = L['skin']; X = s.X
        # the neck is an elliptic column: its two contours are the screen extremes of rings on it
        q = np.linspace(0, 2 * np.pi, 37)[:-1]; rx = A([math.cos(H.roll), math.sin(H.roll)])
        def ext(P): v = P @ rx; return A([P[np.argmin(v)], P[np.argmax(v)]])
        up = ext(H.P3(np.c_[np.cos(q) * .50, np.full(36, -.28), -.22 + np.sin(q) * .40])); mh = ext(H.P3(np.c_[np.cos(q) * .47, np.full(36, -.80), -.20 + np.sin(q) * .36]))
        dn = ext(s.P(np.c_[np.cos(q) * .52, np.full(36, .02), -.14 + np.sin(q) * .40]))
        ml, mr = [mh[i] * .55 + (up[i] * .4 + dn[i] * .6) * .45 for i in (0, 1)]
        fl, fr = ext(s.P(np.c_[np.cos(q) * .84, np.full(36, -.16), -.10 + np.sin(q) * .40]))
        Ll = spl([up[0], ml, dn[0], fl], n=10); Rr = spl([up[1], mr, dn[1], fr], n=10)
        pit = s.P([(0, -.42, .42)])[0]
        c.fill(s.S([(-.86, .30, -.25), (-.74, .02, .30), (-.30, -.30, .62), (.30, -.30, .62), (.74, .02, .30), (.86, .30, -.25), (.6, .45, -.6), (-.6, .45, -.6)], True), L.get('hooddeep', mixc(L['hood'][1], s.ink, .45)))
        N = np.vstack([Ll, [pit], Rr[::-1]]); c.fill(N, b)
        c.clip([N])
        sg = -L['side']
        # cast shadow of the jaw: the jaw line pushed down along the light
        j = np.vstack([H.jaw(-1), H.jaw(1)[::-1][1:]]); d = A([sg * .10, .24 + .10 * max(0, -H.pitch)]) * s.u
        if H.pitch > .25: d = d * .75
        top = np.vstack([H.jaw(-1)[:1] - A([s.u, 2 * s.u]), H.jaw(1)[:1] + A([s.u, -2 * s.u])])
        jj = j + d; zz = jj.copy(); zz[len(jj) // 2, 1] += .04 * s.u
        c.fill(np.vstack([zz, [jj[-1] + A([2 * s.u, 0])], [jj[-1] + A([2 * s.u, -3 * s.u])], [jj[0] + A([-2 * s.u, -3 * s.u])], [jj[0] - A([2 * s.u, 0])]]), sh)
        # side plane of the neck away from the light + tendon (sternomastoid) shapes
        e = H.P3(A([(sg * .56, -.36, -.12)]))[0]; p0 = s.P([(sg * .10, -.36, .40)])[0]; q0 = s.P([(sg * .34, -.10, .20)])[0]
        c.fill(np.vstack([spl([e + A([-sg * .06 * s.u, 0]), (e * .5 + q0 * .5) + A([-sg * .05 * s.u, .0]), q0, p0], n=8), [p0 + A([sg * 2 * s.u, s.u])], [e + A([sg * 2 * s.u, -s.u])]]), sh)
        t = H.P3(A([(-sg * .52, -.40, -.10)]))[0]; tq = s.P([(-sg * .30, -.12, .24)])[0]; tp = s.P([(-sg * .07, -.38, .42)])[0]
        c.fill(np.vstack([spl([t * .35 + tq * .65 + A([-sg * .03, -.10]) * s.u, tq, tp], n=8), spl([tp + A([0, -.03]) * s.u, tq + A([sg * .085, -.02]) * s.u, t * .35 + tq * .65 + A([sg * .05, -.10]) * s.u], n=8)]), sh)
        c.fill(np.vstack([s.P([(-.12, -.33, .44), (0, -.28, .45), (.12, -.33, .44), (.05, -.50, .46), (-.05, -.50, .46)])]), dp if L['style'] != 'day' else sh)
        c.unclip()
        s.c.line(Ll, s.w * .85, s.ink, taper=(.1, .1), press=.2, seed=s.seed + 11); s.c.line(Rr, s.w * .85, s.ink, taper=(.1, .1), press=.2, seed=s.seed + 12)
        vv = 1. + .5 * X['vein']
        if X['vein'] > .6:                                          # strained: both cords stand out, a black behind each, two thin bands of skin pulled between jaw and collar
            kv = min(1., (X['vein'] - .6) / .4)
            for q in (-1, 1):
                a = H.P3(A([(q * .47, -.62, .02)]))[0]; m = s.P([(q * .30, -.06, .24)])[0]; z = s.P([(q * .075, -.40, .44)])[0]; u_ = s.u
                R_ = spl([a * .5 + m * .5, m, z], n=10); L_ = spl([z + A([q * .012, -.01]) * u_, m + A([q * (.075 if q == sg else .050), .015]) * u_, a * .5 + m * .5 + A([q * .030, 0]) * u_], n=10)
                c.fill(np.vstack([R_, L_]), mixc(dp, s.ink, .25) if q == sg else dp, .85 * kv)
                b0, b1 = H.P3(A([(q * .20, -.88, .30)]))[0], s.P([(q * .15, -.24, .38)])[0]
                s.ln(spl([b0 * .70 + b1 * .30, b0 * .3 + b1 * .7 + A([q * .014 * u_, 0]), b1 + A([0, .05 * u_])], n=8), .42 * kv, seed=300 + q, taper=(.5, .6)); s.ln(spl([b0 * .60 + b1 * .40 + A([q * .07, .02]) * u_, b1 + A([q * .080, -.06]) * u_], n=6), .34 * kv, seed=303 + q, taper=(.5, .6))
            vv *= 1.25
        for q in (-1, 1):
            a = H.P3(A([(q * .50, -.52, -.08)]))[0]; m = s.P([(q * .33, -.02, .18)])[0]; z = s.P([(q * .06, -.40, .44)])[0]
            P = spl([a * .62 + m * .38, a * .3 + m * .7 + A([-q * .03 * s.u, 0]), m, z], n=12); n = len(P)
            s.ln(P[int(n * .10):int(n * .98)], .42 * vv, seed=13 + q, taper=(.5, .15))
            if X['vein'] > .3:
                P2 = spl([a * .55 + m * .45 + A([-q * .14, .02]) * s.u, m + A([-q * .13, -.08]) * s.u, z + A([q * .05, -.12]) * s.u], n=10); s.ln(P2, .34, seed=17 + q, taper=(.5, .4))
        ad = (H.P3(A([(0, -.80, .28)]))[0] * .35 + s.P([(0, -.16, .40)])[0] * .65)
        s.ln([ad + A([-.05, -.03]) * s.u, ad + A([-.015, .05]) * s.u, ad + A([.04, .03]) * s.u], .36, seed=20, taper=(.4, .5))
        s.ln(s.P([(-.10, -.36, .45), (-.03, -.42, .46), (0, -.40, .46)]), .36, seed=21, taper=(.4, .4)); s.ln(s.P([(.10, -.36, .45), (.03, -.42, .46)]), .36, seed=22, taper=(.4, .4))

    # ---- jacket
    def _sil(s):
        u = s.up; L_ = []
        half = lambda q: [(q * .70, .10 + u * .1, -.22), (q * 1.02, -.07 + u * .45, -.22), (q * 1.34, -.20 + u * .8, -.20), (q * 1.60, -.40 + u * .9, -.16), (q * 1.74, -.72 + u * .6, -.10), (q * 1.79, -1.12 + u * .3, -.06), (q * 1.74, -1.62, -.04), (q * 1.78, -2.2, .0), (q * 1.80, -3.2, 0)]
        return half(-1), half(1)

    def jacket(s):
        c, L = s.c, s.L; b, hi, bk = L['cloth']; u = s.up; br = s.br * .02; sg = L['side']
        hl, hr = s._sil(); Ol, Or = s.S(hl), s.S(hr); poly = np.vstack([Ol[::-1], Or]); c.fill(poly, b); c.clip([poly])
        a_ = {sg: 1., -sg: .55}
        F = lambda P, col, **k: c.fill(s.S(P, True, corners=range(len(P)), n=6) if not k else s.S(P, True, **k), col)
        for q in (-1, 1):
            m = mixc(b, hi, a_[q]); m2 = mixc(b, hi, a_[q] * .8)
            # deltoid: light rides the top of the shoulder and breaks into two points down the sleeve
            c.fill(s.S([(q * 1.06, -.14 + u * .5, .0), (q * 1.34, -.25 + u * .8, -.02), (q * 1.56, -.46 + u * .9, .0), (q * 1.67, -.80 + u * .6, .05), (q * 1.66, -1.08, .08), (q * 1.58, -.82 + u * .6, .13), (q * 1.52, -1.00, .14), (q * 1.47, -.66 + u * .7, .16), (q * 1.36, -.44 + u * .8, .14), (q * 1.18, -.30 + u * .5, .10)], True, corners=(0, 4, 5, 6)), m)
            # collar-to-chest plane
            c.fill(s.S([(q * 1.00, -.20, .10), (q * .97, -.40, .24), (q * .80, -.70, .46), (q * .54, -.90, .58), (q * .70, -.66, .50), (q * .84, -.42, .36), (q * .90, -.22, .18)], True, corners=(0, 3)), m)
            # chest folds pulled from the armpit
            c.fill(s.S([(q * 1.02, -1.02, .22), (q * .80, -1.08, .48), (q * .46, -1.16, .62), (q * .74, -1.22, .50), (q * 1.00, -1.18, .22)], True, corners=(0, 2, 4)), m2)
            c.fill(s.S([(q * .98, -1.66, .24), (q * .76, -1.86, .46), (q * .44, -1.96, .58), (q * .78, -2.02, .44), (q * .98, -1.86, .22)], True, corners=(0, 2, 4)), m2)
            # sleeve folds
            c.fill(s.S([(q * 1.72, -1.30, .0), (q * 1.58, -1.50, .12), (q * 1.28, -1.64, .18), (q * 1.52, -1.64, .12), (q * 1.70, -1.52, .02)], True, corners=(0, 2, 4)), m2)
            c.fill(s.S([(q * 1.72, -1.96, .02), (q * 1.56, -2.2, .12), (q * 1.30, -2.34, .16), (q * 1.54, -2.38, .12), (q * 1.72, -2.2, .02)], True, corners=(0, 2, 4)), m2)
            # armpit: black shapes that pull out of the joint, different on each side
            j = np.random.default_rng(31 + q).normal(0, .03, (24, 2)); J = lambda P, k: [(x + j[(i + k) % 24, 0], y + j[(i + k) % 24, 1], z) for i, (x, y, z) in enumerate(P)]
            c.fill(s.S(J([(q * 1.03, -.64, .22), (q * .90, -.84, .38), (q * .62, -.98, .55), (q * .90, -.97, .34), (q * 1.02, -1.10, .22), (q * 1.00, -1.50, .20), (q * 1.06, -1.86, .16), (q * 1.12, -1.40, .14), (q * 1.14, -1.06, .16), (q * 1.36, -.96, .16), (q * 1.15, -.82, .18)], 0), True, corners=(0, 2, 4, 6, 8, 9)), bk)
            if q == sg: c.fill(s.S(J([(q * 1.03, -1.34, .20), (q * .84, -1.48, .40), (q * .60, -1.58, .52), (q * .86, -1.60, .36), (q * 1.02, -1.58, .20)], 5), True, corners=(0, 2, 4)), bk)
            else: c.fill(s.S(J([(q * 1.10, -1.30, .16), (q * 1.28, -1.42, .16), (q * 1.52, -1.46, .1), (q * 1.30, -1.54, .14), (q * 1.12, -1.52, .16)], 9), True, corners=(0, 2, 4)), bk)
            c.fill(s.S([(q * 1.80, -1.18, -.06), (q * 1.60, -1.30, .08), (q * 1.36, -1.46, .15), (q * 1.60, -1.42, .08), (q * 1.80, -1.36, -.06)], True, corners=(0, 2, 4)), bk)
            c.fill(s.S([(q * 1.80, -1.80, -.02), (q * 1.58, -1.88, .08), (q * 1.36, -2.06, .15), (q * 1.60, -2.02, .08), (q * 1.80, -1.94, -.02)], True, corners=(0, 2, 4)), bk)
        s.wear('jacket')
        # cowl's cast shadow
        c.fill(s.S([(-1.10, .02, -.1), (-.98, -.36, .30), (-.60, -.80, .56), (.02, -1.00, .68), (.60, -.80, .56), (.98, -.36, .30), (1.10, .02, -.1), (.6, .3, 0), (-.6, .3, 0)], True), bk)
        c.unclip()
        il = s.L.get('clothline', s.ink)
        s.c.line(Ol, s.w * .95, s.ink, taper=(.05, .02), press=.25, seed=s.seed + 31); s.c.line(Or, s.w * .95, s.ink, taper=(.05, .02), press=.25, seed=s.seed + 32)
        for q in (-1, 1):
            s.ln(s.P([(q * 1.00, -.60, .24), (q * 1.06, -.84, .22), (q * 1.08, -1.2, .20), (q * 1.06, -1.8, .18), (q * 1.08, -2.8, .16)]), .62, seed=36 + q, taper=(.25, .05))
            for k, (P, w) in enumerate([([(q * 1.00, -.58, .26), (q * 1.14, -.72, .24), (q * 1.36, -.80, .20)], .50), ([(q * 1.03, -.70, .24), (q * .86, -.90, .40), (q * .60, -1.02, .55)], .52), ([(q * 1.05, -.84, .22), (q * 1.20, -1.02, .20), (q * 1.40, -1.14, .16)], .42),
                                        ([(q * 1.04, -1.02, .22), (q * .86, -1.34, .40), (q * .62, -1.52, .5)], .42), ([(q * 1.80, -1.20, -.06), (q * 1.60, -1.32, .08), (q * 1.34, -1.48, .15)], .50), ([(q * 1.76, -1.60, 0), (q * 1.58, -1.66, .08), (q * 1.32, -1.80, .14)], .42),
                                        ([(q * 1.30, -.19 + u * .8, -.14), (q * 1.29, -.36 + u * .7, .08), (q * 1.36, -.58 + u * .6, .16)], .40), ([(q * 1.42, -.25 + u * .8, -.10), (q * 1.45, -.38 + u * .7, .06)], .34), ([(q * 1.06, -.10 + u * .4, -.1), (q * 1.04, -.28, .14), (q * .97, -.44, .26)], .40)]):
                s.ln(s.P(P), w, seed=38 + k + 10 * q, taper=(.1, .65))

        # front flap: seam from under the cowl to the button and down
        s.ln(s.P([(-.62, -.72, .55), (-.28, -.90, .66), (.06, -.98, .70), (.30, -1.00, .68)]), .44, seed=51, taper=(.4, .1))
        s.ln(s.P([(.30, -1.00, .68), (.36, -1.6, .66), (.34, -2.6, .62)]), .50, seed=52, taper=(.1, .1))
        s.ln(s.P([(.40, -1.02, .66), (.46, -1.6, .64), (.44, -2.6, .60)]), .26, seed=53, taper=(.2, .1))
        # gold spiral button
        g0, g1, g2 = L['gold']; a = np.linspace(0, 2 * np.pi, 21)[:-1]; r = .085
        Q = lambda R, dx=0, dy=0: s.P(np.c_[.33 + dx + np.cos(a) * R, -1.00 + dy + np.sin(a) * R, np.full(20, .70)])
        c.fill(Q(r * 1.12), s.ink); c.fill(Q(r), g0); c.fill(Q(r * .86, -L['side'] * .018, -.014), g1); c.fill(Q(r * .66, L['side'] * .004, .006), g0)
        sp = np.linspace(0, 1, 40); th = sp * 4.2 * math.pi; rr = r * .72 * (1 - sp * .9)
        s.c.line(s.P(np.c_[.33 + np.cos(th) * rr, -1.0 + np.sin(th) * rr, np.full(40, .705)]), max(.8, s.w * .26), g1, taper=(.1, .4), seed=3)
        c.fill(Q(r * .18, L['side'] * .036, .036), g2)

    def cowl(s):
        c, L = s.c, s.L; b, sh, hi = L['hood']; sg = L['side']; dp = L.get('hooddeep', mixc(sh, s.ink, .45))
        lo = [(-1.03, .36, -.38), (-1.05, .16, -.20), (-.99, -.12, .10), (-.83, -.42, .40), (-.50, -.70, .62), (-.12, -.84, .72), (.10, -.85, .72), (.48, -.72, .62), (.81, -.44, .40), (.99, -.14, .10), (1.05, .14, -.20), (1.03, .34, -.38)]
        hi_ = [(.82, .50, -.40), (.78, .30, -.12), (.66, .10, .22), (.46, -.10, .50), (.22, -.22, .62), (-.02, -.26, .65), (-.26, -.21, .62), (-.48, -.08, .50), (-.66, .11, .22), (-.78, .30, -.12), (-.82, .50, -.40)]
        Pl, Ph = s.S(lo), s.S(hi_); poly = np.vstack([Pl, Ph]); c.fill(poly, b); c.clip([poly])
        # under-fold shadows: three wraps crossing toward the front
        c.fill(s.S([(-1.02, .08, -.2), (-.86, -.10, .22), (-.60, -.30, .50), (-.20, -.46, .68), (.10, -.46, .70), (-.16, -.58, .70), (-.52, -.46, .58), (-.80, -.26, .32), (-.99, -.04, 0.)], True, corners=(0, 4)), sh)
        c.fill(s.S([(1.02, .10, -.2), (.88, -.12, .22), (.66, -.36, .48), (.30, -.58, .68), (-.06, -.66, .72), (.26, -.70, .68), (.60, -.56, .52), (.86, -.30, .28), (1.0, -.06, 0.)], True, corners=(0, 4)), sh)
        c.fill(s.S([(-.66, .30, -.1), (-.60, .06, .26), (-.42, -.14, .52), (-.14, -.26, .64), (.18, -.26, .64), (.42, -.14, .52), (.62, .06, .26), (.68, .30, -.1), (.54, -.08, .42), (.24, -.36, .66), (-.04, -.40, .68), (-.30, -.34, .64), (-.54, -.12, .42)], True, corners=(0, 7)), sh)
        c.fill(s.S([(-.97, -.14, .12), (-.82, -.42, .40), (-.50, -.70, .62), (-.12, -.84, .72), (.10, -.85, .72), (.48, -.72, .62), (.24, -.76, .70), (-.10, -.75, .72), (-.44, -.62, .63), (-.74, -.38, .44)], True, corners=(0, 5)), sh)
        q = sg
        c.fill(s.S([(q * .92, .02, .02), (q * .84, -.16, .28), (q * .66, -.34, .48), (q * .74, -.16, .36), (q * .82, .02, .12)], True, corners=(0, 2)), hi)
        c.fill(s.S([(q * .50, -.16, .50), (q * .30, -.30, .64), (q * .06, -.36, .68), (q * .28, -.26, .64)], True, corners=(0, 2)), hi)
        s.wear('hood')
        c.unclip()
        s.c.line(Pl, s.w * .95, s.ink, taper=(.05, .05), press=.28, seed=s.seed + 71); s.c.line(Ph, s.w * .75, s.ink, taper=(.1, .1), press=.28, seed=s.seed + 72)
        c.clip([poly])
        for k, (P, w) in enumerate([([(-.98, .06, -.1), (-.84, -.12, .24), (-.58, -.32, .50), (-.20, -.46, .68), (.08, -.46, .70)], .62), ([(1.0, .08, -.1), (.86, -.14, .24), (.64, -.38, .48), (.30, -.59, .68), (.0, -.66, .72)], .62),
                                    ([(-.62, .12, .22), (-.50, -.12, .44), (-.28, -.33, .62), (-.06, -.39, .68)], .44), ([(.64, .14, .22), (.52, -.10, .42), (.30, -.31, .62)], .44), ([(-.86, -.30, .34), (-.60, -.52, .55), (-.30, -.66, .68)], .40),
                                    ([(-.50, -.70, .62), (-.42, -.60, .63), (-.30, -.54, .66)], .36), ([(.48, -.72, .62), (.44, -.64, .62), (.34, -.58, .66)], .36), ([(.80, -.06, .2), (.70, -.16, .36), (.58, -.22, .46)], .32), ([(.12, -.85, .72), (.08, -.76, .73), (.10, -.70, .72)], .34)]):
            s.ln(s.P(P), w, seed=73 + k, taper=(.12, .55))
        c.unclip()

    def wear(s, where='jacket'):
        """second half of the film: rips, scorch, dust (level s.dmg)"""
        c, L, d = s.c, s.L, s.dmg; b, hi, bk = L['cloth']; r = np.random.default_rng(77)
        if d < .05: return
        def frag(x, y, z, w, h, a, m, seed):                    # ragged angular patch
            q = np.random.default_rng(seed); th = np.linspace(0, 2 * np.pi, m, endpoint=False) + q.normal(0, .12, m); rr = q.uniform(.55, 1.2, m); ca, sa = math.cos(a), math.sin(a)
            px, py = np.cos(th) * w * rr, np.sin(th) * h * rr; return s.P(np.c_[x + px * ca - py * sa, y + px * sa + py * ca, np.full(m, z)])
        if where == 'jacket':
            for k, (x, y, z, l, h, a) in enumerate(((-1.38, -.50, .12, .26, .052, -.95), (.80, -.80, .46, .20, .040, .55), (1.54, -1.16, .1, .24, .058, 1.25), (-.86, -1.46, .40, .28, .045, -.25), (1.36, -.40, .12, .13, .03, -.5))):
                if k / 5 > d: break
                m = 9; u = np.linspace(-1, 1, m); q = np.random.default_rng(60 + k); ca, sa = math.cos(a), math.sin(a)
                up, dn = h * (1 - u ** 2) * q.uniform(.55, 1.25, m), -h * (1 - u ** 2) * q.uniform(.45, 1.1, m)
                R2 = lambda X, Y: s.P(np.c_[x + X * ca - Y * sa, y + X * sa + Y * ca, np.full(len(X), z)])
                Pu, Pd = R2(u * l, up), R2(u * l, dn); P = np.vstack([Pu, Pd[::-1]]); c.fill(P, mixc(bk, '#000000', .4))
                c.fill(np.vstack([Pd, R2(u * l, dn - h * .42 * (1 - u ** 2))[::-1]]), mixc(b, hi, .75))          # turned-out lower edge catches light
                s.ln(Pu, .34, seed=k, taper=(.2, .2)); s.ln(R2(u * l, dn - h * .42 * (1 - u ** 2)), .24, seed=k + 5, taper=(.3, .3))
                for jx in (-.55, -.2, .12, .5):
                    i = int((jx + 1) / 2 * (m - 1)); s.ln([Pu[i], (Pu[i] + Pd[i]) / 2 + A([.012, 0]) * s.u, Pd[i] + q.normal(0, .008, 2) * s.u], .16, col=mixc(hi, b, .25), seed=i, taper=(.1, .2))
                for jx in (-1, 1):
                    e = R2(A([jx * l]), A([0.]))[0]; e2 = R2(A([jx * l * 1.25]), A([jx * .02]))[0]; s.ln([e, (e + e2) / 2 + q.normal(0, .01, 2) * s.u, e2], .20, col=mixc(hi, b, .4), seed=7, taper=(.1, .7))
            dust = mixc(L['skin'][0], b, .55)
            for k in range(int(26 * d)):
                q = r.choice([-1, 1]); x, y = q * r.uniform(.3, 1.6), -r.uniform(.3, 2.2); z = .5 - .3 * abs(x); P = frag(x, y, z, r.uniform(.015, .07), r.uniform(.01, .03), r.uniform(-.8, .8), 7, 500 + k); c.fill(P, dust, .30)
            for k in range(int(34 * d)):
                q = r.choice([-1, 1]); x, y = q * r.uniform(.3, 1.6), -r.uniform(.25, 2.2); z = .5 - .3 * abs(x); l = r.uniform(.02, .08); a = -.6 * q + r.normal(0, .25)
                s.ln(s.P([(x - l * math.cos(a), y - l * math.sin(a), z), (x, y, z), (x + l * math.cos(a), y + l * math.sin(a), z)]), .22, col=dust, seed=k, taper=(.4, .6), alpha=.55)
        else:                                                      # hood: scorched edge and a torn notch
            sc = mixc(L['hood'][1], '#120a0c', .78)
            for x, y, z, w, h, a in ((.86, -.20, .20, .17, .07, 1.0), (-.46, -.66, .60, .19, .06, -.4), (-.98, .06, .0, .09, .13, .2)):
                P = frag(x, y, z, w, h, a, 13, int(abs(x) * 90)); c.fill(P, sc, .92); g = P.mean(0); c.fill(g + (P - g) * 1.28, sc, .30)
            for k in range(int(16 * d)):
                th = r.uniform(-2.2, 2.2); x, y, z = .92 * math.sin(th), -.45 * math.cos(th) - .1, .6 * math.cos(th); P = frag(x, y, z, r.uniform(.01, .03), r.uniform(.008, .02), r.uniform(0, 3), 6, 900 + k); c.fill(P, sc, .7)

    def draw_back(s): s.hood_back()

    def draw_front(s): s.jacket()


def _bust(H, expr='neutral', light='day', f=0, wind=0., seed=1, base=None, byaw=None, bpitch=0., broll=0., damage=0., wet=0., raise_=0., breath=0., cel=None, rims=True, head=True, hood=True, w=W, h=HH, **over):
    X = getx(expr, **over); L = LIGHT[light] if isinstance(light, str) else light
    hd = YHead.of(H, X['jaw'], X['jx'], X['slim'])
    c1 = Cel(w, h); bd = _Body(c1, hd, L, seed, base, byaw, bpitch, broll, damage, wet, raise_, breath, f, wind); bd.X = X
    if hood: bd.draw_back()
    bd.draw_front(); bd.neck()
    if head: Yuji.head(c1, hd, X, L, f, wind, seed, wet, damage, front=False, nape=-.55 + max(0., hd.pitch) * max(0., math.cos(hd.yaw)) * 1.3)
    bd.cowl()
    a = c1.layer(h, w)
    if head:
        c2 = Cel(w, h); Yuji.head(c2, hd, X, L, f, wind, seed, wet, damage, back=False); b = c2.layer(h, w)
    else: b = np.zeros_like(a)
    for R in (L['rim'] or []) if rims else []:
        k = H.s / 330.; kw = dict(amt=R.get('amt', 1.), inset=lw(H.s) * .62, over=R.get('over', False))
        if R.get('head', True): b = rim_in(b, R['d'][0] * k, R['d'][1] * k, R['col'], close=min(16, int(.018 * H.s)), **kw)
        a = rim_in(a, R['d'][0] * k, R['d'][1] * k, R['col'], **kw)
    out = a; bb = _bbox(b[..., 3], 1, h, w)
    if bb is not None: y0, y1, x0, x1 = bb; B_ = b[y0:y1, x0:x1]; out[y0:y1, x0:x1] = B_ + a[y0:y1, x0:x1] * (1 - B_[..., 3:4])
    for p in L.get('para', []):
        cx, cy = H.cx, H.cy; out = para_fast(out, (cx + p['p0'][0] * H.s, cy + p['p0'][1] * H.s), (cx + p['p1'][0] * H.s, cy + p['p1'][1] * H.s), p['stops'], p.get('mode', 'mul'), p.get('amt', 1.))
    return out


Yuji.bust = staticmethod(_bust)
