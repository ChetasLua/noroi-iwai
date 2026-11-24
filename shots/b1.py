"""B1  In a patch of sun, someone is asleep.  72 f."""
import numpy as np, cv2, math
from jj import comp, kit, W, H
from jj.paint import rgb, f32, smooth
from jj.grass import Hill, Foreground, motes, PW, PH, Q, GW, GH
from chars import chibi as C

NX, NY, NS, NR = 1450., 742., 122., .58                     # Gojo's neck in the painting, px per unit, roll
BLAND, POP = 45, 54
BUB = np.array([-1.46, -.74])                               # bubble centre from the nostril, in radii


def e(t): t = min(1., max(0., t)); return t * t * (3 - 2 * t)
def out(t, p=3): return 1 - (1 - min(1., max(0., t))) ** p


def hill(season='summer', **k):
    """the hill of b1 (g3 rhymes with it)"""
    kw = dict(seed=11, crest=[(-60, 540), (360, 430), (800, 322), (1240, 316), (1740, 405), (2300, 560)], crest2=[(-60, 248), (420, 292), (900, 212), (1500, 232), (2300, 168)],
              blobs=[(1500, 780, 640, 330, -10, .085, .8), (300, 1060, 640, 300, 18, -.17, .6), (760, 640, 560, 120, -9, -.07, .7), (1960, 1080, 420, 200, -20, -.12, .6)],
              quiet=[(1440, 730, 330)],
              trees=[(262, 300, 30), (318, 282, 44), (388, 296, 36), (452, 300, 24), (214, 306, 20), (1672, 318, 30), (1728, 300, 46), (1800, 322, 36), (1866, 348, 24), (1034, 196, 17), (1076, 190, 22), (1120, 197, 16), (604, 204, 13), (640, 200, 17), (1560, 214, 15), (1598, 208, 19), (2040, 168, 20), (2088, 172, 15)],
              far_blobs=[(760, 285, 420, 40, -4, .10, .6), (1500, 380, 420, 30, 3, -.05, .5), (300, 170, 420, 36, 5, .07, .6), (1850, 120, 420, 30, -5, -.07, .6), (1150, 130, 380, 24, 2, .06, .6)],
              strips=[([(1240, 150), (1420, 138), (1640, 150), (1900, 130), (2260, 136)], 13, -.22), ([(1200, 172), (1420, 158), (1660, 170), (1920, 148), (2260, 154)], 11, .20), ([(-40, 196), (200, 176), (470, 196), (760, 168)], 11, -.20), ([(-40, 214), (220, 196), (500, 214), (720, 192)], 10, .18),
                      ([(520, 318), (700, 286), (900, 268), (1120, 262)], 10, .16), ([(1180, 268), (1400, 280), (1580, 298)], 12, -.16)],
              carpets=[(560, 560, 260, 60, -12, 50, 'daisy'), (2000, 620, 200, 60, 18, 30, 'clover'), (260, 760, 160, 40, 0, 30, 'butter')],
              drifts=[(620, 520, 150, 50, 10, 'daisy'), (420, 640, 120, 40, 7, 'butter'), (980, 980, 190, 60, 14, 'daisy'), (1800, 560, 110, 40, 7, 'clover'), (1120, 470, 90, 30, 7, 'butter'), (2010, 900, 90, 30, 6, 'daisy'), (720, 860, 120, 40, 6, 'clover')],
              cloud_seed=3, lam=1080., period=18., weak=.45, wave=1.55, wind=14., gust_ref=(4, 430, 700), sunny=[(1470, 720, 470)])
    kw.update(k); return Hill(season, **kw)


def add_path(G):
    G.add_path([(2260, 250), (1990, 236), (1850, 262), (1640, 262), (1470, 292), (1300, 300), (1180, 278), (1040, 246), (960, 236)], 9., 3.5, col='#f4eecb', amt=.85)
    G.add_path([(-40, 330), (150, 318), (300, 346), (470, 336), (640, 290), (800, 250), (940, 236)], 4., 6.5, col='#f4eecb', amt=.75)


def torii(im, M, x=948., y=236., h=44., tint=('#e0604a', '#a9564f'), haze=.34, air='#bfe3d6'):
    """a small red gate where the two paths meet, seen through the air"""
    px, py, s, _ = C.through(M, x, y, h); d = C.D(int(px - s * 1.2), int(py - s * 1.5), int(s * 2.4), int(s * 2), 2); u = lambda P: np.array(P, float) * s + [px, py]
    a, b = C._mix(tint[0], air, haze), C._mix(tint[1], air, haze); sh = C._mix('#3a7a72', air, haze * .6)
    d.detail(u([(-.36, .0), (.40, .04), (.86, .34), (.10, .30)]), sh)
    for sx in (-.34, .34): d.detail(u([(sx - .05, .02), (sx - .04, -.80), (sx + .05, -.80), (sx + .07, .02)]), a); d.detail(u([(sx + .02, .02), (sx + .02, -.80), (sx + .05, -.80), (sx + .07, .02)]), b)
    d.detail(u([(-.46, -.62), (.46, -.62), (.46, -.55), (-.46, -.55)]), b); d.detail(u([(-.60, -.94), (-.2, -.885), (.2, -.885), (.60, -.94), (.56, -.80), (.2, -.775), (-.2, -.775), (-.56, -.80)]), a)
    d.detail(u([(-.62, -.98), (-.2, -.925), (.2, -.925), (.62, -.98), (.60, -.93), (.2, -.875), (-.2, -.875), (-.60, -.93)]), C._mix('#3c4a55', air, haze)); d.detail(u([(-.05, -.78), (.05, -.78), (.05, -.60), (-.05, -.60)]), b)
    return C.stamp(im, d, outer=None)


class Shot:
    id = 'b1'; n = 72
    finish = dict(bloom=.22, vig=.12, grade=('#12343a', (1.02, 1.0, .96), '#fff3cf', .25))
    EVENTS = [(0, 'whiteout_settle', dict(dur=12, note='arrives out of a white-out, settles by 12')),
              (0, 'wind_bed', dict(dur=72, size=.35, note='summer hill air, grass, far cicada')),
              (0, 'gust', dict(size=1.0, dur=28, pan=(-1, 1), note='big grass wave enters left on the downbeat, mid frame 14, lays over Gojo at 18, leaves right at 28')),
              (18, 'gust', dict(size=.4, dur=28, pan=(-1, 1), note='small wave, passes Gojo at 36')),
              (36, 'gust', dict(size=1.0, dur=28, pan=(-1, 1), note='big wave on the second downbeat, mid frame 50, over Gojo at 54 with the pop, leaves 64')),
              (54, 'gust', dict(size=.45, dur=18, pan=(-1, .3), note='small wave')),
              *[(9 * k, 'bubble_swell', dict(size=.5 + .1 * k, pan=.3, note='sleep bubble swells on the beat, pitch rises with size')) for k in range(5)],
              (0, 'snore', dict(dur=18, pan=.3, note='tiny breath in')), (18, 'snore', dict(dur=18, pan=.3, note='breath out')), (36, 'snore', dict(dur=9, pan=.3, note='breath in')),
              (4, 'butterfly_flutter', dict(dur=40, size=.2, pan=(-.8, .25), note='white butterfly crosses from the left, very soft')),
              (BLAND, 'butterfly_land', dict(size=.3, pan=.25, note='lands on the bubble: rubbery creak')), (49, 'bubble_creak', dict(size=.3, pan=.25, note='bubble sags under it')),
              (POP, 'bubble_pop', dict(size=1.0, pan=.28, note='pop')), (POP, 'butterfly_startle', dict(size=.5, dur=16, pan=(.25, -.3), note='flies up and away')),
              (55, 'twitch', dict(size=.4, pan=.35, note='body jolt, cloth and grass')), (57, 'lip_smack', dict(size=.4, pan=.35)), (59, 'arm_lift', dict(size=.3, pan=.4, note='sleeve, grass springs back')),
              (61, 'scratch', dict(size=.6, pan=.38, note='belly scratch stroke 1')), (63, 'scratch', dict(size=.6, pan=.38)), (65, 'scratch', dict(size=.6, pan=.38, note='stroke 3')),
              (62, 'lip_smack', dict(size=.3, pan=.35)), (66, 'lip_smack', dict(size=.3, pan=.35)), (68, 'sigh', dict(size=.4, dur=4, pan=.35, note='sinks back, still asleep'))]

    # -------------------------------------------------- setup
    def setup(self):
        self.G = hill(); self.G.setup(); add_path(self.G)
        body = self._cel(0, None, seat='body'); arm = self._cel(0, None, seat='arm'); lay = np.clip(arm - body, 0, 1)
        lay = cv2.morphologyEx(lay, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        self.seat = self.G.seat([body, lay], sun=(15, 15))
        self.fg = Foreground([(40, 1190, 360, 50, .12, None), (360, 1200, 230, 30, .3, 'daisy'), (1880, 1190, 300, 34, -.3, 'butter')], blur_px=7)

    def cam(self, f):
        t = e(f / 80.); return (1120 + 34 * t, 630 + 14 * t, 1.0 + .085 * t, -.004 + .010 * t)

    def gust_at(self, f, x, y):
        g, _, _ = self._fld(f); return float(g[int(np.clip(y / Q, 0, GH - 1)), int(np.clip(x / Q, 0, GW - 1))])

    def _fld(self, f):
        return self.G.fields(f)

    # -------------------------------------------------- acting
    def pose(self, f):
        """everything he does at frame f"""
        k = dict(t=f, breath=math.sin(2 * math.pi * f / 36. - 1.2), mouth=None, hroll=0., hx=0., hy=0., pitch=-.16, lift=1.)
        A = [(.44, .20), (.92, .34), (1.34, .52)]; B = [(.46, .20), (.92, .52), (.38, .80)]
        k['aR'] = A; k['over'] = ''; k['hk'] = ('open', 'open'); k['pose'] = 'starfish'
        # the jolt: anticipation 2 f, stretch, settle
        j = f - POP
        if j >= 0:
            k['hy'] = [.0, -.07, -.10, -.05, .015, .02, .01][j] if j < 7 else .0
            k['hroll'] = {0: 0., 1: -.06, 2: -.09, 3: -.07}.get(j, -.05 + .05 * min(1, (j - 3) / 4)) if j < 8 else .0
            k['breath'] = 1.4 if j in (1, 2) else k['breath']
        if f >= 55: k['mouth'] = {55: 'gape', 56: 'gape', 57: 'wave', 58: 'wave', 59: 'flat', 60: 'cat', 61: 'cat', 62: 'wave', 63: 'wave', 64: 'flat', 65: 'flat', 66: 'cat', 67: 'cat', 68: 'smile', 69: 'smile'}.get(f, None)
        # arm: rest -> up off the grass -> belly, scratch, stay
        if f >= 58:
            L1 = [(.45, .20), (1.02, .30), (1.26, .06)]; L2 = [(.46, .20), (.92, .40), (1.02, .80)]
            k['over'] = 'R'; k['pose'] = 'scratch'; k['hk'] = ('open', 'flat'); k['scratch'] = 0.
            if f == 58: k['aR'] = L1; k['over'] = ''; k['hk'] = ('open', 'open'); k['pose'] = 'starfish'; k['lift'] = .45
            elif f == 59: k['aR'] = L2; k['lift'] = .0; k['hk'] = ('open', 'open')
            else:
                k['lift'] = 0.; k['aR'] = B; k['scratch'] = {60: -.3, 61: 1., 62: .2, 63: -1., 64: .1, 65: 1., 66: .0, 67: -.5}.get(f, 0.)
                if f >= 68: k['aR'] = [(.46, .20), (.92, .55), (.40, .84)]
        # sinks back: head rolls a little to the other side, deep breath out
        if f >= 66: q = e((f - 66) / 6.); k['hroll'] += .10 * q; k['hy'] += .03 * q; k['breath'] = -.9 * q + (1 - q) * .5
        return k

    def bubble_r(self, f):
        """swell on the beat (4 f up with overshoot), eases part of the way back, grows over the shot"""
        if f >= POP: return 14 * e((f - 69) / 3.) if f >= 69 else 0.
        b = f // 9; u = (f % 9); base = [26, 36, 46, 57, 68, 72][min(b, 5)]; prev = [12, 20, 29, 39, 50, 66][min(b, 5)]
        up = out(u / 4., 3) * (1 + .10 * math.sin(math.pi * min(1, u / 5.))); dn = e((u - 4) / 5.) * .22
        return prev + (base - prev) * up - (base - prev) * dn * 1.0 if b < 5 else 70 + 2.2 * math.sin(f * .9)

    def butterfly(self, f):
        """painting position, heading, flap override, size, alive"""
        if f < 4: return None
        nose = self._nose0; land = nose + BUB * 71 + np.array([-8., -71. + 3.])
        keys = [(4, (330, 610)), (11, (520, 470)), (18, (760, 560)), (25, (930, 440)), (32, (1090, 540)), (39, (1200, 415)), (45, tuple(land))]
        if f <= BLAND:
            from jj.ink import spl
            ts = np.array([k for k, _ in keys], float); P = np.array([p for _, p in keys], float); x = np.interp(f, ts, np.arange(len(ts)))
            S = spl(P, n=24); i = min(len(S) - 1, int(round(x * 24))); p = S[i] + [6 * math.sin(f * .9), 9 * math.sin(f * 1.35)] if f < BLAND - 2 else S[i]
            d = S[min(len(S) - 1, i + 6)] - S[max(0, i - 6)]; return p, math.atan2(d[1], d[0]) * .6, None, 1.0
        if f < POP:
            sag = 5 * e((f - 45) / 4.) ; return land + [0, sag + 1.5 * math.sin(f * .9)], .15, .15 + .55 * math.sin((f - 45) * .62), 1.0
        u = (f - POP) / 17.; p = land + np.array([-20 * u - 250 * u * u + 18 * math.sin(f * 1.1), -30 * out(u * 3) - 240 * u ** 1.25 + 14 * math.sin(f * 1.7)]); return p, -2.3, None, 1.0

    def _cel(self, f, M, seat=None):
        k = self.pose(f); M = Hill.matrix((PW / 2, PH / 2, 1., 0.)) if M is None else M
        x, y, S, rot = C.through(M, NX, NY, NS, NR)
        if seat:                                                       # masks for the bed, in the painting
            d = C.D(0, 0, PW, PH, 1); kw = dict(aR=[(.44, .2), (.40, .6), (.30, .8)]) if seat == 'body' else {}
            C.put(d, 'gojo', 'starfish', NX, NY, NS, rot=NR, t=0, breath=0., seed=1, **kw, over='R' if seat == 'body' else ''); return d.rgba(outer=None)[..., 3]
        fs = f - f % 2                                                 # hair on twos, body on ones
        g = self.gust_at(f, NX + 120, NY - 120)
        d = C.D(int(x - 400), int(y - 420), 860, 840, 2)
        r = C.put(d, 'gojo', k['pose'], x, y, S, rot=rot, t=fs, wind=(1, .3), gust=.03 + .15 * g, wamp=.035 + .05 * g, seed=1, breath=k['breath'], mouth=k['mouth'], hroll=k['hroll'], hx=k['hx'], hy=k['hy'], pitch=k['pitch'],
                  aR=k['aR'], over=k['over'], hk=k['hk'], **({'scratch': k['scratch']} if 'scratch' in k else {}))
        return d.layer(), r, k

    # -------------------------------------------------- frame
    def frame(self, f):
        if not hasattr(self, '_nose0'):
            d = C.D(0, 0, 8, 8, 1); r = C.put(d, 'gojo', 'starfish', NX, NY, NS, rot=NR, t=0, breath=0.); self._nose0 = r['head'](np.array([(.02, .40)]), 'd')[0]
        M = Hill.matrix(self.cam(f)); z = math.hypot(M[0, 0], M[1, 0])
        L, r, k = self._cel(f, M); L = C.daylight(L, 300, 950)
        w = [1., {58: .45, 59: .0}.get(f, 1. if f < 58 else max(.0, .12 - .02 * (f - 60)))]
        im = self.G.frame(f, layer=L, cam=self.cam(f), seat=self.seat, w=w)
        im = torii(im, M)
        # bubble
        nose = r['head'](np.array([(.02, .40)]), 'd')[0]; R = self.bubble_r(f) * z
        if f < POP and R > 2:
            sq = .07 * e((f - 45) / 3.) * (1 + .5 * math.sin((f - 45) * 1.3)) if f >= 45 else .04 * math.sin((f % 9) * 1.4) * math.exp(-(f % 9) / 3.)
            c = nose + BUB * R; im = C.bubble(im, c[0], c[1] + (4 * z if f >= 46 else 0), R, f, nose[0], nose[1], squash=sq)
        elif POP <= f < POP + 5:
            R = 71 * z; c = nose + BUB * R; im = C.bubble(im, c[0], c[1], R, f, pop=float(f - POP))
        elif R > 2: c = nose + BUB * R; im = C.bubble(im, c[0], c[1], R, f, nose[0], nose[1])
        # butterfly and its shadow on the grass
        b = self.butterfly(f)
        if b is not None:
            p, hd, fl, _ = b; x, y, S, _ = C.through(M, p[0], p[1], 40.)
            if -80 < x < W + 80 and -80 < y < H + 80:
                d = C.D(int(x - 110), int(y - 110), 220, 220, 2); C.butterfly(d, x, y, S, f, head=hd, flap=fl, speed=1.0 if f < POP else 1.45); l = d.layer(outer=None)
                if f < BLAND - 3 or f > POP + 2: im = kit.cast(im, l, 44 * z, 92 * z, '#1d5a56', .22, 5)
                elif f <= BLAND: h = (BLAND - f) / 3.; im = kit.cast(im, l, 44 * z * h + 6, 92 * z * h + 8, '#1d5a56', .20, 4)
                im = C.stamp(im, d, outer=None)
        # letters of sleep
        a = e((f - 6) / 8.) * (1 - e((f - 52) / 3.))
        if a > .02:
            x, y, _, _ = C.through(M, 1724, 486, 1.); d = C.D(int(x - 120), int(y - 320), 420, 400, 2); C.zzz(d, x, y, 44 * z, f - f % 2); im = C.stamp(im, d, a, outer=None)
        cx, cy, _, _ = self.cam(f)
        im = comp.over(im, self.fg.layer(f, gust=.25 + .5 * self.gust_at(f, cx - 500, cy + 420), z=1 + (z - 1) * 1.7, dx=-(cx - 1120) * 1.5, dy=-(cy - 630) * 1.5))
        im = motes(im, f, n=40)
        if f < 14:                                                     # out of the white
            q = [1.0, .96, .86, .72, .56, .42, .31, .22, .155, .105, .068, .04, .02, .008][f]
            im = im * (1 + 1.1 * q) + q * np.array([1.0, .99, .95], f32) * .95
        return im
