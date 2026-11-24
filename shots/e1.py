"""E1  The box.  His face, almost smiling; the living cube shuts round him eye by eye; it falls in an empty hall.  72 f."""
import numpy as np, cv2, math, os
from jj import comp, kit, W, H
from jj.paint import rgb, f32, smooth, fbm, grid, mix
from jj.ink import Cel, spl, resample, noise1
from jj.kit import outline, hatch
from jj import world
from jj.realm import Face, warp_face, Cube, FACES, quat, qmul, qmat
from chars.gojo import Gojo, gojo_head, PAL
from chars import gojo_coat as GC

D = math.radians
CX, CY, S0 = 966., 520., 352.                                                  # head: centre at eye level, px per unit
A_, DD = 1.52, 7.                                                             # half box (head units), camera distance
FF = S0 * DD
SHUT = dict(L=28, R=31, T=34, B=37)
LAND1, LAND2, REST = 54., 63., 67.4
TUBE = {50: .25, 55: .25, 59: .25, 60: 0., 61: .25, 65: .25}
def ease(t): t = min(1., max(0., t)); return t * t * (3 - 2 * t)


def prj(P):
    P = np.asarray(P, float); k = FF / (DD - P[..., 2]); return np.stack([CX + P[..., 0] * k, CY + P[..., 1] * k], -1)


class Shot:
    id = 'e1'; n = 72
    finish = dict(bloom=.22, vig=.26, soft=.10, grain=.014, halation='#d4f0e4', sat=.92)
    EVENTS = [(0, 'void_room', dict(size=.3, note='inside the living cube: wet stone, no air')), (6, 'flesh_creak', dict(size=.3, note='sinew tightening')), (11, 'eyes_open', dict(size=.35, note='he lifts his eyes to us, almost smiling')), (21, 'slow_blink', dict(size=.15, note='one slow blink, shut 23-25')),
              (28, 'box_close', dict(v=0, pan=-.5, size=.8, note='left side shuts, its eye opens')), (31, 'box_close', dict(v=1, pan=.5, size=.85, note='right side')), (34, 'box_close', dict(v=2, pan=0., size=.9, note='top')), (37, 'box_close', dict(v=3, pan=0., size=.95, note='bottom')), (38, 'box_lid', dict(size=.9, dur=7, note='front face rises, its eye opening')),
              (45, 'box_shut', dict(size=1., note='shut; one black frame')), (46, 'fluoro_buzz', dict(dur=26, size=.5, dips=sorted(TUBE), note='one tube failing; dips on these frames')), (47, 'box_fall', dict(size=.2, note='air round a small falling stone')), (54, 'box_clink', dict(size=1., pan=-.1, note='first hit on tile, ring of dust')), (63, 'box_clink', dict(size=.5, pan=.05, note='second hit')), (63, 'box_roll', dict(dur=5, size=.6, pan=.1, note='tips over an edge, rest at 67')), (71, 'eye_shut', dict(size=.2, note='the last eye closes, 68 to 71'))]

    def setup(self):
        self.face = dict(K=Face(11, 480), L=Face(12, 480), R=Face(13, 480), T=Face(14, 480), B=Face(15, 480), F=Face(16, 960))
        self.cube = Cube(.047, (16, 12, 13, 14, 15, 11), 360)
        hp = dict(w=2112, h=1188, eye=(-.30, -1.05, .19), target=(.10, 6., .66), focal=1300.)
        self.hl = {}
        for tb in (1., .25, 0.):
            p = world.plate('hall', tube=tb, **hp); self.cam = p.cam; z = smooth(1.4, 9., np.minimum(p.z, 60.))[..., None]; soft = cv2.GaussianBlur(p.img, (0, 0), 1.7); far = cv2.GaussianBlur(p.img, (0, 0), 3.4); im = p.img * (1 - np.clip(z * 2, 0, 1)) + soft * np.clip(z * 2, 0, 1) * (1 - np.clip(z * 2 - 1, 0, 1)) + far * np.clip(z * 2 - 1, 0, 1); self.hl[tb] = im.astype(np.float16); del p, soft, far, im, z
        self._track(); self.busts = {}
        r = np.random.default_rng(4); self.motes = np.c_[r.uniform(0, W, 46), r.uniform(0, H, 46), r.uniform(.4, 1.6, 46), r.uniform(0, 6.28, 46)]

    # =============================================================== A: inside the closing cube
    def bust(self, f):
        k = f // 2
        if k not in self.busts:
            ff = k * 2; up = ease((ff - 8) / 4.)                              # eyes: lowered, then on us
            bl = float(np.interp(ff, [17, 19, 21, 23, 25, 27, 29, 32], [1., .78, .38, 0., 0., .34, .74, 1.])); e = dict(band=False, lid='sad', brow='sad', mouth='sad', open=(.62 + .38 * up) * bl, gaze=(-.05 * (1 - up) + .02 * up, -.16 * (1 - up) + .01), pupil=.30, glow=.10 + .25 * up)
            br = math.sin(ff * .13) * .006; self.busts_make(k, ff, e, br, f)
        return self.busts[k]

    def busts_make(self, k, ff, e, br, f):
        pal = PAL['under']; Hd = gojo_head(True).pose(CX, CY + br * S0, S0, .13, .035 - .03 * ease((ff - 8) / 4.), .055, .10); B = GC.Body().pose(CX, CY + br * S0 + 1.42 * S0, S0, .05, -.03, .02)
        kw = dict(state='down', expr=e, light=pal, f=ff, wind=(0., 0.), seed=5, key='T'); c = Cel(W, H)
        GC.collar_back(c, B, pal, 5); Gojo.head(c, Hd, parts=('back',), neck_len=.15, **kw); GC.torso(c, B, pal, 5, arms='down', left=True, right=True); back = c.layer()
        c = Cel(W, H); Gojo.head(c, Hd, parts=('face',), **{**kw, 'key': None}); face = c.layer()
        # one hard shadow that turns at brow, cheekbone, nose and jaw (light high on his right); it lies on the skin only, under the drawing
        P = lambda pts, co: Hd.P(spl(np.array(pts, float), True, corners=co, n=7)); m = Cel(W, H, ss=2); d = Cel(W, H, ss=2)
        for pts, co in (([(.62, .78), (.535, .52), (.50, .30), (.455, .205), (.50, .13), (.572, .085), (.560, -.005), (.615, -.085), (.520, -.150), (.405, -.215), (.352, -.315), (.372, -.420), (.318, -.520), (.262, -.600), (.268, -.690), (.205, -.775), (.168, -.860), (.085, -.915), (.040, -1.03), (.75, -1.03), (.84, -.3), (.84, .78)], (3, 5, 7, 10, 11, 14, 16, 18, 19, 21, 0)),
                        ([(.205, .118), (.130, .072), (.080, .010), (.060, -.080), (.074, -.190), (.104, -.290), (.126, -.352), (.092, -.388), (.044, -.380), (.024, -.345), (.034, -.260), (.012, -.130), (.018, .000), (.040, .075), (.110, .125)], (0, 6, 9, 14)),
                        ([(-.535, .120), (-.420, .142), (-.280, .118), (-.165, .070), (-.090, .012), (-.062, -.075), (-.115, -.035), (-.180, .030), (-.290, .078), (-.410, .098)], (0, 5)), ([(.540, .120), (.420, .142), (.280, .118), (.200, .085), (.290, .078), (.410, .098)], (0, 3)),
                        ([(-.655, .02), (-.600, -.085), (-.540, -.135), (-.470, -.245), (-.430, -.370), (-.385, -.500), (-.330, -.610), (-.420, -.540), (-.520, -.380), (-.600, -.240), (-.690, -.12)], (2, 6, 10)),
                        ([(-.205, -.845), (-.110, -.895), (.0, -.912), (.10, -.895), (.06, -1.03), (-.16, -1.03)], (0, 3)), ([(-.052, -.800), (.0, -.808), (.056, -.798), (.030, -.826), (-.026, -.826)], (0, 2))): m.fill(P(pts, co), '#ffffff')
        m = m.rgba()[..., 3]; sk = rgb(pal['skin']); dist = np.abs(face[..., :3] - sk * face[..., 3:4]).max(-1); skin = (1 - smooth(.05, .20, dist)) * face[..., 3]; a_ = (m * skin)[..., None]; face[..., :3] = face[..., :3] * (1 - a_) + mix(rgb(pal['skin_sh']), rgb('#4f5f5a'), .35) * a_
        glow_ = Cel(W, H, ss=2)                                                # cold light caught on the brow, cheekbone and nose ridge
        for pts, co in (([(-.30, -.170), (-.20, -.185), (-.22, -.235), (-.34, -.270), (-.43, -.215)], (0, 2, 4)), ([(-.010, -.06), (.004, -.20), (.010, -.31), (-.018, -.33), (-.030, -.20)], (0, 2, 3)), ([(-.16, -.60), (-.06, -.585), (-.08, -.62)], (0, 1, 2))): glow_.fill(P(pts, co), '#ffffff')
        a_ = (glow_.rgba()[..., 3] * skin * .85)[..., None]; face[..., :3] = face[..., :3] * (1 - a_) + rgb('#d6e2cc') * a_
        c = Cel(W, H); Gojo.head(c, Hd, parts=('front',), **kw); sw = .012 * math.sin(ff * .16 + 1.)
        for (ra, tp, rb), bd, ww in ((((.165, .50), (.462 + sw, -.250), (.320, .47)), -.07, 3.4), (((.30, .46), (.520 + sw * 1.3, -.05), (.375, .44)), -.05, 2.6), (((.10, .42), (.285 + sw * 1.6, -.02), (.165, .42)), -.06, 2.4)):
            z = kit.spike(*[Hd.P(np.array([v_]))[0] for v_ in (ra, tp, rb)], bulge=.07, bend=bd, n=14); c.fill(z, pal['hair']); n_ = len(z) // 2; c.fill(np.vstack([z[n_:], (z[:n_ + 1] * .35 + z[::-1][:n_ + 1] * .65)]), pal['hair_sh']); c.line(z[:n_ + 1], ww, pal['hair_line'], taper=(.5, .08), press=.4, seed=3); c.line(z[n_:], ww * 1.25, pal['hair_line'], taper=(.08, .5), press=.4, seed=4)
        hair = c.layer()
        # a hard shadow of the fringe on the forehead
        y = np.arange(H, dtype=f32)[:, None]; sh = kit.shifted(hair[..., 3], .035 * S0, .105 * S0) * np.clip(face[..., 3], 0, 1) * (y < CY + .30 * S0); sh = smooth(.45, .55, sh)[..., None]; face[..., :3] *= 1 - sh * (1 - np.array([.60, .66, .70], f32))
        # clumps: planes that turn from the light, and four that catch it, cold
        q = lambda pts: np.array([(CX + (px + .06 * py) * S0, CY + br * S0 - py * S0) for px, py in pts]); m = Cel(W, H, ss=2)
        for pts in ([(-.64, .94), (-.57, .72), (-.50, .44), (-.53, .22), (-.61, .48), (-.66, .74)], [(-.27, 1.05), (-.22, .80), (-.16, .50), (-.12, .30), (-.22, .52), (-.28, .82)], [(.26, 1.04), (.31, .80), (.33, .50), (.31, .28), (.25, .54), (.23, .84)], [(.68, .92), (.68, .70), (.65, .44), (.66, .22), (.72, .46), (.74, .74)], [(-.88, .36), (-.86, .10), (-.81, -.18), (-.91, -.02), (-.94, .18)]):
            m.fill(spl(q(pts), True, corners=(0, 3), n=7), '#ffffff'); z = np.array(pts)
            for dx, t0, t1 in ((.075, .18, .80), (-.085, .30, .95)): m.line(spl(q([(z[0, 0] * (1 - u) + z[3, 0] * u + dx, z[0, 1] * (1 - u) + z[3, 1] * u) for u in (t0, (t0 + t1) / 2, t1)]), n=8), 9., '#ffffff', taper=(.5, .5), seed=int(abs(dx) * 999))
        dk = m.rgba()[..., 3] * hair[..., 3]; m = Cel(W, H, ss=2)
        for pts in ([(-.50, 1.10), (-.46, .82), (-.38, .56), (-.36, .36), (-.31, .60), (-.36, .90)], [(.00, 1.20), (.05, .92), (.08, .62), (.10, .42), (.16, .68), (.13, .98)], [(.42, 1.10), (.47, .84), (.49, .58), (.49, .40), (.56, .62), (.56, .88)], [(-.80, .62), (-.74, .42), (-.72, .18), (-.69, .02), (-.66, .24), (-.70, .50)]):
            m.fill(spl(q(pts), True, corners=(0, 3), n=7), '#ffffff')
            for dx, t0, t1 in ((.055, .10, .62), (-.045, .25, .80)): z = np.array(pts); m.line(spl(q([(z[0, 0] * (1 - u) + z[3, 0] * u + dx, z[0, 1] * (1 - u) + z[3, 1] * u) for u in (t0, (t0 + t1) / 2, t1)]), n=8), 5., '#ffffff', taper=(.4, .6), seed=int(abs(dx) * 999))
        lt = m.rgba()[..., 3] * hair[..., 3]; L_ = hair[..., :3].max(-1); ok = smooth(.40, .58, L_)
        hair[..., :3] = hair[..., :3] * (1 - (dk * .23 * ok)[..., None] * np.array([1.08, .96, .92], f32)); a_ = (lt * ok)[..., None]; hair[..., :3] = hair[..., :3] * (1 - a_) + np.array([.93, 1.06, 1.05], f32) * hair[..., 3:4] * a_
        hair = kit.rim(hair, 2.5, 5.5, '#eafffb', .75, .5)
        c = Cel(W, H); GC.collar_front(c, B, pal, 5); fr = c.layer(); ch = smooth(.45, .55, kit.shifted(np.clip(face[..., 3], 0, 1) * (y > CY + .30 * S0), .085 * S0, .24 * S0))[..., None]; self.chin = ch[..., 0].astype(np.float16)
        back[..., :3] *= 1 - ch * .62; fr[..., :3] *= 1 - ch * .62; out = back
        for l in (face, hair, fr): out = out * (1 - l[..., 3:4]) + l
        if len(self.busts) > 2: self.busts.clear()
        self.busts[k] = out.astype(f32)

    def strand(self, cel, pts, w, seed, lit=1.0, flare=(False, False)):
        """a rope of wet grey sinew: hard-edged planes, long fibres, knots, pen outline heavy below, broken wet light"""
        r = np.random.default_rng(seed); C = resample(spl(np.asarray(pts, float), n=14), 4.); m = len(C); t = np.linspace(0, 1, m); T = np.gradient(C, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-6; N = np.c_[-T[:, 1], T[:, 0]]
        kn = np.sort(r.uniform(.06, .94, max(2, int(m / 70)))); off = flare; e0 = (1 + .22 * np.clip(1 - t / .22, 0, 1) ** 2) if off[0] else (.22 + .78 * np.clip(t / .20, 0, 1) ** .6); e1_ = (1 + .75 * np.clip(1 - (1 - t) / .22, 0, 1) ** 2) if off[1] else (.22 + .78 * np.clip((1 - t) / .20, 0, 1) ** .6); wd = w * (1 + .10 * noise1(m, 40, r)) * e0 * e1_
        for k in kn: wd = wd * (1 - .24 * np.exp(-((t - k) * m * 4 / (w * .40)) ** 2))
        s = np.where(N[:, 1] < 0, 1., -1.); up = N * s[:, None]; A, B = C + up * (wd * .5)[:, None], C - up * (wd * .5)[:, None]; poly = np.vstack([A, B[::-1]]); edge = lambda o, j, sc: C + up * ((o + j * noise1(m, sc, r) + .5 * j * noise1(m, sc * .3, r)) * wd)[:, None]
        pal = ('#7d807b', '#a7aaa0', '#4b5053', '#2b2f33') if lit > .6 else ('#565b5a', '#6c716d', '#363b3e', '#222629'); cel.fill(poly, pal[0]); cel.clip(poly)
        e1 = edge(.22, .08, 24); cel.fill(np.vstack([A + up * 4, e1[::-1]]), pal[1]); e2 = edge(-.10, .10, 22); cel.fill(np.vstack([e2, (B - up * 4)[::-1]]), pal[2]); e3 = edge(-.36, .07, 16); cel.fill(np.vstack([e3, (B - up * 4)[::-1]]), pal[3])
        if w > 26:                                                            # tendons inside the rope: each a lit ridge and its own dark
            for k in range(int(2 + w / 34)):
                o0, o1 = r.uniform(-.40, .40, 2); a_, b_ = sorted((r.uniform(-.05, .45), r.uniform(.5, 1.05))); i0, i1 = int(max(0, a_) * (m - 1)), int(min(1, b_) * (m - 1)); tt = np.linspace(0, 1, i1 - i0); o = o0 + (o1 - o0) * tt + .07 * np.sin(tt * 5 + k); q = C[i0:i1] + up[i0:i1] * (o * wd[i0:i1])[:, None]; cel.line(q, w * .055 + 1.5, '#17191d', taper=(.25, .35), press=.5, seed=seed * 50 + k); cel.line(q + up[i0:i1] * (w * .055 + 1.5), w * .035 + 1.2, pal[1], taper=(.3, .4), press=.5, seed=seed * 51 + k, alpha=.8)
        for k in range(int(4 + w * .10)):
            o = r.uniform(-.46, .46); a_ = r.uniform(0, .85); i0, i1 = int(a_ * (m - 1)), int(min(1, a_ + r.uniform(.06, .22)) * (m - 1))
            if i1 - i0 > 6: cel.line(C[i0:i1] + up[i0:i1] * ((o + .04 * np.sin(t[i0:i1] * 20 + k)) * wd[i0:i1])[:, None], r.uniform(1.1, 2.2), '#23272a', taper=(.35, .5), seed=seed * 60 + k, alpha=.8)
        for k in kn:                                                          # creases at the knots
            i = int(k * (m - 1))
            for j in range(4): q = min(m - 2, max(1, i + int(r.normal(0, 2.4)))); u0, u1 = sorted(r.uniform(-.5, .4, 2)); P0, P1 = C[q] + up[q] * u0 * wd[q], C[q] + up[q] * u1 * wd[q]; cel.line(spl([P0, (P0 + P1) / 2 + T[q] * wd[q] * .10, P1], n=6), r.uniform(1.6, 3.2), '#0f1215', taper=(.2, .5), seed=seed * 70 + j + i)
        for k in range(int(m / 9)):                                           # hatching rolls up from the under side
            i = int(r.integers(2, m - 3)); L_ = wd[i] * r.uniform(.16, .40); p0 = B[i]; p1 = p0 + up[i] * L_ + T[i] * L_ * .30; cel.line(spl([p0, (p0 + p1) / 2 - T[i] * 2, p1], n=5), r.uniform(1.4, 2.6), '#0f1214', taper=(.02, .85), seed=seed * 90 + k)
        for k in range(int(2 + m / 60)):                                      # wet light, broken
            i = int(r.uniform(.04, .92) * m); j = min(m - 1, i + int(r.uniform(4, 16)))
            if j - i > 3: cel.line(A[i:j] - up[i:j] * (wd[i:j] * r.uniform(.12, .22))[:, None], r.uniform(1.8, 3.4) + w * .012, '#f1f6ea' if lit > .6 else '#9aa39b', taper=(.35, .45), seed=seed * 33 + k, alpha=.95)
        cel.unclip(); cel.line(A, 2.6 + w * .006, '#0d0f11', taper=(.02, .02), press=.5, seed=seed * 7); cel.line(B, 3.8 + w * .012, '#0b0c0f', taper=(.02, .02), press=.5, seed=seed * 7 + 1)

    def anc(self, k, d, v, th):
        """a point on the inner side of a wall: d = 0 hinge .. 1 outer edge, v = -1..1 along the hinge -> screen px"""
        if k == 'K': return prj(np.array([d, v, -A_]))
        D_ = 2 * A_ * d; c, s = D_ * math.cos(D(th[k])), -A_ + D_ * math.sin(D(th[k]))
        return prj(np.array({'L': (-A_ - c, v * A_, s), 'R': (A_ + c, v * A_, s), 'T': (v * A_, -A_ - c, s), 'B': (v * A_, A_ + c, s)}[k]))

    def root(self, cel, p, w, toward, seed, clip):
        """where a rope grows out of the stone (painted on the wall, under the rope): a low swelling, tendons running out into the grain, dark round the foot"""
        r = np.random.default_rng(seed + 300); a0 = math.atan2(*(np.asarray(toward) - p)[::-1]); w = max(w, 24.); n = 9; hi, md, lo = '#7f847c', '#5c615d', '#2a2f32'; cel.clip(clip)
        th = np.linspace(0, 2 * math.pi, 48, endpoint=False); c, s_ = math.cos(a0), math.sin(a0); ring = lambda k, j: (np.c_[np.cos(th) * w * k * .92, np.sin(th) * w * k] * (1 + j * noise1(48, 6, r))[:, None]) @ np.array([[c, s_], [-s_, c]]) + p
        cel.fill(ring(1.55, .16), '#0b0e10', .45)
        for j in range(n):
            an = a0 + math.pi + (j - (n - 1) / 2) / ((n - 1) / 2) * 2.5 + r.normal(0, .10); L = w * r.uniform(1.0, 1.75); d = np.array([math.cos(an), math.sin(an)]); nn = np.array([-d[1], d[0]]) * r.normal(0, .10) * L; q = spl([p + d * w * .3, p + d * L * .6 + nn, p + d * L * 1.15 + nn * .6], n=10); ww = w * r.uniform(.20, .34)
            cel.line(q + [0, ww * .3], ww * 1.3, lo, taper=(.0, .95), press=.3, seed=seed + j, tmin=.04); cel.line(q, ww, md, taper=(.0, .95), press=.3, seed=seed + j, tmin=.04); cel.line(q - [0, ww * .22], ww * .30, hi, taper=(.0, .9), press=.4, seed=seed + j + 9, tmin=.05)
        X = ring(.95, .08); cel.fill(X, md); cel.clip(X); cel.fill((X - p) * .82 + p - [w * .05, w * .22], hi, .75); cel.fill((X - p) * [1, .5] + p + [0, w * .68], lo)
        for k in range(16): t = r.uniform(0, 6.28); d = np.array([math.cos(t), math.sin(t)]); cel.line(spl([p + d * w * .98, p + d * w * .72 + r.normal(0, w * .04, 2), p + d * w * r.uniform(.40, .58)], n=5), 1.8 + w * .012, '#14181b', taper=(.02, .85), seed=seed + k)
        cel.unclip(); cel.unclip()

    def sinew(self, cel, f, front, th, roots):
        s = S0; b = math.sin(f * .21) * 2.4; cl = ease((f - 18) / 20.); E = lambda x, y: np.array([CX + x * s, CY + y * s + b * .4])
        def rope(an, e, sag, w, seed, lit=1., d2=None):
            A = self.anc(*an, th); Z = E(*e); L = np.hypot(*(Z - A)); sg = (sag + .10 * cl) * s + b; N = np.array([0, 1.]); pts = [A * (1 - t) + Z * t + N * sg * 4 * t * (1 - t) * (.85 + .3 * t) for t in (0, .25, .5, .75, 1)]
            self.strand(cel, pts, w, seed, lit, flare=(True, False)); self.root(roots, A, w * .80, pts[1], seed, prj(self.quad(an[0], th.get(an[0], 0)))); self.ends.append((A, w))
        if not front:
            for an, e, sag, w, sd in ((('K', -1.22, -1.02), (-.46, .55), .10, 56, 1), (('K', 1.24, -.92), (.46, .55), .10, 64, 2), (('T', .42, -.62), (-.40, .40), .0, 42, 3), (('T', .38, .66), (.40, .40), .0, 38, 4), (('K', -1.30, .10), (-.50, .95), .08, 34, 5), (('K', 1.30, .22), (.50, .95), .08, 32, 6)): rope(an, e, sag, w, sd, .4)
            return
        for an, e, sag, w, sd in ((('L', .78, .96), (.20, 1.58), .02, 128, 22), (('R', .80, .98), (-.20, 1.62), .02, 136, 32), (('L', .72, .84), (-.22, 1.26), .12, 88, 21), (('R', .70, .86), (.26, 1.27), .12, 94, 31), (('B', .26, -.62), (-.48, 1.10), -.04, 48, 24), (('B', .28, .66), (.54, 1.11), -.04, 54, 25),
                                  (('L', .58, .64), (-.56, 1.10), .12, 38, 23), (('R', .60, .67), (.62, 1.12), .12, 42, 33)): rope(an, e, sag, w, sd)
        for q, w, sd in ((((-.66, 1.12), (-.32, 1.22), (.05, 1.24), (.42, 1.20), (.70, 1.12)), 46, 26), (((-1.0, 1.30), (-.6, 1.38), (-.1, 1.33), (.5, 1.42), (1.05, 1.32)), 20, 41)): self.strand(cel, [E(*v) for v in q], w, sd, 1.)
        for an, e, sag, w, sd in ((('L', .88, .26), (-.73, .94), .16, 25, 27), (('R', .88, .18), (.79, .98), .16, 28, 28), (('L', .40, -.74), (-.66, .84), .20, 14, 29), (('R', .42, -.72), (.70, .90), .22, 12, 30)): rope(an, e, sag, w, sd)

    def walls(self, f):
        """angles of the four sides (deg, 90 = shut) and of the front lid"""
        def th(k, a0, f0):
            f1 = SHUT[k]; br = 1.3 * math.sin(f * .19 + ord(k)); u = (f - f0) / (f1 - f0)
            if u <= 0: return a0 + br
            if u < 1: return a0 + br * (1 - u) + (90 - a0) * u ** 2.3
            g = f - f1; return 90 - 3.2 * math.exp(-g * .9) * math.cos(g * 2.2) * (g > 0)
        return dict(L=th('L', 34, 17), R=th('R', 30, 20), T=th('T', 32, 23), B=th('B', 26, 26))

    def quad(self, k, th):
        c, s = math.cos(D(th)) * 2 * A_, math.sin(D(th)) * 2 * A_; a = A_
        if k == 'K': return np.array([(-a, -a, -a), (a, -a, -a), (a, a, -a), (-a, a, -a)])
        if k == 'L': return np.array([(-a - c, -a, -a + s), (-a, -a, -a), (-a, a, -a), (-a - c, a, -a + s)])
        if k == 'R': return np.array([(a, -a, -a), (a + c, -a, -a + s), (a + c, a, -a + s), (a, a, -a)])
        if k == 'T': return np.array([(-a, -a - c, -a + s), (a, -a - c, -a + s), (a, -a, -a), (-a, -a, -a)])
        return np.array([(-a, a, -a), (a, a, -a), (a, a + c, -a + s), (-a, a + c, -a + s)])

    def inside(self, f):
        x, y = grid(); th = self.walls(f); cl = ease((f - 18) / 26.)
        base = rgb('#0a0f0e') + (rgb('#1c2a27') - rgb('#0a0f0e')) * np.clip(1 - np.hypot((x - CX) / 900., (y - 380) / 620.), 0, 1)[..., None] ** 1.4 * (1 - .6 * cl)
        amb = np.array([.88, 1.0, .93]) * (1 - .46 * cl)
        op = {k: float(np.interp(f, *v)) for k, v in dict(K=([8, 15, 60], [0, .62, .62]), L=([19, 28, 60], [0, .9, .9]), R=([22, 31, 60], [0, .95, .95]), T=([25, 34, 60], [0, .9, .9]), B=([28, 37, 60], [0, .9, .9])).items()}
        gz = dict(K=(0, .2), L=(.9, 0.), R=(-.9, 0.), T=(0., .8), B=(0., -.8))
        def put(k, t, sh, bl, front=None):
            nonlocal base
            Q = self.quad(k, t); tx = self.face[k].cached(op[k], gz[k])
            if front is not None:                                              # only the part of the floor that passes in front of him
                u0 = (front + A_) / (2 * A_ * max(.05, math.sin(D(t))))
                if u0 >= .98: return
                Q = np.array([Q[0] + (Q[3] - Q[0]) * u0, Q[1] + (Q[2] - Q[1]) * u0, Q[2], Q[3]]); tx = cv2.resize(tx[int(u0 * len(tx)):], (len(tx), len(tx)), interpolation=cv2.INTER_LINEAR)
            q = prj(Q); l = warp_face(tx, q, sh, light=amb); l = cv2.GaussianBlur(l, (0, 0), bl) if bl > .4 else l; base = base * (1 - l[..., 3:4]) + l[..., :3]
        put('K', 0, (.52, .52, .30, .30), 6.5)
        for k in ('T', 'B', 'L', 'R'):
            s = th[k] / 90.; inn, out = .62 * (1 - .25 * cl), 1.25 - .45 * s; c4 = dict(L=(out, inn, inn * .8, out * .8), R=(inn, out, out * .8, inn * .8), T=(out, out, inn, inn), B=(inn * .7, inn * .7, out * .62, out * .62))[k]
            put(k, th[k], c4, 3.0)
        def ropes(front):
            c, rc = Cel(W, H), Cel(W, H); self.ends = []; self.sinew(c, f, front, th, rc); l = c.layer()
            for A, w in self.ends:                                             # each rope sinks into its root
                q_ = max(10., w * .62); x0_, x1_, y0_, y1_ = [int(v) for v in (max(0, A[0] - 4 * q_), min(W, A[0] + 4 * q_), max(0, A[1] - 4 * q_), min(H, A[1] + 4 * q_))]
                if x1_ > x0_ and y1_ > y0_: d_ = np.hypot(x[y0_:y1_, x0_:x1_] - A[0], y[y0_:y1_, x0_:x1_] - A[1]) / q_; k_ = smooth(.15, 1.05, d_)[..., None]; l[y0_:y1_, x0_:x1_] *= k_; l[y0_:y1_, x0_:x1_, :3] *= .45 + .55 * smooth(.6, 2.2, d_)[..., None]
            return l, cv2.GaussianBlur(rc.layer(), (0, 0), 1.7)
        R_, rt = ropes(True); wl = .80 - .42 * cl; base = base * (1 - rt[..., 3:4]) + rt[..., :3] * wl * amb.astype(f32)
        l, rt = ropes(False); base = base * (1 - rt[..., 3:4]) + rt[..., :3] * wl * .6 * amb.astype(f32); l = cv2.GaussianBlur(l, (0, 0), 2.0); sh = np.clip(.62 - .3 * cl, 0, 1); base = base * (1 - l[..., 3:4]) + l[..., :3] * sh
        base = base * (1 - .30 * cv2.GaussianBlur(self.bust(f)[..., 3], (0, 0), 34)[..., None])              # he darkens the wall behind him
        B = self.bust(f); R_[..., :3] *= 1 - .45 * self.chin.astype(f32)[..., None]; B = B * (1 - R_[..., 3:4]) + R_
        B = kit.para(B, (CX, CY - 420), (CX, CY + 640), [(0, '#ffffff'), (.45, '#e9f2ea'), (.8, '#7d8c86'), (1, '#39443f')], 'mul', 1.)
        # light on him: closes to a slit across the eyes
        yt = float(np.interp(f, [18, 26, 32, 37, 41, 44], [-700, -330, -150, -78, -44, -16])) + CY; yb = float(np.interp(f, [18, 26, 31, 36, 41, 44], [900, 560, 300, 130, 46, 20])) + CY; ed = float(np.interp(f, [18, 44], [120, 9]))
        tilt = (x - CX) * .045; band = smooth(yt - ed, yt + ed, y - tilt) * smooth(yb + ed, yb - ed, y - tilt); amb_ = float(np.interp(f, [18, 34, 44], [1., .30, .05])); lit = amb_ + (1 - amb_) * band; gain = 1 + .34 * ease((f - 34) / 9.)
        lit = lit[..., None] * (np.array([1., 1., 1.]) * (1 - band[..., None]) + gain * band[..., None]); lit = np.where(lit < 1, lit * np.array([.80, .95, .92]) ** ((1 - np.clip(lit, 0, 1)) * 1.4), lit)
        if f > 34:
            e = ease((f - 34) / 8.) * .22; base = base + (band[..., None] * np.array([.55, .80, .72]) * e * np.clip(1 - np.abs(x - CX) / 900., 0, 1)[..., None])
        base = base * (1 - B[..., 3:4]) + B[..., :3] * lit
        s = th['B'] / 90.; put('B', th['B'], (.28 * (1 - .25 * cl),) * 2 + (.46 - .16 * s,) * 2, 3.0, front=.40)
        # iris catch light returns through the dark
        # ---- front lid, rising
        if f >= 34:
            psi = float(np.interp(f, [34, 37, 39, 41, 42, 43, 44, 45], [100, 105.5, 108, 110.8, 112.6, 114.8, 119.4, 180])); yy, zz = A_, A_; e = np.array([0, math.cos(D(psi)), math.sin(D(psi))]) * 2 * A_; a = A_ * 1.04
            Q = np.array([(-a, yy + e[1], zz + e[2]), (a, yy + e[1], zz + e[2]), (a, yy, zz), (-a, yy, zz)]); Q[:, 2] = np.minimum(Q[:, 2], DD - 1.6); q = prj(Q)
            if psi > 106.2:
                o = float(np.interp(f, [38, 41, 43, 44], [.0, .35, .8, 1.])); l = warp_face(self.face['F'].cached(o, (0, -.3), .30), q, (.20, .20, .44, .44), light=np.array([.82, 1., .94])); bl = float(np.interp(f, [36, 40, 43, 44], [9, 5, 2.4, 1.4])); l = cv2.GaussianBlur(l, (0, 0), bl)
                ed_ = q[0][1]; base = base * (1 - .85 * np.clip(smooth(ed_ - 46, ed_ + 4, y), 0, 1)[..., None] * (np.abs(x - CX) < 900)[..., None]); base = base * (1 - l[..., 3:4]) + l[..., :3]
                rimy = np.exp(-((y - ed_ - 3) / 3.0) ** 2) * (x > q[0][0]) * (x < q[1][0]); base = base + (rimy * .30 * (1 - cl * .4))[..., None] * np.array([.75, 1., .9])
        # motes
        c = Cel(W, H)
        for mx, my, sz, ph in self.motes:
            px, py = (mx + 14 * math.sin(f * .05 + ph) + f * .6) % W, (my + f * .5 * sz) % H; c.fill(np.array([(px + math.cos(a_) * sz * 1.7, py + math.sin(a_) * sz * 1.7) for a_ in np.linspace(0, 6.28, 9)]), '#cfe6da', .30 * (1 - cl))
        l = cv2.GaussianBlur(c.layer(), (0, 0), 1.3); base = base + l[..., :3]
        return comp.cam2d(base.astype(f32), CX - 4, CY + 22 - 18 * ease(f / 44.), 1.02 + .075 * ease(f / 46.), D(-.5) * (f / 44.), interp=cv2.INTER_CUBIC)

    # =============================================================== B: the hall
    def _track(self):
        h = self.cube.h; cr = np.array([(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float) * h
        self.low = lambda q: float(-(cr @ qmat(q).T)[:, 2].min())
        mv = np.array([.94, .34, 0]); ax = np.cross([0, 0, 1], mv); qr = quat((0, 0, 1), D(-31))                                # rest: turned a little, an eye to us
        q63 = qmul(quat(ax, D(-52)), qr); q54 = qmul(quat((.50, .62, .60), D(-300)), q63); self.qk = (qr, q63, q54, ax)
        self.p54 = np.array([-.330, -.452]); self.p63 = np.array([-.212, -.420]); self.prest = self.p63 + mv[:2] / np.hypot(*mv[:2]) * .068

    def pose(self, t):
        qr, q63, q54, ax = self.qk; g = 9.81
        if t <= LAND1:
            q = qmul(quat((-.35, .75, .55), D(-265) * (LAND1 - t) / 8.), q54); tau = (LAND1 - t) / 24.; z = self.low(q54) + 1.85 * tau; xy = self.p54 + np.array([-.10, .035]) * (LAND1 - t) / 8.; z = max(z, self.low(q))
        elif t <= LAND2:
            u = (t - LAND1) / (LAND2 - LAND1); q = qmul(quat((.50, .62, .60), D(300) * u), q54); tau = (t - LAND1) / 24.; z0, z1 = self.low(q54), self.low(q63); z = z0 + (z1 - z0) * u + 1.84 * tau - .5 * g * tau * tau; z = max(z, self.low(q)); xy = self.p54 + (self.p63 - self.p54) * u
        else:
            u = min(1., (t - LAND2) / (REST - LAND2)); k = u ** 1.6; a = 52 * k
            if t > REST: gk = t - REST; a = 52 - 3.0 * math.exp(-gk * 1.1) * abs(math.sin(gk * 2.3))
            q = qmul(quat(ax, D(a)), q63); z = self.low(q); xy = self.p63 + (self.prest - self.p63) * (1 - (1 - u) ** 1.5)
        return np.array([xy[0], xy[1], z]), q

    def hall_frame(self, f):
        tb = TUBE.get(f, 1.); OX, OY = (2112 - W) / 2, (1188 - H) / 2; img = self.hl[tb].astype(f32).copy(); hh, ww = img.shape[:2]; cam = self.cam; t = f + .0; sc = .30 + .70 * tb
        lights = [((0., 8.2, 2.8), np.array([.95, 1.12, 1.06]) * 1.25 * tb), ((-2.3, .2, 1.3), np.array([.52, .66, .66]) * 1.1 * sc), ((2.3, -.2, 1.2), np.array([.26, .40, .42]) * .9 * sc), ((-.3, -1.4, 1.1), np.array([.34, .40, .40]) * 1.0 * sc)]; amb = np.array([.33, .38, .39]) * (.55 + .45 * tb)
        def opens(tt): e = float(np.interp(tt, [46, 54, 62, 67, 68, 69, 70, 71], [.75, 1., .92, .84, .80, .55, .22, 0.])); o = float(np.interp(tt, [46, 56, 62, 66], [.7, .6, .3, 0.])); return [o, o * .8, e, o * .7, o, o * .9]
        p, q = self.pose(t)
        # floor: shadow, reflection, dust
        x, y = grid(hh, ww); rays = cam.rays(hh, ww); tz = -cam.eye[2] / np.where(rays[..., 2] < -1e-4, rays[..., 2], -1e9); fx, fy = cam.eye[0] + rays[..., 0] * tz, cam.eye[1] + rays[..., 1] * tz; fl = (rays[..., 2] < -1e-3) & (tz < 14)
        k = math.exp(-max(0, p[2] - .05) / .16); d1 = np.hypot((fx - p[0]) / (.058 + p[2] * .12), (fy - p[1] + .012) / (.064 + p[2] * .16)); d2 = np.hypot((fx - p[0] - .010) / (.060 + p[2] * .25), (fy - p[1] + .085 + p[2] * .5) / (.115 + p[2] * .3)); sh = np.where(fl, .78 * k * smooth(1.25, .55, d1) + .42 * k * tb * smooth(1.35, .2, d2), 0)[..., None]; img = img * (1 - np.clip(sh, 0, .9))
        cl_ = self.cube.draw(cam, p, qmat(q), opens(t), lights, amb, 2, False, h=hh, w=ww); R_ = qmat(q); c = Cel(ww, hh); hc = self.cube.h   # drawn edges: pen outside, a cold glint on the edges that meet the tube
        vis = {}
        for i, (n, u_, v_) in enumerate(FACES):
            n, u_, v_ = [R_ @ np.asarray(k_, float) for k_ in (n, u_, v_)]; ce = p + n * hc
            if (cam.eye - ce) @ n > 0: vis[i] = (n, cam.project(np.array([ce + hc * (-u_ - v_), ce + hc * (u_ - v_), ce + hc * (u_ + v_), ce + hc * (-u_ + v_)]))[0])
        tl = np.array([0., 8.2, 2.8]) - p; tl /= np.linalg.norm(tl)
        for i, (n, qd) in vis.items():
            for j in range(4): c.line(np.array([qd[j], (qd[j] + qd[(j + 1) % 4]) / 2, qd[(j + 1) % 4]]), 3.4, '#0a0d0e', taper=(.06, .06), press=.4, seed=i * 4 + j, tmin=.5)
        for i, (n, qd) in vis.items():
            if n[2] > .35 or n @ tl > .1:
                for j in range(4): c.line(np.array([qd[j] * .97 + qd[(j + 1) % 4] * .03, qd[(j + 1) % 4] * .97 + qd[j] * .03]), 1.8, '#dff6ee', taper=(.2, .2), press=.5, seed=i * 9 + j, alpha=.75 * tb * min(1., max(n[2], n @ tl) * 1.3))
        l = c.layer(hh, ww); cl_ = cl_ * (1 - l[..., 3:4]) + l
        if f in (51, 52): cl_ = self.smear(cl_, (150, 175)[f - 51], f)
        rf = self.cube.draw(cam, p, qmat(q), opens(t), lights, amb * .8, 1, True, h=hh, w=ww); rf = cv2.GaussianBlur(rf, (0, 0), sigmaX=1.6, sigmaY=3.2 + 14 * max(0, p[2] - .05)); fr = np.where(fl, .66, 0)[..., None]; img = img * (1 - rf[..., 3:4] * fr) + rf[..., :3] * fr
        for t0, amp, c0, sd in ((LAND1, 1., self.p54, 3), (LAND2, .5, self.p63, 4)):
            a = t - t0
            if 0 <= a < 17:
                r = np.random.default_rng(sd); d = np.hypot(fx - c0[0], fy - c0[1]); R = .040 + .185 * amp * (1 - math.exp(-(a + .5) / 4.6)); w_ = .016 + .0040 * a; an = np.arctan2(fy - c0[1], fx - c0[0]); nz = sum(np.sin(an * k + p_) * am for k, p_, am in zip((3, 5, 8, 13), r.uniform(0, 6, 4), (.5, .4, .35, .25)))
                ring = np.where(fl, np.exp(-((d - R * (1 + .06 * nz)) / w_) ** 2) * np.clip(.75 + .35 * nz, .1, 1.5) + .45 * np.exp(-(d / (R * .8 + .01)) ** 2) * math.exp(-a / 3.), 0).astype(f32)
                up = ring.copy(); hpx = 10 + 26 * amp * (1 - math.exp(-a / 3.5))                                  # it stands a little off the tiles
                for k in range(1, 7): up = np.maximum(up, kit.shifted(ring, 1.2 * k, -hpx * k / 6.) * (1 - k / 7.5))
                tex = .55 + .45 * np.clip(fbm(hh, ww, (30, 12, 5), (1, .7, .45), 60 + sd) * .6 + .5, 0, 1.4); up = cv2.GaussianBlur(up * tex, (0, 0), 2.0 + .35 * a); al = np.clip(up, 0, 1) * .46 * amp * math.exp(-a / 6.0) * sc; img = img * (1 - al[..., None] * .55) + al[..., None] * rgb('#b9d2cc') * .80
                c = Cel(ww, hh, ss=1)
                for i in range(int(26 * amp)):
                    an, vv, vz, r_ = r.uniform(0, 6.28), r.uniform(.2, .9) * amp, r.uniform(.3, 1.1) * amp, r.uniform(.012, .075); tt = a / 24.; zz_ = .004 + vz * tt - 4.9 * tt * tt
                    if zz_ < 0: continue
                    (gx, gy), gz = cam.project(np.array([c0[0] + math.cos(an) * (r_ + vv * tt), c0[1] + math.sin(an) * (r_ + vv * tt), zz_])); sz = r.uniform(.0008, .0022) * cam.f / gz; c.fill(np.array([(gx - sz, gy), (gx, gy - sz), (gx + sz, gy + sz * .3), (gx, gy + sz)]), '#9fb4b0', .9)
                l = c.layer(hh, ww); img = img * (1 - l[..., 3:4]) + l[..., :3]
        img = img * (1 - cl_[..., 3:4]) + cl_[..., :3]
        sh_ = kit.shake(f, 3.0, seed=7, decay=1.1, t0=54) + kit.shake(f, 1.3, seed=8, decay=1.0, t0=63)
        u = (f - 46) / 26.; return comp.cam2d(img.astype(f32), ww / 2 + 40 + sh_[0] + 8 * u, hh / 2 + 112 + sh_[1] + 12 * u, 1.13 + .045 * u, D(-1.2), interp=cv2.INTER_CUBIC)

    def smear(self, L, ln, seed):
        """a drawn smear: the cube pulled out upward along its fall, cel tones, edges torn like a dry brush"""
        ys, xs = np.nonzero(L[..., 3] > .5); x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1; n = x1 - x0; r = np.random.default_rng(seed); out = np.zeros_like(L)
        top = np.array([np.argmax(L[y0:y1, x, 3] > .5) + y0 for x in range(x0, x1)]); bot = np.array([y1 - 1 - np.argmax(L[y0:y1, x, 3][::-1] > .5) for x in range(x0, x1)])
        co, fi, hi = noise1(n, 46, r), noise1(n, 12, r), noise1(n, 3.2, r); u = np.linspace(-1, 1, n); hg = ln * np.clip(.74 + .22 * co + .16 * fi + .10 * hi - .34 * np.abs(u) ** 3, .16, 1.2)
        stq = np.round((.5 + .30 * noise1(n, 30, r) + .22 * noise1(n, 8, r)) * 4) / 4.; gap = np.abs(noise1(n, 5.5, r)) + .35 * np.abs(noise1(n, 2.2, r)); inkc = np.abs(noise1(n, 17, r)) < .11
        mc = L[y0:y1, x0:x1][L[y0:y1, x0:x1, 3] > .9][:, :3].mean(0); pal = [mc * k_ for k_ in (.22, .58, .95, 1.4, 1.85)]
        for i, x in enumerate(range(x0, x1)):
            h_ = int(hg[i]); yt = int(top[i]); ya = max(0, yt - h_); m = yt - ya
            if m < 2: continue
            v = (yt - np.arange(ya, yt)) / float(h_); dry = gap[i] + .30 * np.sin(v * 9 + i * .9) * .4; al = np.clip((dry - (v ** 2.2) * .95) * 6 + .9, 0, 1) * np.clip((1 - v) * 14, 0, 1); k = int(np.clip(stq[i] * 4 + .5, 0, 4)); col = pal[k] if not inkc[i] else pal[0]
            out[ya:yt, x, :3] = col * al[:, None]; out[ya:yt, x, 3] = al
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=.55, sigmaY=1.6); out[..., :3] = np.minimum(out[..., :3], out[..., 3:4] * 1.05)
        c = Cel(L.shape[1], L.shape[0])                                        # pen down both flanks, and a few hairs of speed
        for i, ww in ((2, 6.), (n - 3, 7.5), (int(n * .34), 3.), (int(n * .67), 3.4)): c.line(np.array([(x0 + i, bot[i] if i in (2, n - 3) else top[i] + 8), (x0 + i + r.normal(0, 2), top[i] - hg[i] * .45), (x0 + i + r.normal(0, 3), top[i] - hg[i] * r.uniform(.85, 1.1))]), ww, '#0a0d0e', taper=(.05, .75), press=.5, seed=seed * 9 + i)
        for _ in range(9): i = int(r.uniform(0, n - 1)); c.line(np.array([(x0 + i, top[i] - hg[i] * r.uniform(.8, 1.0)), (x0 + i + r.normal(0, 2), top[i] - hg[i] * r.uniform(1.15, 1.5))]), r.uniform(1.4, 3.2), '#8f9a96' if r.random() < .5 else '#14181a', taper=(.3, .7), seed=seed * 5 + i)
        l = c.layer(L.shape[0], L.shape[1]); out = out * (1 - l[..., 3:4]) + l
        yy = np.arange(L.shape[0], dtype=f32)[:, None]; k = smooth(y0 + (y1 - y0) * .10, y0 + (y1 - y0) * .46, yy)[..., None]; body = L.copy(); st_ = cv2.GaussianBlur(L, (0, 0), sigmaX=.3, sigmaY=9.)                    # the face itself is dragged upward, crisp at the leading edge
        body = (st_ * (1 - k) + L * k) * np.clip(L[..., 3:4] * 1.0, 0, 1); body[..., 3] = L[..., 3]; return out * (1 - body[..., 3:4]) + body

    def frame(self, f):
        if f < 45: return np.maximum(self.inside(f), 0).astype(f32)
        if f == 45: return np.zeros((H, W, 3), f32) + rgb('#050807') * .4
        return np.maximum(self.hall_frame(f), 0).astype(f32)
