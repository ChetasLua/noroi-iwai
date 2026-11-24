"""NOBARA KUGISAKI.  Nobara.head(cel, H, expr, light, f, wind, seed, patch=False) / Nobara.bust(cx, cy, s, yaw, pitch, roll, ...) -> layer"""
import numpy as np, math
from chars import mn
from chars.mn import planes, face_line, Lock, LIGHT, lw, hair_group, scalp, wind_off, eye, brow, nose, mouth, face_shadow, ear, neck, INK
from jj.head3 import Head
from jj.ink import Cel, spl
from jj.kit import outline, shape

A = [(-1.0, .105), (-.96, .175), (-.86, .300), (-.72, .425), (-.56, .535), (-.36, .622), (-.12, .672), (.12, .695), (.35, .70), (.6, .675), (.8, .54), (.93, .33), (1.0, .05)]

HAIR = {'day': dict(base='#bd6b35', sh='#874426', hi='#e3a263', line='#37190f', hip=0, band=(.30, .36, ('sideL', 'sideR', 'fr', 'frL', 'frR', 'back'), .8)), 'sunset': dict(base='#bb5a30', sh='#763029', hi='#f7b563', line='#300f0e', hip=0, band=(.30, .36, ('sideL', 'sideR', 'fr', 'frL', 'frR', 'back'), .8)),
        'under': dict(base='#7d5646', sh='#4f3740', hi='#a98a72', line='#1c1218', hip=0, band=(.30, .36, ('sideL', 'sideR', 'fr', 'frL', 'frR', 'back'), .8)), 'battle': dict(base='#955034', sh='#4b2a44', hi='#d79a78', line='#1b0f18', hip=0, band=(.30, .36, ('sideL', 'sideR', 'fr', 'frL', 'frR', 'back'), .8)),
        'snow': dict(base='#bb7240', sh='#85504a', hi='#eebf8d', line='#301b18', hip=0, band=(.30, .36, ('sideL', 'sideR', 'fr', 'frL', 'frR', 'back'), .8))}
HAIR['gold'] = dict(HAIR['day'], base='#b96430', sh='#78392a', hi='#ffc374')
HAIR['blue'] = dict(base='#b0623a', sh='#5d3050', hi='#f0b887', line='#24101a', hip=0, band=(.30, .36, ('sideL', 'sideR', 'fr', 'frL', 'frR', 'back'), .8))

C = dict(
    eye=(.322, -.085),
    upper=[(-.170, -.054), (-.132, .026), (-.064, .084), (.026, .102), (.110, .082), (.178, .022)],
    lower=[(-.170, -.054), (-.124, -.094), (-.044, -.119), (.052, -.113), (.130, -.072), (.178, .022)],
    iris=(.082, .110), irisc=(.002, -.008),
    ic=dict(ring='#2a120c', base='#9a541d', top='#4f2410', lo='#e09a3c', pup='#160805', spoke=1),
    lash=dict(col='#1c0f10', tk=[.005, .018, .029, .036, .041, .036], wing=(.052, .010), up=[(.84, .062, .042, .024)],
              low=[(.80, .022, -.022)], lowr=(.25, .97), loww=.9),
    crease=[(-.085, .042), (.0, .044), (.085, .046), (.150, .056)],
    brow=[(-.170, .138, .014), (-.14, .150, .036), (-.04, .196, .032), (.085, .238, .022), (.225, .214, .001)], browc='#5a2a16')

GR = [(-.135, -.648), (-.06, -.662), (.02, -.658), (.115, -.628), (.205, -.560)]
EXPR = {
    'calm': dict(lid=.95, brow=(0, 0, 0), mouth=dict(kind='line', pts=[(-.080, -.668), (-.03, -.664), (.02, -.668), (.080, -.658)], lip=[(-.038, -.735), (0, -.742), (.038, -.735)])),
    'point': dict(lid=1.10, brow=(.030, .10, 0), iris=.96, mouth=dict(kind='open', up=[(-.130, -.596), (-.068, -.610), (0, -.616), (.068, -.608), (.136, -.590)], lo=[(-.130, -.596), (-.112, -.700), (-.052, -.782), (.048, -.786), (.114, -.700), (.136, -.590)],
                  teeth=[(-.130, -.616), (-.066, -.638), (0, -.645), (.066, -.636), (.136, -.610)], tongue=[(-.078, -.790), (-.045, -.722), (0, -.712), (.048, -.722), (.078, -.786), (.0, -.83)], lip=[(-.035, -.838), (0, -.845), (.035, -.838)], dark='#5b1a24')),
    'half': dict(lid=1.12, brow=(.030, .04, 0), mouth=dict(kind='open', up=[(-.072, -.630), (0, -.634), (.076, -.626)], lo=[(-.072, -.630), (-.048, -.690), (.0, -.706), (.052, -.688), (.076, -.626)], teeth=[(-.072, -.642), (0, -.650), (.076, -.638)], lip=[(-.035, -.765), (0, -.772), (.035, -.765)], dark='#5b1a24')),
    'smug': dict(lid=(.74, .80), squint=.16, brow=((-.020, .020), (.34, .10), .012), gaze=(0, -.004), iris=.95, mouth=dict(kind='open', up=GR, lo=[(-.135, -.648), (-.085, -.706), (.015, -.736), (.130, -.694), (.205, -.560)], teeth=[(-.135, -.655), (-.075, -.700), (.015, -.722), (.125, -.680), (.205, -.570)],
                 tw='#fbf6f0', dark='#4a1c26', lip=[(-.03, -.792), (.015, -.800), (.065, -.788)], crease=[(.232, -.505), (.250, -.560), (.232, -.612)])),
    'fierce': dict(lid=.94, squint=.20, brow=(-.058, .40, .035), frown=1, gaze=(0, .004), iris=.86, mouth=dict(kind='open', up=[(-.120, -.652), (-.05, -.636), (.03, -.640), (.130, -.628)], lo=[(-.120, -.652), (-.09, -.705), (.02, -.716), (.10, -.694), (.130, -.628)], teeth=[(-.12, -.662), (-.08, -.698), (.02, -.708), (.10, -.684), (.13, -.640)], tw='#f8f2ee', lip=[(-.04, -.775), (0, -.782), (.04, -.775)])),
    'alarm': dict(lid=1.24, brow=(.050, -.06, 0), iris=.72, pupil=.85, mouth=dict(kind='open', up=[(-.070, -.650), (0, -.636), (.070, -.650)], lo=[(-.070, -.650), (-.052, -.740), (0, -.762), (.052, -.740), (.070, -.650)], teeth=[(-.07, -.664), (0, -.658), (.07, -.664)], lip=[(-.04, -.80), (0, -.806), (.04, -.80)])),
    'teary': dict(lid=.82, squint=.26, brow=(.030, -.22, 0), gaze=(0, -.010), tear=1, mouth=dict(kind='line', pts=[(-.112, -.632), (-.055, -.662), (.0, -.670), (.055, -.660), (.114, -.628)], lip=[(-.038, -.740), (0, -.747), (.038, -.740)], tick=.010)),
    'dots': dict(dots=1, brow=(.10, -.06, 0), mouth=dict(kind='open', up=[(-.045, -.668), (0, -.656), (.045, -.668)], lo=[(-.045, -.668), (-.02, -.694), (.02, -.694), (.045, -.668)])),
    'shut': dict(lid=.0, brow=(.01, -.05, 0), mouth=dict(kind='line', pts=[(-.080, -.662), (0, -.672), (.080, -.658)])),
}

ELL = ((0, .20, -.06), (.80, .885, .93))


class NHead(Head):
    """shorter lower face: everything below the nose is drawn up a little"""
    def P3(self, Q):
        Q = np.array(Q, float); y = Q[..., 1]; t = np.clip((-.40 - y) / .60, 0, 1.3); Q[..., 1] = y + .085 * t * t * (3 - 2 * np.minimum(t, 1.)); return Head.P3(self, Q)


def make_head(): return NHead(a=A, nose=(.085, -.09, -.420, -.470, .132), ear=(.688, .06, -.40, -.10), cheek=.036)


def locks(H):
    L = []; n = [0]
    def add(l): L.append(l)
    def fr(th, y, tip, w, **k): n[0] += 1; add(Lock.fr(H, th, y, tip, w, id=n[0], tag=k.pop('tag', 'fr'), **k))
    c0, rr = np.array(ELL[0]), np.array(ELL[1])
    def el(th, e, k=1.): return c0 + rr * k * np.array([math.cos(e) * math.sin(th), math.sin(e), math.cos(e) * math.cos(th)])
    def bob(th, tip, w, k=1.015, e0=1.05, flick=.0, hook=(0, 0, 0), **kw):
        """big clump: crown -> over the widest part -> a hooked tip (x, y, z)"""
        n[0] += 1; tip = np.array(tip, float); q = el(th, -.10, k); m = q * .50 + tip * .50 + np.array([math.sin(th), 0, math.cos(th)]) * (.03 + flick) - np.array(hook, float) * .55
        add(Lock.pa(H, [el(th, e0, .985), el(th, e0 * .62, k), el(th, .25, k), q, m, tip], w, kind=kw.pop('kind', 'flip'), id=n[0], tag=kw.pop('tag', 'sideR' if th > 0 else 'sideL'), **kw))
    def sw(tip, w, part=(.30, 1.12), k=1.03, dz=.07, drop=0., **kw):
        """swept fringe: from the part, over the hairline, to a tip on the face"""
        n[0] += 1; tx, ty = tip; p2 = np.array([tx, ty, float(H.z(np.clip(tx, -.66, .66), ty)) + dz]); th = math.atan2(tx, .62) * .78 + .04
        a = el(part[0], part[1], .985); b = el(part[0] * .70 + th * .30, .88, k); c = el(part[0] * .34 + th * .66, .52, k); c2 = c * .46 + p2 * .54 + np.array([np.sign(tx - part[0] * .8) * .035 + drop, .045, .06])
        add(Lock.pa(H, [a, b, c, c2, p2], w, kind=kw.pop('kind', 'swp'), id=n[0], tag='frR' if tx > .45 else 'frL' if tx < -.45 else 'fr', n=(math.sin(th), .3, math.cos(th)), **kw))
    # back of the head: four clumps, the nape shows between their tips
    bob(3.14, (-.06, -.76, -.74), .62, tag='back', asym=.1); bob(2.50, (.62, -.92, -.60), .54, tag='back', hook=(.10, .02, 0)); bob(-2.50, (-.66, -.84, -.60), .56, tag='back', hook=(-.12, .02, 0)); bob(-1.98, (-.90, -.68, -.36), .40, tag='back', hook=(-.14, .05, 0)); bob(2.0, (.88, -.74, -.36), .40, tag='back', hook=(.13, .04, 0))
    # viewer-left: three unequal clumps, flipped hem, one hook in toward the jaw
    bob(-1.70, (-.930, -.640, -.16), .44, asym=-.12, hook=(-.16, .06, 0)); bob(-1.36, (-.800, -.960, .14), .46, asym=-.10, hook=(-.10, -.02, 0)); bob(-1.08, (-.500, -.800, .50), .30, e0=1.0, hook=(.08, 0, .04), asym=.12)
    bob(-1.22, (-.700, -.600, .34), .16, e0=.98, hook=(-.05, .02, 0))
    # viewer-right: two big ones swept back, a short hooked one by the cheek
    bob(1.72, (.900, -.600, -.22), .44, asym=.12, hook=(.16, .06, 0)); bob(1.36, (.820, -.900, .10), .42, asym=.12, hook=(.14, .0, 0)); bob(1.10, (.610, -.520, .38), .22, e0=.98, hook=(-.04, 0, .03), asym=-.1)
    # thin flyaways
    bob(-1.32, (-.970, -.760, .06), .05, k=1.05, flick=.04, hook=(-.10, .06, 0), kind='ink'); bob(1.52, (.990, -.700, -.06), .05, k=1.05, flick=.04, hook=(.12, .06, 0), kind='ink'); bob(-1.10, (-.430, -.700, .58), .035, hook=(.05, 0, .03), kind='ink')
    # fringe: one big sweep to the viewer's left, a short wing on the right, forehead open between them
    sw((-.640, -.300), .30, asym=-.16); sw((-.500, -.020), .40, asym=-.24); sw((-.270, .130), .34, asym=-.20); sw((-.060, .285), .20, asym=-.10)
    sw((.660, -.040), .20, asym=.14, dz=.03); sw((.530, .150), .28, asym=.18, dz=.04); sw((.350, .280), .14, asym=.06)
    sw((-.150, -.030), .085, dz=.12, asym=-.05, kind='ink'); sw((.060, .110), .065, dz=.12, kind='ink', drop=.03); sw((-.390, -.160), .075, dz=.12, asym=-.05, kind='ink')
    return L


def hl(t): return .56 - .64 * min(abs(t) / 1.15, 1) ** 1.8 - .46 * max(0., min((abs(t) - 1.40) / .8, 1.))


BODY = dict(
    ty=[0, .15, .3, .5, .8, 1.3, 2.0, 2.8, 3.6], ta=[.50, .68, .92, 1.16, 1.26, 1.16, 1.04, .86, 1.0], tf=[.28, .34, .42, .50, .60, .70, .60, .46, .5],
    body_r=[(.34, .04, .18), (.50, -.08, 0), (.76, -.22, -.04), (1.00, -.34, -.05), (1.16, -.48, -.05), (1.15, -.82, 0), (1.06, -1.22, .0), (1.03, -1.60, .0), (.94, -2.10, 0), (.84, -2.60, 0), (.80, -2.9, 0)],
    bc=(4,),
    sleeve_r=[(.84, -.28, -.02), (1.12, -.38, -.04), (1.28, -.52, -.05), (1.38, -.80, -.05), (1.42, -1.12, -.05), (1.44, -1.50, -.05), (1.50, -1.82, -.08), (1.48, -2.20, -.08), (1.52, -2.65, 0), (1.50, -3.3, .1), (1.14, -3.3, .1), (1.08, -2.7, 0), (1.10, -2.25, -.02), (1.02, -1.9, .02), (1.05, -1.52, .05), (.98, -1.20, .10), (.92, -.82, .16), (.84, -.50, .16)],
    sc=(9, 10, 13),
    sleeve_sh=[(1.20, -.4, .1), (1.16, -.9, .1), (1.22, -1.35, .1), (1.15, -1.7, .1), (1.28, -1.85, .1), (1.20, -2.4, .1), (1.3, -3.4, .1), (.9, -3.4, .1), (.8, -1.2, .1), (.9, -.4, .1)],
    sleeve_hi=[[(1.10, -.42, .05), (1.28, -.58, .0), (1.36, -.88, 0), (1.37, -1.18, 0), (1.30, -.95, .05), (1.22, -.68, .05)]],
    sleeve_ln=[[(1.44, -1.74, .0), (1.30, -1.78, .1), (1.16, -1.92, .1)], [(1.02, -1.88, .05), (1.18, -1.80, .1), (1.30, -1.84, .05)], [(1.48, -2.14, .0), (1.32, -2.06, .1), (1.2, -2.10, .1)], [(1.06, -2.06, .05), (1.2, -2.18, .1), (1.28, -2.36, .05)], [(.98, -1.22, .1), (1.10, -1.36, .1), (1.14, -1.58, .1)], [(1.42, -1.44, 0), (1.30, -1.54, .1), (1.2, -1.56, .1)]],
    body_sh=[(.42, .0), (.56, -.40), (.40, -.70), (.64, -.95), (.78, -1.25), (.52, -1.5), (.62, -2.0), (.50, -2.9), (.6, -3.9), (1.8, -3.9), (1.8, 0)],
    dark=[[(-.42, -.06), (-.30, -.28), (0, -.36), (.30, -.28), (.42, -.06), (.0, .0)], [(-.86, -1.50), (-.5, -1.62), (-.12, -1.52), (-.1, -1.62), (-.5, -1.80), (-.9, -1.66)]],
    hi=[[(-.54, -.14), (-.82, -.28), (-1.14, -.45), (-1.00, -.49), (-.78, -.38), (-.66, -.28)], [(.50, -.18), (.80, -.30), (1.02, -.42), (.82, -.40)]],
    seams=[[(.06, -.32), (.05, -1.0), (.06, -2.0), (.08, -3.0)]],
    lines=[[(-1.08, -1.04), (-.88, -1.00), (-.66, -1.08), (-.50, -1.22)], [(-1.03, -1.26), (-.80, -1.28), (-.60, -1.42)], [(1.08, -1.02), (.88, -.98), (.68, -1.06)], [(1.03, -1.28), (.82, -1.32), (.66, -1.46)], [(-.32, -.30), (-.48, -.56), (-.54, -.86), (-.48, -1.16)], [(.36, -.30), (.55, -.52), (.64, -.80)], [(-.86, -1.50), (-.50, -1.62), (-.12, -1.56)], [(.80, -1.52), (.52, -1.64), (.22, -1.60)], [(-.10, -.9), (-.20, -1.3), (-.16, -1.7)]],
    over=[],
    collar=dict(ty=.15, tz=-.04, trx=.50, trz=.46, tilt=.17, by=-.12, bz=-.02, brx=.55, brz=.44, btilt=.10, bow=.0,
                hi=[[(-1.35, .10), (-.95, .16), (-.50, .30), (-.90, .40), (-1.25, .34)], [(.10, .12), (.20, .16), (.20, .85), (.12, .8)]],
                ln=[[(.08, 0), (.09, .5), (.10, 1.0)], [(-.9, .62), (-.5, .7), (-.2, .64)]]),
    cbutton=(.16, -.42), br=.075)


class _Nob:
    def pose(self, cx, cy, s, yaw=0., pitch=0., roll=0., persp=.10): return make_head().pose(cx, cy, s, yaw, pitch, roll, persp)

    def _hair(self, H, f, wind):
        L = locks(H)
        for l in L:
            if (l.tag == 'frL' and H.yaw < -.15) or (l.tag == 'frR' and H.yaw > .15):        # far side of the fringe opens so the far eye reads
                k = min(1., (abs(H.yaw) - .15) / .25); sg = -1 if l.tag == 'frL' else 1; l.w *= 1 - .45 * k; l.path = l.path.copy(); l.path[-1] += np.array([sg * .06, .20, -.10]) * k; l.path[-2] += np.array([sg * .05, .08, -.06]) * k
            l.build(H, wind_off(H, wind, f, (l.id * .618) % 1, .7 if l.tag[:2] == 'fr' else 1.))
        return L

    @staticmethod
    def _isback(l): return l.tag == 'back' or (l.tag[:4] == 'side' and l.vis < -.25)

    @staticmethod
    def _far(l, H): return (H.yaw > .15 and l.tag == 'sideR') or (H.yaw < -.15 and l.tag == 'sideL')

    def back(self, cel, H, light='day', f=0, wind=(0, 0), seed=0, L=None):
        w = lw(H.s); pal = HAIR[light]; LT = LIGHT[light]
        bk = sorted([l for l in (L or self._hair(H, f, wind)) if self._isback(l)], key=lambda l: l.depth)
        front_view = math.cos(H.yaw) > .2
        cols = dict(pal, base=pal['sh'], hi=None, band=None) if front_view else pal
        hair_group(cel, bk, dict(cols, sh=_dk(pal['sh'], .72), cast=np.array([.02, .045]) * H.s), *w, seed=seed, key=LT['key'], caps=[scalp(H, lambda t: -.30, 1.10, front=False, ell=ELL)], line=pal['line'])

    def head(self, cel, H, expr='calm', light='day', f=0, wind=(0, 0), seed=0, parts='bnf', L=None, patch=False, **ex):
        E = dict(EXPR[expr]) if isinstance(expr, str) else dict(expr); E.update(ex)
        LT = LIGHT[light]; w = lw(H.s); pal = HAIR[light]; L = L or self._hair(H, f, wind)
        if 'b' in parts: self.back(cel, H, light, f, wind, seed, L)
        if 'n' in parts: neck(cel, H, LT, w, .285, seed=seed)
        if 'f' not in parts: return cel
        if H.pitch > .05: cel.fill(H.under(), LT['skin2'])
        for s in (-1, 1): ear(cel, H, s, LT, w, seed)
        kx, ky = LT['key']; fl = sorted([l for l in L if not self._isback(l) and self._far(l, H)], key=lambda l: l.depth)
        if fl: hair_group(cel, fl, dict(pal, cast=np.array([-kx * .030 + .01, .050]) * H.s), *w, seed=seed, key=LT['key'], line=pal['line'], inner=.22)
        F = H.face(); cel.fill(F, LT['skin']); (mn.lit_plane if LT.get('lit') else face_shadow)(cel, H, LT, F)
        fr = sorted([l for l in L if not self._isback(l) and not self._far(l, H)], key=lambda l: l.depth)
        kx, ky = LT['key']; off = np.array([-kx * .085, .105]) * H.s
        cel.clip(F)
        for l in fr:
            if l.tag != 'back': cel.fill(l.poly + off, LT['skin2'])
        if E.get('blush'):
            for s in (-1, 1): cel.fill(H.P(np.array([(s * .36 + .13 * math.cos(a), -.27 + .05 * math.sin(a)) for a in np.linspace(0, 6.28, 20)])), LT['blush'], .55)
        cel.unclip()
        planes(cel, H, C, E, LT, F)
        for s in (-1, 1):
            if patch and s == 1: continue
            far = s * H.yaw > .3
            if far: cel.clip(F)
            eye(cel, H, s, C, E, LT, w, seed)
            if far: cel.unclip()
        nose(cel, H, C, LT, w, seed); mouth(cel, H, E['mouth'], LT, w, seed)
        for s in (-1, 1): brow(cel, H, s, C, E, LT, w, seed)
        if E.get('tear'): self._tears(cel, H, E, LT, w, patch)
        if H.yaw * H.yaw > .75 ** 2: cel.fill(H.P3(H.nose_profile()), LT['skin'])
        if patch: self._patch(cel, H, F, LT, w, seed)
        face_line(cel, H, F, w, LT['line'], seed + 2)
        kx, ky = LT['key']; hair_group(cel, fr, dict(pal, cast=np.array([-kx * .030 + .01, .050]) * H.s), *w, seed=seed, key=LT['key'], caps=[scalp(H, hl, 1.1, ell=ELL)], line=pal['line'], inner=.22)
        cel.ctx.push_group()                               # brows read through the fringe
        for s in (-1, 1): brow(cel, H, s, C, dict(E, frown=0), LT, w, seed, col=pal['line'])
        cel.ctx.pop_group_to_source(); cel.ctx.paint_with_alpha(.55)
        return cel

    def _tears(self, cel, H, E, LT, w, patch):
        for s in ((-1,) if patch else (-1, 1)):
            q = spl([(s * .40, -.135), (s * .405, -.25), (s * .37, -.40)], n=8)
            cel.line(H.P(q), w[1] * 2.2, '#ffffff', taper=(.1, .5), alpha=.75, seed=3); cel.line(H.P(q + [s * .012, 0]), w[2], LT['skin2'], taper=(.2, .6), seed=4)

    def _patch(self, cel, H, F, LT, w, seed):
        ex, ey = C['eye']; cel.clip(F)
        q = np.array([(-.215, .085), (-.10, .128), (.05, .140), (.175, .118), (.240, .060), (.215, -.060), (.125, -.165), (.0, -.205), (-.115, -.150), (-.200, -.030)]); P = H.P(spl(np.c_[ex + q[:, 0], ey + q[:, 1]], True, corners=(0, 4), n=8), .02)
        s1 = H.P(spl([(ex - .16, ey + .10), (ex - .42, ey + .30), (ex - .75, ey + .50)], n=8), .02); s2 = H.P(spl([(ex - .13, ey + .14), (ex - .38, ey + .345), (ex - .72, ey + .56)], n=8), .02)
        cel.fill(np.vstack([s1, s2[::-1]]), '#17141c')
        s3 = H.P(spl([(ex + .18, ey - .02), (ex + .30, ey - .10), (ex + .42, ey - .16)], n=8), .02); s4 = H.P(spl([(ex + .19, ey + .05), (ex + .31, ey - .03), (ex + .43, ey - .09)], n=8), .02)
        cel.fill(np.vstack([s3, s4[::-1]]), '#17141c')
        cel.fill(P, '#231f2b'); kx = -1 if LT['key'][0] < 0 else 1
        cel.shade(H.P(np.array([(ex + kx * .02, ey + .2), (ex - kx * .02, ey + .02), (ex + kx * .05, ey - .05), (ex - kx * .01, ey - .25), (ex - kx * .4, ey - .25), (ex - kx * .4, ey + .2)]), .02), '#100e15', within=P)
        cel.fill(H.P(np.array([(ex + kx * .06, ey + .085), (ex + kx * .17, ey + .070), (ex + kx * .175, ey - .03), (ex + kx * .135, ey + .045)]), .02), '#57516a')
        st = P + (P.mean(0) - P) * .10; cel.line(np.vstack([st, st[:1]]), w[2] * .8, '#4a4558', taper=(.02, .02), seed=seed + 2, alpha=.7)
        cel.line(np.vstack([P, P[:2]]), w[1] * 1.3, '#08070a', taper=(.02, .02), seed=seed); cel.unclip()

    def bust(self, cx, cy, s, yaw=0., pitch=0., roll=0., expr='calm', light='day', f=0, wind=(0, 0), seed=0, torso=None, w=1920, h=1080, finish=True, under=None, over=None, patch=False, **ex):
        cel = Cel(w, h); H = self.pose(cx, cy, s, yaw, pitch, roll); LT = LIGHT[light]; W_ = lw(s)
        T = mn.torso_under(H, **(torso or {})); L = self._hair(H, f, wind)
        self.back(cel, H, light, f, wind, seed, L)
        mn.uniform(cel, T, BODY, LT, W_, seed, 'back'); neck(cel, H, LT, W_, .285, seed=seed)
        if under: under(cel, H, T)
        mn.uniform(cel, T, BODY, LT, W_, seed, 'front')
        self.head(cel, H, expr, light, f, wind, seed, parts='f', L=L, patch=patch, **ex)
        if over: over(cel, H, T)
        lay = cel.layer(h, w)
        return mn.grade(lay, LT, (cx - 1.7 * s, cy - 1.4 * s, cx + 1.7 * s, cy + 3 * s), max(.5, s / 200)) if finish else lay


def _dk(c, k):
    from jj.paint import rgb
    return rgb(c) * k


Nobara = _Nob()
