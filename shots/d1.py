"""d1 MEGUMI: hands lock, the shadow boils, the Divine Dog erupts, leaps, jaws at the lens."""
import numpy as np, cv2, math
from jj import comp, W, H
from jj.paint import rgb, f32, fbm, smooth, ramp, grid
from jj.ink import Cel, spl
from jj.kit import para, focus_lines, speed_lines, shifted, shake, smear, zoomblur, outline, splatter
from chars import mn, dog
from chars.mn_keys import licks, boil_layer, hpal, nz
from chars.megumi import Megumi as M, BODY, EXPR

P = dog.PAL['battle']
FAR = dict(P, base='#0b1030', sh='#070a20', hi='#16245f', hi2='#2c58c0', nose='#0d1338', nosehi='#274ba6', gum='#1b1030', gumhi='#2a1846', gumdk='#100a20', tongue='#1c1436', tonguehi='#2a2050', tooth='#7fa2e6', toothsh='#3f5fb4', toothdk='#263c86', mark='#8fb4f4', lip='#080a1c', line='#040616')
dog.PAL['_far'] = FAR
E3 = lambda t, p=3: 1 - (1 - np.clip(t, 0, 1)) ** p


def ink_streaks(seed, ang, n, L=(700, 1700), wd=(50, 190), box=(-300, W + 300, -100, H + 100), bend=.05):
    """body passing the lens: long fur blades along the motion"""
    r = np.random.default_rng(seed); d = np.array([math.cos(ang), math.sin(ang)]); q = np.array([-d[1], d[0]]); out = []
    for i in range(n):
        p = np.array([r.uniform(box[0], box[1]), r.uniform(box[2], box[3])]); l = r.uniform(*L); b = l * bend * r.normal(0, 1); w = wd[0] + (wd[1] - wd[0]) * r.random() ** 1.8
        out.append(dog.Lock.d2([p + d * l * .5, p + d * l * .15 + q * b, p - d * l * .2 + q * b * 1.2, p - d * l * .5 + q * b * .4], w, 'ink', r.uniform(-.2, .2), seed * 100 + i, depth=r.random()))
    return sorted(out, key=lambda l: l.depth)


def thrown(seed, cx, cy, n, r0, r1, f, col='#05060f', hot='#8cc8ff', sz=(3, 16), asp=.62, p_hot=.25, v=1.0):
    """ink drops flung out of a point, stretched along their flight"""
    c = Cel(); r = np.random.default_rng(seed)
    for _ in range(n):
        a = r.uniform(0, 6.283); d0 = (r0 + (r1 - r0) * r.random()) * (.5 + v * .5); p = np.array([cx + math.cos(a) * d0, cy + math.sin(a) * d0 * asp]); e = np.array([math.cos(a), math.sin(a) * asp]); e /= np.hypot(*e); q = np.array([-e[1], e[0]]); s = sz[0] + (sz[1] - sz[0]) * r.random() ** 2.2
        c.fill(spl([p - e * s * 2.2, p + q * s * .42, p + e * s * 1.1, p - q * s * .42], True, corners=(0,), n=5), hot if r.random() < p_hot else col)
    return c.layer()


class Shot:
    id = 'd1'; n = 72
    finish = dict(bloom=.34, vig=.20)
    EVENTS = [(0, 'cloth_whoosh', dict(size=.5, note='two hands snap together, sleeves')), (3, 'hand_lock', dict(size=.8, note='palms slap, dry crack, low boom under it')), (9, 'shadow_boil', dict(len=18, size=.8, note='wet bubbling rumble rising until 27')),
              (18, 'eyes_ignite', dict(size=.7, note='two pale eyes open in the dark behind him, sharp breath of the beast')), (21, 'growl', dict(len=6, size=.8)), (27, 'impact', dict(size=1.0, note='the dog erupts past him, air torn open')),
              (28, 'fur_whoosh', dict(len=4, size=1.0, pan='L>R')), (31, 'paw_hit', dict(size=.5, note='bursts out of the dust far down the street')), (33, 'paw_land', dict(size=.6, note='claws on asphalt, rubble')), (37, 'paw_land', dict(size=.9, note='nearer, road cracks')), (39, 'paw_hit', dict(size=.7, note='push-off')), (41, 'leap', dict(size=.8, note='push-off, whoosh toward us')),
              (45, 'roar', dict(len=17, size=1.0, note='jaws at the lens, full roar')), (45, 'impact', dict(size=.8)), (54, 'roar_accent', dict(size=.8, note='second, deeper push of the roar')), (60, 'inhale', dict(len=3, note='sound sucked out')), (63, 'bite', dict(size=1.0, note='jaws slam shut: hardest hit of the shot')), (65, 'silence', dict(len=7, note='black, one low tail'))]

    def setup(self):
        x, y = grid(); self.C = {}
        n = fbm(H, W, (420, 160, 60), seed=8); d = np.sqrt(((x - 1120) / 1150) ** 2 + ((y - 760) / 820) ** 2)
        self.bg2 = ramp(np.clip(d + n * .05, 0, 1.4), [(0, '#5f9af0'), (.22, '#2f56c4'), (.5, '#182b78'), (.85, '#0b1136'), (1.4, '#070819')])
        d = np.sqrt(((x - 930) / 1150) ** 2 + ((y - 640) / 760) ** 2); self.bg1 = ramp(np.clip(d + n * .05, 0, 1.4), [(0, '#3f74de'), (.3, '#1f3fa4'), (.6, '#111d58'), (1., '#080b26'), (1.4, '#050614')])
        d = np.sqrt(((x - 1010) / 1250) ** 2 + ((y - 420) / 820) ** 2); nn = fbm(H, W, (300, 120, 50), seed=5)
        self.bg5 = ramp(np.clip(d + nn * .05, 0, 1.3), [(0, '#5a92f0'), (.25, '#2a4fc0'), (.55, '#15246e'), (.9, '#090d2a'), (1.3, '#05060f')])
        from jj import world
        p = world.plate('ruin', mode='night', view='low', w=2400, h=1350); im = p.img.astype(f32); l = np.clip(im[..., 0] * .35 + im[..., 1] * .45 + im[..., 2] * .30, 0, 1.2); g = ramp(np.clip(l * 1.75, 0, 1.2), [(0, '#02030a'), (.14, '#060b26'), (.30, '#0f1f62'), (.50, '#2350b8'), (.75, '#5f9cf4'), (1.2, '#d4eaff')])
        warm = cv2.dilate(smooth(.04, .16, im[..., 0] - im[..., 2]).astype(f32), np.ones((9, 9), np.uint8)); warm = cv2.GaussianBlur(warm, (0, 0), 4)[..., None]; yy, xx = np.mgrid[0:1350, 0:2400].astype(f32); vg = np.clip(1.12 - .62 * (((xx - 1210) / 1150) ** 2 + ((yy - 800) / 760) ** 2), .22, 1.12)[..., None]; self.street = np.asarray((g * .82 + im * 1.3 * .18) * (1 - warm) * vg + rgb('#0b1640') * warm * vg * .8, f32)

    def cached(self, k, fn):
        if k not in self.C:
            if len(self.C) > 8: self.C.pop(next(iter(self.C)))
            self.C[k] = fn()
        return self.C[k]

    def frame(self, f):
        if f < 9: return self.hands(f)
        if f < 31: return self.boil(f)
        if f < 45: return self.leap(f)
        if f < 63: return self.jaws(f)
        return self.bite(f)

    # ---------------------------------------------------------------- 0-9 hands
    def hands(self, f):
        from jj import hands as HD
        L = mn.LIGHT['battle']; w = mn.lw(300); k = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}.get(f, 5)
        gap, rot, sx = [(1.55, 30, 250), (.95, 20, 190), (.36, 9, 120), (-.02, -3, 0), (.0, 2, 0), (.0, 0, 0)][k]; jit = {3: (0, 12), 4: (0, -5)}.get(f, (0, 0))
        def base():
            s = 540; cel = Cel(); Hd = M.pose(1090, -70, s, -.22, -.06, -.05); T = mn.torso_under(Hd, yaw=-.18, roll=.04, drop=1.10); ww = mn.lw(s); hair = M._hair(Hd, 0, (0, 0))
            mn.uniform(cel, T, BODY, L, ww, 3, 'back'); mn.neck(cel, Hd, L, ww, .335, seed=3); mn.uniform(cel, T, BODY, L, ww, 3, 'front'); M.head(cel, Hd, EXPR['focus'], 'battle', 0, (0, 0), 3, parts='f', L=hair)
            return comp.bokeh(mn.grade(cel.layer(), L, (500, -300, 1600, 1080), 1.4), 10, 0)
        body = self.cached('b1', base)
        def drawhands():
            hc = Cel(); pal = hpal(L); cam = HD.Cam(930 + jit[0], 600 + jit[1], 318, 6.2)
            far = HD.Hand('dog_far', 'R', HD.euler(170, 6, 66 + rot), (.03 - gap * .95, .02 - gap * .25, -.17), HD.SLIM); near = HD.Hand('dog_near', 'L', HD.euler(-10, 6, 66 - rot), (gap * 1.05, -gap * .18, .165), HD.SLIM)
            W_ = lambda h, y_: cam.px(h.world(np.array([0, y_, 0])))
            a = W_(far, -.10); g1 = [far, near]
            if gap > .2:
                for h, s_, sd in ((far, 1, 21), (near, 1, 33)):
                    g = HD.render([h], cam, light=(-.75, -.25, .55)); HD.ink(hc, g, pal, w[0] * .9, 3, inner=.6); b = W_(h, -.28); e = W_(h, -1.4); mn.sleeve2(hc, [b, (b + e) / 2 + [0, 14], e + [0, 40], e + (e - b) * .8], [(0, 106), (.4, 150), (1, 210)], L, w, sd, bunch=(.2,))
            else:
                mn.sleeve2(hc, [a + [46, 12], (930, 960), (700, 1180), (520, 1330)], [(0, 108), (.5, 160), (1, 210)], L, w, 21, bunch=(.18,))
                g = HD.render(g1, cam, light=(-.75, -.25, .55)); HD.ink(hc, g, pal, w[0] * .9, 3, inner=.6)
                b = W_(near, -.28); e = W_(near, -1.4); e2 = W_(near, -2.7); mn.sleeve2(hc, [b, e + [6, 16], e2 + [14, 30], e2 + (e2 - e) + [0, 60]], [(0, 104), (.3, 150), (.6, 185), (1, 230)], L, w, 33, bunch=(.14,))
            return mn.grade(hc.layer(), L, (400, 200, 1500, 1100), .8)
        hl = self.cached(('h', k), drawhands)
        im = comp.over(self.bg1.copy(), body); sh = cv2.GaussianBlur(shifted(hl[..., 3], 26, 40), (0, 0), 7) * body[..., 3]; im = im * (1 - sh[..., None] * .55 * (1 - rgb('#7a70b4')))
        if f < 3:
            c = Cel(); speed_lines(c, .12, 34, 7, (400, 1500), '#8fc2ff', 10 + f, alpha=.75); speed_lines(c, .12, 30, 10, (400, 1500), '#05071a', 20 + f); im = comp.over(im, c.layer(), .55); hl = .55 * smear(hl, sx, 30, 9) + .45 * hl
        g = np.clip((f - 2) / 5., 0, 1); fk = f // 2 * 2
        if f >= 3:
            im = comp.over(im, boil_layer([licks(-90, 2000, 1130, 330, 9, 14, lean=.06, f=fk, boil=.30, grow=.35 + .65 * g)], P['base'], '#2f4aa8', P['line'], 3.0, (18, 16), hi2='#5d8fe6', rimc='#7cc6ff', drops=int(10 * g), seed=14 + fk, box=(100, 1800, 300, 900)))
        im = comp.over(im, hl)
        if f == 3:                                           # contact: white wedges, one drawing
            c = Cel(); focus_lines(c, 880, 560, 330, 1700, 90, 16, '#eaf4ff', 5, ry=.7); focus_lines(c, 880, 560, 420, 1700, 70, 12, '#050818', 6, ry=.7); im = comp.over(im, c.layer(), .9); im = im + comp.radial(880, 560, 520, 420, 1.6)[..., None] * rgb('#9fd0ff') * .5
        if f in (4, 5): im = im + comp.radial(880, 560, 620, 460, 1.8)[..., None] * rgb('#5fa4ff') * (.30 if f == 4 else .14)
        im = comp.screen(im, comp.radial(930, 1120, 1100, 480, 2.2)[..., None] * rgb('#2c62e0') * (.2 + .3 * g))
        if f >= 4: im = comp.over(im, thrown(40 + fk, 900, 700, 16, 250, 900, f, sz=(3, 13)))
        dx, dy = shake(f, 9, 3, 2.2, 3) if f >= 3 else (0, 0); z = 1.03 + .018 * E3(f / 9.)
        return comp.cam2d(im, W / 2 + dx, H / 2 + dy, z, .004 * math.sin(f))

    # ---------------------------------------------------------------- 9-31 the pool boils, eyes open, the dog bursts past
    def megumi(self, k, wind, expr, f=0):
        L = mn.LIGHT['battle']; s = 262; cx, cy = 1190, 440; cel = Cel(); Hd = M.pose(cx, cy, s, -.30, -.10, -.06); w = mn.lw(s); hair = M._hair(Hd, k, wind); T = mn.torso_under(Hd, yaw=-.22, roll=.05, drop=1.10)
        M.back(cel, Hd, 'battle', k, (0, 0), 3, hair); mn.uniform(cel, T, BODY, L, w, 3, 'back'); mn.neck(cel, Hd, L, w, .335, seed=3); mn.uniform(cel, T, BODY, L, w, 3, 'front')
        M.head(cel, Hd, expr, 'battle', k, (0, 0), 3, parts='f', L=hair)
        return mn.grade(cel.layer(), L, (cx - 420, 60, cx + 420, 1080), 1.0)

    def sign(self):
        from jj import hands as HD
        L = mn.LIGHT['battle']; w = mn.lw(262); hc = Cel(); pal = hpal(L); cam = HD.Cam(985, 772, 190, 7.5)
        far = HD.Hand('dog_far', 'R', HD.euler(170, 6, 68), (0.03, .02, -.17), HD.SLIM); near = HD.Hand('dog_near', 'L', HD.euler(-10, 6, 68), (0, 0, .165), HD.SLIM); W_ = lambda h, y_: cam.px(h.world(np.array([0, y_, 0])))
        a = W_(far, -.10); mn.sleeve2(hc, [a + [26, 6], (1010, 960), (900, 1110), (800, 1260)], [(0, 66), (.5, 100), (1, 130)], L, w, 21, bunch=(.2,))
        g = HD.render([far, near], cam, light=(-.75, -.25, .55), clip=(500, 250, 1500, 1080)); HD.ink(hc, g, pal, w[0] * .8, 3, inner=.6)
        b = W_(near, -.28); e = W_(near, -1.4); e2 = W_(near, -2.7); mn.sleeve2(hc, [b, e + [4, 10], e2 + [8, 18], e2 + (e2 - e) + [0, 40]], [(0, 64), (.3, 92), (.6, 114), (1, 142)], L, w, 33, bunch=(.14,))
        return mn.grade(hc.layer(), L, (600, 400, 1500, 1100), .62)

    def boil(self, f):
        fk = f // 2 * 2; f3 = f // 3 * 3; t = f - 9
        grow = .10 + .95 * float(E3(t / 13., 2)); burst = f >= 27
        rise = 560 * (1 - float(E3((fk - 10) / 11., 3))); eyes = 1. if f >= 18 else 0.; jaw = .05 + .40 * float(E3((fk - 20) / 6., 2))
        im = self.bg2.copy()
        if not burst:
            def wolf():
                c = Cel(); dy = rise; dog.fur_fill(c, dog.mane(430, 360 + dy, 300, 270, 30, 470, 170, 12, swirl=.28 + .05 * math.sin(fk * .5)), FAR, 3.0, 4, wedge=.5)
                D = dog.DogHead().pose(455, 330 + dy, 225, .42, .02, .05, jaw=jaw, persp=.28); dog.head(c, D, '_far', fk, 3, w=2.8, eyes=eyes, wind=(.03, -.06, .6)); return dog.rim2(c.layer(), -4, 5, '#3f7be8', .7, .8), [D.P3(np.array([(q * .44, .40, .44)]))[0] for q in (-1, 1)]
            if f >= 11:
                lay, ey = self.cached(('w', fk), wolf); im = comp.over(im, comp.blur(lay, 1.2))
                if eyes:
                    k = 1.5 if f in (18, 19) else 1.0
                    for e in ey: im = comp.glow(im, e[0], e[1], 210 * k, '#5fb0ff', min(1., .75 * k), 2.4)
        im = comp.over(im, boil_layer([licks(-80, 2000, 985, 340, 15, 5, lean=.10 + .5 * burst, f=fk, boil=.28, grow=grow)], '#0b1136', '#1d378f', '#050819', 2.4, (14, 12), rimc='#4d93f2', drops=int(14 * grow), seed=5 + fk, box=(700, 1900, 250, 760)))
        # hair and collar blown upward, then torn sideways by the dog
        g = min(1., t / 8.); w3 = f3 if not burst else f
        wind = (.10, -.10 - .20 * g, .45) if not burst else ((.55, -.16, .35) if f == 27 else (.70, -.02, .3) if f == 28 else (.62, -.06, .3))
        ex = dict(EXPR['focus'], gaze=(-.012, .004), lid=.92 - .10 * smooth(16, 18, f) + (.22 if burst else 0)) if True else None
        body = self.cached(('m', w3, burst), lambda: self.megumi(w3, wind, ex)); hl = self.cached('sign', self.sign)
        yb = int(round(3 * math.sin(f3 * .7)))
        im = comp.over(im, shifted(body, 0, yb) if yb else body); sh = cv2.GaussianBlur(shifted(hl[..., 3], 16, 26), (0, 0), 2.0) * body[..., 3]; im = im * (1 - sh[..., None] * (1 - rgb('#8478b4'))); im = comp.over(im, shifted(hl, 0, yb) if yb else hl)
        im = comp.over(im, boil_layer([licks(-90, 860, 975, 300, 8, 14, lean=.28 + .4 * burst, f=fk, boil=.30, grow=grow), licks(1320, 2020, 1000, 300, 6, 23, lean=-.18 + .6 * burst, f=fk, boil=.30, grow=grow), licks(500, 1700, 1110, 110, 11, 31, lean=.05, f=fk, boil=.4, grow=.4 + .6 * grow)], P['base'], '#2f4aa8', P['line'], 3.0, (18, 16), hi2='#5d8fe6', rimc='#7cc6ff', drops=int(16 * grow), seed=14 + fk, box=(60, 900, 420, 900)))
        im = comp.screen(im, comp.radial(1010, 1120, 1000, 420, 2.2)[..., None] * rgb('#2c62e0') * (.2 + .25 * grow))
        im = comp.over(im, comp.bokeh(boil_layer([licks(-200, 420, 1100, 300, 3, 41, lean=.35, f=fk, boil=.25, grow=grow), licks(1560, 2150, 1130, 300, 3, 47, lean=-.35, f=fk, boil=.25, grow=grow)], '#05060f', '#0f1c4a', '#030409', 2, (26, 20), drops=5, seed=44, box=(1500, 1900, 500, 900)), 9, 0))
        if not burst:                                        # ink specks sucked upward
            c = Cel(); r = np.random.default_rng(12)
            for i in range(46):
                x0, y0, v, sz = r.uniform(0, W), r.uniform(0, 1300), r.uniform(10, 34), 2 + 8 * r.random() ** 2.6; y_ = 1180 - ((y0 + v * t) % 1300); p = np.array([x0 + 24 * math.sin(t * .3 + i), y_]); c.fill(spl([p - [0, sz * 2.4], p + [sz * .42, 0], p + [0, sz], p - [sz * .42, 0]], True, corners=(0,), n=5), '#05060f' if r.random() < .7 else '#8cc8ff')
            im = comp.over(im, c.layer(), min(1., t / 4.))
            z = 1.03 + .05 * (t / 18.); sx, sy = (shake(f, 1.2 + 2.2 * grow, 5)); fl = .22 if f == 18 else .10 if f == 19 else 0
            im = im + fl * rgb('#4f8cff') * comp.radial(520, 240, 900, 600, 1.5)[..., None] if fl else im
            return comp.cam2d(im, W / 2 + sx - 10 * (z - 1) * 10, H / 2 + sy, z, .006 * (t / 18.))
        # the dog tears past the lens, four smear drawings on ones
        k = f - 27; c = Cel()
        if k == 0: sp = dict(path=[(60, 980), (90, 700), (300, 520), (520, 450)], width=[(0, 330), (.4, 280), (.75, 235), (1, 190)], ln=1.35, spread=.75, head=dict(cx=690, cy=440, s=285, yaw=.34, pitch=.02, roll=.03, jaw=.72, persp=.33), mane=dict(rx=310, ry=270, n=26, ln=480, w=170, swirl=.35, a0=1.5, a1=5.6)); mv = (230, 40)
        elif k == 1: sp = dict(path=[(-200, 900), (60, 640), (420, 520), (740, 500)], width=[(0, 470), (.4, 420), (.75, 360), (1, 290)], ln=1.4, spread=.8, head=dict(cx=1000, cy=520, s=470, yaw=.22, pitch=.02, roll=.06, jaw=.95, persp=.37), mane=dict(rx=500, ry=420, n=28, ln=700, w=260, swirl=.35, a0=1.5, a1=5.6), legs=[dict(path=[(640, 800), (900, 960), (1180, 930), (1370, 985)], width=[(0, 190), (.5, 150), (1, 120)], paw=(220, .5))]); mv = (380, 80)
        if k < 2:
            lay = self.cached(('e', k), lambda: dog.draw(sp, 'battle', f, 3, w=3.6)); sm = smear(shifted(lay, -mv[0] * .45, -mv[1] * .45), mv[0], mv[1], 11)
            im = comp.over(im, sm, .9); im = comp.over(im, lay)
            c = Cel(); speed_lines(c, math.atan2(mv[1], mv[0]), 40, 8, (300, 1300), '#a9d6ff', 70 + k, box=(-200, 150, W, H), alpha=.85); im = comp.over(im, c.layer(), .7)
            D = dog.DogHead().pose(**sp['head'])
            for q in (-1, 1):
                e = D.P3(np.array([(q * .44, .40, .44)]))[0]; im = comp.screen(im, smear(comp.radial(e[0] - mv[0] * .4, e[1] - mv[1] * .4, 90, 60, 1.6), mv[0] * .8, mv[1] * .8, 9)[..., None] * rgb('#7fc8ff') * 1.2)
        else:
            def wall():
                c = Cel(); c.fill(np.array([(0, 0), (W, 0), (W, H), (0, H)]), P['sh'], .0); r = np.random.default_rng(90 + k)
                Ls = ink_streaks(90 + k, .24, 64 if k == 2 else 54, (800, 2100), (70, 300), box=(-200, W + 200, 120 if k == 2 else -100, H + 200)); dog.fur_fill(c, Ls, P, 4.0, 5, wedge=.7); return dog.rim2(c.layer(), 5, 7, '#79c4ff', .95, .7)
            lay = self.cached(('s', k), wall); im = comp.over(im, smear(lay, 150, 38, 9) * .5 + lay * .5) if k == 2 else comp.over(im * .35, smear(lay, 420, 100, 13) * .65 + lay * .35)
            c = Cel(); speed_lines(c, .24, 50, 7, (500, 1800), '#bfe2ff', 80 + k, alpha=.9); im = comp.over(im, c.layer(), .8)
        dx, dy = shake(f, 16, 9, 3.2, 27)
        return comp.cam2d(im, W / 2 + dx, H / 2 + dy, 1.09, .012 * (-1) ** f)

    # ---------------------------------------------------------------- 31-45 down the ruined street
    LEAP = [(31, 'leap', 0, .33, 1120, 560, 1, .50), (33, 'gather', 0, .56, 1010, 618, 0, .85), (35, 'leap', 1, .62, 900, 530, 1, .62), (37, 'gather', 1, .84, 1000, 632, 0, .95), (39, 'leap', 0, .92, 1060, 500, 1, .74), (41, 'leap', 0, 1.14, 1010, 480, 1, .92), (43, 'leap', 0, 1.55, 975, 520, 1, 1.)]

    def leap(self, f):
        i = [k for k, d in enumerate(self.LEAP) if d[0] <= f][-1]; f0, key, flip, sc, ox, oy, air, spr = self.LEAP[i]; u = f - f0; sc *= 1 + .05 * u * (1.6 if air else .6)
        z = .90 + .020 * (f - 31) / 14.; dx, dy = shake(f, {31: 5, 33: 9, 37: 14}.get(f0, 3.0) * (1 if u == 0 else .6), 21)
        bg = comp.cam2d(self.street, 1200 + dx * .6, 742 + dy * .6, z)
        x, y = grid(); bg = comp.screen(bg, comp.radial(1010, 500, 900, 420, 1.8)[..., None] * rgb('#2f6cf0') * .50); bg = bg + comp.radial(1010, 560, 520, 190, 1.8)[..., None] * rgb('#8fc6ff') * .42
        bg = comp.screen(bg, (smooth(470, 600, y) * smooth(860, 620, y))[..., None] * rgb('#2a5ad0') * (.30 + .10 * nz((260, 90), (1, .5), 71))[..., None])
        # the shadow it drags: brush trails on the road from the vanishing point to its feet
        fk = f // 2 * 2; r = np.random.default_rng(60 + i); tr = []; hw = 430 * sc; gy = oy + (330 if air else 355) * sc
        for q in range(15):
            e = np.array([ox + r.uniform(-1, 1) * hw, gy + r.uniform(-60, 90) * sc]); v = np.array([1010, 590]); m = v * .45 + e * .55 + r.normal(0, 26, 2) * sc; tr.append(dog.Lock.d2([e, e * .72 + m * .28 + r.normal(0, 9, 2), m, v * .82 + e * .18], r.uniform(26, 110) * sc, 'ink', r.uniform(-.2, .2), 700 + i * 20 + q))
        c = Cel(); dog.fur_fill(c, tr, dict(P, base='#060812', hi='#203a94', hi2='#4f8cf0'), 2.0, 3, wedge=.5); bg = comp.over(bg, c.layer(), .92)
        sp = dog.scaled(dog.SPEC[key], sc, (W - ox) if flip else ox, oy, spread=spr)
        lay = self.cached(('d', i, u), lambda: dog.draw(sp, 'battle', f, 3 + i, w=1.4 + 2.2 * min(sc, 1.1), rimd=(3 + 3 * sc, 3 + 2 * sc)))
        if flip: lay = lay[:, ::-1]
        # ground shadow
        if air: sh = comp.blur((((x - ox) / (560 * sc + 60)) ** 2 + ((y - (oy + 430 * sc + 30)) / (70 * sc + 12)) ** 2 < 1).astype(f32), 14); bg = bg * (1 - .65 * sh[..., None])
        from jj import fx
        if i >= 1:                                           # thrown road of the last push-off
            c = Cel(); pf = [d for d in self.LEAP[:i + 1] if d[1] == 'gather']
            for d in pf:
                px = W - d[4] if False else d[4]; fx.debris(c, f - d[0], (px, d[5] + 400 * d[3]), n=int(16 + 34 * d[3]), seed=d[0], speed=(320 * d[3] + 120, 1350 * d[3] + 200), size=(3, 70 * d[3] + 6), grav=1700 * d[3] + 300, pal=fx.NIGHTROCK, ang=(-2.8, -.35), floor=None)
            bg = comp.over(bg, c.layer())
            c = Cel()
            for d in pf: fx.dust_skirt(c, f - d[0] + 1, d[4], d[5] + 380 * d[3], 430 * d[3], 90 * d[3], 210 * d[3], seed=d[0], n=22, pal=dict(lit='#5f86dc', mid='#263f8e', dark='#101a4a', ink='#060818'), life=1.0)
            bg = comp.over(bg, c.layer(), .85)
        if air: sm = smear(lay, 0, 70 * sc + 20, 7); bg = comp.over(bg, shifted(sm, 0, -50 * sc), .45)
        im = comp.over(bg, lay)
        im = comp.over(im, thrown(120 + f, ox, oy + 60, int(14 + 22 * sc), 240 * sc, 900 * sc + 120, f, sz=(2, 6 + 10 * sc)))
        if f >= 41:
            c = Cel(); focus_lines(c, ox, oy - 20, 560 + 40 * (f % 2), 1700, 90, 9, '#9fcbff', 5 + f, ry=.62); im = comp.over(im, c.layer(), .42 if f < 43 else .62)
        if f >= 43: im = zoomblur(im, ox, oy, .06 + .05 * u, 9) * .6 + im * .4
        return comp.cam2d(im, W / 2 + dx, H / 2 + dy, 1.04, .012 * math.sin(f * 1.3))

    # ---------------------------------------------------------------- 45-63 jaws
    def jawpose(self, f):
        t = f - 45
        jaw = [.42, .74, .98][t] if t < 3 else .86 + .03 * math.sin((t // 2) * 1.1) if t < 8 else [.70, 1.05, .99][min(t - 8, 2)] + .02 * math.sin(t)     # second roar on the beat (f54)
        s = [300, 360, 352][t] if t < 3 else 336 + 3.0 * (t - 3) + (-10 if t == 8 else 26 * math.exp(-max(0, t - 9) / 3.) if t > 8 else 0) + max(0, t - 12) ** 2 * 7.5
        cx = 1010 + 8 * math.sin((t // 2) * .9); cy = 382 + (10 if t == 0 else 0) - max(0, t - 12) * 9 + 5 * math.sin((t // 2) * 1.4)
        if t >= 15: j = min(t - 15, 2); jaw = [.97, .70, .34][j]; s = [400, 430, 470][j]; cy = [300, 170, 60][j]
        return dict(cx=cx, cy=cy, s=s, yaw=-.16 + .02 * math.sin((t // 2) * .8), pitch=-.05 + .02 * math.sin((t // 2) * 1.2), roll=.14 - .012 * t * .6, jaw=jaw, persp=.445)

    def jaws(self, f):
        t = f - 45; tk = t if t < 4 or t in (8, 9, 10) or t >= 15 else t // 2 * 2; fk = 45 + tk; ps = self.jawpose(fk); sw = .05 * math.sin(tk * .6)
        def lines():
            fl = Cel(); focus_lines(fl, 1010, 420, 620, 1700, 120, 9, '#8fc2ff', 3 + tk, ry=.62); focus_lines(fl, 1010, 420, 520, 1700, 150, 12, '#05071a', 4 + tk, ry=.62); return fl.layer()
        im = comp.over(self.bg5, self.cached(('l', tk), lines), .75)
        def far():
            c = Cel(); pal = dict(P, base='#0a0d22', sh='#060816', hi='#182766', hi2='#2d53b0'); dog.fur_fill(c, dog.mane(1000, 380, 760, 520, 34, 600, 230, 4, swirl=.30 + sw), pal, 3.2, 1, wedge=.45); return dog.rim2(c.layer(), 5, 6, '#5aa8ff', .8, .8)
        im = comp.over(im, self.cached(('f', tk // 4), far)); im = comp.screen(im, comp.radial(1000, 120, 1100, 520, 2.)[..., None] * rgb('#1b3f9f') * .5)
        def near():
            c = Cel(); k = ps['s'] / 345.; dog.fur_fill(c, dog.mane(1000, 420, 620 * k, 430 * k, 30, 470 * k * (1 + .12 * math.exp(-max(0, t - 9) / 3.) * (t >= 9)), 190, 9, swirl=-.22 + sw * 1.4), P, 3.4, 2, wedge=.6)
            D = dog.DogHead().pose(**ps); dog.head(c, D, 'battle', fk, 3, w=3.8, wind=(0, -.05, .9)); lay = para(c.layer(), (500, 0), (1300, 1080), [(0, '#f4f8ff'), (.45, '#ffffff'), (1, '#8a86c8')], 'mul', .8); return dog.rim2(lay, 6, 5, '#79c4ff', .9, .7)
        lay = self.cached(('n', tk), near); D = dog.DogHead().pose(**ps)
        if t == 0: lay = zoomblur(lay, 1000, 480, .28, 11) * .7 + lay * .3
        elif t >= 16: lay = zoomblur(lay, 1000, 620, .10 * (t - 15), 9) * .55 + lay * .45
        im = comp.over(im, lay)
        for q in (-1, 1):
            e = D.P3(np.array([(q * .44, .40, .44)]))[0]; im = comp.glow(im, e[0], e[1], 250, '#3f8cff', .5, 2.6)
        # saliva string that stretches, breath, flung ink
        c = Cel(); a = D.P3(np.array([(.205, -.60, 1.30)]))[0]; b = D.P3(D.low(np.array([(.245, -.06, 1.25)])))[0]
        if t < 14: m = (a + b) / 2 + [14 * math.sin(t * .8), 26]; c.line(spl([a, m, b], n=12), 4.2, '#cfe0ff', taper=(.05, .05), wfun=lambda v: .35 + 1.3 * (2 * v - 1) ** 2, alpha=.8, seed=3)
        im = comp.over(im, c.layer())
        u = (tk % 8) / 8.; im = comp.over(im, comp.bokeh(thrown(200 + tk // 8, 1000, 520, 26, 300 + 500 * u, 700 + 700 * u, f, sz=(4, 20), p_hot=.05), 3.0, 0)); s_ = Cel(); r = np.random.default_rng(300 + tk)
        for _ in range(34):
            a_ = r.uniform(0, 6.283); d0 = r.uniform(380, 1100); p = np.array([1000 + math.cos(a_) * d0, 500 + math.sin(a_) * d0 * .6]); e = np.array([math.cos(a_), math.sin(a_) * .62]); L_ = r.uniform(10, 60); s_.line(np.array([p, p + e * L_ * .5, p + e * L_]), r.uniform(1.2, 3.4), '#bfe4ff', taper=(.5, .2), seed=int(a_ * 99))
        im = im + s_.layer()[..., :3] * 1.1
        if t >= 12: im = im * (1 - .5 * smooth(12, 18, t) * (1 - comp.radial(1000, 560, 1100, 700, .6))[..., None])     # light dies into the throat
        dx, dy = shake(f, 20, 33, 4.0, 45); q = shake(f, 2.6, 34); e9 = shake(f, 11, 35, 2.4, 54); q = q + e9 if f >= 54 else q; rl = .016 * math.exp(-t / 3.) * (-1) ** t
        return comp.cam2d(im, W / 2 + dx + q[0], H / 2 + dy + q[1], 1.05, rl)

    # ---------------------------------------------------------------- 63-72 bite
    def bite(self, f):
        t = f - 63
        if t < 2:
            bg, ink, fl, sd, dk = (('#f4f9ff', '#04050c', '#ffffff', '#8fb2f4', '#2b5fe0'), ('#04050c', '#f4f9ff', '#04050c', '#12245e', '#2b5fe0'))[t]; c = Cel(); c.fill(np.array([(-200, -200), (W + 200, -200), (W + 200, H + 200), (-200, H + 200)]), bg)
            focus_lines(c, 960, 560, 150 + 120 * t, 1900, 150, 30, ink, 11 + t, ry=.5, clump=.9); focus_lines(c, 960, 560, 420, 1900, 60, 16, dk, 21 + t, ry=.5, clump=.9)
            ro = lambda q: (np.asarray(q, float) - [960, 560]) @ np.array([[math.cos(.11), math.sin(.11)], [-math.sin(.11), math.cos(.11)]]) + [960, 560]
            up = [(-60, 250, 360), (170, 190, 250), (430, 320, 560), (700, 170, 250), (860, 170, 230), (1030, 170, 250), (1300, 330, 600), (1590, 210, 320), (1800, 260, 400), (2010, 220, 300)]; lo = [(60, 230, 330), (300, 190, 270), (570, 290, 520), (790, 150, 220), (945, 150, 210), (1130, 200, 280), (1430, 290, 500), (1690, 210, 330), (1910, 230, 340)]
            def fang(x, wd, ln, yb, sg, sd_):
                P0 = spl([(x - wd * .52, yb - 60 * sg), (x - wd * .50, yb), (x - wd * .30, yb + sg * ln * .45), (x - wd * .02, yb + sg * ln)], n=8); P1 = spl([(x - wd * .02, yb + sg * ln), (x + wd * .34, yb + sg * ln * .40), (x + wd * .50, yb), (x + wd * .52, yb - 60 * sg)], n=8); Q = ro(np.vstack([P0, P1]))
                outline(c, Q, 13, ink, sd_); c.fill(Q, fl); m = spl([(x + wd * .04, yb - 60 * sg), (x + wd * .10, yb + sg * ln * .2), (x + wd * .04, yb + sg * ln * .6), (x - wd * .02, yb + sg * ln)], n=8); c.fill(ro(np.vstack([m, P1])), sd); m2 = spl([(x + wd * .30, yb - 60 * sg), (x + wd * .30, yb + sg * ln * .25), (x + wd * .12, yb + sg * ln * .7), (x - wd * .02, yb + sg * ln)], n=8); c.fill(ro(np.vstack([m2, P1])), dk)
            for i, (x, wd, ln) in enumerate(lo): fang(x, wd, ln, 840, -1, 40 + i)
            gum = spl([(-200, 830), (60, 880), (300, 850), (570, 900), (790, 850), (945, 866), (1130, 850), (1430, 905), (1690, 850), (1910, 880), (2150, 830)], n=8); c.fill(ro(np.vstack([gum, [(2200, 1400), (-200, 1400)]])), ink if t == 0 else '#0a1a5a')
            for i, (x, wd, ln) in enumerate(up): fang(x, wd, ln, 260, 1, 60 + i)
            gum = spl([(-200, 270), (-60, 216), (170, 250), (430, 200), (700, 252), (860, 236), (1030, 252), (1300, 196), (1590, 250), (1800, 220), (2010, 250), (2200, 230)], n=8); c.fill(ro(np.vstack([gum, [(2200, -400), (-200, -400)]])), ink if t == 0 else '#0a1a5a')
            splatter(c, 960, 560, 420, 70, 18, dk if t == 0 else '#cfe2ff', 5)
            im = c.layer()[..., :3]; dx, dy = shake(f, 24, 41); return comp.cam2d(im, W / 2 + dx, H / 2 + dy, 1.10 + .06 * t, .012 * (-1) ** t)
        im = np.zeros((H, W, 3), f32) + rgb('#030409'); k = math.exp(-(t - 2) / 2.2)
        # the eyes burn on for a moment, then nothing
        for (ex, ey, ang) in ((760, 300, .16), (1230, 330, -.10)):
            g, hh = grid(); u = (g - ex) * math.cos(ang) + (hh - ey) * math.sin(ang); v = -(g - ex) * math.sin(ang) + (hh - ey) * math.cos(ang); im = im + (np.exp(-(u / 85) ** 2 - (v / (9 + 2 * t)) ** 2))[..., None] * rgb('#4f9cff') * k * .9 + comp.radial(ex, ey, 260, 160, 2.2)[..., None] * rgb('#1a3fae') * k * .5
        return im
