"""fx orb: Gojo's Blue, Red, Hollow Purple, erasing trail, and the Unlimited Void kit."""
from .fx_core import *
from .fx_flash import kite, kite_star, bolt_line, bolt_poly, bolt_tree, ragged_ring, hairlines, distort_ring
from .comp import radial, blur


def _blob(x, y, r, seed, t=0., amp=.06, n=48, k=3.):
    a = np.linspace(0, 6.283, n, endpoint=False); w = 1 + amp * (noise(np.cos(a) * k * .5, np.sin(a) * k * .5, t * 2, seed) + .5 * noise(np.cos(a) * k, np.sin(a) * k, t * 3, seed + 1)); return np.c_[x + np.cos(a) * r * w, y + np.sin(a) * r * w]


def _crescent(x, y, r, a0, span, w, n=30, spiral=0., bulge=.5):
    """thick-thin arc shape, optionally spiralling inward"""
    v = np.linspace(0, 1, n); a = a0 + span * v; rr = r * (1 - spiral * v); hw = w * np.sin(np.pi * v ** (.5 + bulge)) ** .9
    return np.vstack([np.c_[x + np.cos(a) * (rr + hw), y + np.sin(a) * (rr + hw)], np.c_[x + np.cos(a) * (rr - hw), y + np.sin(a) * (rr - hw)][::-1]])


def _streaks(cel, x, y, r0, r1, n, f, seed, cols, inward=True, swirl=1.2, w=2.6, speed=1.6, asp=1., length=.35):
    t = f / 24.
    for i in range(n):
        ph = (t * speed * (.6 + .8 * h1(seed, i, 1)) + h1(seed, i, 2)) % 1.; cyc = int(t * speed * (.6 + .8 * h1(seed, i, 1)) + h1(seed, i, 2)); a = h1(seed, i, cyc, 3) * 6.283; u = ph if inward else 1 - ph; pos = r0 + (r1 - r0) * (1 - u ** 1.9) if inward else r0 + (r1 - r0) * (ph ** .55)
        ln = (r1 - r0) * length * (.3 + .9 * h1(seed, i, 4)) * (.35 + u if inward else 1.2 - ph); v = np.linspace(0, 1, 12); rr = pos + ln * v; aa = a + swirl * np.log(np.maximum(rr / r0, 1.)) * (1 if inward else -.4); P = np.c_[x + np.cos(aa) * rr, y + np.sin(aa) * rr * asp]
        ww = w * (.35 + 1.3 * bms(h1(seed, i, 5))) * min(1., ph * 6) * min(1., (1 - ph) * 5); p = Path(P, 4.); s = np.linspace(0, p.L, 14); hw = ww * np.sin(np.pi * (s / p.L) ** .45) ** .8
        if ww > .3: cel.fill(np.vstack([p.at(s, hw), p.at(s, -hw)[::-1]]), cols[i % len(cols)], .55 + .45 * h1(seed, i, 6))
    return cel


def orb_blue(x, y, r, f, power=1., seed=0, streaks=1., step=1, reach=6.):
    """Lapse: Blue.  implosion: light, air and shapes are dragged in along a spiral; dark pinched halo; navy ball, cyan shapes, white heart"""
    k = int(f) // step * step; t = k / 24.; cel = Cel(); p = power
    if streaks > 0: _streaks(cel, x, y, r * 1.08, r * reach, int(54 * streaks), k, seed, ('#eaffff', '#7fdcff', '#2f8cff', '#1a4dff'), True, 1.15, 1.2 + r * .022)
    for i in range(7):                                       # broad arms of drawn light
        a = i / 7 * 6.283 - t * 3.2 + .5 * h1(seed, i); cel.fill(_crescent(x, y, r * (1.9 + .9 * h1(seed, i, 2)), a, 1.5 + .7 * h1(seed, i, 3), r * .10 * (.5 + h1(seed, i, 4)), spiral=.42), ('#1d52ff', '#2f9bff', '#0f2bd0')[i % 3], .85)
    for i in range(5): cel.fill(_crescent(x, y, r * (1.22 + .10 * h1(seed, i, 7)), -t * 2.6 + i * 1.256, 1.0 + .4 * h1(seed, i, 8), r * .075, spiral=.16), '#020515', .8)
    cel.fill(_blob(x, y, r * 1.02, seed + 2, t, .05), '#0c22c8'); cel.clip(_blob(x, y, r * 1.02, seed + 2, t, .05))
    for i in range(6):
        a = -t * 5 + i * 1.05; cel.fill(_crescent(x, y, r * (.86 - .10 * (i % 3)), a, 1.4, r * .13, spiral=.5), ('#1c6bff', '#34b4ff', '#113df0')[i % 3])
    cel.fill(_blob(x - r * .04, y - r * .05, r * .56, seed + 3, t * 2, .16), '#39b8ff'); cel.fill(_blob(x - r * .06, y - r * .08, r * .40, seed + 4, t * 2, .18), '#a8f2ff'); cel.fill(_blob(x - r * .07, y - r * .09, r * .25, seed + 5, t * 3, .16), '#ffffff')
    cel.unclip()
    for i in range(4): cel.fill(_crescent(x, y, r * 1.01, -t * 4 + i * 1.7 + h1(seed, i, 9), .9 + .5 * h1(seed, i, 10), r * .022 + .8, spiral=.04), '#d8fbff')
    lay = cel.layer(); L = (radial(x, y, r * 7, power=3.2)[..., None] * C('#1b5cff') * .75 + radial(x, y, r * 2.6, power=2.)[..., None] * C('#3fa8ff') * .55) * p; L *= (1 - .7 * lay[..., 3:4]); L += hot(lay, .75, (3, 14, 40), (.35, .3, .2), '#bfeaff', .5 * p)
    return FX(lay, L)


def pinch(img, x, y, r, amt=.35, twist=.6):
    """space dragged into a point (negative amt = pushed out).  r = outer radius px"""
    h, w = img.shape[:2]; gx, gy = grid(h, w); dx, dy = gx - x, gy - y; d = np.sqrt(dx * dx + dy * dy) + 1e-3; k = np.clip(1 - d / r, 0, 1); k = k * k * (3 - 2 * k); s = 1 + amt * k; a = twist * k * k; c, q = np.cos(a), np.sin(a)
    return cv2.remap(img, (x + (dx * c - dy * q) * s).astype(f32), (y + (dx * q + dy * c) * s).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def orb_red(x, y, r, f, power=1., seed=0, streaks=1., step=1, reach=5.5):
    """Reversal: Red.  repulsion: shapes are thrown out in pulses; crimson ball, black-red edge, white-hot heart, spikes"""
    k = int(f) // step * step; t = k / 24.; cel = Cel()
    if streaks > 0: _streaks(cel, x, y, r * 1.1, r * reach, int(46 * streaks), k, seed, ('#fff1e0', '#ff7a4a', '#ff2a2a', '#c0041c'), False, .5, 1.2 + r * .024, 1.9, length=.28)
    for i in range(3):
        ph = (t * 1.5 + i / 3.) % 1.; ragged_ring(cel, x, y, r * (1.1 + 3.4 * ph ** .6), r * (1.1 + 3.4 * ph ** .6), r * .16 * (1 - ph) ** 1.2, seed + i * 5 + int(t * 1.5 + i / 3.), '#ff2a30' if i % 2 else '#8a0418', gaps=.25 + .4 * ph, n=8)
    for K, K2 in kite_star(x, y, r * 1.02, r * (2.1 + .4 * power), 24, seed + k // 2, None, .06, r * .06): cel.fill(K2, '#2a0006'); cel.fill(K, '#ff2634')
    cel.fill(_blob(x, y, r * 1.045, seed, t * 2, .06), '#2a0006'); cel.fill(_blob(x, y, r, seed + 2, t, .05), '#c00418'); cel.clip(_blob(x, y, r, seed + 2, t, .05))
    for i in range(6): a = t * 4 + i * 1.05; cel.fill(_crescent(x, y, r * (.84 - .1 * (i % 3)), a, 1.3, r * .14, spiral=-.15), ('#ff3a2a', '#ff6a3a', '#8e0012')[i % 3])
    cel.fill(_blob(x, y, r * .60, seed + 3, t * 2, .18), '#ff4a34'); cel.fill(_blob(x, y, r * .44, seed + 4, t * 2, .2), '#ffb08a'); cel.fill(star_core(x, y, r * .20, r * .50, 11, seed + k), '#fff6ea'); cel.unclip()
    for i in range(4): cel.fill(_crescent(x, y, r * 1.0, t * 5 + i * 1.6 + h1(seed, i, 9), .8 + .5 * h1(seed, i, 10), r * .022 + .8, spiral=-.04), '#ffd9c8')
    lay = cel.layer(); L = (radial(x, y, r * 7, power=3.2)[..., None] * C('#ff1a22') * .7 + radial(x, y, r * 2.6, power=2.)[..., None] * C('#ff4a30') * .5) * power; L *= (1 - .7 * lay[..., 3:4]); return FX(lay, L + hot(lay, .75, (3, 14, 40), (.35, .3, .2), '#ffd0c0', .5 * power))


def star_core(x, y, r0, r1, n, seed):
    r = R(seed, 4); P = []
    for i in range(n): a = (i + .4 * r.random()) / n * 6.283; l = r0 + (r1 - r0) * bms(r.random(), .3, .3); P += [(x + math.cos(a) * l, y + math.sin(a) * l)]; b = (i + .5 + .2 * r.random()) / n * 6.283; P += [(x + math.cos(b) * r0 * .7, y + math.sin(b) * r0 * .7)]
    return np.array(P)


def orb_purple(x, y, r, f, power=1., seed=0, bolts=1., step=1, squash=1., rot=0.):
    """Hollow Purple: dark violet sphere, off-centre black depth, white thick-thin rim, violet shapes turning inside, drawn lightning, blue and red fringes"""
    k = int(f) // step * step; t = k / 24.; cel = Cel(); r0 = R(seed, k // 2)
    for i in range(int(10 * bolts)):                         # lightning behind
        a = r0.uniform(0, 6.283); l = r * r0.uniform(1.25, 2.5); P0 = (x + math.cos(a) * r * .9, y + math.sin(a) * r * .9); b = a + r0.normal(0, .5); P1 = (x + math.cos(b) * l, y + math.sin(b) * l); ps, ln = bolt_tree(P0, P1, r * .035 + 1.2, seed + i * 7 + k // 2, .4, .5, 1, edge=r * .035)
        for K, K2 in ps: cel.fill(K2, '#6a1fe0')
        for K, K2 in ps: cel.fill(K, '#f4e6ff')
    ragged_ring(cel, x + r * .05, y, r * 1.10, r * 1.10, r * .07, seed + k, '#ff2a6a', gaps=.4, n=7, spurs=.2); ragged_ring(cel, x - r * .05, y, r * 1.10, r * 1.10, r * .07, seed + k + 9, '#2a5cff', gaps=.4, n=7, spurs=.2)
    body = _blob(x, y, r, seed + 2, t, .02); cel.fill(_blob(x, y, r * 1.075, seed + 1, t, .04), '#fdf6ff'); cel.fill(body, '#2a0a66'); cel.clip(body)
    for i in range(9):
        a = t * 2.6 * (1 if i % 2 else -.7) + i * .7; cel.fill(_crescent(x, y, r * (.93 - .085 * i * .9), a, 1.7 + .5 * h1(seed, i), r * .10, spiral=.25 * (1 if i % 2 else -1)), ('#4a14a8', '#7a2cf0', '#3a0f88', '#9a55ff')[i % 4])
    cel.fill(_blob(x + r * .08, y + r * .06, r * .56, seed + 3, t * 1.5, .10), '#170338'); cel.fill(_blob(x + r * .10, y + r * .08, r * .40, seed + 4, t * 2, .12), '#05010f')
    for i in range(5): a = -t * 5 + i * 1.26; cel.fill(_crescent(x + r * .10, y + r * .08, r * .44, a, .9, r * .020, spiral=.3), '#c79bff')
    cel.fill(_crescent(x, y, r * .84, 3.5, 1.5, r * .055), '#d9baff'); cel.fill(_crescent(x, y, r * .86, 3.75, .8, r * .03), '#ffffff')
    for i in range(int(4 * bolts)):                          # lightning over the surface
        a = r0.uniform(0, 6.283); b = a + r0.uniform(1.5, 3.2); P = bolt_line((x + math.cos(a) * r * .9, y + math.sin(a) * r * .9), (x + math.cos(b) * r * .85, y + math.sin(b) * r * .85), seed + i + k, .45, 6)
        for K, K2 in bolt_poly(P, r * .018 + 1, seed + i + k, edge=r * .02): cel.fill(K2, '#a45cff'); cel.fill(K, '#ffffff')
    cel.unclip(); ragged_ring(cel, x, y, r * 1.02, r * 1.02, r * .045 + 1, seed + k // 2, '#ffffff', gaps=.08, n=9, spurs=.25); lay = cel.layer(); L = (radial(x, y, r * 5, power=3.)[..., None] * C('#6a22ff') * .7 + radial(x, y, r * 2.2, power=2.)[..., None] * C('#b070ff') * .4) * power; L *= (1 - .8 * lay[..., 3:4])
    return FX(lay, L + hot(lay, .85, (3, 12, 34), (.3, .28, .2), '#e6d0ff', .5 * power))


def purple_merge(x, y, r, t, f, gap=420., seed=0, ang=0.):
    """t 0..1: Blue (one side) and Red (other) circle each other, stretch into tomoe commas, collapse (.72), flash, Purple is born (.8-1) -> FX"""
    if t >= .80: s = min(1., (t - .80) / .12); s = 1 + .22 * math.sin(s * math.pi) if s < 1 else 1.; return orb_purple(x, y, r * s, f, 1 + (1 - min(1, (t - .8) / .2)) * 1.2, seed, 1.6)
    e = t / .80; d = gap * (1 - e ** 1.7) * .5; a = ang + 3.6 * e * e * e * 1.6; c, s = math.cos(a), math.sin(a); rb = r * (.62 - .12 * e); B = orb_blue(x - c * d, y - s * d, rb, f, 1., seed, .7, reach=4.); Q = orb_red(x + c * d, y + s * d, rb, f, 1., seed + 3, .7, reach=4.); cel = Cel(); k = int(f)
    for sg, col, col2 in ((-1, '#1c5cff', '#8fe6ff'), (1, '#e0122a', '#ffb08a')):      # comma tails
        P = []; v = np.linspace(0, 1, 26); th = a + (math.pi if sg < 0 else 0) - v * (1.0 + 2.4 * e); rr = d + r * (.2 + .9 * v) * e; q = np.c_[x + np.cos(th) * rr, y + np.sin(th) * rr]; pth = Path(q, 4.); ss = np.linspace(0, pth.L, 24); hw = rb * .62 * (1 - ss / pth.L) ** .9 * e
        cel.fill(np.vstack([pth.at(ss, hw), pth.at(ss, -hw)[::-1]]), col); cel.fill(np.vstack([pth.at(ss, hw * .45), pth.at(ss, -hw * .25)[::-1]]), col2)
    r0 = R(seed, k)
    for i in range(int(2 + 7 * e)):
        P0 = np.array([x - c * d, y - s * d]) + r0.normal(0, rb * .5, 2); P1 = np.array([x + c * d, y + s * d]) + r0.normal(0, rb * .5, 2); ps, _ = bolt_tree(P0, P1, 2 + 5 * e, seed + i * 3 + k, .4, .4, 1, edge=5)
        for K, K2 in ps: cel.fill(K2, '#7a2cff')
        for K, K2 in ps: cel.fill(K, '#fbf0ff')
    m = merge(FX(cel.layer(), None), B, Q)
    if e > .86: u = (e - .86) / .14; m = FX(m.rgba, m.light + radial(x, y, 200 + 900 * u, power=2.2)[..., None] * C('#d9b8ff') * 1.6 * u)
    return m


def purple_trail(base, P, w, f, seed=0, under='#08030f', rim='#f6eaff', mid='#7a2cff', taper=.25, persp=None):
    """what Hollow Purple passed through is gone: a ragged erased tunnel along path P (start -> orb), white torn rim, violet inner shapes, sparks.  persp(v) = width scale along the path -> RGB"""
    k = int(f) // 2; p = Path(P, 6.); L = p.L; ws = persp or (lambda v: 1.); cel = Cel()
    def tun(kw, amp, sd):
        s = np.linspace(0, L, max(24, int(L / 9))); v = s / L; hw = np.array([w * kw * ws(q) for q in v]) * np.minimum(1., .15 + v / max(taper, 1e-3)); z = []
        for sg in (1, -1): n = hw * (1 + amp * noise(s / (w * .30) , sg * 5., k * .25, sd) + amp * .7 * np.maximum(0, noise(s / (w * .11), sg * 9., k * .5, sd)) ** 2); z.append(p.at(s, n * sg))
        return np.vstack([z[0], z[1][::-1]])
    cel.fill(tun(1.16, .16, seed), mid); cel.fill(tun(1.06, .14, seed + 1), rim); cel.fill(tun(.96, .10, seed + 2), '#2a0a66'); cel.fill(tun(.86, .10, seed + 3), under)
    for i in range(16):
        s0 = h1(seed, i, k) * L * .9; l = L * (.05 + .14 * h1(seed, i, 2)); o = (h1(seed, i, 3) * 2 - 1) * .78; v0 = s0 / L; q = p.sub(s0, min(L, s0 + l), lambda v: o * w * ws(v0) * (.9 + .1 * v), 12); thin(cel, q, 1.2 + 2.2 * h1(seed, i, 4), mid if i % 3 else rim, 1., .4, .5, i)
    for i in range(34):                                      # torn crumbs at the edge
        s0 = h1(seed, i, 11) * L; sg = 1 if i % 2 else -1; v0 = s0 / L; n0 = w * ws(v0) * (1.10 + .35 * ((h1(seed, i, 12) + f / 40.) % 1.)); c = p.at(s0, sg * n0); s = w * .05 * (.3 + h1(seed, i, 13)); th = np.sort(R(seed, i).uniform(0, 6.283, 4)) + f * .2; cel.fill(np.c_[c[0] + np.cos(th) * s * 1.6, c[1] + np.sin(th) * s], '#12061f' if i % 3 else mid)
    lay = cel.layer(); out = base * (1 - lay[..., 3:4]) + lay[..., :3]; return out + hot(lay, .8, (4, 16, 50), (.3, .25, .2), '#c9a0ff', .5)


# ------------------------------------------------------------------ Unlimited Void
def glyph(cel, x, y, s, rot, seed, col='#ffffff', alpha=1.0, w=None, flip=1.):
    """unreadable ideograph-like mark: 4-8 brush strokes in a square (no real character)"""
    r = R(seed, 6); w = w or s * .11; c, q = math.cos(rot), math.sin(rot)
    def T(P): P = np.asarray(P, float) * [s * flip, s]; return np.c_[P[:, 0] * c - P[:, 1] * q + x, P[:, 0] * q + P[:, 1] * c + y]
    for i in range(int(r.integers(4, 8))):
        u = r.random(); a, b = r.uniform(-.45, .45, 2)
        if u < .36: P = [[-.48 * r.uniform(.4, 1), a], [0, a - .03], [.48 * r.uniform(.4, 1), a - .05]]
        elif u < .62: P = [[b, -.5 * r.uniform(.5, 1)], [b + .01, 0], [b, .5 * r.uniform(.5, 1)]]
        elif u < .80: P = [[b, -.3], [b - .10, .1], [b - .34, .46]]
        elif u < .92: P = [[b, -.1], [b + .14, .2], [b + .36, .44]]
        else: P = [[-.4, a], [.34, a - .02], [.36, a + .06], [.30, a + .44], [.20, a + .38]]
        thin(cel, T(spl(P, n=4)), w * r.uniform(.7, 1.3), col, alpha, .06, .35, seed + i, .25, 12)
    return cel


_vbg = {}


def void_bg(seed=0, cx=1180., cy=430.):
    """painted nothing: dark gouache strokes turning round the sun, indigo / violet / teal, almost black (cached)"""
    key = (seed, int(cx), int(cy))
    if key not in _vbg:
        from .paint import Canvas, arcs
        r = np.random.default_rng(seed + 5); cv = Canvas(H, W, color='#03040d'); n1 = fbm(H, W, (520, 240, 90), None, seed + 11); n2 = fbm(H, W, (420, 180), None, seed + 12)
        for hw, n, al in ((42, 900, .75), (20, 2200, .7), (9, 4200, .65), (3.5, 5200, .6)):
            d = 140 + 1650 * r.random(n) ** .75; a = r.uniform(0, 6.283, n); x, y = cx + np.cos(a) * d, cy + np.sin(a) * d * .92; ok = (x > -60) & (x < W + 60) & (y > -60) & (y < H + 60); x, y, a, d = x[ok], y[ok], a[ok], d[ok]
            u = n1[np.clip(y, 0, H - 1).astype(int), np.clip(x, 0, W - 1).astype(int)]; v = n2[np.clip(y, 0, H - 1).astype(int), np.clip(x, 0, W - 1).astype(int)]; k = np.clip(.42 + .32 * u, 0, 1)[:, None]; q = np.clip(.5 + .4 * v, 0, 1)[:, None]
            col = C('#04050f') * (1 - k) + (C('#14226e') * (1 - q) + C('#3a1a78') * q) * k; col = col + np.clip(u - 1.25, 0, 1)[:, None] * C('#2f74b0') * .5; col = col * (.55 + .6 * np.exp(-d / 620.))[:, None] * r.uniform(.6, 1.15, (len(x), 1))
            cv.strokes(arcs(x, y, a + np.pi / 2 + .35 + r.normal(0, .10, len(x)), hw * r.uniform(5, 11, len(x)), hw * 7 / np.maximum(d, 80)), hw * r.uniform(.5, 1.1, len(x)), np.clip(col, 0, 1).astype(f32), op=al, dry=.35 * r.random(len(x)), seed=seed + int(hw), shape='round', bristle=.12)
        _vbg[key] = cv.img
    return _vbg[key]


def void(f, cx=1180., cy=430., rad=210., seed=0, fall=1., zoom=1., lines=300, glyphs=64, tilt=.22, rot=-.35, drift=(0., 0.), freeze=False, sun=1., vp=None, bg=None, streams=1.):
    """Unlimited Void picture: deep indigo nothing, black sun with a drawn white accretion ring (far arc, near band, lensed halo), rivers of thin light running from the
    vanishing point past the camera, glyph-like debris with depth.  fall = speed of the rivers, freeze = hold time.  -> RGB frame"""
    t = (0. if freeze else f / 24.) * fall; img = (void_bg(seed, cx, cy) if bg is None else bg).copy(); vx, vy = vp or (cx, cy); cel = Cel(); far = Cel(ss=1); r = R(seed, 2)
    for i in range(420):                                     # stars, three depths
        z = h1(seed, i, 1); a = h1(seed, i, 2) * 6.283; d = (70 + 1500 * h1(seed, i, 3)) * (1 + t * .03 * z); x, y = vx + math.cos(a) * d + drift[0] * z, vy + math.sin(a) * d * .8 + drift[1] * z
        if 0 < x < W and 0 < y < H: s = .5 + 1.2 * z ** 4; far.fill(np.array([[x - s, y], [x, y - s], [x + s, y], [x, y + s]]), '#cfe0ff', .25 + .7 * z ** 2)
    n = int(lines * streams)                                 # rivers of light
    for i in range(n):
        a = (i + h1(seed, i, 11)) / n * 6.283 * 1.0; bundle = int(h1(seed, i, 12) * 11); ab = bundle / 11. * 6.283 + .5 * h1(seed, bundle, 40); a = ab + (h1(seed, i, 13) - .5) * (.12 + 1.0 * h1(seed, i, 14) ** 3); rr = 1.1 + 2.6 * h1(seed, i, 15); tw = .55 + .5 * h1(seed, bundle, 16); ph = (t * (.16 + .20 * h1(seed, i, 17)) + h1(seed, i, 18)) % 1.
        z0 = 1 / (.012 + .55 * ph ** 2.2); z1 = z0 * (1.7 + 1.6 * h1(seed, i, 19)); zz = np.exp(np.linspace(math.log(z0), math.log(z1), 26)); aa = a + tw * .55 * np.log(zz + 1) + .06 * np.sin(zz * .4 + i); X = vx + np.cos(aa) * rr * rad * 4.2 / zz; Y = vy + np.sin(aa) * rr * rad * 3.4 / zz; dd = np.hypot(X - cx, Y - cy)
        if (dd < rad * .9).all(): continue
        fade = min(1., ph * 5) * min(1., (1 - ph) * 6); w = (.5 + 2.6 * ph ** 2 * bms(h1(seed, i, 20))) * 1.0; col = ('#ffffff', '#bfe4ff', '#8fb4ff', '#c9a8ff', '#e8f6ff')[i % 5]; thin(cel if w > 1.0 else far, np.c_[X, Y], w * 1.7, col, .22 + .7 * fade * h1(seed, i, 21), .35, .6, i, .25)
    img = img + far.layer()[..., :3] * .9
    if glyphs:
        G = []
        for i in range(glyphs):
            ph = (t * (.10 + .10 * h1(seed, i, 31)) + h1(seed, i, 32)) % 1.; z = 1 / (.05 + .95 * ph ** 1.8); a = h1(seed, i, 33) * 6.283; rr = 1.0 + 2.2 * h1(seed, i, 34); x = vx + math.cos(a) * rr * rad * 3.1 / z + drift[0] * ph; y = vy + math.sin(a) * rr * rad * 2.5 / z + drift[1] * ph; s = 46 / z * (.6 + .8 * h1(seed, i, 35)) + 3; G.append((z, x, y, s, i, ph))
        G.sort(key=lambda q: -q[0]); near = Cel(W // 2, H // 2)
        for z, x, y, s, i, ph in G:
            if -80 < x < W + 80 and -80 < y < H + 80 and math.hypot(x - cx, y - cy) > rad * 1.05: glyph(near, x / 2, y / 2, s / 2, t * (h1(seed, i, 36) - .5) * 2 + i, seed + i, '#dfeaff' if i % 4 else '#b9a0ff', .25 + .65 * min(1., ph * 3) * min(1., (1 - ph) * 6), flip=.25 + .75 * abs(math.cos(t * .9 * (h1(seed, i, 37) + .3) + i))) if ph > .75 else glyph(cel, x, y, s, t * (h1(seed, i, 36) - .5) * 2 + i, seed + i, '#dfeaff' if i % 4 else '#b9a0ff', .25 + .65 * min(1., ph * 3) * min(1., (1 - ph) * 6), flip=.25 + .75 * abs(math.cos(t * .9 * (h1(seed, i, 37) + .3) + i)))
        nl = cv2.resize(cv2.GaussianBlur(near.layer(H // 2, W // 2), (0, 0), 1.6), (W, H), interpolation=cv2.INTER_LINEAR); img = img * (1 - nl[..., 3:4] * .8) + nl[..., :3] * .8
    lay = cel.layer(); img = img * (1 - lay[..., 3:4]) + lay[..., :3] + hot(lay, .5, (3, 12, 40), (.25, .22, .18), '#9fc4ff', .7)
    if sun > 0: s = black_sun(cx, cy, rad * sun * zoom, f if not freeze else 0, seed, tilt, rot); img = put(img, s)
    return img


def black_sun(cx, cy, r, f, seed=0, tilt=.22, rot=-.35, step=2):
    """black disc, hand-drawn white photon ring, lensed halo and a tilted accretion band crossing in front -> FX"""
    k = int(f) // step; t = k * step / 24.; cel = Cel(); cr, sr = math.cos(rot), math.sin(rot)
    def ell(R0, a, asp, h=0.): X, Y = np.cos(a) * R0, np.sin(a) * R0 * asp + h; return np.c_[cx + X * cr - Y * sr, cy + X * sr + Y * cr]
    def arc(R0, a0, sp, asp, w, col, h=0., al=1.): a = np.linspace(a0, a0 + sp, max(12, int(sp * 22))); v = np.linspace(0, 1, len(a)); hw = w * np.sin(np.pi * v) ** .6; O = ell(1., a, asp, 0) - [cx, cy]; A = ell(R0, a, asp, h); u = O / (np.hypot(*O.T)[:, None] + 1e-6); cel.fill(np.vstack([A + u * hw[:, None], (A - u * hw[:, None])[::-1]]), col, al)
    for i in range(34):                                      # accretion, far half (behind)
        R0 = r * (1.35 + 1.35 * h1(seed, i, 1) ** 1.4); a0 = math.pi + h1(seed, i, 2) * 2.3 + .3 * math.sin(t + i); arc(R0, a0, .5 + 1.2 * h1(seed, i, 3), tilt, r * .012 * (.4 + 2.2 * h1(seed, i, 4) ** 2), ('#ffffff', '#cfe6ff', '#9fb8ff')[i % 3], 0., .55 + .45 * h1(seed, i, 5))
    for i in range(22):                                      # lensed halo: light bent over the pole
        R0 = r * (1.10 + .42 * h1(seed, i, 11) ** 1.6); a0 = h1(seed, i, 12) * 6.283 + t * (.5 + .4 * h1(seed, i, 13)); arc(R0, a0, .5 + 1.5 * h1(seed, i, 14), 1., r * .010 * (.3 + 1.8 * h1(seed, i, 15) ** 2), ('#e9f3ff', '#b4ccff', '#ffffff')[i % 3], 0., .5 + .5 * h1(seed, i, 16))
    cel.fill(_blob(cx, cy, r * 1.045, seed, t * .4, .012, 90), '#ffffff'); cel.fill(_blob(cx, cy, r * 1.010, seed + 1, t * .4, .016, 90), '#05060c'); cel.fill(_blob(cx, cy, r * .99, seed + 1, t * .3, .01, 90), '#000000')
    ragged_ring(cel, cx, cy, r * 1.035, r * 1.035, r * .028, seed + k, '#ffffff', gaps=.12, n=10, rag=.012, spurs=.3)
    for i in range(44):                                      # near half of the band crosses the disc
        R0 = r * (1.22 + 1.6 * h1(seed, i, 21) ** 1.5); a0 = -.15 + h1(seed, i, 22) * 2.6; sp = .4 + 1.3 * h1(seed, i, 23); a0 = (a0 + t * 1.1 * (1.6 / (R0 / r))) % 3.3 - .1; arc(R0, a0, min(sp, 3.25 - a0), tilt, r * .012 * (.4 + 2.6 * h1(seed, i, 24) ** 2), ('#ffffff', '#ffffff', '#d4e6ff', '#a9bcff')[i % 4], 0., .6 + .4 * h1(seed, i, 25))
    lay = cel.layer(); g = radial(cx, cy, r * 4.2, power=3.)[..., None] * C('#5a86ff') * .42; d = np.sqrt((grid()[0] - cx) ** 2 + (grid()[1] - cy) ** 2); g = g * smooth(r * .9, r * 1.15, d)[..., None]
    return FX(lay, (g + hot(lay, .6, (4, 16, 54), (.35, .3, .26), '#cfe0ff', .75)) * (1 - .25 * lay[..., 3:4]))
