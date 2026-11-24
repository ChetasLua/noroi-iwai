"""qa/ink/sheet.jpg + qa/ink/panels.jpg: every letter, how it is written, and the manga page toolkit."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2, math
cv2.setNumThreads(4)
from jj.sumi import *; from jj.letters import *; from jj.ink import Cel
from jj.comp import to8, over
import jj.manga as M
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'qa', 'ink'); os.makedirs(OUT, exist_ok=True)
wr = lambda n, im: cv2.imwrite(os.path.join(OUT, n), to8(np.clip(im, 0, 1))[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])


def strip(ch, sz=300, part=.55):
    n = len(KANJI[ch]); pg = Page(sz, sz, seed=3); write(pg, ch, (sz * .05, sz * .10, sz * (.80 if ch != '生' else .86))); P = paper(sz, sz); out = []
    for i in range(n):
        d, w_, p = pg.ink(i + part + .45 * (i == n - 1), full=True); out.append(lay(P, d, w_, p))
    return np.hstack(out)


def grid_row(imgs, W_): r = np.hstack(imgs); return np.pad(r, ((0, 0), (0, max(0, W_ - r.shape[1])), (0, 0)), constant_values=.85)[:, :W_]


if __name__ == '__main__':
    what = sys.argv[1:] or ['sheet', 'panels']
    if 'sheet' in what:
        W_ = 2700; rows = [grid_row([strip('呪', 300)], W_), grid_row([strip('祝', 300)], W_), grid_row([strip('生', 300)], W_)]
        rows[0][:, 2400:] = paper(300, 300); rows[2][:, 1500:] = paper(300, 1200)
        a = rows[2][:, 1500:]; L = [latin('a fan film', 600, 120, 70, seed=3, align='center'), latin('characters by Gege Akutami', 600, 236, 70, seed=4, align='center')]; rows[2][:, 1500:] = lay(a, latin_ink(L, 1., 300, 1200))
        rows[0][:, 2400:] = stamp(rows[0][:, 2400:], 150, 150, 190, seed=2)
        pg = Page(640, W_, seed=8)
        for i, c in enumerate('領域展開'): write(pg, c, (20 + i * 350, 20, 330), wt=1.2, seed=i)
        for i, c in enumerate('五条悟'): write(pg, c, (1460 + i * 410, 10, 390), wt=1.35, seed=i)
        d, w_, p = pg.ink(full=True); r = lay(paper(640, W_), d, w_, p)
        c = Cel(W_, 640)
        sfx(c, 'ガコン', [(40, 400), (760, 380), (740, 600), (20, 620)], seed=3, hatch=4); sfx(c, 'ドドド', [(800, 450), (1400, 390), (1420, 640), (780, 560)], seed=5, grow=.16); sfx(c, 'ドン', [(1480, 430), (1960, 400), (1980, 620), (1440, 640)], seed=1, hatch=4); sfx(c, 'バッ', [(2020, 420), (2300, 400), (2280, 630), (2040, 640)], seed=4, hatch=3)
        sfx(c, 'ゴゴゴ', [(2400, 420), (2680, 400), (2680, 620), (2380, 560)], seed=2, grow=.2)
        rows.append(over(r, c.layer(640, W_)))
        wr('sheet.jpg', np.vstack(rows))
    if 'panels' in what:
        H_ = 620; A = [M.art(n) for n in ('explosion', 'runner', 'fist', 'street', 'eye', 'fingers')]
        R = lambda g: cv2.resize(g, (int(g.shape[1] * H_ / g.shape[0]), H_), interpolation=cv2.INTER_AREA)
        t = np.hstack([np.pad(R(g), ((6, 6), (6, 6)), constant_values=.5) for g in A[:3]]); b = np.hstack([np.pad(R(g), ((6, 6), (6, 6)), constant_values=.5) for g in A[3:]]); w_ = max(t.shape[1], b.shape[1]); f = lambda g: np.pad(g, ((0, 0), (0, w_ - g.shape[1])), constant_values=.5)
        wr('panels.jpg', np.repeat(np.vstack([f(t), f(b)])[..., None], 3, -1))
        S = M.page(); c = Cel()
        sfx(c, 'ドン', [(1440, 30), (1910, 0), (1910, 320), (1490, 350)], seed=1, grow=.2, hatch=3, hatch_dir=(.6, 1)); sfx(c, 'ゴゴゴ', [(40, 36), (170, 26), (196, 420), (30, 432)], seed=2, vertical=True, grow=.16); sfx(c, 'バッ', [(300, 840), (440, 830), (430, 1060), (270, 1060)], seed=4, vertical=True, hatch=2)
        S.ink(c); wr('page.jpg', S.rgb())
