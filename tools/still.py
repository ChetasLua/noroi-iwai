"""python tools/still.py d4 40 [80 ...] [--crop x,y[,w,h]] [--sheet] [--strip a:b:step] [--raw] [--scale .5] [--out DIR]
Finished stills of a shot -> qa/<shot>/fNNNN.jpg.  --sheet = one contact sheet (3 per row, 640 wide).
--strip 20:44:2 = consecutive-frame sheet for judging motion.  Crops are 1:1 (600x450 default)."""
import sys, os, importlib, argparse, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, cv2
cv2.setNumThreads(4)
from jj import comp
ap = argparse.ArgumentParser(); ap.add_argument('shot'); ap.add_argument('frames', nargs='*', type=int)
ap.add_argument('--crop', action='append', default=[]); ap.add_argument('--sheet', action='store_true'); ap.add_argument('--strip'); ap.add_argument('--raw', action='store_true')
ap.add_argument('--scale', type=float, default=1.0); ap.add_argument('--out', default=None); ap.add_argument('--cols', type=int, default=3)
a = ap.parse_args()
fr = list(a.frames)
if a.strip: fr += list(range(*[int(v) for v in a.strip.split(':')])); a.sheet = True
m = importlib.import_module('shots.' + a.shot); S = m.Shot(); t = time.time(); S.setup(); print('setup %.1fs' % (time.time() - t))
d = a.out or f'qa/{a.shot}'; os.makedirs(d, exist_ok=True); ims = []
for f in fr:
    t = time.time(); im = S.frame(f); t1 = time.time() - t
    assert im.shape == (1080, 1920, 3), im.shape
    if not a.raw: im = comp.finish(im, f, **getattr(S, 'finish', {}))
    im8 = comp.to8(np.clip(im, 0, 1), f); print(f, '%.2fs' % t1)
    if a.sheet:
        s = cv2.resize(im8, (640, 360), interpolation=cv2.INTER_AREA); cv2.putText(s, str(f), (8, 350), 0, .6, (255, 255, 0), 2); ims.append(s)
    else:
        o = im8 if a.scale == 1 else cv2.resize(im8, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
        comp.save(f'{d}/f{f:04d}.jpg', o, 90)
    for k, c in enumerate(a.crop):
        v = [int(q) for q in c.split(',')]; w, h = (v[2], v[3]) if len(v) > 2 else (600, 450)
        comp.save(f'{d}/f{f:04d}_c{k}.jpg', im8[v[1]:v[1] + h, v[0]:v[0] + w], 92)
if a.sheet:
    n = a.cols; rows = [np.hstack((ims[i:i + n] + [np.zeros_like(ims[0])] * n)[:n]) for i in range(0, len(ims), n)]
    comp.save(f'{d}/sheet.jpg', np.vstack(rows), 88); print(f'{d}/sheet.jpg')
