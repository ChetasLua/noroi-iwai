"""python tools/fig_demo.py [sheet] : QA sheet of the figure kit -> qa/figure/sheet.jpg"""
import sys, os; ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, cv2, math, time
cv2.setNumThreads(4)
from jj.cam import Camera3D
from jj.figure import *
from jj.fig_smear import limb_smear
from jj.figure import POSES
from jj import comp
os.makedirs('qa/figure', exist_ok=True)
BG = np.array([.80, .79, .77], np.float32)


def tile(w, h): return np.full((h, w, 3), BG, np.float32)


def ground(im, cam, w=3.):
    p = cam.project(np.array([[-w, 0, 0], [w, 0, 0]]))[0]; cv2.line(im, tuple(int(v) for v in p[0]), tuple(int(v) for v in p[1]), (.62, .60, .58), 1, cv2.LINE_AA); return im


def lab(im, s):
    q = (np.clip(im, 0, 1) * 255).astype(np.uint8).copy(); cv2.putText(q, s, (10, im.shape[0] - 10), 0, .5, (60, 58, 56), 1, cv2.LINE_AA); return q.astype(np.float32) / 255


rows = []
t0 = time.time()
# 1: everybody
W1 = 1920; im = tile(W1, 620); cam = Camera3D((0, -10, 1.15), (0, 0, 1.08), focal=2120, w=W1, h=620); ground(im, cam, 5)
for i, (c, p, y) in enumerate((('nobara', 'contra', 2.8), ('megumi', 'pockets', 2.8), ('yuji', 'contra', 2.75), ('gojo', 'cast', 2.85), ('nanami', 'contra', 2.9), ('todo', 'stand', 2.85), ('sukuna', 'contra', 2.9), ('crowd', 'walk_contact', 2.6), ('crowdf', 'contra', 2.8), ('curse', 'hulk', 2.75))):
    F = Fig(c).pose(p).place((i - 4.62) * .84 + (.26 if i > 8 else 0.), 0, 0, y); im = comp.over(im, F.draw(cam, 0))
rows.append(lab(im, 'cast'))
# 2: action poses
cells = []
for c, p, y, e in (('yuji', 'punch_hit', 2.2, 'shout'), ('yuji', 'dive_punch', 1.9, 'rage'), ('sukuna', 'kick', 2.2, 'grin'), ('sukuna', 'palm', 2.2, 'grin'), ('megumi', 'leap', 2.4, 'firm'), ('todo', 'punch_wind', 2.3, 'grin'), ('nobara', 'hammer_up', 2.4, 'firm'), ('gojo', 'float', 2.8, None), ('nanami', 'stagger', 2.5, None), ('yuji', 'slide', 2.2, 'rage'), ('yuji', 'land', 2.4, 'rage'), ('yuji', 'knees', 2.3, 'sad')):
    cam = Camera3D((0, -6.4, .75), (0, 0, .95), focal=1320, w=480, h=480); F = Fig(c).pose(p).place(0, 0, -.12 if p in ('leap', 'dive_punch') else 0, y); im = ground(tile(480, 480), cam)
    im = comp.over(im, F.draw(cam, 0, expr=dict(mood=e) if e else None, rim=dict(dir=(.9, .2, -.45), col='#ffd9a8', thr=.55, amt=.75))); cells.append(lab(im, c + '  ' + p))
for k in range(0, 12, 4): rows.append(np.hstack(cells[k:k + 4]))
# 3: sprint cycle, 12 drawings on twos (one stride)
cells = []
for i in range(12):
    cam = Camera3D((0, -7, 1.0), (0, 0, .88), focal=1060, w=320, h=300); F = Fig('yuji').anim(lambda f: dict(pose=sprint(f / 24.), heading=math.pi / 2 + .2, x=0, y=0, z=0)); F.at(i * 2); cells.append(lab(comp.over(ground(tile(320, 300), cam), F.draw(cam, i * 2)), str(i * 2)))
rows.append(np.hstack(cells[:6])); rows.append(np.hstack(cells[6:]))
# 4: punch with smear, consecutive frames
cells = []; M = mirror; KEYS = [(0, M(POSES['guard']), 'in'), (4, M(POSES['punch_wind']), 'out'), (8, M(add_pose(POSES['punch_wind'], root=(0, -.015, -.01))), 'snap'), (10, M(POSES['punch_hit']), 'hold'), (12, M(POSES['punch_hit']), 'in'), (15, M(POSES['punch_follow']), 'out'), (22, M(POSES['guard']), 'io')]
fn = lambda f: dict(pose=seq(f, KEYS), x=-.45 + .7 * float(np.interp(f, [8, 10, 13], [0, .75, 1])), y=0, z=0, heading=-math.pi / 2 - .15)
for f in (0, 3, 5, 8, 9, 10, 12, 15):
    cam = Camera3D((0, -6.6, .95), (0, 0, .95), focal=1180, w=480, h=360); F = Fig('yuji').anim(fn); F.at(f); im = ground(tile(480, 360), cam); L = F.draw(cam, f, expr=dict(mood='rage'))
    if f in (9, 10): im = comp.over(im, F.smear(cam, f - 1, f, 'armL', seed=f))
    cells.append(lab(comp.over(im, L), 'f%d' % f))
rows.append(np.hstack(cells[:4])); rows.append(np.hstack(cells[4:]))
comp.save('qa/figure/sheet.jpg', np.vstack([r[:, :1920] for r in rows]), 91); print('%.1fs' % (time.time() - t0))
