"""QA: model sheet of Nanami and Todo -> qa/nanami/sheet.jpg"""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import cv2, numpy as np; cv2.setNumThreads(4)
from chars.ntkit import *
from chars.nanami import Nanami
from chars.todo import Todo
from chars.nt_props import goggles3d, watch, sandwich
from chars.nanami import PAL as NP
W2, H2 = 2800, 3640
base = np.ones((H2, W2, 3), np.float32) * rgb('#d6d8df')
def band(y0, y1, c): base[y0:y1] = rgb(c)
band(0, 900, '#d6d8df'); band(900, 1500, '#c3c8d6')
c = Cel(W2, 1500)
heads = [(0, 0, 0, 'deadpan', 1), (.42, 0, 0, 'deadpan', 1), (.75, 0, 0, 'calm', 0), (1.5, 0, 0, 'far', 0), (-.45, 0, 0, 'corner', 0), (-.8, 0, 0, 'gentle', 0), (-1.5, 0, 0, 'closed', 0),
         (0, .32, 0, 'closed', 0), (.3, -.35, .05, 'look_down', 0), (.5, .05, -.05, 'gentle', 0), (-.35, 0, .04, 'speak_a', 0), (.35, .05, 0, 'speak_o', 0), (0, 0, 0, 'grit', 0), (-1.5, -.1, 0, 'deadpan', 0)]
for i, (yw, pt, rl, ex, g) in enumerate(heads):
    Nanami.head(c, Nanami.pose(210 + 398 * (i % 7), 250 + 450 * (i // 7), 150, yw, pt, rl), ex, 'day', goggles=bool(g), lit=-1 if yw >= 0 else 1)
th = [(0, 0, 0, 'focus'), (.5, 0, 0, 'smug'), (-.5, 0, 0, 'grin'), (1.5, 0, 0, 'focus'), (0, .32, .06, 'weep'), (.4, .25, 0, 'joy'), (-.4, -.25, 0, 'warcry')]
for i, (yw, pt, rl, ex) in enumerate(th): Todo.head(c, Todo.pose(210 + 398 * i, 1190, 150, yw, pt, rl), ex, 'day', lit=-1 if yw >= 0 else 1)
l = c.layer(1500, W2); base[:1500] = comp.over(base[:1500], l)
y = 1500
for i, (lt, bgc, of, kw) in enumerate([('window', '#2d3a63', 'suit', dict(yaw=.5, expr='corner', goggles=False, lit=1, loosen=1.)), ('ember', '#1c0c0c', 'suit', dict(yaw=-.45, expr='grit', goggles=True, lit=-1)), ('beach', '#bfe2f2', 'shirt', dict(yaw=-.4, expr='gentle', goggles=False, lit=-1, loose=1., wind=.05)), ('night', '#1b1f3a', 'suit', dict(yaw=.1, expr='deadpan', goggles=True, lit=-1))]):
    L_ = Nanami.bust(350, 260, 128, light=lt, outfit=of, w=700, h=760, pitch=0., byaw=kw['yaw'] * .7, **kw)
    if lt == 'blue': L_ = rim(L_, -5, 2, '#ffd08a', .95, .6)
    if lt == 'ember': L_ = rim(L_, 5, -2, '#ff9a42', .95, .6)
    if lt == 'night': L_ = rim(L_, 4, 2, '#8fd0ff', .8, .6)
    b_ = np.ones((760, 700, 3), np.float32) * rgb(bgc); base[y:y + 760, 700 * i:700 * i + 700] = comp.over(b_, L_)
y = 2260
for i, (po, ex, lt, bgc, kw) in enumerate([('windup', 'focus', 'day', '#d6d8df', dict(bpitch=.05)), ('clap', 'warcry', 'night', '#1b1f3a', dict(bpitch=.05)), ('spread', 'weep', 'glory', '#e48a6a', dict(pitch=.2, bpitch=.2, persp=.11)), ('crossed', 'smug', 'day', '#c3c8d6', dict(yaw=.4, byaw=.5))]):
    L_ = Todo.bust(350, 200, 66, pose=po, expr=ex, light=lt, w=700, h=760, **kw)
    if lt == 'night': L_ = rim(L_, 4, 2, '#8fd0ff', .8, .6)
    b_ = np.ones((760, 700, 3), np.float32) * rgb(bgc); base[y:y + 760, 700 * i:700 * i + 700] = comp.over(b_, L_)
y = 3020; base[y:] = rgb('#59637f'); c = Cel(W2, H2 - y)
cw = Cel(1300, H2 - y); watch(cw, 520, 300, 200, lw=4.5); base[y:, 900:2200] = comp.over(base[y:, 900:2200], cw.layer(H2 - y, 1300))
goggles3d(c, 230, 160, 330, roll=.05, lw=4.5); goggles3d(c, 640, 170, 330, yaw=.7, pitch=.3, roll=-.5, lw=4.5); goggles3d(c, 260, 430, 330, yaw=-1., pitch=-.6, roll=.9, lw=4.5); goggles3d(c, 640, 440, 330, yaw=2.6, pitch=.2, roll=.4, lw=4.5)
sandwich(c, 2480, 330, 110, lw=3.6)
base[y:] = comp.over(base[y:], c.layer(H2 - y, W2))
os.makedirs('qa/nanami', exist_ok=True); comp.save('qa/nanami/sheet.jpg', base, 90)
