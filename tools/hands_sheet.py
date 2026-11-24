"""python tools/hands_sheet.py  -> qa/hands/sheet.jpg : every pose of jj/hands.py at working size, each with its 120 px version"""
import sys, os; ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, cv2
from multiprocessing import Pool
from jj import hands as HD
from jj.ink import Cel
E = HD.euler; M, SL, HV, FM = HD.MALE, HD.SLIM, HD.HEAVY, HD.FEMALE
CELLS = [  # label, builder -> (hands, props), scale, dist, pal
 ('relaxed', lambda: ([HD.Hand('relaxed', 'R', E(20, 10, -6))], ()), .36, 8, 'day'), ('relaxed side', lambda: ([HD.Hand('relaxed', 'L', E(-100, 0, 0), bld=FM)], ()), .36, 8, 'day'),
 ('open', lambda: ([HD.Hand('open', 'R', E(168, -10, 6))], ()), .34, 8, 'day'), ('open back', lambda: ([HD.Hand('open', 'R', E(-22, 6, -6))], ()), .34, 8, 'day'),
 ('splay', lambda: ([HD.Hand('splay', 'R', E(162, 18, 0), bld=HV, nails='dark', bands=((.14, .23), (.34, .43)))], ()), .34, 5, 'day'), ('thrust at lens', lambda: ([HD.Hand('thrust', 'L', E(166, 64, 26), bld=HV)], ()), .34, 1.9, 'fire'),
 ('fist at lens', lambda: ([HD.Hand('fist', 'R', E(180, 78, 0), bld=HV)], ()), .52, 3.2, 'day'), ('fist side', lambda: ([HD.Hand('fist', 'R', E(90, 0, 80))], ()), .42, 8, 'day'), ('fist 3/4', lambda: ([HD.Hand('fist', 'R', E(148, 58, 0), bld=HV, tape=dict(fingers=HD.FING))], ()), .50, 4, 'day'),
 ('punch', lambda: ([HD.Hand('fist', 'L', E(196, 62, -70), bld=HV)], ()), .44, 2.6, 'night'), ('fist up', lambda: ([HD.Hand('fist', 'R', E(180, 0, 0), bld=SL)], ()), .42, 8, 'day'), ('fist back', lambda: ([HD.Hand('fist', 'R', E(-36, 28, 0))], ()), .42, 7, 'day'),
 ('point', lambda: ([HD.Hand('point', 'R', E(150, 18, 0), bld=SL)], ()), .34, 7, 'day'), ('point at lens', lambda: ([HD.Hand('point', 'R', E(150, 72, 10))], ()), .40, 2.6, 'day'),
 ('gojo cross', lambda: ([HD.Hand('cross', 'R', E(162, -8, -4), bld=SL, cuff=dict(kind='uniform', y=.16))], ()), .34, 8, 'day'), ('gojo two', lambda: ([HD.Hand('two', 'R', E(158, -8, 6), bld=SL, cuff=dict(kind='uniform', y=.2))], ()), .33, 8, 'pale'),
 ('pinch', lambda: ([HD.Hand('pinch', 'R', E(84, 0, 0), bld=FM)], ()), .36, 8, 'day'), ('flick', lambda: ([HD.Hand('flick', 'R', E(104, 0, 0), bld=SL)], ()), .36, 8, 'day'),
 ('pray', lambda: (HD.two('pray', R=E(-24, -14, -5)), ()), .27, 6, 'day'), ('sukuna sign', lambda: (HD.two('sukuna', R=E(-10, 8, 0), bld=HV, nails='claw', bands=((.16, .26), (.38, .48))), ()), .25, 7, 'ink'),
 ('megumi dog', lambda: (HD.two('dog', R=E(-25, 12, 0), bld=SL), ()), .28, 7, 'night'), ('clap open', lambda: (HD.two('clap', gap=.75, bld=HV), ()), .21, 7, 'day'), ('clap', lambda: (HD.two('clap', R=E(16, -8, 4), bld=HV), ()), .25, 7, 'day'),
 ('nails', lambda: (lambda h: ([h], HD.nail_props(h)))(HD.Hand('nails', 'R', E(178, 48, 0), bld=FM)), .34, 5, 'day'), ('hammer', lambda: (lambda h: ([h], HD.hammer_props(h)))(HD.Hand('hammer', 'R', E(112, 8, -30), bld=FM, pos=(0, -.2, 0))), .24, 8, 'day'),
 ('grip wrist', lambda: (lambda a: ([a, HD.grip_wrist(a, at=.35, turn=180)], ()))(HD.Hand('palm_up', 'R', E(170, -40, 30))), .25, 8, 'day'), ('rest', lambda: ([HD.Hand('rest', 'R', E(10, 42, 0))], ()), .36, 7, 'day'),
 ('fuga rear', lambda: ([HD.Hand('rear', 'R', E(118, 6, 20), bld=HV, nails='dark', bands=((.14, .23), (.34, .43)))], ()), .40, 7, 'fire'), ('palm up', lambda: ([HD.Hand('palm_up', 'R', E(174, -66, 14))], ()), .36, 4, 'night'),
 ('almost touching', lambda: (HD.two('reach', gap=.10), ()), .19, 8, 'day'), ('hook', lambda: ([HD.Hand('hook', 'R', E(100, 0, 0), bld=SL)], ()), .38, 8, 'day'), ('carry', lambda: ([HD.Hand('carry', 'R', E(150, 0, 176), bld=FM)], ()), .36, 8, 'day'),
 ('slap', lambda: ([HD.Hand('slap', 'R', E(176, 0, -12))], ()), .34, 8, 'day'), ('kimono', lambda: ([HD.Hand('relaxed', 'R', E(150, -10, 4), bld=HV, nails='dark', bands=((.1, .18), (.25, .33)), cuff=dict(kind='kimono', y=.42))], ()), .34, 8, 'day'),
 ('wraps', lambda: ([HD.Hand('open', 'R', E(165, 0, 0), tape=dict(fingers=HD.FING))], ()), .34, 8, 'day')]
BG = dict(day='#b9bfd0', night='#39405c', fire='#8f2f1e', pale='#6f86c8', ink='#6e1a20')


def cell(a):
    cv2.setNumThreads(1)
    lab, fn, sc, ds, pal = CELLS[a]; out = []
    for S in (460, 120):
        cel = Cel(S, S); cel.fill(np.array([(0, 0), (S, 0), (S, S), (0, S)]), BG[pal]); hs, pr = fn()
        g = HD.render(hs, HD.Cam(S / 2, S / 2, S * sc, ds), light=(-.5, .75, .6) if pal != 'fire' else (.5, -.5, .5), res=520, clip=(-8, -8, S + 8, S + 8), props=pr); HD.ink(cel, g, pal, 3.4 if S > 200 else 1.25)
        out.append(np.ascontiguousarray((np.clip(cel.rgba()[..., :3], 0, 1) * 255).astype(np.uint8)))
    im, sm = out; im[-120:, -120:] = sm; cv2.rectangle(im, (339, 339), (459, 459), (255, 255, 255), 1); cv2.putText(im, lab, (8, 20), 0, .55, (255, 255, 255) if pal != 'day' else (60, 0, 90), 1)
    return im


if __name__ == '__main__':
    sel = sys.argv[1:]; L = [i for i, c in enumerate(CELLS) if not sel or c[0] in sel]
    with Pool(4) as P: ims = P.map(cell, L)
    n = min(7, len(ims)); rows = [np.hstack((ims[i:i + n] + [np.full_like(ims[0], 255)] * n)[:n]) for i in range(0, len(ims), n)]
    os.makedirs('qa/hands', exist_ok=True); cv2.imwrite('qa/hands/sheet.jpg' if not sel else 'qa/hands/part.jpg', np.vstack(rows)[..., ::-1], [1, 88])
