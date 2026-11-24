import sys; import os; sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
import numpy as np, math, cv2
from jj.head3 import Head
from jj.ink import Cel, spl
from jj.kit import outline, shape
from jj import comp
W2,H2=1920,1080
c=Cel(W2,H2); c.fill(np.array([(0,0),(W2,0),(W2,H2),(0,H2)]),'#c9ccd6')
poses=[(0,0,0),(.35,0,0),(.7,0,0),(1.1,0,0),(1.5,0,0),(0,.35,0),(0,-.35,0),(.6,.3,.1),(-.5,-.3,-.1),(-.9,0,0)]
for i,(yw,pt,rl) in enumerate(poses):
    cx=200+380*(i%5); cy=270+520*(i//5)
    H=Head().pose(cx,cy,190,yw,pt,rl)
    sk=H.skull(); c.fill(sk,'#8a8fa0')
    for s in (-1,1):
        e=H.ear(s)
        if e['vis']>0: shape(c,e['outer'],'#e8bfa4',2.4); c.line(e['inner'],1.5)
    if pt>0: c.fill(H.under(),'#b98a78')
    f=H.face(); c.fill(f,'#f0cdb4'); outline(c,f,3.2)
    for s in (-1,1):
        if s*yw<=0.2: pass
        c.line(H.jaw(s),2.0,alpha=.4,col='#a03030')
    for s in (-1,1):
        ex=.33*s
        if H.facing(ex,0)>.12:
            c.line(H.P([(ex-.17,-.02),(ex-.10,.055),(ex,.075),(ex+.10,.06),(ex+.17,.0)]),5)
            c.line(H.P([(ex-.11,-.085),(ex,-.10),(ex+.10,-.085)]),2)
            c.fill(H.P(spl([(ex+.075*math.cos(a),-.01+.085*math.sin(a)) for a in np.linspace(0,6.283,14)[:-1]],True)),'#31415f')
            c.line(H.P([(ex-.2,.18),(ex,.245),(ex+.2,.2)]),6)
    c.line(H.P3(H.nose_profile()[2:7]),2.4)
    c.line(H.P([(-.17,-.665),(0,-.655),(.17,-.665)]),2.6); c.line(H.P([(-.05,-.75),(.05,-.75)]),1.6)
im=c.rgba()[...,:3]; comp.save(sys.argv[1],im)
