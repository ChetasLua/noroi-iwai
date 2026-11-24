"""figure kit hero still (also a usage example): python tools/fig_hero2.py -> qa/figure"""
import sys, os; ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, cv2, math, time
cv2.setNumThreads(3)
from jj.cam import Camera3D
from jj.figure import *
from jj.fig_street import *
from jj.ink import Cel, spl
from jj import comp, kit, fx
from jj.paint import fbm, grid, smooth, Canvas, rgb
cam=Camera3D((-.10,-3.05,.40),(.02,0,1.04),focal=1330,roll=math.radians(8))
st=lambda p,x,y,z,h: dict(pose=p,x=x,y=y,z=z,heading=h)
YP=mirror(POSES['dive_punch']); YP=dict(YP); YP['hL']=(-.06,.96,-.30)
KP=mirror(POSES['block_back'])
y1=st(YP,-.60,-.06,-.04,-math.pi/2-.10); y0=st(mirror(POSES['punch_wind']),-1.05,-.06,.30,-math.pi/2-.10)
k1=st(KP,.72,.05,0,math.pi/2+.10)
Y=Fig('yuji').pose(y1['pose']).place(y1['x'],y1['y'],y1['z'],y1['heading']); Y._fx={'hood':np.array([-.08,0,.05]),'hair':np.array([-.5,0,.15]),'hem':np.array([-.05,0,.02])}
K=Fig('sukuna',seed=3).pose(k1['pose']).place(k1['x'],k1['y'],k1['z'],k1['heading']); K._fx={'hem':np.array([.07,0,.03]),'sleeveL':np.array([.9,0,.3]),'sleeveR':np.array([.6,.1,.2]),'hair':np.array([.6,0,.2])}
key='cache/fig_hero2_bg.npy'
if os.path.exists(key) and '--bg' not in sys.argv: bg=np.load(key)
else: bg=street(cam,12,ASH,width=9.5,lit=.03,damage=1.6); np.save(key,bg)
t=time.time()
lg=dict(light=(-.35,.8,.45),kick=(-.85,.35,-.35))
pal=dict(mul='#f4e2d2',shmul='#c6a6a8',litmul='#fff0dc'); rim=dict(dir=(.55,.35,-.75),col='#ffb060',thr=.50,amt=.85)
LK=K.draw(cam,0,expr=dict(mood='grin',look=(.6,.3)),pal=pal,rim=rim,**lg)
LY=Y.draw(cam,0,expr=dict(mood='shout',look=(.4,-.2)),pal=pal,rim=rim,**lg)
SM=Y.smear(cam,y0,y1,'armL',pal=pal,seed=3,tail=.8)
print('figs',time.time()-t,Y.info['px'],K.info['px'])
hit=np.array(Y.info['hands']['L'],float)+np.array([26,4]); print(hit)
img=comp.blur(bg,2.0)
x,yy=grid()
# fire glow and ash air
img=comp.screen(img,(np.exp(-((yy-760)/230)**2)*np.exp(-((x-1180)/620)**2))[...,None]*np.array([1.,.55,.25],np.float32)*.42)
rb=rubble(cam,170,31,(-9,9,2.2,40.),(.10,.85),pal=ASH,light=(.5,-.75,.3)); img=comp.over(img,rb)
# shadows on the ground
g=fx.ground_proj(cam,centre=(.72,.05))
sh=np.zeros((1080,1920),np.float32)
for cx,w,hh in ((K.info['feet']['R'][0],300,46),(K.info['feet']['L'][0],260,40)):
    py=cam.project(np.array([.72,.05,0]))[0][1]; cv2.ellipse(sh,(int(cx),int(py)+20),(w,hh),8,0,360,1.,-1)
py=cam.project(np.array([-.6,0,0]))[0]; cv2.ellipse(sh,(int(py[0]),int(py[1])),(300,34),8,0,360,.7,-1)
img*=1-comp.blur(sh,16)[...,None]*.55
c=Cel(); fx.ground_crack(c,g,1.15,.5,seed=9,n=12,col='#1a1216',lip='#d0b8a4',side='#3a2c30',top='#7a6a66',topdark='#5c4e50',w=.08,lift=.10,lw=2.); img=comp.over(img,c.layer())
c=Cel(); fx.dust_skirt(c,6,float(py[0])+620,float(py[1])+60,560,80,130,seed=2,n=18,pal=dict(lit='#d8b89a',mid='#a08474',dark='#6a5654',ink='#2a1e20')); img=comp.over(img,c.layer(),.8)
# shock behind the bodies
img=comp.screen(img,kit.radial(hit[0],hit[1],620,power=2.)[...,None]*np.array([1.,.86,.66],np.float32)*.55)
c=Cel(); kit.focus_lines(c,hit[0],hit[1],240,1500,n=90,w=4.2,col='#fff2dc',seed=2,ry=.8); img=comp.over(img,c.layer(),.40)
c=Cel(); kit.focus_lines(c,hit[0],hit[1],330,1500,n=80,w=5,col='#140c10',seed=5,ry=.8); img=comp.over(img,c.layer(),.55)
tl=kit.smear(LY,-60,-24) if False else None
from jj.figure import tails
img=comp.over(img,tails(LY,(150,52),n=90,seed=4,width=(2,8)),.8)
img=comp.over(img,tails(LK,(95,10),n=50,seed=6,width=(2,7)),.7)
img=comp.over(img,LK)
img=comp.over(img,SM)
img=comp.over(img,LY)
# the hit: hard star, ring, sparks
c=Cel(); P=fx.star_poly(hit[0],hit[1],26,150,13,5,asp=.8) if hasattr(fx,'star_poly') else None
from jj.fx_flash import star_poly
P=star_poly(hit[0],hit[1],22,135,11,5,asp=.75,rot=.3); c.fill(P,'#fff6e2'); kit.outline(c,P,2.0,'#17131a',seed=2)
P2=star_poly(hit[0],hit[1],14,70,9,9,asp=.75,rot=.1); c.fill(P2,'#ffffff')
r=np.random.default_rng(8)
for i in range(26):
    a=r.uniform(0,6.28); d0=r.uniform(60,160); d1=d0+r.uniform(40,220); p0=hit+np.array([math.cos(a),math.sin(a)*.8])*d0; p1=hit+np.array([math.cos(a),math.sin(a)*.8])*d1
    c.line(np.array([p0,(p0+p1)/2,p1]),r.uniform(1.5,5),'#fff0d0' if r.random()<.6 else '#17131a',taper=(.2,.8),seed=i)
img=comp.over(img,c.layer())
c=Cel(); fx.debris(c,7,(hit[0]-260,hit[1]+430),n=20,seed=3,speed=(300,900),ang=(-2.9,-1.2),size=(5,30),spread=(260,30),pal=dict(lit='#f0c090',mid='#7a6462',dark='#3a2c30',ink='#17131a')); img=comp.over(img,c.layer())
img=comp.screen(img,kit.radial(hit[0],hit[1],330,power=2.6)[...,None]*np.array([1.,.8,.55],np.float32)*.5)
# embers
c=Cel(); r=np.random.default_rng(5)
for i in range(70):
    p=np.array([r.uniform(0,1920),r.uniform(0,1080)]); l=r.uniform(3,16); a=r.uniform(-.9,-.2)
    c.line(np.array([p,p+np.array([math.cos(a),math.sin(a)])*l*.5,p+np.array([math.cos(a),math.sin(a)])*l]),r.uniform(1.2,3.4),'#ffb45a' if r.random()<.7 else '#ffe6b0',taper=(.3,.5),seed=i)
img=comp.over(img,c.layer())
out=comp.finish(img,0,bloom=.34,vig=.2)
comp.save('qa/figure/hero_fight.jpg',comp.to8(out),93)
