"""figure kit hero still (also a usage example): python tools/fig_hero1.py -> qa/figure"""
import sys, os; ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, cv2, math, time
cv2.setNumThreads(3)
from jj.cam import Camera3D
from jj.figure import *
from jj.fig_street import *
from jj.ink import Cel, spl
from jj import comp, kit, fx
from jj.paint import fbm, grid, smooth, Canvas
cam=Camera3D((.30,-1.50,.25),(.06,-.35,.75),focal=950,roll=math.radians(-12))
pose=P(lead='root',root=(.01,0,-.02),pel=(44,-6,-26),wst=(8,-2,10),cst=(8,-2,22),nck=(-22,2,-8),hed=(-30,1,-4),cL=(2,8),cR=(6,-5),
    hL=(.34,.82,-.52),hR=(.62,-.60,.34),eL=(1,.1,.1),eR=(.5,-.6,1),gL='fist',gR='fist',wL=(-85,-10,0),wR=(-30,-10,0),
    fR=(.18,.32,.085),fL=(.05,-.50,.36),kR=(.65,1,.5),kL=(.1,1,-.7),tR=(12,14,0),tL=(6,-84,0))
F=Fig('yuji').pose(pose).place(0,0,0,math.pi+.08); F._fx={'hood':np.array([.03,.12,.06]),'hair':np.array([.05,.5,.1])}
key='cache/fig_hero1_bg.npy'
if os.path.exists(key) and '--bg' not in sys.argv: bg=np.load(key)
else: bg=street(cam,3); np.save(key,bg)
t=time.time()
L=F.draw(cam,0,light=(-.55,.75,.35),kick=(-.85,.30,-.35),expr=dict(mood='shout'),rim=dict(dir=(.92,.12,-.42),col='#ff9c48',thr=.52,amt=.82),pal=dict(mul='#cdd2f0',shmul='#a9addb',litmul='#dfe6ff'))
print('fig',time.time()-t,F.info['px'])
Lz=F.mask('handL','farmL')
import jj.fx_break as FB
RK=dict(lit='#d98a52',mid='#4a4658',dark='#232234',ink='#0b0a12')
# ---- background: depth of field, zoom streaks at the edges, atmosphere
img=comp.blur(bg,2.6)
x,y=grid(); d=np.sqrt(((x-930)/960)**2+((y-470)/620)**2); w=smooth(.55,1.25,d)[...,None]
zb=kit.zoomblur(img,930,430,.16,15); img=img*(1-w)+zb*w
img=comp.screen(img,(np.exp(-((y-860)/190)**2)*np.exp(-((x-1250)/480)**2))[...,None]*np.array([1.,.52,.25],np.float32)*.30)
# road: crater at the striking foot, contact shadow, rubble
g=fx.ground_proj(cam,centre=(.20,-.60))
rb=rubble(cam,130,9,(-7.5,7.5,1.5,26.),(.15,1.2)); img=comp.over(img,rb)
c=Cel(); fx.ground_crack(c,g,.95,.62,seed=5,n=12,col='#07070f',lip='#8f86a0',side='#1c1a2a',top='#3c3a4e',topdark='#2a2a3c',glow='#58c8ff',w=.10,lift=.16,lw=2.0)
img=comp.over(img,c.layer())
foot=F.info['feet']['R']; 
sh=np.zeros((1080,1920),np.float32); cv2.ellipse(sh,(int(foot[0])-10,int(foot[1])+92),(230,44),-12,0,360,1.,-1); img*=1-comp.blur(sh,14)[...,None]*.62
# dust thrown back from the foot
c=Cel(); fx.dust_skirt(c,7,foot[0]+20,foot[1]+60,330,70,120,seed=4,n=14,pal=dict(lit='#8a8098',mid='#565268',dark='#363448',ink='#14121e')); img=comp.over(img,c.layer(),.92)
# energy
bk,fr=fx.cursed(Lz,6,stream=(.75,-.95),size=150,seed=3,amount=1.0,wisps=1.1)
img=fx.put(img,bk)
img=comp.over(img,L)
img=fx.put(img,fr)
h=F.info['hands']['L']
img=comp.screen(img,kit.radial(h[0],h[1],520,power=2.2)[...,None]*np.array([.16,.52,1.],np.float32)*.34)
# flying stones, near ones out of focus
c=Cel(); fx.debris(c,9,(foot[0]+20,foot[1]+50),pal=RK,light=(.8,-.35,.2),n=26,seed=7,speed=(300,1500),size=(6,46),spread=(60,10),trails=True); img=comp.over(img,c.layer())
nd=Cel(); r=np.random.default_rng(3)
for X,Y,q,s in ((120,930,120,1),(1770,900,150,2),(1660,120,70,3),(260,180,60,4)):
    fx.rock(nd,X,Y,q,tuple(r.uniform(0,6,3)),seed=s,pal=RK,light=(.8,-.3,.1))
img=comp.over(img,comp.bokeh(nd.layer(),9,1.0))
# speed lines
c=Cel(); kit.focus_lines(c,900,430,760,1500,n=70,w=3.2,col='#e8ecff',seed=4,ry=.62); img=comp.over(img,c.layer(),.16)
c=Cel(); kit.focus_lines(c,900,430,820,1500,n=60,w=4.2,col='#05060c',seed=8,ry=.62); img=comp.over(img,c.layer(),.5)
out=comp.finish(img,0,bloom=.34,vig=.2)
comp.save('qa/figure/hero_sprint.jpg',comp.to8(out),93)
