"""Yuji hero keys: 1 laugh in evening light, 2 scream in the rain, 3 cold rage in ash.
   python chars/yuji_keys.py 1 2 3  ->  qa/yuji/hero<n>.jpg"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, math, cv2
cv2.setNumThreads(4)
from jj import comp, W, H
from jj.ink import Cel, spl
from jj.paint import Canvas, rgb, fbm, smooth, ramp, grid, arcs, f32
from chars import yuji as Y
A1 = np.ones((1, 3))


def bg1():
    x,y=grid(); t=np.clip((y/H)*.9+(x/W)*.25,0,1)
    im=ramp(t,[(0,'#f6b15f'),(.25,'#fcd58c'),(.5,'#f3b779'),(.75,'#c98a6e'),(1,'#7a6a6b')])
    # far hillside, foliage masses as blurred gouache dabs
    cv=Canvas(H,W,img=im); r=np.random.default_rng(4)
    n=420; xs=r.uniform(-100,W+100,n); base=640+70*np.sin(xs/310.)+30*np.sin(xs/97.+1); ys=base+np.abs(r.normal(0,1,n))*260
    d=np.clip((ys-base)/420,0,1)[:,None]; cols=(1-d)*rgb('#b98a55')+d*rgb('#4f5a49'); cols=cols+r.normal(0,.05,(n,1))*A1
    warm=(r.random(n)<.22*(xs<1000)); cols[warm]=rgb('#ffd98a')+r.normal(0,.04,(warm.sum(),1))
    cv.strokes(arcs(xs,ys,r.uniform(-.6,.6,n),r.uniform(90,240,n),r.normal(0,.6,n)),r.uniform(24,62,n),cols,op=.85,dry=.2,seed=2)
    im=cv2.GaussianBlur(cv.img,(0,0),17)
    # bokeh discs of light through leaves
    lay=np.zeros((H,W,3),f32)
    for k in range(46):
        cx,cy=r.uniform(-50,W),r.uniform(0,760); R=r.uniform(16,54); c=rgb(['#ffe9b0','#ffd07c','#fff3cf','#e9c07a'][k%4])*r.uniform(.05,.30)
        yy,xx=np.ogrid[:H,:W]; 
        x0,x1,y0,y1=int(max(0,cx-R-3)),int(min(W,cx+R+3)),int(max(0,cy-R-3)),int(min(H,cy+R+3))
        if x1<=x0 or y1<=y0: continue
        Yy,Xx=np.mgrid[y0:y1,x0:x1]; dd=np.hypot(Xx-cx,Yy-cy); m=np.clip(R-dd+.5,0,1)*(.7+.5*np.clip(dd/R,0,1)**4)
        lay[y0:y1,x0:x1]+=m[...,None]*c
    im=comp.screen(im,cv2.GaussianBlur(lay,(0,0),1.6))
    im=im+comp.radial(170,120,1300,power=2.2)[...,None]*rgb('#ffdf9e')*.55
    return im
def key1(out):
    im=bg1()
    h=Y.YHead().pose(1080,560,372,-.46,.0,-.10)
    L=Y.light('sunset',side=-1,amt=.02)
    a=Y.Yuji.bust(h,'laugh',L,f=6,wind=(.55,.12),seed=5,byaw=-.30,broll=.04)
    im=comp.over(im,a)
    # light wrap + air
    wrap=comp.blur(a[...,3],26)*(1-a[...,3]); 
    inner=np.clip(comp.blur(1-a[...,3],20)*a[...,3],0,1)
    im=comp.screen(im,(inner[...,None]*rgb('#ffcf86')*comp.radial(100,200,1500,power=1.)[...,None])*.35)
    im=im+comp.radial(120,60,1250,power=2.4)[...,None]*rgb('#ffd48a')*.30
    r=np.random.default_rng(9); lay=np.zeros((H,W,3),f32)
    for k in range(26):
        cx,cy=r.uniform(0,W),r.uniform(0,H); R=r.choice([2.2,3.2,5,9,16],p=[.3,.3,.2,.13,.07]); cv2.circle(lay,(int(cx*8),int(cy*8)),int(R*8),tuple(float(v) for v in rgb('#ffe6ae')*r.uniform(.15,.7)/(1+R*.12)),-1,cv2.LINE_AA,3)
    im=comp.screen(im,cv2.GaussianBlur(lay,(0,0),1.3))
    im=comp.finish(im,0,bloom=.34,vig=.16)
    comp.save(out,im,94)


def bg2():
    x,y=grid(); t=np.clip(y/H,0,1)
    im=ramp(t,[(0,'#5c6c92'),(.3,'#3a4468'),(.7,'#22283f'),(1,'#171a2b')])
    cv=Canvas(H,W,img=im); r=np.random.default_rng(4); n=260
    xs=r.uniform(-100,W+100,n); ys=r.uniform(300,H+100,n)
    c=rgb('#2b3250')[None]+r.normal(0,.03,(n,1)); cv.strokes(arcs(xs,ys,np.pi/2+r.normal(0,.15,n),r.uniform(120,420,n),0),r.uniform(20,70,n),c,op=.7,dry=.3,seed=3)
    im=cv2.GaussianBlur(cv.img,(0,0),24)
    im=comp.screen(im,comp.radial(430,40,900,power=2.)[...,None]*rgb('#a9c2f0')*.75)
    return im
def rain(cel,n,seed,L,w,col,al,ang=math.pi/2+.10):
    r=np.random.default_rng(seed); d=np.array([math.cos(ang),math.sin(ang)]); nn=np.array([-d[1],d[0]])
    for _ in range(n):
        p=np.array([r.uniform(-60,W+60),r.uniform(-200,H+60)]); ln=r.uniform(*L); ww=w*r.uniform(.5,1.4)
        cel.fill(np.array([p,p+d*ln*.8+nn*ww*.5,p+d*ln,p+d*ln*.8-nn*ww*.5]),col,al*r.uniform(.4,1))
def key2(out):
    im=bg2()
    c=Cel(); rain(c,260,1,(40,150),2.2,'#c9d8f4',.45); im=comp.over(im,cv2.GaussianBlur(c.layer(),(0,0),1.0))
    h=Y.YHead().pose(1000,440,690,.26,.20,.17)
    L=Y.light('rain',side=-1,amt=.60)
    a=Y.Yuji.bust(h,'scream',L,f=0,wind=0,seed=5,wet=1.,byaw=.06,raise_=.12)
    im=comp.over(im,a)
    inner=np.clip(comp.blur(1-a[...,3],26)*a[...,3],0,1)
    im=comp.screen(im,inner[...,None]*rgb('#9fb9ec')*.40*comp.radial(300,60,1500,power=1.)[...,None])
    im=comp.screen(im,comp.radial(380,-40,1200,power=2.2)[...,None]*rgb('#8fa9df')*.28)
    c=Cel(); rain(c,70,7,(120,420),4.2,'#e6eefc',.55); im=comp.over(im,cv2.GaussianBlur(c.layer(),(0,0),1.6))
    c=Cel(); rain(c,9,11,(380,760),15,'#e6eefc',.30); im=comp.over(im,comp.blur(c.layer(),9))
    im=comp.finish(im,0,bloom=.34,vig=.24)
    comp.save(out,im,94)


def bg3():
    x,y=grid(); t=np.clip(y/H,0,1)
    im=ramp(t,[(0,'#4d4c55'),(.35,'#7d7a7c'),(.62,'#b3a79c'),(.8,'#8c7468'),(1,'#4a3a3c')])
    r=np.random.default_rng(4)
    # smoke masses
    cv=Canvas(H,W,img=im); n=520
    xs=r.uniform(-100,W+100,n); ys=r.uniform(-60,H,n); d=np.clip(ys/H,0,1)[:,None]
    sm=fbm(H,W,(420,190,80),None,7); v=sm[np.clip(ys.astype(int),0,H-1),np.clip(xs.astype(int),0,W-1)][:,None]
    cols=(1-d)*rgb('#5a5860')+d*rgb('#9a8678')+v*.055+r.normal(0,.012,(n,1)); cv.strokes(arcs(xs,ys,-.25+r.normal(0,.5,n),r.uniform(120,360,n),r.normal(0,.8,n)),r.uniform(34,90,n),cols,op=.6,dry=.35,seed=2)
    im=cv2.GaussianBlur(cv.img,(0,0),9)
    # ruined towers leaning, silhouettes by air
    for k,(X,Wd,top,lean,col,bl) in enumerate([(60,190,330,30,'#77706f',7),(330,130,520,-24,'#8a8079',8),(1420,170,260,-60,'#6b6566',6),(1660,240,120,-40,'#5a565b',5),(1300,90,540,10,'#8c817a',9),(1840,200,380,20,'#4e4a50',5)]):
        c=Cel(); q=np.random.default_rng(k+20); m=8; xs_=np.linspace(0,Wd,m); tops=top+q.normal(0,24,m)+np.linspace(0,50*q.choice([-1,1]),m)
        P=[(X+lean+a,b) for a,b in zip(xs_,tops)]+[(X+Wd,H),(X,H)]; c.fill(np.array(P),col)
        for j in range(26):
            wx,wy=X+q.uniform(.1,.85)*Wd,q.uniform(top+60,H); c.fill(np.array([(wx,wy),(wx+16,wy),(wx+16,wy+24),(wx,wy+24)]),'#3f3b42',.5)
        im=comp.over(im,cv2.GaussianBlur(c.layer(),(0,0),bl))
    im=comp.screen(im,comp.radial(1560,1120,1000,620,power=1.6)[...,None]*rgb('#ff7a34')*.75)
    im=comp.screen(im,comp.radial(500,-100,900,power=2.)[...,None]*rgb('#e9edf2')*.35)
    return im
def ash(n,seed,smin,smax,bl,cols):
    c=Cel(); r=np.random.default_rng(seed)
    for _ in range(n):
        p=np.array([r.uniform(-20,W+20),r.uniform(-20,H+20)]); s=r.uniform(smin,smax); m=7; a=np.sort(r.uniform(0,2*np.pi,m)); rr=s*r.uniform(.4,1.2,m); th=r.uniform(0,3.14); sq=r.uniform(.3,1)
        Q=np.c_[np.cos(a)*rr,np.sin(a)*rr*sq]; R=np.array([[math.cos(th),-math.sin(th)],[math.sin(th),math.cos(th)]]); c.fill(Q@R.T+p,cols[r.integers(len(cols))],r.uniform(.5,1))
    return cv2.GaussianBlur(c.layer(),(0,0),bl) if bl>.3 else c.layer()
def key3(out):
    im=bg3()
    im=comp.over(im,ash(170,1,2,5,1.2,['#d9d2cc','#b8b0ac','#8f898a']))
    h=Y.YHead().pose(850,490,395,.36,.31,-.09,persp=.17)
    L=Y.light('ash',side=1,amt=.30,rim=Y.LIGHT['ash']['rim'])
    a=Y.Yuji.bust(h,Y.getx('rage',gy=-.016,gx=0.),L,f=3,wind=(.32,.10),seed=5,byaw=.30,bpitch=.22,damage=.9,broll=-.03)
    im=comp.over(im,a)
    inner=np.clip(comp.blur(1-a[...,3],22)*a[...,3],0,1)
    im=comp.screen(im,inner[...,None]*(rgb('#ff8a48')*comp.radial(1600,1100,1100,power=1.)[...,None]*.55+rgb('#dfe6ee')*comp.radial(400,0,1100,power=1.)[...,None]*.25))
    im=comp.screen(im,comp.radial(1500,1180,900,600,power=2.)[...,None]*rgb('#ff6a2a')*.35)
    im=comp.over(im,ash(70,5,4,9,1.0,['#e6dfd8','#c6bdb8','#6d6668']))
    im=comp.over(im,ash(16,9,16,34,7,['#d8d0ca','#7a7274','#a89e9a']),.8)
    c=Cel(); r=np.random.default_rng(3); g=np.zeros((H,W,3),f32)
    for k in range(95):
        cx,cy=r.uniform(380,W+40),r.uniform(60,H+20)
        if 640<cx<1150 and 150<cy<860 and r.random()<.93: continue
        R=r.choice([1.5,2.3,3.2,5.0],p=[.45,.3,.18,.07]); th=-.75+r.normal(0,.35); l=R*r.uniform(2.,6.); bend=r.normal(0,.5)
        u=np.linspace(-1,1,9); P=np.c_[cx+np.cos(th+bend*u*.5)*u*l, cy+np.sin(th+bend*u*.5)*u*l]
        c.line(P,R*1.6,['#ffc46a','#ff8c3c','#fff0bc'][k%3],taper=(.5,.4),seed=k)
    em=c.layer(); im=comp.over(im,em)+comp.blur(em[...,:3],5)*1.2+comp.blur(em[...,:3],16)*1.4
    im=comp.finish(im,0,bloom=.36,vig=.26)
    comp.save(out,im,94)


if __name__ == '__main__':
    os.makedirs('qa/yuji', exist_ok=True)
    for a in sys.argv[1:] or ['1', '2', '3']: {'1': key1, '2': key2, '3': key3}[a](f'qa/yuji/hero{a}.jpg')
