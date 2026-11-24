"""python tools/look.py out.jpg d4:40 d5:20 ... [--w 960] [--cols 2]   contact sheet of finished frames from any shots
   python tools/look.py out.jpg --film 24                          one frame every 24 film frames from out/*.mkv (what is rendered)"""
import sys, os, importlib, subprocess
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, cv2
cv2.setNumThreads(4)
from jj import comp
a = sys.argv[1:]; out = a.pop(0); w = 960; n = 2; film = None; toks = []
while a:
    t = a.pop(0)
    if t == '--w': w = int(a.pop(0))
    elif t == '--cols': n = int(a.pop(0))
    elif t == '--film': film = int(a.pop(0))
    else: toks.append(t)
ims = []; h = w * 9 // 16
if film:
    import film as FM
    cap = {}
    for s, a0, ln in FM.SHOTS:
        p = f'out/{s}.mkv'
        for f in range(a0, a0 + ln):
            if f % film: continue
            if not os.path.exists(p): ims.append(np.zeros((h, w, 3), np.uint8)); continue
            raw = subprocess.run(['.mm/env/bin/ffmpeg', '-v', 'error', '-ss', str((f - a0) / 24.), '-i', p, '-frames:v', '1', '-vf', f'scale={w}:{h}:flags=area', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True).stdout
            im = np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy() if len(raw) == h * w * 3 else np.zeros((h, w, 3), np.uint8)
            cv2.putText(im, f'{s}:{f-a0}', (6, h - 8), 0, .5, (255, 255, 0), 1); ims.append(im)
else:
    cache = {}
    for t in toks:
        k, f = t.split(':'); f = int(f)
        if k not in cache:
            S = importlib.import_module('shots.' + k).Shot(); S.setup(); cache[k] = S
        S = cache[k]; im = comp.to8(comp.finish(S.frame(f), f, **getattr(S, 'finish', {})), f)
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA); cv2.putText(im, t, (6, h - 8), 0, .5, (255, 255, 0), 1); ims.append(im)
while len(ims) % n: ims.append(np.zeros_like(ims[0]))
comp.save(out, np.vstack([np.hstack(ims[i:i + n]) for i in range(0, len(ims), n)]), 86); print(out, len(ims))
