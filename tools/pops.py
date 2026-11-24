"""frame-difference report of the finished film: single-frame spikes that are not cuts, long frozen runs, near-black / near-white frames"""
import sys, subprocess, numpy as np
sys.path.insert(0, '.')
import film
p = subprocess.run(['.mm/env/bin/ffmpeg', '-nostdin', '-v', 'error', '-i', sys.argv[1], '-vf', 'scale=192:108:flags=area', '-f', 'rawvideo', '-pix_fmt', 'gray', '-'], capture_output=True).stdout
a = np.frombuffer(p, np.uint8).reshape(-1, 108, 192).astype(np.float32); print('frames', len(a))
d = np.r_[0, np.abs(np.diff(a, axis=0)).mean((1, 2))]
cuts = {s[1] for s in film.SHOTS}
sh = lambda f: next((s, f - b) for s, b, n in reversed(film.SHOTS) if f >= b)
# returns to the same picture after one odd frame = flicker/glitch candidate
for f in range(2, len(a) - 2):
    if f in cuts or f + 1 in cuts: continue
    back = np.abs(a[f + 1] - a[f - 1]).mean()
    if d[f] > 14 and d[f + 1] > 14 and back < .35 * min(d[f], d[f + 1]): print('one-frame flash', f, sh(f), 'd=%.0f back=%.0f' % (d[f], back))
run = 0
for f in range(1, len(a)):
    if d[f] < .35: run += 1
    else:
        if run >= 14: print('frozen run of %d frames ends at' % run, f, sh(f))
        run = 0
m = a.mean((1, 2)); 
for lab, idx in (('near black', np.where(m < 4)[0]), ('near white', np.where(m > 246)[0])):
    if len(idx):
        g = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1); print(lab, [(int(q[0]), len(q), sh(int(q[0]))[0]) for q in g])
