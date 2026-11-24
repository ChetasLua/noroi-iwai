"""QA for the band: python band_qa.py spec in.wav out.jpg [t0 t1]  |  bands in.wav   |  (log-f spectrogram, band balance per section)"""
import sys, numpy as np, cv2
from scipy.io import wavfile
from scipy.signal import stft
cv2.setNumThreads(2)
def load(p):
    sr, x = wavfile.read(p); dt = x.dtype; x = x.astype(np.float32)
    if dt != np.float32: x = x / (2**31 if dt == np.int32 else 32768.)
    return sr, x if x.ndim == 2 else np.stack([x, x], 1)
def spec(p, out, t0=0, t1=None, W=1800, H=700):
    sr, x = load(p); x = x[int(t0*sr):int(t1*sr) if t1 else None].mean(1)
    n = 4096; hop = max(128, len(x)//W); f, t, Z = stft(x, sr, nperseg=n, noverlap=n-hop if hop<n else 0, boundary=None, padded=True)
    S = 20*np.log10(np.abs(Z)+1e-7); fr = np.geomspace(25, 20000, H); idx = np.clip(fr/(sr/n), 0, S.shape[0]-1.001)
    i = idx.astype(int); a = (idx-i)[:, None]; L = S[i]*(1-a)+S[i+1]*a; L = cv2.resize(L[::-1].astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    v = np.clip((L+100)/90, 0, 1); im = cv2.applyColorMap((v*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    for q in (50, 100, 200, 500, 1000, 2000, 5000, 10000):
        y = int(H-1-(np.log(q/25)/np.log(20000/25))*(H-1)); cv2.line(im, (0, y), (W, y), (90, 90, 90), 1); cv2.putText(im, str(q), (2, y-2), 0, .4, (200, 200, 200), 1)
    dur = len(x)/sr; step = 1 if dur < 30 else 6
    for s in np.arange(0, dur, step): cv2.putText(im, '%.1f' % (s+t0), (int(s/dur*W), H-4), 0, .4, (255, 255, 255), 1)
    e = np.abs(x); k = max(1, len(e)//W); env = e[:k*W].reshape(W, k).max(1); wv = np.zeros((140, W, 3), np.uint8)
    for c in range(W): cv2.line(wv, (c, 70-int(env[c]*68)), (c, 70+int(env[c]*68)), (200, 200, 120), 1)
    cv2.imwrite(out, np.vstack([im, wv]), [1, 88])
SEC = [('A', 0, 4), ('B12', 4, 8), ('B3', 8, 10), ('B4', 10, 12), ('B5', 12, 14), ('B6', 14, 16), ('C1', 16, 20), ('C2', 20, 22), ('C4', 22, 24), ('D1', 24, 26), ('D2', 26, 28), ('D3', 28, 30), ('D4', 30, 31), ('D4b', 31.5, 32), ('D5', 32, 34), ('D6', 34, 36), ('E12', 36, 40), ('E34', 40, 44), ('E5', 44, 47.5), ('F1', 48, 50), ('F2', 50, 52), ('F3', 52, 54), ('F4', 54, 56), ('F5', 56, 58), ('H1', 58, 60), ('H2', 60, 62), ('H3', 62, 64), ('H4', 64, 66), ('G1', 66, 70), ('G3', 70, 73), ('G4', 73, 76)]
BANDS = [(20, 60), (60, 120), (120, 300), (300, 800), (800, 2000), (2000, 5000), (5000, 10000), (10000, 20000)]
def bands(p):
    sr, x = load(p); print('sec    rms   ' + ' '.join('%6s' % ('%d' % a) for a, b in BANDS) + '   corr  (dB rel. total)')
    for nm, a, b in SEC:
        s = x[int(a*1.5*sr):int(b*1.5*sr)]
        if len(s) < 100: continue
        m = s.mean(1); P = np.abs(np.fft.rfft(m*np.hanning(len(m))))**2; f = np.fft.rfftfreq(len(m), 1/sr); tot = P.sum()+1e-12
        c = (s[:, 0]*s[:, 1]).mean()/(np.sqrt((s[:, 0]**2).mean()*(s[:, 1]**2).mean())+1e-12)
        print('%-5s %6.1f  ' % (nm, 10*np.log10((s**2).mean()+1e-12)) + ' '.join('%6.1f' % (10*np.log10(P[(f >= u) & (f < v)].sum()/tot+1e-12)) for u, v in BANDS) + '  %5.2f' % c)
if __name__ == '__main__':
    a = sys.argv
    if a[1] == 'spec': spec(a[2], a[3], *(float(v) for v in a[4:6]))
    elif a[1] == 'bands': bands(a[2])
HITS = [54, 108, 126, 198, 306, 333, 351, 360, 369, 378, 387, 414, 486, 504, 558, 576, 612, 621, 630, 639, 702, 720, 738, 756, 774, 792, 801, 810, 819, 864, 891, 909, 954, 972, 1035, 1044, 1053, 1062, 1071, 1134, 1152, 1206, 1476, 1728, 1764, 1782, 1854, 1872, 1890, 1908, 1917, 1926, 1944, 1962, 1980, 1998, 2016, 2088, 2124, 2142, 2196, 2205, 2214, 2223, 2232, 2241, 2250, 2259, 2268, 2304, 2358, 2520, 2700]
SIL = [(1123, 1133.8, 'f1116 drop'), (1284, 1290, 'void end'), (1290, 1295.8, 'silence'), (1345, 1362, 'clinks'), (1711, 1727, 'cut'), (2024, 2051.8, 'whisper bar'), (2328, 2357.8, 'before impact')]
def hits(p):
    sr, x = load(p); m = np.abs(x).max(1); h = 48; e = np.add.reduceat(x.mean(1)**2, np.arange(0, len(x), h))[:len(x)//h]
    import scipy.ndimage as nd
    from scipy.signal import butter, sosfilt
    y = sosfilt(butter(4, 160, 'hp', fs=sr, output='sos'), x.mean(1)); e = np.add.reduceat(y**2, np.arange(0, len(y), h))[:len(y)//h]   # subs ripple: judge the attack above 160 Hz
    f = nd.uniform_filter1d(e, 6); db = 10*np.log10(f+1e-9); bad = 0
    for fr in HITS:
        c = int(fr/24*sr/h); pre = db[max(0, c-70):c-8].mean(); seg = db[c-40:c+60]; d = np.diff(nd.uniform_filter1d(seg, 4)); k = int(np.argmax(d)); rise = seg[min(99, k+14)]-seg[max(0, k-10)]
        off = (k-40)*h/sr*1000; post = db[c+2:c+60].max()
        flag = '' if abs(off) < 12 and rise > 2 else '   <--'; bad += bool(flag)
        print('f%4d  onset %+6.1f ms  rise %5.1f dB  level before %6.1f after %6.1f%s' % (fr, off, rise, pre, post, flag))
    for a, b, n in SIL: s = x[int(a/24*sr):int(b/24*sr)]; print('silence f%d-%d %-14s %6.1f dB rms' % (a, b, n, 10*np.log10((s**2).mean()+1e-12)))
    print('flagged', bad)
if __name__ == '__main__' and sys.argv[1] == 'hits': hits(sys.argv[2])
