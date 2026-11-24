"""plates on disk: float16 .npy per plate, opened memory-mapped, turned to float32 only while a frame uses them"""
import numpy as np, os, glob


class Plates:
    def __init__(s, ver, build):
        s.ver = ver; os.makedirs('cache', exist_ok=True); ok = f'cache/{ver}__ok'
        if not os.path.exists(ok):
            old = f'cache/{ver}.npz'
            D = dict(np.load(old)) if os.path.exists(old) else build()
            for k, v in D.items():
                v = np.asarray(v); v = v.astype(np.float16) if v.dtype in (np.float32, np.float64) else v
                t = f'cache/{ver}__{k}.{os.getpid()}.npy'; np.save(t, v); os.replace(t, f'cache/{ver}__{k}.npy')
            open(ok, 'w').close()
            if os.path.exists(old): os.remove(old)
        s.m = {p[len(f'cache/{ver}__'):-4]: np.load(p, mmap_mode='r') for p in glob.glob(f'cache/{ver}__*.npy')}

    def __contains__(s, k): return k in s.m

    def __getitem__(s, k):
        a = s.m[k]; return a.astype(np.float32) if a.dtype == np.float16 else np.asarray(a)
