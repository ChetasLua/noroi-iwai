"""Cached drawings for Mahoraga's shots.  A shot module defines VER (int or {function: int}), JOBS = {key: (function name, args)}; each function returns a
premultiplied RGBA layer (H,W,4), an RGB plate, a mono map (H,W), or a dict of them.  ensure(shot) draws what is missing, one after another in this process.
Stored as float16 .npy cropped to the ink, read with mmap; Lay composites only its own rectangle."""
import os, sys, json, gc, hashlib, importlib, numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
f32 = np.float32


def _mod(shot): return importlib.import_module('shots.' + shot)


def base(shot, key):
    m = _mod(shot); fn = m.JOBS[key][0]; v = m.VER.get(fn, 1) if isinstance(m.VER, dict) else m.VER
    return os.path.join(ROOT, 'cache', f'{shot}_{key}_' + hashlib.md5(repr((v, m.JOBS[key])).encode()).hexdigest()[:8])


class Lay:
    """a cropped drawing: a (float16, mmap), x, y, full shape"""
    __slots__ = ('a', 'x', 'y', 'shape')
    def __init__(s, a, x, y, shape): s.a, s.x, s.y, s.shape = a, int(x), int(y), tuple(shape)
    def roi(s): return np.asarray(s.a, f32)
    def full(s):
        o = np.zeros(s.shape, f32); h, w = s.a.shape[:2]
        if h: o[s.y:s.y + h, s.x:s.x + w] = s.a
        return o
    def over(s, im, op=1.0):
        """composite onto an RGB float frame, in place"""
        h, w = s.a.shape[:2]
        if not h: return im
        b = im[s.y:s.y + h, s.x:s.x + w]; a = s.roi(); b *= (1 - a[..., 3:4] * op); b += a[..., :3] * op; return im


def _pack(a):
    a = np.asarray(a); m = np.abs(a).max(-1) > 1e-3 if a.ndim == 3 else np.abs(a) > 1e-3; ys, xs = np.nonzero(m)
    if len(ys) == 0: return np.zeros((0, 0) + a.shape[2:], np.float16), (0, 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1; return a[y0:y1, x0:x1].astype(np.float16), (int(y0), int(x0))


def draw_one(shot, key):
    m = _mod(shot); fn, args = m.JOBS[key]; r = getattr(m, fn)(*args); r = r if isinstance(r, dict) else dict(a=r); p = base(shot, key); meta = {}
    for k, v in r.items():
        a, o = _pack(v); t = f'{p}.{k}.{os.getpid()}.tmp.npy'; np.save(t, a); os.replace(t, f'{p}.{k}.npy'); meta[k] = [*o, *[int(q) for q in np.shape(v)]]
    t = p + '.%d.tmp' % os.getpid(); json.dump(meta, open(t, 'w')); os.replace(t, p + '.json'); del r; gc.collect()


def missing(shot): return [k for k in _mod(shot).JOBS if not os.path.exists(base(shot, k) + '.json')]


def ensure(shot, verbose=True):
    import time
    for k in missing(shot):
        t = time.time(); draw_one(shot, k)
        if verbose: print(shot, k, '%.1fs' % (time.time() - t), flush=True)


def get(shot, key):
    p = base(shot, key); meta = json.load(open(p + '.json')); out = {k: Lay(np.load(f'{p}.{k}.npy', mmap_mode='r'), v[1], v[0], v[2:]) for k, v in meta.items()}
    return out['a'] if list(out) == ['a'] else out


def val(shot, key, name=None):
    """full float32 array"""
    r = get(shot, key); return (r if name is None else r[name]).full()


def prune(shot):
    """delete cached drawings of older versions"""
    keep = {os.path.basename(base(shot, k)) for k in _mod(shot).JOBS}; d = os.path.join(ROOT, 'cache'); n = 0
    for f in os.listdir(d):
        if f.startswith(shot + '_') and f.split('.')[0] not in keep: os.remove(os.path.join(d, f)); n += 1
    return n


if __name__ == '__main__':
    shot, only = sys.argv[1], sys.argv[2:]
    if only:
        import time
        for k in _mod(shot).JOBS:
            if any(k.startswith(o) for o in only): t = time.time(); draw_one(shot, k); print(k, '%.1fs' % (time.time() - t), flush=True)
    else: ensure(shot)
    print('pruned', prune(shot))
