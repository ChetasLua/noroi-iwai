"""python tools/render.py d4 [--range a:b] [--jobs 6] [--every 1]  ->  out/<shot>.mkv (near-lossless)"""
import sys, os, importlib, argparse, time, subprocess
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault('NUMBA_NUM_THREADS', '2')
import numpy as np, cv2
from multiprocessing import Pool
from jj import comp, W, H
FF = os.path.join(ROOT, '.mm/env/bin/ffmpeg'); S = None


def init(name):
    global S
    cv2.setNumThreads(1); S = importlib.import_module('shots.' + name).Shot(); S.setup()


def one(f):
    return f, comp.to8(comp.finish(S.frame(f), f, **getattr(S, 'finish', {})), f)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('shot'); ap.add_argument('--range'); ap.add_argument('--jobs', type=int, default=5); ap.add_argument('--out')
    a = ap.parse_args(); t = time.time()
    s0 = importlib.import_module('shots.' + a.shot).Shot(); s0.setup(); n = s0.n; del s0
    r = [int(v) for v in a.range.split(':')] if a.range else [0, n]
    out = a.out or f'out/{a.shot}.mkv'
    p = subprocess.Popen([FF, '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', '24', '-i', '-', '-threads', '3', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '6', '-pix_fmt', 'yuv444p', out], stdin=subprocess.PIPE)
    with Pool(a.jobs, init, (a.shot,)) as pool:
        for f, im in pool.imap(one, range(*r), chunksize=2): p.stdin.write(im.tobytes())
    p.stdin.close(); p.wait(); print(out, '%d frames %.0fs' % (r[1] - r[0], time.time() - t))
