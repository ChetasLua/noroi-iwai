"""python3 tools/share.py b1 b4 ...  -> out/share/<id>.jpg : six frames of a rendered shot (2 x 3, 960 wide each).  Light: ffmpeg only, one at a time."""
import sys, subprocess, os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, '.')
import film
for s in sys.argv[1:]:
    n = film.LEN[s]; k = [int(n * t) for t in (.04, .22, .4, .56, .74, .93)]
    sel = '+'.join(f'eq(n\\,{q})' for q in k)
    r = subprocess.run(['.mm/env/bin/ffmpeg', '-y', '-v', 'error', '-threads', '2', '-i', f'out/{s}.mkv', '-vf', f"select='{sel}',scale=960:-2,tile=2x3", '-frames:v', '1', '-q:v', '3', f'out/share/{s}.jpg'])
    subprocess.run(['.mm/env/bin/ffmpeg', '-y', '-v', 'error', '-i', f'out/share/{s}.jpg', '-vf', 'scale=1000:-2', '-q:v', '5', f'out/share/small_{s}.jpg'])
    print(s, r.returncode)
