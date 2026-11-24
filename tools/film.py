"""python tools/film.py [--jobs 14] [--only a1,d4] [--assemble]
Renders every shot to out/<id>.mkv (near-lossless) if missing or older than its sources (--only forces a list), then --assemble joins them
with out/soundtrack.wav into dist/noroi-iwai.mp4 (1920x1080, 24 fps, H.264 high bitrate, AAC 320k, English lyric subtitles as a soft track)."""
import sys, os, argparse, subprocess, json, time, glob
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); os.chdir(ROOT)
import film as FM
FF = '.mm/env/bin/ffmpeg'; PY = '.mm/env/bin/python'


def srt():
    L = json.load(open('audio/score.json'))['lines']
    def ts(s): h = int(s // 3600); m = int(s % 3600 // 60); return '%02d:%02d:%06.3f' % (h, m, s % 60)
    out = []
    for i, l in enumerate(L):
        if l['voice'] == 'whisper' or not l['notes']: pass
        a = l['notes'][0]['t'] * .1875 - .05; b = max(n['t'] + n['d'] for n in l['notes']) * .1875 + .1
        if i + 1 < len(L): b = min(b, L[i + 1]['notes'][0]['t'] * .1875 - .06)
        out.append('%d\n%s --> %s\n%s\n' % (i + 1, ts(a).replace('.', ','), ts(b).replace('.', ','), l['en']))
    open('dist/noroi-iwai.en.srt', 'w').write('\n'.join(out))


def check():
    bad = []
    for s, a, n in FM.SHOTS:
        p = f'out/{s}.mkv'
        if not os.path.exists(p): bad.append((s, 'missing')); continue
        r = subprocess.run(['.mm/env/bin/ffprobe', '-v', 'error', '-count_packets', '-select_streams', 'v', '-show_entries', 'stream=nb_read_packets', '-of', 'csv=p=0', p], capture_output=True, text=True).stdout.strip().strip(',')
        if r != str(n): bad.append((s, r, n))
    return bad


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--jobs', type=int, default=14); ap.add_argument('--only'); ap.add_argument('--assemble', action='store_true'); ap.add_argument('--crf', default='13'); ap.add_argument('--out', default='dist/noroi-iwai.mp4')
    a = ap.parse_args()
    if not a.assemble or a.only:
        ids = a.only.split(',') if a.only else [s for s, _, _ in FM.SHOTS if os.path.exists(f'shots/{s}.py')]
        for s in ids:
            t = time.time(); r = subprocess.run([PY, 'tools/render.py', s, '--jobs', str(a.jobs)], capture_output=True, text=True)
            print(s, 'ok' if r.returncode == 0 else 'FAIL ' + r.stderr[-600:], '%.0fs' % (time.time() - t), flush=True)
    if a.assemble:
        bad = check(); print('check', bad or 'all shots complete')
        if bad and not os.environ.get('FORCE'): sys.exit(1)
        srt(); open('out/list.txt', 'w').write(''.join(f"file '{os.path.abspath('out/' + s + '.mkv')}'\n" for s, _, _ in FM.SHOTS if os.path.exists(f'out/{s}.mkv')))
        cmd = [FF, '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-r', '24', '-i', 'out/list.txt', '-i', 'out/soundtrack.wav', '-i', 'dist/noroi-iwai.en.srt', '-map', '0:v', '-map', '1:a', '-map', '2:s',
               '-vf', 'scale=in_range=tv:out_range=tv:in_color_matrix=bt601:out_color_matrix=bt709:flags=lanczos+accurate_rnd+full_chroma_int,format=yuv420p,setsar=1',
               '-c:v', 'libx264', '-preset', 'slow', '-crf', a.crf, '-profile:v', 'high', '-level', '4.2', '-g', '48', '-x264-params', 'aq-mode=3:deblock=-1,-1:psy-rd=1.0,0.15',
               '-color_range', 'tv', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-c:a', 'aac', '-b:a', '320k', '-c:s', 'mov_text', '-metadata:s:s:0', 'language=eng', '-disposition:s:0', '0', '-t', '114', '-movflags', '+faststart', a.out]
        rc = subprocess.run(cmd).returncode; print('rc', rc, a.out, '%.1f MB' % (os.path.getsize(a.out) / 1e6))
