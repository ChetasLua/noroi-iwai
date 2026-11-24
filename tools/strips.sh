#!/bin/zsh
# tools/strips.sh <film.mp4>  -> qa/crit/<shot>_m.jpg : every 2nd frame (6 x 6) of each shot at 400 px, for judging motion.  ffmpeg only, one at a time.
cd "$(dirname "$0")/.."
python3 - <<'PY' | while read s n; do
import sys; sys.path.insert(0,'.')
import film
for s,a,n in film.SHOTS: print(s,n)
PY
  step=2; [ $n -gt 72 ] && step=3; [ $n -gt 110 ] && step=4; [ $n -lt 40 ] && step=1
  nice -n 10 .mm/env/bin/ffmpeg -nostdin -y -v error -threads 2 -i out/$s.mkv -vf "select='not(mod(n\,$step))',scale=400:-2,tile=6x6" -frames:v 1 -q:v 4 qa/crit/${s}_m.jpg
done
