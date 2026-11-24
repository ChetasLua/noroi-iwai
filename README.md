# 呪い / 祝い (Noroi / Iwai)

A 114-second anime fan film set in the world of *Jujutsu Kaisen*, drawn, animated, scored and sung entirely in code.
1920×1080, 24 fps, 2,736 frames, 37 shots, 160 BPM.

No image models, no image files, no samples, no text-to-speech, no fonts on screen: the picture is Python (numpy, opencv, scipy, pycairo, numba, Pillow) and the sound is zero-dependency JavaScript for Node.

Unofficial fan work. *Jujutsu Kaisen* and its characters are by Gege Akutami; this project is not made or endorsed by the author or the rights holders.

## Build

pycairo has no macOS wheel, so the environment comes from conda-forge ([micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)). Run from the repository root. Sound needs Node 24 or newer.

```bash
micromamba create -y -p .mm/env -c conda-forge python=3.12 numpy opencv pycairo scipy pillow numba ffmpeg pysoundfile
```

Picture: one shot per module in `shots/`; plates are cached on the first run, so it is slow the first time.

```bash
./.mm/env/bin/python tools/film.py --jobs 6
```

Events for the sound department:

```bash
./.mm/env/bin/python tools/events.py
```

Voice (two engines: the film uses the first engine's choir), band, effects and mix:

```bash
node audio/voice.mjs --all 4
```

```bash
node audio/voice_v1/voice.mjs --all 1
```

```bash
mkdir -p out/stems_old_voice && cp out/stems_v1_rebuild/vox_choir.wav out/stems_old_voice/
```

```bash
node audio/band.mjs all && node audio/mix.mjs
```

Join picture and sound into `dist/noroi-iwai.mp4`:

```bash
./.mm/env/bin/python tools/film.py --assemble
```

One frame, with 1:1 crops:

```bash
./.mm/env/bin/python tools/still.py d4 54 --crop 900,400
```

## What is where

| Path | What it does |
|---|---|
| `film.py` | the timeline: every shot, its start and length |
| `jj/head3.py` | 3-D head construction: a face drawn once in front view, correct at any angle |
| `jj/hands.py` `hands_sleeve.py` | hands as anatomy, with two-hand signs |
| `jj/figure.py` `fig_*` | full figures from a 3-D skeleton: runs, punches, smears, cloth |
| `jj/kit.py` `ink.py` `sumi.py` `manga.py` | cel ink, dry brush, hatching, screentone, manga panels |
| `jj/fx*.py` | flat-shape fire, black flash, shatter, orbs, weather |
| `jj/world*.py` `grass.py` `clouds.py` | Shibuya, ruins, subway, hill, beach, shrine, skies |
| `chars/` | the characters, one module each |
| `shots/` | the 37 shots, one module each |
| `audio/` | the song: score, a glottal-pulse voice with a 44-tube vocal tract, a synthesized band, effects and the mix |
| `tools/` | stills, shot renders, film assembly, event collection |
