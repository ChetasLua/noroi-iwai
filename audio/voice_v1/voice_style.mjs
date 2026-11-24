// voice_style.mjs : how each line is performed.  eff = effort at start / peak / end (0 whisper-close .. 1.2 everything he has)
const B = { rd: .28, breath: 0.26, vib: 40, vibRate: 5.3, accent: .9, wn: 40, zeta: .6, scoop0: -80, rel: .18, onset: .032, cons: .95, breathIn: .9, brAmp: .020, sib: .9, jit: .95, decay: 1.8 };
const C = { rd: .22, tiltK: 1, belt: .5, open: .15, breath: 0.2, vib: 48, vibRate: 5.6, accent: 1.1, wn: 44, zeta: .6, scoop0: -90, rel: .12, onset: .026, breathIn: 1, brAmp: .022, jit: .955, sib: .95 };
const D = { rd: .2, tiltK: 1, breath: 0.12, vib: 62, vibRate: 5.8, accent: 1.4, wn: 47, zeta: .56, scoop0: -150, rel: .10, onset: .02, rasp: .12, belt: 1, fall: 45, breathIn: 1.1, brAmp: .028, jit: .95, sib: 1, open: .32, brLen: .30 };
const E = { rd: .3, breath: 0.5, vib: 34, vibRate: 4.9, accent: .6, wn: 38, zeta: .72, scoop0: -60, rel: .24, onset: .05, cons: .95, breathIn: 1.2, brAmp: .020, jit: .95, tremor: .5, sib: .8, decay: 2.2 };
const H = { rd: .1, tiltK: 1, breath: 0.14, vib: 72, vibRate: 6.0, accent: 1.6, wn: 52, zeta: .58, scoop0: -180, rel: .09, onset: .018, rasp: .22, belt: 1, fall: 70, breathIn: 1.2, brAmp: .034, jit: .95, sib: 1, open: .34, brLen: .28 };
const G = { rd: .26, breath: 0.28, vib: 44, vibRate: 5.2, accent: .7, wn: 42, zeta: .68, scoop0: -70, rel: .22, onset: .035, cons: .95, breathIn: 1, brAmp: .018, sib: .85, jit: .95 };
export const STYLE = {
  b1: { ...B, eff: [.30, .40, .28] }, b2: { ...B, eff: [.36, .46, .34] }, b3: { ...B, eff: [.40, .46, .40], accent: 1 }, b4: { ...B, eff: [.46, .56, .44], }, b5: { ...B, eff: [.46, .56, .44] }, b6: { ...B, eff: [.48, .54, .36], },
  c1: { ...C, eff: [.50, .60, .56] }, c2: { ...C, eff: [.55, .66, .58] }, c3: { ...C, eff: [.62, .78, .86], peak: .7 }, c4: { ...C, eff: [.80, .92, 1.02], peak: .4, rasp: .05 },
  d1: { ...D, eff: [.88, .98, .90] }, d2: { ...D, eff: [.86, .96, .88] }, d3: { ...D, eff: [.86, .98, .92] }, d4: { ...D, eff: [.92, 1.0, 1.10], peak: .6 }, d5: { ...D, eff: [.86, .98, 1.06], peak: .6 }, d6: { ...D, eff: [.95, .86, .60], peak: .4, rasp: .04 },
  e1: { ...E, eff: [.26, .34, .20] }, e2: { ...E, eff: [.26, .38, .20] }, e3: { ...E, eff: [.28, .40, .22] }, e4: { ...E, eff: [.28, .42, .22] }, e5: { ...E, eff: [.42, .62, .24], peak: .25, vib: 44, tremor: .9 },
  h1: { ...H, eff: [1.0, 1.08, 1.0] }, h2: { ...H, eff: [.98, 1.08, 1.0] }, h3: { ...H, eff: [1.0, 1.08, 1.16], peak: .6 }, h4: { ...H, eff: [1.1, 1.16, .92], peak: .35 },
  g2: { ...G, eff: [.36, .46, .30] }, g3: { ...G, eff: [.44, .54, .46] }, g4: { ...G, eff: [.44, .56, .30], peak: .5 },
  f1: { eff: [.8, .8, .8] }, f2: { eff: [.8, .9, .9] }, f3: { eff: [1, 1, 1] }, f4: { eff: [.9, .9, .9] }, f5: { eff: [.4, .4, .3], breath: 1 },
};
// per-note acting: [line][index] -> extras
export const ACT = {
  b1: { 11: { vib: 28, fall: 60 } }, b6: { 10: { glideAt: .45, vib: 31 } },
  c3: { 4: { cresc: .10 }, 8: { cresc: .18, vib: 46 } }, c4: { 2: { cresc: .28, vib: 62, vibOn: .45, rasp: 0.09 } },
  d1: { 2: { scoop: -120 }, 10: { vib: 53 } }, d2: { 11: { vib: 53 } }, d4: { 10: { scoop: -220, scoopT: .09, cresc: .10, vib: 62, rasp: 0.16 } }, d5: { 10: { scoop: -180, scoopT: .08, cresc: .08, vib: 62, rasp: 0.14 } }, d6: { 11: { fall: 180, fallT: .25, vib: 31 } },
  e5: { 2: { vib: 46, vibOn: .5, cresc: .06 }, 7: { vib: 34, vibOn: .75, flip: [[.10, .19, 430, .7], [1.18, 1.27, 260, .6]], glideAt: .30, rel: .45, fall: 160, fallT: .4 } },
  h1: { 2: { scoop: -150 }, 11: { vib: 62, rasp: 0.18 } }, h3: { 2: { scoop: -150 }, 10: { scoop: -240, scoopT: .09, cresc: .12, vib: 71, rasp: 0.27, raspT: .35 } }, h4: { 3: { vib: 71, cresc: .05, rasp: 0.27, raspT: .3 }, 8: { vib: 53, fall: 120 } },
  g4: { 2: { vib: 43, vibOn: .7, rel: .5 } },
};
// twelve men: different throats, ages and tempers
export const CHOIR = [
  { Lt: 18.8, oct: -12, pan: -.15, cents: -8, lag: 0.006, rasp: 0.07, vib: 12, jit: 1.2, eff: 1, gain: 1.15, st: { rd: 0.35, breath: .4 } },
  { Lt: 19.4, oct: -12, pan: .30, cents: 12, lag: 0.024, rasp: 0.20, vib: 8, jit: 1.7, eff: 1.05, gain: 1.1, st: { rd: 0.25, breath: .5 } },
  { Lt: 18.2, oct: -12, pan: -.55, cents: -19, lag: 0.033, rasp: 0.05, vib: 16, jit: 1.1, eff: .95, gain: 1.0, st: { rd: 0.55, breath: .5 } },
  { Lt: 17.9, oct: -12, pan: .65, cents: 5, lag: -0.006, rasp: 0.14, vib: 10, jit: 1.4, eff: 1, gain: 1.0, st: { rd: 0.45, breath: .4 } },
  { Lt: 18.5, oct: -12, pan: 0, cents: 16, lag: 0.015, rasp: 0.32, vib: 0, jit: 2.1, eff: 1.1, gain: .95, st: { rd: 0.20, breath: .7 } },
  { Lt: 17.4, oct: 0, pan: -.35, cents: -11, lag: 0.012, rasp: 0.02, vib: 20, jit: 0.9, eff: .95, gain: .62, st: { rd: 0.55, breath: .5 } },
  { Lt: 17.0, oct: 0, pan: .45, cents: 14, lag: 0.027, rasp: 0.05, vib: 18, jit: 1.0, eff: .9, gain: .60, st: { rd: 0.65, breath: .7 } },
  { Lt: 17.8, oct: 0, pan: -.80, cents: 4, lag: -0.009, rasp: 0.11, vib: 14, jit: 1.2, eff: 1, gain: .58, shout: 1, st: { rd: 0.45, breath: .4 } },
  { Lt: 16.7, oct: 0, pan: .85, cents: -16, lag: 0.039, rasp: 0.07, vib: 22, jit: 1.0, eff: .95, gain: .55, shout: 1, st: { rd: 0.60, breath: .6 } },
  { Lt: 17.2, oct: 0, pan: .12, cents: 8, lag: 0.018, rasp: 0.16, vib: 10, jit: 1.4, eff: 1.05, gain: .55, shout: 1, st: { rd: 0.40, breath: .5 } },
  { Lt: 19.0, oct: -12, pan: -.70, cents: 22, lag: 0.045, rasp: 0.09, vib: 6, jit: 1.4, eff: .85, gain: .8, st: { rd: 0.75, breath: .9 } },
  { Lt: 18.0, oct: -12, pan: .78, cents: -24, lag: -0.015, rasp: 0.25, vib: 8, jit: 1.8, eff: 1, gain: .8, st: { rd: 0.50, breath: .6 } },
];
