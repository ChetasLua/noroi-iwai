// voice_fit.mjs : shapes area functions until their resonances land on the singer's formant targets.  node audio/voice_fit.mjs -> audio/voice_vowels.mjs
import fs from 'node:fs';
import { NSEC, formants } from './voice_tract.mjs';
const LT = 16.0;
// starting tube shapes (glottis -> lips, cm2); any length, resampled to 44
const BASE = {
  i: [.33,.30,.36,.34,.68,.50,2.43,3.15,2.66,2.49,3.39,3.80,3.78,4.35,4.50,4.43,4.68,4.52,4.15,4.09,3.51,2.95,2.03,1.66,1.38,1.05,.60,.35,.32,.22,.20,.22,.25,.28,.38,.40,.50,.75,1.4,1.8,1.9,1.6],
  a: [.45,.30,.32,.30,.35,.40,.53,1.05,1.12,.95,.83,.69,.56,.48,.43,.42,.42,.48,.55,.75,1.2,1.35,1.72,2.09,2.56,2.78,2.96,3.32,3.75,4.4,4.99,5.5,5.9,6.0,5.9,5.6,5.1,4.6,4.1,4.1,4.2,4.3,4.5,4.7],
  u: [.40,.38,.35,.43,.60,1.6,2.6,2.7,2.5,2.6,3.2,4.2,4.4,4.2,3.9,3.6,3.2,2.8,2.5,2.2,2.0,1.8,1.5,1.2,.95,.75,.55,.45,.45,.50,.60,.75,.95,1.2,1.5,1.9,2.3,2.5,2.6,2.4,2.0,1.5,1.1,.9],
  e: [.30,.25,.26,.26,.35,.78,1.25,1.24,1.1,1.0,1.1,1.4,1.77,1.97,2.46,2.70,2.92,3.03,2.84,2.84,2.83,2.36,2.14,2.00,1.78,1.6,1.4,1.25,1.15,1.15,1.2,1.4,1.7,2.0,2.2,2.2,2.3,2.5,2.2,1.9],
  o: [.28,.25,.28,.30,.59,1.4,1.5,1.2,1.0,1.1,1.8,2.1,1.96,1.92,1.70,1.66,1.52,1.28,1.3,1.2,.95,.95,.95,.85,.70,.56,.50,.50,.67,1.06,1.5,2.29,2.99,3.74,4.39,5.0,5.6,5.3,4.2,2.9,1.8,1.1,.75,.6],
};
// F1..F5 per vowel for three efforts: 0 close/soft, 1 open chest, 2 full belt up high
export const T = {
  a: [[740, 1230, 2650, 3480, 4200], [860, 1350, 2700, 3450, 4200], [900, 1400, 2700, 3450, 4200]],
  i: [[322, 2230, 3020, 3560, 4250], [447, 2130, 2920, 3500, 4250], [562, 2050, 2880, 3520, 4250]],
  u: [[350, 1290, 2380, 3400, 4150], [450, 1290, 2420, 3400, 4150], [560, 1320, 2450, 3400, 4150]],
  e: [[489, 1930, 2620, 3500, 4200], [624, 1880, 2620, 3450, 4200], [707, 1840, 2640, 3480, 4250]],
  o: [[470, 840, 2560, 3350, 4150], [590, 960, 2600, 3350, 4150], [680, 1050, 2620, 3350, 4150]],
};
const W = [3, 3, 2, 1, .4], NB = 14, LIPS = { a: [3.8, 4.4, 4.8], i: [1.6, 2.4, 3.2], u: [.75, 1.0, 1.3], e: [2.2, 3.0, 3.8], o: [.65, .95, 1.4] };
const resample = b => Array.from({ length: NSEC }, (_, i) => { const x = i * (b.length - 1) / (NSEC - 1), k = Math.min(b.length - 2, Math.floor(x)), u = x - k; return Math.exp(Math.log(b[k]) * (1 - u) + Math.log(b[k + 1]) * u); });
function shape(base, c) { return base.map((a, i) => { let s = 0; const x = (i + .5) / NSEC; for (let k = 0; k < NB; k++) s += c[k] * Math.cos(Math.PI * k * x); return Math.min(6, Math.max(.2, a * Math.exp(s))); }); }
let KK = 0;
function resid(base, c, tgt, lip) { const A = shape(base, c), f = formants(A, LT, 5, 5600), r = []; for (let i = 0; i < 5; i++) r.push(W[i] * (i > 1 && KK ? .45 : 1) * Math.log((f[i] || tgt[i] * 1.6) / tgt[i]) * 12); for (let k = 0; k < NB; k++) r.push(c[k] * (k < 1 ? .35 : .10 + .035 * k)); r.push(.9 * Math.log(A[NSEC - 1] / lip), .6 * Math.log(A[NSEC - 2] / lip), .3 * Math.log(A[NSEC - 3] / lip)); return r; }
function solve(M, b) { const n = b.length; for (let i = 0; i < n; i++) { let p = i; for (let j = i + 1; j < n; j++) if (Math.abs(M[j][i]) > Math.abs(M[p][i])) p = j;[M[i], M[p]] = [M[p], M[i]];[b[i], b[p]] = [b[p], b[i]]; for (let j = i + 1; j < n; j++) { const q = M[j][i] / M[i][i]; for (let k = i; k < n; k++) M[j][k] -= q * M[i][k]; b[j] -= q * b[i]; } } for (let i = n - 1; i >= 0; i--) { for (let k = i + 1; k < n; k++) b[i] -= M[i][k] * b[k]; b[i] /= M[i][i]; } return b; }
function fit(base, tgt, c0, lip) {
  let c = c0 ? c0.slice() : new Array(NB).fill(0), lam = .05, r = resid(base, c, tgt, lip), E = r.reduce((s, x) => s + x * x, 0);
  for (let it = 0; it < 60; it++) {
    const J = []; for (let k = 0; k < NB; k++) { const d = c.slice(); d[k] += .01; const q = resid(base, d, tgt, lip); J.push(q.map((x, i) => (x - r[i]) / .01)); }
    const M = [], b = []; for (let i = 0; i < NB; i++) { M.push([]); for (let j = 0; j < NB; j++) { let s = 0; for (let m = 0; m < r.length; m++) s += J[i][m] * J[j][m]; M[i].push(s + (i === j ? lam * (s + .02) : 0)); } let s = 0; for (let m = 0; m < r.length; m++) s -= J[i][m] * r[m]; b.push(s); }
    const dc = solve(M, b), n = c.map((x, i) => x + Math.max(-.4, Math.min(.4, dc[i]))), q = resid(base, n, tgt, lip), e = q.reduce((s, x) => s + x * x, 0);
    if (e < E) { if (E - e < 1e-7) break; c = n; r = q; E = e; lam = Math.max(1e-4, lam * .5); } else { lam *= 3; if (lam > 1e5) break; }
  }
  return c;
}
 let txt = '// voice_vowels.mjs : written by voice_fit.mjs.  44 areas (cm2, glottis to lips) per vowel and effort (0 soft, 1 chest, 2 belt), tract length 16.0 cm.\nexport const VOWELS = {\n';
for (const v of 'aiueo') { const base = resample(BASE[v]); let c = null; txt += ' ' + v + ': [\n'; for (let k = 0; k < 3; k++) { KK = k; c = fit(base, T[v][k], c, LIPS[v][k]); const A = shape(base, c), f = formants(A, LT, 5, 5600); console.log(v, k, T[v][k].join(' '), '->', f.map(x => x.toFixed(0)).join(' '), ' areas min', Math.min(...A).toFixed(2), 'max', Math.max(...A).toFixed(2), 'lips', A[43].toFixed(2)); txt += '  [' + A.map(x => +x.toFixed(3)).join(',') + '],\n'; } txt += ' ],\n'; }
txt += '};\n'; fs.writeFileSync(new URL('./voice_vowels.mjs', import.meta.url), txt);
