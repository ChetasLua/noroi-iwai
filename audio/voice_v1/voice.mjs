// voice_v1 : the first engine, kept because the film uses its choir in F (out/stems_old_voice/vox_choir.wav).  node audio/voice_v1/voice.mjs --all 1 -> out/stems_v1_rebuild
// voice.mjs : the singer.  node audio/voice.mjs            -> every stem into out/stems + out/vocal_dry.wav (4 threads)
//             node audio/voice.mjs --lead out.wav [ids]    -> dry lead only (mono, 16 bit), optional subset of lines
import fs from 'node:fs';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { SR, writeWav, db, todb, rng, midi, clamp, smooth, Biquad } from './voice_dsp.mjs';
import { synth, SPF } from './voice_core.mjs';
import { Tract } from './voice_tract.mjs';
import { buildLine, E8 } from './voice_perf.mjs';
import { STYLE, ACT, CHOIR } from './voice_style.mjs';
import * as fx from './voice_fx.mjs';
export const SCORE = JSON.parse(fs.readFileSync(new URL('../score.json', import.meta.url)));
export const TOTAL = 114 * SR, LT = 17.6;
export function lineNotes(line) { const act = ACT[line.id] || {}; return line.notes.map((n, i) => ({ ...n, x: { ...(act[i] || {}) } })); }
export function renderLine(line, o = {}) {
  const st = { ...STYLE[line.id], ...(o.style || {}) }, b = buildLine(o.notes || lineNotes(line), st, o), r = synth(b.s, { tract: o.tract, seed: (o.seed ?? 1) * 1009 + line.bar, hfj: .3, hfs: .5, ...(o.synth || {}) });
  return { y: r.y, start: Math.round(b.t0 * SR) - r.pad, b };
}
export function nextStart(l) { const e = l.notes[l.notes.length - 1], te = (e.t + e.d) * E8; let m; for (const q of SCORE.lines) { const t = q.notes[0].t * E8; if (q.voice === l.voice && t > l.notes[0].t * E8 && t < te + .5 && (m === undefined || t < m)) m = t; } return m; }
export function addTo(buf, r, g = 1) { for (let i = 0; i < r.y.length; i++) { const j = r.start + i; if (j >= 0 && j < buf.length) buf[j] += r.y[i] * g; } }
export function renderLead(ids, o = {}) { const buf = new Float32Array(TOTAL), tract = o.tract || new Tract(o.Lt ?? LT); for (const l of SCORE.lines) { if (l.voice !== 'lead' || (ids && !ids.includes(l.id))) continue; const fpo = o.Lt && o.Lt !== LT ? { fp: 40 + Math.round(o.Lt * 100) % 31, fpd: .35 } : {}, so = o.styleAdd ? Object.fromEntries(Object.entries(o.styleAdd).map(([k, v]) => [k, (STYLE[l.id][k] ?? 0) + v])) : {}; addTo(buf, renderLine(l, { ...o, style: { ...(o.style || {}), ...so }, synth: { ...fpo, ...(o.synth || {}) }, tract, seed: (o.seed ?? 1), nextT: nextStart(l), notes: o.notesFor ? o.notesFor(l) : undefined })); } return buf; }

// ---------- harmony ----------
const PC = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
export function chordPcs(c) { if (c === 'N') return []; const m = /^([A-G])([#b]?)([^/]*)/.exec(c), r = (PC[m[1]] + (m[2] === '#' ? 1 : m[2] === 'b' ? -1 : 0) + 12) % 12, q = m[3]; const iv = /^m(?!aj)/.test(q) ? [0, 3, 7] : /sus4/.test(q) ? [0, 5, 7] : q === '5' ? [0, 7] : [0, 4, 7]; if (/maj7/.test(q)) iv.push(11); else if (/7/.test(q)) iv.push(10); return iv.map(x => (r + x) % 12); }
// diatonic line that keeps thirds to sixths away from the tune, leans on chord tones, moves smoothly.  dir -1 below, +1 above
export function harmony(line, dir, lo = 3, hi = 9) {
  const sharp = line.bar >= 58, scale = new Set(sharp ? [7, 9, 11, 0, 2, 4, 6] : [5, 7, 9, 10, 0, 2, 4]), ns = line.notes, S = [];
  ns.forEach((n, k) => { const m = midi(n.pitch.split('>')[0]), ch = chordPcs(SCORE.chords[Math.min(75, Math.floor(n.t / 8))]), strong = n.t % 4 === 0 || n.d >= 2, row = []; for (let iv = lo; iv <= hi; iv++) { const h = m + dir * iv, pc = ((h % 12) + 12) % 12; let c = (iv === 3 || iv === 4) ? 0 : (iv === 8 || iv === 9) ? .8 : iv === 5 ? .9 : iv === 7 ? 1.4 : 3; if (!scale.has(pc) && !ch.includes(pc)) c += 6; if (!ch.includes(pc)) c += strong ? 2.2 : .5; let best = 0, from = -1; if (k) { best = 1e9; S[k - 1].forEach((p, j) => { const d = Math.abs(p.h - h), v = p.cost + .28 * d + (d > 5 ? 1 : 0); if (v < best) { best = v; from = j; } }); } row.push({ h, cost: c + best, from }); } S.push(row); });
  let j = S[S.length - 1].reduce((a, p, i, r) => p.cost < r[a].cost ? i : a, 0); const out = new Array(ns.length); for (let k = ns.length - 1; k >= 0; k--) { out[k] = S[k][j].h; j = S[k][j].from; }
  return ns.map((n, k) => ({ ...n, pitch: n.pitch.split('>')[0], h: out[k] - midi(n.pitch.split('>')[0]), x: { ...((ACT[line.id] || {})[k] || {}), scoop: undefined, flip: undefined, fry: undefined, rasp: undefined, glideAt: undefined } }));
}

// ---------- tasks (one per thread job) ----------
const has = (l, ids) => ids.includes(l.id);
const DLINES = ['c3', 'c4', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6'], HLINES = ['h1', 'h2', 'h3', 'h4'];
export const TASKS = {
  lead: () => renderLead(null, { seed: 1 }),
  dblL: () => renderLead([...DLINES, ...HLINES, 'g3', 'b4', 'b6'], { seed: 23, Lt: LT * 1.018, hum: .011, lag: .006, detune: -7, wander: 5, styleAdd: { rd: .18, breathIn: -2, rasp: .03 } }),
  dblR: () => renderLead([...DLINES, ...HLINES, 'g3', 'b4', 'b6'], { seed: 47, Lt: LT * .984, hum: .012, lag: -.003, detune: 8, wander: 5, styleAdd: { rd: .22, breathIn: -2, rasp: .03 } }),
  oct: () => renderLead(['c4', ...DLINES.slice(2), ...HLINES], { seed: 61, Lt: LT * 1.05, shift: -12, hum: .010, detune: 3, wander: 4, styleAdd: { rd: -.1, breathIn: -2, vib: -30 }, style: { rasp: .1 } }),
  har1: () => renderLead([...DLINES.slice(2), ...HLINES, 'g3', 'g4'], { seed: 83, Lt: LT * 1.01, hum: .010, detune: -4, wander: 5, styleAdd: { rd: .2, breathIn: -2, rasp: -.05 }, notesFor: l => harmony(l, -1) }),
  har2: () => renderLead([...HLINES, 'd4', 'd5', 'g3', 'g4'], { seed: 97, Lt: LT * .985, hum: .010, detune: 5, wander: 5, styleAdd: { rd: .5, breathIn: -2, rasp: -.1, breath: .4 }, notesFor: l => has(l, ['g3', 'g4']) ? harmony(l, -1, 5, 9).map((n, k) => n) : harmony(l, 1, 3, 5) }),
  harg: () => { // hands join: a second voice enters on "te wo tsunagu"
    const l = SCORE.lines.find(q => q.id === 'g2'), k0 = l.notes.findIndex(n => n.mora === 'te'), h = harmony(l, 1, 3, 5), buf = new Float32Array(TOTAL); addTo(buf, renderLine(l, { tract: new Tract(LT * .99), seed: 71, notes: h.filter((n, k) => k >= 5 && n.t >= 550), style: { rd: .5, breathIn: .5, eff: [.34, .4, .3] } })); return buf; },
  whisper: () => { const buf = new Float32Array(TOTAL), tract = new Tract(17.2, { r: 900, c1: 9 }); const L = SCORE.lines.find(q => q.voice === 'whisper'); const W = { whisper: true, tract, hum: .004 };
    const a2 = [{ mora: 'no', pitch: 'D3', t: 21.4, d: .9, x: { acc: 0, vib: 0 } }, { mora: 'ro', pitch: 'D3', t: 22.4, d: .8, x: { acc: 0 } }, { mora: 'i', pitch: 'D3', t: 23.3, d: 1.6, x: { acc: 0, vib: 0, rel: .3 } }];
    const st = { ...STYLE.f5, breathIn: .35, brAmp: .02, sib: 1.0, cons: 1.2 };
    addTo(buf, renderLine({ id: 'f5', bar: 3 }, { ...W, seed: 5, notes: a2, style: st })); const f5 = L.notes.map((n, i) => ({ ...n, d: i === 2 ? 4 : 1.2, t: n.t + i * .35, x: { acc: i === 2 ? .6 : 0, vib: 0, rel: i === 2 ? .45 : .1 } })); addTo(buf, renderLine(L, { ...W, seed: 8, notes: f5, style: st }), 1.15); return buf; },
};
for (let v = 0; v < 12; v++) TASKS['choir' + v] = () => {
  const c = CHOIR[v], buf = new Float32Array(TOTAL), tract = new Tract(c.Lt, { piri: c.piri }); for (const l of SCORE.lines) { if (l.voice !== 'choir') continue; const up = l.id === 'f3', sh = c.oct + (up ? -12 : 0) + (up && c.shout ? 12 : 0); const notes = l.notes.map((n, i) => ({ ...n, x: { acc: up ? .5 + .22 * i * (.75 + .5 * ((v * 7 + i * 3) % 5) / 4) : (n.t % 4 === 0 ? .8 : .2), eff: up ? -.18 + .045 * i : 0, vib: c.vib, scoop: up ? -180 : -60, rasp: c.rasp, raspT: up ? .2 : 9, fall: i === l.notes.length - 1 && !up ? 60 : undefined, rel: .14 } }));
    addTo(buf, renderLine(l, { tract, seed: 200 + v * 13, shift: sh, detune: c.cents, wander: 7, lag: c.lag, hum: .020, notes, nextT: nextStart(l), style: { ...STYLE[l.id], ...c.st, eff: STYLE[l.id].eff.map(e => e * c.eff), breathIn: 0, jit: c.jit, tau: .026, wn: 34, zeta: .75, drift: 1.8, fricDb: -8, belt: .4, tiltK: .5, air: .02 }, synth: { live: 1.4, balance: .5, fp: 3 + v * 5, fpd: .6, hfc: 5200, hfn: 4 } }), c.gain); } return buf; };

export function foldChoir(R, v, y) { const N = TOTAL; if (!R.cl) { R.cl = [new Float32Array(N), new Float32Array(N)]; R.cm = new Float32Array(N); } const p = (CHOIR[v].pan + 1) / 2, d = Math.round(Math.abs(CHOIR[v].pan) * .0006 * SR), gl = Math.cos(p * Math.PI / 2), gr = Math.sin(p * Math.PI / 2), dl = CHOIR[v].pan > 0 ? d : 0, dr = CHOIR[v].pan < 0 ? d : 0; for (let i = 0; i < N; i++) { const s = y[i]; if (!s) continue; R.cl[0][Math.min(N - 1, i + dl)] += s * gl; R.cl[1][Math.min(N - 1, i + dr)] += s * gr; R.cm[i] += s * .5; } }
// ---------- mix of the stems ----------
function norm(x, target) { let e = 0, c = 0; for (const v of x) if (Math.abs(v) > 2e-3) { e += v * v; c++; } const g = c ? target / Math.sqrt(e / c) : 1; return x.map(v => v * g); }
const secAt = (t, tab) => { let i = 0; while (i < tab.length - 1 && tab[i + 1][0] <= t) i++; const a = tab[i], p = tab[i - 1]; if (!p) return a[1]; const u = clamp((t - a[0]) / .5, 0, 1); return p[1] + (a[1] - p[1]) * smooth(u); };
const ride = (x, tab) => { const y = new Float32Array(x.length); let g = 0; for (let i = 0; i < x.length; i++) { if ((i & 63) === 0) g = secAt(i / SR, tab); y[i] = x[i] * g; } return y; };
const T = b => b * 1.5;
export function mixStems(R) {
  const S = {}, N = TOTAL;
  // lead
  let x = Float32Array.from(R.lead); let pk = 0; for (const v of x) pk = Math.max(pk, Math.abs(v)); const dry = x.map(v => v * .7 / pk);
  x = norm(x, .12);
  x = ride(x, [[0, db(5)], [T(16), db(4)], [T(20), db(2)], [T(24), db(0)], [T(35.2), db(3)], [T(36), db(7)], [T(44), db(3.5)], [T(46), db(6)], [T(58), db(-1)], [T(66), db(5.5)], [T(71), db(4.5)]]);
  x = fx.eq(x, [['hp', 95, .7, 0], ['ls', 260, .7, 1.5], ['pk', 2900, 1.2, -1.2], ['hs', 8500, .7, 2]]);
  x = fx.comp(x, { thr: -26, ratio: 2.6, atk: .010, rel: .14, makeup: 3 }); x = fx.comp(x, { thr: -16, ratio: 3.5, atk: .003, rel: .08, makeup: 1.5 });
  const grit = ride(fx.warm(x, 3.2, 1, .2), [[0, 0], [T(16), .08], [T(24), .25], [T(36), 0], [T(44), .1], [T(46), 0], [T(58), .4], [T(66), 0]]); x = fx.warm(x, 1.7, .35); fx.add(x, grit, .5); x = fx.deess(x, 6000, -33, 3);
  const pl = fx.plate(ride(x, [[0, .10], [T(16), .14], [T(22), .18], [T(24), .15], [T(36), .24], [T(44), .30], [T(58), .17], [T(66), .20], [T(73), .28]]), { decay: .66, pre: .030, damp: 2900, hicut: 4600, lowcut: 240 }), rm = fx.room(x, 5, { n: 34, lp: 3000, len: .038 }), ec = fx.echo(ride(x, [[0, 0], [T(16), .10], [T(22), .2], [T(24), .13], [T(36), .05], [T(44), .22], [T(58), .12], [T(66), .06], [T(73), .16]]), .28125, .42), ec2 = fx.echo(ride(x, [[0, 0], [T(22), .12], [T(24), .09], [T(36), .05], [T(44), .16], [T(58), .09], [T(66), .05]]), .375, .42);
  delete R.lead; S.vox_lead = [0, 1].map(c => { const o = new Float32Array(N); for (let i = 0; i < N; i++) o[i] = x[i] + .14 * rm[c][i] + .85 * pl[c][i] + (c ? ec2[i] : ec[i]); return o; });
  // doubles
  const dbl = k => { let y = norm(R[k], .10); y = fx.eq(y, [['hp', 190, .7, 0], ['pk', 2800, 1, -2], ['hs', 7000, .7, -2]]); y = fx.comp(y, { thr: -28, ratio: 4, atk: .006, rel: .1, makeup: 5 }); return fx.deess(y, 5600, -36, 5); }, dl = dbl('dblL'), dr = dbl('dblR'), dp = fx.plate(dl.map((v, i) => (v + dr[i]) * .5), { decay: .6, pre: .018, damp: 2800, hicut: 4200 });
  delete R.dblL; delete R.dblR; S.vox_double = [0, 1].map(c => { const o = new Float32Array(N); for (let i = 0; i < N; i++) o[i] = .42 * ((c ? .25 : .95) * dl[i] + (c ? .95 : .25) * dr[i]) + .22 * dp[c][i]; return o; });
  // harmonies
  const hm = (k, g, hp = 170) => { let y = norm(R[k], .10); y = fx.eq(y, [['hp', hp, .7, 0], ['pk', 320, .9, -2.5], ['pk', 2900, 1, -2.5], ['hs', 6500, .7, -3]]); y = fx.comp(y, { thr: -28, ratio: 4, atk: .006, rel: .12, makeup: 5 }); return fx.deess(y, 5600, -38, 5).map(v => v * g); }, h1 = hm('har1', .55), h2 = hm('har2', .42), oc = hm('oct', .5, 115), hg = hm('harg', .4);
  const hb = new Float32Array(N); for (let i = 0; i < N; i++) hb[i] = h1[i] + h2[i] + .6 * oc[i] + hg[i]; const hp = fx.plate(hb, { decay: .68, pre: .03, damp: 2800, hicut: 4200 });
  for (const k of ['har1', 'har2', 'oct', 'harg']) delete R[k]; S.vox_harm = [0, 1].map(c => { const o = new Float32Array(N); for (let i = 0; i < N; i++) o[i] = (c ? .35 : .9) * h1[i] + (c ? .9 : .35) * h2[i] + .7 * oc[i] + (c ? .8 : .5) * hg[i] + .28 * hp[c][i]; return o; });
  // choir: twelve men in a dead stone room
  const { cl, cm } = R;
  let ce = 0, cc = 0; for (let i = 0; i < N; i++) { const v = cl[0][i] + cl[1][i]; if (Math.abs(v) > .02) { ce += v * v; cc++; } } const cg = .14 / Math.sqrt(ce / Math.max(1, cc)); for (const c of cl) for (let i = 0; i < N; i++) c[i] *= cg; for (let i = 0; i < N; i++) cm[i] *= cg;
  const cp = fx.plate(cm, { decay: .80, pre: .032, damp: 1900, lowcut: 110, hicut: 2600, size: 1.25 }), cr = fx.room(cm, 31, { len: .10, n: 44, lp: 2400 });
  S.vox_choir = [0, 1].map(c => { let o = fx.eq(cl[c], [['hp', 50, .7, 0], ['ls', 160, .7, 2], ['pk', 420, .9, 1.5], ['hs', 5200, .7, -3]]); o = fx.warm(fx.comp(o, { thr: -22, ratio: 3, atk: .012, rel: .16, makeup: 4 }), 2.4, .45); for (let i = 0; i < N; i++) o[i] = .85 * o[i] + .5 * cr[c][i] + .42 * cp[c][i]; return o; });
  // whisper: mouth at the ear, a little wet echo in the dark
  { let y = norm(R.whisper, .09); y = fx.eq(y, [['hp', 140, .7, 0], ['ls', 300, .7, 4], ['pk', 3400, 1, 1.5], ['hs', 9000, .7, 1]]); y = fx.comp(y, { thr: -30, ratio: 4, atk: .004, rel: .1, makeup: 8 }); y = fx.deess(y, 6500, -34, 3); const wp = fx.plate(y, { decay: .8, pre: .06, damp: 3600, lowcut: 300 }), e1 = fx.echo(y, .19, .5, { lp: 2600, hp: 500, taps: 3 }), e2 = fx.echo(y, .31, .5, { lp: 2400, hp: 500, taps: 3 }); S.vox_whisper = [0, 1].map(c => { const o = new Float32Array(N); for (let i = 0; i < N; i++) { const t = i / SR, far = t < 10 ? (c ? .55 : 1) : (c ? 1 : .6); o[i] = y[i] * far + .05 * wp[c][i] + .05 * (c ? e2[i] : e1[i]); } return o; }); }
  return { S, dry };
}

if (!isMainThread) { parentPort.on('message', k => { if (k === null) process.exit(0); const y = TASKS[k](); parentPort.postMessage({ k, y }, [y.buffer]); }); }
else if (process.argv[2] === '--lead') {
  const ids = process.argv[4] ? process.argv[4].split(',') : null, t = Date.now(), y = renderLead(ids); let pk = 0; for (const v of y) pk = Math.max(pk, Math.abs(v)); const g = .8 / pk; console.error('peak', pk.toFixed(3), 'ms', Date.now() - t);
  let a = 0, b = TOTAL; if (ids) { const L = SCORE.lines.filter(l => ids.includes(l.id)); a = Math.max(0, Math.round((Math.min(...L.map(l => l.bar)) * 1.5 - .8) * SR)); b = Math.min(TOTAL, Math.round((Math.max(...L.map(l => l.bar + l.beats8 / 8)) * 1.5 + .6) * SR)); }
  writeWav(process.argv[3], [y.subarray(a, b).map(v => v * g)], { float: false });
}
else if (process.argv[1] && import.meta.url.endsWith(process.argv[1].split('/').pop())) {
  const t0 = Date.now(), keys = Object.keys(TASKS), R = {}, jobs = +(process.argv[3] || 1); let next = 0, done = 0;
  await new Promise((res, rej) => { for (let w = 0; w < jobs; w++) { const W = new Worker(new URL(import.meta.url)); W.on('error', rej); W.on('message', m => { if (m.k.startsWith('choir')) foldChoir(R, +m.k.slice(5), m.y); else R[m.k] = m.y; done++; if (next < keys.length) W.postMessage(keys[next++]); else W.postMessage(null); if (done === keys.length) res(); }); W.postMessage(keys[next++]); } });
  console.log('voices', ((Date.now() - t0) / 1000).toFixed(1), 's');
  const { S, dry } = mixStems(R), dir = new URL('../../out/stems_v1_rebuild/', import.meta.url).pathname; fs.mkdirSync(dir, { recursive: true });
  for (const k in S) { let pk = 0; for (const c of S[k]) for (const v of c) pk = Math.max(pk, Math.abs(v)); fx.limit(S[k], .95); writeWav(dir + k + '.wav', S[k]); console.log(k, 'peak before limit', todb(pk).toFixed(1), 'dB'); }
  writeWav(new URL('../../out/stems_v1_rebuild/vocal_dry.wav', import.meta.url).pathname, [dry], { float: false }); console.log('total', ((Date.now() - t0) / 1000).toFixed(1), 's');
}
