// band.mjs: renders the arrangement (band_arr.mjs) with our own instruments (inst_*.mjs) into stems and a band mix.
//   node audio/band.mjs all            render every stem one after the other, then mix
//   node audio/band.mjs stem drums     one stem     |   node audio/band.mjs mix     |   node audio/band.mjs check  (harmony + level report)
//   --bars 24 36   only play events inside those bars      --out dir
import { spawnSync } from 'node:child_process';
import os from 'node:os';
import fs from 'node:fs';
import path from 'node:path';
import * as C from './inst_core.mjs';
import * as D from './inst_drums.mjs';
import * as P from './inst_pluck.mjs';
import * as B from './inst_bow.mjs';
import * as K from './inst_keys.mjs';
import * as Y from './inst_synth.mjs';
import { build, at } from './band_arr.mjs';
const { SR, N, st, mk, add, addS, pan, gain, gainS, filt, filtS, bqc, db, todb, rng, clamp, F, mtof, TAU } = C;
try { os.setPriority(0, 15); } catch {}
const argv = process.argv.slice(2), opt = k => { const i = argv.indexOf('--' + k); return i < 0 ? null : argv.slice(i + 1); };
const OUT = opt('out') ? opt('out')[0] : path.join(C.ROOT, 'out'), STEMS = path.join(OUT, 'stems');
const bars = opt('bars') ? opt('bars').slice(0, 2).map(Number) : null;
const X = build(); if (bars) for (const k in X) X[k] = X[k].filter(e => e.t >= bars[0] * 1.5 - .7 && e.t < bars[1] * 1.5);
const ev = k => X[k] || [];
const n = N;
const S = t => Math.floor(t * SR);
const toS = (m, p = 0, g = 1) => { const s = st(n); pan(s, m, p, g); return s; };
const send = (dst, src, g) => addS(dst, src, g);
// time varying send level, [[sec, linear]]
const sendA = (dst, src, pts) => { for (let i = 0; i < n; i++) { const g = (i & 63) ? sendA.g : (sendA.g = C.envAt(pts, i / SR)); dst.L[i] += src.L[i] * g; dst.R[i] += src.R[i] * g; } };
const secs = (...a) => { const o = []; for (let i = 0; i < a.length; i++) { const [bar, v] = a[i]; if (i) o.push([bar * 1.5 - .02, a[i - 1][1]]); o.push([bar * 1.5, v]); } return o; };
const fadeGate = (b, t0, t1, floor, inS = .02, outS = .004) => { for (const x of [b.L, b.R]) for (let i = S(t0); i < Math.min(n, S(t1)); i++) { const t = i / SR, u = clamp((t - t0) / inS, 0, 1), w = clamp((t1 - t) / outS, 0, 1); const k = 1 - (1 - floor) * Math.min(u, w); x[i] *= k; } };
// ---------------------------------------------------------------- stems
const R_ = {};
R_.drums = () => {
  const kickB = mk(n), snB = st(n), tomB = st(n), hatB = st(n), cymB = st(n), prB = st(n), orB = st(n), R = rng('kitperf'); const cnt = {};
  const rr = (p, k = 8) => { cnt[p] = (cnt[p] || 0) + 1 + (R.u() < .25 ? 1 : 0); return cnt[p] % k; };
  const L = D.layer; let slow = 0;
  const choke = (b, s) => { const o = { L: Float32Array.from(b.L), R: Float32Array.from(b.R) }; const a = S(s), r = S(.045); for (const x of [o.L, o.R]) for (let i = 0; i < x.length; i++) if (i > a) x[i] *= Math.max(0, 1 - (i - a) / r) ** 2; return o; };
  const E = ev('kit').slice().sort((a, b) => a.t - b.t); const hatEnds = []; // open hats are choked by the next closed hat
  const hatT = E.filter(e => 'HEhOf'.includes(e.p)).map(e => e.t);
  for (const e of E) { const v = clamp(e.v * (1 + .05 * R.g()), .04, 1), l = L(v), f = v / l; slow = slow * .97 + .03 * R.g() * .006; const off = { K: .0015, S: .0022, R: .0016, G: .004, H: .0028, E: .0028, h: .003, O: .003, T1: .003, T2: .003, T3: .003, T4: .003 }[e.p] ?? .0025; const t = Math.max(0, e.t + slow + off * R.g() * .7 + (e.p === 'S' || e.p === 'R' ? .001 : 0)), i = S(t); const c3 = v < .35 ? .25 : v < .7 ? .55 : 1;
    switch (e.p) {
      case 'K': add(kickB, D.kick(l, rr('K'), 'rock'), f * .98, i); break; case 'k': add(kickB, D.kick(l, rr('k'), 'soft'), f * .85, i); break;
      case 'S': pan(snB, D.snare(l, rr('S'), 'center'), -.03, f * .8, i); break; case 'R': pan(snB, D.snare(l, rr('R'), 'rim'), -.03, f * .92, i); break; case 'G': pan(snB, D.snare(l, rr('G'), 'ghost'), -.03, f * .7, i); break; case 'snr': pan(snB, D.snare(l, rr('G'), 'ghost'), .1, f * 1.1, i); break; case 'x': pan(snB, D.snare(l, rr('x'), 'xstick'), -.1, f * .62, i); break;
      case 'bt': pan(snB, D.brushTap(l, rr('bt')), -.05, f * .55, i); break; case 'g': pan(snB, D.brushTap(l, rr('bt')), -.05, f * .45, i); break; case 'sw': pan(snB, D.brushSweep(e.dur, 1, rr('sw', 4)), -.15, v * .4, i); break;
      case 'H': addS(hatB, D.cym('hatc', c3, rr('H')), v / c3 * .46, i); break; case 'E': addS(hatB, choke(D.cym('hath', c3, rr('h', 4)), .5), v / c3 * .52, i); break;
      case 'h': case 'O': { const nx = hatT.find(q => q > e.t + .02); const sm = D.cym(e.p === 'O' ? 'hato' : 'hath', c3, rr(e.p, 4)); addS(hatB, choke(sm, e.choke ?? (nx ? Math.min(1.6, nx - e.t) : 1.2)), v / c3 * .55, i); break; }
      case 'f': addS(hatB, D.cym('foot', c3, rr('f')), v / c3 * .3, i); break;
      case 'C': case 'D': case 'N': case 'P': { const id = { C: 'crashA', D: 'crashB', N: 'china', P: 'splash' }[e.p], sm = D.cym(id, c3, rr(e.p, 4)), pn = { C: -.45, D: .45, N: .6, P: -.2 }[e.p], b = e.choke ? choke(sm, e.choke) : sm, g = v / c3 * { C: .62, D: .6, N: .5, P: .5 }[e.p]; add(cymB.L, b.L, g * (1 - Math.max(0, pn) * .7), i); add(cymB.R, b.R, g * (1 + Math.min(0, pn) * .7), i); break; }
      case 'I': { const b = D.cym('ride', c3, rr('I')); add(cymB.L, b.L, v / c3 * .13, i); add(cymB.R, b.R, v / c3 * .2, i); break; } case 'B': addS(cymB, D.cym('bell', c3, rr('B')), v / c3 * .25, i); break;
      case 'T1': case 'T2': case 'T3': case 'T4': { const k = +e.p[1] - 1; pan(tomB, D.tom(l, rr(e.p), [148, 116, 90, 70][k]), [-.35, -.05, .25, .5][k], f * [.62, .68, .74, .8][k], i); break; }
      case 'c': addS(prB, D.clap(1, rr('c')), v * .62, i); break; case 'c2': { const b = D.clap(1, rr('c')); add(prB.L, b.R, v * .4, i + S(.009)); add(prB.R, b.L, v * .4, i + S(.009)); break; }
      case 's': pan(prB, D.shaker(1, rr('s', 8), R.r(.08, .1)), .32, v * .34, i); break; case 't': pan(prB, D.tamb(1, rr('t', 6)), -.38, v * .24, i); break;
      case 'w': pan(prB, D.wood(1, rr('w'), 880), .3, v * .4, i); break; case 'w2': pan(prB, D.wood(1, rr('w'), 690), .3, v * .4, i); break; case 'rimk': pan(prB, D.wood(1, rr('rk'), 1500), -.3, v * .3, i); break; case 'col': pan(prB, D.wood(1, rr('cl', 8), 1180 + 260 * R.g()), e.pan || 0, v * .16, i); break;
      case 'ka': pan(prB, D.taiko(1, rr('ka'), 'ka'), .2, v * .5, i); break; case 'kane': pan(orB, D.kane(1, rr('kn')), e.pan ?? .25, v * .3, i); break; case 'hyo': pan(orB, D.hyoshigi(1, rr('hy')), -.2, v * .4, i); break;
      case 'chimes': for (let k = 0; k < 16; k++) pan(orB, D.triangle(1, k), -.7 + 1.4 * k / 16, v * .05 * R.r(.5, 1), i + S(k * .045 + .01 * R.u())); break;
      case 'swell': { const b = D.cymSwell(Math.round(e.dur * 100) / 100, 0, 'crashA'); addS(orB, e.dur < 3 ? b : b, v * .5, i); break; }
      case 'gc': pan(orB, D.granCassa(1, rr('gc')), 0, v * .62, i); break; case 'timp': pan(orB, D.timp(l, rr('tp', 4), e.f), .15, f * .42, i); break; case 'tam': addS(orB, D.tamtam(1, 0), v * .3, i); break; case 'tamq': addS(orB, D.tamtam(1, 1), v * .3, i); break;
      case 'anv': pan(orB, D.anvil(1, rr('an', 4), 1), .3, v * .24, i); break; case 'anvL': pan(orB, D.anvil(1, rr('an', 4), .42), -.15, v * .34, i); break; case 'hb': break; default: console.error('piece', e.p); } }
  for (const e of ev('tri')) pan(orB, D.triangle(1, 0), .4, e.v * .16, S(e.t));
  filt(kickB, bqc.hp(38, .8), bqc.pk(72, -2, 1), bqc.pk(300, -4, .9), bqc.pk(3600, 6, .9), bqc.lp(10000, .6)); C.comp(kickB, { thr: -10, ratio: 3, att: .012, rel: .08, makeup: -2 });
  filtS(snB, bqc.pk(205, 3, 1.2), bqc.pk(480, -3, 1), bqc.pk(3200, -1.5, .9), bqc.hs(4800, -1.5)); C.comp(snB, { thr: -14, ratio: 3, att: .011, rel: .09, makeup: 7 });
  filtS(tomB, bqc.pk(95, 2, .9), bqc.pk(450, -4, .9)); filtS(hatB, bqc.hp(380, .7), bqc.pk(3400, -2, 1), bqc.hs(5500, -3), bqc.lp(10500, .6)); filtS(cymB, bqc.hp(260, .7), bqc.pk(3200, -3.5, .8), bqc.hs(5200, -3.5), bqc.lp(10500, .6));
  const kS = toS(kickB), dry = st(n); addS(dry, kS, 1); addS(dry, snB, 1); addS(dry, tomB, 1); addS(dry, hatB, 1); addS(dry, cymB, 1); addS(dry, prB, 1);
  // rooms: dry and close in B, bigger in the rock sections
  const rs = st(n), lvl = secs([0, .5], [4, .12], [16, .3], [24, .5], [28, .18], [30, .5], [36, .3], [48, .6], [58, .62], [66, .12]); const tmp = st(n); addS(tmp, kS, .35); addS(tmp, snB, 1); addS(tmp, tomB, .9); addS(tmp, hatB, .4); addS(tmp, cymB, .45); addS(tmp, prB, .7); sendA(rs, tmp, lvl);
  const rm = C.room(rs, { t60: .75, size: .5, hiMul: .6, er: 1.3 }); C.comp(rm, { thr: -24, ratio: 4, att: .003, rel: .11, makeup: 8 }); filtS(rm, bqc.hp(140, .7), bqc.lp(7500, .7)); addS(dry, rm, .75);
  const ps = st(n); const pl = secs([0, .2], [4, .07], [16, .2], [24, .24], [28, .1], [30, .24], [48, .3], [58, .28], [66, .08]); const tp = st(n); addS(tp, snB, 1); addS(tp, tomB, .7); addS(tp, prB, .5); sendA(ps, tp, pl); addS(dry, C.plate(ps, { t60: 1.1 }), .9);
  // parallel crush for the rock kit
  const par = C.copyS(dry); const pg = secs([0, .2], [4, 0], [24, .35], [36, 0], [48, .35], [66, 0]); filtS(par, bqc.hs(3500, -6), bqc.lp(6500, .7)); C.comp(par, { thr: -30, ratio: 8, att: .004, rel: .07, makeup: 12 }); sendA(dry, par, pg);
  const oh = st(n); send(oh, orB, .5); const ohp = st(n); send(ohp, prB, .05); send(ohp, snB, .04); addS(oh, ohp, 1); addS(dry, orB, 1); addS(dry, C.hall(oh, { t60: 3 }), .8);
  C.comp(dry, { thr: -8, ratio: 2, att: .014, rel: .12, makeup: 0 }); return dry;
};
R_.bass = () => { const p = P.bassAmp(P.bassDI(ev('bass'), 'pk', { n, style: 'pick' }), { drive: .6 }), s = P.bassAmp(P.bassDI(ev('slap'), 'sl', { n, style: 'slap' }), { drive: .22 }), u = P.bassAmp(P.bassDI(ev('ubass'), 'ub', { n, style: 'up' }), { up: true }); filt(s, bqc.pk(2600, 3, 1)); filt(p, bqc.hp(34, .7), bqc.pk(85, 1, .8), bqc.pk(420, -2, .8), bqc.pk(950, 2, .9)); const o = st(n); pan(o, p, 0, .62); pan(o, s, 0, .72); pan(o, u, 0, 1.15); const r = st(n); pan(r, u, .1, .12); pan(r, s, 0, .06); addS(o, C.room(r), .8); C.comp(o, { thr: -14, ratio: 3, att: .01, rel: .1, makeup: 2 }); return o; };
R_.guitars = () => {
  const o = st(n), hl = st(n), pl = st(n);
  const rl = ev('rg'); const L = P.amp(P.guitarDI(rl, 'L', { n }), { type: 'hi', drive: 1.25, cab: 'L9' }), Rr = P.amp(P.guitarDI(rl, 'R', { n }), { type: 'hiB', drive: 1.15, cab: 'R9', cabo: { lp: 4350, mid: -1, notch: 3350, body: 4.5 } }); const eq = x => filt(x, bqc.hp(88, .8), bqc.pk(3200, -2, 1.2)); eq(L); eq(Rr); const gr = st(n); pan(gr, L, -.98, .5); pan(gr, Rr, .98, .5); const qv = rl.filter(e => e.t > 35.9); const Q1 = P.amp(P.guitarDI(qv, 'L3', { n, jitter: .008, pickup: .13 }), { type: 'crunch', drive: 2.6, cab: 'L3' }), Q2 = P.amp(P.guitarDI(qv, 'R3', { n, jitter: .008, pickup: .13 }), { type: 'crunch', drive: 2.4, cab: 'R3' }); for (const x of [Q1, Q2]) filt(x, bqc.hp(105, .7), bqc.pk(3100, -3, 1), bqc.lp(4600, .7)); pan(gr, Q1, -.72, .24); pan(gr, Q2, .72, .24); addS(o, gr, 1); const gm = C.room(gr, { t60: .45, size: .5, er: 1.2, lp: 4200 }); addS(o, gm, .1);
  const r2 = ev('rg2'); const a = P.amp(P.guitarDI(r2, 'L2', { n, pickup: .12 }), { type: 'crunch', drive: 1.6, cab: 'L2' }), b = P.amp(P.guitarDI(r2, 'R2', { n, pickup: .12 }), { type: 'crunch', drive: 1.4, cab: 'R2' }); filt(a, bqc.hp(180, .7)); filt(b, bqc.hp(180, .7)); pan(o, a, -.7, .34); pan(o, b, .7, .34); pan(hl, a, -.7, .05); pan(hl, b, .7, .05);
  // lead: one voice, sustained, with delay and plate
  const le = [...ev('lead'), ...ev('leadq')].map(e => ({ ...e, n: Array.isArray(e.n) ? e.n : [e.n], o: e.fb ? { t60: 30, hi: [4000, 8] } : { t60: 14, hi: [4000, 3] }, vib: e.vib || (e.dur > .3 ? [5.6, 26, .16] : null) })); const di = P.guitarDI(le, 'lead', { n, pickup: .07, drift: .002 }); for (const e of le) if (e.env) for (let i = S(e.t); i < S(e.t + e.dur); i++) di[i] *= Math.pow((i - S(e.t)) / (e.dur * SR), 1.2); const ld = P.amp(di, { type: 'lead', drive: 1, cab: 'ld' }); filt(ld, bqc.hp(220, .7), bqc.pk(1500, 1.5, 1)); const ls = toS(ld, .12, .3); addS(o, ls, 1); addS(o, C.delayFx(ls, { t: .28125, fb: .28, g: .3 }), 1); send(pl, ls, .22);
  // clean funk
  const cl = [...ev('clean'), ...ev('funk').filter(e => e.side < 0)], cr = [...ev('clean'), ...ev('funk').filter(e => e.side > 0)]; const spec = { t60: 2.2, hi: [4500, .5], bright: 9000, nz: .5 }; const A1 = P.amp(P.guitarDI(cl, 'cl', { n, spec, pickup: .18 }), { type: 'clean', cab: 'cl' }), A2 = P.amp(P.guitarDI(cr, 'cr', { n, spec, pickup: .18 }), { type: 'clean', cab: 'cr' }); for (const x of [A1, A2]) { filt(x, bqc.hp(180, .7), bqc.pk(2600, 2, 1)); C.comp(x, { thr: -22, ratio: 4, att: .004, rel: .08, makeup: 8 }); } const cs = st(n); pan(cs, A1, -.75, .5); pan(cs, A2, .75, .5); addS(o, cs, .55); send(pl, cs, .06);
  // nylon
  const ny = P.nylonDI(ev('nylon'), 'n', { n }); const ns = toS(ny, -.42, 1.0); C.comp(ns, { thr: -20, ratio: 2, att: .01, rel: .12, makeup: 2 }); addS(o, ns, 1); send(hl, ns, .1); send(pl, ns, .1);
  addS(o, C.hall(hl, { t60: 2.2 }), .9); addS(o, C.plate(pl), .9); return o;
};
R_.keys = () => {
  const o = st(n), hl = st(n), pl = st(n), rm = st(n);
  const fe = K.piano(ev('felt'), { seed: 'f', n, felt: 1 }); addS(o, fe, 2.6); send(hl, fe, .34); send(rm, fe, .5); const fg = K.piano(ev('feltG'), { seed: 'g', n, felt: 1 }); addS(o, fg, 2.8); send(hl, fg, .9);
  const jz = K.piano(ev('jazzp'), { seed: 'j', n, felt: 0 }); filtS(jz, bqc.hs(3000, -3)); addS(o, jz, 1.7); send(rm, jz, .7); send(pl, jz, .12);
  const gp = K.piano([...ev('piano')], { seed: 'p', n, felt: 0 }); filtS(gp, bqc.pk(220, -2, .8), bqc.hs(1800, 3)); addS(o, gp, 1.9); sendA(hl, gp, secs([0, .25], [30, .18], [36, .75], [46, .5], [58, .2]));
  const lo = K.piano(ev('pianoLow'), { seed: 'l', n, felt: 0 }); addS(o, lo, 1.5); send(hl, lo, .6);
  const mb = K.musicBox(ev('mbox'), { n }); addS(o, mb, 1.0); send(hl, mb, .6); const ce = K.celesta(ev('celesta'), { n }); addS(o, ce, .9); send(hl, ce, .5);
  const eye = st(n); for (const e of ev('eye')) { pan(eye, D.crotale(1, mtof(e.n)), e.pan, e.v * .13, S(e.t)); const p = K.pianoNote(e.n, .5, .5, { seed: e.n }); addS(eye, p, .9 * e.v, S(e.t)); } addS(o, eye, .8); send(hl, eye, .9);
  const cv = P.clavDI(ev('clav'), 'cv', { n }), cvs = toS(cv, -.22, .8); addS(o, cvs, 1); send(rm, cvs, .35); send(pl, cvs, .05); const sp = K.piano(ev('stabs').map(e => ({ ...e, v: e.v * .8 })), { seed: 's', n }); filtS(sp, bqc.pk(3000, -3, 1)); addS(o, sp, .9); send(hl, sp, .3);
  const re = K.winds(ev('reed'), 'reed', { seed: 'r', n, players: 1 }); addS(o, re, .8); send(rm, re, .4);
  addS(o, C.hall(hl), .8); addS(o, C.plate(pl), .8); addS(o, C.room(rm), .55); return o;
};
R_.strings = () => {
  const o = st(n), hl = st(n); const H = e => ({ ...e });
  const sb = ev('stabs').map(e => ({ ...e, art: 'mar' })); const v1 = B.section(ev('v1').map(H), { seed: 'v1', n, players: 9, body: 'violin', center: -.38, width: .42 }), v2 = B.section([...ev('v2').map(H), ...sb.filter(e => e.n >= 62)], { seed: 'v2', n, players: 8, body: 'violin', center: .12, width: .5, bright: .9 }), va = B.section([...ev('va').map(H), ...sb.filter(e => e.n < 62 && e.n >= 52)], { seed: 'va', n, players: 6, body: 'viola', center: .42, width: .34, bright: .9 }), vc = B.section([...ev('vc').map(H), ...sb.filter(e => e.n < 52)], { seed: 'vc', n, players: 6, body: 'cello', center: .3, width: .3 }), cb = B.section(ev('cb').map(H), { seed: 'cb', n, players: 3, body: 'cello', center: .08, width: .15, bright: .7 });
  filtS(v1, bqc.pk(3200, -3.5, 1.1), bqc.hs(6000, -2)); filtS(v2, bqc.pk(3200, -3, 1.1), bqc.hs(6000, -2)); filtS(va, bqc.ls(300, -2.5)); filtS(vc, bqc.pk(180, -2.5, .8), bqc.hs(1500, 2)); filtS(cb, bqc.lp(1800, .7)); addS(o, v1, 1.25); addS(o, v2, 1.1); addS(o, va, 1.05); addS(o, vc, 1.15); addS(o, cb, 1.3); for (const [b, g] of [[v1, .42], [v2, .42], [va, .4], [vc, .3], [cb, .12]]) send(hl, b, g);
  const pz = P.pizzDI(ev('pizz'), 'pz', { n, body: 'violin' }), pL = P.pizzDI(ev('pizzL'), 'pl', { n, body: 'cello', players: 4 }); addS(o, pz, .85); addS(o, pL, .9); send(hl, pz, .22); send(hl, pL, .2);
  const vs = B.solo(ev('violinS'), { seed: 'vs', n, body: 'violin', bright: .95 }), cs = B.solo(ev('celloS'), { seed: 'cs', n, body: 'cello', vd: 18, bright: .95 }); const vS = toS(vs, -.14, 1), cS = toS(cs, .16, 1); filtS(vS, bqc.pk(3100, -3, 1.1)); filtS(cS, bqc.pk(250, -2, .8), bqc.pk(1600, 2, .9)); addS(o, vS, .7); addS(o, cS, .62); send(hl, vS, .5); send(hl, cS, .34);
  const scale = { F: [0, 2, 4, 5, 7, 9, 10].map(x => (x + 5) % 12), D: [2, 4, 6, 9, 11], G: [7, 9, 11, 2, 6] }; const hv = [...ev('harp')]; for (const e of ev('harpgl')) { const ns = []; for (let m = e.from; m <= e.to; m++) if (scale[e.scale].includes(((m % 12) + 12) % 12)) ns.push(m); ns.forEach((m, i) => hv.push({ t: e.t + e.dur * Math.pow(i / ns.length, .9), dur: 1.6, n: m, v: e.v * (.6 + .4 * i / ns.length) })); } const ha = P.harpDI(hv, 'h', { n }); const hS = toS(ha, -.3, .95); addS(o, hS, 1); send(hl, hS, .35);
  addS(o, C.hall(hl, { t60: 2.9, er: .6 }), 1.0); return o;
};
R_.jp = () => {
  const o = st(n), hl = st(n), pl = st(n); const ko = P.kotoDI(ev('koto'), 'k', { n }), ks = toS(ko, .36, .42); addS(o, ks, 1); send(hl, ks, .4); const uk = P.kotoDI(ev('ukoto'), 'u', { n }), us = toS(uk, .36, .36); addS(o, us, 1); send(hl, us, .6);
  const s1 = P.shamiDI(ev('shami'), 'a', { n }), s2 = P.shamiDI(ev('shami').map(e => ({ ...e, v: e.v * .9 })), 'b', { n }); const ss = st(n); pan(ss, s1, -.52, .3); pan(ss, s2, .52, .3); C.comp(ss, { thr: -18, ratio: 3, att: .006, rel: .07, makeup: 2 }); addS(o, ss, 1); send(hl, ss, .12); send(pl, ss, .08);
  const sk = K.shakuhachi(ev('shaku'), { n }), sks = toS(sk, -.2, .7); addS(o, sks, 1); send(hl, sks, .8);
  const tb = st(n), R = rng('tkperf'), cnt = {}; for (const e of ev('taiko')) { const ens = e.ens || 1; for (let k = 0; k < ens; k++) { const v = clamp(e.v * R.r(.86, 1.04), .05, 1), l = D.layer(v); cnt[e.p] = (cnt[e.p] || 0) + 1; const x = D.taiko(l, cnt[e.p] % 6, e.p); const g = v / l * { oo: .78, naga: .6, hira: .52, shime: .3, ka: .3 }[e.p] / Math.pow(ens, .62); pan(tb, x, e.p === 'oo' ? (k - (ens - 1) / 2) * .3 : clamp((ens > 1 ? (k / (ens - 1) - .5) * 1.3 : .1) + .1 * R.g(), -.9, .9), g, S(Math.max(0, e.t + (k ? Math.abs(R.g()) * .009 : 0) + .0015 * R.g()))); } } filtS(tb, bqc.hp(36, .7), bqc.pk(85, -3, .9), bqc.pk(230, -1.5, .9), bqc.pk(2600, 2.5, .9)); C.comp(tb, { thr: -14, ratio: 2.5, att: .012, rel: .14, makeup: 2 }); addS(o, tb, 1); const th = st(n); sendA(th, tb, secs([0, .34], [16, .28], [24, .16], [36, .3], [48, .34], [58, .18], [66, .3])); addS(o, C.hall(th, { t60: 2.4, er: .9, erSpan: .1, hp: 70 }), .8);
  addS(o, C.hall(hl), .8); addS(o, C.plate(pl), .8); return o;
};
R_.synth = () => {
  const o = st(n), hl = st(n), ca = st(n); const sb = Y.sub(ev('drone'), { n }); pan(o, sb, 0, .5); const hb = mk(n), R = rng('hb'); ev('kit').filter(e => e.p === 'hb').forEach((e, i) => add(hb, D.kick(D.layer(e.v), i % 4, 'heart'), e.v * .75, S(e.t))); filt(hb, bqc.lp(900, .7)); pan(o, hb, 0, 1);
  const bo = Y.boom(ev('boom'), { n }); pan(o, bo, 0, .62); const pu = Y.pulse(ev('pulse'), { n }); pan(o, pu, 0, .62); const pS = toS(pu, 0, .2); send(hl, pS, .25);
  const brt = Y.breath(ev('breath'), { n }); addS(o, brt, .5); send(hl, brt, .5);
  const cr = st(n); for (const e of ev('crot')) pan(cr, D.crotale(1, mtof(e.n)), rng(e.t).r(-.5, .5), e.v * .12, S(e.t)); for (const e of ev('glassbow')) { const x = mk(S(e.dur + .6)); const f = mtof(e.n); let ph = 0; for (let i = 0; i < x.length; i++) { const t = i / SR; ph += TAU * f * (1 + .0016 * Math.sin(TAU * 4.6 * t)) / SR; x[i] = (Math.sin(ph) + .2 * Math.sin(2.76 * ph)) * Math.min(1, t / .25) * (t > e.dur ? Math.max(0, 1 - (t - e.dur) / .6) : 1); } pan(cr, x, -.2, e.v * .045, S(e.t)); } addS(o, cr, 1); send(ca, cr, .9);
  const bm = B.bowedMetal(ev('bmetal'), { seed: 'bm', n }); addS(o, bm, .9); send(ca, bm, .5);
  const src = { crash: () => D.cym('crashA', 1, 0), tam: () => D.tamtam(1, 0), kane: () => { const s = st(S(1)); pan(s, D.kane(1, 0), 0, 1); return s; } }; for (const e of ev('rev')) Y.reverseInto(o, src[e.src](), e.t, e.dur, e.v * .55);
  const rs = Y.riser(ev('riser'), { n }); addS(o, rs, .85); send(hl, rs, .2); const sh = Y.shimmer(ev('shim'), { n }); addS(o, sh, .6); send(ca, sh, .9);
  const gl = Y.pad(ev('glassp'), { n, type: 'glass', voices: 4, det: 7, fc: 4200, seed: 'gp' }); addS(o, gl, .55); send(ca, gl, .8);
  for (const e of ev('flash')) { const Lq = S(e.dur + .05), x = mk(Lq), q = rng('fl' + e.t); let p = 0, m = 0; const f = mtof(e.n); for (let i = 0; i < Lq; i++) { const t = i / SR; p += f / SR; m += f * 1.414 / SR; const s = Math.sin(TAU * p) + .8 * Math.sin(TAU * p * 2.01 + 2.5 * Math.sin(TAU * m)); x[i] = Math.tanh(s * 3) * Math.min(1, t / .002) * Math.pow(Math.max(0, 1 - t / e.dur), 1.4); } filt(x, bqc.lp(2600, .8), bqc.hp(60, .7)); const c = st(Lq); pan(c, x, 0, e.v * .2); addS(o, c, 1, S(e.t)); }
  { const cho = K.choir(ev('hornpad'), { n, players: 6 }), cw = C.hall(cho, { t60: 2.2 }); gainS(cho, .5); addS(cho, cw, .42); fadeGate(cho, F(1710) - .006, F(1728) - .003, 0, .03, .002); addS(o, cho, 1); } const hm = K.hum(ev('hum'), { n, players: 5 }); addS(o, hm, .62); send(hl, hm, .5);
  for (const e of ev('ring')) { const Lq = S(e.dur), x = mk(Lq), f = mtof(e.n); for (let i = 0; i < Lq; i++) { const t = i / SR; x[i] = (Math.sin(TAU * f * t) + .5 * Math.sin(TAU * f * 1.0017 * t)) * Math.min(1, t / .03) * Math.pow(1 - t / e.dur, 1.6); } const c = st(Lq); pan(c, x, 0, e.v * .035); addS(o, c, 1, S(e.t)); addS(hl, c, .6, S(e.t)); }
  addS(o, C.hall(hl), .8); addS(o, C.cathedral(ca), .9); return o;
};
// ---------------------------------------------------------------- the picture acts on every stem
const tailGate = (b, t0, t1, tau) => { for (const x of [b.L, b.R]) for (let i = S(t0); i < Math.min(n, S(t1)); i++) { const t = i / SR; x[i] *= Math.exp(-(t - t0) / tau) * Math.min(1, (t1 - t) / .002 + .0); } };
function act(name, b) {
  // f333 slap (bar 9.25): stop, everything but the hit; f864 a breath before the chorus; f1116 sound dies; f1341-1360 clink; f1710 cut; f2304-2358 hold
  const g = (t0, t1, fl, a = .03, z = .004) => fadeGate(b, t0, t1, fl, a, z);
  const drum = name === 'drums';
  for (const f of [909, 972, 1152, 1206, 1764, 1782, 1854, 1872, 1890, 1908, 1917, 1926, 1944, 1962, 1980, 1998, 2124, 2142, 2196, 2205, 2214, 2223, 2232, 2241, 2250, 2259, 2268]) g(F(f) - .034, F(f) - .004, .1, .016, .002);
  for (const f of [54, 126, 558, 702, 954]) g(F(f) - .036, F(f) - .004, .14, .018, .002); g(F(504) - .045, F(504) - .004, .3, .025, .002); g(F(2520) - .045, F(2520) - .004, .35, .03, .002); for (const f of [891, 1035]) g(F(f) - .03, F(f) - .004, .12, .014, .002);
  for (const f of [414, 576, 612, 621, 630, 639, 720, 738, 756, 774, 792]) g(F(f) - .042, F(f) - .004, f === 414 ? .25 : .16, .02, .002); g(F(486) - .05, F(486) - .004, .4, .03, .002); g(F(2700) - .05, F(2700) - .004, .45, .03, .002);
  g(F(2088) - .085, F(2088) - .004, .015, .02, .003); g(F(2304) - .05, F(2304) - .004, .06, .02, .003);
  g(F(333) + (drum ? .5 : .1), F(351) - .02, drum ? .25 : .05, .32, .015);
  g(F(108) - .06, F(108) - .004, .03, .03, .003); g(F(864) - .05, F(864) - .004, .02, .012, .003);
  tailGate(b, F(1116) + .03, F(1134) - .003, .034);
  if (name === 'keys' || name === 'strings') g(F(1341) - .02, F(1363), .05, .16, .12);
  if (name !== 'synth') { g(F(1710) - .006, F(1728) - .003, 0, .03, .002); }
  tailGate(b, F(2016) + .16, F(2052) - .003, .05);
  tailGate(b, F(2322), F(2358) - .003, .05);
  // d6: swallowed into a reversed void at f1242
  if (name !== 'synth') { const t0 = F(1242), m = S(.75), a0 = S(t0); const seg = { L: b.L.slice(a0 - m, a0), R: b.R.slice(a0 - m, a0) }; g(t0, F(1296) - .003, 0, .03, .002); const Lq = S(1.5), v = st(n); let pos = 0; const s1 = new C.SVF(), s2 = new C.SVF(); for (let i = 0; i < Lq; i++) { const t = i / SR; if ((i & 3) === 0) { const fc = 5200 * Math.exp(-t / .32) + 120; s1.set(fc, .8); s2.set(fc, .8); } const q = m - 1 - pos; pos += 1.05 * Math.exp(-t / .9); if (q < 1) break; const ii = Math.floor(q), f = q - ii, l = seg.L[ii] * (1 - f) + seg.L[ii + 1 < m ? ii + 1 : ii] * f, r = seg.R[ii] * (1 - f) + seg.R[ii + 1 < m ? ii + 1 : ii] * f; const e = Math.pow(Math.max(0, 1 - t / 1.15), 1.2) * Math.min(1, t / .004); v.L[a0 + i] = s1.run(l) * e * .8; v.R[a0 + i] = s2.run(r) * e * .8; } const w = C.hall(v, { t60: 1.6, er: .2 }); addS(b, v, .9); for (const x of [w.L, w.R]) for (let i = 0; i < n; i++) { const t = i / SR; if (t > F(1290)) x[i] = 0; else if (t > F(1272)) x[i] *= Math.exp(-(t - F(1272)) / .07); } addS(b, w, .7); }
  else tailGate(b, F(1276), F(1296) - .003, .06);
  return b;
}
// the story in numbers: LUFS of every stem in every stretch of the film (null = silent or leave alone)
const NAMES0 = ['drums', 'bass', 'guitars', 'keys', 'strings', 'jp', 'synth'];
const TARGET = [ // [bar0, bar1, drums, bass, guitars, keys, strings, jp, synth]
  [0, 2, -31, null, null, -28, null, -26, -23], [2, 3, -33, null, null, -31, -26, -24, -26], [3, 4, -19.5, -24, -21, -21, -19, -18.5, -19],
  [4, 6, -41, -31, -29, -26, -36, -28.5, -39], [6, 8, -28, -27, -29, -26.5, -29, -32, null], [8, 10, -27, -27, null, -26.5, -26.5, null, null], [10, 12, -24.5, -24.5, -30, -26.5, null, null, null], [12, 14, -30.5, -26.5, -32, -26.5, -31, -35, null], [14, 16, -29, -30, null, -24.5, -20.5, -31, -31],
  [16, 18, -31, null, null, -25, -24.5, -24.5, -23.5], [18, 20, -42, null, null, -25, -23, -27, -22.5], [20, 22, -23, null, -26, -28, -19.5, -19.5, -22], [22, 23, -19, -22, -20.5, -24, -18.5, -18.5, -22.5], [23, 24, -20, -23.5, -22.5, null, -17, -20.5, -20.5],
  [24, 26, -15, -20.5, -18, null, -20.5, -25, -27], [26, 28, -15, -20.5, -18.5, -29, -22, -24, -27], [28, 30, -15.5, -19, -22.5, -24.5, -24, -26, null], [30, 31, -17.5, -21.5, -20, null, -18.5, -21.5, -25], [31, 32, -14, -21, -17.5, -22.5, -18.5, -21, -18.5], [32, 34, -14.5, -20, -16.5, -22, -18.5, -25, -24], [34, 36, -15, -20, -16.5, -22, -19, -25, -23],
  [36, 38, null, null, null, -24, null, null, -34], [38, 40, null, null, null, -26, -24, null, null], [40, 42, null, null, null, -26.5, -23.5, null, -32], [42, 44, null, null, null, -26.5, -23, null, -37], [44, 46, -32, -33, -33, -22, -18, null, null], [46, 48, -21, -25, -23, -19, -14.5, -20.5, -23],
  [48, 50, -28, -22.5, -19.5, null, -21, -18.5, -22], [50, 52, -16, -20.5, -18.5, -26, -22, -18, -29], [52, 54, -16, -20.5, -18.5, -24.5, -22.5, -17.5, -19], [54, 56, -15.5, -20, -18, -23, -19.5, -18.5, -20.5], [56, 57, -21, -24, -20, null, null, -21, -25], [57, 58, -18, -23, -20, null, -17, -18, -20],
  [58, 60, -13.5, -18.5, -15.5, -21, -16.5, -21.5, -19], [60, 62, -13, -18.5, -15.5, -21, -16.5, -20, -23], [62, 64, -13, -18.5, -15, -20.5, -15.5, -21, -20], [64, 65, -14, -19, -15.5, -21, -15.5, -20.5, -22.5], [65, 66, -16, -21, -18, -19, -18, -20, -18],
  [66, 68, null, null, null, -25, -34, null, -28], [68, 70, -39, null, null, -26, -27.5, -31, -26], [70, 71, -35, -30, null, -24.5, -21, -37, -25.5], [71, 73, -33, -29, -28, -25.5, -27, -28, -30], [73, 76, -37, -30, null, -23, -22.5, -31, -26]];
for (const r of TARGET) { const D = r[0] >= 24 && r[0] < 36, H = r[0] >= 58 && r[0] < 66, Fq = r[0] >= 48 && r[0] < 56; const add_ = D ? [-1.5, -1.2, .6, -1.5, -2.6, -1.5, -1.5] : H ? [1.2, 1.5, 2.8, 1.2, .9, 1.5, 1.8] : Fq ? [0, 0, 1.2, 0, 0, 0, 0] : null; if (add_) add_.forEach((v, i) => { if (r[i + 2] !== null) r[i + 2] += v; }); }
{ const q = TARGET.find(r => r[0] === 10); q[5] = -25; q[6] = -27.5; q[4] = -28.5; } { const q = TARGET.find(r => r[0] === 22); q[8] = -23; } for (const r of TARGET) if (r[0] >= 58 && r[0] < 65 && r[8] === null) r[8] = -20;
function levels(name, b) { const c = NAMES0.indexOf(name) + 2, pts = []; let last = null; for (const r of TARGET) { const tg = r[c]; let g = last; if (tg !== null) { const m = C.lufs(b, S(r[0] * 1.5), Math.min(n, S(r[1] * 1.5))); if (m > -62) g = clamp(tg - m, -30, 24); } if (g === null) g = -12; if (last !== null) pts.push([r[0] * 1.5 - .012, last]); pts.push([r[0] * 1.5 + .004, g]); last = g; } for (const x of [b.L, b.R]) { let k = 0; for (let i = 0; i < n; i += 24) { const g = db(C.envAt(pts, i / SR)); for (let j = i; j < i + 24 && j < n; j++) x[j] *= g; } } return pts; }
const TRIM = -6.5, MAKEUP = 6.5; // stems leave with headroom; band_mix puts the level back before its own master
const NAMES = ['drums', 'bass', 'guitars', 'keys', 'strings', 'jp', 'synth'];
// while anyone sings (times from score.json), 1.2-3.5 kHz of guitars, strings and Japanese strings steps back 3 dB, and the middle a little more
const SUNG = (() => { const sc = JSON.parse(fs.readFileSync(path.join(C.HERE, 'score.json'))), iv = []; for (const l of sc.lines) for (const q of l.notes) iv.push([q.t / 8 * 1.5 - .06, (q.t + q.d) / 8 * 1.5 + .05]); iv.sort((x, y) => x[0] - y[0]); const o = []; for (const q of iv) { const z = o[o.length - 1]; if (z && q[0] - z[1] < .35) z[1] = Math.max(z[1], q[1]); else o.push(q.slice()); } return o; })();
function sungGate() { const g = mk(n), up = 1 - Math.exp(-1 / (.03 * SR)), dn = 1 - Math.exp(-1 / (.16 * SR)); let k = 0, e = 0; for (let i = 0; i < n; i++) { const t = i / SR; while (k < SUNG.length - 1 && t > SUNG[k][1]) k++; const on = t >= SUNG[k][0] && t <= SUNG[k][1] ? 1 : 0; e += (on > e ? up : dn) * (on - e); g[i] = e; } return g; }
function pocket(name, b) { const wide = { strings: 1.22, guitars: 1.05, keys: 1.06, jp: 1.1 }[name] || 1, dip = { guitars: -3, strings: -3, jp: -3, keys: -1.5, synth: -1 }[name] || 0, mid = { guitars: -2, strings: -2, jp: -1.5, keys: -2, synth: -1.5 }[name] || 0; if (!dip && wide === 1) return; const g = sungGate(); for (let i = 0; i < n; i++) { const m = (b.L[i] + b.R[i]) * .5, q = (b.L[i] - b.R[i]) * .5 * wide; b.L[i] = m; b.R[i] = q; } const side = Float32Array.from(b.R), mm = Float32Array.from(b.L); filt(mm, bqc.pk(2050, dip + mid, .85)); filt(side, bqc.pk(2050, dip, .85)); for (let i = 0; i < n; i++) { const m = b.L[i] + (mm[i] - b.L[i]) * g[i], q = b.R[i] + (side[i] - b.R[i]) * g[i]; b.L[i] = m + q; b.R[i] = m - q; } }
function renderStem(name) { const t0 = Date.now(); let b = R_[name](); pocket(name, b); const pts = levels(name, b); if (opt('v')) console.log(name, pts.filter((_, i) => !(i & 1)).map(q => (q[0] / 1.5).toFixed(0) + ':' + q[1].toFixed(1)).join(' ')); if (name === 'drums') { for (const x of [b.L, b.R]) for (let i = 0; i < n; i++) { const v = x[i]; x[i] = Math.abs(v) < .4 ? v : Math.sign(v) * (.4 + .42 * Math.tanh((Math.abs(v) - .4) / .42)); } } b = act(name, b); gainS(b, db(TRIM)); C.limiter(b, { ceil: -3.4, rel: .06, look: .003 }); for (const x of [b.L, b.R]) { filt(x, bqc.hp(18, .7)); for (let i = 0; i < x.length; i++) if (Number.isNaN(x[i])) throw new Error('NaN in ' + name); C.fade(x, .005, .2); } C.writeWav(path.join(STEMS, `band_${name}.wav`), b); console.log(`${name}: ${((Date.now() - t0) / 1000).toFixed(1)} s, peak ${todb(Math.max(C.peak(b.L), C.peak(b.R))).toFixed(1)} dB, LUFS ${C.lufs(b).toFixed(1)}`); }
export function master(m) { // gentle bus: glue, tape, limiter
  filtS(m, bqc.hp(24, .7), bqc.pk(3000, -.8, .8), bqc.hs(9500, 1)); C.comp(m, { thr: -15, ratio: 1.7, att: .025, rel: .18, knee: 10, makeup: 1.5 }); for (const x of [m.L, m.R]) for (let i = 0; i < x.length; i++) { const v = x[i]; x[i] = v - .06 * v * v * v * (Math.abs(v) < 1.4 ? 1 : 0); } // bass to mono below 110 Hz
  const s = mk(n); for (let i = 0; i < n; i++) s[i] = (m.L[i] - m.R[i]) * .5; filt(s, bqc.hp(110, .7)); for (let i = 0; i < n; i++) { const c = (m.L[i] + m.R[i]) * .5; m.L[i] = c + s[i]; m.R[i] = c - s[i]; } return C.limiter(m, { ceil: -1.2, rel: .09 }); }
function mix() { const m = st(n); for (const k of NAMES) { const f = path.join(STEMS, `band_${k}.wav`); if (fs.existsSync(f)) addS(m, C.readWav(f), 1); else console.log('missing', k); } const pre = Math.max(C.peak(m.L), C.peak(m.R)); gainS(m, db(MAKEUP)); master(m); C.fade(m.L, .005, .3); C.fade(m.R, .005, .3); C.writeWav24(path.join(OUT, 'band_mix.wav'), m); console.log('pre-master peak', todb(pre).toFixed(1), 'mix LUFS', C.lufs(m).toFixed(1), 'TP', C.truePeak(m).toFixed(2)); for (const [nm, a, z] of [['A', 0, 4], ['B', 4, 14], ['B6', 14, 16], ['C', 16, 22], ['C4', 22, 24], ['D1-3', 24, 30], ['D4-5', 30, 34], ['D6', 34, 36], ['E1-4', 36, 44], ['E5', 44, 47.5], ['F', 48, 56], ['F5', 56, 58], ['H', 58, 65], ['G', 66, 76]]) console.log(nm.padEnd(5), 'LUFS', C.lufs(m, S(a * 1.5), S(z * 1.5)).toFixed(1)); }
// ---------------------------------------------------------------- checks
function check() { const sc = JSON.parse(fs.readFileSync(path.join(C.HERE, 'score.json'))); const PCS = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'], pc = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 }; const ct = c => { if (c === 'N') return null; const [h, bass] = c.split('/'); const m = /^([A-G][b#]?)(.*)$/.exec(h); let r = pc[m[1][0]] + (m[1][1] === 'b' ? -1 : m[1][1] === '#' ? 1 : 0); const q = m[2]; const iv = { '': [0, 4, 7], m: [0, 3, 7], m7: [0, 3, 7, 10], 7: [0, 4, 7, 10], maj7: [0, 4, 7, 11], 5: [0, 7], '7sus4': [0, 5, 7, 10], '7b9': [0, 1, 4, 7, 10] }[q]; const s = new Set(iv.map(x => (x + r + 12) % 12)); if (bass) s.add(pc[bass[0]] + (bass[1] === '#' ? 1 : 0)); return s; }; const inst = ['v1', 'v2', 'va', 'vc', 'cb', 'piano', 'felt', 'feltG', 'nylon', 'bass', 'ubass', 'slap', 'brass', 'stabs', 'hornpad', 'rg', 'rg2', 'koto', 'hum', 'mbox', 'celloS', 'violinS', 'lead', 'leadq', 'celesta', 'pizz', 'jazzp', 'glassp']; const out = []; for (const k of inst) for (const e of ev(k)) for (const m of Array.isArray(e.n) ? e.n : [e.n]) { const b = Math.floor(e.t / 1.5 + .01), s = ct(sc.chords[b]); if (!s) continue; const p = ((m % 12) + 12) % 12; if (!s.has(p) && e.dur > .3) out.push(`bar ${(e.t / 1.5).toFixed(2)} ${sc.chords[b]} ${k} ${PCS[p]}${Math.floor(m / 12) - 1} ${e.dur.toFixed(2)}s`); } console.log(out.length, 'non-chord tones longer than 0.3 s'); console.log(out.join('\n')); }
// ---------------------------------------------------------------- main
const cmd = argv[0];
if (cmd === 'stem') renderStem(argv[1]);
else if (cmd === 'mix') mix();
else if (cmd === 'check') check();
else if (cmd === 'all') { // strictly one process at a time (docs/BRIEF.md, MACHINE RULES)
  const q = opt('only') ? opt('only')[0].split(',') : NAMES, extra = [...(bars ? ['--bars', ...bars.map(String)] : []), ...(opt('out') ? ['--out', OUT] : [])]; for (const k of q) { const r = spawnSync(process.execPath, ['--max-old-space-size=2500', process.argv[1], 'stem', k, ...extra], { stdio: 'inherit' }); if (r.status) { console.log(k, 'failed', r.status); process.exit(1); } } mix(); }
