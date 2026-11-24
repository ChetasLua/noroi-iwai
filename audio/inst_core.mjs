// inst_core.mjs: DSP core for the band. Plain JavaScript, no packages, no samples.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
export const HERE = path.dirname(fileURLToPath(import.meta.url));
export const ROOT = path.resolve(HERE, '..');
export const SR = 48000, SECS = 114, N = SECS * SR, BAR = 1.5, E8 = 0.1875;
export const T = (bar, e = 0) => bar * BAR + e * E8;           // bar, eighths -> seconds
export const F = f => f / 24;                                  // film frame -> seconds
export const TAU = Math.PI * 2;
export const mk = (n = N) => new Float32Array(n);
export const st = (n = N) => ({ L: new Float32Array(n), R: new Float32Array(n) });
export const db = x => Math.pow(10, x / 20);
export const todb = x => 20 * Math.log10(Math.max(1e-12, x));
export const clamp = (x, a, b) => x < a ? a : x > b ? b : x;
export const lerp = (a, b, t) => a + (b - a) * t;
export const mtof = m => 440 * Math.pow(2, (m - 69) / 12);
const PC = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
export function nt(s) { if (typeof s === 'number') return s; const m = /^([A-G])([#b]*)(-?\d)$/.exec(s); if (!m) throw new Error('note ' + s); let p = PC[m[1]] + 12 * (+m[3] + 1); for (const c of m[2]) p += c === '#' ? 1 : -1; return p; }
export function hash(s) { let h = 2166136261; s = String(s); for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; }
export function rng(seed) {
  let a = hash(seed) || 1, sp = null;
  const u = () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  const g = () => { if (sp !== null) { const v = sp; sp = null; return v; } let x, y, r; do { x = u() * 2 - 1; y = u() * 2 - 1; r = x * x + y * y; } while (r >= 1 || r === 0); const k = Math.sqrt(-2 * Math.log(r) / r); sp = y * k; return x * k; };
  return { u, g, r: (lo, hi) => lo + (hi - lo) * u(), pick: arr => arr[Math.floor(u() * arr.length)] };
}
// ------------------------------------------------------------ filters
export const bqc = {
  lp(f, q = .7071, sr = SR) { const w = TAU * Math.min(f, sr * .49) / sr, c = Math.cos(w), a = Math.sin(w) / (2 * q), n = 1 + a; return [(1 - c) / 2 / n, (1 - c) / n, (1 - c) / 2 / n, -2 * c / n, (1 - a) / n]; },
  hp(f, q = .7071, sr = SR) { const w = TAU * Math.min(f, sr * .49) / sr, c = Math.cos(w), a = Math.sin(w) / (2 * q), n = 1 + a; return [(1 + c) / 2 / n, -(1 + c) / n, (1 + c) / 2 / n, -2 * c / n, (1 - a) / n]; },
  bp(f, q = 1, sr = SR) { const w = TAU * Math.min(f, sr * .49) / sr, c = Math.cos(w), a = Math.sin(w) / (2 * q), n = 1 + a; return [a / n, 0, -a / n, -2 * c / n, (1 - a) / n]; },
  pk(f, g, q = 1, sr = SR) { const A = Math.pow(10, g / 40), w = TAU * Math.min(f, sr * .49) / sr, c = Math.cos(w), a = Math.sin(w) / (2 * q), n = 1 + a / A; return [(1 + a * A) / n, -2 * c / n, (1 - a * A) / n, -2 * c / n, (1 - a / A) / n]; },
  ls(f, g, q = .7071, sr = SR) { const A = Math.pow(10, g / 40), w = TAU * f / sr, c = Math.cos(w), al = Math.sin(w) / (2 * q), s = 2 * Math.sqrt(A) * al, n = (A + 1) + (A - 1) * c + s; return [A * ((A + 1) - (A - 1) * c + s) / n, 2 * A * ((A - 1) - (A + 1) * c) / n, A * ((A + 1) - (A - 1) * c - s) / n, -2 * ((A - 1) + (A + 1) * c) / n, ((A + 1) + (A - 1) * c - s) / n]; },
  hs(f, g, q = .7071, sr = SR) { const A = Math.pow(10, g / 40), w = TAU * Math.min(f, sr * .49) / sr, c = Math.cos(w), al = Math.sin(w) / (2 * q), s = 2 * Math.sqrt(A) * al, n = (A + 1) - (A - 1) * c + s; return [A * ((A + 1) + (A - 1) * c + s) / n, -2 * A * ((A - 1) + (A + 1) * c) / n, A * ((A + 1) + (A - 1) * c - s) / n, 2 * ((A - 1) - (A + 1) * c) / n, ((A + 1) - (A - 1) * c - s) / n]; },
  ap(f, q = .7071, sr = SR) { const w = TAU * f / sr, c = Math.cos(w), a = Math.sin(w) / (2 * q), n = 1 + a; return [(1 - a) / n, -2 * c / n, 1, -2 * c / n, (1 - a) / n]; },
  res(f, t60, sr = SR) { const r = Math.pow(10, -3 / (t60 * sr)), w = TAU * f / sr; return [Math.sin(w), 0, 0, -2 * r * Math.cos(w), r * r]; },   // modal resonator, impulse response sin(w(n+1)) r^n
};
export function filt(x, ...cs) {                              // in place cascade
  const n = x.length;
  for (const c of cs) { if (!c) continue; const [b0, b1, b2, a1, a2] = c; let z1 = 0, z2 = 0;
    for (let i = 0; i < n; i++) { const v = x[i], y = b0 * v + z1; z1 = b1 * v - a1 * y + z2; z2 = b2 * v - a2 * y; x[i] = y; } }
  return x;
}
export function filtS(b, ...cs) { filt(b.L, ...cs); filt(b.R, ...cs); return b; }
export class SVF { // TPT state variable, modulated
  constructor() { this.a = 0; this.b = 0; }
  set(f, q, sr = SR) { this.g = Math.tan(Math.PI * Math.min(f, sr * .45) / sr); this.k = 1 / q; this.d = 1 / (1 + this.g * (this.g + this.k)); }
  run(x) { const hp = (x - (this.g + this.k) * this.a - this.b) * this.d, v1 = this.g * hp, bp = v1 + this.a; this.a = bp + v1; const v2 = this.g * bp, lp = v2 + this.b; this.b = lp + v2; this.hp = hp; this.bp = bp; this.lp = lp; return lp; }
}
export function dc(x, r = .9985) { let px = 0, py = 0; for (let i = 0; i < x.length; i++) { const y = x[i] - px + r * py; px = x[i]; py = y; x[i] = y; } return x; }
// ------------------------------------------------------------ buffers
export function add(dst, src, g = 1, at = 0) { const n = Math.min(src.length, dst.length - at); for (let i = Math.max(0, -at); i < n; i++) dst[at + i] += src[i] * g; return dst; }
export function addS(dst, src, g = 1, at = 0) { add(dst.L, src.L, g, at); add(dst.R, src.R, g, at); return dst; }
export function pan(dst, mono, p = 0, g = 1, at = 0) { const a = (p + 1) * Math.PI / 4; add(dst.L, mono, g * Math.cos(a) * 1.4142, at); add(dst.R, mono, g * Math.sin(a) * 1.4142, at); return dst; }
export function gain(x, g) { for (let i = 0; i < x.length; i++) x[i] *= g; return x; }
export function gainS(b, g) { gain(b.L, g); gain(b.R, g); return b; }
export function copyS(b) { return { L: Float32Array.from(b.L), R: Float32Array.from(b.R) }; }
export function peak(x) { let p = 0; for (let i = 0; i < x.length; i++) { const a = Math.abs(x[i]); if (a > p) p = a; } return p; }
export function rms(x, a = 0, b = x.length) { let s = 0; for (let i = a; i < b; i++) s += x[i] * x[i]; return Math.sqrt(s / Math.max(1, b - a)); }
export function norm(x, p = 1) { const k = peak(x); if (k > 0) gain(x, p / k); return x; }
export function fade(x, inS, outS) { const a = Math.floor(inS * SR), b = Math.floor(outS * SR), n = x.length; for (let i = 0; i < a && i < n; i++) x[i] *= .5 - .5 * Math.cos(Math.PI * i / a); for (let i = 0; i < b && i < n; i++) x[n - 1 - i] *= .5 - .5 * Math.cos(Math.PI * i / b); return x; }
// piecewise gain automation in seconds: [[t,dB],...] (linear in dB)
export function autom(b, pts) {
  const bufs = b.L ? [b.L, b.R] : [b]; const n = bufs[0].length; let k = 0;
  for (let i = 0; i < n; i += 32) { const t = i / SR; while (k < pts.length - 2 && t >= pts[k + 1][0]) k++;
    const p = pts[k], q = pts[Math.min(k + 1, pts.length - 1)]; const g0 = t <= p[0] ? p[1] : t >= q[0] ? q[1] : p[1] + (q[1] - p[1]) * (t - p[0]) / (q[0] - p[0]);
    const t2 = (i + 32) / SR; let g1 = t2 <= p[0] ? p[1] : t2 >= q[0] ? q[1] : p[1] + (q[1] - p[1]) * (t2 - p[0]) / (q[0] - p[0]);
    const a = g0 <= -90 ? 0 : db(g0), c = g1 <= -90 ? 0 : db(g1);
    for (let j = 0; j < 32 && i + j < n; j++) { const g = a + (c - a) * j / 32; for (const x of bufs) x[i + j] *= g; } }
  return b;
}
export function envAt(pts, t) { if (t <= pts[0][0]) return pts[0][1]; for (let k = 0; k < pts.length - 1; k++) if (t < pts[k + 1][0]) { const p = pts[k], q = pts[k + 1]; return p[1] + (q[1] - p[1]) * (t - p[0]) / (q[0] - p[0]); } return pts[pts.length - 1][1]; }
// ------------------------------------------------------------ FFT convolution (two reals at once)
export class FFT {
  constructor(n) { this.n = n; this.c = new Float64Array(n / 2); this.s = new Float64Array(n / 2); for (let i = 0; i < n / 2; i++) { this.c[i] = Math.cos(TAU * i / n); this.s[i] = Math.sin(TAU * i / n); } this.rev = new Uint32Array(n); let b = Math.log2(n); for (let i = 0; i < n; i++) { let r = 0; for (let j = 0; j < b; j++) if (i >> j & 1) r |= 1 << (b - 1 - j); this.rev[i] = r; } }
  run(re, im, inv) { const n = this.n, rev = this.rev, C = this.c, S = this.s; for (let i = 0; i < n; i++) { const j = rev[i]; if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
    for (let sz = 2; sz <= n; sz <<= 1) { const h = sz >> 1, step = n / sz; for (let i = 0; i < n; i += sz) for (let k = 0, t = 0; k < h; k++, t += step) { const c = C[t], s = inv ? S[t] : -S[t], a = i + k, b = a + h; const xr = re[b] * c - im[b] * s, xi = re[b] * s + im[b] * c; re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi; } }
    if (inv) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; } }
}
export function convolve2(xL, xR, ir, g = 1) {                // returns [yL,yR], same length as input, real IR
  const n = xL.length, M = ir.length; let sz = 1024; while (sz < M * 3) sz <<= 1; const B = sz - M, f = new FFT(sz);
  const hr = new Float64Array(sz), hi = new Float64Array(sz); for (let i = 0; i < M; i++) hr[i] = ir[i] * g; f.run(hr, hi, false);
  const yL = new Float32Array(n), yR = new Float32Array(n), re = new Float64Array(sz), im = new Float64Array(sz);
  for (let p = 0; p < n; p += B) { let any = false; re.fill(0); im.fill(0); const e = Math.min(B, n - p);
    for (let i = 0; i < e; i++) { const a = xL[p + i], b = xR ? xR[p + i] : 0; re[i] = a; im[i] = b; if (a !== 0 || b !== 0) any = true; }
    if (!any) continue; f.run(re, im, false); for (let i = 0; i < sz; i++) { const a = re[i], b = im[i]; re[i] = a * hr[i] - b * hi[i]; im[i] = a * hi[i] + b * hr[i]; } f.run(re, im, true);
    const q = Math.min(sz, n - p); for (let i = 0; i < q; i++) { yL[p + i] += re[i]; yR[p + i] += im[i]; } }
  return [yL, yR];
}
export const convM = (x, ir, g = 1) => convolve2(x, null, ir, g)[0];
// ------------------------------------------------------------ resampling (oversampling for nonlinear stages)
function kaiser(n, beta) { const I0 = x => { let s = 1, t = 1; for (let k = 1; k < 30; k++) { t *= (x / 2 / k) ** 2; s += t; } return s; }; const w = new Float64Array(n), d = I0(beta); for (let i = 0; i < n; i++) { const r = 2 * i / (n - 1) - 1; w[i] = I0(beta * Math.sqrt(1 - r * r)) / d; } return w; }
const protos = {};
function proto(L, per = 24) { const key = L + '_' + per; if (protos[key]) return protos[key]; const n = L * per + 1, h = new Float64Array(n), w = kaiser(n, 8), fc = .46 / L; let s = 0; for (let i = 0; i < n; i++) { const t = i - (n - 1) / 2; h[i] = (t === 0 ? 2 * fc : Math.sin(TAU * fc * t) / (Math.PI * t)) * w[i]; s += h[i]; } for (let i = 0; i < n; i++) h[i] /= s; return protos[key] = h; }
export function upsample(x, L) { const h = proto(L), n = h.length, half = (n - 1) / 2, y = new Float32Array(x.length * L);
  for (let ph = 0; ph < L; ph++) { const taps = []; for (let k = 0; k < n; k += 1) if ((k % L) === ((ph + half) % L)) taps.push(k); const m = taps.length, c = new Float64Array(m); for (let j = 0; j < m; j++) c[j] = h[taps[j]] * L; const off = new Int32Array(m); for (let j = 0; j < m; j++) off[j] = (ph + half - taps[j]) / L;
    for (let i = 0; i < x.length; i++) { let s = 0; for (let j = 0; j < m; j++) { const q = i + off[j]; if (q >= 0 && q < x.length) s += x[q] * c[j]; } y[i * L + ph] = s; } }
  return y; }
export function downsample(y, L) { const h = proto(L), n = h.length, half = (n - 1) / 2, m = Math.floor(y.length / L), x = new Float32Array(m);
  for (let i = 0; i < m; i++) { const c = i * L; let s = 0; const a = Math.max(0, c - half), b = Math.min(y.length - 1, c + half); for (let q = a; q <= b; q++) s += y[q] * h[q - c + half]; x[i] = s; } return x; }
// ------------------------------------------------------------ dynamics
export function comp(b, { thr = -18, ratio = 3, att = .01, rel = .12, knee = 6, makeup = 0, mix = 1, hp = 0, look = 0 } = {}) {
  const L = b.L || b, R = b.R || b, n = L.length, aA = Math.exp(-1 / (att * SR)), aR = Math.exp(-1 / (rel * SR)); let e = 0, g = 0, hx = 0, hy = 0, lo = 0, hi = 0; const hc = hp ? Math.exp(-TAU * hp / SR) : 0, mu = db(makeup), la = Math.floor(look * SR);
  for (let i = 0; i < n; i++) { const j = Math.min(n - 1, i + la); let s = Math.max(Math.abs(L[j]), Math.abs(R[j])); if (hp) { const sl = L[j], sr_ = R[j]; lo = hc * lo + (1 - hc) * sl; hi = hc * hi + (1 - hc) * sr_; s = Math.max(Math.abs(sl - lo), Math.abs(sr_ - hi)); }
    const d = 8.685889638 * Math.log(s + 1e-9), o = d - thr; let r = 0; if (2 * o > knee) r = o * (1 / ratio - 1); else if (2 * o > -knee) r = (1 / ratio - 1) * (o + knee / 2) ** 2 / (2 * knee);
    g = r < g ? aA * g + (1 - aA) * r : aR * g + (1 - aR) * r; const k = Math.exp(g * .115129255) * mu * mix + (1 - mix); L[i] *= k; if (R !== L) R[i] *= k; }
  return b;
}
export function satS(b, drive = 1, mix = 1) { for (const x of b.L ? [b.L, b.R] : [b]) { const k = 1 / Math.tanh(drive); for (let i = 0; i < x.length; i++) { const v = x[i]; x[i] = v * (1 - mix) + mix * Math.tanh(v * drive) / drive; } } return b; }
export function limiter(b, { ceil = -1, rel = .08, look = .004 } = {}) {                       // look-ahead, 4x inter-sample aware
  const { L, R } = b, n = L.length, la = Math.max(8, Math.floor(look * SR)), c = db(ceil), need = new Float32Array(n);
  const k1 = [-.0166, .0979, -.3127, .5 * 1.8, .3936, -.1209, .0233].map(x => x);
  for (let i = 0; i < n; i++) { let p = Math.max(Math.abs(L[i]), Math.abs(R[i])); if (i > 2 && i < n - 3 && p > c * .6) for (const x of [L, R]) for (const ph of [.25, .5, .75]) { // 6-point Lagrange
      let s = 0; for (let j = -2; j <= 3; j++) { let w = 1; for (let k = -2; k <= 3; k++) if (k !== j) w *= (ph - k) / (j - k); s += x[i + j] * w; } const a = Math.abs(s); if (a > p) p = a; }
    need[i] = p > c ? c / p : 1; }
  // sliding min then smooth
  const mn = new Float32Array(n), dq = new Int32Array(n); let h = 0, t = 0; for (let i = 0; i < n + la; i++) { if (i < n) { while (t > h && need[dq[t - 1]] >= need[i]) t--; dq[t++] = i; } const o = i - la; if (o >= 0) { while (h < t && dq[h] < o - 0) h++; mn[o] = h < t ? need[dq[h]] : 1; } }
  const aR = Math.exp(-1 / (rel * SR)); let g = 1; const gs = new Float32Array(n); for (let i = 0; i < n; i++) { g = mn[i] < g ? mn[i] : aR * g + (1 - aR) * mn[i]; gs[i] = g; }
  // box smooth over look-ahead (causal average of min-held curve)
  let acc = 0; const w = la; const out = new Float32Array(n); for (let i = 0; i < n; i++) { acc += gs[i]; if (i >= w) acc -= gs[i - w]; out[i] = acc / Math.min(i + 1, w); }
  for (let i = 0; i < n; i++) { const k = Math.min(out[i], need[i] < 1 ? need[i] : 1); L[i] *= k; R[i] *= k; }
  return b;
}
// ------------------------------------------------------------ reverbs: 16-line FDN, Hadamard, modulated, frequency dependent decay
const PRIMES = [1433, 1601, 1867, 2053, 2251, 2399, 2617, 2797, 3089, 3323, 3559, 3881, 4127, 4409, 4733, 5077];
export function fdn(b, { t60 = 1.8, hiMul = .45, loMul = 1.1, fHi = 4200, fLo = 220, pre = .012, size = 1, mod = 6, hp = 160, lp = 9500, er = .5, erSpan = .06, width = 1, diff = .62, seed = 'fdn', g = 1 } = {}) {
  const n = b.L.length, R = rng(seed), NL = 16, D = [], len = new Int32Array(NL), pos = new Int32Array(NL);
  const gL = new Float64Array(NL), gM = new Float64Array(NL), gH = new Float64Array(NL), zL = new Float64Array(NL), zH = new Float64Array(NL), ph = new Float64Array(NL), dp = new Float64Array(NL), ma = new Float64Array(NL);
  for (let k = 0; k < NL; k++) { const L = Math.max(60, Math.round(PRIMES[k] * size * (1 + .02 * (R.u() - .5)))); len[k] = L + 64; D.push(new Float32Array(len[k])); const s = L / SR; gM[k] = Math.pow(10, -3 * s / t60); gL[k] = Math.pow(10, -3 * s / (t60 * loMul)) / gM[k]; gH[k] = Math.pow(10, -3 * s / (t60 * hiMul)) / gM[k]; ph[k] = R.u() * TAU; dp[k] = TAU * R.r(.15, .9) / SR; ma[k] = mod * R.r(.5, 1); pos[k] = 0; }
  const cL = 1 - Math.exp(-TAU * fLo / SR), cH = 1 - Math.exp(-TAU * fHi / SR), outL = new Float32Array(n), outR = new Float32Array(n);
  // pre filter
  const iL = Float32Array.from(b.L), iR = Float32Array.from(b.R); for (const x of [iL, iR]) filt(x, hp ? bqc.hp(hp, .7) : null, lp < 22000 ? bqc.lp(lp, .6) : null);
  // early reflections
  const nE = 22, eT = [], eG = [], eP = []; for (let k = 0; k < nE; k++) { const u = (k + R.u()) / nE; eT.push(Math.floor((pre + .002 + u * u * erSpan) * SR)); eG.push(er * .5 * Math.exp(-2.6 * u) * (R.u() < .5 ? -1 : 1)); eP.push(R.r(-.9, .9)); }
  const preN = Math.floor(pre * SR), apN = [Math.floor(142 * Math.max(.4, size)), Math.floor(379 * Math.max(.4, size)), Math.floor(107 * Math.max(.4, size)), Math.floor(277 * Math.max(.4, size))], ap = [0, 1].map(c => apN.map((l, i) => ({ b: new Float32Array(l + (c ? 13 + i * 5 : 0)), p: 0 })));
  const v = new Float64Array(NL), SGN = [1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1];
  // last active sample: stop early when tail has died
  let last = 0; for (let i = n - 1; i >= 0; i--) if (iL[i] !== 0 || iR[i] !== 0) { last = i; break; } const stop = Math.min(n, last + Math.floor((t60 * Math.max(1, loMul) * 1.6 + .2) * SR)); let first = 0; while (first < n && iL[first] === 0 && iR[first] === 0) first++;
  for (let i = first; i < stop; i++) {
    const q = i - preN; let xl = q >= 0 ? iL[q] : 0, xr = q >= 0 ? iR[q] : 0;
    for (let c = 0; c < 2; c++) { let x = c ? xr : xl; const A = ap[c]; for (let k = 0; k < 4; k++) { const a = A[k], d = a.b[a.p], y = -diff * x + d; a.b[a.p] = x + diff * y; if (++a.p >= a.b.length) a.p = 0; x = y; } if (c) xr = x; else xl = x; }
    for (let k = 0; k < NL; k++) { const B = D[k], Ln = len[k]; ph[k] += dp[k]; let rp = pos[k] + 32 - 1 + ma[k] * Math.sin(ph[k]); let ii = Math.floor(rp); const fr = rp - ii; let p0 = ii % Ln, p1 = p0 + 1; if (p1 >= Ln) p1 = 0; let s = B[p0] * (1 - fr) + B[p1] * fr;
      zL[k] += cL * (s - zL[k]); zH[k] += cH * (s - zH[k]); const low = zL[k], hi = s - zH[k], mid = s - low - hi; v[k] = gM[k] * (low * gL[k] + mid + hi * gH[k]); }
    let ol = 0, or = 0; for (let k = 0; k < NL; k += 2) { ol += v[k] * SGN[k]; or += v[k + 1] * SGN[k + 1]; }
    // Hadamard 16
    for (let h = 8; h > 0; h >>= 1) for (let a = 0; a < 16; a += h << 1) for (let j = a; j < a + h; j++) { const x = v[j], y = v[j + h]; v[j] = x + y; v[j + h] = x - y; }
    for (let k = 0; k < NL; k++) { const B = D[k]; B[pos[k]] = v[k] * .25 + (k & 1 ? xr : xl) * .25 * SGN[15 - k] + 1e-20; if (++pos[k] >= len[k]) pos[k] = 0; }
    let e1 = 0, e2 = 0; if (er > 0) for (let k = 0; k < nE; k++) { const z = i - eT[k]; if (z >= 0) { const a = (iL[z] * (1 - eP[k]) + iR[z] * (1 + eP[k])) * eG[k]; e1 += a * (1 - eP[k]) * .5; e2 += a * (1 + eP[k]) * .5; } }
    const m = (ol + or) * .5, s = (ol - or) * .5 * width; outL[i] = g * (m + s + e1); outR[i] = g * (m - s + e2);
  }
  return { L: outL, R: outR };
}
export const room = (b, o = {}) => fdn(b, { t60: .42, hiMul: .5, loMul: .9, size: .34, pre: .004, er: 1.1, erSpan: .035, hp: 110, lp: 8200, mod: 2, seed: 'room', ...o });
export const plate = (b, o = {}) => fdn(b, { t60: 1.7, hiMul: .5, loMul: .85, size: .62, pre: .018, er: .05, hp: 220, lp: 10000, mod: 9, diff: .7, seed: 'plate', ...o });
export const hall = (b, o = {}) => fdn(b, { t60: 2.5, hiMul: .34, loMul: 1.0, size: 1.15, pre: .034, er: .45, erSpan: .085, hp: 230, lp: 6000, mod: 8, seed: 'hall', ...o });
export const cathedral = (b, o = {}) => fdn(b, { t60: 5.5, hiMul: .25, loMul: 1.2, size: 1.5, pre: .04, er: .25, erSpan: .11, hp: 110, lp: 6000, mod: 12, seed: 'cath', ...o });
export function delayFx(b, { t = .28125, fb = .3, lp = 3500, hp = 250, g = .3, ping = true } = {}) { const n = b.L.length, d = Math.round(t * SR), L = new Float32Array(n), R = new Float32Array(n); const a = 1 - Math.exp(-TAU * lp / SR), h = 1 - Math.exp(-TAU * hp / SR); let l1 = 0, l2 = 0, h1 = 0, h2 = 0;
  for (let i = d; i < n; i++) { const fl = L[i - d], fr = R[i - d]; let xl = (ping ? (b.L[i - d] + b.R[i - d]) * .5 : b.L[i - d]) + fb * (ping ? fr : fl), xr = (ping ? 0 : b.R[i - d]) + fb * (ping ? fl : fr); if (ping) { xr = fl; xl = (b.L[i - d] + b.R[i - d]) * .5 + fb * fr; } l1 += a * (xl - l1); l2 += a * (xr - l2); h1 += h * (l1 - h1); h2 += h * (l2 - h2); L[i] = l1 - h1; R[i] = l2 - h2; } return { L: gain(L, g), R: gain(R, g) }; }
export function chorusFx(b, { rate = .7, depth = .0022, base = .012, mix = .5 } = {}) { const n = b.L.length, o = st(n); for (let i = 0; i < n; i++) for (let c = 0; c < 2; c++) { const x = c ? b.R : b.L, y = c ? o.R : o.L; let s = x[i] * (1 - mix * .4); for (let v = 0; v < 2; v++) { const d = (base + depth * Math.sin(TAU * rate * (1 + v * .23) * i / SR + c * 1.6 + v * 2.2)) * SR, q = i - d; if (q > 1) { const ii = Math.floor(q), f = q - ii; s += mix * .5 * (x[ii] * (1 - f) + x[ii + 1] * f); } } y[i] = s; } return o; }
// ------------------------------------------------------------ loudness
export function lufs(b, a = 0, z = b.L.length) { const L = Float32Array.from(b.L.subarray(a, z)), R = Float32Array.from(b.R.subarray(a, z)); const c1 = [1.53512485958697, -2.69169618940638, 1.19839281085285, -1.69065929318241, .73248077421585], c2 = [1, -2, 1, -1.99004745483398, .99007225036621]; filt(L, c1, c2); filt(R, c1, c2); const w = Math.floor(.4 * SR), h = Math.floor(.1 * SR), bl = []; if (L.length < w) { let s = 0; for (let i = 0; i < L.length; i++) s += L[i] ** 2 + R[i] ** 2; return -.691 + 10 * Math.log10(s / L.length + 1e-12); } for (let p = 0; p + w <= L.length; p += h) { let s = 0; for (let i = p; i < p + w; i++) s += L[i] * L[i] + R[i] * R[i]; bl.push(s / w); } const l = x => -.691 + 10 * Math.log10(x + 1e-12); let g = bl.filter(x => l(x) > -70); if (!g.length) return -70; let m = g.reduce((p, c) => p + c, 0) / g.length; g = g.filter(x => l(x) > l(m) - 10); m = g.reduce((p, c) => p + c, 0) / g.length; return l(m); }
export function truePeak(b) { let p = 0; for (const x of [b.L, b.R]) { for (let i = 3; i < x.length - 4; i++) { const a0 = Math.abs(x[i]); if (a0 > p) p = a0; if (a0 > p * .7) for (const ph of [.25, .5, .75]) { let s = 0; for (let j = -2; j <= 3; j++) { let w = 1; for (let k = -2; k <= 3; k++) if (k !== j) w *= (ph - k) / (j - k); s += x[i + j] * w; } const a = Math.abs(s); if (a > p) p = a; } } } return todb(p); }
// ------------------------------------------------------------ wav
export function writeWav(file, b) { const L = b.L || b, R = b.R || b, n = L.length, buf = Buffer.alloc(58 + n * 8); buf.write('RIFF', 0); buf.writeUInt32LE(50 + n * 8, 4); buf.write('WAVEfmt ', 8); buf.writeUInt32LE(18, 16); buf.writeUInt16LE(3, 20); buf.writeUInt16LE(2, 22); buf.writeUInt32LE(SR, 24); buf.writeUInt32LE(SR * 8, 28); buf.writeUInt16LE(8, 32); buf.writeUInt16LE(32, 34); buf.writeUInt16LE(0, 36); buf.write('fact', 38); buf.writeUInt32LE(4, 42); buf.writeUInt32LE(n, 46); buf.write('data', 50); buf.writeUInt32LE(n * 8, 54); const f = new Float32Array(n * 2); for (let i = 0; i < n; i++) { f[2 * i] = L[i]; f[2 * i + 1] = R[i]; } Buffer.from(f.buffer).copy(buf, 58); fs.mkdirSync(path.dirname(file), { recursive: true }); fs.writeFileSync(file, buf); }
export function writeWav24(file, b) { const { L, R } = b, n = L.length, buf = Buffer.alloc(44 + n * 6); buf.write('RIFF', 0); buf.writeUInt32LE(36 + n * 6, 4); buf.write('WAVEfmt ', 8); buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(2, 22); buf.writeUInt32LE(SR, 24); buf.writeUInt32LE(SR * 6, 28); buf.writeUInt16LE(6, 32); buf.writeUInt16LE(24, 34); buf.write('data', 36); buf.writeUInt32LE(n * 6, 40); let o = 44; for (let i = 0; i < n; i++) { for (const x of [L, R]) { const v = Math.round(clamp(x[i], -1, 1) * 8388607); buf.writeIntLE(v, o, 3); o += 3; } } fs.mkdirSync(path.dirname(file), { recursive: true }); fs.writeFileSync(file, buf); }
export function readWav(file) { const b = fs.readFileSync(file); let p = 12, fmt, ch, bits, sr; while (p < b.length) { const id = b.toString('ascii', p, p + 4), sz = b.readUInt32LE(p + 4); if (id === 'fmt ') { fmt = b.readUInt16LE(p + 8); ch = b.readUInt16LE(p + 10); sr = b.readUInt32LE(p + 12); bits = b.readUInt16LE(p + 22); } if (id === 'data') { const bytes = bits / 8, n = Math.floor(sz / bytes / ch), o = st(n); for (let i = 0; i < n; i++) for (let c = 0; c < 2; c++) { const q = p + 8 + (i * ch + Math.min(c, ch - 1)) * bytes; const v = fmt === 3 ? b.readFloatLE(q) : bits === 24 ? b.readIntLE(q, 3) / 8388608 : b.readInt16LE(q) / 32768; (c ? o.R : o.L)[i] = v; } o.sr = sr; return o; } p += 8 + sz + (sz & 1); } throw new Error('no data'); }
// ------------------------------------------------------------ modal bank
// modes: [{f,a,t60,ph?,rise?}] adds into out
export function modal(out, modes, at = 0, g = 1) {
  const n = out.length;
  for (const m of modes) { if (m.f > SR * .45 || m.a === 0) continue; const parts = m.rise ? [[m.a, m.t60], [-m.a, 1 / (1 / m.t60 + 6.9078 / (m.rise * 6.9078))]] : [[m.a, m.t60]];
    for (const [a, t60] of parts) { const r = Math.pow(10, -3 / (t60 * SR)), w = TAU * m.f / SR, k = 2 * r * Math.cos(w), r2 = r * r, ph = m.ph || 0; let y1 = a * g * Math.sin(ph - w) / r, y0 = a * g * Math.sin(ph); /* y[n]=r^n sin(ph+wn) */ let p2 = a * g * Math.sin(ph - 2 * w) / r2, p1 = y1; const L = Math.min(n - at, Math.ceil(t60 * 1.6 * SR)); for (let i = 0; i < L; i++) { const y = k * p1 - r2 * p2; p2 = p1; p1 = y; out[at + i] += y; } } }
  return out;
}
export function noise(n, R) { const x = new Float32Array(n); for (let i = 0; i < n; i++) x[i] = R.u() * 2 - 1; return x; }
export function expEnv(x, tau, att = 0) { const a = Math.exp(-1 / (tau * SR)); let e = 1; const A = Math.floor(att * SR); for (let i = 0; i < x.length; i++) { x[i] *= e * (i < A ? i / A : 1); if (i >= A) e *= a; } return x; }
// cache for synthesised one-shots
const CDIR = path.join(ROOT, 'cache', 'band');
export function cached(key, fn) { const f = path.join(CDIR, key + '.f32'); if (fs.existsSync(f)) { const b = fs.readFileSync(f); const hdr = b.readUInt32LE(0); const all = new Float32Array(b.buffer.slice(b.byteOffset + 4, b.byteOffset + b.length)); if (hdr === 1) return all; const h = all.length / 2; return { L: all.slice(0, h), R: all.slice(h) }; }
  const r = fn(); fs.mkdirSync(CDIR, { recursive: true }); const isS = !!r.L; const total = isS ? r.L.length * 2 : r.length, b = Buffer.alloc(4 + total * 4); b.writeUInt32LE(isS ? 2 : 1, 0); const f32 = new Float32Array(total); if (isS) { f32.set(r.L, 0); f32.set(r.R, r.L.length); } else f32.set(r, 0); Buffer.from(f32.buffer).copy(b, 4); const tmp = f + '.' + process.pid + Math.random().toString(36).slice(2); fs.writeFileSync(tmp, b); fs.renameSync(tmp, f); return r; }
// body impulse response: dense random modes under a target envelope (dB points), used for guitar/violin/koto bodies
export function bodyIR(seed, { len = .12, f0 = 80, f1 = 8000, n = 160, env = [[80, 0], [8000, -20]], t0 = .09, t1 = .006, strong = [], direct = .0 } = {}) {
  const R = rng('body' + seed), out = new Float32Array(Math.floor(len * SR)), modes = []; const lg = Math.log(f1 / f0);
  for (let k = 0; k < n; k++) { const u = (k + R.u()) / n, f = f0 * Math.exp(lg * u), g = envAt(env.map(e => [Math.log(e[0]), e[1]]), Math.log(f)); const t60 = t0 * Math.pow(t1 / t0, u) * R.r(.6, 1.5); modes.push({ f, a: db(g) * Math.sqrt(f / 500) * R.r(.35, 1) * (R.u() < .5 ? -1 : 1) / Math.sqrt(t60 * 40), t60: t60 * 1.0, ph: R.r(-.4, .4) }); }
  for (const s of strong) modes.push({ f: s[0], a: db(s[1]), t60: s[2], ph: 0 }); modal(out, modes); fade(out, .0002, len * .3); let e = 0; for (const v of out) e += v * v; gain(out, 1 / Math.sqrt(e * (1 + direct * direct))); out[0] += direct / Math.sqrt(1 + direct * direct); return out; }

// bowed-instrument body: sparse, high-Q modes (Q 22..60) so that vibrato makes every partial flicker the way wood does
export function fiddleIR(seed, { scale = 1, len = .16, hill = 2500, hillDb = 3, top = 9000, low = [[275, 9, 12], [400, 3, 16], [462, 10, 18], [540, 10, 18]], tilt = 7, direct = .35 } = {}) {
  const R = rng('fid' + seed), out = new Float32Array(Math.floor(len * SR)), md = []; for (const [f, d, q] of low) md.push({ f: f * scale * R.r(.985, 1.015), d, q });
  let f = 610 * scale; while (f < top) { const x = f / scale; const base = tilt * clamp(1 - Math.log2(Math.max(1, x / 900)) / 1.6, 0, 1); let dev = 4.6 * R.g(); if (f > 2200) dev = Math.min(dev, .5) - 2.5 * Math.exp(-Math.pow(Math.log(f / 3250) / .3, 2)); const g = base + hillDb * Math.exp(-Math.pow(Math.log(x / hill) / .42, 2)) - Math.max(0, Math.log2(x / (hill * 1.5))) * 11 + dev; md.push({ f, d: g, q: 24 + 34 * Math.min(1, x / 5000) * R.r(.7, 1.3) }); f += scale * (x < 1200 ? R.r(55, 100) : R.r(38, 85)); }
  modal(out, md.map(m => { const t60 = 2.2 * m.q / m.f; return { f: m.f, t60, a: db(m.d) * m.f / 1000 / m.q * 12 * (R.u() < .5 ? -1 : 1) * R.r(.75, 1.25), ph: R.r(-.35, .35) }; })); fade(out, .0001, len * .35); let e = 0; for (const v of out) e += v * v; e = Math.sqrt(e); for (let i = 0; i < out.length; i++) out[i] /= e; out[0] += direct; return out; }
