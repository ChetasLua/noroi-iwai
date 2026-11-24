// DSP kernel of the sound department.  Plain JavaScript, Node 25, no packages, no samples.
// Everything is 48 kHz double precision.  A mono signal is a Float64Array, a stereo one is an `St` ({l, r}).
// House rules (the client's ears): noise is always band-limited, nothing is saturated, energy above 8 kHz is restrained.
export const SR = 48000, TAU = Math.PI * 2;
export const N = s => Math.max(1, Math.round(s * SR));
export const buf = s => new Float64Array(N(s));
export const clamp = (x, a = 0, b = 1) => x < a ? a : x > b ? b : x;
export const lerp = (a, b, t) => a + (b - a) * t;
export const db = d => Math.pow(10, d / 20);
export const todb = x => 20 * Math.log10(Math.max(x, 1e-12));
export const mtof = m => 440 * Math.pow(2, (m - 69) / 12);
export const smooth = x => { x = clamp(x); return x * x * (3 - 2 * x); };

// ---------------------------------------------------------------- random
export function rng(seed = 1) {
  let a = (Math.imul((seed * 7919 + 17) | 0, 2654435761) >>> 0) || 1;
  const r = () => { a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  for (let i = 0; i < 8; i++) r();
  r.u = (lo, hi) => lo + (hi - lo) * r();
  r.log = (lo, hi) => lo * Math.pow(hi / lo, r());
  r.g = () => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(TAU * r());
  r.exp = m => -m * Math.log(1 - r());
  r.pick = arr => arr[Math.floor(r() * arr.length)];
  r.sgn = () => r() < .5 ? -1 : 1;
  r.i = (lo, hi) => Math.floor(r.u(lo, hi + 1));
  return r;
}
export function noise(r, n) { const x = new Float64Array(n); for (let i = 0; i < n; i++) x[i] = r() * 2 - 1; return x; }
export function pink(r, n) {            // Paul Kellet's refined filter: -3 dB/oct
  const x = new Float64Array(n); let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
  for (let i = 0; i < n; i++) {
    const w = r() * 2 - 1;
    b0 = .99886 * b0 + w * .0555179; b1 = .99332 * b1 + w * .0750759; b2 = .969 * b2 + w * .153852; b3 = .8665 * b3 + w * .3104856; b4 = .55 * b4 + w * .5329522; b5 = -.7616 * b5 - w * .016898;
    x[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * .5362) * .18; b6 = w * .115926;
  }
  return x;
}
export function brown(r, n) { const x = new Float64Array(n); let y = 0; for (let i = 0; i < n; i++) { y = y * .9985 + (r() * 2 - 1) * .055; x[i] = y; } return x; }
// slow random curve in [-1,1]: smooth interpolation of values at `rate` Hz
export function wander(r, n, rate) {
  const x = new Float64Array(n), step = SR / rate; let a = r.u(-1, 1), b = r.u(-1, 1), p = r();
  for (let i = 0; i < n; i++) { x[i] = a + (b - a) * p * p * (3 - 2 * p); p += 1 / step; if (p >= 1) { p -= 1; a = b; b = r.u(-1, 1); } }
  return x;
}

// ---------------------------------------------------------------- envelopes
// pts: [[t, v, curve], ...]  curve of the segment that ENDS at this point: undefined linear, 'e' exponential, 's' smooth, number = power
export function env(n, pts) {
  const x = new Float64Array(n); let k = 0;
  for (let i = 0; i < n; i++) {
    const t = i / SR; while (k < pts.length - 1 && t >= pts[k + 1][0]) k++;
    const A = pts[k], B = pts[k + 1];
    if (t < A[0] || !B) { x[i] = t < A[0] ? pts[0][1] : pts[pts.length - 1][1]; continue; }
    const p = (t - A[0]) / Math.max(1e-9, B[0] - A[0]), c = B[2];
    x[i] = c === 'e' ? A[1] * Math.pow(Math.max(B[1], 1e-5) / Math.max(A[1], 1e-5), p) : c === 's' ? lerp(A[1], B[1], smooth(p)) : typeof c === 'number' ? lerp(A[1], B[1], Math.pow(p, c)) : lerp(A[1], B[1], p);
  }
  return x;
}
export const curve = pts => { const f = t => { let k = 0; while (k < pts.length - 1 && t >= pts[k + 1][0]) k++; const A = pts[k], B = pts[k + 1]; if (t <= pts[0][0]) return pts[0][1]; if (!B) return A[1];
  const p = (t - A[0]) / Math.max(1e-9, B[0] - A[0]), c = B[2]; return c === 'e' ? A[1] * Math.pow(Math.max(B[1], 1e-5) / Math.max(A[1], 1e-5), p) : c === 's' ? lerp(A[1], B[1], smooth(p)) : typeof c === 'number' ? lerp(A[1], B[1], Math.pow(p, c)) : lerp(A[1], B[1], p); }; return f; };
export const mul = (x, e) => { for (let i = 0; i < x.length; i++) x[i] *= typeof e === 'number' ? e : typeof e === 'function' ? e(i / SR) : (e[i] ?? 0); return x; };
export const mix = (y, x, g = 1, at = 0) => { const o = Math.round(at * SR); for (let i = Math.max(0, -o); i < x.length && i + o < y.length; i++) y[i + o] += x[i] * g; return y; };
export const peakOf = x => { let m = 0; for (let i = 0; i < x.length; i++) { const a = Math.abs(x[i]); if (a > m) m = a; } return m; };
export const rms = x => { let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * x[i]; return Math.sqrt(s / x.length); };
export const unit = (x, p = 1) => { const m = peakOf(x); if (m > 0) for (let i = 0; i < x.length; i++) x[i] *= p / m; return x; };
export const unitr = (x, p = .2) => { const m = rms(x); if (m > 0) for (let i = 0; i < x.length; i++) x[i] *= p / m; return x; };
export const reverse = x => { const y = new Float64Array(x.length); for (let i = 0; i < x.length; i++) y[i] = x[x.length - 1 - i]; return y; };
export const fadeEnds = (x, a = .004, b = .02) => { const na = N(a), nb = N(b), n = x.length; for (let i = 0; i < na && i < n; i++) x[i] *= .5 - .5 * Math.cos(Math.PI * i / na); for (let i = 0; i < nb && i < n; i++) x[n - 1 - i] *= .5 - .5 * Math.cos(Math.PI * i / nb); return x; };

// ---------------------------------------------------------------- filters (RBJ biquads, TDF2; in place, return x)
export function coef(type, f, q = .7071, g = 0) {
  f = clamp(f, 5, SR * .48); const w = TAU * f / SR, c = Math.cos(w), s = Math.sin(w), al = s / (2 * q), A = Math.pow(10, g / 40); let b0, b1, b2, a0, a1, a2;
  if (type === 'lp') { b0 = (1 - c) / 2; b1 = 1 - c; b2 = b0; a0 = 1 + al; a1 = -2 * c; a2 = 1 - al; }
  else if (type === 'hp') { b0 = (1 + c) / 2; b1 = -(1 + c); b2 = b0; a0 = 1 + al; a1 = -2 * c; a2 = 1 - al; }
  else if (type === 'bp') { b0 = al; b1 = 0; b2 = -al; a0 = 1 + al; a1 = -2 * c; a2 = 1 - al; }
  else if (type === 'notch') { b0 = 1; b1 = -2 * c; b2 = 1; a0 = 1 + al; a1 = -2 * c; a2 = 1 - al; }
  else if (type === 'ap') { b0 = 1 - al; b1 = -2 * c; b2 = 1 + al; a0 = 1 + al; a1 = -2 * c; a2 = 1 - al; }
  else if (type === 'pk') { b0 = 1 + al * A; b1 = -2 * c; b2 = 1 - al * A; a0 = 1 + al / A; a1 = -2 * c; a2 = 1 - al / A; }
  else { const sq = 2 * Math.sqrt(A) * al, sg = type === 'ls' ? 1 : -1;
    b0 = A * ((A + 1) - sg * (A - 1) * c + sq); b1 = sg * 2 * A * ((A - 1) - sg * (A + 1) * c); b2 = A * ((A + 1) - sg * (A - 1) * c - sq);
    a0 = (A + 1) + sg * (A - 1) * c + sq; a1 = -sg * 2 * ((A - 1) + sg * (A + 1) * c); a2 = (A + 1) + sg * (A - 1) * c - sq; }
  return [b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0];
}
export function bq(x, k) { let z1 = 0, z2 = 0; const [b0, b1, b2, a1, a2] = k; for (let i = 0; i < x.length; i++) { const v = x[i], y = b0 * v + z1; z1 = b1 * v - a1 * y + z2; z2 = b2 * v - a2 * y; x[i] = y; } return x; }
export const lp = (x, f, q) => bq(x, coef('lp', f, q)), hp = (x, f, q) => bq(x, coef('hp', f, q)), bp = (x, f, q = 1) => bq(x, coef('bp', f, q));
export const lp4 = (x, f) => lp(lp(x, f, .5412), f, 1.3066), hp4 = (x, f) => hp(hp(x, f, .5412), f, 1.3066);
export const pk = (x, f, q, g) => bq(x, coef('pk', f, q, g)), ls = (x, f, g) => bq(x, coef('ls', f, .7071, g)), hs = (x, f, g) => bq(x, coef('hs', f, .7071, g));
export const ap = (x, f, q = .7) => bq(x, coef('ap', f, q));
// time-varying state-variable filter (TPT).  f: function(t) or number.  mode lp | bp | hp
export function svf(x, f, q = 1, mode = 'bp') {
  let a = 0, b = 0; const k = 1 / q; let g = 0, a1 = 0, a2 = 0, a3 = 0; const fn = typeof f === 'function' ? f : () => f;
  for (let i = 0; i < x.length; i++) {
    if ((i & 15) === 0) { g = Math.tan(Math.PI * clamp(fn(i / SR), 8, SR * .45) / SR); a1 = 1 / (1 + g * (g + k)); a2 = g * a1; a3 = g * a2; }
    const v3 = x[i] - b, v1 = a1 * a + a2 * v3, v2 = b + a2 * a + a3 * v3; a = 2 * v1 - a; b = 2 * v2 - b;
    x[i] = mode === 'lp' ? v2 : mode === 'bp' ? k * v1 : x[i] - k * v1 - v2;
  }
  return x;
}
export const band = (r, n, lo, hi, kind = noise) => lp4(hp4(kind(r, n), lo), hi);      // band-limited noise, 24 dB/oct skirts

// ---------------------------------------------------------------- physics
// contact force: raised cosine, unit area (a hard contact is short, so bright).  tc = nominal contact time; the effective pulse is half of it
export function pulse(y, at, tc, a = 1) { const n = Math.max(2, Math.round(tc * .5 * SR)), o = Math.round(at * SR); for (let i = 0; i < n; i++) { const k = o + i; if (k >= 0 && k < y.length) y[k] += a * (1 - Math.cos(TAU * (i + .5) / n)) / n; } return y; }
// resonator bank: modes [[f, t60, a], ...] driven by force x
export function bank(x, modes, n = x.length) {
  const y = new Float64Array(n);
  for (const [f, t60, a] of modes) {
    if (f > 16000 || f < 8) continue; const r = Math.pow(10, -3 / (t60 * SR)), w = TAU * f / SR, c1 = 2 * r * Math.cos(w), c2 = -r * r, b = Math.sin(w) * a; let y1 = 0, y2 = 0;
    for (let i = 0; i < n; i++) { const v = b * (i < x.length ? x[i] : 0) + c1 * y1 + c2 * y2; y2 = y1; y1 = v; y[i] += v; }
  }
  return y;
}
export function hit(dur, tc, modes, a = 1) { const x = buf(dur); pulse(x, 0, tc, a); return bank(x, modes); }
// one damped sinusoid with a soft attack and optional pitch bend (per second, exponential)
export function ping(y, at, f, t60, a, bend = 0, atk = .0006) {
  const o = Math.round(at * SR), n = Math.min(Math.round(t60 * 1.15 * SR), y.length - o); if (n <= 0) return y; const d = Math.pow(10, -3 / (t60 * SR)), na = Math.max(1, atk * SR); let e = a, ph = 0, inc = TAU * f / SR; const bf = bend ? Math.exp(bend / SR) : 1;
  for (let i = 0; i < n; i++) { const k = o + i; const at_ = i < na ? .5 - .5 * Math.cos(Math.PI * i / na) : 1; if (k >= 0) y[k] += e * at_ * Math.sin(ph); ph += inc; inc *= bf; e *= d; if (inc > 2.2) break; }
  return y;
}
// air bubble of radius mm (Minnaert; van den Doel's damping), pitch rises as it nears the surface
export function bubble(y, at, mm, a, xi = .1) { const f = 3260 / mm, rr = mm / 1000, d = .13 / rr + .0072 * Math.pow(rr, -1.5); return ping(y, at, f, .7 * 6.9 / d, a, Math.min(5, xi * d * .4), .25 / f); }
export const bubbleF = mm => 3260 / mm;
// poisson event times with rate(t) per second
export function poisson(r, dur, rate, max = 1e9) { const out = [], fn = typeof rate === 'function' ? rate : () => rate; let mx = 1; for (let i = 0; i <= 200; i++) mx = Math.max(mx, fn(dur * i / 200)); let t = 0; while (out.length < max) { t += r.exp(1 / mx); if (t >= dur) break; if (r() < fn(t) / mx) out.push(t); } return out; }
// stick-slip force: pulses at rate(t) with jitter; strength follows the waiting time
export function slip(r, dur, rate, { jit = .35, tc = .0007, amp = 1 } = {}) { const y = buf(dur), fn = typeof rate === 'function' ? rate : () => rate, am = typeof amp === 'function' ? amp : () => amp; let t = 0; while (t < dur) { const f = Math.max(2, fn(t)), w = (1 / f) * Math.max(.15, 1 + jit * r.g()); t += w; if (t < dur) pulse(y, t, tc * r.u(.7, 1.4), am(t) * w * f * r.sgn() * r.u(.6, 1.2)); } return y; }
// turbulent air: band of noise whose centre and loudness follow speed u(t) in 0..1 (aeolian: f ~ U/d, p ~ U^n)
export function air(r, dur, u, { f = 900, q = 1.2, pw = 2.4, lo = 60, hi = 7000, src = pink } = {}) { const n = N(dur), x = src(r, n); svf(x, t => Math.max(lo, f * u(t)), q, 'bp'); for (let i = 0; i < n; i++) x[i] *= Math.pow(clamp(u(i / SR), 0, 4), pw); return lp4(hp(x, lo), hi); }
// glottal source: KLGLOTT88 derivative, f0(t) Hz, oq open quotient, with jitter, subharmonic growl and breath
export function glottis(r, dur, f0, { oq = .6, jit = .01, shim = .05, growl = 0, breath = .05, tilt = 3000, sub = 2 } = {}) {
  const n = N(dur), y = new Float64Array(n), fn = typeof f0 === 'function' ? f0 : () => f0, gw = typeof growl === 'function' ? growl : () => growl, br = typeof breath === 'function' ? breath : () => breath; let ph = 0, T = 1, amp = 1, k = 0, j = 1;
  const bn = lp(hp(noise(r, n), 400), 6000);
  for (let i = 0; i < n; i++) {
    const t = i / SR; if (ph >= 1) { ph -= 1; k++; j = 1 + jit * r.g(); amp = (1 + shim * r.g()) * (k % sub === 0 ? 1 : 1 - .75 * gw(t)); }
    const o = oq, p = ph / o; let v = 0; if (p < 1) v = (2 * p - 3 * p * p) * 1.5;
    y[i] = v * amp + bn[i] * br(t) * (p < 1 ? 1 : .35); ph += fn(t) * j / SR;
  }
  return lp(y, tilt);
}
// parallel formants; each [f(t)|number, bandwidth, gain]
export function formants(x, list) { const y = new Float64Array(x.length); for (const [f, bw, g] of list) { const c = typeof f === 'function' ? f(.1) : f; mix(y, svf(Float64Array.from(x), f, c / bw, 'bp'), g); } return y; }
// karplus-like plucked string with lowpass loss
export function pluck(r, dur, f, { bright = .5, decay = .996, pos = .2 } = {}) { const n = N(dur), L = SR / f, M = Math.ceil(L) + 2, d = new Float64Array(M), y = new Float64Array(n), ex = lp(noise(r, M), 600 + 6000 * bright); for (let i = 0; i < M; i++) d[i] = ex[i] - (i - Math.round(pos * M) >= 0 ? ex[i - Math.round(pos * M)] : 0); let w = 0, last = 0; for (let i = 0; i < n; i++) { const rp = (w - L + 2 * M) % M, a = Math.floor(rp), fr = rp - a, v = d[a % M] * (1 - fr) + d[(a + 1) % M] * fr; last = last + (v - last) * (.25 + .7 * bright); d[w] = last * decay; y[i] = v; w = (w + 1) % M; } return hp(y, 30); }
// varispeed read of x with speed(t)
export function vari(x, dur, speed) { const n = N(dur), y = new Float64Array(n); let p = 0; for (let i = 0; i < n; i++) { const k = Math.floor(p); if (k + 1 >= x.length) break; y[i] = x[k] + (x[k + 1] - x[k]) * (p - k); p += Math.max(0, speed(i / SR)); } return y; }

// ---------------------------------------------------------------- stereo
export class St {
  constructor(dur, A = Float64Array) { this.n = N(dur); this.l = new A(this.n); this.r = new A(this.n); }      // long buses use Float32Array (memory)
  add(x, { at = 0, g = 1, pan = 0 } = {}) {       // pan -1..1 constant power with a small interaural delay; pan may be a function of t
    const o = Math.round(at * SR);
    if (typeof pan === 'function') { for (let i = Math.max(0, -o); i < x.length && i + o < this.n; i++) { const a = (clamp(pan(i / SR), -1, 1) + 1) * Math.PI / 4; this.l[i + o] += x[i] * g * Math.cos(a) * 1.414; this.r[i + o] += x[i] * g * Math.sin(a) * 1.414; } return this; }
    const a = (clamp(pan, -1, 1) + 1) * Math.PI / 4, gl = Math.cos(a) * 1.414 * g, gr = Math.sin(a) * 1.414 * g, itd = Math.round(Math.abs(pan) * 16), ol = pan > 0 ? o + itd : o, or = pan < 0 ? o + itd : o;
    for (let i = 0; i < x.length; i++) { const kl = i + ol, kr = i + or; if (kl >= 0 && kl < this.n) this.l[kl] += x[i] * gl; if (kr >= 0 && kr < this.n) this.r[kr] += x[i] * gr; }
    return this;
  }
  addSt(s, { at = 0, g = 1 } = {}) { mix(this.l, s.l, g, at); mix(this.r, s.r, g, at); return this; }
  wide(fn, { at = 0, g = 1 } = {}) { mix(this.l, fn(0), g, at); mix(this.r, fn(1), g, at); return this; }       // two decorrelated takes
  each(fn) { fn(this.l); fn(this.r); return this; }
  mulEnv(e) { mul(this.l, e); mul(this.r, e); return this; }
  ping(at, f, t60, a, pan = 0, bend = 0, atk = .0006) { const g = (clamp(pan, -1, 1) + 1) * Math.PI / 4; ping(this.l, at, f, t60, a * Math.cos(g) * 1.414, bend, atk); ping(this.r, at + pan * .0002, f, t60, a * Math.sin(g) * 1.414, bend, atk); return this; }
  bubble(at, mm, a, pan = 0, xi = .1) { const g = (clamp(pan, -1, 1) + 1) * Math.PI / 4; bubble(this.l, at, mm, a * Math.cos(g) * 1.414, xi); bubble(this.r, at, mm, a * Math.sin(g) * 1.414, xi); return this; }
  peak() { return Math.max(peakOf(this.l), peakOf(this.r)); }
  gain(g) { mul(this.l, g); mul(this.r, g); return this; }
  fade(a = .002, b = .03) { fadeEnds(this.l, a, b); fadeEnds(this.r, a, b); return this; }
  mono() { const x = new Float64Array(this.n); for (let i = 0; i < this.n; i++) x[i] = (this.l[i] + this.r[i]) * .5; return x; }
  trim(thr = 1e-5) { let e = this.n - 1; while (e > 100 && Math.abs(this.l[e]) < thr && Math.abs(this.r[e]) < thr) e--; e = Math.min(this.n, e + 480); this.l = this.l.slice(0, e); this.r = this.r.slice(0, e); this.n = e; return this; }
}

// ---------------------------------------------------------------- FFT, convolution, rooms
const TW = new Map(); const tw = n => { let t = TW.get(n); if (!t) { const h = n >> 1, c = new Float64Array(h), q = new Float64Array(h); for (let k = 0; k < h; k++) { c[k] = Math.cos(TAU * k / n); q[k] = Math.sin(TAU * k / n); } TW.set(n, t = [c, q]); } return t; };
export function fft(re, im, inv = false) {
  const n = re.length; for (let i = 1, j = 0; i < n; i++) { let b = n >> 1; for (; j & b; b >>= 1) j ^= b; j ^= b; if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
  const [C, Q] = tw(n), sg = inv ? 1 : -1; for (let len = 2; len <= n; len <<= 1) { const h = len >> 1, st = n / len; for (let i = 0; i < n; i += len) for (let k = 0, q = 0; k < h; k++, q += st) { const a = i + k, b = a + h, cs = C[q], sn = sg * Q[q], tr = re[b] * cs - im[b] * sn, ti = re[b] * sn + im[b] * cs; re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti; } }
  if (inv) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}
const KC = new Map();        // at most two kernel spectra are kept (memory)
export function convolvePair(xl, xr, hl, hr) {       // two real convolutions at once (left in the real part, right in the imaginary part), overlap-add
  let K = KC.get(hl); if (!K) { let e = hl.length - 1, m = 0; for (let i = 0; i < hl.length; i++) m = Math.max(m, Math.abs(hl[i]), Math.abs(hr[i])); while (e > 64 && Math.abs(hl[e]) < m * 3e-4 && Math.abs(hr[e]) < m * 3e-4) e--; const L = e + 1; let n = 1024; while (n < 2 * L) n <<= 1; const mk = h => { const R = new Float64Array(n), I = new Float64Array(n); R.set(h.subarray(0, L)); fft(R, I); return [Float32Array.from(R), Float32Array.from(I)]; }; K = { L, n, l: mk(hl), r: mk(hr) }; if (KC.size >= 2) KC.delete(KC.keys().next().value); KC.set(hl, K); }
  const { L, n, l: [HLr, HLi], r: [HRr, HRi] } = K, B = n - L + 1, yl = new Float64Array(xl.length + L - 1), yr = new Float64Array(xl.length + L - 1);
  for (let o = 0; o < xl.length; o += B) { const R = new Float64Array(n), I = new Float64Array(n); let any = false; for (let i = 0; i < B && o + i < xl.length; i++) { R[i] = xl[o + i]; I[i] = xr[o + i]; if (R[i] || I[i]) any = true; } if (!any) continue; fft(R, I); const OR = new Float64Array(n), OI = new Float64Array(n);
    for (let k = 0; k < n; k++) { const j = (n - k) & (n - 1), Ar = (R[k] + R[j]) * .5, Ai = (I[k] - I[j]) * .5, Br = (I[k] + I[j]) * .5, Bi = (R[j] - R[k]) * .5, pr = Ar * HLr[k] - Ai * HLi[k], pi = Ar * HLi[k] + Ai * HLr[k], qr = Br * HRr[k] - Bi * HRi[k], qi = Br * HRi[k] + Bi * HRr[k]; OR[k] = pr - qi; OI[k] = pi + qr; }
    fft(OR, OI, true); for (let i = 0; i < n && o + i < yl.length; i++) { yl[o + i] += OR[i]; yr[o + i] += OI[i]; } }
  return { l: yl, r: yr };
}
export function convolve(x, h) {        // overlap-add with a block about the length of h
  const L = h.length; let n = 1024; while (n < 2 * L) n <<= 1; const B = n - L + 1, Hr = new Float64Array(n), Hi = new Float64Array(n); Hr.set(h); fft(Hr, Hi); const y = new Float64Array(x.length + L - 1);
  for (let o = 0; o < x.length; o += B) { const R = new Float64Array(n), I = new Float64Array(n); let any = false; for (let i = 0; i < B && o + i < x.length; i++) { const v = x[o + i]; R[i] = v; if (v) any = true; } if (!any) continue; fft(R, I); for (let i = 0; i < n; i++) { const a = R[i] * Hr[i] - I[i] * Hi[i]; I[i] = R[i] * Hi[i] + I[i] * Hr[i]; R[i] = a; } fft(R, I, true); for (let i = 0; i < n && o + i < y.length; i++) y[o + i] += R[i]; }
  return y;
}
// synthetic room: early reflections from explicit [ms, gain, pan, lp] + diffuse tail with a decay time per octave band
export function room({ t60 = 1, seed = 5, pre = 0, early = [], tail = 1, bands = {}, build = .01, hicut = 7000, locut = 60, width = 1, mod = 0 }) {
  const r = rng(seed), cen = [63, 125, 250, 500, 1000, 2000, 4000, 8000], T = cen.map(f => t60 * (bands[f] ?? 1)), dur = Math.min(9, Math.max(...T) * 1.05 + pre + .05), n = N(dur), out = [new Float64Array(n), new Float64Array(n)], common = noise(r, n);
  for (let c = 0; c < 2; c++) {
    const w = noise(r, n); cen.forEach((f, k) => {
      const x = new Float64Array(n), co = f < 200 ? .85 : (1 - width) * .7; for (let i = 0; i < n; i++) x[i] = w[i] * Math.sqrt(1 - co * co) + common[i] * co;
      bp(x, f, 1.2); const d = Math.pow(10, -3 / (T[k] * SR)), np = N(pre), nb = Math.max(1, N(build)); let e = tail * Math.sqrt(1 / (f + 40)) * 1.1; e = tail * .5;
      for (let i = 0; i < n; i++) { if (i < np) { x[i] = 0; continue; } const b = i - np < nb ? (i - np) / nb : 1; x[i] *= e * b; e *= d; } mix(out[c], x, 1);
    });
    let s = 0; for (const v of out[c]) s += v * v; const g = tail ? tail / Math.sqrt(s) : 0; mul(out[c], g);
  }
  for (const [ms, g, pan = 0, cut = 6000] of early) { const len = N(.004), x = new Float64Array(len); x[0] = 1; lp(x, cut); const a = (pan + 1) * Math.PI / 4; mix(out[0], x, g * Math.cos(a) * 1.2, ms / 1000); mix(out[1], x, g * Math.sin(a) * 1.2, ms / 1000); }
  for (const o of out) { hp(o, locut); lp(o, hicut); }
  return { l: out[0], r: out[1] };
}

// ---------------------------------------------------------------- meters
// BS.1770-4 integrated + max momentary + max 100 ms + short-term, streaming (no copies of the signal)
export function loud(l, r) {
  const hop = N(.1), n = l.length, e = []; const ch = [l, r].map(x => ({ x, a1: 0, a2: 0, b1: 0, b2: 0 }));
  for (let o = 0; o < n; o += hop) { let s = 0; const end = Math.min(n, o + hop); for (const c of ch) { const x = c.x; let { a1, a2, b1, b2 } = c; for (let i = o; i < end; i++) { const v = x[i], y = 1.53512485958697 * v + a1; a1 = -2.69169620938640 * v + 1.69065929318241 * y + a2; a2 = 1.19839281085285 * v - .73248077421585 * y; const z = y + b1; b1 = -2 * y + 1.99004745483398 * z + b2; b2 = y - .99007225036621 * z; s += z * z; } c.a1 = a1; c.a2 = a2; c.b1 = b1; c.b2 = b2; } e.push(s / hop); }
  const L = p => -0.691 + 10 * Math.log10(Math.max(p, 1e-14)), mom = [], st = []; for (let k = 0; k + 4 <= e.length; k++) mom.push((e[k] + e[k + 1] + e[k + 2] + e[k + 3]) / 4); for (let k = 0; k + 30 <= e.length; k++) { let s = 0; for (let j = 0; j < 30; j++) s += e[k + j]; st.push(s / 30); }
  const g1 = mom.filter(p => L(p) > -70), m1 = g1.reduce((x, y) => x + y, 0) / Math.max(1, g1.length), g2 = g1.filter(p => L(p) > L(m1) - 10), m2 = g2.reduce((x, y) => x + y, 0) / Math.max(1, g2.length);
  const g3 = st.map(L).filter(v => v > -70).sort((x, y) => x - y), stm = st.length ? st.reduce((x, y) => x + y, 0) / st.length : 0, gated = g3.filter(v => v > L(stm) - 20), lra = gated.length > 2 ? gated[Math.floor(gated.length * .95)] - gated[Math.floor(gated.length * .1)] : 0;
  return { I: L(m2), M: L(Math.max(0, ...mom)), S: L(Math.max(0, ...st)), k100: L(Math.max(0, ...e)), lra, mom: mom.map(L), st: st.map(L) };
}
// 4x oversampled true peak
const TPF = (() => { const T = 48, k = []; for (let p = 0; p < 4; p++) { const h = new Float64Array(T); for (let i = 0; i < T; i++) { const x = i - T / 2 + 1 - p / 4 + 0, s = Math.abs(x) < 1e-9 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x); h[i] = s * (.5 + .5 * Math.cos(Math.PI * x / (T / 2 + 1))); } k.push(h); } return { T, k }; })();
export function truePeakEnv(x) { const { T, k } = TPF, n = x.length, y = new Float32Array(n); for (let i = 0; i < n; i++) { let m = Math.abs(x[i]); if (m > .25) for (let p = 1; p < 4; p++) { let s = 0; const h = k[p]; for (let j = 0; j < T; j++) { const q = i + j - T / 2 + 1; if (q >= 0 && q < n) s += x[q] * h[j]; } s = Math.abs(s); if (s > m) m = s; } y[i] = m; } return y; }
// average power spectrum (Hann, hop n/2)
export function spectrum(x, n = 4096) { const P = new Float64Array(n / 2 + 1); let c = 0; const w = new Float64Array(n); for (let i = 0; i < n; i++) w[i] = .5 - .5 * Math.cos(TAU * i / n); for (let o = 0; o < Math.max(1, x.length - n / 2); o += n / 2) { const R = new Float64Array(n), I = new Float64Array(n); for (let i = 0; i < n && o + i < x.length; i++) R[i] = x[o + i] * w[i]; fft(R, I); for (let i = 0; i <= n / 2; i++) P[i] += R[i] * R[i] + I[i] * I[i]; c++; } for (let i = 0; i < P.length; i++) P[i] /= c; return P; }

// ---------------------------------------------------------------- WAV
import fs from 'node:fs';
export function writeWav(path, l, r, bits = 24) { const n = l.length, ch = 2, bps = bits / 8, b = Buffer.alloc(44 + n * ch * bps); b.write('RIFF', 0); b.writeUInt32LE(36 + n * ch * bps, 4); b.write('WAVEfmt ', 8); b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20); b.writeUInt16LE(ch, 22); b.writeUInt32LE(SR, 24); b.writeUInt32LE(SR * ch * bps, 28); b.writeUInt16LE(ch * bps, 32); b.writeUInt16LE(bits, 34); b.write('data', 36); b.writeUInt32LE(n * ch * bps, 40);
  let o = 44; const M = Math.pow(2, bits - 1) - 1; for (let i = 0; i < n; i++) for (const x of [l, r]) { const v = Math.round(clamp(x[i], -1, 1) * M); if (bits === 16) b.writeInt16LE(v, o); else b.writeIntLE(v, o, 3); o += bps; } fs.writeFileSync(path, b); }
export function readWav(path) {
  const b = fs.readFileSync(path); let o = 12, fmt = null, data = null; while (o + 8 <= b.length) { const id = b.toString('latin1', o, o + 4), sz = b.readUInt32LE(o + 4); if (id === 'fmt ') fmt = { tag: b.readUInt16LE(o + 8), ch: b.readUInt16LE(o + 10), sr: b.readUInt32LE(o + 12), bits: b.readUInt16LE(o + 22), sub: sz >= 26 ? b.readUInt16LE(o + 32) : 0 }; if (id === 'data') { data = [o + 8, Math.min(sz, b.length - o - 8)]; break; } o += 8 + sz + (sz & 1); }
  const tag = fmt.tag === 65534 ? fmt.sub : fmt.tag, by = fmt.bits / 8, n = Math.floor(data[1] / by / fmt.ch), X = Array.from({ length: fmt.ch }, () => new Float64Array(n)); let p = data[0];
  for (let i = 0; i < n; i++) for (let c = 0; c < fmt.ch; c++) { X[c][i] = tag === 3 ? (by === 8 ? b.readDoubleLE(p) : b.readFloatLE(p)) : by === 2 ? b.readInt16LE(p) / 32768 : by === 3 ? b.readIntLE(p, 3) / 8388608 : by === 4 ? b.readInt32LE(p) / 2147483648 : (b[p] - 128) / 128; p += by; }
  let l = X[0], r = X[1] ?? X[0]; if (fmt.sr !== SR) { l = resample(l, fmt.sr, SR); r = X[1] ? resample(r, fmt.sr, SR) : l; } return { l, r, sr: fmt.sr, ch: fmt.ch };
}
// adds a wav file straight into two long buffers (no intermediate copy when it is already 48 kHz)
export function addWav(path, L, R, g = 1) { const h = fs.readFileSync(path); let o = 12, f = null, d = null; while (o + 8 <= h.length) { const id = h.toString('latin1', o, o + 4), sz = h.readUInt32LE(o + 4); if (id === 'fmt ') f = { tag: h.readUInt16LE(o + 8), ch: h.readUInt16LE(o + 10), sr: h.readUInt32LE(o + 12), bits: h.readUInt16LE(o + 22), sub: sz >= 26 ? h.readUInt16LE(o + 32) : 0 }; if (id === 'data') { d = [o + 8, Math.min(sz, h.length - o - 8)]; break; } o += 8 + sz + (sz & 1); }
  if (f.sr !== SR) { const w = readWav(path); for (let i = 0; i < w.l.length && i < L.length; i++) { L[i] += w.l[i] * g; R[i] += w.r[i] * g; } return; } const tag = f.tag === 65534 ? f.sub : f.tag, by = f.bits / 8, n = Math.min(L.length, Math.floor(d[1] / by / f.ch)), at = p => tag === 3 ? (by === 8 ? h.readDoubleLE(p) : h.readFloatLE(p)) : by === 2 ? h.readInt16LE(p) / 32768 : by === 3 ? h.readIntLE(p, 3) / 8388608 : h.readInt32LE(p) / 2147483648; for (let i = 0, p = d[0]; i < n; i++, p += by * f.ch) { const a = at(p); L[i] += a * g; R[i] += (f.ch > 1 ? at(p + by) : a) * g; } }
export function resample(x, a, b) { const ratio = a / b, n = Math.floor(x.length / ratio), y = new Float64Array(n), W = 24, fc = Math.min(1, b / a) * .97; for (let i = 0; i < n; i++) { const p = i * ratio, k = Math.floor(p); let s = 0; for (let j = k - W + 1; j <= k + W; j++) { if (j < 0 || j >= x.length) continue; const d = (p - j) * fc, z = Math.abs(d) < 1e-9 ? 1 : Math.sin(Math.PI * d) / (Math.PI * d); s += x[j] * z * fc * (.5 + .5 * Math.cos(Math.PI * (p - j) / W)); } y[i] = s; } return y; }
