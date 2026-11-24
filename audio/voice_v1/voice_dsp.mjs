// voice_dsp.mjs : small numeric kit for the singer.  Node 25, zero packages.
import fs from 'node:fs';
export const SR = 48000;

// ---------- random ----------
export function rng(seed) {            // xorshift128-ish, deterministic
  let a = seed >>> 0 || 1, b = 0x9e3779b9, c = 0x243f6a88, d = 0xb7e15162 ^ a;
  const u = () => { const t = (a + b | 0) + d | 0; d = d + 1 | 0; a = b ^ b >>> 9; b = c + (c << 3) | 0; c = c << 21 | c >>> 11; c = c + t | 0; return (t >>> 0) / 4294967296; };
  for (let i = 0; i < 20; i++) u();
  let spare = null;
  const g = () => { if (spare !== null) { const s = spare; spare = null; return s; } let x, y, r; do { x = 2 * u() - 1; y = 2 * u() - 1; r = x * x + y * y; } while (r >= 1 || r === 0); const m = Math.sqrt(-2 * Math.log(r) / r); spare = y * m; return x * m; };
  return { u, g, range: (lo, hi) => lo + (hi - lo) * u() };
}
// band-limited slow noise: sum of value noise octaves, unit-ish variance
export function slowNoise(seed) {
  const r = rng(seed), tab = new Float64Array(4096); for (let i = 0; i < 4096; i++) tab[i] = r.g();
  return (t, f) => { const x = t * f + 1000.37; const i = Math.floor(x), u = x - i, w = u * u * (3 - 2 * u); return tab[i & 4095] * (1 - w) + tab[(i + 1) & 4095] * w; };
}

// ---------- FFT (real, packed) ----------
export class RFFT {
  constructor(n) {                      // n real points, power of two
    this.n = n; const m = this.m = n >> 1; this.lg = Math.log2(m) | 0;
    this.cos = new Float64Array(m / 2); this.sin = new Float64Array(m / 2);
    for (let i = 0; i < m / 2; i++) { this.cos[i] = Math.cos(2 * Math.PI * i / m); this.sin[i] = Math.sin(2 * Math.PI * i / m); }
    this.wc = new Float64Array(m + 1); this.ws = new Float64Array(m + 1);
    for (let i = 0; i <= m; i++) { this.wc[i] = Math.cos(Math.PI * i / m); this.ws[i] = Math.sin(Math.PI * i / m); }
    this.rev = new Uint32Array(m); for (let i = 0; i < m; i++) { let r = 0; for (let b = 0; b < this.lg; b++) r |= ((i >> b) & 1) << (this.lg - 1 - b); this.rev[i] = r; }
    this.re = new Float64Array(m); this.im = new Float64Array(m);
  }
  _c(re, im) {                          // forward complex in place
    const m = this.m, { cos, sin, rev } = this;
    for (let i = 0; i < m; i++) { const j = rev[i]; if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
    for (let s = 2; s <= m; s <<= 1) { const h = s >> 1, st = m / s;
      for (let k = 0; k < m; k += s) for (let j = 0, p = 0; j < h; j++, p += st) { const a = k + j, b = a + h, wr = cos[p], wi = -sin[p]; const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr; re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi; } }
  }
  forward(x, Xr, Xi) {                  // x[n] -> X[0..m]
    const m = this.m, re = this.re, im = this.im;
    for (let i = 0; i < m; i++) { re[i] = x[2 * i]; im[i] = x[2 * i + 1]; }
    this._c(re, im);
    for (let k = 0; k <= m; k++) { const a = k === m ? 0 : k, b = k === 0 ? 0 : m - k; const er = .5 * (re[a] + re[b]), ei = .5 * (im[a] - im[b]), or = .5 * (im[a] + im[b]), oi = -.5 * (re[a] - re[b]); const wr = this.wc[k], wi = -this.ws[k]; Xr[k] = er + or * wr - oi * wi; Xi[k] = ei + or * wi + oi * wr; }
  }
  inverse(Xr, Xi, x) {                  // X[0..m] -> x[n]
    const m = this.m, re = this.re, im = this.im;
    for (let k = 0; k < m; k++) { const b = m - k; const er = .5 * (Xr[k] + Xr[b]), ei = .5 * (Xi[k] - Xi[b]), dr = .5 * (Xr[k] - Xr[b]), di = .5 * (Xi[k] + Xi[b]); const wr = this.wc[k], wi = this.ws[k]; const or = dr * wr - di * wi, oi = dr * wi + di * wr; re[k] = er - oi; im[k] = -(ei + or); }   // conj for inverse via forward
    this._c(re, im);
    for (let i = 0; i < m; i++) { x[2 * i] = re[i] / m; x[2 * i + 1] = -im[i] / m; }
  }
}

// ---------- WAV ----------
export function writeWav(path, chans, { float = true } = {}) {
  const nc = chans.length, n = chans[0].length, bps = float ? 4 : 2, buf = Buffer.alloc(44 + n * nc * bps);
  buf.write('RIFF', 0); buf.writeUInt32LE(36 + n * nc * bps, 4); buf.write('WAVEfmt ', 8); buf.writeUInt32LE(16, 16); buf.writeUInt16LE(float ? 3 : 1, 20); buf.writeUInt16LE(nc, 22); buf.writeUInt32LE(SR, 24); buf.writeUInt32LE(SR * nc * bps, 28); buf.writeUInt16LE(nc * bps, 32); buf.writeUInt16LE(bps * 8, 34); buf.write('data', 36); buf.writeUInt32LE(n * nc * bps, 40);
  let o = 44;
  for (let i = 0; i < n; i++) for (let c = 0; c < nc; c++) { const v = chans[c][i]; if (float) { buf.writeFloatLE(v, o); o += 4; } else { buf.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(v * 32767))), o); o += 2; } }
  fs.writeFileSync(path, buf);
}
export function readWav(path) {
  const b = fs.readFileSync(path); let o = 12, fmt, data; while (o < b.length) { const id = b.toString('latin1', o, o + 4), sz = b.readUInt32LE(o + 4); if (id === 'fmt ') fmt = { f: b.readUInt16LE(o + 8), nc: b.readUInt16LE(o + 10), sr: b.readUInt32LE(o + 12), bits: b.readUInt16LE(o + 22) }; if (id === 'data') data = b.subarray(o + 8, o + 8 + sz); o += 8 + sz + (sz & 1); }
  const n = data.length / (fmt.nc * fmt.bits / 8) | 0, ch = Array.from({ length: fmt.nc }, () => new Float32Array(n));
  for (let i = 0, p = 0; i < n; i++) for (let c = 0; c < fmt.nc; c++) { if (fmt.bits === 32) { ch[c][i] = data.readFloatLE(p); p += 4; } else if (fmt.bits === 16) { ch[c][i] = data.readInt16LE(p) / 32768; p += 2; } else { ch[c][i] = data.readIntLE(p, 3) / 8388608; p += 3; } }
  return { sr: fmt.sr, ch };
}

// ---------- filters ----------
export class Biquad {
  constructor(type, f, q = Math.SQRT1_2, g = 0) { this.set(type, f, q, g); this.z1 = 0; this.z2 = 0; }
  set(type, f, q, g) {
    const w = 2 * Math.PI * Math.min(f, SR * .49) / SR, cs = Math.cos(w), sn = Math.sin(w), al = sn / (2 * q), A = Math.pow(10, g / 40); let b0, b1, b2, a0, a1, a2;
    if (type === 'lp') { b0 = b2 = (1 - cs) / 2; b1 = 1 - cs; a0 = 1 + al; a1 = -2 * cs; a2 = 1 - al; }
    else if (type === 'hp') { b0 = b2 = (1 + cs) / 2; b1 = -(1 + cs); a0 = 1 + al; a1 = -2 * cs; a2 = 1 - al; }
    else if (type === 'bp') { b0 = al; b1 = 0; b2 = -al; a0 = 1 + al; a1 = -2 * cs; a2 = 1 - al; }
    else if (type === 'pk') { b0 = 1 + al * A; b1 = -2 * cs; b2 = 1 - al * A; a0 = 1 + al / A; a1 = -2 * cs; a2 = 1 - al / A; }
    else if (type === 'ls') { const s = 2 * Math.sqrt(A) * al; b0 = A * ((A + 1) - (A - 1) * cs + s); b1 = 2 * A * ((A - 1) - (A + 1) * cs); b2 = A * ((A + 1) - (A - 1) * cs - s); a0 = (A + 1) + (A - 1) * cs + s; a1 = -2 * ((A - 1) + (A + 1) * cs); a2 = (A + 1) + (A - 1) * cs - s; }
    else if (type === 'hs') { const s = 2 * Math.sqrt(A) * al; b0 = A * ((A + 1) + (A - 1) * cs + s); b1 = -2 * A * ((A - 1) + (A + 1) * cs); b2 = A * ((A + 1) + (A - 1) * cs - s); a0 = (A + 1) - (A - 1) * cs + s; a1 = 2 * ((A - 1) - (A + 1) * cs); a2 = (A + 1) - (A - 1) * cs - s; }
    this.b0 = b0 / a0; this.b1 = b1 / a0; this.b2 = b2 / a0; this.a1 = a1 / a0; this.a2 = a2 / a0;
  }
  p(x) { const y = this.b0 * x + this.z1; this.z1 = this.b1 * x - this.a1 * y + this.z2; this.z2 = this.b2 * x - this.a2 * y; return y; }
  run(x) { for (let i = 0; i < x.length; i++) x[i] = this.p(x[i]); return x; }
}
export const db = x => Math.pow(10, x / 20), todb = x => 20 * Math.log10(Math.max(x, 1e-12));
export const clamp = (x, a, b) => x < a ? a : x > b ? b : x, lerp = (a, b, t) => a + (b - a) * t, smooth = t => { t = clamp(t, 0, 1); return t * t * (3 - 2 * t); }, ss5 = t => { t = clamp(t, 0, 1); return t * t * t * (t * (6 * t - 15) + 10); };
const NN = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
export function midi(p) { const m = /^([A-G])([#b]?)(-?\d)$/.exec(p); return 12 * (+m[3] + 1) + NN[m[1]] + (m[2] === '#' ? 1 : m[2] === 'b' ? -1 : 0); }
export const hz = m => 440 * Math.pow(2, (m - 69) / 12);
