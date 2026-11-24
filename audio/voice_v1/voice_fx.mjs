// voice_fx.mjs : the vocal chain (equaliser, level riding, warmth, plate, small room, slap/eighth delays).  Everything written here, no packages.
import { SR, Biquad, db, todb, clamp, rng } from './voice_dsp.mjs';
export function eq(x, bands) { const y = Float32Array.from(x); for (const [type, f, q, g] of bands) new Biquad(type, f, q, g).run(y); return y; }
// soft-knee feed-forward compressor, RMS + peak detector
export function comp(x, { thr = -20, ratio = 3, atk = .008, rel = .12, knee = 8, makeup = 0, sc = null } = {}) {
  const y = new Float32Array(x.length), a = Math.exp(-1 / (atk * SR)), r = Math.exp(-1 / (rel * SR)), ra = Math.exp(-1 / (.004 * SR)); let e = 0, g = 0; const s = sc || x;
  for (let i = 0; i < x.length; i++) { e = ra * e + (1 - ra) * s[i] * s[i]; const l = 10 * Math.log10(e + 1e-10), o = l - thr; let t = o <= -knee / 2 ? 0 : o < knee / 2 ? (o + knee / 2) ** 2 / (2 * knee) : o; t *= -(1 - 1 / ratio); g = t < g ? a * g + (1 - a) * t : r * g + (1 - r) * t; y[i] = x[i] * Math.pow(10, (g + makeup) / 20); }
  return y;
}
export function deess(x, f = 6200, thr = -30, ratio = 3) { const hi = Float32Array.from(x); new Biquad('hp', f, .7).run(hi); new Biquad('hp', f, .7).run(hi); const y = new Float32Array(x.length); let e = 0; const a = Math.exp(-1 / (.0015 * SR)), r = Math.exp(-1 / (.05 * SR)); for (let i = 0; i < x.length; i++) { const v = Math.abs(hi[i]); e = v > e ? a * e + (1 - a) * v : r * e + (1 - r) * v; const l = todb(e) - thr, g = l > 0 ? Math.pow(10, -l * (1 - 1 / ratio) / 20) : 1; y[i] = x[i] - hi[i] * (1 - g); } return y; }
// even-order warmth, 2x oversampled by linear halves (gentle drive only)
export function warm(x, drive = 1.6, mix = .35, bias = .12) { const y = new Float32Array(x.length); let p = 0; const dc = Math.tanh(bias); let h = 0, hx = 0; const sat = v => (Math.tanh(v * drive + bias) - dc) / drive / (1 - dc * dc); for (let i = 0; i < x.length; i++) { const m = (p + x[i]) * .5, s = .5 * (sat(m) + sat(x[i])); p = x[i]; const d = s; h = .9995 * (h + d - hx); hx = d; y[i] = x[i] * (1 - mix) + h * mix; } return y; }
class Delay { constructor(n) { this.b = new Float32Array(Math.max(2, n + 4)); this.n = n; this.i = 0; } rd(d) { let p = this.i - d; while (p < 0) p += this.b.length; const k = p | 0, u = p - k; return this.b[k] * (1 - u) + this.b[(k + 1) % this.b.length] * u; } wr(v) { this.b[this.i] = v; this.i = (this.i + 1) % this.b.length; } }
// plate (figure-of-eight tank after Dattorro, retuned for 48 kHz, modulated)
export function plate(x, { decay = .72, pre = .025, damp = 5200, lowcut = 220, hicut = 7500, size = 1 } = {}) {
  const K = 48000 / 29761 * size, n = x.length, L = new Float32Array(n), Rr = new Float32Array(n), pd = new Delay(Math.round(pre * SR) + 2), hp = new Biquad('hp', lowcut, .7), lp = new Biquad('lp', hicut, .6);
  const ap = (d, g) => { const D = new Delay(Math.round(d * K) + 40), dl = Math.round(d * K); return { D, f: (v, m = 0) => { const z = D.rd(dl + m); const w = v - g * z; D.wr(w); return z + g * w; }, tap: t => D.rd(t * K) }; };
  const i1 = ap(142, .75), i2 = ap(107, .75), i3 = ap(379, .625), i4 = ap(277, .625), a1 = ap(672, -.7), a2 = ap(1800, .5), b1 = ap(908, -.7), b2 = ap(2656, .5), mk = d => { const D = new Delay(Math.round(d * K) + 8); return { D, dl: Math.round(d * K), tap: t => D.rd(t * K) }; }, d1 = mk(4453), d2 = mk(3720), e1 = mk(4217), e2 = mk(3163), dk = Math.exp(-2 * Math.PI * damp / SR); let l1 = 0, l2 = 0, fa = 0, fb = 0;
  for (let i = 0; i < n; i++) {
    pd.wr(x[i]); let v = lp.p(hp.p(pd.rd(Math.round(pre * SR)))); v = i4.f(i3.f(i2.f(i1.f(v))));
    const m = 9 * Math.sin(2 * Math.PI * .83 * i / SR), m2 = 9 * Math.sin(2 * Math.PI * .61 * i / SR + 1);
    let a = a1.f(v + decay * fb, 12 + m); d1.D.wr(a); a = d1.D.rd(d1.dl); l1 = a * (1 - dk) + l1 * dk; a = a2.f(l1 * decay); d2.D.wr(a); fa = d2.D.rd(d2.dl);
    let b = b1.f(v + decay * fa, 12 + m2); e1.D.wr(b); b = e1.D.rd(e1.dl); l2 = b * (1 - dk) + l2 * dk; b = b2.f(l2 * decay); e2.D.wr(b); fb = e2.D.rd(e2.dl);
    L[i] = .6 * (e1.tap(266) + e1.tap(2974) - b2.tap(1913) + e2.tap(1996) - d1.tap(1990) - a2.tap(187) - d2.tap(1066));
    Rr[i] = .6 * (d1.tap(353) + d1.tap(3627) - a2.tap(1228) + d2.tap(2673) - e1.tap(2111) - b2.tap(335) - e2.tap(121));
  }
  return [L, Rr];
}
// early reflections of a small treated room: sparse, dark, different per ear
export function room(x, seed = 7, { len = .045, n = 14, lp = 4200, gain = .5 } = {}) { const R = rng(seed), out = [new Float32Array(x.length), new Float32Array(x.length)]; for (const o of out) { const f = new Biquad('lp', lp, .6), src = Float32Array.from(x); f.run(src); for (let k = 0; k < n; k++) { const t = Math.round((.0035 + len * Math.pow((k + R.u()) / n, 1.2)) * SR), g = gain * (R.u() < .5 ? -1 : 1) * Math.pow(10, -1.1 * k / n) * (.5 + .5 * R.u()) / Math.sqrt(n) * 1.8; for (let i = t; i < x.length; i++) o[i] += src[i - t] * g; } } return out; }
export function echo(x, time, fb, { lp = 3200, hp = 350, taps = 4 } = {}) { const y = new Float32Array(x.length); let src = Float32Array.from(x); for (let k = 1; k <= taps; k++) { new Biquad('lp', lp, .7).run(src); new Biquad('hp', hp, .7).run(src); const d = Math.round(time * k * SR), g = Math.pow(fb, k - 1); for (let i = d; i < x.length; i++) y[i] += src[i - d] * g; } return y; }
export const add = (a, b, g = 1) => { for (let i = 0; i < a.length; i++) a[i] += b[i] * g; return a; };
export function limit(chs, ceil = .97) { const n = chs[0].length, look = 96, rel = Math.exp(-1 / (.08 * SR)); let g = 1; const need = new Float32Array(n); for (let i = 0; i < n; i++) { let p = 0; for (const c of chs) p = Math.max(p, Math.abs(c[i])); need[i] = p > ceil ? ceil / p : 1; } const q = new Float32Array(n); let h = 1; for (let i = n - 1; i >= 0; i--) { let m = need[i]; for (let k = 1; k <= look && i + k < n; k += 8) m = Math.min(m, need[i + k]); q[i] = m; } for (let i = 0; i < n; i++) { g = q[i] < g ? g + (q[i] - g) * .06 : 1 - (1 - g) * rel - 0; g = Math.min(g, 1); if (need[i] < g) g = need[i]; for (const c of chs) c[i] *= g; } return chs; }
