// Registry + shared physical building blocks for the effects library.
import { SR, TAU, N, St, rng, buf, noise, pink, brown, band, lp, hp, bp, lp4, hp4, svf, pk, hs, ls, env, curve, mul, mix, unit, unitr, pulse, bank, hit, ping, bubble, poisson, slip, air, wander, clamp, lerp, smooth, db, todb, loud, peakOf, fadeEnds } from './sfx_dsp.mjs';

export const SFX = {};
// def(name, meta, fn)   meta: { d: description, hear: what a blind listener should say, sync: seconds from file start to the moment that must sit on the frame,
//                        lo, hi: final band limits, cat, fade: [in, out] }
export function def(name, meta, fn) { if (SFX[name]) throw new Error('duplicate effect ' + name); SFX[name] = { name, sync: 0, lo: 24, hi: 8500, cat: 'foley', ...meta, fn }; }
// chord of the moment (Hz, low to high); tonal effects tune themselves to it.  Default D minor.
export const CH = o => o.chord ?? [146.83, 174.61, 220];
export function render(name, o = {}) {
  const m = SFX[name]; if (!m) throw new Error('no effect called ' + name);
  const opts = { seed: 1, size: 1, v: 0, ...(m.def ?? {}), ...o }, r = rng(opts.seed * 101 + hash(name)), st = m.fn(opts, r);
  st.each(x => { hp(x, m.lo); lp(x, m.hi * .9, .6); });                 // gentle roof: air stays, hiss goes
  const tail = m.fade ?? [.001, .03]; st.fade(tail[0], tail[1]).trim(2e-5);
  const L = loud(st.l, st.r), g = Math.min(db(-18 - (Number.isFinite(L.k100) ? L.k100 : -18)), .84 / Math.max(1e-9, st.peak()));   // reference level: -18 LUFS in the loudest 100 ms, never above -1.5 dBFS
  st.gain(g); st.sync = typeof m.sync === 'function' ? m.sync(opts) : m.sync; st.name = name; return st;
}
export function hash(s) { let h = 2166136261; for (const c of s) h = Math.imul(h ^ c.charCodeAt(0), 16777619); return (h >>> 0) % 100003; }

// ---------------------------------------------------------------- shared parts
// low body: a sine whose pitch falls from f0 to f1
export function thud(st, at, f0, f1, t60, a, { glide = .05, pan = 0, atk = .004, h2 = 0 } = {}) {
  const n = N(t60 * 1.15), x = new Float64Array(n), d = Math.pow(10, -3 / (t60 * SR)), na = atk * SR; let ph = 0, e = 1;
  for (let i = 0; i < n; i++) { const f = f1 + (f0 - f1) * Math.exp(-i / SR / glide); ph += TAU * f / SR; x[i] = e * (i < na ? .5 - .5 * Math.cos(Math.PI * i / na) : 1) * (Math.sin(ph) + h2 * Math.sin(2 * ph + .4)); e *= d; }
  fadeEnds(x, 0.0005, .02); st.add(x, { at, g: a * .36, pan }); return st;
}
// band-limited noise burst with attack and t60
export function nb(st, r, at, lo, hi, atk, t60, a, { pan = 0, wide = 0, src = noise, hold = 0 } = {}) {
  atk = Math.max(atk, .0009); const n = N(atk + hold + t60 * 1.1), mk = () => { const x = unitr(band(r, n + 400, lo, hi, src).slice(400), .3), na = atk * SR, nh = (atk + hold) * SR, d = Math.pow(10, -3 / (t60 * SR)); let e = 1; for (let i = 0; i < n; i++) { if (i < na) x[i] *= .5 - .5 * Math.cos(Math.PI * i / na); else if (i > nh) { x[i] *= e; e *= d; } } return fadeEnds(x, .0003, .01); };
  if (wide) { const m = mk(); st.add(m, { at, g: a * (1 - wide * .5), pan }); const s1 = mk(), s2 = mk(); mix(st.l, s1, a * wide * .7, at); mix(st.r, s2, a * wide * .7, at); } else st.add(mk(), { at, g: a, pan });
  return st;
}
// a short band-passed grain (friction, paper, cloth, tearing fibres)
export function grain(st, r, at, f, q, ms, a, pan = 0) { const n = N(ms / 1000 * 2.2), x = noise(r, n); svf(x, f, q, 'bp'); const d = Math.pow(10, -3 / (ms / 1000 * 1.6 * SR)); let e = 1; const na = Math.max(16, .2 * n); for (let i = 0; i < n; i++) { x[i] *= e * (i < na ? .5 - .5 * Math.cos(Math.PI * i / na) : 1); e *= d; } unit(fadeEnds(x, .0002, .0006)); st.add(x, { at, g: a, pan }); }
// crumbs of stone, bone, glass: poisson cloud of small two-mode impacts
export function crumbs(st, r, { at = 0, dur = .6, rate = 200, fall = 3, lo = 250, hi = 1800, t60 = .02, a = .1, spread = .8, pw = 2, big = 0, pan = 0, ratio = 1.63 } = {}) {
  for (const t of poisson(r, dur, t => rate * Math.exp(-fall * t / dur))) { const s = Math.pow(r(), pw), f = r.log(lo, hi) * (1 - .5 * s * big), p = clamp(pan + r.u(-spread, spread), -1, 1), am = a * (.12 + .88 * s) * Math.exp(-fall * .4 * t / dur), tt = t60 * r.u(.5, 1.6) * (1 + s * big * 2); st.ping(at + t, f, tt, am, p, 0, .0004 + .4 / f); st.ping(at + t, f * ratio * r.u(.94, 1.07), tt * .6, am * .5, p, 0, .0004); }
}
// rustle of paper, film or leaves: power-law crumple events
export function rustle(st, r, { at = 0, dur = .4, rate = 300, lo = 900, hi = 4500, a = .1, ms = [.5, 3], shape = () => 1, spread = .5, pan = 0, pw = 3, q = 1.4 } = {}) {
  for (const t of poisson(r, dur, t => rate * shape(t / dur))) { const s = Math.pow(r(), pw); grain(st, r, at + t, r.log(lo, hi), q * r.u(.6, 1.6), r.u(ms[0], ms[1]) * (1 + s), a * (.1 + .9 * s) * Math.sqrt(shape(t / dur)), clamp(pan + r.u(-spread, spread), -1, 1)); }
}
// cluster of bubbles: thick or thin liquids.  Most events are short noisy blips (film breaking, cavity closing), a minority ring
export function bubbles(st, r, { at = 0, dur = .3, rate = 80, mm = [2, 8], a = .1, shape = () => 1, spread = .5, pan = 0, xi = .05, pw = 1.6, ring = .3 } = {}) {
  for (const t of poisson(r, dur, t => rate * 1.5 * shape(t / dur))) { const m = mm[0] * Math.pow(mm[1] / mm[0], Math.pow(r(), pw)), f = 3260 / m, p = clamp(pan + r.u(-spread, spread), -1, 1), g = a * r.u(.25, 1) * Math.pow(m / mm[1], .6);
    grain(st, r, at + t, f * r.u(.8, 1.6), r.u(1.5, 4), clamp(2600 / f, 1.4, 16), g * .8, p); if (r() < ring) { const q = (clamp(p, -1, 1) + 1) * Math.PI / 4, tt = clamp(9 / f * 2.4, .006, .09) * r.u(.6, 1.3), b = Math.min(6, xi * 60) * r.u(-.3, 1); ping(st.l, at + t, f, tt, g * .8 * Math.cos(q) * 1.414, b, .2 / f); ping(st.r, at + t, f, tt, g * .8 * Math.sin(q) * 1.414, b, .2 / f); } }
}
// dense inharmonic body (plate, beam, shell, slab): n modes between lo and hi, decay falling with frequency
export function dense(r, { lo = 120, hi = 4000, n = 36, t60 = .2, roll = .6, tilt = .5, jit = .9 } = {}) { const m = []; for (let i = 0; i < n; i++) { const f = lo * Math.pow(hi / lo, clamp((i + jit * r.u(-.5, .5)) / (n - 1))); m.push([f, t60 * Math.pow(lo / f, roll) * r.u(.6, 1.35), Math.pow(lo / f, tilt) * r.u(.35, 1) * r.sgn()]); } return m; }
// strike a modal body: contact pulse + rough micro-contact; two slightly different pickups make it stereo; peak = a
export function strike(st, r, at, modes, { tc = .002, rough = .5, rt = .006, a = 1, pan = 0, width = .5, dur = 0, hits = [[0, 1]] } = {}) {
  const D = dur || Math.min(5, Math.max(...modes.map(m => m[1])) * 1.1 + .05), x = buf(D), nr = N(rt * 3); for (const [t, g] of hits) { pulse(x, t, tc, g); const o = N(t); let e = rough * g / Math.max(8, tc * SR * .5); const d = Math.pow(10, -3 / (rt * SR * 2)); for (let i = 0; i < nr && o + i < x.length; i++) { x[o + i] += (r() * 2 - 1) * e * 3; e *= d; } }
  const L = bank(x, modes.map(m => [m[0], m[1], m[2] * (1 + width * r.u(-1, 1))])), R = bank(x, modes.map(m => [m[0], m[1], m[2] * (1 + width * r.u(-1, 1))])), pk = Math.max(peakOf(L), peakOf(R), 1e-9); const g = (clamp(pan, -1, 1) + 1) * Math.PI / 4; mix(st.l, L, a / pk * Math.cos(g) * 1.414, at); mix(st.r, R, a / pk * Math.sin(g) * 1.414, at); return st;
}
// meat: no ringing at all, three damped layers of mass, muscle and skin
export function flesh(st, r, at, { a = 1, pan = 0, mass = 1, skin = 1, f = 1 } = {}) { nb(st, r, at, 55 * f, 210 * f, .003, .12, 1.5 * a * mass, { pan, src: pink }); nb(st, r, at, 190 * f, 800 * f, .0012, .07, 1.5 * a, { pan }); nb(st, r, at, 700 * f, 2600 * f, .0005, .032, 1.3 * a * skin, { pan }); }
// wet: noise through a cavity that keeps changing shape
export function squelch(st, r, { at = 0, dur = .2, f0 = 300, f1 = 1600, q = 2.4, rate = 45, a = .5, pan = 0, shape = u => Math.sin(Math.PI * u), gate = .7 } = {}) { const n = N(dur), x = band(r, n, f0 * .5, Math.min(6000, f1 * 2), noise), w = wander(r, n, rate), g = wander(r, n, rate * 1.7); svf(x, t => { const i = Math.min(n - 1, Math.floor(t * SR)); return f0 * Math.pow(f1 / f0, .5 + .5 * w[i]); }, q, 'bp'); for (let i = 0; i < n; i++) x[i] *= Math.pow(Math.max(0, shape(i / n)), 1.1) * (1 - gate + gate * Math.pow(.5 + .5 * g[i], 2)); unit(fadeEnds(x, .003, .008)); st.add(x, { at, g: a, pan }); }
// thick liquid gulps: a cavity fills, its pitch climbs, it closes
export function glugs(st, r, { at = 0, dur = 2, rate = 3, f = 110, a = .5, spread = .7, pan = 0, shape = () => 1, q = 4 } = {}) { for (const t of poisson(r, dur, t => rate * shape(t / dur))) { const L = r.u(.14, .32), n = N(L), x = band(r, n, f * .6, f * 7, pink), ff = f * r.log(.75, 1.9), w = wander(r, n, 28), up = r.u(.35, .8); svf(x, u => ff * (1 + up * Math.pow(u / L, 1.5)), q, 'bp'); for (let i = 0; i < n; i++) x[i] *= Math.pow(Math.sin(Math.PI * Math.pow(i / n, .7)), 1.5) * (.75 + .25 * w[i]); unit(fadeEnds(x, .008, .01)); const p = clamp(pan + r.u(-spread, spread), -1, 1); st.add(x, { at: at + t, g: a * r.u(.4, 1), pan: p }); nb(st, r, at + t + L * .85, f * 2, f * 8, .006, .06, a * .22, { pan: p, src: pink }); } }
// brittle fracture of stone, asphalt, concrete: cracks that start fast and find their way outward (no shock waves: those belong to Black Flash alone)
export function fracture(st, r, { at = 0, dur = .5, n = 24, lo = 180, hi = 1600, a = 1, pan = 0, spread = .6, pw = 1.7 } = {}) { for (let k = 0; k < n; k++) { const t = dur * Math.pow((k + r()) / n, pw), u = t / dur, f = r.log(lo, hi) * (1 - .35 * u), g = a * (1 - .7 * u) * r.u(.35, 1), p = clamp(pan + r.u(-spread, spread) * (.3 + u), -1, 1); grain(st, r, at + t, f, .9, r.u(3, 9), .7 * g, p); strike(st, r, at + t, dense(r, { lo: f * .5, hi: Math.min(3600, f * 4), n: 9, t60: .045, roll: .5, tilt: .2 }), { tc: .0015, rough: 1.8, a: .55 * g, pan: p, width: .2, dur: .09 }); } }
// close lightning: a crooked channel is a chain of short shock segments, each arrives as an N-wave at its own time.  This IS a tear.
export function lightning(st, r, { at = 0, dur = .5, n = 420, crack = .05, wave = [.0006, .0042], a = 1, pan = 0, fall = 5, claps = 3 } = {}) {
  const pre = new St(dur + .05), tc = Array.from({ length: claps }, () => Math.pow(r(), 1.5) * dur * .8); for (let k = 0; k < n; k++) { const u = r(); const t = u < .42 ? crack * Math.pow(r(), 1.7) : u < .72 ? r.pick(tc) + Math.abs(r.g()) * .012 : Math.pow(r(), 1.6) * dur, w = r.log(wave[0], wave[1]) * (1 + 1.6 * t / dur), m = Math.max(6, N(w)), g = Math.exp(-fall * t / dur) * r.u(.2, 1) * (r() < .5 ? 1 : .5) * (t < crack ? 1.2 : 1), p = clamp(pan + r.u(-1, 1) * clamp(.1 + t / dur * 2), -1, 1), pa = (p + 1) * Math.PI / 4, gl = g * Math.cos(pa) * 1.414, gr = g * Math.sin(pa) * 1.414, o = N(t), sg = r.sgn(); for (let i = 0; i < m && o + i < pre.n; i++) { const q = i / m, v = sg * (1 - 2 * q) * Math.min(1, q * 10) * Math.min(1, (1 - q) * 10); pre.l[o + i] += v * gl; pre.r[o + i] += v * gr; } }
  const pk = pre.peak(); st.addSt(pre, { at, g: a / pk }); return st;
}
// cloth: soft noise that follows speed, with slow irregular weave
export function cloth(st, r, { at = 0, dur = .3, lo = 250, hi = 2200, a = .2, shape = u => Math.sin(Math.PI * u), pan = 0, wide = .4 } = {}) {
  const n = N(dur), mk = () => { const x = band(r, n, lo, hi, pink), w = wander(r, n, 35), w2 = wander(r, n, 9); for (let i = 0; i < n; i++) x[i] *= Math.pow(Math.max(0, shape(i / n)), 1.2) * (.62 + .38 * w[i]) * (.75 + .25 * w2[i]); return unit(fadeEnds(x, .006, .01)); };
  st.add(mk(), { at, g: a * (1 - wide * .4), pan }); if (wide) { mix(st.l, mk(), a * wide * .6, at); mix(st.r, mk(), a * wide * .6, at); }
}
// whoosh: turbulent air around a moving body, velocity u(x) over 0..1, with a pan path
export function whoosh(st, r, { at = 0, dur = .3, f = 900, q = 1.1, a = .5, u = x => Math.sin(Math.PI * Math.pow(x, 1.4)), pan = 0, pw = 2.2, hi = 5500, lo = 60 } = {}) { const x = air(r, dur, t => Math.max(.02, u(t / dur)), { f, q, pw, hi, lo }); unit(fadeEnds(x, .004, .01)); st.add(x, { at, g: a, pan: typeof pan === 'function' ? t => pan(t / dur) : pan }); }
// tonal aeolian whistle of a thin object (diameter sets pitch), doppler given by pitch(x)
export function whistle(st, r, { at = 0, dur = .4, f = 1800, q = 14, a = .3, u = x => Math.sin(Math.PI * x), pan = 0, pitch = () => 1 } = {}) { const n = N(dur), x = band(r, n, f * .35, Math.min(7500, f * 2.2)); svf(x, t => f * pitch(t / dur), q, 'bp'); mul(x, t => Math.pow(Math.max(0, u(t / dur)), 1.6)); unit(fadeEnds(x, .006, .012)); st.add(x, { at, g: a, pan: typeof pan === 'function' ? t => pan(t / dur) : pan }); }
// slow reversed swell made from any stereo builder
export function reversed(st) { for (const c of ['l', 'r']) { const x = st[c]; for (let i = 0, j = x.length - 1; i < j; i++, j--) { const t = x[i]; x[i] = x[j]; x[j] = t; } } return st; }
// sustained partial cluster with individual drift and entry times: [[f, amp, pan, t0, rise]]
export function cluster(st, r, parts, { at = 0, dur = 2, drift = .003, am = .15, rel = .2, rate = .7, breath = .8 } = {}) {
  const n = N(dur); for (const [f, a, pan = 0, t0 = 0, rise = .4] of parts) { const w = wander(r, n, rate * r.u(.6, 1.5)), w2 = wander(r, n, rate * 2.1), b1 = wander(r, n, Math.min(55, 9 + f * .03)), b2 = wander(r, n, 23), x = new Float64Array(n); let ph = r() * TAU; for (let i = 0; i < n; i++) { const t = i / SR; ph += TAU * f * (1 + drift * w[i]) / SR; const e = smooth((t - t0) / rise) * smooth((dur - t) / rel); x[i] = Math.sin(ph + breath * b1[i]) * e * (1 + am * w2[i]) * (1 + .4 * breath * b2[i]); } st.add(x, { at, g: a, pan }); }
}
export { SR, TAU, N, St, rng, buf, noise, pink, brown, band, lp, hp, bp, lp4, hp4, svf, pk, hs, ls, env, curve, mul, mix, unit, unitr, pulse, bank, hit, ping, bubble, poisson, slip, air, wander, clamp, lerp, smooth, db, todb, loud, peakOf, fadeEnds };
