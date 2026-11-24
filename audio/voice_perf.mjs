// voice_perf.mjs : reads the score like a singer.  Turns notes + morae into articulator gestures, pitch, effort, breath.
import { rng, slowNoise, clamp, lerp, smooth, ss5, midi } from './voice_dsp.mjs';
import { CR, VI } from './voice_core.mjs';
export const E8 = .1875;

// ---------- morae ----------
export function parseMora(m) {
  if (m === 'N' || m === 'Q') return { t: m };
  const x = /^(sh|ch|ts|[kgsztdnhbpmyrwj])?(y)?([aiueo])$/.exec(m); if (!x) throw Error('mora ' + m);
  let c = x[1] || '', v = x[3]; if (c === 'h' && v === 'i') c = 'hy'; if (c === 'h' && v === 'u') c = 'f';
  return { t: 'CV', c, v, pal: !!x[2] || c === 'sh' || c === 'ch' || c === 'j' || c === 'y' };
}
const VOICED = new Set(['g', 'z', 'j', 'd', 'n', 'b', 'm', 'y', 'r', 'w', '', 'N']);
// closure / burst / friction / glide lengths in seconds
export const CT = {
  k: { cl: .042, vot: .030 }, t: { cl: .044, vot: .017 }, p: { cl: .05, vot: .015 }, d: { cl: .034, vot: 0 }, b: { cl: .042, vot: 0 }, g: { cl: .046, vot: 0 }, n: { cl: .052, vot: 0 }, m: { cl: .060, vot: 0 },
  s: { fr: .078 }, sh: { fr: .082 }, z: { fr: .066 }, j: { cl: .030, fr: .040 }, ch: { cl: .034, fr: .048 }, ts: { cl: .034, fr: .042 }, h: { fr: .056 }, hy: { fr: .066 }, f: { fr: .060 }, r: { cl: .017, vot: 0 }, w: { gl: .070 }, y: { gl: .072 },
};
const unit = { a: [1, 0, 0, 0, 0], i: [0, 1, 0, 0, 0], u: [0, 0, 1, 0, 0], e: [0, 0, 0, 1, 0], o: [0, 0, 0, 0, 1] };
const mix = (a, b, t) => a.map((x, i) => x * (1 - t) + b[i] * t);
// tongue-body posture asked for by a consonant and how strongly it insists
const BODY = {
  t: [[0, .25, .25, .5, 0], .30], d: [[0, .25, .25, .5, 0], .30], n: [[0, .25, .25, .5, 0], .28], s: [[0, .3, .3, .4, 0], .30], z: [[0, .3, .3, .4, 0], .30], ts: [[0, .25, .5, .25, 0], .35], r: [[0, .1, .4, .5, 0], .22],
  sh: [[0, .8, .1, .1, 0], .75], ch: [[0, .85, .1, .05, 0], .75], j: [[0, .85, .1, .05, 0], .75], y: [[0, .92, 0, .08, 0], .95], hy: [[0, 1, 0, 0, 0], .6], k: [[.1, .1, .5, .1, .2], .14], g: [[.1, .1, .5, .1, .2], .14], w: [[0, 0, .6, 0, .4], .85], h: [[0, 0, 0, 0, 0], 0], f: [[0, 0, 1, 0, 0], .3], m: [[0, 0, 0, 0, 0], 0], b: [[0, 0, 0, 0, 0], 0],
};
const VELLOC = { a: 27, o: 26, u: 27.5, e: 30.5, i: 32.5 };

// ---------- a line becomes a control segment ----------
export function buildLine(notes0, st, o = {}) {
  const R = rng((o.seed ?? 1) * 7919 + 13), nz = [0, 1, 2, 3, 4, 5, 6].map(k => slowNoise((o.seed ?? 1) * 977 + k * 31));
  const hum = o.hum ?? .007, shift = o.shift ?? 0;
  const notes = notes0.map((n, i) => { const [p0, p1] = n.pitch.split('>'); const m = parseMora(n.mora); let t = n.t * E8 + (o.lag ?? 0) + (i ? R.g() * hum : R.g() * hum * .5); return { ...m, mora: n.mora, t, d: n.d * E8, m0: midi(p0) + shift + (n.h ?? 0), m1: p1 ? midi(p1) + shift + (n.h ?? 0) : null, x: n.x || {} }; });
  const T0 = notes[0].t - 1.0, T1 = notes[notes.length - 1].t + notes[notes.length - 1].d + .8, n = Math.ceil((T1 - T0) * CR), F = v => new Float32Array(n).fill(v), idx = t => clamp(Math.round((t - T0) * CR), 0, n - 1), tm = i => T0 + i / CR;
  const s = { n, f0: F(0), ee: F(0), rd: F(1.5), asp: F(0), aspfl: F(.15), rasp: F(0), air: F(0), jit: F(1), tilt: F(6000), mvf: F(7000), bwo: F(120), bwc: F(10), eff: F(0), velum: F(0), w: [F(0), F(0), F(0), F(0), F(0)], gest: [], fric: [] };
  const gate = F(0), cdip = F(1), aspC = F(0), cents = F(0);
  const post = [], volt = [];            // posture targets {t, w, tau}, voiced intervals
  const ramp = (t0, t1, t2, t3) => { const a = F(0); for (let i = idx(t0); i <= idx(t3); i++) { const t = tm(i); a[i] = t < t1 ? ss5((t - t0) / Math.max(1e-4, t1 - t0)) : t < t2 ? 1 : 1 - ss5((t - t2) / Math.max(1e-4, t3 - t2)); } return a; };
  const close = (loc, hw, amin, t0, t1, t2, t3, deg = 1) => { const m = ramp(t0, t1, t2, t3); if (deg !== 1) for (let i = 0; i < n; i++) m[i] *= deg; s.gest.push({ loc, hw, amin, m }); };
  const fr = {
    s: { j: 40, fc: 6000, hp: 3400, hpn: 2, slope: 1.6, amn: .05, ag: .25, gn: 85 }, sh: { j: 38, fc: 3600, hp: 2000, hpn: 2, slope: 1.6, amn: .06, ag: .25, gn: 65 }, hy: { j: 36, fc: 2300, hp: 1500, slope: 2.0, amn: .08, ag: .3, gn: 75 }, f: { j: 43, fc: 2500, hp: 500, slope: 1.0, amn: .06, ag: .3, gn: 25 },
    ka: { j: 29, fc: 1500, hp: 700, slope: 1.9, amn: .03, ag: .2, gn: 130 }, ko: { j: 28, fc: 1200, hp: 500, slope: 1.9, amn: .03, ag: .2, gn: 190 }, ku: { j: 29, fc: 1300, hp: 600, slope: 1.9, amn: .03, ag: .2, gn: 160 }, ke: { j: 32, fc: 2100, hp: 1000, slope: 1.9, amn: .03, ag: .2, gn: 170 }, ki: { j: 34, fc: 2600, hp: 1200, slope: 1.9, amn: .03, ag: .2, gn: 190 },
    t: { j: 41, fc: 4200, hp: 2600, hpn: 2, slope: 1.35, amn: .04, ag: .2, gn: 95 }, z: { j: 40, fc: 6000, hp: 3200, hpn: 2, slope: 1.6, amn: .05, ag: .12, mod: .6, gn: 85 }, j: { j: 38, fc: 3600, hp: 2000, hpn: 2, slope: 1.6, amn: .06, ag: .12, mod: .6, gn: 65 }, p: { j: 43, fc: 1500, hp: 250, slope: 1.2, amn: .05, ag: .2, gn: 45 } };
  for (const k in fr) fr[k].amp = F(0);
  const puff = (name, t0, t1, t2, t3, a) => { const r = ramp(t0, t1, t2, t3), e = fr[name].amp; for (let i = idx(t0); i <= idx(t3); i++) e[i] = Math.max(e[i], a * r[i]); };
  const burst = (name, t, a, tau) => { const e = fr[name].amp; for (let i = idx(t); i <= idx(t + tau * 5); i++) { const x = Math.max(0, tm(i) - t); e[i] = Math.max(e[i], a * Math.exp(-x / tau)); } };
  const dip = (t0, t1, t2, t3, g) => { const r = ramp(t0, t1, t2, t3); for (let i = idx(t0); i <= idx(t3); i++) cdip[i] = Math.min(cdip[i], 1 - (1 - g) * r[i]); };
  const breathy = (t0, t1, t2, t3, a) => { const r = ramp(t0, t1, t2, t3); for (let i = idx(t0); i <= idx(t3); i++) aspC[i] = Math.max(aspC[i], a * r[i]); };
  const CS = st.cons ?? 1, WH = !!o.whisper;
  let cur = null, micro = []; const tau0 = st.tau ?? .021;
  // a coronal drags the tongue body forward, more so out of back vowels
  const COR = { t: 1, d: 1, n: 1, s: .9, z: .9, ts: .95, r: .85 }, CW = [.55, .25, .48, .30, .70], CB = [0, .32, .14, .54, 0];
  const bodyFor = (c, pv, nv) => { const b = BODY[c]; const base = mix(pv, nv, .55); if (COR[c]) { let w = 0; for (let v = 0; v < 5; v++) w += base[v] * CW[v]; return mix(base, CB, w * COR[c] * (st.loc ?? 1)); } return b && b[1] > 0 ? mix(base, b[0], b[1]) : base; };
  for (let k = 0; k < notes.length; k++) {
    const nt = notes[k], pr = notes[k - 1], nx = notes[k + 1], gapBefore = !pr || nt.t - (pr.t + pr.d) > .04, tv = nt.t;
    const pv = !pr || gapBefore && !cur ? null : cur;
    if (nt.t === undefined) throw 0;
    nt.start = gapBefore; nt.end = !nx || nx.t - (nt.t + nt.d) > .04;
    if (nt.mora === 'Q') { nt.kind = 'Q'; continue; }
    if (nt.mora === 'N') { nt.kind = 'N'; continue; }
    const nv = unit[nt.v], from = gapBefore ? nv : cur || nv; cur = nv; nt.kind = 'V';
    const avail = gapBefore ? .25 : (pr.kind === 'Q' ? pr.d * .62 : pr.kind === 'N' ? pr.d * .55 : Math.min(pr.d * .52, .12)), q = CT[nt.c], len = q ? (q.cl || 0) + (q.vot || 0) + (q.fr || 0) + (q.gl || 0) : 0, sc = q ? Math.min(1.25, CS * (1 + .12 * R.g() * .5), avail / len) : 1, L = x => (q[x] || 0) * sc;
    let c0 = tv, c1 = tv;                // constriction start, release
    const c = nt.c, afterQ = pr && pr.kind === 'Q' && !gapBefore, afterN = pr && pr.kind === 'N' && !gapBefore;
    if (c) {
      const voiced = VOICED.has(c);
      if (q.gl) { c0 = tv - L('gl'); c1 = tv - L('gl') * .25; post.push({ t: c0 - .02, w: bodyFor(c, from, nv), tau: tau0 }, { t: c1, w: nv, tau: tau0 * 1.15 }); if (c === 'w') close(43, 2.0, .35, c0 - .02, c0 + .03, c1, tv + .05, .9); else close(33, 4, .30, c0 - .02, c0 + .03, c1, tv + .045, .75); dip(c0 - .02, c0 + .025, c1, tv + .035, .72); }
      else if (c === 'h' || c === 'hy' || c === 'f') {
        c0 = tv - L('fr'); c1 = tv; post.push({ t: c0 - .03, w: bodyFor(c, from, nv), tau: tau0 }, { t: c0 + .012, w: nv, tau: tau0 });
        if (c === 'h') { breathy(c0 - .022, c0 + .016, tv - .012, tv + .04, 1); }
        else if (c === 'hy') { close(33.5, 4.5, .10, c0 - .04, c0, tv - .012, tv + .035, .93); puff('hy', c0 - .012, c0 + .02, tv - .02, tv + .012, .75); breathy(c0 - .02, c0 + .015, tv - .012, tv + .035, .55); }
        else { close(43, 2, .10, c0 - .035, c0, tv - .012, tv + .03, .93); puff('f', c0 - .012, c0 + .02, tv - .02, tv + .012, .8); breathy(c0 - .02, c0 + .015, tv - .012, tv + .035, .5); }
      }
      else if (q.fr) {                   // s sh z j ch ts
        const cl = L('cl'), f = L('fr'); c0 = tv - cl - f; c1 = tv - .004; if (afterQ) c0 = pr.t + pr.d * .40;
        post.push({ t: c0 - .03, w: bodyFor(c, from, nv), tau: tau0 }, { t: tv - .022, w: nv, tau: nt.pal ? tau0 * 1.25 : tau0 });
        const post_ = c === 'sh' || c === 'ch' || c === 'j', loc = post_ ? 36.0 : 39.0, hw = post_ ? 2.6 : 2.0, am = post_ ? .12 : .075;
        if (cl) { close(loc + (post_ ? .8 : 0), hw, .003, c0 - .035, c0, tv - f - .004, tv - f + .010); const nm = c === 'j' ? 'j' : post_ ? 'sh' : 's'; burst(nm, tv - f, c === 'j' ? .25 : .8, .006); }
        close(loc, hw, am, c0 - .035, c0 - .002, c1 - .008, tv + .034, .97);
        const name = c === 'z' ? 'z' : c === 'j' ? 'j' : post_ ? 'sh' : 's', amp = { s: .85, ts: .80, sh: .80, ch: .85, z: 1.0, j: .60 }[c] * (st.sib ?? 1);
        puff(name, tv - f - .008, tv - f * .62, tv - f * .30, tv + .014, amp);
        if (voiced) dip(c0 - .02, c0 + .02, tv - .012, tv + .03, .42);
      }
      else {                             // stops, nasals, tap
        const cl = L('cl'), vot = L('vot') * (st.vot ?? 1); c0 = tv - cl - vot; c1 = tv - vot; if (afterQ) c0 = pr.t + pr.d * .40; if (afterN) c0 = Math.max(pr.t + pr.d * .6, c0);
        const front = nt.pal || nt.v === 'i' ? 1 : nt.v === 'e' ? .6 : 0; const bw = nt.pal ? mix(nv, unit.i, .8) : bodyFor(c, from, nv);
        post.push({ t: c0 - .032, w: bw, tau: tau0 }, { t: c1 - (nt.pal ? -.004 : .010), w: nv, tau: nt.pal ? tau0 * 1.3 : tau0 });
        const loc = { k: VELLOC[nt.pal ? 'i' : nt.v], g: VELLOC[nt.pal ? 'i' : nt.v], t: 39.8, d: 39.8, n: 39.8, r: 39.0, b: 43, p: 43, m: 43 }[c], hw = { k: 3.0, g: 3.0, t: 2.2, d: 2.2, n: 2.4, r: 1.6, b: 1.4, p: 1.4, m: 1.4 }[c];
        const tin = { k: .042, g: .042, t: .034, d: .030, n: .034, r: .017, b: .034, p: .034, m: .036 }[c], tout = { k: .048, g: .046, t: .036, d: .034, n: .036, r: .020, b: .036, p: .036, m: .038 }[c], am = c === 'r' ? .08 : c === 'g' && !gapBefore ? .03 : .003;
        const hold = voiced ? 0 : Math.min(.014, vot * .7);      // loci stay audible a moment after the release
        close(loc, hw, am, c0 - tin, c0, c1, c1 + 0.002 + hold * .25, 1); close(loc, hw, .12, c0 - tin, c0, c1, c1 + tout + hold, .88);
        if (c === 'k') { const nm = 'k' + (nt.pal ? 'i' : nt.v); burst(nm, c1 - .001, 1.0, .006); puff(nm, c1, c1 + .004, c1 + vot * .5, tv + .012, .22); breathy(c1 - .003, c1 + .006, tv - .006, tv + .035, .7); }
        if (c === 't') { burst('t', c1 - .001, .95, .0045); puff('t', c1, c1 + .003, c1 + vot * .3, tv + .004, .22); breathy(c1 - .003, c1 + .006, tv - .004, tv + .032, .6); }
        if (c === 'p') { burst('p', c1 - .001, .9, .005); breathy(c1 - .003, c1 + .006, tv - .004, tv + .03, .5); }
        if (c === 'd') { burst('t', c1 - .001, .22, .004); dip(c0 - .02, c0 + .012, c1 - .004, c1 + .03, .55); micro.push([c0 + cl * .5, -40, .03]); }
        if (c === 'b') { burst('p', c1 - .001, .22, .004); dip(c0 - .02, c0 + .012, c1 - .004, c1 + .03, .55); micro.push([c0 + cl * .5, -40, .03]); }
        if (c === 'g') { const nas = gapBefore ? 0 : st.gn ?? 0; if (nas) addVel(c0 - .035, c0 + .005, c1 - .012, c1 + .03, .30 * nas); dip(c0 - .02, c0 + .012, c1 - .004, c1 + .03, .55); burst('k' + (nt.pal ? 'i' : nt.v), c1 - .002, gapBefore || afterN ? .45 : .28, .004); micro.push([c0 + cl * .5, -30, .03]); }
        if (c === 'n' || c === 'm') { addVel(c0 - .030, c0 + .006, c1 - .006, c1 + .024, .55); dip(c0 - .02, c0 + .015, c1 - .006, c1 + .03, 1.0); micro.push([c0 + cl * .5, -12, .03]); }
        if (c === 'r') { dip(c0 - .018, c0 + .004, c1, c1 + .022, .82); micro.push([c0 + cl * .5, -25, .02]); }
        if (afterN && pr) {}
      }
    } else if (gapBefore) { }
    else { const same = pr.v === nt.v; post.push({ t: tv - .052, w: nv, tau: tau0 * 1.12 }); if (same) dip(tv - .035, tv - .004, tv + .002, tv + .04, .62); else dip(tv - .04, tv - .005, tv + .004, tv + .045, .86); }
    if (nt.d > .4 && (st.belt ?? 0)) { const o2 = { a: 'a', i: 'e', u: 'o', e: 'a', o: 'a' }[nt.v], k = { a: 0, i: .20, u: .14, e: .13, o: .12 }[nt.v] * st.belt; post.push({ t: tv + .12, w: mix(nv, unit[o2], k * .6), tau: .06 }, { t: tv + nt.d * .45, w: mix(nv, unit[o2], k), tau: .09 }, { t: tv + nt.d * .82, w: mix(nv, [.2, .1, .3, .3, .1], .16), tau: .06 }); }
    if (gapBefore) post.push({ t: c0 - .42, w: mix(nv, [.25, 0, .25, .5, 0], .5), tau: .04 }); if (c === '') post.push({ t: c0 - .10, w: nv, tau: .025 });
    nt.c0 = c0; nt.c1 = c1;
    const voiced = VOICED.has(c);
    // voicing bookkeeping
    if (gapBefore || !voiced) volt.push({ on: voiced ? (c === '' ? tv - .012 : c === 'y' || c === 'w' ? c0 : c0 + (c === 'n' || c === 'm' ? -.02 : .004)) : c === 'h' ? tv - .022 : tv - .006, r0: gapBefore && c === '' ? (st.onset ?? .03) : voiced ? .03 : .022, off: null, r1: .02, phr: gapBefore });
    if (!voiced && volt.length > 1 && volt[volt.length - 2].off === null) { const v = volt[volt.length - 2]; v.off = c === 'h' || c === 'hy' || c === 'f' ? c0 + .012 : c0 + .006; v.r1 = c === 'h' ? .03 : .018; }
    if (nt.end) { const v = volt[volt.length - 1]; const rel = nt.x.rel ?? st.rel ?? .11; nt.offT = nt.t + nt.d - (nx ? .05 : .02) - rel * .35; if (!nx && o.nextT !== undefined) nt.offT = Math.min(nt.offT, Math.max(nt.t + nt.d * .55, o.nextT - .16 - rel * .55)); v.off = nt.offT; v.r1 = rel; }
  }
  function addVel(t0, t1, t2, t3, a) { const r = ramp(t0, t1, t2, t3); for (let i = idx(t0); i <= idx(t3); i++) s.velum[i] = Math.max(s.velum[i], a * r[i]); }
  // moraic nasal and geminate
  notes.forEach((nt, k) => {
    const nx = notes[k + 1], pr = notes[k - 1];
    if (nt.mora === 'N') { const c = nx && !nx.start ? nx.c : 'n', vel = c === 'g' || c === 'k', t0 = nt.t, t1 = nx ? (nx.c1 ?? nx.t) : nt.t + nt.d, ls = vel ? (nx.pal ? 31 : 28) : 40; close(ls, vel ? 3.2 : 2.4, .004, t0 - .03, t0 + .03, t1 - .01, t1 + .03, 1); addVel(t0 - .040, t0 + .012, t1 - .02, t1 + .020, .60); post.push({ t: t0 - .02, w: mix(unit[pr.v], [0, .3, .3, .4, 0], vel ? .3 : .45), tau: .03 }); dip(t0 - .03, t0 + .03, t1 - .02, t1 + .02, .84); nt.kind = 'N'; }
  });
  post.sort((a, b) => a.t - b.t);
  // articulators follow their targets (critically damped, twice)
  { let k = 0, tg = post[0].w, tau = post[0].tau; const a = tg.slice(), b = tg.slice(); for (let i = 0; i < n; i++) { const t = tm(i); while (k < post.length && post[k].t <= t) { tg = post[k].w; tau = post[k].tau; k++; } const g = 1 - Math.exp(-1 / (CR * tau)); for (let v = 0; v < 5; v++) { a[v] += (tg[v] - a[v]) * g; b[v] += (a[v] - b[v]) * g; s.w[v][i] = b[v]; } } }
  for (const v of volt) { const off = v.off ?? T1 - .2; for (let i = idx(v.on - v.r0 * .4); i <= idx(off + v.r1); i++) { const t = tm(i), u = smooth((t - (v.on - v.r0 * .4)) / v.r0), d = 1 - smooth((t - off) / v.r1); gate[i] = Math.max(gate[i], u * d); } }

  // ---------- pitch ----------
  const sung = notes, wn = st.wn ?? 46, zeta = st.zeta ?? .68, steps = [];
  sung.forEach((nt, k) => { const pr = sung[k - 1], lead = nt.start ? .30 : clamp(.052 + .006 * Math.abs(nt.m0 - pr.m0), .05, .09); steps.push({ t: nt.t - lead - (nt.x.late ? -nt.x.late : 0), m: nt.m0, nt }); if (nt.m1 !== null) steps.push({ t: nt.t + nt.d * (nt.x.glideAt ?? .42), m: nt.m1, nt, slow: 1 }); });
  { let k = 0, x = steps[0].m, v = 0, tg = x, w = wn; for (let i = 0; i < n; i++) { const t = tm(i); while (k < steps.length && steps[k].t <= t) { const s0 = steps[k]; if (s0.nt.start && !s0.slow) { x = s0.m; v = 0; } tg = s0.m; w = s0.slow ? 18 : wn; k++; } const dt = 1 / CR; for (let q = 0; q < 2; q++) { const ac = w * w * (tg - x) - 2 * zeta * w * v; v += ac * dt / 2; x += v * dt / 2; } cents[i] = x * 100; } }
  const bump = (tc, a, sg) => { for (let i = idx(tc - 3.2 * sg); i <= idx(tc + 3.2 * sg); i++) { const z = (tm(i) - tc) / sg; cents[i] += a * Math.exp(-.5 * z * z); } };
  for (const [t, a, sg] of micro) bump(t, a * (st.micro ?? 1.5) * (.6 + .8 * R.u()), sg);
  sung.forEach((nt, k) => { const dv = 7 * R.g(); bump(nt.t + nt.d * .5, dv, Math.min(.12, nt.d * .32)); if (nt.d < .3) bump(nt.t + nt.d * .86, -9 - 6 * R.u(), .03); });
  sung.forEach((nt, k) => {
    const pr = sung[k - 1]; if (pr && !nt.start && nt.m0 - pr.m0 >= 3) bump(nt.t - .10, -(8 + 5 * (nt.m0 - pr.m0)), .026);       // preparation
    if (pr && !nt.start && pr.m0 - nt.m0 >= 4) bump(nt.t - .10, 10, .026);
    const sc = nt.x.scoop ?? (nt.start ? (st.scoop0 ?? -70) * (.7 + .6 * R.u()) : R.u() < .6 ? -38 * R.u() : 0); if (sc) { const tau = nt.x.scoopT ?? .055, t0 = nt.start ? nt.c0 ?? nt.t : nt.t - .06; for (let i = idx(t0 - .1); i < idx(nt.t + .5); i++) { const t = tm(i) - t0; cents[i] += sc * (t < 0 ? (nt.start ? 1 : smooth(1 + t / .06)) : Math.exp(-Math.pow(t / tau, 1.5) * .5)); } }
    if (nt.x.fall || (nt.end && st.fall)) { const a = nt.x.fall ?? st.fall, te = nt.offT + .02, L = nt.x.fallT ?? .16; for (let i = idx(te - L); i < Math.min(n, idx(te + .25)); i++) { const u = (tm(i) - (te - L)) / L; cents[i] -= a * u * u; } }
    if (nt.x.bend) for (const [t, a, sg] of nt.x.bend) bump(nt.t + t, a, sg);
  });
  // vibrato: late, uneven, deeper when the note lives
  const vibPhase = F(0); let ph = R.u() * 6.28;
  notes.forEach((nt, k) => {
    let d = nt.d; if (notes[k + 1] && notes[k + 1].mora === 'Q') d += 0; if (d < (st.vibMin ?? .45)) return; const depth0 = (nt.x.vib ?? st.vib ?? 32) * (.62 + .62 * R.u()); if (depth0 <= 0) return;
    const on = nt.t + Math.max(.25, (nt.x.vibOn ?? clamp(d * .34, .25, .50)) * (.85 + .4 * R.u())), off = nt.offT ? nt.offT + .05 : nt.t + d - .05, rate = (nt.x.vibRate ?? st.vibRate ?? 5.6) * (.91 + .18 * R.u()); let cyc = 1 + .12 * R.g(), p = -Math.PI / 2 * 0 + 0, lastc = 0, rr = rate;
    for (let i = idx(on); i <= idx(off); i++) { const t = tm(i), u = smooth((t - on) / (nt.x.vibIn ?? clamp(d * .28, .14, .5))), e = 1 - smooth((t - off + .10) / .10), r2 = rr * (1 + .05 * Math.sin((t - on) * 2.2) + .03 * nz[3](t, 1.3)); p += 2 * Math.PI * r2 / CR; const c = Math.floor(p / (2 * Math.PI)); if (c !== lastc) { lastc = c; cyc = 1 + .16 * R.g(); rr = rate * (1 + .045 * R.g()); } const vv = Math.sin(p) + .14 * Math.sin(2 * p + .7) + .10 * Math.sin(.5 * p + 1); cents[i] += depth0 * u * e * cyc * vv; vibPhase[i] = u * e * Math.sin(p); }
  });
  const drift = st.drift ?? 1, dt = o.detune ?? 0;
  for (let i = 0; i < n; i++) { const t = tm(i); cents[i] += drift * (7 * nz[0](t, 1.1) + 6 * nz[1](t, 3.7) + 4 * nz[2](t, 8.5)) + dt + (o.wander ?? 0) * nz[4](t, .5); }
  if (st.tremor) for (let i = 0; i < n; i++) { const t = tm(i); cents[i] += st.tremor * (nz[5](t, 6.5) * 10); }

  // ---------- effort, loudness, voice quality ----------
  const tA = notes[0].t, tB = notes[notes.length - 1].t + notes[notes.length - 1].d, [e0, e1, e2] = st.eff, pk = st.peak ?? .55;
  for (let i = 0; i < n; i++) {
    const t = tm(i), u = clamp((t - tA) / (tB - tA), 0, 1); let e = u < pk ? lerp(e0, e1, smooth(u / pk)) : lerp(e1, e2, smooth((u - pk) / (1 - pk))); s.eff[i] = e;
  }
  const vfade = F(1), evo = F(0), eadd = F(0), raspE = F(0), gdb = F(0), rdAdd = F(0), aspAdd = F(0);
  notes.forEach((nt, k) => {
    const pr = notes[k - 1], nx = notes[k + 1], acc = nt.x.acc ?? ((Math.round(nt.t / E8) % 8 === 0) ? 1 : (Math.round(nt.t / E8) % 4 === 0) ? .6 : (Math.round(nt.t / E8) % 2 === 0) ? .25 : 0) * (st.accent ?? 1), end = nx && !nt.end ? nx.t : nt.t + nt.d;
    const hi = pr ? clamp((nt.m0 - pr.m0) / 12, -.4, .4) : 0, rnd = R.g() * .6;
    for (let i = idx(nt.t - .04); i < idx(end + .02); i++) { const t = tm(i) - nt.t, w = smooth((t + .04) / .05) * (1 - smooth((tm(i) - end) / .02)); const dd = Math.min(nt.d, .42), uu = clamp(t / dd, 0, 1), arc = -(st.arc ?? 2.2) * (Math.pow(1 - smooth(Math.min(1, t / .07)), 2) * .9 + .9 * smooth((uu - .35) / .65) * (nt.d > .5 ? .45 : 1)); const env = acc * 1.7 * Math.exp(-Math.max(0, t) / .16) + 1.3 * hi + rnd + arc; gdb[i] += w * env; eadd[i] += w * (acc * .06 * Math.exp(-Math.max(0, t) / .15) + .10 * hi); }
    if (nt.d > .35) for (let i = idx(nt.t - .02); i < idx(end + .1); i++) { const u = clamp((tm(i) - nt.t) / nt.d, 0, 1.15); evo[i] += .22 * Math.sin(Math.PI * Math.min(1, u * 1.1)) - .12; }
    if (nt.d > .4) { const bl = (st.belt ?? 0); for (let i = idx(nt.t - .02); i < idx(end + .12); i++) { const t = tm(i) - nt.t, u = clamp(t / nt.d, 0, 1.2), z = (u - .30) / .26, g = Math.exp(-z * z), tail = smooth((u - .74) / .26);
      gdb[i] += (2.2 + .6 * bl) * g - (nt.x.cresc ? .6 : nt.end ? 4.2 : 2.6) * tail + 1.1 * nz[4](tm(i) + 11, 3.1); rdAdd[i] += -(.10 + .08 * bl) * g + (nt.x.cresc ? .15 : nt.end ? .65 : .40) * tail + .10 * nz[5](tm(i) + 5, 2.6); aspAdd[i] += (nt.end ? 1.3 : .55) * tail; evo[i] += .16 * bl * smooth(t / .22);
      if (bl) { const q = nz[6](tm(i) + 3, 3.3); if (q > .55 && u < .7) raspE[i] = Math.max(raspE[i], (st.rasp ?? 0) * 1.1 * clamp((q - .55) * 2, 0, 1) * smooth(t / .1)); } } }
    if (nt.x.airTail) for (let i = idx(nt.t + nt.d * .42); i < n; i++) { const u = smooth((tm(i) - nt.t - nt.d * .42) / (nt.d * .42)); vfade[i] = Math.min(vfade[i], 1 - .96 * u * nt.x.airTail); aspAdd[i] += 3.2 * u; rdAdd[i] += .5 * u; }
    if (nt.start && st.slowAtk) for (let i = idx(nt.t - .3); i < idx(nt.t + st.slowAtk); i++) gdb[i] -= 10 * (1 - smooth((tm(i) - nt.t + .05) / st.slowAtk));
    if (nt.x.cresc) for (let i = idx(nt.t); i < idx(end + .3); i++) { const u = clamp((tm(i) - nt.t) / nt.d, 0, 1); eadd[i] += nt.x.cresc * ss5(u); }
    else if (nt.d > .6 && (st.swell ?? 1)) for (let i = idx(nt.t); i < idx(end + .3); i++) { const u = clamp((tm(i) - nt.t) / nt.d, 0, 1); eadd[i] += (st.swell ?? 1) * (.07 * Math.sin(Math.PI * Math.min(1, u * 1.15)) - .05 * u * u); }
    if (nt.x.eff) for (let i = idx(nt.t - .03); i < idx(end); i++) eadd[i] += nt.x.eff * smooth((tm(i) - nt.t + .03) / .06) * (1 - smooth((tm(i) - end + .05) / .05));
    const rs = nt.x.rasp ?? (acc >= .6 ? (st.rasp ?? 0) : (st.rasp ?? 0) * .25); if (rs) for (let i = idx(nt.t - .02); i < idx(nt.t + Math.min(nt.d, .5)); i++) { const t = tm(i) - nt.t; raspE[i] = Math.max(raspE[i], rs * smooth((t + .02) / .04) * Math.exp(-Math.max(0, t - .03) / (nt.x.raspT ?? .11)) * (.75 + .5 * Math.abs(nz[6](tm(i), 14)))); }
    if (nt.x.flip) for (const [t0, t1, cst, rdx] of nt.x.flip) for (let i = idx(nt.t + t0 - .03); i < idx(nt.t + t1 + .05); i++) { const u = smooth((tm(i) - nt.t - t0 + .012) / .024) * (1 - smooth((tm(i) - nt.t - t1) / .035)); cents[i] += cst * u; rdAdd[i] += rdx * u; gdb[i] -= 3.5 * u; aspAdd[i] += .5 * u; }
    if (nt.x.fry) { const L = nt.x.fry; for (let i = idx(nt.t - L - .02); i < idx(nt.t + .03); i++) { const t = tm(i) - nt.t, u = smooth((t + L + .02) / .02) * (1 - smooth((t + .012) / .036)); cents[i] = lerp(cents[i], 100 * (midi('D2') + 1.5 * nz[6](tm(i), 20)), u); gate[i] = Math.max(gate[i], .5 * u); s.jit[i] += 4 * u; rdAdd[i] -= .4 * u; } }
  });
  // breaths before phrases
  const brs = []; notes.forEach((nt, k) => { if (!nt.start) return; const pr = notes[k - 1], room = pr ? nt.c0 - (pr.offT + .10) : .8, L = Math.min(st.brLen ?? .36, room - .05); if (L > .12 && (st.breathIn ?? 1) > 0 && !(nt.x.nobreath)) { const gp = nt.c === '' || !nt.c ? .10 : .045; brs.push([nt.c0 - L - gp, nt.c0 - gp, (st.breathIn ?? 1)]); } });
  const brE = F(0); for (const [a, b, g] of brs) for (let i = idx(a); i <= idx(b); i++) { const u = clamp((tm(i) - a) / (b - a), 0, 1); brE[i] = g * Math.pow(Math.max(0, Math.sin(Math.PI * Math.pow(u, 1.5))), 1.1) * (.75 + .25 * nz[2](tm(i), 9)); }
  s.eff0 = F(.4); for (let i = 0; i < n; i++) s.eff0[i] = clamp(s.eff[i] + eadd[i], .05, 1.4);
  const lv = st.gain ?? 0, brth = s.brth = F(0);
  for (let i = 0; i < n; i++) {
    const t = tm(i), e = clamp(s.eff[i] + eadd[i], .05, 1.4), g = gate[i], f0 = 440 * Math.pow(2, (cents[i] / 100 - 69) / 12);
    s.f0[i] = f0; const life = st.life ?? 1, dbv = -17 + 17 * Math.min(e, 1.15) + gdb[i] + lv + life * (.9 * nz[0](t + 50, 2.3) + .7 * nz[1](t + 70, 5.5)) + 1.1 * vibPhase[i], amp = Math.pow(10, dbv / 20) * g * cdip[i];
    const hb = clamp(aspC[i], 0, 1);
    s.ee[i] = WH ? 0 : amp * vfade[i] * (1 - .9 * hb * .92); s.rasp[i] = raspE[i];
    const hv = smooth((cents[i] / 100 - 65) / 9); s.rd[i] = clamp((st.rd ?? 0) + 2.12 - 1.12 * e + (st.head ?? .45) * hv + rdAdd[i] + (1 - Math.min(1, g)) * 1.0 + (1 - Math.min(1, cdip[i])) * (st.dipRd ?? 0) + hb * .8 + .10 * nz[5](t, 2.1) + .06 * nz[6](t + 9, 6) - vibPhase[i] * .08 - .045 * clamp(gdb[i], -6, 4), .55, 2.7);
    const br = (st.breath ?? .5), lvA = Math.pow(10, (-17 + 17 * Math.min(e, 1.1) + lv) / 20);
    s.asp[i] = Math.max(amp * (.004 + .070 * br * (st.brK2 ?? 1.5) + aspAdd[i] * .06), hb * lvA * (.16 + .03 * br)) * (WH ? 0 : 1); brth[i] = brE[i] * (st.brAmp ?? .035) * (st.brK ?? 1);
    s.aspfl[i] = clamp((st.aspFl ?? .3) + hb * .7, 0, 1);
    s.jit[i] *= (st.jit ?? 1) * (1 + .5 * (1 - g));
    s.tilt[i] = (st.tiltK ?? 1) * (2300 + 4200 * Math.min(e, 1.2)) * (.35 + .65 * g * g) * (1 - .3 * hv) * Math.pow(10, clamp(gdb[i], -8, 5) / 40); s.air[i] = amp * (st.air ?? .008) * (1 + 1.2 * br); s.mvf[i] = (st.mvfK ?? 1) * (e < .65 ? lerp(8600, 10000, smooth((e - .3) / .35)) : lerp(10000, 12600, smooth((e - .65) / .4))) * (1 - .25 * hv) * (.7 + .3 * g);
    s.bwo[i] = 94 + 95 * clamp(s.rd[i] - .7, 0, 2); s.bwc[i] = 32 + 14 * clamp(s.rd[i] - 1, 0, 2);
    // tract opening follows pitch and effort: F1 stays above F0
    s.eff[i] = clamp((f0 - 225) / 215 * .95 + .07 * vibPhase[Math.max(0, i - 20)] + (e - .5) * .55 + (st.open ?? 0) + .16 * life * nz[2](t + 30, 1.9) + evo[i], 0, 1.98);
  }
  if (o.airy) for (let i = 0; i < n; i++) { const hb = clamp(aspC[i], 0, 1), g = gate[i] * cdip[i]; s.asp[i] = Math.max(g, hb * .8) * .13 * Math.pow(10, clamp(gdb[i], -8, 4) / 20) * Math.pow(10, (-17 + 17 * Math.min(s.eff0 ? s.eff0[i] : .4, 1)) / 20); s.aspfl[i] = .8; s.ee[i] *= .06; s.rd[i] = 2.7; s.mvf[i] = 1500; s.air[i] *= .4; }
  for (const k in fr) { const f = fr[k]; const g0 = Math.pow(10, (st.fricDb ?? -6) / 20) * (o.airy ? .45 : 1); let any = false; for (let i = 0; i < n; i++) if (f.amp[i] > 0) { const e = clamp(s.ee ? 0 : 0, 0, 1); f.amp[i] *= f.gn * g0 * Math.pow(10, (-9 + 9 * Math.min(1, st.eff[1]) + lv) / 20); any = true; } if (any) s.fric.push(f); }
  if (WH) {                              // unvoiced: the open glottis is the voice
    for (let i = 0; i < n; i++) { const g = Math.max(gate[i] * cdip[i], aspC[i] * .9), e = s.eff[i]; s.ee[i] = 0; s.asp[i] = g * .12 * Math.pow(10, gdb[i] / 20); s.aspfl[i] = 1; }
  }
  return { s, t0: T0, notes };
}
