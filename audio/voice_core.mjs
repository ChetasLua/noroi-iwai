// voice_core.mjs : pitch-synchronous singer.  Every glottal cycle gets its own LF pulse (own Rd, period, amplitude), is sent through the tract
// transfer function of that instant and overlap-added; breath noise is gated by the open glottis; consonant noise is born at the constriction.
import { SR, RFFT, rng, slowNoise, clamp, lerp } from './voice_dsp.mjs';
import { NSEC, NG, GF, GBIN, BI, BW, KMAX, NFFT, DF, Tract } from './voice_tract.mjs';
import { VOWELS } from './voice_vowels.mjs';
export const CR = 500, SPF = SR / CR;       // control rate, samples per control frame
export const VI = { a: 0, i: 1, u: 2, e: 3, o: 4 };
const DV = ['a', 'i', 'u', 'e', 'o'].map(v => VOWELS[v].map(A => Float64Array.from(A, Math.sqrt)));   // sqrt-area (diameter-like)

const smoothstep = (x, a, b) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
// ---------- LF model ----------
function lfSolve(Rd) {
  const Ra = Math.max(.002, (-1 + 4.8 * Rd) / 100), Rk = (22.4 + 11.8 * Rd) / 100, Rg = (Rk / 4) * (.5 + 1.2 * Rk) / (.11 * Rd - Ra * (.5 + 1.2 * Rk));
  const tp = 1 / (2 * Rg), te = tp * (1 + Rk), ta = Ra, wg = Math.PI / tp, tb = 1 - te;
  let eps = 1 / ta; for (let i = 0; i < 60; i++) { const f = eps * ta - 1 + Math.exp(-eps * tb), d = ta - tb * Math.exp(-eps * tb); eps -= f / d; }
  const ret = -(1 / (eps * ta)) * ((1 - Math.exp(-eps * tb)) / eps - tb * Math.exp(-eps * tb));
  const open = al => { const E0 = -1 / (Math.exp(al * te) * Math.sin(wg * te)); return E0 * (Math.exp(al * te) * (al * Math.sin(wg * te) - wg * Math.cos(wg * te)) + wg) / (al * al + wg * wg); };
  let lo = -30, hi = 60; for (let i = 0; i < 80; i++) { const m = (lo + hi) / 2; if (open(m) + ret > 0) lo = m; else hi = m; }   // more growth -> less positive flow before closure
  const al = (lo + hi) / 2; return { te, tp, ta, wg, tb, eps, al };
}
const RDT = []; for (let i = 0; i <= 125; i++) RDT.push(lfSolve(.35 + i * .02));
function lf(Rd, T) { const x = clamp((Rd - .35) / .02, 0, 124.999), i = x | 0, u = x - i, a = RDT[i], b = RDT[i + 1], L = k => a[k] * (1 - u) + b[k] * u; return { te: L('te') * T, tp: L('tp') * T, ta: L('ta') * T, wg: L('wg') / T, tb: L('tb') * T, eps: L('eps') / T, al: L('al') / T, T }; }
// spectrum on the grid, time origin at the closure instant
function lfSpec(p, out, oi) {
  const { te, wg, tb, eps, al, ta } = p, ea = Math.exp(al * te), sg = Math.sin(wg * te), cg = Math.cos(wg * te), E0 = -1 / (ea * sg), A1 = ea * sg, A2 = ea * wg * cg, k = -1 / (eps * ta), eb = Math.exp(-eps * tb);
  for (let g = 0; g < NG; g++) {
    const w = 2 * Math.PI * GF[g], c1 = Math.cos(w * te), s1 = Math.sin(w * te), c2 = Math.cos(w * tb), s2 = Math.sin(w * tb);
    // numerator: A1*(al - jw) - A2 + wg e^{jw te}
    const nr = A1 * al - A2 + wg * c1, ni = -A1 * w + wg * s1, dr = al * al - w * w + wg * wg, di = -2 * al * w, dd = dr * dr + di * di;
    let xr = E0 * (nr * dr + ni * di) / dd, xi = E0 * (ni * dr - nr * di) / dd;
    // return phase
    const er = eps, ei = w, ee = er * er + ei * ei, ar = 1 - eb * c2, ai = eb * s2; let yr = (ar * er + ai * ei) / ee, yi = (ai * er - ar * ei) / ee;
    // - eb (1 - e^{-jw tb})/(jw) = -eb * ( s2 - j(1-c2) )/w
    yr -= eb * s2 / w; yi += eb * (1 - c2) / w;
    xr += k * yr; xi += k * yi; out[g] = xr; oi[g] = xi;
  }
}
export { lf, lfSpec };

const fft = new RFFT(NFFT), NF2 = 2048, fft2 = new RFFT(NF2);
// cross modes and soft walls: a real throat stops behaving like a pipe above 5 kHz
const hfRoll = (fc, n) => Float64Array.from(GF, f => 1 / Math.sqrt(1 + Math.pow(f / fc, n)));
// every throat has its own small pole/zero pairs (windpipe, ventricles, teeth, sinuses): the fingerprint that keeps neighbouring partials uneven
export function fingerprint(seed, depth = 1) {
  const R = rng(seed * 7 + 3), pr = [[600, 640], [1430, 1520], [2120, 2010], [2880, 3040], [3650, 3480], [4450, 4680], [5350, 5120], [6250, 6550], [7300, 7000], [8600, 8950]], r = new Float64Array(NG).fill(1), im = new Float64Array(NG);
  for (const [z0, p0] of pr) { const k = 1 + .05 * R.g(), fz = z0 * k, fp = fz + (p0 - z0) * depth * (.7 + .6 * R.u()), bz = 60 + .022 * fz, bp = 80 + .03 * fp; for (let g = 0; g < NG; g++) { const f = GF[g], ar = fz * fz + bz * bz / 4 - f * f, ai = bz * f, br = fp * fp + bp * bp / 4 - f * f, bi = bp * f, d = br * br + bi * bi, n = (fp * fp) / (fz * fz), qr = (ar * br + ai * bi) / d * n, qi = (ai * br - ar * bi) / d * n, x = r[g] * qr - im[g] * qi; im[g] = r[g] * qi + im[g] * qr; r[g] = x; } }
  return { r, i: im };
}
const HOPF = 2;                          // tract frame every 2 control frames (4 ms)

export function areaOf(s, i, D) {        // fills D with sqrt-area at control frame i
  const e = clamp(s.eff[i], 0, 1.999), k = e | 0, u = e - k; D.fill(0);
  for (let v = 0; v < 5; v++) { const w = s.w[v][i]; if (w < 1e-4) continue; const a = DV[v][k], b = DV[v][k + 1]; for (let q = 0; q < NSEC; q++) D[q] += w * (a[q] * (1 - u) + b[q] * u); }
  for (const g of s.gest) { const m = g.m[i]; if (m < 1e-4) continue; const lo = Math.max(0, Math.ceil(g.loc - g.hw * 2.2)), hi = Math.min(NSEC - 1, Math.floor(g.loc + g.hw * 2.2)), dm = Math.sqrt(g.amin); for (let q = lo; q <= hi; q++) { const x = (q - g.loc) / g.hw, kk = m * Math.exp(-.693 * x * x * (Math.abs(x) < 1 ? x * x * .6 + .4 : 1)), c = Math.min(1, kk * 1.0); if (D[q] > dm) D[q] = D[q] * (1 - c) + dm * c; } }
  return D;
}

export function synth(s, opt = {}) {
  const N = s.n * SPF, pad = 4800, out = new Float32Array(N + pad * 2), R = rng(opt.seed ?? 1), fl = [0, 1, 2, 3, 4, 5].map(k => slowNoise((opt.seed ?? 1) * 131 + k * 17 + 5));
  const tract = opt.tract, nFr = Math.ceil(s.n / HOPF) + 1, Hs = new Array(nFr), Dprev = new Float64Array(NSEC), D = new Float64Array(NSEC), A = new Float64Array(NSEC); let have = null, anPrev = -1;
  const HFR = hfRoll(opt.hfc ?? 5200, opt.hfn ?? 6), wj = opt.white ?? 3, FP = fingerprint(opt.fp ?? 1, opt.fpd ?? 0), flutter = opt.flutter ?? 1, wall = opt.wall ?? .035;
  for (let f = 0; f < nFr; f++) {
    const i = Math.min(s.n - 1, f * HOPF), t = i / CR; areaOf(s, i, D);
    // living tract: slow random breathing of tongue height, pharynx width, lip opening
    const f1 = fl[0](t, 2.3) * .035 * flutter, f2 = fl[1](t, 3.1) * .03 * flutter, f3 = fl[2](t, 1.7) * .03 * flutter;
    for (let q = 0; q < NSEC; q++) { const x = q / (NSEC - 1); D[q] *= 1 + f1 * Math.cos(Math.PI * x) + f2 * Math.cos(2 * Math.PI * x) + f3 * Math.cos(3 * Math.PI * x); }
    const an = s.velum[i]; let dmax = 0; for (let q = 0; q < NSEC; q++) dmax = Math.max(dmax, Math.abs(D[q] - Dprev[q]) / (D[q] + .02));
    if (have && dmax < .004 && Math.abs(an - anPrev) < .004) { Hs[f] = have; continue; }
    Dprev.set(D); anPrev = an; let amin = 9; for (let q = 0; q < NSEC; q++) { A[q] = D[q] * D[q]; if (q > 18 && A[q] < amin) amin = A[q]; }
    const h = { r: new Float64Array(NG), i: new Float64Array(NG), amin }; tract.H(A, an, h.r, h.i, opt.nasGain ?? 1);
    for (let g = 0; g < NG; g++) { const x = h.r[g], y = h.i[g]; h.r[g] = (x * FP.r[g] - y * FP.i[g]) * HFR[g]; h.i[g] = (x * FP.i[g] + y * FP.r[g]) * HFR[g]; }
    const wl = wall * clamp(1 - amin / .08, 0, 1);      // closed mouth: voicing leaks through cheeks and throat
    if (wl > 0) for (let g = 0; g < NG; g++) { const x = GF[g] / 260; h.r[g] += wl / (1 + x * x * x * x) * 4; }
    Hs[f] = have = h;
  }
  // a singer balances vowels: predict each vowel's loudness at its pitch and even most of it out
  const CH = 8, nC = Math.ceil(s.n / CH) + 2, comp = new Float32Array(nC).fill(1), kap = opt.balance ?? .62;
  if (kap > 0) { const hr = new Float64Array(NG), hi = new Float64Array(NG), gr = new Float64Array(NG), gi = new Float64Array(NG), sv = { ...s, gest: [] }; let last = null, lf0 = 0, lv = 1;
    for (let c = 0; c < nC; c++) { const i = Math.min(s.n - 1, c * CH); if (s.ee[i] < 1e-5 && !opt.whisperBal) { comp[c] = lv; continue; } areaOf(sv, i, D); for (let q = 0; q < NSEC; q++) A[q] = D[q] * D[q]; tract.H(A, 0, hr, hi); const f0 = s.f0[i] || 200; lfSpec(lf(clamp(s.rd[i], .4, 2.7), 1 / f0), gr, gi); let P = 0; for (let k = 1; k * f0 < 5000; k++) { const x = k * f0 / DF, kk = Math.min(KMAX - 1, Math.round(x)), g = BI[kk], w = .25 + .75 * Math.min(1, k * f0 / 700); P += w * (hr[g] ** 2 + hi[g] ** 2) * (gr[g] ** 2 + gi[g] ** 2); } const rms = f0 * Math.sqrt(2 * P); lv = clamp(Math.pow(.42 / Math.max(.02, rms), kap), .3, 2.6); comp[c] = lv; } 
    for (let r = 0; r < 2; r++) for (let c = 1; c < nC - 1; c++) comp[c] = .25 * comp[c - 1] + .5 * comp[c] + .25 * comp[c + 1]; }
  const cAt = tt => { const x = clamp(tt * CR / CH, 0, nC - 1.001), i = x | 0, u = x - i; return comp[i] * (1 - u) + comp[i + 1] * u; };
  // ---------- voiced pulses ----------
  const tmpN = new Float64Array(NFFT), Wr = new Float64Array(NFFT / 2 + 1), Wi = new Float64Array(NFFT / 2 + 1), Vw = new Float64Array(NG), Nw = new Float64Array(NG), nk = opt.nk ?? 0, nfl = opt.nfl ?? .22, sdN = Array.from({ length: 22 }, () => R.g());
  const Gr = new Float64Array(NG), Gi = new Float64Array(NG), Xr = new Float64Array(NFFT / 2 + 1), Xi = new Float64Array(NFFT / 2 + 1), y = new Float64Array(NFFT), Sr = new Float64Array(NG), Si = new Float64Array(NG);
  const aspEnv = new Float32Array(N + pad), at = (a, t) => { const x = clamp(t * CR, 0, s.n - 1.001), i = x | 0, u = x - i; return a[i] * (1 - u) + a[i + 1] * u; };
  let t = 0, np = 0; const tEnd = s.n / CR; let jitLP = 0, shLP = 0;
  const LB = 22, lbF = Float64Array.from({ length: LB }, (_, b) => 120 * Math.pow(1.262, b)), lgN = Array.from({ length: LB }, (_, b) => slowNoise((opt.seed ?? 1) * 389 + b * 7 + 3)), ldN = Array.from({ length: LB }, (_, b) => slowNoise((opt.seed ?? 1) * 577 + b * 11 + 9)), lg = new Float64Array(LB), ld = new Float64Array(LB), lvA = opt.live ?? 1, lbI = new Int32Array(NG), lbW = new Float64Array(NG);
  for (let g = 0, b = 0; g < NG; g++) { while (b < LB - 2 && lbF[b + 1] < GF[g]) b++; lbI[g] = b; lbW[g] = clamp((GF[g] - lbF[b]) / (lbF[b + 1] - lbF[b]), 0, 1); }
  const bands = [2600, 4200, 6200, 9000, 14000], bd = [0, 0, 0, 0, 0], bg = [0, 0, 0, 0, 0];
  while (t < tEnd) {
    const f0 = at(s.f0, t); let T = 1 / Math.max(45, f0); const tm = t + T * .6, ee0 = at(s.ee, tm), rasp = at(s.rasp, tm), rdv = at(s.rd, tm), aspv = at(s.asp, tm) * cAt(tm), jb = at(s.jit, tm);
    // natural perturbation: white + slightly correlated jitter/shimmer
    jitLP = jitLP * .6 + R.g() * .8; shLP = shLP * .55 + R.g() * .83;
    let jit = jb * (.00022 * wj * jitLP) + jb * .0011 * (fl[3](t, 11) + .8 * fl[4](t, 21)), sh = jb * (.0025 * wj * shLP + .030 * fl[5](t, 9) + .02 * fl[4](t + 9, 17)); let sub = 0;
    if (rasp > .01) { const ph = np % 2; sub = ph ? -1 : 1; const r2 = R.g(); jit += rasp * (.010 * sub + .012 * r2); sh += rasp * (.16 * sub + .14 * R.g()); }
    T *= 1 + clamp(jit, -.09, .09);
    const cg = cAt(tm), ee = ee0 * cg * Math.max(.08, 1 + sh), rd = clamp(rdv + .004 * wj * R.g() * jb + .04 * fl[3](t + 5, 9) * jb - rasp * .08 * sub, .4, 2.75), p = lf(rd, T);
    const i0 = Math.round(t * SR), i1 = Math.round((t + T) * SR);
    // breath envelope follows the open glottis
    { const floor = at(s.aspfl, tm), o0 = p.tp * .35, o1 = p.te + p.ta * 1.5; for (let i = i0; i < i1 && i < aspEnv.length; i++) { const x = (i / SR - t); let g = x < o0 ? 0 : x > o1 ? 0 : Math.sin(Math.PI * (x - o0) / (o1 - o0)); aspEnv[i] = aspv * (floor + (1 - floor) * Math.pow(Math.max(0, g), .8)); } }
    if (ee > 1e-5) {
      lfSpec(p, Gr, Gi);
      const tcl = t + p.te, fx = clamp(tcl * CR / HOPF, 0, nFr - 1.001), fi = fx | 0, fu = fx - fi, h0 = Hs[fi], h1 = Hs[fi + 1];
      const tl = at(s.tilt, tm), mvf = at(s.mvf, tm);
      for (let b = 0; b < 5; b++) { const k = (opt.hfj ?? 1) * jb * [0, .000012, .000022, .00003, .00003][b] * (1 + 2 * rasp); bd[b] = R.g() * k; bg[b] = Math.exp(R.g() * [.02, .06, .10, .12, .12][b] * (opt.hfs ?? 1)); }
      let bb = 0; for (let b = 0; b < LB; b++) { const f = lbF[b], q = smoothstep(f, 500, 3200); lg[b] = lvA * (.10 + .16 * q) * (.9 * lgN[b](t, 6.5) + .45 * lgN[b](t + 31, 13)); ld[b] = (opt.disp ?? 0) * 1e-6 * ((4 + 130 * smoothstep(f, 1000, 3000)) * ldN[b](t, 3.2) + 230 * smoothstep(f, 1200, 3400) * sdN[b]); }
      for (let g = 0; g < NG; g++) {
        const f = GF[g]; let hr = h0.r[g] * (1 - fu) + h1.r[g] * fu, hi = h0.i[g] * (1 - fu) + h1.i[g] * fu;
        // extra tilt pole and voiced-band limit
        const q = f / tl, tr = 1 / (1 + q * q), ti = -q * tr, v = f / mvf, v3 = v * v * v, vd = 1 / Math.sqrt(1 + v3 * v3); Vw[g] = vd; Nw[g] = v3 * vd * nk; const vw = 1;
        while (bb < 4 && f > bands[bb]) bb++; const b0 = bb === 0 ? 0 : bands[bb - 1], bu = clamp((f - b0) / (bands[bb] - b0), 0, 1), dl = bb === 0 ? bd[0] * bu : lerp(bd[bb - 1], bd[bb], bu), gn = (bb === 0 ? lerp(1, bg[0], bu) : lerp(bg[bb - 1], bg[bb], bu)) * vw * ee;
        const lb = lbI[g], lw = lbW[g], lgg = Math.exp(lg[lb] * (1 - lw) + lg[lb + 1] * lw), ldd = ld[lb] * (1 - lw) + ld[lb + 1] * lw; const ph = -2 * Math.PI * f * (dl + ldd), gn2 = gn * lgg, cp = Math.cos(ph) * gn2, sp = Math.sin(ph) * gn2;
        const ar = Gr[g] * tr - Gi[g] * ti, ai = Gr[g] * ti + Gi[g] * tr, br = ar * hr - ai * hi, bi = ar * hi + ai * hr; Sr[g] = br * cp - bi * sp; Si[g] = br * sp + bi * cp;
      }
      // place: closure instant at sample position c = tcl*SR ; buffer start at floor(c) - pre
      const c = tcl * SR, pre = Math.ceil(p.te * SR) + 24, bs = Math.floor(c) - pre, off = c - bs;     // off >= pre
      // upper half of the spectrum: noise, born while the glottis is open, through the same throat, at the same level the partials would have had
      if (nk > 0) { tmpN.fill(0); const z0 = off - p.te * SR, n1 = Math.min(NFFT - 1, Math.ceil(z0 + T * SR)), o0 = p.tp * .3, o1 = p.te + p.ta * 2.5; let e2 = 0; for (let i = Math.max(0, Math.floor(z0)); i < n1; i++) { const x = (i - z0) / SR, w = x < o0 || x > o1 ? 0 : Math.sin(Math.PI * (x - o0) / (o1 - o0)), en = nfl + (1 - nfl) * w; tmpN[i] = R.g() * en; e2 += en * en; } const q = 1 / Math.sqrt(e2 || 1); for (let i = Math.max(0, Math.floor(z0)); i < n1; i++) tmpN[i] *= q; fft.forward(tmpN, Wr, Wi); }
      Xr.fill(0); Xi.fill(0); const dth = -2 * Math.PI * off / NFFT;
      let cr = 1, ci = 0; const dr = Math.cos(dth), di = Math.sin(dth);
      for (let k = 1; k <= KMAX; k++) {
        const ncr = cr * dr - ci * di; ci = cr * di + ci * dr; cr = ncr; const g = BI[k], u = BW[k], sr = Sr[g] * (1 - u) + Sr[g + 1] * u, si = Si[g] * (1 - u) + Si[g + 1] * u; let w = k < 1150 ? 1 : (KMAX - k) / 150;
        const vv = (Vw[g] * (1 - u) + Vw[g + 1] * u), nn = (Nw[g] * (1 - u) + Nw[g + 1] * u), er = (vv * cr + nn * Wr[k]) * w, ei = (vv * ci + nn * Wi[k]) * w; Xr[k] = sr * er - si * ei; Xi[k] = sr * ei + si * er;
        if ((k & 255) === 0) { const th = dth * k; cr = Math.cos(th); ci = Math.sin(th); }
      }
      fft.inverse(Xr, Xi, y);
      // damping: extra losses while the glottis is open
      const dBo = at(s.bwo, tm), dBc = at(s.bwc, tm), aO = Math.exp(-Math.PI * dBo / SR), aC = Math.exp(-Math.PI * dBc / SR), per = T * SR, cl = p.tb * SR, n0 = off; let wv = 1, phs = 0; const len = Math.min(NFFT, pre + 2700);
      const o = bs + pad, fadeS = len - 500;
      for (let i = 0; i < len; i++) { if (i > n0) { phs++; if (phs > per) phs -= per; wv *= phs < cl ? aC : aO; } const j = o + i; if (j >= 0 && j < out.length) out[j] += y[i] * wv * (i > fadeS ? (len - i) / 500 : 1) * SR; }
    }
    t += T; np++;
  }
  // ---------- noise paths ----------
  const tmp = new Float64Array(NF2), Nr = new Float64Array(NF2 / 2 + 1), Ni = new Float64Array(NF2 / 2 + 1), HOPN = 96, WIN = 384; const win = Float64Array.from({ length: WIN }, (_, i) => .5 - .5 * Math.cos(2 * Math.PI * (i + .5) / WIN)), wsum = WIN / 2 / HOPN;
  const noiseThrough = (sig, hAt, shape) => {
    for (let b = -WIN; b < N; b += HOPN) {
      let en = 0; tmp.fill(0); for (let i = 0; i < WIN; i++) { const j = b + i; if (j < 0 || j >= N) continue; const v = sig[j]; if (v !== 0) { tmp[i] = v * win[i] / wsum; en += v * v; } } if (en < 1e-14) continue;
      const h = hAt((b + WIN / 2) / SR); if (!h) continue; fft2.forward(tmp, Nr, Ni);
      for (let k = 0; k <= NF2 / 2; k++) { const kk = k * 2; if (kk > KMAX || k === 0) { Nr[k] = Ni[k] = 0; continue; } const g = BI[kk], u = BW[kk], hr = h.r[g] * (1 - u) + h.r[g + 1] * u, hi = h.i[g] * (1 - u) + h.i[g + 1] * u, sh = shape(kk * DF) * (kk > 1150 ? (KMAX - kk) / 150 : 1); const r = (Nr[k] * hr - Ni[k] * hi) * sh; Ni[k] = (Nr[k] * hi + Ni[k] * hr) * sh; Nr[k] = r; }
      fft2.inverse(Nr, Ni, tmp); for (let i = 0; i < NF2; i++) { const j = b + i + pad; if (j >= 0 && j < out.length) out[j] += tmp[i] * (i > NF2 - 256 ? (NF2 - i) / 256 : 1); }
    }
  };
  const hAtG = tt => { const f = clamp(Math.round(tt * CR / HOPF), 0, nFr - 1); return Hs[f]; };
  { const sig = new Float32Array(N); for (let i = 0; i < N; i++) sig[i] = R.g() * aspEnv[i]; const hp = opt.aspHp ?? 800, lp = opt.aspLp ?? 5200; noiseThrough(sig, hAtG, f => { const a = f / hp, b = f / lp; return a * a / (1 + a * a) / Math.sqrt(1 + b * b * b * b); }); }
  { const kA = opt.air ?? 1; if (kA > 0 && s.air) { const sig = new Float32Array(N); for (let i = 0; i < N; i++) { const x = i / SPF, k = Math.min(s.n - 2, x | 0), a = s.air[k] * (1 - x + k) + s.air[k + 1] * (x - k); if (a > 1e-7) { const fh = Hs[Math.min(nFr - 1, (i / SPF / HOPF) | 0)], op = clamp((fh.amin - .03) / .16, 0, 1); if (op <= 0) continue; const e = aspEnv[i] / (1e-6 + at(s.asp, i / SR)); sig[i] = R.g() * a * op * kA * (.55 + .6 * Math.min(1.2, e)); } } const one = { r: new Float64Array(NG).fill(1), i: new Float64Array(NG) }; const FPa = fingerprint((opt.fp ?? 1) + 77, 1.6); noiseThrough(sig, () => FPa, f => { const a = f / 2700, b = f / 12500; return Math.pow(a, 3) / (1 + Math.pow(a, 3)) / (1 + b * b * b * b); }); } }
  if (s.brth) { const sig = new Float32Array(N); let any = false; for (let i = 0; i < N; i++) { const x = i / SPF, k = Math.min(s.n - 2, x | 0), a = s.brth[k] * (1 - x + k) + s.brth[k + 1] * (x - k); if (a > 1e-6) { sig[i] = R.g() * a; any = true; } } if (any) noiseThrough(sig, hAtG, f => { const a = f / 950, b = f / 3600, a4 = a * a * a * a; return a4 / (1 + a4) / (1 + b * b); }); }
  for (const fr of s.fric) {
    const sig = new Float32Array(N); let any = false, lastI = -1; for (let i = 0; i < N; i++) { const x = i / SPF, k = Math.min(s.n - 2, x | 0), a = fr.amp[k] * (1 - x + k) + fr.amp[k + 1] * (x - k); if (a > 1e-6) { let v = R.g() * a; if (fr.mod) { v *= 1 - fr.mod + fr.mod * Math.min(1, aspEnv[i] / (1e-6 + at(s.asp, i / SR))); } sig[i] = v; any = true; } }
    if (!any) continue; const cache = new Map();
    const hAt = tt => { const f = clamp(Math.round(tt * CR / HOPF), 0, nFr - 1); let h = cache.get(f); if (!h) { areaOf(s, Math.min(s.n - 1, f * HOPF), D); let mn = 99; for (let q = 0; q < NSEC; q++) A[q] = Math.max(D[q] * D[q], q >= 24 ? fr.amn ?? .02 : .05); h = { r: new Float64Array(NG), i: new Float64Array(NG) }; tract.HF(A, fr.j, fr.ag ?? .2, h.r, h.i); cache.set(f, h); } return h; };
    const fc = fr.fc ?? 5000, hpf = fr.hp ?? 500, sl = fr.slope ?? 1, hpn = fr.hpn ?? 1;
    noiseThrough(sig, hAt, f => { const q = f / fc, z = f / hpf; return (f / 1000) / Math.pow(1 + q * q, sl) * Math.pow(z * z / (1 + z * z), hpn); });
  }
  return { y: out, pad };
}
