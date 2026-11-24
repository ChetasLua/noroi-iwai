// voice_tract.mjs : one-dimensional vocal tract as a chain of 44 lossy tubes (Sondhi-Schroeter losses with yielding walls),
// piriform side cavity, nasal branch with a sinus, radiating lips.  Gives complex transfer functions on a warped frequency grid.
import { SR } from './voice_dsp.mjs';
export const NSEC = 44, CS = 35300, RHO = 0.00114, RC = RHO * CS;
export const NFFT = 4096, DF = SR / NFFT;
// frequency grid: dense where formants are narrow
const gb = [];
for (let k = 1; k < 110; k += 1) gb.push(k);
for (let k = 110; k < 430; k += 2) gb.push(k);
for (let k = 430; k < 1300; k += 6) gb.push(k);
gb.push(1300);
export const GBIN = Int32Array.from(gb), NG = GBIN.length, GF = Float64Array.from(gb, k => k * DF), KMAX = 1300;   // 15.2 kHz
// bin -> grid interpolation
export const BI = new Int32Array(KMAX + 1), BW = new Float64Array(KMAX + 1);
{ let g = 0; for (let k = 0; k <= KMAX; k++) { while (g < NG - 2 && GBIN[g + 1] < k) g++; BI[k] = g; BW[k] = Math.max(0, Math.min(1, (k - GBIN[g]) / (GBIN[g + 1] - GBIN[g]))); } }

const NASAL = [0, .7, 1.1, 1.8, 2.6, 3.2, 3.5, 3.4, 3.1, 2.8, 2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.25, 1.1, .95, .85, .8, .9];   // [0],[1] coupled to the velum port
export const VELUM = 22;                 // branch above section 22 (about 8.3 cm above the glottis)
const PIRI = 4;                          // piriform fossae join above section 4
export class Tract {
  constructor(Lt = 16.4, opt = {}) {
    this.Lt = Lt; this.l = Lt / NSEC; const l = this.l;
    const mk = () => new Float64Array(NG);
    this.chr = mk(); this.chi = mk(); this.gsr = mk(); this.gsi = mk(); this.sgr = mk(); this.sgi = mk(); this.pyr = mk(); this.pyi = mk(); this.nhr = mk(); this.nhi = mk(); this.ngr = mk(); this.ngi = mk(); this.nsr = mk(); this.nsi = mk(); this.snr = mk(); this.sni = mk();
    const a = 130 * Math.PI, b = (30 * Math.PI) ** 2, c1 = opt.c1 ?? 4, w02 = (406 * Math.PI) ** 2, r = opt.r ?? 408;
    const lps = opt.piri2 ?? [1.55, 2.02], aps = opt.piriA2 ?? [.26, .22];
    for (let g = 0; g < NG; g++) {
      const w = 2 * Math.PI * GF[g];
      const calc = (len, lossk) => {
        // alpha = sqrt(j w c1)
        const ar = Math.sqrt(w * c1 * lossk * lossk / 2), ai = ar;
        // beta = j w w02 / ((j w + a) j w + b) + alpha
        const dr = b - w * w, di = a * w, dd = dr * dr + di * di; let br = (w * w02 * di) / dd + ar, bi = (w * w02 * dr) / dd + ai;
        // gamma = sqrt((r + j w)/(beta + j w))
        const nr = r * lossk, ni = w, er = br, ei = bi + w, ee = er * er + ei * ei; let qr = (nr * er + ni * ei) / ee, qi = (ni * er - nr * ei) / ee; const m = Math.sqrt(Math.hypot(qr, qi)), ph = Math.atan2(qi, qr) / 2; const gr = m * Math.cos(ph), gi = m * Math.sin(ph);
        // sigma = gamma (beta + j w)
        const sr = gr * er - gi * ei, si = gr * ei + gi * er, xr = sr * len / CS, xi = si * len / CS;
        const chr = Math.cosh(xr) * Math.cos(xi), chi = Math.sinh(xr) * Math.sin(xi), shr = Math.sinh(xr) * Math.cos(xi), shi = Math.cosh(xr) * Math.sin(xi);
        const gg = gr * gr + gi * gi;
        return { chr, chi, gsr: gr * shr - gi * shi, gsi: gr * shi + gi * shr, sgr: (shr * gr + shi * gi) / gg, sgi: (shi * gr - shr * gi) / gg };
      };
      const us = Math.max(0, Math.min(1, (GF[g] - 900) / 1600)), lk = 1 + (opt.lk ?? 0) * us * us * (3 - 2 * us) + .6 * Math.pow(Math.max(0, (GF[g] - 3600) / 2500), 1.3);
      const o = calc(l, lk); this.chr[g] = o.chr; this.chi[g] = o.chi; this.gsr[g] = o.gsr * RC; this.gsi[g] = o.gsi * RC; this.sgr[g] = o.sgr / RC; this.sgi[g] = o.sgi / RC;
      const n = calc(l, 2.6); this.nhr[g] = n.chr; this.nhi[g] = n.chi; this.ngr[g] = n.gsr * RC; this.ngi[g] = n.gsi * RC; this.nsr[g] = n.sgr / RC; this.nsi[g] = n.sgi / RC;
      // piriform admittance (closed side tube): (A/rc) tanh/gamma = sg/ch
      for (let q = 0; q < 2; q++) { const p = calc(lps[q], 5.0), cd = p.chr * p.chr + p.chi * p.chi; this.pyr[g] += aps[q] / RC * (p.sgr * p.chr + p.sgi * p.chi) / cd; this.pyi[g] += aps[q] / RC * (p.sgi * p.chr - p.sgr * p.chi) / cd; }
      // sinus: series RLC shunt, f0 = 520 Hz, Q about 5
      const f0 = 520, Q = 4.5, Z0 = RC / .35, X = Z0 * (GF[g] / f0 - f0 / GF[g]), R = Z0 / Q, dz = R * R + X * X; this.snr[g] = R / dz; this.sni[g] = -X / dz;
    }
    this.P = [mk(), mk()]; this.U = [mk(), mk()]; this.nP = [mk(), mk()]; this.nU = [mk(), mk()]; this.nCache = new Map();
    this.zb = [mk(), mk()];
  }
  _radiate(a, Pr, Pi, Ur, Ui) {
    const R = 128 * RC / (9 * Math.PI * Math.PI * a), L = 8 * RHO / (3 * Math.PI * Math.sqrt(Math.PI * a));
    for (let g = 0; g < NG; g++) { const X = 2 * Math.PI * GF[g] * L, d = R * R + X * X; Pr[g] = R * X * X / d; Pi[g] = R * R * X / d; Ur[g] = 1; Ui[g] = 0; }
  }
  _step(a, Pr, Pi, Ur, Ui, hr, hi, gr, gi, sr, si) {
    const ia = 1 / a;
    for (let g = 0; g < NG; g++) {
      const pr = Pr[g], pi = Pi[g], ur = Ur[g], ui = Ui[g], cr = hr[g], ci = hi[g], br = gr[g] * ia, bi = gi[g] * ia, dr = sr[g] * a, di = si[g] * a;
      Pr[g] = cr * pr - ci * pi + br * ur - bi * ui; Pi[g] = cr * pi + ci * pr + br * ui + bi * ur;
      Ur[g] = dr * pr - di * pi + cr * ur - ci * ui; Ui[g] = dr * pi + di * pr + cr * ui + ci * ur;
    }
  }
  nasal(an) {                            // returns {yr,yi: input admittance, tr,ti: nostril flow per unit pressure at the port}
    const key = Math.round(Math.log2(Math.max(an, .003)) * 6); let o = this.nCache.get(key); if (o) return o;
    const av = Math.pow(2, key / 6), mk = () => new Float64Array(NG), Pr = mk(), Pi = mk(), Ur = mk(), Ui = mk();
    this._radiate(.9, Pr, Pi, Ur, Ui);
    for (let i = NASAL.length - 1; i >= 0; i--) {
      const a = i === 0 ? av : i === 1 ? .5 * (av + .7) : i === 2 ? Math.min(1.1, av * .4 + .7) : NASAL[i];
      if (i === 14) for (let g = 0; g < NG; g++) { const pr = Pr[g], pi = Pi[g]; Ur[g] += pr * this.snr[g] - pi * this.sni[g]; Ui[g] += pr * this.sni[g] + pi * this.snr[g]; }
      this._step(a, Pr, Pi, Ur, Ui, this.nhr, this.nhi, this.ngr, this.ngi, this.nsr, this.nsi);
    }
    o = { yr: mk(), yi: mk(), tr: mk(), ti: mk() };
    for (let g = 0; g < NG; g++) { const d = Pr[g] * Pr[g] + Pi[g] * Pi[g]; o.tr[g] = Pr[g] / d; o.ti[g] = -Pi[g] / d; o.yr[g] = (Ur[g] * Pr[g] + Ui[g] * Pi[g]) / d; o.yi[g] = (Ui[g] * Pr[g] - Ur[g] * Pi[g]) / d; }
    this.nCache.set(key, o); return o;
  }
  // glottal flow -> radiated flow (mouth + nostrils).  A: 44 areas glottis..lips, an: velar port area (cm2), leak: pressure leak through the walls/cheeks
  H(A, an, Hr, Hi, nasGain = 1) {
    const [Pr, Pi] = this.P, [Ur, Ui] = this.U;
    this._radiate(Math.max(A[NSEC - 1], .02), Pr, Pi, Ur, Ui);
    const n = an > .008 ? this.nasal(an) : null; let nr = null, ni = null;
    for (let i = NSEC - 1; i >= 0; i--) {
      this._step(Math.max(A[i], .0015), Pr, Pi, Ur, Ui, this.chr, this.chi, this.gsr, this.gsi, this.sgr, this.sgi);
      if (i === VELUM && n) { nr = this.nP[0]; ni = this.nP[1]; for (let g = 0; g < NG; g++) { const pr = Pr[g], pi = Pi[g]; nr[g] = nasGain * (pr * n.tr[g] - pi * n.ti[g]); ni[g] = nasGain * (pr * n.ti[g] + pi * n.tr[g]); Ur[g] += pr * n.yr[g] - pi * n.yi[g]; Ui[g] += pr * n.yi[g] + pi * n.yr[g]; } }
      if (i === PIRI) for (let g = 0; g < NG; g++) { const pr = Pr[g], pi = Pi[g]; Ur[g] += pr * this.pyr[g] - pi * this.pyi[g]; Ui[g] += pr * this.pyi[g] + pi * this.pyr[g]; }
    }
    for (let g = 0; g < NG; g++) { const ur = Ur[g], ui = Ui[g], d = ur * ur + ui * ui, tr = nr ? 1 + nr[g] : 1, ti = ni ? ni[g] : 0; Hr[g] = (tr * ur + ti * ui) / d; Hi[g] = (ti * ur - tr * ui) / d; }
  }
  // supraglottal pressure source just downstream of section j -> lip flow.  ag: glottal opening (cm2)
  HF(A, j, ag, Hr, Hi) {
    const [Pr, Pi] = this.P, [Ur, Ui] = this.U, [zr, zi] = this.zb;
    this._radiate(Math.max(A[NSEC - 1], .02), Pr, Pi, Ur, Ui);
    for (let i = NSEC - 1; i > j; i--) this._step(Math.max(A[i], .0015), Pr, Pi, Ur, Ui, this.chr, this.chi, this.gsr, this.gsi, this.sgr, this.sgi);
    const Rg = 60 / Math.max(ag, .02), Lg = RHO * .3 / Math.max(ag, .02);
    for (let g = 0; g < NG; g++) { zr[g] = Rg; zi[g] = 2 * Math.PI * GF[g] * Lg; }
    for (let i = 0; i <= j; i++) { const a = Math.max(A[i], .0015), ia = 1 / a;   // Zout = (Zin ch + B)/(Zin D + ch)
      for (let g = 0; g < NG; g++) { const cr = this.chr[g], ci = this.chi[g], br = this.gsr[g] * ia, bi = this.gsi[g] * ia, dr = this.sgr[g] * a, di = this.sgi[g] * a, x = zr[g], y = zi[g]; const nr = x * cr - y * ci + br, ni = x * ci + y * cr + bi, er = x * dr - y * di + cr, ei = x * di + y * dr + ci, e = er * er + ei * ei; zr[g] = (nr * er + ni * ei) / e; zi[g] = (ni * er - nr * ei) / e; } }
    for (let g = 0; g < NG; g++) { const dr = Pr[g] + zr[g] * Ur[g] - zi[g] * Ui[g], di = Pi[g] + zr[g] * Ui[g] + zi[g] * Ur[g], d = dr * dr + di * di; Hr[g] = dr / d; Hi[g] = -di / d; }
  }
}
// formants of an area function: lossless with wall compliance, zero crossings of U(glottis)
export function formants(A, Lt, n = 6, fmax = 6500) {
  const l = Lt / NSEC, w0 = 2 * Math.PI * 190, out = [];
  const f = fr => {
    const w = 2 * Math.PI * fr, kk = w * Math.sqrt(Math.max(1e-6, 1 - (w0 / w) ** 2)), ga = 1 / Math.sqrt(Math.max(1e-6, 1 - (w0 / w) ** 2)), x = kk * l / CS, c = Math.cos(x), s = Math.sin(x);
    let U = 1, p = w * 8 * RHO / (3 * Math.PI * Math.sqrt(Math.PI * Math.max(A[NSEC - 1], .02)));
    for (let i = NSEC - 1; i >= 0; i--) { const a = Math.max(A[i], .0015), p2 = c * p + RC * ga / a * s * U, u2 = -a / (RC * ga) * s * p + c * U; p = p2; U = u2; if (i === PIRI) { const y = .9 / (RC * ga) * Math.tan(kk * 1.75 / CS); U -= p * y; } }
    return U;
  };
  let fa = 195, va = f(fa);
  for (let fb = 205; fb < fmax && out.length < n; fb += 10) { const vb = f(fb); if (va * vb < 0 && Math.abs(va) < 50 && Math.abs(vb) < 50) { let a = fa, b = fb, x = va; for (let i = 0; i < 22; i++) { const m = (a + b) / 2, vm = f(m); if (vm * x < 0) b = m; else { a = m; x = vm; } } if (Math.abs(f((a + b) / 2)) < 5) out.push((a + b) / 2); } fa = fb; va = vb; }
  return out;
}
