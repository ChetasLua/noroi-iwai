// The effects library of the film: original synthesis, no samples.  import { SFX, render } from './sfx_lib.mjs'
//   render(name, { seed, size, pan, dur, v, chord, ... }) -> St { l, r, n, sync }    (reference level: -18 LUFS in its loudest 100 ms, peak <= -1.5 dBFS)
//   `sync` = seconds from the first sample to the moment that must sit on the picture's frame.
//   CLI:  node audio/sfx_lib.mjs list            names and one-line descriptions
//         node audio/sfx_lib.mjs demo            out/sfx_demo.wav + out/sfx_demo.txt (every effect in order)
//         node audio/sfx_lib.mjs one NAME out.wav '{"seed":2}'
export { SFX, render, CH, def } from './sfx_core.mjs';
import { SFX, render } from './sfx_core.mjs';
import './sfx_1.mjs';
import './sfx_2.mjs';
import './sfx_3.mjs';
import './sfx_4.mjs';
import { SR, N, St, writeWav, db } from './sfx_dsp.mjs';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url)), out = path.join(here, '..', 'out');
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [cmd, a, b, c] = process.argv.slice(2);
  if (cmd === 'list') for (const m of Object.values(SFX)) console.log(m.name.padEnd(17), m.d);
  else if (cmd === 'one') { const s = render(a, c ? JSON.parse(c) : {}); writeWav(b, s.l, s.r, 16); console.log(a, (s.n / SR).toFixed(2) + 's'); }
  else if (cmd === 'demo') {
    const all = [], rows = []; let t = .5; const add = (name, o, label) => { const s = render(name, o); all.push([t, s]); rows.push(`${(t + s.sync).toFixed(2).padStart(7)}  ${(label ?? name).padEnd(20)} ${SFX[name].d}`); t += s.n / SR + .45; };
    const variants = { eye_tear: [{ seed: 1 }, { seed: 2, v: 1 }], panel_slam: [{ v: 0 }, { v: 1 }, { v: 2 }], footstep_road: [{ shoe: 0 }, { shoe: 1 }, { shoe: 2 }, { shoe: 3 }], curse_eye: [0, 1, 2, 3, 4, 5].map(v => ({ v })), todo_clap: [0, 1, 2, 3].map(v => ({ v, seed: v + 1 })), blow: [0, 1, 2, 3].map(v => ({ v })), rain: [0, 1, 2].map(v => ({ v })), black_flash: [{ seed: 1, size: 1.2 }, { seed: 2, size: .9, len: .55 }], block_die: [0, 1, 2, 3].map(v => ({ v, seed: v + 1 })) };
    for (const name of Object.keys(SFX)) { const v = variants[name] ?? [{}]; v.forEach((o, i) => add(name, o, v.length > 1 ? `${name} #${i + 1}` : name)); }
    const st = new St(t + .5, Float32Array); for (const [at, s] of all) st.addSt(s, { at, g: 1 }); fs.mkdirSync(out, { recursive: true }); writeWav(path.join(out, 'sfx_demo.wav'), st.l, st.r, 24); fs.writeFileSync(path.join(out, 'sfx_demo.txt'), 'seconds  effect (the time is the moment that sits on the picture frame)\n' + rows.join('\n') + '\n'); console.log(rows.length, 'entries,', t.toFixed(1), 's');
  }
}
