"""Collects every picture event of the film for the sound department.   ./.mm/env/bin/python tools/events.py  -> out/events.json
Each shots/<id>.py that exists gives Shot.EVENTS (read as a literal; imported only when it is computed); film offsets are added.  Shots without a module keep the draft cues of audio/events_draft.json."""
import ast, json, os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import film
def literal(path):
    try: tree = ast.parse(open(path).read())
    except SyntaxError: return 'BROKEN'        # the artist is editing it right now
    for c in tree.body:
        if isinstance(c, ast.ClassDef) and c.name == 'Shot':
            for s in c.body:
                if isinstance(s, ast.Assign) and any(getattr(t, 'id', '') == 'EVENTS' for t in s.targets):
                    try: return ast.literal_eval(s.value)
                    except Exception:
                        try: return eval(compile(ast.Expression(s.value), path, 'eval'), {'dict': dict, 'list': list, 'range': range, 'round': round, 'int': int, 'float': float, 'min': min, 'max': max, 'len': len, 'zip': zip, 'enumerate': enumerate, '__builtins__': {}})
                        except Exception: return None
    return None
def imported(sid):
    code = "import sys,json;sys.path.insert(0,%r);import importlib;m=importlib.import_module('shots.%s');print('EVENTS='+json.dumps(m.Shot.EVENTS,default=str))" % (ROOT, sid)
    try:
        r = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=240)
        for l in r.stdout.splitlines():
            if l.startswith('EVENTS='): return json.loads(l[7:])
        print('  import failed', sid, r.stderr.strip().splitlines()[-1:] )
    except subprocess.TimeoutExpired: print('  import timed out', sid)
    return None
OUT = os.path.join(ROOT, 'out/events.json'); last = json.load(open(OUT))['events'] if os.path.exists(OUT) else []
draft = json.load(open(os.path.join(ROOT, 'audio/events_draft.json'))); ev, info = [], {}
for sid, a, n in film.SHOTS:
    p = os.path.join(ROOT, 'shots', sid + '.py'); E = None
    if os.path.exists(p):
        E = literal(p)
        if E is None: E = imported(sid)
        if E == 'BROKEN' or E is None:       # keep the last good list of this shot
            E = None; keep = [e for e in last if e['shot'] == sid and e['source'] == 'shot']
            if keep: E = [(e['lf'], e['kind'], e['opts']) for e in keep]; print('  kept last good events of', sid)
    info[sid] = dict(start=a, n=n, module=os.path.exists(p), events=None if E is None else len(E))
    if E is not None:
        for e in E:
            lf, kind = float(e[0]), str(e[1]); d = dict(e[2]) if len(e) > 2 and isinstance(e[2], dict) else (dict(note=str(e[2])) if len(e) > 2 else {})
            ev.append(dict(shot=sid, lf=lf, f=a + lf, t=round((a + lf) / 24, 5), kind=kind, source='shot', opts=json.loads(json.dumps(d, default=str))))
    else: ev += [e for e in draft if e['shot'] == sid]
ev.sort(key=lambda e: e['f']); os.makedirs(os.path.join(ROOT, 'out'), exist_ok=True)
json.dump(dict(fps=24, frames=film.N, shots=info, events=ev), open(OUT, 'w'), ensure_ascii=False, indent=0)
print('shots with picture events:', [k for k, v in info.items() if v['events'] is not None] or 'none', '| draft:', sum(1 for v in info.values() if v['events'] is None), 'shots |', len(ev), 'events')
