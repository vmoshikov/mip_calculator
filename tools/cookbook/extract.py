"""Собирает cookbook.json из скачанных config/benchmark сниппетов SGLang cookbook (каталог configs/)."""
import json, re, glob, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jsx2json import tokenize_to_json

def grab(src, key, start=0):
    m = re.search(r'\n\s*' + key + r'\s*:\s*', src[start:])
    if not m: return None
    i = start + m.end()
    open_c = src[i]; close_c = {'{':'}','[':']'}.get(open_c)
    if not close_c: return None
    depth=0; j=i; in_str=None
    while j < len(src):
        c = src[j]
        if in_str:
            if c == '\\': j += 2; continue
            if c == in_str: in_str = None
        elif src.startswith('//', j): j = src.find('\n', j); continue
        elif c in '"\'`': in_str = c
        elif c == open_c: depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0: break
        j += 1
    return src[i:j+1]

def parse(lit):
    try: return json.loads(tokenize_to_json(lit))
    except Exception as e:
        return None

out = {}
for f in sorted(glob.glob('configs/*.jsx')):  # запускать из каталога со скачанными сниппетами
    if f.endswith('-benchmarks.jsx'): continue
    src = open(f, encoding='utf-8').read()
    name = re.search(r'modelName:\s*"([^"]+)"', src).group(1)
    cells_lit = grab(src, 'cells')
    cells = parse(cells_lit) if cells_lit else None
    if cells is None and cells_lit:
        # parse cell by cell
        cells = []
        for m in re.finditer(r'\{\s*\n\s*match:', cells_lit):
            k = m.start(); depth=0; j=k; in_str=None
            while j < len(cells_lit):
                c=cells_lit[j]
                if in_str:
                    if c=='\\': j+=2; continue
                    if c==in_str: in_str=None
                elif cells_lit.startswith('//', j): j=cells_lit.find('\n', j); continue
                elif c in '"\'`': in_str=c
                elif c=='{': depth+=1
                elif c=='}':
                    depth-=1
                    if depth==0: break
                j+=1
            p = parse(cells_lit[k:j+1])
            if p: cells.append(p)
    docker = parse(grab(src, 'dockerImages') or '{}') or {}
    names = parse(grab(src, 'modelNames') or '{}') or {}
    bfile = f.replace('.jsx', '-benchmarks.jsx')
    bench = []
    try:
        bsrc = open(bfile, encoding='utf-8').read()
        lit = re.search(r'export\s+const\s+benchmarks\s*=\s*', bsrc)
        bench = parse(bsrc[lit.end():bsrc.rfind(']')+1]) or []
    except FileNotFoundError:
        pass
    out[f.split('/')[-1][:-4]] = dict(modelName=name, dockerImages=docker, modelNames=names,
        cells=[dict(match=c.get('match'), verified=c.get('verified'), nnodes=c.get('nnodes'),
                    flags=[x for x in (c.get('flags') or []) if isinstance(x, str)]) for c in (cells or []) if isinstance(c, dict)],
        benchmarks=bench)
    print(f, name, 'cells', len(out[f.split('/')[-1][:-4]]['cells']), 'bench', len(bench), 'with speed', sum(1 for b in bench if b.get('speed')))
json.dump(out, open('cookbook.json','w'), ensure_ascii=False, indent=1)
