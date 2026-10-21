"""MMDM JSON, actor and typed decision API contract."""
import hashlib
import json
import math
import os
from pathlib import Path

def dump(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, allow_nan=False)

def digest(x):
    return hashlib.sha256(dump(x).encode()).hexdigest()

def file_sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def save(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + f'.{os.getpid()}.partial')
    t.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    t.replace(p)

def rows(p):
    with Path(p).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def jsonl(p, values):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + f'.{os.getpid()}.partial')
    with t.open('w') as f:
        for value in values:
            f.write(dump(value) + '\n')
    t.replace(p)

def actor(r):
    return {k: r[k] for k in ['state', 'question', 'options', 'task', 'score_values'] if k in r}

def gold_kind(q):
    assert q and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 0 <= x <= 1 for x in q)
    assert abs(math.fsum(q) - 1) < 1e-6
    return 'hard' if q.count(1.0) == 1 and q.count(0.0) == len(q)-1 else 'soft'

def valid_p(p, n):
    assert isinstance(p, list) and len(p) == n
    assert all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 0 <= x <= 1 for x in p)
    s = math.fsum(p)
    assert s > 0 and abs(s - 1) <= .020000001
    return [x/s for x in p]

def wire(r, model):
    opts = r['options']; names = [o['name'] for o in opts]
    kind = r['task']; q = {'type': kind, 'instructions': r['question']}
    if kind == 'noul' and [x.lower() for x in names] == ['false', 'true']:
        keys = ['no', 'yes']
        q['criteria'] = {k: o['criteria'] for k, o in zip(['false', 'true'], opts) if o.get('criteria') is not None}
    elif kind == 'score' and len(opts) <= 10:
        assert len(r['score_values']) == len(opts)
        keys = list(map(str, range(len(opts))))
        q['criteria'] = [dict(name=o['name'], criteria=o.get('criteria'), score_value=v) for o, v in zip(opts, r['score_values'])]
    else:
        q['type'] = 'choice'; keys = names if len(set(names)) == len(names) else [o['id'] for o in opts]
        q['criteria'] = {}
        for i, (k, o) in enumerate(zip(keys, opts)):
            c = o.get('criteria')
            if keys != names or kind == 'score':
                c = dict(name=o['name'], criteria=c)
                if kind == 'score': c['score_value'] = r['score_values'][i]
            q['criteria'][k] = c
    return {'state': r['state'], 'questions': {'decision': q}, 'model': model}, keys

def parse_native(body, keys, response):
    a = response['answers']['decision']
    if a.get('type') != body['questions']['decision']['type']:
        raise ValueError('response_type_mismatch')
    if a['type'] == 'noul':
        p = a['noul']
        if isinstance(p, bool): raise ValueError('boolean_probability')
        return valid_p([1-float(p), float(p)], 2)
    p = a['probabilities']
    if set(p) != set(keys): raise ValueError('option_key_mismatch')
    return valid_p([p[k] for k in keys], len(keys))
