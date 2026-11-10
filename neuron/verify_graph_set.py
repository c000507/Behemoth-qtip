#!/usr/bin/env python3
"""List the buckets of a NEFF graph directory with the tool versions recorded in their meta.json and flag problems.

    neuron/verify_graph_set.py GRAPH_DIR [--json]

Per bucket: kind / shape, neuronx-cc and libnrt versions, git commit, build timestamp (meta.json "build", written by
trace_model.py; older buckets have none and are listed as "no build info"). Exit status 1 when the set mixes neuronx-cc
versions, mixes head formats (fp8 / bf16, which the runtime refuses), lacks meta.json or graph.neff in a bucket directory, or
has buckets without build info; 0 otherwise. Read-only; needs no Neuron device."""
import json
import sys
from pathlib import Path


def bucket_rows(graph_dir):
    rows, problems = [], []
    for d in sorted(p for p in Path(graph_dir).iterdir() if p.is_dir()):
        meta_path, neff = d / 'meta.json', d / 'graph.neff'
        if not meta_path.exists():
            if neff.exists():
                problems.append(f'{d.name}: graph.neff without meta.json (the runtime ignores this directory)')
            continue
        try:
            m = json.loads(meta_path.read_text())
        except json.JSONDecodeError as error:
            problems.append(f'{d.name}: meta.json is not valid JSON ({error})')
            continue
        if not neff.exists():
            problems.append(f'{d.name}: meta.json without graph.neff (the runtime fails to load it)')
        b = m.get('build') or {}
        kind = m.get('kind', '?')
        shape = {'layer': lambda: f"R={m.get('R')} Q={m.get('Q')} P={m.get('P')}" + (' shared' if m.get('shared_table') else ''),
                 'head': lambda: f"T={m.get('T')} S={m.get('S')} {'fp8' if m.get('fp8') else 'bf16'}" + (' argmax' if m.get('argmax') else ''),
                 'model': lambda: f"R={m.get('R')} Q={m.get('Q')} P={m.get('P')} L={m.get('L')}"}.get(kind, lambda: '')()
        rows.append({'bucket': d.name, 'kind': kind, 'shape': shape, 'neuronx_cc': b.get('neuronx_cc_bin') or b.get('neuronx_cc'),
                     'libnrt': b.get('libnrt'), 'commit': (b.get('git_commit') or '')[:12] + ('+' if b.get('git_dirty') else ''),
                     'timestamp': b.get('timestamp'), 'fp8_head': m.get('fp8') if kind == 'head' else None})
    compilers = {r['neuronx_cc'] for r in rows if r['neuronx_cc']}
    if len(compilers) > 1:
        problems.append(f"mixed neuronx-cc versions: {sorted(compilers)}")
    missing = [r['bucket'] for r in rows if not r['neuronx_cc']]
    if missing:
        problems.append(f"{len(missing)} bucket(s) without build info (traced before meta.json 'build'): {' '.join(missing)}")
    head_formats = {r['fp8_head'] for r in rows if r['kind'] == 'head'}
    if len(head_formats) > 1:
        problems.append('head buckets mix fp8 and bf16 lm_head formats (the runtime refuses the set)')
    return rows, problems


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if len(args) != 1:
        raise SystemExit(__doc__)
    rows, problems = bucket_rows(args[0])
    if '--json' in sys.argv:
        print(json.dumps({'buckets': rows, 'problems': problems}, indent=1))
    else:
        width = max([len(r['bucket']) for r in rows] + [6])
        for r in rows:
            print(f"{r['bucket']:<{width}}  {r['kind']:<5} {r['shape']:<28} neuronx-cc {r['neuronx_cc'] or '-':<22} libnrt {r['libnrt'] or '-':<22} "
                  f"{r['commit'] or '-':<13} {r['timestamp'] or 'no build info'}")
        for p in problems:
            print('PROBLEM:', p)
        if not problems:
            print(f'OK: {len(rows)} buckets, one compiler version {next(iter({r["neuronx_cc"] for r in rows}), None)}')
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
