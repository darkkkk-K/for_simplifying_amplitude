"""New source-disjoint512 development +512 sealed test inputs. No inverse labels."""
from collections import Counter
import importlib.util
import json
from pathlib import Path
import random

from project.rl.packed32_common import REPO, Ledger, append, digest, key, read_json, write_json
from data.generation_full32 import token_id, accept_lengths, public_record

DATA_ID = 'lambda20_eval1024_20261024_v1'
CAPS = {'attempts': 4096, 'monomial_calls': 16384, 'random_identity_calls': 65536}


def exclusion_metadata():
    paths = sorted((REPO/'datasets').glob('**/samples.jsonl'))
    paths += [REPO/'datasets/packed_full32_20261023_v1'/n for n in ('train.json', 'validation.json')]
    paths += [REPO/'datasets/single2048_20261024_v1'/n for n in ('train.json', 'test.json')]
    sources, inputs, hashes = set(), set(), {}
    for path in paths:
        hashes[path.relative_to(REPO).as_posix()] = digest(path)
        rows = [json.loads(line) for line in path.read_text().splitlines()] if path.suffix == '.jsonl' else read_json(path)
        for row in rows:
            if row.get('source_id'): sources.add(row['source_id'])
            for name in ('id', 'sample_id'):
                if row.get(name): inputs.add(row[name])
    return sources, inputs, hashes


def split_records(rows):
    if len(rows) != 1024 or len({r['source_id'] for r in rows}) != 1024 or len({r['id'] for r in rows}) != 1024:
        raise ValueError('Need1024 distinct sources/inputs')
    ordered = sorted(rows, key=lambda r: key(DATA_ID, 'split', r['source_id']))
    return [dict(r, index=i, split='dev' if i < 512 else 'test') for i, r in enumerate(ordered)]


def run():
    from project.rl.legal_actions import build_policy_input
    data, out, contract = REPO/'datasets'/DATA_ID, REPO/'artifacts/runs'/DATA_ID, REPO/'experiments'/DATA_ID
    if any(p.exists() for p in (data, out, contract)): raise FileExistsError('No overwrite/retry')
    excluded_sources, excluded_inputs, dependencies = exclusion_metadata()
    initial_sources, initial_inputs = set(excluded_sources), set(excluded_inputs)
    engine = Path(importlib.util.find_spec('forsythia_hep').origin)
    protocol = {'id': DATA_ID, 'count': 1024, 'dev': 512, 'test': 512, 'n_point': 4,
        'source_monomials': [1, 4], 'scramble_attempts': [4, 8, 12, 16], 'mass_dimensions': [6, 8, 10],
        'helicities': [[-1,-1,1,1], [0,0,0,0]], 'accept': 'whole source_terms<input_terms<=32; source1..4 terms; no subset',
        'split': 'sort by source hash-derived order, first512 dev,last512 sealed test', 'python_seed': 2026102402,
        'native_rng': 'not exposed; saved exact inputs authoritative', 'caps': CAPS,
        'excluded_metadata_sha256': dependencies, 'storage_cap_bytes': 256*1024**2,
        'test_use': 'only after model selection; no policy/model use during generation',
        'code_sha256': {p: digest(REPO/p) for p in ('data/generation_eval1024.py', 'data/generation_full32.py',
            'project/rl/legal_actions.py', 'project/rl/packed32_common.py', 'scripts/generate_eval1024.py')},
        'engine_sha256': {str(p): digest(p) for p in [engine, *engine.parent.glob('*.pyd')]}}
    data.mkdir(); out.mkdir(parents=True); contract.mkdir(parents=True)
    write_json(contract/'protocol.json', protocol); write_json(out/'started.json', {'protocol_sha256': digest(contract/'protocol.json')})
    ledger = Ledger(CAPS); rng = random.Random(protocol['python_seed']); accepted = []; rejected = Counter()
    try:
        import forsythia_hep as api
        while len(accepted) < 1024 and ledger.counts['attempts'] < CAPS['attempts']:
            ledger.hit('attempts'); serial = ledger.counts['attempts']
            helicity, dimension = rng.choice(protocol['helicities']), rng.choice(protocol['mass_dimensions'])
            source = 0*api.abk(1, 2)
            for _ in range(rng.randint(1, 4)):
                ledger.hit('monomial_calls'); call = ledger.counts['monomial_calls']
                append(out/'calls.jsonl', {'event': 'start', 'kind': 'monomial', 'attempt': serial, 'serial': call})
                term = api.random_monomial(helicity, dimension)
                source += rng.choice((-3,-2,-1,1,2,3))*term
                append(out/'calls.jsonl', {'event': 'end', 'kind': 'monomial', 'attempt': serial, 'serial': call, 'id': token_id(term.tokenize())})
            clean_tokens = source.tokenize(); sid = token_id(clean_tokens); source_terms = len(clean_tokens)
            reason = None if 1 <= source_terms <= 4 and sid not in excluded_sources else 'source_invalid_or_duplicate'
            current = source; steps = protocol['scramble_attempts'][len(accepted)%4]
            if reason is None:
                for step in range(steps):
                    kind = rng.choice(('schouten', 'momentum')); ledger.hit('random_identity_calls'); call = ledger.counts['random_identity_calls']
                    append(out/'calls.jsonl', {'event': 'start', 'kind': kind, 'serial': call, 'attempt': serial, 'step': step})
                    error = None
                    try:
                        current = (api.random_apply_schouten(current, 1) if kind == 'schouten' else api.random_apply_momentum_conservation(current, 4, 1))[1]
                    except RuntimeError as exc:
                        if str(exc) != 'no nontrivial identity can be applied': raise
                        error = str(exc)
                    tokens = current.tokenize()
                    append(out/'calls.jsonl', {'event': 'end', 'kind': kind, 'serial': call, 'attempt': serial,
                        'id': token_id(tokens), 'terms': len(tokens), 'error': error})
                    if len(tokens) > 128 or max(map(len, tokens), default=0) > 128:
                        reason = 'intermediate_capacity'; break
            tokens = current.tokenize(); iid = token_id(tokens)
            if reason is None and not accept_lengths(source_terms, len(tokens)): reason = 'no_growth_or_length'
            if reason is None and iid in excluded_inputs: reason = 'duplicate_input'
            if reason is None:
                build_policy_input([tokens], [4], [20], [20], max_terms=32)
                accepted.append(public_record(len(accepted), 'pending', tokens, sid)); excluded_sources.add(sid); excluded_inputs.add(iid)
            else: rejected[reason] += 1
            append(out/'attempts.jsonl', {'attempt': serial, 'accepted': reason is None, 'reason': reason,
                'source_id': sid, 'input_id': iid, 'input_terms': len(tokens)})
            if reason is None and len(accepted)%128 == 0:
                print(f'new evaluation inputs {len(accepted)}/1024; attempts={serial}; identity={ledger.counts["random_identity_calls"]}', flush=True)
                if sum(p.stat().st_size for p in out.iterdir() if p.is_file()) > protocol['storage_cap_bytes']: raise RuntimeError('Storage cap')
        if len(accepted) != 1024: raise RuntimeError('Generation cap exhausted')
        ordered = split_records(accepted)
        if {r['source_id'] for r in ordered} & initial_sources or {r['id'] for r in ordered} & initial_inputs:
            raise ValueError('Existing-source/input overlap')
        for split in ('dev', 'test'): write_json(data/(split+'.json'), [r for r in ordered if r['split'] == split])
        result = {'status': 'completed', 'count': 1024, 'dev': 512, 'test': 512, 'source_count': 1024,
            'source_overlap': 0, 'overlap_with_existing_sources': 0, 'input_only': True,
            'counts': dict(ledger.counts), 'rejections': dict(rejected),
            'terms': {'min': min(r['terms'] for r in ordered), 'max': max(r['terms'] for r in ordered), 'mean': sum(r['terms'] for r in ordered)/1024},
            'files_sha256': {s+'.json': digest(data/(s+'.json')) for s in ('dev', 'test')}}
        for name, expected in {**protocol['code_sha256'], **dependencies}.items():
            if digest(REPO/name) != expected: raise ValueError('Generation dependency changed')
        write_json(data/'manifest.json', result); write_json(out/'completed.json', result); write_json(contract/'summary.json', result)
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as exc:
        write_json(out/'failed.json', {'error': repr(exc), 'counts': dict(ledger.counts), 'accepted': len(accepted)}); raise
