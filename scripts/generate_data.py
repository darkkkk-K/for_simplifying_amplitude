"""Generate encoder contrastive groups or complete RL training expressions."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('encoder', 'rl'), default='rl')
    p.add_argument('--count', type=int, default=1024)
    p.add_argument('--n-point', type=int, choices=(4, 5, 6), default=4)
    p.add_argument('--dimension-min', type=int, default=2)
    p.add_argument('--dimension-max', type=int, default=10)
    p.add_argument('--min-terms', type=int, default=2)
    p.add_argument('--max-terms', type=int, default=36)
    p.add_argument('--scramble-steps', type=int, default=32)
    p.add_argument('--candidate-cap', type=int, default=24576)
    p.add_argument('--seed', type=int, default=103)
    p.add_argument('--output', type=Path)
    p.add_argument('--run', action='store_true')
    args = p.parse_args()
    print(json.dumps(vars(args), default=str, indent=2))
    if not args.run:
        return
    if args.output is None or args.output.exists() or not 1 <= args.count or not 2 <= args.dimension_min <= args.dimension_max:
        p.error('Use a new --output directory and valid count/dimension range.')
    if not 1 <= args.min_terms <= args.max_terms <= 128 or args.scramble_steps < 0:
        p.error('Invalid term bounds or scramble steps.')
    import numpy as np
    from data.generation import amplitude_config_gen, data_gen_for_pre
    random.seed(args.seed); np.random.seed(args.seed)
    args.output.mkdir(parents=True)
    if args.stage == 'encoder':
        all_groups = data_gen_for_pre(total_samples=args.count, output_file=None, if_chunk=False)
        groups = next(all_groups)
        split = max(1, int(len(groups)*.8))
        for label, samples in (('train_data', groups[:split]), ('val_data', groups[split:])):
            if not samples:
                raise ValueError('Need enough groups for both train and validation')
            folder = args.output/label; folder.mkdir()
            shard = folder/'amp_data_part_000.pkl.gz'
            with gzip.open(shard, 'wb') as f: pickle.dump(samples, f)
            metadata = dict(total_num_samples=len(samples), num_files=1, filesample_index_mapping={str(shard.resolve()):
                dict(start_idx=0, end_idx=len(samples)-1, num_samples=len(samples))})
            with gzip.open(folder/'metadata.pkl.gz', 'wb') as f: pickle.dump(metadata, f)
        return
    import forsythia_hep as api
    from data.features import validate, expression_key
    rows, originals, seen, counts = [], [], set(), Counter()
    for _ in range(args.candidate_cap):
        if len(rows) == args.count: break
        counts['candidates'] += 1
        hels, dim = amplitude_config_gen(args.n_point, random.randint(args.dimension_min, args.dimension_max))
        hels = list(map(int, hels))
        source = 0*api.abk(1, 2)
        for _ in range(random.randint(1, 5)):
            counts['monomial_calls'] += 1
            mono = api.random_monomial(hels, dim)
            validate(mono.tokenize(), args.n_point, dim, hels)
            source += random.choice((-3, -2, -1, 1, 2, 3))*mono
        source = source.normal_order().expand()
        clean = source.tokenize(); origin = expression_key(clean, args.n_point)
        if not 1 <= len(clean) <= 5 or origin in seen: continue
        current = source
        for _ in range(args.scramble_steps):
            counts['random_identity_calls'] += 1
            current = (api.random_apply_schouten(current, 1) if random.random() < .5
                       else api.random_apply_momentum_conservation(current, args.n_point, 1))[1]
            if len(current.tokenize()) > 128: break
        tokens = current.tokenize(); key = expression_key(tokens, args.n_point)
        if not max(args.min_terms, len(clean)+1) <= len(tokens) <= args.max_terms or key in seen: continue
        validate(tokens, args.n_point, dim, hels)
        sid = hashlib.sha256(repr(clean).encode()).hexdigest()
        rows.append(dict(index=len(rows), split='train', source_id=sid, n_point=args.n_point,
                         dimension=dim, helicities_twice=hels, terms=len(tokens), tokens=tokens))
        originals.append(dict(source_id=sid, source_terms=len(clean), tokens=clean))
        seen.update((origin, key))
        if len(rows) % 128 == 0: print(f'{len(rows)}/{args.count}', flush=True)
    for name, value in [('train.json', rows), ('origins.json', originals), ('generation.json', dict(counts=counts, settings=vars(args)))]:
        (args.output/name).write_text(json.dumps(value, default=str, indent=2), encoding='utf-8')
    if len(rows) != args.count: raise RuntimeError('Candidate budget reached; partial data saved.')


if __name__ == '__main__':
    main()
