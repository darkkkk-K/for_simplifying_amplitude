"""Generate complete four-point amplitudes with shared physical configurations."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=1024)
    parser.add_argument('--dimension-min', type=int, default=2)
    parser.add_argument('--dimension-max', type=int, default=10)
    parser.add_argument('--min-terms', type=int, default=21)
    parser.add_argument('--max-terms', type=int, default=40)
    parser.add_argument('--scramble-steps', type=int, default=128)
    parser.add_argument('--seed', type=int, default=2026102811)
    parser.add_argument('--candidate-cap', type=int, default=24576)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not (1 <= args.count and 2 <= args.dimension_min <= args.dimension_max <= 10
            and 1 <= args.min_terms <= args.max_terms <= 128 and args.scramble_steps >= 0):
        parser.error('Check count, dimension, term bounds and scramble steps.')
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import forsythia_hep as api
    from project.rl.point5_curriculum_data import original_config_generator
    from project.reducibility.features import expression_key, validate
    from project.showcase.evaluation import monomial_capacities
    rng, nrng = random.Random(args.seed), np.random.RandomState(args.seed)
    config = original_config_generator(rng, nrng)
    dimensions = [d for d in range(args.dimension_min, args.dimension_max+1)
                  if max(monomial_capacities(d).values()) >= args.min_terms]
    if not dimensions:
        parser.error('This dimension range has too few distinct monomials for min-terms.')
    rows, origins, seen, counts = [], [], set(), Counter()
    for _ in range(args.candidate_cap):
        if len(rows) >= args.count:
            break
        counts['candidates'] += 1
        helicities, dim = config(4, rng.choice(dimensions))
        helicities = list(map(int, helicities))
        capacity = monomial_capacities(dim)[tuple(helicities)]
        if capacity < args.min_terms:
            continue
        original = 0*api.abk(1, 2)
        for _ in range(rng.randint(1, 5)):
            counts['monomial_calls'] += 1
            mono = api.random_monomial(helicities, dim)
            validate(mono.tokenize(), 4, dim, helicities)
            original += rng.choice((-3, -2, -1, 1, 2, 3))*mono
        original = original.normal_order().expand()
        clean = original.tokenize()
        origin_key = expression_key(clean, 4)
        if not 1 <= len(clean) <= 5 or origin_key in seen:
            continue
        target = rng.randint(args.min_terms, min(args.max_terms, capacity))
        expr = original
        for _ in range(args.scramble_steps):
            counts['random_identity_calls'] += 1
            expr = (api.random_apply_schouten(expr, 1) if rng.random() < .5
                    else api.random_apply_momentum_conservation(expr, 4, 1))[1]
            tokens = expr.tokenize()
            validate(tokens, 4, dim, helicities)
            if target <= len(tokens) <= args.max_terms or len(tokens) > 128:
                break
        tokens = expr.tokenize()
        input_key = expression_key(tokens, 4)
        if not args.min_terms <= len(tokens) <= args.max_terms or input_key in seen:
            continue
        sid = hashlib.sha256(repr(clean).encode()).hexdigest()
        rows.append(dict(index=len(rows), source_id=sid, split='generated', n_point=4, dimension=dim,
                         helicities_twice=helicities, terms=len(tokens), tokens=tokens))
        origins.append(dict(source_id=sid, source_terms=len(clean), tokens=clean))
        seen.update((origin_key, input_key))
        if len(rows) % 32 == 0:
            print(f'{len(rows)}/{args.count}', flush=True)
    for name, value in [('inputs.json', rows), ('origins.json', origins), ('generation.json', dict(counts=counts, settings=vars(args)))]:
        (args.output/name).write_text(json.dumps(value, default=str, ensure_ascii=True, indent=2), encoding='utf-8')
    print(json.dumps(dict(accepted=len(rows), counts=counts)))
    if len(rows) != args.count:
        raise RuntimeError('Candidate budget reached; generated records were saved.')


if __name__ == '__main__':
    main()
