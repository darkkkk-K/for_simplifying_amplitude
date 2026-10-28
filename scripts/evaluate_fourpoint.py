"""Run the released policy on a saved four-point dataset."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=ROOT/'datasets/fourpoint1024/test.json')
    parser.add_argument('--policy', type=Path, default=ROOT/'artifacts/models/rl/policy_096.pt')
    parser.add_argument('--encoder', type=Path, default=ROOT/'artifacts/models/pretrained/temp15_final_model_sl_256.pth')
    parser.add_argument('--topk', type=int, default=4)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--seed', type=int, default=2026102812)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    import torch
    from scripts.simplify_policy import PolicySimplifier
    from project.rl.packed32_common import expression_from_tokens
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rows = json.loads(args.inputs.read_text(encoding='utf-8'))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output or ROOT/'artifacts/evaluations'/stamp
    output.mkdir(parents=True, exist_ok=False)
    solver = PolicySimplifier(args.policy, args.encoder)
    results = []
    for row in rows:
        seed_data = ['portfolio_fourpoint1024_20261028_v1', row['source_id'], args.seed]
        seed = int(hashlib.sha256(json.dumps(seed_data, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()[:15], 16)
        result = solver.simplify(expression_from_tokens(row['tokens']), row['n_point'],
                                 topk=args.topk, budget_steps=args.steps, seed=seed)
        info = solver.last_info
        results.append(dict(source_id=row['source_id'], dimension=row['dimension'], seed=seed,
                            **info, returned_tokens=result.tokenize(), returned_expression=str(result)))
        if len(results) % 32 == 0:
            print(f'{len(results)}/{len(rows)}', flush=True)
    from statistics import fmean
    metrics = {}
    for name, lo, hi in [('all', 2, 10), ('2-4', 2, 4), ('5-7', 5, 7), ('8-10', 8, 10)]:
        group = [r for r in results if lo <= r['dimension'] <= hi]
        if group:
            metrics[name] = dict(count=len(group), mean_reduction=fmean(1-r['best_terms']/r['initial_terms'] for r in group),
                mean_input_terms=fmean(r['initial_terms'] for r in group), mean_output_terms=fmean(r['best_terms'] for r in group),
                total_term_reduction=1-sum(r['best_terms'] for r in group)/sum(r['initial_terms'] for r in group))
    (output/'results.json').write_text(json.dumps(dict(settings=vars(args), metrics=metrics, records=results),
        default=str, ensure_ascii=True, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))
    print('Results:', output)


if __name__ == '__main__':
    main()
