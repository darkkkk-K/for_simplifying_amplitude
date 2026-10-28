"""Current H24 PPO: 32 fresh streams, minibatch16, two epochs."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/ppo.json')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--run', action='store_true', help='Execute the configured training budget')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    print(json.dumps(config, ensure_ascii=False, indent=2))
    if not args.run:
        return
    if args.output is None or args.output.exists() or args.checkpoint and args.resume:
        parser.error('Provide a new --output directory; choose checkpoint or resume.')
    import numpy as np
    import torch
    from models.loading import load_policy
    from training.runtime import CurriculumDeck, TrainingRuntime, collect_rollout
    from training.ppo import update_ppo
    if (config['lanes'], config['horizon'], config['batch_size'], config['epochs']) != (32, 24, 16, 2):
        raise ValueError('Current preset uses 32×24/batch16/epoch2')
    if (config['learning_rate'] != 1e-4 or config['entropy_coef'] != .001 or config['gae_lambda'] != .93
            or config['gamma'] != 1. or config['beta'] != .4 or config['top_k'] != 8):
        raise ValueError('Current PPO preset: lr1e-4, entropy.001, GAE.93')
    if sum(p['quota'] for p in config['pools']) != 32 or not 1 <= config['rounds'] <= 60:
        raise ValueError('Invalid rollout count or quotas')
    data, quotas, fingerprints = {}, {}, {}
    from data.features import validate
    seen = set()
    for pool in config['pools']:
        path = ROOT/pool['path']
        rows = json.loads(path.read_text(encoding='utf-8'))
        if not rows or any(r['split'] != 'train' for r in rows):
            raise ValueError('Training split required')
        for row in rows:
            validate(row['tokens'], row['n_point'], row['dimension'], row['helicities_twice'])
            if row['source_id'] in seen:
                raise ValueError('Repeated source across training pools')
            seen.add(row['source_id'])
        data[pool['name']], quotas[pool['name']] = rows, pool['quota']
        fingerprints[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    seed = config['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_policy(ROOT/config['encoder'], args.checkpoint, device=device, top_k=config['top_k'])
    if model.top_k != config['top_k']:
        raise ValueError('Checkpoint sampling configuration differs from training config')
    model.train()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4, weight_decay=0.)
    deck = CurriculumDeck(data, quotas, seed)
    first, updates = 1, 0
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if saved['config'] != config or saved['data_sha256'] != fingerprints:
            raise ValueError('Resume config/data mismatch')
        model.load_checkpoint_payload(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        deck.load_state_dict(saved['deck'])
        random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'])
        if device == 'cuda': torch.cuda.set_rng_state_all(saved['cuda_rng'])
        first, updates = saved['round']+1, saved['updates']
    args.output.mkdir(parents=True)
    (args.output/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    runtime = TrainingRuntime(model, horizon=24, c_punishment=config['c_punishment'])
    counts = Counter()
    recent = []
    for round_index in range(first, config['rounds']+1):
        states, buffer = collect_rollout(runtime, deck, seed, round_index)
        folder = args.output/f'round_{round_index:03d}'
        folder.mkdir()
        with gzip.open(folder/'buffer.json.gz', 'wt', encoding='utf-8') as stream:
            json.dump(buffer, stream)
        update_records = []
        update_ppo(model, optimizer, states, buffer['rows'], buffer['advantages'], buffer['targets'],
            generator=torch.Generator().manual_seed(seed+round_index), hit=lambda k, n=1: counts.update({k: n}),
            on_step=update_records.append, completed_updates=updates, enabled=True)
        updates += 96
        with gzip.open(folder/'updates.json.gz', 'wt', encoding='utf-8') as stream:
            json.dump(update_records, stream)
        torch.save(dict(model=model.checkpoint_payload(), optimizer=optimizer.state_dict(), deck=deck.state_dict(),
            config=config, data_sha256=fingerprints, round=round_index, updates=updates,
            python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device == 'cuda' else []), args.output/'recovery.pt')
        if round_index in config['save_rounds'] or round_index == config['rounds']:
            torch.save(model.checkpoint_payload(), args.output/f'policy_{round_index:03d}.pt')
        recent.extend(runtime.finished); runtime.finished = []
        if round_index % 4 == 0 or round_index == config['rounds']:
            print(json.dumps(dict(round=round_index, episodes=len(recent), updates=updates,
                mean_final_reduce=sum(r['final_reduce'] for r in recent)/len(recent) if recent else None,
                mean_loss=sum(r['loss'] for r in update_records)/len(update_records),
                counts=dict(runtime.counts), optimizer_counts=dict(counts))), flush=True)
            recent = []


if __name__ == '__main__':
    main()
