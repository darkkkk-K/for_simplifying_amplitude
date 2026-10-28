"""32x16x24 PPO after a frozen encoder, with fresh policy and strict top16."""
from collections import Counter
from dataclasses import asdict
import gzip
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
from unittest.mock import patch

from .packed32_common import REPO, Ledger, append, check_sources, digest, expression_from_tokens, key, paired, read_json, source_hashes, tensor_digest, write_json
from .single24_experiment import DATA, SourceDeck, inputs, save_json_gz, save_model, seed_for as previous_seed

RUN = 'early_fusion_top16_20261024_v1'
CAPS = {'checkpoint_loads': 1, 'old_rl_checkpoint_loads': 0, 'encoder_forwards': 40000,
        'policy_forwards': 40000, 'preparations': 40000, 'train_engine': 12288, 'eval_engine': 24576,
        'sampling': 36864, 'reconstructions': 14000, 'generation': 0,
        'update_forwards': 1536, 'autograd': 1536, 'optimizer': 1536, 'encoder_feature_reuses': 24576,
        'small_state_restores': 1, 'cache_hits': 0}


def locations():
    return REPO/'artifacts/runs'/RUN, REPO/'experiments'/RUN


def seed_for(*parts):
    return int(key(RUN, *parts)[:15], 16)


def plan():
    return {'id': RUN, 'dataset': DATA, 'seed': 103, 'top_k': 16, 'rounds': 32, 'lanes': 16,
        'steps_per_lane': 24, 'steps_per_round': 384, 'minibatch': 16, 'epochs': 2,
        'optimizer_per_round': 48, 'optimizer_total': 1536, 'only_encoder_frozen': True,
        'policy_initialization': 'fresh Xavier; actor final normal std .01; no old RL parameters or cosine base',
        'symbolic_features': '32 algebra + 3 n_point one-hot before term attention',
        'support': 'all structural legal actions; explicit n_point; no encoder threshold/query-history exclusions',
        'behavior': 'dynamic current-policy top16; stable tie order; categorical within top16; one real successor accepted',
        'ppo': {'gamma': 1., 'gae_lambda': .95, 'clip': .2, 'value_coef': .5, 'grad_clip': .5, 'lr': .0003, 'weight_decay': 0., 'entropy_coef': 0.},
        'rewards': {'segment_enabled': True, 'potential_enabled': True, 'segment_scale': .5, 'potential_scale': .25},
        'episode_horizon': 24, 'max_terms': 128, 'max_tokens': 128, 'max_actions': 32768,
        'restart': 'new training source after terminal; carry unfinished episodes across rollout windows; GAE terminal mask and tail bootstrap',
        'reports': 'each4 rounds, actual completed episode final reduce; cumulative final reduce; pending segments separate; best supplemental',
        'test': 'consumed256-source development comparison only, seed401/503, trained endpoint vs fresh initial; no model selection',
        'caps': CAPS, 'buffer_cap_bytes': 2*1024**3, 'storage_cap_bytes': 2*1024**3,
        'stopping': '32 rounds except mechanical failures; no automatic retry, budget extension or deployment',
        'topk_support_caveat': 'current topk recomputed in PPO; excluded old actions have new probability0; support changes may make old-new KL infinite',
        'automatic_deployment': False, 'new_data_generation': False}


def freeze():
    from project.paths import asset_path
    out, contract = locations()
    if out.exists() or (contract/'protocol.json').exists():
        raise FileExistsError('No overwrite or implicit resume')
    rows = inputs('train')
    encoder_path = asset_path('checkpoints/temp15_final_model_sl_256.pth')
    baseline_path = asset_path('checkpoints_p1_depth2_full160_160u_seed42/grpo_update_0096.pth')
    engine = Path(importlib.util.find_spec('forsythia_hep').origin)
    code = source_hashes(('project/rl/early_fusion_policy.py', 'project/rl/early_fusion_experiment.py',
        'project/rl/full32_common.py', 'project/rl/packed_ppo.py', 'project/rl/single24_policy.py',
        'project/rl/single24_experiment.py', 'scripts/run_early_fusion.py', 'tests/test_early_fusion.py'))
    protocol = plan() | {'code_sha256': code, 'encoder_path': str(encoder_path), 'encoder_sha256': digest(encoder_path),
        'protected_baseline_path': str(baseline_path), 'protected_baseline_sha256': digest(baseline_path),
        'dataset_sha256': digest(REPO/'datasets'/DATA/'manifest.json'),
        'input_sha256': read_json(REPO/'datasets'/DATA/'manifest.json')['files_sha256'],
        'engine_sha256': {str(p): digest(p) for p in [engine, *engine.parent.glob('*.pyd')]},
        'git_index_sha256': digest(REPO/'.git/index')}
    contract.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True)
    for name in ('training', 'snapshots', 'progress', 'evaluation'):
        (out/name).mkdir()
    write_json(contract/'protocol.json', protocol)
    write_json(out/'started.json', {'protocol_sha256': digest(contract/'protocol.json'), 'pid': os.getpid()})
    return out, contract, protocol, Ledger(CAPS), rows


class Runtime:
    def __init__(self, out, model, ledger, rewards):
        import forsythia_hep as api
        self.out, self.model, self.ledger, self.rewards = out, model, ledger, rewards
        self.phase, self.round, self.serial = 'train', 0, 0
        self.apply = api.apply_identity_at
        self.seen = set(); self.stream = gzip.open(out/'states.jsonl.gz', 'wt', encoding='utf-8')
        self.finished = []; self.lanes = [None]*16
        self.encoder_before = tensor_digest(model.encoder.state_dict())
        self.hook = model.encoder.register_forward_pre_hook(lambda *_: ledger.hit('encoder_forwards'))
        def forbid(*a, **k):
            ledger.hit('generation'); raise RuntimeError('Generation forbidden')
        self.guards = [patch.object(api, name, forbid) for name in ('random_monomial', 'random_apply_schouten', 'random_apply_momentum_conservation', 'simplify')]
        for guard in self.guards: guard.start()

    def state(self, expr):
        from .legal_actions import content_id
        tokens = expr.tokenize(); sid = content_id(tokens)
        if sid not in self.seen:
            self.stream.write(json.dumps({'id': sid, 'tokens': tokens}, separators=(',', ':'))+'\n'); self.seen.add(sid)
        return sid

    def frozen_check(self):
        if tensor_digest(self.model.encoder.state_dict()) != self.encoder_before or any(p.grad is not None or p.requires_grad for p in self.model.encoder.parameters()):
            raise ValueError('Pretrained encoder changed')
        if not all(p.requires_grad for p in self.model.trainable_parameters()):
            raise ValueError('An intended trainable policy parameter is frozen')

    def close(self):
        self.stream.close(); self.hook.remove()
        for guard in reversed(self.guards): guard.stop()

    def prepare(self, expr, progress, n_point):
        from .legal_actions import build_policy_input
        tokens = expr.tokenize()
        if not tokens: return None, 'zero_expression'
        if len(tokens) > 128 or max(map(len, tokens), default=0) > 128: return None, 'capacity'
        self.ledger.hit('preparations')
        batch = build_policy_input([tokens], [n_point], [24-progress.step], [24])
        state = self.model.prepare(batch, progress)
        return (state, None) if state is not None else (None, 'no_legal_actions')

    def score(self, state):
        import torch
        self.ledger.hit('policy_forwards')
        with torch.no_grad(): return self.model([state])

    def finish(self, ep):
        row = {'phase': self.phase, 'round_ended': self.round, 'episode': ep['episode'],
            'source_id': ep['source']['source_id'], 'n_point': ep['source']['n_point'],
            'initial_state': ep['initial_state'], 'initial_terms': ep['progress'].initial,
            'final_state': self.state(ep['expr']), 'final_terms': len(ep['expr'].tokenize()),
            'best_state': ep['best_state'], 'best_terms': ep['progress'].best,
            'steps': ep['progress'].step, 'reason': ep['reason']}
        row['final_reduce'] = (row['initial_terms']-row['final_terms'])/row['initial_terms']
        row['best_reduce'] = (row['initial_terms']-row['best_terms'])/row['initial_terms']
        self.finished.append(row); append(self.out/'episodes.jsonl', row)

    def start(self, source):
        from .single24_policy import Progress
        self.ledger.hit('reconstructions'); self.serial += 1
        expr = expression_from_tokens(source['tokens']); sid = self.state(expr)
        progress = Progress(len(source['tokens']), len(source['tokens']))
        state, reason = self.prepare(expr, progress, source['n_point'])
        ep = {'source': source, 'expr': expr, 'episode': self.serial, 'progress': progress,
            'initial_state': sid, 'initial_phi': state.potential if state else 0., 'state': state, 'reason': reason, 'best_state': sid}
        if state is None: self.finish(ep)
        return ep

    def transition(self, ep, generator):
        from .early_fusion_policy import condition, sample_one
        from .single24_policy import Progress
        state, progress, expr = ep['state'], ep['progress'], ep['expr']
        logits, mask, values = self.score(state)
        self.ledger.hit('sampling'); index, logp, top = sample_one(logits, mask, 16, generator)
        t, i, j = state.actions[index].tolist(); action = (t, (i, j))
        tokens = expr.tokenize(); parent = self.state(expr); n = ep['source']['n_point']
        self.ledger.hit(self.phase+'_engine'); serial = self.ledger.counts[self.phase+'_engine']
        append(self.out/'calls.jsonl', {'event': 'start', 'phase': self.phase, 'serial': serial, 'parent': parent, 'action': action, 'n_point': n})
        try: child = self.apply(expr, action, n)[0]
        except BaseException as exc:
            append(self.out/'calls.jsonl', {'event': 'error', 'phase': self.phase, 'serial': serial, 'error': repr(exc)}); raise
        if expr.tokenize() != tokens: raise ValueError('Parent mutated')
        cid = self.state(child); length = len(child.tokenize())
        append(self.out/'calls.jsonl', {'event': 'end', 'phase': self.phase, 'serial': serial, 'child': cid, 'terms': length})
        reason = 'horizon' if progress.step+1 == 24 else None
        next_state = None
        if reason is None:
            preliminary = Progress(progress.initial, progress.best, progress.step+1, progress.best_stage, progress.paid_bonus)
            next_state, reason = self.prepare(child, preliminary, n)
        if length > 128 or max(map(len, child.tokenize()), default=0) > 128:
            reason, next_state = 'capacity', None
        effective = progress.best if reason == 'capacity' else length
        new_progress, reward = progress.advance(effective, state.potential, next_state.potential if next_state else 0., reason is not None, self.rewards)
        if next_state is not None: next_state = condition(next_state, new_progress)
        if new_progress.best < progress.best: ep['best_state'] = cid
        ep.update(expr=child, progress=new_progress, state=next_state, reason=reason)
        row = {'episode': ep['episode'], 'source_id': ep['source']['source_id'], 'n_point': n,
            'initial_state': ep['initial_state'], 'initial_terms': progress.initial, 'initial_phi': ep['initial_phi'],
            'step': new_progress.step, 'parent': parent, 'next': cid, 'before': len(tokens), 'after': length, 'serial': serial,
            'actions': state.actions.tolist(), 'logits': logits[0, mask[0]].detach().cpu().double().tolist(),
            'context_id': state.context_id, 'selected_index': index, 'top_indices': top, 'old_logp': logp,
            'old_value': float(values[0]), 'progress_before': asdict(progress), 'progress_after': asdict(new_progress),
            'phi_before': state.potential, 'phi_after': next_state.potential if next_state else 0.,
            'reward': reward, 'terminal': reason is not None, 'reason': reason}
        if reason is not None: self.finish(ep)
        return state, row


def collect(runtime, deck, u):
    import torch
    from .packed_ppo import compute_gae
    runtime.round = u
    states, rows, advantages, targets, streams = [], [], [], [], []
    buffer_bytes = 0
    used = {ep['source']['source_id'] for ep in runtime.lanes if ep is not None and ep['state'] is not None}
    for lane in range(16):
        gen = torch.Generator().manual_seed(seed_for('train', u, lane)); ep = runtime.lanes[lane]
        local = []
        for tick in range(24):
            while ep is None or ep['state'] is None:
                ep = runtime.start(deck.sample(used))
            state, row = runtime.transition(ep, gen)
            row.update(lane=lane, tick=tick, round=u)
            buffer_bytes += state.tensor_bytes()
            if buffer_bytes > 2*1024**3: raise RuntimeError('Rollout memory cap')
            states.append(state); local.append(row)
        runtime.lanes[lane] = ep
        bootstrap = float(runtime.score(ep['state'])[2][0]) if ep['state'] is not None else 0.
        a, target = compute_gae([r['reward']['total'] for r in local], [r['old_value'] for r in local],
            [r['terminal'] for r in local], bootstrap_value=bootstrap, gamma=1., gae_lambda=.95)
        rows.extend(local); advantages.extend(a.tolist()); targets.extend(target.tolist())
        streams.append({'lane': lane, 'bootstrap': bootstrap, 'terminal_at_end': local[-1]['terminal']})
    size = sum(s.tensor_bytes() for s in states)
    if len(states) != 384 or size > 2*1024**3: raise RuntimeError('Rollout size/memory cap')
    return states, {'rows': rows, 'streams': streams, 'advantages': advantages, 'targets': targets, 'buffer_bytes': size}


def progress_report(finished, lanes, u):
    recent = [r for r in finished if r['phase'] == 'train' and u-3 <= r['round_ended'] <= u]
    cumulative = [r for r in finished if r['phase'] == 'train' and r['round_ended'] <= u]
    mean = lambda rows, key: statistics.fmean(r[key] for r in rows) if rows else None
    pending = [{'episode': ep['episode'], 'step': ep['progress'].step,
                'initial_terms': ep['progress'].initial, 'current_terms': len(ep['expr'].tokenize()),
                'current_reduce': (ep['progress'].initial-len(ep['expr'].tokenize()))/ep['progress'].initial}
               for ep in lanes if ep is not None and ep['state'] is not None]
    return {'round': u, 'round_block': [u-3, u], 'completed_episodes': len(recent),
        'mean_actual_final_reduce': mean(recent, 'final_reduce'), 'mean_best_reduce': mean(recent, 'best_reduce'),
        'cumulative_completed_episodes': len(cumulative), 'cumulative_final_reduce': mean(cumulative, 'final_reduce'),
        'pending_segments': pending, 'reason_counts': dict(Counter(r['reason'] for r in recent))}


def evaluate(runtime, test, tag):
    import torch
    results = []; folder = runtime.out/'evaluation'/tag; folder.mkdir()
    for i, row in enumerate(test):
        for seed in (401, 503):
            ep = runtime.start(row); trace = []; gen = torch.Generator().manual_seed(previous_seed('test', seed, row['id']))
            while ep['state'] is not None:
                _, r = runtime.transition(ep, gen); trace.append(r)
            record = {**runtime.finished[-1], 'model': tag, 'index': row['index'], 'evaluation_seed': seed, 'trace': trace}
            path = folder/f"case_{row['index']}_s{seed}.json.gz"; save_json_gz(path, record)
            results.append({k: v for k, v in record.items() if k != 'trace'} | {'file': path.relative_to(runtime.out).as_posix(), 'sha256': digest(path)})
        if (i+1) % 32 == 0: print(f'eval {tag}: {i+1}/256', flush=True)
    return results


def run():
    out, contract, protocol, ledger, training = freeze(); runtime = None
    try:
        from .full32_common import configure_torch
        from project.pretraining.encoder import read_checkpoint, construct_encoder
        from .early_fusion_policy import EarlyFusionActorCritic, update_ppo
        from .single24_policy import RewardConfig
        torch = configure_torch(); original_load = torch.load
        def counted(path, *a, **kw):
            if Path(path).resolve() != Path(protocol['encoder_path']).resolve():
                ledger.hit('old_rl_checkpoint_loads'); raise RuntimeError('Only the standalone encoder may be loaded')
            ledger.hit('checkpoint_loads'); return original_load(path, *a, **kw)
        with patch('torch.load', counted):
            state, config = read_checkpoint(protocol['encoder_path'], num_heads=8)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(103)
            encoder = construct_encoder(state, config, 'cpu')
            model = EarlyFusionActorCritic(encoder, enabled=True, top_k=16).to('cuda').train()
        del state
        initial = model.small_state_dict()
        write_json(out/'model.json', {'encoder_config': config, 'encoder_sha256': protocol['encoder_sha256'],
            'trainable_parameters': sum(p.numel() for p in model.trainable_parameters()),
            'trainable_groups': sorted({n.split('.')[0] for n, p in model.named_parameters() if p.requires_grad}),
            'old_rl_checkpoint_loads': 0, 'initialization_seed': 103})
        save_model(out/'snapshots/round_000.pt', model, round=0)
        optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=.0003, weight_decay=0.)
        runtime = Runtime(out, model, ledger, RewardConfig(**protocol['rewards']))
        deck = SourceDeck(training, previous_seed('source_deck'))
        summaries = []; reports = []
        for u in range(1, 33):
            folder = out/'training'/f'rollout_{u:03d}'; folder.mkdir()
            before = model.small_state_dict()
            states, buffer = collect(runtime, deck, u)
            if tensor_digest(before) != tensor_digest(model.small_state_dict()): raise ValueError('Policy changed during collection')
            save_json_gz(folder/'buffer.json.gz', buffer)
            with gzip.open(folder/'optimizer_steps.jsonl.gz', 'xt', encoding='utf-8') as stream:
                updates = update_ppo(model, optimizer, states, buffer['rows'], buffer['advantages'], buffer['targets'],
                    generator=torch.Generator().manual_seed(seed_for('update', u)), hit=ledger.hit,
                    on_step=lambda row: stream.write(json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n'))
            if len(updates) != 48: raise ValueError('Not48 updates')
            after = model.small_state_dict()
            groups = sorted({n.split('.')[0] for n in before if n != 'charges'})
            changes = {g: math.sqrt(sum(float((after[n].double()-before[n].double()).square().sum()) for n in before if n.split('.')[0] == g)) for g in groups}
            info = {'round': u, 'real_transitions': len(states), 'optimizer_steps': len(updates), 'buffer_bytes': buffer['buffer_bytes'],
                'buffer_sha256': digest(folder/'buffer.json.gz'), 'weight_changes': changes,
                'epoch_loss': {str(ep): {k: statistics.fmean(r[k] for r in updates if r['epoch'] == ep)
                    for k in ('actor_loss', 'value_loss', 'loss', 'support_changed_fraction', 'selected_action_excluded_fraction', 'clip_fraction')} for ep in (1, 2)},
                'counts': dict(ledger.counts)}
            write_json(folder/'completed.json', info); summaries.append(info)
            runtime.frozen_check()
            save_model(out/'snapshots/latest_recovery.pt', model, optimizer, round=u,
                deck_order=deck.order, deck_rng=deck.rng.getstate(), source_visits=deck.visits,
                lane_context=[{'tokens': ep['expr'].tokenize(), 'progress': asdict(ep['progress']), 'source': ep['source'], 'reason': ep['reason'], 'episode': ep['episode']} for ep in runtime.lanes],
                counts=dict(ledger.counts))
            print(f'rollout {u}/32; real_transitions=384; optimizer={ledger.counts["optimizer"]}', flush=True)
            if u % 4 == 0:
                progress = progress_report(runtime.finished, runtime.lanes, u)
                reports.append(progress); write_json(out/'progress'/f'round_{u:03d}.json', progress)
                print('PROGRESS '+json.dumps(progress, ensure_ascii=True), flush=True)
            if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > protocol['storage_cap_bytes']: raise RuntimeError('Storage cap')
            del states, buffer, before, after, updates
        save_model(out/'snapshots/round_032.pt', model, round=32)
        write_json(out/'training_completed.json', {'rounds': summaries, 'progress': reports, 'source_visits': deck.visits,
            'unique_training_sources': len(set(deck.visits)), 'counts': dict(ledger.counts)})
        if ledger.counts['optimizer'] != 1536 or ledger.counts['train_engine'] != 12288: raise ValueError('Training totals')
        test = inputs('test'); runtime.phase = 'eval'
        if {r['source_id'] for r in test} & {r['source_id'] for r in training}: raise ValueError('Source overlap')
        results = evaluate(runtime, test, 'trained_top16')
        ledger.hit('small_state_restores'); model.load_small_state(initial)
        results += evaluate(runtime, test, 'initial_top16')
        write_json(out/'evaluation.json', results)
        runtime.frozen_check(); runtime.close(); runtime = None
        vectors = {}; means = {}
        for tag in ('trained_top16', 'initial_top16'):
            vectors[tag] = {}
            for metric in ('final_reduce', 'best_reduce'):
                vectors[tag][metric] = [statistics.fmean(r[metric] for r in results if r['model'] == tag and r['source_id'] == sid) for sid in sorted({r['source_id'] for r in test})]
            means[tag] = {metric: statistics.fmean(vector) for metric, vector in vectors[tag].items()}
        report = {'means': means, 'trained_minus_initial': {metric: paired(vectors['trained_top16'][metric], vectors['initial_top16'][metric]) for metric in ('final_reduce', 'best_reduce')},
            'counts': {k: ledger.counts[k] for k in CAPS}, 'progress': reports,
            'train_unique_sources': len(set(deck.visits)), 'default_model_changed': False,
            'consumed_test_development_only': True, 'encoder_unchanged': True}
        check_sources(protocol)
        for path, expected in [(protocol['encoder_path'], protocol['encoder_sha256']), (protocol['protected_baseline_path'], protocol['protected_baseline_sha256']), (REPO/'.git/index', protocol['git_index_sha256'])]:
            if digest(path) != expected: raise ValueError('Protected file changed')
        write_json(out/'report.json', report)
        write_json(contract/'result_index.json', {'files': [{'path': p.relative_to(REPO).as_posix(), 'sha256': digest(p)} for p in sorted(out.rglob('*')) if p.is_file()]})
        write_json(out/'completed.json', {'report_sha256': digest(out/'report.json'), 'protocol_sha256': digest(contract/'protocol.json'), 'counts': report['counts']})
        print(json.dumps({k: report[k] for k in ('means', 'trained_minus_initial', 'counts')}, indent=2), flush=True)
    except BaseException as exc:
        write_json(out/'failed.json', {'error': repr(exc), 'counts': dict(ledger.counts), 'automatic_retry': False}); raise
    finally:
        if runtime is not None: runtime.close()
