"""Expression transitions, curriculum sampling, and fresh rollout collection."""
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import random
import torch
from data.symbolic import content_id
from data.features import expression_from_tokens
from models.policy import build_observation, sample_one
from .rewards import LongContext, LongProgress, frozen_potential
from .ppo import compute_gae


class SourceDeck:
    def __init__(self, rows, seed):
        if not rows:
            raise ValueError('A training pool is empty')
        self.rows, self.rng = rows, random.Random(seed)
        self.order, self.visits = [], []

    def sample(self, used):
        for _ in range(2*len(self.rows)):
            if not self.order:
                self.order = list(range(len(self.rows)))
                self.rng.shuffle(self.order)
            row = self.rows[self.order.pop()]
            if row['source_id'] in used:
                continue
            used.add(row['source_id'])
            self.visits.append(row['source_id'])
            return row
        raise RuntimeError('No unused source in this rollout')


class CurriculumDeck:
    def __init__(self, data, quotas, seed):
        if set(data) != set(quotas) or any(v < 1 for v in quotas.values()):
            raise ValueError('Positive quota for every training pool required')
        self.quotas = quotas
        self.pools = {k: SourceDeck(rows, seed+i) for i, (k, rows) in enumerate(data.items())}
        self.rng, self.order = random.Random(seed), []

    def sample(self, used):
        if not self.order:
            self.order = [k for k, count in self.quotas.items() for _ in range(count)]
            self.rng.shuffle(self.order)
        return self.pools[self.order.pop()].sample(used)

    def begin_rollout(self, used):
        self.order = []
        return [self.sample(used) for _ in range(sum(self.quotas.values()))]

    def state_dict(self):
        return dict(order=self.order, rng=self.rng.getstate(), pools={k:
            dict(order=v.order, visits=v.visits, rng=v.rng.getstate()) for k, v in self.pools.items()})

    def load_state_dict(self, state):
        if set(state['pools']) != set(self.pools):
            raise ValueError('Training pools differ from recovery file')
        self.order = list(state['order'])
        self.rng.setstate(state['rng'])
        for key, saved in state['pools'].items():
            pool = self.pools[key]
            pool.order, pool.visits = list(saved['order']), list(saved['visits'])
            pool.rng.setstate(saved['rng'])


class TrainingRuntime:
    def __init__(self, model, horizon=24, c_punishment=1.5):
        self.model, self.horizon, self.c_punishment = model, horizon, c_punishment
        self.counts, self.finished = Counter(), []

    def prepare(self, expression, n_point, best):
        tokens = expression.tokenize()
        if not tokens:
            return None, 'zero_expression'
        if len(tokens) > 128 or max(map(len, tokens), default=0) > 128:
            return None, 'capacity'
        observation = build_observation(tokens, n_point, best)
        if not len(observation.actions):
            return None, 'no_legal_actions'
        self.counts['encoder_forwards'] += 1
        return self.model.prepare(observation), None

    @torch.no_grad()
    def score(self, state):
        self.counts['policy_forwards'] += 1
        self.counts['value_forwards'] += 1
        return self.model([state])

    def potential(self, state):
        self.counts['potential_evaluations'] += 1
        return frozen_potential(state, self.model.charges, device=self.model.token_project.weight.device)

    def start(self, source):
        self.counts['reconstructions'] += 1
        expr = expression_from_tokens(source['tokens'])
        length = len(source['tokens'])
        context = LongContext(LongProgress(length, length, horizon=self.horizon), length,
                              last_terms=length, coefficient=self.c_punishment)
        state, reason = self.prepare(expr, source['n_point'], length)
        return dict(source=source, expr=expr, context=context, state=state, reason=reason,
                    potential=self.potential(state))

    def transition(self, ep, generator):
        import forsythia_hep as api
        state, context, expr = ep['state'], ep['context'], ep['expr']
        tokens, n = expr.tokenize(), ep['source']['n_point']
        logits, mask, values = self.score(state)
        selected, logp, top = sample_one(logits, mask, self.model.top_k, generator)
        t, i, j = state.actions[selected].tolist()
        self.counts['identity_calls'] += 1
        self.counts['sampling'] += 1
        child = api.apply_identity_at(expr, (t, (i, j)), n)[0]
        if expr.tokenize() != tokens:
            raise RuntimeError('Symbolic action mutated its parent')
        length = len(child.tokenize())
        reason = 'horizon' if context.progress.step+1 == self.horizon else None
        successor = None
        if reason is None:
            successor, reason = self.prepare(child, n, min(context.progress.best, length))
        if length > 128 or max(map(len, child.tokenize()), default=0) > 128:
            reason, successor = 'capacity', None
        phi_after = self.potential(successor)
        updated, reward, _ = context.advance(length, ep['potential'], phi_after, reason is not None,
                                              capacity=reason == 'capacity')
        row = dict(source_id=ep['source']['source_id'], selected_index=selected, old_logp=logp,
                   old_value=float(values[0]), logits=logits[0, mask[0]].detach().cpu().double().tolist(),
                   top_indices=top, reward=reward, terminal=reason is not None, before=len(tokens), after=length)
        ep.update(expr=child, context=updated, state=successor, reason=reason, potential=phi_after)
        if reason:
            self.finished.append(dict(source_id=row['source_id'], initial_terms=updated.progress.initial,
                final_terms=length, best_terms=updated.progress.best, steps=updated.progress.step,
                final_reduce=1-length/updated.progress.initial, reason=reason))
        return state, row


def collect_rollout(runtime, deck, seed, round_index, *, buffer_cap=6*1024**3):
    states, rows, advantages, targets = [], [], [], []
    used, bytes_used = set(), 0
    # All initial sources are reserved before any automatic reset.
    initial = deck.begin_rollout(used)
    for lane, source in enumerate(initial):
        generator = torch.Generator().manual_seed(int(content_id([seed, round_index, lane])[:15], 16))
        ep, local = runtime.start(source), []
        for _ in range(runtime.horizon):
            for _ in range(1024):
                if ep['state'] is not None:
                    break
                ep = runtime.start(deck.sample(used))
            else:
                raise RuntimeError('Too many unusable starts')
            state, row = runtime.transition(ep, generator)
            bytes_used += state.tensor_bytes()
            if bytes_used > buffer_cap:
                raise RuntimeError('Frozen feature buffer limit')
            states.append(state)
            local.append(row)
        bootstrap = float(runtime.score(ep['state'])[2][0]) if ep['state'] is not None else 0.
        adv, target = compute_gae([r['reward']['total'] for r in local], [r['old_value'] for r in local],
            [r['terminal'] for r in local], bootstrap_value=bootstrap, gamma=1., gae_lambda=.93)
        rows.extend(local)
        advantages.extend(adv.tolist())
        targets.extend(target.tolist())
    return states, dict(rows=rows, advantages=advantages, targets=targets, buffer_bytes=bytes_used)
