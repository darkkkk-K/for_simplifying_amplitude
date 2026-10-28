"""Training environment adapter with a separate restricted policy observation."""
from dataclasses import asdict

from .early_fusion_experiment import Runtime
from .early_fusion_policy import sample_one
from .packed32_common import append, expression_from_tokens
from .restricted_environment import TrainingContext, frozen_potential
from .restricted_policy import SCHEMA, build_observation


class RestrictedTrainingRuntime(Runtime):
    def __init__(self, out, model, ledger, rewards, *, c_punishment, gae_lambda=.93, enabled=False):
        if not enabled:
            raise RuntimeError('Separated training runtime is off')
        if model.observation_schema != SCHEMA or model.top_k != 8:
            raise ValueError('Expected restricted top8 policy')
        super().__init__(out, model, ledger, rewards)
        self.c_punishment = c_punishment
        self.gae_lambda = gae_lambda
        self.lanes = [None]*24
        self.value_hook = model.value_head.register_forward_pre_hook(lambda *_: ledger.hit('value_forwards'))

    def close(self):
        self.value_hook.remove()
        super().close()

    def prepare(self, expr, best_terms, n_point):
        tokens = expr.tokenize()
        if not tokens:
            return None, 'zero_expression'
        if len(tokens) > 128 or max(map(len, tokens), default=0) > 128:
            return None, 'capacity'
        self.ledger.hit('preparations')
        observation = build_observation(tokens, n_point, best_terms)
        if not len(observation.actions):
            return None, 'no_legal_actions'
        return self.model.prepare(observation), None

    def potential(self, state):
        if state is None:
            return 0.
        self.ledger.hit('potential_evaluations')
        return frozen_potential(state, self.model.charges, device=self.model.token_project.weight.device)

    def start(self, source):
        self.ledger.hit('reconstructions')
        self.serial += 1
        expr = expression_from_tokens(source['tokens'])
        sid = self.state(expr)
        context = TrainingContext.start(len(source['tokens']), coefficient=self.c_punishment,
                                        rewards=self.rewards, enabled=True)
        state, reason = self.prepare(expr, context.progress.best, source['n_point'])
        phi = self.potential(state)
        episode = {'source': source, 'expr': expr, 'episode': self.serial,
                   'training_context': context, 'progress': context.progress,
                   'initial_state': sid, 'initial_phi': phi, 'potential': phi,
                   'state': state, 'reason': reason, 'best_state': sid}
        if state is None:
            self.finish(episode)
        return episode

    def transition(self, ep, generator):
        if self.phase != 'train':
            raise RuntimeError('Use reward-free inference for evaluation')
        state, context, expr = ep['state'], ep['training_context'], ep['expr']
        progress = context.progress
        if state.best_terms != progress.best or context.punishment.last_terms != len(expr.tokenize()):
            raise ValueError('Environment and expression observations diverged')
        logits, mask, values = self.score(state)
        self.ledger.hit('sampling')
        index, logp, top = sample_one(logits, mask, self.model.top_k, generator)
        t, i, j = state.actions[index].tolist()
        action, tokens = (t, (i, j)), expr.tokenize()
        parent, n = self.state(expr), ep['source']['n_point']
        self.ledger.hit('train_engine')
        serial = self.ledger.counts['train_engine']
        append(self.out/'calls.jsonl', {'event': 'start', 'phase': 'train', 'serial': serial,
                                      'parent': parent, 'action': action, 'n_point': n})
        try:
            child = self.apply(expr, action, n)[0]
        except BaseException as exc:
            append(self.out/'calls.jsonl', {'event': 'error', 'serial': serial, 'error': repr(exc)})
            raise
        if expr.tokenize() != tokens:
            raise ValueError('Symbolic action mutated its parent')
        cid, length = self.state(child), len(child.tokenize())
        append(self.out/'calls.jsonl', {'event': 'end', 'phase': 'train', 'serial': serial, 'child': cid, 'terms': length})
        reason = 'horizon' if progress.step+1 == 20 else None
        next_state = None
        if reason is None:
            next_state, reason = self.prepare(child, min(progress.best, length), n)
        if length > 128 or max(map(len, child.tokenize()), default=0) > 128:
            reason, next_state = 'capacity', None
        phi_before, phi_after = ep['potential'], self.potential(next_state)
        updated, reward, punishment = context.advance(length, phi_before, phi_after, reason is not None,
                                                       capacity=reason == 'capacity')
        if next_state is not None and next_state.best_terms != updated.progress.best:
            raise ValueError('Historical minimum not updated')
        if updated.progress.best < progress.best:
            ep['best_state'] = cid
        ep.update(expr=child, training_context=updated, progress=updated.progress, potential=phi_after,
                  state=next_state, reason=reason)
        row = {'episode': ep['episode'], 'source_id': ep['source']['source_id'], 'n_point': n,
               'initial_state': ep['initial_state'], 'initial_terms': progress.initial, 'initial_phi': ep['initial_phi'],
               'step': updated.progress.step, 'parent': parent, 'next': cid, 'before': len(tokens), 'after': length,
               'serial': serial, 'actions': state.actions.tolist(),
               'logits': logits[0, mask[0]].detach().cpu().double().tolist(), 'policy_context': state.context.tolist(),
               'context_id': state.context_id, 'selected_index': index, 'top_indices': top,
               'old_logp': logp, 'old_value': float(values[0]), 'progress_before': asdict(progress),
               'progress_after': asdict(updated.progress), 'phi_before': phi_before, 'phi_after': phi_after,
               'reward': reward, 'terminal': reason is not None, 'reason': reason, 'observation_schema': SCHEMA}
        if self.c_punishment:
            row['checkpoint_punishment'] = punishment
        if reason is not None:
            self.finish(ep)
        return state, row
