"""Reward-free actor inference. Budgets belong only to the outer controller."""
from dataclasses import dataclass

import torch

from .legal_actions import content_id
from .restricted_policy import build_observation


@dataclass(frozen=True)
class InferenceResult:
    expression: object
    best_expression: object
    steps: int
    reason: str
    trace: list


def select_argmax(logits, mask):
    if logits.ndim != 2 or logits.shape[0] != 1 or mask.shape != logits.shape or mask.dtype != torch.bool:
        raise ValueError('Single-state logits/mask required')
    if not bool(mask.any()) or not bool(torch.isfinite(logits[mask]).all()):
        raise ValueError('Empty or invalid legal support')
    scores = logits.detach().masked_fill(~mask, -torch.inf)
    index = int(scores[0].argmax())
    return index, int((mask[0] & (scores[0] == scores[0, index])).sum())


def simplify_argmax(model, expression, n_point, *, max_actions, apply_action, hit=None, enabled=False):
    if not enabled:
        raise RuntimeError('Restricted inference is off')
    if type(max_actions) is not int or max_actions < 0:
        raise ValueError('Nonnegative external action budget required')
    hit = hit or (lambda key, amount=1: None)
    current = best_expression = expression
    best = len(current.tokenize())
    trace = []
    reason = 'horizon'
    for step in range(max_actions):
        tokens = current.tokenize()
        if not tokens:
            reason = 'zero_expression'; break
        if len(tokens) > 128 or max(map(len, tokens), default=0) > 128:
            reason = 'capacity'; break
        hit('preparations')
        observation = build_observation(tokens, n_point, best)
        if not len(observation.actions):
            reason = 'no_legal_actions'; break
        hit('encoder_forwards')
        state = model.prepare(observation)
        hit('policy_forwards')
        with torch.no_grad():
            logits, mask = model.policy_logits([state])
        index, ties = select_argmax(logits, mask)
        t, i, j = state.actions[index].tolist()
        action = (t, (i, j))
        hit('argmax_selections'); hit('engine')
        child = apply_action(current, action, n_point)[0]
        if current.tokenize() != tokens:
            raise ValueError('Symbolic action mutated its parent')
        child_tokens = child.tokenize()
        after = len(child_tokens)
        if after < best:
            best, best_expression = after, child
        trace.append({'step': step+1, 'before': len(tokens), 'after': after,
                      'parent_tokens': tokens, 'next_tokens': child_tokens,
                      'parent': content_id(tokens), 'next': content_id(child_tokens),
                      'best_terms': best, 'context': state.context.tolist(), 'context_id': state.context_id,
                      'actions': state.actions.tolist(), 'logits': logits[0, mask[0]].cpu().double().tolist(),
                      'selected_index': index, 'selected_action': action, 'maximum_ties': ties, 'selection': 'argmax'})
        current = child
        if not child_tokens:
            reason = 'zero_expression'; break
        if after > 128 or max(map(len, child_tokens), default=0) > 128:
            reason = 'capacity'; break
    return InferenceResult(current, best_expression, len(trace), reason, trace)
