"""20-step reward/context and480-sample PPO adapter; old frozen code unchanged."""
from dataclasses import dataclass, replace
from fractions import Fraction
import math

import torch
from torch import nn

from .early_fusion_policy import EarlyFusionActorCritic, topk_log_probs, ppo_loss
from .legal_actions import content_id

HORIZON = 20
BETA = .4


def checkpoints(horizon=HORIZON, beta=BETA):
    fraction = Fraction(str(beta))
    if type(horizon) is not int or horizon <= 0 or not 0 < fraction < 1:
        raise ValueError('Positive horizon and0<beta<1 required')
    t = 0; cuts = []
    while t < horizon:
        t += ((horizon-t)*fraction.numerator+fraction.denominator-1)//fraction.denominator
        cuts.append(t)
    return tuple(cuts)


CUTS = checkpoints()


@dataclass(frozen=True)
class Progress20:
    initial: int
    best: int
    step: int = 0
    best_stage: int = 0
    paid_bonus: float = 0.

    def advance(self, length, phi_before, phi_after, terminal, config):
        if self.initial <= 0 or length < 0 or not 0 <= self.step < HORIZON:
            raise ValueError('Invalid20-step progress')
        if not all(math.isfinite(v) and 0 <= v <= 1+1e-6 for v in (phi_before, phi_after)):
            raise ValueError('Invalid fixed potential')
        t = self.step+1
        if t == HORIZON and not terminal: raise ValueError('Task horizon is terminal')
        stage = next(j for j, cut in enumerate(CUTS, 1) if t <= cut)
        best = min(self.best, length); first = stage if best < self.best else self.best_stage
        progress = (self.initial-best)/self.initial
        bonus = config.segment_scale/self.initial*progress*((len(CUTS)-first)/(len(CUTS)-1) if first else 0.)
        paid = bonus if terminal or t in CUTS else self.paid_bonus
        main = (self.best-best)/self.initial
        segment = paid-self.paid_bonus if config.segment_enabled else 0.
        potential = config.potential_scale/self.initial*((0. if terminal else phi_after)-phi_before) if config.potential_enabled else 0.
        return Progress20(self.initial, best, t, first, paid), {'main': main, 'segment': segment, 'potential': potential, 'total': main+segment+potential}


def condition20(state, progress):
    if not 0 <= progress.step < HORIZON: raise ValueError('Terminal state has no decision')
    cut = next(c for c in CUTS if c > progress.step)
    env = [float(state.n_point == n) for n in (4,5,6)]+[math.log1p(len(state.encoded)), (HORIZON-progress.step)/HORIZON, math.log1p(HORIZON)]
    history = [math.log1p(progress.initial), progress.best/progress.initial, progress.best_stage/len(CUTS),
        progress.paid_bonus*progress.initial/.5, (cut-progress.step)/HORIZON, state.potential]
    return replace(state, context=torch.tensor(env+history, dtype=torch.float32),
        context_id=content_id([state.state_id, state.n_point, HORIZON, BETA, vars(progress)]))


class Horizon20ActorCritic(EarlyFusionActorCritic):
    @torch.no_grad()
    def prepare(self, batch, progress):
        if not bool((batch.initial_budget == HORIZON).all()) or not bool((batch.remaining_budget == HORIZON-progress.step).all()):
            raise ValueError('Not20-step budget context')
        # Reuse only the frozen encoder/structural preparation. Replace the
        # inherited24-step context before any actor/critic forward can see it.
        state = super().prepare(batch, progress)
        return condition20(state, progress) if state is not None else None


def update_ppo480(model, optimizer, states, records, advantages, targets, *, generator, hit, on_step):
    if len(states) != 480 or len(records) != 480: raise ValueError('Expected480 real transitions')
    device = model.token_project.weight.device
    adv = torch.tensor(advantages, dtype=torch.float64, device=device)
    adv = (adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-8)
    targets = torch.tensor(targets, dtype=torch.float64, device=device)
    old = torch.tensor([r['old_logp'] for r in records], dtype=torch.float64, device=device)
    updates = []
    for epoch in (1, 2):
        order = torch.randperm(480, generator=generator).tolist()
        for start in range(0, 480, 16):
            indices = order[start:start+16]
            selected = torch.tensor([records[i]['selected_index'] for i in indices], device=device)
            optimizer.zero_grad(set_to_none=True)
            hit('update_forwards'); hit('encoder_feature_reuses', 16)
            logits, mask, values = model([states[i] for i in indices])
            lp, keep = topk_log_probs(logits, mask, model.top_k)
            chosen = lp[torch.arange(16, device=device), selected]
            if epoch == 1 and start == 0 and not torch.allclose(chosen.detach(), old[indices], atol=1e-5, rtol=0.):
                raise ValueError('Initial PPO batch does not reproduce behavior probability')
            loss, actor, value, ratio = ppo_loss(chosen, old[indices], adv[indices], values, targets[indices])
            if not bool(torch.isfinite(loss)): raise ValueError('Nonfinite loss')
            hit('autograd'); loss.backward()
            gradients = {}
            for name, parameter in model.named_parameters():
                if not name.startswith('encoder.') and parameter.grad is not None:
                    group = name.split('.')[0]
                    gradients[group] = gradients.get(group, 0.)+float(parameter.grad.detach().double().square().sum())
            norm = float(nn.utils.clip_grad_norm_(tuple(model.trainable_parameters()), .5, error_if_nonfinite=True))
            hit('optimizer'); optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in model.trainable_parameters()): raise ValueError('Nonfinite parameters')
            supports = [keep[j].nonzero().flatten().tolist() for j in range(16)]
            changed = [supports[j] != records[i]['top_indices'] for j,i in enumerate(indices)]
            retained = torch.isfinite(chosen); log_ratio = chosen[retained]-old[indices][retained]
            row = {'epoch': epoch, 'indices': indices, 'batch_size': 16,
                'loss': float(loss.detach()), 'actor_loss': float(actor.detach()), 'value_loss': float(value.detach()),
                'gradient_norm': norm, 'gradient_groups': {k: math.sqrt(v) for k,v in gradients.items()},
                'new_logits': [logits[j,mask[j]].detach().cpu().double().tolist() for j in range(16)],
                'new_top_indices': supports, 'new_chosen_logp': [float(v) if math.isfinite(float(v)) else None for v in chosen.detach().cpu()],
                'values': values.detach().cpu().double().tolist(), 'ratio': ratio.detach().cpu().tolist(),
                'support_changed_fraction': sum(changed)/16, 'selected_action_excluded_fraction': float((~retained).double().mean()),
                'conditional_approx_kl_retained_samples': float((log_ratio.exp()-1-log_ratio).mean()) if bool(retained.any()) else None,
                'old_to_new_kl_infinite_for_any_state': any(changed), 'clip_fraction': float(((ratio-1).abs() > .2).double().mean())}
            on_step(row); updates.append({k:v for k,v in row.items() if k not in ('new_logits','new_top_indices')})
    optimizer.zero_grad(set_to_none=True)
    return updates
