"""Opt-in entropy regularization; original20-step PPO implementation unchanged."""
import math

import torch
from torch import nn

from .early_fusion_policy import ppo_loss, topk_log_probs


def entropy_from_log_probs(log_probs, keep):
    """Entropy of exactly the current top-k categorical distribution, per state."""
    if log_probs.ndim != 2 or keep.shape != log_probs.shape or keep.dtype != torch.bool:
        raise ValueError('Expected[B,A] log probabilities and Boolean support')
    if not bool(keep.any(-1).all()) or not bool(torch.isfinite(log_probs[keep]).all()):
        raise ValueError('Empty or nonfinite categorical support')
    if not bool(torch.isneginf(log_probs[~keep]).all()):
        raise ValueError('Excluded actions must have zero probability')
    # Mask before multiplication: never form0*(-inf), including in backward.
    safe_log_probs = log_probs.masked_fill(~keep, 0.)
    return -(log_probs.exp()*safe_log_probs).sum(-1)


def entropy_ppo_loss(log_probs, keep, selected, old, advantages, values, targets, *, entropy_coef=0.):
    if not math.isfinite(entropy_coef) or entropy_coef < 0:
        raise ValueError('Entropy coefficient must be finite and nonnegative')
    chosen = log_probs[torch.arange(len(selected), device=selected.device), selected]
    base, actor, value, ratio = ppo_loss(chosen, old, advantages, values, targets)
    entropy = entropy_from_log_probs(log_probs, keep)
    # Exact baseline path when disabled, not an altered surrogate objective.
    loss = base if entropy_coef == 0. else base-entropy_coef*entropy.mean()
    return loss, actor, value, ratio, entropy


def update_ppo480_entropy(model, optimizer, states, records, advantages, targets, *,
                          generator, hit, on_step, enabled=False, entropy_coef=0.):
    if not enabled:
        raise RuntimeError('entropy20 update branch is off')
    if not math.isfinite(entropy_coef) or entropy_coef < 0:
        raise ValueError('Invalid entropy coefficient')
    if len(states) != 480 or len(records) != 480:
        raise ValueError('Expected480 real transitions')
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
            hit('entropy_batches')
            loss, actor, value, ratio, entropy = entropy_ppo_loss(
                lp, keep, selected, old[indices], adv[indices], values, targets[indices], entropy_coef=entropy_coef)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite loss')
            hit('autograd'); loss.backward()
            gradients = {}
            for name, parameter in model.named_parameters():
                if not name.startswith('encoder.') and parameter.grad is not None:
                    group = name.split('.')[0]
                    gradients[group] = gradients.get(group, 0.)+float(parameter.grad.detach().double().square().sum())
            norm = float(nn.utils.clip_grad_norm_(tuple(model.trainable_parameters()), .5, error_if_nonfinite=True))
            hit('optimizer'); optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in model.trainable_parameters()):
                raise ValueError('Nonfinite parameters')
            supports = [keep[j].nonzero().flatten().tolist() for j in range(16)]
            changed = [supports[j] != records[i]['top_indices'] for j, i in enumerate(indices)]
            retained = torch.isfinite(chosen); log_ratio = chosen[retained]-old[indices][retained]
            sizes = keep.sum(-1)
            normalized = torch.where(sizes > 1, entropy/sizes.double().clamp_min(2).log(), torch.zeros_like(entropy))
            row = {'epoch': epoch, 'indices': indices, 'batch_size': 16,
                'loss': float(loss.detach()), 'actor_loss': float(actor.detach()), 'value_loss': float(value.detach()),
                'entropy': float(entropy.detach().mean()), 'entropy_per_state': entropy.detach().cpu().tolist(),
                'normalized_entropy': float(normalized.detach().mean()), 'entropy_coef': entropy_coef,
                'entropy_loss': float(-entropy_coef*entropy.detach().mean()),
                'ppo_loss_without_entropy': float((actor+.5*value).detach()),
                'gradient_norm': norm, 'gradient_groups': {k: math.sqrt(v) for k, v in gradients.items()},
                'new_logits': [logits[j, mask[j]].detach().cpu().double().tolist() for j in range(16)],
                'new_top_indices': supports, 'new_chosen_logp': [float(v) if math.isfinite(float(v)) else None for v in chosen.detach().cpu()],
                'values': values.detach().cpu().double().tolist(), 'ratio': ratio.detach().cpu().tolist(),
                'support_changed_fraction': sum(changed)/16, 'selected_action_excluded_fraction': float((~retained).double().mean()),
                'conditional_approx_kl_retained_samples': float((log_ratio.exp()-1-log_ratio).mean()) if bool(retained.any()) else None,
                'old_to_new_kl_infinite_for_any_state': any(changed), 'clip_fraction': float(((ratio-1).abs() > .2).double().mean())}
            on_step(row)
            updates.append({k: v for k, v in row.items() if k not in ('new_logits', 'new_top_indices')})
    optimizer.zero_grad(set_to_none=True)
    return updates
