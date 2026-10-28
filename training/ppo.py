"""PPO-Clip, per-trajectory GAE, top-k entropy, and batch16 updates."""
import math
import torch
from torch import nn
from models.policy import topk_log_probs
ATOL=1e-4
def ppo_loss(new_logp, old_logp, advantage, values, targets, *, epsilon=.2, value_coef=.5):
    if bool(torch.isnan(new_logp).any() or torch.isposinf(new_logp).any()):
        raise ValueError("Invalid new probability")
    if not all(bool(torch.isfinite(x).all()) for x in (old_logp, advantage, values, targets)):
        raise ValueError("Nonfinite PPO input")
    ratio = (new_logp-old_logp.detach()).exp()  # -inf is an exact zero probability.
    if not bool(torch.isfinite(ratio).all()):
        raise ValueError("Nonfinite PPO ratio")
    actor = -torch.minimum(ratio*advantage.detach(), ratio.clamp(1-epsilon, 1+epsilon)*advantage.detach()).mean()
    value = (values-targets.detach()).square().mean()
    return actor+value_coef*value, actor, value, ratio
@torch.no_grad()
def compute_gae(rewards, values, terminated, *, bootstrap_value=0., gamma=1., gae_lambda=.95):
    """One contiguous trajectory segment; truncation must supply V(next_state).

    Budget exhaustion in a finite 64-call task is termination, not a training
    buffer truncation. This function never carries GAE between trajectories.
    """
    r, v = (torch.as_tensor(x, dtype=torch.float64).detach() for x in (rewards, values))
    d = torch.as_tensor(terminated, dtype=torch.bool, device=r.device)
    if r.ndim != 1 or r.shape != v.shape or d.shape != r.shape:
        raise ValueError("GAE expects equal one-dimensional trajectories")
    if not (0 <= gamma <= 1 and 0 <= gae_lambda <= 1) or not math.isfinite(float(bootstrap_value)):
        raise ValueError("Invalid discount, lambda or bootstrap")
    if not bool(torch.isfinite(r).all() and torch.isfinite(v).all()):
        raise ValueError("Nonfinite rewards/values")
    if len(d) and bool(d[-1]) and float(bootstrap_value) != 0.:
        raise ValueError("A terminated state has zero bootstrap value")
    advantage = torch.zeros_like(r)
    following_value, following_advantage = float(bootstrap_value), 0.
    for t in reversed(range(len(r))):
        live = (~d[t]).to(r.dtype)
        delta = r[t] + gamma * live * following_value - v[t]
        advantage[t] = delta + gamma * gae_lambda * live * following_advantage
        following_value, following_advantage = v[t], advantage[t]
    return advantage, advantage + v
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
def verify_behavior_batch(logits, mask, records, top_k):
    """Before any update: exact top-k support, bounded all-supported logp error."""
    if len(records) != len(logits):
        raise ValueError('Behavior batch size mismatch')
    saved = torch.zeros_like(logits, dtype=torch.float64)
    for j, r in enumerate(records):
        count = int(mask[j].sum())
        if count != len(r['logits']) or not bool(mask[j, :count].all()) or bool(mask[j, count:].any()):
            raise ValueError('Legal action layout differs from behavior')
        saved[j, :count] = torch.tensor(r['logits'], dtype=torch.float64, device=logits.device)
    lp, keep = topk_log_probs(logits, mask, top_k)
    old_lp, old_keep = topk_log_probs(saved, mask, top_k)
    if not torch.equal(keep, old_keep):
        raise ValueError('Behavior top-k support changed before update')
    selected_errors = []
    for j, r in enumerate(records):
        if old_keep[j].nonzero().flatten().tolist() != r['top_indices']:
            raise ValueError('Stored behavior support mismatch')
        a = r['selected_index']
        if not 0 <= a < logits.shape[1] or not bool(old_keep[j, a]):
            raise ValueError('Stored action outside behavior support')
        if abs(float(old_lp[j, a].detach()) - r['old_logp']) > 1e-8:
            raise ValueError('Stored behavior logp mismatch')
        selected_errors.append(abs(float(lp[j, a].detach()) - r['old_logp']))
    error = float((lp[keep].detach() - old_lp[keep]).abs().max())
    if not math.isfinite(error) or error > ATOL or max(selected_errors) > ATOL:
        raise ValueError('Behavior logp error exceeds frozen numerical tolerance')
    return {'mode': 'batch_numeric_v1', 'atol': ATOL, 'support_equal': True,
            'all_supported_logp_max_abs': error, 'selected_logp_max_abs': max(selected_errors),
            'relative_probability_error_bound': math.expm1(error)}
class FirstBatchReference:
    def __init__(self,model,states,records,hit,*,enabled=False):
        if not enabled:raise RuntimeError('Numerical reference branch is off')
        self.model=model;self.records={id(s):r for s,r in zip(states,records)};self.hit=hit
        self.checked=False;self.evidence=None

    def __getattr__(self,name):return getattr(self.model,name)

    def __call__(self,states):
        result=self.model(states)
        if self.checked:return result
        rows=[self.records[id(s)] for s in states];self.checked=True
        try:
            verify_behavior_batch(result[0],result[1],rows,self.model.top_k)
            self.evidence={'mode':'batched','reference_forwards':0};return result
        except ValueError as exc:
            if str(exc)!='Behavior logp error exceeds frozen numerical tolerance':raise
        # The validator has already required exact support and valid saved logp.
        lp,keep=topk_log_probs(result[0].detach(),result[1],self.model.top_k)
        errors=[]
        for j,r in enumerate(rows):
            saved=torch.tensor(r['logits'],dtype=torch.float64,device=lp.device)[None]
            old,mask=topk_log_probs(saved,torch.ones_like(saved,dtype=torch.bool),self.model.top_k)
            errors.append(float((lp[j,:saved.shape[1]][mask[0]]-old[mask]).abs().max()))
        self.evidence={'mode':'single_state_reference','discarded_batch_max_logp_error':max(errors),'reference_forwards':len(states)}
        if max(errors)>1e-3:raise ValueError('Batch discrepancy is too large for the certified numerical fallback')
        del result,lp,keep
        outputs=[]
        for state in states:
            self.hit('reference_policy_forwards');self.hit('reference_feature_reuses')
            outputs.append(self.model([state]))
        logits=nn.utils.rnn.pad_sequence([r[0][0] for r in outputs],batch_first=True)
        mask=nn.utils.rnn.pad_sequence([r[1][0] for r in outputs],batch_first=True,padding_value=False)
        values=torch.cat([r[2] for r in outputs])
        report=verify_behavior_batch(logits,mask,rows,self.model.top_k)
        if report['all_supported_logp_max_abs']>1e-8:raise ValueError('Single-state reference does not reproduce behavior')
        self.evidence['reference_max_logp_error']=report['all_supported_logp_max_abs']
        self.hit('numerical_fallback_batches')
        return logits,mask,values
def update_ppo(model, optimizer, states, records, advantages, targets, *, generator, hit,
             on_step, completed_updates, enabled=False):
    if not enabled or len(states) != 768 or len(records) != 768:
        raise ValueError('Explicit 768-transition update required')
    if type(completed_updates) is not int or not 0 <= completed_updates <= 5664 or completed_updates % 96:
        raise ValueError('60-round update boundary required')
    proxy = FirstBatchReference(model, states, records, hit, enabled=True)
    device = model.token_project.weight.device
    adv = torch.tensor(advantages, dtype=torch.float64, device=device)
    adv = (adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-8)
    target = torch.tensor(targets, dtype=torch.float64, device=device)
    old = torch.tensor([r['old_logp'] for r in records], dtype=torch.float64, device=device)
    updates = []
    for epoch in (1, 2):
        order = torch.randperm(768, generator=generator).tolist()
        for start in range(0, 768, 16):
            ids = order[start:start+16]
            update = completed_updates+len(updates)+1
            for group in optimizer.param_groups:
                group['lr'] = 1e-4
            optimizer.zero_grad(set_to_none=True)
            selected = torch.tensor([records[i]['selected_index'] for i in ids], device=device)
            hit('update_forwards')
            hit('encoder_feature_reuses', 16)
            logits, mask, values = proxy([states[i] for i in ids])
            lp, keep = topk_log_probs(logits, mask, model.top_k)
            chosen = lp[torch.arange(16, device=device), selected]
            if epoch == 1 and start == 0:
                behavior = verify_behavior_batch(logits, mask, [records[i] for i in ids], model.top_k)
            hit('entropy_batches')
            loss, actor, value, ratio, entropy = entropy_ppo_loss(lp, keep, selected, old[ids], adv[ids], values, target[ids], entropy_coef=.001)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite PPO loss')
            hit('autograd')
            loss.backward()
            norm = float(nn.utils.clip_grad_norm_(tuple(model.trainable_parameters()), .5, error_if_nonfinite=True))
            hit('optimizer')
            optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in model.trainable_parameters()):
                raise ValueError('Nonfinite parameters')
            support = [keep[j].nonzero().flatten().tolist() for j in range(16)]
            row = {'epoch': epoch, 'indices': ids, 'batch_size': 16, 'course_update': update,
                'lr_used': 1e-4, 'gradient_norm': norm,
                'loss': float(loss.detach()), 'actor_loss': float(actor.detach()), 'value_loss': float(value.detach()),
                'entropy': float(entropy.detach().mean()), 'clip_fraction': float(((ratio-1).abs() > .2).double().mean()),
                'selected_action_excluded_fraction': float((~torch.isfinite(chosen)).double().mean()),
                'support_changed_fraction': sum(support[j] != records[i]['top_indices'] for j, i in enumerate(ids))/16,
                'new_logits': [logits[j, mask[j]].detach().cpu().double().tolist() for j in range(16)],
                'new_top_indices': support,
                'new_chosen_logp': [float(v) if math.isfinite(float(v)) else None for v in chosen.detach().cpu()],
                'ratio': ratio.detach().cpu().tolist(), 'values': values.detach().cpu().double().tolist()}
            if epoch == 1 and start == 0:
                row.update(behavior_check=behavior, forward_execution=proxy.evidence)
            on_step(row)
            updates.append({k: v for k, v in row.items() if k not in ('indices', 'new_logits', 'new_top_indices', 'new_chosen_logp', 'ratio', 'values')})
    optimizer.zero_grad(set_to_none=True)
    assert len(updates) == 96
    return updates
